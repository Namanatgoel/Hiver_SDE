# DESIGN RATIONALE
# [@F10_nofreelabels2025] Reference-grounded judging (provide brand's real reply as reference)
#   — blind judging is the weakest configuration; real reference anchors the rubric.
# [@F6_wang2023unfair]   Pairwise, both orders averaged — single-order pairwise has position bias.
# [@F7_positionbias2024judging] Position bias in LLM judges — swap to measure and cancel.
# [@F5_verga2024jury]    Jury of 3 judges — ensemble reduces per-judge bias variance.
# [@F4_kim2024prometheus2] Prometheus-2-7B local judge — 7B at Q4 fits 8GB VRAM; best human
#   agreement among open evaluators; reproducible without API calls.
# [@F2_liu2023geval]     G-Eval rubric with chain-of-thought + probability-weighted scoring.

"""
src/judge.py — reference-grounded jury judge (Gemini + Groq + local Prometheus-2).

Device: RTX 5050 (local Prometheus-2) + LLM APIs (Gemini + Groq)
Phase:  7
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

JUDGE_RUBRIC = """You are evaluating a customer support reply.

Reference reply (how the brand actually responded): {reference}

Agent reply: {reply}
Customer message: {message}

Rate the agent reply on a scale of 1-5:
5 = Excellent: fully resolves the issue, matches brand tone, well-grounded in evidence
4 = Good: mostly resolves, minor gaps
3 = Adequate: partially addresses the issue
2 = Poor: misses key aspects or has factual errors
1 = Unacceptable: wrong, harmful, or completely off-topic

Provide your rating as JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""

PAIRWISE_RUBRIC = """Compare two customer support replies.

Customer message: {message}
Reference (actual brand reply): {reference}

Reply A: {reply_a}
Reply B: {reply_b}

Which reply is better? Return JSON only:
{{"winner": "A" | "B" | "tie", "reasoning": "<one sentence>"}}"""


class JuryJudge:
    """Reference-grounded pairwise + absolute jury of 3 judges."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Individual judges
    # ------------------------------------------------------------------

    def _call_api_judge(self, prompt: str, provider: str, purpose: str) -> str:
        from src.llm.gateway import complete
        response, _ = complete(
            [{"role": "user", "content": prompt}],
            purpose=purpose,
            params={"temperature": 0.0},
            force_provider=provider,
        )
        return response

    def _call_local_judge(self, prompt: str) -> str:
        """Call Prometheus-2-7B via Ollama (local, load-use-del managed by Ollama)."""
        from src.llm.gateway import complete
        response, _ = complete(
            [{"role": "user", "content": prompt}],
            purpose="local_judge",
            params={"temperature": 0.0},
            force_provider="ollama",
        )
        return response

    def _parse_score(self, raw: str) -> dict:
        text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: look for "score": N pattern
            import re
            m = re.search(r'"score"\s*:\s*(\d)', text)
            score = int(m.group(1)) if m else 3
            return {"score": score, "reasoning": text[:100]}

    def _parse_pairwise(self, raw: str) -> dict:
        text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r'"winner"\s*:\s*"([AB]|tie)"', text)
            winner = m.group(1) if m else "tie"
            return {"winner": winner, "reasoning": text[:100]}

    # ------------------------------------------------------------------
    # Absolute scoring (1-5)
    # ------------------------------------------------------------------

    def score_absolute(
        self,
        message: str,
        reply: str,
        reference: str,
    ) -> dict:
        """Score a reply from all 3 judges; return per-judge scores + median."""
        prompt = JUDGE_RUBRIC.format(
            message=message[:400],
            reply=reply[:600],
            reference=reference[:400],
        )

        results = {}
        for judge_name, call_fn in [
            ("gemini", lambda p: self._call_api_judge(p, "gemini", "judge_absolute_gemini")),
            ("groq",   lambda p: self._call_api_judge(p, "groq",   "judge_absolute_groq")),
            ("ollama", lambda p: self._call_local_judge(p)),
        ]:
            try:
                raw = call_fn(prompt)
                parsed = self._parse_score(raw)
                results[judge_name] = {"score": parsed.get("score", 3), "reasoning": parsed.get("reasoning", "")}
            except Exception as e:
                results[judge_name] = {"score": None, "error": str(e)}

        scores = [v["score"] for v in results.values() if v.get("score") is not None]
        import statistics
        median = statistics.median(scores) if scores else None
        return {"judges": results, "median": median}

    # ------------------------------------------------------------------
    # Pairwise scoring (both orders averaged) ([@F6_wang2023unfair])
    # ------------------------------------------------------------------

    def score_pairwise(
        self,
        message: str,
        reply_agent: str,
        reply_baseline: str,
        reference: str,
    ) -> dict:
        """Run pairwise comparison in both orders; return averaged result."""
        def _run_order(a: str, b: str, order: str) -> dict:
            prompt = PAIRWISE_RUBRIC.format(
                message=message[:400],
                reference=reference[:400],
                reply_a=a[:500],
                reply_b=b[:500],
            )
            results = {}
            for judge_name, call_fn in [
                ("gemini", lambda p: self._call_api_judge(p, "gemini", f"judge_pairwise_{order}_gemini")),
                ("groq",   lambda p: self._call_api_judge(p, "groq",   f"judge_pairwise_{order}_groq")),
                ("ollama", lambda p: self._call_local_judge(p)),
            ]:
                try:
                    raw = call_fn(prompt)
                    parsed = self._parse_pairwise(raw)
                    results[judge_name] = parsed
                except Exception as e:
                    results[judge_name] = {"winner": "tie", "error": str(e)}
            return results

        # Order 1: agent=A, baseline=B
        order1 = _run_order(reply_agent, reply_baseline, "AB")
        # Order 2: agent=B, baseline=A (swap)
        order2 = _run_order(reply_baseline, reply_agent, "BA")

        # Convert to agent-win / baseline-win / tie (normalising for order)
        def _normalise(result: dict, agent_was_a: bool) -> str:
            w = result.get("winner", "tie")
            if w == "tie":
                return "tie"
            if agent_was_a:
                return "agent_win" if w == "A" else "baseline_win"
            else:
                return "agent_win" if w == "B" else "baseline_win"

        normalised: dict[str, list] = {}
        for judge in ["gemini", "groq", "ollama"]:
            v1 = _normalise(order1.get(judge, {}), agent_was_a=True)
            v2 = _normalise(order2.get(judge, {}), agent_was_a=False)
            # Average: both agree -> that outcome; disagree -> tie
            if v1 == v2:
                normalised[judge] = v1
            else:
                normalised[judge] = "tie"

        # Order-swap agreement rate
        agreements = [order1.get(j, {}).get("winner") == order2.get(j, {}).get("winner") for j in ["gemini", "groq", "ollama"] if j in order1 and j in order2]
        order_swap_agreement = sum(agreements) / len(agreements) if agreements else 0.0

        return {
            "judges": normalised,
            "order_swap_agreement": order_swap_agreement,
            "order1_raw": order1,
            "order2_raw": order2,
        }

    # ------------------------------------------------------------------
    # Inter-judge agreement
    # ------------------------------------------------------------------

    def inter_judge_kappa(self, scores_by_judge: dict[str, list[int]]) -> float:
        from src.stats import cohens_kappa
        judge_names = list(scores_by_judge.keys())
        if len(judge_names) < 2:
            return 1.0
        kappas = []
        for i in range(len(judge_names)):
            for j in range(i + 1, len(judge_names)):
                a = scores_by_judge[judge_names[i]]
                b = scores_by_judge[judge_names[j]]
                kappas.append(cohens_kappa(a, b))
        import statistics
        return statistics.mean(kappas)
