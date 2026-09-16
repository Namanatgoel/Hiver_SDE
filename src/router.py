# DESIGN RATIONALE
# [@C13_rebedea2023nemoguardrails] Hard rails for billing/legal/safety — deterministic rails
#   intercepted 70-95% of unanswerable prompts; "escalate by rule, not vibes".
# [@C6_angelopoulos2021conformal] Conformal threshold τ — lets us state "auto-handled subset has
#   ≤α error with probability ≥1-δ"; highest value-per-page upgrade over a hand-picked 0.4.
# [@C1_wen2025abstention]     Coverage@Accuracy and risk-coverage curves — map of metrics to report.
# [@C2_bachar2026lpp]         Attribution-based reason codes over boolean — makes escalation auditable.
# [@C7_xiong2023uncertainty]  Do NOT gate on raw verbalised confidence — poorly calibrated.

"""
src/router.py — three-layer routing: hard rails -> calibrated gate -> reason codes.

Layer 1: regex/policy hard rails (always escalate, fixed reason code)
Layer 2: calibrated conformal gate (tau from calibration split)
Layer 3: reason codes for attribution

Device: CPU (rules + threshold lookup; no GPU)
Phase:  6
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hard rail patterns ([@C13_rebedea2023nemoguardrails])
# ---------------------------------------------------------------------------

RAIL_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("billing_legal", "billing_legal",
     re.compile(
         r"(refund|charge[d]?|bill[ed]?|fraud|unauthorized|sue|lawsuit|attorney|lawyer"
         r"|court|regulator|fcc|ftc|bbb|consumer\s+protection|arbitration)",
         re.IGNORECASE,
     )),
    ("safety", "safety",
     re.compile(
         r"(kill\s+myself|suicide|self.harm|hurt\s+myself|end\s+my\s+life"
         r"|don.t\s+want\s+to\s+live)",
         re.IGNORECASE,
     )),
    ("account_takeover", "account_takeover",
     re.compile(
         r"(hacked|someone\s+(else\s+)?logged\s+in|unauthorized\s+(access|login)"
         r"|account\s+(stolen|compromised|takeover))",
         re.IGNORECASE,
     )),
    ("minor", "minor",
     re.compile(r"\b(i\s+am\s+|i'm\s+)?(a\s+)?(minor|under\s+18|underage|child)\b", re.IGNORECASE)),
    ("pii_sensitive", "pii_sensitive",
     re.compile(
         r"(social\s+security|ssn|passport\s+number|credit\s+card\s+number|\bccn\b"
         r"|\bcvv\b|bank\s+account\s+number)",
         re.IGNORECASE,
     )),
    ("competitor_abuse", "competitor_abuse",
     re.compile(
         r"(competitor|switch\s+to|moving\s+to|cancell?ing\s+and\s+(going|switching))",
         re.IGNORECASE,
     )),
]


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    action: str          # "auto_handle" | "escalate"
    reason_code: str     # why this decision was made
    layer: int           # 1=hard rail, 2=gate, 3=reason
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """
    Three-layer router. Tau is set by conformal calibration in src/calibrate.py.
    """

    def __init__(self, tau: float = 0.5) -> None:
        self.tau = tau

    # ------------------------------------------------------------------
    # Layer 1: hard rails
    # ------------------------------------------------------------------

    def _check_hard_rails(self, message: str) -> RouteDecision | None:
        for name, code, pattern in RAIL_PATTERNS:
            if pattern.search(message):
                return RouteDecision(
                    action="escalate",
                    reason_code=code,
                    layer=1,
                    details={"rail": name, "pattern": pattern.pattern[:60]},
                )
        return None

    # ------------------------------------------------------------------
    # Layer 2: calibrated gate ([@C6_angelopoulos2021conformal])
    # ------------------------------------------------------------------

    def _check_gate(
        self,
        calibrated_conf: float,
        retrieval_empty: bool,
        intent_low_conf: bool,
        low_groundedness: bool,
        repeated_failure: bool,
    ) -> RouteDecision | None:
        if retrieval_empty:
            return RouteDecision("escalate", "retrieval_empty", layer=2, details={"conf": calibrated_conf})
        if calibrated_conf < self.tau:
            # Determine the most informative reason code
            if intent_low_conf:
                code = "low_intent_confidence"
            elif low_groundedness:
                code = "low_groundedness"
            elif repeated_failure:
                code = "repeated_failure"
            else:
                code = "low_calibrated_confidence"
            return RouteDecision("escalate", code, layer=2, details={"conf": calibrated_conf, "tau": self.tau})
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        message: str,
        calibrated_conf: float,
        retrieval_results: list[dict],
        groundedness_score: float = 1.0,
        repeated_failure: bool = False,
        intent_margin: float = 1.0,
    ) -> RouteDecision:
        """
        Route a message. Returns a RouteDecision with action and reason_code.
        Does NOT use raw verbalised confidence ([@C7_xiong2023uncertainty]).
        """
        # Layer 1: hard rails
        rail = self._check_hard_rails(message)
        if rail is not None:
            return rail

        # Derived signals for gate
        retrieval_empty = len(retrieval_results) == 0
        intent_low_conf = intent_margin < 0.1  # very small margin between top-2 intents
        low_groundedness = groundedness_score < 0.3

        # Layer 2: calibrated gate
        gate = self._check_gate(
            calibrated_conf, retrieval_empty, intent_low_conf, low_groundedness, repeated_failure
        )
        if gate is not None:
            return gate

        # Layer 3: auto-handle
        return RouteDecision(
            action="auto_handle",
            reason_code="passed_all_gates",
            layer=3,
            details={"conf": calibrated_conf, "tau": self.tau},
        )

    def reason_code_distribution(self, decisions: list[RouteDecision]) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(d.reason_code for d in decisions))
