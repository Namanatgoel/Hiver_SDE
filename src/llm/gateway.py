# DESIGN RATIONALE
# [@C12_chen2023frugalgpt] JSONL response cache + provider cascade — FrugalGPT shows cascade
#   architectures cut cost while maintaining quality; our cache makes re-runs free (0 calls).
# [@T16_ollama] cu128/cu130 torch pin + sm_120 check — RTX 5050 Blackwell needs PyTorch >= 2.7
#   built with CUDA >= 12.8; default wheels silently fail with "no kernel image" errors.

"""
src/llm/gateway.py — LLM gateway with JSONL cache, provider chain, and budget ledger.

Provider chain: gemini -> groq -> cerebras -> ollama
Cache: sha256(provider+model+messages+params) keyed JSONL in .cache/llm/
Budget: budget_ledger.json, hard daily stop per provider
Backoff: exponential on 429, honours x-ratelimit-remaining-* headers
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import tenacity
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache/llm"))
LEDGER_PATH = Path("budget_ledger.json")
BUDGET_PER_DAY = int(os.getenv("BUDGET_PER_DAY", "250"))

PROVIDERS = ["gemini", "groq", "cerebras", "ollama"]

# Models to use per provider (overridable via env)
DEFAULT_MODELS: dict[str, str] = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    "groq": os.getenv("GROQ_MODEL", "groq/compound"),
    "cerebras": os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
    "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
}

# Per-provider disable flags (set DISABLE_GEMINI=1 etc.)
DISABLED: dict[str, bool] = {
    p: bool(int(os.getenv(f"DISABLE_{p.upper()}", "0"))) for p in PROVIDERS
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(provider: str, model: str, messages: list[dict], params: dict) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, "messages": messages, "params": params},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.jsonl"


def _load_cache(key: str) -> str | None:
    p = _cache_path(key)
    if p.exists():
        data = json.loads(p.read_text().strip().splitlines()[-1])
        return data["response"]
    return None


def _save_cache(key: str, provider: str, model: str, response: str, purpose: str) -> None:
    record = {
        "key": key,
        "provider": provider,
        "model": model,
        "response": response,
        "purpose": purpose,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _cache_path(key).write_text(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Budget ledger
# ---------------------------------------------------------------------------


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {}


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2))


def _check_and_increment(provider: str) -> None:
    """Raise if daily budget exceeded; otherwise increment the counter."""
    ledger = _load_ledger()
    today = _today_utc()
    day_key = f"{today}/{provider}"
    count = ledger.get(day_key, 0)
    if count >= BUDGET_PER_DAY:
        raise RuntimeError(
            f"Daily budget exhausted for {provider}: {count}/{BUDGET_PER_DAY} calls today."
        )
    ledger[day_key] = count + 1
    _save_ledger(ledger)
    log.debug("Budget %s: %d/%d", day_key, count + 1, BUDGET_PER_DAY)


# ---------------------------------------------------------------------------
# Provider call implementations
# ---------------------------------------------------------------------------


def _call_gemini(model: str, messages: list[dict], params: dict) -> str:
    import google.genai as genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    # Convert messages to Gemini Content format
    contents = []
    system_text = None
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system_text = content
        elif role == "user":
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))
        elif role == "assistant":
            contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))

    # Build GenerateContentConfig with only the fields the new SDK accepts
    cfg_kwargs: dict[str, Any] = {}
    if "temperature" in params:
        cfg_kwargs["temperature"] = params["temperature"]
    if "max_output_tokens" in params:
        cfg_kwargs["max_output_tokens"] = params["max_output_tokens"]
    if "top_p" in params:
        cfg_kwargs["top_p"] = params["top_p"]
    if system_text:
        cfg_kwargs["system_instruction"] = system_text

    config = genai_types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    return resp.text


def _call_groq(model: str, messages: list[dict], params: dict) -> str:
    from groq import Groq  # type: ignore

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    allowed = {"temperature", "max_tokens", "top_p", "stop"}
    kw = {k: v for k, v in params.items() if k in allowed}
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    return resp.choices[0].message.content


def _call_cerebras(model: str, messages: list[dict], params: dict) -> str:
    from cerebras.cloud.sdk import Cerebras  # type: ignore

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise EnvironmentError("CEREBRAS_API_KEY not set")
    client = Cerebras(api_key=api_key)
    allowed = {"temperature", "max_tokens", "top_p"}
    kw = {k: v for k, v in params.items() if k in allowed}
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    return resp.choices[0].message.content


def _call_ollama(model: str, messages: list[dict], params: dict) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {k: v for k, v in params.items() if k in ("temperature", "top_p", "num_predict")},
    }
    resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_PROVIDER_FNS = {
    "gemini": _call_gemini,
    "groq": _call_groq,
    "cerebras": _call_cerebras,
    "ollama": _call_ollama,
}


# ---------------------------------------------------------------------------
# Retry logic — honours x-ratelimit-remaining-* headers
# ---------------------------------------------------------------------------


def _extract_retry_after(exc: Exception) -> float:
    """Return seconds to wait from rate-limit headers or exponential default."""
    for attr in ("response", "headers"):
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        remaining = headers.get("x-ratelimit-remaining-requests", None)
        reset_ms = headers.get("x-ratelimit-reset-requests", None)
        if remaining is not None and int(remaining) == 0 and reset_ms is not None:
            # reset_ms is usually "Xs" or "Xms"
            try:
                val = reset_ms.rstrip("ms").rstrip("s")
                secs = float(val) / 1000 if "ms" in reset_ms else float(val)
                return min(secs + 1, 60)
            except ValueError:
                pass
    return 0  # let tenacity's wait handle it


def _should_retry(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "too many", "quota", "overloaded"))


def _call_with_retry(provider: str, model: str, messages: list[dict], params: dict) -> str:
    fn = _PROVIDER_FNS[provider]
    last_exc: Exception | None = None

    for attempt in range(5):
        try:
            return fn(model, messages, params)
        except Exception as exc:
            if not _should_retry(exc):
                raise
            wait = _extract_retry_after(exc) or (2 ** attempt)
            log.warning("Provider %s 429/rate-limit (attempt %d); sleeping %.1fs", provider, attempt + 1, wait)
            last_exc = exc
            time.sleep(wait)

    raise RuntimeError(f"Provider {provider} failed after 5 retries") from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    purpose: str = "unspecified",
    params: dict | None = None,
    force_provider: str | None = None,
) -> tuple[str, str]:
    """
    Call the LLM provider chain with caching and budget enforcement.

    Returns (response_text, provider_used).

    Cache contract: identical inputs always return the cached response; no provider
    call is made. A cache-warm re-run costs exactly 0 provider calls.
    """
    params = params or {}
    providers_to_try = [force_provider] if force_provider else PROVIDERS

    # Try cache first (provider-agnostic: any cached hit wins)
    for provider in providers_to_try:
        if DISABLED.get(provider):
            continue
        m = model or DEFAULT_MODELS[provider]
        key = _cache_key(provider, m, messages, params)
        cached = _load_cache(key)
        if cached is not None:
            log.debug("Cache HIT key=%s provider=%s", key[:12], provider)
            return cached, f"{provider}(cached)"

    # No cache hit — call live, walking the chain
    errors: list[str] = []
    for provider in providers_to_try:
        if DISABLED.get(provider):
            continue
        m = model or DEFAULT_MODELS[provider]
        key = _cache_key(provider, m, messages, params)
        try:
            _check_and_increment(provider)
            log.info("Calling provider=%s model=%s purpose=%s", provider, m, purpose)
            response = _call_with_retry(provider, m, messages, params)
            _save_cache(key, provider, m, response, purpose)
            log.info("Provider %s responded (%d chars)", provider, len(response))
            return response, provider
        except RuntimeError as exc:
            if "budget" in str(exc).lower():
                log.warning("Budget exhausted for %s, trying next provider", provider)
                errors.append(f"{provider}: budget exhausted")
                continue
            raise
        except EnvironmentError as exc:
            log.warning("Provider %s skipped: %s", provider, exc)
            errors.append(f"{provider}: {exc}")
            continue
        except Exception as exc:
            log.warning("Provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")
            continue

    raise RuntimeError(
        f"All providers failed or unavailable. Errors: {'; '.join(errors)}"
    )


def cache_stats() -> dict:
    """Return cache hit counts and total cached entries."""
    files = list(CACHE_DIR.glob("*.jsonl"))
    providers: dict[str, int] = {}
    for f in files:
        try:
            data = json.loads(f.read_text().strip().splitlines()[-1])
            p = data.get("provider", "unknown")
            providers[p] = providers.get(p, 0) + 1
        except Exception:
            pass
    return {"total_cached": len(files), "by_provider": providers}


def ledger_summary() -> dict:
    """Return today's call counts per provider."""
    ledger = _load_ledger()
    today = _today_utc()
    return {k.split("/")[1]: v for k, v in ledger.items() if k.startswith(today)}
