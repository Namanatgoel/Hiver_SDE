"""tests/test_gateway.py — Phase 0 gateway tests

Tests:
1. Cache-hit behaviour: second call must produce 0 new provider calls.
2. Budget enforcement: calling past the daily limit raises RuntimeError.
3. Cache key is deterministic: same inputs always give the same key.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Each test gets its own cache dir and ledger."""
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir()
    ledger = tmp_path / "ledger.json"

    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("BUDGET_PER_DAY", "10")

    # Patch module-level constants after env vars are set
    import importlib
    import src.llm.gateway as gw
    monkeypatch.setattr(gw, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(gw, "LEDGER_PATH", ledger)
    monkeypatch.setattr(gw, "BUDGET_PER_DAY", 10)

    yield cache_dir, ledger


# ---------------------------------------------------------------------------
# Test 1: deterministic cache key
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    from src.llm.gateway import _cache_key
    msgs = [{"role": "user", "content": "hello"}]
    k1 = _cache_key("gemini", "gemini-2.5-flash", msgs, {"temperature": 0.0})
    k2 = _cache_key("gemini", "gemini-2.5-flash", msgs, {"temperature": 0.0})
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_differs_on_content():
    from src.llm.gateway import _cache_key
    msgs_a = [{"role": "user", "content": "hello"}]
    msgs_b = [{"role": "user", "content": "world"}]
    assert _cache_key("gemini", "m", msgs_a, {}) != _cache_key("gemini", "m", msgs_b, {})


# ---------------------------------------------------------------------------
# Test 2: cache hit returns cached response, 0 new provider calls
# ---------------------------------------------------------------------------


def test_cache_hit_costs_zero_calls(isolated_cache, monkeypatch):
    """A cache-warm re-run must make 0 provider calls."""
    from src.llm.gateway import _save_cache, _cache_key, complete

    msgs = [{"role": "user", "content": "test cache"}]
    params = {}
    key = _cache_key("gemini", "gemini-2.5-flash", msgs, params)

    # Seed the cache manually
    _save_cache(key, "gemini", "gemini-2.5-flash", "cached response", "test")

    call_count = {"n": 0}
    def fake_call(*args, **kwargs):
        call_count["n"] += 1
        return "live response"

    import src.llm.gateway as gw
    monkeypatch.setattr(gw, "_call_with_retry", lambda *a, **k: fake_call(*a, **k))
    monkeypatch.setattr(gw, "DISABLED", {p: False for p in gw.PROVIDERS})

    response, provider = complete(msgs, purpose="test")

    assert response == "cached response"
    assert "cached" in provider
    assert call_count["n"] == 0, "Cache hit must not make any provider calls"


# ---------------------------------------------------------------------------
# Test 3: live call writes to cache; second call is a cache hit
# ---------------------------------------------------------------------------


def test_live_call_then_cache_hit(isolated_cache, monkeypatch):
    from src.llm.gateway import complete

    msgs = [{"role": "user", "content": "live then cached"}]

    import src.llm.gateway as gw

    call_count = {"n": 0}
    def fake_provider(model, messages, params):
        call_count["n"] += 1
        return "live answer"

    monkeypatch.setattr(gw, "DISABLED", {p: p != "gemini" for p in gw.PROVIDERS})
    monkeypatch.setattr(gw, "_call_with_retry", lambda p, m, msgs, params: fake_provider(m, msgs, params))
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    # First call — hits provider
    r1, p1 = complete(msgs, purpose="first")
    assert r1 == "live answer"
    assert call_count["n"] == 1

    # Second call — must hit cache
    r2, p2 = complete(msgs, purpose="second")
    assert r2 == "live answer"
    assert call_count["n"] == 1, "Second call must use cache, not provider"
    assert "cached" in p2


# ---------------------------------------------------------------------------
# Test 4: budget exhaustion raises RuntimeError
# ---------------------------------------------------------------------------


def test_budget_exhaustion(isolated_cache, monkeypatch):
    from src.llm.gateway import _check_and_increment

    import src.llm.gateway as gw
    monkeypatch.setattr(gw, "BUDGET_PER_DAY", 3)

    for _ in range(3):
        _check_and_increment("gemini")  # should succeed

    with pytest.raises(RuntimeError, match="budget"):
        _check_and_increment("gemini")  # 4th call must fail
