"""scripts/smoke_gateway.py — Phase 0 smoke test

Makes 3 real gateway calls, then re-runs them and asserts 0 new provider calls.
Prints a 5-line phase report at the end.

Usage:
    conda activate hiver_sde
    python scripts/smoke_gateway.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.gateway import cache_stats, complete, ledger_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SMOKE_MESSAGES = [
    [{"role": "user", "content": "Reply with exactly: SMOKE_TEST_1"}],
    [{"role": "user", "content": "Reply with exactly: SMOKE_TEST_2"}],
    [{"role": "user", "content": "Reply with exactly: SMOKE_TEST_3"}],
]


def main() -> None:
    print("\n" + "=" * 60)
    print("PHASE 0 SMOKE TEST")
    print("=" * 60)

    # --- Run 1: cold (hits live providers) ---
    print("\n[Run 1] Cold run — expect 3 provider calls")
    t0 = time.time()
    results: list[tuple[str, str]] = []
    for i, msgs in enumerate(SMOKE_MESSAGES):
        resp, provider = complete(msgs, purpose=f"smoke_test_{i+1}")
        results.append((resp, provider))
        print(f"  call {i+1}: provider={provider!r}  response_len={len(resp)}")

    cold_elapsed = time.time() - t0
    stats_after_cold = ledger_summary()
    cache_after_cold = cache_stats()
    print(f"  Cold run elapsed: {cold_elapsed:.2f}s")
    print(f"  Provider calls today: {stats_after_cold}")
    print(f"  Cache entries: {cache_after_cold['total_cached']}")

    # --- Run 2: warm (must hit cache) ---
    print("\n[Run 2] Warm run — expect 0 new provider calls")
    t1 = time.time()
    calls_before = sum(stats_after_cold.values())

    for i, msgs in enumerate(SMOKE_MESSAGES):
        resp2, provider2 = complete(msgs, purpose=f"smoke_test_{i+1}_rerun")
        assert "cached" in provider2, (
            f"Expected cache hit on call {i+1}, got provider={provider2!r}"
        )
        assert resp2 == results[i][0], (
            f"Cache returned different content on call {i+1}!"
        )
        print(f"  call {i+1}: provider={provider2!r}  ✓ cache hit")

    warm_elapsed = time.time() - t1
    stats_after_warm = ledger_summary()
    calls_after = sum(stats_after_warm.values())
    new_calls = calls_after - calls_before

    if new_calls != 0:
        print(f"\nFAIL: warm run made {new_calls} new provider calls (expected 0)")
        sys.exit(1)

    # --- Phase 0 report ---
    print("\n" + "=" * 60)
    print("PHASE 0 COMPLETE")
    print(f"  ran:      scripts/smoke_gateway.py")
    print(f"  numbers:  3 cold calls OK; 3 warm calls = 0 new provider calls")
    print(f"  VRAM:     N/A (CPU-only phase) | provider calls: {calls_before} cold + 0 warm")
    print(f"  .bib:     @C12_chen2023frugalgpt @T16_ollama")
    status = "ACCEPT" if new_calls == 0 else "REJECT"
    print(f"  status:   {status}")
    print("=" * 60)

    if new_calls != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
