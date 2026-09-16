# DESIGN RATIONALE
# [@D2_prabhakar2018towards] Union-find thread reconstruction from response_tweet_id chains —
#   the only prior paper on this exact corpus; its thread structure is the canonical reference.
#   channel_switch kept as its own field (v1 mistakenly marked it as "resolved").

"""
scripts/02_build_threads.py — thread reconstruction via union-find.

Outputs: data/processed/threads.jsonl
Device:  CPU
Phase:   2
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
CONFIG_PATH = Path("brand_config.json")
OUT_PATH = Path("data/processed/threads.jsonl")


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        root = x
        while root in self._parent and self._parent[root] != root:
            root = self._parent[root]
        curr = x
        while curr in self._parent and self._parent[curr] != root:
            nxt = self._parent[curr]
            self._parent[curr] = root
            curr = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


DM_REDIRECT_RE = __import__("re").compile(
    r"(please\s+(dm|direct\s+message)|send\s+(us|a)\s+(dm|direct\s+message)"
    r"|click\s+here\s+to\s+chat|please\s+follow\s+(us\s+)?and\s+(dm|direct))",
    __import__("re").IGNORECASE,
)


def is_dm_redirect(text: str) -> bool:
    return bool(DM_REDIRECT_RE.search(str(text))) and len(str(text).split()) < 25


def build_threads(df: pd.DataFrame, brand_handle: str) -> list[dict]:
    """Reconstruct conversation threads using union-find on reply chains."""
    uf = UnionFind()
    brand_lower = brand_handle.lower()

    # Build edges from reply relationships fast
    tids = df["tweet_id"].astype(str).values
    resps = df["response_tweet_id"].fillna("").astype(str).values
    inresps = df["in_response_to_tweet_id"].fillna("").astype(str).values

    for tid, resp, inresp in zip(tids, resps, inresps):
        if resp and resp != "nan":
            for r in resp.split(","):
                r = r.strip()
                if r:
                    uf.union(tid, r)
        if inresp and inresp != "nan":
            uf.union(tid, inresp)

    # Prune to roots that contain at least one message by the brand
    brand_tids = df[df["author_id"].str.lower() == brand_lower]["tweet_id"].astype(str).values
    brand_roots = {uf.find(t) for t in brand_tids}

    # Group only tweets that belong to brand_roots
    groups: dict[str, list] = defaultdict(list)
    # Convert relevant subset to dicts
    for row in df.itertuples(index=False):
        tid = str(row.tweet_id)
        root = uf.find(tid)
        if root in brand_roots:
            groups[root].append({
                "tweet_id": tid,
                "author_id": str(row.author_id),
                "inbound": row.inbound,
                "created_at": row.created_at,
                "text": str(row.text),
                "response_tweet_id": str(row.response_tweet_id),
                "in_response_to_tweet_id": str(row.in_response_to_tweet_id),
            })

    # Build thread objects
    brand_lower = brand_handle.lower()
    threads = []
    for root, tweets in groups.items():
        # Sort by created_at
        tweets_sorted = sorted(
            tweets,
            key=lambda r: r.get("created_at") if pd.notna(r.get("created_at")) else pd.Timestamp.min,
        )

        has_customer = any(
            str(t.get("inbound", "")).lower() in ("true", "1") for t in tweets_sorted
        )
        brand_replies = [
            t for t in tweets_sorted
            if str(t.get("author_id", "")).lower() == brand_lower
            and not is_dm_redirect(str(t.get("text", "")))
        ]
        has_brand = len(brand_replies) > 0

        if not (has_customer and has_brand):
            continue

        # Detect channel switch (customer or brand mentions moving to DM)
        channel_switch = any(
            is_dm_redirect(str(t.get("text", "")))
            for t in tweets_sorted
            if str(t.get("author_id", "")).lower() == brand_lower
        )

        first_customer = next(
            (t for t in tweets_sorted if str(t.get("inbound", "")).lower() in ("true", "1")),
            None,
        )

        thread = {
            "thread_id": root,
            "tweet_count": len(tweets_sorted),
            "customer_turn_count": sum(
                1 for t in tweets_sorted if str(t.get("inbound", "")).lower() in ("true", "1")
            ),
            "brand_turn_count": len(brand_replies),
            "channel_switch": channel_switch,
            "first_customer_message": str(first_customer.get("text", "")) if first_customer else "",
            "first_customer_created_at": str(first_customer.get("created_at", "")) if first_customer else "",
            "brand_replies": [
                {
                    "tweet_id": str(t.get("tweet_id", "")),
                    "text": str(t.get("text", "")),
                    "created_at": str(t.get("created_at", "")),
                }
                for t in brand_replies
            ],
            "turns": [
                {
                    "tweet_id": str(t.get("tweet_id", "")),
                    "author_id": str(t.get("author_id", "")),
                    "inbound": str(t.get("inbound", "")),
                    "text": str(t.get("text", "")),
                    "created_at": str(t.get("created_at", "")),
                }
                for t in tweets_sorted
            ],
        }
        threads.append(thread)

    return threads


def main() -> None:
    print("\n" + "=" * 60)
    print("PHASE 2: Thread Reconstruction (union-find)")
    print("  Plan ref: §Phase 2 | .bib: @D2_prabhakar2018towards")
    print("  Device: CPU")
    print("=" * 60)

    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found — run Phase 1 first.", file=sys.stderr)
        sys.exit(1)

    config = json.loads(CONFIG_PATH.read_text())
    brand_handle = config["brand_handle"]
    print(f"\nBrand: {brand_handle!r}")

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, dtype=str, low_memory=False)
    df["inbound"] = df["inbound"].map({"True": True, "False": False, True: True, False: False})
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    print(f"  {len(df):,} rows")

    print("\nBuilding threads ...")
    threads = build_threads(df, brand_handle)
    print(f"  Found {len(threads):,} valid threads")

    if len(threads) < 8_000:
        print(f"\n⚠  STOP: only {len(threads):,} threads (need ≥ 8k)")
        sys.exit(1)

    # Stats
    turn_counts = [t["tweet_count"] for t in threads]
    avg_turns = sum(turn_counts) / len(turn_counts)
    channel_switches = sum(1 for t in threads if t["channel_switch"])
    print(f"  Avg turns per thread: {avg_turns:.2f}")
    print(f"  Threads with channel_switch: {channel_switches:,}")

    # Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for thread in threads:
            f.write(json.dumps(thread) + "\n")
    print(f"\nWrote {len(threads):,} threads → {OUT_PATH}")

    # Phase report
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print(f"  ran:      scripts/02_build_threads.py")
    print(f"  numbers:  threads={len(threads):,} | avg_turns={avg_turns:.2f} | channel_switch={channel_switches:,}")
    print(f"  VRAM:     N/A (CPU) | provider calls: 0")
    print(f"  .bib:     @D2_prabhakar2018towards")
    print(f"  status:   {'ACCEPT' if len(threads) >= 8_000 else 'REJECT'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
