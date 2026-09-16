# DESIGN RATIONALE
# [@D2_prabhakar2018towards] Handle detection via verified list + mention regex — the only paper
#   on this corpus shows naive inbound==False picks the wrong brand (NaN rows + mention rows).
#   Also: drop DM-redirect replies and split by TIME (train=older, test=newer) to avoid leakage.
# [@D1_axelbrooke2017twcs]   Primary corpus — fields tweet_id, author_id, inbound, created_at,
#   text, response_tweet_id, in_response_to_tweet_id; PII already masked.

"""
scripts/01_eda.py — EDA, brand selection, and time split.

Outputs: brand_config.json
Device:  CPU
Phase:   1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/raw/twcs.csv")
OUT_CONFIG = Path("brand_config.json")
TIME_SPLIT = "2017-11-15"
DEDUP_THRESHOLD = 0.95
TOP_N_BRANDS = 10

# Verified brand handles from the TWCS dataset (from Axelbrooke 2017 dataset docs)
KNOWN_BRAND_HANDLES = {
    "applesupport", "applsupport", "amazonhelp", "AmazonHelp", "sprintcare",
    "tmobilehelp", "tmobileus", "ask_spectrum", "xboxsupport", "hulu_support",
    "spotifycares", "comcastcares", "delta", "deltassist", "united",
    "americanair", "southwestair", "virgintrains", "bostoncalling",
    "samsungsupport", "sonyplaystation", "microsoftsupport", "googlepixelteam",
    "lyftsupport", "uberhelp", "uber_support", "tesco", "british_airways",
    "airbnbhelp", "hiltonhelps", "marriottbonvoy", "bankofamerica",
    "chasebank", "wellsfargo", "usairwaysteam", "nvidiahelp", "gstore",
}

# DM redirect patterns to drop
DM_REDIRECT_RE = re.compile(
    r"(please\s+(dm|direct\s+message)|send\s+(us|a)\s+(dm|direct\s+message)"
    r"|click\s+here\s+to\s+chat|please\s+follow\s+(us\s+)?and\s+(dm|direct))",
    re.IGNORECASE,
)

URL_ONLY_RE = re.compile(
    r"^\s*(https?://\S+\s*)+$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    # Normalise inbound to bool
    df["inbound"] = df["inbound"].map({"True": True, "False": False, True: True, False: False})
    # Parse timestamps with explicit Twitter API format for high performance
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def detect_brands(df: pd.DataFrame) -> pd.DataFrame:
    """Detect brand accounts using both handle list and @mention regex."""
    author_lower = df["author_id"].str.lower().fillna("")

    # Method 1: handle list match
    handle_set = {h.lower() for h in KNOWN_BRAND_HANDLES}
    method1 = df[~df["inbound"].fillna(True) & author_lower.isin(handle_set)]["author_id"].str.lower()

    # Method 2: @mention regex on inbound messages
    mention_counts: dict[str, int] = {}
    inbound_texts = df[df["inbound"].fillna(False)]["text"].dropna()
    for text in inbound_texts:
        for handle in re.findall(r"@(\w+)", text):
            h = handle.lower()
            mention_counts[h] = mention_counts.get(h, 0) + 1

    # Count outbound by author
    outbound = df[df["inbound"] == False].copy()  # noqa: E712
    brand_counts = outbound["author_id"].str.lower().value_counts()

    # Build report table
    all_handles = set(brand_counts.index.tolist()) | set(mention_counts.keys())
    rows = []
    for h in all_handles:
        rows.append({
            "handle": h,
            "outbound_count": brand_counts.get(h, 0),
            "mention_count": mention_counts.get(h, 0),
            "in_known_list": h in handle_set,
        })

    brand_df = (
        pd.DataFrame(rows)
        .sort_values("outbound_count", ascending=False)
        .reset_index(drop=True)
    )
    return brand_df


def hygiene_filter(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Apply corpus hygiene rules from @D2_prabhakar2018towards."""
    brand_mask = df["author_id"].str.lower() == brand_handle.lower()
    brand_replies = df[brand_mask].copy()

    n_start = len(brand_replies)

    # 1. Drop DM-redirect-only replies
    is_dm_only = brand_replies["text"].str.strip().apply(
        lambda t: bool(DM_REDIRECT_RE.search(str(t))) and len(str(t).split()) < 25
    )
    brand_replies = brand_replies[~is_dm_only]
    print(f"  Dropped {is_dm_only.sum():,} DM-redirect replies")

    # 2. Drop URL-only / empty
    is_url_empty = brand_replies["text"].str.strip().apply(
        lambda t: not str(t).strip() or bool(URL_ONLY_RE.match(str(t)))
    )
    brand_replies = brand_replies[~is_url_empty]
    print(f"  Dropped {is_url_empty.sum():,} URL-only/empty replies")

    # 3. Deduplicate near-identical brand replies (rapidfuzz >= 0.95 similarity)
    # Fast length + prefix bucketing avoids O(N^2) comparison on 100k+ strings
    norm_texts = brand_replies["text"].str.replace(r"^(@\w+\s*)+", "", regex=True).str.strip()
    unique_texts = norm_texts.unique()
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for t in unique_texts:
        k = (t[:15].lower(), len(t) // 10)
        groups[k].append(t)

    duplicates: set[str] = set()
    for (prefix, l_bin), bucket in groups.items():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            t1 = bucket[i]
            if t1 in duplicates:
                continue
            for j in range(i + 1, len(bucket)):
                t2 = bucket[j]
                if t2 in duplicates:
                    continue
                if fuzz.ratio(t1, t2) >= DEDUP_THRESHOLD * 100.0:
                    duplicates.add(t2)

    is_near_dup = norm_texts.isin(duplicates)
    brand_replies = brand_replies[~is_near_dup]
    print(f"  Dropped {is_near_dup.sum():,} near-duplicate replies (≥{DEDUP_THRESHOLD:.0%} similarity)")

    n_end = len(brand_replies)
    print(f"  Hygiene: {n_start:,} → {n_end:,} brand reply rows ({n_start - n_end:,} removed)")
    return brand_replies


def time_split_counts(df: pd.DataFrame, brand_handle: str) -> tuple[int, int]:
    """Count inbound messages in train/test windows for the chosen brand."""
    # Get all tweet_ids that the brand replied to
    brand_reply_ids = set(
        df[df["author_id"].str.lower() == brand_handle.lower()]["in_response_to_tweet_id"].dropna()
    )
    # Inbound messages addressed to brand
    inbound = df[df["inbound"].fillna(False) & df["tweet_id"].isin(brand_reply_ids)]
    cutoff = pd.Timestamp(TIME_SPLIT, tz="UTC")
    train = inbound[inbound["created_at"] < cutoff]
    test = inbound[inbound["created_at"] >= cutoff]
    return len(train), len(test)


def prompt_brand_choice(brand_df: pd.DataFrame, default_choice: str | None = None) -> str:
    """Display top brands and ask user to confirm the choice."""
    if default_choice:
        return default_choice.lower().lstrip("@")

    print(f"\nTop {TOP_N_BRANDS} brands by outbound message count:")
    print(f"{'#':<4} {'handle':<25} {'outbound':>10} {'mentions':>10} {'known_list':>12}")
    print("-" * 65)
    for i, row in brand_df.head(TOP_N_BRANDS).iterrows():
        flag = "✓" if row["in_known_list"] else " "
        print(f"{i+1:<4} {row['handle']:<25} {int(row['outbound_count']):>10,} "
              f"{int(row['mention_count']):>10,} {flag:>12}")

    auto_brand = brand_df.iloc[0]["handle"]
    print(f"\nAuto-selected: #1 — {auto_brand!r}")
    try:
        answer = input("\nPress ENTER to confirm, or type a different number/handle: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"Non-interactive session: proceeding with auto-selected {auto_brand!r}")
        return auto_brand

    if not answer:
        return auto_brand
    elif answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(brand_df):
            return brand_df.iloc[idx]["handle"]
        else:
            print(f"Invalid choice {answer}, using auto-selection.")
            return auto_brand
    else:
        return answer.lower().lstrip("@")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: EDA, brand selection, time split")
    parser.add_argument("--brand", type=str, default=None, help="Brand handle to select (bypasses prompt)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 1: EDA + Brand Selection + Time Split")
    print(f"  Plan ref: §Phase 1 | .bib: @D2_prabhakar2018towards @D1_axelbrooke2017twcs")
    print(f"  Device: CPU")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print(f"\n[Brand Detection]")
    brand_df = detect_brands(df)

    chosen = prompt_brand_choice(brand_df, default_choice=args.brand)
    print(f"\nSelected brand: {chosen!r}")

    print(f"\n[Corpus Hygiene] Applying @D2 rules to brand={chosen!r}")
    _ = hygiene_filter(df, chosen)

    print(f"\n[Time Split] cutoff={TIME_SPLIT}")
    train_n, test_n = time_split_counts(df, chosen)
    print(f"  Train inbound (< {TIME_SPLIT}): {train_n:,}")
    print(f"  Test  inbound (≥ {TIME_SPLIT}): {test_n:,}")

    if train_n < 20_000:
        print(f"\n⚠  STOP: brand {chosen!r} has only {train_n:,} train messages (need ≥ 20k)")
        print("   Please re-run and choose a different brand.")
        sys.exit(1)

    if test_n < 3_000:
        print(f"\n⚠  STOP: brand {chosen!r} has only {test_n:,} test messages (need ≥ 3k)")
        print("   Please re-run and choose a different brand.")
        sys.exit(1)

    config = {
        "brand_handle": chosen,
        "time_split_cutoff": TIME_SPLIT,
        "train_inbound_count": train_n,
        "test_inbound_count": test_n,
        "top_brands": brand_df.head(TOP_N_BRANDS).to_dict(orient="records"),
    }
    OUT_CONFIG.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {OUT_CONFIG}")

    # Phase report
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"  ran:      scripts/01_eda.py")
    print(f"  numbers:  brand={chosen!r} | train={train_n:,} | test={test_n:,}")
    print(f"  VRAM:     N/A (CPU) | provider calls: 0")
    print(f"  .bib:     @D2_prabhakar2018towards @D1_axelbrooke2017twcs")
    accept = train_n >= 20_000 and test_n >= 3_000
    print(f"  status:   {'ACCEPT' if accept else 'REJECT'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

