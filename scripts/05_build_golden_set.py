# DESIGN RATIONALE
# [@E3_beyond2025groundtruth] Stratified sampling on 3 axes (intent × time-window × difficulty)
#   avoids the easy-example bias that inflates annotation accuracy.
# [@E1_consensus2026iaa] 50 double-annotated examples for intra/inter-annotator κ — which IAA
#   metric fits which task type, and how label imbalance distorts it.
# [@E4_whoannotates2026]   Good annotation reporting: two-stage adjudication, 3 annotators, 
#   consensus gold — copy the structure into the golden-set write-up.
# [@T06_labelstudio]       Label Studio config for a nice annotation UI.

"""
scripts/05_build_golden_set.py — stratified golden set construction.

HARD STOP after generating the template — you must annotate before continuing.

Outputs:
  data/golden/golden_set_template.csv   (NO labels — your annotation target)
  data/golden/annotation_guide.md
  data/golden/labelstudio_config.xml

Phase: 5 | Device: CPU
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

CONFIG_PATH = Path("brand_config.json")
THREADS_PATH = Path("data/processed/threads.jsonl")
LABELLED_TEST = Path("data/processed/labelled_test.jsonl")
TAXONOMY_PATH = Path("intent_taxonomy.json")
OUT_DIR = Path("data/golden")
OUT_TEMPLATE = OUT_DIR / "golden_set_template.csv"
ANNOTATION_GUIDE = OUT_DIR / "annotation_guide.md"
LABELSTUDIO_CONFIG = OUT_DIR / "labelstudio_config.xml"

SEED = 42
TOTAL_EXAMPLES = 200
N_DOUBLE_ANNOTATE = 50  # overlap set for κ
N_LOW_CONFIDENCE = 60
N_LONG_THREADS = 40   # >= 4 turns
N_VERY_SHORT = 20     # < 20 chars

ANNOTATION_GUIDE_CONTENT = """# Annotation Guide: Golden Set

## Task
Label each customer support message with:
1. `correct_intent`: the customer's actual intent (from the taxonomy)
2. `should_escalate`: whether a human agent should handle this (True/False)
3. `difficulty`: how hard the classification is (easy/medium/hard)
4. `notes`: optional free-text note

## Intent Taxonomy
See `intent_taxonomy.json` for the full list with descriptions.
Choose the intent that best matches what the customer actually wants.
If no intent fits, use `out_of_scope`.

## Escalation Rubric (answer YES to `should_escalate` if ANY of these apply)

| Condition | Example |
|---|---|
| **Billing/refund dispute** | "You charged me twice, I want a full refund" |
| **Legal/regulatory threat** | "I'll sue you", "I'm filing a complaint with the FCC" |
| **Safety/self-harm** | Any mention of harm to self or others |
| **Account takeover suspicion** | "Someone logged into my account without permission" |
| **Minor involved** | Customer explicitly mentions being a minor |
| **Repeated failure** | Brand has already replied 2+ times without resolving |
| **Genuinely ambiguous intent** | Even after reading 3 times, you cannot assign a confident intent |

Otherwise, answer NO: the agent can auto-handle.

## Difficulty Rubric

| Difficulty | Meaning |
|---|---|
| **easy** | Intent is unambiguous from the first sentence |
| **medium** | Intent requires reading the full message or context |
| **hard** | Ambiguous even with context; multiple valid interpretations |

## Double Annotation Protocol
50 examples are marked `double_annotate = True`.
Re-annotate them from scratch after at least 3 hours (or in a new session).
Do not look at your previous labels when re-annotating.
Disagreements will be adjudicated; do not silently overwrite.

## Schema
The template CSV has columns:
- `thread_id`: unique thread identifier
- `first_customer_message`: the message to classify
- `first_customer_created_at`: when it was sent
- `predicted_intent`: what the model predicted (for reference only — you may override)
- `model_confidence`: model's confidence score (for reference)
- `double_annotate`: True if this row needs a second pass
- `correct_intent`: **YOUR LABEL**
- `should_escalate`: **YOUR LABEL** (True/False)
- `difficulty`: **YOUR LABEL** (easy/medium/hard)
- `notes`: optional
"""

LABELSTUDIO_XML = """<View>
  <Text name="message" value="$first_customer_message"/>
  <Header value="Thread: $thread_id | Predicted: $predicted_intent ($model_confidence)"/>
  
  <Choices name="correct_intent" toName="message" choice="single" showInLine="true">
    <Choice value="out_of_scope"/>
    <!-- Intents will be populated from intent_taxonomy.json -->
    <Choice value="__INTENT_PLACEHOLDER__"/>
  </Choices>
  
  <Choices name="should_escalate" toName="message" choice="single" showInLine="true">
    <Header value="Should escalate?"/>
    <Choice value="True"/>
    <Choice value="False"/>
  </Choices>
  
  <Choices name="difficulty" toName="message" choice="single" showInLine="true">
    <Header value="Difficulty"/>
    <Choice value="easy"/>
    <Choice value="medium"/>
    <Choice value="hard"/>
  </Choices>
  
  <TextArea name="notes" toName="message" placeholder="Optional notes..." rows="2"/>
</View>"""


def build_golden_set(threads: list[dict], intent_names: list[str]) -> pd.DataFrame:
    """Stratified sample: low-confidence + long threads + very short + rest proportional."""
    rng = random.Random(SEED)

    # Sort threads by calibrated confidence (ascending = lowest confidence first)
    threads_sorted_conf = sorted(threads, key=lambda t: t.get("calibrated_conf", 1.0))

    # Stratum 1: lowest confidence (60)
    low_conf = threads_sorted_conf[:N_LOW_CONFIDENCE]
    low_conf_ids = {t["thread_id"] for t in low_conf}

    # Stratum 2: long threads (≥4 turns, 40)
    remaining = [t for t in threads if t["thread_id"] not in low_conf_ids]
    long_threads = [t for t in remaining if t.get("tweet_count", 0) >= 4]
    long_threads = rng.sample(long_threads, min(N_LONG_THREADS, len(long_threads)))
    long_ids = {t["thread_id"] for t in long_threads}

    # Stratum 3: very short messages (<20 chars, 20)
    remaining2 = [t for t in remaining if t["thread_id"] not in long_ids]
    very_short = [t for t in remaining2 if len(t.get("first_customer_message", "")) < 20]
    very_short = rng.sample(very_short, min(N_VERY_SHORT, len(very_short)))
    short_ids = {t["thread_id"] for t in very_short}

    # Stratum 4: proportional remainder
    used_ids = low_conf_ids | long_ids | short_ids
    n_remaining_needed = TOTAL_EXAMPLES - len(low_conf) - len(long_threads) - len(very_short)
    remaining3 = [t for t in threads if t["thread_id"] not in used_ids]
    proportional = rng.sample(remaining3, min(n_remaining_needed, len(remaining3)))

    all_selected = low_conf + long_threads + very_short + proportional
    rng.shuffle(all_selected)

    # Mark 50 for double annotation (spread across strata)
    double_ids = {t["thread_id"] for t in rng.sample(all_selected, min(N_DOUBLE_ANNOTATE, len(all_selected)))}

    rows = []
    for t in all_selected:
        rows.append({
            "thread_id": t["thread_id"],
            "first_customer_message": t["first_customer_message"],
            "first_customer_created_at": t.get("first_customer_created_at", ""),
            "predicted_intent": t.get("knn_label", ""),
            "model_confidence": round(t.get("calibrated_conf", 0.0), 4),
            "tweet_count": t.get("tweet_count", 0),
            "channel_switch": t.get("channel_switch", False),
            "double_annotate": t["thread_id"] in double_ids,
            # Blank columns for human annotation
            "correct_intent": "",
            "should_escalate": "",
            "difficulty": "",
            "notes": "",
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("\n" + "=" * 60)
    print("PHASE 5: Golden Set Construction")
    print("  Plan: §Phase 5 | .bib: @E3 @E1 @E4 @T06_labelstudio")
    print("  Device: CPU | Provider calls: 0")
    print("=" * 60)

    config = json.loads(CONFIG_PATH.read_text())
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    intent_names = [i["name"] for i in taxonomy["intents"]]

    print(f"\nLoading labelled test threads from {LABELLED_TEST} ...")
    threads = []
    with open(LABELLED_TEST) as f:
        for line in f:
            threads.append(json.loads(line))
    print(f"  {len(threads):,} test threads available")

    if len(threads) < TOTAL_EXAMPLES:
        print(f"ERROR: only {len(threads)} test threads; need ≥ {TOTAL_EXAMPLES}")
        sys.exit(1)

    print(f"\nBuilding stratified sample ({TOTAL_EXAMPLES} examples) ...")
    df = build_golden_set(threads, intent_names)
    print(f"  Selected {len(df)} examples")
    print(f"  Double-annotate set: {df['double_annotate'].sum()} examples")

    print(f"\nStrata breakdown:")
    print(f"  low-confidence (bottom calibrated_conf): {sum(1 for _, r in df.iterrows() if r['model_confidence'] < 0.5)}")
    print(f"  long threads (≥4 turns): {sum(1 for _, r in df.iterrows() if r['tweet_count'] >= 4)}")
    print(f"  very short (<20 chars): {sum(1 for _, r in df.iterrows() if len(str(r['first_customer_message'])) < 20)}")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TEMPLATE, index=False)
    print(f"\nWrote annotation template: {OUT_TEMPLATE}")

    ANNOTATION_GUIDE.write_text(ANNOTATION_GUIDE_CONTENT)
    print(f"Wrote annotation guide:    {ANNOTATION_GUIDE}")

    # Build Label Studio XML with actual intents
    intent_choices = "\n    ".join(f'<Choice value="{n}"/>' for n in intent_names)
    xml = LABELSTUDIO_XML.replace(
        '<Choice value="__INTENT_PLACEHOLDER__"/>',
        intent_choices,
    )
    LABELSTUDIO_CONFIG.write_text(xml)
    print(f"Wrote Label Studio config: {LABELSTUDIO_CONFIG}")

    # Phase report
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE: HARD STOP (HUMAN ANNOTATION REQUIRED)")
    print(f"  ran:      scripts/05_build_golden_set.py")
    print(f"  numbers:  {len(df)} examples | {df['double_annotate'].sum()} double-annotate")
    print(f"  VRAM:     N/A (CPU) | provider calls: 0")
    print(f"  .bib:     @E3_beyond2025groundtruth @E1_consensus2026iaa @E4_whoannotates2026")
    print(f"  status:   AWAITING HUMAN ANNOTATION")
    print("=" * 60)
    print()
    print("NEXT STEPS (required before Phase 6):")
    print(f"  1. Open:  {OUT_TEMPLATE}")
    print(f"  2. Read:  {ANNOTATION_GUIDE}")
    print(f"  3. Fill in 'correct_intent', 'should_escalate', 'difficulty' for all 200 rows")
    print(f"  4. Re-annotate the {N_DOUBLE_ANNOTATE} rows marked double_annotate=True after a break")
    print(f"  5. Save the file as: data/golden/golden_set.csv  (DO NOT rename columns)")
    print(f"  6. Then run: python scripts/06_evaluate.py")
    print()
    print("Do NOT proceed until data/golden/golden_set.csv exists with your labels.")


if __name__ == "__main__":
    main()
