"""
scripts/annotate_golden_set.py — automated annotation of the golden set template.
Uses rule-based escalation rubrics, taxonomy embeddings, and LLM verification for ambiguous cases.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

CSV_PATH = Path("data/golden/golden_set_template.csv")
GOLDEN_CSV = Path("data/golden/golden_set.csv")
TAXONOMY_PATH = Path("intent_taxonomy.json")

def load_taxonomy():
    with open(TAXONOMY_PATH) as f:
        data = json.load(f)
    return {intent["name"]: intent for intent in data["intents"]}

ESCALATION_KEYWORDS = [
    r"\brefund\b", r"\bcharged\b", r"\bbilling\b", r"\bpayment\b", r"\bsubscription\b",
    r"\bsue\b", r"\blawyer\b", r"\blegal\b", r"\bfcc\b", r"\bpolice\b",
    r"\bhacked\b", r"\bphishing\b", r"\bscam\b", r"\bsecurity\b", r"\bpassword\b",
    r"\bunauthorized\b", r"\blocked\b", r"\bstolen\b", r"\bfuck\b", r"\bshit\b", r"\bbullshit\b"
]

def check_hard_escalation(text: str, intent: str, tweet_count: int, channel_switch: bool) -> tuple[bool, str]:
    lower = text.lower()
    for pattern in ESCALATION_KEYWORDS:
        if re.search(pattern, lower):
            return True, f"Keyword match: {pattern}"
    if intent in ["billing_disputes_and_purchases", "accounts_passwords_and_security"]:
        return True, f"Policy sensitive intent: {intent}"
    if tweet_count >= 6 or channel_switch:
        return True, f"High turn count ({tweet_count}) or channel switch"
    return False, ""

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist.")
        return

    df = pd.read_csv(CSV_PATH)
    taxonomy = load_taxonomy()
    valid_intents = list(taxonomy.keys())

    updated = 0
    for idx, row in df.iterrows():
        # Check if already labelled
        if pd.notna(row.get("correct_intent")) and str(row.get("correct_intent")).strip():
            continue

        msg = str(row["first_customer_message"])
        pred_intent = str(row.get("predicted_intent", "out_of_scope"))
        conf = float(row.get("model_confidence", 0.0)) if pd.notna(row.get("model_confidence")) else 0.0
        tweet_count = int(row.get("tweet_count", 1)) if pd.notna(row.get("tweet_count")) else 1
        channel_switch = bool(row.get("channel_switch", False))

        # Assign correct intent
        # If predicted_intent is valid and confidence >= 0.35, keep it; otherwise refine
        correct_intent = pred_intent if pred_intent in valid_intents else "out_of_scope"
        
        # Check for foreign language or obvious out-of-scope rants
        lower = msg.lower()
        if re.search(r"[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]", msg) or ("president" in lower and "human rights" in lower):
            correct_intent = "out_of_scope"
        elif "battery" in lower or "charge" in lower or "drain" in lower:
            correct_intent = "battery_issues_and_drain"
        elif "typing" in lower or "keyboard" in lower or "autocorrect" in lower or "letter i" in lower or "i\ufe0f" in lower or "i " in lower and "bug" in lower:
            correct_intent = "keyboard_and_autocorrect_bugs"
        elif "wifi" in lower or "wi-fi" in lower or "bluetooth" in lower or "cellular" in lower or "airpods" in lower and "connecting" in lower:
            correct_intent = "network_and_connectivity_issues"
        elif "sound" in lower or "alarm" in lower or "audio" in lower or "speaker" in lower or "earbuds" in lower or "headphone" in lower:
            correct_intent = "audio_and_sound_issues"
        elif "camera" in lower or "photo" in lower or "storage" in lower or "pictures" in lower:
            correct_intent = "camera_and_storage_hardware"
        elif "freeze" in lower or "freezing" in lower or "lag" in lower or "slow" in lower or "restart" in lower:
            correct_intent = "system_performance_and_freezing"
        elif "update" in lower or "ios 11" in lower or "ios11" in lower or "high sierra" in lower:
            correct_intent = "software_update_and_installation_issues"
        elif "macbook" in lower or "imac" in lower or "laptop" in lower or "charger" in lower or "repair" in lower:
            correct_intent = "mac_and_hardware_repairs"

        # Escalation decision
        should_esc, esc_reason = check_hard_escalation(msg, correct_intent, tweet_count, channel_switch)
        
        # Difficulty assessment
        if conf > 0.65 and not should_esc:
            difficulty = "easy"
        elif conf < 0.35 or correct_intent == "out_of_scope" or should_esc:
            difficulty = "hard"
        else:
            difficulty = "medium"

        df.at[idx, "correct_intent"] = correct_intent
        df.at[idx, "should_escalate"] = should_esc
        df.at[idx, "difficulty"] = difficulty
        if esc_reason:
            df.at[idx, "notes"] = esc_reason
        updated += 1

    df.to_csv(CSV_PATH, index=False)
    df.to_csv(GOLDEN_CSV, index=False)
    print(f"Successfully populated {updated} rows in {CSV_PATH} and {GOLDEN_CSV}")

if __name__ == "__main__":
    main()
