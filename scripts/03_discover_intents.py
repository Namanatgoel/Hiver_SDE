# DESIGN RATIONALE
# [@A1_rodriguez2024intentgpt] Training-free intent discovery via KMeans k=30 + in-context
#   prompt merge — avoids fine-tuning on 8GB VRAM; evaluated on CLINC and BANKING datasets.
# [@A2_parikh2023zerofewshot] Retrieve top-5 intent exemplars by similarity to shrink the
#   merge prompt; PEFT beats zero-shot with even 1 example/intent.
# [@A6_larson2019clinc150] Canonical argument for explicit out_of_scope intent — without it
#   the escalation layer has no home in the intent taxonomy.
# [@G9_chen2021abcd]        ABCD 55-intent cross-check to validate taxonomy coverage.
# [@A4_casanueva2020banking77] Banking77 77-intent cross-check for financial support domains.

"""
scripts/03_discover_intents.py — embedding-based KMeans clustering + LLM merge into taxonomy.

Outputs: intent_taxonomy.json
Device:  RTX 5050 (embeddings) + ~8 API calls (LLM merge)
Phase:   3
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

CONFIG_PATH = Path("brand_config.json")
THREADS_PATH = Path("data/processed/threads.jsonl")
OUT_TAXONOMY = Path("intent_taxonomy.json")

SEED = 42
N_CLUSTERS = 30
N_EXEMPLARS = 10
TARGET_INTENTS = (10, 14)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_threads() -> list[dict]:
    threads = []
    with open(THREADS_PATH) as f:
        for line in f:
            t = json.loads(line)
            if t.get("first_customer_message", "").strip():
                threads.append(t)
    return threads


def embed_messages(messages: list[str], device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"  Loading embedding model {EMBEDDING_MODEL!r} on {device} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embeddings = model.encode(
        messages,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
    print(f"  Embeddings shape: {embeddings.shape} | peak VRAM: {vram:.0f} MB")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeddings, vram


def cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    print(f"\n  KMeans k={k}, seed={SEED} ...")
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


def get_exemplars(messages: list[str], labels: np.ndarray, k: int, n: int) -> dict[int, list[str]]:
    from collections import defaultdict
    groups: dict[int, list[str]] = defaultdict(list)
    for msg, lbl in zip(messages, labels):
        groups[int(lbl)].append(msg)
    exemplars = {}
    for cluster_id, msgs in groups.items():
        sampled = random.Random(SEED).sample(msgs, min(n, len(msgs)))
        exemplars[cluster_id] = sampled
    return exemplars


def merge_clusters_with_llm(exemplars: dict[int, list[str]], brand: str) -> list[dict]:
    """One LLM call to merge 30 clusters into 10-14 customer-perspective intents + out_of_scope."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.llm.gateway import complete

    cluster_text = ""
    for cid, exs in sorted(exemplars.items()):
        cluster_text += f"\nCluster {cid}:\n"
        for ex in exs[:5]:  # send 5 exemplars per cluster to fit context
            cluster_text += f"  - {ex[:120]}\n"

    prompt = f"""You are a customer support taxonomy expert for brand "{brand}".

Below are {len(exemplars)} message clusters (k-means on embeddings) from real customers.
Each cluster has 5 example messages.

YOUR TASK:
1. Merge these clusters into exactly 10-14 distinct customer-perspective intents.
2. Add one mandatory "out_of_scope" intent for spam, gibberish, competitor abuse, or truly off-topic messages.
3. For each intent, provide:
   - "name": short snake_case label (e.g. "billing_dispute")
   - "description": one sentence from the customer's perspective
   - "source_clusters": list of cluster IDs you merged into this intent

Return ONLY a JSON array. No markdown, no explanation.

Format:
[
  {{"name": "...", "description": "...", "source_clusters": [0, 3, 7]}},
  ...
  {{"name": "out_of_scope", "description": "Messages that are spam, abusive, off-topic, or not addressed to this brand.", "source_clusters": [...]}}
]

Clusters:
{cluster_text}
"""

    response, provider = complete(
        [{"role": "user", "content": prompt}],
        purpose="intent_taxonomy_merge",
        params={"temperature": 0.0},
    )
    print(f"  LLM response from {provider!r} ({len(response)} chars)")

    # Parse JSON
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    intents = json.loads(text)
    return intents


def check_coverage(messages: list[str], intents: list[dict], embeddings: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> dict:
    """Compute OOS share and verify ≥90% map to non-OOS intents."""
    cluster_to_intent: dict[int, str] = {}
    for intent in intents:
        for cid in intent.get("source_clusters", []):
            cluster_to_intent[cid] = intent["name"]

    mapped = sum(1 for lbl in labels if cluster_to_intent.get(int(lbl), "out_of_scope") != "out_of_scope")
    oos = len(labels) - mapped
    coverage = mapped / len(labels)
    return {"total": len(labels), "mapped_non_oos": mapped, "oos": oos, "coverage": coverage}


def main() -> None:
    import time
    import torch

    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 3: Intent Taxonomy Discovery")
    print("  Plan: §Phase 3 | .bib: @A1 @A2 @A6 @G9 @A4")
    print("  Device: RTX 5050 (embeddings) + ~8 LLM API calls")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"  GPU capability: {cap}")

    random.seed(SEED)
    np.random.seed(SEED)

    config = json.loads(CONFIG_PATH.read_text())
    brand = config["brand_handle"]
    cutoff = config["time_split_cutoff"]

    print(f"\nLoading threads from {THREADS_PATH} ...")
    threads = load_threads()

    # Filter to train window only
    from datetime import timezone
    import pandas as pd
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train_threads = [
        t for t in threads
        if pd.to_datetime(
            t.get("first_customer_created_at", ""),
            format="ISO8601",
            utc=True,
            errors="coerce",
        ) < cutoff_ts
    ]
    print(f"  Total threads: {len(threads):,} | Train threads: {len(train_threads):,}")

    messages = [t["first_customer_message"] for t in train_threads]
    print(f"  Embedding {len(messages):,} first customer messages ...")

    embeddings, vram_peak = embed_messages(messages, device)

    print(f"\nClustering k={N_CLUSTERS} ...")
    labels, centers = cluster_kmeans(embeddings, N_CLUSTERS)

    exemplars = get_exemplars(messages, labels, N_CLUSTERS, N_EXEMPLARS)

    # Print cluster sizes
    from collections import Counter
    size_counts = Counter(int(l) for l in labels)
    print("\nCluster sizes:")
    for cid in sorted(size_counts.keys()):
        pct = size_counts[cid] / len(labels) * 100
        print(f"  Cluster {cid:2d}: {size_counts[cid]:5d} ({pct:.1f}%)")

    print(f"\nMerging {N_CLUSTERS} clusters into intents via LLM (~1 API call) ...")
    intents = merge_clusters_with_llm(exemplars, brand)
    print(f"  Got {len(intents)} intents from LLM")

    # Validate count
    n = len(intents)
    oos_present = any(i["name"] == "out_of_scope" for i in intents)
    if not oos_present:
        print("  WARNING: no out_of_scope intent — appending one")
        all_covered = set(cid for i in intents for cid in i.get("source_clusters", []))
        uncovered = [cid for cid in range(N_CLUSTERS) if cid not in all_covered]
        intents.append({"name": "out_of_scope", "description": "Messages that are spam, abusive, off-topic, or not addressed to this brand.", "source_clusters": uncovered})

    if not (TARGET_INTENTS[0] <= len(intents) <= TARGET_INTENTS[1]):
        print(f"  INFO: got {len(intents)} intents (target: {TARGET_INTENTS[0]}-{TARGET_INTENTS[1]})")

    # Check coverage
    cov = check_coverage(messages, intents, embeddings, centers, labels)
    print(f"\nCoverage: {cov['mapped_non_oos']:,}/{cov['total']:,} = {cov['coverage']:.1%} non-OOS")

    if cov["coverage"] < 0.90:
        print(f"  WARNING: coverage {cov['coverage']:.1%} < 90% target")

    # Attach exemplars to intents for the taxonomy file
    cluster_to_intent_idx = {}
    for idx, intent in enumerate(intents):
        for cid in intent.get("source_clusters", []):
            cluster_to_intent_idx[cid] = idx
    for idx, intent in enumerate(intents):
        ex_list = []
        for cid in intent.get("source_clusters", []):
            ex_list.extend(exemplars.get(cid, []))
        intent["exemplars"] = ex_list[:N_EXEMPLARS]

    # Check each intent has >= 5 exemplars
    for intent in intents:
        if len(intent.get("exemplars", [])) < 5:
            print(f"  WARNING: intent {intent['name']!r} has < 5 exemplars")

    # Save taxonomy
    taxonomy = {
        "brand": brand,
        "version": "v1",
        "seed": SEED,
        "embedding_model": EMBEDDING_MODEL,
        "k_clusters": N_CLUSTERS,
        "n_intents": len(intents),
        "coverage": cov,
        "intents": intents,
    }
    OUT_TAXONOMY.write_text(json.dumps(taxonomy, indent=2))
    print(f"\nWrote {OUT_TAXONOMY}")

    # Check stop triggers
    for intent in intents:
        n_msgs = sum(size_counts.get(cid, 0) for cid in intent.get("source_clusters", []))
        pct = n_msgs / len(labels)
        if intent["name"] != "out_of_scope":
            if pct > 0.35:
                print(f"\n⚠  STOP: intent {intent['name']!r} covers {pct:.1%} > 35% — review needed")
                sys.exit(1)
            if pct < 0.02:
                print(f"\n⚠  STOP: intent {intent['name']!r} covers {pct:.1%} < 2% — review needed")
                sys.exit(1)

    # Update RUNLOG
    elapsed_s = time.time() - t0
    runlog = Path("reports/RUNLOG.md")
    if runlog.exists():
        with open(runlog, "a") as f:
            f.write(f"| 3 | 03_discover_intents.py | {device} | {vram_peak:.0f} | {elapsed_s:.1f} | 1 |\n")

    # Phase report
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print(f"  ran:      scripts/03_discover_intents.py")
    print(f"  numbers:  intents={len(intents)} | coverage={cov['coverage']:.1%} | OOS={cov['oos']:,}")
    print(f"  VRAM:     {vram_peak:.0f} MB | provider calls: ~1")
    print(f"  .bib:     @A1_rodriguez2024intentgpt @A2_parikh2023zerofewshot @A6_larson2019clinc150")
    all_have_exemplars = all(len(i.get("exemplars", [])) >= 5 for i in intents)
    accept = cov["coverage"] >= 0.90 and all_have_exemplars
    print(f"  status:   {'ACCEPT' if accept else 'REJECT'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
