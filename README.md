# AI Customer Support Agent for AppleSupport

An evidence-grounded, audit-ready AI customer support agent developed for the Hiver SDE Intern Assignment.
The system classifies incoming customer inquiries, retrieves verified historical resolution threads via hybrid BM25/dense fusion with cross-encoder reranking, drafts evidence-grounded replies with citations, and implements a 3-layer calibrated routing engine to decide whether to auto-handle or escalate with structured attribution.

---

## Deliverables Checklist

| Deliverable | Location | Description |
|---|---|---|
| **1. Runnable Pipeline** | `Makefile`, `scripts/`, `src/` | End-to-end runnable pipeline reproducible in under 15 minutes |
| **2. Golden Evaluation Set** | [`data/golden/golden_set.csv`](data/golden/golden_set.csv) | 200 stratified customer inquiries with verified intents and escalation labels |
| **3. Evaluation Harness** | [`scripts/06_evaluate.py`](scripts/06_evaluate.py) | Automated metrics + LLM-as-a-judge rubric with 95% bootstrap CIs |
| **4. Comprehensive Report** | [`REPORT.md`](REPORT.md) | Problem framing, baseline results, top-5 failure modes, headline critique, and next steps |
| **5. Decision Log** | [`DECISION_LOG.md`](DECISION_LOG.md) | 15 non-obvious architecture and methodology trade-offs backed by literature |

---

## Headline Results

All metrics evaluated on the held-out 200-example golden set.
Point estimates are reported alongside empirical 95% bootstrap confidence intervals (2,000 resamples, seed=42).

| System | Intent Accuracy (95% CI) | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 | Over-Esc Rate |
|---|---|---|---|---|---|---|
| Trivial (random + always escalate) | ~10% | — | — | 100% | ~0.47 | 100% |
| Simple (majority intent + regex) | 22.5% [17.0%, 28.5%] | — | — | — | 0.164 | ~14% |
| **Proposed Agent** (tau=0.35) | **72.5%** [66.5%, 78.5%] | **0.779** | 0.351 | 0.656 | **0.457** | 53.2% |

LLM jury judge evaluation (Gemini + Groq + local Prometheus-2): run `make eval-judge` with valid API keys.
*Full analysis in [REPORT.md](REPORT.md), including an honest breakdown of what these numbers do and do not prove.*

---

## Reproduce in Under 15 Minutes

### Step 1: Environment Setup
Ensure you have Conda installed. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate hiver_sde
```

### Step 2: Configure Environment Keys
Copy the example environment file:
```bash
cp .env.example .env
```
Add your API keys to `.env` (Gemini, Groq, or Cerebras). If running offline, the local cache and rules execute deterministically without making live provider calls.

### Step 3: Run Test Suite
Validate cache determinism, budget accounting, and routing rails:
```bash
make test
```

### Step 4: Run Evaluation Harness
Run the full evaluation suite across the golden evaluation set:
```bash
make eval
```
This executes intent scoring, escalation auditing, baseline comparisons, judge jury scoring with position-swap debiasing, and generates `reports/eval_results.json`.

### Step 5: Regenerate Report
Compile and regenerate the complete markdown report directly from evaluation artifacts:
```bash
make report
```

### Step 6: Interactive Live Demo
Execute the agent on representative customer inquiries to inspect routing and response drafting:
```bash
make run
```

---

## System Architecture

```
Incoming Customer Message
           |
           v
+-----------------------------------------------------------+
| Layer 1: Guardrail Rails (Regex & Policy Checks)          |
| -> Billing disputes, legal threats, safety, account PII   |
| -> Immediate escalation with fixed reason codes           |
+-----------------------------------------------------------+
           | (passes hard rails)
           v
+-----------------------------------------------------------+
| Layer 2: Intent Discovery & Centroid Classification      |
| -> Dense semantic representations (all-MiniLM-L6-v2)      |
| -> Calibrated confidence & out-of-scope detection         |
+-----------------------------------------------------------+
           |
           v
+-----------------------------------------------------------+
| Layer 3: Hybrid Retrieval & Cross-Encoder Reranking       |
| -> BM25 token search + dense vector retrieval             |
| -> Reciprocal Rank Fusion (RRF) -> Top-20 candidates      |
| -> Cross-Encoder rerank (BAAI/bge-reranker-base) -> Top-3|
+-----------------------------------------------------------+
           |
           v
+-----------------------------------------------------------+
| Layer 4: Calibrated Conformal Routing Engine              |
| -> Conformal risk thresholding on joint confidence        |
| -> Escalate if risk exceeds tau with structured reason    |
+-----------------------------------------------------------+
           | (auto-handle approved)
           v
+-----------------------------------------------------------+
| Layer 5: Evidence-Grounded Reply Drafting                 |
| -> Strictly grounded prompt with required citations       |
| -> Verifiable reference to historical brand resolutions   |
+-----------------------------------------------------------+
```

---

## Repository Structure

```
├── brand_config.json               # Brand metadata and temporal partition config
├── intent_taxonomy.json            # 13 verified intent categories and descriptions
├── environment.yml                 # Pinned conda runtime dependencies
├── Makefile                        # Command targets: test, eval, report, run
├── references.bib                  # Academic literature and benchmark citations
├── DECISION_LOG.md                 # 15 non-obvious engineering decisions
├── REPORT.md                       # Comprehensive evaluation and failure analysis
├── data/
│   └── golden/                     # Hand-labelled evaluation set (Deliverable #2)
│       ├── annotation_guide.md     # Rubric for intents, escalation, and difficulty
│       ├── golden_set.csv          # 200 annotated evaluation records
│       └── labelstudio_config.xml  # Label Studio UI configuration
├── src/
│   ├── agent.py                    # Evidence-grounded generation and interactive demo
│   ├── retriever.py                # Hybrid BM25/Dense retrieval with cross-encoder
│   ├── router.py                   # 3-layer guardrail and conformal routing engine
│   ├── calibrate.py                # Isotonic calibration and conformal bounds
│   ├── judge.py                    # Reference-grounded evaluation judge
│   ├── stats.py                    # Bootstrap CIs, Cohen's kappa, correlations
│   └── llm/
│       └── gateway.py              # Tiered provider chain with SHA-256 caching
├── scripts/
│   ├── 00_download_data.py         # Data setup and raw tweet symlinking
│   ├── 01_eda.py                   # Corpus exploration and temporal split cutoff
│   ├── 02_build_threads.py         # Thread reconstruction and DM filtering
│   ├── 03_discover_intents.py      # HDBSCAN/KMeans intent cluster discovery
│   ├── 04_label_intents.py         # Pseudo-label generation with confidence margins
│   ├── 05_build_golden_set.py      # Stratified sampling for evaluation set
│   ├── 06_evaluate.py              # Full evaluation harness runner
│   └── 09_build_report.py          # Dynamic report generation from eval artifacts
└── tests/
    └── test_gateway.py             # Provider cache and budget unit tests
```

---

## Citations and Academic Foundation

All architectural choices and evaluation methodologies are grounded in academic literature cataloged in [`references.bib`](references.bib).
Key foundations include:
- Temporal splitting avoiding lookahead leakage: Prabhakar et al. (2018) `[@D2_prabhakar2018towards]`
- Hybrid retrieval with cross-encoder reranking: Wu et al. (2026) `[@B2_wu2026evidence]`, Xu et al. (2024) `[@B1_xu2024ragkg]`
- Conformal risk control for abstention: Angelopoulos et al. (2021) `[@C6_angelopoulos2021conformal]`, Wen et al. (2025) `[@C1_wen2025abstention]`
- Non-parametric bootstrap error bars: Miller et al. (2024) `[@G1_miller2024errorbars]`
- LLM Judge debiasing and position-swap: Wang et al. (2023) `[@F6_wang2023unfair]`, Verga et al. (2024) `[@F5_verga2024jury]`
