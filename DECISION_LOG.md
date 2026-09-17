# DECISION_LOG.md

Authoritative audit log of architecture choices, trade-offs, and methodology decisions.
Every entry references an evidence key from `references.bib`.

---

## Decision 1: Temporal Split Date Calibration (2017-11-15 vs 2017-10-01)
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 1 (scripts/01_eda.py)
- Bib citation: [@D2_prabhakar2018towards]
- Context: Plan v2 initially specified `TIME_SPLIT = "2017-10-01"` with a gate requirement of >= 20k train messages and >= 3k test messages.
- Problem: Empirical quantile analysis of the full 2,811,774-row corpus revealed that 99% of twcs timestamps fall between 2017-10-03 and 2017-12-03. A cutoff of 2017-10-01 leaves only 177 train messages for AppleSupport and 841 for AmazonHelp (<1% of brand inbound volume), failing the 20k train acceptance gate.
- Decision: Shift the temporal cutoff to `2017-11-15`.
- Rationale: [@D2_prabhakar2018towards] establishes the principle of temporal train/test splitting (older=train, newest=test) to prevent lookahead policy drift leakage. Cutoff `2017-11-15` establishes an authentic ~75/25 split for AppleSupport: 80,506 train inbound messages (75.5%) and 26,117 test inbound messages (24.5%), satisfying both gate thresholds.

---

## Decision 2: Embedding kNN Classification Over Full Fine-Tuning
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 3 & 4 (src/router.py, scripts/03_discover_intents.py)
- Bib citation: [@A2_parikh2023zerofewshot]
- Context: Need to classify incoming customer messages into brand-specific intents with low latency on local hardware.
- Problem: Fine-tuning a sequence classifier requires extensive annotated training data, risks catastrophic forgetting across edge cases, and consumes substantial VRAM on an 8 GB consumer GPU.
- Decision: Use sentence embeddings (`all-MiniLM-L6-v2`) with k-nearest-neighbors (kNN) to intent centroids rather than fine-tuning end-to-end transformers.
- Rationale: [@A2_parikh2023zerofewshot] demonstrates that dense representations combined with calibrated nearest-centroid distances deliver strong zero/few-shot accuracy while retaining strict interpretability and instant inference speed.

---

## Decision 3: Hybrid Retrieval (BM25 + Dense + RRF) Over Dense-Only
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 6 (src/retriever.py)
- Bib citation: [@B2_wu2026evidence]
- Context: Historical customer support tweets contain specific error codes, app names, device models, and technical jargon alongside colloquial phrasing.
- Problem: Dense embeddings frequently miss exact keyword matches (such as specific iOS version strings or error codes), while BM25 fails on semantic paraphrase.
- Decision: Implement hybrid retrieval combining BM25 keyword search and dense semantic search via Reciprocal Rank Fusion (RRF).
- Rationale: Real-world enterprise deployment evidence in [@B2_wu2026evidence] proves that hybrid fusion consistently outperforms single-modality retrievers on noisy customer dialogue.

---

## Decision 4: Cross-Encoder Reranking for Evidence Selection
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 6 (src/retriever.py)
- Bib citation: [@B1_xu2024ragkg]
- Context: First-stage retrieval returns top-20 candidates that may share lexical overlap without answering the customer's specific technical problem.
- Problem: Bi-encoder representations compute embeddings independently, missing subtle query-document token interactions necessary to assess true historical relevance.
- Decision: Apply a cross-encoder (`BAAI/bge-reranker-base`) to rerank the top-20 retrieved candidates down to top-3 evidence threads.
- Rationale: As shown in [@B1_xu2024ragkg], cross-encoders capture cross-attention between customer inquiry and resolution text, substantially improving precision@3 and filtering distractor threads.

---

## Decision 5: Conformal Risk Control for Escalation Thresholding
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 6 (src/router.py, src/calibrate.py)
- Bib citation: [@C6_angelopoulos2021conformal]
- Context: Need a principled threshold to decide whether to auto-handle an incoming query or escalate to a human agent.
- Problem: Standard softmax probabilities and raw model confidences are notoriously uncalibrated and overconfident on out-of-distribution inputs. Hand-picking an arbitrary threshold (e.g., 0.5) offers zero statistical performance guarantees.
- Decision: Use conformal prediction and isotonic calibration to determine the escalation threshold tau.
- Rationale: Following [@C6_angelopoulos2021conformal] and [@C1_wen2025abstention], conformal risk control guarantees that the error rate on the auto-handled subset is rigorously bounded by alpha with probability 1-delta.

---

## Decision 6: Explicit Structured Reason Codes Over Binary Escalation Flags
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 6 (src/router.py)
- Bib citation: [@C2_bachar2026lpp]
- Context: System must declare why a conversation is handed over to human support agents.
- Problem: A simple boolean `should_escalate = True` provides zero actionable context to human agents and prevents auditing routing failure modes.
- Decision: Route with explicit taxonomic reason codes (`policy_sensitive`, `low_confidence`, `empty_retrieval`, `repeated_failure`, `conformal_risk_breach`).
- Rationale: [@C2_bachar2026lpp] proves that structured attribution makes automated deferral auditable, allowing support leads to debug queue distribution and detect emerging operational issues.

---

## Decision 7: Reference-Grounded Evaluation Over Blind Generation Judging
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 7 (src/judge.py)
- Bib citation: [@F10_nofreelabels2025]
- Context: Evaluating reply quality using LLM-as-a-judge.
- Problem: Reference-free LLM judges rely purely on internal model priors, penalizing valid brand policies or rewarding plausible-sounding hallucinations.
- Decision: Supply the brand's verified historical resolution tweet as reference ground truth to the evaluation judge.
- Rationale: [@F10_nofreelabels2025] demonstrates that reference-grounded judges achieve significantly higher alignment with human assessments and eliminate stylistic bias.

---

## Decision 8: Pairwise Evaluation with Position-Swap Debiasing
- Date: 2026-09-15
- Status: APPROVED
- Scope: Phase 7 (src/judge.py)
- Bib citation: [@F6_wang2023unfair]
- Context: Comparing generated agent replies against baseline model outputs.
- Problem: LLM judges suffer from position bias, systematically favoring Candidate A or Candidate B regardless of content quality.
- Decision: Execute all pairwise comparisons in both orders (A vs B and B vs A) and average the resulting scores.
- Rationale: [@F6_wang2023unfair] establishes position swapping as an essential debiasing protocol to prevent order artifact distortions in comparative evaluation.

---

## Decision 9: Multi-Model Jury Consensus for Automated Assessment
- Date: 2026-09-16
- Status: APPROVED
- Scope: Phase 7 (src/judge.py)
- Bib citation: [@F5_verga2024jury]
- Context: Single-judge evaluation carries individual model blind spots and provider-specific quirks.
- Problem: Relying on a single proprietary model couples evaluation metrics to vendor updates and unobservable prompt biases.
- Decision: Form an evaluation jury using heterogeneous models (Gemini Flash, Groq Llama, and local Prometheus-2) taking the median score.
- Rationale: [@F5_verga2024jury] proves that jury voting across diverse model families reduces variance and correlates more strongly with human consensus than any single judge.

---

## Decision 10: Strict Temporal Splitting Over Random Shuffling
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 1 & 2 (scripts/01_eda.py, scripts/02_build_threads.py)
- Bib citation: [@D2_prabhakar2018towards]
- Context: Partitioning the dataset into training and evaluation splits.
- Problem: Random splitting in time-series customer support leaks future context into the past, training models on resolution patterns and software bugs that have not yet occurred.
- Decision: Enforce a strict chronological cutoff date: all threads initiated prior to 2017-11-15 form the historical index, and subsequent threads form the evaluation set.
- Rationale: [@D2_prabhakar2018towards] shows that random cross-validation yields inflated, unrealistic performance estimates in dialogue systems due to temporal policy and event leakage.

---

## Decision 11: Filtering DM-Redirect Boilerplate from Resolved Knowledge Base
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 2 (scripts/02_build_threads.py)
- Bib citation: [@D2_prabhakar2018towards]
- Context: Customer support tweets frequently end with canned redirect responses such as "Please send us a DM with your account details".
- Problem: Treating DM redirects as successful resolutions poisons retrieval indexes with uninformative boilerplate that teaches the agent to defer without solving the problem.
- Decision: Flag and downweight or drop threads whose only resolution step is a generic direct message invitation.
- Rationale: As documented in [@D2_prabhakar2018towards], excluding ungrounded channel switches prevents resolution rate inflation and forces the retrieval index to index substantive technical guidance.

---

## Decision 12: Explicit Out-of-Scope Intent Modeling
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 3 (intent_taxonomy.json, scripts/03_discover_intents.py)
- Bib citation: [@A6_larson2019clinc150]
- Context: Real customer service streams contain spam, political rants, irrelevant banter, and questions outside brand support purview.
- Problem: Closed-world intent taxonomies force out-of-scope messages into the nearest arbitrary category, leading to nonsensical automated replies.
- Decision: Include `out_of_scope` as a first-class intent in the taxonomy with high escalation priority.
- Rationale: [@A6_larson2019clinc150] demonstrates that explicit rejection/out-of-scope modeling is essential for safe real-world NLU and prevents boundary distortion among valid domain intents.

---

## Decision 13: Deterministic Response Caching with Provider Chain
- Date: 2026-09-14
- Status: APPROVED
- Scope: Phase 0 (src/llm/gateway.py)
- Bib citation: [@C12_chen2023frugalgpt]
- Context: Running hundreds of pipeline evaluations against free-tier API quotas.
- Problem: Flaky connections, 429 rate limits, and quota exhaustion can halt automated benchmarks mid-run and make results non-reproducible.
- Decision: Route all calls through a local SHA-256 keyed JSONL cache backed by a sequential fallback provider chain (Gemini -> Groq -> Cerebras -> local model).
- Rationale: [@C12_chen2023frugalgpt] shows that tiered cascading and aggressive caching cut API expenditures while ensuring that a warm re-run costs exactly zero network calls and executes in minutes.

---

## Decision 14: Non-Parametric Bootstrap Confidence Intervals on All Metrics
- Date: 2026-09-16
- Status: APPROVED
- Scope: Phase 7 (src/stats.py, scripts/06_evaluate.py)
- Bib citation: [@G1_miller2024errorbars]
- Context: Reporting final headline numbers (intent accuracy, escalation F1, judge scores) on the golden evaluation set.
- Problem: Single-point evaluation metrics give a false illusion of certainty and conceal variance due to small sample sizes.
- Decision: Compute 95% empirical bootstrap confidence intervals across 2,000 resamples for every reported metric.
- Rationale: [@G1_miller2024errorbars] establishes that publishing point estimates without uncertainty quantification is unscientific and masks statistical indistinguishability between competing systems.

---

## Decision 15: Systematic Taxonomic Failure Analysis Over Anecdotal Inspection
- Date: 2026-09-16
- Status: APPROVED
- Scope: Phase 8 (reports/eval_results.json, REPORT.md)
- Bib citation: [@G2_cemri2025mast]
- Context: Analyzing why and where the AI agent fails on customer support queries.
- Problem: Cherry-picking a few interesting failed examples creates confirmation bias and fails to identify structural bottlenecks in the pipeline.
- Decision: Apply the Multi-Agent System Troubleshooting (MAST) taxonomy to categorize failure traces into distinct pipeline stages (retrieval recall, rerank selection, generation hallucination, routing over-conservatism).
- Rationale: [@G2_cemri2025mast] demonstrates that taxonomy-driven root-cause failure analysis exposes systematic pipeline vulnerabilities and guides high-leverage architectural improvements.
