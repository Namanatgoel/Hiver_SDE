# AI Customer Support Agent: AppleSupport Evaluation Report

## 1. Problem Framing: What "Good" Means for AppleSupport

Customer support for Apple on Twitter (@AppleSupport) represents a high-volume, high-stakes technical support environment.
Customers frequently reach out with terse messages during software rollouts (such as iOS 11 updates) and hardware launches.

### Defining "Good" for AppleSupport:
1. **Factual Groundedness Over Generative Fluency**: The system must never hallucinate return policies, replacement terms, or diagnostic steps. Every recommendation must directly cite documented historical resolutions.
2. **Conservative Escalation on Ambiguity**: A false auto-handle on account security, billing disputes, or severe system crashes degrades user trust and can trigger regulatory liability. Escalating safely with a structured reason code is superior to a confident wrong answer.
3. **Low-Latency Hybrid Retrieval**: Customer inquiries often hinge on exact software builds ("iOS 11.0.3") or specific hardware terms alongside colloquial phrasing. The system must index both exact lexical tokens and semantic intent.

### What We Chose NOT to Build:
- **Unconstrained Free Generation**: We explicitly avoided ungrounded generation models that draft solutions without historical retrieval evidence.
- **Single-Turn Blind Automation**: We refused to automate multi-turn threads involving private customer information (credentials, serial numbers). Those are routed directly to human agents.
- **Monolithic End-to-End Fine-Tuning**: Rather than black-box fine-tuning an LLM on noisy Twitter text, we decoupled classification, retrieval, and generation for strict auditability.

---

## 2. Results vs. Baselines

All metrics are evaluated on the held-out golden evaluation set (n = 200).
Every metric reports an empirical 95% non-parametric bootstrap confidence interval across 2,000 resamples.

| System / Model | Intent Accuracy (95% CI) | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 | Over-Escalation Rate |
|---|---|---|---|---|---|---|
| **Trivial Baseline** (Random Intent + Always Escalate) | 11.0% [7.0%, 15.5%] | — | — | 100.0% | 0.467 | 100.0% |
| **Simple Baseline** (Majority Intent + Regex Escalation) | 22.5% [17.0%, 28.5%] | — | — | — | 0.164 | — |
| **Proposed Agent** (Embedding kNN + Hybrid RAG + Guardrails, tau=0.35) | **72.5%** [66.5%, 78.5%] | **0.779** | 0.338 | 0.803 | **0.476** | 69.1% |

### LLM-as-a-Judge Evaluation:
- **Status: UNVERIFIED** — Run with --judge flag to execute LLM jury evaluation (requires API keys, ~150 calls).
- The judge harness is implemented in `src/judge.py` and uses a jury of Gemini + Groq + local Prometheus-2. Run `make eval` with valid API keys to populate this section.

---

## 3. Failure Analysis: Top 5 Failure Modes

Through systematic inspection of misclassified and misrouted traces, we identified the following top 5 failure modes:

1. **Foreign Language Customer Inquiries (Prevalence: ~12%)**:
   - *Example*: "Тук някой друг нещастен мак юзър, ъпдейтнал до High Sierra и изпитващ затруднения с InDesign..."
   - *Stage*: Intent Classification / Routing.
   - *Hypothesis*: The English sentence embedding model (`all-MiniLM-L6-v2`) mapped Cyrillic and non-English scripts into erratic centroid clusters rather than identifying them as out-of-scope for the English support queue.

2. **Compound Multi-Issue Inquiries (Prevalence: ~9%)**:
   - *Example*: "Phone battery drains in 10 minutes and the screen freezes when opening Camera after iOS 11 update."
   - *Stage*: Single-label Intent Boundary.
   - *Hypothesis*: The customer reports three distinct defects simultaneously (battery, system performance, camera). Single-label classification is forced to pick one, causing technical misclassification.

3. **Over-Conservative Escalation on Benign Slang (Prevalence: ~15%)**:
   - *Example*: "This autocorrect glitch is killing me, fix it."
   - *Stage*: Layer 1 Hard Rail Routing.
   - *Hypothesis*: Hard rail regex caught the word "killing" under safety/self-harm filters, unnecessarily triggering a human escalation for a trivial keyboard bug.

4. **Empty or Distractor Evidence Retrieval (Prevalence: ~8%)**:
   - *Example*: Obscure hardware issues on legacy Apple accessories.
   - *Stage*: Hybrid BM25/Dense Retrieval.
   - *Hypothesis*: The historical corpus had insufficient high-resolution threads for legacy accessories, leading the cross-encoder to assign low similarity scores and forcing deferral.

5. **Atypical Unicode Punctuation and Bug Artifacts (Prevalence: ~6%)**:
   - *Example*: "I [?] help wtf @AppleSupport" (representing the infamous iOS 11.1 autocorrect bug rendering 'I' as a symbol).
   - *Stage*: Tokenization and Intent Embedding.
   - *Hypothesis*: Unicode replacement glyphs were stripped or distorted during preprocessing, shifting the text away from the `keyboard_and_autocorrect_bugs` cluster.

---

## 4. What Is Misleading About My Headline Number?

Publishing an Intent Accuracy of **72.5%** without qualification would be fundamentally misleading for three reasons:

1. **Sample Size Uncertainty (Bootstrap Margin of Error)**:
   On a golden evaluation set of 200 examples, the 95% confidence interval spans **66.5% to 78.5%** (a wide 12.0 percentage point window).
   Reporting a single point estimate masks the statistical variance inherent in small test sets.

2. **Severe Class Imbalance Distortions**:
   In AppleSupport Twitter data during late 2017, bugs related to iOS 11 software updates and the letter 'I' autocorrect bug account for over 35% of all inbound traffic.
   A classifier that performs well on these dominant clusters achieves an inflated headline accuracy while remaining unreliable on lower-frequency intents like hardware repairs or account recovery.

3. **Over-Escalation Masking Real Automation Efficiency**:
   Our agent achieves high safety and a recall of 80.3% on required escalations, but suffers an unnecessary escalation rate of 69.1%.
   In a production contact center, escalating 69.1% of auto-handlable queries would overwhelm human agent queues, negating the operational cost savings implied by the headline accuracy.

---

## 5. What We Would Do Next With One More Week

If given one additional week to advance the system toward enterprise production, we would execute the following:

1. **Fine-Tune Domain Cross-Encoder Reranker**:
   Train a specialized MiniLM cross-encoder on hard negative support pairs mined from AppleSupport thread trees to improve Precision@3 by 15-20%.
2. **Multi-Label Intent and Entity Extraction**:
   Upgrade the intent classifier from single-label kNN to a multi-label hierarchical tagger capable of parsing compound complaints and tagging specific hardware models and OS versions.
3. **Context-Aware Safety Filtering**:
   Replace rigid regex hard rails with a lightweight token-level semantic guardrail to distinguish literal safety emergencies from conversational hyperbole ("this glitch is killing me").
4. **Active Learning and Disagreement Mining**:
   Deploy an iterative annotation loop prioritizing examples where the hybrid retriever and intent classifier exhibit maximal uncertainty for continuous dataset enrichment.
5. **Interactive Multi-Turn State Tracking**:
   Extend the agent architecture from single-turn retrieval drafting to dialogue state tracking capable of maintaining troubleshooting memory across 3 to 5 turn customer interactions.
