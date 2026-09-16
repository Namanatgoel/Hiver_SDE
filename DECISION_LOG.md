# DECISION_LOG.md

Authoritative audit log of design choices and plan deviations.
Every entry references a key from `references.bib`.
If no entry exists, it is explicitly marked as `no supporting entry in references.bib`.

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
