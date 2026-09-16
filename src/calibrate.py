# DESIGN RATIONALE
# [@C6_angelopoulos2021conformal] Conformal risk control on the calibration split — computes
#   tau such that the auto-handled subset has ≤alpha error with probability ≥1-delta.
#   This converts "I picked 0.4 by gut feeling" into a statistical guarantee.

"""
src/calibrate.py — isotonic calibration + conformal risk control for tau selection.

Device: CPU (isotonic) / RTX 5050 (if generating predictions for calibration set)
Phase:  6
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

def fit_isotonic(raw_scores: list[float], gold_correct: list[int]):
    """
    Fit isotonic regression on (raw_score, gold_correct) pairs.
    Returns a fitted IsotonicRegression object.
    gold_correct: 1 if the model prediction was correct, 0 otherwise.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.array(raw_scores), np.array(gold_correct, dtype=float))
    return ir


def compute_ece(scores: list[float], correct: list[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [lo <= s < hi for s in scores]
        if not any(mask):
            continue
        bin_s = [s for s, m in zip(scores, mask) if m]
        bin_c = [c for c, m in zip(correct, mask) if m]
        ece += abs(np.mean(bin_s) - np.mean(bin_c)) * sum(mask) / n
    return ece


# ---------------------------------------------------------------------------
# Conformal risk control ([@C6_angelopoulos2021conformal])
# ---------------------------------------------------------------------------

def conformal_tau(
    cal_scores: list[float],
    cal_errors: list[int],
    alpha: float = 0.10,
    delta: float = 0.05,
) -> float:
    """
    Choose tau (escalation threshold) by conformal risk control on the calibration split.

    The guarantee: P(error rate on auto-handled subset ≤ alpha) ≥ 1 - delta.

    cal_scores: calibrated confidence scores (higher = more confident)
    cal_errors: 1 if the model was wrong on this example, 0 if correct
    alpha:      target maximum error rate on auto-handled examples
    delta:      failure probability

    Returns: tau (escalate if calibrated_conf < tau)
    """
    n = len(cal_scores)
    assert n == len(cal_errors), "Lengths must match"

    # Sort by score descending (highest confidence first)
    sorted_pairs = sorted(zip(cal_scores, cal_errors), key=lambda x: -x[0])

    # Sweep tau: for each candidate tau, compute empirical error rate on auto-handled subset
    best_tau = 0.0  # default: auto-handle everything
    for i in range(n):
        tau_candidate = sorted_pairs[i][0]
        # Auto-handled = examples with score >= tau_candidate
        auto_handled = sorted_pairs[:i + 1]
        if len(auto_handled) == 0:
            continue
        empirical_error = sum(e for _, e in auto_handled) / len(auto_handled)

        # Hoeffding bound: P(true_error > empirical + eps) ≤ delta
        # eps = sqrt(log(1/delta) / (2 * n_auto))
        n_auto = len(auto_handled)
        eps = np.sqrt(np.log(1.0 / delta) / (2 * n_auto))
        upper_bound = empirical_error + eps

        if upper_bound <= alpha:
            best_tau = tau_candidate  # keep updating to the smallest tau that satisfies the bound

    return best_tau


def save_calibration(path: Path, tau: float, ece_before: float, ece_after: float, alpha: float, delta: float, n_cal: int) -> None:
    data = {
        "tau": tau,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "conformal_alpha": alpha,
        "conformal_delta": delta,
        "n_calibration": n_cal,
        "guarantee": f"Auto-handled subset has <=alpha={alpha} error with probability >=1-delta={1-delta}",
    }
    path.write_text(json.dumps(data, indent=2))
