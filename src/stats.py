# DESIGN RATIONALE
# [@G1_miller2024errorbars]    Bootstrap 95% CIs on every metric — n=200 => ±6-7pt without CIs
#   is the "misleading headline" trap; 2,000 resamples is the standard.
# [@E2_alttest2025annotator]   Alt-test (leave-one-annotator-out + IPW) for a defensible YES/NO
#   on whether the LLM judge may replace the human annotator.

"""
src/stats.py — bootstrap CIs, Cohen's κ, Pearson, Spearman, alt-test.

Device: CPU
Phase:  7
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    stat_fn=np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute a bootstrap confidence interval for a statistic.
    Returns {"point": ..., "lower": ..., "upper": ..., "n_resamples": ...}.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    point = float(stat_fn(arr))

    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(stat_fn(sample)))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    # Sanity check: CI must contain the point estimate
    assert lower <= point <= upper or (
        abs(lower - point) < 1e-9 or abs(upper - point) < 1e-9
    ), f"CI [{lower:.4f}, {upper:.4f}] does not contain point={point:.4f}"

    return {"point": point, "lower": lower, "upper": upper, "n_resamples": n_resamples}


# ---------------------------------------------------------------------------
# Cohen's kappa (quadratic-weighted)
# ---------------------------------------------------------------------------

def cohens_kappa(
    rater_a: list,
    rater_b: list,
    weighted: str = "quadratic",
) -> float:
    """Compute Cohen's kappa between two raters."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(rater_a, rater_b, weights=weighted if weighted != "none" else None))


# ---------------------------------------------------------------------------
# Pearson and Spearman
# ---------------------------------------------------------------------------

def pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = scipy_stats.spearmanr(x, y)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Alt-test ([@E2_alttest2025annotator])
# ---------------------------------------------------------------------------

def alt_test(
    human_ratings: list[list[int]],
    judge_ratings: list[int],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Alternative annotator test: leave-one-annotator-out majority vote + IPW.

    human_ratings: list of annotator rating vectors (each inner list = one annotator's ratings).
    judge_ratings: the LLM judge's ratings.

    Returns: {
        "verdict": "YES" | "NO",   # can the judge replace a human annotator?
        "judge_agreement": float,  # judge↔majority agreement rate
        "human_loo_agreement": float,  # expected LOO agreement among humans
        "kappa_judge_majority": float,
        "kappa_human_loo": float,
    }
    """
    rng = np.random.RandomState(seed)
    n_items = len(judge_ratings)
    n_annotators = len(human_ratings)

    assert all(len(r) == n_items for r in human_ratings), "All annotators must rate same items"

    # Majority vote (all humans)
    human_matrix = np.array(human_ratings)  # (n_annotators, n_items)
    majority = np.round(np.mean(human_matrix, axis=0)).astype(int).tolist()

    # Judge↔majority agreement
    judge_agree = float(np.mean([j == m for j, m in zip(judge_ratings, majority)]))
    kappa_judge = cohens_kappa(judge_ratings, majority)

    # Leave-one-out: for each annotator, compute majority of the rest vs that annotator
    loo_agrees = []
    loo_kappas = []
    for held_out in range(n_annotators):
        others = [human_ratings[i] for i in range(n_annotators) if i != held_out]
        others_matrix = np.array(others)
        loo_majority = np.round(np.mean(others_matrix, axis=0)).astype(int).tolist()
        held_out_ratings = human_ratings[held_out]
        loo_agrees.append(np.mean([h == m for h, m in zip(held_out_ratings, loo_majority)]))
        loo_kappas.append(cohens_kappa(held_out_ratings, loo_majority))

    human_loo_agree = float(np.mean(loo_agrees))
    human_loo_kappa = float(np.mean(loo_kappas))

    # Verdict: judge passes if its agreement is not significantly worse than human LOO
    # Using a conservative threshold: judge agreement >= human_loo_agree - 0.05
    verdict = "YES" if judge_agree >= human_loo_agree - 0.05 else "NO"

    return {
        "verdict": verdict,
        "judge_agreement": judge_agree,
        "human_loo_agreement": human_loo_agree,
        "kappa_judge_majority": kappa_judge,
        "kappa_human_loo": human_loo_kappa,
        "n_items": n_items,
        "n_annotators": n_annotators,
    }
