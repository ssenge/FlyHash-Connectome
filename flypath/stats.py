"""Inference for the wiring comparison.

Two questions are kept apart because they need different procedures.

1. Is the measured wiring unusual among degree-preserving random wirings?
   Test statistic: retrieval score s(M) on the benchmark. Under the null
   hypothesis the measured matrix is exchangeable with draws from the uniform
   fixed-margin distribution, so its rank among B null draws is uniform and the
   randomization p-value is exact *conditional on the benchmark*
   (`randomization_test`).

2. How large is the difference, and how uncertain is it across odorant sets?
   Estimand: theta = E_D[ s_D(M_real) - E_null s_D(M) ] relative to
   E_D E_null s_D(M), where D is a benchmark generated from the source-odorant
   population by the stated mixture generator. Interval: a two-stage
   bootstrap that resamples source odorants, regenerates the benchmark,
   recomputes retrieval, and re-estimates the null mean from a fresh subset of
   null matrices (`two_stage_bootstrap`). Its coverage is checked by
   simulation in `experiments.coverage`.

Variation *across individual random networks* (the null's standard deviation)
and uncertainty *in the null mean* (that SD over sqrt(B)) are reported
separately; they answer different questions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def randomization_test(real: float, null: np.ndarray) -> dict:
    """Rank-based p-values with the +1 correction; ties count against the claim."""
    null = np.asarray(null, float)
    b = len(null)
    p_upper = (1 + int((null >= real).sum())) / (b + 1)   # H1: real scores higher
    p_lower = (1 + int((null <= real).sum())) / (b + 1)   # H1: real scores lower
    return {"B": b, "p_upper": p_upper, "p_lower": p_lower,
            "p_two_sided": min(1.0, 2 * min(p_upper, p_lower)),
            "rank_from_top": int((null > real).sum()) + 1}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def shift_power(null: np.ndarray, deltas: list[float], alpha: float = 0.05,
                direction: str = "upper") -> dict:
    """Power of the randomization test against a relative location shift.

    Model: the measured wiring behaves like a random wiring whose score is
    multiplied by (1 + delta). For each held-out null draw b, its score is
    shifted and tested against the remaining B-1 draws; power is the rejection
    rate. This uses the empirical null distribution, not a normal
    approximation, but the shift model itself is an assumption.
    """
    null = np.asarray(null, float)
    b = len(null)
    out = {}
    for d in deltas:
        rej = 0
        for i in range(b):
            rest = np.delete(null, i)
            s = null[i] * (1 + d)
            if direction == "upper":
                p = (1 + int((rest >= s).sum())) / b
            else:
                p = (1 + int((rest <= s).sum())) / b
            rej += p <= alpha
        out[f"{d:.4f}"] = rej / b
    return out


def smallest_detectable(power: dict, target: float = 0.8) -> float | None:
    """Smallest shift in a power table reaching `target` power, if any."""
    hits = [float(k) for k, v in power.items() if v >= target]
    return min(hits, key=abs) if hits else None


def two_stage_bootstrap(replicate: Callable[[np.random.Generator], tuple[float, np.ndarray]],
                        draws: int, seed: int = 0, level: float = 0.90) -> dict:
    """Percentile interval for the relative difference real - mean(null).

    `replicate(rng)` must build a fresh benchmark from resampled source
    odorants and return (real score, scores of a fresh subset of null
    matrices) on it. Returning the null subset lets the interval carry
    finite-null-ensemble uncertainty; drawing fewer nulls per replicate than
    the point estimate uses makes that component conservative.
    """
    rng = np.random.default_rng(seed)
    rel = np.empty(draws)
    absd = np.empty(draws)
    for r in range(draws):
        real, null = replicate(rng)
        base = float(np.mean(null))
        absd[r] = real - base
        rel[r] = (real - base) / base
    a = (1 - level) / 2
    lo, hi = np.quantile(rel, [a, 1 - a])
    return {"level": level, "draws": draws,
            "relative_ci": [float(lo), float(hi)],
            "absolute_ci": [float(np.quantile(absd, a)), float(np.quantile(absd, 1 - a))],
            "replicates_relative": rel.tolist()}


def equivalent(ci: list[float], margin: float) -> bool:
    """Two one-sided tests at level (1 - level)/2 each: the interval lies
    inside (-margin, +margin)."""
    return bool(-margin < ci[0] and ci[1] < margin)


# ----------------------------------------------------------------- superseded

def legacy_group_bootstrap(real_ap: np.ndarray, null_aps: np.ndarray,
                           cluster: np.ndarray, draws: int = 2000,
                           seed: int = 0, level: float = 0.90) -> list[float]:
    """The procedure used in the previous revision, kept only so the coverage
    study can show why it was replaced.

    It resamples groups of already-computed per-query scores and averages over
    the *same* null matrices every time, so neither the benchmark nor the null
    ensemble is re-drawn. With per-query null scores that are constant but
    differ between matrices it returns a zero-width interval.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster)
    index = {g: np.flatnonzero(cluster == g) for g in uniq}
    null_aps = np.atleast_2d(null_aps)
    rel = np.empty(draws)
    for b in range(draws):
        rows = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        base = null_aps[:, rows].mean()
        rel[b] = (real_ap[rows].mean() - base) / base
    a = (1 - level) / 2
    return [float(np.quantile(rel, a)), float(np.quantile(rel, 1 - a))]
