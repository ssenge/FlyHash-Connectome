"""E3: collision law of the fly hash (theory).

Pre-registered 2026-09-28 in experiments/PREREGISTRATION_2026-09-28.md (also
UPDATES.md section 14), before this code was written. The criteria below are
copied from there and are not to be changed after results are seen.

Design
------
Common protocol (flypath.replication, the 2017 benchmark): 10,000 items per
dataset (`replication.load_benchmark`, seed 0), 1,000 random queries per trial,
ground truth = the 200 (2%) nearest items by Euclidean distance on the input
features before row-centring, AP@200 with all 200 true neighbours in the
denominator (`replication.average_precision`). Every hash sees the row-centred
input (`replication.centre`). Hash codes are binary top-k winner sets (`_top_k`,
the tie rule of `replication.winners` made exact) ranked by Hamming distance
(`replication.rank_binary`).

  fly            sparse binary d x m matrix, every Kenyon cell samples s distinct
                 inputs (`replication.fly_matrix`)
  Gaussian-WTA   dense d x m matrix of i.i.d. N(0, 1) entries, the same top-k
  (GWTA)         (concomitant rank-order hash, Eshghi and Rajaram 2008)

(a) Equivalence. SIFT, GloVe, MNIST raw; 10 trials x 1,000 queries; m = 10d,
    s = 0.1d, k in {4, 16}. Pass: |AP_fly - AP_GWTA| / AP_GWTA <= 5% in >= 5 of
    the 6 (dataset, k) cells.
(b) Collapse. theta_half = the angle at which the mean winner-set overlap
    fraction |T(x) & T(y)| / k equals 0.5, from >= 1e5 pairs binned by angle,
    linear interpolation between bins. Predictor theta_half = c / t with
    t = isf(k / m) (inverse upper-tail standard normal). One c is fitted on
    MNIST random fly matrices over all 12 (k, m) cells and frozen (stored in the
    results before any held-out cell is computed). Held-out: k in {4, 16, 64,
    256} x m in {20k, 10d, 40d} on SIFT, GloVe and odour mixtures (36 cells),
    plus the 7 connectome matrices (`connectomes.projection`, binarised,
    `flyhash.load_odours` + `flyhash.align` to that hemisphere's DoOR
    glomeruli) at their own m with k = round(0.05 m), on odour mixtures of that
    hemisphere's DoOR profiles. Angles in degrees.
    Pass: |theta_half - c/t| < 3 deg in >= 90% of the 36 held-out random cells
    AND in >= 6 of 7 connectome matrices.
(c) Fan-in boundary. PCA-51 SIFT, MNIST, GloVe and odour mixtures (d = 35);
    m = 1,838; k in {16, 92}; s in {1, 2, 3, 5, 8, 13, 26, d} (26 capped at d).
    deficit(s) = 1 - AP_fly(s) / AP_GWTA (same m, k); d_eff = participation
    ratio (sum lambda)^2 / sum lambda^2 of the input covariance.
    Pass on >= 3 of 4 inputs, both k: Spearman(s, deficit) <= -0.9; deficit >
    10% for every s <= 5; deficit < 5% for every s >= 0.3 d_eff.

Results go to results/e3_collision.json. The runner checkpoints after every
unit (one trial of (a), one cell of (b), one trial of (c)) and resumes;
`criteria` recomputes every verdict from the stored per-cell numbers.

    VECLIB_MAXIMUM_THREADS=5 PYTHONPATH=. python -m flypath.collision --budget-minutes 7

stops cleanly before starting a new unit after 7 minutes; repeat it until it
prints "E3 complete" (exit status 0; 3 means "incomplete, run again"). A
results file written under another design is refused, not mixed.

Unspecified details fixed before running
----------------------------------------
Fixed on 2026-09-28 before any full-scale run; only a smoke run on a tiny
configuration (written to a scratch file) preceded the full run.

 1. Input and truth. Hashes see row-centred inputs (2017 protocol, step 1);
    GWTA gets the same input as the fly. Truth: raw features for (a) and the
    odour mixtures; for PCA-51 inputs in (c), the PCA features themselves
    (as in `replication.dimension_sweep`). PCA is fitted on the 10,000 items.
 2. s = max(1, round(0.1 d)) (the replication convention): SIFT 13, GloVe 30,
    MNIST 78, odours 4. Random fly matrices in (b) use the same s.
 3. Winner ties. Exact ties in drive (cells with identical inputs, frequent
    for small s and in connectome matrices; also cells whose inputs have equal
    integer sums, as in MNIST and SIFT) are broken by a fixed random
    priority over cells, drawn once per matrix draw and shared by all items:
    the tied cell with the higher priority wins (the rule of
    `replication.winners(y, k, prio)`, the repo's "random" tie rule, applied
    exactly by `_top_k`). The row-centred drive is computed as
    x @ w - mean(x) * colsum(w) (see `_wta`) so that drives tied in exact
    arithmetic stay bitwise tied. Code-review fix, before any full run:
    centring the input first in float32 left about 2% of MNIST items (k = 16)
    with a tie broken by rounding noise, and the jitter of
    `replication.winners` can vanish below float resolution; neither changes
    the drive, only which of two tied cells wins. The same
    priority is used for the fly and GWTA matrices of a trial (it has no effect
    on continuous Gaussian drives). Hamming ties in retrieval are broken at
    random (`replication.rank_binary`), with the same tie draws for every
    method within a trial.
 4. (a) Per trial: fresh queries, one fly and one GWTA matrix, both shared by
    k = 4 and 16. AP_fly and AP_GWTA are means over the 10 trials (the ratio
    of means is compared with the 5% tolerance).
 5. (b) Angle = angle between the row-centred input vectors (what the
    projection sees; for a Gaussian projection it is the arccos of the drive
    correlation), computed as 2 arcsin(|u - v| / 2) on unit vectors.
    Items whose row-centred vector is zero are excluded from pairs.
 6. (b) Pair sampling, identical for every dataset and fixed before any
    cell was computed: the 20 nearest neighbours by angle of every item (all
    items are anchors, so the smallest angles present in the data are
    covered) plus 100,000 uniformly random pairs (a != b); the union is
    deduplicated as unordered pairs (>= 1e5 pairs asserted). One pair set per
    dataset (seeded), shared by all cells of that dataset.
 7. (b) Binning: fixed 2 deg bins from 0 deg; bin position = mean angle of
    its pairs, bin value = mean overlap fraction; bins with < 100 pairs are
    dropped. theta_half = first downward crossing of 0.5 scanning upward in
    angle, by linear interpolation between adjacent retained bins. If the
    first retained bin is already below 0.5 ("below_range") or the curve never
    falls below 0.5 ("above_range"), theta_half is undefined and the cell
    counts as a failure (conservative). All non-empty bins with their counts
    are stored so theta_half can be recomputed.
 8. (b) Each random cell averages R = 5 independent fly matrices (each with
    its own tie priority): the per-pair overlap fraction is averaged over the
    5 draws before binning (the expected collision curve of the random
    construction). Per-draw theta_half values are stored too. A connectome
    matrix is fixed; its R = 5 draws are tie priorities only.
 9. (b) Grid cells are evaluated as labelled, each with independent draws,
    also where two labels give the same m (SIFT: k = 64 has 20k = 10d = 1280;
    k = 256 has 20k = 40d = 5120). The denominator is 36.
10. (b) c: least squares through the origin in degrees,
    c = sum(theta_i / t_i) / sum(1 / t_i^2), over MNIST cells with a defined
    theta_half. Stored once and never refitted on resume. A held-out cell
    with k/m >= 0.5 (t <= 0: odours, k = 256, m = 10d = 350) has no valid
    prediction and counts as a failure.
11. (b) Connectomes: binary matrices; odour mixtures = 10,000 Dirichlet
    mixtures (`flyhash.mixtures`, seed 0) of that hemisphere's DoOR profiles,
    as in `replication.controls`. The held-out random "odours" cells use the
    standard odour benchmark (MaleCNS R DoOR glomeruli, d = 35).
12. (c) 20 trials x 1,000 queries per input. Per trial: fresh queries, one
    GWTA matrix, one fly matrix per s (independent draws), all shared by
    k = 16 and 92; one tie priority and one retrieval tie seed for all
    methods of the trial. deficit(s) uses trial-mean APs (ratio of means).
13. (c) d_eff = (tr C)^2 / tr(C^2), C = covariance over items of the
    row-centred hash input. The value for the input before row-centring is
    stored as well (not used by the criteria).
14. (c) s grid per input: sorted set of {1, 2, 3, 5, 8, 13, min(26, d), d}.
    Spearman via scipy.stats.spearmanr over the 8 s values (nan fails).
    "Pass on >= 3 of 4 inputs, both k" is read strictly: an input passes if
    all three conditions hold at k = 16 AND at k = 92; E3(c) passes if >= 3
    inputs pass. The per-k counts (lenient reading) are reported too.
15. Seeds: numpy SeedSequence([20260928, part, keys...]) with string keys
    mapped by CRC-32; see `_rng`.

Design issue noted before running (no deviation taken)
------------------------------------------------------
At s = d every Kenyon cell of `replication.fly_matrix` samples all d inputs,
so all m cells are identical and have the same drive for every item. The
winner set is then fixed by tie-breaking alone and is the same for every item,
so AP_fly(d) is at chance and deficit(d) is close to 1. Because s = d >=
0.3 d_eff always, the third condition of (c) ("deficit < 5% for every
s >= 0.3 d_eff") fails at s = d by construction, and the extra high-deficit
point at the largest s also breaks Spearman <= -0.9. The pre-registered
verdict is computed exactly as registered. `criteria` also reports, labelled
"not pre-registered", the same three conditions with s = d left out; it
changes no pre-registered verdict. (Row-centring adds a related symmetry: a
cell sampling S has the negated drive of one sampling the complement of S, so
s = 26 of 35 behaves like s = 9.)

Two more design issues were found in code review before any full run. Both
come from the inputs alone, not from any hash or AP value, and no deviation
was taken for either.

(c) The low-s and high-s conditions contradict each other when
0.3 d_eff <= 5. Any s <= 5 that is also >= 0.3 d_eff would need a deficit
above 10% and below 5% at the same time. On the 10,000 benchmark items:

      input     d_eff (row-centred / before)   0.3 d_eff   s in both sets
      SIFT       6.68 / 6.70                   2.0         3, 5
      MNIST     21.05 / 21.22                  6.3         none
      GloVe     39.39 / 40.00                  11.8        none
      odours    11.57 / 8.23                   3.5 / 2.5   5 (3, 5 before)

SIFT and odours therefore cannot pass under either d_eff reading, so at most
2 of 4 inputs can pass and E3(c) fails by construction. This holds with or
without s = d, so the non-registered variant fails too. `criteria` reports
it as "contradictory_s" per input and k, and as "n_inputs_satisfiable".

(b) Natural data pairs cannot reach the angles the prediction needs on
GloVe. With the pair set of detail 6, the first 2-degree bin with >= 100
pairs is [46, 48) deg for GloVe; there are about 240 GloVe pairs below 46 deg.
For comparison, the same bin is [12, 14) for MNIST, [6, 8) for SIFT and
[0, 2) for odours. The measured theta_half is at least the angle of the first
kept bin, so a held-out cell can pass only if c/t > ~44 deg. For the 12
GloVe cells (t = 1.37 to 3.40) that needs c >= 60 (1 cell), 72 (4), 89 (2),
112 (2), 132 (2) or 150 deg (1). The pilot (theta_half about 1/t radians)
and the smoke run put c near 57-60 deg. Odours k = 256, m = 10d already fails
(t < 0), so the 90% criterion allows at most two GloVe failures, which needs
c >= 132 deg. So E3(b) (held-out) fails by construction for any plausible c.
`criteria` reports "first_retained_deg" and "pass_possible" for every
held-out cell, and "heldout_n_pass_possible".

Deviations
----------
None.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, spearmanr

from . import flyhash as fh
from . import replication as rp
from .config import ROOT, Config
from .connectomes import HEMISPHERES

RESULTS = ROOT / "results" / "e3_collision.json"
SEED = 20260928
M_LABELS = ("20k", "10d", "40d")

DESIGN = {
    "n_items": rp.N_DATA,
    "queries": rp.N_QUERIES,
    "top_fraction": rp.TOP,
    "sampling": rp.SAMPLING,
    "a": {"datasets": ["sift", "glove", "mnist"], "trials": 10, "ks": [4, 16],
          "m_per_d": 10, "tolerance": 0.05, "cells_needed": 5},
    "b": {"fit_dataset": "mnist", "heldout": ["sift", "glove", "odours"],
          "ks": [4, 16, 64, 256], "ms": list(M_LABELS), "draws": 5,
          "nn_per_item": 20, "random_pairs": 100_000, "min_pairs": 100_000,
          "bin_width_deg": 2.0, "min_bin": 100, "level": 0.5,
          "tolerance_deg": 3.0, "pass_fraction": 0.9,
          "hemispheres": [list(h) for h in HEMISPHERES],
          "connectome_sparsity": 0.05, "connectome_draws": 5, "connectomes_needed": 6},
    "c": {"inputs": ["sift", "mnist", "glove", "odours"], "pca_dims": 51, "m": 1838,
          "ks": [16, 92], "s_grid": [1, 2, 3, 5, 8, 13, 26, "d"], "trials": 20,
          "spearman_max": -0.9, "low_s_max": 5, "low_s_deficit": 0.10,
          "high_s_factor": 0.3, "high_s_deficit": 0.05, "inputs_needed": 3},
}

# A tiny configuration that only checks that every code path runs.
SMOKE = copy.deepcopy(DESIGN)
SMOKE.update({"n_items": 2000, "queries": 100})
SMOKE["a"].update({"datasets": ["sift"], "trials": 1})
SMOKE["b"].update({"heldout": ["odours"], "ks": [4, 16], "ms": ["20k", "10d"], "draws": 2,
                   "nn_per_item": 5, "random_pairs": 5000, "min_pairs": 5000, "min_bin": 20,
                   "hemispheres": [["malecns", "R"]], "connectomes_needed": 1})
SMOKE["c"].update({"inputs": ["sift", "odours"], "trials": 1})


# ---------------------------------------------------------------- estimators

def unit_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows scaled to unit norm, and a mask of rows with non-zero norm."""
    nrm = np.linalg.norm(x, axis=1)
    ok = nrm > 0
    u = np.zeros_like(x, dtype=np.float64)
    u[ok] = x[ok] / nrm[ok, None]
    return u, ok


def angles_deg(ua: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Angle in degrees between unit vectors, 2 arcsin(|u - v| / 2): accurate at
    small angles, where arccos of the dot product is not."""
    half = np.linalg.norm(ua - ub, axis=1) / 2
    return np.degrees(2 * np.arcsin(np.clip(half, 0.0, 1.0)))


def sample_pairs(xc: np.ndarray, nn_per_item: int, n_random: int, rng,
                 chunk: int = 1000) -> tuple[np.ndarray, np.ndarray, dict]:
    """Unordered item pairs (a < b): every item's `nn_per_item` nearest
    neighbours by angle plus `n_random` uniform random pairs, deduplicated.
    Items with a zero vector are left out."""
    u, ok = unit_rows(xc)
    idx = np.flatnonzero(ok)
    U = u[idx]
    n = len(idx)
    parts_a, parts_b = [], []
    if nn_per_item > 0:
        for i in range(0, n, chunk):
            sim = U[i:i + chunk] @ U.T
            rows = np.arange(len(sim))
            sim[rows, i + rows] = -np.inf
            nb = np.argpartition(-sim, nn_per_item - 1, axis=1)[:, :nn_per_item]
            parts_a.append(np.repeat(i + rows, nn_per_item))
            parts_b.append(nb.ravel())
    ra = rng.integers(0, n, n_random)
    rb = (ra + rng.integers(1, n, n_random)) % n          # uniform over b != a
    parts_a.append(ra)
    parts_b.append(rb)
    a, b = np.concatenate(parts_a), np.concatenate(parts_b)
    pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0)
    meta = {"items_used": int(n), "items_zero_norm": int((~ok).sum()),
            "nn_ordered": int(n * nn_per_item), "random_ordered": int(n_random),
            "unique_pairs": int(len(pairs))}
    return idx[pairs[:, 0]], idx[pairs[:, 1]], meta


def pair_overlap(tags: sp.csr_matrix, a: np.ndarray, b: np.ndarray,
                 chunk: int = 50_000) -> np.ndarray:
    """|T(a) & T(b)| / k for each pair, k = number of winners of item a."""
    k = np.diff(tags.indptr)
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        sa, sb = a[i:i + chunk], b[i:i + chunk]
        shared = np.asarray(tags[sa].multiply(tags[sb]).sum(1)).ravel()
        out[i:i + chunk] = shared / k[sa]
    return out


def bin_curve(angle: np.ndarray, value: np.ndarray, width: float) -> dict:
    """Non-empty fixed-width angle bins starting at 0: lower edge, mean angle,
    mean value and count per bin."""
    b = np.floor(np.asarray(angle) / width).astype(np.int64)
    cnt = np.bincount(b)
    s_ang = np.bincount(b, weights=angle)
    s_val = np.bincount(b, weights=value)
    nz = np.flatnonzero(cnt)
    return {"lo": (nz * width).tolist(), "angle": (s_ang[nz] / cnt[nz]).tolist(),
            "value": (s_val[nz] / cnt[nz]).tolist(), "count": cnt[nz].tolist()}


def theta_half(curve: dict, min_count: int, level: float = 0.5) -> tuple[float, str, int]:
    """Angle at which the binned curve first falls through `level`, by linear
    interpolation between adjacent bins with >= `min_count` pairs.

    Returns (theta, status, number of downward crossings); theta is nan when
    the first retained bin is already below `level` ("below_range"), when the
    curve never falls below it ("above_range") or when no bin is retained.
    """
    ang = np.asarray(curve["angle"], float)
    val = np.asarray(curve["value"], float)
    keep = np.asarray(curve["count"]) >= min_count
    ang, val = ang[keep], val[keep]
    if len(ang) == 0:
        return float("nan"), "no_bins", 0
    above = val >= level
    cross = np.flatnonzero(above[:-1] & ~above[1:])
    if not above[0]:
        return float("nan"), "below_range", int(len(cross))
    if len(cross) == 0:
        return float("nan"), "above_range", 0
    i = int(cross[0])
    th = ang[i] + (val[i] - level) * (ang[i + 1] - ang[i]) / (val[i] - val[i + 1])
    return float(th), "ok", int(len(cross))


def first_retained(curve: dict, min_count: int) -> float | None:
    """Mean angle of the first bin with >= `min_count` pairs: theta_half can
    only be measured at or above it (diagnostic)."""
    keep = np.flatnonzero(np.asarray(curve["count"]) >= min_count)
    return float(curve["angle"][keep[0]]) if len(keep) else None


def threshold(k: int, m: int) -> float:
    """t = inverse upper-tail standard normal of k/m."""
    return float(norm.isf(k / m))


def fit_c(theta: np.ndarray, t: np.ndarray) -> float:
    """Least-squares c in theta = c / t (through the origin in 1/t)."""
    theta, t = np.asarray(theta, float), np.asarray(t, float)
    return float(np.sum(theta / t) / np.sum(1.0 / t ** 2))


def participation_ratio(z: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the covariance of the rows of z,
    computed as (tr C)^2 / tr(C^2) without an eigendecomposition."""
    c = np.cov(np.asarray(z, float), rowvar=False)
    return float(np.trace(c) ** 2 / np.sum(c * c))


def s_values(grid, d: int) -> list[int]:
    """The fan-in grid for input dimension d: "d" means d, values capped at d."""
    return sorted({d if s == "d" else min(int(s), d) for s in grid})


def equivalence_verdict(ap_fly: float, ap_gwta: float, tolerance: float) -> dict:
    rel = abs(ap_fly - ap_gwta) / ap_gwta
    return {"ap_fly": ap_fly, "ap_gwta": ap_gwta, "relative_difference": rel,
            "pass": bool(rel <= tolerance)}


def fanin_verdict(s: list[int], deficit: list[float], d_eff: float, c: dict) -> dict:
    """The three (c) conditions for one input and one k."""
    s, deficit = np.asarray(s), np.asarray(deficit, float)
    with warnings.catch_warnings():                   # constant input -> nan, which fails
        warnings.simplefilter("ignore")
        rho = float(spearmanr(s, deficit).statistic) if len(s) > 1 else float("nan")
    is_low = s <= c["low_s_max"]
    is_high = s >= c["high_s_factor"] * d_eff
    low, high = deficit[is_low], deficit[is_high]
    ok_rho = bool(rho <= c["spearman_max"])            # nan fails
    ok_low = bool(len(low) > 0 and (low > c["low_s_deficit"]).all())
    ok_high = bool(len(high) > 0 and (high < c["high_s_deficit"]).all())
    # an s in both sets would need deficit > 10% and < 5% at once: the input
    # cannot pass whatever the data (diagnostic only; the verdict is unchanged)
    both = s[is_low & is_high] if c["low_s_deficit"] >= c["high_s_deficit"] else s[:0]
    return {"spearman": rho, "spearman_ok": ok_rho,
            "low_s": s[is_low].tolist(), "low_s_ok": ok_low,
            "high_s": s[is_high].tolist(), "high_s_ok": ok_high,
            "contradictory_s": both.tolist(),
            "pass": ok_rho and ok_low and ok_high}


# ---------------------------------------------------------------- hashing

def _rng(*keys) -> np.random.Generator:
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng([SEED, *ints])


def _top_k(y: np.ndarray, k: int, prio: np.ndarray) -> sp.csr_matrix:
    """Exactly k winners per row as a sparse 0/1 matrix: every cell whose
    drive exceeds the row's k-th largest drive and, among the cells tied at
    it, those with the highest `prio` (a fixed priority over cells, the same
    for every row). This is the rule of `replication.winners(y, k, prio)`, but
    exact: `winners` adds a jitter of 0.25 x the smallest gap between distinct
    values in the first 200 rows, which can fall below the floating-point
    resolution of the largest drives and then leaves ties to argpartition."""
    n, m = y.shape
    kth = -np.partition(-y, k - 1, axis=1)[:, k - 1:k]
    win = y > kth
    tied = y == kth
    need = k - win.sum(axis=1)                        # >= 1 tied cells still to take
    easy = tied.sum(axis=1) == need                   # all tied cells win
    win[easy] |= tied[easy]
    hard = np.flatnonzero(~easy)
    if len(hard):
        order = np.argsort(-prio, kind="stable")      # highest priority first
        t = tied[hard][:, order]
        take = np.zeros_like(t)
        take[:, order] = t & (np.cumsum(t, axis=1) <= need[hard, None])
        win[hard] |= take
    rows, cols = np.nonzero(win)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, m))


def _wta(x: np.ndarray, w: np.ndarray, ks, prio: np.ndarray) -> dict:
    """Top-k winner sets (`_top_k`) of the row-centred drive centre(x) @ w
    for every k, computed in row chunks. `x` is the input BEFORE row-centring.

    The drive is computed as x @ w - mean(x) * colsum(w) (float32 product,
    float64 shift), which equals centre(x) @ w. Centring first in floating
    point would round every x_i - mean(x) separately, so two cells whose drives
    are tied in exact arithmetic (e.g. integer pixels summed over different
    input sets of the same size) would differ by rounding noise and the tie
    would be broken by that noise, differently for every item, instead of by
    `prio`. Here equal integer sums stay bitwise equal (exact in float32 below
    2**24), and so do the drives of cells with identical inputs.
    """
    n, m = len(x), w.shape[1]
    x32 = np.asarray(x, np.float32)
    mu = np.asarray(x, np.float64).mean(axis=1)
    colsum = np.asarray(w, np.float64).sum(axis=0)
    chunk = max(200, int(2e7 // m))
    parts = {k: [] for k in ks}
    for i in range(0, n, chunk):
        y = (x32[i:i + chunk] @ w).astype(np.float64)
        y -= mu[i:i + chunk, None] * colsum[None, :]
        for k in ks:
            parts[k].append(_top_k(y, k, prio))
    return {k: sp.vstack(v).tocsr() for k, v in parts.items()}


def _retrieval(tags: sp.csr_matrix, q, true, tie_seed: int) -> dict:
    pred = rp.rank_binary(tags, q, true.shape[1], np.random.default_rng(tie_seed))
    return {"ap": rp.average_precision(pred, true), "recall": rp.recall(pred, true)}


# ---------------------------------------------------------------- data

class _Data:
    """Datasets loaded once per process."""

    def __init__(self, cfg: Config, n: int, log):
        self.cfg, self.n, self.log, self._raw, self._pairs = cfg, n, log, {}, {}

    def raw(self, name: str) -> np.ndarray:
        if name not in self._raw:
            t0 = time.time()
            self._raw[name] = rp.load_benchmark(self.cfg, name, n=self.n)
            self.log(f"  loaded {name} {self._raw[name].shape} ({time.time() - t0:.0f}s)")
        return self._raw[name]

    def input_c(self, name: str, dims: int) -> np.ndarray:
        """Input of (c): raw odour mixtures, otherwise PCA to `dims` components."""
        key = f"c:{name}:{dims}"
        if key not in self._raw:
            x = self.raw(name)
            self._raw[key] = x if name == "odours" else rp.pca(x, dims)
        return self._raw[key]

    def connectome(self, ds: str, side: str):
        from . import connectomes as C
        key = f"{ds}_{side}"
        if key not in self._raw:
            p = C.projection(self.cfg, ds, side)
            pb = fh.Projection("b", p.binary, p.glomeruli, meta=p.meta)
            od = fh.load_odours(self.cfg, p.glomeruli)
            po = fh.align(pb, od.glomeruli)
            x = rp.load_benchmark(self.cfg, "odours", n=self.n, odours=od.x)
            self._raw[key] = (po.binary.astype(np.float32), x, od.glomeruli)
        return self._raw[key]

    def pairs(self, key: str, x: np.ndarray, b: dict):
        if key not in self._pairs:
            xc = rp.centre(x)
            a_, b_, meta = sample_pairs(xc, b["nn_per_item"], b["random_pairs"],
                                        _rng("b-pairs", key))
            if len(a_) < b["min_pairs"]:
                raise RuntimeError(f"{key}: only {len(a_)} pairs")
            u, _ = unit_rows(xc)
            ang = np.concatenate([angles_deg(u[a_[i:i + 50_000]], u[b_[i:i + 50_000]])
                                  for i in range(0, len(a_), 50_000)])
            h = hashlib.sha1(np.stack([a_, b_]).astype(np.int64).tobytes()).hexdigest()
            meta.update({"sha1": h, "angle_quantiles_deg": dict(zip(
                ("0", "1", "5", "25", "50", "75", "95", "100"),
                np.percentile(ang, [0, 1, 5, 25, 50, 75, 95, 100]).tolist()))})
            # the hash input is returned uncentred: `_wta` centres it exactly
            self._pairs[key] = (x, a_, b_, ang, meta)
        return self._pairs[key]


# ---------------------------------------------------------------- units

def _unit_a(x: np.ndarray, name: str, t: int, D: dict) -> dict:
    n, d = x.shape
    a = D["a"]
    m = a["m_per_d"] * d
    s = max(1, int(round(D["sampling"] * d)))
    top = int(D["top_fraction"] * n)
    rng = _rng("a", name, t)
    q = rng.choice(n, D["queries"], replace=False)
    true = rp.truth(x, q, top)
    w = {"fly": rp.fly_matrix(d, m, rng, sampled=s),
         "gwta": rng.normal(size=(d, m)).astype(np.float32)}
    prio = rng.random(m)
    tie_seed = int(rng.integers(2**31))
    out = {}
    for method, wm in w.items():
        tags = _wta(x, wm, a["ks"], prio)              # row-centred drive
        out[method] = {str(k): _retrieval(tags[k], q, true, tie_seed) for k in a["ks"]}
    return out


def _cell_b(x, pa, pb, ang, draws, k: int, b: dict) -> dict:
    """Collision curve and theta_half of one (matrix family, k) cell; `x` is
    the input before row-centring (`_wta` centres it); `draws` yields (matrix,
    tie priority) pairs."""
    acc = np.zeros(len(pa))
    per_draw = []
    r = 0
    for w, prio in draws:
        ov = pair_overlap(_wta(x, w, [k], prio)[k], pa, pb)
        acc += ov
        per_draw.append(theta_half(bin_curve(ang, ov, b["bin_width_deg"]),
                                   b["min_bin"], b["level"])[0])
        r += 1
    curve = bin_curve(ang, acc / r, b["bin_width_deg"])
    th, status, ncross = theta_half(curve, b["min_bin"], b["level"])
    return {"theta_half": th, "status": status, "n_crossings": ncross,
            "theta_half_draws": per_draw, "draws": r, "n_pairs": int(len(pa)), "curve": curve}


def _unit_c(z: np.ndarray, name: str, t: int, D: dict) -> dict:
    n, d = z.shape
    c = D["c"]
    m = c["m"]
    top = int(D["top_fraction"] * n)
    rng = _rng("c", name, t)
    q = rng.choice(n, D["queries"], replace=False)
    true = rp.truth(z, q, top)
    prio = rng.random(m)
    tie_seed = int(rng.integers(2**31))
    wg = rng.normal(size=(d, m)).astype(np.float32)
    tags = _wta(z, wg, c["ks"], prio)                  # row-centred drive
    out = {"gwta": {str(k): _retrieval(tags[k], q, true, tie_seed) for k in c["ks"]}, "fly": {}}
    for s in s_values(c["s_grid"], d):
        w = rp.fly_matrix(d, m, _rng("c", name, t, "s", s), sampled=s)
        tags = _wta(z, w, c["ks"], prio)
        out["fly"][str(s)] = {str(k): _retrieval(tags[k], q, true, tie_seed) for k in c["ks"]}
    return out


# ---------------------------------------------------------------- criteria

def _m_of(label: str, k: int, d: int) -> int:
    return {"20k": 20 * k, "10d": 10 * d, "40d": 40 * d}[label]


def b_cells(D: dict) -> list[dict]:
    """The random-matrix cells of (b): fit cells first, then held-out ones."""
    b = D["b"]
    out = []
    for role, names in (("fit", [b["fit_dataset"]]), ("heldout", b["heldout"])):
        for name in names:
            for k in b["ks"]:
                for lab in b["ms"]:
                    out.append({"key": f"{name}|k={k}|m={lab}", "role": role, "dataset": name,
                                "k": k, "m_label": lab})
    return out


def _mean(trials: dict, *path) -> float:
    vals = []
    for tr in trials.values():
        v = tr
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.mean(vals))


def criteria(res: dict) -> dict:
    """Every pre-registered verdict, recomputed from the stored numbers."""
    D = res["design"]
    out = {}

    # (a)
    a, rows, done = D["a"], [], True
    for name in a["datasets"]:
        tr = res.get("a", {}).get(name, {}).get("trials", {})
        done &= len(tr) == a["trials"]
        for k in a["ks"]:
            if tr:
                v = equivalence_verdict(_mean(tr, "fly", str(k), "ap"),
                                        _mean(tr, "gwta", str(k), "ap"), a["tolerance"])
                rows.append({"dataset": name, "k": k, "trials": len(tr), **v})
    n_ok = sum(r["pass"] for r in rows)
    out["a"] = {"complete": done, "cells": rows, "n_pass": n_ok,
                "n_cells": len(a["datasets"]) * len(a["ks"]),
                "pass": bool(n_ok >= a["cells_needed"]) if done else None}

    # (b)
    b = D["b"]
    rb = res.get("b", {})
    cells = rb.get("cells", {})
    fit = rb.get("fit")
    c_deg = fit["c_deg"] if fit else None
    held, conn = [], []
    for key, cell in cells.items():
        if cell["role"] == "fit":
            continue
        t = cell["t"]
        pred = c_deg / t if (c_deg is not None and t > 0) else None
        th = cell["theta_half"]
        err = th - pred if (pred is not None and th is not None) else None
        # diagnostic: the measured theta_half is >= the first retained bin, so a
        # prediction more than the tolerance below it cannot pass
        fr = first_retained(cell["curve"], b["min_bin"]) if "curve" in cell else None
        possible = pred is not None and (fr is None or pred + b["tolerance_deg"] > fr)
        row = {"key": key, "k": cell["k"], "m": cell["m"], "t": t, "theta_half": th,
               "status": cell["status"], "predicted": pred, "error_deg": err,
               "first_retained_deg": fr, "pass_possible": bool(possible),
               "pass": bool(err is not None and abs(err) < b["tolerance_deg"])}
        (conn if cell["role"] == "connectome" else held).append(row)
    n_held = len(b["heldout"]) * len(b["ks"]) * len(b["ms"])
    n_conn = len(b["hemispheres"])
    done_b = fit is not None and len(held) == n_held and len(conn) == n_conn
    frac = sum(r["pass"] for r in held) / n_held
    n_conn_ok = sum(r["pass"] for r in conn)
    out["b"] = {"complete": done_b, "c_deg": c_deg, "heldout": held, "connectomes": conn,
                "heldout_n_pass": sum(r["pass"] for r in held), "heldout_n_cells": n_held,
                "heldout_n_pass_possible": sum(r["pass_possible"] for r in held),
                "heldout_fraction": frac, "connectome_n_pass": n_conn_ok,
                "connectome_n": n_conn,
                "pass_heldout": bool(frac >= b["pass_fraction"]) if done_b else None,
                "pass_connectomes": bool(n_conn_ok >= b["connectomes_needed"]) if done_b else None}
    out["b"]["pass"] = (out["b"]["pass_heldout"] and out["b"]["pass_connectomes"]) if done_b else None

    # (c)
    c = D["c"]
    per_input, per_input_x, done_c = {}, {}, True
    for name in c["inputs"]:
        r = res.get("c", {}).get(name)
        if not r or not r.get("trials"):
            done_c = False
            continue
        tr = r["trials"]
        done_c &= len(tr) == c["trials"]
        s = r["s_grid"]
        ks = {}
        ks_x = {}
        for k in c["ks"]:
            g = _mean(tr, "gwta", str(k), "ap")
            f = [_mean(tr, "fly", str(si), str(k), "ap") for si in s]
            dfc = [1 - fi / g for fi in f]
            ks[str(k)] = {"ap_gwta": g, "ap_fly": f, "deficit": dfc,
                          **fanin_verdict(s, dfc, r["d_eff"], c)}
            keep = [i for i, si in enumerate(s) if si != r["d"]]
            ks_x[str(k)] = fanin_verdict([s[i] for i in keep], [dfc[i] for i in keep],
                                         r["d_eff"], c)
        per_input[name] = {"d": r["d"], "d_eff": r["d_eff"], "s": s, "trials": len(tr),
                           "k": ks,
                           "satisfiable": not any(v["contradictory_s"] for v in ks.values()),
                           "pass": all(v["pass"] for v in ks.values())}
        per_input_x[name] = {"k": ks_x, "pass": all(v["pass"] for v in ks_x.values())}
    n_in = sum(v["pass"] for v in per_input.values())
    per_k = {str(k): sum(v["k"][str(k)]["pass"] for v in per_input.values()) for k in c["ks"]}
    out["c"] = {"complete": done_c, "inputs": per_input, "n_inputs_pass_both_k": n_in,
                "n_inputs_pass_per_k": per_k,
                "n_inputs_satisfiable": sum(v["satisfiable"] for v in per_input.values()),
                "pass": bool(n_in >= c["inputs_needed"]) if done_c else None,
                "pass_lenient_per_k": bool(all(v >= c["inputs_needed"] for v in per_k.values()))
                if done_c else None,
                "not_preregistered_without_s_eq_d": {
                    "note": "s = d is degenerate (all Kenyon cells identical); this "
                            "evaluation leaves it out and is NOT pre-registered",
                    "inputs": per_input_x,
                    "n_inputs_pass_both_k": sum(v["pass"] for v in per_input_x.values())}}
    out["complete"] = bool(out["a"]["complete"] and out["b"]["complete"] and out["c"]["complete"])
    return out


# ---------------------------------------------------------------- runner

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    os.replace(tmp, path)


def _sha256_self() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(cfg: Config, out: Path | str = RESULTS, design: dict | None = None,
        budget_minutes: float | None = None, log=print) -> dict:
    """Run E3, resuming from `out` if it exists. Stops cleanly before starting
    a new unit once `budget_minutes` have passed; call again to continue."""
    D = json.loads(json.dumps(design or DESIGN))
    out = Path(out)
    t0 = time.time()
    deadline = None if budget_minutes is None else t0 + 60 * budget_minutes

    if out.exists():
        res = json.loads(out.read_text())
        if res["design"] != D:
            raise SystemExit(f"{out} was produced with a different design; refusing to mix")
        log(f"resuming {out}")
    else:
        res = {"experiment": "E3 collision law of the fly hash",
               "preregistration": "experiments/PREREGISTRATION_2026-09-28.md (UPDATES.md s. 14)",
               "design": D, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "code_sha256_at_creation": _sha256_self(), "invocations": [],
               "a": {}, "b": {"pairs": {}, "cells": {}}, "c": {}}
    res["invocations"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "code_sha256": _sha256_self(), "python": platform.python_version(),
                               "numpy": np.__version__})
    _save(res, out)

    def time_left() -> bool:
        return deadline is None or time.time() < deadline

    data = _Data(cfg, D["n_items"], log)
    finished = True

    # (a) equivalence
    for name in D["a"]["datasets"]:
        entry = res["a"].setdefault(name, {"trials": {}})
        for t in range(D["a"]["trials"]):
            if str(t) in entry["trials"]:
                continue
            if not time_left():
                finished = False
                break
            x = data.raw(name)
            n, d = x.shape
            entry.update({"d": d, "m": D["a"]["m_per_d"] * d,
                          "s": max(1, int(round(D["sampling"] * d)))})
            t1 = time.time()
            r = _unit_a(x, name, t, D)
            r["seconds"] = time.time() - t1
            entry["trials"][str(t)] = _clean(r)
            _save(res, out)
            log(f"  (a) {name} trial {t + 1}/{D['a']['trials']}: " + "  ".join(
                f"k={k} fly {r['fly'][str(k)]['ap']:.4f} gwta {r['gwta'][str(k)]['ap']:.4f}"
                for k in D["a"]["ks"]) + f"  ({r['seconds']:.0f}s)")

    # (b) collapse: MNIST fit cells, then the frozen c, then held-out cells
    B = D["b"]
    for cell in b_cells(D):
        if cell["key"] in res["b"]["cells"]:
            continue
        if cell["role"] == "heldout" and "fit" not in res["b"]:
            _fit(res, D, out, log)
            if "fit" not in res["b"]:
                finished = False
                break
        if not time_left():
            finished = False
            break
        name = cell["dataset"]
        x = data.raw(name)
        d = x.shape[1]
        xh, pa, pb, ang, meta = data.pairs(name, x, B)
        _check_pairs(res, name, meta)
        s = max(1, int(round(D["sampling"] * d)))
        k = cell["k"]
        m = _m_of(cell["m_label"], k, d)

        def draws(key=cell["key"], d=d, m=m, s=s):
            for r in range(B["draws"]):
                g = _rng("b", key, r)
                yield rp.fly_matrix(d, m, g, sampled=s), g.random(m)
        t1 = time.time()
        r = _cell_b(xh, pa, pb, ang, draws(), k, B)
        res["b"]["cells"][cell["key"]] = _clean({**cell, "m": m, "d": d, "s": s,
                                                 "t": threshold(k, m), "k_over_m": k / m, **r,
                                                 "seconds": time.time() - t1})
        _save(res, out)
        log(f"  (b) {cell['key']}: m={m} theta_half {r['theta_half']:.2f} ({r['status']}), "
            f"1/t {np.degrees(1 / threshold(k, m)) if k / m < 0.5 else float('nan'):.2f} deg "
            f"({time.time() - t1:.0f}s)")
    if finished and "fit" not in res["b"]:
        _fit(res, D, out, log)

    # (b) connectome matrices
    if finished and "fit" in res["b"]:
        for ds, side in B["hemispheres"]:
            key = f"connectome|{ds}_{side}"
            if key in res["b"]["cells"]:
                continue
            if not time_left():
                finished = False
                break
            t1 = time.time()
            w, x, gloms = data.connectome(ds, side)
            d, m = w.shape
            k = int(round(B["connectome_sparsity"] * m))
            pkey = f"odours_{ds}_{side}"
            xh, pa, pb, ang, meta = data.pairs(pkey, x, B)
            _check_pairs(res, pkey, meta)

            def draws(key=key, w=w, m=m):
                for r in range(B["connectome_draws"]):
                    yield w, _rng("b", key, r).random(m)
            r = _cell_b(xh, pa, pb, ang, draws(), k, B)
            res["b"]["cells"][key] = _clean({"key": key, "role": "connectome", "dataset": pkey,
                                             "hemisphere": f"{ds}_{side}", "glomeruli": gloms,
                                             "d": d, "m": m, "k": k, "nnz": int(w.sum()),
                                             "inputs_mean": float(w.sum(0).mean()),
                                             "t": threshold(k, m), "k_over_m": k / m, **r,
                                             "seconds": time.time() - t1})
            _save(res, out)
            log(f"  (b) {key}: d={d} m={m} k={k} theta_half {r['theta_half']:.2f} "
                f"({r['status']}) ({time.time() - t1:.0f}s)")

    # (c) fan-in boundary
    C_ = D["c"]
    for name in C_["inputs"] if finished else []:
        entry = res["c"].setdefault(name, {"trials": {}})
        for t in range(C_["trials"]):
            if str(t) in entry["trials"]:
                continue
            if not time_left():
                finished = False
                break
            z = data.input_c(name, C_["pca_dims"])
            if "d_eff" not in entry:
                d = z.shape[1]
                entry.update({"d": d, "m": C_["m"], "s_grid": s_values(C_["s_grid"], d),
                              "input": "odour mixtures, raw" if name == "odours"
                              else f"PCA-{C_['pca_dims']}",
                              "d_eff": participation_ratio(rp.centre(z)),
                              "d_eff_before_row_centring": participation_ratio(z)})
            t1 = time.time()
            r = _unit_c(z, name, t, D)
            r["seconds"] = time.time() - t1
            entry["trials"][str(t)] = _clean(r)
            _save(res, out)
            k0 = str(C_["ks"][0])
            log(f"  (c) {name} trial {t + 1}/{C_['trials']} (d_eff {entry['d_eff']:.1f}): k={k0} "
                f"gwta {r['gwta'][k0]['ap']:.4f} fly " + " ".join(
                    f"s{s}:{v[k0]['ap']:.4f}" for s, v in r["fly"].items())
                + f" ({r['seconds']:.0f}s)")
        if not finished:
            break

    res["criteria"] = criteria(res)
    res["complete"] = bool(finished and res["criteria"]["complete"])
    res["invocations"][-1]["seconds"] = time.time() - t0
    _save(res, out)
    log("E3 complete" if res["complete"] else "E3 incomplete: run the same command again to resume")
    return res


def _check_pairs(res: dict, key: str, meta: dict) -> None:
    """The pair set of a dataset is stored once; a resumed run must rebuild it
    identically."""
    stored = res["b"]["pairs"].get(key)
    if stored is None:
        res["b"]["pairs"][key] = meta
    elif stored["sha1"] != meta["sha1"]:
        raise RuntimeError(f"pair set of {key} changed between invocations")


def _fit(res: dict, D: dict, out: Path, log) -> None:
    """Fit c on the MNIST cells once they are all done, and freeze it."""
    B = D["b"]
    fit_cells = [c for c in res["b"]["cells"].values() if c["role"] == "fit"]
    if len(fit_cells) < len(B["ks"]) * len(B["ms"]):
        return
    use = [c for c in fit_cells if c["theta_half"] is not None and np.isfinite(c["theta_half"])
           and c["t"] > 0]
    if not use:
        raise RuntimeError("no MNIST cell has a defined theta_half; c cannot be fitted")
    c = fit_c([u["theta_half"] for u in use], [u["t"] for u in use])
    res["b"]["fit"] = {
        "c_deg": c, "c_rad": float(np.radians(c)), "n_cells_used": len(use),
        "n_cells": len(fit_cells), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "in_sample": [{"key": u["key"], "t": u["t"], "theta_half": u["theta_half"],
                       "predicted": c / u["t"], "error_deg": u["theta_half"] - c / u["t"]}
                      for u in use]}
    _save(res, out)
    log(f"  (b) c fitted on {len(use)} MNIST cells and frozen: c = {c:.3f} deg")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flypath.collision",
                                description="E3: collision law of the fly hash (pre-registered)")
    p.add_argument("--config", help="a YAML config overriding config.example.yaml")
    p.add_argument("--out", default=str(RESULTS), help="results JSON (checkpoint and resume)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny configuration that only checks the code runs; needs --out")
    p.add_argument("--budget-minutes", type=float, default=None,
                   help="stop cleanly before starting a new unit after this many minutes")
    args = p.parse_args(argv)
    if args.smoke and Path(args.out).resolve() == RESULTS.resolve():
        p.error("--smoke must not write to results/e3_collision.json; pass --out")
    from . import config
    cfg = config.load(args.config)
    res = run(cfg, out=args.out, design=SMOKE if args.smoke else DESIGN,
              budget_minutes=args.budget_minutes,
              log=lambda s: print(s, flush=True))
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
