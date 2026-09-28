"""E4 stage 0: the spectral fan-out allocation f*(Lambda, k), derived and frozen
before any retrieval (AP) run.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, E4, stage 0:
"Derive f*(Lambda, k) under a Gaussian-drive, order-statistic top-k model at
nnz = 9,967 with MaleCNS R inputs per cell. Freeze it (code plus SHA-256)
before any new AP run." The allocation is chosen to maximise the model's
collision slope, -d E[overlap] / d angle, at the data's median angle between
items and their 200 nearest neighbours. Nothing in this module computes a
retrieval score.

`fstar(lam, k, inputs_per_cell, nnz, theta)` returns f*: one non-negative
integer fan-out per input dimension, summing to nnz. `run()` (CLI:
`python -m flypath.spectral`) freezes f* for the stage-1 synthetic spectra in
results/e4_frozen.json together with the SHA-256 of this file and a UTC time.
For real inputs (stage 2), f* depends on the training split's spectrum and
angle, so only the code is frozen: stage 2 calls `verify_frozen()` and then
`fstar(spectrum(train), k, c, nnz, median_nn_angle(train))` on the training
split alone.

This is the second freeze, re-frozen before any AP run to correct the
neighbour pair model (see "Re-freeze" at the end of this docstring).


The model
---------
  inputs    x ~ N(0, Lambda), Lambda = diag(lambda_1..lambda_d): the raw
            coordinates the benchmark hashes (PCA components for real data).
            The hash sees the row-centred vector x~ = x - mean_j(x_j) 1, as in
            the 2017 protocol (`replication.centre`).
  wiring    a binary d x m matrix M. Column sums are the MaleCNS right
            hemisphere's inputs per cell c_i (m = 1886 cells, sum c = nnz =
            9967); row sums are the fan-out f, the design variable. Given
            (c, f), M is uniform over binary matrices with these margins: a
            greedy realisation of the margins followed by 30 curveball sweeps,
            the construction of `flyhash.margin_control`.
  drives    y = M^T x~. Cell i's drive is Gaussian with variance
            sum_{j in S_i} lambda_j (S_i = the cell's inputs), up to the
            row-centring correction. Cells that share inputs are correlated,
            and the model keeps that correlation.
  top-k     T(x) = the k cells with the largest drives. Exact ties between
            cells with identical input sets are broken by one fixed random
            priority over cells per wiring, the same for both items of a pair.
  pairs     neighbour pairs at a water level D (derivation 1).
  objective S(f) = -d E[|T(x) & T(x')| / k] / d theta at theta = theta_data,
            estimated as a central difference over theta +- 2.5 deg.


Derivation
----------
1. Neighbour pairs (analytic). The ground truth ranks by Euclidean distance on
   the raw input, so a neighbour of an item x ~ N(0, Lambda) is an item
   x' ~ N(0, Lambda) conditioned on a small |x - x'|. Replace the hard radius
   by its exponential tilt exp(-s |x - x'|^2 / 2), the Lagrangian form of the
   radius constraint, and write D = 1/s. Given x, the density of x' is
   proportional to N(x'; 0, Lambda) exp(-s |x - x'|^2 / 2): Gaussian and
   independent across dimensions, with precision 1/lambda_j + s and mean
   s x_j / (1/lambda_j + s). That is

       x'_j | x ~ N(rho_j x_j, lambda_j (1 - rho_j)),
       rho_j = s lambda_j / (1 + s lambda_j) = lambda_j / (lambda_j + D)

   (`tilt_coefficient`), so x'_j = rho_j x_j + sqrt(1 - rho_j) z_j with
   z ~ N(0, Lambda) independent of x (`partner`). The anchor x keeps its
   marginal N(0, lambda_j), since queries are typical items. The neighbour's
   variance is lambda_j q_j with q_j = 1 - rho_j + rho_j^2 <= 1 (neighbours
   lie closer to the centre), and its correlation with x is
   rho_j / sqrt(q_j) (`neighbour_correlation`). D is a water level. The
   difference x'_j - x_j has variance
   lambda_j D (lambda_j + 2 D) / (lambda_j + D)^2. That is about D for
   lambda_j >> D, where neighbours share the input, and 2 lambda_j for
   lambda_j << D, where they do not: the reverse-water-filling shape. For an
   isotropic Lambda the correlation is the same for every input and equals
   cos theta. D is set so that the median angle between the row-centred
   model vectors equals theta_data. That uses the angle convention of E3
   (`collision`): 2 arcsin(|u - v| / 2) on unit rows. The angle is
   increasing in D, and dtheta/dD does not depend on f. Maximising
   -dE[O]/dtheta is therefore the same as maximising -dE[O]/dD. The frozen
   run checks the model against the synthetic data ("pair_model_check" in
   the results). It compares the per-input correlation and variance ratio of
   200-NN pairs, in raw coordinates, with rho_j / sqrt(q_j) and q_j.

2. The mean-field collision law (analytic), and why it cannot be optimised
   directly. Treat the drives as independent Gaussians. Cell i's drive has
   variance v_i = sum_{S_i} lambda_j for x and w_i = sum_{S_i} lambda_j q_j
   for x'. Their covariance is u_i = sum_{S_i} lambda_j rho_j and their
   correlation is r_i = u_i / sqrt(v_i w_i). Take the order-statistic
   thresholds tau and tau' from sum_i Phi_bar(tau / sqrt(v_i)) = k and
   sum_i Phi_bar(tau' / sqrt(w_i)) = k (large-m limit). Then

       P(i in T(x) & T(x')) = Phi_bar_2(a_i, b_i; r_i),
       a_i = tau / sqrt(v_i),  b_i = tau' / sqrt(w_i).

   a_i does not depend on D. Three results give dE[O]/dD in closed form:
     - Plackett's identity, d Phi_bar_2 / dr = phi_2(a, b; r);
     - d Phi_bar_2 / db = -phi(b) Phi_bar((a - r b) / sqrt(1 - r^2));
     - d rho_j / dD = -lambda_j / (lambda_j + D)^2, with dq_j/dD =
       (2 rho_j - 1) d rho_j/dD and the implicit derivative of the tau'
       equation.
   Then

       -dE[O]/dD = -(1/k) sum_i [phi_2(a_i, b_i; r_i) dr_i/dD
                                 + d Phi_bar_2/db (a_i, b_i; r_i) db_i/dD]

   (`meanfield_slope`, checked against numerical differentiation). Two
   consequences follow.
   (a) Isotropic spectrum. Whatever f is, v_i = c_i lambda,
       w_i = c_i lambda q, b_i = a_i, and every r_i = rho / sqrt(q) =
       cos theta. The mean-field objective does not depend on f, and f* is
       even by symmetry (derivation 4). In the exact model, where drives are
       correlated through shared inputs, exploratory runs put even fan-out at
       or above every uneven alternative tried, for k = 4, 16 and 94. Those
       runs used the first freeze's pair model (see "Re-freeze"). The
       alternatives were fan-out concentrated on 40, 30, 20 or 12 inputs, a
       Dirichlet(5) allocation and a linear 1:3 ramp. The frozen run repeats
       this check under the present model ("isotropic_check").
   (b) Degeneracy. Maximising the mean-field objective freely over f is an
       exploratory computation, not part of the frozen run. It was done with
       the first freeze's pair model. The degeneracy comes from the
       independence assumption, not from the pair model. It used 51 free
       conditional-Poisson weights and importance-weighted Monte Carlo over
       input sets; for beta = 1, k = 16 it gave
       f = [1854, 1854, 1753, 4, 4, 1, ..., 834, 1080, 1226]. The highest-
       variance inputs go on about 98% of cells, and the remaining input
       slots go to the lowest-variance inputs. The cells become near copies
       of each other, which the independence assumption still counts as
       independent. An input on every cell shifts every drive equally, so
       top-k cannot see it. With the drives centred across cells, the same
       optimisation parks the spare slots on "dummy" inputs carried by about
       m cells. The exact model without the cap of derivation 3 has the same
       failure: along the power family its slope was largest for
       complement-coded allocations, with the top inputs on about 97% of
       cells. So the optimisation uses the exact joint distribution of the
       drives (Monte Carlo of the model, derivation 5), and it needs the
       constraint of derivation 3.

3. Visibility cap (analytic). Top-k does not change when a constant is added
   to every drive. Input j therefore reaches the ranking only through the
   across-cell-centred column M_j - (f_j / m) 1, whose energy is
   x_j^2 f_j (1 - f_j / m). That energy is largest at f_j = m/2 and falls
   beyond it: connections past m/2 make the input less visible to top-k.
   Hence 0 <= f_j <= floor(m / 2) = 943. Feasibility of the margins is the
   Gale-Ryser condition: the prefix sums of f sorted in decreasing order are
   at most sum_i min(c_i, r).

4. Symmetry. Inputs with equal lambda are exchangeable in the model, so f*
   is a function of lambda_j alone. For an isotropic spectrum it is even
   fan-out, with CV about 0.002 after rounding 9967 / 51.

5. Numerical solution. The search runs over the one-parameter family

       f_j(alpha) = min(cap, s lambda_j^alpha),  s set by sum_j f_j = nnz,

   projected onto the Gale-Ryser set and rounded to integers that follow the
   order of lambda (`power_allocation`). alpha runs from -2 to 6 plus the
   limit alpha = inf (greedy: fill the highest-variance inputs to the cap in
   turn); see ALPHAS. This is the family of the pre-registered grid, so f*
   is also a grid member with exponent alpha*. For each distinct allocation,
   S(f) is the mean over 8 wirings of the central-difference slope on 20,000
   model pairs. Common random numbers (the same pairs, wiring seeds and
   priorities) are shared across allocations. f*(Lambda, k) is the
   allocation with the largest mean slope.

   Exploratory scans (synthetic spectra, 6 wirings, before this module, with
   the first freeze's pair model) showed where the exact-model slope is
   largest:
     beta = 0.5            rises with alpha up to the greedy limit, at every k
     beta = 1, k = 16, 94  rises with alpha up to the greedy limit
     beta = 1, k = 4       flat within noise
     beta = 2, k = 4, 16   peaks near alpha = 0 on a rough curve
     beta = 2, k = 94      rises with alpha
   The collision-slope criterion therefore mostly prefers to concentrate
   fan-out on the highest-variance inputs, up to the cap. It measures only
   the change in mean overlap, not how finely a code separates items, so
   coarser codes are not penalised. This is a property of the pre-registered
   criterion, reported here before any AP is seen.


Unspecified details fixed before running
----------------------------------------
 1. Pair model and water level as in derivation 1. D is matched on the median
    angle of 20,000 row-centred model pairs. The slope is a central difference
    over theta +- 2.5 deg (HALF_WIDTH_DEG), in overlap fraction per degree.
 2. theta_data (`median_nn_angle`): every item is an anchor. Its neighbours
    are the round(0.02 n) nearest items by Euclidean distance on the raw
    input, as in the ground truth; that is 200 at n = 10,000. The angle is
    taken between the row-centred vectors, and theta is the median over all
    (item, neighbour) pairs.
 3. Lambda. Stage 1 uses the nominal lambda_j = j^-beta. For real data, Lambda
    is the per-dimension variance of the raw training inputs in the hashed
    coordinates (`spectrum`); the model applies row-centring itself.
 4. Stage-1 data: `synthetic(beta)` = N(0, diag(j^-beta)), n = 10,000,
    d = 51, numpy default_rng(0). There is one data set per beta, shared by
    all trials, and theta is computed from it.
 5. The wiring ensemble, tie rule and objective are as in "The model". The
    Monte Carlo sizes and seeds are N_PAIRS, N_WIRINGS and SEED.
 6. The search family and grid are ALPHAS (derivation 5). The cap is
    floor(m / 2) (derivation 3). Rounding is largest remainder under the cap
    and the Gale-Ryser bounds, then the values are re-sorted to follow
    lambda. Several alpha can give the same integer allocation (always for an
    isotropic spectrum); it is evaluated once, and alpha* is the one of
    smallest |alpha|, with the positive one chosen on a tie.
 7. The grid allocations of stages 1 and 2 are `power_allocation(lam, alpha,
    c)`, the same function, cap and rounding. f* is then the grid member
    alpha*, and "f* inside the best-grid-alpha interval" compares alpha* with
    that interval. alpha = inf is the greedy member.
 8. For real data, f* is computed from the training split only, by `fstar`
    with the code whose hash `verify_frozen()` checks.
 9. nnz must equal sum(inputs_per_cell): the margins fix it.


Deviations
----------
 1. The pre-registration puts the freeze record ("code plus SHA-256") in
    UPDATES.md. This task does not edit UPDATES.md. The SHA-256 of this file
    and the UTC time of the freeze are in results/e4_frozen.json, to be
    copied into UPDATES.md.
 2. f* is derived within the capped one-parameter family of derivation 5,
    not over all allocations. Reason: the unconstrained problem has no
    usable optimum under any Gaussian-drive variant (derivation 2b). The
    exact-model objective is a Monte Carlo estimate too rough for a
    51-dimensional search, and the family is the pre-registered grid's. The
    cap m/2 is added to the pre-registered model for the reason given in
    derivation 3.
 3. The pre-registered wording gives each cell's drive variance as
    sum_{S_i} lambda_j. The model keeps that variance but also keeps the
    correlation between cells that share inputs, and it applies the
    protocol's row-centring. Without the correlation the model is degenerate
    (derivation 2b).
 4. Stage 0 was frozen twice, both times before any AP run. The first freeze
    had an error in derivation 1 (see "Re-freeze"). The pre-registration
    requires the freeze to precede any new AP run, and that still holds.


Re-freeze
---------
Re-frozen before any AP run. The first freeze (spectral.py sha256
72e35ec418e6be2e9c86cd72a68a1fac467de6ea7d2c804fcdc25722652f3be0, frozen
2026-09-28T18:12:15Z) derived its pair model wrongly. Its premise was
x' ~ N(0, Lambda) conditioned on a small |x - x'|, which gives
x'_j | x ~ N(rho_j x_j, lambda_j (1 - rho_j)). It used instead:
  - the symmetric joint tilt of both items, with precision
    [[1/lambda + s, -s], [-s, 1/lambda + s]], which also tilts the anchor
    x and drops the normaliser that depends on x;
  - and then reset the neighbour's marginal to N(0, lambda_j):
    x'_j = rho_j x_j + sqrt(1 - rho_j^2) z_j, with correlation rho_j.
An adversarial review found this before any AP was computed. The review
checked both models against the synthetic data's 200-NN pairs (first 2,000
anchors, raw coordinates, D matched to theta_data in each model):
  - The rms error of the per-input correlation was 0.016 / 0.022 / 0.026
    for the first model and 0.007 / 0.005 / 0.004 for the corrected one, at
    beta = 0.5 / 1 / 2.
  - The data neighbours' variance on inputs 1 to 10 is 0.69 to 0.88
    lambda_j. The corrected model gives 0.75 to 0.95; the first model has
    1.
The mean-field law (derivation 2) is updated to match. The pair-model check
now compares raw rather than row-centred coordinates, as derivation 1
states it, and it also reports the variance ratio. Nothing else changed:
the objective, the family and grid, the cap, the Monte Carlo sizes and
seeds, the wiring ensemble and the tie rule are those of the first freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from . import flyhash as fh
from . import replication as rp
from .config import ROOT

NNZ = 9967                  # MaleCNS right hemisphere, all glomeruli, all connections
KS = (4, 16, 94)
BETAS = (0.0, 0.5, 1.0, 1.5, 2.0)
D_SYNTH = 51
N_SYNTH = 10_000
N_PAIRS = 20_000
N_WIRINGS = 8
HALF_WIDTH_DEG = 2.5
SWEEPS = 30
SEED = 0
ALPHAS = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
          1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, math.inf)
CHUNK = 4000
RESULTS = ROOT / "results" / "e4_frozen.json"
REFREEZE = {
    "note": "re-frozen before any AP run: the first freeze's neighbour pair model "
            "(derivation 1) did not follow from its premise; it used the symmetric joint "
            "tilt with the neighbour's marginal reset to N(0, lambda), correlation rho. "
            "Corrected to the conditional tilt x' | x ~ N(rho x, lambda (1 - rho)), which "
            "fits the synthetic 200-NN pairs' per-input correlation 2-7 times better (rms "
            "error; see the module docstring, 'Re-freeze'); the mean-field law and the "
            "pair-model check were updated to match; nothing else changed.",
    "supersedes": {"sha256": "72e35ec418e6be2e9c86cd72a68a1fac467de6ea7d2c804fcdc25722652f3be0",
                   "frozen_utc": "2026-09-28T18:12:15Z"},
}


# ---------------------------------------------------------------- data side

def spectrum(x: np.ndarray) -> np.ndarray:
    """Lambda for real data: per-dimension variance of the raw training inputs
    (in the coordinates the benchmark hashes, e.g. PCA-51)."""
    return np.asarray(x, float).var(axis=0)


def synthetic(beta: float, n: int = N_SYNTH, d: int = D_SYNTH,
              seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stage-1 inputs: n draws of N(0, diag(j^-beta)), j = 1..d, and lambda."""
    lam = np.arange(1, d + 1, dtype=float) ** -float(beta)
    x = np.random.default_rng(seed).normal(size=(n, d)) * np.sqrt(lam)
    return x, lam


def _unit_centred(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = rp.centre(np.asarray(x, float))
    nrm = np.linalg.norm(xc, axis=-1, keepdims=True)
    return np.divide(xc, nrm, out=np.zeros_like(xc), where=nrm > 0), nrm[..., 0] > 0


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit rows, 2 arcsin(|u - v| / 2) (accurate at small angles)."""
    return np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(u - v, axis=-1) / 2, 0.0, 1.0)))


def nn_pairs(x: np.ndarray, top: int | None = None, chunk: int = 1000):
    """For every item, its `top` nearest items by Euclidean distance on the raw
    input (the ground-truth rule); top defaults to round(0.02 n)."""
    x = np.asarray(x, float)
    n = len(x)
    top = top or max(1, int(round(rp.TOP * n)))
    sq = (x * x).sum(1)
    for i in range(0, n, chunk):
        q = np.arange(i, min(i + chunk, n))
        dist = sq[q][:, None] + sq[None, :] - 2 * x[q] @ x.T
        dist[np.arange(len(q)), q] = np.inf
        yield q, np.argpartition(dist, top - 1, axis=1)[:, :top]


def median_nn_angle(x: np.ndarray, top: int | None = None) -> float:
    """theta_data: median angle (deg) between row-centred items and their
    Euclidean nearest neighbours (see "Unspecified details" 2)."""
    u, ok = _unit_centred(x)
    out = []
    for q, nb in nn_pairs(x, top):
        a = _angle_deg(u[q][:, None, :], u[nb])
        out.append(a[ok[q][:, None] & ok[nb]])
    return float(np.median(np.concatenate(out)))


# ---------------------------------------------------------------- pair model

def tilt_coefficient(lam: np.ndarray, D: float) -> np.ndarray:
    """rho_j = lambda_j / (lambda_j + D): E[x'_j | x] = rho_j x_j (derivation 1)."""
    lam = np.asarray(lam, float)
    return lam / (lam + D)


def neighbour_variance(lam: np.ndarray, D: float) -> np.ndarray:
    """q_j = Var(x'_j) / lambda_j = 1 - rho_j + rho_j^2 (derivation 1)."""
    rho = tilt_coefficient(lam, D)
    return 1.0 - rho + rho ** 2


def neighbour_correlation(lam: np.ndarray, D: float) -> np.ndarray:
    """corr(x_j, x'_j) = rho_j / sqrt(q_j) (derivation 1)."""
    return tilt_coefficient(lam, D) / np.sqrt(neighbour_variance(lam, D))


def partner(x: np.ndarray, z: np.ndarray, lam: np.ndarray, D: float) -> np.ndarray:
    """x' = rho x + sqrt(1 - rho) z, z ~ N(0, Lambda): x' | x ~ N(rho x, Lambda (1 - rho))."""
    rho = tilt_coefficient(lam, D)
    return rho * x + np.sqrt(1.0 - rho) * z


def model_draws(lam: np.ndarray, n: int = N_PAIRS, seed: int = SEED):
    """The (x, z) Gaussian draws shared by every allocation (common random numbers)."""
    rng = np.random.default_rng([seed, 17])
    s = np.sqrt(np.asarray(lam, float))
    return rng.normal(size=(n, len(s))) * s, rng.normal(size=(n, len(s))) * s


def model_median_angle(lam, D: float, x: np.ndarray, z: np.ndarray) -> float:
    u, _ = _unit_centred(x)
    v, _ = _unit_centred(partner(x, z, lam, D))
    return float(np.median(_angle_deg(u, v)))


def water_level(lam, theta: float, x: np.ndarray, z: np.ndarray) -> float:
    """D such that the median angle of the row-centred model pairs is theta (deg)."""
    lam = np.asarray(lam, float)
    if not 0.0 < theta < 90.0:
        raise ValueError(f"theta must lie in (0, 90) deg, got {theta}")
    c = math.log(lam.max())
    L = brentq(lambda L: model_median_angle(lam, math.exp(L), x, z) - theta,
               c - 60.0, c + 60.0, xtol=1e-7)
    return math.exp(L)


# ---------------------------------------------------------------- allocations

def gale_ryser_bounds(inputs_per_cell, d: int) -> np.ndarray:
    """g[r] = sum_i min(c_i, r), r = 0..d: the largest total fan-out any r inputs can have."""
    c = np.asarray(inputs_per_cell, int)
    return np.array([np.minimum(c, r).sum() for r in range(d + 1)], float)


def gale_ryser_ok(f, inputs_per_cell) -> bool:
    """Is there a binary matrix with row sums f and column sums c?"""
    f = np.asarray(f)
    c = np.asarray(inputs_per_cell, int)
    if (f < 0).any() or f.sum() != c.sum() or f.max(initial=0) > len(c):
        return False
    g = gale_ryser_bounds(c, len(f))
    return bool((np.cumsum(np.sort(f)[::-1]) <= g[1:] + 1e-9).all())


def _fill(t: np.ndarray, amount: float, logw: np.ndarray | None, cap: float) -> np.ndarray:
    """Add `amount` to t (sorted by decreasing lambda), each entry capped at `cap`:
    in proportion to exp(logw) (water-filling), or greedily in order if logw is None."""
    t = t.copy()
    room = cap - t
    if amount <= 0:
        return t
    if room.sum() < amount - 1e-9:
        raise ValueError("fan-out exceeds the cap on every input")
    if logw is None:
        for j in range(len(t)):
            add = min(room[j], amount)
            t[j] += add
            amount -= add
            if amount <= 1e-12:
                break
        return t
    lw = logw - logw.max()

    def added(L):
        return np.minimum(room, np.exp(np.minimum(L + lw, math.log(cap) + 1.0))).sum()

    hi = math.log(cap) + 1.0 - lw.min()
    lo = math.log(amount / len(t)) - 1.0
    L = brentq(lambda L: added(L) - amount, lo, hi, xtol=1e-12)
    return t + np.minimum(room, np.exp(np.minimum(L + lw, math.log(cap) + 1.0)))


def _project(t: np.ndarray, logw: np.ndarray | None, g: np.ndarray, cap: float) -> np.ndarray:
    """Move fan-out from over-full top blocks to the inputs below them until every
    prefix sum satisfies the Gale-Ryser bound (t sorted by decreasing lambda)."""
    t = t.copy()
    for _ in range(10 * len(t)):
        P = np.cumsum(np.sort(t)[::-1])
        viol = np.flatnonzero(P > g[1:] + 1e-7)
        if not len(viol):
            return t
        r = viol[0] + 1
        order = np.argsort(-t, kind="stable")
        top, rest = order[:r], np.sort(order[r:])
        excess = P[r - 1] - g[r]
        t[top] *= g[r] / P[r - 1]
        t[rest] = _fill(t[rest], excess, None if logw is None else logw[rest], cap)
    raise RuntimeError("Gale-Ryser projection did not converge")


def _round(t: np.ndarray, c: np.ndarray, cap: int) -> np.ndarray:
    """Integers summing to nnz, at most cap, Gale-Ryser feasible; largest
    remainder, then re-sorted so that the integers follow the order of t
    (the largest integer where t is largest; ties in t by position)."""
    nnz = int(c.sum())
    g = gale_ryser_bounds(c, len(t))
    f = np.minimum(np.floor(t + 1e-9).astype(int), cap)
    frac = t - f
    order = np.argsort(-frac, kind="stable")
    need = nnz - int(f.sum())
    while need > 0:
        given = 0
        for j in order:                      # one unit per input per pass
            if need == 0:
                break
            if f[j] >= cap:
                continue
            f[j] += 1
            if (np.cumsum(np.sort(f)[::-1]) <= g[1:] + 1e-9).all():
                need -= 1
                given += 1
            else:
                f[j] -= 1
        if not given:
            raise RuntimeError("no feasible integer rounding")
    ranks = np.argsort(np.argsort(-t, kind="stable"), kind="stable")
    return np.sort(f)[::-1][ranks]


def power_allocation(lam, alpha: float, inputs_per_cell, cap: int | None = None) -> np.ndarray:
    """f_j = min(cap, s lambda_j^alpha) with sum f = nnz, projected onto the
    Gale-Ryser set, rounded, and ordered like lambda (non-decreasing in lambda
    for alpha > 0, non-increasing for alpha < 0, even for alpha = 0 or equal
    lambdas). alpha = inf fills the highest-variance inputs to the cap in turn.
    The default cap is floor(m / 2) (derivation 3)."""
    lam = np.asarray(lam, float)
    c = np.asarray(inputs_per_cell, int)
    d, m, nnz = len(lam), len(c), int(c.sum())
    cap = m // 2 if cap is None else int(cap)
    if cap * d < nnz:
        raise ValueError(f"cap {cap} x {d} inputs cannot hold nnz = {nnz}")
    if (lam < 0).any() or not np.isfinite(lam).all():
        raise ValueError("lambda must be finite and non-negative")
    order = np.argsort(-lam, kind="stable")          # highest variance first
    ls = np.maximum(lam[order], 1e-300)
    iso = ls.max() == ls.min()
    if iso or alpha == 0:
        logw = np.zeros(d)
    elif alpha == math.inf:
        logw = None
    elif alpha == -math.inf:
        raise ValueError("alpha = -inf is not part of the family")
    else:
        logw = float(alpha) * (np.log(ls) - np.log(ls.max()))
    if logw is None:
        # greedy, but inputs of equal lambda share alike
        t = np.zeros(d)
        left = float(nnz)
        for v in np.unique(ls)[::-1]:
            grp = np.flatnonzero(ls == v)
            add = min(left, cap * len(grp))
            t[grp] = add / len(grp)
            left -= add
            if left <= 1e-12:
                break
    else:
        t = _fill(np.zeros(d), float(nnz), logw, cap)
    t = _project(t, logw, gale_ryser_bounds(c, d), cap)
    f = np.empty(d, int)
    f[order] = _round(t, c, cap)
    return f


def realise(inputs_per_cell, fanout, seed, sweeps: int = SWEEPS) -> np.ndarray:
    """A binary d x m matrix with row sums `fanout` and column sums
    `inputs_per_cell`, uniform over such matrices: greedy realisation (cells
    with most inputs first, each taking the inputs with most remaining
    capacity) then curveball, as in `flyhash.margin_control`."""
    c = np.asarray(inputs_per_cell, int)
    f = np.asarray(fanout, int)
    if not gale_ryser_ok(f, c):
        raise ValueError("margins are not realisable (Gale-Ryser)")
    rng = np.random.default_rng(seed)
    d, m = len(f), len(c)
    cap = f.copy()
    out = np.zeros((d, m))
    for i in np.argsort(-c, kind="stable"):
        chosen = np.lexsort((rng.random(d), -cap))[:c[i]]
        if (cap[chosen] <= 0).any():
            raise RuntimeError("greedy realisation failed")
        out[chosen, i] = 1.0
        cap[chosen] -= 1
    q = fh.curveball(fh.Projection("e4", out, [str(j) for j in range(d)]),
                     seed=int(rng.integers(2**31)), sweeps=sweeps)
    return (q.matrix > 0).astype(float)


# ---------------------------------------------------------------- the model's slope

class Setup:
    """Model pairs at theta -+ half-width for one spectrum (shared across allocations)."""

    def __init__(self, lam, theta: float, pairs: int = N_PAIRS, seed: int = SEED,
                 half_width: float = HALF_WIDTH_DEG):
        self.lam = np.asarray(lam, float)
        self.theta, self.half_width, self.seed = float(theta), float(half_width), seed
        x, z = model_draws(self.lam, pairs, seed)
        self.D = water_level(self.lam, theta, x, z)
        self.D_minus = water_level(self.lam, theta - half_width, x, z)
        self.D_plus = water_level(self.lam, theta + half_width, x, z)
        self.x = rp.centre(x)
        self.x_minus = rp.centre(partner(x, z, self.lam, self.D_minus))
        self.x_plus = rp.centre(partner(x, z, self.lam, self.D_plus))

    def describe(self) -> dict:
        return {"theta_deg": self.theta, "half_width_deg": self.half_width,
                "D": self.D, "D_minus": self.D_minus, "D_plus": self.D_plus,
                "pairs": int(len(self.x)), "seed": self.seed}


def _top_sorted(y: np.ndarray, K: int) -> np.ndarray:
    idx = np.argpartition(-y, K - 1, axis=1)[:, :K]
    o = np.argsort(-np.take_along_axis(y, idx, 1), axis=1, kind="stable")
    return np.take_along_axis(idx, o, 1)


def _overlaps(M: np.ndarray, S: Setup, ks, prio: np.ndarray) -> dict:
    """Mean overlap fraction at theta - h and theta + h, for every k, on wiring M."""
    U, inv = np.unique(M.T, axis=0, return_inverse=True)   # identical cells get identical drives
    inv = np.asarray(inv).ravel()
    Mu = U.T
    K = max(ks)
    tot = {k: np.zeros(2) for k in ks}
    n = len(S.x)
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        tops = []
        for xx in (S.x[sl], S.x_minus[sl], S.x_plus[sl]):
            y = (xx @ Mu)[:, inv]
            scale = np.abs(y).max(1, keepdims=True)
            tops.append(_top_sorted(y + 1e-9 * scale * prio[None, :], K))
        for k in ks:
            A = np.zeros((tops[0].shape[0], M.shape[1]), bool)
            np.put_along_axis(A, tops[0][:, :k], True, 1)
            for s, t in enumerate(tops[1:]):
                tot[k][s] += np.take_along_axis(A, t[:, :k], 1).sum() / k
    return {k: v / n for k, v in tot.items()}


def meanfield_slope(M: np.ndarray, lam, k: int, D: float) -> float:
    """-dE[O]/dD of derivation 2 (independent drives, order-statistic thresholds
    for the item and its neighbour)."""
    lam = np.asarray(lam, float)
    rho = tilt_coefficient(lam, D)
    drho = -lam / (lam + D) ** 2
    v = lam @ M
    keep = v > 0
    v = v[keep]
    w = ((lam * neighbour_variance(lam, D)) @ M)[keep]
    u = ((lam * rho) @ M)[keep]
    du = ((lam * drho) @ M)[keep]
    dw = ((lam * (2 * rho - 1) * drho) @ M)[keep]
    r = np.clip(u / np.sqrt(v * w), -1.0, 1.0 - 1e-12)
    dr = r * (du / u - dw / (2 * w))
    sv, sw = np.sqrt(v), np.sqrt(w)
    tau = brentq(lambda t: norm.sf(t / sv).sum() - k, -100 * sv.max(), 100 * sv.max())
    tau2 = brentq(lambda t: norm.sf(t / sw).sum() - k, -100 * sw.max(), 100 * sw.max())
    a, b = tau / sv, tau2 / sw
    pb = norm.pdf(b)
    dtau2 = (pb * b * dw / (2 * w)).sum() / (pb / sw).sum()
    db = dtau2 / sw - b * dw / (2 * w)
    s = np.sqrt(1 - r ** 2)
    phi2 = np.exp(-(a * a - 2 * r * a * b + b * b) / (2 * s * s)) / (2 * np.pi * s)
    dP_db = -pb * norm.sf((a - r * b) / s)
    return float(-(phi2 * dr + dP_db * db).sum() / k)


def model_slope(S: Setup, fanout, inputs_per_cell, ks=KS, wirings: int = N_WIRINGS,
                seed: int = SEED, meanfield: bool = True) -> dict:
    """S(f) per k: mean and SE over wirings of the central-difference slope
    (overlap fraction per degree), plus the mean overlaps at theta -+ h and,
    for reference, the mean-field slope on the same wirings (per degree,
    converted with the model's dD/dtheta)."""
    c = np.asarray(inputs_per_cell, int)
    ks = tuple(int(k) for k in ks)
    per = {k: [] for k in ks}
    om = {k: [] for k in ks}
    op = {k: [] for k in ks}
    mf = {k: [] for k in ks}
    dD_dtheta = (S.D_plus - S.D_minus) / (2 * S.half_width)
    for w in range(wirings):
        M = realise(c, fanout, seed=[seed, 1000 + w])
        prio = np.random.default_rng([seed, 2000 + w]).random(M.shape[1])
        o = _overlaps(M, S, ks, prio)
        for k in ks:
            om[k].append(o[k][0])
            op[k].append(o[k][1])
            per[k].append((o[k][0] - o[k][1]) / (2 * S.half_width))
            if meanfield:
                mf[k].append(meanfield_slope(M, S.lam, k, S.D) * dD_dtheta)
    out = {}
    for k in ks:
        v = np.array(per[k])
        out[k] = {"slope": float(v.mean()),
                  "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                  "per_wiring": v.tolist(),
                  "overlap_minus": float(np.mean(om[k])), "overlap_plus": float(np.mean(op[k])),
                  "meanfield_slope": float(np.mean(mf[k])) if meanfield else None}
    return out


def _key(f: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(f, np.int64).tobytes()).hexdigest()[:16]


def _alpha_label(a: float) -> str:
    return "inf" if a == math.inf else f"{a:g}"


def fstar_all(lam, ks, inputs_per_cell, nnz: int, theta: float, *, alphas=ALPHAS,
              pairs: int = N_PAIRS, wirings: int = N_WIRINGS, seed: int = SEED,
              half_width: float = HALF_WIDTH_DEG, cap: int | None = None,
              cache: dict | None = None, on_eval=None, log=None) -> tuple[dict, dict]:
    """f*(Lambda, k) for every k in `ks` from one scan of the family (derivation 5).

    Returns ({k: f*}, info). `cache` maps allocation keys to earlier
    evaluations (checkpoint and resume). `on_eval(key, record)` is called
    after each new evaluation."""
    lam = np.asarray(lam, float)
    c = np.asarray(inputs_per_cell, int)
    if int(nnz) != int(c.sum()):
        raise ValueError(f"nnz = {nnz} but the inputs per cell sum to {int(c.sum())}")
    ks = tuple(int(k) for k in ks)
    if max(ks) >= len(c):
        raise ValueError("k must be smaller than the number of cells")
    cache = {} if cache is None else cache
    allocs = {}
    for a in alphas:
        f = power_allocation(lam, a, c, cap)
        allocs[_alpha_label(a)] = (a, f, _key(f))
    S = Setup(lam, theta, pairs, seed, half_width)
    for label, (a, f, key) in allocs.items():
        if key in cache and all(str(k) in cache[key]["k"] for k in ks):
            continue
        t0 = time.time()
        r = model_slope(S, f, c, ks, wirings, seed)
        rec = {"fanout": f.tolist(), "alphas": [], "seconds": time.time() - t0,
               "k": {str(k): v for k, v in r.items()}}
        cache[key] = rec
        if on_eval:
            on_eval(key, rec)
        if log:
            log(f"    alpha {label:>5}: " + "  ".join(
                f"k={k} {100 * r[k]['slope']:.3f}%/deg" for k in ks) + f"  ({rec['seconds']:.0f}s)")
    for key in {v[2] for v in allocs.values()}:
        cache[key]["alphas"] = sorted({lbl for lbl, v in allocs.items() if v[2] == key},
                                      key=lambda s: float(s))
    out, info = {}, {"setup": S.describe(), "curve": {}, "k": {}}
    for label, (a, f, key) in allocs.items():
        info["curve"][label] = {"key": key, **{str(k): cache[key]["k"][str(k)]["slope"] for k in ks}}
    for k in ks:
        best = max(cache[v[2]]["k"][str(k)]["slope"] for v in allocs.values())
        cands = [v for v in allocs.values() if cache[v[2]]["k"][str(k)]["slope"] == best]
        a, f, key = min(cands, key=lambda v: (abs(v[0]), -v[0]))
        out[k] = f.copy()
        info["k"][str(k)] = {"alpha_star": _alpha_label(a), "key": key, "slope": best,
                             "se": cache[key]["k"][str(k)]["se"],
                             "cv": float(f.std() / f.mean()),
                             "alphas_with_same_allocation": cache[key]["alphas"]}
    return out, info


def fstar(lam, k: int, inputs_per_cell, nnz: int, theta: float, *, return_info: bool = False,
          **kw):
    """The fan-out allocation f*(Lambda, k): non-negative integers summing to
    nnz, one per input dimension, aligned with `lam`. `theta` is the median
    nearest-neighbour angle in degrees (`median_nn_angle`)."""
    out, info = fstar_all(lam, (k,), inputs_per_cell, nnz, theta, **kw)
    return (out[int(k)], info) if return_info else out[int(k)]


# ---------------------------------------------------------------- freezing

def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_frozen(path: Path | str = RESULTS) -> dict:
    """Load the frozen record and check that this file is the frozen code."""
    res = json.loads(Path(path).read_text())
    if not res.get("complete"):
        raise RuntimeError(f"{path} is not a complete freeze")
    if res["code"]["sha256"] != code_sha256():
        raise RuntimeError("flypath/spectral.py differs from the frozen code "
                           f"({res['code']['sha256'][:12]} frozen, {code_sha256()[:12]} now)")
    return res


def frozen_fstar(beta: float, k: int, path: Path | str = RESULTS) -> np.ndarray:
    """The frozen stage-1 f* for a synthetic spectrum (beta in BETAS)."""
    res = verify_frozen(path)
    return np.array(res["synthetic"][f"{float(beta):g}"]["f_star"][str(int(k))], int)


DESIGN = {"betas": list(BETAS), "ks": list(KS), "alphas": [_alpha_label(a) for a in ALPHAS],
          "pairs": N_PAIRS, "wirings": N_WIRINGS, "half_width_deg": HALF_WIDTH_DEG,
          "seed": SEED, "n_synthetic": N_SYNTH, "d_synthetic": D_SYNTH, "sweeps": SWEEPS,
          "cap": "floor(m/2)", "nnz": NNZ, "inputs_per_cell": "MaleCNS R, experiments.context(cfg).full"}
SMOKE = {**DESIGN, "betas": [0.0, 1.0], "ks": [4, 16], "alphas": ["0", "1", "inf"],
         "pairs": 1000, "wirings": 2, "n_synthetic": 2000}


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
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    os.replace(tmp, path)


def _pair_model_check(x: np.ndarray, lam: np.ndarray, theta: float, anchors: int = 2000) -> dict:
    """Per-input correlation and neighbour variance ratio of the 200-NN pairs
    of the first `anchors` items (raw coordinates, as in derivation 1) against
    the model at the matched water level."""
    q, nb = next(nn_pairs(x, None, chunk=anchors))
    a = np.repeat(x[q], nb.shape[1], axis=0)
    b = x[nb.ravel()]
    corr = (a * b).mean(0) / np.sqrt((a * a).mean(0) * (b * b).mean(0))
    xs, zs = model_draws(lam, N_PAIRS, SEED)
    D = water_level(lam, theta, xs, zs)
    mc = neighbour_correlation(lam, D)
    return {"anchors": int(len(q)), "D": D, "nn_correlation": corr.tolist(),
            "model_correlation": mc.tolist(),
            "rms_correlation_error": float(np.sqrt(((corr - mc) ** 2).mean())),
            "nn_variance_ratio": ((b * b).mean(0) / lam).tolist(),
            "model_variance_ratio": neighbour_variance(lam, D).tolist()}


def _isotropic_check(c: np.ndarray, theta: float, D_: dict, log) -> dict:
    """Exact-model slope at an isotropic spectrum: even fan-out against uneven ones."""
    d = D_["d_synthetic"]
    nnz = int(c.sum())
    cap = len(c) // 2
    g = gale_ryser_bounds(c, d)
    lam = np.ones(d)
    rng = np.random.default_rng([D_["seed"], 31])
    targets = {"even": np.full(d, nnz / d)}
    for used in (40, 30, 20, 12):
        t = np.full(d, 2.0)
        t[:used] = (nnz - 2.0 * (d - used)) / used
        targets[f"{used} inputs"] = t
    targets["dirichlet5"] = rng.dirichlet(np.full(d, 5.0)) * nnz
    ramp = np.linspace(1.0, 3.0, d)
    targets["ramp 1:3"] = ramp / ramp.sum() * nnz
    S = Setup(lam, theta, D_["pairs"], D_["seed"], D_["half_width_deg"])
    out = {}
    for name, t in targets.items():
        o = np.argsort(-t, kind="stable")
        ts = _project(np.minimum(t[o], cap), None, g, cap)
        fi = _round(ts, c, cap)
        f = np.empty(d, int)
        f[o] = fi
        r = model_slope(S, f, c, D_["ks"], D_["wirings"], D_["seed"], meanfield=False)
        out[name] = {"fanout": f.tolist(), "cv": float(f.std() / f.mean()),
                     **{str(k): {"slope": v["slope"], "se": v["se"]} for k, v in r.items()}}
        log(f"  isotropic check {name:>10}: " + "  ".join(
            f"k={k} {100 * v['slope']:.3f}%/deg" for k, v in r.items()))
    return out


def run(out: Path | str = RESULTS, design: dict | None = None,
        budget_minutes: float | None = None, log=print) -> dict:
    """Freeze f* for the stage-1 spectra, resuming from `out` if it exists.
    Stops cleanly after the current allocation once `budget_minutes` have
    passed; call again to continue."""
    from . import config
    from . import experiments as ex
    D_ = json.loads(json.dumps(design or DESIGN))
    out = Path(out)
    deadline = None if budget_minutes is None else time.time() + 60 * budget_minutes
    sha = code_sha256()
    if out.exists():
        res = json.loads(out.read_text())
        if res["design"] != D_:
            raise SystemExit(f"{out} was produced with a different design; refusing to mix")
        if res["code"]["sha256"] != sha:
            raise SystemExit(f"{out} was produced by different code; delete it to restart")
        if res.get("complete"):
            log(f"{out} is already complete")
            return res
        log(f"resuming {out}")
    else:
        res = {"experiment": "E4 stage 0: spectral fan-out allocation f*(Lambda, k)",
               "preregistration": "experiments/PREREGISTRATION_2026-09-28.md, E4 stage 0",
               "note": "model quantities only; no retrieval (AP) was computed",
               "refreeze": REFREEZE,
               "design": D_, "code": {"path": "flypath/spectral.py", "sha256": sha},
               "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "complete": False, "invocations": [], "synthetic": {}}
    res["invocations"].append({"start_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "python": platform.python_version(), "numpy": np.__version__})
    _save(res, out)

    cfg = config.load()
    c = ex.context(cfg).full.inputs().astype(int)
    if int(c.sum()) != D_["nnz"]:
        raise SystemExit(f"MaleCNS R inputs per cell sum to {int(c.sum())}, not {D_['nnz']}")
    res["inputs_per_cell"] = {"m": int(len(c)), "nnz": int(c.sum()), "cap": int(len(c) // 2),
                              "histogram": np.bincount(c).tolist(),
                              "sha1": hashlib.sha1(c.astype(np.int64).tobytes()).hexdigest()}
    alphas = tuple(math.inf if a == "inf" else float(a) for a in D_["alphas"])

    class _Stop(Exception):
        pass

    for beta in D_["betas"]:
        label = f"{float(beta):g}"
        e = res["synthetic"].setdefault(label, {"beta": beta})
        if "f_star" in e:
            continue
        x, lam = synthetic(beta, D_["n_synthetic"], D_["d_synthetic"], D_["seed"])
        if "theta_deg" not in e:
            e["theta_deg"] = median_nn_angle(x)
            e["lambda"] = lam.tolist()
            e["pair_model_check"] = _pair_model_check(x, lam, e["theta_deg"])
            _save(res, out)
        log(f"beta {label}: theta = {e['theta_deg']:.3f} deg")
        cache = e.setdefault("evaluations", {})

        def on_eval(key, rec):
            _save(res, out)
            if deadline is not None and time.time() > deadline:
                raise _Stop

        try:
            fs, info = fstar_all(lam, D_["ks"], c, D_["nnz"], e["theta_deg"], alphas=alphas,
                                 pairs=D_["pairs"], wirings=D_["wirings"], seed=D_["seed"],
                                 half_width=D_["half_width_deg"], cache=cache,
                                 on_eval=on_eval, log=log)
        except _Stop:
            log("  budget reached; call again to continue")
            _save(res, out)
            return res
        e["setup"] = info["setup"]
        e["curve"] = info["curve"]
        e["selection"] = info["k"]
        e["f_star"] = {str(k): f.tolist() for k, f in fs.items()}
        _save(res, out)
        for k in D_["ks"]:
            s = info["k"][str(k)]
            log(f"  k={k}: alpha* = {s['alpha_star']}, f* = {e['f_star'][str(k)][:6]}..., "
                f"CV {s['cv']:.3f}")

    if "isotropic_check" not in res:
        if deadline is not None and time.time() > deadline - 180:
            log("  budget reached before the isotropic check; call again to continue")
            _save(res, out)
            return res
        th0 = res["synthetic"]["0"]["theta_deg"]
        res["isotropic_check"] = _isotropic_check(c, th0, D_, log)
        _save(res, out)
    res["complete"] = True
    res["frozen_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res["code"]["sha256"] = code_sha256()
    _save(res, out)
    log(f"frozen: {out} (spectral.py sha256 {res['code']['sha256'][:12]})")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flypath.spectral",
                                description="E4 stage 0: derive and freeze f*(Lambda, k)")
    p.add_argument("--out", default=str(RESULTS), help="results JSON (checkpoint and resume)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny configuration that only checks the code runs; needs --out")
    p.add_argument("--budget-minutes", type=float, default=None,
                   help="stop cleanly after the current allocation once this many minutes passed")
    args = p.parse_args(argv)
    if args.smoke and Path(args.out).resolve() == RESULTS.resolve():
        p.error("--smoke must not write to results/e4_frozen.json; pass --out")
    res = run(out=args.out, design=SMOKE if args.smoke else DESIGN,
              budget_minutes=args.budget_minutes, log=lambda s: print(s, flush=True))
    return 0 if res.get("complete") else 3


if __name__ == "__main__":
    sys.exit(main())
