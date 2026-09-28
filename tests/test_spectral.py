"""E4 stage 0: the spectral fan-out allocation f*(Lambda, k) (flypath.spectral)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from flypath import replication as rp
from flypath import spectral as sp

FAST = dict(pairs=1500, wirings=2, alphas=(-1.0, 0.0, 0.5, 1.0, 2.0, math.inf))


def _cells(m: int = 240, seed: int = 0) -> np.ndarray:
    """A small inputs-per-cell sequence (1..8 inputs), MaleCNS-like in shape."""
    return np.random.default_rng(seed).integers(1, 9, m)


# ------------------------------------------------------------------ f*

def test_fstar_sums_to_nnz_and_is_realisable():
    c = _cells()
    lam = np.arange(1, 17, dtype=float) ** -1.0
    f = sp.fstar(lam, 4, c, int(c.sum()), 45.0, **FAST)
    assert f.dtype.kind == "i" and f.shape == lam.shape
    assert f.sum() == c.sum()
    assert f.min() >= 0 and f.max() <= len(c) // 2
    assert sp.gale_ryser_ok(f, c)


def test_fstar_rejects_nnz_that_the_margins_cannot_have():
    c = _cells()
    with pytest.raises(ValueError):
        sp.fstar(np.ones(16), 4, c, int(c.sum()) + 1, 45.0, **FAST)


def test_fstar_flat_for_isotropic_spectrum():
    c = _cells(m=1886, seed=1)
    lam = np.full(51, 0.7)
    f, info = sp.fstar(lam, 16, c, int(c.sum()), 70.0, return_info=True, **FAST)
    assert f.sum() == c.sum()
    assert f.std() / f.mean() < 0.05
    assert np.ptp(f) <= 1                              # even up to rounding
    assert info["k"]["16"]["alpha_star"] == "0"        # every alpha gives this allocation


def test_fstar_monotone_in_lambda():
    c = _cells()
    lam = np.random.default_rng(5).permutation(np.arange(1, 17, dtype=float) ** -1.5)
    f = sp.fstar(lam, 16, c, int(c.sum()), 40.0, **FAST)
    fo = np.diff(f[np.argsort(-lam)])
    assert (fo <= 0).all() or (fo >= 0).all()


def test_fstar_is_aligned_with_input_order():
    c = _cells()
    lam = np.arange(1, 17, dtype=float) ** -1.0
    perm = np.random.default_rng(2).permutation(16)
    f = sp.fstar(lam, 4, c, int(c.sum()), 45.0, **FAST)
    fp = sp.fstar(lam[perm], 4, c, int(c.sum()), 45.0, **FAST)
    # same spectrum, permuted: the same fan-out follows each lambda value
    assert (np.sort(f) == np.sort(fp)).all()
    assert (np.diff(fp[np.argsort(-lam[perm])]) <= 0).all() or \
           (np.diff(fp[np.argsort(-lam[perm])]) >= 0).all()


# ------------------------------------------------------------------ the family

@pytest.mark.parametrize("alpha", [-2.0, -1.0, -0.25, 0.25, 1.0, 3.0, math.inf])
def test_power_allocation_monotone_in_lambda(alpha):
    c = _cells()
    lam = np.random.default_rng(3).permutation(np.arange(1, 21, dtype=float) ** -1.5)
    f = sp.power_allocation(lam, alpha, c)
    step = np.diff(f[np.argsort(-lam)])
    if alpha > 0:
        assert (step <= 0).all()          # more variance, at least as much fan-out
    else:
        assert (step >= 0).all()
    assert f.sum() == c.sum() and sp.gale_ryser_ok(f, c) and f.max() <= len(c) // 2


def test_power_allocation_even_at_alpha_zero_and_equal_lambdas_share():
    c = _cells()
    lam = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
    assert np.ptp(sp.power_allocation(lam, 0.0, c)) <= 1
    for alpha in (0.5, 2.0, math.inf):
        f = sp.power_allocation(lam, alpha, c)
        for v in np.unique(lam):
            assert np.ptp(f[lam == v]) <= 1


def test_power_allocation_respects_cap_and_the_largest_cell():
    # one cell with 12 inputs forces at least 12 inputs with fan-out >= 1
    c = np.concatenate([np.full(200, 3), [12]])
    lam = np.arange(1, 16, dtype=float) ** -2.0
    f = sp.power_allocation(lam, math.inf, c)
    assert (f > 0).sum() >= 12 and f.max() <= len(c) // 2 and sp.gale_ryser_ok(f, c)


def test_gale_ryser():
    assert sp.gale_ryser_ok([3, 2, 0], [2, 2, 1])      # input 1 on every cell, input 2 on two
    assert not sp.gale_ryser_ok([4, 1, 0], [2, 2, 1])  # more cells than there are
    assert sp.gale_ryser_ok([2, 2, 1], [3, 1, 1])
    assert not sp.gale_ryser_ok([3, 2, 0], [3, 1, 1])  # a 3-input cell needs 3 used inputs
    assert not sp.gale_ryser_ok([2, 2, 2], [3, 1, 1])  # wrong total


def test_realise_has_exact_margins():
    c = _cells()
    f = sp.power_allocation(np.arange(1, 17, dtype=float) ** -1.0, 1.0, c)
    M = sp.realise(c, f, seed=3, sweeps=5)
    assert (M.sum(0) == c).all() and (M.sum(1) == f).all()
    assert set(np.unique(M)) <= {0.0, 1.0}


# ------------------------------------------------------------------ pair model and angle

def test_water_level_matches_the_angle_and_orders_inputs():
    lam = np.arange(1, 31, dtype=float) ** -1.0
    x, z = sp.model_draws(lam, 4000, 0)
    D = sp.water_level(lam, 50.0, x, z)
    assert abs(sp.model_median_angle(lam, D, x, z) - 50.0) < 1e-3
    assert sp.water_level(lam, 40.0, x, z) < D < sp.water_level(lam, 60.0, x, z)
    rho = sp.tilt_coefficient(lam, D)
    assert np.allclose(rho, lam / (lam + D))
    corr = sp.neighbour_correlation(lam, D)
    assert (np.diff(corr) < 0).all() and np.allclose(corr, rho / np.sqrt(1 - rho + rho ** 2))


def test_pair_model_is_the_conditional_tilt():
    # derivation 1: x' ~ N(0, l) tilted by exp(-(x' - x)^2 / (2 D)) given x is
    # N(rho x, l (1 - rho)); checked by quadrature against the exact conditional
    from scipy.integrate import quad
    from scipy.stats import norm
    for l, D, x0 in ((1.0, 0.3, 1.2), (0.2, 0.5, -0.7), (0.05, 2.0, 0.3)):
        def w(y):
            return norm.pdf(y, 0, math.sqrt(l)) * math.exp(-(y - x0) ** 2 / (2 * D))
        Z = quad(w, -30, 30)[0]
        mu = quad(lambda y: y * w(y), -30, 30)[0] / Z
        var = quad(lambda y: (y - mu) ** 2 * w(y), -30, 30)[0] / Z
        rho = sp.tilt_coefficient(np.array([l]), D)[0]
        assert abs(mu - rho * x0) < 1e-9 and abs(var - l * (1 - rho)) < 1e-9


def test_partner_has_the_model_moments():
    lam = np.array([2.0, 1.0, 0.3, 0.05])
    D = 0.4
    x, z = sp.model_draws(lam, 200_000, 3)
    xp = sp.partner(x, z, lam, D)
    rho = sp.tilt_coefficient(lam, D)
    slope = (x * xp).mean(0) / (x * x).mean(0)
    assert np.allclose(slope, rho, atol=0.01)
    assert np.allclose(((xp - rho * x) ** 2).mean(0) / lam, 1 - rho, atol=0.01)
    assert np.allclose((xp * xp).mean(0) / lam, sp.neighbour_variance(lam, D), atol=0.01)
    corr = (x * xp).mean(0) / np.sqrt((x * x).mean(0) * (xp * xp).mean(0))
    assert np.allclose(corr, sp.neighbour_correlation(lam, D), atol=0.01)


def test_isotropic_pairs_have_correlation_cos_theta():
    lam = np.ones(400)
    x, z = sp.model_draws(lam, 3000, 1)
    D = sp.water_level(lam, 60.0, x, z)
    rho = sp.neighbour_correlation(lam, D)[0]
    assert abs(math.degrees(math.acos(rho)) - 60.0) < 1.0


def test_median_nn_angle_against_brute_force():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(150, 6)) * np.array([3.0, 2.0, 1.0, 1.0, 0.5, 0.5])
    top = 7
    xc = rp.centre(x)
    angles = []
    for i in range(len(x)):
        dist = ((x - x[i]) ** 2).sum(1)
        dist[i] = np.inf
        for j in np.argsort(dist)[:top]:
            cos = xc[i] @ xc[j] / np.linalg.norm(xc[i]) / np.linalg.norm(xc[j])
            angles.append(math.degrees(math.acos(np.clip(cos, -1, 1))))
    assert abs(sp.median_nn_angle(x, top=top) - np.median(angles)) < 1e-6


def test_synthetic_spectrum():
    x, lam = sp.synthetic(1.5, n=20_000)
    assert x.shape == (20_000, 51) and np.allclose(lam, np.arange(1, 52) ** -1.5)
    assert np.allclose(x.var(0) / lam, 1.0, atol=0.05)


# ------------------------------------------------------------------ analytic step

@pytest.mark.parametrize("D", [0.1, 0.4, 2.0])
def test_meanfield_slope_is_the_derivative_of_the_meanfield_collision(D):
    from scipy.optimize import brentq
    from scipy.stats import norm
    rng = np.random.default_rng(6)
    d, m, k = 5, 40, 3
    lam = np.array([1.0, 0.6, 0.3, 0.2, 0.1])
    M = np.zeros((d, m))
    for i in range(m):
        M[rng.choice(d, int(rng.integers(1, 4)), replace=False), i] = 1

    def collision(D):
        # independent drives; x has variance v, the neighbour w, covariance u
        v = lam @ M
        w = (lam * sp.neighbour_variance(lam, D)) @ M
        u = (lam * sp.tilt_coefficient(lam, D)) @ M
        r = u / np.sqrt(v * w)
        tau = brentq(lambda t: norm.sf(t / np.sqrt(v)).sum() - k, -50, 50)
        tau2 = brentq(lambda t: norm.sf(t / np.sqrt(w)).sum() - k, -50, 50)
        a, b = tau / np.sqrt(v), tau2 / np.sqrt(w)
        both = [multivariate_normal(mean=[0, 0], cov=[[1, ri], [ri, 1]]).cdf([-ai, -bi])
                for ai, bi, ri in zip(a, b, r)]
        return float(np.sum(both) / k)

    h = 1e-3 * D
    numeric = -(collision(D + h) - collision(D - h)) / (2 * h)
    assert abs(sp.meanfield_slope(M, lam, k, D) - numeric) < 2e-3 * abs(numeric) + 1e-6


# ------------------------------------------------------------------ the frozen record

def test_frozen_record_is_consistent():
    if not sp.RESULTS.exists():
        pytest.skip("no results/e4_frozen.json yet")
    res = json.loads(sp.RESULTS.read_text())
    if not res.get("complete"):
        pytest.skip("freeze still running")
    assert res["code"]["sha256"] == sp.code_sha256(), "spectral.py changed after the freeze"
    assert res["refreeze"]["supersedes"]["sha256"] != res["code"]["sha256"]
    assert res["refreeze"]["note"].startswith("re-frozen before any AP run")
    hist = res["inputs_per_cell"]["histogram"]
    c = np.repeat(np.arange(len(hist)), hist)
    assert c.sum() == sp.NNZ == 9967
    for beta, e in res["synthetic"].items():
        lam = np.array(e["lambda"])
        # the pair model reproduces the data's per-input neighbour correlation
        assert e["pair_model_check"]["rms_correlation_error"] < 0.01
        for k in ("4", "16", "94"):
            f = np.array(e["f_star"][k])
            assert f.sum() == sp.NNZ and f.min() >= 0 and sp.gale_ryser_ok(f, c)
            step = np.diff(f[np.argsort(-lam, kind="stable")])
            assert (step <= 0).all() or (step >= 0).all()
            if float(beta) == 0.0:
                assert f.std() / f.mean() < 0.05


@pytest.mark.data
def test_malecns_inputs_per_cell_give_nnz():
    from flypath import config, experiments as ex
    cfg = config.load()
    if not (cfg.graph_dir / "nodes.parquet").exists():
        pytest.skip("no built graph; run: python -m flypath build")
    c = ex.context(cfg).full.inputs()
    assert int(c.sum()) == sp.NNZ and len(c) == 1886
