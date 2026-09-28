"""E4 stages 1 and 2 runner (flypath.spectral_run)."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from flypath import replication as rp
from flypath import spectral as spc
from flypath import spectral_run as sr


def _cells(m: int = 240, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 9, m)


# ------------------------------------------------------------------ the hash

def test_top_k_exact_breaks_ties_by_priority():
    y = np.array([[5.0, 3.0, 3.0, 3.0, 1.0],
                  [1.0, 2.0, 3.0, 4.0, 5.0],
                  [2.0, 2.0, 0.0, 0.0, 0.0]])
    order = np.array([3, 1, 2, 0, 4])                  # cell 3 has the highest priority
    t = sr.top_k_exact(y, 2, order).toarray()
    assert (t.sum(1) == 2).all()
    np.testing.assert_array_equal(t[0], [1, 0, 0, 1, 0])   # 5 wins; tie at 3 -> cell 3
    np.testing.assert_array_equal(t[1], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(t[2], [1, 1, 0, 0, 0])   # all tied cells fit


def test_top_k_exact_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, (200, 30)).astype(float)     # many ties
    prio = rng.random(30)
    order = np.argsort(-prio, kind="stable")
    for k in (1, 4, 11):
        t = sr.top_k_exact(y, k, order).toarray()
        for i in range(len(y)):
            want = np.lexsort((-prio, -y[i]))[:k]        # by drive, then priority
            assert set(np.flatnonzero(t[i])) == set(want)


def test_identical_cells_get_identical_drive_and_priority_decides():
    rng = np.random.default_rng(1)
    d, m = 8, 40
    M = (rng.random((d, m)) < 0.3).astype(float)
    M[:, 0] = 0
    M[[1, 4], 0] = 1
    M[:, 1] = M[:, 0]                                    # cells 0 and 1 identical
    M[:, M.sum(0) == 0] = 1
    prio = rng.random(m)
    prio[1], prio[0] = 2.0, -1.0                          # cell 1 beats cell 0 on ties
    order = np.argsort(-prio, kind="stable")
    xc = rp.centre(rng.normal(size=(500, d)))
    t = sr.tags(xc, M, [3, 7], order)
    for k, tk in t.items():
        a = tk.toarray().astype(bool)
        assert (a.sum(1) == k).all()
        assert not (a[:, 0] & ~a[:, 1]).any()             # 0 never wins without 1
        y = xc @ M
        for i in range(0, 500, 50):                       # brute force with the same rule
            want = np.lexsort((-prio, -np.round(y[i], 12)))[:k]
            assert set(np.flatnonzero(a[i])) == set(want)


# ------------------------------------------------------------------ inputs

def test_prescale():
    z = np.random.default_rng(0).normal(size=(5, 4))
    lam = np.array([4.0, 1.0, 0.25, 0.0625])
    assert sr.prescale(z, lam, 0.0) is z
    np.testing.assert_allclose(sr.prescale(z, lam, 1.0), z * np.sqrt(lam))
    np.testing.assert_allclose(sr.prescale(z, lam, 2.0), z * lam)


def test_pca_fit_matches_replication_pca_and_fixes_signs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 12)) @ rng.normal(size=(12, 12)) + 3.0
    mu, v = sr.pca_fit(x, 5)
    z = sr.pca_apply(x, mu, v)
    ref = rp.pca(x, 5)
    np.testing.assert_allclose(np.abs(z), np.abs(ref), atol=1e-8)
    lam = spc.spectrum(z)
    assert (np.diff(lam) <= 0).all()
    assert (v[np.abs(v).argmax(0), np.arange(5)] > 0).all()
    np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-10)


def test_split_indices_are_disjoint():
    for name, n_pool in (("mnist", 60_000), ("glove", 400_000)):
        items, tr, va = sr.split_indices(name, n_pool, 10_000, 10_000, 10_000)
        assert len(set(items)) == len(set(tr)) == len(set(va)) == 10_000
        assert not (set(items) & set(tr)) and not (set(items) & set(va)) and not (set(tr) & set(va))
        # the items are load_benchmark's draw
        want = np.random.default_rng(0).choice(n_pool, 10_000, replace=False)
        np.testing.assert_array_equal(items, np.sort(want) if name == "glove" else want)
    items, tr, va = sr.split_indices("sift", 25_000, 10_000, 10_000, 10_000)
    assert items is None and not (set(tr) & set(va))
    with pytest.raises(RuntimeError):
        sr.split_indices("sift", 25_000, 10_000, 20_000, 10_000)


# ------------------------------------------------------------------ arms

def test_allocations_even_grid_and_fstar():
    c = _cells()
    lam = np.arange(1, 17, dtype=float) ** -1.0
    f4 = spc.power_allocation(lam, 6.0, c)
    al = sr.allocations(lam, c, {4: f4}, sr.GRID)
    assert set(al) == {"even", "fstar:4", *(f"grid:{a:g}" for a in sr.GRID)}
    np.testing.assert_array_equal(al["even"], al["grid:0"])
    np.testing.assert_array_equal(al["fstar:4"], f4)
    for f in al.values():
        assert f.sum() == c.sum() and spc.gale_ryser_ok(f, c)
    assert np.ptp(al["even"]) <= 1
    extra = al["even"] == al["even"].max()
    assert extra[: extra.sum()].all()                   # extra units on the top-variance inputs
    for a in sr.GRID[1:]:
        assert (np.diff(al[f"grid:{a:g}"]) <= 0).all()    # lam decreasing -> f non-increasing


def test_arm_source():
    choice = {"4": {"alpha": "1.25"}}
    assert sr.arm_source("even", 4) == ("even", 0.0)
    assert sr.arm_source("fstar", 16) == ("fstar:16", 0.0)
    assert sr.arm_source("grid:0.5", 4) == ("grid:0.5", 0.0)
    assert sr.arm_source("prescale:0.75", 4) == ("even", 0.75)
    assert sr.arm_source("prescale_best", 4, choice) == ("even", 1.25)
    arms = sr.test_arms(sr.DESIGN)
    assert len(arms) == 2 + 8 + 8 + 1 and len(set(arms)) == len(arms)
    assert sr.validation_arms(sr.DESIGN) == [f"prescale:{a}" for a in sr.DESIGN["grid_alphas"]]


def test_choose_prescale_picks_the_best_mean_and_the_smallest_alpha_on_ties():
    tr = {"0": {"ap": {"prescale:0": {"4": 0.1, "16": 0.3}, "prescale:0.5": {"4": 0.2, "16": 0.2},
                       "prescale:1": {"4": 0.2, "16": 0.1}}},
          "1": {"ap": {"prescale:0": {"4": 0.1, "16": 0.3}, "prescale:0.5": {"4": 0.2, "16": 0.2},
                       "prescale:1": {"4": 0.2, "16": 0.1}}}}
    ch = sr.choose_prescale(tr, ["0", "0.5", "1"], [4, 16])
    assert ch["4"]["alpha"] == "0.5" and ch["16"]["alpha"] == "0"
    assert ch["4"]["validation_trials"] == 2


def test_coincident_arms():
    c = _cells()
    lam = np.ones(16)
    al = sr.allocations(lam, c, {4: spc.power_allocation(lam, 0.0, c)}, sr.GRID)
    co = sr.coincident(al, [4], lam, sr.DESIGN["prescale_alphas"])
    assert co["isotropic_lambda"]
    assert len(co["even"]) == 8 + 8                     # every grid and prescale arm
    assert "even" in co["fstar@4"]
    lam = np.arange(1, 17, dtype=float) ** -2.0
    al = sr.allocations(lam, c, {4: spc.power_allocation(lam, 1.0, c)}, sr.GRID)
    co = sr.coincident(al, [4], lam, sr.DESIGN["prescale_alphas"])
    assert co["even"] == ["grid:0", "prescale:0"] and "grid:1" in co["fstar@4"]


def test_run_trial_uses_common_random_numbers():
    rng = np.random.default_rng(3)
    c = _cells()
    lam = np.arange(1, 17, dtype=float) ** -1.0
    z = rng.normal(size=(600, 16)) * np.sqrt(lam)
    al = sr.allocations(lam, c, {4: spc.power_allocation(lam, 1.0, c)}, sr.GRID)
    arms = ["even", "fstar", "grid:0", "grid:1", "grid:2", "prescale:0", "prescale:0.5",
            "prescale_best"]
    choice = {"4": {"alpha": "0.5"}}
    kw = dict(n_queries=60, top=12, choice=choice, sweeps=2)
    r = sr.run_trial(z, lam, c, al, arms, [4], [0, 1, 2], **kw)
    ap = {a: r["ap"][a]["4"] for a in arms}
    assert ap["even"] == ap["grid:0"] == ap["prescale:0"]
    assert ap["fstar"] == ap["grid:1"]                   # same allocation -> same matrix
    assert ap["prescale_best"] == ap["prescale:0.5"]
    assert all(0.0 <= v <= 1.0 for v in ap.values())
    assert r["distinct_matrices"] == 3                   # even, grid:1 (= f*), grid:2
    assert r["distinct_hashes"] == 4                     # + prescale:0.5 on the even matrix
    r2 = sr.run_trial(z, lam, c, al, arms, [4], [0, 1, 2], **kw)
    assert r2["ap"] == r["ap"]
    r3 = sr.run_trial(z, lam, c, al, arms, [4], [0, 1, 3], **kw)
    assert r3["ap"] != r["ap"]


# ------------------------------------------------------------------ the freeze

def test_check_frozen_passes_on_the_frozen_code():
    rec = sr.check_frozen()
    assert rec["code"]["sha256"] == hashlib.sha256(open(spc.__file__, "rb").read()).hexdigest()


def test_check_frozen_aborts_on_another_hash_or_an_incomplete_freeze(tmp_path):
    rec = json.loads(sr.FROZEN.read_text())
    bad = copy.deepcopy(rec)
    bad["code"]["sha256"] = "0" * 64
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        sr.check_frozen(p)
    bad = copy.deepcopy(rec)
    bad["complete"] = False
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        sr.check_frozen(p)


def test_run_refuses_before_writing_when_the_freeze_does_not_match(tmp_path):
    from flypath import config
    rec = json.loads(sr.FROZEN.read_text())
    rec["code"]["sha256"] = "f" * 64
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(rec))
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        sr.run(config.load(), out=out, design=sr.SMOKE, frozen_path=p, log=lambda s: None)
    assert not out.exists()


def test_smoke_cannot_write_the_results_file():
    with pytest.raises(SystemExit):
        sr.main(["--smoke"])


def test_design_uses_the_frozen_model_defaults():
    F = sr.DESIGN["fstar_model"]
    assert F["pairs"] == spc.N_PAIRS and F["wirings"] == spc.N_WIRINGS
    assert F["alphas"] == [spc._alpha_label(a) for a in spc.ALPHAS]
    assert F["half_width_deg"] == spc.HALF_WIDTH_DEG and F["seed"] == spc.SEED
    assert sr.DESIGN["ks"] == [4, 16, 94] and sr.DESIGN["stage1"]["trials"] == 10
    assert sr.DESIGN["stage2"]["trials"] == 20
    assert sr.DESIGN["grid_alphas"] == ["0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"]


# ------------------------------------------------------------------ with data

TINY = copy.deepcopy(sr.SMOKE)
TINY.update({"n_items": 300, "n_queries": 30, "ks": [4]})
TINY["stage1"].update({"betas": [1.0], "trials": 2, "validation_trials": 1})
TINY["stage2"].update({"datasets": []})


@pytest.mark.data
def test_stage1_tiny_run_uses_frozen_fstar_and_resumes_identically(tmp_path):
    from flypath import config
    cfg = config.load()
    out = tmp_path / "e4.json"
    res = sr.run(cfg, out=out, design=TINY, log=lambda s: None)
    assert res["complete"]
    e = res["stage1"]["1"]
    np.testing.assert_array_equal(e["fstar"]["4"]["fanout"], spc.frozen_fstar(1.0, 4))
    t = e["test"]["trials"]
    assert set(t) == {"0", "1"}
    a = e["prescale_choice"]["4"]["alpha"]
    for tr in t.values():
        assert set(tr["ap"]) == set(sr.test_arms(TINY))
        assert tr["ap"]["even"] == tr["ap"]["grid:0"] == tr["ap"]["prescale:0"]
        assert tr["ap"]["prescale_best"] == tr["ap"][f"prescale:{a}"]
    # drop a trial and resume: it is recomputed identically
    saved = json.loads(out.read_text())
    keep = saved["stage1"]["1"]["test"]["trials"].pop("1")
    del saved["stage1"]["1"]["summary"]
    saved["complete"] = False
    out.write_text(json.dumps(saved))
    res2 = sr.run(cfg, out=out, design=TINY, log=lambda s: None)
    assert res2["complete"]
    got = res2["stage1"]["1"]["test"]["trials"]["1"]
    assert got["ap"] == keep["ap"] and got["recall"] == keep["recall"]
    # a different design is refused
    other = copy.deepcopy(TINY)
    other["stage1"]["trials"] = 3
    with pytest.raises(SystemExit):
        sr.run(cfg, out=out, design=other, log=lambda s: None)


@pytest.mark.data
@pytest.mark.parametrize("name", ["sift", "mnist"])
def test_load_splits(name):
    from flypath import config
    cfg = config.load()
    items, train, val, meta = sr.load_splits(cfg, name, 10_000, 2000, 1000)
    np.testing.assert_array_equal(items, rp.load_benchmark(cfg, name))
    assert train.shape == (2000, items.shape[1]) and val.shape == (1000, items.shape[1])
    assert meta["n_train"] == 2000 and meta["n_validation"] == 1000


# ------------------------------------------------------------------ review checks
# (adversarial review against E4: exact margins, equal nnz, AP@200 against an
# independent implementation, Lambda/PCA/theta from the training split only,
# pre-scaling chosen on validation before any test trial)

def _small_problem(seed: int = 3):
    rng = np.random.default_rng(seed)
    c = _cells()
    lam = np.arange(1, 17, dtype=float) ** -1.0
    z = rng.normal(size=(600, 16)) * np.sqrt(lam)
    fs = {4: spc.power_allocation(lam, math.inf, c), 16: spc.power_allocation(lam, -0.5, c)}
    return z, lam, c, sr.allocations(lam, c, fs, sr.GRID)


def test_run_trial_matrices_have_exact_margins_and_equal_nnz(monkeypatch):
    z, lam, c, al = _small_problem()
    seen = []
    real = spc.realise

    def spy(cc, f, seed, sweeps=spc.SWEEPS):
        M = real(cc, f, seed=seed, sweeps=sweeps)
        seen.append((np.asarray(f, int).copy(), M))
        return M
    monkeypatch.setattr(spc, "realise", spy)
    r = sr.run_trial(z, lam, c, al, sr.test_arms(sr.DESIGN), [4, 16], [0, 7, 1],
                     n_queries=60, top=12, choice={"4": {"alpha": "1"}, "16": {"alpha": "2"}},
                     sweeps=3)
    assert len(seen) == r["distinct_matrices"] >= 9           # even, 7 grid, f* (2 distinct)
    for f, M in seen:
        assert set(np.unique(M)) <= {0.0, 1.0}
        np.testing.assert_array_equal(M.sum(1).astype(int), f)  # row sums = the arm's fan-out
        np.testing.assert_array_equal(M.sum(0).astype(int), c)  # column sums = inputs per cell
        assert int(M.sum()) == int(c.sum())                     # equal nnz in every arm
    # every arm's allocation was realised (pre-scaling arms use the even matrix)
    keys = {sr._key(f) for f, _ in seen}
    assert {sr._key(f) for f in al.values()} == keys


def _brute_ap(z, lam, c, f, pre, k, seeds, n_queries, top, sweeps):
    """AP@top of one arm, written independently of spectral_run and replication
    (only the documented seed streams and `spectral.realise` are shared)."""
    n, m = len(z), len(c)
    q = np.random.default_rng(seeds + [1]).choice(n, n_queries, replace=False)
    true = []
    for qi in q:
        d = ((z - z[qi]) ** 2).sum(1)
        d[qi] = np.inf
        true.append(set(np.argsort(d, kind="stable")[:top].tolist()))
    rank = np.empty(m, int)
    rank[np.argsort(-np.random.default_rng(seeds + [2]).random(m), kind="stable")] = np.arange(m)
    M = spc.realise(c, f, seed=seeds + [4], sweeps=sweeps)
    x = z * lam ** (pre / 2.0)
    y = np.round((x - x.mean(1, keepdims=True)) @ M, 9)
    T = np.zeros((n, m), bool)
    np.put_along_axis(T, np.lexsort((np.broadcast_to(rank, y.shape), -y), axis=1)[:, :k], True, 1)
    assert (T.sum(1) == k).all()
    dist = (2 * k - 2 * (T[q].astype(int) @ T.T.astype(int))).astype(float)
    dist += 0.5 * np.random.default_rng(seeds + [3, k]).random(dist.shape)
    dist[np.arange(len(q)), q] = np.inf
    pred = np.argsort(dist, axis=1, kind="stable")[:, :top]
    aps = []
    for p, t in zip(pred, true):
        hit = np.array([i in t for i in p], float)
        aps.append(float((np.cumsum(hit) / np.arange(1, top + 1) * hit).sum() / top))
    return float(np.mean(aps))


def test_run_trial_ap_matches_an_independent_implementation():
    z, lam, c, al = _small_problem()
    seeds = [0, 11, 5]
    arms = ["even", "fstar", "grid:2", "prescale:1.5"]
    r = sr.run_trial(z, lam, c, al, arms, [4, 16], seeds, n_queries=60, top=12, sweeps=3)
    for arm in arms:
        for k in (4, 16):
            alloc, pre = sr.arm_source(arm, k)
            want = _brute_ap(z, lam, c, al[alloc], pre, k, seeds, 60, 12, 3)
            assert r["ap"][arm][str(k)] == pytest.approx(want, abs=1e-12), (arm, k)


def test_average_precision_counts_misses_in_the_denominator():
    true = np.array([[0, 1, 2, 3]])
    assert rp.average_precision(np.array([[0, 9, 8, 7]]), true) == pytest.approx(1 / 4)
    assert rp.average_precision(np.array([[9, 0, 1, 8]]), true) == pytest.approx((1 / 2 + 2 / 3) / 4)
    assert rp.average_precision(np.array([[3, 2, 1, 0]]), true) == pytest.approx(1.0)


def test_stage2_fits_lambda_pca_and_theta_on_training_only_and_chooses_prescale_first(
        monkeypatch, tmp_path):
    """Stage 2 end to end on stand-in data (no data files needed): Lambda, the
    PCA and theta come from the training split alone; f* is computed from
    them; the pre-scaling exponent is chosen from validation trials only,
    before the first test trial; every realised matrix has exact margins."""
    from flypath import config
    rng = np.random.default_rng(5)
    c = _cells()
    raw = 60
    mix = rng.normal(size=(raw, raw))
    train = rng.normal(size=(300, raw)) * np.linspace(3.0, 0.2, raw) @ mix
    val = rng.normal(size=(300, raw)) * np.linspace(3.0, 0.2, raw) @ mix
    items = rng.normal(size=(300, raw)) * np.linspace(0.2, 3.0, raw) @ mix + 5.0  # other spectrum
    meta = {"source": "stand-in", "n_items": 300, "n_train": 300, "n_validation": 300,
            "raw_dim": raw, "split_sha1": "x"}
    monkeypatch.setattr(sr, "load_splits", lambda *a, **kw: (items, train, val, meta))
    monkeypatch.setattr(sr, "inputs_per_cell", lambda cfg, frozen=None: c)
    calls = {}

    def fake_fstar_all(lam, ks, cc, nnz, theta, **kw):
        calls.update(lam=np.array(lam), theta=theta, nnz=nnz)
        f = {int(k): spc.power_allocation(lam, 1.0, cc) for k in ks}
        info = {"setup": {"theta_deg": theta}, "curve": {},
                "k": {str(int(k)): {"alpha_star": "1", "cv": 0.0, "slope": 0.0, "se": 0.0,
                                    "alphas_with_same_allocation": ["1"]} for k in ks}}
        return f, info
    monkeypatch.setattr(spc, "fstar_all", fake_fstar_all)
    order = []
    real_trial, real_choose, real_realise = sr.run_trial, sr.choose_prescale, spc.realise

    def trial_spy(z, *a, **kw):
        split = "test" if "fstar" in a[3] else "validation"
        order.append(("trial", split, z))
        return real_trial(z, *a, **kw)

    def choose_spy(trials, *a, **kw):
        order.append(("choose", sorted(trials)))
        return real_choose(trials, *a, **kw)

    def realise_spy(cc, f, seed, sweeps=spc.SWEEPS):
        M = real_realise(cc, f, seed=seed, sweeps=sweeps)
        np.testing.assert_array_equal(M.sum(1).astype(int), f)
        np.testing.assert_array_equal(M.sum(0).astype(int), cc)
        return M
    monkeypatch.setattr(sr, "run_trial", trial_spy)
    monkeypatch.setattr(sr, "choose_prescale", choose_spy)
    monkeypatch.setattr(spc, "realise", realise_spy)

    D = copy.deepcopy(sr.SMOKE)
    D.update({"n_items": 300, "n_queries": 30, "ks": [4, 16], "sweeps": 3})
    D["stage1"].update({"betas": []})
    D["stage2"].update({"datasets": ["sift"], "trials": 2, "validation_trials": 2,
                        "n_train": 300, "n_validation": 300})
    res = sr.run(config.load(), out=tmp_path / "e4.json", design=D, log=lambda s: None)
    assert res["complete"]
    e = res["stage2"]["sift"]
    mu, v = sr.pca_fit(train, 51)
    z_train = sr.pca_apply(train, mu, v)
    lam_train = spc.spectrum(z_train)
    np.testing.assert_array_equal(e["lambda"], lam_train)
    np.testing.assert_array_equal(calls["lam"], lam_train)
    assert calls["theta"] == e["theta_deg"] == spc.median_nn_angle(z_train)
    assert calls["nnz"] == D["nnz"]
    # the items' own PCA spectrum differs, so an items-fitted Lambda would be caught
    assert not np.allclose(spc.spectrum(rp.pca(items, 51)), lam_train, rtol=0.1)
    # the hashed items are the items mapped through the training PCA
    test_z = [o[2] for o in order if o[0] == "trial" and o[1] == "test"]
    np.testing.assert_array_equal(test_z[0], sr.pca_apply(items, mu, v))
    # validation trials, then the choice (from validation trials only), then test trials
    kinds = [o[0] if o[0] == "choose" else o[1] for o in order]
    assert kinds == ["validation", "validation", "choose", "test", "test"]
    assert order[2][1] == ["0", "1"]
    ch = real_choose(e["validation"]["trials"], D["prescale_alphas"], D["ks"])
    assert {k: v["alpha"] for k, v in ch.items()} == {k: v["alpha"] for k, v in
                                                      e["prescale_choice"].items()}
    for tr in e["test"]["trials"].values():
        for k in ("4", "16"):
            a = e["prescale_choice"][k]["alpha"]
            assert tr["ap"]["prescale_best"][k] == tr["ap"][f"prescale:{a}"][k]
