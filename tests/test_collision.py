"""Scorers and estimators of flypath.collision (E3) on hand-checkable inputs."""

import json

import numpy as np
import pytest
import scipy.sparse as sp

from flypath import collision as c
from flypath import replication as rp
from flypath.connectomes import HEMISPHERES


def _curve(angle, value, count=None):
    return {"angle": list(angle), "value": list(value),
            "count": list(count if count is not None else [1000] * len(angle))}


# ---------------------------------------------------------------- theta_half

def test_theta_half_interpolates_linearly_between_bins():
    # 0.7 at 15 deg, 0.4 at 25 deg: 0.5 is reached 1/3 of the way, at 15 + 20/3
    th, status, n = c.theta_half(_curve([5, 15, 25, 35], [1.0, 0.7, 0.4, 0.2]), 1)
    assert status == "ok" and n == 1
    assert np.isclose(th, 15 + 20 / 3)


def test_theta_half_exact_hit_on_a_bin():
    th, status, _ = c.theta_half(_curve([10, 20, 30], [0.9, 0.5, 0.1]), 1)
    # 0.5 is not below 0.5, so the crossing is between 20 and 30, starting at 20
    assert status == "ok" and np.isclose(th, 20.0)


def test_theta_half_undefined_outside_the_observed_range():
    th, status, _ = c.theta_half(_curve([30, 40], [0.4, 0.2]), 1)
    assert np.isnan(th) and status == "below_range"
    th, status, _ = c.theta_half(_curve([30, 40], [0.9, 0.6]), 1)
    assert np.isnan(th) and status == "above_range"
    th, status, _ = c.theta_half(_curve([30], [0.9], [5]), 10)
    assert np.isnan(th) and status == "no_bins"


def test_theta_half_takes_the_first_crossing_and_counts_them():
    th, status, n = c.theta_half(_curve([10, 20, 30, 40, 50], [0.8, 0.4, 0.6, 0.3, 0.1]), 1)
    assert status == "ok" and n == 2
    assert np.isclose(th, 10 + 0.3 / 0.4 * 10)


def test_theta_half_skips_sparse_bins():
    # the middle bin has too few pairs, so interpolation spans 10 -> 30
    curve = _curve([10, 20, 30], [1.0, 0.0, 0.0], [500, 3, 500])
    th, status, _ = c.theta_half(curve, 100)
    assert status == "ok" and np.isclose(th, 20.0)


# ---------------------------------------------------------------- binning, angles, pairs

def test_bin_curve_means_and_counts():
    angle = np.array([0.5, 1.5, 2.5, 3.9, 7.0])
    value = np.array([1.0, 0.0, 0.5, 0.5, 0.2])
    b = c.bin_curve(angle, value, 2.0)
    assert b["lo"] == [0.0, 2.0, 6.0]
    np.testing.assert_allclose(b["angle"], [1.0, 3.2, 7.0])
    np.testing.assert_allclose(b["value"], [0.5, 0.5, 0.2])
    assert b["count"] == [2, 2, 1]


def test_angles_deg():
    u = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    v = np.array([[0, 1.0, 0], [1.0, 0, 0], [-1.0, 0, 0],
                  [np.cos(np.radians(60)), np.sin(np.radians(60)), 0]])
    np.testing.assert_allclose(c.angles_deg(u, v), [90, 0, 180, 60], atol=1e-9)


def test_angles_deg_is_accurate_for_tiny_angles():
    a = 1e-7                                            # radians
    u = np.array([[1.0, 0.0]])
    v = np.array([[np.cos(a), np.sin(a)]])
    assert np.isclose(c.angles_deg(u, v)[0], np.degrees(a), rtol=1e-6)


def test_unit_rows_flags_zero_vectors():
    u, ok = c.unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(u[0], [0.6, 0.8])
    assert ok.tolist() == [True, False]


def test_sample_pairs_are_unordered_unique_and_include_nearest_neighbours():
    x = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99], [-1.0, 0.0], [0.0, 0.0]])
    a, b, meta = c.sample_pairs(x, nn_per_item=1, n_random=50, rng=np.random.default_rng(0))
    pairs = set(zip(a.tolist(), b.tolist()))
    assert (a < b).all() and len(pairs) == len(a)
    assert 5 not in a and 5 not in b                     # the zero vector is left out
    assert meta["items_zero_norm"] == 1
    assert {(0, 1), (2, 3)} <= pairs                      # each item's nearest neighbour


# ---------------------------------------------------------------- overlap and winners

def test_pair_overlap_fraction():
    t = sp.csr_matrix(np.array([[1, 1, 0, 0, 0],
                                [0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1]], dtype=np.float32))
    a, b = np.array([0, 0, 0, 1]), np.array([1, 2, 0, 2])
    np.testing.assert_allclose(c.pair_overlap(t, a, b, chunk=2), [0.5, 0.0, 1.0, 0.0])


def test_wta_breaks_exact_ties_the_same_way_for_every_item():
    # s = 1: cells 0-2 sample input 0, cells 3-5 input 1; with k = 2 the
    # winners of an item whose larger (row-centred) input is 0 are two of
    # cells 0-2, and the same two for every such item
    w = np.zeros((2, 6), np.float32)
    w[0, :3] = 1
    w[1, 3:] = 1
    x = np.array([[3.0, 1.0], [5.0, -2.0], [1.0, 4.0], [0.5, 2.0]])
    prio = np.random.default_rng(3).random(6)
    tags = c._wta(x, w, [2], prio)[2].toarray().astype(bool)
    assert (tags.sum(1) == 2).all()
    assert (tags[0] == tags[1]).all() and tags[0, :3].sum() == 2
    assert (tags[2] == tags[3]).all() and tags[2, 3:].sum() == 2
    assert np.array_equal(np.flatnonzero(tags[0]), np.sort(np.argsort(-prio[:3])[:2]))


def test_wta_is_top_k_of_the_row_centred_drive():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 20)) + 3.0                  # non-zero row means
    for w in (rng.normal(size=(20, 500)).astype(np.float32),
              rp.fly_matrix(20, 500, rng, sampled=4)):
        prio = rng.random(500)
        t = c._wta(x, w, [8], prio)[8]
        ref = rp.winners(rp.centre(x) @ w.astype(np.float64), 8, prio)
        assert (t != ref).nnz == 0


def test_wta_breaks_integer_sum_ties_by_the_priority():
    # integer inputs, d = 100 (the row mean is not exact in binary floating
    # point): cells with different input sets but equal sums are tied in exact
    # arithmetic and must be resolved by `prio`, as in an exact reference.
    # Centring first in float32 broke some of these ties by rounding noise.
    rng = np.random.default_rng(0)
    d, m, k = 100, 2000, 16
    x = rng.integers(0, 256, size=(1500, d)) * (rng.random((1500, d)) < 0.3)
    w = rp.fly_matrix(d, m, rng, sampled=10)
    prio = rng.random(m)
    raw = x @ w.astype(np.int64)                          # exact integer drives
    top = np.lexsort((-np.broadcast_to(prio, raw.shape), -raw), axis=1)[:, :k]
    ref = np.zeros(raw.shape, bool)
    ref[np.arange(len(raw))[:, None], top] = True
    srt = -np.sort(-raw, 1)
    assert (srt[:, k - 1] == srt[:, k]).mean() > 0.05     # ties are common here
    got = c._wta(x.astype(float), w, [k], prio)[k].toarray().astype(bool)
    assert np.array_equal(got, ref)


def test_wta_chunks_agree_with_one_call():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(700, 20))
    w = rng.normal(size=(20, 40000)).astype(np.float32)   # forces several row chunks
    prio = rng.random(40000)
    t = c._wta(x, w, [4], prio)[4]
    y = (x.astype(np.float32) @ w).astype(np.float64)
    y -= x.mean(1)[:, None] * w.sum(0, dtype=np.float64)
    assert (t != c._top_k(y, 4, prio)).nnz == 0
    assert (t != rp.winners(y, 4, prio)).nnz == 0        # same rule without ties


def test_top_k_breaks_ties_by_priority_and_keeps_exactly_k():
    prio = np.array([0.1, 0.9, 0.2, 0.5, 0.3])
    y = np.array([[3.0, 5.0, 5.0, 5.0, 1.0],               # 3 tied for 2 places
                  [9.0, 5.0, 5.0, 1.0, 0.0],               # 1 above, 2 tied for 1 place
                  [9.0, 8.0, 7.0, 1.0, 0.0]])              # no tie
    got = c._top_k(y, 2, prio).toarray().astype(bool)
    assert got.sum(1).tolist() == [2, 2, 2]
    assert np.flatnonzero(got[0]).tolist() == [1, 3]
    assert np.flatnonzero(got[1]).tolist() == [0, 1]
    assert np.flatnonzero(got[2]).tolist() == [0, 1]
    # all tied cells win when they exactly fill the remaining places
    assert np.flatnonzero(c._top_k(y[1:2], 3, prio).toarray()[0]).tolist() == [0, 1, 2]


def test_top_k_is_exact_where_the_winners_jitter_vanishes():
    # a tiny gap among the first 200 rows makes the jitter of
    # replication.winners smaller than the float resolution at 1000
    m = 6
    prio = np.array([0.05, 0.9, 0.1, 0.2, 0.3, 0.4])
    y = np.zeros((3, m))
    y[0, :2] = [1e-13, 3e-13]
    y[1] = [1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0]           # cells 0 and 1 tied for 1 place
    y[2] = [0.0, 0.0, 1000.0, 1000.0, 1000.0, 1.0]        # cells 2-4 tied for 1 place
    got = c._top_k(y, 1, prio).toarray().astype(bool)
    assert np.flatnonzero(got[1]).tolist() == [1] and np.flatnonzero(got[2]).tolist() == [4]


def test_fly_matrix_at_full_fan_in_is_degenerate():
    # documented design issue: at s = d all Kenyon cells are identical
    w = rp.fly_matrix(7, 30, np.random.default_rng(0), sampled=7)
    assert (w == 1).all()


# ---------------------------------------------------------------- predictor and fit

def test_threshold_is_the_upper_tail_quantile():
    assert np.isclose(c.threshold(1, 2), 0.0)
    assert np.isclose(c.threshold(2275, 100000), 2.0, atol=1e-3)
    assert c.threshold(3, 4) < 0


def test_fit_c_recovers_exact_constant_and_is_least_squares():
    t = np.array([1.5, 2.0, 3.0])
    assert np.isclose(c.fit_c(28.0 / t, t), 28.0)
    theta = np.array([20.0, 14.0, 9.0])
    grid = np.linspace(20, 40, 20001)
    sse = ((theta[None, :] - grid[:, None] / t[None, :]) ** 2).sum(1)
    assert abs(c.fit_c(theta, t) - grid[sse.argmin()]) < 1e-3


# ---------------------------------------------------------------- d_eff and (c) verdict

def test_participation_ratio():
    # two equal-variance orthogonal directions -> 2; one direction -> 1
    z = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
    assert np.isclose(c.participation_ratio(z), 2.0)
    assert np.isclose(c.participation_ratio(np.array([[1.0, 0], [-1.0, 0]])), 1.0)
    # eigenvalues 2, 1, 1 -> 16 / 6
    z3 = np.array([[np.sqrt(2), 0, 0], [-np.sqrt(2), 0, 0], [0, 1, 0], [0, -1, 0],
                   [0, 0, 1], [0, 0, -1]]) * np.sqrt(3)
    assert np.isclose(c.participation_ratio(z3), 16 / 6)


def test_s_values_caps_and_dedupes():
    grid = [1, 2, 3, 5, 8, 13, 26, "d"]
    assert c.s_values(grid, 51) == [1, 2, 3, 5, 8, 13, 26, 51]
    assert c.s_values(grid, 35) == [1, 2, 3, 5, 8, 13, 26, 35]
    assert c.s_values(grid, 20) == [1, 2, 3, 5, 8, 13, 20]


def test_fanin_verdict_pass_and_each_failure():
    cc = c.DESIGN["c"]
    s = [1, 2, 3, 5, 8, 13, 26, 51]
    good = [0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]
    v = c.fanin_verdict(s, good, d_eff=40.0, c=cc)    # 0.3 d_eff = 12 -> s in {13, 26, 51}
    assert v["pass"] and np.isclose(v["spearman"], -1.0) and v["high_s"] == [13, 26, 51]
    low = list(good)
    low[3] = 0.09                                    # s = 5 not > 10%
    assert not c.fanin_verdict(s, low, 40.0, cc)["low_s_ok"]
    high = list(good)
    high[-1] = 0.99                                  # degenerate s = d
    v = c.fanin_verdict(s, high, 40.0, cc)
    assert not v["high_s_ok"] and not v["spearman_ok"] and not v["pass"]
    assert not c.fanin_verdict(s, [0.2] * 8, 40.0, cc)["spearman_ok"]   # nan fails


def test_fanin_verdict_flags_contradictory_low_and_high_s():
    cc = c.DESIGN["c"]
    s = [1, 2, 3, 5, 8, 13, 26, 51]
    dfc = [0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]
    # SIFT-like d_eff 6.68: 0.3 d_eff = 2.004, so s = 3 and 5 are in both sets
    v = c.fanin_verdict(s, dfc, 6.68, cc)
    assert v["contradictory_s"] == [3, 5] and not v["pass"]
    assert c.fanin_verdict(s, dfc, 21.05, cc)["contradictory_s"] == []   # MNIST-like


def test_first_retained():
    curve = _curve([1, 3, 5, 7], [1.0, 0.9, 0.7, 0.3], [5, 50, 150, 900])
    assert c.first_retained(curve, 100) == 5.0
    assert c.first_retained(curve, 1000) is None


def test_equivalence_verdict():
    assert c.equivalence_verdict(0.0951, 0.100, 0.05)["pass"]
    assert c.equivalence_verdict(0.1049, 0.100, 0.05)["pass"]
    v = c.equivalence_verdict(0.094, 0.100, 0.05)
    assert not v["pass"] and np.isclose(v["relative_difference"], 0.06)


# ---------------------------------------------------------------- criteria from stored numbers

def _fake_results():
    D = json.loads(json.dumps(c.DESIGN))
    res = {"design": D, "a": {}, "b": {"cells": {}, "fit": {"c_deg": 30.0}}, "c": {}}
    for i, name in enumerate(D["a"]["datasets"]):
        fly = 0.10 if i < 2 else 0.12                # the third dataset misses by 20%
        res["a"][name] = {"trials": {str(t): {"fly": {"4": {"ap": fly}, "16": {"ap": fly}},
                                              "gwta": {"4": {"ap": 0.10}, "16": {"ap": 0.10}}}
                                     for t in range(D["a"]["trials"])}}
    n = 0
    for cell in c.b_cells(D):
        t = 2.0
        th = 15.0 + (2.0 if n % 10 else 5.0)            # every 10th cell misses by 5 deg
        res["b"]["cells"][cell["key"]] = {**cell, "m": 1000, "t": t, "theta_half": th,
                                          "status": "ok"}
        n += cell["role"] == "heldout"
    for ds, side in D["b"]["hemispheres"]:
        res["b"]["cells"][f"connectome|{ds}_{side}"] = {
            "role": "connectome", "k": 92, "m": 1838, "t": 1.645,
            "theta_half": 30.0 / 1.645 + 1.0, "status": "ok"}
    for name in D["c"]["inputs"]:
        d = 35 if name == "odours" else 51
        s = c.s_values(D["c"]["s_grid"], d)
        dfc = [0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.99]  # degenerate s = d
        res["c"][name] = {"d": d, "d_eff": 40.0, "s_grid": s, "trials": {
            str(t): {"gwta": {"16": {"ap": 0.2}, "92": {"ap": 0.2}},
                     "fly": {str(si): {"16": {"ap": 0.2 * (1 - x)}, "92": {"ap": 0.2 * (1 - x)}}
                             for si, x in zip(s, dfc)}}
            for t in range(D["c"]["trials"])}}
    return res


def test_criteria_recomputes_every_verdict():
    cr = c.criteria(_fake_results())
    assert cr["a"]["complete"] and cr["a"]["n_pass"] == 4 and cr["a"]["pass"] is False
    b = cr["b"]
    assert b["complete"] and b["heldout_n_cells"] == 36
    assert b["heldout_n_pass"] == 36 - 4                 # held-out cells 0, 10, 20, 30 miss
    assert b["pass_heldout"] is False                    # 32/36 < 90%
    assert b["connectome_n_pass"] == 7 and b["pass_connectomes"] is True
    assert b["pass"] is False
    cc = cr["c"]
    assert cc["complete"] and cc["n_inputs_pass_both_k"] == 0 and cc["pass"] is False
    x = cc["not_preregistered_without_s_eq_d"]
    assert x["n_inputs_pass_both_k"] == 4
    assert cr["complete"]


def test_criteria_counts_undefined_cells_and_t_below_zero_as_failures():
    res = _fake_results()
    cells = [v for v in res["b"]["cells"].values() if v["role"] == "heldout"]
    cells[1]["theta_half"] = None                        # undefined theta_half
    cells[2]["t"] = -0.6                                 # k/m > 0.5
    cr = c.criteria(res)
    rows = {r["key"]: r for r in cr["b"]["heldout"]}
    assert not rows[cells[1]["key"]]["pass"] and rows[cells[1]["key"]]["error_deg"] is None
    assert not rows[cells[2]["key"]]["pass"] and rows[cells[2]["key"]]["predicted"] is None


def test_criteria_marks_cells_whose_prediction_is_below_the_measurable_range():
    res = _fake_results()
    cells = [v for v in res["b"]["cells"].values() if v["role"] == "heldout"]
    # predicted 30 / 2 = 15 deg, but the first bin with >= 100 pairs is at 47 deg
    cells[3]["curve"] = _curve([41.0, 47.0, 49.0], [0.6, 0.55, 0.45], [40, 150, 300])
    cells[3]["theta_half"] = 48.0
    cells[4]["curve"] = _curve([13.0, 15.0], [0.6, 0.45], [150, 300])
    rows = {r["key"]: r for r in c.criteria(res)["b"]["heldout"]}
    assert rows[cells[3]["key"]]["first_retained_deg"] == 47.0
    assert rows[cells[3]["key"]]["pass_possible"] is False
    assert rows[cells[4]["key"]]["pass_possible"] is True
    assert c.criteria(res)["b"]["heldout_n_pass_possible"] == 35


def test_criteria_counts_unsatisfiable_inputs():
    res = _fake_results()
    res["c"]["sift"]["d_eff"] = 6.68                     # 0.3 d_eff < 5
    cc = c.criteria(res)["c"]
    assert cc["inputs"]["sift"]["satisfiable"] is False
    assert cc["inputs"]["mnist"]["satisfiable"] is True
    assert cc["n_inputs_satisfiable"] == 3


# ---------------------------------------------------------------- the runner, end to end

class _FakeData(c._Data):
    """Synthetic stand-in for the benchmarks and connectomes (no downloads)."""
    DIMS = {"sift": 16, "mnist": 24, "glove": 20, "odours": 12}

    @staticmethod
    def _synth(n, d, seed):
        r = np.random.default_rng(seed)
        centres = np.repeat(r.normal(size=(n // 10, d)), 10, 0)
        x = centres + r.normal(size=(n, d)) * r.uniform(0.05, 1.0, size=(n, 1))
        return np.round(np.abs(x) * 10)                  # non-negative integers

    def raw(self, name):
        if name not in self._raw:
            self._raw[name] = self._synth(self.n, self.DIMS[name], len(name))
        return self._raw[name]

    def connectome(self, ds, side):
        key = f"{ds}_{side}"
        if key not in self._raw:
            r = np.random.default_rng(len(key))
            w = (r.random((12, 150)) < 0.3).astype(np.float32)
            w = w[:, w.sum(0) > 0]
            self._raw[key] = (w, self._synth(self.n, 12, 99), [f"g{i}" for i in range(12)])
        return self._raw[key]


def _tiny_design():
    D = json.loads(json.dumps(c.DESIGN))
    D.update({"n_items": 400, "queries": 30})
    D["a"].update({"datasets": ["sift"], "trials": 2})
    D["b"].update({"heldout": ["odours", "sift"], "ks": [4, 8], "ms": ["20k", "10d"], "draws": 2,
                   "nn_per_item": 3, "random_pairs": 800, "min_pairs": 800, "min_bin": 5,
                   "hemispheres": [["malecns", "R"], ["hemibrain", "R"]],
                   "connectomes_needed": 1})
    D["c"].update({"inputs": ["sift", "odours"], "pca_dims": 10, "m": 200, "ks": [4, 8],
                   "trials": 2})
    return D


def test_run_resumes_exactly_and_never_refits_c(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "_Data", _FakeData)
    D, out = _tiny_design(), tmp_path / "e3.json"
    quiet = lambda s: None                               # noqa: E731

    # nothing is computed without budget
    r0 = c.run(None, out=out, design=D, budget_minutes=0, log=quiet)
    assert not r0["complete"] and not r0["b"]["cells"] and not r0["a"]["sift"]["trials"]

    r1 = c.run(None, out=out, design=D, log=quiet)
    assert r1["complete"] and r1["criteria"]["complete"]
    assert len(r1["b"]["cells"]) == 4 + 8 + 2
    c1 = r1["b"]["fit"]["c_deg"]
    fit = [(v["theta_half"], v["t"]) for v in r1["b"]["cells"].values()
           if v["role"] == "fit" and v["theta_half"] is not None]
    assert np.isclose(c1, c.fit_c(*zip(*fit)))

    # remove one unit of every kind and tamper with c: the units come back
    # identical and c is not refitted
    res = json.loads(out.read_text())
    gone = {k: res["b"]["cells"].pop(k) for k in ("sift|k=8|m=10d", "connectome|hemibrain_R")}
    gone_a = res["a"]["sift"]["trials"].pop("1")
    gone_c = res["c"]["odours"]["trials"].pop("0")
    res["b"]["fit"]["c_deg"] = 12.345
    out.write_text(json.dumps(res))
    r2 = c.run(None, out=out, design=D, log=quiet)
    assert r2["complete"] and r2["b"]["fit"]["c_deg"] == 12.345
    for k, v in gone.items():
        for f in ("theta_half", "theta_half_draws", "curve"):
            assert r2["b"]["cells"][k][f] == v[f]
    assert r2["a"]["sift"]["trials"]["1"]["fly"] == gone_a["fly"]
    assert r2["c"]["odours"]["trials"]["0"]["fly"] == gone_c["fly"]

    # a file from another design is refused
    D2 = _tiny_design()
    D2["a"]["trials"] = 3
    with pytest.raises(SystemExit):
        c.run(None, out=out, design=D2, log=quiet)


def test_design_matches_preregistration():
    D = c.DESIGN
    assert D["a"]["datasets"] == ["sift", "glove", "mnist"] and D["a"]["ks"] == [4, 16]
    assert D["a"]["trials"] == 10 and D["queries"] == 1000 and D["a"]["m_per_d"] == 10
    b = D["b"]
    assert b["ks"] == [4, 16, 64, 256] and b["ms"] == ["20k", "10d", "40d"]
    assert b["heldout"] == ["sift", "glove", "odours"] and b["fit_dataset"] == "mnist"
    assert b["min_pairs"] >= 1e5
    assert b["tolerance_deg"] == 3.0 and b["pass_fraction"] == 0.9
    assert [tuple(h) for h in b["hemispheres"]] == list(HEMISPHERES)
    assert len(c.b_cells(D)) == 12 + 36
    cc = D["c"]
    assert cc["m"] == 1838 and cc["ks"] == [16, 92] and cc["pca_dims"] == 51
    assert cc["trials"] >= 5


def test_smoke_cannot_overwrite_the_results_file():
    with pytest.raises(SystemExit):
        c.main(["--smoke"])
