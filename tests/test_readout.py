"""E2 (flypath.readout): scorers, budgets, estimators and the criteria logic
on small hand-checkable examples."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from flypath import flyhash as fh
from flypath import readout as ro
from flypath import replication as rp


# ------------------------------------------------------------------ budgets

def test_information_theoretic_budget_is_exact():
    assert ro.info_bits(4, 2) == 3            # C = 6
    assert ro.info_bits(5, 2) == 4            # C = 10
    assert ro.info_bits(8, 1) == 3            # C = 8, a power of two
    assert ro.info_bits(9, 1) == 4
    for m, k in ((1280, 2), (3000, 16), (7840, 32), (350, 32)):
        c, b = math.comb(m, k), ro.info_bits(m, k)
        assert 2 ** (b - 1) < c <= 2 ** b


def test_fixed_width_budget_and_value_tag_winners():
    assert ro.index_bits(8) == 3 and ro.index_bits(9) == 4 and ro.index_bits(1280) == 11
    assert ro.fixed_bits(9, 2) == 8
    assert ro.budget("fixed", 7840, 4) == 4 * 13
    assert ro.value_tag_winners(23, 3000) == 1              # 23 // (12 + 4)
    assert ro.value_tag_winners(252, 3000) == 15
    assert ro.value_tag_winners(15, 3000) == 0


def test_value_tag_has_a_winner_at_every_protocol_budget():
    for d in (128, 300, 784, 35):             # sift, glove, mnist, odours
        m = 10 * d
        for k in ro.KS:
            for acc in ro.ACCOUNTING:
                assert ro.value_tag_winners(ro.budget(acc, m, k), m) >= 1


def test_pq_layout_spends_exactly_the_budget():
    per, chunks = ro.pq_layout(23, 300)
    assert per == [8, 8, 7]
    assert np.array_equal(np.concatenate(chunks), np.arange(300))
    per, chunks = ro.pq_layout(288, 35)       # more bits than 8 per dimension
    assert len(per) == 35 and sum(per) == 288 and set(per) == {8, 9}
    assert all(len(c) == 1 for c in chunks)


# ------------------------------------------------------------------ scorers

def test_cell_stats_and_standardise():
    y = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 7.0]], np.float32)
    mu, sd = ro.cell_stats(y, chunk=1)
    assert np.allclose(mu, [2, 2, 6]) and np.allclose(sd, [1, 0, 1])
    z = ro.standardise(np.array([[4.0, 9.0, 6.0]]), mu, sd)
    assert np.allclose(z, [[2.0, 0.0, 0.0]])  # the constant cell contributes 0


def test_fly_asymmetric_score_sums_query_drive_over_item_winners():
    idx = np.array([[0, 2], [1, 2], [0, 2]])
    u = np.array([[1.0, 10.0, 100.0], [0.0, -1.0, 1.0]])
    assert np.allclose(ro.set_scores(idx, u), [[101, 110, 101], [1, 0, 1]])


def test_fly_asymmetric_score_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 20)).astype(np.float32)
    t = rp.winners(y, 3)
    t.sort_indices()
    idx = t.indices.reshape(30, 3)
    u = rng.normal(size=(4, 20))
    brute = np.array([[u[a, idx[j]].sum() for j in range(30)] for a in range(4)])
    assert np.allclose(ro.set_scores(idx, u), brute)
    # the same as graded query . binary tag
    assert np.allclose(ro.set_scores(idx, u), u @ t.toarray().T)


def test_sign_code_asymmetric_score():
    bits = np.array([[1, 0], [0, 1], [1, 1]], bool)
    u = np.array([[2.0, -1.0]])
    assert np.allclose(ro.sign_scores(bits, u), [[3, -3, 1]])
    # ranks as u . b does (they differ by a per-query constant)
    assert np.array_equal(np.argsort(ro.sign_scores(bits, u)[0]), np.argsort(u @ bits.T)[0])


def test_quantiser():
    v = np.arange(17.0)
    lev, rec, rng_ = ro.quantise(v, 4)
    assert rng_ == [0.0, 16.0]
    assert np.array_equal(lev, np.r_[np.arange(16), 15])
    assert np.allclose(rec, lev + 0.5)
    assert np.abs(rec - v).max() <= 0.5
    lev, rec, _ = ro.quantise(np.full(5, 3.0))
    assert (lev == 0).all() and (rec == 3.0).all()


def test_value_tag_codes_scores_and_distances():
    y = np.array([[5, 1, 3, 0], [0, 4, 1, 2]], np.float32)
    idx, lev, rec = ro.value_tag(y, 2)
    assert idx.tolist() == [[0, 2], [1, 3]]
    w = 3 / 16                                  # range [2, 5], 16 levels
    assert lev.tolist() == [[15, 5], [10, 0]]
    assert np.allclose(rec, 2 + (lev + 0.5) * w)
    s = ro.value_scores(idx, lev, rec, np.array([[1.0, 0, 0, 0], [0, 1.0, 1.0, 0]]))
    assert np.allclose(s, [[rec[0, 0], 0.0], [rec[0, 1], rec[1, 0]]])
    u = np.array([[1.0, 2.0, 0.0, -1.0]])
    adc = ro.value_adc(idx, lev, rec, u)
    dense = np.zeros((2, 4))
    np.put_along_axis(dense, idx, rec, 1)
    assert np.allclose(adc, ((u[:, None, :] - dense[None]) ** 2).sum(-1))
    dist = ro.value_distances(idx, lev, rec, np.array([0]), 4)
    assert np.isclose(dist[0, 0], 0.0)
    assert np.isclose(dist[0, 1], (rec ** 2).sum())   # disjoint supports


def test_pq_adc_with_a_hand_codebook():
    books = [np.array([[0.0], [10.0]]), np.array([[1.0, 1.0], [-1.0, -1.0]])]
    codes = np.array([[0, 0], [1, 1], [0, 0]])
    d = ro.pq_adc(np.array([[0.0, 1.0, 1.0]]), codes, books)
    assert np.allclose(d, [[0.0, 108.0, 0.0]])


def test_pq_fit_uses_the_budget():
    x = np.random.default_rng(0).normal(size=(300, 12))
    codes, books = ro.pq_fit(x, 23, np.random.default_rng(1), iters=5)
    per, chunks = ro.pq_layout(23, 12)
    assert codes.shape == (300, 3)
    for s, (b, cols) in enumerate(zip(per, chunks)):
        assert books[s].shape == (2 ** b, len(cols))
        assert codes[:, s].min() >= 0 and codes[:, s].max() < 2 ** b
    rec = np.hstack([books[s][codes[:, s]] for s in range(3)])
    assert np.allclose(ro.pq_adc(x[:2], codes, books), rp._sqdist(x[:2], rec))


# ------------------------------------------------------------------ ranking

def test_rank_scores_orders_excludes_self_and_breaks_ties_at_random():
    s = np.array([[3.0, 1.0, 3.0, 2.0, 0.0]])
    firsts = set()
    for seed in range(40):
        r = ro.rank_scores(s, np.array([1]), 3, np.random.default_rng(seed))[0]
        assert set(r[:2]) == {0, 2} and r[2] == 3
        firsts.add(int(r[0]))
    assert firsts == {0, 2}
    r = ro.rank_scores(np.array([[9.0, 1.0, 2.0]]), np.array([0]), 2, np.random.default_rng(0))
    assert r.tolist() == [[2, 1]]


def test_rank_scores_with_all_scores_tied():
    s = np.zeros((2, 6))
    r = ro.rank_scores(s, np.array([0, 1]), 5, np.random.default_rng(0))
    assert sorted(r[0]) == [1, 2, 3, 4, 5] and sorted(r[1]) == [0, 2, 3, 4, 5]


def test_rank_scores_matches_a_full_sort():
    rng = np.random.default_rng(4)
    for levels in (3, 50, None):              # heavy ties, some ties, none
        s = rng.normal(size=(7, 400)) if levels is None else rng.integers(0, levels, (7, 400)).astype(float)
        q = rng.choice(400, 7, replace=False)
        r = ro.rank_scores(s, q, 30, np.random.default_rng(1))
        ref = s.copy()
        ref[np.arange(7), q] = -np.inf
        best = -np.sort(-ref, axis=1)[:, :30]
        assert np.array_equal(np.take_along_axis(s, r, 1), best)   # right scores, right order
        assert all(len(set(row)) == 30 and qq not in row for row, qq in zip(r.tolist(), q))


def test_rank_hamming_matches_rank_binary():
    bits = np.random.default_rng(3).random((60, 9)) > 0.5
    q = np.array([0, 5, 7])
    a = ro.rank_hamming(bits, q, 10, np.random.default_rng(11))
    b = rp.rank_binary(sp.csr_matrix(bits, dtype=np.float32), q, 10, np.random.default_rng(11))
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ one trial

def test_codes_trial_runs_and_is_deterministic():
    x = np.abs(np.random.default_rng(5).normal(size=(300, 12)))
    ks = (2, 4)
    r = ro.codes_trial(x, 20, ks, (7, 0))
    names = {"fly_sym", "fly_asym_raw", "fly_asym_std"}
    for acc in ("info", "fixed"):
        for f in ro.ASYM_FAMILIES:
            names |= {f"{f}_{acc}_sym", f"{f}_{acc}_asym_raw", f"{f}_{acc}_asym_std"}
        names |= {f"pq_{acc}_adc", f"vtag_{acc}_adc"}
    assert set(r["ap"]) == names == set(r["recall"])
    for nm in names:
        vals = [v for v in r["ap"][nm] if v is not None]
        assert len(r["ap"][nm]) == len(ks) and all(0.0 <= v <= 1.0 for v in vals)
    assert r == ro.codes_trial(x, 20, ks, (7, 0))
    v = ro.codes_trial(x, 20, ks, (7, 1), accountings=("info",), pq=False)
    assert not any(nm.startswith("pq") or "_fixed_" in nm for nm in v["ap"])


def test_validation_choice():
    def tr(raw, std):
        ap = {"fly_asym_raw": raw, "fly_asym_std": std}
        for f in ro.ASYM_FAMILIES:
            ap[f"{f}_info_asym_raw"] = [0.5, 0.5]
            ap[f"{f}_info_asym_std"] = [0.4, None]
        return {"ap": ap}
    choice, means = ro.choose({"0": tr([0.1, 0.2], [0.2, 0.2]), "1": tr([0.1, 0.2], [0.1, 0.2])})
    assert choice["fly"] == "std" and np.isclose(means["fly"]["std"], 0.175)
    assert choice["gauss_info"] == "raw"
    choice, _ = ro.choose({"0": tr([0.3, 0.1], [0.1, 0.3])})     # equal -> raw
    assert choice["fly"] == "raw"


# ------------------------------------------------------------------ estimators

def test_ratio_ci_constant_trials():
    r = ro.ratio_ci(np.full(5, 2.0), np.array([np.ones(5), np.full(5, 0.5)]), draws=200)
    assert np.isclose(r["ratio"], 2.0) and np.allclose(r["ci95"], [2.0, 2.0]) and r["best"] == 0


def test_ratio_ci_retakes_the_best_competitor_in_every_resample():
    fly = np.ones(4)
    comps = np.array([[2.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 2.0]])
    r = ro.ratio_ci(fly, comps, draws=500)
    assert np.isclose(r["ratio"], 1.0)
    assert r["ci95"][1] <= 1.0 + 1e-12      # max of the two means is >= 1 in any resample


def test_competitor_set():
    choice = {"fly": "std", "gauss_info": "raw", "densefly_info": "std", "vtag_info": "std"}
    # primary: every competitor's asymmetric scorer uses its own validated query variant
    assert ro.competitor_set("info", "primary", choice) == [
        "fly_sym", "gauss_info_sym", "gauss_info_asym_raw", "densefly_info_sym",
        "densefly_info_asym_std", "vtag_info_sym", "vtag_info_asym_std", "vtag_info_adc"]
    with pytest.raises(ValueError):
        ro.competitor_set("info", "primary")
    assert ro.competitor_set("info", "raw_asym") == [
        "fly_sym", "gauss_info_sym", "gauss_info_asym_raw", "densefly_info_sym",
        "densefly_info_asym_raw", "vtag_info_sym", "vtag_info_asym_raw"]
    ext = ro.competitor_set("fixed", "all_variants")
    assert {"gauss_fixed_asym_std", "gauss_fixed_asym_raw", "vtag_fixed_adc",
            "densefly_fixed_asym_raw"} <= set(ext)
    for table in ro.COMPARISONS:
        cs = ro.competitor_set("info", table, choice)
        assert not any(c.startswith(("pq", "fly_asym")) for c in cs) and "fly_sym" in cs
        assert set(cs) <= set(ro.competitor_set("info", "all_variants"))


def test_validation_chooses_for_every_asymmetric_scorer_and_accounting():
    x = np.abs(np.random.default_rng(6).normal(size=(250, 10)))
    val = {str(v): ro.codes_trial(x, 20, (2, 4), (ro.SEED_VAL, v), pq=False) for v in range(2)}
    choice, means = ro.choose(val)
    assert set(choice) == {"fly", *(f"{f}_{a}" for f in ro.ASYM_FAMILIES for a in ro.ACCOUNTING)}
    assert set(choice.values()) <= set(ro.VARIANTS)
    for fam, mm in means.items():
        pre = "fly_asym" if fam == "fly" else f"{fam}_asym"
        for var in ro.VARIANTS:
            vals = [v for tr in val.values() for v in tr["ap"][f"{pre}_{var}"]]
            assert np.isclose(mm[var], np.mean(vals))
        assert choice[fam] == ("std" if mm["std"] > mm["raw"] else "raw")


def _fake_dataset(fly_ap, comp_ap, n_trials=4, ks=ro.KS, choice="std"):
    rng = np.random.default_rng(0)
    trials = {}
    for t in range(n_trials):
        ap = {"fly_asym_raw": [0.01] * len(ks), "fly_asym_std": [0.01] * len(ks)}
        ap[f"fly_asym_{choice}"] = [a + 1e-4 * rng.random() for a in fly_ap]
        ap["fly_sym"] = [0.001] * len(ks)
        for acc in ("info", "fixed"):
            for f in ro.ASYM_FAMILIES:
                for sc in ("sym", "asym_raw", "asym_std"):
                    ap[f"{f}_{acc}_{sc}"] = [0.002] * len(ks)
            ap[f"pq_{acc}_adc"] = [0.9] * len(ks)
            ap[f"vtag_{acc}_adc"] = [0.002] * len(ks)
        ap["densefly_info_asym_raw"] = [c + 1e-4 * rng.random() for c in comp_ap]
        trials[str(t)] = {"ap": ap, "recall": ap}
    ch = {"fly": choice, **{f"{f}_{a}": "raw" for f in ro.ASYM_FAMILIES for a in ro.ACCOUNTING}}
    return {"validation": {"choice": ch},
            "bits": {"info": [10] * len(ks), "fixed": [12] * len(ks)}, "trials": trials}


def _settings(ks=ro.KS):
    return {"ks": list(ks), "trials": 4, "n_items": 10, "n_queries": 5, "datasets": ["glove", "mnist"],
            "val_items": 1, "val_queries": 1, "conn_ks": [4, 16], "conn_trials": 1, "conn_nulls": 1,
            "conn_datasets": []}


def test_analysis_and_primary_criteria():
    ks = ro.KS
    glove = _fake_dataset([0.5, 0.5, 0.5, 0.1, 0.5], [0.2] * 5)       # fly loses at k = 16 only
    mnist = _fake_dataset([0.5, 0.1, 0.5, 0.5, 0.5], [0.2] * 5)       # fly loses at k = 4
    res = {"settings": {"ks": list(ks), "trials": 4, "n_items": 10, "n_queries": 5,
                        "datasets": ["glove", "mnist"], "val_items": 1, "val_queries": 1,
                        "conn_ks": [4, 16], "conn_trials": 1, "conn_nulls": 1, "conn_datasets": []},
           "datasets": {"glove": glove, "mnist": mnist}}
    out = ro.analyse(res)
    rows = out["datasets"]["glove"]["comparisons"]["info"]["primary"]
    assert [r["lower_ge_1"] for r in rows] == [True, True, True, False, True]
    assert rows[0]["best_competitor"] == "densefly_info_asym_raw"
    assert out["datasets"]["glove"]["fly_readout"] == "fly_asym_std"
    cr = out["criteria"]
    assert cr["a_glove"]["pass"] and cr["a_glove"]["n_passing"] == 4
    assert cr["b_mnist"]["k2"] and not cr["b_mnist"]["k4"] and not cr["b_mnist"]["pass"]
    assert cr["primary_pass"] is False and cr["protocol_as_preregistered"] is False
    assert cr["all_competitors_present"] is True
    assert "pq_reference" in out["datasets"]["glove"]["comparisons"]["info"]
    assert set(out["datasets"]["glove"]["comparisons"]["info"]) == {*ro.COMPARISONS, "pq_reference"}


def test_primary_set_uses_the_competitors_validated_variant():
    ks = ro.KS
    ds = _fake_dataset([0.5] * 5, [0.2] * 5)                      # densefly raw far below the fly
    for tr in ds["trials"].values():
        tr["ap"]["densefly_info_asym_std"] = [0.8] * 5            # its std variant beats the fly
    res = {"settings": _settings(), "datasets": {"glove": ds}}
    tabs = ro.analyse(res)["datasets"]["glove"]["comparisons"]["info"]
    assert all(r["lower_ge_1"] for r in tabs["primary"])          # validation chose raw
    assert not any(r["lower_ge_1"] for r in tabs["all_variants"])
    ds["validation"]["choice"]["densefly_info"] = "std"           # validation chose std
    tabs = ro.analyse(res)["datasets"]["glove"]["comparisons"]["info"]
    assert not any(r["lower_ge_1"] for r in tabs["primary"])
    assert {r["best_competitor"] for r in tabs["primary"]} == {"densefly_info_asym_std"}
    assert all(r["lower_ge_1"] for r in tabs["raw_asym"])         # the raw-only comparison


def test_value_tag_adc_is_a_primary_competitor_and_missing_ones_are_flagged():
    ks = ro.KS
    ds = _fake_dataset([0.5] * 5, [0.2] * 5)
    for tr in ds["trials"].values():
        tr["ap"]["vtag_info_adc"] = [0.9] * 5
        tr["ap"]["gauss_info_sym"] = [0.002, None, 0.002, 0.002, 0.002]
    res = {"settings": _settings(), "datasets": {"glove": ds}}
    out = ro.analyse(res)
    rows = out["datasets"]["glove"]["comparisons"]["info"]["primary"]
    assert {r["best_competitor"] for r in rows} == {"vtag_info_adc"}
    assert rows[1]["missing_competitors"] == ["gauss_info_sym"] and rows[0]["missing_competitors"] == []
    assert out["criteria"]["all_competitors_present"] is False


def test_connectome_analysis_inside_rule():
    def trial(real, null, even):
        return {"real": {r: [real, real] for r in ro.READOUTS},
                "null_mean": {r: [null, null] for r in ro.READOUTS},
                "out_equal": {r: [even, even] for r in ro.READOUTS}}
    cn = {"ks": [4, 16], "trials": {"0": trial(0.30, 0.31, 0.33), "1": trial(0.29, 0.30, 0.31),
                                    "2": trial(0.31, 0.31, 0.32)}}
    a = ro.analyse_connectome(cn, "std")
    assert a["dataset_passes"] and len(a["rows"]) == 4       # identical readouts: inside
    cn["trials"]["0"]["out_equal"]["asym_std"] = [0.60, 0.60]
    cn["trials"]["1"]["out_equal"]["asym_std"] = [0.60, 0.60]
    cn["trials"]["2"]["out_equal"]["asym_std"] = [0.60, 0.60]
    a = ro.analyse_connectome(cn, "std")
    assert not a["dataset_passes"]
    assert ro.analyse_connectome(cn, "raw")["dataset_passes"]


def test_connectome_trial_on_a_small_wiring():
    rng = np.random.default_rng(2)
    g, m = 6, 40
    w = np.zeros((g, m))
    for c in range(m):
        w[rng.choice(g, 2, replace=False), c] = 1.0
    pb = fh.Projection("b", w, [f"g{i}" for i in range(g)])
    nulls = [fh.curveball(pb, seed=s, sweeps=3).matrix.astype(np.float32) for s in range(2)]
    z = rng.normal(size=(200, g))
    r = ro.connectome_trial("sift", z, w.astype(np.float32), nulls, pb, 0, (2, 4), 20)
    for kind in ("real", "out_equal", "null_mean"):
        for rd in ro.READOUTS:
            assert len(r[kind][rd]) == 2 and all(0 <= v <= 1 for v in r[kind][rd])
    assert len(r["null"]["sym"]) == 2


@pytest.mark.parametrize("name", ["sift", "odours"])
def test_connectome_symmetric_readout_reproduces_the_papers_trial_controls(name):
    """Same seeds, queries, permutations, tie draws and even-fan-out draw as
    replication.trial_controls: the symmetric AP must be identical."""
    rng = np.random.default_rng(8)
    g, m = 8, 60
    w = np.zeros((g, m))
    for c in range(m):
        w[rng.choice(g, int(rng.integers(1, 4)), replace=False), c] = 1.0
    pb = fh.Projection("b", w, [f"g{i}" for i in range(g)])
    pool = [fh.curveball(pb, seed=s, sweeps=3) for s in range(3)]
    x = np.abs(rng.normal(size=(1100, g)))           # trial_controls draws 1,000 queries
    paper = rp.trial_controls(name, x, pb, pool, 2, log=lambda s: None)
    z = x if name == "odours" else rp.pca(x, g)
    nulls = [p.matrix.astype(np.float32) for p in pool]
    ks = (4, 16)
    cols = [paper["sizes"].index(k) for k in ks]
    for t in range(2):
        r = ro.connectome_trial(name, z, w.astype(np.float32), nulls, pb, t, ks, rp.N_QUERIES)
        assert np.array_equal(r["real"]["sym"], np.array(paper["real"][t])[cols])
        assert np.allclose(r["null_mean"]["sym"], np.array(paper["null"][t])[cols], rtol=0, atol=1e-15)
        assert np.array_equal(r["out_equal"]["sym"], np.array(paper["out_equal"][t])[cols])


def test_paper_check():
    trials = {str(t): {"real": {"sym": [0.1 + t, 0.2]}, "null_mean": {"sym": [0.3, 0.4]},
                       "out_equal": {"sym": [0.5, 0.6]}} for t in range(2)}
    cn = {"ks": [4, 16], "n_nulls": 10, "trials": trials}
    paper = {"trials": 2, "B": 10, "sizes": [2, 4, 8, 16],
             "real": [[9, 0.1, 9, 0.2], [9, 1.1, 9, 0.2]], "null": [[9, 0.3, 9, 0.4]] * 2,
             "out_equal": [[9, 0.5, 9, 0.6]] * 2}
    c = ro.paper_check(cn, paper)
    assert c["available"] and c["identical"]
    paper["out_equal"] = [[9, 0.5, 9, 0.61]] * 2
    c = ro.paper_check(cn, paper)
    assert not c["identical"] and np.isclose(c["max_abs_diff"]["out_equal"], 0.01)
    assert not ro.paper_check({**cn, "n_nulls": 20}, paper)["available"]
    assert not ro.paper_check(cn, None)["available"]


# ------------------------------------------------------------------ data (optional)

@pytest.mark.data
@pytest.mark.parametrize("name", ["sift", "mnist"])
def test_validation_items_are_disjoint_from_test_items(name):
    from flypath import config
    cfg = config.load()
    try:
        x = rp.load_benchmark(cfg, name, n=500)
    except Exception as e:                   # no downloaded benchmark data
        pytest.skip(f"benchmark data unavailable: {e}")
    xv, info = ro.validation_set(cfg, name, 50, 500, x)
    assert xv.shape == (50, x.shape[1]) and info["test_subset_reproduced"]
    keys = {r.tobytes() for r in x}
    assert not any(r.tobytes() in keys for r in xv)
