"""F1 weak-connection audit: thresholding, null seeding and checkpoint resume."""

import json

import numpy as np
import pytest

from flypath import experiments as ex
from flypath import f1_audit as F
from flypath import flyhash as fh


def toy(seed=0, n_g=6, n_c=40):
    rng = np.random.default_rng(seed)
    m = rng.integers(1, 7, (n_g, n_c)) * (rng.random((n_g, n_c)) < 0.35)
    m[:, 0] = 0                                            # a cell without any input
    m[:, 1] = 0
    m[0, 1] = 1                                            # a cell with one weak connection
    return fh.Projection("real", m.astype(float), [f"G{i}" for i in range(n_g)])


def loader(cfg, ds, side):
    return toy(seed={"R": 0, "L": 1}[side])


HEMIS = [("toy", "R"), ("toy", "L")]


def test_threshold_binarises_glomerulus_weights_and_drops_empty_cells():
    p = fh.Projection("real", np.array([[1., 0, 3, 2], [2, 0, 0, 1], [0, 0, 5, 0]]),
                      ["A", "B", "C"])
    b = F.threshold(p, 2)
    np.testing.assert_array_equal(b.matrix, [[0, 1, 1], [1, 0, 0], [0, 1, 0]])
    assert b.meta["cells_kept"].tolist() == [True, False, True, True]
    d = F.describe(p, b)
    assert (d["n_cells"], d["n_cells_dropped"], d["nnz"]) == (3, 1, 4)
    assert d["single_input_cells"] == 2
    assert d["connections_retained"] == pytest.approx(4 / 6)
    assert d["synapses_retained"] == pytest.approx(12 / 14)
    assert F.threshold(p, 1).matrix.shape == (3, 3)       # w >= 1 is the plain binary matrix


def test_null_q_matches_null_pool_seeding():
    b = F.threshold(toy(), 2)
    seed, q = F.null_q((b.binary.astype(np.uint8), 10_003, 30))
    ref = fh.curveball(fh.Projection("b", b.binary, b.glomeruli), seed=10_003, sweeps=30)
    assert seed == 10_003
    assert q == ex._cooccurrence_q(fh.Projection("c", (ref.matrix > 0).astype(float), b.glomeruli))
    assert (ref.matrix > 0).sum(0).tolist() == b.inputs().tolist()


def test_resumed_run_equals_one_shot_run(tmp_path):
    kw = dict(B=6, thresholds=(1, 3), hemispheres=HEMIS, load=loader, workers=1, chunk=2,
              log=lambda *_: None)
    one = F.run(out=tmp_path / "one.json", max_seconds=1e9, **kw)
    assert one["complete"]
    calls = 0
    while True:
        calls += 1
        r = F.run(out=tmp_path / "part.json", max_seconds=0, **kw)   # one chunk per call
        if r["complete"]:
            break
        assert calls < 50
    assert calls == 2 * 2 * 3                              # hemispheres x thresholds x chunks
    part = json.loads((tmp_path / "part.json").read_text())
    for k in ("toy_R", "toy_L"):
        for w in ("1", "3"):
            a, b = one["hemispheres"][k]["thresholds"][w], part["hemispheres"][k]["thresholds"][w]
            assert a["q_null"] == b["q_null"] and a["z"] == b["z"]
            null = np.array(a["q_null"])
            assert a["p_upper"] == (1 + (null >= a["q_real"]).sum()) / 7
            assert a["z"] == pytest.approx((a["q_real"] - null.mean()) / null.std(ddof=1))
    assert set(one["table"]["3"]) == {"toy_R", "toy_L", "n_departing"}


def test_resume_refuses_a_changed_matrix(tmp_path):
    kw = dict(B=4, thresholds=(2,), hemispheres=[("toy", "R")], workers=1, chunk=2,
              log=lambda *_: None, out=tmp_path / "x.json", max_seconds=0)
    F.run(load=loader, **kw)
    with pytest.raises(ValueError, match="matrix changed"):
        F.run(load=lambda cfg, ds, side: toy(seed=5), **kw)


def test_resume_refuses_to_shrink_the_holm_family(tmp_path):
    kw = dict(B=4, thresholds=(1,), load=loader, workers=1, chunk=4, log=lambda *_: None,
              out=tmp_path / "x.json", max_seconds=1e9)
    full = F.run(hemispheres=HEMIS, **kw)
    assert full["complete"] and full["holm_family"] == ["toy_R", "toy_L"]
    before = (tmp_path / "x.json").read_text()
    with pytest.raises(ValueError, match="not requested"):
        F.run(hemispheres=[("toy", "R")], **kw)                # e.g. --only toy_R
    assert (tmp_path / "x.json").read_text() == before


def test_resumed_checkpoint_is_incomplete_until_finished(tmp_path):
    kw = dict(thresholds=(1,), hemispheres=HEMIS, load=loader, workers=1, chunk=2,
              log=lambda *_: None, out=tmp_path / "x.json")
    assert F.run(B=4, max_seconds=1e9, **kw)["complete"]
    r = F._load(tmp_path / "x.json", 8, (1,), HEMIS)          # extending B
    assert r["complete"] is False and "table" not in r and "hemibrain_statement" not in r
    part = F.run(B=8, max_seconds=0, **kw)
    assert not part["complete"] and "table" not in part
    while not F.run(B=8, max_seconds=0, **kw)["complete"]:
        pass
    done = json.loads((tmp_path / "x.json").read_text())
    assert done["complete"] and all(len(c["q_null"]) == 8 and c["B"] == 8
                                    for h in done["hemispheres"].values()
                                    for c in h["thresholds"].values())


def test_finish_applies_the_hemibrain_rule_at_w2_and_holm_per_threshold():
    def cell(p_up, p_lo):
        return {"z": 0.0, "p_upper": p_up, "p_lower": p_lo, "q_null": []}

    def res(hb_w2):
        return {"hemispheres": {
            "hemibrain_R": {"thresholds": {"1": cell(0.49, 0.51), "2": hb_w2}},
            "flywire_R": {"thresholds": {"1": cell(0.001, 1.0), "2": cell(0.001, 1.0)}}}}

    hs = [("hemibrain", "R"), ("flywire", "R")]
    r = res(cell(0.04, 0.97))
    F.finish(r, hs, (1, 2))
    st = r["hemibrain_statement"]
    assert st["departs_at_w2"] and st["revise_paper_statement"] and st["p_upper_at_w2"] == 0.04
    assert r["table"]["1"]["n_departing"] == 1 and r["table"]["2"]["n_departing"] == 2
    assert not r["table"]["1"]["hemibrain_R"]["departs"]
    # Holm over the two hemispheres within w >= 2: 0.001 -> 0.002, 0.04 -> max(0.002, 0.04)
    assert r["hemispheres"]["hemibrain_R"]["thresholds"]["2"]["p_upper_holm"] == pytest.approx(0.04)
    assert r["hemispheres"]["flywire_R"]["thresholds"]["2"]["p_upper_holm"] == pytest.approx(0.002)
    assert r["holm_family"] == ["hemibrain_R", "flywire_R"]

    r = res(cell(0.06, 0.95))                                  # not below alpha at w >= 2
    F.finish(r, hs, (1, 2))
    assert not r["hemibrain_statement"]["revise_paper_statement"]

    r = res(cell(0.99, 0.01))                                  # lower tail is reported, not the rule
    F.finish(r, hs, (1, 2))
    st = r["hemibrain_statement"]
    assert not st["revise_paper_statement"] and st["reported_not_rule"]["2"]["lower_tail_below_alpha"]
