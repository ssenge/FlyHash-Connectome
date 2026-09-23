"""The hash, the wiring, the nulls and the data handling. Needs the built graph."""

from __future__ import annotations

import numpy as np
import pytest

from flypath import config, data, flyhash as fh

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def full(cfg):
    if not (cfg.graph_dir / "nodes.parquet").exists():
        pytest.skip("no built graph; run: python -m flypath build")
    return fh.mushroom_body(data.load_graph(cfg))


@pytest.fixture(scope="module")
def odours(cfg, full):
    return fh.load_odours(cfg, full.glomeruli)


@pytest.fixture(scope="module")
def proj(full, odours):
    return fh.align(full, odours.glomeruli)


@pytest.fixture(scope="module")
def items(odours):
    return fh.mixtures(odours.x, 300, seed=0)


# ------------------------------------------------------------------ wiring

def test_glomerulus_parser():
    assert fh.glomerulus_of("DA1_lPN") == "DA1"
    assert fh.glomerulus_of("VA1d_adPN") == "VA1d"
    assert fh.glomerulus_of("M_l2PNm16") is None      # multiglomerular
    assert fh.glomerulus_of("CB2004") is None         # unnamed
    assert fh.glomerulus_of(float("nan")) is None


def test_projection_dimensions(full, proj):
    assert full.matrix.shape == (51, 1886)
    assert proj.matrix.shape == (35, 1838)
    assert proj.nnz == 7110


def test_inputs_are_distinct_glomeruli_not_claws(full, proj):
    """Distinct glomerular inputs and PN partners are reported separately."""
    assert full.inputs().mean() == pytest.approx(5.285, abs=0.01)
    assert proj.inputs().mean() == pytest.approx(3.868, abs=0.01)
    pn = proj.meta["pn_partners"]
    assert len(pn) == proj.matrix.shape[1]
    assert (pn >= proj.inputs()).all()        # aggregation can only merge partners


def test_fan_out_is_lopsided(proj):
    fan = proj.fan_out()
    assert fan.max() > 10 * fan.min()


# ------------------------------------------------------------------ nulls

@pytest.mark.parametrize("which", ["aligned", "full"])
@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_curveball_preserves_both_margins_elementwise(full, proj, which, seed):
    p = proj if which == "aligned" else full
    q = fh.curveball(p, seed=seed)
    assert np.array_equal(q.inputs(), p.inputs())      # every cell, not the sorted multiset
    assert np.array_equal(q.fan_out(), p.fan_out())    # every glomerulus
    assert not np.array_equal(q.matrix > 0, p.matrix > 0)


def test_curveball_keeps_each_cells_weights(proj):
    q = fh.curveball(proj, seed=3)
    for c in range(0, proj.matrix.shape[1], 37):
        a = np.sort(proj.matrix[:, c][proj.matrix[:, c] > 0])
        b = np.sort(q.matrix[:, c][q.matrix[:, c] > 0])
        assert np.array_equal(a, b)


def test_curveball_checkpoints(proj):
    snaps = fh.curveball(proj, seed=0, checkpoints=(0, 1, 5))
    assert [s.meta["sweeps"] for s in snaps] == [0.0, 1.0, 5.0]
    assert np.array_equal(snaps[0].matrix > 0, proj.matrix > 0)


def test_shuffle_inputs_keeps_input_counts_and_flattens_fan_out(proj):
    q = fh.shuffle_inputs(proj, seed=0)
    assert np.array_equal(q.inputs(), proj.inputs())
    assert q.fan_out().std() < proj.fan_out().std()


def test_uniform_inputs_uses_six_by_default(proj):
    q = fh.uniform_inputs(proj, seed=0)
    assert set(q.inputs().tolist()) == {6}
    with pytest.raises(ValueError):
        fh.uniform_inputs(proj, inputs=0)


def test_balanced_fanout_is_exact(proj):
    q = fh.balanced_fanout(proj, seed=0)
    assert np.array_equal(q.inputs(), proj.inputs())
    assert q.fan_out().max() - q.fan_out().min() <= 1


# ------------------------------------------------------------------ the hash

@pytest.mark.parametrize("rule", ["random", "index"])
def test_exactly_k_winners(proj, items, rule):
    for k in (8, 64, 92):
        assert (fh.tags(items, proj, k, tie=rule).sum(1) == k).all()


def test_include_rule_keeps_every_tied_cell(proj, items):
    inc = fh.tags(items, proj, 64, tie="include").sum(1)
    assert (inc >= 64).all()


def test_tie_rule_on_a_constructed_tie():
    """Cells 0 and 1 have identical inputs, so their drive is always equal."""
    m = np.array([[1, 1, 0], [0, 0, 1]], float)
    p = fh.Projection("t", m, ["a", "b"])
    x = np.array([[3.0, 1.0]])                  # cells 0 and 1 tie above cell 2
    assert fh.tags(x, p, 1, tie="index").tolist() == [[True, False, False]]
    assert fh.tags(x, p, 1, tie="include").tolist() == [[True, True, False]]
    picks = {tuple(fh.tags(x, p, 1, tie="random", tie_seed=s)[0]) for s in range(20)}
    assert picks == {(True, False, False), (False, True, False)}


def test_tags_are_deterministic_under_a_seed(proj, items):
    a = fh.tags(items, proj, 32, tie="random", tie_seed=5)
    b = fh.tags(items, proj, 32, tie="random", tie_seed=5)
    assert np.array_equal(a, b)


def test_normalisation_removes_concentration(proj, items):
    assert np.array_equal(fh.tags(items, proj, 16), fh.tags(items * 7.5, proj, 16))


def test_storage_bits_exact():
    assert fh.storage_bits(4, 2) == 3        # C(4,2)=6
    assert fh.storage_bits(5, 1) == 3        # C(5,1)=5
    assert fh.storage_bits(8, 4) == 7        # C(8,4)=70
    assert fh.storage_bits(1838, 64) == 397


def test_computation_bits(proj):
    assert fh.computation_bits(proj) == round(proj.nnz / (2 * proj.matrix.shape[0]))


# ------------------------------------------------------------------ retrieval

def test_hamming_ties_follow_the_priority():
    tag = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], bool)
    truth = np.array([[1], [0], [0], [0]])
    pri = np.tile(np.arange(4, dtype=np.int32), (4, 1))      # index order
    assert fh.mean_average_precision(tag, truth, pri, per_item=True)[0] == 1.0
    pri[0] = [0, 3, 1, 2]                                     # item 2 now preferred
    assert fh.mean_average_precision(tag, truth, pri, per_item=True)[0] == 0.0


def test_map_bounded_and_better_than_random_codes(proj, items):
    truth = fh.true_neighbours(items, 10)
    pri = fh.retrieval_priority(len(items), 0)
    s = fh.mean_average_precision(fh.tags(items, proj, 32), truth, pri)
    rnd = np.random.default_rng(0).random((len(items), proj.matrix.shape[1])) < 0.02
    assert 0.0 <= s <= 1.0
    assert s > fh.mean_average_precision(rnd, truth, pri)


def test_metrics_differ_but_share_the_self_exclusion(items):
    for metric in ("euclidean", "normalised", "angular"):
        t = fh.true_neighbours(items, 5, metric)
        assert not (t == np.arange(len(items))[:, None]).any()


# ------------------------------------------------------------------ odour data

def test_door_is_pinned():
    assert fh.DOOR_COMMIT in fh.DOOR_BASE and "master" not in fh.DOOR_BASE


def test_missing_fraction_is_reported(odours):
    assert odours.x.shape == (172, 35)
    assert odours.missing_fraction == pytest.approx(0.287, abs=0.001)


@pytest.mark.parametrize("how", ["glomerulus_mean", "odour_mean", "lowrank"])
def test_imputation_changes_only_unmeasured_entries(cfg, full, odours, how):
    alt = fh.load_odours(cfg, full.glomeruli, missing=how)
    assert alt.names == odours.names
    obs = odours.observed
    assert np.allclose(alt.x[obs], odours.x[obs])
    assert not np.allclose(alt.x[~obs], 0.0)
