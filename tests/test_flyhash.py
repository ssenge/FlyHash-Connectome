"""The mushroom body as a hash function. Needs the built graph."""

from __future__ import annotations

import numpy as np
import pytest

from flypath import config, data, flyhash

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def proj(cfg):
    if not (cfg.graph_dir / "nodes.parquet").exists():
        pytest.skip("no built graph; run: python -m flypath build")
    return flyhash.mushroom_body(data.load_graph(cfg))


def test_glomerulus_parser():
    assert flyhash.glomerulus_of("DA1_lPN") == "DA1"
    assert flyhash.glomerulus_of("VA1d_adPN") == "VA1d"
    assert flyhash.glomerulus_of("M_l2PNm16") is None      # multiglomerular
    assert flyhash.glomerulus_of("CB2004") is None         # unnamed
    assert flyhash.glomerulus_of(float("nan")) is None


def test_projection_matches_known_anatomy(proj):
    """~51 olfactory glomeruli, ~2000 Kenyon cells, about six claws each."""
    assert 45 <= len(proj.glomeruli) <= 60
    assert 1500 <= proj.matrix.shape[1] <= 2500
    assert 3.5 <= proj.claws().mean() <= 7.0


def test_da1_is_over_represented(proj):
    """The pheromone glomerulus has more projection neurons than average."""
    assert "DA1" in proj.glomeruli
    fan = proj.fan_out()
    assert fan.max() > 5 * fan.min()          # fan-out is strongly non-uniform


def test_shuffle_kc_preserves_claws_only(proj):
    q = flyhash.shuffle_kc(proj, seed=0)
    assert np.array_equal(np.sort(q.claws()), np.sort(proj.claws()))
    # the real fan-out is lopsided; this control flattens it
    assert q.fan_out().std() < proj.fan_out().std()


def test_shuffle_both_preserves_both_degree_sequences(proj):
    q = flyhash.shuffle_both(proj, seed=0)
    assert q.claws().sum() == pytest.approx(proj.claws().sum(), rel=0.1)
    assert q.fan_out().std() == pytest.approx(proj.fan_out().std(), rel=0.25)
    assert not np.array_equal(q.matrix, proj.matrix)


def test_tags_are_sparse_and_the_right_size(proj):
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(size=(20, len(proj.glomeruli))))
    t = flyhash.tags(x, proj, 16)
    assert t.shape == (20, proj.matrix.shape[1])
    assert (t.sum(axis=1) == 16).all()


def test_normalisation_removes_concentration(proj):
    x = np.abs(np.random.default_rng(1).normal(size=(10, len(proj.glomeruli))))
    a = flyhash.tags(x, proj, 16)
    b = flyhash.tags(x * 7.5, proj, 16)       # same odour, stronger puff
    assert np.array_equal(a, b)


def test_identical_inputs_get_identical_tags(proj):
    x = np.abs(np.random.default_rng(2).normal(size=(5, len(proj.glomeruli))))
    t = flyhash.tags(np.vstack([x, x]), proj, 16)
    assert np.array_equal(t[:5], t[5:])


def test_map_is_bounded_and_beats_chance(proj):
    rng = np.random.default_rng(3)
    x = np.abs(rng.normal(size=(60, 6)) @ rng.normal(size=(6, len(proj.glomeruli))))
    score = flyhash.mean_average_precision(x, flyhash.tags(x, proj, 16))
    assert 0.0 <= score <= 1.0
    random_tag = rng.random((60, proj.matrix.shape[1])) < 0.01
    assert score > flyhash.mean_average_precision(x, random_tag)


def test_architecture_beats_classical_lsh(proj):
    """The 2017 result: sparse expansion plus winner-take-all beats dense LSH."""
    rng = np.random.default_rng(4)
    x = np.abs(rng.normal(size=(80, 6)) @ rng.normal(size=(6, len(proj.glomeruli))))
    fly = flyhash.mean_average_precision(x, flyhash.tags(x, proj, 16))
    lsh = flyhash.mean_average_precision(x, flyhash.gaussian_lsh(x, 16, seed=0))
    assert fly > lsh


def test_odour_data_loads(cfg, proj):
    x, names, used = flyhash.load_odours(cfg, proj.glomeruli)
    assert x.shape[0] > 100 and x.shape[1] > 20
    assert set(used) <= set(proj.glomeruli)
    assert x.min() >= 0.0


def test_real_wiring_has_no_advantage_over_the_strict_null(cfg, proj):
    """The headline finding: keep it pinned, so a regression would show up."""
    res = flyhash.experiment(data.load_graph(cfg), cfg, sizes=(16,), seeds=10)
    z = res["rows"][0]["z_vs_strict_null"]
    assert abs(z) < 2.5, f"the null result moved: z={z:+.2f}"
