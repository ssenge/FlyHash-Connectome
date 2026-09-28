"""F2 runner (flypath.f2_audit) on toy wiring. No data needed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flypath import experiments as ex
from flypath import f2_audit as f2
from flypath import flyhash as fh
from flypath import stats


def _toy(n_g=10, n_c=200, seed=0):
    rng = np.random.default_rng(seed)
    m = (rng.random((n_g, n_c)) < 0.3).astype(float)
    m[rng.integers(0, n_g, n_c), np.arange(n_c)] = 1.0          # every cell has an input
    x = np.abs(rng.normal(size=(30, n_g)))
    return fh.Projection("b", m, [f"G{i}" for i in range(n_g)]), x


@pytest.fixture
def root(monkeypatch, tmp_path):
    """Keep every cache write (null_pool included) inside tmp_path."""
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(ex, "N_ITEMS", 300)
    return tmp_path


def test_draws_and_cache_key_match_null_pool(root):
    p, _ = _toy()
    pool = ex.null_pool(p, 4)
    assert f2.pool_cache_path(p, 4).exists()                    # same key as null_pool
    for b, q in enumerate(pool):
        np.testing.assert_array_equal(f2.draw(p, b), q.matrix > 0)
    np.testing.assert_array_equal(f2.draw(p, 0).sum(1), p.fan_out())
    np.testing.assert_array_equal(f2.draw(p, 0).sum(0), p.inputs())


def test_written_cache_is_what_null_pool_returns(root, monkeypatch):
    p, _ = _toy(seed=1)
    mats = np.stack([f2.draw(p, b) for b in range(3)])
    f2.write_pool_cache(p, mats)

    def boom(*a, **k):
        raise AssertionError("null_pool regenerated instead of reading the cache")
    monkeypatch.setattr(fh, "curveball", boom)
    pool = ex.null_pool(p, 3)
    assert all(np.array_equal(q.matrix > 0, m > 0) for q, m in zip(pool, mats))


def test_summarise_floor_and_ties():
    null = np.linspace(0.40, 0.42, 99)
    s = f2.summarise(0.30, null)
    assert s["p_two_sided"] == pytest.approx(2 / 100) == pytest.approx(s["p_floor"])
    assert s["relative_difference"] == pytest.approx(0.30 / null.mean() - 1)
    tie = f2.summarise(float(null[0]), null)                    # ties count against the claim
    assert tie["p_lower"] == pytest.approx(2 / 100)


def test_holm_family_matches_stats_holm():
    recs = {k: {"p_two_sided": p} for k, p in zip("abcdefg", [0.001, 0.01, 0.02, 0.3, 0.04, 0.5, 0.007])}
    h = f2.holm_family(recs)
    assert list(h["p_holm"].values()) == stats.holm([r["p_two_sided"] for r in recs.values()])
    assert h["n_significant"] == sum(v < 0.05 for v in h["p_holm"].values())


def test_advance_resumes_after_interruption_and_equals_null_pool_scoring(root, monkeypatch):
    po, x = _toy(seed=2)
    a, b = root / "a", root / "b"
    a.mkdir()
    b.mkdir()
    real_scores = ex.scores
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("interrupted")
        return real_scores(*args, **kw)
    monkeypatch.setattr(ex, "scores", flaky)
    with pytest.raises(RuntimeError):
        f2.advance("toy_R", po, x, a, n_null=6, chunk=2, log=lambda *_: None)
    assert len(list(a.glob("toy_R-B6-c2-*.npz"))) == 1          # the first chunk survived
    monkeypatch.setattr(ex, "scores", real_scores)
    resumed = f2.advance("toy_R", po, x, a, n_null=6, chunk=2, log=lambda *_: None)
    fresh = f2.advance("toy_R", po, x, b, n_null=6, chunk=3, log=lambda *_: None)
    assert resumed["null_scores"] == fresh["null_scores"]
    assert resumed["checks"]["pool_cache_equals_chunks"]

    # identical to _odour_test's route: null_pool, then experiments.scores at the primary k
    b0 = ex.bench(fh.mixtures(x, ex.N_ITEMS, seed=0))
    kP = ex.primary_k(po)
    ref = [real_scores(b0, q, (kP,))[0] for q in ex.null_pool(po, 6)]
    np.testing.assert_allclose(resumed["null_scores"], ref, rtol=0, atol=0)
    assert resumed["real"] == real_scores(b0, po, (kP,))[0]
    assert not list(a.glob("toy_R-B6-c2-*.npz"))                # chunks removed once cached
    assert f2.advance("toy_R", po, x, a, n_null=6, chunk=2) == resumed   # done: no recompute


def test_run_writes_output_only_when_complete(root):
    po, x = _toy(seed=3)
    ck = root / "ck"
    ck.mkdir()
    f2._save_npz(ck / "malecns_R-setup.npz", binary=po.binary.astype(np.uint8),
                 glomeruli=np.array(po.glomeruli), x=x)
    out = root / "out.json"
    st = f2.run(cfg=object(), n_null=4, chunk=2, budget=0, only=["malecns_R"], out=out,
                checkpoint_dir=ck, log=lambda *_: None)
    assert not st["complete"] and not out.exists()
    st = f2.run(cfg=object(), n_null=4, chunk=2, budget=600, only=["malecns_R"], out=out,
                checkpoint_dir=ck, log=lambda *_: None)
    assert st["complete"]
    res = json.loads(out.read_text())
    assert res["B"] == 4 and res["holm"] is None                # Holm needs all seven
    assert len(res["hemispheres"]["malecns_R"]["null_scores"]) == 4
    assert not list(root.rglob("*.tmp")) and not list(root.rglob("*.tmp.npz"))   # atomic writes


def test_run_never_overwrites_the_f2_result_with_a_partial_run(root):
    po, x = _toy(seed=4)
    ck = root / "ck"
    ck.mkdir()
    f2._save_npz(ck / "malecns_R-setup.npz", binary=po.binary.astype(np.uint8),
                 glomeruli=np.array(po.glomeruli), x=x)
    with pytest.raises(ValueError, match="unknown hemispheres"):  # a typo used to mean "nothing to do: complete"
        f2.run(cfg=object(), n_null=4, chunk=2, budget=600, only=["malecnsR"],
               out=root / "o.json", checkpoint_dir=ck, log=lambda *_: None)
    for kw in ({"n_null": 4, "only": ["malecns_R"]},              # subset and other B
               {"n_null": f2.B, "only": ["malecns_R"]},           # subset at the registered B
               {"n_null": 4}):                                    # all seven, other B
        with pytest.raises(ValueError, match="explicit out"):
            f2.run(cfg=object(), chunk=2, budget=600, checkpoint_dir=ck, log=lambda *_: None, **kw)
    assert not f2.default_out().exists()
    assert not list(ck.glob("malecns_R-B*"))                      # refused before any work
