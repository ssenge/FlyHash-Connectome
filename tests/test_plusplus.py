"""Independent invariants and adversarial cases for the experimental suite."""
import json
from pathlib import Path

import numpy as np
import pytest

from flypath import plusplus as pp


def config():
    c = json.loads(Path('experiments/flyhashpp_pilot.json').read_text())
    c.update(train=32, validation=8, queries=8, gallery=40, top=5,
             dimensions=8, epochs=2, diversity_steps=30, timing_repeats=1,
             candidate_budget=10, seeds=[0], expansions=[2], fanins=[2], winners=[4])
    return c


def test_diversity_preserves_both_margins_and_reduces_cycles():
    w = pp.rp.fly_matrix(12, 50, np.random.default_rng(2), sampled=4)
    v, accepted = pp.diversify(w, 300, 9)
    np.testing.assert_array_equal(w.sum(0), v.sum(0))
    np.testing.assert_array_equal(w.sum(1), v.sum(1))
    assert set(np.unique(v)) <= {0, 1}
    assert pp.overlap_energy(v) < pp.overlap_energy(w)
    assert accepted > 0


def test_top_ties_do_not_change_with_batch_or_scale():
    y = np.array([[1., 1., 1., 0.], [1., 2., 2., 2.]])
    priority = np.array([3, 2, 1, 0])
    np.testing.assert_array_equal(pp.fixed_top(y, 2, priority), [[2, 1], [3, 2]])
    np.testing.assert_array_equal(pp.fixed_top(y[:1], 2, priority), pp.fixed_top(y, 2, priority)[:1])
    np.testing.assert_array_equal(pp.fixed_top(y * 100, 2, priority), pp.fixed_top(y, 2, priority))


def test_bio_update_matches_explicit_equations_and_mask():
    w = np.array([[.2, .7, -.3], [.6, -.1, .2]])
    x = np.array([[1., 2.], [-1., 1.]])
    expected = np.zeros_like(w)
    for row in x:
        currents = row @ w
        order = np.argsort(-currents)
        for neuron, sign in [(order[0], 1.), (order[1], -.4)]:
            expected[:, neuron] += sign * (row - currents[neuron] * w[:, neuron])
    expected /= np.abs(expected).max()
    np.testing.assert_allclose(pp.bio_update(w, x), expected)
    mask = np.array([[1, 0, 1], [0, 1, 0]])
    update = pp.bio_update(w * mask, x, mask=mask)
    assert (update[mask == 0] == 0).all()


def test_scores_missing_retrieval_and_exact_truth():
    truth = np.arange(200)[None]
    pred = np.full((1, 200), -1); pred[0, 0] = 0
    ap, recall = pp.scores(pred, truth)
    assert ap[0] == recall[0] == .005
    assert pp.scores(truth, truth)[0][0] == 1


def test_disjoint_splits_and_training_only_pca():
    c = config(); x = np.random.default_rng(5).normal(size=(100, 12))
    blocks, ids = pp.split_data(x, c, 0)
    assert len(set(sum(ids, []))) == sum(map(len, ids))
    transformed, _ = pp.preprocess(blocks, 8)
    perturbed = [a.copy() for a in blocks]; perturbed[2] += 1000
    again, _ = pp.preprocess(perturbed, 8)
    np.testing.assert_array_equal(transformed[0], again[0])
    np.testing.assert_array_equal(transformed[3], again[3])


@pytest.mark.parametrize('method', pp.METHODS)
def test_models_are_finite_deterministic_and_costed(method):
    c = config(); rng = np.random.default_rng(3)
    train, raw = rng.normal(size=(32, 8)).astype('f'), rng.normal(size=(32, 8)).astype('f')
    spec = pp.Spec(method, 16, 2, 4)
    a = pp.Model(spec, c, 0).fit(train, raw)
    b = pp.Model(spec, c, 0).fit(train, raw)
    y, yp = a.transform(train, raw)
    np.testing.assert_array_equal(y, b.transform(train, raw)[0])
    assert np.isfinite(y).all()
    if method in ('fly', 'balanced', 'diverse', 'calibrated', 'zscore', 'balanced_calibrated',
                  'diverse_calibrated', 'two_concat', 'two_or', 'biohash', 'biohash_untrained',
                  'sparse_bio', 'sparse_untrained'):
        assert (y.sum(1) == 4).all()
    if method == 'densefly': np.testing.assert_array_equal(y, train @ a.w >= 0)
    stats = a.resources(y)
    assert stats['projection_operations'] > 0 and stats['code_bits_fixed_width'] > 0
    pred, n = pp.rank_codes(y[:4], y[4:], a, 5,
                            None if yp is None else yp[:4], None if yp is None else yp[4:])
    assert pred.shape == (4, 5)
    assert all(len(set(row[row >= 0])) == len(row[row >= 0]) for row in pred)
    if method in ('two_or', 'densefly_probe'): assert (n <= c['candidate_budget']).all()


def test_validation_frontier_not_test_selection():
    a = dict(key='a', projection_operations=10, code_bits_fixed_width=10, validation_ap=.5, test_ap=.9)
    b = dict(key='b', projection_operations=10, code_bits_fixed_width=10, validation_ap=.6, test_ap=.1)
    assert pp.frontier([a, b]) == ['b']


def test_full_smoke_writes_all_rows_and_pairs(tmp_path):
    c = config(); c['datasets'] = ['synthetic']
    path = tmp_path / 'smoke.json'
    out = pp.run(c, path)
    assert out['status'] == 'complete'
    assert len(out['rows']) == len(pp.METHODS)
    assert out['paired_ablations']
    saved = json.loads(path.read_text())
    assert saved['implementation_sha256']
    assert path.with_suffix('.md').exists()


def test_budget_baselines_and_independent_table_ablation():
    c = config(); x = np.random.default_rng(1).normal(size=(32, 8)).astype('f')
    models = {kind: pp.Model(pp.Spec(kind, 32, 3, 4), c, 2).fit(x, x)
              for kind in ('fly', 'gaussian_ops', 'gaussian_storage', 'two_global', 'two_concat', 'two_or')}
    resource = lambda key: models[key].resources(models[key].transform(x, x)[0])
    assert resource('gaussian_ops')['projection_operations'] <= resource('fly')['projection_operations']
    assert resource('gaussian_storage')['code_bits_fixed_width'] == resource('fly')['code_bits_fixed_width']
    for key in ('two_concat', 'two_or'):
        np.testing.assert_array_equal(models[key].w, models['two_global'].w)
    assert not np.array_equal(models['two_global'].w[:, :16], models['two_global'].w[:, 16:])
    assert (models['two_global'].w.sum(0) == 3).all()


def test_calibration_frozen_and_measured_sparse_mask_preserved():
    c = config(); x = np.random.default_rng(1).normal(size=(32, 8)).astype('f')
    m = pp.Model(pp.Spec('calibrated', 16, 2, 4), c, 2).fit(x, x)
    original = m.sorted_train.copy()
    together = m.transform(np.concatenate([x[:1], x[1:2] * 1000]), x[:2])[0]
    alone = m.transform(x[:1], x[:1])[0]
    np.testing.assert_array_equal(together[:1], alone)
    np.testing.assert_array_equal(original, m.sorted_train)
    base = pp.rp.fly_matrix(8, 16, np.random.default_rng(4), sampled=2)
    learned = pp.Model(pp.Spec('measured_sparse_bio', 16, 0, 4), c, 2).fit(x, x, base=base)
    assert (learned.w[base == 0] == 0).all()
    assert learned.resources(learned.transform(x, x)[0])['projection_multiplications'] > 0
