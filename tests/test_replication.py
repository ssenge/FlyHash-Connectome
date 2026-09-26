"""Scores and the fly construction used in flypath.replication."""

import numpy as np

from flypath import replication as r


def test_ap_counts_unretrieved_neighbours():
    # hits at ranks 1 and 3 among 4 true neighbours: (1/1 + 2/3) / 4
    pred = np.array([[0, 5, 1, 6]])
    true = np.array([[0, 1, 2, 3]])
    assert np.isclose(r.average_precision(pred, true), (1 + 2 / 3) / 4)
    assert np.isclose(r.recall(pred, true), 0.5)


def test_ap_rank_one_only_is_not_perfect():
    # the reviewer's counterexample: one correct neighbour at rank 1 of 200
    true = np.arange(200)[None, :]
    pred = np.concatenate([[0], np.arange(1000, 1199)])[None, :]
    assert np.isclose(r.average_precision(pred, true), 1 / 200)
    assert r.ap_retrieved(pred, true) == 1.0          # the flawed convention


def test_ap_extremes():
    true = np.arange(10)[None, :]
    assert r.average_precision(true.copy(), true) == 1.0
    assert r.recall(true.copy(), true) == 1.0
    none = np.arange(100, 110)[None, :]
    assert r.average_precision(none, true) == 0.0
    assert r.ap_retrieved(none, true) == 0.0


def test_ap_perfect_set_wrong_order_below_one():
    true = np.arange(4)[None, :]
    pred = np.array([[9, 0, 1, 2]])                   # three hits, shifted by one
    assert np.isclose(r.average_precision(pred, true), (1 / 2 + 2 / 3 + 3 / 4) / 4)


def test_list_overlap_matches_reference_code():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = np.array([rng.permutation(50)[:10] for _ in range(5)])
        p = np.array([rng.permutation(50)[:10] for _ in range(5)])
        p[0] = t[0]
        assert np.isclose(r.list_overlap(p, t), r._ap_reference(p, t))


def test_fly_matrix_samples_exactly():
    w = r.fly_matrix(40, 300, np.random.default_rng(1), sampled=6)
    assert (w.sum(0) == 6).all() and set(np.unique(w)) <= {0.0, 1.0}


def test_winners_are_top_k():
    y = np.random.default_rng(2).normal(size=(30, 100))
    t = r.winners(y, 5).toarray().astype(bool)
    assert (t.sum(1) == 5).all()
    assert (np.where(t, y, np.inf).min(1) >= np.sort(y, 1)[:, -5]).all()


def test_winners_random_ties_break_exact_ties_only():
    y = np.array([[3.0, 1.0, 1.0, 1.0, 0.0]])
    prio = np.array([0.0, 0.1, 0.9, 0.5, 0.99])
    t = r.winners(y, 2, prio).toarray()[0]
    assert t[0] == 1 and t[2] == 1 and t.sum() == 2   # the top cell, then the tied cell with top priority


def test_contrast_is_ratio_of_means_with_interval():
    c = r.contrast(np.array([1.1, 1.2, 1.0, 1.1]), np.array([1.0, 1.0, 1.0, 1.0]), draws=500)
    assert np.isclose(c["estimate"], 10.0)
    assert c["ci95"][0] <= c["estimate"] <= c["ci95"][1]
