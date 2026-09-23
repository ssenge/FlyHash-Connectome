"""The 2017 scores and the fly construction used in flypath.replication."""

import numpy as np

from flypath import replication as r


def test_average_precision_hand_example():
    # hits at ranks 1 and 3: (1/1 + 2/3) / 2
    pred = np.array([[0, 5, 1, 6]])
    true = np.array([[0, 1, 2, 3]])
    assert np.isclose(r.average_precision(pred, true), (1 + 2 / 3) / 2)
    assert r.average_precision(np.array([[7, 8]]), np.array([[0, 1]])) == 0.0


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
