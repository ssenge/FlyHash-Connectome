"""Inference helpers. No data needed."""

from __future__ import annotations

import numpy as np
import pytest

from flypath import stats


def test_randomization_extremes():
    null = np.arange(10, dtype=float)
    top = stats.randomization_test(100.0, null)
    assert top["p_upper"] == pytest.approx(1 / 11)
    assert top["p_lower"] == pytest.approx(1.0)
    assert top["p_two_sided"] == pytest.approx(2 / 11)
    bottom = stats.randomization_test(-1.0, null)
    assert bottom["p_lower"] == pytest.approx(1 / 11)


def test_randomization_counts_ties_against_the_claim():
    null = np.array([1.0, 1.0, 2.0])
    r = stats.randomization_test(1.0, null)
    assert r["p_lower"] == pytest.approx(3 / 4)     # both ties count
    assert r["p_upper"] == pytest.approx(4 / 4)


def test_reviewer_rank_example():
    """The reviewer's figure: with 38 of 40 nulls scoring higher, 2 lie at or
    below the real score, so the lower-tail p is (1 + 2) / 41 = 0.073."""
    null = np.r_[np.full(38, 1.0), np.full(2, -1.0)]
    assert stats.randomization_test(0.0, null)["p_lower"] == pytest.approx(3 / 41)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_shift_power_is_monotone():
    null = np.random.default_rng(0).normal(1.0, 0.01, 200)
    pw = stats.shift_power(null, [0.001, 0.01, 0.03, 0.08])
    vals = list(pw.values())
    assert vals == sorted(vals)
    assert vals[0] < 0.2 and vals[-1] > 0.95
    assert stats.smallest_detectable(pw) in (0.03, 0.08)


def test_two_stage_carries_null_matrix_variation():
    """The reviewer's diagnostic: two null matrices with different constant
    per-query scores. The superseded procedure returns a zero-width interval;
    the two-stage bootstrap, which re-draws the null subset, does not."""
    real_ap = np.full(50, 0.5)
    null_aps = np.vstack([np.full(50, 0.4), np.full(50, 0.6)])
    cluster = np.repeat(np.arange(10), 5)
    lo, hi = stats.legacy_group_bootstrap(real_ap, null_aps, cluster, draws=500)
    assert hi - lo == pytest.approx(0.0)

    def replicate(rng):
        pick = rng.choice(2, 1)
        return 0.5, null_aps[pick, 0]
    ci = stats.two_stage_bootstrap(replicate, 400, seed=1)["relative_ci"]
    assert ci[1] - ci[0] > 0.1


def test_equivalence_needs_the_whole_interval_inside():
    assert stats.equivalent([-0.02, 0.01], 0.05)
    assert not stats.equivalent([-0.06, 0.01], 0.05)
    assert not stats.equivalent([-0.02, 0.05], 0.05)
