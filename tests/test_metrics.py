import numpy as np
import pytest

from gei.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    js_distance,
    kl_divergence,
    mcnemar_exact,
    paired_bootstrap_ci,
    pass_at_1,
    reliability_curve,
    tokens_per_joule,
)


def test_kl_is_zero_for_identical_distributions():
    p = np.array([0.1, 0.2, 0.7])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_grows_with_divergence():
    p = np.array([0.9, 0.05, 0.05])
    near = np.array([0.8, 0.1, 0.1])
    far = np.array([0.05, 0.05, 0.9])
    assert kl_divergence(p, far) > kl_divergence(p, near) > 0


def test_js_distance_is_symmetric_and_bounded():
    p = np.array([0.9, 0.1])
    q = np.array([0.2, 0.8])
    assert js_distance(p, q) == pytest.approx(js_distance(q, p))
    assert 0.0 <= js_distance(p, q) <= 1.0


def test_pass_at_1():
    assert pass_at_1([True, True, False, False]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pass_at_1([])


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(0.7, 0.1, size=200)
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < values.mean() < hi


def test_perfect_calibration_scores_zero():
    predicted = [0.0] * 50 + [1.0] * 50
    outcomes = [False] * 50 + [True] * 50
    assert expected_calibration_error(predicted, outcomes) == pytest.approx(0.0)


def test_overconfident_launch_check_is_penalised():
    predicted = [0.95] * 100
    outcomes = [True] * 50 + [False] * 50  # actually only 50% finish
    assert expected_calibration_error(predicted, outcomes) == pytest.approx(0.45, abs=0.01)


def test_reliability_curve_shapes():
    curve = reliability_curve([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert len(curve["confidence"]) == len(curve["accuracy"]) == len(curve["count"])
    assert sum(curve["count"]) == 4


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [True])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.5], [True])


def test_tokens_per_joule():
    assert tokens_per_joule(1000, 50.0) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        tokens_per_joule(1000, 0.0)


def test_paired_bootstrap_ci_is_zero_for_identical_results():
    same = [True, False, True, True, False] * 20
    assert paired_bootstrap_ci(same, same) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_detects_what_overlapping_intervals_miss():
    """A consistent 10-point drop on the same problems: unpaired 95% intervals
    overlap, but the paired interval on the difference excludes zero."""
    rng = np.random.default_rng(0)
    b = rng.random(100) < 0.6
    a = b.copy()
    a[np.flatnonzero(b)[:10]] = False  # a loses exactly 10 problems b solved
    _, hi_a = bootstrap_ci(a)
    lo_b, _ = bootstrap_ci(b)
    assert hi_a >= lo_b  # the unpaired intervals overlap
    mean, _, hi = paired_bootstrap_ci(a, b)
    assert mean == pytest.approx(-0.10)
    assert hi < 0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 0], [1])


def test_mcnemar_exact_matches_the_binomial():
    # 1 vs 6 discordant: p = 2 * (C(7,0) + C(7,1)) / 2**7 = 16/128
    a = [True] * 1 + [False] * 6 + [True] * 10
    b = [False] * 1 + [True] * 6 + [True] * 10
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (1, 6)
    assert r["p"] == pytest.approx(16 / 128)


def test_mcnemar_exact_without_discordance_is_one():
    same = [True, False, True]
    assert mcnemar_exact(same, same) == {"a_only": 0, "b_only": 0, "p": 1.0}


def test_mcnemar_exact_is_symmetric_and_capped():
    a, b = [True, False, True, False], [False, True, False, True]
    assert mcnemar_exact(a, b)["p"] == 1.0
    assert mcnemar_exact(a, b)["p"] == mcnemar_exact(b, a)["p"]
