import numpy as np
import pytest

from gei.envelope import GB, Envelope, ModelConfig
from gei.length_prior import LengthPrior
from gei.schedule import reactive_schedule, solve_schedule


@pytest.fixture
def family():
    return [
        ModelConfig("large", 8 * GB, 1.00),
        ModelConfig("mid", 5 * GB, 0.93),
        ModelConfig("small", 3 * GB, 0.85),
    ]


@pytest.fixture
def tight_env():
    """Sized so the large model alone cannot reach the far tail of the prior."""
    return Envelope(
        memory_bytes=16 * GB,
        kv_bytes_per_token=512 * 1024,
        reserve_bytes=1 * GB,
    )


@pytest.fixture
def prior():
    return LengthPrior.lognormal(median_tokens=8000, sigma=0.6, max_tokens=32_000)


def test_prior_is_a_distribution(prior):
    assert prior.pmf.sum() == pytest.approx(1.0)
    assert np.all(np.diff(prior.grid) > 0)


def test_conditioning_moves_mass_forward(prior):
    later = prior.condition_on_progress(10_000)
    assert later.grid.min() > 10_000
    assert later.mean() > prior.mean()


def test_schedule_burns_when_memory_is_tight(family, tight_env, prior):
    sched = solve_schedule(family, tight_env, prior, segment_tokens=512)
    assert sched.burn_points(), "a tight envelope should force at least one burn"
    assert sched.p_success > 0


def test_burning_beats_never_burning(family, tight_env, prior):
    """GEI's core claim, in miniature."""
    fixed_only = [family[0]] + [
        ModelConfig(f"{family[0].name}-clone{i}", family[0].weight_bytes - i, 1.0)
        for i in (1, 2)
    ]
    burnable = solve_schedule(family, tight_env, prior, segment_tokens=512)
    stuck = solve_schedule(fixed_only, tight_env, prior, segment_tokens=512)
    assert burnable.p_success > stuck.p_success


def test_config_index_never_decreases(family, tight_env, prior):
    """Burning is irreversible within a task."""
    sched = solve_schedule(family, tight_env, prior, segment_tokens=512)
    path = sched.config_index_per_segment
    assert all(b >= a for a, b in zip(path, path[1:]))


def test_lambda_delays_burning(family, tight_env, prior):
    """Pricing quality should not make GEI burn sooner than it must."""
    greedy = solve_schedule(family, tight_env, prior, segment_tokens=512, lam=0.0)
    careful = solve_schedule(family, tight_env, prior, segment_tokens=512, lam=5.0)
    assert careful.expected_quality >= greedy.expected_quality - 1e-9


def test_roomy_envelope_needs_no_burn(family, prior):
    roomy = Envelope(memory_bytes=200 * GB, kv_bytes_per_token=64 * 1024)
    sched = solve_schedule(family, roomy, prior, segment_tokens=1024)
    assert sched.burn_points() == []
    assert sched.p_success == pytest.approx(1.0, abs=1e-6)


def test_planned_beats_or_matches_reactive(family, tight_env, prior):
    """The planning ablation: a schedule must never do worse than reacting."""
    planned = solve_schedule(family, tight_env, prior, segment_tokens=512)
    reactive = reactive_schedule(family, tight_env, prior, segment_tokens=512)
    assert planned.p_success >= reactive.p_success - 1e-9


def test_energy_budget_can_bind(family, prior):
    """With energy scarce, the schedule must shed weight to cut per-token cost."""
    env = Envelope(
        memory_bytes=200 * GB,
        kv_bytes_per_token=64 * 1024,
        alpha_joules_per_byte=1e-9,
        energy_joules=250.0,
    )
    sched = solve_schedule(family, env, prior, segment_tokens=1024, energy_buckets=64)
    assert sched.burn_points() or sched.p_success < 1.0


def test_hopeless_envelope_reports_failure(family, prior):
    env = Envelope(memory_bytes=3 * GB, kv_bytes_per_token=512 * 1024)
    sched = solve_schedule(family, env, prior, segment_tokens=512)
    assert sched.p_success == 0.0


def test_burn_points_align_to_segments(family, tight_env, prior):
    sched = solve_schedule(family, tight_env, prior, segment_tokens=512)
    for token, _ in sched.burn_points():
        assert token % 512 == 0
