import pytest

from gei.controller import Binding, binding_constraint, fly
from gei.elastic import KVPolicy, MockElasticModel
from gei.envelope import GB, Envelope, ModelConfig
from gei.launch import launch_check
from gei.length_prior import LengthPrior


@pytest.fixture
def family():
    return [
        ModelConfig("large", 8 * GB, 1.00),
        ModelConfig("mid", 5 * GB, 0.93),
        ModelConfig("small", 3 * GB, 0.85),
    ]


@pytest.fixture
def prior():
    return LengthPrior.lognormal(median_tokens=6000, sigma=0.5, max_tokens=24_000)


def roomy():
    return Envelope(memory_bytes=200 * GB, kv_bytes_per_token=64 * 1024)


def tight():
    return Envelope(memory_bytes=16 * GB, kv_bytes_per_token=512 * 1024,
                    reserve_bytes=1 * GB)


def hopeless():
    return Envelope(memory_bytes=4 * GB, kv_bytes_per_token=1024 * 1024)


def test_launch_prefers_the_largest_viable_start(family, prior):
    """Quality is highest at the top; start lean only when forced."""
    decision = launch_check(family, roomy(), prior, tau=0.95)
    assert decision.go
    assert decision.start_index == 0


def test_launch_declines_an_impossible_task(family, prior):
    decision = launch_check(family, hopeless(), prior, tau=0.95)
    assert not decision.go
    assert decision.p_success < 0.95
    assert "no starting configuration" in decision.reason


def test_launch_goes_leaner_before_giving_up(family, prior):
    """A tight envelope should produce a GO, possibly from a smaller start."""
    decision = launch_check(family, tight(), prior, tau=0.80)
    assert decision.go
    assert decision.schedule is not None


def test_higher_tau_is_never_easier(family, prior):
    lenient = launch_check(family, tight(), prior, tau=0.50)
    strict = launch_check(family, tight(), prior, tau=0.999)
    assert lenient.go or not strict.go


def test_binding_constraint_detects_memory_pressure(family):
    env = tight()
    model = MockElasticModel(family)
    near_limit = env.max_tokens(family[0])
    assert binding_constraint(env, model, token=near_limit, energy_used=0.0) is (
        Binding.MEMORY
    )
    assert binding_constraint(env, model, token=0, energy_used=0.0) is Binding.NONE


def test_binding_constraint_detects_thermal(family):
    model = MockElasticModel(family)
    limit = binding_constraint(
        roomy(), model, token=100, energy_used=0.0, thermal_headroom=0.0
    )
    assert limit is Binding.THERMAL


def test_flight_completes_a_short_task(family, prior):
    model = MockElasticModel(family)
    log = fly(model, roomy(), prior, true_length=2000, check_every=256)
    assert log.completed
    assert log.tokens == 2000


def test_flight_burns_under_pressure_and_still_finishes(family, prior):
    env = tight()
    model = MockElasticModel(family)
    log = fly(model, env, prior, true_length=20_000, tau=0.70, check_every=512)
    assert log.burns, "a long task in a tight envelope must burn"
    assert model.current_index > 0


def test_fixed_model_dies_where_gei_survives(family, prior):
    """Side-by-side: the headline experiment, in miniature."""
    env = tight()
    length = env.max_tokens(family[0]) + 5000

    # A "family" with nothing worth burning to: the fixed-model baseline.
    no_burn = [family[0], ModelConfig("large-clone", family[0].weight_bytes - 1, 1.0)]
    stuck = MockElasticModel(no_burn)
    stuck_log = fly(stuck, env, prior, true_length=length, tau=0.10, check_every=512)

    gei = MockElasticModel(family)
    gei_log = fly(gei, env, prior, true_length=length, tau=0.10, check_every=512)

    assert not stuck_log.completed
    assert gei_log.tokens > stuck_log.tokens


def test_declined_launch_does_not_run(family, prior):
    model = MockElasticModel(family)
    log = fly(model, hopeless(), prior, true_length=5000, tau=0.95)
    assert not log.completed
    assert log.aborted_reason == "launch check declined"
    assert log.tokens == 0


def test_model_refuses_to_grow(family):
    model = MockElasticModel(family)
    model.switch_to(2)
    with pytest.raises(ValueError, match="irreversible"):
        model.switch_to(0)


def test_switch_shock_decays(family):
    """Quality dips right after a switch, then recovers."""
    model = MockElasticModel(family, switch_shock=0.3, shock_decay_tokens=100)
    model.prefill(list(range(10)))
    model.switch_to(1, kv_policy=KVPolicy.REUSE)
    right_after = model.effective_quality()
    for _ in range(150):
        model.step()
    assert model.effective_quality() > right_after


def test_recompute_avoids_switch_shock(family):
    """Condition D in Experiment 1: pay compute, skip the shock."""
    model = MockElasticModel(family, switch_shock=0.3)
    model.switch_to(1, kv_policy="recompute")
    assert model.effective_quality() == pytest.approx(family[1].quality)
