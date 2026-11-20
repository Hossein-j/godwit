import pytest

from gei.envelope import GB, Envelope, ModelConfig, range_gain, validate_family


@pytest.fixture
def family():
    return [
        ModelConfig("large", 8 * GB, 1.00),
        ModelConfig("mid", 5 * GB, 0.93),
        ModelConfig("small", 3 * GB, 0.85),
    ]


@pytest.fixture
def env():
    return Envelope(
        memory_bytes=24 * GB,
        kv_bytes_per_token=256 * 1024,
        alpha_joules_per_byte=1e-9,
        energy_joules=5000.0,
        reserve_bytes=2 * GB,
    )


def test_family_must_shrink(family):
    validate_family(family)
    with pytest.raises(ValueError, match="strictly decrease"):
        validate_family([family[2], family[0]])


def test_family_quality_must_not_increase():
    with pytest.raises(ValueError, match="not increase in quality"):
        validate_family(
            [ModelConfig("a", 8 * GB, 0.8), ModelConfig("b", 4 * GB, 0.95)]
        )


def test_burning_extends_the_horizon(family, env):
    """The whole premise: a smaller model reaches more tokens."""
    assert env.max_tokens(family[2]) > env.max_tokens(family[0])


def test_range_gain_matches_the_formula(family, env):
    gain = range_gain(family, env)
    free_large = env.usable_memory - family[0].weight_bytes
    free_small = env.usable_memory - family[2].weight_bytes
    assert gain["fixed_max_tokens"] == int(free_large // env.kv_bytes_per_token)
    assert gain["burned_max_tokens"] == int(free_small // env.kv_bytes_per_token)
    assert gain["extension_ratio"] > 1.0


def test_feasibility_flips_exactly_at_max_tokens(family, env):
    cfg = family[0]
    limit = env.max_tokens(cfg)
    assert env.feasible(cfg, limit)
    assert not env.feasible(cfg, limit + 1)


def test_reserve_reduces_usable_memory(family):
    big = Envelope(memory_bytes=24 * GB, kv_bytes_per_token=1024)
    small = Envelope(memory_bytes=24 * GB, kv_bytes_per_token=1024, reserve_bytes=4 * GB)
    assert small.max_tokens(family[0]) < big.max_tokens(family[0])


def test_model_too_big_yields_zero_tokens(env):
    assert env.max_tokens(ModelConfig("huge", 64 * GB, 1.0)) == 0


def test_energy_grows_with_context(family, env):
    cfg = family[0]
    assert env.energy_per_token(cfg, 10_000) > env.energy_per_token(cfg, 0)


def test_segment_energy_equals_sum_of_tokens(family, env):
    """The midpoint shortcut must be exact for linear KV growth."""
    cfg = family[1]
    brute = sum(env.energy_per_token(cfg, t) for t in range(100, 164))
    assert env.segment_energy(cfg, 100, 64) == pytest.approx(brute)


def test_smaller_model_costs_less_energy_per_token(family, env):
    assert env.energy_per_token(family[2], 5000) < env.energy_per_token(family[0], 5000)


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        ModelConfig("bad", -1, 0.5)
    with pytest.raises(ValueError):
        ModelConfig("bad", 1024, 1.7)


def test_batch_size_scales_kappa_and_prompt_per_sequence(family):
    """B concurrent sequences share the weights but each pays its own cache."""
    one = Envelope(memory_bytes=24 * GB, kv_bytes_per_token=8 * 1024,
                   prompt_kv_bytes=64 * 1024**2)
    many = Envelope(memory_bytes=24 * GB, kv_bytes_per_token=8 * 1024,
                    prompt_kv_bytes=64 * 1024**2, batch_size=32)
    assert many.kappa == 32 * one.kappa
    assert many.kv_bytes_at(1000) == 32 * one.kv_bytes_at(1000)
    large = family[0]
    assert many.memory_used(large, 1000) == large.weight_bytes + 32 * one.kv_bytes_at(1000)
    free = 24 * GB - large.weight_bytes - 32 * 64 * 1024**2
    assert many.max_tokens(large) == int(free // (32 * 8 * 1024))


def test_batching_makes_a_small_cache_bind(family):
    """A cache too small to ever bind alone binds once enough sequences share it."""
    single = Envelope(memory_bytes=10 * GB, kv_bytes_per_token=8 * 1024)
    batched = Envelope(memory_bytes=10 * GB, kv_bytes_per_token=8 * 1024, batch_size=12)
    t_max = 32_768
    assert single.feasible(family[0], t_max)
    assert not batched.feasible(family[0], t_max)
    assert batched.feasible(family[-1], t_max)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        Envelope(memory_bytes=GB, kv_bytes_per_token=1024, batch_size=0)
