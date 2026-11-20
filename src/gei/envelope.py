# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Memory and energy envelope for Godwit Endurance Inference (GEI).

The central constraint of GEI: weights and context share one fixed memory
envelope.

    W(c_t) + K_t <= M

A conventional system holds W fixed, so the task dies at
T = (M - W)/kappa. GEI lets W shrink along a planned path, extending the
reachable horizon to (M - W_min)/kappa.

Everything in this module is pure arithmetic: no GPU, no model, fully testable
on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024 ** 3
MB = 1024 ** 2


@dataclass(frozen=True)
class ModelConfig:
    """One nested sub-model of an elastic family (MatFormer / Nemotron Elastic).

    Attributes
    ----------
    name:
        Human readable identifier, e.g. "E4B" or "12B".
    weight_bytes:
        Resident weight memory for this configuration.
    quality:
        Relative per-token quality in [0, 1]. Used only to rank configurations
        in the burn schedule; absolute values do not matter, differences do.
    """

    name: str
    weight_bytes: int
    quality: float

    def __post_init__(self) -> None:
        if self.weight_bytes <= 0:
            raise ValueError("weight_bytes must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must lie in [0, 1]")


def validate_family(configs: list[ModelConfig]) -> None:
    """A burnable family is ordered largest-first and quality must not increase.

    Burning is irreversible within a task, so a schedule only ever moves to a
    higher index. That is only sensible if higher index means smaller and no
    better.
    """
    if len(configs) < 2:
        raise ValueError("need at least two configurations to burn between")
    for a, b in zip(configs, configs[1:]):
        if b.weight_bytes >= a.weight_bytes:
            raise ValueError(
                f"configs must strictly decrease in weight_bytes: "
                f"{a.name}={a.weight_bytes} then {b.name}={b.weight_bytes}"
            )
        if b.quality > a.quality:
            raise ValueError(
                f"configs must not increase in quality: "
                f"{a.name}={a.quality} then {b.name}={b.quality}"
            )


@dataclass(frozen=True)
class Envelope:
    """The fixed resource envelope a task must finish inside.

    Attributes
    ----------
    memory_bytes:
        Total device memory available to the process (M).
    energy_joules:
        Total energy budget for the task (E). Use float('inf') to ignore.
    kv_bytes_per_token:
        Context memory one sequence adds per generated token, after any KV
        compression already in use. A per-sequence model constant; see
        `batch_size` for the envelope-wide kappa.
    alpha_joules_per_byte:
        Energy per byte read per token. Decoding is memory-bandwidth bound, so
        e_t ~ alpha * (W + K_t). Calibrate this empirically per device.
    prompt_kv_bytes:
        Context memory already occupied by the prompt at t = 0.
    reserve_bytes:
        Headroom held back for activations, fragmentation and the allocator.
    batch_size:
        Sequences decoded concurrently, in lockstep, sharing one copy of the
        weights. Every context term scales with it: kappa = batch_size *
        kv_bytes_per_token, and the prompt cache is paid once per sequence.
        With batch_size > 1 a "token" index is a decode step (one token per
        sequence), and energy_per_token is the energy of that whole step.
        This is how a model with a small per-token cache is made to bind
        honestly on a large device (docs/MODEL_SELECTION.md).
    """

    memory_bytes: int
    kv_bytes_per_token: float
    alpha_joules_per_byte: float = 0.0
    energy_joules: float = float("inf")
    prompt_kv_bytes: float = 0.0
    reserve_bytes: int = 0
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def kappa(self) -> float:
        """Context memory the whole envelope gains per decode step."""
        return self.batch_size * self.kv_bytes_per_token

    @property
    def usable_memory(self) -> float:
        """Memory actually available to weights plus context."""
        return self.memory_bytes - self.reserve_bytes

    def kv_bytes_at(self, token: float) -> float:
        """Context memory K_t after `token` generated tokens."""
        return self.batch_size * self.prompt_kv_bytes + self.kappa * token

    def memory_used(self, config: ModelConfig, token: float) -> float:
        """Total resident memory W(c) + K_t."""
        return config.weight_bytes + self.kv_bytes_at(token)

    def headroom(self, config: ModelConfig, token: float) -> float:
        """Bytes still free. Negative means the envelope is already violated."""
        return self.usable_memory - self.memory_used(config, token)

    def feasible(self, config: ModelConfig, token: float) -> bool:
        """Does the envelope hold at this token index for this configuration?"""
        return self.headroom(config, token) >= 0

    def max_tokens(self, config: ModelConfig) -> int:
        """Last token index reachable without ever leaving the envelope.

        This is the horizon a conventional fixed-size system is stuck with.
        """
        free = self.usable_memory - self.memory_used(config, 0)
        if free < 0:
            return 0
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        return int(free // self.kappa)

    def energy_per_token(self, config: ModelConfig, token: float) -> float:
        """Energy for the token generated at index `token`."""
        return self.alpha_joules_per_byte * self.memory_used(config, token)

    def segment_energy(
        self, config: ModelConfig, start_token: int, n_tokens: int
    ) -> float:
        """Energy for `n_tokens` tokens starting at `start_token`.

        Uses the midpoint of the segment, which is exact for the linear growth
        of K_t assumed here.
        """
        if n_tokens <= 0:
            return 0.0
        midpoint = start_token + (n_tokens - 1) / 2.0
        return n_tokens * self.energy_per_token(config, midpoint)


def range_gain(configs: list[ModelConfig], envelope: Envelope) -> dict:
    """Headline numbers for the paper: how far burning extends the horizon.

    Returns the fixed-model horizon (largest config, never burning), the GEI
    horizon (burned all the way down) and their ratio.
    """
    validate_family(configs)
    fixed = envelope.max_tokens(configs[0])
    burned = envelope.max_tokens(configs[-1])
    return {
        "fixed_max_tokens": fixed,
        "burned_max_tokens": burned,
        "extension_tokens": burned - fixed,
        "extension_ratio": (burned / fixed) if fixed > 0 else float("inf"),
        "fixed_config": configs[0].name,
        "smallest_config": configs[-1].name,
    }
