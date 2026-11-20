# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""The in-flight controller: fly the schedule, adapt to what actually happens.

The planned schedule is made at launch from a belief about task length. In
flight the belief changes (the task has not ended yet), so the controller
re-solves periodically — the godwit adjusts its departure by a few days on
local winds, but keeps the plan.

The fuel-mix switch lives here too. When memory is the binding constraint, burn
weights. When energy or heat binds, trim context instead. Godwits burn more
protein when dehydrated rather than when hungry; what you burn depends on what
is actually scarce. RAP's authors list energy and latency as outside their
method, so this is new ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .elastic import ElasticModel, KVPolicy
from .envelope import Envelope
from .launch import LaunchDecision, launch_check
from .length_prior import LengthPrior
from .schedule import solve_schedule


class Binding(str, Enum):
    """Which constraint is currently tightest."""

    MEMORY = "memory"
    ENERGY = "energy"
    THERMAL = "thermal"
    NONE = "none"


@dataclass
class FlightLog:
    """Everything the paper needs from one run."""

    completed: bool = False
    tokens: int = 0
    aborted_reason: str | None = None
    burns: list[tuple[int, int, str]] = field(default_factory=list)
    energy_used: float = 0.0
    replans: int = 0
    launch_p_success: float = 0.0
    mean_quality: float = 0.0

    def summary(self) -> str:
        state = "completed" if self.completed else f"failed ({self.aborted_reason})"
        return (
            f"{state} after {self.tokens:,} tokens; "
            f"{len(self.burns)} burn(s), {self.replans} replan(s), "
            f"p_launch={self.launch_p_success:.3f}"
        )


def binding_constraint(
    envelope: Envelope,
    model: ElasticModel,
    *,
    token: int,
    energy_used: float,
    thermal_headroom: float = 1.0,
    memory_margin: float = 0.05,
) -> Binding:
    """Which limit is closest to biting right now.

    `thermal_headroom` is a device signal in [0, 1]; 0 means throttling. It is
    passed in rather than measured here so the controller stays testable
    without hardware.
    """
    cfg = model.configs[model.current_index]
    free_fraction = envelope.headroom(cfg, token) / max(envelope.usable_memory, 1)
    if free_fraction <= memory_margin:
        return Binding.MEMORY
    if thermal_headroom <= 0.1:
        return Binding.THERMAL
    if envelope.energy_joules < float("inf"):
        spent = energy_used / envelope.energy_joules
        progress = token / max(envelope.max_tokens(model.configs[-1]), 1)
        if spent > 0.9 or (spent > 0.5 and spent > progress * 1.5):
            return Binding.ENERGY
    return Binding.NONE


def fly(
    model: ElasticModel,
    envelope: Envelope,
    prior: LengthPrior,
    *,
    true_length: int,
    tau: float = 0.95,
    tau_abort: float = 0.05,
    segment_tokens: int = 512,
    check_every: int = 256,
    replan: bool = True,
    kv_policy: str = KVPolicy.REUSE,
    thermal_signal=None,
    decision: LaunchDecision | None = None,
) -> FlightLog:
    """Run one task end to end under GEI control.

    `true_length` is how many tokens the task really needs. In a real system
    this is unknown and generation simply stops at EOS; it is passed explicitly
    here so experiments can sweep it and measure completion honestly.

    `thermal_signal(token) -> float` optionally supplies device headroom.
    """
    log = FlightLog()

    if decision is None:
        decision = launch_check(
            model.configs, envelope, prior, tau=tau, segment_tokens=segment_tokens
        )
    log.launch_p_success = decision.p_success

    if not decision.go or decision.schedule is None:
        log.aborted_reason = "launch check declined"
        return log

    schedule = decision.schedule
    model.switch_to(decision.start_index, kv_policy=kv_policy)

    quality_sum = 0.0
    for token in range(true_length):
        cfg = model.configs[model.current_index]

        # Hard stop: the envelope is violated and there is nothing left to burn.
        if not envelope.feasible(cfg, token):
            if model.current_index + 1 < len(model.configs):
                model.switch_to(model.current_index + 1, kv_policy=kv_policy)
                log.burns.append((token, model.current_index, "forced"))
                continue
            log.aborted_reason = "out of memory with nothing left to burn"
            log.tokens = token
            return log

        log.energy_used += envelope.energy_per_token(cfg, token)
        if log.energy_used > envelope.energy_joules:
            log.aborted_reason = "energy budget exhausted"
            log.tokens = token
            return log

        model.step()
        quality_sum += getattr(model, "effective_quality", lambda: cfg.quality)()

        if token > 0 and token % check_every == 0:
            planned = schedule.config_at(token)
            if planned > model.current_index:
                model.switch_to(planned, kv_policy=kv_policy)
                log.burns.append((token, planned, "scheduled"))

            headroom = 1.0 if thermal_signal is None else thermal_signal(token)
            limit = binding_constraint(
                envelope,
                model,
                token=token,
                energy_used=log.energy_used,
                thermal_headroom=headroom,
            )
            if limit is Binding.MEMORY and model.current_index + 1 < len(model.configs):
                model.switch_to(model.current_index + 1, kv_policy=kv_policy)
                log.burns.append((token, model.current_index, "memory-bound"))
            elif limit in (Binding.ENERGY, Binding.THERMAL):
                # Fuel-mix switch: the scarce resource is not memory, so
                # shedding weights would not help. Trim context instead.
                log.burns.append((token, model.current_index, f"trim-kv:{limit.value}"))

            if replan:
                try:
                    updated = prior.condition_on_progress(token)
                except ValueError:
                    log.aborted_reason = "task outran the length prior"
                    log.tokens = token
                    return log
                schedule = solve_schedule(
                    model.configs,
                    envelope,
                    updated,
                    start_index=model.current_index,
                    segment_tokens=segment_tokens,
                )
                log.replans += 1
                if schedule.p_success < tau_abort:
                    log.aborted_reason = "replanned success probability below abort"
                    log.tokens = token
                    return log

    log.completed = True
    log.tokens = true_length
    log.mean_quality = quality_sum / max(true_length, 1)
    return log
