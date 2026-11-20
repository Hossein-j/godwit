# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""The burn schedule: when to drop to the next smaller configuration.

This is the heart of GEI and the part that distinguishes it from prior work.
RAP chooses one configuration per request and keeps it. PM-KVQ shrinks the
context progressively but needs the maximum length up front. Here the schedule
is planned across a task whose length is a *distribution*, and it moves down
the nested family while the generation is still running.

Formulation
-----------
Choose a non-decreasing configuration index k(t) maximising

    J = sum_s P(T ends in segment s) * 1{feasible through s}
        - lam * sum_s P(T >= start of s) * (q_max - q(c_s)) * weight_s

The first term is the probability the task finishes. The second is the quality
paid for burning early, weighted by the chance the task is still running (there
is no point charging for quality the task never used). This decomposition is
separable over segments, which is what makes an exact dynamic program possible.

Burning is irreversible within a task (the godwit cannot regrow organs
mid-flight), so k(t) only ever increases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .envelope import Envelope, ModelConfig, validate_family
from .length_prior import LengthPrior

_FAIL = -1.0e18  # sentinel for "this path cannot complete"


@dataclass
class BurnSchedule:
    """A planned sequence of configuration drops."""

    segment_tokens: int
    config_index_per_segment: list[int]
    configs: list[ModelConfig] = field(repr=False)
    p_success: float = 0.0
    expected_quality: float = 0.0
    objective: float = 0.0
    start_index: int = 0

    def config_at(self, token: int) -> int:
        """Configuration index in force at a given token."""
        seg = min(token // self.segment_tokens, len(self.config_index_per_segment) - 1)
        return self.config_index_per_segment[seg]

    def burn_points(self) -> list[tuple[int, int]]:
        """(token, new config index) for each drop — what the controller acts on."""
        points: list[tuple[int, int]] = []
        prev = self.config_index_per_segment[0]
        for seg, k in enumerate(self.config_index_per_segment):
            if k != prev:
                points.append((seg * self.segment_tokens, k))
                prev = k
        return points

    def describe(self) -> str:
        lines = [
            f"start: {self.configs[self.start_index].name}",
            f"p(success) = {self.p_success:.3f}",
            f"E[quality] = {self.expected_quality:.3f}",
        ]
        for token, k in self.burn_points():
            lines.append(f"  burn at token {token:>7,} -> {self.configs[k].name}")
        if not self.burn_points():
            lines.append("  (no burn needed)")
        return "\n".join(lines)


def solve_schedule(
    configs: list[ModelConfig],
    envelope: Envelope,
    prior: LengthPrior,
    *,
    start_index: int = 0,
    segment_tokens: int = 512,
    lam: float = 0.0,
    energy_buckets: int = 64,
) -> BurnSchedule:
    """Exact dynamic program for the optimal burn schedule.

    Parameters
    ----------
    start_index:
        Configuration to launch in. The launch check sweeps this.
    segment_tokens:
        Granularity of decisions. The controller re-checks every this many
        tokens; smaller is finer but costs DP time.
    lam:
        Price on quality loss. lam = 0 maximises completion probability alone
        and is the right setting when finishing is all that matters.
    energy_buckets:
        Discretisation of cumulative energy. Ignored when the energy budget is
        infinite, which collapses the state space.

    State is (segment, config index, energy bucket); the DP is solved backwards.
    """
    validate_family(configs)
    if not 0 <= start_index < len(configs):
        raise ValueError("start_index out of range")
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")

    horizon = int(prior.grid[-1])
    n_seg = int(np.ceil(horizon / segment_tokens))
    n_cfg = len(configs)
    track_energy = np.isfinite(envelope.energy_joules)
    n_buckets = energy_buckets if track_energy else 1
    bucket_size = (envelope.energy_joules / n_buckets) if track_energy else 1.0

    # Per-segment probability mass: P(T ends in segment s) and P(T >= segment s).
    seg_end_mass = np.zeros(n_seg)
    for t, p in zip(prior.grid, prior.pmf):
        seg_end_mass[min(int((t - 1) // segment_tokens), n_seg - 1)] += p
    seg_survival = np.concatenate([[1.0], 1.0 - np.cumsum(seg_end_mass)[:-1]])

    q_max = max(c.quality for c in configs)
    seg_weight = segment_tokens / max(horizon, 1)

    # Precompute per (segment, config): memory feasibility and energy cost.
    feasible = np.zeros((n_seg, n_cfg), dtype=bool)
    energy = np.zeros((n_seg, n_cfg))
    for s in range(n_seg):
        start_tok = s * segment_tokens
        end_tok = min((s + 1) * segment_tokens, horizon)
        for k, cfg in enumerate(configs):
            # The envelope must hold at the *end* of the segment, the worst case.
            feasible[s, k] = envelope.feasible(cfg, end_tok)
            energy[s, k] = envelope.segment_energy(
                cfg, start_tok, max(end_tok - start_tok, 0)
            )

    # Backwards DP. value[k, b] = expected future objective from segment s.
    value = np.zeros((n_cfg, n_buckets))
    choice = np.zeros((n_seg, n_cfg, n_buckets), dtype=np.int16)

    for s in range(n_seg - 1, -1, -1):
        nxt = value
        value = np.zeros((n_cfg, n_buckets))
        for k in range(n_cfg):
            for b in range(n_buckets):
                # Reaching a segment no configuration can serve is not a
                # catastrophe: the task simply stops there. Probability mass
                # already banked by shorter tasks is kept, and future reward is
                # zero. Treating this as -inf (an earlier bug) made the DP
                # discard schedules that succeed for all but the longest tail.
                best, best_k = 0.0, k
                for k2 in range(k, n_cfg):  # burning is irreversible
                    if not feasible[s, k2]:
                        continue
                    if track_energy:
                        b2 = b + int(np.ceil(energy[s, k2] / bucket_size))
                        if b2 >= n_buckets:
                            continue
                    else:
                        b2 = 0
                    reward = seg_end_mass[s] - lam * seg_survival[s] * (
                        q_max - configs[k2].quality
                    ) * seg_weight
                    total = reward + nxt[k2, b2]
                    if total > best:
                        best, best_k = total, k2
                value[k, b] = best
                choice[s, k, b] = best_k

    # Forward pass: read the policy out and score it honestly.
    path: list[int] = []
    k, b = start_index, 0
    p_success = 0.0
    quality = 0.0
    weight_seen = 0.0
    alive = True
    for s in range(n_seg):
        if alive:
            k2 = int(choice[s, k, b])
            if not feasible[s, k2]:
                alive = False
            else:
                if track_energy:
                    b2 = b + int(np.ceil(energy[s, k2] / bucket_size))
                    if b2 >= n_buckets:
                        alive = False
                    else:
                        b = b2
                if alive:
                    k = k2
                    p_success += seg_end_mass[s]
                    # Survival-weighted so quality the task never reaches does
                    # not count; normalised at the end to read as an average.
                    quality += seg_survival[s] * configs[k].quality * seg_weight
                    weight_seen += seg_survival[s] * seg_weight
        path.append(k)
    quality = quality / weight_seen if weight_seen > 0 else 0.0

    return BurnSchedule(
        segment_tokens=segment_tokens,
        config_index_per_segment=path,
        configs=configs,
        p_success=float(p_success),
        expected_quality=float(quality),
        objective=float(value[start_index, 0]),
        start_index=start_index,
    )


def reactive_schedule(
    configs: list[ModelConfig],
    envelope: Envelope,
    prior: LengthPrior,
    *,
    start_index: int = 0,
    segment_tokens: int = 512,
) -> BurnSchedule:
    """Ablation baseline: burn only when the memory limit is actually hit.

    This is the obvious thing to do without a plan, and the ablation in the
    paper that shows whether planning ahead buys anything.
    """
    validate_family(configs)
    horizon = int(prior.grid[-1])
    n_seg = int(np.ceil(horizon / segment_tokens))
    seg_end_mass = np.zeros(n_seg)
    for t, p in zip(prior.grid, prior.pmf):
        seg_end_mass[min(int((t - 1) // segment_tokens), n_seg - 1)] += p
    seg_survival = np.concatenate([[1.0], 1.0 - np.cumsum(seg_end_mass)[:-1]])
    seg_weight = segment_tokens / max(horizon, 1)

    path, k, p_success, quality, alive = [], start_index, 0.0, 0.0, True
    weight_seen = 0.0
    for s in range(n_seg):
        end_tok = min((s + 1) * segment_tokens, horizon)
        while alive and not envelope.feasible(configs[k], end_tok):
            if k + 1 < len(configs):
                k += 1
            else:
                alive = False
        if alive:
            p_success += seg_end_mass[s]
            quality += seg_survival[s] * configs[k].quality * seg_weight
            weight_seen += seg_survival[s] * seg_weight
        path.append(k)
    quality = quality / weight_seen if weight_seen > 0 else 0.0

    return BurnSchedule(
        segment_tokens=segment_tokens,
        config_index_per_segment=path,
        configs=configs,
        p_success=float(p_success),
        expected_quality=float(quality),
        objective=float(p_success),
        start_index=start_index,
    )
