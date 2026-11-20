# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""The launch check: decide before committing whether the task can finish.

A godwit does not take off and hope. It departs when fuel load and winds make
the crossing feasible. GEI does the same: before the first token, estimate the
task length, solve for the best schedule from each possible starting
configuration, and either go, go leaner, or decline.

No prior method does this. RAP and PM-KVQ are told the length in advance, so
the question never arises.
"""

from __future__ import annotations

from dataclasses import dataclass

from .envelope import Envelope, ModelConfig
from .length_prior import LengthPrior
from .schedule import BurnSchedule, solve_schedule


@dataclass
class LaunchDecision:
    """What the launch check decided, and why."""

    go: bool
    start_index: int | None
    schedule: BurnSchedule | None
    p_success: float
    tau: float
    reason: str

    def describe(self) -> str:
        if self.go and self.schedule is not None:
            head = (
                f"GO from {self.schedule.configs[self.start_index].name} "
                f"(p={self.p_success:.3f} >= tau={self.tau:.2f})"
            )
            return head + "\n" + self.schedule.describe()
        return f"NO-GO: {self.reason} (best p={self.p_success:.3f}, tau={self.tau:.2f})"


def launch_check(
    configs: list[ModelConfig],
    envelope: Envelope,
    prior: LengthPrior,
    *,
    tau: float = 0.95,
    segment_tokens: int = 512,
    lam: float = 0.0,
    energy_buckets: int = 64,
) -> LaunchDecision:
    """Pick the largest starting configuration that still clears the threshold.

    Sweeping largest-first matters: quality is highest at the top of the
    family, so GEI should only start lean when it has to. This mirrors the
    godwit reducing organs before departure only as far as the crossing
    demands.

    Returns the first start index whose optimal schedule reaches p >= tau, or a
    NO-GO carrying the best probability achievable.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0, 1]")

    best_p, best_schedule, best_index = -1.0, None, None
    for start in range(len(configs)):
        schedule = solve_schedule(
            configs,
            envelope,
            prior,
            start_index=start,
            segment_tokens=segment_tokens,
            lam=lam,
            energy_buckets=energy_buckets,
        )
        if schedule.p_success > best_p:
            best_p, best_schedule, best_index = schedule.p_success, schedule, start
        if schedule.p_success >= tau:
            return LaunchDecision(
                go=True,
                start_index=start,
                schedule=schedule,
                p_success=schedule.p_success,
                tau=tau,
                reason="threshold met",
            )

    return LaunchDecision(
        go=False,
        start_index=best_index,
        schedule=best_schedule,
        p_success=max(best_p, 0.0),
        tau=tau,
        reason="no starting configuration reaches tau; envelope too small "
        "or task too long",
    )
