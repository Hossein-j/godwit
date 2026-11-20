# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Belief over task length T.

The godwit cannot know in advance exactly how long the crossing will take; it
departs on an estimate of distance and wind. GEI is in the same position: an
open-ended agentic task has no known token count at launch. Everything else in
GEI (the launch check, the burn schedule) is conditioned on this belief.

Prior work (RAP, PM-KVQ) assumes the length is known up front. Modelling it as
a distribution is what makes the launch check meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LengthPrior:
    """Discrete distribution over total generated tokens T.

    Attributes
    ----------
    grid:
        Strictly increasing token counts (the right edge of each bin).
    pmf:
        P(T falls in this bin). Sums to 1.
    """

    grid: np.ndarray
    pmf: np.ndarray

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.int64)
        pmf = np.asarray(self.pmf, dtype=np.float64)
        if grid.ndim != 1 or pmf.shape != grid.shape:
            raise ValueError("grid and pmf must be 1-D arrays of equal length")
        if np.any(np.diff(grid) <= 0):
            raise ValueError("grid must be strictly increasing")
        if np.any(pmf < 0):
            raise ValueError("pmf must be non-negative")
        total = pmf.sum()
        if not np.isclose(total, 1.0):
            raise ValueError(f"pmf must sum to 1, got {total}")

    # -- constructors ----------------------------------------------------

    @classmethod
    def lognormal(
        cls,
        median_tokens: float,
        sigma: float,
        max_tokens: int,
        n_bins: int = 64,
    ) -> "LengthPrior":
        """Lognormal prior, the usual shape for generation lengths.

        A long right tail is exactly the risk GEI exists to handle: most tasks
        are short, a few run far past the fixed-model horizon.
        """
        if median_tokens <= 0 or sigma <= 0 or max_tokens <= 0:
            raise ValueError("median_tokens, sigma and max_tokens must be positive")
        grid = np.unique(
            np.linspace(max_tokens / n_bins, max_tokens, n_bins).astype(np.int64)
        )
        mu = np.log(median_tokens)
        # P(T <= edge) under the lognormal, differenced into bin mass.
        from math import erf, sqrt

        def cdf(x: float) -> float:
            if x <= 0:
                return 0.0
            z = (np.log(x) - mu) / (sigma * sqrt(2.0))
            return 0.5 * (1.0 + erf(z))

        edges = np.concatenate([[0], grid])
        mass = np.array(
            [cdf(float(hi)) - cdf(float(lo)) for lo, hi in zip(edges[:-1], edges[1:])]
        )
        if mass.sum() <= 0:
            raise ValueError("prior has no mass on the grid; raise max_tokens")
        # Mass beyond max_tokens is folded into the last bin: a task longer than
        # the grid is still a task that must finish, and dropping it would make
        # the launch check optimistic.
        mass[-1] += 1.0 - cdf(float(grid[-1]))
        return cls(grid=grid, pmf=mass / mass.sum())

    @classmethod
    def empirical(cls, observed_lengths, n_bins: int = 64) -> "LengthPrior":
        """Prior fitted to logged task lengths — the production path.

        Once runs are logged, replace the lognormal guess with this.
        """
        lengths = np.asarray(list(observed_lengths), dtype=np.float64)
        if lengths.size == 0:
            raise ValueError("need at least one observed length")
        counts, edges = np.histogram(lengths, bins=n_bins)
        grid = np.unique(np.ceil(edges[1:]).astype(np.int64))
        counts = counts[: grid.size].astype(np.float64)
        if counts.sum() <= 0:
            raise ValueError("histogram is empty")
        return cls(grid=grid, pmf=counts / counts.sum())

    # -- queries ---------------------------------------------------------

    def survival(self) -> np.ndarray:
        """P(T >= grid[i]): probability the task is still running at each bin."""
        return np.concatenate([[1.0], 1.0 - np.cumsum(self.pmf)[:-1]])

    def p_at_most(self, tokens: int) -> float:
        """P(T <= tokens) — the success probability of a horizon this long."""
        return float(self.pmf[self.grid <= tokens].sum())

    def condition_on_progress(self, tokens_done: int) -> "LengthPrior":
        """Update the belief mid-flight: the task has not ended yet.

        The controller calls this every N tokens. Truncating already-passed
        mass is the minimum sensible update; a learned estimator that also
        reads the generated text would replace this.
        """
        keep = self.grid > tokens_done
        if not keep.any():
            raise ValueError(
                f"progress {tokens_done} exceeds the whole prior grid; "
                "widen max_tokens"
            )
        pmf = self.pmf[keep]
        if pmf.sum() <= 0:
            # All surviving bins had zero mass; fall back to uniform over them
            # rather than crashing a run in flight.
            pmf = np.ones_like(pmf)
        return LengthPrior(grid=self.grid[keep], pmf=pmf / pmf.sum())

    def mean(self) -> float:
        return float((self.grid * self.pmf).sum())

    def quantile(self, q: float) -> int:
        """Smallest grid point with cumulative mass >= q."""
        if not 0.0 < q <= 1.0:
            raise ValueError("q must lie in (0, 1]")
        idx = int(np.searchsorted(np.cumsum(self.pmf), q))
        return int(self.grid[min(idx, self.grid.size - 1)])
