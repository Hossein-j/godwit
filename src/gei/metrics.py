# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Metrics for the GEI experiments.

Experiment 1 needs switch shock (how much the distribution moves right after a
mid-generation switch) and task accuracy. Experiment 4 needs calibration of the
launch check. Everything reports an interval, not a bare number: with 100
problems per condition, a difference without error bars says nothing.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, for next-token distributions.

    Experiment 1 uses this as "switch shock": p from the switched run
    (condition C), q from the small model run from scratch (condition B). Near
    zero means the shrunken model is behaving as if it had been small all
    along.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("distributions must have the same shape")
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], _EPS))))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance: symmetric, bounded, safe when supports differ."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    m = 0.5 * (p + q)
    return float(np.sqrt(max(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m), 0.0)))


def pass_at_1(correct: list[bool]) -> float:
    """Fraction of problems solved on a single sample."""
    arr = np.asarray(correct, dtype=bool)
    if arr.size == 0:
        raise ValueError("need at least one result")
    return float(arr.mean())


def bootstrap_ci(
    values, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean.

    Report this beside every accuracy number in the paper.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("need at least one value")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a, b, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b with a percentile bootstrap interval, for paired results.

    Use when both conditions ran on the same problems (Experiment 1): problems
    are resampled jointly, so per-problem difficulty cancels and the interval
    is much tighter than comparing two unpaired intervals for overlap.
    Returns (mean difference, low, high).
    """
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two non-empty results of the same length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    lo, hi = np.quantile(diff[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def mcnemar_exact(a, b) -> dict:
    """McNemar's exact test for paired binary outcomes (two-sided).

    Only discordant pairs carry information: `a_only` problems solved by a but
    not b, and `b_only` the reverse. Under the null each discordant pair is a
    fair coin, so p = 2 * P(Binomial(n, 1/2) <= min(a_only, b_only)), capped at 1.
    Exact, so valid at the small counts a 50-problem cell produces.
    """
    from math import comb

    a = np.asarray(list(a), dtype=bool)
    b = np.asarray(list(b), dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two non-empty results of the same length")
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": min(1.0, 2 * tail)}


def expected_calibration_error(
    predicted: list[float], outcomes: list[bool], *, n_bins: int = 10
) -> float:
    """ECE for the launch check (Experiment 4).

    The launch check claims p(finish). ECE asks whether, among tasks it gave
    0.8, about 80% actually finished. A confident but wrong launch check is
    worse than none.
    """
    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("predicted and outcomes must align")
    if p.size == 0:
        raise ValueError("need at least one prediction")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_curve(
    predicted: list[float], outcomes: list[bool], *, n_bins: int = 10
) -> dict:
    """Points for the reliability diagram in the Experiment 4 figure."""
    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf.append(float(p[mask].mean()))
        acc.append(float(y[mask].mean()))
        count.append(int(mask.sum()))
    return {"confidence": conf, "accuracy": acc, "count": count}


def tokens_per_joule(tokens: int, joules: float) -> float:
    """Experiment 5's headline number."""
    if joules <= 0:
        raise ValueError("joules must be positive")
    return tokens / joules
