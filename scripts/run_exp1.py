#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Experiment 1: can a shrunken model read its larger self's KV cache?

This single question decides whether GEI is viable. Four conditions:

    A  large throughout                      upper bound
    B  small throughout                      reference
    C  large, switch to small mid-generation, reuse the KV cache   <- GEI
    D  same switch, but recompute the cache with the small model   fallback

Decision rule (from the plan):
    C matches B            -> KV reuse works; build the controller
    C below B, D recovers  -> add partial recomputation, budget its cost
    both fail              -> switch-aware ("burnability") training, then rerun

Modes:
    --backend mock   runs anywhere, exercises the pipeline, proves nothing
                     about real models
    --backend hf     the real experiment; requires HFElasticModel to be
                     implemented against your chosen checkpoint
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gei.elastic import KVPolicy, MockElasticModel
from gei.envelope import GB, ModelConfig
from gei.metrics import bootstrap_ci, js_distance, kl_divergence


@dataclass
class TrialResult:
    condition: str
    problem_id: int
    seed: int
    switch_point: float
    correct: bool
    mean_quality: float
    switch_shock_kl: float | None = None


def _family() -> list[ModelConfig]:
    return [
        ModelConfig("large", int(4.4 * GB), 1.00),
        ModelConfig("small", int(2.0 * GB), 0.88),
    ]


def run_mock_trial(
    condition: str, problem_id: int, seed: int, switch_point: float, n_tokens: int
) -> TrialResult:
    """One generation under one condition, using the CPU mock.

    The mock's `switch_shock` is a *parameter*, not a measurement. This path
    verifies the harness and the statistics; it cannot answer the scientific
    question.
    """
    family = _family()
    model = MockElasticModel(family, switch_shock=0.15, shock_decay_tokens=200, seed=seed)
    model.prefill(list(range(64)))

    if condition == "B_small":
        model.switch_to(1, kv_policy=KVPolicy.RECOMPUTE)

    switch_at = int(n_tokens * switch_point)
    qualities, shock_probs, ref_probs = [], [], []

    for t in range(n_tokens):
        if condition in ("C_reuse", "D_recompute") and t == switch_at:
            policy = KVPolicy.REUSE if condition == "C_reuse" else KVPolicy.RECOMPUTE
            model.switch_to(1, kv_policy=policy)
        step = model.step()
        qualities.append(model.effective_quality())
        # Collect the window used for switch shock: the 200 tokens after the switch.
        if condition in ("C_reuse", "D_recompute") and switch_at <= t < switch_at + 200:
            shock_probs.append(np.exp(step.logprobs))

    shock = None
    if shock_probs:
        # Reference: the same model run small from the start, same seed.
        ref = MockElasticModel(family, switch_shock=0.15, seed=seed)
        ref.prefill(list(range(64)))
        ref.switch_to(1, kv_policy=KVPolicy.RECOMPUTE)
        for _ in range(switch_at):
            ref.step()
        ref_probs = [np.exp(ref.step().logprobs) for _ in range(len(shock_probs))]
        shock = float(
            np.mean([kl_divergence(p, q) for p, q in zip(shock_probs, ref_probs)])
        )

    mean_quality = float(np.mean(qualities))
    rng = np.random.default_rng(seed * 10_000 + problem_id)
    correct = bool(rng.random() < mean_quality)  # quality as success probability
    return TrialResult(
        condition=condition,
        problem_id=problem_id,
        seed=seed,
        switch_point=switch_point,
        correct=correct,
        mean_quality=mean_quality,
        switch_shock_kl=shock,
    )


def summarise(results: list[TrialResult]) -> dict:
    out: dict[str, dict] = {}
    for cond in sorted({r.condition for r in results}):
        rows = [r for r in results if r.condition == cond]
        acc = [float(r.correct) for r in rows]
        lo, hi = bootstrap_ci(acc)
        shocks = [r.switch_shock_kl for r in rows if r.switch_shock_kl is not None]
        out[cond] = {
            "n": len(rows),
            "accuracy": float(np.mean(acc)),
            "ci95": [lo, hi],
            "mean_quality": float(np.mean([r.mean_quality for r in rows])),
            "switch_shock_kl": float(np.mean(shocks)) if shocks else None,
        }
    return out


def verdict(summary: dict) -> str:
    """Apply the plan's decision rule to the numbers."""
    b, c, d = summary.get("B_small"), summary.get("C_reuse"), summary.get("D_recompute")
    if not (b and c):
        return "incomplete: need both B_small and C_reuse"
    c_overlaps_b = c["ci95"][1] >= b["ci95"][0]
    if c_overlaps_b:
        return (
            "PASS (1): C matches B within noise -> KV reuse works. "
            "Proceed to the full controller."
        )
    if d and d["ci95"][1] >= b["ci95"][0]:
        return (
            "PARTIAL (2): C is below B but D recovers -> add partial KV "
            "recomputation and budget its cost."
        )
    return (
        "FAIL (3): neither reuse nor recompute matches B -> switch-aware "
        "('burnability') training is required before GEI is viable."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--switch-points", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out", type=Path, default=Path("results/exp1.json"))
    args = ap.parse_args()

    if args.backend == "hf":
        raise SystemExit(
            "The hf backend needs HFElasticModel implemented first "
            "(src/gei/elastic.py). Pick the checkpoint, confirm its nesting "
            "structure, verify switch_to actually frees VRAM, then wire it here."
        )

    print(f"Experiment 1 [{args.backend}]: {args.n_problems} problems x "
          f"{len(args.seeds)} seeds x {len(args.switch_points)} switch points")
    print("NOTE: the mock backend validates the harness only. Its switch shock "
          "is a parameter, not a measurement.\n")

    results: list[TrialResult] = []
    for sp in args.switch_points:
        for seed in args.seeds:
            for pid in range(args.n_problems):
                for cond in ("A_large", "B_small", "C_reuse", "D_recompute"):
                    if cond in ("A_large", "B_small") and sp != args.switch_points[0]:
                        continue  # these do not depend on the switch point
                    results.append(run_mock_trial(cond, pid, seed, sp, args.tokens))

    summary = summarise(results)
    print(f"{'condition':<14}{'n':>6}{'accuracy':>12}{'95% CI':>22}{'shock KL':>12}")
    for cond, s in summary.items():
        ci = f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
        shock = f"{s['switch_shock_kl']:.4f}" if s["switch_shock_kl"] else "-"
        print(f"{cond:<14}{s['n']:>6}{s['accuracy']:>12.3f}{ci:>22}{shock:>12}")

    print("\n" + verdict(summary))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "config": vars(args) | {"out": str(args.out)},
                "summary": summary,
                "trials": [asdict(r) for r in results],
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
