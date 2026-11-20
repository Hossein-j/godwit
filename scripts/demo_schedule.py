#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Show what GEI does, end to end, with no GPU and no model.

Run this first to see whether the idea behaves the way the paper claims:
burning extends the horizon, the launch check declines impossible tasks, and a
planned schedule beats reacting at the limit.

    python scripts/demo_schedule.py
    python scripts/demo_schedule.py --config configs/rtx4090.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gei import (
    GB,
    Envelope,
    LengthPrior,
    MockElasticModel,
    ModelConfig,
    fly,
    launch_check,
    range_gain,
    reactive_schedule,
    solve_schedule,
)


def load_config(path: Path | None):
    """Build the family, envelope and prior from YAML, or use demo defaults."""
    if path is None:
        family = [
            ModelConfig("E4B", int(4.4 * GB), 1.00),
            ModelConfig("E3B", int(3.2 * GB), 0.95),
            ModelConfig("E2B", int(2.0 * GB), 0.88),
        ]
        envelope = Envelope(
            memory_bytes=8 * GB,  # deliberately tight, as on a phone or shared GPU
            kv_bytes_per_token=128 * 1024,
            reserve_bytes=1 * GB,
        )
        prior = LengthPrior.lognormal(8000, 0.6, 32_000)
        return family, envelope, prior, 512, 0.95

    import yaml

    cfg = yaml.safe_load(path.read_text())
    family = [
        ModelConfig(f["name"], int(f["weight_gb"] * GB), float(f["quality"]))
        for f in cfg["family"]
    ]
    dev, envc = cfg["device"], cfg["envelope"]
    envelope = Envelope(
        memory_bytes=int(dev["total_memory_gb"] * GB),
        kv_bytes_per_token=float(envc["kv_bytes_per_token"]),
        alpha_joules_per_byte=float(envc.get("alpha_joules_per_byte", 0.0)),
        energy_joules=float(envc.get("energy_joules", float("inf"))),
        prompt_kv_bytes=float(envc.get("prompt_kv_bytes", 0)),
        reserve_bytes=int(dev.get("reserve_gb", 0) * GB),
    )
    p = cfg["prior"]
    prior = LengthPrior.lognormal(
        p["median_tokens"], p["sigma"], p["max_tokens"]
    )
    return (
        family,
        envelope,
        prior,
        int(cfg["schedule"]["segment_tokens"]),
        float(cfg["launch"]["tau"]),
    )


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    family, envelope, prior, segment, tau = load_config(args.config)

    rule("Envelope")
    print(f"  usable memory   {envelope.usable_memory / GB:.1f} GB")
    print(f"  KV per token    {envelope.kv_bytes_per_token / 1024:.0f} KiB")
    print(f"  expected length {prior.mean():,.0f} tokens "
          f"(p95 = {prior.quantile(0.95):,})")

    rule("1. What burning buys")
    gain = range_gain(family, envelope)
    print(f"  fixed at {gain['fixed_config']:>4}: {gain['fixed_max_tokens']:>8,} tokens")
    print(f"  burned to {gain['smallest_config']:>3}: {gain['burned_max_tokens']:>8,} tokens")
    print(f"  extension:       {gain['extension_ratio']:>8.2f}x "
          f"(+{gain['extension_tokens']:,} tokens)")

    rule("2. Launch check")
    decision = launch_check(family, envelope, prior, tau=tau, segment_tokens=segment)
    for line in decision.describe().splitlines():
        print("  " + line)

    rule("3. Planned schedule vs reacting at the limit")
    planned = solve_schedule(family, envelope, prior, segment_tokens=segment)
    reactive = reactive_schedule(family, envelope, prior, segment_tokens=segment)
    print(f"  planned   p(success) = {planned.p_success:.4f}  "
          f"E[quality] = {planned.expected_quality:.4f}")
    print(f"  reactive  p(success) = {reactive.p_success:.4f}  "
          f"E[quality] = {reactive.expected_quality:.4f}")

    rule("4. Flying it (mock model, no GPU)")
    for length in (2_000, 12_000, 30_000):
        model = MockElasticModel(family)
        log = fly(
            model,
            envelope,
            prior,
            true_length=length,
            tau=min(tau, 0.5),
            segment_tokens=segment,
            check_every=segment,
        )
        print(f"  task of {length:>6,} tokens: {log.summary()}")

    print(
        "\nNumbers above come from the config, not from hardware. "
        "Run scripts/calibrate.py on the 4090 before quoting any of them."
    )


if __name__ == "__main__":
    main()
