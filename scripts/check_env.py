#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Verify the machine can run the GEI experiments. Run this first.

Checks GPU, VRAM, torch/CUDA, system RAM and disk, and flags anything that
would bite later (a 24 GB card is enough, but only with the right precision).
"""

from __future__ import annotations

import shutil
import sys

MIN_VRAM_GB = 20.0
MIN_RAM_GB = 32.0
MIN_DISK_GB = 200.0


def ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    failures = 0
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        bad("Python 3.10+ required")
        failures += 1

    print("\nGPU")
    try:
        import torch
    except ImportError:
        bad("torch not installed — run: pip install -e '.[gpu]'")
        return 1

    print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        bad("CUDA not available to torch; check driver and the torch build")
        failures += 1
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            line = f"{props.name}: {vram:.1f} GB, capability {props.major}.{props.minor}"
            if vram >= MIN_VRAM_GB:
                ok(line)
            else:
                warn(line + f" (< {MIN_VRAM_GB} GB; use 8-bit or a smaller model)")
            if props.major >= 8:
                ok("bf16 supported (Ampere or newer)")
            else:
                warn("no bf16; fp16 only, watch for overflow in long generations")

    print("\nHost")
    try:
        import psutil

        ram = psutil.virtual_memory().total / 1024**3
        (ok if ram >= MIN_RAM_GB else warn)(
            f"{ram:.0f} GB RAM" + ("" if ram >= MIN_RAM_GB else " — 64 GB recommended")
        )
    except ImportError:
        warn("psutil not installed; skipped RAM check")

    free = shutil.disk_usage(".").free / 1024**3
    (ok if free >= MIN_DISK_GB else warn)(f"{free:.0f} GB free disk")

    print("\nPackages")
    for name in ("numpy", "yaml", "transformers", "datasets", "accelerate"):
        try:
            mod = __import__(name)
            ok(f"{name} {getattr(mod, '__version__', '?')}")
        except ImportError:
            if name in ("numpy", "yaml"):
                bad(f"{name} missing — run: pip install -e .")
                failures += 1
            else:
                warn(f"{name} missing — needed for GPU experiments only")

    print("\nGEI core (CPU)")
    try:
        from gei import GB, Envelope, LengthPrior, ModelConfig, launch_check, range_gain

        fam = [
            ModelConfig("large", 8 * GB, 1.0),
            ModelConfig("small", 3 * GB, 0.85),
        ]
        env = Envelope(memory_bytes=24 * GB, kv_bytes_per_token=128 * 1024,
                       reserve_bytes=2 * GB)
        gain = range_gain(fam, env)
        ok(
            f"range gain {gain['fixed_max_tokens']:,} -> "
            f"{gain['burned_max_tokens']:,} tokens "
            f"({gain['extension_ratio']:.2f}x)"
        )
        prior = LengthPrior.lognormal(8000, 0.6, 32_000)
        ok(f"launch check runs: go={launch_check(fam, env, prior, tau=0.9).go}")
    except Exception as exc:  # noqa: BLE001
        bad(f"GEI core failed: {exc}")
        failures += 1

    print("\n" + ("All good." if failures == 0 else f"{failures} blocking problem(s)."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
