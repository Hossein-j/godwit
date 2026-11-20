#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Measure the envelope constants on this machine. Do this before any result.

Three numbers in configs/*.yaml are guesses until measured here:

  kv_bytes_per_token   derived from the model's own architecture
  reserve_bytes        activations + fragmentation, measured under load
  alpha_joules_per_byte  energy per byte read, from power draw and throughput

Usage:
    python scripts/calibrate.py --model google/gemma-3n-E4B-it
    python scripts/calibrate.py --model <id> --skip-power   # no nvidia-smi
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

GB = 1024 ** 3


def kv_bytes_per_token(cfg) -> tuple[int, dict]:
    """KV cache bytes added per token: 2 * layers * kv_heads * head_dim * dtype.

    Grouped-query attention makes this much smaller than naive estimates, so
    read kv_heads from the config rather than assuming it equals the head count.
    """
    layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads if hidden and heads else None)
    if not all([layers, kv_heads, head_dim]):
        raise SystemExit(
            "Could not read layers/kv_heads/head_dim from the model config; "
            "fill kv_bytes_per_token in the YAML by hand."
        )
    dtype_bytes = 2  # fp16/bf16 cache
    per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return per_token, {
        "layers": layers,
        "attention_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "note": "halve for an 8-bit KV cache; quarter for 4-bit",
    }


def gpu_power_watts() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Hugging Face model id or path")
    ap.add_argument("--tokens", type=int, default=512, help="tokens to time")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/calibration.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from gei.revisions import pinned

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device; calibration must run on the target GPU.")

    print(f"Model: {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, revision=pinned(args.model), trust_remote_code=False)
    per_token, detail = kv_bytes_per_token(cfg)
    print(f"  KV per token: {per_token / 1024:.1f} KiB  ({detail})")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / GB:.1f} GB")

    print("\nLoading model (this is the slow part)...")
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.model, revision=pinned(args.model))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=pinned(args.model), torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    weights = torch.cuda.memory_allocated()
    print(f"  resident weights: {weights / GB:.2f} GB")

    print(f"\nGenerating {args.tokens} tokens to measure overhead and speed...")
    inputs = tok("Explain the plan step by step.", return_tensors="pt").to("cuda:0")
    power_before = None if args.skip_power else gpu_power_watts()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    power_after = None if args.skip_power else gpu_power_watts()

    generated = out.shape[-1] - inputs["input_ids"].shape[-1]
    peak = torch.cuda.max_memory_allocated()
    expected_kv = per_token * out.shape[-1]
    # Everything resident beyond weights and the KV cache is the reserve the
    # envelope must hold back: activations, workspace, fragmentation.
    reserve = max(peak - weights - expected_kv, 0)
    tps = generated / elapsed

    print(f"  {generated} tokens in {elapsed:.1f}s = {tps:.1f} tok/s")
    print(f"  peak allocated: {peak / GB:.2f} GB")
    print(f"  implied reserve: {reserve / GB:.2f} GB")

    alpha = None
    if power_before and power_after:
        watts = (power_before + power_after) / 2
        joules_per_token = watts / tps
        bytes_read = weights + expected_kv / 2  # average over the generation
        alpha = joules_per_token / bytes_read
        print(f"  ~{watts:.0f} W, {joules_per_token:.3f} J/token")
        print(f"  alpha_joules_per_byte: {alpha:.3e}")
    else:
        print("  power not sampled; set alpha_joules_per_byte by hand or rerun "
              "without --skip-power")

    result = {
        "model": args.model,
        "gpu": props.name,
        "total_memory_gb": props.total_memory / GB,
        "kv_bytes_per_token": per_token,
        "kv_detail": detail,
        "weight_bytes": weights,
        "peak_bytes": peak,
        "reserve_bytes": reserve,
        "reserve_gb": reserve / GB,
        "tokens_per_second": tps,
        "alpha_joules_per_byte": alpha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}. Copy these into your config's `envelope` block.")
    print("Reserve measured on a short run is a floor: rerun near your longest "
          "context before trusting it.")


if __name__ == "__main__":
    main()
