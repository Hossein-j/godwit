#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Measure GEI's range gain: tokens completed before the memory envelope fails.

One condition per process, so every run starts from a clean allocator:

  A  large configuration, never burns (the fixed-model baseline)
  B  GEI: start large, burn to the small configuration when the envelope binds
  C  small configuration from the start (lower bound on the horizon cost)
  A-dynamic, B-dynamic  A and B with HF's DynamicCache instead of the block
             cache (the ablation for the cache)

The envelope is enforced with torch.cuda.set_per_process_memory_fraction: the
caching allocator may reserve at most M bytes, fragmentation and transients
included (the CUDA context sits outside it). The cache is gei.kvcache's block
cache: 512-token blocks allocated as the context grows, never pre-sized.

"Binds" (B, B-dynamic) is detected by probing: before a step that will
allocate, try to allocate what that step needs plus a margin; if it fails,
burn, then take the step. The block cache allocates one block row every
block_tokens tokens. DynamicCache reallocates every layer on every step via
torch.cat, so its probe is one layer's new K and V at length T+1, every step. Generation ignores end-of-sequence and runs to failure.

The model's nesting is untrained (prefix slicing of a dense FFN), so this
measures horizon and memory only, never quality.

Usage:
    python scripts/range_gain.py --condition A --envelope-gib 15.75
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # never touch the network

import torch

from gei.elastic import HFElasticModel, KVPolicy
from gei.revisions import pinned
from gei.kvcache import BlockLayer

GiB, MiB = 1024**3, 1024**2
PROMPT = "Write a very long, detailed story about a bird crossing an ocean."


def next_allocation(m) -> int:
    """Bytes the next step must allocate at once (0 if none)."""
    if m._kv_cache == "block":
        return row_bytes(m) if needs_new_row(m) else 0
    layer = m._cache.layers[0]  # torch.cat: new K and V at length T+1, old still live
    per_token = layer.keys.nbytes // layer.keys.shape[-2]
    return 2 * (layer.keys.nbytes + per_token)


def needs_new_row(m) -> bool:
    """Will the next step allocate a new block in every full-attention layer?"""
    layer = next(x for x in m._cache.layers if isinstance(x, BlockLayer))
    return layer.length == len(layer.key_blocks) * layer.block_tokens


def row_bytes(m) -> int:
    layers = [x for x in m._cache.layers if isinstance(x, BlockLayer)]
    return sum(b.nbytes for x in layers for b in (x.key_blocks[0], x.value_blocks[0]))


def fits(nbytes: int) -> bool:
    try:
        probe = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        return False
    del probe
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", choices=["A", "B", "C", "A-dynamic", "B-dynamic"],
                    required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--widths", type=int, nargs=2, default=[12288, 6144])
    ap.add_argument("--envelope-gib", type=float, default=15.75)
    ap.add_argument("--block-tokens", type=int, default=512)
    ap.add_argument("--margin-mib", type=int, default=64,
                    help="headroom beyond one block row before B burns")
    ap.add_argument("--max-tokens", type=int, default=60_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    total = torch.cuda.get_device_properties(0).total_memory
    envelope = int(args.envelope_gib * GiB)
    torch.cuda.set_per_process_memory_fraction(envelope / total)

    t_load = time.time()
    kv_cache = "dynamic" if args.condition.endswith("-dynamic") else "block"
    m = HFElasticModel.from_pretrained(args.model, args.widths, device="cuda",
                                       kv_cache=kv_cache, block_tokens=args.block_tokens)
    tok = AutoTokenizer.from_pretrained(args.model, revision=pinned(args.model))
    load_s = time.time() - t_load
    torch.cuda.synchronize()
    after_load = {"allocated": torch.cuda.memory_allocated(),
                  "reserved": torch.cuda.memory_reserved()}

    if args.condition == "C":
        m.switch_to(1, kv_policy=KVPolicy.REUSE)
    torch.cuda.synchronize()
    before_gen = {"allocated": torch.cuda.memory_allocated(),
                  "reserved": torch.cuda.memory_reserved()}

    ids = tok(PROMPT).input_ids
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m.prefill(ids)
    generated, burn, failure = 0, None, None
    burns = args.condition in ("B", "B-dynamic")
    try:
        while generated < args.max_tokens:
            need = next_allocation(m) if burns and m.current_index == 0 else 0
            if need and not fits(need + args.margin_mib * MiB):
                t_burn = time.time()
                torch.cuda.synchronize()
                a = torch.cuda.memory_allocated()
                m.switch_to(1, kv_policy=KVPolicy.REUSE)
                torch.cuda.synchronize()
                burn = {"at_generated_token": generated, "context_tokens": len(ids) + generated,
                        "probe_bytes": need + args.margin_mib * MiB,
                        "freed_bytes": a - torch.cuda.memory_allocated(),
                        "switch_ms": (time.time() - t_burn) * 1e3,
                        "peak_reserved_before": torch.cuda.max_memory_reserved()}
                print(f"burn at generated token {generated}: freed "
                      f"{burn['freed_bytes'] / GiB:.4f} GiB in {burn['switch_ms']:.0f} ms",
                      flush=True)
            m.step()
            generated += 1
            if generated % 2000 == 0:
                print(f"  {generated} tokens, kv {m.kv_bytes() / GiB:.3f} GiB, "
                      f"{(time.time() - t0) / generated * 1e3:.1f} ms/token", flush=True)
    except torch.OutOfMemoryError as exc:
        failure = str(exc).split(".")[0]
    gen_s = time.time() - t0
    torch.cuda.synchronize()

    result = {
        "condition": args.condition, "kv_cache": kv_cache, "model": args.model,
        "widths": args.widths,
        "envelope_bytes": envelope, "device_total_bytes": total,
        "block_tokens": args.block_tokens, "prompt_tokens": len(ids),
        "generated_tokens": generated, "context_tokens": len(ids) + generated,
        "stopped_by": "out_of_memory" if failure else "max_tokens", "failure": failure,
        "kv_bytes_at_end": m.kv_bytes(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "after_load": after_load, "before_generation": before_gen,
        "weight_bytes": [c.weight_bytes for c in m.configs],
        "burn": burn, "load_seconds": load_s, "generation_seconds": gen_s,
        "ms_per_token": gen_s / max(generated, 1) * 1e3,
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("after_load",)}, indent=1))
    out = Path(args.out or f"results/range_gain_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
