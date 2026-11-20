#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Range gain on a deployable trained-nesting checkpoint: Gemma 3n E4B, batched.

Pre-registered in docs/RANGE_GAIN_BATCHED_PREREG.md. The batch-1 run on
Qwen3-8B (docs/RANGE_GAIN.md) validated the mechanism on a model whose nesting
is untrained; the scoping criterion (docs/MODEL_SELECTION.md) says a
trained-nesting checkpoint like Gemma 3n only binds under batching. This runs
it there, one condition per process:

  A          large configuration, never burns
  B          burn to the small configuration when the envelope binds
  C          small configuration from the start
  A-dynamic, B-dynamic   A and B with HF's DynamicCache (the cache ablation)

All B sequences decode the same prompt in lockstep (no padding, so the fused
decode kernel applies); generation ignores end-of-sequence and runs until the
envelope fails. The small configuration is the mix-and-match variant used in
Experiment 1 (layers 0-9 at full width). This measures horizon and memory, not
quality.
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
MODEL = "google/gemma-3n-E4B-it"
PROMPT = "Write a very long, detailed story about a bird crossing an ocean."


def next_allocation(m) -> int:
    """Bytes the next step must allocate at once beyond its transients (0 if none)."""
    if m._kv_cache == "block":
        blocks = [x for x in m._cache.layers if isinstance(x, BlockLayer)]
        first = blocks[0]
        if first.length == len(first.key_blocks) * first.block_tokens:  # new row due
            return sum(b.nbytes for x in blocks for b in (x.key_blocks[0], x.value_blocks[0]))
        return 0
    # DynamicCache: torch.cat allocates each layer's new K and V at length T+1
    # while the old ones are live; the largest layer bounds the step's peak.
    sizes = [x.keys.nbytes for x in m._cache.layers if getattr(x, "keys", None) is not None
             and x.keys.numel()]
    biggest = max(sizes)
    layer = max((x for x in m._cache.layers if getattr(x, "keys", None) is not None
                 and x.keys.numel()), key=lambda x: x.keys.nbytes)
    per_token = layer.keys.nbytes // layer.keys.shape[-2]
    return 2 * (biggest + per_token)


def fits(nbytes: int) -> bool:
    try:
        probe = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        return False
    del probe
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", required=True,
                    choices=["A", "B", "C", "A-dynamic", "B-dynamic"])
    ap.add_argument("--envelope-gib", type=float, default=13.40,
                    help="allocator cap; 0 disables it (smoke tests only)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block-tokens", type=int, default=512)
    ap.add_argument("--margin-mib", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=30_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Gemma3nForCausalLM

    total = torch.cuda.get_device_properties(0).total_memory
    envelope = int(args.envelope_gib * GiB)
    if envelope:
        torch.cuda.set_per_process_memory_fraction(envelope / total)

    kv_cache = "dynamic" if args.condition.endswith("-dynamic") else "block"
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), device="cuda",
                       kv_cache=kv_cache, block_tokens=args.block_tokens)
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
    m.prefill_batch([ids] * args.batch)
    steps, burn, failure = 0, None, None
    burns = args.condition in ("B", "B-dynamic")
    try:
        while steps < args.max_tokens:
            need = next_allocation(m) if burns and m.current_index == 0 else 0
            if need and not fits(need + args.margin_mib * MiB):
                t_burn = time.time()
                torch.cuda.synchronize()
                a = torch.cuda.memory_allocated()
                m.switch_to(1, kv_policy=KVPolicy.REUSE)
                torch.cuda.synchronize()
                burn = {"at_step": steps, "context_tokens": len(ids) + steps,
                        "probe_bytes": need + args.margin_mib * MiB,
                        "freed_bytes": a - torch.cuda.memory_allocated(),
                        "switch_ms": (time.time() - t_burn) * 1e3}
                print(f"burn at step {steps}: freed {burn['freed_bytes'] / GiB:.4f} GiB "
                      f"in {burn['switch_ms']:.0f} ms", flush=True)
            m.step_batch()
            steps += 1
            if steps % 2000 == 0:
                print(f"  {steps} steps, kv {m.kv_bytes() / GiB:.3f} GiB, "
                      f"{(time.time() - t0) / steps * 1e3:.1f} ms/step", flush=True)
    except torch.OutOfMemoryError as exc:
        failure = str(exc).split(".")[0]
    gen_s = time.time() - t0
    torch.cuda.synchronize()

    result = {
        "condition": args.condition, "kv_cache": kv_cache, "model": MODEL,
        "batch": args.batch, "envelope_bytes": envelope, "device_total_bytes": total,
        "block_tokens": args.block_tokens, "margin_mib": args.margin_mib,
        "prompt_tokens": len(ids), "steps": steps, "context_tokens": len(ids) + steps,
        "stopped_by": "out_of_memory" if failure else "max_tokens", "failure": failure,
        "kv_bytes_at_end": m.kv_bytes(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "after_load": after_load, "before_generation": before_gen,
        "weight_bytes": [c.weight_bytes for c in m.configs], "burn": burn,
        "load_seconds": load_s, "generation_seconds": gen_s,
        "ms_per_step": gen_s / max(steps, 1) * 1e3,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "after_load"}, indent=1))
    out = Path(args.out or f"results/range_gain_batched_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
