# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Decode throughput by attention path, at matched context lengths.

Paths (one per process, so each starts from a clean allocator):

  kernel   block cache, fused Triton paged-decode kernel (gei.paged_decode)
  loop     block cache, Python loop over blocks (the reference path)
  sdpa     HF DynamicCache, SDPA with PyTorch's default backend choice
  flash    HF DynamicCache, SDPA restricted to its FlashAttention backend
           (PyTorch's built-in FlashAttention-2 kernel; the flash_attn
           package is not installed here)

The model runs through HFElasticModel._forward, the call the range-gain runs
used, at the small configuration (the one that decodes the long contexts in
those runs). The context is grown by chunked prefill to each length, then
greedy decode steps are timed; the per-step logprob copy to the host that
step_batch makes is left out, since it is the same for every path. Token ids
are random: attention cost does not depend on them.

  python scripts/bench_decode_attention.py --model qwen3-8b --path kernel
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import torch

from gei import HFElasticModel, KVPolicy, kvcache, paged_decode
from gei.revisions import pinned

MODELS = {
    # name: (HF id, widths, exclude_layers, batch, prefill chunk)
    "qwen3-8b": ("Qwen/Qwen3-8B", [12288, 6144], (), 1, 1024),
    "gemma3n-e4b": ("google/gemma-3n-E4B-it", [16384, 8192], range(10), 16, 512),
}
CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32000]


def load(name: str, path: str) -> HFElasticModel:
    hf_id, widths, exclude, _, _ = MODELS[name]
    kv = "block" if path in ("kernel", "loop") else "dynamic"
    if name == "qwen3-8b":
        m = HFElasticModel.from_pretrained(hf_id, widths, device="cuda", kv_cache=kv,
                                           block_tokens=512)
    else:
        from transformers import Gemma3nForCausalLM

        model = Gemma3nForCausalLM.from_pretrained(hf_id, revision=pinned(hf_id), dtype=torch.bfloat16)
        m = HFElasticModel(model, widths, exclude_layers=exclude, device="cuda",
                           kv_cache=kv, block_tokens=512)
    m.switch_to(1, kv_policy=KVPolicy.REUSE)
    return m


def greedy_step(m: HFElasticModel) -> None:
    tokens = m._forward(m._width - 1, m._width).argmax(dim=-1).tolist()
    for row, t in zip(m._rows, tokens):
        row.append(t)


def grow_to(m: HFElasticModel, target: int, chunk: int, gen: torch.Generator,
            vocab: int) -> None:
    """Append random tokens and prefill them in chunks until the cache holds
    target - 1 columns (the next step feeds column target - 1)."""
    have = m._width - 1
    add = target - m._width
    new = torch.randint(0, vocab, (len(m._rows), add), generator=gen).tolist()
    for row, extra in zip(m._rows, new):
        row.extend(extra)
    for s in range(have, target - 1, chunk):
        m._forward(s, min(s + chunk, target - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--path", choices=["kernel", "loop", "sdpa", "flash"], required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    ap.add_argument("--chunk", type=int, default=None, help="prefill chunk (tokens)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default="results/bench_decode_attention.jsonl")
    args = ap.parse_args()

    if args.path == "loop":
        paged_decode.supported = lambda query, kv: False
    sdpa_ctx = nullcontext
    if args.path == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def sdpa_ctx():
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

    _, _, _, batch, chunk = MODELS[args.model]
    chunk = args.chunk or chunk
    m = load(args.model, args.path)
    text = m.model.config.get_text_config()
    # Gemma 3n's per-layer embeddings cover fewer ids than its vocabulary.
    vocab = min(text.vocab_size, getattr(text, "vocab_size_per_layer_input", None)
                or text.vocab_size)
    gen = torch.Generator().manual_seed(0)
    info = {"model": args.model, "path": args.path, "batch": batch,
            "attn_implementation": m.model.config._attn_implementation,
            "kv_cache": m._kv_cache, "config": "small (after burn)",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    print(json.dumps(info), flush=True)
    m.prefill_batch(torch.randint(0, vocab, (batch, 16), generator=gen).tolist())
    for ctx in args.contexts:
        row = dict(info, context=ctx, prefill_chunk=chunk)
        phase = "prefill"
        try:
            grow_to(m, ctx, chunk, gen, vocab)  # prefill: default backends everywhere
            phase = "decode"
            calls0 = kvcache.STATS["paged_decode_calls"]
            with sdpa_ctx():  # the path under test applies to the decode steps
                for _ in range(args.warmup):
                    greedy_step(m)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    greedy_step(m)
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            row.update(
                ms_per_step=dt / args.steps * 1e3,
                tokens_per_s=batch * args.steps / dt,
                kernel_calls_per_step=(kvcache.STATS["paged_decode_calls"] - calls0)
                / (args.warmup + args.steps),
                kv_bytes=m.kv_bytes(), peak_reserved=torch.cuda.max_memory_reserved(),
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            row["failed_in"] = phase
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" in row:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
