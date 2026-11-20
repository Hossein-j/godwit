#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Does shrinking the FFN actually free VRAM, and
can the KV cache claim the freed bytes?

What it does, on one model:

  1. Load the large configuration and record torch.cuda.memory_allocated().
  2. View trap: slice one layer's FFN *without* copying. A view keeps the parent
     storage alive, so this must free nothing. It is the easy bug to ship in
     HFElasticModel.switch_to, so it is shown explicitly, then undone.
  3. Fill the GPU with ballast so only a small margin is free. The envelope is
     now tight: any KV headroom after the shrink can only come from the weights.
  4. Measure KV capacity before the shrink (pages, or one contiguous tensor,
     allocated until OOM, then freed).
  5. Shrink every FFN to its first `small_width` neurons (MatFormer-style prefix
     nesting) and optionally empty_cache. Assert the drop in memory_allocated()
     equals the expected weight difference. Two strategies:
       split (default)  prefix and suffix are separate tensors from load time;
                        the burn is a pure `del`, whole segments are released.
       clone            copy the kept slice at burn time, then drop the original.
                        Frees the right byte count, but the copies land inside
                        freed segments and strand ~2% of them (measured on
                        Qwen3-8B), so the KV cache cannot claim it all.
  6. Measure KV capacity again. The gain must match the freed bytes.
  7. Run a forward pass on the shrunk model to confirm it is still coherent.

Memory behaviour does not depend on whether the weights were *trained* nested,
so a dense checkpoint is a valid stand-in for this test (its outputs after the
shrink will be poor; that is irrelevant here). The model is loaded with
local_files_only=True: this script never downloads anything.

Usage:
    python scripts/verify_vram_free.py --model Qwen/Qwen3-8B
    python scripts/verify_vram_free.py --model Qwen/Qwen3-8B --kv-layout contiguous
    python scripts/verify_vram_free.py --model Qwen/Qwen3-8B --strategy clone
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

# Hard guarantee of no network access; local_files_only alone still lets
# transformers query the Hub for metadata.
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

MiB = 1024**2
GiB = 1024**3


def alloc() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def gib(n: int) -> str:
    return f"{n / GiB:.3f} GiB"


def find_mlps(model) -> list:
    mlps = [
        m
        for m in model.modules()
        if all(isinstance(getattr(m, p, None), torch.nn.Linear)
               for p in ("gate_proj", "up_proj", "down_proj"))
    ]
    if not mlps:
        raise SystemExit("no gate/up/down MLPs found; this model is not supported")
    return mlps


def bytes_removed(mlp, k: int) -> int:
    """Bytes freed by keeping the first k intermediate neurons of one MLP."""
    total = 0
    for name in ("gate_proj", "up_proj"):
        lin = getattr(mlp, name)
        per_row = lin.weight.shape[1] * lin.weight.element_size()
        total += (lin.weight.shape[0] - k) * per_row
        if lin.bias is not None:
            total += (lin.bias.shape[0] - k) * lin.bias.element_size()
    w = mlp.down_proj.weight
    total += w.shape[0] * (w.shape[1] - k) * w.element_size()
    return total


def shrink_mlp(mlp, k: int, *, copy: bool = True) -> None:
    """Keep the first k intermediate neurons. copy=False leaves views (the trap)."""

    def keep(t: torch.Tensor) -> torch.Tensor:
        return t.clone() if copy else t

    for name in ("gate_proj", "up_proj"):
        lin = getattr(mlp, name)
        lin.weight = torch.nn.Parameter(keep(lin.weight.data[:k]), requires_grad=False)
        if lin.bias is not None:
            lin.bias = torch.nn.Parameter(keep(lin.bias.data[:k]), requires_grad=False)
        lin.out_features = k
    lin = mlp.down_proj
    # Column slice is non-contiguous, so it must be materialised either way;
    # contiguous() on the view would also copy, so the trap only applies to rows.
    lin.weight = torch.nn.Parameter(
        lin.weight.data[:, :k].contiguous() if copy else lin.weight.data[:, :k],
        requires_grad=False,
    )
    lin.in_features = k


def split_mlp(mlp, k: int) -> None:
    """Store the FFN as two allocations: kept prefix (live) + burnable suffix.

    Done once at load time, so a later burn is a pure free with no copy. The
    suffix is held as non-persistent buffers so .to("cuda") moves it. A real
    backend needs a split forward (two GEMMs) while the suffix is resident; this
    script only measures memory, so the large configuration is never run here.
    """
    for name in ("gate_proj", "up_proj"):
        lin = getattr(mlp, name)
        if lin.bias is not None:
            raise SystemExit("--strategy split does not handle biased MLPs")
        w = lin.weight.data
        mlp.register_buffer(f"_burn_{name}", w[k:].clone(), persistent=False)
        lin.weight = torch.nn.Parameter(w[:k].clone(), requires_grad=False)
        lin.out_features = k
    lin = mlp.down_proj
    w = lin.weight.data
    mlp.register_buffer("_burn_down_proj", w[:, k:].contiguous(), persistent=False)
    lin.weight = torch.nn.Parameter(w[:, :k].contiguous(), requires_grad=False)
    lin.in_features = k


def burn_split(mlp) -> None:
    for name in ("gate_proj", "up_proj", "down_proj"):
        delattr(mlp, f"_burn_{name}")


def view_trap(m0, k: int, width: int) -> int:
    """Slice one layer without copying, measure what was freed, then undo it."""
    saved = (m0.gate_proj.weight, m0.up_proj.weight, m0.down_proj.weight)
    before = alloc()
    shrink_mlp(m0, k, copy=False)
    view_freed = before - alloc()
    m0.gate_proj.weight, m0.up_proj.weight, m0.down_proj.weight = saved
    m0.gate_proj.out_features = m0.up_proj.out_features = m0.down_proj.in_features = width
    del saved
    one_layer = bytes_removed(m0, k)
    print(f"[2] view trap (layer 0, slice without copy): freed {view_freed / MiB:.1f} MiB "
          f"of {one_layer / MiB:.1f} MiB expected  "
          f"-> {'confirmed: views free nothing' if view_freed < MiB else 'UNEXPECTED'}")
    return view_freed


def kv_page_bytes(cfg, page_tokens: int, dtype_size: int) -> int:
    """One layer's K+V block for page_tokens tokens: 2 * tokens * kv_heads * head_dim."""
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return 2 * page_tokens * kv_heads * head_dim * dtype_size


def claim_paged(page_bytes: int, dtype) -> int:
    """Allocate KV pages until OOM, free them, return bytes claimed."""
    pages = []
    n = page_bytes // torch.empty((), dtype=dtype).element_size()
    try:
        while True:
            pages.append(torch.empty(n, dtype=dtype, device="cuda"))
    except torch.OutOfMemoryError:
        pass
    claimed = len(pages) * page_bytes
    del pages
    return claimed


def claim_contiguous(dtype, hi: int, resolution: int = 16 * MiB) -> int:
    """Largest single KV tensor that fits, by binary search to `resolution`."""
    elsize = torch.empty((), dtype=dtype).element_size()

    def fits(nbytes: int) -> bool:
        try:
            t = torch.empty(nbytes // elsize, dtype=dtype, device="cuda")
        except torch.OutOfMemoryError:
            return False
        del t
        return True

    lo = 0
    while hi - lo > resolution:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF id or local path (no downloads)")
    ap.add_argument("--small-ratio", type=float, default=0.5,
                    help="small FFN width as a fraction of the large one")
    ap.add_argument("--strategy", choices=["split", "clone"], default="split",
                    help="clone: copy the kept slice at burn time; split: store "
                         "prefix and suffix as separate tensors at load, burn = del")
    ap.add_argument("--kv-layout", choices=["paged", "contiguous"], default="paged")
    ap.add_argument("--page-tokens", type=int, default=16,
                    help="tokens per KV page (vLLM default block size is 16)")
    ap.add_argument("--no-empty-cache", action="store_true",
                    help="skip torch.cuda.empty_cache() after the shrink")
    ap.add_argument("--margin-mib", type=int, default=0,
                    help="free VRAM left beside the ballast (default: 2x the "
                         "largest FFN tensor, the transient a clone needs)")
    ap.add_argument("--out", default=None, help="JSON results path")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gei.revisions import pinned

    dtype = torch.bfloat16
    print(f"Loading {args.model} (bf16, local files only)")
    tok = AutoTokenizer.from_pretrained(args.model, revision=pinned(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=pinned(args.model), dtype=dtype, local_files_only=True
    )
    model.requires_grad_(False).eval()

    cfg = model.config
    mlps = find_mlps(model)
    width = mlps[0].gate_proj.out_features
    k = int(width * args.small_ratio)
    expected = sum(bytes_removed(m, k) for m in mlps)
    largest_tensor = max(m.gate_proj.weight.nbytes for m in mlps)
    print(f"  {len(mlps)} FFN blocks, width {width} -> {k}; "
          f"expected weight drop {gib(expected)}; strategy {args.strategy}")
    if args.strategy == "split":
        for m in mlps:
            split_mlp(m, k)

    model.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()

    a_large = alloc()
    print(f"\n[1] large config resident: memory_allocated = {gib(a_large)}")

    # [2] View trap on one layer: no bytes are freed.
    view_freed = None
    if args.strategy == "split":
        print("[2] view trap: skipped (split layout never slices at burn time)")
    else:
        view_freed = view_trap(mlps[0], k, width)

    # [3] Ballast: leave only `margin` free on the device.
    margin = args.margin_mib * MiB or 2 * largest_tensor
    free, total = torch.cuda.mem_get_info()
    ballast = torch.empty(max(free - margin, 0), dtype=torch.uint8, device="cuda")
    free_after_ballast, _ = torch.cuda.mem_get_info()
    print(f"[3] ballast {gib(ballast.numel())}; device free now "
          f"{free_after_ballast / MiB:.0f} MiB (target margin {margin / MiB:.0f} MiB)")

    page_bytes = kv_page_bytes(cfg, args.page_tokens, torch.empty((), dtype=dtype).element_size())

    def capacity() -> int:
        if args.kv_layout == "paged":
            return claim_paged(page_bytes, dtype)
        return claim_contiguous(dtype, hi=expected + 4 * margin + GiB)

    # [4] KV capacity before shrinking. Release what it cached, so the
    # measurement does not lend memory to the shrink.
    cap_before = capacity()
    torch.cuda.empty_cache()
    layout = (f"paged, {page_bytes / 1024:.0f} KiB pages ({args.page_tokens} tok/layer)"
              if args.kv_layout == "paged" else "one contiguous tensor")
    print(f"[4] KV capacity before shrink ({layout}): {gib(cap_before)}")

    # [5] Shrink for real.
    torch.cuda.reset_peak_memory_stats()
    a0 = alloc()
    for m in mlps:
        if args.strategy == "split":
            burn_split(m)
        else:
            shrink_mlp(m, k, copy=True)
    gc.collect()
    if not args.no_empty_cache:
        torch.cuda.empty_cache()
    a_small = alloc()
    drop = a0 - a_small
    transient = torch.cuda.max_memory_allocated() - a0
    reserved_slack = torch.cuda.memory_reserved() - a_small
    print(f"[5] shrink done ({'with' if not args.no_empty_cache else 'WITHOUT'} empty_cache)")
    print(f"    memory_allocated {gib(a0)} -> {gib(a_small)}: drop {gib(drop)}, "
          f"expected {gib(expected)}, diff {(drop - expected) / MiB:+.2f} MiB")
    print(f"    transient peak above start: {transient / MiB:.1f} MiB; "
          f"reserved-but-unallocated after: {gib(reserved_slack)}")
    drop_ok = abs(drop - expected) <= max(MiB, expected * 1e-3)

    # [6] KV capacity after shrinking.
    cap_after = capacity()
    gain = cap_after - cap_before
    # Page granularity and the contiguous search resolution bound the error.
    tol = max(2 * page_bytes if args.kv_layout == "paged" else 32 * MiB, expected * 0.01)
    claim_ok = gain >= expected - tol
    print(f"[6] KV capacity after shrink: {gib(cap_after)}; gain {gib(gain)} "
          f"= {gain / expected:.1%} of freed weight bytes")

    # [7] The shrunk model still runs.
    del ballast
    torch.cuda.empty_cache()
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        logits = model(ids).logits
    fwd_ok = bool(torch.isfinite(logits).all())
    print(f"[7] forward pass on shrunk model: logits {tuple(logits.shape)}, "
          f"finite={fwd_ok}")

    results = {
        "model": args.model, "strategy": args.strategy, "ffn_blocks": len(mlps), "width_large": width,
        "width_small": k, "kv_layout": args.kv_layout, "page_bytes": page_bytes,
        "empty_cache": not args.no_empty_cache, "device_total": total,
        "alloc_large": a_large, "alloc_small": a_small, "drop": drop,
        "expected_drop": expected, "view_trap_freed": view_freed,
        "shrink_transient_peak": transient, "margin": margin,
        "kv_capacity_before": cap_before, "kv_capacity_after": cap_after,
        "kv_gain": gain, "drop_ok": drop_ok, "claim_ok": claim_ok, "forward_ok": fwd_ok,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")

    print("\nRESULT:",
          f"weight drop {'PASS' if drop_ok else 'FAIL'},",
          f"KV claim {'PASS' if claim_ok else 'FAIL'},",
          f"forward {'PASS' if fwd_ok else 'FAIL'}")
    assert drop_ok, f"drop {drop} != expected {expected}"
    assert claim_ok, f"KV gain {gain} < freed {expected} - tol {tol}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
