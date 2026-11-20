# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Tight-memory reclaim test for split-at-load FFN chunks, packed or unpacked.

The same measurement as tests/test_hf_elastic.py::
test_freed_chunks_are_claimable_at_every_size, reported as a fraction instead
of pass/fail, with the weight layout selectable so the allocator setting can be
compared on both sides of Requirement 1:

  packed       each burn level is one flat buffer (HFElasticModel as shipped)
  interleaved  every chunk its own tensor, moved to the device in module order
               with the kept weights (the layout before per-level packing)
  grouped      every chunk its own tensor, allocated after all kept weights

Ballast fills the GPU; KV-sized pages are claimed until out-of-memory before
and after the burn: 64 KiB (one layer's K+V for 16 tokens at vLLM's default
block size; the allocator's small pool) or 2 MiB (512 tokens; the large pool).
Conversion = (pages after - pages before) / freed weight bytes.

The allocator setting is read once at CUDA initialisation, so run one
configuration per process, e.g.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python scripts/verify_chunk_reclaim.py --chunk-mib 2 --layout interleaved
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_hf_elastic import IDS, _claim_pages, _fill_device, tiny_llama

from gei import HFElasticModel


def build(layout: str, chunk_mib: float, hidden: int, layers: int) -> HFElasticModel:
    half = int(chunk_mib * 1024**2 // (hidden * 4))  # neurons per fp32 chunk
    base = tiny_llama(hidden=hidden, intermediate=2 * half, layers=layers)
    if layout == "packed":
        return HFElasticModel(base, [2 * half, half], device="cuda")
    m = HFElasticModel(base, [2 * half, half])  # packed on the host
    if layout == "interleaved":
        m.model.to("cuda")  # one device tensor per parameter, in module order
    else:  # grouped: kept weights first, then each chunk on its own
        dropped = {id(p) for c in m._level_chunks(1) for p in c.parameters()}
        for p in m.model.parameters():
            if id(p) not in dropped:
                p.data = p.data.to("cuda")
        for b_name, b in list(m.model.named_buffers()):
            mod = m.model.get_submodule(b_name.rpartition(".")[0]) if "." in b_name else m.model
            mod._buffers[b_name.rpartition(".")[2]] = b.to("cuda")
        for c in m._level_chunks(1):
            for p in c.parameters():
                p.data = p.data.to("cuda").clone()
    m.device = torch.device("cuda")
    m._generator = torch.Generator(device="cuda").manual_seed(0)
    return m


def storages_of_level(m: HFElasticModel) -> int:
    return len({p.untyped_storage().data_ptr()
                for c in m._level_chunks(1) for p in c.parameters()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chunk-mib", type=float, required=True)
    ap.add_argument("--layout", choices=["packed", "interleaved", "grouped"], required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--page-kib", type=int, default=64, help="KV page size claimed")
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

    torch.manual_seed(0)
    m = build(args.layout, args.chunk_mib, args.hidden, args.layers)
    m.prefill(IDS)
    freed = m.configs[0].weight_bytes - m.configs[1].weight_bytes
    level_allocs = storages_of_level(m)
    ballast = []
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _fill_device(ballast)
    page = args.page_kib * 1024
    before = _claim_pages(page)
    torch.cuda.empty_cache()
    alloc0 = torch.cuda.memory_allocated()
    m.switch_to(1)
    torch.cuda.synchronize()
    drop = alloc0 - torch.cuda.memory_allocated()
    after = _claim_pages(page)
    m.step()  # the shrunk model still runs
    row = {
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "layout": args.layout, "chunk_mib": args.chunk_mib,
        "layers": args.layers, "hidden": args.hidden, "page_bytes": page,
        "level_allocations": level_allocs,
        "freed_bytes": freed, "allocated_drop_bytes": drop,
        "claim_before_bytes": before, "claim_after_bytes": after,
        "conversion": (after - before) / freed,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
    }
    del ballast, m
    gc.collect()
    print(json.dumps(row))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
