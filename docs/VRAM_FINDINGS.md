# Step 2 findings: does burning weights free VRAM the KV cache can use?

**Short answer: yes, but only with the split-at-load weight layout.** Shrinking
the FFN frees exactly the expected bytes in every layout tested, but only the
split layout lets the KV cache claim all of them. This is the layout
`HFElasticModel` must use.

Measured 2026-10-20 with `scripts/verify_vram_free.py`.

## Caveat: dense stand-in, not a nested checkpoint

These numbers come from **Qwen3-8B, a dense model**, not a MatFormer-style
nested checkpoint. None was available locally, and Gemma 3n E4B (~16 GB) does
not fit on the remaining disk. Each FFN was cut to its first half of neurons
(width 12288 → 6144, all 36 layers), which is the same prefix slicing MatFormer
nesting uses.

The memory arithmetic does not depend on whether the weights were *trained*
nested: a tensor of a given shape occupies the same bytes either way, and the
allocator behaves the same. What the stand-in cannot tell us is output quality
after the cut, which is Experiment 1's question, not this one's. Repeat the run
on the Experiment 1 checkpoint once it is chosen.

## Setup

| | |
|---|---|
| GPU | RTX 4090, 23.6 GiB (≈21.4 GiB free; display uses ≈1.8 GiB) |
| Software | torch 2.12.0+cu126, transformers 4.57.6 |
| Model | Qwen3-8B, bf16, 15.258 GiB resident |
| Burn | FFN width 12288 → 6144 in all 36 layers |
| Expected weight drop | 5.0625 GiB (computed from tensor shapes) |

## Method: the tight-memory test

Checking `memory_allocated()` alone is not enough: bytes can leave the
allocated count yet stay unusable for a new allocation. So the script makes
the memory limit tight and measures what the KV cache can actually get.

1. Load the large configuration; record `memory_allocated()`.
2. Fill the GPU with one dummy "ballast" tensor, leaving only a small margin
   free (≈190 MiB, twice the largest FFN tensor). From here on, any extra KV
   room can only come from the burned weights.
3. Measure KV capacity *before* the burn: allocate KV-shaped tensors until
   out-of-memory, record the total, free them.
4. Burn: shrink every FFN, optionally call `torch.cuda.empty_cache()`. Assert
   the drop in `memory_allocated()` equals the expected weight difference.
5. Measure KV capacity again. **Pass condition: the gain equals the freed
   weight bytes** (within page size / search resolution, max 1%).
6. Run a forward pass on the shrunk model to confirm it still works.

KV capacity is measured two ways: as fixed-size pages (one layer's K+V for 16
tokens = 64 KiB, vLLM's default block size; also 512 tokens = 2 MiB), and as
the largest single contiguous tensor (binary search, 16 MiB resolution).

## Results

Weight drop was exactly 5.0625 GiB (0.00 MiB error) in every run except the
view layout, which freed nothing.

| Layout | KV layout | `empty_cache` | KV gain / freed | Extra memory during burn |
|---|---|---|---|---|
| **View** (slice, no copy) | n/a | n/a | **0%** (0.0 of 144 MiB, one layer) | 0 |
| **Clone at switch** | 64 KiB pages | yes | 97.9% | 48 MiB |
| | 64 KiB pages | no | 97.9% | 48 MiB |
| | 2 MiB pages | yes | 98.9% | 48 MiB |
| | contiguous | yes | 97.8% | 48 MiB |
| | 64 KiB pages, `expandable_segments` | yes | 91.0% | 48 MiB |
| | contiguous, `expandable_segments` | yes | 91.2% | 48 MiB |
| **Split at load** | 64 KiB pages | yes | **100.0%** | 0 |
| | 64 KiB pages | no | **100.0%** | 0 |
| | 2 MiB pages | yes | 99.9% | 0 |
| | contiguous | yes | 100.1% | 0 |

Raw results: `results/vram_free_*.json`, one file per Clone and Split row
above; the View row is the `view_trap_freed` field of the Clone runs. A forward pass on the shrunk model succeeded in every run.

## Why the split layout is required

**View.** `W[:k]` is a view: it shares the original tensor's storage, which
stays allocated in full. The model gets smaller on paper and frees nothing. A
`switch_to` written this way would make GEI's range gain silently zero.

**Clone at switch.** Copying the kept slice, then dropping the original, frees
the right byte count but strands 2–9% of it. Each new 48 MiB copy is placed
inside a block freed by an earlier layer's 96 MiB original. That leaves memory
segments that are partly in use, which the caching allocator can neither return
to the driver nor hand to a KV allocation of a different size. About 110–190
MiB never became usable cache, failing the pass condition. `expandable_segments`
made this worse, not better. The copy also needs spare memory during the burn
(one tensor's worth), exactly when memory is shortest, and costs a
device-to-device copy of the whole kept model mid-generation.

**Split at load.** Each FFN is stored from load time as two separate tensors,
the kept prefix and the droppable suffix. Burning is a plain `del` of the
suffix: whole allocator segments become free and nothing new is allocated, so
there is nothing to fragment. 100% of the freed bytes became KV capacity, with
zero extra memory during the burn and no copy.

The cost is in the large configuration: while both halves are resident, each
FFN projection runs as two matmuls (gate/up: two row blocks concatenated;
down: two column blocks summed) instead of one.

## Smaller observations

- `empty_cache()` made no difference to reuse. On an out-of-memory error,
  PyTorch's allocator frees its cached blocks and retries on its own. Calling it
  is still useful so that `torch.cuda.mem_get_info()` reports the true state.
- vLLM pre-allocates its KV pool once at startup and does not grow it, so
  freed weight memory would not reach a stock vLLM cache. A GEI backend needs
  a KV allocator that can grow at runtime (HF's dynamic cache does).
- transformers 4.57 contacts the Hub for metadata even with
  `local_files_only=True`; the script sets `HF_HUB_OFFLINE=1` to guarantee no
  network access.

## Under batching: two more losses (HFElasticModel, 2026-10-20)

Measured with `HFElasticModel` itself, batch of 64, ballast filling the GPU
after prefill, then decoding until out-of-memory (tests in
`tests/test_hf_elastic.py`; tiny random Llama, fp32, RTX 4090, torch 2.12).

**1. Small chunks need one allocation per burn level.** With every droppable
chunk a separate tensor, `memory_allocated()` dropped by exactly the right
amount, yet the cache could not use the bytes: at 36 layers with 2 MiB chunks,
burning gave 4.4% of the freed bytes as cache, the same as not burning at all.
PyTorch packs allocations of 1–10 MiB into shared 20 MiB segments, so freed
chunks left holes next to kept weights that never return to the driver. Step 2
passed only because Qwen3-8B's chunks are 48 MiB, large enough to get their own
segments. `HFElasticModel` now gives each burn level one flat buffer, with the
chunk weights as views of it (safe, because the whole buffer is dropped at
once). After the burn, KV-sized pages can claim **100%** of the freed bytes at
both 4 and 36 layers. Real checkpoints mostly have large chunks (Gemma 3n E4B:
32 MiB), but the backend no longer depends on that.

**2. HF's growing cache realises only part of the freed bytes.**
`DynamicCache` grows each layer by `torch.cat`, so the old and new tensors are
live at once and old blocks are not reused as the cache grows. SDPA adds
per-step transients on top (`repeat_kv` copies when a mask is passed).

| Layers (growing KV) | Freed | Cache grown after burn | Without burn |
|---|---|---|---|
| 4 | 192 MiB | 86 MiB (44.8%) | 0 |
| 16 | 192 MiB | 121 MiB (63.2%) | n/a |
| 36 | 216 MiB | 163 MiB (75.6%) | 2.3 MiB (1.1%) |

The loss shrinks with more KV-writing layers, because each layer's share of the
transient is smaller. On Qwen3-8B (36 layers, batch 8) the per-step transient was
1.5% of the cache, but the allocator held 1.9 GiB reserved-but-unused for a
1.2 GiB cache at T=1024. **Gemma 3n is the bad case:** only 4 layers grow, the
shape that realised 45%.

Consequence: the range gain must be reported as measured with the cache
actually used. A cache pre-sized to the maximum context is no fix: it would
claim the memory before the burn frees it, which defeats the experiment. The
fix is a cache that allocates incrementally without recopying, below.

## Block-allocated cache (gei.kvcache, 2026-10-20)

`HFElasticModel(kv_cache="block", block_tokens=512)` keeps every
full-attention layer's cache as fixed-size blocks of `block_tokens` tokens.
Appending writes into the current block; a new block is allocated only when it
is full; existing blocks are never copied, resized or moved (a test checks
their storage pointers survive later steps). Nothing is pre-sized. Because K
and V are never one contiguous tensor, attention runs block by block with an
online softmax (registered with transformers as "gei_blocked"), handling
grouped-query attention natively. Sliding-window layers keep HF's own bounded
layer, since they stop growing at the window.

It is exact: batched, mixed prompt lengths, switched mid-generation with REUSE
or RECOMPUTE, on Llama and on Gemma 3n (sliding and KV-shared layers), it gives
the same tokens as each sequence decoded alone, with logprob differences around
5e-7 (4-token blocks, so every sequence spans many blocks).

**Measured.** Same harness as above: batch 64, prefill, fill the GPU with
ballast, burn (REUSE), decode until out-of-memory. Net growth = growth with the
burn minus growth without it (2–10 MiB of allocator slack), as a fraction of
the freed weight bytes. 16-token blocks; identical over three runs.

| Shape | Freed | HF DynamicCache | Block cache |
|---|---|---|---|
| 36 KV layers | 216 MiB | 72.2% | **100.0%** |
| 4 KV layers (Gemma-3n-like) | 192 MiB | 49.0% | **94.3%** |

The rest of the 4-layer gap is per-step transients (FFN activations, attention
scratch) and the last block row that does not fit, a few MiB that loom large
against a 192 MiB toy budget; real models free GiB. Block size matters only
through granularity: a new block row allocates one block per layer at once, so
the row must be small next to the freed budget. Measured on these shapes,
8/16/64-token blocks gave 91–94% (4 layers) and 102–106% gross (36 layers).
At batch 16 on Gemma 3n E4B, a 512-token row is 64 MiB against 2.34 GiB freed.

Cost: the reference attention is a Python loop over blocks, slower than fused
SDPA. It is correct and memory-faithful. Decode now runs a fused Triton kernel
over the block table (`gei.paged_decode`); the loop remains the reference and
the fallback (prefill, padding masks, softcapping). Throughput of both is below,
"Decode throughput by attention path".

## Does `expandable_segments` remove Requirement 1? (2026-11-01)

**No.** With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the unpacked
layout still converts **0%** of the freed bytes in every clean failure cell
(2 MiB chunks with 64 KiB pages, at 12 and 36 layers; 0.5 MiB chunks with
2 MiB pages, at 48 layers), and only 27.8% for 8 MiB chunks with 64 KiB pages.
The flag also makes the packed layout worse with small KV pages. Packing each
burn level into one buffer stays necessary. The flag does not replace it.

**Why this was asked.** The Step 2 table reports `expandable_segments` only for
clone at switch. If the flag made the *unpacked* split layout reclaim 100%,
Requirement 1 would be a property of the default caching allocator, and "set
this flag" would be a cheaper fix than "pack your weights".

**Method.** `scripts/verify_chunk_reclaim.py` runs the same tight-memory test as
`test_freed_chunks_are_claimable_at_every_size`, reporting the conversion
instead of pass/fail: a tiny Llama (hidden 256, 12 layers, fp32) whose FFN
chunks are 0.5, 2, 8 or 24 MiB; ballast fills the GPU; KV-sized pages are
claimed until out-of-memory before and after the burn. Conversion = (pages
after − pages before) / freed weight bytes. Freed budgets are 18, 72, 288 and
864 MiB. Plus two larger shapes: 36 layers × 2 MiB (216 MiB, where the 4.4% was
first seen) and 48 layers × 0.5 MiB (72 MiB, because an 18 MiB budget is below
one 20 MiB allocator segment and cannot resolve the 2 MiB-page cells). Three
weight layouts:

- **packed**: one flat buffer per burn level (`HFElasticModel` as shipped);
- **unpacked**: every chunk its own tensor, moved to the GPU in module order
  with the kept weights. This is the layout before per-level packing, the one
  the "2 and 8 MiB fail" result came from;
- **unpacked, grouped**: every chunk its own tensor, but allocated after all
  kept weights.

Two KV page sizes, because the allocator serves them from different pools:
64 KiB (as in the existing test; requests of ≤1 MiB come from the *small* pool,
in 2 MiB segments) and 2 MiB (the *large* pool; requests of 1–10 MiB are served
from 20 MiB segments). The block cache uses both in practice: a 512-token block
is 1 MiB per K or V tensor on Qwen3-8B at batch 1 (small pool) and 8 MiB on
Gemma 3n at batch 16 (large pool). One process per cell (the allocator setting
is read at CUDA init), three repeats. Every cell freed exactly the expected
bytes from `memory_allocated()`, and the shrunk model decoded afterwards. Raw
rows: `results/chunk_reclaim.jsonl`.

**Results**, % of freed bytes that became KV pages; three repeats, identical
unless listed:

| KV page | Layout | Allocator | 0.5 MiB | 2 MiB | 8 MiB | 24 MiB | 36 L × 2 MiB | 48 L × 0.5 MiB |
|---|---|---|---|---|---|---|---|---|
| 64 KiB | packed | default | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 64 KiB | packed | expandable | **0.0** | 83.3 | 90.3 | 97.2 | 92.6 | 83.3 |
| 64 KiB | unpacked | default | 100.0 | **0.0** | **0.0** (19.4 once) | 100.0 | **0.0** | 100.0 |
| 64 KiB | unpacked | expandable | 100.0 | **0.0** | 27.8 | 78.7 | **0.0** | 100.0 |
| 64 KiB | unpacked, grouped | default | 100.0 | 83.3 | **0.0** | 100.0 | 92.6 | 100.0 |
| 64 KiB | unpacked, grouped | expandable | 100.0 | 83.3 | 97.2 | 97.2 | 92.6 | 100.0 |
| 2 MiB | packed | default | ‡ | 111.1 † | 97.2–104.2 † | 99.5 | 101.9 † | 111.1 † |
| 2 MiB | packed | expandable | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2 MiB | unpacked | default | ‡ | 100.0 | 100.0 | 99.5 | 100.0 | **0.0** |
| 2 MiB | unpacked | expandable | ‡ | 100.0 | 100.0 | 100.0 | 100.0 | **0.0** |
| 2 MiB | unpacked, grouped | default | ‡ | 100.0 | 100.0 | 99.5 | 100.0 | 111.1 † |
| 2 MiB | unpacked, grouped | expandable | ‡ | 100.0 | 100.0 | 100.0 | 100.0 | 111.1 † |

† Above 100%: claims of 2 MiB pages move in 20 MiB steps (one 20 MiB segment
under the default allocator, one 20 MiB large-pool mapping under expandable
segments), so a budget that is not a multiple of 20 MiB can complete one more
step out of slack the ballast left: 80 MiB claimed against 72 freed is 111.1%.

‡ Not resolvable at 12 layers: the 18 MiB budget is below one 20 MiB step, so
these cells read 0% or 111.1% regardless of layout (packed, default: 111.1, 0.0,
0.0 over the three repeats). The 48-layer column answers the question instead.

**What the data supports.**

1. *The flag does not make unpacked reclaim 100%.* The clean failures of the
   unpacked layout (2 MiB chunks with 64 KiB pages at 12 and 36 layers; 0.5 MiB
   chunks with 2 MiB pages at 48 layers) are 0% with or without the flag, and
   the 8 MiB cell rises only to 27.8%. Requirement 1 is not an artefact of the
   default allocator's segment policy.
2. *Whether unpacked fails depends on the chunk size and the cache's allocation
   size together, not on chunk size alone.* With 64 KiB pages, 2 and 8 MiB
   chunks fail and 0.5 MiB chunks work; with 2 MiB pages it is the reverse. The
   pattern fits this account: a freed chunk inside a segment that some live
   tensor still occupies cannot return to the driver, so its space can serve
   only requests from the same pool. In the unpacked layout that live tensor is
   a kept weight. 0.5 MiB chunks sit in small-pool segments, usable by 64 KiB
   pages but not by 2 MiB ones; 2 and 8 MiB chunks sit in 20 MiB large-pool
   segments, usable by 2 MiB pages but not by 64 KiB ones; 24 MiB chunks get
   their own segments and work either way. The grouped layout, which keeps the
   chunks away from kept weights, recovers most cases (111.1% for 0.5 MiB with
   2 MiB pages), but not all: 8 MiB grouped chunks still gave 0% with 64 KiB
   pages under the default allocator, which kept-weight sharing alone does not
   predict. Any live tensor in the segment pins it; here the likely pins are
   ballast pieces placed in the segments' free remainders. Either way the
   outcome is hard to predict, which is the argument for packing.
3. *Packing works across pools under the default allocator* (100% at 64 KiB;
   within one 20 MiB step at 2 MiB). *Under `expandable_segments` it does not
   always:* with 64 KiB pages, packed fell to 0% for an 18 MiB buffer and 83–97%
   elsewhere; with 2 MiB pages it was exactly 100% everywhere. Every claim made
   from freed memory under the flag with 64 KiB pages is a whole multiple of
   20 MiB (60 of 72, 200 of 216, 260 of 288, 840 of 864 MiB, and 0 of 18). This
   is consistent with expandable segments mapping large-pool memory in 20 MiB
   units (2 MiB for the small pool): freed large-pool memory reaches the small
   pool only as whole, aligned 20 MiB units, so up to ~28 MiB of each budget is
   lost to the edges, regardless of the budget's size.

**Consequence.** Requirement 1 stands, and it is a requirement on the weight
layout, not on an allocator setting. A backend should pack each burn level. Under
the default allocator, packing alone was enough at both page sizes. With
`expandable_segments` on, packed weights still reclaimed exactly 100% when the KV
allocations came from the large pool (> 1 MiB), but lost up to ~28 MiB per burn
when they came from the small pool. Every range-gain run used the default
allocator.

## Decode throughput by attention path (2026-11-01)

**Question.** What does the block cache cost in speed, and which attention path
did the range-gain runs use?

**Method.** `scripts/bench_decode_attention.py`, one process per model and path.
The model runs through `HFElasticModel._forward`, the call the range-gain scripts
use, at the small configuration (the one that decodes the long contexts in those
runs), bf16, with 512-token blocks. The context is grown by chunked prefill to
each length, then 4 warm-up and 32 timed greedy decode steps (8 for the Python
loop). The time per step is the whole model's forward pass plus argmax, not
attention alone. The per-step logprob copy to the host that `step_batch` makes
is left out, since it is the same for every path. Token ids are random:
attention cost does not depend on them. Contexts 1,024 to 32,000 tokens
(Gemma 3n's limit is 32,768); one run per cell on an RTX 4090 that also drives
the display (≈3 GiB in use), so differences of a few percent are not meaningful.
Raw rows: `results/bench_decode_attention.jsonl`.

Paths:

- **kernel**: block cache, fused Triton paged-decode kernel;
- **loop**: block cache, the Python block loop (kernel disabled);
- **SDPA**: HF `DynamicCache` with SDPA, PyTorch's default backend choice;
- **flash**: HF `DynamicCache` with SDPA restricted to its FlashAttention
  backend. The `flash_attn` package is not installed here, so HF's
  `flash_attention_2` path could not run; this is PyTorch's built-in
  FlashAttention-2 kernel instead.

**Results**, decode tokens/s (all sequences in the batch; higher is better):

| Context | Qwen3-8B b=1: kernel | loop | SDPA | flash | Gemma 3n b=16: kernel | loop | SDPA | flash |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 55.7 | 40.8 | 65.8 | 64.1 | 528 | 491 | 520 | n/a |
| 2,048 | 55.3 | 31.1 | 63.7 | 62.0 | 519 | 464 | 512 | n/a |
| 4,096 | 55.5 | 22.0 | 60.7 | 60.9 | 499 | 423 | 486 | n/a |
| 8,192 | 54.2 | 13.5 | 52.1 | 54.6 | 461 | 355 | 437 | n/a |
| 16,384 | 49.4 | 7.8 | 44.6 | 45.1 | 393 | 269 | 354 | n/a |
| 24,576 | | | | | 337 | 206 | 300 § | n/a |
| 32,000 | 42.5 | 4.4 | 31.5 | 31.9 | 302 | 184 | 261 § | n/a |

The 24,576 row (Gemma only) was run separately, growing straight to that length,
to give a matched point between 16k and 32k.

§ Context grown by decoding, one token at a time, as in the range-gain runs.
Grown by chunked prefill, SDPA ran out of memory *during the prefill* at 24k and
32k: with an attention mask (always passed for a multi-token chunk after the
first), HF's SDPA path expands each full-attention layer's K and V to all 8
query heads (`repeat_kv`), about 1.5 GiB per layer at 24k and batch 16. That
failure belongs to the benchmark's prefill, not to decoding, and is not
reported as a result.

- **Qwen3-8B, batch 1.** SDPA is faster at short context (kernel 0.85× SDPA at
  1k, 0.91× at 4k); the two cross near 8k, and the kernel is 1.11× at 16k and
  1.35× at 32k. `DynamicCache` copies the whole cache every step (`torch.cat`),
  so its step time grows with context faster than the kernel's. This matches
  the earlier full-run wall-clock in `docs/RANGE_GAIN.md` (SDPA faster at short
  context, the block cache faster over a long generation). Forcing the flash
  backend performed the same as default SDPA (which backend the default
  picked was not logged). The Python
  loop is 1.4× slower than the kernel at 1k and 9.6× slower at 32k (63 blocks ×
  36 layers of small kernel launches per step).
- **Gemma 3n E4B, batch 16.** The kernel matches or beats SDPA at every length
  where SDPA runs (1.01× at 1k, 1.11× at 16k). Only 7 of 35 layers have
  full-attention KV, so attention is a small part of the step and the paths are
  closer than on Qwen; the kernel's lead grows with context (1.13× at 24k,
  1.16× at 32k). The loop is 1.08× slower at 1k and 1.64× at 32k. The copying
  cache needs more memory for the same context: at 32k, peak reserved was
  21.7 GiB with SDPA against 18.6 GiB with the block cache, for about the same
  cache (4.47 vs 4.50 GB), on a card where about 21 GiB is usable next to the
  display. That is Requirement 2's transient, seen from the speed benchmark.
  **Forced flash cannot run on Gemma 3n:** its 28 sliding-window layers
  pass an explicit attention mask to SDPA at every decode step, and SDPA's
  flash backend takes no mask ("No available kernel"). Logged per call: the 7
  full-attention calls are mask-free with grouped-query attention, the 28
  sliding calls carry a mask. Default SDPA already uses flash wherever a call
  allows it.

**Which path the range-gain runs used: the Triton kernel**, for every
full-attention layer at every decode step. Checked directly: running the range-gain
scripts themselves (condition C, 40 steps) on the current code, which differs
from the code of those runs only in license headers (and, since the Qwen runs,
in host-memory handling at load and `logits_to_keep=1`, both outside
attention), counted 1,440 kernel calls on Qwen3-8B (36 per
step, all layers) and 280 on Gemma 3n (7 per step, all full-attention layers;
its 28 sliding-window layers use HF's bounded cache and SDPA, as in every
condition). `docs/RANGE_GAIN.md` records the kernel for the Qwen runs. The
`-dynamic` ablations used `DynamicCache` with default SDPA. Prefill always runs
the Python loop (the kernel is decode-only), as it did in those runs.

## What framework support in-generation burning requires

Two conditions, each measured here; without either, `memory_allocated()`
reports the burn correctly and the range gain still does not appear.

1. **Flat per-burn-level weight buffers.** Each burn level must be one
   allocation that no kept weight shares. With each FFN chunk as its own
   tensor, chunks of roughly 1–10 MiB land in the allocator's shared 20 MiB
   segments beside kept weights. At 36 layers with 2 MiB chunks the drop in
   `memory_allocated()` was exact, yet burning gave **4.4%** of the freed bytes
   as cache, the same as not burning: in effect **0%**. Packed per level: 100%
   claimable at every chunk size tested (0.5–24 MiB) under the default
   allocator (64 KiB pages; within one 20 MiB step with 2 MiB pages).
   Unpacked, with 64 KiB KV pages, 0.5 MiB and 24 MiB chunks happened to work
   and 2 MiB and 8 MiB did not. With 2 MiB pages the outcome flips (0.5 MiB
   fails at 48 layers; 2 and 8 MiB work), so whether an unpacked implementation
   silently fails depends on the chunk size *and* the cache's allocation size
   together. `expandable_segments` is no substitute for packing (above, "Does
   `expandable_segments` remove Requirement 1?").
   A regression test guards this on any shape (`tests/test_hf_elastic.py`,
   `test_each_burn_level_is_one_allocation_shared_with_nothing_kept`).
2. **Non-copying incremental cache allocation.** The cache must grow into the
   freed memory a block at a time, without recopying what it already holds and
   without pre-sizing. A copying cache must hold each layer's new K and V
   alongside the old ones every step, so it loses roughly one layer's share of
   the cache, plus fragmentation. How much that costs depends on the shape:
   - toy shapes at batch 64 (above): HF's cache converted **49.0%** (4 KV
     layers) and **72.2%** (36 KV layers); the block cache **94.3%** and
     **100.0%**;
   - Qwen3-8B at batch 1, 36 KV layers, 5.06 GiB freed (`docs/RANGE_GAIN.md`):
     HF's cache converted **97.2%** (70 of 72 freed rows), the block cache
     **100%**. Worth 1,024 tokens of horizon, 2.8% of the gain.

   So this condition matters most where one layer is a large share of the
   cache: few KV-writing layers (Gemma 3n grows only 4) or a small burn budget.
   On a 36-layer model at batch 1 it is a small correction. It is still
   needed for an exact range gain, and for the 4-KV-layer shape it is the
   difference between half and nearly all of the freed memory.

Star Elastic's model card says in-generation switching "currently requires a
custom inference path" and that "efficient native vLLM integration is actively
being worked on". The card is about switching in general, and its switch goes
up, so it never frees weights. For the case GEI needs, burning down and handing
the freed memory to the cache, these two conditions are, in our reading, the
concrete content of that framework support. Stock vLLM fails both: weights
are loaded as fixed tensors, and its KV pool is sized once at startup.
