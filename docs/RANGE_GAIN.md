# Range gain on Qwen3-8B (batch 1)

**Result: burning extended the horizon by exactly the predicted 36,864 tokens.**
Under one fixed memory envelope, the large configuration ran out of KV room at
a 2,049-token context. GEI, burning to the small configuration when the
envelope bound, reached 38,913 tokens, the same horizon as running the small
configuration from the start. With HF's copying `DynamicCache` instead of the
block cache, the same burn reached 37,889: the block cache is worth 1,024
tokens of horizon at this shape.

Measured 2026-10-20 with `scripts/range_gain.py`, on the shipping code (the
Triton decode kernel with unpadded partials).

## What this does and does not measure

**Qwen3-8B's nesting is untrained.** The small configuration keeps the first
6,144 of each FFN's 12,288 neurons of a dense checkpoint that was never trained
to be sliced. Its output after the burn is not a meaningful model of anything.
This experiment therefore measures **horizon and memory only, never quality**:
how many tokens fit before the envelope fails, how much memory is used, and how
long it takes. No accuracy, perplexity or quality claim follows from it. The
quality cost of burning needs a checkpoint with trained nesting (Experiment 2,
`docs/MODEL_SELECTION.md`).

Generation is greedy, ignores end-of-sequence and runs until out-of-memory.
The tokens are filler for the cache; their content is irrelevant here.

**Determinism and n = 1.** Token counts here are deterministic: decoding is
greedy, and where the run fails is fixed by allocation arithmetic (the bytes
the envelope holds and the bytes each step requests), not by sampling. Re-running
B and C on the shipping kernel reproduced the earlier horizons to the token
(below). One run per condition is therefore sufficient for the horizon claim.
Wall-clock is not deterministic: differences of a few seconds between runs are
not meaningful.

## Setup

| | |
|---|---|
| GPU | RTX 4090, 23.62 GiB (25,358,630,912 B) |
| Software | torch 2.12.0+cu126, transformers 4.57.6, Triton 3.7.0 |
| Model | Qwen3-8B (cached locally), bf16, batch 1 |
| Configurations | FFN width 12,288 (large) and 6,144 (small), all 36 layers, split at load, one buffer per burn level |
| KV cache | `gei.kvcache` block cache, 512-token blocks, allocated as the context grows (never pre-sized), fused Triton decode attention. Ablation: HF `DynamicCache` with SDPA |
| Prompt | "Write a very long, detailed story about a bird crossing an ocean." (14 tokens) |

**Envelope.** M = **15.75 GiB = 16,911,433,728 B**, enforced with
`torch.cuda.set_per_process_memory_fraction`. This caps everything PyTorch's
caching allocator reserves: weights, cache, activations, transients and
fragmentation. The CUDA context and cuBLAS handles sit outside it. M was chosen
so the large configuration fails early in a long generation and the small one
fails before Qwen3's 40,960-position limit.

**Conditions**, one process each, so each starts from a clean allocator:

- **A**: large configuration throughout, never burns (the fixed-model baseline).
- **B**: GEI. Start large and burn to small when the envelope binds (cache kept,
  REUSE). The trigger is a probe, described below.
- **C**: small configuration from the start (the large weights are loaded, then
  burned before the prompt).
- **B-dynamic**: B with HF's `DynamicCache` instead of the block cache, same
  trigger rule. The ablation for the cache. **A-dynamic** (A with
  `DynamicCache`) is its baseline.

## Results

| | A: large, never burns | B: GEI | C: small from start | B-dynamic: GEI, HF cache | A-dynamic |
|---|---|---|---|---|---|
| Tokens generated before OOM | 2,035 | **38,899** | 38,899 | 37,875 | 2,035 |
| Context at OOM (prompt + generated) | 2,049 | **38,913** | 38,913 | 37,889 | 2,049 |
| Horizon gain over its baseline | | **+36,864** | | +35,840 | |
| Burn at context | n/a | 1,537 | before the prompt | 2,050 | n/a |
| Bytes freed by the burn | n/a | 5,435,817,984 (exact) | 5,435,817,984 | 5,435,817,984 | n/a |
| Switch time | n/a | 80 ms | n/a | 71 ms | n/a |
| Peak allocated | 16,745,565,696 (15.596 GiB) | 16,745,873,408 (15.596 GiB) | 16,745,873,408 (15.596 GiB) | 16,769,614,336 (15.618 GiB) | 16,698,486,272 (15.552 GiB) |
| Peak reserved | 16,909,336,576 (15.748 GiB) | 16,909,336,576 | 16,909,336,576 | 16,909,336,576 | 16,892,559,360 (15.733 GiB) |
| Generation wall-clock | 50.0 s | 818.1 s | 810.0 s | 912.2 s | 43.8 s |
| Mean time per token | 24.6 ms | 21.0 ms | 20.8 ms | 24.1 ms | 21.5 ms |

Every run reached the envelope: peak reserved is within one allocator page
(2 MiB) of M, or 18 MiB for A-dynamic. Load took 5.2–5.9 s in every run,
before generation.

**Reproduced on the shipping kernel.** B and C were first run with an earlier
version of the decode kernel that padded its per-block partials. Re-run on the
shipping (unpadded) kernel, both horizons are **unchanged at 38,913**, and the
burn point (1,537) and freed bytes are identical. Peak allocated fell by 7.7 MB
(16,753,579,520 to 16,745,873,408), the partials no longer padded. Generation
time was 832.3 / 827.0 s before and 818.1 / 810.0 s after.

## Per-token arithmetic

**KV cost.** Qwen3-8B: 36 layers, 8 KV heads, head dim 128, every layer full
attention.

    κ = 2 (K,V) × 36 × 8 × 128 × 2 B = 147,456 B/token = 144 KiB/token

The block cache grows in rows of one 512-token block per layer:

    row = 512 × 147,456 B = 75,497,472 B = 72 MiB

**Burnable budget.** Each layer drops 6,144 neurons from gate, up and down
projections: 3 × 4,096 × 6,144 × 2 B = 150,994,944 B per layer, × 36 layers:

    ΔW = 5,435,817,984 B = 5.0625 GiB   (measured freed: 5,435,817,984 B, exact)

**Prediction.**

    ΔW / κ = 5,435,817,984 / 147,456 = 36,864 tokens = 72 rows exactly

**A.** Headroom after loading = M − reserved = 16,911,433,728 − 16,536,043,520
= 375,390,208 B = 4.97 rows. Four rows fit: 4 × 512 = 2,048 cached tokens, so
the step that needs position 2,049 fails. Measured context at OOM: **2,049**. ✓

**B.** The 0.97 of a row left over in A (73,400,320 B) is too small for a row.
After the burn it is joined by ΔW: 73,400,320 + 5,435,817,984 = 5,509,218,304 B
= 72.97 rows, so 72 more rows. 76 rows × 512 = 38,912 cached tokens: predicted
context at OOM **38,913**, measured **38,913**. ✓

**C.** Headroom = 16,911,433,728 − 11,100,225,536 = 5,811,208,192 B = 76.97
rows, so 76 rows, the same 38,913. ✓

**Gain.** B − A = 38,913 − 2,049 = **36,864 tokens**, exactly the prediction.

## The cache ablation: what the block cache is worth

| | Gain over its own baseline | Of 36,864 predicted | Freed rows converted |
|---|---|---|---|
| Block cache (B − A) | 36,864 tokens | 100.0% | 72 of 72 |
| HF DynamicCache (B-dynamic − A-dynamic) | 35,840 tokens | 97.2% | 70 of 72 (row-equivalents) |

**The block cache is worth 1,024 tokens of horizon here** (2 rows, 151 MB).
Both caches share the same baseline (2,049), so the difference is entirely
after the burn.

The shortfall matches what `torch.cat` must hold in flight. Every step,
`DynamicCache` allocates each layer's new K and V at length T+1 while the old
ones are still live. At T = 37,889 one layer's new K and V take
2 × 37,889 × 2,048 B = 155,193,344 B, against a shortfall of 150,994,944 B
(1,024 tokens). At OOM, `DynamicCache` left 378,849,024 B of the envelope not
holding cache, against 178,689,792 B for the block cache (whose figure includes
the partly allocated 77th row).

**Scope of this number.** At batch 1 on a 36-layer model with 5 GiB freed, the
copying cache loses only 2.8%, because one layer's in-flight copy is about 1/36
of the cache. The loss grows when a layer's share of the cache grows. On the
toy shapes in `docs/VRAM_FINDINGS.md` (batch 64, small budgets) HF's cache
converted 49–72%, and the worst shape had 4 KV-writing layers, which is Gemma 3n
E4B's structure. For Gemma 3n the block cache should matter far more than it
does here; that is still to be measured on the real checkpoint.

Wall-clock: at short context SDPA beats the block kernel (A-dynamic 21.5 ms vs
A 24.6 ms per token); over a long generation the block cache is faster (B 818 s
vs B-dynamic 912 s), because `torch.cat` copies the whole cache every step.

## The burn trigger: a measured safety-vs-quality trade-off

GEI must burn *before* the step that would run out of memory: an out-of-memory
error mid-forward leaves the cache half-updated. The trigger is a probe. Before
any step that will allocate, try to allocate what that step needs plus a 64 MiB
margin; if the probe fails, burn first, then take the step.

The margin buys safety and costs quality. B triggered at a **1,537-token
context**, one row earlier than strictly necessary. At that check the nominal
headroom was 148,897,792 B, and the probe asked for one contiguous 142,606,336 B
block (a 72 MiB row plus the margin). A ~6 MB nominal margin (4%) is within
allocator rounding and fragmentation, so the probe failed, and B burned.

The cost is measured: **512 tokens run on the small configuration that could
have run at full width, and zero horizon**. B still ended exactly where C did.
A smaller margin would keep the large configuration longer, at greater risk of
an out-of-memory error inside a step. With trained nesting, those 512 tokens
are where the quality cost would land. B-dynamic, whose per-step allocations
are smaller, triggered at 2,050.

## Why the block-cache run has no gap

What remains with the block cache is quantisation, which lands exactly here:

- The envelope is used in whole rows of 72 MiB. In every condition 0.97 of a
  row (≈73 MB) stays unusable. Because ΔW is exactly 72 rows, B's fraction is
  the same as A's, and the gain comes out at exactly ΔW/κ. With a ΔW that is not
  a whole number of rows, B − A would sit within one row (512 tokens) of ΔW/κ.
- At OOM the cache held 5,787,090,944 B: 76 rows (5,737,807,872 B) plus 47
  layer blocks of the 77th row (49,283,072 B), allocated before the step failed.
- Weight chunks are packed one buffer per burn level, so the bytes freed equal
  ΔW to the byte and all of them are claimable.

## Caveats

- Batch 1 only. The batched range-gain runs on a trained-nesting checkpoint
  (Gemma 3n E4B, `docs/MODEL_SELECTION.md`) are still to do.
- The envelope is a PyTorch allocator cap on a 24 GB card, not a physically
  smaller device; the CUDA context (~0.5 GB) is outside it.
- Reproduce with `python scripts/range_gain.py --condition {A,B,C,A-dynamic,B-dynamic}
  --envelope-gib 15.75`; raw JSON goes to `results/range_gain_*.json`
  (included).
