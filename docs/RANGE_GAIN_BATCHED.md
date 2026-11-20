# Batched range gain on Gemma 3n E4B (trained nesting, batch 16)

Pre-registered in `docs/RANGE_GAIN_BATCHED_PREREG.md` (commit `d58b6be`,
before any run under the registered envelope). Measured 2026-10-27 with
`scripts/range_gain_batched.py`; raw results in
`results/range_gain_batched_*.json`.

**Result: on a deployable trained-nesting checkpoint, burning added 19,455
tokens of context per sequence for each of 16 sequences, and the burned run
reached exactly the small configuration's horizon.** Two of the three
pre-registered predictions were met; the primary prediction missed its
registered value by one token (below).

## Setup (as registered)

Gemma 3n E4B text model, bf16, layers 0–9 excluded from burning (the
Experiment 1 mix-and-match small configuration); 16 copies of a 15-token prompt
decoded in lockstep; block cache with 512-token blocks (ablation: HF
`DynamicCache`); allocator cap M = 13.40 GiB (14,388,140,441 B); burn probe for
the next allocation plus 128 MiB; greedy, end-of-sequence ignored, until out of
memory. One process and one run per condition. This measures horizon and
memory, not quality.

## Results

| Condition | Context at OOM (per sequence) | Steps | Burn at context | Bytes freed | Generation |
|---|---|---|---|---|---|
| A: large, never burns | 1,026 | 1,011 | — | — | 33.4 s |
| **B: GEI, burn when the envelope binds** | **20,481** | 20,466 | 1,025 | 2,516,582,400 (exact) | 755.7 s |
| C: small from the start | 20,481 | 20,466 | before the prompt | 2,516,582,400 | 758.1 s |
| A-dynamic | 1,153 | 1,138 | — | — | 36.2 s |
| B-dynamic | 13,314 | 13,299 | 400 | 2,516,582,400 | 482.5 s |

The switch took 79 ms (B) and 80 ms (B-dynamic). Every run reached the
envelope: peak reserved 14,369,685,504–14,386,462,720 B against
M = 14,388,140,441 B.

## The pre-registered predictions

**P1: gain ∈ {18,944, 19,456} tokens per sequence: NOT MET as registered.**
Measured context(B) − context(A) = 20,481 − 1,026 = **19,455**, one token
below the registered 19,456.

*Post-hoc explanation (not a reinterpretation of the prediction):* the
registered values assumed that every run fails when its next 512-token block
row does not fit. B and C did: each ended with exactly 40 full rows
(14 prefilled + 20,466 decoded = 20,480 cached tokens) and failed on the step
that needed row 41. A did not: it allocated its third row, wrote one token into
it (14 + 1,011 = 1,025 cached tokens), and failed on the next step's transient
(~66 MiB of logits and activations at batch 16, measured in calibration), not on
a row allocation. Counted in rows, the gain is 38, one of the two predicted row
counts; counted in tokens as registered, it is 19,455. ΔW / (16 × 8 KiB) =
19,200 tokens = 37.5 rows, so the measurement lies within one row of the
unrounded arithmetic either way.

**P2: context(B) = context(C): MET.** Both 20,481.

**P3: context(B-dynamic) ≤ 0.8 × context(B): MET.** 13,314 / 20,481 = **0.650**.
The bound from the one-layer copy (4 equal full-attention KV layers, so a
quarter of the growing cache in flight) was stated before the data. The
copying cache lost more than the bound alone: the other per-step transients and
fragmentation add to it, as the account predicts they can.

## What the cache is worth here

| | Gain over its own baseline | Share of the block cache's gain |
|---|---|---|
| Block cache (B − A) | 19,455 tokens | 100% |
| HF `DynamicCache` (B-dynamic − A-dynamic) | 12,161 tokens | 62.5% |

On this checkpoint, with only 4 KV-writing full-attention layers, the copying
cache gives up 7,294 tokens per sequence, 37.5% of the gain. On Qwen3-8B at
batch 1 (36 such layers) it gave up 1,024 tokens, 2.8%
(`docs/RANGE_GAIN.md`). This is the real-scale counterpart of the 4-KV-layer
toy shape in `docs/VRAM_FINDINGS.md` (49.0%).

A-dynamic reached a slightly longer context than A (1,153 vs 1,026) because a
copying cache grows one token at a time, while the block cache commits a
64 MiB row (512 tokens for 16 sequences) at once.

## Arithmetic

    κ per sequence   = 4 layers × 2 (K,V) × 2 KV heads × 256 × 2 B = 8 KiB/token
    batch per step   = 16 × 8 KiB = 128 KiB
    row (512 tokens) = 512 × 128 KiB = 67,108,864 B
    ΔW               = 25 layers × 3 × 2,048 × 8,192 × 2 B = 2,516,582,400 B = 37.5 rows
    ΔW / (16 κ)      = 19,200 tokens per sequence

## Limits

One model, one batch size, one prompt repeated 16 times (content does not
affect allocation), greedy decoding, one run per condition (token counts are
deterministic), and an allocator cap rather than a smaller device.
