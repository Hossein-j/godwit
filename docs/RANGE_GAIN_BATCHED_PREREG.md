# Pre-registration: batched range gain on Gemma 3n E4B

Written and committed **before** any run under the registered envelope
(2026-10-28). Script: `scripts/range_gain_batched.py`. Any deviation from what
follows will be reported as such.

## Why this run

The batch-1 range gain (`docs/RANGE_GAIN.md`) was measured on Qwen3-8B, whose
nesting is untrained, so the model that shows the gain cannot deploy it. The
scoping criterion (`docs/MODEL_SELECTION.md`) says a trained-nesting checkpoint
such as Gemma 3n E4B binds only under batching (B ≥ 9 at full context). This
run measures the range gain there, at batch 16.

## Fixed in advance

| | |
|---|---|
| Model | `google/gemma-3n-E4B-it`, text-only, bf16, `HFElasticModel` with layers 0–9 excluded (the Experiment 1 mix-and-match small configuration) |
| Batch | 16 sequences, the same 14-token prompt, decoded in lockstep (no padding) |
| Cache | block cache, 512-token blocks; ablation: HF `DynamicCache` |
| Envelope | allocator cap M = **13.40 GiB** |
| Burn trigger | probe for the next allocation plus a **128 MiB** margin (raised from the Qwen run's 64 MiB because calibration measured ~66 MiB of non-cache per-step transients at batch 16) |
| Generation | greedy, end-of-sequence ignored, until out of memory or 30,000 steps |
| Conditions | A (large, never burns), B (burn when the envelope binds), C (small from the start), A-dynamic and B-dynamic (A and B with `DynamicCache`); one process each, one run each |

Calibration before this document (no envelope cap, so no outcome data):
13,748,928,512 B reserved after loading; ΔW = 2,516,582,400 B; one block row
= 67,108,864 B; 32.5 ms per step; worst per-step transient 129.7 MiB including
one new row.

## Predictions

**P1 (primary): the range gain is 18,944 or 19,456 tokens per sequence.**
Per sequence, the cache grows by κ = 8 KiB per token (4 full-attention KV
layers × K,V × 2 KV heads × 256 × 2 B), so the batch grows by 16 × 8 KiB =
128 KiB per step and one 512-token block row is 64 MiB. Burning frees
ΔW = 2,516,582,400 B = 37.5 rows. A run ends when the next whole row does not
fit, so the gain is 37 or 38 whole rows, depending on the fraction of a row A
leaves unused: **gain = context(B) − context(A) ∈ {18,944, 19,456}**. Any other
value falsifies the arithmetic.

**P2: B and C reach the same horizon.** context(B) = context(C), as on
Qwen3-8B: after the burn, B holds the same weights as C, and its leftover
partial row plus ΔW is what C has from the start.

**P3 (secondary): the copying cache loses at least the one-layer copy.** With 4
equal full-attention KV layers, `torch.cat` holds one layer's new K and V, a
quarter of the growing cache, in flight at each step, so the cache can use at
most 4/5 of the room available to the block cache. Sliding-window copies and
fragmentation can only add loss. **Prediction: context(B-dynamic) ≤ 0.8 ×
context(B).** A B-dynamic horizon above that would falsify the one-layer-copy
account of the copy cost. This is the first test of that account made before the
data; on Qwen3-8B it was fitted after the fact.

## Analysis

Report every condition's context at out-of-memory, the burn point, bytes freed,
peak memory and wall-clock. Token counts are deterministic (greedy decoding,
failure fixed by allocation arithmetic), so one run per condition is the whole
experiment; no extension or rerun is planned. If B runs out of memory inside a
step rather than burning first, that is reported as a failure of the trigger.
