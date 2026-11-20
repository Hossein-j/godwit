# Results: GEI against KV-cache eviction at equal memory

Design and hypotheses: `docs/PREREG.md` (frozen 2026-11-18 10:34 PST, SHA-256 in
`docs/PREREG.sha256`, emailed to the co-author before any eviction data).
Data: `results/evict_stream_trials.jsonl`, `results/evict_snap_trials.jsonl`;
A, B, C0 from the stored worst-case run, `../results/exp1_worst_trials.jsonl`.
Analysis: `scripts/report.py` -> `results/evict_summary.json`, `logs/report.txt`.
Completed 2026-11-18.

## Checks

1. **Reproduction:** the reinstalled environment (torch 2.12.0+cu126,
   transformers 4.57.6, RTX 4090, driver 615.71.09) reproduced 30/30 stored A and
   C0 records exactly (text and token count; 1k, 4k, 16k, problems 0-4). The
   stored A, B, C0 records are used as they are.
2. **Exact zero:** both eviction policies with an unlimited budget gave tokens
   identical to stored A on 20/20 anchor-cell records, with no evictions.

## Accuracy (GSM8K problems 0-49, budget L = 1,024 growing-cache tokens)

Bootstrap 95% CIs in brackets.

| Cell | A (large, full cache) | B (small) | C0 (GEI) | E-stream | E-snap |
|---|---|---|---|---|---|
| 1k | 0.84 [0.74, 0.94] | 0.72 | 0.76 [0.64, 0.86] | 0.80 [0.68, 0.90] | 0.92 [0.84, 0.98] |
| 4k | 0.82 [0.70, 0.92] | 0.56 | 0.62 [0.48, 0.76] | 0.00 [0.00, 0.00] | 0.36 [0.22, 0.50] |
| 16k | 0.64 [0.50, 0.76] | 0.26* | 0.40 [0.26, 0.54] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.14] |

\* B below the weak-baseline threshold at 16k; no comparison here uses B.

## Hypotheses

Paired bootstrap CI of the difference; McNemar exact test on discordant pairs
(C0-only / eviction-only). No multiplicity correction.

| Cell | Comparison | Difference | Discordant | p |
|---|---|---|---|---|
| 4k | **H1** C0 − E-stream | +0.62 [+0.48, +0.76] | 31 / 0 | 9.3e-10 |
| 16k | **H1** C0 − E-stream | +0.40 [+0.26, +0.54] | 20 / 0 | 1.9e-06 |
| 4k | **H2** C0 − E-snap | +0.26 [+0.10, +0.42] | 16 / 3 | 0.0044 |
| 16k | **H2** C0 − E-snap | +0.34 [+0.18, +0.50] | 19 / 2 | 0.00022 |
| 1k (descriptive) | C0 − E-stream | −0.04 [−0.20, +0.12] | 7 / 9 | 0.80 |
| 1k (descriptive) | C0 − E-snap | −0.16 [−0.30, −0.04] | 2 / 10 | 0.039 |

- **H1 supported** at 4k and 16k: GEI solves 31 and 20 problems StreamingLLM
  misses, and StreamingLLM solves none that GEI misses.
- **H2 (two-sided):** GEI is higher than SnapKV at 4k and 16k.
- **At 1k SnapKV is higher than GEI** (and higher than A, 0.92 against 0.84,
  4 / 0 discordant, p = 0.12). The budget binds in 46/50 SnapKV trials but only
  a small part of the ~820-token prompt is removed; a plausible reading, not
  tested here, is that removing filler helps the large model.

## Secondary measures

| Cell | Policy | Question tokens kept (mean, min-max) | Runaway | Wrong question | A − eviction |
|---|---|---|---|---|---|
| 1k | stream | 1.00 | 1 | 0 | +0.04 (p 0.50) |
| 1k | snap | 1.00 (0.97-1.00) | 0 | 0 | −0.08 (p 0.12) |
| 4k | stream | 0.00 | 5 | 12 | +0.82 (p 9.1e-13) |
| 4k | snap | 0.77 (0.59-0.96) | 5 | 5 | +0.46 (p 5.6e-06) |
| 16k | stream | 0.00 | 6 | 24 | +0.64 (p 4.7e-10) |
| 16k | snap | 0.38 (0.18-0.62) | 16 | 17 | +0.58 (p 3.0e-08) |

The mechanism predicted for H1 holds: StreamingLLM keeps no question token at
4k or 16k, and "wrong question" (answering a filler example) is its most common
failure at 16k. SnapKV keeps less of the question as the prompt grows (0.77 at
4k, 0.38 at 16k) and fails more often by running away to the 8,192-token cap at
16k. The "wrong question" rule is the same classifier as `EXP1_WORST_PREREG.md`,
with its known misfire; failure-class counts have not been hand-checked.

## Reading

Under the paper's deployable envelope, keeping the whole prompt on the small
configuration (GEI) beats keeping the large model and evicting cache once the
prompt is several times the budget (4k, 16k), against both a recency policy and
an attention-scored one. When only a little must be evicted (1k), SnapKV is
better than GEI. The gap between GEI and eviction grows with the share of the
prompt that eviction must discard.

## Limitations

- The budget comes from the memory arithmetic of the batched envelope; the
  quality runs are at batch 1, not a physical OOM at batch 16.
- One model (Gemma 3n E4B), one task (GSM8K with foreign worked examples as
  filler), 50 problems per cell, greedy decoding, one seed.
- Two eviction methods; H2O-style per-step attention scoring, KV quantisation,
  and eviction combined with burning are not tested.
- The question sits at the top of the prompt by design, which is the worst case
  for recency eviction.
- The model revision is not recorded in the paper; this study used
  `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10` (Hub main, 2025-07-14), and the
  30/30 reproduction indicates the same weights.
