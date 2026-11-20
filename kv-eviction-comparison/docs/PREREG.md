# Pre-registration: GEI against KV-cache eviction at equal memory

**Status:** design approved by the first author on 2026-11-18 10:34 PST, before any eviction
code or data existed. Frozen with the SHA-256 in `PREREG.sha256`. A third-party
timestamp (for example an email of this file to the co-author) is still to be
made; until then this is an internal pre-specification, not a public
pre-registration.

## Question

Under the same memory envelope, which works better for a long-prompt task:
keeping the **large** model and **evicting** KV cache to fit, or **burning** to
the small configuration and keeping the **full** cache (GEI)?

This is the baseline a reviewer will ask about first: KV eviction is the
standard way to fit a long context into fixed memory, and it never runs out of
memory, so the comparison is quality at a fixed task, not horizon.

## Setting: the paper's deployable envelope

Gemma 3n E4B, the registered batched envelope from `RANGE_GAIN_BATCHED_PREREG.md`:
13.40 GiB allocator cap, batch 16. Measured per-sequence horizons
(`RANGE_GAIN_BATCHED.md`):

| | Context per sequence that fits |
|---|---|
| Large, never burns (block cache) | 1,026 tokens |
| GEI, burn when the envelope binds | 20,481 tokens |

So under this envelope the large model can hold **L = 1,024** tokens of
growing cache per sequence (two 512-token block rows), and GEI can hold the
whole prompt. The quality runs are done at batch 1, as in the paper's Section 5,
with the large model's growing cache **capped at L = 1,024 tokens** by the
eviction policy. The memory arithmetic, not a physical OOM, sets the budget;
this is stated as a limitation.

On Gemma 3n only the four full-attention KV-writing layers (in layers 0-19)
grow; sliding-window layers are fixed at 512 tokens and layers 20-34 reuse
earlier caches. Eviction is applied to the growing full-attention caches only,
which is exactly the memory the budget counts. Layers that reuse a cache see
the evicted cache.

## Conditions (cells 1k, 4k, 16k; GSM8K test problems 0-49; same prompts)

Reused from the paper's worst-case run (`results/exp1_worst_trials.jsonl`), after
the reproduction check below:

- **A** large throughout, full cache (needs more memory than the envelope allows
  at 4k and 16k: reference only).
- **B** small throughout, full cache.
- **C0** GEI: large prefill, burn at token 0, small generates every token,
  reusing the cache. This is what GEI does when the prompt fills the envelope.

New:

- **E-stream** large throughout, StreamingLLM eviction: keep the first 4 tokens
  (attention sinks) and the most recent L − 4 tokens, applied after prefill and
  after every step.
- **E-snap** large throughout, SnapKV-style eviction: after prefill, score
  prompt positions per KV head by the attention of the last 32 prompt tokens
  (summed over the query heads of the group, max-pooled with kernel 7); keep
  the top L − 32 − 256 positions plus the 32-token window, leaving 256 tokens of
  room for the answer. During generation, once the cache exceeds L, the oldest
  generated token is evicted; the compressed prompt is kept.

The anchor cell (88-token prompts) is used only for the zero check.

Decoding as in the paper: greedy, one seed, bf16, 8,192-token cap, SDPA where
attention scores are not needed.

## Checks before results are trusted

1. **Reproduction.** The software environment used for the paper is no longer
   installed. Reinstall torch 2.12.0+cu126 and transformers 4.57.6, rerun A and
   C0 on problems 0-4 of each cell, and compare text and token count with the
   stored records. Any mismatch means rerunning A, B and C0 in full in the new
   environment before any comparison.
2. **Exact zero for eviction.** With the budget set above prompt + answer
   length (anchor cell, problems 0-9), E-stream and E-snap must produce token
   sequences identical to A. Any mismatch is a bug and is fixed before data is
   collected.

## Hypotheses and predictions

- **H1 (directional).** At 4k and 16k, C0 scores higher than E-stream. Reason:
  the question sits at the top of the prompt and StreamingLLM keeps only the
  sinks and the most recent tokens, so the question is evicted while the
  closing instruction ("solve the PROBLEM stated at the very top") is kept.
- **H2 (two-sided, no predicted direction).** C0 against E-snap at 4k and 16k.
- At 1k the budget binds only when prompt + answer exceeds 1,024 tokens;
  reported descriptively.

## Measures and tests

- Accuracy per condition with bootstrap 95% CIs.
- Paired differences C0 − E-stream and C0 − E-snap with paired bootstrap CIs,
  and McNemar's exact test on discordant pairs, per cell. No multiplicity
  correction; the two comparisons and three cells are reported in full.
- Secondary: A − E-snap (what eviction costs the large model), runaway rate,
  and for E-snap whether the question tokens survived compression (fraction of
  question token positions kept, averaged over KV heads and layers).
- Failure classes as in `EXP1_WORST_PREREG.md`, with the known misfire of the
  "wrong question" rule reported as before.

## Weak-baseline flag

As in the worst-case run, B solves 13/50 at 16k; accuracy comparisons that use
B at 16k are flagged as not meaningful. The C0-against-eviction comparisons do
not use B.

## What this does not test

Other eviction methods (H2O and its variants need per-step attention scores),
KV quantisation, combining eviction with burning, natural long-document tasks,
and a physical OOM at batch 16 rather than a token budget.
