# Switch shock vs foreign-cache length (Gemma 3n E4B)

**Verdict:**

- **Switch-shock KL saturates by about 1k foreign tokens.** It rises
  modestly from the ~180-token anchor to ~920 tokens (0.037 → 0.047; paired
  difference +0.0098 [−0.0002, +0.0195], borderline), then stays **flat from
  1k to 16k** (paired 16k − 1k: +0.0009 [−0.014, +0.016]). An 18× longer
  foreign cache adds no measurable shock.
- **C never separates from D.** C − D accuracy is +0.04, 0.00, +0.02, +0.04
  at the four sizes; every 95% interval includes zero.
- **The pilot's runaway signal does not grow with length.** Runaways rise
  with length in every condition, the large model included, and C equals D at
  16k (12 vs 12). Across all 200 paired problems, C-only runaways number 4
  against 6 D-only.

**Limit of the design (read before citing):** in C and D the large model
writes the first half of each answer, restating the question from the far end
of the prompt. After the switch the small model never has to retrieve
information from deep in the foreign cache. So this shows that a 16k-token
foreign cache costs nothing compared with recomputing it, and that its far
entries add no shock. It does **not** show the small model can *retrieve*
from deep in a foreign cache. A switch at token 0 (below) would test that.

Measured 2026-10-26 with `scripts/exp1_dose.py` (354 minutes on one RTX 4090);
summary by `scripts/exp1_dose_report.py`.

## Design

**Task.** A GSM8K test question at the very top of the prompt, then N worked
solutions to unrelated GSM8K *train* problems as filler (calculator
annotations removed), then "Now solve the PROBLEM stated at the very top (not
the examples)". The question is not repeated. Prompt length is set by N; the
same 50 test problems are used in every cell, each with its own seeded filler.

Why this task: the answer is scored exactly as in the pilot; length can be set
to the token; train and test do not overlap; the license is the pilot's; the
only download was the 2 MB train split (disk is nearly full). Long-context
benchmarks with natural documents could not have their length controlled, and
synthetic haystack generators bring their own filler and licenses.

**Anchor.** The ~100-token cell uses the pilot's prompt unchanged. It
reproduces the pilot exactly: all 200 trials (50 problems × 4 conditions)
match the pilot's correctness and token counts, and the shock KL values are
identical.

**Conditions** per problem and cell, as in the pilot: A large throughout (fixes
the switch point at half of A's own answer length); B small throughout; C burn
at the switch, reuse the cache; D burn at the switch, recompute the cache. The
small configuration is the pilot's mix-and-match variant (layers 0–9 at full
width, 10–34 halved), not the trained E2B. HF `DynamicCache` with SDPA, bf16,
greedy, stop at `<end_of_turn>`, cap 8,192 tokens.

**Foreign-cache size** is measured per trial: the tokens in the cache at the
switch (prompt minus the pending token, plus the generated prefix). The
x-axis below is the median per cell.

**Metrics.** Switch-shock KL as in the pilot: at each of the 200 positions
after the switch, KL between the run's next-token distribution and the small
model's on the same prefix with a freshly computed cache. D's value is the
bf16 noise floor. *Runaway*: hit the 8,192-token cap. *Loop*: some 16-token
sequence occurs at least 4 times in the last 512 tokens.

## Results

| Cell | Foreign cache (median) | Shock KL C [95% CI] | KL D (noise floor) | C / D |
|---|---|---|---|---|
| anchor | 184 | 0.0371 [0.0296, 0.0455] | 0.00090 | 41× |
| 1k | 920 | 0.0469 [0.0380, 0.0568] | 0.00169 | 28× |
| 4k | 4,022 | 0.0496 [0.0343, 0.0716] | 0.00110 | 45× |
| 16k | 16,308 | 0.0478 [0.0356, 0.0618] | 0.00088 | 55× |

Paired differences in C's shock KL between cells (same problems):

| | Difference [95% CI] |
|---|---|
| 1k − anchor | +0.0098 [−0.0002, +0.0195] |
| 4k − 1k | +0.0027 [−0.0151, +0.0258] |
| 16k − 4k | −0.0018 [−0.0240, +0.0180] |
| 16k − 1k | +0.0009 [−0.0136, +0.0163] |

**Shock by offset after the switch** (mean KL for C): it is largest
immediately after the switch and falls to about a quarter of that by 100–200
tokens, the same in every cell.

| Cell | tokens 0–10 | 10–50 | 50–100 | 100–200 |
|---|---|---|---|---|
| anchor | 0.094 | 0.040 | 0.022 | 0.023 |
| 1k | 0.086 | 0.042 | 0.038 | 0.026 |
| 4k | 0.080 | 0.049 | 0.038 | 0.026 |
| 16k | 0.105 | 0.053 | 0.036 | 0.028 |

**Accuracy** (n = 50 per cell):

| Cell | A | B | C | D | C − D [95% CI] | C − B [95% CI] |
|---|---|---|---|---|---|---|
| anchor | 0.90 | 0.82 | 0.88 | 0.84 | +0.04 [−0.04, +0.12] | +0.06 [−0.06, +0.18] |
| 1k | 0.84 | 0.72 | 0.76 | 0.76 | +0.00 [−0.12, +0.12] | +0.04 [−0.08, +0.16] |
| 4k | 0.82 | 0.56 | 0.74 | 0.72 | +0.02 [−0.08, +0.12] | +0.18 [+0.04, +0.32] |
| 16k | 0.64 | 0.26 | 0.58 | 0.54 | +0.04 [−0.06, +0.16] | +0.32 [+0.14, +0.50] |

**Runaways and loops** (count of 50):

| Cell | Runaway A / B / C / D | Loop A / B / C / D | C-only vs D-only runaway |
|---|---|---|---|
| anchor | 0 / 3 / 2 / 1 | 0 / 5 / 2 / 1 | 2 vs 1 |
| 1k | 1 / 5 / 2 / 2 | 1 / 4 / 2 / 2 | 0 vs 0 |
| 4k | 0 / 7 / 1 / 4 | 0 / 6 / 1 / 4 | 0 vs 3 |
| 16k | 7 / 7 / 12 / 12 | 7 / 9 / 11 / 11 | 2 vs 2 |

## Reading the results

**Why the curve saturates.** Only part of Gemma 3n's cache can be foreign.
Layers 0–9 keep their FFNs, so the residual stream reaching the attention of
layers 0–10 is identical in both configurations, and so are their K and V.
Foreign KV lives only in layers 11–19 (layers 20–34 reuse the caches of
layers 18 and 19). Of those, only layers 14 and 19 are full attention and keep
the whole history; the rest are sliding windows that hold at most 512 tokens.
So past ~512 tokens the only thing that grows is far history in two layers,
and the measurement shows it adds no shock. This is derived from the config,
not measured directly.

**Shock is local in time.** In every cell the KL is highest in the first 10
tokens after the switch (0.08–0.105) and falls to ~0.026 by 100–200 tokens. The
sliding windows refill with the small model's own KV as it generates, which
fits this decay.

**C and D track each other at every length; B does not.** B collapses with
length (0.82 → 0.26) far faster than A (0.90 → 0.64). The mix-and-match small
configuration is much worse at reading a long prompt itself. C and D stay
close to A because the large model writes the first half of the answer (the
design limit above). The large C − B gap at 16k (+0.32) therefore measures
"half the answer by the large model" more than any property of the reused
cache.

**Runaways are a length effect, not a reuse effect.** At 16k, A (large, no
switch) runs away 7 times; C and D 12 each. The pilot's 5-vs-1 C-over-D signal
does not reappear: summed over all cells, 4 C-only against 6 D-only.

## What this does not establish, and the next test

- **Retrieval from deep in a foreign cache.** Needs the switch *before* the
  answer starts, so the small model must read the question from the far end
  of the large model's prompt cache. C switched at token 0 against B
  (recomputing at token 0 equals B: verified 5/5 in the pilot) isolates exactly
  that. One extra decode per problem per cell.
- Longer foreign caches than 16k (Gemma 3n's limit is 32k), other tasks,
  sampling instead of greedy, more seeds.
- The trained E2B: `HFElasticModel` supports width nesting only, and E2B also
  drops 5 layers.

## Licenses (artifact statement)

| Artifact | Used for | License | Source |
|---|---|---|---|
| GSM8K test split (`openai/gsm8k` @ `740312add8`, `main`) | questions | MIT | dataset card |
| GSM8K train split (same revision) | filler (worked solutions) | MIT | dataset card |
| Gemma 3n E4B-it (`google/gemma-3n-E4B-it`) | model | Gemma Terms of Use | model card |

## Reproduce

    python scripts/exp1_dose.py             # 4 cells x 50 problems, ~6 h on an RTX 4090
    python scripts/exp1_dose_report.py      # summary table

Trials go to `results/exp1_dose_trials.jsonl` (included), one record per
problem, cell and condition, with per-offset KL; the run resumes from it.
