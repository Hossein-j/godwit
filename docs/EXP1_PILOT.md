# Experiment 1 pilot: can the small configuration continue from the large one's cache?

**Verdict: case (1), KV reuse works**, on this pilot. Switching mid-answer
from the large to the small configuration and keeping the large model's cache
(C) scored no worse than running small from the start (B): 0.85 vs 0.82,
paired difference +0.03 [−0.04, +0.10]. Both decision tests agree, and so do
both scoring rules below.

The reuse is not free: it shifts the next-token distribution by 39× the
numerical noise floor, and C ran into repetition loops more often than D
(5 vs 1 runs). Neither shows up as an accuracy loss at n = 100, but both are
the first things to watch in the full experiment.

Measured 2026-10-21 with `scripts/exp1_pilot.py` (86 minutes on one RTX 4090).

## What the small configuration is

**The small configuration is a mix-and-match variant, not the trained E2B.**
It keeps FFN layers 0–9 at full width (16,384) and halves layers 10–34 to
8,192 neurons. Layers 0–9 are excluded because their activation sparsity
thresholds over all FFN neurons and cannot be split into chunks. Gemma 3n's
trained E2B instead halves every layer and drops 5 whole layers, which
`HFElasticModel` does not support. The variant sits inside Gemma 3n's trained
MatFormer nesting (prefix FFN slices), but this exact width pattern was never
evaluated by Google. **So condition B is a weaker reference than it appears:**
it is a plausible but untrained combination, not a released sub-model, and its
accuracy (0.82) is not E2B's.

## Setup

| | |
|---|---|
| Model | `google/gemma-3n-E4B-it`, text-only (`Gemma3nForCausalLM`; vision and audio not loaded), bf16 |
| Configurations | large 12.79 GiB; small 10.45 GiB (2.34 GiB freed per burn, exact) |
| Task | GSM8K test, first 100 problems (`openai/gsm8k`, revision `740312add8`) |
| Prompt | question + "Solve the problem step by step. End with a final line of the form 'The answer is N'." (chat template) |
| Decoding | greedy, one seed, stop at `<end_of_turn>`, cap 8,192 tokens |
| Switch point | half of A's own generated length per problem (median token 95, range 42–217) |
| Cache | HF `DynamicCache` with SDPA attention in every condition. The block cache and its Triton kernel are not used: this measures quality, not memory |
| Scoring | final answer = number after "The answer is", else the last number; equals the gold answer after `####` |

**Conditions:** A large throughout; B small throughout (burned before the
prompt); C burn at the switch point, keep the cache (REUSE); D burn at the
switch point, rebuild the cache with the small model (RECOMPUTE).

**Precision.** The Gemma 3n text model needs 25.6 GiB in fp32, more than the
card has, so the pilot runs in bf16 with these pre-flight checks on the first 5
problems:

- **B = D switched at token 0:** identical tokens on **5/5**. The small model
  reached through a switch computes exactly what it computes from the start.
- **A via `HFElasticModel` = plain HF `generate`:** identical on **4/5**. On
  problem 3 one near-tie word differs mid-answer ("Total meters per week" vs
  "Meters per week"), from bf16 rounding of the two-chunk FFN sum; same final
  answer. This cannot bias C against B: all four conditions run through the
  same `HFElasticModel` path, and C and B differ only in where the cache
  came from.

## Results

| | A: large | B: small | C: switch, reuse | D: switch, recompute |
|---|---|---|---|---|
| Accuracy | 0.90 | 0.82 | **0.85** | 0.86 |
| 95% CI (bootstrap) | [0.84, 0.95] | [0.74, 0.89] | [0.78, 0.92] | [0.79, 0.93] |
| Strict accuracy (runaway counts as wrong) | 0.90 | 0.82 | 0.84 | 0.86 |
| Mean quality (log-prob under A) | −0.042 | −0.267 | −0.132 | −0.136 |
| Switch-shock KL (mean / median / p90) | | | 0.0302 / 0.0217 / 0.0693 | 0.00078 / 0.00056 / 0.00144 |
| Runaways (hit 8,192 without stopping) | 0 | 5 | 5 | 1 |
| Generated tokens, median (mean) | 190 (205) | 198 (637) | 197 (625) | 194 (313) |

**Paired accuracy differences** (same 100 problems, bootstrap over problems,
95%):

| | Standard scoring | Strict scoring |
|---|---|---|
| C − B | **+0.03 [−0.04, +0.10]** | +0.02 [−0.05, +0.09] |
| D − B | +0.04 [−0.02, +0.11] | +0.04 [−0.02, +0.11] |
| C − D | −0.01 [−0.06, +0.04] | −0.02 [−0.07, +0.03] |
| A − B | +0.08 [+0.01, +0.15] | +0.08 [+0.01, +0.15] |

Discordant problems: C right and B wrong on 9, the reverse on 6; C right and D
wrong on 3, the reverse on 4.

## Decision rule

- **`run_exp1.py`'s rule** (C's 95% interval overlaps B's): **PASS, case (1)**.
  This rule is lenient: at n = 100 it would also pass a drop of up to ~15 points.
- **Paired test** (C − B interval): C is not below B. The interval excludes
  any drop larger than 4 points (5 under strict scoring). **Case (1)**.

The two agree. Case (2) would need C below B with D recovering; C − D is
−0.01 [−0.06, +0.04], so there is nothing for recomputation to recover.

## Reading the numbers

**The burn has a detectable cost; the switch does not add to it.** A − B =
+0.08 [+0.01, +0.15]: halving layers 10–34 costs accuracy. C and D both land
between B and A, as expected when half the answer was written by the large
model.

**Mean quality** is the mean log-probability the large model assigns to each
condition's own output tokens (teacher-forced), so it measures drift from what
A would have written. C (−0.132) and D (−0.136) are the same, half way between
A and B. Reusing the cache does not move the output further from the large
model than recomputing it does.

**Switch-shock KL** is computed on each switched run's own tokens: KL between
the run's next-token distribution and the small model's on the same prefix
with a freshly computed cache, over the 200 tokens after the switch. For D it
is the bf16 noise floor (same model, cache built in one pass vs
incrementally): 0.00078. For C it is **39× that** (0.030), and 94 of C's 100
values exceed the largest of D's. So the reused cache really does change what
the small model predicts. On GSM8K at this n the change does not cost
accuracy.

**Runaways are the one sign of trouble.** C failed to stop on 5 problems where
D stopped (5, 26, 78, 92, 93); D failed to stop on 1 where C stopped (21). On 3
of them (26, 78, 93) both B and D stopped with the right answer, so the loop
looks specific to the reused cache. 5 vs 1 is not significant (exact sign test
p ≈ 0.22), but it is the failure mode to measure in the full run. One runaway
(C, problem 5) was scored correct because "The answer is 64" appears inside
the loop; strict scoring counts it as wrong.

## Limits of the pilot

- n = 100, one seed, one switch point: the paired test can rule out C being
  more than ~4–5 points worse than B, not smaller effects.
- GSM8K answers are short (median ~195 tokens), so the switch lands early
  (median token 95) and the reused cache holds only ~100 generated tokens plus
  the prompt. Longer generations, which put more of the parent's cache in front
  of the small model, are the harder test.
- B is a mix-and-match width pattern, not a trained sub-model (above).
- Greedy decoding only.

## Licenses (artifact statement)

| Artifact | Used for | License | Source |
|---|---|---|---|
| GSM8K (`openai/gsm8k`, revision `740312add8`, `main` test split) | evaluation data | MIT | dataset card: "The GSM8K dataset is licensed under the MIT License" |
| Gemma 3n E4B-it (`google/gemma-3n-E4B-it`) | model under test | Gemma Terms of Use (Hub license field `gemma`; gated, accepted on the account used) | model card |

No other evaluation dataset was used in this pilot.

## Reproduce

    python scripts/exp1_pilot.py            # 100 problems, ~1.5 h on an RTX 4090

Per-trial records (tokens, switch point, prediction, gold, text, metrics) go
to `results/exp1_pilot_trials.jsonl`, the summary to `results/exp1_pilot.json`
(both included in this repository). The run resumes from the trials file if interrupted.
