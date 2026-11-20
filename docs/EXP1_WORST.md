# Switch at token 0: the worst case for cache reuse (Gemma 3n E4B)

Pre-registered in `docs/EXP1_WORST_PREREG.md` (commit `40a7dfa`, before any
data). Measured 2026-10-26 with `scripts/exp1_worst.py` (150 minutes on one
RTX 4090); analysis by `scripts/exp1_worst_report.py`.

## The exact measurement first

At token 0, C0 (large model wrote the prompt cache, small model generates) and
B (small model wrote its own cache) condition on **identical context**. Their
first-token distributions can therefore be compared exactly, with no teacher
forcing. And the recompute path gives a true zero: D0 (large prefill, switch
at token 0, recompute) produced first-token log-probabilities **bitwise
identical** to B's on all 40 problems checked, so KL(D0 ‖ B) = **0 exactly**
(the KL routine's own floor is ≤ 3 × 10⁻⁸). Any nonzero KL(C0 ‖ B) is
therefore the cache and nothing else.

| Foreign cache (median tokens) | First-token KL(C0 ‖ B), mean [95% CI] | median | p90 | KL(D0 ‖ B) |
|---|---|---|---|---|
| 88 | 0.244 [0.174, 0.321] | 0.107 | 0.590 | 0 (exact) |
| 822 | 0.344 [0.246, 0.450] | 0.237 | 0.800 | 0 (exact) |
| 3,922 | 0.561 [0.390, 0.755] | 0.294 | 1.296 | 0 (exact) |
| 16,206 | 0.543 [0.390, 0.716] | 0.321 | 1.422 | 0 (exact) |

**The worst-case shock grows with foreign-cache length up to about 4k tokens,
then saturates.** Paired differences (same problems): 4k − 1k = +0.22
[+0.01, +0.44]; 16k − ~100 = +0.30 [+0.12, +0.49]; but 16k − 4k = −0.02
[−0.25, +0.22].

It is also much larger than in the typical case. With the switch mid-answer
(`docs/EXP1_DOSE.md`), the first token after the switch is also an exact
comparison, and its KL was 0.02–0.12 with no trend in length. Without a
bridge, the first-token shock is 2–14× larger, and only here does it grow with
length.

The shock is brief: over the 200 tokens after the switch, mean KL falls from
0.24–0.57 at the first token to 0.02–0.03 by tokens 50–200 in every cell.

## Verdict

**The large shock does not turn into an accuracy loss.** C0 scored at or
above B in every cell, and no difference reached significance:

| Foreign cache | A | B | C0 | C0 − B [95% CI] | McNemar: C0-only vs B-only, p | P(C0 ✓ \| B ✓) |
|---|---|---|---|---|---|---|
| 88 | 0.90 | 0.82 | 0.88 | +0.06 [−0.02, +0.16] | 4 vs 1, p = 0.38 | 0.98 |
| 822 | 0.84 | 0.72 | 0.76 | +0.04 [−0.08, +0.16] | 5 vs 3, p = 0.73 | 0.92 |
| 3,922 | 0.82 | 0.56 | 0.62 | +0.06 [−0.08, +0.20] | 9 vs 6, p = 0.61 | 0.79 |
| 16,206 ⚠ | 0.64 | 0.26 | 0.40 | +0.14 [−0.02, +0.30] | 12 vs 5, p = 0.14 | 0.62 |

⚠ **16k: B too weak for the accuracy comparison** (13 of 50 solved, below the
pre-registered 20). KL and failure analysis remain valid there.

**The extension rule was not triggered.** It required C0 < B in accuracy, which
no cell showed, so every cell stays at 50 problems as pre-registered.

**Power.** At 4k (B solves 28 of 50) the 95% interval on C0 − B still spans
−0.08 to +0.20, so a C0 deficit of up to ~8 points cannot be excluded there.
At ~100 and 1k, the intervals exclude deficits beyond 2 and 8 points. What can
be said is narrower than "no effect": **no sign of harm, and the direction
favours C0 in every cell.**

## How C0 and B fail

**Pre-registered classes** (fixed rules, applied to every miss):

| Foreign cache | B misses: runaway/loop · wrong question · no question · slip | C0 misses: runaway/loop · wrong question · no question · slip |
|---|---|---|
| 88 | 3 · 0 · 0 · 6 | 1 · 0 · 0 · 5 |
| 822 | 5 · 0 · 0 · 9 | 4 · 0 · 0 · 8 |
| 3,922 | 7 · 3 · 0 · 12 | 6 · 2 · 0 · 11 |
| 16,206 | 9 · 10 · 0 · 18 | 11 · 6 · 0 · 13 |

**Hand check: the "wrong question" rule does not work at long lengths.** All
21 "wrong question" cases were read by hand, along with 8 slips flagged as
possible misreadings (question beside the first ~170 characters of the
response). **None answered a filler problem outright.** Every one works on the
target problem. The rule fired because the predicted number coincided with one of the
~70 filler answers in the prompt, which are mostly small integers, so
collisions are common. This is a disagreement with the pre-registered rule,
reported as the pre-registration requires.

**What the hand check found instead: question misreading** (post hoc, my
labels, not pre-registered). The response restates the right problem with a
number or relation corrupted. All cases are at 16k:

- **B: 8** (#12 "6 lemons" for 7; #14 invents a "female students" question; #18;
  #20 "spilled 10 liters" for 1; #29; #40 "his sister Sara"; #42 "24 pieces"
  for 8; #47 reverses the red/blue ratio).
- **C0: 3** (#18; #34 "Aaron has 2 fewer jewels than Raymond"; #40 "his sister
  Ben's iPhone ... 10 years old").
- **#18 is the same corruption, word for word, in B and C0** ("How many more
  eggs does she need to buy if she wants to make 4 omelets?"). That misreading
  belongs to the small model, not to the foreign cache.
- **Three are blends with the filler**, checked against the prompts: B #14's
  "female" and B #40's "sister ... Sara" occur in filler examples in the same
  prompt but not in the question, and C0 #40's "sister" comes from the same
  example. So B has 2 blends and 6 pure misreadings; C0 has 1 blend and 2 pure
  misreadings. The other corrupted numbers ("6 lemons", "24 pieces", "fewer
  jewels than Raymond") appear in neither the question nor the filler.

So the retrieval failure this experiment was designed to catch is real at
16k, but it is **mostly B's**: the small model misreads a far-back question
more often with its own cache (8) than with the large model's (3), and blends
in filler content more often (2 vs 1). Neither condition ever answered a filler
problem outright or failed to find the question. C0's other
failures are the same kinds as B's (arithmetic and reasoning slips, loops) at
similar rates. That is stronger evidence that the reused cache is not the
cause than matched accuracy alone.

**Runaways** (hit the 8,192 cap), B vs C0: 3 vs 1, 5 vs 3, 7 vs 6, 7 vs 12.
At 16k, C0 ran away more: 9 C0-only against 4 B-only (McNemar p = 0.27). Not
significant, but in the same direction as the pilot's 5-vs-1 signal, and the
worst-case 16k cell is where it should show if it is real. Of the measured
risks, this is the one to test with more problems.

## Two operating regimes, not a typical case and a bound

| | Typical: burn mid-answer (`EXP1_DOSE.md`) | Worst: burn at token 0 (this) |
|---|---|---|
| Who wrote the recent context | the large model (the bridge) | nobody: generation starts after the switch |
| First-token shock, exact | 0.02–0.12, no trend in length | 0.24–0.56, grows to ~4k then saturates |
| C vs the matched reference | C − D: +0.04, 0.00, +0.02, +0.04 | C0 − B: +0.06, +0.04, +0.06, +0.14 |

**Real GEI operation lands in both.** In the usual case, memory binds deep
into a long generation. By then the large model has written thousands of
tokens and surfaced what the rest of the task needs. That is the typical
case.

But a prompt that nearly fills the memory envelope forces a burn
**immediately after prefill**, which is exactly C0. This is not a contrived
stress test: long-document question answering, large retrieved contexts and
long system prompts all produce it, and GEI's launch check would be choosing
to start large on exactly those tasks. So C0 is a **second operating regime**,
and the result here applies to it directly. In this regime on this task,
reusing the large model's prompt cache gave no measurable accuracy cost, and
fewer misreadings of a far-back question than the small model's own cache.

A third situation sits between the two: a long task that later depends on
something far back, such as an early instruction. It needs retrieval from deep
in a foreign cache even mid-answer. C0 tests that retrieval with no bridge,
which is why it bounds that case too.

## Checks and deviations from the pre-registration

- **Reused A and B reproduce exactly.** Rerun with the current code (including
  the last-position logits change): A matched the earlier run on 20 of 20
  problems, B on 40 of 40, by text and token count. The earlier run stored no
  token ids, so this check is text plus count, not bitwise.
- **D0 = B: 40 of 40,** identical token ids and bitwise-identical first-token
  log-probabilities.
- **Deviation:** the pre-registered "wrong question" rule misfires (above). Its
  counts are reported unchanged; the hand-checked reading, and the post-hoc
  "question misreading" category, are labelled as such.
- **The foreign-cache size here is the prompt alone** (88 tokens median at the
  anchor). In the typical case it was the prompt plus the generated prefix (184).
- Everything else ran as pre-registered.

## Limits

- One task (GSM8K with filler), one model, greedy decoding, one seed.
- The small configuration is the mix-and-match variant (layers 0–9 at full
  width), not the trained E2B; B is a weaker reference than a trained sub-model.
- 50 problems per cell: no harm detected, but deficits up to ~8 points at 4k
  are not excluded.

## Licenses

As in `docs/EXP1_DOSE.md`: GSM8K test and train splits (`openai/gsm8k` @
`740312add8`), MIT; Gemma 3n E4B-it, Gemma Terms of Use.

## Reproduce

    python scripts/exp1_worst.py            # 4 cells x 50 problems, ~2.5 h; needs exp1_dose trials for A and B
    python scripts/exp1_worst_report.py     # pre-registered analysis + examples for the hand check
