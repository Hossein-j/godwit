# Pre-registration: switch at token 0 (worst case for cache reuse)

Written and committed **before** any C0 or D0 data was generated
(2026-10-26). The analysis follows these rules; any deviation will be
reported as such.

## Question

Can the small configuration answer correctly when the large configuration
wrote the entire prompt cache and the small one generates every token? This
is the worst case for cache reuse: no bridge, since the large model has
written no part of the answer. It is also a deployment regime, not only a
bound: a prompt that nearly fills the memory envelope forces a burn
immediately after prefill, which is exactly C0.

## Conditions, per foreign-cache size (anchor, 1k, 4k, 16k; problems 0-49)

- **A** large throughout, and **B** small throughout: reused from
  `exp1_dose.py` after a spot-check (below).
- **C0** large prefill, switch at token 0, REUSE the cache.
- **D0** large prefill, switch at token 0, RECOMPUTE: should equal B by
  construction. Run on problems 0-9 per size as a check on the recompute path.

The comparison that matters is **C0 vs B**: the same model generates every
token; only the author of the prompt cache differs.

## Checks before the results are trusted

1. **Reuse of A and B:** rerun A and B on problems 0-4 per size with the
   current code and compare with the earlier run (text and token count; the
   earlier run did not store token ids). Any mismatch means rerunning A and B
   in full.
2. **D0 = B:** on problems 0-9 per size, compare D0's and a fresh B's token
   ids and first-token log-probabilities. If identical, the D0-vs-B
   first-token KL is exactly 0 and is reported as such. Any mismatch is
   investigated before C0 is interpreted.

## Measures

- **Primary: first-token shock KL, KL(C0 ‖ B).** At token 0 both condition on
  the identical prompt, so this needs no teacher forcing. B's first-token
  distribution comes from B's own code path (prefill + one step).
- Secondary: mean KL over the 200 tokens after the switch, against the small
  model teacher-forced on C0's own prefix (as in the earlier runs).
- Accuracy per condition with bootstrap CIs; paired C0 − B difference with a
  paired bootstrap CI; **McNemar's exact test** on the C0-vs-B discordant
  pairs; conditional accuracy P(C0 correct | B correct) and the reverse.
- Runaway rate (hit the 8,192-token cap) and loop rate (a 16-token sequence
  at least 4 times in the last 512 tokens).

## Failure classification (every C0 miss and every B miss)

Assigned by fixed rules, in this order:

1. **runaway/loop**: hit the cap, or the loop criterion holds.
2. **wrong question**: the predicted answer equals the gold answer of one of
   the filler examples in that prompt and not the target's, or the response
   refers to "Example N".
3. **no question found**: the response says no problem was given or asks
   which problem to solve (matches: "no problem", "not provided", "didn't
   provide", "did not provide", "which problem", "please provide", "there is no
   problem").
4. **reasoning slip**: none of the above; the right question, answered wrong.

A sample of each class is read by hand to check the rules; disagreements are
reported. If C0's misses fall into the same classes as B's, at similar rates,
that is evidence the reused cache is not the cause.

## Power and the extension rule (fixed now)

At 50 problems, discordant pairs come from a small pool, especially where B is
weak.

- **B too weak:** a cell's C0-vs-B accuracy comparison is flagged as not
  meaningful if B solves **fewer than 20 of 50**. KL and failure types are
  still reported for it.
- **Extension:** an unflagged cell is extended from 50 to 100 problems
  (adding problems 50-99, with B and C0 only) if **C0 accuracy < B accuracy
  and McNemar's exact two-sided p < 0.5**. Flagged cells are not extended.
  No other cell is extended, whatever the data show.
