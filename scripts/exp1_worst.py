#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Switch at token 0: the worst case for cache reuse (Gemma 3n E4B).

Pre-registered in docs/EXP1_WORST_PREREG.md. The large configuration writes
the whole prompt cache, then the small configuration generates every token:
  C0  switch at token 0, REUSE the cache
  D0  switch at token 0, RECOMPUTE (should equal B; checked on problems 0-9)
A and B are reused from exp1_dose.py for problems 0-49, after a spot-check on
problems 0-4 (rerun with the current code, compared by text and token count).

Primary measure: first-token KL(C0 || B). Both condition on the identical
prompt, so no teacher forcing: B's first-token distribution comes from B's own
code path (small model, prefill, one step).

Prompts are rebuilt with exp1_dose.build_prompt and the same seeds, so they are
the same token ids as the earlier run; the indices and answers of the filler
examples are recorded for classifying "wrong question" failures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_dose import CELLS, build_prompt, load_filler, looped, per_position_kl
from exp1_pilot import (
    MODEL,
    decode,
    extract_answer,
    is_correct,
    load_problems,
    teacher_forced,
)

from gei.elastic import HFElasticModel, KVPolicy
from gei.revisions import pinned
from gei.metrics import kl_divergence


def filler_answer(example: str) -> str:
    return example.rsplit("The answer is", 1)[1].strip().rstrip(".").replace(",", "")


def first_step_small(m, prompt):
    """B's exact first step: small model, its own prefill, one token."""
    m.reset()
    m.switch_to(1, kv_policy=KVPolicy.RECOMPUTE)
    m.prefill(prompt)
    return m.step().logprobs.astype(np.float32)


def summary_of(out, text, gold, stop):
    return {"tokens": len(out), "correct": is_correct(text, gold), "pred": extract_answer(text),
            "runaway": out[-1] not in stop, "loop": looped(out), "text": text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--dose-trials", default="results/exp1_dose_trials.jsonl")
    ap.add_argument("--trials", default="results/exp1_worst_trials.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Gemma3nForCausalLM

    problems = load_problems(args.end)
    filler = load_filler()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    dose = {(r["cell"], r["problem_id"], r["condition"]): r
            for r in map(json.loads, Path(args.dose_trials).read_text().splitlines())}
    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), restorable=True,
                       names=["E4B", "mixmatch_l0-9_full"], device="cuda")

    path = Path(args.trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["cell"], r["problem_id"]) for r in map(json.loads, path.read_text().splitlines())}
    t0 = time.time()
    with path.open("a") as log:
        for cell in args.cells:
            for p in problems[args.start:args.end]:
                pid = p["id"]
                if (cell, pid) in done:
                    continue
                t_p = time.time()
                rng = random.Random(1000 * pid + CELLS[cell])
                prompt, used = build_prompt(tok, p["question"], CELLS[cell], filler, rng,
                                            return_examples=True)
                rec = {"cell": cell, "problem_id": pid, "prompt_tokens": len(prompt),
                       "foreign_tokens": len(prompt) - 1, "gold": p["gold"],
                       "filler_answers": sorted({filler_answer(filler[i]) for i in used})}

                b_first = first_step_small(m, prompt)
                out, post = decode(m, prompt, stop, args.max_new, switch_at=0,
                                   policy=KVPolicy.REUSE)
                _, ref = teacher_forced(m.model, prompt, out, dist_from=0, dist_n=len(post))
                text = tok.decode(out, skip_special_tokens=True)
                rec["C0"] = summary_of(out, text, p["gold"], stop) | {
                    "kl_first": kl_divergence(np.exp(post[0]), np.exp(b_first)),
                    "kl_by_offset": per_position_kl(post, ref)}

                # B: reused from the earlier run where it exists, fresh otherwise.
                old_b = dose.get((cell, pid, "B_small"))
                if pid < 10 or old_b is None:
                    b_out, _ = decode(m, prompt, stop, args.max_new, small=True)
                    b_text = tok.decode(b_out, skip_special_tokens=True)
                    rec["B"] = summary_of(b_out, b_text, p["gold"], stop) | {"source": "fresh"}
                    if old_b is not None:
                        rec["check_B_matches_earlier"] = (b_text == old_b["text"]
                                                          and len(b_out) == old_b["tokens"])
                else:
                    rec["B"] = {k: old_b[k] for k in ("tokens", "correct", "pred", "runaway",
                                                      "loop", "text")} | {"source": "exp1_dose"}

                if pid < 10:  # D0 = B check, token ids and first-step log-probs
                    d0_out, d0_post = decode(m, prompt, stop, args.max_new, switch_at=0,
                                             policy=KVPolicy.RECOMPUTE)
                    rec["check_D0"] = {
                        "tokens_equal_B": d0_out == b_out,
                        "first_logprobs_equal_B": bool(np.array_equal(d0_post[0], b_first)),
                        "kl_first_D0_B": kl_divergence(np.exp(d0_post[0]), np.exp(b_first))}
                if pid < 5 and (cell, pid, "A_large") in dose:  # spot-check reuse of A
                    a_out, _ = decode(m, prompt, stop, args.max_new)
                    a_text = tok.decode(a_out, skip_special_tokens=True)
                    old_a = dose[(cell, pid, "A_large")]
                    rec["check_A_matches_earlier"] = (a_text == old_a["text"]
                                                      and len(a_out) == old_a["tokens"])
                old_a = dose.get((cell, pid, "A_large"))
                rec["A"] = ({k: old_a[k] for k in ("tokens", "correct", "runaway", "loop")}
                            if old_a else None)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"{cell:6s} problem {pid:2d}: prompt {len(prompt):5d}, C0 "
                      f"{'ok ' if rec['C0']['correct'] else 'MISS'} B "
                      f"{'ok ' if rec['B']['correct'] else 'MISS'}, KL first "
                      f"{rec['C0']['kl_first']:.4f}"
                      + (f", D0=B {rec['check_D0']['tokens_equal_B']}" if pid < 10 else "")
                      + f" | {time.time() - t_p:.0f}s, {(time.time() - t0) / 60:.0f} min",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
