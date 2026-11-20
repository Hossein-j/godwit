#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Switch shock as a function of foreign-cache length (Gemma 3n E4B).

The Experiment 1 pilot only put ~100-200 tokens of the large model's cache in
front of the small model. GEI operates at tens of thousands. This varies the
foreign-cache size directly, by prompt length, instead of hoping generations
run long.

Task: a GSM8K test question at the very top of the prompt, then N worked
solutions to unrelated GSM8K *train* problems as filler, then "now solve the
problem stated at the very top". The question is not repeated, so the small
model must read it from the far end of a cache the large model wrote. The
"anchor" cell uses the pilot's plain prompt unchanged (no filler).

Cells: foreign-cache size in {anchor ~100, 1k, 4k, 16k}; the same problems in
every cell. Per problem: A (large; fixes the switch point at half of A's own
length, as in the pilot), B (small throughout), C (burn at the switch, reuse
the cache), D (burn at the switch, recompute it). Foreign-cache size = tokens
in the cache at the switch (prompt minus the pending token, plus the generated
prefix), recorded per trial.

Per trial: correct; runaway (hit the token cap); loop (some 16-token n-gram
occurs >= 4 times in the last 512 tokens); for C and D the switch-shock KL at
each of the 200 positions after the switch, against the small model on the
same prefix with a freshly computed cache (as in the pilot; D's value is the
bf16 noise floor).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_pilot import (
    DATASET,
    DATASET_REV,
    INSTRUCTION,
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

CELLS = {"anchor": 0, "1k": 1024, "4k": 4096, "16k": 16384}  # target foreign tokens
TOP = ("PROBLEM: {q}\n\nBelow are worked solutions to other, unrelated problems. They "
       "are for reference only; do not answer them.\n\n")
BOTTOM = ("\n\nNow solve the PROBLEM stated at the very top (not the examples). Solve it "
          "step by step. End with a final line of the form 'The answer is N'.")


def load_filler() -> list[str]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "main/train-00000-of-00001.parquet",
                           repo_type="dataset", revision=DATASET_REV)
    df = pd.read_parquet(path)
    out = []
    for q, a in zip(df.question, df.answer):
        body, final = a.rsplit("####", 1)
        body = re.sub(r"<<.*?>>", "", body).strip()  # drop calculator annotations
        out.append(f"Q: {q}\nA: {body}\nThe answer is {final.strip()}.")
    return out


def build_prompt(tok, question, target, filler, rng, *, return_examples=False):
    """Chat-templated ids; for target > 0, add whole filler examples until the
    prompt reaches about `target` tokens (the generated prefix adds the rest).
    With return_examples, also return the indices of the filler used."""
    if target == 0:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": question + INSTRUCTION}], add_generation_prompt=True)
        return (ids, []) if return_examples else ids
    examples, budget = [], target - 100  # ~100 tokens of answer precede the switch
    order = rng.sample(range(len(filler)), len(filler))
    for i in order:
        examples.append(f"Example {len(examples) + 1}:\n{filler[i]}")
        if len(examples) % 8 == 0:
            text = TOP.format(q=question) + "\n\n".join(examples) + BOTTOM
            ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                          add_generation_prompt=True)
            if len(ids) >= budget:
                break
    while True:  # trim back to the budget one example at a time
        text = TOP.format(q=question) + "\n\n".join(examples) + BOTTOM
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True)
        if len(ids) <= budget or len(examples) == 1:
            used = order[:len(examples)]  # examples are taken in shuffled order
            return (ids, used) if return_examples else ids
        examples.pop()


def looped(out, n=16, times=4, tail=512) -> bool:
    t = out[-tail:]
    grams = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    return bool(grams) and max(grams.values()) >= times


def per_position_kl(run, ref):
    return [kl_divergence(np.exp(p), np.exp(q)) for p, q in zip(run, ref)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--trials", default="results/exp1_dose_trials.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Gemma3nForCausalLM

    problems = load_problems(args.n_problems)
    filler = load_filler()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), restorable=True,
                       names=["E4B", "mixmatch_l0-9_full"], device="cuda")

    path = Path(args.trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["cell"], r["problem_id"]) for r in map(json.loads, path.read_text().splitlines())
                if r["condition"] == "D_recompute"}
    t0 = time.time()
    with path.open("a") as log:
        for cell in args.cells:
            for p in problems:
                if (cell, p["id"]) in done:
                    continue
                rng = random.Random(1000 * p["id"] + CELLS[cell])
                prompt = build_prompt(tok, p["question"], CELLS[cell], filler, rng)
                t_p = time.time()
                out_a, _ = decode(m, prompt, stop, args.max_new)
                switch_at = max(1, len(out_a) // 2)
                foreign = len(prompt) - 1 + switch_at
                runs = {"A_large": (out_a, None),
                        "B_small": (decode(m, prompt, stop, args.max_new, small=True)[0], None)}
                for cond, policy in (("C_reuse", KVPolicy.REUSE), ("D_recompute", KVPolicy.RECOMPUTE)):
                    out, post = decode(m, prompt, stop, args.max_new, switch_at=switch_at,
                                       policy=policy)
                    _, ref = teacher_forced(m.model, prompt, out, dist_from=switch_at,
                                            dist_n=len(post))
                    runs[cond] = (out, per_position_kl(post, ref))
                for cond, (out, kls) in runs.items():
                    text = tok.decode(out, skip_special_tokens=True)
                    rec = {"cell": cell, "condition": cond, "problem_id": p["id"],
                           "prompt_tokens": len(prompt), "switch_at": switch_at,
                           "foreign_tokens": foreign if cond[0] in "CD" else None,
                           "tokens": len(out), "correct": is_correct(text, p["gold"]),
                           "pred": extract_answer(text), "gold": p["gold"],
                           "runaway": out[-1] not in stop, "loop": looped(out),
                           "kl_mean": float(np.mean(kls)) if kls else None,
                           "kl_by_offset": kls, "text": text}
                    log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"{cell:6s} problem {p['id']:2d}: prompt {len(prompt):5d}, foreign "
                      f"{foreign:5d}, A {len(out_a)} tok, KL C {np.mean(runs['C_reuse'][1]):.4f} "
                      f"D {np.mean(runs['D_recompute'][1]):.5f} | {time.time() - t_p:.0f}s, "
                      f"{(time.time() - t0) / 60:.0f} min total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
