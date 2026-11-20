#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""GEI against KV eviction at equal memory: the eviction conditions.

Design and hypotheses: docs/PREREG.md (frozen; SHA-256 in docs/PREREG.sha256).

The large configuration decodes every token (no burn) through the same
HFElasticModel wrapper and decoding loop as the paper's A runs, with its
growing full-attention cache held to `--budget` tokens by the eviction policy.
Prompts are rebuilt exactly as in the paper's worst-case run (same builder,
same seeds), so every trial pairs with a stored A, B and C0 record.

Modes:
  --policy stream|snap --cells 1k 4k 16k     the registered runs
  --check-zero                               exact-zero check: both policies
                                             with an unlimited budget on the
                                             anchor cell, problems 0-9, must
                                             reproduce the stored A tokens

Reads the repository code and stored results (../src, ../scripts, ../results)
without modifying them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GODWIT = HERE.parents[1]  # repository root
sys.path[:0] = [str(HERE), str(GODWIT / "scripts"), str(GODWIT / "src")]

from evict import Evictor  # noqa: E402

BUDGET = 1024  # docs/PREREG.md: the large model's per-sequence cache under the envelope
DOSE_TRIALS = GODWIT / "results" / "exp1_dose_trials.jsonl"
WORST_TRIALS = GODWIT / "results" / "exp1_worst_trials.jsonl"


def question_span(tok, question: str, prompt: list[int]) -> tuple[int, int]:
    """Token positions [start, end) of the question text in the prompt, by
    longest common prefix with the templated prompt cut before and after it
    (exact up to one token at either boundary)."""
    def lcp(content):
        ids = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True)
        n = 0
        while n < min(len(ids), len(prompt)) and ids[n] == prompt[n]:
            n += 1
        return n
    return lcp("PROBLEM:"), lcp("PROBLEM: " + question)


def decode_evict(m, prompt, stop, max_new, ev: Evictor, q_span=None):
    """The paper's greedy decode (exp1_pilot.decode, large model throughout)
    with eviction applied to m's cache after prefill and after every step."""
    m.reset()
    ev.attach(m.model)
    m.prefill(prompt)
    ev.after_prefill(m._cache)
    kept_q = ev.kept_fraction(*q_span) if q_span else None
    out = []
    for _ in range(max_new):
        step = m.step()
        out.append(step.token_id)
        ev.after_step(m._cache, position=len(m._rows[0]) - 2)
        if step.token_id in stop:
            break
    return out, kept_q


def load_stored():
    dose = {(r["cell"], r["problem_id"], r["condition"]): r
            for r in map(json.loads, DOSE_TRIALS.read_text().splitlines())}
    worst = {(r["cell"], r["problem_id"]): r
             for r in map(json.loads, WORST_TRIALS.read_text().splitlines())}
    return dose, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", choices=["stream", "snap"])
    ap.add_argument("--cells", nargs="+", default=["1k", "4k", "16k"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--check-zero", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from exp1_dose import CELLS, build_prompt, load_filler, looped
    from exp1_pilot import MODEL, extract_answer, is_correct, load_problems
    from gei.elastic import HFElasticModel
    from gei.revisions import pinned
    from transformers import AutoTokenizer, Gemma3nForCausalLM

    if args.check_zero:
        runs = [(pol, "anchor", 10**9) for pol in ("stream", "snap")]
        args.start, args.end = 0, 10
        out_path = Path(args.out or HERE.parent / "results" / "check_zero.jsonl")
    else:
        if not args.policy:
            ap.error("--policy is required unless --check-zero")
        runs = [(args.policy, c, args.budget) for c in args.cells]
        out_path = Path(args.out or HERE.parent / "results" / f"evict_{args.policy}_trials.jsonl")

    problems = load_problems(args.end)
    filler = load_filler()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    dose, worst = load_stored()
    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), restorable=True,
                       names=["E4B", "mixmatch_l0-9_full"], device="cuda")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(r["policy"], r["cell"], r["problem_id"])
                for r in map(json.loads, out_path.read_text().splitlines())}
    t0 = time.time()
    with out_path.open("a") as log:
        for policy, cell, budget in runs:
            for p in problems[args.start:args.end]:
                pid = p["id"]
                if (policy, cell, pid) in done:
                    continue
                t_p = time.time()
                rng = random.Random(1000 * pid + CELLS[cell])
                prompt = build_prompt(tok, p["question"], CELLS[cell], filler, rng)
                span = question_span(tok, p["question"], prompt) if CELLS[cell] else None
                ev = Evictor(model.config.get_text_config(), policy, budget)
                out, kept_q = decode_evict(m, prompt, stop, args.max_new, ev, span)
                text = tok.decode(out, skip_special_tokens=True)
                rec = {"policy": policy, "cell": cell, "problem_id": pid, "budget": budget,
                       "prompt_tokens": len(prompt), "question_span": span,
                       "question_kept_fraction": kept_q, "evicted_entries": ev.evictions,
                       "tokens": len(out), "correct": is_correct(text, p["gold"]),
                       "pred": extract_answer(text), "gold": p["gold"],
                       "runaway": out[-1] not in stop, "loop": looped(out), "text": text}
                stored_a = dose.get((cell, pid, "A_large"))
                if stored_a is not None:
                    rec["same_as_stored_A"] = (text == stored_a["text"]
                                               and len(out) == stored_a["tokens"])
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"{policy:6s} {cell:6s} problem {pid:2d}: prompt {len(prompt):5d}, "
                      f"{'ok  ' if rec['correct'] else 'MISS'} {len(out):4d} tok, q kept "
                      f"{'-' if kept_q is None else f'{kept_q:.2f}'}, same as A "
                      f"{rec.get('same_as_stored_A')} | {time.time() - t_p:.0f}s, "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
