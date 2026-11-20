#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Check 1 of docs/PREREG.md: does the reinstalled environment reproduce the
paper's stored records?

Reruns A (large throughout) and C0 (burn at token 0, reuse the cache) on
problems 0-4 of the 1k, 4k and 16k cells with the paper's own decode loop, and
compares text and token count with ../results. Any mismatch means A, B
and C0 must be rerun in full before the eviction runs are compared with them.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GODWIT = HERE.parents[1]  # repository root
sys.path[:0] = [str(HERE), str(GODWIT / "scripts"), str(GODWIT / "src")]


def main() -> int:
    from exp1_dose import CELLS, build_prompt, load_filler
    from exp1_pilot import MODEL, decode, load_problems
    from gei.elastic import HFElasticModel, KVPolicy
    from gei.revisions import pinned
    from run_eviction import load_stored
    from transformers import AutoTokenizer, Gemma3nForCausalLM

    problems = load_problems(5)
    filler = load_filler()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    dose, worst = load_stored()
    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), restorable=True,
                       names=["E4B", "mixmatch_l0-9_full"], device="cuda")

    out_path = HERE.parent / "results" / "repro_check.jsonl"
    n_ok = n_all = 0
    with out_path.open("w") as log:
        for cell in ("1k", "4k", "16k"):
            for p in problems:
                pid, t = p["id"], time.time()
                prompt = build_prompt(tok, p["question"], CELLS[cell], filler,
                                      random.Random(1000 * pid + CELLS[cell]))
                a, _ = decode(m, prompt, stop, 8192)
                c0, _ = decode(m, prompt, stop, 8192, switch_at=0, policy=KVPolicy.REUSE)
                old_a, old_c0 = dose[(cell, pid, "A_large")], worst[(cell, pid)]["C0"]
                res = {"cell": cell, "problem_id": pid, "prompt_tokens": len(prompt),
                       "prompt_tokens_stored": worst[(cell, pid)]["prompt_tokens"]}
                for name, out, old in (("A", a, old_a), ("C0", c0, old_c0)):
                    text = tok.decode(out, skip_special_tokens=True)
                    res[name] = {"same_text": text == old["text"],
                                 "same_tokens": len(out) == old["tokens"],
                                 "tokens": len(out), "tokens_stored": old["tokens"]}
                    n_all += 1
                    n_ok += res[name]["same_text"] and res[name]["same_tokens"]
                log.write(json.dumps(res) + "\n")
                log.flush()
                print(f"{cell:4s} problem {pid}: prompt {len(prompt)} (stored "
                      f"{res['prompt_tokens_stored']}), A same {res['A']['same_text']}, "
                      f"C0 same {res['C0']['same_text']} | {time.time() - t:.0f}s", flush=True)
    print(f"reproduced {n_ok}/{n_all} records exactly")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
