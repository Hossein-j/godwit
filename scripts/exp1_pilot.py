#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Experiment 1 pilot on Gemma 3n E4B: can the small configuration continue from
the large one's KV cache?

Pilot only: GSM8K test (first N problems), one switch point, one seed, greedy.

  A_large      large throughout
  B_small      small throughout (burned before the prompt)
  C_reuse      large, burn at the switch point, keep the cache      <- GEI
  D_recompute  same burn, rebuild the cache with the small model

The switch point is half of A's own generated length for that problem, so it
always lands mid-answer (8,192 tokens is only the cap; answers stop at
<end_of_turn>).

The small configuration is a mix-and-match variant: FFN width 16,384 -> 8,192
in layers 10-34, with layers 0-9 kept at full width because their activation
sparsity cannot be chunked. It is not the trained E2B (which also drops 5
layers), so B is a weaker reference than a trained sub-model would be.

Metrics per trial:
  correct          final answer equals the GSM8K gold answer
  mean_quality     mean log-probability the LARGE model assigns to the
                   condition's own output tokens (teacher-forced): drift from A
  switch_shock_kl  C and D only: mean KL(p_run || p_ref) over the 200 tokens
                   after the switch, where p_ref is the small model on the same
                   prefix with a freshly computed cache. For C it isolates the
                   effect of reusing the large model's cache; for D it is the
                   bf16 numerical noise floor (same model, cache built
                   incrementally vs in one pass).

Pre-flight on the first problems: (a) A through HFElasticModel vs plain HF
generate; (b) B vs a switch at token 0 with recompute. Both should give
identical tokens; mismatches are reported, not hidden.

HF's DynamicCache with SDPA attention throughout: this measures quality, not
memory, so the block cache and its kernel are not used.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_exp1 import TrialResult, summarise, verdict

from gei.elastic import HFElasticModel, KVPolicy
from gei.revisions import pinned
from gei.metrics import kl_divergence, paired_bootstrap_ci

MODEL = "google/gemma-3n-E4B-it"
DATASET, DATASET_REV = "openai/gsm8k", "740312add8"
INSTRUCTION = ("\nSolve the problem step by step. End with a final line of the form "
               "'The answer is N'.")
CONDITIONS = ("A_large", "B_small", "C_reuse", "D_recompute")


def load_problems(n: int) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "main/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=DATASET_REV)
    df = pd.read_parquet(path).head(n)
    return [{"id": i, "question": q, "gold": a.split("####")[-1].strip().replace(",", "")}
            for i, (q, a) in enumerate(zip(df.question, df.answer))]


_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def extract_answer(text: str) -> str | None:
    """'The answer is N' if present, else the last number in the text."""
    m = re.findall(r"answer is[:\s]*\**\s*(" + _NUM + ")", text, flags=re.IGNORECASE)
    nums = m or re.findall(_NUM, text)
    if not nums:
        return None
    return nums[-1].replace("$", "").replace(",", "").rstrip(".")


def is_correct(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def decode(m, prompt, stop, max_new, *, small=False, switch_at=None, policy=None,
           window=200):
    """Greedy generation; returns (tokens, logprob arrays for the shock window)."""
    m.reset()
    if small:
        m.switch_to(1, kv_policy=KVPolicy.RECOMPUTE)
    m.prefill(prompt)
    out, post = [], []
    for t in range(max_new):
        if switch_at is not None and t == switch_at:
            m.switch_to(1, kv_policy=policy)
        step = m.step()
        out.append(step.token_id)
        if switch_at is not None and switch_at <= t < switch_at + window:
            post.append(step.logprobs.astype(np.float32))
        if step.token_id in stop:
            break
    return out, post


@torch.inference_mode()
def teacher_forced(model, prompt, out, *, dist_from=None, dist_n=0, chunk=512):
    """Per-token log-probs of `out` given `prompt`, run in chunks with a cache so
    an 8k-token sequence never materialises 8k x vocab logits. Optionally also
    returns full log-prob distributions for generated positions
    [dist_from, dist_from + dist_n)."""
    from transformers import DynamicCache

    ids = prompt + out
    device = next(model.parameters()).device
    cache, lp_tokens, dists = DynamicCache(config=model.config), [], []
    for start in range(0, len(ids) - 1, chunk):
        end = min(start + chunk, len(ids) - 1)
        x = torch.tensor([ids[start:end]], device=device)
        logits = model(input_ids=x, past_key_values=cache, use_cache=True).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(ids[start + 1:end + 1], device=device)
        picked = lp.gather(1, targets[:, None])[:, 0].tolist()
        for pos in range(start, end):  # logits at pos predict ids[pos + 1]
            gen = pos + 1 - len(prompt)  # index of that token within `out`
            if gen >= 0:
                lp_tokens.append(picked[pos - start])
                if dist_from is not None and dist_from <= gen < dist_from + dist_n:
                    dists.append(lp[pos - start].cpu().numpy())
    return lp_tokens, dists


def shock_kl(run_lps, ref_lps) -> float | None:
    pairs = list(zip(run_lps, ref_lps))
    if not pairs:
        return None
    return float(np.mean([kl_divergence(np.exp(p), np.exp(q)) for p, q in pairs]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--switch-fraction", type=float, default=0.5)
    ap.add_argument("--preflight", type=int, default=5)
    ap.add_argument("--trials", default="results/exp1_pilot_trials.jsonl")
    ap.add_argument("--out", default="results/exp1_pilot.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Gemma3nForCausalLM

    problems = load_problems(args.n_problems)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=pinned(MODEL))
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["question"] + INSTRUCTION}],
                                       add_generation_prompt=True) for p in problems]

    model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
    model.requires_grad_(False).eval()

    # Pre-flight (a): plain HF generate on the unmodified model, before wrapping.
    trials_path = Path(args.trials)
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["problem_id"], r["condition"])] = r
    preflight = {"plain_generate": {}}
    pre_path = trials_path.with_suffix(".preflight.json")
    if pre_path.exists():
        preflight = json.loads(pre_path.read_text())
    else:
        model.to("cuda")
        for i in range(args.preflight):
            with torch.inference_mode():
                g = model.generate(torch.tensor([prompts[i]], device="cuda"),
                                   max_new_tokens=args.max_new, do_sample=False,
                                   eos_token_id=sorted(stop))[0, len(prompts[i]):].tolist()
            preflight["plain_generate"][str(i)] = g
        # Reload from disk (page-cached) instead of copying back from the GPU:
        # a second host copy of the weights does not fit in RAM beside the split.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = Gemma3nForCausalLM.from_pretrained(MODEL, revision=pinned(MODEL), dtype=torch.bfloat16)
        model.requires_grad_(False).eval()

    m = HFElasticModel(model, [16384, 8192], exclude_layers=range(10), restorable=True,
                       names=["E4B", "mixmatch_l0-9_full"], device="cuda")
    print(f"configs: {[(c.name, round(c.weight_bytes / 2**30, 3)) for c in m.configs]}; "
          f"burnable layers {m.burnable_layers[0]}-{m.burnable_layers[-1]}", flush=True)

    t_start = time.time()
    with trials_path.open("a") as log:
        for p, prompt in zip(problems, prompts):
            pid = p["id"]
            if all((pid, c) in done for c in CONDITIONS):
                continue
            # A first: it fixes this problem's switch point.
            out_a, _ = decode(m, prompt, stop, args.max_new)
            switch_at = max(1, int(len(out_a) * args.switch_fraction))
            runs = {"A_large": (out_a, None)}
            runs["B_small"] = (decode(m, prompt, stop, args.max_new, small=True)[0], None)
            for cond, policy in (("C_reuse", KVPolicy.REUSE), ("D_recompute", KVPolicy.RECOMPUTE)):
                out, post = decode(m, prompt, stop, args.max_new, switch_at=switch_at,
                                   policy=policy)
                # Reference: small model (resident now), same prefix, fresh cache.
                _, ref = teacher_forced(m.model, prompt, out, dist_from=switch_at,
                                        dist_n=len(post))
                runs[cond] = (out, shock_kl(post, ref))
            if pid < args.preflight:
                d0 = decode(m, prompt, stop, args.max_new, switch_at=0,
                            policy=KVPolicy.RECOMPUTE)[0]
                preflight.setdefault("b_equals_d0", {})[str(pid)] = d0 == runs["B_small"][0]
                preflight.setdefault("a_equals_plain", {})[str(pid)] = (
                    out_a == preflight["plain_generate"][str(pid)])
                pre_path.write_text(json.dumps(preflight))
            m.reset()  # large again, for the quality pass
            for cond in CONDITIONS:
                out, shock = runs[cond]
                lps, _ = teacher_forced(m.model, prompt, out)
                text = tok.decode(out, skip_special_tokens=True)
                rec = asdict(TrialResult(condition=cond, problem_id=pid, seed=0,
                                         switch_point=args.switch_fraction,
                                         correct=is_correct(text, p["gold"]),
                                         mean_quality=float(np.mean(lps)) if lps else 0.0,
                                         switch_shock_kl=shock))
                rec |= {"tokens": len(out), "switch_at": switch_at if cond[0] in "CD" else None,
                        "pred": extract_answer(text), "gold": p["gold"],
                        "stopped": out[-1] in stop, "text": text}
                log.write(json.dumps(rec) + "\n")
                done[(pid, cond)] = rec
            log.flush()
            el = time.time() - t_start
            accs = {c: np.mean([done[(i, c)]["correct"] for i in range(pid + 1) if (i, c) in done])
                    for c in CONDITIONS}
            print(f"problem {pid}: len_A {len(out_a)}, switch at {switch_at}; running acc "
                  + " ".join(f"{c[0]}={a:.2f}" for c, a in accs.items())
                  + f" | {el / 60:.1f} min elapsed", flush=True)

    rows = [done[(p["id"], c)] for p in problems for c in CONDITIONS if (p["id"], c) in done]
    trials = [TrialResult(**{k: r[k] for k in TrialResult.__dataclass_fields__}) for r in rows]
    summary = summarise(trials)
    by = {c: [float(done[(p["id"], c)]["correct"]) for p in problems] for c in CONDITIONS}
    paired = {f"{x}-{y}": paired_bootstrap_ci(by[x], by[y])
              for x, y in (("C_reuse", "B_small"), ("D_recompute", "B_small"),
                           ("C_reuse", "D_recompute"), ("A_large", "B_small"))}
    result = {"model": MODEL, "dataset": f"{DATASET}@{DATASET_REV}", "args": vars(args),
              "summary": summary, "verdict_interval_overlap": verdict(summary),
              "paired_accuracy_diff_ci95": paired, "preflight": {
                  k: v for k, v in preflight.items() if k != "plain_generate"},
              "configs": [(c.name, c.weight_bytes) for c in m.configs]}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("summary", "verdict_interval_overlap",
                                             "paired_accuracy_diff_ci95", "preflight")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
