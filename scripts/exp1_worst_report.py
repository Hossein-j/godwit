#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Analyse exp1_worst.py trials by the rules in docs/EXP1_WORST_PREREG.md."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from gei.metrics import bootstrap_ci, mcnemar_exact, paired_bootstrap_ci

NO_QUESTION = ("no problem", "not provided", "didn't provide", "did not provide",
               "which problem", "please provide", "there is no problem")
WEAK_B = 20  # B solving fewer than this many of 50 flags the accuracy comparison


def same_number(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def classify(run: dict, gold: str, filler_answers: list[str]) -> str | None:
    """Pre-registered failure class for a miss; None if the run is correct."""
    if run["correct"]:
        return None
    if run["runaway"] or run["loop"]:
        return "runaway/loop"
    pred = run["pred"]
    if (any(same_number(pred, f) for f in filler_answers) and not same_number(pred, gold)) \
            or re.search(r"\bExample\s+\d+", run["text"]):
        return "wrong question"
    low = run["text"].lower()
    if any(s in low for s in NO_QUESTION):
        return "no question found"
    return "reasoning slip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default="results/exp1_worst_trials.jsonl")
    ap.add_argument("--out", default="results/exp1_worst_summary.json")
    ap.add_argument("--examples", type=int, default=2, help="examples printed per class")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.trials).read_text().splitlines()]
    out = {}
    for cell in dict.fromkeys(r["cell"] for r in rows):
        rs = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["problem_id"])
        b = [r["B"]["correct"] for r in rs]
        c0 = [r["C0"]["correct"] for r in rs]
        a = [r["A"]["correct"] for r in rs if r["A"]]
        kl_first = [r["C0"]["kl_first"] for r in rs]
        kl_win = [float(np.mean(r["C0"]["kl_by_offset"])) for r in rs if r["C0"]["kl_by_offset"]]
        offs = [r["C0"]["kl_by_offset"] for r in rs]
        bins = {f"{lo}-{hi}": float(np.mean([np.mean(o[lo:hi]) for o in offs if len(o) > lo]))
                for lo, hi in ((0, 1), (1, 10), (10, 50), (50, 200))}
        checks = [r for r in rs if "check_D0" in r]
        classes = {who: Counter(c for r in rs if (c := classify(r[who], r["gold"], r["filler_answers"])))
                   for who in ("B", "C0")}
        mc = mcnemar_exact(c0, b)
        n_b = int(sum(b))
        weak = n_b < WEAK_B * len(rs) / 50
        extend = (not weak) and np.mean(c0) < np.mean(b) and mc["p"] < 0.5
        out[cell] = {
            "n": len(rs), "foreign_tokens_median": float(np.median([r["foreign_tokens"] for r in rs])),
            "accuracy": {"A": [float(np.mean(a)), *bootstrap_ci(a)] if a else None,
                         "B": [float(np.mean(b)), *bootstrap_ci(b)],
                         "C0": [float(np.mean(c0)), *bootstrap_ci(c0)]},
            "paired_C0_minus_B": list(paired_bootstrap_ci(c0, b)),
            "mcnemar_C0_vs_B": mc,
            "conditional": {"P(C0 | B correct)": float(np.mean([x for x, y in zip(c0, b) if y])) if n_b else None,
                            "P(B | C0 correct)": float(np.mean([y for x, y in zip(c0, b) if x])) if any(c0) else None},
            "kl_first": {"mean": float(np.mean(kl_first)), "median": float(np.median(kl_first)),
                         "ci95": list(bootstrap_ci(kl_first)), "p90": float(np.quantile(kl_first, 0.9))},
            "kl_window_mean": float(np.mean(kl_win)), "kl_by_offset": bins,
            "D0_check": {"n": len(checks),
                         "tokens_equal_B": sum(r["check_D0"]["tokens_equal_B"] for r in checks),
                         "first_logprobs_bitwise_equal_B": sum(r["check_D0"]["first_logprobs_equal_B"] for r in checks),
                         "max_abs_kl_routine": max((abs(r["check_D0"]["kl_first_D0_B"]) for r in checks), default=None)},
            "reuse_spot_check": {
                "A": [r["check_A_matches_earlier"] for r in rs if "check_A_matches_earlier" in r],
                "B": [r["check_B_matches_earlier"] for r in rs if "check_B_matches_earlier" in r]},
            "runaway": {w: int(sum(r[w]["runaway"] for r in rs)) for w in ("B", "C0")},
            "loop": {w: int(sum(r[w]["loop"] for r in rs)) for w in ("B", "C0")},
            "failure_classes": {w: dict(classes[w]) for w in classes},
            "B_too_weak": bool(weak), "extend_to_100": bool(extend),
        }
    Path(args.out).write_text(json.dumps(out, indent=2))

    for cell, s in out.items():
        acc = s["accuracy"]
        print(f"\n== {cell}: n {s['n']}, foreign {s['foreign_tokens_median']:.0f} tokens"
              f"{'  [B TOO WEAK for accuracy comparison]' if s['B_too_weak'] else ''}"
              f"{'  [EXTEND to 100 by the pre-registered rule]' if s['extend_to_100'] else ''}")
        print(f"   accuracy A {acc['A'][0] if acc['A'] else float('nan'):.2f}  B {acc['B'][0]:.2f} "
              f"[{acc['B'][1]:.2f},{acc['B'][2]:.2f}]  C0 {acc['C0'][0]:.2f} [{acc['C0'][1]:.2f},{acc['C0'][2]:.2f}]")
        d = s["paired_C0_minus_B"]
        mc = s["mcnemar_C0_vs_B"]
        print(f"   C0 - B {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]; McNemar C0-only {mc['a_only']} vs "
              f"B-only {mc['b_only']}, p = {mc['p']:.3f}; conditional {s['conditional']}")
        k = s["kl_first"]
        print(f"   first-token KL(C0||B) mean {k['mean']:.4f} [{k['ci95'][0]:.4f},{k['ci95'][1]:.4f}] "
              f"median {k['median']:.4f} p90 {k['p90']:.4f}; 200-token window {s['kl_window_mean']:.4f}; "
              f"by offset {{{', '.join(f'{a}: {v:.4f}' for a, v in s['kl_by_offset'].items())}}}")
        print(f"   D0 check {s['D0_check']}; reuse spot-check {s['reuse_spot_check']}")
        print(f"   runaway {s['runaway']} loop {s['loop']}; failure classes {s['failure_classes']}")

    if args.examples:
        print("\n-- examples per failure class (for the hand check) --")
        for who in ("C0", "B"):
            seen = Counter()
            for r in rows:
                c = classify(r[who], r["gold"], r["filler_answers"])
                if c and seen[c] < args.examples:
                    seen[c] += 1
                    t = r[who]["text"]
                    print(f"[{who} | {c} | {r['cell']} #{r['problem_id']} gold {r['gold']} pred {r[who]['pred']}] "
                          f"{t[:220]!r} ... {t[-160:]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
