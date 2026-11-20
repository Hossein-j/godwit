#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Analysis of the eviction comparison by the rules in docs/PREREG.md.

Per cell (1k, 4k, 16k): accuracy of A, B, C0 (stored paper records) and
E-stream, E-snap (results/evict_*_trials.jsonl) with bootstrap 95% CIs; paired
C0 - E-stream and C0 - E-snap with paired bootstrap CIs and McNemar's exact
test (H1: C0 > E-stream at 4k and 16k; H2: C0 vs E-snap, two-sided); secondary
A - E-snap, runaway and loop counts, question-kept fraction, and failure
classes with the worst-case run's classifier. No multiplicity correction.

Run: .venv/bin/python scripts/report.py   (works on partial data)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GODWIT = HERE.parents[1]  # repository root
sys.path[:0] = [str(HERE), str(GODWIT / "scripts"), str(GODWIT / "src")]

from exp1_worst_report import WEAK_B, classify  # noqa: E402
from gei.metrics import bootstrap_ci, mcnemar_exact, paired_bootstrap_ci  # noqa: E402

RESULTS = HERE.parent / "results"
CELLS = ("1k", "4k", "16k")
POLICIES = {"E-stream": "stream", "E-snap": "snap"}


def acc(xs):
    return [float(np.mean(xs)), *map(float, bootstrap_ci(xs))] if xs else None


def paired(a, b):
    d = list(map(float, paired_bootstrap_ci(a, b)))
    mc = mcnemar_exact(a, b)
    return {"diff": d, "mcnemar": mc}


def main() -> int:
    worst = {(r["cell"], r["problem_id"]): r
             for r in map(json.loads, (GODWIT / "results" / "exp1_worst_trials.jsonl").read_text().splitlines())}
    ev = {}
    for name, pol in POLICIES.items():
        f = RESULTS / f"evict_{pol}_trials.jsonl"
        if f.exists():
            for r in map(json.loads, f.read_text().splitlines()):
                ev[(name, r["cell"], r["problem_id"])] = r

    out = {}
    for cell in CELLS:
        s = {}
        for name in POLICIES:
            pids = sorted(pid for (n, c, pid) in ev if n == name and c == cell)
            if not pids:
                continue
            e = [ev[(name, cell, p)] for p in pids]
            w = [worst[(cell, p)] for p in pids]
            ec = [r["correct"] for r in e]
            c0 = [r["C0"]["correct"] for r in w]
            a = [r["A"]["correct"] for r in w]
            b = [r["B"]["correct"] for r in w]
            qk = [r["question_kept_fraction"] for r in e if r["question_kept_fraction"] is not None]
            s[name] = {
                "n": len(pids),
                "accuracy": {"A": acc(a), "B": acc(b), "C0": acc(c0), name: acc(ec)},
                f"C0_minus_{name}": paired(c0, ec),
                f"A_minus_{name}": paired(a, ec),
                "B_too_weak": sum(b) < WEAK_B * len(b) / 50,
                "runaway": sum(r["runaway"] for r in e), "loop": sum(r["loop"] for r in e),
                "evicted_entries_median": float(np.median([r["evicted_entries"] for r in e])),
                "budget_bound": sum(r["evicted_entries"] > 0 for r in e),
                "question_kept_fraction": {"mean": float(np.mean(qk)), "min": float(np.min(qk)),
                                           "max": float(np.max(qk))} if qk else None,
                "failure_classes": dict(Counter(
                    c for r, wr in zip(e, w) if (c := classify(r, r["gold"], wr["filler_answers"])))),
            }
        if s:
            out[cell] = s
    (RESULTS / "evict_summary.json").write_text(json.dumps(out, indent=2))

    def ci(v):
        return f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" if v else "-"

    for cell, s in out.items():
        for name, t in s.items():
            a = t["accuracy"]
            p, q = t[f"C0_minus_{name}"], t[f"A_minus_{name}"]
            hyp = "H1 (C0 > E-stream)" if name == "E-stream" else "H2 (two-sided)"
            print(f"\n== {cell} {name}: n {t['n']}, budget bound in {t['budget_bound']}/{t['n']}"
                  f"{'  [B too weak: B comparisons not meaningful]' if t['B_too_weak'] else ''}")
            print(f"   accuracy A {ci(a['A'])}  B {ci(a['B'])}  C0 {ci(a['C0'])}  {name} {ci(a[name])}")
            print(f"   {hyp}: C0 - {name} {p['diff'][0]:+.2f} [{p['diff'][1]:+.2f},{p['diff'][2]:+.2f}], "
                  f"McNemar C0-only {p['mcnemar']['a_only']} vs {name}-only {p['mcnemar']['b_only']}, "
                  f"p = {p['mcnemar']['p']:.3f}")
            print(f"   A - {name} {q['diff'][0]:+.2f} [{q['diff'][1]:+.2f},{q['diff'][2]:+.2f}], "
                  f"p = {q['mcnemar']['p']:.3f}; runaway {t['runaway']} loop {t['loop']}; "
                  f"question kept {t['question_kept_fraction']}")
            print(f"   failure classes {t['failure_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
