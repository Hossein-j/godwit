#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Summarise exp1_dose.py trials: the switch-shock dose-response curve.

Per cell (foreign-cache size): switch-shock KL for C and D (mean, median, p90,
C/D ratio), KL by offset after the switch, accuracy per condition with
bootstrap CIs, paired C - D and C - B accuracy differences, and runaway / loop
rates per condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gei.metrics import bootstrap_ci, paired_bootstrap_ci

CONDS = ("A_large", "B_small", "C_reuse", "D_recompute")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", default="results/exp1_dose_trials.jsonl")
    ap.add_argument("--out", default="results/exp1_dose_summary.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.trials).read_text().splitlines()]
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    summary = {}
    for cell in cells:
        by = {c: {r["problem_id"]: r for r in rows if r["cell"] == cell and r["condition"] == c}
              for c in CONDS}
        pids = sorted(set.intersection(*(set(by[c]) for c in CONDS)))  # complete problems only
        acc = {c: [float(by[c][p]["correct"]) for p in pids] for c in CONDS}
        kl = {c: np.array([by[c][p]["kl_mean"] for p in pids if by[c][p]["kl_mean"] is not None])
              for c in ("C_reuse", "D_recompute")}
        foreign = [by["C_reuse"][p]["foreign_tokens"] for p in pids]
        offsets = {}
        for c in ("C_reuse", "D_recompute"):
            curves = [by[c][p]["kl_by_offset"] for p in pids]
            width = max((len(x) for x in curves), default=0)
            offsets[c] = [float(np.mean([x[i] for x in curves if len(x) > i]))
                          for i in range(width)]
        s = {
            "n_problems": len(pids),
            "foreign_tokens": {"median": float(np.median(foreign)), "min": int(min(foreign)),
                               "max": int(max(foreign))},
            "accuracy": {c: [float(np.mean(acc[c])), *bootstrap_ci(acc[c])] for c in CONDS},
            "paired": {f"{x}-{y}": list(paired_bootstrap_ci(acc[x], acc[y]))
                       for x, y in (("C_reuse", "D_recompute"), ("C_reuse", "B_small"),
                                    ("D_recompute", "B_small"), ("A_large", "B_small"))},
            "kl": {c: {"mean": float(kl[c].mean()), "median": float(np.median(kl[c])),
                       "p90": float(np.quantile(kl[c], 0.9))} for c in kl},
            "kl_ratio_C_over_D": float(kl["C_reuse"].mean() / kl["D_recompute"].mean()),
            "kl_by_offset_bins": {c: {f"{a}-{b}": float(np.mean(offsets[c][a:b]))
                                      for a, b in ((0, 10), (10, 50), (50, 100), (100, 200))
                                      if len(offsets[c]) > a}
                                  for c in offsets},
            "runaway": {c: int(sum(by[c][p]["runaway"] for p in pids)) for c in CONDS},
            "loop": {c: int(sum(by[c][p]["loop"] for p in pids)) for c in CONDS},
            "c_only_failures_vs_d": {
                "runaway": int(sum(by["C_reuse"][p]["runaway"] and not by["D_recompute"][p]["runaway"] for p in pids)),
                "runaway_d_only": int(sum(by["D_recompute"][p]["runaway"] and not by["C_reuse"][p]["runaway"] for p in pids)),
                "loop": int(sum(by["C_reuse"][p]["loop"] and not by["D_recompute"][p]["loop"] for p in pids)),
                "loop_d_only": int(sum(by["D_recompute"][p]["loop"] and not by["C_reuse"][p]["loop"] for p in pids)),
            },
        }
        summary[cell] = s
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(f"{'cell':7s}{'n':>4}{'foreign':>9}{'KL C':>9}{'KL D':>10}{'C/D':>7}"
          f"{'acc A':>7}{'B':>6}{'C':>6}{'D':>6}{'C-D [95% CI]':>22}{'C-B [95% CI]':>22}"
          f"{'runaway A/B/C/D':>17}{'loop A/B/C/D':>15}")
    for cell, s in summary.items():
        a = s["accuracy"]
        cd, cb = s["paired"]["C_reuse-D_recompute"], s["paired"]["C_reuse-B_small"]
        print(f"{cell:7s}{s['n_problems']:>4}{s['foreign_tokens']['median']:>9.0f}"
              f"{s['kl']['C_reuse']['mean']:>9.4f}{s['kl']['D_recompute']['mean']:>10.5f}"
              f"{s['kl_ratio_C_over_D']:>7.1f}"
              + "".join(f"{a[c][0]:>6.2f}" if i else f"{a[c][0]:>7.2f}" for i, c in enumerate(CONDS))
              + f"  {cd[0]:+.2f} [{cd[1]:+.2f},{cd[2]:+.2f}]  {cb[0]:+.2f} [{cb[1]:+.2f},{cb[2]:+.2f}]"
              + f"{'/'.join(str(s['runaway'][c]) for c in CONDS):>17}"
              + f"{'/'.join(str(s['loop'][c]) for c in CONDS):>15}")
    print("\nKL by offset after the switch (C_reuse):")
    for cell, s in summary.items():
        print(f"  {cell:7s}", {k: round(v, 4) for k, v in s["kl_by_offset_bins"]["C_reuse"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
