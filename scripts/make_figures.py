#!/usr/bin/env python3
# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Generate the figures from committed result files only.

Writes PDF (for print) and PNG (for the README) of each figure to docs/figures/,
or to the directory given with --out.

  fig_shock_vs_length      first-token switch-shock KL vs foreign-cache length,
                           typical case (switch mid-answer) and worst case
                           (switch at token 0), mean with bootstrap 95% CI
  fig_kl_decay             switch-shock KL by token offset after the switch
  fig_range_gain           context reached before the envelope fails, per
                           condition (Qwen3-8B batch 1; Gemma 3n batch 16 if run)
  fig_mechanism            weights W and cache K against the envelope M, never
                           burning vs GEI (Qwen3-8B batch 1); measured points from
                           the run logs, curve from the block-row arithmetic,
                           which is checked against every logged point first
  fig_layouts              schematic of allocator segments under each weight
                           layout, with the measured KV gain per byte freed

Inputs (all committed): results/exp1_dose_trials.jsonl,
results/exp1_worst_trials.jsonl, results/range_gain_*.json,
results/range_gain_batched_*.json. Palette: categorical slots 1-2 of the
dataviz reference palette (blue, orange), validated on white; markers and
line styles give a second encoding for grayscale print.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gei.metrics import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a85", "#1f1f1e", "#6b6b66"
CELLS = ("anchor", "1k", "4k", "16k")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)


def load_jsonl(name):
    return [json.loads(line) for line in (ROOT / "results" / name).read_text().splitlines()]


def typical_first_token():
    """Typical case: KL at offset 0 after the switch (exact: no divergence yet)."""
    rows = load_jsonl("exp1_dose_trials.jsonl")
    out = []
    for cell in CELLS:
        c = [r for r in rows if r["cell"] == cell and r["condition"] == "C_reuse"]
        kl = [r["kl_by_offset"][0] for r in c if r["kl_by_offset"]]
        out.append((float(np.median([r["foreign_tokens"] for r in c])), np.mean(kl),
                    *bootstrap_ci(kl)))
    return np.array(out)


def worst_first_token():
    rows = load_jsonl("exp1_worst_trials.jsonl")
    out = []
    for cell in CELLS:
        c = [r for r in rows if r["cell"] == cell]
        kl = [r["C0"]["kl_first"] for r in c]
        out.append((float(np.median([r["foreign_tokens"] for r in c])), np.mean(kl),
                    *bootstrap_ci(kl)))
    return np.array(out)


def fig_shock_vs_length():
    typ, worst = typical_first_token(), worst_first_token()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for data, color, marker, style, label in (
            (worst, ORANGE, "s", "-", "Worst case: switch at token 0 (C0 vs B)"),
            (typ, BLUE, "o", "--", "Typical: switch mid-answer (C vs fresh cache)")):
        x, m, lo, hi = data.T
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=color, marker=marker, ms=5,
                    linestyle=style, capsize=2, elinewidth=1, label=label)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.text(2200, -0.075, "recompute (D0 vs B): exactly 0", color=MUTED, fontsize=7,
            ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Foreign-cache length (tokens, median per cell)")
    ax.set_ylabel("First-token KL (nats)")
    ax.set_ylim(-0.10, 0.85)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_shock_vs_length")
    plt.close(fig)
    return typ, worst


EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 200)  # log-spaced offset bins


def offset_curve(kls):
    """Mean KL in log-spaced offset bins (the raw per-offset means are noisy late,
    where fewer answers are still running). Returns (bin centre, mean)."""
    xs, ys = [], []
    for lo, hi in itertools.pairwise(EDGES):
        vals = [v for k in kls for v in k[lo:hi]]
        if vals:
            xs.append(np.sqrt((lo + 1) * hi))
            ys.append(np.mean(vals))
    return np.array(xs), np.array(ys)


def fig_kl_decay():
    dose = load_jsonl("exp1_dose_trials.jsonl")
    worst = load_jsonl("exp1_worst_trials.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    shades = {"anchor": 0.35, "1k": 0.55, "4k": 0.78, "16k": 1.0}
    labels = {"anchor": "~100", "1k": "1k", "4k": "4k", "16k": "16k"}
    for ax, title, getter, color in (
            (axes[0], "Typical: switch mid-answer (C)",
             lambda cell: [r["kl_by_offset"] for r in dose
                           if r["cell"] == cell and r["condition"] == "C_reuse"], BLUE),
            (axes[1], "Worst case: switch at token 0 (C0)",
             lambda cell: [r["C0"]["kl_by_offset"] for r in worst if r["cell"] == cell], ORANGE)):
        for cell in CELLS:
            x, y = offset_curve(getter(cell))
            ax.plot(x, y, color=color, alpha=shades[cell], linewidth=1.6, marker="o",
                    ms=3.5, label=f"{labels[cell]} tokens")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, color=INK)
        ax.set_xlabel("Tokens after the switch")
    nx, ny = offset_curve([r["kl_by_offset"] for r in dose if r["condition"] == "D_recompute"])
    for ax in axes:
        ax.plot(nx, ny, color=GRAY, linewidth=1, linestyle=":", marker="^", ms=3,
                label="D: bf16 noise floor (typical case)")
    axes[0].set_ylabel("Mean KL (nats)")
    # One shared legend: each panel's colour names its case, so cache length is
    # shown in neutral shades (lighter = shorter) to avoid repeating it per panel.
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK, alpha=shades[c], marker="o", ms=3.5, linewidth=1.6,
                      label=f"{labels[c]} tokens") for c in CELLS]
    handles.append(Line2D([], [], color=GRAY, linestyle=":", marker="^", ms=3, linewidth=1,
                          label="D: bf16 noise floor"))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.26), title="Foreign-cache length (lighter = shorter)",
               title_fontsize=8, fontsize=7.5)
    save(fig, "fig_kl_decay")
    plt.close(fig)


def fig_range_gain():
    def ctx(name):
        p = ROOT / "results" / name
        return json.loads(p.read_text())["context_tokens"] if p.exists() else None

    panels = [("Qwen3-8B, batch 1 (untrained nesting)", {
        "Large, never burns": ctx("range_gain_A.json"),
        "Small from the start": ctx("range_gain_C.json"),
        "GEI: burn when the envelope binds": ctx("range_gain_B.json"),
        "GEI, HF DynamicCache": ctx("range_gain_B-dynamic.json"),
        "Large, never burns, DynamicCache": ctx("range_gain_A-dynamic.json")})]
    batched = {
        "Large, never burns": ctx("range_gain_batched_A.json"),
        "Small from the start": ctx("range_gain_batched_C.json"),
        "GEI: burn when the envelope binds": ctx("range_gain_batched_B.json"),
        "GEI, HF DynamicCache": ctx("range_gain_batched_B-dynamic.json"),
        "Large, never burns, DynamicCache": ctx("range_gain_batched_A-dynamic.json")}
    if all(v is not None for v in batched.values()):
        panels.append(("Gemma 3n E4B, batch 16 (trained nesting)", batched))
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.8, 1.75 * len(panels) + 0.3),
                             squeeze=False, gridspec_kw={"hspace": 0.75})
    for ax, (title, data) in zip(axes[:, 0], panels):
        names = list(data)[::-1]
        vals = [data[n] for n in names]
        colors = [BLUE if n.startswith("GEI: burn") else
                  ("#9ec5f4" if n.startswith("GEI,") else GRAY) for n in names]
        bars = ax.barh(names, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() * 1.0 + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:,}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(vals) * 1.13)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("Context reached before the envelope fails (tokens per sequence)")
    save(fig, "fig_range_gain")
    plt.close(fig)
    return len(panels)


ROW = 512 * 147_456  # one 512-token block row on Qwen3-8B (docs/RANGE_GAIN.md)
GIB = 1024**3


def cache_bytes(context):
    """Block-cache bytes at a given context: whole 512-token rows, ceil(cached/512)."""
    cached = np.maximum(np.asarray(context) - 1, 0)
    return np.ceil(cached / 512) * ROW


def logged_points(name, prompt_tokens):
    """(context, cache bytes) measured every 2,000 tokens in a run log."""
    pts = []
    for line in (ROOT / "results" / name).read_text().splitlines():
        m = re.match(r"\s*(\d+) tokens, kv ([\d.]+) GiB", line)
        if m:
            pts.append((prompt_tokens + int(m.group(1)), float(m.group(2)) * GIB))
    return pts


def fig_mechanism():
    """W(t) and K(t) under one envelope: large never burns vs GEI (Qwen3-8B, batch 1)."""
    a = json.loads((ROOT / "results" / "range_gain_A.json").read_text())
    b = json.loads((ROOT / "results" / "range_gain_B.json").read_text())
    M = a["envelope_bytes"]
    w_large, w_small = a["weight_bytes"]
    burn_ctx = b["burn"]["context_tokens"]
    runs = [("Large configuration, never burns", a, None),
            ("GEI: burn when the envelope binds", b, burn_ctx)]
    # The row arithmetic must reproduce every logged cache size before it is drawn.
    for _, r, _ in runs:
        for ctx, kv in logged_points(f"range_gain_{r['condition']}.log", r["prompt_tokens"]):
            assert abs(cache_bytes(ctx) - kv) / GIB < 0.001, (ctx, kv)
    xmax = 40_000
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, (title, r, burn) in zip(axes, runs):
        end = r["context_tokens"]
        x = np.arange(1, end + 1)
        w = np.full(x.shape, float(w_large))
        if burn is not None:
            w[x >= burn] = w_small
        k = cache_bytes(x)
        ax.fill_between(x, 0, w / GIB, color="#b7d3f6", linewidth=0, label="weights $W$")
        ax.fill_between(x, w / GIB, (w + k) / GIB, color=ORANGE, alpha=0.85, linewidth=0,
                        label="KV cache $K$")
        ax.axhline(M / GIB, color=INK, linewidth=1.2)
        ax.text(xmax * 0.98, M / GIB + 0.25, f"envelope $M$ = {M / GIB:.2f} GiB",
                ha="right", va="bottom", fontsize=7.5, color=INK)
        pts = logged_points(f"range_gain_{r['condition']}.log", r["prompt_tokens"])
        if pts:
            px, pk = zip(*pts)
            pw = [w_small if (burn is not None and c >= burn) else w_large for c in px]
            ax.plot(px, (np.array(pw) + np.array(pk)) / GIB, "o", ms=3, color=INK,
                    label="measured (run log)")
        ax.plot([end], [(w[-1] + k[-1]) / GIB], marker="X", ms=8, color=INK, linestyle="none")
        ax.annotate(f"out of memory\nat {end:,}", xy=(end, (w[-1] + k[-1]) / GIB),
                    xytext=(end + (1_500 if end < 10_000 else -1_500), 7.5),
                    ha="left" if end < 10_000 else "right", fontsize=7.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        if burn is not None:
            ax.annotate(f"burn at {burn:,}:\n$W$ drops by {(w_large - w_small) / GIB:.4f} GiB",
                        xy=(burn, w_small / GIB), xytext=(9_000, 4.0), fontsize=7.5,
                        color=INK, arrowprops={"arrowstyle": "-", "color": MUTED,
                                               "linewidth": 0.8})
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 17.5)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Context (tokens)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Memory (GiB)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    save(fig, "fig_mechanism")
    plt.close(fig)


def fig_layouts():
    """Schematic (not to scale): allocator segments before and after a burn."""
    from matplotlib.patches import Rectangle

    KEEP, DROP, FREE, STRAND = "#b7d3f6", ORANGE, "white", "#d9d9d6"
    rows = [
        ("View (slice, no copy)", "0%",
         [[("k", 1), ("d", 1)]], [[("k", 1), ("vd", 1)]],
         "the view keeps the\noriginal tensor alive"),
        ("Clone at switch", "97.8–98.9%",
         [[("k", 1), ("d", 1)], [("k", 1), ("d", 1)]],
         [[("k", 0.5), ("k", 0.5), ("s", 1)], [("r", 2)]],
         "copies land in freed blocks;\nthe remainders are stranded"),
        ("Split at load, large chunks", "99.9–100.1%",
         [[("k", 2)], [("d", 2)]], [[("k", 2)], [("r", 2)]],
         "48 MiB chunks get their own\nsegments; dropping frees them"),
        ("Split at load, small chunks", "4.4%\n(= no burn)",
         [[("k", 0.5), ("d", 0.5), ("k", 0.5), ("d", 0.5)]],
         [[("k", 0.5), ("s", 0.5), ("k", 0.5), ("s", 0.5)]],
         "1–10 MiB chunks share 20 MiB\nsegments with kept weights"),
        ("One buffer per burn level", "100%",
         [[("k", 0.5), ("k", 0.5), ("k", 0.5), ("k", 0.5)], [("d", 2)]],
         [[("k", 0.5), ("k", 0.5), ("k", 0.5), ("k", 0.5)], [("r", 2)]],
         "the level's chunks are one\nallocation, freed whole"),
    ]
    colors = {"k": KEEP, "d": DROP, "s": STRAND, "r": FREE, "vd": KEEP}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = len(rows) - 1
    seg_w, gap = 2.0, 0.25

    def draw(x0, yy, segments):
        for seg in segments:
            # the segment outline (an allocator segment)
            ax.add_patch(Rectangle((x0, yy), seg_w, 0.55, fill=False, edgecolor=INK,
                                   linewidth=1.0))
            x = x0
            for kind, w in seg:
                ax.add_patch(Rectangle((x + 0.03, yy + 0.06), w - 0.06, 0.43,
                                       facecolor=colors[kind],
                                       edgecolor=MUTED if kind in ("r", "s") else "none",
                                       linestyle="--" if kind == "r" else "-",
                                       hatch="////" if kind == "s" else None,
                                       linewidth=0.8))
                if kind == "r":
                    ax.text(x + w / 2, yy + 0.27, "returned", ha="center", va="center",
                            fontsize=6.5, color=MUTED)
                if kind == "vd":
                    ax.text(x + w / 2, yy + 0.27, "still held", ha="center", va="center",
                            fontsize=6.5, color=INK)
                x += w
            x0 += seg_w + gap
        return x0

    for name, gain, before, after, why in rows:
        ax.text(0, y + 0.47, name, fontsize=8, color=INK, ha="left", va="center",
                fontweight="bold")
        ax.text(0, y + 0.08, why, fontsize=6.8, color=MUTED, ha="left", va="center",
                linespacing=1.1)
        draw(6.2, y, before)
        ax.annotate("", xy=(11.35, y + 0.27), xytext=(10.9, y + 0.27),
                    arrowprops={"arrowstyle": "->", "color": MUTED})
        draw(11.6, y, after)
        ax.text(16.4, y + 0.27, gain, fontsize=8.5, color=INK, ha="left", va="center",
                linespacing=1.1)
        y -= 1
    ax.text(8.3, len(rows) - 0.05, "before the burn", ha="center", fontsize=7.5, color=MUTED)
    ax.text(13.7, len(rows) - 0.05, "after the burn", ha="center", fontsize=7.5, color=MUTED)
    ax.text(16.4, len(rows) - 0.05, "KV gain / freed", ha="left", fontsize=7.5, color=MUTED)
    handles = [Rectangle((0, 0), 1, 1, facecolor=KEEP, label="kept weights"),
               Rectangle((0, 0), 1, 1, facecolor=DROP, label="droppable chunk"),
               Rectangle((0, 0), 1, 1, facecolor=STRAND, hatch="////", edgecolor=MUTED,
                         label="freed but stranded"),
               Rectangle((0, 0), 1, 1, facecolor=FREE, edgecolor=MUTED, linestyle="--",
                         label="returned to the driver")]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    ax.text(0, -0.55, "Boxes: allocator segments (schematic, not to scale).", fontsize=6.8,
            color=MUTED)
    ax.set_xlim(0, 18.6)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.axis("off")
    save(fig, "fig_layouts")
    plt.close(fig)


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output directory")
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    typ, worst = fig_shock_vs_length()
    fig_kl_decay()
    n_panels = fig_range_gain()
    fig_mechanism()
    fig_layouts()
    print("typical first-token KL (foreign, mean, lo, hi):")
    for row in typ:
        print("  " + ", ".join(f"{v:.3f}" for v in row))
    print("worst first-token KL (foreign, mean, lo, hi):")
    for row in worst:
        print("  " + ", ".join(f"{v:.3f}" for v in row))
    print(f"range-gain panels: {n_panels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
