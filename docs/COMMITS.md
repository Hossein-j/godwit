# Commits cited in the paper

The paper's provenance appendix cites each source document by the commit that
last changed its cited content. Those commits belong to the full development
history (38 commits, 13 to 30 October 2026), which is archived and available
to editors and reviewers on request. This repository is
a clean release of the same documents; the table below maps every cited
commit to what it contains.

| Commit | Date | Message | Cited for |
|---|---|---|---|
| `ebadf1e` | 2026-10-13 | GEI: envelope, burn schedule, launch check, controller, Experiment 1 harness | CODE: `src/gei/schedule.py`, `launch.py`, `controller.py`; DR: `scripts/run_exp1.py` (Experiment 1 decision rule) |
| `71ab1a4` | 2026-10-20 | Add model selection survey and Envelope batch_size | CODE: `src/gei/envelope.py` |
| `ccc5096` | 2026-10-20 | Measure range gain on Qwen3-8B at batch 1: +36,864 tokens as predicted | First record of the range-gain prediction together with its results |
| `283c58b` | 2026-10-20 | Range gain: rerun on shipping kernel, add DynamicCache ablation | MS: `docs/MODEL_SELECTION.md`; VF: `docs/VRAM_FINDINGS.md`; RG: `docs/RANGE_GAIN.md` |
| `5852ba1` | 2026-10-21 | Experiment 1 pilot on Gemma 3n E4B: case (1), KV reuse works | P1: `docs/EXP1_PILOT.md` |
| `40a7dfa` | 2026-10-26 | Pre-register the switch-at-token-0 (worst case) experiment | PR: `docs/EXP1_WORST_PREREG.md` |
| `8b0e57a` | 2026-10-26 | Switch shock vs foreign-cache length: saturates by ~1k, C never separates from D | DS: `docs/EXP1_DOSE.md` |
| `55fcae1` | 2026-10-26 | Worst case, switch at token 0: shock grows to ~4k, no accuracy cost | RM: `README.md`; WC: `docs/EXP1_WORST.md` |
| `d58b6be` | 2026-10-27 | Pre-register the batched range gain on Gemma 3n E4B | PB: `docs/RANGE_GAIN_BATCHED_PREREG.md` |
| `0cef765` | 2026-10-27 | Batched range gain on Gemma 3n E4B: +19,455 tokens per sequence | RB: `docs/RANGE_GAIN_BATCHED.md`, `results/range_gain_batched_*.json` |
| `373265f` | 2026-10-27 | Commit the per-trial records and run summaries | TR: `results/` per-trial records and run summaries |

The eviction comparison's pre-specification (KP) is identified by its SHA-256,
`833d7167291ac0e9d9d772ad6fcbd973006cf1e04734670459f8dbc06b83de46`, not by a
commit.

## Checking without the history

The pre-specified documents can be checked here directly:

```bash
sha256sum -c PREREG_SHA256SUMS
```

`docs/EXP1_WORST_PREREG.md`, `docs/RANGE_GAIN_BATCHED_PREREG.md` and
`kv-eviction-comparison/docs/PREREG.md` are byte-identical to their registered
versions; `scripts/run_exp1.py` differs from `ebadf1e` only by the licence
header added before release. Every reported number can be recomputed from the
per-trial records in `results/` and `kv-eviction-comparison/results/`.
