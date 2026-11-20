# GEI against KV-cache eviction at equal memory

Under the same memory envelope, is it better to keep the large model and
**evict** KV cache to fit, or to **burn** to the small configuration and keep
the **full** cache (GEI)? Results: [`RESULTS.md`](RESULTS.md).

| Foreign cache | GEI (C0) | StreamingLLM | SnapKV |
|---|---|---|---|
| 822 tokens | 0.76 | 0.80 | 0.92 |
| 3,922 tokens | **0.62** | 0.00 | 0.36 |
| 16,206 tokens | **0.40** | 0.00 | 0.06 |

Gemma 3n E4B, GSM8K problems 0-49 per cell, eviction budget 1,024 cache tokens
per sequence (the large model's share of the batch-16 envelope in
`../docs/RANGE_GAIN_BATCHED.md`).

## Pre-specification

The design (conditions, budget, both policies and their parameters, hypotheses
and checks) is in [`docs/PREREG.md`](docs/PREREG.md), frozen on 2026-11-18 at
10:34 PST before any eviction code or data existed, and sent to the co-author
before any eviction data. Verify it is unchanged:

```bash
cd kv-eviction-comparison/docs && sha256sum -c PREREG.sha256
```

## Files

| Path | Contents |
|---|---|
| `docs/PREREG.md`, `docs/PREREG.sha256` | Frozen design and its SHA-256. |
| `scripts/evict.py` | StreamingLLM and SnapKV-style eviction for Gemma 3n's growing caches. |
| `scripts/test_evict.py` | Engineering test on a tiny random Gemma 3n: eviction equals a masked full-cache reference. |
| `scripts/repro_check.py` | Check 1: the environment reproduces the stored A and C0 records. |
| `scripts/run_eviction.py` | Check 2 (`--check-zero`) and the registered runs (`--policy stream\|snap`). |
| `scripts/report.py` | Pre-registered analysis -> `results/evict_summary.json`. |
| `results/` | Per-trial records of every run (text, prediction, tokens kept). |
| `logs/` | Run logs, analysis output, environment record (`environment.txt`, `pip_freeze.txt`). |

The A, B and C0 conditions are the stored records of the worst-case run
(`../results/exp1_worst_trials.jsonl`); check 1 reproduced 30/30 of them exactly
in this environment.

## Environment and run order

Python 3.11.18, torch 2.12.0+cu126, transformers 4.57.6, Triton 3.7.0, one
RTX 4090 (`logs/pip_freeze.txt`). Model `google/gemma-3n-E4B-it` at revision
`c1221e9c62e34a43ab7ffacd1be0ea71f126ef10` (gated: accept the licence on
Hugging Face first).

```bash
python kv-eviction-comparison/scripts/test_evict.py          # engineering test
python kv-eviction-comparison/scripts/repro_check.py         # check 1
python kv-eviction-comparison/scripts/run_eviction.py --check-zero   # check 2
python kv-eviction-comparison/scripts/run_eviction.py --policy stream
python kv-eviction-comparison/scripts/run_eviction.py --policy snap
python kv-eviction-comparison/scripts/report.py
```

Runs are resumable; about 1.5 h for StreamingLLM and 2 h for SnapKV on an
RTX 4090.
