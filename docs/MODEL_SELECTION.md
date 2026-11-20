# Model selection

Which checkpoints can test GEI, and for which experiment. Every number below
comes from the model's real `config.json` (or its model card where stated),
fetched 2026-10-20. None is from memory. No weights were downloaded.

## The scoping criterion

GEI frees weight memory so the context can grow into it. That only matters if
the context would otherwise run out of room. Put precisely:

> **GEI's memory benefit requires κ·T, the context memory a task accumulates,
> to be comparable to or larger than ΔW, the burnable weight budget.**
> When ΔW ≫ κ·T_max, memory never binds: the large model plus the longest
> possible context already fits, burning buys tokens the model cannot use,
> and only the energy benefit remains.

Here κ is bytes of KV cache added per token (`kv_bytes_per_token` × batch
size), T_max is the longest context the model supports, and ΔW is the bytes
freed by burning from the largest to the smallest configuration.

The ratio κ·T_max / ΔW decides it. Below 1, there is no task the large model
cannot finish for lack of memory, so the range gain is zero by construction.

### Worked example: Gemma 3n E4B fails

| | Gemma 3n E4B |
|---|---|
| κ (per sequence) | **8 KiB/token** |
| Fixed KV (sliding-window layers) | 16 MiB |
| T_max | 32,768 tokens |
| κ·T_max + fixed | **0.27 GiB** |
| ΔW (FFN 16384 → 8192, layers 10–34) | **2.34 GiB** |
| κ·T_max / ΔW | **0.11** |

The cache is small for two architectural reasons, both visible in the config:

- **KV sharing.** `num_kv_shared_layers: 15`: layers 20–34 reuse the cache of
  earlier layers, so only layers 0–19 write KV.
- **Sliding windows.** Of those 20 layers, 16 are `sliding_attention` with a
  512-token window, so their cache stops growing. Only 4 full-attention layers
  (4, 9, 14, 19) grow: 4 × 2 (K,V) × 2 KV heads × 256 head_dim × 2 B = 8 KiB.

Burning frees about 9× the most cache a single sequence can ever use. On one
sequence, Gemma 3n cannot show a range gain at any envelope size.

ΔW is computed from tensor shapes, not measured on the GPU. Layers 0–9 are
excluded because their activation sparsity thresholds over all FFN neurons
and cannot be split into chunks (`HFElasticModel(exclude_layers=...)`). Without
the exclusion, ΔW would be 3.28 GiB and the ratio worse (0.08).

### Worked example: Qwen3-4B passes

| | Qwen3-4B |
|---|---|
| κ (per sequence) | **144 KiB/token** (36 layers × 2 × 8 KV heads × 128 × 2 B) |
| T_max | 40,960 tokens |
| κ·T_max | **5.63 GiB** |
| ΔW (hypothetical: FFN 9728 → 4864, all layers) | 2.50 GiB |
| κ·T_max / ΔW | **2.25** |

Every layer is full attention with no sharing, so the cache grows fast enough
to bind. But Qwen3 has **no trained nesting**: halving its FFN is plain
slicing, and quality after the cut says nothing about GEI. It passes the
memory criterion and fails the nesting one.

### Making the envelope bind honestly: batching

`Envelope(batch_size=B)` scales κ by B: B sequences decode in lockstep, sharing
one copy of the weights, each with its own cache. This is a real serving
regime, not an artificial constraint. The Star Elastic model card makes the
same point, citing batch 224 vs 36 for its 12B vs 30B variants on one GPU.

Gemma 3n passes at full context once B ≥ 2.34 / 0.27 ≈ **9** sequences. A
tight simulated envelope (M set just above W + κ·T) is the other way to make it
bind, but it is less defensible because no real device has that envelope.

## How κ is derived

κ counts only cache that keeps growing with the sequence. All figures are bf16
(2 bytes), K and V both stored, one sequence.

**Dense GQA (Qwen3, MiniMax M2.x):** every layer is full attention.

    κ = 2 (K,V) × layers × kv_heads × head_dim × 2 B

Qwen3-4B: `num_hidden_layers: 36`, `num_key_value_heads: 8`, `head_dim: 128`,
no sliding window (`use_sliding_window: false`):
2 × 36 × 8 × 128 × 2 B = **147,456 B = 144 KiB/token**. Qwen3-8B has the same
three values, so the same κ; its larger `hidden_size` does not enter.

**Gemma 3n / Gemma 4 (KV sharing + sliding windows):** only the first
`layers − num_kv_shared_layers` layers write KV; among those, only
`full_attention` layers grow, and `sliding_attention` layers hold a fixed
`sliding_window` tokens.

    κ     = 2 × (full layers among the KV-writing ones) × global_kv_heads × global_head_dim × 2 B
    fixed = 2 × (sliding layers among them) × sliding_window × kv_heads × head_dim × 2 B

Gemma 3n E4B: layers 0–19 write KV (35 − 15); full ones are 4, 9, 14, 19;
no separate global heads, so 2 KV heads × 256: κ = 2 × 4 × 2 × 256 × 2 B =
8 KiB/token, fixed = 2 × 16 × 512 × 2 × 256 × 2 B = 16 MiB. Gemma 4 uses
`num_global_key_value_heads` and `global_head_dim` (512) for the full layers.

**Hybrid Mamba / linear attention (Nemotron, MiniMax-Text-01):** only softmax
attention layers grow; Mamba and linear-attention layers carry fixed-size state.

    κ = 2 × (attention layers) × kv_heads × head_dim × 2 B

Star Elastic 30B-A3B: `hybrid_override_pattern` has 6 `*` (attention) of 52
layers, 2 KV heads × 128: κ = 2 × 6 × 2 × 128 × 2 B = 6 KiB/token.

The simple dense formula (as in `scripts/calibrate.py`) over-counts the last
two families: applied to Gemma 3n E4B it gives 2 × 35 × 2 × 256 × 2 B = 70 KiB,
almost 9× the true 8 KiB.

## Survey

(a) trained nesting; (b) κ = KV bytes per token per sequence in bf16, counting
only layers whose cache grows. Sizes are total weight files in the repo, in
decimal GB as the Hub reports them. Free disk is about 14 GB.

| Model | (a) Trained nesting | (b) κ | Fixed KV | T_max | κ·T_max | Size | Fits 14 GB? |
|---|---|---|---|---|---|---|---|
| **Gemma 3n E4B** | **Yes**: MatFormer, E2B nested inside (card). FFN width 16384 → 8192; E2B also drops 5 layers (35 → 30) | **8 KiB** | 16 MiB | 32k | 0.27 GiB | 15.7 GB | **No** |
| Gemma 3n E2B | Is the nested child; no larger parent in the repo | 8 KiB | 16 MiB | 32k | 0.27 GiB | 10.9 GB | Yes |
| Gemma 4 E2B | No ("E" = per-layer embeddings, per card; one FFN width in config) | 6 KiB | 6 MiB | 131k | 0.76 GiB | 10.3 GB | Yes |
| Gemma 4 E4B | No (same) | 16 KiB | 20 MiB | 131k | 2.02 GiB | 16.0 GB | Cached already |
| Gemma 4 12B | No | 16 KiB¹ | 320 MiB | 262k | 4.31 GiB | 23.9 GB | Cached already |
| Gemma 4 26B-A4B | No (MoE) | 20 KiB¹ | 200 MiB | 262k | 5.20 GiB | 51.6 GB | No |
| Gemma 4 31B | No | 80 KiB¹ | 800 MiB | 262k | 20.8 GiB | 62.6 GB | Cached already |
| **Qwen3-4B** | No | **144 KiB** (2 × 36 × 8 × 128 × 2 B) | 0 | 41k | 5.63 GiB | 8.0 GB | Cached already |
| Qwen3-8B | No | 144 KiB (2 × 36 × 8 × 128 × 2 B) | 0 | 41k | 5.63 GiB | 16.4 GB | Cached already |
| MiniMax-M2.7 | None in config (MoE, 256 experts) | 248 KiB | 0 | 205k | 48.4 GiB | 230 GB | No |
| MiniMax-M3 | None in config (MoE, 128 experts) | 120 KiB² | 0 | 1M | 120 GiB | 854 GB | No |
| MiniMax-Text-01 | None in config (MoE) | 40 KiB³ | linear-attn state | 10M | 391 GiB | 915 GB | No |
| Nemotron-Elastic-12B | **Yes**: 6B/9B nested (card); slicing prunes embedding dims and MLP | ≈16 KiB⁴ | Mamba state | 131k⁴ | ≈2 GiB | 24.6 GB | No |
| Star Elastic 30B-A3B (BF16) | **Yes**: 23B/12B nested; embedding dim + MoE FFN dim differ, heads identical (card) | **6 KiB** | Mamba state | 262k | 1.50 GiB | 63.2 GB (FP8 32.7, NVFP4 19.3) | No |
| atrost/climbmix-matformer-353m | **Yes**, third-party: FFN *and attention heads* nested (s/m/l/xl) | 30 KiB (xl) | 0 | 2k | 0.06 GiB | 1.4 GB (fp32) | Yes |
| Can111/ShellElastic-MatFormer-675M-1.69B | **Yes**, third-party baseline: q/k/v/o and FFN all nested | Not in config⁵ | ? | 2k | ? | 13.2 GB (fp32) | Barely |

¹ These Gemma 4 models set `attention_k_eq_v: true`. The figure assumes K and
V are both stored. If the runtime stores K once, κ halves. Not checked: the
local transformers 4.57 has no Gemma 4 code.
² Excludes the sparse-attention index keys (`sparse_index_dim: 128`, 4 index
heads on layers 3–59), which add to κ.
³ Only the 10 softmax layers (7, 15, …, 79) grow. The other 70 are linear
attention with fixed-size state.
⁴ Config is gated; not read. The card says "just four Attention layers";
the parent Nemotron Nano 12B v2 config has six. Estimate = 4 layers × the
parent's 8 KV heads × 128 head_dim. Treat as unverified.
⁵ `ElasticForCausalLM` custom code; head counts are not in the config.

Also checked and dropped: MiniMax-H3 is a video/audio diffusion pipeline, not
an LLM; `varshu23/gemma3n-text-matformer` contains no weights (placeholder
files).

### Criterion scores for the nested candidates

| Model | ΔW (largest → smallest) | κ·T_max | Ratio | Batch to bind at T_max |
|---|---|---|---|---|
| Gemma 3n E4B → mix-n-match (0–9 fixed) | 2.34 GiB | 0.27 GiB | 0.11 | ≥ 9 |
| Star Elastic 30B → 12B | ≈ 33.5 GiB⁶ | 1.50 GiB | 0.045 | ≥ 23 |
| Nemotron-Elastic 12B → 6B | ≈ 11 GiB⁶ | ≈ 2 GiB | ≈ 0.18 | ≥ 6 |
| atrost MatFormer xl → s | ≥ 0.26 GiB⁷ | 0.06 GiB | ≤ 0.22 | ≥ 5 |

⁶ From parameter counts on the model card × 2 bytes, not tensor shapes.
⁷ FFN only, from the config; nested attention heads add more.

**Every checkpoint with trained nesting fails the criterion for a single
sequence.** This is structural rather than bad luck. The architectures that
ship with nesting (Gemma 3n, Nemotron hybrids) are also the ones built for
small caches: KV sharing, sliding windows, Mamba layers with fixed state. The
models with large κ (Qwen3, MiniMax M2.x) are not nested. The paper should
state this openly: on current nested checkpoints, GEI's memory benefit
appears under batched serving, not single long generations.

## Recommendation

**Experiment 1 (cache reuse after a switch): Gemma 3n E4B.**

- It is the only trained-nesting checkpoint whose nesting is FFN-only with
  shared attention, so the cache the large model writes is shape-compatible
  with the small one, and that is the question Experiment 1 asks.
- It is native to transformers and works with `HFElasticModel` as written,
  with layers 0–9 excluded.
- The criterion does not matter for Experiment 1, which measures quality after
  a switch, not memory.
- **Blocker:** 15.7 GB does not fit the ~14 GB free. Free about 3 GB more.
  E2B alone does not help: it is the nested child, with nothing to burn from.
- Fallback that fits today: `atrost/climbmix-matformer-353m` (1.4 GB). It is
  trained-nested but third-party, unvetted, 353M parameters, 2k context,
  custom code, and nests attention heads, so its cache layout changes across
  sizes. Use it only as a smoke test of the harness on real nested weights.
- Star Elastic is the stronger Experiment 1 model scientifically (its card says
  the nested variants "preserve the Mamba and attention layer structure,
  enabling cache-state transplantation"), but at 19.3 GB even in NVFP4 it does
  not fit, and `HFElasticModel` supports neither embedding-dimension nor MoE
  nesting.

**Range-gain experiments: Gemma 3n E4B under batching, with Qwen3-4B as a
mechanics control.**

- Primary: Gemma 3n E4B with `batch_size ≥ 9` at full context (more at shorter
  T). Same checkpoint as Experiment 1, trained nesting, and a regime the
  field already treats as real. Report the range gain as measured under that
  batch size, never as the single-sequence ideal.
- Control: Qwen3-4B, already cached (8.0 GB). It passes the criterion per
  sequence (ratio 2.25), so it can show the horizon extension on one long
  generation. Its FFN slicing is untrained, so use it only for memory and
  horizon mechanics, never for quality claims.
- Use `HFElasticModel(kv_cache="block")` with batched decoding
  (`prefill_batch` / `step_batch`) for every range-gain run. Under a full GPU,
  HF's `DynamicCache` loses about one layer's share of the cache: 2.8% of the
  gain on Qwen3-8B (36 KV layers, batch 1), but half the freed bytes on a
  4-KV-layer toy shape, which is Gemma 3n's structure (docs/VRAM_FINDINGS.md,
  docs/RANGE_GAIN.md). Report the range gain as measured with the cache
  actually used.

## Sources

Configs and file listings from the Hugging Face Hub, 2026-10-20:
`google/gemma-3n-E4B-it` and `-E2B-it` (configs from the ungated
`unsloth/` mirrors; Google's own are gated, and the E4B mirror config matches
the transformers 4.57 `Gemma3nTextConfig` defaults exactly), `google/gemma-4-*`,
`Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `MiniMaxAI/MiniMax-M2.7`, `-M3`,
`-Text-01-hf`, `nvidia/Nemotron-Elastic-12B` (card and file listing only),
`nvidia/NVIDIA-Nemotron-Nano-12B-v2`,
`nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16` (config and card),
`atrost/climbmix-matformer-353m-1p2b-h100`,
`Can111/ShellElastic-baseline-MatFormer-675M-1.69B`.
