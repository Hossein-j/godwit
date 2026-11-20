# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Backend interface for an elastic (nested) model, plus a CPU mock.

GEI needs a model that contains smaller versions of itself and can switch
between them *while generating*. MatFormer and Nemotron Elastic provide the
nesting; neither switches mid-generation, which is the gap GEI fills.

The hard open question lives here: after switching down, the smaller model must
continue from a KV cache written by its larger parent. Experiment 1 tests
exactly that, so the interface exposes `switch_to` as a first-class operation
with a `kv_policy`.

The mock backend makes every other part of the system testable on CPU, with no
GPU and no model download. HFElasticModel is the real backend; it imports torch
lazily, so this module still loads on a CPU-only install.
"""

from __future__ import annotations

import abc
import functools
import gc
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .envelope import ModelConfig


class KVPolicy:
    """What happens to the existing KV cache when the model shrinks."""

    REUSE = "reuse"  # condition C: keep the parent's cache as-is (GEI's proposal)
    RECOMPUTE = "recompute"  # condition D: rebuild it with the smaller model
    DROP = "drop"  # diagnostic only: discard context entirely


@dataclass
class StepResult:
    """One decoding step."""

    token_id: int
    logprobs: np.ndarray | None = None
    config_index: int = 0


class ElasticModel(abc.ABC):
    """A nested model family that can change size mid-generation."""

    @property
    @abc.abstractmethod
    def configs(self) -> list[ModelConfig]:
        """Nested configurations, largest first."""

    @property
    @abc.abstractmethod
    def current_index(self) -> int:
        """Index of the configuration currently resident."""

    @abc.abstractmethod
    def prefill(self, prompt_token_ids: list[int]) -> None:
        """Run the prompt and populate the KV cache."""

    @abc.abstractmethod
    def step(self) -> StepResult:
        """Generate one token from the current state."""

    @abc.abstractmethod
    def switch_to(self, index: int, kv_policy: str = KVPolicy.REUSE) -> None:
        """Shrink to a smaller nested configuration.

        Must refuse to grow: burning is irreversible within a task.
        """

    @abc.abstractmethod
    def kv_bytes(self) -> int:
        """Current KV cache size in bytes."""

    def reset(self) -> None:  # pragma: no cover - optional for backends
        raise NotImplementedError


class MockElasticModel(ElasticModel):
    """Deterministic CPU stand-in used by the tests and the dry-run path.

    It imitates the one behaviour that matters for the control logic: smaller
    configurations are cheaper and a little worse, and a mid-generation switch
    costs some accuracy for a while afterwards ("switch shock"). The magnitude
    of that shock is a *simulation parameter*, not a measurement — Experiment 1
    exists to measure the real one.
    """

    def __init__(
        self,
        configs: list[ModelConfig],
        *,
        kv_bytes_per_token: int = 128 * 1024,
        vocab_size: int = 256,
        switch_shock: float = 0.15,
        shock_decay_tokens: int = 200,
        seed: int = 0,
    ) -> None:
        self._configs = list(configs)
        self._index = 0
        self._tokens = 0
        self._kv_per_token = kv_bytes_per_token
        self._vocab = vocab_size
        self._switch_shock = switch_shock
        self._shock_decay = max(shock_decay_tokens, 1)
        self._tokens_since_switch = 10**9
        self._rng = np.random.default_rng(seed)
        self._last_kv_policy = KVPolicy.REUSE

    @property
    def configs(self) -> list[ModelConfig]:
        return self._configs

    @property
    def current_index(self) -> int:
        return self._index

    def prefill(self, prompt_token_ids: list[int]) -> None:
        self._tokens = len(prompt_token_ids)

    def effective_quality(self) -> float:
        """Per-token quality, degraded briefly after a switch.

        RECOMPUTE pays compute to avoid the shock; REUSE is free but may not be.
        """
        base = self._configs[self._index].quality
        if self._last_kv_policy == KVPolicy.RECOMPUTE:
            return base
        decay = max(0.0, 1.0 - self._tokens_since_switch / self._shock_decay)
        return float(np.clip(base - self._switch_shock * decay, 0.0, 1.0))

    def step(self) -> StepResult:
        q = self.effective_quality()
        logits = self._rng.normal(size=self._vocab)
        logits[0] += 8.0 * q  # token 0 stands in for "the correct continuation"
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        token = int(self._rng.choice(self._vocab, p=probs))
        self._tokens += 1
        self._tokens_since_switch += 1
        return StepResult(
            token_id=token, logprobs=np.log(probs), config_index=self._index
        )

    def switch_to(self, index: int, kv_policy: str = KVPolicy.REUSE) -> None:
        if index < self._index:
            raise ValueError("cannot grow mid-task: burning is irreversible")
        if index >= len(self._configs):
            raise IndexError("configuration index out of range")
        if index == self._index:
            return
        self._index = index
        self._tokens_since_switch = 0
        self._last_kv_policy = kv_policy
        if kv_policy == KVPolicy.DROP:
            self._tokens = 0

    def kv_bytes(self) -> int:
        return self._tokens * self._kv_per_token

    def reset(self) -> None:
        self._index = 0
        self._tokens = 0
        self._tokens_since_switch = 10**9
        self._last_kv_policy = KVPolicy.REUSE




def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-only installs
        raise ImportError(
            "HFElasticModel needs torch and transformers: pip install -e '.[gpu]'"
        ) from exc
    return torch


@functools.cache
def _nested_mlp_class():
    """Build the nn.Module subclass lazily, so this module imports without torch."""
    torch = _require_torch()
    nn = torch.nn

    class NestedMLP(nn.Module):
        """A gated FFN stored as separately allocated neuron-range chunks.

        A gated FFN is a sum over its intermediate neurons, so for chunks c,

            y = sum_c down_c( act(gate_c x) * up_c x )

        Chunk 0 holds neurons [0, w_min); each later chunk holds the next range
        up to the largest width. Burning deletes trailing chunks: every chunk is
        its own allocation, so this frees whole allocator segments and copies
        nothing (docs/VRAM_FINDINGS.md: the only layout with 100% KV reuse).
        """

        def __init__(self, mlp, bounds: list[int]) -> None:
            super().__init__()
            self.act_fn = mlp.act_fn
            self.chunks = nn.ModuleList()
            for start, end in itertools.pairwise(bounds):
                chunk = nn.Module()
                chunk.gate = _slice_linear(mlp.gate_proj, start, end, rows=True)
                chunk.up = _slice_linear(mlp.up_proj, start, end, rows=True)
                chunk.down = _slice_linear(mlp.down_proj, start, end, rows=False)
                self.chunks.append(chunk)
            down_bias = mlp.down_proj.bias
            self.down_bias = (
                None if down_bias is None
                else nn.Parameter(down_bias.detach().clone(), requires_grad=False)
            )
            self.bounds = list(bounds)

        @property
        def width(self) -> int:
            return self.bounds[len(self.chunks)]

        def forward(self, x):
            out = None
            for c in self.chunks:
                y = c.down(self.act_fn(c.gate(x)) * c.up(x))
                out = y if out is None else out + y
            return out if self.down_bias is None else out + self.down_bias

    return NestedMLP


def _slice_linear(lin, start: int, end: int, *, rows: bool):
    """New bias-free-or-not Linear holding a copy of one neuron range."""
    nn = _require_torch().nn
    w = lin.weight.detach()
    if rows:  # gate/up: neurons are output rows
        weight = w[start:end].clone()
        bias = None if lin.bias is None else lin.bias.detach()[start:end].clone()
    else:  # down: neurons are input columns; its bias is added once, not per chunk
        weight = w[:, start:end].contiguous()
        bias = None
    out = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                    device="meta")
    out.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        out.bias = nn.Parameter(bias, requires_grad=False)
    return out


def _pack_level(slots, device) -> None:
    """Give one burn level a single allocation; its parameters become views of it.

    `slots` is a list of (module, attribute, source tensor). Packing matters
    because the caching allocator puts tensors of 1-10 MiB into shared 20 MiB
    segments: freed separately, small chunks leave holes beside kept weights
    that never return to the driver, so `memory_allocated()` drops but the
    cache cannot use the bytes. One buffer per level is freed whole. Views are
    safe here, unlike the view trap, because the entire buffer is dropped at
    once: nothing kept shares its storage.
    """
    torch = _require_torch()
    align = 64  # elements; keeps every view 128-byte aligned for the GEMM kernels
    by_dtype: dict = {}
    for slot in slots:
        by_dtype.setdefault(slot[2].dtype, []).append(slot)
    for dtype, group in by_dtype.items():
        sizes = [-(-t.numel() // align) * align for _, _, t in group]
        flat = torch.empty(sum(sizes), dtype=dtype, device=device)
        offset = 0
        for (module, attr, src), size in zip(group, sizes):
            view = flat[offset:offset + src.numel()].view(src.shape)
            view.copy_(src)
            setattr(module, attr, torch.nn.Parameter(view, requires_grad=False))
            offset += size


def _find_gated_mlps(model) -> list[tuple[object, str, object]]:
    """(parent, attribute, mlp) for every gate/up/down FFN, in layer order."""
    nn = _require_torch().nn
    found = []
    for parent in model.modules():
        for attr, child in parent.named_children():
            if all(isinstance(getattr(child, p, None), nn.Linear)
                   for p in ("gate_proj", "up_proj", "down_proj")):
                if not callable(getattr(child, "act_fn", None)):
                    raise ValueError(f"FFN {type(child).__name__} has no act_fn")
                found.append((parent, attr, child))
    if not found:
        raise ValueError("no gated FFNs (gate_proj/up_proj/down_proj) found")
    return found


def _cache_bytes(cache) -> int:
    if cache is None:
        return 0
    layers = getattr(cache, "layers", None)
    if layers is not None:  # transformers >= 4.56
        tensors = [t for layer in layers for t in (layer.keys, layer.values)]
    else:
        tensors = [*cache.key_cache, *cache.value_cache]
    return sum(t.nbytes for t in tensors if t is not None and hasattr(t, "nbytes"))


class HFElasticModel(ElasticModel):
    """Real backend over a Hugging Face causal LM with MatFormer-style FFN nesting.

    Every configuration keeps a *prefix* of each FFN's intermediate neurons;
    attention is untouched, so the KV layout is shared across sizes and REUSE
    is mechanically possible. Nemotron Elastic (Mamba state) is not supported.

    Split-at-load layout (proved in docs/VRAM_FINDINGS.md):
        At construction each FFN is replaced by a NestedMLP whose neuron ranges
        are separate tensors, *before* the model moves to the device. While a
        larger configuration is resident its FFNs run one matmul per chunk and
        sum; `switch_to` deletes the dropped chunks, which frees exactly their
        bytes with no copy and no transient allocation. Slicing views frees
        nothing, and cloning the kept slice at switch time strands 2-9% of the
        freed memory in fragmented segments.

    Pass the model on CPU so the chunks are allocated once on the device. A
    model already on the device works but fragments it during the split.

    Parameters
    ----------
    model:
        A transformers causal LM (any model whose FFNs have gate_proj, up_proj,
        down_proj and act_fn: Llama, Qwen, Gemma, ...).
    widths:
        FFN width of each configuration, largest first. Each entry is one int
        for every layer, or a per-layer sequence (MatFormer mix-n-match). Widths
        must not increase from one configuration to the next in any layer.
    names, qualities:
        For the ModelConfig of each configuration. Qualities default to the
        width fraction, a placeholder until Experiment 1 measures them.
    exclude_layers:
        FFN indices (in layer order, i.e. decoder layer index) kept at a fixed
        width: their original module is left untouched and never burned, and
        their entries in `widths` are ignored. Needed for Gemma 3n's layers
        0-9, whose activation sparsity thresholds over *all* neurons, so the FFN
        no longer splits into a sum over chunks.
    kv_cache, block_tokens:
        "dynamic" uses HF's DynamicCache. "block" uses gei.kvcache.BlockCache:
        full-attention layers grow in fixed blocks of `block_tokens` tokens with
        no recopying, and attention runs block by block ("gei_blocked"). Use
        "block" for range-gain runs: under a full GPU, DynamicCache converted
        49-72% of burned weight bytes into cache on toy shapes and 97.2% on
        Qwen3-8B at batch 1 (docs/VRAM_FINDINGS.md, docs/RANGE_GAIN.md).
    restorable:
        Keep a host copy of the droppable chunks (made at load, so the burn is
        still a pure delete) so `reset()` can regrow without reloading the
        checkpoint. Costs host RAM equal to the largest-minus-smallest weights.
    temperature, seed:
        Sampling; temperature 0 is greedy.
    """

    def __init__(
        self,
        model,
        widths: Sequence[int | Sequence[int]],
        *,
        names: Sequence[str] | None = None,
        qualities: Sequence[float] | None = None,
        device: str | None = None,
        exclude_layers: Iterable[int] = (),
        restorable: bool = False,
        kv_cache: str = "dynamic",
        block_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        torch = _require_torch()
        found = _find_gated_mlps(model)
        n_layers = len(found)
        excluded = sorted(set(exclude_layers))
        if any(not 0 <= i < n_layers for i in excluded):
            raise ValueError(f"exclude_layers {excluded} outside 0..{n_layers - 1}")
        self.burnable_layers = [i for i in range(n_layers) if i not in excluded]
        if not self.burnable_layers:
            raise ValueError("every layer is excluded; nothing to burn")
        per_layer = [
            [int(w)] * n_layers if isinstance(w, (int, np.integer)) else [int(x) for x in w]
            for w in widths
        ]
        if len(per_layer) < 2:
            raise ValueError("need at least two widths to burn between")
        for i, ws in enumerate(per_layer):
            if len(ws) != n_layers:
                raise ValueError(f"widths[{i}] has {len(ws)} layers, model has {n_layers}")
        # From here on, "layer" indexes the burnable FFNs only.
        mlps = [found[i] for i in self.burnable_layers]
        per_layer = [[ws[i] for i in self.burnable_layers] for ws in per_layer]
        for layer, (_, _, mlp) in enumerate(mlps):
            if getattr(mlp, "activation_sparsity", 0):
                raise NotImplementedError(
                    f"FFN {self.burnable_layers[layer]} uses activation sparsity, "
                    "which thresholds over all neurons, so it cannot be chunked; "
                    "pass it in exclude_layers"
                )
            col = [ws[layer] for ws in per_layer]
            full = mlp.gate_proj.out_features
            if not (0 < col[-1] and col[0] <= full):
                raise ValueError(f"layer {self.burnable_layers[layer]}: widths {col} outside (0, {full}]")
            if any(b > a for a, b in itertools.pairwise(col)):
                raise ValueError(
                    f"layer {self.burnable_layers[layer]}: widths {col} increase"
                )

        nested_cls = _nested_mlp_class()
        self._mlps = []
        del found, mlp
        # Drop each original FFN as soon as its chunked copy replaces it, so the
        # split holds at most one extra layer: holding all originals until the
        # end costs the full FFN size again in host RAM (4.7 GB on Gemma 3n).
        for layer in range(len(mlps)):
            parent, attr, original = mlps[layer]
            mlps[layer] = None
            bounds = [0, *sorted({ws[layer] for ws in per_layer})]
            nested = nested_cls(original, bounds)
            setattr(parent, attr, nested)
            del original
            self._mlps.append(nested)
        self._widths = per_layer
        n_configs = len(per_layer)

        # Host copies of every chunk the smallest configuration drops.
        self._spare = None
        if restorable:
            self._spare = [
                [{k: v.detach().to("cpu", copy=True) for k, v in c.state_dict().items()}
                 for c in nested.chunks[self._keep(n_configs - 1, layer):]]
                for layer, nested in enumerate(self._mlps)
            ]

        # Take the droppable chunks out before the move, so the kept weights go
        # to the device on their own and each burn level lands in one buffer.
        model.requires_grad_(False).eval()
        target = (torch.device(device) if device is not None
                  else next(model.parameters()).device)
        staged = []
        for index in range(1, n_configs):
            slots = []
            for chunk in self._level_chunks(index):
                for lin in (chunk.gate, chunk.up, chunk.down):
                    for attr in ("weight", "bias"):
                        t = getattr(lin, attr)
                        if t is not None:
                            slots.append((lin, attr, t.detach()))
                            setattr(lin, attr, torch.nn.Parameter(
                                torch.empty(0, dtype=t.dtype), requires_grad=False))
            staged.append(slots)
        model.to(target)
        for slots in staged:
            _pack_level(slots, target)
        del staged
        self.model = model
        self.device = target

        if kv_cache not in ("dynamic", "block"):
            raise ValueError(f"kv_cache must be 'dynamic' or 'block', not {kv_cache!r}")
        self._kv_cache, self._block_tokens = kv_cache, block_tokens
        if kv_cache == "block":
            from .kvcache import register

            name = register()
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation(name)
            else:  # pragma: no cover - older transformers
                model.config._attn_implementation = name

        total = sum(t.nbytes for t in model.parameters()) + sum(
            t.nbytes for t in model.buffers()
        )
        sizes = [total - self._dropped_bytes(i) for i in range(len(per_layer))]
        names = list(names) if names else [f"cfg{i}" for i in range(len(per_layer))]
        if qualities is None:
            top = sum(per_layer[0])
            qualities = [sum(ws) / top for ws in per_layer]
        self._configs = [
            ModelConfig(n, int(b), float(q)) for n, b, q in zip(names, sizes, qualities)
        ]
        self._index = 0
        self._temperature = temperature
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._seed = seed
        self._cache = None
        self._rows: list[list[int]] = []
        self._pads: list[int] = []

    @classmethod
    def from_pretrained(
        cls, path: str, widths: Sequence[int | Sequence[int]], *, device: str = "cuda",
        dtype=None, **kwargs,
    ) -> HFElasticModel:
        """Load a local checkpoint on CPU, split it, then move it to `device`.

        Never downloads: local_files_only=True.
        """
        torch = _require_torch()
        from transformers import AutoModelForCausalLM

        from gei.revisions import pinned

        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype or torch.bfloat16, local_files_only=True,
            revision=pinned(path),
        )
        return cls(model, widths, device=device, **kwargs)

    # -- configuration bookkeeping -------------------------------------------

    def _keep(self, index: int, layer: int) -> int:
        """Number of chunks configuration `index` keeps in `layer`."""
        return self._mlps[layer].bounds.index(self._widths[index][layer])

    def _level_chunks(self, index: int) -> list:
        """Chunks dropped on the move from configuration index - 1 to index."""
        return [
            c
            for layer, nested in enumerate(self._mlps)
            for c in nested.chunks[self._keep(index, layer):self._keep(index - 1, layer)]
        ]

    def _dropped_bytes(self, index: int) -> int:
        """Bytes of FFN chunks configuration `index` does not keep."""
        dropped = 0
        for layer, nested in enumerate(self._mlps):
            for c in nested.chunks[self._keep(index, layer):]:
                dropped += sum(p.nbytes for p in c.parameters())
        return dropped

    @property
    def configs(self) -> list[ModelConfig]:
        return self._configs

    @property
    def current_index(self) -> int:
        return self._index

    # -- decoding ------------------------------------------------------------
    #
    # Sequences are decoded in lockstep as a left-padded batch; single-sequence
    # use is a batch of one. Invariant: the cache covers every column but the
    # last, and the last column holds each row's pending token. step_batch()
    # feeds that column, so every emitted token comes from the model resident
    # at that moment, reading a cache possibly written by its parent.

    @property
    def _width(self) -> int:
        return len(self._rows[0]) + self._pads[0]

    def _layout(self):
        """Left-padded ids, attention mask and position ids for all columns."""
        torch = _require_torch()
        pad_id = getattr(self.model.config, "pad_token_id", None) or 0
        ids = torch.tensor(
            [[pad_id] * p + row for p, row in zip(self._pads, self._rows)],
            dtype=torch.long, device=self.device,
        )
        cols = torch.arange(self._width, device=self.device)[None, :]
        pads = torch.tensor(self._pads, device=self.device)[:, None]
        mask = (cols >= pads).long()
        positions = (cols - pads).clamp(min=0)
        return ids, mask, positions

    def _forward(self, start: int, end: int):
        """Run columns [start, end) against the cache; return last-column logits."""
        torch = _require_torch()
        ids, mask, positions = self._layout()
        if self._cache is None and self._kv_cache == "block":
            from .kvcache import BlockCache

            self._cache = BlockCache(self.model.config, self._block_tokens)
        with torch.inference_mode():
            out = self.model(
                input_ids=ids[:, start:end], attention_mask=mask[:, :end],
                position_ids=positions[:, start:end], past_key_values=self._cache,
                use_cache=True,
                # Only the last column is used. Full logits for a 16k-token
                # prefill would be 16k x vocab (8.6 GB on Gemma 3n).
                logits_to_keep=1,
            )
        self._cache = out.past_key_values
        return out.logits[:, -1]

    def prefill_batch(self, prompts: list[list[int]]) -> None:
        """Start len(prompts) sequences decoded together; lengths may differ."""
        if not prompts or any(not p for p in prompts):
            raise ValueError("need at least one prompt, each with at least one token")
        self._cache = None
        longest = max(len(p) for p in prompts)
        self._rows = [list(p) for p in prompts]
        self._pads = [longest - len(p) for p in prompts]
        if longest > 1:
            self._forward(0, longest - 1)

    def prefill(self, prompt_token_ids: list[int]) -> None:
        self.prefill_batch([prompt_token_ids])

    def step_batch(self) -> list[StepResult]:
        """One token for every sequence."""
        torch = _require_torch()
        if not self._rows:
            raise RuntimeError("call prefill() or prefill_batch() first")
        logits = self._forward(self._width - 1, self._width).float()
        logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        if self._temperature > 0:
            probs = torch.softmax(logits / self._temperature, dim=-1)
            tokens = torch.multinomial(probs, 1, generator=self._generator)[:, 0]
        else:
            tokens = logits.argmax(dim=-1)
        tokens = tokens.tolist()
        for row, token in zip(self._rows, tokens):
            row.append(token)
        return [
            StepResult(token_id=t, logprobs=lp, config_index=self._index)
            for t, lp in zip(tokens, logprobs)
        ]

    def step(self) -> StepResult:
        if self.batch_size != 1:
            raise RuntimeError("step() decodes one sequence; use step_batch()")
        return self.step_batch()[0]

    def switch_to(self, index: int, kv_policy: str = KVPolicy.REUSE) -> None:
        if index < self._index:
            raise ValueError("cannot grow mid-task: burning is irreversible")
        if index >= len(self._configs):
            raise IndexError("configuration index out of range")
        if index == self._index:
            return
        if kv_policy not in (KVPolicy.REUSE, KVPolicy.RECOMPUTE, KVPolicy.DROP):
            raise ValueError(f"unknown kv_policy {kv_policy!r}")
        for layer, nested in enumerate(self._mlps):
            del nested.chunks[self._keep(index, layer):]
        self._index = index
        if kv_policy != KVPolicy.REUSE:
            self._cache = None  # free the parent's cache before anything else
        gc.collect()
        torch = _require_torch()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if kv_policy == KVPolicy.RECOMPUTE and self._rows and self._width > 1:
            self._forward(0, self._width - 1)
        elif kv_policy == KVPolicy.DROP:
            self._rows = [row[-1:] for row in self._rows]
            self._pads = [0] * len(self._rows)

    def kv_bytes(self) -> int:
        """Bytes the cache holds; for "block", every allocated block, full or not."""
        if self._kv_cache == "block":
            from .kvcache import cache_nbytes

            return cache_nbytes(self._cache)
        return _cache_bytes(self._cache)

    @property
    def batch_size(self) -> int:
        """Sequences currently being decoded (0 before prefill)."""
        return len(self._rows)

    @property
    def batch_token_ids(self) -> list[list[int]]:
        """Prompt plus generated tokens so far, one list per sequence."""
        return [list(row) for row in self._rows]

    @property
    def token_ids(self) -> list[int]:
        """Prompt plus generated tokens so far (single-sequence use)."""
        if self.batch_size != 1:
            raise RuntimeError("token_ids is for one sequence; use batch_token_ids")
        return list(self._rows[0])

    def reset(self) -> None:
        """Clear the context and, if restorable, regrow to the largest config."""
        torch = _require_torch()
        self._cache = None
        self._rows, self._pads = [], []
        self._generator.manual_seed(self._seed)
        if self._index == 0:
            return
        if self._spare is None:
            raise RuntimeError(
                "weights were burned; construct with restorable=True or reload"
            )
        nn = torch.nn
        last = len(self._configs) - 1
        # Regrow level by level, most recently burned first, so every layer's
        # chunks are appended in neuron order and each level is packed again.
        for index in range(self._index, 0, -1):
            slots = []
            for layer, nested in enumerate(self._mlps):
                base = self._keep(last, layer)  # spare[j] is chunk number base + j
                for pos in range(self._keep(index, layer), self._keep(index - 1, layer)):
                    state = self._spare[layer][pos - base]
                    chunk = nn.Module()
                    for name in ("gate", "up", "down"):
                        w = state[f"{name}.weight"]
                        b = state.get(f"{name}.bias")
                        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None,
                                        device="meta")
                        slots.append((lin, "weight", w))
                        if b is not None:
                            slots.append((lin, "bias", b))
                        setattr(chunk, name, lin)
                    nested.chunks.append(chunk)
            _pack_level(slots, self.device)
        self._index = 0
