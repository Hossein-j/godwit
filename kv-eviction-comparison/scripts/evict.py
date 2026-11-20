# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""KV-cache eviction for Gemma 3n under a per-sequence token budget.

Design: docs/PREREG.md. Only the growing caches are evicted: the full-attention
layers that write their own KV (layer index < first KV-shared layer). Layers
that share a cache read `past_key_values.shared_layers`, which the source layer
rebuilds from its (evicted) cache on every forward pass. Sliding-window layers
are fixed-size and left alone.

Keys are stored after RoPE, so kept entries keep their original positions and
new tokens get their true positions (as in SnapKV and NVIDIA kvpress); the
model never sees re-numbered positions.

Policies, budget L tokens of growing cache per sequence:
  stream  StreamingLLM: keep the first `sinks` entries and the most recent
          L - sinks, after prefill and after every step.
  snap    SnapKV-style: after prefill, score prompt positions per KV head by the
          attention of the last `window` prompt queries (summed over the query
          heads of the group, max-pooled with `kernel`), keep the top
          L - window - reserve positions plus the window. During generation,
          once the cache exceeds L, evict the oldest generated entry.
  none    no eviction (the exact-zero check).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def growing_layers(config) -> list[int]:
    """Full-attention layers that write their own KV cache."""
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    return [i for i, t in enumerate(config.layer_types[:first_shared]) if t == "full_attention"]


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """t [1, kv_heads, n, d], idx [kv_heads, k] sorted -> [1, kv_heads, k, d]."""
    return torch.gather(t, 2, idx[None, :, :, None].expand(1, -1, -1, t.shape[-1])).contiguous()


@dataclass
class Evictor:
    """Holds the policy state for one sequence; call after prefill and each step."""

    config: object
    policy: str
    budget: int
    sinks: int = 4
    window: int = 32
    kernel: int = 7
    reserve: int = 256
    layers: list[int] = field(default_factory=list)
    kept_prompt: int | None = None      # snap: entries of the compressed prompt
    evictions: int = 0
    # Per layer, per KV head: the absolute positions held in the cache (for the
    # question-survival measure and for the reference test).
    positions: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.policy not in ("stream", "snap", "none"):
            raise ValueError(f"unknown policy {self.policy!r}")
        self.layers = growing_layers(self.config)
        self._queries: dict[int, torch.Tensor] = {}
        self._hooks = []

    # -- SnapKV: capture the observation-window queries during prefill --------

    def attach(self, model) -> None:
        """Register hooks that keep the last `window` queries of each growing
        layer during the next forward pass (prefill). snap only."""
        if self.policy != "snap":
            return
        from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb

        attn = {i: model.model.language_model.layers[i].self_attn if hasattr(model.model, "language_model")
                else model.model.layers[i].self_attn for i in self.layers}

        def make(i, mod):
            def hook(_m, args, kwargs):
                h = kwargs.get("hidden_states", args[0] if args else None)
                cos, sin = kwargs["position_embeddings"]
                w = min(self.window, h.shape[1])
                hs = h[:, -w:]
                q = mod.q_proj(hs).view(*hs.shape[:-1], -1, mod.config.head_dim)
                q = mod.q_norm(q)
                q = apply_rotary_pos_emb(q, cos[:, -w:], sin[:, -w:], unsqueeze_dim=2)
                self._queries[i] = q.transpose(1, 2).detach()  # [1, H, w, d]
            return hook

        for i, mod in attn.items():
            self._hooks.append(mod.register_forward_pre_hook(make(i, mod), with_kwargs=True))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # -- eviction ---------------------------------------------------------------

    def _layer(self, cache, i):
        return cache.layers[i]

    def after_prefill(self, cache) -> None:
        self.detach()
        n = self._layer(cache, self.layers[0]).keys.shape[-2]
        kv_heads = self._layer(cache, self.layers[0]).keys.shape[1]
        base = torch.arange(n, device=self._layer(cache, self.layers[0]).keys.device)
        self.positions = {i: base.expand(kv_heads, n).clone() for i in self.layers}
        if self.policy == "none":
            return
        if self.policy == "stream":
            self._stream(cache)
            return
        # snap
        keep_n = self.budget - self.reserve
        if n <= keep_n:
            self.kept_prompt = n
            self._queries.clear()
            return
        for i in self.layers:
            layer = self._layer(cache, i)
            k = layer.keys  # [1, kv, n, d]
            q = self._queries.pop(i)  # [1, H, w, d]
            w = q.shape[2]
            groups = q.shape[1] // k.shape[1]
            kk = k.repeat_interleave(groups, dim=1)  # [1, H, n, d]
            logits = torch.matmul(q.float(), kk.float().transpose(-1, -2))  # scaling 1.0 (Gemma 3n)
            # causal within the window: query j (absolute n - w + j) sees keys <= n - w + j
            qpos = torch.arange(n - w, n, device=k.device)[:, None]
            kpos = torch.arange(n, device=k.device)[None, :]
            logits = logits.masked_fill(kpos > qpos, float("-inf"))
            attn = logits.softmax(dim=-1)  # [1, H, w, n]
            score = attn.sum(dim=2).view(1, k.shape[1], groups, n).sum(dim=2)[0]  # [kv, n]
            prefix = score[:, : n - w]
            pooled = F.max_pool1d(prefix[:, None, :], self.kernel, stride=1,
                                  padding=self.kernel // 2)[:, 0, :]
            top = pooled.topk(keep_n - w, dim=-1).indices
            win = torch.arange(n - w, n, device=k.device).expand(k.shape[1], w)
            idx = torch.cat([top, win], dim=-1).sort(dim=-1).values  # [kv, keep_n]
            layer.keys, layer.values = _gather(layer.keys, idx), _gather(layer.values, idx)
            self.positions[i] = torch.gather(self.positions[i], 1, idx)
        self.kept_prompt = keep_n
        self.evictions += n - keep_n

    def after_step(self, cache, position: int) -> None:
        """Call after a step that appended the token at absolute `position`."""
        for i in self.layers:
            p = self.positions[i]
            self.positions[i] = torch.cat(
                [p, torch.full((p.shape[0], 1), position, device=p.device, dtype=p.dtype)], dim=1)
        if self.policy == "none":
            return
        if self.policy == "stream":
            self._stream(cache)
            return
        n = self._layer(cache, self.layers[0]).keys.shape[-2]
        if n <= self.budget:
            return
        drop = self.kept_prompt  # oldest generated entry
        for i in self.layers:
            layer = self._layer(cache, i)
            keep = torch.cat([torch.arange(0, drop), torch.arange(drop + 1, n)]).to(layer.keys.device)
            layer.keys = layer.keys.index_select(2, keep).contiguous()
            layer.values = layer.values.index_select(2, keep).contiguous()
            self.positions[i] = self.positions[i].index_select(1, keep)
        self.evictions += 1

    def _stream(self, cache) -> None:
        n = self._layer(cache, self.layers[0]).keys.shape[-2]
        if n <= self.budget:
            return
        dev = self._layer(cache, self.layers[0]).keys.device
        keep = torch.cat([torch.arange(0, self.sinks), torch.arange(n - (self.budget - self.sinks), n)]).to(dev)
        for i in self.layers:
            layer = self._layer(cache, i)
            layer.keys = layer.keys.index_select(2, keep).contiguous()
            layer.values = layer.values.index_select(2, keep).contiguous()
            self.positions[i] = self.positions[i].index_select(1, keep)
        self.evictions += n - self.budget

    def kept_fraction(self, start: int, end: int) -> float:
        """Mean over growing layers and KV heads of the share of absolute
        positions [start, end) still in the cache."""
        if end <= start:
            return float("nan")
        fr = []
        for p in self.positions.values():
            inside = ((p >= start) & (p < end)).sum(dim=1).float() / (end - start)
            fr.extend(inside.tolist())
        return sum(fr) / len(fr)
