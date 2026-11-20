# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Block-allocated KV cache that grows without recopying, plus blocked attention.

GEI's claim is that the cache grows *into* memory freed by burning weights
later in the task, so the cache must allocate incrementally and must not waste
the freed headroom. HF's DynamicCache fails the second condition: it grows each
layer by `torch.cat`, so the old and new tensors are live at once and old
blocks are not reused. Under a full GPU it converted 49-72% of freed bytes
into cache on toy shapes at batch 64, and 97.2% on Qwen3-8B at batch 1: the
loss is about one layer's share of the cache, so it is largest for models with
few KV-writing layers such as Gemma 3n (docs/VRAM_FINDINGS.md). A cache pre-sized to the maximum
context fails the first: it would claim the memory up front.

This cache keeps each full-attention layer as a list of fixed-size blocks of
`block_tokens` tokens. Appending writes into the current block; a new block is
allocated only when it is full; existing blocks are never copied or resized.
Because K and V are never one contiguous tensor, attention runs block by block
with an online softmax (the flash-decoding merge), registered with transformers
as the attention implementation "gei_blocked". Single-token decode on CUDA uses
a fused Triton kernel over a table of block addresses (gei.paged_decode); the
Python loop here is the reference and handles prefill, masks and softcapping. It is exact, and it handles
grouped-query attention natively, without materialising repeat_kv copies.

Sliding-window layers keep HF's DynamicSlidingWindowLayer: their cache stops
growing at the window, so it is bounded and plays no part in the range gain.
Gemma 3n's KV-shared layers read their source layer's cache as before.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import AttentionInterface

ATTN_IMPLEMENTATION = "gei_blocked"

# How often the fused Triton decode path ran (tests use it to prove the path is live).
STATS = {"paged_decode_calls": 0}


class BlockedKV:
    """Read-only view over a layer's blocks: the first `length` tokens."""

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor],
                 length: int, block_tokens: int) -> None:
        self.keys, self.values = keys, values
        self.length, self.block_tokens = length, block_tokens

    def spans(self):
        """(key block, value block, first token, tokens used) for every block."""
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            start = i * self.block_tokens
            used = min(self.block_tokens, self.length - start)
            if used <= 0:
                return
            yield k[:, :, :used], v[:, :, :used], start, used

    def to(self, device, *args, **kwargs) -> BlockedKV:
        """Gemma 3n's KV-shared layers call .to(device) on their source's cache."""
        if torch.device(device) != self.keys[0].device:
            raise NotImplementedError("BlockedKV does not move between devices")
        return self

    @property
    def shape(self):  # [batch, kv heads, length, head dim], like the tensor it replaces
        b, h, _, d = self.keys[0].shape
        return torch.Size((b, h, self.length, d))


class BlockLayer(DynamicLayer):
    """One full-attention layer's cache as fixed-size blocks, appended in place."""

    is_sliding = False

    def __init__(self, block_tokens: int) -> None:
        super().__init__()
        if block_tokens < 1:
            raise ValueError("block_tokens must be at least 1")
        self.block_tokens = block_tokens
        self.key_blocks: list[torch.Tensor] = []
        self.value_blocks: list[torch.Tensor] = []
        self.length = 0

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def _new_block(self, like: torch.Tensor) -> torch.Tensor:
        b, h, _, d = like.shape
        return torch.empty(b, h, self.block_tokens, d, dtype=like.dtype, device=like.device)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict[str, Any] | None = None) -> tuple[BlockedKV, BlockedKV]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        written, n = 0, key_states.shape[-2]
        while written < n:
            slot = self.length % self.block_tokens
            if slot == 0 and self.length // self.block_tokens == len(self.key_blocks):
                self.key_blocks.append(self._new_block(key_states))
                self.value_blocks.append(self._new_block(value_states))
            take = min(n - written, self.block_tokens - slot)
            blk = self.length // self.block_tokens
            self.key_blocks[blk][:, :, slot:slot + take] = key_states[:, :, written:written + take]
            self.value_blocks[blk][:, :, slot:slot + take] = value_states[:, :, written:written + take]
            written += take
            self.length += take
        view = BlockedKV(self.key_blocks, self.value_blocks, self.length, self.block_tokens)
        return view, view

    def get_seq_length(self) -> int:
        return self.length

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.length + cache_position.shape[0], 0

    @property
    def nbytes(self) -> int:
        """Bytes actually allocated, including the unfilled part of the last block."""
        return sum(t.nbytes for t in (*self.key_blocks, *self.value_blocks))

    def crop(self, max_length: int) -> None:  # pragma: no cover - not used by GEI
        raise NotImplementedError("BlockLayer does not support crop")


class BlockCache(DynamicCache):
    """DynamicCache layout (sliding / shared layers as HF does) with BlockLayer for
    every full-attention layer."""

    def __init__(self, config, block_tokens: int = 512) -> None:
        super().__init__(config=config)
        self.block_tokens = block_tokens
        self.layers = [
            BlockLayer(block_tokens) if type(layer) is DynamicLayer else layer
            for layer in self.layers
        ]
        if not self.layers:
            raise ValueError("BlockCache needs a config with per-layer structure")


def cache_nbytes(cache: Cache | None) -> int:
    """Allocated bytes of any HF cache, BlockCache included."""
    if cache is None:
        return 0
    total = 0
    for layer in cache.layers:
        if isinstance(layer, BlockLayer):
            total += layer.nbytes
        else:
            total += sum(t.nbytes for t in (layer.keys, layer.values)
                         if t is not None and hasattr(t, "nbytes"))
    return total


def blocked_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                              scaling=None, softcap=None, **kwargs):
    """Attention over a BlockedKV, block by block, merged with an online softmax.

    Plain tensors (sliding-window layers, or calls without a cache) are passed
    to HF's SDPA implementation unchanged.
    """
    if not isinstance(key, BlockedKV):
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if dropout:
        raise NotImplementedError("blocked attention is inference-only")
    b, hq, q_len, d = query.shape
    if attention_mask is None and softcap is None and query.is_cuda:
        from . import paged_decode

        if paged_decode.supported(query, key):
            STATS["paged_decode_calls"] += 1
            return paged_decode.paged_decode_attention(
                query, key, value, scaling if scaling is not None else d ** -0.5
            ), None
    hkv = key.keys[0].shape[1]
    groups = hq // hkv
    scale = scaling if scaling is not None else d ** -0.5
    q = query.float().view(b, hkv, groups, q_len, d) * scale
    kv_len = key.length
    if attention_mask is not None:
        mask = attention_mask.bool()[:, :, -q_len:, :]  # [b, 1, q, kv]
        mask = mask.view(mask.shape[0], 1, 1, q_len, mask.shape[-1])
    q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]

    m = torch.full((b, hkv, groups, q_len, 1), float("-inf"), device=query.device)
    denom = torch.zeros_like(m)
    acc = torch.zeros(b, hkv, groups, q_len, d, device=query.device)
    for (k, _, start, used), (_, v, _, _) in zip(key.spans(), value.spans()):
        scores = torch.einsum("bhgqd,bhnd->bhgqn", q, k.float())
        if softcap is not None:
            scores = softcap * torch.tanh(scores / softcap)
        if attention_mask is not None:
            allowed = mask[..., start:start + used]
        else:  # mask skipped by HF: causal only
            k_pos = torch.arange(start, start + used, device=query.device)[None, :]
            allowed = (k_pos <= q_pos).view(1, 1, 1, q_len, used)
        scores = scores.masked_fill(~allowed, float("-inf"))
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        p = torch.exp(scores - safe)
        correction = torch.exp(m - safe)
        denom = denom * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + torch.einsum("bhgqn,bhnd->bhgqd", p, v.float())
        m = m_new
    out = acc / denom.clamp(min=torch.finfo(acc.dtype).tiny)  # fully masked rows -> 0
    out = out.view(b, hq, q_len, d).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def register() -> str:
    """Register "gei_blocked" with transformers (idempotent); return its name.

    Global registration: HF's mask code looks the name up in the global mapping.
    """
    AttentionInterface.register(ATTN_IMPLEMENTATION, blocked_attention_forward)
    AttentionMaskInterface.register(ATTN_IMPLEMENTATION, sdpa_mask)
    return ATTN_IMPLEMENTATION
