# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Triton decode attention over a list of separately allocated KV blocks.

The block cache (gei.kvcache) never concatenates K and V, so attention has to
read every block where it lies. The reference path loops over blocks in Python,
several kernels per block, which is far slower than fused SDPA at long context.
This kernel reads all blocks in one launch per layer, from a table of block
addresses: program (block i, batch-head) computes a partial softmax over its
block for the query heads sharing that KV head, and a small vectorised merge
combines the partials (the flash-decoding reduction). Exact up to fp32
accumulation order.

Decode only: one query token, no mask (no padding), no softcapping. Anything
else uses the reference path in gei.kvcache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _partial_kernel(
    q_ptr, k_table, v_table, m_out, l_out, o_out,
    length, scale, n_bh,
    BLOCK_TOKENS: tl.constexpr, TILE: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr, GROUPS_PAD: tl.constexpr, DTYPE: tl.constexpr,
):
    blk = tl.program_id(0)
    bh = tl.program_id(1)  # batch * kv_heads + kv_head
    g = tl.arange(0, GROUPS_PAD)
    d = tl.arange(0, HEAD_DIM)
    g_ok = g < GROUPS
    q = tl.load(q_ptr + (bh * GROUPS + g)[:, None] * HEAD_DIM + d[None, :],
                mask=g_ok[:, None], other=0.0)
    q = (q.to(tl.float32) * scale).to(DTYPE)

    k_base = tl.load(k_table + blk).to(tl.pointer_type(DTYPE))
    v_base = tl.load(v_table + blk).to(tl.pointer_type(DTYPE))
    head_off = bh * BLOCK_TOKENS * HEAD_DIM  # block layout [batch, kv_heads, tokens, dim]
    used = tl.minimum(length - blk * BLOCK_TOKENS, BLOCK_TOKENS)

    m = tl.full((GROUPS_PAD,), float("-inf"), tl.float32)
    lsum = tl.zeros((GROUPS_PAD,), tl.float32)
    acc = tl.zeros((GROUPS_PAD, HEAD_DIM), tl.float32)
    for start in range(0, BLOCK_TOKENS, TILE):
        t = start + tl.arange(0, TILE)
        t_ok = t < used
        offs = head_off + t[:, None] * HEAD_DIM + d[None, :]
        k = tl.load(k_base + offs, mask=t_ok[:, None], other=0.0)
        v = tl.load(v_base + offs, mask=t_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee")  # [GROUPS_PAD, TILE]
        s = tl.where(t_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - safe[:, None])
        corr = tl.exp(m - safe)
        lsum = lsum * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(DTYPE), v, input_precision="ieee")
        m = m_new

    # Only the real query-head rows are stored: the partials are transient memory
    # taken from the same envelope the cache grows into, so no padding is kept.
    row = (blk * n_bh + bh) * GROUPS + g
    tl.store(m_out + row, m, mask=g_ok)
    tl.store(l_out + row, lsum, mask=g_ok)
    tl.store(o_out + row[:, None] * HEAD_DIM + d[None, :], acc, mask=g_ok[:, None])


def _block_table(blocks: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in blocks], dtype=torch.int64,
                        device=blocks[0].device)


def supported(query: torch.Tensor, kv) -> bool:
    """Whether the kernel applies: CUDA, one query token, a supported dtype, and
    power-of-two head dim and block size (Triton tiles), each at least 16."""
    d, blk = query.shape[-1], kv.block_tokens
    return (query.is_cuda and query.shape[2] == 1 and query.dtype in _TL_DTYPES
            and d >= 16 and d & (d - 1) == 0 and blk >= 16 and blk & (blk - 1) == 0
            and kv.keys[0].dtype == query.dtype)


def paged_decode_attention(query: torch.Tensor, key, value, scale: float) -> torch.Tensor:
    """query [B, Hq, 1, D] against BlockedKV key/value; returns [B, 1, Hq, D]."""
    b, hq, _, d = query.shape
    hkv = key.keys[0].shape[1]
    groups = hq // hkv
    groups_pad = max(16, triton.next_power_of_2(groups))  # tl.dot needs M >= 16
    n_blocks = -(-key.length // key.block_tokens)
    n_bh = b * hkv
    k_table = _block_table(key.keys[:n_blocks])
    v_table = _block_table(value.values[:n_blocks])
    q = query.reshape(n_bh, groups, d).contiguous()
    m = torch.empty(n_blocks, n_bh, groups, device=query.device, dtype=torch.float32)
    lsum = torch.empty_like(m)
    o = torch.empty(n_blocks, n_bh, groups, d, device=query.device, dtype=torch.float32)
    # About 16 KiB of K per tile keeps K, V and their pipeline stages in shared memory.
    tile = max(16, min(key.block_tokens, 16384 // (d * query.element_size())))
    _partial_kernel[(n_blocks, n_bh)](
        q, k_table, v_table, m, lsum, o, key.length, scale, n_bh,
        BLOCK_TOKENS=key.block_tokens, TILE=tile, HEAD_DIM=d,
        GROUPS=groups, GROUPS_PAD=groups_pad, DTYPE=_TL_DTYPES[query.dtype],
        num_stages=2,
    )
    # Merge the per-block partials.
    m_all = m.amax(dim=0)
    w = torch.exp(m - m_all)  # empty blocks never occur: every listed block has tokens
    out = (o * w[..., None]).sum(0) / (lsum * w).sum(0)[..., None]
    out = out.reshape(b, hq, 1, d).to(query.dtype)
    return out.transpose(1, 2).contiguous()
