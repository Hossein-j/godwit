# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Engineering test of evict.py on a tiny random Gemma 3n (no download needed).

Reference: the full, un-evicted cache with an attention mask that hides
exactly the entries the evictor dropped (per KV head, applied to the growing
layers and to the layers that share their cache). Eviction must give the same
next-token distribution as this reference at every step, up to float error.
Not part of the pre-registration; it checks the implementation.

Run: .venv/bin/python scripts/test_evict.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evict import Evictor, growing_layers

from transformers import AttentionInterface, DynamicCache, Gemma3nForCausalLM, Gemma3nTextConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward

EXTRA: dict[int, torch.Tensor] = {}  # source layer -> bool [H, kv_len] (True = attend)


def ref_attention(module, query, key, value, attention_mask, **kw):
    src = module.kv_shared_layer_index if module.is_kv_shared_layer else module.layer_idx
    if src in EXTRA:
        keep = EXTRA[src][None, :, None, :].to(query.device)  # [1, H, 1, kv]
        if attention_mask is None:
            attention_mask = keep
        else:
            am = attention_mask[..., -key.shape[-2]:]
            attention_mask = (am if am.dtype == torch.bool else am == 0) & keep
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("ref_masked", ref_attention)


def tiny(dtype):
    n = 10
    cfg = Gemma3nTextConfig(
        vocab_size=256, vocab_size_per_layer_input=256, hidden_size=64,
        hidden_size_per_layer_input=16, intermediate_size=128, num_hidden_layers=n,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, sliding_window=16,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 2,
        num_kv_shared_layers=4, laurel_rank=8, activation_sparsity_pattern=[0.0] * n,
        altup_num_inputs=2)
    torch.manual_seed(0)
    m = Gemma3nForCausalLM(cfg).to("cuda", dtype).eval()
    return m, cfg


@torch.inference_mode()
def run(policy, dtype, n_prompt=60, steps=40, budget=24):
    model, cfg = tiny(dtype)
    groups = cfg.num_attention_heads // cfg.num_key_value_heads
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 256, (n_prompt + steps,), generator=g).tolist()

    model.set_attn_implementation("sdpa")
    ev = Evictor(cfg, policy, budget, sinks=2, window=8, kernel=3, reserve=6)
    cache = DynamicCache(config=cfg)
    ev.attach(model)
    model(input_ids=torch.tensor([ids[:n_prompt]], device="cuda"),
          position_ids=torch.arange(n_prompt, device="cuda")[None], past_key_values=cache,
          use_cache=True, logits_to_keep=1)
    full = copy.deepcopy(cache)
    full_ref = copy.deepcopy(cache)
    ev.after_prefill(cache)

    worst, agree, vs_full = 0.0, 0, 0.0
    for t in range(steps):
        pos = n_prompt + t
        x = torch.tensor([[ids[pos]]], device="cuda")
        p = torch.tensor([[pos]], device="cuda")
        # Reference mask from the evictor's positions before this step, plus this token.
        EXTRA.clear()
        for i in ev.layers:
            kept = torch.zeros(cfg.num_key_value_heads, pos + 1, dtype=torch.bool, device="cuda")
            kept.scatter_(1, ev.positions[i], True)
            kept[:, pos] = True
            EXTRA[i] = kept.repeat_interleave(groups, dim=0)
        model.set_attn_implementation("ref_masked")
        ref = model(input_ids=x, position_ids=p, past_key_values=full_ref, use_cache=True,
                    logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        model.set_attn_implementation("sdpa")
        out = model(input_ids=x, position_ids=p, past_key_values=cache, use_cache=True,
                    logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        unmasked = model(input_ids=x, position_ids=p, past_key_values=full, use_cache=True,
                         logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        ev.after_step(cache, pos)
        worst = max(worst, (ref - out).abs().max().item())
        vs_full = max(vs_full, (unmasked - out).abs().max().item())
        agree += int(ref.argmax() == out.argmax())
        n_cache = cache.layers[ev.layers[0]].keys.shape[-2]
        assert policy == "none" or n_cache <= budget, (policy, t, n_cache)
    return worst, agree, steps, ev.evictions, vs_full


@torch.inference_mode()
def runner_path(policy, n_prompt=60, steps=40, budget=24):
    """run_eviction.decode_evict goes through HFElasticModel's own forward
    (left-padded layout, attention mask over all columns). Its greedy tokens
    must equal a direct greedy loop over the verified eviction path above."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from gei.elastic import HFElasticModel
    from run_eviction import decode_evict

    model, cfg = tiny(torch.float32)
    model.to("cpu")
    m = HFElasticModel(model, [128, 64], device="cuda")
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(3, 256, (n_prompt,), generator=g).tolist()
    ev = Evictor(cfg, policy, budget, sinks=2, window=8, kernel=3, reserve=6)
    via_runner, _ = decode_evict(m, prompt, stop=set(), max_new=steps, ev=ev)

    ev2 = Evictor(cfg, policy, budget, sinks=2, window=8, kernel=3, reserve=6)
    cache = DynamicCache(config=cfg)
    model.set_attn_implementation("sdpa")
    ev2.attach(model)
    model(input_ids=torch.tensor([prompt[:-1]], device="cuda"),
          position_ids=torch.arange(n_prompt - 1, device="cuda")[None], past_key_values=cache,
          use_cache=True, logits_to_keep=1)
    ev2.after_prefill(cache)
    direct, tok, pos = [], prompt[-1], n_prompt - 1
    for _ in range(steps):
        logits = model(input_ids=torch.tensor([[tok]], device="cuda"),
                       position_ids=torch.tensor([[pos]], device="cuda"), past_key_values=cache,
                       use_cache=True, logits_to_keep=1).logits[0, -1]
        ev2.after_step(cache, pos)
        tok, pos = int(logits.argmax()), pos + 1
        direct.append(tok)
    return via_runner == direct, ev.evictions, ev2.evictions


if __name__ == "__main__":
    ok = True
    for policy in ("none", "stream", "snap"):
        same, e1, e2 = runner_path(policy)
        ok &= same and e1 == e2
        print(f"runner path    {policy:6s} tokens identical to direct loop: {same}  "
              f"evicted {e1}/{e2}  {'PASS' if same and e1 == e2 else 'FAIL'}")
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 0.1)):
        for policy in ("none", "stream", "snap"):
            worst, agree, steps, ev, vs_full = run(policy, dtype)
            # vs ref must be tiny; vs the plain full cache it must be large when
            # something was evicted (otherwise the test would be vacuous).
            good = worst < tol and (policy == "none" or vs_full > 10 * worst)
            ok &= good
            print(f"{str(dtype):15s} {policy:6s} vs masked ref {worst:.2e}  vs full cache "
                  f"{vs_full:.2e}  argmax agree {agree}/{steps}  evicted {ev:4d}  "
                  f"{'PASS' if good else 'FAIL'}")
    print("growing layers in the tiny model:", growing_layers(tiny(torch.float32)[1]))
    raise SystemExit(0 if ok else 1)
