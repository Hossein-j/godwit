"""HFElasticModel on a tiny random Llama built in code: no checkpoint, no download.

Skipped entirely when torch or transformers is missing (the CPU-only install);
the memory test is additionally skipped without a CUDA device.
"""

import copy
import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gei.elastic import HFElasticModel, KVPolicy

FULL, SMALL = 256, 128


def tiny_llama(hidden=64, intermediate=FULL, layers=3, seed=0, max_positions=128):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=97, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=max_positions,
    )
    return transformers.LlamaForCausalLM(cfg).float().eval()


def tiny_gemma3n(seed=0):
    """Gemma 3n text stack in miniature: sparse FFNs in layers 0-1, KV sharing."""
    if not hasattr(transformers, "Gemma3nForCausalLM"):
        pytest.skip("transformers too old for Gemma 3n")
    torch.manual_seed(seed)
    cfg = transformers.Gemma3nTextConfig(
        vocab_size=97, vocab_size_per_layer_input=97, hidden_size=64,
        hidden_size_per_layer_input=8, intermediate_size=[FULL] * 4,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        activation_sparsity_pattern=[0.95, 0.95, 0.0, 0.0], num_kv_shared_layers=1,
        layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=8,
        laurel_rank=4, max_position_embeddings=128,
    )
    return transformers.Gemma3nForCausalLM(cfg).float().eval()


def truncated(model, width, layers=None):
    """Reference: the plain model with FFNs cut to their first `width` neurons."""
    ref = copy.deepcopy(model)
    for i, layer in enumerate(ref.model.layers):
        if layers is not None and i not in layers:
            continue
        mlp = layer.mlp
        for lin in (mlp.gate_proj, mlp.up_proj):
            lin.weight = torch.nn.Parameter(lin.weight.data[:width].clone())
            lin.out_features = width
        mlp.down_proj.weight = torch.nn.Parameter(
            mlp.down_proj.weight.data[:, :width].contiguous()
        )
        mlp.down_proj.in_features = width
    return ref


def logits(model, ids):
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0]


IDS = [3, 14, 15, 92, 65, 35, 89, 79]


def test_large_config_matches_the_original_model():
    """Two matmuls per projection must compute the same FFN as one."""
    base = tiny_llama()
    expected = logits(base, IDS)
    m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    torch.testing.assert_close(logits(m.model, IDS), expected, atol=1e-5, rtol=1e-5)


def test_switch_matches_the_truncated_model():
    base = tiny_llama()
    m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    m.switch_to(1)
    torch.testing.assert_close(
        logits(m.model, IDS), logits(truncated(base, SMALL), IDS), atol=1e-5, rtol=1e-5
    )


def test_configs_are_ordered_and_sized_by_the_dropped_chunks():
    m = HFElasticModel(tiny_llama(), [FULL, 192, SMALL], names=["L", "M", "S"])
    big, mid, small = m.configs
    assert [c.name for c in m.configs] == ["L", "M", "S"]
    # Per layer, a neuron costs one row of gate and up and one column of down.
    per_neuron = 3 * 64 * 4  # hidden * fp32 bytes
    assert big.weight_bytes - mid.weight_bytes == 3 * (FULL - 192) * per_neuron
    assert mid.weight_bytes - small.weight_bytes == 3 * (192 - SMALL) * per_neuron
    assert big.quality == 1.0 and small.quality == pytest.approx(0.5)


def test_switch_refuses_to_grow_and_rejects_out_of_range():
    m = HFElasticModel(tiny_llama(), [FULL, SMALL])
    m.switch_to(1)
    with pytest.raises(ValueError, match="irreversible"):
        m.switch_to(0)
    with pytest.raises(IndexError):
        m.switch_to(2)


def test_per_layer_widths_mix_and_match():
    m = HFElasticModel(tiny_llama(), [[FULL] * 3, [SMALL, FULL, 64]])
    m.switch_to(1)
    widths = [nested.width for nested in m._mlps]
    assert widths == [SMALL, FULL, 64]


def test_invalid_widths_are_rejected():
    with pytest.raises(ValueError, match="increase"):
        HFElasticModel(tiny_llama(), [SMALL, FULL])
    with pytest.raises(ValueError, match="outside"):
        HFElasticModel(tiny_llama(), [FULL * 2, SMALL])


def test_greedy_decoding_matches_the_original_and_logprobs_normalise():
    base = tiny_llama()
    m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    m.prefill(IDS)
    steps = [m.step() for _ in range(5)]
    assert np.exp(steps[0].logprobs).sum() == pytest.approx(1.0, abs=1e-4)
    assert all(s.config_index == 0 for s in steps)
    with torch.inference_mode():
        out = base.generate(torch.tensor([IDS]), max_new_tokens=5, do_sample=False)
    assert [s.token_id for s in steps] == out[0, len(IDS):].tolist()


def test_kv_policies():
    base = tiny_llama()
    reuse = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    reuse.prefill(IDS)
    reuse.step()
    before = reuse.kv_bytes()
    assert before > 0
    reuse.switch_to(1, kv_policy=KVPolicy.REUSE)
    assert reuse.kv_bytes() == before
    assert reuse.step().config_index == 1

    # RECOMPUTE rebuilds the cache with the small model: the next token must be
    # what the truncated model predicts from scratch on the same context.
    rec = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    rec.prefill(IDS)
    rec.step()
    rec.switch_to(1, kv_policy=KVPolicy.RECOMPUTE)
    assert rec.kv_bytes() == before
    ctx = rec.token_ids
    expected = int(logits(truncated(base, SMALL), ctx)[-1].argmax())
    assert rec.step().token_id == expected

    drop = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    drop.prefill(IDS)
    drop.switch_to(1, kv_policy=KVPolicy.DROP)
    assert drop.kv_bytes() == 0
    assert drop.token_ids == IDS[-1:]


def test_reset_regrows_only_when_restorable():
    base = tiny_llama()
    m = HFElasticModel(copy.deepcopy(base), [FULL, 192, SMALL], restorable=True)
    m.switch_to(2)
    m.reset()
    assert m.current_index == 0
    torch.testing.assert_close(logits(m.model, IDS), logits(base, IDS), atol=1e-5,
                               rtol=1e-5)

    fixed = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    fixed.switch_to(1)
    with pytest.raises(RuntimeError, match="restorable"):
        fixed.reset()


def test_sparse_layers_must_be_excluded():
    with pytest.raises(NotImplementedError, match="exclude_layers"):
        HFElasticModel(tiny_gemma3n(), [FULL, SMALL])


def test_excluded_layers_stay_at_fixed_width():
    base = tiny_gemma3n()
    m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL], exclude_layers=[0, 1])
    assert m.burnable_layers == [2, 3]
    torch.testing.assert_close(logits(m.model, IDS), logits(base, IDS), atol=1e-5,
                               rtol=1e-5)
    untouched = [m.model.model.layers[i].mlp for i in (0, 1)]

    m.switch_to(1)
    assert [m.model.model.layers[i].mlp for i in (0, 1)] == untouched
    assert all(mlp.gate_proj.out_features == FULL for mlp in untouched)
    assert [nested.width for nested in m._mlps] == [SMALL, SMALL]
    torch.testing.assert_close(
        logits(m.model, IDS), logits(truncated(base, SMALL, layers={2, 3}), IDS),
        atol=1e-5, rtol=1e-5,
    )
    # Only the two burnable layers count towards the weight drop.
    big, small = m.configs
    assert big.weight_bytes - small.weight_bytes == 2 * 3 * (FULL - SMALL) * 64 * 4


def test_gemma3n_decodes_across_a_switch_with_shared_kv():
    m = HFElasticModel(tiny_gemma3n(), [FULL, SMALL], exclude_layers=[0, 1])
    m.prefill(IDS)
    m.step()
    before = m.kv_bytes()
    m.switch_to(1, kv_policy=KVPolicy.REUSE)
    assert m.kv_bytes() == before
    step = m.step()
    assert step.config_index == 1 and np.isfinite(step.logprobs).all()


def test_exclude_layers_out_of_range():
    with pytest.raises(ValueError, match="exclude_layers"):
        HFElasticModel(tiny_llama(), [FULL, SMALL], exclude_layers=[3])
    with pytest.raises(ValueError, match="nothing to burn"):
        HFElasticModel(tiny_llama(), [FULL, SMALL], exclude_layers=[0, 1, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_switch_to_frees_exactly_the_dropped_weight_bytes_on_cuda():
    """The step-2 property: burning returns the dropped bytes, to the byte."""
    gc.collect()
    torch.cuda.empty_cache()
    m = HFElasticModel(
        tiny_llama(hidden=256, intermediate=2048, layers=4), [2048, 1024, 512],
        device="cuda",
    )
    m.prefill(IDS)
    m.step()
    for index in (1, 2):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        expected = m.configs[index - 1].weight_bytes - m.configs[index].weight_bytes
        m.switch_to(index, kv_policy=KVPolicy.REUSE)
        torch.cuda.synchronize()
        assert before - torch.cuda.memory_allocated() == expected
        assert expected == 3 * (1024 if index == 1 else 512) * 256 * 4 * 4
    del m
    gc.collect()
    torch.cuda.empty_cache()


# -- batched decoding ---------------------------------------------------------

PROMPTS = [IDS, [5, 7, 11], [42, 1, 2, 3, 9, 8, 13, 21, 34, 55, 89]]  # mixed lengths


def decode_alone(base, prompt, policy, widths=(FULL, SMALL), switch_at=3, n=6, **kw):
    """Reference: one sequence on its own, same switch at the same step."""
    m = HFElasticModel(copy.deepcopy(base), list(widths), **kw)
    m.prefill(prompt)
    out = []
    for t in range(n):
        if t == switch_at:
            m.switch_to(1, kv_policy=policy)
        out.append(m.step())
    return out


def decode_batch(base, policy, widths=(FULL, SMALL), switch_at=3, n=6, **kw):
    m = HFElasticModel(copy.deepcopy(base), list(widths), **kw)
    m.prefill_batch(PROMPTS)
    steps = []
    for t in range(n):
        if t == switch_at:
            m.switch_to(1, kv_policy=policy)
        steps.append(m.step_batch())
    return m, [[s[b] for s in steps] for b in range(len(PROMPTS))]


@pytest.mark.parametrize("policy", [KVPolicy.REUSE, KVPolicy.RECOMPUTE])
def test_batched_switch_matches_each_sequence_alone(policy):
    """Padding, masks and positions must not change any sequence's output."""
    base = tiny_llama()
    m, rows = decode_batch(base, policy)
    assert m.batch_size == len(PROMPTS)
    for prompt, row in zip(PROMPTS, rows):
        alone = decode_alone(base, prompt, policy)
        assert [s.token_id for s in row] == [s.token_id for s in alone]
        assert [s.config_index for s in row] == [0, 0, 0, 1, 1, 1]
        np.testing.assert_allclose(row[-1].logprobs, alone[-1].logprobs, atol=1e-4)


def test_batched_switch_on_gemma3n_matches_each_sequence_alone():
    base = tiny_gemma3n()
    _, rows = decode_batch(base, KVPolicy.REUSE, exclude_layers=[0, 1])
    for prompt, row in zip(PROMPTS, rows):
        alone = decode_alone(base, prompt, KVPolicy.REUSE, exclude_layers=[0, 1])
        assert [s.token_id for s in row] == [s.token_id for s in alone]


def test_batched_reuse_keeps_the_cache_and_drop_clears_it():
    base = tiny_llama()
    m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    m.prefill_batch(PROMPTS)
    m.step_batch()
    before = m.kv_bytes()
    m.switch_to(1, kv_policy=KVPolicy.REUSE)
    assert m.kv_bytes() == before
    m.step_batch()
    # One more column for every sequence: 3 layers x K,V x 2 heads x 16 dim x fp32.
    assert m.kv_bytes() - before == len(PROMPTS) * 3 * 2 * 2 * 16 * 4

    d = HFElasticModel(copy.deepcopy(base), [FULL, SMALL])
    d.prefill_batch(PROMPTS)
    d.step_batch()
    last = [row[-1] for row in d.batch_token_ids]
    d.switch_to(1, kv_policy=KVPolicy.DROP)
    assert d.kv_bytes() == 0
    assert d.batch_token_ids == [[t] for t in last]
    assert len(d.step_batch()) == len(PROMPTS)


def test_single_sequence_api_refuses_a_batch():
    m = HFElasticModel(tiny_llama(), [FULL, SMALL])
    m.prefill_batch(PROMPTS)
    with pytest.raises(RuntimeError, match="step_batch"):
        m.step()
    with pytest.raises(RuntimeError, match="batch_token_ids"):
        _ = m.token_ids
    with pytest.raises(ValueError):
        m.prefill_batch([IDS, []])


def _fill_device(ballast):
    """Allocate until nothing more fits, in shrinking chunks."""
    for chunk in (256 * 1024**2, 16 * 1024**2, 2 * 1024**2):
        try:
            while True:
                ballast.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
        except torch.OutOfMemoryError:
            pass


def _decode_until_oom(m):
    kv0 = m.kv_bytes()
    try:
        while True:
            m.step_batch()
    except torch.OutOfMemoryError:
        pass
    return m.kv_bytes() - kv0


def _claim_pages(page=64 * 1024):
    pages = []
    try:
        while True:
            pages.append(torch.empty(page, dtype=torch.uint8, device="cuda"))
    except torch.OutOfMemoryError:
        pass
    claimed = len(pages) * page
    del pages
    return claimed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("burn", [False, True])
def test_freed_bytes_become_usable_cache_under_batching(burn):
    """Under a tight envelope, a batch's cache can only grow into burned weights.

    36 layers with small (2 MiB) chunks: the shape where unpacked chunks failed,
    because the allocator shares 20 MiB segments between them and kept weights.
    Ballast fills the device after prefill; then:
      no burn -> the cache can barely grow;
      burn    -> memory_allocated drops by exactly the freed bytes, KV-sized
                 pages can claim all of them, and HF's own growing cache claims
                 most of them while decoding a batch of 64.
    """
    gc.collect()
    torch.cuda.empty_cache()
    m = HFElasticModel(
        tiny_llama(hidden=256, intermediate=4096, layers=36, max_positions=4096),
        [4096, 2048], device="cuda",
    )
    m.prefill_batch([IDS] * 64)
    m.step_batch()
    freed = m.configs[0].weight_bytes - m.configs[1].weight_bytes
    ballast = []
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _fill_device(ballast)
        if not burn:
            assert _decode_until_oom(m) < 0.05 * freed
            return

        before = torch.cuda.memory_allocated()
        m.switch_to(1, kv_policy=KVPolicy.REUSE)
        torch.cuda.synchronize()
        assert before - torch.cuda.memory_allocated() == freed
        # The backend's guarantee: every freed byte can become cache.
        assert _claim_pages() >= freed
        # HF's DynamicCache grows by torch.cat (old and new tensor live at once,
        # old blocks not reused) and SDPA adds per-step transients, so it
        # realises less: 75.6% measured on this shape (RTX 4090, torch 2.12).
        assert _decode_until_oom(m) >= 0.6 * freed
    finally:
        del ballast
        del m
        gc.collect()
        torch.cuda.empty_cache()


# -- block-allocated cache -------------------------------------------------------


@pytest.mark.parametrize("policy", [KVPolicy.REUSE, KVPolicy.RECOMPUTE])
@pytest.mark.parametrize("model", ["llama", "gemma3n"])
def test_block_cache_batched_switch_matches_each_sequence_alone(policy, model):
    """Same equivalence as the batched tests, through the block cache and its
    blocked attention; 4-token blocks so every sequence spans many blocks."""
    base = tiny_llama() if model == "llama" else tiny_gemma3n()
    kw = {} if model == "llama" else {"exclude_layers": [0, 1]}
    m, rows = decode_batch(base, policy, kv_cache="block", block_tokens=4, **kw)
    from gei.kvcache import BlockLayer

    assert any(isinstance(layer, BlockLayer) for layer in m._cache.layers)
    for prompt, row in zip(PROMPTS, rows):
        alone = decode_alone(base, prompt, policy, **kw)
        assert [s.token_id for s in row] == [s.token_id for s in alone]
        np.testing.assert_allclose(row[-1].logprobs, alone[-1].logprobs, atol=1e-4)


def test_block_cache_appends_without_recopying_and_is_not_presized():
    from gei.kvcache import BlockLayer

    m = HFElasticModel(tiny_llama(), [FULL, SMALL], kv_cache="block", block_tokens=4)
    m.prefill_batch(PROMPTS)  # longest prompt 11 tokens: 10 cached -> 3 blocks
    layer = m._cache.layers[0]
    assert isinstance(layer, BlockLayer)
    assert len(layer.key_blocks) == 3
    first = [b.data_ptr() for b in layer.key_blocks]
    for _ in range(7):
        m.step_batch()
    assert layer.get_seq_length() == 17 and len(layer.key_blocks) == 5
    assert [b.data_ptr() for b in layer.key_blocks[:3]] == first  # never moved
    # Allocated bytes are whole blocks: 3 layers x K,V x 5 blocks x [3, 2, 4, 16] fp32.
    assert m.kv_bytes() == 3 * 2 * 5 * (3 * 2 * 4 * 16 * 4)


def test_unknown_kv_cache_is_rejected():
    with pytest.raises(ValueError, match="kv_cache"):
        HFElasticModel(tiny_llama(), [FULL, SMALL], kv_cache="paged")


# -- regression: small chunks must be packed per burn level ---------------------


def _storage(p):
    return p.untyped_storage().data_ptr()


@pytest.mark.parametrize("restored", [False, True])
def test_each_burn_level_is_one_allocation_shared_with_nothing_kept(restored):
    """Structural guard for the small-chunk failure, runnable without CUDA.

    Freed separately, chunks of 1-10 MiB leave holes in shared allocator segments
    and the cache gets 0% of them although memory_allocated() drops correctly.
    So every burn level must live in exactly one storage that no kept parameter
    uses, including after reset() regrows the weights.
    """
    m = HFElasticModel(tiny_llama(), [FULL, 192, SMALL], restorable=True)
    if restored:
        m.switch_to(2)
        m.reset()
    level_storages = []
    for index in (1, 2):
        params = [p for c in m._level_chunks(index) for p in c.parameters()]
        storages = {_storage(p) for p in params}
        assert len(storages) == 1, f"level {index} spans {len(storages)} allocations"
        level_storages.append(storages.pop())
    dropped = {id(p) for i in (1, 2) for c in m._level_chunks(i) for p in c.parameters()}
    kept = {_storage(p) for p in m.model.parameters() if id(p) not in dropped}
    assert not kept & set(level_storages)
    assert len(set(level_storages)) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("chunk_mib", [0.5, 2, 8, 24])
def test_freed_chunks_are_claimable_at_every_size(chunk_mib):
    """Whatever the chunk size, every freed byte must be claimable under a full GPU.

    2 MiB chunks gave 0% before per-level packing; 24 MiB chunks always worked.
    """
    gc.collect()
    torch.cuda.empty_cache()
    hidden, layers = 256, 12
    half = int(chunk_mib * 1024**2 // (hidden * 4))  # neurons per fp32 chunk
    m = HFElasticModel(
        tiny_llama(hidden=hidden, intermediate=2 * half, layers=layers),
        [2 * half, half], device="cuda",
    )
    m.prefill(IDS)
    freed = m.configs[0].weight_bytes - m.configs[1].weight_bytes
    ballast = []
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _fill_device(ballast)
        before = _claim_pages()
        torch.cuda.empty_cache()
        m.switch_to(1)
        assert _claim_pages() - before >= freed
    finally:
        del ballast
        del m
        gc.collect()
        torch.cuda.empty_cache()


# -- the range-gain measurement: real cache growth into burned weights ----------

SHAPES = {"36-layer": (36, 4096), "gemma3n-like-4-kv-layers": (4, 32768)}


def _growth_into_burn(shape, cache, burn):
    layers, inter = SHAPES[shape]
    kw = {"kv_cache": cache} | ({"block_tokens": 16} if cache == "block" else {})
    gc.collect()
    torch.cuda.empty_cache()
    m = HFElasticModel(
        tiny_llama(hidden=256, intermediate=inter, layers=layers, max_positions=8192),
        [inter, inter // 2], device="cuda", **kw,
    )
    m.prefill_batch([IDS] * 64)
    m.step_batch()
    freed = m.configs[0].weight_bytes - m.configs[1].weight_bytes
    ballast = []
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _fill_device(ballast)
        if burn:
            m.switch_to(1, kv_policy=KVPolicy.REUSE)
        return _decode_until_oom(m), freed
    finally:
        del ballast
        del m
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("shape", list(SHAPES))
def test_block_cache_turns_freed_weights_into_cache(shape):
    """Net cache growth (burn minus no-burn) as a fraction of freed weight bytes.

    Measured on an RTX 4090, torch 2.12, batch 64, 16-token blocks
    (docs/VRAM_FINDINGS.md): block 100.0% / 94.3%, HF DynamicCache 72.2% / 49.0%
    for the 36-layer and 4-KV-layer shapes, identical over three runs. The block cache must convert nearly
    all of it and beat the copying cache on both shapes.
    """
    fractions = {}
    for cache in ("block", "dynamic"):
        base, _ = _growth_into_burn(shape, cache, burn=False)
        grew, freed = _growth_into_burn(shape, cache, burn=True)
        fractions[cache] = (grew - base) / freed
    print(f"\n{shape}: block {fractions['block']:.1%}, dynamic {fractions['dynamic']:.1%}")
    assert fractions["block"] >= 0.85
    assert fractions["block"] > fractions["dynamic"] + 0.1


# -- fused Triton decode over blocks ----------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("shape", [
    (1, 8, 4, 128, 512, 1000),   # Qwen3-8B heads, partial last block
    (3, 2, 2, 64, 16, 37),       # batch, small blocks
    (2, 1, 8, 256, 64, 130),     # Gemma-style head dim, one KV head
])
def test_paged_decode_matches_dense_attention(shape, dtype, tol):
    from gei import paged_decode
    from gei.kvcache import BlockedKV

    b, hkv, groups, d, blk, length = shape
    torch.manual_seed(0)
    n = -(-length // blk) + 1  # plus an unfilled spare block, as in a live cache
    ks = [torch.randn(b, hkv, blk, d, device="cuda", dtype=dtype) for _ in range(n)]
    vs = [torch.randn(b, hkv, blk, d, device="cuda", dtype=dtype) for _ in range(n)]
    kv = BlockedKV(ks, vs, length, blk)
    q = torch.randn(b, hkv * groups, 1, d, device="cuda", dtype=dtype)
    k = torch.cat(ks, 2)[:, :, :length].float()
    v = torch.cat(vs, 2)[:, :, :length].float()
    scores = torch.einsum("bhgd,bhtd->bhgt", q.float().view(b, hkv, groups, d), k) * d**-0.5
    dense = torch.einsum("bhgt,bhtd->bhgd", scores.softmax(-1), v)
    dense = dense.reshape(b, hkv * groups, 1, d).transpose(1, 2)
    out = paged_decode.paged_decode_attention(q, kv, kv, d**-0.5)
    torch.testing.assert_close(out.float(), dense, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_block_cache_on_cuda_uses_the_kernel_and_matches_dynamic():
    """End to end on CUDA: same tokens as HF's cache across a switch, and the
    fused kernel (not the Python loop) did the decode attention."""
    from gei import kvcache

    base = tiny_llama(max_positions=512)
    prompts = [IDS, IDS[::-1]]  # equal lengths: no padding mask, so the kernel applies
    out = {}
    for cache in ("dynamic", "block"):
        kw = {"kv_cache": cache, "block_tokens": 16} if cache == "block" else {}
        m = HFElasticModel(copy.deepcopy(base), [FULL, SMALL], device="cuda", **kw)
        m.prefill_batch(prompts)
        calls = kvcache.STATS["paged_decode_calls"]
        steps = []
        for t in range(40):  # crosses two 16-token block boundaries
            if t == 20:
                m.switch_to(1, kv_policy=KVPolicy.REUSE)
            steps.append(m.step_batch())
        out[cache] = [[s.token_id for s in step] for step in steps]
        if cache == "block":
            assert kvcache.STATS["paged_decode_calls"] - calls == 40 * 3  # steps x layers
    assert out["block"] == out["dynamic"]
