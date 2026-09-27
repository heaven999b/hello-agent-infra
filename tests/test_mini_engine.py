"""mini_engine 核心正确性：引擎里的每个优化都不许改变 greedy 输出。

参照物是 reference_greedy：每生成一个 token 都把整条序列从头算一遍，不用任何缓存。
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_engine import (BlockAllocator, EngineConfig, LLMEngine, PagedKVCache, PrefixCache,
                         TinyTransformer, compute_block_hashes, reference_greedy)
from mini_engine.model import SeqChunk

MAX_TOKENS = 10


def _toks(seed: int, n: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, 512, n)]


@pytest.fixture(scope="module")
def model() -> TinyTransformer:
    return TinyTransformer()


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    shared = _toks(100, 40)                      # 40 个共享 token = 2 个满块 + 半块
    return [shared + _toks(1, 10), shared + _toks(2, 25), _toks(3, 32),
            _toks(4, 33), shared + _toks(5, 50), _toks(6, 7)]


@pytest.fixture(scope="module")
def reference(model, prompts) -> list[list[int]]:
    return [reference_greedy(model, p, MAX_TOKENS) for p in prompts]


def _run(model, prompts, cfg: EngineConfig, sequential: bool = False) -> tuple[list[list[int]], LLMEngine]:
    eng = LLMEngine(model, cfg)
    if sequential:                               # 一个跑完再加下一个：后面的请求才能命中前面的缓存
        outs = [eng.generate([p], MAX_TOKENS)[0] for p in prompts]
    else:
        outs = eng.generate(prompts, MAX_TOKENS)
    eng.allocator.assert_no_leak()
    return outs, eng


# ---------------- 模型层：分页注意力 ----------------
def test_paged_forward_matches_dense_and_reads_blocks(model):
    tokens = _toks(7, 50)
    dense = model.forward_dense(tokens)
    outs = []
    for table in ([0, 1, 2, 3], [9, 4, 7, 1]):   # 物理块号连续 / 打乱，结果必须一样
        kv = PagedKVCache(model.cfg, num_blocks=10, block_size=16)
        last = []
        for p0, p1 in ((0, 20), (20, 50)):       # 分两块 prefill：第二块要从分页 KV 里读第一块
            pos = np.arange(p0, p1)
            slots = np.asarray(table)[pos // 16] * 16 + pos % 16
            logits = model.forward_paged(np.asarray(tokens[p0:p1]), pos, slots,
                                         [SeqChunk(0, p1 - p0, p1, table)], kv)
            last.append(logits[0])
        np.testing.assert_allclose(last[0], dense[19], atol=1e-9)
        np.testing.assert_allclose(last[1], dense[49], atol=1e-9)
        outs.append(last[1])
        # 把第 0 个物理块清零后再算最后一个 token：结果必须变——证明注意力真的在读块里的 KV
        kv.k[:, table[0]] = 0.0
        pos = np.array([49])
        corrupted = model.forward_paged(np.asarray(tokens[49:50]), pos, np.asarray(table)[pos // 16] * 16 + pos % 16,
                                        [SeqChunk(0, 1, 50, table)], kv)[0]
        assert not np.allclose(corrupted, dense[49], atol=1e-3)
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-12)


def test_outputs_depend_on_context(model):
    """随机权重也必须对上下文敏感，否则下面的"输出一致"测试就没有意义。"""
    a = reference_greedy(model, _toks(8, 20) + [42], 4)
    b = reference_greedy(model, _toks(9, 20) + [42], 4)
    assert a != b


# ---------------- 引擎：各种优化都不改变输出 ----------------
def test_engine_matches_naive_reference(model, prompts, reference):
    outs, _ = _run(model, prompts, EngineConfig())
    assert outs == reference


def test_prefix_cache_on_off_identical(model, prompts, reference):
    on, eng_on = _run(model, prompts, EngineConfig(enable_prefix_caching=True), sequential=True)
    off, eng_off = _run(model, prompts, EngineConfig(enable_prefix_caching=False), sequential=True)
    assert on == off == reference
    assert eng_on.stats.cached_tokens > 0 and eng_off.stats.cached_tokens == 0
    assert eng_on.stats.prefill_tokens_computed == eng_off.stats.prefill_tokens_computed - eng_on.stats.cached_tokens


@pytest.mark.parametrize("budget", [7, 16, 23])
def test_chunked_prefill_matches_one_shot(model, prompts, reference, budget):
    chunked, eng_c = _run(model, prompts, EngineConfig(max_num_batched_tokens=budget))
    one_shot, eng_o = _run(model, prompts, EngineConfig(max_num_batched_tokens=512, enable_chunked_prefill=False))
    assert chunked == one_shot == reference
    assert eng_c.stats.steps > eng_o.stats.steps          # 确实被切成了更多步


def test_batched_equals_individual(model, prompts, reference):
    eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=64))
    ids = [eng.add_request(p, MAX_TOKENS) for p in prompts]
    noncontiguous = False
    while eng.has_unfinished():
        eng.step()
        for r in eng.scheduler.running:                  # 并发请求的 block table 会交错，不再连续
            noncontiguous |= any(b - a != 1 for a, b in zip(r.block_ids, r.block_ids[1:]))
    batched = [eng.requests[i].output_token_ids for i in ids]
    single = [_run(model, [p], EngineConfig(enable_prefix_caching=False))[0][0] for p in prompts]
    assert batched == single == reference
    assert noncontiguous
    assert eng.stats.steps < len(prompts) * MAX_TOKENS    # 确实混在一起跑了
    eng.allocator.assert_no_leak()


@pytest.mark.parametrize("prefix_caching", [True, False])
def test_preemption_recompute_keeps_outputs(model, prompts, reference, prefix_caching):
    cfg = EngineConfig(num_blocks=10, max_num_batched_tokens=32, max_pinned_frac=0.0,
                       enable_prefix_caching=prefix_caching)
    outs, eng = _run(model, prompts, cfg)
    assert eng.stats.preemptions > 0
    assert outs == reference
    assert eng.stats.prefill_tokens_computed > sum(map(len, prompts)) - eng.stats.cached_tokens  # 有重算


# ---------------- 前缀缓存：命中数量精确 ----------------
def test_shared_prefix_second_request_hits_exact_blocks(model):
    shared = _toks(11, 40)
    eng = LLMEngine(model, EngineConfig())
    eng.generate([shared + _toks(12, 10)], 4)
    before = eng.stats.prefill_tokens_computed
    rid = eng.add_request(shared + _toks(13, 10), 4)
    eng.run_until_done()
    # 40 个共享 token 只有前 2 个满块（32 token）的 hash 相同；第 3 块混入了不同的后缀
    assert eng.requests[rid].num_cached_tokens == 32
    assert eng.stats.prefill_tokens_computed - before == 50 - 32
    eng.allocator.assert_no_leak()


def test_multiturn_hits_blocks_filled_by_generated_tokens(model):
    eng = LLMEngine(model, EngineConfig())
    turn1 = _toks(14, 50)
    out1 = eng.generate([turn1], 10)[0]
    # 第 1 轮算过 KV 的 token = 50 + 10 - 1 = 59（最后一个输出还没喂回模型）→ 3 个满块
    turn2 = turn1 + out1 + _toks(15, 20)                 # 80 token，只追加
    before = eng.stats.prefill_tokens_computed
    rid = eng.add_request(turn2, 4)
    eng.run_until_done()
    assert eng.requests[rid].num_cached_tokens == 48
    assert eng.stats.prefill_tokens_computed - before == 80 - 48
    assert eng.requests[rid].output_token_ids == reference_greedy(model, turn2, 4)


def test_full_hit_still_recomputes_last_block(model):
    """整段 prompt 都在缓存里也要重算最后一个 token 拿 logits；只匹配 (len-1) 个 token 内的满块。"""
    p = _toks(16, 32)
    eng = LLMEngine(model, EngineConfig())
    eng.generate([p], 3)
    rid = eng.add_request(p, 3)
    eng.run_until_done()
    assert eng.requests[rid].num_cached_tokens == 16
    assert eng.requests[rid].output_token_ids == reference_greedy(model, p, 3)


def test_hash_chain_prevents_cross_context_false_hits(model):
    """同样 16 个 token 的块，前文不同 → KV 不同 → 不能共享。块 hash 必须包含父块 hash。"""
    x, w, z, y = (_toks(s, 16) for s in (30, 31, 32, 33))
    eng = LLMEngine(model, EngineConfig())
    eng.generate([x + w + [5]], 2)                      # 登记 [x] [x→w]
    eng.generate([z + y + [5]], 2)                      # 登记 [z] [z→y]（y 的 KV 是在 z 之后算的）
    target = x + y + [5]
    rid = eng.add_request(target, 4)
    eng.run_until_done()
    assert eng.requests[rid].num_cached_tokens == 16    # 只命中 x；y 块虽然 token 相同但前文不同
    assert eng.requests[rid].output_token_ids == reference_greedy(model, target, 4)


def test_lru_evicts_tail_first_and_keeps_chain_head():
    alloc = BlockAllocator(6)
    cache = PrefixCache(alloc)
    toks = list(range(2, 50))                            # 48 token = 3 个满块
    hashes = compute_block_hashes(toks, 16)
    blocks = alloc.allocate(3, now=0.0)
    for h, b in zip(hashes, blocks):
        cache.register(h, b)
    alloc.free(blocks)                                   # 逆序入队：尾块离队头更近
    assert cache.lookup(hashes) == blocks                # 空闲但仍可命中
    alloc.allocate(4, now=0.0)                           # 3 个从未用过的块 + 1 个被淘汰
    assert alloc.num_evictions == 1
    assert cache.lookup(hashes) == blocks[:2]            # 被淘汰的是尾块，链头还在
    alloc.touch(blocks[:2])                              # 命中：从空闲队列摘出，不会再被淘汰
    assert alloc.allocate(1, now=0.0) is None
    alloc.free(blocks[:2])


def test_pin_protects_blocks_until_deadline(model):
    now = [0.0]
    eng = LLMEngine(model, EngineConfig(num_blocks=16, max_pinned_frac=0.5), clock=lambda: now[0])
    agent = _toks(17, 64)                                # 4 个满块
    eng.add_request(agent, 1, pin_seconds=10.0)          # 结束后 pin 10 秒（等工具）
    eng.run_until_done()
    assert eng.allocator.num_pinned(now[0]) == 4
    for seed in (18, 19):                                # 两个别人的请求把块池用了两遍
        eng.generate([_toks(seed, 120)], 8)
    rid2 = eng.add_request(agent + _toks(20, 8), 1)
    eng.run_until_done()
    assert eng.requests[rid2].num_cached_tokens == 64    # pin 住的 KV 还在
    # 对照：不 pin 时同样的负载会把它淘汰
    eng2 = LLMEngine(model, EngineConfig(num_blocks=16, max_pinned_frac=0.5), clock=lambda: now[0])
    eng2.generate([agent], 1)
    for seed in (18, 19):
        eng2.generate([_toks(seed, 120)], 8)
    rid3 = eng2.add_request(agent + _toks(20, 8), 1)
    eng2.run_until_done()
    assert eng2.requests[rid3].num_cached_tokens == 0
    assert eng.requests[rid2].output_token_ids == eng2.requests[rid3].output_token_ids
    # pin 到期后可以被正常淘汰
    now[0] = 11.0
    assert eng.allocator.num_pinned(now[0]) == 0
    eng.allocator.assert_no_leak()


def test_pin_budget_is_capped():
    alloc = BlockAllocator(8)
    ids = alloc.allocate(6, now=0.0)
    assert alloc.pin(ids, until=5.0, now=0.0, max_pinned=4) == 4
    alloc.free(ids)
    assert alloc.allocate(5, now=1.0) is None            # 4 个被 pin，只剩 4 个可用
    assert alloc.allocate(5, now=6.0) is not None        # 到期后可用


def test_rejects_request_that_can_never_fit(model):
    eng = LLMEngine(model, EngineConfig(num_blocks=8, max_pinned_frac=0.5))
    with pytest.raises(ValueError):
        eng.add_request(_toks(21, 70), 1)                # 需要 5 块 > 8 - 4


# ---------------- 同一步共享、写入顺序、副本补位 ----------------
def test_same_step_requests_share_prefix_grpo_style(model):
    """同一 prompt 一次提交 G 份（GRPO 式同题多采样）：调度时登记，同一步里后接纳的 G-1 份直接命中。"""
    p, g = _toks(40, 50), 8                              # 50 token：最多匹配 49 个 → 3 个满块
    eng = LLMEngine(model, EngineConfig())
    outs = eng.generate([p] * g, 4)
    assert eng.stats.steps == 4                          # 一步 prefill 全部 G 份，之后 3 步 decode
    assert eng.stats.cached_tokens == (g - 1) * 48
    assert eng.stats.prefill_tokens_computed == 50 + (g - 1) * 2
    assert outs == [reference_greedy(model, p, 4)] * g
    eng.allocator.assert_no_leak()


def test_forward_paged_writes_all_kv_before_attention(model):
    """同一次前向里，排在前面的序列要读排在后面的序列本步才写的块（调度时登记的安全前提）。"""
    tokens = _toks(41, 40)
    dense = model.forward_dense(tokens)
    kv = PagedKVCache(model.cfg, num_blocks=6, block_size=16)
    table_a, table_b = [0, 1], [0, 1, 2]                # B 共享 A 的前两个块
    pos_a, pos_b = np.arange(0, 32), np.arange(32, 40)
    slots_a = np.asarray(table_a)[pos_a // 16] * 16 + pos_a % 16
    slots_b = np.asarray(table_b)[pos_b // 16] * 16 + pos_b % 16
    # 故意把"读者" B 放在"写者" A 前面
    logits = model.forward_paged(np.asarray(tokens[32:40] + tokens[0:32]), np.concatenate([pos_b, pos_a]),
                                 np.concatenate([slots_b, slots_a]),
                                 [SeqChunk(0, 8, 40, table_b), SeqChunk(8, 32, 32, table_a)], kv)
    np.testing.assert_allclose(logits[0], dense[39], atol=1e-9)
    np.testing.assert_allclose(logits[1], dense[31], atol=1e-9)


def test_duplicate_block_is_promoted_when_canonical_is_evicted():
    alloc = BlockAllocator(4)
    cache = PrefixCache(alloc)
    h = compute_block_hashes(list(range(2, 18)), 16)[0]
    first, second = alloc.allocate(2, now=0.0)          # 两个请求各自算出了同一个块
    cache.register(h, first, chain_pos=0)
    cache.register(h, second, chain_pos=0)
    assert cache.lookup([h]) == [first]
    alloc.free([first])
    alloc.allocate(3, now=0.0)                           # 把 first 淘汰掉
    assert cache.lookup([h]) == [second]                 # 副本顶上，前缀依然可以命中
    alloc.free([second])


def test_batch_changes_logits_only_at_rounding_level(model):
    """"输出一致"是 token 级的：同一序列单独算 vs 和别人拼 batch，logits 只在浮点舍入级别不同。"""
    a, b = _toks(42, 40), _toks(43, 57)
    kv = PagedKVCache(model.cfg, num_blocks=16, block_size=16)
    pa, pb = np.arange(40), np.arange(57)
    ta, tb = [0, 1, 2], [3, 4, 5, 6]
    sa, sb = np.asarray(ta)[pa // 16] * 16 + pa % 16, np.asarray(tb)[pb // 16] * 16 + pb % 16
    alone = model.forward_paged(np.asarray(a), pa, sa, [SeqChunk(0, 40, 40, ta)], kv)[0]
    both = model.forward_paged(np.asarray(b + a), np.concatenate([pb, pa]), np.concatenate([sb, sa]),
                               [SeqChunk(0, 57, 57, tb), SeqChunk(57, 40, 40, ta)], kv)[1]
    np.testing.assert_allclose(both, alone, rtol=0, atol=1e-9)
    assert np.argmax(both) == np.argmax(alone)
    top2 = np.sort(alone)[-2:]
    assert top2[1] - top2[0] > 1e-6                      # argmax 的间隔远大于舍入误差


# ---------------- 容量检查的边界 ----------------
def test_capacity_counts_kv_tokens_not_generated_tokens(model):
    """最后一个生成的 token 不写 KV：一个请求最多 prompt + max_tokens - 1 个 KV token。"""
    eng = LLMEngine(model, EngineConfig(num_blocks=1))
    eng.add_request(_toks(44, 16), 1)                    # 16 个 KV token，正好 1 块
    eng.add_request(_toks(45, 15), 2)                    # 15 + 2 - 1 = 16，也是 1 块
    eng.run_until_done()
    eng.allocator.assert_no_leak()
    with pytest.raises(ValueError):
        eng.add_request(_toks(46, 16), 2)                # 17 个 KV token → 2 块


def test_no_chunked_prefill_budget_boundary(model):
    """关闭 chunked prefill 时，被抢占的请求要一步重算完 prompt + 已生成 token，所以按 KV token 数检查。"""
    eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=32, enable_chunked_prefill=False))
    rid = eng.add_request(_toks(47, 30), 3)              # 30 + 3 - 1 = 32：刚好
    eng.run_until_done()
    assert eng.requests[rid].output_token_ids == reference_greedy(model, _toks(47, 30), 3)
    with pytest.raises(ValueError):
        eng.add_request(_toks(47, 30), 4)                # 33 > 32


def test_pin_is_opt_in(model):
    eng = LLMEngine(model, EngineConfig(num_blocks=256))
    eng.add_request(_toks(48, 2000), 1)                  # 默认不为 pin 预留块：长请求能进来
    with pytest.raises(ValueError):
        eng.add_request(_toks(49, 10), 1, pin_seconds=1.0)


def test_tiny_config_validates_structure():
    from mini_engine import TinyConfig
    for bad in (dict(d_model=100), dict(n_kv_heads=3), dict(d_model=120, n_heads=8), dict(vocab_size=2)):
        with pytest.raises(ValueError):
            TinyConfig(**bad)


# ---------------- 中止、回收、权重更新、租户隔离 ----------------
def test_abort_mid_prefill_frees_blocks_and_keeps_others_correct(model):
    eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=16))
    long_p, other = _toks(50, 60), _toks(51, 20)
    rid = eng.add_request(long_p, 5)
    oid = eng.add_request(other, 5)
    eng.step()
    eng.step()                                           # long 还在 chunked prefill 中途
    assert 0 < eng.requests[rid].num_computed_tokens < len(long_p)
    assert eng.abort_request(rid)
    assert not eng.abort_request(rid)                    # 重复中止无副作用
    eng.run_until_done()
    assert eng.requests[oid].output_token_ids == reference_greedy(model, other, 5)
    eng.allocator.assert_no_leak()
    assert eng.pop_request(rid).status.value == "aborted" and rid not in eng.requests


def test_weight_update_without_cache_reset_hits_stale_kv(model):
    """RL 权重同步后：块哈希只看 token，不 reset 前缀缓存就会命中旧权重算的 KV。"""
    new_model = TinyTransformer(type(model.cfg)(seed=1))
    p = _toks(52, 64)
    for reset in (False, True):
        eng = LLMEngine(model, EngineConfig())
        eng.generate([p], 2)
        eng.update_weights(new_model, reset_prefix_cache=reset)
        rid = eng.add_request(p, 6)
        eng.run_until_done()
        req = eng.requests[rid]
        fresh = reference_greedy(new_model, p, 6)
        if reset:
            assert req.num_cached_tokens == 0 and req.output_token_ids == fresh
        else:
            assert req.num_cached_tokens == 48 and req.output_token_ids != fresh


def test_reset_prefix_cache_refuses_while_blocks_in_use(model):
    eng = LLMEngine(model, EngineConfig())
    eng.add_request(_toks(53, 40), 4)
    eng.step()
    assert eng.reset_prefix_cache() is False
    with pytest.raises(RuntimeError):
        eng.update_weights(model)
    eng.run_until_done()
    assert eng.reset_prefix_cache() is True and not eng.prefix_cache.table


def test_cache_salt_isolates_tenants(model):
    p = _toks(54, 50)
    eng = LLMEngine(model, EngineConfig())
    eng.add_request(p, 2, cache_salt=7)
    eng.run_until_done()
    other = eng.add_request(p, 2, cache_salt=8)          # 另一个租户：同样的 token 也不命中
    same = eng.add_request(p, 2, cache_salt=7)
    eng.run_until_done()
    assert eng.requests[other].num_cached_tokens == 0
    assert eng.requests[same].num_cached_tokens == 48


def test_long_prefill_token_threshold_caps_chunks_without_changing_output(model, prompts, reference):
    outs, eng = _run(model, prompts, EngineConfig(long_prefill_token_threshold=8))
    assert outs == reference
    assert eng.stats.steps > _run(model, prompts, EngineConfig())[1].stats.steps
