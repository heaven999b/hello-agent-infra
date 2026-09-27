"""Capstone 练习测试。默认测 exercise.py（没写完会失败，这是正常的）；INFRA_SOLUTION=1 时测 solution.py。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from codesign_sim.testing import load_exercise
from mini_engine import (BlockAllocator, EngineConfig, LLMEngine, PrefixCache, TinyTransformer,
                         compute_block_hashes, reference_greedy)
from mini_engine.agent_harness import HarnessConfig, run_harness_sync

ex = load_exercise(__file__)


def _toks(seed: int, n: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, 512, n)]


@pytest.fixture(scope="module")
def model() -> TinyTransformer:
    return TinyTransformer()


# ---------------- (a) 路由 ----------------
def test_prefix_match_blocks_stops_at_first_miss():
    assert ex.prefix_match_blocks([1, 2, 3], {1, 3}) == 1
    assert ex.prefix_match_blocks([1, 2, 3], {1, 2, 3, 4}) == 3
    assert ex.prefix_match_blocks([1, 2, 3], set()) == 0


def test_pick_instance_rules():
    p = _toks(1, 64)                                       # 4 个满块
    h = compute_block_hashes(p, 16)
    caches = [set(h[:1]), set(h[:3]), set()]
    assert ex.pick_instance(p, caches, loads=[0, 0, 0], load_cap=4) == 1        # 命中最多
    assert ex.pick_instance(p, caches, loads=[0, 4, 0], load_cap=4) == 0        # 1 满载，退而求其次
    assert ex.pick_instance(p, [set(), set(), set()], [3, 1, 1], load_cap=4) == 1   # 都不命中：负载小、下标小
    assert ex.pick_instance(p, caches, loads=[5, 6, 5], load_cap=4) == 0        # 全满：负载最小


def test_cache_aware_routing_with_approximate_view_beats_round_robin(model):
    """3 个 agent × 3 轮只追加对话，分给 2 个真实引擎实例。

    路由器看不到引擎内部：它只记得自己把哪些 prompt 发去了哪台（近似视图，不知道哪些块已被淘汰），
    负载按"驻留在该实例上的 agent 数"计。agent 数取奇数，轮询才会把同一 agent 的相邻两轮分到不同实例。
    """
    def run(policy: str) -> tuple[int, list[int]]:
        engines = [LLMEngine(model, EngineConfig()) for _ in range(2)]
        views: list[set[int]] = [set(), set()]
        ctx = {a: _toks(10 + a, 40) for a in range(3)}
        home: dict[int, int] = {}
        rr = 0
        for turn in range(3):
            for a in range(3):
                if policy == "affinity":
                    loads = [sum(1 for b, h in home.items() if h == i and b != a) for i in range(2)]
                    i = ex.pick_instance(ctx[a], views, loads, load_cap=2)
                else:
                    i, rr = rr % 2, rr + 1
                home[a] = i
                views[i] |= set(compute_block_hashes(ctx[a], 16))
                out = engines[i].generate([ctx[a]], 4)[0]
                ctx[a] = ctx[a] + out + _toks(100 * turn + a, 20)
        return sum(e.stats.prefill_tokens_computed for e in engines), [e.stats.steps for e in engines]

    affinity, steps = run("affinity")
    round_robin, _ = run("round_robin")
    assert affinity < round_robin
    assert min(steps) > 0                                  # 两个实例都分到了活


# ---------------- (b) 会话感知淘汰 ----------------
def test_choose_victims_rules():
    B = ex.FreeBlockInfo
    blocks = [B(0, last_used=1.0, pin_until=0.0, chain_pos=0, n_active_sessions=1, cached=True),
              B(1, last_used=5.0, pin_until=0.0, chain_pos=1, n_active_sessions=0, cached=True),
              B(2, last_used=2.0, pin_until=0.0, chain_pos=0, n_active_sessions=0, cached=True),
              B(3, last_used=0.5, pin_until=99.0, chain_pos=2, n_active_sessions=0, cached=True),   # pin 住了
              B(4, last_used=9.0, pin_until=0.0, chain_pos=-1, n_active_sessions=3, cached=False),  # 没挂前缀
              B(5, last_used=1.0, pin_until=0.0, chain_pos=1, n_active_sessions=1, cached=True)]
    assert ex.choose_victims(blocks, 3, now=10.0) == [4, 2, 1]
    assert ex.choose_victims(blocks, 5, now=10.0) == [4, 2, 1, 5, 0]      # 同时释放的链：尾块 5 先于链头 0
    assert ex.choose_victims(blocks, 6, now=10.0) is None
    assert ex.choose_victims(blocks, 6, now=100.0) == [4, 3, 2, 1, 5, 0]  # pin 到期


def test_choose_victims_evicts_chain_tail_first_on_real_cache():
    """同一时刻释放的 3 块链：淘汰 1 块之后，前 2 块仍然能命中。"""
    alloc = BlockAllocator(3)
    cache = PrefixCache(alloc)
    hashes = compute_block_hashes(list(range(2, 50)), 16)
    blocks = alloc.allocate(3, now=0.0)
    for i, (h, b) in enumerate(zip(hashes, blocks)):
        cache.register(h, b, chain_pos=i)
    alloc.free(blocks, now=1.0)
    infos = [ex.FreeBlockInfo(b, 1.0, alloc.blocks[b].pin_until, i, 0, True) for i, b in enumerate(blocks)]
    victim = ex.choose_victims(infos, 1, now=2.0)
    assert victim == [blocks[2]]
    alloc.allocate(1, now=2.0, exclude=blocks[:2])        # 真的把它分配出去
    assert cache.lookup(hashes) == blocks[:2]


def test_session_aware_allocator_is_output_preserving(model):
    """差分：显存很紧、会抢占的情况下，换淘汰策略不改变任何生成结果，结束时无泄漏。"""
    shared = _toks(20, 40)
    prompts = [shared + _toks(21 + i, 5 + 7 * i) for i in range(6)] + [_toks(30, 33)] * 2
    cfg = EngineConfig(num_blocks=10, max_num_batched_tokens=32)
    eng = LLMEngine(model, cfg, allocator_cls=ex.SessionAwareAllocator)
    outs = eng.generate(prompts, 8)
    assert outs == [reference_greedy(model, p, 8) for p in prompts]
    assert eng.stats.preemptions > 0 and eng.allocator.num_evictions > 0
    eng.allocator.assert_no_leak()


def test_session_aware_allocator_reduces_recompute_for_agents(model):
    """会话长短不一的 agent 负载（虚拟时钟，结果确定）：已结束会话的块先走，正在等工具的会话的 KV 留下。"""
    cfg = HarnessConfig(n_turns=5, turn_spread=3)
    tight = EngineConfig(num_blocks=32)
    lru = run_harness_sync(model, cfg, tight)
    sa = run_harness_sync(model, cfg, tight, allocator_cls=ex.SessionAwareAllocator)
    assert sa.outputs == lru.outputs
    assert sa.prefill_computed < lru.prefill_computed


# ---------------- (c) 投机解码 ----------------
@pytest.mark.parametrize("k", [1, 3, 4])
def test_speculative_matches_greedy_and_counts_calls(model, k):
    prompt, n = _toks(5, 30), 12
    ref = reference_greedy(model, prompt, n)

    def oracle(seq: list[int], kk: int) -> list[int]:        # 永远猜对
        done = len(seq) - len(prompt)
        return ref[done:done + kk]

    def wrong(seq: list[int], kk: int) -> list[int]:         # 永远猜错
        return [(ref[len(seq) - len(prompt)] + 1) % 512] * kk

    out, calls = ex.speculative_greedy(model, prompt, n, oracle, k)
    assert out == ref and calls == math.ceil(n / (k + 1))
    out, calls = ex.speculative_greedy(model, prompt, n, wrong, k)
    assert out == ref and calls == n


@pytest.mark.parametrize("alpha,k", [(0.6, 3), (0.8, 4)])
def test_speculative_yield_matches_expected_formula(model, alpha, k):
    """每个位置的草稿以概率 α 猜对（按位置固定 seed）：每次前向的平均产出 ≈ E(α,k) = (1-α^(k+1))/(1-α)。"""
    prompt, n = _toks(5, 20), 120
    ref = LLMEngine(model).generate([prompt], n)[0]          # 引擎输出 = 朴素 greedy（由 tests/ 保证）
    correct = np.random.default_rng(1).random(n) < alpha

    def draft(seq: list[int], kk: int) -> list[int]:
        d = len(seq) - len(prompt)
        return [ref[d + j] if correct[d + j] else (ref[d + j] + 1) % 512 for j in range(kk)]

    out, calls = ex.speculative_greedy(model, prompt, n, draft, k)
    expected = (1 - alpha ** (k + 1)) / (1 - alpha)
    assert out == ref
    assert abs(n / calls - expected) / expected < 0.15


def test_speculative_with_random_draft_is_still_exact(model):
    prompt = _toks(6, 25)
    rng = np.random.default_rng(0)
    out, _ = ex.speculative_greedy(model, prompt, 10, lambda s, kk: [int(t) for t in rng.integers(0, 512, kk)], 3)
    assert out == reference_greedy(model, prompt, 10)
