"""模块 02 练习测试。离线、确定性：五个函数都是纯函数。

运行：make module N=02
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/02_inference_engines
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)
COST = ex.LLAMA3_70B_8XH100


# ================================================================== (a) paged_allocate


def test_a_docstring_example():
    r = ex.paged_allocate([5, 16, 17], block_size=16, num_blocks=8)
    assert r.block_tables == [[0], [1], [2, 3]]
    assert r.admitted == 3
    assert r.used_blocks == 4
    assert r.waste_tokens == 26
    assert r.waste_ratio == pytest.approx(26 / 64)


def test_a_waste_is_at_most_one_block_per_request():
    rng = np.random.default_rng(0)
    lens = [int(x) for x in rng.integers(1, 4000, size=200)]
    r = ex.paged_allocate(lens, block_size=16, num_blocks=10**6)
    assert r.admitted == 200
    assert 0 <= r.waste_tokens <= 200 * 15  # 每个请求最多浪费 block_size - 1 个槽位
    assert r.used_blocks == sum(math.ceil(n / 16) for n in lens)
    assert r.waste_ratio < 0.01  # 平均长度 ~2000 时，碎片率远小于 1%


def test_a_mean_waste_is_b_minus_1_over_2():
    # L mod B 均匀分布时，每序列的期望浪费是 (B-1)/2，B=16 时为 7.5
    lens = list(range(1, 16 * 100 + 1))
    r = ex.paged_allocate(lens, block_size=16, num_blocks=10**5)
    assert r.waste_tokens / len(lens) == pytest.approx(7.5)


def test_a_fcfs_all_or_nothing_head_of_line_blocking():
    # 第 2 个请求要 3 块，只剩 2 块 -> 不接纳它，也不许跳过它去接纳后面只要 1 块的请求
    r = ex.paged_allocate([16, 40, 1], block_size=16, num_blocks=3)
    assert r.admitted == 1
    assert r.block_tables == [[0]]
    assert r.used_blocks == 1
    assert r.waste_tokens == 0


def test_a_uses_given_free_list_non_contiguous():
    r = ex.paged_allocate([20, 10], block_size=16, num_blocks=10, free_blocks=[7, 2, 9, 4])
    assert r.block_tables == [[7, 2], [9]]
    assert r.used_blocks == 3
    assert r.waste_tokens == 3 * 16 - 30


def test_a_edge_cases_and_validation():
    r = ex.paged_allocate([], block_size=16, num_blocks=4)
    assert (r.admitted, r.used_blocks, r.waste_tokens, r.waste_ratio) == (0, 0, 0, 0.0)
    r = ex.paged_allocate([0, 16], block_size=16, num_blocks=1)
    assert r.block_tables == [[], [0]]
    with pytest.raises(ValueError):
        ex.paged_allocate([1], block_size=0, num_blocks=1)
    for bad in ([-1], [-17]):
        with pytest.raises(ValueError):
            ex.paged_allocate(bad, block_size=16, num_blocks=8)
    with pytest.raises(ValueError):
        ex.paged_allocate([20], block_size=16, num_blocks=4, free_blocks=[0, 0])  # 重复块号
    with pytest.raises(ValueError):
        ex.paged_allocate([20], block_size=16, num_blocks=4, free_blocks=[1, 9])  # 越界块号


def test_a_larger_block_means_more_waste():
    lens = [37, 250, 1001, 64, 5]
    small = ex.paged_allocate(lens, block_size=16, num_blocks=10**4)
    big = ex.paged_allocate(lens, block_size=256, num_blocks=10**4)
    assert big.waste_tokens > small.waste_tokens


# ================================================================== (b) plan_chunked_prefill


def test_b_docstring_example():
    plan = ex.plan_chunked_prefill([300, 100], n_decodes=2, token_budget=256)
    assert plan == [[(0, 254)], [(0, 46), (1, 100)]]


def test_b_finished_prompts_join_decode_next_step():
    # 第 0 步：预算 10，decode 0 -> [(0, 4), (1, 6)]；prompt 0 做完，第 1 步起 decode 占 1
    # 第 1 步：预算 10 - 1 = 9 -> [(1, 9)]；第 2 步：9 -> [(1, 9)]；第 3 步：9 -> [(1, 1)]
    plan = ex.plan_chunked_prefill([4, 25], n_decodes=0, token_budget=10)
    assert plan == [[(0, 4), (1, 6)], [(1, 9)], [(1, 9)], [(1, 1)]]


def test_b_invariants_budget_and_totals():
    rng = np.random.default_rng(1)
    prompts = [int(x) for x in rng.integers(1, 5000, size=30)]
    budget, n_dec = 2048, 64
    plan = ex.plan_chunked_prefill(prompts, n_decodes=n_dec, token_budget=budget)
    got = [0] * len(prompts)
    decoding, order = n_dec, []
    for step in plan:
        assert sum(t for _, t in step) + decoding <= budget
        assert all(t > 0 for _, t in step)
        for idx, t in step:
            got[idx] += t
            order.append(idx)
        decoding += sum(1 for idx, _ in step if got[idx] == prompts[idx])
    assert got == prompts
    assert order == sorted(order)  # FCFS：下标只增不减


def test_b_smaller_budget_needs_more_steps_for_long_prompt():
    steps_small = len(ex.plan_chunked_prefill([32_768], n_decodes=0, token_budget=512))
    steps_big = len(ex.plan_chunked_prefill([32_768], n_decodes=0, token_budget=8192))
    assert steps_small == 64
    assert steps_big == 4


def test_b_errors_and_empty():
    assert ex.plan_chunked_prefill([], n_decodes=5, token_budget=16) == []
    with pytest.raises(ValueError):
        ex.plan_chunked_prefill([10], n_decodes=16, token_budget=16)  # decode 吃光预算
    bad_calls = [
        dict(prompt_lens=[10], n_decodes=0, token_budget=0),
        dict(prompt_lens=[10], n_decodes=-3, token_budget=16),
        dict(prompt_lens=[10, -5], n_decodes=0, token_budget=16),
        dict(prompt_lens=[0], n_decodes=0, token_budget=16),
        dict(prompt_lens=[10], n_decodes=0, token_budget=16, max_steps=0),
    ]
    for kwargs in bad_calls:
        with pytest.raises(ValueError):
            ex.plan_chunked_prefill(**kwargs)


# ================================================================== (c1) expected_accepted_tokens


def test_c1_formula_values():
    assert ex.expected_accepted_tokens(0.8, 4) == pytest.approx((1 - 0.8**5) / 0.2)
    assert ex.expected_accepted_tokens(0.85, 1) == pytest.approx(1.85)  # MTP 草稿 1 个 token
    assert ex.expected_accepted_tokens(0.3, 0) == pytest.approx(1.0)
    assert ex.expected_accepted_tokens(1.0, 7) == pytest.approx(8.0)
    assert ex.expected_accepted_tokens(0.0, 7) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ex.expected_accepted_tokens(1.2, 3)


def test_c1_matches_monte_carlo():
    rng = np.random.default_rng(2)
    alpha, gamma, n = 0.7, 5, 200_000
    accepted = rng.random((n, gamma)) < alpha
    # 连续接受的前缀长度 + 目标模型补的 1 个
    first_reject = np.where(accepted.all(axis=1), gamma, np.argmin(accepted, axis=1))
    empirical = float(np.mean(first_reject + 1))
    assert empirical == pytest.approx(ex.expected_accepted_tokens(alpha, gamma), rel=0.01)


# ================================================================== (c2) spec_speedup_at_batch


def test_c2_small_batch_reduces_to_leviathan_formula():
    # batch 1：访存瓶颈，验证 5 个位置和算 1 个一样快 -> S = E / (1 + gamma·c)
    s = ex.spec_speedup_at_batch(0.8, 4, COST, batch=1, ctx=4096, c=0.05)
    assert s == pytest.approx(ex.expected_accepted_tokens(0.8, 4) / 1.2, rel=1e-6)


def test_c2_depends_on_context_length_at_large_batch():
    # 同样 batch 256：短上下文时验证变成计算瓶颈，投机反而更慢；上下文变长后 KV 读取主导，验证近乎免费
    s = [ex.spec_speedup_at_batch(0.8, 4, COST, batch=256, ctx=ctx) for ctx in (512, 2048, 5000)]
    assert s == pytest.approx([0.75, 1.24, 2.11], abs=0.01)
    assert s[0] < 1 < s[1] < s[2]
    long_ctx = ex.spec_speedup_at_batch(0.8, 4, COST, batch=32, ctx=32_768)
    assert long_ctx == pytest.approx(ex.expected_accepted_tokens(0.8, 4), rel=1e-6)


def test_c2_best_gamma_shrinks_as_batch_grows_at_short_context():
    def best_gamma(batch: int, ctx: int) -> int:
        speedups = [ex.spec_speedup_at_batch(0.8, g, COST, batch, ctx, c=0.05) for g in range(9)]
        return int(np.argmax(speedups))

    assert best_gamma(1, 512) == 8
    assert best_gamma(128, 512) == 1
    assert best_gamma(256, 512) == 0  # 该关掉投机了
    assert best_gamma(256, 4096) == 2  # 同样的 batch，上下文长了还值得开


def test_c2_gamma_zero_and_validation():
    assert ex.spec_speedup_at_batch(0.8, 0, COST, batch=64, ctx=1000) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ex.spec_speedup_at_batch(0.8, 4, COST, batch=0, ctx=1000)
    with pytest.raises(ValueError):
        ex.spec_speedup_at_batch(1.5, 4, COST, batch=8, ctx=1000)
    with pytest.raises(ValueError):
        ex.spec_speedup_at_batch(0.8, 4, COST, batch=8, ctx=1000, c=-0.1)


# ================================================================== (d) infer_chunk_tokens


def _chunk_step_time(chunk: int, done: int) -> float:
    flops = COST.linear_flops_per_token * chunk + COST.attn_flops_per_pair * chunk * (done + chunk / 2)
    return flops / COST.flops_per_s


def test_d_recovers_budget_from_stall():
    stall = _chunk_step_time(8192, 24576)
    assert stall == pytest.approx(0.373, abs=0.001)  # 讲义第 5 节：32k prompt 的最后一块
    assert ex.infer_chunk_tokens(stall, 24576, COST) == 8192


def test_d_attention_term_matters():
    # 同样 0.373 秒的卡顿，如果这是 prompt 的第一块（前面没有已算的 token），块会大得多
    first_chunk = ex.infer_chunk_tokens(0.373, 0, COST)
    assert first_chunk > 11_000
    for chunk, done in [(2048, 0), (4096, 60_000), (16384, 100_000)]:
        assert ex.infer_chunk_tokens(_chunk_step_time(chunk, done), done, COST) == chunk


def test_d_memory_bound_stall_cannot_be_inverted():
    weight_read_s = COST.weight_bytes / COST.bytes_per_s  # ≈ 6.6 ms
    with pytest.raises(ValueError):
        ex.infer_chunk_tokens(weight_read_s * 0.9, 0, COST)
    with pytest.raises(ValueError):
        ex.infer_chunk_tokens(-1.0, 0, COST)
