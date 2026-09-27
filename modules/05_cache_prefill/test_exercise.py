"""模块 05 练习测试。离线、确定性：四个函数都是纯函数。

运行：make module N=05
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/05_cache_prefill
"""

from __future__ import annotations

import math

import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)

# 公共场景：上下文 4k token，其他正在跑的请求共 50k token，batch 共 60k token，
# prefill 1 万 token/s，CPU<->GPU 换入换出 20 万 token/s
BASE = dict(context_tokens=4_000, other_tokens=50_000, batch_tokens=60_000, prefill_tok_per_s=10_000.0)


# ================================================================== (a) min_waste_action


def test_a_short_tool_call_preserves():
    action, w = ex.min_waste_action(intercept_s=0.5, swap_tok_per_s=200_000.0, **BASE)
    assert w["discard"] == pytest.approx(0.4 * 54_000)
    assert w["preserve"] == pytest.approx(2_000)
    assert w["swap"] == pytest.approx(2_400)
    assert action == "preserve"


def test_a_long_tool_call_swaps_when_pcie_is_fast():
    action, _ = ex.min_waste_action(intercept_s=60.0, swap_tok_per_s=200_000.0, **BASE)
    assert action == "swap"


def test_a_long_tool_call_discards_when_swap_is_slow():
    action, w = ex.min_waste_action(intercept_s=60.0, swap_tok_per_s=5_000.0, **BASE)
    assert w["swap"] == pytest.approx(96_000)
    assert action == "discard"


def test_a_tie_prefers_preserve():
    action, w = ex.min_waste_action(intercept_s=0.6, swap_tok_per_s=200_000.0, **BASE)
    assert w["preserve"] == pytest.approx(w["swap"])
    assert action == "preserve"


def test_a_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.min_waste_action(intercept_s=1.0, swap_tok_per_s=0.0, **BASE)
    with pytest.raises(ValueError):
        ex.min_waste_action(4_000, 50_000, 60_000, -1.0, 10_000.0, 200_000.0)
    with pytest.raises(ValueError):
        ex.min_waste_action(4_000, 50_000, 60_000, math.nan, 10_000.0, 200_000.0)


# ================================================================== (b) route_min_est_ttft


def hbm(n: int) -> tuple[int, int, int]:
    return (n, 0, 0)


def test_b_trades_cache_hit_against_queue():
    # 只有 HBM 命中时退化成 queue + miss/prefill：实例 0 = 0.1 + 1.0；实例 1 = 0.2 + 0.2；实例 2 = 1.0 + 0.1
    hits = [hbm(0), hbm(8_000), hbm(9_000)]
    assert ex.route_min_est_ttft(10_000, hits, [0.1, 0.2, 1.0], 10_000.0) == 1


def test_b_long_queue_beats_full_hit():
    # 实例 1 全命中但排队 5 秒；实例 0 全不命中只要 1 秒
    assert ex.route_min_est_ttft(10_000, [hbm(0), hbm(10_000)], [0.0, 5.0], 10_000.0) == 0


def test_b_small_hbm_hit_vs_big_but_slow_ssd_hit():
    # 实例 0：HBM 命中 2k、排队 0.1 s → 0.1 + 0.8 = 0.9
    # 实例 1：SSD 命中 9k、不排队；SSD 5k token/s → 加载 1.8 s，比重算 0.9 s 还慢，按重算计 → 0.9 + 0.1 = 1.0
    hits = [(2_000, 0, 0), (0, 0, 9_000)]
    assert ex.route_min_est_ttft(10_000, hits, [0.1, 0.0], 10_000.0, load_tok_per_s=(math.inf, 5_000.0)) == 0
    # SSD 快到 50k token/s：实例 1 = 0.18 + 0.1 = 0.28，反超
    assert ex.route_min_est_ttft(10_000, hits, [0.1, 0.0], 10_000.0, load_tok_per_s=(math.inf, 50_000.0)) == 1


def test_b_slow_tier_counts_as_recompute_not_worse():
    # SSD 比重算慢时，SSD 命中至少不比没命中差（取 min(加载, 重算)）
    hits = [(0, 0, 0), (0, 0, 5_000)]
    assert ex.route_min_est_ttft(10_000, hits, [0.0, 0.0], 10_000.0, load_tok_per_s=(math.inf, 1_000.0)) == 0  # 并列取小下标


def test_b_dram_tier_is_charged():
    # 实例 0：DRAM 命中 8k，DRAM 40k token/s → 0.2 + 0.2 = 0.4；实例 1：HBM 命中 6k → 0.4 + 0.05 排队 = 0.45
    hits = [(0, 8_000, 0), (6_000, 0, 0)]
    assert ex.route_min_est_ttft(10_000, hits, [0.0, 0.05], 10_000.0, load_tok_per_s=(40_000.0, 5_000.0)) == 0


def test_b_decode_interference_shifts_choice():
    hits = [hbm(9_000), hbm(8_000)]
    assert ex.route_min_est_ttft(10_000, hits, [0.0, 0.0], 10_000.0) == 0
    assert ex.route_min_est_ttft(10_000, hits, [0.0, 0.0], 10_000.0, decode_interference_s=[0.5, 0.0]) == 1


def test_b_hits_are_clamped_in_tier_order():
    # 实例 0 的 HBM 命中已覆盖整条 prompt，SSD 段被截掉，不应再算加载时间
    hits = [(1_000, 0, 50_000), (0, 0, 0)]
    assert ex.route_min_est_ttft(1_000, hits, [0.0, 0.0], 10_000.0, load_tok_per_s=(math.inf, 1.0)) == 0


def test_b_tie_goes_to_lowest_index():
    assert ex.route_min_est_ttft(1_000, [hbm(500), hbm(500)], [0.1, 0.1], 10_000.0) == 0


def test_b_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [], [], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [hbm(1), hbm(2)], [0.1], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [hbm(1)], [0.1], 0.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(-1, [hbm(1)], [0.1], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [(-5, 0, 0)], [0.1], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [hbm(1)], [-0.1], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [hbm(1)], [math.nan], 10_000.0)
    with pytest.raises(ValueError):
        ex.route_min_est_ttft(1_000, [hbm(1)], [0.1], 10_000.0, load_tok_per_s=(0.0, 1.0))


# ================================================================== (c) route_affinity_with_cap


def test_c_best_match_among_instances_under_cap():
    assert ex.route_affinity_with_cap([100, 500, 300], [1, 10, 2], load_cap=5) == 2


def test_c_infinite_cap_is_pure_affinity():
    assert ex.route_affinity_with_cap([100, 500, 300], [1, 10, 2], load_cap=math.inf) == 1


def test_c_all_over_cap_falls_back_to_least_loaded():
    assert ex.route_affinity_with_cap([100, 500, 300], [9, 7, 8], load_cap=5) == 1


def test_c_ties_break_on_load_then_index():
    assert ex.route_affinity_with_cap([500, 500], [3, 2], load_cap=10) == 1
    assert ex.route_affinity_with_cap([500, 500], [2, 2], load_cap=10) == 0
    assert ex.route_affinity_with_cap([0, 0, 0], [4, 1, 1], load_cap=10) == 1


def test_c_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([], [], 1.0)
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([1, 2], [1], 1.0)
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([1, 2], [1.0, -1.0], 5.0)
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([1, 2], [1.0, math.nan], 5.0)
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([1, 2], [1.0, 2.0], math.nan)
    with pytest.raises(ValueError):
        ex.route_affinity_with_cap([-1, 2], [1.0, 2.0], 5.0)


# ================================================================== (d) load_or_recompute


def test_d_fast_tier_loads():
    choice, t_load, t_re = ex.load_or_recompute(10_000, 327_680, 25.0, 0.001, 5_000.0)
    assert t_load == pytest.approx(0.001 + 10_000 * 327_680 / 25e9)
    assert t_re == pytest.approx(2.0)
    assert choice == "load"


def test_d_slow_tier_recomputes():
    choice, t_load, t_re = ex.load_or_recompute(10_000, 327_680, 1.0, 0.001, 5_000.0)
    assert t_load > t_re
    assert choice == "recompute"


def test_d_breakeven_bandwidth_matches_formula():
    kv, rate = 327_680, 5_000.0
    b_star = kv * rate / 1e9  # GB/s
    assert ex.load_or_recompute(100_000, kv, b_star * 1.01, 0.0, rate)[0] == "load"
    assert ex.load_or_recompute(100_000, kv, b_star * 0.99, 0.0, rate)[0] == "recompute"


def test_d_fixed_latency_matters_for_short_prefixes():
    # 64 个 token：重算 6.4 ms；即使带宽无穷大，10 ms 的固定延迟也让加载不划算
    assert ex.load_or_recompute(64, 70_272, 1e6, 0.010, 10_000.0)[0] == "recompute"


def test_d_zero_tokens():
    assert ex.load_or_recompute(0, 70_272, 10.0, 0.002, 10_000.0) == ("recompute", 0.002, 0.0)


def test_d_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.load_or_recompute(10, 100, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        ex.load_or_recompute(-1, 100, 1.0, 0.0, 1.0)
