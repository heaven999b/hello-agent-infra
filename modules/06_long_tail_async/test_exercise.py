"""模块 06 练习测试。离线、确定性：四个函数都是纯函数。

运行：make module N=06
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/06_long_tail_async
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)


# ================================================================== (a) sync_busy_frac_pareto


def test_busy_frac_n1_is_one() -> None:
    # 只有一条轨迹时没有屏障，也就没有气泡
    for alpha in (1.2, 1.5, 2.0, 3.0):
        assert ex.sync_busy_frac_pareto(alpha, 1) == pytest.approx(1.0, rel=1e-12)


def test_busy_frac_known_values() -> None:
    assert ex.sync_busy_frac_pareto(2.0, 64) == pytest.approx(0.1407, abs=5e-4)
    assert ex.sync_busy_frac_pareto(1.5, 256) == pytest.approx(0.0278, abs=5e-4)
    assert ex.sync_busy_frac_pareto(3.0, 1024) == pytest.approx(0.1099, abs=5e-4)


def test_busy_frac_large_n_no_overflow() -> None:
    v = ex.sync_busy_frac_pareto(2.0, 10**6)
    assert 0.0 < v < 0.01
    # 渐近式：busy ≈ α / ((α-1) Γ(1-1/α)) · n^(-1/α)
    asym = 2.0 / (1.0 * math.gamma(0.5)) * (10**6) ** -0.5
    assert v == pytest.approx(asym, rel=1e-3)


def test_busy_frac_monotone() -> None:
    # batch 越大越糟
    vals = [ex.sync_busy_frac_pareto(1.5, n) for n in (2, 8, 64, 512)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    # n 足够大时，尾越重越糟
    heavy = [ex.sync_busy_frac_pareto(a, 4096) for a in (3.0, 2.0, 1.5, 1.2)]
    assert all(a > b for a, b in zip(heavy, heavy[1:]))


def test_busy_frac_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    alpha, n = 3.0, 32
    t = (1.0 - rng.random((40_000, n))) ** (-1.0 / alpha)  # Pareto(α, 1) 采样
    mc = t.mean() / t.max(axis=1).mean()
    assert ex.sync_busy_frac_pareto(alpha, n) == pytest.approx(mc, rel=0.03)


def test_busy_frac_invalid() -> None:
    with pytest.raises(ValueError):
        ex.sync_busy_frac_pareto(1.0, 10)
    with pytest.raises(ValueError):
        ex.sync_busy_frac_pareto(2.0, 0)


# ================================================================== (b) truncated_is_weights


def test_tis_on_policy_all_ones() -> None:
    lp = np.log([0.5, 0.2, 0.9, 0.7])
    for level in ("token", "sequence"):
        w = ex.truncated_is_weights(lp, lp, cap=2.0, level=level)
        assert isinstance(w, np.ndarray)
        assert w.shape == (4,)
        np.testing.assert_allclose(w, 1.0)


def test_tis_token_level() -> None:
    lr = np.log([0.5, 0.5, 0.5])
    lp = np.log([0.25, 0.5, 1.0])  # 比率 0.5, 1.0, 2.0
    np.testing.assert_allclose(ex.truncated_is_weights(lp, lr, cap=1.5, level="token"), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(ex.truncated_is_weights(lp, lr, cap=5.0, level="token"), [0.5, 1.0, 2.0])


def test_tis_sequence_level_product_and_broadcast() -> None:
    lr = np.log([0.5, 0.5, 0.5])
    lp = np.log([0.6, 0.6, 0.6])  # 每个 token 比率 1.2，乘积 1.728
    w = ex.truncated_is_weights(lp, lr, cap=10.0, level="sequence")
    np.testing.assert_allclose(w, [1.728] * 3, rtol=1e-9)
    w_capped = ex.truncated_is_weights(lp, lr, cap=1.5, level="sequence")
    np.testing.assert_allclose(w_capped, [1.5] * 3)


def test_tis_length_trap() -> None:
    # 每个 token 比率 1.1：序列级随长度爆炸（约 13,780），token 级不随长度变化
    lr = np.zeros(100)
    lp = np.full(100, math.log(1.1))
    seq = ex.truncated_is_weights(lp, lr, cap=1e9, level="sequence")
    tok = ex.truncated_is_weights(lp, lr, cap=1e9, level="token")
    assert seq[0] == pytest.approx(1.1**100, rel=1e-6)
    np.testing.assert_allclose(tok, 1.1, rtol=1e-9)


def test_tis_overflow_safe_and_capped() -> None:
    lr = np.array([-1000.0, 0.0])
    lp = np.array([0.0, -1000.0])
    with np.errstate(over="raise"):
        w = ex.truncated_is_weights(lp, lr, cap=3.0, level="token")
    assert np.all(np.isfinite(w))
    assert w[0] == pytest.approx(3.0)
    assert w[1] == pytest.approx(math.exp(-20.0))
    rng = np.random.default_rng(1)
    for level in ("token", "sequence"):
        w = ex.truncated_is_weights(rng.normal(size=50), rng.normal(size=50), cap=2.0, level=level)
        assert np.all(w <= 2.0 + 1e-12) and np.all(w > 0)


def test_tis_empty_and_invalid() -> None:
    assert ex.truncated_is_weights([], [], cap=2.0).shape == (0,)
    with pytest.raises(ValueError):
        ex.truncated_is_weights([0.0, 0.0], [0.0], cap=2.0)
    with pytest.raises(ValueError):
        ex.truncated_is_weights([0.0], [0.0], cap=0.0)
    # 几何平均不是 IS 权重（verl 只把它用于拒绝采样），不接受
    for bad in ("geometric", "trajectory"):
        with pytest.raises(ValueError):
            ex.truncated_is_weights([0.0], [0.0], cap=2.0, level=bad)


# ================================================================== (c) plan_async_pools


def test_plan_docstring_example_matches_areal_ratio() -> None:
    plan = ex.plan_async_pools(64, 1.0, 3.0, 512, 60.0)
    assert plan["n_gen"] == 48 and plan["n_train"] == 16  # 3:1，与 AReaL 一致
    assert plan["traj_per_s"] == pytest.approx(48.0)
    assert plan["train_step_s"] == pytest.approx(512 / 48)
    assert plan["min_staleness"] == 6  # ceil(60 / 10.67) = ceil(5.625)
    assert plan["admission_window"] == 7 * 512
    assert plan["bottleneck"] == "train"


def test_plan_tie_prefers_fewer_trainers_and_exact_ratio() -> None:
    # g = τ：5/5 是唯一最优；p99 恰好是步长的整数倍时不应多算一步
    plan = ex.plan_async_pools(10, 1.0, 1.0, 10, 4.0)
    assert (plan["n_gen"], plan["n_train"]) == (5, 5)
    assert plan["train_step_s"] == pytest.approx(2.0)
    assert plan["min_staleness"] == 2
    # 奇数卡、g = τ：4/5 与 5/4 吞吐都是 4，取训练卡少的
    plan = ex.plan_async_pools(9, 1.0, 1.0, 8, 0.0)
    assert (plan["n_gen"], plan["n_train"]) == (5, 4)
    assert plan["bottleneck"] == "train"
    assert plan["min_staleness"] == 0
    assert plan["admission_window"] == 8


def test_plan_generation_bound() -> None:
    # 训练极快：只需 1 张训练卡，瓶颈在生成
    plan = ex.plan_async_pools(8, 0.5, 100.0, 64, 30.0)
    assert (plan["n_gen"], plan["n_train"]) == (7, 1)
    assert plan["bottleneck"] == "gen"
    assert plan["traj_per_s"] == pytest.approx(3.5)
    assert plan["min_staleness"] == math.ceil(30.0 / (64 / 3.5))


def test_plan_longer_tail_needs_more_staleness() -> None:
    a = ex.plan_async_pools(128, 2.0, 5.0, 1024, 120.0)
    b = ex.plan_async_pools(128, 2.0, 5.0, 1024, 600.0)
    assert (a["n_gen"], a["n_train"]) == (b["n_gen"], b["n_train"])
    assert b["min_staleness"] > a["min_staleness"]
    assert b["admission_window"] == (b["min_staleness"] + 1) * 1024


def test_plan_invalid() -> None:
    with pytest.raises(ValueError):
        ex.plan_async_pools(1, 1.0, 1.0, 8, 1.0)
    with pytest.raises(ValueError):
        ex.plan_async_pools(8, 0.0, 1.0, 8, 1.0)
    with pytest.raises(ValueError):
        ex.plan_async_pools(8, 1.0, 1.0, 0, 1.0)
    with pytest.raises(ValueError):
        ex.plan_async_pools(8, 1.0, 1.0, 8, -1.0)


# ================================================================== (d) lpt_schedule


def test_lpt_docstring_example() -> None:
    makespan, assignment = ex.lpt_schedule([7, 5, 4, 3, 3, 2], 2)
    assert makespan == pytest.approx(12.0)
    assert assignment == [0, 1, 1, 0, 1, 0]


def test_lpt_edge_cases() -> None:
    assert ex.lpt_schedule([], 3) == (0.0, [])
    makespan, assignment = ex.lpt_schedule([3.0, 9.0], 4)
    assert makespan == pytest.approx(9.0)
    assert assignment == [1, 0]  # 9 先派给 worker 0，3 派给 worker 1
    with pytest.raises(ValueError):
        ex.lpt_schedule([1.0], 0)
    for bad in ([1.0, -2.0], [float("nan")], [1.0, float("inf")]):
        with pytest.raises(ValueError):
            ex.lpt_schedule(bad, 2)


def test_lpt_assignment_consistent() -> None:
    rng = np.random.default_rng(2)
    lengths = list(rng.pareto(1.5, size=40) + 1.0)
    makespan, assignment = ex.lpt_schedule(lengths, 6)
    loads = [0.0] * 6
    for length, w in zip(lengths, assignment):
        loads[w] += length
    assert makespan == pytest.approx(max(loads))
    assert makespan >= max(lengths) - 1e-12
    assert makespan >= sum(lengths) / 6 - 1e-12


def _brute_force_opt(lengths: list[float], m: int) -> float:
    best = math.inf
    for assign in itertools.product(range(m), repeat=len(lengths)):
        loads = [0.0] * m
        for length, w in zip(lengths, assign):
            loads[w] += length
        best = min(best, max(loads))
    return best


def test_lpt_within_graham_bound() -> None:
    # Graham (1969)：LPT 的 makespan ≤ (4/3 - 1/(3m)) · OPT
    rng = np.random.default_rng(3)
    for m in (2, 3):
        for _ in range(5):
            lengths = [float(x) for x in rng.integers(1, 20, size=7)]
            lpt, _ = ex.lpt_schedule(lengths, m)
            opt = _brute_force_opt(lengths, m)
            assert opt - 1e-9 <= lpt <= (4 / 3 - 1 / (3 * m)) * opt + 1e-9
