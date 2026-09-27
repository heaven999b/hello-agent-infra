"""模块 07 练习测试。纯 numpy、确定性。

运行：make module N=07
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/07_multi_agent_credit
"""

from __future__ import annotations

import numpy as np
import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)


# ------------------------------------------------------------------ group_advantages


def test_grpo_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2, size=(5, 8)).astype(float)
    r[0] = [1, 0, 0, 0, 0, 0, 0, 1]  # 保证至少一组有方差
    a = ex.group_advantages(r)
    assert a.shape == r.shape
    np.testing.assert_allclose(a.mean(axis=-1), 0.0, atol=1e-9)
    for row, s in zip(a, r.std(axis=-1)):
        if s > 0:
            assert row.std() == pytest.approx(1.0, abs=1e-4)


def test_grpo_constant_group_is_zero_not_nan():
    r = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    a = ex.group_advantages(r, mode="grpo")
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a[0], 0.0)
    np.testing.assert_array_equal(a[1], 0.0)
    assert a[2, 0] > 0 > a[2, 1]


def test_mean_mode_is_centering():
    r = np.array([[3.0, 1.0, 2.0, 6.0]])
    np.testing.assert_allclose(ex.group_advantages(r, mode="mean"), [[0.0, -2.0, -1.0, 3.0]])


def test_rloo_small_example():
    a = ex.group_advantages(np.array([1.0, 0.0, 0.0, 1.0]), mode="rloo")
    np.testing.assert_allclose(a, [1 - 1 / 3, -2 / 3, -2 / 3, 1 - 1 / 3])


def test_rloo_equals_scaled_mean_centering():
    rng = np.random.default_rng(1)
    r = rng.normal(size=(6, 5))
    g = r.shape[-1]
    np.testing.assert_allclose(
        ex.group_advantages(r, mode="rloo"), g / (g - 1) * ex.group_advantages(r, mode="mean"), atol=1e-12
    )


@pytest.mark.parametrize("mode", ["grpo", "mean", "rloo"])
def test_higher_dimensional_input_uses_last_axis(mode):
    rng = np.random.default_rng(2)
    r = rng.normal(size=(2, 3, 5))
    a = ex.group_advantages(r, mode=mode)
    assert a.shape == r.shape
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(a[i, j], ex.group_advantages(r[i, j][None], mode=mode)[0], atol=1e-12)
    np.testing.assert_allclose(a.mean(axis=-1), 0.0, atol=1e-9)


def test_rloo_needs_two_samples_and_bad_mode_raises():
    with pytest.raises(ValueError):
        ex.group_advantages(np.array([[1.0]]), mode="rloo")
    with pytest.raises(ValueError):
        ex.group_advantages(np.array([[1.0, 0.0]]), mode="ppo")


def test_group_advantages_does_not_mutate_input():
    r = np.array([[1.0, 0.0, 0.0, 1.0]])
    before = r.copy()
    for mode in ("grpo", "mean", "rloo"):
        ex.group_advantages(r, mode=mode)
    np.testing.assert_array_equal(r, before)


# ------------------------------------------------------------------ difference_rewards


def _counting(fn):
    calls: list = []

    def wrapped(x):
        calls.append(x)
        return fn(x)

    return wrapped, calls


def test_difference_rewards_additive_team():
    # 团队奖励 = 各 agent 动作价值之和，默认动作 0 的价值为 0 → D_i 就是 agent i 自己的价值
    worth = [0.0, 1.0, 2.5]
    reward_fn, calls = _counting(lambda a: sum(worth[x] for x in a))
    d = ex.difference_rewards(reward_fn, (1, 2, 0), default_action=0)
    np.testing.assert_allclose(d, [1.0, 2.5, 0.0])
    assert len(calls) == 3 + 1


def test_difference_rewards_all_or_nothing():
    # 只有三人都选 1 才成功：每个人都是「关键的」，D_i 都等于 1
    reward_fn, calls = _counting(lambda a: float(all(x == 1 for x in a)))
    d = ex.difference_rewards(reward_fn, [1, 1, 1], default_action=0)
    np.testing.assert_allclose(d, [1.0, 1.0, 1.0])
    # 失败时谁也没有贡献
    d_fail = ex.difference_rewards(reward_fn, [1, 0, 1], default_action=0)
    np.testing.assert_allclose(d_fail, [0.0, 0.0, 0.0])
    assert len(calls) == 2 * (3 + 1)


# ------------------------------------------------------------------ exact_shapley


def test_shapley_additive_game_recovers_weights():
    w = np.array([0.5, 2.0, -1.0, 3.0])
    phi = ex.exact_shapley(lambda s: float(sum(w[i] for i in s)), 4)
    np.testing.assert_allclose(phi, w, atol=1e-12)


def test_shapley_glove_game():
    # 经典手套博弈：0 号有左手套，1、2 号各有右手套；凑成一副价值 1
    def v(s: frozenset[int]) -> float:
        return float(0 in s and (1 in s or 2 in s))

    np.testing.assert_allclose(ex.exact_shapley(v, 3), [2 / 3, 1 / 6, 1 / 6], atol=1e-12)


def test_shapley_efficiency_symmetry_null_player():
    # 0、1 对称；3 号是 null player（加入任何联盟都不改变价值）
    def v(s: frozenset[int]) -> float:
        core = s - {3}
        return float(len(core) ** 2) + (2.0 if 2 in core else 0.0)

    phi = ex.exact_shapley(v, 4)
    assert phi.sum() == pytest.approx(v(frozenset(range(4))) - v(frozenset()))
    assert phi[0] == pytest.approx(phi[1])
    assert phi[3] == pytest.approx(0.0)


def test_shapley_evaluates_each_coalition_once():
    seen: list[frozenset[int]] = []

    def v(s: frozenset[int]) -> float:
        seen.append(s)
        return float(len(s))

    ex.exact_shapley(v, 5)
    assert len(seen) == len(set(seen)) == 2**5


def test_shapley_refuses_large_n():
    with pytest.raises(ValueError):
        ex.exact_shapley(lambda s: 0.0, 17)


# ------------------------------------------------------------------ branch_token_cost


def test_branch_token_cost_three_strategies():
    # 历史：20k 外部 + 3k 生成；每条分支追加 2k 外部、生成 1k；4 条分支
    args = dict(history_ext=20_000, history_gen=3_000, append_ext=2_000, branch_gen=1_000, n_branches=4)
    assert ex.branch_token_cost(**args, strategy="rerun") == {"prefill": 88_000, "decode": 16_000}
    assert ex.branch_token_cost(**args, strategy="restore") == {"prefill": 100_000, "decode": 4_000}
    assert ex.branch_token_cost(**args, strategy="kv_fork") == {"prefill": 8_000, "decode": 4_000}


def test_branch_token_cost_fork_never_worse_and_zero_branches():
    for strategy in ("rerun", "restore", "kv_fork"):
        assert ex.branch_token_cost(5, 5, 5, 5, 0, strategy) == {"prefill": 0, "decode": 0}
    fork = ex.branch_token_cost(100, 50, 10, 20, 3, "kv_fork")
    for other in ("rerun", "restore"):
        cost = ex.branch_token_cost(100, 50, 10, 20, 3, other)
        assert fork["prefill"] <= cost["prefill"] and fork["decode"] <= cost["decode"]


def test_branch_token_cost_rejects_bad_input():
    with pytest.raises(ValueError):
        ex.branch_token_cost(1, 1, 1, 1, 1, "teleport")
    with pytest.raises(ValueError):
        ex.branch_token_cost(-1, 1, 1, 1, 1, "kv_fork")
