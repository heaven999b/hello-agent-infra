"""credit：Shapley 闭式解（均匀 / 策略参考）、估计器的方差与策略梯度偏差、共同随机数、学习速度区间。"""

import math
from itertools import combinations, product

import numpy as np
import pytest

from codesign_sim.credit import (
    METHODS,
    credit_error_stats,
    estimate_credit,
    gradient_bias,
    make_team_task,
    train_policies,
)


def brute_force_shapley(task, a_row, ref):
    n, K = task.n_agents, task.n_actions

    def v(S):  # 缺席者按参考分布 ref 取期望
        others = [i for i in range(n) if i not in S]
        total = 0.0
        for g in product(range(K), repeat=len(others)):
            x = np.array(a_row)
            x[others] = g
            total += np.prod([ref[i, gi] for i, gi in zip(others, g)]) * task.expected_reward(x[None])[0]
        return total

    phi = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(n):
            for S in combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi[i] += w * (v(set(S) | {i}) - v(set(S)))
    return phi


def test_true_contribution_is_exact_shapley():
    task = make_team_task(4, 3, seed=1)
    a = np.array([[0, 2, 1, 1], [2, 2, 0, 1]])
    uniform = np.full((4, 3), 1 / 3)
    policy = np.random.default_rng(0).dirichlet(np.ones(3), size=4)
    for row, p in zip(a, task.true_contribution(a)):
        assert np.allclose(p, brute_force_shapley(task, row, uniform))
    for row, p in zip(a, task.true_contribution(a, policy)):
        assert np.allclose(p, brute_force_shapley(task, row, policy))
    mean_f = task.expected_reward(np.array(list(product(range(3), repeat=4)))).mean()
    assert np.allclose(task.true_contribution(a).sum(axis=1), task.expected_reward(a) - mean_f)  # 效率性


def test_counterfactual_rewards():
    task = make_team_task(5, 3, seed=2)
    a = np.random.default_rng(0).integers(0, 3, size=(4, 5))
    F = task.counterfactual_rewards(a)
    for b, i, x in product(range(4), range(5), range(3)):
        y = a[b].copy()
        y[i] = x
        assert F[b, i, x] == pytest.approx(task.expected_reward(y[None])[0])


def test_estimator_error_variance_and_offset():
    task = make_team_task(6, 4, seed=0)
    st = {m: credit_error_stats(task, m, n_samples=4000) for m in METHODS}
    assert st["difference"]["error_var"] < 0.5 * st["shared"]["error_var"]
    assert st["shapley"]["error_var"] < 0.1 * st["shared"]["error_var"]
    assert st["counterfactual_baseline"]["error_var"] < st["shared"]["error_var"]
    assert abs(st["group_baseline"]["mean_offset_vs_shapley"]) < 0.05
    assert st["shared"]["bias"] == st["shared"]["mean_offset_vs_shapley"]  # 旧字段名保留为别名
    assert st["shapley"]["corr"] > st["difference"]["corr"] > st["shared"]["corr"]
    # "相对 Shapley 的偏移" ≠ 策略梯度偏差：difference 有偏移，但期望梯度无偏；shapley(uniform) 相反
    assert abs(st["difference"]["mean_offset_vs_shapley"]) > 0.1 and st["difference"]["grad_rel_err"] < 1e-9
    assert abs(st["shapley"]["mean_offset_vs_shapley"]) < 0.01 and st["shapley"]["grad_rel_err"] > 0.05


def test_policy_gradient_bias():
    task = make_team_task(5, 3, seed=2)
    pi = np.random.default_rng(1).dirichlet(np.ones(3), size=5)
    for m in ("shared", "group_baseline_loo", "difference", "counterfactual_baseline"):
        assert gradient_bias(task, m, pi)["grad_rel_err"] < 1e-9
    assert gradient_bias(task, "shapley", pi, shapley_ref="policy")["grad_rel_err"] < 1e-9  # 以当前策略为背景：无偏
    assert gradient_bias(task, "shapley", pi)["grad_rel_err"] > 0.05  # 以均匀策略为背景：有偏
    gb = gradient_bias(task, "group_baseline", pi, batch=16)
    assert gb["grad_rel_err"] == pytest.approx(1 / 16) and gb["grad_cos"] == pytest.approx(1.0)  # 只缩放 (B-1)/B


def test_group_baseline_loo_and_crn():
    task = make_team_task(6, 3, seed=0)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=(8, 6))
    r = task.sample_rewards(a, rng)
    loo = estimate_credit(task, a, r, "group_baseline_loo")
    assert loo[0, 0] == pytest.approx(r[0] - r[1:].mean())
    # 共同随机数：反事实与实际 rollout 共享噪声，差分后噪声完全抵消
    crn = estimate_credit(task, a, r, "difference", cf_noise="common")
    F = task.counterfactual_rewards(a)
    assert np.allclose(crn, task.expected_reward(a)[:, None] - F[:, :, 0])
    t8 = make_team_task(8, 4, seed=0)
    gv = {c: np.mean(train_policies(t8, "difference", n_iters=40, cf_noise=c)["grad_var"])
          for c in ("common", "none", "independent")}
    assert gv["common"] < gv["none"] < gv["independent"]


def test_training_grad_variance_ordering():
    task = make_team_task(8, 4, seed=0)
    gv = {m: np.mean(train_policies(task, m, n_iters=40)["grad_var"]) for m in METHODS}
    assert gv["difference"] < gv["shared"] and gv["shapley"] < gv["difference"]
    assert gv["group_baseline"] < gv["shared"]


def test_variance_turns_into_learning_speed_only_with_many_agents():
    # 8 个 agent、batch 64：各估计器几乎一样；32 个 agent、batch 8：shared 的每 agent 信噪比 ~1/n，明显落后
    few = make_team_task(8, 4, seed=0)
    f_sh, f_df = (train_policies(few, m, n_iters=150)["final_return"] for m in ("shared", "difference"))
    assert f_sh == pytest.approx(f_df, rel=0.05)
    many = make_team_task(32, 4, seed=0)
    m_sh, m_df = (train_policies(many, m, n_iters=150, batch=8)["final_return"] for m in ("shared", "difference"))
    assert m_sh < 0.8 * m_df


def test_final_return_is_after_last_update_and_deterministic():
    task = make_team_task(6, 4, seed=3)
    out = train_policies(task, "difference")
    assert len(out["returns"]) == len(out["grad_var"]) == 300
    assert out["returns"][-1] > out["returns"][0] + 1.0
    assert out["final_return"] != out["returns"][-1]  # returns[t] 是第 t 次更新之前
    assert out["final_return"] <= out["best_return"] + 1e-9
    assert train_policies(task, "difference")["returns"] == out["returns"]


def test_bad_arguments():
    task = make_team_task(3, 2)
    with pytest.raises(ValueError):
        estimate_credit(task, np.zeros((1, 3), dtype=int), np.zeros(1), "magic")
    with pytest.raises(ValueError):
        estimate_credit(task, np.zeros((2, 3), dtype=int), np.zeros(2), "difference", cf_noise="x")
