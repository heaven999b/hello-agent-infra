"""模块 07 实验：组内基线、多 agent 信用分配的偏差 / 方差 / 学习速度，以及反事实评估的系统代价。

运行：python modules/07_multi_agent_credit/lab.py      （M1 笔记本上 < 10 秒）

六部分：
0. 同一 prompt 采 G 个样本的组内基线（纯 numpy）：二阶矩比值、零优势组比例随难度分布的变化
1. 训练中途（非均匀策略）的单样本梯度：信噪比、期望梯度是否等于真梯度、额外评估次数；
   反事实查询分「完美 oracle」和「每次查询是一次独立带噪 rollout」两种口径
2. REINFORCE 训练 150 轮（20 个 seed，均值 ± 标准误）：学习速度（走完 80% 差距的轮数）与 150 轮后的回报
3. agent 数 n 变大时，共享奖励与 difference reward 的梯度方差
4. 把额外评估换算成 rollout：用模块 06 的 simulate_rollout 看 sync / async 各付多少
5. 盈亏判据：信噪比提升倍数 vs 有效吞吐下降倍数
"""

from __future__ import annotations

from itertools import product
from math import exp, factorial, lgamma

import numpy as np

from codesign_sim.credit import estimate_credit, gradient_bias, make_team_task, train_policies
from codesign_sim.rollout import RolloutConfig, simulate_rollout

N_AGENTS, N_ACTIONS = 6, 4


# ---------------------------------------------------------------- 小工具


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)  # 先减最大值，避免 exp 溢出
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_actions(policy: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """每个 agent 独立按 policy[i] 采样动作，返回 (n_samples, n_agents)。"""
    u = rng.random((n_samples, policy.shape[0], 1))
    return np.minimum((u > np.cumsum(policy, axis=1)[None]).sum(axis=2), policy.shape[1] - 1)


def per_sample_grads(credit: np.ndarray, actions: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """softmax logits 的单样本梯度：credit_i × (onehot(a_i) − π_i)，形状 (batch, n, K)。"""
    return credit[:, :, None] * (np.eye(policy.shape[1])[actions] - policy[None])


def snr(grads: np.ndarray, true_grad: np.ndarray) -> float:
    """单样本信噪比 = ‖真梯度‖² / 单样本梯度的总方差。"""
    noise = ((grads - grads.mean(axis=0)) ** 2).sum(axis=(1, 2)).mean()
    return float((true_grad**2).sum() / noise)


def shapley_by_rollouts(task, actions, ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """真实系统口径的 Shapley：每个联盟 v(S) 只用**一次带噪 rollout** 估计（缺席者按 ref 采样动作）。"""
    batch, n = actions.shape
    masks = list(product((0, 1), repeat=n))
    v = {}
    for m in masks:
        keep = np.array(m, dtype=bool)
        filled = np.where(keep[None], actions, sample_actions(ref, batch, rng))
        v[m] = task.expected_reward(filled) + task.noise_std * rng.standard_normal(batch)
    phi = np.zeros((batch, n))
    for m in masks:
        s = sum(m)
        if s == n:
            continue
        w = factorial(s) * factorial(n - s - 1) / factorial(n)
        for i in range(n):
            if not m[i]:
                phi[:, i] += w * (v[m[:i] + (1,) + m[i + 1 :]] - v[m])
    return phi


def steps_to_fraction(returns: list[float], best: float, frac: float = 0.9) -> int:
    """回报第一次达到「初始值 + frac × (最优 − 初始)」的迭代数；没达到返回 len(returns)。"""
    r = np.asarray(returns)
    hit = np.nonzero(r >= r[0] + frac * (best - r[0]))[0]
    return int(hit[0]) if len(hit) else len(r)


# ---------------------------------------------------------------- 第 0 部分


def part0_group_baselines() -> None:
    g = 8
    rng = np.random.default_rng(0)
    print(f"== 0. 同一 prompt 采 G={g} 个样本的组内基线（纯 numpy，20 万组）")
    print("-- (a) 奖励乘子的二阶矩 / 不用基线时的二阶矩")
    print(f"{'p':>6}{'true mean':>11}{'RLOO (MC)':>11}{'RLOO theory':>13}{'mean incl. self':>17}")
    for p in (0.05, 0.125, 0.5, 0.9):
        r = (rng.random((200_000, g)) < p).astype(float)
        m2 = (r**2).mean()
        loo = (r.sum(axis=1, keepdims=True) - r) / (g - 1)
        incl = r.mean(axis=1, keepdims=True)
        print(f"{p:>6}{((r - p) ** 2).mean() / m2:>11.3f}{((r - loo) ** 2).mean() / m2:>11.3f}"
              f"{(1 - p) * g / (g - 1):>13.3f}{((r - incl) ** 2).mean() / m2:>17.3f}")
    print("-- (b) 平均成功率都是 0.05 时，零优势（全对/全错）组的比例与超额采样倍数")
    print(f"{'difficulty distribution':<26}{'zero-adv (MC)':>14}{'closed form':>13}{'oversample':>12}")

    def lbeta(a: float, b: float) -> float:
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    for name, ab in (("all p = 0.05", None), ("Beta(1, 19)", (1.0, 19.0)), ("Beta(0.25, 4.75)", (0.25, 4.75))):
        p = np.full(200_000, 0.05) if ab is None else rng.beta(*ab, size=200_000)
        k = rng.binomial(g, p)
        mc = float(np.mean((k == 0) | (k == g)))
        if ab is None:
            exact = 0.95**g + 0.05**g
        else:
            a, b = ab
            exact = exp(lbeta(a, b + g) - lbeta(a, b)) + exp(lbeta(a + g, b) - lbeta(a, b))
        print(f"{name:<26}{mc:>14.3f}{exact:>13.3f}{1 / (1 - exact):>11.2f}x")


# ---------------------------------------------------------------- 第 1 部分

PART1_ROWS = [
    # (显示名, method, 关键字参数, 额外评估次数)
    ("shared", "shared", {}, 0),
    ("group mean (incl. self)", "group_baseline", {}, 0),
    ("group LOO", "group_baseline_loo", {}, 0),
    ("COMA, oracle cf", "counterfactual_baseline", {}, 0),
    ("COMA, noisy cf", "counterfactual_baseline", {"cf_noise": "independent"}, N_AGENTS * (N_ACTIONS - 1)),
    ("difference, oracle cf", "difference", {}, 0),
    ("difference, noisy cf", "difference", {"cf_noise": "independent"}, N_AGENTS),
    ("difference, CRN cf", "difference", {"cf_noise": "common"}, N_AGENTS),
    ("Shapley uniform, oracle", "shapley", {"shapley_ref": "uniform"}, 0),
    ("Shapley policy, oracle", "shapley", {"shapley_ref": "policy"}, 0),
]


def part1_gradient_quality() -> dict[str, float]:
    n_samples, n_rollout = 30_000, 4_000
    print(f"\n== 1. 训练中途（非均匀策略）的单样本梯度（n={N_AGENTS} agents, K={N_ACTIONS} actions）")
    task = make_team_task(N_AGENTS, N_ACTIONS, seed=0)
    rng = np.random.default_rng(0)
    policy = softmax(1.5 * rng.standard_normal((N_AGENTS, N_ACTIONS)))  # 模拟训练到一半、已经有偏好的策略
    true_grad = task.true_gradient(policy)
    a = sample_actions(policy, n_samples, rng)
    r = task.sample_rewards(a, rng)
    print(f"{'estimator':<26}{'SNR/sample':>11}{'vs shared':>11}{'cos(E[g],∇J)':>14}{'rel.err':>9}{'extra evals':>13}")
    out: dict[str, float] = {}
    for name, method, kw, extra in PART1_ROWS:
        c = estimate_credit(task, a, r, method, policy, rng=np.random.default_rng(1), **kw)
        s = snr(per_sample_grads(c, a, policy), true_grad)
        ref = {"shapley_ref": kw["shapley_ref"]} if "shapley_ref" in kw else {}
        bias = gradient_bias(task, method, policy, batch=8, **ref)  # 期望梯度（对噪声取期望）与真梯度比较
        out[name] = s
        print(f"{name:<26}{s:>11.4f}{s / out['shared']:>10.1f}x{bias['grad_cos']:>14.3f}{bias['grad_rel_err']:>9.3f}{extra:>13d}")
    # 真实系统口径：每个联盟一次带噪 rollout（期望与 oracle 版相同，所以 cos / rel.err 与上面对应行一致）
    for label, ref in (("uniform", np.full((N_AGENTS, N_ACTIONS), 1.0 / N_ACTIONS)), ("policy", policy)):
        name = f"Shapley {label}, rollouts"
        c = shapley_by_rollouts(task, a[:n_rollout], ref, rng)
        s = snr(per_sample_grads(c, a[:n_rollout], policy), true_grad)
        bias = gradient_bias(task, "shapley", policy, shapley_ref=label)
        out[name] = s
        print(f"{name:<26}{s:>11.4f}{s / out['shared']:>10.1f}x{bias['grad_cos']:>14.3f}{bias['grad_rel_err']:>9.3f}"
              f"{2**N_AGENTS - 1:>13d}")
    print("   (group mean 的 rel.err 按 batch=8 计：均值含自身 → 期望梯度缩小 (B-1)/B)")
    return out


# ---------------------------------------------------------------- 第 2 部分

PART2_ROWS = [
    ("shared", "shared", {}),
    ("group LOO", "group_baseline_loo", {}),
    ("COMA, noisy cf", "counterfactual_baseline", {"cf_noise": "independent"}),
    ("difference, noisy cf", "difference", {"cf_noise": "independent"}),
    ("Shapley uniform", "shapley", {"shapley_ref": "uniform"}),
    ("Shapley policy", "shapley", {"shapley_ref": "policy"}),
]


def part2_training() -> None:
    seeds, batch, lr, iters = range(20), 8, 0.3, 150
    print(f"\n== 2. REINFORCE 训练 {iters} 轮（n={N_AGENTS}, K={N_ACTIONS}, batch={batch}, lr={lr}；{len(seeds)} 个 seed，均值 ± 标准误）")
    print(f"{'estimator':<24}{'grad_var(first 50)':>19}{'iters to 80%':>18}{'at 75 / best':>16}{'at 150 / best':>18}")
    for name, method, kw in PART2_ROWS:
        gv, iters_to, half, ratio = [], [], [], []
        for s in seeds:
            out = train_policies(make_team_task(N_AGENTS, N_ACTIONS, seed=s), method, n_iters=iters, batch=batch,
                                 lr=lr, seed=s, **kw)
            gv.append(np.mean(out["grad_var"][:50]))
            iters_to.append(steps_to_fraction(out["returns"], out["best_return"], frac=0.8))
            half.append(out["returns"][iters // 2] / out["best_return"])
            ratio.append(out["final_return"] / out["best_return"])
        se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))  # noqa: E731
        print(f"{name:<24}{np.mean(gv):>19.3f}{np.mean(iters_to):>11.1f} ± {se(iters_to):>4.1f}"
              f"{np.mean(half):>9.3f} ± {se(half):.3f}{np.mean(ratio):>11.3f} ± {se(ratio):.3f}")


# ---------------------------------------------------------------- 第 3 部分


def part3_scaling() -> None:
    print(f"\n== 3. agent 数变大：早期梯度方差（前 30 轮平均，K={N_ACTIONS}，batch 64，3 个 seed）")
    rows = [("shared", "shared", {}), ("group LOO", "group_baseline_loo", {}),
            ("difference", "difference", {"cf_noise": "independent"})]
    print(f"{'n_agents':<10}" + "".join(f"{name:>14}" for name, _, _ in rows) + f"{'shared/difference':>20}")
    for n in (2, 4, 6, 12):
        vals = []
        for _, method, kw in rows:
            v = [np.mean(train_policies(make_team_task(n, N_ACTIONS, seed=s), method, n_iters=30, batch=64, lr=0.1,
                                        seed=s, **kw)["grad_var"]) for s in range(3)]
            vals.append(float(np.mean(v)))
        print(f"{n:<10}" + "".join(f"{v:>14.3f}" for v in vals) + f"{vals[0] / vals[2]:>19.1f}x")


# ---------------------------------------------------------------- 第 4 部分

PART4_ROWS = [
    # (显示名, 额外评估, 模式, 额外配置)
    ("shared, sync", 0, "sync", {}),
    ("difference (+n), sync", N_AGENTS, "sync", {}),
    ("COMA by sim (+n(K-1)), sync", N_AGENTS * (N_ACTIONS - 1), "sync", {}),
    ("exact Shapley (+2^n-1), sync", 2**N_AGENTS - 1, "sync", {}),
    ("shared, async", 0, "async", {}),
    ("difference (+n), async", N_AGENTS, "async", {}),
    ("COMA by sim (+n(K-1)), async", N_AGENTS * (N_ACTIONS - 1), "async", {}),
]


def part4_system_cost() -> dict[str, float]:
    print(f"\n== 4. 反事实评估的 rollout 代价（n={N_AGENTS} agents；模块 06 模拟器默认参数：64 workers，Pareto α=1.5，5 步）")
    print(f"{'setting':<32}{'trained/step':>13}{'step_time_s':>12}{'busy':>6}{'stale':>7}{'wasted':>8}{'useful/h':>10}"
          f"{'vs shared':>11}")
    base: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for name, extra, mode, kw in PART4_ROWS:
        out = simulate_rollout(RolloutConfig(batch_size=256 * (1 + extra), mode=mode, **kw), n_steps=5, seed=0)
        useful = out["traj_per_hour"] / (1 + extra)  # 只有原始联合 episode 是「新数据」，反事实分支只用来算信用
        base.setdefault(mode, useful)  # 每种模式的第一行是 shared
        ratios[name] = useful / base[mode]
        print(f"{name:<32}{out['mean_batch']:>13.0f}{out['step_time_s']:>12.0f}{out['worker_busy_frac']:>6.2f}"
              f"{out['mean_staleness']:>7.2f}{out['wasted_frac']:>8.3f}{useful:>10.0f}{ratios[name]:>10.2f}x")
    return ratios


# ---------------------------------------------------------------- 第 5 部分


def part5_break_even(snrs: dict[str, float], ratios: dict[str, float]) -> None:
    print("\n== 5. 盈亏判据：T_方法 / T_shared ≈ (SNR_shared / SNR_方法) / (有效吞吐比)，< 1 才赚")
    cases = [
        ("difference, noisy cf", "difference (+n), sync", "difference (+n), async"),
        ("COMA, noisy cf", "COMA by sim (+n(K-1)), sync", "COMA by sim (+n(K-1)), async"),
        ("Shapley policy, rollouts", "exact Shapley (+2^n-1), sync", None),
    ]
    print(f"{'estimator (real rollouts)':<28}{'SNR gain':>10}{'sync thr':>10}{'T sync':>9}{'async thr':>11}{'T async':>9}")
    for est, sync_key, async_key in cases:
        gain = snrs[est] / snrs["shared"]
        t_sync = (1 / gain) / ratios[sync_key]
        line = f"{est:<28}{gain:>9.1f}x{ratios[sync_key]:>9.2f}x{t_sync:>9.2f}"
        if async_key:
            line += f"{ratios[async_key]:>10.2f}x{(1 / gain) / ratios[async_key]:>9.2f}"
        else:
            line += f"{'—':>11}{'—':>9}"
        print(line)


if __name__ == "__main__":
    part0_group_baselines()
    snrs = part1_gradient_quality()
    part2_training()
    part3_scaling()
    ratios = part4_system_cost()
    part5_break_even(snrs, ratios)
