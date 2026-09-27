"""多 agent 信用分配（credit assignment）：一个"已知真实个体贡献"的合作博弈。

任务：n_agents 个 agent 各选一个动作 a_i ∈ {0..n_actions-1}，团队只拿到一个标量奖励
    r = f(a) + ε,   f(a) = Σ_i W[i, a_i] + Σ_{i<j} Q_ij(a_i, a_j),   ε ~ N(0, noise_std²)
W 是个体项、Q 是两两协同项（量级小于 W）。

真实个体贡献 = f 的 Shapley 值（缺席的 agent 按参考分布 ρ 取期望；默认 ρ = 均匀随机动作）。
对这种"一元 + 二元"博弈有闭式解：
    φ_i = (W[i,a_i] - E_ρ W[i,·]) + ½ Σ_{j≠i} [ Q_ij(a_i,a_j) - E_{b~ρ_i} Q_ij(b,a_j) + E_{b~ρ_j} Q_ij(a_i,b) - E_{ρ_i,ρ_j} Q_ij ]
（每对协同项的"交互收益"两人平分；Σ_i φ_i = f(a) - E_ρ[f]）

估计器（都只拿到观测到的 r，外加对 f 的"反事实查询"权限——相当于有一个模拟器/critic）：
- shared：每个 agent 都拿 r（最朴素；方差含所有其他 agent 的贡献）
- group_baseline：r - batch 平均（GRPO 风格；均值**含自身**，期望梯度被缩放 (B-1)/B，方向不变）
- group_baseline_loo：r - 其余样本的平均（leave-one-out，严格无偏）
- difference：r - f(a_{-i}, c_i)，c_i = 默认动作 0（Wolpert & Tumer 的 difference reward）
- counterfactual_baseline：r - Σ_b π_i(b) f(a_{-i}, b)（COMA 式的反事实基线，π 为当前策略，默认均匀）
- shapley：φ_i(a) + (r - f(a)) / n（精确 Shapley，噪声平摊；真实系统里需要 2^n 次查询或学出来的价值模型）。
  shapley_ref="uniform"（默认）以均匀随机队友为背景——方差最低但**不是**当前策略的梯度（有偏）；
  shapley_ref="policy" 以当前策略为背景——对独立策略的 REINFORCE 严格无偏

反事实查询的噪声（cf_noise，只影响 difference / counterfactual_baseline）：
- "none"（默认）：反事实用无噪声的 f（完美 critic）
- "independent"：每次反事实查询都是一次独立的带噪仿真（真实的"再跑一遍"）
- "common"：共同随机数（CRN）——反事实查询与实际 rollout 共享外生噪声 ε，差分时噪声完全抵消
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

METHODS = ("shared", "group_baseline", "group_baseline_loo", "difference", "shapley", "counterfactual_baseline")
_CF_NOISE = ("none", "independent", "common")


@dataclass
class TeamTask:
    n_agents: int
    n_actions: int
    unary: np.ndarray  # W: (n_agents, n_actions)
    pairwise: np.ndarray  # Q: (n_agents, n_agents, n_actions, n_actions)，Q[j,i] = Q[i,j].T，对角为 0
    noise_std: float = 0.5
    default_action: int = 0

    # ---------------------------------------------------------------- 奖励

    def expected_reward(self, joint_actions: np.ndarray) -> np.ndarray:
        """f(a)：无噪声团队奖励。joint_actions 形状 (batch, n_agents)。"""
        a = np.atleast_2d(joint_actions)
        idx = np.arange(self.n_agents)
        unary = self.unary[idx, a].sum(axis=1)
        # Q 是对称存储（Q[j,i] = Q[i,j].T、对角为 0），所以 Σ_{i<j} = ½ Σ_{i≠j}
        pair = self.pairwise[idx[:, None], idx[None, :], a[:, :, None], a[:, None, :]]  # (batch, n, n)
        return unary + 0.5 * pair.sum(axis=(1, 2))

    def sample_rewards(self, joint_actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        f = self.expected_reward(joint_actions)
        return f + self.noise_std * rng.standard_normal(f.shape)

    def counterfactual_rewards(self, joint_actions: np.ndarray) -> np.ndarray:
        """F[b, i, x] = f(a_{-i}, a_i ← x)：把第 i 个 agent 的动作换成 x、其他不变时的奖励（无噪声）。"""
        a = np.atleast_2d(joint_actions)
        n = a.shape[1]
        f = self.expected_reward(a)
        idx = np.arange(n)
        # 第 i 个 agent 取 x 时，它与其他人的协同项之和：C[b, i, x] = Σ_j Q[i, j][x, a_j]
        qa = self.pairwise[:, idx, :, a]  # (batch, n_j, n_i, K)
        contrib = qa.sum(axis=1)  # (batch, n_i, K)
        own = self.unary[idx, a]  # (batch, n)
        own_pair = np.take_along_axis(contrib, a[:, :, None], axis=2)[:, :, 0]
        base = f[:, None] - own - own_pair  # 去掉 agent i 的所有项
        return base[:, :, None] + self.unary[None, :, :] + contrib

    def true_contribution(self, joint_actions: np.ndarray, ref_probs: np.ndarray | None = None) -> np.ndarray:
        """个体贡献 φ（闭式 Shapley 值），形状 (batch, n_agents)。

        ref_probs：缺席 agent 的参考分布 (n_agents, n_actions)；默认 None = 均匀（这就是本任务定义的"真实贡献"）。
        """
        a = np.atleast_2d(joint_actions)
        n, K = self.n_agents, self.n_actions
        rho = np.full((n, K), 1.0 / K) if ref_probs is None else np.asarray(ref_probs, dtype=float)
        idx = np.arange(n)
        I, J = idx[:, None], idx[None, :]
        ai, aj = a[:, :, None], a[:, None, :]
        Q = self.pairwise
        col_m = np.einsum("ix,ijxy->ijy", rho, Q)  # E_{b~ρ_i} Q_ij(b, y)
        row_m = np.einsum("ijxy,jy->ijx", Q, rho)  # E_{b~ρ_j} Q_ij(x, b)
        both = np.einsum("ix,ijxy,jy->ij", rho, Q, rho)  # E_{ρ_i,ρ_j} Q_ij
        q = Q[I, J, ai, aj]  # Q_ij(a_i, a_j)，形状 (batch, n_i, n_j)
        pair_share = 0.5 * (q - col_m[I, J, aj] + row_m[I, J, ai] - both[None]).sum(axis=2)  # 对角项全为 0
        unary_ref = (self.unary * rho).sum(axis=1)
        return self.unary[idx, a] - unary_ref[None, :] + pair_share

    def expected_return(self, probs: np.ndarray) -> float:
        """独立策略 π_i（形状 (n_agents, n_actions)）下的期望团队奖励（精确计算）。"""
        pair = np.einsum("ix,ijxy,jy->", probs, self.pairwise, probs)  # = 2 × Σ_{i<j}
        return float((self.unary * probs).sum() + 0.5 * pair)

    def true_gradient(self, probs: np.ndarray) -> np.ndarray:
        """∇_logits E_π[f]（独立 softmax 策略），形状 (n_agents, n_actions)，精确计算。"""
        m = self.unary + np.einsum("ijxy,jy->ix", self.pairwise, probs)  # E[f | a_i = x] 中依赖 x 的部分
        return probs * (m - (probs * m).sum(axis=1, keepdims=True))

    def best_return(self) -> float:
        """最优联合动作的奖励（动作空间 ≤ 2^16 时穷举，否则返回 nan）。"""
        if self.n_actions ** self.n_agents > 2 ** 16:
            return float("nan")
        grid = np.array(list(product(range(self.n_actions), repeat=self.n_agents)))
        return float(self.expected_reward(grid).max())


def make_team_task(n_agents: int, n_actions: int, seed: int = 0, *, pair_scale: float = 0.3,
                   noise_std: float = 0.5) -> TeamTask:
    """随机生成一个合作任务：W ~ N(0,1)，协同项 ~ N(0, pair_scale²)，观测噪声 N(0, noise_std²)。"""
    if n_agents < 2 or n_actions < 2:
        raise ValueError("n_agents 与 n_actions 都必须 >= 2")
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_agents, n_actions))
    Q = np.zeros((n_agents, n_agents, n_actions, n_actions))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            Q[i, j] = pair_scale * rng.standard_normal((n_actions, n_actions))
            Q[j, i] = Q[i, j].T
    return TeamTask(n_agents, n_actions, W, Q, noise_std=noise_std)


def _uniform(task: TeamTask) -> np.ndarray:
    return np.full((task.n_agents, task.n_actions), 1.0 / task.n_actions)


def _noisy_cf(task: TeamTask, a: np.ndarray, r: np.ndarray, cf_noise: str, rng: np.random.Generator | None) -> np.ndarray:
    F = task.counterfactual_rewards(a)
    if cf_noise == "none":
        return F
    if cf_noise == "common":  # 共同随机数：反事实与实际 rollout 共享外生噪声
        return F + (r - task.expected_reward(a))[:, None, None]
    rng = rng if rng is not None else np.random.default_rng(0)
    return F + task.noise_std * rng.standard_normal(F.shape)  # 每次查询一次独立仿真


def estimate_credit(task: TeamTask, joint_actions: np.ndarray, rewards: np.ndarray, method: str,
                    policies: np.ndarray | None = None, *, shapley_ref: str = "uniform",
                    cf_noise: str = "none", rng: np.random.Generator | None = None) -> np.ndarray:
    """给每个样本的每个 agent 估计信用，返回 (batch, n_agents)。

    policies：当前策略概率 (n_agents, n_actions)，counterfactual_baseline 与 shapley_ref="policy" 用；默认均匀。
    shapley_ref："uniform" | "policy"（见模块说明）。cf_noise："none" | "independent" | "common"；
    rng：cf_noise="independent" 时的随机数发生器（默认固定 seed 0）。
    """
    if cf_noise not in _CF_NOISE:
        raise ValueError(f"cf_noise 必须是 {_CF_NOISE} 之一")
    a = np.atleast_2d(joint_actions)
    r = np.asarray(rewards, dtype=float).reshape(-1)
    n = task.n_agents
    pi = _uniform(task) if policies is None else np.asarray(policies, dtype=float)
    if method == "shared":
        return np.repeat(r[:, None], n, axis=1)
    if method == "group_baseline":
        return np.repeat((r - r.mean())[:, None], n, axis=1)
    if method == "group_baseline_loo":
        if len(r) < 2:
            raise ValueError("group_baseline_loo 需要 batch >= 2")
        loo = (r.sum() - r) / (len(r) - 1)
        return np.repeat((r - loo)[:, None], n, axis=1)
    if method == "difference":
        F = _noisy_cf(task, a, r, cf_noise, rng)
        return r[:, None] - F[:, :, task.default_action]
    if method == "counterfactual_baseline":
        F = _noisy_cf(task, a, r, cf_noise, rng)
        return r[:, None] - (F * pi[None, :, :]).sum(axis=2)
    if method == "shapley":
        if shapley_ref not in ("uniform", "policy"):
            raise ValueError("shapley_ref 必须是 'uniform' 或 'policy'")
        f = task.expected_reward(a)
        ref = None if shapley_ref == "uniform" else pi
        return task.true_contribution(a, ref) + ((r - f) / n)[:, None]
    raise ValueError(f"method 必须是 {METHODS} 之一，收到 {method!r}")


def expected_gradient(task: TeamTask, method: str, probs: np.ndarray | None = None, *, batch: int = 64,
                      shapley_ref: str = "uniform", n_mc: int = 20_000, seed: int = 0) -> np.ndarray:
    """估计器的**期望**梯度 E[credit_i · ∇log π_i(a_i)]（对观测噪声取期望，所以噪声项消失）。

    联合动作空间 ≤ 2^16 时精确枚举，否则用 n_mc 个样本的蒙特卡洛（分块计算，控制内存）。
    group_baseline 按 batch 大小计入 (B-1)/B 缩放。
    """
    pi = _uniform(task) if probs is None else np.asarray(probs, dtype=float)
    n, K = task.n_agents, task.n_actions
    if K**n <= 2**16:
        a_all = np.array(list(product(range(K), repeat=n)))
        w_all = np.prod(pi[np.arange(n), a_all], axis=1)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((n_mc, n, 1))
        a_all = np.minimum((u > np.cumsum(pi, axis=1)[None]).sum(axis=2), K - 1)
        w_all = np.full(n_mc, 1.0 / n_mc)
    mean_f = task.expected_return(pi)
    g = np.zeros((n, K))
    for lo in range(0, len(a_all), 8192):
        a, w = a_all[lo : lo + 8192], w_all[lo : lo + 8192]
        f = task.expected_reward(a)
        if method == "shared":
            c = np.repeat(f[:, None], n, axis=1)
        elif method == "group_baseline":
            c = np.repeat(((1 - 1 / batch) * (f - mean_f))[:, None], n, axis=1)
        elif method == "group_baseline_loo":
            c = np.repeat((f - mean_f)[:, None], n, axis=1)
        else:  # 其余估计器对 f 的依赖与 r 相同（噪声项期望为 0），直接用 f 代替 r
            c = estimate_credit(task, a, f, method, pi, shapley_ref=shapley_ref)
        score = np.eye(K)[a] - pi[None]  # (N, n, K)
        g += (w[:, None, None] * c[:, :, None] * score).sum(axis=0)
    return g


def gradient_bias(task: TeamTask, method: str, probs: np.ndarray | None = None, **kw) -> dict:
    """期望梯度与真梯度 ∇E_π[f] 的相对误差与余弦相似度（衡量"对策略梯度有没有偏"）。"""
    pi = _uniform(task) if probs is None else probs
    g, g_true = expected_gradient(task, method, pi, **kw), task.true_gradient(pi)
    rel = float(np.linalg.norm(g - g_true) / np.linalg.norm(g_true))
    cos = float((g * g_true).sum() / (np.linalg.norm(g) * np.linalg.norm(g_true)))
    return {"grad_rel_err": rel, "grad_cos": cos}


def credit_error_stats(task: TeamTask, method: str, n_samples: int = 4096, seed: int = 0, *,
                       grad_probs: np.ndarray | None = None) -> dict:
    """在均匀随机联合动作上比较估计值与真实贡献（均匀参考的 Shapley），并给出策略梯度偏差。

    - mean_offset_vs_shapley：估计值减 Shapley 真值的平均偏移（旧字段名 bias 保留为别名；
      它**不是**策略梯度的偏差——常数偏移不改变期望梯度）
    - error_var：误差方差；corr：与真值的相关系数
    - grad_rel_err / grad_cos：在策略 grad_probs 处，期望梯度相对真梯度的误差（见 gradient_bias）。
      默认 grad_probs = 每个 agent 一个 Dirichlet(1) 随机策略（固定 seed），代表"训练到一半、已不均匀"的策略——
      在均匀策略处 shapley(uniform) 恰好无偏，看不出问题
    """
    if grad_probs is None:
        grad_probs = np.random.default_rng(seed + 1).dirichlet(np.ones(task.n_actions), size=task.n_agents)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, task.n_actions, size=(n_samples, task.n_agents))
    r = task.sample_rewards(a, rng)
    est = estimate_credit(task, a, r, method)
    true = task.true_contribution(a)
    err = est - true
    corr = np.mean([np.corrcoef(est[:, i], true[:, i])[0, 1] for i in range(task.n_agents)])
    offset = float(err.mean())
    return {"mean_offset_vs_shapley": offset, "bias": offset, "error_var": float(err.var(axis=0).mean()),
            "corr": float(corr), **gradient_bias(task, method, grad_probs)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_policies(task: TeamTask, method: str, n_iters: int = 300, batch: int = 64, lr: float = 0.1,
                   seed: int = 0, *, shapley_ref: str = "uniform", cf_noise: str = "none") -> dict:
    """独立 softmax 策略 + REINFORCE，用指定估计器的信用做每个 agent 的回报。

    返回：
    - returns：长度 n_iters，returns[t] 是第 t 次更新**之前**的策略的期望团队奖励（精确值）
    - grad_var：每轮单样本梯度的方差（对所有 agent、动作取平均）
    - final_return：全部 n_iters 次更新**之后**的策略的期望回报
    - best_return：最优联合动作的奖励（动作空间太大时为 nan）

    经验区间（见 docs/codesign-sim.md §9）：agent 少、batch 大时各估计器的学习曲线几乎重合；
    agent 多（≥ 32）、batch 小（≤ 16）时 shared 的每 agent 信噪比 ~ 1/n_agents，差距才转化为学习速度差。
    """
    if method not in METHODS:
        raise ValueError(f"method 必须是 {METHODS} 之一，收到 {method!r}")
    rng = np.random.default_rng(seed)
    n, K = task.n_agents, task.n_actions
    logits = np.zeros((n, K))
    eye = np.eye(K)
    returns, grad_var = [], []
    for _ in range(n_iters):
        pi = _softmax(logits)
        returns.append(task.expected_return(pi))
        u = rng.random((batch, n, 1))
        a = (u > np.cumsum(pi, axis=1)[None, :, :]).sum(axis=2)  # 逐 agent 按 π_i 采样
        a = np.minimum(a, K - 1)
        r = task.sample_rewards(a, rng)
        credit = estimate_credit(task, a, r, method, policies=pi, shapley_ref=shapley_ref, cf_noise=cf_noise, rng=rng)
        score = eye[a] - pi[None, :, :]  # ∇ log π_i(a_i) 对 logits：onehot - π
        per_sample = credit[:, :, None] * score  # (batch, n, K)
        grad_var.append(float(per_sample.var(axis=0).mean()))
        logits += lr * per_sample.mean(axis=0)
    return {
        "returns": returns,
        "grad_var": grad_var,
        "final_return": task.expected_return(_softmax(logits)),
        "best_return": task.best_return(),
    }
