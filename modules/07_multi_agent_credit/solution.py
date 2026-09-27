"""模块 07 参考答案（与 exercise.py 同签名）。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from math import factorial

import numpy as np


def group_advantages(rewards: np.ndarray, *, mode: str = "grpo", eps: float = 1e-6) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    g = r.shape[-1]
    mean = r.mean(axis=-1, keepdims=True)
    if mode == "mean":
        return r - mean
    if mode == "grpo":
        std = r.std(axis=-1, keepdims=True)  # ddof=0
        # std == 0 的组：r - mean 本来就全是 0，这里显式置零，避免 0/eps 之外的数值问题
        return np.where(std > 0, (r - mean) / (std + eps), 0.0)
    if mode == "rloo":
        if g < 2:
            raise ValueError("leave-one-out 需要每组至少 2 个样本")
        baseline = (r.sum(axis=-1, keepdims=True) - r) / (g - 1)
        return r - baseline
    raise ValueError(f"mode 必须是 grpo / mean / rloo，收到 {mode!r}")


def difference_rewards(
    reward_fn: Callable[[tuple[int, ...]], float],
    joint_action: Sequence[int],
    default_action: int = 0,
) -> np.ndarray:
    base = tuple(int(a) for a in joint_action)
    r_joint = float(reward_fn(base))
    out = np.empty(len(base), dtype=np.float64)
    for i in range(len(base)):
        # 只替换第 i 个 agent 的动作，其余 agent 的动作保持不变
        counterfactual = base[:i] + (default_action,) + base[i + 1 :]
        out[i] = r_joint - float(reward_fn(counterfactual))
    return out


def exact_shapley(value_fn: Callable[[frozenset[int]], float], n_agents: int) -> np.ndarray:
    if n_agents > 16:
        raise ValueError("精确 Shapley 需要 2^n 次联盟求值，n > 16 请改用蒙特卡洛排列采样")
    players = range(n_agents)
    # 1) 每个联盟只求值一次
    values: dict[frozenset[int], float] = {}
    for k in range(n_agents + 1):
        for coalition in combinations(players, k):
            key = frozenset(coalition)
            values[key] = float(value_fn(key))
    # 2) 按公式加权组合边际贡献
    n_fact = factorial(n_agents)
    phi = np.zeros(n_agents, dtype=np.float64)
    for s, v_s in values.items():
        if len(s) == n_agents:
            continue
        weight = factorial(len(s)) * factorial(n_agents - len(s) - 1) / n_fact
        for i in players:
            if i not in s:
                phi[i] += weight * (values[s | {i}] - v_s)
    return phi


def branch_token_cost(
    history_ext: int,
    history_gen: int,
    append_ext: int,
    branch_gen: int,
    n_branches: int,
    strategy: str,
) -> dict[str, int]:
    if min(history_ext, history_gen, append_ext, branch_gen, n_branches) < 0:
        raise ValueError("token 数和分支数都不能为负")
    per_branch = {
        "rerun": (history_ext + append_ext, history_gen + branch_gen),
        "restore": (history_ext + history_gen + append_ext, branch_gen),
        "kv_fork": (append_ext, branch_gen),
    }
    if strategy not in per_branch:
        raise ValueError(f"strategy 必须是 {sorted(per_branch)} 之一，收到 {strategy!r}")
    prefill, decode = per_branch[strategy]
    return {"prefill": prefill * n_branches, "decode": decode * n_branches}
