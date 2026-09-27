"""模块 07 练习：信用分配的四个基本零件（三个算法侧、一个系统侧）。

全部是纯函数，只依赖 numpy，不需要 codesign_sim。
运行测试：make module N=07
看参考答案能否通过：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/07_multi_agent_credit
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def group_advantages(rewards: np.ndarray, *, mode: str = "grpo", eps: float = 1e-6) -> np.ndarray:
    """同一个 prompt 的 G 个样本的组内优势（最后一维是组）。

    要求：
    - rewards 的形状是 (..., G)，所有计算都沿最后一维进行，前面可以有任意多维。
    - mode="grpo"（GRPO 原版）：A = (r - mean) / (std + eps)，mean 与总体标准差 std（ddof=0）都含样本自己。
      某一组所有奖励都相同（std == 0）时，这一组的优势必须全为 0，不能出现 nan / inf。
    - mode="mean"（Dr. GRPO 的做法）：A = r - mean。注意 mean 含样本自己，期望梯度会被缩小 (G-1)/G。
    - mode="rloo"（留一法）：A = r - mean_{j != i} r_j，基线不含样本自己，严格无偏。
      G < 2 时抛 ValueError；用 O(G) 的向量化写法（总和减去自己），不要写双重循环。
    - 其他 mode 抛 ValueError。
    - 返回 float64 数组，形状与输入相同；不要原地修改输入。
    提示：rloo 的结果恰好等于 G/(G-1) × mode="mean" 的结果，测试会检查这个恒等式。
    """
    raise NotImplementedError("TODO: 实现三种组内优势")


def difference_rewards(
    reward_fn: Callable[[tuple[int, ...]], float],
    joint_action: Sequence[int],
    default_action: int = 0,
) -> np.ndarray:
    """difference reward（Wolpert & Tumer 的 Wonderful Life Utility 一族）。

    D_i = R(a) - R(a_{-i}, c)：把第 i 个 agent 的动作换成默认动作 c，看团队奖励掉了多少。

    要求：
    - reward_fn 接收一个 tuple[int, ...]（联合动作），返回团队奖励 float。
    - 返回长度为 n_agents 的 float64 数组。
    - 总共只能调用 reward_fn n_agents + 1 次（1 次原始联合动作 + 每个 agent 1 次反事实）。
      这正是 difference reward 在训练系统里的代价：每个 agent 一次额外评估。
    """
    raise NotImplementedError("TODO: 实现 difference reward")


def exact_shapley(value_fn: Callable[[frozenset[int]], float], n_agents: int) -> np.ndarray:
    """小规模精确 Shapley 值。

    phi_i = sum_{S ⊆ N \\ {i}} |S|! (n - |S| - 1)! / n! * [v(S ∪ {i}) - v(S)]

    要求：
    - value_fn 接收一个 frozenset[int]（联盟里的 agent 编号，0..n-1），返回该联盟的价值。
    - 返回长度为 n_agents 的 float64 数组。
    - 每个联盟最多求值一次（先把 2^n 个联盟的价值算好缓存起来，再组合），
      因为在真实系统里一次 v(S) 可能就是一次完整的多 agent rollout。
    - n_agents > 16 时抛 ValueError（2^16 = 65536 次求值已经是上限）。
    """
    raise NotImplementedError("TODO: 实现精确 Shapley 值")


def branch_token_cost(
    history_ext: int,
    history_gen: int,
    append_ext: int,
    branch_gen: int,
    n_branches: int,
    strategy: str,
) -> dict[str, int]:
    """系统侧：在一个决策点做 n_branches 条反事实分支，要多算多少 token。

    一条轨迹在决策点之前有 history_ext 个外部 token（prompt、工具结果）和 history_gen 个模型生成的 token；
    每条分支在决策点之后追加 append_ext 个外部 token（新的工具结果、其他 agent 的消息），自己生成 branch_gen 个 token。

    三种做法（每条分支的代价，再乘以 n_branches）：
    - "rerun"：从头重跑整条 episode。决策点之前的外部 token 要重新 prefill，模型要把历史重新 decode 一遍。
      prefill = history_ext + append_ext；decode = history_gen + branch_gen。
    - "restore"（C3 式 checkpoint 恢复）：把决策点之前的文本历史原样恢复，当作 prompt 喂回去。
      不用重新 decode 历史，但整段历史（含模型生成的部分）要重新 prefill。
      prefill = history_ext + history_gen + append_ext；decode = branch_gen。
    - "kv_fork"：从决策点的 KV 直接 fork（前缀命中）。历史既不 prefill 也不 decode。
      prefill = append_ext；decode = branch_gen。
    返回 {"prefill": 总 prefill token 数, "decode": 总 decode token 数}。
    - 任何 token 数为负、n_branches < 0 或 strategy 不认识时抛 ValueError。
    """
    raise NotImplementedError("TODO: 实现三种分支方式的 token 账")
