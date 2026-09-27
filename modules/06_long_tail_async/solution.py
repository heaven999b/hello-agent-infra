"""模块 06 参考答案：长尾与异步的四个基本功。

与 exercise.py 同签名。先自己写，卡住了再看。
"""

from __future__ import annotations

import heapq
import math
from typing import Sequence

import numpy as np

# 与 verl 相同的数值保护：对数比率先截到 [-20, 20] 再取 exp
_LOG_RATIO_CLAMP = 20.0


# ---------------------------------------------------------------------------
# (a) 同步 rollout 的槽位忙碌比例（Pareto 尾闭式解）
# ---------------------------------------------------------------------------


def sync_busy_frac_pareto(alpha: float, n: int) -> float:
    """E[T] / E[max(T_1..T_n)]，T ~ Pareto(alpha, x_m)，结果与 x_m 无关。"""
    if alpha <= 1.0:
        raise ValueError("alpha 必须 > 1，否则均值无穷")
    if n < 1:
        raise ValueError("n 必须 >= 1")
    mean = alpha / (alpha - 1.0)
    # E[M_n] / x_m = Γ(n+1) Γ(1-1/α) / Γ(n+1-1/α)，用 lgamma 避免大 n 溢出
    log_emax = math.lgamma(n + 1) + math.lgamma(1 - 1 / alpha) - math.lgamma(n + 1 - 1 / alpha)
    return mean / math.exp(log_emax)


# ---------------------------------------------------------------------------
# (b) 截断重要性采样权重
# ---------------------------------------------------------------------------


def truncated_is_weights(
    logp_train: Sequence[float] | np.ndarray,
    logp_rollout: Sequence[float] | np.ndarray,
    cap: float,
    level: str = "token",
) -> np.ndarray:
    """一条序列的截断 IS 权重，形状与输入相同。"""
    lp = np.asarray(logp_train, dtype=float)
    lr = np.asarray(logp_rollout, dtype=float)
    if lp.shape != lr.shape or lp.ndim != 1:
        raise ValueError("logp_train 与 logp_rollout 必须是同长度的一维数组")
    if cap <= 0:
        raise ValueError("cap 必须 > 0")
    if level not in ("token", "sequence"):
        raise ValueError(f"未知 level: {level}（几何平均不是 IS 权重，只用于拒绝采样）")
    if lp.size == 0:
        return np.zeros(0)

    log_ratio = lp - lr
    if level == "token":
        log_w = log_ratio  # 每个 token 自己一个比率
    else:
        log_w = np.full_like(lp, log_ratio.sum())  # 整条序列的似然比，广播到每个 token

    w = np.exp(np.clip(log_w, -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP))
    return np.minimum(w, cap)


# ---------------------------------------------------------------------------
# (c) 分离部署的池配比与 staleness 预算
# ---------------------------------------------------------------------------


def plan_async_pools(
    n_gpus: int,
    gen_rate_per_gpu: float,
    train_rate_per_gpu: float,
    batch_size: int,
    traj_latency_p99_s: float,
) -> dict:
    """推理 / 训练卡数配比、训练步时长、最小 η 与 AReaL 准入窗口。"""
    if n_gpus < 2 or batch_size < 1:
        raise ValueError("n_gpus 必须 >= 2，batch_size 必须 >= 1")
    if gen_rate_per_gpu <= 0 or train_rate_per_gpu <= 0 or traj_latency_p99_s < 0:
        raise ValueError("吞吐必须 > 0，时延必须 >= 0")
    best_train, best_x = 1, -1.0
    for n_train in range(1, n_gpus):  # 训练卡从少到多试，吞吐严格更大才替换（并列取训练卡少的）
        x = min((n_gpus - n_train) * gen_rate_per_gpu, n_train * train_rate_per_gpu)
        if x > best_x:
            best_train, best_x = n_train, x
    n_gen = n_gpus - best_train
    step_s = batch_size / best_x  # 慢的一方决定每步多久
    min_staleness = max(0, math.ceil(traj_latency_p99_s / step_s - 1e-9))
    return {
        "n_gen": n_gen,
        "n_train": best_train,
        "traj_per_s": best_x,
        "train_step_s": step_s,
        "bottleneck": "train" if best_train * train_rate_per_gpu <= n_gen * gen_rate_per_gpu else "gen",
        "min_staleness": min_staleness,
        "admission_window": (min_staleness + 1) * batch_size,  # AReaL 式 (3)：已提交未消费的上限
    }


# ---------------------------------------------------------------------------
# (d) LPT 调度
# ---------------------------------------------------------------------------


def lpt_schedule(lengths: Sequence[float], n_workers: int) -> tuple[float, list[int]]:
    """最长的先派，派给当前负载最小的 worker（负载相同取编号小的）。"""
    if n_workers < 1:
        raise ValueError("n_workers 必须 >= 1")
    if any(not math.isfinite(x) or x < 0 for x in lengths):
        raise ValueError("长度必须是非负有限数（先清洗长度预测器的输出）")
    # 长度降序；长度相同按下标升序，保证结果确定
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    heap: list[tuple[float, int]] = [(0.0, w) for w in range(n_workers)]
    assignment = [-1] * len(lengths)
    for i in order:
        load, w = heapq.heappop(heap)  # (负载, 编号) 最小者
        assignment[i] = w
        heapq.heappush(heap, (load + float(lengths[i]), w))
    makespan = max(load for load, _ in heap)
    return makespan, assignment
