"""模块 06 练习：长尾与异步的四个基本功。

你要实现四个函数：

    sync_busy_frac_pareto(alpha, n)                                   —— (a) 同步 rollout 的槽位忙碌比例（Pareto 尾闭式解）
    truncated_is_weights(logp_train, logp_rollout, cap, level)       —— (b) 截断重要性采样（TIS）权重：token / sequence 两级
    plan_async_pools(n_gpus, gen_rate_per_gpu, train_rate_per_gpu,
                     batch_size, traj_latency_p99_s)                  —— (c) 系统设计：分离部署的池配比与最小 staleness
    lpt_schedule(lengths, n_workers)                                  —— (d) 按预测长度做 LPT（最长优先）调度

写完后运行：
    make module N=06
    # 或
    .venv/bin/python -m pytest modules/06_long_tail_async

提示：
- 四个函数都是纯函数，只依赖标准库和 numpy，不依赖 codesign_sim，结果确定。
- (a) 是 README 第 3.2 节的公式；(b) 是第 5.2、5.4 节；(c) 是第 4.5、7.2 节；(d) 是第 4.1 节。
- 看参考答案：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/06_long_tail_async
"""

from __future__ import annotations

import heapq  # noqa: F401  实现 (d) 时会用到
import math  # noqa: F401  实现 (a) 时会用到
from typing import Sequence

import numpy as np

# 与 verl 相同的数值保护：对数比率先截到 [-20, 20] 再取 exp
_LOG_RATIO_CLAMP = 20.0


# ---------------------------------------------------------------------------
# TODO (a)：同步 rollout 的槽位忙碌比例
# ---------------------------------------------------------------------------


def sync_busy_frac_pareto(alpha: float, n: int) -> float:
    """同步 rollout 中槽位的期望忙碌比例 E[T] / E[max(T_1, ..., T_n)]。

    设 n 条轨迹同时开始，时长独立同分布 T ~ Pareto(alpha, x_m)，
    即 P(T > t) = (x_m / t) ** alpha（t >= x_m）。同步屏障要等最慢的那条。

    要求：
    - 返回 E[T] / E[M_n]，其中 M_n = max_i T_i。结果与 x_m 无关（可取 x_m = 1）。
    - E[T] = alpha / (alpha - 1)（x_m = 1 时）。
    - E[M_n] = Γ(n + 1) · Γ(1 - 1/alpha) / Γ(n + 1 - 1/alpha)（x_m = 1 时）。
      n 很大时 Γ 会溢出，请用 math.lgamma 在对数空间计算。
    - alpha <= 1（均值无穷）或 n < 1 时抛 ValueError。
    - n = 1 时应当恰好返回 1.0（没有屏障就没有气泡）。

    例：alpha = 2, n = 64 时约为 0.141；alpha = 1.5, n = 256 时约为 0.028。
    """
    raise NotImplementedError("TODO (a): 用 lgamma 实现 E[T] / E[max_n T] 的闭式解")


# ---------------------------------------------------------------------------
# TODO (b)：截断重要性采样权重
# ---------------------------------------------------------------------------


def truncated_is_weights(
    logp_train: Sequence[float] | np.ndarray,
    logp_rollout: Sequence[float] | np.ndarray,
    cap: float,
    level: str = "token",
) -> np.ndarray:
    """计算一条序列上每个 token 的截断 IS 权重 w_t。

    输入是同一条序列上逐 token 的对数概率：
    - logp_train[t]   = log π_θ(y_t | y_<t)，训练端算的
    - logp_rollout[t] = log μ(y_t | y_<t)，推理引擎采样时记录的
    记对数比率 d_t = logp_train[t] - logp_rollout[t]。

    两个级别（对应 verl rollout correction 的 rollout_is = "token" / "sequence"）：
    - "token"：     w_t = min(exp(d_t), cap)                   每个 token 各自截断（有偏、方差小）
    - "sequence"：  w_t = min(exp(Σ_s d_s), cap)               整条序列的似然比，广播到每个 token

    注意：几何平均 exp(mean_s d_s) 不是任何两个分布之间的似然比，没有 IS 的无偏性；
    verl 只把它用于拒绝采样（Geo-RS），所以这里不把它当作权重级别。

    要求：
    - 返回与输入同长度的一维 np.ndarray（float）。
    - 取 exp 之前，把对应的对数量（d_t 或 Σd）clip 到 [-20, 20]，防止溢出。
    - 两个输入长度不同、或不是一维时抛 ValueError；cap <= 0 抛 ValueError；
      level 不是 "token" / "sequence" 抛 ValueError（包括 "geometric"）。
    - 空序列返回长度为 0 的数组。

    例：logp_train == logp_rollout 时两种级别都返回全 1（cap >= 1 时）。
    """
    raise NotImplementedError("TODO (b): 计算 token / sequence 两种级别的截断 IS 权重")


# ---------------------------------------------------------------------------
# TODO (c)：分离部署的池配比与 staleness 预算（系统设计题）
# ---------------------------------------------------------------------------


def plan_async_pools(
    n_gpus: int,
    gen_rate_per_gpu: float,
    train_rate_per_gpu: float,
    batch_size: int,
    traj_latency_p99_s: float,
) -> dict:
    """给分离部署的异步 RL 做容量规划（README 第 7.2 节）。

    输入：
    - n_gpus：总卡数，要分成推理池 n_gen 与训练池 n_train（两边都至少 1 张）
    - gen_rate_per_gpu：g，一张推理卡稳态每秒产出多少条轨迹
    - train_rate_per_gpu：τ，一张训练卡每秒消化多少条轨迹
    - batch_size：B，每个训练步消费的轨迹数
    - traj_latency_p99_s：一条轨迹从开始生成到完成的 p99 时长（秒）

    计算规则（测试按这个规则算期望值）：
    1. 对 n_train = 1, 2, ..., n_gpus - 1 逐个试，吞吐 X = min(n_gen · g, n_train · τ)（条/秒），
       取 X 最大的 n_train；并列时取 n_train 最小的（多留推理余量吸收长尾）。
    2. train_step_s = B / X：慢的一方决定每步多久。
    3. bottleneck = "train" 若 n_train · τ <= n_gen · g，否则 "gen"。
    4. min_staleness = ceil(traj_latency_p99_s / train_step_s)：一条 p99 轨迹生成期间训练端走的步数，
       也就是让它不因超过 η 被中止或阻塞所需的最小 η。为避免浮点误差，先减 1e-9 再取 ceil，且不小于 0。
    5. admission_window = (min_staleness + 1) · B：AReaL 式 ⌊(N_r - 1)/B⌋ <= i + η 允许的
       「已提交但未被训练消费」的轨迹上限。

    返回 dict，键为 n_gen、n_train、traj_per_s、train_step_s、bottleneck、min_staleness、admission_window。
    n_gpus < 2、batch_size < 1、吞吐 <= 0 或时延 < 0 时抛 ValueError。

    例：plan_async_pools(64, 1.0, 3.0, 512, 60.0) →
        n_gen = 48, n_train = 16（正好是 AReaL 的 3:1），traj_per_s = 48.0，train_step_s ≈ 10.67，
        min_staleness = 6，admission_window = 3584。
    """
    raise NotImplementedError("TODO (c): 枚举配比求最大吞吐，再算步时长、最小 η 与准入窗口")


# ---------------------------------------------------------------------------
# TODO (d)：LPT 调度
# ---------------------------------------------------------------------------


def lpt_schedule(lengths: Sequence[float], n_workers: int) -> tuple[float, list[int]]:
    """按预测长度做 LPT（Longest Processing Time first）调度。

    规则（测试按这个规则算期望值）：
    1. 把任务按长度从大到小排序；长度相同的按下标从小到大。
    2. 依次把每个任务派给「当前总负载最小」的 worker；负载相同的取编号最小的。
    3. worker 编号为 0 .. n_workers - 1，初始负载都是 0。

    返回 (makespan, assignment)：
    - makespan：所有 worker 总负载的最大值（float）；没有任务时为 0.0
    - assignment：长度与 lengths 相同，assignment[i] 是任务 i 被派去的 worker 编号

    n_workers < 1 时抛 ValueError；任何长度为负、NaN 或 inf 时抛 ValueError（长度预测器的输出要先清洗）。

    提示：用 heapq 维护 (负载, worker 编号) 的小顶堆，天然满足规则 2 的并列处理。
    例：lengths = [7, 5, 4, 3, 3, 2], n_workers = 2 → (12.0, [0, 1, 1, 0, 1, 0])。
    """
    raise NotImplementedError("TODO (d): 实现 LPT 调度，返回 (makespan, assignment)")
