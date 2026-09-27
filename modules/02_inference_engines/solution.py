"""模块 02 参考答案：与 exercise.py 同签名。

先自己写 exercise.py，卡住了再来看。运行 `INFRA_SOLUTION=1 make module N=02` 用这份答案跑测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# (a) PagedAttention：块表分配与内部碎片
# ---------------------------------------------------------------------------


@dataclass
class PagedAllocation:
    """一次块分配的结果。"""

    block_tables: list[list[int]]  # 每个被接纳请求的块表：第 i 个逻辑块 -> 物理块号
    admitted: int  # 被接纳的请求数
    used_blocks: int  # 一共用掉的物理块数
    waste_tokens: int  # 已分配但没装 token 的槽位数（只可能出现在每个请求的最后一个块里）
    waste_ratio: float  # waste_tokens / (used_blocks * block_size)；没分配任何块时为 0.0


def paged_allocate(
    seq_lens: list[int],
    block_size: int,
    num_blocks: int,
    free_blocks: list[int] | None = None,
) -> PagedAllocation:
    if block_size <= 0 or num_blocks < 0:
        raise ValueError("block_size 必须为正，num_blocks 不能为负")
    if any(length < 0 for length in seq_lens):
        raise ValueError("序列长度不能为负")
    free = list(range(num_blocks)) if free_blocks is None else list(free_blocks)
    if len(set(free)) != len(free) or any(not 0 <= b < num_blocks for b in free):
        raise ValueError("free_blocks 必须是 [0, num_blocks) 内互不重复的物理块号")
    tables: list[list[int]] = []
    waste = 0
    cursor = 0  # 空闲链表里下一个可用块的位置
    for length in seq_lens:
        need = -(-length // block_size)  # 向上取整
        if cursor + need > len(free):
            break  # FCFS：队头放不下就停，不跳过它去接纳后面的请求
        tables.append(free[cursor : cursor + need])
        cursor += need
        waste += need * block_size - length
    slots = cursor * block_size
    return PagedAllocation(
        block_tables=tables,
        admitted=len(tables),
        used_blocks=cursor,
        waste_tokens=waste,
        waste_ratio=waste / slots if slots else 0.0,
    )


# ---------------------------------------------------------------------------
# (b) Chunked prefill：decode 优先 + token 预算
# ---------------------------------------------------------------------------


def plan_chunked_prefill(
    prompt_lens: list[int],
    n_decodes: int,
    token_budget: int,
    max_steps: int = 100_000,
) -> list[list[tuple[int, int]]]:
    if token_budget <= 0 or n_decodes < 0 or max_steps <= 0:
        raise ValueError("要求 token_budget > 0、n_decodes >= 0、max_steps > 0")
    if any(n < 1 for n in prompt_lens):
        raise ValueError("每个 prompt 至少 1 个 token")
    remaining = list(prompt_lens)
    decoding = n_decodes
    plan: list[list[tuple[int, int]]] = []
    head = 0  # 当前正在 prefill 的 prompt 下标（FCFS）
    while head < len(remaining):
        if len(plan) >= max_steps:
            raise RuntimeError("超过 max_steps 仍未规划完")
        budget = token_budget - decoding  # 先给所有 decode 各 1 个 token
        if budget <= 0:
            raise ValueError("decode 已经吃光 token 预算，prefill 永远排不上")
        step: list[tuple[int, int]] = []
        while budget > 0 and head < len(remaining):
            chunk = min(budget, remaining[head])
            step.append((head, chunk))
            remaining[head] -= chunk
            budget -= chunk
            if remaining[head] == 0:
                head += 1
        plan.append(step)
        decoding += sum(1 for idx, _ in step if remaining[idx] == 0)  # 做完的下一步起开始 decode
    return plan


# ---------------------------------------------------------------------------
# (c) 投机解码：期望接受长度与「随 batch 和上下文变化」的加速比
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeCostModel:
    """一次前向的 roofline 成本模型（整个部署合计）。"""

    weight_bytes: float  # 每步必须从 HBM 读一遍的权重字节
    linear_flops_per_token: float  # 每个 token 的线性层 FLOPs ≈ 2 × 激活参数量
    attn_flops_per_pair: float  # 每个 (query token, 上下文 token) 对的注意力 FLOPs = 4 × 层数 × query 头数 × head_dim
    kv_bytes_per_token: float  # 每个上下文 token 的 KV 字节
    flops_per_s: float  # 有效算力 = 卡数 × 稠密峰值 × MFU
    bytes_per_s: float  # 有效带宽 = 卡数 × HBM 带宽 × MBU


# Llama-3-70B（BF16，70.6B 参数；80 层、64 个 query 头、8 个 KV 头、head_dim 128）跑在 8 × H100-SXM 上，
# 稠密 BF16 989 TFLOPS、HBM 3.35 TB/s，MFU 0.6、MBU 0.8
LLAMA3_70B_8XH100 = DecodeCostModel(
    weight_bytes=2 * 70.6e9,
    linear_flops_per_token=2 * 70.6e9,
    attn_flops_per_pair=4 * 80 * 64 * 128,
    kv_bytes_per_token=2 * 80 * 8 * 128 * 2,
    flops_per_s=8 * 989e12 * 0.6,
    bytes_per_s=8 * 3.35e12 * 0.8,
)


def expected_accepted_tokens(alpha: float, gamma: int) -> float:
    if not 0.0 <= alpha <= 1.0 or gamma < 0:
        raise ValueError("要求 0 <= alpha <= 1 且 gamma >= 0")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def _step_time(cost: DecodeCostModel, batch: int, ctx: int, q: int) -> float:
    """batch 条序列、每条上下文 ctx、每条这一步算 q 个位置：max(计算时间, 访存时间)。"""
    flops = batch * q * (cost.linear_flops_per_token + cost.attn_flops_per_pair * ctx)
    bytes_ = cost.weight_bytes + batch * ctx * cost.kv_bytes_per_token  # 同一序列的 q 个 query 共用一份 KV
    return max(flops / cost.flops_per_s, bytes_ / cost.bytes_per_s)


def spec_speedup_at_batch(
    alpha: float, gamma: int, cost: DecodeCostModel, batch: int, ctx: int, c: float = 0.0
) -> float:
    if batch < 1 or ctx < 0 or c < 0:
        raise ValueError("要求 batch >= 1、ctx >= 0、c >= 0")
    e = expected_accepted_tokens(alpha, gamma)
    t1 = _step_time(cost, batch, ctx, 1)
    return e * t1 / (_step_time(cost, batch, ctx, gamma + 1) + gamma * c * t1)


# ---------------------------------------------------------------------------
# (d) 逆问题：从观测到的卡顿反推 prefill 块大小
# ---------------------------------------------------------------------------


def infer_chunk_tokens(stall_s: float, prefix_done: int, cost: DecodeCostModel) -> int:
    if stall_s <= 0 or prefix_done < 0:
        raise ValueError("要求 stall_s > 0、prefix_done >= 0")
    if stall_s <= cost.weight_bytes / cost.bytes_per_s:
        raise ValueError("卡顿不超过读一遍权重的时间：访存瓶颈区，步长与块大小无关，无法反推")
    # (a/2)·C² + (lin + a·done)·C − stall·F = 0，取正根
    half_a = cost.attn_flops_per_pair / 2
    b = cost.linear_flops_per_token + cost.attn_flops_per_pair * prefix_done
    rhs = stall_s * cost.flops_per_s
    if half_a == 0:
        return round(rhs / b)
    return round((-b + math.sqrt(b * b + 4 * half_a * rhs)) / (2 * half_a))
