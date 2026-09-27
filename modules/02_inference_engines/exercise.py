"""模块 02 练习：把推理引擎里几个「面试会让你当场推」的机制写成纯函数。

你要实现五个函数（四道题）：

    paged_allocate(seq_lens, block_size, num_blocks, free_blocks)   —— (a) PagedAttention 块表分配 + 内部碎片率
    plan_chunked_prefill(prompt_lens, n_decodes, token_budget)      —— (b) decode 优先的 chunked prefill 分块计划
    expected_accepted_tokens(alpha, gamma)                          —— (c1) 投机解码每次验证平均产出几个 token
    spec_speedup_at_batch(alpha, gamma, cost, batch, ctx, c)        —— (c2) 加速比如何随 batch 和上下文长度变化
    infer_chunk_tokens(stall_s, prefix_done, cost)                  —— (d) 逆问题：从观测到的卡顿反推 prefill 块大小

写完后运行：
    make module N=02
    # 或
    .venv/bin/python -m pytest modules/02_inference_engines

看参考答案跑测试：
    INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/02_inference_engines

提示：
- 五个函数都是纯函数，不依赖 codesign_sim，结果确定。
- 讲义对应小节：(a) 第 3 节，(b) 第 5 节，(c) 第 6 节，(d) 第 5 节的步长公式。
"""

from __future__ import annotations

import math  # noqa: F401  实现 (d) 时会用到
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 已实现：返回类型与成本模型
# ---------------------------------------------------------------------------


@dataclass
class PagedAllocation:
    """一次块分配的结果。"""

    block_tables: list[list[int]]  # 每个被接纳请求的块表：第 i 个逻辑块 -> 物理块号
    admitted: int  # 被接纳的请求数
    used_blocks: int  # 一共用掉的物理块数
    waste_tokens: int  # 已分配但没装 token 的槽位数（只可能出现在每个请求的最后一个块里）
    waste_ratio: float  # waste_tokens / (used_blocks * block_size)；没分配任何块时为 0.0


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


# ---------------------------------------------------------------------------
# TODO (a)：PagedAttention 块表分配
# ---------------------------------------------------------------------------


def paged_allocate(
    seq_lens: list[int],
    block_size: int,
    num_blocks: int,
    free_blocks: list[int] | None = None,
) -> PagedAllocation:
    """按到达顺序（FCFS）给每个序列分配 KV cache 物理块，返回块表和碎片统计。

    规则：
    - 每个长度为 L 的序列需要 ceil(L / block_size) 个块；L = 0 时需要 0 个块（块表为空列表）。
    - 物理块从空闲链表里按顺序取：free_blocks 为 None 时空闲链表是 [0, 1, ..., num_blocks-1]；
      否则就用 free_blocks 给定的顺序（模拟运行一段时间后空闲块已经「东一块西一块」）。
      注意：物理块号不需要连续，这正是 PagedAttention 的意义。
    - all-or-nothing：一个请求要么拿到它需要的全部块，要么一个都不拿。
    - FCFS：第一个放不下的请求会挡住它后面所有请求（不许跳过它先接纳后面的短请求）。
    - waste_tokens = 已分配块的总槽位数 - 被接纳序列的 token 总数（内部碎片）。
    - 以下情况抛 ValueError：block_size <= 0；num_blocks < 0；任一序列长度为负；
      free_blocks 里有重复的块号，或有块号不在 [0, num_blocks) 内（同一物理块不能分给两个逻辑块）。

    例：seq_lens=[5, 16, 17], block_size=16, num_blocks=8
        -> block_tables=[[0], [1], [2, 3]], admitted=3, used_blocks=4,
           waste_tokens=64-38=26, waste_ratio=26/64
    """
    raise NotImplementedError("TODO (a): 实现 PagedAttention 的块表分配与碎片统计")


# ---------------------------------------------------------------------------
# TODO (b)：chunked prefill 分块计划
# ---------------------------------------------------------------------------


def plan_chunked_prefill(
    prompt_lens: list[int],
    n_decodes: int,
    token_budget: int,
    max_steps: int = 100_000,
) -> list[list[tuple[int, int]]]:
    """规划每一步（一次模型前向）里各 prompt 分到多少 prefill token。

    调度规则（和 vLLM V1 的默认策略一致：decode 优先，剩余预算装 prefill 分块）：
    - 每一步的 token 预算是 token_budget。
    - 先给每个正在 decode 的请求 1 个 token。一开始有 n_decodes 个 decode 请求，
      并假设它们在规划期间一直没结束。
    - 剩余预算按 prompt 的到达顺序（下标顺序）切分：当前 prompt 还剩多少就切多少（不超过剩余预算）；
      当前 prompt 切完了，同一步里继续切下一个 prompt，直到预算用完或所有 prompt 都切完。
    - 某个 prompt 的最后一块在第 s 步被调度，它就从第 s+1 步起加入 decode（每步多占 1 个 token）。
    - 抛 ValueError：token_budget <= 0、n_decodes < 0、max_steps <= 0、任一 prompt 长度 < 1，
      或者某一步 decode 已经把预算吃光（剩余预算 <= 0）而还有 prompt 没做完。
    - 超过 max_steps 步还没规划完，抛 RuntimeError。

    返回：plan[s] 是第 s 步的 [(prompt 下标, 本步分到的 token 数), ...]，按调度顺序排列；
    所有 prompt 都切完时停止（prompt_lens 为空时返回 []）。

    例：prompt_lens=[300, 100], n_decodes=2, token_budget=256
        第 0 步：decode 2 个 -> 剩 254 -> [(0, 254)]
        第 1 步：decode 2 个 -> 剩 254 -> [(0, 46), (1, 100)]
        返回 [[(0, 254)], [(0, 46), (1, 100)]]
    """
    raise NotImplementedError("TODO (b): 实现 decode 优先的 chunked prefill 分块计划")


# ---------------------------------------------------------------------------
# TODO (c)：投机解码
# ---------------------------------------------------------------------------


def expected_accepted_tokens(alpha: float, gamma: int) -> float:
    """草稿模型先猜 gamma 个 token，目标模型一次前向验证；返回这一轮平均产出的 token 数。

    假设每个草稿 token 被接受的概率都是 alpha，且相互独立；遇到第一个被拒绝的 token 就停，
    目标模型会在停下的位置自己补 1 个 token（全部接受时也会多补 1 个）。
    所以产出数在 1 到 gamma+1 之间，期望是（Leviathan et al. 2023, Eq. 1）：

        E = (1 - alpha^(gamma+1)) / (1 - alpha)       alpha = 1 时 E = gamma + 1

    - gamma = 0 表示不投机，E = 1。
    - alpha 不在 [0, 1] 或 gamma < 0 时抛 ValueError。
    """
    raise NotImplementedError("TODO (c1): 实现期望接受长度公式")


def spec_speedup_at_batch(
    alpha: float, gamma: int, cost: DecodeCostModel, batch: int, ctx: int, c: float = 0.0
) -> float:
    """batch 条序列、每条上下文长 ctx 时，投机解码相对普通 decode 的吞吐加速比。

    一步里每条序列算 q 个位置（普通 decode q = 1，验证 q = gamma + 1）的耗时，用 roofline：

        t(q) = max( batch·q·(linear_flops_per_token + attn_flops_per_pair·ctx) / flops_per_s ,
                    (weight_bytes + batch·ctx·kv_bytes_per_token) / bytes_per_s )

    注意访存项里 KV 只算一次：同一条序列的 q 个 query 共用这条序列的 KV。
    草稿每步耗时记为 c·t(1)（c 是草稿与目标单步的耗时比），一轮投机 = gamma 次草稿 + 1 次验证：

        speedup = expected_accepted_tokens(alpha, gamma) · t(1) / ( t(gamma+1) + gamma·c·t(1) )

    - gamma = 0 时 speedup = 1。
    - batch < 1、ctx < 0、c < 0，或 alpha / gamma 非法时抛 ValueError。

    做完后试试：LLAMA3_70B_8XH100、alpha=0.8、gamma=4、batch=256，
    ctx=512 和 ctx=4096 的结果为什么一个 < 1、一个 > 1？（讲义第 6 节）
    """
    raise NotImplementedError("TODO (c2): 实现考虑 KV 读取的 batch 级加速比")


# ---------------------------------------------------------------------------
# TODO (d)：逆问题——从卡顿反推块大小
# ---------------------------------------------------------------------------


def infer_chunk_tokens(stall_s: float, prefix_done: int, cost: DecodeCostModel) -> int:
    """线上看到 decode 用户的一次卡顿（最长 token 间隔）是 stall_s 秒，反推这一步塞进了多大的 prefill 块。

    假设这一步是计算瓶颈，时间主要花在这个 prefill 块上（忽略同批 decode 的少量 token）。
    块长 C、这条 prompt 之前已经 prefill 了 prefix_done 个 token，则这一步的 FLOPs 是：

        linear_flops_per_token·C + attn_flops_per_pair·C·(prefix_done + C/2)

    令它等于 stall_s·flops_per_s，解关于 C 的二次方程取正根，返回四舍五入后的整数。

    - 如果 stall_s 不大于「读一遍权重的时间」weight_bytes / bytes_per_s，说明这一步处在访存瓶颈区，
      块多大步长都差不多，无法反推，抛 ValueError。
    - stall_s <= 0 或 prefix_done < 0 时抛 ValueError。

    例：LLAMA3_70B_8XH100 上，32k prompt 的最后一块（prefix_done = 24576）让卡顿达到约 0.373 秒，
    反推出块长约 8192——也就是 token 预算是 8192。
    """
    raise NotImplementedError("TODO (d): 解二次方程，从卡顿时长反推 prefill 块大小")
