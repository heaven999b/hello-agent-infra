"""模块 03 练习：两个「照公式写」的基本功，加两个「自己推」的判断题。

你要实现四个函数：

    kv_bytes_per_token(attn, n_layers, ...)         —— (a) MHA / GQA / MQA / MLA 每 token 的 KV cache 字节数（K/V 可不同维）
    eplb_gpu_loads(expert_loads, n_gpus, ...)       —— (b) EPLB：贪心冗余专家 + 均衡装箱，支持全局策略和层次策略
    mla_decode_roofline(h_q, s_q, ...)              —— (c) 推导 MLA decode 注意力 kernel 的算术强度，判断受限于算力还是带宽
    day6_per_gpu(...)                               —— (d) 从 Day 6 公开账本反推每张 decode GPU 的并发、KV、专家 batch 和速度上限

写完后运行：
    make module N=03
    # 或
    .venv/bin/python -m pytest modules/03_deepseek_stack

看参考答案能否通过：
    INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/03_deepseek_stack

提示：
- 四个函数都是纯函数，只用标准库 + numpy，结果确定。
- (c) (d) 的 docstring 只给了物理含义，公式要你自己推；README 第 2.1 节和第 4.5 节有推导思路。
"""

from __future__ import annotations

import math  # noqa: F401  实现 (a) 时会用到

import numpy as np  # noqa: F401  实现 (b) 时会用到

# ---------------------------------------------------------------------------
# TODO (a)：每 token 的 KV cache 字节数
# ---------------------------------------------------------------------------


def kv_bytes_per_token(
    attn: str,
    n_layers: int,
    *,
    n_kv_heads: int = 0,
    head_dim: int = 0,
    k_head_dim: int | None = None,
    v_head_dim: int | None = None,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    dtype_bytes: float = 2,
) -> int:
    """返回一个 token 在所有层上占用的 KV cache 字节数（int）。

    规则（DeepSeek-V2 论文表 1，arXiv:2405.04434）：
      - "mha" / "gqa"：每层存 K 和 V，每个 KV 头 K 占 k_head_dim、V 占 v_head_dim 个元素
                        → n_layers × n_kv_heads × (k_head_dim + v_head_dim) × dtype_bytes
                        k_head_dim / v_head_dim 为 None 时都取 head_dim（K、V 同维的常见情形）
      - "mqa"：同上，但只有 1 个 KV 头（忽略 n_kv_heads）
      - "mla"：每层只存压缩 latent c_kv（kv_lora_rank 维）和解耦 RoPE 的共享 key（qk_rope_head_dim 维）
                        → n_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
      - 其他字符串：raise ValueError

    dtype_bytes 可以是 2（BF16）、1（FP8）或 0.5（FP4）。不足 1 字节时按物理存储向上取整：
    1 个 FP4 元素也要占 1 个字节。

    对照数字（BF16，ISCA'25 论文表 1，arXiv:2505.09343）：
      DeepSeek-V3（MLA，61 层，512 + 64）       → 70_272
      Qwen-2.5 72B（GQA，80 层，8 头 × 128）    → 327_680
      LLaMA-3.1 405B（GQA，126 层，8 头 × 128） → 516_096
    V3 的反事实 MHA 版本：61 层 × 128 头，K = 192（128 nope + 64 rope）、V = 128 → 4_997_120
    """
    raise NotImplementedError("TODO (a): 按 attn 类型实现 KV 字节公式，K/V 维度分开算")


# ---------------------------------------------------------------------------
# TODO (b)：EPLB 负载均衡（冗余专家 + 装箱；全局 / 层次两种策略）
# ---------------------------------------------------------------------------


def eplb_gpu_loads(
    expert_loads: np.ndarray,
    n_gpus: int,
    n_redundant: int,
    *,
    n_groups: int = 1,
    n_nodes: int = 1,
) -> np.ndarray:
    """模拟 deepseek-ai/EPLB 的 rebalance_experts，返回长度为 n_gpus 的 float 数组：每张 GPU 分到的负载。

    输入：
      expert_loads：长度 n_experts，每个逻辑专家的负载（例如一段时间内路由到它的 token 数）
      n_gpus：EP 大小；n_redundant：额外的冗余副本数
      n_groups / n_nodes：专家分组数（V3 的 n_group = 8）和节点数；GPU 编号按节点连续排列

    先做输入校验，下列情况 raise ValueError：
      expert_loads 不是非空一维数组或含负数；n_gpus / n_groups / n_nodes ≤ 0；n_redundant < 0；
      (n_experts + n_redundant) 不能被 n_gpus 整除；n_experts 不能被 n_groups 整除；n_gpus 不能被 n_nodes 整除。

    两个基本操作（与官方 eplb.py 相同）：
      replicate(loads, n_slots)：每个专家先有 1 份；重复加副本直到总数为 n_slots，
          每次加给「负载 / 当前副本数」最大的专家（并列取下标最小，np.argmax 的默认行为）。
      pack(weights, n_packs)：每个箱子恰好装 len(weights) / n_packs 件；按重量从大到小（稳定排序）
          依次放进「当前总重最小、且还有空位」的箱子（并列取下标最小）。
      每份副本的负载 = 该专家负载 / 该专家副本数（token 在副本间均分）。

    策略选择：
      - n_groups % n_nodes == 0 → 层次策略：
          1) 把每组的总负载 pack 到 n_nodes 个节点上（每节点 n_groups / n_nodes 组）；
          2) 每个节点内，对本节点的专家 replicate 到 (n_experts + n_redundant) / n_nodes 个槽位；
          3) 每个节点内，把本节点的副本 pack 到该节点的 n_gpus / n_nodes 张 GPU 上。
      - 否则 → 全局策略：等价于 n_groups = n_nodes = 1 的层次策略。

    返回的顺序无所谓，测试会先排序再比较。
    """
    raise NotImplementedError("TODO (b): 校验输入 → 分组到节点 → 节点内复制 → 节点内装箱")


# ---------------------------------------------------------------------------
# TODO (c)：MLA decode 注意力 kernel 的 roofline
# ---------------------------------------------------------------------------


def mla_decode_roofline(
    h_q: int,
    s_q: int,
    *,
    d_k: int = 576,
    d_v: int = 512,
    kv_entry_bytes: float = 1152,
    peak_tflops: float = 865.0,
    hbm_tbps: float = 3.35,
) -> dict:
    """判断 MLA 在 MQA 模式（矩阵吸收后）下的 decode 注意力 kernel 受限于算力还是显存带宽。

    场景：一条请求本步有 s_q 个 query token（开 MTP 时 s_q = 2），每个 token 有 h_q 个 query 头（TP 会把头切小）。
    所有头共享同一份 KV：每个 KV 条目（一层、一个历史 token）只从 HBM 读一次，占 kv_entry_bytes 字节
    （BF16 时 576 × 2 = 1152；V3.2 的 FP8 格式是 656）。K 用满 d_k 维，V 是其中的前 d_v 维，不需要额外读取。
    一次乘加算 2 FLOPs；query、输出的读写相对 KV 可忽略。

    返回 {"intensity": 每读 1 字节 KV 做的 FLOPs, "ridge": peak_tflops / hbm_tbps（FLOP/B），
          "bound": "compute"（intensity ≥ ridge）或 "memory"}。
    默认的 865 TFLOPS 是 FlashMLA 深度解读里 H800 降频后的实际峰值，3.35 TB/s 是 H800 的 HBM 带宽。

    思考：结果里有没有 batch 大小？为什么？
    """
    raise NotImplementedError("TODO (c): 数清每个 KV 条目对应的 FLOPs 和字节，再与 ridge 比较")


# ---------------------------------------------------------------------------
# TODO (d)：从 Day 6 账本反推每张 decode GPU
# ---------------------------------------------------------------------------


def day6_per_gpu(
    node_decode_tok_s: float = 14.8e3,
    user_tok_s: float = 21.0,
    avg_kv_len: int = 4989,
    kv_bytes_per_token: int = 70_272,
    *,
    gpus_per_node: int = 8,
    ep: int = 144,
    n_experts: int = 256,
    n_redundant: int = 32,
    top_k: int = 8,
    s_q: int = 2,
    tokens_per_step: float = 1.85,
    micro_batches: int = 2,
    n_layers: int = 61,
    hidden: int = 7168,
    n_targets: int = 8,
    bw_gb_s: float = 50.0,
    alpha_us: float = 0.0,
) -> dict:
    """只用 Day 6 公开的数字，推出一张 decode GPU 上的状态。返回 dict，键如下：

      "concurrent"：每张 GPU 同时在生成的请求数。已知每节点输出吞吐 node_decode_tok_s、每节点 gpus_per_node 张卡、
                    每条请求平均输出速度 user_tok_s。（提示：Little 定律）
      "kv_gb"：这些请求在这张 GPU 上占用的 KV cache，单位 GB（1e9 字节）。每条请求的平均 KV 长度为 avg_kv_len。
      "tokens_per_replica_mb"：每个路由专家副本在每个 micro-batch 里分到的 token 数（按平均摊）。
                    每条请求每步送进 s_q 个 token，一步拆成 micro_batches 个 micro-batch；每个 token 选 top_k 个路由专家；
                    全系统共有 ep 张卡、n_experts + n_redundant 个专家副本。
      "step_ms"：假设 EP 通信是瓶颈、两个 micro-batch 完全重叠时，一个 decode step 的耗时（毫秒）。
                    每个 micro-batch 每层做一次 dispatch（FP8，1 字节/元素）和一次 combine（BF16，2 字节/元素），
                    每个 token 发往 n_targets 个远端专家、每次传 hidden 维；带宽 bw_gb_s（1 GB/s = 1e9 B/s），
                    dispatch 和 combine 各另加一个固定延迟 alpha_us。每层耗时 = micro_batches × 单个 micro-batch 的通信时间。
      "user_tok_s_bound"：由 step_ms 决定的每条请求输出速度上限。每步平均产出 tokens_per_step 个 token（MTP 接受率约 85%）。

    默认参数下的对照：concurrent ≈ 88.1，kv_gb ≈ 30.9，tokens_per_replica_mb ≈ 352，step_ms ≈ 37.0，user_tok_s_bound ≈ 50.0
    """
    raise NotImplementedError("TODO (d): 依次推并发 → KV → 每副本 token → 通信时间 → 速度上限")
