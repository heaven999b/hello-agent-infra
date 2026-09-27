"""模块 03 参考答案（与 exercise.py 同签名）。

先自己写 exercise.py，卡住了再来看这里。
运行测试：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/03_deepseek_stack
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# (a) 每 token 的 KV cache 字节数：MHA / GQA / MQA / MLA
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
    attn = attn.lower()
    if attn in ("mha", "gqa", "mqa"):
        heads = 1 if attn == "mqa" else n_kv_heads
        d_k = head_dim if k_head_dim is None else k_head_dim
        d_v = head_dim if v_head_dim is None else v_head_dim
        # K 与 V 各存一份，维度可以不同（例如 K = 128 nope + 64 rope，V = 128）
        elems = n_layers * heads * (d_k + d_v)
    elif attn == "mla":
        # 只存压缩后的 latent c_kv（kv_lora_rank 维）+ 解耦 RoPE 的共享 key（qk_rope_head_dim 维）
        elems = n_layers * (kv_lora_rank + qk_rope_head_dim)
    else:
        raise ValueError(f"unknown attn type: {attn}")
    # FP4 这类不足 1 字节的格式按向上取整计物理字节（1e-9 容忍浮点误差）
    return math.ceil(elems * dtype_bytes - 1e-9)


# ---------------------------------------------------------------------------
# (b) EPLB：贪心冗余专家 + 均衡装箱（全局策略 / 层次策略）
# ---------------------------------------------------------------------------


def _replicate_counts(loads: np.ndarray, n_slots: int) -> np.ndarray:
    """每次把一份副本给「负载 / 副本数」最大的专家（与 deepseek-ai/EPLB 的 replicate_experts 相同）。"""
    counts = np.ones(len(loads), dtype=np.int64)
    for _ in range(n_slots - len(loads)):
        counts[int(np.argmax(loads / counts))] += 1
    return counts


def _balanced_packing(weights: np.ndarray, n_packs: int) -> np.ndarray:
    """按重量从大到小，依次放进「当前最轻且还有空位」的箱子。返回每个物品所在的箱子编号。"""
    per_pack = len(weights) // n_packs
    pack_weight = np.zeros(n_packs)
    pack_items = np.zeros(n_packs, dtype=np.int64)
    where = np.empty(len(weights), dtype=np.int64)
    for idx in np.argsort(-weights, kind="stable"):
        candidates = np.flatnonzero(pack_items < per_pack)
        best = candidates[int(np.argmin(pack_weight[candidates]))]
        where[idx] = best
        pack_weight[best] += weights[idx]
        pack_items[best] += 1
    return where


def _validate(loads: np.ndarray, n_gpus: int, n_redundant: int, n_groups: int, n_nodes: int) -> None:
    if loads.ndim != 1 or loads.size == 0 or np.any(loads < 0):
        raise ValueError("expert_loads must be a non-empty 1-D array of non-negative loads")
    if n_gpus <= 0 or n_redundant < 0 or n_groups <= 0 or n_nodes <= 0:
        raise ValueError("n_gpus / n_groups / n_nodes must be positive and n_redundant non-negative")
    if (loads.size + n_redundant) % n_gpus != 0:
        raise ValueError("(n_experts + n_redundant) must be divisible by n_gpus")
    if loads.size % n_groups != 0 or n_gpus % n_nodes != 0:
        raise ValueError("n_experts must be divisible by n_groups and n_gpus by n_nodes")


def eplb_gpu_loads(
    expert_loads: np.ndarray,
    n_gpus: int,
    n_redundant: int,
    *,
    n_groups: int = 1,
    n_nodes: int = 1,
) -> np.ndarray:
    loads = np.asarray(expert_loads, dtype=np.float64)
    _validate(loads, n_gpus, n_redundant, n_groups, n_nodes)
    if n_groups % n_nodes != 0:  # 组数不能被节点数整除 → 退化为全局策略（与官方 rebalance_experts 相同）
        n_groups, n_nodes = 1, 1
    n_slots = loads.size + n_redundant
    group_size = loads.size // n_groups
    gpus_per_node = n_gpus // n_nodes

    # 第 1 步：把专家组均衡地分到各节点
    group_loads = loads.reshape(n_groups, group_size).sum(axis=1)
    node_of_group = _balanced_packing(group_loads, n_nodes)

    gpu_loads = np.zeros(n_gpus)
    for node in range(n_nodes):
        groups = np.flatnonzero(node_of_group == node)
        node_experts = loads.reshape(n_groups, group_size)[groups].ravel()
        # 第 2 步：节点内复制热门专家
        counts = _replicate_counts(node_experts, n_slots // n_nodes)
        replica_loads = np.repeat(node_experts / counts, counts)
        # 第 3 步：节点内把副本装到各 GPU
        where = _balanced_packing(replica_loads, gpus_per_node)
        gpu_loads[node * gpus_per_node:(node + 1) * gpus_per_node] = np.bincount(
            where, weights=replica_loads, minlength=gpus_per_node
        )
    return gpu_loads


# ---------------------------------------------------------------------------
# (c) MLA decode 注意力 kernel 的 roofline 判断
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
    # 每读一个 KV 条目（一层、一个 token），h_q·s_q 个 query 各做一次 QK（d_k）和 PV（d_v），一次乘加算 2 FLOPs
    flops_per_entry = 2 * h_q * s_q * (d_k + d_v)
    intensity = flops_per_entry / kv_entry_bytes
    ridge = peak_tflops / hbm_tbps  # TFLOPS / (TB/s) = FLOP/B
    return {"intensity": intensity, "ridge": ridge, "bound": "compute" if intensity >= ridge else "memory"}


# ---------------------------------------------------------------------------
# (d) 从 Day 6 账本反推每张 decode GPU 的状态
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
    # Little 定律：每卡吞吐 / 每条请求的速度 = 每卡同时在跑的请求数
    concurrent = node_decode_tok_s / gpus_per_node / user_tok_s
    kv_gb = concurrent * avg_kv_len * kv_bytes_per_token / 1e9
    tokens_per_mb = concurrent * s_q / micro_batches  # 每卡每个 micro-batch 送进模型的 token
    # 所有卡的「token→路由专家」分配平摊到 n_experts + n_redundant 个副本上
    tokens_per_replica_mb = tokens_per_mb * top_k * ep / (n_experts + n_redundant)
    # 每个 micro-batch 一次 dispatch（FP8，1 B）+ 一次 combine（BF16，2 B），各带一个固定延迟 α
    comm_us = 2 * alpha_us + 3 * tokens_per_mb * n_targets * hidden / (bw_gb_s * 1e9) * 1e6
    # 双 micro-batch 重叠且通信是瓶颈时，每层 ≈ 2 × 单个 micro-batch 的通信时间
    step_ms = n_layers * micro_batches * comm_us / 1000
    return {
        "concurrent": concurrent,
        "kv_gb": kv_gb,
        "tokens_per_replica_mb": tokens_per_replica_mb,
        "step_ms": step_ms,
        "user_tok_s_bound": tokens_per_step / (step_ms / 1000),
    }
