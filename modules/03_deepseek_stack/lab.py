"""模块 03 实验：用四个小实验核对 DeepSeek 的系统账。

A. MLA 到底省了多少 KV：每 token KV 字节，以及 8 / 16 / 144 张 H800 按不同部署方式能放下多少并发会话
B. EPLB 冗余专家值不值：GPU 最大/平均负载比，拆成「结构性下限」和「采样噪声」两部分
C. 跨节点 EP 的通信天花板：ISCA'25 的纯带宽算式 + 用 DeepEP 实测拟合的固定延迟 α
D. 核对 Day 6 账本，并反推每张 decode GPU 的状态

运行：.venv/bin/python modules/03_deepseek_stack/lab.py   （纯 CPU，约 1 秒）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS, ep_comm_time_s, kv_bytes_per_token, moe_params

GB = 1e9
H800 = GPUS["H800-SXM"]

# ============================================================================
# A. KV cache 与显存：MHA / GQA / MLA，以及权重怎么摆
# ============================================================================

# DeepSeek-V3 结构参数（官方 config.json）：61 层，前 3 层 dense，其余 58 层 MoE；hidden 7168；
# 每层 256 个路由专家，专家中间维 2048；总参数 671B（不含 14B MTP 模块）。参数拆分直接用 codesign_sim 的 moe_params
V3_MODEL = MODELS["deepseek-v3"]
V3_SPLIT = moe_params(V3_MODEL)
V3_MOE_LAYERS, V3_HIDDEN, V3_EXPERTS = V3_SPLIT["moe_layers"], V3_MODEL.d_model, V3_MODEL.n_routed_experts
PARAMS_PER_EXPERT = V3_SPLIT["expert_params"]  # 3 × 7168 × 2048：gate / up / down 三个矩阵
V3_NONROUTED_PARAMS = V3_SPLIT["always_active_params"]  # 注意力、共享专家、dense 层、embedding 与输出头 ≈ 17.1B

CONTEXTS = (4_989, 32_768, 131_072)  # 4,989 = Day 6 公布的平均 KV 长度；32K / 128K 代表 agent 长会话
HBM_UTIL = 0.9  # 类似 vLLM 的 gpu_memory_utilization：留 10% 给激活和碎片


@dataclass(frozen=True)
class KVModel:
    name: str
    attn: str
    kv_bytes: int  # 每 token KV 字节（BF16）
    weight_bytes: float  # 总权重字节
    moe: bool  # True：按 DeepSeek 方式部署（注意力 DP、路由专家 EP）


def gqa_kv_bytes(n_layers: int, n_kv_heads: int, d_k: int, d_v: int, dtype_bytes: int = 2) -> int:
    """K 与 V 维度可以不同：V3 的 K = 128 nope + 64 rope = 192，V = 128。"""
    return n_layers * n_kv_heads * (d_k + d_v) * dtype_bytes


def build_models() -> list[KVModel]:
    dense = [MODELS[k] for k in ("llama3.1-8b", "qwen3-32b", "llama3.1-70b")]
    out = [KVModel(m.name, m.attn, kv_bytes_per_token(m), m.total_params_b * 1e9 * 2, False) for m in dense]
    # V3 官方以 FP8 发布权重（1 字节/参数）；两个反事实版本只换 KV 形状，权重按 V3 不变（假设）
    v3 = MODELS["deepseek-v3"]
    out += [
        KVModel("v3-if-MHA", "mha", gqa_kv_bytes(61, 128, 192, 128), 671e9, True),
        KVModel("v3-if-GQA8", "gqa", gqa_kv_bytes(61, 8, 192, 128), 671e9, True),
        KVModel(v3.name, v3.attn, kv_bytes_per_token(v3), 671e9, True),
    ]
    return out


def weight_per_gpu(m: KVModel, n_gpus: int, deploy: str, n_redundant: int = 0) -> float:
    """每张卡上的权重字节。
    tp / pooled：权重平均切到每张卡（dense 模型 TP8 就是这样；对 V3 只是理想化上界）。
    dp+ep：注意力、共享专家等非路由参数每卡一份；路由专家按 (256 + 冗余) / n_gpus 个槽位分片。"""
    if deploy in ("tp", "pooled"):
        return m.weight_bytes / n_gpus
    slots_per_gpu = (V3_EXPERTS + n_redundant) / n_gpus
    return V3_NONROUTED_PARAMS + slots_per_gpu * V3_MOE_LAYERS * PARAMS_PER_EXPERT


def sessions(m: KVModel, n_gpus: int, deploy: str, context: int, n_redundant: int = 0) -> int:
    """能同时放下的会话数；显存不够放权重时返回 -1。
    dp+ep 下一条请求的 KV 整条落在一张卡上，所以先按卡取整再乘卡数。"""
    kv_per_gpu = H800.hbm_gb * GB * HBM_UTIL - weight_per_gpu(m, n_gpus, deploy, n_redundant)
    if kv_per_gpu <= 0:
        return -1
    per_session = m.kv_bytes * context
    if deploy == "dp+ep":
        return n_gpus * int(kv_per_gpu // per_session)
    return int(n_gpus * kv_per_gpu // per_session)


def part_a() -> None:
    models = {m.name: m for m in build_models()}
    base = models["deepseek-v3"].kv_bytes
    print("A1. KV cache per token (BF16)")
    print(f"{'model':<13}{'attn':<6}{'KV B/token':>12}{'KV @32K':>11}{'vs V3-MLA':>11}")
    for m in models.values():
        print(f"{m.name:<13}{m.attn:<6}{m.kv_bytes:>12,}{m.kv_bytes * 32_768 / GB:>9.1f}GB{m.kv_bytes / base:>10.1f}x")

    print(f"\nA2. Concurrent sessions ({H800.name}, {H800.hbm_gb:.0f} GB/GPU, {HBM_UTIL:.0%} usable, upper bound)")
    header = "".join(f"{f'@{c:,}':>10}" for c in CONTEXTS)
    print(f"{'model':<12}{'GPUs':>5}{'deploy':>8}{'W/GPU':>7}{'KV/GPU':>8}{header}")
    rows = [("llama3.1-70b", 8, "tp", 0), ("qwen3-32b", 8, "tp", 0),
            ("deepseek-v3", 8, "pooled", 0), ("deepseek-v3", 8, "dp+ep", 0),
            ("deepseek-v3", 16, "pooled", 0), ("deepseek-v3", 16, "dp+ep", 0),
            ("v3-if-GQA8", 16, "dp+ep", 0), ("v3-if-MHA", 16, "dp+ep", 0),
            ("deepseek-v3", 144, "dp+ep", 32)]
    for name, n_gpus, deploy, red in rows:
        m = models[name]
        w = weight_per_gpu(m, n_gpus, deploy, red)
        kv = H800.hbm_gb * GB * HBM_UTIL - w
        cells = [sessions(m, n_gpus, deploy, c, red) for c in CONTEXTS]
        text = "".join(f"{'OOM':>10}" if n < 0 else f"{n:>10,}" for n in cells)
        kv_text = f"{kv / GB:>7.0f}G" if kv > 0 else f"{'-':>8}"
        print(f"{name:<12}{n_gpus:>5}{deploy:>8}{w / GB:>6.0f}G{kv_text}{text}")


# ============================================================================
# B. EP 负载均衡：结构性下限 + 采样噪声；全局策略 vs 层次策略
# ============================================================================

TOP_K, N_GROUPS = 8, 8  # V3 config：num_experts_per_tok = 8，n_group = 8
# (名字, EP 大小, 每卡 token 数, 冗余专家数, 节点数)
# EP32 / EP144 与各 32 个冗余来自 Day 6；每卡 token 数与 EP128 来自 deepseek-ai/profile-data
# （预填充每卡 16K token；解码每卡 128 条请求）。EP128 的冗余数未公开，这里取 0 作对照。
STAGES = [("prefill EP32", 32, 16_384, 32, 4), ("decode EP144", 144, 128, 32, 18), ("decode EP128", 128, 128, 0, 16)]
SKEWS = (0.0, 0.5, 1.0)  # 专家热度 p ∝ exp(σ·z)，z~N(0,1)；σ 越大越偏斜（合成假设，非官方数据）
N_STEPS = 8  # 规划窗口与评估窗口的步数


def sample_loads(rng: np.random.Generator, popularity: np.ndarray, n_tokens: int) -> np.ndarray:
    """用多项分布近似 top-8 路由：共 n_tokens×8 次「token→专家」分配。
    忽略了同一 token 不重复选同一专家，也忽略了组受限路由（每个 token 只在 4 个组里选）。"""
    return rng.multinomial(n_tokens * TOP_K, popularity).astype(np.float64)


def round_robin_gpu_loads(loads: np.ndarray, n_gpus: int) -> np.ndarray:
    """朴素放置：专家 e 放在 GPU e % n_gpus 上，不做任何复制。"""
    return np.bincount(np.arange(len(loads)) % n_gpus, weights=loads, minlength=n_gpus)


def replicate(loads: np.ndarray, n_slots: int) -> np.ndarray:
    """EPLB 的冗余规则：每次给「负载/副本数」最大的专家加一份副本。返回每个专家的副本数。"""
    counts = np.ones(len(loads), dtype=np.int64)
    for _ in range(n_slots - len(loads)):
        counts[int(np.argmax(loads / counts))] += 1
    return counts


def pack(weights: np.ndarray, n_packs: int) -> np.ndarray:
    """EPLB 的装箱规则：从重到轻，放进当前最轻且还有空位的箱子。返回每件物品所在的箱子。"""
    per_pack = len(weights) // n_packs
    pack_weight, pack_items = np.zeros(n_packs), np.zeros(n_packs, dtype=np.int64)
    where = np.empty(len(weights), dtype=np.int64)
    for i in np.argsort(-weights, kind="stable"):
        open_packs = np.flatnonzero(pack_items < per_pack)
        p = open_packs[int(np.argmin(pack_weight[open_packs]))]
        where[i], pack_weight[p], pack_items[p] = p, pack_weight[p] + weights[i], pack_items[p] + 1
    return where


def plan_eplb(loads: np.ndarray, n_gpus: int, n_redundant: int,
              n_groups: int = 1, n_nodes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """官方 eplb.py 的放置方案：(每个物理副本对应的逻辑专家, 每个物理副本所在 GPU)。
    n_groups % n_nodes == 0 时走层次策略（组→节点，节点内复制、装箱），否则退化为全局策略。"""
    if n_groups % n_nodes != 0:
        n_groups, n_nodes = 1, 1
    group_size, gpus_per_node = len(loads) // n_groups, n_gpus // n_nodes
    slots_per_node = (len(loads) + n_redundant) // n_nodes
    experts_by_group = np.arange(len(loads)).reshape(n_groups, group_size)
    node_of_group = pack(loads[experts_by_group].sum(axis=1), n_nodes)
    phy2log, where = [], []
    for node in range(n_nodes):
        experts = experts_by_group[node_of_group == node].ravel()
        counts = replicate(loads[experts], slots_per_node)
        replicas = np.repeat(experts, counts)
        phy2log.append(replicas)
        where.append(pack(loads[replicas] / np.repeat(counts, counts), gpus_per_node) + node * gpus_per_node)
    return np.concatenate(phy2log), np.concatenate(where)


def apply_plan(actual_loads: np.ndarray, plan: tuple[np.ndarray, np.ndarray], n_gpus: int) -> np.ndarray:
    """把「实际负载」按方案分摊：同一专家的 token 在它的副本之间均分。"""
    phy2log, where = plan
    counts = np.bincount(phy2log, minlength=len(actual_loads))
    return np.bincount(where, weights=actual_loads[phy2log] / counts[phy2log], minlength=n_gpus)


def imbalance(gpu_loads: np.ndarray) -> float:
    return float(gpu_loads.max() / gpu_loads.mean())


def pad(loads: np.ndarray, n_redundant: int) -> np.ndarray:
    """「只装箱不复制」时，把冗余槽位当成负载为 0 的空槽。"""
    return np.concatenate([loads, np.zeros(n_redundant)])


def eval_stage(rng: np.random.Generator, popularity: np.ndarray, ep: int, n_tokens: int,
               n_redundant: int, n_nodes: int) -> list[float | None]:
    """用过去 N_STEPS 步的累计负载做静态规划，在接下来 N_STEPS 步上评估，返回各列的平均值。"""
    history = sum(sample_loads(rng, popularity, n_tokens) for _ in range(N_STEPS))
    greedy = plan_eplb(pad(history, n_redundant), ep, 0)
    global_ = plan_eplb(history, ep, n_redundant)
    hier = plan_eplb(history, ep, n_redundant, N_GROUPS, n_nodes) if N_GROUPS % n_nodes == 0 else None
    rows = []
    for _ in range(N_STEPS):
        now = sample_loads(rng, popularity, n_tokens)
        rows.append([
            imbalance(round_robin_gpu_loads(now, ep)),
            imbalance(apply_plan(pad(now, n_redundant), greedy, ep)),
            imbalance(apply_plan(now, global_, ep)),
            imbalance(apply_plan(now, hier, ep)) if hier else np.nan,
            imbalance(apply_plan(now, plan_eplb(now, ep, n_redundant), ep)),
        ])
    expected = popularity * n_tokens * TOP_K  # 每卡 token → ∞ 时的负载：没有采样噪声
    noise_free = imbalance(apply_plan(expected, plan_eplb(expected, ep, n_redundant), ep))
    means = [float(x) for x in np.mean(rows, axis=0)]
    return [None if np.isnan(x) else x for x in means] + [noise_free]


def part_b(seed: int = 0) -> None:
    print("\nB. EP load balance: max/mean dispatch-receive tokens per GPU (1.00 = perfect)")
    print(f"{'stage':<14}{'sigma':>6}{'naive':>8}{'greedy':>8}{'global':>8}{'hier':>8}{'now':>8}{'no-noise':>10}")
    rng = np.random.default_rng(seed)
    for sigma in SKEWS:
        popularity = np.exp(sigma * rng.standard_normal(V3_EXPERTS))
        popularity /= popularity.sum()
        for name, ep, tokens_per_gpu, red, nodes in STAGES:
            cols = eval_stage(rng, popularity, ep, ep * tokens_per_gpu, red, nodes)
            cells = "".join(f"{'n/a':>8}" if c is None else f"{c:>8.2f}" for c in cols[:-1])
            print(f"{name:<14}{sigma:>6.1f}{cells}{cols[-1]:>10.3f}")


# ============================================================================
# C. 跨节点 EP 的通信时间：纯带宽（ISCA'25 §2.3.2）vs 加上固定延迟 α
# ============================================================================

# DeepEP V1 低延迟 kernel 实测（docs/legacy.md）：H800 + CX7 400Gb/s IB，每批 128 token、hidden 7168、top-8，
# FP8 dispatch + BF16 combine；EP128 时 dispatch 192 µs、combine 369 µs
DEEPEP_TOKENS, DEEPEP_HIDDEN, DEEPEP_TOPK = 128, 7168, 8
DEEPEP_EP128_US = 192 + 369
IB_GB_S = 50  # V3 报告：IB 约 50 GB/s


def bw_only_us(tokens: int, hidden: int, n_targets: int, bw_gb_s: float) -> float:
    """dispatch（FP8，1 B）+ combine（BF16，2 B）的纯带宽时间。"""
    return 3 * tokens * n_targets * hidden / (bw_gb_s * GB) * 1e6


# 把「实测 − 纯带宽模型」全部记到 α 上：每次 all-to-all（dispatch 或 combine）一个 α。单点拟合，
# α 里同时混着固定延迟和带宽没跑满两部分，只能用来看数量级。
DEEPEP_MODEL_US = bw_only_us(DEEPEP_TOKENS, DEEPEP_HIDDEN, DEEPEP_TOPK, IB_GB_S)
ALPHA_US = (DEEPEP_EP128_US - DEEPEP_MODEL_US) / 2


def part_c() -> None:
    print("\nC. EP all-to-all per micro-batch (ISCA setup: 9 targets, hidden 7000, FP8+BF16; 61 layers x 2 micro-batches)")
    print(f"  alpha fit: DeepEP EP128 measured {DEEPEP_EP128_US} us vs bandwidth model {DEEPEP_MODEL_US:.1f} us"
          f" -> alpha = {ALPHA_US:.1f} us per all-to-all")
    print(f"{'interconnect':<16}{'tok/GPU/mb':>11}{'bw us':>9}{'+alpha us':>10}{'alpha %':>9}"
          f"{'TPOT bw':>9}{'TPOT +a':>9}")
    for label, bw, alpha in [("IB 50GB/s", IB_GB_S, ALPHA_US), ("NVL72 900GB/s", 900, None)]:
        for tokens in (8, 32, 128):
            bw_us = bw_only_us(tokens, 7_000, 9, bw)
            tpot_bw = 61 * 2 * bw_us / 1000
            if alpha is None:  # α 是在 IB/RDMA 上拟合的，不外推到 NVLink 域
                print(f"{label:<16}{tokens:>11}{bw_us:>9.2f}{'-':>10}{'-':>9}{tpot_bw:>9.2f}{'-':>9}")
                continue
            full = bw_us + 2 * alpha
            print(f"{label:<16}{tokens:>11}{bw_us:>9.2f}{full:>10.2f}{2 * alpha / full:>9.0%}"
                  f"{tpot_bw:>9.2f}{61 * 2 * full / 1000:>9.2f}")

    # 同一份流量的两种解释：把缺口记成固定延迟 α，还是记成链路效率（codesign_sim：80% 效率 + 每阶段 3.7 µs）
    lib_ep128 = ep_comm_time_s(H800, V3_MODEL, 128 * DEEPEP_TOKENS, 128) / V3_MOE_LAYERS * 1e6
    print(f"\nC2. Day 6 setup (EP144, 8 remote targets, hidden 7168), us per MoE layer per micro-batch")
    print(f"  check at EP128, 128 tok/GPU: DeepEP measured {DEEPEP_EP128_US} us, codesign_sim {lib_ep128:.1f} us")
    print(f"{'tok/GPU/mb':>11}{'bw only':>9}{'+alpha':>9}{'codesign_sim':>14}{'+alpha / sim':>14}")
    for tokens in (8, 32, 88, 128):
        bw_us = bw_only_us(tokens, V3_HIDDEN, 8, IB_GB_S)
        sim_us = ep_comm_time_s(H800, V3_MODEL, 144 * tokens, 144) / V3_MOE_LAYERS * 1e6
        full = bw_us + 2 * ALPHA_US
        print(f"{tokens:>11}{bw_us:>9.1f}{full:>9.1f}{sim_us:>14.1f}{full / sim_us:>13.1f}x")


# ============================================================================
# D. Day 6 账本（2025-02-27 12:00 至 02-28 12:00，UTC+8）与每张 GPU 的视角
# ============================================================================

AVG_NODES, GPU_HOUR_USD = 226.75, 2.0
INPUT_TOK, HIT_TOK, OUTPUT_TOK = 608e9, 342e9, 168e9
PREFILL_PER_NODE, DECODE_PER_NODE = 73.7e3, 14.8e3  # token/s/节点；prefill 的 73.7k 含缓存命中
USER_TOK_S, AVG_KV_LEN = 21.0, 4_989  # 平均输出速度 20–22 token/s 取中值；每个输出 token 的平均 KV 长度
DAY = 86_400


def per_gpu(s_q: int, tokens_per_step: float, alpha_us: float) -> dict:
    """与练习 (d) 同样的推导：并发 → KV → 每副本 token → 通信受限的 step 时间 → 速度上限。"""
    concurrent = DECODE_PER_NODE / 8 / USER_TOK_S
    tokens_mb = concurrent * s_q / 2
    per_replica_mb = tokens_mb * TOP_K * 144 / (V3_EXPERTS + 32)
    comm_us = 2 * alpha_us + bw_only_us(tokens_mb, V3_HIDDEN, 8, IB_GB_S)  # Day 6：共享专家在本卡，8 个远端目标
    step_ms = 61 * 2 * comm_us / 1000
    return {"concurrent": concurrent, "kv_gb": concurrent * AVG_KV_LEN * 70_272 / GB,
            "per_replica_mb": per_replica_mb, "step_ms": step_ms, "user_bound": tokens_per_step / step_ms * 1000}


def part_d() -> None:
    cost = AVG_NODES * 8 * 24 * GPU_HOUR_USD
    revenue = (HIT_TOK * 0.14 + (INPUT_TOK - HIT_TOK) * 0.55 + OUTPUT_TOK * 2.19) / 1e6  # 按 R1 价格
    prefill_nodes = INPUT_TOK / DAY / PREFILL_PER_NODE
    decode_nodes = OUTPUT_TOK / DAY / DECODE_PER_NODE
    print("\nD1. Re-deriving the Day 6 ledger")
    print(f"  daily cost   = 226.75 nodes x 8 GPUs x 24 h x $2 = ${cost:,.0f}   (official $87,072)")
    print(f"  revenue @R1  = ${revenue:,.0f}   (official $562,027; gap = rounded token counts)")
    print(f"  cost margin  = {(revenue - cost) / cost:.0%}   (official 545%)")
    print(f"  node-equiv.  = prefill {prefill_nodes:.1f} + decode {decode_nodes:.1f}"
          f" = {prefill_nodes + decode_nodes:.1f}   (official daily average {AVG_NODES})")
    miss_per_node = (INPUT_TOK - HIT_TOK) / DAY / prefill_nodes
    read_ms = 70_272 * 32_768 / (40 * 2**30) * 1e3  # 3FS：每个客户端节点 KVCache 读取峰值 40+ GiB/s
    print(f"  prefill miss-only = {miss_per_node / 1e3:.1f}k tok/s/node; re-prefill 32K = "
          f"{32_768 / miss_per_node:.2f} node-s vs disk read {read_ms:.0f} ms")

    routed_gb = 2 * V3_MOE_LAYERS * PARAMS_PER_EXPERT / GB  # EP144 + 32 冗余：每卡 2 个路由专家
    print(f"\nD2. One decode GPU (EP144, weights ~{V3_NONROUTED_PARAMS / GB:.1f} + {routed_gb:.1f} GB in FP8)")
    print(f"{'case':<22}{'req/GPU':>8}{'KV GB':>7}{'tok/replica/mb':>15}{'step ms':>9}{'tok/s bound':>12}")
    for label, s_q, tps, alpha in [("no MTP, bw only", 1, 1.0, 0.0), ("MTP, bw only", 2, 1.85, 0.0),
                                   ("MTP, +alpha", 2, 1.85, ALPHA_US)]:
        r = per_gpu(s_q, tps, alpha)
        print(f"{label:<22}{r['concurrent']:>8.1f}{r['kv_gb']:>7.1f}{r['per_replica_mb']:>15.0f}"
              f"{r['step_ms']:>9.1f}{r['user_bound']:>12.1f}")
    # 对照：codesign_sim 的 EP 通信项（只算 58 个 MoE 层，链路效率 80%，每阶段 3.7 µs）
    r = per_gpu(2, 1.85, 0.0)
    tokens_mb = r["concurrent"] * 2 / 2
    sim_step_ms = 2 * ep_comm_time_s(H800, V3_MODEL, 144 * tokens_mb, 144) * 1e3
    print(f"{'MTP, codesign_sim':<22}{r['concurrent']:>8.1f}{r['kv_gb']:>7.1f}{r['per_replica_mb']:>15.0f}"
          f"{sim_step_ms:>9.1f}{1.85 / sim_step_ms * 1000:>12.1f}")
    print(f"  observed average: {USER_TOK_S - 1:.0f}-{USER_TOK_S + 1:.0f} tok/s per request")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
