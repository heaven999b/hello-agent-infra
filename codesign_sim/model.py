"""模型结构参数 + 推理成本模型（FLOPs、访存字节、通信、roofline 步时间）。

结构参数全部来自官方 config / 官方代码仓库，见每个条目的 ``source``。
成本模型只算"主项"：
- GEMM：非 embedding 的激活参数 2·N FLOPs/token；lm_head 只在需要 logits 的位置算（prefill 每条序列 1 个）；
  embedding 是查表，不算 FLOPs，读权重时也只读 batch 行
- 注意力：QK^T 与 PV（可以用与 GEMM 不同的精度计时，见 step_time_s 的 attn_flops / attn_dtype）
- 访存：本步触达的权重（MoE 按专家粒度）+ KV 读写
- 通信（可选）：TP 每层 2 次 all-reduce（NVLink）、EP dispatch/combine（节点内 NVLink、跨节点 IB）
忽略 norm、激活函数、softmax、采样等次要项。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hardware import (
    GPU,
    IB_GBPS_PER_GPU,
    IB_LATENCY_S,
    LINK_EFFICIENCY,
    NVLINK_LATENCY_S,
    peak_tflops,
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    attn: str  # "mha" | "gqa" | "mla"
    total_params_b: float
    active_params_b: float
    kv_lora_rank: int = 0  # MLA
    qk_rope_head_dim: int = 0  # MLA
    source: str = ""
    # ---- 以下是库扩展字段（契约之外，都有默认值；0 表示"未知/不适用"，相关修正自动关闭）----
    vocab_size: int = 0
    tie_embeddings: bool = False  # 输入 embedding 与 lm_head 是否共享权重
    max_context: int = 0  # 官方支持的最大上下文（token）
    n_routed_experts: int = 0  # MoE：路由专家数 E
    experts_per_tok: int = 0  # MoE：每 token 激活的路由专家数 k
    moe_intermediate_size: int = 0  # MoE：每个专家 FFN 的中间维度
    n_dense_layers: int = 0  # MoE：前几层是 dense FFN（DeepSeek 的 first_k_dense_replace）
    n_shared_experts: int = 0  # MoE：共享专家数（算在"每步必读"的参数里）

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts > 0 and self.experts_per_tok > 0


_LLAMA_SRC = (
    "https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py (llama3_1 arch_args); "
    "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md (context 128k); "
)

MODELS: dict[str, ModelConfig] = {
    # Llama 3.1 与 Llama 3 结构完全相同（sku_list 里只差 use_scaled_rope），但上下文 128k（Llama 3 只有 8k），
    # 与本课程的长上下文 agent 负载一致。HF 仓库是 gated，结构取 Meta 官方 llama-models；
    # 参数量取 HF API 的 safetensors 统计（8,030,261,248 / 70,553,706,496）。词表 128256，lm_head 不共享。
    "llama3.1-8b": ModelConfig(
        name="llama3.1-8b",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,  # dim / n_heads = 4096 / 32
        attn="gqa",
        total_params_b=8.03,
        active_params_b=8.03,
        source=_LLAMA_SRC + "https://huggingface.co/meta-llama/Llama-3.1-8B",
        vocab_size=128256,
        tie_embeddings=False,
        max_context=131072,
    ),
    "llama3.1-70b": ModelConfig(
        name="llama3.1-70b",
        n_layers=80,
        d_model=8192,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,  # 8192 / 64
        attn="gqa",
        total_params_b=70.55,
        active_params_b=70.55,
        source=_LLAMA_SRC + "https://huggingface.co/meta-llama/Llama-3.1-70B",
        vocab_size=128256,
        tie_embeddings=False,
        max_context=131072,
    ),
    # Qwen3-32B config.json：64 层、hidden 5120、64 Q 头 / 8 KV 头、head_dim 128、vocab 151936、不共享 lm_head；
    # model card：32.8B 参数，原生 32,768、YaRN 扩展到 131,072 token
    "qwen3-32b": ModelConfig(
        name="qwen3-32b",
        n_layers=64,
        d_model=5120,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        attn="gqa",
        total_params_b=32.8,
        active_params_b=32.8,
        source="https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json",
        vocab_size=151936,
        tie_embeddings=False,
        max_context=131072,
    ),
    # DeepSeek-V3 config.json：61 层、hidden 7168、128 头、kv_lora_rank 512、qk_rope_head_dim 64、
    # qk_nope_head_dim = v_head_dim = 128、n_routed_experts 256、num_experts_per_tok 8、moe_intermediate_size 2048、
    # first_k_dense_replace 3、n_shared_experts 1、vocab 129280；权重官方即 FP8（e4m3，128×128 block）。
    # model card：671B 总参数 / 37B 激活（不含 14B MTP 模块）、128K 上下文
    "deepseek-v3": ModelConfig(
        name="deepseek-v3",
        n_layers=61,
        d_model=7168,
        n_heads=128,
        n_kv_heads=128,
        head_dim=128,  # qk_nope_head_dim = v_head_dim = 128
        attn="mla",
        total_params_b=671.0,
        active_params_b=37.0,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        source="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",
        vocab_size=129280,
        tie_embeddings=False,
        max_context=131072,
        n_routed_experts=256,
        experts_per_tok=8,
        moe_intermediate_size=2048,
        n_dense_layers=3,
        n_shared_experts=1,
    ),
}
# 旧键名别名（早期版本叫 llama3-8b / llama3-70b；结构参数相同，指向同一个对象）
MODELS["llama3-8b"] = MODELS["llama3.1-8b"]
MODELS["llama3-70b"] = MODELS["llama3.1-70b"]


# ---------------------------------------------------------------------------
# 参数拆分：embedding / lm_head / MoE 专家
# ---------------------------------------------------------------------------


def embedding_params(cfg: ModelConfig) -> tuple[float, float]:
    """(输入 embedding 表参数量, lm_head 独立参数量)。共享权重时 lm_head 独立参数为 0；vocab 未知时都为 0。"""
    table = float(cfg.vocab_size * cfg.d_model)
    return table, (0.0 if cfg.tie_embeddings else table)


def body_active_params(cfg: ModelConfig) -> float:
    """每个 token 参与 GEMM 的激活参数（不含 embedding 表与 lm_head）。"""
    emb, head = embedding_params(cfg)
    return cfg.active_params_b * 1e9 - emb - head


def moe_params(cfg: ModelConfig) -> dict:
    """MoE 参数拆分（按 config 结构推算）：每专家参数、路由专家总量、每步必读的非路由参数。

    每专家 = 3 · d_model · moe_intermediate_size（gate/up/down 三个矩阵）；
    路由专家总量 = MoE 层数 · E · 每专家；非路由（注意力、共享专家、dense 层、embedding、gate）= 总量 - 路由专家。
    DeepSeek-V3：每专家 44.0M、路由专家 653.9B、非路由约 17.1B。
    """
    if not cfg.is_moe:
        return {"expert_params": 0.0, "moe_layers": 0, "routed_params": 0.0,
                "always_active_params": cfg.total_params_b * 1e9}
    expert = 3.0 * cfg.d_model * cfg.moe_intermediate_size
    moe_layers = cfg.n_layers - cfg.n_dense_layers
    routed = moe_layers * cfg.n_routed_experts * expert
    return {"expert_params": expert, "moe_layers": moe_layers, "routed_params": routed,
            "always_active_params": cfg.total_params_b * 1e9 - routed}


def expected_unique_experts(n_experts: int, k: int, n_tokens: int, skew: float = 0.0) -> float:
    """n_tokens 个 token、每个选 k 个专家时，一层里平均被选中的不同专家数。

    skew=0：均匀路由，E·(1 - (1 - k/E)^n)；skew>0：专家热度 ∝ 1/(e+1)^skew（热门专家反复被选，覆盖更少）。
    """
    if n_tokens <= 0:
        return 0.0
    if skew <= 0:
        return n_experts * (1.0 - (1.0 - k / n_experts) ** n_tokens)
    p = 1.0 / np.arange(1, n_experts + 1) ** skew
    q = np.minimum(1.0, k * p / p.sum())
    return float((1.0 - (1.0 - q) ** n_tokens).sum())


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


def kv_bytes_per_token(cfg: ModelConfig, kv_dtype_bytes: int = 2) -> int:
    """每个 token 的 KV cache 字节数（所有层合计）。

    - mha/gqa：K 和 V 各一份 -> 2 · n_layers · n_kv_heads · head_dim · bytes
    - mla：只缓存压缩后的 latent c_KV（kv_lora_rank）和解耦的 RoPE key（qk_rope_head_dim），
      所有头共享 -> n_layers · (kv_lora_rank + qk_rope_head_dim) · bytes
    """
    if cfg.attn == "mla":
        return cfg.n_layers * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * kv_dtype_bytes
    if cfg.attn in ("mha", "gqa"):
        return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * kv_dtype_bytes
    raise ValueError(f"未知 attn 类型: {cfg.attn!r}")


def weight_bytes_total(cfg: ModelConfig, weight_bytes: float = 2) -> float:
    """全部权重的字节数（MoE 含所有专家）。"""
    return cfg.total_params_b * 1e9 * weight_bytes


def weight_bytes_touched(cfg: ModelConfig, n_tokens: int, weight_bytes: float = 2, *, expert_skew: float = 0.0) -> float:
    """一步前向（本步共 n_tokens 个 token）实际要从 HBM 读出的权重字节数。

    - 输入 embedding 是查表：只读 n_tokens 行，不读整张表（共享 lm_head 时整张表本来就要读）
    - lm_head 整张读（每步都要出 logits）
    - dense：其余参数整份读一遍
    - MoE（按专家粒度）：非路由参数每步必读 + 每个 MoE 层读"被选中的不同专家"，
      均匀路由时期望为 E·(1-(1-k/E)^n)；expert_skew > 0 模拟热点专家
    """
    if n_tokens <= 0:
        return 0.0
    emb, head = embedding_params(cfg)
    rows = 0.0 if (emb == 0 or cfg.tie_embeddings) else min(n_tokens, cfg.vocab_size) * cfg.d_model
    if not cfg.is_moe:
        return (cfg.total_params_b * 1e9 - emb + (emb if cfg.tie_embeddings else 0.0) + rows) * weight_bytes
    mp = moe_params(cfg)
    always = mp["always_active_params"] - emb + (emb if cfg.tie_embeddings else 0.0) + rows
    uniq = expected_unique_experts(cfg.n_routed_experts, cfg.experts_per_tok, n_tokens, expert_skew)
    return (always + mp["moe_layers"] * uniq * mp["expert_params"]) * weight_bytes


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def attn_flops_per_pair(cfg: ModelConfig, phase: str = "prefill") -> float:
    """每个 (query token, key token) 对、所有层所有头合计的注意力 FLOPs（QK^T + PV，乘加算 2）。

    - mha/gqa：2 · L · H · (d_qk + d_v) = 4 · L · H · head_dim（GQA 省的是 KV 存储，不省 FLOPs）
    - mla prefill（"MHA 模式"，先把 latent 展开成每头 K/V）：d_qk = head_dim + rope，d_v = head_dim
    - mla decode（"吸收"权重后的 MQA 模式，直接在 latent 上做注意力）：
      d_qk = kv_lora_rank + rope，d_v = kv_lora_rank —— 计算更多、访存更少，这是 MLA 的关键取舍
    """
    L, H = cfg.n_layers, cfg.n_heads
    if cfg.attn == "mla":
        if phase == "decode":
            d_qk = cfg.kv_lora_rank + cfg.qk_rope_head_dim
            d_v = cfg.kv_lora_rank
        else:
            d_qk = cfg.head_dim + cfg.qk_rope_head_dim
            d_v = cfg.head_dim
    else:
        d_qk = d_v = cfg.head_dim
    return 2.0 * L * H * (d_qk + d_v)


def _lm_head_flops(cfg: ModelConfig) -> float:
    return 2.0 * cfg.vocab_size * cfg.d_model


def prefill_flops_split(cfg: ModelConfig, n_new: int, n_cached: int = 0, logits_positions: int = 1) -> tuple[float, float]:
    """prefill 的 (GEMM FLOPs, 注意力 FLOPs)。

    GEMM = 2 · N_body · n_new + lm_head 只算 logits_positions 个位置（默认 1：只要最后一个 token 的 logits）。
    注意力：第 k 个新 token（k=1..n_new）看 n_cached + k 个 token，共 n_new·n_cached + n_new(n_new+1)/2 对。
    """
    if n_new <= 0:
        return 0.0, 0.0
    gemm = 2.0 * body_active_params(cfg) * n_new + _lm_head_flops(cfg) * min(logits_positions, n_new)
    pairs = n_new * n_cached + n_new * (n_new + 1) / 2.0
    return gemm, attn_flops_per_pair(cfg, "prefill") * pairs


def prefill_flops(cfg: ModelConfig, n_new: int, n_cached: int = 0, *, logits_positions: int = 1) -> float:
    """prefill n_new 个新 token（前面已有 n_cached 个 token 的 KV）所需 FLOPs（GEMM + 注意力）。

    ≈ 2 · N_body · n_new（N_body = 去掉 embedding 与 lm_head 的激活参数）+ 1 个位置的 lm_head + 注意力项。
    注意：分块 prefill 时每块默认都算 1 个 logits 位置（只有最后一块真的需要），多出的量可以忽略。
    """
    gemm, attn = prefill_flops_split(cfg, n_new, n_cached, logits_positions)
    return gemm + attn


def decode_flops_split(cfg: ModelConfig, batch: int, context_len: int) -> tuple[float, float]:
    """一次 decode 步的 (GEMM FLOPs, 注意力 FLOPs)；每个 token 都要算 lm_head。"""
    if batch <= 0:
        return 0.0, 0.0
    gemm = batch * (2.0 * body_active_params(cfg) + _lm_head_flops(cfg))
    return gemm, batch * attn_flops_per_pair(cfg, "decode") * context_len


def decode_flops(cfg: ModelConfig, batch: int, context_len: int) -> float:
    """一次 decode 步（batch 个序列各生成 1 个 token，平均上下文 context_len）的 FLOPs。"""
    gemm, attn = decode_flops_split(cfg, batch, context_len)
    return gemm + attn


def decode_bytes(
    cfg: ModelConfig,
    batch: int,
    context_len: int,
    weight_bytes: float = 2,
    kv_dtype_bytes: int = 2,
    *,
    kv_replicas: int = 1,
    expert_skew: float = 0.0,
) -> float:
    """一次 decode 步要从 HBM 读的字节：权重（MoE 按专家粒度）+ 每个序列读自己的整段 KV cache。

    kv_replicas：MLA 在 TP 下 latent KV 无法按头切分，每个 TP rank 都要读完整 KV，此时传 tp。
    """
    if batch <= 0:
        return 0.0
    w = weight_bytes_touched(cfg, batch, weight_bytes, expert_skew=expert_skew)
    kv = batch * context_len * kv_bytes_per_token(cfg, kv_dtype_bytes) * kv_replicas
    return w + kv


# ---------------------------------------------------------------------------
# 通信
# ---------------------------------------------------------------------------


def allreduce_time_s(nbytes: float, n_ranks: int, link_gbps_bidir: float, latency_s: float = NVLINK_LATENCY_S,
                     efficiency: float = LINK_EFFICIENCY, algo: str = "two_shot") -> float:
    """all-reduce 时间 = 带宽项 + 延迟项。两种算法的带宽项相同：每卡收发 2(p-1)/p · 数据量。

    - "two_shot"（默认）：NVSwitch 全互联时，reduce-scatter 与 all-gather 各一步点对点完成，
      延迟项 2α。vLLM 的自定义 all-reduce、NCCL 在 NVSwitch 上的小消息算法属于这一类。
    - "ring"：p 张卡连成环，两个阶段各 p-1 步，延迟项 2(p-1)α；小消息时比 two_shot 慢得多。
    """
    if algo not in ("two_shot", "ring"):
        raise ValueError(f"algo 必须是 'two_shot' 或 'ring'，收到 {algo!r}")
    if n_ranks <= 1 or nbytes <= 0:
        return 0.0
    bw = link_gbps_bidir / 2.0 * 1e9 * efficiency
    steps = 2.0 if algo == "two_shot" else 2.0 * (n_ranks - 1)
    return 2.0 * (n_ranks - 1) / n_ranks * nbytes / bw + steps * latency_s


def tp_comm_time_s(gpu: GPU, cfg: ModelConfig, n_tokens: int, tp: int, act_bytes: int = 2) -> float:
    """TP 通信：每层 2 次 all-reduce（注意力输出 + FFN 输出），数据量 n_tokens · d_model · act_bytes，走 NVLink。"""
    if tp <= 1 or n_tokens <= 0:
        return 0.0
    per = allreduce_time_s(n_tokens * cfg.d_model * act_bytes, tp, gpu.nvlink_gbps)
    return cfg.n_layers * 2 * per


def ep_comm_time_s(gpu: GPU, cfg: ModelConfig, n_tokens: int, ep: int, *, gpus_per_node: int = 8,
                   ib_gbps: float = IB_GBPS_PER_GPU, dispatch_bytes: int = 1, combine_bytes: int = 2) -> float:
    """EP 通信（每个 MoE 层一次 dispatch + 一次 combine）。

    每卡负责 n_tokens/ep 个 token，每个 token 发往 k 个专家，其中 (ep-1)/ep 在别的卡上；
    dispatch 用 FP8（1 字节）、combine 用 BF16（2 字节），与 DeepSeek 推理系统的做法一致。
    ep > gpus_per_node 时，远端专家里有 (ep - gpus_per_node)/(ep - 1) 在其他节点，走每卡 ib_gbps 的 IB；
    其余走 NVLink（单向 = nvlink_gbps/2）。两条链路并行，取较慢者；再加每阶段一个消息延迟。
    简化：不做 DeepSeek 的"同节点去重 + NVLink 转发"，也不建模专家负载不均。
    """
    if not cfg.is_moe or ep <= 1 or n_tokens <= 0:
        return 0.0
    tok = n_tokens / ep
    remote = tok * cfg.experts_per_tok * cfg.d_model * (dispatch_bytes + combine_bytes) * (ep - 1) / ep
    cross = max(0, ep - gpus_per_node) / (ep - 1)
    t_nvl = remote * (1 - cross) / (gpu.nvlink_gbps / 2.0 * 1e9 * LINK_EFFICIENCY) if gpu.nvlink_gbps else 0.0
    t_ib = remote * cross / (ib_gbps * 1e9 * LINK_EFFICIENCY) if cross > 0 else 0.0
    latency = 2 * (IB_LATENCY_S if cross > 0 else NVLINK_LATENCY_S)
    return moe_params(cfg)["moe_layers"] * (max(t_nvl, t_ib) + latency)


# ---------------------------------------------------------------------------
# roofline 步时间
# ---------------------------------------------------------------------------


def step_time_s(
    gpu: GPU,
    cfg: ModelConfig,
    flops: float,
    bytes_: float,
    n_gpus: int = 1,
    mfu: float = 0.6,
    mbu: float = 0.8,
    dtype: str = "bf16",
    *,
    attn_flops: float = 0.0,
    attn_dtype: str = "bf16",
    comm_s: float = 0.0,
    comm_overlap: float = 0.0,
) -> float:
    """roofline 步时间 = max(计算时间, 访存时间) + 未被掩盖的通信时间。

    计算时间 = flops / (n_gpus · 峰值[dtype] · mfu) + attn_flops / (n_gpus · 峰值[attn_dtype] · mfu)
    （flops 按 dtype 计时——通常是 GEMM；attn_flops 可以单独按注意力精度计时，默认 0 即旧行为）
    访存时间 = bytes_ / (n_gpus · 带宽 · mbu)
    通信：comm_s 中 comm_overlap 比例可与计算/访存重叠（取 max），其余串行相加。
    假设 n_gpus 张卡完美切分工作量。``cfg`` 参数保留给接口一致性。
    """
    del cfg  # 成本已经体现在 flops / bytes_ 里
    if n_gpus <= 0:
        raise ValueError("n_gpus 必须 >= 1")
    if not 0.0 <= comm_overlap <= 1.0:
        raise ValueError("comm_overlap 必须在 [0, 1]")
    compute = flops / (n_gpus * peak_tflops(gpu, dtype) * 1e12 * mfu)
    if attn_flops:
        compute += attn_flops / (n_gpus * peak_tflops(gpu, attn_dtype) * 1e12 * mfu)
    memory = bytes_ / (n_gpus * gpu.hbm_tbps * 1e12 * mbu)
    core = max(compute, memory)
    return max(core, comm_s * comm_overlap) + comm_s * (1.0 - comm_overlap)


def kv_capacity_tokens(
    gpu: GPU,
    cfg: ModelConfig,
    n_gpus: int = 1,
    weight_bytes: float = 2,
    kv_dtype_bytes: int = 2,
    mem_frac: float = 0.9,
) -> int:
    """n_gpus 张卡装下权重后还能放多少 token 的 KV cache。

    (n_gpus · HBM · mem_frac - 权重) / 每 token KV 字节；mem_frac 预留给激活与碎片。
    权重放不下时返回负数（调用方自行报错）。
    """
    free = n_gpus * gpu.hbm_gb * 1e9 * mem_frac - weight_bytes_total(cfg, weight_bytes)
    return int(free // kv_bytes_per_token(cfg, kv_dtype_bytes))
