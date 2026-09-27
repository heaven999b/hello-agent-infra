"""model：结构参数、KV 公式（含 MLA）、embedding/lm_head 修正、MoE 专家粒度字节、FP8、通信项。"""

import pytest

from codesign_sim.hardware import GPUS
from codesign_sim.model import (
    MODELS,
    allreduce_time_s,
    body_active_params,
    decode_bytes,
    decode_flops,
    ep_comm_time_s,
    expected_unique_experts,
    kv_bytes_per_token,
    kv_capacity_tokens,
    moe_params,
    prefill_flops,
    prefill_flops_split,
    step_time_s,
    tp_comm_time_s,
    weight_bytes_touched,
)

L8, L70, DS = MODELS["llama3.1-8b"], MODELS["llama3.1-70b"], MODELS["deepseek-v3"]


def test_structural_params_and_aliases():
    assert (DS.n_layers, DS.d_model, DS.n_heads, DS.kv_lora_rank, DS.qk_rope_head_dim) == (61, 7168, 128, 512, 64)
    assert (DS.n_routed_experts, DS.experts_per_tok, DS.moe_intermediate_size, DS.n_dense_layers) == (256, 8, 2048, 3)
    assert DS.attn == "mla" and DS.total_params_b == 671 and DS.active_params_b == 37 and DS.vocab_size == 129280
    q = MODELS["qwen3-32b"]
    assert (q.n_layers, q.d_model, q.n_heads, q.n_kv_heads, q.head_dim, q.vocab_size) == (64, 5120, 64, 8, 128, 151936)
    assert (L8.n_layers, L70.n_layers, L8.max_context) == (32, 80, 131072)
    assert MODELS["llama3-8b"] is L8 and MODELS["llama3-70b"] is L70  # 旧键名别名
    for cfg in MODELS.values():
        assert cfg.source


def test_kv_bytes_gqa_and_mla():
    assert kv_bytes_per_token(L8) == 2 * 32 * 8 * 128 * 2 == 131072  # 128 KiB
    assert kv_bytes_per_token(L70) == 2 * 80 * 8 * 128 * 2 == 327680  # 320 KiB
    assert kv_bytes_per_token(L8, kv_dtype_bytes=1) == 65536  # FP8 KV 减半
    assert kv_bytes_per_token(DS) == 61 * (512 + 64) * 2 == 70272  # MLA：只存 latent + RoPE key
    assert kv_bytes_per_token(L70) / kv_bytes_per_token(DS) > 4


def test_embedding_and_lm_head_not_counted_as_gemm():
    # llama3.1-8b：embedding 与 lm_head 各 128256×4096 ≈ 0.525B，占 8.03B 的 13%
    assert body_active_params(L8) == pytest.approx(8.03e9 - 2 * 128256 * 4096)
    gemm, _ = prefill_flops_split(L8, 1000)
    assert gemm == pytest.approx(2 * body_active_params(L8) * 1000 + 2 * 128256 * 4096)  # lm_head 只算最后 1 个位置
    assert prefill_flops(L8, 1000) < 2 * 8.03e9 * 1000  # 比 2·N·n 少约 13%
    assert decode_flops(L8, 1, 0) == pytest.approx(2 * 8.03e9 - 2 * 128256 * 4096)  # decode 每步都要 lm_head
    # 读权重：输入 embedding 只读 batch 行，不读整张表
    assert weight_bytes_touched(L8, 4) == pytest.approx((8.03e9 - 128256 * 4096 + 4 * 4096) * 2)


def test_chunked_prefill_flops_additive():
    whole = prefill_flops(L8, 6000)
    chunked = (prefill_flops(L8, 2048, logits_positions=0) + prefill_flops(L8, 2048, 2048, logits_positions=0)
               + prefill_flops(L8, 1904, 4096))
    assert chunked == pytest.approx(whole)  # 因果注意力的 (q,k) 对数可加
    assert prefill_flops(L8, 100_000) / (2 * body_active_params(L8) * 100_000) > 1.3  # 长上下文注意力项显著


def test_moe_weight_bytes_expert_granular():
    mp = moe_params(DS)
    assert mp["expert_params"] == 3 * 7168 * 2048 and mp["moe_layers"] == 58
    assert 16e9 < mp["always_active_params"] < 18e9  # 非路由参数约 17.1B
    assert weight_bytes_touched(DS, 1, 1) == pytest.approx(37e9, rel=0.03)  # 单 token ≈ 激活参数
    b8 = weight_bytes_touched(DS, 8, 1)
    assert 150e9 < b8 < 170e9  # 专家级：8 个 token 约 160 GB（"参数独立被选中"的旧近似给出 245 GB）
    assert weight_bytes_touched(DS, 4096, 1) == pytest.approx(671e9 - 129280 * 7168, rel=0.01)
    assert expected_unique_experts(256, 8, 32) == pytest.approx(256 * (1 - (1 - 8 / 256) ** 32))
    assert expected_unique_experts(256, 8, 32, skew=1.0) < expected_unique_experts(256, 8, 32)  # 热点专家覆盖更少
    # dense：除了 embedding 的 batch 行，读的字节与 batch 无关
    assert weight_bytes_touched(L8, 512) - weight_bytes_touched(L8, 1) == pytest.approx(511 * 4096 * 2)


def test_decode_bytes_and_mla_tp_replication():
    b = decode_bytes(L70, 4, 1000)
    assert b == pytest.approx(weight_bytes_touched(L70, 4) + 4 * 1000 * 327680)
    # MLA 的 latent KV 不能按头切：TP8 下每个 rank 都读完整 KV
    assert decode_bytes(DS, 64, 4096, 1, kv_replicas=8) - decode_bytes(DS, 64, 4096, 1) == pytest.approx(
        7 * 64 * 4096 * 70272)


def test_step_time_regimes_and_fp8():
    gpu = GPUS["H100-SXM"]
    f, b = decode_flops(L8, 1, 1000), decode_bytes(L8, 1, 1000)
    t = step_time_s(gpu, L8, f, b)
    assert t == pytest.approx(b / (gpu.hbm_tbps * 1e12 * 0.8))  # batch=1 decode：访存受限
    assert 4e-3 < t < 8e-3
    gemm, attn = prefill_flops_split(L8, 8192)
    t_bf16 = step_time_s(gpu, L8, gemm, 16e9, attn_flops=attn)
    t_fp8 = step_time_s(gpu, L8, gemm, 8e9, dtype="fp8", attn_flops=attn)
    assert t_bf16 == pytest.approx((gemm + attn) / (989e12 * 0.6))  # 算力受限
    assert t_fp8 == pytest.approx(gemm / (1979e12 * 0.6) + attn / (989e12 * 0.6))  # GEMM 用 FP8 峰值，注意力仍是 BF16
    assert step_time_s(gpu, L8, gemm, 16e9, n_gpus=2, attn_flops=attn) == pytest.approx(t_bf16 / 2)


def test_communication_terms():
    # ring all-reduce：2(p-1)/p · 数据 / 单向带宽 + 2α
    t = allreduce_time_s(1e9, 8, 900, latency_s=0, efficiency=1.0)
    assert t == pytest.approx(2 * 7 / 8 * 1e9 / 450e9)
    h100, h800 = GPUS["H100-SXM"], GPUS["H800-SXM"]
    # H800 只砍了 NVLink：TP 通信慢约 2.25 倍（带宽项），这是 H100 与 H800 唯一的差别
    big = 8192
    ratio = tp_comm_time_s(h800, L70, big, 8) / tp_comm_time_s(h100, L70, big, 8)
    assert 2.0 < ratio < 2.3
    assert tp_comm_time_s(h100, L70, 128, 1) == 0.0
    # EP：跨节点走 IB（每卡 50 GB/s）比节点内 NVLink 慢得多
    intra = ep_comm_time_s(h800, DS, 4096, 8)
    cross = ep_comm_time_s(h800, DS, 4096 * 4, 32)  # 每卡 token 数相同，但 3/4 的远端专家在别的节点
    assert cross > 2 * intra > 0
    assert ep_comm_time_s(h800, L70, 4096, 8) == 0.0  # dense 模型没有 EP
    # 通信重叠：全部重叠时取 max，不重叠时相加
    assert step_time_s(h100, L8, 0.0, 1e9, comm_s=0.1, comm_overlap=1.0) == pytest.approx(0.1)
    assert step_time_s(h100, L8, 0.0, 1e9, comm_s=0.1) == pytest.approx(0.1 + 1e9 / (3.35e12 * 0.8))


def test_kv_capacity():
    gpu = GPUS["H100-SXM"]
    cap = kv_capacity_tokens(gpu, L70, n_gpus=8)
    assert cap == int((8 * 80e9 * 0.9 - 70.55e9 * 2) // 327680)
    assert 1.2e6 < cap < 1.4e6
    assert kv_capacity_tokens(gpu, DS, n_gpus=8) < 0  # BF16 权重 1.34 TB，8 张 80GB 放不下
    assert kv_capacity_tokens(GPUS["B200"], DS, n_gpus=8, weight_bytes=1) > 0


def test_allreduce_ring_latency_grows_with_ranks():
    from codesign_sim.model import allreduce_time_s
    small = 1024.0                                          # 小消息：延迟项主导
    two_shot = allreduce_time_s(small, 8, 900.0)
    ring = allreduce_time_s(small, 8, 900.0, algo="ring")
    assert ring > 5 * two_shot                              # ring 延迟项 2(p-1)α = 14α，two-shot 为 2α
    big = 1e9                                               # 大消息：两者带宽项相同
    assert abs(allreduce_time_s(big, 8, 900.0) - allreduce_time_s(big, 8, 900.0, algo="ring")) / allreduce_time_s(big, 8, 900.0) < 0.01
