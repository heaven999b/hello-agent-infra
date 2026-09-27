"""模块 03 练习测试。离线、确定性。

运行：make module N=03
验证参考答案：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/03_deepseek_stack
"""

from __future__ import annotations

import numpy as np
import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)


# ================================================================== (a) kv_bytes_per_token


def test_a_mla_deepseek_v3_matches_isca_table():
    # DeepSeek-V3：61 层，kv_lora_rank=512，qk_rope_head_dim=64（官方 config.json）
    assert ex.kv_bytes_per_token("mla", 61, kv_lora_rank=512, qk_rope_head_dim=64) == 70_272


def test_a_gqa_models_match_isca_table():
    assert ex.kv_bytes_per_token("gqa", 80, n_kv_heads=8, head_dim=128) == 327_680  # Qwen-2.5 72B
    assert ex.kv_bytes_per_token("gqa", 126, n_kv_heads=8, head_dim=128) == 516_096  # LLaMA-3.1 405B


def test_a_asymmetric_k_and_v():
    # V3 若改用 MHA：K = 128 nope + 64 rope = 192，V = 128
    assert ex.kv_bytes_per_token("mha", 61, n_kv_heads=128, k_head_dim=192, v_head_dim=128) == 4_997_120
    assert ex.kv_bytes_per_token("gqa", 61, n_kv_heads=8, k_head_dim=192, v_head_dim=128) == 312_320
    # 只给 head_dim 时 K、V 同维
    assert ex.kv_bytes_per_token("mha", 32, n_kv_heads=32, head_dim=128) == 2 * 32 * 32 * 128 * 2


def test_a_mqa_ignores_n_kv_heads():
    assert ex.kv_bytes_per_token("mqa", 32, n_kv_heads=32, head_dim=128) == 2 * 32 * 128 * 2


def test_a_dtype_bytes_and_sub_byte_rounding():
    bf16 = ex.kv_bytes_per_token("mla", 61, kv_lora_rank=512, qk_rope_head_dim=64)
    fp8 = ex.kv_bytes_per_token("mla", 61, kv_lora_rank=512, qk_rope_head_dim=64, dtype_bytes=1)
    fp4 = ex.kv_bytes_per_token("mla", 61, kv_lora_rank=512, qk_rope_head_dim=64, dtype_bytes=0.5)
    assert (bf16, fp8, fp4) == (70_272, 35_136, 17_568)
    assert isinstance(fp4, int)
    # 奇数个 FP4 元素：物理上仍要占整字节，向上取整
    assert ex.kv_bytes_per_token("mqa", 1, head_dim=1, dtype_bytes=0.5) == 1
    assert ex.kv_bytes_per_token("mla", 1, kv_lora_rank=3, qk_rope_head_dim=0, dtype_bytes=0.5) == 2


def test_a_mla_equals_gqa_with_2_25_groups_for_v2_shape():
    # V2 论文：d_c = 4·d_h，d_h^R = d_h/2 → MLA 的 KV 等于 2.25 组的 GQA
    mla = ex.kv_bytes_per_token("mla", 60, kv_lora_rank=4 * 128, qk_rope_head_dim=64)
    per_group = ex.kv_bytes_per_token("gqa", 60, n_kv_heads=1, head_dim=128)
    assert mla / per_group == pytest.approx(2.25)


def test_a_unknown_attn_raises():
    with pytest.raises(ValueError):
        ex.kv_bytes_per_token("linear", 10, n_kv_heads=1, head_dim=64)


# ================================================================== (b) eplb_gpu_loads


def test_b_hand_example_with_and_without_redundancy():
    loads = np.array([10.0, 2.0, 2.0, 2.0])
    # 无冗余：2 张卡、每卡 2 个槽位 → 10 与 2 挤在一张卡上
    assert sorted(ex.eplb_gpu_loads(loads, 2, 0)) == pytest.approx([4.0, 12.0])
    # 2 个冗余副本全部给热门专家 0 → 3 份各 10/3，再装箱
    assert sorted(ex.eplb_gpu_loads(loads, 2, 2)) == pytest.approx([22 / 3, 26 / 3])


# deepseek-ai/EPLB README 的官方示例：12 个专家、4 组、4 个冗余、2 节点 × 4 卡（走层次策略）
EPLB_README_WEIGHT = np.array([
    [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
    [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
], dtype=float)
# 由官方 README 给出的 phy2log 输出换算出的每卡负载（已排序）
EPLB_README_GPU_LOADS = [
    [86.5, 113.0, 121.5, 125.0, 131.5, 147.5, 152.0, 156.0],
    [117.5, 118.5, 120.5, 123.0, 152.0, 172.0, 173.0, 179.5],
]


@pytest.mark.parametrize("layer", [0, 1])
def test_b_hierarchical_matches_official_readme_example(layer):
    out = ex.eplb_gpu_loads(EPLB_README_WEIGHT[layer], 8, 4, n_groups=4, n_nodes=2)
    assert sorted(out) == pytest.approx(EPLB_README_GPU_LOADS[layer])


def test_b_indivisible_groups_fall_back_to_global():
    rng = np.random.default_rng(3)
    loads = rng.lognormal(0.0, 1.0, size=256) * 1000
    # 8 组不能被 18 个节点整除 → 与全局策略结果相同（Day 6 decode EP144 的情形）
    fallback = np.sort(ex.eplb_gpu_loads(loads, 144, 32, n_groups=8, n_nodes=18))
    global_ = np.sort(ex.eplb_gpu_loads(loads, 144, 32))
    assert fallback == pytest.approx(global_)


def test_b_conserves_total_load_and_shape():
    rng = np.random.default_rng(0)
    loads = rng.lognormal(0.0, 1.0, size=256) * 1000
    for kwargs in ({}, {"n_groups": 8, "n_nodes": 4}):
        out = np.asarray(ex.eplb_gpu_loads(loads, 32, 32, **kwargs))
        assert out.shape == (32,)
        assert out.sum() == pytest.approx(loads.sum())


def test_b_uniform_load_structural_floor():
    # 热度均匀时：EP128 无冗余完美均衡；EP144 配 32 冗余（288 槽）下限是 2 / (256/144) = 1.125
    ones = np.ones(256)
    ep128 = np.asarray(ex.eplb_gpu_loads(ones, 128, 0))
    ep144 = np.asarray(ex.eplb_gpu_loads(ones, 144, 32))
    assert ep128.max() / ep128.mean() == pytest.approx(1.0)
    assert ep144.max() / ep144.mean() == pytest.approx(1.125)


def test_b_redundancy_helps_on_skewed_load():
    rng = np.random.default_rng(1)
    loads = rng.lognormal(0.0, 1.0, size=256) * 1000
    no_red = np.asarray(ex.eplb_gpu_loads(loads, 32, 0))
    red = np.asarray(ex.eplb_gpu_loads(loads, 32, 32))
    assert red.max() / red.mean() < no_red.max() / no_red.mean()


@pytest.mark.parametrize(
    "loads, n_gpus, n_redundant, kwargs",
    [
        (np.ones(10), 4, 1, {}),  # 11 个槽位不能均分到 4 卡
        (np.ones(8), 0, 0, {}),  # n_gpus 非正
        (np.ones(8), 4, -4, {}),  # 冗余数为负
        (np.array([1.0, -1.0, 1.0, 1.0]), 2, 0, {}),  # 负载为负
        (np.array([]), 1, 0, {}),  # 空数组
        (np.ones(12), 4, 0, {"n_groups": 5}),  # 专家数不能被组数整除
    ],
)
def test_b_invalid_inputs_raise(loads, n_gpus, n_redundant, kwargs):
    with pytest.raises(ValueError):
        ex.eplb_gpu_loads(loads, n_gpus, n_redundant, **kwargs)


# ================================================================== (c) mla_decode_roofline


def test_c_no_tp_no_mtp_is_borderline_memory_bound():
    r = ex.mla_decode_roofline(128, 1)
    assert r["intensity"] == pytest.approx(2 * 128 * 1088 / 1152)  # ≈ 241.8
    assert r["ridge"] == pytest.approx(865 / 3.35)  # ≈ 258.2
    assert r["bound"] == "memory"


def test_c_mtp_makes_it_compute_bound():
    r = ex.mla_decode_roofline(128, 2)
    assert r["intensity"] == pytest.approx(483.6, abs=0.1)
    assert r["bound"] == "compute"


def test_c_tp4_splits_heads_and_goes_memory_bound():
    # V3 报告的旧方案：decode 注意力 TP4 → 每卡 32 个头
    assert ex.mla_decode_roofline(32, 2)["bound"] == "memory"
    assert ex.mla_decode_roofline(32, 2)["intensity"] == pytest.approx(120.9, abs=0.1)


def test_c_fp8_kv_raises_intensity():
    # V3.2 的 FP8 稀疏 decode 格式：每条目 656 B
    r = ex.mla_decode_roofline(128, 1, kv_entry_bytes=656)
    assert r["intensity"] == pytest.approx(424.6, abs=0.1)
    assert r["bound"] == "compute"


# ================================================================== (d) day6_per_gpu


def test_d_default_day6_numbers():
    r = ex.day6_per_gpu()
    assert r["concurrent"] == pytest.approx(88.1, abs=0.05)
    assert r["kv_gb"] == pytest.approx(30.9, abs=0.05)
    assert r["tokens_per_replica_mb"] == pytest.approx(352.4, abs=0.1)
    assert r["step_ms"] == pytest.approx(36.98, abs=0.01)
    assert r["user_tok_s_bound"] == pytest.approx(50.0, abs=0.05)


def test_d_reproduces_isca_formula_per_micro_batch():
    # ISCA 的设定：每个 micro-batch 32 个 token、9 个目标、hidden 7000 → 单次通信 120.96 µs，61 层 × 2 → 14.76 ms
    # 让每卡每个 micro-batch 恰好 32 个 token：concurrent × s_q / micro_batches = 32
    r = ex.day6_per_gpu(node_decode_tok_s=8 * 32, user_tok_s=1.0, s_q=2, hidden=7000, n_targets=9)
    assert r["step_ms"] == pytest.approx(14.76, abs=0.01)


def test_d_alpha_adds_fixed_latency():
    base = ex.day6_per_gpu()["step_ms"]
    with_alpha = ex.day6_per_gpu(alpha_us=60)["step_ms"]
    # 61 层 × 2 个 micro-batch × (dispatch + combine) × 60 µs = 14.64 ms
    assert with_alpha - base == pytest.approx(61 * 2 * 2 * 60 / 1000)


def test_d_without_mtp_halves_expert_batch():
    r = ex.day6_per_gpu(s_q=1, tokens_per_step=1.0)
    assert r["tokens_per_replica_mb"] == pytest.approx(176.2, abs=0.1)
    assert r["concurrent"] == pytest.approx(88.1, abs=0.05)
