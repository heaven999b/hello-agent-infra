"""模块 01 练习测试。离线、确定性：四个函数都是纯函数，只调用 codesign_sim 的公开接口。

运行：make module N=01
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/01_inference_basics
"""

from __future__ import annotations

import pytest

from codesign_sim.hardware import GPUS, ridge_point
from codesign_sim.model import (
    MODELS,
    decode_bytes,
    decode_flops,
    kv_bytes_per_token,
    prefill_flops,
    step_time_s,
)
from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)

H100 = GPUS["H100-SXM"]
L8 = MODELS["llama3-8b"]
L70 = MODELS["llama3-70b"]
DSV3 = MODELS["deepseek-v3"]


def intensity(cfg, batch: int, ctx: int, weight_bytes: float = 2, kv_dtype_bytes: int = 2) -> float:
    return decode_flops(cfg, batch, ctx) / decode_bytes(cfg, batch, ctx, weight_bytes, kv_dtype_bytes)


# ================================================================== (a) max_concurrency


def test_a_readme_examples():
    assert ex.max_concurrency(H100, L8, 8192) == 52
    assert ex.max_concurrency(H100, L70, 32768, n_gpus=8) == 40


def test_a_matches_the_documented_formula():
    for cfg, n_gpus, ctx in [(L8, 1, 4096), (L70, 8, 8192), (MODELS["qwen3-32b"], 2, 16384)]:
        usable = n_gpus * H100.hbm_gb * 1e9 * 0.9
        weights = cfg.total_params_b * 1e9 * 2
        expected = int((usable - weights) // (ctx * kv_bytes_per_token(cfg)))
        assert ex.max_concurrency(H100, cfg, ctx, n_gpus=n_gpus) == expected


def test_a_weights_do_not_fit_returns_zero():
    # DeepSeek-V3 的 BF16 权重约 1.34 TB，一台 8 × H100 放不下
    assert ex.max_concurrency(H100, DSV3, 4096, n_gpus=8) == 0


def test_a_fp8_kv_roughly_doubles_concurrency():
    bf16 = ex.max_concurrency(H100, L70, 32768, n_gpus=8, kv_dtype_bytes=2)
    fp8 = ex.max_concurrency(H100, L70, 32768, n_gpus=8, kv_dtype_bytes=1)
    assert fp8 in (2 * bf16, 2 * bf16 + 1)


def test_a_fp8_weights_free_memory_for_kv():
    assert ex.max_concurrency(H100, L70, 32768, n_gpus=8, weight_bytes=1) > ex.max_concurrency(H100, L70, 32768, n_gpus=8)


def test_a_longer_context_means_fewer_sequences():
    values = [ex.max_concurrency(H100, L8, ctx) for ctx in (2048, 8192, 32768)]
    assert values[0] > values[1] > values[2]


def test_a_rejects_bad_context():
    with pytest.raises(ValueError):
        ex.max_concurrency(H100, L8, 0)


# ================================================================== (b) critical_batch


def test_b_short_context_becomes_compute_bound():
    b = ex.critical_batch(H100, L8, 128)
    assert b is not None
    ridge = ridge_point(H100)
    assert intensity(L8, b, 128) >= ridge
    assert intensity(L8, b - 1, 128) < ridge


def test_b_gqa_at_agent_length_context_never_reaches_ridge():
    # README 第 5 节：Llama-3-8B 在 4k 上下文时 intensity 上限约 32，远低于 H100 的 ridge（约 295）
    assert ex.critical_batch(H100, L8, 4096) is None
    assert ex.critical_batch(H100, L70, 32768) is None


def test_b_mla_reaches_ridge_where_gqa_cannot():
    # 同样 4k 上下文：GQA 的 Llama-3-8B 永远达不到 ridge；MLA 的 DeepSeek-V3 在足够大的 batch 下可以
    # （它的权重太大，所以需要的 batch 远超 4096，这也是它要靠大规模 EP 聚合 batch 的原因之一）
    assert ex.critical_batch(H100, L8, 4096, max_batch=16384) is None
    b = ex.critical_batch(H100, DSV3, 4096, max_batch=16384)
    assert b is not None and b > 4096
    assert intensity(DSV3, b, 4096) >= ridge_point(H100)
    assert intensity(DSV3, b - 1, 4096) < ridge_point(H100)


def test_b_fp8_kv_is_passed_through():
    # README 第 11 节：512 token 上下文时，BF16 KV 的上限低于 ridge，FP8 KV 把上限翻倍才越过
    assert ex.critical_batch(H100, L8, 512, max_batch=100000) is None
    b = ex.critical_batch(H100, L8, 512, max_batch=100000, kv_dtype_bytes=1)
    assert b == 826
    assert intensity(L8, b, 512, kv_dtype_bytes=1) >= ridge_point(H100)
    assert intensity(L8, b - 1, 512, kv_dtype_bytes=1) < ridge_point(H100)


def test_b_fp8_weights_are_passed_through():
    # 权重字节减半，线性层部分的 intensity 翻倍，临界 batch 约减半
    bf16 = ex.critical_batch(H100, L8, 128)
    fp8 = ex.critical_batch(H100, L8, 128, weight_bytes=1)
    assert fp8 is not None and fp8 < bf16
    assert intensity(L8, fp8, 128, weight_bytes=1) >= ridge_point(H100)
    assert intensity(L8, fp8 - 1, 128, weight_bytes=1) < ridge_point(H100)


def test_b_lower_ridge_gpu_needs_smaller_batch():
    h20 = GPUS["H20"]  # ridge ≈ 37
    assert ex.critical_batch(h20, L8, 128) < ex.critical_batch(H100, L8, 128)


def test_b_respects_max_batch():
    b = ex.critical_batch(H100, L8, 128)
    assert ex.critical_batch(H100, L8, 128, max_batch=b - 1) is None
    assert ex.critical_batch(H100, L8, 128, max_batch=b) == b


# ================================================================== (c) estimate_ttft


def expected_ttft(cfg, n_gpus: int, prompt: int, cached: int, queue: float = 0.0) -> float:
    n_new = max(prompt - cached, 1)
    flops = prefill_flops(cfg, n_new, prompt - n_new)
    bytes_ = cfg.total_params_b * 1e9 * 2 + prompt * kv_bytes_per_token(cfg)
    return queue + step_time_s(H100, cfg, flops, bytes_, n_gpus, 0.6, 0.8)


def test_c_readme_example():
    assert ex.estimate_ttft(H100, L8, 4096) == pytest.approx(0.10, abs=0.01)


def test_c_matches_the_documented_formula():
    for cfg, n_gpus, prompt, cached in [(L8, 1, 4096, 0), (L70, 8, 65536, 32768), (L70, 8, 8192, 1024)]:
        got = ex.estimate_ttft(H100, cfg, prompt, cached, n_gpus=n_gpus, queue_s=0.25)
        assert got == pytest.approx(expected_ttft(cfg, n_gpus, prompt, cached, 0.25), rel=1e-9)


def test_c_cache_hits_cut_ttft():
    cold = ex.estimate_ttft(H100, L70, 65536, 0, n_gpus=8)
    half = ex.estimate_ttft(H100, L70, 65536, 32768, n_gpus=8)
    warm = ex.estimate_ttft(H100, L70, 65536, 65536 - 512, n_gpus=8)
    assert cold > half > warm
    # 后一半 token 看到的上下文更长，所以命中一半省下的不到一半
    assert half > cold / 2


def test_c_small_incremental_prefill_is_memory_bound():
    # 只算 128 个新 token：计算约 3.5 ms，读权重 + 4k 的 KV 约 6.2 ms，由带宽决定
    got = ex.estimate_ttft(H100, L8, 4096, 4096 - 128)
    memory_s = (L8.total_params_b * 1e9 * 2 + 4096 * kv_bytes_per_token(L8)) / (H100.hbm_tbps * 1e12 * 0.8)
    assert got == pytest.approx(memory_s, rel=1e-9)


def test_c_full_hit_still_computes_one_token():
    got = ex.estimate_ttft(H100, L8, 4096, 4096)
    assert got == pytest.approx(expected_ttft(L8, 1, 4096, 4096), rel=1e-9)
    assert got > 0


def test_c_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.estimate_ttft(H100, L8, 100, 200)
    with pytest.raises(ValueError):
        ex.estimate_ttft(H100, L8, 100, -1)
    with pytest.raises(ValueError):
        ex.estimate_ttft(H100, L8, 0, 0)


# ================================================================== (d) tool_wait_kv_action


def test_d_short_tool_wait_keeps_kv():
    assert ex.tool_wait_kv_action(H100, L8, 4096, tool_s=5) == ("keep", 0.0)
    assert ex.tool_wait_kv_action(H100, L70, 65536, tool_s=30, n_gpus=8) == ("keep", 0.0)


def test_d_break_even_tool_wait():
    # keep 的门槛 = recompute_s / share；70B、64k、8 × H100 约 62 s
    recompute = ex.estimate_ttft(H100, L70, 65536, 0, n_gpus=8)
    share = 65536 * kv_bytes_per_token(L70) / (8 * H100.hbm_gb * 1e9 * 0.9 - L70.total_params_b * 1e9 * 2)
    threshold = recompute / share
    assert 60 < threshold < 67
    assert ex.tool_wait_kv_action(H100, L70, 65536, threshold * 0.99, n_gpus=8)[0] == "keep"
    assert ex.tool_wait_kv_action(H100, L70, 65536, threshold * 1.01, n_gpus=8)[0] == "recompute"


def test_d_long_wait_offloads_when_a_fast_tier_exists():
    action, extra = ex.tool_wait_kv_action(H100, L70, 65536, tool_s=300, n_gpus=8, offload_gbps=512)
    assert action == "offload"
    assert extra == pytest.approx(65536 * kv_bytes_per_token(L70) / 512e9, rel=1e-9)


def test_d_long_wait_without_offload_tier_recomputes():
    action, extra = ex.tool_wait_kv_action(H100, L70, 65536, tool_s=300, n_gpus=8)
    assert action == "recompute"
    assert extra == pytest.approx(ex.estimate_ttft(H100, L70, 65536, 0, n_gpus=8), rel=1e-9)


def test_d_slow_tier_loses_to_recompute_for_short_context():
    # 8B、4k：重算只要约 0.10 s；1 GB/s 的慢存储搬回 0.54 GB 要 0.54 s，不如重算；64 GB/s 则该卸载
    assert ex.tool_wait_kv_action(H100, L8, 4096, tool_s=60, offload_gbps=1.0)[0] == "recompute"
    assert ex.tool_wait_kv_action(H100, L8, 4096, tool_s=60, offload_gbps=64)[0] == "offload"


def test_d_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.tool_wait_kv_action(H100, DSV3, 4096, tool_s=10)  # 权重放不下
    with pytest.raises(ValueError):
        ex.tool_wait_kv_action(H100, L8, 4096, tool_s=-1)
    with pytest.raises(ValueError):
        ex.tool_wait_kv_action(H100, L8, 0, tool_s=1)
    with pytest.raises(ValueError):
        ex.tool_wait_kv_action(H100, L8, 4096, tool_s=1, offload_gbps=-5)
