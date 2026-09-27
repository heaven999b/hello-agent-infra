"""engine：守恒、roofline 量级、chunked prefill 降低抖动、前缀命中、KV 共享与抢占、FP8、通信与 MLA 的 TP 复制。"""

import warnings

import numpy as np
import pytest

from codesign_sim.engine import Engine, EngineConfig, Request, simulate
from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS, body_active_params

L8, L70, DS = MODELS["llama3.1-8b"], MODELS["llama3.1-70b"], MODELS["deepseek-v3"]


def make_requests(n: int = 120, seed: int = 0, rate: float = 4.0) -> list[Request]:
    rng = np.random.default_rng(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.exponential(1 / rate)
        prompt = int(rng.integers(200, 6000))
        reqs.append(Request(i, t, prompt, int(rng.integers(0, prompt)), int(rng.integers(1, 400)), session_id=i % 7))
    return reqs


CFG = EngineConfig(GPUS["H100-SXM"], L70, n_gpus=8)


def test_conservation():
    reqs = make_requests()
    out = simulate(reqs, CFG)
    assert out["n_requests"] == len(reqs) and len(out["per_request"]) == len(reqs)
    assert out["output_tokens"] == sum(r.output_tokens for r in reqs)
    assert out["prefill_tokens_computed"] == sum(r.prompt_tokens - min(r.cached_tokens, r.prompt_tokens - 1) for r in reqs)
    for rec, req in zip(out["per_request"], sorted(reqs, key=lambda r: r.req_id)):
        assert rec["req_id"] == req.req_id and req.arrival_s < rec["first_token_s"] <= rec["finish_s"]
    assert 0 < out["busy_frac"] <= 1 and out["preemptions"] == 0
    assert out["ttft_p99"] >= out["ttft_p50"] > 0 and out["tpot_p99"] >= out["tpot_p50"] > 0


def test_deterministic():
    a, b = simulate(make_requests(), CFG), simulate(make_requests(), CFG)
    assert a["per_request"] == b["per_request"]


def test_single_request_matches_roofline():
    cfg = EngineConfig(GPUS["H100-SXM"], L8, n_gpus=1)
    out = simulate([Request(0, 0.0, 16, 0, 101)], cfg)
    # 每步读一遍权重（不含输入 embedding 表）：(8.03e9 - 0.525e9) · 2 B / (3.35 TB/s · 0.8) ≈ 5.6 ms
    assert out["tpot_p50"] == pytest.approx((8.03e9 - 128256 * 4096) * 2 / (3.35e12 * 0.8), rel=0.02)


def test_chunked_prefill_reduces_decode_jitter():
    reqs = [Request(i, 0.0, 500, 0, 600) for i in range(16)]
    reqs += [Request(100 + i, 0.5 + 0.8 * i, 12000, 0, 20) for i in range(8)]
    base = dict(gpu=GPUS["H100-SXM"], model=L70, n_gpus=8, max_batch_tokens=2048)
    chunked = simulate(reqs, EngineConfig(**base, chunked_prefill=True))
    plain = simulate(reqs, EngineConfig(**base, chunked_prefill=False))
    worst = lambda out: max(r["max_itl_s"] for r in out["per_request"] if r["req_id"] < 16)  # noqa: E731
    assert worst(chunked) < 0.5 * worst(plain)
    assert chunked["itl_max"] < plain["itl_max"]


def test_prefix_hits_reduce_ttft():
    reqs = [Request(i, i * 0.5, 16000, 0, 50) for i in range(20)]
    hit = [Request(i, i * 0.5, 16000, 15000, 50) for i in range(20)]
    assert simulate(hit, CFG)["ttft_p50"] < 0.3 * simulate(reqs, CFG)["ttft_p50"]


def test_load_delay_from_dram():
    fast = simulate([Request(0, 0.0, 8000, 7000, 10)], CFG)
    slow = simulate([Request(0, 0.0, 8000, 7000, 10, load_s=0.5)], CFG)
    assert slow["ttft_p50"] == pytest.approx(fast["ttft_p50"] + 0.5)


def test_cached_prefix_kv_is_shared():
    # 10 个请求共享 9000 token 的已缓存前缀：KV 只按各自的私有部分计，能同时跑
    cfg = EngineConfig(GPUS["H100-SXM"], L8, n_gpus=1, kv_capacity_tokens=20000)
    reqs = [Request(i, 0.0, 10000, 9000, 200) for i in range(10)]
    eng = Engine(cfg)
    for r in reqs:
        eng.submit(r)
    eng.step()
    assert len(eng.prefilling) + eng.n_dec == 10  # 若不共享，每个要 10200，只能放下 1 个


def test_oracle_reservation_limits_concurrency():
    cfg = EngineConfig(GPUS["H100-SXM"], L8, n_gpus=1, kv_capacity_tokens=5000)
    reqs = [Request(i, 0.0, 1500, 0, 100) for i in range(10)] + [Request(99, 0.0, 9000, 0, 10)]
    eng = Engine(cfg)
    for r in reqs:
        eng.submit(r)
    while eng.has_work():
        eng.step()
        assert eng.kv_used <= cfg.kv_capacity_tokens or len(eng.prefilling) + eng.n_dec == 1
    assert len(eng.finished) == len(reqs) and eng.preemptions == 0


def test_incremental_reservation_preempts_long_outputs():
    # RL 式长输出：按 prompt 准入、decode 中途显存不够 -> 抢占最后准入的序列并重算
    reqs = [Request(i, 0.0, 1000, 0, 3000) for i in range(8)]
    base = dict(gpu=GPUS["H100-SXM"], model=L8, n_gpus=1, kv_capacity_tokens=16000)
    inc = simulate(reqs, EngineConfig(**base, kv_reserve="incremental"))
    ora = simulate(reqs, EngineConfig(**base, kv_reserve="oracle"))
    assert inc["preemptions"] > 0 and inc["recomputed_tokens"] > 0 and ora["preemptions"] == 0
    assert inc["output_tokens"] == ora["output_tokens"] == 8 * 3000  # token 守恒
    assert inc["prefill_tokens_computed"] > ora["prefill_tokens_computed"]  # 重算的代价
    assert all(r["preemptions"] >= 0 for r in inc["per_request"])


def test_fp8_gemm_speeds_up_compute_bound_prefill():
    # H20 上 FP8 权重 + FP8 GEMM：compute-bound 的 prefill 明显变快（注意力核心仍按 BF16）
    reqs = [Request(0, 0.0, 8192, 0, 1)]
    base = dict(gpu=GPUS["H20"], model=DS, n_gpus=8, weight_bytes=1)
    fp8 = simulate(reqs, EngineConfig(**base))  # gemm_dtype=None 跟随 weight_bytes=1 -> fp8
    bf16 = simulate(reqs, EngineConfig(**base, gemm_dtype="bf16"))
    assert EngineConfig(**base).resolved_gemm_dtype() == "fp8"
    assert 0.5 < fp8["ttft_p50"] / bf16["ttft_p50"] < 0.75


def test_tp_communication_and_mla_replication():
    reqs = [Request(i, 0.0, 2000, 0, 200) for i in range(32)]
    h100 = simulate(reqs, EngineConfig(GPUS["H100-SXM"], L70, n_gpus=8, tp=8))
    h800 = simulate(reqs, EngineConfig(GPUS["H800-SXM"], L70, n_gpus=8, tp=8))
    ideal = simulate(reqs, EngineConfig(GPUS["H800-SXM"], L70, n_gpus=8))
    assert h800["tpot_p50"] > h100["tpot_p50"] > ideal["tpot_p50"]  # 通信让 H800 与 H100 有了差别
    # MLA：TP8 下 latent KV 每卡完整读取；DP-attention 避免复制
    base = dict(gpu=GPUS["B200"], model=DS, n_gpus=8, weight_bytes=1)
    long_ctx = [Request(i, 0.0, 30000, 29999, 100) for i in range(64)]  # 纯 decode、长上下文
    tp = simulate(long_ctx, EngineConfig(**base, tp=8))
    dp = simulate(long_ctx, EngineConfig(**base, tp=8, dp_attention=True))
    assert tp["tpot_p50"] > 1.5 * dp["tpot_p50"]


def test_body_params_used_in_cost():
    assert body_active_params(L8) < 8.03e9


def test_model_must_fit_and_context_warning():
    with pytest.raises(ValueError):
        simulate([Request(0, 0, 10, 0, 10)], EngineConfig(GPUS["H100-SXM"], DS, n_gpus=8))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simulate([Request(0, 0, 140000, 0, 10)], EngineConfig(GPUS["B200"], L8, n_gpus=1))
    assert any("max_context" in str(x.message) for x in w)
