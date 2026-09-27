"""pd：P/D 分离与同卡数共置的对比、KV 传输与按层流水。"""

import numpy as np
import pytest

from codesign_sim.engine import EngineConfig, Request
from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS, kv_bytes_per_token
from codesign_sim.pd import simulate_colocated, simulate_pd

CFG = EngineConfig(GPUS["H100-SXM"], MODELS["llama3.1-70b"], n_gpus=8)


def workload(n: int = 200, rate: float = 3.0, seed: int = 0) -> list[Request]:
    rng = np.random.default_rng(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.exponential(1 / rate)
        reqs.append(Request(i, t, int(rng.integers(2000, 16000)), 0, int(rng.integers(100, 500))))
    return reqs


def test_pd_stabilizes_decode_at_equal_gpus():
    reqs = workload()
    co = simulate_colocated(reqs, CFG, n_instances=2)
    pd = simulate_pd(reqs, CFG, n_prefill=1, n_decode=1)
    assert co["n_gpus_total"] == pd["n_gpus_total"] == 16
    assert pd["n_requests"] == co["n_requests"] == len(reqs)
    assert pd["tpot_p99"] < 0.5 * co["tpot_p99"]  # decode 不再被长 prompt 打断
    assert pd["itl_p99"] < co["itl_p99"]
    assert pd["ttft_p50"] > co["ttft_p50"]  # 代价：只有一半的卡做 prefill，TTFT 排队


def test_kv_transfer_and_layerwise_pipelining():
    reqs = workload(40, rate=0.5)
    slow_link = dict(kv_link_gbps=2.0)  # 故意很慢的链路
    lw = simulate_pd(reqs, CFG, layerwise=True, **slow_link)
    bulk = simulate_pd(reqs, CFG, layerwise=False, **slow_link)
    mean_prompt = np.mean([r.prompt_tokens for r in reqs if r.output_tokens > 1])
    expect = mean_prompt * kv_bytes_per_token(CFG.model) / (2.0e9 * 0.8)
    assert lw["kv_transfer_s_mean"] == pytest.approx(expect, rel=1e-6)
    assert bulk["kv_transfer_exposed_s_mean"] == pytest.approx(bulk["kv_transfer_s_mean"])
    assert lw["kv_transfer_exposed_s_mean"] < bulk["kv_transfer_exposed_s_mean"]
    assert lw["tpot_p99"] < bulk["tpot_p99"]  # 暴露的传输时间落在第 1→2 个 token 之间


def test_mla_kv_is_cheap_to_move():
    assert kv_bytes_per_token(MODELS["deepseek-v3"]) * 4 < kv_bytes_per_token(MODELS["llama3.1-70b"])


def test_decode_side_kv_capacity_counts_transferred_prompt():
    # decode 实例按 prompt + 已生成 token 计 KV（与共置同口径）；容量受限时在 decode 侧排队，goodput 随之下降
    from dataclasses import replace

    reqs = workload()
    slo = dict(ttft_slo_s=2.0, tpot_slo_s=0.03)
    roomy = simulate_pd(reqs, CFG, n_prefill=1, n_decode=1, **slo)  # 默认 max_num_seqs=256，不需要绕开
    tight_cfg = replace(CFG, kv_capacity_tokens=40_000)
    tight = simulate_pd(reqs, CFG, tight_cfg, n_prefill=1, n_decode=1, **slo)
    assert tight["decode_kv_peak_frac"] <= 1.0 and roomy["decode_kv_peak_frac"] <= 1.0
    assert tight["decode_kv_peak_frac"] > 0.9  # 容量真的被用满、成了约束
    assert tight["goodput_req_s"] < 0.5 * roomy["goodput_req_s"]
    assert tight["tpot_p99"] > 5 * roomy["tpot_p99"]  # 排队体现在第 1→2 个 token 的间隔上
    inc = simulate_pd(reqs, CFG, replace(tight_cfg, kv_reserve="incremental"), **slo)
    assert inc["decode_kv_peak_frac"] <= 1.0 and inc["n_requests"] == len(reqs)
    co = simulate_colocated(reqs, CFG, n_instances=2, **slo)
    assert 0 < co["slo_attainment"] <= 1 and co["kv_peak_frac"] <= 1.0
