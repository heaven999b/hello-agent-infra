"""router：闭环多轮负载下的命中率、热点代价与负载上限、写缓存时机、非 oracle 提示、两层缓存。"""

import pytest

from codesign_sim.engine import EngineConfig
from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS
from codesign_sim.router import route_and_simulate
from codesign_sim.workload import generate_sessions

CFG = EngineConfig(GPUS["H100-SXM"], MODELS["llama3.1-8b"], n_gpus=1)
SESSIONS = generate_sessions(30, 6, seed=0, arrival_rate=1.0)
N_TURNS = sum(len(s.turns) for s in SESSIONS)
# 热点场景：只有 1 个 project、到达率高，所有 session 共享同一段前缀
HOT = generate_sessions(100, 1, seed=0, arrival_rate=3.0, mean_turns=6)


@pytest.fixture(scope="module")
def results():
    pols = ["round_robin", "least_loaded", "cache_affinity", "affinity_with_load_cap", "session_sticky"]
    return {p: route_and_simulate(SESSIONS, 4, p, CFG, cache_capacity_blocks=8000) for p in pols}


@pytest.fixture(scope="module")
def hot():
    runs = [("cache_affinity", "prefill_done"), ("affinity_with_load_cap", "prefill_done"), ("cache_affinity", "finish")]
    return {(p, ins): route_and_simulate(HOT, 4, p, CFG, 8000, insert_on=ins) for p, ins in runs}


def test_all_requests_complete(results):
    for out in results.values():
        assert out["n_requests"] == N_TURNS == sum(out["requests_per_instance"])
        assert out["prefill_tokens_computed"] <= out["prefill_tokens_total"]
        assert out["load_imbalance"] >= 1.0


def test_cache_affinity_beats_round_robin(results):
    rr, aff = results["round_robin"], results["cache_affinity"]
    assert aff["hit_rate"] > rr["hit_rate"] + 0.1
    assert aff["prefill_tokens_computed"] < rr["prefill_tokens_computed"]
    assert aff["ttft_p50"] < rr["ttft_p50"]


def test_session_sticky_baseline(results):
    st = results["session_sticky"]
    inst = {}
    for r in st["per_request"]:  # 同一会话的所有请求都落在同一实例
        assert inst.setdefault(r["session_id"], r["instance"]) == r["instance"]
    assert st["migrations"] == 0
    assert st["hit_rate"] == pytest.approx(results["cache_affinity"]["hit_rate"], abs=0.05)  # 默认负载下与亲和接近
    moved = route_and_simulate(HOT, 4, "session_sticky", CFG, 8000, sticky_migrate=True, load_metric="requests",
                               balance_abs_threshold=2)
    assert moved["migrations"] > 0  # 允许迁移时，失衡的实例上的会话会被挪走


def test_hotspot_cost_and_load_cap(hot):
    pure, cap = hot[("cache_affinity", "prefill_done")], hot[("affinity_with_load_cap", "prefill_done")]
    assert pure["max_instance_share"] > 0.95  # 纯亲和：所有请求被吸到一个实例，其余实例饿死
    assert cap["max_instance_share"] < 0.8
    assert cap["load_imbalance"] < pure["load_imbalance"]
    assert cap["ttft_p99"] < 0.7 * pure["ttft_p99"]  # 负载上限把尾延迟救回来
    assert cap["hit_rate"] > 0.8  # 代价：命中率只损失一点


def test_insert_on_finish_masks_the_hotspot(hot):
    # 只在请求完成时写缓存：早期同 project 的 session 在路由时打平、被分散，热点形成不起来
    assert hot[("cache_affinity", "finish")]["max_instance_share"] < hot[("cache_affinity", "prefill_done")]["max_instance_share"] - 0.3


def test_hint_modes_and_default_is_not_oracle():
    kw = dict(cache_policy="session_aware", cache_capacity_blocks=1500)
    sess = SESSIONS[:25]
    default = route_and_simulate(sess, 2, "cache_affinity", CFG, **kw)
    noisy = route_and_simulate(sess, 2, "cache_affinity", CFG, hint_mode="noisy", **kw)
    oracle = route_and_simulate(sess, 2, "cache_affinity", CFG, hint_mode="oracle", **kw)
    fixed = route_and_simulate(sess, 2, "cache_affinity", CFG, hint_mode="fixed", hint_ttl_s=5.0, **kw)
    assert default["per_request"] == noisy["per_request"]  # 默认是带噪声的估计，不是 oracle
    assert oracle["per_request"] != noisy["per_request"] and fixed["per_request"] != noisy["per_request"]


def test_dram_tier_loads_instead_of_recompute():
    kw = dict(cache_capacity_blocks=1500)
    hbm_only = route_and_simulate(SESSIONS[:25], 2, "cache_affinity", CFG, **kw)
    tiered = route_and_simulate(SESSIONS[:25], 2, "cache_affinity", CFG, cache_dram_blocks=40000, **kw)
    assert tiered["dram_hit_rate"] > 0 and tiered["dram_loaded_blocks"] > 0
    assert tiered["prefill_tokens_computed"] < hbm_only["prefill_tokens_computed"]
    # PCIe 慢到离谱时，搬运不如重算：DRAM 命中被当作未命中
    slow = route_and_simulate(SESSIONS[:25], 2, "cache_affinity", CFG, cache_dram_blocks=40000, dram_gbps=0.01, **kw)
    assert slow["dram_loaded_blocks"] == 0 and slow["dram_recomputed_blocks"] > 0


def test_deterministic_and_validation():
    a = route_and_simulate(SESSIONS[:20], 2, "least_loaded", CFG, 4000, seed=1)
    b = route_and_simulate(SESSIONS[:20], 2, "least_loaded", CFG, 4000, seed=1)
    assert a["per_request"] == b["per_request"]
    for policy, kw in [("random", {}), ("affinity_with_load_cap", dict(load_cap_factor=0.0)),
                       ("least_loaded", dict(insert_on="never")), ("least_loaded", dict(hint_mode="magic"))]:
        with pytest.raises(ValueError):
            route_and_simulate(SESSIONS[:2], 2, policy, CFG, 100, **kw)
