"""rollout：等资源比较、AReaL 可中断续写 vs 中止、限流性质、长运行稳定、partial 固定 batch 与边界、token 加权 staleness。"""

from dataclasses import replace

import pytest

from codesign_sim.rollout import RolloutConfig, compare_rollout_modes, simulate_rollout, simulate_rollout_seeds

KEYS = ["worker_busy_frac", "traj_per_hour", "mean_staleness", "p99_staleness", "step_time_s"]
SMALL = RolloutConfig(n_workers=16, batch_size=64)


@pytest.fixture(scope="module")
def runs():
    return {m: simulate_rollout(RolloutConfig(mode=m), n_steps=30) for m in ["sync", "partial", "async"]}


def test_keys_and_ranges(runs):
    for out in runs.values():
        assert all(k in out for k in KEYS)
        assert 0 < out["worker_busy_frac"] <= out["worker_occupied_frac"] <= 1 + 1e-9
        assert 0 < out["cluster_util"] <= 1 + 1e-9
        assert out["mean_batch"] == 256  # 三种模式的训练 batch 都是 batch_size


def test_sync_long_tail_idles_workers(runs):
    s = runs["sync"]
    assert s["staleness_max"] == 0 and s["worker_busy_frac"] < 0.35
    assert s["gate_idle_frac"] > 0.5  # 等最慢的那条：大部分槽位空转
    light = simulate_rollout(RolloutConfig(mode="sync", tail_alpha=4.0), n_steps=30)
    assert light["worker_busy_frac"] > s["worker_busy_frac"]


def test_equal_resource_decomposition():
    cmp = compare_rollout_modes(n_steps=30)
    assert cmp["tail_gain"] > 2.0  # 主要收益来自"不等长尾"
    assert 0.8 < cmp["overlap_gain"] < 1.25  # 等 GPU 总数时，训练与生成重叠几乎不带来额外吞吐
    assert cmp["runs"]["async_disaggregated"]["n_slots"] < cmp["runs"]["async_colocated"]["n_slots"]


def test_interrupt_keeps_long_trajectories_abort_drops_them():
    it = simulate_rollout(RolloutConfig(mode="async"), n_steps=30)
    ab = simulate_rollout(RolloutConfig(mode="async", stale_policy="abort"), n_steps=30)
    assert it["wasted_frac"] == 0 and it["dropped_trajectories"] == 0 and it["interrupts"] > 0
    assert it["recompute_frac"] > 0  # 打断后要用新权重重算 KV
    assert ab["wasted_frac"] > 0 and ab["staleness_max"] <= 1  # 中止换来了硬上界……
    assert it["length_ratio"] > ab["length_ratio"] + 0.08  # ……代价是训练数据偏向短轨迹
    assert it["staleness_max"] > 1  # 可中断续写：长轨迹跨越多个版本（由多个版本的片段组成）
    retain = simulate_rollout(RolloutConfig(mode="async", resume_kv="retain"), n_steps=30)
    assert retain["recompute_frac"] == 0


def test_rate_limit_property():
    for eta in (0, 1, 3):
        out = simulate_rollout(replace(SMALL, mode="async", deploy="disaggregated", max_staleness=eta), n_steps=20)
        assert out["max_gate_lead"] <= eta
    a0 = simulate_rollout(replace(SMALL, mode="async", max_staleness=0), n_steps=10, seed=3)
    s0 = simulate_rollout(replace(SMALL, mode="sync"), n_steps=10, seed=3)
    assert a0["traj_per_hour"] == pytest.approx(s0["traj_per_hour"])  # η=0 退化为 sync


def test_long_runs_do_not_collapse():
    for pol in ("interrupt", "abort"):
        cfg = replace(SMALL, mode="async", stale_policy=pol, deploy="disaggregated")
        short = simulate_rollout(cfg, n_steps=30)["train_tokens_per_hour"]
        long = simulate_rollout(cfg, n_steps=240)["train_tokens_per_hour"]
        assert long == pytest.approx(short, rel=0.1)


def test_rate_limit_starves_workers_when_workers_exceed_window():
    # n_workers ≥ (η+1)·B：限流让大部分槽位闲着；调大 η 才能用上
    many = replace(SMALL, mode="async", deploy="disaggregated", n_workers=256, batch_size=64)
    tight = simulate_rollout(replace(many, max_staleness=1), n_steps=20)
    loose = simulate_rollout(replace(many, max_staleness=4), n_steps=20)
    few = simulate_rollout(replace(SMALL, mode="async", deploy="disaggregated"), n_steps=20)
    assert tight["gate_idle_frac"] > 0.4 > few["gate_idle_frac"]
    assert loose["gate_idle_frac"] < tight["gate_idle_frac"]
    assert loose["train_tokens_per_hour"] > tight["train_tokens_per_hour"]


def test_partial_fixed_batch_and_token_staleness(runs):
    p = runs["partial"]
    assert p["worker_busy_frac"] > runs["sync"]["worker_busy_frac"]
    assert p["token_staleness_mean"] > p["mean_staleness"] > 0  # 长轨迹跨越更多版本，按 token 加权更陈旧
    assert p["token_staleness_max"] == p["staleness_max"] > 1


def test_partial_tiny_budget_boundary():
    cfg = RolloutConfig(mode="partial", n_workers=4, batch_size=16, mean_gen_s=50.0, partial_budget_s=0.5)
    out = simulate_rollout(cfg, n_steps=10)
    assert out["mean_batch"] == 16 and out["step_time_s"] >= cfg.train_step_s
    for k in ("worker_busy_frac", "worker_occupied_frac", "recompute_frac", "total_time_s"):
        assert out[k] >= 0
    assert out["worker_occupied_frac"] <= 1 + 1e-9
    tok = simulate_rollout(replace(cfg, partial_budget_tokens=200), n_steps=10)
    assert tok["mean_batch"] == 16


def test_no_extra_batch_on_last_step():
    cfg = replace(SMALL, mode="async", deploy="disaggregated", train_step_s=90.0, max_staleness=2)
    assert simulate_rollout(cfg, n_steps=12)["trained_trajectories"] == 12 * 64


def test_multi_seed_helper():
    out = simulate_rollout_seeds(replace(SMALL, mode="partial"), n_steps=10, seeds=(0, 1, 2))
    assert len(out["runs"]) == 3 and out["std"]["traj_per_hour"] > 0
    assert out["mean"]["traj_per_hour"] == pytest.approx(sum(r["traj_per_hour"] for r in out["runs"]) / 3)


def test_bad_config():
    for bad in (dict(mode="magic"), dict(deploy="cloud"), dict(stale_policy="drop"), dict(resume_kv="x")):
        with pytest.raises(ValueError):
            simulate_rollout(replace(SMALL, **bad), n_steps=2)
