"""模块 05 实验：路由策略、淘汰策略、加载 vs 重算、命中率对 TTFT 的影响。

跑法：
    .venv/bin/python modules/05_cache_prefill/lab.py

五个部分：
A. 多实例路由：按请求轮询（稻草人）/ least_loaded / session_sticky（会话粘性）/ cache_affinity / affinity_with_load_cap，
   看命中率、负载不均衡、TTFT、JCT，把 cache-aware 路由的收益拆成「会话粘性」和「前缀感知」两部分；
   A2 给一组热点参数，看纯亲和与带负载上限的差别。
B. 单实例淘汰（重放、只看命中率）：lru / 会话软保护（只知道「会回来」）/ 点估计 pin（精确 = 上界、带噪声、带上限）。
C. Continuum 的 τ*：从负载估 Benefit（重算时间 + 排队延迟 × 记忆性 η），按工具耗时的经验分布算最优 TTL（README 2.3）。
D. 端到端（闭环、有排队）：同一批会话，不同的淘汰策略与非 oracle 的 pin 提示，比 TTFT 和任务完成时间（JCT）。
E. 加载 vs 重算：每个模型的「盈亏平衡带宽」B*（BF16 / FP8 GEMM），和 PCIe / 分布式 SSD 的单卡带宽比一比（README 4.2）。
F. 命中率 × chunked prefill：同一批长 prompt 请求，命中率 0% / 50% / 90% 时的 TTFT 与 TPOT（README 5.1–5.2）。
"""

from __future__ import annotations

import math
import random

import numpy as np

from codesign_sim.cache import PrefixCache
from codesign_sim.engine import EngineConfig, Request, simulate
from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS, kv_bytes_per_token, prefill_flops_split, step_time_s
from codesign_sim.router import route_and_simulate
from codesign_sim.workload import AgentSession, context_tokens, generate_sessions, to_blocks

SEED = 0
LINE = "=" * 104
QWEN_4xH100 = dict(gpu=GPUS["H100-SXM"], model=MODELS["qwen3-32b"], n_gpus=4)


def jct_stats(res: dict, sessions: list[AgentSession]) -> tuple[float, float]:
    """每个会话的任务完成时间 = 最后一轮完成时刻 − 会话到达时刻；返回 (均值, p90)。"""
    last: dict[int, float] = {}
    for r in res["per_request"]:
        last[r["session_id"]] = max(last.get(r["session_id"], 0.0), r["finish_s"])
    jct = np.array([last[s.session_id] - s.arrival_s for s in sessions])
    return float(jct.mean()), float(np.percentile(jct, 90))


# ---------------------------------------------------------------------------
# A. 多实例路由
# ---------------------------------------------------------------------------


POLICIES_A = ("round_robin", "least_loaded", "session_sticky", "cache_affinity", "affinity_with_load_cap")


def print_routing(sessions: list[AgentSession], cfg: EngineConfig, cap: int, policies: tuple[str, ...]) -> None:
    print(f"   {'策略':<24}{'命中率':>8}{'负载不均衡':>11}{'最热实例占比':>13}{'TTFT p50':>10}{'TTFT p99':>10}{'JCT 均值':>10}")
    for policy in policies:
        r = route_and_simulate(sessions, 4, policy, cfg, cache_capacity_blocks=cap, seed=SEED)
        jct_mean, _ = jct_stats(r, sessions)
        label = policy + ("（稻草人）" if policy == "round_robin" else "")
        print(
            f"   {label:<24}{r['token_hit_rate']:>8.1%}{r['load_imbalance']:>11.2f}{r['max_instance_share']:>13.1%}"
            f"{r['ttft_p50']:>9.2f}s{r['ttft_p99']:>9.2f}s{jct_mean:>9.1f}s"
        )


def part_a() -> None:
    cfg = EngineConfig(**QWEN_4xH100)
    sessions = generate_sessions(120, 4, seed=SEED, arrival_rate=6.0)
    work = sum(context_tokens(s, len(s.turns) - 1, include_output=True) for s in sessions)
    print(LINE)
    print("A1. 路由策略：4 个实例（每个 4×H100 跑 Qwen3-32B），120 个 agent 会话 / 4 个项目（热度 Zipf s=1），闭环")
    print(f"    全部会话最终上下文之和约 {work / 1e6:.1f}M token；缓存容量按比例缩小（真实 4 卡实例约能放 77 万 token）")
    print("    round_robin 按请求轮询，同一会话的相邻轮次被打散，是个稻草人；真正的对手是 session_sticky（A2 里有 least_loaded）")
    for cap in (5_000, 20_000):
        print("-" * 104)
        print(f"    每实例前缀缓存 {cap:,} 个 block（×16 token = {cap * 16 / 1e3:.0f}k token）")
        print_routing(sessions, cfg, cap, ("round_robin", "session_sticky", "cache_affinity", "affinity_with_load_cap"))
    hot = generate_sessions(150, 4, seed=SEED, arrival_rate=10.0, project_zipf_s=3.0)
    print(LINE)
    print("A2. 热点：4 个实例（每个只有 2×H100），150 个会话 / 4 个项目，热度极度集中（Zipf s=3），更高到达率；缓存 20,000 block")
    print("-" * 104)
    print_routing(hot, EngineConfig(gpu=GPUS["H100-SXM"], model=MODELS["qwen3-32b"], n_gpus=2), 20_000, POLICIES_A[1:])


# ---------------------------------------------------------------------------
# B. 单实例淘汰：重放，只看命中率
# ---------------------------------------------------------------------------

BLOCK_B = 64
DECODE_S_PER_TOKEN = 0.02  # 假设：单会话 decode 约 50 token/s，只用来排时间顺序
FAR = 1e9  # 「只知道会回来」：pin_until = FAR − 完成时刻，让暂停中的会话之间仍按 LRU 排


def build_events(sessions: list[AgentSession]) -> list[tuple[float, int, int, int]]:
    """(时刻, 类型, session 下标, 轮次)；类型 0 = 本轮算完（写缓存），1 = 本轮请求到达（查缓存）。"""
    events = []
    for i, s in enumerate(sessions):
        t = s.arrival_s
        for k, turn in enumerate(s.turns):
            done = t + turn.output_tokens * DECODE_S_PER_TOKEN
            events += [(t, 1, i, k), (done, 0, i, k)]
            t = done + turn.tool_latency_s
    events.sort()
    return events


def replay_eviction(sessions, events, blocks, capacity: int, policy: str, hint: str, noise: float = 0.0,
                    cap_s: float = math.inf) -> tuple[float, int]:
    """返回 (token 命中率, 被拒绝插入的 block 数)。

    hint：本轮算完时给缓存的 pin_until 提示（最后一轮不给，harness 知道会话结束了）
      "none"   不给提示
      "alive"  只说「会回来」，不说何时（会话软保护，类似 SGLang session radix cache）
      "eta"    完成时刻 + min(工具耗时 × exp(N(0, noise)), cap_s)；noise=0 且 cap=inf 即 oracle 上界
    """
    rng = random.Random(SEED)
    cache = PrefixCache(capacity, policy=policy)
    prompt_total = hit_total = 0
    for now, kind, i, k in events:
        s = sessions[i]
        if kind == 1:
            n = context_tokens(s, k)
            hit = cache.match(blocks[(i, k)][: n // BLOCK_B], now)
            prompt_total += n
            hit_total += hit * BLOCK_B
            continue
        pin = None
        if k < len(s.turns) - 1 and hint != "none":
            if hint == "alive":
                pin = FAR - now
            else:
                eta = s.turns[k].tool_latency_s * (math.exp(rng.gauss(0.0, noise)) if noise > 0 else 1.0)
                pin = now + min(eta, cap_s)
        cache.insert(blocks[(i, k)], now, pin_until=pin)
    return hit_total / prompt_total, cache.stats["rejected_blocks"]


def tool_latencies(sessions: list[AgentSession]) -> np.ndarray:
    return np.array([t.tool_latency_s for s in sessions for t in s.turns[:-1]])


def part_b() -> None:
    sessions = generate_sessions(120, 8, seed=SEED, arrival_rate=2.0)
    events = build_events(sessions)
    blocks = {(i, k): to_blocks(s, k, BLOCK_B, include_output=True) for i, s in enumerate(sessions) for k in range(len(s.turns))}
    p90 = float(np.percentile(tool_latencies(sessions), 90))
    variants = [
        ("lru", "lru", "none", 0.0, math.inf),
        ("会话软保护（只知会回来）", "session_aware", "alive", 0.0, math.inf),
        ("点估计 pin（精确，上界）", "ttl", "eta", 0.0, math.inf),
        ("点估计 pin（σ=1）", "ttl", "eta", 1.0, math.inf),
        (f"点估计 pin（σ=1，≤p90={p90:.1f}s）", "ttl", "eta", 1.0, p90),
        ("软提示 ETA（σ=1）", "session_aware", "eta", 1.0, math.inf),
    ]
    caps = (1_000, 2_000, 4_000)
    print(LINE)
    print(f"B. 淘汰策略（只看命中率）：单实例，{len(sessions)} 个会话 / 8 个项目，按时间顺序重放")
    print("   σ=1：提示的工具耗时乘上 exp(N(0,1))，误差中位数约 2 倍；括号里是被拒绝插入的 block 数（全被 pin 住时新 KV 进不来）")
    print("-" * 104)
    header = "".join(f"{f'{c * BLOCK_B // 1000}k token':>20}" for c in caps)
    print(f"   {'策略 \\ 缓存容量':<30}{header}")
    for name, policy, hint, noise, cap_s in variants:
        cells = ""
        for c in caps:
            rate, rejected = replay_eviction(sessions, events, blocks, c, policy, hint, noise, cap_s)
            cells += f"{rate:>11.1%}{f'({rejected})':>9}"
        print(f"   {name:<30}{cells}")


# ---------------------------------------------------------------------------
# C. Continuum 的 τ*
# ---------------------------------------------------------------------------


def continuum_tau(samples: np.ndarray, benefit_s: float) -> tuple[float, float]:
    """τ* = argmax_τ P(T ≤ τ)·benefit − τ，在所有历史耗时（含 0）里枚举（Continuum §4.2 式 (2)）。返回 (τ*, P(τ*))。"""
    xs = np.sort(samples)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    gain = cdf * benefit_s - xs
    j = int(np.argmax(gain))
    if gain[j] <= 0.0:
        return 0.0, 0.0
    return float(xs[j]), float(cdf[j])


def d_sessions() -> list[AgentSession]:
    return generate_sessions(80, 8, seed=SEED, arrival_rate=2.0)


D_CACHE_BLOCKS = 3_000


def continuum_inputs(sessions: list[AgentSession]) -> dict:
    """按 Continuum §4.1 从负载里估 Benefit = 重载时间 + T̄·η 的各项（本库没有 swap，重载 = 重算 prefill）。"""
    cfg = MODELS["qwen3-32b"]
    ctx = np.array([context_tokens(s, k) for s in sessions for k in range(len(s.turns))])
    ks = [k for s in sessions for k in range(len(s.turns))]
    remaining = [len(s.turns) - k for s in sessions for k in range(len(s.turns))]
    eta = -float(np.corrcoef(ks, remaining)[0, 1])  # 记忆性因子：已做轮数能否预示剩余轮数

    def reload_s(n: int) -> float:
        gemm, attn = prefill_flops_split(cfg, n, 0)
        return step_time_s(GPUS["H100-SXM"], cfg, gemm, 0.0, n_gpus=4, attn_flops=attn)

    lru = route_and_simulate(sessions, 1, "round_robin", EngineConfig(**QWEN_4xH100), D_CACHE_BLOCKS, seed=SEED,
                             cache_policy="lru", hint_mode="none")
    t_queue = float(np.mean([r["ttft_s"] for r in lru["per_request"]]))  # 用 LRU 下的平均 TTFT 近似 T̄（偏大）
    return {"lru": lru, "eta": eta, "t_queue": t_queue, "ctx_mean": float(ctx.mean()), "ctx_p90": float(np.percentile(ctx, 90)),
            "reload_mean": reload_s(int(ctx.mean())), "reload_p90": reload_s(int(np.percentile(ctx, 90))),
            "samples": tool_latencies(sessions)}


def part_c(ci: dict) -> None:
    print(LINE)
    print("C. Continuum 的 τ*：Benefit = 重载时间 + T̄·η，τ* = argmax P(τ)·Benefit − τ（README 2.3），工具耗时用本负载的经验分布")
    print(f"   本负载（D 部分的 80 个会话）：η = −Corr(已做轮数, 剩余轮数) = {ci['eta']:+.2f}；"
          f"T̄ ≈ LRU 下平均 TTFT = {ci['t_queue']:.2f} s")
    print(f"   上下文均值 {ci['ctx_mean'] / 1e3:.0f}k token → 重算 {ci['reload_mean']:.2f} s；"
          f"p90 {ci['ctx_p90'] / 1e3:.0f}k token → 重算 {ci['reload_p90']:.2f} s（4×H100，Qwen3-32B）")
    print("-" * 104)
    print(f"   {'情形':<28}{'Benefit（秒）':>14}{'τ*（秒）':>10}{'P(工具在 τ* 内返回)':>22}")
    cases = [
        ("本负载，平均上下文", ci["reload_mean"] + ci["t_queue"] * ci["eta"]),
        ("本负载，p90 上下文", ci["reload_p90"] + ci["t_queue"] * ci["eta"]),
        ("假设：Benefit = 10 s", 10.0),
        ("假设：Benefit = 30 s", 30.0),
        ("假设：Benefit = 100 s", 100.0),
    ]
    for name, benefit in cases:
        tau, p = continuum_tau(ci["samples"], benefit)
        print(f"   {name:<28}{benefit:>14.2f}{tau:>10.2f}{p:>22.1%}")


# ---------------------------------------------------------------------------
# D. 端到端：淘汰策略 × pin 提示，看 TTFT 与 JCT
# ---------------------------------------------------------------------------


def part_d(ci: dict) -> None:
    cfg = EngineConfig(**QWEN_4xH100)
    sessions = d_sessions()
    tau_here, _ = continuum_tau(ci["samples"], ci["reload_p90"] + ci["t_queue"] * ci["eta"])
    tau_10, _ = continuum_tau(ci["samples"], 10.0)
    rows = [
        ("lru", "lru", "none", 0.0),
        (f"Continuum τ*（本负载）={tau_here:.1f}s", "ttl", "fixed", tau_here),
        (f"ttl 固定 τ*={tau_10:.1f}s（Benefit=10s）", "ttl", "fixed", tau_10),
        ("ttl + 带噪声 ETA（σ=1）", "ttl", "noisy", 0.0),
        ("session_aware + 带噪声 ETA", "session_aware", "noisy", 0.0),
        ("ttl + 精确 ETA（上界）", "ttl", "oracle", 0.0),
    ]
    print(LINE)
    print(f"D. 端到端（闭环、真实排队）：1 个实例（4×H100，Qwen3-32B），{len(sessions)} 个会话，"
          f"前缀缓存 {D_CACHE_BLOCKS:,} block（{D_CACHE_BLOCKS * 16 // 1000}k token）")
    print("   注意：本库的前缀缓存容量独立于引擎运行时的 KV，pin 的代价只体现为挤掉别人的缓存、不会挡住新请求准入")
    print("-" * 104)
    print(f"   {'淘汰策略 + 提示':<34}{'命中率':>8}{'TTFT p50':>10}{'TTFT p99':>10}{'JCT 均值':>10}{'JCT p90':>10}")
    for name, cache_policy, hint_mode, ttl in rows:
        if cache_policy == "lru" or (hint_mode == "fixed" and ttl == 0.0):
            r = ci["lru"]  # τ* = 0 即不 pin，和 LRU 是同一次模拟（continuum_inputs 里已经跑过）
        else:
            r = route_and_simulate(sessions, 1, "round_robin", cfg, cache_capacity_blocks=D_CACHE_BLOCKS, seed=SEED,
                                   cache_policy=cache_policy, hint_mode=hint_mode, hint_ttl_s=ttl)
        jct_mean, jct_p90 = jct_stats(r, sessions)
        print(f"   {name:<34}{r['token_hit_rate']:>8.1%}{r['ttft_p50']:>9.2f}s{r['ttft_p99']:>9.2f}s"
              f"{jct_mean:>9.1f}s{jct_p90:>9.1f}s")


# ---------------------------------------------------------------------------
# E. 加载 vs 重算
# ---------------------------------------------------------------------------

PCIE_GB_S = 64.0  # H100 SXM 标称 PCIe Gen5 128 GB/s（双向合计），单向约 64 GB/s
SSD_PEAK_GB_S = 40 * 2**30 / 8 / 1e9  # 3FS KVCache 读：每客户端峰值约 40 GiB/s，按 8 卡平分（峰值口径）


def part_e() -> None:
    gpu = GPUS["H800-SXM"]
    n = 32_768  # 重算一段 32k token 的前缀（含注意力项）
    print(LINE)
    print(f"E. 加载 vs 重算：在 {gpu.name} 上重算 32k token 前缀（含注意力，MFU 0.6）vs 从下层存储加载")
    print("   B* = 每 token KV 字节 × 单卡重算速率；加载带宽高于 B* 就该加载（README 4.2）。注意力统一按 BF16 计时")
    print("-" * 104)
    print(f"   {'模型':<13}{'注意力':>6}{'KV/token':>11}{'ρ=2N/b_kv':>12}{'B*（BF16）':>12}{'B*（FP8）':>11}{'PCIe 余量':>11}{'SSD 峰值余量':>13}")
    for name in ("llama3.1-8b", "qwen3-32b", "llama3.1-70b", "deepseek-v3"):
        cfg = MODELS[name]
        kv = kv_bytes_per_token(cfg)
        gemm, attn = prefill_flops_split(cfg, n, 0)
        b_star = {}
        for dtype in ("bf16", "fp8"):
            t = step_time_s(gpu, cfg, gemm, 0.0, n_gpus=1, dtype=dtype, attn_flops=attn, attn_dtype="bf16")
            b_star[dtype] = kv * n / t / 1e9
        rho = 2 * cfg.active_params_b * 1e9 / kv
        print(
            f"   {name:<13}{cfg.attn:>6}{kv / 1024:>9.1f}KiB{rho / 1e3:>10.0f}k{b_star['bf16']:>9.2f}GB/s{b_star['fp8']:>8.2f}GB/s"
            f"{PCIE_GB_S / b_star['bf16']:>10.0f}×{SSD_PEAK_GB_S / b_star['bf16']:>12.1f}×"
        )
    print(f"   （余量按 BF16 的 B* 算；PCIe 单向按 {PCIE_GB_S:.0f} GB/s/卡；SSD 按 3FS 峰值 40 GiB/s/节点 ÷ 8 = {SSD_PEAK_GB_S:.2f} GB/s/卡）")


# ---------------------------------------------------------------------------
# F. 命中率 × chunked prefill
# ---------------------------------------------------------------------------


def part_f() -> None:
    rng = np.random.default_rng(SEED)
    n_req, prompt, out = 96, 40_000, 300
    arrivals = np.cumsum(rng.exponential(1.25, n_req))  # 平均每 1.25 s 来一个 4 万 token 的 agent 请求
    print(LINE)
    print(f"F. 命中率 × chunked prefill：{n_req} 个请求，每个 prompt {prompt // 1000}k token、输出 {out} token，平均 1.25 s 到达一个（4×H100，Qwen3-32B）")
    print("-" * 104)
    print(f"   {'命中率':>6}{'chunked':>9}{'TTFT p50':>10}{'TTFT p99':>10}{'TPOT p50':>10}{'TPOT p99':>10}{'GPU 忙碌':>10}")
    for h in (0.0, 0.5, 0.9):
        for chunked in (True, False):
            reqs = [Request(i, float(arrivals[i]), prompt, int(prompt * h), out, i) for i in range(n_req)]
            r = simulate(reqs, EngineConfig(chunked_prefill=chunked, **QWEN_4xH100))
            print(
                f"   {h:>6.0%}{str(chunked):>9}{r['ttft_p50']:>9.2f}s{r['ttft_p99']:>9.2f}s"
                f"{r['tpot_p50'] * 1e3:>8.0f}ms{r['tpot_p99'] * 1e3:>8.0f}ms{r['busy_frac']:>10.0%}"
            )


if __name__ == "__main__":
    part_a()
    part_b()
    inputs = continuum_inputs(d_sessions())
    part_c(inputs)
    part_d(inputs)
    part_e()
    part_f()
