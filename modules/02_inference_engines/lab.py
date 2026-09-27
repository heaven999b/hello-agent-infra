"""模块 02 实验：chunked prefill、token 预算、前缀命中与 P/D 分离如何改变延迟、容量和 goodput。

运行：python modules/02_inference_engines/lab.py   （或 make lab N=02）

全部在「Llama-3.1-70B（BF16），每个实例 8 × H100-SXM」上跑（codesign_sim 不计通信）：
  实验 1：调度方式——prefill 优先（decode 暂停）、decode 优先但不分块、decode 优先 + 分块
  实验 2：token 预算（max_batch_tokens）从 128 扫到 32768：延迟 + 闭环容量
  实验 3：只有 32k agent 请求，扫前缀命中比例 cached_tokens / prompt_tokens
  实验 4：同样 32 张卡，4 个共置实例 vs P/D 分离（2P+2D、3P+1D），两种 SLO 下的 goodput

指标：
- TTFT / TPOT：请求级分位数；实验 1–3 的负载按泊松过程开环到达。
- ITL max：所有请求里最长的一次 token 间隔，即「最严重的一次卡顿」（simulate 公开提供的 itl_max 字段）。
- 容量：闭环测量——固定 64 个并发客户端，每个请求完成后立刻发下一个，直到跑完同一批请求；
  报告输出 token 吞吐。开环负载下的输出吞吐几乎恒等于提供负载，不能拿来比较调度策略。
- goodput（DistServe 的定义）：至少 90% 的请求同时满足 TTFT 与 TPOT SLO 时能承受的最大请求率；
  达标率用 pd 模块按 ttft_slo_s / tpot_slo_s 算出的 slo_attainment。
模拟器是成本模型（roofline + 离散事件），数字用来比较「谁更好、差几倍」，不是真实毫秒数。
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from codesign_sim.engine import Engine, EngineConfig, Request, simulate
from codesign_sim.hardware import GPUS
from codesign_sim.model import MODELS
from codesign_sim.pd import simulate_colocated, simulate_pd

GPU = GPUS["H100-SXM"]
MODEL = MODELS["llama3.1-70b"]
N_GPUS = 8  # 每个实例的卡数

AGENT_PROMPT = 32_768  # agent 请求的 prompt 长度（仓库上下文 + 工具定义 + 历史）
AGENT_OUTPUT = 200
CONCURRENCY = 64  # 闭环容量测量的并发客户端数


def poisson_arrivals(rng: np.random.Generator, rate: float, duration_s: float) -> np.ndarray:
    """[0, duration_s) 内的泊松到达时刻。"""
    n = rng.poisson(rate * duration_s)
    return np.sort(rng.uniform(0.0, duration_s, size=n))


def mixed_workload(seed: int = 0, duration_s: float = 60.0) -> list[Request]:
    """短对话（4 req/s，prompt 200–1000，输出 100–400）+ agent 请求（0.25 req/s，prompt 32k，输出 200）。"""
    rng = np.random.default_rng(seed)
    reqs: list[Request] = []
    for t in poisson_arrivals(rng, rate=4.0, duration_s=duration_s):
        reqs.append(Request(len(reqs), float(t), int(rng.integers(200, 1001)), 0, int(rng.integers(100, 401))))
    for t in poisson_arrivals(rng, rate=0.25, duration_s=duration_s):
        reqs.append(Request(len(reqs), float(t), AGENT_PROMPT, 0, AGENT_OUTPUT))
    return reqs


def agent_workload(cached_frac: float, seed: int = 1, rate: float = 0.7, duration_s: float = 120.0) -> list[Request]:
    """只有 agent 请求，每个请求 prompt 的前 cached_frac 已在前缀缓存里。"""
    rng = np.random.default_rng(seed)
    cached = int(AGENT_PROMPT * cached_frac)
    return [
        Request(i, float(t), AGENT_PROMPT, cached, AGENT_OUTPUT)
        for i, t in enumerate(poisson_arrivals(rng, rate=rate, duration_s=duration_s))
    ]


def engine_cfg(budget: int = 8192, chunked: bool = True) -> EngineConfig:
    return EngineConfig(gpu=GPU, model=MODEL, n_gpus=N_GPUS, max_batch_tokens=budget, chunked_prefill=chunked)


def closed_loop_capacity(reqs: list[Request], cfg: EngineConfig, concurrency: int = CONCURRENCY) -> float:
    """固定并发的闭环：先发 concurrency 个请求，每完成一个就立刻发下一个；返回输出 token 吞吐。"""
    todo = deque(sorted(reqs, key=lambda r: (r.arrival_s, r.req_id)))
    engine = Engine(cfg)
    for _ in range(min(concurrency, len(todo))):
        engine.submit(replace(todo.popleft(), arrival_s=0.0))
    while engine.has_work():
        for rec in engine.step():
            if todo:
                engine.submit(replace(todo.popleft(), arrival_s=rec["finish_s"]))
    return engine.summary()["throughput_tok_s"]


LAT_COLS = ["TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99", "ITL max"]


def print_header(first_col: str, extra: list[str] | None = None) -> None:
    cols = LAT_COLS + (extra or [])
    print(f"{first_col:>24} | " + " ".join(f"{c:>9}" for c in cols))
    print("-" * (27 + 10 * len(cols)))


def print_row(label: str, r: dict, extra: list[float] | None = None) -> None:
    lat = [r["ttft_p50"], r["ttft_p99"], r["tpot_p50"], r["tpot_p99"], r["itl_max"]]
    cells = [f"{x * 1000:9.0f}" for x in lat] + [f"{x:9.0f}" for x in (extra or [])]
    print(f"{label:>24} | " + " ".join(cells))


def exp1_scheduling() -> None:
    reqs = mixed_workload()
    n_agent = sum(r.prompt_tokens == AGENT_PROMPT for r in reqs)
    print(f"\n=== 实验 1：调度方式（{len(reqs)} 个请求，其中 {n_agent} 个 32k agent 请求；延迟单位 ms）===")
    print_header("mode")
    print_row("prefill-first (Orca/V0)", simulate(reqs, engine_cfg(chunked=False)))
    print_row("decode-first, no chunk", simulate(reqs, engine_cfg(budget=32768)))
    print_row("decode-first, chunk 8192", simulate(reqs, engine_cfg(budget=8192)))
    print_row("decode-first, chunk 2048", simulate(reqs, engine_cfg(budget=2048)))


def exp2_budget_sweep() -> None:
    reqs = mixed_workload()
    print(f"\n=== 实验 2：token 预算（decode 优先 + 分块；负载同实验 1；延迟单位 ms；容量 = {CONCURRENCY} 并发闭环）===")
    print_header("budget", ["容量 tok/s"])
    for budget in (128, 256, 512, 2048, 8192, 32768):
        cfg = engine_cfg(budget=budget)
        print_row(str(budget), simulate(reqs, cfg), [closed_loop_capacity(reqs, cfg)])


def exp3_prefix_hits() -> None:
    print("\n=== 实验 3：前缀命中比例（只有 32k agent 请求，0.7 req/s，预算 8192；延迟单位 ms）===")
    print_header("cached")
    for frac in (0.0, 0.5, 0.75, 0.9, 0.99):
        print_row(f"{frac:.0%}", simulate(agent_workload(frac), engine_cfg()))


def pd_workload(rate: float, n: int = 80, seed: int = 3) -> list[Request]:
    """长输入短输出的 agent 负载：prompt 32k、输出 200，泊松到达。"""
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=n)
    return [Request(i, float(t), AGENT_PROMPT, 0, AGENT_OUTPUT) for i, t in enumerate(np.cumsum(gaps))]


def exp4_pd_goodput() -> None:
    rates = (1.5, 2.0, 2.5, 3.0, 3.5)
    slos = {"strict TPOT<=30ms": (5.0, 0.03), "loose TPOT<=500ms": (5.0, 0.5)}
    cfg = engine_cfg()
    systems = {
        "colocated x4": lambda reqs, **slo: simulate_colocated(reqs, cfg, n_instances=4, **slo),
        "P/D 2P+2D": lambda reqs, **slo: simulate_pd(reqs, cfg, cfg, n_prefill=2, n_decode=2, **slo),
        "P/D 3P+1D": lambda reqs, **slo: simulate_pd(reqs, cfg, cfg, n_prefill=3, n_decode=1, **slo),
    }
    print("\n=== 实验 4：同样 32 张 H100，共置 vs P/D 分离的 goodput（prompt 32k / 输出 200；TTFT SLO 均为 5 s；90% 达标）===")
    print(f"{'system':>12} | " + " ".join(f"{s:>18}" for s in slos)
          + " | @2.5 req/s: TTFT p99 / TPOT p99 (ms), KV peak")
    print("-" * 104)
    for name, run in systems.items():
        goodput, at_25 = [], None
        for ttft, tpot in slos.values():
            ok = []
            for rate in rates:
                r = run(pd_workload(rate), ttft_slo_s=ttft, tpot_slo_s=tpot)
                if r["slo_attainment"] >= 0.9:
                    ok.append(rate)
                if rate == 2.5 and at_25 is None:
                    at_25 = r
            if not ok:
                goodput.append(f"< {rates[0]:.1f} req/s")
            else:
                goodput.append(f"{'>= ' if max(ok) == rates[-1] else ''}{max(ok):.1f} req/s")
        kv_peak = at_25.get("decode_kv_peak_frac", at_25.get("kv_peak_frac"))
        print(f"{name:>12} | " + " ".join(f"{g:>18}" for g in goodput)
              + f" | {at_25['ttft_p99'] * 1000:6.0f} / {at_25['tpot_p99'] * 1000:4.0f}, {kv_peak:.2f}")


def main() -> None:
    print(f"模型 {MODEL.name}，每实例 {N_GPUS} × {GPU.name}")
    exp1_scheduling()
    exp2_budget_sweep()
    exp3_prefix_hits()
    exp4_pd_goodput()


if __name__ == "__main__":
    main()
