"""Capstone 演示：在 CPU 上真实跑 mini_engine，打印 README 里引用的全部数字。

运行：.venv/bin/python capstone/demo.py                  （M1 笔记本上约 6–12 秒，取决于机器负载）
      .venv/bin/python capstone/demo.py --with-solution  （额外跑第 5 节：练习 (b) 参考答案的会话感知淘汰）

第 1–5 节只统计 token 数或用虚拟时钟，每次运行结果完全一致；第 6–8 节是真实墙钟，随机器负载变化。
1. 正确性：各种优化开/关，greedy 输出都与"每步从头重算"的朴素实现逐 token 一致
2. 共享前缀：先后两个请求、同一步 G 份同 prompt
3. 多 agent harness：三种上下文策略的前缀命中率与 prefill 实算量（虚拟时钟）
4. 显存紧张时，等工具期间不 pin / oracle TTL pin / 固定 TTL pin（虚拟时钟）
5. （可选）会话感知淘汰 vs LRU（虚拟时钟）
6. 真实墙钟：并发 vs 串行，工具等待真实重叠；事件循环被 step 阻塞多久
7. chunk 大小：长 prompt 的 prefill 总耗时与 KV 重读量
8. CPU 上 prefill / decode 每 token 耗时
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from mini_engine import AsyncLLMEngine, EngineConfig, LLMEngine, PagedKVCache, TinyTransformer, reference_greedy
from mini_engine.agent_harness import HarnessConfig, HarnessReport, run_harness_sync
from mini_engine.model import SeqChunk


def toks(seed: int, n: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, 512, n)]


def section(title: str) -> None:
    print(f"\n== {title} ==")


def correctness(model: TinyTransformer) -> None:
    section("1. 正确性：与朴素重算逐 token 对比（6 个请求 × 10 token）")
    shared = toks(100, 40)
    prompts = [shared + toks(1, 10), shared + toks(2, 25), toks(3, 32), toks(4, 33), shared + toks(5, 50), toks(6, 7)]
    ref = [reference_greedy(model, p, 10) for p in prompts]
    configs = {
        "默认（一次全部提交）": (EngineConfig(), False),
        "逐个提交": (EngineConfig(), True),
        "关闭前缀缓存": (EngineConfig(enable_prefix_caching=False), False),
        "每步只给 7 个 token": (EngineConfig(max_num_batched_tokens=7), False),
        "只有 10 个块（逼出抢占）": (EngineConfig(num_blocks=10, max_num_batched_tokens=32), False),
    }
    print(f"{'配置':<22}{'步数':>6}{'prefill实算':>12}{'命中':>6}{'抢占':>6}  输出一致")
    for name, (cfg, sequential) in configs.items():
        eng = LLMEngine(model, cfg)
        outs = [eng.generate([p], 10)[0] for p in prompts] if sequential else eng.generate(prompts, 10)
        eng.allocator.assert_no_leak()
        s = eng.stats
        print(f"{name:<22}{s.steps:>6}{s.prefill_tokens_computed:>12}{s.cached_tokens:>6}{s.preemptions:>6}  {outs == ref}")
    diffs, gaps = [], []
    for i in range(10):                                  # 同一序列：单独算 vs 和别的序列拼成一个 batch
        a, b = toks(42 + i, 40), toks(430 + i, 57)
        kv = PagedKVCache(model.cfg, 16, 16)
        pa, pb, ta, tb = np.arange(40), np.arange(57), [0, 1, 2], [3, 4, 5, 6]
        sa, sb = np.asarray(ta)[pa // 16] * 16 + pa % 16, np.asarray(tb)[pb // 16] * 16 + pb % 16
        alone = model.forward_paged(np.asarray(a), pa, sa, [SeqChunk(0, 40, 40, ta)], kv)[0]
        both = model.forward_paged(np.asarray(b + a), np.concatenate([pb, pa]), np.concatenate([sb, sa]),
                                   [SeqChunk(0, 57, 57, tb), SeqChunk(57, 40, 40, ta)], kv)[1]
        diffs.append(float(np.abs(both - alone).max()))
        gaps.append(float(np.diff(np.sort(alone)[-2:])[0]))
    print(f"logits 不逐位相同：单独算 vs 拼 batch 最大差 {min(diffs):.1e}–{max(diffs):.1e}；"
          f"top-1 与 top-2 的间隔 {min(gaps):.3f}–{max(gaps):.2f}（10 个随机序列）")


def shared_prefix(model: TinyTransformer) -> None:
    section("2. 共享前缀")
    shared = toks(11, 40)
    eng = LLMEngine(model, EngineConfig())
    for i, suffix in enumerate((toks(12, 10), toks(13, 10))):
        before = eng.stats.prefill_tokens_computed
        rid = eng.add_request(shared + suffix, 4)
        eng.run_until_done()
        r = eng.requests[rid]
        print(f"先后两个请求 #{i}: prompt 50（前 40 共享），命中 {r.num_cached_tokens:>2}，实算 {eng.stats.prefill_tokens_computed - before}")
    p = toks(40, 200)
    for g in (1, 8):
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=2048))
        outs = eng.generate([p] * g, 4)
        s = eng.stats
        print(f"同一步提交 G={g} 份 200 token 的 prompt：prompt 共 {s.prompt_tokens:>4}，命中 {s.cached_tokens:>4}，"
              f"实算 {s.prefill_tokens_computed:>4}，步数 {s.steps}，G 份输出相同 {all(o == outs[0] for o in outs)}")


HEADER = f"{'':<16}{'prompt':>8}{'命中':>8}{'命中率':>9}{'prefill实算':>10}{'抢占':>6}{'淘汰':>6}{'步数':>6}{'耗时(s)':>9}"


def harness_row(name: str, r: HarnessReport) -> None:
    print(f"{name:<16}{r.prompt_tokens:>8}{r.cached_tokens:>8}{r.hit_rate:>9.1%}{r.prefill_computed:>10}"
          f"{r.preemptions:>6}{r.evictions:>6}{r.steps:>6}{r.elapsed_s:>9.3f}")


def harness(model: TinyTransformer, with_solution: bool) -> None:
    base = HarnessConfig()
    print(f"\n（harness：{base.n_agents} 个 agent × {base.n_turns} 轮；共享 system prompt {base.system_prompt_len} token；"
          f"每轮输出 {base.output_len}、工具结果 {base.tool_result_len} token；工具耗时 {base.tool_latency_s * 1e3:.0f} ms ±50%；"
          f"虚拟时钟：每步 1 ms + 0.1 ms/token）")
    section("3. harness 上下文策略（块充足：256 块；虚拟时钟）")
    print(HEADER)
    for s in ("append", "compress", "timestamp"):
        harness_row(s, run_harness_sync(model, replace(base, strategy=s)))

    section("4. 显存紧张（36 块）时，等工具期间 pin 与不 pin（append；虚拟时钟）")
    print(HEADER)
    tight = EngineConfig(num_blocks=36, max_pinned_frac=0.5)
    for policy, label in (("none", "不 pin"), ("oracle", "oracle TTL"), ("fixed", "固定 TTL")):
        harness_row(label, run_harness_sync(model, replace(base, pin_policy=policy), tight))

    if with_solution:
        section("5. 会话感知淘汰 vs LRU（练习 (b) 参考答案；会话长短不一：5/4/3/2 轮；32 块；虚拟时钟）")
        path = Path(__file__).with_name("solution.py")
        spec = importlib.util.spec_from_file_location("capstone_solution", path)
        sol = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = sol
        spec.loader.exec_module(sol)
        cfg, cap = replace(base, n_turns=5, turn_spread=3), EngineConfig(num_blocks=32)
        print(HEADER)
        lru = run_harness_sync(model, cfg, cap)
        sa = run_harness_sync(model, cfg, cap, allocator_cls=sol.SessionAwareAllocator)
        harness_row("LRU", lru)
        harness_row("会话感知", sa)
        print(f"两者输出一致：{lru.outputs == sa.outputs}")


def real_clock(model: TinyTransformer) -> None:
    section("6. 真实墙钟（随机器负载变化）")
    base = replace(HarnessConfig(), clock="real")
    conc = run_harness_sync(model, base)
    serial = run_harness_sync(model, replace(base, concurrent=False))
    print(f"工具等待之和 {conc.tool_wait_sum_s:.2f} s；并发墙钟 {conc.elapsed_s:.2f} s；串行墙钟 {serial.elapsed_s:.2f} s；"
          f"输出一致 {conc.outputs == serial.outputs}；并发时在别人等工具期间执行的步数 {conc.steps_during_others_tool_wait()}/{conc.steps}")

    async def loop_lag(in_executor: bool) -> tuple[float, float, float]:
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=1024, num_blocks=512))
        lags: list[float] = []
        done = False

        async def heartbeat():
            while not done:
                t = time.perf_counter()
                await asyncio.sleep(0.001)
                lags.append((time.perf_counter() - t - 0.001) * 1e3)

        async with AsyncLLMEngine(eng, step_in_executor=in_executor) as ae:
            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.01)
            t0 = time.perf_counter()
            await asyncio.gather(ae.generate(toks(600, 768), 2), *(ae.generate(toks(610 + i, 20), 8) for i in range(4)))
            wall = (time.perf_counter() - t0) * 1e3
            done = True
            await hb
        return float(np.median(lags)), max(lags), wall

    for label, flag in (("step 在事件循环线程里", False), ("step 放进线程池", True)):
        p50, mx, wall = asyncio.run(loop_lag(flag))
        print(f"{label:<16} 1 个 768 token 长 prompt + 4 个短请求：心跳延迟 p50 {p50:6.2f} ms、max {mx:7.2f} ms；总耗时 {wall:6.0f} ms")


def chunk_cost(model: TinyTransformer) -> None:
    section("7. chunk 大小：一个 768 token prompt 的 prefill（真实墙钟）")
    length = 768
    print(f"{'每步预算':>8}{'步数':>6}{'prefill 总耗时(ms)':>18}{'KV 重读(token)':>16}{'分数矩阵元素(百万,每层每头)':>20}")
    for budget in (768, 256, 64):
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=budget, num_blocks=512, enable_prefix_caching=False))
        eng.kv.k[:] = 0.0                               # 先触碰内存页
        eng.add_request(toks(700, length), 1)
        t = time.perf_counter()
        eng.run_until_done()
        ms = (time.perf_counter() - t) * 1e3
        chunks = [(i, min(budget, length - i)) for i in range(0, length, budget)]
        kv_reads = sum(i for i, _ in chunks)             # 每块要重读前面已经写好的 KV
        scores = sum(n * (i + n) for i, n in chunks)     # 我们的 _attend 物化完整的 [n, ctx] 分数矩阵
        print(f"{budget:>8}{eng.stats.steps:>6}{ms:>18.1f}{kv_reads:>16}{scores / 1e6:>18.2f}")


def cpu_timing(model: TinyTransformer) -> None:
    section("8. CPU 上每 token 耗时（float64 numpy，中位数；prefill 一步含首 token 的 logits 与调度开销）")
    eng = LLMEngine(model, EngineConfig(num_blocks=512, max_num_batched_tokens=512, max_num_seqs=64,
                                        enable_prefix_caching=False))
    eng.kv.k[:] = 0.0
    eng.kv.v[:] = 0.0

    def step_ms(prompts: list[list[int]], decode_steps: int = 0) -> float:
        for p in prompts:
            eng.add_request(p, decode_steps + 1)
        times = []
        while eng.has_unfinished():
            t = time.perf_counter()
            eng.step()
            times.append((time.perf_counter() - t) * 1e3)
        return float(np.median(times[1:])) if decode_steps else times[0]

    for n in (16, 64, 256):
        ms = float(np.median([step_ms([toks(200 + n + i, n)]) for i in range(3)]))
        print(f"prefill {n:>3} token 一步: {ms:6.2f} ms  → {ms / n * 1e3:7.1f} µs/token")
    for b in (1, 16):
        ms = step_ms([toks(300 + i, 64) for i in range(b)], decode_steps=16)
        print(f"decode batch={b:>2}（上下文 ~70）一步: {ms:6.2f} ms  → {ms / b * 1e3:7.1f} µs/token")


def main() -> None:
    t0 = time.perf_counter()
    model = TinyTransformer()
    c = model.cfg
    print(f"模型：vocab {c.vocab_size}, d_model {c.d_model}, {c.n_layers} 层, {c.n_heads} 个 query 头 / "
          f"{c.n_kv_heads} 个 KV 头, head_dim {c.head_dim}, SwiGLU {c.ffn_dim}；block_size 16")
    correctness(model)
    shared_prefix(model)
    harness(model, with_solution="--with-solution" in sys.argv)
    real_clock(model)
    chunk_cost(model)
    cpu_timing(model)
    print(f"\n总耗时 {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
