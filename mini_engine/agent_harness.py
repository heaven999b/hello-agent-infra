"""多 agent harness 演示：N 个 agent 共享 system prompt，多轮"生成 → 等工具 → 追加工具结果"。

对比三种 harness 上下文策略对前缀缓存的影响：
- append：只追加（上一轮的完整上下文是这一轮的前缀）→ 历史几乎全部命中；
- compress：每轮把历史压缩改写成"摘要标记 + 最近 K 个 token"→ 只有 system prompt 命中；
- timestamp：把每轮都变的内容（如时间戳）放在最前面 → 一个 token 变了，后面全部失效。
以及等工具期间 pin 与不 pin 的差异：显存紧张时，别人的 prefill 会把正在等工具的 agent **已登记的满块**淘汰掉
（尾部没写满的块本来就不能复用，下一轮总要重算）。

两种时钟：
- clock="virtual"（默认）：VirtualClock 离散事件模拟，引擎每步按成本模型推进虚拟时间，工具等待按虚拟时间计。
  结果逐次完全一致、与机器负载无关，测试和讲义里的对比表都用它；
- clock="real"：工具等待是真实的 `await asyncio.sleep(latency)`，墙钟时间真实重叠（demo 里演示用）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from .async_engine import AsyncLLMEngine, RealClock, StepSpan, VirtualClock
from .engine import EngineConfig, LLMEngine
from .kv_cache import BlockAllocator
from .model import TinyTransformer

STRATEGIES = ("append", "compress", "timestamp")
PIN_POLICIES = ("none", "oracle", "fixed")
SUMMARY_MARK = 1   # compress 策略的"摘要开始"标记 token


@dataclass(frozen=True)
class HarnessConfig:
    n_agents: int = 6
    n_turns: int = 4
    turn_spread: int = 0             # >0 时第 i 个 agent 少跑 i % (turn_spread+1) 轮（会话长短不一）
    system_prompt_len: int = 96      # 所有 agent 共享（block_size=16 时是 6 个满块）
    task_len: int = 16               # 每个 agent 自己的任务描述
    output_len: int = 8              # 每轮生成的 token 数（相当于一次工具调用指令）
    tool_result_len: int = 24        # 每轮工具返回追加的 token 数
    tool_latency_s: float = 0.04
    latency_jitter: float = 0.5      # 每次工具耗时 ∈ [1-j, 1+j] × tool_latency_s（固定 seed）
    start_stagger_s: float = 0.005   # 第 i 个 agent 在 i × stagger 秒后开始
    strategy: str = "append"
    compress_keep: int = 48          # compress 策略保留最近多少个历史 token
    pin_policy: str = "none"         # none | oracle（本次工具真实耗时 × margin，看得见未来）| fixed（按分布上界定 TTL）
    pin_margin: float = 1.5
    clock: str = "virtual"           # virtual | real
    concurrent: bool = True          # False：agent 一个接一个跑（串行基线）
    step_in_executor: bool = False
    seed: int = 0

    def turns_of(self, agent_id: int) -> int:
        return self.n_turns - agent_id % (self.turn_spread + 1)

    def pin_seconds(self, latency: float) -> float:
        if self.pin_policy == "oracle":
            return latency * self.pin_margin
        if self.pin_policy == "fixed":
            return self.tool_latency_s * (1 + self.latency_jitter)
        return 0.0


@dataclass
class HarnessReport:
    clock: str
    elapsed_s: float                 # 从开始到最后一个 agent 结束（virtual 时是虚拟时间）
    tool_wait_sum_s: float           # 所有工具等待时间之和（串行时光等工具就要这么久）
    prompt_tokens: int
    lookup_tokens: int               # 查缓存的 token 总数（含抢占后重新接纳），命中率的分母
    cached_tokens: int
    prefill_computed: int
    decode_tokens: int
    preemptions: int
    evictions: int
    steps: int
    outputs: dict[int, list[list[int]]] = field(default_factory=dict)
    tool_intervals: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    step_log: list[StepSpan] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.lookup_tokens if self.lookup_tokens else 0.0

    def steps_during_others_tool_wait(self) -> int:
        """有多少步是在"某个 agent 正在等工具、且它不在本步 batch 里"时执行的——计算与工具等待重叠的直接证据。"""
        n = 0
        for span in self.step_log:
            in_batch = {e.session_id for e in span.record.entries}
            if any(a not in in_batch and any(t0 <= span.start < t1 for t0, t1 in ivs)
                   for a, ivs in self.tool_intervals.items()):
                n += 1
        return n


def _tokens(cfg: HarnessConfig, vocab: int, n: int, *key: int) -> list[int]:
    """确定性的"文本"：由 (seed, *key) 决定。token 从 2 开始，0/1 留作特殊标记。"""
    return [int(t) for t in np.random.default_rng([cfg.seed, *key]).integers(2, vocab, n)]


def tool_latencies(cfg: HarnessConfig, agent_id: int) -> list[float]:
    u = np.random.default_rng([cfg.seed, agent_id, 10_000]).random(max(cfg.turns_of(agent_id) - 1, 0))
    return [cfg.tool_latency_s * (1 + cfg.latency_jitter * (2 * x - 1)) for x in u]


def build_prompt(cfg: HarnessConfig, system: list[int], history: list[int], agent_id: int, turn: int,
                 vocab: int = 512) -> list[int]:
    if cfg.strategy == "timestamp":
        stamp = 2 + (agent_id * 131 + turn * 17) % min(500, vocab - 2)   # 每个 agent 每轮都不同的"时间戳"，不越出词表
        return [stamp] + system + history
    return system + history


def update_history(cfg: HarnessConfig, history: list[int], output: list[int], tool_result: list[int],
                   turn: int, vocab: int = 512) -> list[int]:
    full = history + output + tool_result
    if cfg.strategy == "compress":
        return [SUMMARY_MARK, 2 + turn % (vocab - 2)] + full[-cfg.compress_keep:]   # 改写：前缀从这里开始变了
    return full


async def _run_agent(ae: AsyncLLMEngine, clock: VirtualClock | RealClock, cfg: HarnessConfig,
                     system: list[int], agent_id: int, vocab: int,
                     intervals: list[tuple[float, float]]) -> list[list[int]]:
    if cfg.concurrent:
        await clock.sleep(agent_id * cfg.start_stagger_s)
    history = _tokens(cfg, vocab, cfg.task_len, agent_id, 0)
    latencies = tool_latencies(cfg, agent_id)
    n_turns = cfg.turns_of(agent_id)
    outputs: list[list[int]] = []
    for turn in range(n_turns):
        last = turn == n_turns - 1
        pin = 0.0 if last else cfg.pin_seconds(latencies[turn])
        prompt = build_prompt(cfg, system, history, agent_id, turn, vocab)
        res = await ae.generate(prompt, cfg.output_len, pin_seconds=pin, session_id=agent_id)
        outputs.append(res.output_token_ids)
        if last:
            break
        t0 = clock()
        await clock.sleep(latencies[turn])                           # 工具调用
        intervals.append((t0, clock()))
        tool_result = _tokens(cfg, vocab, cfg.tool_result_len, agent_id, turn + 1)
        history = update_history(cfg, history, res.output_token_ids, tool_result, turn, vocab)
    ae.end_session(agent_id)                                         # 告诉引擎：这个会话结束了
    return outputs


async def run_harness(model: TinyTransformer, cfg: HarnessConfig, engine_cfg: EngineConfig | None = None,
                      allocator_cls: type[BlockAllocator] = BlockAllocator) -> HarnessReport:
    if cfg.strategy not in STRATEGIES or cfg.pin_policy not in PIN_POLICIES or cfg.clock not in ("virtual", "real"):
        raise ValueError(f"strategy ∈ {STRATEGIES}, pin_policy ∈ {PIN_POLICIES}, clock ∈ (virtual, real)")
    clock: VirtualClock | RealClock = VirtualClock() if cfg.clock == "virtual" else RealClock()
    engine = LLMEngine(model, engine_cfg, clock=clock, allocator_cls=allocator_cls)
    vocab = model.cfg.vocab_size
    system = _tokens(cfg, vocab, cfg.system_prompt_len, 999_999)
    intervals: dict[int, list[tuple[float, float]]] = {i: [] for i in range(cfg.n_agents)}
    t0, w0 = clock(), time.perf_counter()
    async with AsyncLLMEngine(engine, step_in_executor=cfg.step_in_executor, record_steps=True,
                              virtual_clock=clock if isinstance(clock, VirtualClock) else None) as ae:
        run = [_run_agent(ae, clock, cfg, system, i, vocab, intervals[i]) for i in range(cfg.n_agents)]
        outs = await asyncio.gather(*run) if cfg.concurrent else [await c for c in run]
    elapsed = clock() - t0 if cfg.clock == "virtual" else time.perf_counter() - w0
    engine.allocator.assert_no_leak()
    s = engine.stats
    return HarnessReport(
        clock=cfg.clock, elapsed_s=elapsed,
        tool_wait_sum_s=sum(sum(tool_latencies(cfg, i)) for i in range(cfg.n_agents)),
        prompt_tokens=s.prompt_tokens, lookup_tokens=s.lookup_tokens, cached_tokens=s.cached_tokens,
        prefill_computed=s.prefill_tokens_computed, decode_tokens=s.decode_tokens,
        preemptions=s.preemptions, evictions=engine.allocator.num_evictions, steps=s.steps,
        outputs=dict(enumerate(outs)), tool_intervals=intervals, step_log=ae.step_log)


def run_harness_sync(model: TinyTransformer, cfg: HarnessConfig, engine_cfg: EngineConfig | None = None,
                     allocator_cls: type[BlockAllocator] = BlockAllocator) -> HarnessReport:
    return asyncio.run(run_harness(model, cfg, engine_cfg, allocator_cls))
