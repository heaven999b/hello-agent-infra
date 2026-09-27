"""多实例路由 + 闭环 agent 负载模拟。

每个 session 是闭环的：第 t 轮完成 → 等工具 tool_latency_s → 第 t+1 轮到达。
每个实例 = 一个 Engine（n_gpus 张卡，有真实的排队：KV 容量、token 预算、max_num_seqs）+ 一个自己的 PrefixCache。

请求的生命周期：
1. 到达：算出本轮 prompt 的 block hash 序列；router 按策略选实例
2. 在该实例的缓存上 match：HBM 命中免费；DRAM 命中（cache_dram_blocks > 0 时）要按 PCIe 带宽搬回，
   搬运比重算还慢就当作未命中、重算
3. 写缓存的时机（insert_on）：
   - "prefill_done"（默认）：prefill 完成即写入 prompt 的 block——之后到达的同前缀请求就能命中
     （SGLang RadixCache 对未完成请求的处理）
   - "route"：路由时立刻写入（SGLang router 的近似树就是这样更新的；最乐观）
   - "finish"：请求完成才写（旧版行为；会掩盖 cache_affinity 的热点代价）
   三种方式在请求完成时都会再写入 prompt + 输出，并附带 pin_until 提示（给 ttl / session_aware 用）
4. pin_until 提示（hint_mode）：
   - "noisy"（默认）：完成时刻 + 真实工具延迟 × exp(σ·N(0,1))，σ = hint_noise_sigma
   - "fixed"：完成时刻 + hint_ttl_s（固定 TTL）
   - "oracle"：完成时刻 + 真实工具延迟——等于提前知道下一轮何时到达，是**上界**，需显式选择
   - "none"：不给提示
   最后一轮不给提示（harness 知道 session 已结束）

路由策略：
- round_robin：按请求轮询（同一 session 的相邻轮次会落到不同实例）
- least_loaded：选负载最低的实例
- cache_affinity：只看命中，选前缀命中最长的实例（平局选负载低）——**极端对照**，会产生热点
- session_sticky：会话第一轮按负载最低选实例，之后固定回到同一实例（会话粘性基线）；
  sticky_migrate=True 时，若粘住的实例相对最空闲实例失衡（判据同下），迁移到最空闲实例并改粘到那里
- affinity_with_load_cap：SGLang router cache_aware 策略的结构（sgl-model-gateway policies/cache_aware.rs）：
  若 (max 负载 - min 负载) > balance_abs_threshold 且 max > load_cap_factor · min，判定失衡，改选负载最低的；
  否则命中率 > cache_threshold 选命中最长的，不到阈值选负载最低的
负载（load_metric）："tokens"（默认，剩余 prefill + 预计 decode）或 "requests"（在途请求数，SGLang 的口径）；
count_tool_waiting=True 时再计入"正在等工具、马上会回来"的 session。
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .cache import PrefixCache
from .engine import Engine, EngineConfig, Request, summarize
from .hardware import PCIE_GEN5_GBPS
from .model import kv_bytes_per_token, prefill_flops_split, step_time_s
from .workload import AgentSession, BlockHasher, append_output, append_prompt, build_hasher, context_tokens

POLICIES = ("round_robin", "least_loaded", "cache_affinity", "affinity_with_load_cap", "session_sticky")
INSERT_ON = ("prefill_done", "route", "finish")
HINT_MODES = ("noisy", "fixed", "oracle", "none")
_PREFILL_DONE, _COMPLETE, _ARRIVE = 0, 1, 2  # 同一时刻的处理顺序


class _Router:
    def __init__(self, n: int, policy: str, rng: np.random.Generator, *, cache_threshold: float,
                 balance_abs: float, balance_rel: float):
        self.n, self.policy, self.rng = n, policy, rng
        self.cache_threshold, self.balance_abs, self.balance_rel = cache_threshold, balance_abs, balance_rel
        self._rr = 0
        self.sticky: dict[int, int] = {}
        self.sticky_migrate = False
        self.migrations = 0

    def _imbalanced(self, load: float, lo: float) -> bool:
        return load - lo > self.balance_abs and load > self.balance_rel * lo

    def choose(self, blocks: list[int], loads: list[float], caches: list[PrefixCache], session_id: int = -1) -> int:
        if self.policy == "session_sticky":
            idx = self.sticky.get(session_id)
            if idx is None or (self.sticky_migrate and self._imbalanced(loads[idx], min(loads))):
                if idx is not None:
                    self.migrations += 1
                idx = self._least_loaded(range(self.n), loads)
                self.sticky[session_id] = idx
            return idx
        if self.policy == "round_robin":
            idx = self._rr % self.n
            self._rr += 1
            return idx
        if self.policy == "least_loaded":
            return self._least_loaded(range(self.n), loads)
        if self.policy == "affinity_with_load_cap":
            if self._imbalanced(max(loads), min(loads)):  # 失衡：退回最短队列
                return self._least_loaded(range(self.n), loads)
        matches = [c.peek(blocks) for c in caches]
        best = max(matches)
        if self.policy == "affinity_with_load_cap" and best <= self.cache_threshold * max(1, len(blocks)):
            return self._least_loaded(range(self.n), loads)
        return self._least_loaded([i for i in range(self.n) if matches[i] == best], loads)

    def _least_loaded(self, cands, loads: list[float]) -> int:
        cands = list(cands)
        low = min(loads[i] for i in cands)
        ties = [i for i in cands if loads[i] == low]
        return ties[0] if len(ties) == 1 else int(self.rng.choice(ties))


def route_and_simulate(
    sessions: list[AgentSession],
    n_instances: int,
    policy: str,
    cfg: EngineConfig,
    cache_capacity_blocks: int,
    seed: int = 0,
    *,
    block_size: int = 16,
    cache_policy: str = "lru",
    load_cap_factor: float = 1.5,
    cache_threshold: float = 0.3,
    balance_abs_threshold: float | None = None,
    load_metric: str = "tokens",
    est_output_tokens: float = 300.0,
    count_tool_waiting: bool = False,
    est_return_tokens: float = 1300.0,
    insert_on: str = "prefill_done",
    hint_mode: str = "noisy",
    hint_ttl_s: float = 10.0,
    hint_noise_sigma: float = 1.0,
    cache_dram_blocks: int = 0,
    dram_gbps: float | None = None,
    sticky_migrate: bool = False,
) -> dict:
    """在 n_instances 个实例上闭环模拟所有 session，返回命中率、负载不均衡度与延迟分位数。

    参数（除契约的前 6 个外都是扩展参数）：
    - load_cap_factor：affinity_with_load_cap 的相对失衡阈值（SGLang --balance-rel-threshold，CLI 默认 1.5），必须 > 0
    - cache_threshold：命中率阈值（SGLang --cache-threshold，CLI 默认 0.3）
    - balance_abs_threshold：绝对失衡阈值（SGLang --balance-abs-threshold，CLI 默认 64 个请求，那是生产规模）；
      None 时按小规模模拟缩放：load_metric="requests" 取 4，"tokens" 取 4 × max_batch_tokens
    - load_metric / est_output_tokens / count_tool_waiting / est_return_tokens：负载度量（见模块说明）
    - insert_on / hint_mode / hint_ttl_s / hint_noise_sigma：写缓存时机与 pin 提示（见模块说明）
    - cache_dram_blocks / dram_gbps：第二层（主机 DRAM）容量与每实例搬运带宽（默认 n_gpus × 64 GB/s，PCIe Gen5 单向）
    - sticky_migrate：session_sticky 策略下，粘住的实例失衡时是否迁移（默认不迁移）

    返回字段：
    - hit_rate：block 级前缀命中率（HBM + DRAM）；hbm_hit_rate / dram_hit_rate 分层
    - token_hit_rate：1 - 实际 prefill token / prompt token 总数
    - per_instance_hit_rate, requests_per_instance, busy_s_per_instance, max_instance_share（最热实例的请求占比）
    - load_imbalance：max(实例忙碌时间) / mean(实例忙碌时间)，1.0 表示完全均衡
    - ttft_p50/p99, tpot_p50/p99, itl_p50/p99, throughput_tok_s, busy_frac, makespan_s
    - prefill_tokens_total, prefill_tokens_computed, evictions, dram_loaded_blocks, dram_recomputed_blocks, per_request
    """
    if n_instances <= 0:
        raise ValueError("n_instances 必须 >= 1")
    if policy not in POLICIES:
        raise ValueError(f"policy 必须是 {POLICIES} 之一，收到 {policy!r}")
    if load_cap_factor <= 0:
        raise ValueError("load_cap_factor 必须 > 0")
    for name, val, allowed in [("insert_on", insert_on, INSERT_ON), ("hint_mode", hint_mode, HINT_MODES),
                               ("load_metric", load_metric, ("tokens", "requests"))]:
        if val not in allowed:
            raise ValueError(f"{name} 必须是 {allowed} 之一，收到 {val!r}")
    if balance_abs_threshold is None:
        balance_abs_threshold = 4.0 if load_metric == "requests" else 4.0 * cfg.max_batch_tokens
    rng = np.random.default_rng(seed)
    router = _Router(n_instances, policy, rng, cache_threshold=cache_threshold,
                     balance_abs=balance_abs_threshold, balance_rel=load_cap_factor)
    router.sticky_migrate = sticky_migrate
    engines = [Engine(cfg) for _ in range(n_instances)]
    caches = [PrefixCache(cache_capacity_blocks, cache_policy, dram_blocks=cache_dram_blocks) for _ in range(n_instances)]
    dram_bw = (dram_gbps if dram_gbps is not None else cfg.n_gpus * PCIE_GEN5_GBPS / 2.0) * 1e9
    kvb = kv_bytes_per_token(cfg.model, cfg.kv_dtype_bytes)
    hint_rng = np.random.default_rng(seed + 7919)

    seq = itertools.count()
    events: list = []
    for s in sessions:
        heapq.heappush(events, (s.arrival_s, _ARRIVE, next(seq), (s, 0)))
    hashers: dict[int, BlockHasher] = {}
    inflight: dict[int, tuple[AgentSession, int, int, list[int]]] = {}
    tool_waiting: list[dict[int, float]] = [dict() for _ in range(n_instances)]
    req_ids = itertools.count()
    counters = {"prompt_total": 0, "dram_loaded": 0, "dram_recomputed": 0}

    def loads() -> list[float]:
        out = []
        for i, e in enumerate(engines):
            base = e.inflight_tokens(est_output_tokens) if load_metric == "tokens" else float(e.n_inflight)
            if count_tool_waiting:
                w = tool_waiting[i]
                base += len(w) * (est_return_tokens if load_metric == "tokens" else 1.0)
            out.append(base)
        return out

    def recompute_s(n_new: int, n_cached: int) -> float:
        gemm, attn = prefill_flops_split(cfg.model, n_new, n_cached)
        return step_time_s(cfg.gpu, cfg.model, gemm, 0.0, cfg.n_gpus, cfg.mfu, cfg.mbu, cfg.resolved_gemm_dtype(),
                           attn_flops=attn, attn_dtype=cfg.attn_dtype)

    def hint(now: float, tool_s: float) -> float | None:
        if hint_mode == "none":
            return None
        if hint_mode == "fixed":
            return now + hint_ttl_s
        if hint_mode == "oracle":
            return now + tool_s
        return now + tool_s * float(np.exp(hint_noise_sigma * hint_rng.standard_normal()))

    def on_arrive(now: float, s: AgentSession, turn: int) -> None:
        h = _prompt_hasher(hashers, s, turn, block_size)
        blocks = list(h.hashes)
        i = router.choose(blocks, loads(), caches, s.session_id)
        for w in tool_waiting:
            w.pop(s.session_id, None)
        hbm, dram = caches[i].match_tiers(blocks, now)
        load_s = 0.0
        if dram:
            t_load = dram * block_size * kvb / dram_bw
            if t_load <= recompute_s(dram * block_size, hbm * block_size):
                load_s = t_load
                counters["dram_loaded"] += dram
                caches[i].insert(blocks[: hbm + dram], now)  # 搬回 HBM（提升）
            else:
                counters["dram_recomputed"] += dram
                dram = 0
        prompt = context_tokens(s, turn)
        counters["prompt_total"] += prompt
        rid = next(req_ids)
        engines[i].submit(Request(rid, now, prompt, (hbm + dram) * block_size, s.turns[turn].output_tokens,
                                  s.session_id, load_s=load_s))
        inflight[rid] = (s, turn, i, blocks)
        if insert_on == "route":
            caches[i].insert(blocks, now)

    def on_complete(now: float, i: int, rec: dict) -> None:
        s, turn, _, _ = inflight.pop(rec["req_id"])
        h = hashers[s.session_id]
        append_output(h, s, turn)
        last = turn == len(s.turns) - 1
        tool_s = s.turns[turn].tool_latency_s
        caches[i].insert(list(h.hashes), now, pin_until=None if last else hint(now, tool_s))
        if last:
            del hashers[s.session_id]
        else:
            tool_waiting[i][s.session_id] = now + tool_s
            heapq.heappush(events, (now + tool_s, _ARRIVE, next(seq), (s, turn + 1)))

    while True:
        t_ev = events[0][0] if events else math.inf
        idx, t_inst, t_second = _earliest(engines)
        if t_ev == math.inf and t_inst == math.inf:
            break
        if t_ev <= t_inst:
            now, kind, _, payload = heapq.heappop(events)
            if kind == _ARRIVE:
                on_arrive(now, *payload)
            elif kind == _COMPLETE:
                on_complete(now, *payload)
            else:
                i, rid = payload
                if rid in inflight:
                    caches[i].insert(inflight[rid][3], now)
            continue
        # 连续推进最早的实例，直到它追上下一个事件或另一个实例（与逐步交替推进等价，只是更快）
        engine = engines[idx]
        bound = min(t_ev, t_second)
        while True:
            for rec in engine.step():
                heapq.heappush(events, (rec["finish_s"], _COMPLETE, next(seq), (idx, rec)))
                bound = min(bound, rec["finish_s"])
            if insert_on == "prefill_done" and engine.prefill_done_log:
                for rid, t_done in engine.prefill_done_log:
                    heapq.heappush(events, (t_done, _PREFILL_DONE, next(seq), (idx, rid)))
                    bound = min(bound, t_done)
                engine.prefill_done_log.clear()
            if engine.next_step_time() >= bound:
                break

    out = _collect(engines, caches, counters)
    out["migrations"] = router.migrations
    return out


def _earliest(engines: list[Engine]) -> tuple[int, float, float]:
    """返回 (下一步最早的实例下标, 它的时刻, 第二早的时刻)。"""
    best_i, best_t, second = -1, math.inf, math.inf
    for i, e in enumerate(engines):
        t = e.next_step_time()
        if t < best_t:
            best_i, best_t, second = i, t, best_t
        elif t < second:
            second = t
    return best_i, best_t, second


def _prompt_hasher(hashers: dict[int, BlockHasher], s: AgentSession, turn: int, block_size: int) -> BlockHasher:
    """增量维护每个 session 的 block hash：没有 rewrite 就在上一轮基础上追加，rewrite 则整体重建。"""
    h = hashers.get(s.session_id)
    if h is None or turn == 0 or s.turns[turn].rewrite:
        h = build_hasher(s, turn, block_size)
    else:
        append_prompt(h, s, turn)
    hashers[s.session_id] = h
    return h


def _collect(engines: list[Engine], caches: list[PrefixCache], counters: dict) -> dict:
    records = [{**r, "instance": i} for i, e in enumerate(engines) for r in e.finished]
    itl = [c for e in engines for c in e._itl_chunks]
    busy = np.array([e.busy_s for e in engines])
    st = {k: sum(c.stats[k] for c in caches) for k in ("hit_blocks", "miss_blocks", "hbm_hit_blocks",
                                                        "dram_hit_blocks", "evictions")}
    lookups = st["hit_blocks"] + st["miss_blocks"]
    computed = sum(e.prefill_tokens_computed for e in engines)
    reqs = [len(e.finished) for e in engines]
    out = summarize(records, float(busy.mean()), itl)
    per_request = out.pop("per_request")
    prompt_total = counters["prompt_total"]
    out.update(
        {
            "hit_rate": st["hit_blocks"] / lookups if lookups else 0.0,
            "hbm_hit_rate": st["hbm_hit_blocks"] / lookups if lookups else 0.0,
            "dram_hit_rate": st["dram_hit_blocks"] / lookups if lookups else 0.0,
            "token_hit_rate": 1.0 - computed / prompt_total if prompt_total else 0.0,
            "per_instance_hit_rate": [c.hit_rate for c in caches],
            "load_imbalance": float(busy.max() / busy.mean()) if busy.mean() > 0 else 1.0,
            "requests_per_instance": reqs,
            "max_instance_share": max(reqs) / max(1, sum(reqs)),
            "busy_s_per_instance": busy.tolist(),
            "prefill_tokens_total": prompt_total,
            "prefill_tokens_computed": computed,
            "evictions": st["evictions"],
            "dram_loaded_blocks": counters["dram_loaded"],
            "dram_recomputed_blocks": counters["dram_recomputed"],
            "per_request": per_request,
        }
    )
    return out
