"""单实例推理引擎模拟：continuous batching + chunked prefill，按"步"推进。

每一步（一次前向）：
1. 准入：按到达顺序（跳过还在从 DRAM 搬 KV、尚未就绪的请求）放进运行集合，直到 KV 显存或 max_num_seqs 用满。
   KV 预留两种模式（EngineConfig.kv_reserve）：
   - "oracle"（默认）：准入时按 (prompt - 命中前缀) + output 一次性预留。注意这用到了真实输出长度——
     真实调度器拿不到，是 oracle；好处是永远不需要抢占
   - "incremental"（vLLM 式）：准入只按 prompt 分配，decode 每步按需增长；显存不够时抢占最后准入的序列
     （recompute：释放它的 KV，放回队头，之后连同已生成的 token 重新 prefill）
   命中前缀缓存的 token 与缓存**物理共享**，不重复计入本请求的 KV（own_cached_kv=True 的请求除外：
   例如 P/D 分离时 decode 侧收到的 prompt KV 归该请求独占，照常计入）
2. 组 batch：
   - chunked_prefill=True（Sarathi-Serve / vLLM V1 风格）：所有 decode 序列各出 1 个 token，
     剩下的 token 预算（max_batch_tokens - n_decode）按 FIFO 切给 prefill 块
   - chunked_prefill=False（经典 prefill 优先）：只要有待 prefill 的请求，这一步只做 prefill（整条 prompt，
     多条能塞进预算就一起做），decode 全部暂停 —— 这就是长 prompt 造成 decode 抖动的来源
3. 步时间 = roofline（见 model.step_time_s）：GEMM 按 gemm_dtype 的峰值、注意力按 attn_dtype 的峰值计时；
   字节 = 本步触达的权重（MoE 按专家粒度）+ 读 KV + 写新 KV；可选通信项（tp / ep）
4. prefill 完成的那一步末尾吐出第 1 个 token（TTFT）；之后每个 decode 步吐 1 个 token

简化：忽略 CUDA graph / kernel launch 开销（可用 step_overhead_s 补一个常数）、采样与调度 CPU 开销。
"""

from __future__ import annotations

import heapq
import math
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .hardware import GPU, IB_GBPS_PER_GPU, peak_tflops
from .model import (
    ModelConfig,
    attn_flops_per_pair,
    body_active_params,
    ep_comm_time_s,
    kv_bytes_per_token,
    kv_capacity_tokens,
    tp_comm_time_s,
    weight_bytes_touched,
)


@dataclass
class Request:
    req_id: int
    arrival_s: float
    prompt_tokens: int
    cached_tokens: int  # 已命中前缀缓存的 token，不需要重新 prefill
    output_tokens: int
    session_id: int = -1
    load_s: float = 0.0  # 扩展：命中的 KV 需要先从 DRAM 搬回 HBM 的时间，就绪前不参与调度
    own_cached_kv: bool = False  # 扩展：cached_tokens 的 KV 归本请求独占（例如 P/D 分离时从 prefill 传来的 KV），计入显存


@dataclass
class EngineConfig:
    gpu: GPU
    model: ModelConfig
    n_gpus: int = 8
    max_batch_tokens: int = 8192  # 每步 token 预算（prefill chunk + decode）
    chunked_prefill: bool = True
    kv_capacity_tokens: int | None = None  # None 时按显存推算
    # ---- 以下是库扩展字段（契约之外，都有默认值）----
    max_num_seqs: int = 256  # 同时在跑的最大序列数（vLLM 同名参数）
    mfu: float = 0.6  # 可达算力利用率
    mbu: float = 0.8  # 可达带宽利用率
    weight_bytes: float = 2.0  # 权重每参数字节（BF16=2，FP8=1）
    kv_dtype_bytes: int = 2
    mem_frac: float = 0.9  # 显存里可用于权重+KV 的比例
    step_overhead_s: float = 0.0  # 每步固定开销（调度 / launch），默认不计
    gemm_dtype: str | None = None  # GEMM 计算精度："bf16" | "fp8"；None = 跟随 weight_bytes（<=1 字节即 fp8）
    attn_dtype: str = "bf16"  # 注意力核心的计算精度（DeepSeek 推理系统里 MLA 核心用 BF16）
    kv_reserve: str = "oracle"  # "oracle" | "incremental"（见模块说明）
    tp: int | None = None  # None = 理想切分、不计通信（旧行为）；设为整数则计 TP all-reduce
    ep: int = 1  # MoE 专家并行度；>1 时计 dispatch/combine 通信
    dp_attention: bool = False  # 注意力按请求做数据并行（MLA 常用）：不做 TP all-reduce，latent KV 不复制
    gpus_per_node: int = 8
    ib_gbps: float = IB_GBPS_PER_GPU  # 每卡跨节点带宽（单向 GB/s）
    comm_overlap: float = 0.0  # 通信可与计算重叠的比例（DeepSeek 双 micro-batch 重叠可接近 1）
    expert_skew: float = 0.0  # MoE 专家热度偏斜（0 = 均匀路由）

    def resolved_gemm_dtype(self) -> str:
        if self.gemm_dtype is not None:
            return self.gemm_dtype
        return "fp8" if self.weight_bytes <= 1 else "bf16"


class _Seq:
    __slots__ = ("req", "cur_prompt", "todo", "done", "cached", "emitted", "first_token_s", "run_start",
                 "runs", "base_alloc", "admit_no", "active", "preemptions")

    def __init__(self, req: Request, admit_no: int):
        self.req = req
        self.cur_prompt = req.prompt_tokens  # 被抢占重算后 = 原 prompt + 已生成的 token
        # 最后一个 prompt token 总要算（要出 logits），所以命中最多 prompt-1 个
        self.cached = max(0, min(req.cached_tokens, req.prompt_tokens - 1))
        self.todo = self.cur_prompt - self.cached  # 还要 prefill 的 token
        self.done = self.cached  # 已经在 KV 里的 token
        self.emitted = 0
        self.first_token_s = math.nan
        self.run_start = -1  # 当前这段 decode 从第几个 decode 步开始
        self.runs: list[tuple[int, float]] = []  # 吐 token 的记录：(-1, 时刻) 或 (起始 decode 步, token 数)
        self.base_alloc = 0
        self.admit_no = admit_no
        self.active = True
        self.preemptions = 0


def resolve_kv_capacity(cfg: EngineConfig) -> int:
    """EngineConfig 的 KV 容量（token 数）：显式给定就用它，否则按显存推算。"""
    if cfg.kv_capacity_tokens is not None:
        return int(cfg.kv_capacity_tokens)
    cap = kv_capacity_tokens(cfg.gpu, cfg.model, cfg.n_gpus, cfg.weight_bytes, cfg.kv_dtype_bytes, cfg.mem_frac)
    if cap <= 0:
        raise ValueError(
            f"{cfg.model.name} 的权重放不进 {cfg.n_gpus}×{cfg.gpu.name}"
            f"（mem_frac={cfg.mem_frac}, weight_bytes={cfg.weight_bytes}），请加卡或降低权重精度"
        )
    return cap


class Engine:
    """可逐步推进的单实例引擎（simulate、router、pd 共用）。"""

    def __init__(self, cfg: EngineConfig):
        if cfg.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens 必须 >= 1")
        if cfg.kv_reserve not in ("oracle", "incremental"):
            raise ValueError("kv_reserve 必须是 'oracle' 或 'incremental'")
        if cfg.tp is not None and not 1 <= cfg.tp <= cfg.n_gpus:
            raise ValueError("tp 必须在 [1, n_gpus]")
        self.cfg = cfg
        self.kv_capacity = resolve_kv_capacity(cfg)
        self.max_seqs = max(1, min(cfg.max_num_seqs, cfg.max_batch_tokens))
        self.t = 0.0
        self.waiting: deque[_Seq | Request] = deque()
        self.prefilling: deque[_Seq] = deque()
        self.decoding: list = []  # 堆：(finish_d, admit_no, _Seq)；被抢占的序列 active=False，懒删除
        self.n_dec = 0
        self.ctx_sum = 0  # 所有 decode 序列当前上下文长度之和
        self.n_decode_steps = 0  # 已完成的 decode 步数
        self.dec_end: list[float] = []  # 每个 decode 步的结束时刻
        self.kv_used = 0
        self.busy_s = 0.0
        self.n_steps = 0
        self.prefill_tokens_computed = 0
        self.output_tokens_emitted = 0
        self.preemptions = 0
        self.recomputed_tokens = 0
        self.kv_peak = 0  # KV 占用的峰值（token），用来检查是否超过容量
        self.finished: list[dict] = []
        self.prefill_done_log: list[tuple[int, float]] = []  # (req_id, 首 token 时刻)，router 用来在 prefill 完成时写缓存
        self._itl_chunks: list[np.ndarray] = []
        self._admit_counter = 0
        self._warned_ctx = False
        m = cfg.model
        self._kvb = kv_bytes_per_token(m, cfg.kv_dtype_bytes)
        self._body2 = 2.0 * body_active_params(m)
        self._lm = 2.0 * m.vocab_size * m.d_model
        self._coef_pre = attn_flops_per_pair(m, "prefill")
        self._coef_dec = attn_flops_per_pair(m, "decode")
        self._gemm_dtype = cfg.resolved_gemm_dtype()
        tp = cfg.tp or 1
        self._kv_rep = tp if (m.attn == "mla" and tp > 1 and not cfg.dp_attention) else 1
        # 步时间的分母预先算好（与 model.step_time_s 完全等价，只是省掉每步的函数调用）
        self._gemm_den = cfg.n_gpus * peak_tflops(cfg.gpu, self._gemm_dtype) * 1e12 * cfg.mfu
        self._attn_den = cfg.n_gpus * peak_tflops(cfg.gpu, cfg.attn_dtype) * 1e12 * cfg.mfu
        self._mem_den = cfg.n_gpus * cfg.gpu.hbm_tbps * 1e12 * cfg.mbu
        if not m.is_moe:  # dense：触达权重 = 常数 + 每个 token 一行 embedding
            w1 = weight_bytes_touched(m, 1, cfg.weight_bytes)
            self._w_row = weight_bytes_touched(m, 2, cfg.weight_bytes) - w1
            self._w_const = w1 - self._w_row
        self._has_comm = (cfg.tp is not None and cfg.tp > 1 and not cfg.dp_attention) or cfg.ep > 1

    # ------------------------------------------------------------------ 外部接口

    def submit(self, req: Request) -> None:
        if req.prompt_tokens < 1 or req.output_tokens < 1:
            raise ValueError(f"请求 {req.req_id}: prompt_tokens 与 output_tokens 必须 >= 1")
        mc = self.cfg.model.max_context
        if mc and req.prompt_tokens + req.output_tokens > mc and not self._warned_ctx:
            warnings.warn(f"有请求的上下文超过 {self.cfg.model.name} 的 max_context={mc}（模拟照常进行）", stacklevel=2)
            self._warned_ctx = True
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.prefilling or self.n_dec)

    @property
    def n_inflight(self) -> int:
        """已提交但未完成的请求数（排队 + 运行）。"""
        return len(self.waiting) + len(self.prefilling) + self.n_dec

    def inflight_tokens(self, est_output: float = 300.0) -> float:
        """在途工作量（token）：剩余 prefill + 预计剩余 decode（输出长度未知，用 est_output 估计）。"""
        total = 0.0
        for w in self.waiting:
            if isinstance(w, Request):
                total += w.prompt_tokens - min(w.cached_tokens, w.prompt_tokens - 1) + est_output
            else:
                total += w.todo + max(0.0, est_output - w.emitted)
        for s in self.prefilling:
            total += s.todo + max(0.0, est_output - s.emitted)
        for _, no, s in self.decoding:
            if s.active and s.admit_no == no:
                total += max(0.0, est_output - s.emitted - (self.n_decode_steps - s.run_start))
        return total

    def next_step_time(self) -> float:
        if self.prefilling or self.n_dec:
            return self.t
        if self.waiting:
            return max(self.t, min(_ready(w) for w in self.waiting))
        return math.inf

    def step(self) -> list[dict]:
        """执行一步前向，返回本步完成的请求记录。"""
        if not (self.prefilling or self.n_dec):
            if not self.waiting:
                return []
            self.t = max(self.t, min(_ready(w) for w in self.waiting))
        self._admit()
        chunks, n_dec = self._build_batch()
        if n_dec and self.cfg.kv_reserve == "incremental":
            while self.kv_used + n_dec > self.kv_capacity and self._preempt_one():
                chunks, n_dec = self._build_batch()
        if not chunks and n_dec == 0:
            return []
        dt = self._step_cost(chunks, n_dec)
        t_end = self.t + dt
        done: list[dict] = []
        if n_dec:
            self._apply_decode(n_dec, t_end, done)
        self._apply_prefill(chunks, t_end, done)
        self.t = t_end
        self.busy_s += dt
        self.n_steps += 1
        self.finished.extend(done)
        return done

    def run(self) -> None:
        while self.has_work():
            self.step()

    # ------------------------------------------------------------------ 调度

    def _need(self, s: _Seq) -> int:
        """准入时要分配的 KV（token）：命中的前缀与缓存共享，不重复计。"""
        own_prompt = s.cur_prompt - (0 if s.req.own_cached_kv else s.cached)
        if self.cfg.kv_reserve == "oracle":
            return own_prompt + (s.req.output_tokens - s.emitted)
        return own_prompt

    def _admit(self) -> None:
        idx = 0
        while idx < len(self.waiting):
            w = self.waiting[idx]
            if _ready(w) > self.t:  # 还在搬 KV，跳过但不阻塞后面已就绪的请求
                idx += 1
                continue
            running = len(self.prefilling) + self.n_dec
            if running >= self.max_seqs:
                break
            s = w if isinstance(w, _Seq) else _Seq(w, 0)
            need = self._need(s)
            # 放不下就等；但如果引擎是空的，超大请求也硬放进去，避免死锁
            if self.kv_used + need > self.kv_capacity and running > 0:
                break
            del self.waiting[idx]
            self._admit_counter += 1
            s.admit_no = self._admit_counter
            s.base_alloc = need
            s.active = True
            self.kv_used += need
            self.kv_peak = max(self.kv_peak, self.kv_used)
            self.prefilling.append(s)

    def _preempt_one(self) -> bool:
        """抢占最后准入的序列（vLLM 的 recompute 策略）。只剩一个序列时不抢占。"""
        live = list(self.prefilling) + [s for _, no, s in self.decoding if s.active and s.admit_no == no]
        if len(live) <= 1:
            return False
        victim = max(live, key=lambda s: s.admit_no)
        if victim in self.prefilling:
            self.prefilling.remove(victim)
            self.kv_used -= victim.base_alloc
        else:
            steps = self.n_decode_steps - victim.run_start
            victim.runs.append((victim.run_start, steps))
            victim.emitted += steps
            victim.active = False
            self.n_dec -= 1
            self.ctx_sum -= victim.cur_prompt + 1 + steps
            self.kv_used -= victim.base_alloc + steps
        # 重算：已生成的 token 并入 prompt，从命中前缀之后重新 prefill
        self.recomputed_tokens += victim.done - victim.cached
        if victim.req.own_cached_kv:  # 独占的"已缓存"KV 也被释放了，只能整段重算
            victim.cached = 0
        victim.cur_prompt = victim.req.prompt_tokens + victim.emitted
        victim.todo = victim.cur_prompt - victim.cached
        victim.done = victim.cached
        victim.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(victim)
        return True

    def _build_batch(self) -> tuple[list[tuple[_Seq, int]], int]:
        budget = self.cfg.max_batch_tokens
        chunks: list[tuple[_Seq, int]] = []
        if self.cfg.chunked_prefill:
            n_dec = self.n_dec  # decode 优先，每个序列 1 个 token
            budget -= n_dec
            for s in self.prefilling:
                if budget <= 0:
                    break
                c = min(s.todo, budget)
                chunks.append((s, c))
                budget -= c
            return chunks, n_dec
        if not self.prefilling:
            return chunks, self.n_dec
        for s in self.prefilling:  # prefill 优先：整条 prompt，decode 暂停
            if chunks and s.todo > budget:
                break
            chunks.append((s, s.todo))
            budget -= s.todo
            if budget <= 0:
                break
        return chunks, 0

    def _step_cost(self, chunks: list[tuple[_Seq, int]], n_dec: int) -> float:
        gemm, attn, kv_read, n_tok = 0.0, 0.0, 0, n_dec
        for s, c in chunks:
            gemm += self._body2 * c + self._lm  # lm_head 只算每块最后一个位置
            attn += self._coef_pre * (c * s.done + c * (c + 1) / 2.0)
            kv_read += s.done  # 读已有的 KV（缓存命中的前缀 + 之前的 chunk）
            n_tok += c
        if n_dec:
            gemm += n_dec * (self._body2 + self._lm)
            attn += self._coef_dec * self.ctx_sum
            kv_read += self.ctx_sum
        cfg = self.cfg
        if cfg.model.is_moe:
            w = weight_bytes_touched(cfg.model, n_tok, cfg.weight_bytes, expert_skew=cfg.expert_skew)
        else:
            w = self._w_const + self._w_row * min(n_tok, cfg.model.vocab_size or n_tok)
        bytes_ = w + (kv_read * self._kv_rep + n_tok) * self._kvb
        core = max(gemm / self._gemm_den + attn / self._attn_den, bytes_ / self._mem_den)  # = model.step_time_s
        if self._has_comm:
            comm = self._comm_s(n_tok)
            core = max(core, comm * cfg.comm_overlap) + comm * (1.0 - cfg.comm_overlap)
        return core + cfg.step_overhead_s

    def _comm_s(self, n_tok: int) -> float:
        cfg = self.cfg
        comm = 0.0
        if cfg.tp is not None and cfg.tp > 1 and not cfg.dp_attention:
            comm += tp_comm_time_s(cfg.gpu, cfg.model, n_tok, cfg.tp)
        if cfg.ep > 1:
            comm += ep_comm_time_s(cfg.gpu, cfg.model, n_tok, cfg.ep, gpus_per_node=cfg.gpus_per_node,
                                   ib_gbps=cfg.ib_gbps)
        return comm

    def _apply_decode(self, n_dec: int, t_end: float, done: list[dict]) -> None:
        self.n_decode_steps += 1
        self.dec_end.append(t_end)
        self.ctx_sum += n_dec
        self.output_tokens_emitted += n_dec
        if self.cfg.kv_reserve == "incremental":
            self.kv_used += n_dec  # 每个序列这一步写入 1 个 token 的 KV
            self.kv_peak = max(self.kv_peak, self.kv_used)
        while self.decoding and self.decoding[0][0] <= self.n_decode_steps:
            _, no, s = heapq.heappop(self.decoding)
            if not (s.active and s.admit_no == no):
                continue  # 已被抢占（或是抢占前留下的旧条目）
            steps = self.n_decode_steps - s.run_start
            s.runs.append((s.run_start, steps))
            s.emitted += steps
            self.n_dec -= 1
            self.ctx_sum -= s.cur_prompt + 1 + steps
            if self.cfg.kv_reserve == "incremental":
                self.kv_used -= steps
            done.append(self._finish(s, t_end))

    def _apply_prefill(self, chunks: list[tuple[_Seq, int]], t_end: float, done: list[dict]) -> None:
        for s, c in chunks:
            s.todo -= c
            s.done += c
            self.prefill_tokens_computed += c
        while self.prefilling and self.prefilling[0].todo == 0:  # FIFO 分配，完成的一定在队头
            s = self.prefilling.popleft()
            if s.emitted == 0:
                s.first_token_s = t_end
                self.prefill_done_log.append((s.req.req_id, t_end))
            s.runs.append((-1, t_end))
            s.emitted += 1
            self.output_tokens_emitted += 1
            if s.emitted >= s.req.output_tokens:
                done.append(self._finish(s, t_end))
                continue
            s.run_start = self.n_decode_steps
            heapq.heappush(self.decoding, (s.run_start + s.req.output_tokens - s.emitted, s.admit_no, s))
            self.n_dec += 1
            self.ctx_sum += s.cur_prompt + 1

    def _finish(self, s: _Seq, t_end: float) -> dict:
        self.kv_used -= s.base_alloc  # incremental 模式下 decode 增长的部分已在 _apply_decode 里释放
        r = s.req
        times: list[float] = []
        for start, n in s.runs:
            if start < 0:
                times.append(n)
            else:
                times.extend(self.dec_end[start : start + int(n)])
        gaps = np.diff(np.asarray(times)) if len(times) > 1 else np.array([])
        if len(gaps):
            self._itl_chunks.append(gaps)
        n_gaps = r.output_tokens - 1
        return {
            "req_id": r.req_id,
            "session_id": r.session_id,
            "arrival_s": r.arrival_s,
            "first_token_s": s.first_token_s,
            "finish_s": t_end,
            "ttft_s": s.first_token_s - r.arrival_s,
            "tpot_s": (t_end - s.first_token_s) / n_gaps if n_gaps > 0 else 0.0,  # 只有 1 个输出 token 时记 0，不计入分位数
            "max_itl_s": float(gaps.max()) if len(gaps) else 0.0,
            "prompt_tokens": r.prompt_tokens,
            "cached_tokens": s.cached,
            "output_tokens": r.output_tokens,
            "preemptions": s.preemptions,
        }

    # ------------------------------------------------------------------ 汇总

    def summary(self) -> dict:
        return summarize(self.finished, self.busy_s, self._itl_chunks, extra={
            "n_steps": self.n_steps,
            "kv_capacity_tokens": self.kv_capacity,
            "prefill_tokens_computed": self.prefill_tokens_computed,
            "output_tokens": self.output_tokens_emitted,
            "preemptions": self.preemptions,
            "recomputed_tokens": self.recomputed_tokens,
        })


def _ready(w: _Seq | Request) -> float:
    req = w.req if isinstance(w, _Seq) else w
    return req.arrival_s + req.load_s


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else math.nan


def summarize(records: list[dict], busy_s: float, itl_chunks: list[np.ndarray], extra: dict | None = None) -> dict:
    """把请求记录汇总成 TTFT / TPOT / ITL 分位数、吞吐与忙碌比例。"""
    if not records:
        raise ValueError("没有完成的请求")
    ttft = np.array([r["ttft_s"] for r in records])
    tpot = np.array([r["tpot_s"] for r in records if r["output_tokens"] > 1])
    itl = np.concatenate(itl_chunks) if itl_chunks else np.array([])
    start = min(r["arrival_s"] for r in records)
    end = max(r["finish_s"] for r in records)
    makespan = max(end - start, 1e-12)
    out_tokens = sum(r["output_tokens"] for r in records)
    result = {
        "ttft_p50": _pct(ttft, 50),
        "ttft_p99": _pct(ttft, 99),
        "tpot_p50": _pct(tpot, 50),
        "tpot_p99": _pct(tpot, 99),
        "itl_p50": _pct(itl, 50),
        "itl_p99": _pct(itl, 99),
        "itl_max": float(itl.max()) if len(itl) else math.nan,
        "throughput_tok_s": out_tokens / makespan,
        "busy_frac": min(1.0, busy_s / makespan),
        "makespan_s": makespan,
        "n_requests": len(records),
    }
    if extra:
        result.update(extra)
    result["per_request"] = sorted(records, key=lambda r: r["req_id"])
    return result


def simulate(requests: list[Request], cfg: EngineConfig) -> dict:
    """模拟一个推理实例处理一批（开环到达的）请求。

    返回 {"ttft_p50","ttft_p99","tpot_p50","tpot_p99","throughput_tok_s","busy_frac","per_request": [...]}，
    另附 itl_p50/itl_p99/itl_max（逐 token 间隔）、makespan_s、n_steps、kv_capacity_tokens、
    prefill_tokens_computed、output_tokens、preemptions、recomputed_tokens。throughput_tok_s 是输出 token 吞吐。
    """
    engine = Engine(cfg)
    for req in sorted(requests, key=lambda r: (r.arrival_s, r.req_id)):
        engine.submit(req)
    engine.run()
    return engine.summary()
