"""LLMEngine：把模型、分页 KV、前缀缓存、调度器串起来的同步推理引擎（greedy 解码）。

一步（step）= 调度（进度前移、登记本步会算满的块）→ 拼扁平 batch（token、位置、KV 写入槽位）
→ 一次前向 → 算到序列末尾的请求采样一个 token。
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .kv_cache import BlockAllocator, PagedKVCache
from .model import SeqChunk, TinyTransformer
from .prefix_cache import PrefixCache, root_hash
from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig


@dataclass
class EngineConfig:
    num_blocks: int = 256
    block_size: int = 16
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 16
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_pinned_frac: float = 0.0      # 为 pin 预留的块比例；默认 0 = 不能 pin，要用时显式打开
    long_prefill_token_threshold: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    prompt_tokens: int = 0              # 所有请求的 prompt token 总数
    prefill_tokens_computed: int = 0    # 真正送进模型的 prefill token（含抢占后重算）
    decode_tokens: int = 0
    lookup_tokens: int = 0              # 每次接纳请求（含抢占后重新接纳）时查缓存的 token 总数
    cached_tokens: int = 0              # 其中命中前缀缓存、跳过计算的 token
    preemptions: int = 0
    aborted: int = 0

    @property
    def prefix_hit_rate(self) -> float:
        return self.cached_tokens / self.lookup_tokens if self.lookup_tokens else 0.0


@dataclass
class BatchEntry:
    request_id: str
    session_id: int | None
    num_new_tokens: int


@dataclass
class StepRecord:
    """一步里谁在 batch 里（用于证明 continuous batching / 计算与工具等待重叠）。"""
    index: int
    entries: list[BatchEntry] = field(default_factory=list)


class LLMEngine:
    def __init__(self, model: TinyTransformer, config: EngineConfig | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 allocator_cls: type[BlockAllocator] = BlockAllocator):
        config = config if config is not None else EngineConfig()
        self.model = model
        self.config = config
        self.clock = clock
        # 前向算完、采样与记录完成时间之前调用（参数：本步实算的 token 数）。
        # 虚拟时钟用它把时间推进到本步结束，这样 finish_time / first_token_time / 工具期间 pin 的起点都包含本步耗时。
        self.on_model_done: Callable[[int], None] | None = None
        self.kv = PagedKVCache(model.cfg, config.num_blocks, config.block_size)
        self.allocator = allocator_cls(config.num_blocks)
        self.prefix_cache = PrefixCache(self.allocator, enabled=config.enable_prefix_caching)
        self.max_pinned_blocks = int(config.num_blocks * config.max_pinned_frac)
        self.scheduler = Scheduler(
            SchedulerConfig(config.block_size, config.max_num_batched_tokens, config.max_num_seqs,
                            config.enable_chunked_prefill, self.max_pinned_blocks,
                            config.long_prefill_token_threshold),
            self.allocator, self.prefix_cache, clock)
        self.stats = EngineStats()
        # 所有请求（含已完成的），方便测试和调试检查；长时间运行时用 pop_request 回收
        self.requests: dict[str, Request] = {}
        self.last_step: StepRecord | None = None
        self._ids = itertools.count()

    # ---------------- 对外接口 ----------------
    def validate_request(self, prompt_token_ids: list[int], max_tokens: int, pin_seconds: float = 0.0) -> None:
        """只检查、不改状态（异步前端在把请求交给引擎线程之前调用）。"""
        cfg = self.config
        if not prompt_token_ids or max_tokens < 1:
            raise ValueError("prompt must be non-empty and max_tokens >= 1")
        # 最后一个生成的 token 不会再喂回模型，所以一个请求最多有 prompt + max_tokens - 1 个 token 的 KV
        max_kv_tokens = len(prompt_token_ids) + max_tokens - 1
        need = math.ceil(max_kv_tokens / cfg.block_size)
        # 单个请求必须能在"未被 pin 预留的块"里放下，否则抢占再多也跑不完
        if need > cfg.num_blocks - self.max_pinned_blocks:
            raise ValueError(f"request needs {need} blocks but only {cfg.num_blocks - self.max_pinned_blocks} "
                             f"are available ({self.max_pinned_blocks} reserved for pins)")
        # 关闭 chunked prefill 时，被抢占的请求要把 prompt + 已生成 token 一步重算完，
        # 最坏情况是 max_kv_tokens 个 token，所以按它检查而不是只看 prompt
        if not cfg.enable_chunked_prefill and max_kv_tokens > cfg.max_num_batched_tokens:
            raise ValueError("without chunked prefill, prompt + max_tokens - 1 must fit one step's token budget "
                             "(a preempted request is recomputed in a single step)")
        if pin_seconds > 0 and self.max_pinned_blocks == 0:
            raise ValueError("pin_seconds > 0 needs EngineConfig.max_pinned_frac > 0")

    def add_request(self, prompt_token_ids: list[int], max_tokens: int, request_id: str | None = None,
                    pin_seconds: float = 0.0, session_id: int | None = None, cache_salt: int = 0) -> str:
        self.validate_request(prompt_token_ids, max_tokens, pin_seconds)
        rid = request_id if request_id is not None else f"req-{next(self._ids)}"
        if rid in self.requests:
            raise ValueError(f"duplicate request id {rid}")
        req = Request(rid, list(prompt_token_ids), max_tokens, self.clock(), pin_seconds, session_id,
                      root_hash(cache_salt))
        self.requests[rid] = req
        self.scheduler.add(req)
        self.stats.prompt_tokens += len(prompt_token_ids)
        return rid

    def abort_request(self, request_id: str) -> bool:
        """中止一个未完成的请求：从调度队列移除、释放块。已完成或不存在时返回 False。"""
        req = self.requests.get(request_id)
        if req is None or not self.scheduler.abort(req):
            return False
        self.stats.aborted += 1
        return True

    def pop_request(self, request_id: str) -> Request:
        """取走一个已结束（完成或中止）的请求，不再保留在 self.requests 里。"""
        req = self.requests[request_id]
        if req.status not in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            raise ValueError(f"request {request_id} is still {req.status.value}")
        return self.requests.pop(request_id)

    def end_session(self, session_id: int) -> None:
        """harness 告诉引擎：这个 agent 会话结束了，它的块可以优先淘汰（会话感知策略用）。"""
        self.allocator.ended_sessions.add(session_id)

    def reset_prefix_cache(self) -> bool:
        """清空前缀缓存。和 vLLM 一样，只有在没有块被占用时才执行（否则返回 False）。"""
        if any(b.ref_count for b in self.allocator.blocks):
            return False
        self.prefix_cache.reset()
        return True

    def update_weights(self, model: TinyTransformer, reset_prefix_cache: bool = True) -> None:
        """换权重（RL 里每个训练步之后的权重同步）。

        前缀缓存里的 KV 是旧权重算的；块哈希只看 token，不看权重版本。不 reset 的话，
        新请求会命中旧 KV，输出既不是旧策略也不是新策略。这里要求引擎空闲（先 drain 或 abort），
        in-flight 更新（带着旧 KV 继续生成）的取舍见 modules/06_long_tail_async。
        """
        if self.has_unfinished():
            raise RuntimeError("drain or abort in-flight requests before updating weights")
        if model.cfg.__class__ is not self.model.cfg.__class__ or \
                (model.cfg.n_layers, model.cfg.n_kv_heads, model.cfg.head_dim) != \
                (self.model.cfg.n_layers, self.model.cfg.n_kv_heads, self.model.cfg.head_dim):
            raise ValueError("new weights must have the same KV layout")
        self.model = model
        if reset_prefix_cache:
            self.reset_prefix_cache()

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[Request]:
        """执行一步，返回本步完成的请求。"""
        batch = self.scheduler.schedule()
        self.stats.preemptions += len(batch.preempted)
        if not batch.items:
            self.last_step = None
            return []
        logits = self._run_model(batch.items)
        self.stats.steps += 1
        self.stats.prefill_tokens_computed += batch.num_prefill_tokens
        self.stats.decode_tokens += batch.num_decode_tokens
        self.stats.cached_tokens += batch.num_cached_tokens
        self.stats.lookup_tokens += batch.num_lookup_tokens
        self.last_step = StepRecord(self.stats.steps, [BatchEntry(r.request_id, r.session_id, n)
                                                       for r, n in batch.items])

        if self.on_model_done is not None:
            self.on_model_done(batch.num_prefill_tokens + batch.num_decode_tokens)
        now = self.clock()
        finished: list[Request] = []
        for (req, _), row in zip(batch.items, logits):
            if req.num_computed_tokens < req.num_tokens:
                continue                              # chunked prefill 还没算到末尾，不采样
            req.output_token_ids.append(int(np.argmax(row)))   # greedy
            if req.first_token_time is None:
                req.first_token_time = now
            if req.is_done:
                self.scheduler.finish(req)
                finished.append(req)
        return finished

    def run_until_done(self, max_steps: int = 1_000_000) -> None:
        for _ in range(max_steps):
            if not self.has_unfinished():
                return
            self.step()
        raise RuntimeError("engine did not finish within max_steps")

    def generate(self, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
        ids = [self.add_request(p, max_tokens) for p in prompts]
        self.run_until_done()
        return [self.requests[i].output_token_ids for i in ids]

    # ---------------- 拼 batch、跑模型 ----------------
    def _run_model(self, items: list[tuple[Request, int]]) -> np.ndarray:
        bs = self.config.block_size
        toks, poss, slots, chunks = [], [], [], []
        start = 0
        for req, n_new in items:
            p0 = req.num_computed_tokens - n_new      # 调度时进度已经前移过了
            pos = np.arange(p0, p0 + n_new)
            table = np.asarray(req.block_ids, dtype=np.int64)
            toks.append(np.asarray(req.all_token_ids[p0:p0 + n_new], dtype=np.int64))
            poss.append(pos)
            slots.append(table[pos // bs] * bs + pos % bs)
            chunks.append(SeqChunk(start, n_new, p0 + n_new, req.block_ids))
            start += n_new
        return self.model.forward_paged(np.concatenate(toks), np.concatenate(poss),
                                        np.concatenate(slots), chunks, self.kv)
