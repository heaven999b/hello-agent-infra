"""Continuous batching 调度器：每步 token 预算、chunked prefill、按 recompute 方式抢占。

统一视角（与 vLLM V1 调度器同思路）：不区分"prefill 阶段 / decode 阶段"，
每个请求只有一个进度 num_computed_tokens，每步给它分配 num_new_tokens 个 token 去算：
- decode：num_new_tokens = 1；
- prefill：num_new_tokens = 剩余 prompt，受本步预算限制时切成多块（chunked prefill）；
- 抢占后重算：num_computed_tokens 归零，prompt + 已生成的 token 全部当 prefill 再算一遍。

与 vLLM V1 一样，num_computed_tokens 在**调度时**就前移（"本步结束后会有 KV 的 token 数"），
并且在调度时就把本步会算满的块登记进前缀缓存。所以同一步里后接纳的请求可以直接命中
前面请求本步才算的块（例如同一 prompt 的 G 份采样在一步里只算一遍前缀）。
这之所以安全，是因为模型每层先写完本步所有 K/V 再做注意力（见 model.forward_paged）。

每步的顺序：
1. 先给 running 里的请求分配（按进入 running 的先后）；需要新块而块不够时，
   抢占 running 队尾（最晚进入、优先级最低）的请求，释放它的块，放回 waiting 队首；
2. 本步没有发生抢占时，再按 FCFS 从 waiting 接纳新请求：先查前缀缓存，只为未命中的部分分配块。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .kv_cache import BlockAllocator
from .prefix_cache import ROOT_HASH, PrefixCache, extend_block_hashes


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(eq=False)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float
    pin_seconds: float = 0.0          # 结束后把它的缓存块 pin 多久（等工具时保留 KV）
    session_id: int | None = None     # 属于哪个 agent 会话（会话感知淘汰用）
    root: int = ROOT_HASH             # 哈希链的根（混入租户 salt / 权重版本）
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0      # 本步结束后会有 KV 的 token 数（调度时前移）
    block_ids: list[int] = field(default_factory=list)       # block table
    block_hashes: list[int] = field(default_factory=list)    # 满块的哈希链（按需补算）
    num_registered_blocks: int = 0    # 前多少个块已登记进前缀缓存
    num_cached_tokens: int = -1       # 第一次被调度时前缀缓存命中的 token 数
    num_preemptions: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_done(self) -> bool:
        return len(self.output_token_ids) >= self.max_tokens


@dataclass
class SchedulerConfig:
    block_size: int = 16
    max_num_batched_tokens: int = 256   # 每步 token 预算（prefill 块 + decode 一起算）
    max_num_seqs: int = 16              # 同时 running 的请求上限
    enable_chunked_prefill: bool = True
    max_pinned_blocks: int = 0          # pin 的块数上限（防止 pin 太多把活跃请求饿死）
    long_prefill_token_threshold: int = 0   # >0 时单个请求一步最多算这么多 token（vLLM V1 同名参数）


@dataclass
class ScheduledBatch:
    items: list[tuple[Request, int]]    # (请求, 本步新算的 token 数)；请求的 num_computed_tokens 已前移
    preempted: list[Request]
    num_prefill_tokens: int             # 其中 prefill（含抢占后重算）的 token 数
    num_decode_tokens: int
    num_cached_tokens: int              # 本步新接纳的请求命中前缀缓存的 token 数
    num_lookup_tokens: int              # 本步新接纳的请求的总 token 数（命中率的分母）


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, allocator: BlockAllocator, prefix_cache: PrefixCache,
                 clock: Callable[[], float]):
        self.cfg = cfg
        self.allocator = allocator
        self.prefix_cache = prefix_cache
        self.clock = clock
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ---------------- 调度一步 ----------------
    def schedule(self) -> ScheduledBatch:
        now = self.clock()
        budget = self.cfg.max_num_batched_tokens
        items: list[tuple[Request, int]] = []
        preempted: list[Request] = []
        n_decode = 0

        # 1) running：decode 或 继续 chunked prefill
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n_new = self._cap(req.num_tokens - req.num_computed_tokens, budget)
            if not self._grow(req, n_new, now, preempted):
                break                      # req 自己被抢占了（它必然是队尾），后面没有请求了
            n_decode += self._is_decode(req, n_new)
            self._advance(req, n_new)
            items.append((req, n_new))
            budget -= n_new
            i += 1

        # 2) waiting：没有抢占时才接纳新请求（有抢占说明块已经不够了）
        cached = looked = 0
        while not preempted and self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            req = self.waiting[0]
            n_new, n_hit = self._admit(req, budget, now)
            if n_new == 0:
                break                      # 块不够或预算不够：FCFS，后面的也不越过它
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            n_decode += self._is_decode(req, n_new)
            self._advance(req, n_new)
            items.append((req, n_new))
            budget -= n_new
            cached += n_hit
            looked += req.num_tokens

        n_total = sum(n for _, n in items)
        return ScheduledBatch(items, preempted, n_total - n_decode, n_decode, cached, looked)

    def _cap(self, n_remaining: int, budget: int) -> int:
        n = min(n_remaining, budget)
        th = self.cfg.long_prefill_token_threshold
        return th if 0 < th < n else n

    @staticmethod
    def _is_decode(req: Request, n_new: int) -> bool:
        """前移之前判断：只喂上一步刚采样出的那 1 个 token，才算 decode。"""
        return n_new == 1 and bool(req.output_token_ids) and req.num_computed_tokens == req.num_tokens - 1

    def _advance(self, req: Request, n_new: int) -> None:
        """前移进度，并立刻登记本步会算满的块（本步所有 K/V 都在注意力之前写入，所以安全）。"""
        req.num_computed_tokens += n_new
        self._cache_full_blocks(req)
        if req.session_id is not None:
            self.allocator.mark_session(req.block_ids, req.session_id)

    def _grow(self, req: Request, n_new: int, now: float, preempted: list[Request]) -> bool:
        """保证 req 的 block table 放得下 num_computed + n_new 个 token；不够就从队尾抢占。"""
        need = math.ceil((req.num_computed_tokens + n_new) / self.cfg.block_size) - len(req.block_ids)
        while need > 0:
            new_ids = self.allocator.allocate(need, now)
            if new_ids is not None:
                req.block_ids.extend(new_ids)
                return True
            victim = self.running.pop()          # 优先级最低：最晚进入 running 的
            self._preempt(victim)
            preempted.append(victim)
            if victim is req:
                return False
        return True

    def _admit(self, req: Request, budget: int, now: float) -> tuple[int, int]:
        """尝试接纳一个 waiting 请求。返回 (本步新算 token 数, 前缀命中 token 数)；(0, 0) 表示接纳失败。"""
        bs = self.cfg.block_size
        # 全部命中时也要重算最后一个 token 才能拿到 logits，所以最多匹配 (num_tokens - 1) 个 token
        n_match_blocks = (req.num_tokens - 1) // bs
        extend_block_hashes(req.block_hashes, req.all_token_ids, bs, n_match_blocks, req.root)
        hit_ids = self.prefix_cache.lookup(req.block_hashes[:n_match_blocks])
        n_hit = len(hit_ids) * bs
        remaining = req.num_tokens - n_hit
        if self.cfg.enable_chunked_prefill:
            n_new = self._cap(remaining, budget)
        elif remaining <= budget:
            n_new = remaining
        else:
            return 0, 0
        need = math.ceil((n_hit + n_new) / bs) - len(hit_ids)
        # 命中的块可能正躺在空闲队列里：必须排除，否则会被当成"新块"分给同一个请求
        new_ids = self.allocator.allocate(need, now, exclude=hit_ids)
        if new_ids is None:
            return 0, 0
        self.allocator.touch(hit_ids)
        req.block_ids = hit_ids + new_ids
        req.num_computed_tokens = n_hit
        req.num_registered_blocks = len(hit_ids)
        if req.num_cached_tokens < 0:
            req.num_cached_tokens = n_hit
        return n_new, n_hit

    def _preempt(self, req: Request) -> None:
        """recompute 式抢占：释放全部块、进度归零，放回 waiting 队首，之后整段重算。"""
        self.allocator.free(req.block_ids, self.clock())
        req.block_ids = []
        req.num_computed_tokens = 0
        req.num_registered_blocks = 0
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.waiting.appendleft(req)

    def _cache_full_blocks(self, req: Request) -> None:
        """把"本步结束后已写满 KV"的块登记进前缀缓存。只能登记到 num_computed_tokens 为止。"""
        bs = self.cfg.block_size
        n_full = req.num_computed_tokens // bs
        if n_full <= req.num_registered_blocks:
            return
        extend_block_hashes(req.block_hashes, req.all_token_ids, bs, n_full, req.root)
        for i in range(req.num_registered_blocks, n_full):
            self.prefix_cache.register(req.block_hashes[i], req.block_ids[i], chain_pos=i)
        req.num_registered_blocks = n_full

    # ---------------- 结束 / 中止 ----------------
    def finish(self, req: Request) -> None:
        now = self.clock()
        self.running.remove(req)
        if req.pin_seconds > 0:
            # pin 的是前缀缓存表里"这条链实际指向的块"，从链头开始，遇到断点就停。
            # 注意只 pin 已登记的满块：尾部未满的块本来就不能被别的请求复用，下一轮会重算。
            chain: list[int] = []
            for h in req.block_hashes[:req.num_registered_blocks]:
                bid = self.prefix_cache.get(h)
                if bid is None:
                    break
                chain.append(bid)
            self.allocator.pin(chain, now + req.pin_seconds, now, self.cfg.max_pinned_blocks)
        self.allocator.free(req.block_ids, now)
        req.block_ids = []
        req.status = RequestStatus.FINISHED
        req.finish_time = now

    def abort(self, req: Request) -> bool:
        """从 waiting / running 里摘掉一个未完成的请求并释放它的块（已登记的满块仍留在缓存里）。"""
        if req.status == RequestStatus.WAITING:
            self.waiting.remove(req)
        elif req.status == RequestStatus.RUNNING:
            self.running.remove(req)
        else:
            return False
        self.allocator.free(req.block_ids, self.clock())
        req.block_ids = []
        req.status = RequestStatus.ABORTED
        req.finish_time = self.clock()
        return True
