"""PagedAttention 式分块 KV cache：物理存储 + 块分配器（引用计数、LRU 空闲队列、pin）。

- PagedKVCache：真正存 K/V 的大数组，形状 [n_layers, num_blocks, block_size, n_kv_heads, head_dim]。
  一条序列的 KV 不必连续：它的 block table 记录"第 i 个逻辑块 -> 哪个物理块"。
- BlockAllocator：管理物理块的生命周期。
  * ref_count：有几个运行中的请求在用这个块（共享前缀时 > 1）；
  * free_queue：ref_count 为 0 的块，按"最久未用在队头"排列（LRU）。
    这些块上的 KV 仍然有效、仍可被前缀缓存命中，直到被重新分配出去——那一刻才算"淘汰"；
  * pin_until：到这个时间之前不允许被淘汰（模拟 agent 等工具时保留它的 KV）；
  * 淘汰策略是可替换的钩子 pick_victims：默认 = 空闲队列顺序（LRU + 逆序释放 → 尾块先淘汰），
    capstone 练习里的会话感知策略就是重写它。为此每个块还记录 last_used、chain_pos、sessions。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import islice
from typing import Callable, Iterable, Iterator

import numpy as np

from .model import TinyConfig


class PagedKVCache:
    def __init__(self, cfg: TinyConfig, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (cfg.n_layers, num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = np.zeros(shape)
        self.v = np.zeros(shape)

    def write(self, layer: int, slot_mapping: np.ndarray, k: np.ndarray, v: np.ndarray) -> None:
        """把新 token 的 K/V 写到各自的物理槽位（slot = block_id * block_size + offset）。"""
        n_kv, hd = self.k.shape[-2:]
        self.k[layer].reshape(-1, n_kv, hd)[slot_mapping] = k
        self.v[layer].reshape(-1, n_kv, hd)[slot_mapping] = v

    def gather(self, layer: int, block_table: list[int], context_len: int) -> tuple[np.ndarray, np.ndarray]:
        """按 block table 读出一条序列前 context_len 个 token 的 K/V（[ctx, n_kv, hd]）。"""
        n_blocks = math.ceil(context_len / self.block_size)
        ids = np.asarray(block_table[:n_blocks], dtype=np.int64)
        n_kv, hd = self.k.shape[-2:]
        k = self.k[layer, ids].reshape(-1, n_kv, hd)[:context_len]
        v = self.v[layer, ids].reshape(-1, n_kv, hd)[:context_len]
        return k, v


@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None    # 满块登记进前缀缓存后才有
    pin_until: float = -math.inf
    last_used: float = 0.0           # 最近一次引用计数降到 0 的时间
    chain_pos: int = -1              # 在前缀链里的位置（0 = 链头）；未登记为 -1
    sessions: set[int] = field(default_factory=set)   # 用过这个块的会话（共享前缀会有多个）

    def reset_cache_meta(self) -> None:
        self.block_hash = None
        self.chain_pos = -1
        self.sessions.clear()


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        # 队头 = 最久未用 = 最先被淘汰；OrderedDict 让"从中间摘掉一个块"是 O(1)
        self.free_queue: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.on_evict: Callable[[KVBlock], None] | None = None   # 前缀缓存注册的回调
        self.ended_sessions: set[int] = set()                     # harness 告知"已结束"的会话
        self.num_evictions = 0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free(self) -> int:
        return len(self.free_queue)

    def num_pinned(self, now: float) -> int:
        return sum(1 for b in self.blocks if b.pin_until > now)

    def num_active_sessions(self, blk: KVBlock) -> int:
        return sum(1 for s in blk.sessions if s not in self.ended_sessions)

    def _evictable(self, now: float, skip: set[int]) -> Iterator[KVBlock]:
        """按空闲队列顺序（LRU → MRU）给出可分配的块：跳过 exclude 与 pin 未到期的。"""
        for bid in self.free_queue:
            if bid not in skip and self.blocks[bid].pin_until <= now:
                yield self.blocks[bid]

    def pick_victims(self, candidates: Iterator[KVBlock], n: int, now: float) -> list[int] | None:
        """淘汰策略钩子（子类可重写）：从候选里选 n 个块；不够返回 None。

        默认：空闲队列前 n 个 = LRU；因为请求结束时逆序释放，同一批释放的块里尾块排在前面。
        """
        picked = [b.block_id for b in islice(candidates, n)]
        return picked if len(picked) == n else None

    def allocate(self, n: int, now: float, exclude: Iterable[int] = ()) -> list[int] | None:
        """分配 n 个块（引用计数置 1）；不够就返回 None 且不做任何修改。"""
        if n == 0:
            return []
        picked = self.pick_victims(self._evictable(now, set(exclude)), n, now)
        if picked is None:
            return None
        for bid in picked:
            del self.free_queue[bid]
            blk = self.blocks[bid]
            if blk.block_hash is not None:       # 这个块还挂着一段可复用的前缀：现在被淘汰
                if self.on_evict is not None:
                    self.on_evict(blk)
                self.num_evictions += 1
            blk.reset_cache_meta()
            blk.ref_count = 1
        return picked

    def touch(self, block_ids: Iterable[int]) -> None:
        """前缀命中：引用计数 +1；原本空闲的块从空闲队列里摘出来。"""
        for bid in block_ids:
            blk = self.blocks[bid]
            if blk.ref_count == 0:
                del self.free_queue[bid]
            blk.ref_count += 1

    def free(self, block_ids: list[int], now: float = 0.0) -> None:
        """引用计数 -1；降到 0 的块放到队尾（最近使用）。

        逆序释放：一条序列的尾块先入队、离队头更近，所以同一批释放的块里尾块先被淘汰，
        共享的前缀头部活得最久（前缀链从头断掉会让后面所有块都无法命中）。
        """
        for bid in reversed(block_ids):
            blk = self.blocks[bid]
            if blk.ref_count <= 0:
                raise RuntimeError(f"double free of block {bid}")
            blk.ref_count -= 1
            if blk.ref_count == 0:
                blk.last_used = now
                self.free_queue[bid] = None

    def mark_session(self, block_ids: Iterable[int], session_id: int) -> None:
        for bid in block_ids:
            self.blocks[bid].sessions.add(session_id)

    def pin(self, block_ids: list[int], until: float, now: float, max_pinned: int) -> int:
        """把块 pin 到 until 之前不被淘汰；总 pin 数不超过 max_pinned。返回实际 pin 住的块数。"""
        budget = max_pinned - self.num_pinned(now)
        pinned = 0
        for bid in block_ids:                 # 调用方按前缀链顺序传入：先保住链头
            blk = self.blocks[bid]
            if blk.pin_until > now:           # 已经 pin 着：只延长时间，不占新额度
                blk.pin_until = max(blk.pin_until, until)
                pinned += 1
            elif budget > 0:
                blk.pin_until = until
                budget -= 1
                pinned += 1
            else:
                break
        return pinned

    def assert_no_leak(self) -> None:
        """所有请求结束后：每个块引用计数为 0，且都在空闲队列里。"""
        busy = [b.block_id for b in self.blocks if b.ref_count != 0]
        if busy or self.num_free != self.num_blocks:
            raise AssertionError(f"leaked blocks: {busy}, free={self.num_free}/{self.num_blocks}")
