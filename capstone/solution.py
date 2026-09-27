"""Capstone 练习参考答案（与 exercise.py 同签名）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from mini_engine import BlockAllocator, TinyTransformer, compute_block_hashes
from mini_engine.kv_cache import KVBlock


# ---------------------------------------------------------------------------
# (a) cache-aware 多实例路由
# ---------------------------------------------------------------------------
def prefix_match_blocks(block_hashes: list[int], cached: set[int]) -> int:
    n = 0
    for h in block_hashes:
        if h not in cached:
            break
        n += 1
    return n


def pick_instance(prompt_token_ids: list[int], instance_hashes: list[set[int]], loads: list[int],
                  load_cap: int, block_size: int = 16) -> int:
    hashes = compute_block_hashes(prompt_token_ids, block_size)
    ok = [i for i in range(len(loads)) if loads[i] < load_cap]
    if not ok:
        return min(range(len(loads)), key=lambda i: (loads[i], i))
    return min(ok, key=lambda i: (-prefix_match_blocks(hashes, instance_hashes[i]), loads[i], i))


# ---------------------------------------------------------------------------
# (b) 会话感知淘汰：先写纯函数，再接进引擎的分配器
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreeBlockInfo:
    block_id: int
    last_used: float          # 最近一次引用计数降到 0 的时间
    pin_until: float          # 到这个时间之前不能淘汰
    chain_pos: int            # 在前缀链里的位置（0 = 链头；未登记为 -1）
    n_active_sessions: int    # 还有几个未结束的会话用过这个块（共享的 system prompt 块可能有很多个）
    cached: bool              # 是否挂着一段可复用的前缀（没挂的块淘汰了也不损失什么）


def choose_victims(free_blocks: list[FreeBlockInfo], n: int, now: float) -> list[int] | None:
    candidates = [b for b in free_blocks if b.pin_until <= now]
    if len(candidates) < n:
        return None
    candidates.sort(key=lambda b: (b.cached, b.n_active_sessions, b.last_used, -b.chain_pos, b.block_id))
    return [b.block_id for b in candidates[:n]]


class SessionAwareAllocator(BlockAllocator):
    def pick_victims(self, candidates: Iterator[KVBlock], n: int, now: float) -> list[int] | None:
        infos = [FreeBlockInfo(b.block_id, b.last_used, b.pin_until, b.chain_pos, self.num_active_sessions(b),
                               b.block_hash is not None) for b in candidates]
        return choose_victims(infos, n, now)


# ---------------------------------------------------------------------------
# (c) greedy 投机解码
# ---------------------------------------------------------------------------
DraftFn = Callable[[list[int], int], list[int]]


def speculative_greedy(model: TinyTransformer, prompt: list[int], max_tokens: int, draft: DraftFn,
                       k: int) -> tuple[list[int], int]:
    seq = list(prompt)
    out: list[int] = []
    n_calls = 0
    while len(out) < max_tokens:
        kk = min(k, max_tokens - len(out) - 1)
        proposal = list(draft(seq, kk))[:kk] if kk > 0 else []
        logits = model.forward_dense(seq + proposal)
        n_calls += 1
        target = np.argmax(logits[len(seq) - 1:], axis=-1)      # target 在每个位置的 greedy 选择
        accepted = 0
        while accepted < len(proposal) and proposal[accepted] == int(target[accepted]):
            accepted += 1
        new = proposal[:accepted] + [int(target[accepted])]      # 接受的前缀 + 1 个 target 自己的 token
        out.extend(new)
        seq.extend(new)
    return out[:max_tokens], n_calls
