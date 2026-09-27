"""vLLM 式按 block 的哈希链前缀缓存。

- 第 i 个满块的 hash = H(第 i-1 块的 hash, 第 i 块的 block_size 个 token)，第 0 块的父 hash 是 root。
  在忽略哈希碰撞的前提下，"两个请求第 i 块 hash 相同"等价于"两者前 (i+1)*block_size 个 token 完全相同"——
  用一个扁平 dict 就能做最长前缀匹配，不需要显式的树。
  （这里用 sha256 截断到 64 位：N 个块两两碰撞的概率约 N^2 / 2^65，N = 1e9 时约 3%。生产系统用完整哈希，
  还会把 LoRA ID、多模态 hash、租户 salt 等 extra keys 混进去；这里的 root 参数就是混入 salt / 权重版本的位置。）
- 只有满块才登记、才共享：满块内容不再变化，多个请求同时引用它不需要 copy-on-write。
- 同一个 hash 可以挂多个物理块（两个请求在不同步里各自算出了同一段前缀）。查找返回最早登记的那个；
  它被淘汰时，其余副本自动顶上（与 vLLM BlockPool 里 hash -> {block_id: block} 的做法一致）。
- 命中的块 ref_count +1；ref_count 回到 0 的块留在空闲队列里仍可被命中，
  直到分配器把它重新分配出去（淘汰），这时通过 on_evict 回调把它从表里删掉。
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .kv_cache import BlockAllocator, KVBlock

ROOT_HASH = 0  # 默认的"第 0 块的父 hash"


def hash_block(parent_hash: int, token_ids: Sequence[int]) -> int:
    """确定性的 64 位块哈希（sha256 截断；不用 Python 内置 hash，跨进程也稳定）。"""
    h = hashlib.sha256(parent_hash.to_bytes(8, "little"))
    h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def root_hash(salt: int = 0) -> int:
    """链的根：salt = 0 时就是 ROOT_HASH；非 0 时（租户隔离、权重版本）整条链都不同。"""
    return ROOT_HASH if salt == 0 else hash_block(ROOT_HASH, [salt])


def extend_block_hashes(hashes: list[int], token_ids: Sequence[int], block_size: int, n_blocks: int,
                        root: int = ROOT_HASH) -> list[int]:
    """把 hashes 就地补到前 n_blocks 个满块（已有的不重算）。token 不够组成满块时停止。"""
    n_blocks = min(n_blocks, len(token_ids) // block_size)
    while len(hashes) < n_blocks:
        i = len(hashes)
        parent = hashes[-1] if hashes else root
        hashes.append(hash_block(parent, token_ids[i * block_size:(i + 1) * block_size]))
    return hashes


def compute_block_hashes(token_ids: Sequence[int], block_size: int, root: int = ROOT_HASH) -> list[int]:
    """整条 token 序列的哈希链（只含满块）。"""
    return extend_block_hashes([], token_ids, block_size, len(token_ids) // block_size, root)


class PrefixCache:
    def __init__(self, allocator: BlockAllocator, enabled: bool = True):
        self.allocator = allocator
        self.enabled = enabled
        self.table: dict[int, dict[int, None]] = {}      # block hash -> 有序的物理块号集合
        allocator.on_evict = self._on_evict

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.table

    def get(self, block_hash: int) -> int | None:
        ids = self.table.get(block_hash)
        return next(iter(ids)) if ids else None

    def lookup(self, block_hashes: Sequence[int]) -> list[int]:
        """最长前缀命中：从第 0 块开始逐块查表，遇到第一个未命中就停。无副作用。"""
        if not self.enabled:
            return []
        hit: list[int] = []
        for h in block_hashes:
            bid = self.get(h)
            if bid is None:
                break
            hit.append(bid)
        return hit

    def register(self, block_hash: int, block_id: int, chain_pos: int = -1) -> None:
        """一个块被算满（或本步必定算满）后登记。同一 hash 已有别的块时作为副本挂在后面。"""
        blk = self.allocator.blocks[block_id]
        if not self.enabled or blk.block_hash is not None:
            return
        self.table.setdefault(block_hash, {})[block_id] = None
        blk.block_hash = block_hash
        blk.chain_pos = chain_pos

    def reset(self) -> int:
        """清空整个前缀缓存（例如权重更新之后）。返回失效的块数。"""
        n = 0
        for ids in self.table.values():
            for bid in ids:
                self.allocator.blocks[bid].reset_cache_meta()
                n += 1
        self.table.clear()
        return n

    def _on_evict(self, blk: KVBlock) -> None:
        ids = self.table.get(blk.block_hash)
        if ids is not None and blk.block_id in ids:
            del ids[blk.block_id]
            if not ids:
                del self.table[blk.block_hash]
