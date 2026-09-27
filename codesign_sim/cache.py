"""前缀缓存：按 block 的 radix tree（压缩前缀树），和 SGLang RadixAttention 的结构一致；可选第二层（主机 DRAM）。

- 每条边是一串 block hash（压缩：没有分叉的链合并成一个节点）
- match：从根往下走，返回命中的前缀 block 数；只刷新**真正命中**的部分（停在边中间时先分裂节点）
- insert：插入一条 block 序列，必要时分裂节点；超出容量就淘汰
- 淘汰只能从"叶子一侧"进行：内部节点是别人的前缀，删了会让子孙失效

两层存储（dram_blocks > 0 时启用）：
- HBM 满了不直接丢，而是把 HBM 边界节点（没有 HBM 孩子的 HBM 节点）的尾部**降级**到 DRAM；DRAM 满了再真正淘汰 DRAM 叶子
- 不变式：从根往下，一条路径上先是 HBM 节点、后是 DRAM 节点
- match_tiers 分别返回 HBM 命中和 DRAM 命中的 block 数；DRAM 命中要按 PCIe 带宽搬回来（或者选择重算），
  由调用方（router）计费；insert 经过 DRAM 节点时会把它们**提升**回 HBM

三种淘汰策略（作用于 HBM 层）：
- "lru"：最久没访问的先淘汰，忽略 pin_until
- "ttl"：pin_until 之前**硬保留**（模拟工具调用期间保留 KV）；全部被 pin 住时新插入会被截断
  （stats["rejected_blocks"]）。若仍超容量（例如路径上的 DRAM 节点被提升回 HBM、又没有新插入可截断），
  作为最后手段按 pin 到期先后强制淘汰（stats["pin_overrides"]），保证 size <= capacity 始终成立。
  同一节点被多个请求 pin 时取最晚的 pin_until
- "session_aware"：pin_until 是 harness 给的"预计下次使用时间"（软提示）：
  先按 LRU 淘汰没有提示或提示已过期的，不够时再淘汰"预计回来最晚"的（近似 Belady）。
  同一节点有多个未来使用者时取**最近**的一次（它最早会被用到）
"""

from __future__ import annotations

import heapq
import itertools

_POLICIES = ("lru", "ttl", "session_aware")
_HBM, _DRAM = 0, 1


class _Node:
    __slots__ = ("key", "children", "parent", "last_access", "pin_until", "stamp", "alive", "tier", "n_hbm_children")

    def __init__(self, key: list[int], parent: _Node | None, now: float = 0.0, pin_until: float | None = None,
                 tier: int = _HBM):
        self.key = key
        self.children: dict[int, _Node] = {}
        self.parent = parent
        self.last_access = now
        self.pin_until = pin_until
        self.stamp = 0  # 每次访问/状态变化 +1，让堆里的旧条目失效（lazy deletion）
        self.alive = True
        self.tier = tier
        self.n_hbm_children = 0


def _common_prefix(key: list[int], blocks: list[int], start: int) -> int:
    """key 与 blocks[start:] 的最长公共前缀长度（二分 + 切片比较，比逐个比较快得多）。"""
    limit = min(len(key), len(blocks) - start)
    if key[:limit] == blocks[start : start + limit]:
        return limit
    lo, hi = 0, limit - 1  # 不变式：前 lo 个相同，前 hi+1 个不全相同
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if key[:mid] == blocks[start : start + mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class PrefixCache:
    def __init__(self, capacity_blocks: int, policy: str = "lru", *, dram_blocks: int = 0):
        if policy not in _POLICIES:
            raise ValueError(f"policy 必须是 {_POLICIES} 之一，收到 {policy!r}")
        if capacity_blocks < 0 or dram_blocks < 0:
            raise ValueError("capacity_blocks / dram_blocks 必须 >= 0")
        self.capacity = int(capacity_blocks)
        self.dram_capacity = int(dram_blocks)
        self.policy = policy
        self.root = _Node([], None)
        self.size = 0  # HBM 里的 block 数
        self.dram_size = 0
        self.stats: dict = {
            "hit_blocks": 0,  # 命中（HBM + DRAM）
            "miss_blocks": 0,
            "evictions": 0,  # 真正丢弃的 block 数（不含降级）
            "inserted_blocks": 0,
            "rejected_blocks": 0,  # 因容量/pin 无法插入的 block 数
            "pin_overrides": 0,  # 全部被 pin 住仍超容量时，被强制淘汰的 pin 节点数
            "lookups": 0,
            "hbm_hit_blocks": 0,
            "dram_hit_blocks": 0,
            "demoted_blocks": 0,  # HBM -> DRAM
            "promoted_blocks": 0,  # DRAM -> HBM
        }
        self._seq = itertools.count()
        self._lru: list = []  # (last_access, seq, node, stamp)：HBM 边界节点，按 LRU
        self._pinned: list = []  # (pin_until, seq, node, stamp)：带未来提示的 HBM 边界节点，按到期时间
        self._far: list = []  # (-pin_until, seq, node, stamp)：session_aware 用，预计回来最晚的先出
        self._dram_lru: list = []  # (last_access, seq, node, stamp)：DRAM 叶子

    # ------------------------------------------------------------------ 查询

    def match(self, blocks: list[int], now: float) -> int:
        """返回命中的前缀 block 数（HBM + DRAM 两层合计），并刷新命中部分的访问时间。"""
        hbm, dram = self.match_tiers(blocks, now)
        return hbm + dram

    def match_tiers(self, blocks: list[int], now: float) -> tuple[int, int]:
        """返回 (HBM 命中 block 数, DRAM 命中 block 数)；不启用 DRAM 层时第二项恒为 0。"""
        self.stats["lookups"] += 1
        hbm, dram = self._walk(blocks, now, touch=True)
        self.stats["hit_blocks"] += hbm + dram
        self.stats["hbm_hit_blocks"] += hbm
        self.stats["dram_hit_blocks"] += dram
        self.stats["miss_blocks"] += len(blocks) - hbm - dram
        return hbm, dram

    def peek(self, blocks: list[int]) -> int:
        """和 match 一样返回命中长度（两层合计），但不刷新访问时间、不计入统计（给 router 打分用）。"""
        hbm, dram = self._walk(blocks, 0.0, touch=False)
        return hbm + dram

    def _walk(self, blocks: list[int], now: float, touch: bool) -> tuple[int, int]:
        node, i, n = self.root, 0, len(blocks)
        hits = [0, 0]
        while i < n:
            child = node.children.get(blocks[i])
            if child is None:
                break
            key = child.key
            if blocks[i : i + len(key)] == key:  # 整条边命中（C 层比较，很快）
                i += len(key)
                hits[child.tier] += len(key)
                if touch:
                    self._touch(child, now)
                node = child
                continue
            j = _common_prefix(key, blocks, i)  # 部分命中：停在这条边中间
            if j > 0:
                hits[child.tier] += j
                if touch:  # 先分裂，只刷新命中的前半段，未用到的后缀保持原来的访问时间
                    self._touch(self._split(node, child, j), now)
            break
        return hits[0], hits[1]

    @property
    def hit_rate(self) -> float:
        total = self.stats["hit_blocks"] + self.stats["miss_blocks"]
        return self.stats["hit_blocks"] / total if total else 0.0

    def __len__(self) -> int:
        return self.size

    def leaves(self) -> list[list[int]]:
        """所有叶子节点的完整 block 路径（调试/测试用）。"""
        out: list[list[int]] = []
        stack = [(self.root, [])]
        while stack:
            node, path = stack.pop()
            path = path + node.key
            if not node.children and node is not self.root:
                out.append(path)
            stack.extend((c, path) for c in node.children.values())
        return out

    def tier_sizes(self) -> tuple[int, int]:
        """(HBM block 数, DRAM block 数)。"""
        return self.size, self.dram_size

    # ------------------------------------------------------------------ 插入

    def insert(self, blocks: list[int], now: float, pin_until: float | None = None) -> None:
        """插入一条 block 序列；已存在的前缀只刷新时间和 pin 提示，路径上的 DRAM 节点提升回 HBM。"""
        if not blocks:
            return
        if self.capacity == 0:
            self.stats["rejected_blocks"] += len(blocks)
            return
        node, i, n = self.root, 0, len(blocks)
        while i < n:
            child = node.children.get(blocks[i])
            if child is None:
                break
            j = _common_prefix(child.key, blocks, i)
            if j < len(child.key):
                child = self._split(node, child, j)
            if child.tier == _DRAM:
                self._promote(child)
            self._touch(child, now, pin_until)
            node, i = child, i + j
        leaf = None
        if i < n:
            leaf = self._attach(node, blocks[i:], now, pin_until)
        self._enforce_capacity(now, protect=leaf)
        if leaf is not None and leaf.alive:
            self.stats["inserted_blocks"] += len(leaf.key)
            self._push(leaf, now)

    def _attach(self, parent: _Node, key: list[int], now: float, pin_until: float | None) -> _Node:
        leaf = _Node(key, parent, now, pin_until)
        parent.children[key[0]] = leaf
        self._set_hbm_child(parent, +1)
        self.size += len(key)
        return leaf

    def _split(self, parent: _Node, child: _Node, j: int) -> _Node:
        """把 child 的边在第 j 个 block 处切开，返回新的中间节点（继承 child 的时间、提示与层级）。"""
        mid = _Node(child.key[:j], parent, child.last_access, child.pin_until, child.tier)
        parent.children[mid.key[0]] = mid
        child.key = child.key[j:]
        child.parent = mid
        mid.children[child.key[0]] = child
        mid.n_hbm_children = 1 if child.tier == _HBM else 0
        child.stamp += 1  # child 的边变短了，旧堆条目作废后重新入堆
        if self._is_frontier(child) or self._is_dram_leaf(child):
            self._push(child, child.last_access)
        if self._is_frontier(mid):
            self._push(mid, mid.last_access)
        return mid

    # ------------------------------------------------------------------ 层级与候选

    @staticmethod
    def _is_frontier(node: _Node) -> bool:
        """HBM 边界节点：本身在 HBM、且没有 HBM 孩子——只有它们可以被淘汰/降级。"""
        return node.alive and node.tier == _HBM and node.n_hbm_children == 0 and node.parent is not None

    @staticmethod
    def _is_dram_leaf(node: _Node) -> bool:
        return node.alive and node.tier == _DRAM and not node.children

    def _set_hbm_child(self, parent: _Node, delta: int) -> None:
        parent.n_hbm_children += delta
        parent.stamp += 1
        if self._is_frontier(parent):
            self._push(parent, parent.last_access)

    def _promote(self, node: _Node) -> None:
        node.tier = _HBM
        n = len(node.key)
        self.dram_size -= n
        self.size += n
        self.stats["promoted_blocks"] += n
        node.stamp += 1
        self._set_hbm_child(node.parent, +1)

    def _touch(self, node: _Node, now: float, pin_until: float | None = None) -> None:
        node.last_access = max(node.last_access, now)
        if pin_until is not None:
            node.pin_until = self._merge_hint(node.pin_until, pin_until, now)
        node.stamp += 1
        if self._is_frontier(node) or self._is_dram_leaf(node):
            self._push(node, now)

    def _merge_hint(self, old: float | None, new: float, now: float) -> float:
        if old is None or old <= now:
            return new
        # ttl：保留到最晚的那个 pin；session_aware：最早会被用到的那次才决定淘汰顺序
        return max(old, new) if self.policy != "session_aware" else min(old, new)

    def _hinted(self, node: _Node, now: float) -> bool:
        return self.policy != "lru" and node.pin_until is not None and node.pin_until > now

    def _push(self, node: _Node, now: float) -> None:
        seq = next(self._seq)
        if node.tier == _DRAM:
            heapq.heappush(self._dram_lru, (node.last_access, seq, node, node.stamp))
        elif self._hinted(node, now):
            heapq.heappush(self._pinned, (node.pin_until, seq, node, node.stamp))
            if self.policy == "session_aware":
                heapq.heappush(self._far, (-node.pin_until, seq, node, node.stamp))
        else:
            heapq.heappush(self._lru, (node.last_access, seq, node, node.stamp))

    def _pop_hbm_victim(self, now: float, protect: _Node | None) -> _Node | None:
        valid = lambda nd, st: self._is_frontier(nd) and nd.stamp == st and nd is not protect  # noqa: E731
        while self._pinned and self._pinned[0][0] <= now:  # pin 到期的回到 LRU 队列
            _, _, node, stamp = heapq.heappop(self._pinned)
            if valid(node, stamp):
                heapq.heappush(self._lru, (node.last_access, next(self._seq), node, stamp))
        while self._lru:
            _, _, node, stamp = heapq.heappop(self._lru)
            if valid(node, stamp):
                return node
        if self.policy == "session_aware":  # 再淘汰"预计回来最晚"的；ttl 则不动被 pin 的
            while self._far:
                _, _, node, stamp = heapq.heappop(self._far)
                if valid(node, stamp) and self._hinted(node, now):
                    return node
        return None

    def _pop_pinned_victim(self, now: float) -> _Node | None:
        """仍在 pin 期内的 HBM 边界节点里，pin 最早到期的那个（只在容量无法满足时使用）。"""
        while self._pinned:
            _, _, node, stamp = heapq.heappop(self._pinned)
            if self._is_frontier(node) and node.stamp == stamp:
                return node
        return None

    def _pop_dram_victim(self) -> _Node | None:
        while self._dram_lru:
            _, _, node, stamp = heapq.heappop(self._dram_lru)
            if self._is_dram_leaf(node) and node.stamp == stamp:
                return node
        return None

    # ------------------------------------------------------------------ 容量

    def _enforce_capacity(self, now: float, protect: _Node | None) -> None:
        while self.size > self.capacity:
            victim = self._pop_hbm_victim(now, protect)
            if victim is None:
                break
            self._shrink_hbm(victim, self.size - self.capacity, now)
        if self.size > self.capacity and protect is not None and protect.alive:
            self._truncate_new(protect, self.size - self.capacity, now)  # 其余都被 pin 住（ttl）
        # 最后手段：全部被 pin 住、又没有新插入可截断（例如路径上的 DRAM 节点被提升回 HBM），
        # 按 pin 到期先后强制淘汰，保证 size <= capacity 始终成立
        while self.size > self.capacity:
            victim = self._pop_pinned_victim(now)
            if victim is None:
                break
            self.stats["pin_overrides"] += 1
            self._shrink_hbm(victim, self.size - self.capacity, now)
        while self.dram_size > self.dram_capacity:
            victim = self._pop_dram_victim()
            if victim is None:
                break
            self._drop_tail(victim, self.dram_size - self.dram_capacity, now)

    def _shrink_hbm(self, victim: _Node, need: int, now: float) -> None:
        """从 HBM 边界节点 victim 的尾部腾出 need 个 block：有 DRAM 层就降级，否则丢弃。"""
        if self.dram_capacity == 0:
            self._drop_tail(victim, need, now)
            return
        k = len(victim.key)
        if k > need:  # 只降级尾部 need 个：切开后 victim 本身就是后半段，前半段留在 HBM
            self._split(victim.parent, victim, k - need)
        victim.tier = _DRAM
        n = len(victim.key)
        self.size -= n
        self.dram_size += n
        self.stats["demoted_blocks"] += n
        victim.stamp += 1
        self._set_hbm_child(victim.parent, -1)
        if self._is_dram_leaf(victim):
            self._push(victim, now)

    def _drop_tail(self, victim: _Node, need: int, now: float) -> None:
        """从叶子 victim 的尾部丢掉 need 个 block（不够就整个删掉）。"""
        k = len(victim.key)
        tier_size = "size" if victim.tier == _HBM else "dram_size"
        cut = min(k, need)
        setattr(self, tier_size, getattr(self, tier_size) - cut)
        self.stats["evictions"] += cut
        if k <= need:
            self._remove_leaf(victim, now)
        else:
            victim.key = victim.key[: k - need]
            victim.stamp += 1
            self._push(victim, now)  # 仍是候选，重新入堆

    def _truncate_new(self, leaf: _Node, excess: int, now: float) -> None:
        self.stats["rejected_blocks"] += min(excess, len(leaf.key))
        if excess >= len(leaf.key):
            self.size -= len(leaf.key)
            self._remove_leaf(leaf, now)
        else:
            leaf.key = leaf.key[: len(leaf.key) - excess]
            self.size -= excess

    def _remove_leaf(self, leaf: _Node, now: float) -> None:
        parent = leaf.parent
        leaf.alive = False
        del parent.children[leaf.key[0]]
        if leaf.tier == _HBM:
            self._set_hbm_child(parent, -1)
        elif self._is_dram_leaf(parent):
            parent.stamp += 1
            self._push(parent, now)
