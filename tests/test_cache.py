"""cache：radix 前缀缓存的命中、分裂、LRU、只从叶子侧淘汰、pin 语义、两层存储；附随机操作的不变式检查。"""

import random

import pytest

from codesign_sim.cache import PrefixCache


def seq(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


def test_match_and_stats():
    c = PrefixCache(100)
    c.insert(seq(0, 10), now=0)
    assert c.match(seq(0, 10) + [999], now=1) == 10
    assert c.match(seq(0, 5) + [999, 1000], now=2) == 5  # 停在边中间
    assert c.match([42], now=3) == 0
    assert c.stats["hit_blocks"] == 15 and c.stats["miss_blocks"] == 4
    assert c.hit_rate == pytest.approx(15 / 19)
    assert c.peek(seq(0, 10)) == 10 and c.stats["lookups"] == 3  # peek 不计统计


def test_split_shares_prefix():
    c = PrefixCache(100)
    c.insert([1, 2, 3, 4], now=0)
    c.insert([1, 2, 9], now=1)
    assert len(c) == 5  # [1,2] 只存一份
    assert sorted(c.leaves()) == [[1, 2, 3, 4], [1, 2, 9]]


def test_partial_match_refreshes_only_matched_prefix():
    c = PrefixCache(26)
    c.insert(seq(0, 10), now=0)  # A
    c.insert(seq(100, 10), now=1)  # B
    c.match(seq(0, 4) + [999], now=2)  # 只用到 A 的前 4 个 block：先分裂，只刷新这 4 个
    c.insert(seq(200, 12), now=3)  # 需要腾 6 个
    assert c.peek(seq(100, 10)) == 10  # B 没被动（若整条 A 都被刷新，B 就成了最旧的）
    assert c.peek(seq(0, 10)) == 4  # A 没用到的后缀先走


def test_lru_evicts_least_recent():
    c = PrefixCache(10)
    c.insert(seq(0, 5), now=0)
    c.insert(seq(100, 5), now=1)
    c.match(seq(0, 5), now=2)
    c.insert(seq(200, 5), now=3)
    assert c.peek(seq(0, 5)) == 5 and c.peek(seq(100, 5)) == 0
    assert c.stats["evictions"] == 5 and len(c) == 10


def test_only_leaves_are_evicted():
    c = PrefixCache(10)
    prefix = seq(0, 4)
    c.insert(prefix + seq(100, 3), now=0)
    c.insert(prefix + seq(200, 3), now=1)
    c.insert(prefix + seq(300, 3), now=2)
    assert c.peek(prefix) == 4  # 共享前缀（内部节点）没被淘汰
    assert c.peek(prefix + seq(100, 3)) == 4  # 最久没用的叶子被淘汰
    assert len(c) == 10


def test_ttl_pins_until_expiry():
    lru, ttl = PrefixCache(10, "lru"), PrefixCache(10, "ttl")
    for c in (lru, ttl):
        c.insert(seq(0, 5), now=0, pin_until=100)
        c.insert(seq(100, 5), now=1)
        c.insert(seq(200, 5), now=2)
    assert lru.peek(seq(0, 5)) == 0 and lru.peek(seq(100, 5)) == 5
    assert ttl.peek(seq(0, 5)) == 5 and ttl.peek(seq(100, 5)) == 0
    c = PrefixCache(10, "ttl")
    c.insert(seq(0, 5), now=0, pin_until=100)
    c.insert(seq(100, 5), now=0, pin_until=100)
    c.insert(seq(200, 5), now=1)  # 全被 pin：新插入被拒绝
    assert c.peek(seq(200, 5)) == 0 and c.stats["rejected_blocks"] == 5
    c.insert(seq(200, 5), now=101)  # pin 过期后可以淘汰
    assert c.peek(seq(200, 5)) == 5 and len(c) == 10


def test_session_aware_order_and_nearest_hint():
    c = PrefixCache(15, "session_aware")
    c.insert(seq(0, 5), now=0, pin_until=50)
    c.insert(seq(100, 5), now=1, pin_until=500)
    c.insert(seq(200, 5), now=2)  # 没有提示（session 已结束）
    c.insert(seq(300, 5), now=3)  # 先淘汰没有提示的
    assert c.peek(seq(200, 5)) == 0 and c.peek(seq(0, 5)) == 5 and c.peek(seq(100, 5)) == 5
    c.insert(seq(400, 5), now=4, pin_until=60)
    assert c.peek(seq(300, 5)) == 0
    c.insert(seq(500, 5), now=5, pin_until=70)
    assert c.peek(seq(100, 5)) == 0  # 预计回来最晚的被淘汰
    # 同一前缀有两个未来使用者：取最近的一次（它很快会被用到，不该被当成"很晚才回来"）
    d = PrefixCache(10, "session_aware")
    d.insert(seq(0, 5), now=0, pin_until=20)
    d.insert(seq(0, 5), now=1, pin_until=900)
    d.insert(seq(100, 5), now=2, pin_until=300)
    d.insert(seq(200, 5), now=3, pin_until=100)
    assert d.peek(seq(0, 5)) == 5 and d.peek(seq(100, 5)) == 0


def test_two_tier_demote_and_promote():
    c = PrefixCache(10, dram_blocks=10)
    c.insert(seq(0, 8), now=0)
    c.insert(seq(100, 8), now=1)  # HBM 放不下：旧序列尾部降级到 DRAM 而不是丢掉
    assert c.tier_sizes() == (10, 6) and c.stats["evictions"] == 0 and c.stats["demoted_blocks"] == 6
    assert c.match_tiers(seq(0, 8), now=2) == (2, 6)  # 前 2 个在 HBM，后 6 个在 DRAM
    c.insert(seq(0, 8), now=3)  # 用到了就提升回 HBM（挤出另一条）
    assert c.match_tiers(seq(0, 8), now=4) == (8, 0) and c.stats["promoted_blocks"] == 6
    c.insert(seq(200, 10), now=5)
    c.insert(seq(300, 10), now=6)  # DRAM 也满了：真正淘汰
    assert c.stats["evictions"] > 0 and c.tier_sizes()[1] <= 10


def _check_invariants(c: PrefixCache) -> None:
    hbm = dram = 0
    stack = [c.root]
    while stack:
        n = stack.pop()
        n_hbm = 0
        for k, ch in n.children.items():
            assert ch.key and ch.key[0] == k and ch.parent is n and ch.alive
            if n is not c.root and n.tier == 1:
                assert ch.tier == 1  # DRAM 节点的孩子也在 DRAM
            hbm, dram = (hbm + len(ch.key), dram) if ch.tier == 0 else (hbm, dram + len(ch.key))
            n_hbm += ch.tier == 0
            stack.append(ch)
        assert n_hbm == n.n_hbm_children
    assert (hbm, dram) == c.tier_sizes()
    assert c.dram_size <= c.dram_capacity and (c.policy == "ttl" or c.size <= c.capacity)


def test_random_operations_keep_invariants():
    rnd = random.Random(0)
    for pol in ("lru", "ttl", "session_aware"):
        for dram in (0, 30):
            for _ in range(25):
                c = PrefixCache(rnd.randint(5, 60), pol, dram_blocks=dram)
                base = [[rnd.randint(0, 10**6) for _ in range(rnd.randint(1, 20))] for _ in range(6)]
                now = 0.0
                for _ in range(60):
                    now += rnd.random()
                    b = rnd.choice(base)
                    s = b[: rnd.randint(1, len(b))] + [rnd.randint(0, 10**6) for _ in range(rnd.randint(0, 6))]
                    if rnd.random() < 0.3:
                        assert sum(c.match_tiers(s, now)) <= len(s)
                    else:
                        c.insert(s, now, pin_until=now + rnd.random() * 5 if rnd.random() < 0.5 else None)
                        if pol == "lru" and c.capacity >= len(s):
                            assert c.peek(s) == len(s)  # 刚插入的总能完整命中
                    _check_invariants(c)


def test_zero_capacity_and_bad_policy():
    c = PrefixCache(0)
    c.insert(seq(0, 3), now=0)
    assert c.match(seq(0, 3), now=1) == 0 and len(c) == 0
    with pytest.raises(ValueError):
        PrefixCache(10, "fifo")


def test_ttl_capacity_holds_when_promotion_meets_all_pinned():
    """全部被 pin 住、又没有新插入可截断时（路径上的 DRAM 节点被提升回 HBM），容量不变式仍成立。"""
    c = PrefixCache(4, "ttl", dram_blocks=8)
    c.insert([1, 2, 3, 4], now=0.0)                      # 不 pin
    c.insert([5, 6, 7, 8], now=1.0, pin_until=100.0)     # 挤走前一条，它降级到 DRAM
    c.insert([1, 2, 3, 4], now=2.0, pin_until=100.0)     # 提升回 HBM 且被 pin，HBM 里没有可淘汰的
    assert c.size <= c.capacity
    assert c.stats["pin_overrides"] >= 1
    assert c.match([1, 2, 3, 4], 3.0) == 4 and c.match([5, 6, 7, 8], 3.0) == 4   # 被挤的留在 DRAM，仍可命中
