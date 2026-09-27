"""模块 04 练习测试。离线、确定性：三个函数都是纯函数。

运行：make module N=04
用参考答案验证测试本身：INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/04_harness_workloads
"""

from __future__ import annotations

import math

import pytest

from codesign_sim.testing import load_exercise

ex = load_exercise(__file__)


# ================================================================== (a) chain_block_hashes


def test_a_only_full_blocks_are_hashed():
    assert len(ex.chain_block_hashes(list(range(10)), 4)) == 2
    assert ex.chain_block_hashes([1, 2, 3], 4) == []
    assert ex.chain_block_hashes([], 4) == []


def test_a_matches_the_documented_formula():
    tokens = [7, 8, 9, 10, 11, 12]
    h0 = hash((0, (7, 8)))
    h1 = hash((h0, (9, 10)))
    h2 = hash((h1, (11, 12)))
    assert ex.chain_block_hashes(tokens, 2) == [h0, h1, h2]


def test_a_shared_prefix_gives_identical_hashes():
    a = ex.chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 2)
    b = ex.chain_block_hashes([1, 2, 3, 4, 99, 98, 97, 96], 2)
    assert a[:2] == b[:2]
    assert a[2] != b[2] and a[3] != b[3]


def test_a_early_change_invalidates_every_later_block():
    """把第 0 个 token 换掉（相当于 system prompt 开头放了个时间戳）：所有 block 的 hash 都变。"""
    base = list(range(64))
    changed = [999] + base[1:]
    ha = ex.chain_block_hashes(base, 16)
    hb = ex.chain_block_hashes(changed, 16)
    assert all(x != y for x, y in zip(ha, hb))


def test_a_same_block_content_at_different_position_differs():
    """同样内容的 block 出现在不同位置，hash 不同 —— 前缀缓存不认「非前缀」的重复内容。"""
    hashes = ex.chain_block_hashes([5, 5, 5, 5, 5, 5], 2)
    assert len(set(hashes)) == 3


def test_a_salt_isolates_tenants():
    tokens = list(range(32))
    assert ex.chain_block_hashes(tokens, 8, salt=0) != ex.chain_block_hashes(tokens, 8, salt=1)
    assert ex.chain_block_hashes(tokens, 8, salt=42) == ex.chain_block_hashes(tokens, 8, salt=42)


def test_a_rejects_bad_block_size():
    with pytest.raises(ValueError):
        ex.chain_block_hashes([1, 2, 3], 0)


# ================================================================== (b) replay_prefix_cache


def test_b_docstring_example():
    assert ex.replay_prefix_cache([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]], 2) == (4, 8)


def test_b_empty():
    assert ex.replay_prefix_cache([], 16) == (0, 0)


def test_b_append_only_session_reuses_everything_but_the_tail():
    system = list(range(100, 132))  # 32 token 的「system prompt」
    turn1 = system + [1] * 16
    turn2 = turn1 + [2] * 16
    turn3 = turn2 + [3] * 16
    reused, recomputed = ex.replay_prefix_cache([turn1, turn2, turn3], 16)
    assert reused == 48 + 64  # 第 2 轮复用 3 个 block，第 3 轮复用 4 个 block
    assert recomputed == 48 + 16 + 16
    assert reused + recomputed == len(turn1) + len(turn2) + len(turn3)


def test_b_timestamp_at_front_kills_all_reuse():
    system = list(range(100, 132))
    reqs = [[t] + system + [7] * 16 for t in range(5)]  # 每次请求开头的「时间戳」都不同
    reused, recomputed = ex.replay_prefix_cache(reqs, 16)
    assert reused == 0
    assert recomputed == sum(len(r) for r in reqs)


def test_b_timestamp_at_tail_keeps_prefix_reuse():
    system = list(range(100, 132))
    reqs = [system + [t] * 16 for t in range(5)]  # 动态内容放在末尾
    reused, _ = ex.replay_prefix_cache(reqs, 16)
    assert reused == 4 * 32


def test_b_cross_session_sharing_of_project_prefix():
    project = list(range(1000, 1064))  # 同一 project 的 64 token 共享前缀
    s1 = project + [1] * 20
    s2 = project + [2] * 20
    reused, _ = ex.replay_prefix_cache([s1, s2], 16)
    assert reused == 64


def test_b_same_content_after_different_prefix_is_not_a_hit():
    """block 1 的内容和之前某个请求的 block 1 一样，但 block 0 不同 —— 链式 hash 下它的 hash 也不同。"""
    reused, _ = ex.replay_prefix_cache([[1, 2, 3, 4], [9, 9, 3, 4]], 2)
    assert reused == 0


def test_b_hash_present_but_not_contiguous_is_not_a_hit(monkeypatch):
    """构造「第 0 个 block 不在缓存、第 1 个 block 的 hash 恰好在缓存」的情形：只能数连续前缀，命中为 0。"""
    fake = {(1, 2, 3, 4): [101, 102], (9, 9, 3, 4): [999, 102]}
    monkeypatch.setattr(ex, "chain_block_hashes", lambda tokens, block_size, salt=0: fake[tuple(tokens)])
    reused, recomputed = ex.replay_prefix_cache([[1, 2, 3, 4], [9, 9, 3, 4]], 2)
    assert reused == 0
    assert recomputed == 8


# ================================================================== (c) compaction_breakeven_turns


def test_c_docstring_examples():
    got = ex.compaction_breakeven_turns(0, 100_000, 5_000)
    assert got == pytest.approx((10_000 + 25_000 + 6_250) / 9_500)
    assert 4.3 < got < 4.4
    assert ex.compaction_breakeven_turns(20_000, 80_000, 5_000) == pytest.approx(5.5)


def test_c_stable_prefix_makes_payback_slower():
    """同样 10 万 token 的上下文，稳定前缀越大（可压缩的历史越少），回本越慢。"""
    a = ex.compaction_breakeven_turns(0, 100_000, 5_000)
    b = ex.compaction_breakeven_turns(40_000, 60_000, 5_000)
    assert b > a


def test_c_small_history_takes_much_longer_to_pay_back():
    big = ex.compaction_breakeven_turns(0, 100_000, 5_000)
    small = ex.compaction_breakeven_turns(0, 30_000, 5_000)
    assert small > 3 * big


def test_c_never_pays_back_when_summary_not_shorter():
    assert ex.compaction_breakeven_turns(0, 10_000, 10_000) == math.inf
    assert ex.compaction_breakeven_turns(5_000, 10_000, 20_000) == math.inf


def test_c_cheaper_output_makes_compaction_pay_back_sooner():
    assert ex.compaction_breakeven_turns(0, 100_000, 5_000, output_mult=1.0) < ex.compaction_breakeven_turns(
        0, 100_000, 5_000, output_mult=5.0
    )


def test_c_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ex.compaction_breakeven_turns(0, -1, 10)
    with pytest.raises(ValueError):
        ex.compaction_breakeven_turns(-1, 100, 10)
    with pytest.raises(ValueError):
        ex.compaction_breakeven_turns(0, 100, 10, read_mult=0)


# ================================================================== (d) render_request

TOOLS = [
    {"name": "shell_run", "description": "run a command", "params": {"cmd": "string", "timeout": "int"}},
    {"name": "browser_open", "params": {"url": "string"}, "description": "open a page"},
]


def test_d_exact_format():
    rendered, stable_end = ex.render_request(
        [{"name": "b", "x": 1}, {"name": "a"}], "你是助手", [{"role": "user", "content": "hi"}], {"now": "09:00"}
    )
    expected_stable = 'TOOL {"name":"a"}\nTOOL {"name":"b","x":1}\nSYSTEM 你是助手\n'
    assert rendered == expected_stable + 'MSG {"content":"hi","role":"user"}\n' + 'VOLATILE {"now":"09:00"}\n'
    assert stable_end == len(expected_stable)


def test_d_tool_order_and_key_order_do_not_matter():
    shuffled = [
        {"params": {"url": "string"}, "description": "open a page", "name": "browser_open"},
        {"params": {"timeout": "int", "cmd": "string"}, "name": "shell_run", "description": "run a command"},
    ]
    a = ex.render_request(TOOLS, "sys", [], {"t": 1})
    b = ex.render_request(shuffled, "sys", [], {"t": 1})
    assert a == b


def test_d_volatile_only_changes_the_last_line():
    hist = [{"role": "user", "content": "task"}]
    a, end_a = ex.render_request(TOOLS, "sys", hist, {"now": "2026-09-28T09:00:00"})
    b, end_b = ex.render_request(TOOLS, "sys", hist, {"now": "2026-09-28T09:00:01"})
    assert end_a == end_b
    cut = a.rindex("VOLATILE ")
    assert a[:cut] == b[:cut] and a != b


def test_d_append_only_history_keeps_old_request_as_prefix():
    h1 = [{"role": "user", "content": "task"}]
    h2 = h1 + [{"role": "assistant", "content": "call shell_run"}, {"role": "tool", "content": "ok"}]
    old, _ = ex.render_request(TOOLS, "sys", h1, {"budget": 900})
    new, _ = ex.render_request(TOOLS, "sys", h2, {"budget": 800})
    assert new.startswith(old[: old.rindex("VOLATILE ")])


def test_d_system_change_moves_the_stable_prefix():
    a, end_a = ex.render_request(TOOLS, "sys v1", [], {})
    b, end_b = ex.render_request(TOOLS, "sys v2", [], {})
    assert a[:end_a] != b[:end_b]


def test_d_rejects_duplicate_or_missing_names():
    with pytest.raises(ValueError):
        ex.render_request([{"name": "a"}, {"name": "a"}], "s", [], {})
    with pytest.raises(ValueError):
        ex.render_request([{"description": "no name"}], "s", [], {})
