"""workload：确定性、均值校准、热度分布、rewrite 压缩、auto-compact、block hash 的共享与失效。"""

import numpy as np
import pytest

from codesign_sim.workload import (
    append_output,
    append_prompt,
    build_hasher,
    context_tokens,
    generate_sessions,
    session_stats,
    to_blocks,
)


def test_deterministic_and_well_formed():
    a = generate_sessions(50, 5, seed=3)
    b = generate_sessions(50, 5, seed=3)
    c = generate_sessions(50, 5, seed=4)
    assert a == b and a != c
    for s in a:
        assert s.turns and s.turns[-1].tool_latency_s == 0.0
        assert all(t.new_prompt_tokens > 0 and t.output_tokens > 0 for t in s.turns)
    assert [s.arrival_s for s in a] == sorted(s.arrival_s for s in a)
    by_proj = {}
    for s in a:  # 同一 project 的共享前缀长度相同
        assert by_proj.setdefault(s.project_id, s.shared_prefix_tokens) == s.shared_prefix_tokens


def test_tail_alpha_changes_tail_not_mean():
    heavy = session_stats(generate_sessions(1000, 4, seed=0, tail_alpha=1.2))
    light = session_stats(generate_sessions(1000, 4, seed=0, tail_alpha=3.0))
    for st in (heavy, light):  # 截断后的均值已校准：8 轮、300 token
        assert st["turns_mean"] == pytest.approx(8, rel=0.08)
        assert st["output_tokens_mean"] == pytest.approx(300, rel=0.08)
    assert heavy["output_tokens_p99"] > 1.5 * light["output_tokens_p99"]
    assert heavy["turns_p99"] > light["turns_p99"]


def test_project_popularity_parameter():
    count = lambda s: np.bincount([x.project_id for x in generate_sessions(1000, 8, seed=0, project_zipf_s=s)],
                                  minlength=8) / 1000  # noqa: E731
    uniform, default, skewed = count(0.0), count(1.0), count(2.0)
    assert uniform.max() < 0.17  # 均匀：每个约 1/8
    assert skewed[0] > default[0] > uniform[0] + 0.1  # 指数越大越集中在头部 project
    assert default[0] == pytest.approx(1 / sum(1 / k for k in range(1, 9)), abs=0.05)  # 默认 = 旧版 Zipf(1)


def test_rewrite_compacts_context():
    ss = generate_sessions(200, 2, seed=5, rewrite_prob=0.5, rewrite_keep_frac=0.3)
    s = next(s for s in ss if len(s.turns) >= 3 and s.turns[2].rewrite)
    prefix = s.shared_prefix_tokens
    before = context_tokens(s, 1, include_output=True)
    after = context_tokens(s, 2)
    assert after == prefix + round((before - prefix) * 0.3) + s.turns[2].new_prompt_tokens
    same = generate_sessions(200, 2, seed=5, rewrite_prob=0.5, rewrite_keep_frac=1.0)
    s1 = next(x for x in same if x.session_id == s.session_id)
    assert context_tokens(s1, 2) == context_tokens(s1, 1, include_output=True) + s1.turns[2].new_prompt_tokens


def test_auto_compact_caps_context():
    ss = generate_sessions(300, 4, seed=1, mean_turns=20)
    longest = max(context_tokens(s, t, True) for s in ss for t in range(len(s.turns)))
    assert longest <= 100_000 + 16000 + 4096  # 超过阈值前自动压缩（最多再加一轮）
    raw = generate_sessions(300, 4, seed=1, mean_turns=20, auto_compact_tokens=None)
    assert max(context_tokens(s, t, True) for s in raw for t in range(len(s.turns))) > longest


def test_blocks_length_and_append_only():
    s = max(generate_sessions(30, 2, seed=0), key=lambda s: len(s.turns))
    assert len(s.turns) >= 3
    for t in range(len(s.turns)):
        assert len(to_blocks(s, t)) == context_tokens(s, t) // 16
    b0, b1 = to_blocks(s, 0, include_output=True), to_blocks(s, 1)
    assert b1[: len(b0)] == b0  # 没有 rewrite：下一轮 prompt 以上一轮 prompt+输出 为前缀
    assert len(to_blocks(s, 0, block_size=64)) == context_tokens(s, 0) // 64


def test_shared_prefix_same_across_sessions_of_project():
    ss = generate_sessions(40, 3, seed=2)
    s1, s2 = [s for s in ss if s.project_id == 0][:2]
    n_shared = s1.shared_prefix_tokens // 16
    assert to_blocks(s1, 0)[:n_shared] == to_blocks(s2, 0)[:n_shared]
    assert to_blocks(s1, 0)[n_shared + 1 :] != to_blocks(s2, 0)[n_shared + 1 :]
    other = next(s for s in ss if s.project_id != 0)
    assert to_blocks(other, 0)[0] != to_blocks(s1, 0)[0]


def test_rewrite_invalidates_private_blocks_only():
    ss = generate_sessions(200, 2, seed=5, rewrite_prob=0.5)
    s = next(s for s in ss if len(s.turns) >= 3 and s.turns[2].rewrite)
    before, after = to_blocks(s, 1, include_output=True), to_blocks(s, 2)
    n_shared = s.shared_prefix_tokens // 16
    assert after[:n_shared] == before[:n_shared]
    assert not set(after[n_shared + 1 :]) & set(before[n_shared + 1 :])


def test_incremental_hasher_matches_to_blocks():
    ss = generate_sessions(60, 3, seed=7, rewrite_prob=0.2)
    s = max(ss, key=lambda s: len(s.turns))
    h = build_hasher(s, 0)
    for t in range(len(s.turns)):
        if t > 0:
            if s.turns[t].rewrite:
                h = build_hasher(s, t)
            else:
                append_prompt(h, s, t)
        assert h.hashes == to_blocks(s, t)
        append_output(h, s, t)
        assert h.hashes == to_blocks(s, t, include_output=True)
