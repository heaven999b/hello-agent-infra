"""模块 04 实验：同一批 agent 会话，不同的 harness 写法，前缀缓存能省下多少 prefill、多少钱。

跑法：
    .venv/bin/python modules/04_harness_workloads/lab.py

做了什么：
1. 用 codesign_sim.workload 生成一批多项目、多轮的 agent 会话（固定 seed，结果确定）。
2. 按请求时间顺序送进一个「足够大」的 PrefixCache（不淘汰），只看 harness 写法本身的影响。
   每种写法都显式地构造「这次请求真正发出去的 token 段序列」，再按 vLLM 的方式做链式 block hash：
   - append_only        ：只追加，从不改写历史（理想写法；自托管引擎语义，同项目会话共享前缀）
   - api_auto_only      ：托管 API 只用自动缓存、没在共享前缀末尾放断点 → 只剩会话内复用（README 2.3）；
                          而且 API 只缓存请求的输入：上一轮的输出要等下一轮作为输入提交时才写入（按写入价）
   - ts_head_seconds    ：system prompt 最前面放精确到秒的时间戳
   - ts_head_day        ：同上，但只精确到「日期」
   - ts_tail_kept       ：时间戳放在每轮新消息末尾，并且留在历史里（只追加）
   - ts_tail_dropped    ：时间戳放在本轮末尾，下一轮从历史里删掉 → 上一轮输出的 KV 是在时间戳之后算的，对不上
   - dynamic_tools_0.2  ：每轮以 20% 概率增删工具（工具定义在前缀最前面）
   - rewrite / compact  ：用 codesign_sim 的 rewrite：只改写不缩短（keep_frac=1），或把历史压成 10% 的摘要（keep_frac=0.1）
3. 打印：prompt 总量、token 命中率、需要重算的 prefill token、账单（按 Claude Sonnet 5 价格估）。
   压缩那几行同时计入「做摘要」的代价：读一遍整段上下文（命中价）+ 生成摘要（输出价）。
4. 再按会话轮数 T 分组，验证复用倍数 R ∈ [T/2, T]（README 3.1）。
"""

from __future__ import annotations

import random

from codesign_sim.cache import PrefixCache
from codesign_sim.workload import AgentSession, context_tokens, generate_sessions, to_blocks

SEED = 0
N_SESSIONS = 160
N_PROJECTS = 8
BLOCK = 64  # 块越大模拟越快；vLLM 默认 16，Mooncake trace 用 512
CAPACITY_BLOCKS = 10**8  # 足够大：本实验不研究淘汰（那是模块 05 的事）
DECODE_S_PER_TOKEN = 0.02  # 假设：单会话 decode 约 50 token/s，只用来给请求排时间顺序
TS_TOKENS = 16  # 假设：一行时间戳约 16 个 token
# Claude Sonnet 5，$/百万 token：输入、5 分钟缓存写入、缓存读取、输出（官方文档，查询于 2026-09-28）
# 计费假设：每个请求末尾都有缓存断点（自动缓存），所以未命中的 token 都按写入价计
PRICE_INPUT, PRICE_WRITE, PRICE_READ, PRICE_OUTPUT = 2.0, 2.5, 0.2, 10.0

Seg = tuple[tuple[int, ...], int]  # (段的身份, token 数)；段的身份只用整数元组，hash 可复现


class SegHasher:
    """把 token 段按 BLOCK 切块做链式 hash（与 vLLM 相同的语义：block 的 hash 依赖它之前的全部内容）。

    root 是链的根：放进不同的 root，相当于在最前面多了一段不同的内容（例如 cache_salt、租户隔离）。
    只输出满 block；不满一个 block 的尾巴留到下一段补齐。
    注：用 Python 内置 hash（int 元组），同一 Python 版本下可复现；真实系统用 sha256 这类稳定 hash。
    """

    def __init__(self, root: tuple[int, ...] = (0,)):
        self.hashes: list[int] = []
        self._h = hash(root)
        self._pending: list[tuple] = []
        self._pending_len = 0

    def copy(self) -> SegHasher:
        other = SegHasher()
        other.hashes, other._h = list(self.hashes), self._h
        other._pending, other._pending_len = list(self._pending), self._pending_len
        return other

    def add(self, seg: Seg) -> SegHasher:
        key, n = seg
        off = 0
        if self._pending_len and n > 0:  # 先补齐上一段留下的半个 block
            take = min(n, BLOCK - self._pending_len)
            self._pending.append((key, 0, take))
            self._pending_len += take
            off = take
            if self._pending_len == BLOCK:
                self._h = hash((self._h, *self._pending))
                self.hashes.append(self._h)
                self._pending, self._pending_len = [], 0
        while n - off >= BLOCK:
            self._h = hash((self._h, key, off))
            self.hashes.append(self._h)
            off += BLOCK
        if n - off > 0:
            self._pending, self._pending_len = [(key, off, n - off)], n - off
        return self


# ---------------------------------------------------------------------------
# 请求时间表与账单
# ---------------------------------------------------------------------------


def request_schedule(sessions: list[AgentSession]) -> list[tuple[float, int, int]]:
    """(请求时刻, session 下标, 轮次)，按时间排序。时刻 = 到达 + 之前各轮的 decode 时间 + 工具耗时。"""
    events = []
    for i, s in enumerate(sessions):
        t = s.arrival_s
        for k, turn in enumerate(s.turns):
            events.append((t, i, k))
            t += turn.output_tokens * DECODE_S_PER_TOKEN + turn.tool_latency_s
    events.sort()
    return events


def bill(prompt: int, hit: int, out: int, extra_read: int = 0, extra_out: int = 0) -> tuple[float, float]:
    """返回 (有缓存的账单, 不缓存的账单)，单位美元。extra_*：做摘要那次调用读的上下文与生成的摘要。"""
    cached = ((prompt - hit) * PRICE_WRITE + (hit + extra_read) * PRICE_READ + (out + extra_out) * PRICE_OUTPUT) / 1e6
    nocache = ((prompt + extra_read) * PRICE_INPUT + (out + extra_out) * PRICE_OUTPUT) / 1e6
    return cached, nocache


# ---------------------------------------------------------------------------
# 各种 harness 写法：每一轮「发出去的 prompt 段」和「引擎算完后缓存里的段」
# ---------------------------------------------------------------------------


def replay_segments(sessions: list[AgentSession], variant: str, tool_version: dict | None = None) -> dict:
    """用段模型重放一种 harness 写法。返回 prompt 总量、命中量、输出量，以及每个会话的 [prompt, miss]。"""
    cache = PrefixCache(CAPACITY_BLOCKS, policy="lru")
    state: dict[int, SegHasher] = {}  # 每个会话「下一轮 prompt 的历史部分」的增量 hash 状态
    prompt_total = hit_total = out_total = 0
    per_session = {s.session_id: [0, 0] for s in sessions}
    for now, i, k in request_schedule(sessions):
        s, turn = sessions[i], sessions[i].turns[k]
        sid, pid = s.session_id, s.project_id
        prefix_key = (0, pid)
        root = (0,)
        if variant in ("api_auto_only", "isolated"):
            root = (1, sid)  # 跨会话拿不到共享前缀：等价于每个会话一条独立的链
        if variant == "dynamic_tools_0.2" and tool_version[(sid, k)] > 0:
            prefix_key = (0, pid, sid, tool_version[(sid, k)])  # 工具集变了：前缀从第 0 个 token 起就不同
        head: list[Seg] = []
        if variant == "ts_head_seconds":
            head = [((9, int(now)), TS_TOKENS)]
        elif variant == "ts_head_day":
            head = [((9, int(now // 86400)), TS_TOKENS)]
        if variant in ("ts_head_seconds", "ts_head_day", "dynamic_tools_0.2") or k == 0:
            # 前缀可能变了：从头重建历史（这些写法只有前缀会变，历史本身仍是只追加）
            h = SegHasher(root)
            for seg in head + [(prefix_key, s.shared_prefix_tokens)]:
                h.add(seg)
            for j in range(k):
                h.add(((1, sid, j), s.turns[j].new_prompt_tokens))
                if variant == "ts_tail_kept":
                    h.add(((2, sid, j), TS_TOKENS))
                h.add(((3, sid, j), s.turns[j].output_tokens))
        else:
            h = state[sid]
        h.add(((1, sid, k), turn.new_prompt_tokens))
        sent = h.copy()
        n_prompt = sum(n for _, n in head) + s.shared_prefix_tokens + sum(
            t.new_prompt_tokens + t.output_tokens for t in s.turns[:k]
        ) + turn.new_prompt_tokens
        if variant in ("ts_tail_kept", "ts_tail_dropped"):
            sent.add(((2, sid, k), TS_TOKENS))  # 本轮时间戳放在最后
            n_prompt += TS_TOKENS * (k + 1 if variant == "ts_tail_kept" else 1)
        hit = cache.match(sent.hashes[: n_prompt // BLOCK], now)
        if variant == "api_auto_only":
            cache.insert(sent.hashes, now)  # 托管 API 只写请求的输入前缀；本轮输出下一轮作为新输入再写入
        else:
            computed = sent.copy().add(((3, sid, k), turn.output_tokens))  # 自托管引擎：输出的 KV 紧跟在本轮 prompt 之后
            cache.insert(computed.hashes, now)
        # 下一轮的历史：dropped 版本的历史里没有时间戳，所以和引擎里算过的 KV 在时间戳处分叉
        base = sent if variant == "ts_tail_kept" else h
        state[sid] = base.copy().add(((3, sid, k), turn.output_tokens))
        prompt_total += n_prompt
        hit_total += hit * BLOCK
        out_total += turn.output_tokens
        per_session[sid][0] += n_prompt
        per_session[sid][1] += n_prompt - hit * BLOCK
    return {"prompt": prompt_total, "hit": hit_total, "out": out_total, "per_session": per_session,
            "extra_read": 0, "extra_out": 0}


def replay_library(sessions: list[AgentSession]) -> dict:
    """压缩 / 改写的几行：直接用 codesign_sim 的 to_blocks（rewrite 轮的上下文按 keep_frac 缩短）。"""
    cache = PrefixCache(CAPACITY_BLOCKS, policy="lru")
    prompt_total = hit_total = out_total = extra_read = extra_out = 0
    for now, i, k in request_schedule(sessions):
        s = sessions[i]
        n_prompt = context_tokens(s, k)
        full = to_blocks(s, k, BLOCK, include_output=True)
        hit = cache.match(full[: n_prompt // BLOCK], now)
        cache.insert(full, now)
        prompt_total += n_prompt
        hit_total += hit * BLOCK
        out_total += s.turns[k].output_tokens
        if k > 0 and s.turns[k].rewrite and s.turns[k].keep_frac < 1.0:
            # 真正的压缩（keep_frac < 1）才有摘要调用：读一遍压缩前的整段上下文（按命中计），生成摘要（按输出计）
            before = context_tokens(s, k - 1, include_output=True)
            summary = n_prompt - s.shared_prefix_tokens - s.turns[k].new_prompt_tokens
            extra_read += before
            extra_out += max(0, summary)
    return {"prompt": prompt_total, "hit": hit_total, "out": out_total, "per_session": None,
            "extra_read": extra_read, "extra_out": extra_out}


def tool_versions(sessions: list[AgentSession], p_change: float) -> dict[tuple[int, int], int]:
    """每个会话每轮以 p_change 概率换一次工具集；版本 > 0 时整条前缀都和别人不同。"""
    rng = random.Random(SEED)
    version: dict[tuple[int, int], int] = {}
    for s in sessions:
        v = 0
        for k in range(len(s.turns)):
            if k > 0 and rng.random() < p_change:
                v += 1
            version[(s.session_id, k)] = v
    return version


def sessions_with(**kw) -> list[AgentSession]:
    # auto_compact_tokens=None：关掉库里「接近上限自动压缩」，让每一行只体现这一行的写法
    return generate_sessions(N_SESSIONS, N_PROJECTS, seed=SEED, auto_compact_tokens=None, **kw)


def part_a() -> None:
    base = sessions_with()
    n_req = sum(len(s.turns) for s in base)
    rows: list[tuple[str, dict]] = [
        ("append_only", replay_segments(base, "append_only")),
        ("api_auto_only", replay_segments(base, "api_auto_only")),
        ("ts_head_seconds", replay_segments(base, "ts_head_seconds")),
        ("ts_head_day", replay_segments(base, "ts_head_day")),
        ("ts_tail_kept", replay_segments(base, "ts_tail_kept")),
        ("ts_tail_dropped", replay_segments(base, "ts_tail_dropped")),
        ("dynamic_tools_0.2", replay_segments(base, "dynamic_tools_0.2", tool_versions(base, 0.2))),
        ("rewrite_p0.1_keep1.0", replay_library(sessions_with(rewrite_prob=0.1, rewrite_keep_frac=1.0))),
        ("compact_p0.1_keep0.1", replay_library(sessions_with(rewrite_prob=0.1, rewrite_keep_frac=0.1))),
        ("compact_p0.3_keep0.1", replay_library(sessions_with(rewrite_prob=0.3, rewrite_keep_frac=0.1))),
        ("rewrite_every_turn", replay_library(sessions_with(rewrite_prob=1.0, rewrite_keep_frac=1.0))),
    ]
    ideal = rows[0][1]
    print("=" * 104)
    print(f"A. 负载概况：{N_SESSIONS} 个会话 / {N_PROJECTS} 个项目 / {n_req} 次请求（seed={SEED}，block={BLOCK} token）")
    print(
        f"   只追加时 prompt 总量 {ideal['prompt'] / 1e6:.2f}M token，输出总量 {ideal['out'] / 1e6:.3f}M token，"
        f"按任务累计的输入:输出 ≈ {ideal['prompt'] / ideal['out']:.0f}:1"
    )
    print("-" * 104)
    print(f"{'harness 写法':<22}{'prompt 总量':>11}{'命中率':>9}{'重算 prefill':>13}{'× 理想':>8}{'账单($)':>10}{'不缓存($)':>11}{'省下':>7}")
    for name, r in rows:
        cost, nocache = bill(r["prompt"], r["hit"], r["out"], r["extra_read"], r["extra_out"])
        miss = r["prompt"] - r["hit"]
        print(
            f"{name:<22}{r['prompt'] / 1e6:>10.2f}M{r['hit'] / r['prompt']:>9.1%}{miss / 1e6:>12.2f}M"
            f"{miss / (ideal['prompt'] - ideal['hit']):>8.1f}{cost:>10.2f}{nocache:>11.2f}{1 - cost / nocache:>7.0%}"
        )


def part_b() -> None:
    sessions = sessions_with()
    shared = replay_segments(sessions, "append_only")["per_session"]
    alone = replay_segments(sessions, "isolated")["per_session"]  # 引擎语义，但每个会话一条独立的链
    print("=" * 104)
    print("B. 只追加的 harness：按会话轮数 T 分组看复用倍数 R = prompt 总量 / 重算量（README 3.1）")
    print("   「会话内」只看本会话的复用，应落在 [T/2, T]；「含跨会话」再加上同项目共享前缀带来的复用")
    print("-" * 104)
    buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 64)]
    print(f"{'轮数 T':>10}{'会话数':>8}{'平均 T':>9}{'[T/2, T]':>14}{'R（会话内）':>14}{'R（含跨会话）':>16}")
    for lo, hi in buckets:
        group = [s for s in sessions if lo <= len(s.turns) <= hi]
        if not group:
            continue
        t_mean = sum(len(s.turns) for s in group) / len(group)
        r_alone = sum(alone[s.session_id][0] for s in group) / sum(alone[s.session_id][1] for s in group)
        r_shared = sum(shared[s.session_id][0] for s in group) / sum(shared[s.session_id][1] for s in group)
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        bound = f"[{t_mean / 2:.1f}, {t_mean:.1f}]"
        print(f"{label:>10}{len(group):>8}{t_mean:>9.1f}{bound:>14}{r_alone:>14.1f}{r_shared:>16.1f}")


if __name__ == "__main__":
    part_a()
    part_b()
