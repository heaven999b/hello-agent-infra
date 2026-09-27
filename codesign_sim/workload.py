"""Agent 负载生成器：多轮 session、同 project 共享前缀、重尾的轮数与输出长度、工具延迟、上下文改写/压缩。

一个 session 的第 t 轮 prompt = 共享前缀 + 前 t 轮的 (new_prompt + output) + 第 t 轮 new_prompt。
这正是 agent harness 的典型形态：上下文只增不减（append-only），所以前缀缓存命中率天然很高；
一旦 harness 改写/压缩历史（rewrite），共享前缀之后的所有 block 都失效，但上下文也变短了
（私有历史只保留 keep_frac，模拟 compaction 的收益）。

数值分布（都是教学用的"合理量级"，不是某个真实产品的统计）：
- 共享前缀（system prompt + 工具定义 + 仓库上下文）：lognormal，中位数 6k token，截断到 [2k, 20k]
- 第 0 轮用户消息：lognormal，中位数 300 token；之后每轮是工具结果：lognormal，中位数 1k token，截断到 [50, 16k]
- 每轮输出：Pareto（形状 tail_alpha），截断到 4096（相当于 max_new_tokens）；**截断后**均值校准为 300
- 轮数：Pareto（形状 tail_alpha），截断到 64（相当于 harness 的 max_turns）；**截断后**均值校准为 mean_turns
  （校准的意义：扫 tail_alpha 时只改尾部、不改平均负载）
- 工具延迟：lognormal，中位数 2 s，截断到 120 s
- project 热度：Zipf，权重 ∝ 1/(k+1)^project_zipf_s（默认 1；0 为均匀）
- session 到达：Poisson 过程，速率 arrival_rate（session/s）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

_HASH_SEED = 0x5EED_CAFE
_MAX_TURNS = 64
_MAX_OUTPUT = 4096


@dataclass
class Turn:
    new_prompt_tokens: int  # 本轮新追加的 token（用户消息 / 工具结果）
    output_tokens: int
    tool_latency_s: float  # 本轮输出之后等待工具的时间（最后一轮为 0）
    rewrite: bool = False  # harness 是否在本轮前改写/压缩了上下文（导致前缀失效）
    keep_frac: float = 1.0  # rewrite 时私有历史保留的比例（1.0 = 只改内容不改长度，<1 = 压缩）


@dataclass
class AgentSession:
    session_id: int
    project_id: int
    shared_prefix_tokens: int  # 同一 project 内所有 session 共享的前缀
    arrival_s: float
    turns: list[Turn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 采样工具
# ---------------------------------------------------------------------------


def _capped_pareto_mean(x_m: float, alpha: float, cap: float) -> float:
    """E[min(X, cap)]，X ~ Pareto(x_m, alpha)，要求 cap >= x_m。"""
    return x_m * alpha / (alpha - 1.0) - x_m**alpha * cap ** (1.0 - alpha) / (alpha - 1.0)


@lru_cache(maxsize=256)
def pareto_scale_for_mean(mean: float, alpha: float, cap: float) -> float:
    """反解 x_m，使 min(Pareto(x_m, alpha), cap) 的均值恰好是 mean（二分；截断后仍保持名义均值）。"""
    if alpha <= 1.0:
        raise ValueError("tail_alpha 必须 > 1")
    if not 0 < mean < cap:
        raise ValueError(f"均值 {mean} 必须在 (0, cap={cap}) 之间")
    lo, hi = 1e-9, mean
    for _ in range(100):
        mid = (lo + hi) / 2
        if _capped_pareto_mean(mid, alpha, cap) < mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _pareto(rng: np.random.Generator, mean: float, alpha: float, cap: float, size: int | None = None):
    """截断到 cap 后均值为 mean 的 Pareto 样本。α 越小尾越长。"""
    x_m = pareto_scale_for_mean(float(mean), float(alpha), float(cap))
    return np.minimum(x_m * (1.0 + rng.pareto(alpha, size)), cap)


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int | None = None):
    return median * np.exp(sigma * rng.standard_normal(size))


def generate_sessions(
    n_sessions: int,
    n_projects: int,
    seed: int = 0,
    *,
    mean_turns: float = 8,
    tail_alpha: float = 1.5,
    rewrite_prob: float = 0.0,
    arrival_rate: float = 1.0,
    rewrite_keep_frac: float = 0.5,
    project_zipf_s: float = 1.0,
    auto_compact_tokens: int | None = 100_000,
) -> list[AgentSession]:
    """生成 n_sessions 个 agent session（在同一 Python/NumPy 版本下，结果只由参数和 seed 决定）。

    - tail_alpha 控制轮数与输出长度的尾部（均值已校准，不随 tail_alpha 变）
    - rewrite_prob：第 1 轮起每轮"改写/压缩上下文"的概率；rewrite_keep_frac：压缩后私有历史保留的比例
    - project_zipf_s：project 热度的 Zipf 指数（默认 1.0 与旧版相同；0 = 均匀；越大越集中在头部 project）
    - auto_compact_tokens：某轮上下文（含本轮输出）将超过这个长度时，harness 在该轮前自动压缩
      （rewrite=True，keep_frac = min(rewrite_keep_frac, 0.5)），模拟 agent 接近上下文上限时的 auto-compact；
      None 表示不自动压缩
    """
    if n_sessions <= 0 or n_projects <= 0:
        raise ValueError("n_sessions 和 n_projects 必须 >= 1")
    if not 0.0 <= rewrite_keep_frac <= 1.0:
        raise ValueError("rewrite_keep_frac 必须在 [0, 1]")
    if project_zipf_s < 0:
        raise ValueError("project_zipf_s 必须 >= 0")
    rng = np.random.default_rng(seed)

    # 每个 project 一个固定的共享前缀长度
    prefix = np.clip(_lognormal(rng, 6000, 0.5, n_projects), 2000, 20000).astype(int)
    weights = 1.0 / np.arange(1, n_projects + 1) ** project_zipf_s
    weights /= weights.sum()
    projects = rng.choice(n_projects, size=n_sessions, p=weights)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, n_sessions))
    arrivals -= arrivals[0]  # 第一个 session 在 t=0 到达

    sessions: list[AgentSession] = []
    for sid in range(n_sessions):
        n_turns = int(max(1, round(float(_pareto(rng, mean_turns, tail_alpha, _MAX_TURNS)))))
        prompts = np.clip(_lognormal(rng, 1000, 1.0, n_turns), 50, 16000).astype(int)
        prompts[0] = int(np.clip(_lognormal(rng, 300, 0.8), 20, 4000))
        outputs = np.maximum(1, np.round(_pareto(rng, 300, tail_alpha, _MAX_OUTPUT, n_turns))).astype(int)
        tools = np.clip(_lognormal(rng, 2.0, 1.0, n_turns), 0.05, 120.0)
        rewrites = rng.random(n_turns) < rewrite_prob
        pid = int(projects[sid])
        turns, priv = [], 0  # priv：共享前缀之后的私有历史长度（与 _private_segments 的算法一致）
        for t in range(n_turns):
            rw, keep = (bool(rewrites[t]) if t > 0 else False), rewrite_keep_frac
            step = int(prompts[t]) + int(outputs[t])
            if t > 0 and not rw and auto_compact_tokens and prefix[pid] + priv + step > auto_compact_tokens:
                rw, keep = True, min(rewrite_keep_frac, 0.5)
            if rw:
                priv = int(round(priv * keep))
            priv += step
            turns.append(
                Turn(
                    new_prompt_tokens=int(prompts[t]),
                    output_tokens=int(outputs[t]),
                    tool_latency_s=0.0 if t == n_turns - 1 else float(round(tools[t], 3)),
                    rewrite=rw,
                    keep_frac=keep if rw else 1.0,
                )
            )
        sessions.append(
            AgentSession(
                session_id=sid,
                project_id=pid,
                shared_prefix_tokens=int(prefix[pid]),
                arrival_s=float(round(arrivals[sid], 3)),
                turns=turns,
            )
        )
    return sessions


# ---------------------------------------------------------------------------
# 上下文的"段"表示：共享前缀 + 私有段（rewrite 时私有历史被压缩成一段摘要）
# ---------------------------------------------------------------------------


def _seg_id(*key: int) -> int:
    # 只用整数元组：CPython 对 int/tuple 的 hash 不随进程随机化（PYTHONHASHSEED 无影响），
    # 同一 CPython 版本下结果可复现；不同 Python 实现/版本之间不保证
    return hash(key)


def _rewrite_version(session: AgentSession, upto_turn: int) -> int:
    return sum(1 for t in session.turns[1 : upto_turn + 1] if t.rewrite)


def _private_segments(session: AgentSession, upto_turn: int, include_output: bool) -> list[tuple[int, int]]:
    """共享前缀之后的私有段 [(段 id, token 数)]。

    第 i 轮（i>=1）rewrite=True 时：之前的私有历史被压缩成一段"摘要"，长度 = round(原长度 · keep_frac)，
    内容全新（段 id 变化），所以共享前缀之后的所有 block hash 都会变。
    """
    sid, priv, version = session.session_id, [], 0
    for i, t in enumerate(session.turns[: upto_turn + 1]):
        if i > 0 and t.rewrite:
            version += 1
            kept = int(round(sum(n for _, n in priv) * t.keep_frac))
            priv = [(_seg_id(2, sid, version), kept)] if kept > 0 else []
        priv.append((_seg_id(1, sid, i, 0, version), t.new_prompt_tokens))
        if i < upto_turn or include_output:
            priv.append((_seg_id(1, sid, i, 1, version), t.output_tokens))
    return priv


def context_tokens(session: AgentSession, upto_turn: int, include_output: bool = False) -> int:
    """第 upto_turn 轮送进模型的 prompt 长度（include_output=True 时再加上本轮输出）；已计入 rewrite 压缩。"""
    _check_turn(session, upto_turn)
    return session.shared_prefix_tokens + sum(n for _, n in _private_segments(session, upto_turn, include_output))


class BlockHasher:
    """把"段"（segment：一段内容相同的 token）按 block_size 切块并做链式 hash。

    block i 的 hash = hash(block i-1 的 hash, 本 block 覆盖的各段 (段 id, 段内偏移, 长度))，
    和 vLLM 的 prefix caching 一样用链式 hash 近似 block 身份：**在忽略 hash 碰撞的前提下**，
    hash 相同 ⇔ 从开头到这个 block 的内容完全相同。
    只输出满 block；不满一个 block 的尾巴留在 pending，等下一段补齐。
    """

    def __init__(self, block_size: int = 16):
        if block_size <= 0:
            raise ValueError("block_size 必须 >= 1")
        self.block_size = block_size
        self.hashes: list[int] = []
        self._h = _HASH_SEED
        self._pending: list[tuple[int, int, int]] = []
        self._pending_len = 0

    def copy(self) -> BlockHasher:
        other = BlockHasher(self.block_size)
        other.hashes = list(self.hashes)
        other._h = self._h
        other._pending = list(self._pending)
        other._pending_len = self._pending_len
        return other

    def append(self, seg_id: int, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        bs, off = self.block_size, 0
        if self._pending_len:  # 先补齐上一段留下的半个 block
            take = min(n_tokens, bs - self._pending_len)
            self._pending.append((seg_id, 0, take))
            self._pending_len += take
            off = take
            if self._pending_len == bs:
                self._h = hash((self._h, *self._pending))
                self.hashes.append(self._h)
                self._pending, self._pending_len = [], 0
        h, out = self._h, self.hashes
        while n_tokens - off >= bs:  # 整块落在同一段里
            h = hash((h, seg_id, off))
            out.append(h)
            off += bs
        self._h = h
        if n_tokens - off > 0:
            self._pending = [(seg_id, off, n_tokens - off)]
            self._pending_len = n_tokens - off


@lru_cache(maxsize=4096)
def _prefix_hasher(project_id: int, prefix_tokens: int, block_size: int) -> BlockHasher:
    h = BlockHasher(block_size)
    h.append(_seg_id(0, project_id), prefix_tokens)
    return h


def build_hasher(session: AgentSession, upto_turn: int, block_size: int = 16, include_output: bool = False) -> BlockHasher:
    """构造到第 upto_turn 轮为止的 BlockHasher（router 用它做增量 hash）。"""
    _check_turn(session, upto_turn)
    h = _prefix_hasher(session.project_id, session.shared_prefix_tokens, block_size).copy()
    for seg, n in _private_segments(session, upto_turn, include_output):
        h.append(seg, n)
    return h


def append_output(hasher: BlockHasher, session: AgentSession, turn: int) -> None:
    """在已经包含第 turn 轮 prompt 的 hasher 上追加第 turn 轮的输出（router 的增量路径）。"""
    version = _rewrite_version(session, turn)
    hasher.append(_seg_id(1, session.session_id, turn, 1, version), session.turns[turn].output_tokens)


def append_prompt(hasher: BlockHasher, session: AgentSession, turn: int) -> None:
    """在包含第 turn-1 轮输出的 hasher 上追加第 turn 轮 prompt（要求第 turn 轮没有 rewrite）。"""
    version = _rewrite_version(session, turn)
    hasher.append(_seg_id(1, session.session_id, turn, 0, version), session.turns[turn].new_prompt_tokens)


def to_blocks(session: AgentSession, upto_turn: int, block_size: int = 16, include_output: bool = False) -> list[int]:
    """把 session 第 upto_turn 轮的 prompt（include_output=True 时含本轮输出）变成 block hash 序列。

    - 同 project 的 session：共享前缀部分的 hash 完全相同（前缀长度不是 block_size 整数倍时，
      跨越边界的那个 block 混入了私有内容，hash 不同）
    - 某轮 rewrite=True：从该轮起，共享前缀之后的所有 block hash 都变化（上下文按 keep_frac 缩短）
    - 只返回满 block（和 vLLM 一样，最后不满一个 block 的 token 不参与缓存）
    """
    return list(build_hasher(session, upto_turn, block_size, include_output).hashes)


def _check_turn(session: AgentSession, upto_turn: int) -> None:
    if not 0 <= upto_turn < len(session.turns):
        raise IndexError(f"upto_turn={upto_turn} 超出范围 [0, {len(session.turns)})")


def session_stats(sessions: list[AgentSession]) -> dict:
    """负载概览：轮数、每轮 prompt 长度、输出长度、prompt:输出比等（都是**实测**值，方便 lab 打印）。"""
    n_turns = np.array([len(s.turns) for s in sessions])
    prompt_lens, outputs, rewrites = [], [], 0
    for s in sessions:
        for t in range(len(s.turns)):
            prompt_lens.append(context_tokens(s, t))
            outputs.append(s.turns[t].output_tokens)
            rewrites += s.turns[t].rewrite
    prompt_lens_a = np.array(prompt_lens)
    return {
        "n_sessions": len(sessions),
        "n_requests": int(n_turns.sum()),
        "turns_mean": float(n_turns.mean()),
        "turns_p99": float(np.percentile(n_turns, 99)),
        "prompt_tokens_mean": float(prompt_lens_a.mean()),
        "prompt_tokens_p50": float(np.percentile(prompt_lens_a, 50)),
        "prompt_tokens_p99": float(np.percentile(prompt_lens_a, 99)),
        "output_tokens_mean": float(np.mean(outputs)),
        "output_tokens_p99": float(np.percentile(outputs, 99)),
        "prompt_to_output_ratio": float(prompt_lens_a.sum() / max(1, np.sum(outputs))),
        "rewrite_frac": rewrites / max(1, int(n_turns.sum())),
    }
