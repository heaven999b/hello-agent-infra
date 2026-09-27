"""RL rollout 调度模拟：同步（sync）vs 部分 rollout（partial）vs 异步（async），在**相同 GPU 总数**下比较。

抽象：
- 集群 = n_workers 个"单位"。一个 rollout 槽位（slot）一次跑一条轨迹，占 1 个单位
- 一条轨迹 = 生成时间 gen（Pareto 重尾，截断到 50×均值，**截断后**均值校准为 mean_gen_s，相当于 max_new_tokens）
  + 工具等待 tool（指数分布，均值 mean_tool_s）；两者按比例均匀交织。生成 token 数 = gen × decode_tok_s
- 一个训练步在整个集群上耗时 train_step_s（GPU·秒 = n_workers · train_step_s）

部署（deploy）：
- "colocated"（共置，默认）：所有单位轮流做 rollout 和训练；训练期间生成全部暂停
- "disaggregated"（分离）：trainer_frac 的单位专门训练（一步耗时 train_step_s / trainer_frac），
  其余 round(n_workers·(1-trainer_frac)) 个槽位专门 rollout；只有 async 能让两边重叠

模式：
- sync：每步生成 batch_size 条新轨迹，等**最慢**的一条结束才训练（= async 且 max_staleness=0）
- partial（APRIL / Kimi k1.5 思路）：过量发射，凑满 batch_size 条完成的轨迹就结束本轮，在途的暂停、下一轮用新策略续写；
  partial_budget_s 是每条轨迹**每轮**最多推进的时长（或用 partial_budget_tokens 指定每轮最多生成的 token），
  超出的也暂停到下一轮。训练 batch 始终是 batch_size。生成与训练不重叠
- async（AReaL，arXiv 2505.24298）：rollout 连续进行，trainer 凑满 batch 就训练（优先用旧轨迹）。
  提交新轨迹时按 AReaL 式 (3) 限流：⌊(N_r - 1)/B⌋ ≤ i + η（N_r = 已提交且未被丢弃的轨迹数，i = 当前版本）。
  stale_policy：
  - "interrupt"（默认，AReaL 语义）：权重更新时打断所有在途生成，丢弃旧 KV、用新权重**重算**后继续解码，
    一条轨迹由多个版本的片段组成；不丢任何数据
  - "abort"（对照）：在途轨迹一旦版本差 > η 就中止，训练时丢弃过旧样本（释放限流名额）——会系统性丢掉长轨迹

重算成本（resume_kv="recompute"，默认）：续写/被打断的轨迹要把当前上下文（prompt_tokens + 已生成 token）重新 prefill。
- 批量场景（权重更新时所有在途轨迹同时被打断、新一轮开始时成批续写）：整个 rollout 集群一起做重算，
  所有轨迹等待同样的时长 = Σ prefill_flops(ctx_i) / (槽位数 · gpus_per_worker · 峰值 · mfu)
- 单条续写（partial 里单条轨迹预算用完后在本轮内被续写）：按一个推理引擎（engine_gpus 张卡）的速度算
重算期间槽位被占用（计入 GPU 忙碌，但不产出新 token）；"retain" 表示沿用旧 KV、不计成本。

staleness：按轨迹 = 训练版本 - 最旧片段的版本；按 token 加权 = Σ 片段 token ×（训练版本 - 片段版本）/ Σ token。
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .hardware import GPUS, peak_tflops
from .model import MODELS, prefill_flops
from .workload import pareto_scale_for_mean

_MODES = ("sync", "partial", "async")


@dataclass
class RolloutConfig:
    n_workers: int = 64
    batch_size: int = 256
    tail_alpha: float = 1.5
    mean_gen_s: float = 20.0
    mean_tool_s: float = 5.0
    train_step_s: float = 30.0
    mode: str = "sync"  # "sync" | "partial" | "async"
    max_staleness: int = 1  # async 模式下的 η
    partial_budget_s: float = 60.0  # partial：每条轨迹每轮最多推进多久，未完成的留到下一轮续写
    # ---- 以下是库扩展字段（契约之外，都有默认值）----
    deploy: str = "colocated"  # "colocated" | "disaggregated"
    trainer_frac: float = 0.25  # disaggregated：训练占用的单位比例
    stale_policy: str = "interrupt"  # async："interrupt"（AReaL）| "abort"（对照）
    resume_kv: str = "recompute"  # "recompute" | "retain"
    prompt_tokens: int = 2048
    decode_tok_s: float = 100.0  # 每条轨迹的解码速度（token/s）
    model: str = "qwen3-32b"  # 重算成本用的模型
    gpu: str = "H100-SXM"
    gpus_per_worker: float = 0.125  # 每个槽位分到的 GPU（批量重算时的算力份额）
    engine_gpus: int = 8  # 一个推理引擎的卡数（单条续写的重算速度）
    mfu: float = 0.6
    partial_budget_tokens: int | None = None  # partial：每条轨迹每轮最多生成的 token（优先于 partial_budget_s）
    over_provision: float = 1.0  # partial：每轮最多新发射 batch_size · over_provision + n_workers 条


class _Traj:
    __slots__ = ("tid", "gen", "gen_frac", "v_start", "remaining", "gen_done", "pending_rc", "segments", "budget_left")

    def __init__(self, tid: int, gen: float, tool: float, version: int):
        self.tid = tid
        self.gen = gen
        self.gen_frac = gen / (gen + tool)
        self.v_start = version
        self.remaining = gen + tool  # 剩余墙钟推进量（不含重算）
        self.gen_done = 0.0  # 已完成的生成秒数
        self.pending_rc = 0.0  # 下次开跑前要先做的重算秒数
        self.segments: list[tuple[int, float]] = []  # (生成该片段的策略版本, 生成秒数)
        self.budget_left = math.inf


class _Stats:
    def __init__(self) -> None:
        self.gen_busy = 0.0  # 槽位在生成的秒数
        self.rc_busy = 0.0  # 槽位在重算 KV 的秒数
        self.occupied = 0.0  # 槽位被轨迹占用（生成 + 工具 + 重算）的秒数
        self.wasted = 0.0  # 被中止轨迹已生成的秒数
        self.gate_idle = 0.0  # 槽位因限流空闲的秒数
        self.dropped = 0
        self.started_gen: list[float] = []
        self.trained: list[_Traj] = []
        self.train_versions: list[int] = []
        self.interrupts = 0
        self.max_gate_lead = 0  # 提交时 ⌊(N_r-1)/B⌋ - 当前版本 的最大值（AReaL 限流保证它 ≤ η）


class _Sim:
    """共用部件：采样、重算成本、片段记账。"""

    def __init__(self, cfg: RolloutConfig, seed: int):
        if cfg.tail_alpha <= 1.0:
            raise ValueError("tail_alpha 必须 > 1")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cap = 50.0 * cfg.mean_gen_s
        self.x_m = pareto_scale_for_mean(float(cfg.mean_gen_s), float(cfg.tail_alpha), self.cap)
        self.ids = itertools.count()
        self.st = _Stats()
        self.model = MODELS[cfg.model]
        self.gpu_speed = peak_tflops(GPUS[cfg.gpu]) * 1e12 * cfg.mfu  # 每卡有效 FLOP/s

    def new(self, version: int) -> _Traj:
        c = self.cfg
        gen = min(self.x_m * (1.0 + self.rng.pareto(c.tail_alpha)), self.cap)
        tool = self.rng.exponential(c.mean_tool_s) if c.mean_tool_s > 0 else 0.0
        tr = _Traj(next(self.ids), gen, tool, version)
        self.st.started_gen.append(gen)
        return tr

    def _ctx_flops(self, tr: _Traj) -> float:
        ctx = int(self.cfg.prompt_tokens + tr.gen_done * self.cfg.decode_tok_s)
        return prefill_flops(self.model, ctx)

    def recompute_s(self, tr: _Traj) -> float:
        """单条续写：按一个推理引擎的速度重算上下文。"""
        if self.cfg.resume_kv == "retain":
            return 0.0
        return self._ctx_flops(tr) / (self.cfg.engine_gpus * self.gpu_speed)

    def pooled_recompute_s(self, trajs: list[_Traj], n_slots: int) -> float:
        """批量续写：整个 rollout 集群一起重算，所有轨迹等待同样的时长。"""
        if self.cfg.resume_kv == "retain" or not trajs:
            return 0.0
        return sum(self._ctx_flops(tr) for tr in trajs) / (n_slots * self.cfg.gpus_per_worker * self.gpu_speed)

    def advance(self, tr: _Traj, seg_start: float, version: int, until: float) -> None:
        """记账：轨迹从 seg_start 起（先重算 pending_rc，再推进）跑到 until。"""
        elapsed = max(0.0, until - seg_start)
        rc = min(elapsed, tr.pending_rc)
        prog = min(elapsed - rc, tr.remaining)
        tr.pending_rc -= rc
        tr.remaining -= prog
        g = prog * tr.gen_frac
        tr.gen_done += g
        if g > 0:
            tr.segments.append((version, g))
        tr.budget_left -= prog
        self.st.rc_busy += rc
        self.st.gen_busy += g
        self.st.occupied += rc + prog

    @staticmethod
    def seg_end(tr: _Traj, seg_start: float) -> float:
        return seg_start + tr.pending_rc + min(tr.remaining, tr.budget_left)

    def train_on(self, batch: list[_Traj], version: int) -> None:
        self.st.trained.extend(batch)
        self.st.train_versions.extend([version] * len(batch))


def _slots_and_train(cfg: RolloutConfig) -> tuple[int, float]:
    if cfg.deploy == "colocated":
        return cfg.n_workers, cfg.train_step_s
    if not 0.0 < cfg.trainer_frac < 1.0:
        raise ValueError("trainer_frac 必须在 (0, 1)")
    return max(1, round(cfg.n_workers * (1.0 - cfg.trainer_frac))), cfg.train_step_s / cfg.trainer_frac


# ---------------------------------------------------------------------------
# sync / async：同一个事件驱动模拟（sync = η=0）
# ---------------------------------------------------------------------------


def _run_streaming(cfg: RolloutConfig, n_steps: int, sim: _Sim, eta: int) -> float:
    B, st = cfg.batch_size, sim.st
    R, T = _slots_and_train(cfg)
    colocated = cfg.deploy == "colocated"
    abort = cfg.stale_policy == "abort"
    version, admitted, steps = 0, 0, 0
    now = 0.0
    idle = list(range(R))
    running: dict[int, tuple[_Traj, float, int, int]] = {}  # slot -> (轨迹, 段开始, 段版本, 事件 token)
    buffer: list[_Traj] = []
    events: list = []  # (时刻, 类型 0=训练结束 1=轨迹段结束, 序号, 数据)
    seq = itertools.count()
    training = False
    batch: list[_Traj] = []

    def schedule(slot: int, tr: _Traj, start: float, v: int) -> None:
        tok = next(seq)
        running[slot] = (tr, start, v, tok)
        heapq.heappush(events, (sim.seg_end(tr, start), 1, tok, slot))

    def start_idle() -> None:
        nonlocal admitted
        if training and colocated:
            return
        while idle and admitted < (version + eta + 1) * B:  # AReaL 限流：⌊(N_r-1)/B⌋ ≤ i + η
            slot = idle.pop()
            admitted += 1
            st.max_gate_lead = max(st.max_gate_lead, (admitted - 1) // B - version)
            schedule(slot, sim.new(version), now, version)

    def try_train() -> None:
        nonlocal training, batch, buffer, admitted
        if training:
            return
        if abort:
            fresh = [tr for tr in buffer if version - tr.v_start <= eta]
            st.dropped += len(buffer) - len(fresh)
            admitted -= len(buffer) - len(fresh)  # 丢弃的样本不再占限流名额
            buffer = fresh
        if len(buffer) < B:
            return
        buffer.sort(key=lambda tr: tr.tid)  # 优先用旧轨迹
        batch, buffer = buffer[:B], buffer[B:]
        training = True
        if colocated:  # 共置：训练期间暂停所有在途生成
            for slot, (tr, s0, v, _) in list(running.items()):
                sim.advance(tr, s0, v, now)
                running[slot] = (tr, math.nan, v, -1)
        heapq.heappush(events, (now + T, 0, next(seq), None))

    def on_version_bump() -> None:
        nonlocal admitted
        resumed = []
        for slot, (tr, s0, v, _) in list(running.items()):
            if not math.isnan(s0):
                sim.advance(tr, s0, v, now)
            if abort:
                if version - tr.v_start > eta:  # 过旧：中止，已生成的算浪费，释放名额
                    st.wasted += tr.gen_done
                    admitted -= 1
                    del running[slot]
                    idle.append(slot)
                    continue
                if colocated:
                    schedule(slot, tr, now, v)  # 训练结束，接着用旧片段版本跑（abort 模式不做在途更新）
                else:
                    running[slot] = (tr, now, v, running[slot][3])  # 已记账到 now，从 now 继续，事件不变
                    heapq.heappush(events, (sim.seg_end(tr, now), 1, running[slot][3], slot))
                continue
            st.interrupts += 1  # interrupt：丢旧 KV，用新权重重算后继续
            resumed.append((slot, tr))
        delay = sim.pooled_recompute_s([tr for _, tr in resumed], R)
        for slot, tr in resumed:
            tr.pending_rc = delay
            schedule(slot, tr, now, version)

    last = 0.0
    start_idle()
    while steps < n_steps:
        now, kind, tok, data = heapq.heappop(events)
        if not (training and colocated):
            st.gate_idle += len(idle) * (now - last)
        last = now
        if kind == 1:
            slot = data
            if slot not in running or running[slot][3] != tok:
                continue  # 过期事件
            tr, s0, v, _ = running.pop(slot)
            sim.advance(tr, s0, v, now)
            buffer.append(tr)
            idle.append(slot)
        else:
            training = False
            sim.train_on(batch, version)
            steps += 1
            version += 1
            if steps == n_steps:
                break
            on_version_bump()
        try_train()
        start_idle()
    for tr, s0, v, _ in running.values():  # 截止到最后一次训练结束
        if not math.isnan(s0):
            sim.advance(tr, s0, v, now)
    return now


# ---------------------------------------------------------------------------
# partial：凑满 batch 即停，在途的下一轮续写
# ---------------------------------------------------------------------------


def _run_partial(cfg: RolloutConfig, n_steps: int, sim: _Sim) -> float:
    B = cfg.batch_size
    R, T = _slots_and_train(cfg)
    t = 0.0
    carry: list[_Traj] = []  # 上一轮暂停、本轮优先续写的轨迹
    budget = cfg.partial_budget_s
    for version in range(n_steps):
        if cfg.partial_budget_tokens is not None:
            per_gen_s = cfg.partial_budget_tokens / cfg.decode_tok_s
        pending = deque(sorted(carry, key=lambda tr: tr.tid))
        carry, paused_now, buffer = [], deque(), []
        new_quota = int(B * cfg.over_provision) + R
        running: list = []  # 堆：(段结束, tid, 段开始, traj)

        def take(at: float):
            nonlocal new_quota
            if pending:
                tr = pending.popleft()
            elif new_quota > 0:
                new_quota -= 1
                tr = sim.new(version)
            elif paused_now:
                tr = paused_now.popleft()
            else:
                return
            tr.budget_left = budget if cfg.partial_budget_tokens is None else per_gen_s / tr.gen_frac
            heapq.heappush(running, (sim.seg_end(tr, at), tr.tid, at, tr))

        head = list(pending)[:R]  # 本轮开头成批续写：集群一起重算，大家等同样久
        batch_rc = sim.pooled_recompute_s(head, R)
        for tr in head:
            tr.pending_rc = batch_rc
        for _ in range(R):
            take(t)
        phase_end = t
        while len(buffer) < B and running:
            end, _, s0, tr = heapq.heappop(running)
            sim.advance(tr, s0, version, end)
            if tr.remaining <= 1e-12:
                buffer.append(tr)
                phase_end = end
            else:  # 本轮预算用完：暂停；本轮新名额用完后才会在本轮内续写
                tr.pending_rc = sim.recompute_s(tr)
                paused_now.append(tr)
            take(end)
        for _, _, s0, tr in running:  # 凑满 batch：暂停所有在途轨迹（APRIL）
            sim.advance(tr, s0, version, phase_end)
            carry.append(tr)
        carry.extend(pending)
        carry.extend(paused_now)
        for tr in carry:  # 下一轮权重已更新：续写要重算
            tr.pending_rc = sim.recompute_s(tr)
        buffer.sort(key=lambda tr: tr.tid)
        sim.train_on(buffer[:B], version)
        t = phase_end + T  # 生成与训练不重叠
    return t


# ---------------------------------------------------------------------------
# 入口与汇总
# ---------------------------------------------------------------------------


def simulate_rollout(cfg: RolloutConfig, n_steps: int = 50, seed: int = 0) -> dict:
    """模拟 n_steps 个训练步，返回
    {"worker_busy_frac","traj_per_hour","mean_staleness","p99_staleness","step_time_s"}，另附：
    - train_tokens_per_hour：每小时训练用掉的生成 token（与轨迹长度无关的主指标）
    - staleness_max；token_staleness_mean / token_staleness_max（按 token 加权、按片段逐段统计）
    - gen_mean_started / gen_mean_trained / length_ratio（被训练轨迹与全部启动轨迹的平均生成时长之比，<1 表示偏向短轨迹）
    - cluster_util（(生成 + 重算 + 训练) GPU·秒 / 总 GPU·秒）、gate_idle_frac（槽位因限流空闲的比例）、
      recompute_frac（重算占槽位忙碌时间的比例）、interrupts、max_gate_lead（限流量：提交时 ⌊(N_r-1)/B⌋ - 版本 的最大值）
    - total_time_s、trained_trajectories、mean_batch、wasted_frac、dropped_trajectories、worker_occupied_frac、n_slots
    worker_busy_frac = 生成秒数 / (rollout 槽位数 × 总时长)，只计"在生成"（GPU 在算）的时间。
    """
    if cfg.mode not in _MODES:
        raise ValueError(f"mode 必须是 {_MODES} 之一，收到 {cfg.mode!r}")
    if cfg.deploy not in ("colocated", "disaggregated"):
        raise ValueError("deploy 必须是 'colocated' 或 'disaggregated'")
    if cfg.stale_policy not in ("interrupt", "abort"):
        raise ValueError("stale_policy 必须是 'interrupt' 或 'abort'")
    if cfg.resume_kv not in ("recompute", "retain"):
        raise ValueError("resume_kv 必须是 'recompute' 或 'retain'")
    if n_steps <= 0 or cfg.n_workers <= 0 or cfg.batch_size <= 0 or cfg.max_staleness < 0:
        raise ValueError("n_steps / n_workers / batch_size 必须 >= 1，max_staleness 必须 >= 0")
    sim = _Sim(cfg, seed)
    if cfg.mode == "partial":
        total = _run_partial(cfg, n_steps, sim)
    else:
        total = _run_streaming(cfg, n_steps, sim, 0 if cfg.mode == "sync" else cfg.max_staleness)
    return _result(cfg, n_steps, total, sim)


def _result(cfg: RolloutConfig, n_steps: int, total: float, sim: _Sim) -> dict:
    st = sim.st
    R, _ = _slots_and_train(cfg)
    trained = st.trained
    stale = np.array([v - tr.v_start for tr, v in zip(trained, st.train_versions)], dtype=float)
    tok_w = tok_s = 0.0
    tok_max = 0
    for tr, v in zip(trained, st.train_versions):
        for seg_v, g in tr.segments:
            tok_w += g * (v - seg_v)
            tok_s += g
            tok_max = max(tok_max, v - seg_v)
    gen_trained = np.array([tr.gen for tr in trained]) if trained else np.zeros(1)
    gen_started = np.mean(st.started_gen) if st.started_gen else 0.0
    train_gpu_s = n_steps * cfg.n_workers * cfg.train_step_s
    return {
        "worker_busy_frac": st.gen_busy / (R * total),
        "traj_per_hour": len(trained) / total * 3600.0,
        "mean_staleness": float(stale.mean()) if len(stale) else 0.0,
        "p99_staleness": float(np.percentile(stale, 99)) if len(stale) else 0.0,
        "step_time_s": total / n_steps,
        "staleness_max": int(stale.max()) if len(stale) else 0,
        "token_staleness_mean": tok_w / tok_s if tok_s else 0.0,
        "token_staleness_max": tok_max,
        "train_tokens_per_hour": float(gen_trained.sum()) * cfg.decode_tok_s / total * 3600.0 if trained else 0.0,
        "gen_mean_started": float(gen_started),
        "gen_mean_trained": float(gen_trained.mean()),
        "length_ratio": float(gen_trained.mean() / gen_started) if gen_started else 0.0,
        "cluster_util": (st.gen_busy + st.rc_busy + train_gpu_s) / (cfg.n_workers * total),
        "gate_idle_frac": st.gate_idle / (R * total),
        "recompute_frac": st.rc_busy / (st.rc_busy + st.gen_busy) if st.gen_busy else 0.0,
        "interrupts": st.interrupts,
        "max_gate_lead": st.max_gate_lead,
        "total_time_s": total,
        "trained_trajectories": len(trained),
        "mean_batch": len(trained) / n_steps,
        "wasted_frac": st.wasted / st.gen_busy if st.gen_busy > 0 else 0.0,
        "dropped_trajectories": st.dropped,
        "worker_occupied_frac": st.occupied / (R * total),
        "n_slots": R,
    }


def mean_std(results: list[dict]) -> dict:
    """把多次运行（多个 seed）的结果汇总成 {key: (均值, 标准差)}，只处理数值字段。"""
    out = {}
    for k, v in results[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = np.array([r[k] for r in results], dtype=float)
            out[k] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out


def simulate_rollout_seeds(cfg: RolloutConfig, n_steps: int = 50, seeds=(0, 1, 2, 3, 4)) -> dict:
    """多 seed 运行 simulate_rollout，返回 {"mean": {...}, "std": {...}, "runs": [...]}，方便 lab 报 均值±标准差。"""
    runs = [simulate_rollout(cfg, n_steps, s) for s in seeds]
    ms = mean_std(runs)
    return {"mean": {k: m for k, (m, _) in ms.items()}, "std": {k: s for k, (_, s) in ms.items()}, "runs": runs}


def compare_rollout_modes(cfg: RolloutConfig | None = None, n_steps: int = 50, seed: int = 0) -> dict:
    """同样的 GPU 总数下把收益拆成两步：
    tail_gain = async 共置 / sync 共置（不等长尾，训练仍占用全部 GPU）；
    overlap_gain = async 分离 / async 共置（训练与生成重叠）。
    吞吐按 train_tokens_per_hour 计（与轨迹长度无关）。另给出 partial 共置作对照。
    """
    cfg = cfg or RolloutConfig()
    runs = {
        "sync_colocated": simulate_rollout(replace(cfg, mode="sync", deploy="colocated"), n_steps, seed),
        "partial_colocated": simulate_rollout(replace(cfg, mode="partial", deploy="colocated"), n_steps, seed),
        "async_colocated": simulate_rollout(replace(cfg, mode="async", deploy="colocated"), n_steps, seed),
        "async_disaggregated": simulate_rollout(replace(cfg, mode="async", deploy="disaggregated"), n_steps, seed),
    }
    tph = {k: v["train_tokens_per_hour"] for k, v in runs.items()}
    return {
        "runs": runs,
        "tail_gain": tph["async_colocated"] / tph["sync_colocated"],
        "overlap_gain": tph["async_disaggregated"] / tph["async_colocated"],
        "total_gain": tph["async_disaggregated"] / tph["sync_colocated"],
    }
