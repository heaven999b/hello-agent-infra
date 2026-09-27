"""模块 06 实验：长尾让同步 rollout 空转多少？partial / async 在**相同 GPU 总数**下能拿回多少？代价是多少 staleness？

运行：
    .venv/bin/python modules/06_long_tail_async/lab.py
    # 或
    make lab N=06

六张表（主指标是「每小时训练的 token 数」，与轨迹长短无关；staleness 按 token 加权）：
  [1] 尾部越重：sync / partial / async 三种模式（共置、等资源、10 个 seed 的均值与区间）
  [2] η 的作用：默认 64 个单位的共置 vs 256 个单位的分离部署，吞吐–陈旧度曲线
  [3] 分离部署的配比：训练占比 1/8 … 1/2，对照共置
  [4] 中止并丢弃（abort）vs 可中断续写（interrupt）：丢弃带来的长度偏置有多大
  [5] partial 的每轮推进预算
  [6] 把 README 第 3.2 节的闭式解和模拟器对上
"""

from __future__ import annotations

import math

import numpy as np

from codesign_sim.rollout import RolloutConfig, simulate_rollout

N_STEPS = 20  # 每个配置模拟的训练步数（多 seed 平均；控制在 10 秒内跑完）
SEEDS_MAIN = tuple(range(10))  # 表 [1] 用 10 个 seed
SEEDS_SIDE = (0, 1, 2)  # 其余表用 3 个 seed 的均值


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """等宽打印一张表：第一列左对齐，其余右对齐。"""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        first = cells[0].ljust(widths[0])
        rest = [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join([first, *rest])

    print(f"\n{title}")
    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def runs(seeds: tuple[int, ...], **overrides: float | int | str) -> list[dict]:
    """默认配置（64 个单位、batch 256、生成均值 20 s、工具均值 5 s、训练 30 s、共置）上改几个参数，跑多个 seed。"""
    cfg = RolloutConfig(**overrides)
    return [simulate_rollout(cfg, n_steps=N_STEPS, seed=s) for s in seeds]


def mean(rs: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rs]))


def mtok(rs: list[dict]) -> float:
    """每小时训练的 token 数（百万），多个 seed 的均值。"""
    return mean(rs, "train_tokens_per_hour") / 1e6


# ---------------------------------------------------------------------------
# [1] 尾部越重：三种模式（等资源）
# ---------------------------------------------------------------------------


def table_modes_vs_tail() -> None:
    rows = []
    for alpha in (3.0, 1.5, 1.2):
        sync = runs(SEEDS_MAIN, tail_alpha=alpha, mode="sync")
        for mode in ("sync", "partial", "async"):
            rs = sync if mode == "sync" else runs(SEEDS_MAIN, tail_alpha=alpha, mode=mode)
            # 每个 seed 各自算相对 sync 的倍数，再报均值与 [最小, 最大]
            ratios = [r["train_tokens_per_hour"] / s["train_tokens_per_hour"] for r, s in zip(rs, sync)]
            rows.append([
                f"alpha={alpha:<3} {mode}",
                f"{mtok(rs):.1f}",
                f"{np.mean(ratios):.1f}x [{min(ratios):.1f}-{max(ratios):.1f}]",
                f"{mean(rs, 'worker_busy_frac'):.0%}",
                f"{mean(rs, 'token_staleness_mean'):.2f}",
                f"{max(r['staleness_max'] for r in rs)}",
                f"{mean(rs, 'length_ratio'):.2f}",
            ])
    print_table(
        "[1] 尾部越重，同步越吃亏（共置、GPU 总数相同；partial 每轮预算 60 s，async η = 1；10 个 seed）",
        ["配置", "M token/h", "相对 sync [区间]", "生成忙碌", "token staleness", "最大 staleness", "长度比"],
        rows,
    )


# ---------------------------------------------------------------------------
# [2] η 的作用
# ---------------------------------------------------------------------------


def table_eta_curve() -> None:
    rows = []
    for eta in (0, 1, 2, 4):
        a = runs(SEEDS_SIDE, mode="async", max_staleness=eta)
        b = runs(SEEDS_SIDE, mode="async", max_staleness=eta, n_workers=256, deploy="disaggregated")
        rows.append([
            f"eta={eta}",
            f"{mtok(a):.1f}",
            f"{mean(a, 'token_staleness_mean'):.2f} / {max(r['staleness_max'] for r in a)}",
            f"{mtok(b):.1f}",
            f"{mean(b, 'token_staleness_mean'):.2f} / {max(r['staleness_max'] for r in b)}",
            f"{mean(b, 'gate_idle_frac'):.0%}",
        ])
    print_table(
        "[2] async 的 η（tail_alpha = 1.5，3 个 seed）：左 = 默认共置 64 单位；右 = 分离部署 256 单位（槽位 192，训练占 1/4）",
        ["配置", "共置 M token/h", "staleness 均值/最大", "分离 M token/h", "staleness 均值/最大", "槽位被限流空闲"],
        rows,
    )


# ---------------------------------------------------------------------------
# [3] 分离部署的配比
# ---------------------------------------------------------------------------


def table_trainer_split() -> None:
    colo = runs(SEEDS_SIDE, mode="async", max_staleness=2)
    rows = [["共置（参照）", f"{mtok(colo):.1f}", f"{mean(colo, 'token_staleness_mean'):.2f}", f"{mean(colo, 'gate_idle_frac'):.0%}"]]
    for frac in (0.125, 0.25, 0.375, 0.5):
        rs = runs(SEEDS_SIDE, mode="async", max_staleness=2, deploy="disaggregated", trainer_frac=frac)
        rows.append([
            f"分离 训练占 {frac:g}",
            f"{mtok(rs):.1f}",
            f"{mean(rs, 'token_staleness_mean'):.2f}",
            f"{mean(rs, 'gate_idle_frac'):.0%}",
        ])
    print_table(
        "[3] 分离部署的配比（async，η = 2，tail_alpha = 1.5，64 单位，3 个 seed）",
        ["配置", "M token/h", "token staleness", "槽位被限流空闲"],
        rows,
    )


# ---------------------------------------------------------------------------
# [4] 中止并丢弃 vs 可中断续写
# ---------------------------------------------------------------------------


def table_abort_vs_interrupt() -> None:
    rows = []
    for alpha in (1.5, 1.2):
        for policy in ("interrupt", "abort"):
            rs = runs(SEEDS_SIDE, mode="async", tail_alpha=alpha, stale_policy=policy)
            rows.append([
                f"alpha={alpha:<3} {policy}",
                f"{mean(rs, 'traj_per_hour'):,.0f}",
                f"{mtok(rs):.1f}",
                f"{mean(rs, 'length_ratio'):.2f}",
                f"{mean(rs, 'wasted_frac'):.1%}",
                f"{mean(rs, 'recompute_frac'):.1%}",
                f"{max(r['staleness_max'] for r in rs)}",
            ])
    print_table(
        "[4] async（共置，η = 1，3 个 seed）：AReaL 式中断续写 vs 超过 η 就中止并丢弃",
        ["配置", "轨迹/小时", "M token/h", "长度比", "被中止浪费", "重算占比", "最大 staleness"],
        rows,
    )


# ---------------------------------------------------------------------------
# [5] partial 的每轮推进预算
# ---------------------------------------------------------------------------


def table_partial_budget() -> None:
    rows = []
    for budget in (15.0, 30.0, 60.0, 120.0):
        rs = runs(SEEDS_SIDE, mode="partial", partial_budget_s=budget)
        rows.append([
            f"partial budget={budget:.0f}s",
            f"{mtok(rs):.1f}",
            f"{mean(rs, 'token_staleness_mean'):.2f}",
            f"{max(r['staleness_max'] for r in rs)}",
            f"{mean(rs, 'recompute_frac'):.1%}",
            f"{mean(rs, 'length_ratio'):.2f}",
        ])
    print_table(
        "[5] partial：每条轨迹每轮最多推进多久（tail_alpha = 1.5，共置，3 个 seed）",
        ["配置", "M token/h", "token staleness", "最大 staleness", "重算占比", "长度比"],
        rows,
    )


# ---------------------------------------------------------------------------
# [6] 闭式解 vs 模拟器
# ---------------------------------------------------------------------------


def pareto_busy_closed_form(alpha: float, n: int) -> float:
    """README 第 3.2 节：E[T] / E[max_n T]，T ~ Pareto(alpha)，不截断。"""
    log_emax = math.lgamma(n + 1) + math.lgamma(1 - 1 / alpha) - math.lgamma(n + 1 - 1 / alpha)
    return alpha / (alpha - 1) / math.exp(log_emax)


def pareto_busy_capped(alpha: float, n: int, cap_over_scale: float) -> float:
    """同上，但 T 截断在 cap_over_scale × x_m 处（比例与尺度无关）。E[min(X, L)] = ∫_0^L P(X > t) dt，数值积分。"""
    x_m, L = 1.0, cap_over_scale
    t = np.geomspace(x_m, L, 200_000)
    tail_one = (x_m / t) ** alpha  # P(T > t)
    tail_max = 1.0 - (1.0 - tail_one) ** n  # P(max > t)
    e_min = x_m + np.trapezoid(tail_one, t)
    e_max = x_m + np.trapezoid(tail_max, t)
    return float(e_min / e_max)


def cap_over_scale(alpha: float, cap_over_mean: float) -> float:
    """模拟器把生成时间截断在 cap_over_mean × 均值，且截断后的均值等于名义均值。
    求截断点与 Pareto 下限 x_m 之比 c：c = cap_over_mean · E[min(X, c)]，x_m = 1 时
    E[min(X, c)] = 1 + (1 - c^(1-α)) / (α - 1)。不动点迭代几次就收敛。
    """
    c = cap_over_mean * alpha / (alpha - 1)
    for _ in range(100):
        c = cap_over_mean * (1.0 + (1.0 - c ** (1.0 - alpha)) / (alpha - 1.0))
    return c


def table_closed_form_check() -> None:
    rows = []
    n = 64
    for alpha in (3.0, 2.0, 1.5):
        # 一条轨迹一个槽（n_workers = batch_size）、无工具、无训练：只剩「等最慢的那条」
        cfg = RolloutConfig(n_workers=n, batch_size=n, tail_alpha=alpha, mean_tool_s=0.0, train_step_s=0.0, mode="sync")
        r = simulate_rollout(cfg, n_steps=200, seed=0)
        cap = cap_over_scale(alpha, 50.0)  # 模拟器截断在 50 × 均值（均值按截断后校准）
        rows.append([
            f"alpha={alpha}, n={n}",
            f"{pareto_busy_closed_form(alpha, n):.1%}",
            f"{pareto_busy_capped(alpha, n, cap):.1%}",
            f"{r['worker_busy_frac']:.1%}",
        ])
    print_table(
        "[6] 同步忙碌比例：闭式解（不截断）/ 截断在 50× 均值处的数值解 / 模拟器（200 步）",
        ["配置", "闭式解", "截断数值解", "simulate_rollout"],
        rows,
    )


def main() -> None:
    table_modes_vs_tail()
    table_eta_curve()
    table_trainer_split()
    table_abort_vs_interrupt()
    table_partial_budget()
    table_closed_form_check()


if __name__ == "__main__":
    main()
