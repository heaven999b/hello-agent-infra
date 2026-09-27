"""模块 01 实验：用 codesign_sim 的成本模型，把 README 里的心算全部打印成表。

运行：
    .venv/bin/python modules/01_inference_basics/lab.py
    # 或
    make lab N=01

6 组、共 8 张表（[5] 和 [6] 各两张）：
  [1] GPU roofline：峰值、带宽、ridge point，以及不同 intensity 下能拿到多少算力
  [2] KV cache 每 token 大小与一条长序列的 KV 体积（GQA vs MLA）
  [3] 显存预算（理想上界）：装下权重后能放多少 token 的 KV、最多多少条序列
  [4] decode 的 arithmetic intensity 随 batch 怎么变（以及它的上限）
  [5] decode 步时间：batch 变大时 TPOT 与吞吐怎么变（两种部署各一张）
  [6] [6a] TTFT：prompt 长度 × 前缀命中比例；[6b] Little's law 下工具等待放大的 KV 占用

所有时间都来自 roofline 模型 max(计算, 访存)，不含 TP all-reduce / EP all-to-all 等通信（README 第 5 节）。
"""

from __future__ import annotations

import math

from codesign_sim.hardware import GPUS, GPU, attainable_tflops, ridge_point
from codesign_sim.model import (
    MODELS,
    ModelConfig,
    decode_bytes,
    decode_flops,
    kv_bytes_per_token,
    prefill_flops,
    step_time_s,
)

MEM_FRAC = 0.9  # 显存里留 10% 给激活、临时缓冲和碎片
GB = 1e9
GIB = 2**30


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


def kv_budget_bytes(gpu: GPU, cfg: ModelConfig, n_gpus: int, weight_bytes: float) -> float:
    """装下全部权重后剩给 KV 的字节数（可能为负：权重都放不下）。"""
    return n_gpus * gpu.hbm_gb * GB * MEM_FRAC - cfg.total_params_b * 1e9 * weight_bytes


def intensity(cfg: ModelConfig, batch: int, ctx: int, weight_bytes: float = 2) -> float:
    return decode_flops(cfg, batch, ctx) / decode_bytes(cfg, batch, ctx, weight_bytes)


# ---------------------------------------------------------------------------
# [1] GPU roofline
# ---------------------------------------------------------------------------


def table_roofline() -> None:
    probes = [1, 8, 64, 242]  # decode batch=1 / GQA-8 注意力 / batch=64 / MLA 注意力
    rows = []
    for g in GPUS.values():
        fp8 = f"{g.fp8_tflops:.0f}" if g.fp8_tflops else "-"
        attain = [f"{attainable_tflops(g, i):.0f}" for i in probes]
        rows.append([g.name, f"{g.bf16_tflops:.0f}", fp8, f"{g.hbm_gb:.0f}", f"{g.hbm_tbps:.2f}",
                     f"{ridge_point(g):.0f}", *attain])
    headers = ["GPU", "BF16 TF", "FP8 TF", "HBM GB", "TB/s", "ridge", *[f"@I={i}" for i in probes]]
    print_table("[1] roofline（dense BF16；@I=x 表示 intensity 为 x 时能拿到的 TFLOPS）", headers, rows)


# ---------------------------------------------------------------------------
# [2] KV cache 大小
# ---------------------------------------------------------------------------


MODEL_KEYS = ["llama3-8b", "llama3-70b", "qwen3-32b", "deepseek-v3"]


def table_kv() -> None:
    base = kv_bytes_per_token(MODELS["llama3-70b"])
    rows = []
    for key in MODEL_KEYS:
        m = MODELS[key]
        kv = kv_bytes_per_token(m)
        rows.append([key, m.attn, f"{kv:,}", f"{kv / 1024:.1f}",
                     f"{32768 * kv / GIB:.2f}", f"{131072 * kv / GIB:.1f}", f"{kv / base:.2f}x"])
    headers = ["model", "attn", "B/token", "KiB/token", "32k seq GiB", "128k seq GiB", "vs 70B"]
    print_table("[2] KV cache（BF16）", headers, rows)


# ---------------------------------------------------------------------------
# [3] 显存预算与最大并发
# ---------------------------------------------------------------------------

DEPLOYMENTS: list[tuple[str, str, int, float]] = [
    # (模型, GPU, 卡数, 每参数字节)
    ("llama3-8b", "H100-SXM", 1, 2),
    ("qwen3-32b", "H100-SXM", 2, 2),
    ("llama3-70b", "H100-SXM", 8, 2),
    ("deepseek-v3", "H800-SXM", 8, 1),
    ("deepseek-v3", "H800-SXM", 32, 1),
]


def table_capacity() -> None:
    rows = []
    for name, gname, n, wb in DEPLOYMENTS:
        cfg, gpu = MODELS[name], GPUS[gname]
        weights = cfg.total_params_b * 1e9 * wb
        budget = kv_budget_bytes(gpu, cfg, n, wb)
        kv = kv_bytes_per_token(cfg)
        if budget <= 0:
            rows.append([name, f"{n}x{gname}", f"{wb:g}", f"{weights / GB:.0f}", "放不下", "-", "-", "-"])
            continue
        tokens = budget / kv
        rows.append([name, f"{n}x{gname}", f"{wb:g}", f"{weights / GB:.0f}", f"{budget / GB:.0f}",
                     f"{tokens / 1e6:.2f}M", f"{int(tokens // 8192)}", f"{int(tokens // 32768)}"])
    headers = ["model", "deploy", "W bytes", "weights GB", "KV GB", "KV tokens", "seqs@8k", "seqs@32k"]
    print_table("[3] 显存预算的理想上界（HBM × 0.9 − 权重；权重在所有卡上只存一份、KV 均匀切分；KV 为 BF16）",
                headers, rows)


# ---------------------------------------------------------------------------
# [4] decode intensity 随 batch 的变化
# ---------------------------------------------------------------------------


def intensity_limit(cfg: ModelConfig, ctx: int) -> float:
    """B → ∞ 时的上限：每条序列的 FLOPs ÷ 每条序列的 KV 字节（权重被摊薄到可以忽略）。"""
    return decode_flops(cfg, 1, ctx) / (ctx * kv_bytes_per_token(cfg))


def table_intensity() -> None:
    batches = [1, 8, 64, 256, 1024, 4096]
    cases = [("llama3-8b", 128), ("llama3-8b", 4096), ("llama3-70b", 4096),
             ("llama3-70b", 32768), ("deepseek-v3", 4096), ("deepseek-v3", 32768)]
    rows = []
    for name, ctx in cases:
        cfg = MODELS[name]
        cells = [f"{intensity(cfg, b, ctx):.1f}" for b in batches]
        rows.append([f"{name} ctx={ctx}", *cells, f"{intensity_limit(cfg, ctx):.1f}"])
    headers = ["case", *[f"B={b}" for b in batches], "B→∞"]
    ridge = ridge_point(GPUS["H100-SXM"])
    print_table(f"[4] decode arithmetic intensity（FLOPs/Byte，BF16 权重与 KV；H100 ridge = {ridge:.0f}）",
                headers, rows)


# ---------------------------------------------------------------------------
# [5] decode 步时间：TPOT 与吞吐
# ---------------------------------------------------------------------------


def table_decode_steps(name: str, gname: str, n_gpus: int, ctx: int, batches: list[int]) -> None:
    cfg, gpu = MODELS[name], GPUS[gname]
    max_seqs = int(kv_budget_bytes(gpu, cfg, n_gpus, 2) // (ctx * kv_bytes_per_token(cfg)))
    peak_s = n_gpus * gpu.bf16_tflops * 1e12 * 0.6
    bw_s = n_gpus * gpu.hbm_tbps * 1e12 * 0.8
    rows = []
    for b in batches:
        f, by = decode_flops(cfg, b, ctx), decode_bytes(cfg, b, ctx)
        t = step_time_s(gpu, cfg, f, by, n_gpus=n_gpus)
        bound = "compute" if f / peak_s >= by / bw_s else "memory"
        fits = "yes" if b <= max_seqs else "NO"
        rows.append([str(b), fits, f"{by / GB:.1f}", f"{f / by:.1f}", bound, f"{t * 1e3:.2f}", f"{b / t:,.0f}"])
    headers = ["batch", "fits", "GB/step", "I", "bound", "TPOT ms", "tok/s"]
    title = (f"[5] decode 步时间：{name} on {n_gpus}x{gname}, ctx={ctx}"
             f"（MFU 0.6 / MBU 0.8；显存最多 {max_seqs} 条序列）")
    print_table(title, headers, rows)


# ---------------------------------------------------------------------------
# [6] TTFT 与 Little's law
# ---------------------------------------------------------------------------


def ttft_s(gpu: GPU, cfg: ModelConfig, n_gpus: int, prompt: int, cached: int) -> float:
    """prefill 未命中部分的时间：FLOPs 按 prefill_flops，字节 = 权重 + 读写整段 KV。"""
    n_new = max(prompt - cached, 1)
    flops = prefill_flops(cfg, n_new, prompt - n_new)
    bytes_ = cfg.total_params_b * 1e9 * 2 + prompt * kv_bytes_per_token(cfg)
    return step_time_s(gpu, cfg, flops, bytes_, n_gpus=n_gpus)


def table_ttft() -> None:
    gpu = GPUS["H100-SXM"]
    hit_rates = [0.0, 0.5, 0.9, 0.99]
    rows = []
    for name, n_gpus in [("llama3-8b", 1), ("llama3-70b", 8)]:
        cfg = MODELS[name]
        for prompt in [4096, 16384, 65536]:
            cells = [f"{ttft_s(gpu, cfg, n_gpus, prompt, int(prompt * h)):.3f}" for h in hit_rates]
            rows.append([f"{name} {n_gpus}xH100", f"{prompt:,}", *cells])
    headers = ["deploy", "prompt", *[f"hit {int(h * 100)}% s" for h in hit_rates]]
    print_table("[6a] TTFT ≈ prefill 未命中部分（不含排队；MFU 0.6）", headers, rows)


def poisson_quantile(mean: float, q: float) -> int:
    """泊松分布的 q 分位数（逐项累加 pmf，mean 在几百以内足够快且确定）。"""
    cdf, k = 0.0, 0
    while True:
        cdf += math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
        if cdf >= q:
            return k
        k += 1


def table_littles_law() -> None:
    cfg, gpu = MODELS["llama3-70b"], GPUS["H100-SXM"]
    rate, gen_s, ctx = 2.0, 30.0, 32768
    per_node = kv_budget_bytes(gpu, cfg, 8, 2)
    per_session = ctx * kv_bytes_per_token(cfg)
    rows = []
    for tool_s in [0.0, 60.0, 300.0]:
        sessions = rate * (gen_s + tool_s)  # Little's law：L = λ·W（均值）
        p99 = poisson_quantile(sessions, 0.99)  # 泊松到达时，系统内会话数 ~ Poisson(L)
        kv = sessions * per_session
        rows.append([f"{tool_s:.0f}", f"{sessions:.0f}", f"{p99}", f"{kv / GB:,.0f}",
                     f"{p99 * per_session / GB:,.0f}", f"{kv / per_node:.1f}"])
    headers = ["tool_s", "sessions (mean)", "sessions p99", "KV GB mean", "KV GB p99", "8xH100 nodes (mean)"]
    print_table(f"[6b] Little's law：{rate:g} 会话/s（泊松到达），生成 {gen_s:.0f} s，32k 上下文，llama3-70b，KV 全程 pin 住",
                headers, rows)


def main() -> None:
    table_roofline()
    table_kv()
    table_capacity()
    table_intensity()
    table_decode_steps("llama3-8b", "H100-SXM", 1, 4096, [1, 8, 32, 64, 128])
    table_decode_steps("llama3-70b", "H100-SXM", 8, 32768, [1, 8, 32, 40, 64])
    table_ttft()
    table_littles_law()


if __name__ == "__main__":
    main()
