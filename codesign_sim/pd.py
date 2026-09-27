"""P/D 分离（prefill / decode disaggregation）的最小模拟，以及同等卡数下的共置对照。

- simulate_colocated：n 个普通实例，prefill 与 decode 混跑（continuous batching + chunked prefill），
  请求按"在途 token 最少"路由
- simulate_pd：n_prefill 个 prefill 实例 + n_decode 个 decode 实例。
  1. 请求先到 prefill 池，prefill 完成吐出首 token（TTFT 在这里确定）
  2. 整段 prompt 的 KV 通过 RDMA 传到 decode 实例：传输时间 = prompt · KV字节 / 链路带宽；
     layerwise=True 时按层流水（第 l 层算完就发），只有约 1/n_layers 的传输暴露在关键路径上
     （若传输总时长超过 prefill 本身的计算时间，超出部分也会暴露）
  3. decode 实例上这个请求只剩 decode：以"命中了整段 prompt"的形式提交，不再做 prefill；
     但这段 prompt KV 归它独占（own_cached_kv=True），按 prompt + 已生成 token 计入 decode 实例的 KV 容量
     （与共置对照同一口径）。容量不足时在 decode 侧排队；kv_reserve="incremental" 时按引擎的抢占模式处理
     （被抢占的请求在 decode 实例上整段重算）

取舍：P/D 分离让 decode 不再被长 prompt 打断（TPOT/ITL 稳定），代价是 KV 传输、两个池子的配比难调、
prefill 池忙时 TTFT 排队。MLA 每 token KV 小（DeepSeek-V3 约 69 KB），传输代价低，这是 P/D 分离的底气之一。
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .engine import Engine, EngineConfig, Request, summarize
from .hardware import IB_GBPS_PER_GPU, LINK_EFFICIENCY
from .model import kv_bytes_per_token, prefill_flops_split, step_time_s


def _least_tokens(engines: list[Engine]) -> int:
    loads = [e.inflight_tokens() for e in engines]
    return int(np.argmin(loads))


def _run(engines: list[Engine], events: list, handle) -> None:
    """通用事件循环：按时间顺序处理事件，并推进最早的引擎。"""
    while True:
        t_ev = events[0][0] if events else math.inf
        times = [e.next_step_time() for e in engines]
        idx = int(np.argmin(times))
        if t_ev == math.inf and times[idx] == math.inf:
            return
        if t_ev <= times[idx]:
            now, _, _, payload = heapq.heappop(events)
            handle(now, payload)
            continue
        for rec in engines[idx].step():
            handle_done = ("done", idx, rec)
            heapq.heappush(events, (rec["finish_s"], 0, id(rec), handle_done))


def _goodput(out: dict, ttft_slo_s: float | None, tpot_slo_s: float | None) -> None:
    """goodput：同时满足 TTFT / TPOT SLO 的请求数 / 秒（SLO 为 None 表示不设限）。"""
    ok = [r for r in out["per_request"]
          if (ttft_slo_s is None or r["ttft_s"] <= ttft_slo_s) and (tpot_slo_s is None or r["tpot_s"] <= tpot_slo_s)]
    out["slo_attainment"] = len(ok) / len(out["per_request"])
    out["goodput_req_s"] = len(ok) / out["makespan_s"]


def simulate_colocated(requests: list[Request], cfg: EngineConfig, n_instances: int = 1, *,
                       ttft_slo_s: float | None = None, tpot_slo_s: float | None = None) -> dict:
    """n_instances 个共置实例（prefill + decode 混跑）处理开环请求；返回与 simulate 相同的汇总字段，
    另附 goodput_req_s / slo_attainment（按 ttft_slo_s / tpot_slo_s）、kv_peak_frac、n_gpus_total。"""
    engines = [Engine(cfg) for _ in range(n_instances)]
    seq = itertools.count()
    events = [(r.arrival_s, 1, next(seq), ("arrive", r)) for r in requests]
    heapq.heapify(events)

    def handle(now: float, payload) -> None:
        if payload[0] == "arrive":
            engines[_least_tokens(engines)].submit(payload[1])

    _run(engines, events, handle)
    records = [r for e in engines for r in e.finished]
    busy = float(np.mean([e.busy_s for e in engines]))
    out = summarize(records, busy, [c for e in engines for c in e._itl_chunks])
    out["n_gpus_total"] = n_instances * cfg.n_gpus
    out["kv_peak_frac"] = max(e.kv_peak / e.kv_capacity for e in engines)
    _goodput(out, ttft_slo_s, tpot_slo_s)
    return out


def simulate_pd(
    requests: list[Request],
    prefill_cfg: EngineConfig,
    decode_cfg: EngineConfig | None = None,
    n_prefill: int = 1,
    n_decode: int = 1,
    *,
    kv_link_gbps: float | None = None,
    layerwise: bool = True,
    ttft_slo_s: float | None = None,
    tpot_slo_s: float | None = None,
) -> dict:
    """P/D 分离模拟。kv_link_gbps：prefill→decode 的聚合链路带宽（GB/s，单向），
    默认 = min(两侧卡数) × 每卡 IB 50 GB/s。返回 TTFT/TPOT/ITL 分位数、吞吐、KV 传输统计、两池忙碌比例，
    以及 goodput_req_s / slo_attainment、decode_kv_peak_frac（decode 侧 KV 峰值 / 容量）、decode_preemptions。
    """
    decode_cfg = decode_cfg or prefill_cfg
    pre = [Engine(prefill_cfg) for _ in range(n_prefill)]
    dec = [Engine(decode_cfg) for _ in range(n_decode)]
    model = prefill_cfg.model
    kvb = kv_bytes_per_token(model, prefill_cfg.kv_dtype_bytes)
    link = kv_link_gbps if kv_link_gbps is not None else min(prefill_cfg.n_gpus, decode_cfg.n_gpus) * IB_GBPS_PER_GPU
    link_bw = link * 1e9 * LINK_EFFICIENCY
    by_id = {r.req_id: r for r in requests}
    first: dict[int, float] = {}
    transfer: list[tuple[float, float]] = []  # (总传输时间, 暴露在关键路径上的时间)
    seq = itertools.count()
    events = [(r.arrival_s, 1, next(seq), ("arrive", r)) for r in requests]
    heapq.heapify(events)
    engines = pre + dec
    done_records: dict[int, dict] = {}

    def prefill_time(r: Request) -> float:
        gemm, attn = prefill_flops_split(model, r.prompt_tokens - r.cached_tokens, r.cached_tokens)
        c = prefill_cfg
        return step_time_s(c.gpu, model, gemm, 0.0, c.n_gpus, c.mfu, c.mbu, c.resolved_gemm_dtype(),
                           attn_flops=attn, attn_dtype=c.attn_dtype)

    def handle(now: float, payload) -> None:
        kind = payload[0]
        if kind == "arrive":
            r = payload[1]
            pre[_least_tokens(pre)].submit(Request(r.req_id, r.arrival_s, r.prompt_tokens, r.cached_tokens, 1,
                                                   r.session_id, r.load_s))
        elif kind == "done":
            idx, rec = payload[1], payload[2]
            if idx < n_prefill:  # prefill 完成 -> 传 KV
                r = by_id[rec["req_id"]]
                first[r.req_id] = rec["first_token_s"]
                if r.output_tokens == 1:
                    done_records[r.req_id] = {**rec, "output_tokens": 1}
                    return
                total = r.prompt_tokens * kvb / link_bw
                exposed = max(total / model.n_layers, total - prefill_time(r)) if layerwise else total
                transfer.append((total, exposed))
                heapq.heappush(events, (now + exposed, 1, next(seq), ("to_decode", r)))
            else:
                done_records[rec["req_id"]] = rec
        else:  # to_decode
            r = payload[1]
            dec[_least_tokens(dec)].submit(Request(r.req_id, now, r.prompt_tokens + 1, r.prompt_tokens,
                                                   r.output_tokens - 1, r.session_id, own_cached_kv=True))

    _run(engines, events, handle)
    records, itl = [], [c for e in dec for c in e._itl_chunks]
    first_gaps = []
    for r in requests:
        rec = done_records[r.req_id]
        t1 = first[r.req_id]
        if r.output_tokens > 1:
            first_gaps.append(rec["first_token_s"] - t1)  # 第 1 → 第 2 个 token：传输 + decode 侧排队
        records.append({
            "req_id": r.req_id, "session_id": r.session_id, "arrival_s": r.arrival_s,
            "first_token_s": t1, "finish_s": rec["finish_s"], "ttft_s": t1 - r.arrival_s,
            "tpot_s": (rec["finish_s"] - t1) / (r.output_tokens - 1) if r.output_tokens > 1 else 0.0,
            "max_itl_s": max(rec.get("max_itl_s", 0.0), first_gaps[-1] if r.output_tokens > 1 else 0.0),
            "prompt_tokens": r.prompt_tokens, "cached_tokens": min(r.cached_tokens, r.prompt_tokens - 1),
            "output_tokens": r.output_tokens,
        })
    if first_gaps:
        itl.append(np.asarray(first_gaps))
    tr = np.array(transfer) if transfer else np.zeros((1, 2))
    out = summarize(records, 0.0, itl)
    makespan = out["makespan_s"]
    out.pop("busy_frac")
    out.update({
        "prefill_busy_frac": float(np.mean([e.busy_s for e in pre])) / makespan,
        "decode_busy_frac": float(np.mean([e.busy_s for e in dec])) / makespan,
        "kv_transfer_s_mean": float(tr[:, 0].mean()),
        "kv_transfer_exposed_s_mean": float(tr[:, 1].mean()),
        "n_gpus_total": n_prefill * prefill_cfg.n_gpus + n_decode * decode_cfg.n_gpus,
        "decode_kv_peak_frac": max(e.kv_peak / e.kv_capacity for e in dec),
        "decode_preemptions": sum(e.preemptions for e in dec),
    })
    _goodput(out, ttft_slo_s, tpot_slo_s)
    return out
