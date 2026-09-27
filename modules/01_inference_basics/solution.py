"""模块 01 参考答案。签名与 exercise.py 完全一致。"""

from __future__ import annotations

from codesign_sim.hardware import GPU, ridge_point
from codesign_sim.model import (
    ModelConfig,
    decode_bytes,
    decode_flops,
    kv_bytes_per_token,
    prefill_flops,
    step_time_s,
)


def max_concurrency(
    gpu: GPU,
    cfg: ModelConfig,
    context_len: int,
    n_gpus: int = 1,
    weight_bytes: float = 2.0,
    kv_dtype_bytes: int = 2,
    mem_frac: float = 0.9,
) -> int:
    if context_len <= 0:
        raise ValueError("context_len 必须 > 0")
    usable = n_gpus * gpu.hbm_gb * 1e9 * mem_frac
    weights = cfg.total_params_b * 1e9 * weight_bytes
    per_seq = context_len * kv_bytes_per_token(cfg, kv_dtype_bytes)
    return max(int((usable - weights) // per_seq), 0)


def _decode_intensity(cfg: ModelConfig, batch: int, context_len: int, weight_bytes: float, kv_dtype_bytes: int) -> float:
    flops = decode_flops(cfg, batch, context_len)
    bytes_ = decode_bytes(cfg, batch, context_len, weight_bytes, kv_dtype_bytes)
    return flops / bytes_


def critical_batch(
    gpu: GPU,
    cfg: ModelConfig,
    context_len: int,
    max_batch: int = 4096,
    weight_bytes: float = 2.0,
    kv_dtype_bytes: int = 2,
) -> int | None:
    ridge = ridge_point(gpu, "bf16")

    def reaches(b: int) -> bool:
        return _decode_intensity(cfg, b, context_len, weight_bytes, kv_dtype_bytes) >= ridge

    if not reaches(max_batch):
        return None
    # intensity 随 batch 单调不减 -> 二分找第一个达标的 batch
    lo, hi = 1, max_batch
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def estimate_ttft(
    gpu: GPU,
    cfg: ModelConfig,
    prompt_tokens: int,
    cached_tokens: int = 0,
    n_gpus: int = 1,
    mfu: float = 0.6,
    mbu: float = 0.8,
    queue_s: float = 0.0,
    weight_bytes: float = 2.0,
) -> float:
    if prompt_tokens < 1 or not 0 <= cached_tokens <= prompt_tokens:
        raise ValueError("需要 prompt_tokens >= 1 且 0 <= cached_tokens <= prompt_tokens")
    n_new = max(prompt_tokens - cached_tokens, 1)  # 最后一个位置的 logits 必须现算
    n_cached = prompt_tokens - n_new
    flops = prefill_flops(cfg, n_new, n_cached)
    # 权重读一遍 + 读已缓存的 KV、写新 KV
    bytes_ = cfg.total_params_b * 1e9 * weight_bytes + prompt_tokens * kv_bytes_per_token(cfg)
    return queue_s + step_time_s(gpu, cfg, flops, bytes_, n_gpus, mfu, mbu)


def tool_wait_kv_action(
    gpu: GPU,
    cfg: ModelConfig,
    context_len: int,
    tool_s: float,
    n_gpus: int = 1,
    offload_gbps: float = 0.0,
    mfu: float = 0.6,
    mem_frac: float = 0.9,
) -> tuple[str, float]:
    if context_len <= 0 or tool_s < 0 or offload_gbps < 0:
        raise ValueError("需要 context_len > 0、tool_s >= 0、offload_gbps >= 0")
    kv_bytes = context_len * kv_bytes_per_token(cfg)
    kv_budget = n_gpus * gpu.hbm_gb * 1e9 * mem_frac - cfg.total_params_b * 1e9 * 2
    if kv_budget <= 0:
        raise ValueError("权重放不下，没有 KV 空间")
    share = kv_bytes / kv_budget
    recompute_s = estimate_ttft(gpu, cfg, context_len, 0, n_gpus, mfu)
    if tool_s * share <= recompute_s:  # 占住的「整机秒」不超过重算的「整机秒」
        return "keep", 0.0
    if offload_gbps > 0:
        reload_s = kv_bytes / (offload_gbps * 1e9)
        if reload_s < recompute_s:
            return "offload", reload_s
    return "recompute", recompute_s
