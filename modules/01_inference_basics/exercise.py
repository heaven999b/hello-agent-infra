"""模块 01 练习：三个「面试时要能当场心算」的推理成本函数，加一个需要做判断的决策函数。

你要实现四个函数：

    max_concurrency(gpu, cfg, context_len, ...)             —— (a) 显存装下权重后，最多能同时跑多少条序列
    critical_batch(gpu, cfg, context_len, ...)              —— (b) decode 从 memory-bound 变成 compute-bound 的最小 batch
    estimate_ttft(gpu, cfg, prompt_tokens, ...)             —— (c) 估算首 token 延迟（考虑前缀缓存命中）
    tool_wait_kv_action(gpu, cfg, context_len, tool_s, ...) —— (d) 工具调用期间，这段 KV 该留、该卸载还是该丢掉重算

写完后运行：
    make module N=01
    # 或
    .venv/bin/python -m pytest modules/01_inference_basics

提示：
- 只用 codesign_sim 的公开接口：GPU / ridge_point（hardware），ModelConfig / kv_bytes_per_token /
  prefill_flops / decode_flops / decode_bytes / step_time_s（model）。
- 单位：GPU.hbm_gb 是 GB（1e9 字节），ModelConfig.total_params_b 是十亿参数。
- 对应 README 第 5–8 节。卡住了先跑一遍 lab.py，看看表格里的数是怎么来的。
- (d) 可以直接调用你写好的 estimate_ttft。
"""

from __future__ import annotations

from codesign_sim.hardware import GPU, ridge_point  # noqa: F401  实现时会用到
from codesign_sim.model import (  # noqa: F401  实现时会用到
    ModelConfig,
    decode_bytes,
    decode_flops,
    kv_bytes_per_token,
    prefill_flops,
    step_time_s,
)

# ---------------------------------------------------------------------------
# TODO (a)：给定显存，最多能同时跑多少条序列
# ---------------------------------------------------------------------------


def max_concurrency(
    gpu: GPU,
    cfg: ModelConfig,
    context_len: int,
    n_gpus: int = 1,
    weight_bytes: float = 2.0,
    kv_dtype_bytes: int = 2,
    mem_frac: float = 0.9,
) -> int:
    """n_gpus 张 gpu 装下 cfg 的全部权重后，还能同时容纳多少条长度为 context_len 的序列的 KV。

    这是「聚合显存」的乐观上界：假设 n_gpus 张卡组成一个模型并行组，权重只存一份、KV 均匀切分。
    真实部署要按每张卡算（权重复制、KV 复制、冗余专家等，见 README 第 6 节）。

    规则（测试按这个规则算期望值）：
    1. 可用显存 = n_gpus × gpu.hbm_gb × 1e9 × mem_frac（其余留给激活、临时缓冲、碎片）。
    2. 权重字节 = cfg.total_params_b × 1e9 × weight_bytes（MoE 要放下所有专家，所以用 total 不用 active）。
    3. 每条序列的 KV = context_len × kv_bytes_per_token(cfg, kv_dtype_bytes)。
    4. 返回 floor((可用显存 − 权重字节) / 每条序列的 KV)；权重放不下（结果为负）时返回 0。
    5. context_len <= 0 → ValueError。

    例：Llama-3-8B、1 × H100、8k 上下文 → 52；Llama-3-70B、8 × H100、32k 上下文 → 40。
    """
    raise NotImplementedError("TODO (a): floor((n_gpus·HBM·mem_frac − 权重) / (context_len·KV 每 token))，负数返回 0")


# ---------------------------------------------------------------------------
# TODO (b)：decode 的临界 batch
# ---------------------------------------------------------------------------


def critical_batch(
    gpu: GPU,
    cfg: ModelConfig,
    context_len: int,
    max_batch: int = 4096,
    weight_bytes: float = 2.0,
    kv_dtype_bytes: int = 2,
) -> int | None:
    """返回让 decode 一步的 arithmetic intensity 达到 ridge point 的最小 batch；到 max_batch 还达不到就返回 None。

    规则：
    1. intensity(B) = decode_flops(cfg, B, context_len) / decode_bytes(cfg, B, context_len, weight_bytes, kv_dtype_bytes)。
       注意把 weight_bytes 和 kv_dtype_bytes 传下去——测试会用 FP8 权重 / FP8 KV 检查。
    2. ridge = ridge_point(gpu, "bf16")。weight_bytes=1 在这里表示「只把权重存成 FP8、矩阵乘仍按 BF16 算」
       （weight-only 量化）；如果矩阵乘也用 FP8，ridge 同样翻倍，两者抵消（README 第 3.4 节）。
    3. 返回 1..max_batch 中第一个满足 intensity(B) >= ridge 的 B；都不满足返回 None。
    4. 不考虑显存装不装得下（那是 (a) 的事）——这里只回答「算术上能不能」。
       达到 ridge 也只是必要条件：真实 kernel 在小矩阵上拿不到峰值（README 第 5 节「roofline 是上界」）。

    为什么会返回 None：B 很大时分母被 KV 读取主导，intensity 趋近一个与 B 无关的上限
    （README 第 5 节）。GQA 模型在几千 token 的上下文下，这个上限就已经低于 H100 的 ridge（约 295）。

    提示：intensity(B) 随 B 单调不减，可以二分；直接线性扫描 4096 次也足够快。
    """
    raise NotImplementedError("TODO (b): 找最小的 B 使 decode_flops/decode_bytes >= ridge_point(gpu)，否则 None")


# ---------------------------------------------------------------------------
# TODO (c)：估算 TTFT
# ---------------------------------------------------------------------------


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
    """TTFT ≈ 排队时间 + prefill 未命中部分的时间（单位：秒；不含通信、采样、网络，是一个下界）。

    规则：
    1. prompt_tokens >= 1 且 0 <= cached_tokens <= prompt_tokens，否则 ValueError。
    2. 要算的新 token 数 n_new = max(prompt_tokens − cached_tokens, 1)：输出第一个 token 需要最后一个位置的
       logits，即使整段都命中缓存，也至少要算 1 个 token；此时已缓存的 token 数 n_cached = prompt_tokens − n_new。
    3. FLOPs = prefill_flops(cfg, n_new, n_cached)；
       字节 = cfg.total_params_b × 1e9 × weight_bytes（权重读一遍）
            + prompt_tokens × kv_bytes_per_token(cfg)（读已缓存的 KV、写新算出的 KV，BF16）。
    4. 返回 queue_s + step_time_s(gpu, cfg, FLOPs, 字节, n_gpus, mfu, mbu)。

    例：Llama-3-8B、4k prompt、1 × H100、MFU 0.6、无缓存 → 约 0.10 s（compute-bound）；
    同样 4k prompt 但命中 3,968 个 token、只算 128 个新 token → 约 6.2 ms，此时变成 memory-bound（README 第 3.2 节）。
    """
    raise NotImplementedError("TODO (c): queue_s + step_time_s(prefill_flops(n_new, n_cached), 权重字节 + KV 字节)")


# ---------------------------------------------------------------------------
# TODO (d)：工具调用期间 KV 留、卸载还是重算
# ---------------------------------------------------------------------------


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
    """一个 agent 会话发出工具调用，预计 tool_s 秒后带着结果回来。它的 KV（context_len 个 token）该怎么办？

    返回 (动作, 工具返回后额外付出的延迟秒数)，动作是 "keep" / "offload" / "recompute" 之一。

    成本模型（README 第 11 节有推导和它的适用边界）：
    - 假设这个实例的并发受 KV 显存限制，所以占住 KV 容量的一定份额、占多久，就等于损失了多少「整机秒」的吞吐。
    - 重算要占用整机算力 recompute_s 秒，也按「整机秒」计。两者可以直接比较。
    - 卸载（到 CPU DRAM / SSD / 远端）几乎不占算力，代价是回来时要把 KV 搬回来，增加延迟。

    规则（BF16 权重、BF16 KV）：
    1. context_len <= 0、tool_s < 0、offload_gbps < 0 → ValueError。
    2. kv_bytes = context_len × kv_bytes_per_token(cfg)。
    3. kv_budget = n_gpus × gpu.hbm_gb × 1e9 × mem_frac − cfg.total_params_b × 1e9 × 2；<= 0（权重都放不下）→ ValueError。
    4. share = kv_bytes / kv_budget（这段 KV 占整机 KV 容量的份额）。
    5. recompute_s = estimate_ttft(gpu, cfg, context_len, 0, n_gpus, mfu)（从头 prefill 整段上下文）。
    6. 如果 tool_s × share <= recompute_s：返回 ("keep", 0.0)。
    7. 否则，如果 offload_gbps > 0 且 reload_s = kv_bytes / (offload_gbps × 1e9) < recompute_s：返回 ("offload", reload_s)。
       offload_gbps 是整个实例搬回 KV 的合计带宽（GB/s），例如 8 张卡各走 PCIe Gen5 单向约 64 GB/s → 512。
    8. 否则返回 ("recompute", recompute_s)。

    例：Llama-3-70B、8 × H100、64k 上下文：recompute_s ≈ 3.08 s，share ≈ 4.9%，所以工具等待在约 62 s 以内就该留；
    等 300 s、有 512 GB/s 的卸载带宽时该卸载（回来约 42 ms）；没有卸载层时只能重算。
    """
    raise NotImplementedError("TODO (d): 先比 tool_s×share 和 recompute_s，再比 reload_s 和 recompute_s")
