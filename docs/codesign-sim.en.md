[中文](codesign-sim.md) · **English**

# codesign_sim API Reference

> `codesign_sim` is the course's teaching simulator. Using a roofline cost model plus discrete-event simulation, it reproduces the **directional conclusions and orders of magnitude** of inference engines, prefix caching, multi-instance routing, P/D disaggregation, RL rollout scheduling and multi-agent credit assignment in a few seconds on an M1 laptop. It depends only on the Python standard library + numpy, and all randomness is fixed by `seed`. This version is a major revision after two independent reviews: wherever "changing a parameter flipped the conclusion", the model was either fixed or its range of validity is stated below (change list in [§11](#11-behavior-and-default-changes-since-the-first-version)).

## 0. Quick start

```bash
make setup                      # 创建 .venv 并以 editable 方式安装 codesign_sim
.venv/bin/python -m pytest tests -q   # 跑模拟器自身的测试
```

The script below exercises each main submodule once (copy-paste runnable):

```python
import numpy as np
from codesign_sim import (GPUS, MODELS, EngineConfig, Request, RolloutConfig, generate_sessions,
                          kv_bytes_per_token, make_team_task, ridge_point, route_and_simulate,
                          simulate, credit_error_stats)
from codesign_sim.model import weight_bytes_touched
from codesign_sim.pd import simulate_colocated, simulate_pd
from codesign_sim.rollout import compare_rollout_modes

# 1) roofline 拐点、KV 字节、MoE 按专家粒度的权重字节
for g in ["A100-80G", "H100-SXM", "H20", "B200"]:
    print(f"{g:9s} ridge(bf16) = {ridge_point(GPUS[g]):6.1f} FLOPs/B")
for m in ["llama3.1-70b", "deepseek-v3"]:
    print(f"{m:12s} KV/token = {kv_bytes_per_token(MODELS[m]) / 1024:6.1f} KiB")
for n in (1, 8, 64):
    print(f"deepseek-v3 FP8 weights touched, {n:3d} tokens/step = {weight_bytes_touched(MODELS['deepseek-v3'], n, 1) / 1e9:5.0f} GB")

# 2) 单实例：chunked prefill 对 decode 抖动的影响
reqs = [Request(i, 0.0, 500, 0, 600) for i in range(16)]
reqs += [Request(100 + i, 0.5 + 0.8 * i, 12000, 0, 20) for i in range(8)]
for chunked in (False, True):
    cfg = EngineConfig(GPUS["H100-SXM"], MODELS["llama3.1-70b"], n_gpus=8,
                       max_batch_tokens=2048, chunked_prefill=chunked)
    out = simulate(reqs, cfg)
    print(f"chunked={chunked!s:5s} ttft_p50={out['ttft_p50']:.2f}s itl_max={out['itl_max']*1e3:.0f}ms")

# 3) 多实例路由：热点场景（1 个 project）下纯亲和 vs SGLang 式负载上限
hot = generate_sessions(100, 1, seed=0, arrival_rate=3.0, mean_turns=6)
cfg = EngineConfig(GPUS["H100-SXM"], MODELS["llama3.1-8b"], n_gpus=1)
for policy in ["least_loaded", "cache_affinity", "affinity_with_load_cap", "session_sticky"]:
    out = route_and_simulate(hot, 4, policy, cfg, cache_capacity_blocks=8000)
    print(f"{policy:23s} hit={out['hit_rate']:.2f} ttft_p99={out['ttft_p99']:.2f}s "
          f"imbalance={out['load_imbalance']:.2f} busiest_share={out['max_instance_share']:.2f}")

# 4) P/D 分离 vs 共置（同样 16 张 H100）
rng = np.random.default_rng(0)
t, pd_reqs = 0.0, []
for i in range(200):
    t += rng.exponential(1 / 3.0)
    pd_reqs.append(Request(i, t, int(rng.integers(2000, 16000)), 0, int(rng.integers(100, 500))))
cfg70 = EngineConfig(GPUS["H100-SXM"], MODELS["llama3.1-70b"], n_gpus=8)
co, pd = simulate_colocated(pd_reqs, cfg70, n_instances=2), simulate_pd(pd_reqs, cfg70, n_prefill=1, n_decode=1)
for name, o in [("colocated 2x8", co), ("P/D 1x8 + 1x8", pd)]:
    print(f"{name:14s} ttft_p50={o['ttft_p50']:.2f}s tpot_p99={o['tpot_p99']*1e3:.0f}ms itl_p99={o['itl_p99']*1e3:.0f}ms")

# 5) RL rollout：同样 GPU 总数下拆解收益
cmp = compare_rollout_modes(RolloutConfig(), n_steps=50)
for k, r in cmp["runs"].items():
    print(f"{k:20s} Mtok/h={r['train_tokens_per_hour']/1e6:5.1f} busy={r['worker_busy_frac']:.2f} "
          f"staleness(traj/token)={r['mean_staleness']:.2f}/{r['token_staleness_mean']:.2f} length_ratio={r['length_ratio']:.2f}")
print(f"tail_gain={cmp['tail_gain']:.2f} overlap_gain={cmp['overlap_gain']:.2f}")

# 6) 信用分配：相对 Shapley 的偏移 ≠ 策略梯度偏差
task = make_team_task(6, 4, seed=0)
for method in ["shared", "difference", "shapley"]:
    st = credit_error_stats(task, method)
    print(f"{method:10s} error_var={st['error_var']:.3f} offset={st['mean_offset_vs_shapley']:+.2f} "
          f"grad_rel_err={st['grad_rel_err']:.3f}")
```

Actual output (M1, about 2 s):

```text
A100-80G  ridge(bf16) =  153.0 FLOPs/B
H100-SXM  ridge(bf16) =  295.2 FLOPs/B
H20       ridge(bf16) =   37.0 FLOPs/B
B200      ridge(bf16) =  281.2 FLOPs/B
llama3.1-70b KV/token =  320.0 KiB
deepseek-v3  KV/token =   68.6 KiB
deepseek-v3 FP8 weights touched,   1 tokens/step =    37 GB
deepseek-v3 FP8 weights touched,   8 tokens/step =   163 GB
deepseek-v3 FP8 weights touched,  64 tokens/step =   584 GB
chunked=False ttft_p50=0.20s itl_max=393ms
chunked=True  ttft_p50=0.21s itl_max=69ms
least_loaded            hit=0.65 ttft_p99=3.11s imbalance=1.24 busiest_share=0.26
cache_affinity          hit=0.80 ttft_p99=3.40s imbalance=4.00 busiest_share=1.00
affinity_with_load_cap  hit=0.89 ttft_p99=0.94s imbalance=1.86 busiest_share=0.71
session_sticky          hit=0.91 ttft_p99=1.05s imbalance=1.37 busiest_share=0.30
colocated 2x8  ttft_p50=0.33s tpot_p99=22ms itl_p99=255ms
P/D 1x8 + 1x8  ttft_p50=0.74s tpot_p99=8ms itl_p99=8ms
sync_colocated       Mtok/h=  3.5 busy=0.15 staleness(traj/token)=0.00/0.00 length_ratio=1.00
partial_colocated    Mtok/h= 12.8 busy=0.56 staleness(traj/token)=0.31/0.81 length_ratio=0.97
async_colocated      Mtok/h= 13.2 busy=0.58 staleness(traj/token)=0.24/0.54 length_ratio=0.99
async_disaggregated  Mtok/h= 12.6 busy=0.75 staleness(traj/token)=0.97/1.25 length_ratio=0.99
tail_gain=3.78 overlap_gain=0.96
shared     error_var=3.026 offset=-0.64 grad_rel_err=0.000
difference error_var=0.602 offset=+0.47 grad_rel_err=0.000
shapley    error_var=0.007 offset=+0.00 grad_rel_err=0.145
```

How to read it:

- The H20's ridge point is only 37 FLOPs/B (compute cut, bandwidth actually higher), so it is naturally suited to decode; MLA makes the per-token KV of the 671B DeepSeek-V3 about 1/4.7 of the 70B GQA model.
- MoE weights are read at expert granularity: 1 token reads only 37 GB (FP8), 8 tokens already 163 GB, 64 tokens 584 GB — a slightly larger decode batch touches nearly all experts, which is why EP spreads experts over many GPUs.
- Chunked prefill cuts the worst inter-token latency (ITL) from 393 ms to 69 ms.
- Hotspot scenario: pure `cache_affinity` pulls 100% of requests onto one instance (the other three starve), and its TTFT p99 of 3.40 s is even worse than `least_loaded`; SGLang-style `affinity_with_load_cap` falls back to the shortest queue when imbalanced, bringing p99 down to 0.94 s with an even higher hit rate; the session-sticky baseline `session_sticky` is also strong.
- P/D disaggregation: decode is no longer interrupted by long prompts, so TPOT p99 drops from 22 ms to 8 ms; the cost is that only half the GPUs do prefill, so TTFT p50 rises from 0.33 s to 0.74 s.
- RL rollout at the **same total GPU count**: nearly all of the gain comes from "not waiting for the long tail" (tail_gain 3.78×); "overlapping training with generation" adds almost no throughput at equal resources (overlap_gain 0.96×) — part of async's advantage in the old version came from free trainer GPUs. Token-weighted staleness is higher than per-trajectory staleness (long trajectories span more versions).
- Credit assignment: difference has a +0.47 offset relative to Shapley, yet its expected gradient is exactly unbiased; Shapley against a uniformly random teammate background has almost no offset and the lowest variance, but a 14.5% gradient bias — "low variance ≠ unbiased".

## 1. Design principles and shared simplifications

| Principle | How |
|---|---|
| Pedagogically correct | Mechanisms match real systems: the radix prefix tree evicts only from the leaf side, continuous batching advances step by step, the first token is emitted at the end of the step that finishes prefill, MLA decode uses the "absorbed weights" path, AReaL's interruptible continuation and rate limiting |
| Credible orders of magnitude | Hardware and model parameters come from official or the most authoritative available sources (Sections 2 and 3; H20 is reported + inferred); step time is roofline: `max(GEMM/(peak[gemm_dtype]·mfu) + attention/(peak[attn_dtype]·mfu), bytes/(bandwidth·mbu))` + optional communication, default mfu=0.6, mbu=0.8 |
| Conclusions survive parameter sweeps | The ranges covered by the reviewers' sweep/counterexample scripts are in the tests or in the "range of validity" notes here |
| Readable code | 150–500 lines per file, short functions, comments in Chinese; public interfaces follow the course's agreed contract, and the library only **adds** helper functions and defaulted fields/parameters beyond it |
| Reproducible | All randomness uses `numpy.random.default_rng(seed)`; block hashes only apply CPython's built-in `hash` to integer tuples, unaffected by `PYTHONHASHSEED`. Results are bit-for-bit reproducible **on the same Python / NumPy version**, not guaranteed across versions |

Still ignored by default: kernel launch and CPU scheduling overhead (`EngineConfig.step_overhead_s` can add a constant), sampling, quantization error, memory fragmentation (roughly reserved via `mem_frac=0.9`). Communication is now **optional**: without `EngineConfig.tp` / `ep`, the engine still assumes "perfect sharding, no communication" (same as the first version).

## 2. hardware —— GPU specs, interconnect constants and roofline

```python
@dataclass(frozen=True)
class GPU:
    name: str; bf16_tflops: float; fp8_tflops: float | None
    hbm_gb: float; hbm_tbps: float; nvlink_gbps: float; source: str
GPUS: dict[str, GPU]
def ridge_point(gpu, dtype="bf16") -> float            # 峰值 TFLOPS / 带宽 TB/s，单位 FLOPs/Byte
def attainable_tflops(gpu, intensity, dtype="bf16") -> float   # min(峰值, 强度 × 带宽)
def peak_tflops(gpu, dtype="bf16") -> float            # 辅助：取某精度的 dense 峰值；不支持 FP8 时抛 ValueError
IB_GBPS_PER_GPU = 50.0; LINK_EFFICIENCY = 0.8; NVLINK_LATENCY_S = 3.33e-6; IB_LATENCY_S = 3.7e-6; PCIE_GEN5_GBPS = 128.0
```

Spec table (all **dense** values; FLOPs count "one multiply-add = 2 FLOPs"). NVIDIA's H100 table lists BF16 as "1,979 teraFLOPS\*" with the footnote "\* With sparsity", so dense is 989 — do not be misled by claims that 1,979 is dense; the A100 table lists both "312 | 624\*", which lets you check the convention:

| Name | BF16 TFLOPS | FP8 TFLOPS | HBM | Bandwidth | NVLink (bidirectional) | Source |
|---|---|---|---|---|---|---|
| A100-80G | 312 | unsupported | 80 GB | 2.039 TB/s | 600 GB/s | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) (SXM column) |
| H100-SXM | 989 | 1979 | 80 GB | 3.35 TB/s | 900 GB/s | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) (1,979\* / 3,958\* are sparse; PCIe Gen5 128 GB/s) |
| H800-SXM | 989 | 1979 | 80 GB | 3.35 TB/s | 400 GB/s | No public NVIDIA spec page; [DeepSeek arXiv 2505.09343](https://arxiv.org/abs/2505.09343) §4.1: Hopper like the H100, with only FP64 and NVLink (900→400 GB/s) reduced; [Tencent Cloud instance docs](https://www.tencentcloud.com/document/product/1236/62812): BF16 989 TFLOPS, 80GB, NVLink 400GB/s |
| H20 | 148 | 296 | 96 GB | 4.0 TB/s | 900 GB/s | **Reported + inferred**, no public NVIDIA spec page: [leaked specs reposted on Tencent Cloud's developer community](https://cloud.tencent.com/developer/news/1284500) (FP8 296 / FP16 148 TFLOPS, 96GB, 4.0TB/s; dense/sparse unstated, treated as dense here); the [Alibaba Cloud gn8v/ebmgn8v families](https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families) page says "96 GB HBM3, 4 TB/s, 900 GB/s NVLINK per GPU" but **does not name the model** (inferred to be H20 from the matching specs), and its FP32 figure of 39.5 TFLOPS differs from the widely reported 44 TFLOPS |
| B200 | 2250 | 4500 | 180 GB | 8.0 TB/s | 1800 GB/s | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/): 8-GPU HGX B200 FP8 72 PF, BF16 36 PF, footnote "sparse; dense is half"; [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/): 1,440 GB, 64 TB/s (180 GB and 8 TB/s per GPU) |

Interconnect constants (all overridable in `EngineConfig` / `route_and_simulate`): 400 Gb/s IB per GPU ≈ 50 GB/s, H800 NVLink 200 GB/s per direction with about 160 GB/s achievable (efficiency 0.8), 64B small-message latency NVLink 3.33 µs / IB 2.8–3.7 µs — all from [DeepSeek ISCA'25](https://arxiv.org/abs/2505.09343) §4.1 and Table 5; PCIe Gen5 x16 at 128 GB/s bidirectional (64 GB/s per direction) from the H100 spec page.

Ridge points you must be able to compute mentally: H100 BF16 ≈ 295 FLOPs/B, A100 ≈ 153, B200 ≈ 281, H20 ≈ 37. Batch-1 decode has intensity of about 1 FLOP/B, far left of the ridge, hence bandwidth-bound.

## 3. model —— model structure and cost model

```python
@dataclass(frozen=True)
class ModelConfig:
    name: str; n_layers: int; d_model: int; n_heads: int; n_kv_heads: int; head_dim: int
    attn: str                    # "mha" | "gqa" | "mla"
    total_params_b: float; active_params_b: float
    kv_lora_rank: int = 0; qk_rope_head_dim: int = 0; source: str = ""
    # 扩展字段（0 = 未知，相关修正自动关闭）
    vocab_size: int = 0; tie_embeddings: bool = False; max_context: int = 0
    n_routed_experts: int = 0; experts_per_tok: int = 0; moe_intermediate_size: int = 0
    n_dense_layers: int = 0; n_shared_experts: int = 0
MODELS: dict[str, ModelConfig]   # "llama3.1-8b", "llama3.1-70b", "qwen3-32b", "deepseek-v3"；旧键 "llama3-8b"/"llama3-70b" 是别名
def kv_bytes_per_token(cfg, kv_dtype_bytes=2) -> int
def prefill_flops(cfg, n_new, n_cached=0, *, logits_positions=1) -> float
def decode_flops(cfg, batch, context_len) -> float
def decode_bytes(cfg, batch, context_len, weight_bytes=2, kv_dtype_bytes=2, *, kv_replicas=1, expert_skew=0.0) -> float
def step_time_s(gpu, cfg, flops, bytes_, n_gpus=1, mfu=0.6, mbu=0.8, dtype="bf16", *,
                attn_flops=0.0, attn_dtype="bf16", comm_s=0.0, comm_overlap=0.0) -> float
# 辅助函数
def prefill_flops_split(cfg, n_new, n_cached=0, logits_positions=1) -> (gemm, attn)
def decode_flops_split(cfg, batch, context_len) -> (gemm, attn)
def weight_bytes_touched(cfg, n_tokens, weight_bytes=2, *, expert_skew=0.0) -> float
def embedding_params(cfg) -> (table, lm_head); def body_active_params(cfg) -> float
def moe_params(cfg) -> dict; def expected_unique_experts(E, k, n_tokens, skew=0.0) -> float
def allreduce_time_s(nbytes, n_ranks, link_gbps_bidir, latency_s=3.33e-6, efficiency=0.8) -> float
def tp_comm_time_s(gpu, cfg, n_tokens, tp, act_bytes=2) -> float
def ep_comm_time_s(gpu, cfg, n_tokens, ep, *, gpus_per_node=8, ib_gbps=50, dispatch_bytes=1, combine_bytes=2) -> float
def kv_capacity_tokens(gpu, cfg, n_gpus=1, weight_bytes=2, kv_dtype_bytes=2, mem_frac=0.9) -> int
```

| Name | Layers | d_model | Q heads / KV heads | head_dim | Attention | Total / active | Vocab | Context | KV bytes/token | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | 32 | 4096 | 32 / 8 | 128 | GQA | 8.03B | 128256 | 128k | 131,072 | [Meta llama-models `sku_list.py`](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py) (Llama 3.1 has the same structure as Llama 3, differing only in `use_scaled_rope`), [Llama 3.1 MODEL_CARD](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) (128k); parameter count from [HF](https://huggingface.co/meta-llama/Llama-3.1-8B) safetensors metadata 8,030,261,248 |
| llama3.1-70b | 80 | 8192 | 64 / 8 | 128 | GQA | 70.55B | 128256 | 128k | 327,680 | Same; [HF](https://huggingface.co/meta-llama/Llama-3.1-70B) metadata 70,553,706,496 |
| qwen3-32b | 64 | 5120 | 64 / 8 | 128 | GQA | 32.8B | 151936 | 32k native / 128k (YaRN) | 262,144 | [Qwen3-32B config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) and model card |
| deepseek-v3 | 61 | 7168 | 128 / 128 | 128 (+64 RoPE) | MLA (kv_lora_rank 512) | 671B / 37B | 129280 | 128k | 70,272 | [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) (256 routed experts, 8 per token, moe_intermediate_size 2048, first 3 layers dense, 1 shared expert) and model card |

Formulas:

- **KV**: GQA/MHA is `2 · L · n_kv_heads · head_dim · bytes`; MLA caches only the latent and the decoupled RoPE key: `L · (kv_lora_rank + qk_rope_head_dim) · bytes`.
- **GEMM**: `2 · N_body` FLOPs/token, where `N_body` = active parameters minus the embedding table and lm_head; lm_head is computed only at positions that need logits (1 per sequence in prefill by default, every token in decode); the embedding lookup costs no FLOPs. Embedding + lm_head are 13% of llama3.1-8b's parameters; counting them as GEMMs (as the old version did) overstates prefill FLOPs by about 13%.
- **Attention**: `c_attn · (n_new · n_cached + n_new(n_new+1)/2)`, with `c_attn = 2 · L · H · (d_qk + d_v)`: `4·L·H·head_dim` for GQA; MLA prefill uses the expanded `d_qk = 128+64, d_v = 128`, MLA decode the absorbed `d_qk = 512+64, d_v = 512`. Chunked prefill has exactly the same attention FLOPs as one-shot prefill (and the same GEMM FLOPs with `logits_positions=0`).
- **Weight reads**: the input embedding reads only batch rows; lm_head is read in full; the rest of a dense model is read once. **MoE at expert granularity**: non-routed parameters (attention, shared expert, the 3 dense FFN layers, embedding, gate — about 17.1B for V3) are read every step; each MoE layer (58 for V3) reads the "distinct selected experts", whose expectation under uniform routing is `E·(1-(1-k/E)^n)`; `expert_skew > 0` models hot experts. The old "parameters selected independently" approximation overstates by 30–50% for n=4–32.
- **FP8**: in `step_time_s`, `flops` is timed at `dtype` and `attn_flops` at `attn_dtype`. In the engine, GEMMs follow the weight precision by default (`weight_bytes=1` means FP8 GEMM) and the core attention defaults to BF16 — matching the [DeepSeek inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): "matrix multiplications and dispatch transmissions adopt FP8, while core MLA computations and combine transmissions use BF16".
- **Communication (optional)**: TP does 2 all-reduces per layer, `2(p-1)/p · n_tokens·d_model·2B / (NVLink per direction · 0.8) + 2α` (by default the two-shot algorithm on NVSwitch: one step each for reduce-scatter and all-gather, latency term 2α; `allreduce_time_s(..., algo="ring")` switches to a pure ring with latency term 2(p-1)α); EP does one dispatch (FP8, 1 B) + combine (BF16, 2 B) per MoE layer, `n_tokens/ep · k · d_model` elements per GPU with `(ep-1)/ep` going to other GPUs, of which `(ep-8)/(ep-1)` cross nodes over IB; the slower link wins. In `step_time_s`, a `comm_overlap` fraction overlaps with compute/memory (max), the rest is added serially. This is the only place H800 differs from H100: the TP8 all-reduce bandwidth term is 2.25× slower.
- **MLA + TP**: the latent KV is shared by all heads and cannot be split by heads, so every TP rank reads the full KV (`decode_bytes(..., kv_replicas=tp)`); this is why DeepSeek/SGLang use DP-attention for MLA.

## 4. workload —— multi-turn agent workload

```python
@dataclass
class Turn: new_prompt_tokens: int; output_tokens: int; tool_latency_s: float; rewrite: bool = False; keep_frac: float = 1.0
@dataclass
class AgentSession: session_id: int; project_id: int; shared_prefix_tokens: int; arrival_s: float; turns: list[Turn]
def generate_sessions(n_sessions, n_projects, seed=0, *, mean_turns=8, tail_alpha=1.5,
                      rewrite_prob=0.0, arrival_rate=1.0,
                      rewrite_keep_frac=0.5, project_zipf_s=1.0, auto_compact_tokens=100_000) -> list[AgentSession]
def to_blocks(session, upto_turn, block_size=16, include_output=False) -> list[int]
# 辅助函数
def context_tokens(session, upto_turn, include_output=False) -> int
def session_stats(sessions) -> dict           # 实测的轮数 / prompt / 输出均值与分位数、prompt:输出比、rewrite 比例
def pareto_scale_for_mean(mean, alpha, cap) -> float
class BlockHasher; def build_hasher(...); def append_prompt(...); def append_output(...)   # router 的增量 hash
```

The prompt of turn t = shared prefix + (new_prompt + output) of the previous t turns + new_prompt of turn t (append-only). Sampling distributions (plausible magnitudes for teaching, **not** statistics of any product):

| Quantity | Distribution |
|---|---|
| Shared prefix (system prompt + tool definitions + repo context) | One per project, lognormal median 6k tokens, clipped to [2k, 20k] |
| Turn-0 user message / tool result of each later turn | lognormal median 300 / 1k tokens (tool results clipped to [50, 16k]) |
| Output per turn | Pareto(shape `tail_alpha`), clipped at 4096; mean **after clipping** calibrated to 300 |
| Number of turns | Pareto(shape `tail_alpha`), clipped at 64; mean **after clipping** calibrated to `mean_turns` |
| Tool latency | lognormal median 2 s, clipped at 120 s; 0 for the last turn |
| Project popularity | Zipf, weight ∝ 1/(k+1)^`project_zipf_s` (default 1, same as the first version; 0 = uniform) |
| Arrivals | Poisson(`arrival_rate` sessions/s) |
| rewrite | True with probability `rewrite_prob` for each turn from turn 1 on; plus one automatic compaction whenever the context would exceed `auto_compact_tokens` |

- Mean calibration: in the old version the clipped mean shifted with `tail_alpha` (only 4.6 turns and 194 tokens on average at α=1.2), mixing tail and average load when sweeping α; now the Pareto scale is solved so α changes only the tail. Measured (1000 sessions): at α=1.2/1.5/2/3 the mean turns are 7.9/8.0/8.1/8.0 and the mean output 298/305/300/301.
- rewrite = rewrite **and compact**: the private history after the shared prefix becomes new content of length = original × `keep_frac` (default 0.5), so the cache is invalidated but prefill/KV also shrinks — enough to compute the break-even of compaction; `rewrite_keep_frac=1.0` means content changes without length change (e.g. the anti-pattern of inserting a timestamp into the prefix).
- auto-compact: models a harness compacting automatically near the context limit, keeping contexts within the models' 128k; `auto_compact_tokens=None` disables it.
- The prompt:output token ratio varies with seed and `tail_alpha`: 71–118:1 for 200 sessions and seeds 0–4, about 51:1 at α=3 — agent workloads are prefill-dominated.

`to_blocks` approximates block identity with **chained hashes** (as vLLM prefix caching does): **ignoring hash collisions**, equal hash ⇔ identical content from the start through this block. Only full blocks are returned; sessions of one project share the first `shared_prefix_tokens // block_size` blocks; without rewrite each turn has the previous prompt+output as a prefix; after a rewrite every hash past the shared prefix changes.

## 5. cache —— block-level radix prefix cache (optionally two tiers)

```python
class PrefixCache:
    def __init__(self, capacity_blocks: int, policy: str = "lru", *, dram_blocks: int = 0): ...
    def match(self, blocks, now) -> int          # 命中的前缀 block 数（HBM + DRAM）；只刷新真正命中的部分
    def insert(self, blocks, now, pin_until=None) -> None
    stats: dict   # hit_blocks, miss_blocks, evictions, inserted_blocks, rejected_blocks, lookups,
                  # hbm_hit_blocks, dram_hit_blocks, demoted_blocks, promoted_blocks
    # 辅助
    def match_tiers(self, blocks, now) -> (hbm_hits, dram_hits)
    def peek(self, blocks) -> int                # 不刷新、不计统计（router 打分用）
    hit_rate: float; size: int; def tier_sizes(self) -> (hbm, dram); def leaves(self) -> list[list[int]]
```

- Structure: a compressed radix tree, as in SGLang RadixAttention; nodes are split on insert as needed and shared prefixes are stored once. **When match stops in the middle of an edge it splits first** and refreshes only the matched head; the unused suffix keeps its old access time (the first version refreshed the whole edge, over-protecting long suffixes).
- Eviction happens only from the leaf side, trimming exactly the needed number of blocks from the tail.
- `lru`: ignores `pin_until`. `ttl`: **hard-keeps** until `pin_until`; if everything is pinned, the new insert is truncated (`rejected_blocks`); when a node is pinned repeatedly the **latest** pin wins. `session_aware`: `pin_until` is an "expected next use time" (soft hint): leaves without a hint or with an expired hint go first by LRU, then the one expected back latest; when a node has several future users the **nearest** one counts.
- Two tiers (`dram_blocks > 0`): when HBM is full, the tail of an "HBM frontier node" (an HBM node with no HBM children) is **demoted** to DRAM; only a full DRAM tier really drops blocks. Along a path HBM comes first, DRAM after. insert **promotes** DRAM nodes it passes back to HBM. The caller pays for moving DRAM hits (the router charges PCIe bandwidth and treats a hit as a miss when recompute is cheaper).

## 6. engine —— single-instance continuous batching + chunked prefill

```python
@dataclass
class Request: req_id: int; arrival_s: float; prompt_tokens: int; cached_tokens: int; output_tokens: int
               session_id: int = -1; load_s: float = 0.0
@dataclass
class EngineConfig:
    gpu: GPU; model: ModelConfig; n_gpus: int = 8
    max_batch_tokens: int = 8192; chunked_prefill: bool = True; kv_capacity_tokens: int | None = None
    # 扩展字段（都有默认值）
    max_num_seqs: int = 256; mfu: float = 0.6; mbu: float = 0.8; weight_bytes: float = 2.0
    kv_dtype_bytes: int = 2; mem_frac: float = 0.9; step_overhead_s: float = 0.0
    gemm_dtype: str | None = None; attn_dtype: str = "bf16"; kv_reserve: str = "oracle"
    tp: int | None = None; ep: int = 1; dp_attention: bool = False; gpus_per_node: int = 8
    ib_gbps: float = 50.0; comm_overlap: float = 0.0; expert_skew: float = 0.0
def simulate(requests, cfg) -> dict
class Engine    # 可逐步推进：submit / step / has_work / next_step_time / inflight_tokens / summary（router、pd 复用）
```

Each step:

1. **Admission**: requests enter the running set in arrival order (a request with `load_s > 0` is not scheduled until its KV has been moved back from DRAM, and does not block ready requests behind it) until KV capacity or `max_num_seqs` is exhausted. Default KV capacity = `(n_gpus·HBM·mem_frac - weights) / KV bytes`; if the weights do not fit, `ValueError` is raised.
   - `kv_reserve="oracle"` (default): reserves `(prompt - cached prefix) + output` up front. **This uses the true output length, which is an oracle** — real schedulers do not know it; the upside is that preemption is never needed.
   - `kv_reserve="incremental"` (vLLM style): allocates only the prompt and grows per decode step; when memory runs out the most recently admitted sequence is preempted (recompute: free its KV, put it back at the head of the queue, re-prefill it together with its generated tokens). Use this for long-output RL scenarios to see the preemption cost (`preemptions`, `recomputed_tokens`).
   - Tokens that hit the prefix cache **physically share** KV with the cache and are not counted again for the request.
2. **Batch formation**: with `chunked_prefill=True`, decode goes first and the remaining budget is carved into prefill chunks FIFO; with `False`, whenever a request awaits prefill the step does only whole-prompt prefill and all decodes pause.
3. **Step time**: GEMMs at the peak of `gemm_dtype` (None follows `weight_bytes`: ≤1 byte means FP8), attention at the peak of `attn_dtype`; bytes = weights touched (MoE at expert granularity) + KV read (replicated tp times for MLA when `tp` is set without `dp_attention`) + new KV written; TP all-reduce when `tp` is set, EP dispatch/combine when `ep > 1`, overlapped per `comm_overlap`.
4. The first token is emitted at the end of the step that completes prefill; each decode step then emits 1 token. `cached_tokens` is capped at `prompt-1`.

Return value: `ttft_p50/p99`, `tpot_p50/p99`, `throughput_tok_s` (output-token throughput), `busy_frac`, `per_request` (including `preemptions`), plus `itl_p50/itl_p99/itl_max`, `makespan_s`, `n_steps`, `kv_capacity_tokens`, `prefill_tokens_computed`, `output_tokens`, `preemptions`, `recomputed_tokens`. A context longer than `model.max_context` emits one `UserWarning` (the simulation proceeds).

## 7. router —— multi-instance routing (closed-loop agent workload)

```python
def route_and_simulate(sessions, n_instances, policy, cfg, cache_capacity_blocks, seed=0, *,
                       block_size=16, cache_policy="lru", load_cap_factor=1.5, cache_threshold=0.3,
                       balance_abs_threshold=None, load_metric="tokens", est_output_tokens=300.0,
                       count_tool_waiting=False, est_return_tokens=1300.0, insert_on="prefill_done",
                       hint_mode="noisy", hint_ttl_s=10.0, hint_noise_sigma=1.0,
                       cache_dram_blocks=0, dram_gbps=None, sticky_migrate=False) -> dict
```

- Closed loop: turn t finishes → wait `tool_latency_s` → turn t+1 arrives. Each instance = one `Engine` (with real queueing: KV capacity, token budget, `max_num_seqs`) + its own `PrefixCache`.
- **When to write the cache**, `insert_on`: `"prefill_done"` (default: the prompt's blocks are written as soon as prefill finishes, so later requests with the same prefix hit — SGLang RadixCache's handling of unfinished requests); `"route"` (written immediately at routing, as SGLang router's approximate tree does; most optimistic); `"finish"` (first-version behavior: written only on completion — masks hotspots, see the table below). All three also write prompt+output on completion.
- **Pin hints**, `hint_mode` (for `ttl` / `session_aware`): `"noisy"` (default: finish time + true tool latency × exp(σ·N(0,1))), `"fixed"` (fixed TTL `hint_ttl_s`), `"oracle"` (true tool latency — equivalent to knowing when the next turn arrives, an **upper bound** that must be chosen explicitly), `"none"`. The first version defaulted to the oracle.
- **Load**, `load_metric`: `"tokens"` (default: remaining prefill + expected remaining decode, with unknown output lengths estimated by `est_output_tokens`) or `"requests"` (in-flight requests, SGLang's measure); `count_tool_waiting=True` also counts sessions "waiting on a tool and about to return" (`est_return_tokens` each).
- **Policies**:
  - `round_robin`: per-request rotation.
  - `least_loaded`: lowest load.
  - `cache_affinity`: hits only, longest prefix match (ties to the less loaded) — an **extreme control** that creates hotspots and starves cold instances.
  - `affinity_with_load_cap`: the structure of the [SGLang router cache_aware policy](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs): if `(max load - min load) > balance_abs_threshold` and `max > load_cap_factor · min`, the system is imbalanced and the least loaded instance is chosen; otherwise the longest match is chosen if the match rate > `cache_threshold`, else the least loaded. The SGLang gateway CLI defaults are cache_threshold 0.3, balance_abs_threshold 64, balance_rel_threshold 1.5 ([main.rs](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/main.rs)); 64 requests is production scale, so with `balance_abs_threshold=None` the library scales it down to 4 requests (`load_metric="requests"`) or 4 × `max_batch_tokens` (`"tokens"`). To reproduce SGLang's values: `load_metric="requests", balance_abs_threshold=64, cache_threshold=0.3, load_cap_factor=1.5`. `load_cap_factor` must be > 0.
  - `session_sticky`: a session's first turn goes to the least loaded instance and later turns return to the same one (session-stickiness baseline); with `sticky_migrate=True`, a session migrates to the least loaded instance when its sticky instance is imbalanced (same criterion as above).
- **Two-tier cache**: `cache_dram_blocks > 0` enables a DRAM tier; moving DRAM hits takes blocks × block_size × KV bytes / `dram_gbps` (default n_gpus × 64 GB/s, one PCIe Gen5 x16 link per GPU, one direction), compared against the prefill time to recompute those tokens, whichever is smaller (`dram_loaded_blocks` / `dram_recomputed_blocks`).
- Returns: `hit_rate` (HBM+DRAM), `hbm_hit_rate`, `dram_hit_rate`, `token_hit_rate`, `per_instance_hit_rate`, `load_imbalance` (max/mean instance busy time), `requests_per_instance`, `max_instance_share` (request share of the busiest instance), `busy_s_per_instance`, TTFT/TPOT/ITL percentiles, `throughput_tok_s`, `prefill_tokens_total/computed`, `evictions`, `dram_loaded_blocks`, `dram_recomputed_blocks`, `migrations`, `per_request` (including `instance`).

When the cache is written decides whether a hotspot can appear (1 project, 100 sessions, 3 sessions/s, 4 × H100 + llama3.1-8b, seed 0, measured):

| Policy | insert_on | Busiest-instance share | Load imbalance | TTFT p99 | Hit rate |
|---|---|---|---|---|---|
| cache_affinity | finish (first version) | 0.41 | 1.81 | 0.67 s | 0.91 |
| cache_affinity | prefill_done | 1.00 | 4.00 | 3.40 s | 0.80 |
| affinity_with_load_cap | prefill_done | 0.71 | 1.86 | 0.94 s | 0.89 |
| session_sticky | prefill_done | 0.30 | 1.37 | 1.05 s | 0.91 |

If the cache is written only on completion, early sessions of the same project tie at routing time and get spread out, so no hotspot forms; once prefill writes the cache, pure affinity pulls every request onto one instance and has the worst tail latency, and SGLang's imbalance fallback rescues it. The hotspot shows only when the busiest instance is saturated — at low load with many projects, pure affinity is still among the best policies for TTFT.

## 8. pd —— P/D disaggregation (new)

```python
from codesign_sim.pd import simulate_colocated, simulate_pd
def simulate_colocated(requests, cfg, n_instances=1, *, ttft_slo_s=None, tpot_slo_s=None) -> dict   # 共置对照
def simulate_pd(requests, prefill_cfg, decode_cfg=None, n_prefill=1, n_decode=1, *,
                kv_link_gbps=None, layerwise=True, ttft_slo_s=None, tpot_slo_s=None) -> dict
```

- Requests first enter the prefill pool; finishing prefill emits the first token (TTFT is decided here); the whole prompt's KV then goes over RDMA to the decode pool: transfer time = `prompt · KV bytes / (link bandwidth · 0.8)`, with the link defaulting to min(GPUs on either side) × 50 GB/s. With `layerwise=True` the transfer is pipelined by layer and only about 1/n_layers of it is on the critical path (anything beyond the prefill compute time is exposed too). In the decode pool the request is submitted as "the whole prompt is cached", with no further prefill; but that prompt KV is owned by the request (`Request.own_cached_kv=True`) and counted against the decode instance's KV capacity as prompt + generated tokens (the same accounting as the colocated baseline). When capacity runs out the request queues on the decode side, and with `kv_reserve="incremental"` the engine's preemption mode applies (a preempted request is fully recomputed on the decode instance). So decode concurrency no longer needs to be capped by hand with `max_num_seqs`.
- Returns: TTFT/TPOT/ITL percentiles (the gap from token 1 to token 2 includes transfer and decode-side queueing), `throughput_tok_s`, `prefill_busy_frac`, `decode_busy_frac`, `kv_transfer_s_mean`, `kv_transfer_exposed_s_mean`, `n_gpus_total`, `decode_kv_peak_frac` (decode-side peak KV / capacity), `decode_preemptions`, plus `goodput_req_s` (requests meeting both SLOs per second) and `slo_attainment` computed from `ttft_slo_s` / `tpot_slo_s`. `simulate_colocated` also returns goodput and `kv_peak_frac`.
- Decode-side KV is a real constraint: on the same 16 H100s with llama3.1-70b and 200 requests (TTFT ≤ 2 s, TPOT ≤ 30 ms), goodput is 2.11 requests/s with unlimited decode KV; capping it at 40k tokens fills the KV to its peak (0.999), raises TPOT p99 to 347 ms and drops goodput to 0.36 requests/s (asserted in the tests).
- Trade-off (measured in §0): on the same 16 H100s with llama3.1-70b, P/D 1+1 cuts TPOT p99 from 22 ms to 8 ms and raises TTFT p50 from 0.33 s to 0.74 s. The P:D ratio must be tuned to the load; MLA's 69 KB of KV per token makes transfer cheap, one reason DeepSeek can afford P/D disaggregation.

## 9. rollout —— RL rollout: sync / partial / async (equal-resource comparison)

```python
@dataclass
class RolloutConfig:
    n_workers: int = 64; batch_size: int = 256; tail_alpha: float = 1.5
    mean_gen_s: float = 20.0; mean_tool_s: float = 5.0; train_step_s: float = 30.0
    mode: str = "sync"; max_staleness: int = 1; partial_budget_s: float = 60.0
    # 扩展字段
    deploy: str = "colocated"; trainer_frac: float = 0.25; stale_policy: str = "interrupt"
    resume_kv: str = "recompute"; prompt_tokens: int = 2048; decode_tok_s: float = 100.0
    model: str = "qwen3-32b"; gpu: str = "H100-SXM"; gpus_per_worker: float = 0.125; engine_gpus: int = 8
    mfu: float = 0.6; partial_budget_tokens: int | None = None; over_provision: float = 1.0
def simulate_rollout(cfg, n_steps=50, seed=0) -> dict
def simulate_rollout_seeds(cfg, n_steps=50, seeds=(0, 1, 2, 3, 4)) -> {"mean", "std", "runs"}
def compare_rollout_modes(cfg=None, n_steps=50, seed=0) -> {"runs", "tail_gain", "overlap_gain", "total_gain"}
def mean_std(results: list[dict]) -> {key: (mean, std)}
```

- **Resources**: the cluster = `n_workers` units. With `deploy="colocated"` (default) all units alternate between rollout and training, and generation pauses while training; with `"disaggregated"` a `trainer_frac` share of units trains (one step takes `train_step_s / trainer_frac`) and the remaining `round(n_workers·(1-trainer_frac))` slots only roll out. Both deployments have the same total GPU-seconds, so they are directly comparable.
- **Trajectories**: generation time is Pareto (shape `tail_alpha`, clipped at 50× the mean, with the mean **after clipping** calibrated to `mean_gen_s`) + tool waiting (exponential, mean `mean_tool_s`); generated tokens = generation seconds × `decode_tok_s`.
- **sync**: generate `batch_size` trajectories per step and wait for the slowest before training (= async with η=0).
- **partial** (APRIL / Kimi k1.5 idea): over-provision, end the round as soon as `batch_size` trajectories have completed, pause the in-flight ones and continue them next round under the new policy; `partial_budget_s` (or `partial_budget_tokens`) is how far each trajectory may advance **per round**, beyond which it is also paused until the next round. **The training batch is always batch_size** (the first version trained on an under-filled batch of only 153–197 when a 60 s wall-clock budget expired). Generation and training do not overlap.
- **async** ([AReaL](https://arxiv.org/abs/2505.24298)): rollout runs continuously and the trainer trains as soon as a batch is ready, preferring older trajectories. New trajectories are rate-limited at submission per AReaL's Eq. (3): `⌊(N_r − 1)/B⌋ ≤ i + η` (N_r = trajectories submitted and not discarded, i = current version).
  - `stale_policy="interrupt"` (default, AReaL semantics): a weight update interrupts all in-flight generation, discards the old KV, recomputes it with the new weights and continues decoding; a trajectory consists of segments from several versions and no data is lost. The rate limit constrains the **submission side**; a single long trajectory can span many versions between start and training (`staleness_max` may exceed η).
  - `stale_policy="abort"` (control): in-flight trajectories with version lag > η are aborted and too-stale samples are dropped at training time (and their rate-limit quota is released — the first version did not release it, so the window shrank monotonically and long-run throughput collapsed to near sync). The hard bound `staleness_max ≤ η` costs **systematically dropping long trajectories** (`length_ratio` clearly < 1).
- **Recompute cost** (`resume_kv="recompute"`, default): a continued/interrupted trajectory re-prefills its context (`prompt_tokens` + generated tokens). In bulk cases (everything interrupted at a weight update, a new round resuming in bulk) the whole rollout cluster recomputes together and everyone waits the same `Σ prefill_flops(ctx_i) / (slots · gpus_per_worker · peak · mfu)`; a single continuation runs at the speed of one engine (`engine_gpus` GPUs). Recompute occupies the slot and counts as GPU-busy but produces no tokens (`recompute_frac`). `"retain"` reuses the old KV at no cost.
- **Staleness**: per trajectory = training version − oldest segment version; token-weighted = Σ segment tokens × (training version − segment version) / Σ tokens (PPO/GRPO losses are token-weighted, and long trajectories span the most versions).
- Returns: the contract's `worker_busy_frac` (slot time spent **generating**), `traj_per_hour`, `mean_staleness`, `p99_staleness`, `step_time_s`, plus `train_tokens_per_hour` (the length-independent main metric), `staleness_max`, `token_staleness_mean/max`, `gen_mean_started/trained`, `length_ratio`, `cluster_util` ((generation + recompute + training) GPU-seconds / total GPU-seconds), `gate_idle_frac` (share of slot time idle due to the rate limit / waiting for the tail), `recompute_frac`, `interrupts`, `max_gate_lead`, `total_time_s`, `trained_trajectories`, `mean_batch`, `wasted_frac`, `dropped_trajectories`, `worker_occupied_frac`, `n_slots`.

**Gain decomposition** (output of §0, default parameters, the same 64 units): sync colocated 3.5 M tokens/h → async colocated 13.2 (`tail_gain` 3.78×, from "not waiting for the tail") → async disaggregated 12.6 (`overlap_gain` 0.96×). At equal resources, "overlapping training with generation" adds almost no throughput: colocation already keeps every GPU busy during training; the real benefits of disaggregation (each side picks its best parallelism, no switching/resharding, no pause-then-recompute) are mostly outside this model. Also, colocated async is nearly equivalent to partial when the rate limit does not bind.

**Range of validity**: the rate-limit window is (η+1)·B trajectories. When n_workers approaches or exceeds (η+1)·B (in real systems a batch usually runs concurrently, so n_workers ≥ B is common), most slots are held back by the rate limit. Measured with the disaggregated deployment over 50 steps:

```text
n_workers/B  mode      eta  Mtok/h  busy  gate_idle  token_staleness
      0.25  partial     1     6.7  0.40       0.00  0.80
      0.25  async       1    12.6  0.75       0.03  1.25
      0.25  async       4    12.7  0.75       0.00  1.29
         1  partial     1    10.9  0.16       0.00  1.43
         1  async       1    14.0  0.21       0.74  1.30
         1  async       4    15.1  0.23       0.71  3.75
         2  partial     1    11.6  0.09       0.00  1.86
         2  async       1    14.1  0.10       0.87  1.30
         2  async       4    15.1  0.12       0.85  3.76
```

With n_workers ≥ B, async's `gate_idle_frac` reaches 0.7–0.9: partly the rate limit (raising η lifts throughput from 14.0 to 15.1, at the cost of token staleness rising from 1.3 to 3.8), partly the trainer becoming the bottleneck in the disaggregated deployment (120 s per step). Rule of thumb: keep (η+1)·B ≥ n_workers and size trainer_frac to the generation/training work. The "async worse than partial" the first version showed in this range came from abort+drop and free trainer GPUs; with AReaL semantics and equal resources async is still slightly ahead, but with many idle slots.

## 10. credit —— multi-agent credit assignment

```python
@dataclass
class TeamTask:          # n_agents, n_actions, unary W, pairwise Q, noise_std=0.5, default_action=0
    def expected_reward(self, joint_actions) -> np.ndarray          # f(a)
    def sample_rewards(self, joint_actions, rng) -> np.ndarray      # f(a) + 噪声
    def counterfactual_rewards(self, joint_actions) -> np.ndarray   # F[b,i,x] = f(a_{-i}, a_i←x)
    def true_contribution(self, joint_actions, ref_probs=None) -> np.ndarray   # Shapley；ref_probs=None 为均匀
    def true_gradient(self, probs) -> np.ndarray                    # ∇E_π[f]（精确）
    def expected_return(self, probs) -> float; def best_return(self) -> float
def make_team_task(n_agents, n_actions, seed=0, *, pair_scale=0.3, noise_std=0.5) -> TeamTask
def estimate_credit(task, joint_actions, rewards, method, policies=None, *,
                    shapley_ref="uniform", cf_noise="none", rng=None) -> np.ndarray
def train_policies(task, method, n_iters=300, batch=64, lr=0.1, seed=0, *,
                   shapley_ref="uniform", cf_noise="none") -> dict   # {"returns","grad_var","final_return","best_return"}
def credit_error_stats(task, method, n_samples=4096, seed=0, *, grad_probs=None) -> dict
def expected_gradient(task, method, probs=None, *, batch=64, shapley_ref="uniform", n_mc=20000, seed=0)
def gradient_bias(task, method, probs=None, **kw) -> {"grad_rel_err", "grad_cos"}
```

- Reward: `r = Σ_i W[i,a_i] + Σ_{i<j} Q_ij(a_i,a_j) + ε`. The **true individual contribution** is the Shapley value of f (absent agents averaged over uniformly random actions); the closed form is checked against brute-force enumeration.
- Estimators (`method`): `shared`, `group_baseline` (subtracts the batch mean including itself; the expected gradient is scaled by (B−1)/B with direction unchanged), `group_baseline_loo` (leave-one-out, strictly unbiased), `difference` (r − f(self replaced by default action 0)), `counterfactual_baseline` (COMA style, r − Σ_b π_i(b)·f(self replaced by b)), `shapley` (exact Shapley + noise split evenly). `shapley_ref="uniform"` (default) uses a uniformly random teammate background — lowest variance but not the current policy's gradient; `shapley_ref="policy"` uses the current policy as background and is strictly unbiased for independent-policy REINFORCE (verified with exact expected gradients in the tests).
- Counterfactual query noise, `cf_noise` (affects difference / counterfactual_baseline): `"none"` (default, a perfect critic), `"independent"` (each query is an independent noisy simulation), `"common"` (common random numbers: counterfactuals share the exogenous noise with the actual rollout, so the noise cancels exactly in the difference). Single-sample gradient variance: common < none < independent.
- `credit_error_stats`: `mean_offset_vs_shapley` (mean of estimate minus true Shapley; the old field name `bias` is kept as an alias — it is **not** a policy-gradient bias, since a constant offset does not change the expected gradient), `error_var`, `corr`, and `grad_rel_err` / `grad_cos` at a non-uniform policy (default: one Dirichlet(1) random policy per agent). At the uniform policy `shapley(uniform)` happens to be unbiased, hiding the problem.
- `train_policies`: `returns[t]` is the exact expected return of the policy **before** update t; `final_return` is after all updates (the first version reported the one before the last update).

**When variance differences become learning-speed differences**: with few agents and large batches the estimators' learning curves nearly coincide; only with many agents and small batches, where shared's per-agent signal-to-noise ratio is about 1/n_agents, does shared clearly fall behind. Final expected return after 300 iterations (mean of 3 seeds, 4 actions):

```text
n= 8 B= 8 shared=   7.4 group_base=   7.6 difference=   7.4 counterfac=   7.4 shapley=   7.4
n= 8 B=64 shared=   7.5 group_base=   7.4 difference=   7.4 counterfac=   7.4 shapley=   7.4
n=32 B= 8 shared=  35.3 group_base=  61.8 difference=  62.1 counterfac=  63.4 shapley=  63.6
n=32 B=64 shared=  58.4 group_base=  62.7 difference=  63.4 counterfac=  63.3 shapley=  63.3
n=64 B= 8 shared=  56.0 group_base= 137.6 difference= 146.9 counterfac= 146.8 shapley= 145.8
n=64 B=64 shared= 102.6 group_base= 147.7 difference= 147.5 counterfac= 147.3 shapley= 147.0
```

(`group_base` is `group_baseline_loo`, `shapley` is `shapley_ref="policy"`.) Takeaway: to see the value of credit assignment in an experiment, use n_agents ≥ 32 and batch ≤ 16.

## 11. Behavior and default changes since the first version

| Function / field | Change | Effect on old conclusions |
|---|---|---|
| `MODELS` | Renamed to `llama3.1-8b` / `llama3.1-70b` (old keys are aliases; `cfg.name` is the new name); new vocab / max_context / MoE fields | Structural parameters unchanged |
| `prefill_flops` / `decode_flops` | Embedding / lm_head no longer counted as GEMMs; prefill computes lm_head for 1 position | Prefill FLOPs about −13% for llama3.1-8b, −3% for 70B, −5% for V3 |
| `weight_bytes_touched` / `decode_bytes` | Input embedding reads only batch rows; MoE at expert granularity | 8B single-request TPOT about −6.5%; V3 decode bytes at small/medium batch down 30–50% |
| `step_time_s` | New `attn_flops/attn_dtype/comm_s/comm_overlap` (old calls unaffected) | None |
| `EngineConfig` | New gemm_dtype (**defaults to following weight_bytes**), attn_dtype, kv_reserve, tp, ep, dp_attention, etc. | Configs with `weight_bytes=1` are now timed with FP8 GEMMs; compute-bound prefill about 2× faster |
| `simulate` / `Engine` | KV of cached prefixes is shared with the cache, not counted again (except for `Request.own_cached_kv=True`) | Higher concurrency under KV pressure |
| `simulate_pd` | Decode side counts prompt + generated tokens against KV capacity and queues/preempts when full; new SLO goodput and `decode_kv_peak_frac` | The first revision had no decode-side KV constraint (needed a `max_num_seqs` workaround); now PD goodput drops noticeably when decode KV is tight |
| `generate_sessions` | Mean calibrated after clipping; rewrite compacts context (`rewrite_keep_frac` default 0.5); `auto_compact_tokens` default 100k; new `project_zipf_s` (default 1 = old behavior) | All session values for a given seed change; sweeping tail_alpha no longer changes the average load; workloads with rewrites get shorter contexts |
| `PrefixCache` | Partial matches split before refreshing; session_aware takes the nearest of several hints; new DRAM tier | Long suffixes no longer over-protected |
| `route_and_simulate` | Defaults `insert_on="prefill_done"` (old: finish), `hint_mode="noisy"` (old: oracle), `load_metric="tokens"` (old: request count); `affinity_with_load_cap` now follows SGLang's cache_aware structure (old: exclude instances with load > 1.5·(mean+1)); new `session_sticky` | cache_affinity is no longer "always best" under hotspots; ttl/session_aware gains shrink (no longer oracle) |
| `simulate_rollout` | Default `deploy="colocated"`: async no longer gets free trainer GPUs; async defaults to AReaL interruptible continuation (old: abort + drop); partial trains only on a full batch and `partial_budget_s` is now a per-trajectory per-round budget (old: per-round wall-clock budget); recompute cost counted by default; fixed async counting one extra batch at the last step | async/sync goes from about 5.4× (per trajectory, async with free trainer GPUs) to about 3.8× (per token, equal resources, colocated); async no longer collapses over long runs; partial's mean_batch is always batch_size |
| `credit_error_stats` | New `mean_offset_vs_shapley`, `grad_rel_err`, `grad_cos` (`bias` kept as alias) | None |
| `train_policies` | `final_return` is now the return of the policy after training | Slightly different values |
| `estimate_credit` | New `group_baseline_loo`, `shapley_ref`, `cf_noise`, `rng` | Defaults unchanged |

## 12. Known limitations

- The cost model counts only leading terms; communication is off by default (set `tp` / `ep` to enable it), all-reduce and all-to-all use a simplified bandwidth + latency model, without DeepSeek's "per-node dedup + NVLink forwarding" and without expert load imbalance (`expert_skew` only affects weight reads).
- The engine has no swap or speculative decoding; `kv_reserve="oracle"` uses the true output length.
- Router cache capacity is still independent of the engine's runtime KV; the two-tier cache has only HBM/DRAM, and SSD can be approximated with a lower `dram_gbps`.
- pd is a minimal model: no DP/EP inside the prefill/decode pools and no link contention between transfer and compute.
- rollout abstracts a trajectory as "a span of time" with no token-level batching; recompute cost uses the bulk/single approximations; weight broadcast time is not modeled (fold it into `train_step_s`). Colocated async pauses generation during training and does not correspond to any particular system.
- The credit task is a single-step cooperative game with unary + pairwise structure, and counterfactual queries default to a perfect critic.
- H20 compute figures are reported values treated as dense (inferred); H800 FP8 and HBM take the H100 SXM values.

## 13. Tests

`tests/` has one file per submodule (`test_hardware.py`, `test_model.py`, `test_workload.py`, `test_cache.py`, `test_engine.py`, `test_router.py`, `test_pd.py`, `test_rollout.py`, `test_credit.py`), covering: spec-convention consistency, KV formulas (including MLA), the embedding/lm_head correction, expert-granular MoE bytes, FP8 timing, TP/EP communication and the H800/H100 difference, MLA KV replication under TP, mean calibration and popularity distribution, rewrite compaction and auto-compact, radix hit/split/refresh-only-the-match/eviction/pins/two tiers (including invariant checks under random operations), engine conservation and chunked prefill, KV sharing and preemption, DRAM load delay, router hotspots and the load cap, cache-write timing, non-oracle hints, session stickiness, P/D disaggregation, rollout's equal-resource decomposition, interrupt vs abort length bias, the rate-limit property, no collapse over long runs, the rate-limit window range, partial's fixed batch and tiny-budget boundary, no extra batch at the last step, multi-seed aggregation, and the Shapley closed form (uniform / policy reference), expected-gradient bias, LOO, common random numbers, and the range where "variance becomes learning speed". Source code: [`codesign_sim/engine.py`](../codesign_sim/engine.py) and its sibling files.
