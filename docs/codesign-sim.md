**中文** · [English](codesign-sim.en.md)

# codesign_sim API 参考

> `codesign_sim` 是本课程配套的教学模拟器：用 roofline 成本模型 + 离散事件模拟，在一台 M1 笔记本上几秒内复现推理引擎、前缀缓存、多实例路由、P/D 分离、RL rollout 调度和多 agent 信用分配的**方向性结论与数量级**。只依赖 Python 标准库 + numpy，所有随机性都由 `seed` 固定。本版按两轮独立审查做了一次大修订：凡是"参数一改结论就反转"的地方，要么修了模型，要么在下文写清了适用区间（改动清单见 [§11](#11-相对首版的行为与默认值变化)）。

## 0. 快速上手

```bash
make setup                      # 创建 .venv 并以 editable 方式安装 codesign_sim
.venv/bin/python -m pytest tests -q   # 跑模拟器自身的测试
```

下面这段脚本把主要子模块各用了一次（完整代码可直接粘贴运行）：

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

真实输出（M1，约 2 秒）：

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

怎么读：

- H20 的拐点只有 37 FLOPs/B（算力被砍、带宽反而更高），天生适合 decode；MLA 让 671B 的 DeepSeek-V3 每 token KV 只有 70B GQA 模型的约 1/4.7。
- MoE 按专家粒度读权重：1 个 token 只读 37 GB（FP8），8 个 token 已经 163 GB，64 个 token 584 GB——decode batch 稍大就接近读全部专家，这就是 EP 要把专家摊到很多卡上的原因。
- chunked prefill 把最坏 token 间隔（ITL）从 393 ms 压到 69 ms。
- 热点场景：纯 `cache_affinity` 把 100% 的请求吸到一个实例（其余三个饿死），TTFT p99 3.40 s 反而比 `least_loaded` 还差；SGLang 式的 `affinity_with_load_cap` 在失衡时退回最短队列，p99 降到 0.94 s、命中率反而更高；会话粘性基线 `session_sticky` 也很强。
- P/D 分离：decode 不再被长 prompt 打断，TPOT p99 从 22 ms 降到 8 ms；代价是只有一半的卡做 prefill，TTFT p50 从 0.33 s 升到 0.74 s。
- RL rollout 在**同样 GPU 总数**下：收益几乎全部来自"不等长尾"（tail_gain 3.78×）；"训练与生成重叠"在等资源时几乎不增加吞吐（overlap_gain 0.96×）——旧版里 async 的一部分优势来自白送的训练 GPU。按 token 加权的 staleness 比按轨迹统计的高（长轨迹跨越更多版本）。
- 信用分配：difference 相对 Shapley 有 +0.47 的偏移，但期望梯度完全无偏；以均匀随机队友为背景的 Shapley 几乎没有偏移、方差最低，却有 14.5% 的梯度偏差——"低方差 ≠ 无偏"。

## 1. 设计原则与共同简化

| 原则 | 具体做法 |
|---|---|
| 教学上正确 | 机制与真实系统一致：radix 前缀树只从叶子侧淘汰、continuous batching 按步推进、prefill 完成的那一步吐出首 token、MLA 的 decode 走"吸收权重"路径、AReaL 的可中断续写与限流 |
| 数量级可信 | 硬件与模型参数来自官方或可得的最权威来源（见第 2、3 节；H20 为据报道 + 推断）；步时间用 roofline：`max(GEMM/(峰值[gemm_dtype]·mfu) + 注意力/(峰值[attn_dtype]·mfu), 字节/(带宽·mbu))` + 可选通信，默认 mfu=0.6、mbu=0.8 |
| 结论经得起扫参 | 审查者的扫参/反例脚本覆盖的区间都写进了测试或本文的"适用区间" |
| 代码可读 | 每个文件 150–500 行，函数短，注释中文；公开接口遵循课程约定的契约，库只在契约之外**增加**辅助函数和带默认值的字段/参数 |
| 可复现 | 所有随机数用 `numpy.random.default_rng(seed)`；block hash 只对整数元组做 CPython 内置 `hash`，不受 `PYTHONHASHSEED` 影响。结果在**同一 Python / NumPy 版本**下逐位可复现，跨版本不保证 |

默认仍然忽略的东西：kernel launch 与 CPU 调度开销（`EngineConfig.step_overhead_s` 可以补一个常数）、采样、量化误差、显存碎片（用 `mem_frac=0.9` 粗略预留）。通信现在是**可选项**：`EngineConfig.tp` / `ep` 不设时仍按"理想切分、不计通信"计算（与首版一致）。

## 2. hardware —— GPU 规格、互联常数与 roofline

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

规格表（全部是 **dense** 值，FLOPs 按"一次乘加 = 2 FLOPs"计）。NVIDIA H100 规格表里 BF16 写的是 "1,979 teraFLOPS\*"，表尾脚注 "\* With sparsity"，所以 dense 是 989——不要被"1,979 是 dense"的说法带偏；A100 表同时列出 "312 | 624\*" 两列，可以对照口径：

| 名称 | BF16 TFLOPS | FP8 TFLOPS | HBM | 带宽 | NVLink（双向） | 来源 |
|---|---|---|---|---|---|---|
| A100-80G | 312 | 不支持 | 80 GB | 2.039 TB/s | 600 GB/s | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)（SXM 列） |
| H100-SXM | 989 | 1979 | 80 GB | 3.35 TB/s | 900 GB/s | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)（1,979\* / 3,958\* 为稀疏值；PCIe Gen5 128 GB/s） |
| H800-SXM | 989 | 1979 | 80 GB | 3.35 TB/s | 400 GB/s | NVIDIA 无公开规格页；[DeepSeek arXiv 2505.09343](https://arxiv.org/abs/2505.09343) §4.1：与 H100 同为 Hopper，只削减 FP64 与 NVLink（900→400 GB/s）；[腾讯云实例文档](https://www.tencentcloud.com/document/product/1236/62812)：BF16 989 TFLOPS、80GB、NVLink 400GB/s |
| H20 | 148 | 296 | 96 GB | 4.0 TB/s | 900 GB/s | **据报道 + 推断**，NVIDIA 无公开规格页：[腾讯云开发者社区转载的曝光参数](https://cloud.tencent.com/developer/news/1284500)（FP8 296 / FP16 148 TFLOPS、96GB、4.0TB/s，未说明 dense/sparse，本库按 dense 处理）；[阿里云 gn8v/ebmgn8v 规格族](https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families)写"每卡 96 GB HBM3、4 TB/s、900 GB/s NVLINK"，但**页面没有写明型号**（按规格吻合推断为 H20），且该页的 FP32 39.5 TFLOPS 与媒体报道的 44 TFLOPS 不一致 |
| B200 | 2250 | 4500 | 180 GB | 8.0 TB/s | 1800 GB/s | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/)：HGX B200 八卡 FP8 72 PF、BF16 36 PF，脚注"稀疏值，dense 为一半"；[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)：1,440 GB、64 TB/s（每卡 180 GB、8 TB/s） |

互联常数（都可在 `EngineConfig` / `route_and_simulate` 里覆盖）：每卡 400 Gb/s IB ≈ 50 GB/s、H800 NVLink 单向 200 GB/s 实测约 160 GB/s 可用（可达比例 0.8）、64B 小消息延迟 NVLink 3.33 µs / IB 2.8–3.7 µs——都来自 [DeepSeek ISCA'25](https://arxiv.org/abs/2505.09343) §4.1 与 Table 5；PCIe Gen5 x16 双向 128 GB/s（单向 64 GB/s）来自 H100 规格页。

必须会心算的拐点：H100 BF16 ≈ 295 FLOPs/B，A100 ≈ 153，B200 ≈ 281，H20 ≈ 37。batch=1 的 decode 强度约 1 FLOP/B，远在拐点左边，所以是带宽受限。

## 3. model —— 模型结构与成本模型

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

| 名称 | 层数 | d_model | Q 头 / KV 头 | head_dim | 注意力 | 总参 / 激活 | 词表 | 上下文 | KV 字节/token | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | 32 | 4096 | 32 / 8 | 128 | GQA | 8.03B | 128256 | 128k | 131,072 | [Meta llama-models `sku_list.py`](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)（Llama 3.1 与 Llama 3 结构相同，只差 `use_scaled_rope`）、[Llama 3.1 MODEL_CARD](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)（128k）；参数量取 [HF](https://huggingface.co/meta-llama/Llama-3.1-8B) safetensors 统计 8,030,261,248 |
| llama3.1-70b | 80 | 8192 | 64 / 8 | 128 | GQA | 70.55B | 128256 | 128k | 327,680 | 同上；[HF](https://huggingface.co/meta-llama/Llama-3.1-70B) 统计 70,553,706,496 |
| qwen3-32b | 64 | 5120 | 64 / 8 | 128 | GQA | 32.8B | 151936 | 32k 原生 / 128k（YaRN） | 262,144 | [Qwen3-32B config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) 与 model card |
| deepseek-v3 | 61 | 7168 | 128 / 128 | 128（+64 RoPE） | MLA（kv_lora_rank 512） | 671B / 37B | 129280 | 128k | 70,272 | [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)（256 路由专家、每 token 8 个、moe_intermediate_size 2048、前 3 层 dense、1 个共享专家）与 model card |

公式：

- **KV**：GQA/MHA 为 `2 · L · n_kv_heads · head_dim · bytes`；MLA 只缓存 latent 与解耦 RoPE key，为 `L · (kv_lora_rank + qk_rope_head_dim) · bytes`。
- **GEMM**：`2 · N_body` FLOPs/token，`N_body` = 激活参数去掉 embedding 表与 lm_head；lm_head 只在需要 logits 的位置算（prefill 默认每条序列 1 个，decode 每个 token 都算）；embedding 查表不算 FLOPs。llama3.1-8b 的 embedding+lm_head 占参数 13%，旧版把它们当 GEMM，会让 prefill FLOPs 高估约 13%。
- **注意力**：`c_attn · (n_new · n_cached + n_new(n_new+1)/2)`，`c_attn = 2 · L · H · (d_qk + d_v)`：GQA 是 `4·L·H·head_dim`；MLA prefill 用展开后的 `d_qk = 128+64, d_v = 128`，MLA decode 用吸收权重后的 `d_qk = 512+64, d_v = 512`。分块 prefill 的注意力 FLOPs 与一次性 prefill 完全相同（`logits_positions=0` 时 GEMM 也完全相同）。
- **读权重**：输入 embedding 只读 batch 行；lm_head 整张读；dense 其余参数整份读。**MoE 按专家粒度**：非路由参数（注意力、共享专家、前 3 层 dense FFN、embedding、gate，V3 约 17.1B）每步必读，每个 MoE 层（V3 有 58 层）读"被选中的不同专家"，均匀路由时期望为 `E·(1-(1-k/E)^n)`，`expert_skew > 0` 模拟热点专家。旧版"参数独立被选中"的近似在 n=4–32 时高估 30–50%。
- **FP8**：`step_time_s` 的 `flops` 按 `dtype` 计时、`attn_flops` 按 `attn_dtype` 计时。引擎里 GEMM 默认跟随权重精度（`weight_bytes=1` 即 FP8 GEMM），注意力核心默认 BF16——与 [DeepSeek 推理系统说明](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)"矩阵乘与 dispatch 用 FP8、MLA 核心计算与 combine 用 BF16"一致。
- **通信（可选）**：TP 每层 2 次 all-reduce，`2(p-1)/p · n_tokens·d_model·2B / (NVLink 单向 · 0.8) + 2α`（默认按 NVSwitch 上的 two-shot 算法：reduce-scatter 与 all-gather 各一步，延迟项 2α；`allreduce_time_s(..., algo="ring")` 可改为纯 ring，延迟项 2(p-1)α）；EP 每个 MoE 层一次 dispatch（FP8，1 B）+ combine（BF16，2 B），每卡 `n_tokens/ep · k · d_model` 个元素、`(ep-1)/ep` 发往别的卡，其中 `(ep-8)/(ep-1)` 跨节点走 IB，两条链路取较慢者。`step_time_s` 里 `comm_overlap` 比例与计算/访存重叠（取 max），其余串行相加。H800 与 H100 的差别只在这里体现：TP8 all-reduce 的带宽项慢 2.25 倍。
- **MLA + TP**：latent KV 所有头共享、无法按头切，TP 下每个 rank 都读完整 KV（`decode_bytes(..., kv_replicas=tp)`）；这就是 DeepSeek/SGLang 对 MLA 用 DP-attention 的原因。

## 4. workload —— agent 多轮负载

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

第 t 轮 prompt = 共享前缀 + 前 t 轮的 (new_prompt + output) + 第 t 轮 new_prompt（append-only）。生成分布（教学用的合理量级，**不是**某个产品的统计）：

| 量 | 分布 |
|---|---|
| 共享前缀（system prompt + 工具定义 + 仓库上下文） | 每个 project 一个，lognormal 中位数 6k token，截断 [2k, 20k] |
| 第 0 轮用户消息 / 之后每轮工具结果 | lognormal 中位数 300 / 1k token（工具结果截断 [50, 16k]） |
| 每轮输出 | Pareto(形状 `tail_alpha`)，截断 4096；**截断后**均值校准为 300 |
| 轮数 | Pareto(形状 `tail_alpha`)，截断 64；**截断后**均值校准为 `mean_turns` |
| 工具延迟 | lognormal 中位数 2 s，截断 120 s；最后一轮为 0 |
| project 热度 | Zipf，权重 ∝ 1/(k+1)^`project_zipf_s`（默认 1 与首版相同；0 为均匀） |
| 到达 | Poisson(`arrival_rate` 个 session/s) |
| rewrite | 第 1 轮起每轮以 `rewrite_prob` 概率为 True；此外上下文将超过 `auto_compact_tokens` 时自动压缩一次 |

- 均值校准：旧版截断后均值会随 `tail_alpha` 变（α=1.2 时平均只有 4.6 轮、194 token），扫 α 时尾部和平均负载混在一起；现在反解 Pareto 的尺度参数，α 只改尾部。实测（1000 个 session）：α=1.2/1.5/2/3 时平均轮数 7.9/8.0/8.1/8.0，平均输出 298/305/300/301。
- rewrite = 改写**并压缩**：共享前缀之后的私有历史变成一段新内容，长度 = 原长度 × `keep_frac`（默认 0.5），所以缓存失效的同时 prefill/KV 也变少了——可以用来算"压缩值不值"的盈亏平衡；`rewrite_keep_frac=1.0` 表示只改内容不改长度（例如在前缀里插入时间戳这类反模式）。
- auto-compact：模拟 harness 在接近上下文上限时自动压缩，保证上下文不超过模型的 128k；`auto_compact_tokens=None` 关闭。
- prompt:输出 token 比随 seed 与 `tail_alpha` 变化，200 个 session、seed 0–4 实测 71–118:1，α=3 时约 51:1——agent 负载是 prefill 主导的。

`to_blocks` 用**链式 hash**近似 block 身份（和 vLLM prefix caching 一样）：**在忽略 hash 碰撞的前提下**，hash 相同 ⇔ 从开头到这个 block 内容完全相同。只返回满 block；同 project 的 session 前 `shared_prefix_tokens // block_size` 个 block 相同；没有 rewrite 时下一轮以上一轮 prompt+输出为前缀；rewrite 后共享前缀之后的所有 hash 都变。

## 5. cache —— 按 block 的 radix 前缀缓存（可选两层）

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

- 结构：压缩 radix tree，与 SGLang RadixAttention 相同；插入时按需分裂节点，共享前缀只存一份。**match 停在一条边中间时先分裂**，只刷新命中的前半段，没用到的后缀保持原来的访问时间（首版会刷新整条边，过度保护长后缀）。
- 淘汰只从叶子一侧进行，并且只从尾部裁掉刚好需要的 block 数。
- `lru`：忽略 `pin_until`。`ttl`：`pin_until` 之前**硬保留**，全被 pin 住时新插入被截断（`rejected_blocks`）；同一节点被多次 pin 时取**最晚**的。`session_aware`：`pin_until` 是"预计下次使用时间"（软提示），先 LRU 淘汰没有提示或提示过期的，再淘汰预计回来最晚的；同一节点有多个未来使用者时取**最近**的一次。
- 两层（`dram_blocks > 0`）：HBM 满了把"HBM 边界节点"（没有 HBM 孩子的 HBM 节点）的尾部**降级**到 DRAM；DRAM 满了才真正丢弃。路径上 HBM 在前、DRAM 在后。insert 经过 DRAM 节点时把它**提升**回 HBM。DRAM 命中的搬运成本由调用方计（router 按 PCIe 带宽计费，比重算还慢就当作未命中）。

## 6. engine —— 单实例 continuous batching + chunked prefill

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

每一步：

1. **准入**：按到达顺序放入运行集合（`load_s > 0` 的请求在 KV 从 DRAM 搬回之前不参与调度，也不阻塞后面已就绪的请求），直到 KV 容量或 `max_num_seqs` 用满。KV 容量默认 = `(n_gpus·HBM·mem_frac - 权重) / KV字节`；权重放不下直接抛 `ValueError`。
   - `kv_reserve="oracle"`（默认）：准入时按 `(prompt - 命中前缀) + output` 一次性预留。**这用到了真实输出长度，是 oracle**——真实调度器拿不到；好处是永远不需要抢占。
   - `kv_reserve="incremental"`（vLLM 式）：只按 prompt 分配，decode 每步按需增长；显存不够时抢占最后准入的序列（recompute：释放 KV、放回队头、连同已生成 token 重新 prefill）。长输出的 RL 场景用它才能看到抢占代价（`preemptions`、`recomputed_tokens`）。
   - 命中前缀缓存的 token 与缓存**物理共享**，不重复计入本请求的 KV。
2. **组 batch**：`chunked_prefill=True` 时 decode 优先、剩余预算按 FIFO 切给 prefill 块；`False` 时只要有待 prefill 的请求就只做整条 prompt 的 prefill，decode 全部暂停。
3. **步时间**：GEMM 按 `gemm_dtype`（None 时跟随 `weight_bytes`：≤1 字节即 FP8）的峰值、注意力按 `attn_dtype` 的峰值；字节 = 触达权重（MoE 按专家粒度）+ 读 KV（MLA 在 `tp` 设置且未开 `dp_attention` 时按 tp 份复制）+ 写新 KV；`tp` 设置时加 TP all-reduce，`ep > 1` 时加 EP dispatch/combine，按 `comm_overlap` 与计算重叠。
4. prefill 完成的那一步末尾吐出首 token；之后每个 decode 步吐 1 个 token。`cached_tokens` 最多算 `prompt-1`。

返回值：`ttft_p50/p99`、`tpot_p50/p99`、`throughput_tok_s`（输出 token 吞吐）、`busy_frac`、`per_request`（含 `preemptions`），另附 `itl_p50/itl_p99/itl_max`、`makespan_s`、`n_steps`、`kv_capacity_tokens`、`prefill_tokens_computed`、`output_tokens`、`preemptions`、`recomputed_tokens`。上下文超过 `model.max_context` 时发一次 `UserWarning`（模拟照常进行）。

## 7. router —— 多实例路由（闭环 agent 负载）

```python
def route_and_simulate(sessions, n_instances, policy, cfg, cache_capacity_blocks, seed=0, *,
                       block_size=16, cache_policy="lru", load_cap_factor=1.5, cache_threshold=0.3,
                       balance_abs_threshold=None, load_metric="tokens", est_output_tokens=300.0,
                       count_tool_waiting=False, est_return_tokens=1300.0, insert_on="prefill_done",
                       hint_mode="noisy", hint_ttl_s=10.0, hint_noise_sigma=1.0,
                       cache_dram_blocks=0, dram_gbps=None, sticky_migrate=False) -> dict
```

- 闭环：第 t 轮完成 → 等 `tool_latency_s` → 第 t+1 轮到达。每个实例 = 一个 `Engine`（有真实排队：KV 容量、token 预算、`max_num_seqs`）+ 一个自己的 `PrefixCache`。
- **写缓存时机** `insert_on`：`"prefill_done"`（默认，prefill 完成即写入 prompt 的 block，之后到达的同前缀请求就能命中，对应 SGLang RadixCache 对未完成请求的处理）；`"route"`（路由时立刻写，SGLang router 近似树的做法，最乐观）；`"finish"`（首版行为：完成才写——会掩盖热点，见下表）。三种方式在完成时都会再写 prompt+输出。
- **pin 提示** `hint_mode`（给 `ttl` / `session_aware` 用）：`"noisy"`（默认：完成时刻 + 真实工具延迟 × exp(σ·N(0,1))）、`"fixed"`（固定 TTL `hint_ttl_s`）、`"oracle"`（真实工具延迟——等于提前知道下一轮何时到达，是**上界**，必须显式选择）、`"none"`。首版默认就是 oracle。
- **负载** `load_metric`：`"tokens"`（默认：剩余 prefill + 预计剩余 decode，输出长度未知，按 `est_output_tokens` 估计）或 `"requests"`（在途请求数，SGLang 的口径）；`count_tool_waiting=True` 再计入"正在等工具、马上会回来"的 session（每个按 `est_return_tokens`）。
- **策略**：
  - `round_robin`：按请求轮询。
  - `least_loaded`：负载最低。
  - `cache_affinity`：只看命中，选前缀命中最长的（平局选负载低）——**极端对照**，会产生热点、饿死冷实例。
  - `affinity_with_load_cap`：[SGLang router cache_aware 策略](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)的结构：若 `(max 负载 - min 负载) > balance_abs_threshold` 且 `max > load_cap_factor · min`，判定失衡，改选负载最低的；否则命中率 > `cache_threshold` 选命中最长的，不到阈值选负载最低的。SGLang 网关的 CLI 默认值是 cache_threshold 0.3、balance_abs_threshold 64、balance_rel_threshold 1.5（[main.rs](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/main.rs)）；64 个请求是生产规模，本库 `balance_abs_threshold=None` 时按小规模缩放为 4 个请求（`load_metric="requests"`）或 4 × `max_batch_tokens`（`"tokens"`）。想复现 SGLang 原值：`load_metric="requests", balance_abs_threshold=64, cache_threshold=0.3, load_cap_factor=1.5`。`load_cap_factor` 必须 > 0。
  - `session_sticky`：会话第一轮按负载最低选实例，之后固定回同一实例（会话粘性基线）；`sticky_migrate=True` 时粘住的实例失衡（判据同上）就迁移到最空闲实例。
- **两层缓存**：`cache_dram_blocks > 0` 时启用 DRAM 层；DRAM 命中的搬运时间 = block 数 × block_size × KV字节 / `dram_gbps`（默认 n_gpus × 64 GB/s，即每卡一条 PCIe Gen5 x16 单向），与重算这些 token 的 prefill 时间比较，取较小者（`dram_loaded_blocks` / `dram_recomputed_blocks`）。
- 返回：`hit_rate`（HBM+DRAM）、`hbm_hit_rate`、`dram_hit_rate`、`token_hit_rate`、`per_instance_hit_rate`、`load_imbalance`（max/mean 实例忙碌时间）、`requests_per_instance`、`max_instance_share`（最热实例的请求占比）、`busy_s_per_instance`、TTFT/TPOT/ITL 分位数、`throughput_tok_s`、`prefill_tokens_total/computed`、`evictions`、`dram_loaded_blocks`、`dram_recomputed_blocks`、`migrations`、`per_request`（含 `instance`）。

写缓存时机决定了热点能不能显现（1 个 project、100 个 session、3 session/s、4 × H100 + llama3.1-8b，seed 0，实测）：

| 策略 | insert_on | 最热实例占比 | 负载不均衡 | TTFT p99 | 命中率 |
|---|---|---|---|---|---|
| cache_affinity | finish（首版） | 0.41 | 1.81 | 0.67 s | 0.91 |
| cache_affinity | prefill_done | 1.00 | 4.00 | 3.40 s | 0.80 |
| affinity_with_load_cap | prefill_done | 0.71 | 1.86 | 0.94 s | 0.89 |
| session_sticky | prefill_done | 0.30 | 1.37 | 1.05 s | 0.91 |

只在完成时写缓存，早期同 project 的 session 在路由时打平、被分散出去，热点形成不起来；prefill 完成即写入后，纯亲和把所有请求吸到一个实例，尾延迟反而最差，SGLang 式的失衡回退把它救回来。热点只在"最热实例被压满"时才显现——低负载、多 project 时纯亲和仍然是 TTFT 最好的策略之一。

## 8. pd —— P/D 分离（新增）

```python
from codesign_sim.pd import simulate_colocated, simulate_pd
def simulate_colocated(requests, cfg, n_instances=1, *, ttft_slo_s=None, tpot_slo_s=None) -> dict   # 共置对照
def simulate_pd(requests, prefill_cfg, decode_cfg=None, n_prefill=1, n_decode=1, *,
                kv_link_gbps=None, layerwise=True, ttft_slo_s=None, tpot_slo_s=None) -> dict
```

- 请求先进 prefill 池，prefill 完成吐出首 token（TTFT 在这里确定）；整段 prompt 的 KV 通过 RDMA 传到 decode 池：传输时间 = `prompt · KV字节 / (链路带宽 · 0.8)`，链路默认 = min(两侧卡数) × 50 GB/s。`layerwise=True` 时按层流水，只有约 1/n_layers 的传输暴露在关键路径上（传输总时长超过 prefill 计算时间时，超出部分也暴露）。decode 池里这个请求以"命中了整段 prompt"的形式提交，不再 prefill；但这段 prompt KV 归它独占（`Request.own_cached_kv=True`），按 prompt + 已生成 token 计入 decode 实例的 KV 容量（与共置对照同一口径），容量不足时在 decode 侧排队，`kv_reserve="incremental"` 时按引擎的抢占模式处理（被抢占的请求在 decode 实例上整段重算）。因此不再需要用 `max_num_seqs` 手动限制 decode 并发。
- 返回：TTFT/TPOT/ITL 分位数（第 1→2 个 token 的间隔包含传输与 decode 侧排队）、`throughput_tok_s`、`prefill_busy_frac`、`decode_busy_frac`、`kv_transfer_s_mean`、`kv_transfer_exposed_s_mean`、`n_gpus_total`、`decode_kv_peak_frac`（decode 侧 KV 峰值 / 容量）、`decode_preemptions`，以及按 `ttft_slo_s` / `tpot_slo_s` 计算的 `goodput_req_s`（同时满足两个 SLO 的请求数 / 秒）与 `slo_attainment`。`simulate_colocated` 同样返回 goodput 与 `kv_peak_frac`。
- decode 侧 KV 是真约束：同样 16 张 H100、llama3.1-70b、200 个请求（TTFT ≤ 2 s、TPOT ≤ 30 ms），decode 实例 KV 容量不设限时 goodput 2.11 请求/s；压到 4 万 token 时 KV 峰值打满（0.999）、TPOT p99 升到 347 ms、goodput 降到 0.36 请求/s（测试有断言）。
- 取舍（§0 实测）：同样 16 张 H100、llama3.1-70b，P/D 1+1 把 TPOT p99 从 22 ms 降到 8 ms，TTFT p50 从 0.33 s 升到 0.74 s。P:D 配比要按负载调；MLA 每 token KV 只有 69 KB，传输代价低，这是 DeepSeek 做 P/D 分离的底气之一。

## 9. rollout —— RL rollout：sync / partial / async（等资源比较）

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

- **资源**：集群 = `n_workers` 个单位。`deploy="colocated"`（默认）时所有单位轮流做 rollout 和训练，训练期间生成暂停；`"disaggregated"` 时 `trainer_frac` 的单位专门训练（一步耗时 `train_step_s / trainer_frac`），其余 `round(n_workers·(1-trainer_frac))` 个槽位专门 rollout。两种部署的 GPU·秒总量相同，所以可以直接比较。
- **轨迹**：生成时间 Pareto（形状 `tail_alpha`，截断 50×均值，**截断后**均值校准为 `mean_gen_s`）+ 工具等待（指数，均值 `mean_tool_s`）；生成 token 数 = 生成秒数 × `decode_tok_s`。
- **sync**：每步生成 `batch_size` 条，等最慢的一条再训练（= async 且 η=0）。
- **partial**（APRIL / Kimi k1.5 思路）：过量发射，凑满 `batch_size` 条完成的轨迹就结束本轮，在途的暂停、下一轮用新策略续写；`partial_budget_s`（或 `partial_budget_tokens`）是每条轨迹**每轮**最多推进的量，超出的也暂停到下一轮。**训练 batch 始终是 batch_size**（首版在 60 s 墙钟预算到时就用不满的 batch 训练，只有 153–197 条）。生成与训练不重叠。
- **async**（[AReaL](https://arxiv.org/abs/2505.24298)）：rollout 连续进行，trainer 凑满 batch 就训练，优先用旧轨迹。提交新轨迹时按 AReaL 式 (3) 限流：`⌊(N_r − 1)/B⌋ ≤ i + η`（N_r = 已提交且未被丢弃的轨迹数，i = 当前版本）。
  - `stale_policy="interrupt"`（默认，AReaL 语义）：权重更新时打断所有在途生成、丢旧 KV、用新权重重算后继续解码，一条轨迹由多个版本的片段组成，不丢数据。限流约束的是**提交端**；单条长轨迹从开始到被训练可以跨越很多版本（`staleness_max` 可以 > η）。
  - `stale_policy="abort"`（对照）：在途轨迹版本差 > η 就中止，训练时丢弃过旧样本（并释放限流名额——首版不释放，名额随步数单调缩小，长运行下吞吐塌陷到接近 sync）。硬上界 `staleness_max ≤ η` 的代价是**系统性丢掉长轨迹**（`length_ratio` 明显 < 1）。
- **重算成本**（`resume_kv="recompute"`，默认）：续写/被打断的轨迹要把上下文（`prompt_tokens` + 已生成 token）重新 prefill。批量场景（权重更新时全部打断、新一轮成批续写）整个 rollout 集群一起算，大家等同样久 = `Σ prefill_flops(ctx_i) / (槽位数 · gpus_per_worker · 峰值 · mfu)`；单条续写按一个引擎（`engine_gpus` 张卡）的速度。重算占用槽位、计入 GPU 忙碌但不产出 token（`recompute_frac`）。`"retain"` 表示沿用旧 KV、不计成本。
- **staleness**：按轨迹 = 训练版本 − 最旧片段版本；按 token 加权 = Σ 片段 token ×（训练版本 − 片段版本）/ Σ token（PPO/GRPO 的 loss 是按 token 加权的，长轨迹恰恰跨越最多版本）。
- 返回：契约的 `worker_busy_frac`（只计"在生成"的槽位时间）、`traj_per_hour`、`mean_staleness`、`p99_staleness`、`step_time_s`，另附 `train_tokens_per_hour`（与轨迹长度无关的主指标）、`staleness_max`、`token_staleness_mean/max`、`gen_mean_started/trained`、`length_ratio`、`cluster_util`（(生成 + 重算 + 训练) GPU·秒 / 总 GPU·秒）、`gate_idle_frac`（槽位因限流/等长尾空闲的比例）、`recompute_frac`、`interrupts`、`max_gate_lead`、`total_time_s`、`trained_trajectories`、`mean_batch`、`wasted_frac`、`dropped_trajectories`、`worker_occupied_frac`、`n_slots`。

**收益拆解**（§0 输出，默认参数、同样 64 个单位）：sync 共置 3.5 M token/h → async 共置 13.2（`tail_gain` 3.78×，来自"不等长尾"）→ async 分离 12.6（`overlap_gain` 0.96×）。等资源时"训练与生成重叠"几乎不增加吞吐：共置已经让所有 GPU 在训练时也在干活；分离部署的真实好处（各自选最优并行策略、免去切换与重分片、不必暂停后重算）大部分不在这个模型里。另外，共置下的 async 在限流不起作用时与 partial 几乎等价。

**适用区间**：限流窗口是 (η+1)·B 条轨迹。n_workers 接近或超过 (η+1)·B 时（真实系统里一个 batch 通常并发跑完，n_workers ≥ B 很常见），大部分槽位会被限流挡住。分离部署、50 步的实测：

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

n_workers ≥ B 时 async 的 `gate_idle_frac` 高达 0.7–0.9：一部分是限流（调大 η 吞吐从 14.0 升到 15.1，代价是 token staleness 从 1.3 升到 3.8），另一部分是分离部署下的 trainer 成了瓶颈（每步 120 s）。经验规则：让 (η+1)·B ≥ n_workers，并按生成/训练的工作量配 trainer_frac。首版在这个区间看到的"async 不如 partial"来自中止+丢弃与白送的训练 GPU；按 AReaL 语义、等资源比较后，async 仍略优，但槽位大量闲置。

## 10. credit —— 多 agent 信用分配

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

- 奖励：`r = Σ_i W[i,a_i] + Σ_{i<j} Q_ij(a_i,a_j) + ε`。**真实个体贡献**定义为 f 的 Shapley 值（缺席 agent 按均匀随机动作取期望），闭式解已与暴力枚举核对。
- 估计器（`method`）：`shared`、`group_baseline`（减含自身的 batch 均值，期望梯度缩放 (B−1)/B、方向不变）、`group_baseline_loo`（leave-one-out，严格无偏）、`difference`（r − f(把自己换成默认动作 0)）、`counterfactual_baseline`（COMA 式，r − Σ_b π_i(b)·f(把自己换成 b)）、`shapley`（精确 Shapley + 噪声平摊）。`shapley_ref="uniform"`（默认）以均匀随机队友为背景，方差最低但不是当前策略的梯度；`shapley_ref="policy"` 以当前策略为背景，对独立策略的 REINFORCE 严格无偏（测试里用精确期望梯度验证）。
- 反事实查询的噪声 `cf_noise`（影响 difference / counterfactual_baseline）：`"none"`（默认，完美 critic）、`"independent"`（每次查询一次独立带噪仿真）、`"common"`（共同随机数：反事实与实际 rollout 共享外生噪声，差分后噪声完全抵消）。单样本梯度方差：common < none < independent。
- `credit_error_stats`：`mean_offset_vs_shapley`（估计值减 Shapley 真值的平均偏移；旧字段名 `bias` 保留为别名——它**不是**策略梯度偏差，常数偏移不改变期望梯度）、`error_var`、`corr`，以及在一个非均匀策略（默认每个 agent 一个 Dirichlet(1) 随机策略）处的 `grad_rel_err` / `grad_cos`。在均匀策略处 `shapley(uniform)` 恰好无偏，看不出问题。
- `train_policies`：`returns[t]` 是第 t 次更新**之前**的策略的精确期望回报；`final_return` 是全部更新**之后**的（首版是最后一次更新之前）。

**方差差异什么时候变成学习速度差异**：agent 少、batch 大时各估计器的学习曲线几乎重合；agent 多、batch 小时，shared 的每 agent 信噪比约为 1/n_agents，才明显落后。300 轮后的最终期望回报（3 个 seed 平均，4 个动作）：

```text
n= 8 B= 8 shared=   7.4 group_base=   7.6 difference=   7.4 counterfac=   7.4 shapley=   7.4
n= 8 B=64 shared=   7.5 group_base=   7.4 difference=   7.4 counterfac=   7.4 shapley=   7.4
n=32 B= 8 shared=  35.3 group_base=  61.8 difference=  62.1 counterfac=  63.4 shapley=  63.6
n=32 B=64 shared=  58.4 group_base=  62.7 difference=  63.4 counterfac=  63.3 shapley=  63.3
n=64 B= 8 shared=  56.0 group_base= 137.6 difference= 146.9 counterfac= 146.8 shapley= 145.8
n=64 B=64 shared= 102.6 group_base= 147.7 difference= 147.5 counterfac= 147.3 shapley= 147.0
```

（`group_base` 为 `group_baseline_loo`，`shapley` 为 `shapley_ref="policy"`。）结论：想在实验里看到信用分配的价值，用 n_agents ≥ 32、batch ≤ 16。

## 11. 相对首版的行为与默认值变化

| 函数 / 字段 | 变化 | 对旧结论的影响 |
|---|---|---|
| `MODELS` | 改名 `llama3.1-8b` / `llama3.1-70b`（旧键为别名，`cfg.name` 变成新名字）；新增 vocab / max_context / MoE 字段 | 结构参数不变 |
| `prefill_flops` / `decode_flops` | 不再把 embedding / lm_head 当 GEMM；prefill 只算 1 个位置的 lm_head | llama3.1-8b 的 prefill FLOPs 约 −13%，70B 约 −3%，V3 约 −5% |
| `weight_bytes_touched` / `decode_bytes` | 输入 embedding 只读 batch 行；MoE 改为专家粒度 | 8B 单请求 TPOT 约 −6.5%；V3 小/中 batch 的 decode 字节下降 30–50% |
| `step_time_s` | 新增 `attn_flops/attn_dtype/comm_s/comm_overlap`（默认不影响旧调用） | 无 |
| `EngineConfig` | 新增 gemm_dtype（**默认跟随 weight_bytes**）、attn_dtype、kv_reserve、tp、ep、dp_attention 等 | `weight_bytes=1` 的配置现在按 FP8 GEMM 计时，compute-bound 的 prefill 快约 2 倍 |
| `simulate` / `Engine` | 命中前缀的 KV 与缓存共享，不再重复计入（`Request.own_cached_kv=True` 的除外） | KV 紧张时并发更高 |
| `simulate_pd` | decode 侧按 prompt + 已生成 token 计 KV，容量不足时排队/抢占；新增 SLO goodput、`decode_kv_peak_frac` | 首版修订里 decode 侧没有 KV 约束（需要 `max_num_seqs` 绕开）；现在 decode KV 紧张时 PD 的 goodput 会明显下降 |
| `generate_sessions` | 截断后均值校准；rewrite 压缩上下文（`rewrite_keep_frac` 默认 0.5）；`auto_compact_tokens` 默认 100k；新增 `project_zipf_s`（默认 1 = 旧行为） | 同 seed 生成的 session 数值全部改变；扫 tail_alpha 时平均负载不再变化；有 rewrite 的负载上下文变短 |
| `PrefixCache` | 部分命中先分裂再刷新；session_aware 多个提示取最近；新增 DRAM 层 | 长后缀不再被过度保护 |
| `route_and_simulate` | 默认 `insert_on="prefill_done"`（旧：finish）、`hint_mode="noisy"`（旧：oracle）、`load_metric="tokens"`（旧：请求数）；`affinity_with_load_cap` 改为 SGLang cache_aware 结构（旧：排除负载 > 1.5·(均值+1) 的实例）；新增 `session_sticky` | 热点场景下 cache_affinity 不再"恒优"；ttl/session_aware 的收益变小（不再是 oracle） |
| `simulate_rollout` | 默认 `deploy="colocated"`：async 不再白送训练 GPU；async 默认 AReaL 可中断续写（旧：中止 + 丢弃）；partial 改为凑满 batch 才训练，`partial_budget_s` 改为每条轨迹每轮的推进预算（旧：整轮墙钟预算）；默认计入重算成本；修掉 async 最后一步多计一批 | async/sync 从约 5.4×（按条计、async 白送训练 GPU）变为约 3.8×（按 token 计、等资源、共置）；async 长运行不再塌陷；partial 的 mean_batch 恒为 batch_size |
| `credit_error_stats` | 新增 `mean_offset_vs_shapley`、`grad_rel_err`、`grad_cos`（`bias` 保留为别名） | 无 |
| `train_policies` | `final_return` 改为训练结束后的策略回报 | 数值略有变化 |
| `estimate_credit` | 新增 `group_baseline_loo`、`shapley_ref`、`cf_noise`、`rng` | 默认不变 |

## 12. 已知局限

- 成本模型只算主项；通信默认不计（设 `tp` / `ep` 才计），all-reduce 与 all-to-all 用简化的带宽 + 延迟模型，不做 DeepSeek 的"同节点去重 + NVLink 转发"，也不建模专家负载不均（`expert_skew` 只影响读权重）。
- engine 没有 swap、投机解码；`kv_reserve="oracle"` 用到了真实输出长度。
- router 的缓存容量仍独立于引擎运行时的 KV；两层缓存只有 HBM/DRAM，SSD 可以用更低的 `dram_gbps` 近似。
- pd 只做了最小模型：没有 prefill/decode 池内部的 DP/EP，也没有传输与计算争用链路。
- rollout 把一条轨迹抽象成"一段时间"，没有 token 级批处理；重算成本用批量/单条两种近似；不建模权重广播时间（可并入 `train_step_s`）。共置 async 在训练时暂停生成，并不等于某个具体系统。
- credit 任务是单步、一元 + 二元结构的合作博弈，反事实查询默认是完美 critic。
- H20 的算力是据报道值并按 dense 处理（推断）；H800 的 FP8 与 HBM 取 H100 SXM 的值。

## 13. 测试

`tests/` 下每个子模块一个文件（`test_hardware.py`、`test_model.py`、`test_workload.py`、`test_cache.py`、`test_engine.py`、`test_router.py`、`test_pd.py`、`test_rollout.py`、`test_credit.py`），覆盖：规格口径一致性、KV 公式（含 MLA）、embedding/lm_head 修正、MoE 专家粒度字节、FP8 计时、TP/EP 通信与 H800/H100 差异、MLA 在 TP 下的 KV 复制、均值校准与热度分布、rewrite 压缩与 auto-compact、radix 命中/分裂/只刷新命中部分/淘汰/pin/两层存储（含随机操作的不变式检查）、engine 守恒与 chunked prefill、KV 共享与抢占、DRAM 搬运延迟、router 热点与负载上限、写缓存时机、非 oracle 提示、会话粘性、P/D 分离、rollout 的等资源拆解、可中断续写 vs 中止的长度偏置、限流性质、长运行不塌陷、限流窗口区间、partial 固定 batch 与极小预算边界、最后一步不多计、多 seed 汇总，以及 Shapley（均匀/策略参考）闭式解、期望梯度偏差、LOO、共同随机数和"方差何时变成学习速度"的区间。源码见 [`codesign_sim/engine.py`](../codesign_sim/engine.py) 等同目录文件。
