[中文](numbers-cheatsheet.md) · **English**

# Numbers You Must Know by Heart

> In interviews and design reviews, stating the right order of magnitude on the spot matters more than reciting exact values. Every number here has a source; "derived" means computed from a formula (shown alongside), and "reported" means there is no official vendor spec page.
>
> Units: TFLOPS = 1e12 FLOP/s; TB/s = 1e12 B/s; GB = 1e9 B; GiB = 2³⁰ B; KiB = 1,024 B. All GPU compute numbers are **dense** (no 2:4 sparsity). The Tensor Core numbers on NVIDIA's H100 / B200 spec pages include sparsity and must be divided by 2. **Vendor interconnect numbers are mostly bidirectional totals; use one-way numbers when estimating "how long one transfer takes"** (Section 4).

## 1. GPU specs

| GPU | BF16 dense TFLOPS | FP8 dense TFLOPS | HBM | HBM bandwidth | Ridge point (BF16, FLOPs/B) | NVLink per GPU (bidirectional total / one way) | Source |
|---|---|---|---|---|---|---|---|
| A100 80GB SXM | 312 | not supported | 80 GB HBM2e | 2.039 TB/s | ≈ 153 | 600 / 300 GB/s | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| H100 SXM | 989 | 1,979 | 80 GB HBM3 | 3.35 TB/s | ≈ 295 | 900 / 450 GB/s | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) (the page's 1,979 / 3,958 are sparse values) |
| H800 SXM | 989 | 1,979 | 80 GB | 3.35 TB/s | ≈ 295 | 400 / 200 GB/s | NVLink cut from 900 to 400 GB/s, 200 GB/s one way, per [arXiv:2505.09343](https://arxiv.org/abs/2505.09343); compute and HBM taken as H100 (no public English NVIDIA spec page) |
| H20 | 148 | 296 | 96 GB HBM3 | 4.0 TB/s | ≈ 37 | 900 / 450 GB/s | **Reported**, no public NVIDIA spec page. All five figures come from [a 芯榜 report reposted on the Tencent Cloud developer community (2024-01-02)](https://cloud.tencent.com/developer/news/1284500), with compute treated as dense (the report does not say); [Alibaba Cloud's gn8v instance family doc](https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families) lists 96 GB HBM3, 4 TB/s, and 900 GB/s NVLink but does not name the GPU, so matching it to H20 is an inference |
| B200 | 2,250 | 4,500 | 180 GB HBM3e | 8 TB/s | ≈ 281 | 1,800 / 900 GB/s | [HGX B200](https://www.nvidia.com/en-us/data-center/hgx/) (8-GPU BF16 36 PF sparse → 2.25 PF dense per GPU), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) (1,440 GB and 64 TB/s across 8 GPUs) |

Mental-math notes:

- **The BF16 ridge of Hopper / Blackwell ≈ 300 FLOPs/Byte**; FP8 doubles the compute and so doubles the ridge (H100 FP8 ≈ 591). What doubles is the Tensor Core matmul peak; attention's KV reads, dequantization, and so on do not double with it.
- Ridge = peak ÷ bandwidth. Decode intensity is usually single digits to a few dozen, so **bandwidth is the currency of decode**.
- H20's ridge is only about 37 (derived from reported specs): weak compute, strong bandwidth; good for decode, poor for prefill. Even an incremental prefill of a few dozen new tokens may still be compute-bound on H20.

## 2. Model structure and KV cache

| Model | Layers | Query heads / KV heads | head_dim | Params (total / active) | KV bytes/token (BF16) | KV of a 32k sequence | Source |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 32 / 8 | 128 | 8.03B | 131,072 B = 128 KiB | 4 GiB | [Llama 3 paper, Table 3](https://arxiv.org/abs/2407.21783) |
| Llama-3-70B | 80 | 64 / 8 | 128 | 70.6B | 327,680 B = 320 KiB | 10 GiB | Same |
| Qwen3-32B | 64 | 64 / 8 | 128 | 32.8B | 262,144 B = 256 KiB | 8 GiB | [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3-32B) |
| DeepSeek-V3 | 61 | 128 heads, MLA with one latent shared by all heads | QK 192 (128 + 64) / V 128; after weight absorption decode works in 576 / 512 dims | 671B / 37B (256 routed experts, 8 chosen per token) | 70,272 B ≈ 68.6 KiB | 2.14 GiB | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |

- Formulas: GQA / MHA is 2 × layers × KV heads × head_dim × bytes; MLA is layers × (kv_lora_rank + rope dim) × bytes.
- Cross-check: Table 1 of DeepSeek's [ISCA paper (arXiv:2505.09343)](https://arxiv.org/abs/2505.09343) gives per-token KV of 70.272 KB for DeepSeek-V3, 327.680 KB for Qwen-2.5 72B, and 516.096 KB for LLaMA-3.1 405B.
- Anchor: **large GQA models are hundreds of KiB per token, MLA is tens of KiB; one 128k-token 70B sequence = 40 GiB = half an H100.**

## 3. Compute and speed

| Quantity | Typical value | How to derive it | Source |
|---|---|---|---|
| Forward FLOPs / token (linear layers) | 2N | one multiply-add per weight; N is total parameters for dense models and **active parameters for MoE** (37B for DeepSeek-V3, not 671B); the input embedding is a lookup and not in N; attention is extra | [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) |
| Attention FLOPs / token / layer | 4 · ctx · n_heads · head_dim | QKᵀ and AV are 2 · ctx · d_attn each | derived |
| Dense decode linear-layer intensity | ≈ batch | weights read once per step | derived |
| MoE routed-expert intensity | ≈ B · k / E (B / 32 for DeepSeek-V3) | each expert gets only B·k/E tokens; crossing the BF16 ridge takes about 9,400 tokens globally; FP8 weights double I and the FP8 ridge doubles too, so still about 9,400 | derived, see [Module 01](../modules/01_inference_basics/README.en.md) Section 3.4 |
| Decode attention intensity | n_heads / n_kv_heads | assumes decode, BF16 KV (×2 with FP8), no prefix shared between sequences. MLA with DP attention ≈ 242, ÷ t when attention is split t ways by TP; × number of sequences for the shared-prefix part | derived, see [Module 01](../modules/01_inference_basics/README.en.md) Sections 3.3 and 3.5 |
| Lower bound for incremental prefill / chunked-prefill chunks | compute-bound only when new tokens ≳ ridge (about 300 for H100 BF16) | I ≈ tokens processed at once | derived |
| Llama-3-8B, 1 × H100, batch 1, 4k context | ≈ 5.8 ms/token at MBU 0.8 (≈ 172 tokens/s); 4.6 ms at full bandwidth (≈ 215 tokens/s) | 15.5 GB (15.0 GB of weights without the embedding table + 0.54 GB of KV) ÷ (3.35 TB/s × 0.8 or × 1.0) | derived |
| Llama-3-70B, 8 × H100, 32k context, batch 40 | TPOT ≈ 26.5 ms, ≈ 1,500 tokens/s (TP all-reduce excluded) | 568 GB ÷ (26.8 TB/s × 0.8) | derived |
| Llama-3-8B prefill of a 4k prompt, 1 × H100 | ≈ 0.10 s | 6.2e13 FLOPs ÷ (989e12 × 0.6) | derived |
| Llama-3-8B computing only 128 new tokens (first 4k cached) | ≈ 6.2 ms, set by reading weights and KV | compute about 3.5 ms < memory about 6.2 ms (full weights + 4k of KV) | derived |
| Llama-3-70B prefill of a 64k prompt, 8 × H100 | ≈ 3.1 s | linear 9.0e15 + attention 5.6e15 FLOPs | derived |
| DeepSeek production, per node (8 × H800) | prefill ≈ 73.7k tokens/s (incl. hits; roughly removing the 56.3% global hit rate, about 32k tokens/s actually computed), decode ≈ 14.8k tokens/s | production statistics; the prefill unit is 4 nodes EP32 and the decode unit 18 nodes EP144, not the same deployment | [DeepSeek inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| DeepSeek production, per-user output speed | 20–22 tokens/s; average KV length per output token 4,989 | production statistics | Same |
| Concurrent sequences per DeepSeek decode node | ≈ 700 | Little's law: 14.8k ÷ 21 | derived |
| MTP (multi-token prediction) | 80–90% acceptance for the second token, 1.8× generation TPS | measured in production | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| DeepSeek-V3 TPOT floor set by EP communication over 400G IB | ≈ 14.76 ms (≈ 67 tokens/s) | the paper assumes about 32 tokens per device; see Section 7 | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| PaLM 540B inference | 29 ms/token at small batch (int8 weights); 76% MFU on large-batch input processing | measured in the paper | [Pope et al. (arXiv:2211.05102)](https://arxiv.org/abs/2211.05102) |

## 4. Interconnect bandwidth

Use the **one-way** column when estimating "how long one transfer takes"; NVLink and PCIe numbers on vendor spec pages are bidirectional totals.

| Link | Vendor figure | One way | Effective / measured | Source |
|---|---|---|---|---|
| NVLink 3 (A100) | 600 GB/s per GPU, bidirectional total | 300 GB/s | — | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| NVLink 4 (H100) | 900 GB/s per GPU, bidirectional total; 18 links | 450 GB/s | — | [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) |
| H800 NVLink | 400 GB/s per GPU (cut for compliance) | 200 GB/s | about 160 GB/s | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343), [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) |
| NVLink 5 (B200) | 1,800 GB/s per GPU; 130 TB/s total for NVL72 | 900 GB/s (the ISCA paper says GB200 NVL72 has 900 GB/s one-way) | — | [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/), [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| InfiniBand NDR (ConnectX-7) | 400 Gb/s per port | 50 GB/s | about 40 GB/s once small messages and latency are considered | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) (an H800 node has 8 × 400G CX7 NICs), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| PCIe Gen4 x16 | 64 GB/s bidirectional total | 32 GB/s | lower in practice | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| PCIe Gen5 x16 | 128 GB/s bidirectional total | 64 GB/s | lower in practice; KV offload to CPU memory goes here | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| NVLink-C2C (Grace Hopper) | 900 GB/s (only a total is given) | — | — | [NVIDIA GH200](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) (NVIDIA says 7× PCIe Gen5) |

Anchor (one way): **HBM 3.35 TB/s ≫ NVLink 0.2–0.45 TB/s ≫ PCIe Gen5 64 GB/s ≈ IB 50 GB/s**. Intra- vs inter-node on H800 is about 4:1 (200 : 50; with the measured 160 : 50 it is about 3.2:1, the figure the DeepSeek-V3 technical report uses). Moving a 64k-token Llama-3-70B KV (about 21.5 GB) takes about 0.43 s over one IB NIC, about 54 ms over eight.

## 5. Storage tiers: where KV can live

"Tier bandwidth" is the storage device's own bandwidth; "as seen from the GPU" is the bottleneck link the GPU must go through, and is the column to use for offload and reload estimates.

| Tier | Typical capacity | Tier bandwidth | As seen from the GPU (bottleneck link) | Example | Source |
|---|---|---|---|---|---|
| HBM | 80–180 GB / GPU | 3.35–8 TB/s / GPU | same | H100, B200 | see Section 1 |
| Host DRAM | hundreds of GB / socket | hundreds of GB/s / socket (Grace: 384 GB/s for the 480 GB configuration, 512 GB/s for 120 / 240 GB) | about 64 GB/s one way per GPU over PCIe Gen5 x16 (theoretical, lower in practice; a socket's memory bandwidth is shared by several GPUs) | Grace CPU | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/), [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| Local NVMe SSD | TBs / drive | about 13 GB/s / drive (PCIe 5.0 sequential read) | limited by drive count and PCIe lanes; several drives can be aggregated | Samsung PM1743 | [Samsung Newsroom](https://news.samsung.com/global/samsung-develops-high-performance-pcie-5-0-ssd-for-enterprise-servers) |
| Distributed SSD storage | PBs | about 6.6 TiB/s aggregate read over 180 storage nodes | limited by the client NIC: KVCache read peak about 40 GiB/s (client with 1 × 400 Gbps NIC), about 5 GB/s per GPU when split across 8 | DeepSeek 3FS | [3FS](https://github.com/deepseek-ai/3FS) |

Anchor: **as seen from the GPU, HBM → host DRAM drops about 50×** (3.35 TB/s vs 64 GB/s), and remote SSD split per GPU drops roughly another order of magnitude; capacity grows one to two orders of magnitude per tier. 56.3% of DeepSeek's production input tokens hit KV cache **on disk**: at its scale, DeepSeek chose to store KV on SSD for reuse rather than recompute prefill every time.

## 6. Queueing and cache economics

| Quantity | Typical value | Source |
|---|---|---|
| Little's law | average number in system L = arrival rate λ × average time in system W | [Little 1961](https://pubsonline.informs.org/doi/abs/10.1287/opre.9.3.383) |
| Planning by percentile | with Poisson arrivals the number in system is Poisson with mean L (M/G/∞ steady state), p99 ≈ L + 2.33√L; about 211 for L = 180, about 17% more | derived, see [Module 01](../modules/01_inference_basics/README.en.md) Section 8 |
| Average KV when context grows each turn | about half the final length under linear growth | derived |
| Agent input:output tokens | about 100:1 | [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) |
| Price of cache-hit input | Anthropic: cache reads 0.1× base input price; 5-minute cache writes 1.25×, 1-hour 2× | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| OpenAI cache discount | cached input up to 90% cheaper | [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) |
| Anthropic default cache TTL | 5 minutes (1 hour optional) | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| DeepSeek production KV cache hit rate | 56.3% of input tokens hit the on-disk KV cache within 24 hours | [DeepSeek inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| Multi-agent token amplification | agents about 4× chat, multi-agent about 15× | [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system) |

Three decision rules (all derived; limits in parentheses):

1. **Store or recompute**: loading KV from a tier beats recomputing ⟺ that tier's bandwidth to the GPU > the break-even bandwidth B\* = KV bytes per token × single-GPU prefill throughput (tokens/s). For Llama-3-8B on H100 it is about 131,072 B × 37k tokens/s ≈ 4.8 GB/s; for DeepSeek-V3 (MLA + large active parameters) about 0.56 GB/s per GPU. Linear layers only, fixed latency ignored; the derivation and more models are in Section 4.2 of [Module 05](../modules/05_cache_prefill/README.en.md).
2. **Which tier**: among tiers with bandwidth > B\*, pick the one with enough capacity and the lowest unit cost. Per Section 5, host DRAM (64 GB/s per GPU) is almost always worth it; remote SSD (about 5 GB/s per GPU) has ample margin for large MLA models and roughly breaks even for small dense models like 8B.
3. **How long to keep during a tool wait**: assuming the instance's concurrency is limited by KV memory, holding a share s of the node's KV capacity for t seconds loses t × s "node-seconds"; recomputing costs r node-seconds. Keep if t × s ≤ r, otherwise offload or drop. For Llama-3-70B, 8 × H100, 64k context, s ≈ 4.9% and r ≈ 3.1 s, so the threshold is about 62 s. Tool latency is random, so pick the TTL from its distribution ([Continuum](https://arxiv.org/abs/2511.02230)). The exercise is [Module 01](../modules/01_inference_basics/README.en.md) exercise (d).

## 7. Communication mental math

| Communication | Volume per step | Example | Source |
|---|---|---|---|
| TP all-reduce (decode) | 2 per layer (after attention and after the MLP), each message = B × d_model × bytes; ring all-reduce sends and receives about 2(n − 1)/n × message per GPU | Llama-3-70B, TP8, B = 40: 160 per step, about 180 MB per GPU in total, about 0.4 ms at 450 GB/s one-way NVLink; bandwidth is not the issue, the fixed latency of 160 collectives adds up | derived |
| EP dispatch + combine (MoE) | per token per layer (dispatch bytes + combine bytes) × (k + shared experts) × hidden | DeepSeek-V3: (1 byte + 2 bytes) × 9 × 7K ≈ 189 KB; with 32 tokens per device over 50 GB/s IB, 120.96 µs per layer, 241.92 µs per layer under dual micro-batch overlap, 14.76 ms over 61 layers | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| Moving KV for PD disaggregation | prompt length × KV bytes per token (plus block metadata) | 64k: Llama-3-70B about 21.5 GB (0.43 s over one IB NIC), DeepSeek-V3 about 4.6 GB (0.09 s) | derived |

## 8. One-minute estimation templates

| Question | Template | Limits |
|---|---|---|
| How long is one decode step? | (weight bytes + B × ctx × KV/token) ÷ (GPUs × HBM bandwidth × 0.8) | a lower bound when memory-bound with similar context lengths in the batch; use Σ ctx_i when they differ; add TP / EP communication and fixed overheads |
| Maximum concurrency? | (GPUs × HBM × 0.9 − weight bytes) ÷ (ctx × KV/token) | an aggregate upper bound; real deployments are per GPU: min over GPUs of ((HBM − this GPU's weights − reserve) ÷ this GPU's KV per request), and DP attention, redundant experts, or TP beyond the KV-head count all shrink it |
| TTFT? | queueing + max((2N × n + 2 × n² × d_attn × layers) ÷ (GPUs × peak × 0.6), (weights + whole KV) ÷ (GPUs × bandwidth × 0.8)) | a lower bound; add cache loading, communication, and waits from chunked-prefill interleaving |
| Compute-bound or not? | FLOPs ÷ bytes, compared with the ridge (Hopper BF16 ≈ 300) | necessary but not sufficient: small matmuls rarely reach peak MFU |
| How long to move the KV once? | ctx × KV/token ÷ one-way link bandwidth (PCIe Gen5 ≈ 64 GB/s, IB ≈ 50 GB/s per port) | can be pipelined layer by layer behind compute |
| How much memory for pinned KV? | Little's law: arrival rate × (generation time + tool wait) × ctx × KV/token | the mean; add margin for p99 with L + 2.33√L |

## 9. Mental-math drills

1. 8 × H100, Llama-3-70B, batch 32, 16k context each: roughly what is TPOT?
   <details><summary>Answer</summary>Each step reads about 139 GB of weights (without the embedding table) + 32 × 16,384 × 327,680 B ≈ 172 GB of KV ≈ 311 GB; 8 GPUs give 26.8 TB/s, about 14.5 ms at MBU 0.8. Compute is only about 1.2 ms, so it is memory-bound. This is a lower bound that ignores TP's 160 all-reduces.</details>
2. How long does it take to move a 64k-token Llama-3-70B KV from the prefill pool to the decode pool over 1 and over 8 400G NICs? What fraction of the prefill time is that?
   <details><summary>Answer</summary>The KV is about 21.5 GB. About 0.43 s over 1 NIC and about 54 ms over 8; the same prefill takes about 3.1 s on 8 × H100, so 14% and 1.7%, mostly hidden by layer-wise pipelining. With a 90% prefix hit, only about 0.41 s of compute is needed and the transfer takes as long as the compute.</details>
3. An agent platform gets 20 new sessions per second, each averaging 10 s of generation and 30 s of tool waits, 8k context, Llama-3-8B, KV pinned throughout. How much memory for the mean and for p99?
   <details><summary>Answer</summary>L = 20 × 40 = 800 sessions; each 8,192 × 131,072 B ≈ 1.07 GB; the mean is about 859 GB, the KV space of about 15 H100s after holding 8B (about 55.9 GB each). With Poisson arrivals p99 ≈ 800 + 2.33 × √800 ≈ 866 sessions, about 8% more. With context growing turn by turn, average occupancy roughly halves.</details>
4. How many tokens per step does DeepSeek-V3 decode need globally for the routed-expert GEMMs to cross the BF16 ridge? What if each sequence verifies 1 extra token per step with MTP?
   <details><summary>Answer</summary>Each expert gets B × 8 / 256 = B / 32 tokens, so B ≈ 295 × 32 ≈ 9,400 tokens; with 2 tokens per sequence per step, about 4,700 concurrent sequences suffice. This is what the ISCA paper means when it says MTP "increases the inference batch size, which is crucial for boosting EP computational intensity".</details>
