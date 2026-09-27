[中文](interview-questions.md) · **English**

# Interview Question Bank: 67 Questions

> **How to use**: 8 topic groups; each question is tagged with a difficulty: ★ basic, ★★ intermediate, ★★★ deep dive.
>
> **Answer keys** always follow "**conclusion → order of magnitude → trade-off**", which is also the order to speak in during an interview: give a one-line judgment, back it with a number you can compute in your head, then state the cost and where it applies.
>
> **Where the numbers come from**: every number without a citation is derived from a formula; the derivations are in [Module 08 §3 parameter card](../modules/08_system_design/README.en.md#3-parameter-card-shared-inputs-for-every-estimate-in-this-module).
>
> **H800 parameters**: NVIDIA has no official spec page for the H800's compute or HBM bandwidth, so this bank estimates with the H100 SXM dense values (BF16 989 / FP8 1,979 TFLOPS, 3.35 TB/s); the reasoning is in the same section.
>
> **Suggested practice**: cover the answer key, answer in 2 minutes, then compare.

| Group | Questions | Module |
|---|---|---|
| [1. Inference fundamentals](#1-inference-fundamentals-9-questions) | 9 | [Module 01](../modules/01_inference_basics/README.en.md) |
| [2. Inference engines](#2-inference-engines-9-questions) | 9 | [Module 02](../modules/02_inference_engines/README.en.md) |
| [3. DeepSeek systems](#3-deepseek-systems-11-questions) | 11 | [Module 03](../modules/03_deepseek_stack/README.en.md) |
| [4. Agent workloads and caching](#4-agent-workloads-and-caching-8-questions) | 8 | [Module 04](../modules/04_harness_workloads/README.en.md), [Module 05](../modules/05_cache_prefill/README.en.md) |
| [5. Long tail and async RL](#5-long-tail-and-async-rl-9-questions) | 9 | [Module 06](../modules/06_long_tail_async/README.en.md) |
| [6. Credit assignment](#6-credit-assignment-8-questions) | 8 | [Module 07](../modules/07_multi_agent_credit/README.en.md) |
| [7. System design](#7-system-design-7-questions) | 7 | [Module 08](../modules/08_system_design/README.en.md) |
| [8. Behavioral and project](#8-behavioral-and-project-6-questions) | 6 | [Module 08 §9](../modules/08_system_design/README.en.md#9-how-to-talk-about-your-harnessinference-co-design-project-in-interviews) |

## 1. Inference fundamentals (9 questions)

### 1.1 ★ Why is decode usually memory-bound while prefill is compute-bound?

- **Conclusion**: each decode step produces only 1 token per sequence but has to read all the weights and the KV once; prefill processes hundreds or thousands of tokens at once, so a single read of the weights is shared across all of them.
- **Order of magnitude**:
  - In BF16, each parameter costs 2 FLOPs per token and takes 2 bytes, so dense decode with batch B has an arithmetic intensity of about B FLOPs/Byte.
  - The BF16 ridge point of H100/H800 ≈ 989 / 3.35 ≈ 295 FLOPs/Byte. B needs to reach about 300 before decode can become compute-bound; once KV reads are counted as well, that is even harder to reach.
- **Trade-off**: a larger batch raises throughput, but TPOT gets longer and KV usage grows.
- **Sources**: [Roofline (Williams et al.)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf), [H100 spec page](https://www.nvidia.com/en-us/data-center/h100/).

### 1.2 ★ Write the per-token KV cache formula and compute it for Llama-3-70B and DeepSeek-V3.

- **Conclusion**: GQA is $2L\,n_{kv}\,d_h\,b$; MLA is $L\,(r_{kv}+d_{rope})\,b$.
- **Order of magnitude**:
  - Llama-3-70B: 2 × 80 × 8 × 128 × 2 = 320 KB.
  - DeepSeek-V3: 61 × (512+64) × 2 = 70,272 B ≈ 70 KB (68.6 KiB).
  - At 128K context, that is about 40 GiB and 8.6 GiB respectively.
- **Trade-off**: MLA saves memory, but at decode time you either "absorb" the weights and run attention on the latent (more compute), or expand the latent into per-head K/V (more memory and bandwidth).
- **Sources**: [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json), [Meta llama-models](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py).

### 1.3 ★ What are TTFT, TPOT, throughput, and goodput? Why do they conflict?

- **Conclusion**:
  - TTFT (time to first token) is set by queueing and prefill; TPOT (time per output token) is set by the time of each decode step.
  - Throughput wants a large batch, but a large batch stretches TPOT; and when prefill is inserted into a decode batch, TPOT jitters.
  - Goodput is the throughput of requests that meet the SLO.
- **Order of magnitude**: DistServe optimizes for goodput and reports serving 7.4× more requests, or meeting a 12.6× tighter SLO, while keeping over 90% of requests within the latency constraints.
- **Trade-off**: chunked prefill spreads the interference out on the same machine; PD disaggregation separates the two physically.
- **Source**: [DistServe](https://arxiv.org/abs/2401.09670).

### 1.4 ★★ Going from MHA to MQA, GQA, and MLA, what does each save, and what does it cost?

- **Conclusion**:
  - MQA: all heads share one KV.
  - GQA: heads are grouped, and each group shares one KV.
  - MLA: caches a low-rank latent, plus a small decoupled RoPE key.
- **Order of magnitude**:
  - DeepSeek-V2 reports that MLA reduces the KV cache by 93.3% relative to DeepSeek 67B and raises maximum generation throughput to 5.76×.
  - Stored uncompressed as plain MHA, DeepSeek-V3's KV would take 61 × 128 × (192+128) × 2 ≈ 5.0 MB per token, about 71× the MLA size.
- **Trade-off**: MQA/GQA can lose quality; the GQA paper compensates by uptraining with about 5% of the original pretraining compute. MLA adds compute and implementation complexity, and RoPE has to be decoupled.
- **Sources**: [MQA](https://arxiv.org/abs/1911.02150), [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

### 1.5 ★★ At what context length does attention cost more FLOPs than the linear layers?

- **Conclusion**: per token, the linear layers cost $2N_{act}$ FLOPs and attention costs $a\cdot K$, where $a = 2L\,n_h(d_{qk}+d_v)$. The two are equal at $K^* = 2N_{act}/a$.
- **Order of magnitude**: for DeepSeek-V3 under prefill (non-absorbed) accounting, $a$ ≈ 5.0M and $K^*$ ≈ 70G / 5.0M ≈ 14.1K; for Llama-3-70B, ≈ 137G / 2.6M ≈ 52K. Here $N_{act}$ excludes the embedding lookup and lm_head (35.1B for DeepSeek-V3, 68.4B for Llama-3-70B). MoE models activate few parameters, so attention takes over earlier.
- **Trade-off**:
  - For long-context agent workloads, prefill optimization has to focus on attention kernels (FlashAttention/FlashMLA) and sparse attention (NSA/DSA); optimizing GEMMs alone is not enough.
  - "MLA saves memory, not compute" holds only under this accounting. In decode's absorbed mode, each attention pair actually costs 17.0M FLOPs (3.4×): compute is spent to save memory traffic.

### 1.6 ★★ When does speculative decoding help, and when does it slow things down?

- **Conclusion**: when decode is memory-bound and compute is sitting idle, trading a cheap draft for several extra tokens per step pays off. When the acceptance rate is low, or verification pushes the tokens per step past the ridge point, the gain disappears.
- **Order of magnitude**:
  - Leviathan et al. report a 2–3× speedup on T5X with an unchanged output distribution.
  - DeepSeek-V3's MTP: the second token is accepted 85–90% of the time, giving about 1.8× TPS. With a single draft token, the expected output per step ≈ 1 + p (p = acceptance rate).
  - **Mental-math threshold for dense models**: verifying k drafts turns the tokens per step into B·(k+1). As long as B·(k+1) stays below the ridge point (about 295 in BF16), verification is nearly free; with k = 3, the gain starts to vanish once B exceeds about 74.
  - **MoE**: verifying more tokens activates more experts. DeepSeek-V3 picks 8 of 256 experts per token; approximating the choice as uniform, 1 token covers 8 experts and 4 tokens about 30, so the bytes of expert weights read grow with it, and the "verification is nearly free" premise weakens.
  - **MLA**: MTP puts 2 query tokens in each step, so at 32K context decode attention moves from sitting right on the ridge to being compute-bound ([Module 08 §4 step 3](../modules/08_system_design/README.en.md#4-drill-a-serving-10000-concurrent-coding-agents-on-an-h800-cluster-prefix-caching--pd-disaggregation)).
- **Trade-off**: you pay for drafting and verification, and KV management gets more complex; the larger the batch, the smaller the gain. Conversely, the long-tail phase of RL rollout has only a few long sequences left and a tiny batch, which is exactly where speculative decoding shines: [Seer](https://arxiv.org/abs/2511.14617) uses adaptive grouped speculative decoding to speed up rollout generation, and context-aware scheduling to cut the latency of long-tail requests.
- **Sources**: [Speculative decoding](https://arxiv.org/abs/2211.17192), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [Seer](https://arxiv.org/abs/2511.14617).

### 1.7 ★★★ Why does MoE decode need a very large global batch and large-scale EP?

- **Conclusion**: each expert receives only a small slice of the batch's tokens, so the cost of reading its weights is not amortized.
- **Order of magnitude**:
  - DeepSeek-V3 activates 8 of 256 routed experts per token, so each expert's average batch ≈ global batch / 32.
  - FP8 weights take 1 byte per parameter and cost 2 FLOPs per token, so an expert GEMM's arithmetic intensity ≈ 2 × per-expert batch.
  - To approach the FP8 ridge point (≈ 1,979 / 3.35 ≈ 590 FLOPs/Byte), each expert's batch has to be about 300, which means a global batch of about 10K tokens per step.
  - DeepSeek's online decode uses EP144 across 18 nodes.
- **Trade-off**:
  - Large EP means heavy all-to-all traffic and sensitivity to slow GPUs, so it has to be paired with DeepEP, dual-batch overlap, and EPLB.
  - **Counterexample: long contexts break this premise.** Under a 32K agent load, KV capacity limits an EP144 instance's global batch to about 2,930 streams, so each expert gets only about 91 tokens. The main value of large EP then becomes "freeing memory for KV": EP144 puts 21.5 GB of weights on each GPU, EP32 37 GB.
- **Sources**: [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [Module 08 §4](../modules/08_system_design/README.en.md#4-drill-a-serving-10000-concurrent-coding-agents-on-an-h800-cluster-prefix-caching--pd-disaggregation).

### 1.8 ★ What problem does FlashAttention solve?

- **Conclusion**: attention is bottlenecked on IO. FlashAttention computes in tiles, so the intermediate N×N matrix is never written back to HBM. It still computes exact attention, not an approximation.
- **Order of magnitude**:
  - Extra memory beyond inputs and outputs drops from O(N²) to O(N).
  - FlashAttention-3 on H100 reaches up to 740 TFLOPS in FP16 (about 75% utilization) and close to 1.2 PFLOPS in FP8.
- **Trade-off**: the kernel has to be rewritten for every hardware generation; it does not reduce FLOPs, which is where it differs from sparse attention.
- **Sources**: [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-3](https://arxiv.org/abs/2407.08608).

### 1.9 ★★★ After an MLA prefix hit, how is prefill attention computed?

- **Conclusion**: the cache stores 512-dim latents, not per-head K/V, so there are two algorithms:
  - Up-project the cached latents into per-head K/V, then compute ordinary MHA. vLLM calls this "compute friendly" and uses it for prefill and chunked prefill.
  - Skip the up-projection and compute directly on the latent with absorbed weights. vLLM calls this "data-movement friendly" and uses it for decode.
- **Order of magnitude**:
  - Up-projection costs about 2 × 512 × 128 × 256 × 61 ≈ 2.05 GFLOPs per cached token. A 32K prefix costs about 67 TFLOPs, 13.7% more than computing the 2K increment alone, so the full-recompute-to-hit ratio drops from 11.2× to about 9.8×.
  - Absorbed mode costs 17.0M FLOPs per (q,k) pair, 12.0M more than the 5.0M after up-projection. With an increment Δ < 2.05G / 12.0M ≈ 171 tokens, absorbed mode is cheaper.
- **Trade-off**: up-projection needs extra workspace memory, which is why vLLM processes the context in fixed-size chunks; absorbed mode saves memory but computes more. Small-chunk local prefill on the decode side suits absorbed mode.
- **Sources**: [vLLM mla_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py), [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json).

## 2. Inference engines (9 questions)

### 2.1 ★ What problem does PagedAttention solve?

- **Conclusion**: reserving contiguous memory per request for KV causes heavy fragmentation and waste. PagedAttention allocates KV in blocks, like virtual memory, and blocks can be shared across requests.
- **Order of magnitude**: the vLLM blog says earlier systems wasted 60%–80% of KV memory, versus under 4% for vLLM; the paper reports a 2–4× throughput improvement.
- **Trade-off**: indirect addressing needs dedicated kernels; the block size trades fragmentation against metadata volume.
- **Sources**: [vLLM paper](https://arxiv.org/abs/2309.06180), [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm).

### 2.2 ★ What is continuous batching (iteration-level scheduling)?

- **Conclusion**: schedule by "iteration" instead of by "request". Every step can admit new requests and remove finished ones, without waiting for the whole batch to end.
- **Order of magnitude**: Orca (OSDI'22) reports 36.9× the throughput of FasterTransformer on GPT-3 175B at the same latency.
- **Trade-off**: mixing prefill and decode in one batch disturbs TPOT, which is what led to chunked prefill and PD disaggregation. Orca uses selective batching to handle the irregularly shaped parts of attention.
- **Source**: [Orca](https://www.usenix.org/conference/osdi22/presentation/yu).

### 2.3 ★★ What does chunked prefill buy you, and what does it cost?

- **Conclusion**: a long prefill is split into roughly equal chunks that are scheduled together with decode, so one long prompt no longer stalls every decode (stall-free).
- **Order of magnitude**: relative to vLLM, Sarathi-Serve reports 2.6× the serving capacity for Mistral-7B on a single A100, up to 3.7× for Yi-34B, and up to 5.6× for Falcon-180B.
- **Trade-off**: smaller chunks make TPOT smoother but lower prefill efficiency and lengthen TTFT; each chunk also has to re-read the KV that already exists before it.
- **Source**: [Sarathi-Serve](https://arxiv.org/abs/2403.02310).

### 2.4 ★★ How does prefix caching (RadixAttention) work, and why does eviction start from leaves?

- **Conclusion**: KV is organized by token in a radix tree, and a new request walks the tree to match its longest common prefix. SGLang uses LRU and evicts leaves first, because after a leaf goes, its ancestors can still be reused by other branches until they in turn become leaves.
- **Order of magnitude**: SGLang reports up to 6.4× higher throughput; in DeepSeek's production service, 56.3% of input tokens hit the on-disk cache.
- **Trade-off**: the hit rate depends on the harness keeping prefixes stable; plain LRU handles agents' tool waits poorly (see 4.2).
- **Sources**: [SGLang](https://arxiv.org/abs/2312.07104), [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).

### 2.5 ★★ What does PD disaggregation buy you, and what does it cost?

- **Conclusion**: prefill and decode run on separate instances, so they stop interfering with each other and each side can pick its own parallelism and batch size. The cost is moving KV between them, plus having to tune the ratio of the two pools.
- **Order of magnitude**:
  - DistServe reports 7.4× more requests / 12.6× tighter SLOs.
  - Splitwise reports 1.4× throughput at 20% lower cost, or 2.35× throughput under the same cost and power budgets.
  - The full KV of one DeepSeek-V3 turn (32K prefix + 2K increment = 34.8K tokens) is about 2.45 GB, about 49 ms over one 50 GB/s NIC; sending only the increment is about 0.14 GB (see 2.9).
- **Trade-off**: requests with very short increments prefill locally (conditional disaggregation, see 2.6).
- **Sources**: [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677).

### 2.6 ★★★ When should prefill run directly on the decode node?

- **Conclusion**: when the "effective input length after subtracting the prefix hit" is short, or is a small fraction of the raw length, doing it locally is cheaper. But on a decode instance with DP attention + large EP, local prefill stalls the whole instance, so the threshold must be much lower.
- **Order of magnitude**: with 32K context and a 2K increment (Module 08 Drill A):
  - Decode uses DP144 attention + EP144 experts, and every MoE layer has a synchronous all-to-all. While one rank runs a local prefill, the other 143 GPUs wait for it, stalling the instance's roughly 2,900 streams.
  - Running the whole 2K on one rank takes about 0.69 s (600 TFLOPS); if it may add only 10 ms per step, just about 215 tokens fit in one second.
  - Doing it remotely means moving KV: 2.45–4.75 GB per turn for the full KV, or about 0.14 GB for the increment only (see 2.9).
- **Trade-off**: NVIDIA Dynamo decides with two thresholds, `eff_isl_threshold` (default 2048) and `eff_isl_ratio_threshold` (default 0.7), plus `decode_busy_threshold`, which forces remote prefill when decode-side KV usage is too high. By the estimate above, an MLA deployment with DP attention should lower the length threshold to a few hundred tokens and balance the local prefill budget across DP ranks.
- **Sources**: [Dynamo conditional disaggregation](https://docs.nvidia.com/dynamo/dev/advanced-customizations/conditional-disaggregation), [Module 08 §4](../modules/08_system_design/README.en.md#4-drill-a-serving-10000-concurrent-coding-agents-on-an-h800-cluster-prefix-caching--pd-disaggregation).

### 2.7 ★★ When KV memory runs out and requests must be preempted, swap or recompute?

- **Conclusion**: swap moves the KV out to host memory and brings it back on resume; recompute drops the KV and runs prefill again on resume. The vLLM paper compares the two. In vLLM V1 the default is already RECOMPUTE, and the official docs give the reason: "recomputation has lower overhead in the V1 architecture".
- **Order of magnitude**: DeepSeek-V3's KV at 34.8K context is 2.45 GB.
  - Swap: bringing it back over PCIe (about 50 GB/s effective) takes about 49 ms.
  - Recompute: about 5.5 PFLOPs, or about 9.1 GPU-seconds at 600 TFLOPS/GPU. But with prefix caching, the blocks a preempted request released often still hit, so only the missed part actually needs recomputing.
- **Trade-off**: today's mainstream combination is "recompute + prefix cache + tiered offload". Swap's role has passed to a KV offload tier, such as vLLM's OffloadingConnector (which offloads completed KV blocks to slower but larger tiers) or LMCache. Short sequences are cheaper to recompute; for long ones, make sure the offload tier can catch them.
- **Sources**: [vLLM](https://arxiv.org/abs/2309.06180), [vLLM tuning docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [vLLM KV Offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/).

### 2.8 ★★★ How do you get both cache-affinity routing and load balancing?

- **Conclusion**:
  - First, pick a preferred instance by hashing the prefix or session.
  - When the preferred instance exceeds its load cap, spill to a secondary one (e.g. power-of-two choices) and pull the prefix over from the distributed cache.
- **Order of magnitude**:
  - In Module 08 Drill C, reading every hit from L2/L3 would need about 4.9 TB/s of network bandwidth, so affinity routing is essential to keep most hits in local HBM.
  - The Tail at Scale notes that picking between two servers by queue length at enqueue time does much better than uniform random assignment (in its words, "exponentially improves").
- **Trade-off**: a load cap set too low hurts the hit rate; set too high, hot spots queue. Dynamo's KV router maintains a global prefix tree from KV events reported by each worker.
- **Sources**: [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/), [Dynamo router](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/router-design).

### 2.9 ★★★ Under PD disaggregation, how is KV transferred?

- **Conclusion**: get the data flow straight before talking bandwidth. Pushing the full KV, pushing only the increment, and keeping the prefix on both sides via session affinity differ by an order of magnitude in bytes per turn. Transfer should stream layer by layer, overlap with compute, and use a transfer engine that aggregates bandwidth across NICs.
- **Order of magnitude** (DeepSeek-V3, 32K prefix + 2K increment, 328 turns/s, Module 08 Drill A):
  - Prefill first reads the 2.30 GB prefix, then pushes the full 2.45 GB: 4.75 GB per turn, 1.56 TB/s cluster-wide.
  - Prefill reads the prefix and pushes back only the 0.14 GB increment: 2.45 GB per turn.
  - Both sides keep the prefix: about 0.17 GB per turn, 56 GB/s cluster-wide, about 14× less.
  - A hit turn takes about 50 ms on one EP32 instance, and the full 2.45 GB takes about 49 ms over one NIC, so layer-wise overlap just hides it.
- **Trade-off**:
  - **Layer-wise transfer**: in Splitwise's words, "Layer-wise KV-cache transfer happens in parallel with the prompt computation for the next layer", with serial transfer for small prompts; Mooncake uses layer-wise prefill to overlap transfer and dumping with compute.
  - **Transfer engines**: Mooncake Transfer Engine aggregates bandwidth across multiple NICs, and NIXL is the point-to-point transfer library that Dynamo uses; DistServe pulls rather than pushes.
  - **Relayout when layouts differ**: GQA under TP is split by KV head, so different TP degrees on P and D mean re-splitting; MLA's latent is shared by all heads and not split by head, so it can be moved whole per request.
  - **Cost of keeping the prefix on both sides**: the prefill side must keep context for every session, about 25 TB in Drill A, mostly in DRAM or on SSD.
- **Sources**: [Splitwise](https://arxiv.org/abs/2311.18677), [Mooncake](https://arxiv.org/abs/2407.00079), [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake), [NIXL](https://github.com/ai-dynamo/nixl), [DistServe](https://arxiv.org/abs/2401.09670).

## 3. DeepSeek systems (11 questions)

### 3.1 ★ What does MLA cache, and why does RoPE need to be decoupled?

- **Conclusion**: each layer caches only a 512-dim compressed latent $c_{KV}$ and a 64-dim RoPE key shared by all heads. DeepSeek-V2 points out that RoPE is incompatible with low-rank KV compression: the rotation depends on position, so it cannot be absorbed into the projection matrices. A small position-carrying key is therefore kept separately.
- **Order of magnitude**: 61 × (512+64) × 2 = 70,272 B/token.
- **Trade-off**: running attention directly on the latent at decode time costs more compute. FlashMLA reaches 3000 GB/s (memory-bound configuration) and 660 TFLOPS (compute-bound configuration) on H800.
- **Sources**: [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

### 3.2 ★★ How is DeepSeek-V3 inference deployed in production, and why is EP sized differently for prefill and decode?

- **Conclusion**:
  - Day 6: a prefill unit spans 4 nodes, with routed experts on EP32 and MLA/shared experts on DP32; a decode unit spans 18 nodes, on EP144/DP144.
  - The technical report gives minimum deployment units of 4 nodes / 32 GPUs for prefill and 40 nodes / 320 GPUs (EP320) for decode.
- **Order of magnitude**: each decode GPU hosts 2 routed experts + 1 shared expert; each prefill GPU hosts 9 routed experts + 1 shared expert; both also carry 32 redundant experts.
- **Trade-off**: decode needs a large batch, hence large EP; prefill already has plenty of tokens, so smaller EP is enough, but its heavy communication has to be hidden with dual-batch overlap.
- **Sources**: [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [DeepSeek-V3](https://arxiv.org/abs/2412.19437).

### 3.3 ★★ What problem does dual-batch overlap solve?

- **Conclusion**: under large EP, all-to-all communication takes a big share of the time. Splitting a batch into two micro-batches lets one compute while the other communicates. In decode, the attention layer is additionally split into two steps and run as a 5-stage pipeline.
- **Order of magnitude**: intra-node NVLink on H800 actually achieves about 160 GB/s, while cross-node IB is about 50 GB/s per NIC (40 GB/s taken as effective), a 3–4× gap.
- **Trade-off**: smaller micro-batches lower GEMM efficiency, and scheduling gets more complex.
- **Sources**: [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [DeepSeek ISCA 2025](https://arxiv.org/abs/2505.09343).

### 3.4 ★★ What is the difference between DeepEP's normal kernels and low-latency kernels?

- **Conclusion**:
  - Normal kernels aim for high throughput and are used in training and prefill.
  - Low-latency kernels use RDMA only and are used in decode.
  - There is also a hook-based way to overlap communication with computation that occupies no SMs.
- **Order of magnitude**: the legacy docs test on H800 + CX7 400Gb/s:
  - Intra-node EP8 dispatch bandwidth is 153 GB/s; cross-node EP32 is about 58 GB/s.
  - The low-latency kernel's EP8 dispatch takes 77 µs.
- **Trade-off**: the current DeepEP README has moved to V2 (NCCL Gin backend), and these numbers now live in the legacy docs. State the version when citing them.
- **Source**: [DeepEP legacy docs](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md).

### 3.5 ★★ At what level do auxiliary-loss-free load balancing and EPLB each operate?

- **Conclusion**:
  - In training, DeepSeek-V3 adds a bias term $b_i$ to each expert's affinity score. The bias is used only to select experts and is adjusted dynamically to balance load, without an auxiliary loss that would interfere with the main objective.
  - In inference, EPLB replicates hot experts as redundant experts and rearranges where experts are placed.
- **Order of magnitude**:
  - Technical report: the prefill stage uses 32 redundant experts, adjusted roughly every 10 minutes.
  - EPLB: a hierarchical policy for prefill (smaller EP), a global policy for decode (larger EP).
- **Trade-off**: load balanced during training does not mean load balanced in production, because the traffic distribution shifts; redundant experts also take memory.
- **Sources**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [EPLB](https://github.com/deepseek-ai/EPLB).

### 3.6 ★★★ How should you read the 545% cost profit margin in Day 6?

- **Conclusion**: it is a theoretical figure computed "as if every token were billed at R1 prices". DeepSeek itself states that actual revenue is substantially lower: V3 is priced lower, web and app access are free, and nighttime discounts apply automatically.
- **Order of magnitude**:
  - Daily cost = 226.75 nodes × 8 GPUs × $2 × 24 h = $87,072.
  - Theoretical daily revenue: $562,027.
  - 56.3% of input tokens hit the on-disk cache.
- **Trade-off**: in an interview, be able to break cost into "average node count × unit price". The two big levers are the hit rate and day/night utilization: DeepSeek shrinks inference nodes during low-load nighttime hours and gives those resources to research and training.
- **Source**: [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).

### 3.7 ★★ What are the key design choices in DeepSeek-V3's FP8 training?

- **Conclusion**: fine-grained quantization to tame outliers: scaling factors are computed per 1×128 tile for activations and per 128×128 block for weights.
- **Order of magnitude**:
  - DeepGEMM's first release reached about 1358 TFLOPS for FP8 GEMM on H800, updated to 1550 TFLOPS in 2025-04.
  - For comparison, the FP8 dense peak estimated from the H100 is 1,979 TFLOPS.
- **Trade-off**: block-level scaling makes kernels more complex; in Day 6 inference, core MLA computation and combine transfers still use BF16.
- **Sources**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

### 3.8 ★★★ What does DeepSeek-V3.2's DSA change, and what does it leave unchanged?

- **Conclusion**: DSA uses a lightning indexer to pick the top-2048 KV tokens for each query, cutting the main model's core attention from O(L²) to O(Lk). The indexer itself is still O(L²), but it is light.
- **Order of magnitude**: `index_topk` = 2048 in config.json. At 128K context, main attention looks at only about 1.6% of the tokens.
- **Trade-off**: DSA saves compute, not KV storage. The KV of every token still has to be kept, plus the indexer's own key cache, so the cache capacity problem is unchanged. NSA, earlier work in the same direction, won an ACL 2025 best paper award.
- **Sources**: [DeepSeek-V3.2-Exp report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json), [NSA](https://arxiv.org/abs/2502.11089).

### 3.9 ★★ Why does DeepSeek run MLA with DP attention instead of TP?

- **Conclusion**: MLA has only one KV "head", i.e. the latent shared by all heads. When attention is split with TP, every GPU must store the full latent, so the KV cache is duplicated as many times as the TP degree within the TP group. With DP attention, each GPU stores only the KV of its own batch of requests.
- **Order of magnitude**:
  - Under TP8 the same latent is stored 8 times, so a node's effective KV capacity is only 1/8 of DP's.
  - In SGLang's words: "If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage". Its DP attention raises decode throughput by up to 1.9×.
  - DeepSeek's production service runs MLA and shared experts with DP32 in prefill and DP144 in decode.
- **Trade-off**: DP attention requires balancing attention compute across ranks, which is exactly what Day 6's Prefill/Decode Load Balancers do; every GPU also stores its own copy of the attention weights (MLA is about 11.4B parameters, about 11.4 GB in FP8).
- **Sources**: [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).

### 3.10 ★★★ How is an FP8 KV cache stored, and what wall does it hit?

- **Conclusion**: FP8 KV nearly doubles capacity, but you pay for scales and accuracy risk. For MLA, whose decode attention already sits near the ridge, doubling capacity also means doubling compute.
- **Order of magnitude**:
  - The format FlashMLA designed for DeepSeek-V3.2: 656 B per token per layer = 512 FP8 (e4m3) values + 4 FP32 scales (each covering 128 values) + 64 unquantized BF16 RoPE values. BF16 takes 1,152 B, so capacity is 1.76×, not 2×.
  - The V4.1 format is 528 B: RoPE is quantized too, with one e8m0 scale per 32 values.
  - At 32K context, capacity ×1.76 raises streams per GPU from about 20 to about 36, MLA decode attention from about 18 ms to about 32 ms, and about 40 ms including weight reads, close to the 50 ms TPOT budget.
- **Trade-off**: the V3.2 format keeps the RoPE part in BF16 while V4.1 quantizes it too, a trade-off between accuracy and capacity; finer scale granularity is more accurate but carries more metadata. How many GPUs you save depends on both capacity and compute ([Module 08 §4 step 3](../modules/08_system_design/README.en.md#4-drill-a-serving-10000-concurrent-coding-agents-on-an-h800-cluster-prefix-caching--pd-disaggregation)).
- **Source**: [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

### 3.11 ★★ What does DeepSeek-V4's compressed attention mean for inference systems?

- **Conclusion**: V4 interleaves CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) into a hybrid attention that compresses KV along the sequence dimension, shrinking the KV capacity problem by an order of magnitude. But the "unit" of prefix caching and PD transfer changes too.
- **Order of magnitude**:
  - CSA compresses every 4 tokens into 1 KV entry, then selects top-k (1024 for Pro, 512 for Flash); HCA compresses every 128 tokens into 1.
  - At 1M context, V4-Pro needs only 27% of V3.2's single-token inference FLOPs and 10% of its KV cache.
  - V4-Pro has 1.6T total / 49B activated parameters and supports 1M context.
- **Trade-off**:
  - A compressed entry covers several tokens, so prefix caching must align to compression boundaries (an inference). DeepSeek's API context-caching docs changed too: because of Sliding Window Attention, the cache matches in independent, complete "cache prefix units", and a hit requires matching a whole unit.
  - Bytes for PD transfer and tiered storage drop proportionally. In Module 08's V3-based drills, every KV-related conclusion (capacity limits, transfer cost) has to be recomputed.
- **Sources**: [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348), [DeepSeek context caching docs](https://api-docs.deepseek.com/guides/kv_cache), [Module 03 §5.1](../modules/03_deepseek_stack/README.en.md#51-deepseek-v4-hybrid-attention-for-one-million-tokens).

## 4. Agent workloads and caching (8 questions)

### 4.1 ★ What are the key differences between agent workloads and chat workloads?

- **Conclusion**: many turns, append-only context, very long context, tool waits between turns, one task fanning out into many requests, and heavy-tailed lengths and latencies.
- **Order of magnitude**:
  - Anthropic reports that agents use about 4× the tokens of chat, and multi-agent systems about 15×.
  - Module 08 Drill A assumes each turn appends only about 2K tokens while carrying 32K of context.
- **Trade-off**: the inference engine should optimize per session rather than per request: cache retention, session affinity, per-turn prefill.
- **Source**: [Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

### 4.2 ★★ During a tool call, should the KV stay in HBM, be offloaded, or be dropped?

- **Conclusion**: a three-way choice, depending on how long the tool will take and whether a lower tier exists. Sub-second waits or HBM with slack → keep in HBM; seconds to minutes → offload to DRAM and prefetch; recompute only when every tier misses.
- **Order of magnitude**: for DeepSeek-V3 at a 35K decode context:
  - Keeping it in HBM, one stream takes 4.9% of a GPU's KV space.
  - Offloading to DRAM: swap-out finishes asynchronously during the wait; reading 2.46 GB back takes about 49 ms (single-stream lower bound) and costs no GPU compute; with a completion hint from the harness, it can be prefetched so the restore hides inside the tool wait.
  - Recomputing costs about 9.1 GPU-seconds, about 10× a hit.
  - The "keep in HBM vs recompute" break-even is about 185 s, which only matters when there is no lower tier.
- **Trade-off**: pinning too long crowds out decode capacity; offloading uses PCIe and DRAM, and restores burst when many tools return at once. Getting this decision right requires the harness to supply the expected tool duration and a completion hint.

### 4.3 ★★ Which harness habits break the prefix cache?

- **Conclusion**: anything that changes the prefix bytes will:
  - a timestamp or random ID at the start of the system prompt;
  - a tool list whose order is not fixed;
  - non-deterministic JSON serialization;
  - compressing or rewriting the context;
  - inserting content into the middle of the history.
- **Order of magnitude**: at 32K context, a full miss costs about 10× a hit (including MLA up-projection); if just 10% of turns miss entirely, the prefill load is ×1.88.
- **Trade-off**: compression saves tokens but throws away the cache. Place compression points on block boundaries and compress in batches. Anthropic's cache prefix order is tools → system → messages, so dynamic content belongs at the end.
- **Source**: [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### 4.4 ★★ How do you plan capacity when many projects share a cache?

- **Conclusion**: working set = the sum of every project's shared prefix + the sum of every active session's private suffix, converted to memory by KV bytes per token.
- **Order of magnitude**: each 20K prefix needs 1.41 GB on DeepSeek-V3, so 50 GB of KV space per GPU holds only about 35 of them; on Llama-3-70B it holds only about 7.6.
- **Trade-off**: routing by project affinity raises the hit rate, but a popular project can overwhelm a single instance. So set a load cap and spill to other instances once it is exceeded.

### 4.5 ★★★ How do you find the upper bound on the hit rate?

- **Conclusion**: replay traces offline with Belady's optimal replacement (which assumes the future access sequence is known) for an upper bound, then plot the "cache capacity vs hit rate" curve. But state when Belady applies to prefix caching.
- **Order of magnitude**:
  - Belady assumes equal-sized objects with equal miss costs. Prefix caching violates both: a block is only usable if all its ancestor blocks hit, and recompute cost per byte of KV differs by more than 4× across prefix lengths (about 1.0 MFLOPs/B for a 1K prefix, about 4.6 MFLOPs/B for 100K).
  - So block-level Belady is only an upper bound on "blocks hit", possibly a loose one, not a bound on "FLOPs saved". Report both the Belady block hit rate and FLOPs saved bucketed by prefix length.
  - If LRU is already close to the bound, changing the eviction policy won't help; you can only add capacity or change the harness. Only when the gap is large is policy work worth the effort.
- **Trade-off**: Belady needs future information, so it only serves as a bound; an online policy can approach it with hints from the harness (expected tool duration, session end). When the goal is FLOPs saved, use a cost-aware offline bound.
- **Source**: [Belady 1966](https://doi.org/10.1147/sj.52.0078).

### 4.6 ★★ Roughly what latency and capacity does each tier of KV storage (HBM / host memory / SSD) have?

- **Conclusion**: HBM is the fastest but smallest; host memory is TB-scale and takes tens of milliseconds to read back over PCIe; SSD or a distributed file system holds tens of TB per node and is read back over RDMA.
- **Order of magnitude**:
  - DGX H100 reference configuration: 2 TB host memory, 8 × 3.84 TB NVMe.
  - PCIe Gen5 x16 is rated at 128 GB/s (both directions combined).
  - 3FS reaches about 6.6 TiB/s aggregate read throughput across 180 storage nodes; a KVCache client node peaks at about 40 GiB/s of reads.
- **Trade-off**: deeper tiers are cheaper but slower to read back. For comparison, reading 2.3 GB back from host memory takes about 46 ms, while recomputing costs 9 GPU-seconds.
- **Sources**: [DGX H100](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html), [3FS](https://github.com/deepseek-ai/3FS).

### 4.7 ★★ What cost structure does public prompt caching pricing reflect?

- **Conclusion**: writes cost more than ordinary input, because they pay for write bandwidth and storage; reads are an order of magnitude cheaper; longer TTLs are charged extra.
- **Order of magnitude** (official docs as of 2026-09):
  - Anthropic: 5-minute writes 1.25×, 1-hour writes 2×, reads 0.1×.
  - Gemini: explicit caches pay a storage fee per token-hour.
  - DeepSeek: on-disk by default, with a hit price far below the miss price.
- **Trade-off**:
  - Recomputing a 14K prefix (DeepSeek-V3) takes about 2.5 GPU-seconds, while reading it back takes about 20 ms, so charging 0.1× for reads still leaves a profit.
  - But the token hit rate is not the compute saved: a 70% token hit rate saves only about 60% of the FLOPs, because the uncached suffix still attends to the whole prefix.
  - The TTL is the knob vendors use to trade storage cost against hit rate.
- **Sources**: [Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [Gemini](https://ai.google.dev/gemini-api/docs/generate-content/caching), [DeepSeek](https://api-docs.deepseek.com/guides/kv_cache).

### 4.8 ★★★ What are the security risks of sharing a KV cache across tenants, and how do you defend?

- **Conclusion**: whether a request hits shows up in TTFT, forming a timing side channel that an attacker can use to infer the prefixes of other people's prompts. A per-tenant salt closes this class, but not every timing side channel.
- **Order of magnitude**: Gu et al. (ICML 2025) detected global cache sharing across users at 7 API providers; PROMPTPEEK at NDSS 2025 demonstrated leaking prompts through KV sharing.
- **Trade-off**:
  - vLLM's `cache_salt` is injected into the first block's hash, so only requests with the same salt can reuse the cache; the cost is losing cross-tenant dedup. Both Anthropic's and OpenAI's docs state that caches are not shared across organizations.
  - Side channels that remain after the salt: queueing delay in shared queues, HBM/PCIe/SSD bandwidth contention, hot-spot queueing, hits in a public namespace, and a shared eviction pool.
  - Corresponding mitigations: per-tenant queues and quotas, resource partitioning for sensitive tenants, padding or bucketing TTFT on hits, and statistical audits with enough samples.
- **Sources**: [Gu et al.](https://arxiv.org/abs/2502.07776), [PROMPTPEEK](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/), [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/).

## 5. Long tail and async RL (9 questions)

### 5.1 ★ Why does the long tail slow down synchronous RL rollout?

- **Conclusion**: a whole batch of trajectories has to finish before the update. In AReaL's words, generation must "wait until the longest output in the batch is completed"; DAPO also writes that in synchronous systems generation time is "typically dominated by the generation of long-tail samples".
- **Order of magnitude**: under a Pareto distribution with α = 1.5, the longest of 4,096 trajectories exceeds 85× the mean with 63% probability. Even when capped at 3× the original mean ($9x_m$), the capped mean is only $2.33x_m$, while one trajectory almost surely hits the cap, so slot utilization is about 2.33 / 9 ≈ 26%.
- **Trade-off**: truncation introduces bias, because long trajectories tend to be the hard problems.
- **Sources**: [AReaL](https://arxiv.org/abs/2505.24298), [DAPO](https://arxiv.org/abs/2503.14476).

### 5.2 ★★ What is partial rollout?

- **Conclusion**: each iteration caps output with a fixed token budget; trajectories that don't finish are saved to the replay buffer and continued in the next iteration (Kimi k1.5). Kimi K2 also uses it to keep long-tail trajectories from blocking the whole rollout.
- **Order of magnitude**: there is no single hard number; it is usually evaluated with two quantities: the fraction of slots left idle waiting on the tail each step, and the share of trajectories that span multiple versions.
- **Trade-off**: one trajectory spans several policy versions, so the version number and logprob have to be recorded per token.
- **Sources**: [Kimi k1.5](https://arxiv.org/abs/2501.12599), [Kimi K2](https://arxiv.org/abs/2507.20534).

### 5.3 ★★ How do you control staleness in async RL, and how do you correct for it?

- **Conclusion**:
  - **Control**: cap the maximum version gap η. AReaL calls it the maximum permitted staleness; η = 0 is synchronous.
  - **Correction**: use an objective such as decoupled PPO, which separates the behavior policy from the proximal policy, then apply importance-sampling correction.
- **Order of magnitude**:
  - AReaL reports training up to 2.77× faster than synchronous systems with the same number of GPUs, with matched or improved final performance.
  - Noukhovitch et al. find async about 40% faster than sync on LLaMA 3.1 8B, with matching final performance.
- **Trade-off**:
  - A larger η gives higher throughput, but the data drifts further from the current policy (off-policy). Monitor the distribution and clip fraction of IS ratios, the KL, and the train–inference logprob gap (see 5.9).
  - **At what level to take the ratio**: for a 10K-token trajectory, multiplying per-token ratios into a sequence-level ratio makes the variance grow exponentially with length, which is unusable; so use token-level ratios with clipping, which is biased.
  - **One trajectory spanning several versions** (partial rollout, interruptible generation): record the version per token, and compute each token's ratio against its own behavior policy.
  - **GRPO's group baseline**: a group mean is only comparable when its samples come from the same or nearby versions, so control staleness per group.
- **Sources**: [AReaL](https://arxiv.org/abs/2505.24298), [Asynchronous RLHF](https://arxiv.org/abs/2410.18252).

### 5.4 ★★ Where does weight sync spend its time, and how do you make it fast?

- **Conclusion**: the bottleneck is usually not bandwidth but three things: resharding the training parallel layout into the inference layout, quantization, and global coordination. But get the physical bytes right first.
- **Order of magnitude**:
  - A 671B model is 671 GB in FP8, but the inference side actually lands more: the non-routed 17B is replicated on every GPU under DP. Each decode GPU takes about 22 GB and each prefill GPU about 40 GB. One decode instance + 3 prefill instances land about 7 TB in total.
  - Each GPU receiving through its own 50 GB/s NIC needs only 0.44–0.8 s at minimum.
  - Kimi K2 notes that resharding and broadcasting through a network file system needs an aggregate bandwidth that "reaches several petabytes per second", so it uses a checkpoint engine co-located with the training nodes, completing a full update of 1T parameters in under 30 s.
  - LlamaRL updates a TB-scale model in about 2 s with DDMA; for comparison, the paper cites OpenRLHF taking 111 s to transfer 70B weights.
- **Trade-off**: double-buffering reduces pauses but takes extra memory; in an async setup, new weights can be loaded while generation continues.
- **Sources**: [Kimi K2](https://arxiv.org/abs/2507.20534), [LlamaRL](https://arxiv.org/abs/2505.24034).

### 5.5 ★★★ Why does GRPO make the long tail reappear at the group level?

- **Conclusion**: the group-relative advantage has to wait until all G samples of a prompt are done, so a group finishes when the slowest of its G samples finishes.
- **Order of magnitude**: with G = 16, the probability that all 16 samples fall within p90 is only 0.9¹⁶ ≈ 18.5%; with G = 64 it is about 0.1%.
- **Trade-off**: a smaller G gives a worse baseline estimate. You can also use a baseline such as leave-one-out once part of the group has finished, but that introduces bias.
- **Source**: [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300).

### 5.6 ★★ How do you choose between colocating training and inference and deploying them separately?

- **Conclusion**:
  - Colocation: for example, verl/HybridFlow reshards weights between training and generation with its 3D-HybridEngine. It avoids moving weights across clusters, and GPUs never sit idle for a whole phase.
  - Separate deployment: makes async easier, and training and inference can scale independently.
- **Order of magnitude**:
  - HybridFlow reports 1.53–20.57× the baseline's throughput.
  - In Kimi k1.5's colocated setup, switching from training to inference takes under a minute, and switching back takes about ten seconds.
- **Trade-off**: colocation is still synchronous at heart, so the long-tail problem remains.
- **Sources**: [HybridFlow](https://arxiv.org/abs/2409.19256), [Kimi k1.5](https://arxiv.org/abs/2501.12599).

### 5.7 ★★ What does DAPO's dynamic sampling do to the system?

- **Conclusion**: oversample, then filter out prompt groups whose accuracy is 1 or 0 (their gradient is zero), so that every batch keeps enough useful samples.
- **Order of magnitude**:
  - DAPO scores 50 on AIME 2024 with Qwen2.5-32B, beating DeepSeek-R1-Zero-Qwen-32B's 47 with half the training steps.
  - In the ablation, adding dynamic sampling as the last step raises the score from 42 to 50.
- **Trade-off**: more samples have to be generated, which makes rollout load and the long tail heavier. DAPO argues that in a synchronous system generation time is already set by the long tail, so the extra efficiency cost is not necessarily large.
- **Source**: [DAPO](https://arxiv.org/abs/2503.14476).

### 5.8 ★★★ Why does an async buffer develop a length bias, and how do you mitigate it?

- **Conclusion**: short trajectories finish first and enter the buffer first. Taking batches in arrival order is equivalent to truncated sampling by duration, which systematically favors short tasks, and short tasks also tend to be easier.
- **Order of magnitude**: one way to quantify it is to compare the distribution of trajectory length (or task difficulty) inside batches against the overall distribution and see how far apart they are; then bucket by difficulty and look at each bucket's drop or over-age rate.
- **Trade-off**: mitigations include sampling by group or by prompt, stratifying by length, and reserving slots for over-age trajectories; the cost is extra waiting.

### 5.9 ★★★ Where does train–inference mismatch come from, and how do you fix it?

- **Conclusion**: even with identical weights, the inference engine (vLLM/SGLang) and the training framework (FSDP/Megatron) can compute very different token probabilities, so on-policy RL quietly becomes off-policy. MoE adds another layer: the two sides may pick different experts.
- **Order of magnitude**:
  - In Yao et al.'s words: "the implementation gap implicitly turns the on-policy RL to be off-policy". They tried having vLLM return the probabilities actually used for sampling and forcing lm_head to FP32, and the gap "still exists".
  - One fix is TIS (truncated importance sampling): weight by min(π_learner / π_sampler, C), where C is a hyperparameter.
  - R3 (Rollout Routing Replay) records the inference-side routing distributions and replays them during training, lowering the train–inference KL and preventing MoE RL training from collapsing.
- **Trade-off**:
  - Recomputing the old logprobs on the trainer is the most robust, but costs one extra forward. In Module 08 Drill B, that moves the training step from about 585 s to about 731 s.
  - TIS introduces bias, and C needs tuning; R3 must record and ship routing information from the inference side.
  - Monitoring: the KL between inference-side and trainer-side logprobs on the same tokens, and the MoE routing agreement rate.
- **Sources**: [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl), [R3](https://arxiv.org/abs/2510.11370).

## 6. Credit assignment (8 questions)

### 6.1 ★ What is the core problem of multi-agent credit assignment?

- **Conclusion**: the team gets one total reward. When every agent uses it as its own learning signal, the randomness caused by other agents' actions becomes noise in its own gradient.
- **Order of magnitude**: one way to quantify it: on a cooperative task where each agent's true contribution is known, compare the gradient variance of different estimators and the number of samples each needs to reach the same return. The more agents, the more noise.
- **Trade-off**: the finer the credit estimate, the more compute or counterfactual evaluations it needs.

### 6.2 ★★ What are difference rewards?

- **Conclusion**: $D_i = G(z) - G(z_{-i} \cup c_i)$: the total reward minus the total reward after "replacing agent i's action with a default action $c_i$", which keeps only the part that i itself contributed.
- **Order of magnitude**: each agent needs one extra counterfactual evaluation each time, so n agents need n.
- **Trade-off**: it assumes you can replay the environment or have a reward model; different choices of the default action give different biases.
- **Source**: [Wolpert & Tumer, An Introduction to Collective Intelligence](https://arxiv.org/abs/cs/9908014).

### 6.3 ★★ How is COMA's counterfactual baseline computed?

- **Conclusion**: a centralized critic estimates $Q(s,\mathbf{u})$. For agent a, hold the other agents' actions fixed and marginalize a's own action under its policy: $A^a = Q(s,\mathbf{u}) - \sum_{u'_a}\pi^a(u'_a\mid\tau^a)\,Q(s,(\mathbf{u}^{-a},u'_a))$.
- **Order of magnitude**: the critic's design lets it compute this baseline "in a single forward pass", with no extra environment interaction.
- **Trade-off**: the critic itself is hard to learn because the joint action space is large; for LLM agents, actions are text, so "all actions" cannot be enumerated at all.
- **Source**: [COMA](https://arxiv.org/abs/1705.08926).

### 6.4 ★★★ Why is the Shapley value fair, why is it expensive, and how is it approximated?

- **Conclusion**: the Shapley value is each agent's marginal contribution averaged over all joining orders, and it satisfies four axioms: efficiency, symmetry, zero for null contributors, and additivity.
- **Order of magnitude**: an exact computation evaluates $2^n$ coalitions, which is 1,024 for n = 10. It is usually estimated by Monte Carlo sampling of random permutations, with error roughly ∝ $1/\sqrt{\text{samples}}$.
- **Trade-off**: each coalition evaluation means running (part of) the task, which is expensive; the noise in the estimate also adds gradient variance.

### 6.5 ★★ What does a group baseline (GRPO-style) buy you in the multi-agent setting, and what are its limits?

- **Conclusion**: sampling the same task several times and using the group mean as the baseline lowers variance, but it cannot tell which agent in the group made the contribution.
- **Order of magnitude**: GRPO uses the group mean as the baseline and divides by the group's standard deviation, with no critic needed (DeepSeekMath).
- **Trade-off**: it removes only the variance that comes from "task difficulty" and leaves credit assignment between agents unsolved; it can be combined with difference rewards or counterfactual baselines.
- **Source**: [DeepSeekMath](https://arxiv.org/abs/2402.03300).

### 6.6 ★★ How do you assign credit over time in a multi-turn agent?

- **Conclusion**: with only a final reward, every turn shares the same advantage, and the longer the trajectory, the thinner the signal each step receives. Options include process rewards, per-turn value estimates, and counterfactual replay.
- **Order of magnitude**: one way to quantify it: fix the first t turns, resample only the later ones, and see how the variance of the final reward changes with t, which reveals the key turns.
- **Trade-off**: process reward models can be hacked; value models are hard to train on long contexts.

### 6.7 ★★★ What forms of reward hacking are specific to multi-agent RL?

- **Conclusion**:
  - "collusion" between agents, e.g. one agent helping another bypass a check;
  - pushing blame onto another agent;
  - exploiting holes in a judge agent;
  - single-agent hacking (e.g. editing tests) still exists as well.
- **Order of magnitude**: one way to quantify it: look for a scissors gap where "reward keeps rising but human evaluation or isolated tests don't", and check whether the correlation between agents' rewards rises abnormally.
- **Trade-off**: the evaluation environment has to be independent and read-only, and judge models need adversarial testing. The cost is higher evaluation cost and latency.

### 6.8 ★★★ How do you estimate the system cost of counterfactual credit assignment, and how do you cut it?

- **Conclusion**: the cost of methods like difference rewards and Shapley ultimately has to be converted into GPU-seconds. The biggest saving is forking from a shared prefix: a counterfactual only reruns the suffix after the fork point and reuses the prefix's KV directly.
- **Order of magnitude** (using Module 08 Drill B's trajectory: 40K tokens, 20 turns, DeepSeek-V3):
  - One full trajectory is about 11 PFLOPs (about 5.8 prefill, about 5.0 decode), about 18 GPU-seconds at 600 TFLOPS.
  - One full-trajectory counterfactual for each of 5 agents adds 5 × 18 ≈ 90 GPU-seconds, making the total cost 6×.
  - Forking at turn 10 and reusing the prefix KV, each counterfactual costs about 5.9 PFLOPs (about 10 GPU-seconds), about 54% of a full trajectory. The second half has longer contexts, so the saving is less than half.
- **Trade-off**:
  - Forking requires the environment to snapshot and restore at the fork point, e.g. the sandbox's filesystem state.
  - The later the fork point, the cheaper, but only the contribution of later actions can be assessed.
  - Shapley needs $2^n$ coalitions, so only sampled approximations are possible, and each sample pays as above.

## 7. System design (7 questions)

### 7.1 ★★★ Serve 10,000 concurrent coding agents on an H800 cluster, with prefix caching and PD disaggregation.

- **Conclusion**: the heart of this problem is tiered KV cache, cache-aware routing, and admission control for the miss tail. Decode approaches both the HBM capacity limit and the MLA attention compute limit; the size and the p99 of prefill are both set by "the fraction of full misses".
- **Order of magnitude**:
  - With cache hits, prefill needs about 300–550 GPUs (600/330 TFLOPS); with the cache completely gone, about 3,000. Including MLA up-projection, a full miss costs about 9.8× a hit.
  - Decode: about 20 streams of 35K per GPU, about 2,930 per EP144 instance, so 3–4 instances (54–72 nodes).
  - Queueing at the busy-time rate (600 TFLOPS): with 10% full misses, p99 TTFT ≤ 2 s requires prefill utilization of at most about 0.8. Deploy at ρ ≈ 0.55 to leave headroom for the optimistic 600 TFLOPS assumption, which also matches Day 6's 330 / 600.
  - Summed at deployment granularity, with prefill scaled by the 1.23 peak-to-mean ratio: about 145–235 nodes.
- **Trade-off**: move the full KV or only the increment between P and D; whether KV waiting on tools goes to HBM, host memory, or gets recomputed; local prefill stalls the whole instance under large EP; FP8 KV hits the compute wall; the blast radius of large EP.
- **Details**: [Module 08 Drill A](../modules/08_system_design/README.en.md#4-drill-a-serving-10000-concurrent-coding-agents-on-an-h800-cluster-prefix-caching--pd-disaggregation).

### 7.2 ★★★ Design an async agentic RL training system for a 600B-class MoE, including the sandbox cluster and weight sync.

- **Conclusion**: rollout and training are separated and run asynchronously; every token records its version, inference-side logprob, and MoE routing; the long tail is handled with partial rollout or interruptible generation; the bottleneck in weight sync is resharding, not bandwidth.
- **Order of magnitude**:
  - Each step trains 164M tokens. A 40K sequence costs 2.11× per token relative to 4K; adding the old/ref forward passes, that is about 731 s on 512 GPUs.
  - Per-GPU memory: parameters, gradients, and optimizer states about 25 GB, activations for a 40K sequence about 35 GB, so context parallelism or recomputation is needed.
  - About 3,100 trajectories in flight, needing the same number of sandboxes; decode needs 1 EP144 instance and prefill 2–4 EP32 instances.
  - Weight sync lands about 7 TB, with a per-GPU bandwidth lower bound of 0.44–0.8 s; public systems measure seconds to 30 s.
- **Trade-off**: the size of η, the bias from dropping trajectories, GRPO's group-level tail, train–inference mismatch, the memory taken by double-buffering.
- **Details**: [Module 08 Drill B](../modules/08_system_design/README.en.md#5-drill-b-an-async-agentic-rl-training-system-for-a-600b-class-moe-with-sandbox-cluster-and-weight-sync).

### 7.3 ★★★ Design a multi-tenant prompt caching API, covering cache tiers, billing, and isolation.

- **Conclusion**:
  - The cache key is a block-level hash chain that includes the tenant salt, model version, and tokenizer version.
  - Three tiers: HBM, host memory, SSD.
  - Eviction score = reuse probability × recompute cost / bytes.
  - Billing is by the number of blocks actually matched.
  - The salt closes only the cross-tenant-reuse class of side channel; the rest needs per-tenant queues, resource partitioning, and audits.
- **Order of magnitude**:
  - Sanity check first: 5,000 requests/s × 20K is 14× the whole-site average input in DeepSeek's Day 6, and the uncached part alone needs about 8,000 H800s.
  - One 14K prefix on DeepSeek-V3 is 0.98 GB; recomputing it costs 2.5 GPU-seconds, while reading it back takes about 20 ms. A 70% token hit rate saves only about 60% of the FLOPs.
  - If every hit were read remotely, it would need 4.9 TB/s of bandwidth, so hot prefixes must stay resident in HBM.
- **Trade-off**: isolation gives up dedup; a longer TTL raises storage cost; larger blocks make billing coarser.
- **Details**: [Module 08 Drill C](../modules/08_system_design/README.en.md#6-drill-c-a-multi-tenant-prompt-caching-api-cache-tiers-billing-isolation).

### 7.4 ★★★ Set an end-to-end latency SLO for a multi-agent orchestration platform, accounting for the critical path, fan-out, and timeouts.

- **Conclusion**: output tokens are the most expensive thing on the critical path; fan-out amplifies tail latency, into $p^k$ under independence. The fixes include quorum + deadline, hedged requests, passing deadlines down level by level, and propagating cancellation, and the design must finally be accepted by its end-to-end quantiles.
- **Order of magnitude**:
  - Under independence, the probability that all 5 subagents are within their own p90 is only 59%, and switching to "4 of 5" raises it to 92%; with pairwise correlation 0.5 the figures are 71% and 88%, so the gain drops from 33 to 17 percentage points.
  - Every 1,000 output tokens take about 25 s (TPOT 25 ms).
  - Monte Carlo check: cutting planning output and adding a quorum still leaves p90 at 106 s; also cutting subagent and aggregation output brings p90 down to 76 s.
- **Trade-off**: answer completeness against latency; the extra load from hedging, and a hedge sent to a replica without the prefix is a full miss; the retry budget.
- **Details**: [Module 08 Drill D](../modules/08_system_design/README.en.md#7-drill-d-end-to-end-latency-slo-for-a-multi-agent-orchestration-platform-critical-path-fan-out-timeouts).

### 7.5 ★★★ Online throughput suddenly drops 30%. How do you troubleshoot?

- **Conclusion**: first confirm the metric is real, then split the problem in two by asking "did GPU busyness change?": if the GPUs got idler, look at the demand side and upstream; if they are just as busy but produce fewer tokens, each token got more expensive. Then estimate, branch by branch, how much of the drop each one can explain at most.
- **Order of magnitude**:
  - A hit-rate drop from 80% to 60% can cost 23–29% of throughput.
  - One GPU slowed to 70% makes the whole EP group about 30% slower.
  - An MTP acceptance drop from 0.85 to 0.6 explains only 13.5%.
- **Trade-off**: `nvidia-smi` utilization only counts the time "some kernel is running" and says nothing about real efficiency; look at per-step time and MFU/MBU instead.
- **Details**: [Module 08 Drill E](../modules/08_system_design/README.en.md#8-drill-e-troubleshooting-a-sudden-30-drop-in-online-throughput).

### 7.6 ★★ Use Little's law to size an RL sandbox cluster.

- **Conclusion**: concurrent sandboxes = trajectory completion rate × trajectory lifetime.
- **Order of magnitude**: under Drill B's assumptions, the training step is about 731 s, so the completion rate is 4,096 / 731 ≈ 5.61/s; the lifetime is 557 s (500 s of decode + 19 tool calls × 3 s). Concurrent sandboxes ≈ 5.61 × 557 ≈ 3,100, plus a 10–20% warm pool.
- **Trade-off**: a larger warm pool starts faster but wastes more; stateful sandboxes cannot be migrated mid-trajectory. Underestimating the training step overestimates the sandbox count, so get the trainer side right first.

### 7.7 ★★ Design a KV cache storage service (Mooncake / 3FS style).

- **Conclusion**: build it as a distributed object store keyed by block hash, with three tiers (HBM / host memory / SSD), RDMA reads and writes, and a global index for the router to use.
- **Order of magnitude**:
  - 3FS reaches about 6.6 TiB/s aggregate read throughput across 180 nodes.
  - In DeepSeek's production service, 56.3% of input tokens hit the on-disk cache.
  - Mooncake reports that under real workloads it let Kimi handle 75% more requests.
- **Trade-off**: write-through or write-back; keep replicas or recompute on loss; the balance between block size and metadata volume.
- **Sources**: [Mooncake](https://arxiv.org/abs/2407.00079), [3FS](https://github.com/deepseek-ai/3FS).

## 8. Behavioral and project (6 questions)

### 8.1 ★ "Tell me about your co-design project."

- **Conclusion**: use the five-part structure: motivation → why it's hard → entry point → how you'll validate → expected trade-offs. State the confidentiality constraint at the start, and use wording like studying / exploring throughout.
- **What you can quantify**: put a number you can compute in your head into the motivation, e.g. "a full recompute costs about 10× a hit"; grade your evidence into three levels: paper results, simulation results, and hypotheses not yet validated.
- **Trade-off**: being specific makes you credible, but you must not leak the partner's details.
- **Details**: [Module 08 §9](../modules/08_system_design/README.en.md#9-how-to-talk-about-your-harnessinference-co-design-project-in-interviews).

### 8.2 ★★ "Tell me about a time you found your own experimental conclusion was wrong."

- **Conclusion**: tell it in STAR form, focusing on three things: how you found out, how you fixed it, and which check you added afterwards.
- **What you can quantify**: name the signal that made you suspicious, e.g. the measurement and the first-principles estimate differed by an order of magnitude, or the conclusion flipped with a different random seed; then say how large the gap was after the fix.
- **Trade-off**: don't invent a perfect story. The interviewer wants to see your habit of checking your own work.

### 8.3 ★★ "When research novelty and engineering delivery conflict, how do you decide?"

- **Conclusion**: define the success metric first, time-box the exploration, and use the smallest testable experiment to decide whether to continue.
- **What you can quantify**: e.g. "within two days, use the simulator to check whether this direction's upper bound exceeds 10%; drop it if it doesn't". The upper bound can come from an offline optimum or idealized assumptions.
- **Trade-off**: engineering too early wastes exploration time; never shipping means no one can use your results.

### 8.4 ★★ "What is the biggest contribution of your first-author paper? What would you change if you redid it?"

- **Conclusion**: state the contribution in one sentence, add one key number, then volunteer a limitation.
- **What you can quantify**: use the paper's real results, without rounding them into nicer numbers; say under what setting they hold.
- **Trade-off**: volunteering a limitation comes across as mature, but follow it immediately with how you plan to address it.

### 8.5 ★ "Why infra, and why our team?"

- **Conclusion**: connect the reason to your research question: agent workloads break the assumptions that inference and training systems were built on (many turns, tool waits, long tails), and that is exactly the problem infra teams have to solve.
- **What you can quantify**: give an example you can work out precisely, e.g. where the KV goes during a tool wait: keeping it in HBM crowds out capacity, offloading to DRAM restores in about 50 ms, and recomputing costs about 10×, so the key is timing offload and prefetch by tool duration.
- **Trade-off**: show enthusiasm, but don't speculate about the team's internal systems.

### 8.6 ★★★ "What do you do when you disagree with a senior engineer about a design?"

- **Conclusion**: align on assumptions first, since disagreements often come from different workload assumptions; then let estimates and small experiments do the talking; be ready to be proven wrong.
- **What you can quantify**: turn the disagreement into a measurable question, e.g. "is the full-miss fraction 2% or 20%?", and settle it by replaying traces.
- **Trade-off**: balance moving fast against getting it right. For disagreements that don't matter much, go with their approach for now and write it down.
