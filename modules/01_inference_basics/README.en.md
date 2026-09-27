[中文](README.md) · **English**

# Module 01: Inference First Principles — FLOPs, Bytes, and the Roofline

> 🕐 Suggested time: 2 hours ｜ 🎯 After this you can: given a model and a GPU, estimate in a minute the KV size per token, the maximum concurrency, whether decode is memory-bound or compute-bound, and the order of magnitude of TTFT; know where these estimates break under MoE, shared prefixes, incremental prefill, and multi-GPU communication; explain the difference between TTFT / TPOT / ITL / goodput, and use Little's law to work out how much memory you need ｜ 🧪 Lab: [`lab.py`](lab.py) ｜ 📖 Must-read primary sources: [All About Transformer Inference (How To Scale Your Model)](https://jax-ml.github.io/scaling-book/inference/), [Efficiently Scaling Transformer Inference (arXiv:2211.05102)](https://arxiv.org/abs/2211.05102)

## 0. One sentence

**LLM inference is two completely different workloads glued together**: prefill (processing the whole input at once) feeds in thousands of tokens at a time and is usually a compute-bound (limited by FLOPs) large matrix multiply; decode (generating output tokens one by one) produces just 1 token per sequence per step, yet must read the weights and the sequence's whole KV from HBM (the GPU's high-bandwidth memory), so it is usually memory-bound (limited by memory bandwidth). There is only one test: compare this computation's "FLOPs ÷ bytes" with the hardware's "peak FLOP/s ÷ bandwidth". Every system design that follows — batching, KV cache management, PD disaggregation, MLA, large-scale EP — is a trade-off around this ratio.

## 1. Why big labs care

DeepSeek's public production inference data ([inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)) has several numbers worth memorizing:

- On 8 × H800 nodes, **prefill does about 73.7k tokens/s (including cache hits) and decode about 14.8k tokens/s**;
- Of the 608B input tokens in 24 hours, **56.3%** hit the on-disk KV cache;
- The average KV cache length per output token is **4,989** tokens, and the average output speed each user sees is **20–22 tokens/s**.

Do not jump to "prefill is 5× faster than decode". The 73.7k includes hit tokens, which do no prefill computation at all. Roughly removing the global hit rate, the prefill each node actually computes is about 73.7k × (1 − 0.563) ≈ **32k tokens/s, about 2.2× decode**. And the two are not the same deployment: the prefill unit is 4 nodes (routed experts EP32), the decode unit 18 nodes (EP144); DeepSeek also does not say whether the 14.8k includes the speedup from MTP (multi-token prediction), which DeepSeek's [ISCA paper (arXiv:2505.09343)](https://arxiv.org/abs/2505.09343) reports raises generation TPS by 1.8×. **The first step in reading anyone's throughput number is to ask how it was measured.**

Now read it with Little's law (Section 8): each node outputs 14.8k tokens per second and each user gets about 21 per second, so each decode node has about 14.8k ÷ 21 ≈ **700 sequences** running at once. Why so many? 20 tokens/s is already fast enough to read, so why not make each sequence faster? This module answers these questions with two quantities — **FLOPs (floating-point operations) and bytes**. Agents make it sharper: each round of an agent inputs tens of thousands of tokens and outputs a few hundred ([Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) reports an input:output ratio of about 100:1), so TTFT (time to first token) decides how long every round waits; and how many agents you can serve at once depends on how much KV fits in memory.

## 2. How many FLOPs and bytes does one token cost

### 2.1 FLOPs

- **Linear layers**: each weight does one multiply-add per token, i.e., 2 FLOPs. So the forward pass costs about **2N FLOPs per token**, where N is the number of parameters that take part in matmuls: total parameters for dense models, **active parameters** for MoE models (37B for DeepSeek-V3, not 671B). The input embedding is a table lookup, not a matmul, so it is not in N; the output lm_head is a real matmul that decode computes for every token. Llama-3-8B's vocabulary has 128,256 × 4,096 ≈ 0.53B parameters, so in decode N ≈ 7.5B rather than 8.03B (codesign_sim counts the same way). This is C_forward ≈ 2N from [Kaplan et al. (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361).
- **Attention**: the new token's query takes a dot product with every key in the context (QKᵀ), then a weighted sum over every value (AV). Per layer per token that is about 2·ctx·d_attn + 2·ctx·d_attn = **4·ctx·d_attn**, where d_attn = n_heads × head_dim and ctx is the context length it can see.
- **Attention in prefill**: processing n tokens at once, under a causal mask token i sees only the first i tokens, so the average ctx ≈ n/2 and each layer costs about **2·n²·d_attn** in total.

When can attention not be ignored? For Llama-3-70B (N ≈ 70.6B, 80 layers, d_attn = 64 × 128 = 8192), the ratio of attention FLOPs to linear FLOPs per decoded token is 4·ctx·8192·80 / (2 × 70.6e9): about 15% at ctx = 8k and about 122% at ctx = 64k. **At long context, attention is the main cost.**

### 2.2 Bytes

- **Weights**: bytes per parameter (2 for BF16, 1 for FP8) × the parameters **used** in this step. A dense model reads its weights from HBM every step (only the looked-up rows of the input embedding table), **and weights are read only once no matter how many sequences are in the batch**. An MoE model reads only the selected experts each step, and a bigger batch selects more experts (Section 3.4).
- **KV cache**: with a standard decode kernel, every sequence reads its own whole KV at every step, so **KV reads grow linearly with the batch**. The exception is sequences that share a prefix: physically the KV is stored once, and a specialized kernel can also read it once (Section 3.5).

So one decode step of a dense model reads about weight bytes + B × ctx × KV bytes per token (B is the number of sequences in the batch).

## 3. Arithmetic intensity and the roofline

### 3.1 Definition and ridge point

**Arithmetic intensity** I = FLOPs ÷ bytes read from or written to memory, in FLOPs/Byte. The [Roofline model](https://dl.acm.org/doi/abs/10.1145/1498765.1498785) (Williams, Waterman, Patterson, CACM 2009) says:

```
attainable FLOPs = min(peak FLOPs, memory bandwidth × I)

 FLOPs ▲
       │            ┌──────────────── peak FLOPs (compute-bound region)
       │          ／
       │        ／   ← slope = memory bandwidth (memory-bound region)
       │      ／
       │    ／
       └──┴─────────┴──────────────▶ I (FLOPs/Byte)
                    ridge point = peak FLOPs ÷ memory bandwidth
```

The **ridge point** is where the two segments meet: below it, time is set by bandwidth; above it, by compute. The H100 SXM's BF16 dense peak is 989 TFLOPS (the 1,979 on NVIDIA's spec page includes sparsity) with 3.35 TB/s bandwidth, so the ridge ≈ **295 FLOPs/Byte**; FP8 doubles the compute and so doubles the ridge (about 591). Values for each GPU are in the lab output in Section 10.

### 3.2 Prefill vs decode: intensity ≈ "tokens processed at once"

A [B × d] activation times a [d × d′] BF16 weight: FLOPs = 2·B·d·d′, bytes ≈ 2·d·d′ (weights dominate when B ≪ d), so **I ≈ B**, the number of tokens in this matrix multiply.

| Phase | Tokens in one matmul | I (linear layers) | On H100 (BF16) |
|---|---|---|---|
| prefill of a 4k prompt, all uncached | 4,096 | ≈ thousands | far above 295, compute-bound |
| **incremental prefill**, only 128 new tokens after a cache hit | 128 | ≈ 128 | below 295, **memory-bound**, and it must also read the whole cached KV |
| decode, batch = 1 | 1 | ≈ 1 | memory-bound, over 99% of compute idle |
| decode, batch = 64 | 64 | ≈ 64 | still memory-bound |
| MoE decode (DeepSeek-V3), global batch = B | each routed expert gets only B × 8 / 256 | ≈ B / 32 | see Section 3.4 |

This is exactly what [Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677) means by "compute-intensive prompt computation" and "memory-intensive token generation", and what the [JAX scaling book](https://jax-ml.github.io/scaling-book/inference/) means by "during prefill, all matrix multiplications are basically always compute-bound". But in an agent workload with a high hit rate, **each round often prefills only a few hundred new tokens (the tool result), and then prefill is memory-bound too**. Example: Llama-3-8B on 1 H100 with the first 3,968 tokens cached, computing only 128 new tokens, takes about 3.5 ms of compute (MFU 60%), while reading the weights and 4k of KV once takes about 6.2 ms (MBU 80%). Three consequences:

1. chunked prefill (splitting a long prompt into chunks mixed with decode, see [Module 02](../02_inference_engines/README.en.md)) needs chunks of at least about ridge tokens (about 300 for H100 BF16) to use the compute;
2. piggybacking a small prefill onto a decode step is almost free, because that step reads the weights anyway;
3. at high hit rates TTFT is set by "(weights + cached KV bytes) ÷ bandwidth", not by FLOPs (Section 7).

**Mental-math example**: Llama-3-8B decoding on 1 H100, batch = 1, 4k context: each step reads about 15.5 GB (about 15.0 GB of weights without the embedding table, plus 0.54 GB of 4k KV), which takes about 4.6 ms even at the full 3.35 TB/s, i.e., at most about 215 tokens/s for one sequence; the step's linear-layer computation is only about 15 GFLOPs, about 15 µs at peak. They differ by 300×, which is the ideal room for batching **if KV can be ignored**; at a 4k context, KV reads cap what batching can recover at about 32× (Section 5).

### 3.3 Attention in decode: intensity = query heads served by each KV head

Attention differs from linear layers: **when sequences share no prefix, each sequence reads its own KV, and no batch size amortizes it**. Per layer, per context token:

- FLOPs = 4 × n_heads × head_dim (QKᵀ and AV)
- bytes = 2 (K and V) × n_kv_heads × head_dim × 2 B (BF16)

So I_attn = n_heads / n_kv_heads, **independent of the batch** (doubled with FP8 KV).

| Attention | Example | I_attn (BF16 KV) |
|---|---|---|
| MHA (multi-head attention, each query head has its own KV) | n_heads = n_kv_heads | 1 |
| GQA (grouped-query attention, several query heads share one KV group, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) | Llama-3-8B: 32 / 8 | 4 |
| GQA | Llama-3-70B: 64 / 8 | 8 |
| MLA (multi-head latent attention, all heads share one compressed latent vector), all 128 heads on one GPU | DeepSeek-V3: 128 heads share a 576-dim latent | ≈ 242 (derivation below) |
| MLA, attention split t ways by TP | each GPU computes only 128 / t heads but reads the full latent | ≈ 242 / t (about 60 at TP4, about 30 at TP8) |

The MLA derivation (the weight-absorbed decode form, using kv_lora_rank = 512, qk_rope_head_dim = 64, num_attention_heads = 128 from [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)): per layer, per context token, each of the 128 heads does one 576-dim dot product and one 512-dim weighted sum, FLOPs = 128 × 2 × (576 + 512) = 278,528; bytes read are 576 × 2 = 1,152. I ≈ 242, already close to the H800 ridge (same as H100, about 295). That is why DeepSeek's [FlashMLA](https://github.com/deepseek-ai/FlashMLA) reports both "up to 3000 GB/s in the memory-bound configuration" and "660 TFLOPS in the compute-bound configuration": **MLA turns decode attention from a pure bandwidth problem into something close to a compute problem.**

**The 242 assumes DP attention**: one GPU holds all 128 heads, and attention is split across GPUs by request rather than by head. If you split attention with TP as for an ordinary dense model, the latent is shared by all heads and cannot be split by head, so every GPU must store a full copy of the KV and read the whole latent while computing only 128 / t heads. The deployment units in the DeepSeek-V3 technical report use TP4 + SP with DP for attention ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Section 3.4), while the production overview says "MLA / Shared Expert DP32 (prefill), DP144 (decode)". The cost of DP attention: non-routed weights such as attention and the shared expert are stored on every GPU. Derived from config.json, that is about 17B parameters (attention about 11.4B, shared experts about 2.6B, dense FFN of the first 3 layers about 1.2B, vocabulary about 1.9B).

### 3.4 MoE: each expert gets only B·k/E tokens

"Linear-layer I ≈ B" holds only for dense models. Each routed expert of an MoE is a separate small FFN that processes only the tokens routed to it in a step: with global batch B, each expert gets on average **B·k/E** tokens (k experts chosen per token, E experts in total). DeepSeek-V3 has k = 8 and E = 256 (num_experts_per_tok and n_routed_experts in [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)), so the expert GEMMs have **I ≈ B / 32**.

- To push the expert GEMMs past the H100 / H800 BF16 ridge (about 295), each step needs about 295 × 32 ≈ **9,400 tokens** globally.
- With FP8 weights and FP8 compute: halving weight bytes doubles I (≈ B / 16), and the FP8 ridge also doubles (about 591), so the two cancel and it still takes about 9,400 tokens.
- Attention and the shared expert are dense, with I ≈ B (the per-GPU B); when all experts are read and KV is ignored, the whole-model average I is about B × 37 / 671 ≈ B / 18.

This is the first-principles reason DeepSeek spreads decode over EP144 (production) and EP320 (technical report): **one GPU cannot gather thousands of concurrent sequences, so requests on many GPUs must share the same experts to pool each expert's batch.** Using the estimate from Section 1: a decode unit of 18 nodes with about 700 sequences each runs about 12.7k sequences, so each routed expert gets about 12.7k × 8 / 256 ≈ 400 tokens per step (not counting MTP), already past the BF16 ridge. The ISCA paper also notes that MTP, by predicting several tokens per step, "increases the inference batch size, which is crucial for boosting EP computational intensity". On the other hand, when the ISCA paper estimates the communication floor it assumes each device holds one expert and processes about 32 tokens at a time, and says this count balances compute-to-memory ratio against communication latency: **large EP also trades intensity against latency; a bigger batch is not always better.**

### 3.5 Shared-prefix attention: many sequences reading the same KV

Section 3.3's "each reads its own KV" breaks when sequences share a prefix, and shared prefixes are exactly this project's core scenario: GRPO samples a group of answers for the same prompt, multiple subagents share a system prompt and tool definitions, requests in the same project share repository context.

[Hydragen (arXiv:2402.05099)](https://arxiv.org/abs/2402.05099) and FlashInfer's [cascade attention](https://flashinfer.ai/2024/02/02/cascade-inference.html) split attention into a "shared prefix" part and a "unique suffix" part: for the prefix, the queries of all sequences are stacked into one matrix and multiplied against a single copy of the prefix KV, which is read only once. The intensity of the prefix part becomes **n_seq × n_heads / n_kv_heads**.

Example (derived): Llama-3-8B, 16 samples share a 4k prompt and each generates 512 tokens. The naive attention I = 4; once the prefix is read only once, the overall I = 4 × 16 × (4096 + 512) / (4096 + 16 × 512) = **24**. Hydragen reports up to 32× end-to-end throughput on CodeLlama-13b, and FlashInfer up to 31× over vLLM's PagedAttention kernel. **When the harness makes requests share a prefix, it saves not only memory and prefill but also directly raises decode's arithmetic intensity.**

## 4. How big is the KV cache

$$\text{KV bytes/token} = \begin{cases} 2 \times L \times n_{kv} \times d_{head} \times b & \text{MHA / GQA} \\ L \times (d_c + d_h^R) \times b & \text{MLA} \end{cases}$$

L is the number of layers, b the bytes per element (2 for BF16), d_c MLA's kv_lora_rank, and d_h^R the decoupled RoPE key dimension. The [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) states that only the compressed latent vector and the decoupled key need to be cached during generation (d_c = 512, d_h^R = 64).

| Model | Attention | Structure (from official config / paper) | KV bytes/token (BF16) |
|---|---|---|---|
| Llama-3-8B | GQA | 32 layers, 8 KV heads, head_dim 128 | 2 × 32 × 8 × 128 × 2 = 131,072 B = **128 KiB** |
| Llama-3-70B | GQA | 80 layers, 8 KV heads, head_dim 128 | 2 × 80 × 8 × 128 × 2 = 327,680 B = **320 KiB** |
| Qwen3-32B | GQA | 64 layers, 8 KV heads, head_dim 128 | 2 × 64 × 8 × 128 × 2 = 262,144 B = **256 KiB** |
| DeepSeek-V3 | MLA | 61 layers, 512 + 64 | 61 × 576 × 2 = 70,272 B ≈ **68.6 KiB** |

Structure sources: Llama 3 from [the paper's Table 3 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783), Qwen3-32B from its [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json), DeepSeek-V3 from its [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json).

A few counterfactuals to remember how much GQA / MLA save (all derived from the formula):

- If Llama-3-70B used MHA (64 KV heads), it would be 2 × 80 × 64 × 128 × 2 B = **2.5 MiB** per token, 8× the GQA size.
- If DeepSeek-V3's 128 heads cached full K (128 + 64 = 192 dims) and V (128 dims) as in MHA, it would be 61 × 128 × 320 × 2 B ≈ **4.77 MiB** per token, about 71× MLA. DeepSeek-V2's abstract puts it this way: compared with DeepSeek 67B, the KV cache shrinks by 93.3% and maximum generation throughput rises to 5.76× ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)).

**One 128k-token Llama-3-70B sequence needs 128k × 320 KiB = 40 GiB of KV**, half an H100.

## 5. How batch size changes decode

Putting Sections 2 and 3 together, the intensity of one decode step (B sequences, each with context ctx, BF16, no prefix shared between sequences):

$$I(B) = \frac{2N \cdot B + 4 L\, d_{attn}\, ctx \cdot B}{W(B) + B \cdot ctx \cdot kv}$$

N is the active parameter count and W(B) the **weight bytes actually read** in this step: for dense models W(B) = 2N, independent of B; for MoE models W(B) = non-routed weights + routed weights × (1 − (1 − k/E)^B), i.e., the more experts the batch covers, the more weights are read, until every expert is read once B is large enough.

- When B is small, weights dominate the denominator and dense I ≈ B: **adding batch is almost free throughput**, because weights are read once.
- When B is large, KV dominates the denominator and I approaches an **upper limit** (2N + 4·L·d_attn·ctx) / (ctx·kv) that depends only on the context length and attention structure, not on the batch.

For Llama-3-8B (derived, BF16, N ≈ 7.5B): at context 128 the limit is about 900, far above the H100's 295, so with enough batch it becomes compute-bound; at context 4k the limit is only about 32, **so no batch size gets it out of memory-bound**; at context 32k the limit is about 7.5, of which 4 comes from attention itself (GQA's 32 / 8) and about 3.5 from linear-layer FLOPs spread over KV bytes.

Conclusion: **when sequences share no prefix, decode on GQA models at real agent context lengths (thousands to tens of thousands of tokens) is memory-bound, and batching cannot rescue it**. To make decode use the compute, either raise attention's intensity (MLA with DP attention, cascade attention over shared prefixes), cut bytes per token (FP8 / lower-precision KV), or verify more than one token per step (speculative decoding, see [Module 02](../02_inference_engines/README.en.md)).

The time model (this is what `step_time_s` in the library does):

$$t_{step} = \max\left(\frac{\text{FLOPs}}{n_{gpu} \cdot \text{peak} \cdot \text{MFU}},\ \frac{\text{bytes}}{n_{gpu} \cdot \text{bandwidth} \cdot \text{MBU}}\right)$$

MFU (model FLOPs utilization) and MBU (model bandwidth utilization) are the achievable fractions of peak compute and bandwidth. In decode, TPOT ≈ t_step and node throughput ≈ B / t_step. While weight reads dominate, t_step barely changes with B, so **throughput grows linearly with B while TPOT stays almost flat**; once KV reads dominate, t_step grows linearly with B and throughput stops growing.

**What this model leaves out** (interviewers will ask):

1. **Communication**. TP does 2 all-reduces per layer (after attention and after the MLP), so one decode step of Llama-3-70B has 160 collectives; with ring all-reduce each GPU sends and receives about 2(n − 1)/n × B × d_model × 2 bytes per collective. For 70B, TP8, B = 40 that totals about 180 MB per GPU per step, only about 0.4 ms at 450 GB/s one-way NVLink, so bandwidth is not the issue, but the fixed latency of 160 collectives adds up. MoE EP does two all-to-alls (dispatch and combine): DeepSeek's ISCA paper assumes each token goes to 8 routed + 1 shared expert, FP8 dispatch, BF16 combine, hidden size about 7K, 32 tokens per device, and 400G IB at 50 GB/s, giving (1 byte + 2 bytes) × 32 × 9 × 7K ÷ 50 GB/s = 120.96 µs per layer, 241.92 µs per layer under dual micro-batch overlap, and **a 14.76 ms TPOT floor over 61 layers that has nothing to do with HBM bandwidth**.
2. **Fixed overheads**: kernel launch, scheduling, sampling. They matter most when the batch is small and a step is only a few milliseconds (CUDA Graph exists to cut this).
3. **The roofline is an upper bound, not a predictor**. MFU / MBU depend heavily on GEMM shapes, kernel fusion, and batch bucketing. Small-batch matrix × vector work almost never reaches peak FLOP/s, so "I crosses the ridge" is necessary but not sufficient for compute-bound. When estimating, try at least 2–3 MFU / MBU settings and check that the conclusion holds.

A more realistic form is t_step ≈ max(compute, memory) + communication that is not overlapped + fixed overheads. In this project's `codesign_sim`, `step_time_s` computes only the first two terms by default (communication must be passed explicitly through the `comm_s` argument), and this module's lab uses only those two; every conclusion should say so.

## 6. Memory budget and maximum concurrency

$$\text{max concurrency} \approx \left\lfloor \frac{n_{gpu} \cdot \text{HBM} \cdot (1 - r) - \text{weight bytes}}{ctx \times \text{KV bytes/token}} \right\rfloor$$

r is the fraction reserved for activations, temporary buffers, and fragmentation (10% in this project). **This formula is an optimistic upper bound**: it assumes the n_gpu GPUs form one model-parallel group, weights are stored once, and KV is split evenly across all GPUs. Real deployments must be computed **per GPU**: usable memory on the GPU − that GPU's weight shard − reserve, divided by that GPU's KV per sequence. Things that make the per-GPU result worse:

- TP larger than the number of KV heads (e.g., Llama-3-70B has only 8 KV heads, so at TP16 there are not enough KV heads to split), or TP over MLA: KV is duplicated across GPUs;
- DP attention: non-routed weights such as attention on every GPU;
- EP: redundant experts (DeepSeek places 32 extra copies of high-load experts);
- PP: each GPU holds weights and KV for only some layers, but filling the pipeline needs more sequences in flight;
- workspace, the CUDA Graph memory pool, and wasted tail blocks of paged KV.

Three examples to do in your head (HBM with 1 GB = 1e9 B):

| Deployment | Weights | Left for KV | KV per token | Max concurrency at 8k context | Max concurrency at 32k context |
|---|---|---|---|---|---|
| Llama-3-8B, 1 × H100 (aggregate formula) | ≈ 16.1 GB | 80 × 0.9 − 16.1 ≈ 55.9 GB | 128 KiB | ≈ 52 | ≈ 13 |
| Llama-3-70B, 8 × H100 (aggregate formula) | ≈ 141 GB | 640 × 0.9 − 141 ≈ 435 GB | 320 KiB | ≈ 162 | ≈ 40 |
| DeepSeek-V3, 32 × H800, FP8 weights, DP attention + EP32 (per GPU) | ≈ 40 GB per GPU | 72 − 40 ≈ 32 GB per GPU, about 1,024 GB in total | 68.6 KiB | ≈ 1,779 | ≈ 444 |

How the DeepSeek-V3 row is derived: about 17.0B non-routed parameters on every GPU; 256 routed experts plus 32 redundant copies spread over 32 GPUs gives 9 experts per GPU per layer (matching the production overview's "each GPU handles 9 routed experts"), each expert has 3 × 7168 × 2048 ≈ 44M parameters, about 23B over 58 MoE layers; FP8 approximated as 1 byte per parameter. **With the same 32 GPUs, the aggregate formula in table [3] of Section 10 gives about 2,836 sequences of 8k, an overestimate of about 1.6×**, because the non-routed weights are copied 32 times and there are redundant experts.

Takeaways:

1. **Memory keeps the batch below the ridge.** 70B at 8k context fits at most about 162 sequences, well short of 295, and Section 5 showed the limit is lower still at long context.
2. **4× the context, ¼ the concurrency.** As agent contexts keep growing, the number of sessions one machine can serve falls in proportion.
3. DeepSeek-V3's FP8 weights are about 671 GB, so **an 8 × 80 GB machine cannot even hold the weights**. That is one reason it must be deployed across machines and use EP to spread experts over many GPUs; DeepSeek's production units are 4 nodes for prefill and 18 nodes for decode ([inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)).

## 7. Metrics: TTFT / TPOT / ITL / throughput / goodput / SLO

| Metric | Definition | Mainly set by | What it means for agents / RL |
|---|---|---|---|
| **TTFT** (time to first token) | from submitting the request to receiving the first output token | queueing + prefill (of the part not in cache) | an agent waits one TTFT every round; dozens of rounds in series make it most of the task latency |
| **TPOT** (time per output token) | (end-to-end latency − TTFT) ÷ (output tokens − 1), one value per request | decode step time, i.e., batch and context length | decides how long long outputs (reasoning, code) take |
| **ITL** (inter-token latency) | the gap between two consecutive output tokens, one sample per gap | same, plus stalls when prefill cuts in | look at p99 ITL to catch "occasional hiccups" |
| **Throughput** | tokens output (or processed) per second by the whole system | batch size, MFU / MBU | what RL rollout cares about most; per-user speed × concurrency |
| **Goodput** | the maximum request rate sustainable while meeting an SLO attainment target | scheduling policy, PD disaggregation | throughput that misses the SLO does not count |
| **SLO** (service level objective) | e.g., "TTFT p99 < 2 s and TPOT p99 < 50 ms" | set by the business | the bar for goodput |

Easy-to-confuse points:

- NVIDIA's [benchmarking article](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) treats TPOT as another name for ITL, with the formula (e2e − TTFT) / (output tokens − 1). The [Anyscale docs](https://docs.anyscale.com/llm/serving/benchmarking/metrics) distinguish the aggregation: **average TPOT weights every request equally, average ITL weights by tokens**, so they differ when many requests are long. Ask which convention is meant in an interview.
- Goodput has no single convention either. [DistServe (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) uses **per-GPU goodput**: the maximum request rate each GPU can sustain while meeting an SLO attainment goal (the paper's example is 90% of requests meeting both TTFT and TPOT). It is a rate and must be stated with the attainment percentage, since 100% attainment is impossible. The Anyscale docs define goodput as "the percentage of requests meeting all SLOs", a percentage closer to SLO attainment. State the unit before you talk about goodput.
- **TTFT mental math**: TTFT ≈ queueing + max(FLOPs ÷ (n_gpu × peak × MFU), bytes ÷ (n_gpu × bandwidth × MBU)) for the uncached part of prefill, with bytes = weights + reading and writing the whole KV. Llama-3-8B, 4k prompt, 1 × H100, MFU 60%: (2 × 6.98e9 × 4096 + 2 × 4096² × 4096 × 32) ≈ 6.2e13 FLOPs → about **0.10 s**, compute-bound (6.98B are the parameters without the input embedding and lm_head; prefill computes lm_head only at the last position). With half the prefix cached, only the second half is computed (the second half's tokens see a longer context, so the attention term does not halve exactly). With 3,968 tokens cached and only 128 computed, compute takes about 3.5 ms but reading the weights and 4k of KV takes about 6.2 ms: **at high hit rates, TTFT is set by bandwidth** (estimated here by reading the full weights once, slightly conservative).
- This estimate is a **lower bound**. Production TTFT adds: prefix-cache lookup and loading KV back from DRAM / SSD / remote, TP / PP communication, waiting while a chunked prefill is interleaved with decode, sampling, network, and detokenization. The first output token comes straight from the logits at the last position of the prefill forward pass; there is no separate "first decode step".

## 8. Little's law: the identity linking concurrency, throughput, and latency

[Little's law](https://pubsonline.informs.org/doi/abs/10.1287/opre.9.3.383) (Little, Operations Research 1961): in steady state, **the average number of items in a system L = arrival rate λ × average time each item stays W**. It makes no distributional assumptions and is the most useful formula in capacity planning.

Three uses:

1. **Request concurrency**: 20 requests arrive per second and each stays 15 s on average, so there are 300 requests in the system on average and you need KV for 300 sequences.
2. **Token form**: in decode, the number of sequences in the batch B = throughput (tokens/s) × TPOT. For 5,000 tokens/s, if TPOT sits exactly at 40 ms you need 200 sequences running at once; a lower TPOT needs fewer. TPOT itself changes with B (Section 5), so what you really want is a B with B / TPOT(B) ≥ 5,000 and TPOT(B) ≤ 40 ms, then check memory with Section 6. Section 1's "about 700 sequences per DeepSeek decode node" is computed this way.
3. **Agent form**: if the KV is pinned (kept in HBM) during tool calls, W must include tool wait time. With 2 new sessions per second, 30 s of generation and 60 s of tool waits per session, on average 2 × (30 + 60) = 180 sessions hold KV, 2/3 of them waiting on tools. **Tool waits multiply memory occupancy by 3×** — this is where the magnitudes in Section 3.2 "keep the KV or drop it" of [Module 00](../00_map/README.en.md) come from, and it is the decision exercise (d) asks you to make.

**Plan capacity with percentiles, not just the mean.** Little's law gives only the mean: provisioning memory for the mean means it falls short half the time, and the engine keeps preempting. If sessions arrive as a Poisson process, the number in the system follows a Poisson distribution with mean L (the steady-state result for an M/G/∞ queue, independent of how long each session stays), so p99 is about L + 2.33√L. For L = 180 that is about 211, 17% more. Bursty arrivals or widely varying KV sizes per session make the swings larger. Also, context grows turn by turn within a session: with linear growth, a session's average KV is about half its final size. Table [6b] of the lab uses the full 32k throughout, a conservative estimate.

## 9. Numbers you must know by heart

| Quantity | Typical value | How to derive it | Source |
|---|---|---|---|
| Forward FLOPs / token | 2N (active parameters for MoE) | one multiply-add per weight | [Kaplan et al.](https://arxiv.org/abs/2001.08361) |
| H100 SXM BF16 dense peak / bandwidth | 989 TFLOPS / 3.35 TB/s | the spec page's 1,979 is sparse, ÷2 | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| H100 ridge point | BF16 ≈ 295 FLOPs/B, FP8 ≈ 591 | peak ÷ 3.35e12 | derived |
| A100 ridge point | ≈ 153 FLOPs/B | 312e12 ÷ 2.039e12 | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| Dense decode linear-layer intensity | ≈ batch size | weights read once | derived |
| MoE routed-expert intensity | ≈ B·k/E (B/32 for DeepSeek-V3; about 9,400 tokens to cross the BF16 ridge) | each expert gets only B·k/E tokens | derived, k and E from [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| Decode attention intensity | n_heads / n_kv_heads (MLA with DP attention ≈ 242, ÷ t under TP t) | each sequence reads its own KV | derived |
| When incremental prefill turns compute-bound | new tokens ≳ ridge (about 300 for H100 BF16) | I ≈ tokens processed at once | derived |
| Llama-3-8B / 70B KV | 128 / 320 KiB/token | 2·L·n_kv·d·2 | [Llama 3 paper](https://arxiv.org/abs/2407.21783) |
| DeepSeek-V3 KV | ≈ 68.6 KiB/token | 61 × 576 × 2 | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| 8B single-GPU batch = 1 decode | ≈ 215 tokens/s at full bandwidth; ≈ 172 tokens/s at MBU 0.8 | 3.35 TB/s ÷ 15.5 GB (weights without the embedding table) | derived |
| TTFT of an 8B 4k prompt | ≈ 0.10 s (1 × H100, MFU 60%) | 6.2e13 ÷ (989e12 × 0.6) | derived |
| DeepSeek production prefill / decode per node | 73.7k (incl. hits; about 32k after removing the 56.3% hits) / 14.8k tokens/s | production statistics (8 × H800) | [DeepSeek inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| DeepSeek-V3 TPOT floor set by EP communication | ≈ 14.76 ms (400G IB) | 61 layers × 2 × 120.96 µs | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |

The full table is in [Numbers you must know](../../docs/numbers-cheatsheet.en.md).

## 10. Lab

```bash
.venv/bin/python modules/01_inference_basics/lab.py
# or
make lab N=01
```

It finishes in under a second on an M1 laptop and prints 6 groups, 8 tables ([5] and [6] have two each). Below is real output with the Llama-3-8B table [5] omitted:

```text
[1] roofline（dense BF16；@I=x 表示 intensity 为 x 时能拿到的 TFLOPS）
GPU       BF16 TF  FP8 TF  HBM GB  TB/s  ridge  @I=1  @I=8  @I=64  @I=242
--------  -------  ------  ------  ----  -----  ----  ----  -----  ------
A100-80G      312       -      80  2.04    153     2    16    130     312
H100-SXM      989    1979      80  3.35    295     3    27    214     811
H800-SXM      989    1979      80  3.35    295     3    27    214     811
H20           148     296      96  4.00     37     4    32    148     148
B200         2250    4500     180  8.00    281     8    64    512    1936

[2] KV cache（BF16）
model        attn  B/token  KiB/token  32k seq GiB  128k seq GiB  vs 70B
-----------  ----  -------  ---------  -----------  ------------  ------
llama3-8b     gqa  131,072      128.0         4.00          16.0   0.40x
llama3-70b    gqa  327,680      320.0        10.00          40.0   1.00x
qwen3-32b     gqa  262,144      256.0         8.00          32.0   0.80x
deepseek-v3   mla   70,272       68.6         2.14           8.6   0.21x

[3] 显存预算的理想上界（HBM × 0.9 − 权重；权重在所有卡上只存一份、KV 均匀切分；KV 为 BF16）
model             deploy  W bytes  weights GB  KV GB  KV tokens  seqs@8k  seqs@32k
-----------  -----------  -------  ----------  -----  ---------  -------  --------
llama3-8b     1xH100-SXM        2          16     56      0.43M       52        13
qwen3-32b     2xH100-SXM        2          66     78      0.30M       36         9
llama3-70b    8xH100-SXM        2         141    435      1.33M      162        40
deepseek-v3   8xH800-SXM        1         671    放不下          -        -         -
deepseek-v3  32xH800-SXM        1         671   1633     23.24M     2836       709

[4] decode arithmetic intensity（FLOPs/Byte，BF16 权重与 KV；H100 ridge = 295）
case                   B=1   B=8  B=64  B=256  B=1024  B=4096    B→∞
---------------------  ---  ----  ----  -----  ------  ------  -----
llama3-8b ctx=128      1.0   8.0  60.0  199.9   479.5   737.2  898.6
llama3-8b ctx=4096     1.1   7.1  22.2   28.8    31.1    31.7   32.0
llama3-70b ctx=4096    1.1   8.0  42.6   79.4   101.3   108.8  111.6
llama3-70b ctx=32768   1.5   8.0  17.4   19.9    20.7    20.9   20.9
deepseek-v3 ctx=4096   1.9   3.5   7.6   25.7    88.8   230.5  492.4
deepseek-v3 ctx=32768  8.3  14.6  30.6   83.4   174.1   239.1  273.1

[5] decode 步时间：llama3-70b on 8xH100-SXM, ctx=32768（MFU 0.6 / MBU 0.8；显存最多 40 条序列）
batch  fits  GB/step     I   bound  TPOT ms  tok/s
-----  ----  -------  ----  ------  -------  -----
1       yes    149.7   1.5  memory     6.98    143
8       yes    224.9   8.0  memory    10.49    763
32      yes    482.6  14.9  memory    22.51  1,422
40      yes    568.5  15.8  memory    26.52  1,509
64       NO    826.2  17.4  memory    38.54  1,661

[6a] TTFT ≈ prefill 未命中部分（不含排队；MFU 0.6）
deploy             prompt  hit 0% s  hit 50% s  hit 90% s  hit 99% s
-----------------  ------  --------  ---------  ---------  ---------
llama3-8b 1xH100    4,096     0.104      0.054      0.011      0.006
llama3-8b 1xH100   16,384     0.504      0.282      0.061      0.007
llama3-8b 1xH100   65,536     3.439      2.194      0.515      0.053
llama3-70b 8xH100   4,096     0.123      0.063      0.013      0.007
llama3-70b 8xH100  16,384     0.547      0.292      0.061      0.007
llama3-70b 8xH100  65,536     3.076      1.834      0.414      0.043

[6b] Little's law：2 会话/s（泊松到达），生成 30 s，32k 上下文，llama3-70b，KV 全程 pin 住
tool_s  sessions (mean)  sessions p99  KV GB mean  KV GB p99  8xH100 nodes (mean)
------  ---------------  ------------  ----------  ---------  -------------------
0                    60            79         644        848                  1.5
60                  180           212       1,933      2,276                  4.4
300                 660           720       7,087      7,731                 16.3
```

How to read it:

1. **Table [1]**: H20's ridge is only 37, so at intensity = 64 it already gets its full 148 TFLOPS, while H100 gets only 214 / 989. Decode (intensity from single digits to a few dozen) is therefore far more "relatively efficient" on H20, and prefill far slower. B200's ridge (281) is close to H100's (295): compute and bandwidth grew in proportion, so decode is still memory-bound.
2. **Tables [2], [3]**: DeepSeek-V3's per-token KV is 0.21× Llama-3-70B's. Table [3] is **the aggregate formula's ideal upper bound**: 8 × H800 cannot even hold DeepSeek-V3's FP8 weights; for 32 GPUs (the size of DeepSeek's prefill unit) the aggregate formula gives about 2,836 sequences of 8k, while the real per-GPU computation with DP attention + EP32 gives only about 1,779 (Section 6), because the non-routed weights are copied 32 times.
3. **Table [4]**: the "B→∞" column is the upper limit from Section 5. Llama-3-8B at 4k context tops out at 32.0 and Llama-3-70B at 32k at 20.9, both far below 295, **so no batch size gets them out of memory-bound**. The DeepSeek-V3 rows use Section 5's W(B) in the denominator: a small batch reads only the selected experts, a large batch reads all of them. At 4k context the limit is 492.4, above the ridge, but the weights are 671B, so at B = 4096 the whole-model intensity is still only 230.5; from the per-expert view of Section 3.4, it takes about 9,400 concurrent tokens for the expert GEMMs to cross the ridge, which is exactly what large-scale EP is for. At 32k context DeepSeek-V3's limit is only 273.1, so with BF16 KV it too falls back to memory-bound.
4. **Table [5]**: as batch grows from 1 to 40, throughput rises from 143 to 1,509 tokens/s (10.6×) and TPOT from 6.98 ms to 26.52 ms (3.8×); at 64 sequences memory no longer fits (fits = NO). At 32k context each sequence's KV reads dominate, so TPOT rises noticeably; with short contexts, where weight reads dominate, TPOT stays much flatter. None of these times include TP all-reduces (Section 5).
5. **Table [6a]**: a 50% hit does not save 50%. Llama-3-70B's 64k prompt drops from 3.076 s to 1.834 s (about 40% saved), because the second half's tokens see a longer context; a 90% hit brings it to 0.414 s and a 99% hit to 0.043 s. A 4k prompt with a 99% hit computes only a few dozen tokens, and its TTFT of about 6–7 ms is almost all weight-reading time: **here prefill is memory-bound**.
6. **Table [6b]**: as tool waits go from 0 to 60 s, the mean pinned KV grows from 644 GB to 1,933 GB, all the KV space of 4.4 8 × H100 machines; at 300 s it takes 16.3. The mean is not enough: with Poisson arrivals, the 60 s row's p99 is 212 sessions and 2,276 GB, about 18% above the mean; the fewer the sessions, the larger the relative margin (79 vs 60 in the first row). In the other direction, with context growing turn by turn the average occupancy roughly halves (Section 8).

Change a parameter and watch:

- In `table_kv`, change `kv_bytes_per_token(m)` to `kv_bytes_per_token(m, 1)` (FP8 KV): every sequence shrinks by half. Then add `kv_dtype_bytes=1` to `decode_bytes` in `intensity` and change `kv_bytes_per_token(cfg)` to `kv_bytes_per_token(cfg, 1)` in `intensity_limit`, and watch the limits in table [4] double.
- Replace `"H100-SXM"` with `"H20"` in `DEPLOYMENTS` and `table_ttft`: each GPU has 16 GB more memory and higher bandwidth, but prefill is compute-limited and TTFT gets about 989 / 148 ≈ 6.7× slower. Then look at the 99%-hit column: a 4k prompt computes only a few dozen tokens, which is already memory-bound on H100 but still compute-bound on H20 (its ridge is only about 37), so H20 is still slower.
- In `main()`, change the second `table_decode_steps` context from 32768 to 4096: about 8× more sequences fit in memory and intensity rises clearly.

## 11. Exercises

Implement four functions in [`exercise.py`](exercise.py), using only `codesign_sim`'s public interface (`ridge_point`, `kv_bytes_per_token`, `prefill_flops`, `decode_flops`, `decode_bytes`, `step_time_s`). The first three turn formulas into code; the fourth requires a judgment:

| Function | What it does | Section |
|---|---|---|
| `max_concurrency` | after holding all weights, how many sequences of a given length fit in KV (the optimistic aggregate bound); return 0 if the weights do not fit | Section 6 |
| `critical_batch` | the smallest batch at which decode intensity reaches the ridge point; `None` if not reached by `max_batch`; FP8 weights / KV must be passed through | Section 5 |
| `estimate_ttft` | queueing + the uncached part of prefill, with bytes counting weights and the whole KV; even on a full hit, at least 1 token must be computed | Section 7 |
| `tool_wait_kv_action` | during a tool call, whether to keep this KV, offload it, or drop it and recompute, plus the extra latency when the tool returns | Section 8, [Module 00](../00_map/README.en.md) Section 3.2 |

The cost model of `tool_wait_kv_action`: assume the instance's concurrency is limited by KV memory; then holding a share of the node's KV capacity for tool_s seconds loses tool_s × share "node-seconds" of throughput, while recomputing occupies the node's compute for recompute_s seconds. Both are in the same unit and compare directly: **keep if tool_s × share ≤ recompute_s**. If keeping does not pay, compare "time to bring it back" with "time to recompute". The model's limits: when the instance is not short of memory, keeping costs almost nothing; offload bandwidth is shared with other requests; tool latency is random, so the TTL should be set from its distribution rather than its mean (which is what [Continuum](https://arxiv.org/abs/2511.02230) does).

```bash
make module N=01                         # test your implementation (it fails until you finish; that is expected)
INFRA_SOLUTION=1 make module N=01        # run the same tests against the reference solution.py
```

Three follow-up experiments once you are done:

1. Use `critical_batch` to compare BF16 KV and FP8 KV (`kv_dtype_bytes=1`): for Llama-3-8B at a 512-token context with `max_batch=100000`, BF16 KV returns `None` and FP8 KV returns 826. **Halving KV bytes doubles the limit, and only then does it cross the ridge.**
2. Use `tool_wait_kv_action` to find the "keep" threshold: about 62 s for Llama-3-70B, 8 × H100, 64k context; only about 11 s for Llama-3-8B, 1 × H100, 4k context. Why is it less worth keeping for the smaller model with the shorter context? (Hint: recomputing is cheap, and 4k of KV is not small on a single GPU running 8B.)
3. Lower `offload_gbps` from 512 down to 1 and see when 70B / 64k switches from offload to recompute, then compare with the "break-even bandwidth" in Section 4.2 of [Module 05](../05_cache_prefill/README.en.md).

## 12. Self-check

1. Is the H100 SXM's BF16 peak 989 or 1,979 TFLOPS? What is its ridge point?
   <details><summary>Answer</summary>Dense is 989 TFLOPS; the 1,979 on NVIDIA's spec page includes sparsity (table footnote "With sparsity"). Ridge ≈ 989e12 / 3.35e12 ≈ 295 FLOPs/Byte.</details>
2. Why is the arithmetic intensity of linear layers in dense-model decode roughly the batch size? What about MoE?
   <details><summary>Answer</summary>[B × d] times [d × d′]: FLOPs = 2Bdd′, bytes are mostly the weights 2dd′ (BF16), so the ratio is about B. Weights are read once per step and shared by all sequences in the batch. Each MoE routed expert gets only B·k/E tokens, B/32 for DeepSeek-V3, so it takes about 9,400 concurrent tokens to cross the BF16 ridge — the first-principles reason for large-scale EP.</details>
3. Why can a bigger batch not rescue decode attention at long context? When is there an exception?
   <details><summary>Answer</summary>When sequences share no prefix, each reads its own KV and nothing is amortized within the batch; attention intensity = n_heads / n_kv_heads (8 for GQA with 8 groups), independent of batch and far below the ridge. The exception is a shared prefix: cascade attention / Hydragen read the prefix KV once, so the prefix part's intensity becomes n_seq × n_heads / n_kv_heads.</details>
4. How many sequences can Llama-3-70B run at once on 8 × H100 with 32k context? Is this an upper or a lower bound?
   <details><summary>Answer</summary>(640 × 0.9 − 141) GB ≈ 435 GB; each sequence is 32,768 × 327,680 B ≈ 10.7 GB; about 40. It is the aggregate formula's upper bound, assuming weights are stored once and KV is split evenly, before subtracting workspace, the CUDA Graph memory pool, and wasted tail blocks of paged KV.</details>
5. Why can MLA bring decode attention close to compute-bound? What does it assume?
   <details><summary>Answer</summary>128 heads share one 576-dim latent vector, so every 1,152 bytes read feed about 128 × 2 × (576 + 512) ≈ 2.8e5 FLOPs, an intensity of about 242, close to the H800 ridge (about 295). It assumes all 128 heads are on one GPU (DP attention); split t ways by TP, each GPU still reads the whole latent, I ≈ 242 / t, and the KV is copied t times.</details>
6. When do average TPOT and average ITL differ?
   <details><summary>Answer</summary>Average TPOT weights every request equally; average ITL weights by tokens. With a mix of long and short requests, long requests contribute more ITL samples and the two diverge.</details>
7. A service needs 5,000 tokens/s of decode throughput and TPOT must not exceed 40 ms. How many sequences must be in the batch?
   <details><summary>Answer</summary>Little's law: B = throughput × TPOT. At exactly 40 ms TPOT you need 200, the most allowed under the SLO; a lower TPOT needs fewer. TPOT itself depends on B, so find B with B / TPOT(B) ≥ 5,000 and TPOT(B) ≤ 40 ms. Under this project's cost model, Llama-3-8B at 1k context on one H100 already reaches 5,000 tokens/s at B = 38 with TPOT about 7.5 ms.</details>
8. DeepSeek reports about 73.7k tokens/s of prefill and about 14.8k tokens/s of decode per 8 × H800 node. Can you say "prefill is 5× faster than decode"? How do you explain the remaining gap?
   <details><summary>Answer</summary>Not directly. The 73.7k includes cache hits; removing the 56.3% hits leaves about 32k tokens/s actually computed, about 2.2× decode; the deployment units differ (4 nodes EP32 vs 18 nodes EP144); whether 14.8k includes MTP speedup is not stated. Only the remaining gap belongs to the roofline: prefill processes many tokens at once, so its matmuls have high intensity and are compute-limited; decode produces 1 token per sequence per step and keeps re-reading weights and KV, so it is bandwidth-limited.</details>

## 13. Interview follow-ups

1. **"Why is decode memory-bound? How do you make it less so?"**
   Key points: conclusion first — decode produces 1 token per sequence per step; dense linear-layer intensity ≈ batch, attention intensity = n_heads / n_kv_heads, both far below the H100 ridge (about 295). Remedies: bigger batch (capped by KV memory), fewer bytes per token (GQA / MLA / FP8 KV / weight quantization), cascade attention over shared prefixes, speculative decoding to verify several tokens per step. Trade-offs: bigger batches lengthen TPOT; quantization risks accuracy; speculative decoding wastes compute when acceptance is low.
2. **"Given 8 H100s and Llama-3-70B at 32k context, how much concurrency can you hold and what throughput?"**
   Key points: weights about 141 GB, about 435 GB for KV, about 10.7 GB per sequence, so about 40 sequences (upper bound). Each step reads about 139 GB of weights (without the embedding table) + 40 × 10.7 GB ≈ 568 GB; 8 GPUs give 26.8 TB/s, at 80% MBU about 26.5 ms per step, throughput about 40 / 0.0265 ≈ 1,500 tokens/s. This ignores the latency of TP's 160 all-reduces per step, so real TPOT will be higher. To improve: FP8 KV (about double the concurrency), prefix sharing, PD disaggregation so decode machines do no prefill.
3. **"TTFT is too high. How do you break it down?"**
   Key points: TTFT = queueing + prefix-cache lookup and KV loading + the uncached part of prefill (plus waiting while interleaved with decode under chunked prefill) + TP / PP communication + sampling, network, detokenization. The first token comes from the logits at the last prefill position; there is no separate "first decode step". Check queueing first (Little's law, arrival rate vs service capacity), then prefill: hit rate, which tier the hit is in, chunk size, TP degree. At high hit rates prefill is already memory-bound and more compute does not help. Trade-offs: chunked prefill protects TPOT but lengthens TTFT of a single long prompt; PD disaggregation removes interference but must move KV.
4. **"What is the difference between goodput and throughput? Why does DistServe optimize goodput?"**
   Key points: throughput ignores latency; DistServe's per-GPU goodput is the maximum request rate each GPU can sustain at a given attainment level (e.g., 90% of requests meeting both TTFT and TPOT). Colocated prefill and decode interfere: a long prefill cutting in makes decode stall. DistServe separates them and reports serving 7.4× more requests or a 12.6× tighter SLO while meeting latency constraints.
5. **"Why does DeepSeek-V3 spread decode over EP across hundreds of GPUs? What role does MLA play?"**
   Key points: conclusion first — to pool each expert's batch. Each routed expert gets only B × 8 / 256 = B/32 tokens, so crossing the BF16 ridge takes about 9,400 concurrent tokens, which one GPU cannot gather; an 8-GPU machine also cannot hold 671 GB of FP8 weights. MLA shrinks KV to about 68.6 KiB/token (Llama-3-70B: 320 KiB), so each GPU holds more sequences; with DP attention, attention intensity is about 242, close to compute-bound. Costs: about 17B non-routed parameters on every GPU plus memory for redundant experts; EP's all-to-all over 400G IB sets a TPOT floor of about 14.76 ms, and MTP is used to enlarge the batch further.
6. **"Use Little's law: an agent platform gets 2 new sessions per second, each spends 30 s generating and 60 s waiting on tools, the KV stays pinned throughout, average context 32k, Llama-3-70B. How much memory holds the KV?"**
   Key points: L = 2 × 90 = 180 sessions; each 32k × 320 KiB ≈ 10.7 GB; about 1.9 TB on average, roughly all the KV space of 4–5 8 × H100 machines. With Poisson arrivals p99 is about 180 + 2.33√180 ≈ 211 sessions, about 17% more; with context growing turn by turn the average occupancy roughly halves. 2/3 of it is waiting on tools, so "offload to DRAM during tools" or "drop by TTL" saves most of it.

## 14. Further reading

- [All About Transformer Inference (How To Scale Your Model)](https://jax-ml.github.io/scaling-book/inference/): derives every formula in this module in more detail, including the critical batch on TPU / GPU.
- [Efficiently Scaling Transformer Inference (arXiv:2211.05102)](https://arxiv.org/abs/2211.05102): Google's partitioning and latency / MFU trade-offs for PaLM 540B inference; multi-query attention lets the context scale up to 32× longer.
- [Transformer Inference Arithmetic (kipply)](https://kipp.ly/transformer-inference-arithmetic/): one of the first blog posts to lay out the "FLOPs / bytes" accounting clearly.
- [Roofline (CACM 2009)](https://dl.acm.org/doi/abs/10.1145/1498765.1498785): the original roofline paper.
- [LLM Inference Unveiled: Survey and Roofline Model Insights (arXiv:2402.16363)](https://arxiv.org/abs/2402.16363): a systematic roofline analysis of inference optimization techniques.
- [DistServe (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670): the origin of goodput, TTFT / TPOT SLOs, and PD disaggregation.
- [DeepSeek-V3 technical report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437): MLA's KV cache form and the inference deployment units.
- [Insights into DeepSeek-V3 (arXiv:2505.09343)](https://arxiv.org/abs/2505.09343): the derivation of the 14.76 ms EP communication floor, MTP acceptance rates, and H800 bandwidth conventions.
- [Hydragen (arXiv:2402.05099)](https://arxiv.org/abs/2402.05099) and [FlashInfer cascade inference](https://flashinfer.ai/2024/02/02/cascade-inference.html): how to read shared-prefix KV only once.
