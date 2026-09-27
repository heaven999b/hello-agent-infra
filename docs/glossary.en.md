[中文](glossary.md) · **English**

# Glossary

> Every term used across the project: English and Chinese names, a one-line explanation, and the module where it first appears (searched in module order 00 → 09, then the capstone). The lectures keep English terms in the Chinese text; the Chinese names are listed here.

## Hardware and performance model

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **HBM (high-bandwidth memory)** | HBM（高带宽显存） | Stacked memory next to the GPU, tens to hundreds of GB at several TB/s; decode speed is mostly set by its bandwidth. | [M00](../modules/00_map/README.en.md) |
| **NVLink** | NVLink | NVIDIA's GPU-to-GPU interconnect; bidirectional totals per GPU are 900 GB/s on H100, cut to 400 GB/s on H800, 1.8 TB/s on B200; use one way (half) to estimate transfer times. | [M00](../modules/00_map/README.en.md) |
| **RDMA (remote direct memory access)** | RDMA（远程直接内存访问） | The NIC reads and writes a remote host's memory or GPU memory (with GPUDirect RDMA) without the remote CPU and without staging through host memory; cross-node KV transfer and expert-parallel traffic rely on it. | [M00](../modules/00_map/README.en.md) |
| **InfiniBand (IB)** | InfiniBand（IB） | An RDMA network common in data centers; NDR is 400 Gb/s ≈ 50 GB/s per port. | [M00](../modules/00_map/README.en.md) |
| **RoCE (RDMA over Converged Ethernet)** | RoCE | RDMA carried over Ethernet, the main alternative to InfiniBand. | [M00](../modules/00_map/README.en.md) |
| **PCIe** | PCIe | The bus between GPU and CPU / NIC / SSD; Gen5 x16 is about 64 GB/s one way, and KV offload to CPU memory goes over it. | [M00](../modules/00_map/README.en.md) |
| **FLOPs** | FLOPs | Number of floating-point operations (a multiply-add counts as 2); the per-second rate is FLOP/s or TFLOPS. | [M00](../modules/00_map/README.en.md) |
| **Dense vs sparse FLOPs** | dense / 稀疏算力 | Vendor spec pages often headline 2:4 structured-sparsity FLOPs, which are 2× dense; use dense for inference estimates. | [M00](../modules/00_map/README.en.md) |
| **Arithmetic intensity** | 算术强度 | FLOPs ÷ bytes moved to and from memory; for dense decode linear layers it is about the batch size, for MoE experts about B·k/E, and for decode attention about query heads ÷ KV heads (with BF16 KV and no prefix shared between sequences). | [M00](../modules/00_map/README.en.md) |
| **Roofline model** | roofline（屋顶线模型） | Attainable FLOP/s = min(peak, bandwidth × intensity); shows at a glance whether compute or bandwidth is the limit. | [M01](../modules/01_inference_basics/README.en.md) |
| **Ridge point** | ridge point（屋脊点） | Peak FLOP/s ÷ memory bandwidth; about 295 FLOPs/Byte for H100 BF16, and anything below it is memory-bound. | [M01](../modules/01_inference_basics/README.en.md) |
| **Compute-bound / memory-bound** | compute-bound / memory-bound | Time set by compute / by memory bandwidth, decided by whether FLOPs ÷ bytes exceeds the ridge. In common serving regimes prefill is usually the former and decode the latter; but an incremental prefill of a few hundred new tokens can be memory-bound, and decode attention with MLA or shared prefixes can approach compute-bound. | [M00](../modules/00_map/README.en.md) |
| **MFU (model FLOPs utilization)** | MFU | Achieved useful FLOP/s as a fraction of peak; prefill is often estimated at 40–60%. | [M00](../modules/00_map/README.en.md) |
| **MBU (model bandwidth utilization)** | MBU | Achieved memory bandwidth as a fraction of peak; the efficiency metric for decode. | [M01](../modules/01_inference_basics/README.en.md) |
| **BF16 / FP8** | BF16 / FP8 | Floating-point formats with 2 / 1 bytes per element. When weights or KV are quantized to FP8 and scale metadata is negligible, bytes roughly halve; the FP8 Tensor Core peak on Hopper / Blackwell is about 2× BF16, but only for matmuls, and end-to-end gains are limited by bandwidth, dequantization, and accuracy. | [M00](../modules/00_map/README.en.md) |

## Model architecture

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **KV cache** | KV cache | Stored per-layer keys and values of processed tokens so decode need not recompute them; size = tokens × KV bytes per token. | [M00](../modules/00_map/README.en.md) |
| **MHA (multi-head attention)** | MHA（多头注意力） | Every query head has its own K / V head; the largest KV. | [M01](../modules/01_inference_basics/README.en.md) |
| **MQA (multi-query attention)** | MQA（多查询注意力） | All query heads share one K / V head; the smallest KV but possible quality loss. | [M01](../modules/01_inference_basics/README.en.md) |
| **GQA (grouped-query attention)** | GQA（分组查询注意力） | Groups of query heads share one K / V head, between MHA and MQA; Llama-3-70B has 64 query heads and 8 KV heads. | [M00](../modules/00_map/README.en.md) |
| **MLA (multi-head latent attention)** | MLA（多头潜在注意力） | DeepSeek's attention: caches only a compressed latent (512 dims) plus a decoupled RoPE key (64 dims), shared by all heads. | [M00](../modules/00_map/README.en.md) |
| **Weight absorption** | 权重吸收 | In MLA decode, fold the up-projections into the query / output projections and attend directly over the latent: fewer bytes, more FLOPs. | [M01](../modules/01_inference_basics/README.en.md) |
| **MoE (mixture of experts)** | MoE（混合专家） | Each token activates only a few expert FFNs; DeepSeek-V3 has 671B total and 37B active parameters per token. | [M00](../modules/00_map/README.en.md) |
| **Active parameters** | 激活参数 | Parameters that actually compute for each token, which set FLOPs; memory must hold the total parameters. | [M01](../modules/01_inference_basics/README.en.md) |
| **MTP (multi-token prediction)** | MTP（多 token 预测） | A module trained to predict several future tokens; at inference it can serve as a speculative-decoding draft. | [M01](../modules/01_inference_basics/README.en.md) |
| **Auxiliary-loss-free load balancing** | 无辅助损失负载均衡 | DeepSeek-V3 adds a per-expert bias to routing scores, used only for top-K selection and nudged up or down by each step's load, and relies mainly on it for balance; it keeps only a tiny (α = 0.0001) sequence-wise balance loss to prevent extreme imbalance within a single sequence. | [M03](../modules/03_deepseek_stack/README.en.md) |

## Inference engines

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **Prefill** | prefill（预填充） | Processing the whole input prompt at once and writing its KV; compute-bound when there are many new tokens, possibly memory-bound when only a few hundred are new; sets TTFT. | [M00](../modules/00_map/README.en.md) |
| **Decode** | decode（解码） | Generating output tokens one at a time, reading the weights and each sequence's KV every step; usually memory-bound and sets TPOT. | [M00](../modules/00_map/README.en.md) |
| **Continuous batching** | continuous batching（连续批处理） | Schedule per iteration rather than per request: requests can join or leave the batch at every step (Orca's iteration-level scheduling). | [M00](../modules/00_map/README.en.md) |
| **PagedAttention / paged KV cache** | PagedAttention / 分页 KV cache | Split KV into fixed-size blocks mapped by a block table, like virtual memory, cutting waste to about one tail block (vLLM). Block size is a trade-off: smaller blocks mean less fragmentation and finer prefix sharing but more table lookups and indirect addressing. | [M00](../modules/00_map/README.en.md) |
| **Preemption** | 抢占 | When memory runs out, pause some requests and free their KV; resume by recomputing or by swapping back from CPU. | [M00](../modules/00_map/README.en.md) |
| **Chunked prefill** | chunked prefill（分块预填充） | Split a long prompt's prefill into chunks mixed into the same steps as decode, so a long prefill does not stall all decodes (Sarathi-Serve). | [M00](../modules/00_map/README.en.md) |
| **Speculative decoding** | 投机解码 | A cheap draft model or MTP guesses several tokens and the big model verifies them in one step, getting more out of memory-bound decode. | [M01](../modules/01_inference_basics/README.en.md) |
| **Quantization** | 量化 | Store weights, activations, or KV in fewer bits (e.g., FP8, INT4), saving bytes and possibly FLOPs at some accuracy risk. | [M01](../modules/01_inference_basics/README.en.md) |
| **PD disaggregation (prefill-decode disaggregation)** | PD 分离 | Run prefill and decode on separate GPU pools to remove interference, at the cost of moving KV across machines (DistServe, Splitwise, Mooncake). | [M00](../modules/00_map/README.en.md) |
| **CUDA Graph** | CUDA Graph | Record a sequence of GPU kernels and replay it as a whole to cut launch overhead; it needs static shapes, which clashes with a batch that changes every step. | [M01](../modules/01_inference_basics/README.en.md) |
| **FlashMLA** | FlashMLA | DeepSeek's open-source MLA decode kernel for Hopper; it reports up to 3000 GB/s memory-bound and 660 TFLOPS compute-bound on H800. | [M01](../modules/01_inference_basics/README.en.md) |
| **vLLM / SGLang / TensorRT-LLM** | vLLM / SGLang / TensorRT-LLM | Three mainstream inference engines: vLLM grew out of PagedAttention, SGLang out of RadixAttention, and TensorRT-LLM is NVIDIA's engine. | [M01](../modules/01_inference_basics/README.en.md) |

## Parallelism and communication

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **TP (tensor parallelism)** | TP（张量并行） | Split each layer's matrices across GPUs with an all-reduce per layer; suits high-bandwidth NVLink inside a node. | [M01](../modules/01_inference_basics/README.en.md) |
| **PP (pipeline parallelism)** | PP（流水线并行） | Place different layers on different GPUs and pipeline micro-batches; little traffic but pipeline bubbles. | [M01](../modules/01_inference_basics/README.en.md) |
| **DP (data parallelism)** | DP（数据并行） | Each replica serves different requests. Inference also uses DP attention: only attention is split across GPUs by request, while experts still use EP. For MLA (whose latent cannot be split by head) or when TP exceeds the KV-head count, TP duplicates KV across GPUs, which DP attention avoids; the cost is a copy of attention and other non-routed weights on every GPU. | [M01](../modules/01_inference_basics/README.en.md) |
| **EP (expert parallelism)** | EP（专家并行） | Spread MoE experts across GPUs and send tokens to them via all-to-all; DeepSeek decode uses EP144 and above. | [M00](../modules/00_map/README.en.md) |
| **All-to-all / all-reduce** | all-to-all / all-reduce | Collective primitives: all-to-all sends different data from every GPU to every GPU (EP dispatch); all-reduce sums results and broadcasts them (TP). | [M00](../modules/00_map/README.en.md) |
| **Node-limited routing** | node-limited routing（限节点路由） | DeepSeek-V3 sends each token to at most 4 nodes because cross-node IB bandwidth is about 1/3 of NVLink. | [M00](../modules/00_map/README.en.md) |
| **DeepEP** | DeepEP | DeepSeek's open-source expert-parallel communication library: all-to-all kernels for MoE dispatch / combine, high-throughput (training, prefill) and low-latency (decode), with FP8 support. | [M03](../modules/03_deepseek_stack/README.en.md) |
| **DualPipe** | DualPipe | DeepSeek-V3's bidirectional pipeline parallelism: micro-batches enter from both ends so all-to-all and PP traffic hide behind compute, with fewer bubbles. | [M03](../modules/03_deepseek_stack/README.en.md) |
| **EPLB (expert-parallel load balancer)** | EPLB（专家并行负载均衡器） | Duplicates heavily loaded experts (redundant experts) and heuristically places them across GPUs so each GPU's EP load is similar. | [M02](../modules/02_inference_engines/README.en.md) |

## Cache and routing

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **Prefix cache / prefix caching** | 前缀缓存 | Reuse already-computed KV for a prefix that is identical from the first token; change one token and everything after it is invalidated. | [M00](../modules/00_map/README.en.md) |
| **Prompt caching** | prompt caching | Prefix caching exposed in API billing: hits are billed at a discount (e.g., Anthropic cache reads at 0.1× the input price). | [M00](../modules/00_map/README.en.md) |
| **Block hash** | block hash（块哈希） | Split tokens into blocks whose hashes chain on the previous block, used to test prefix equality (vLLM). | [M02](../modules/02_inference_engines/README.en.md) |
| **Radix tree / RadixAttention** | radix tree / RadixAttention | A tree that organizes KV by prefix so requests share common prefixes and keep their branches (SGLang). | [M00](../modules/00_map/README.en.md) |
| **Cascade attention (shared-prefix attention)** | cascade attention（共享前缀注意力） | Split attention into a shared prefix and unique suffixes; for the prefix, stack the queries of many sequences and read one copy of the KV once, multiplying intensity by the number of sequences (Hydragen, FlashInfer). | [M01](../modules/01_inference_basics/README.en.md) |
| **Hit rate** | 命中率 | Fraction of input tokens served from cache; about 56.3% over 24 hours in DeepSeek production. | [M00](../modules/00_map/README.en.md) |
| **Eviction policy (LRU / TTL / session-aware)** | 淘汰策略（LRU / TTL / session-aware） | Who goes first when the cache is full: least recently used, by time-to-live, or by whether the session will come back. | [M00](../modules/00_map/README.en.md) |
| **TTL (time to live)** | TTL（生存时间） | How long an entry survives without access; giving KV a TTL during tool calls is the middle ground between keep and drop. | [M00](../modules/00_map/README.en.md) |
| **Pin** | pin（钉住） | Make a piece of KV non-evictable for a while. | [M00](../modules/00_map/README.en.md) |
| **KV offload / tiered storage** | KV 卸载 / 分层存储 | Move idle KV from HBM to CPU DRAM or SSD and bring it back when needed (Mooncake, 3FS). | [M00](../modules/00_map/README.en.md) |
| **Cache-affinity routing** | cache-affinity 路由 | Send requests to the instance that already caches their prefix; it conflicts with load balancing. | [M00](../modules/00_map/README.en.md) |
| **Load imbalance** | 负载不均衡 | Uneven load across instances; worsens when affinity routing piles popular prefixes onto a few instances. | [M00](../modules/00_map/README.en.md) |
| **cache_salt** | cache_salt | vLLM's per-request cache salt: folded into the first block's hash so only requests with the same salt share cache, for tenant isolation against timing side channels. | [M02](../modules/02_inference_engines/README.en.md) |
| **Mooncake** | Mooncake | Kimi's KVCache-centric disaggregated serving architecture that pools idle CPU, DRAM, and SSD in the cluster into a distributed KV cache. | [M02](../modules/02_inference_engines/README.en.md) |
| **3FS (Fire-Flyer File System)** | 3FS | DeepSeek's open-source distributed file system on SSD + RDMA, also used to store reusable KV cache. | [M03](../modules/03_deepseek_stack/README.en.md) |

## Metrics and queueing

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **TTFT (time to first token)** | TTFT（首 token 延迟） | From request submission to the first output token; = queueing + prefill of the uncached part. | [M00](../modules/00_map/README.en.md) |
| **TPOT (time per output token)** | TPOT（每输出 token 时间） | (end-to-end latency − TTFT) ÷ (output tokens − 1), one value per request. | [M00](../modules/00_map/README.en.md) |
| **ITL (inter-token latency)** | ITL（token 间延迟） | Gap between consecutive output tokens, one sample per gap; average ITL is token-weighted. | [M01](../modules/01_inference_basics/README.en.md) |
| **Throughput** | 吞吐 | Tokens processed or produced per second by the system; says nothing about any one request's wait. | [M00](../modules/00_map/README.en.md) |
| **Goodput** | goodput（有效吞吐） | DistServe's definition: the maximum request rate per GPU while meeting an SLO attainment goal (e.g., 90% of requests meeting both TTFT and TPOT); the Anyscale docs instead define it as the percentage of requests meeting all SLOs. | [M01](../modules/01_inference_basics/README.en.md) |
| **SLO (service level objective)** | SLO（服务等级目标） | A commitment on latency percentiles, e.g., "TTFT p99 < 2 s and TPOT p99 < 50 ms". | [M01](../modules/01_inference_basics/README.en.md) |
| **p99 / tail latency** | p99 / 尾延迟 | The 99th-percentile latency; serial multi-turn agents amplify the tail. | [M01](../modules/01_inference_basics/README.en.md) |
| **JCT (job completion time)** | JCT（任务完成时间） | Total time for a multi-turn agent job from start to finish; closer to user experience than per-request TTFT / TPOT. | [M05](../modules/05_cache_prefill/README.en.md) |
| **Little's law** | Little's law（利特尔法则） | Average number in system = arrival rate × average time in system (L = λW); turns throughput and latency into concurrency and memory. | [M01](../modules/01_inference_basics/README.en.md) |

## Agent harness

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **Agent harness** | agent harness | The program around the model that assembles context, calls tools, manages memory, and orchestrates agents; it decides what the model sees each round. | [M00](../modules/00_map/README.en.md) |
| **Context assembly** | 上下文组装 | Concatenating system prompt, tool definitions, history, and retrieved content into one request; the order decides whether prefixes can be reused. | [M00](../modules/00_map/README.en.md) |
| **Compaction** | 上下文压缩（compaction） | Summarizing or trimming history when the context grows too long; invalidates the prefix cache after the compaction point. | [M00](../modules/00_map/README.en.md) |
| **Memory injection** | 记忆注入 | Inserting long-term memory or retrieved content into the context; inserting it near the front breaks prefix reuse. | [M00](../modules/00_map/README.en.md) |
| **Tool call** | 工具调用 | The model emits a structured call, the harness runs it and appends the result; meanwhile the KV waits. | [M00](../modules/00_map/README.en.md) |
| **Append-only context** | append-only 上下文 | Only append at the end and never edit earlier content, so each round hits the previous round's prefix. | [M00](../modules/00_map/README.en.md) |
| **Multi-agent orchestration** | 多 agent 编排 | A lead agent spawns subagents that work in parallel and then merges results (fan-out / fan-in); token volume multiplies. | [M00](../modules/00_map/README.en.md) |
| **Session / turn** | 会话 / 轮 | An agent task's whole multi-round interaction / one "model output + tool result" round within it. | [M00](../modules/00_map/README.en.md) |
| **Logit masking** | logit 掩码 | Instead of deleting tool definitions from the context, mask disallowed tools during decoding to keep the prefix stable (Manus's approach). | [M01](../modules/01_inference_basics/README.en.md) |

## Training systems and RL

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **Agentic RL** | agentic RL | Training agents with RL: trajectories are multi-turn sessions with tool calls, and rewards come from task outcomes. | [M00](../modules/00_map/README.en.md) |
| **Rollout** | rollout（轨迹采样） | Running the current policy to produce trajectories for training; in agentic RL it is essentially a large batch of inference requests. | [M00](../modules/00_map/README.en.md) |
| **Long tail** | 长尾 | A few trajectories or requests are far longer than average; under heavy-tailed distributions (e.g., Pareto), waiting synchronously for the slowest is very expensive. | [M00](../modules/00_map/README.en.md) |
| **Synchronous / asynchronous RL** | 同步 / 异步 RL | Synchronous: wait for the whole batch before training; asynchronous: generate and train concurrently, trading slightly stale data for utilization (AReaL). | [M00](../modules/00_map/README.en.md) |
| **Partial rollout** | partial rollout（部分展开） | Generate only up to a time or length budget per round and continue unfinished trajectories next round, trimming the tail. | [M00](../modules/00_map/README.en.md) |
| **Staleness** | staleness（策略滞后） | How many versions the data-generating policy lags the policy being trained; asynchronous RL caps it. | [M00](../modules/00_map/README.en.md) |
| **On-policy / off-policy** | on-policy / off-policy | Whether training data comes from the current policy; data with staleness > 0 is mildly off-policy. | [M00](../modules/00_map/README.en.md) |
| **Importance sampling** | 重要性采样 | Reweight old data by the ratio of new to old policy probabilities to correct off-policy bias (PPO's ratio and clip follow this idea). | [M00](../modules/00_map/README.en.md) |
| **PPO (proximal policy optimization)** | PPO | A policy-gradient algorithm that limits each update with a clipped ratio of new to old policy probabilities. | [M03](../modules/03_deepseek_stack/README.en.md) |
| **GRPO (group relative policy optimization)** | GRPO | A PPO variant that samples a group of answers per prompt and uses group-relative rewards as advantages, dropping the value network (DeepSeekMath). | [M01](../modules/01_inference_basics/README.en.md) |
| **Decoupled PPO** | decoupled PPO | AReaL's objective: separate the behavior policy that sampled the data from the proximal policy that anchors the trust region, so stale data trains stably. | [M06](../modules/06_long_tail_async/README.en.md) |
| **Weight sync** | 权重同步 | Pushing updated weights from trainer to inference / rollout engines; hundreds of GB for large models. | [M00](../modules/00_map/README.en.md) |

## Multi-agent credit assignment

| English | 中文 | Explanation | First appears |
|---|---|---|---|
| **Credit assignment** | 信用分配 | When a team gets one total reward, deciding how much each agent's actions contributed. | [M00](../modules/00_map/README.en.md) |
| **Shared reward** | 共享奖励 | Every agent receives the team reward; unbiased but high variance, and free riders get rewarded too. | [M00](../modules/00_map/README.en.md) |
| **Group baseline** | 组内基线（group baseline） | Use the mean reward of a group of samples for the same prompt as the baseline (GRPO; RLOO leaves the sample itself out) to reduce variance. | [M06](../modules/06_long_tail_async/README.en.md) |
| **Difference reward** | difference reward（差分奖励） | Team reward minus the team reward with this agent's action replaced by a default, measuring its marginal contribution. | [M00](../modules/00_map/README.en.md) |
| **Shapley value** | Shapley 值 | A fair split that averages marginal contributions over all join orders; exact computation is exponential. | [M00](../modules/00_map/README.en.md) |
| **Counterfactual baseline** | 反事实基线 | Fix the other agents' actions and take the expectation over this agent's actions as the baseline (the COMA idea). | [M07](../modules/07_multi_agent_credit/README.en.md) |
| **Gradient variance** | 梯度方差 | Variance of the policy-gradient estimate; better credit assignment means lower variance and faster learning. | [M00](../modules/00_map/README.en.md) |

## Easily confused pairs

| Pair | Difference | Why it matters |
|---|---|---|
| **TPOT vs ITL** | TPOT is one value per request, (end-to-end − TTFT) ÷ (output tokens − 1), and its average weights requests equally; ITL is one sample per token gap and its average weights by tokens. NVIDIA's article treats them as synonyms. | They diverge when many requests are long; ask which convention is meant. |
| **Token hit rate vs request hit rate** | Token hit rate = cached input tokens ÷ all input tokens (DeepSeek's 56.3% is this); request hit rate = the fraction of requests with any hit (or a full hit). | Compute and cost savings follow the token hit rate; the TTFT distribution follows how much each request hits. |
| **MFU vs MBU** | MFU = achieved useful FLOP/s ÷ peak; MBU = achieved useful bandwidth ÷ peak. | Judge prefill by MFU and decode by MBU; a high MFU does not mean fast decode. |
| **Prefix cache vs semantic cache** | A prefix cache reuses already-computed KV for an identical prefix and still runs the model, so the output is the same as without the cache; a semantic cache returns an earlier answer for a semantically similar request and skips the model. | The former saves only prefill and never changes semantics; the latter saves more but can return a wrong or stale answer. |
| **DP attention vs ordinary DP** | Ordinary DP is several full replicas of the model, each serving different requests; DP attention splits only attention across GPUs by request, while experts stay shared across the whole EP group. | DP attention avoids duplicating MLA's KV within a TP group, at the cost of a copy of non-routed weights on every GPU. |
| **Goodput (DistServe) vs goodput (Anyscale)** | DistServe: the maximum request rate per GPU at a given attainment level (e.g., 90%), a rate; Anyscale: the percentage of requests meeting all SLOs, a percentage. | State the unit and the attainment level before quoting goodput. |
| **Bidirectional vs one-way bandwidth** | NVLink and PCIe numbers on spec pages are usually the sum of both directions (H100 NVLink 900 GB/s, PCIe Gen5 x16 128 GB/s); one transfer uses one direction (450 and 64 GB/s). | Using the bidirectional number underestimates transfer time by 2×. |
