[中文](reading-list.md) · **English**

# Reading List

> Primary sources for the whole project: organized into eight topics, plus a 6-hour shortest path. Every arXiv paper cited in the module lectures is included. Every entry's link, title and authors were opened and verified by the author; the year is the first public release (arXiv v1 or conference/blog publication).
> Depth: **Deep** = read methods and experiments, be able to restate the key formulas and numbers; **Skim** = read abstract, introduction, one core figure and the conclusion; know what it solves and what it costs.
> As of September 2026.

## 0. Shortest path: the 12 papers to read if you only have 6 hours

| Order | Title | Why read it first | Module |
|---|---|---|---|
| 1 | [Roofline](https://doi.org/10.1145/1498765.1498785) | The shared language for every "compute-bound vs bandwidth-bound" judgment | [M1](../modules/01_inference_basics/README.en.md) |
| 2 | [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) | Orders of magnitude for prefill / decode / KV cache in one place | [M1](../modules/01_inference_basics/README.en.md) |
| 3 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | The memory-management baseline of modern inference engines | [M2](../modules/02_inference_engines/README.en.md) |
| 4 | [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention and the "LM program" abstraction, the starting point for agent workloads | [M2](../modules/02_inference_engines/README.en.md) |
| 5 | [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | A top team's real production inference system with public cost numbers | [M3](../modules/03_deepseek_stack/README.en.md) |
| 6 | [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) | How model architecture, hardware and systems are co-designed | [M3](../modules/03_deepseek_stack/README.en.md) |
| 7 | [Mooncake](https://arxiv.org/abs/2407.00079) | KVCache-centric P/D disaggregation (Kimi's serving platform) | [M5](../modules/05_cache_prefill/README.en.md) |
| 8 | [Autellix](https://arxiv.org/abs/2502.13965) | Agent programs as first-class citizens of scheduling | [M4](../modules/04_harness_workloads/README.en.md) / [M9](../modules/09_frontier/README.en.md) |
| 9 | [AReaL](https://arxiv.org/abs/2505.24298) | Fully asynchronous RL and staleness control | [M6](../modules/06_long_tail_async/README.en.md) |
| 10 | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) | Origin of the group baseline, the template for later critic-free methods | [M7](../modules/07_multi_agent_credit/README.en.md) |
| 11 | [COMA](https://arxiv.org/abs/1705.08926) | The classic statement of the multi-agent counterfactual baseline and difference rewards | [M7](../modules/07_multi_agent_credit/README.en.md) |
| 12 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | How "evaluation" becomes the main workload in a self-improvement loop | [M9](../modules/09_frontier/README.en.md) |

## 1. Inference fundamentals

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 | Starting point of every FLOPs and KV cache formula; the architecture and complexity table are enough | Skim |
| [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) | Williams, Waterman, Patterson (CACM 52(4)) | 2009 | Original definition of arithmetic intensity and the ridge point, the tool for prefill/decode bottlenecks | Deep |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | kipply (blog) | 2022 | A model of pen-and-paper math for KV cache size and memory time | Deep |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) | Pope et al. | 2022 | Systematic analysis of inference partitioning and latency/cost trade-offs; frequent in interviews | Deep |
| [How To Scale Your Model: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) | Austin et al. (Google DeepMind) | 2025 | Inference order-of-magnitude derivations as a textbook; pairs with this project's mental-math tables | Deep |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. | 2020 | Power laws linking loss to model size, data size and training compute | Skim |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | Shoeybi et al. | 2019 | The original intra-layer tensor parallelism (TP), the basis of inference partitioning | Skim |
| [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) | Korthikanti et al. | 2022 | Sequence parallelism and selective activation recomputation: the training-side memory bill | Skim |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | Shazeer | 2019 | MQA: fewer KV heads to cut decode memory traffic | Skim |
| [GQA](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 2023 | The compromise between MHA and MQA, today's default in most open models | Skim |
| [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | Rotary position embedding, also implemented in the capstone's mini engine | Skim |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Shazeer | 2020 | Origin of gated feed-forward layers such as SwiGLU, which set FFN parameters and FLOPs | Skim |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | Zhang, Sennrich | 2019 | RMSNorm: a normalization layer common in modern open models (also used by the capstone's mini engine) | Skim |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Dao et al. | 2022 | The IO-aware idea: design kernels around the memory hierarchy | Skim |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao | 2023 | Better parallelism and work partitioning | Skim |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Shah et al. | 2024 | Asynchrony and low precision on Hopper | Skim |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Leviathan, Kalman, Matias | 2022 | A small draft model turns bandwidth-bound decode into verification | Skim |
| [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) | Yuan et al. | 2024 | A survey organizing inference optimizations through the roofline lens | Skim |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Grattafiori et al. | 2024 | Architecture parameters and training/inference infrastructure details of an open model | Skim |
| [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) | Dean, Barroso (CACM) | 2013 | Tail latency grows with fan-out, plus classic remedies such as hedged requests: the systems origin of the long-tail problem | Deep |
| [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383) | Little (Operations Research) | 1961 | Little's law: concurrency = arrival rate × time in system, the tool for estimating concurrent sandboxes, in-flight requests and KV occupancy | Skim |

## 2. Inference engines

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Yu et al. (Seoul National University, FriendliAI) | 2022 | Origin of iteration-level scheduling (continuous batching) | Deep |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Kwon et al. | 2023 | vLLM's paged KV management, the memory-management baseline of modern engines | Deep |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | Zheng et al. | 2023 | RadixAttention prefix reuse + `fork`/`gen` program primitives | Deep |
| [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) | LMSYS (blog) | 2024 | An intuitive illustration of RadixAttention | Skim |
| [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) | LMSYS (blog) | 2024 | Zero-overhead batch scheduler and cache-aware load balancer | Skim |
| [SARATHI](https://arxiv.org/abs/2308.16369) | Agrawal et al. | 2023 | The predecessor of Sarathi-Serve: chunked prefill with decodes piggybacked on prefill batches | Skim |
| [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Agrawal et al. | 2024 | Chunked prefill: split long prefills and batch them with decodes | Deep |
| [DistServe](https://arxiv.org/abs/2401.09670) | Zhong et al. | 2024 | Prefill/decode disaggregation to optimize goodput | Deep |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Patel et al. | 2023 | Put prefill and decode phases on different machines | Skim |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | Qin et al. | 2024 | Kimi's serving platform: P/D disaggregation + a distributed KV cache on CPU/DRAM/SSD | Deep |
| [LMCache](https://arxiv.org/abs/2510.09665) | Liu et al. | 2025 | A standalone KV cache layer for enterprise scale | Skim |
| [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | vLLM project (blog) | 2025 | The V1 rewrite of vLLM's core architecture | Skim |
| [vLLM: Automatic Prefix Caching design doc](https://docs.vllm.ai/en/stable/design/prefix_caching/) | vLLM project | — | How a production engine implements prefix caching by block hash | Deep |
| [Hydragen](https://arxiv.org/abs/2402.05099) | Juravsky et al. | 2024 | Computes attention over the shared prefix and unique suffixes separately, so large-batch decode throughput barely drops with long shared prefixes | Skim |
| [Cascade Inference](https://flashinfer.ai/2024/02/02/cascade-inference.html) | FlashInfer (blog) | 2024 | Cascade attention over shared prefixes to save memory bandwidth | Skim |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Frantar et al. | 2022 | A classic post-training weight quantization method | Skim |
| [AWQ](https://arxiv.org/abs/2306.00978) | Lin et al. | 2023 | Activation-aware weight quantization that protects salient weights | Skim |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | Chen et al. | 2023 | Another landmark speculative sampling paper: a modified rejection sampling scheme preserves the target model's distribution | Skim |
| [EAGLE](https://arxiv.org/abs/2401.15077) | Li et al. | 2024 | Speculative decoding with feature-level autoregressive drafts | Skim |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | Li et al. | 2025 | Switches to direct token prediction with multi-layer feature fusion (training-time test), so the draft model benefits from more training data | Skim |
| [MagicDec](https://arxiv.org/abs/2408.11049) | Sadhukhan et al. | 2024 | Speculative decoding still speeds up the high-throughput regime with moderate-to-long contexts and large batches | Skim |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) | Oliaro et al. | 2024 | Model-free speculative decoding for the highly repetitive, predictable sequences of agent workloads | Skim |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | GeeeekExplorer | — | A lightweight vLLM implementation in about 1,200 lines of Python, a shortcut to understanding engine structure | Skim |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | vLLM project | — | Code: scheduler and KV management | Skim |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | SGLang project | — | Code: radix cache and routing | Skim |

## 3. DeepSeek

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | Dai et al. | 2024 | Fine-grained plus shared experts, the origin of DeepSeek's MoE structure | Deep |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | DeepSeek-AI | 2024 | Origin of MLA (multi-head latent attention), the core of KV cache compression | Deep (MLA section) |
| [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) | Wang et al. | 2024 | The auxiliary-loss-free load balancing adopted by V3 | Skim |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek-AI | 2024 | MoE, FP8 training, DualPipe and the deployment plan | Deep (infrastructure sections) |
| [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) | Zhao et al. (DeepSeek-AI) | 2025 | Reflections on model–hardware co-design; the closest to an "infra engineer's view" | Deep |
| [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | DeepSeek-AI (open-infra-index) | 2025 | Production EP deployment, P/D disaggregation, and cost/profit numbers | Deep |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Shao et al. | 2024 | Origin of GRPO (Section 4) | Deep (Section 4) |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek-AI | 2025 | Large-scale RL eliciting reasoning; where post-training load comes from | Skim |
| [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | Yuan et al. | 2025 | Hardware-aligned, natively trainable sparse attention | Skim |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek-AI | 2025 | DSA sparse attention, post-training compute >10% of pre-training, agentic task synthesis with 1,827 environments | Deep |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | DeepSeek-AI | 2026 | V4-Pro (1.6T parameters, 49B activated) and V4-Flash (284B, 13B activated), hybrid attention with million-token context | Deep |
| [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) | DeepSeek-AI | 2026 | Pushes KV cache compression to the limit: long-horizon agents make workloads input-heavy, so prefill and KV storage and bandwidth become the main bottleneck | Deep |
| [DeepSeek Elastic Compute (DSec)](https://arxiv.org/abs/2609.22978) | Huang et al. (DeepSeek-AI) | 2026 | A production sandbox platform: over 380,000 concurrent sandboxes, co-designed with the RL framework | Deep |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | DeepSeek-AI | 2025 | MLA decode kernels | Skim |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | DeepSeek-AI | 2025 | Expert-parallel communication library | Skim |
| [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) | DeepSeek-AI | 2025 | Bidirectional pipeline parallelism overlapping computation and communication | Skim |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | DeepSeek-AI | 2025 | A clean and efficient BLAS (GEMM) kernel library for GPUs | Skim |
| [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB) | DeepSeek-AI | 2025 | Expert-parallelism load balancer | Skim |
| [deepseek-ai/3FS](https://github.com/deepseek-ai/3FS) | DeepSeek-AI | 2025 | A distributed file system for training and inference; DSec's images live on it too | Skim |
| [DeepSeek API: Context Caching](https://api-docs.deepseek.com/guides/kv_cache) | DeepSeek | — | Default-on disk prefix cache; prefix-only matching, best effort | Skim |

## 4. Agent workloads and caching

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | Yichao 'Peak' Ji (Manus, blog) | 2025 | Production-agent view: KV-cache hit rate is the single most important metric; input:output ratio ~100:1 | Deep |
| [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Anthropic (engineering blog) | 2025 | Multi-agent systems use ~15× the tokens of chat; token usage explains 80% of performance variance | Deep |
| [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Anthropic (engineering blog) | 2025 | How a harness manages context; where "rewrite/compaction" comes from | Skim |
| [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934) | Gim et al. | 2023 | Modular reuse of attention states | Skim |
| [CacheGen](https://arxiv.org/abs/2310.07240) | Liu et al. | 2023 | Compressing and streaming KV caches: the bandwidth bill of fetching KV remotely | Skim |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | Yao et al. | 2024 | KV reuse at non-prefix positions (RAG) | Skim |
| [EPIC](https://arxiv.org/abs/2410.15332) | Hu et al. | 2024 | Position-independent caching: reuse KV without an exact prefix match | Skim |
| [CachedAttention](https://arxiv.org/abs/2403.19708) | Gao et al. | 2024 | Reuses the KV of historical tokens across turns instead of recomputing it each turn | Skim |
| [Preble](https://arxiv.org/abs/2407.00023) | Srivatsa et al. | 2024 | Distributed prompt scheduling that balances cache reuse and load across instances | Skim |
| [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312) | Cao et al. | 2025 | The tension between prefix locality and multi-client fairness, and scheduling algorithms for it | Skim |
| [Consistent Hashing with Bounded Loads](https://arxiv.org/abs/1608.01350) | Mirrokni, Thorup, Zadimoghaddam | 2016 | Consistent hashing with a load bound: the classic algorithm behind "affinity routing + overflow cap" | Skim |
| [KVCache Cache in the Wild](https://arxiv.org/abs/2506.02634) | Wang et al. | 2025 | Characterizes real KV cache workloads at a large cloud provider: eviction policy quality is highly workload-dependent | Deep |
| [TokenCake](https://arxiv.org/abs/2510.18586) | Bian et al. | 2025 | KV management for multi-agent apps: spatial contention and idling during function calls | Skim |
| [InferCept](https://arxiv.org/abs/2402.01869) | Abhyankar et al. | 2024 | Preserve/swap/recompute decisions for KV when a tool call intercepts generation | Deep |
| [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) | Lin et al. (Shanghai Jiao Tong University, Microsoft Research; [arXiv](https://arxiv.org/abs/2405.19888)) | 2024 | Let the service see application-level data dependencies | Skim |
| [Autellix](https://arxiv.org/abs/2502.13965) | Luo et al. | 2025 | Program-level scheduling (PLAS/ATLAS) and program-level head-of-line blocking | Deep |
| [Continuum](https://arxiv.org/abs/2511.02230) | Li et al. | 2025 | TTL-based KV retention during tool calls | Deep |
| [KVFlow](https://arxiv.org/abs/2507.07400) | Pan et al. | 2025 | Predict reuse from the workflow graph and prefetch KV | Skim |
| [CacheScout (Learning Agent Execution for KV-Cache Management)](https://arxiv.org/abs/2608.14624) | Zhang et al. | 2026 | Learn agent transitions online to manage KV | Skim |
| [TOPAS](https://arxiv.org/abs/2608.25523) | Ni et al. | 2026 | Joint decisions on prefix retention and request scheduling | Skim |
| [ConServe (Observation, Not Prediction)](https://arxiv.org/abs/2606.01839) | Ding et al. | 2026 | Conversation-level P/D disaggregated scheduling | Skim |
| [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849) | Liu et al. | 2026 | Tools report progress while running instead of predicting durations | Deep |
| [Dynamo: Agent Hints doc](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-hints.md) | NVIDIA Dynamo project | — | An existing industrial harness→engine hint interface (priority, osl, speculative_prefill) | Skim |
| [SGLang HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) | LMSYS (blog) | 2025 | Hierarchical KV caching with pluggable storage backends | Skim |
| [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) | llm-d (blog) | 2025 | From prefix caching in vLLM to distributed cache-aware scheduling | Skim |
| [The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) | Song et al. | 2024 | Timing side channels from shared KV caches: the isolation cost of multi-tenant caching | Skim |
| [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/abs/2502.07776) | Gu et al. | 2025 | Audits via timing differences whether commercial APIs share prompt caches across users | Skim |
| [I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/) | Wu et al. (Southern University of Science and Technology, ByteDance; NDSS) | 2026 | An attack that recovers other users' prompts under multi-tenant KV sharing | Skim |

## 5. RL training systems (sync / partial / async)

### 5.1 Systems and frameworks

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [IMPALA](https://arxiv.org/abs/1802.01561) | Espeholt et al. | 2018 | Actor–learner decoupling and V-trace off-policy correction, the classic of async RL | Deep (V-trace section) |
| [Asynchronous RLHF](https://arxiv.org/abs/2410.18252) | Noukhovitch et al. | 2024 | How much staleness async, off-policy RL can tolerate for LLMs | Deep |
| [HybridFlow (verl)](https://arxiv.org/abs/2409.19256) | Sheng et al. | 2024 | The synchronous colocated baseline and 3D-HybridEngine resharding | Deep |
| [OpenRLHF](https://arxiv.org/abs/2405.11143) | Hu et al. | 2024 | Open-source RLHF framework on Ray + vLLM | Skim |
| [AReaL](https://arxiv.org/abs/2505.24298) | Fu et al. | 2025 | Fully async + interruptible rollouts + staleness-enhanced PPO, up to 2.77× | Deep |
| [Beyond Ten Turns (ASearcher)](https://arxiv.org/abs/2508.07976) | Gao et al. | 2025 | Fully asynchronous RL for long-horizon search agents, with over 100 turns of tool calls during training | Skim |
| [PipelineRL](https://arxiv.org/abs/2509.19128) | Piché et al. | 2025 | In-flight weight updates: keep generating with the old KV; measured KL only slightly higher than recomputing | Deep |
| [StreamRL](https://arxiv.org/abs/2504.15930) | Zhong et al. | 2025 | Pipeline bubbles and skewness bubbles in disaggregated RL | Deep |
| [APRIL](https://arxiv.org/abs/2509.18521) | Zhou et al. | 2025 | Over-provision rollouts, stop at the target count, recycle unfinished ones | Skim |
| [Seer](https://arxiv.org/abs/2511.14617) | Qin et al. | 2025 | Exploit similarity among same-prompt samples for load balancing and speculative decoding | Skim |
| [RollPacker](https://arxiv.org/abs/2509.21009) | Gao et al. | 2025 | Tail batching: consolidate prompts that produce long-tail responses into a few "long rounds" so most rounds run short, balanced rollouts, without relaxing synchronization | Skim |
| [ROLL](https://arxiv.org/abs/2506.06122) | Wang et al. | 2025 | An RL library for scale: rollout scheduler, environment/reward workers | Skim |
| [ROLL Flash](https://arxiv.org/abs/2510.11345) | Lu et al. | 2025 | ROLL's native async extension: environment-level asynchrony and redundant environments | Skim |
| [RollArt](https://arxiv.org/abs/2512.22560) | Gao et al. | 2025 | Heterogeneous disaggregated deployment for multi-task agentic RL, run on 3,000+ GPUs | Skim |
| [AgentRL](https://arxiv.org/abs/2510.04206) | Zhang et al. | 2025 | Async framework and containerized environments for multi-turn multi-task agentic RL | Skim |
| [LlamaRL](https://arxiv.org/abs/2505.24034) | Wu et al. | 2025 | A distributed async RL framework with direct device-memory weight sync | Skim |
| [ProRL Agent](https://arxiv.org/abs/2603.18815) | Zhang et al. | 2026 | Rollout-as-a-service: the rollout lifecycle served through an API | Skim |
| [Polar: Agentic RL on Any Harness at Scale](https://arxiv.org/abs/2605.24220) | Xu et al. | 2026 | Treats the harness as a black box and rebuilds token-faithful trajectories; gains depend strongly on the harness | Skim |
| [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) | Moonshot AI | — | Middleware for updating model weights inside inference engines | Skim |
| [THUDM/slime](https://github.com/THUDM/slime) | THUDM | — | Megatron + SGLang RL post-training framework behind the GLM family | Skim |
| [verl-project/verl](https://github.com/verl-project/verl) | verl project | — | Open-source implementation of HybridFlow | Skim |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | AReaL project | — | Open-source implementation of async RL | Skim |

### 5.2 Off-policy correction and training–inference consistency

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | Schulman et al. | 2015 | Trust regions and the local surrogate: why token-level surrogates only hold when policies are close | Skim |
| [Safe and Efficient Off-Policy Reinforcement Learning (Retrace)](https://arxiv.org/abs/1606.02647) | Munos et al. | 2016 | Off-policy return estimation with truncated traces, the same family as V-trace | Skim |
| [Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641) | Hilton, Cobbe, Schulman | 2021 | Origin of decoupling the proximal policy from the behavior policy (decoupled PPO) | Deep |
| [Mastering Complex Control in MOBA Games with Deep Reinforcement Learning](https://arxiv.org/abs/1912.09729) | Ye et al. | 2019 | Dual-clip PPO: a lower bound on the objective for negative advantages, especially important on stale data | Skim |
| [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) | Zheng et al. | 2025 | Sequence-level importance ratios and sequence-level clipping | Skim |
| [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374) | Zheng et al. | 2025 | When the token-level surrogate is valid: both training–inference discrepancy and staleness must be small | Deep |
| [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370) | Ma et al. | 2025 | Training–inference router mismatch in MoE and how to align it | Skim |
| [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) | Qi et al. | 2025 | Attributes training–inference mismatch to BF16 precision and switches to FP16 | Skim |
| [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Horace He (Thinking Machines Lab, blog) | 2025 | The root cause of inference nondeterminism is kernels lacking batch invariance | Deep |

### 5.3 RL infrastructure in model technical reports

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Kimi Team | 2025 | Partial rollouts to handle the long tail of long CoT | Skim |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | Kimi Team | 2025 | Agentic data synthesis and colocated RL infrastructure | Skim |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | GLM-4.5 Team | 2025 | Open MoE for agentic/reasoning/coding and its post-training | Skim |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) | MiniMax | 2025 | CISPO: clip the importance weights instead of the update | Skim |
| [Magistral](https://arxiv.org/abs/2506.10910) | Mistral-AI | 2025 | Replace weights mid-generation and continue with a slightly stale KV | Skim |
| [rStar2-Agent](https://arxiv.org/abs/2508.20722) | Shang et al. | 2025 | Agentic RL on 64 MI300X: high-throughput code environment and GRPO-RoC | Skim |

## 6. Credit assignment

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Buy 4 REINFORCE Samples, Get a Baseline for Free!](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/) | Kool, van Hoof, Welling (ICLR 2019 workshop) | 2019 | Origin of the leave-one-out baseline (RLOO) | Skim |
| [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) | Ahmadian et al. | 2024 | Brings RLOO back to LLM post-training; argues many PPO parts are unnecessary | Deep |
| [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) | Liu et al. | 2025 | GRPO's length bias and difficulty bias; drop the 1/\|o\| and std normalizations | Deep |
| [DAPO](https://arxiv.org/abs/2503.14476) | Yu et al. | 2025 | Four tricks incl. Clip-Higher and Dynamic Sampling; dynamic sampling means extra rollouts | Deep |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | Lightman et al. | 2023 | Process supervision beats outcome supervision; PRM800K | Skim |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | Wang et al. | 2023 | Auto-label process rewards with N completions from a given step | Skim |
| [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) | Cui et al. | 2025 | Online implicit PRM updates from outcome labels only | Skim |
| [VinePPO](https://arxiv.org/abs/2410.01679) | Kazemnejad et al. | 2024 | Monte Carlo rollouts instead of a value network: better credit, costlier steps | Deep |
| [GiGPO: Group-in-Group Policy Optimization](https://arxiv.org/abs/2505.10978) | Feng et al. | 2025 | Step-level groups via "anchor states", no extra rollouts | Deep |
| [Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment](https://arxiv.org/abs/2505.11821) | Wei et al. | 2025 | GRPO/PPO variants under turn-level rewards | Skim |
| [TRACE](https://arxiv.org/abs/2607.13988) | Tao et al. | 2026 | Turn-level TD rewards from a frozen reference model's gold-answer log-probs | Skim |
| [GACA](https://arxiv.org/abs/2609.12424) | Liang et al. | 2026 | Adaptively mix step- and episode-level advantages by NLL | Skim |
| [Coverage, Not Targeting](https://arxiv.org/abs/2609.02417) | Zhou et al. | 2026 | With terminal-state verifiers, uniformly spread dense reward beats "targeting critical turns"; includes control designs | Deep |
| [TreeRL](https://arxiv.org/abs/2506.11902) | Hou et al. | 2025 | Branches from high-uncertainty intermediate steps rather than randomly, under the same token budget | Skim |
| [Agentic Reinforced Policy Optimization (ARPO)](https://arxiv.org/abs/2507.19849) | Dong et al. | 2025 | Entropy-based adaptive branching after tool calls | Skim |
| [First Return, Entropy-Eliciting Explore (FR3E)](https://arxiv.org/abs/2507.07017) | Zheng et al. | 2025 | Targeted rollouts at high-uncertainty decision points | Skim |
| [TreePO](https://arxiv.org/abs/2508.17445) | Li et al. | 2025 | Tree sampling that amortizes common prefixes, saving 22%–43% of GPU hours | Deep |
| [Tree Search for LLM Agent Reinforcement Learning (Tree-GRPO)](https://arxiv.org/abs/2509.21240) | Ji et al. | 2025 | Shares prefixes to get more rollouts within a fixed token / tool-call budget | Skim |
| [Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) | Foerster et al. | 2017 | Counterfactual baseline; a clear discussion of what difference rewards cost | Deep |
| [Optimal Payoff Functions for Members of Collectives](https://doi.org/10.1142/S0219525901000188) | Wolpert, Tumer (Advances in Complex Systems 4(2–3)) | 2001 | The difference reward / Wonderful Life Utility family of rewards | Skim |
| [A Value for n-Person Games](https://doi.org/10.1515/9781400881970-018) | Shapley (Contributions to the Theory of Games II) | 1953 | Original definition of the Shapley value | Skim |
| [Value-Decomposition Networks](https://arxiv.org/abs/1706.05296) | Sunehag et al. | 2017 | VDN: team Q as a sum over agents; the lazy-agent problem | Skim |
| [QMIX](https://arxiv.org/abs/1803.11485) | Rashid et al. | 2018 | Value factorization with a monotonic mixing network | Skim |
| [Shapley Q-value](https://arxiv.org/abs/1907.05707) | Wang et al. | 2019 | Shapley value as each agent's critic | Skim |
| [LLM Collaboration With Multi-Agent Reinforcement Learning (MAGRPO)](https://arxiv.org/abs/2508.04652) | Liu et al. | 2025 | Multi-agent GRPO: joint reward + group-centered advantage + decentralized execution | Deep |
| [Stronger-MAS (AT-GRPO)](https://arxiv.org/abs/2510.11062) | Zhao et al. | 2025 | Why GRPO grouping breaks for multi-agent; agent×turn grouping and a multi-policy training system | Deep |
| [Exact Is Easier: Credit Assignment for Cooperative LLM Agents (C3)](https://arxiv.org/abs/2603.06859) | Chen et al. | 2026 | Text histories restore exactly → fixed-history sampling + leave-one-out baseline; agent-removal counterfactuals are biased | Deep |
| [Counterfactual Credit Policy Optimization (CCPO)](https://arxiv.org/abs/2603.21563) | Li et al. | 2026 | Marginal contribution by removing an agent; one extra rollout per sample | Skim |
| [MAPoRL](https://arxiv.org/abs/2502.18439) | Park et al. | 2025 | Multi-LLM discussion + verifier scoring for multi-agent co-training | Skim |
| [MALT](https://arxiv.org/abs/2412.01928) | Motwani et al. | 2024 | Generator–verifier–refiner chain; credit via search tree + value iteration | Skim |
| [ReMA](https://arxiv.org/abs/2503.09501) | Wan et al. | 2025 | Hierarchical multi-agent RL with a meta-thinking agent and a reasoning agent | Skim |
| [SPIRAL](https://arxiv.org/abs/2506.24119) | Liu et al. | 2025 | Zero-sum self-play; role-conditioned advantage baselines (RAE) prevent thinking collapse | Skim |
| [SWEET-RL](https://arxiv.org/abs/2503.15478) | Zhou et al. | 2025 | A step-level critic trained with training-time privileged information | Skim |
| [MARFT](https://arxiv.org/abs/2504.16129) | Liao et al. | 2025 | Formalization and open problems of multi-agent reinforcement fine-tuning for LLMs | Skim |
| [From Reasoning to Agentic: Credit Assignment in RL for LLMs](https://arxiv.org/abs/2604.09459) | Zhang | 2026 | A credit assignment survey covering January 2024 to July 2026 | Skim |
| [RL for LLM-based Multi-Agent Systems through Orchestration Traces](https://arxiv.org/abs/2605.02801) | Zhang | 2026 | Reward and credit units of multi-agent RL seen through orchestration traces | Skim |

## 7. Reward design, LLM judges and reward hacking

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Zheng et al. | 2023 | Position, verbosity and self-enhancement biases of LLM judges | Skim |
| [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) | Gunjal et al. | 2025 | Rubrics as rewards for non-verifiable domains | Skim |
| [Checklists Are Better Than Reward Models For Aligning Language Models](https://arxiv.org/abs/2507.18624) | Viswanathan et al. | 2025 | Item-by-item checklists instead of a scalar reward model | Skim |
| [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) | Gao, Schulman, Hilton | 2022 | Quantifying reward hacking: as the proxy is optimized, gold reward rises then falls | Deep |
| [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) | Zhao et al. | 2025 | A single symbol can elicit positive rewards from a judge | Skim |
| [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | Lilian Weng (blog) | 2024 | A systematic review of reward hacking | Skim |

## 8. Frontier (agent serving, agentic RL, RSI)

| Title | Authors / Org | Year | Why read | Depth |
|---|---|---|---|---|
| [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547) | Zhang et al. | 2025 | A taxonomy of agentic RL plus a compendium of open environments/frameworks | Skim |
| [Chain-of-Agents](https://arxiv.org/abs/2508.13167) | Li et al. | 2025 | Distill a multi-agent system into one model, then agentic RL | Skim |
| [STaR](https://arxiv.org/abs/2203.14465) | Zelikman et al. | 2022 | Iteratively train on its own correct rationales | Skim |
| [Voyager](https://arxiv.org/abs/2305.16291) | Wang et al. | 2023 | An open-ended agent with an ever-growing skill library | Skim |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | Yuan et al. | 2024 | Iterative training where the model rewards itself | Skim |
| [Gödel Agent](https://arxiv.org/abs/2410.04444) | Yin et al. | 2024 | A self-referential agent framework that modifies its own logic | Skim |
| [Absolute Zero](https://arxiv.org/abs/2505.03335) | Zhao et al. | 2025 | Self-proposed, self-solved tasks with zero external data; a code executor as reward | Skim |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Zhang et al. | 2025 | An archive of self-modifying coding agents, SWE-bench 20.0% → 50.0% | Deep |
| [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614) | Wang et al. | 2025 | Benchmark score ≠ self-improvement potential; allocate budget by potential estimated from descendants | Deep |
| [Hyperband](https://arxiv.org/abs/1603.06560) | Li et al. | 2016 | Early-stopping resource allocation, a mature baseline for scheduling evaluations | Skim |
| [SEAL: Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) | Zweiger et al. | 2025 | The model generates its own finetuning data and update directives | Skim |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | Novikov et al. (Google DeepMind white paper) | 2025 | An evolutionary coding agent improving infrastructure and its own training | Deep |
| [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046) | Gao et al. | 2025 | A survey organized by what, when and how to evolve | Skim |
| [A Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407) | Fang et al. | 2025 | Self-evolution through a unified feedback-loop framework | Skim |
| [SEAGym](https://arxiv.org/abs/2606.17546) | Zheng et al. | 2026 | An evaluation environment for self-evolving harnesses: train/validate/replay/cost records | Skim |
| [ModularRSI](https://arxiv.org/abs/2609.14857) | Wu et al. | 2026 | Modular harness self-improvement disjoint from evaluation benchmarks | Skim |
| [ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://recursive-workshop.github.io/) | Organizers from KAUST, ByteDance, NYU/DeepMind and others | 2026 | Problem framing of the RSI direction and accepted papers | Skim |
| [SWE-bench/experiments](https://github.com/SWE-bench/experiments) | SWE-bench project | — | Each submission's predictions, execution logs and trajectories: public traces for studying agent workloads | Skim |
| [CoMLRL (MAGRPO code)](https://github.com/OpenMLRL/CoMLRL) | MAGRPO authors | — | Open implementation of multi-agent GRPO; read how advantages are computed | Skim |
| [PettingLLMs (AT-GRPO code)](https://github.com/pettingllms-ai/PettingLLMs) | AT-GRPO authors | — | Open implementation of agent×turn grouping and multi-policy training | Skim |
| [C3 code](https://github.com/EIT-EAST-Lab/C3) | C3 authors | — | Open implementation of fixed-history re-sampling and the leave-one-out baseline | Skim |
| [Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) | xxzcc | — | A continuously updated catalog of credit assignment papers | Skim |
