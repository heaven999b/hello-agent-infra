**中文** · [English](reading-list.en.md)

# 阅读清单

> 全项目的一手资料清单：按八个主题组织，另附一条 6 小时最短路径。各模块讲义里引用过的 arXiv 论文都已收录。每一条都由作者打开核实过链接、标题和作者；年份按首次公开（arXiv 首版或会议/博客发布）计。
> 深度说明：**精读** = 读方法与实验、能复述关键公式和数字；**略读** = 读摘要、引言、一张核心图和结论，知道它解决什么、代价是什么。
> 截至 2026 年 9 月。

## 0. 最短路径：只有 6 小时读论文时读这 12 篇

| 顺序 | 标题 | 为什么先读它 | 对应模块 |
|---|---|---|---|
| 1 | [Roofline](https://doi.org/10.1145/1498765.1498785) | 所有「算力受限 / 带宽受限」判断的共同语言 | [M1](../modules/01_inference_basics/README.md) |
| 2 | [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) | 把 prefill / decode / KV cache 的数量级一次讲透 | [M1](../modules/01_inference_basics/README.md) |
| 3 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 现代推理引擎的显存管理基线 | [M2](../modules/02_inference_engines/README.md) |
| 4 | [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention 与「LM 程序」抽象，agent 负载的起点 | [M2](../modules/02_inference_engines/README.md) |
| 5 | [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | 一家顶级团队公开的真实线上推理系统与成本数字 | [M3](../modules/03_deepseek_stack/README.md) |
| 6 | [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) | 模型结构、硬件与系统如何协同设计 | [M3](../modules/03_deepseek_stack/README.md) |
| 7 | [Mooncake](https://arxiv.org/abs/2407.00079) | 以 KV cache 为中心的 P/D 分离架构（Kimi 的服务平台） | [M5](../modules/05_cache_prefill/README.md) |
| 8 | [Autellix](https://arxiv.org/abs/2502.13965) | 把 agent 程序当调度的一等公民 | [M4](../modules/04_harness_workloads/README.md) / [M9](../modules/09_frontier/README.md) |
| 9 | [AReaL](https://arxiv.org/abs/2505.24298) | 完全异步 RL 与 staleness 控制 | [M6](../modules/06_long_tail_async/README.md) |
| 10 | [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300) | 组内基线的出处，后面所有 critic-free 方法的母版 | [M7](../modules/07_multi_agent_credit/README.md) |
| 11 | [COMA](https://arxiv.org/abs/1705.08926) | 多 agent 反事实基线与 difference reward 的经典表述 | [M7](../modules/07_multi_agent_credit/README.md) |
| 12 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | 自我改进回路里「评估」如何变成主负载 | [M9](../modules/09_frontier/README.md) |

## 1. 推理基础

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 | 所有 FLOPs 与 KV cache 公式的出发点，只需看结构与复杂度表 | 略读 |
| [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) | Williams, Waterman, Patterson（CACM 52(4)） | 2009 | 算术强度与 ridge point 的原始定义，判断 prefill/decode 瓶颈的工具 | 精读 |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | kipply（博客） | 2022 | 用纸笔算 KV cache 大小和访存时间的入门范本 | 精读 |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) | Pope et al. | 2022 | 推理的切分策略与延迟/成本权衡的系统分析，面试高频 | 精读 |
| [How To Scale Your Model：All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) | Austin et al.（Google DeepMind） | 2025 | 把推理的数量级推导写成教科书，适合对照本项目的心算表 | 精读 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. | 2020 | loss 与模型规模、数据量、训练算力之间的幂律关系 | 略读 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | Shoeybi et al. | 2019 | 层内张量并行（TP）的原始做法，推理切分的基础 | 略读 |
| [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) | Korthikanti et al. | 2022 | 序列并行与选择性激活重算：训练侧的显存账 | 略读 |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | Shazeer | 2019 | MQA：减少 KV 头数来降低 decode 访存 | 略读 |
| [GQA](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 2023 | MHA 与 MQA 之间的折中，今天多数开源模型的默认 | 略读 |
| [RoFormer（RoPE）](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | 旋转位置编码，capstone 的迷你引擎也实现了它 | 略读 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Shazeer | 2020 | SwiGLU 等门控前馈层的出处，决定 FFN 的参数量与 FLOPs | 略读 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | Zhang, Sennrich | 2019 | RMSNorm：现代开源模型常用的归一化层（capstone 的迷你引擎也用它） | 略读 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Dao et al. | 2022 | IO-aware 的思想：按显存层级设计 kernel | 略读 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao | 2023 | 并行与工作划分的改进 | 略读 |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Shah et al. | 2024 | Hopper 上的异步与低精度 | 略读 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Leviathan, Kalman, Matias | 2022 | 用小模型草稿把带宽受限的 decode 变成验证 | 略读 |
| [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) | Yuan et al. | 2024 | 用 roofline 统一梳理推理优化技术的综述 | 略读 |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Grattafiori et al. | 2024 | 开源大模型的结构参数与训练/推理基础设施细节 | 略读 |
| [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) | Dean, Barroso（CACM） | 2013 | 扇出越大尾延迟越被放大，以及对冲请求等经典对策：长尾问题的系统出处 | 精读 |
| [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383) | Little（Operations Research） | 1961 | Little 定律：并发数 = 到达率 × 停留时间，估算并发沙箱、在途请求与 KV 占用的工具 | 略读 |

## 2. 推理引擎

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Yu et al.（Seoul National University、FriendliAI） | 2022 | iteration-level 调度（continuous batching）的起源 | 精读 |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Kwon et al. | 2023 | vLLM 的分页 KV 管理，现代引擎的显存管理基线 | 精读 |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | Zheng et al. | 2023 | RadixAttention 前缀复用 + `fork`/`gen` 程序原语 | 精读 |
| [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) | LMSYS（博客） | 2024 | RadixAttention 的直观图解 | 略读 |
| [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) | LMSYS（博客） | 2024 | 零开销批调度器与 cache-aware 负载均衡器 | 略读 |
| [SARATHI](https://arxiv.org/abs/2308.16369) | Agrawal et al. | 2023 | Sarathi-Serve 的前作：chunked prefill + 把 decode 捎带进 prefill 批次 | 略读 |
| [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Agrawal et al. | 2024 | chunked prefill：把长 prefill 切块与 decode 混批 | 精读 |
| [DistServe](https://arxiv.org/abs/2401.09670) | Zhong et al. | 2024 | prefill/decode 分离以优化 goodput | 精读 |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Patel et al. | 2023 | 按阶段把 prefill 与 decode 放到不同机器 | 略读 |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | Qin et al. | 2024 | Kimi 的服务平台：P/D 分离 + 利用 CPU/DRAM/SSD 的分布式 KV cache | 精读 |
| [LMCache](https://arxiv.org/abs/2510.09665) | Liu et al. | 2025 | 面向企业规模的独立 KV cache 层 | 略读 |
| [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | vLLM 项目（博客） | 2025 | vLLM 核心架构的 V1 重写 | 略读 |
| [vLLM：Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/stable/design/prefix_caching/) | vLLM 项目 | — | 生产引擎里前缀缓存按 block hash 的实现方式 | 精读 |
| [Hydragen](https://arxiv.org/abs/2402.05099) | Juravsky et al. | 2024 | 把共享前缀与各自后缀的注意力分开计算，长共享前缀下大 batch 解码吞吐几乎不掉 | 略读 |
| [Cascade Inference](https://flashinfer.ai/2024/02/02/cascade-inference.html) | FlashInfer（博客） | 2024 | 共享前缀的级联注意力，省显存带宽 | 略读 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Frantar et al. | 2022 | 训练后权重量化的经典方法 | 略读 |
| [AWQ](https://arxiv.org/abs/2306.00978) | Lin et al. | 2023 | 按激活分布保护重要权重的量化 | 略读 |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | Chen et al. | 2023 | 投机采样的另一篇代表作：修正的拒绝采样保证输出分布与目标模型一致 | 略读 |
| [EAGLE](https://arxiv.org/abs/2401.15077) | Li et al. | 2024 | 在特征层做自回归草稿的投机解码 | 略读 |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | Li et al. | 2025 | 改为直接预测 token、融合多层特征（training-time test），让草稿模型能吃到更多训练数据的收益 | 略读 |
| [MagicDec](https://arxiv.org/abs/2408.11049) | Sadhukhan et al. | 2024 | 中长上下文、大 batch 的高吞吐区间里投机解码仍能加速 | 略读 |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) | Oliaro et al. | 2024 | 面向 agent 负载里高度重复、可预测序列的无模型投机解码 | 略读 |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | GeeeekExplorer | — | 约 1,200 行 Python 的轻量 vLLM 实现，读懂引擎结构的捷径 | 略读 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | vLLM 项目 | — | 代码：调度器与 KV 管理 | 略读 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | SGLang 项目 | — | 代码：radix cache 与路由 | 略读 |

## 3. DeepSeek

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | Dai et al. | 2024 | 细粒度专家 + 共享专家，DeepSeek MoE 结构的出处 | 精读 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | DeepSeek-AI | 2024 | MLA（多头潜在注意力）的出处，KV cache 压缩的核心 | 精读（MLA 一节） |
| [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) | Wang et al. | 2024 | V3 采用的无辅助损失负载均衡 | 略读 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek-AI | 2024 | MoE、FP8 训练、DualPipe 与部署方案 | 精读（基础设施各节） |
| [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) | Zhao et al.（DeepSeek-AI） | 2025 | 模型—硬件协同设计的反思，最接近「infra 工程师视角」的一篇 | 精读 |
| [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | DeepSeek-AI（open-infra-index） | 2025 | 线上推理系统的 EP 部署、P/D 分离与成本利润数字 | 精读 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Shao et al. | 2024 | GRPO 的出处（第 4 节） | 精读（第 4 节） |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek-AI | 2025 | 大规模 RL 激发推理能力，理解后训练负载从哪来 | 略读 |
| [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | Yuan et al. | 2025 | 硬件对齐、可训练的稀疏注意力 | 略读 |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek-AI | 2025 | DSA 稀疏注意力、后训练算力 >10% 预训练、1,827 个环境的 agentic 任务合成 | 精读 |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | DeepSeek-AI | 2026 | V4-Pro（1.6T 参数、49B 激活）与 V4-Flash（284B、13B 激活），百万 token 上下文的混合注意力 | 精读 |
| [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) | DeepSeek-AI | 2026 | 把 KV cache 压缩推到极限：长程 agent 让负载偏向输入，prefill 与 KV 的存储、带宽成为主瓶颈 | 精读 |
| [DeepSeek Elastic Compute (DSec)](https://arxiv.org/abs/2609.22978) | Huang et al.（DeepSeek-AI） | 2026 | 生产级沙箱平台：超过 38 万个并发沙箱，与 RL 框架协同设计 | 精读 |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | DeepSeek-AI | 2025 | MLA decode kernel | 略读 |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | DeepSeek-AI | 2025 | 专家并行通信库 | 略读 |
| [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) | DeepSeek-AI | 2025 | 双向流水线并行，计算与通信重叠 | 略读 |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | DeepSeek-AI | 2025 | GPU 上简洁高效的 BLAS（GEMM）kernel 库 | 略读 |
| [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB) | DeepSeek-AI | 2025 | 专家并行的负载均衡器 | 略读 |
| [deepseek-ai/3FS](https://github.com/deepseek-ai/3FS) | DeepSeek-AI | 2025 | 面向训练与推理的分布式文件系统，DSec 的镜像也放在上面 | 略读 |
| [DeepSeek API：Context Caching](https://api-docs.deepseek.com/guides/kv_cache) | DeepSeek | — | 默认开启的硬盘前缀缓存，只匹配前缀、尽力而为 | 略读 |

## 4. agent 负载与缓存

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | Yichao 'Peak' Ji（Manus，博客） | 2025 | 生产 agent 视角：KV cache 命中率是最重要的指标，输入输出比约 100:1 | 精读 |
| [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Anthropic（工程博客） | 2025 | 多 agent 系统约为聊天的 15 倍 token 消耗，token 量解释了 80% 的表现方差 | 精读 |
| [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Anthropic（工程博客） | 2025 | harness 如何管理上下文，理解「改写/压缩」从何而来 | 略读 |
| [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934) | Gim et al. | 2023 | 模块化的注意力状态复用 | 略读 |
| [CacheGen](https://arxiv.org/abs/2310.07240) | Liu et al. | 2023 | KV cache 的压缩与流式传输：从远端取 KV 的带宽账 | 略读 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | Yao et al. | 2024 | 非前缀位置的 KV 复用（RAG 场景） | 略读 |
| [EPIC](https://arxiv.org/abs/2410.15332) | Hu et al. | 2024 | 位置无关缓存：不要求前缀完全匹配也能复用 KV | 略读 |
| [CachedAttention](https://arxiv.org/abs/2403.19708) | Gao et al. | 2024 | 多轮对话中复用历史 token 的 KV，避免每轮重算 | 略读 |
| [Preble](https://arxiv.org/abs/2407.00023) | Srivatsa et al. | 2024 | 分布式的前缀感知调度：多实例间兼顾缓存复用与负载 | 略读 |
| [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312) | Cao et al. | 2025 | 前缀局部性与多租户公平之间的矛盾及调度算法 | 略读 |
| [Consistent Hashing with Bounded Loads](https://arxiv.org/abs/1608.01350) | Mirrokni, Thorup, Zadimoghaddam | 2016 | 带负载上界的一致性哈希：「亲和路由 + 溢出上限」的经典算法 | 略读 |
| [KVCache Cache in the Wild](https://arxiv.org/abs/2506.02634) | Wang et al. | 2025 | 大型云厂商的真实 KV 缓存负载刻画：淘汰策略的好坏高度依赖负载 | 精读 |
| [TokenCake](https://arxiv.org/abs/2510.18586) | Bian et al. | 2025 | 多 agent 应用的 KV 管理：空间争用与工具调用期间的闲置 | 略读 |
| [InferCept](https://arxiv.org/abs/2402.01869) | Abhyankar et al. | 2024 | 工具调用打断生成时 KV 的保留/换出/重算决策 | 精读 |
| [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) | Lin et al.（上海交通大学、Microsoft Research；[arXiv](https://arxiv.org/abs/2405.19888)） | 2024 | 让服务端看见应用层的数据依赖 | 略读 |
| [Autellix](https://arxiv.org/abs/2502.13965) | Luo et al. | 2025 | 程序级调度（PLAS/ATLAS）与程序级队头阻塞 | 精读 |
| [Continuum](https://arxiv.org/abs/2511.02230) | Li et al. | 2025 | 工具调用期间 KV 的 TTL 保留策略 | 精读 |
| [KVFlow](https://arxiv.org/abs/2507.07400) | Pan et al. | 2025 | 按 workflow 图预测复用、预取 KV | 略读 |
| [CacheScout（Learning Agent Execution for KV-Cache Management）](https://arxiv.org/abs/2608.14624) | Zhang et al. | 2026 | 在线学习 agent 转移来管理 KV | 略读 |
| [TOPAS](https://arxiv.org/abs/2608.25523) | Ni et al. | 2026 | 前缀保留与请求调度的联合决策 | 略读 |
| [ConServe（Observation, Not Prediction）](https://arxiv.org/abs/2606.01839) | Ding et al. | 2026 | 以会话为单位的 P/D 分离调度 | 略读 |
| [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849) | Liu et al. | 2026 | 让工具运行中上报进度，替代时长预测 | 精读 |
| [Dynamo：Agent Hints 文档](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-hints.md) | NVIDIA Dynamo 项目 | — | 工业界现成的 harness→引擎 hint 接口（priority、osl、speculative_prefill） | 略读 |
| [SGLang HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) | LMSYS（博客） | 2025 | 分层 KV 缓存与可插拔存储后端 | 略读 |
| [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) | llm-d（博客） | 2025 | 从 vLLM 前缀缓存到分布式缓存感知调度 | 略读 |
| [The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) | Song et al. | 2024 | 共享 KV cache 带来的时序侧信道：多租户缓存的隔离代价 | 略读 |
| [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/abs/2502.07776) | Gu et al. | 2025 | 用时序差异审计商用 API 的提示缓存是否跨用户共享 | 略读 |
| [I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/) | Wu et al.（南方科技大学、ByteDance；NDSS） | 2026 | 多租户 KV 共享下恢复他人 prompt 的攻击 | 略读 |

## 5. RL 训练系统（同步 / partial / 异步）

### 5.1 系统与框架

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [IMPALA](https://arxiv.org/abs/1802.01561) | Espeholt et al. | 2018 | actor–learner 解耦与 V-trace off-policy 修正，异步 RL 的经典 | 精读（V-trace 一节） |
| [Asynchronous RLHF](https://arxiv.org/abs/2410.18252) | Noukhovitch et al. | 2024 | LLM 场景下异步、off-policy RL 能容忍多少陈旧度 | 精读 |
| [HybridFlow（verl）](https://arxiv.org/abs/2409.19256) | Sheng et al. | 2024 | 同步 colocate 基线与 3D-HybridEngine 重分片 | 精读 |
| [OpenRLHF](https://arxiv.org/abs/2405.11143) | Hu et al. | 2024 | 基于 Ray + vLLM 的开源 RLHF 框架 | 略读 |
| [AReaL](https://arxiv.org/abs/2505.24298) | Fu et al. | 2025 | 完全异步 + 可中断 rollout + staleness-enhanced PPO，最高 2.77× | 精读 |
| [Beyond Ten Turns（ASearcher）](https://arxiv.org/abs/2508.07976) | Gao et al. | 2025 | 完全异步 RL 训练长程搜索 agent，训练中出现超过 100 轮工具调用 | 略读 |
| [PipelineRL](https://arxiv.org/abs/2509.19128) | Piché et al. | 2025 | in-flight 权重更新：带旧 KV 继续生成，实测 KL 只比重算略高 | 精读 |
| [StreamRL](https://arxiv.org/abs/2504.15930) | Zhong et al. | 2025 | 解耦架构下的 pipeline bubble 与 skewness bubble | 精读 |
| [APRIL](https://arxiv.org/abs/2509.18521) | Zhou et al. | 2025 | 超额发起 rollout、够数即停、未完成的回收续写 | 略读 |
| [Seer](https://arxiv.org/abs/2511.14617) | Qin et al. | 2025 | 利用同 prompt 样本的相似性做负载均衡与投机解码 | 略读 |
| [RollPacker](https://arxiv.org/abs/2509.21009) | Gao et al. | 2025 | tail batching：把会产生长尾回答的 prompt 集中到少数「长轮」，其余轮只跑短而均衡的 rollout，不放松同步 | 略读 |
| [ROLL](https://arxiv.org/abs/2506.06122) | Wang et al. | 2025 | 面向大规模的 RL 库：rollout 调度器、环境/奖励 worker | 略读 |
| [ROLL Flash](https://arxiv.org/abs/2510.11345) | Lu et al. | 2025 | ROLL 的原生异步扩展：环境级异步与冗余环境 | 略读 |
| [RollArt](https://arxiv.org/abs/2512.22560) | Gao et al. | 2025 | 多任务 agentic RL 的异构解耦部署，3,000+ GPU 实践 | 略读 |
| [AgentRL](https://arxiv.org/abs/2510.04206) | Zhang et al. | 2025 | 多轮多任务 agentic RL 的异步框架与容器化环境 | 略读 |
| [LlamaRL](https://arxiv.org/abs/2505.24034) | Wu et al. | 2025 | 分布式异步 RL 框架与跨设备直接显存访问的权重同步 | 略读 |
| [ProRL Agent](https://arxiv.org/abs/2603.18815) | Zhang et al. | 2026 | rollout-as-a-service：把 rollout 生命周期做成 API 服务 | 略读 |
| [Polar: Agentic RL on Any Harness at Scale](https://arxiv.org/abs/2605.24220) | Xu et al. | 2026 | 把 harness 当黑盒、重建 token-faithful 轨迹；收益强烈依赖 harness | 略读 |
| [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) | Moonshot AI | — | 推理引擎里更新模型权重的中间件 | 略读 |
| [THUDM/slime](https://github.com/THUDM/slime) | THUDM | — | Megatron + SGLang 的 RL 后训练框架，支撑 GLM 系列 | 略读 |
| [verl-project/verl](https://github.com/verl-project/verl) | verl 项目 | — | HybridFlow 的开源实现 | 略读 |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | AReaL 项目 | — | 异步 RL 的开源实现 | 略读 |

### 5.2 off-policy 修正与训推一致性

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | Schulman et al. | 2015 | 信任域与局部近似目标：理解 token 级代理目标为什么只在策略接近时成立 | 略读 |
| [Safe and Efficient Off-Policy Reinforcement Learning（Retrace）](https://arxiv.org/abs/1606.02647) | Munos et al. | 2016 | 截断迹系数的 off-policy 回报估计，与 V-trace 同族 | 略读 |
| [Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641) | Hilton, Cobbe, Schulman | 2021 | 把近端策略与行为策略解耦（decoupled PPO）的出处 | 精读 |
| [Mastering Complex Control in MOBA Games with Deep Reinforcement Learning](https://arxiv.org/abs/1912.09729) | Ye et al. | 2019 | dual-clip PPO：负优势时给目标加下界，陈旧数据上尤其重要 | 略读 |
| [Group Sequence Policy Optimization（GSPO）](https://arxiv.org/abs/2507.18071) | Zheng et al. | 2025 | 序列级重要性比率与序列级裁剪 | 略读 |
| [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374) | Zheng et al. | 2025 | token 级代理目标何时有效：训推差异与策略陈旧都要小 | 精读 |
| [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370) | Ma et al. | 2025 | MoE 的训推路由不一致与对齐 | 略读 |
| [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) | Qi et al. | 2025 | 训推不一致的根因归到 BF16 精度，改用 FP16 | 略读 |
| [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Horace He（Thinking Machines Lab，博客） | 2025 | 推理非确定性的根因是 kernel 缺乏 batch invariance | 精读 |

### 5.3 模型技术报告中的 RL 基础设施

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Kimi Team | 2025 | partial rollouts 应对长 CoT 的长尾 | 略读 |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | Kimi Team | 2025 | agentic 数据合成与 colocate RL 基础设施 | 略读 |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | GLM-4.5 Team | 2025 | 面向 agent/推理/编码的开源 MoE 及其后训练 | 略读 |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) | MiniMax | 2025 | CISPO：裁剪重要性权重而不是裁剪更新 | 略读 |
| [Magistral](https://arxiv.org/abs/2506.10910) | Mistral-AI | 2025 | 生成中途替换权重、带着稍旧的 KV 继续生成 | 略读 |
| [rStar2-Agent](https://arxiv.org/abs/2508.20722) | Shang et al. | 2025 | 在 64 张 MI300X 上做 agentic RL：高吞吐代码环境与 GRPO-RoC | 略读 |

## 6. 信用分配

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Buy 4 REINFORCE Samples, Get a Baseline for Free!](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/) | Kool, van Hoof, Welling（ICLR 2019 workshop） | 2019 | 留一法基线（RLOO）的出处 | 略读 |
| [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) | Ahmadian et al. | 2024 | 把 RLOO 带回 LLM 后训练，论证 PPO 的很多部件不必要 | 精读 |
| [Understanding R1-Zero-Like Training（Dr. GRPO）](https://arxiv.org/abs/2503.20783) | Liu et al. | 2025 | GRPO 的长度偏置与难度偏置，去掉 1/\|o\| 与 std 归一化 | 精读 |
| [DAPO](https://arxiv.org/abs/2503.14476) | Yu et al. | 2025 | Clip-Higher、Dynamic Sampling 等四个技巧；动态采样意味着额外 rollout | 精读 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | Lightman et al. | 2023 | 过程监督优于结果监督；PRM800K | 略读 |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | Wang et al. | 2023 | 用从某一步出发的 N 次补全自动标注过程奖励 | 略读 |
| [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) | Cui et al. | 2025 | 只用结果标签在线更新隐式 PRM | 略读 |
| [VinePPO](https://arxiv.org/abs/2410.01679) | Kazemnejad et al. | 2024 | 用蒙特卡洛 rollout 代替 value network，信用更准但每步更贵 | 精读 |
| [GiGPO: Group-in-Group Policy Optimization](https://arxiv.org/abs/2505.10978) | Feng et al. | 2025 | 用「锚点状态」在组内构造 step 级分组，无额外 rollout | 精读 |
| [Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment](https://arxiv.org/abs/2505.11821) | Wei et al. | 2025 | turn 级奖励下的 GRPO/PPO 变体 | 略读 |
| [TRACE](https://arxiv.org/abs/2607.13988) | Tao et al. | 2026 | 用冻结参考模型的答案对数概率构造 turn 级 TD 奖励 | 略读 |
| [GACA](https://arxiv.org/abs/2609.12424) | Liang et al. | 2026 | 按 NLL 自适应地混合 step 级与 episode 级优势 | 略读 |
| [Coverage, Not Targeting](https://arxiv.org/abs/2609.02417) | Zhou et al. | 2026 | 终局验证器下均匀分配的稠密奖励胜过「瞄准关键 turn」，附对照设计 | 精读 |
| [TreeRL](https://arxiv.org/abs/2506.11902) | Hou et al. | 2025 | 相同 token 预算下从高不确定性的中间步分叉，而不是随机分叉 | 略读 |
| [Agentic Reinforced Policy Optimization（ARPO）](https://arxiv.org/abs/2507.19849) | Dong et al. | 2025 | 工具调用之后按熵自适应分支 | 略读 |
| [First Return, Entropy-Eliciting Explore（FR3E）](https://arxiv.org/abs/2507.07017) | Zheng et al. | 2025 | 找出高不确定性的决策点做定向 rollout | 略读 |
| [TreePO](https://arxiv.org/abs/2508.17445) | Li et al. | 2025 | 摊销公共前缀的树形采样，节省 22%–43% GPU 小时 | 精读 |
| [Tree Search for LLM Agent Reinforcement Learning（Tree-GRPO）](https://arxiv.org/abs/2509.21240) | Ji et al. | 2025 | 共享前缀，在固定 token / 工具调用预算内得到更多 rollout | 略读 |
| [Counterfactual Multi-Agent Policy Gradients（COMA）](https://arxiv.org/abs/1705.08926) | Foerster et al. | 2017 | 反事实基线；对 difference reward 代价的清晰讨论 | 精读 |
| [Optimal Payoff Functions for Members of Collectives](https://doi.org/10.1142/S0219525901000188) | Wolpert, Tumer（Advances in Complex Systems 4(2–3)） | 2001 | difference reward / Wonderful Life Utility 一族奖励 | 略读 |
| [A Value for n-Person Games](https://doi.org/10.1515/9781400881970-018) | Shapley（Contributions to the Theory of Games II） | 1953 | Shapley 值的原始定义 | 略读 |
| [Value-Decomposition Networks](https://arxiv.org/abs/1706.05296) | Sunehag et al. | 2017 | VDN：团队 Q 值按 agent 相加分解；lazy agent 问题 | 略读 |
| [QMIX](https://arxiv.org/abs/1803.11485) | Rashid et al. | 2018 | 单调混合网络的值分解 | 略读 |
| [Shapley Q-value](https://arxiv.org/abs/1907.05707) | Wang et al. | 2019 | 把 Shapley 值用作每个 agent 的 critic | 略读 |
| [LLM Collaboration With Multi-Agent Reinforcement Learning（MAGRPO）](https://arxiv.org/abs/2508.04652) | Liu et al. | 2025 | 多 agent 版 GRPO：联合奖励 + 组内中心化优势 + 分散执行 | 精读 |
| [Stronger-MAS（AT-GRPO）](https://arxiv.org/abs/2510.11062) | Zhao et al. | 2025 | 为什么 GRPO 分组在多 agent 下失效；按 agent×turn 分组与多策略训练系统 | 精读 |
| [Exact Is Easier: Credit Assignment for Cooperative LLM Agents（C3）](https://arxiv.org/abs/2603.06859) | Chen et al. | 2026 | 文本历史可精确恢复 → 固定历史采样 + 留一法基线；agent 移除式反事实有偏 | 精读 |
| [Counterfactual Credit Policy Optimization（CCPO）](https://arxiv.org/abs/2603.21563) | Li et al. | 2026 | 移除某个 agent 的反事实边际贡献，每个样本一条额外 rollout | 略读 |
| [MAPoRL](https://arxiv.org/abs/2502.18439) | Park et al. | 2025 | 多 LLM 讨论 + verifier 打分的多 agent 共训练 | 略读 |
| [MALT](https://arxiv.org/abs/2412.01928) | Motwani et al. | 2024 | 生成—验证—修正的异构 agent 链，用搜索树 + 价值迭代分配信用 | 略读 |
| [ReMA](https://arxiv.org/abs/2503.09501) | Wan et al. | 2025 | 元思考 agent 与推理 agent 的分层多 agent RL | 略读 |
| [SPIRAL](https://arxiv.org/abs/2506.24119) | Liu et al. | 2025 | 零和博弈自博弈；按角色的优势基线（RAE）防止思考崩塌 | 略读 |
| [SWEET-RL](https://arxiv.org/abs/2503.15478) | Zhou et al. | 2025 | 用训练期特权信息训练 step 级 critic | 略读 |
| [MARFT](https://arxiv.org/abs/2504.16129) | Liao et al. | 2025 | LLM 多 agent 强化微调的形式化与开放问题 | 略读 |
| [From Reasoning to Agentic: Credit Assignment in RL for LLMs](https://arxiv.org/abs/2604.09459) | Zhang | 2026 | 信用分配综述，覆盖 2024 年 1 月至 2026 年 7 月 | 略读 |
| [RL for LLM-based Multi-Agent Systems through Orchestration Traces](https://arxiv.org/abs/2605.02801) | Zhang | 2026 | 从编排轨迹看多 agent RL 的奖励与信用单位 | 略读 |

## 7. 奖励设计、LLM judge 与 reward hacking

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Zheng et al. | 2023 | LLM judge 的位置、冗长、自我偏好偏差 | 略读 |
| [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) | Gunjal et al. | 2025 | 把 rubric 作为不可验证领域的奖励 | 略读 |
| [Checklists Are Better Than Reward Models For Aligning Language Models](https://arxiv.org/abs/2507.18624) | Viswanathan et al. | 2025 | 用逐条 checklist 代替标量奖励模型 | 略读 |
| [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) | Gao, Schulman, Hilton | 2022 | reward hacking 的定量刻画：代理奖励越优化、真实奖励先升后降 | 精读 |
| [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) | Zhao et al. | 2025 | 单个符号就能骗出 judge 的正奖励 | 略读 |
| [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | Lilian Weng（博客） | 2024 | reward hacking 的系统综述 | 略读 |

## 8. 前沿（agent 服务化、agentic RL、RSI）

| 标题 | 作者 / 机构 | 年份 | 为什么读 | 深度 |
|---|---|---|---|---|
| [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547) | Zhang et al. | 2025 | agentic RL 的全景分类与开源环境/框架汇编 | 略读 |
| [Chain-of-Agents](https://arxiv.org/abs/2508.13167) | Li et al. | 2025 | 把多 agent 系统蒸馏进单个模型，再做 agentic RL | 略读 |
| [STaR](https://arxiv.org/abs/2203.14465) | Zelikman et al. | 2022 | 用自己答对的推理链迭代训练自己 | 略读 |
| [Voyager](https://arxiv.org/abs/2305.16291) | Wang et al. | 2023 | 技能库不断增长的开放式 agent | 略读 |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | Yuan et al. | 2024 | 模型给自己打奖励的迭代训练 | 略读 |
| [Gödel Agent](https://arxiv.org/abs/2410.04444) | Yin et al. | 2024 | 自指式地修改自身逻辑的 agent 框架 | 略读 |
| [Absolute Zero](https://arxiv.org/abs/2505.03335) | Zhao et al. | 2025 | 零外部数据的自出题自解题，代码执行器作奖励 | 略读 |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Zhang et al. | 2025 | 自改代码的 agent 档案库，SWE-bench 20.0% → 50.0% | 精读 |
| [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614) | Wang et al. | 2025 | 基准分 ≠ 自我改进潜力；按后代表现估计潜力来分配预算 | 精读 |
| [Hyperband](https://arxiv.org/abs/1603.06560) | Li et al. | 2016 | 逐步淘汰式的资源分配，评估调度的成熟基线 | 略读 |
| [SEAL: Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) | Zweiger et al. | 2025 | 模型生成自己的微调数据与更新指令 | 略读 |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | Novikov et al.（Google DeepMind 白皮书） | 2025 | 进化式编码 agent 改进基础设施与自身训练 | 精读 |
| [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046) | Gao et al. | 2025 | 按「改什么、何时改、怎么改」组织的综述 | 略读 |
| [A Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407) | Fang et al. | 2025 | 用统一反馈回路框架梳理自我进化 | 略读 |
| [SEAGym](https://arxiv.org/abs/2606.17546) | Zheng et al. | 2026 | 自我进化 harness 的评估环境：训练/验证/回放/成本记录 | 略读 |
| [ModularRSI](https://arxiv.org/abs/2609.14857) | Wu et al. | 2026 | 与评测集不相交的模块化 harness 自我改进 | 略读 |
| [ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://recursive-workshop.github.io/) | 组织者来自 KAUST、ByteDance、NYU/DeepMind 等 | 2026 | RSI 方向的问题框架与已接收论文 | 略读 |
| [SWE-bench/experiments](https://github.com/SWE-bench/experiments) | SWE-bench 项目 | — | 各提交的预测、执行日志与 trajectories：研究 agent 负载的公开 trace | 略读 |
| [CoMLRL（MAGRPO 代码）](https://github.com/OpenMLRL/CoMLRL) | MAGRPO 作者 | — | 多 agent GRPO 的开源实现，可直接读 advantage 的计算代码 | 略读 |
| [PettingLLMs（AT-GRPO 代码）](https://github.com/pettingllms-ai/PettingLLMs) | AT-GRPO 作者 | — | agent×turn 分组与多策略训练的开源实现 | 略读 |
| [C3 代码](https://github.com/EIT-EAST-Lab/C3) | C3 作者 | — | 固定历史重采与留一法基线的开源实现 | 略读 |
| [Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) | xxzcc | — | 持续更新的信用分配论文目录 | 略读 |
