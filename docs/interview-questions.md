**中文** · [English](interview-questions.en.md)

# 面试题库：67 题

> **用法**：按主题分 8 组，每题标难度：★ 基础、★★ 进阶、★★★ 深挖。
>
> **答题要点**统一按"**结论 → 数量级 → 取舍**"组织，这也是面试时开口的顺序：先给一句判断，再用一个能心算的数字支撑，最后说代价和适用边界。
>
> **数字的来历**：凡是没有标出处的数字，都是按公式推出来的，推导过程见[模块 08 §3 参数卡](../modules/08_system_design/README.md#3-参数卡本模块所有估算的共同输入)。
>
> **H800 参数**：H800 的算力和 HBM 带宽没有 NVIDIA 官方规格页，本题库按 H100 SXM 的 dense 值估算（BF16 989 / FP8 1,979 TFLOPS，3.35 TB/s），理由见同一节。
>
> **建议练法**：先遮住要点自己答，限时 2 分钟，再对照。

| 组 | 题数 | 对应模块 |
|---|---|---|
| [1. 推理基础](#1-推理基础9-题) | 9 | [模块 01](../modules/01_inference_basics/README.md) |
| [2. 推理引擎](#2-推理引擎9-题) | 9 | [模块 02](../modules/02_inference_engines/README.md) |
| [3. DeepSeek 系统](#3-deepseek-系统11-题) | 11 | [模块 03](../modules/03_deepseek_stack/README.md) |
| [4. Agent 负载与缓存](#4-agent-负载与缓存8-题) | 8 | [模块 04](../modules/04_harness_workloads/README.md)、[模块 05](../modules/05_cache_prefill/README.md) |
| [5. 长尾与异步 RL](#5-长尾与异步-rl9-题) | 9 | [模块 06](../modules/06_long_tail_async/README.md) |
| [6. 信用分配](#6-信用分配8-题) | 8 | [模块 07](../modules/07_multi_agent_credit/README.md) |
| [7. 系统设计](#7-系统设计7-题) | 7 | [模块 08](../modules/08_system_design/README.md) |
| [8. 行为与项目](#8-行为与项目6-题) | 6 | [模块 08 §9](../modules/08_system_design/README.md#9-如何在面试里讲你的-harnessinference-co-design-项目) |

## 1. 推理基础（9 题）

### 1.1 ★ 为什么 decode 通常受限于访存，而 prefill 受限于算力？

- **结论**：decode 每步每条序列只产出 1 个 token，却要把权重和 KV 全读一遍；prefill 一次处理成百上千个 token，读一次权重就能摊给所有 token。
- **数量级**：
  - BF16 下每个参数每个 token 算 2 FLOPs、占 2 字节，所以 batch 为 B 的 dense decode，算术强度约为 B FLOPs/Byte。
  - H100/H800 BF16 的屋脊点 ≈ 989 / 3.35 ≈ 295 FLOPs/Byte。B 要到约 300 才可能受限于算力；把读 KV 也算进去，实际更难达到。
- **取舍**：加大 batch 能提高吞吐，但 TPOT 变长，KV 占用也变大。
- **出处**：[Roofline（Williams et al.）](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)、[H100 规格页](https://www.nvidia.com/en-us/data-center/h100/)。

### 1.2 ★ 写出每 token KV cache 的公式，并算出 Llama-3-70B 和 DeepSeek-V3 的值。

- **结论**：GQA 为 $2L\,n_{kv}\,d_h\,b$；MLA 为 $L\,(r_{kv}+d_{rope})\,b$。
- **数量级**：
  - Llama-3-70B：2 × 80 × 8 × 128 × 2 = 320 KB。
  - DeepSeek-V3：61 × (512+64) × 2 = 70,272 B ≈ 70 KB（68.6 KiB）。
  - 128K 上下文时分别约 40 GiB 和 8.6 GiB。
- **取舍**：MLA 省显存，但 decode 时要么在 latent 上"吸收"权重做注意力（计算更多），要么把 latent 展开成每头的 K/V（显存和带宽更多）。
- **出处**：[DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)、[Meta llama-models](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)。

### 1.3 ★ TTFT、TPOT、吞吐、goodput 分别是什么？它们为什么互相冲突？

- **结论**：
  - TTFT（首 token 延迟）由排队和 prefill 决定；TPOT（每个输出 token 的间隔）由每步 decode 的时间决定。
  - 吞吐要大 batch，而大 batch 会拉长 TPOT；prefill 插进 decode 批次，又会让 TPOT 抖动。
  - goodput 指满足 SLO 的请求吞吐。
- **数量级**：DistServe 以 goodput 为优化目标，报告在 90% 以上请求满足延迟约束的前提下，可服务 7.4 倍的请求量，或满足紧 12.6 倍的 SLO。
- **取舍**：chunked prefill 是在同机上摊平干扰，PD 分离是从物理上隔离。
- **出处**：[DistServe](https://arxiv.org/abs/2401.09670)。

### 1.4 ★★ 从 MHA 到 MQA、GQA、MLA，各省了什么，又付出了什么？

- **结论**：
  - MQA：所有头共享一份 KV。
  - GQA：头分组，组内共享 KV。
  - MLA：缓存低秩的 latent，外加一小块解耦的 RoPE key。
- **数量级**：
  - DeepSeek-V2 报告，MLA 相对 DeepSeek 67B 把 KV cache 减少 93.3%，最大生成吞吐提升到 5.76 倍。
  - DeepSeek-V3 如果不压缩、按朴素 MHA 存 KV，每 token 要 61 × 128 × (192+128) × 2 ≈ 5.0 MB，约为 MLA 的 71 倍。
- **取舍**：MQA/GQA 可能损失质量，GQA 论文用约 5% 的原始预训练算力做 uptrain 来弥补；MLA 增加了计算量和实现复杂度，RoPE 需要解耦处理。
- **出处**：[MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)、[DeepSeek-V2](https://arxiv.org/abs/2405.04434)。

### 1.5 ★★ 上下文多长时，注意力的 FLOPs 会超过线性层？

- **结论**：每 token 的线性层 FLOPs 为 $2N_{act}$，注意力为 $a\cdot K$，其中 $a = 2L\,n_h(d_{qk}+d_v)$。两者相等时 $K^* = 2N_{act}/a$。
- **数量级**：DeepSeek-V3 按 prefill（不吸收权重）口径 $a$ ≈ 5.0M，$K^*$ ≈ 70G / 5.0M ≈ 14.1K；Llama-3-70B ≈ 137G / 2.6M ≈ 52K。这里 $N_{act}$ 不含 embedding 查表和 lm_head（DeepSeek-V3 为 35.1B，Llama-3-70B 为 68.4B）。MoE 的激活参数少，注意力更早占主导。
- **取舍**：
  - 在长上下文的 agent 负载下，prefill 优化要盯注意力 kernel（FlashAttention/FlashMLA）和稀疏注意力（NSA/DSA），光优化 GEMM 不够。
  - "MLA 省显存、不省算力"只在这个口径下成立。decode 用吸收模式时，每对注意力反而要 17.0M FLOPs（3.4 倍），是用算力换访存。

### 1.6 ★★ 投机解码什么时候有效，什么时候反而变慢？

- **结论**：decode 受限于访存、有空闲算力时，用便宜的草稿换每步多出几个 token，是划算的。接受率低，或者验证把每步的 token 数推过了屋脊点时，收益就没了。
- **数量级**：
  - Leviathan et al. 在 T5X 上加速 2–3 倍，输出分布不变。
  - DeepSeek-V3 的 MTP：第二个 token 的接受率为 85–90%，TPS 约为 1.8 倍。只有 1 个草稿 token 时，每步期望产出 ≈ 1 + p（p 为接受率）。
  - **dense 模型的心算阈值**：验证 k 个草稿，每步的 token 数变成 B·(k+1)。只要 B·(k+1) 低于屋脊点（BF16 约 295），验证几乎免费；k = 3 时，B 超过约 74 收益就开始消失。
  - **MoE**：验证的 token 多了，被激活的专家也多了。DeepSeek-V3 每 token 选 256 个专家中的 8 个，按均匀选择近似，1 个 token 覆盖 8 个专家，4 个 token 约 30 个，读专家权重的字节随之增加，"验证几乎免费"的前提变弱。
  - **MLA**：MTP 让每步有 2 个 query token，32K 上下文下 decode 注意力从正好在屋脊上变成算力受限（[模块 08 §4 第 3 步](../modules/08_system_design/README.md#4-演练-a用-h800-集群服务-1-万个并发-coding-agent前缀缓存--pd-分离)）。
- **取舍**：要付出草稿计算和验证的开销，KV 管理也更复杂；batch 越大收益越小。反过来，RL rollout 的长尾阶段只剩少数长序列、batch 很小，正是投机解码的主场：[Seer](https://arxiv.org/abs/2511.14617) 用自适应的分组投机解码加速 rollout 生成，并用上下文感知的调度缓解长尾请求的延迟。
- **出处**：[Speculative decoding](https://arxiv.org/abs/2211.17192)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[Seer](https://arxiv.org/abs/2511.14617)。

### 1.7 ★★★ 为什么 MoE 的 decode 需要很大的全局 batch 和大规模 EP？

- **结论**：每个专家只分到 batch 里的一小部分 token，读专家权重的开销摊不薄。
- **数量级**：
  - DeepSeek-V3 每 token 激活 256 个路由专家中的 8 个，每个专家的平均 batch ≈ 全局 batch / 32。
  - FP8 权重每参数 1 字节、每 token 2 FLOPs，所以专家 GEMM 的算术强度 ≈ 2 × 每专家 batch。
  - 要接近 FP8 屋脊点（≈ 1,979 / 3.35 ≈ 590 FLOPs/Byte），每专家 batch 要约 300，全局 batch 就要约 1 万 token/步。
  - DeepSeek 线上 decode 用的是 EP144，跨 18 个节点。
- **取舍**：
  - 大 EP 意味着 all-to-all 通信重、对慢卡敏感，需要 DeepEP、双 batch 重叠和 EPLB 配合。
  - **反例：长上下文会破坏这个前提。** 32K 的 agent 负载下，KV 容量把一个 EP144 实例的全局 batch 限在约 2,930 条流，每专家只有约 91 个 token。这时大 EP 的主要价值变成"给 KV 腾显存"：EP144 每卡权重 21.5 GB，EP32 是 37 GB。
- **出处**：[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)、[模块 08 §4](../modules/08_system_design/README.md#4-演练-a用-h800-集群服务-1-万个并发-coding-agent前缀缓存--pd-分离)。

### 1.8 ★ FlashAttention 解决了什么问题？

- **结论**：注意力的瓶颈在 IO。FlashAttention 用分块（tiling）计算，让中间的 N×N 矩阵不写回 HBM。它算的仍是精确注意力，不是近似。
- **数量级**：
  - 除输入输出外的额外显存从 O(N²) 降到 O(N)。
  - FlashAttention-3 在 H100 上，FP16 最高 740 TFLOPS（约 75% 利用率），FP8 接近 1.2 PFLOPS。
- **取舍**：每一代硬件都要重写 kernel；它不减少 FLOPs，这一点与稀疏注意力不同。
- **出处**：[FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-3](https://arxiv.org/abs/2407.08608)。

### 1.9 ★★★ MLA 前缀命中之后，prefill 的注意力怎么算？

- **结论**：缓存里存的是 512 维 latent，不是每头的 K/V，所以有两种算法：
  - 把缓存的 latent 升维成每头 K/V，再按普通 MHA 算。vLLM 称之为 "compute friendly"，用于 prefill 和 chunked prefill。
  - 不升维，直接在 latent 上用吸收后的权重算。vLLM 称之为 "data-movement friendly"，用于 decode。
- **数量级**：
  - 升维每个缓存 token 约 2 × 512 × 128 × 256 × 61 ≈ 2.05 GFLOPs。32K 前缀约 67 TFLOPs，比只算 2K 增量多 13.7%，整段重算与命中之比因此从 11.2 倍降到约 9.8 倍。
  - 吸收模式每个 (q,k) 对 17.0M FLOPs，比升维后的 5.0M 多 12.0M。增量 Δ < 2.05G / 12.0M ≈ 171 个 token 时，吸收模式更省。
- **取舍**：升维需要额外的 workspace 显存，vLLM 为此把上下文按固定大小分块处理；吸收模式省显存，但算得多。decode 侧小 chunk 的本地 prefill 适合用吸收模式。
- **出处**：[vLLM mla_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py)、[DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)。

## 2. 推理引擎（9 题）

### 2.1 ★ PagedAttention 解决了什么问题？

- **结论**：按请求预留连续显存来放 KV，会产生大量碎片和浪费。PagedAttention 像虚拟内存一样按块分配 KV，块还能在请求之间共享。
- **数量级**：vLLM 博客称此前的系统浪费了 60%–80% 的 KV 显存，vLLM 的浪费不到 4%；论文报告吞吐提升 2–4 倍。
- **取舍**：间接寻址需要专门的 kernel；块大小要在碎片和元数据量之间权衡。
- **出处**：[vLLM 论文](https://arxiv.org/abs/2309.06180)、[vLLM 博客](https://vllm.ai/blog/2023-06-20-vllm)。

### 2.2 ★ continuous batching（iteration-level scheduling）是什么？

- **结论**：按"迭代"而不是按"请求"调度。每一步都可以加入新请求、移出已完成的请求，不必等整批结束。
- **数量级**：Orca（OSDI'22）在 GPT-3 175B 上，同等延迟下吞吐是 FasterTransformer 的 36.9 倍。
- **取舍**：prefill 和 decode 混在同一批里会干扰 TPOT，由此才有了 chunked prefill 和 PD 分离。Orca 用 selective batching 处理注意力里形状不规则的部分。
- **出处**：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)。

### 2.3 ★★ chunked prefill 带来什么，又要付出什么？

- **结论**：把长 prefill 切成大小相近的块，和 decode 一起调度，避免一个长 prompt 让所有 decode 停顿（stall-free）。
- **数量级**：Sarathi-Serve 报告，相比 vLLM，Mistral-7B 单卡 A100 的服务容量提升到 2.6 倍，Yi-34B 最高 3.7 倍，Falcon-180B 最高 5.6 倍。
- **取舍**：块越小，TPOT 越平稳，但 prefill 效率下降、TTFT 变长；而且每一块都要重新读一遍前面已有的 KV。
- **出处**：[Sarathi-Serve](https://arxiv.org/abs/2403.02310)。

### 2.4 ★★ 前缀缓存（RadixAttention）怎么做？为什么淘汰时从叶子开始？

- **结论**：用 radix tree 按 token 组织 KV，新请求沿树匹配最长的公共前缀。SGLang 用 LRU 先淘汰叶子，因为叶子被淘汰后，祖先节点仍可被其他分支复用，直到它们也变成叶子。
- **数量级**：SGLang 报告吞吐最高提升 6.4 倍；DeepSeek 线上有 56.3% 的输入 token 命中落盘缓存。
- **取舍**：命中率取决于 harness 能否保持前缀稳定；纯 LRU 对 agent 的工具等待不友好（见 4.2）。
- **出处**：[SGLang](https://arxiv.org/abs/2312.07104)、[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)。

### 2.5 ★★ PD 分离带来什么，又要付出什么？

- **结论**：prefill 和 decode 放在不同实例上，彼此不再干扰，各自可以选并行方式和 batch 大小。代价是要在两者之间搬 KV，还要调两个池子的配比。
- **数量级**：
  - DistServe 报告 7.4 倍请求量 / 12.6 倍更紧的 SLO。
  - Splitwise 报告同等成本下吞吐 1.4 倍（成本还低 20%），或同等成本和功耗预算下吞吐 2.35 倍。
  - DeepSeek-V3 一轮（32K 前缀 + 2K 增量 = 34.8K token）的整段 KV 约 2.45 GB，走单块 50 GB/s 网卡约 49 ms；只传增量约 0.14 GB（见 2.9）。
- **取舍**：增量很短的请求直接在本地做 prefill（条件分离，见 2.6）。
- **出处**：[DistServe](https://arxiv.org/abs/2401.09670)、[Splitwise](https://arxiv.org/abs/2311.18677)。

### 2.6 ★★★ 什么时候应该直接在 decode 节点上做 prefill？

- **结论**："扣掉前缀命中后的有效输入长度"很短，或者它占原始长度的比例很低时，本地做更划算。但在 DP 注意力 + 大 EP 的 decode 实例上，本地 prefill 会拖住整个实例，阈值要低得多。
- **数量级**：32K 上下文、2K 增量的情况下（模块 08 演练 A）：
  - decode 用 DP144 注意力 + EP144 专家，每个 MoE 层都有同步的 all-to-all。一个 rank 做本地 prefill，其余 143 张卡都在等它，卡住的是整个实例约 2,900 条流。
  - 2K 全在一个 rank 上做约 0.69 s（600 TFLOPS）；如果每步只给它加 10 ms，1 秒内只能做约 215 个 token。
  - 远程做则要搬 KV：整段每轮 2.45–4.75 GB，或者只传增量约 0.14 GB（见 2.9）。
- **取舍**：NVIDIA Dynamo 用 `eff_isl_threshold`（默认 2048）和 `eff_isl_ratio_threshold`（默认 0.7）两个阈值做这个决定，另有 `decode_busy_threshold` 在 decode 侧 KV 占用过高时强制走远程。按上面的估算，DP 注意力的 MLA 部署应把长度阈值降到几百 token，并在各 DP rank 之间均衡本地 prefill 的预算。
- **出处**：[Dynamo conditional disaggregation](https://docs.nvidia.com/dynamo/dev/advanced-customizations/conditional-disaggregation)、[模块 08 §4](../modules/08_system_design/README.md#4-演练-a用-h800-集群服务-1-万个并发-coding-agent前缀缓存--pd-分离)。

### 2.7 ★★ KV 显存不够、要抢占请求时，该用 swap 还是 recompute？

- **结论**：swap 把 KV 换出到主存，恢复时再搬回；recompute 直接丢掉 KV，恢复时重新 prefill。vLLM 论文对比过这两种方式。到了 vLLM V1，默认已经是 RECOMPUTE，官方文档的理由是 "recomputation has lower overhead in the V1 architecture"。
- **数量级**：DeepSeek-V3 34.8K 上下文的 KV 为 2.45 GB。
  - swap：经 PCIe（有效约 50 GB/s）搬回约 49 ms。
  - recompute：约 5.5 PFLOPs，按 600 TFLOPS/卡约 9.1 卡·秒。但有前缀缓存时，被抢占请求释放的块往往还能命中，实际要重算的只是没命中的部分。
- **取舍**：今天的主流组合是"重算 + 前缀缓存 + 分层下放"。swap 的角色交给了 KV 下放层，例如 vLLM 的 OffloadingConnector（把已完成的 KV 块下放到更慢、更大的层）或 LMCache。短序列重算更便宜，长序列要确保下放层接得住。
- **出处**：[vLLM](https://arxiv.org/abs/2309.06180)、[vLLM 调优文档](https://docs.vllm.ai/en/stable/configuration/optimization/)、[vLLM KV Offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)。

### 2.8 ★★★ 缓存亲和路由和负载均衡怎么兼顾？

- **结论**：
  - 先按前缀或会话的哈希选首选实例。
  - 首选实例超过负载上限时，溢出到次选实例（例如 power-of-two choices），同时从分布式缓存把前缀拉过去。
- **数量级**：
  - 模块 08 演练 C 中，如果每次命中都要从 L2/L3 读，需要约 4.9 TB/s 的网络带宽。所以必须靠亲和路由让大部分命中留在本地 HBM。
  - The Tail at Scale 指出，按入队时的队列长度在两台服务器中二选一，负载均衡效果比均匀随机好得多（原文："exponentially improves"）。
- **取舍**：负载上限设得太低，命中率下降；设得太高，热点排队。Dynamo 的 KV router 用各 worker 上报的 KV 事件维护一棵全局前缀树。
- **出处**：[The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)、[Dynamo router](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/router-design)。

### 2.9 ★★★ PD 分离下，KV 怎么传？

- **结论**：先想清楚数据流，再谈带宽。整段推送、只推增量、两侧都按会话亲和保留前缀，三者每轮搬的字节差一个数量级。传输要逐层流水、与计算重叠，并用能聚合多网卡带宽的传输引擎。
- **数量级**（DeepSeek-V3，32K 前缀 + 2K 增量，328 轮/s，模块 08 演练 A）：
  - prefill 先读 2.30 GB 前缀，再推整段 2.45 GB：4.75 GB/轮，全集群 1.56 TB/s。
  - prefill 读前缀，只推回增量 0.14 GB：2.45 GB/轮。
  - 两侧都保留前缀：约 0.17 GB/轮，全集群 56 GB/s，少约 14 倍。
  - 命中轮在一个 EP32 实例上约 50 ms，整段 2.45 GB 走单网卡约 49 ms，逐层重叠正好能藏住。
- **取舍**：
  - **逐层传输**：Splitwise 原话 "Layer-wise KV-cache transfer happens in parallel with the prompt computation for the next layer"，小 prompt 则改用串行传输；Mooncake 用 layer-wise prefill 把传输和落盘与计算重叠。
  - **传输引擎**：Mooncake Transfer Engine 做多网卡带宽聚合，NIXL 是 Dynamo 用的点对点传输库；DistServe 用 pull 而不是 push。
  - **布局不一致时要重排**：GQA 在 TP 下按 KV 头切分，P/D 两侧 TP 度不同就要重新切；MLA 的 latent 所有头共用，不按头切，按请求整体搬即可。
  - **两侧都保留前缀的代价**：prefill 侧要为所有会话留上下文，演练 A 里约 25 TB，主要放在主存或 SSD。
- **出处**：[Splitwise](https://arxiv.org/abs/2311.18677)、[Mooncake](https://arxiv.org/abs/2407.00079)、[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)、[NIXL](https://github.com/ai-dynamo/nixl)、[DistServe](https://arxiv.org/abs/2401.09670)。

## 3. DeepSeek 系统（11 题）

### 3.1 ★ MLA 缓存的是什么？为什么要把 RoPE 单独解耦？

- **结论**：每层只缓存 512 维的压缩 latent $c_{KV}$ 和 64 维所有头共享的 RoPE key。DeepSeek-V2 指出 RoPE 与低秩 KV 压缩不兼容：旋转与位置有关，无法被吸收进投影矩阵。所以要单独留一小块带位置信息的 key。
- **数量级**：61 × (512+64) × 2 = 70,272 B/token。
- **取舍**：decode 时直接在 latent 上做注意力，计算量更大。FlashMLA 在 H800 上达到 3000 GB/s（受限于访存的配置）和 660 TFLOPS（受限于算力的配置）。
- **出处**：[DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

### 3.2 ★★ DeepSeek-V3 线上推理怎么部署？prefill 和 decode 的 EP 规模为什么不同？

- **结论**：
  - Day 6：prefill 单元 4 节点，路由专家 EP32、MLA/共享专家 DP32；decode 单元 18 节点，EP144/DP144。
  - 技术报告：给出的最小部署单元是 prefill 4 节点 32 卡，decode 40 节点 320 卡（EP320）。
- **数量级**：decode 每卡放 2 个路由专家 + 1 个共享专家；prefill 每卡放 9 个路由专家 + 1 个共享专家；两者都各有 32 个冗余专家。
- **取舍**：decode 需要大 batch，所以用大 EP；prefill 本身 token 就多，较小的 EP 就够了，但通信量大，要靠双 batch 重叠来隐藏。
- **出处**：[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)。

### 3.3 ★★ 双 batch 重叠（dual-batch overlap）解决什么问题？

- **结论**：大 EP 下 all-to-all 通信占比很高。把一个 batch 拆成两个微 batch，一个在计算时另一个在通信。decode 阶段还把注意力层拆成两步，用 5 级流水线。
- **数量级**：H800 节点内 NVLink 实际可达约 160 GB/s，跨节点每块 IB 网卡约 50 GB/s（按 40 GB/s 有效计），相差 3–4 倍。
- **取舍**：微 batch 变小，GEMM 效率下降；调度也更复杂。
- **出处**：[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)、[DeepSeek ISCA 2025](https://arxiv.org/abs/2505.09343)。

### 3.4 ★★ DeepEP 的 normal kernel 和 low-latency kernel 有什么区别？

- **结论**：
  - normal kernel 追求高吞吐，用于训练和 prefill。
  - low-latency kernel 只走 RDMA，用于 decode。
  - 另外还有一种基于 hook 的通信计算重叠方式，不占用 SM。
- **数量级**：legacy 文档里的测试环境是 H800 + CX7 400Gb/s：
  - 节点内 EP8 的 dispatch 带宽 153 GB/s；跨节点 EP32 约 58 GB/s。
  - 低延迟 kernel 在 EP8 下 dispatch 耗时 77 µs。
- **取舍**：当前 DeepEP README 已换成 V2（NCCL Gin 后端），上面这些数字在 legacy 文档里。引用时要说明版本。
- **出处**：[DeepEP legacy 文档](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)。

### 3.5 ★★ 无辅助损失的负载均衡和 EPLB 各在哪个层面起作用？

- **结论**：
  - 训练时，DeepSeek-V3 给每个专家的亲和分数加一个偏置项 $b_i$，这个偏置只用于选专家，动态调整以均衡负载，不引入干扰主目标的辅助损失。
  - 推理时，EPLB 复制热点专家作为冗余专家，并重新摆放专家的位置。
- **数量级**：
  - 技术报告：prefill 阶段设 32 个冗余专家，大约每 10 分钟调整一次。
  - EPLB：EP 规模较小的 prefill 用分层策略，EP 规模较大的 decode 用全局策略。
- **取舍**：训练时负载均衡，不等于线上也均衡，因为线上流量分布会变；冗余专家也要占显存。
- **出处**：[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[EPLB](https://github.com/deepseek-ai/EPLB)。

### 3.6 ★★★ 怎么解读 Day 6 里 545% 的成本利润率？

- **结论**：这是"假设所有 token 都按 R1 价格计费"算出的理论值。DeepSeek 自己说明，实际收入显著更低：V3 定价更低、网页和 App 免费、夜间自动打折。
- **数量级**：
  - 日成本 = 226.75 节点 × 8 卡 × 2 美元 × 24 h = 87,072 美元。
  - 理论日收入 562,027 美元。
  - 56.3% 的输入 token 命中落盘缓存。
- **取舍**：面试时要能把成本拆成"平均节点数 × 单价"。两大杠杆是命中率和昼夜利用率：DeepSeek 在夜间低负载时缩减推理节点，把资源给研究和训练。
- **出处**：[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)。

### 3.7 ★★ DeepSeek-V3 的 FP8 训练有哪些关键设计？

- **结论**：用细粒度量化缓解离群值：激活按 1×128 的 tile、权重按 128×128 的 block 分别计算缩放因子。
- **数量级**：
  - DeepGEMM 初版在 H800 上的 FP8 GEMM 最高约 1358 TFLOPS，2025-04 更新到 1550 TFLOPS。
  - 作为对照，按 H100 估的 FP8 dense 峰值是 1,979 TFLOPS。
- **取舍**：块级缩放让 kernel 更复杂；Day 6 的推理里，MLA 核心计算和 combine 传输仍用 BF16。
- **出处**：[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。

### 3.8 ★★★ DeepSeek-V3.2 的 DSA 改变了什么，没改变什么？

- **结论**：DSA 用一个 lightning indexer 为每个 query 挑出 top-2048 个 KV token，主模型的核心注意力从 O(L²) 降到 O(Lk)。indexer 本身仍是 O(L²)，但计算很轻。
- **数量级**：config.json 里 `index_topk` = 2048。在 128K 上下文下，主注意力只看约 1.6% 的 token。
- **取舍**：DSA 省的是计算，不是 KV 存储。所有 token 的 KV 仍然要存，而且还多出 indexer 自己的 key cache，所以缓存容量问题不变。同方向的前作 NSA 获得 ACL 2025 最佳论文。
- **出处**：[DeepSeek-V3.2-Exp 报告](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)、[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json)、[NSA](https://arxiv.org/abs/2502.11089)。

### 3.9 ★★ 为什么 DeepSeek 的 MLA 要用 DP 注意力，而不是 TP？

- **结论**：MLA 只有一个 KV "头"，也就是所有头共用的 latent。用 TP 切注意力时，每张卡都要存完整的 latent，KV cache 在 TP 组里重复存了 TP 度那么多份。改用 DP 注意力，每张卡只存自己那批请求的 KV。
- **数量级**：
  - TP8 时同一份 latent 存 8 份，节点的有效 KV 容量只有 DP 的 1/8。
  - SGLang 的原话："If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage"。它用 DP 注意力把 decode 吞吐最高提升到 1.9 倍。
  - DeepSeek 线上 prefill 用 DP32、decode 用 DP144 跑 MLA 和共享专家。
- **取舍**：DP 注意力要在各 rank 之间均衡注意力计算量，Day 6 的 Prefill/Decode Load Balancer 就是干这个的；每张卡还要各存一份注意力权重（MLA 约 11.4B 参数，FP8 约 11.4 GB）。
- **出处**：[SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)、[Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)。

### 3.10 ★★★ FP8 KV cache 怎么存？它会撞上什么墙？

- **结论**：FP8 KV 让容量接近翻倍，但要付 scale 的开销和精度风险。对 MLA 这种 decode 注意力本来就接近屋脊的结构，容量翻倍也意味着算力要翻倍。
- **数量级**：
  - FlashMLA 为 DeepSeek-V3.2 设计的格式：每层每 token 656 B = 512 个 FP8（e4m3）值 + 4 个 FP32 scale（每个管 128 个值）+ 64 个不量化的 BF16 RoPE 值。BF16 是 1,152 B，所以容量是 1.76 倍，不是 2 倍。
  - V4.1 的格式是 528 B：RoPE 也量化，scale 改成每 32 个值一个 e8m0。
  - 32K 上下文下，容量 ×1.76 让每卡流数从约 20 条变成约 36 条，MLA decode 注意力从约 18 ms 变成约 32 ms，加读权重约 40 ms，已经逼近 50 ms 的 TPOT 预算。
- **取舍**：V3.2 的格式把 RoPE 部分保留为 BF16，V4.1 则把它也量化了，这是精度和容量之间的取舍；scale 粒度越细越准，元数据也越多。卡数能省多少，要同时看容量和算力（[模块 08 §4 第 3 步](../modules/08_system_design/README.md#4-演练-a用-h800-集群服务-1-万个并发-coding-agent前缀缓存--pd-分离)）。
- **出处**：[FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

### 3.11 ★★ DeepSeek-V4 的压缩注意力对推理系统意味着什么？

- **结论**：V4 用 CSA（Compressed Sparse Attention）和 HCA（Heavily Compressed Attention）交错组成混合注意力，沿序列维压缩 KV，把 KV 容量问题缩小了一个数量级。但前缀缓存和 PD 传输的"单位"也变了。
- **数量级**：
  - CSA 每 4 个 token 压成 1 个 KV 条目，再选 top-k（Pro 为 1024，Flash 为 512）；HCA 每 128 个 token 压成 1 个。
  - 1M 上下文下，V4-Pro 的单 token 推理 FLOPs 只有 V3.2 的 27%，KV cache 只有 10%。
  - V4-Pro 为 1.6T 总参数 / 49B 激活，支持 1M 上下文。
- **取舍**：
  - 压缩条目覆盖多个 token，前缀缓存要按压缩边界对齐（推论）。DeepSeek API 的上下文缓存文档也改了：因为引入了 Sliding Window Attention，缓存以独立、完整的 "cache prefix unit" 为单位，必须完整匹配某个 unit 才算命中。
  - PD 传输和分层存储的字节数同比下降。模块 08 用 V3 做的演练里，和 KV 相关的结论（容量受限、传输代价）都要重新算。
- **出处**：[DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348)、[DeepSeek 上下文缓存文档](https://api-docs.deepseek.com/guides/kv_cache)、[模块 03 §5.1](../modules/03_deepseek_stack/README.md#51-deepseek-v4面向-100-万-token-的混合注意力)。

## 4. Agent 负载与缓存（8 题）

### 4.1 ★ agent 负载和聊天负载有哪些关键区别？

- **结论**：多轮对话、上下文只追加（append-only）、上下文很长、每轮之间要等工具、一次任务会扇出成多个请求，而且长度和时延都是重尾分布。
- **数量级**：
  - Anthropic 报告，agent 的 token 用量约为聊天的 4 倍，多 agent 系统约为 15 倍。
  - 模块 08 演练 A 的假设：每轮只追加约 2K token，却要带着 32K 的上下文。
- **取舍**：推理引擎应该以"会话"而不是"请求"为单位来优化：缓存保留、会话亲和、按轮次做 prefill。
- **出处**：[Anthropic 多 agent 研究系统](https://www.anthropic.com/engineering/multi-agent-research-system)。

### 4.2 ★★ 工具调用期间，KV 应该留在 HBM、下放，还是丢掉？

- **结论**：三选一，看工具要等多久，以及有没有下一级存储。亚秒级等待或 HBM 有富余 → 留在 HBM；秒到分钟级 → 下放主存并预取；所有层都 miss 才重算。
- **数量级**：以 DeepSeek-V3、decode 期间 35K 上下文为例：
  - 留在 HBM，一条流占每卡 KV 空间的 4.9%。
  - 下放主存：换出在等待期间异步完成；读回 2.46 GB 约 49 ms（单流下限），不耗 GPU 算力；有 harness 的完成提示时可以预取，把读回藏在工具等待里。
  - 重算约 9.1 卡·秒，是命中的约 10 倍。
  - "留在 HBM 与重算"的盈亏点约 185 s，只在没有下一级存储时才有意义。
- **取舍**：pin 得太久会挤占 decode 容量；下放要占 PCIe 和主存，很多工具同时返回时读回会出现突发。要做对这个决定，需要 harness 提供预计的工具时长和完成提示。

### 4.3 ★★ 哪些 harness 写法会破坏前缀缓存？

- **结论**：任何让前缀字节发生变化的写法都会：
  - system prompt 开头放时间戳或随机 ID；
  - 工具列表顺序不固定；
  - JSON 序列化结果不确定；
  - 压缩或改写上下文；
  - 在历史中间插入内容。
- **数量级**：32K 上下文下，整段 miss 的计算量约是命中的 10 倍（含 MLA 升维）；只要 10% 的轮次整段 miss，prefill 负载就 ×1.88。
- **取舍**：压缩能省 token，但会丢掉缓存。压缩点应放在块边界上，并批量压缩。Anthropic 的缓存前缀顺序是 tools → system → messages，所以动态内容要放在后面。
- **出处**：[Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。

### 4.4 ★★ 多个项目共享缓存时，容量怎么规划？

- **结论**：工作集 = 各项目共享前缀之和 + 各活跃会话私有后缀之和，再按每 token 的 KV 字节数换算成显存。
- **数量级**：每个 20K 前缀，DeepSeek-V3 需要 1.41 GB，每卡 50 GB KV 空间只放得下约 35 个；Llama-3-70B 只放得下约 7.6 个。
- **取舍**：按项目亲和路由能提高命中率，但热门项目会压垮单个实例。所以要设负载上限，超了就溢出到别的实例。

### 4.5 ★★★ 怎么知道命中率的上限在哪？

- **结论**：离线回放 trace，用 Belady 最优替换（假设知道未来的访问序列）算上限，再画出"缓存容量–命中率"曲线。但要讲清 Belady 在前缀缓存上的适用条件。
- **数量级**：
  - Belady 假设对象等大、未命中代价相同。前缀缓存不满足：一个块要可用，它的所有祖先块都得命中；不同长度前缀每字节 KV 的重算代价相差 4 倍以上（1K 前缀约 1.0 MFLOPs/B，100K 约 4.6 MFLOPs/B）。
  - 所以块级 Belady 只是"命中块数"的上界，可能相当宽松，不是"省下 FLOPs"的上界。报告时要同时给 Belady 块命中率和按前缀长度分桶的 FLOPs 节省。
  - 如果 LRU 已经接近上界，换淘汰策略没用，只能加容量或改 harness；差距大时，才值得在策略上下功夫。
- **取舍**：Belady 要用到未来信息，只能当上界；在线策略可以借助 harness 的提示（预计工具时长、会话结束）去逼近它。以省下的 FLOPs 为目标时，要用考虑代价的离线界。
- **出处**：[Belady 1966](https://doi.org/10.1147/sj.52.0078)。

### 4.6 ★★ KV 分层存储（HBM / 主存 / SSD）各层的时延和容量大致在什么量级？

- **结论**：HBM 最快但最小；主存是 TB 级，经 PCIe 读回要几十毫秒；SSD 或分布式文件系统每节点几十 TB，经 RDMA 读回。
- **数量级**：
  - DGX H100 参考配置：2 TB 主存，8 × 3.84 TB NVMe。
  - PCIe Gen5 x16 标称 128 GB/s（双向合计）。
  - 3FS 在 180 个存储节点上的聚合读吞吐约 6.6 TiB/s；KVCache 客户端单节点峰值读约 40 GiB/s。
- **取舍**：层级越深越便宜，但读回越慢。对比：从主存读回 2.3 GB 约 46 ms，而重算要 9 卡·秒。
- **出处**：[DGX H100](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)、[3FS](https://github.com/deepseek-ai/3FS)。

### 4.7 ★★ 公开的 prompt caching 定价反映了怎样的成本结构？

- **结论**：写入比普通输入贵，因为要付写带宽和存储；读取便宜一个数量级；更长的 TTL 另外收费。
- **数量级**（截至 2026-09 的官方文档）：
  - Anthropic：5 分钟写入 1.25×，1 小时写入 2×，读取 0.1×。
  - Gemini：显式缓存按 token·小时收存储费。
  - DeepSeek：默认落盘，命中价远低于未命中价。
- **取舍**：
  - 重算一个 14K 前缀（DeepSeek-V3）约 2.5 卡·秒，读回只要约 20 ms，所以读取按 0.1× 收费仍然有利润。
  - 但 token 命中率不等于省下的算力：70% 的 token 命中只省下约 60% 的 FLOPs，因为未命中的后缀仍要对整个前缀做注意力。
  - TTL 是厂商在存储成本和命中率之间调节的旋钮。
- **出处**：[Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[Gemini](https://ai.google.dev/gemini-api/docs/generate-content/caching)、[DeepSeek](https://api-docs.deepseek.com/guides/kv_cache)。

### 4.8 ★★★ 多租户共享 KV cache 有什么安全风险？怎么防？

- **结论**：是否命中会反映在 TTFT 上，形成计时侧信道，攻击者可以借此推断别人 prompt 的前缀。按租户加 salt 能堵住这一类，但堵不住所有计时侧信道。
- **数量级**：Gu et al.（ICML 2025）在 7 家 API 提供商处检测到跨用户的全局缓存共享；NDSS 2025 的 PROMPTPEEK 演示了通过 KV 共享泄露 prompt 的攻击。
- **取舍**：
  - vLLM 的 `cache_salt` 会被注入第一块的哈希，只有 salt 相同的请求才能复用缓存；代价是失去跨租户的去重。Anthropic 和 OpenAI 的文档都写明缓存不在组织之间共享。
  - salt 之后还剩的侧信道：共享队列的排队时延、HBM/PCIe/SSD 带宽争用、热点排队、公开命名空间的命中、共享淘汰池。
  - 对应的缓解：按租户排队和配额、对敏感租户做资源分区、对命中请求的 TTFT 做填充或分桶、样本量足够的统计审计。
- **出处**：[Gu et al.](https://arxiv.org/abs/2502.07776)、[PROMPTPEEK](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)、[vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)。

## 5. 长尾与异步 RL（9 题）

### 5.1 ★ 为什么同步 RL 的 rollout 会被长尾拖慢？

- **结论**：一批轨迹要全部完成才能更新。AReaL 原话是："wait until the longest output in the batch is completed"；DAPO 也写道，同步系统的生成时间"typically dominated by the generation of long-tail samples"。
- **数量级**：在 Pareto α = 1.5 的分布下，4,096 条轨迹里的最长一条，有 63% 的概率超过均值的 85 倍。即使截断在原均值的 3 倍（$9x_m$），截断后的均值也只有 $2.33x_m$，而几乎必然有一条触到上限，槽位利用率约 2.33 / 9 ≈ 26%。
- **取舍**：截断会引入偏差，因为长轨迹往往对应难题。
- **出处**：[AReaL](https://arxiv.org/abs/2505.24298)、[DAPO](https://arxiv.org/abs/2503.14476)。

### 5.2 ★★ partial rollout 是什么？

- **结论**：每轮给输出设一个固定的 token 预算，没生成完的轨迹存进 replay buffer，下一轮接着生成（Kimi k1.5）。Kimi K2 也用它来防止长尾轨迹阻塞整个 rollout。
- **数量级**：没有统一的硬数字，一般用两个量来评估：每步因等待长尾而闲置的槽位比例，以及跨多个版本的轨迹占比。
- **取舍**：一条轨迹会跨多个策略版本，所以要按 token 记录版本号和 logprob。
- **出处**：[Kimi k1.5](https://arxiv.org/abs/2501.12599)、[Kimi K2](https://arxiv.org/abs/2507.20534)。

### 5.3 ★★ 异步 RL 的 staleness 怎么控制，又怎么修正？

- **结论**：
  - **控制**：限制最大版本差 η。AReaL 称之为 maximum permitted staleness，η = 0 就是同步。
  - **修正**：用 decoupled PPO 这类目标，把行为策略和 proximal 策略分开，再做重要性采样修正。
- **数量级**：
  - AReaL 报告，在相同卡数下，训练速度最高是同步系统的 2.77 倍，最终性能持平或更好。
  - Noukhovitch et al. 在 LLaMA 3.1 8B 上，异步比同步快约 40%，最终性能相当。
- **取舍**：
  - η 越大吞吐越高，但数据越偏离当前策略（off-policy）。要监控 IS 比率的分布和被裁剪的比例、KL，以及训推 logprob 差（见 5.9）。
  - **比率取在哪一级**：10K token 的轨迹，如果把逐 token 比率连乘成序列级比率，方差随长度指数增长，不可用；所以用 token 级比率加裁剪，这是有偏的。
  - **一条轨迹跨多个版本**（partial rollout、可中断生成）：按 token 记版本号，每个 token 用它自己的行为策略算比率。
  - **GRPO 的组内基线**：组均值只在样本来自相同或相近版本时才可比，所以要按组控制 staleness。
- **出处**：[AReaL](https://arxiv.org/abs/2505.24298)、[Asynchronous RLHF](https://arxiv.org/abs/2410.18252)。

### 5.4 ★★ 权重同步的开销在哪？怎么做快？

- **结论**：瓶颈通常不在带宽，而在三件事：把训练的并行布局重排成推理布局、量化、全局协调。但先要把物理字节算对。
- **数量级**：
  - 671B 模型 FP8 权重是 671 GB，但推理侧实际要落地的更多：非路由的 17B 在 DP 下每张卡各存一份。decode 每卡约 22 GB，prefill 每卡约 40 GB。1 个 decode 实例 + 3 个 prefill 实例共落地约 7 TB。
  - 每张卡经自己的 50 GB/s 网卡收完的下限只有 0.44–0.8 s。
  - Kimi K2 指出，用网络文件系统重排和广播所需的聚合带宽 "reaches several petabytes per second"，所以用和训练节点同机的 checkpoint engine，1T 参数全量更新不到 30 s。
  - LlamaRL 用 DDMA 更新 TB 级模型约 2 s；作为对照，论文里 OpenRLHF 传输 70B 权重要 111 s。
- **取舍**：双缓冲能减少停顿，但要多占显存；异步场景下可以一边生成一边加载新权重。
- **出处**：[Kimi K2](https://arxiv.org/abs/2507.20534)、[LlamaRL](https://arxiv.org/abs/2505.24034)。

### 5.5 ★★★ 为什么 GRPO 会让长尾在"组"这一级重新出现？

- **结论**：组内相对优势要等同一个 prompt 的 G 个样本全部完成，一组的完成时间就是这 G 个里最慢的那个。
- **数量级**：G = 16 时，16 个样本全部落在 p90 以内的概率只有 0.9¹⁶ ≈ 18.5%；G = 64 时约 0.1%。
- **取舍**：减小 G 会让基线估计变差。也可以在组内部分样本完成后，先用留一法（leave-one-out）之类的基线，但会引入偏差。
- **出处**：[DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)。

### 5.6 ★★ 训推同卡（colocate）和分离部署怎么选？

- **结论**：
  - colocate：例如 verl/HybridFlow 用 3D-HybridEngine 在训练和生成之间重排权重。好处是省掉跨集群搬权重，卡也不会整段闲置。
  - 分离部署：便于做异步，训练和推理可以各自扩缩容。
- **数量级**：
  - HybridFlow 报告吞吐是基线的 1.53–20.57 倍。
  - Kimi k1.5 的 colocate 方案，从训练切到推理不到 1 分钟，从推理切回训练约 10 秒。
- **取舍**：colocate 本质上仍是同步的，长尾问题依旧。
- **出处**：[HybridFlow](https://arxiv.org/abs/2409.19256)、[Kimi k1.5](https://arxiv.org/abs/2501.12599)。

### 5.7 ★★ DAPO 的 dynamic sampling 对系统有什么影响？

- **结论**：过采样，再过滤掉准确率为 1 或 0 的 prompt 组（这些组梯度为零），保证每个 batch 里有效样本的数量。
- **数量级**：
  - DAPO 在 Qwen2.5-32B 上 AIME 2024 得 50 分，超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分，训练步数只用了一半。
  - 消融实验中，加上 dynamic sampling 这一步把分数从 42 提到 50。
- **取舍**：要多生成样本，rollout 的负载和长尾都更重。DAPO 认为在同步系统里，生成时间本来就由长尾决定，所以额外的效率损失不一定大。
- **出处**：[DAPO](https://arxiv.org/abs/2503.14476)。

### 5.8 ★★★ 异步 buffer 为什么会产生长度偏差？怎么缓解？

- **结论**：短轨迹先完成、先进 buffer。按到达顺序取 batch，相当于按时长做了截尾采样，会系统性地偏向短的任务，而短任务往往也更容易。
- **数量级**：可以这样量化：比较 batch 内轨迹长度（或任务难度）的分布和整体分布，看两者差多少；再按难度分桶，看各桶被丢弃或超龄的比例。
- **取舍**：缓解办法有按组 / 按 prompt 取样、按长度分层取样、给超龄轨迹留名额；代价是要多等一会儿。

### 5.9 ★★★ 训推不一致从哪来？怎么修？

- **结论**：即使权重相同，推理引擎（vLLM/SGLang）和训练框架（FSDP/Megatron）算出的 token 概率也可能差很多，on-policy 的 RL 因此悄悄变成了 off-policy。MoE 还多一层：两侧选中的专家可能不同。
- **数量级**：
  - Yao et al. 的原话："the implementation gap implicitly turns the on-policy RL to be off-policy"。他们试过让 vLLM 返回实际采样用的概率、强制 lm_head 用 FP32，差距 "still exists"。
  - 修法之一是 TIS（截断重要性采样）：用 min(π_learner / π_sampler, C) 加权，C 是超参数。
  - R3（Rollout Routing Replay）把推理侧的路由分布记录下来，在训练时重放，用来降低训推 KL、防止 MoE 的 RL 训练崩溃。
- **取舍**：
  - 在训练侧重算 old logprob 最稳，但多一次前向。模块 08 演练 B 里，训练步长因此从约 585 s 变成约 731 s。
  - TIS 引入偏差，C 要调；R3 要在推理侧记录并传输路由信息。
  - 监控：同一批 token 在推理侧和训练侧的 KL，以及 MoE 路由一致率。
- **出处**：[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)、[R3](https://arxiv.org/abs/2510.11370)。

## 6. 信用分配（8 题）

### 6.1 ★ 多 agent 信用分配的核心问题是什么？

- **结论**：团队只拿到一个总奖励。每个 agent 都拿它当自己的学习信号，别的 agent 的动作带来的随机性，就变成了自己梯度里的噪声。
- **数量级**：可以这样量化：在每个 agent 真实贡献已知的合作任务上，比较各种估计器给出的梯度方差，以及学到同一回报所需的样本数。agent 越多，噪声越大。
- **取舍**：信用估计得越细，需要的计算或反事实评估就越多。

### 6.2 ★★ difference rewards 是什么？

- **结论**：$D_i = G(z) - G(z_{-i} \cup c_i)$，即总奖励减去"把 agent i 的动作换成默认动作 $c_i$"之后的总奖励，只保留 i 自己带来的那部分。
- **数量级**：每个 agent 每次要多做一次反事实评估，n 个 agent 就是 n 次。
- **取舍**：前提是能重放环境，或者有奖励模型；默认动作选得不同，偏差也不同。
- **出处**：[Wolpert & Tumer, An Introduction to Collective Intelligence](https://arxiv.org/abs/cs/9908014)。

### 6.3 ★★ COMA 的反事实基线怎么算？

- **结论**：用一个集中式 critic 估计 $Q(s,\mathbf{u})$。对 agent a，固定其他 agent 的动作，把 a 自己的动作按它的策略边缘化掉：$A^a = Q(s,\mathbf{u}) - \sum_{u'_a}\pi^a(u'_a\mid\tau^a)\,Q(s,(\mathbf{u}^{-a},u'_a))$。
- **数量级**：critic 的设计让它"in a single forward pass"就能算出这个基线，不需要额外和环境交互。
- **取舍**：critic 本身难学，因为联合动作空间很大；在 LLM agent 里，动作是文本，根本无法枚举"所有动作"。
- **出处**：[COMA](https://arxiv.org/abs/1705.08926)。

### 6.4 ★★★ Shapley 值为什么公平？为什么贵？怎么近似？

- **结论**：Shapley 值是每个 agent 在所有加入顺序下边际贡献的平均值，满足四条公理：效率、对称、零贡献者得零、可加性。
- **数量级**：精确计算要评估 $2^n$ 个联盟，n = 10 就是 1,024 次。通常用随机排列采样做蒙特卡洛估计，误差约 ∝ $1/\sqrt{\text{采样数}}$。
- **取舍**：每评估一个联盟就要跑一次（部分）任务，成本很高；估计本身的噪声也会增加梯度方差。

### 6.5 ★★ 把组内基线（GRPO 风格）用到多 agent 上，效果和局限是什么？

- **结论**：同一任务采样多次，用组均值做基线，能降低方差，但分不清组内是哪个 agent 做出了贡献。
- **数量级**：GRPO 用组均值做基线、再除以组内标准差，不需要 critic（DeepSeekMath）。
- **取舍**：它只消除了"任务难度"带来的方差，没有解决 agent 之间的信用分配；可以和 difference rewards 或反事实基线结合使用。
- **出处**：[DeepSeekMath](https://arxiv.org/abs/2402.03300)。

### 6.6 ★★ 多轮 agent 里，时间维度上的信用分配怎么做？

- **结论**：如果只有终局奖励，所有轮次共享同一个优势，轨迹越长，每一步分到的信号越稀薄。可选的办法有过程奖励、按轮次估计价值、反事实重放。
- **数量级**：可以这样量化：固定前 t 轮，只重采后面的轮次，看终局奖励的方差怎么随 t 变化，据此找出关键轮次。
- **取舍**：过程奖励模型可能被 hack；价值模型在长上下文上很难训练。

### 6.7 ★★★ 多 agent RL 里的 reward hacking 有哪些特有的形式？

- **结论**：
  - agent 之间"合谋"，例如一个 agent 帮另一个绕过检查；
  - 把责任推给别的 agent；
  - 利用评审 agent 的漏洞；
  - 单 agent 那类 hacking（比如改测试）依然存在。
- **数量级**：可以这样量化：看"奖励在涨、人工评估或隔离测试不涨"的剪刀差，以及各 agent 奖励之间的相关性有没有异常升高。
- **取舍**：评测环境要独立且只读，裁判模型要做对抗测试。代价是评测成本和延迟都会上升。

### 6.8 ★★★ 反事实信用分配的系统成本怎么算？怎么省？

- **结论**：difference rewards、Shapley 这类方法的成本，最终要折算成卡·秒。最大的省法是从共享前缀分叉：反事实只需要重跑分叉点之后的后缀，前缀的 KV 直接复用。
- **数量级**（按模块 08 演练 B 的轨迹：40K token、20 轮，DeepSeek-V3）：
  - 一条完整轨迹约 11 PFLOPs（prefill 约 5.8，decode 约 5.0），按 600 TFLOPS 约 18 卡·秒。
  - 5 个 agent 各做一次整条轨迹的反事实，要多 5 × 18 ≈ 90 卡·秒，总成本变成 6 倍。
  - 从第 10 轮分叉、复用前缀 KV 时，每次反事实约 5.9 PFLOPs（约 10 卡·秒），约为整条的 54%。后半段的上下文更长，所以省不到一半。
- **取舍**：
  - 分叉要求环境能在分叉点快照和复原，例如沙箱的文件系统状态。
  - 分叉点选得越晚越省，但只能评估后面那些动作的贡献。
  - Shapley 要 $2^n$ 个联盟，只能采样近似，每个样本都要按上面的方式付费。

## 7. 系统设计（7 题）

### 7.1 ★★★ 用 H800 集群服务 1 万个并发 coding agent，要求前缀缓存和 PD 分离。

- **结论**：这道题的核心是 KV cache 分层、按缓存路由，以及 miss 尾部的准入控制。decode 同时逼近 HBM 容量和 MLA 注意力算力；prefill 的规模和 p99 都由"整段 miss 的比例"决定。
- **数量级**：
  - 缓存命中时 prefill 约 300–550 卡（600/330 TFLOPS），全部失效时约 3,000 卡。含 MLA 升维时，整段 miss 约是命中的 9.8 倍。
  - decode：每卡约 20 条 35K 的流，一个 EP144 实例约 2,930 条，要 3–4 个实例（54–72 节点）。
  - 排队按忙时速率（600 TFLOPS）算：10% 整段 miss 时，p99 TTFT ≤ 2 s 要求 prefill 利用率不超过约 0.8。实际按 ρ ≈ 0.55 部署，给 600 TFLOPS 这个乐观假设留余量，也与 Day 6 的 330 / 600 对得上。
  - 按部署粒度汇总、prefill 再乘峰均比 1.23，约 145–235 节点。
- **取舍**：PD 之间搬整段还是只搬增量；等待工具的 KV 放 HBM、主存还是重算；本地 prefill 在大 EP 下会卡住整个实例；FP8 KV 会撞上算力墙；大 EP 带来的爆炸半径。
- **详解**：[模块 08 演练 A](../modules/08_system_design/README.md#4-演练-a用-h800-集群服务-1-万个并发-coding-agent前缀缓存--pd-分离)。

### 7.2 ★★★ 为 600B 级 MoE 设计异步 agentic RL 训练系统，包括沙箱集群和权重同步。

- **结论**：rollout 和训练分离、异步进行；每个 token 记录版本、推理侧 logprob 和 MoE 路由；用 partial rollout 或可中断生成处理长尾；权重同步的瓶颈在重排，不在带宽。
- **数量级**：
  - 每步训练 164M token。40K 序列的每 token 成本是 4K 的 2.11 倍，再加 old/ref 两次前向，512 卡上约 731 s。
  - 每张卡的显存：参数、梯度、优化器状态约 25 GB，40K 序列的激活约 35 GB，要靠上下文并行或重计算。
  - 在途轨迹约 3,100 条，需要同样数量的沙箱；decode 1 个 EP144 实例，prefill 2–4 个 EP32 实例。
  - 权重同步要落地约 7 TB，每卡的带宽下限 0.44–0.8 s；公开系统实测是秒级到 30 s。
- **取舍**：η 的大小、丢弃带来的偏差、GRPO 的组级长尾、训推不一致、双缓冲占用的显存。
- **详解**：[模块 08 演练 B](../modules/08_system_design/README.md#5-演练-b为-600b-级-moe-设计异步-agentic-rl-训练系统含沙箱集群与权重同步)。

### 7.3 ★★★ 设计一个多租户 prompt caching API，覆盖缓存分层、计费和隔离。

- **结论**：
  - 缓存键是块级哈希链，里面要带租户 salt、模型版本和 tokenizer 版本。
  - 分三层：HBM、主存、SSD。
  - 淘汰分 = 复用概率 × 重算代价 / 字节数。
  - 计费按实际匹配到的块数。
  - salt 只堵住跨租户复用这一类侧信道，其余的要靠按租户排队、资源分区和审计。
- **数量级**：
  - 先做合理性检查：5,000 请求/s × 20K 是 DeepSeek Day 6 全站平均输入的 14 倍，光未命中部分就要约 8,000 张 H800。
  - DeepSeek-V3 一个 14K 前缀是 0.98 GB，重算要 2.5 卡·秒，读回约 20 ms。70% 的 token 命中只省下约 60% 的 FLOPs。
  - 如果所有命中都从远端读，需要 4.9 TB/s 带宽，所以热前缀必须常驻 HBM。
- **取舍**：隔离会失去去重；TTL 越长存储成本越高；块越大计费粒度越粗。
- **详解**：[模块 08 演练 C](../modules/08_system_design/README.md#6-演练-c多租户-prompt-caching-api缓存分层计费隔离)。

### 7.4 ★★★ 为多 agent 编排平台定端到端延迟 SLO，要考虑关键路径、fan-out 和超时。

- **结论**：关键路径上最贵的是输出 token；fan-out 会放大尾延迟，独立时是 $p^k$。解法包括 quorum + deadline、对冲请求、deadline 逐级下传、取消传播，最后要用端到端分位数验收。
- **数量级**：
  - 独立时，5 个子 agent 都在各自 p90 以内的概率只有 59%，改成"5 取 4"后是 92%；两两相关系数 0.5 时是 71% 和 88%，收益从 33 个百分点降到 17 个。
  - 每 1,000 个输出 token 约 25 s（TPOT 25 ms）。
  - 蒙特卡洛核算：只砍规划输出、加 quorum，p90 仍有 106 s；再砍子 agent 和汇总的输出，p90 降到 76 s。
- **取舍**：答案完整性和延迟之间的取舍；对冲带来的额外负载，发到没有前缀的副本等于整段 miss；重试预算。
- **详解**：[模块 08 演练 D](../modules/08_system_design/README.md#7-演练-d多-agent-编排平台的端到端延迟-slo关键路径fan-out超时)。

### 7.5 ★★★ 线上吞吐突然下降 30%，怎么排查？

- **结论**：先确认指标是真的，再用"GPU 忙碌度有没有变"把问题一分为二：变闲了，查需求侧和上游；一样忙但 token 变少，说明每个 token 变贵了。然后逐个分支估算它最多能解释多少下降。
- **数量级**：
  - 命中率从 80% 降到 60%，吞吐可以掉 23–29%。
  - 一张卡慢到 70%，整个 EP 组就慢约 30%。
  - MTP 接受率从 0.85 降到 0.6，只能解释 13.5%。
- **取舍**：`nvidia-smi` 的利用率只统计"有 kernel 在跑"的时间，看不出真实效率，要看每步耗时和 MFU/MBU。
- **详解**：[模块 08 演练 E](../modules/08_system_design/README.md#8-演练-e线上吞吐突然下降-30-的排查)。

### 7.6 ★★ 用 Little 定律估算 RL 沙箱集群的规模。

- **结论**：并发沙箱数 = 轨迹完成速率 × 单条轨迹的生命周期。
- **数量级**：按演练 B 的假设，训练步长约 731 s，完成速率 4,096 / 731 ≈ 5.61 条/s；生命周期 557 s（decode 500 s + 19 次工具 × 3 s）。并发沙箱 ≈ 5.61 × 557 ≈ 3,100 个，另加 10–20% 的预热池。
- **取舍**：预热池越大，启动越快，但浪费也越多；有状态的沙箱没法在轨迹中途迁移。训练步长估小了，沙箱数就会估大，所以要先把训练端算对。

### 7.7 ★★ 设计一个 KV cache 存储服务（Mooncake / 3FS 风格）。

- **结论**：做成以块哈希为键的分布式对象存储，分 HBM / 主存 / SSD 三层，用 RDMA 读写，并提供全局索引给路由使用。
- **数量级**：
  - 3FS 在 180 个节点上的聚合读吞吐约 6.6 TiB/s。
  - DeepSeek 线上 56.3% 的输入 token 命中落盘缓存。
  - Mooncake 报告，在真实负载下让 Kimi 多处理了 75% 的请求。
- **取舍**：写穿还是回写；存副本还是丢了重算；块大小和元数据量之间的平衡。
- **出处**：[Mooncake](https://arxiv.org/abs/2407.00079)、[3FS](https://github.com/deepseek-ai/3FS)。

## 8. 行为与项目（6 题）

### 8.1 ★ "讲讲你的 co-design 项目。"

- **结论**：按五段式来讲：动机 → 为什么难 → 切入点 → 怎么验证 → 预期取舍。一开头就交代保密约束，全程用 studying / exploring 这类措辞。
- **可量化的点**：动机里放一个能心算的数字，例如"整段重算约是命中的 10 倍"；把证据分成三级：论文结论、模拟结果、尚未验证的假设。
- **取舍**：讲得具体才可信，但不能泄露合作方细节。
- **详解**：[模块 08 §9](../modules/08_system_design/README.md#9-如何在面试里讲你的-harnessinference-co-design-项目)。

### 8.2 ★★ "讲一次你发现自己实验结论错了的经历。"

- **结论**：用 STAR 结构讲，重点放在三件事：你是怎么发现的，怎么修正的，之后加了哪条检查。
- **可量化的点**：说出触发你怀疑的信号，例如实测和第一性原理估算差了一个数量级、换一个随机种子结论就反转；再说清修正之后的差距。
- **取舍**：别编一个完美故事。面试官想看的是你的自我校验习惯。

### 8.3 ★★ "研究的新颖性和工程落地冲突时，你怎么取舍？"

- **结论**：先定义成功指标，再给探索设时间盒，用最小的可验证实验判断是否继续。
- **可量化的点**：例如"两天内用模拟器判断这个方向的上界是否超过 10%，达不到就放弃"。上界可以用离线最优或理想化假设来算。
- **取舍**：过早工程化会浪费探索时间；一直不落地，又没人能用你的结果。

### 8.4 ★★ "你一作论文最大的贡献是什么？如果重做会怎么改？"

- **结论**：一句话讲清贡献，加一个关键数字，再主动说一个局限。
- **可量化的点**：用论文里真实的结果，别四舍五入成更好看的数；说明它在什么设置下成立。
- **取舍**：主动承认局限会显得成熟，但要紧接着说你打算怎么补上。

### 8.5 ★ "为什么想做 infra，为什么是我们团队？"

- **结论**：把理由和你的研究问题连起来：agent 负载改变了推理系统和训练系统原有的假设（多轮、工具等待、长尾），而这正是 infra 团队要解决的问题。
- **可量化的点**：举一个你能算清楚的例子，例如工具等待期间 KV 的去留：留在 HBM 挤占容量，下放主存读回约 50 ms，重算约 10 倍，所以关键是按工具时长决定下放和预取的时机。
- **取舍**：要表现出热情，但别对这个团队的内部系统做任何猜测。

### 8.6 ★★★ "你和资深工程师在设计上意见不一致，怎么办？"

- **结论**：先对齐假设，因为分歧常常来自负载假设不同；再用估算和小实验说话；准备好接受自己被证伪。
- **可量化的点**：把分歧变成一个可测的问题，例如"整段 miss 的比例到底是 2% 还是 20%"，然后回放 trace 来判断。
- **取舍**：推进速度和把事情做对之间要平衡。不关键的分歧，先按对方的方案走，同时记录下来。
