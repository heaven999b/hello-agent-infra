**中文** · [English](glossary.en.md)

# 术语表

> 全项目用到的术语：中英对照、一句话解释、首次出现的模块（按模块编号 00 → 09 查找，最后是综合实战）。讲义正文里术语保留英文原词，这里给出中文说法。

## 硬件与性能模型

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **HBM（高带宽显存）** | HBM (high-bandwidth memory) | 堆叠在 GPU 旁的显存，容量几十到上百 GB、带宽数 TB/s；decode 的速度主要由它的带宽决定。 | [M00](../modules/00_map/README.md) |
| **NVLink** | NVLink | NVIDIA GPU 之间的高速互联；每卡双向合计 H100 900 GB/s、H800 削到 400 GB/s、B200 1.8 TB/s，估搬运时间用单向（一半）。 | [M00](../modules/00_map/README.md) |
| **RDMA（远程直接内存访问）** | RDMA (remote direct memory access) | 网卡直接读写远端主机内存或 GPU 显存（配合 GPUDirect RDMA），不经过远端 CPU、也不必先拷到主机内存；跨机搬 KV、专家并行通信都靠它。 | [M00](../modules/00_map/README.md) |
| **InfiniBand（IB）** | InfiniBand (IB) | 数据中心常用的 RDMA 网络；NDR 每端口 400 Gb/s ≈ 50 GB/s。 | [M00](../modules/00_map/README.md) |
| **RoCE** | RoCE (RDMA over Converged Ethernet) | 在以太网上跑 RDMA，是 InfiniBand 之外的另一种选择。 | [M00](../modules/00_map/README.md) |
| **PCIe** | PCIe | GPU 与 CPU、网卡、SSD 之间的总线；Gen5 x16 单向约 64 GB/s，KV 卸载到 CPU 内存走这里。 | [M00](../modules/00_map/README.md) |
| **FLOPs** | FLOPs | 浮点运算次数（乘加算 2 次）；每秒的浮点运算次数写作 FLOP/s 或 TFLOPS。 | [M00](../modules/00_map/README.md) |
| **dense / 稀疏算力** | Dense vs sparse FLOPs | 厂商规格页常把 2:4 结构化稀疏的算力当标称值，它是 dense 算力的 2 倍；估算推理时用 dense。 | [M00](../modules/00_map/README.md) |
| **算术强度** | Arithmetic intensity | FLOPs ÷ 读写显存的字节数；dense decode 的线性层约等于 batch，MoE 专家约为 B·k/E，decode 注意力约为 query 头数 ÷ KV 头数（BF16 KV、序列不共享前缀时）。 | [M00](../modules/00_map/README.md) |
| **roofline（屋顶线模型）** | Roofline model | 可达算力 = min(峰值算力, 带宽 × 算术强度)，一眼看出瓶颈在算力还是带宽。 | [M01](../modules/01_inference_basics/README.md) |
| **ridge point（屋脊点）** | Ridge point | 峰值算力 ÷ 显存带宽；H100 BF16 约 295 FLOPs/Byte，强度低于它就是 memory-bound。 | [M01](../modules/01_inference_basics/README.md) |
| **compute-bound / memory-bound** | Compute-bound / memory-bound | 耗时由算力决定 / 由显存带宽决定，看 FLOPs ÷ 字节是否超过 ridge。常见 serving 区间里 prefill 多为前者、decode 多为后者；但只算几百个新 token 的增量 prefill 可能 memory-bound，MLA 或共享前缀的 decode 注意力可以接近 compute-bound。 | [M00](../modules/00_map/README.md) |
| **MFU** | MFU (model FLOPs utilization) | 实际达到的有效算力占峰值的比例；prefill 常按 40–60% 估算。 | [M00](../modules/00_map/README.md) |
| **MBU** | MBU (model bandwidth utilization) | 实际达到的显存带宽占峰值的比例；decode 的效率指标。 | [M01](../modules/01_inference_basics/README.md) |
| **BF16 / FP8** | BF16 / FP8 | 每元素 2 字节 / 1 字节的浮点格式。权重或 KV 用 FP8 量化且 scale 等元数据开销可忽略时，字节约减半；Hopper / Blackwell 的 FP8 Tensor Core 峰值约为 BF16 的 2 倍，但只针对矩阵乘，端到端收益受带宽、反量化和精度约束。 | [M00](../modules/00_map/README.md) |

## 模型结构

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **KV cache** | KV cache | 把已处理 token 每层的 key / value 存下来，decode 时不必重算；大小 = token 数 × 每 token KV 字节。 | [M00](../modules/00_map/README.md) |
| **MHA（多头注意力）** | MHA (multi-head attention) | 每个 query 头都有自己的 K / V 头；KV 最大。 | [M01](../modules/01_inference_basics/README.md) |
| **MQA（多查询注意力）** | MQA (multi-query attention) | 所有 query 头共享一组 K / V；KV 最小，但可能损失质量。 | [M01](../modules/01_inference_basics/README.md) |
| **GQA（分组查询注意力）** | GQA (grouped-query attention) | 若干 query 头共享一组 K / V，介于 MHA 和 MQA 之间；Llama-3-70B 为 64 个 query 头、8 个 KV 头。 | [M00](../modules/00_map/README.md) |
| **MLA（多头潜在注意力）** | MLA (multi-head latent attention) | DeepSeek 的注意力：只缓存一个压缩后的潜在向量（512 维）加解耦的 RoPE key（64 维），所有头共享。 | [M00](../modules/00_map/README.md) |
| **权重吸收** | Weight absorption | MLA decode 时把上投影矩阵并进 query / 输出投影，直接在潜在向量上做注意力：读得少、算得多。 | [M01](../modules/01_inference_basics/README.md) |
| **MoE（混合专家）** | MoE (mixture of experts) | 每个 token 只激活少数专家 FFN；DeepSeek-V3 总参数 671B、每 token 激活 37B。 | [M00](../modules/00_map/README.md) |
| **激活参数** | Active parameters | 每个 token 实际参与计算的参数量，决定 FLOPs；显存要放下的是总参数量。 | [M01](../modules/01_inference_basics/README.md) |
| **MTP（多 token 预测）** | MTP (multi-token prediction) | 训练时额外预测后续多个 token 的模块；推理时可当投机解码的草稿。 | [M01](../modules/01_inference_basics/README.md) |
| **无辅助损失负载均衡** | Auxiliary-loss-free load balancing | DeepSeek-V3 给每个专家的路由分数加一个偏置，只用于选 top-K，按每步的负载动态调大或调小，主要靠它做均衡；另外只保留一个权重极小（α = 0.0001）的序列级 balance loss，防止单条序列内极端不均。 | [M03](../modules/03_deepseek_stack/README.md) |

## 推理引擎

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **prefill（预填充）** | Prefill | 一次性处理整段输入 prompt、写出它的 KV；新 token 多时是 compute-bound，只算几百个新 token 时可能 memory-bound；决定 TTFT。 | [M00](../modules/00_map/README.md) |
| **decode（解码）** | Decode | 逐个生成输出 token，每步读权重和每条序列的 KV；通常 memory-bound，决定 TPOT。 | [M00](../modules/00_map/README.md) |
| **continuous batching（连续批处理）** | Continuous batching | 按迭代而不是按请求调度：每一步都可以有请求加入或离开 batch（Orca 的 iteration-level scheduling）。 | [M00](../modules/00_map/README.md) |
| **PagedAttention / 分页 KV cache** | PagedAttention / paged KV cache | 像虚拟内存一样把 KV 切成固定大小的 block，用 block table 映射，把浪费压到约一个尾块（vLLM）。block 大小是折中：越小碎片越少、前缀共享粒度越细，但查表和间接寻址开销越大。 | [M00](../modules/00_map/README.md) |
| **抢占** | Preemption | 显存不够时暂停一些请求、释放它们的 KV；恢复时要么重算（recompute），要么从 CPU 换回（swap）。 | [M00](../modules/00_map/README.md) |
| **chunked prefill（分块预填充）** | Chunked prefill | 把长 prompt 的 prefill 切成小块，和 decode 混在同一步里，避免长 prefill 卡住所有 decode（Sarathi-Serve）。 | [M00](../modules/00_map/README.md) |
| **投机解码** | Speculative decoding | 用便宜的草稿模型或 MTP 先猜若干 token，大模型一步验证，把 memory-bound 的 decode 变得更「值」。 | [M01](../modules/01_inference_basics/README.md) |
| **量化** | Quantization | 用更少的比特存权重、激活或 KV（如 FP8、INT4），省字节也可能省算力，代价是精度风险。 | [M01](../modules/01_inference_basics/README.md) |
| **PD 分离** | PD disaggregation (prefill-decode disaggregation) | prefill 和 decode 放在不同的 GPU 池上，消除互相干扰，代价是要跨机传 KV（DistServe、Splitwise、Mooncake）。 | [M00](../modules/00_map/README.md) |
| **CUDA Graph** | CUDA Graph | 把一串 GPU kernel 录下来整体重放，省掉逐个启动的开销；要求形状固定，和每步都变的 batch 有冲突。 | [M01](../modules/01_inference_basics/README.md) |
| **FlashMLA** | FlashMLA | DeepSeek 开源的 Hopper 上 MLA decode kernel；H800 上报告 memory-bound 配置最高 3000 GB/s、compute-bound 配置 660 TFLOPS。 | [M01](../modules/01_inference_basics/README.md) |
| **vLLM / SGLang / TensorRT-LLM** | vLLM / SGLang / TensorRT-LLM | 三个主流推理引擎：vLLM 起源于 PagedAttention，SGLang 起源于 RadixAttention，TensorRT-LLM 是 NVIDIA 的引擎。 | [M01](../modules/01_inference_basics/README.md) |

## 并行与通信

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **TP（张量并行）** | TP (tensor parallelism) | 把一层的矩阵切到多卡上，每层都要 all-reduce；适合节点内高带宽 NVLink。 | [M01](../modules/01_inference_basics/README.md) |
| **PP（流水线并行）** | PP (pipeline parallelism) | 把不同层放到不同卡上，按微批流水执行；通信少，但有流水线气泡。 | [M01](../modules/01_inference_basics/README.md) |
| **DP（数据并行）** | DP (data parallelism) | 每份副本处理不同的请求。推理里还有 DP attention：只把注意力按请求分到不同卡，专家仍走 EP。对 MLA（潜在向量不能按头切）或 TP 超过 KV 头数的模型，TP 会让 KV 在多张卡上重复，DP attention 避免了这一点；代价是注意力等非 routed 权重每卡一份。 | [M01](../modules/01_inference_basics/README.md) |
| **EP（专家并行）** | EP (expert parallelism) | 把 MoE 的专家分到不同卡上，token 通过 all-to-all 发给对应专家；DeepSeek decode 用到 EP144 及以上。 | [M00](../modules/00_map/README.md) |
| **all-to-all / all-reduce** | All-to-all / all-reduce | 集合通信原语：all-to-all 让每张卡给每张卡发不同数据（EP 分发），all-reduce 把各卡结果求和再广播（TP）。 | [M00](../modules/00_map/README.md) |
| **node-limited routing（限节点路由）** | Node-limited routing | DeepSeek-V3 限制每个 token 最多发往 4 个节点，因为跨节点 IB 带宽只有 NVLink 的约 1/3。 | [M00](../modules/00_map/README.md) |
| **DeepEP** | DeepEP | DeepSeek 开源的专家并行通信库：MoE dispatch / combine 的 all-to-all kernel，分高吞吐（训练、prefill）和低延迟（decode）两类，支持 FP8。 | [M03](../modules/03_deepseek_stack/README.md) |
| **DualPipe** | DualPipe | DeepSeek-V3 的双向流水线并行：从流水线两端同时送入微批，把 all-to-all 和 PP 通信藏到计算后面，减少气泡。 | [M03](../modules/03_deepseek_stack/README.md) |
| **EPLB（专家并行负载均衡器）** | EPLB (expert-parallel load balancer) | 复制高负载专家（冗余专家）并启发式地摆放到各卡上，让 EP 下每张卡的负载接近。 | [M02](../modules/02_inference_engines/README.md) |

## 缓存与路由

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **前缀缓存** | Prefix cache / prefix caching | 复用「从第一个 token 起完全相同」的前缀已算好的 KV；前缀中间改一个字，之后全部失效。 | [M00](../modules/00_map/README.md) |
| **prompt caching** | Prompt caching | API 层面的前缀缓存计费：命中部分按折扣价计费（如 Anthropic 缓存读取 0.1× 输入价）。 | [M00](../modules/00_map/README.md) |
| **block hash（块哈希）** | Block hash | 把 token 序列按 block 切分，每块的 hash 链式依赖前一块，用来判断前缀是否相同（vLLM）。 | [M02](../modules/02_inference_engines/README.md) |
| **radix tree / RadixAttention** | Radix tree / RadixAttention | 按前缀组织 KV 的树，多个请求共享公共前缀、分叉各自保存（SGLang）。 | [M00](../modules/00_map/README.md) |
| **cascade attention（共享前缀注意力）** | Cascade attention (shared-prefix attention) | 把注意力拆成共享前缀和各自后缀两段，前缀部分把多条序列的 query 拼起来、对同一份 KV 只读一次，强度乘上序列数（Hydragen、FlashInfer）。 | [M01](../modules/01_inference_basics/README.md) |
| **命中率** | Hit rate | 命中缓存的输入 token 占比；DeepSeek 线上 24 小时约 56.3%。 | [M00](../modules/00_map/README.md) |
| **淘汰策略（LRU / TTL / session-aware）** | Eviction policy (LRU / TTL / session-aware) | 缓存满了先扔谁：最久未用、按生存时间、或按会话是否还会回来。 | [M00](../modules/00_map/README.md) |
| **TTL（生存时间）** | TTL (time to live) | 缓存条目在无访问时保留多久；工具调用期间给 KV 设 TTL 是「留还是丢」的折中。 | [M00](../modules/00_map/README.md) |
| **pin（钉住）** | Pin | 让某段 KV 在一段时间内不可被淘汰。 | [M00](../modules/00_map/README.md) |
| **KV 卸载 / 分层存储** | KV offload / tiered storage | 把暂时不用的 KV 从 HBM 挪到 CPU DRAM 或 SSD，要用时再搬回（Mooncake、3FS）。 | [M00](../modules/00_map/README.md) |
| **cache-affinity 路由** | Cache-affinity routing | 把共享前缀的请求送到已经缓存了这段前缀的实例上；和负载均衡相互冲突。 | [M00](../modules/00_map/README.md) |
| **负载不均衡** | Load imbalance | 各实例负载差异；亲和路由把热门前缀堆到少数实例上时会变严重。 | [M00](../modules/00_map/README.md) |
| **cache_salt** | cache_salt | vLLM 的按请求缓存盐：混进第一个 block 的 hash，只有同 salt 的请求能共享缓存，用于多租户隔离、防计时侧信道。 | [M02](../modules/02_inference_engines/README.md) |
| **Mooncake** | Mooncake | Kimi 的以 KVCache 为中心的 PD 分离架构，用集群里闲置的 CPU、DRAM、SSD 组成分布式 KV 缓存。 | [M02](../modules/02_inference_engines/README.md) |
| **3FS** | 3FS (Fire-Flyer File System) | DeepSeek 开源的基于 SSD + RDMA 的分布式文件系统，也用来存放可复用的 KV cache。 | [M03](../modules/03_deepseek_stack/README.md) |

## 指标与排队

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **TTFT（首 token 延迟）** | TTFT (time to first token) | 从提交请求到收到第一个输出 token；= 排队 + 未命中部分的 prefill。 | [M00](../modules/00_map/README.md) |
| **TPOT（每输出 token 时间）** | TPOT (time per output token) | (端到端延迟 − TTFT) ÷ (输出 token 数 − 1)，每个请求一个值。 | [M00](../modules/00_map/README.md) |
| **ITL（token 间延迟）** | ITL (inter-token latency) | 相邻两个输出 token 的间隔，每个间隔一个样本；平均 ITL 按 token 加权。 | [M01](../modules/01_inference_basics/README.md) |
| **吞吐** | Throughput | 系统每秒处理或产出的 token 数；不关心单个请求等了多久。 | [M00](../modules/00_map/README.md) |
| **goodput（有效吞吐）** | Goodput | DistServe 的定义：在达到 SLO 达标率目标（如 90% 的请求同时满足 TTFT 和 TPOT）时，每张 GPU 能承受的最大请求速率；Anyscale 文档另把它定义为满足全部 SLO 的请求占比。 | [M01](../modules/01_inference_basics/README.md) |
| **SLO（服务等级目标）** | SLO (service level objective) | 对延迟分位数的承诺，如「TTFT p99 < 2 s 且 TPOT p99 < 50 ms」。 | [M01](../modules/01_inference_basics/README.md) |
| **p99 / 尾延迟** | p99 / tail latency | 第 99 百分位的延迟；agent 串行多轮时，尾部会被放大。 | [M01](../modules/01_inference_basics/README.md) |
| **JCT（任务完成时间）** | JCT (job completion time) | 一个多轮 agent 任务从开始到结束的总时间；比单次请求的 TTFT / TPOT 更贴近用户体验。 | [M05](../modules/05_cache_prefill/README.md) |
| **Little's law（利特尔法则）** | Little's law | 系统内平均数量 = 到达率 × 平均停留时间（L = λW）；用来从吞吐和延迟反推并发与显存。 | [M01](../modules/01_inference_basics/README.md) |

## Agent harness

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **agent harness** | Agent harness | 包在模型外面的那层程序：拼上下文、调工具、管记忆、编排多个 agent，决定每轮发给模型什么。 | [M00](../modules/00_map/README.md) |
| **上下文组装** | Context assembly | 把 system prompt、工具定义、历史、检索结果按顺序拼成一次请求；顺序决定前缀能否复用。 | [M00](../modules/00_map/README.md) |
| **上下文压缩（compaction）** | Compaction | 上下文太长时把历史总结或裁剪；会让压缩点之后的前缀缓存失效。 | [M00](../modules/00_map/README.md) |
| **记忆注入** | Memory injection | 把长期记忆或检索结果插进上下文；插在前面会破坏前缀复用。 | [M00](../modules/00_map/README.md) |
| **工具调用** | Tool call | 模型输出结构化的调用请求，由 harness 执行后把结果追加回上下文；期间 KV 处于等待状态。 | [M00](../modules/00_map/README.md) |
| **append-only 上下文** | Append-only context | 只在末尾追加、从不修改前面的内容，让每一轮都能命中上一轮的前缀。 | [M00](../modules/00_map/README.md) |
| **多 agent 编排** | Multi-agent orchestration | lead agent 派出多个 subagent 并行工作再汇总（fan-out / fan-in）；token 量成倍增长。 | [M00](../modules/00_map/README.md) |
| **会话 / 轮** | Session / turn | 一个 agent 任务的完整多轮交互 / 其中一次「模型输出 + 工具结果」。 | [M00](../modules/00_map/README.md) |
| **logit 掩码** | Logit masking | 不从上下文里删掉工具定义，而是在解码时屏蔽不该用的工具，保持前缀稳定（Manus 的做法）。 | [M01](../modules/01_inference_basics/README.md) |

## 训练系统与 RL

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **agentic RL** | Agentic RL | 用强化学习训练 agent：轨迹是多轮、带工具调用的会话，奖励来自任务结果。 | [M00](../modules/00_map/README.md) |
| **rollout（轨迹采样）** | Rollout | 用当前策略跑出一批轨迹用于训练；在 agentic RL 里它本质上是一大批推理请求。 | [M00](../modules/00_map/README.md) |
| **长尾** | Long tail | 少数轨迹或请求远长于平均值；在重尾分布（如 Pareto）下，同步等待最慢者的代价很高。 | [M00](../modules/00_map/README.md) |
| **同步 / 异步 RL** | Synchronous / asynchronous RL | 同步：等一批轨迹全部结束再训练；异步：生成和训练并行，用稍旧的策略数据换利用率（AReaL）。 | [M00](../modules/00_map/README.md) |
| **partial rollout（部分展开）** | Partial rollout | 每轮只生成固定时间或长度，没完成的轨迹留到下一轮续写，削掉长尾。 | [M00](../modules/00_map/README.md) |
| **staleness（策略滞后）** | Staleness | 生成数据的策略版本落后于正在训练的版本多少步；异步 RL 要给它设上限。 | [M00](../modules/00_map/README.md) |
| **on-policy / off-policy** | On-policy / off-policy | 训练数据是否来自当前策略；staleness > 0 的数据是轻度 off-policy。 | [M00](../modules/00_map/README.md) |
| **重要性采样** | Importance sampling | 用新旧策略概率之比给旧数据重新加权，修正 off-policy 偏差（PPO 的 ratio 与 clip 就是这个思路）。 | [M00](../modules/00_map/README.md) |
| **PPO** | PPO (proximal policy optimization) | 用新旧策略概率比加裁剪（clip）限制每次更新幅度的策略梯度算法。 | [M03](../modules/03_deepseek_stack/README.md) |
| **GRPO** | GRPO (group relative policy optimization) | PPO 的变体：对同一 prompt 采样一组回答，用组内相对奖励作优势，省掉价值网络（DeepSeekMath）。 | [M01](../modules/01_inference_basics/README.md) |
| **decoupled PPO** | Decoupled PPO | AReaL 的目标函数：把采样用的行为策略和用作信任域中心的近端策略分开，让旧数据也能稳定训练。 | [M06](../modules/06_long_tail_async/README.md) |
| **权重同步** | Weight sync | 训练更新后把新权重推送给推理 / rollout 引擎；大模型要搬几百 GB。 | [M00](../modules/00_map/README.md) |

## 多 agent 信用分配

| 中文 | English | 一句话解释 | 首次出现 |
|---|---|---|---|
| **信用分配** | Credit assignment | 团队只拿到一个总奖励时，把功劳分给每个 agent 的每个动作。 | [M00](../modules/00_map/README.md) |
| **共享奖励** | Shared reward | 每个 agent 都拿团队总奖励；无偏但方差大，搭便车的 agent 也被奖励。 | [M00](../modules/00_map/README.md) |
| **组内基线（group baseline）** | Group baseline | 用同一 prompt 的一组采样的平均奖励作基线（GRPO；RLOO 则去掉样本自己），降低方差。 | [M06](../modules/06_long_tail_async/README.md) |
| **difference reward（差分奖励）** | Difference reward | 团队奖励减去「把这个 agent 换成默认动作」后的团队奖励，衡量它的边际贡献。 | [M00](../modules/00_map/README.md) |
| **Shapley 值** | Shapley value | 对所有加入顺序平均边际贡献的公平分配；精确计算是指数级的。 | [M00](../modules/00_map/README.md) |
| **反事实基线** | Counterfactual baseline | 固定其他 agent 的动作，对这个 agent 的动作求期望作基线（COMA 的思路）。 | [M07](../modules/07_multi_agent_credit/README.md) |
| **梯度方差** | Gradient variance | 策略梯度估计的方差；信用分配越准，方差越小，学得越快。 | [M00](../modules/00_map/README.md) |

## 易混概念对照

| 概念对 | 区别 | 为什么要分清 |
|---|---|---|
| **TPOT vs ITL** | TPOT 每个请求一个值：(端到端 − TTFT) ÷ (输出 token 数 − 1)，平均时每个请求权重相同；ITL 每个 token 间隔一个样本，平均时按 token 加权。NVIDIA 的文章把两者当同义词。 | 长请求多时两者不同；面试先问清口径。 |
| **token 命中率 vs 请求命中率** | token 命中率 = 命中缓存的输入 token ÷ 全部输入 token（DeepSeek 的 56.3% 就是这个）；请求命中率 = 有命中（或整段命中）的请求占比。 | 省多少算力和钱看 token 命中率；TTFT 的分布看每个请求命中了多少。 |
| **MFU vs MBU** | MFU = 实际有效算力 ÷ 峰值算力；MBU = 实际有效带宽 ÷ 峰值带宽。 | prefill 看 MFU，decode 看 MBU；MFU 高不代表 decode 快。 |
| **prefix cache vs semantic cache** | prefix cache 复用「完全相同的前缀」已算好的 KV，模型照样生成新输出，结果和不缓存一样；semantic cache 对「意思相近」的请求直接返回以前的回答，跳过模型。 | 前者只省 prefill、不改变语义；后者省得多，但可能答错或答旧。 |
| **DP attention vs 普通 DP** | 普通 DP 是整个模型的多份副本，各自服务不同请求；DP attention 只把注意力按请求分到不同卡，专家仍由整个 EP 组共享。 | DP attention 避免 MLA 的 KV 在 TP 组内重复，代价是非 routed 权重每卡一份。 |
| **goodput（DistServe）vs goodput（Anyscale）** | DistServe：给定达标率（如 90%）下每 GPU 能承受的最大请求速率，是速率；Anyscale：满足全部 SLO 的请求占比，是百分比。 | 说 goodput 先说单位和达标率。 |
| **双向带宽 vs 单向带宽** | NVLink、PCIe 规格页上的数字多是两个方向相加（H100 NVLink 900 GB/s、PCIe Gen5 x16 128 GB/s）；一次搬运只用一个方向（450、64 GB/s）。 | 拿双向数字估搬运时间会快估 2 倍。 |
