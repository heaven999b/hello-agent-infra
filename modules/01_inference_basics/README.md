**中文** · [English](README.en.md)

# 模块 01：推理第一性原理——FLOPs、字节与 roofline

> 🕐 建议时长：2 小时 ｜ 🎯 学完你能：给定模型和 GPU，一分钟内心算出每 token 的 KV 大小、最大并发、decode 是 memory-bound 还是 compute-bound、TTFT 的量级；知道这些心算在 MoE、共享前缀、增量 prefill 和多卡通信下分别在哪里失效；说清 TTFT / TPOT / ITL / goodput 的区别，并用 Little's law 反推需要多少显存 ｜ 🧪 实验：[`lab.py`](lab.py) ｜ 📖 必读一手资料：[All About Transformer Inference（How To Scale Your Model）](https://jax-ml.github.io/scaling-book/inference/)、[Efficiently Scaling Transformer Inference（arXiv:2211.05102）](https://arxiv.org/abs/2211.05102)

## 0. 一句话

**LLM 推理是两种完全不同的负载拼在一起**：prefill（预填充，一次性处理整段输入）一次喂进成千上万个 token，通常是 compute-bound（受算力限制）的大矩阵乘；decode（解码，逐个生成输出 token）每步每条序列只产出 1 个 token，却要把权重和这条序列的全部 KV 从 HBM（GPU 上的高带宽显存）读一遍，通常是 memory-bound（受显存带宽限制）的。判断标准只有一个：这次计算的「FLOPs ÷ 字节」和硬件的「峰值算力 ÷ 带宽」谁大。后面所有的系统设计——batching、KV cache 管理、PD 分离、MLA、大规模 EP——都是在围绕这个比值做取舍。

## 1. 大厂为什么在意

DeepSeek 公开的线上推理数据（[推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)）里有几个数字值得背下来：

- 同样是 8 × H800 的节点，**prefill 约 73.7k token/s（含缓存命中），decode 约 14.8k token/s**；
- 24 小时内输入 608B token，其中 **56.3%** 命中磁盘上的 KV cache；
- 平均每个输出 token 对应的 KV cache 长度是 **4,989** 个 token，每个用户平均看到的输出速度是 **20–22 token/s**。

先别急着说「prefill 比 decode 快 5 倍」。73.7k 里含命中的 token，这部分根本没做 prefill 计算。按全局命中率粗略扣掉，每个节点真正算出来的 prefill 约 73.7k × (1 − 0.563) ≈ **32k token/s，约为 decode 的 2.2 倍**。而且两者不是同一种部署：prefill 单元是 4 个节点（路由专家 EP32），decode 单元是 18 个节点（EP144）；DeepSeek 也没说明 14.8k 是否包含 MTP（多 token 预测）带来的加速，而 DeepSeek 的 [ISCA 论文（arXiv:2505.09343）](https://arxiv.org/abs/2505.09343)报告 MTP 让生成 TPS 提升 1.8 倍。**读别人的吞吐数字，第一步是问清口径。**

再用 Little's law（第 8 节）读一遍：每节点每秒出 14.8k token，每个用户每秒拿到约 21 个，那么每个 decode 节点上同时约有 14.8k ÷ 21 ≈ **700 条序列**在跑。为什么要这么多条？每条 20 token/s 已经够用户看，为什么不让每条更快？本模块用两个量——**FLOPs（浮点运算次数）和字节**——回答这些问题。放到 agent 场景里，问题更尖锐：agent 每轮输入几万 token、输出几百 token（[Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) 报告输入:输出约 100:1），TTFT（首 token 延迟）决定每一轮等多久；而能同时服务多少个 agent，取决于显存里放得下多少 KV。

## 2. 一个 token 要多少 FLOPs、读多少字节

### 2.1 FLOPs

- **线性层**：每个权重对每个 token 做一次乘加，也就是 2 FLOPs。所以前向每 token 约 **2N FLOPs**，N 是参与矩阵乘的参数量：dense 模型是总参数量，MoE 模型是**激活参数量**（DeepSeek-V3 用 37B，不能用 671B）。输入端的 embedding 是查表，不做矩阵乘，不算在 N 里；输出端的 lm_head 是真正的矩阵乘，decode 每个 token 都要算。Llama-3-8B 的词表有 128,256 × 4,096 ≈ 5.3 亿参数，所以 decode 时 N ≈ 7.5B 而不是 8.03B（codesign_sim 也按这个口径算）。这就是 [Kaplan 等人（arXiv:2001.08361）](https://arxiv.org/abs/2001.08361)的 C_forward ≈ 2N。
- **注意力**：新 token 的 query 要和上下文里每个 key 点积（QKᵀ），再对每个 value 加权求和（AV）。每层每 token 约 2·ctx·d_attn + 2·ctx·d_attn = **4·ctx·d_attn**，其中 d_attn = n_heads × head_dim，ctx 是它能看到的上下文长度。
- **prefill 的注意力**：一次处理 n 个 token，因果掩码下第 i 个 token 只看前 i 个，平均 ctx ≈ n/2，每层合计约 **2·n²·d_attn**。

注意力什么时候不能忽略？以 Llama-3-70B（N ≈ 70.6B，80 层，d_attn = 64 × 128 = 8192）为例，decode 时每 token 的注意力 FLOPs 和线性层之比是 4·ctx·8192·80 / (2 × 70.6e9)：ctx = 8k 时约 15%，ctx = 64k 时约 122%。**长上下文时注意力就是大头。**

### 2.2 字节

- **权重**：每参数字节数（BF16 为 2，FP8 为 1）× 这一步**被用到**的参数量。dense 模型每一步都要把权重从 HBM 读一遍（输入 embedding 表只读被查到的那几行），**不论 batch 里有几条序列，权重只读一次**。MoE 模型每步只读被选中的专家，batch 越大，被选中的专家越多（第 3.4 节）。
- **KV cache**：用标准的 decode kernel 时，每条序列每一步都要把自己的全部 KV 读一遍，**KV 读取量随 batch 线性增长**。例外是多条序列共享同一段前缀：物理上 KV 只存一份，专门的 kernel 还能让它只读一次（第 3.5 节）。

所以 dense 模型 decode 一步的字节数 ≈ 权重字节 + B × ctx × 每 token KV 字节（B 是 batch 里的序列数）。

## 3. Arithmetic intensity 与 roofline

### 3.1 定义与 ridge point

**arithmetic intensity（算术强度）** I = FLOPs ÷ 从显存读写的字节数，单位 FLOPs/Byte。[Roofline 模型](https://dl.acm.org/doi/abs/10.1145/1498765.1498785)（Williams、Waterman、Patterson，CACM 2009）说：

```
可达算力 = min(峰值算力, 显存带宽 × I)

  算力 ▲
       │            ┌──────────────── 峰值算力（compute-bound 区）
       │          ／
       │        ／   ← 斜率 = 显存带宽（memory-bound 区）
       │      ／
       │    ／
       └──┴─────────┴──────────────▶ I（FLOPs/Byte）
                    ridge point = 峰值算力 ÷ 显存带宽
```

**ridge point（屋脊点）** 是两段的交点：I 低于它，时间由带宽决定；高于它，由算力决定。H100 SXM 的 BF16 dense 峰值 989 TFLOPS（NVIDIA 规格页写的 1,979 是带稀疏的数字）、带宽 3.35 TB/s，ridge ≈ **295 FLOPs/Byte**；FP8 算力翻倍，ridge 也翻倍（约 591）。各卡的数值见第 10 节的实验输出。

### 3.2 prefill vs decode：intensity 约等于「一次处理的 token 数」

一个 [B × d] 的激活乘 [d × d′] 的 BF16 权重：FLOPs = 2·B·d·d′，字节 ≈ 2·d·d′（B ≪ d 时权重占绝大部分），所以 **I ≈ B**，B 是这次矩阵乘里的 token 数。

| 阶段 | 一次矩阵乘里的 token 数 | I（线性层） | 在 H100（BF16）上 |
|---|---|---|---|
| prefill，4k prompt 全部未命中 | 4,096 | ≈ 数千 | 远高于 295，compute-bound |
| **增量 prefill**，命中后只算 128 个新 token | 128 | ≈ 128 | 低于 295，**memory-bound**，而且还要读整段已缓存的 KV |
| decode，batch = 1 | 1 | ≈ 1 | memory-bound，算力闲置 99% 以上 |
| decode，batch = 64 | 64 | ≈ 64 | 仍然 memory-bound |
| MoE decode（DeepSeek-V3），全局 batch = B | 每个 routed expert 只分到 B × 8 / 256 | ≈ B / 32 | 见第 3.4 节 |

这正是 [Splitwise（arXiv:2311.18677）](https://arxiv.org/abs/2311.18677) 所说的「prompt computation 是 compute-intensive，token generation 是 memory-intensive」，也是 [JAX scaling book](https://jax-ml.github.io/scaling-book/inference/) 所说的「prefill 时矩阵乘基本都是 compute-bound」。但在高命中率的 agent 负载里，**每轮真正要 prefill 的往往只有几百个新 token（工具结果），这时 prefill 也是 memory-bound 的**。例：Llama-3-8B 在 1 张 H100 上、前面 3,968 个 token 已命中、只算 128 个新 token，计算约 3.5 ms（MFU 60%），读一遍权重和 4k 的 KV 却要约 6.2 ms（MBU 80%）。三个推论：

1. chunked prefill（把长 prompt 切块和 decode 混跑，见[模块 02](../02_inference_engines/README.md)）的块至少要约 ridge 个 token（H100 BF16 约 300），才能吃满算力；
2. 把一小段 prefill 捎带进 decode 步几乎是免费的，因为那一步本来就要读一遍权重；
3. 命中率高时，TTFT 由「权重 + 已缓存 KV 的字节 ÷ 带宽」决定，而不是由 FLOPs 决定（第 7 节）。

**心算例子**：Llama-3-8B 在 1 张 H100 上 decode，batch = 1、上下文 4k：每步读约 15.5 GB（不含 embedding 表的权重约 15.0 GB，加 4k 的 KV 0.54 GB），按 3.35 TB/s 满带宽也要约 4.6 ms，也就是单序列最多约 215 token/s；而这一步线性层的计算量只有约 15 GFLOPs，按峰值算只要约 15 µs。两者差 300 倍，这是**假设 KV 可以忽略时** batching 的理想空间；在 4k 上下文下，KV 读取会把能收回的倍数卡在约 32 倍（第 5 节）。

### 3.3 decode 里的注意力：intensity = 每个 KV 头服务的 query 头数

注意力和线性层不同：**序列之间不共享前缀时，每条序列读的是自己的 KV，batch 再大也摊不薄**。每层每个上下文 token：

- FLOPs = 4 × n_heads × head_dim（QKᵀ 和 AV）
- 字节 = 2（K 和 V）× n_kv_heads × head_dim × 2 B（BF16）

所以 I_attn = n_heads / n_kv_heads，**与 batch 无关**（FP8 KV 时翻倍）。

| 注意力 | 例子 | I_attn（BF16 KV） |
|---|---|---|
| MHA（多头注意力，每个 query 头有自己的 KV） | n_heads = n_kv_heads | 1 |
| GQA（分组查询注意力，多个 query 头共享一组 KV，[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)） | Llama-3-8B：32 / 8 | 4 |
| GQA | Llama-3-70B：64 / 8 | 8 |
| MLA（多头潜在注意力，所有头共享一个压缩后的潜在向量），128 个头都在同一张卡上 | DeepSeek-V3：128 个头共享 576 维潜在向量 | ≈ 242（推导见下） |
| MLA，注意力按 TP 切成 t 份 | 每张卡只算 128 / t 个头，却要读完整的潜在向量 | ≈ 242 / t（TP4 约 60，TP8 约 30） |

MLA 的推导（吸收权重后的 decode 形式，按 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) 的 kv_lora_rank = 512、qk_rope_head_dim = 64、num_attention_heads = 128）：每层每个上下文 token，128 个头各做一次 576 维点积和一次 512 维加权求和，FLOPs = 128 × 2 × (576 + 512) = 278,528；读的字节是 576 × 2 = 1,152。I ≈ 242，已经接近 H800 的 ridge（与 H100 相同，约 295）。这就是为什么 DeepSeek 的 [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 会同时报告「memory-bound 配置下最高 3000 GB/s」和「compute-bound 配置下 660 TFLOPS」：**MLA 把 decode 注意力从纯带宽问题变成了接近算力问题。**

**242 的前提是 DP attention**：一张卡上有全部 128 个头，注意力按请求（而不是按头）分到不同卡上。如果像普通 dense 模型那样用 TP 切注意力，潜在向量是所有头共享的，没法按头切开，每张卡都要存一份完整的 KV、读完整的潜在向量，却只算 128 / t 个头。DeepSeek-V3 技术报告里的部署单元注意力用的是 TP4 + SP 配合 DP（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) 第 3.4 节），线上系统概览写的是「MLA / Shared Expert DP32（prefill）、DP144（decode）」。DP attention 的代价是：注意力、共享专家这些非 routed 权重每张卡都要存一份。按 config.json 推算，这部分约 17B 参数（注意力约 11.4B、共享专家约 2.6B、前 3 层 dense FFN 约 1.2B、词表约 1.9B）。

### 3.4 MoE：每个专家只分到 B·k/E 个 token

「线性层 I ≈ B」只对 dense 模型成立。MoE 的每个 routed expert 是一个独立的小 FFN，一步里只处理被路由给它的 token：全局 batch 为 B 时，每个专家平均分到 **B·k/E** 个 token（k 是每 token 选几个专家，E 是专家总数）。DeepSeek-V3 的 k = 8、E = 256（[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) 的 num_experts_per_tok、n_routed_experts），所以专家 GEMM 的 **I ≈ B / 32**。

- 要让专家 GEMM 越过 H100 / H800 的 BF16 ridge（约 295），全局每步要有约 295 × 32 ≈ **9,400 个 token**。
- 换成 FP8 权重 + FP8 算力：权重字节减半让 I 翻倍（≈ B / 16），FP8 的 ridge 也翻倍（约 591），两者抵消，仍然要约 9,400 个 token。
- 注意力和共享专家这部分是 dense 的，I ≈ B（每张卡上的 B）；所有专家都被读到、忽略 KV 时，整模型平均 I 约为 B × 37 / 671 ≈ B / 18。

这就是 DeepSeek 要把 decode 铺到 EP144（线上）、EP320（技术报告）的第一性原因：**一张卡上凑不出几千条并发序列，只能让很多张卡上的请求共用同一批专家，把每个专家的 batch 聚起来。** 用第 1 节的推算：一个 decode 单元 18 个节点、每节点约 700 条序列，合计约 1.27 万条，每步每个 routed expert 分到约 1.27 万 × 8 / 256 ≈ 400 个 token（不算 MTP），已经越过 BF16 ridge。ISCA 论文还指出，MTP 每步预测多个 token，「增大了推理 batch，对提高 EP 的计算强度至关重要」。另一方面，ISCA 论文在估算通信下限时假设每台设备放 1 个专家、一次处理约 32 个 token，并说这个数是在计算访存比和通信延迟之间的折中：**大 EP 在强度和延迟之间也要取舍，不是 batch 越大越好。**

### 3.5 共享前缀的注意力：多条序列一起读同一段 KV

第 3.3 节的「KV 各读各的」在序列共享前缀时不成立，而共享前缀恰好是本项目的核心场景：GRPO 对同一个 prompt 采样一组回答、多个 subagent 共享 system prompt 和工具定义、同一个项目的请求共享仓库上下文。

[Hydragen（arXiv:2402.05099）](https://arxiv.org/abs/2402.05099)和 FlashInfer 的 [cascade attention](https://flashinfer.ai/2024/02/02/cascade-inference.html) 把注意力拆成「共享前缀」和「各自后缀」两部分：前缀部分把所有序列的 query 拼成一个矩阵，对同一份前缀 KV 做一次矩阵乘，KV 只读一次。前缀部分的强度就变成 **n_seq × n_heads / n_kv_heads**。

例（推导值）：Llama-3-8B，16 条样本共享 4k 的 prompt，各自生成 512 个 token。朴素做法注意力 I = 4；前缀只读一次后，整体 I = 4 × 16 × (4096 + 512) / (4096 + 16 × 512) = **24**。Hydragen 报告在 CodeLlama-13b 上端到端吞吐最高提升 32 倍，FlashInfer 报告相对 vLLM 的 PagedAttention kernel 最高 31 倍。**harness 让多条请求共享前缀，不只省了显存和 prefill，还直接提高了 decode 的算术强度。**

## 4. KV cache 有多大

$$\text{KV 字节/token} = \begin{cases} 2 \times L \times n_{kv} \times d_{head} \times b & \text{MHA / GQA} \\ L \times (d_c + d_h^R) \times b & \text{MLA} \end{cases}$$

L 是层数，b 是每个元素的字节数（BF16 为 2），d_c 是 MLA 的 kv_lora_rank，d_h^R 是解耦的 RoPE key 维度。[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)写明生成时只需要缓存压缩后的潜在向量和解耦的 key（d_c = 512，d_h^R = 64）。

| 模型 | 注意力 | 结构参数（来自官方 config / 论文） | KV 字节/token（BF16） |
|---|---|---|---|
| Llama-3-8B | GQA | 32 层，8 个 KV 头，head_dim 128 | 2 × 32 × 8 × 128 × 2 = 131,072 B = **128 KiB** |
| Llama-3-70B | GQA | 80 层，8 个 KV 头，head_dim 128 | 2 × 80 × 8 × 128 × 2 = 327,680 B = **320 KiB** |
| Qwen3-32B | GQA | 64 层，8 个 KV 头，head_dim 128 | 2 × 64 × 8 × 128 × 2 = 262,144 B = **256 KiB** |
| DeepSeek-V3 | MLA | 61 层，512 + 64 | 61 × 576 × 2 = 70,272 B ≈ **68.6 KiB** |

结构参数来源：Llama 3 见[论文 Table 3（arXiv:2407.21783）](https://arxiv.org/abs/2407.21783)，Qwen3-32B 见 [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json)，DeepSeek-V3 见 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。

几个反事实，帮你记住 GQA / MLA 省了多少（均为按公式推导）：

- 如果 Llama-3-70B 用 MHA（64 个 KV 头），每 token 是 2 × 80 × 64 × 128 × 2 B = **2.5 MiB**，是 GQA 的 8 倍。
- 如果 DeepSeek-V3 的 128 个头都按 MHA 缓存完整的 K（128 + 64 = 192 维）和 V（128 维），每 token 是 61 × 128 × 320 × 2 B ≈ **4.77 MiB**，约是 MLA 的 71 倍。DeepSeek-V2 的摘要给的口径是：与 DeepSeek 67B 相比，KV cache 减少 93.3%，最大生成吞吐提升到 5.76 倍（[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)）。

**一条 128k 上下文的 Llama-3-70B 序列就要 128k × 320 KiB = 40 GiB 的 KV**，半张 H100。

## 5. batch size 如何改变 decode

把第 2、3 节合起来，一步 decode 的 intensity（B 条序列，每条上下文 ctx，BF16，序列间不共享前缀）：

$$I(B) = \frac{2N \cdot B + 4 L\, d_{attn}\, ctx \cdot B}{W(B) + B \cdot ctx \cdot kv}$$

N 是激活参数量，W(B) 是这一步**被读到的权重字节**：dense 模型 W(B) = 2N，与 B 无关；MoE 模型 W(B) = 非 routed 权重 + routed 权重 × (1 − (1 − k/E)^B)，即 batch 覆盖到的专家越多，读的权重越多，B 大到一定程度后所有专家都被读到。

- B 小的时候，分母被权重主导，dense 模型 I ≈ B：**加 batch 几乎白送吞吐**，因为权重只读一次。
- B 大的时候，分母被 KV 主导，I 趋近一个**上限** (2N + 4·L·d_attn·ctx) / (ctx·kv)，它只和上下文长度与注意力结构有关，与 batch 无关。

以 Llama-3-8B 为例（推导值，BF16，N ≈ 7.5B）：上下文 128 时上限约 900，远高于 H100 的 295，加 batch 最终能变 compute-bound；上下文 4k 时上限只有约 32，**batch 再大也只能停在 memory-bound**；上下文 32k 时上限约 7.5，其中 4 来自注意力本身（GQA 的 32 / 8），约 3.5 来自线性层 FLOPs 摊到 KV 字节上。

结论：**序列之间不共享前缀时，真实 agent 上下文（几千到几万 token）下 GQA 模型的 decode 是 memory-bound 的，加 batch 救不回来**。想让 decode 吃满算力，要么提高注意力的 intensity（MLA 配 DP attention、共享前缀的 cascade attention），要么减少每 token 的字节（FP8 / 更低精度 KV），要么让每一步验证不止一个 token（投机解码，speculative decoding，见[模块 02](../02_inference_engines/README.md)）。

时间模型（库里的 `step_time_s` 就是这个）：

$$t_{step} = \max\left(\frac{\text{FLOPs}}{n_{gpu} \cdot \text{峰值} \cdot \text{MFU}},\ \frac{\text{字节}}{n_{gpu} \cdot \text{带宽} \cdot \text{MBU}}\right)$$

MFU（model FLOPs utilization）和 MBU（model bandwidth utilization）是实际能达到的算力和带宽比例。decode 时 TPOT ≈ t_step，整机吞吐 ≈ B / t_step。权重读取占大头时，t_step 几乎不随 B 变，**吞吐随 B 线性涨，TPOT 几乎不变**；KV 读取占大头之后，t_step 随 B 线性涨，吞吐不再增长。

**这个模型少了什么**（面试官一定会问）：

1. **通信**。TP 每层要做 2 次 all-reduce（注意力之后、MLP 之后），Llama-3-70B 每一步 decode 就是 160 次集合通信；按 ring all-reduce，每次每卡收发约 2(n − 1)/n × B × d_model × 2 字节。70B、TP8、B = 40 时每步每卡合计约 180 MB，按 NVLink 单向 450 GB/s 算只要约 0.4 ms，带宽不是问题，但 160 次通信各自的固定延迟会累加。MoE 的 EP 要做两次 all-to-all（dispatch 和 combine）：DeepSeek 的 ISCA 论文按每 token 发往 8 个 routed + 1 个共享专家、dispatch 用 FP8、combine 用 BF16、隐藏维约 7K、每设备 32 个 token、400G IB 50 GB/s 计算，每层通信 (1 字节 + 2 字节) × 32 × 9 × 7K ÷ 50 GB/s = 120.96 µs，双 micro-batch 重叠下每层 241.92 µs，61 层合计 **14.76 ms 的 TPOT 下限，与 HBM 带宽无关**。
2. **固定开销**：kernel launch、调度、采样。batch 小、步长只有几毫秒时，它们占比最大（CUDA Graph 就是用来压这部分的）。
3. **roofline 是上界，不是预测器**。MFU / MBU 强烈依赖 GEMM 形状、kernel 融合、batch 分桶。小 batch 的矩阵 × 向量几乎拿不到峰值算力，所以「I 越过 ridge」只是 compute-bound 的必要条件，不是充分条件。做估算时至少换 2–3 组 MFU / MBU 看结论稳不稳。

更贴近真实的写法是 t_step ≈ max(计算, 访存) + 没能被重叠的通信 + 固定开销。本项目 `codesign_sim` 的 `step_time_s` 默认只算前两项（通信要通过 `comm_s` 参数显式传入），本模块的实验也只用前两项，这一点要在任何结论里说明。

## 6. 显存预算与最大并发

$$\text{最大并发} \approx \left\lfloor \frac{n_{gpu} \cdot \text{HBM} \cdot (1 - r) - \text{权重字节}}{ctx \times \text{KV 字节/token}} \right\rfloor$$

r 是给激活、临时缓冲区和碎片预留的比例（本项目取 10%）。**这个公式是乐观上界**：它假设 n_gpu 张卡组成一个模型并行组，权重只存一份，KV 在所有卡上均匀切分。真实部署要按**每张卡**算：卡上可用显存 − 这张卡上的权重分片 − 预留，再除以这张卡上每条序列的 KV。会让每卡结果变差的因素有：

- TP 大于 KV 头数（例如 Llama-3-70B 只有 8 个 KV 头，TP16 时 KV 头不够切），或者 MLA 用 TP：KV 要在多张卡上各存一份；
- DP attention：注意力等非 routed 权重每卡一份；
- EP：冗余专家（DeepSeek 为高负载专家额外放 32 个副本）；
- PP：每张卡只放部分层的权重和 KV，但流水线要填满需要更多在途序列；
- workspace、CUDA Graph 显存池、分页 KV 的尾块浪费。

心算三个例子（HBM 按 1 GB = 1e9 B）：

| 部署 | 权重 | 可用于 KV | 每 token KV | 8k 上下文最大并发 | 32k 上下文最大并发 |
|---|---|---|---|---|---|
| Llama-3-8B，1 × H100（聚合公式） | ≈ 16.1 GB | 80 × 0.9 − 16.1 ≈ 55.9 GB | 128 KiB | ≈ 52 | ≈ 13 |
| Llama-3-70B，8 × H100（聚合公式） | ≈ 141 GB | 640 × 0.9 − 141 ≈ 435 GB | 320 KiB | ≈ 162 | ≈ 40 |
| DeepSeek-V3，32 × H800，FP8 权重，DP attention + EP32（按每卡算） | 每卡 ≈ 40 GB | 每卡 72 − 40 ≈ 32 GB，共约 1,024 GB | 68.6 KiB | ≈ 1,779 | ≈ 444 |

DeepSeek-V3 这一行的推算：非 routed 参数约 17.0B 每卡一份；256 个 routed 专家加 32 个冗余副本分到 32 张卡，每卡每层 9 个专家（与线上系统概览「每张 GPU 负责 9 个 routed 专家」一致），每个专家 3 × 7168 × 2048 ≈ 44M 参数，58 个 MoE 层合计约 23B；FP8 按每参数 1 字节近似。**同样 32 张卡，第 10 节表 [3] 的聚合公式会给出约 2,836 条 8k 序列，高估约 1.6 倍**，差在非 routed 权重被复制了 32 份、再加上冗余专家。

要点：

1. **显存把 batch 卡在 ridge 下面**。70B 在 8k 上下文最多约 162 条，离 295 还远，而且第 5 节已经说明长上下文时上限更低。
2. **上下文翻 4 倍，并发除以 4**。agent 的上下文越滚越长，同一台机器能服务的会话数就按比例下降。
3. DeepSeek-V3 的 FP8 权重约 671 GB，**一台 8 × 80 GB 的机器连权重都放不下**，这是它必须多机部署、并用 EP 把专家摊到很多卡上的原因之一；DeepSeek 线上 prefill 用 4 个节点、decode 用 18 个节点为一个单元（[推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)）。

## 7. 指标：TTFT / TPOT / ITL / 吞吐 / goodput / SLO

| 指标 | 定义 | 主要由什么决定 | 在 agent / RL 里意味着什么 |
|---|---|---|---|
| **TTFT**（time to first token，首 token 延迟） | 从提交请求到收到第一个输出 token | 排队 + prefill（未命中缓存的那部分）| agent 每轮都要等一次 TTFT，几十轮串起来就是任务延迟的大头 |
| **TPOT**（time per output token，每输出 token 时间） | (端到端延迟 − TTFT) ÷ (输出 token 数 − 1)，每个请求算一个值 | decode 步长，即 batch 和上下文长度 | 决定长输出（推理链、代码）的耗时 |
| **ITL**（inter-token latency，token 间延迟） | 相邻两个输出 token 之间的间隔，每个间隔一个样本 | 同上，外加被 prefill 插队造成的卡顿 | 看 p99 ITL 才能发现「偶尔卡一下」 |
| **吞吐** | 整个系统每秒输出（或处理）的 token 数 | batch 大小、MFU / MBU | RL rollout 最关心；每用户速度 × 并发数 |
| **goodput** | 在达到 SLO 达标率目标的前提下能承受的最大请求速率 | 调度策略、PD 分离 | 不满足 SLO 的吞吐不算数 |
| **SLO**（service level objective，服务等级目标） | 例如「TTFT p99 < 2 s 且 TPOT p99 < 50 ms」 | 业务定 | 决定 goodput 的门槛 |

几个容易混的点：

- NVIDIA 的[基准测试文章](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)把 TPOT 当作 ITL 的别名，公式都是 (e2e − TTFT) / (输出 token 数 − 1)。[Anyscale 文档](https://docs.anyscale.com/llm/serving/benchmarking/metrics)则区分了聚合方式：**平均 TPOT 每个请求权重相同，平均 ITL 按 token 加权**，所以长请求多的时候两者会不一样。面试时先问清楚口径。
- goodput 的口径也不统一。[DistServe（arXiv:2401.09670）](https://arxiv.org/abs/2401.09670)用的是 **per-GPU goodput**：在达到 SLO 达标率目标（论文举例 90% 的请求同时满足 TTFT 和 TPOT）的前提下，每张 GPU 能承受的最大请求速率。它是一个速率，必须带着达标率百分比说，因为要求 100% 达标是做不到的。Anyscale 文档则把 goodput 定义为「满足全部 SLO 的请求占比」，是一个百分比，更接近 SLO 达标率。说 goodput 之前先说清单位。
- **TTFT 心算**：TTFT ≈ 排队 + prefill 未命中部分的 max(FLOPs ÷ (n_gpu × 峰值 × MFU)，字节 ÷ (n_gpu × 带宽 × MBU))，字节 = 权重 + 读写整段 KV。Llama-3-8B、4k prompt、1 × H100、MFU 60%：(2 × 6.98e9 × 4096 + 2 × 4096² × 4096 × 32) ≈ 6.2e13 FLOPs → 约 **0.10 s**，compute-bound（6.98B 是去掉输入 embedding 和 lm_head 的参数；prefill 只在最后一个位置算 lm_head）。命中一半前缀，就只算后一半（后一半 token 看到的上下文更长，注意力项不会正好减半）。命中 3,968 个 token、只算 128 个时，计算只要约 3.5 ms，读权重和 4k 的 KV 却要约 6.2 ms：**命中率高时 TTFT 由带宽决定**（这里按整份权重读一遍估算，略偏保守）。
- 这个心算是**下界**。线上 TTFT 还要加上：前缀缓存的查找和从 DRAM / SSD / 远端搬回 KV、TP / PP 通信、chunked prefill 被切片后和 decode 交错造成的等待、采样、网络和 detokenize。第一个输出 token 直接来自 prefill 那次前向最后一个位置的 logits，不存在单独的「第一步 decode」。

## 8. Little's law：并发、吞吐与延迟的恒等式

[Little's law](https://pubsonline.informs.org/doi/abs/10.1287/opre.9.3.383)（Little，Operations Research 1961）：稳态下，**系统里平均有多少个东西 L = 到达率 λ × 每个东西平均停留时间 W**。它不依赖分布假设，是容量规划里最好用的一条公式。

三种用法：

1. **请求并发**：每秒到达 20 个请求，每个在系统里平均待 15 s，系统里平均就有 300 个请求，你要为 300 条序列准备 KV。
2. **token 形式**：decode 时 batch 里的序列数 B = 吞吐（token/s）× TPOT。要 5,000 token/s，如果 TPOT 正好卡在 40 ms，就需要 200 条序列同时在跑；TPOT 更低，需要的序列就更少。TPOT 本身又随 B 变（第 5 节），所以真正要找的是满足 B / TPOT(B) ≥ 5,000 且 TPOT(B) ≤ 40 ms 的 B，再用第 6 节检查显存。第 1 节「每个 DeepSeek decode 节点约 700 条序列」也是这么算的。
3. **agent 形式**：如果工具调用期间 KV 被 pin 住（留在显存），W 要算上工具等待时间。每秒新来 2 个会话、每个会话生成 30 s、工具等待 60 s，那么平均有 2 × (30 + 60) = 180 个会话占着 KV，其中 2/3 是在等工具。**工具等待直接把显存占用放大了 3 倍**，这就是[模块 00](../00_map/README.md)第 3.2 节「KV 留还是丢」的数量级来源，也是练习 (d) 要做的判断。

**容量规划要看分位数，不能只看均值**。Little's law 只给均值：按均值配显存，一半时间会不够，引擎就要频繁抢占。如果会话按泊松过程到达，系统里的会话数服从均值为 L 的泊松分布（M/G/∞ 排队的稳态结论，与每个会话待多久的分布无关），p99 约为 L + 2.33√L。L = 180 时约 211，要多留 17%。到达有突发、或者每个会话的 KV 大小差别很大时，波动更大。另外，会话内的上下文是逐轮增长的：如果线性增长，一个会话平均占用的 KV 约是它最终长度的一半。实验表 [6b] 按全程 32k 计算，是偏保守的估计。

## 9. 必须会心算的数字

| 量 | 典型值 | 怎么推出来 | 来源 |
|---|---|---|---|
| 前向 FLOPs / token | 2N（MoE 用激活参数） | 每个权重一次乘加 | [Kaplan et al.](https://arxiv.org/abs/2001.08361) |
| H100 SXM BF16 dense 峰值 / 带宽 | 989 TFLOPS / 3.35 TB/s | 规格页 1,979 是稀疏数字，÷2 | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| H100 ridge point | BF16 ≈ 295 FLOPs/B，FP8 ≈ 591 | 峰值 ÷ 3.35e12 | 推导 |
| A100 ridge point | ≈ 153 FLOPs/B | 312e12 ÷ 2.039e12 | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| dense decode 线性层 intensity | ≈ batch 大小 | 权重只读一次 | 推导 |
| MoE routed expert intensity | ≈ B·k/E（DeepSeek-V3 为 B/32，越过 BF16 ridge 要约 9,400 个 token） | 每个专家只分到 B·k/E 个 token | 推导，k、E 见 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| decode 注意力 intensity | n_heads / n_kv_heads（MLA 配 DP attention ≈ 242，TP t 时 ÷ t） | 每条序列 KV 各读各的 | 推导 |
| 增量 prefill 变 compute-bound 的门槛 | 新 token 数 ≳ ridge（H100 BF16 约 300） | I ≈ 一次处理的 token 数 | 推导 |
| Llama-3-8B / 70B KV | 128 / 320 KiB/token | 2·L·n_kv·d·2 | [Llama 3 论文](https://arxiv.org/abs/2407.21783) |
| DeepSeek-V3 KV | ≈ 68.6 KiB/token | 61 × 576 × 2 | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| 8B 单卡 batch = 1 decode | 满带宽 ≈ 215 token/s；MBU 0.8 时 ≈ 172 token/s | 3.35 TB/s ÷ 15.5 GB（权重不含 embedding 表） | 推导 |
| 8B 4k prompt 的 TTFT | ≈ 0.10 s（1 × H100，MFU 60%） | 6.2e13 ÷ (989e12 × 0.6) | 推导 |
| DeepSeek 线上每节点 prefill / decode | 73.7k（含命中；扣掉 56.3% 命中约 32k）/ 14.8k token/s | 线上统计（8 × H800） | [DeepSeek 推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| DeepSeek-V3 EP 通信决定的 TPOT 下限 | ≈ 14.76 ms（400G IB） | 61 层 × 2 × 120.96 µs | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |

完整的数字表见 [必须会心算的数字](../../docs/numbers-cheatsheet.md)。

## 10. 实验

```bash
.venv/bin/python modules/01_inference_basics/lab.py
# 或
make lab N=01
```

在 M1 笔记本上 1 秒内跑完，共 6 组、8 张表（[5] 和 [6] 各两张）。下面是真实输出，省略了 Llama-3-8B 那张 [5]：

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

怎么读：

1. **表 [1]**：H20 的 ridge 只有 37，intensity = 64 时已经拿满自己的 148 TFLOPS，而 H100 只拿到 214 / 989。所以 decode（intensity 几到几十）在 H20 上「相对效率」高得多，prefill 在 H20 上则慢得多。B200 的 ridge（281）和 H100（295）差不多：算力和带宽同比例涨，decode 照样 memory-bound。
2. **表 [2]、[3]**：DeepSeek-V3 每 token 的 KV 是 Llama-3-70B 的 0.21 倍。表 [3] 是**聚合公式的理想上界**：8 × H800 连 DeepSeek-V3 的 FP8 权重都放不下；32 张卡（DeepSeek prefill 单元的规模）按聚合公式能放约 2,836 条 8k 序列，按真实的 DP attention + EP32 每卡计算只有约 1,779 条（第 6 节），差在非 routed 权重复制了 32 份。
3. **表 [4]**：「B→∞」列就是第 5 节说的上限。Llama-3-8B 在 4k 上下文的上限是 32.0，Llama-3-70B 在 32k 是 20.9，都远低于 295，**batch 再大也是 memory-bound**。DeepSeek-V3 行的分母用的是第 5 节的 W(B)：小 batch 只读被选中的专家，batch 大了所有专家都要读。它在 4k 上下文的上限是 492.4，高于 ridge，但权重有 671B，B = 4096 时整模型 intensity 还只有 230.5；按第 3.4 节的每专家视角，要约 9,400 个并发 token 才能让专家 GEMM 越过 ridge，这正是大规模 EP 要做的事。上下文到 32k 时，DeepSeek-V3 的上限也只有 273.1，BF16 KV 下同样落回 memory-bound。
4. **表 [5]**：batch 从 1 涨到 40，吞吐从 143 涨到 1,509 token/s（10.6 倍），TPOT 从 6.98 ms 涨到 26.52 ms（3.8 倍）；到 64 条时显存已经放不下（fits = NO）。32k 上下文下每条序列的 KV 读取占了大头，所以 TPOT 涨得明显；上下文短、权重读取占大头时，TPOT 会平得多。这些时间都不含 TP 的 all-reduce（第 5 节）。
5. **表 [6a]**：命中一半并不省一半。Llama-3-70B 的 64k prompt 从 3.076 s 降到 1.834 s（省约 40%），因为后一半 token 看到的上下文更长；命中 90% 降到 0.414 s，命中 99% 降到 0.043 s。4k prompt 命中 99% 时只算几十个 token，TTFT 约 6–7 ms，几乎全是读权重的时间：**这时 prefill 是 memory-bound 的**。
6. **表 [6b]**：工具等待从 0 涨到 60 s，pin 住的 KV 均值从 644 GB 涨到 1,933 GB，相当于 4.4 台 8 × H100 的全部 KV 空间；300 s 时要 16.3 台。均值不够用：泊松到达时，60 s 那一行的 p99 是 212 个会话、2,276 GB，比均值多约 18%；会话数越少，相对余量越大（第一行 79 对 60）。反过来，上下文逐轮增长时平均占用约减半（第 8 节）。

改参数看现象：

- 把 `table_kv` 里的 `kv_bytes_per_token(m)` 改成 `kv_bytes_per_token(m, 1)`（FP8 KV）：所有序列体积减半。再把 `intensity` 里的 `decode_bytes` 加上 `kv_dtype_bytes=1`、`intensity_limit` 里的 `kv_bytes_per_token(cfg)` 改成 `kv_bytes_per_token(cfg, 1)`，看表 [4] 的上限怎么翻倍。
- 把 `DEPLOYMENTS` 和 `table_ttft` 里的 `"H100-SXM"` 换成 `"H20"`：每卡多 16 GB 显存、带宽更高，但 prefill 受算力限制，TTFT 会慢约 989 / 148 ≈ 6.7 倍。再看命中 99% 那一列：4k prompt 只算几十个 token，H100 上已经是 memory-bound，H20 上却仍是 compute-bound（它的 ridge 只有约 37），所以 H20 依然更慢。
- 把 `main()` 里第二个 `table_decode_steps` 的上下文从 32768 改成 4096：显存能放的序列数约变为 8 倍，intensity 也明显变高。

## 11. 练习

在 [`exercise.py`](exercise.py) 里实现四个函数，只用 `codesign_sim` 的公开接口（`ridge_point`、`kv_bytes_per_token`、`prefill_flops`、`decode_flops`、`decode_bytes`、`step_time_s`）。前三个是把公式写成代码，第四个需要做判断：

| 函数 | 要做什么 | 对应小节 |
|---|---|---|
| `max_concurrency` | 装下全部权重后，最多能同时放多少条给定长度序列的 KV（聚合公式的乐观上界）；权重放不下返回 0 | 第 6 节 |
| `critical_batch` | decode 的 intensity 达到 ridge point 的最小 batch；到 `max_batch` 还达不到返回 `None`；FP8 权重 / KV 要传下去 | 第 5 节 |
| `estimate_ttft` | 排队 + prefill 未命中部分，字节算上权重和整段 KV；整段命中时也至少要算 1 个 token | 第 7 节 |
| `tool_wait_kv_action` | 工具调用期间，这段 KV 该留（keep）、卸载（offload）还是丢掉重算（recompute），并给出工具返回后额外的延迟 | 第 8 节、[模块 00](../00_map/README.md) 第 3.2 节 |

`tool_wait_kv_action` 的成本模型：假设实例的并发受 KV 显存限制，那么占住整机 KV 容量的 share 这么多份额、占 tool_s 秒，就相当于损失了 tool_s × share 个「整机秒」的吞吐；重算要占整机算力 recompute_s 秒。两者单位相同，可以直接比：**tool_s × share ≤ recompute_s 就留**。留不住时，再比「搬回来的时间」和「重算的时间」。这个模型的边界：实例不缺显存时，留着几乎不花钱；卸载的带宽和其他请求共享；工具时延是随机的，应该按分布而不是均值设 TTL（[Continuum](https://arxiv.org/abs/2511.02230) 就是这么做的）。

```bash
make module N=01                         # 测你的实现（没写完会失败，这是正常的）
INFRA_SOLUTION=1 make module N=01        # 用参考答案 solution.py 跑同一套测试
```

做完之后的三个追加实验：

1. 用 `critical_batch` 对比 BF16 KV 和 FP8 KV（`kv_dtype_bytes=1`）：Llama-3-8B 在 512 token 的上下文、`max_batch=100000` 时，BF16 KV 返回 `None`，FP8 KV 返回 826。**KV 字节减半，上限翻倍，才越过了 ridge。**
2. 用 `tool_wait_kv_action` 找「留」的门槛：Llama-3-70B、8 × H100、64k 上下文约 62 s；Llama-3-8B、1 × H100、4k 上下文只有约 11 s。为什么小模型、短上下文反而更不值得留？（提示：重算便宜，而 4k 的 KV 在 8B 的单卡上并不算小。）
3. 把 `offload_gbps` 从 512 一路降到 1，看 70B / 64k 什么时候从 offload 变成 recompute，再和[模块 05](../05_cache_prefill/README.md) 第 4.2 节的「盈亏平衡带宽」对照。

## 12. 自测题

1. H100 SXM 的 BF16 峰值是 989 还是 1,979 TFLOPS？ridge point 是多少？
   <details><summary>答案</summary>dense 是 989 TFLOPS，NVIDIA 规格页写的 1,979 带稀疏（表格脚注 "With sparsity"）。ridge ≈ 989e12 / 3.35e12 ≈ 295 FLOPs/Byte。</details>
2. 为什么 dense 模型 decode 时线性层的 arithmetic intensity 约等于 batch 大小？MoE 呢？
   <details><summary>答案</summary>[B × d] 乘 [d × d′]：FLOPs = 2Bdd′，字节主要是权重 2dd′（BF16），比值约为 B。权重每步只读一次，被 batch 里所有序列共享。MoE 的每个 routed expert 只分到 B·k/E 个 token，DeepSeek-V3 是 B/32，所以要约 9,400 个并发 token 才越过 BF16 ridge，这是大规模 EP 的第一性原因。</details>
3. 为什么增大 batch 救不了长上下文的 decode 注意力？什么情况下例外？
   <details><summary>答案</summary>序列之间不共享前缀时，每条序列读的是自己的 KV，batch 内摊不薄；注意力的 intensity = n_heads / n_kv_heads（GQA-8 组时为 8），与 batch 无关，远低于 ridge。例外是共享前缀：cascade attention / Hydragen 让前缀 KV 只读一次，前缀部分的强度变成 n_seq × n_heads / n_kv_heads。</details>
4. Llama-3-70B 在 8 × H100 上、32k 上下文，最多能同时跑多少条序列？这个数是上界还是下界？
   <details><summary>答案</summary>(640 × 0.9 − 141) GB ≈ 435 GB；每条序列 32,768 × 327,680 B ≈ 10.7 GB；约 40 条。这是聚合公式的上界，假设权重只存一份、KV 均匀切分，还没扣 workspace、CUDA Graph 显存池和分页 KV 的尾块浪费。</details>
5. MLA 为什么能让 decode 注意力接近 compute-bound？前提是什么？
   <details><summary>答案</summary>128 个头共享同一个 576 维潜在向量，每读 1,152 字节要做约 128 × 2 × (576 + 512) ≈ 2.8e5 FLOPs，intensity ≈ 242，接近 H800 的 ridge（约 295）。前提是 128 个头都在同一张卡上（DP attention）；用 TP 切成 t 份时，每卡还是要读完整的潜在向量，I ≈ 242 / t，而且 KV 要复制 t 份。</details>
6. 平均 TPOT 和平均 ITL 什么时候会不一样？
   <details><summary>答案</summary>平均 TPOT 每个请求权重相同，平均 ITL 按 token 加权。长短请求混合时，长请求贡献了更多 ITL 样本，两者就会不同。</details>
7. 一个服务要 5,000 token/s 的 decode 吞吐，TPOT 不能超过 40 ms，batch 里要多少条序列？
   <details><summary>答案</summary>Little's law：B = 吞吐 × TPOT。TPOT 正好 40 ms 时需要 200 条，这是满足 SLO 时的上限；TPOT 更低则需要更少。TPOT 本身随 B 变，所以要找满足 B / TPOT(B) ≥ 5,000 且 TPOT(B) ≤ 40 ms 的 B。按本项目的成本模型，Llama-3-8B、1k 上下文、单张 H100，B = 38 时 TPOT 约 7.5 ms，已经达到 5,000 token/s。</details>
8. DeepSeek 报告每个 8 × H800 节点 prefill 约 73.7k token/s、decode 约 14.8k token/s。能说「prefill 比 decode 快 5 倍」吗？怎么解释剩下的差距？
   <details><summary>答案</summary>不能直接这么说。73.7k 含缓存命中，扣掉 56.3% 的命中，真正计算的约 32k token/s，约为 decode 的 2.2 倍；两者的部署单元不同（4 节点 EP32 vs 18 节点 EP144）；14.8k 是否含 MTP 加速没有说明。剩下的差距才归到 roofline：prefill 一次处理大量 token，矩阵乘 intensity 高，受算力限制；decode 每步每条序列只产出 1 个 token，要反复读权重和 KV，受带宽限制。</details>

## 13. 面试追问

1. **「为什么 decode 是 memory-bound？怎么让它不那么 memory-bound？」**
   要点：先说结论——decode 每步每条序列只出 1 个 token，dense 线性层 intensity ≈ batch，注意力 intensity = n_heads / n_kv_heads，都远低于 H100 的 ridge（约 295）。办法：加大 batch（受 KV 显存限制）、减小每 token 字节（GQA / MLA / FP8 KV / 权重量化）、共享前缀时用 cascade attention、投机解码让一步验证多个 token。取舍：batch 大了 TPOT 变长；量化有精度风险；投机解码在接受率低时白算。
2. **「给你 8 张 H100 和 Llama-3-70B，32k 上下文，能撑多少并发？吞吐大概多少？」**
   要点：权重约 141 GB，可用 KV 约 435 GB，每条序列约 10.7 GB，约 40 条（上界）。每步读约 139 GB 权重（不含 embedding 表）+ 40 × 10.7 GB ≈ 568 GB，8 卡合计 26.8 TB/s，按 MBU 80% 算约 26.5 ms 一步，吞吐约 40 / 0.0265 ≈ 1,500 token/s。这个估算没算 TP 每步 160 次 all-reduce 的延迟，实际 TPOT 会偏高。要提高：FP8 KV（并发约翻倍）、prefix 共享、PD 分离让 decode 机器不做 prefill。
3. **「TTFT 太高，怎么拆解？」**
   要点：TTFT = 排队 + 前缀缓存查找与 KV 加载 + prefill 未命中部分（chunked prefill 下还有和 decode 交错造成的等待）+ TP / PP 通信 + 采样、网络、detokenize。第一个 token 就来自 prefill 最后一个位置的 logits，没有单独的「第一步 decode」。先看排队（Little's law，到达率 vs 服务能力），再看 prefill：命中率、命中在哪一层、chunk 大小、TP 并行度。命中率高时 prefill 已经 memory-bound，再加算力没用。取舍：chunked prefill 保护 TPOT 但会拉长单个长 prompt 的 TTFT；PD 分离消除干扰但要搬 KV。
4. **「goodput 和吞吐有什么区别？为什么 DistServe 要优化 goodput？」**
   要点：吞吐不管延迟；DistServe 的 per-GPU goodput 是在给定达标率（例如 90% 的请求同时满足 TTFT 和 TPOT）下每张卡能承受的最大请求速率。prefill 和 decode 放在一起会互相干扰：长 prefill 插进来让 decode 卡顿。DistServe 把两者分开，报告在满足延迟约束的前提下可服务 7.4 倍请求或 12.6 倍更紧的 SLO。
5. **「DeepSeek-V3 的 decode 为什么要铺到上百张卡的 EP？MLA 在这里起什么作用？」**
   要点：先说结论——为了把每个专家的 batch 聚起来。每个 routed expert 只分到 B × 8 / 256 = B/32 个 token，越过 BF16 ridge 要约 9,400 个并发 token，一张卡凑不出来；一台 8 卡机器也放不下 671 GB 的 FP8 权重。MLA 把 KV 压到约 68.6 KiB/token（Llama-3-70B 是 320 KiB），每卡能放更多序列；配合 DP attention，注意力 intensity 约 242，接近 compute-bound。代价：非 routed 的约 17B 参数每卡一份，冗余专家再占一些显存；EP 的 all-to-all 在 400G IB 上给出约 14.76 ms 的 TPOT 下限，MTP 用来进一步放大 batch。
6. **「用 Little's law 估一下：agent 平台每秒新增 2 个会话，每个会话 30 s 生成、60 s 工具等待，KV 全程 pin 住，平均上下文 32k，Llama-3-70B。要多少显存放 KV？」**
   要点：L = 2 × 90 = 180 个会话；每个 32k × 320 KiB ≈ 10.7 GB；均值约 1.9 TB，约等于 4–5 台 8 × H100 的全部 KV 空间。泊松到达时 p99 约 180 + 2.33√180 ≈ 211 个会话，要多留约 17%；上下文逐轮增长时平均占用约减半。其中 2/3 是在等工具，所以「工具期间卸载到 DRAM」或「按 TTL 丢弃」可以省掉大部分。

## 14. 延伸阅读

- [All About Transformer Inference（How To Scale Your Model）](https://jax-ml.github.io/scaling-book/inference/)：把本模块的每个公式推得更细，含 TPU / GPU 的临界 batch。
- [Efficiently Scaling Transformer Inference（arXiv:2211.05102）](https://arxiv.org/abs/2211.05102)：Google 在 PaLM 540B 上做推理的分区与 latency / MFU 取舍，multi-query attention 让上下文长度能扩到 32 倍。
- [Transformer Inference Arithmetic（kipply）](https://kipp.ly/transformer-inference-arithmetic/)：最早把「FLOPs / 字节」算账方法讲清楚的博客之一。
- [Roofline（CACM 2009）](https://dl.acm.org/doi/abs/10.1145/1498765.1498785)：roofline 模型原始论文。
- [LLM Inference Unveiled: Survey and Roofline Model Insights（arXiv:2402.16363）](https://arxiv.org/abs/2402.16363)：用 roofline 系统分析各种推理优化技术。
- [DistServe（arXiv:2401.09670）](https://arxiv.org/abs/2401.09670)：goodput、TTFT / TPOT SLO 与 PD 分离的出处。
- [DeepSeek-V3 技术报告（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437)：MLA 的 KV 缓存形式和推理部署单元。
- [Insights into DeepSeek-V3（arXiv:2505.09343）](https://arxiv.org/abs/2505.09343)：EP 通信下限 14.76 ms 的推导、MTP 的接受率、H800 的带宽口径。
- [Hydragen（arXiv:2402.05099）](https://arxiv.org/abs/2402.05099)与 [FlashInfer cascade inference](https://flashinfer.ai/2024/02/02/cascade-inference.html)：共享前缀时怎样让 KV 只读一次。
