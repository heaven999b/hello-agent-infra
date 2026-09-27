**中文** · [English](numbers-cheatsheet.en.md)

# 必须会心算的数字

> 面试和设计评审里，能当场说出数量级，比背出精确值更重要。本页每个数字都给出来源；标「推导」的由公式算出（公式写在旁边），标「据报道」的没有厂商官方规格页。
>
> 单位约定：TFLOPS = 1e12 FLOP/s；TB/s = 1e12 B/s；GB = 1e9 B；GiB = 2³⁰ B；KiB = 1,024 B。所有 GPU 算力都是 **dense**（不含 2:4 稀疏）。NVIDIA 规格页上 H100 / B200 的 Tensor Core 数字是带稀疏的，要除以 2。**互联带宽的厂商数字多是双向合计，估算「搬一次要多久」时用单向**（第 4 节）。

## 1. GPU 规格

| GPU | BF16 dense TFLOPS | FP8 dense TFLOPS | HBM | HBM 带宽 | ridge point（BF16，FLOPs/B） | NVLink 每卡（双向合计 / 单向） | 来源 |
|---|---|---|---|---|---|---|---|
| A100 80GB SXM | 312 | 不支持 | 80 GB HBM2e | 2.039 TB/s | ≈ 153 | 600 / 300 GB/s | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| H100 SXM | 989 | 1,979 | 80 GB HBM3 | 3.35 TB/s | ≈ 295 | 900 / 450 GB/s | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)（规格页 1,979 / 3,958 为稀疏值） |
| H800 SXM | 989 | 1,979 | 80 GB | 3.35 TB/s | ≈ 295 | 400 / 200 GB/s | NVLink 从 900 削到 400 GB/s、单向 200 GB/s 见 [arXiv:2505.09343](https://arxiv.org/abs/2505.09343)；算力与 HBM 按 H100 取值（NVIDIA 无公开英文规格页） |
| H20 | 148 | 296 | 96 GB HBM3 | 4.0 TB/s | ≈ 37 | 900 / 450 GB/s | **据报道**，NVIDIA 无公开规格页。全部五项数字见[腾讯云开发者社区转载的芯榜报道（2024-01-02）](https://cloud.tencent.com/developer/news/1284500)，算力按 dense 处理（报道未说明）；[阿里云 gn8v 规格族文档](https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families)写了 96 GB HBM3、4 TB/s、900 GB/s NVLink，但没写 GPU 型号，和 H20 的对应关系是推断 |
| B200 | 2,250 | 4,500 | 180 GB HBM3e | 8 TB/s | ≈ 281 | 1,800 / 900 GB/s | [HGX B200](https://www.nvidia.com/en-us/data-center/hgx/)（8 卡 BF16 36 PF 稀疏 → 每卡 dense 2.25 PF）、[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)（1,440 GB、64 TB/s 共 8 卡） |

心算要点：

- **Hopper / Blackwell 的 BF16 ridge ≈ 300 FLOPs/Byte**；FP8 算力翻倍，ridge 也翻倍（H100 FP8 ≈ 591）。注意翻倍的是 Tensor Core 的矩阵乘峰值，注意力的 KV 读取、反量化等不会跟着翻倍。
- ridge = 峰值 ÷ 带宽。decode 的 intensity 通常只有个位数到几十，所以**带宽才是 decode 的货币**。
- H20 的 ridge 只有约 37（推导，基于据报道的规格）：算力很弱、带宽很强，适合 decode，不适合 prefill；即使是只算几十个新 token 的增量 prefill，在 H20 上也可能仍是 compute-bound。

## 2. 模型结构与 KV cache

| 模型 | 层数 | query 头 / KV 头 | head_dim | 参数（总 / 激活） | KV 字节/token（BF16） | 32k 序列的 KV | 来源 |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 32 / 8 | 128 | 8.03B | 131,072 B = 128 KiB | 4 GiB | [Llama 3 论文 Table 3](https://arxiv.org/abs/2407.21783) |
| Llama-3-70B | 80 | 64 / 8 | 128 | 70.6B | 327,680 B = 320 KiB | 10 GiB | 同上 |
| Qwen3-32B | 64 | 64 / 8 | 128 | 32.8B | 262,144 B = 256 KiB | 8 GiB | [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json)、[model card](https://huggingface.co/Qwen/Qwen3-32B) |
| DeepSeek-V3 | 61 | 128 头，MLA 所有头共享一个潜在向量 | QK 192（128 + 64）/ V 128；吸收权重后 decode 在 576 / 512 维上做 | 671B / 37B（256 个 routed 专家，每 token 选 8 个） | 70,272 B ≈ 68.6 KiB | 2.14 GiB | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |

- 公式：GQA / MHA 为 2 × 层数 × KV 头 × head_dim × 字节；MLA 为 层数 × (kv_lora_rank + rope 维度) × 字节。
- 交叉核对：DeepSeek 的 [ISCA 论文（arXiv:2505.09343）](https://arxiv.org/abs/2505.09343) Table 1 给出每 token KV：DeepSeek-V3 70.272 KB、Qwen-2.5 72B 327.680 KB、LLaMA-3.1 405B 516.096 KB。
- 心算锚点：**GQA 大模型每 token 几百 KiB，MLA 几十 KiB；一条 128k 的 70B 序列 = 40 GiB = 半张 H100。**

## 3. 计算量与速度

| 量 | 典型值 | 怎么推出来 | 来源 |
|---|---|---|---|
| 前向 FLOPs / token（线性层） | 2N | 每个权重一次乘加；dense 的 N 是总参数，**MoE 的 N 是激活参数**（DeepSeek-V3 用 37B，不能用 671B）；输入 embedding 是查表，不算在 N 里；注意力另算 | [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) |
| 注意力 FLOPs / token / 层 | 4 · ctx · n_heads · head_dim | QKᵀ 和 AV 各 2 · ctx · d_attn | 推导 |
| dense decode 线性层 intensity | ≈ batch | 权重每步只读一次 | 推导 |
| MoE routed expert 的 intensity | ≈ B · k / E（DeepSeek-V3 为 B / 32） | 每个专家只分到 B·k/E 个 token；越过 BF16 ridge 要全局约 9,400 个 token；FP8 权重让 I 翻倍、FP8 ridge 也翻倍，仍约 9,400 | 推导，见[模块 01](../modules/01_inference_basics/README.md) 第 3.4 节 |
| decode 注意力 intensity | n_heads / n_kv_heads | 前提：decode、BF16 KV（FP8 时 ×2）、序列不共享前缀。MLA 配 DP attention ≈ 242，注意力按 TP 切 t 份时 ÷ t；共享前缀的部分 × 序列数 | 推导，见[模块 01](../modules/01_inference_basics/README.md) 第 3.3、3.5 节 |
| 增量 prefill / chunked prefill 块的下限 | 新 token 数 ≳ ridge（H100 BF16 约 300）才是 compute-bound | I ≈ 一次处理的 token 数 | 推导 |
| Llama-3-8B，1 × H100，batch 1，4k 上下文 | MBU 0.8 时 ≈ 5.8 ms/token（≈ 172 token/s）；满带宽 4.6 ms（≈ 215 token/s） | 15.5 GB（不含 embedding 表的权重 15.0 GB + KV 0.54 GB）÷ (3.35 TB/s × 0.8 或 × 1.0) | 推导 |
| Llama-3-70B，8 × H100，32k 上下文，batch 40 | TPOT ≈ 26.5 ms，≈ 1,500 token/s（不含 TP all-reduce） | 568 GB ÷ (26.8 TB/s × 0.8) | 推导 |
| Llama-3-8B 4k prompt 的 prefill，1 × H100 | ≈ 0.10 s | 6.2e13 FLOPs ÷ (989e12 × 0.6) | 推导 |
| Llama-3-8B 只算 128 个新 token（前面 4k 命中） | ≈ 6.2 ms，由读权重和 KV 决定 | 计算约 3.5 ms < 访存约 6.2 ms（按整份权重 + 4k 的 KV） | 推导 |
| Llama-3-70B 64k prompt 的 prefill，8 × H100 | ≈ 3.1 s | 线性 9.0e15 + 注意力 5.6e15 FLOPs | 推导 |
| DeepSeek 线上每节点（8 × H800） | prefill ≈ 73.7k token/s（含命中；按 56.3% 的全局命中率粗略扣除后，真正计算的约 32k token/s），decode ≈ 14.8k token/s | 线上统计；prefill 单元 4 节点 EP32，decode 单元 18 节点 EP144，不是同一种部署 | [DeepSeek 推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| DeepSeek 线上每用户输出速度 | 20–22 token/s；每个输出 token 平均 KV 长度 4,989 | 线上统计 | 同上 |
| 每个 DeepSeek decode 节点的并发序列 | ≈ 700 条 | Little's law：14.8k ÷ 21 | 推导 |
| MTP（多 token 预测） | 第二个 token 接受率 80–90%，生成 TPS 提升 1.8 倍 | 线上实测 | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| DeepSeek-V3 在 400G IB 上的 EP 通信决定的 TPOT 理论下限 | ≈ 14.76 ms（≈ 67 token/s） | 论文假设每设备约 32 个 token，见第 7 节 | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| PaLM 540B 推理 | 小 batch 29 ms/token（int8 权重）；大 batch 处理输入 76% MFU | 论文实测 | [Pope et al.（arXiv:2211.05102）](https://arxiv.org/abs/2211.05102) |

## 4. 互联带宽

估「搬一次要多久」时用**单向**列；厂商规格页上的 NVLink、PCIe 数字是双向合计。

| 链路 | 厂商口径 | 单向 | 有效 / 实测 | 来源 |
|---|---|---|---|---|
| NVLink 3（A100） | 600 GB/s 每卡，双向合计 | 300 GB/s | — | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| NVLink 4（H100） | 900 GB/s 每卡，双向合计；18 条链路 | 450 GB/s | — | [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) |
| H800 的 NVLink | 400 GB/s 每卡（合规削减） | 200 GB/s | 约 160 GB/s | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343)、[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) |
| NVLink 5（B200） | 1,800 GB/s 每卡；NVL72 合计 130 TB/s | 900 GB/s（ISCA 论文称 GB200 NVL72 单向 900 GB/s） | — | [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)、[arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| InfiniBand NDR（ConnectX-7） | 400 Gb/s 每端口 | 50 GB/s | 考虑小消息和延迟约 40 GB/s | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343)（H800 节点配 8 张 400G CX7 网卡）、[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| PCIe Gen4 x16 | 64 GB/s 双向合计 | 32 GB/s | 实际更低 | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) |
| PCIe Gen5 x16 | 128 GB/s 双向合计 | 64 GB/s | 实际更低；KV 卸载到 CPU 内存走这里 | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| NVLink-C2C（Grace Hopper） | 900 GB/s（官方只给总数） | — | — | [NVIDIA GH200](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)（官方称是 PCIe Gen5 的 7 倍） |

心算锚点（单向）：**HBM 3.35 TB/s ≫ NVLink 0.2–0.45 TB/s ≫ PCIe Gen5 64 GB/s ≈ IB 50 GB/s**。H800 节点内外约 4:1（200 : 50；按实测 160 : 50 约 3.2:1，这是 DeepSeek-V3 技术报告用的数）。一段 64k 的 Llama-3-70B KV（约 21.5 GB）走一张 IB 网卡约 0.43 s，走 8 张约 54 ms。

## 5. 存储层级：KV 可以放在哪

「本层带宽」是存储设备自己的带宽；「从 GPU 看」是 GPU 要拿到数据时的瓶颈链路，估算卸载、搬回用这一列。

| 层 | 典型容量 | 本层带宽 | 从 GPU 看（瓶颈链路） | 例子 | 来源 |
|---|---|---|---|---|---|
| HBM | 80–180 GB / 卡 | 3.35–8 TB/s / 卡 | 同左 | H100、B200 | 见第 1 节 |
| 主机 DRAM | 数百 GB / 插槽 | 数百 GB/s / 插槽（Grace：480 GB 配置 384 GB/s，120 / 240 GB 配置 512 GB/s） | 每卡经 PCIe Gen5 x16 单向约 64 GB/s（理论值，实际更低；一个插槽的内存带宽由多张卡分享） | Grace CPU | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)、[NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| 本地 NVMe SSD | TB 级 / 盘 | 约 13 GB/s / 盘（PCIe 5.0 顺序读） | 受盘数和 PCIe 通道限制，多盘可以聚合 | Samsung PM1743 | [Samsung Newsroom](https://news.samsung.com/global/samsung-develops-high-performance-pcie-5-0-ssd-for-enterprise-servers) |
| 分布式 SSD 存储 | PB 级 | 180 个存储节点聚合读约 6.6 TiB/s | 受客户端网卡限制：KVCache 读取峰值约 40 GiB/s（客户端 1 × 400 Gbps 网卡），分给 8 卡约 5 GB/s / 卡 | DeepSeek 3FS | [3FS](https://github.com/deepseek-ai/3FS) |

心算锚点：**从 GPU 看，HBM → 主机 DRAM 约掉 50 倍**（3.35 TB/s 对 64 GB/s），再到远端 SSD 按卡分摊又掉一个数量级左右；容量则每层涨一到两个数量级。DeepSeek 线上 56.3% 的输入 token 命中的是**磁盘上**的 KV cache：在它的规模下，DeepSeek 选择了把 KV 存到 SSD 上复用，而不是每次重算 prefill。

## 6. 排队与缓存经济学

| 量 | 典型值 | 来源 |
|---|---|---|
| Little's law | 系统内平均数量 L = 到达率 λ × 平均停留时间 W | [Little 1961](https://pubsonline.informs.org/doi/abs/10.1287/opre.9.3.383) |
| 按分位数规划 | 泊松到达时，系统内数量服从均值为 L 的泊松分布（M/G/∞ 稳态），p99 ≈ L + 2.33√L；L = 180 时约 211，多约 17% | 推导，见[模块 01](../modules/01_inference_basics/README.md) 第 8 节 |
| 上下文逐轮增长时的平均 KV | 线性增长时约为最终长度的一半 | 推导 |
| agent 输入:输出 token | 约 100:1 | [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) |
| 缓存命中输入的价格 | Anthropic：缓存读取 0.1× 基础输入价；5 分钟缓存写入 1.25×、1 小时 2× | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| OpenAI 缓存折扣 | 缓存输入最高便宜 90% | [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) |
| Anthropic 默认缓存 TTL | 5 分钟（可选 1 小时） | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| DeepSeek 线上 KV cache 命中率 | 24 小时内 56.3% 的输入 token 命中磁盘 KV cache | [DeepSeek 推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| 多 agent 的 token 放大 | agent 约 4× 聊天，多 agent 约 15× | [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system) |

三个判定式（都是推导，适用边界写在括号里）：

1. **存还是重算**：从某一层加载 KV 比重算快 ⟺ 这一层到 GPU 的带宽 > 盈亏平衡带宽 B\* = 每 token KV 字节 × 单卡 prefill 吞吐（token/s）。Llama-3-8B 在 H100 上约 131,072 B × 3.7 万 token/s ≈ 4.8 GB/s；DeepSeek-V3（MLA + 大激活参数）约 0.56 GB/s / 卡。只算线性层、不含固定延迟；推导和更多模型见[模块 05](../modules/05_cache_prefill/README.md) 第 4.2 节。
2. **放哪一层**：在带宽 > B\* 的层里，选容量够、单位成本最低的那层。按第 5 节，主机 DRAM（每卡 64 GB/s）几乎永远值得；远端 SSD（每卡约 5 GB/s）对 MLA 大模型余量很足，对 8B 这类小 dense 模型基本打平。
3. **工具等待期间留多久**：假设实例并发受 KV 显存限制，占住整机 KV 容量的份额 s、占 t 秒，相当于损失 t × s 个「整机秒」；重算要 r 秒整机时间。t × s ≤ r 就留，否则卸载或丢弃。Llama-3-70B、8 × H100、64k 上下文，s ≈ 4.9%、r ≈ 3.1 s，门槛约 62 s。工具时延是随机的，TTL 应按分布选（[Continuum](https://arxiv.org/abs/2511.02230)）。练习见[模块 01](../modules/01_inference_basics/README.md) 练习 (d)。

## 7. 通信心算

| 通信 | 每步的量 | 例子 | 来源 |
|---|---|---|---|
| TP all-reduce（decode） | 每层 2 次（注意力后、MLP 后），每次消息 = B × d_model × 字节；ring all-reduce 每卡收发约 2(n − 1)/n × 消息 | Llama-3-70B、TP8、B = 40：每步 160 次，每卡合计约 180 MB，NVLink 单向 450 GB/s 下约 0.4 ms；带宽不是问题，160 次的固定延迟会累加 | 推导 |
| EP dispatch + combine（MoE） | 每 token 每层 (dispatch 字节 + combine 字节) × (k + 共享专家数) × hidden | DeepSeek-V3：(1 字节 + 2 字节) × 9 × 7K ≈ 189 KB；每设备 32 个 token、50 GB/s IB 时每层 120.96 µs，双 micro-batch 重叠下每层 241.92 µs，61 层合计 14.76 ms | [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) |
| PD 分离搬 KV | prompt 长度 × 每 token KV 字节（外加 block 元数据） | 64k：Llama-3-70B 约 21.5 GB（单张 IB 0.43 s），DeepSeek-V3 约 4.6 GB（0.09 s） | 推导 |

## 8. 一分钟心算模板

| 问题 | 模板 | 适用边界 |
|---|---|---|
| decode 一步多久？ | (权重字节 + B × ctx × KV/token) ÷ (卡数 × HBM 带宽 × 0.8) | memory-bound、batch 内上下文长度相近时的下界；上下文长短不一时用 Σ ctx_i；另加 TP / EP 通信和固定开销 |
| 最多多少并发？ | (卡数 × HBM × 0.9 − 权重字节) ÷ (ctx × KV/token) | 聚合上界；真实部署按每卡算：min over 卡 ((HBM − 本卡权重 − 预留) ÷ 本卡每请求 KV)，DP attention、冗余专家、TP 超过 KV 头数都会让它变小 |
| TTFT 多少？ | 排队 + max((2N × n + 2 × n² × d_attn × 层数) ÷ (卡数 × 峰值 × 0.6)，(权重 + 整段 KV) ÷ (卡数 × 带宽 × 0.8)) | 下界；另加缓存加载、通信、chunked prefill 的交错等待 |
| 是 compute-bound 吗？ | FLOPs ÷ 字节，和 ridge（Hopper BF16 ≈ 300）比 | 必要不充分：小矩阵很难达到峰值 MFU |
| KV 搬一次多久？ | ctx × KV/token ÷ 链路单向带宽（PCIe Gen5 ≈ 64 GB/s，IB ≈ 50 GB/s / 端口） | 可以按层流水藏在计算后面 |
| 要多少显存放 pin 住的 KV？ | Little's law：到达率 × (生成时间 + 工具等待) × ctx × KV/token | 均值；p99 按 L + 2.33√L 留余量 |

## 9. 心算练习

1. 8 × H100、Llama-3-70B、batch 32、每条 16k 上下文，TPOT 大约多少？
   <details><summary>答案</summary>每步读约 139 GB 权重（不含 embedding 表）+ 32 × 16,384 × 327,680 B ≈ 172 GB 的 KV ≈ 311 GB；8 卡 26.8 TB/s，按 MBU 0.8 约 14.5 ms。计算只要约 1.2 ms，所以是 memory-bound。这是下界，没算 TP 的 160 次 all-reduce。</details>
2. 把一段 64k 的 Llama-3-70B KV 从 prefill 池搬到 decode 池，用 1 张和 8 张 400G 网卡各要多久？占 prefill 时间的多少？
   <details><summary>答案</summary>KV 约 21.5 GB。1 张网卡约 0.43 s，8 张约 54 ms；同一段 prefill 在 8 × H100 上约 3.1 s，分别占 14% 和 1.7%，按层流水基本能藏住。如果前缀命中 90%，只需算约 0.41 s，传输就和计算一样长了。</details>
3. agent 平台每秒新来 20 个会话，平均生成 10 s、平均工具等待 30 s，上下文 8k，Llama-3-8B，KV 全程 pin 住。均值和 p99 各要多少显存？
   <details><summary>答案</summary>L = 20 × 40 = 800 个会话；每个 8,192 × 131,072 B ≈ 1.07 GB；均值约 859 GB，约 15 张 H100 装下 8B 后的 KV 空间（每张约 55.9 GB）。泊松到达时 p99 ≈ 800 + 2.33 × √800 ≈ 866 个会话，多约 8%。上下文逐轮增长时平均占用约减半。</details>
4. DeepSeek-V3 decode 要让 routed expert 的 GEMM 越过 BF16 ridge，全局每步要多少个 token？如果每条序列每步用 MTP 多验证 1 个 token 呢？
   <details><summary>答案</summary>每个专家分到 B × 8 / 256 = B / 32 个 token，要 B ≈ 295 × 32 ≈ 9,400 个 token；每条序列每步 2 个 token 时，约 4,700 条并发序列就够了。这就是 ISCA 论文说 MTP「增大推理 batch，对提高 EP 计算强度至关重要」的意思。</details>
