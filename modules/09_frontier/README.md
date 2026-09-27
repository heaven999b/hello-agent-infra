**中文** · [English](README.en.md)

# 模块 09：前沿与开放问题

> 🕐 建议时长：0.5 小时（只计「30 分钟核心」：第 0–2 节 + 第 12 节前 3 题；标了「选读」的部分不计入 20 小时） ｜ 🎯 学完你能：用四段话讲清 2025–2026 年 agent × infra 的四条前沿线索；把两个与你方向最相关的开放问题讲到「已有答案是什么、真正没解决的是什么、第一张图怎么做」 ｜ 🧪 实验：无（纯讲义） ｜ 📖 必读一手资料：[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)（读摘要与第 3 节）；[Autellix](https://arxiv.org/abs/2502.13965) 已在 [模块 04](../04_harness_workloads/README.md) 出现，这里只需从「程序级服务化」的角度重读摘要

**怎么用这 30 分钟**：约 10 分钟读第 2.1–2.4 节的四条线索，约 15 分钟读第 2.5–2.6 节的两个精选问题，5 分钟做第 12 节前 3 题。第 3 节以后是选读，留到你做研究选题时查。

本模块只收录作者打开核实过的论文与报告。文中所有「提升 X 倍」都是**原论文在其自身设置下报告的数字**，不同论文之间不可直接比较。时间截至 2026 年 9 月。

## 0. 一句话

**推理引擎和训练系统正在从「服务一个请求」变成「服务一个程序」：调度单位从请求变成会话/程序，缓存策略开始读 harness（模型外面那一圈 agent 代码）的信号，RL 系统开始把整个 agent harness 当作黑盒来做 rollout（按当前策略采样轨迹）；而递归自我改进（RSI）把「训练—部署—评估」这个环闭合起来，让评估和数据生成本身变成一等公民的负载。**

## 1. 大厂为什么在意

一个具体场景：一个编码 agent 平台，每个用户任务是一个 20–100 轮的 agent 会话（读仓库、跑测试、改代码），每轮之间夹着几秒到几分钟的工具调用。同一套模型还要做 agentic RL：每个训练 step 要跑几千条这样的轨迹。

- **推理侧**：引擎只看到一个个孤立的请求。工具调用一返回，上一轮的 KV cache（注意力的键值缓存）可能已经被 LRU 淘汰，要重新 prefill（把整段上下文重新算一遍前向）。InferCept 在其负载下测到，这种「已经算过的上下文被重算」占了模型前向总时间的 37–40%（[InferCept](https://arxiv.org/abs/2402.01869)）。
- **训练侧**：OpenRLHF 的作者称推理（生成）阶段常占 RLHF / RLVR 总运行时间的 90% 以上，这是一句笼统描述，没有给测量设置（[OpenRLHF](https://arxiv.org/abs/2405.11143)；APRIL 也引用了这一说法）。长尾轨迹拖住整批，环境（沙箱、浏览器、编译器）是 CPU 密集且不稳定的。
- **闭环侧**：模型/harness 开始自己改自己（DGM 在 SWE-bench 上从 20.0% 自我改进到 50.0%，[DGM](https://arxiv.org/abs/2505.22954)），每一次「改进」都要跑一整套评估来验证。评估从「发布前做一次」变成了「内循环里每一步都做」。

## 2. 30 分钟核心

### 2.1 线索一：把 agent 当「程序」来服务

调度与缓存的基本单位正在从单个请求升级为程序或会话：SGLang 让应用用 `fork` / `gen` 写成程序并在程序间复用前缀 KV（[SGLang](https://arxiv.org/abs/2312.07104)），Autellix 按程序此前所有调用的累计运行时间排优先级，在其负载下同等延迟的程序吞吐提升 4–15×（[Autellix](https://arxiv.org/abs/2502.13965)）。数量级：一个 50 轮会话、共享前缀 20k token、每轮外部追加 2k token（工具结果、用户输入，缓存命中时只需对它们做 append-prefill；模型自己生成的 token 在 decode 时已写入 KV，这里忽略）。缓存全丢时第 k 轮要 prefill 20k + 2k·k，50 轮累计 ≈ 1.0M + 2.55M ≈ 3.6M token；全命中时只要 20k + 50×2k = 0.12M，差约 30 倍。程序级调度的价值就在于把系统推向后者，代价是要处理会话粘性带来的负载不均（[模块 05](../05_cache_prefill/README.md)）。

### 2.2 线索二：harness 感知的推理引擎

harness 手里有引擎看不到的信号：工具大概跑多久、下一步调哪个 agent、什么时候要压缩上下文。Continuum 用该工具历史时长的分布决定工具调用期间 KV 保留多久（[Continuum](https://arxiv.org/abs/2511.02230)）；Ask the Tool, Don't Guess 让工具在运行中上报进度，替代调用前的时长猜测（[Ask the Tool](https://arxiv.org/abs/2609.18849)）；KVFlow、CacheScout、TOPAS 用声明的或在线学到的 workflow 图管理 KV。工业界已经有现成接口：NVIDIA Dynamo 的 `nvext.agent_hints`（`priority`、`strict_priority`、`osl`、`speculative_prefill`）以及 SGLang 的按优先级淘汰 radix 叶子，见 [模块 05 第 7 节](../05_cache_prefill/README.md#7-协同设计接口提案harness-应该告诉引擎什么)。剩下的问题是：在这些字段之上，再加哪个 hint 的边际收益最大。

### 2.3 线索三：大规模 agentic RL 系统

演进路线是「同步 → partial rollout → 完全异步 → 解耦/异构 → rollout-as-a-service」，每一步都在消掉上一步的气泡，也都用一点 off-policy 偏差换利用率（数学骨架见 [模块 06 第 5 节](../06_long_tail_async/README.md#5-代价一staleness-与-off-policy-修正数学)）。最新一步是把整个 harness 当黑盒：Polar 代理 LLM API 调用、事后重建 token 级轨迹，但收益强烈依赖 harness：同一个 Qwen3.5-4B 在 SWE-bench Verified 上，用 Codex harness 提升 22.6 分，用 Qwen Code 只提升 0.6 分（[Polar](https://arxiv.org/abs/2605.24220)）。瓶颈正在从 GPU 算力移向「环境 + 长尾 + staleness + 轨迹保真」：DeepSeek 的沙箱平台 DSec 在生产中支撑超过 38 万个并发沙箱（[DSec](https://arxiv.org/abs/2609.22978)）。

### 2.4 线索四：RSI 对基础设施的含义

自我改进回路改的可以是数据、奖励、课程、权重或 harness 代码（DGM、AlphaEvolve 等，第 6 节）。对基础设施有三个含义：

1. **一切都要版本化**：KV 是某个权重版本的中间结果，缓存键必须带上版本。权重一更新，是立即清空还是保留旧版本 KV 继续用，是一个策略选择，已有两种答案：AReaL 中断在途生成、丢弃 KV 后重算；PipelineRL 带着旧 KV 继续生成，在 7B 模型上实测 KL 只比重算略高（[模块 06 第 4.6 节](../06_long_tail_async/README.md#46-in-flight-权重更新连-kv-都不重算)）。前缀缓存的 key 默认是 token 序列的 hash、不含权重版本，更新后不清缓存就会静默复用旧 KV，SGLang、vLLM 对此提供的开关见 [模块 06 第 7.5 节](../06_long_tail_async/README.md#75-权重版本--kv--前缀缓存)。
2. **数据管线自己生成**：DeepSeek-V3.2 的 agentic 任务合成管线生成了 1,827 个环境和 85,000 条 prompt（[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)），「造环境 + 造任务 + 过滤」本身就是大规模推理负载。
3. **评估进入内循环**：每个候选都要跑基准；Huxley-Gödel Machine 还指出基准分不等于「继续自我改进的潜力」，按后代表现估计潜力来分配预算，用更少的 CPU 小时超过了此前的方法（[HGM](https://arxiv.org/abs/2510.21614)）。

### 2.5 精选问题 A：反事实信用的系统代价（多 agent 奖励 × 长尾异步 × prefill）

- **已有答案**：单 agent 场景里「在哪分叉、怎么摊销前缀」已被直接研究过。TreeRL 在相同的生成 token 预算下从高不确定性的中间步分叉，而不是随机分叉（[TreeRL](https://arxiv.org/abs/2506.11902)）；ARPO 在工具调用之后按熵自适应分支（[ARPO](https://arxiv.org/abs/2507.19849)）；FR3E 找出高不确定性的决策点做定向 rollout（[FR3E](https://arxiv.org/abs/2507.07017)）；TreePO 靠摊销公共前缀在采样设计上节省 22%–43% 的 GPU 小时（[TreePO](https://arxiv.org/abs/2508.17445)）；Tree-GRPO 共享前缀，在固定 token 或工具调用预算内得到更多 rollout（[Tree-GRPO](https://arxiv.org/abs/2509.21240)）。多 agent 场景里，AT-GRPO 每轮为每个 agent 分支（[AT-GRPO](https://arxiv.org/abs/2510.11062)），C3 靠 checkpoint 恢复、不重新生成历史，比 MAPPO 少用 32% token（[C3](https://arxiv.org/abs/2603.06859)）。
- **真正开放**：
  1. 多 agent 下要选的是「哪个 agent × 哪一轮」，而一个样本的信用要等原始轨迹和它的全部反事实分支都跑完才能算：这是一个落在长尾关键路径上的**组屏障**；
  2. 编码 agent 的环境状态（文件系统、进程）不在文本里，分支需要沙箱快照或 fork；DSec 提供把沙箱做增量磁盘快照、再恢复成新沙箱的接口，但论文没有给出快照延迟，分支数又会乘上沙箱并发（[模块 03 第 5.3 节](../03_deepseek_stack/README.md#53-dsec智能体-rl-的沙箱平台)）；
  3. 账要分开算：KV fork 省的是决策点之前的 **prefill**，checkpoint 恢复（C3）省的是决策点之前的 **decode**，分叉之后追加的工具结果仍然要 prefill，而且前缀要命中还得把分支路由到同一实例。
- **第一张图**：在 [模块 07](../07_multi_agent_credit/README.md) 的 team task 上固定额外评估预算，只给 n 个 agent 中的 k 个算 difference reward，其余用组内基线，画「梯度信噪比 vs k」，再用 `simulate_rollout` 把 k 换算成有效吞吐。现在就能做；如果要「选决策点」，需要库新增一个多步的 team task。

### 2.6 精选问题 B：权重在变、前缀在共享：版本化的多项目缓存（多项目 cache × 长尾异步 × prefill）

- **已有答案**：在途请求跨过权重更新时怎么办，已有两种做法和实测（上面 2.4 第 1 条）；多轮 agent 两轮之间换了权重也会形成「新权重 + 旧 KV」，[模块 06 第 7.5 节](../06_long_tail_async/README.md#75-权重版本--kv--前缀缓存)列出了续写时 KV 的四种处理及代价；多租户隔离有 vLLM 的 `cache_salt` 和时序侧信道的研究（[模块 05 第 2.4 节](../05_cache_prefill/README.md#24-多租户共享还是隔离)）。
- **真正开放**：一个项目的共享前缀（system prompt + 工具定义 + 仓库上下文）会被很多**新**请求反复使用。权重更新后，新请求能否继续复用上一版本的前缀 KV？复用几个版本？这条「命中率 vs KL」曲线还没有人画过；再叠加多项目争抢容量、按隔离域切分缓存，就是读者「多项目 cache」方向的核心问题。
- **第一张图**：命中率这一侧现在就能做。用 `generate_sessions` + `PrefixCache` 跑两条边界：每次权重更新（周期 = `train_step_s`）都新建一个空缓存（立即清空），和从不清空（完全复用陈旧 KV）。两条曲线之间的面积，就是「允许复用陈旧前缀」最多能换回的 prefill。KL 那一侧需要一个小模型做真实生成。

## 3. 选读 A：程序级服务化的细节

| 系统 | 把什么作为一等公民 | 关键机制 | 原文报告的效果 |
|---|---|---|---|
| [SGLang](https://arxiv.org/abs/2312.07104)（2023） | LM program | 前端原语 `fork` / `join` / `gen`；运行时 RadixAttention 在程序间复用前缀 KV | 吞吐最高 6.4× |
| [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)（OSDI 2024） | Semantic Variable（语义变量） | 应用把多次 LLM 调用之间的数据依赖暴露给服务端，服务端做数据流分析 | 最高一个数量级 |
| [InferCept](https://arxiv.org/abs/2402.01869)（2024） | 被工具调用「打断」的生成 | 工具调用期间对 KV 在「保留 / 换出到 CPU / 丢弃后重算」之间选择，目标是最小化显存浪费（min-waste） | 吞吐 1.6–2× |
| [Autellix](https://arxiv.org/abs/2502.13965)（2025） | 整个 agent 程序 | 截获程序的 LLM 调用，按程序**此前所有调用的累计运行时间**（PLAS，program-level attained service）抢占与排优先级，缓解程序级队头阻塞（head-of-line blocking） | 同等延迟下程序吞吐 4–15×（对比 vLLM 等） |
| [ConServe](https://arxiv.org/abs/2606.01839)（2026） | 整个会话 | 调度单位从 turn 提升到 conversation：首轮长 prefill 送到高吞吐 prefill 节点，KV 只做这一次跨节点迁移；之后各轮的 append-prefill 和 decode 都在固定的那台 decode 节点上本地执行 | p95 首个有效 token 延迟 −51% |

**取舍**：

- 程序级「已服务越多、优先级越低」属于 LAS（least attained service）一类策略。在程序规模重尾、可以抢占时，它往往能改善平均完成时间；但在 LLM 服务里 prefill 通常不可细粒度抢占，还要考虑 cache 局部性、工具等待和优先级分级，收益不是无条件的，还要防止长程序饿死（老化或配额）。
- 会话粘性（ConServe、cache affinity 路由）提升命中率，但会造成实例间负载不均，这正是 [模块 05](../05_cache_prefill/README.md) 里 `affinity_with_load_cap` 要平衡的东西。
- 程序信息从哪来？要么由应用显式声明（Parrot、SGLang 前端），要么由服务端推断（Autellix 截获调用）。显式声明更准，但要改 harness；推断不侵入，但会猜错。

## 4. 选读 B：harness 感知引擎的细节

| 引擎读取的 harness 信号 | 代表工作 | 用它做什么 | 原文报告的效果 |
|---|---|---|---|
| 该工具历史调用时长的分布（经验 CDF） | [Continuum](https://arxiv.org/abs/2511.02230) | 工具调用期间把 KV pin 在显存里，按下面的目标函数选 TTL，超时自动释放；配合程序级 FCFS | 平均任务完成时间（JCT）改善 8× 以上 |
| 工具**运行中**上报的进度 | [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849) | 论证调用开始前的任何时长估计都可能连排序都排不对；让工具在运行中显式上报「还剩多少」，引擎据此决定 KV 去留 | 工具返回后 p90 TTFT −20.7%（对比 LRU），接近 oracle |
| 声明的 workflow 图 | [KVFlow](https://arxiv.org/abs/2507.07400) | 把 agent 调度抽象为 Agent Step Graph，按「距离下次执行还有几步」淘汰 KV，并提前从 CPU 预取 | 对比 SGLang 分层 radix cache 最高 1.83× / 2.19× |
| 在线学到的 agent 转移 | [CacheScout](https://arxiv.org/abs/2608.14624) | 不需要预先声明 workflow，在线学习 agent 之间的执行转移，用来指导淘汰和预取 | 命中率 +10–18 个百分点，平均 TTFT −18–45% |
| 工作流 DAG + 前缀状态 | [TOPAS](https://arxiv.org/abs/2608.25523) | 联合决定「留哪些 agent 的前缀」和「先跑哪些请求」，权衡关键路径与前缀复用 | 合成负载上 mean/p99 JCT 最高 −39.8% / −49.4% |

**TTL 的目标函数，写严谨一点**：设工具时长 T 是随机变量，B 是「工具返回时 KV 还在」能省下的代价（重算或重载 + 被驱逐后的排队），c 是 pin 住这份 KV 每秒的机会成本。保留到 τ 的期望净收益是

J(τ) = P(T ≤ τ) · B − c · E[min(T, τ)]

对 τ 求导：J'(τ) = f(τ)·B − c·P(T > τ)。所以**只要「工具在下一刻返回」的风险率 h(τ) = f(τ)/P(T > τ) 大于 c/B，就值得继续保留**。Continuum 论文写的是 argmax_τ P(τ, f)×Benefit − Cost(τ)，把占用代价近似为随 τ 线性增长（即用 τ 代替 min(T, τ)，是一个上界）。一个推论：若工具时长服从指数分布（风险率为常数 1/E[T]），最优 TTL 要么是 0 要么是无穷——**当且仅当 B > c·E[T] 时才 pin**。第 12 节第 4 题会让你代入数字。

**hint 接口的宽窄**：

- 最窄：什么都不传，引擎靠 LRU 和历史猜。
- 中间：每次请求带几个字段。这一档已有工业实现：Dynamo 的 `nvext.agent_hints`（`priority`、`strict_priority`、`osl` 预计输出长度、`speculative_prefill` 本轮结束后预先 prefill 下一轮前缀）加会话 ID，以及 SGLang 的 `--radix-eviction-policy priority`；[模块 05 第 7 节](../05_cache_prefill/README.md#7-协同设计接口提案harness-应该告诉引擎什么)还提出了 `tool_eta`、fork 组、可丢弃段等新增字段。
- 最宽：harness 把整个程序（DAG）交给引擎（SGLang 前端、Parrot）。

越宽收益越大，但耦合越重、越难跨 harness 复用，也更依赖 harness 说实话。[模块 04](../04_harness_workloads/README.md) 讲的「上下文改写导致前缀失效」就是一个 harness 最清楚、引擎最不清楚的信号。

**把这些放进一个成本模型**：任务完成时间 ≈ 排队 + prefill(未命中 token) + decode + 工具等待；显存 = 活跃 KV + pin 住的 KV + 已换出的 KV。TTL、会话粘性、LAS 优先级、按项目配额都是这个模型里的决策变量。模块 05 给出了统一的 TTFT 代价估计（[第 3.2 节](../05_cache_prefill/README.md#32-统一代价估计-ttft)）和「重算 vs 加载」的盈亏公式（[第 4.2 节](../05_cache_prefill/README.md#42-重算-vs-加载盈亏平衡公式)），[模块 08](../08_system_design/README.md) 把它们用在完整的系统设计题里。

## 5. 选读 C：agentic RL 系统的细节

```
同步 colocate（verl/HybridFlow）
   └─ 痛：批内最长轨迹拖住整批 → 长尾气泡
partial rollout（Kimi k1.5、APRIL）
   └─ 未完成的轨迹留到下一轮续写；痛：一条轨迹跨多个策略版本
完全异步（AReaL、Asynchronous RLHF）
   └─ 生成与训练解耦；痛：staleness（数据陈旧度），需要控制最大版本差 + off-policy 修正
解耦 / 异构（StreamRL、RollArt）
   └─ prefill、decode、环境、奖励分别放到最合适的硬件；痛：跨阶段同步与调度
rollout-as-a-service / harness 黑盒（ProRL Agent → Polar）
   └─ 不再把 harness 移植进 RL 环境接口，而是代理 LLM API 调用、事后重建 token 级轨迹
```

| 系统 | 关键点 | 原文报告 |
|---|---|---|
| [HybridFlow / verl](https://arxiv.org/abs/2409.19256) | single-controller + multi-controller 混合；3D-HybridEngine 在训练和生成间重分片 actor | 吞吐 1.53–20.57× |
| [AReaL](https://arxiv.org/abs/2505.24298) | 完全异步；可中断的 rollout worker（换权重时丢弃 KV、用新权重重算后续写）；staleness-enhanced PPO | 同卡数下最高 2.77× 训练加速 |
| [StreamRL](https://arxiv.org/abs/2504.15930) | 解耦架构；stream generation 消除 pipeline bubble；输出长度 ranker 识别长尾 | 吞吐最高 2.66× |
| [Seer](https://arxiv.org/abs/2511.14617) | 观察到同一 prompt 的多个样本输出长度相似；divided rollout + context-aware 调度 + 分组投机解码 | rollout 吞吐最高 2.04×，长尾延迟 −72–94% |
| [RollArt](https://arxiv.org/abs/2512.22560) | 多任务 agentic RL；prefill 重的任务去算力型 GPU、decode 重的去带宽型 GPU、环境去 CPU 集群；奖励走 serverless | 训练时间缩短 1.31–2.05×；在 3,000+ GPU 集群上训练了数千亿参数的 MoE |
| [ProRL Agent](https://arxiv.org/abs/2603.18815) → [Polar](https://arxiv.org/abs/2605.24220) | ProRL Agent 把 rollout 生命周期做成 API 服务；Polar 在其基础上把任意 harness 当黑盒，代理 LLM API 调用、重建 token-faithful 轨迹 | Qwen3.5-4B 在 SWE-bench Verified 上提升 0.6–22.6 分，视 harness 而定（Codex 22.6、Claude Code 4.8、Qwen Code 0.6、Pi 6.2） |

厂商技术报告里公开的规模锚点：

- **Kimi K2**：从 GitHub 抓取 3000+ 真实 MCP 工具，合成 20,000+ 工具；RL 用 colocate 架构，靠大量并发 rollout 摊薄昂贵交互的延迟，并用 partial rollout 防止长尾未完成任务阻塞整个 rollout（[Kimi K2](https://arxiv.org/abs/2507.20534)）。
- **DeepSeek-V3.2**：agentic 任务合成管线生成 1,827 个环境、85,000 条复杂 prompt；后训练计算预算超过预训练成本的 10%（[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)）。
- **DeepSeek DSec**：单个规模单元约 160 个节点、每天约 300 万个沙箱，生产中超过 38 万个并发沙箱、每秒超过 5,000 次创建（[DSec](https://arxiv.org/abs/2609.22978)）。
- **GLM 系列**：由 [slime](https://github.com/THUDM/slime)（Megatron 训练 + SGLang rollout + data buffer）支撑后训练。

**跨版本轨迹的最小数学骨架**（完整推导见 [模块 06 第 5 节](../06_long_tail_async/README.md#5-代价一staleness-与-off-policy-修正数学)）：设一条 800 token 的轨迹，前 400 个 token 由版本 v10 生成，后 400 个由 v11 生成，训练时策略是 v12。每个 token 要存四样东西：生成它的版本号、**采样时**的 logprob（含温度与 top-p 截断）、奖励 / 优势、loss mask。重要性采样只要求分母是真正用来采样的概率，所以按 token 记录的 logprob 就足以写出这条混合版本轨迹的精确似然比；AReaL 的 decoupled PPO 再把「信任域中心」（最近的策略）和「行为策略」（各段各自的版本）拆开。实际用逐 token 比率而不用序列级乘积，原因是乘积的方差随长度指数增长（第 12 节第 6 题）。

## 6. 选读 D：RSI 的细节

RSI（recursive self-improvement，递归自我改进）在 2026 年已经有了专门的学术场合：[ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://recursive-workshop.github.io/)，它把问题拆成「改什么（参数、世界模型、记忆、工具与技能、架构）、何时改（episode 内、测试时、部署后）、怎么改（奖励/价值学习、模仿、进化搜索）」以及安全与评估。

| 改什么 | 代表工作 | 闭环里的「评估器」 |
|---|---|---|
| 训练数据 / 推理轨迹 | [STaR](https://arxiv.org/abs/2203.14465)：用自己答对的推理链再训练自己 | 答案是否正确 |
| 奖励本身 | [Self-Rewarding LM](https://arxiv.org/abs/2401.10020)：模型用 LLM-as-a-Judge 给自己打分，迭代 DPO | 自己 |
| 课程（出题） | [Absolute Zero](https://arxiv.org/abs/2505.03335)：同一个模型出题又解题，零外部数据 | 代码执行器 |
| 权重更新指令 | [SEAL](https://arxiv.org/abs/2506.10943)：模型生成自己的微调数据和更新指令，用更新后模型的下游表现作 RL 奖励 | 下游任务 |
| agent 代码 / harness | [DGM](https://arxiv.org/abs/2505.22954)、[HGM](https://arxiv.org/abs/2510.21614)、[ModularRSI](https://arxiv.org/abs/2609.14857) | 编码基准（SWE-bench 等）；HGM 改用后代表现估计的「自我改进潜力」 |
| 算法与基础设施代码 | [AlphaEvolve](https://arxiv.org/abs/2506.13131)：进化式编码 agent，改进了 Google 数据中心调度，并加速了驱动它自己的 LLM 的训练 | 一个或多个自动评估器 |

**对基础设施的三个含义（展开）**：

1. **版本化语义，而不是「一更新就全失效」**：KV、路由表、评估基线都带模型版本号。强制所有 worker 立即切到新版本，等价于清空缓存，prefix cache 的寿命就被权重更新周期截断；保留多个版本则命中率取决于流量怎么在版本间分配、旧版本保留多久，代价是多版本权重与 KV 占用的显存。在途请求已有两种答案（AReaL 重算、PipelineRL 复用陈旧 KV），跨请求的共享前缀还没有（第 2.6 节）。
2. **自生成数据管线**：Kimi K2 和 DeepSeek-V3.2 的合成管线都说明，「造环境 + 造任务 + 过滤」本身就是一个大规模推理负载，瓶颈常在沙箱/执行环境与评审（LLM judge 或执行器），而不在训练。
3. **评估闭环**：DGM 这类系统每产生一个候选都要跑基准；[SEAGym](https://arxiv.org/abs/2606.17546) 指出频繁更新可能不提升 held-out 表现、中间好快照可能后来崩溃；[ModularRSI](https://arxiv.org/abs/2609.14857) 用与评测集不相交的 2,000 个进化任务来避免「在基准上进化 = 过拟合基准」。于是评估需要像训练一样被调度：提前淘汰差候选、方差感知的晋级、回归门禁和回滚。

## 7. 选读 E：开放问题全表

标签：**A** 长尾异步 · **B** 多 agent 奖励 · **C** 多项目 cache · **D** prefill · **E** efficient system · **F** RSI。「抓手」一列注明现在能不能用本项目的代码做。

| # | 真正开放的部分 | 方向 | 已有答案（锚点） | 最小实验的抓手 |
|---|---|---|---|---|
| 1 | 长程 agent 轨迹（10⁴–10⁵ token、跨几十个版本段）上，逐 token 截断 IS 的阈值怎么随段数与长度自适应？有效样本量（ESS）何时崩？ | A B | 跨版本分段轨迹用逐 token 行为概率 + decoupled PPO（AReaL，见 [模块 06 第 5.3 节](../06_long_tail_async/README.md#53-decoupled-ppo把信任域中心和行为策略拆开)）；PipelineRL 的 token lag | 现在可做：用 numpy 合成逐 token 对数比率序列，比较 token / sequence 两级 TIS 的偏差与方差（模块 06 练习已有这两级的实现） |
| 2 | harness 信号（任务类型、计划的工具序列）能否比只看 prompt 更好地预测轨迹长度？ | A E | StreamRL 用输出长度 ranker；Seer 利用同一 prompt 多个样本的长度相似性 | 部分可做：公开 trace 可用 [SWE-bench/experiments](https://github.com/SWE-bench/experiments) 里各提交的 trajectories（Ask the Tool 用了 mini-SWE-agent 排行榜 runs 与 OpenHands 评测 runs）；把预测接入调度模拟需要库新增「按预测长度派发」 |
| 3 | staleness 上界是否应随工具延迟分布自适应？ | A | AReaL 用固定的 η 并靠配比控制 staleness；ROLL Flash 报告 α=2 对多数场景够用 | 系统侧现在可做：在 [模块 06](../06_long_tail_async/README.md) 的模拟器里扫 `mean_tool_s` × `max_staleness`；算法侧需要真实训练 |
| 4 | 多 agent 下的预算化反事实：选「哪个 agent × 哪一轮」分支？分支还要环境快照、还在组屏障上 | B A | 单 agent 选分叉点：TreeRL、ARPO、FR3E；多 agent 每轮分支：AT-GRPO；按 NLL 调粒度：[GACA](https://arxiv.org/abs/2609.12424)；[Coverage, Not Targeting](https://arxiv.org/abs/2609.02417)（仅在终局验证器、信息密度 V_d 远低于约 0.8 时，均匀分配胜过瞄准） | agent 级简化版现在可做（第 2.5 节「第一张图」）；轮次级需要库新增多步 team task |
| 5 | 多轮 agent 里，KV fork 做反事实什么时候反而不省？（分叉后的工具结果仍要 prefill、分支要路由到同一实例、分支 KV 驻留挤占 batch） | B C D | TreePO 摊销公共前缀节省 22%–43% GPU 小时；Tree-GRPO 固定预算下更多 rollout；C3 靠 checkpoint 恢复少用 32% token | 现在可做：直接构造分叉的 block 序列喂给 `PrefixCache.match` / `insert`，统计命中与 prefill token，再用模块 01 的公式换算时间 |
| 6 | 两个 agent 共享一个 judge 奖励时，会不会形成只对彼此有效的「暗号」来骗 judge？ | B F | [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)：单个符号或「Let's solve this step by step」这类开头就能骗出正奖励（单模型攻击） | 需要小模型真实训练：生成者 + 审阅者共享 judge 奖励，比较联合训练与分开训练时「judge 分 − 可验证分」的背离；再做「交换伙伴」对照：换成独立训练的伙伴后分数下跌，说明形成了私有约定 |
| 7 | 多项目共享前缀缓存：隔离粒度（租户 / 项目 / 公开前缀）、配额与命中率、p95 TTFT 的三方权衡 | C E | vLLM 的 `cache_salt`；[已有工作](https://arxiv.org/abs/2409.20002)展示了通过共享 KV cache 的时序侧信道推断他人 prompt；[模块 05 第 2.4 节](../05_cache_prefill/README.md#24-多租户共享还是隔离) | 现在可做：用 `generate_sessions(project_zipf_s=...)` 扫项目热度的集中度，比较共享一个 LRU `PrefixCache` 与「每个项目一个 `PrefixCache`、容量按配额切分」（近似按项目隔离，等价于不跨项目共享）的命中率；p95 TTFT 再接 `route_and_simulate` |
| 8 | 权重更新后，**新请求**能否复用上一版本的共享前缀 KV？复用几个版本？命中率 vs KL 的曲线 | C A F | 在途请求已有答案：AReaL 中断并重算；PipelineRL、Magistral 带旧 KV 继续，PipelineRL 实测 KL 只比重算略高 | 命中率侧现在可做（第 2.6 节「第一张图」）；KL 侧需要小模型 |
| 9 | P/D 分离按 turn 还是按会话？分界条件（append 长度、KV 迁移带宽、decode 节点负载）是什么？ | D E | ConServe 按会话（后续 append-prefill 在 decode 节点本地做）；Splitwise / DistServe 按请求 | 解析模型现在可做（推出两种方案的 KV 迁移量与排队）；`codesign_sim.pd` 已有按请求的 P/D 分离模拟（`simulate_pd` 与同卡数的 `simulate_colocated`），会话级方案需要在它上面加一个「首轮后固定 decode 节点、append-prefill 本地做」的变体 |
| 10 | harness 的上下文压缩/改写如何做成 cache 友好（只追加摘要、按 block 对齐）？省下的 prefill 值不值得一点质量损失？ | D C | Manus 的实践：保持前缀稳定、上下文只追加、序列化确定（[Manus 博客](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)）；[模块 04](../04_harness_workloads/README.md) | prefill 侧现在可做：`generate_sessions(rewrite_prob=...)`；质量侧需要真实任务 |
| 11 | 在 Dynamo 现有 hint 与模块 05 的提案之上，哪个新增字段（`tool_eta`、fork 组、可丢弃段）的边际收益最大？ | E | Dynamo `agent_hints`；SGLang 优先级淘汰；Continuum（工具时长）；Ask the Tool（进度） | 部分可做：`route_and_simulate` 已支持工具时长类的 pin 提示（`hint_mode` 取 none / fixed / noisy / oracle，oracle 是上界），可以先量化这一个字段的价值；fork 组、可丢弃段等其他字段需要库扩展 |
| 12 | 自我改进回路里的评估调度：候选共享祖先带来的**相关噪声**、评估集被反复使用导致的过拟合、候选质量非平稳 | F E | successive halving / [Hyperband](https://arxiv.org/abs/1603.06560) 是超参搜索里的成熟方法；HGM 用后代表现估计潜力来分配扩展与评估预算 | 现在可做：用 numpy 合成「祖先树 + 相关噪声」的候选，比较 successive halving 在相关噪声与独立噪声下的误判率与成本 |

## 8. 选读 F：把一个开放问题做成论文的最小实验设计模板

把下面十行填满，基本就是一篇 workshop 论文的骨架；填不满的那一行，就是你还没想清楚的地方。第 5 行里的「迷你引擎」指 [综合实战](../../capstone/README.md) 里真正执行 Transformer 前向的 `mini_engine`：机制类的想法（新的淘汰策略、会话感知调度）可以先在它上面验证正确性，再回到模拟器里放大规模。

```
1. 问题（一句话，含一个可测的量）：        在 ___ 负载下，___ 导致 ___ 指标恶化 ___ 倍
2. 为什么是现在（一个数量级锚点 + 来源）：  例：生成阶段常占 RL 运行时间 90% 以上（OpenRLHF）
3. 假设 H1 与反假设 H0：                  H1：___ 能把 ___ 降到 ___；H0：收益 < 噪声
4. oracle 上界（先证明值得做）：            假设完美信息时最多能省多少？< 10% 就换题
5. 最小环境（由便宜到贵）：                 解析模型 → 模拟器（codesign_sim）→ 真实执行的迷你引擎（mini_engine）→ 小模型真实执行 → 公开 trace
6. 基线（至少三条）：                       朴素做法 / 最强已发表方法 / oracle（只作上界）
7. 指标（系统 + 算法各一，缺一不可）：        系统：吞吐、p99、GPU·h；算法：回报、样本效率、最终精度
8. 对照与消融：                             同预算、同硬件、同 trace、同 batching；多 seed 并报置信区间；随机化对照（例：打乱位置但保持集中度）
9. 规模外推：                               用数量级论证：放大到千卡、百万 token 上下文时结论是否还成立、哪一项先变成瓶颈
10. 效度威胁：                              模拟器假设、负载代表性、与已有工作的差异点
```

**关于 oracle 的两条纪律**：oracle 用了部署时拿不到的信息（真实工具时长、真实输出长度、未来请求），只能作为离线上界，不能当成可部署的基线参与排名；它的信息也不能泄漏进被比较的方法（例如用真实长度调出来的阈值）。

**一个填好的例子（问题 5）**：

1. 问题：多 agent RL 中反事实信用要为每个样本多跑 n 条分支，同步训练下 step time 明显变长。
2. 锚点：经典 difference reward 需要能评估反事实世界，实际环境里往往要额外仿真（COMA 论文对 difference reward 的总结）；COMA 本身用中心化 critic 对单个 agent 的动作做边缘化，不需要真实重跑。本题研究的是**真的去跑反事实分支**时系统能省多少：C3 报告靠 checkpoint 恢复、不重新生成历史，比 MAPPO 少用 32% token。
3. H1：分支从决策点的 KV fork 出发且前缀全部命中时，额外成本只剩「分叉后追加内容的 prefill + 分支的 decode」，step time 的增幅远小于 n 倍；H0：长尾 decode 主导，fork 省下的历史 prefill 可以忽略。
4. oracle：假设决策点之前的 prefill 完全免费，额外成本最多下降多少？先用 [模块 01](../01_inference_basics/README.md) 的 prefill / decode 时间公式算。
5. 环境：`codesign_sim.credit`（信用质量）+ `PrefixCache`（分支命中率）+ `simulate_rollout`（step time）。
6. 基线：shared reward（零额外成本）、LOO 全量重跑、LOO + checkpoint 恢复（C3 式，不重新 decode 历史）、LOO + KV fork。
7. 指标：梯度方差、达到目标回报所需的 GPU·s；p99 step time。
8. 对照：固定总 GPU·s 预算；20 个 seed 并报置信区间。
9. 外推：上下文 100k token、n=8 个 agent 时，prefill 与 decode 的比例如何变化。
10. 威胁：真实系统中分支 KV 驻留会挤占 batch；分支必须路由到同一实例才能命中；编码 agent 还要沙箱快照。

## 9. 必须会心算的数字

| 量 | 典型值 | 怎么推出来 | 来源 |
|---|---|---|---|
| 50 轮会话「缓存全丢」vs「全命中」的 prefill token 比 | ≈ 30× | 共享前缀 20k、每轮外部追加 2k：累计 ≈3.6M vs 0.12M（第 2.1 节） | 本模块推导 |
| 工具调用打断导致的重算占前向时间 | 37–40% | InferCept 在其负载下的测量 | [InferCept](https://arxiv.org/abs/2402.01869) |
| 生成阶段占 RL 总运行时间 | 常超过 90% | OpenRLHF 的笼统描述（APRIL 引用） | [OpenRLHF](https://arxiv.org/abs/2405.11143) |
| 异步 RL 相对同步的训练加速 | 最高 2.77× | AReaL 同卡数对比 | [AReaL](https://arxiv.org/abs/2505.24298) |
| 后训练算力 / 预训练算力 | > 10% | DeepSeek-V3.2 的公开数字 | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| 立即清空时的前缀命中率 | 1 − 请求间隔 / 更新周期 | 每个更新周期的第一次请求必 miss：每 5 s 一次请求、30 s 一次更新 → 5/6 ≈ 83% | 本模块推导 |
| 指数工具时长下 pin 住 KV 的条件 | B > c · E[T] | 风险率恒为 1/E[T]，最优 TTL 只能取 0 或 ∞（第 4 节） | 本模块推导 |
| Qwen3-32B 一个 100k token 上下文的 KV | ≈ 26 GB | 64 层 × 8 个 KV 头 × 128 维 × 2（K、V）× 2 字节 = 256 KiB/token | [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) |

## 10. 实验

本模块是纯讲义，没有 `lab.py`。第 7 节「抓手」一列标了「现在可做」的问题，都可以用前面模块的代码在 10 秒量级跑出第一张图。

## 11. 练习（选读）

本模块没有代码练习。书面练习：从第 7 节挑一个问题，按第 8 节模板写满十行，控制在一页以内。

## 12. 自测题

前 3 题属于 30 分钟核心，后 3 题是需要计算的选读题。

1. 为什么说「程序」比「请求」更适合作为 agent 负载的调度单位？给一个数量级论证。
2. 权重更新时，在途请求的 KV 有哪两种处理方式？各付出什么代价？
3. 在多 agent 反事实信用里，KV fork 和 checkpoint 恢复分别省掉了哪一笔计算？分叉之后还剩哪些计算？
4. 一个 Qwen3-32B 实例跑在 8×H100 上（TP8），某会话上下文 100k token，正在等工具。假设：重算 prefill 约 3.6 s（按模块 01 的公式、MFU 0.6）；实例可用于 KV 的显存约 570 GB，且 batch 受 KV 容量限制、吞吐与可用 KV 成正比；工具时长服从指数分布。平均工具时长低于多少时值得 pin 住这份 KV？
5. rollout 侧每 10 s 同步一次权重并立即清空前缀缓存，某项目的共享前缀每 2 s 被新请求使用一次。命中率是多少？改成每 60 s 同步一次呢？如果改为保留上一个版本的前缀 KV 给新请求用，命中率和代价各变成什么？
6. 一条 800 token 的轨迹：前 400 个 token 由 v10 生成，平均每 token 对数比率 log(π_v12/μ) = 0.005；后 400 个由 v11 生成，平均为 0.002。训练时策略为 v12。序列级重要性权重是多少？逐 token 截断 IS（C=2）会截断吗？

<details><summary>答案</summary>

1. 程序知道「接下来还会有调用、共享什么前缀、已经服务了多少」。按请求调度且缓存全丢时，50 轮会话要 prefill ≈3.6M token；缓存全命中只要 ≈0.12M，差约 30 倍（第 2.1 节）。程序级调度还能缓解程序级队头阻塞（Autellix）。
2. AReaL：中断在途生成、丢弃 KV、用新权重重算后续写，版本干净但要多付一次 prefill；PipelineRL / Magistral：带着旧 KV 继续生成，不浪费计算，但序列内是「新权重 + 旧 KV」的混合，PipelineRL 实测 KL 只比重算略高。两者都要求按 token 记录采样 logprob 和版本号。
3. KV fork 省掉决策点之前的 **prefill**；checkpoint 恢复（C3）省掉决策点之前的 **decode**（不用重新生成历史）。分叉之后追加的工具结果、其他 agent 的消息仍要 prefill，分支自己的输出仍要 decode；前缀要命中，分支还得路由到保存这份 KV 的同一实例。
4. KV = 100k × 256 KiB ≈ 26.2 GB，占可用 KV 的 26.2 / 570 ≈ 4.6%，所以 pin 住它每秒的机会成本 c ≈ 0.046 个实例·秒。B ≈ 3.6 实例·秒（忽略被驱逐后的排队）。指数分布下 pin 当且仅当 B > c·E[T]，即 E[T] < 3.6 / 0.046 ≈ 78 s。常见错误：忘了 KV 在 8 张卡上分摊，或把 B 当成单卡时间。
5. 每个 10 s 周期里有 5 次请求，第一次必 miss：命中率 = 1 − 2/10 = 80%；60 s 周期时为 1 − 2/60 ≈ 97%。保留上一版本 KV 给新请求用时，除了第一次冷启动之外全部命中（接近 100%），代价是这些请求的前缀是旧权重算出来的（KL 增加，需要像 PipelineRL 那样测），并且要同时容纳两个版本的前缀 KV。
6. 序列级 log W = 400×0.005 + 400×0.002 = 2.8，W = e^2.8 ≈ 16.4，一条样本的权重被放大 16 倍，方差随长度指数增长。逐 token 比率约为 e^0.005 ≈ 1.005 和 e^0.002 ≈ 1.002，远低于 C=2，不会被截断。所以长轨迹用逐 token 比率，而不是序列级乘积（模块 06 第 5.2 节）。

</details>

## 13. 面试追问（选读）

1. **「agent 负载和聊天负载，对推理引擎最大的不同是什么？」**
   结论：请求之间有强依赖且中间夹着工具等待，调度和缓存必须按程序/会话看。数量级：50 轮会话缓存全丢 vs 全命中，prefill 差约 30 倍。取舍：会话粘性提升命中但造成负载不均；LAS 类程序级优先级在程序规模重尾时往往改善平均 JCT，但要防饿死，且在不可抢占的 prefill 与 cache 局部性约束下收益不保证。
2. **「工具调用期间 KV 留在显存还是换出？」**
   结论：比较「保留的机会成本 c·E[min(T, τ)]」和「命中时省下的重算/重载 + 排队 B」，按风险率 h(τ) > c/B 决定是否继续保留。数量级：用模块 01 的 KV 公式算占用；第 12 节第 4 题的设定下，平均工具时长低于约 78 s 才值得 pin。取舍：固定 TTL 简单但对时长方差不鲁棒（Continuum 用历史分布 + 程序级 FCFS）；运行中上报进度更准但要改 harness（Ask the Tool）。
3. **「设计一个支持 agentic RL 的 rollout 系统，你会先解决什么？」**
   结论：先量化 rollout 时间分解（生成 / 环境 / 奖励），通常长尾和环境先成为瓶颈。数量级：生成阶段常占 90% 以上（OpenRLHF 的描述）；DSec 生产中超过 38 万个并发沙箱。取舍：partial rollout vs 完全异步 vs 解耦（第 5 节），每一步都用算法上的 off-policy 代价换系统利用率。
4. **「如果模型每小时更新一次权重并在线服务，你的推理基础设施要改什么？」**
   结论：版本化一切：KV cache、路由、评估基线都要带模型版本。数量级：立即清空时前缀命中率 = 1 − 请求间隔 / 更新周期。取舍：灰度发布通常是两个版本的实例池并存、按比例切流，每张卡只放一个版本：总容量要预留余量，KV 与路由都要带版本号以免跨版本命中；回滚要求旧版本实例池能快速恢复。
5. **「自我改进的 agent 系统里，最容易被忽略的基础设施成本是什么？」**
   结论：评估。每个候选都要跑基准，且要防止在基准上过拟合。取舍：评估越省越容易误判；successive halving / Hyperband 是成熟工具，但 RSI 里候选共享祖先、噪声相关，要用与评测集不相交的进化任务和回归门禁换可靠性（HGM、ModularRSI、SEAGym）。

## 14. 延伸阅读

- [Autellix](https://arxiv.org/abs/2502.13965)：把 agent 程序当一等公民调度的代表作，读它如何定义程序级队头阻塞。
- [Continuum](https://arxiv.org/abs/2511.02230) 与 [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849)：工具调用期间 KV 去留的「预测」与「观测」两条路线。
- [KVFlow](https://arxiv.org/abs/2507.07400)、[CacheScout](https://arxiv.org/abs/2608.14624)、[TOPAS](https://arxiv.org/abs/2608.25523)：多 agent workflow 的 KV 管理三种思路（声明 / 学习 / 联合调度）。
- [ConServe](https://arxiv.org/abs/2606.01839)：会话级 P/D 分离，prefill 方向的新视角。
- [PipelineRL](https://arxiv.org/abs/2509.19128)：in-flight 权重更新与陈旧 KV 的实测，版本化缓存问题的起点。
- [TreePO](https://arxiv.org/abs/2508.17445)、[Tree-GRPO](https://arxiv.org/abs/2509.21240)、[TreeRL](https://arxiv.org/abs/2506.11902)：树形 rollout 与前缀摊销，反事实信用系统代价的直接前作。
- [AReaL](https://arxiv.org/abs/2505.24298)、[StreamRL](https://arxiv.org/abs/2504.15930)、[RollArt](https://arxiv.org/abs/2512.22560)、[Polar](https://arxiv.org/abs/2605.24220)：agentic RL 系统演进的四个节点。
- [DSec](https://arxiv.org/abs/2609.22978)：沙箱平台与 RL 的协同设计，分支需要环境快照时的一手参考。
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)、[Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)、[AlphaEvolve](https://arxiv.org/abs/2506.13131)：自我改进回路里「评估器」如何成为核心部件。
- [SEAGym](https://arxiv.org/abs/2606.17546)、[ModularRSI](https://arxiv.org/abs/2609.14857)：自我进化 harness 的评估与泛化问题。
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/)：RSI 方向的问题框架与已接收论文列表。
- 全项目阅读清单见 [docs/reading-list.md](../../docs/reading-list.md)。
