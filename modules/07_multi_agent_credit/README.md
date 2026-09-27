**中文** · [English](README.en.md)

# 模块 07：多 agent 奖励与信用分配

> 🕐 建议时长：1.5 小时（第 7.3–7.5 节与第 6 节论文表为选读，不计入） ｜ 🎯 学完你能：推导组内基线（GRPO / RLOO）什么时候无偏、什么时候降方差、什么时候反而有害；说清多轮 agent 的 turn/step 级优势有哪几种来源；区分共享奖励、difference reward、COMA 反事实基线、Shapley / 留一法、值分解五类多 agent 信用分配，并算出每一种的偏差条件和额外评估次数；用「方差下降倍数 vs 有效吞吐下降倍数」判断反事实信用在同步 / 异步系统里值不值 ｜ 🧪 实验：[`lab.py`](lab.py) ｜ 📖 必读一手资料：[COMA](https://arxiv.org/abs/1705.08926)、[DeepSeekMath（GRPO，第 4 节）](https://arxiv.org/abs/2402.03300)

## 0. 一句话

**信用分配（credit assignment）就是把一个稀疏、带噪的团队标量奖励，拆成「每个 agent、每一步」都能用的学习信号；几乎每一种更准的拆法，都要用更多的评估（额外 rollout 即按策略多跑出的轨迹、critic 即价值网络的前向、judge 即打分模型的调用）去换。所以它既是算法问题，也是训练系统的吞吐问题。**

## 1. 大厂为什么在意

一个具体场景：一个 planner + coder + tester 的多 agent 编码系统用 RL 训练。每个 episode 有 20–50 次工具调用，最终奖励只有一个「测试是否通过」（0/1）。

- **稀疏**：训练初期平均成功率可能只有 5%。GRPO 每个 prompt 采 G=8 条，若每道题的成功率都是 0.05，全错的组占 0.95⁸ ≈ 66%，这些组的组内优势全为 0，梯度白算。真实题目难度不均匀，情况更糟：平均成功率同样是 0.05，若成功率服从 Beta(0.25, 4.75)，零优势组占 77%（第 8 节）。DAPO 的 Dynamic Sampling 通过超额采样、过滤掉全对/全错的 prompt 来解决（[DAPO](https://arxiv.org/abs/2503.14476)），代价是多跑 rollout；成功率为 0 的题永远凑不满，只能靠课程或过滤。
- **归因错**：共享奖励下，planner 会因为 coder 碰巧写对而得到正梯度。VDN 论文把这叫做源自队友行为的「虚假奖励信号」（spurious reward），并描述了「一个 agent 学会了、另一个因为自己的探索会拖累团队而不再学」的 lazy agent 现象（[VDN](https://arxiv.org/abs/1706.05296)）；近期 LLM 工作把冗余 agent 白拿正奖励称为 free-riding（[CCPO](https://arxiv.org/abs/2603.21563)）。
- **贵**：更准的信用通常要反事实评估，也就是「如果这个 agent 换个动作 / 不在场，结果会怎样」。COMA 论文对 difference reward 的总结是：每个 agent 都需要一次单独的反事实仿真（[COMA](https://arxiv.org/abs/1705.08926)）。多 agent 系统本来就贵，Anthropic 报告其多 agent 研究系统的 token 消耗约为普通聊天的 15 倍（[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)）。再乘上 n 倍的反事实 rollout，就直接压到 [模块 06](../06_long_tail_async/README.md) 讲的长尾关键路径上。

读者正在研究这个方向：本模块先把算法谱系和每种方法的**偏差条件与系统账单**讲清楚，再用实验看方差、学习速度和吞吐。

## 2. 结果奖励：稀疏、方差与基线

### 2.1 策略梯度与基线

REINFORCE 的梯度估计是 g = (R − b) ∇log π(a)。只要基线 b 不依赖当前样本的动作 a，E[b ∇log π(a)] = b ∇Σπ = 0，所以**基线不改变期望，只改变方差**。

先看奖励乘子 (R − b) 的二阶矩，它决定梯度方差的量级（严格的梯度协方差是 E[(R − b)² s sᵀ] − ∇J ∇Jᵀ，其中 s = ∇log π(a)；这里忽略 s 的大小与 R 的相关性，只为估数量级）。对 0/1 奖励、成功率 p：

- 不用基线：E[R²] = p
- 用**真实**均值 p 作基线：E[(R − p)²] = p(1 − p)，比值 **1 − p**
- 用**留一法**估出来的基线（RLOO）：基线本身有方差 p(1 − p)/(G − 1)，比值 **(1 − p)·G/(G − 1)**
- 减**含自己的组均值**：比值 (1 − p)·(G − 1)/G，看起来更小，但这是有偏的收缩（第 2.2 节）

三个结论：成功率低时基线几乎不省方差（p=0.05 时真实均值也只省 5%）；**p < 1/G 时，估出来的 RLOO 基线反而让二阶矩变大**（p=0.05、G=8 时比值 1.086）；成功率高时基线省得最多（p=0.9 时降到约 1/10）。所以低成功率时该先增大 G、做课程或过滤，而不是指望基线。

### 2.2 GRPO 与 RLOO：用同组样本当基线

对同一个 prompt 采 G 个回答，奖励 r₁…r_G：

| 方法 | 样本 i 的优势 | 期望梯度 |
|---|---|---|
| RLOO | Âᵢ = rᵢ − (1/(G−1)) Σ_{j≠i} rⱼ | 基线不含样本 i 自己，**严格无偏**（[Kool et al. 2019](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/)；[Ahmadian et al. 2024](https://arxiv.org/abs/2402.14740)） |
| 减组均值（Dr. GRPO） | Âᵢ = rᵢ − mean(r)，并去掉目标里的 1/\|oᵢ\| | mean 含 rᵢ/G 这一项，而其余样本与 aᵢ 独立，所以 E[(rᵢ − mean)∇log π(aᵢ)] = (1 − 1/G)·∇J：**方向不变、整体缩小 (G−1)/G**。Dr. GRPO 去掉 1/\|oᵢ\| 与 std，消除长度偏置与难度偏置（[Dr. GRPO](https://arxiv.org/abs/2503.20783)） |
| GRPO（结果监督） | Âᵢ = (rᵢ − mean(r)) / std(r)，同一回答的所有 token 共享这个值 | std 由包括样本 i 在内的整组动作决定，不再是常数缩放：一般**有偏**；它按题目重新加权，std 小的题（过易或过难）权重大（[GRPO](https://arxiv.org/abs/2402.03300)，DeepSeekMath 实验中每题采 64 个输出） |

一个值得背下来的恒等式：rᵢ − mean_{j≠i} rⱼ = **G/(G−1) · (rᵢ − mean(r))**。RLOO 和「减组均值」的**样本值**只差一个常数缩放，但前者的期望是 ∇J、后者是 (G−1)/G·∇J。练习 1 会让你验证它。

DeepSeekMath 还给了过程监督版本：每一步的奖励先在全组内归一化，token t 的优势是**它之后所有步**归一化奖励之和。这就是第 3 节 step 级优势的最简单形式。

### 2.3 什么时候组内基线失效

1. **全对或全错的组**：优势全为 0。比例 = pᴳ + (1−p)ᴳ；题目难度越不均匀，比例越高（第 8 节、实验第 0 部分）。DAPO 原文指出，在同步且生成不流水化的系统里，这种超额采样不一定拖慢训练，因为生成时间通常由长尾样本主导：这正是实验第 4 部分「额外 rollout 填进气泡」的一手旁证。
2. **组内 prompt 不一样**：多 agent、多轮时，每个 agent 每一轮看到的 prompt（角色 + 交互历史）都不同，「同组比较」的前提不成立（[AT-GRPO](https://arxiv.org/abs/2510.11062)，第 6 节）。
3. **组内方差来自别人**：多 agent 共享奖励时，组内方差里混着其他 agent 的贡献，基线只去掉公共均值，去不掉别人的方差（第 4 节和实验）。

## 3. 时间维度：多轮 agent 的 turn/step 级优势

轨迹级优势把一个标量广播给所有 token / 所有轮。轨迹一长，问题就暴露了：TRACE 的作者指出，一条失败的轨迹里可能有很多把 agent 推向目标的好动作，但只看结果的训练会给它们和最终那个错误同样的负优势（[TRACE](https://arxiv.org/abs/2607.13988)）。

| 做法 | 信号从哪来 | 额外代价 | 代表工作 |
|---|---|---|---|
| 轨迹级广播 | 结果奖励 | 0 | GRPO、RLOO |
| 过程奖励模型（PRM）逐步打分 | 人工标注的步级标签（过程监督在 MATH 上显著优于结果监督，发布 80 万条步级标签 PRM800K）；或从某一步出发补全 N 次、按答对比例自动标注；或只用结果标签在线更新的隐式 PRM | 训练并前向一个 PRM（自动标注还要每步 N 条补全）；PRM 本身可被 hack | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)、[Math-Shepherd](https://arxiv.org/abs/2312.08935)、[PRIME](https://arxiv.org/abs/2502.01456) |
| 学出来的 value 网络 | critic | critic 的训练与显存 | PPO；[VinePPO](https://arxiv.org/abs/2410.01679) 发现 PPO 的 value 网络在推理任务上常把坏状态判成好状态 |
| 中间状态的 MC 回报 | 从状态 sₜ 重新采 K 条 rollout 取平均 | 每个状态 K 条；VinePPO 每步慢到 2–5 倍，但达到同等精度的墙钟时间反而更短 | [VinePPO](https://arxiv.org/abs/2410.01679) |
| 树形 rollout | 从中间步分叉，同一前缀下的分支互为基线 | 分叉点的选择与前缀摊销；TreePO 摊销公共前缀节省 22%–43% GPU 小时 | [TreeRL](https://arxiv.org/abs/2506.11902)、[TreePO](https://arxiv.org/abs/2508.17445)、[Tree-GRPO](https://arxiv.org/abs/2509.21240) |
| 组内同状态再分组 | 在同一组轨迹里找重复出现的环境状态（anchor state），在这些状态上比较不同动作 | 几乎为 0，不需要额外 rollout | [GiGPO](https://arxiv.org/abs/2505.10978) |
| turn 级奖励设计 | 可验证的中间奖励或 LLM judge | 奖励设计 / judge 调用 | [Wei et al. 2025](https://arxiv.org/abs/2505.11821)（稠密的逐轮奖励在搜索与游戏 agent 上优于稀疏的终局奖励） |
| 参考模型的对数概率 TD | 冻结参考模型对标准答案的 log-prob，转成状态价值，再取 TD 差 | 参考模型前向 | [TRACE](https://arxiv.org/abs/2607.13988) |
| 自适应粒度 | 按每步 NLL 混合 step 级与 episode 级优势 | 几乎为 0 | [GACA](https://arxiv.org/abs/2609.12424) |

**别急着「瞄准」**：2026 年的一项对照实验发现，当验证器只能看到最终状态（大部分轮的对错不可见，信息密度远低于其相变点约 0.8）时，把稠密奖励**均匀**摊到各轮，胜过把同样的优势集中到「关键轮」上；它还提出了「打乱位置但保持集中度」的对照，任何声称「精准定位关键 turn」的方法都应该过这个对照（[Coverage, Not Targeting](https://arxiv.org/abs/2609.02417)）。

## 4. 空间维度：多 agent 的信用分配

设 n 个 agent，联合动作 a = (a₁, …, aₙ)，团队奖励 R(a)。agent i 的策略梯度用它的信用 Cᵢ 代替 R。

### 4.1 共享奖励

Cᵢ = R。对团队目标无偏，但 Cᵢ 的方差里包含**所有其他 agent 的贡献 + 观测噪声**：只有一元贡献时约随 n 线性增长；有两两协同项时，协同项的个数是 n(n−1)/2，方差随 n 超线性增长（实验第 3 部分）。加上组内均值基线只能去掉公共均值，去不掉别人的方差。

### 4.2 difference reward

Dᵢ = R(a) − R(a₋ᵢ, cᵢ)：把 agent i 的动作换成默认动作 cᵢ，看团队奖励掉多少。这一族奖励由 Wolpert 与 Tumer 提出，称为 Wonderful Life Utility（见 [Wolpert & Tumer 2001](https://doi.org/10.1142/S0219525901000188)）。

- 好处：第二项不依赖 aᵢ，所以任何让 Dᵢ 变大的动作也让 R 变大；不涉及 i 的项在两项里相消，方差大幅下降。但和 i 有关的协同项消不掉，所以 difference 的方差也会随 n 缓慢上升。
- 代价（COMA 论文的总结）：需要模拟器或奖励模型来算 R(a₋ᵢ, cᵢ)；每个 agent 一次额外仿真；默认动作 cᵢ 往往不好选。
- **噪声怎么配对很关键**：如果反事实那次评估用独立的随机性重跑，它自带一份观测噪声，Dᵢ 的方差会**多出一份噪声方差**；如果用共同随机数（common random numbers，同一个环境种子、同一份工具返回），两次评估的噪声相关，差分时大部分被抵消。实验第 1 部分对比了这两种情况。

### 4.3 COMA 反事实基线

Aᵢ(s, a) = Q(s, a) − Σ_{a'ᵢ} πᵢ(a'ᵢ | τᵢ) Q(s, (a₋ᵢ, a'ᵢ))

用一个中心化 critic 把「换默认动作」换成「对 agent i 自己的动作按当前策略取期望」。基线不依赖 aᵢ 的实际取值，所以对梯度的期望贡献为 0；critic 的输入里放其他 agent 的动作、输出 agent i 的每个动作的 Q 值，一次前向就能算完，输出维度是 |U| 而不是 |U|ⁿ（[COMA](https://arxiv.org/abs/1705.08926)）。**系统视角：它把「n 次额外仿真」换成了「critic 的一次批量前向」，不需要真实重跑。** 注意优势的第一项是学出来的 Q(s, a) 而不是真实回报，Q 有误差时优势就有偏差。

**在 LLM 里 COMA 怎么落地**，取决于「动作」的粒度：

- **token 级**：要对整个词表求和（例如 Qwen3-32B 的词表有 151,936 个 token），而且 agent 的一次决策是一整段消息，不是一个 token：不可行。
- **消息级**：没法枚举所有消息，只能对 K 个采样出来的候选取平均，这就退化成「采样版反事实基线」，和 C3 / 留一法是同一类，每个候选都要一个 critic 打分或一段后续 rollout。
- **工具调用级**：动作空间可以枚举（调哪个工具、哪组参数），但评估每个候选要恢复环境状态（第 7.3 节）。

### 4.4 Shapley 值与留一法

Shapley 值（[Shapley 1953](https://doi.org/10.1515/9781400881970-018)）把团队价值 v(N) 按「所有加入顺序下的平均边际贡献」分给每个人：

φᵢ = Σ_{S ⊆ N∖{i}} |S|!(n − |S| − 1)!/n! · [v(S ∪ {i}) − v(S)]

它是唯一同时满足效率（Σφᵢ = v(N) − v(∅)）、对称、零贡献者得 0、可加性的分配。代价是精确计算要 **2ⁿ 个联盟的价值**（原始联合动作那一个已经有了，额外 2ⁿ − 1 个）；蒙特卡洛近似采 M 个排列，要约 M·n 次边际评估，**只有 M·n < 2ⁿ 时才比精确算法便宜**（n=4、M=8 时 32 > 16，反而更贵）。[Shapley Q-value](https://arxiv.org/abs/1907.05707) 把它用作每个 agent 的 critic。

**Shapley 当作策略梯度的信用，有偏还是无偏，取决于「缺席」怎么定义**。设联盟价值 v(S) = E_{a'₋S ∼ π}[f(a_S, a'₋S)]，即缺席的 agent 按**当前策略**边缘化。那么：

1. i ∉ S 时，v(S) 不依赖 aᵢ，所以 E[v(S) ∇log πᵢ(aᵢ)] = 0；
2. i ∈ S 时，样本里的 a_{S∖i} 本来就按 π 采样，缺席者也按 π 取期望，所以 E[v(S) | aᵢ] = Qᵢ(aᵢ)，于是 E[v(S) ∇log πᵢ] = E[Qᵢ(aᵢ) ∇log πᵢ] = ∇ᵢJ；
3. 权重之和 Σ_S w_S = 1，所以 E[φᵢ ∇log πᵢ] = ∇ᵢJ。∎

也就是说，**按当前策略边缘化缺席者时，Shapley 信用对策略梯度严格无偏**；按均匀随机动作、默认动作或「把 agent 移除」来定义缺席，都会引入偏差（策略离参考分布越远，偏差越大）。实验第 1 部分同时给出两种参考分布。

两个系统上的提醒：

- 一个 v(S) 往往不是「跑一条 rollout」就能得到的：缺席者按 π 取期望，本身就要内层采样（或对 K^{n−|S|} 种组合求平均），toy 任务里能闭式算，真实系统里要乘上每个联盟的 rollout 数。
- 留一法（leave-one-out，LOO）只看「最后一个加入」的边际贡献：v(N) − v(N∖{i})，本质上就是「默认动作 = 缺席」的 difference reward。

| 团队结构 | 例子 | LOO | Shapley |
|---|---|---|---|
| 互补（缺一不可） | 3 个 agent 都做对才成功 | 每人 1，总和 3 > 团队价值 1 | 每人 1/3 |
| 替代（任一即可） | 两个 tester 谁发现 bug 都算 | 每人 0 | 每人 1/2 |
| 手套博弈 | 0 号有左手套，1、2 号各有右手套 | (1, 0, 0) | (2/3, 1/6, 1/6) |

LLM 多 agent 里「移除一个 agent」还有一个额外问题：它会改变下游 agent 看到的输入。C3 的论证是，这样测到的差值混入了下游 agent 面对陌生输入时的退化，而且加样本也消除不了这个偏差；它改为**固定完整历史、只重采这个 agent 在该决策点的动作**，再用留一法基线算优势（[C3](https://arxiv.org/abs/2603.06859)）。这个分布偏移对 LOO 成立，对 Shapley 里每个 S ≠ N 的联盟（同样要把 agent 移除）也成立。

### 4.5 值分解（VDN / QMIX，简述）

- VDN：Q_tot = Σᵢ Qᵢ，团队价值按 agent 相加分解（[VDN](https://arxiv.org/abs/1706.05296)）。
- QMIX：用一个混合网络把 Qᵢ 组合成 Q_tot，并**结构上保证单调**（∂Q_tot/∂Qᵢ ≥ 0），这样每个 agent 各自取 argmax 就等于联合 argmax（[QMIX](https://arxiv.org/abs/1803.11485)）。

它们是值方法，在 LLM 上直接用得少（token 级 Q 值很难学），但「把团队价值分解成 agent 项」的思想会出现在 critic 设计里。面试时能说清「加性 → 单调」这条演进即可。

### 4.6 取舍总表

| 方法 | 偏差 | 方差 | 每个样本的额外评估 | 前提 | 适用 |
|---|---|---|---|---|---|
| 共享奖励 | 无偏 | 高，随 n 超线性增长 | 0 | 无 | n 小、强耦合 |
| RLOO 组内基线 | 无偏 | 去掉公共均值；p < 1/G 时二阶矩反而变大 | 0（同组样本本来就要采） | 同一 prompt 多样本 | 默认起点 |
| 减组均值（Dr. GRPO / MAGRPO） | 方向不变，缩小 (G−1)/G | 同上 | 0 | 同上 | 同上 |
| GRPO（除 std） | 一般有偏（按题目重加权） | 同上 | 0 | 同上 | 实践常用 |
| difference reward | 无偏（与团队目标对齐） | 低；独立噪声下多一份噪声方差 | n | 模拟器 + 默认动作 | 环境可重置 |
| COMA 反事实基线 | 基线无偏；Q 有误差时优势有偏 | 低 | 1 次 critic 批量前向（或 n·(K−1) 次仿真） | 准确的中心化 critic | 动作空间小 |
| 精确 Shapley | 按 π 边缘化时无偏；按均匀 / 默认 / 移除定义缺席时有偏 | 低 | 2ⁿ − 1 个联盟 × 每个联盟的评估数 | 能评估任意联盟 | n ≲ 10 |
| MC Shapley | 同上，另有采样误差 | 中 | ≈ M·n（只有 M·n < 2ⁿ 才划算） | 同上 | n 大 |
| LOO / 移除 agent | 互补或替代时分配不公平；LLM 中还有分布偏移 | 低 | n | 能移除 agent | 冗余检测 |
| CCPO（双 agent 移除一个） | 同上 | 低 | 1 | 能让另一个 agent 单独作答 | Think → Solve |
| 固定历史重采（C3） | 无偏 | 低 | 决策点数 × 候选数 × 每个候选的 rollout 数 | 历史可精确恢复 | 纯文本多 agent |
| 值分解 | 结构假设 | 低 | critic 前向 | 训练 critic | 值方法 |

## 5. LLM judge / rubric 作为奖励，以及 reward hacking

很多多 agent 任务没有可验证的答案（写作、研究报告、讨论质量），只能让另一个模型（judge）按评分细则（rubric）打分；而策略会去钻打分器的空子，这就是 reward hacking：

- **judge 有系统偏差**：MT-Bench 论文列出了位置偏差、冗长偏差、自我偏好偏差和有限的推理能力；同时 GPT-4 级别的 judge 与人类偏好一致率超过 80%，与人类之间的一致率相当（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。
- **rubric / checklist 更稳**：Rubrics as Rewards 在 HealthBench 上相对 Likert 直接打分最多提升 31%，且小 judge 用 rubric 更对齐（[RaR](https://arxiv.org/abs/2507.17746)）；RLCF 把指令拆成 checklist 逐条打分，在五个基准上都有提升（[Checklists](https://arxiv.org/abs/2507.18624)）。Kimi K2 用 verifiable reward + self-critique rubric reward 的组合，并用可验证任务上的 on-policy rollout 持续更新 critic（[Kimi K2](https://arxiv.org/abs/2507.20534)）。
- **reward hacking 可以定量**：Gao 等人在合成设定下拟合出，以 d = √KL(π‖π_init) 为横轴，best-of-n 的真实奖励 R(d) = d(α − βd)，RL 的 R(d) = d(α − β log d)，都是先升后降（[Gao et al.](https://arxiv.org/abs/2210.10760)）。
- **judge 很脆**：「:」「.」或「Let's solve this problem step by step.」这类「万能钥匙」就能从多个 judge（包括闭源模型）骗出正奖励（[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)）。

**系统代价**：judge 调用本身就是推理负载，每条轨迹至少一次，逐步 / 逐 rubric 打分时乘以步数或条目数，而且位于 RL 关键路径上。RollArt 把无状态的奖励计算放到 serverless 上（[RollArt](https://arxiv.org/abs/2512.22560)）。常用防御：KL 约束、多 judge 集成、rubric 拆分 + 可验证锚点（CCPO 的 SEPO 让外部任务结果始终占主导）、定期用金标准评估，以及监控「judge 分」与「可验证分」的背离。

## 6. 近期多 agent LLM 强化学习（逐篇核实，选读）

| 论文 | 设定 | 信用怎么分 | 系统上需要什么 |
|---|---|---|---|
| [MALT](https://arxiv.org/abs/2412.01928)（2024-12） | 生成 → 验证 → 修正三个角色的串行流水线 | 每个角色多次采样形成搜索树，用价值迭代把最终对错传回每个角色；off-policy | 树形采样：每个角色重复采样，样本量按分支因子相乘 |
| [MAPoRL](https://arxiv.org/abs/2502.18439)（2025-02） | 多个 LLM 先各自作答，再多轮讨论 | verifier 同时给答案和讨论打分，鼓励纠错与说服 | 多模型共训练 |
| [ReMA](https://arxiv.org/abs/2503.09501)（2025-03） | 高层元思考 agent + 低层推理 agent | 分层 MARL；多轮版本用 turn 级 ratio 与参数共享 | 两个角色共享参数以提高效率 |
| [SWEET-RL](https://arxiv.org/abs/2503.15478)（2025-03） | agent 与人类协作者多轮完成任务（ColBench） | 训练一个能看到训练期额外信息的 critic，给出 step 级奖励 | 额外 critic |
| [SPIRAL](https://arxiv.org/abs/2506.24119)（2025-06） | 零和游戏自博弈 | RAE：按游戏、按角色维护指数滑动平均基线；去掉 RAE 约 200 步后出现「思考崩塌」 | 在线多 agent 多轮 RL 系统 |
| [MAGRPO](https://arxiv.org/abs/2508.04652)（2025-08） | Dec-POMDP，写作与编码协作 | 每轮采 G 个联合样本，联合奖励减组均值（不除 std），分散执行 | 联合采样 G 份 |
| [AT-GRPO](https://arxiv.org/abs/2510.11062)（2025-10） | 游戏、规划、编码、数学 | 分组键 = (环境, agent, 轮次)；每轮每个 agent 采 K 个候选，执行奖励最高者（树形采样）；奖励 = α·团队 + 个体 | 每个策略一个独立资源池，池内按 HybridFlow 方式放一个 RolloutWorker 和一个 UpdateWorker；环境跑在 CPU 池 |
| [C3](https://arxiv.org/abs/2603.06859)（2026-03） | 数学与代码，两种多 agent 拓扑 | 固定历史重采该决策点的动作 + checkpoint 恢复后 rollout 到结束 + 留一法基线 | 论文报告 checkpoint 恢复避免重新生成历史前缀：418.4M token，对比 MAPPO 的 616.1M 少 32%，墙钟时间也少 32% |
| [CCPO](https://arxiv.org/abs/2603.21563)（2026-03） | Think → Solve 双 agent | Δ = R(联合) − R(Solver 单独作答)；冗余 agent 得 0，抑制 free-riding | 每个样本一条额外的「单独作答」rollout |

两篇 2026 年的综述可以当地图用：[信用分配综述](https://arxiv.org/abs/2604.09459)（覆盖 2024 年 1 月到 2026 年 7 月）和 [编排轨迹视角的多 agent RL 综述](https://arxiv.org/abs/2605.02801)（作者指出截至 2026 年 5 月，显式的消息级反事实信用仍很少，且没有找到针对「何时停止」决策的 RL 训练方法）。

## 7. 系统代价：反事实评估 = 额外 rollout

### 7.1 算账

以「每个训练样本（一次联合 episode）要多做几次评估」为单位：

| 方法 | 额外评估 | n=4 agent、batch 256 时总 episode 数 |
|---|---|---|
| 共享 / 组内基线 | 0 | 256 |
| CCPO（双 agent，移除一个） | 1 | 256 × 2 = 512（n=2） |
| difference / LOO | n | 256 × 5 = 1,280 |
| COMA（用仿真代替 critic） | n·(K−1) | K=4 时 256 × 13 = 3,328 |
| 精确 Shapley | 2ⁿ − 1 个其余联盟（每个联盟若要内层采样，再乘以采样数） | 256 × 16 = 4,096 |
| MC Shapley（M 个排列） | ≈ M·n | M=8 时 256 × 33 = 8,448，比精确的还贵 |
| DAPO Dynamic Sampling | 超额采样 1/(1 − pᴳ − (1−p)ᴳ) 倍 | p=0.05、G=8 时约 3 倍 |

如果每次评估都是一整条多轮 episode，这些倍数就是 rollout 算力的倍数。

### 7.2 与长尾的相互作用

1. **更多样本 → 更长的尾**：同步训练的 step time 由最慢的轨迹决定。Pareto(α) 尾下，N 个样本最大值的期望大约按 N^{1/α} 增长；α=1.5 时样本数变成 5 倍，最慢那条的期望长度约变成 5^{2/3} ≈ 2.9 倍。实验第 4 部分用 [模块 06](../06_long_tail_async/README.md) 的 `simulate_rollout` 给出模拟的 step time。
2. **信用组屏障**：一个样本的信用要等原始轨迹和它的全部 n 条反事实都跑完才能算，所以真正的单位是「信用组」而不是单条轨迹。异步系统里，组内任何一条因为过旧被中止，整组信用都要作废或补跑。而且同一任务的原始轨迹和反事实轨迹长度是正相关的（Seer 观察到同一 prompt 的多个样本输出长度相似，[Seer](https://arxiv.org/abs/2511.14617)），一组里往往一起长、一起拖尾。
3. **更长的依赖链**：结果级 difference reward 要等原始联合动作确定之后才能发起反事实；多轮任务里，关键路径变成「原始轨迹走到决策点 + 分支从决策点跑到结束」。
4. **缓解办法**：在决策点**立即** fork 分支；分支复用前缀（第 7.4 节）；只在关键决策点分支（预算化反事实，见 [模块 09](../09_frontier/README.md) 第 2.5 节）；用 critic 代替仿真（COMA），把 rollout 变成前向；异步计算信用，代价是信用本身也有了 staleness（数据陈旧度）。

### 7.3 环境状态的反事实（选读）

本模块开头的编码 agent，环境状态（文件系统、进程、数据库）不在文本历史里。C3 的前提是「交互历史就是可观测文本，没有隐藏状态」；它自己也指出，一般的部分可观测随机博弈里，固定历史需要一个能恢复隐藏环境状态的模拟器。三条路线：

| 路线 | 做法 | 代价 | 什么时候行 |
|---|---|---|---|
| 纯文本重放 | 把决策点之前的文本原样恢复（C3） | 只有推理算力 | 工具无副作用，或工具结果可以原样回放 |
| 沙箱快照 / fork | DSec 支持在任意时刻给沙箱做增量磁盘快照，再恢复成一个新沙箱；microVM 暂停时把内存与执行状态存成快照 | 每个分支点一次快照（磁盘增量，microVM 还要加上内存大小），分支数 × 并发沙箱数；DSec 论文没有公开快照延迟 | 状态在磁盘 / VM 里、可以完整捕获 |
| 记录与回放 | 记录每次工具调用的结果，回放时直接复用（DSec 早期版本在 GPU 任务被抢占后用命令日志对齐 rollout 状态，重放时已完成的操作复用记录结果，避免非幂等命令的重复副作用） | 日志存储；分支之后的新调用仍要真实执行 | 分支点之前的调用可以复用；工具不确定（网络、时间、随机数）时，回放出的状态和真实重跑不一样，会引入偏差 |

并发账用 Little 定律：并发沙箱数 = 创建速率 × 平均寿命。每个样本多出 n 个分支，并发沙箱数大约乘以 (1 + n × 分支存活时间 / 原始存活时间)。DSec 在生产中支撑超过 38 万个并发沙箱（[DSec](https://arxiv.org/abs/2609.22978)；平台细节见 [模块 03 第 5.3 节](../03_deepseek_stack/README.md#53-dsec智能体-rl-的沙箱平台)，沙箱规模化见 [模块 06 第 8 节](../06_long_tail_async/README.md#8-代价四沙箱规模化环境即服务)）。

### 7.4 KV fork 的账（选读）

把决策点之前的历史分成外部 token（prompt、工具结果）H_ext 和模型生成的 token H_gen，每条分支在决策点之后追加外部 token A_ext、自己生成 B_gen：

| 做法 | 每条分支的 prefill | 每条分支的 decode | 例子（20k / 3k / 2k / 1k，4 条分支） |
|---|---|---|---|
| 从头重跑 | H_ext + A_ext | H_gen + B_gen | prefill 88k、decode 16k |
| checkpoint 恢复（C3） | H_ext + H_gen + A_ext | B_gen | prefill 100k、decode 4k |
| KV fork（前缀命中） | A_ext | B_gen | prefill 8k、decode 4k |

- checkpoint 恢复省的是历史的 **decode**，KV fork 再省掉历史的 **prefill**；分叉之后追加的工具结果和其他 agent 的消息，无论哪种做法都要 prefill。
- KV fork 的前提：分支被路由到持有这份前缀 KV 的同一实例，而且在所有分支发出前这份 KV 没被淘汰（[模块 05](../05_cache_prefill/README.md) 的 cache affinity 与 TTL）。
- 显存：paged / radix 缓存里各分支共享前缀块，只有分叉后的块是每条分支独占，占用约为 kv_bytes/token × (前缀长度 + Σ 各分支后缀长度)；而且要一直占到信用组屏障解除。练习 4 实现这张账。

### 7.5 多个 agent 用几个模型、怎么放（选读）

| 维度 | M 个独立模型 | 一个共享模型 + 角色 prompt |
|---|---|---|
| 权重同步 | 每步 M × 参数量 × 2 字节（BF16）：3 × 7B = 42 GB | 1 × 参数量 × 2 字节：14 GB |
| 同步耗时下限 | 走一张 400 Gb/s（50 GB/s）网卡约 0.84 s；节点内按 H100 的 900 GB/s NVLink 约 0.05 s | 约 0.28 s / 0.016 s |
| 训练侧 | 每个策略各自的优化器状态与训练 worker；AT-GRPO 给每个策略一个独立资源池（RolloutWorker + UpdateWorker） | 一份优化器状态；各角色的梯度混在一起，可能互相干扰 |
| 推理侧 KV | 不同模型之间不能共享 KV | 把共享的 system prompt、工具定义放在最前、角色说明放在后面，各角色可以共享前缀 KV |
| loss 与信用 | 每个模型只训练自己的 token | 同一条多角色 transcript 里，每个角色的 token 用自己的优势，其他角色的 token 作为上下文、loss mask 置 0：优势必须是逐 token 的张量，不能是逐序列的一个标量 |
| on-policy | 每个策略按自己的版本号追踪 staleness | 一个版本号 |

权重同步的量级与工程手段（分桶、delta 同步、与训练重叠）见 [模块 06 第 7 节](../06_long_tail_async/README.md#7-代价三部署配比与权重同步)。

### 7.6 值不值：方差与吞吐的盈亏判据

在梯度噪声主导的区间，达到同样进度需要的样本数大约与每个样本的梯度方差（按信号大小归一化，即信噪比的倒数）成正比；时间 = 样本数 / 每秒有效样本数。于是

**T_方法 / T_基线 ≈ (SNR_基线 / SNR_方法) / (q_方法 / q_基线)**

其中 SNR 是单样本梯度信噪比，q 是每小时有效（原始）轨迹数。只有**信噪比提升的倍数大于有效吞吐下降的倍数**时，反事实信用才在墙钟时间上赚。把实验第 1、4 部分的数字代进去：

| 估计器（每个反事实都是一次真实 rollout） | 信噪比提升 | 同步：有效吞吐比 → T 比值 | 异步：有效吞吐比 → T 比值 |
|---|---|---|---|
| difference（+n） | 3.0× | 0.50× → 0.66（赚） | 0.18× → 1.81（亏） |
| COMA 式基线，用仿真算反事实（+n(K−1)） | 5.9× | 0.25× → 0.69（赚） | 0.07× → 2.46（亏） |
| 精确 Shapley，按策略参考（+2ⁿ−1） | 8.9× | 0.10× → 1.17（亏） | — |

（n=6、K=4，数字来自实验第 1、4、5 部分。）同步系统有长尾气泡可填，额外 rollout 打了折；异步系统没有气泡，反事实按全价收费，这时只有把 rollout 换成 critic 前向（COMA 的本意）或把单次反事实的成本压下来（KV fork、共同随机数）才可能赚。注意这是玩具任务上、在一个策略点测出的信噪比，只用来演示判据和数量级，不是对真实系统的预测；真实系统里要在自己的负载上重测两边的倍数。

## 8. 必须会心算的数字

| 量 | 典型值 | 怎么推出来 | 来源 |
|---|---|---|---|
| 0/1 奖励的方差 | p(1−p)，最大 0.25（p=0.5） | 伯努利分布 | 定义 |
| 基线后奖励乘子二阶矩 / 不用基线 | 真实均值 1 − p；RLOO (1 − p)·G/(G − 1)；p=0.05、G=8 时分别为 0.95、1.086 | 估出来的基线多出 p(1−p)/(G−1) 的方差 | 本模块推导 |
| 零优势组的比例（平均成功率 0.05、G=8） | 各题同为 0.05 → 66%；Beta(1, 19) → 70%；Beta(0.25, 4.75) → 77% | E[pᴳ + (1−p)ᴳ]，对 Beta 分布有闭式 | 本模块推导 |
| DAPO 式超额采样倍数 | 上面三种情况分别约 3.0×、3.4×、4.4× | 1 / (1 − 零优势比例) | 本模块推导 |
| RLOO 与减组均值 | 样本值差 G/(G−1)（G=8 → 1.14）；期望梯度分别为 ∇J 与 (G−1)/G·∇J | 恒等式（练习 1） | 本模块推导 |
| 精确 Shapley 联盟数 | 2ⁿ：n=4 → 16，n=8 → 256，n=16 → 65,536 | 子集个数 | 定义 |
| MC Shapley 何时划算 | M·n < 2ⁿ：n=4 时 M ≤ 3，n=10 时 M ≤ 102 | 比较评估次数 | 本模块推导 |
| difference / LOO 总评估倍数 | n + 1 | 1 次原始 + n 次反事实 | [COMA](https://arxiv.org/abs/1705.08926) |
| 重尾下「最慢一条」随样本数增长 | N^{1/α}；α=1.5、5× 样本 → 约 2.9× | Pareto 极值 | 本模块推导 |
| 3 个 7B 独立策略每步权重同步 | 42 GB；50 GB/s 网卡下限约 0.84 s | 3 × 7B × 2 字节 | 本模块推导 |
| VinePPO 每步变慢 | 最多 5×（RhoMath 1.1B）、2×（DeepSeekMath 7B） | K 条 MC rollout / 状态 | [VinePPO](https://arxiv.org/abs/2410.01679) |

## 9. 实验

**怎么跑**：`make lab N=07`（或 `.venv/bin/python modules/07_multi_agent_credit/lab.py`），M1 笔记本上约 5 秒，结果完全确定。

**任务是什么**：第 0 部分是纯 numpy 的伯努利奖励。第 1–4 部分用 `make_team_task` 生成的「已知真实贡献」的合作博弈：n 个 agent 各选一个动作，团队只拿到一个标量 r = Σᵢ W[i, aᵢ] + Σ_{i<j} Q_ij(aᵢ, aⱼ) + 噪声（个体项 + 两两协同项，噪声标准差 0.5）。几个口径：

- **oracle cf**：反事实查询拿到无噪声的 f，相当于一个完美的模拟器或 critic（库的默认）。
- **noisy cf**：每次反事实查询都是一次独立的带噪 rollout（`cf_noise="independent"`），这才是「真的去重跑」。
- **CRN cf**：反事实与实际 rollout 共享同一份外生噪声（`cf_noise="common"`，共同随机数）。
- **Shapley uniform / policy**：缺席的 agent 按均匀随机动作 / 按当前策略取期望（`shapley_ref`）。库里的 Shapley 用无噪声的联盟价值、并把观测噪声除以 n，这是 oracle 口径；`rollouts` 两行由 lab 自己算：每个联盟只用**一次带噪 rollout** 估计，和 noisy cf 同一口径。
- `extra evals` 是如果这些查询要用真实 rollout 实现，每个样本要多花几次评估。

**真实输出**：

```text
== 0. 同一 prompt 采 G=8 个样本的组内基线（纯 numpy，20 万组）
-- (a) 奖励乘子的二阶矩 / 不用基线时的二阶矩
     p  true mean  RLOO (MC)  RLOO theory  mean incl. self
  0.05      0.950      1.086        1.086            0.832
 0.125      0.875      1.000        1.000            0.766
   0.5      0.500      0.571        0.571            0.437
   0.9      0.100      0.114        0.114            0.087
-- (b) 平均成功率都是 0.05 时，零优势（全对/全错）组的比例与超额采样倍数
difficulty distribution    zero-adv (MC)  closed form  oversample
all p = 0.05                       0.663        0.663       2.97x
Beta(1, 19)                        0.704        0.704       3.38x
Beta(0.25, 4.75)                   0.770        0.772       4.38x

== 1. 训练中途（非均匀策略）的单样本梯度（n=6 agents, K=4 actions）
estimator                  SNR/sample  vs shared  cos(E[g],∇J)  rel.err  extra evals
shared                         0.0182       1.0x         1.000    0.000            0
group mean (incl. self)        0.0414       2.3x         1.000    0.125            0
group LOO                      0.0414       2.3x         1.000    0.000            0
COMA, oracle cf                0.1193       6.6x         1.000    0.000            0
COMA, noisy cf                 0.1063       5.9x         1.000    0.000           18
difference, oracle cf          0.0635       3.5x         1.000    0.000            0
difference, noisy cf           0.0547       3.0x         1.000    0.000            6
difference, CRN cf             0.0752       4.1x         1.000    0.000            6
Shapley uniform, oracle        0.3337      18.4x         0.979    0.203            0
Shapley policy, oracle         0.2252      12.4x         1.000    0.000            0
Shapley uniform, rollouts      0.1729       9.5x         0.979    0.203           63
Shapley policy, rollouts       0.1616       8.9x         1.000    0.000           63
   (group mean 的 rel.err 按 batch=8 计：均值含自身 → 期望梯度缩小 (B-1)/B)

== 2. REINFORCE 训练 150 轮（n=6, K=4, batch=8, lr=0.3；20 个 seed，均值 ± 标准误）
estimator                grad_var(first 50)      iters to 80%    at 75 / best     at 150 / best
shared                                1.604       45.3 ±  5.1    0.854 ± 0.020      0.906 ± 0.016
group LOO                             0.403       37.8 ±  2.8    0.900 ± 0.010      0.947 ± 0.008
COMA, noisy cf                        0.133       35.5 ±  3.2    0.911 ± 0.012      0.961 ± 0.007
difference, noisy cf                  0.186       37.5 ±  4.6    0.915 ± 0.013      0.958 ± 0.008
Shapley uniform                       0.041       51.2 ±  8.5    0.851 ± 0.024      0.898 ± 0.021
Shapley policy                        0.085       35.0 ±  2.8    0.909 ± 0.014      0.957 ± 0.007

== 3. agent 数变大：早期梯度方差（前 30 轮平均，K=4，batch 64，3 个 seed）
n_agents          shared     group LOO    difference   shared/difference
2                  0.253         0.193         0.267                0.9x
4                  0.398         0.348         0.270                1.5x
6                  0.732         0.605         0.300                2.4x
12                 4.362         2.219         0.542                8.0x

== 4. 反事实评估的 rollout 代价（n=6 agents；模块 06 模拟器默认参数：64 workers，Pareto α=1.5，5 步）
setting                          trained/step step_time_s  busy  stale  wasted  useful/h  vs shared
shared, sync                              256         651  0.13   0.00   0.000      1415      1.00x
difference (+n), sync                    1792        1298  0.43   0.00   0.000       710      0.50x
COMA by sim (+n(K-1)), sync              4864        2617  0.58   0.00   0.000       352      0.25x
exact Shapley (+2^n-1), sync            16384        6794  0.75   0.00   0.000       136      0.10x
shared, async                             256         134  0.60   0.19   0.000      6872      1.00x
difference (+n), async                   1792         731  0.76   0.03   0.000      1260      0.18x
COMA by sim (+n(K-1)), async             4864        1932  0.78   0.01   0.000       477      0.07x

== 5. 盈亏判据：T_方法 / T_shared ≈ (SNR_shared / SNR_方法) / (有效吞吐比)，< 1 才赚
estimator (real rollouts)     SNR gain  sync thr   T sync  async thr  T async
difference, noisy cf              3.0x     0.50x     0.66      0.18x     1.81
COMA, noisy cf                    5.9x     0.25x     0.69      0.07x     2.46
Shapley policy, rollouts          8.9x     0.10x     1.17          —        —
```

**怎么读**：

0. **第 0 部分（组内基线）**：蒙特卡洛与理论一致。p < 1/G 时 RLOO 的二阶矩比值大于 1（p=0.05 时 1.086，p=1/8 时恰好 1.000）；「减含自己的均值」看起来更小（0.832），但那是有偏的收缩。平均成功率同样是 0.05，难度越不均匀，零优势组越多（0.663 → 0.704 → 0.770），超额采样倍数从 2.97× 涨到 4.38×。
1. **第 1 部分（梯度质量）**：
   - 所有「边际型」估计器的期望梯度都与真梯度同向（cos = 1.000）；「减含自己的组均值」的相对误差正好是 0.125 = 1/8，也就是 batch=8 时 (B−1)/B 的整体收缩，方向不变。
   - 信噪比：group LOO 是 shared 的 2.3×；difference 在 oracle 口径下 3.5×，换成每次反事实一次独立带噪 rollout 降到 3.0×，用共同随机数升到 4.1×；COMA 式基线从 6.6× 降到 5.9×。
   - Shapley：oracle 口径的 18.4× / 12.4× 不公平（联盟价值无噪、噪声还被除以 n）；每个联盟一次带噪 rollout 时是 9.5× / 8.9×，代价是 63 次额外评估。**以均匀随机队友为参考时 cos = 0.979、相对误差 0.203，有偏；以当前策略为参考时 cos = 1.000，无偏**，和第 4.4 节的推导一致。
2. **第 2 部分（学习速度，20 个 seed）**：
   - 反事实类（COMA 35.5、difference 37.5 轮）和按策略参考的 Shapley（35.0 轮）走完 80% 差距最快，150 轮后都到最优的约 0.96；group LOO 居中（37.8 轮、0.947）；shared 最慢（45.3 ± 5.1 轮、0.906）。
   - Shapley uniform 方差最低（0.041），却学得最慢、最终最差（51.2 ± 8.5 轮、0.898）：**低方差换不来有偏的方向**。
   - COMA 的前期梯度方差比 difference 低（0.133 vs 0.186），但学习速度的差别（35.5 ± 3.2 vs 37.5 ± 4.6）在标准误之内：两者期望梯度相同，噪声降到这个量级后，进度主要受学习率和期望梯度限制，再降方差收益递减。少量 seed 下看到的「方差更低反而更慢」只是 seed 噪声。
   - 第 75 轮到第 150 轮所有方法的回报都还在涨（例如 difference 0.915 → 0.958），final/best 小于 1 说明 150 轮还没训完，不是停在局部最优。
3. **第 3 部分（agent 数）**：shared 的梯度方差从 n=2 的 0.253 涨到 n=12 的 4.362（agent 数 6 倍、方差约 17 倍，超线性）；difference 也在涨（0.267 → 0.542），因为和自己有关的协同项消不掉。n=2 时 difference 反而比 shared 略差（0.9×）：默认动作选得不好时，W[i, aᵢ] − W[i, 0] 本身就是方差来源；n=12 时 shared 是 difference 的 8.0 倍。
4. **第 4 部分（系统代价）**：把每个反事实当成一条完整的额外轨迹，放进 [模块 06](../06_long_tail_async/README.md) 的 rollout 模拟器（`useful/h` 只数原始 episode）；所有设置都是同样 64 个单位、共置部署，属于等资源比较。
   - 同步：difference 多跑 7 倍 rollout，有效吞吐降到 0.50×；COMA 式仿真 19 倍 → 0.25×；精确 Shapley 64 倍 → 0.10×。多出来的 rollout 有一部分填进了长尾气泡（busy 从 0.13 升到 0.43 / 0.58 / 0.75），所以代价低于线性。这与 DAPO 原文的观察一致：同步系统里生成时间由长尾主导，多采样不一定拖慢训练。
   - 异步：rollout 槽位本来就忙（0.60），反事实接近全价：difference 0.18×、COMA 式仿真 0.07×。n=6 时，即使先把系统做成异步，difference 的有效吞吐（1260 条/小时）也低于共享奖励的同步（1415）。
   - 大 batch 让单位时间的版本数变少，所以 difference 在异步下的平均 staleness（0.03）反而比 shared（0.19）低；但信用本身的陈旧度没有被统计。
   - 模拟器偏乐观的地方：反事实与原始轨迹同时发起、互相独立；没有建模信用组屏障（一组 n+1 条都完成才能算信用，组内任一条被中止就整组作废）、同一任务原始与反事实轨迹长度的正相关，以及分支复用前缀 KV 带来的节省。
5. **第 5 部分（盈亏判据）**：用「每个反事实都是一次真实 rollout」的信噪比：difference 在同步下 T 比值 0.66（快约三分之一），在异步下 1.81（慢 81%）；COMA 式仿真同步 0.69、异步 2.46；精确 Shapley 即使在同步下也是 1.17，不划算。结论：**真实重跑的反事实信用只在「有气泡可填」的同步系统里划算**；异步系统要么用 critic 前向（COMA 的本意）把 rollout 换成前向，要么靠 KV fork、共同随机数等把单次反事实的成本和噪声压下来。

**改哪个参数看什么**（以下现象都实际跑过）：

- `part2_training` 里把 batch 从 8 改成 2：shared 走完 80% 要 103.6 ± 12.9 轮，difference 57.2 ± 10.3 轮，差距拉大。
- `part1_gradient_quality` 里 `make_team_task(..., pair_scale=1.0)`：协同项变强，Shapley uniform 的 cos 从 0.979 降到 0.969。
- `part4_system_cost` 里给 `RolloutConfig` 加 `tail_alpha=3.0`（尾巴变轻）：同步模式下 difference 的有效吞吐比从 0.50 降到 0.26（气泡少了，反事实更接近全价）；异步模式基本不变（0.18 → 0.19）。

## 10. 练习

在 [`exercise.py`](exercise.py) 里实现四个函数（都是纯 numpy，不依赖模拟器）：

| 函数 | 要实现什么 | 考点 |
|---|---|---|
| `group_advantages` | GRPO / 减组均值 / RLOO 三种组内优势，std 为 0 的组返回 0，支持任意前导维度 | 无偏性、数值稳定、恒等式 RLOO = G/(G−1) × 减组均值 |
| `difference_rewards` | Dᵢ = R(a) − R(a₋ᵢ, c)，只准调用 n+1 次奖励函数 | 反事实评估的成本 |
| `exact_shapley` | 精确 Shapley 值，每个联盟只求值一次 | 效率、对称、零贡献者；手套博弈 |
| `branch_token_cost` | 重跑 / checkpoint 恢复 / KV fork 三种分支方式的 prefill 与 decode token 账 | 系统侧：C3 省 decode、KV fork 省 prefill，追加内容都要 prefill |

```bash
make module N=07                                                 # 测你的实现（没写完会失败，正常）
INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/07_multi_agent_credit   # 看参考答案能否通过
```

## 11. 自测题

1. 成功率 p=0.05、G=8 时，为什么 RLOO 基线不但不省方差、反而让奖励乘子的二阶矩变大？这时真正该做什么？
2. 「减组均值」和 RLOO 的样本值只差一个常数，为什么一个无偏、一个有偏？GRPO 再除以 std 之后呢？
3. 共享奖励下，为什么 agent 越多，每个 agent 的梯度方差越大，而且在有协同项时是超线性的？difference reward 能完全解决吗？
4. 三个 agent 必须都做对才成功。LOO 和 Shapley 分别给每个人多少信用？哪个满足「总和等于团队价值」？
5. 在什么条件下，把 Shapley 值当作策略梯度的信用是无偏的？用三步推导说明。
6. 为什么在 LLM 多 agent 系统里「把某个 agent 移除」来算贡献会有偏？C3 怎么绕开？编码 agent 里 C3 的前提还成立吗？
7. 一个 difference reward 实现把反事实评估用独立的随机种子重跑。和用同一个种子相比，信用的方差会怎样变？
8. 反事实信用的单样本信噪比是共享奖励的 3.0 倍，同步系统里有效吞吐降到 0.50 倍、异步系统里降到 0.18 倍。分别值不值？

<details><summary>答案</summary>

1. RLOO 的基线是从其余 G−1 个样本估出来的，自带 p(1−p)/(G−1) 的方差，二阶矩比值 (1−p)·G/(G−1) = 0.95 × 8/7 ≈ 1.086 > 1；p < 1/G 时都是这样。真正的困难是几乎没有正样本：增大 G、做难度课程或 DAPO 式超额采样过滤全错组，或用过程奖励 / 中间奖励让信号变稠密。
2. 组均值里含 rᵢ/G，而 rᵢ 与 aᵢ 相关，所以 E[(rᵢ − mean)∇log π(aᵢ)] = (1 − 1/G)∇J：方向对、大小被缩小；RLOO 的基线不含样本自己，与 aᵢ 独立，期望正好是 ∇J。GRPO 的 std 也由整组动作（包括 aᵢ）决定，不再是常数缩放，一般有偏，并按题目重新加权（std 小的题权重大）。
3. 共享奖励 R 里包含所有 agent 的贡献，每多一个 agent 就多一项一元方差，还要多 n−1 个与它有关的协同项，协同项总数 n(n−1)/2，所以超线性。difference 只能消掉不涉及 i 的项，和 i 有关的协同项仍在，所以它的方差也随 n 缓慢上升。
4. LOO：每人 v(N) − v(N∖{i}) = 1，总和 3 ≠ 1；Shapley：每人 1/3，总和 1。Shapley 满足效率性。
5. 当联盟价值按当前策略边缘化缺席者，即 v(S) = E_{a'₋S∼π}[f(a_S, a'₋S)] 时无偏：(1) i ∉ S 的项不依赖 aᵢ，期望贡献为 0；(2) i ∈ S 的项满足 E[v(S)|aᵢ] = Qᵢ(aᵢ)，贡献 ∇ᵢJ；(3) Shapley 权重之和为 1。按均匀、默认动作或移除定义缺席都会有偏。
6. 移除 agent 会改变下游 agent 的输入分布，差值里混入了下游在陌生输入下的退化，这是加样本也消不掉的偏差。C3 固定完整历史，只在该决策点重采这个 agent 的动作并 rollout 到结束，用留一法基线比较，前提是历史能精确恢复。编码 agent 的环境状态不在文本里，需要沙箱快照 / fork 或记录回放（第 7.3 节），否则前提不成立。
7. 独立种子时，反事实那次评估自带一份独立的观测噪声，差分后方差多出约一份噪声方差 σ²；同一种子（共同随机数）时两次评估的噪声相关，差分时大部分抵消。
8. 按第 7.6 节的判据：同步 (1/3.0)/0.50 ≈ 0.67，墙钟时间约少三分之一，值；异步 (1/3.0)/0.18 ≈ 1.85，反而慢约 85%，不值（实验第 5 部分用未取整的数得到 0.66 和 1.81）。异步系统没有气泡可填，反事实按全价收费，信噪比要提升到 1/0.18 ≈ 5.6 倍以上才划算。

</details>

## 12. 面试追问

1. **「GRPO 为什么能去掉 value 模型？代价是什么？」**
   结论：用同一 prompt 的 G 个样本的均值（和 std）当基线，省掉了与 policy 同规模的 critic。数量级：DeepSeekMath 每题 64 个样本；平均成功率 0.05、G=8 时零优势组占 66%–77%（看难度分布）。取舍：rollout 量乘以 G；信用只到轨迹级；减含自己的均值把梯度缩小 (G−1)/G，除 std 带来难度偏置，1/\|o\| 带来长度偏置（Dr. GRPO）；想严格无偏就用 RLOO。
2. **「多 agent 共享一个结果奖励，你怎么做信用分配？」**
   结论：先用组内基线（零额外成本），方差不够再上反事实。数量级：difference / LOO 要 n 倍评估，精确 Shapley 要 2ⁿ − 1 个联盟，COMA 是一次 critic 前向。取舍：环境能否重置、历史能否精确恢复（纯文本可以，编码 agent 要沙箱快照）、critic 能否学准；Shapley 要按当前策略边缘化缺席者才无偏；移除式反事实在 LLM 中有分布偏移。
3. **「反事实 rollout 会把训练吞吐拖垮吗？怎么设计系统？」**
   结论：会，而且落在长尾关键路径上，还有信用组屏障；代价大小取决于系统模式。数量级：n=6 时 difference 让 rollout 变 7 倍；实验里同步模式有效吞吐降到 0.50 倍（反事实部分填了气泡），异步模式降到 0.18 倍（接近全价）；按信噪比 3 倍算，同步快约三分之一、异步慢约八成。取舍：用「信噪比提升 vs 吞吐下降」判断；决策点 fork + 前缀 KV 复用、预算化分支、异步信用（信用有 staleness）、critic 代替仿真（有偏差风险）。
4. **「3 个 agent，用 3 个 7B 还是 1 个 7B？rollout 和 update 怎么放？」**
   结论：先问角色之间是否需要不同的能力、训练是否要独立节奏。数量级：3 个独立 7B 每步同步 42 GB，50 GB/s 网卡下限约 0.84 s；共享模型 14 GB。取舍：独立模型各自一个资源池（AT-GRPO 的 RolloutWorker + UpdateWorker），显存、同步、优化器状态都乘以 3，但梯度互不干扰；共享模型能共享前缀 KV，但要逐 token 的优势和按角色的 loss mask，梯度可能互相干扰。
5. **「用 LLM judge 做奖励，最担心什么？怎么监控？」**
   结论：reward hacking 与 judge 偏差。数量级：Gao 等人的拟合里真实奖励随 √KL 先升后降；单个「万能钥匙」token 就能骗出正奖励。取舍：rubric / checklist 更稳但更贵（每条 rubric 一次调用）；KL 约束限制探索；要监控 judge 分与可验证分的背离，并定期金标准评估。
6. **「多轮 agent 的 step 级优势怎么估，不想多跑 rollout？」**
   结论：GiGPO 的 anchor state 分组、GACA（复用 rollout 已记录的 NLL）几乎零额外 rollout；TRACE 只多一次冻结参考模型的前向。数量级：VinePPO 式 MC 每步慢 2–5 倍；树形 rollout 靠前缀摊销能省 22%–43% GPU 小时（TreePO）。取舍：anchor state 要求状态会重复出现（ALFWorld、WebShop 这类环境可以，开放网页未必）；另外要先过「均匀分配」对照，稠密未必要「瞄准」。

## 13. 延伸阅读

- [COMA](https://arxiv.org/abs/1705.08926)：反事实基线的出处，也是对 difference reward 代价最清楚的一段论述。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的原始公式，含过程监督版本。
- [Back to Basics（RLOO）](https://arxiv.org/abs/2402.14740)与 [Dr. GRPO](https://arxiv.org/abs/2503.20783)：组内基线的两个关键细节（无偏性、归一化偏置）。
- [VinePPO](https://arxiv.org/abs/2410.01679)、[GiGPO](https://arxiv.org/abs/2505.10978)、[TreePO](https://arxiv.org/abs/2508.17445)：step 级信用的「多花 rollout」「不多花 rollout」「摊销前缀」三条路线。
- [Coverage, Not Targeting](https://arxiv.org/abs/2609.02417)：做 turn 级信用研究前必读的对照设计。
- [MAGRPO](https://arxiv.org/abs/2508.04652)、[AT-GRPO](https://arxiv.org/abs/2510.11062)、[C3](https://arxiv.org/abs/2603.06859)：多 agent LLM RL 从「共享组内基线」到「精确反事实」的三步。
- [DSec](https://arxiv.org/abs/2609.22978)：反事实分支需要环境快照时的一手系统参考。
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)、[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)：reward hacking 的定量与实例。
- [信用分配综述（2026）](https://arxiv.org/abs/2604.09459)：按粒度与方法分类的最新地图。
- 全项目阅读清单见 [docs/reading-list.md](../../docs/reading-list.md)；本方向的开放问题见 [模块 09](../09_frontier/README.md)。
