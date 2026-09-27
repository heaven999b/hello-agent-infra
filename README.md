<div align="center">

# 👋 Hello Agent Infra

### Harness × Inference 协同设计：20 小时速成

**从「会写 Agent」到「能和大模型 infra 工程师对等讨论」：10 个模块 + 1 个真实迷你引擎 · 中英双语 · 每个实验都能在笔记本上跑。**
推理的第一性原理、推理引擎、DeepSeek 系统栈、Agent 负载画像、缓存与 prefill、长尾与异步 RL、多 Agent 信用分配、系统设计与面试。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/deps-numpy%20only-brightgreen)
![Hardware](https://img.shields.io/badge/runs%20on-M1%208GB%20laptop-lightgrey)
![Built with Claude Opus 5.5](https://img.shields.io/badge/built%20with-Claude%20Opus%205.5-D97757)

本项目由 **Claude Opus 5.5** 协助架构设计

**中文** · [English](README.en.md)

[快速开始](#-快速开始) · [20 小时路线](#-20-小时路线) · [综合实战](capstone/README.md) · [面试题](docs/interview-questions.md) · [心算数字](docs/numbers-cheatsheet.md) · [阅读清单](docs/reading-list.md)

</div>

---

## 🤔 为什么要做这个项目？

Agent 的 harness（上下文组装、压缩、记忆注入、工具调用、多 Agent 编排）和推理/训练系统，过去是两拨人分别优化的。但到了 coding agent 和 agentic RL 的规模，两边的决策开始互相绑架：

- harness 为了省上下文做了一次**压缩改写**，推理引擎的**前缀缓存**就整段失效，只能重新 prefill；
- agent 在**等工具返回**的几秒到几分钟里，它的 KV cache 是留在显存里占位，还是丢掉以后重算？
- 多个项目、成千上万个 session 共享一组 GPU，**按缓存路由**能省 prefill，却会把负载压到少数实例上；
- agentic RL 的 rollout 有**长尾**：一条轨迹跑 20 轮、另一条跑 2 轮，同步训练时整批都在等最慢的那条；
- 多个 agent 合作完成一个任务，只有一个最终奖励，**功劳怎么分**，分的时候又要多跑多少 rollout？

这些问题只站在一边是解不好的，这就是 **harness–inference co-design（协同设计）**。这个项目用 20 小时，把你需要的系统知识、数量级直觉、动手实验和面试表达一次补齐。

> 🎯 **诚实的目标**：20 小时不会让任何人变成资深工程师——那需要几年的线上规模与事故。它能做到的是：让你能用 DeepSeek / Qwen / Seed 这类团队的语言、数量级和取舍讨论问题，扛得住面试深挖，并且手里有一个能跑、能讲的项目。

### 先修要求

- 能读懂 Python 和 numpy，知道 Transformer 的前向由哪几块组成（注意力、MLP、残差）。
- 知道策略梯度和 PPO / GRPO 的基本形式（第 6、7 模块会用到）。
- 不需要 CUDA 或 GPU 经验。

### 学完的验收清单

20 小时后，下面每一条你都应该能在白板上独立完成（每个模块的自测题和面试追问就是在练这些）：

1. 给定模型结构和 GPU 型号，心算每 token 的 KV 字节、单卡最大并发、prefill 与 decode 各自的瓶颈，以及它们在什么 batch / 上下文下会翻转。
2. 说清 continuous batching、PagedAttention、前缀缓存、chunked prefill、投机解码、PD 分离各自解决什么、代价是什么、什么时候不划算。
3. 讲出 DeepSeek 推理系统的关键设计（MLA、大规模 EP、PD 分离、硬盘 KV 缓存），并用公开数字反推每张卡的工作点。
4. 给定一个 agent 负载，判断 harness 的哪些行为会让前缀缓存失效，以及工具等待期间 KV 该留、下放还是重算。
5. 推导同步 RL rollout 的长尾气泡，说清 partial / 异步 rollout 的收益与代价（staleness、off-policy 修正、训推不一致、权重同步）。
6. 比较多 agent 信用分配的几类方法，并算出每种方法额外需要多少 rollout。
7. 用七步框架在 45 分钟内完成一道 LLM infra 系统设计题。

### 事实来源的约定

讲义里的每个硬件规格、模型参数、论文结论和公司系统细节，都链接到一手来源（论文、官方技术报告、官方仓库或文档）。**推断**会明确写成推断；无法核实的写「未公开」或「据报道」；**课程自己的抽象**（比如模拟器的简化假设）写在对应的 API 文档里。面试时请按这三类区分着讲。

## ✨ 你会得到什么

| | |
|---|---|
| 🧮 **数量级直觉** | 每个模块都有「必须会心算的数字」：KV cache 每 token 多少字节、decode 为什么是访存瓶颈、一个 H800 节点能撑多少并发 |
| 🧪 **能跑的实验** | `codesign_sim`：纯 numpy 的推理/训练系统模拟器（roofline 成本模型、前缀缓存、continuous batching、多实例路由、同步/异步 rollout、信用分配），每个实验 10 秒内出结果 |
| ⚙️ **真实实现** | `mini_engine`：真正执行 Transformer 前向的迷你推理引擎——分块 KV、前缀缓存、continuous batching、chunked prefill、抢占、asyncio 并发 agent 前端；差分测试（含随机生成的配置）证明各项优化开关前后，greedy 输出逐 token 一致 |
| 📝 **带测试的练习** | 每个动手模块都有 `exercise.py` + 测试，写完 `make module N=05` 就知道对不对 |
| 🎤 **面试弹药** | DeepSeek 风格的深挖追问、5 道完整的系统设计演练、60 道左右的题库，以及「怎么讲你自己的 co-design 项目」 |

## 🗺 20 小时路线

| # | 模块 | 时长 | 你会搞懂 | 动手 |
|---|---|---|---|---|
| 0 | [全景图](modules/00_map/README.md) | 0.5h | harness / 推理引擎 / 训练系统 / 硬件四层如何互相转嫁成本 | — |
| 1 | [推理的第一性原理](modules/01_inference_basics/README.md) | 2h | prefill vs decode、roofline、KV cache 公式、TTFT/TPOT/goodput | roofline 与 KV 计算器 |
| 2 | [现代推理引擎](modules/02_inference_engines/README.md) | 2h | continuous batching、PagedAttention、前缀缓存、chunked prefill、投机解码、量化、并行、PD 分离 | 引擎模拟：chunked prefill 与 token 预算 |
| 3 | [DeepSeek 系统栈](modules/03_deepseek_stack/README.md) | 2.5h | MLA、DeepSeekMoE、MTP、FP8、DualPipe、DeepEP、EPLB、3FS、PD 分离 + 大规模 EP 推理 | MLA vs GQA 显存、EP 负载均衡 |
| 4 | [Agent 负载画像](modules/04_harness_workloads/README.md) | 1.5h | agent 负载和聊天负载有什么不同、怎样的 harness 对缓存友好 | harness 策略 vs 前缀命中率 |
| 5 | [Cache × Prefill 协同设计](modules/05_cache_prefill/README.md) | 2.5h | 多项目缓存争用、工具等待期间的 KV 去留、缓存路由 vs 负载均衡、KV 分层、program 级调度 | 多实例路由与缓存策略对比 |
| 6 | [长尾与异步](modules/06_long_tail_async/README.md) | 2.5h | rollout 长尾、partial / 异步 RL、staleness 与 off-policy 修正、训推不一致、权重同步 | 同步 / partial / 异步 rollout 对比 |
| 7 | [多 Agent 信用分配](modules/07_multi_agent_credit/README.md) | 1.5h | 组内基线、difference reward、反事实基线、Shapley、reward hacking | 各估计器的方差与学习速度 |
| 8 | [系统设计与面试](modules/08_system_design/README.md) | 2h | 答题框架、5 道完整演练、怎么讲你的项目 | — |
| 9 | [前沿与开放问题](modules/09_frontier/README.md) | 0.5h | 未解决的问题、RSI 对基础设施的含义、怎么把问题做成论文 | — |
| ★ | [综合实战：迷你推理引擎](capstone/README.md) | 2.5h | 读懂并扩展一个真实执行的引擎 + 异步 agent 前端 | 会话感知淘汰、缓存路由等进阶任务 |

**合计 20 小时。** 建议节奏：4 天 × 5 小时，或者每天 2 小时连续 10 天。第 5、6、7 模块是 co-design 的核心，时间紧就优先保证它们。

## 🚀 快速开始

运行只需要 Python 3.10+ 和 numpy；跑测试另需 pytest（`make setup` 会一起装）。命令用到 `git` 和 `make`（macOS 上由 Xcode Command Line Tools 提供）。没有 GPU 也能跑：本项目在 Apple M1 / 8GB、Python 3.14、numpy 2.5 上开发和测试，「每个实验 10 秒内」指的是在这台机器空闲时单独运行一个 `lab.py` 的墙钟时间。

```bash
git clone https://github.com/heaven999b/hello-agent-infra.git
cd hello-agent-infra
make setup            # 创建 .venv，安装 numpy + pytest
make lab N=01         # 跑模块 1 的实验
make module N=05      # 跑模块 5 的练习测试（没写完会失败，这是正常的）
make test-solutions   # 用参考答案跑全部测试
make check-links      # 检查所有文档链接
```

## 📁 仓库结构

```
hello-agent-infra/
├── modules/            # 10 个模块：README（中/英）+ lab.py + exercise.py / solution.py / test_exercise.py
├── codesign_sim/       # 模拟器：hardware / model / workload / cache / engine / router / rollout / credit
├── mini_engine/        # 真实执行的迷你推理引擎 + asyncio agent 前端
├── capstone/           # 综合实战：读懂并扩展 mini_engine
├── docs/               # 术语表、心算数字、面试题、阅读清单、GPU 验证指南、模拟器 API
├── tests/              # 模拟器与迷你引擎的测试
└── scripts/            # 链接检查
```

## ⚖️ 模拟器、迷你引擎和真实系统的边界

- **`codesign_sim` 是成本模型，不是引擎。** 它用 roofline 估算每一步的耗时，用离散事件推进调度，适合比较「策略 A 和策略 B 谁更好、差几倍」，不适合给出精确的毫秒数。每个模拟器的简化假设写在 [API 文档](docs/codesign-sim.md) 里。
  - **可以相信的**：数量级、结论的方向，以及「在什么参数区间结论成立」。凡是参数一变结论就会反转的地方，讲义都会写出反转的条件。
  - **需要保留的**：kernel 实现、通信拥塞、调度器的工程细节都可能改变常数，甚至改变两个接近方案的排序。拿不准时，用 [GPU 验证指南](docs/gpu-validation.md) 在真实系统上复现。
- **`mini_engine` 是真实执行的，但规模是玩具级。** 前缀缓存、分块 KV、调度、抢占都真的在跑，正确性由差分测试证明（在测试覆盖的配置上 greedy 输出逐 token 一致；logits 仍有 1e-15 量级的数值差异，这正是 batch invariance 问题的缩影）；但它跑在 CPU 上、没有 GPU kernel、没有分布式，性能数字不能外推到 GPU。
- **想在真实 GPU 上验证？** 看 [GPU 验证指南](docs/gpu-validation.md)：用 vLLM / SGLang 的官方 benchmark 复现本项目的主要结论，以及模拟器和真实系统的已知偏差。

## 🤝 姊妹项目

[Hello Agent System](https://github.com/heaven999b/hello-agent-system)：从零手写一个生产级 Agent 框架，讲 harness 这一侧。本项目讲 harness 之下的推理与训练系统，以及两边怎么协同设计。

## 📄 License

[MIT](LICENSE)
