<div align="center">

# 👋 Hello Agent Infra

### Harness × Inference Co-Design in 20 Hours

**From "I can build agents" to "I can talk shop with LLM infra engineers": 10 modules + 1 real mini engine · bilingual (中文 / English) · every lab runs on a laptop.**
Inference first principles, inference engines, the DeepSeek systems stack, agent workload profiling, caching and prefill, long tails and async RL, multi-agent credit assignment, system design and interviews.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/deps-numpy%20only-brightgreen)
![Hardware](https://img.shields.io/badge/runs%20on-M1%208GB%20laptop-lightgrey)
![Built with Claude Opus 5.5](https://img.shields.io/badge/built%20with-Claude%20Opus%205.5-D97757)

Architecture designed with the help of **Claude Opus 5.5**

[中文](README.md) · **English**

[Quick start](#-quick-start) · [20-hour path](#-the-20-hour-path) · [Capstone](capstone/README.en.md) · [Interview questions](docs/interview-questions.en.md) · [Numbers cheatsheet](docs/numbers-cheatsheet.en.md) · [Reading list](docs/reading-list.en.md)

</div>

---

## 🤔 Why this project?

An agent's harness (context assembly, compaction, memory injection, tool calls, multi-agent orchestration) and the inference/training systems underneath it used to be optimized by different people. At the scale of coding agents and agentic RL, decisions on one side start to hold the other side hostage:

- the harness **compacts and rewrites** the context to save tokens, and the engine's **prefix cache** is invalidated wholesale, forcing a full re-prefill;
- while an agent **waits for a tool** for seconds to minutes, should its KV cache stay in HBM, or be dropped and recomputed later?
- many projects and thousands of sessions share one GPU pool; **cache-aware routing** saves prefill but piles load onto a few instances;
- agentic RL rollouts have a **long tail**: one trajectory runs 20 turns, another 2, and synchronous training makes the whole batch wait for the slowest;
- several agents cooperate on a task with a single final reward: **who gets the credit**, and how many extra rollouts does it cost to find out?

None of these can be solved from one side alone. That is **harness–inference co-design**. This project fills in the systems knowledge, order-of-magnitude intuition, hands-on labs, and interview vocabulary you need, in 20 hours.

> 🎯 **An honest goal**: 20 hours will not make anyone a senior engineer; that takes years of production scale and incidents. What it can do: let you discuss problems in the language, numbers and trade-offs used by teams like DeepSeek / Qwen / Seed, hold up under deep-dive interviews, and leave you with a project you can run and explain.

### Prerequisites

- You can read Python and numpy, and you know the parts of a Transformer forward pass (attention, MLP, residuals).
- You know the basic form of policy gradients and PPO / GRPO (used in modules 6 and 7).
- No CUDA or GPU experience needed.

### Exit checklist

After 20 hours you should be able to do each of these on a whiteboard on your own (the self-check questions and interview follow-ups in every module drill exactly these):

1. Given a model architecture and a GPU, work out in your head the KV bytes per token, the maximum concurrency per GPU, the bottleneck of prefill and of decode, and at what batch size / context length they flip.
2. Explain what continuous batching, PagedAttention, prefix caching, chunked prefill, speculative decoding and PD disaggregation each solve, what they cost, and when they do not pay off.
3. Describe the key design choices of DeepSeek's inference system (MLA, large-scale EP, PD disaggregation, on-disk KV cache), and back out each GPU's operating point from the published numbers.
4. For a given agent workload, tell which harness behaviours invalidate the prefix cache, and whether KV should be kept, offloaded or recomputed while a tool is running.
5. Derive the long-tail bubble of synchronous RL rollout, and explain the gains and costs of partial / asynchronous rollout (staleness, off-policy correction, train–inference mismatch, weight sync).
6. Compare the families of multi-agent credit assignment and compute how many extra rollouts each one needs.
7. Finish an LLM-infra system design question in 45 minutes with the seven-step framework.

### How facts are sourced

Every hardware spec, model parameter, paper result and company system detail in the lectures links to a primary source (paper, official technical report, official repository or documentation). **Inferences** are labelled as inferences; anything that cannot be verified is marked "not public" or "reportedly"; **the course's own abstractions** (such as the simulators' simplifying assumptions) are documented in the corresponding API docs. In interviews, keep these three categories apart.

## ✨ What you get

| | |
|---|---|
| 🧮 **Order-of-magnitude intuition** | Every module has "numbers you must be able to do in your head": KV bytes per token, why decode is memory-bound, how many concurrent requests one H800 node can hold |
| 🧪 **Runnable labs** | `codesign_sim`: a numpy-only simulator of inference and training systems (roofline cost model, prefix cache, continuous batching, multi-instance routing, sync/async rollout, credit assignment); every lab finishes within 10 seconds |
| ⚙️ **A real implementation** | `mini_engine`: a mini inference engine that actually executes the Transformer forward pass, with paged KV, prefix caching, continuous batching, chunked prefill, preemption, and an asyncio multi-agent front end; differential tests (including randomly generated configurations) show greedy outputs are token-for-token identical with each optimization on and off |
| 📝 **Exercises with tests** | Every hands-on module has `exercise.py` plus tests; run `make module N=05` to check your work |
| 🎤 **Interview ammunition** | DeepSeek-style deep-dive follow-ups, 5 complete system design walkthroughs, a bank of about 60 questions, and how to present your own co-design project |

## 🗺 The 20-hour path

| # | Module | Time | You will understand | Hands-on |
|---|---|---|---|---|
| 0 | [The map](modules/00_map/README.en.md) | 0.5h | How harness / engine / training system / hardware shift costs onto each other | — |
| 1 | [Inference first principles](modules/01_inference_basics/README.en.md) | 2h | Prefill vs decode, roofline, the KV cache formula, TTFT/TPOT/goodput | Roofline and KV calculators |
| 2 | [Modern inference engines](modules/02_inference_engines/README.en.md) | 2h | Continuous batching, PagedAttention, prefix caching, chunked prefill, speculative decoding, quantization, parallelism, PD disaggregation | Engine sim: chunked prefill and token budgets |
| 3 | [The DeepSeek stack](modules/03_deepseek_stack/README.en.md) | 2.5h | MLA, DeepSeekMoE, MTP, FP8, DualPipe, DeepEP, EPLB, 3FS, PD disaggregation + large-scale EP serving | MLA vs GQA memory, EP load balancing |
| 4 | [Agent workload profiling](modules/04_harness_workloads/README.en.md) | 1.5h | How agent traffic differs from chat traffic, and what a cache-friendly harness looks like | Harness policy vs prefix hit rate |
| 5 | [Cache × prefill co-design](modules/05_cache_prefill/README.en.md) | 2.5h | Multi-project cache contention, keeping KV across tool waits, cache-aware routing vs load balancing, KV tiering, program-level scheduling | Routing and cache policy comparison |
| 6 | [Long tails and asynchrony](modules/06_long_tail_async/README.en.md) | 2.5h | Rollout long tails, partial / async RL, staleness and off-policy correction, train–inference mismatch, weight sync | Sync / partial / async rollout comparison |
| 7 | [Multi-agent credit assignment](modules/07_multi_agent_credit/README.en.md) | 1.5h | Group baselines, difference rewards, counterfactual baselines, Shapley values, reward hacking | Estimator variance and learning speed |
| 8 | [System design and interviews](modules/08_system_design/README.en.md) | 2h | An answer framework, 5 full walkthroughs, how to present your project | — |
| 9 | [Frontier and open problems](modules/09_frontier/README.en.md) | 0.5h | Unsolved problems, what RSI means for infrastructure, turning a problem into a paper | — |
| ★ | [Capstone: a mini inference engine](capstone/README.en.md) | 2.5h | Read and extend a real engine with an async agent front end | Session-aware eviction, cache-aware routing, and more |

**20 hours in total.** Suggested pace: 4 days × 5 hours, or 2 hours a day for 10 days. Modules 5, 6 and 7 are the core of co-design; if time is short, protect those.

## 🚀 Quick start

Running needs Python 3.10+ and numpy; the tests also need pytest (`make setup` installs both). The commands use `git` and `make` (on macOS they come with the Xcode Command Line Tools). No GPU required: the project is developed and tested on an Apple M1 with 8 GB, Python 3.14 and numpy 2.5, and "every lab within 10 seconds" means the wall-clock time of running one `lab.py` on its own on that machine while idle.

```bash
git clone https://github.com/heaven999b/hello-agent-infra.git
cd hello-agent-infra
make setup            # create .venv and install numpy + pytest
make lab N=01         # run the module 1 lab
make module N=05      # run the module 5 exercise tests (failing until you finish them is expected)
make test-solutions   # run every test against the reference solutions
make check-links      # check all documentation links
```

## 📁 Repository layout

```
hello-agent-infra/
├── modules/            # 10 modules: README (zh/en) + lab.py + exercise.py / solution.py / test_exercise.py
├── codesign_sim/       # simulators: hardware / model / workload / cache / engine / router / rollout / credit
├── mini_engine/        # a mini inference engine that really executes, plus an asyncio agent front end
├── capstone/           # capstone: understand and extend mini_engine
├── docs/               # glossary, numbers cheatsheet, interview questions, reading list, GPU validation guide, simulator API
├── tests/              # tests for the simulators and the mini engine
└── scripts/            # link checker
```

## ⚖️ Where the simulator, the mini engine and real systems differ

- **`codesign_sim` is a cost model, not an engine.** It estimates each step with a roofline model and advances scheduling with discrete events. Use it to compare "is policy A better than policy B, and by how much"; do not read its outputs as exact milliseconds. Each simulator's simplifying assumptions are listed in the [API docs](docs/codesign-sim.en.md).
  - **What you can trust**: orders of magnitude, the direction of a conclusion, and the parameter range in which it holds. Wherever a conclusion flips when a parameter changes, the lecture states the condition under which it flips.
  - **What to hold loosely**: kernel implementations, communication congestion and scheduler engineering details can change the constants, and can even swap the ranking of two close alternatives. When in doubt, reproduce on a real system with the [GPU validation guide](docs/gpu-validation.en.md).
- **`mini_engine` really executes, at toy scale.** Prefix caching, paged KV, scheduling and preemption actually run, and differential tests show correctness (on the tested configurations, greedy outputs are token-for-token identical; the logits still differ at the 1e-15 level, a miniature of the batch-invariance problem); but it runs on a CPU with no GPU kernels and no distribution, so its timings do not transfer to GPUs.
- **Want to validate on a real GPU?** See the [GPU validation guide](docs/gpu-validation.en.md): reproduce the main results with the official vLLM / SGLang benchmarks, plus the known gaps between the simulator and real systems.

## 🤝 Sister project

[Hello Agent System](https://github.com/heaven999b/hello-agent-system) builds a production-grade agent framework from scratch, covering the harness side. This project covers the inference and training systems beneath the harness, and how to co-design the two.

## 📄 License

[MIT](LICENSE)
