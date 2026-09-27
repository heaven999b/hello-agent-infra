[中文](README.md) · **English**

# Module 09: Frontiers and Open Problems

> 🕐 Suggested time: 0.5 hours (counting only the "30-minute core": Sections 0–2 + the first 3 questions of Section 12; everything marked "optional" does not count toward the 20 hours) ｜ 🎯 After this module you can: explain the four 2025–2026 frontier threads of agent × infra in four paragraphs; take the two open problems most relevant to your direction down to "what is already answered, what is really unsolved, how to make the first plot" ｜ 🧪 Lab: none (lecture only) ｜ 📖 Must-read primary sources: [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (abstract and Section 3); [Autellix](https://arxiv.org/abs/2502.13965) already appears in [Module 04](../04_harness_workloads/README.en.md), so here just re-read its abstract from the "program-level serving" angle

**How to spend the 30 minutes**: about 10 minutes on the four threads in Sections 2.1–2.4, about 15 minutes on the two selected problems in Sections 2.5–2.6, and 5 minutes on the first 3 questions of Section 12. Everything from Section 3 on is optional, for when you pick a research topic.

This module only includes papers and reports the author opened and verified. Every "X× improvement" below is **the number reported by the original paper under its own setup**; numbers across papers are not directly comparable. Coverage is as of September 2026.

## 0. One sentence

**Inference engines and training systems are moving from "serving a request" to "serving a program": the scheduling unit is becoming the session/program, cache policies are starting to read signals from the harness (the agent code wrapped around the model), RL systems are starting to treat the whole agent harness as a black box for rollouts (sampling trajectories with the current policy), and recursive self-improvement (RSI) closes the train–deploy–evaluate loop, turning evaluation and data generation into first-class workloads.**

## 1. Why big labs care

A concrete scenario: a coding-agent platform where each user task is a 20–100 turn agent session (read the repo, run tests, edit code), with tool calls lasting seconds to minutes between turns. The same model is also trained with agentic RL: every training step needs thousands of such trajectories.

- **Inference side**: the engine only sees isolated requests. When a tool call returns, the previous turn's KV cache may already have been evicted by LRU, so the context has to be prefilled again. InferCept measured, on its workload, that this recomputation of already-computed context accounts for 37–40% of total model forward time ([InferCept](https://arxiv.org/abs/2402.01869)).
- **Training side**: the OpenRLHF authors say the inference (generation) phase often accounts for over 90% of total RLHF / RLVR runtime; this is a general statement with no measurement setup ([OpenRLHF](https://arxiv.org/abs/2405.11143); APRIL cites it too). Long-tail trajectories stall the whole batch, and environments (sandboxes, browsers, compilers) are CPU-heavy and flaky.
- **Closed-loop side**: models/harnesses now modify themselves (DGM self-improves from 20.0% to 50.0% on SWE-bench, [DGM](https://arxiv.org/abs/2505.22954)), and every "improvement" must be validated by running a full evaluation. Evaluation moves from "once before release" to "every step of the inner loop".

## 2. The 30-minute core

### 2.1 Thread 1: serving agents as "programs"

The basic unit of scheduling and caching is being upgraded from a single request to a program or session: SGLang lets applications be written as programs with `fork` / `gen` and reuses prefix KV across programs ([SGLang](https://arxiv.org/abs/2312.07104)); Autellix prioritizes by the summed runtime of a program's prior calls and, on its workloads, improves program throughput by 4–15× at the same latency ([Autellix](https://arxiv.org/abs/2502.13965)). Order of magnitude: a 50-turn session with a 20k-token shared prefix and 2k external tokens appended per turn (tool results and user input; with cache hits only these need append-prefill; the model's own tokens were written into the KV during decode and are ignored here). With every cache lost, turn k must prefill 20k + 2k·k, ≈ 1.0M + 2.55M ≈ 3.6M tokens over 50 turns; with every cache hit only 20k + 50×2k = 0.12M, about 30× less. The value of program-level scheduling is pushing the system toward the latter, at the price of handling the load imbalance that session stickiness causes ([Module 05](../05_cache_prefill/README.en.md)).

### 2.2 Thread 2: harness-aware inference engines

The harness holds signals the engine cannot see: roughly how long a tool will take, which agent runs next, when the context will be compacted. Continuum uses the historical distribution of a tool's durations to decide how long to keep KV during a tool call ([Continuum](https://arxiv.org/abs/2511.02230)); Ask the Tool, Don't Guess has tools report progress while running instead of guessing durations before the call ([Ask the Tool](https://arxiv.org/abs/2609.18849)); KVFlow, CacheScout and TOPAS manage KV with declared or online-learned workflow graphs. Industry already has interfaces: NVIDIA Dynamo's `nvext.agent_hints` (`priority`, `strict_priority`, `osl`, `speculative_prefill`) and SGLang's priority-based eviction of radix leaves; see [Module 05 Section 7](../05_cache_prefill/README.en.md#7-a-co-design-interface-proposal-what-the-harness-should-tell-the-engine). The remaining question is which hint on top of these fields has the largest marginal benefit.

### 2.3 Thread 3: large-scale agentic RL systems

The evolution is "synchronous → partial rollout → fully asynchronous → disaggregated/heterogeneous → rollout-as-a-service"; each step removes the previous step's bubbles and each trades a little off-policy bias for utilization (the math skeleton is in [Module 06 Section 5](../06_long_tail_async/README.en.md#5-cost-1-staleness-and-off-policy-correction-the-math)). The latest step treats the whole harness as a black box: Polar proxies LLM API calls and rebuilds token-level trajectories afterwards, but the gain depends heavily on the harness: the same Qwen3.5-4B on SWE-bench Verified gains 22.6 points with the Codex harness and only 0.6 with Qwen Code ([Polar](https://arxiv.org/abs/2605.24220)). The bottleneck is moving from GPU compute to "environments + long tail + staleness + trajectory fidelity": DeepSeek's sandbox platform DSec supports over 380,000 concurrent sandboxes in production ([DSec](https://arxiv.org/abs/2609.22978)).

### 2.4 Thread 4: what RSI means for infrastructure

A self-improvement loop may change data, rewards, curricula, weights or harness code (DGM, AlphaEvolve and others, Section 6). Three implications for infrastructure:

1. **Version everything**: KV is an intermediate result of one weight version, so cache keys must carry the version. When weights update, flushing immediately or keeping old-version KV is a policy choice, and there are already two answers: AReaL interrupts in-flight generation, discards the KV and recomputes; PipelineRL keeps generating with the old KV and, on a 7B model, measures only slightly higher KL than recomputing ([Module 06 Section 4.6](../06_long_tail_async/README.en.md#46-in-flight-weight-updates-not-even-recomputing-kv)). Prefix-cache keys are by default hashes of token sequences without the weight version, so not flushing after an update silently reuses old KV; the switches SGLang and vLLM provide for this are in [Module 06 Section 7.5](../06_long_tail_async/README.en.md#75-weight-versions--kv--prefix-caches).
2. **Data pipelines generate themselves**: DeepSeek-V3.2's agentic task synthesis pipeline produced 1,827 environments and 85,000 prompts ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)); "build environments + build tasks + filter" is itself a large inference workload.
3. **Evaluation enters the inner loop**: every candidate runs a benchmark; the Huxley-Gödel Machine further points out that benchmark score is not the same as "potential for further self-improvement", and by estimating that potential from descendants' performance to allocate budget it beats prior methods with fewer CPU hours ([HGM](https://arxiv.org/abs/2510.21614)).

### 2.5 Selected problem A: the system cost of counterfactual credit (multi-agent reward × long-tail async × prefill)

- **Already answered**: in the single-agent setting, "where to branch and how to amortize prefixes" has been studied directly. TreeRL branches from high-uncertainty intermediate steps rather than randomly, under the same generation token budget ([TreeRL](https://arxiv.org/abs/2506.11902)); ARPO branches adaptively by entropy after tool calls ([ARPO](https://arxiv.org/abs/2507.19849)); FR3E finds high-uncertainty decision points for targeted rollouts ([FR3E](https://arxiv.org/abs/2507.07017)); TreePO amortizes common prefixes and saves 22%–43% of GPU hours in its sampling design ([TreePO](https://arxiv.org/abs/2508.17445)); Tree-GRPO shares prefixes to get more rollouts within a fixed token or tool-call budget ([Tree-GRPO](https://arxiv.org/abs/2509.21240)). In the multi-agent setting, AT-GRPO branches for every agent at every turn ([AT-GRPO](https://arxiv.org/abs/2510.11062)), and C3 restores checkpoints instead of regenerating history, using 32% fewer tokens than MAPPO ([C3](https://arxiv.org/abs/2603.06859)).
- **Really open**:
  1. In multi-agent settings the choice is "which agent × which turn", and a sample's credit can only be computed once the original trajectory and all of its counterfactual branches have finished: a **group barrier** sitting on the long-tail critical path;
  2. A coding agent's environment state (file system, processes) is not in the text, so branches need sandbox snapshots or forks; DSec offers an interface that takes an incremental disk snapshot of a sandbox and restores it as a new sandbox, but the paper gives no snapshot latency, and the branch count multiplies sandbox concurrency ([Module 03 Section 5.3](../03_deepseek_stack/README.en.md#53-dsec-the-sandbox-platform-for-agentic-rl));
  3. Keep the accounts separate: a KV fork saves the **prefill** before the decision point, checkpoint restoration (C3) saves the **decode** before the decision point, tool results appended after the fork still need prefill, and a prefix hit also requires routing the branch to the same instance.
- **First plot**: on the team task of [Module 07](../07_multi_agent_credit/README.en.md), fix an extra-evaluation budget, compute difference rewards for only k of the n agents and a group baseline for the rest, plot "gradient SNR vs k", then convert k into useful throughput with `simulate_rollout`. This can be done now; "choosing decision points" needs a multi-step team task added to the library.

### 2.6 Selected problem B: weights change, prefixes are shared: versioned multi-project caching (multi-project cache × long-tail async × prefill)

- **Already answered**: what to do with in-flight requests across a weight update has two approaches with measurements (item 1 of Section 2.4 above); a multi-turn agent whose weights change between two turns also ends up with "new weights + old KV", and [Module 06 Section 7.5](../06_long_tail_async/README.en.md#75-weight-versions--kv--prefix-caches) lists four ways to handle KV on continuation and their costs; multi-tenant isolation has vLLM's `cache_salt` and research on timing side channels ([Module 05 Section 2.4](../05_cache_prefill/README.en.md#24-multi-tenancy-share-or-isolate)).
- **Really open**: a project's shared prefix (system prompt + tool definitions + repository context) is reused by many **new** requests. After a weight update, can new requests keep reusing the previous version's prefix KV? For how many versions? Nobody has drawn this "hit rate vs KL" curve yet; add several projects competing for capacity and caches split by isolation domain, and you have the core problem of the reader's "multi-project cache" direction.
- **First plot**: the hit-rate side can be done now. Use `generate_sessions` + `PrefixCache` to run two bounds: a fresh empty cache at every weight update (period = `train_step_s`, i.e. immediate flush), and never flushing (fully reusing stale KV). The area between the two curves is the most prefill that "allowing stale prefix reuse" can win back. The KL side needs a small model doing real generation.

## 3. Optional A: details of program-level serving

| System | First-class citizen | Key mechanism | Reported effect |
|---|---|---|---|
| [SGLang](https://arxiv.org/abs/2312.07104) (2023) | LM program | Frontend primitives `fork` / `join` / `gen`; RadixAttention runtime reuses prefix KV across programs | up to 6.4× throughput |
| [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) (OSDI 2024) | Semantic Variable | The application exposes data dependencies between LLM calls to the service, which runs data-flow analysis | up to an order of magnitude |
| [InferCept](https://arxiv.org/abs/2402.01869) (2024) | Generation "intercepted" by a tool call | During a tool call, choose among "preserve / swap to CPU / discard and recompute" for the KV, minimizing GPU memory waste (min-waste) | 1.6–2× throughput |
| [Autellix](https://arxiv.org/abs/2502.13965) (2025) | The whole agent program | Intercepts a program's LLM calls; preempts and prioritizes by **the summed runtime of all the program's prior calls** (PLAS, program-level attained service), mitigating program-level head-of-line blocking | 4–15× program throughput at the same latency (vs vLLM and others) |
| [ConServe](https://arxiv.org/abs/2606.01839) (2026) | The whole conversation | Raises the scheduling unit from turn to conversation: the long first-turn prefill goes to a high-throughput prefiller and the KV is migrated across nodes only this once; every later turn's append-prefill and decode run locally on the one pinned decoder | −51% p95 time-to-first-effective-token |

**Trade-offs**:

- Program-level "the more service received, the lower the priority" belongs to the LAS (least attained service) family. With heavy-tailed program sizes and preemption it often improves mean completion time; but in LLM serving prefill is usually not finely preemptible, and cache locality, tool waits and priority classes all matter, so the gain is not unconditional, and long programs must be protected from starvation (aging or quotas).
- Session stickiness (ConServe, cache-affinity routing) raises hit rate but causes load imbalance across instances. This is exactly what `affinity_with_load_cap` balances in [Module 05](../05_cache_prefill/README.en.md).
- Where does program information come from? Either the application declares it explicitly (Parrot, SGLang frontend) or the server infers it (Autellix intercepting calls). Declaration is more accurate but requires changing the harness; inference is non-invasive but can guess wrong.

## 4. Optional B: details of harness-aware engines

| Harness signal read by the engine | Representative work | What it is used for | Reported effect |
|---|---|---|---|
| Historical distribution of this tool's call durations (empirical CDF) | [Continuum](https://arxiv.org/abs/2511.02230) | Pin KV in GPU memory during a tool call, choose the TTL by the objective below, auto-evict on expiry; combined with program-level FCFS | >8× better average job completion time (JCT) |
| Progress reported by the tool **while running** | [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849) | Argues that no estimate fixed before a call starts can reliably even rank calls; tools report "how much is left" while running and the engine decides whether KV stays | −20.7% p90 TTFT after a tool call (vs LRU), close to an oracle |
| Declared workflow graph | [KVFlow](https://arxiv.org/abs/2507.07400) | Abstracts agent scheduling as an Agent Step Graph, evicts KV by "steps until next execution", and prefetches from CPU ahead of time | up to 1.83× / 2.19× vs SGLang hierarchical radix cache |
| Agent transitions learned online | [CacheScout](https://arxiv.org/abs/2608.14624) | No predeclared workflow; learns agent execution transitions online to guide eviction and prefetching | +10–18 pp hit rate, −18–45% mean TTFT |
| Workflow DAG + prefix state | [TOPAS](https://arxiv.org/abs/2608.25523) | Jointly decides "which agent prefixes to keep" and "which requests to run", trading critical path against prefix reuse | up to −39.8% / −49.4% mean/p99 JCT on synthetic workloads |

**The TTL objective, written more rigorously**: let the tool duration T be a random variable, B the cost saved if the KV is still there when the tool returns (recompute or reload + queueing after eviction), and c the opportunity cost per second of pinning this KV. The expected net benefit of keeping it until τ is

J(τ) = P(T ≤ τ) · B − c · E[min(T, τ)]

Differentiating in τ: J'(τ) = f(τ)·B − c·P(T > τ). So **keep holding as long as the hazard rate of "the tool returns in the next instant", h(τ) = f(τ)/P(T > τ), exceeds c/B**. The Continuum paper writes argmax_τ P(τ, f)×Benefit − Cost(τ), approximating the occupation cost as linear in τ (i.e. using τ in place of min(T, τ), an upper bound). A corollary: if tool durations are exponential (constant hazard 1/E[T]), the optimal TTL is either 0 or infinity: **pin if and only if B > c·E[T]**. Question 4 in Section 12 asks you to plug in numbers.

**How wide the hint interface should be**:

- Narrowest: pass nothing; the engine guesses from LRU and history.
- Middle: each request carries a few fields. This tier already has industrial implementations: Dynamo's `nvext.agent_hints` (`priority`, `strict_priority`, `osl` expected output length, `speculative_prefill` to prefill the next turn's prefix after this turn ends) plus a session ID, and SGLang's `--radix-eviction-policy priority`; [Module 05 Section 7](../05_cache_prefill/README.en.md#7-a-co-design-interface-proposal-what-the-harness-should-tell-the-engine) also proposes new fields such as `tool_eta`, fork groups and droppable segments.
- Widest: the harness hands the whole program (DAG) to the engine (SGLang frontend, Parrot).

Wider gives more benefit but heavier coupling, less reuse across harnesses, and more reliance on the harness telling the truth. The "context rewrite invalidates the prefix" effect in [Module 04](../04_harness_workloads/README.en.md) is a signal the harness knows best and the engine knows least.

**Putting these into one cost model**: job completion time ≈ queueing + prefill(missed tokens) + decode + tool wait; memory = active KV + pinned KV + swapped-out KV. TTL, session stickiness, LAS priority and per-project quotas are all decision variables in this model. Module 05 gives a unified TTFT cost estimate ([Section 3.2](../05_cache_prefill/README.en.md#32-a-unified-cost-estimated-ttft)) and the "recompute vs load" break-even formula ([Section 4.2](../05_cache_prefill/README.en.md#42-recompute-vs-load-the-break-even-formula)), and [Module 08](../08_system_design/README.en.md) uses them in full system design questions.

## 5. Optional C: details of agentic RL systems

```
synchronous colocated (verl/HybridFlow)
   └─ pain: the longest trajectory in a batch stalls the batch → long-tail bubbles
partial rollout (Kimi k1.5, APRIL)
   └─ unfinished trajectories continue next round; pain: one trajectory spans several policy versions
fully asynchronous (AReaL, Asynchronous RLHF)
   └─ generation decoupled from training; pain: staleness, needs a max version gap + off-policy correction
disaggregated / heterogeneous (StreamRL, RollArt)
   └─ prefill, decode, environments, rewards each on best-fit hardware; pain: cross-stage sync and scheduling
rollout-as-a-service / harness as black box (ProRL Agent → Polar)
   └─ instead of porting the harness into an RL env interface, proxy LLM API calls and rebuild token-level trajectories afterwards
```

| System | Key point | Reported |
|---|---|---|
| [HybridFlow / verl](https://arxiv.org/abs/2409.19256) | Hybrid single-controller + multi-controller; 3D-HybridEngine reshards the actor between training and generation | 1.53–20.57× throughput |
| [AReaL](https://arxiv.org/abs/2505.24298) | Fully asynchronous; interruptible rollout workers (on a weight update, discard the KV, recompute it with the new weights, then continue); staleness-enhanced PPO | up to 2.77× training speedup at equal GPU count |
| [StreamRL](https://arxiv.org/abs/2504.15930) | Disaggregated; stream generation removes pipeline bubbles; an output-length ranker identifies long-tail samples | up to 2.66× throughput |
| [Seer](https://arxiv.org/abs/2511.14617) | Observes that samples of the same prompt have similar output lengths; divided rollout + context-aware scheduling + grouped speculative decoding | up to 2.04× rollout throughput, −72–94% long-tail latency |
| [RollArt](https://arxiv.org/abs/2512.22560) | Multi-task agentic RL; prefill-heavy tasks to compute-optimized GPUs, decode-heavy to bandwidth-optimized GPUs, environments to CPU clusters; rewards on serverless | 1.31–2.05× shorter training time; trained a hundreds-of-billions-parameter MoE on a 3,000+ GPU cluster |
| [ProRL Agent](https://arxiv.org/abs/2603.18815) → [Polar](https://arxiv.org/abs/2605.24220) | ProRL Agent serves the rollout lifecycle through an API; Polar builds on it to treat any harness as a black box, proxying LLM API calls and rebuilding token-faithful trajectories | Qwen3.5-4B gains 0.6–22.6 points on SWE-bench Verified depending on the harness (Codex 22.6, Claude Code 4.8, Qwen Code 0.6, Pi 6.2) |

Scale anchors disclosed in vendor technical reports:

- **Kimi K2**: 3000+ real MCP tools fetched from GitHub and 20,000+ synthetic tools; RL uses a colocated architecture, many concurrent rollouts to amortize the latency of expensive interactions, and partial rollout so long-tail unfinished tasks do not block the whole rollout ([Kimi K2](https://arxiv.org/abs/2507.20534)).
- **DeepSeek-V3.2**: the agentic task synthesis pipeline produced 1,827 environments and 85,000 complex prompts; the post-training compute budget exceeds 10% of pre-training cost ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)).
- **DeepSeek DSec**: one production-scale unit spans about 160 nodes and serves about 3 million sandboxes per day; production supports over 380,000 concurrent sandboxes and over 5,000 sandbox creations per second ([DSec](https://arxiv.org/abs/2609.22978)).
- **GLM family**: post-training is powered by [slime](https://github.com/THUDM/slime) (Megatron training + SGLang rollout + data buffer).

**The minimal math skeleton of cross-version trajectories** (full derivation in [Module 06 Section 5](../06_long_tail_async/README.en.md#5-cost-1-staleness-and-off-policy-correction-the-math)): take an 800-token trajectory whose first 400 tokens were generated by version v10 and last 400 by v11, trained when the policy is v12. Each token must store four things: the version that generated it, the logprob **at sampling time** (including temperature and top-p truncation), the reward / advantage, and a loss mask. Importance sampling only needs the denominator to be the probability actually used for sampling, so per-token recorded logprobs are enough to write the exact likelihood ratio of this mixed-version trajectory; AReaL's decoupled PPO then separates the "trust-region center" (the most recent policy) from the "behavior policy" (each segment's own version). Per-token ratios are used instead of the sequence-level product because the product's variance grows exponentially with length (Question 6 of Section 12).

## 6. Optional D: details of RSI

RSI (recursive self-improvement) has a dedicated academic venue as of 2026: the [ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://recursive-workshop.github.io/), which frames the problem as "what changes (parameters, world models, memory, tools and skills, architectures), when changes happen (within an episode, at test time, after deployment), how changes are produced (reward/value learning, imitation, evolutionary search)", plus safety and evaluation.

| What changes | Representative work | The "evaluator" in the loop |
|---|---|---|
| Training data / reasoning traces | [STaR](https://arxiv.org/abs/2203.14465): retrain on its own correct rationales | whether the answer is correct |
| The reward itself | [Self-Rewarding LM](https://arxiv.org/abs/2401.10020): the model scores itself via LLM-as-a-Judge, iterative DPO | itself |
| Curriculum (posing tasks) | [Absolute Zero](https://arxiv.org/abs/2505.03335): one model proposes and solves tasks, zero external data | a code executor |
| Weight-update directives | [SEAL](https://arxiv.org/abs/2506.10943): the model generates its own finetuning data and update directives; the updated model's downstream performance is the RL reward | downstream tasks |
| Agent code / harness | [DGM](https://arxiv.org/abs/2505.22954), [HGM](https://arxiv.org/abs/2510.21614), [ModularRSI](https://arxiv.org/abs/2609.14857) | coding benchmarks (SWE-bench, etc.); HGM instead uses a "self-improvement potential" estimated from descendants' performance |
| Algorithm and infrastructure code | [AlphaEvolve](https://arxiv.org/abs/2506.13131): an evolutionary coding agent that improved Google data-center scheduling and sped up training of the LLM underpinning it | one or more automatic evaluators |

**Three implications for infrastructure (expanded)**:

1. **Versioned semantics, not "everything is invalid after an update"**: KV, routing tables and evaluation baselines carry the model version. Forcing every worker onto the new version immediately is equivalent to flushing the cache, and the prefix cache's lifetime is then cut off by the weight-update period; keeping several versions makes hit rate depend on how traffic is split across versions and how long old versions are retained, at the price of memory for multiple versions of weights and KV. In-flight requests already have two answers (AReaL recomputes, PipelineRL reuses stale KV); shared prefixes across requests do not yet (Section 2.6).
2. **Self-generated data pipelines**: the synthesis pipelines of Kimi K2 and DeepSeek-V3.2 both show that "build environments + build tasks + filter" is itself a large inference workload, often bottlenecked by sandboxes/execution environments and judging (LLM judge or executor), not by training.
3. **Evaluation loop**: systems like DGM run a benchmark for every candidate; [SEAGym](https://arxiv.org/abs/2606.17546) shows frequent updates may fail to improve held-out performance and good intermediate snapshots may later collapse; [ModularRSI](https://arxiv.org/abs/2609.14857) uses 2,000 evolution tasks disjoint from the evaluation benchmarks to avoid "evolving on the benchmark = overfitting the benchmark". Evaluation therefore needs to be scheduled like training: early elimination of weak candidates, variance-aware promotion, regression gating and rollback.

## 7. Optional E: the full open-problem table

Tags: **A** long-tail async · **B** multi-agent reward · **C** multi-project cache · **D** prefill · **E** efficient system · **F** RSI. The "handle" column says whether it can be done now with this project's code.

| # | What is really open | Directions | Already answered (anchors) | Handle for a minimal experiment |
|---|---|---|---|---|
| 1 | On long agent trajectories (10⁴–10⁵ tokens, spanning dozens of version segments), how should the threshold of per-token truncated IS adapt to segment count and length? When does the effective sample size (ESS) collapse? | A B | Cross-version segmented trajectories use per-token behavior probabilities + decoupled PPO (AReaL, see [Module 06 Section 5.3](../06_long_tail_async/README.en.md#53-decoupled-ppo-separating-the-trust-region-center-from-the-behavior-policy)); PipelineRL's token lag | Doable now: synthesize per-token log-ratio sequences in numpy and compare the bias and variance of token- and sequence-level TIS (Module 06's exercise already implements these two levels) |
| 2 | Can harness signals (task type, planned tool sequence) predict trajectory length better than the prompt alone? | A E | StreamRL uses an output-length ranker; Seer exploits length similarity among samples of the same prompt | Partly doable: public traces include the trajectories of submissions in [SWE-bench/experiments](https://github.com/SWE-bench/experiments) (Ask the Tool used mini-SWE-agent leaderboard runs and OpenHands evaluation runs); plugging predictions into a scheduling simulation needs "dispatch by predicted length" added to the library |
| 3 | Should the staleness bound adapt to the tool-latency distribution? | A | AReaL uses a fixed η and controls staleness by balancing workers; ROLL Flash reports α=2 is enough in most scenarios | System side doable now: sweep `mean_tool_s` × `max_staleness` in the [Module 06](../06_long_tail_async/README.en.md) simulator; the algorithm side needs real training |
| 4 | Budgeted counterfactuals in multi-agent settings: branch on "which agent × which turn"? Branches also need environment snapshots and sit behind a group barrier | B A | Choosing branch points in single-agent RL: TreeRL, ARPO, FR3E; branching every turn in multi-agent: AT-GRPO; granularity by NLL: [GACA](https://arxiv.org/abs/2609.12424); [Coverage, Not Targeting](https://arxiv.org/abs/2609.02417) (uniform spreading beats targeting only with terminal-state verifiers, when the information density V_d is well below about 0.8) | The agent-level simplified version is doable now ("first plot" of Section 2.5); the turn level needs a multi-step team task added to the library |
| 5 | In multi-turn agents, when does a KV fork for counterfactuals stop saving? (tool results after the fork still need prefill, branches must be routed to the same instance, resident branch KV squeezes the batch) | B C D | TreePO amortizes common prefixes and saves 22%–43% of GPU hours; Tree-GRPO gets more rollouts under a fixed budget; C3 uses 32% fewer tokens through checkpoint restoration | Doable now: build the forked block sequences directly and feed them to `PrefixCache.match` / `insert`, count hits and prefill tokens, then convert to time with the Module 01 formulas |
| 6 | When two agents share one judge reward, do they form "passwords" that only work between them to fool the judge? | B F | [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794): a single symbol or an opener like "Let's solve this step by step" elicits positive reward (a single-model attack) | Needs real small-model training: a generator + reviewer share a judge reward; compare the divergence "judge score − verifiable score" under joint vs separate training; then a "swap partner" control: if scores drop when paired with an independently trained partner, a private convention has formed |
| 7 | Shared prefix caching across projects: the three-way trade-off among isolation granularity (tenant / project / public prefix), quotas and hit rate, and p95 TTFT | C E | vLLM's `cache_salt`; [prior work](https://arxiv.org/abs/2409.20002) shows timing side channels through shared KV caches can infer other users' prompts; [Module 05 Section 2.4](../05_cache_prefill/README.en.md#24-multi-tenancy-share-or-isolate) | Doable now: sweep the concentration of project popularity with `generate_sessions(project_zipf_s=...)` and compare hit rate between one shared LRU `PrefixCache` and "one `PrefixCache` per project with capacity split by quota" (approximating per-project isolation, i.e. no cross-project sharing); get p95 TTFT via `route_and_simulate` |
| 8 | After a weight update, can **new requests** reuse the previous version's shared-prefix KV? For how many versions? The hit rate vs KL curve | C A F | In-flight requests already have answers: AReaL interrupts and recomputes; PipelineRL and Magistral continue with the old KV, and PipelineRL measures only slightly higher KL than recomputing | Hit-rate side doable now ("first plot" of Section 2.6); the KL side needs a small model |
| 9 | P/D disaggregation per turn or per conversation? What is the crossover condition (append length, KV migration bandwidth, decoder load)? | D E | ConServe is per conversation (later append-prefill runs locally on the decoder); Splitwise / DistServe are per request | Analytic model doable now (derive KV migration volume and queueing for both); `codesign_sim.pd` already simulates per-request P/D disaggregation (`simulate_pd`, with `simulate_colocated` at the same GPU count), and the conversation-level scheme needs a variant on top of it that "pins a decoder after turn 1 and runs append-prefill locally" |
| 10 | How can harness context compaction/rewrite be made cache-friendly (append-only summaries, block alignment)? Is the prefill saved worth a small quality loss? | D C | Manus's practice: keep the prefix stable, keep context append-only, keep serialization deterministic ([Manus blog](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)); [Module 04](../04_harness_workloads/README.en.md) | Prefill side doable now: `generate_sessions(rewrite_prob=...)`; the quality side needs real tasks |
| 11 | On top of Dynamo's existing hints and Module 05's proposal, which new field (`tool_eta`, fork groups, droppable segments) has the largest marginal benefit? | E | Dynamo `agent_hints`; SGLang priority eviction; Continuum (tool duration); Ask the Tool (progress) | Partly doable: `route_and_simulate` already supports tool-duration pin hints (`hint_mode` of none / fixed / noisy / oracle, with oracle as the upper bound), so the value of that one field can be measured first; other fields such as fork groups and droppable segments need library extensions |
| 12 | Scheduling evaluation in self-improvement loops: **correlated noise** from candidates sharing ancestors, overfitting from reusing the evaluation set, non-stationary candidate quality | F E | Successive halving / [Hyperband](https://arxiv.org/abs/1603.06560) are mature methods in hyperparameter search; HGM allocates expansion and evaluation budget by potential estimated from descendants | Doable now: synthesize "ancestor tree + correlated noise" candidates in numpy and compare successive halving's misjudgment rate and cost under correlated vs independent noise |

## 8. Optional F: a minimal experiment design template for turning an open problem into a paper

Fill in the ten lines below and you have the skeleton of a workshop paper; any line you cannot fill is where your thinking is not done yet. The "mini engine" in line 5 is the `mini_engine` from the [capstone](../../capstone/README.en.md), which really executes the Transformer forward pass: mechanism-level ideas (a new eviction policy, session-aware scheduling) can be checked for correctness there first, then scaled up in the simulator.

```
1. Problem (one sentence, with a measurable quantity):  under ___ workload, ___ degrades ___ metric by ___×
2. Why now (one order-of-magnitude anchor + source):     e.g. generation often exceeds 90% of RL runtime (OpenRLHF)
3. Hypothesis H1 and null H0:                            H1: ___ reduces ___ to ___; H0: gain < noise
4. Oracle upper bound (prove it is worth doing first):   how much could perfect information save? < 10% → pick another problem
5. Minimal environment (cheap to expensive):             analytic model → simulator (codesign_sim) → real-execution mini engine (mini_engine) → small model real execution → public traces
6. Baselines (at least three):                           naive / strongest published / oracle (upper bound only)
7. Metrics (one system + one algorithm, both required):  system: throughput, p99, GPU·h; algorithm: return, sample efficiency, final accuracy
8. Controls and ablations:                               same budget, hardware, trace and batching; multiple seeds with confidence intervals; randomized control (e.g. shuffle positions, keep concentration)
9. Scale extrapolation:                                  order-of-magnitude argument: does it hold at 1000 GPUs / 1M-token contexts, which term bottlenecks first
10. Threats to validity:                                 simulator assumptions, workload representativeness, difference from prior work
```

**Two rules for oracles**: an oracle uses information unavailable at deployment (true tool durations, true output lengths, future requests), so it can only serve as an offline upper bound, never be ranked as a deployable baseline; and its information must not leak into the methods being compared (e.g. a threshold tuned with true lengths).

**A filled-in example (Problem 5)**:

1. Problem: counterfactual credit in multi-agent RL runs n extra branches per sample, clearly lengthening step time under synchronous training.
2. Anchor: classic difference rewards need a way to evaluate the counterfactual world, which in real environments often means extra simulation (the COMA paper's summary of difference rewards); COMA itself marginalizes over one agent's actions with a centralized critic and needs no real re-runs. This problem studies how much the system saves when counterfactual branches **are actually run**: C3 reports that restoring checkpoints instead of regenerating history uses 32% fewer tokens than MAPPO.
3. H1: if branches fork from the decision point's KV and every prefix hits, the extra cost is only "prefill of content appended after the fork + the branches' decode", so step time grows far less than n×; H0: long-tail decode dominates and the history prefill saved by forking is negligible.
4. Oracle: if all prefill before the decision point were free, how much would the extra cost drop at most? Compute it first with the prefill / decode time formulas of [Module 01](../01_inference_basics/README.en.md).
5. Environment: `codesign_sim.credit` (credit quality) + `PrefixCache` (branch hit rate) + `simulate_rollout` (step time).
6. Baselines: shared reward (zero extra cost), LOO with full re-rollout, LOO + checkpoint restoration (C3-style, no re-decoding of history), LOO + KV fork.
7. Metrics: gradient variance, GPU·s to reach a target return; p99 step time.
8. Controls: fixed total GPU·s budget; 20 seeds with confidence intervals.
9. Extrapolation: how does the prefill/decode ratio change at 100k-token context and n=8 agents?
10. Threats: in a real system resident branch KV squeezes the batch; branches must be routed to the same instance to hit; coding agents also need sandbox snapshots.

## 9. Numbers you must be able to do in your head

| Quantity | Typical value | How to derive it | Source |
|---|---|---|---|
| Prefill-token ratio, 50-turn session, "every cache lost" vs "every cache hit" | ≈ 30× | 20k shared prefix, 2k external tokens appended per turn: cumulative ≈3.6M vs 0.12M (Section 2.1) | derived in this module |
| Recomputation caused by tool interception, as a share of forward time | 37–40% | InferCept's measurement on its workload | [InferCept](https://arxiv.org/abs/2402.01869) |
| Generation share of total RL runtime | often over 90% | OpenRLHF's general statement (cited by APRIL) | [OpenRLHF](https://arxiv.org/abs/2405.11143) |
| Async vs sync RL training speedup | up to 2.77× | AReaL, equal GPU count | [AReaL](https://arxiv.org/abs/2505.24298) |
| Post-training compute / pre-training compute | > 10% | DeepSeek-V3.2's public number | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| Prefix hit rate with immediate flush | 1 − request interval / update period | the first request of each update period always misses: one request every 5 s, one update every 30 s → 5/6 ≈ 83% | derived in this module |
| Condition to pin KV under exponential tool durations | B > c · E[T] | constant hazard 1/E[T], so the optimal TTL is either 0 or ∞ (Section 4) | derived in this module |
| KV of one 100k-token context for Qwen3-32B | ≈ 26 GB | 64 layers × 8 KV heads × 128 dims × 2 (K, V) × 2 bytes = 256 KiB/token | [config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) |

## 10. Lab

This module is lecture-only and has no `lab.py`. Problems marked "doable now" in the handle column of Section 7 can each produce a first plot in roughly 10 seconds using code from earlier modules.

## 11. Exercise (optional)

No code exercise in this module. Written exercise: pick one problem from Section 7 and fill in all ten lines of the Section 8 template, within one page.

## 12. Self-check questions

The first 3 questions belong to the 30-minute core; the last 3 are optional questions that require calculation.

1. Why is a "program" a better scheduling unit than a "request" for agent workloads? Give an order-of-magnitude argument.
2. When weights update, what are the two ways to handle the KV of in-flight requests? What does each cost?
3. In multi-agent counterfactual credit, which computation does a KV fork save, and which does checkpoint restoration save? What computation remains after the fork?
4. A Qwen3-32B instance runs on 8×H100 (TP8); a session with a 100k-token context is waiting on a tool. Assume: recomputing the prefill takes about 3.6 s (Module 01 formula, MFU 0.6); about 570 GB of the instance's memory is available for KV, and the batch is KV-capacity-bound so throughput is proportional to available KV; tool durations are exponential. Below what mean tool duration is it worth pinning this KV?
5. The rollout side syncs weights every 10 s and flushes the prefix cache immediately; a project's shared prefix is used by a new request every 2 s. What is the hit rate? What if weights sync every 60 s? If instead the previous version's prefix KV is kept for new requests, what happens to hit rate and what does it cost?
6. An 800-token trajectory: the first 400 tokens were generated by v10 with an average per-token log-ratio log(π_v12/μ) = 0.005; the last 400 by v11 with an average of 0.002. The policy at training time is v12. What is the sequence-level importance weight? Does per-token truncated IS (C=2) truncate anything?

<details><summary>Answers</summary>

1. A program knows "more calls are coming, which prefix they share, how much service it has received". With per-request scheduling and every cache lost, a 50-turn session prefills ≈3.6M tokens; with every cache hit only ≈0.12M, about 30× less (Section 2.1). Program-level scheduling also mitigates program-level head-of-line blocking (Autellix).
2. AReaL: interrupt in-flight generation, discard the KV, recompute it with the new weights and continue: versions stay clean but you pay an extra prefill. PipelineRL / Magistral: keep generating with the old KV: no wasted compute, but the sequence mixes "new weights + old KV", and PipelineRL measures only slightly higher KL than recomputing. Both require recording the sampling logprob and version of every token.
3. A KV fork saves the **prefill** before the decision point; checkpoint restoration (C3) saves the **decode** before the decision point (no regenerating history). Tool results and other agents' messages appended after the fork still need prefill, and the branch's own output still needs decode; for the prefix to hit, the branch must also be routed to the same instance that holds the KV.
4. KV = 100k × 256 KiB ≈ 26.2 GB, i.e. 26.2 / 570 ≈ 4.6% of available KV, so pinning it costs c ≈ 0.046 instance-seconds per second. B ≈ 3.6 instance-seconds (ignoring queueing after eviction). Under exponential durations pin iff B > c·E[T], i.e. E[T] < 3.6 / 0.046 ≈ 78 s. Common mistakes: forgetting that the KV is spread over 8 GPUs, or treating B as single-GPU time.
5. Each 10 s period has 5 requests and the first always misses: hit rate = 1 − 2/10 = 80%; with a 60 s period, 1 − 2/60 ≈ 97%. Keeping the previous version's KV for new requests makes everything but the first cold start hit (close to 100%), at the price that these requests' prefix was computed by old weights (higher KL, which must be measured as PipelineRL did) and that prefix KV for two versions must fit at once.
6. Sequence-level log W = 400×0.005 + 400×0.002 = 2.8, so W = e^2.8 ≈ 16.4: one sample's weight is amplified 16×, and the variance grows exponentially with length. The per-token ratios are about e^0.005 ≈ 1.005 and e^0.002 ≈ 1.002, far below C=2, so nothing is truncated. That is why long trajectories use per-token ratios instead of the sequence-level product (Module 06 Section 5.2).

</details>

## 13. Interview deep-dives (optional)

1. **"What is the biggest difference between agent and chat workloads for an inference engine?"**
   Conclusion: requests are strongly dependent with tool waits in between, so scheduling and caching must be done per program/session. Magnitude: a 50-turn session with every cache lost vs every cache hit differs ≈30× in prefill. Trade-off: session stickiness raises hits but unbalances load; LAS-style program-level priority often improves mean JCT when program sizes are heavy-tailed, but must avoid starvation, and the gain is not guaranteed under non-preemptible prefill and cache-locality constraints.
2. **"During a tool call, keep KV in GPU memory or swap it out?"**
   Conclusion: compare "the opportunity cost of holding, c·E[min(T, τ)]" with "the recompute/reload + queueing saved on a hit, B", and keep holding while the hazard rate h(τ) > c/B. Magnitude: compute occupancy with the Module 01 KV formula; under the setting of Question 4 in Section 12 it is worth pinning only if the mean tool duration is below about 78 s. Trade-off: a fixed TTL is simple but not robust to duration variance (Continuum uses the historical distribution + program-level FCFS); in-flight progress reports are more accurate but require harness changes (Ask the Tool).
3. **"Design a rollout system for agentic RL. What do you tackle first?"**
   Conclusion: first measure the rollout time breakdown (generation / environment / reward); usually the long tail and environments bottleneck first. Magnitude: generation often exceeds 90% (OpenRLHF's statement); DSec runs over 380,000 concurrent sandboxes in production. Trade-off: partial rollout vs fully async vs disaggregated (Section 5); each step trades algorithmic off-policy cost for system utilization.
4. **"If the model updates weights hourly while serving online, what must change in your inference infrastructure?"**
   Conclusion: version everything: KV cache, routing and evaluation baselines all carry a model version. Magnitude: with immediate flush, prefix hit rate = 1 − request interval / update period. Trade-off: a canary rollout usually runs two versions' instance pools side by side and splits traffic proportionally, with one version per GPU: reserve headroom in total capacity, and version KV and routing to avoid cross-version hits; rollback requires the old version's pool to come back quickly.
5. **"In a self-improving agent system, what is the most overlooked infrastructure cost?"**
   Conclusion: evaluation. Every candidate runs a benchmark, and overfitting to the benchmark must be prevented. Trade-off: cheaper evaluation misjudges more; successive halving / Hyperband are mature tools, but in RSI candidates share ancestors and their noise is correlated, so buy reliability with evolution tasks disjoint from the evaluation set and regression gating (HGM, ModularRSI, SEAGym).

## 14. Further reading

- [Autellix](https://arxiv.org/abs/2502.13965): the representative work on scheduling agent programs as first-class citizens; read how it defines program-level head-of-line blocking.
- [Continuum](https://arxiv.org/abs/2511.02230) and [Ask the Tool, Don't Guess](https://arxiv.org/abs/2609.18849): the "predict" and "observe" routes to keeping KV during tool calls.
- [KVFlow](https://arxiv.org/abs/2507.07400), [CacheScout](https://arxiv.org/abs/2608.14624), [TOPAS](https://arxiv.org/abs/2608.25523): three approaches to KV management for multi-agent workflows (declared / learned / joint scheduling).
- [ConServe](https://arxiv.org/abs/2606.01839): conversation-level P/D disaggregation, a new angle for the prefill direction.
- [PipelineRL](https://arxiv.org/abs/2509.19128): in-flight weight updates and measurements of stale KV, the starting point of the versioned-cache problem.
- [TreePO](https://arxiv.org/abs/2508.17445), [Tree-GRPO](https://arxiv.org/abs/2509.21240), [TreeRL](https://arxiv.org/abs/2506.11902): tree rollouts and prefix amortization, the direct predecessors of the system cost of counterfactual credit.
- [AReaL](https://arxiv.org/abs/2505.24298), [StreamRL](https://arxiv.org/abs/2504.15930), [RollArt](https://arxiv.org/abs/2512.22560), [Polar](https://arxiv.org/abs/2605.24220): four waypoints in the evolution of agentic RL systems.
- [DSec](https://arxiv.org/abs/2609.22978): a sandbox platform co-designed with RL, the first-hand reference when branches need environment snapshots.
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614), [AlphaEvolve](https://arxiv.org/abs/2506.13131): how the "evaluator" becomes the core component of a self-improvement loop.
- [SEAGym](https://arxiv.org/abs/2606.17546), [ModularRSI](https://arxiv.org/abs/2609.14857): evaluation and generalization problems of self-evolving harnesses.
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/): the problem framing of the RSI direction and the list of accepted papers.
- The project-wide reading list is at [docs/reading-list.en.md](../../docs/reading-list.en.md).
