[中文](README.md) · **English**

# Module 07: Multi-Agent Reward and Credit Assignment

> 🕐 Suggested time: 1.5 hours (Sections 7.3–7.5 and the paper table in Section 6 are optional and not counted) ｜ 🎯 After this module you can: derive when group baselines (GRPO / RLOO) are unbiased, when they reduce variance and when they actually hurt; name the sources of turn/step-level advantages for multi-turn agents; distinguish five families of multi-agent credit assignment (shared reward, difference reward, COMA counterfactual baseline, Shapley / leave-one-out, value decomposition) and state each one's bias conditions and extra evaluation count; judge whether counterfactual credit pays off in sync / async systems via "variance reduction factor vs useful-throughput reduction factor" ｜ 🧪 Lab: [`lab.py`](lab.py) ｜ 📖 Must-read primary sources: [COMA](https://arxiv.org/abs/1705.08926), [DeepSeekMath (GRPO, Section 4)](https://arxiv.org/abs/2402.03300)

## 0. One sentence

**Credit assignment turns a sparse, noisy team-level scalar reward into a learning signal usable "per agent, per step"; almost every more accurate split is paid for with more evaluations (extra rollouts, i.e. more trajectories sampled with the policy; critic forwards, i.e. value-network passes; judge calls, i.e. scoring-model invocations). So it is both an algorithm problem and a training-system throughput problem.**

## 1. Why big labs care

A concrete scenario: a planner + coder + tester multi-agent coding system trained with RL. Each episode has 20–50 tool calls, and the only reward is "did the tests pass" (0/1) at the end.

- **Sparse**: early in training the average success rate may be only 5%. With GRPO sampling G=8 per prompt, if every question has success rate 0.05, all-wrong groups make up 0.95⁸ ≈ 66%; their group advantages are all zero and the gradient compute is wasted. Real question difficulty is heterogeneous, which makes it worse: with the same mean success rate of 0.05, if success rates follow Beta(0.25, 4.75), zero-advantage groups make up 77% (Section 8). DAPO's Dynamic Sampling over-samples and filters out all-correct/all-wrong prompts ([DAPO](https://arxiv.org/abs/2503.14476)), at the price of more rollouts; questions with success rate 0 never fill up and can only be handled by curricula or filtering.
- **Misattribution**: under a shared reward the planner gets a positive gradient because the coder happened to be right. The VDN paper calls these "spurious reward signals" originating from teammates' behaviour, and describes the "lazy agent" phenomenon where one agent learns while another stops learning because its exploration would hurt the team ([VDN](https://arxiv.org/abs/1706.05296)); recent LLM work calls a redundant agent collecting positive reward free-riding ([CCPO](https://arxiv.org/abs/2603.21563)).
- **Expensive**: more accurate credit usually needs counterfactual evaluation, i.e. "what if this agent had acted differently / been absent". The COMA paper's summary of difference rewards is that each agent requires a separate counterfactual simulation ([COMA](https://arxiv.org/abs/1705.08926)). Multi-agent systems are already expensive: Anthropic reports its multi-agent research system uses about 15× the tokens of chat ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). Multiply by n counterfactual rollouts and it lands right on the long-tail critical path from [Module 06](../06_long_tail_async/README.en.md).

The reader is researching this direction: this module first lays out the algorithm family tree and each method's **bias conditions and system bill**, then uses experiments to look at variance, learning speed and throughput.

## 2. Outcome rewards: sparsity, variance and baselines

### 2.1 Policy gradient and baselines

The REINFORCE estimator is g = (R − b) ∇log π(a). As long as the baseline b does not depend on the current sample's action a, E[b ∇log π(a)] = b ∇Σπ = 0, so **a baseline changes the variance, not the expectation**.

Start with the second moment of the reward multiplier (R − b), which sets the magnitude of gradient variance (the exact gradient covariance is E[(R − b)² s sᵀ] − ∇J ∇Jᵀ with s = ∇log π(a); here we ignore the correlation between the size of s and R, only to estimate magnitudes). For a 0/1 reward with success rate p:

- no baseline: E[R²] = p
- the **true** mean p as baseline: E[(R − p)²] = p(1 − p), ratio **1 − p**
- a **leave-one-out** estimated baseline (RLOO): the baseline itself has variance p(1 − p)/(G − 1), ratio **(1 − p)·G/(G − 1)**
- subtracting the **group mean including oneself**: ratio (1 − p)·(G − 1)/G, which looks smaller, but it is a biased shrinkage (Section 2.2)

Three conclusions: at a low success rate a baseline saves almost no variance (even the true mean only saves 5% at p=0.05); **for p < 1/G, the estimated RLOO baseline actually increases the second moment** (ratio 1.086 at p=0.05, G=8); at a high success rate the baseline saves the most (down to about 1/10 at p=0.9). So at low success rates, first increase G, use a curriculum or filter, rather than counting on the baseline.

### 2.2 GRPO and RLOO: using same-group samples as the baseline

Sample G responses for the same prompt, with rewards r₁…r_G:

| Method | Advantage of sample i | Expected gradient |
|---|---|---|
| RLOO | Âᵢ = rᵢ − (1/(G−1)) Σ_{j≠i} rⱼ | The baseline excludes sample i itself: **strictly unbiased** ([Kool et al. 2019](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/); [Ahmadian et al. 2024](https://arxiv.org/abs/2402.14740)) |
| Mean centering (Dr. GRPO) | Âᵢ = rᵢ − mean(r), and drop 1/\|oᵢ\| from the objective | The mean contains rᵢ/G while the other samples are independent of aᵢ, so E[(rᵢ − mean)∇log π(aᵢ)] = (1 − 1/G)·∇J: **same direction, scaled down by (G−1)/G**. Dr. GRPO drops 1/\|oᵢ\| and std to remove the length and difficulty biases ([Dr. GRPO](https://arxiv.org/abs/2503.20783)) |
| GRPO (outcome supervision) | Âᵢ = (rᵢ − mean(r)) / std(r), shared by all tokens of the response | std depends on the whole group's actions including sample i, so it is no longer a constant scale: generally **biased**; it reweights questions, giving more weight to low-std (too-easy or too-hard) ones ([GRPO](https://arxiv.org/abs/2402.03300); DeepSeekMath samples 64 outputs per question in its experiments) |

An identity worth memorizing: rᵢ − mean_{j≠i} rⱼ = **G/(G−1) · (rᵢ − mean(r))**. The **sample values** of RLOO and "subtract the group mean" differ only by a constant scale, but the former's expectation is ∇J and the latter's is (G−1)/G·∇J. Exercise 1 asks you to verify it.

DeepSeekMath also gives a process-supervision version: each step's reward is first normalized across the whole group, and token t's advantage is the sum of normalized rewards of **all steps after it**. This is the simplest form of the step-level advantages in Section 3.

### 2.3 When group baselines fail

1. **All-correct or all-wrong groups**: all advantages are zero. Fraction = pᴳ + (1−p)ᴳ; the more heterogeneous the difficulty, the higher the fraction (Section 8, Part 0 of the lab). The DAPO paper points out that in a system that is synchronous with a non-pipelined generation stage, this over-sampling does not necessarily slow training, because generation time is typically dominated by long-tail samples: first-hand support for "extra rollouts fill the bubbles" in Part 4 of the lab.
2. **Different prompts within a group**: with multiple agents and turns, each agent sees a different prompt (role + interaction history) at each turn, so the premise of "comparing within a group" breaks ([AT-GRPO](https://arxiv.org/abs/2510.11062), Section 6).
3. **Group variance comes from others**: with a shared multi-agent reward, the within-group variance mixes in other agents' contributions; the baseline removes the common mean but not others' variance (Section 4 and the lab).

## 3. The time axis: turn/step-level advantages for multi-turn agents

A trajectory-level advantage broadcasts one scalar to every token / every turn. As trajectories grow, the problem shows: the TRACE authors point out that a failed rollout may contain many useful actions that moved the agent toward the goal, yet outcome-only training gives them the same negative advantage as the eventual mistake ([TRACE](https://arxiv.org/abs/2607.13988)).

| Approach | Where the signal comes from | Extra cost | Representative work |
|---|---|---|---|
| Trajectory-level broadcast | Outcome reward | 0 | GRPO, RLOO |
| Process reward model (PRM) scores each step | Human step labels (process supervision significantly beats outcome supervision on MATH; PRM800K releases 800,000 step labels); or auto-labels from N completions per step, scored by the fraction that reach the correct answer; or an implicit PRM updated online from outcome labels only | Train and run a PRM (auto-labeling also needs N completions per step); the PRM itself can be hacked | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [Math-Shepherd](https://arxiv.org/abs/2312.08935), [PRIME](https://arxiv.org/abs/2502.01456) |
| Learned value network | A critic | Critic training and memory | PPO; [VinePPO](https://arxiv.org/abs/2410.01679) finds PPO's value network often misjudges bad states as good on reasoning tasks |
| MC return of intermediate states | Re-sample K rollouts from state sₜ and average | K per state; each VinePPO step is 2–5× slower, yet wall-clock time to equal accuracy is shorter | [VinePPO](https://arxiv.org/abs/2410.01679) |
| Tree rollouts | Branch from intermediate steps; branches under the same prefix serve as each other's baseline | Choosing branch points and amortizing prefixes; TreePO amortizes common prefixes and saves 22%–43% of GPU hours | [TreeRL](https://arxiv.org/abs/2506.11902), [TreePO](https://arxiv.org/abs/2508.17445), [Tree-GRPO](https://arxiv.org/abs/2509.21240) |
| Regroup by identical states within a group | Find environment states that recur across the group's trajectories (anchor states) and compare actions there | ≈0, no extra rollouts | [GiGPO](https://arxiv.org/abs/2505.10978) |
| Turn-level reward design | Verifiable intermediate rewards or an LLM judge | Reward design / judge calls | [Wei et al. 2025](https://arxiv.org/abs/2505.11821) (dense per-turn rewards beat sparse terminal rewards on search and game agents) |
| TD on reference-model log-probs | A frozen reference model's log-prob of the gold answer, turned into state values, then TD differences | Reference-model forward | [TRACE](https://arxiv.org/abs/2607.13988) |
| Adaptive granularity | Mix step- and episode-level advantages by per-step NLL | ≈0 | [GACA](https://arxiv.org/abs/2609.12424) |

**Don't rush to "target"**: a 2026 controlled study finds that when the verifier only sees the final state (most turns' correctness is invisible, information density far below its phase boundary of about 0.8), spreading dense reward **uniformly** across turns beats concentrating the same advantage on "critical turns"; it also proposes a "shuffle positions, keep concentration" control that any claim of "precisely locating critical turns" should pass ([Coverage, Not Targeting](https://arxiv.org/abs/2609.02417)).

## 4. The agent axis: multi-agent credit assignment

Take n agents, joint action a = (a₁, …, aₙ), team reward R(a). Agent i's policy gradient uses its credit Cᵢ in place of R.

### 4.1 Shared reward

Cᵢ = R. Unbiased for the team objective, but the variance of Cᵢ contains **every other agent's contribution + observation noise**: with only unary contributions it grows roughly linearly with n; with pairwise synergy terms there are n(n−1)/2 of them, and variance grows superlinearly with n (Part 3 of the lab). Adding a group-mean baseline removes only the common mean, not others' variance.

### 4.2 Difference reward

Dᵢ = R(a) − R(a₋ᵢ, cᵢ): replace agent i's action with a default action cᵢ and see how much the team reward drops. This family of rewards was proposed by Wolpert and Tumer as the Wonderful Life Utility (see [Wolpert & Tumer 2001](https://doi.org/10.1142/S0219525901000188)).

- Benefit: the second term does not depend on aᵢ, so any action that increases Dᵢ also increases R; terms not involving i cancel between the two terms, cutting variance sharply. But synergy terms involving i do not cancel, so difference-reward variance also rises slowly with n.
- Cost (as summarized in the COMA paper): a simulator or reward model is needed to compute R(a₋ᵢ, cᵢ); one extra simulation per agent; the default action cᵢ is often hard to choose.
- **How the noise is paired matters**: if the counterfactual evaluation is re-run with independent randomness, it carries its own observation noise and Dᵢ's variance **gains an extra noise variance**; with common random numbers (the same environment seed, the same tool outputs), the two evaluations' noise is correlated and mostly cancels in the difference. Part 1 of the lab compares the two cases.

### 4.3 COMA counterfactual baseline

Aᵢ(s, a) = Q(s, a) − Σ_{a'ᵢ} πᵢ(a'ᵢ | τᵢ) Q(s, (a₋ᵢ, a'ᵢ))

A centralized critic replaces "swap in a default action" with "take the expectation over agent i's own actions under the current policy". The baseline does not depend on the realized aᵢ, so its expected contribution to the gradient is zero; the critic takes the other agents' actions as input and outputs a Q value for each of agent i's actions, so one forward pass suffices and the output size is |U| rather than |U|ⁿ ([COMA](https://arxiv.org/abs/1705.08926)). **Systems view: it trades "n extra simulations" for "one batched critic forward", with no real re-runs.** Note the first term of the advantage is the learned Q(s, a), not the true return, so errors in Q bias the advantage.

**How COMA lands in LLMs** depends on the granularity of an "action":

- **Token level**: the sum runs over the whole vocabulary (e.g. Qwen3-32B has 151,936 tokens), and an agent's decision is a whole message, not one token: infeasible.
- **Message level**: all messages cannot be enumerated, so you can only average over K sampled candidates, which degenerates into a "sampled counterfactual baseline", the same family as C3 / leave-one-out; every candidate needs a critic score or a follow-up rollout.
- **Tool-call level**: the action space can be enumerated (which tool, which arguments), but evaluating each candidate requires restoring the environment state (Section 7.3).

### 4.4 Shapley value and leave-one-out

The Shapley value ([Shapley 1953](https://doi.org/10.1515/9781400881970-018)) splits the team value v(N) by "average marginal contribution over all join orders":

φᵢ = Σ_{S ⊆ N∖{i}} |S|!(n − |S| − 1)!/n! · [v(S ∪ {i}) − v(S)]

It is the unique allocation satisfying efficiency (Σφᵢ = v(N) − v(∅)), symmetry, null player gets 0, and additivity. The cost is that exact computation needs **the values of 2ⁿ coalitions** (the original joint action is already one of them, so 2ⁿ − 1 extra); a Monte Carlo approximation with M permutations needs about M·n marginal evaluations, and **is only cheaper than the exact algorithm when M·n < 2ⁿ** (at n=4, M=8, 32 > 16, so it is more expensive). [Shapley Q-value](https://arxiv.org/abs/1907.05707) uses it as each agent's critic.

**Whether Shapley is biased as policy-gradient credit depends on how "absence" is defined**. Let the coalition value be v(S) = E_{a'₋S ∼ π}[f(a_S, a'₋S)], i.e. absent agents are marginalized under the **current policy**. Then:

1. For i ∉ S, v(S) does not depend on aᵢ, so E[v(S) ∇log πᵢ(aᵢ)] = 0;
2. For i ∈ S, the sample's a_{S∖i} is already drawn from π and absent agents are also averaged under π, so E[v(S) | aᵢ] = Qᵢ(aᵢ), hence E[v(S) ∇log πᵢ] = E[Qᵢ(aᵢ) ∇log πᵢ] = ∇ᵢJ;
3. The weights sum to Σ_S w_S = 1, so E[φᵢ ∇log πᵢ] = ∇ᵢJ. ∎

In other words, **when absent agents are marginalized under the current policy, Shapley credit is strictly unbiased for the policy gradient**; defining absence as uniformly random actions, a default action or "removing the agent" introduces bias (larger the further the policy is from the reference distribution). Part 1 of the lab shows both reference distributions.

Two systems reminders:

- One v(S) is often not "one rollout": marginalizing absent agents under π itself needs inner sampling (or averaging over K^{n−|S|} combinations); toy tasks compute it in closed form, but in real systems multiply by the number of rollouts per coalition.
- Leave-one-out (LOO) only looks at the marginal contribution of "joining last": v(N) − v(N∖{i}), which is a difference reward whose default action is "absent".

| Team structure | Example | LOO | Shapley |
|---|---|---|---|
| Complements (all required) | Success only if all 3 agents are right | 1 each, sum 3 > team value 1 | 1/3 each |
| Substitutes (either suffices) | Either of two testers finding the bug counts | 0 each | 1/2 each |
| Glove game | Agent 0 holds a left glove, agents 1 and 2 each hold a right glove | (1, 0, 0) | (2/3, 1/6, 1/6) |

In LLM multi-agent systems, "removing an agent" has an extra problem: it changes what downstream agents see. C3 argues that the measured difference then mixes in downstream agents' degradation under unfamiliar input, a bias that more samples cannot remove; instead it **fixes the full history and re-samples only this agent's action at the decision point**, then computes advantages with a leave-one-out baseline ([C3](https://arxiv.org/abs/2603.06859)). This distribution shift holds for LOO and also for every coalition S ≠ N in Shapley (which also removes agents).

### 4.5 Value decomposition (VDN / QMIX, briefly)

- VDN: Q_tot = Σᵢ Qᵢ, the team value decomposed additively over agents ([VDN](https://arxiv.org/abs/1706.05296)).
- QMIX: a mixing network combines the Qᵢ into Q_tot with a **structurally enforced monotonicity** (∂Q_tot/∂Qᵢ ≥ 0), so each agent's own argmax equals the joint argmax ([QMIX](https://arxiv.org/abs/1803.11485)).

These are value-based methods and are rarely used directly on LLMs (token-level Q values are hard to learn), but the idea of "decomposing the team value into agent terms" shows up in critic design. In an interview it is enough to explain the "additive → monotonic" progression.

### 4.6 Trade-off summary

| Method | Bias | Variance | Extra evaluations per sample | Requires | Fits |
|---|---|---|---|---|---|
| Shared reward | Unbiased | High, grows superlinearly with n | 0 | Nothing | Small n, tight coupling |
| RLOO group baseline | Unbiased | Removes the common mean; for p < 1/G the second moment actually grows | 0 (group samples are needed anyway) | Multiple samples per prompt | Default starting point |
| Mean centering (Dr. GRPO / MAGRPO) | Same direction, scaled by (G−1)/G | Same as above | 0 | Same as above | Same as above |
| GRPO (divide by std) | Generally biased (reweights questions) | Same as above | 0 | Same as above | Common in practice |
| Difference reward | Unbiased (aligned with the team objective) | Low; with independent noise, one extra noise variance | n | Simulator + default action | Resettable environments |
| COMA counterfactual baseline | Baseline unbiased; advantage biased if Q is wrong | Low | One batched critic forward (or n·(K−1) simulations) | An accurate centralized critic | Small action spaces |
| Exact Shapley | Unbiased when marginalizing under π; biased when absence means uniform / default / removal | Low | 2ⁿ − 1 coalitions × evaluations per coalition | Can evaluate any coalition | n ≲ 10 |
| MC Shapley | Same as above, plus sampling error | Medium | ≈ M·n (only worth it when M·n < 2ⁿ) | Same as above | Large n |
| LOO / agent removal | Unfair allocation for complements or substitutes; plus distribution shift in LLMs | Low | n | Agents can be removed | Redundancy detection |
| CCPO (remove one of two agents) | Same as above | Low | 1 | The other agent can answer alone | Think → Solve |
| Fixed-history re-sampling (C3) | Unbiased | Low | decision points × candidates × rollouts per candidate | Exactly restorable history | Text-only multi-agent |
| Value decomposition | Structural assumption | Low | Critic forward | Train a critic | Value-based methods |

## 5. LLM judges / rubrics as rewards, and reward hacking

Many multi-agent tasks have no verifiable answer (writing, research reports, discussion quality), so another model (a judge) has to score them against a rubric; and the policy will exploit the scorer's loopholes, which is reward hacking:

- **Judges have systematic biases**: the MT-Bench paper lists position, verbosity and self-enhancement biases and limited reasoning ability; at the same time GPT-4-class judges agree with human preferences over 80% of the time, the same level as agreement between humans ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).
- **Rubrics / checklists are steadier**: Rubrics as Rewards improves up to 31% relative on HealthBench over direct Likert scoring, and smaller judges align better with rubrics ([RaR](https://arxiv.org/abs/2507.17746)); RLCF splits instructions into checklists scored item by item and improves on all five benchmarks ([Checklists](https://arxiv.org/abs/2507.18624)). Kimi K2 combines verifiable rewards with a self-critique rubric reward and keeps updating the critic with on-policy rollouts from verifiable tasks ([Kimi K2](https://arxiv.org/abs/2507.20534)).
- **Reward hacking can be quantified**: in a synthetic setup, Gao et al. fit, with d = √KL(π‖π_init) on the x-axis, gold reward R(d) = d(α − βd) for best-of-n and R(d) = d(α − β log d) for RL; both rise and then fall ([Gao et al.](https://arxiv.org/abs/2210.10760)).
- **Judges are brittle**: "master keys" like ":" "." or "Let's solve this problem step by step." elicit false positive rewards from many judges, including proprietary models ([One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)).

**System cost**: judge calls are an inference workload in their own right, at least one per trajectory, multiplied by steps or items for per-step / per-rubric scoring, and they sit on the RL critical path. RollArt offloads stateless reward computation to serverless infrastructure ([RollArt](https://arxiv.org/abs/2512.22560)). Common defenses: KL constraints, judge ensembles, rubric decomposition + verifiable anchors (CCPO's SEPO keeps the external task outcome dominant), periodic gold-standard evaluation, and monitoring the divergence between "judge score" and "verifiable score".

## 6. Recent multi-agent LLM reinforcement learning (each paper verified, optional)

| Paper | Setting | How credit is split | What the system needs |
|---|---|---|---|
| [MALT](https://arxiv.org/abs/2412.01928) (2024-12) | Sequential generator → verifier → refiner roles | Each role is sampled repeatedly into a search tree; value iteration propagates final correctness back to each role; off-policy | Tree sampling: each role is re-sampled, sample count multiplies by the branching factor |
| [MAPoRL](https://arxiv.org/abs/2502.18439) (2025-02) | Several LLMs answer independently, then discuss over multiple turns | A verifier scores both the answer and the discussion, rewarding corrective and persuasive discussion | Multi-model co-training |
| [ReMA](https://arxiv.org/abs/2503.09501) (2025-03) | High-level meta-thinking agent + low-level reasoning agent | Hierarchical MARL; the multi-turn version uses a turn-level ratio and parameter sharing | The two roles share parameters for efficiency |
| [SWEET-RL](https://arxiv.org/abs/2503.15478) (2025-03) | An agent and a human collaborator solve tasks over multiple turns (ColBench) | Trains a critic with access to extra training-time information, giving step-level rewards | An extra critic |
| [SPIRAL](https://arxiv.org/abs/2506.24119) (2025-06) | Zero-sum game self-play | RAE: an exponential-moving-average baseline per game and per role; without RAE, "thinking collapse" appears after about 200 steps | An online multi-agent multi-turn RL system |
| [MAGRPO](https://arxiv.org/abs/2508.04652) (2025-08) | Dec-POMDP, writing and coding collaboration | G joint samples per turn; joint reward minus group mean (no std division); decentralized execution | G joint samples |
| [AT-GRPO](https://arxiv.org/abs/2510.11062) (2025-10) | Games, planning, coding, math | Group key = (environment, agent, turn); each agent samples K candidates per turn and executes the highest-reward one (tree sampling); reward = α·team + local | An independent resource pool per policy, each holding one RolloutWorker and one UpdateWorker in HybridFlow style; environments run in a CPU pool |
| [C3](https://arxiv.org/abs/2603.06859) (2026-03) | Math and code, two multi-agent topologies | Fixed-history re-sampling of the action at a decision point + checkpoint restore and rollout to the end + leave-one-out baseline | The paper reports checkpoint restoration avoids regenerating history prefixes: 418.4M tokens vs MAPPO's 616.1M, 32% fewer, and 32% less wall-clock time |
| [CCPO](https://arxiv.org/abs/2603.21563) (2026-03) | Think → Solve dual agents | Δ = R(joint) − R(Solver answers alone); a redundant agent gets 0, suppressing free-riding | One extra "solo answer" rollout per sample |

Two 2026 surveys work as maps: a [credit assignment survey](https://arxiv.org/abs/2604.09459) (covering January 2024 to July 2026) and a [survey of multi-agent RL through orchestration traces](https://arxiv.org/abs/2605.02801) (the author notes that as of May 2026 explicit message-level counterfactual credit remains sparse, and no RL training method was found for the "when to stop" decision).

## 7. System cost: counterfactual evaluation = extra rollouts

### 7.1 The bill

In units of "extra evaluations per training sample (one joint episode)":

| Method | Extra evaluations | Total episodes at n=4 agents, batch 256 |
|---|---|---|
| Shared / group baseline | 0 | 256 |
| CCPO (two agents, remove one) | 1 | 256 × 2 = 512 (n=2) |
| Difference / LOO | n | 256 × 5 = 1,280 |
| COMA (simulation instead of a critic) | n·(K−1) | K=4: 256 × 13 = 3,328 |
| Exact Shapley | 2ⁿ − 1 other coalitions (multiply by the inner samples if a coalition needs them) | 256 × 16 = 4,096 |
| MC Shapley (M permutations) | ≈ M·n | M=8: 256 × 33 = 8,448, more than exact |
| DAPO Dynamic Sampling | over-sample 1/(1 − pᴳ − (1−p)ᴳ)× | about 3× at p=0.05, G=8 |

If every evaluation is a full multi-turn episode, these multipliers are multipliers on rollout compute.

### 7.2 Interaction with the long tail

1. **More samples → longer tail**: synchronous step time is set by the slowest trajectory. Under a Pareto(α) tail the expected maximum of N samples grows roughly as N^{1/α}; at α=1.5, 5× the samples makes the slowest one about 5^{2/3} ≈ 2.9× longer in expectation. Part 4 of the lab gives simulated step times with `simulate_rollout` from [Module 06](../06_long_tail_async/README.en.md).
2. **Credit-group barrier**: a sample's credit can only be computed once the original trajectory and all n counterfactuals have finished, so the real unit is the "credit group", not a single trajectory. In an async system, if any trajectory in the group is aborted for being too stale, the whole group's credit must be discarded or re-run. Moreover, the original and counterfactual trajectories of the same task have positively correlated lengths (Seer observes that samples of the same prompt have similar output lengths, [Seer](https://arxiv.org/abs/2511.14617)), so a group tends to run long and drag the tail together.
3. **Longer dependency chains**: an outcome-level difference reward can only launch counterfactuals once the original joint action is known; in multi-turn tasks the critical path becomes "original trajectory up to the decision point + branch from the decision point to the end".
4. **Mitigations**: fork branches **immediately** at the decision point; let branches reuse the prefix (Section 7.4); branch only at critical decision points (budgeted counterfactuals, see Section 2.5 of [Module 09](../09_frontier/README.en.md)); replace simulation with a critic (COMA), turning rollouts into forwards; compute credit asynchronously, at the price of stale credit.

### 7.3 Counterfactuals over environment state (optional)

For the coding agent at the start of this module, the environment state (file system, processes, databases) is not in the text history. C3's premise is that "interaction histories are observable text with no hidden state"; it also notes itself that in general partially observable stochastic games, holding history fixed requires a simulator capable of restoring hidden environment state. Three routes:

| Route | How | Cost | When it works |
|---|---|---|---|
| Text-only replay | Restore the text before the decision point as-is (C3) | Inference compute only | Tools have no side effects, or tool results can be replayed verbatim |
| Sandbox snapshot / fork | DSec can take an incremental disk snapshot of a sandbox at any point and restore it as a new sandbox; a paused microVM saves its memory and execution state as a snapshot | One snapshot per branch point (disk increment, plus memory size for microVMs), branch count × concurrent sandboxes; the DSec paper publishes no snapshot latency | State lives on disk / in a VM and can be fully captured |
| Record and replay | Record every tool call's result and reuse it during replay (earlier DSec versions reconciled rollout state after GPU-job preemption with a command log, reusing recorded results for completed operations to avoid duplicate side effects of non-idempotent commands) | Log storage; new calls after the branch still execute for real | Calls before the branch point can be reused; with non-deterministic tools (network, time, randomness) the replayed state differs from a real re-run and introduces bias |

For concurrency, use Little's law: concurrent sandboxes = creation rate × mean lifetime. With n extra branches per sample, concurrent sandboxes grow by roughly (1 + n × branch lifetime / original lifetime). DSec supports over 380,000 concurrent sandboxes in production ([DSec](https://arxiv.org/abs/2609.22978); platform details in [Module 03 Section 5.3](../03_deepseek_stack/README.en.md#53-dsec-the-sandbox-platform-for-agentic-rl), sandbox scaling in [Module 06 Section 8](../06_long_tail_async/README.en.md#8-cost-4-scaling-sandboxes--environments-as-a-service)).

### 7.4 The bill of a KV fork (optional)

Split the history before the decision point into external tokens (prompt, tool results) H_ext and model-generated tokens H_gen; after the decision point each branch appends external tokens A_ext and generates B_gen:

| Approach | Prefill per branch | Decode per branch | Example (20k / 3k / 2k / 1k, 4 branches) |
|---|---|---|---|
| Rerun from scratch | H_ext + A_ext | H_gen + B_gen | prefill 88k, decode 16k |
| Checkpoint restoration (C3) | H_ext + H_gen + A_ext | B_gen | prefill 100k, decode 4k |
| KV fork (prefix hit) | A_ext | B_gen | prefill 8k, decode 4k |

- Checkpoint restoration saves the history's **decode**; a KV fork also saves the history's **prefill**; tool results and other agents' messages appended after the fork need prefill in every approach.
- A KV fork requires routing branches to the same instance that holds the prefix KV, and that KV must not be evicted before all branches are issued (cache affinity and TTL in [Module 05](../05_cache_prefill/README.en.md)).
- Memory: in paged / radix caches the branches share prefix blocks and only the post-fork blocks are per-branch, so occupancy is about kv_bytes/token × (prefix length + Σ branch suffix lengths), held until the credit-group barrier clears. Exercise 4 implements this bill.

### 7.5 How many models for the agents, and how to place them (optional)

| Dimension | M independent models | One shared model + role prompts |
|---|---|---|
| Weight sync | M × parameters × 2 bytes (BF16) per step: 3 × 7B = 42 GB | 1 × parameters × 2 bytes: 14 GB |
| Lower bound on sync time | about 0.84 s over one 400 Gb/s (50 GB/s) NIC; about 0.05 s within a node at H100's 900 GB/s NVLink | about 0.28 s / 0.016 s |
| Training side | Each policy has its own optimizer state and training workers; AT-GRPO gives each policy an independent resource pool (RolloutWorker + UpdateWorker) | One optimizer state; the roles' gradients mix and may interfere |
| Inference-side KV | Different models cannot share KV | Put the shared system prompt and tool definitions first and role instructions after, so the roles can share prefix KV |
| Loss and credit | Each model trains only its own tokens | In one multi-role transcript, each role's tokens use its own advantage and the other roles' tokens are context with loss mask 0: advantages must be a per-token tensor, not one scalar per sequence |
| On-policy | Each policy tracks staleness by its own version | One version |

Magnitudes and engineering techniques of weight sync (bucketing, delta sync, overlapping with training) are in [Module 06 Section 7](../06_long_tail_async/README.en.md#7-cost-3-deployment-pool-sizing-and-weight-synchronization).

### 7.6 Is it worth it: a break-even criterion for variance vs throughput

In the regime dominated by gradient noise, the number of samples needed for the same progress is roughly proportional to per-sample gradient variance normalized by signal size (the inverse of the signal-to-noise ratio); time = samples / useful samples per second. So

**T_method / T_baseline ≈ (SNR_baseline / SNR_method) / (q_method / q_baseline)**

where SNR is the single-sample gradient signal-to-noise ratio and q is useful (original) trajectories per hour. Counterfactual credit pays off in wall-clock time only when **the SNR gain exceeds the useful-throughput loss**. Plugging in the numbers from Parts 1 and 4 of the lab:

| Estimator (every counterfactual is a real rollout) | SNR gain | Sync: useful-throughput ratio → T ratio | Async: useful-throughput ratio → T ratio |
|---|---|---|---|
| difference (+n) | 3.0× | 0.50× → 0.66 (pays off) | 0.18× → 1.81 (loses) |
| COMA-style baseline with simulated counterfactuals (+n(K−1)) | 5.9× | 0.25× → 0.69 (pays off) | 0.07× → 2.46 (loses) |
| exact Shapley, policy reference (+2ⁿ−1) | 8.9× | 0.10× → 1.17 (loses) | — |

(n=6, K=4; numbers from Parts 1, 4 and 5 of the lab.) A synchronous system has long-tail bubbles to fill, so extra rollouts come at a discount; an asynchronous system has no bubbles and counterfactuals cost full price, so the only ways to win there are turning rollouts into critic forwards (COMA's original intent) or pushing down the cost of each counterfactual (KV forks, common random numbers). Note that the SNR here is measured on a toy task at a single policy point, only to demonstrate the criterion and the magnitudes, not to predict a real system; in a real system, re-measure both factors on your own workload.

## 8. Numbers you must be able to do in your head

| Quantity | Typical value | How to derive it | Source |
|---|---|---|---|
| Variance of a 0/1 reward | p(1−p), max 0.25 (p=0.5) | Bernoulli distribution | definition |
| Second moment of the reward multiplier with / without a baseline | true mean 1 − p; RLOO (1 − p)·G/(G − 1); 0.95 and 1.086 at p=0.05, G=8 | the estimated baseline adds p(1−p)/(G−1) of variance | derived in this module |
| Fraction of zero-advantage groups (mean success rate 0.05, G=8) | all questions at 0.05 → 66%; Beta(1, 19) → 70%; Beta(0.25, 4.75) → 77% | E[pᴳ + (1−p)ᴳ], closed form for Beta | derived in this module |
| DAPO-style over-sampling factor | about 3.0×, 3.4×, 4.4× for the three cases above | 1 / (1 − zero-advantage fraction) | derived in this module |
| RLOO vs mean centering | sample values differ by G/(G−1) (G=8 → 1.14); expected gradients ∇J and (G−1)/G·∇J | identity (Exercise 1) | derived in this module |
| Number of coalitions for exact Shapley | 2ⁿ: n=4 → 16, n=8 → 256, n=16 → 65,536 | number of subsets | definition |
| When MC Shapley is cheaper | M·n < 2ⁿ: M ≤ 3 at n=4, M ≤ 102 at n=10 | compare evaluation counts | derived in this module |
| Total evaluation multiplier for difference / LOO | n + 1 | 1 original + n counterfactuals | [COMA](https://arxiv.org/abs/1705.08926) |
| Growth of "the slowest one" with sample count under a heavy tail | N^{1/α}; α=1.5, 5× samples → about 2.9× | Pareto extremes | derived in this module |
| Weight sync per step for 3 independent 7B policies | 42 GB; lower bound about 0.84 s on a 50 GB/s NIC | 3 × 7B × 2 bytes | derived in this module |
| VinePPO per-step slowdown | up to 5× (RhoMath 1.1B), 2× (DeepSeekMath 7B) | K MC rollouts per state | [VinePPO](https://arxiv.org/abs/2410.01679) |

## 9. Lab

**How to run**: `make lab N=07` (or `.venv/bin/python modules/07_multi_agent_credit/lab.py`); about 5 seconds on an M1 laptop, fully deterministic.

**The task**: Part 0 uses pure-numpy Bernoulli rewards. Parts 1–4 use the cooperative game with known true contributions built by `make_team_task`: n agents each pick an action and the team only gets a scalar r = Σᵢ W[i, aᵢ] + Σ_{i<j} Q_ij(aᵢ, aⱼ) + noise (unary terms + pairwise synergy terms, noise std 0.5). Several accounting conventions:

- **oracle cf**: counterfactual queries return the noise-free f, as if there were a perfect simulator or critic (the library default).
- **noisy cf**: every counterfactual query is an independent noisy rollout (`cf_noise="independent"`), i.e. "really re-running".
- **CRN cf**: the counterfactual shares the actual rollout's exogenous noise (`cf_noise="common"`, common random numbers).
- **Shapley uniform / policy**: absent agents are averaged over uniformly random actions / over the current policy (`shapley_ref`). The library's Shapley uses noise-free coalition values and divides observation noise by n, which is the oracle convention; the two `rollouts` rows are computed by the lab itself: each coalition is estimated with **one noisy rollout**, the same convention as noisy cf.
- `extra evals` is how many extra evaluations per sample these queries would cost if implemented with real rollouts.

**Actual output** (the lab prints Chinese section titles; the English meanings are: 0 = group baselines for G samples of one prompt, (a) second moment of the reward multiplier relative to no baseline, (b) zero-advantage fraction and over-sampling factor at mean success rate 0.05; 1 = single-sample gradients under a mid-training non-uniform policy; 2 = REINFORCE training for 150 iterations, mean ± standard error over 20 seeds; 3 = early gradient variance as the number of agents grows; 4 = rollout cost of counterfactual evaluation in the Module 06 simulator; 5 = break-even criterion, a ratio below 1 means the method pays off):

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

**How to read it**:

0. **Part 0 (group baselines)**: Monte Carlo matches theory. For p < 1/G the RLOO second-moment ratio exceeds 1 (1.086 at p=0.05, exactly 1.000 at p=1/8); "subtract the mean including oneself" looks smaller (0.832) but it is a biased shrinkage. With the same mean success rate of 0.05, the more heterogeneous the difficulty, the more zero-advantage groups (0.663 → 0.704 → 0.770), and the over-sampling factor grows from 2.97× to 4.38×.
1. **Part 1 (gradient quality)**:
   - Every "marginal" estimator's expected gradient is aligned with the true gradient (cos = 1.000); the relative error of "subtract the group mean including oneself" is exactly 0.125 = 1/8, i.e. the (B−1)/B shrinkage at batch=8, with the direction unchanged.
   - SNR: group LOO is 2.3× shared; difference is 3.5× under the oracle convention, drops to 3.0× when each counterfactual is an independent noisy rollout, and rises to 4.1× with common random numbers; the COMA-style baseline drops from 6.6× to 5.9×.
   - Shapley: the oracle-convention 18.4× / 12.4× is unfair (noise-free coalition values, noise divided by n); with one noisy rollout per coalition it is 9.5× / 8.9×, at the price of 63 extra evaluations. **With uniformly random teammates as the reference, cos = 0.979 and relative error 0.203: biased; with the current policy as the reference, cos = 1.000: unbiased**, matching the derivation in Section 4.4.
2. **Part 2 (learning speed, 20 seeds)**:
   - The counterfactual family (COMA 35.5, difference 37.5 iterations) and policy-referenced Shapley (35.0) cover 80% of the gap fastest and all reach about 0.96 of optimal after 150 iterations; group LOO is in between (37.8 iterations, 0.947); shared is slowest (45.3 ± 5.1 iterations, 0.906).
   - Shapley uniform has the lowest variance (0.041) yet learns slowest and ends worst (51.2 ± 8.5 iterations, 0.898): **low variance cannot buy back a biased direction**.
   - COMA's early gradient variance is lower than difference's (0.133 vs 0.186), but the gap in learning speed (35.5 ± 3.2 vs 37.5 ± 4.6) is within the standard error: their expected gradients are identical, and once noise is this small, progress is limited by the learning rate and the expected gradient, so further variance reduction has diminishing returns. The "lower variance yet slower" seen with a handful of seeds is just seed noise.
   - Every method's return is still rising from iteration 75 to 150 (e.g. difference 0.915 → 0.958); final/best below 1 means training is not finished at 150 iterations, not that it stopped at a local optimum.
3. **Part 3 (number of agents)**: shared's gradient variance grows from 0.253 at n=2 to 4.362 at n=12 (6× the agents, about 17× the variance: superlinear); difference also grows (0.267 → 0.542) because synergy terms involving oneself do not cancel. At n=2 difference is slightly worse than shared (0.9×): with a poorly chosen default action, W[i, aᵢ] − W[i, 0] is itself a source of variance; at n=12 shared is 8.0× difference.
4. **Part 4 (system cost)**: each counterfactual is treated as one full extra trajectory in the rollout simulator from [Module 06](../06_long_tail_async/README.en.md) (`useful/h` counts only original episodes); every setting uses the same 64 units in colocated deployment, so this is an equal-resource comparison.
   - Synchronous: difference runs 7× the rollouts and useful throughput drops to 0.50×; COMA-style simulation 19× → 0.25×; exact Shapley 64× → 0.10×. Part of the extra rollouts fills long-tail bubbles (busy rises from 0.13 to 0.43 / 0.58 / 0.75), so the cost is sublinear. This matches the DAPO paper's observation that in synchronous systems generation time is dominated by the long tail, so extra sampling does not necessarily slow training.
   - Asynchronous: rollout slots are already busy (0.60), so counterfactuals cost close to full price: difference 0.18×, COMA-style simulation 0.07×. At n=6, even after making the system asynchronous, difference's useful throughput (1260 per hour) is below the shared reward's synchronous throughput (1415).
   - A larger batch means fewer versions per unit time, so difference's mean staleness under async (0.03) is actually lower than shared's (0.19); the staleness of the credit itself is not measured.
   - Where the simulator is optimistic: counterfactuals start at the same time as, and independently of, the original trajectories; it does not model the credit-group barrier (credit needs all n+1 trajectories of a group, and aborting any one invalidates the group), the positive length correlation between a task's original and counterfactual trajectories, or the savings from branches reusing the prefix KV.
5. **Part 5 (break-even)**: using SNR with "every counterfactual is a real rollout": difference has a T ratio of 0.66 in sync (about a third faster) and 1.81 in async (81% slower); COMA-style simulation 0.69 in sync, 2.46 in async; exact Shapley is 1.17 even in sync, not worth it. Conclusion: **counterfactual credit by real re-runs only pays off in synchronous systems that have bubbles to fill**; asynchronous systems must either turn rollouts into forwards with a critic (COMA's original intent) or push down the cost and noise of each counterfactual with KV forks, common random numbers and the like.

**Which parameter to change and what to look for** (all of these were actually run):

- In `part2_training`, change batch from 8 to 2: shared needs 103.6 ± 12.9 iterations to cover 80% of the gap, difference 57.2 ± 10.3, widening the gap.
- In `part1_gradient_quality`, use `make_team_task(..., pair_scale=1.0)`: stronger synergy lowers Shapley uniform's cos from 0.979 to 0.969.
- In `part4_system_cost`, add `tail_alpha=3.0` to `RolloutConfig` (lighter tail): in sync mode difference's useful-throughput ratio falls from 0.50 to 0.26 (fewer bubbles, so counterfactuals get closer to full price); async mode barely changes (0.18 → 0.19).

## 10. Exercise

Implement four functions in [`exercise.py`](exercise.py) (pure numpy, no simulator needed):

| Function | What to implement | What it tests |
|---|---|---|
| `group_advantages` | GRPO / mean-centering / RLOO group advantages; groups with std 0 return 0; any leading dimensions | Unbiasedness, numerical stability, the identity RLOO = G/(G−1) × mean centering |
| `difference_rewards` | Dᵢ = R(a) − R(a₋ᵢ, c), calling the reward function only n+1 times | The cost of counterfactual evaluation |
| `exact_shapley` | Exact Shapley values, each coalition evaluated once | Efficiency, symmetry, null player; the glove game |
| `branch_token_cost` | The prefill and decode token bill of rerun / checkpoint restoration / KV fork branching | Systems side: C3 saves decode, a KV fork saves prefill, appended content always needs prefill |

```bash
make module N=07                                                 # test your implementation (fails until finished, that's expected)
INFRA_SOLUTION=1 .venv/bin/python -m pytest modules/07_multi_agent_credit   # check the reference solution passes
```

## 11. Self-check questions

1. At p=0.05, G=8, why does the RLOO baseline not only fail to reduce variance but actually increase the second moment of the reward multiplier? What should you actually do?
2. The sample values of "mean centering" and RLOO differ only by a constant; why is one unbiased and the other biased? What about GRPO after dividing by std?
3. Under a shared reward, why does each agent's gradient variance grow with more agents, and superlinearly with synergy terms? Does the difference reward fully fix it?
4. Three agents must all be right to succeed. How much credit do LOO and Shapley give each agent? Which one sums to the team value?
5. Under what condition is the Shapley value unbiased as policy-gradient credit? Explain with a three-step derivation.
6. Why is "removing an agent" to measure its contribution biased in LLM multi-agent systems? How does C3 get around it? Does C3's premise hold for coding agents?
7. A difference-reward implementation re-runs counterfactual evaluations with independent random seeds. Compared with using the same seed, how does the variance of the credit change?
8. Counterfactual credit has 3.0× the single-sample SNR of the shared reward; useful throughput drops to 0.50× in a sync system and to 0.18× in an async system. Is it worth it in each case?

<details><summary>Answers</summary>

1. The RLOO baseline is estimated from the other G−1 samples and carries p(1−p)/(G−1) of variance, so the second-moment ratio is (1−p)·G/(G−1) = 0.95 × 8/7 ≈ 1.086 > 1; this holds whenever p < 1/G. The real difficulty is that there are almost no positive samples: increase G, use a difficulty curriculum or DAPO-style over-sampling that filters all-wrong groups, or make the signal denser with process / intermediate rewards.
2. The group mean contains rᵢ/G, which is correlated with aᵢ, so E[(rᵢ − mean)∇log π(aᵢ)] = (1 − 1/G)∇J: right direction, shrunk magnitude; RLOO's baseline excludes the sample itself and is independent of aᵢ, so its expectation is exactly ∇J. GRPO's std also depends on the whole group's actions (including aᵢ), so it is no longer a constant scale, is generally biased, and reweights questions (low-std questions get more weight).
3. The shared reward R contains every agent's contribution; each extra agent adds one unary variance term plus n−1 synergy terms involving it, n(n−1)/2 synergy terms in total, hence superlinear. The difference reward only cancels terms not involving i; synergy terms involving i remain, so its variance also rises slowly with n.
4. LOO: each gets v(N) − v(N∖{i}) = 1, summing to 3 ≠ 1; Shapley: 1/3 each, summing to 1. Shapley satisfies efficiency.
5. It is unbiased when coalition values marginalize absent agents under the current policy, i.e. v(S) = E_{a'₋S∼π}[f(a_S, a'₋S)]: (1) terms with i ∉ S do not depend on aᵢ and contribute zero in expectation; (2) terms with i ∈ S satisfy E[v(S)|aᵢ] = Qᵢ(aᵢ) and contribute ∇ᵢJ; (3) the Shapley weights sum to 1. Defining absence as uniform, a default action or removal is biased.
6. Removing an agent changes the input distribution of downstream agents; the difference mixes in downstream degradation under unfamiliar input, a bias more samples cannot remove. C3 fixes the full history, re-samples only this agent's action at the decision point and rolls out to the end, comparing with a leave-one-out baseline; the premise is that history can be restored exactly. A coding agent's environment state is not in the text, so it needs sandbox snapshots / forks or record-and-replay (Section 7.3); otherwise the premise fails.
7. With independent seeds, the counterfactual evaluation carries its own independent observation noise, so the difference gains about one extra noise variance σ²; with the same seed (common random numbers) the two evaluations' noise is correlated and mostly cancels in the difference.
8. By the criterion of Section 7.6: sync (1/3.0)/0.50 ≈ 0.67, about a third less wall-clock time, worth it; async (1/3.0)/0.18 ≈ 1.85, about 85% slower, not worth it (Part 5 of the lab gets 0.66 and 1.81 from unrounded numbers). An async system has no bubbles to fill, so counterfactuals are charged at full price and the SNR gain must exceed 1/0.18 ≈ 5.6× to pay off.

</details>

## 12. Interview deep-dives

1. **"Why can GRPO drop the value model? What does it cost?"**
   Conclusion: the mean (and std) of G samples of the same prompt serves as the baseline, saving a critic as large as the policy. Magnitude: DeepSeekMath uses 64 samples per question; with mean success rate 0.05 and G=8, zero-advantage groups are 66%–77% depending on the difficulty distribution. Trade-off: rollouts multiply by G; credit stays at trajectory level; subtracting a mean that includes oneself shrinks the gradient by (G−1)/G, dividing by std introduces the difficulty bias and 1/\|o\| the length bias (Dr. GRPO); use RLOO for strict unbiasedness.
2. **"Multiple agents share one outcome reward. How do you assign credit?"**
   Conclusion: start with a group baseline (zero extra cost); move to counterfactuals if variance is still too high. Magnitude: difference / LOO need n× evaluations, exact Shapley 2ⁿ − 1 coalitions, COMA one critic forward. Trade-off: whether the environment can be reset, whether history can be restored exactly (yes for text only; coding agents need sandbox snapshots), whether the critic can be learned accurately; Shapley is unbiased only when absent agents are marginalized under the current policy; removal-based counterfactuals have distribution shift in LLMs.
3. **"Will counterfactual rollouts kill training throughput? How would you design the system?"**
   Conclusion: yes, they land on the long-tail critical path behind a credit-group barrier; how much they cost depends on the system mode. Magnitude: at n=6 difference rewards make rollouts 7×; in the lab, useful throughput drops to 0.50× in sync mode (counterfactuals partly fill bubbles) and to 0.18× in async mode (close to full price); with a 3× SNR gain, sync is about a third faster and async about 80% slower. Trade-off: decide with "SNR gain vs throughput loss"; decision-point forks + prefix KV reuse, budgeted branching, asynchronous credit (stale credit), critic instead of simulation (bias risk).
4. **"Three agents: three 7B models or one 7B? How do you place rollout and update?"**
   Conclusion: first ask whether the roles need different capabilities and whether training needs independent schedules. Magnitude: three independent 7B models sync 42 GB per step, a lower bound of about 0.84 s on a 50 GB/s NIC; a shared model syncs 14 GB. Trade-off: independent models each get a resource pool (AT-GRPO's RolloutWorker + UpdateWorker), and memory, sync and optimizer state all triple, but gradients do not interfere; a shared model can share prefix KV but needs per-token advantages and per-role loss masks, and gradients may interfere.
5. **"Using an LLM judge as the reward, what worries you most? How do you monitor it?"**
   Conclusion: reward hacking and judge bias. Magnitude: in Gao et al.'s fits gold reward rises then falls with √KL; a single "master key" token can elicit a positive reward. Trade-off: rubrics / checklists are steadier but costlier (one call per rubric item); KL constraints limit exploration; monitor the divergence between judge and verifiable scores and run periodic gold-standard evaluations.
6. **"How do you estimate step-level advantages for multi-turn agents without extra rollouts?"**
   Conclusion: GiGPO's anchor-state grouping and GACA (reusing NLL already recorded by the rollout) need almost no extra rollouts; TRACE adds only a frozen reference-model forward. Magnitude: VinePPO-style MC makes each step 2–5× slower; tree rollouts save 22%–43% of GPU hours through prefix amortization (TreePO). Trade-off: anchor states require states to recur (fine for ALFWorld, WebShop; not necessarily for the open web); also pass the "uniform spreading" control first, since dense does not have to mean "targeted".

## 13. Further reading

- [COMA](https://arxiv.org/abs/1705.08926): origin of the counterfactual baseline, and the clearest discussion of what difference rewards cost.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): GRPO's original formulas, including the process-supervision version.
- [Back to Basics (RLOO)](https://arxiv.org/abs/2402.14740) and [Dr. GRPO](https://arxiv.org/abs/2503.20783): two key details of group baselines (unbiasedness, normalization biases).
- [VinePPO](https://arxiv.org/abs/2410.01679), [GiGPO](https://arxiv.org/abs/2505.10978), [TreePO](https://arxiv.org/abs/2508.17445): three routes to step-level credit: "spend more rollouts", "spend no extra rollouts", "amortize prefixes".
- [Coverage, Not Targeting](https://arxiv.org/abs/2609.02417): the control design to read before doing turn-level credit research.
- [MAGRPO](https://arxiv.org/abs/2508.04652), [AT-GRPO](https://arxiv.org/abs/2510.11062), [C3](https://arxiv.org/abs/2603.06859): three steps of multi-agent LLM RL from "shared group baseline" to "exact counterfactuals".
- [DSec](https://arxiv.org/abs/2609.22978): the first-hand systems reference when counterfactual branches need environment snapshots.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760), [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794): quantification and instances of reward hacking.
- [Credit assignment survey (2026)](https://arxiv.org/abs/2604.09459): the latest map organized by granularity and method.
- The project-wide reading list is at [docs/reading-list.en.md](../../docs/reading-list.en.md); open problems in this direction are in [Module 09](../09_frontier/README.en.md).
