"""模块 05 练习：cache × prefill 协同设计里最常被追问的四个决策。

你要实现四个函数：

    min_waste_action(...)          —— (a) 工具调用期间，这个请求的 KV 是 丢弃 / 留在显存 / 换到 CPU？（InferCept 的 min-waste 规则）
    route_min_est_ttft(...)        —— (b) 按「排队 + 分层搬运 + 未命中 prefill + decode 干扰」估计 TTFT 选实例
    route_affinity_with_cap(...)   —— (c) 带负载上限的亲和路由（affinity_with_load_cap 的核心决策）
    load_or_recompute(...)         —— (d) 这段前缀的 KV：从 DRAM/SSD/远端加载，还是直接重算？

写完后运行：
    make module N=05
    # 或
    .venv/bin/python -m pytest modules/05_cache_prefill

提示：
- 四个函数都是纯函数，不依赖 codesign_sim，结果确定。
- 对应 README：(a) 第 2.3 节，(b)(c) 第 3.2–3.3 节，(d) 第 4.2 节。
- 单位约定：token 数是 int；时间是秒；速率是 token/s；带宽是 GB/s（1 GB = 1e9 字节）。
- 所有输入都要做基本校验：负数、NaN 在这些路由/淘汰决策里没有物理意义，一律 ValueError。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# TODO (a)：InferCept 的 min-waste 决策
# ---------------------------------------------------------------------------


def min_waste_action(
    context_tokens: int,
    other_tokens: int,
    batch_tokens: int,
    intercept_s: float,
    prefill_tok_per_s: float,
    swap_tok_per_s: float,
) -> tuple[str, dict[str, float]]:
    """一个请求发起了工具调用（interception），预计 intercept_s 秒后回来。它的 KV 怎么处理最省？

    沿用 InferCept（ICML'24）的「显存浪费 = 占着显存却没干活的 token × 秒」口径（单位 token·s；
    论文里每一项都再乘每 token 的 KV 字节数 M，比较大小时可以约掉）：

        t_fwd    = context_tokens / prefill_tok_per_s          # 回来后重算这段上下文要多久
        discard  = t_fwd * context_tokens + t_fwd * other_tokens
                   # 重算期间：自己的 KV 在长出来，同时正在跑的其他请求（other_tokens）也被拖住
        preserve = intercept_s * context_tokens                # 整个工具调用期间一直占着显存
        swap     = 2 * (context_tokens / swap_tok_per_s) * batch_tokens
                   # 换出 + 换入两次拷贝，每次都拖住当前 batch 里的全部 token（batch_tokens）

    模型边界（README 2.3）：假设拷贝是同步的、完全不与计算重叠、只有一条共享 PCIe；
    真实系统里异步 DMA / 逐层加载会让 swap 更便宜，带宽争用会让它更贵。

    返回 (action, wastes)：
        action 是 "preserve" / "swap" / "discard" 中浪费最小的那个；
        并列时按 "preserve" > "swap" > "discard" 的顺序取前者（能不动就不动）；
        wastes = {"discard": ..., "preserve": ..., "swap": ...}，单位 token·s。
    任一速率 <= 0 或为 NaN，或任一 token 数 / 时间为负或 NaN → ValueError。
    """
    raise NotImplementedError("TODO (a): 按三个公式算浪费，取最小；并列时 preserve > swap > discard")


# ---------------------------------------------------------------------------
# TODO (b)：按估计 TTFT 选实例（带分层搬运与 decode 干扰）
# ---------------------------------------------------------------------------


def route_min_est_ttft(
    prompt_tokens: int,
    hits: list[tuple[int, int, int]],
    queue_s: list[float],
    prefill_tok_per_s: float,
    load_tok_per_s: tuple[float, float] = (math.inf, math.inf),
    decode_interference_s: list[float] | None = None,
) -> int:
    """给一个新请求选实例：估计 TTFT 最小者。

    参数：
        hits[i] = (hbm, dram, ssd)：实例 i 上这条 prompt 的前缀分别有多少 token 在 HBM / DRAM / SSD 里。
                 三段是**连续**的：前缀先是 hbm 个 token，接着 dram 个，再接着 ssd 个。
                 三段之和超过 prompt_tokens 时，按 hbm → dram → ssd 的顺序截到 prompt_tokens。
        queue_s[i]：实例 i 上排在它前面的工作还要多久做完（秒）。
        prefill_tok_per_s：未命中部分的 prefill 速率（所有实例相同）。
        load_tok_per_s = (DRAM 加载速率, SSD 加载速率)，单位 token/s；math.inf 表示加载不花时间。
        decode_interference_s[i]：同机 decode 抢算力导致的额外延迟（秒）；None 表示全为 0。

    估计（README 3.2）：
        对 DRAM 段和 SSD 段，各自取 min(加载时间, 重算时间)：加载比重算还慢时，就当作重算
        （这就是练习 (d) 的判断）。
        TTFT_i = queue_s[i] + t_dram_i + t_ssd_i + miss_i / prefill_tok_per_s + decode_interference_s[i]
        其中 miss_i = prompt_tokens − (截断后的 hbm + dram + ssd)。
    返回 TTFT_i 最小的实例下标；并列时取下标最小的。

    校验（都 → ValueError）：列表为空或长度不一致；prompt_tokens 为负；任一命中 token 数为负；
    queue_s / decode_interference_s 有负数或 NaN；prefill_tok_per_s 不是正的有限数；
    加载速率 <= 0 或为 NaN（math.inf 合法）。
    """
    raise NotImplementedError("TODO (b): 截断三段命中 → 各层取 min(加载, 重算) → 加上排队、未命中 prefill、decode 干扰 → argmin")


# ---------------------------------------------------------------------------
# TODO (c)：带负载上限的亲和路由
# ---------------------------------------------------------------------------


def route_affinity_with_cap(match_tokens: list[int], loads: list[float], load_cap: float) -> int:
    """带负载上限的亲和路由（codesign_sim.router 里 "affinity_with_load_cap" 策略的核心决策）。

    规则：
    1. 候选集 = 所有 loads[i] < load_cap 的实例。
    2. 候选集非空：选 match_tokens 最大的；并列时选 load 更小的；再并列选下标更小的。
    3. 候选集为空（所有实例都到上限了）：退化成 least-loaded —— 选 load 最小的；并列选下标更小的。
    4. 校验（→ ValueError）：列表为空或长度不同；match_tokens 有负数；loads 有负数、NaN 或 inf；
       load_cap 为负数或 NaN（math.inf 合法，表示没有上限）。

    直觉：命中缓存很值钱，但不能为了命中把一台机器压垮。load_cap 就是「愿意为缓存亲和付出多少排队」的旋钮：
    load_cap = inf 退化成纯 cache_affinity；load_cap 很小退化成 least_loaded。
    它和有界负载一致性哈希、SGLang router 的失衡阈值、llm-d 的 TTFT 负载闸门是同一个思路。
    """
    raise NotImplementedError("TODO (c): 先过滤 load < cap 的实例，再按 (-match, load, index) 选；都满了就选最闲的")


# ---------------------------------------------------------------------------
# TODO (d)：加载还是重算
# ---------------------------------------------------------------------------


def load_or_recompute(
    n_tokens: int,
    kv_bytes_per_token: int,
    bandwidth_gb_s: float,
    latency_s: float,
    prefill_tok_per_s: float,
) -> tuple[str, float, float]:
    """n_tokens 个 token 的 KV 在下层存储（DRAM / SSD / 远端）里有一份，是加载回 GPU 还是直接重算 prefill？

        t_load      = latency_s + n_tokens * kv_bytes_per_token / (bandwidth_gb_s * 1e9)
        t_recompute = n_tokens / prefill_tok_per_s

    返回 (choice, t_load, t_recompute)，choice = "load" if t_load < t_recompute else "recompute"
    （相等时选 "recompute"：不用多占一条 IO 通路）。
    n_tokens == 0 时 t_load = latency_s，t_recompute = 0.0，选 "recompute"。
    带宽、速率 <= 0 或为 NaN，或 token 数、字节数、延迟为负 → ValueError。

    推论（README 4.2）：忽略固定延迟时，加载更快 ⟺ bandwidth_gb_s > kv_bytes_per_token * prefill_tok_per_s / 1e9，
    右边就是以 GB/s 为单位的「盈亏平衡带宽」B*（若带宽用字节/秒表示，则不除 1e9）。
    每 token 的 KV 越小（GQA → MLA）、prefill 越慢（模型越大），越值得加载。
    """
    raise NotImplementedError("TODO (d): 算 t_load 和 t_recompute，比较大小")
