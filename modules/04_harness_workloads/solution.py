"""模块 04 参考答案（接口与 exercise.py 完全一致；建议先自己写，卡住超过 10 分钟再来看）。

原练习说明：把 harness 的行为翻译成「引擎能复用多少 KV、要花多少钱」。

你要实现四个函数：

    chain_block_hashes(tokens, block_size, salt)            —— (a) 像 vLLM 一样，给 token 序列算「链式」block hash
    replay_prefix_cache(requests, block_size)               —— (b) 用一个无限大的前缀缓存重放一串请求，数复用/重算的 token
    compaction_breakeven_turns(stable_tokens, history_tokens, summary_tokens, ...)
                                                            —— (c) 压缩（compaction）会话历史要再跑多少轮才回本
    render_request(tools, system, history, volatile)        —— (d) 把一次请求渲染成确定性的字节串，并给出断点位置

写完后运行：
    make module N=04
    # 或
    .venv/bin/python -m pytest modules/04_harness_workloads

提示：
- 四个函数都是纯函数，不依赖 codesign_sim，结果确定。
- (a)(b) 是 README 第 2.3 节「前缀缓存为什么只认前缀」的代码版；(c) 是第 5.4 节「压缩的账」；
  (d) 是第 5.1 节规则 1、2、4、9（稳定在前、动态在后、确定性序列化、断点放在稳定前缀末尾）。
"""

from __future__ import annotations

import json
import math

# ---------------------------------------------------------------------------
# TODO (a)：链式 block hash
# ---------------------------------------------------------------------------


def chain_block_hashes(tokens: list[int], block_size: int, salt: int = 0) -> list[int]:
    """把 token 序列切成 block，给每个**完整** block 算一个链式 hash。

    规则（测试按这个规则算期望值）：
    1. 只处理完整 block：len(tokens) // block_size 个；末尾不满一个 block 的 token 不产生 hash
       （vLLM 也「只缓存满 block」）。
    2. 第 i 个 block 的 hash：
           h_i = hash((h_{i-1}, tuple(tokens[i*block_size:(i+1)*block_size])))
       其中 h_{-1} = salt。
    3. 返回 [h_0, h_1, ...]。block_size <= 0 → ValueError。

    关于 hash 函数：本练习用 Python 内置 hash 只是为了简单。int 元组的内置 hash 在同一个 Python 版本的
    不同进程之间是确定的，但**不保证**跨 Python 版本、跨平台一致，不能用作跨进程、跨机器共享的缓存键。
    真实系统用稳定的密码学 hash（vLLM 自 v0.11 起默认 sha256）。

    为什么要「链式」：h_i 依赖 h_{i-1}，所以第 j 个 token 一旦变了，第 j // block_size 个 block
    **以及它之后所有** block 的 hash 都会变 —— 这就是「system prompt 开头放一个时间戳，整段缓存全废」的原因。
    salt 相当于在链的根部塞了一个看不见的 token：不同租户用不同 salt，缓存就天然隔离（vLLM 的 cache_salt）。
    """
    if block_size <= 0:
        raise ValueError("block_size 必须为正")
    hashes: list[int] = []
    parent = salt
    for i in range(len(tokens) // block_size):
        block = tuple(tokens[i * block_size : (i + 1) * block_size])
        parent = hash((parent, block))  # 链式：当前 block 的 hash 依赖前一个 block
        hashes.append(parent)
    return hashes


# ---------------------------------------------------------------------------
# TODO (b)：用无限大的前缀缓存重放一串请求
# ---------------------------------------------------------------------------


def replay_prefix_cache(requests: list[list[int]], block_size: int) -> tuple[int, int]:
    """按顺序处理 requests（每个元素是一次请求的完整 prompt token 序列），返回 (reused_tokens, recomputed_tokens)。

    缓存模型（自托管引擎、最理想的情况：容量无限、永不淘汰，所有 session 共用一个缓存）：
    1. 每个请求先用 chain_block_hashes(prompt, block_size) 算出 block hash 列表（salt=0）。
       请在函数体内按名字调用 chain_block_hashes（测试会替换它来构造特殊的 hash 序列）。
    2. 命中长度 = 从第 0 个 block 开始，**连续**在缓存里的 block 数 m：遇到第一个不在缓存里的 block 就停，
       即使后面某个 block 的 hash 恰好在缓存里，也不算命中（前缀缓存只认前缀）。
    3. reused += m * block_size；recomputed += len(prompt) - m * block_size
       （没命中的 block 和末尾不满一个 block 的 token 都要重算 prefill）。
    4. 处理完一个请求后，把它所有完整 block 的 hash 加入缓存。
    5. requests 为空 → 返回 (0, 0)。

    例：block_size=2，requests = [[1,2,3,4,5], [1,2,3,4,5,6,7]]
        第 1 个：0 命中，重算 5；缓存 {h(1,2), h(3,4)}
        第 2 个：命中 2 个 block = 4 token，重算 3
        → 返回 (4, 8)
    """
    cache: set[int] = set()
    reused = recomputed = 0
    for prompt in requests:
        hashes = chain_block_hashes(prompt, block_size)
        matched = 0
        for h in hashes:  # 只数从头开始连续命中的 block
            if h not in cache:
                break
            matched += 1
        reused += matched * block_size
        recomputed += len(prompt) - matched * block_size
        cache.update(hashes)
    return reused, recomputed


# ---------------------------------------------------------------------------
# TODO (c)：压缩会话历史的盈亏平衡轮数
# ---------------------------------------------------------------------------


def compaction_breakeven_turns(
    stable_tokens: int,
    history_tokens: int,
    summary_tokens: int,
    read_mult: float = 0.1,
    write_mult: float = 1.25,
    output_mult: float = 5.0,
) -> float:
    """现在压缩会话历史，至少还要再跑多少轮 agent 循环才能在「钱」上回本？

    上下文 = 稳定前缀 P（tools + system + 项目上下文，stable_tokens）+ 会话历史 H（history_tokens），
    两者当前都在缓存里。压缩只把 H 换成 S 个 token 的摘要（summary_tokens），P 不动、仍然命中。

    所有价格都用「未缓存输入 token 的单价」当 1 个单位：
        read_mult   缓存命中读取的价格倍数（Anthropic 为 0.1）
        write_mult  写入缓存的价格倍数（Anthropic 5 分钟 TTL 为 1.25）
        output_mult 输出 token 相对输入 token 的价格倍数（Claude Sonnet 5 为 $10/$2 = 5）

    两个选项之后每轮追加的新 token 相同，只比较差额：
        一次性成本 = read_mult * (P + H)     # 做摘要的那次调用要读一遍整段上下文（命中缓存）
                   + output_mult * S         # 生成摘要
                   + write_mult * S          # 新上下文 P + S 里只有 S 要写缓存
        每轮节省   = read_mult * (H - S)     # 之后每轮少读 H - S 个缓存 token（P 两边都要读，抵消）
        回本轮数   = 一次性成本 / 每轮节省

    返回回本轮数（float，不取整）。summary_tokens >= history_tokens 时永远回不了本，返回 math.inf。
    任一 token 数为负，或任一倍数 <= 0 → ValueError。

    例：P=0, H=100_000, S=5_000，默认倍数 → (10_000 + 25_000 + 6_250) / 9_500 ≈ 4.34 轮；
        P=20_000, H=80_000, S=5_000 → (10_000 + 25_000 + 6_250) / 7_500 = 5.5 轮
        （同样 10 万 token 的上下文，稳定前缀越大，压缩越难回本）。
    """
    if min(stable_tokens, history_tokens, summary_tokens) < 0:
        raise ValueError("token 数不能为负")
    if min(read_mult, write_mult, output_mult) <= 0:
        raise ValueError("价格倍数必须为正")
    if summary_tokens >= history_tokens:
        return math.inf
    one_time = (
        read_mult * (stable_tokens + history_tokens)
        + output_mult * summary_tokens
        + write_mult * summary_tokens
    )
    per_turn_saving = read_mult * (history_tokens - summary_tokens)
    return one_time / per_turn_saving


# ---------------------------------------------------------------------------
# TODO (d)：确定性渲染 + 断点位置
# ---------------------------------------------------------------------------


def render_request(tools: list[dict], system: str, history: list[dict], volatile: dict) -> tuple[str, int]:
    """把一次请求渲染成确定性的字符串，返回 (rendered, stable_end)。

    渲染规则（每一行以 "\\n" 结尾，测试按这个规则算期望值）：
    1. 工具按 tool["name"] 排序，每个工具一行："TOOL " + canon(tool)。
    2. 一行 "SYSTEM " + system。
    3. stable_end = 到这里为止的字符数（含换行）。这就是显式缓存断点该放的位置：稳定前缀的末尾。
    4. 历史消息按原顺序，每条一行："MSG " + canon(message)。
    5. 最后一行："VOLATILE " + canon(volatile)（时间戳、剩余预算这类每次都变的东西，永远放在最后）。
    其中 canon(x) = json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))。

    工具名重复，或某个工具缺少 "name" 字段 → ValueError。

    性质（测试会检查）：
    - 工具列表的顺序、dict 的 key 顺序不影响输出（确定性序列化）；
    - 只改 volatile 时，rendered 在最后一行之前的部分完全不变；
    - 往 history 末尾追加消息时，旧请求去掉 VOLATILE 行之后，是新请求的前缀（只追加）。
    """
    def canon(x: object) -> str:
        return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    names = [t.get("name") for t in tools]
    if any(n is None for n in names) or len(set(names)) != len(names):
        raise ValueError("每个工具都要有唯一的 name")
    lines = [f"TOOL {canon(t)}" for t in sorted(tools, key=lambda t: t["name"])]
    lines.append(f"SYSTEM {system}")
    stable = "".join(line + "\n" for line in lines)
    body = "".join(f"MSG {canon(m)}\n" for m in history)
    tail = f"VOLATILE {canon(volatile)}\n"
    return stable + body + tail, len(stable)
