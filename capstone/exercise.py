"""Capstone 练习：在真实执行的 mini_engine 上做三件 infra 工程师每天在做的事。

你要实现五个函数 / 方法（第一个是热身）：

    prefix_match_blocks(block_hashes, cached)                          —— (a0) 最长前缀命中了几个块
    pick_instance(prompt_token_ids, instance_hashes, loads, load_cap)  —— (a) cache-aware 多实例路由
    choose_victims(free_blocks, n, now)                                —— (b1) 会话感知淘汰的决策规则（纯函数）
    SessionAwareAllocator.pick_victims(candidates, n, now)             —— (b2) 把 (b1) 接进引擎的块分配器
    speculative_greedy(model, prompt, max_tokens, draft, k)            —— (c) greedy 投机解码

(b2) 改的是引擎内部：LLMEngine(model, cfg, allocator_cls=SessionAwareAllocator) 会用你的淘汰策略。
验收用差分测试：换了淘汰策略后，生成的 token 必须与默认 LRU 完全一致、结束时没有块泄漏，
同时 agent 负载下的 prefill 实算量要更少。

写完后运行：
    .venv/bin/python -m pytest capstone
    # 看参考答案：INFRA_SOLUTION=1 .venv/bin/python -m pytest capstone

提示：
- 块哈希链用 mini_engine.compute_block_hashes(token_ids, block_size)：只含满块，第 i 个 hash 已经编码了前 i+1 块的全部 token。
- KVBlock 上有 block_id、block_hash（None = 没挂可复用前缀）、pin_until、last_used、chain_pos、sessions；
  BlockAllocator.num_active_sessions(block) 返回还没结束的会话数（harness 在 agent 结束时调用 end_session）。
- BlockAllocator.pick_victims 收到的 candidates 已经排除了 pin 未到期和正被命中的块，按空闲队列顺序（LRU → MRU）给出。
- model.forward_dense(tokens) 返回每个位置的 logits [T, vocab]，第 t 行是"看完 tokens[:t+1] 之后下一个 token"的分布。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np  # noqa: F401  实现时会用到

from mini_engine import BlockAllocator, TinyTransformer, compute_block_hashes  # noqa: F401  实现时会用到
from mini_engine.kv_cache import KVBlock


# ---------------------------------------------------------------------------
# (a) cache-aware 多实例路由
# ---------------------------------------------------------------------------
def prefix_match_blocks(block_hashes: list[int], cached: set[int]) -> int:
    """从第 0 块开始逐块检查 hash 是否在 cached 里，返回连续命中的块数（遇到第一个不在的就停）。

    例：block_hashes = [h0, h1, h2]，cached = {h0, h2} → 1（h1 断了，h2 即使在也不算）。
    """
    raise NotImplementedError("TODO: 实现最长前缀块匹配")


def pick_instance(prompt_token_ids: list[int], instance_hashes: list[set[int]], loads: list[int],
                  load_cap: int, block_size: int = 16) -> int:
    """为一个请求选择推理实例（返回下标）。

    - instance_hashes[i]：路由器**认为**第 i 个实例缓存着的块 hash 集合（近似视图，可能过期）；
      loads[i]：它当前的负载；
    - 只在 loads[i] < load_cap 的实例里选：前缀命中块数最多的优先；
      命中相同时选 load 更小的；再相同选下标更小的；
    - 如果所有实例都 >= load_cap：退化为选 load 最小的（相同选下标更小的）。
    这就是"亲和 + 负载上限"：只看命中会把热门前缀的流量全压到一台机器上。
    """
    raise NotImplementedError("TODO: 实现 cache-aware 路由")


# ---------------------------------------------------------------------------
# (b) 会话感知淘汰：先写纯函数，再接进引擎的分配器
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreeBlockInfo:
    block_id: int
    last_used: float          # 最近一次引用计数降到 0 的时间
    pin_until: float          # 到这个时间之前不能淘汰
    chain_pos: int            # 在前缀链里的位置（0 = 链头；未登记为 -1）
    n_active_sessions: int    # 还有几个未结束的会话用过这个块（共享的 system prompt 块可能有很多个）
    cached: bool              # 是否挂着一段可复用的前缀（没挂的块淘汰了也不损失什么）


def choose_victims(free_blocks: list[FreeBlockInfo], n: int, now: float) -> list[int] | None:
    """从空闲块里选 n 个淘汰，返回 block_id 列表（按淘汰顺序）；可淘汰的不足 n 个时返回 None。

    规则（依次比较）：
    1. pin_until > now 的块不能淘汰；
    2. 没挂可复用前缀的块（cached=False）先淘汰——淘汰它们不损失任何命中；
    3. 活跃会话数少的先淘汰（0 = 所有用过它的会话都结束了）；
    4. LRU：last_used 小的先淘汰；
    5. 尾块先淘汰：chain_pos 大的先淘汰——同一条链同时释放时，先丢链头会让整条链都查不到；
    6. 最后按 block_id 小的先淘汰。
    纯 LRU 会把"正在等工具、马上就回来"的 agent 的 KV 当成最冷的数据淘汰掉——这正是 agent 负载的痛点。
    """
    raise NotImplementedError("TODO: 实现会话感知淘汰规则")


class SessionAwareAllocator(BlockAllocator):
    def pick_victims(self, candidates: Iterator[KVBlock], n: int, now: float) -> list[int] | None:
        """把每个候选 KVBlock 转成 FreeBlockInfo（n_active_sessions 用 self.num_active_sessions），
        再用 choose_victims 选出 n 个。返回 None 表示不够。"""
        raise NotImplementedError("TODO: 把会话感知淘汰接进分配器")


# ---------------------------------------------------------------------------
# (c) greedy 投机解码
# ---------------------------------------------------------------------------
DraftFn = Callable[[list[int], int], list[int]]


def speculative_greedy(model: TinyTransformer, prompt: list[int], max_tokens: int, draft: DraftFn,
                       k: int) -> tuple[list[int], int]:
    """用"草稿 + 一次 target 前向校验"做 greedy 投机解码，返回 (生成的 token, target 前向次数)。

    每一轮：
    1. kk = min(k, 还需生成的 token 数 - 1)；kk > 0 时让 draft(当前序列, kk) 提出 kk 个候选 token；
    2. 对"当前序列 + 候选"调用一次 model.forward_dense（计为一次 target 前向）；
    3. 从 logits 里取 target 在"当前序列最后一个位置"起每个位置的 argmax；
       从头比较，接受与 target 一致的最长候选前缀，再追加 1 个 target 自己的 token（纠正或额外奖励）。
    要求：输出必须与 mini_engine.reference_greedy(model, prompt, max_tokens) 完全一致；
    草稿全对时 target 前向次数 = ceil(max_tokens / (k + 1))，草稿全错时 = max_tokens；
    草稿每个位置以概率 α 猜对时，每次前向平均产出约 E(α, k) = (1 - α^(k+1)) / (1 - α) 个 token。
    """
    raise NotImplementedError("TODO: 实现 greedy 投机解码")
