"""mini_engine：真正执行计算的迷你推理引擎（numpy，CPU）。

和 codesign_sim（成本模型 / 模拟器）互补：这里每个机制都真的算——
分页 KV（PagedAttention 式）、按 block 哈希链的前缀缓存、continuous batching、
chunked prefill、recompute 抢占、asyncio 前端，以及一个多 agent harness 演示。
阅读顺序与架构说明见 capstone/README.md。
"""

from .async_engine import AsyncLLMEngine, EngineClosedError, GenerationResult, RealClock, VirtualClock
from .engine import EngineConfig, EngineStats, LLMEngine
from .kv_cache import BlockAllocator, PagedKVCache
from .model import TinyConfig, TinyTransformer, reference_greedy
from .prefix_cache import PrefixCache, compute_block_hashes, hash_block

__all__ = [
    "AsyncLLMEngine", "EngineClosedError", "GenerationResult", "RealClock", "VirtualClock", "EngineConfig", "EngineStats", "LLMEngine",
    "BlockAllocator", "PagedKVCache", "TinyConfig", "TinyTransformer", "reference_greedy",
    "PrefixCache", "compute_block_hashes", "hash_block",
]
