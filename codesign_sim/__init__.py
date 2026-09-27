"""codesign_sim：Hello Agent Infra 配套的推理 / 训练系统教学模拟器（只依赖标准库 + numpy）。

子模块：
- hardware：GPU 规格（dense 算力）、互联常数与 roofline
- model：模型结构、KV cache 公式、FLOPs / 访存字节（MoE 按专家粒度）、通信、roofline 步时间
- workload：agent 多轮 session 负载生成与 block hash
- cache：按 block 的 radix 前缀缓存（lru / ttl / session_aware，可选 DRAM 第二层）
- engine：单实例 continuous batching + chunked prefill 模拟
- router：多实例路由（round_robin / least_loaded / cache_affinity / affinity_with_load_cap / session_sticky）
- pd：P/D 分离与共置对照
- rollout：RL rollout 的 sync / partial / async 调度（等资源比较）
- credit：多 agent 信用分配估计器
"""

from .cache import PrefixCache
from .credit import (
    TeamTask,
    credit_error_stats,
    estimate_credit,
    expected_gradient,
    gradient_bias,
    make_team_task,
    train_policies,
)
from .engine import Engine, EngineConfig, Request, simulate
from .hardware import GPU, GPUS, attainable_tflops, peak_tflops, ridge_point
from .model import (
    MODELS,
    ModelConfig,
    decode_bytes,
    decode_flops,
    kv_bytes_per_token,
    kv_capacity_tokens,
    prefill_flops,
    step_time_s,
)
from .pd import simulate_colocated, simulate_pd
from .rollout import RolloutConfig, compare_rollout_modes, simulate_rollout, simulate_rollout_seeds
from .router import route_and_simulate
from .workload import AgentSession, Turn, context_tokens, generate_sessions, session_stats, to_blocks

__version__ = "0.2.0"

__all__ = [
    "GPU", "GPUS", "ridge_point", "attainable_tflops", "peak_tflops",
    "ModelConfig", "MODELS", "kv_bytes_per_token", "prefill_flops", "decode_flops", "decode_bytes",
    "step_time_s", "kv_capacity_tokens",
    "Turn", "AgentSession", "generate_sessions", "to_blocks", "context_tokens", "session_stats",
    "PrefixCache",
    "Request", "EngineConfig", "Engine", "simulate",
    "route_and_simulate",
    "simulate_pd", "simulate_colocated",
    "RolloutConfig", "simulate_rollout", "simulate_rollout_seeds", "compare_rollout_modes",
    "TeamTask", "make_team_task", "estimate_credit", "train_policies", "credit_error_stats",
    "expected_gradient", "gradient_bias",
]
