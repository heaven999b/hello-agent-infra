"""小型 decoder-only Transformer（numpy，float64，确定性随机权重）。

结构与 Llama 系列一致的四个零件：RMSNorm、RoPE、GQA（grouped-query attention）、SwiGLU。
权重是固定 seed 的随机数——它不会"说人话"，但每一步都是真实的矩阵计算，
所以能用来验证推理引擎的正确性：分页 KV、前缀缓存、chunked prefill、抢占重算
都必须和"每次从头算整条序列"的朴素实现给出 token 级完全相同的 greedy 输出。
（logits 并不逐位相同：batch 组成、chunk 切法会改变浮点规约顺序，float64 下差异约 1e-15，
远小于 top-1 与 top-2 的间隔，所以 argmax 不变。GPU 上 bf16 的差异大得多，见 capstone/README.md。）

两条前向路径共用同一套注意力数学（_attend）：
- forward_dense：不用任何缓存，对整条序列做一次因果注意力（参考实现）；
- forward_paged：多条序列的新 token 拼成一个扁平 batch 过线性层，
  注意力按序列从分页 KV cache（见 kv_cache.py）里按 block table 读出历史 K/V。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 只用于类型标注，避免循环导入
    from .kv_cache import PagedKVCache


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8          # query 头数
    n_kv_heads: int = 2       # KV 头数（GQA：每 4 个 query 头共享 1 个 KV 头）
    ffn_dim: int = 384        # SwiGLU 中间维度
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 2 or self.ffn_dim <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size must be > 2 (0/1 are reserved), ffn_dim and n_layers > 0")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads (GQA groups)")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotate-half RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads


@dataclass
class LayerWeights:
    attn_norm: np.ndarray   # [d]
    wq: np.ndarray          # [d, H*hd]
    wk: np.ndarray          # [d, Hkv*hd]
    wv: np.ndarray          # [d, Hkv*hd]
    wo: np.ndarray          # [H*hd, d]
    ffn_norm: np.ndarray    # [d]
    w_gate: np.ndarray      # [d, f]
    w_up: np.ndarray        # [d, f]
    w_down: np.ndarray      # [f, d]


@dataclass
class SeqChunk:
    """一条序列在本步扁平 batch 里的那一段。"""

    start: int               # 在扁平 token 维中的起始下标
    length: int              # 本步新算的 token 数（prefill 块 或 decode 的 1 个）
    context_len: int         # 本步算完后该序列拥有 KV 的 token 总数
    block_table: list[int]   # 逻辑块号 -> 物理块号


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def apply_rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """x: [T, heads, hd]；按绝对位置旋转（rotate-half 写法，与 HF Llama 一致）。"""
    hd = x.shape[-1]
    inv_freq = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)      # [hd/2]
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]    # [T, hd/2]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _attend(q: np.ndarray, k: np.ndarray, v: np.ndarray, q_pos: np.ndarray) -> np.ndarray:
    """因果 GQA 注意力。q: [n, H, hd]；k, v: [ctx, Hkv, hd]（位置 0..ctx-1）；q_pos: [n] 绝对位置。"""
    n, n_heads, hd = q.shape
    ctx, n_kv, _ = k.shape
    g = n_heads // n_kv
    # 同一个 KV 头下的 g 个 query 头拼成 g*n 行，一次矩阵乘完成（这就是 GQA 省 KV 读取的地方）
    qg = np.ascontiguousarray(q.reshape(n, n_kv, g, hd).transpose(1, 2, 0, 3)).reshape(n_kv, g * n, hd)
    kt = np.ascontiguousarray(k.transpose(1, 2, 0))                  # [Hkv, hd, ctx]
    vt = np.ascontiguousarray(v.transpose(1, 0, 2))                  # [Hkv, ctx, hd]
    scores = (qg @ kt).reshape(n_kv, g, n, ctx) / np.sqrt(hd)        # [Hkv, g, n, ctx]
    future = np.arange(ctx)[None, :] > q_pos[:, None]                # [n, ctx] 因果掩码
    scores = np.where(future, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = (probs.reshape(n_kv, g * n, ctx) @ vt).reshape(n_kv, g, n, hd)
    return out.transpose(2, 0, 1, 3).reshape(n, n_heads * hd)


class TinyTransformer:
    def __init__(self, cfg: TinyConfig | None = None):
        cfg = cfg if cfg is not None else TinyConfig()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, hd, f = cfg.d_model, cfg.head_dim, cfg.ffn_dim

        def dense(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.standard_normal((fan_in, fan_out)) / np.sqrt(fan_in)

        self.embed = rng.standard_normal((cfg.vocab_size, d))
        self.layers = [
            LayerWeights(
                attn_norm=np.ones(d), wq=dense(d, cfg.n_heads * hd),
                wk=dense(d, cfg.n_kv_heads * hd), wv=dense(d, cfg.n_kv_heads * hd),
                wo=dense(cfg.n_heads * hd, d), ffn_norm=np.ones(d),
                w_gate=dense(d, f), w_up=dense(d, f), w_down=dense(f, d),
            )
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = np.ones(d)
        self.lm_head = dense(d, cfg.vocab_size)

    # ---- 共享零件 ----
    def _qkv(self, lw: LayerWeights, x: np.ndarray, positions: np.ndarray):
        c = self.cfg
        h = rms_norm(x, lw.attn_norm, c.norm_eps)
        q = (h @ lw.wq).reshape(-1, c.n_heads, c.head_dim)
        k = (h @ lw.wk).reshape(-1, c.n_kv_heads, c.head_dim)
        v = (h @ lw.wv).reshape(-1, c.n_kv_heads, c.head_dim)
        return apply_rope(q, positions, c.rope_theta), apply_rope(k, positions, c.rope_theta), v

    def _ffn(self, lw: LayerWeights, x: np.ndarray) -> np.ndarray:
        h = rms_norm(x, lw.ffn_norm, self.cfg.norm_eps)
        return (silu(h @ lw.w_gate) * (h @ lw.w_up)) @ lw.w_down

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return rms_norm(x, self.final_norm, self.cfg.norm_eps) @ self.lm_head

    # ---- 参考路径：无缓存，整条序列一次算完 ----
    def forward_dense(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        """返回每个位置的 logits [T, vocab]。"""
        ids = np.asarray(token_ids, dtype=np.int64)
        pos = np.arange(len(ids))
        x = self.embed[ids]
        for lw in self.layers:
            q, k, v = self._qkv(lw, x, pos)
            x = x + _attend(q, k, v, pos) @ lw.wo
            x = x + self._ffn(lw, x)
        return self._logits(x)

    # ---- 引擎路径：扁平 batch + 分页 KV ----
    def forward_paged(self, token_ids: np.ndarray, positions: np.ndarray, slot_mapping: np.ndarray,
                      chunks: list[SeqChunk], kv: PagedKVCache) -> np.ndarray:
        """一步前向：所有序列的新 token 拼成 [T] 一起过线性层；注意力从分页 KV 读历史。

        slot_mapping[i] = 第 i 个新 token 的 K/V 要写入的物理槽位（block_id * block_size + offset）。
        返回每条序列本段最后一个 token 的 logits [num_chunks, vocab]。
        """
        x = self.embed[token_ids]
        for layer_idx, lw in enumerate(self.layers):
            q, k, v = self._qkv(lw, x, positions)
            # 安全前提：每层先把本步**所有**序列的新 K/V 写进块，再做任何注意力。
            # 调度器因此可以在调度时就登记"本步会算满"的块，让同一步里后接纳的请求直接命中它们
            # （同一 prompt 的 G 份采样在同一步共享前缀）。改成"逐序列写完就算"会破坏这一点，
            # tests/test_mini_engine.py::test_forward_paged_writes_all_kv_before_attention 守着它。
            kv.write(layer_idx, slot_mapping, k, v)
            attn = np.empty((len(token_ids), self.cfg.n_heads * self.cfg.head_dim))
            for ch in chunks:                                # 再逐序列从块里读出完整历史
                sl = slice(ch.start, ch.start + ch.length)
                k_ctx, v_ctx = kv.gather(layer_idx, ch.block_table, ch.context_len)
                attn[sl] = _attend(q[sl], k_ctx, v_ctx, positions[sl])
            x = x + attn @ lw.wo
            x = x + self._ffn(lw, x)
        last = np.array([ch.start + ch.length - 1 for ch in chunks])
        return self._logits(x[last])


def reference_greedy(model: TinyTransformer, prompt: list[int], max_tokens: int) -> list[int]:
    """最朴素的 greedy：每生成一个 token 都把整条序列从头算一遍（O(n^2)，只用于验证）。"""
    seq = list(prompt)
    out: list[int] = []
    for _ in range(max_tokens):
        nxt = int(np.argmax(model.forward_dense(seq)[-1]))
        out.append(nxt)
        seq.append(nxt)
    return out
