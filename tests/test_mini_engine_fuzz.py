"""随机差分测试（固定 seed）：随机配置 + 随机到达 + 随机中止，每步检查不变量，结束后与朴素重算逐 token 比对。

专门覆盖手写用例覆盖不到的交错：被抢占者持有共享块、重新接纳时命中自己留下的缓存、
同一步里多份完全相同的 prompt、chunk 切在块中间时的登记时机、中止一个正在 prefill 的请求。
"""

from __future__ import annotations

import math
import random

import pytest

from mini_engine import EngineConfig, LLMEngine, TinyTransformer, reference_greedy
from mini_engine.scheduler import RequestStatus

N_TRIALS = 60


@pytest.fixture(scope="module")
def model() -> TinyTransformer:
    return TinyTransformer()


def check_invariants(eng: LLMEngine) -> None:
    al, bs = eng.allocator, eng.config.block_size
    refs: dict[int, int] = {}
    for r in eng.scheduler.running:
        assert len(set(r.block_ids)) == len(r.block_ids), "block table has a duplicate physical block"
        assert len(r.block_ids) == math.ceil(r.num_computed_tokens / bs), "block table size != computed tokens"
        for b in r.block_ids:
            refs[b] = refs.get(b, 0) + 1
    for b in al.blocks:                                   # 引用计数守恒
        assert b.ref_count == refs.get(b.block_id, 0), (b.block_id, b.ref_count, refs.get(b.block_id, 0))
        assert (b.ref_count == 0) == (b.block_id in al.free_queue)
        if b.block_hash is not None:
            assert b.block_id in eng.prefix_cache.table[b.block_hash]
    for h, ids in eng.prefix_cache.table.items():
        assert ids and all(al.blocks[bid].block_hash == h for bid in ids)
    for r in eng.scheduler.waiting:
        assert r.block_ids == [] and r.num_computed_tokens == 0


def random_trial(model: TinyTransformer, trial: int, ref_cache: dict) -> tuple[int, LLMEngine]:
    rng = random.Random(trial)
    bs = rng.choice([4, 8, 16])
    max_len, max_tok = 48, rng.randint(1, 8)
    min_blocks = math.ceil((max_len + max_tok - 1) / bs) + 1
    chunked = rng.random() < 0.8
    tight = rng.random() < 0.5                          # 一半的 trial 显存很紧，逼出抢占
    cfg = EngineConfig(num_blocks=rng.randint(min_blocks, int(min_blocks * (1.5 if tight else 3))), block_size=bs,
                       max_num_batched_tokens=rng.randint(1, 48) if chunked else rng.randint(max_len + max_tok, 120),
                       max_num_seqs=rng.randint(2, 9), enable_prefix_caching=rng.random() < 0.85,
                       enable_chunked_prefill=chunked)
    eng = LLMEngine(model, cfg)
    shared = [rng.randint(2, 511) for _ in range(rng.randint(0, 36))]
    prompts: list[list[int]] = []
    for _ in range(rng.randint(3, 9)):
        if prompts and rng.random() < 0.35:
            prompts.append(list(rng.choice(prompts)))    # 完全相同的 prompt（同题多采样）
        else:
            prompts.append((shared + [rng.randint(2, 511) for _ in range(rng.randint(1, 16))])[:max_len])
    pending = sorted((rng.randint(0, 10), i) for i in range(len(prompts)))
    ids: dict[int, str] = {}
    step = 0
    while pending or eng.has_unfinished():
        while pending and pending[0][0] <= step:
            _, i = pending.pop(0)
            ids[i] = eng.add_request(prompts[i], max_tok)
        if rng.random() < 0.05 and eng.has_unfinished():  # 偶尔中止一个未完成的请求
            victims = list(eng.scheduler.running) + list(eng.scheduler.waiting)
            eng.abort_request(rng.choice(victims).request_id)
        eng.step()
        check_invariants(eng)
        step += 1
        assert step < 5000
    eng.allocator.assert_no_leak()
    mismatches = 0
    for i, rid in ids.items():
        req = eng.requests[rid]
        if req.status == RequestStatus.ABORTED:
            continue
        key = (tuple(prompts[i]), max_tok)
        if key not in ref_cache:
            ref_cache[key] = reference_greedy(model, prompts[i], max_tok)
        mismatches += req.output_token_ids != ref_cache[key]
    return mismatches, eng


def test_random_differential_with_invariants(model):
    ref_cache: dict = {}
    totals = {"preemptions": 0, "cached": 0, "evictions": 0, "aborted": 0}
    for trial in range(N_TRIALS):
        bad, eng = random_trial(model, trial, ref_cache)
        assert bad == 0, f"trial {trial}: {bad} requests differ from reference_greedy"
        totals["preemptions"] += eng.stats.preemptions
        totals["cached"] += eng.stats.cached_tokens
        totals["evictions"] += eng.allocator.num_evictions
        totals["aborted"] += eng.stats.aborted
    # 确认这组随机配置真的走到了抢占、命中、淘汰、中止这些路径
    assert all(v > 0 for v in totals.values()), totals
