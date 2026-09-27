"""mini_engine 的 asyncio 前端与多 agent harness。

这里没有墙钟断言：并发与重叠用"事件时间线"和虚拟时钟证明，机器再忙结果也一样。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import numpy as np
import pytest

from mini_engine import (AsyncLLMEngine, EngineClosedError, EngineConfig, LLMEngine, TinyTransformer,
                         reference_greedy)
from mini_engine.agent_harness import HarnessConfig, run_harness_sync


def _toks(seed: int, n: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, 512, n)]


@pytest.fixture(scope="module")
def model() -> TinyTransformer:
    return TinyTransformer()


async def _until(pred, limit: int = 100_000) -> None:
    """只靠让出事件循环来等条件成立（不用 sleep 计时）。"""
    for _ in range(limit):
        if pred():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


# ---------------- 前端基本正确性 ----------------
@pytest.mark.parametrize("in_executor", [False, True])
def test_async_concurrent_generate_matches_reference(model, in_executor):
    prompts = [_toks(s, n) for s, n in ((1, 30), (2, 45), (3, 17))]

    async def main():
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=32))
        async with AsyncLLMEngine(eng, step_in_executor=in_executor) as ae:
            res = await asyncio.gather(*(ae.generate(p, 6) for p in prompts))
        eng.allocator.assert_no_leak()
        assert not eng.requests                          # 前端取走结果后回收请求记录
        return res

    results = asyncio.run(main())
    assert [r.output_token_ids for r in results] == [reference_greedy(model, p, 6) for p in prompts]


def test_continuous_batching_admits_new_request_mid_flight(model):
    """长请求还在 decode 时到达的短请求，必须在长请求结束之前完成（不是等一批跑完再接下一批）。"""
    async def main():
        eng = LLMEngine(model, EngineConfig())
        async with AsyncLLMEngine(eng, record_steps=True) as ae:
            long_task = asyncio.create_task(ae.generate(_toks(5, 30), 60))
            await _until(lambda: eng.stats.steps >= 3)
            short = await ae.generate(_toks(6, 10), 3)
            assert not long_task.done()
            await long_task
        mixed = [s for s in ae.step_log if len(s.record.entries) == 2]
        return short, mixed

    short, mixed = asyncio.run(main())
    assert short.output_token_ids == reference_greedy(model, _toks(6, 10), 3)
    assert mixed                                          # 有步同时包含两个请求


# ---------------- 取消 / 超时 / 关闭 ----------------
def test_cancel_mid_prefill_aborts_and_engine_keeps_serving(model):
    async def main():
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=16))
        async with AsyncLLMEngine(eng) as ae:
            other = asyncio.create_task(ae.generate(_toks(7, 20), 8))
            victim = asyncio.create_task(ae.generate(_toks(8, 80), 20))
            await _until(lambda: any(r.num_computed_tokens > 0 for r in eng.scheduler.running
                                     if len(r.prompt_token_ids) == 80))
            victim.cancel()                               # 它还在 chunked prefill 中途
            with pytest.raises(asyncio.CancelledError):
                await victim
            res_other = await other
            again = await ae.generate(_toks(9, 12), 4)    # 引擎还能继续服务
            assert not ae._task.done()
        assert eng.stats.aborted == 1
        eng.allocator.assert_no_leak()
        return res_other, again

    res_other, again = asyncio.run(main())
    assert res_other.output_token_ids == reference_greedy(model, _toks(7, 20), 8)
    assert again.output_token_ids == reference_greedy(model, _toks(9, 12), 4)


def test_wait_for_timeout_does_not_break_other_requests(model):
    """调用方超时（asyncio.wait_for 超时时做的就是取消内层任务）不影响其他请求，块全部释放。

    不用真实的超时时长：在引擎跑过确定的步数后显式取消，结果与机器快慢无关。
    """
    async def main():
        eng = LLMEngine(model, EngineConfig())
        async with AsyncLLMEngine(eng) as ae:
            other = asyncio.create_task(ae.generate(_toks(10, 20), 30))
            slow = asyncio.create_task(ae.generate(_toks(11, 40), 400))
            await _until(lambda: eng.stats.steps >= 3)        # 两个请求都已经在引擎里跑
            slow.cancel()
            with pytest.raises(asyncio.CancelledError):
                await slow
            res = await other
        eng.allocator.assert_no_leak()
        return res

    assert asyncio.run(main()).output_token_ids == reference_greedy(model, _toks(10, 20), 30)


def test_close_right_after_cancel_still_frees_blocks(model):
    """竞态窗口：调用方取消后，abort 只是投进了 inbox，后台循环还没来得及执行就开始关闭。

    关闭时不能丢掉这条 abort，也不能只中止还在 _futures 里的请求——被取消的请求已经不在那里了。
    这个时序是确定的：cancel 之后立刻进入 __aexit__，它先置 _closing，再让出事件循环。
    """
    async def main():
        eng = LLMEngine(model, EngineConfig())
        ae = AsyncLLMEngine(eng)
        await ae.__aenter__()
        task = asyncio.create_task(ae.generate(_toks(11, 40), 400))
        await _until(lambda: eng.stats.steps >= 2)       # 请求已经在引擎里跑、占着块
        task.cancel()
        await ae.__aexit__(None, None, None)
        with pytest.raises(asyncio.CancelledError):
            await task
        eng.allocator.assert_no_leak()

    asyncio.run(main())


def test_result_for_already_cancelled_future_is_dropped(model):
    """竞态窗口：step 在线程里跑时调用方被取消，Future 已是 cancelled，但 generate 的清理还没来得及执行。

    这时后台循环交付结果必须跳过它，而不是 set_result 抛 InvalidStateError 把整个循环带崩。
    （这个窗口很难用公开 API 稳定复现，所以这里白盒地构造它。）
    """
    async def main():
        eng = LLMEngine(model, EngineConfig())
        async with AsyncLLMEngine(eng) as ae:
            rid = eng.add_request(_toks(14, 10), 1, request_id="race")
            fut = asyncio.get_running_loop().create_future()
            ae._futures[rid] = fut
            fut.cancel()
            eng.run_until_done()
            ae._deliver(eng.requests[rid])               # 不能抛异常
            res = await ae.generate(_toks(15, 10), 2)    # 循环仍然健康
        return res

    assert asyncio.run(main()).output_token_ids == reference_greedy(model, _toks(15, 10), 2)


def test_closing_with_pending_requests_fails_them_and_frees_blocks(model):
    async def main():
        eng = LLMEngine(model, EngineConfig())
        ae = AsyncLLMEngine(eng)
        await ae.__aenter__()
        pending = asyncio.create_task(ae.generate(_toks(12, 30), 500))
        await _until(lambda: eng.stats.steps >= 2)
        await ae.__aexit__(None, None, None)
        with pytest.raises(EngineClosedError):
            await pending
        eng.allocator.assert_no_leak()

    asyncio.run(main())


def test_step_in_executor_keeps_event_loop_responsive(model):
    """在事件循环线程里 step：step 期间别的协程一次都跑不了；放进线程后，事件循环在 step 期间照常运转。

    线程里的 step 先等一道门，只有心跳协程在 step 进行中真的跑了才会开门——直接证明响应性，与计算快慢无关。
    """
    async def ticks_during_steps(in_executor: bool) -> int:
        eng = LLMEngine(model, EngineConfig(max_num_batched_tokens=512))
        gate = threading.Event()
        if in_executor:
            orig_step = eng.step

            def gated_step():
                assert gate.wait(timeout=10), "step 进行中事件循环没有运行"
                return orig_step()
            eng.step = gated_step
        ticks = 0
        async with AsyncLLMEngine(eng, step_in_executor=in_executor) as ae:
            done = False

            async def heartbeat():
                nonlocal ticks
                while not done:
                    if ae.step_in_flight:
                        ticks += 1
                        gate.set()
                    await asyncio.sleep(0)

            hb = asyncio.create_task(heartbeat())
            await ae.generate(_toks(13, 400), 2)
            done = True
            await hb
        return ticks

    assert asyncio.run(ticks_during_steps(False)) == 0
    assert asyncio.run(ticks_during_steps(True)) > 0


# ---------------- harness（虚拟时钟，确定性） ----------------
def test_harness_is_deterministic_and_overlaps_compute_with_tool_waits(model):
    cfg = HarnessConfig(n_agents=4, n_turns=3, output_len=4)
    a, b = run_harness_sync(model, cfg), run_harness_sync(model, cfg)
    assert (a.outputs, a.prefill_computed, a.steps, a.elapsed_s) == (b.outputs, b.prefill_computed, b.steps, b.elapsed_s)
    assert a.steps_during_others_tool_wait() > 0          # 某个 agent 等工具时，引擎在为别人算
    assert a.elapsed_s < 0.5 * a.tool_wait_sum_s          # 虚拟时间下的重叠：与机器快慢无关
    serial = run_harness_sync(model, replace(cfg, concurrent=False))
    assert serial.steps_during_others_tool_wait() == 0
    assert serial.elapsed_s > serial.tool_wait_sum_s
    assert serial.outputs == a.outputs                    # 交错方式不影响生成结果


def test_virtual_clock_wakes_idle_loop_when_someone_starts_sleeping(model):
    """引擎空闲等新请求时，有协程晚几拍才开始虚拟睡眠：循环必须被叫醒去推进虚拟时间，而不是互相等死。"""
    from mini_engine import VirtualClock

    async def main():
        clock = VirtualClock()
        eng = LLMEngine(model, EngineConfig(), clock=clock)
        async with AsyncLLMEngine(eng, virtual_clock=clock) as ae:
            first = await ae.generate(_toks(16, 10), 2)
            for _ in range(10):                           # 模拟 harness 在调用工具前还有几步异步处理
                await asyncio.sleep(0)
            await clock.sleep(0.5)                        # 此时后台循环正空闲等待
            second = await ae.generate(_toks(17, 10), 2)
        return first, second, clock.now

    first, second, now = asyncio.run(asyncio.wait_for(main(), timeout=10))
    assert now >= 0.5 and second.output_token_ids == reference_greedy(model, _toks(17, 10), 2)


def test_append_only_harness_maximizes_prefix_hits(model):
    cfg = HarnessConfig(n_agents=4, n_turns=3, output_len=4)
    rep = {s: run_harness_sync(model, replace(cfg, strategy=s)) for s in ("append", "compress", "timestamp")}
    assert rep["append"].hit_rate > rep["compress"].hit_rate > 0.0
    assert rep["timestamp"].cached_tokens == 0           # 最前面一个 token 变了，整条前缀失效
    assert rep["append"].prefill_computed < rep["compress"].prefill_computed < rep["timestamp"].prefill_computed


def test_pin_during_tool_calls_reduces_recompute_under_memory_pressure(model):
    tight = EngineConfig(num_blocks=36, max_pinned_frac=0.5)
    rep = {p: run_harness_sync(model, replace(HarnessConfig(), pin_policy=p), tight) for p in ("none", "oracle", "fixed")}
    assert rep["oracle"].prefill_computed < rep["none"].prefill_computed
    assert rep["fixed"].prefill_computed < rep["none"].prefill_computed
    assert rep["oracle"].evictions < rep["none"].evictions
    assert rep["none"].outputs == rep["oracle"].outputs == rep["fixed"].outputs


def test_virtual_clock_latency_includes_last_step(model):
    """虚拟时钟下：完成时间要包含最后一步的耗时，latency_s 与 step 耗时之和一致。"""
    from mini_engine.async_engine import VirtualClock, default_step_cost

    async def main():
        vc = VirtualClock()
        eng = LLMEngine(model, EngineConfig(), clock=vc)
        async with AsyncLLMEngine(eng, virtual_clock=vc) as ae:
            res = await ae.generate(_toks(21, 20), 3)
        return res, eng

    res, eng = asyncio.run(main())
    # 单个请求：第 1 步 prefill 20 token 并采样第 1 个输出，之后每步 decode 1 token，共 3 步
    expected = default_step_cost(20) + 2 * default_step_cost(1)
    assert abs(res.latency_s - expected) < 1e-9


@pytest.mark.parametrize("strategy", ["timestamp", "compress"])
def test_harness_tokens_stay_inside_small_vocab(strategy):
    """小词表（远小于默认 512）下，harness 生成的时间戳 / 摘要 token 不越出词表。"""
    from mini_engine import TinyConfig
    small = TinyTransformer(TinyConfig(vocab_size=64))
    cfg = replace(HarnessConfig(), strategy=strategy, n_agents=2)
    run_harness_sync(small, cfg)                          # 越界会在 embedding 查表时直接 IndexError
