"""asyncio 前端：一个后台循环不停 step，多个协程并发 `await generate(...)`。

- generate() 不直接碰引擎，而是把"加请求"投进收件箱（inbox），后台循环在两步之间统一处理。
  这样即使 step 放到线程里跑（step_in_executor=True），引擎状态也只在一个地方被修改，没有数据竞争；
  vLLM V1 的前端与 EngineCore 之间也是"发消息"而不是共享状态（它们在不同进程，用 ZMQ 通信）。
- 调用方被取消（asyncio.wait_for 超时、task.cancel()）时，generate 捕获 CancelledError，
  投递一条 abort：请求从调度队列里摘掉、块被释放，不会继续占算力，也不会把结果写到已取消的 Future 上。
- 后台循环每跑完一步就 `await asyncio.sleep(0)` 让出事件循环，正在等工具的协程、新到的请求都能插进来，
  这就是 continuous batching 能"中途加人"的前提。
- 默认 step 在事件循环线程里同步执行：一步几毫秒时没问题，但一个长 prompt 的 prefill 可能占住事件循环
  上百毫秒，期间所有协程都停摆。step_in_executor=True 把 step 放进单独的线程（NumPy 的底层运算会释放 GIL），
  事件循环保持响应。vLLM V1 更进一步，把 EngineCore 放进独立进程。
- VirtualClock：离散事件的虚拟时钟。引擎每步按成本模型推进虚拟时间，工具等待是"睡到某个虚拟时刻"。
  这样 harness 实验的结果与机器快慢、负载无关，逐次运行完全一致，可以写进测试。
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .engine import LLMEngine, StepRecord
from .scheduler import Request


class EngineClosedError(RuntimeError):
    """AsyncLLMEngine 已经关闭，请求没有完成。"""


def default_step_cost(n_tokens: int) -> float:
    """虚拟时钟下一步的耗时（秒）：1 ms 固定开销 + 0.1 ms/token，与本机实测的量级接近（见 capstone 第 6 节）。"""
    return 1e-3 + 1e-4 * n_tokens


class VirtualClock:
    def __init__(self, step_cost: Callable[[int], float] = default_step_cost):
        self.now = 0.0
        self.step_cost = step_cost
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self.on_sleep: Callable[[], None] | None = None   # 有人开始睡时通知引擎循环（它可能正空闲等待）

    def __call__(self) -> float:
        return self.now

    async def sleep(self, dt: float) -> None:
        if dt <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self.now + dt, next(self._seq), fut))
        if self.on_sleep is not None:
            self.on_sleep()
        await fut

    def has_sleepers(self) -> bool:
        while self._sleepers and self._sleepers[0][2].done():   # 被取消的睡眠者
            heapq.heappop(self._sleepers)
        return bool(self._sleepers)

    def advance(self, dt: float) -> None:
        self.now += dt
        self._wake()

    def advance_time(self, dt: float) -> None:
        """只推进时间、不唤醒睡眠者：引擎的 step 可能在线程池里跑，唤醒要回到事件循环线程做（wake_due）。"""
        self.now += dt

    def wake_due(self) -> None:
        self._wake()

    def jump_to_next(self) -> None:
        """引擎空闲时：直接跳到下一个睡眠者醒来的时刻。"""
        if self.has_sleepers():
            self.now = max(self.now, self._sleepers[0][0])
            self._wake()

    def _wake(self) -> None:
        while self._sleepers and self._sleepers[0][0] <= self.now:
            _, _, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)


class RealClock:
    """与 VirtualClock 同接口的真实时钟。"""

    def __call__(self) -> float:
        return time.monotonic()

    async def sleep(self, dt: float) -> None:
        await asyncio.sleep(dt)


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt_len: int
    output_token_ids: list[int]
    num_cached_tokens: int       # 第一次调度时前缀缓存命中的 token 数
    num_preemptions: int
    latency_s: float             # 引擎时钟下从加入到完成的时间


@dataclass(frozen=True)
class StepSpan:
    start: float                 # 引擎时钟
    end: float
    record: StepRecord


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine, *, step_in_executor: bool = False,
                 virtual_clock: VirtualClock | None = None, record_steps: bool = False):
        if virtual_clock is not None and engine.clock is not virtual_clock:
            raise ValueError("construct the LLMEngine with clock=virtual_clock so pins use virtual time")
        self.engine = engine
        self.step_in_executor = step_in_executor
        self.vclock = virtual_clock
        self.record_steps = record_steps
        self.step_log: list[StepSpan] = []
        self.step_in_flight = False
        self._futures: dict[str, asyncio.Future[GenerationResult]] = {}
        self._inbox: deque[tuple] = deque()
        self._ids = itertools.count()
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closing = False

    # ---------------- 生命周期 ----------------
    async def __aenter__(self) -> AsyncLLMEngine:
        self._wakeup = asyncio.Event()
        if self.step_in_executor:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-step")
        if self.vclock is not None:
            self.vclock.on_sleep = self._wakeup.set
            vclock = self.vclock
            self.engine.on_model_done = lambda n: vclock.advance_time(vclock.step_cost(n))   # 否则：循环在等新请求、协程在等虚拟时间 → 互相等死
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        """优雅关闭：等当前这一步做完再停，然后中止所有没完成的请求，并让等待者收到 EngineClosedError。"""
        assert self._task is not None and self._wakeup is not None
        self._closing = True
        self._wakeup.set()
        try:
            await self._task
        except Exception:                        # 后台循环已经因错误退出，错误已经发给了等待者
            pass
        for op in self._inbox:                   # 还没交给引擎的操作
            if op[0] == "add":
                self._fail(op[1], EngineClosedError("engine closed before the request started"))
            elif op[0] == "abort" and self.engine.abort_request(op[1]):
                self.engine.pop_request(op[1])   # 调用方已经取消、循环还没来得及执行的 abort
        self._inbox.clear()
        # 引擎里所有没完成的请求都要中止——包括调用方已取消、已经不在 _futures 里的那些
        for req in [*self.engine.scheduler.waiting, *self.engine.scheduler.running]:
            if self.engine.abort_request(req.request_id):
                self.engine.pop_request(req.request_id)
        for rid in list(self._futures):
            self._fail(rid, EngineClosedError("engine closed before the request finished"))
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self.vclock is not None:
            self.vclock.on_sleep = None
            self.engine.on_model_done = None

    # ---------------- 对外接口 ----------------
    async def generate(self, prompt_token_ids: list[int], max_tokens: int, pin_seconds: float = 0.0,
                       session_id: int | None = None, cache_salt: int = 0) -> GenerationResult:
        if self._task is None or self._task.done() or self._closing:
            raise RuntimeError("use `async with AsyncLLMEngine(engine) as ae:` first")
        self.engine.validate_request(prompt_token_ids, max_tokens, pin_seconds)   # 参数错误直接抛给调用方
        rid = f"async-{next(self._ids)}"
        fut: asyncio.Future[GenerationResult] = asyncio.get_running_loop().create_future()
        self._futures[rid] = fut
        self._post(("add", rid, list(prompt_token_ids), max_tokens, pin_seconds, session_id, cache_salt))
        try:
            return await fut
        except asyncio.CancelledError:
            self._futures.pop(rid, None)
            self._post(("abort", rid))           # 调用方不要结果了：别让它继续占算力和块
            raise

    def end_session(self, session_id: int) -> None:
        self._post(("end_session", session_id))

    # ---------------- 内部 ----------------
    def _post(self, op: tuple) -> None:
        self._inbox.append(op)
        assert self._wakeup is not None
        self._wakeup.set()

    def _fail(self, rid: str, exc: BaseException) -> None:
        fut = self._futures.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_exception(exc)

    def _apply_inbox(self) -> None:
        """只在两步之间、在事件循环线程里修改引擎状态。"""
        while self._inbox:
            op = self._inbox.popleft()
            if op[0] == "add":
                _, rid, prompt, max_tokens, pin, sid, salt = op
                if rid not in self._futures:     # 还没加进引擎就被取消了
                    continue
                try:
                    self.engine.add_request(prompt, max_tokens, rid, pin, sid, salt)
                except Exception as exc:         # noqa: BLE001  交给这个请求的等待者
                    self._fail(rid, exc)
            elif op[0] == "abort":
                if self.engine.abort_request(op[1]):
                    self.engine.pop_request(op[1])
            elif op[0] == "end_session":
                self.engine.end_session(op[1])

    async def _step(self) -> list[Request]:
        t0 = self.engine.clock()
        self.step_in_flight = True
        try:
            if self._executor is not None:
                finished = await asyncio.get_running_loop().run_in_executor(self._executor, self.engine.step)
            else:
                finished = self.engine.step()
        finally:
            self.step_in_flight = False
        if self.vclock is not None:
            self.vclock.wake_due()                   # 时间已在 engine.on_model_done 里推进到本步结束
        if self.record_steps and self.engine.last_step is not None:
            self.step_log.append(StepSpan(t0, self.engine.clock(), self.engine.last_step))
        return finished

    def _deliver(self, req: Request) -> None:
        self.engine.pop_request(req.request_id)
        fut = self._futures.pop(req.request_id, None)
        if fut is not None and not fut.done():   # 调用方可能已经取消了
            fut.set_result(GenerationResult(
                req.request_id, len(req.prompt_token_ids), list(req.output_token_ids),
                req.num_cached_tokens, req.num_preemptions, (req.finish_time or 0.0) - req.arrival_time))

    async def _idle(self) -> None:
        """引擎没活干：虚拟时钟下跳到下一个工具返回的时刻；否则睡到有新请求为止。"""
        assert self._wakeup is not None
        if self.vclock is not None and self.vclock.has_sleepers():
            for _ in range(2):                   # 先让刚被唤醒的协程把新请求投进来
                await asyncio.sleep(0)
            if not self._inbox:
                self.vclock.jump_to_next()
            return
        self._wakeup.clear()
        if not self._inbox and not self._closing:
            await self._wakeup.wait()

    async def _loop(self) -> None:
        try:
            while not self._closing:
                self._apply_inbox()
                if not self.engine.has_unfinished():
                    await self._idle()
                    continue
                for req in await self._step():
                    self._deliver(req)
                await asyncio.sleep(0)           # 让出事件循环
        except Exception as exc:                 # 引擎出错时别让调用方永远挂着
            for rid in list(self._futures):
                self._fail(rid, exc)
            raise
