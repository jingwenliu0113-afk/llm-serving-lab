"""FIFO admission with a bounded queue (decision 029 "准入控制與超載保護").

- At most max_concurrency permits are out at any time; a permit covers the /tokenize call and the upstream request,
  so the gateway never has more than max_concurrency HTTP requests in vLLM (the research client's
  --max-concurrency semantics).
- Strict FIFO: a deque of futures. The fast path (no waiting) applies only when a slot is free and nobody waits;
  a released slot is handed to the oldest live waiter (in_flight does not drop), so a newcomer cannot overtake.
- A waiter waits at most min(max_queue_wait_s, deadline - now) (429 queue_timeout); a full queue is 429 queue_full.
- Never leaks a slot: a waiter that is cancelled or times out after its future already received the slot hands it on
  (release), and Permit.release is idempotent, so the endpoint, the response and the guard's scope cleanup can all
  call it.
- reject_waiters (backend down) and set_draining reject the queue with 503; while draining no newcomer is admitted.
  Requests already running finish (or are cancelled by the server's graceful-shutdown timeout).
- deadline and queued_s use the injected clock (time.monotonic by default); the waits themselves run on the event
  loop's timer.
"""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Callable, Mapping

import anyio

STATUS = {"queue_full": 429, "queue_timeout": 429, "backend_unavailable": 503, "draining": 503}
_GRANTED = "granted"   # a waiter's future result: the slot, or the reason it was rejected


class Rejection(Exception):
    """429 (healthy, at the configured capacity) or 503 (cannot serve now), with a fixed Retry-After per reason."""

    def __init__(self, reason: str, retry_after_s: int):
        super().__init__(reason)
        self.reason = reason
        self.status = STATUS[reason]
        self.retry_after_s = retry_after_s


class Permit:
    def __init__(self, ctrl: "AdmissionController", queued_s: float):
        self._ctrl = ctrl
        self._released = False
        self.queued_s = queued_s

    def release(self) -> None:
        """Idempotent; hands the slot to the oldest live waiter."""
        if not self._released:
            self._released = True
            self._ctrl._release()


class AdmissionController:
    def __init__(self, *, max_concurrency: int, max_queue: int, max_queue_wait_s: float,
                 retry_after_s: Mapping[str, int], clock: Callable[[], float] = time.monotonic):
        if max_concurrency < 1 or max_queue < 0 or max_queue_wait_s <= 0:
            raise ValueError("max_concurrency >= 1, max_queue >= 0 and max_queue_wait_s > 0 are required")
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.max_queue_wait_s = max_queue_wait_s
        self._retry_after_s = dict(retry_after_s)
        self._clock = clock
        self._in_flight = 0
        self._waiters: collections.deque[asyncio.Future] = collections.deque()
        self._idle = asyncio.Event()
        self._idle.set()
        self.draining = False

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queued(self) -> int:
        return sum(1 for f in self._waiters if not f.done())

    def _rejection(self, reason: str) -> Rejection:
        return Rejection(reason, self._retry_after_s[reason])

    def _take(self) -> None:
        self._in_flight += 1
        self._idle.clear()

    def _release(self) -> None:
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(_GRANTED)   # the slot moves to this waiter; in_flight is unchanged
                return
        self._in_flight -= 1
        if self._in_flight == 0:
            self._idle.set()

    async def acquire(self, *, deadline: float) -> Permit:
        """A permit, or Rejection. Cancellation-safe: a cancelled caller never holds or loses a slot."""
        if self.draining:
            raise self._rejection("draining")
        queued = self.queued
        if self._in_flight < self.max_concurrency and not queued:
            self._take()
            return Permit(self, 0.0)
        if queued >= self.max_queue:
            raise self._rejection("queue_full")
        start = self._clock()
        timeout = min(self.max_queue_wait_s, deadline - start)
        if timeout <= 0:
            raise self._rejection("queue_timeout")
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        outcome = None
        try:
            with anyio.move_on_after(timeout):
                outcome = await fut
        finally:
            if outcome is None:   # timed out, cancelled or failed before reading the outcome
                if fut.done() and not fut.cancelled() and fut.result() == _GRANTED:
                    self._release()   # the slot arrived while we were leaving: pass it on
                else:
                    fut.cancel()
                    self._discard(fut)
        if outcome is None:
            raise self._rejection("queue_timeout")
        if outcome != _GRANTED:
            raise self._rejection(outcome)
        return Permit(self, self._clock() - start)

    def _discard(self, fut: asyncio.Future) -> None:
        try:
            self._waiters.remove(fut)
        except ValueError:
            pass

    def reject_waiters(self, reason: str) -> None:
        """Every queued request gets Rejection(reason) (backend down or draining); running ones are untouched."""
        waiters, self._waiters = self._waiters, collections.deque()
        for fut in waiters:
            if not fut.done():
                fut.set_result(reason)

    def set_draining(self) -> None:
        self.draining = True
        self.reject_waiters("draining")

    async def wait_idle(self, timeout_s: float) -> bool:
        """Whether in_flight reached 0 within timeout_s."""
        with anyio.move_on_after(timeout_s):
            await self._idle.wait()
        return self._in_flight == 0
