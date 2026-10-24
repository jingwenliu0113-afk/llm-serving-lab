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
- max_concurrency 0 (only with max_queue 0) is a closed controller: a verified refuse class (decision 030 8.7,
  C_PROD INFEASIBLE). Every acquire is a 503 class_not_serviceable at once, with no Retry-After (retrying cannot
  help), and nothing ever waits.
- deadline and queued_s use the injected clock (time.monotonic by default); the waits themselves run on the event
  loop's timer.
- Permit.release records released_s on that clock (044: the access log's t_released_s) and calls on_release with it
  before the slot moves on, so the next request's upstream interval never starts before this one's ended.

PolicyController (decision 044 "產品程式" items 2, 3; the candidate designs of 043 7.3, off unless the settings set
[admission] or [prefill_pacing], and only on a provisional study profile or a verified §12 one: capserve.config):
- One strict FIFO for every request, with head-of-line blocking: only the head is ever admitted, so a request that
  would fit never overtakes one that does not (a newcomer takes the fast path only when nobody waits).
- weighted-v1: each request asks for its class's units; the head is admitted when the units in flight plus its own
  stay within the budget and its class stays within its cap (043 7.3 A: 2K 1 unit, 8K 4, budget 32, caps 32 / 8).
- Prefill pacing (with one class: units 1, budget = its cap): a token bucket of burst tokens refilled at rate tokens/s,
  full at the start; the head must also find its prompt's tokens in it (043 7.3 B). A head with a free slot but too
  few tokens waits on the loop's timer for the deficit; that wait is its prefill_wait_s, the rest of its wait its
  queued_s (prefill_wait_s is None without a bucket). Tokens are taken at admission and never given back.
- Never leaks a slot: a waiter that leaves (timeout, cancel, rejection) is removed and the queue is pumped again, since
  a head that left may unblock the next; one that received the slot while leaving releases it (it moves on).
- reject_waiters, set_draining, wait_idle, the 429 / 503 reasons and their Retry-After are AdmissionController's.
"""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Callable, Mapping, NamedTuple

import anyio

STATUS = {"queue_full": 429, "queue_timeout": 429, "backend_unavailable": 503, "draining": 503,
          "class_not_serviceable": 503}
NOT_SERVICEABLE = "class_not_serviceable"   # a verified 0: never retried, so no Retry-After
_GRANTED = "granted"   # a waiter's future result: the slot, or the reason it was rejected


class Rejection(Exception):
    """429 (healthy, at the configured capacity) or 503 (cannot serve now), with a fixed Retry-After per reason (None
    for class_not_serviceable)."""

    def __init__(self, reason: str, retry_after_s: int | None):
        super().__init__(reason)
        self.reason = reason
        self.status = STATUS[reason]
        self.retry_after_s = retry_after_s


class Demand(NamedTuple):
    """What one request asks of a PolicyController (044): its class, its units and its prompt tokens (the bucket's
    cost)."""
    cls: str
    units: int
    tokens: int


class Permit:
    def __init__(self, ctrl, queued_s: float, *, demand: Demand | None = None, prefill_wait_s: float | None = None,
                 on_release: Callable[[float], None] | None = None):
        self._ctrl = ctrl
        self._released = False
        self._on_release = on_release
        self.queued_s = queued_s
        self.demand = demand
        self.prefill_wait_s = prefill_wait_s
        self.released_s: float | None = None

    def release(self) -> None:
        """Idempotent; records released_s (and calls on_release with it) before the slot moves on, then hands it to
        the oldest live waiter."""
        if not self._released:
            self._released = True
            self.released_s = self._ctrl._clock()
            if self._on_release is not None:
                self._on_release(self.released_s)
            if self.demand is None:   # AdmissionController: one slot per request
                self._ctrl._release()
            else:
                self._ctrl._release(self.demand)


class AdmissionController:
    def __init__(self, *, max_concurrency: int, max_queue: int, max_queue_wait_s: float,
                 retry_after_s: Mapping[str, int], clock: Callable[[], float] = time.monotonic):
        if max_concurrency < 0 or max_queue < 0 or max_queue_wait_s <= 0 or (max_concurrency == 0 and max_queue):
            raise ValueError("max_concurrency >= 0 (0 only with max_queue 0: a closed controller), max_queue >= 0 and "
                             "max_queue_wait_s > 0 are required")
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
        return Rejection(reason, None if reason == NOT_SERVICEABLE else self._retry_after_s[reason])

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

    async def acquire(self, *, deadline: float, on_release: Callable[[float], None] | None = None) -> Permit:
        """A permit, or Rejection. Cancellation-safe: a cancelled caller never holds or loses a slot."""
        if self.max_concurrency == 0:
            raise self._rejection(NOT_SERVICEABLE)
        if self.draining:
            raise self._rejection("draining")
        queued = self.queued
        if self._in_flight < self.max_concurrency and not queued:
            self._take()
            return Permit(self, 0.0, on_release=on_release)
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
        return Permit(self, self._clock() - start, on_release=on_release)

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


# ---------------------------------------------------------------- 044: the candidate policies

TOKEN_EPS = 1e-6   # a bucket within this of the head's prompt covers it (float refill), so a timer never spins


class TokenBucket:
    """043 7.3 B: burst tokens, refilled continuously at rate tokens/s up to burst; full at the start. Times are the
    caller's (the controller's injected clock)."""

    def __init__(self, burst: int, rate: float, *, now: float):
        if burst < 1 or rate <= 0:
            raise ValueError("a token bucket needs burst >= 1 and rate > 0")
        self.burst, self.rate = burst, rate
        self._tokens, self._at = float(burst), now

    def tokens(self, now: float) -> float:
        return min(float(self.burst), self._tokens + max(now - self._at, 0.0) * self.rate)

    def ready(self, n: int, now: float) -> bool:
        return self.tokens(now) >= n - TOKEN_EPS

    def take(self, n: int, now: float) -> None:
        self._tokens, self._at = self.tokens(now) - n, now

    def wait_s(self, n: int, now: float) -> float:
        """Seconds until n tokens are in the bucket (0 when they are)."""
        return max(n - self.tokens(now), 0.0) / self.rate


class _Waiter:
    __slots__ = ("fut", "demand", "short_since", "prefill_wait_s")

    def __init__(self, fut: asyncio.Future, demand: Demand):
        self.fut = fut
        self.demand = demand
        self.short_since: float | None = None   # since when, at the head with a free slot, it lacks only tokens
        self.prefill_wait_s = 0.0


class PolicyController:
    """weighted-v1 and prefill pacing on one strict FIFO (module docstring; 044, 043 7.3)."""

    def __init__(self, *, caps: Mapping[str, int], budget: int, max_queue: int, max_queue_wait_s: float,
                 retry_after_s: Mapping[str, int], bucket: TokenBucket | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if budget < 1 or max_queue < 0 or max_queue_wait_s <= 0 or not caps or min(caps.values()) < 1:
            raise ValueError("budget >= 1, every class cap >= 1, max_queue >= 0 and max_queue_wait_s > 0 are required")
        self.caps = dict(caps)
        self.budget = budget
        self.max_queue = max_queue
        self.max_queue_wait_s = max_queue_wait_s
        self.bucket = bucket
        self._retry_after_s = dict(retry_after_s)
        self._clock = clock
        self._in_flight = 0
        self._units = 0
        self._class_in_flight = dict.fromkeys(self.caps, 0)
        self._waiters: collections.deque[_Waiter] = collections.deque()
        self._timer: asyncio.TimerHandle | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self.draining = False

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def units_in_flight(self) -> int:
        return self._units

    @property
    def class_in_flight(self) -> dict:
        return dict(self._class_in_flight)

    @property
    def queued(self) -> int:
        return sum(1 for w in self._waiters if not w.fut.done())

    def _rejection(self, reason: str) -> Rejection:
        return Rejection(reason, self._retry_after_s[reason])

    def _fits(self, d: Demand) -> bool:
        return self._units + d.units <= self.budget and self._class_in_flight[d.cls] < self.caps[d.cls]

    def _has_tokens(self, d: Demand, now: float) -> bool:
        return self.bucket is None or self.bucket.ready(d.tokens, now)

    def _take(self, d: Demand, now: float) -> None:
        self._in_flight += 1
        self._units += d.units
        self._class_in_flight[d.cls] += 1
        if self.bucket is not None:
            self.bucket.take(d.tokens, now)
        self._idle.clear()

    def _release(self, d: Demand) -> None:
        self._in_flight -= 1
        self._units -= d.units
        self._class_in_flight[d.cls] -= 1
        if self._in_flight == 0:
            self._idle.set()
        self._pump()

    def _pump(self) -> None:
        """Admit from the head while it fits and the bucket covers its prompt; a head short only of tokens waits for
        them on the loop's timer; a head that does not fit blocks everyone behind it."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        now = self._clock()
        while self._waiters:
            w = self._waiters[0]
            if w.fut.done():   # rejected (or cancelled) while queued: no longer waiting
                self._waiters.popleft()
                continue
            if not self._fits(w.demand):
                return
            if not self._has_tokens(w.demand, now):
                if w.short_since is None:
                    w.short_since = now
                self._timer = asyncio.get_running_loop().call_later(self.bucket.wait_s(w.demand.tokens, now),
                                                                    self._pump)
                return
            self._waiters.popleft()
            self._take(w.demand, now)
            if w.short_since is not None:
                w.prefill_wait_s = now - w.short_since
            w.fut.set_result(_GRANTED)

    def _prefill(self, seconds: float) -> float | None:
        return seconds if self.bucket is not None else None

    async def acquire(self, *, deadline: float, demand: Demand | None = None,
                      on_release: Callable[[float], None] | None = None) -> Permit:
        """A permit for demand, or Rejection. Cancellation-safe, as AdmissionController.acquire."""
        if demand is None or demand.cls not in self.caps or demand.units < 1:
            raise ValueError(f"demand {demand!r} is not one of this policy's classes {sorted(self.caps)}")
        if self.bucket is not None and demand.tokens > self.bucket.burst:   # never covered: config refuses it (044)
            raise ValueError(f"a prompt of {demand.tokens} tokens exceeds the pacing burst {self.bucket.burst}")
        if self.draining:
            raise self._rejection("draining")
        start = self._clock()
        if not self.queued and self._fits(demand) and self._has_tokens(demand, start):
            self._take(demand, start)
            return Permit(self, 0.0, demand=demand, prefill_wait_s=self._prefill(0.0), on_release=on_release)
        if self.queued >= self.max_queue:
            raise self._rejection("queue_full")
        timeout = min(self.max_queue_wait_s, deadline - start)
        if timeout <= 0:
            raise self._rejection("queue_timeout")
        w = _Waiter(asyncio.get_running_loop().create_future(), demand)
        self._waiters.append(w)
        self._pump()   # it may be the head already, short only of tokens: start its pacing wait
        outcome = None
        try:
            with anyio.move_on_after(timeout):
                outcome = await w.fut
        finally:
            if outcome is None:   # timed out, cancelled or failed before reading the outcome
                if w.fut.done() and not w.fut.cancelled() and w.fut.result() == _GRANTED:
                    self._release(demand)   # the slot arrived while we were leaving: pass it on
                else:
                    w.fut.cancel()
                    self._discard(w)
                    self._pump()   # the head may have left: the next may fit now
        if outcome is None:
            raise self._rejection("queue_timeout")
        if outcome != _GRANTED:
            raise self._rejection(outcome)
        queued_s = max(self._clock() - start - w.prefill_wait_s, 0.0)
        return Permit(self, queued_s, demand=demand, prefill_wait_s=self._prefill(w.prefill_wait_s),
                      on_release=on_release)

    def _discard(self, w: _Waiter) -> None:
        try:
            self._waiters.remove(w)
        except ValueError:
            pass

    def reject_waiters(self, reason: str) -> None:
        waiters, self._waiters = self._waiters, collections.deque()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for w in waiters:
            if not w.fut.done():
                w.fut.set_result(reason)

    def set_draining(self) -> None:
        self.draining = True
        self.reject_waiters("draining")

    async def wait_idle(self, timeout_s: float) -> bool:
        with anyio.move_on_after(timeout_s):
            await self._idle.wait()
        return self._in_flight == 0
