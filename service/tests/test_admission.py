"""Tests for capserve/admission.py: FIFO admission with a bounded queue that never exceeds max_concurrency and never
loses a permit on cancellation, timeout or rejection (decision 029 "准入控制與超載保護"); the release time a
permit records before its slot moves on, and decision 044's candidate policies on one strict FIFO with head-of-line
blocking: weighted-v1 (a budget of units, a cap per class) and prefill pacing (a token bucket the head's prompt must
fit)."""

import asyncio
import random
import time

import pytest

from capserve.admission import AdmissionController, Demand, PolicyController, Rejection, TokenBucket

pytestmark = pytest.mark.anyio
RETRY = {"queue_full": 2, "queue_timeout": 3, "backend_unavailable": 5, "draining": 7}
FAR = 1e12


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def ctrl(n=1, q=8, wait=60.0, clock=None):
    return AdmissionController(max_concurrency=n, max_queue=q, max_queue_wait_s=wait, retry_after_s=RETRY,
                               clock=clock or Clock())


async def until(pred, limit=2000):
    """Yield to the loop until pred() holds (a bounded number of checkpoints, never wall-clock)."""
    for _ in range(limit):
        if pred():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


def waiter(c, out, tag, deadline=FAR):
    async def run():
        p = await c.acquire(deadline=deadline)
        out.append(tag)
        return p
    return asyncio.create_task(run())


async def test_fast_path_and_idempotent_release():
    c = ctrl(n=2)
    p1 = await c.acquire(deadline=FAR)
    p2 = await c.acquire(deadline=FAR)
    assert (c.in_flight, c.queued, p1.queued_s) == (2, 0, 0.0)
    p1.release()
    p1.release()
    assert c.in_flight == 1
    p2.release()
    assert c.in_flight == 0


async def test_200_random_tasks_never_exceed_the_limit():
    c = ctrl(n=4, q=200)
    rng = random.Random(7)
    live, peak, done = [0], [0], []

    async def job(i):
        p = await c.acquire(deadline=FAR)
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        assert c.in_flight <= 4
        for _ in range(rng.randrange(5)):
            await asyncio.sleep(0)
        live[0] -= 1
        p.release()
        done.append(i)

    await asyncio.gather(*(job(i) for i in range(200)))
    assert peak[0] == 4 and len(done) == 200 and (c.in_flight, c.queued) == (0, 0)


async def test_queue_full_is_a_429_with_its_retry_after():
    c = ctrl(n=1, q=1)
    p = await c.acquire(deadline=FAR)
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR)
    assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("queue_full", 429, 2)
    p.release()
    (await t).release()
    assert c.in_flight == 0


async def test_queue_timeout_from_the_deadline_with_an_injected_clock():
    clock = Clock(500.0)
    c = ctrl(n=1, clock=clock)
    p = await c.acquire(deadline=FAR)
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=500.0)        # no time left before the request deadline: never queued
    assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("queue_timeout", 429, 3)
    assert c.queued == 0
    p.release()


async def test_queue_timeout_after_max_queue_wait():
    c = ctrl(n=1, wait=0.01)
    p = await c.acquire(deadline=FAR)
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR)
    assert exc.value.reason == "queue_timeout" and (c.queued, c.in_flight) == (0, 1)
    p.release()
    assert c.in_flight == 0


async def test_queued_seconds_come_from_the_injected_clock():
    clock = Clock(10.0)
    c = ctrl(n=1, clock=clock)
    p = await c.acquire(deadline=FAR)
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    clock.t = 12.5
    p.release()
    p2 = await t
    assert p2.queued_s == 2.5
    p2.release()


async def test_fifo_order_and_no_overtaking_after_a_release():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    order, tasks = [], []
    for i in range(5):
        tasks.append(waiter(c, order, i))
        await until(lambda i=i: c.queued == i + 1)
    p.release()                                 # the slot is handed to waiter 0 ...
    late = waiter(c, order, "late")             # ... so a newcomer must queue behind the others
    await until(lambda: c.queued == 5)
    for t in tasks:
        (await t).release()
    (await late).release()
    assert order == [0, 1, 2, 3, 4, "late"] and c.in_flight == 0


async def test_cancel_while_queued_does_not_leak():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert (c.queued, c.in_flight) == (0, 1)
    p.release()
    assert c.in_flight == 0


async def test_cancel_during_handoff_passes_the_slot_on():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    order = []
    t1 = waiter(c, order, 1)
    await until(lambda: c.queued == 1)
    t2 = waiter(c, order, 2)
    await until(lambda: c.queued == 2)
    p.release()          # t1's future now holds the slot ...
    t1.cancel()          # ... but t1 is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await t1
    p2 = await t2
    assert order == [2] and c.in_flight == 1
    p2.release()
    assert (c.in_flight, c.queued) == (0, 0)


async def test_cancel_during_handoff_with_no_other_waiter_frees_the_slot():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    p.release()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert c.in_flight == 0
    (await c.acquire(deadline=FAR)).release()


async def test_reject_waiters_rejects_only_the_queue():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    tasks = [waiter(c, [], i) for i in range(3)]
    await until(lambda: c.queued == 3)
    c.reject_waiters("backend_unavailable")
    for t in tasks:
        with pytest.raises(Rejection) as exc:
            await t
        assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("backend_unavailable", 503, 5)
    assert (c.queued, c.in_flight) == (0, 1)
    p.release()
    assert c.in_flight == 0


async def test_drain_rejects_waiters_and_newcomers():
    c = ctrl(n=2)
    p1 = await c.acquire(deadline=FAR)
    p2 = await c.acquire(deadline=FAR)
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    c.set_draining()
    with pytest.raises(Rejection) as exc:
        await t
    assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("draining", 503, 7)
    p1.release()
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR)           # a free slot does not matter while draining
    assert exc.value.reason == "draining" and c.draining
    assert await c.wait_idle(0.01) is False     # p2 still running
    p2.release()
    assert await c.wait_idle(1.0) is True and c.in_flight == 0


async def test_wait_idle_wakes_when_the_last_permit_is_released():
    c = ctrl(n=1)
    p = await c.acquire(deadline=FAR)
    idle = asyncio.create_task(c.wait_idle(5.0))
    await asyncio.sleep(0)
    assert not idle.done()
    p.release()
    assert await idle is True


@pytest.mark.parametrize("kw", [{"max_concurrency": -1}, {"max_concurrency": 0, "max_queue": 1}, {"max_queue": -1},
                                {"max_queue_wait_s": 0}])
def test_bad_limits_are_refused(kw):
    args = {"max_concurrency": 1, "max_queue": 0, "max_queue_wait_s": 1.0, "retry_after_s": RETRY, **kw}
    with pytest.raises(ValueError):
        AdmissionController(**args)


async def test_a_closed_controller_admits_and_queues_nothing():
    """max_concurrency 0 (with max_queue 0) is a verified refuse class (030 8.7): every acquire is a 503
    class_not_serviceable at once, without a Retry-After (retrying cannot help), and nothing ever waits."""
    c = ctrl(n=0, q=0)
    for _ in range(3):
        with pytest.raises(Rejection) as exc:
            await c.acquire(deadline=FAR)
        assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("class_not_serviceable", 503, None)
    assert (c.in_flight, c.queued) == (0, 0)
    c.set_draining()
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR)
    assert exc.value.reason == "class_not_serviceable"
    assert await c.wait_idle(0.01) is True


async def test_zero_queue_rejects_immediately():
    c = ctrl(n=1, q=0)
    p = await c.acquire(deadline=FAR)
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR)
    assert exc.value.reason == "queue_full"
    p.release()


# ------------------------------------------------------------------ 044: the release time --

async def test_release_records_its_time_before_the_slot_moves_on():
    """044 item 1: t_released_s is read in Permit.release before the next waiter gets the slot, so the next request's
    [t_sent, t_released) interval never starts before this one's ended."""
    clock = Clock(5.0)
    c = ctrl(n=1, clock=clock)
    seen = []
    p = await c.acquire(deadline=FAR, on_release=lambda t: seen.append((t, c.queued)))
    t = waiter(c, [], "w")
    await until(lambda: c.queued == 1)
    clock.t = 7.5
    p.release()
    p.release()
    assert seen == [(7.5, 1)] and p.released_s == 7.5   # once; the waiter still queued when the time was taken
    p2 = await t
    assert p2.released_s is None and p2.prefill_wait_s is None
    p2.release()
    assert p2.released_s == 7.5 and c.in_flight == 0


# ------------------------------------------------------------------ 044: the token bucket --

def test_the_token_bucket_starts_full_refills_at_its_rate_and_never_beyond_burst():
    b = TokenBucket(10, 1000.0, now=0.0)
    assert b.tokens(0.0) == 10 and b.ready(10, 0.0) and not b.ready(11, 0.0)
    b.take(8, 0.0)
    assert b.tokens(0.0) == 2 and b.tokens(0.003) == 5 and b.tokens(1.0) == 10
    assert b.wait_s(4, 0.0) == pytest.approx(0.002) and b.wait_s(2, 0.0) == 0.0
    b.take(5, 0.003)
    assert b.tokens(0.003) == 0 and b.wait_s(10, 0.003) == pytest.approx(0.01)
    for burst, rate in ((0, 1.0), (1, 0.0)):
        with pytest.raises(ValueError):
            TokenBucket(burst, rate, now=0.0)


# ------------------------------------------------------------------ 044: weighted-v1 --

UNITS = {"s": 1, "l": 4}   # weighted-v1's shape: a short class of 1 unit, a long one of 4, budget 4 = caps 4 / 1


def policy(*, caps=None, budget=4, q=8, wait=60.0, bucket=None, clock=None):
    return PolicyController(caps=caps or {"s": 4, "l": 1}, budget=budget, max_queue=q, max_queue_wait_s=wait,
                            retry_after_s=RETRY, bucket=bucket, clock=clock or Clock())


def ask(cls, tokens=1, units=None):
    return Demand(cls, UNITS[cls] if units is None else units, tokens)


def policy_waiter(c, out, tag, demand, deadline=FAR):
    async def run():
        p = await c.acquire(deadline=deadline, demand=demand)
        out.append(tag)
        return p
    return asyncio.create_task(run())


async def test_weighted_v1_admits_within_the_budget_and_each_class_cap():
    c = policy()
    shorts = [await c.acquire(deadline=FAR, demand=ask("s")) for _ in range(4)]
    assert (c.in_flight, c.units_in_flight, c.class_in_flight) == (4, 4, {"s": 4, "l": 0})
    order = []
    t = policy_waiter(c, order, "s5", ask("s"))
    await until(lambda: c.queued == 1)
    shorts[0].release()
    p = await t
    assert order == ["s5"] and p.demand == ask("s") and p.prefill_wait_s is None
    for x in shorts[1:] + [p]:
        x.release()
    long = await c.acquire(deadline=FAR, demand=ask("l"))
    assert (c.units_in_flight, c.class_in_flight) == (4, {"s": 0, "l": 1})
    t = policy_waiter(c, order, "s6", ask("s"))                   # the budget is full: a short one waits
    await until(lambda: c.queued == 1)
    long.release()
    (await t).release()
    assert (c.in_flight, c.units_in_flight, c.queued) == (0, 0, 0)


async def test_a_class_cap_binds_below_the_budget():
    c = policy(caps={"s": 4, "l": 1}, budget=8)
    long = await c.acquire(deadline=FAR, demand=ask("l"))
    order = []
    t = policy_waiter(c, order, "l2", ask("l"))                   # 4 + 4 <= 8 units, but the long cap is 1
    await until(lambda: c.queued == 1)
    behind = policy_waiter(c, order, "s", ask("s"))              # fits, but strict FIFO: behind the long head
    await until(lambda: c.queued == 2)
    assert c.units_in_flight == 4 and order == []
    long.release()
    for x in (await t, await behind):
        x.release()
    assert order == ["l2", "s"] and c.in_flight == 0


async def test_the_head_blocks_everyone_behind_it():
    """044 / 043 7.3 A: strict FIFO with head-of-line blocking: a short request that would fit never overtakes a long
    head that does not."""
    c = policy()
    shorts = [await c.acquire(deadline=FAR, demand=ask("s")) for _ in range(3)]   # 3 of 4 units
    order = []
    head = policy_waiter(c, order, "l", ask("l"))
    await until(lambda: c.queued == 1)
    behind = policy_waiter(c, order, "s", ask("s"))
    await until(lambda: c.queued == 2)
    shorts[0].release()
    shorts[1].release()                                          # 1 unit in flight: still no room for 4
    await asyncio.sleep(0)
    assert order == [] and c.units_in_flight == 1
    shorts[2].release()
    long = await head
    assert order == ["l"] and c.queued == 1                      # the short one now waits for the long one
    long.release()
    (await behind).release()
    assert order == ["l", "s"] and (c.in_flight, c.units_in_flight) == (0, 0)


async def test_a_blocked_head_that_leaves_unblocks_the_next():
    c = policy()   # its clock stays at 1000.0: a deadline of 1000.05 times a waiter out after 0.05 s
    shorts = [await c.acquire(deadline=FAR, demand=ask("s")) for _ in range(3)]
    order = []
    head = policy_waiter(c, order, "l", ask("l"))
    await until(lambda: c.queued == 1)
    behind = policy_waiter(c, order, "s", ask("s"))
    await until(lambda: c.queued == 2)
    head.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head
    p = await behind                                             # the free unit goes to it at once
    assert order == ["s"] and c.units_in_flight == 4
    head = policy_waiter(c, order, "l2", ask("l"), deadline=1000.05)   # a long head that times out
    await until(lambda: c.queued == 1)
    behind = policy_waiter(c, order, "s2", ask("s"))
    await until(lambda: c.queued == 2)
    p.release()
    with pytest.raises(Rejection) as exc:
        await head
    assert exc.value.reason == "queue_timeout"
    (await behind).release()
    for x in shorts:
        x.release()
    assert order == ["s", "s2"] and (c.in_flight, c.units_in_flight, c.queued) == (0, 0, 0)


async def test_a_cancel_during_the_handoff_passes_the_units_on():
    c = policy()
    long = await c.acquire(deadline=FAR, demand=ask("l"))
    order = []
    t1 = policy_waiter(c, order, 1, ask("l"))
    await until(lambda: c.queued == 1)
    t2 = policy_waiter(c, order, 2, ask("s"))
    await until(lambda: c.queued == 2)
    long.release()          # t1's future now holds 4 units ...
    t1.cancel()             # ... but t1 is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await t1
    p2 = await t2
    assert order == [2] and (c.in_flight, c.units_in_flight) == (1, 1)
    p2.release()
    assert (c.in_flight, c.units_in_flight, c.queued) == (0, 0, 0)


async def test_queue_full_timeout_drain_and_backend_down_as_the_fifo_controller():
    c = policy(q=1)
    long = await c.acquire(deadline=FAR, demand=ask("l"))
    t = policy_waiter(c, [], "w", ask("s"))
    await until(lambda: c.queued == 1)
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR, demand=ask("s"))
    assert (exc.value.reason, exc.value.status, exc.value.retry_after_s) == ("queue_full", 429, 2)
    c.reject_waiters("backend_unavailable")
    with pytest.raises(Rejection) as exc:
        await t
    assert (exc.value.reason, exc.value.status) == ("backend_unavailable", 503)
    clock = Clock(500.0)
    c2 = policy(clock=clock)
    p = await c2.acquire(deadline=FAR, demand=ask("l"))
    with pytest.raises(Rejection) as exc:
        await c2.acquire(deadline=500.0, demand=ask("s"))       # no time left before the deadline: never queued
    assert exc.value.reason == "queue_timeout" and c2.queued == 0
    c.set_draining()
    with pytest.raises(Rejection) as exc:
        await c.acquire(deadline=FAR, demand=ask("s"))
    assert exc.value.reason == "draining" and await c.wait_idle(0.01) is False
    long.release()
    p.release()
    assert await c.wait_idle(1.0) is True and (c.in_flight, c2.in_flight) == (0, 0)


async def test_200_random_mixed_tasks_never_exceed_the_budget_or_a_cap():
    c = policy(caps={"s": 8, "l": 2}, budget=8, q=300)
    rng = random.Random(11)
    peak = {"units": 0, "s": 0, "l": 0}
    done = []

    async def job(i):
        cls = "l" if rng.random() < 0.3 else "s"
        p = await c.acquire(deadline=FAR, demand=ask(cls))
        peak["units"] = max(peak["units"], c.units_in_flight)
        for k, v in c.class_in_flight.items():
            peak[k] = max(peak[k], v)
        assert c.units_in_flight <= 8 and c.class_in_flight["l"] <= 2
        for _ in range(rng.randrange(5)):
            await asyncio.sleep(0)
        p.release()
        done.append(i)

    await asyncio.gather(*(job(i) for i in range(200)))
    assert len(done) == 200 and peak["units"] == 8 and peak["l"] == 2
    assert (c.in_flight, c.units_in_flight, c.queued) == (0, 0, 0)


@pytest.mark.parametrize("kw", [{"budget": 0}, {"q": -1}, {"wait": 0}, {"caps": {"s": 0, "l": 1}}])
def test_bad_policy_limits_are_refused(kw):
    with pytest.raises(ValueError):
        policy(**kw)


async def test_a_demand_outside_the_policy_is_a_programming_error():
    c = policy(bucket=TokenBucket(4, 1000.0, now=1000.0))
    for demand in (None, Demand("x", 1, 1), Demand("s", 1, 5)):   # unknown class; a prompt above the burst
        with pytest.raises(ValueError):
            await c.acquire(deadline=FAR, demand=demand)


# ------------------------------------------------------------------ 044: prefill pacing --

async def test_pacing_lets_a_burst_through_then_paces_the_head_by_its_tokens():
    """043 7.3 B: the head needs a free slot and its prompt's tokens; prefill_wait_s is the time it waited with a free
    slot for the tokens only, queued_s the rest."""
    c = policy(caps={"s": 4}, budget=4, bucket=TokenBucket(10, 1000.0, now=time.monotonic()), clock=time.monotonic)
    a = await c.acquire(deadline=FAR, demand=Demand("s", 1, 4))
    b = await c.acquire(deadline=FAR, demand=Demand("s", 1, 4))
    assert (a.prefill_wait_s, b.prefill_wait_s, a.queued_s) == (0.0, 0.0, 0.0)
    order = []
    t = policy_waiter(c, order, "c", Demand("s", 1, 4))          # 2 tokens left: waits ~2 ms with a free slot
    await until(lambda: c.queued == 1)
    small = policy_waiter(c, order, "d", Demand("s", 1, 1))     # 1 token would fit, but never overtakes the head
    await until(lambda: c.queued == 2)
    p = await t
    assert order[0] == "c" and p.prefill_wait_s > 0 and p.queued_s < p.prefill_wait_s
    q = await small
    assert order == ["c", "d"]
    for x in (a, b, p, q):
        x.release()
    assert (c.in_flight, c.queued) == (0, 0)


async def test_prefill_wait_counts_only_the_time_at_the_head_with_a_free_slot():
    clock = Clock(0.0)
    c = policy(caps={"s": 1}, budget=1, bucket=TokenBucket(4, 1000.0, now=0.0), clock=clock)
    first = await c.acquire(deadline=FAR, demand=Demand("s", 1, 4))   # the bucket is empty now
    t = policy_waiter(c, [], "w", Demand("s", 1, 4))
    await until(lambda: c.queued == 1)
    clock.t = 0.002                                                   # 2 tokens back; the slot is still taken
    first.release()                                                   # free slot, 2 tokens short: pacing starts
    clock.t = 0.004
    p = await t
    assert p.prefill_wait_s == pytest.approx(0.002) and p.queued_s == pytest.approx(0.002)
    p.release()
    clock.t = 1.0                                                     # refilled: a free slot and the tokens
    q = await c.acquire(deadline=FAR, demand=Demand("s", 1, 4))
    assert (q.prefill_wait_s, q.queued_s) == (0.0, 0.0)
    q.release()


async def test_a_paced_head_that_leaves_lets_the_next_through():
    c = policy(caps={"s": 4}, budget=4, wait=60.0, bucket=TokenBucket(4, 1.0, now=time.monotonic()),
               clock=time.monotonic)
    a = await c.acquire(deadline=FAR, demand=Demand("s", 1, 3))     # 1 token left, refilled at 1 token / s
    order = []
    head = policy_waiter(c, order, "head", Demand("s", 1, 4))       # would wait ~3 s
    await until(lambda: c.queued == 1)
    behind = policy_waiter(c, order, "small", Demand("s", 1, 1))
    await until(lambda: c.queued == 2)
    head.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head
    p = await asyncio.wait_for(behind, 1.0)                          # the 1 token left covers it at once
    assert order == ["small"] and p.prefill_wait_s == 0.0
    for x in (a, p):
        x.release()
    assert (c.in_flight, c.queued) == (0, 0)
