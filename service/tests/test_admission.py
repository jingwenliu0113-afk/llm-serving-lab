"""Tests for capserve/admission.py: FIFO admission with a bounded queue that never exceeds max_concurrency and never
loses a permit on cancellation, timeout or rejection (decision 029 "准入控制與超載保護")."""

import asyncio
import random

import pytest

from capserve.admission import AdmissionController, Rejection

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
