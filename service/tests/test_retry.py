"""Tests for capserve/retry.py: Retry-After for 429s from the observed service time (decision 038 §3)."""

import pytest

from capserve.config import RetryAfter
from capserve.guard import RequestRecord
from capserve.retry import ADAPTIVE_REASONS, RetryAfterPolicy


def rec(outcome="ok", duration_s=10.0, queue_wait_s=0.0):
    return RequestRecord(request_id="r", method="POST", path="/v1/completions", arrival=0.0, route="completions",
                         outcome=outcome, duration_s=duration_s, queue_wait_s=queue_wait_s)


def policy(max_concurrency=4, **kw):
    state = {"queued": 0}
    return RetryAfterPolicy(RetryAfter(**kw), max_concurrency=max_concurrency, queued=lambda: state["queued"]), state


def test_fixed_mode_ignores_the_load():
    p, state = policy(mode="fixed")
    p.observe(rec(duration_s=100.0))
    state["queued"] = 50
    assert [p.seconds(r) for r in ("queue_full", "queue_timeout", "backend_unavailable", "draining")] == [2, 2, 5, 5]


def test_adaptive_falls_back_to_the_fixed_value_before_any_completion():
    p, state = policy(queue_full=7)
    state["queued"] = 3
    assert (p.seconds("queue_full"), p.seconds("queue_timeout")) == (7, 2) and p.mean_service_s is None


def test_estimate_is_queue_position_times_mean_service_over_concurrency():
    p, state = policy(max_concurrency=4)
    p.observe(rec(duration_s=10.0))
    p.observe(rec(duration_s=25.0, queue_wait_s=5.0))      # service 20 s: EWMA 10 + 0.2 x (20 - 10) = 12
    assert p.mean_service_s == pytest.approx(12.0)
    state["queued"] = 3
    assert p.seconds("queue_full") == pytest.approx((3 + 1) * 12.0 / 4)
    assert p.seconds("queue_timeout") == pytest.approx(12.0)


def test_estimate_is_clamped_to_min_and_max():
    p, state = policy(max_concurrency=32, min_s=1.0, max_s=60.0)
    p.observe(rec(duration_s=0.5))
    assert p.seconds("queue_full") == 1.0                  # (0 + 1) x 0.5 / 32 below min_s
    p, state = policy(max_concurrency=1, min_s=1.0, max_s=60.0)
    p.observe(rec(duration_s=30.0))
    state["queued"] = 9
    assert p.seconds("queue_full") == 60.0                 # 10 x 30 above max_s


def test_backend_reasons_stay_fixed_in_adaptive_mode():
    p, state = policy(backend_unavailable=9, draining=11)
    p.observe(rec(duration_s=40.0))
    state["queued"] = 4
    assert (p.seconds("backend_unavailable"), p.seconds("draining")) == (9, 11)
    assert set(ADAPTIVE_REASONS) == {"queue_full", "queue_timeout"}


@pytest.mark.parametrize("record", [rec(outcome="client_error"), rec(outcome="cancelled"), rec(outcome="deadline"),
                                    rec(outcome="upstream_error"), rec(outcome="rejected", queue_wait_s=None),
                                    rec(queue_wait_s=None), rec(duration_s=None)])
def test_only_admitted_successful_requests_count(record):
    p, _ = policy()
    p.observe(record)
    assert p.mean_service_s is None


def test_service_time_excludes_the_queue_wait():
    p, _ = policy()
    p.observe(rec(duration_s=3.0, queue_wait_s=2.0))
    assert p.mean_service_s == pytest.approx(1.0)


@pytest.mark.parametrize("kw", [{"mode": "auto"}, {"min_s": 0}, {"max_s": -1}, {"min_s": 5.0, "max_s": 4.0}])
def test_bad_settings_are_refused(kw):
    with pytest.raises(ValueError):
        RetryAfter(**kw)


def test_fixed_values_are_the_four_reasons():
    assert RetryAfter(queue_full=3).fixed_values() == {"queue_full": 3, "queue_timeout": 2, "backend_unavailable": 5,
                                                       "draining": 5}
