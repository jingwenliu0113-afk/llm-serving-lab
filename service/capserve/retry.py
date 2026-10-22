"""Retry-After for the 429s from the observed service time (decision 038 §3).

- Only queue_full (the guard's pending cap or a full admission queue) and queue_timeout adapt; backend_unavailable and
  draining keep their fixed values (settings [retry_after_s]): how long a restart or a shutdown takes does not depend
  on the load.
- Service time of a finished request = arrival -> response end minus its admission wait, counted only for requests
  that were admitted and ended ok (a cancelled, failed or rejected request says nothing about how long a slot is
  held). mean_service_s is an EWMA with ALPHA, the first observation taken as is.
- Estimate = (queued + 1) x mean_service_s / max_concurrency: the time until the queue ahead of a newcomer, plus the
  newcomer, clears, if slots free at the observed rate. Clamped to [min_s, max_s]. Before any completion, or in
  mode "fixed", the reason's fixed value.
- A hint only: admission never depends on it. openai_api.retry_headers rounds Retry-After up to whole seconds and
  keeps the estimate in retry-after-ms.
"""

from __future__ import annotations

from typing import Callable

from capserve.config import RetryAfter

ADAPTIVE_REASONS = ("queue_full", "queue_timeout")
ALPHA = 0.2


class RetryAfterPolicy:
    def __init__(self, cfg: RetryAfter, *, max_concurrency: int, queued: Callable[[], int]):
        self.cfg = cfg
        self.max_concurrency = max_concurrency
        self._queued = queued
        self.mean_service_s: float | None = None

    def observe(self, rec) -> None:
        """A finished request (the guard's RequestRecord)."""
        if rec.outcome != "ok" or rec.queue_wait_s is None or rec.duration_s is None:
            return
        service = max(rec.duration_s - rec.queue_wait_s, 0.0)
        mean = self.mean_service_s
        self.mean_service_s = service if mean is None else mean + ALPHA * (service - mean)

    def seconds(self, reason: str) -> float:
        fixed = getattr(self.cfg, reason)
        if (self.cfg.mode == "fixed" or reason not in ADAPTIVE_REASONS or self.mean_service_s is None
                or self.max_concurrency <= 0):
            return fixed
        estimate = (self._queued() + 1) * self.mean_service_s / self.max_concurrency
        return min(max(estimate, self.cfg.min_s), self.cfg.max_s)
