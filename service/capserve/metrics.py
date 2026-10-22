"""Operational Prometheus metrics (decision 029 "/metrics"). These numbers are for operating the gateway only and
are never research evidence: research TTFT comes from the research client, never from here.

- An own CollectorRegistry (never prometheus_client.REGISTRY) and disable_created_metrics(), so the exposition has
  no _created series.
- Gauges read live state through callables (in_flight, queued, pending, backend_ready); limits and the profile are
  fixed for the process. Info("capserve_profile") is exposed as capserve_profile_info.
- finish(record) is the only place a finished request becomes counters / histograms (called by the guard):
  requests_total{route, stream, outcome} for the completion routes (stream "unknown" when the request ended before
  its body was parsed), rejections / client errors / upstream errors / disconnects by their label, queue wait for
  admitted requests, ttft (arrival -> first content, queue included) and upstream_ttft (upstream send -> first
  content, the research TTFT definition) for streams, e2e for successful requests.
"""

from __future__ import annotations

from typing import Callable

import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from capserve.profile import CapacityProfile, WorkloadClass

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
E2E_BUCKETS = BUCKETS + (300,)


class Metrics:
    def __init__(self, profile: CapacityProfile, cls: WorkloadClass, *, in_flight: Callable[[], float],
                 queued: Callable[[], float], pending: Callable[[], float], backend_ready: Callable[[], bool]):
        prometheus_client.disable_created_metrics()
        self.registry = r = CollectorRegistry()
        self.requests = Counter("capserve_requests", "Finished requests on the completion routes",
                                ["route", "stream", "outcome"], registry=r)
        self.rejections = Counter("capserve_rejections", "429 / 503 rejections", ["reason"], registry=r)
        self.client_errors = Counter("capserve_client_errors", "4xx client errors", ["code"], registry=r)
        self.upstream_errors = Counter("capserve_upstream_errors", "vLLM failures", ["kind"], registry=r)
        self.client_disconnects = Counter("capserve_client_disconnects", "Clients gone before the end", ["phase"],
                                          registry=r)
        self.key_requests = Counter("capserve_key_requests", "Finished requests per API key name (038 §5)",
                                    ["key", "outcome"], registry=r)
        self.queue_wait = Histogram("capserve_queue_wait_seconds", "Admission wait", buckets=BUCKETS, registry=r)
        self.ttft = Histogram("capserve_ttft_seconds", "Arrival to first content (queue included)", ["route"],
                              buckets=BUCKETS, registry=r)
        self.upstream_ttft = Histogram("capserve_upstream_ttft_seconds", "Upstream send to first content", ["route"],
                                       buckets=BUCKETS, registry=r)
        self.e2e = Histogram("capserve_e2e_seconds", "Arrival to response end (successful requests)",
                             ["route", "stream"], buckets=E2E_BUCKETS, registry=r)
        for name, doc, fn in (("capserve_in_flight", "Admitted requests", in_flight),
                              ("capserve_queued", "Requests waiting for admission", queued),
                              ("capserve_pending", "Requests inside the gateway", pending),
                              ("capserve_backend_ready", "BackendMonitor state", lambda: float(bool(backend_ready())))):
            Gauge(name, doc, registry=r).set_function(fn)
        Gauge("capserve_concurrency_limit", "Active class max_concurrency", registry=r).set(cls.max_concurrency)
        Gauge("capserve_queue_limit", "Active class max_queue", registry=r).set(cls.max_queue)
        Gauge("capserve_profile_verified", "1 = verified profile, 0 = provisional", registry=r).set(
            1 if profile.verified else 0)
        prov = profile.provenance
        Info("capserve_profile", "Capacity profile in use", registry=r).info({
            "profile_id": profile.profile_id, "status": profile.status,
            "slo_id": profile.slo.id if profile.slo else "",
            "source_session": prov.source_session if prov else "",
            "analysis_sha256": prov.analysis_sha256 if prov else "",
            "active_class": cls.name})

    def finish(self, rec) -> None:
        if rec.reason:
            self.rejections.labels(rec.reason).inc()
        if rec.error_code:
            self.client_errors.labels(rec.error_code).inc()
        if rec.upstream_error:
            self.upstream_errors.labels(rec.upstream_error).inc()
        if rec.disconnect_phase:
            self.client_disconnects.labels(rec.disconnect_phase).inc()
        if rec.key:
            self.key_requests.labels(rec.key, rec.outcome or "cancelled").inc()
        if rec.route is None:
            return
        stream = "unknown" if rec.stream is None else ("true" if rec.stream else "false")
        self.requests.labels(rec.route, stream, rec.outcome or "cancelled").inc()
        if rec.queue_wait_s is not None:
            self.queue_wait.observe(rec.queue_wait_s)
        if rec.ttft_s is not None:
            self.ttft.labels(rec.route).observe(rec.ttft_s)
        if rec.upstream_ttft_s is not None:
            self.upstream_ttft.labels(rec.route).observe(rec.upstream_ttft_s)
        if rec.outcome == "ok" and rec.duration_s is not None:
            self.e2e.labels(rec.route, stream).observe(rec.duration_s)

    def exposition(self) -> bytes:
        return prometheus_client.generate_latest(self.registry)
