"""Tests for capserve/metrics.py: an own registry, no _created series, the 029 metric names, and one place that
turns a finished request record into counters and histograms."""

import prometheus_client
import pytest
from prometheus_client.parser import text_string_to_metric_families

from capserve.guard import RequestRecord
from capserve.metrics import Metrics
from capserve.profile import CapacityProfile

PROFILE = CapacityProfile.model_validate({
    "schema_version": 1, "profile_id": "provisional-x", "status": "provisional", "provisional_reason": "test",
    "slo": None, "server": {"vllm_version": "0.30.0", "model_repo": "m", "model_revision": "r",
                            "fingerprint_sha256": None, "num_gpu_blocks": None, "block_size": None},
    "classes": [{"name": "2k", "max_prompt_tokens": 1920, "max_output_tokens": 128, "max_concurrency": 4,
                 "max_queue": 6, "source": "t"}],
    "provenance": None, "content_sha256": "x"})
NAMES = {
    "capserve_requests_total", "capserve_rejections_total", "capserve_client_errors_total",
    "capserve_upstream_errors_total", "capserve_client_disconnects_total", "capserve_queue_wait_seconds",
    "capserve_ttft_seconds", "capserve_upstream_ttft_seconds", "capserve_e2e_seconds", "capserve_in_flight",
    "capserve_queued", "capserve_pending", "capserve_concurrency_limit", "capserve_queue_limit",
    "capserve_backend_ready", "capserve_profile_verified", "capserve_profile_info"}


def metrics(**live):
    values = {"in_flight": 3, "queued": 2, "pending": 9, "backend_ready": True, **live}
    return Metrics(PROFILE, PROFILE.classes[0], **{k: (lambda v=v: v) for k, v in values.items()})


def samples(m):
    text = m.exposition().decode()
    return text, {(s.name, tuple(sorted(s.labels.items()))): s.value
                  for fam in text_string_to_metric_families(text) for s in fam.samples}


def value(m, name, **labels):
    return samples(m)[1].get((name, tuple(sorted(labels.items()))))


def test_exposition_names_and_no_created_series():
    text, _ = samples(metrics())
    assert {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")} == NAMES
    assert "_created" not in text and "capserve_profile_info_info" not in text


def test_own_registry_never_touches_the_global_one():
    a, b = metrics(), metrics()                 # two instances do not collide
    assert a.registry is not b.registry and a.registry is not prometheus_client.REGISTRY
    assert "capserve_in_flight" not in prometheus_client.generate_latest().decode()


def test_gauges_read_live_state_and_limits():
    m = metrics()
    assert (value(m, "capserve_in_flight"), value(m, "capserve_queued"), value(m, "capserve_pending")) == (3, 2, 9)
    assert (value(m, "capserve_concurrency_limit"), value(m, "capserve_queue_limit")) == (4, 6)
    assert (value(m, "capserve_backend_ready"), value(m, "capserve_profile_verified")) == (1, 0)
    assert value(metrics(backend_ready=False), "capserve_backend_ready") == 0


def test_profile_info_labels():
    assert value(metrics(), "capserve_profile_info", profile_id="provisional-x", status="provisional", slo_id="",
                 source_session="", analysis_sha256="", active_class="2k") == 1


def rec(**kw):
    return RequestRecord(request_id="r", method="POST", path="/v1/completions", arrival=0.0,
                         **{"route": "completions", **kw})


def test_finish_counts_an_ok_stream_with_latencies():
    m = metrics()
    m.finish(rec(stream=True, outcome="ok", queue_wait_s=0.3, ttft_s=0.5, upstream_ttft_s=0.2, duration_s=1.5))
    assert value(m, "capserve_requests_total", route="completions", stream="true", outcome="ok") == 1
    assert value(m, "capserve_queue_wait_seconds_count") == 1
    assert value(m, "capserve_ttft_seconds_sum", route="completions") == 0.5
    assert value(m, "capserve_upstream_ttft_seconds_sum", route="completions") == 0.2
    assert value(m, "capserve_e2e_seconds_sum", route="completions", stream="true") == 1.5
    assert value(m, "capserve_e2e_seconds_bucket", route="completions", stream="true", le="300.0") == 1


@pytest.mark.parametrize("kw, counter, labels", [
    ({"outcome": "rejected", "reason": "queue_full"}, "capserve_rejections_total", {"reason": "queue_full"}),
    ({"outcome": "client_error", "error_code": "unsupported_parameter"}, "capserve_client_errors_total",
     {"code": "unsupported_parameter"}),
    ({"outcome": "upstream_error", "upstream_error": "connect"}, "capserve_upstream_errors_total",
     {"kind": "connect"}),
    ({"outcome": "cancelled", "disconnect_phase": "queued"}, "capserve_client_disconnects_total",
     {"phase": "queued"}),
])
def test_finish_counts_failures_by_their_label(kw, counter, labels):
    m = metrics()
    m.finish(rec(stream=False, duration_s=1.0, **kw))
    assert value(m, counter, **labels) == 1
    assert value(m, "capserve_requests_total", route="completions", stream="false", outcome=kw["outcome"]) == 1
    assert value(m, "capserve_e2e_seconds_count", route="completions", stream="false") is None


def test_unknown_stream_and_non_completion_routes():
    m = metrics()
    m.finish(rec(stream=None, outcome="rejected", reason="queue_full"))
    assert value(m, "capserve_requests_total", route="completions", stream="unknown", outcome="rejected") == 1
    m.finish(RequestRecord(request_id="r", method="GET", path="/v1/models", arrival=0.0, outcome="rejected",
                           reason="queue_full"))
    assert value(m, "capserve_rejections_total", reason="queue_full") == 2
    assert value(m, "capserve_requests_total", route="completions", stream="unknown", outcome="rejected") == 1
