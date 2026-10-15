"""Tests for capserve/profile.py: the capacity profile schema, canonical hash, loader rules and server fingerprint
(decision 029 "Capacity profile"; 027 §4)."""

import copy
import json
from pathlib import Path

import pytest

import core_b_plan as bp
from capserve import profile as cp

SERVICE = Path(__file__).resolve().parent.parent
HEX = "ab" * 32


@pytest.fixture(scope="module")
def pins():
    return bp.load_pins()


def _server(pins, **over):
    return {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
            "model_revision": pins["core_b"]["revision"], "fingerprint_sha256": HEX, "num_gpu_blocks": 248,
            "block_size": 528, **over}


def _class(**over):
    return {"name": "2k", "max_prompt_tokens": 1920, "max_output_tokens": 128, "max_concurrency": 4, "max_queue": 4,
            "source": "stage_c:L=2048:uniform", **over}


def verified_doc(pins):
    doc = {"schema_version": 1, "profile_id": "verified-formal-20261101-c1", "status": "verified",
           "provisional_reason": None,
           "slo": {"id": "S1", "terms": {"P95 TTFT <= 2000 ms": 2000}},
           "server": _server(pins),
           "classes": [_class()],
           "provenance": {"source_session": "formal-20261101-c1",
                          "analysis_path": "results/capacity/formal-20261101-c1/analysis.json",
                          "analysis_sha256": HEX, "prereg_section": "§8", "freeze_commit": "146d260",
                          "generator_commit": "24dc89e"}}
    return rehash(doc)


def provisional_doc(pins):
    doc = {"schema_version": 1, "profile_id": "provisional-test", "status": "provisional",
           "provisional_reason": "unit test; not a capacity", "slo": None,
           "server": _server(pins, fingerprint_sha256=None, num_gpu_blocks=None, block_size=None),
           "classes": [_class()], "provenance": None}
    return rehash(doc)


def rehash(doc):
    doc = {k: v for k, v in doc.items() if k != "content_sha256"}
    doc["content_sha256"] = cp.content_sha256(doc)
    return doc


def write(tmp_path, doc, name="p.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def refusal(tmp_path, doc, *, allow_provisional=True, pins=None):
    with pytest.raises(cp.ProfileError) as exc:
        cp.load_profile(write(tmp_path, doc), allow_provisional=allow_provisional, pins=pins)
    return str(exc.value)


# ------------------------------------------------------------------ canonical hash --

def test_canonical_json_is_sorted_compact_utf8():
    assert cp.canonical_json({"b": 1, "a": ["é", None]}) == '{"a":["é",null],"b":1}'
    assert cp.canonical_sha256({"b": 1, "a": 2}) == cp.canonical_sha256({"a": 2, "b": 1})


def test_content_sha256_ignores_only_its_own_field():
    doc = {"a": 1, "content_sha256": "x"}
    assert cp.content_sha256(doc) == cp.canonical_sha256({"a": 1})
    assert cp.content_sha256({"a": 2}) != cp.content_sha256(doc)


# ------------------------------------------------------------------ loader --

def test_a_valid_verified_profile_loads(tmp_path, pins):
    prof = cp.load_profile(write(tmp_path, verified_doc(pins)), allow_provisional=False, pins=pins)
    assert prof.status == "verified" and prof.verified
    assert prof.workload_class("2k").max_concurrency == 4
    with pytest.raises(cp.ProfileError):
        prof.workload_class("8k")


def test_tampering_fails_the_content_hash(tmp_path, pins):
    doc = verified_doc(pins)
    doc["classes"][0]["max_concurrency"] = 64
    assert "content_sha256" in refusal(tmp_path, doc)


def test_provisional_is_refused_without_the_flag_and_allowed_with_it(tmp_path, pins):
    doc = provisional_doc(pins)
    msg = refusal(tmp_path, doc, allow_provisional=False, pins=pins)
    assert "provisional" in msg and "--allow-provisional" in msg and "CAPSERVE_ALLOW_PROVISIONAL=1" in msg
    prof = cp.load_profile(write(tmp_path, doc), allow_provisional=True, pins=pins)
    assert prof.status == "provisional" and not prof.verified


def test_provisional_requires_a_reason(tmp_path, pins):
    for reason in (None, "", "   "):
        doc = provisional_doc(pins)
        doc["provisional_reason"] = reason
        assert "provisional_reason" in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(provisional_reason="why"), "provisional_reason"),
    (lambda d: d.update(provenance=None), "provenance"),
    (lambda d: d.update(slo=None), "slo"),
    (lambda d: d["provenance"].update(source_session="pilot-20261101"), "formal-"),
    (lambda d: d["provenance"].update(analysis_sha256="abc"), "analysis_sha256"),
    (lambda d: d["provenance"].update(analysis_sha256="AB" * 32), "analysis_sha256"),
    (lambda d: d["provenance"].update(freeze_commit="not-a-sha"), "freeze_commit"),
    (lambda d: d["provenance"].update(generator_commit=""), "generator_commit"),
    (lambda d: d["provenance"].update(analysis_path=""), "analysis_path"),
    (lambda d: d["provenance"].update(prereg_section="§7"), "prereg_section"),
    (lambda d: d["server"].update(fingerprint_sha256=None), "fingerprint_sha256"),
    (lambda d: d["server"].update(fingerprint_sha256="0" * 63), "fingerprint_sha256"),
    (lambda d: d["server"].update(num_gpu_blocks=None), "num_gpu_blocks"),
    (lambda d: d["server"].update(block_size=0), "block_size"),
])
def test_verified_rules(tmp_path, pins, edit, needle):
    doc = verified_doc(pins)
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("over, needle", [
    ({"max_prompt_tokens": 32700, "max_output_tokens": 128}, "32768"),
    ({"max_prompt_tokens": 0}, "max_prompt_tokens"),
    ({"max_output_tokens": 0}, "max_output_tokens"),
    ({"max_concurrency": 0}, "max_concurrency"),
    ({"max_concurrency": 129}, "max_concurrency"),
    ({"max_queue": -1}, "max_queue"),
])
def test_class_bounds(tmp_path, pins, over, needle):
    doc = verified_doc(pins)
    doc["classes"] = [_class(**over)]
    assert needle in refusal(tmp_path, rehash(doc), pins=pins)


def test_bounds_at_the_limits_load(tmp_path, pins):
    doc = verified_doc(pins)
    doc["classes"] = [_class(max_prompt_tokens=32640, max_output_tokens=128, max_concurrency=128, max_queue=0),
                      _class(name="1", max_prompt_tokens=1, max_output_tokens=1, max_concurrency=1)]
    prof = cp.load_profile(write(tmp_path, rehash(doc)), allow_provisional=False, pins=pins)
    assert [c.name for c in prof.classes] == ["2k", "1"]


def test_class_names_are_unique_and_classes_non_empty(tmp_path, pins):
    doc = verified_doc(pins)
    doc["classes"] = [_class(), _class(max_concurrency=2)]
    assert "duplicate" in refusal(tmp_path, rehash(doc), pins=pins)
    doc["classes"] = []
    assert "classes" in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(extra_key=1), "extra_key"),
    (lambda d: d["server"].update(gpu="5070"), "gpu"),
    (lambda d: d["classes"][0].update(priority=1), "priority"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.update(status="draft"), "status"),
    (lambda d: d["classes"][0].update(max_concurrency="4"), "max_concurrency"),
    (lambda d: d["classes"][0].update(max_concurrency=True), "max_concurrency"),
    (lambda d: d.pop("provenance"), "provenance"),
])
def test_unknown_fields_and_wrong_types_are_rejected(tmp_path, pins, edit, needle):
    doc = verified_doc(pins)
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("key, value", [("vllm_version", "0.31.0"), ("model_repo", "Qwen/Qwen3.5-9B"),
                                        ("model_revision", "0" * 40)])
def test_server_fields_must_equal_pins(tmp_path, pins, key, value):
    doc = verified_doc(pins)
    doc["server"][key] = value
    assert key in refusal(tmp_path, rehash(doc), pins=pins)


def test_unreadable_or_invalid_json_is_a_profile_error(tmp_path, pins):
    with pytest.raises(cp.ProfileError):
        cp.load_profile(tmp_path / "missing.json", allow_provisional=True, pins=pins)
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(cp.ProfileError):
        cp.load_profile(tmp_path / "bad.json", allow_provisional=True, pins=pins)


def test_loader_defaults_to_the_repo_pins(tmp_path, pins):
    assert cp.load_profile(write(tmp_path, verified_doc(pins)), allow_provisional=False).verified


# ------------------------------------------------------------------ committed profiles --

def test_provisional_dev_profile_is_valid_provisional_and_says_it_is_not_a_capacity(pins):
    path = SERVICE / "profiles" / "provisional-dev.json"
    with pytest.raises(cp.ProfileError):
        cp.load_profile(path, allow_provisional=False, pins=pins)
    prof = cp.load_profile(path, allow_provisional=True, pins=pins)
    assert prof.status == "provisional" and prof.profile_id == "provisional-dev"
    assert "not a capacity" in prof.provisional_reason.lower()
    assert prof.provenance is None and prof.slo is None


def test_every_committed_profile_hashes_to_its_content():
    for path in sorted((SERVICE / "profiles").glob("*.json")):
        doc = json.loads(path.read_text())
        assert doc["content_sha256"] == cp.content_sha256(doc), path.name


# ------------------------------------------------------------------ fingerprint --

def test_fingerprint_drops_only_host_and_port_and_includes_the_env():
    env = {"VLLM_GDN_DECODE_KERNEL": "triton"}
    a = ["/x/vllm", "serve", "m", "--host", "127.0.0.1", "--port", "8000", "--max-num-seqs", "128"]
    b = ["/y/vllm", "serve", "m", "--host", "0.0.0.0", "--port", "8100", "--max-num-seqs", "128"]
    c = ["/x/vllm", "serve", "m", "--host=127.0.0.1", "--port=9", "--max-num-seqs", "128"]
    assert cp.server_fingerprint(a, env) == cp.server_fingerprint(b, env) == cp.server_fingerprint(c, env)
    assert cp.server_fingerprint(a, env) != cp.server_fingerprint(a[:-1] + ["64"], env)
    assert cp.server_fingerprint(a, env) != cp.server_fingerprint(a[:2] + a[3:], env)
    assert cp.server_fingerprint(a, env) != cp.server_fingerprint(a, {"VLLM_GDN_DECODE_KERNEL": "fla"})
    assert len(cp.server_fingerprint(a, env)) == 64


def test_stage_b_fingerprint_is_the_core_b_server_argv_and_recorded_env(pins):
    sb = pins["stage_b"]
    argv, env = bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                               gpu_memory_utilization=sb["gpu_memory_utilization"], port=8000,
                               kv_cache_memory_bytes=sb["kv_cache_memory_bytes"], vllm_bin="/elsewhere/vllm",
                               base_env={"PATH": "/usr/bin", "HOME": "/home/u"})
    assert cp.stage_b_fingerprint(pins) == cp.server_fingerprint(argv, bp.recorded_env(env))
    other = copy.deepcopy(pins)
    other["stage_b"]["wsl2_pin_memory"] = "1"
    assert cp.stage_b_fingerprint(other) != cp.stage_b_fingerprint(pins)
    other = copy.deepcopy(pins)
    other["stage_b"]["kv_cache_memory_bytes"] = 3 * 2**30
    assert cp.stage_b_fingerprint(other) != cp.stage_b_fingerprint(pins)
