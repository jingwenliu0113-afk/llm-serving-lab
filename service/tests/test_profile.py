"""Tests for capserve/profile.py: the capacity profile schema, canonical hash, loader rules and server fingerprint
(decision 029 "Capacity profile"; 027 §4), and the per-class statuses of a verified profile (decision 030 8.7:
verified with a number and its bound, verified 0 = refuse the class, unavailable with a reason and no number)."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
from capserve import profile as cp
from fakes.profiles import FROZEN, HEX, frozen_capacity, frozen_configs, rehash, verified_class
from fakes.profiles import verified_doc as _verified_doc

SERVICE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    return bp.load_pins()


def _server(pins, **over):
    return {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
            "model_revision": pins["core_b"]["revision"], "fingerprint_sha256": HEX, "num_gpu_blocks": 248,
            "block_size": 528, **over}


def _class(**over):
    return {"name": "2k", "max_prompt_tokens": 1920, "max_output_tokens": 128, "max_concurrency": 4, "max_queue": 4,
            "source": "provisional:test", **over}


def verified_doc(pins, **kw):
    return _verified_doc(pins, fingerprint=HEX, **kw)


def provisional_doc(pins):
    doc = {"schema_version": 1, "profile_id": "provisional-test", "status": "provisional",
           "provisional_reason": "unit test; not a capacity", "slo": None,
           "server": _server(pins, fingerprint_sha256=None, num_gpu_blocks=None, block_size=None),
           "classes": [_class()], "provenance": None}
    return rehash(doc)


# one class per Stage C workload: every 030 8.7 status over two profiles
NUMBERED = [verified_class(workload="U2K", status="EXACT", value=7),
            verified_class(workload="U8K", status="LOWER", value=16),
            verified_class(workload="M2K", status="CENSORED", value=70),
            verified_class(workload="M8K", status="INFEASIBLE", value=0)]
UNAVAILABLE = [verified_class(workload="U8K", status="UNSTABLE"), verified_class(workload="M2K", status="NT")]


def write(tmp_path, doc, name="p.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def load(path, **kw):
    """load_profile against the frozen configs of the fakes (a verified profile exists only after the freeze)."""
    kw.setdefault("frozen", FROZEN)
    return cp.load_profile(path, **kw)


def refusal(tmp_path, doc, *, allow_provisional=True, pins=None, frozen=FROZEN):
    with pytest.raises(cp.ProfileError) as exc:
        cp.load_profile(write(tmp_path, doc), allow_provisional=allow_provisional, pins=pins, frozen=frozen)
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
    prof = load(write(tmp_path, verified_doc(pins)), allow_provisional=False, pins=pins)
    assert prof.status == "verified" and prof.verified
    assert prof.workload_class("uniform-2k").max_concurrency == 4
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
    prof = load(write(tmp_path, doc), allow_provisional=True, pins=pins)
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
    (lambda d: d["provenance"].update(freeze_commit="146d260"), "freeze_commit"),        # 40 hex, never abbreviated
    (lambda d: d["provenance"].update(generator_commit=""), "generator_commit"),
    (lambda d: d["provenance"].update(deployed_commit="3" * 39), "deployed_commit"),
    (lambda d: d["provenance"].update(capacity_sha256="cd"), "capacity_sha256"),
    (lambda d: d["provenance"].update(pins_sha256=None), "pins_sha256"),
    (lambda d: d["provenance"].pop("gpu"), "gpu"),
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
    ({"max_concurrency": None}, "max_concurrency"),
    ({"max_queue": -1}, "max_queue"),
    ({"max_queue": None}, "max_queue"),
])
def test_class_bounds(tmp_path, pins, over, needle):
    doc = provisional_doc(pins)
    doc["classes"] = [_class(**over)]
    assert needle in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("over, needle", [
    ({"max_prompt_tokens": 32700, "max_output_tokens": 128}, "32768"),
    ({"value": 129}, "max_concurrency"),
])
def test_verified_class_bounds(tmp_path, pins, over, needle):
    doc = verified_doc(pins, classes=[verified_class(**over)])
    assert needle in refusal(tmp_path, doc, pins=pins)


def test_bounds_at_the_limits_load(tmp_path, pins):
    doc = provisional_doc(pins)
    doc["classes"] = [_class(max_prompt_tokens=32640, max_output_tokens=128, max_concurrency=128, max_queue=0),
                      _class(name="1", max_prompt_tokens=1, max_output_tokens=1, max_concurrency=1)]
    prof = load(write(tmp_path, rehash(doc)), allow_provisional=True, pins=pins)
    assert [c.name for c in prof.classes] == ["2k", "1"]
    doc = verified_doc(pins, classes=[verified_class(workload="M2K", status="CENSORED", value=128)])
    assert load(write(tmp_path, doc), allow_provisional=False, pins=pins).classes[0].max_concurrency == 128


def test_class_names_are_unique_and_classes_non_empty(tmp_path, pins):
    doc = verified_doc(pins)
    doc["classes"] = [verified_class(), verified_class(value=2)]
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
        load(tmp_path / "missing.json", allow_provisional=True, pins=pins)
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(cp.ProfileError):
        load(tmp_path / "bad.json", allow_provisional=True, pins=pins)


def test_loader_defaults_to_the_repo_pins(tmp_path, pins):
    assert load(write(tmp_path, verified_doc(pins)), allow_provisional=False).verified


# ------------------------------------------------------------------ class statuses (030 8.7) --

def test_a_verified_profile_states_each_class_from_its_c_prod(tmp_path, pins):
    """EXACT / LOWER / CENSORED -> verified with that number and bound; INFEASIBLE -> verified 0 (refuse the class);
    UNSTABLE / NT -> unavailable with a reason and no number."""
    got = {}
    for classes in (NUMBERED, UNAVAILABLE):
        prof = load(write(tmp_path, verified_doc(pins, classes=classes)), allow_provisional=False, pins=pins)
        got.update({c.evidence.c_prod.status: (c.status, c.max_concurrency, c.max_queue, c.bound, c.serviceable)
                    for c in prof.classes})
    assert got == {"EXACT": ("verified", 7, 7, "exact", True), "LOWER": ("verified", 16, 16, "lower", True),
                   "CENSORED": ("verified", 70, 70, "censored", True), "INFEASIBLE": ("verified", 0, 0, "exact", False),
                   "UNSTABLE": ("unavailable", None, None, None, False), "NT": ("unavailable", None, None, None, False)}
    assert prof.workload_class("mixed-2k").unavailable_reason


def _edit_class(pins, cls, edit):
    cls = copy.deepcopy(cls)
    edit(cls)
    return verified_doc(pins, classes=[cls])


@pytest.mark.parametrize("edit, needle", [
    (lambda c: c.update(unavailable_reason=None), "unavailable_reason"),
    (lambda c: c.update(unavailable_reason="  "), "unavailable_reason"),
    (lambda c: c.update(max_concurrency=4), "no number"),
    (lambda c: c.update(max_queue=0), "no number"),
    (lambda c: c.update(bound="lower"), "no number"),
    (lambda c: c["evidence"]["c_prod"].update(value=4, status="EXACT"), "UNSTABLE or NT"),
    (lambda c: c.update(source="stage_c:U8K:c_prod:NT"), "source"),
])
def test_an_unavailable_class_has_a_reason_and_never_a_number(tmp_path, pins, edit, needle):
    doc = _edit_class(pins, verified_class(workload="U8K", status="UNSTABLE"), edit)
    assert needle in refusal(tmp_path, doc, pins=pins)


@pytest.mark.parametrize("edit, needle", [
    (lambda c: c.update(bound="lower"), "verified 0"),
    (lambda c: c.update(bound=None), "bound"),
    (lambda c: c.update(max_queue=4), "max_queue"),
    (lambda c: c.update(source="stage_c:M8K:c_prod:EXACT")
     or c["evidence"]["c_prod"].update(status="EXACT"), "INFEASIBLE"),
    (lambda c: c["evidence"]["c_prod"].update(value=None), "C_PROD"),
])
def test_a_verified_0_needs_bound_exact_and_an_infeasible_source(tmp_path, pins, edit, needle):
    doc = _edit_class(pins, verified_class(workload="M8K", status="INFEASIBLE", value=0), edit)
    assert needle in refusal(tmp_path, doc, pins=pins)


@pytest.mark.parametrize("edit, needle", [
    (lambda c: c.update(bound="lower"), "bound"),
    (lambda c: c.update(bound=None), "bound"),
    (lambda c: c.update(max_concurrency=5), "C_PROD"),
    (lambda c: c.update(max_concurrency=None), "max_concurrency"),
    (lambda c: c.update(unavailable_reason="why"), "unavailable_reason"),
    (lambda c: c.update(source="stage_c:L=2048:uniform"), "source"),
    (lambda c: c.update(evidence=None), "evidence"),
    (lambda c: c.update(status=None), "status"),
    (lambda c: c["evidence"]["c_prod"].update(status="UNSTABLE"), "C_PROD"),
    (lambda c: c.update(status="draft"), "status"),
])
def test_a_verified_number_is_its_c_prod_with_its_bound(tmp_path, pins, edit, needle):
    doc = _edit_class(pins, verified_class(status="EXACT", value=4), edit)
    assert needle in refusal(tmp_path, doc, pins=pins)


@pytest.mark.parametrize("over, needle", [
    ({"status": "verified", "bound": "exact"}, "provisional"),
    ({"status": "unavailable", "unavailable_reason": "x", "max_concurrency": None, "max_queue": None}, "provisional"),
    ({"bound": "exact"}, "provisional"),
    ({"unavailable_reason": "x"}, "provisional"),
])
def test_a_provisional_profiles_classes_carry_no_status(tmp_path, pins, over, needle):
    doc = provisional_doc(pins)
    doc["classes"] = [_class(**over)]
    assert needle in refusal(tmp_path, rehash(doc), pins=pins)
    doc["classes"] = [{**_class(), "evidence": verified_class()["evidence"]}]
    assert "provisional" in refusal(tmp_path, rehash(doc), pins=pins)


@pytest.mark.parametrize("name", sorted(cp.CLASS_NAMES.values()))
def test_a_provisional_profile_never_uses_a_verified_class_name(tmp_path, pins, name):
    """Phase 2 closure (036): the Stage C class names belong to verified profiles. A provisional profile with a class
    named uniform-2k could otherwise be served under a production config that selects uniform-2k, so it is refused
    even with --allow-provisional."""
    doc = provisional_doc(pins)
    doc["classes"] = [_class(name=name)]
    msg = refusal(tmp_path, rehash(doc), pins=pins)
    assert name in msg and "reserved" in msg


def test_a_provisional_profile_id_never_starts_with_verified(tmp_path, pins):
    doc = provisional_doc(pins)
    doc["profile_id"] = "verified-formal-20261017-c1b"
    msg = refusal(tmp_path, rehash(doc), pins=pins)
    assert "profile_id" in msg and "verified-" in msg


def test_the_committed_provisional_profiles_use_no_reserved_name(pins):
    for path in sorted((SERVICE / "profiles").glob("provisional-*.json")):
        prof = cp.load_profile(path, allow_provisional=True, pins=pins)
        assert not prof.verified and not prof.profile_id.startswith("verified-")
        assert not {c.name for c in prof.classes} & set(cp.CLASS_NAMES.values()), path.name


def test_serving_class_refuses_an_unavailable_class_and_keeps_a_verified_0(tmp_path, pins):
    prof = load(write(tmp_path, verified_doc(pins, classes=UNAVAILABLE)), allow_provisional=False, pins=pins)
    for name in ("uniform-8k", "mixed-2k"):
        with pytest.raises(cp.ProfileError) as exc:
            prof.serving_class(name)
        assert "unavailable" in str(exc.value) and "never served" in str(exc.value)
    prof = load(write(tmp_path, verified_doc(pins, classes=NUMBERED)), allow_provisional=False, pins=pins)
    assert prof.serving_class("mixed-8k").serviceable is False
    assert prof.serving_class("uniform-2k").serviceable is True
    with pytest.raises(cp.ProfileError):
        prof.serving_class("nope")
    dev = load(write(tmp_path, provisional_doc(pins)), allow_provisional=True, pins=pins)
    assert dev.serving_class("2k").serviceable is True and dev.serving_class("2k").status is None


def test_slo_terms_are_copied_verbatim_with_one_nested_level(tmp_path, pins):
    doc = verified_doc(pins)
    assert doc["slo"]["terms"]["ttft_p95_ms"] == {"2048": 2000, "8192": 6000, "32768": 20000}
    assert load(write(tmp_path, doc), allow_provisional=False, pins=pins).slo.terms == doc["slo"]["terms"]
    doc["slo"]["terms"]["ttft_p95_ms"] = {"2048": {"p95": 2000}}
    assert "terms" in refusal(tmp_path, rehash(doc), pins=pins)


# ------------------------------------------------------------------ provenance at load time (029 M2) --

def forged(pins):
    """The review finding's hand-written verified profile: U8K (whose real C_PROD may be UNSTABLE) as a verified 13
    with matching evidence, under an SLO that is not capacity.toml's, from a session and analysis that do not exist."""
    cls = verified_class(workload="U8K", status="EXACT", value=13)
    doc = verified_doc(pins, classes=[cls], slo={"id": "S3", "terms": {"ttft_p95_ms": {"2048": 99999},
                                                                         "tpot_p95_ms": 999}})
    doc["provenance"]["source_session"] = "formal-does-not-exist"
    doc["provenance"]["analysis_path"] = "results/capacity/nowhere/analysis.json"
    return rehash(doc)


def test_a_forged_verified_profile_is_refused_at_load_time(tmp_path, pins):
    """Review finding: the loader only checked formats, so any self-hashed verified profile loaded and served."""
    msg = refusal(tmp_path, forged(pins), allow_provisional=False, pins=pins)
    assert "profile_id" in msg and "analysis_path" in msg and "slo" in msg


def test_no_verified_profile_loads_before_the_freeze(tmp_path, pins):
    """029: the generator refuses before [freeze] capacity = true, so no verified profile can exist before it."""
    for capacity in (frozen_capacity(capacity=False), frozen_capacity(capacity="true"), {}):
        msg = refusal(tmp_path, verified_doc(pins), allow_provisional=True, pins=pins, frozen=frozen_configs(capacity))
        assert "[freeze] capacity" in msg
    assert load(write(tmp_path, provisional_doc(pins)), allow_provisional=True, pins=pins,
                frozen=frozen_configs(frozen_capacity(capacity=False))).status == "provisional"


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(profile_id="verified-formal-20261102-c1"), "profile_id"),
    (lambda d: d["provenance"].update(source_session="formal-20261102-c1"), "profile_id"),
    (lambda d: d["provenance"].update(analysis_path="results/capacity/formal-20261101-c1/analysis-2.json"),
     "analysis_path"),
    (lambda d: d["provenance"].update(capacity_sha256="00" * 32), "capacity_sha256"),
    (lambda d: d["provenance"].update(pins_sha256="00" * 32), "pins_sha256"),
    (lambda d: d["slo"].update(id="S3"), "slo"),
    (lambda d: d["slo"]["terms"].update(tpot_p95_ms=200), "slo"),
    (lambda d: d["slo"]["terms"].pop("mixed_long"), "slo"),
    (lambda d: d["classes"][0].update(name="2k"), "uniform-2k"),
    (lambda d: d["classes"][0].update(max_prompt_tokens=8064), "max_prompt_tokens"),
    (lambda d: d["classes"][0].update(max_output_tokens=64), "max_output_tokens"),
    (lambda d: d["classes"][0]["evidence"].update(workload="U32K"), "workload"),
    (lambda d: d["classes"].append({**copy.deepcopy(d["classes"][0]), "name": "uniform-2k-again"}), "U2K"),
])
def test_a_verified_profile_must_match_its_session_the_frozen_configs_and_the_stage_c_classes(tmp_path, pins, edit,
                                                                                               needle):
    doc = verified_doc(pins)
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins)


def test_the_frozen_configs_default_to_the_repo_files():
    root = Path(bp.__file__).resolve().parent.parent
    cap, pins = (root / "configs" / "capacity.toml").read_bytes(), (root / "configs" / "pins.toml").read_bytes()
    assert cp.frozen_configs() == (tomllib.loads(cap.decode()), hashlib.sha256(cap).hexdigest(),
                                   hashlib.sha256(pins).hexdigest())


def test_a_missing_capacity_toml_is_not_frozen(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pins.toml").write_bytes(b"x = 1\n")
    assert cp.frozen_configs(tmp_path) == ({}, None, hashlib.sha256(b"x = 1\n").hexdigest())
    (tmp_path / "configs" / "capacity.toml").write_text("[freeze\n")
    with pytest.raises(cp.ProfileError):
        cp.frozen_configs(tmp_path)


def test_the_stage_c_classes_are_the_generators():
    assert {w: cp.stage_c_class(w) for w in cp.CLASS_NAMES} == {
        "U2K": ("uniform-2k", 1920, 128), "U8K": ("uniform-8k", 8064, 128), "M2K": ("mixed-2k", 32640, 128),
        "M8K": ("mixed-8k", 32640, 128)}


# ------------------------------------------------------------------ committed profiles --

def test_provisional_dev_profile_is_valid_provisional_and_says_it_is_not_a_capacity(pins):
    path = SERVICE / "profiles" / "provisional-dev.json"
    with pytest.raises(cp.ProfileError):
        load(path, allow_provisional=False, pins=pins)
    prof = load(path, allow_provisional=True, pins=pins)
    assert prof.status == "provisional" and prof.profile_id == "provisional-dev"
    assert "not a capacity" in prof.provisional_reason.lower()
    assert prof.provenance is None and prof.slo is None


def test_every_committed_profile_hashes_to_its_content():
    for path in sorted((SERVICE / "profiles").glob("*.json")):
        doc = json.loads(path.read_text())
        assert doc["content_sha256"] == cp.content_sha256(doc), path.name


def test_committed_provisional_profiles_still_load_unchanged(pins):
    """The per-class status fields are optional: a provisional profile's classes carry none and load as before."""
    for path in sorted((SERVICE / "profiles").glob("provisional-*.json")):
        prof = load(path, allow_provisional=True, pins=pins)
        assert all(c.status is None and c.evidence is None and c.serviceable for c in prof.classes), path.name


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
