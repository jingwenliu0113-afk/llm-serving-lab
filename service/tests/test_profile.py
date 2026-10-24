"""Tests for capserve/profile.py: the capacity profile schema, canonical hash, loader rules and server fingerprint
(decision 029 "Capacity profile"; 027 §4), the per-class statuses of a verified profile (decision 030 8.7:
verified with a number and its bound, verified 0 = refuse the class, unavailable with a reason and no number), and the
Stage D profiles of decision 043 (§9: their classes, servers, fingerprints and evidence; the serving_class refusals:
an output limit above 128 tokens, on a T profile anything but an M8K-T class with H-D2a supported, and on a B0 profile
a T workload's class), and the Stage V profiles of decision 044 (§12: checked against a frozen configs/stage_v.toml,
committed before Stage V's freeze with [freeze] stage_v = false, so every §12 profile fails closed until then; the
pacing copies, V1's validated classes at V1's pinned C and points, the weighted-v1 admission section, and the §12
fields nowhere else)."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import stage_d_plan as sdplan
from capserve import profile as cp
from fakes.profiles import FROZEN, HEX, frozen_capacity, frozen_configs, frozen_stage_d, rehash, verified_class
from fakes.profiles import ABOVE_CKV_NOT_COVERED, NOT_COVERED, STAGE_D_CLASSES, stage_d_class, stage_d_doc
from fakes.profiles import FROZEN_V, GATEWAY_PATH_SHA256, PACING, SLOW_QUEUE, V1_NOT_COVERED, admission_section
from fakes.profiles import frozen_stage_v, stage_v_copy, stage_v_doc, v1_class, validation
from fakes.profiles import t_fingerprint as hand_t_fingerprint
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


@pytest.mark.parametrize("name", sorted(cp.CLASS_NAMES.values()) + sorted(cp.STAGE_D_CLASS_NAMES.values()))
def test_a_provisional_profile_never_uses_a_verified_class_name(tmp_path, pins, name):
    """Phase 2 closure (036): the Stage C class names belong to verified profiles. A provisional profile with a class
    named uniform-2k could otherwise be served under a production config that selects uniform-2k, so it is refused
    even with --allow-provisional."""
    doc = provisional_doc(pins)
    doc["classes"] = [_class(name=name)]
    msg = refusal(tmp_path, rehash(doc), pins=pins)
    assert name in msg and "reserved" in msg


@pytest.mark.parametrize("pid", ["verified-formal-20261017-c1b", "verified-other", "verified-"])
def test_a_provisional_profile_id_never_starts_with_verified(tmp_path, pins, pid):
    doc = provisional_doc(pins)
    doc["profile_id"] = pid
    msg = refusal(tmp_path, rehash(doc), pins=pins)
    assert "profile_id" in msg and "verified-" in msg


def test_the_committed_provisional_profiles_use_no_reserved_name(pins):
    for path in sorted((SERVICE / "profiles").glob("provisional-*.json")):
        prof = cp.load_profile(path, allow_provisional=True, pins=pins)
        assert not prof.verified and not prof.profile_id.startswith("verified-")
        reserved = set(cp.CLASS_NAMES.values()) | set(cp.STAGE_D_CLASS_NAMES.values())
        assert not {c.name for c in prof.classes} & reserved, path.name


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
    stage_d = (root / "configs" / "stage_d.toml").read_bytes()   # 043: read too (a release's git archive has it)
    stage_v = (root / "configs" / "stage_v.toml").read_bytes()   # 044: the harness's, frozen only at Stage V's freeze
    assert tomllib.loads(stage_v.decode())["freeze"]["stage_v"] is False
    assert cp.frozen_configs() == (tomllib.loads(cap.decode()), hashlib.sha256(cap).hexdigest(),
                                   hashlib.sha256(pins).hexdigest(), tomllib.loads(stage_d.decode()),
                                   hashlib.sha256(stage_d).hexdigest(), tomllib.loads(stage_v.decode()),
                                   hashlib.sha256(stage_v).hexdigest())


def test_a_missing_capacity_toml_is_not_frozen(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pins.toml").write_bytes(b"x = 1\n")
    assert cp.frozen_configs(tmp_path) == ({}, None, hashlib.sha256(b"x = 1\n").hexdigest(), None, None, None, None)
    (tmp_path / "configs" / "capacity.toml").write_text("[freeze\n")
    with pytest.raises(cp.ProfileError):
        cp.frozen_configs(tmp_path)


def test_a_missing_stage_d_toml_is_none_and_a_broken_one_an_error(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pins.toml").write_bytes(b"x = 1\n")
    (tmp_path / "configs" / "stage_d.toml").write_bytes(b"[freeze]\nstage_d = true\n")
    got = cp.frozen_configs(tmp_path)
    assert (got.stage_d, got.stage_d_sha256) == ({"freeze": {"stage_d": True}},
                                                 hashlib.sha256(b"[freeze]\nstage_d = true\n").hexdigest())
    assert (got.capacity, got.capacity_sha256) == ({}, None)
    (tmp_path / "configs" / "stage_d.toml").write_text("[freeze\n")
    with pytest.raises(cp.ProfileError):
        cp.frozen_configs(tmp_path)


def test_frozen_configs_built_from_three_positional_values_have_no_stage_d():
    """043: the new fields have defaults, so FrozenConfigs(cfg, CAP_SHA, PINS_SHA) (the generator, its tests) works."""
    got = cp.FrozenConfigs({"freeze": {"capacity": True}}, "cd" * 32, "ef" * 32)
    assert (got.stage_d, got.stage_d_sha256) == (None, None) and got.capacity_sha256 == "cd" * 32
    assert (got.stage_v, got.stage_v_sha256) == (None, None)


def test_a_stage_v_toml_is_read_only_when_present(tmp_path):
    """044 item 6: the §12 loader reads configs/stage_v.toml (never a Stage V module) when it exists."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pins.toml").write_bytes(b"x = 1\n")
    assert cp.frozen_configs(tmp_path)[5:] == (None, None)
    raw = b'[freeze]\nstage_v = true\ngateway_path_sha256 = "' + b"c3" * 32 + b'"\n'
    (tmp_path / "configs" / "stage_v.toml").write_bytes(raw)
    got = cp.frozen_configs(tmp_path)
    assert (got.stage_v, got.stage_v_sha256) == (tomllib.loads(raw.decode()), hashlib.sha256(raw).hexdigest())
    (tmp_path / "configs" / "stage_v.toml").write_text("[freeze\n")
    with pytest.raises(cp.ProfileError, match="stage_v.toml"):
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


def test_t_fingerprint_is_b0_plus_the_threshold_from_the_frozen_stage_d_argv(pins):
    """043 §1.4: T's fingerprint calls stage_d_plan.server_argv_for (frozen, read-only); the values are the ones 043
    quotes, and the B0 one is the pinned c1b profile's."""
    assert cp.stage_b_fingerprint(pins) == "98b9f41be7b75b6bf431e6c090124156360edfc929ccd6164001468da05b6d4f"
    assert cp.t_fingerprint(pins) == "564d27ecfc45f162c564185cf8c9b4566f84208e788808aa6d302187474dd7cc"
    assert cp.t_fingerprint(pins) == hand_t_fingerprint(pins)
    argv, env = sdplan.server_argv_for(pins, "M8K-T", port=1234, vllm_bin="/other/vllm")
    assert cp.t_fingerprint(pins) == cp.server_fingerprint(argv, bp.recorded_env(env))
    other = copy.deepcopy(pins)
    other["advanced"]["long_prefill_token_threshold"] = 256
    assert cp.t_fingerprint(other) != cp.t_fingerprint(pins)
    assert cp.stage_b_fingerprint(other) == cp.stage_b_fingerprint(pins)


# ------------------------------------------------------------------ Stage D profiles (043, §9) --

def test_the_pinned_c1b_profile_loads_unchanged(pins):
    """043 §3: every new field is optional, so the pinned production profile keeps its bytes, hash and meaning."""
    path = SERVICE / "profiles" / "verified-formal-20261017-c1b.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        "963de428168e688733ad0692fab2a68b1d83ca644e5c9deda1126df8370759c8"
    prof = cp.load_profile(path, allow_provisional=False, pins=pins)
    assert prof.content_sha256 == "1863d7e1766bc41b4351981714c85eaf1530d984ca9d8e07866f56dae786c8bc"
    assert prof.provenance.prereg_section == "§8" and prof.provenance.stage_d_sha256 is None
    assert prof.server.long_prefill_token_threshold is None and prof.server_variant == "B0"
    for c in prof.classes:
        ev = c.evidence
        assert (ev.server, ev.output_tokens, ev.descriptive, ev.at_c_prod, ev.hypotheses, ev.not_covered) == \
            (None,) * 6, c.name
    assert prof.serving_class("uniform-2k").max_concurrency == 32


def test_the_stage_d_classes_are_the_d_a_workloads():
    """043 §2: one class per D-A workload, its name unlike any c1b one; M8K-T takes the 007 mix's 32,640-token long
    requests (WORKLOADS["M8K-T"]["prompt"] is None)."""
    assert set(cp.STAGE_D_CLASS_NAMES) == set(sdplan.SESSIONS["D-A1"]) | set(sdplan.SESSIONS["D-A2"])
    assert {w: cp.stage_d_class(w) for w in cp.STAGE_D_CLASS_NAMES} == {
        "U32K": ("uniform-32k", 32640, 128), "U8K-T": ("uniform-8k-t", 8064, 128),
        "M8K-T": ("mixed-8k-t", 32640, 128), "U2K-O512": ("uniform-2k-o512", 1920, 512),
        "U8K-O512": ("uniform-8k-o512", 8064, 512)}
    assert {w: v[:3] for w, v in STAGE_D_CLASSES.items()} == {w: cp.stage_d_class(w) for w in cp.STAGE_D_CLASS_NAMES}
    assert not set(cp.STAGE_D_CLASS_NAMES.values()) & set(cp.CLASS_NAMES.values())
    for w in ("U2K", "U2K-P", "U8K-P", "U2K-var", "U8K-var", "U2K-gw", "U8K-gw"):
        with pytest.raises(cp.ProfileError):
            cp.stage_d_class(w)


def test_not_covered_is_a_closed_enum_set_by_the_flags():
    """043 §5: every D-A class has the six items; ABOVE_CKV adds in_vllm_wait_beyond_c_kv; no other flag adds any.
    044 adds output_length_distribution to the enum, for V1's classes only."""
    assert list(cp.NOT_COVERED) == NOT_COVERED + [ABOVE_CKV_NOT_COVERED, "output_length_distribution"]
    for w in cp.STAGE_D_CLASS_NAMES:
        assert cp.not_covered_for(w, []) == NOT_COVERED
        assert cp.not_covered_for(w, ["WINDOW_SENSITIVE", "STRICT_COARSE"]) == NOT_COVERED
        assert cp.not_covered_for(w, ["WINDOW_SENSITIVE", "ABOVE_CKV"]) == NOT_COVERED + [ABOVE_CKV_NOT_COVERED]
    with pytest.raises(cp.ProfileError):
        cp.not_covered_for("U2K", [])


B0_CLASSES = [stage_d_class("U32K", value=2), stage_d_class("U2K-O512", value=32, flags=["WINDOW_SENSITIVE",
                                                                                          "ABOVE_CKV"]),
              stage_d_class("U8K-O512", value=12)]
T_CLASSES = [stage_d_class("U8K-T", value=4), stage_d_class("M8K-T", value=4, flags=["WINDOW_SENSITIVE"])]


def test_stage_d_profiles_of_both_servers_load(tmp_path, pins):
    b0 = load(write(tmp_path, stage_d_doc(pins, classes=B0_CLASSES), "b0.json"), allow_provisional=False, pins=pins)
    assert b0.profile_id == "verified-formal-20261101-d-a1-b0" and b0.server_variant == "B0"
    assert b0.server.fingerprint_sha256 == cp.stage_b_fingerprint(pins)
    o512 = b0.workload_class("uniform-2k-o512")
    assert (o512.max_concurrency, o512.bound, o512.evidence.output_tokens, o512.evidence.server) == (32, "exact", 512,
                                                                                                     "B0")
    assert o512.evidence.not_covered[-1] == "in_vllm_wait_beyond_c_kv"
    assert o512.evidence.descriptive.c_late.value == 32 and o512.evidence.at_c_prod.attainment_min == 1.0
    t = load(write(tmp_path, stage_d_doc(pins, server="T", classes=T_CLASSES), "t.json"), allow_provisional=False,
             pins=pins)
    assert t.profile_id == "verified-formal-20261101-d-a1-t" and t.server_variant == "T"
    assert (t.server.long_prefill_token_threshold, t.server.fingerprint_sha256) == (512, cp.t_fingerprint(pins))
    assert t.workload_class("mixed-8k-t").evidence.hypotheses == {"H-D2a": "supported"}


def test_stage_d_unavailable_and_verified_0_classes_load(tmp_path, pins):
    """030 8.7 holds for §9 too, with a stage_d source."""
    classes = [stage_d_class("U32K", "UNSTABLE"), stage_d_class("U2K-O512", "NT"),
               stage_d_class("U8K-O512", "INFEASIBLE", 0)]
    prof = load(write(tmp_path, stage_d_doc(pins, classes=classes)), allow_provisional=False, pins=pins)
    assert [(c.status, c.max_concurrency, c.source) for c in prof.classes] == [
        ("unavailable", None, "stage_d:U32K:c_prod:UNSTABLE"), ("unavailable", None, "stage_d:U2K-O512:c_prod:NT"),
        ("verified", 0, "stage_d:U8K-O512:c_prod:INFEASIBLE")]


def _edit_d_class(i):
    def edit(fn):
        return lambda d: fn(d["classes"][i])
    return edit


def _append_on(server, workload, value):
    """Append workload's class with evidence.server set to the profile's own server, so only the other half of the
    server rule (the workload's server in stage_d_plan.WORKLOADS) can refuse it."""
    def edit(d):
        cls = stage_d_class(workload, value=value)
        cls["evidence"]["server"] = server
        d["classes"].append(cls)
    return edit


@pytest.mark.parametrize("server, edit, needle", [
    ("B0", lambda d: d["provenance"].update(stage_d_sha256=None), "stage_d_sha256"),
    ("B0", lambda d: d["provenance"].pop("stage_d_sha256"), "stage_d_sha256"),
    ("B0", lambda d: d["provenance"].update(stage_d_sha256="AB" * 32), "stage_d_sha256"),
    ("B0", lambda d: d["provenance"].update(stage_d_sha256="00" * 32), "stage_d_sha256"),     # not this checkout's
    ("B0", lambda d: d["provenance"].update(capacity_sha256="00" * 32), "capacity_sha256"),
    ("B0", lambda d: d["provenance"].update(pins_sha256="00" * 32), "pins_sha256"),
    ("B0", lambda d: d["slo"].update(id="S3"), "slo"),
    ("B0", lambda d: d["slo"]["terms"].update(tpot_p95_ms=200), "slo"),
    ("B0", lambda d: d.update(profile_id="verified-formal-20261101-d-a1"), "profile_id"),
    ("B0", lambda d: d.update(profile_id="verified-formal-20261101-d-a1-t"), "profile_id"),
    ("T", lambda d: d.update(profile_id="verified-formal-20261101-d-a1-b0"), "profile_id"),
    ("B0", lambda d: d["provenance"].update(source_session="formal-20261101-d-a2"), "profile_id"),
    ("B0", lambda d: d["provenance"].update(analysis_path="results/capacity/formal-20261101-d-a1/analysis.json"),
     "analysis_path"),
    ("B0", lambda d: d["classes"].append(stage_d_class("U32K", value=3)), "more than one class"),
    ("B0", _edit_d_class(0)(lambda c: c.update(name="uniform-32k-x")), "uniform-32k"),
    ("B0", _edit_d_class(0)(lambda c: c.update(max_prompt_tokens=32000)), "max_prompt_tokens"),
    ("B0", _edit_d_class(2)(lambda c: c.update(max_output_tokens=128)), "max_output_tokens"),
    ("B0", _edit_d_class(0)(lambda c: c["evidence"].update(server="T")), "evidence.server is 'T'"),
    ("B0", _edit_d_class(0)(lambda c: c["evidence"].update(server=None)), "evidence.server is None"),
    ("B0", _edit_d_class(2)(lambda c: c["evidence"].update(output_tokens=128)), "output_tokens"),
    ("B0", _edit_d_class(2)(lambda c: c["evidence"].update(output_tokens=None)), "output_tokens"),
    ("B0", lambda d: d["classes"].append(stage_d_class("U8K-T", value=4)), "runs on server T"),     # a T class on B0
    ("T", lambda d: d["classes"].append(stage_d_class("U32K", value=2)), "runs on server B0"),      # a B0 class on T
    ("B0", _append_on("B0", "U8K-T", 4), "runs on server T"),     # the same, its evidence.server the profile's own
    ("T", _append_on("T", "U32K", 2), "runs on server B0"),
    ("B0", _edit_d_class(0)(lambda c: c.update(source="stage_c:U32K:c_prod:EXACT")), "source"),
    ("B0", _edit_d_class(1)(lambda c: c["evidence"]["not_covered"].pop()), "not_covered"),   # drops the ABOVE_CKV one
    ("B0", _edit_d_class(0)(lambda c: c["evidence"]["not_covered"].append("in_vllm_wait_beyond_c_kv")), "not_covered"),
    ("B0", _edit_d_class(0)(lambda c: c["evidence"]["not_covered"].reverse()), "not_covered"),
    ("B0", _edit_d_class(0)(lambda c: c["evidence"].update(not_covered=None)), "not_covered"),
    ("B0", _edit_d_class(0)(lambda c: c["evidence"].update(not_covered=["anything"])), "not_covered"),   # schema
    ("B0", lambda d: d["server"].update(long_prefill_token_threshold=512), "fingerprint"),  # B0's fingerprint, T's knob
    ("T", lambda d: d["server"].update(long_prefill_token_threshold=None), "fingerprint"),
    ("T", lambda d: d["server"].update(long_prefill_token_threshold=256), "long_prefill_token_threshold"),
    ("T", lambda d: d["server"].update(fingerprint_sha256=HEX), "fingerprint"),
])
def test_a_stage_d_profile_must_match_its_session_server_and_the_d_a_classes(tmp_path, pins, server, edit, needle):
    doc = stage_d_doc(pins, server=server, classes=copy.deepcopy(T_CLASSES if server == "T" else B0_CLASSES))
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins)


@pytest.mark.parametrize("workload", ["U2K", "U2K-P", "U8K-P", "U2K-var", "U8K-var", "U2K-gw", "U8K-gw", "U8K"])
def test_only_the_d_a_workloads_become_stage_d_classes(tmp_path, pins, workload):
    """043 §3: the sentinel, D-B, D-C (and any Stage C workload) never become a §9 class."""
    cls = stage_d_class("U32K", value=2)
    cls["evidence"]["workload"] = workload
    cls["source"] = f"stage_d:{workload}:c_prod:EXACT"
    msg = refusal(tmp_path, stage_d_doc(pins, classes=[cls]), allow_provisional=False, pins=pins)
    assert "workload" in msg and workload in msg


def test_a_stage_d_profile_needs_both_freezes_and_stage_d_tomls_slo(tmp_path, pins):
    doc = stage_d_doc(pins)
    for frozen, needle in ((frozen_configs(stage_d=frozen_stage_d(stage_d=False)), "[freeze] stage_d"),
                           (FROZEN._replace(stage_d=None, stage_d_sha256=None), "[freeze] stage_d"),
                           (frozen_configs(frozen_capacity(capacity=False)), "[freeze] capacity")):
        assert needle in refusal(tmp_path, doc, allow_provisional=False, pins=pins, frozen=frozen)
    s3 = frozen_stage_d()
    s3["slo"] = {**s3["slo"], "tpot_p95_ms": 150}
    msg = refusal(tmp_path, doc, allow_provisional=False, pins=pins, frozen=frozen_configs(stage_d=s3))
    assert "slo" in msg and "stage_d.toml" in msg


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d["provenance"].update(stage_d_sha256="a1" * 32), "stage_d_sha256"),
    (lambda d: d["server"].update(long_prefill_token_threshold=512), "long_prefill_token_threshold"),
    (lambda d: d["classes"][0]["evidence"].update(server="B0"), "server"),
    (lambda d: d["classes"][0]["evidence"].update(output_tokens=128), "output_tokens"),
    (lambda d: d["classes"][0]["evidence"].update(not_covered=list(NOT_COVERED)), "not_covered"),
    (lambda d: d["classes"][0]["evidence"].update(hypotheses={"H-D1": "supported"}), "hypotheses"),
    (lambda d: d["classes"][0].update(source="stage_d:U2K:c_prod:EXACT"), "source"),
])
def test_a_stage_c_profile_carries_none_of_the_stage_d_fields(tmp_path, pins, edit, needle):
    """§8's rules are unchanged; the §9-only fields are simply absent (null) in a §8 profile."""
    doc = verified_doc(pins)
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins)


def test_prereg_sections_are_stage_c_d_and_v():
    assert cp.PREREG_SECTIONS == ("§8", "§9", "§12") and cp.PREREG_SECTION == "§8"
    assert (cp.STAGE_D_SECTION, cp.STAGE_V_SECTION) == ("§9", "§12")


# ------------------------------------------------------------------ serving_class: the 043 refusals --

def test_a_verified_class_with_more_than_128_output_tokens_is_never_served(tmp_path, pins):
    """043 §4: U8K's C_PROD is 12 at 512 output tokens, but C 9 already FAILs at 128; a shorter output than measured
    is not covered, so an O512 class is never the active class, whatever flag is given."""
    prof = load(write(tmp_path, stage_d_doc(pins, classes=B0_CLASSES)), allow_provisional=False, pins=pins)
    for name in ("uniform-2k-o512", "uniform-8k-o512"):
        with pytest.raises(cp.ProfileError) as exc:
            prof.serving_class(name)
        assert "512" in str(exc.value) and "128" in str(exc.value) and "043" in str(exc.value)
    assert prof.serving_class("uniform-32k").max_concurrency == 2
    dev = cp.load_profile(SERVICE / "profiles" / "provisional-rtx.json", allow_provisional=True, pins=pins)
    assert dev.serving_class("rtx-it").max_output_tokens == 1024   # a provisional limit is no measured length


def test_on_a_t_profile_only_m8k_t_with_h_d2a_supported_is_served(tmp_path, pins):
    """030 8.8 via 043 §3: a T profile serves only a declared M8K deployment whose C_PROD beats B0's (H-D2a); U8K-T is
    reported, never served."""
    prof = load(write(tmp_path, stage_d_doc(pins, server="T", classes=T_CLASSES)), allow_provisional=False,
                pins=pins)
    with pytest.raises(cp.ProfileError) as exc:
        prof.serving_class("uniform-8k-t")
    assert "030 8.8" in str(exc.value) and "report" in str(exc.value)
    assert prof.serving_class("mixed-8k-t").max_concurrency == 4
    for hyp in ({"H-D2a": "not supported"}, {"H-D2a": "NT"}, {}):
        doc = stage_d_doc(pins, server="T", classes=[stage_d_class("M8K-T", value=4, hypotheses=hyp)])
        prof = load(write(tmp_path, doc), allow_provisional=False, pins=pins)
        with pytest.raises(cp.ProfileError) as exc:
            prof.serving_class("mixed-8k-t")
        assert "H-D2a" in str(exc.value) and "030 8.8" in str(exc.value)


@pytest.mark.parametrize("workload", ["U8K-T", "M8K-T"])
def test_a_t_workloads_class_on_a_b0_profile_is_never_served(pins, workload):
    """030 8.8, defence in depth: the refusal reads the model only (create_app may get a profile the loader never saw),
    so T's limit is never served on B0 even when the class's evidence.server claims B0."""
    cls = stage_d_class(workload, value=4)
    cls["evidence"]["server"] = "B0"
    prof = cp.CapacityProfile.model_validate(stage_d_doc(pins, classes=[stage_d_class("U32K", value=2), cls]))
    assert prof.server_variant == "B0"
    with pytest.raises(cp.ProfileError) as exc:
        prof.serving_class(cls["name"])
    assert "030 8.8" in str(exc.value) and "server T" in str(exc.value) and workload in str(exc.value)
    assert prof.serving_class("uniform-32k").max_concurrency == 2


def test_a_t_profile_class_without_evidence_is_never_served(pins):
    """The refusal reads the model only (create_app may get a profile the loader never saw): fail closed."""
    doc = stage_d_doc(pins, server="T", classes=T_CLASSES)
    doc["classes"][1]["evidence"] = None
    with pytest.raises(cp.ProfileError, match="030 8.8"):
        cp.CapacityProfile.model_validate(rehash(doc)).serving_class("mixed-8k-t")


# ------------------------------------------------------------------ Stage V profiles (044, §12) --

def v_load(tmp_path, doc, pins, frozen=FROZEN_V):
    return load(write(tmp_path, doc), allow_provisional=False, pins=pins, frozen=frozen)


def test_v1_not_covered_drops_what_v1_measured_and_adds_the_distribution():
    """044 "§12 的 class": V1 measured through the gateway and other output lengths, but only one distribution."""
    assert cp.v1_not_covered("U8K-O512", []) == V1_NOT_COVERED + ["output_length_distribution"]
    assert cp.v1_not_covered("U2K-O512", ["WINDOW_SENSITIVE", "ABOVE_CKV"]) == V1_NOT_COVERED + [
        ABOVE_CKV_NOT_COVERED, "output_length_distribution"]
    for w in cp.STAGE_D_CLASS_NAMES:
        assert "output_length_distribution" not in cp.not_covered_for(w, ["ABOVE_CKV"])   # 043's lists unchanged
    assert cp.V1_CLASS_NAMES == {"U2K-O512": "uniform-2k-o1-512", "U8K-O512": "uniform-8k-o1-512"}


def test_a_stage_v_profile_loads_after_stage_vs_freeze(tmp_path, pins):
    prof = v_load(tmp_path, stage_v_doc(pins), pins)
    assert prof.profile_id == "verified-formal-20261101-v-adm-b0" and prof.provenance.prereg_section == "§12"
    u2k = prof.serving_class("uniform-2k")
    assert (u2k.max_concurrency, u2k.pacing.burst_tokens, u2k.pacing.rate_tokens_per_s) == (32, 9600, 5920)
    assert u2k.pacing.queue.model_dump() == SLOW_QUEUE and u2k.validation.unit == "PACE-2K"
    assert prof.admission.budget == 32 and [c.name for c in prof.admission.classes] == ["uniform-2k", "uniform-8k"]
    assert prof.provenance.gateway_path_sha256 == GATEWAY_PATH_SHA256
    cls = v_load(tmp_path, stage_v_doc(pins, classes=[v1_class("U2K-O512"), v1_class("U8K-O512")], admission=None,
                                       session="formal-20261102-v-cls"), pins)
    o1 = cls.serving_class("uniform-8k-o1-512")   # 044: 512 output tokens validated, so served (043 §4 lifted)
    assert (o1.max_concurrency, o1.max_output_tokens, o1.bound, o1.evidence.c_prod.value) == (8, 512, "validated", 12)
    assert o1.evidence.via_gateway is True and o1.evidence.output_distribution == "uniform[1,512]"
    plain = stage_v_doc(pins, classes=[stage_v_copy("U2K", pacing=False), stage_v_copy("U8K", pacing=False)])
    assert v_load(tmp_path, plain, pins).workload_class("uniform-8k").pacing is None   # W1 alone


def test_no_stage_v_profile_loads_before_stage_vs_freeze(tmp_path, pins):
    """044 item 6: configs/stage_v.toml has [freeze] stage_v = true only from Stage V's freeze; without it, or not
    frozen (this checkout: the harness's stage_v.toml, stage_v = false), every §12 profile is refused: fail closed."""
    doc = stage_v_doc(pins)
    for frozen in (FROZEN, cp.frozen_configs(), frozen_configs(stage_v=frozen_stage_v(stage_v=False)),
                   frozen_configs(stage_v={"slo": frozen_stage_v()["slo"]})):
        msg = refusal(tmp_path, doc, allow_provisional=True, pins=pins, frozen=frozen)
        assert "[freeze] stage_v" in msg
    for frozen, needle in ((frozen_configs(frozen_capacity(capacity=False), stage_v=frozen_stage_v()),
                            "[freeze] capacity"),
                           (frozen_configs(stage_d=frozen_stage_d(stage_d=False), stage_v=frozen_stage_v()),
                            "[freeze] stage_d"),
                           (frozen_configs(stage_v=frozen_stage_v(gateway_path_sha256="00" * 32)),
                            "gateway_path_sha256")):
        assert needle in refusal(tmp_path, doc, allow_provisional=False, pins=pins, frozen=frozen)
    s3 = frozen_stage_v()
    s3["slo"]["tpot_p95_ms"] = 150
    assert "stage_v.toml [slo]" in refusal(tmp_path, doc, allow_provisional=False, pins=pins,
                                           frozen=frozen_configs(stage_v=s3))


def _v_cls(i):
    def edit(fn):
        return lambda d: fn(d["classes"][i])
    return edit


def _v1(fn, needle):
    """An edit of a V-CLS profile (V1's two classes, no admission section)."""
    return ("V1", fn, needle)


@pytest.mark.parametrize("kind, edit, needle", [
    ("ADM", lambda d: d["provenance"].update(stage_v_sha256="00" * 32), "stage_v_sha256"),
    ("ADM", lambda d: d["provenance"].update(stage_v_sha256=None), "stage_v_sha256"),
    ("ADM", lambda d: d["provenance"].update(stage_d_sha256="00" * 32), "stage_d_sha256"),
    ("ADM", lambda d: d["provenance"].update(capacity_sha256="00" * 32), "capacity_sha256"),
    ("ADM", lambda d: d["provenance"].update(gateway_path_sha256="00" * 32), "gateway_path_sha256"),
    ("ADM", lambda d: d["provenance"].update(gateway_path_sha256=None), "gateway_path_sha256"),
    ("ADM", lambda d: d["provenance"].update(capserve_sha256="nope"), "capserve_sha256"),
    ("ADM", lambda d: d["provenance"].update(base_profiles=[]), "base_profiles"),
    ("ADM", lambda d: d["provenance"].update(base_profiles=None), "base_profiles"),
    ("ADM", lambda d: d["provenance"]["base_profiles"].append({"id": "verified-x", "content_sha256": "X"}),
     "base_profiles"),
    ("ADM", lambda d: d["slo"].update(id="S3"), "slo"),
    ("ADM", lambda d: d.update(profile_id="verified-formal-20261101-v-adm"), "profile_id"),
    ("ADM", lambda d: d["provenance"].update(analysis_path="results/stage_d/formal-20261101-v-adm/analysis.json"),
     "analysis_path"),
    ("ADM", lambda d: d["server"].update(long_prefill_token_threshold=512), "B0"),
    # the copies
    ("ADM", _v_cls(0)(lambda c: c.update(name="uniform-2k-paced")), "uniform-2k"),
    ("ADM", _v_cls(0)(lambda c: c.update(max_prompt_tokens=8064)), "max_prompt_tokens"),
    ("ADM", _v_cls(0)(lambda c: c["evidence"].update(workload="M2K")), "uniform-2k or uniform-8k"),
    ("ADM", _v_cls(0)(lambda c: c["evidence"].update(server="B0")), "§8 class"),
    ("ADM", _v_cls(0)(lambda c: c["evidence"].update(via_gateway=True)), "§8 class"),
    ("ADM", _v_cls(0)(lambda c: c["pacing"].update(burst_tokens=1919)), "burst"),
    ("ADM", _v_cls(0)(lambda c: c.update(validation=None)), "validation"),
    ("ADM", _v_cls(0)(lambda c: c["validation"].update(unit="PACE-8K")), "PACE-2K"),
    ("ADM", _v_cls(0)(lambda c: c["validation"].update(study="V4")), "V2"),
    ("ADM", _v_cls(0)(lambda c: c["validation"].update(points=[])), "points"),
    ("ADM", _v_cls(0)(lambda c: c["validation"]["worst_r"].update({"P-X": {"client": 0.5}})), "worst_r"),
    ("ADM", _v_cls(0)(lambda c: c.update(pacing=None)), "validation"),                # validation without pacing
    ("ADM", _v_cls(0)(lambda c: c.update(bound="validated", source="stage_v:PACE-2K")), "V1"),
    ("ADM", lambda d: d["classes"].append(stage_v_copy("U2K")), "more than one class"),
    ("ADM", lambda d: d.update(admission=None, classes=[stage_v_copy("U2K"), stage_v_copy("U8K", pacing=False)]),
     "admission"),                                                          # a plain copy no section names
    # the admission section
    ("ADM", lambda d: d["admission"]["classes"][1].update(name="uniform-8k-o1-512"), "uniform-8k-o1-512"),
    ("ADM", lambda d: d["admission"]["classes"][1].update(max_concurrency=9, units=4), "max_concurrency"),
    ("ADM", lambda d: d["admission"]["classes"][1].update(max_concurrency=16, units=2), "caps are the profile's"),
    ("ADM", lambda d: d["admission"]["classes"][1].update(units=3), "budget"),
    ("ADM", lambda d: d["admission"].update(budget=64), "budget"),
    ("ADM", lambda d: d["admission"]["classes"].reverse(), "increasing"),
    ("ADM", lambda d: d["admission"]["classes"].append(dict(d["admission"]["classes"][0])), "twice"),
    ("ADM", lambda d: d["admission"].update(gateway_path_sha256="00" * 32), "gateway_path_sha256"),
    ("ADM", lambda d: d["admission"].update(measured_mixes=[]), "measured_mixes"),
    ("ADM", lambda d: d["admission"].update(measured_mixes=[0.9, 1.0]), "measured_mixes"),
    ("ADM", lambda d: d["admission"]["validation"].update(unit="W2"), "W1"),
    # V1's classes
    _v1(_v_cls(0)(lambda c: c.update(name="uniform-2k-o512")), "uniform-2k-o1-512"),
    _v1(_v_cls(0)(lambda c: c.update(max_output_tokens=128)), "max_output_tokens"),
    _v1(_v_cls(1)(lambda c: c.update(max_queue=12)), "max_queue"),
    _v1(_v_cls(1)(lambda c: c.update(source="stage_v:V1-2K")), "source"),
    _v1(_v_cls(1)(lambda c: c.update(bound="exact")), "source"),
    _v1(_v_cls(1)(lambda c: c.update(status="unavailable", unavailable_reason="x", max_concurrency=None,
                                     max_queue=None)), "verified"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(workload="U32K")), "workload"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(output_tokens=128)), "output_tokens"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(server="T")), "server"),
    _v1(_v_cls(0)(lambda c: c["evidence"]["not_covered"].insert(1, "via_gateway")), "not_covered"),
    _v1(_v_cls(1)(lambda c: c["evidence"].update(not_covered=NOT_COVERED)), "not_covered"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(output_distribution="uniform[1,128]")), "output_distribution"),
    _v1(_v_cls(0)(lambda c: c["evidence"]["datasets_sha256"].pop()), "datasets_sha256"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(datasets_sha256=["ab" * 32] * 5)), "datasets_sha256"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(via_gateway=False)), "via_gateway"),
    _v1(_v_cls(0)(lambda c: c["evidence"].update(via_gateway=None)), "via_gateway"),
    _v1(_v_cls(0)(lambda c: c.update(pacing={**PACING["U2K"], "queue": dict(SLOW_QUEUE)})), "pacing"),
    _v1(_v_cls(0)(lambda c: c.update(validation=None)), "validation"),
    _v1(_v_cls(0)(lambda c: c["validation"].update(unit="V1-8K")), "V1-2K"),
    # 044 實作補記六: the C is V1's fixed point (V1_C), never the O512 C_PROD, nor a neighbour of the point
    _v1(_v_cls(1)(lambda c: c.update(max_concurrency=12, max_queue=12)), "044 V1: the point's C"),   # the O512 C_PROD
    _v1(_v_cls(1)(lambda c: c.update(max_concurrency=9, max_queue=9)), "044 V1: the point's C"),
    _v1(_v_cls(0)(lambda c: c.update(max_concurrency=64, max_queue=64)), "044 V1: the point's C"),
    _v1(_v_cls(0)(lambda c: c.update(max_concurrency=31, max_queue=31)), "044 V1: the point's C"),
    # ... and validation.points are the unit's points, in [stage_v.units] order
    _v1(_v_cls(0)(lambda c: c["validation"].update(points=["GW-U2K-O512", "L-U2K"])), "the points of V1-2K"),
    _v1(_v_cls(1)(lambda c: c.update(validation=validation("V1", "V1-8K", ["L-U8K"], ["v-uniform-8k-o512-c8"]))),
        "the points of V1-8K"),
    _v1(_v_cls(1)(lambda c: c.update(validation=validation("V1", "V1-8K", ["L-U2K", "GW-U2K-O512"],
                                                           ["v-uniform-8k-o512-c8", "v-uniform-8k-o512"]))),
        "the points of V1-8K"),
])
def test_a_stage_v_profile_must_match_its_session_its_freeze_and_the_044_mapping(tmp_path, pins, kind, edit, needle):
    if kind == "V1":
        doc = stage_v_doc(pins, classes=[v1_class("U2K-O512"), v1_class("U8K-O512")], admission=None,
                          session="formal-20261102-v-cls")
    else:
        doc = stage_v_doc(pins)
    edit(doc)
    msg = refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins, frozen=FROZEN_V)
    assert needle in msg, msg


def test_v1s_pinned_c_and_points_are_the_stage_v_configs():
    """044 實作補記六: the loader pins a V1 class's max_concurrency = max_queue to V1_C (the C of the unit's L point,
    which the GW-O512 point covers at a C >= it) and its validation.points to V1_POINTS, both the harness's
    configs/stage_v.toml values (the loader itself reads only [freeze] and [slo] of that file)."""
    cfg = tomllib.loads((SERVICE.parent / "configs" / "stage_v.toml").read_text())["stage_v"]
    assert cp.V1_C == {"U2K-O512": 32, "U8K-O512": 8}
    for w, unit in cp.V1_UNITS.items():
        assert list(cp.V1_POINTS[w]) == cfg["units"][unit], w
        pts = {p: cfg["points"][p] for p in cp.V1_POINTS[w]}
        assert all(x["unit"] == unit and x["server_base"] == w for x in pts.values()), w
        (l_point,) = [x for x in pts.values() if x["study"] == "V1"]
        assert l_point["C"] == cp.V1_C[w] and l_point["limits"]["max_concurrency"] == cp.V1_C[w], w
        assert all(x["C"] >= cp.V1_C[w] for x in pts.values()), w   # the all-512 GW point covers the C (044 差異 9)


@pytest.mark.parametrize("doc_of, edit, needle", [
    (lambda p: verified_doc(p), lambda d: d["classes"][0].update(pacing={**PACING["U2K"], "queue": SLOW_QUEUE}),
     "pacing"),
    (lambda p: verified_doc(p), lambda d: d["classes"][0].update(validation=validation("V2", "PACE-2K", ["P"], ["v"])),
     "validation"),
    (lambda p: verified_doc(p), lambda d: d.update(admission=admission_section()), "admission"),
    (lambda p: verified_doc(p), lambda d: d["provenance"].update(gateway_path_sha256="c3" * 32), "gateway_path"),
    (lambda p: verified_doc(p), lambda d: d["classes"][0]["evidence"].update(via_gateway=True), "via_gateway"),
    (lambda p: stage_d_doc(p), lambda d: d.update(admission=admission_section()), "admission"),
    (lambda p: stage_d_doc(p), lambda d: d["provenance"].update(stage_v_sha256="b2" * 32), "stage_v_sha256"),
    (lambda p: stage_d_doc(p), lambda d: d["classes"][0]["evidence"].update(datasets_sha256=["ab" * 32]),
     "datasets_sha256"),
    (lambda p: provisional_doc(p), lambda d: d.update(admission=admission_section()), "admission"),
    (lambda p: provisional_doc(p), lambda d: d["classes"][0].update(pacing={**PACING["U2K"], "queue": SLOW_QUEUE}),
     "pacing"),
])
def test_the_stage_v_fields_never_appear_outside_section_12(tmp_path, pins, doc_of, edit, needle):
    """044: a pacing, an admission section, V1's evidence and the Stage V provenance exist only in a §12 profile, so
    no candidate policy can be attached to c1b's, a Stage D or a provisional profile by hand."""
    doc = doc_of(pins)
    edit(doc)
    assert needle in refusal(tmp_path, rehash(doc), allow_provisional=True, pins=pins, frozen=FROZEN_V)


@pytest.mark.parametrize("name", ["uniform-2k-o1-512", "uniform-8k-o1-512"])
def test_a_provisional_profile_never_uses_v1s_class_names(tmp_path, pins, name):
    doc = provisional_doc(pins)
    doc["classes"][0]["name"] = name
    assert "reserved" in refusal(tmp_path, rehash(doc), pins=pins)


def test_bound_validated_is_only_v1s(tmp_path, pins):
    """044: "validated" (a fixed point V1 validated) is no 030 8.7 bound: a §8 / §9 class with it never loads, and a
    model that never went through the loader still gets 043 §4's refusal for its 512 output tokens."""
    doc = verified_doc(pins)
    doc["classes"][0]["bound"] = "validated"
    assert "bound" in refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins, frozen=FROZEN_V)
    doc = stage_d_doc(pins, classes=[stage_d_class("U2K-O512", value=32)])
    doc["classes"][0]["bound"] = "validated"
    assert "bound" in refusal(tmp_path, rehash(doc), allow_provisional=False, pins=pins, frozen=FROZEN_V)
    with pytest.raises(cp.ProfileError, match="043 §4"):
        cp.CapacityProfile.model_validate(rehash(doc)).serving_class("uniform-2k-o512")


# ------------------------------------------------------------------ the Stage V study profile (044 item 9) --

STUDY_BASES = {"v-uniform-2k": ("verified-formal-20261017-c1b", "uniform-2k"),
               "v-uniform-8k": ("verified-formal-20261017-c1b", "uniform-8k"),
               "v-uniform-2k-o512": ("verified-formal-20261023-d-a2-b0", "uniform-2k-o512"),
               "v-uniform-8k-o512": ("verified-formal-20261023-d-a2-b0", "uniform-8k-o512")}
LIMITS = ("max_prompt_tokens", "max_output_tokens", "max_concurrency", "max_queue")


def test_the_stage_v_study_profile_copies_its_base_classes_limits_field_by_field(pins):
    """044 item 9 / "和 043 字面的差異" 3: the v- classes the study serves with --allow-provisional have, field by field,
    the limits of the committed verified class they stand for (only the label and the status differ), on B0;
    v-uniform-8k-o512-c8 is uniform-8k-o512's tokens at the 8 V1-8K would unlock (c1b's uniform-8k)."""
    path = SERVICE / "profiles" / "provisional-stage-v.json"
    with pytest.raises(cp.ProfileError, match="allow-provisional"):
        cp.load_profile(path, allow_provisional=False, pins=pins)
    prof = cp.load_profile(path, allow_provisional=True, pins=pins)
    assert prof.profile_id == "provisional-stage-v" and "not a capacity" in prof.provisional_reason.lower()
    assert prof.server.fingerprint_sha256 == cp.stage_b_fingerprint(pins) and prof.server_variant == "B0"
    assert (prof.server.num_gpu_blocks, prof.server.block_size) == (pins["stage_b"]["num_gpu_blocks"],
                                                                    pins["stage_b"]["block_size"])
    assert [c.name for c in prof.classes] == list(STUDY_BASES) + ["v-uniform-8k-o512-c8"]
    for name, (pid, base) in STUDY_BASES.items():
        b = cp.load_profile(SERVICE / "profiles" / f"{pid}.json", allow_provisional=False, pins=pins)
        assert {k: getattr(prof.serving_class(name), k) for k in LIMITS} == \
            {k: getattr(b.workload_class(base), k) for k in LIMITS}, name
    o512 = cp.load_profile(SERVICE / "profiles" / "verified-formal-20261023-d-a2-b0.json", allow_provisional=False,
                           pins=pins).workload_class("uniform-8k-o512")
    u8k = cp.load_profile(SERVICE / "profiles" / "verified-formal-20261017-c1b.json", allow_provisional=False,
                          pins=pins).workload_class("uniform-8k")
    c8 = prof.serving_class("v-uniform-8k-o512-c8")
    assert (c8.max_prompt_tokens, c8.max_output_tokens) == (o512.max_prompt_tokens, o512.max_output_tokens)
    assert (c8.max_concurrency, c8.max_queue) == (u8k.max_concurrency, u8k.max_queue) == (8, 8)
    assert c8.max_concurrency == v1_class("U8K-O512")["max_concurrency"]   # the value V1-8K would unlock
