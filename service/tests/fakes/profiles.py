"""Schema-faithful capacity profile documents for tests (decision 029 "Capacity profile"; the per-class statuses of
decision 030 8.7). A verified class is built here by hand from a C_PROD status, independently of the generator:
EXACT / LOWER / CENSORED -> verified with that number and bound, INFEASIBLE -> verified 0 (refuse the class),
UNSTABLE / NT -> unavailable with a reason and no number. Its name and token limits are its Stage C workload's
(030 8.7: U2K / U8K take L - 128 prompt tokens, the mixed M2K / M8K the 32,640-token long requests; 128 output).

A verified profile is checked at load time against the frozen configs (029 M2): FROZEN is the repo's capacity.toml
with [freeze] capacity = true and its stage_d.toml with [freeze] stage_d = true, and the capacity.toml / pins.toml /
stage_d.toml sha256 the provenances of verified_doc and stage_d_doc state.

A Stage D profile (decision 043, PREREGISTRATION §9) is built here by hand too: one per (session, server), B0 (no
long_prefill_token_threshold, the Stage B fingerprint) or T (threshold 512 from pins [advanced], the fingerprint of B0's
argv plus --long-prefill-token-threshold 512), its classes named and limited by the D-A workload (043 §2), every class
stating its server, output length, descriptive estimands, the C_PROD point's facts, its hypotheses and what it never
covered (043 §5: six items, plus in_vllm_wait_beyond_c_kv for ABOVE_CKV).

A Stage V profile (decision 044, PREREGISTRATION §12) is built here by hand too: B0, checked against a frozen
configs/stage_v.toml (frozen_stage_v: [freeze] stage_v and gateway_path_sha256, stage_d.toml's [slo]). The repo's file
exists before Stage V's freeze with [freeze] stage_v = false (cp.frozen_configs() is that repo state, and refuses
every §12 profile); FROZEN models a checkout without the file and FROZEN_V adds a frozen one. Its classes are copies
of c1b's uniform-2k / uniform-8k (with the pacing V2 validated, or plain members of the weighted-v1 admission section
V4 validated) and V1's validated classes (uniform-2k-o1-512 / uniform-8k-o1-512: the D-A O512 class's limits and
evidence, not_covered recomputed, the output distribution and its five datasets, at V1's C 32 / 8 over the unit's
points)."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import core_b_plan as bp
from capserve.profile import FrozenConfigs, content_sha256, server_fingerprint, stage_b_fingerprint

HEX = "ab" * 32
CAPACITY_SHA256, PINS_SHA256, STAGE_D_SHA256 = "cd" * 32, "ef" * 32, "a1" * 32
S2 = {"ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100, "mixed_long": "budget"}
BOUND = {"EXACT": "exact", "LOWER": "lower", "CENSORED": "censored", "INFEASIBLE": "exact"}
CLASSES = {"U2K": ("uniform-2k", 1920), "U8K": ("uniform-8k", 8064), "M2K": ("mixed-2k", 32640),
           "M8K": ("mixed-8k", 32640)}


def frozen_capacity(**freeze) -> dict:
    """The repo's configs/capacity.toml after the freeze (or with the given [freeze] values)."""
    cfg = tomllib.loads((Path(bp.__file__).resolve().parent.parent / "configs" / "capacity.toml").read_text())
    cfg["freeze"].update({"capacity": True, **freeze})
    return cfg


def frozen_stage_d(**freeze) -> dict:
    """The repo's configs/stage_d.toml after the freeze (or with the given [freeze] values)."""
    cfg = tomllib.loads((Path(bp.__file__).resolve().parent.parent / "configs" / "stage_d.toml").read_text())
    cfg["freeze"].update({"stage_d": True, **freeze})
    return cfg


GATEWAY_PATH_SHA256, STAGE_V_SHA256, CAPSERVE_SHA256 = "c3" * 32, "b2" * 32, "d4" * 32


def frozen_stage_v(**freeze) -> dict:
    """A frozen configs/stage_v.toml as the §12 loader reads it (044): [freeze] stage_v, mini, gateway_path_sha256,
    and [slo] = stage_d.toml's (or with the given [freeze] values)."""
    cfg = {"schema_version": 1, "freeze": {"stage_v": True, "mini": "smoke-v-20261101",
                                           "gateway_path_sha256": GATEWAY_PATH_SHA256},
           "slo": copy.deepcopy(frozen_stage_d()["slo"])}
    cfg["freeze"].update(freeze)
    return cfg


def frozen_configs(capacity: dict | None = None, *, capacity_sha256=CAPACITY_SHA256, pins_sha256=PINS_SHA256,
                   stage_d: dict | None = None, stage_d_sha256=STAGE_D_SHA256, stage_v: dict | None = None,
                   stage_v_sha256=STAGE_V_SHA256) -> FrozenConfigs:
    """What a verified profile is checked against at load time (default: the frozen repo capacity.toml and
    stage_d.toml, and no stage_v.toml; FrozenConfigs._replace(stage_d=None, stage_d_sha256=None) is a checkout without
    stage_d.toml)."""
    return FrozenConfigs(frozen_capacity() if capacity is None else capacity, capacity_sha256, pins_sha256,
                         frozen_stage_d() if stage_d is None else stage_d, stage_d_sha256, stage_v,
                         None if stage_v is None else stage_v_sha256)


FROZEN = frozen_configs()
FROZEN_V = frozen_configs(stage_v=frozen_stage_v())   # after Stage V's freeze (044)


def rehash(doc: dict) -> dict:
    doc = {k: v for k, v in doc.items() if k != "content_sha256"}
    doc["content_sha256"] = content_sha256(doc)
    return doc


def evidence(workload="U2K", status="EXACT", value=4, *, c_kv=35, c_cap=70) -> dict:
    numbered = bool(value)
    return {"workload": workload, "scope": "test scope", "c_prod": {"value": value, "status": status},
            "c_slo": {"value": value, "status": status}, "c_kv": c_kv, "c_cap": c_cap, "flags": [],
            "binding": ["tpot"] if status != "NT" else None,
            "margins": {"short": 0.5, "tpot": 0.1} if numbered else None,
            "burst_max_ttft_ms": 8000.0 if numbered else None}


def verified_class(name=None, *, workload="U2K", status="EXACT", value=4, max_prompt_tokens=None,
                   max_output_tokens=128) -> dict:
    """A class of a verified profile; its name and max_prompt_tokens default to its workload's (030 8.7)."""
    unavailable = status in ("UNSTABLE", "NT")
    limit = None if unavailable else (0 if status == "INFEASIBLE" else value)
    std_name, std_prompt = CLASSES.get(workload, (f"class-{workload}", 1920))
    name = std_name if name is None else name
    max_prompt_tokens = std_prompt if max_prompt_tokens is None else max_prompt_tokens
    return {"name": name, "max_prompt_tokens": max_prompt_tokens, "max_output_tokens": max_output_tokens,
            "max_concurrency": limit, "max_queue": limit, "source": f"stage_c:{workload}:c_prod:{status}",
            "status": "unavailable" if unavailable else "verified", "bound": None if unavailable else BOUND[status],
            "unavailable_reason": f"C_PROD is {status}: no verified admission limit (test)" if unavailable else None,
            "evidence": evidence(workload, status, limit)}


def verified_doc(pins: dict | None = None, *, classes: list | None = None, fingerprint: str | None = None,
                 **over) -> dict:
    """A valid verified profile for the Stage B server of pins (default: the repo's pins)."""
    pins = pins if pins is not None else bp.load_pins()
    doc = {"schema_version": 1, "profile_id": "verified-formal-20261101-c1", "status": "verified",
           "provisional_reason": None, "slo": {"id": "S2", "terms": S2},   # capacity.toml's [slo] (S2)
           "server": {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
                      "model_revision": pins["core_b"]["revision"],
                      "fingerprint_sha256": fingerprint or stage_b_fingerprint(pins),
                      "num_gpu_blocks": pins["stage_b"]["num_gpu_blocks"], "block_size": pins["stage_b"]["block_size"]},
           "classes": classes if classes is not None else [verified_class()],
           "provenance": {"source_session": "formal-20261101-c1",
                          "analysis_path": "results/capacity/formal-20261101-c1/analysis.json",
                          "analysis_sha256": HEX, "prereg_section": "§8", "freeze_commit": "1" * 40,
                          "generator_commit": "2" * 40, "deployed_commit": "3" * 40,
                          "capacity_sha256": CAPACITY_SHA256, "pins_sha256": PINS_SHA256,
                          "gpu": "NVIDIA GeForce RTX 5070 Ti", "gpu_driver": "617.42"},
           **over}
    return rehash(copy.deepcopy(doc))   # never shares S2 or a caller's classes with the document


# ------------------------------------------------------------------ Stage D (decision 043, §9)

# workload -> (class name, max_prompt_tokens, output tokens, server): 043 §2 (M8K-T: the 007 mix's long requests)
STAGE_D_CLASSES = {"U32K": ("uniform-32k", 32640, 128, "B0"), "U8K-T": ("uniform-8k-t", 8064, 128, "T"),
                   "M8K-T": ("mixed-8k-t", 32640, 128, "T"), "U2K-O512": ("uniform-2k-o512", 1920, 512, "B0"),
                   "U8K-O512": ("uniform-8k-o512", 8064, 512, "B0")}
HYPOTHESES = {"U32K": {"H-D1": "supported"}, "U8K-T": {"H-D2b": "supported"}, "M8K-T": {"H-D2a": "supported"},
              "U2K-O512": {"H-D3": "not supported"}, "U8K-O512": {"H-D3": "supported"}}   # 042's verdicts
NOT_COVERED = ["prompt_length_variation", "via_gateway", "open_loop", "output_length_other_than_measured",
               "start_up_wave", "concurrent_classes"]                                      # 043 §5: every D-A class
ABOVE_CKV_NOT_COVERED = "in_vllm_wait_beyond_c_kv"                                         # 043 §5: ABOVE_CKV adds it


def t_fingerprint(pins: dict) -> str:
    """Stage D's T server, by hand: B0's argv on the product port plus --long-prefill-token-threshold <pins
    [advanced]> (Advanced's condition T), with B0's recorded env."""
    sb = pins["stage_b"]
    argv, env = bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                               gpu_memory_utilization=sb["gpu_memory_utilization"], port=8100,
                               kv_cache_memory_bytes=sb["kv_cache_memory_bytes"])
    argv = argv + ["--long-prefill-token-threshold", str(pins["advanced"]["long_prefill_token_threshold"])]
    return server_fingerprint(argv, bp.recorded_env(env))


def stage_d_class(workload="U32K", status="EXACT", value=2, *, flags=(), hypotheses=None) -> dict:
    """A class of a Stage D profile from its D-A workload's C_PROD (030 8.7 statuses; source stage_d:...)."""
    name, prompt, out, server = STAGE_D_CLASSES[workload]
    unavailable = status in ("UNSTABLE", "NT")
    limit = None if unavailable else (0 if status == "INFEASIBLE" else value)
    numbered = bool(limit)
    est = {"value": limit, "status": status}
    ev = {"workload": workload, "scope": f"test scope, exactly {out} output tokens (ignore-eos)", "c_prod": dict(est),
          "c_slo": dict(est), "c_kv": 3, "c_cap": 6, "flags": list(flags),
          "binding": ["tpot"] if status != "NT" else None, "margins": {"short": 0.5, "tpot": 0.1} if numbered else None,
          "burst_max_ttft_ms": 8000.0 if numbered else None, "server": server, "output_tokens": out,
          "descriptive": {"c_full": dict(est), "c_late": dict(est)},
          "at_c_prod": {"attainment_min": 1.0, "preemptions_max": 0.0, "kv_cache_usage_peak_max": 0.5}
          if numbered else None,
          "hypotheses": HYPOTHESES[workload] if hypotheses is None else hypotheses,
          "not_covered": NOT_COVERED + ([ABOVE_CKV_NOT_COVERED] if "ABOVE_CKV" in flags else [])}
    return {"name": name, "max_prompt_tokens": prompt, "max_output_tokens": out, "max_concurrency": limit,
            "max_queue": limit, "source": f"stage_d:{workload}:c_prod:{status}",
            "status": "unavailable" if unavailable else "verified", "bound": None if unavailable else BOUND[status],
            "unavailable_reason": f"C_PROD is {status}: no verified admission limit (test)" if unavailable else None,
            "evidence": ev}


def stage_d_doc(pins: dict | None = None, *, server: str = "B0", classes: list | None = None,
                session: str = "formal-20261101-d-a1", **over) -> dict:
    """A valid §9 verified profile of one Stage D session's B0 or T server (default: U32K on B0, M8K-T on T)."""
    pins = pins if pins is not None else bp.load_pins()
    t = server == "T"
    if classes is None:
        classes = [stage_d_class("M8K-T", value=4)] if t else [stage_d_class("U32K", value=2)]
    doc = {"schema_version": 1, "profile_id": f"verified-{session}-{'t' if t else 'b0'}", "status": "verified",
           "provisional_reason": None, "slo": {"id": "S2", "terms": S2},   # stage_d.toml's [slo] (S2)
           "server": {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
                      "model_revision": pins["core_b"]["revision"],
                      "fingerprint_sha256": t_fingerprint(pins) if t else stage_b_fingerprint(pins),
                      "num_gpu_blocks": pins["stage_b"]["num_gpu_blocks"], "block_size": pins["stage_b"]["block_size"],
                      "long_prefill_token_threshold": pins["advanced"]["long_prefill_token_threshold"] if t else None},
           "classes": classes,
           "provenance": {"source_session": session, "analysis_path": f"results/stage_d/{session}/analysis.json",
                          "analysis_sha256": HEX, "prereg_section": "§9", "freeze_commit": "1" * 40,
                          "generator_commit": "2" * 40, "deployed_commit": "3" * 40,
                          "capacity_sha256": CAPACITY_SHA256, "pins_sha256": PINS_SHA256,
                          "stage_d_sha256": STAGE_D_SHA256, "gpu": "NVIDIA GeForce RTX 5070 Ti",
                          "gpu_driver": "617.42"},
           **over}
    return rehash(copy.deepcopy(doc))


# ------------------------------------------------------------------ Stage V (decision 044, §12)

SLOW_QUEUE = {"max_queue_wait_s": 120.0, "request_deadline_s": 600.0}           # 044 V2 / V4: "慢慢服務"
PACING = {"U2K": {"burst_tokens": 9600, "rate_tokens_per_s": 5920},              # 044 V2
          "U8K": {"burst_tokens": 32256, "rate_tokens_per_s": 5800}}
V1_NOT_COVERED = ["prompt_length_variation", "open_loop", "start_up_wave", "concurrent_classes"]   # + ABOVE_CKV's
V1_CLASSES = {  # base D-A workload -> (name, validated C, unit, points, the study classes they served)
    "U2K-O512": ("uniform-2k-o1-512", 32, "V1-2K", ["L-U2K", "GW-U2K-O512"], ["v-uniform-2k-o512"]),
    "U8K-O512": ("uniform-8k-o1-512", 8, "V1-8K", ["L-U8K", "GW-U8K-O512"],
                 ["v-uniform-8k-o512-c8", "v-uniform-8k-o512"])}


def validation(study, unit, points, measured, *, n=5) -> dict:
    return {"study": study, "unit": unit, "points": list(points), "measured_classes": list(measured),
            "worst_r": {p: {"client": 0.81, "upstream": 0.74} for p in points}, "n": n}


def stage_v_copy(workload="U2K", value=None, *, pacing=True) -> dict:
    """A §12 copy of c1b's uniform-2k (32) or uniform-8k (8): its §8 class, with the pacing PACE-2K / PACE-8K
    validated (V2), or plain (only a member of the admission section)."""
    tier = {"U2K": "2K", "U8K": "8K"}[workload]
    cls = verified_class(workload=workload, value=value or {"U2K": 32, "U8K": 8}[workload])
    if pacing:
        cls["pacing"] = {**PACING[workload], "queue": dict(SLOW_QUEUE)}
        cls["validation"] = validation("V2", f"PACE-{tier}", [f"P-{workload}"], [f"v-{cls['name']}"])
    return cls


def v1_class(base="U2K-O512") -> dict:
    """V1's validated class of a D-A O512 workload: its limits and evidence, the validated C, not_covered recomputed
    (via_gateway and output_length_other_than_measured covered, output_length_distribution added)."""
    name, c, unit, points, measured = V1_CLASSES[base]
    flags = ["WINDOW_SENSITIVE", "ABOVE_CKV"] if base == "U2K-O512" else []
    cls = stage_d_class(base, value={"U2K-O512": 32, "U8K-O512": 12}[base], flags=flags)
    ev = cls["evidence"]
    ev["not_covered"] = V1_NOT_COVERED + ([ABOVE_CKV_NOT_COVERED] if "ABOVE_CKV" in flags else []) + [
        "output_length_distribution"]
    ev.update(output_distribution="uniform[1,512]", datasets_sha256=[f"{i:02x}" * 32 for i in range(1, 6)],
              via_gateway=True)
    cls.update(name=name, max_concurrency=c, max_queue=c, bound="validated", source=f"stage_v:{unit}",
               validation=validation("V1", unit, points, measured))
    return cls


def admission_section(**over) -> dict:
    """W1's weighted-v1 section (044 V4) over the copies uniform-2k and uniform-8k."""
    return {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
            "classes": [{"name": "uniform-2k", "units": 1, "max_concurrency": 32},
                        {"name": "uniform-8k", "units": 4, "max_concurrency": 8}],
            "queue": dict(SLOW_QUEUE), "measured_mixes": [0.9, 0.5], "gateway_path_sha256": GATEWAY_PATH_SHA256,
            "validation": validation("V4", "W1", ["W-90", "W-50"], ["v-uniform-2k", "v-uniform-8k"]), **over}


def stage_v_doc(pins: dict | None = None, *, classes: list | None = None, admission: dict | None = "default",
                session: str = "formal-20261101-v-adm", **over) -> dict:
    """A valid §12 verified profile of one Stage V session (default: V-ADM's PACE-2K, PACE-8K and W1)."""
    pins = pins if pins is not None else bp.load_pins()
    doc = {"schema_version": 1, "profile_id": f"verified-{session}-b0", "status": "verified",
           "provisional_reason": None, "slo": {"id": "S2", "terms": S2},   # stage_v.toml's [slo] (S2)
           "server": {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
                      "model_revision": pins["core_b"]["revision"], "fingerprint_sha256": stage_b_fingerprint(pins),
                      "num_gpu_blocks": pins["stage_b"]["num_gpu_blocks"], "block_size": pins["stage_b"]["block_size"],
                      "long_prefill_token_threshold": None},
           "classes": [stage_v_copy("U2K"), stage_v_copy("U8K")] if classes is None else classes,
           "admission": admission_section() if admission == "default" else admission,
           "provenance": {"source_session": session, "analysis_path": f"results/stage_v/{session}/analysis.json",
                          "analysis_sha256": HEX, "prereg_section": "§12", "freeze_commit": "1" * 40,
                          "generator_commit": "2" * 40, "deployed_commit": "3" * 40,
                          "capacity_sha256": CAPACITY_SHA256, "pins_sha256": PINS_SHA256,
                          "stage_d_sha256": STAGE_D_SHA256, "stage_v_sha256": STAGE_V_SHA256,
                          "gateway_path_sha256": GATEWAY_PATH_SHA256, "capserve_sha256": CAPSERVE_SHA256,
                          "base_profiles": [{"id": "verified-formal-20261017-c1b", "content_sha256": "e5" * 32}],
                          "gpu": "NVIDIA GeForce RTX 5070 Ti", "gpu_driver": "617.42"},
           **over}
    return rehash(copy.deepcopy(doc))
