"""Schema-faithful capacity profile documents for tests (decision 029 "Capacity profile"; the per-class statuses of
decision 030 8.7). A verified class is built here by hand from a C_PROD status, independently of the generator:
EXACT / LOWER / CENSORED -> verified with that number and bound, INFEASIBLE -> verified 0 (refuse the class),
UNSTABLE / NT -> unavailable with a reason and no number. Its name and token limits are its Stage C workload's
(030 8.7: U2K / U8K take L - 128 prompt tokens, the mixed M2K / M8K the 32,640-token long requests; 128 output).

A verified profile is checked at load time against the frozen configs (029 M2): FROZEN is the repo's capacity.toml
with [freeze] capacity = true, and the capacity.toml / pins.toml sha256 verified_doc's provenance states."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import core_b_plan as bp
from capserve.profile import FrozenConfigs, content_sha256, stage_b_fingerprint

HEX = "ab" * 32
CAPACITY_SHA256, PINS_SHA256 = "cd" * 32, "ef" * 32
S2 = {"ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100, "mixed_long": "budget"}
BOUND = {"EXACT": "exact", "LOWER": "lower", "CENSORED": "censored", "INFEASIBLE": "exact"}
CLASSES = {"U2K": ("uniform-2k", 1920), "U8K": ("uniform-8k", 8064), "M2K": ("mixed-2k", 32640),
           "M8K": ("mixed-8k", 32640)}


def frozen_capacity(**freeze) -> dict:
    """The repo's configs/capacity.toml after the freeze (or with the given [freeze] values)."""
    cfg = tomllib.loads((Path(bp.__file__).resolve().parent.parent / "configs" / "capacity.toml").read_text())
    cfg["freeze"].update({"capacity": True, **freeze})
    return cfg


def frozen_configs(capacity: dict | None = None, *, capacity_sha256=CAPACITY_SHA256,
                   pins_sha256=PINS_SHA256) -> FrozenConfigs:
    """What a verified profile is checked against at load time (default: the frozen repo capacity.toml)."""
    return FrozenConfigs(frozen_capacity() if capacity is None else capacity, capacity_sha256, pins_sha256)


FROZEN = frozen_configs()


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
