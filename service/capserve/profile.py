"""Capacity profile (decision 029 "Capacity profile"; 027 §4): schema, canonical hash, server fingerprint, loader.

- A profile is one JSON document validated by strict pydantic models with extra="forbid": unknown keys and wrong
  types are errors, never ignored or coerced. Every top-level key is required (null where it does not apply).
- content_sha256 = sha256 of the canonical JSON (sorted keys, no whitespace, UTF-8) of the document without that
  key. It only catches accidental edits (anyone can recompute it); a verified profile is trusted because the generator
  reproduces it from the frozen analysis (M2).
- every profile: at least one class, unique class names, max_prompt_tokens + max_output_tokens <= Core B's
  --max-model-len (32768), 1 <= max_concurrency <= Core B's --max-num-seqs (128), max_queue >= 0; vLLM version and
  model repo / revision equal to pins.
- verified: a §8 provenance with a formal- session, 64-hex analysis / fingerprint hashes, git commits, an SLO, the
  KV block geometry the deploy step checks, and a null provisional_reason.
- provisional: a non-empty provisional_reason; loaded only when explicitly allowed (--allow-provisional or
  CAPSERVE_ALLOW_PROVISIONAL=1). Its limits are for development and integration tests, never product limits.

The server fingerprint is sha256 over the vLLM argv without argv[0] (the machine's binary path) and without
--host / --port, together with core_b_plan.recorded_env; the deploy step (M1) refuses a verified profile whose
fingerprint differs from the server it starts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

import core_b_plan as bp

SCHEMA_VERSION = 1
PRODUCT_VLLM_PORT = 8100   # 029: product vLLM port, distinct from research's 8000 (dropped from the fingerprint anyway)
PREREG_SECTION = "§8"      # Stage C (027 §2)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{7,40}")
_DROPPED_FLAGS = ("--host", "--port")


class ProfileError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Slo(_Strict):
    id: str
    terms: dict[str, int | float | str]   # copied verbatim from the frozen §8


class ServerSpec(_Strict):
    vllm_version: str
    model_repo: str
    model_revision: str
    fingerprint_sha256: str | None
    num_gpu_blocks: int | None
    block_size: int | None


class WorkloadClass(_Strict):
    name: str
    max_prompt_tokens: int
    max_output_tokens: int
    max_concurrency: int
    max_queue: int
    source: str


class Provenance(_Strict):
    source_session: str
    analysis_path: str
    analysis_sha256: str
    prereg_section: str
    freeze_commit: str
    generator_commit: str


class CapacityProfile(_Strict):
    schema_version: Literal[1]
    profile_id: str
    status: Literal["verified", "provisional"]
    provisional_reason: str | None
    slo: Slo | None
    server: ServerSpec
    classes: list[WorkloadClass]
    provenance: Provenance | None
    content_sha256: str

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def workload_class(self, name: str) -> WorkloadClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise ProfileError(f"profile {self.profile_id} has no class {name!r} (classes: {[c.name for c in self.classes]})")


# ---------------------------------------------------------------- hashing

def canonical_json(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(doc) -> str:
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def content_sha256(doc: dict) -> str:
    return canonical_sha256({k: v for k, v in doc.items() if k != "content_sha256"})


def server_fingerprint(argv: Sequence[str], recorded_env: dict) -> str:
    """sha256 of argv[1:] minus every --host / --port (with its value, or as --flag=value) plus the recorded env."""
    kept, skip = [], False
    for arg in argv[1:]:
        if skip:
            skip = False
        elif arg in _DROPPED_FLAGS:
            skip = True
        elif not arg.startswith(tuple(f"{f}=" for f in _DROPPED_FLAGS)):
            kept.append(arg)
    return canonical_sha256({"argv": kept, "env": recorded_env})


def stage_b_fingerprint(pins: dict) -> str:
    """The fingerprint of the Core B server at the Stage B values in pins (core_b_plan.server_argv and
    recorded_env, read-only): the server a verified profile was measured on, served on the product port."""
    sb = pins["stage_b"]
    argv, env = bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                               gpu_memory_utilization=sb["gpu_memory_utilization"], port=PRODUCT_VLLM_PORT,
                               kv_cache_memory_bytes=sb["kv_cache_memory_bytes"])
    return server_fingerprint(argv, bp.recorded_env(env))


# ---------------------------------------------------------------- rules

def profile_problems(doc: dict, prof: CapacityProfile, pins: dict) -> list:
    """Every 029 loader rule the profile breaks (empty: valid). doc is the raw JSON the hash is computed over."""
    problems = []
    if doc.get("content_sha256") != content_sha256(doc):
        problems.append("content_sha256 does not match the canonical content (edited after generation?)")
    names = [c.name for c in prof.classes]
    if not names:
        problems.append("classes is empty")
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        problems.append(f"duplicate class names {dups}")
    for c in prof.classes:
        if c.max_prompt_tokens < 1 or c.max_output_tokens < 1:
            problems.append(f"class {c.name}: max_prompt_tokens and max_output_tokens must be >= 1")
        if c.max_prompt_tokens + c.max_output_tokens > bp.MAX_MODEL_LEN:
            problems.append(f"class {c.name}: max_prompt_tokens + max_output_tokens > {bp.MAX_MODEL_LEN}")
        if not 1 <= c.max_concurrency <= bp.MAX_NUM_SEQS:
            problems.append(f"class {c.name}: max_concurrency outside 1..{bp.MAX_NUM_SEQS}")
        if c.max_queue < 0:
            problems.append(f"class {c.name}: max_queue < 0")
    want = {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
            "model_revision": pins["core_b"]["revision"]}
    for key, value in want.items():
        if getattr(prof.server, key) != value:
            problems.append(f"server.{key} {getattr(prof.server, key)!r} != pins {value!r}")
    if prof.verified:
        problems += _verified_problems(prof)
    elif not (prof.provisional_reason or "").strip():
        problems.append("a provisional profile needs a provisional_reason")
    return problems


def _verified_problems(prof: CapacityProfile) -> list:
    problems = []
    if prof.provisional_reason is not None:
        problems.append("a verified profile has provisional_reason null")
    if prof.slo is None:
        problems.append("a verified profile needs the frozen §8 slo")
    if not _HEX64.fullmatch(prof.server.fingerprint_sha256 or ""):
        problems.append("server.fingerprint_sha256 must be 64 lowercase hex")
    for key in ("num_gpu_blocks", "block_size"):
        if not (getattr(prof.server, key) or 0) >= 1:
            problems.append(f"server.{key} must be a positive integer")
    p = prof.provenance
    if p is None:
        return problems + ["a verified profile needs a provenance"]
    if not p.source_session.startswith("formal-"):
        problems.append(f"provenance.source_session {p.source_session!r} is not a formal- session")
    if not p.analysis_path:
        problems.append("provenance.analysis_path is empty")
    if not _HEX64.fullmatch(p.analysis_sha256):
        problems.append("provenance.analysis_sha256 must be 64 lowercase hex")
    if p.prereg_section != PREREG_SECTION:
        problems.append(f"provenance.prereg_section {p.prereg_section!r} != {PREREG_SECTION!r}")
    for key in ("freeze_commit", "generator_commit"):
        if not _GIT_SHA.fullmatch(getattr(p, key)):
            problems.append(f"provenance.{key} must be a git commit (7-40 lowercase hex)")
    return problems


def load_profile(path, *, allow_provisional: bool, pins: dict | None = None) -> CapacityProfile:
    """The profile at path, or ProfileError with every problem. pins default to configs/pins.toml."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f"{path}: cannot read the profile: {exc}") from exc
    try:
        prof = CapacityProfile.model_validate(doc)
    except ValidationError as exc:
        raise ProfileError(f"{path}: {exc}") from exc
    problems = profile_problems(doc, prof, pins if pins is not None else bp.load_pins())
    if problems:
        raise ProfileError(f"{path}: " + "; ".join(problems))
    if not prof.verified and not allow_provisional:
        raise ProfileError(f"{path}: profile {prof.profile_id} is provisional ({prof.provisional_reason}); it is "
                           "refused without --allow-provisional or CAPSERVE_ALLOW_PROVISIONAL=1 (027 §4)")
    return prof
