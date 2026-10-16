"""Capacity profile (decision 029 "Capacity profile"; 027 §4): schema, canonical hash, server fingerprint, loader.

- A profile is one JSON document validated by strict pydantic models with extra="forbid": unknown keys and wrong
  types are errors, never ignored or coerced. Every top-level key is required (null where it does not apply).
- content_sha256 = sha256 of the canonical JSON (sorted keys, no whitespace, UTF-8) of the document without that
  key. It only catches accidental edits (anyone can recompute it); a verified profile is trusted because the generator
  reproduces it from the frozen analysis (M2).
- every profile: at least one class, unique class names, max_prompt_tokens + max_output_tokens <= Core B's
  --max-model-len (32768); vLLM version and model repo / revision equal to pins.
- verified: a §8 provenance with a formal- session, 64-hex analysis / capacity.toml / pins.toml hashes, full 40-hex
  git commits (freeze, generator, the session's DEPLOYED_COMMIT), the GPU, an SLO (its terms copied verbatim), the KV
  block geometry the deploy step checks, and a null provisional_reason. Every class states its status from the
  Stage C C_PROD of its workload (decision 030 8.7), with that C_PROD and its descriptive evidence (C_SLO, C_KV,
  flags, binding terms, margins, start-up burst):
  - "verified" with a number: C_PROD EXACT / LOWER / CENSORED, max_concurrency = C_PROD (1..128), bound "exact" /
    "lower" / "censored";
  - "verified" 0 = refuse the class: only from an INFEASIBLE C_PROD, with bound "exact" and max_queue 0; the
    gateway starts and answers every completion request 503 class_not_serviceable;
  - "unavailable": C_PROD UNSTABLE / NT, a non-empty unavailable_reason and no number (max_concurrency, max_queue,
    bound null); never served (serving_class refuses it), whatever flag is given, and never a fallback number.
  source is "stage_c:<workload>:c_prod:<status>".
- verified, checked at load time against this checkout (029 M2; what the generator writes, so a hand-written profile
  that breaks it never loads): configs/capacity.toml has [freeze] capacity = true (no verified capacity exists before
  the freeze) and the provenance's capacity_sha256 / pins_sha256 are the sha256 of this checkout's configs/capacity.toml
  / pins.toml; the slo is capacity.toml's [slo] verbatim; profile_id is verified-<source_session> and analysis_path
  results/capacity/<source_session>/analysis.json; every class is its Stage C workload's (030 8.7: one class per
  workload, named uniform-2k / uniform-8k / mixed-2k / mixed-8k, prompts of L - 128 tokens for U2K / U8K and of the
  32,640-token long requests for M2K / M8K, 128 output tokens). The numbers themselves are trusted because a verified
  profile is served only as a committed service/profiles/<name>.json, which the Mac test regenerates from the frozen
  analysis (029; capserve.deploy and `capserve serve` both enforce the location).
- provisional: a non-empty provisional_reason; loaded only when explicitly allowed (--allow-provisional or
  CAPSERVE_ALLOW_PROVISIONAL=1). Its limits are for development and integration tests, never product limits. Its
  classes carry no status / bound / unavailable_reason / evidence (the fields are optional, so every provisional
  profile of M0 / M1 loads unchanged) and need 1 <= max_concurrency <= Core B's --max-num-seqs (128), max_queue >= 0.

The server fingerprint is sha256 over the vLLM argv without argv[0] (the machine's binary path) and without
--host / --port, together with core_b_plan.recorded_env; the deploy step (M1) refuses a verified profile whose
fingerprint differs from the server it starts.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

import capacity_plan as cplan
import core_b_plan as bp

SCHEMA_VERSION = 1
PRODUCT_VLLM_PORT = 8100   # 029: product vLLM port, distinct from research's 8000 (dropped from the fingerprint anyway)
PREREG_SECTION = "§8"      # Stage C (027 §2)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_DROPPED_FLAGS = ("--host", "--port")
C_PROD_STATUSES = ("EXACT", "LOWER", "CENSORED", "INFEASIBLE", "UNSTABLE", "NT")   # capacity_analysis.STATUSES
BOUNDS = {"EXACT": "exact", "LOWER": "lower", "CENSORED": "censored"}               # 030 8.7: a verified number
REFUSE_STATUS = "INFEASIBLE"                                                        # 030 8.7: verified 0 = refuse
UNAVAILABLE_STATUSES = ("UNSTABLE", "NT")                                           # 030 8.7: no number, no fallback
CLASS_NAMES = {"U2K": "uniform-2k", "U8K": "uniform-8k", "M2K": "mixed-2k", "M8K": "mixed-8k"}  # 030 8.7 classes
CONFIGS = "configs"


class ProfileError(ValueError):
    pass


class FrozenConfigs(NamedTuple):
    """What a verified profile is checked against at load time: configs/capacity.toml ({} without one: not frozen),
    its sha256 (None without one) and the sha256 of configs/pins.toml."""
    capacity: dict
    capacity_sha256: str | None
    pins_sha256: str


def frozen_configs(root=cplan.ROOT) -> FrozenConfigs:
    """The FrozenConfigs of the checkout at root (default: the one these modules run from, e.g. a release)."""
    configs = Path(root) / CONFIGS
    try:
        pins_raw = (configs / "pins.toml").read_bytes()
        cap_path = configs / "capacity.toml"
        if not cap_path.exists():
            return FrozenConfigs({}, None, hashlib.sha256(pins_raw).hexdigest())
        cap_raw = cap_path.read_bytes()
        return FrozenConfigs(tomllib.loads(cap_raw.decode()), hashlib.sha256(cap_raw).hexdigest(),
                             hashlib.sha256(pins_raw).hexdigest())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"cannot read {configs}/capacity.toml or pins.toml: {exc}") from exc


def stage_c_class(workload: str) -> tuple:
    """(name, max_prompt_tokens, max_output_tokens) of a Stage C workload's class (030 8.7): uniform L takes prompts
    of L - 128 tokens, mixed L the 007 mix's 32,640-token long requests; 128 output tokens."""
    L, mix = cplan.WORKLOAD_SPEC[workload]
    return CLASS_NAMES[workload], bp.LONG_INPUT if mix == "mixed" else L - bp.N_GEN, bp.N_GEN


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Slo(_Strict):
    id: str
    terms: dict[str, int | float | str | dict[str, int | float | str]]   # copied verbatim from the frozen §8


class ServerSpec(_Strict):
    vllm_version: str
    model_repo: str
    model_revision: str
    fingerprint_sha256: str | None
    num_gpu_blocks: int | None
    block_size: int | None


class Estimand(_Strict):
    value: int | None
    status: Literal[C_PROD_STATUSES]


class ClassEvidence(_Strict):
    """What a verified profile states for each class besides its limit (030 8.7); descriptive, never a limit."""
    workload: str                          # the Stage C workload (U2K, U8K, M2K, M8K)
    scope: str                             # where the number holds (8.7: prompts, outputs, steady state, admitted)
    c_prod: Estimand                       # the only estimand a profile may turn into a limit (8.1)
    c_slo: Estimand
    c_kv: int
    c_cap: int
    flags: list[str]
    binding: list[str] | None              # the terms above 1 at K's smallest FAIL point
    margins: dict[str, float] | None       # 1 - worst R at C_PROD, per term
    burst_max_ttft_ms: float | None        # the start-up wave (idx < C) at C_PROD, worst valid run


class WorkloadClass(_Strict):
    name: str
    max_prompt_tokens: int
    max_output_tokens: int
    max_concurrency: int | None            # None only for an unavailable class
    max_queue: int | None
    source: str
    status: Literal["verified", "unavailable"] | None = None    # verified profiles only (030 8.7)
    bound: Literal["exact", "lower", "censored"] | None = None
    unavailable_reason: str | None = None
    evidence: ClassEvidence | None = None

    @property
    def serviceable(self) -> bool:
        """Whether anything may be admitted: False for an unavailable class and for a verified 0 (refuse)."""
        return self.status != "unavailable" and (self.max_concurrency or 0) >= 1


class Provenance(_Strict):
    source_session: str
    analysis_path: str
    analysis_sha256: str
    prereg_section: str
    freeze_commit: str        # the commit that set configs/capacity.toml [freeze] capacity = true
    generator_commit: str
    deployed_commit: str      # the session's DEPLOYED_COMMIT
    capacity_sha256: str
    pins_sha256: str
    gpu: str
    gpu_driver: str


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

    def serving_class(self, name: str) -> WorkloadClass:
        """The class a gateway may serve as its active_class: an unavailable class is refused, whatever flag is given
        (030 8.7: no verified number, never a fallback); a verified 0 is returned (not serviceable: refuse it)."""
        c = self.workload_class(name)
        if c.status == "unavailable":
            raise ProfileError(f"profile {self.profile_id} class {name!r} is unavailable ({c.unavailable_reason}): "
                               "Stage C verified no admission limit for it, so it is never served, whatever flag is "
                               "given (030 8.7)")
        return c


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
        problems += _status_problems(c) if prof.verified else _provisional_class_problems(c)
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


def _limit_problems(c: WorkloadClass) -> list:
    problems = []
    if not (isinstance(c.max_concurrency, int) and 1 <= c.max_concurrency <= bp.MAX_NUM_SEQS):
        problems.append(f"class {c.name}: max_concurrency {c.max_concurrency!r} outside 1..{bp.MAX_NUM_SEQS}")
    if not (isinstance(c.max_queue, int) and c.max_queue >= 0):
        problems.append(f"class {c.name}: max_queue {c.max_queue!r} is not an integer >= 0")
    return problems


def _provisional_class_problems(c: WorkloadClass) -> list:
    extra = [k for k in ("status", "bound", "unavailable_reason", "evidence") if getattr(c, k) is not None]
    problems = [f"class {c.name}: {extra} belong to a verified profile's classes; a provisional profile's classes "
                "carry none (030 8.7)"] if extra else []
    return problems + _limit_problems(c)


def stage_c_source(workload: str, c_prod_status: str) -> str:
    """A verified profile's class source: the Stage C workload and the status of its C_PROD (030 8.7)."""
    return f"stage_c:{workload}:c_prod:{c_prod_status}"


def _status_problems(c: WorkloadClass) -> list:
    """030 8.7 for one class of a verified profile."""
    where = f"class {c.name}"
    if c.status is None:
        return [f"{where}: a verified profile states every class's status (verified | unavailable; 030 8.7)"]
    if c.evidence is None:
        return [f"{where}: a verified profile's class needs its Stage C evidence (C_PROD, C_SLO, C_KV, flags; 8.7)"]
    est, problems = c.evidence.c_prod, []
    if c.source != stage_c_source(c.evidence.workload, est.status):
        problems.append(f"{where}: source {c.source!r} != {stage_c_source(c.evidence.workload, est.status)!r}")
    if c.status == "unavailable":
        if not (c.unavailable_reason or "").strip():
            problems.append(f"{where}: an unavailable class needs an unavailable_reason")
        if (c.max_concurrency, c.max_queue, c.bound) != (None, None, None):
            problems.append(f"{where}: an unavailable class has no number (max_concurrency, max_queue and bound are "
                            "null; 030 8.7: never a number, never a fallback)")
        if est.status not in UNAVAILABLE_STATUSES or est.value is not None:
            problems.append(f"{where}: unavailable needs a C_PROD of UNSTABLE or NT without a value, not "
                            f"{est.status} {est.value!r}")
        return problems
    if c.unavailable_reason is not None:
        problems.append(f"{where}: a verified class has unavailable_reason null")
    if c.max_concurrency == 0:
        if c.bound != "exact" or est.status != REFUSE_STATUS or est.value != 0:
            problems.append(f"{where}: a verified 0 (refuse this class) needs bound exact and an {REFUSE_STATUS} "
                            f"C_PROD of value 0, not bound {c.bound!r} and C_PROD {est.status} {est.value!r} (8.7)")
        if c.max_queue != 0:
            problems.append(f"{where}: a verified 0 admits nothing, so max_queue is 0, not {c.max_queue!r}")
        return problems
    problems += _limit_problems(c)
    if est.status not in BOUNDS or est.value != c.max_concurrency:
        problems.append(f"{where}: max_concurrency {c.max_concurrency!r} is not the C_PROD {est.status} {est.value!r} "
                        f"(a verified number is an EXACT / LOWER / CENSORED C_PROD; 8.7)")
    elif c.bound != BOUNDS[est.status]:
        problems.append(f"{where}: bound {c.bound!r} != {BOUNDS[est.status]!r} for a C_PROD {est.status}")
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
    for key in ("analysis_sha256", "capacity_sha256", "pins_sha256"):
        if not _HEX64.fullmatch(getattr(p, key)):
            problems.append(f"provenance.{key} must be 64 lowercase hex")
    if p.prereg_section != PREREG_SECTION:
        problems.append(f"provenance.prereg_section {p.prereg_section!r} != {PREREG_SECTION!r}")
    for key in ("freeze_commit", "generator_commit", "deployed_commit"):
        if not _HEX40.fullmatch(getattr(p, key)):
            problems.append(f"provenance.{key} must be a full git commit (40 lowercase hex)")
    if not (p.gpu.strip() and p.gpu_driver.strip()):
        problems.append("provenance.gpu and provenance.gpu_driver name the GPU the session ran on")
    return problems


def provenance_problems(doc: dict, prof: CapacityProfile, frozen: FrozenConfigs) -> list:
    """Every rule a verified profile breaks against this checkout's frozen configs and the Stage C classes (029 M2)."""
    problems = []
    if (frozen.capacity.get("freeze") or {}).get("capacity") is not True:
        problems.append("configs/capacity.toml has no [freeze] capacity = true: no verified capacity exists before "
                        "Stage C's freeze (029, 030), so this profile does not come from the generator")
    slo = (frozen.capacity or {}).get("slo")
    if doc.get("slo") is not None and slo != {"id": doc["slo"].get("id"), **(doc["slo"].get("terms") or {})}:
        problems.append(f"slo {doc['slo'].get('id')!r} is not configs/capacity.toml [slo] "
                        f"{(slo or {}).get('id')!r} verbatim (030 8.7)")
    p = prof.provenance
    if p is not None:
        if prof.profile_id != f"verified-{p.source_session}":
            problems.append(f"profile_id {prof.profile_id!r} != 'verified-{p.source_session}' (its source_session)")
        want = f"results/capacity/{p.source_session}/analysis.json"
        if p.analysis_path != want:
            problems.append(f"provenance.analysis_path {p.analysis_path!r} != {want!r} (the session's analysis.json)")
        for key, name in (("capacity_sha256", "capacity.toml"), ("pins_sha256", "pins.toml")):
            if getattr(p, key) != getattr(frozen, key):
                problems.append(f"provenance.{key} {getattr(p, key)} is not this checkout's configs/{name} "
                                f"{getattr(frozen, key)}")
    workloads = [c.evidence.workload for c in prof.classes if c.evidence is not None]
    dups = sorted({w for w in workloads if workloads.count(w) > 1})
    if dups:
        problems.append(f"workloads {dups} have more than one class (030 8.7: one class per Stage C workload)")
    for c in prof.classes:
        if c.evidence is None:
            continue
        w = c.evidence.workload
        if w not in CLASS_NAMES:
            problems.append(f"class {c.name}: evidence.workload {w!r} is not a Stage C workload {list(CLASS_NAMES)}")
            continue
        want = dict(zip(("name", "max_prompt_tokens", "max_output_tokens"), stage_c_class(w)))
        got = {k: getattr(c, k) for k in want}
        if got != want:
            problems.append(f"class {c.name}: {got} is not workload {w}'s class {want} (030 8.7)")
    return problems


def load_profile(path, *, allow_provisional: bool, pins: dict | None = None,
                 frozen: FrozenConfigs | None = None) -> CapacityProfile:
    """The profile at path, or ProfileError with every problem. pins default to configs/pins.toml, frozen (read only
    for a verified profile) to frozen_configs()."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ProfileError(f"{path}: cannot read the profile: {exc}") from exc
    return parse_profile(data, source=path, allow_provisional=allow_provisional, pins=pins, frozen=frozen)


def parse_profile(data, *, source, allow_provisional: bool, pins: dict | None = None,
                  frozen: FrozenConfigs | None = None) -> CapacityProfile:
    """The profile in data (a profile file's bytes or text; source names it in errors), or ProfileError. The deploy
    step parses the very bytes it hands to the gateway, so what was checked is what is served."""
    try:
        doc = json.loads(data)
    except ValueError as exc:
        raise ProfileError(f"{source}: cannot read the profile: {exc}") from exc
    try:
        prof = CapacityProfile.model_validate(doc)
    except ValidationError as exc:
        raise ProfileError(f"{source}: {exc}") from exc
    problems = profile_problems(doc, prof, pins if pins is not None else bp.load_pins())
    if prof.verified:
        try:
            problems += provenance_problems(doc, prof, frozen if frozen is not None else frozen_configs())
        except ProfileError as exc:
            problems.append(str(exc))
    if problems:
        raise ProfileError(f"{source}: " + "; ".join(problems))
    if not prof.verified and not allow_provisional:
        raise ProfileError(f"{source}: profile {prof.profile_id} is provisional ({prof.provisional_reason}); it is "
                           "refused without --allow-provisional or CAPSERVE_ALLOW_PROVISIONAL=1 (027 §4)")
    return prof
