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
- verified from Stage D (decision 043; prereg_section §9, the other value PREREG_SECTIONS allows): one profile per
  (session, server), the server B0 (server.long_prefill_token_threshold null or absent, the Stage B fingerprint) or T
  (the threshold of pins [advanced], t_fingerprint: stage_d_plan.server_argv_for, frozen, read-only). Besides the §8
  format rules (source "stage_d:<workload>:c_prod:<status>" instead), checked against this checkout:
  configs/stage_d.toml has [freeze] stage_d = true and capacity.toml [freeze] capacity = true;
  provenance.stage_d_sha256 / capacity_sha256 / pins_sha256 are this checkout's stage_d.toml / capacity.toml /
  pins.toml; the slo is stage_d.toml's [slo] verbatim; profile_id is verified-<source_session>-<b0|t> (the suffix
  from the threshold) and analysis_path results/stage_d/<source_session>/analysis.json; every class is a D-A
  workload's (stage_d_class: uniform-32k, uniform-8k-t, mixed-8k-t, uniform-2k-o512, uniform-8k-o512; never the
  sentinel's, D-B's or D-C's), at most one per workload, run on this profile's server (stage_d_plan.WORKLOADS and
  evidence.server), with evidence.output_tokens its measured output length and evidence.not_covered exactly
  not_covered_for(workload, flags) (a closed enum: a hand-edited list never loads). Its descriptive evidence
  (descriptive c_full / c_late, at_c_prod, hypotheses) is never a limit. A §8 profile has none of these §9 fields
  (all null or absent: c1b loads, hashes and regenerates unchanged).
- verified from Stage V (decision 044; prereg_section §12): one profile per Stage V session, B0 only, written by the
  generator for the units whose every point PASSed. Checked against this checkout: configs/stage_v.toml (committed
  with the harness before the freeze with [freeze] stage_v = false; until it is true, or without the file, every §12
  profile is refused: fail closed) has [freeze] stage_v = true, and stage_d.toml / capacity.toml their freezes;
  provenance.stage_v_sha256 / stage_d_sha256 / capacity_sha256 / pins_sha256 are this checkout's files and
  provenance.gateway_path_sha256 is stage_v.toml's [freeze] gateway_path_sha256 (the frozen request path the units
  were measured on; capserve_sha256 and base_profiles [{id, content_sha256}] are provenance only); the slo is
  stage_v.toml's [slo] verbatim; profile_id is verified-<source_session>-b0 and analysis_path
  results/stage_v/<source_session>/analysis.json. Its classes:
  - a copy of c1b's uniform-2k / uniform-8k (§8's class rules, its §8 evidence only), with the `pacing` PACE-2K /
    PACE-8K validated ({burst_tokens >= max_prompt_tokens, rate_tokens_per_s, queue}) and that `validation`, or plain
    (no pacing, no validation) as a member of the admission section;
  - V1's validated class of a D-A O512 workload (V1_CLASS_NAMES): that workload's name-mapped token limits, bound
    "validated" (a fixed point V1 validated, not a C_PROD: max_concurrency = max_queue = V1_C, the C of L-U2K / L-U8K,
    32 / 8, never the O512 C_PROD; source stage_v:<unit>), its evidence (server B0, output_tokens 512) with
    not_covered = v1_not_covered(workload, flags), output_distribution "uniform[1,512]", the five datasets' sha256 and
    via_gateway true, and V1's validation over exactly the unit's points (V1_POINTS);
  - at most one class per workload. The profile's `admission` section (W1, V4) names copies by name with weighted-v1's
    parameters (admission_problems), the queue settings it was validated with, measured_mixes in (0, 1), the same
    gateway_path_sha256 and V4's validation. Every number (limits, pacing, the section's parameters) is trusted as for
    §8 / §9: a committed file the Mac test regenerates. The §12 fields exist nowhere else (a §8, §9 or provisional
    profile has them null), so no candidate policy can be attached to another profile by hand.
- provisional: a non-empty provisional_reason; loaded only when explicitly allowed (--allow-provisional or
  CAPSERVE_ALLOW_PROVISIONAL=1). Its limits are for development and integration tests, never product limits. Its
  classes carry no status / bound / unavailable_reason / evidence (the fields are optional, so every provisional
  profile of M0 / M1 loads unchanged), never use a Stage C, Stage D or V1 class name (036, 043, 044), and need
  1 <= max_concurrency <= Core B's --max-num-seqs (128), max_queue >= 0.

serving_class (the active class of `capserve serve`, create_app and `deploy up`; a ProfileError no flag overrides)
refuses an unavailable class (030 8.7), a verified class with max_output_tokens > 128 (043 §4: measured at exactly that
many output tokens; shorter outputs, never measured, are not safer: U8K's C_PROD is 12 at 512 but C 9 FAILs at 128;
044: except a §12 profile's validated class, whose outputs V1 measured uniform on [1, 512]),
on a T profile every class but M8K-T with H-D2a supported (030 8.8: T only for a declared M8K deployment whose
C_PROD beats B0's; uniform-8k-t is report only), and on a B0 profile every class of a T workload
(stage_d_plan.WORKLOADS; 030 8.8: T's limit is never served on B0; the loader refuses such a profile too,
serving_class reads the model only).

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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import capacity_plan as cplan
import core_b_plan as bp
import stage_d_plan as sdplan

SCHEMA_VERSION = 1
PRODUCT_VLLM_PORT = 8100   # 029: product vLLM port, distinct from research's 8000 (dropped from the fingerprint anyway)
PREREG_SECTION = "§8"      # Stage C (027 §2)
STAGE_D_SECTION = "§9"     # Stage D (039; its profiles: 043)
STAGE_V_SECTION = "§12"    # Stage V (044): the validated copies, V1's classes, the admission section
PREREG_SECTIONS = (PREREG_SECTION, STAGE_D_SECTION, STAGE_V_SECTION)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_DROPPED_FLAGS = ("--host", "--port")
C_PROD_STATUSES = ("EXACT", "LOWER", "CENSORED", "INFEASIBLE", "UNSTABLE", "NT")   # capacity_analysis.STATUSES
BOUNDS = {"EXACT": "exact", "LOWER": "lower", "CENSORED": "censored"}               # 030 8.7: a verified number
REFUSE_STATUS = "INFEASIBLE"                                                        # 030 8.7: verified 0 = refuse
UNAVAILABLE_STATUSES = ("UNSTABLE", "NT")                                           # 030 8.7: no number, no fallback
CLASS_NAMES = {"U2K": "uniform-2k", "U8K": "uniform-8k", "M2K": "mixed-2k", "M8K": "mixed-8k"}  # 030 8.7 classes
# 043 §2: the D-A workloads' classes, named unlike any c1b class (a config selecting uniform-8k never gets T's 4 or
# O512's 12)
STAGE_D_CLASS_NAMES = {"U32K": "uniform-32k", "U8K-T": "uniform-8k-t", "M8K-T": "mixed-8k-t",
                       "U2K-O512": "uniform-2k-o512", "U8K-O512": "uniform-8k-o512"}
SERVERS = ("B0", "T")                       # stage_d_plan.WORKLOADS[w]["server"]
T_WORKLOAD = "U8K-T"                        # a T workload: stage_d_plan.server_argv_for reads only its server
T_SERVED = ("M8K-T", "H-D2a")               # 030 8.8: T serves only M8K, and only when its C_PROD beats B0's
# 043 §5: what no D-A class's number covers, and (ABOVE_CKV only) the wait inside vLLM beyond C_KV; 044: V1's classes
# drop the gateway and other output lengths (measured) and add the output distribution (only uniform[1, 512])
NOT_COVERED = ("prompt_length_variation", "via_gateway", "open_loop", "output_length_other_than_measured",
               "start_up_wave", "concurrent_classes", "in_vllm_wait_beyond_c_kv", "output_length_distribution")
ABOVE_CKV_NOT_COVERED = "in_vllm_wait_beyond_c_kv"
V1_NOT_COVERED = "output_length_distribution"
V1_COVERED = ("via_gateway", "output_length_other_than_measured")
STAGE_D_EVIDENCE = ("server", "output_tokens", "descriptive", "at_c_prod", "hypotheses", "not_covered")   # §9 only
# 044 (§12): V1 validates a D-A O512 workload's class at a fixed point; its unit and its reserved name
V1_CLASS_NAMES = {"U2K-O512": "uniform-2k-o1-512", "U8K-O512": "uniform-8k-o1-512"}
V1_UNITS = {"U2K-O512": "V1-2K", "U8K-O512": "V1-8K"}
# 044 V1 (實作補記六): the fixed C of L-U2K / L-U8K, the class's max_concurrency = max_queue (GW-U8K-O512 @ 12 is the
# all-512 worst case and covers 8; never the O512 C_PROD), and the unit's points in configs/stage_v.toml's order
V1_C = {"U2K-O512": 32, "U8K-O512": 8}
V1_POINTS = {"U2K-O512": ("L-U2K", "GW-U2K-O512"), "U8K-O512": ("L-U8K", "GW-U8K-O512")}
V1_DISTRIBUTION = "uniform[1,512]"
V1_DATASETS = 5                                       # 044 V1: five pinned sequences per point
COPY_UNITS = {"U2K": "PACE-2K", "U8K": "PACE-8K"}     # 044 V2 / V4: the copies are c1b's uniform-2k / uniform-8k
ADMISSION_UNIT = "W1"
VALIDATED = "validated"                               # 044: a bound only V1's classes have
STUDIES = ("V1", "V2", "V4")
VIEWS = ("client", "upstream")
STAGE_V_EVIDENCE = ("output_distribution", "datasets_sha256", "via_gateway")                           # §12 only
STAGE_V_PROVENANCE = ("stage_v_sha256", "gateway_path_sha256", "capserve_sha256", "base_profiles")     # §12 only
CONFIGS = "configs"


class ProfileError(ValueError):
    pass


class FrozenConfigs(NamedTuple):
    """What a verified profile is checked against at load time: configs/capacity.toml ({} without one: not frozen),
    its sha256 (None without one), the sha256 of configs/pins.toml, and (§9, 043) configs/stage_d.toml and (§12, 044)
    configs/stage_v.toml with their sha256 (None without one). The Stage D / V fields have defaults, so
    FrozenConfigs(capacity, capacity_sha256, pins_sha256) still builds a Stage C one."""
    capacity: dict
    capacity_sha256: str | None
    pins_sha256: str
    stage_d: dict | None = None
    stage_d_sha256: str | None = None
    stage_v: dict | None = None
    stage_v_sha256: str | None = None


def _read_toml(path: Path) -> tuple:
    """(the TOML at path, its sha256), or (None, None) without the file."""
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return tomllib.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def frozen_configs(root=cplan.ROOT) -> FrozenConfigs:
    """The FrozenConfigs of the checkout at root (default: the one these modules run from, e.g. a release, whose git
    archive holds configs/stage_d.toml too)."""
    configs = Path(root) / CONFIGS
    try:
        pins_raw = (configs / "pins.toml").read_bytes()
        capacity, capacity_sha256 = _read_toml(configs / "capacity.toml")
        stage_d, stage_d_sha256 = _read_toml(configs / "stage_d.toml")
        stage_v, stage_v_sha256 = _read_toml(configs / "stage_v.toml")   # 044: [freeze] stage_v false until frozen
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"cannot read {configs}/capacity.toml, stage_d.toml, stage_v.toml or pins.toml: "
                           f"{exc}") from exc
    return FrozenConfigs({} if capacity is None else capacity, capacity_sha256, hashlib.sha256(pins_raw).hexdigest(),
                         stage_d, stage_d_sha256, stage_v, stage_v_sha256)


def stage_c_class(workload: str) -> tuple:
    """(name, max_prompt_tokens, max_output_tokens) of a Stage C workload's class (030 8.7): uniform L takes prompts
    of L - 128 tokens, mixed L the 007 mix's 32,640-token long requests; 128 output tokens."""
    L, mix = cplan.WORKLOAD_SPEC[workload]
    return CLASS_NAMES[workload], bp.LONG_INPUT if mix == "mixed" else L - bp.N_GEN, bp.N_GEN


def _d_a_workload(workload: str) -> dict:
    if workload not in STAGE_D_CLASS_NAMES:
        raise ProfileError(f"{workload!r} is not a D-A workload {list(STAGE_D_CLASS_NAMES)}: only D-A's closed-loop "
                           "workloads have a Stage D class (043; never the sentinel, D-B or D-C)")
    return sdplan.WORKLOADS[workload]


def stage_d_class(workload: str) -> tuple:
    """(name, max_prompt_tokens, max_output_tokens) of a D-A workload's class (043 §2): its stage_d_plan.WORKLOADS
    prompt (M8K-T, a mixed one: the 007 mix's 32,640-token long requests, as mixed-8k) and its output length."""
    spec = _d_a_workload(workload)
    return STAGE_D_CLASS_NAMES[workload], bp.LONG_INPUT if spec["mix"] == "mixed" else spec["prompt"], spec["out"]


def not_covered_for(workload: str, flags) -> list:
    """043 §5: what a D-A class's limit was never measured under, in NOT_COVERED's order: for every class the six
    common items (prompt lengths, the gateway, open loop, other output lengths, the start-up wave, concurrent
    classes), plus, for an ABOVE_CKV class, requests waiting inside vLLM beyond C_KV. Enums only, never a D-B or D-C
    number."""
    _d_a_workload(workload)
    return [n for n in NOT_COVERED if n != V1_NOT_COVERED and (n != ABOVE_CKV_NOT_COVERED or "ABOVE_CKV" in flags)]


def v1_not_covered(workload: str, flags) -> list:
    """044 "§12 的 class": a V1 class's not_covered: its D-A workload's, without what V1 measured (via the gateway,
    output lengths other than 512), plus the output distribution (V1 measured only uniform[1, 512])."""
    return [n for n in not_covered_for(workload, flags) if n not in V1_COVERED] + [V1_NOT_COVERED]


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
    long_prefill_token_threshold: int | None = None   # 043: null / absent = B0; T = pins [advanced] (§9 only)


class Estimand(_Strict):
    value: int | None
    status: Literal[C_PROD_STATUSES]


class Descriptive(_Strict):
    """043: the C_PROD under the analysis's other windows (descriptive.*.c_prod): all requests, and W' = [2C, N - C)."""
    c_full: Estimand
    c_late: Estimand


class AtCProd(_Strict):
    """043: over the valid runs at C_PROD (points[C_PROD].runs); preemptions and KV are the whole bench process's."""
    attainment_min: float | None
    preemptions_max: float | None
    kv_cache_usage_peak_max: float | None


class QueueSettings(_Strict):
    """044: the gateway queue settings (settings max_queue_wait_s, request_deadline_s) a candidate policy was validated
    with; it serves only with exactly these."""
    max_queue_wait_s: float
    request_deadline_s: float


class PacingParams(_Strict):
    """044 V2 (043 7.3 B): a prefill token bucket of burst_tokens, refilled at rate_tokens_per_s (settings
    [prefill_pacing])."""
    burst_tokens: int = Field(ge=1)
    rate_tokens_per_s: int = Field(ge=1)


class ClassPacing(PacingParams):
    """§12: the pacing a copy's PACE unit validated, and its queue settings."""
    queue: QueueSettings


class AdmissionClass(_Strict):
    """One weighted-v1 class: a class of the profile by name, its units per request and its cap (requests)."""
    name: str
    units: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)


class AdmissionParams(_Strict):
    """044 V4 (043 7.3 A): weighted-v1 over named classes (settings [admission]); a prompt is admitted as the first
    class it fits."""
    policy: Literal["weighted-v1"]
    budget: int = Field(ge=1)
    max_queue: int = Field(ge=0)
    classes: list[AdmissionClass] = Field(min_length=2)

    @property
    def label(self) -> str:
        """The policy as one label (metrics' active_class): weighted-v1:<class>+<class>."""
        return f"{self.policy}:" + "+".join(c.name for c in self.classes)


class Validation(_Strict):
    """§12: which Stage V unit validated this class or section: its study, points, the provisional study classes those
    points served, the worst maxR over each point's runs per view, and the runs per point. Descriptive."""
    study: Literal[STUDIES]
    unit: str
    points: list[str]
    measured_classes: list[str]
    worst_r: dict[str, dict[Literal[VIEWS], float]]
    n: int


class AdmissionSection(AdmissionParams):
    """§12: the weighted-v1 parameters W1 validated, with its queue settings, the mixes measured and its request
    path."""
    queue: QueueSettings
    measured_mixes: list[float]
    gateway_path_sha256: str
    validation: Validation


class ClassEvidence(_Strict):
    """What a verified profile states for each class besides its limit (030 8.7); descriptive, never a limit."""
    workload: str                          # the Stage C workload (U2K, U8K, M2K, M8K) or the D-A one (§9)
    scope: str                             # where the number holds (8.7: prompts, outputs, steady state, admitted)
    c_prod: Estimand                       # the only estimand a profile may turn into a limit (8.1)
    c_slo: Estimand
    c_kv: int
    c_cap: int
    flags: list[str]
    binding: list[str] | None              # the terms above 1 at K's smallest FAIL point
    margins: dict[str, float] | None       # 1 - worst R at C_PROD, per term
    burst_max_ttft_ms: float | None        # the start-up wave (idx < C) at C_PROD, worst valid run
    # §9 only (043), null / absent in a §8 profile
    server: Literal[SERVERS] | None = None                   # the D-A workload's server
    output_tokens: int | None = None                         # exactly this many output tokens (ignore-eos) measured
    descriptive: Descriptive | None = None
    at_c_prod: AtCProd | None = None
    hypotheses: dict[str, str] | None = None                 # {id: verdict} as the analysis states them
    not_covered: list[Literal[NOT_COVERED]] | None = None    # not_covered_for(workload, flags), nothing else
    # §12 V1 classes only (044), null / absent elsewhere
    output_distribution: str | None = None                   # "uniform[1,512]"
    datasets_sha256: list[str] | None = None                 # the five pinned sequences
    via_gateway: bool | None = None


class WorkloadClass(_Strict):
    name: str
    max_prompt_tokens: int
    max_output_tokens: int
    max_concurrency: int | None            # None only for an unavailable class
    max_queue: int | None
    source: str
    status: Literal["verified", "unavailable"] | None = None    # verified profiles only (030 8.7)
    bound: Literal["exact", "lower", "censored", "validated"] | None = None   # validated: §12 V1 only (044)
    unavailable_reason: str | None = None
    evidence: ClassEvidence | None = None
    pacing: ClassPacing | None = None        # §12 only (044): the pacing a PACE unit validated
    validation: Validation | None = None     # §12 only (044)

    @property
    def serviceable(self) -> bool:
        """Whether anything may be admitted: False for an unavailable class and for a verified 0 (refuse)."""
        return self.status != "unavailable" and (self.max_concurrency or 0) >= 1


class BaseProfile(_Strict):
    """§12: a committed profile whose classes a Stage V class copies (044)."""
    id: str
    content_sha256: str


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
    stage_d_sha256: str | None = None   # 043: §9's configs/stage_d.toml; null / absent in a §8 profile
    # §12 only (044), null / absent elsewhere
    stage_v_sha256: str | None = None
    gateway_path_sha256: str | None = None   # the frozen request path the units were measured on
    capserve_sha256: str | None = None       # provenance only (run_stage_d.capserve_sha256)
    base_profiles: list[BaseProfile] | None = None


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
    admission: AdmissionSection | None = None   # §12 only (044): W1's weighted-v1 section

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    @property
    def section(self) -> str | None:
        return self.provenance.prereg_section if self.provenance is not None else None

    def validated(self, c: WorkloadClass) -> bool:
        """044: a §12 profile's class with the bound V1 validated (its outputs measured uniform on [1, 512])."""
        return self.verified and self.section == STAGE_V_SECTION and c.bound == VALIDATED

    @property
    def server_variant(self) -> str:
        """043: "T" for a profile of Stage D's T server (long_prefill_token_threshold set), else "B0"."""
        return "B0" if self.server.long_prefill_token_threshold is None else "T"

    def workload_class(self, name: str) -> WorkloadClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise ProfileError(f"profile {self.profile_id} has no class {name!r} (classes: {[c.name for c in self.classes]})")

    def serving_class(self, name: str) -> WorkloadClass:
        """The class a gateway may serve as its active_class, whatever flag is given: an unavailable class is refused
        (030 8.7: no verified number, never a fallback), and so is a verified class of more than 128 output tokens
        (043 §4; 044: unless V1 validated it, bound "validated" in a §12 profile), on a T profile every class but
        M8K-T with H-D2a supported (030 8.8) and, on a B0 profile, every class of a T workload (030 8.8: T's limit is
        never B0's; the loader refuses it too, this reads the model only); a verified 0 is returned (not serviceable:
        refuse it)."""
        c = self.workload_class(name)
        if c.status == "unavailable":
            raise ProfileError(f"profile {self.profile_id} class {name!r} is unavailable ({c.unavailable_reason}): "
                               "Stage C (or D) verified no admission limit for it, so it is never served, whatever "
                               "flag is given (030 8.7)")
        if self.server_variant == "T":
            workload, hypothesis = T_SERVED
            ev = c.evidence
            if ev is None or ev.workload != workload:
                raise ProfileError(f"profile {self.profile_id} is for the T server and class {name!r} is not its "
                                   f"{workload} class: 030 8.8: report only; a T profile serves only a deployment "
                                   "declared as M8K (mixed-8k-t), whatever flag is given (043)")
            verdict = (ev.hypotheses or {}).get(hypothesis)
            if verdict != "supported":
                raise ProfileError(f"profile {self.profile_id} class {name!r}: {hypothesis} is {verdict!r}, not "
                                   "supported: 030 8.8 serves T's M8K class only when its C_PROD exceeds B0's, so it "
                                   "is never served, whatever flag is given (043)")
        elif c.evidence is not None and sdplan.WORKLOADS.get(c.evidence.workload, {}).get("server") == "T":
            raise ProfileError(f"profile {self.profile_id} is for the B0 server and class {name!r} is workload "
                               f"{c.evidence.workload}'s, measured on server T: 030 8.8: its limit is T's, never B0's, "
                               "so it is never served on B0, whatever flag is given (043)")
        if self.verified and c.max_output_tokens > bp.N_GEN and not self.validated(c):
            raise ProfileError(f"profile {self.profile_id} class {name!r} has max_output_tokens {c.max_output_tokens} "
                               f"> {bp.N_GEN}: its limit holds for exactly {c.max_output_tokens} output tokens "
                               "(ignore-eos) and shorter outputs were never measured (they are not safer: U8K's C_PROD "
                               f"is 12 at 512, but C 9 already FAILs at {bp.N_GEN}), so it is never the active class, "
                               "whatever flag is given (043 §4)")
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


def t_fingerprint(pins: dict) -> str:
    """The fingerprint of Stage D's T server (030 8.8, 039): B0 plus --long-prefill-token-threshold <pins
    [advanced]>, from the frozen stage_d_plan.server_argv_for itself (read-only, never rebuilt here), served on the
    product port."""
    argv, env = sdplan.server_argv_for(pins, T_WORKLOAD, port=PRODUCT_VLLM_PORT)
    return server_fingerprint(argv, bp.recorded_env(env))


def server_variant_fingerprint(variant: str, pins: dict) -> str:
    """The fingerprint a §9 profile of server variant B0 or T states (043)."""
    return t_fingerprint(pins) if variant == "T" else stage_b_fingerprint(pins)


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
    section = prof.provenance.prereg_section if prof.provenance is not None else None
    for c in prof.classes:
        if c.max_prompt_tokens < 1 or c.max_output_tokens < 1:
            problems.append(f"class {c.name}: max_prompt_tokens and max_output_tokens must be >= 1")
        if c.max_prompt_tokens + c.max_output_tokens > bp.MAX_MODEL_LEN:
            problems.append(f"class {c.name}: max_prompt_tokens + max_output_tokens > {bp.MAX_MODEL_LEN}")
        problems += _status_problems(c, section) if prof.verified else _provisional_class_problems(c)
    want = {"vllm_version": pins["vllm"]["version"], "model_repo": pins["core_b"]["repo"],
            "model_revision": pins["core_b"]["revision"]}
    for key, value in want.items():
        if getattr(prof.server, key) != value:
            problems.append(f"server.{key} {getattr(prof.server, key)!r} != pins {value!r}")
    if not (prof.verified and section == STAGE_V_SECTION):
        problems += _stage_v_field_problems(prof)
    if prof.verified:
        problems += _verified_problems(prof)
        if section in (STAGE_D_SECTION, STAGE_V_SECTION):
            problems += _stage_d_server_problems(prof, pins)
    else:
        if not (prof.provisional_reason or "").strip():
            problems.append("a provisional profile needs a provisional_reason")
        if prof.profile_id.startswith("verified-"):   # 036: never mistaken for the generator's verified-<session>
            problems.append(f"profile_id {prof.profile_id!r}: a provisional profile's id never starts with 'verified-'")
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
    # 036 / 043 / 044: a production config selects these names; never a provisional limit
    if c.name in CLASS_NAMES.values() or c.name in STAGE_D_CLASS_NAMES.values() or c.name in V1_CLASS_NAMES.values():
        problems.append(f"class name {c.name!r} is reserved for a verified profile's Stage C / Stage D / V1 class (030 "
                        "8.7, 043, 044); a provisional profile names its classes otherwise")
    return problems + _limit_problems(c)


def stage_c_source(workload: str, c_prod_status: str) -> str:
    """A verified profile's class source: the Stage C workload and the status of its C_PROD (030 8.7)."""
    return f"stage_c:{workload}:c_prod:{c_prod_status}"


def stage_d_source(workload: str, c_prod_status: str) -> str:
    """A §9 profile's class source: the D-A workload and the status of its C_PROD (030 8.7 via 043)."""
    return f"stage_d:{workload}:c_prod:{c_prod_status}"


def stage_v_source(unit: str) -> str:
    """A §12 validated class's source: the Stage V unit that validated it (044)."""
    return f"stage_v:{unit}"


def _status_problems(c: WorkloadClass, section: str | None = PREREG_SECTION) -> list:
    """030 8.7 for one class of a verified profile (its source: stage_c for §8 and a §12 copy, stage_d for §9); a §12
    validated class: _validated_problems."""
    where = f"class {c.name}"
    if section == STAGE_V_SECTION and c.bound == VALIDATED:
        return _validated_problems(c)
    if c.status is None:
        return [f"{where}: a verified profile states every class's status (verified | unavailable; 030 8.7)"]
    if c.evidence is None:
        return [f"{where}: a verified profile's class needs its Stage C evidence (C_PROD, C_SLO, C_KV, flags; 8.7)"]
    est, problems = c.evidence.c_prod, []
    source = (stage_d_source if section == STAGE_D_SECTION else stage_c_source)(c.evidence.workload, est.status)
    if c.source != source:
        problems.append(f"{where}: source {c.source!r} != {source!r}")
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


def _validated_problems(c: WorkloadClass) -> list:
    """044: a V1 class is verified at the fixed point V1 validated, max_concurrency = max_queue = V1_C of its D-A
    workload (32 / 8: the C of L-U2K / L-U8K, never the O512 C_PROD; 實作補記六), from that workload's unit; never a
    C_PROD, so no 030 8.7 status rule."""
    where = f"class {c.name}"
    problems = []
    if c.status != "verified" or c.unavailable_reason is not None:
        problems.append(f"{where}: a validated class is verified with unavailable_reason null (044)")
    problems += _limit_problems(c)
    if c.max_queue != c.max_concurrency:
        problems.append(f"{where}: max_queue {c.max_queue!r} != max_concurrency {c.max_concurrency!r} (044: the C V1 "
                        "validated, as 030 8.7's max_queue = C_PROD)")
    if c.evidence is None:
        return problems + [f"{where}: a validated class states its D-A workload's evidence (044)"]
    w = c.evidence.workload
    if w in V1_C and not c.max_concurrency == c.max_queue == V1_C[w]:
        problems.append(f"{where}: max_concurrency {c.max_concurrency!r} / max_queue {c.max_queue!r} != {V1_C[w]} for "
                        f"{w} (044 V1: the point's C, never the O512 C_PROD)")
    unit = V1_UNITS.get(w)
    if unit is None or c.source != stage_v_source(unit):
        problems.append(f"{where}: source {c.source!r} != {stage_v_source(unit) if unit else 'stage_v:<V1 unit>'!r} "
                        "(044: the unit that validated it)")
    return problems


def _stage_v_field_problems(prof: CapacityProfile) -> list:
    """044: §12's fields (pacing, validation, an admission section, V1's evidence, the Stage V provenance) are null or
    absent in every other profile (§8, §9, provisional): no candidate policy is ever attached to one by hand."""
    problems = []
    if prof.admission is not None:
        problems.append("admission is a §12 (Stage V, 044) profile's section; this profile has it null")
    if prof.provenance is not None:
        extra = [k for k in STAGE_V_PROVENANCE if getattr(prof.provenance, k) is not None]
        if extra:
            problems.append(f"provenance {extra} are §12's (Stage V, 044); this profile has them null")
    for c in prof.classes:
        extra = [k for k in ("pacing", "validation") if getattr(c, k) is not None]
        extra += [k for k in STAGE_V_EVIDENCE if c.evidence is not None and getattr(c.evidence, k) is not None]
        if extra:
            problems.append(f"class {c.name}: {extra} are §12's (Stage V, 044); this class has them null")
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
    if p.prereg_section not in PREREG_SECTIONS:
        problems.append(f"provenance.prereg_section {p.prereg_section!r} is not one of {PREREG_SECTIONS}")
    for key in ("freeze_commit", "generator_commit", "deployed_commit"):
        if not _HEX40.fullmatch(getattr(p, key)):
            problems.append(f"provenance.{key} must be a full git commit (40 lowercase hex)")
    if not (p.gpu.strip() and p.gpu_driver.strip()):
        problems.append("provenance.gpu and provenance.gpu_driver name the GPU the session ran on")
    if p.prereg_section in (STAGE_D_SECTION, STAGE_V_SECTION):
        if not _HEX64.fullmatch(p.stage_d_sha256 or ""):
            problems.append("provenance.stage_d_sha256 must be 64 lowercase hex (§9 / §12: configs/stage_d.toml)")
    elif p.prereg_section == PREREG_SECTION:
        problems += _stage_c_field_problems(prof)
    if p.prereg_section == STAGE_V_SECTION:
        for key in ("stage_v_sha256", "gateway_path_sha256", "capserve_sha256"):
            if not _HEX64.fullmatch(getattr(p, key) or ""):
                problems.append(f"provenance.{key} must be 64 lowercase hex (§12, 044)")
        bases = p.base_profiles or []
        if not bases or len({b.id for b in bases}) != len(bases) or not all(
                b.id.startswith("verified-") and _HEX64.fullmatch(b.content_sha256) for b in bases):
            problems.append("provenance.base_profiles must name the committed verified profiles the copies come from, "
                            "once each, with their 64-hex content_sha256 (§12, 044)")
    return problems


def _stage_c_field_problems(prof: CapacityProfile) -> list:
    """A §8 profile has none of §9's fields (043): Stage C measured only B0, and its generator writes none of them."""
    problems = []
    if prof.provenance.stage_d_sha256 is not None:
        problems.append("provenance.stage_d_sha256 is §9's (Stage D); a §8 profile has it null")
    if prof.server.long_prefill_token_threshold is not None:
        problems.append("server.long_prefill_token_threshold is §9's T server; Stage C (§8) measured only B0, so a §8 "
                        "profile has it null")
    for c in prof.classes:
        extra = [k for k in STAGE_D_EVIDENCE if c.evidence is not None and getattr(c.evidence, k) is not None]
        if extra:
            problems.append(f"class {c.name}: evidence {extra} are §9's (Stage D, 043); a §8 class has them null")
    return problems


def _stage_d_server_problems(prof: CapacityProfile, pins: dict) -> list:
    """§9 (043): the server is Stage D's B0 or T, T at pins [advanced]'s threshold, with that server's fingerprint
    (B0: stage_b_fingerprint, T: t_fingerprint), so a B0 fingerprint never comes with T's threshold or vice versa."""
    problems = []
    threshold, want = prof.server.long_prefill_token_threshold, pins["advanced"]["long_prefill_token_threshold"]
    if threshold is not None and threshold != want:
        problems.append(f"server.long_prefill_token_threshold {threshold!r} is neither null (B0) nor pins [advanced] "
                        f"{want} (T, 030 8.8)")
    variant = prof.server_variant
    fingerprint = server_variant_fingerprint(variant, pins)
    if prof.server.fingerprint_sha256 != fingerprint:
        problems.append(f"server.fingerprint_sha256 {prof.server.fingerprint_sha256} is not the {variant} server's "
                        f"{fingerprint} (043: the fingerprint follows long_prefill_token_threshold)")
    return problems


def provenance_problems(doc: dict, prof: CapacityProfile, frozen: FrozenConfigs) -> list:
    """Every rule a verified profile breaks against this checkout's frozen configs and the Stage C classes (029 M2);
    a §9 profile against the Stage D ones (043)."""
    if prof.section == STAGE_D_SECTION:
        return _stage_d_provenance_problems(doc, prof, frozen)
    if prof.section == STAGE_V_SECTION:
        return _stage_v_provenance_problems(doc, prof, frozen)
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


def _stage_d_provenance_problems(doc: dict, prof: CapacityProfile, frozen: FrozenConfigs) -> list:
    """§9 (043 §3) against this checkout: both freezes, its stage_d.toml / capacity.toml / pins.toml, stage_d.toml's
    SLO, the session's ids, and at most one D-A class per workload, of this profile's server."""
    problems = []
    stage_d = frozen.stage_d or {}
    if (stage_d.get("freeze") or {}).get("stage_d") is not True:
        problems.append("configs/stage_d.toml has no [freeze] stage_d = true: no Stage D capacity exists before the "
                        "freeze (039), so this profile does not come from the generator")
    if (frozen.capacity.get("freeze") or {}).get("capacity") is not True:
        problems.append("configs/capacity.toml has no [freeze] capacity = true: Stage D runs the frozen Stage C "
                        "protocol (039), so this profile does not come from the generator")
    slo = stage_d.get("slo")
    if doc.get("slo") is not None and slo != {"id": doc["slo"].get("id"), **(doc["slo"].get("terms") or {})}:
        problems.append(f"slo {doc['slo'].get('id')!r} is not configs/stage_d.toml [slo] "
                        f"{(slo or {}).get('id')!r} verbatim (043)")
    p, variant = prof.provenance, prof.server_variant
    want = f"verified-{p.source_session}-{variant.lower()}"
    if prof.profile_id != want:
        problems.append(f"profile_id {prof.profile_id!r} != {want!r} (its source_session and server {variant}, 043)")
    want = f"results/stage_d/{p.source_session}/analysis.json"
    if p.analysis_path != want:
        problems.append(f"provenance.analysis_path {p.analysis_path!r} != {want!r} (the session's analysis.json)")
    for key, name in (("stage_d_sha256", "stage_d.toml"), ("capacity_sha256", "capacity.toml"),
                      ("pins_sha256", "pins.toml")):
        if getattr(p, key) != getattr(frozen, key):
            problems.append(f"provenance.{key} {getattr(p, key)} is not this checkout's configs/{name} "
                            f"{getattr(frozen, key)}")
    workloads = [c.evidence.workload for c in prof.classes if c.evidence is not None]
    dups = sorted({w for w in workloads if workloads.count(w) > 1})
    if dups:
        problems.append(f"workloads {dups} have more than one class (043: one class per D-A workload)")
    for c in prof.classes:
        if c.evidence is None:
            continue
        ev, w = c.evidence, c.evidence.workload
        if w not in STAGE_D_CLASS_NAMES:
            problems.append(f"class {c.name}: evidence.workload {w!r} is not a D-A workload "
                            f"{list(STAGE_D_CLASS_NAMES)} (043: the sentinel, D-B and D-C never become a class)")
            continue
        want = dict(zip(("name", "max_prompt_tokens", "max_output_tokens"), stage_d_class(w)))
        got = {k: getattr(c, k) for k in want}
        if got != want:
            problems.append(f"class {c.name}: {got} is not workload {w}'s class {want} (043 §2)")
        spec = sdplan.WORKLOADS[w]
        if spec["server"] != variant or ev.server != variant:
            problems.append(f"class {c.name}: workload {w} runs on server {spec['server']} and its evidence.server is "
                            f"{ev.server!r}, but this profile is for server {variant} (043: one profile per server)")
        if ev.output_tokens != spec["out"]:
            problems.append(f"class {c.name}: evidence.output_tokens {ev.output_tokens!r} != workload {w}'s "
                            f"{spec['out']} (the measured output length)")
        if ev.not_covered != not_covered_for(w, ev.flags):
            problems.append(f"class {c.name}: evidence.not_covered {ev.not_covered!r} != "
                            f"{not_covered_for(w, ev.flags)} (043 §5: not_covered_for its workload and flags)")
    return problems


def admission_problems(adm: AdmissionParams, classes: Sequence[WorkloadClass]) -> list:
    """weighted-v1's parameters against the classes they name, in order (044; settings [admission] and a §12 section
    alike): each class once, its cap its max_concurrency and units x cap = budget (043 7.3 A: a class alone fills the
    budget exactly at its C_PROD, so any mix with sum(n / C_PROD) <= 1 is admitted), strictly increasing
    max_prompt_tokens (a prompt goes to the first class it fits) and one max_output_tokens."""
    problems = []
    names = [a.name for a in adm.classes]
    if len(set(names)) != len(names):
        problems.append(f"admission names a class twice {names}")
    for a, c in zip(adm.classes, classes):
        if a.max_concurrency != c.max_concurrency:
            problems.append(f"admission class {a.name}: max_concurrency {a.max_concurrency} != the class's "
                            f"{c.max_concurrency!r} (044: the caps are the profile's limits)")
        if a.units * a.max_concurrency != adm.budget:
            problems.append(f"admission class {a.name}: units {a.units} x max_concurrency {a.max_concurrency} != "
                            f"budget {adm.budget} (043 7.3 A: units = budget / the class's C_PROD)")
    prompts = [c.max_prompt_tokens for c in classes]
    if any(a >= b for a, b in zip(prompts, prompts[1:])):
        problems.append(f"admission classes {names} are not in strictly increasing max_prompt_tokens {prompts} (a "
                        "prompt is admitted as the first class it fits)")
    if len({c.max_output_tokens for c in classes}) > 1:
        problems.append(f"admission classes {names} differ in max_output_tokens (one output bound per request)")
    return problems


def _validation_problems(v: Validation | None, where: str, *, study: str, unit: str) -> list:
    if v is None:
        return [f"{where}: a validated §12 item states its validation ({study} {unit}, 044)"]
    problems = []
    if (v.study, v.unit) != (study, unit):
        problems.append(f"{where}: validation {v.study} {v.unit} != {study} {unit} (044)")
    if not v.points or len(set(v.points)) != len(v.points) or set(v.worst_r) != set(v.points) or not all(
            v.worst_r.values()):
        problems.append(f"{where}: validation points {v.points} and worst_r {sorted(v.worst_r)}: every point once, "
                        "with its worst R per view")
    if not v.measured_classes or v.n < 1:
        problems.append(f"{where}: validation needs the study classes measured and n >= 1")
    return problems


def _v1_class_problems(c: WorkloadClass) -> list:
    """044: a V1 class is its D-A O512 workload's class under V1's name, B0, 512 output tokens, not_covered by
    v1_not_covered, V1's output distribution and five datasets, measured via the gateway; no pacing; its validation is
    its unit's, over exactly the unit's points (V1_POINTS; its C: _validated_problems)."""
    ev, where = c.evidence, f"class {c.name}"
    w = ev.workload
    if w not in V1_CLASS_NAMES:
        return [f"{where}: evidence.workload {w!r} is not one V1 validates {list(V1_CLASS_NAMES)} (044)"]
    spec = sdplan.WORKLOADS[w]
    want = {"name": V1_CLASS_NAMES[w], "max_prompt_tokens": spec["prompt"], "max_output_tokens": spec["out"]}
    got = {k: getattr(c, k) for k in want}
    problems = [] if got == want else [f"{where}: {got} is not V1's class of {w} {want} (044)"]
    if ev.server != spec["server"] or ev.output_tokens != spec["out"]:
        problems.append(f"{where}: evidence.server {ev.server!r} / output_tokens {ev.output_tokens!r} != {w}'s "
                        f"{spec['server']} / {spec['out']}")
    if ev.not_covered != v1_not_covered(w, ev.flags):
        problems.append(f"{where}: evidence.not_covered {ev.not_covered!r} != {v1_not_covered(w, ev.flags)} (044: "
                        "v1_not_covered)")
    if ev.output_distribution != V1_DISTRIBUTION:
        problems.append(f"{where}: evidence.output_distribution {ev.output_distribution!r} != {V1_DISTRIBUTION!r}")
    shas = ev.datasets_sha256 or []
    if len(shas) != V1_DATASETS or len(set(shas)) != V1_DATASETS or not all(_HEX64.fullmatch(x) for x in shas):
        problems.append(f"{where}: evidence.datasets_sha256 must be the {V1_DATASETS} distinct 64-hex sequences V1 ran")
    if ev.via_gateway is not True:
        problems.append(f"{where}: evidence.via_gateway {ev.via_gateway!r}: V1 measured through the gateway (044)")
    if c.pacing is not None:
        problems.append(f"{where}: a V1 class has no pacing (044: V1 ran with pacing off)")
    problems += _validation_problems(c.validation, where, study="V1", unit=V1_UNITS[w])
    if c.validation is not None and c.validation.points != list(V1_POINTS[w]):
        problems.append(f"{where}: validation.points {c.validation.points} != {list(V1_POINTS[w])}, the points of "
                        f"{V1_UNITS[w]} (044 產品對應)")
    return problems


def _copy_class_problems(c: WorkloadClass, member: bool) -> list:
    """044: a §12 copy is c1b's uniform-2k / uniform-8k (§8's class and evidence only), with its PACE unit's pacing and
    validation, or plain as an admission member."""
    ev, where = c.evidence, f"class {c.name}"
    w = ev.workload
    if w not in COPY_UNITS:
        return [f"{where}: evidence.workload {w!r}: a §12 copy is c1b's uniform-2k or uniform-8k (044 V2 / V4: "
                f"{list(COPY_UNITS)}); V1's classes have bound {VALIDATED!r}"]
    want = dict(zip(("name", "max_prompt_tokens", "max_output_tokens"), stage_c_class(w)))
    got = {k: getattr(c, k) for k in want}
    problems = [] if got == want else [f"{where}: {got} is not workload {w}'s class {want} (a copy, 044)"]
    extra = [k for k in STAGE_D_EVIDENCE + STAGE_V_EVIDENCE if getattr(ev, k) is not None]
    if extra:
        problems.append(f"{where}: evidence {extra}: a copy carries its §8 class's evidence only (044)")
    if c.pacing is None:
        if c.validation is not None:
            problems.append(f"{where}: a copy's validation is its pacing's ({COPY_UNITS[w]}); without pacing it has "
                            "none (an admission member: the section's validation)")
        elif not member:
            problems.append(f"{where}: a copy without pacing is only a weighted-v1 class, but the admission section "
                            "does not name it (044)")
        return problems
    if c.pacing.burst_tokens < c.max_prompt_tokens:
        problems.append(f"{where}: pacing burst_tokens {c.pacing.burst_tokens} < max_prompt_tokens "
                        f"{c.max_prompt_tokens}: such a prompt could never be admitted (044)")
    return problems + _validation_problems(c.validation, where, study="V2", unit=COPY_UNITS[w])


def _stage_v_provenance_problems(doc: dict, prof: CapacityProfile, frozen: FrozenConfigs) -> list:
    """§12 (044) against this checkout: the three freezes (configs/stage_v.toml exists before Stage V's freeze with
    [freeze] stage_v = false; until it is true every §12 profile is refused: fail closed), its stage_v.toml /
    stage_d.toml / capacity.toml / pins.toml, stage_v.toml's SLO and frozen gateway_path_sha256, the session's ids, B0,
    and the §12 classes and admission section."""
    problems = []
    stage_v = frozen.stage_v or {}
    freeze = stage_v.get("freeze") or {}
    if freeze.get("stage_v") is not True:
        problems.append("configs/stage_v.toml is missing or has no [freeze] stage_v = true: nothing is validated "
                        "before Stage V's freeze (044), so this profile does not come from the generator")
    if ((frozen.stage_d or {}).get("freeze") or {}).get("stage_d") is not True:
        problems.append("configs/stage_d.toml has no [freeze] stage_d = true (Stage V reuses Stage D's frozen config)")
    if (frozen.capacity.get("freeze") or {}).get("capacity") is not True:
        problems.append("configs/capacity.toml has no [freeze] capacity = true (Stage V runs the frozen Stage C "
                        "protocol)")
    slo = stage_v.get("slo")
    if doc.get("slo") is not None and slo != {"id": doc["slo"].get("id"), **(doc["slo"].get("terms") or {})}:
        problems.append(f"slo {doc['slo'].get('id')!r} is not configs/stage_v.toml [slo] {(slo or {}).get('id')!r} "
                        "verbatim (044)")
    p = prof.provenance
    if prof.server_variant != "B0":
        problems.append("a §12 profile is for the B0 server (044: every Stage V unit was measured on B0)")
    want = f"verified-{p.source_session}-b0"
    if prof.profile_id != want:
        problems.append(f"profile_id {prof.profile_id!r} != {want!r} (its source_session, 044)")
    want = f"results/stage_v/{p.source_session}/analysis.json"
    if p.analysis_path != want:
        problems.append(f"provenance.analysis_path {p.analysis_path!r} != {want!r} (the session's analysis.json)")
    for key, name in (("stage_v_sha256", "stage_v.toml"), ("stage_d_sha256", "stage_d.toml"),
                      ("capacity_sha256", "capacity.toml"), ("pins_sha256", "pins.toml")):
        if getattr(p, key) != getattr(frozen, key):
            problems.append(f"provenance.{key} {getattr(p, key)} is not this checkout's configs/{name} "
                            f"{getattr(frozen, key)}")
    if p.gateway_path_sha256 != freeze.get("gateway_path_sha256"):
        problems.append(f"provenance.gateway_path_sha256 {p.gateway_path_sha256} is not configs/stage_v.toml [freeze] "
                        f"gateway_path_sha256 {freeze.get('gateway_path_sha256')} (044: the frozen request path)")
    workloads = [c.evidence.workload for c in prof.classes if c.evidence is not None]
    dups = sorted({w for w in workloads if workloads.count(w) > 1})
    if dups:
        problems.append(f"workloads {dups} have more than one class (044: one class per workload)")
    adm = prof.admission
    members = {a.name for a in adm.classes} if adm is not None else set()
    for c in prof.classes:
        if c.evidence is not None:
            problems += _v1_class_problems(c) if c.bound == VALIDATED else _copy_class_problems(c, c.name in members)
    if adm is not None:
        copies = {c.name: c for c in prof.classes if c.bound != VALIDATED}
        missing = [a.name for a in adm.classes if a.name not in copies]
        if missing:
            problems.append(f"admission classes {missing} are not this profile's copies {sorted(copies)} (044)")
        else:
            problems += admission_problems(adm, [copies[a.name] for a in adm.classes])
        if adm.gateway_path_sha256 != p.gateway_path_sha256:
            problems.append(f"admission.gateway_path_sha256 {adm.gateway_path_sha256} != the provenance's "
                            f"{p.gateway_path_sha256} (044)")
        if not adm.measured_mixes or not all(0 < m < 1 for m in adm.measured_mixes):
            problems.append(f"admission.measured_mixes {adm.measured_mixes}: the 2K shares measured, each in (0, 1)")
        problems += _validation_problems(adm.validation, "admission", study="V4", unit=ADMISSION_UNIT)
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
