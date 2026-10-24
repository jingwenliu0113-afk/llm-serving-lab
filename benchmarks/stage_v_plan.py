#!/usr/bin/env python3
"""Stage V plan rules (decision 044; PREREGISTRATION §12), fixed before any Stage V data. Core, Stage C and Stage D
modules are imported, never changed: the block order is capacity_plan.block_order, N core_b_plan.num_prompts, every
server and client stage_d_plan's own functions (server_argv_for, bench_argv_for) of a point's base workloads, and
every timeout stage_d_plan.timeout_s of stage_d_plan.tau1 on the frozen stage_d.toml.

- configs/stage_v.toml must hold exactly the 044 values (stage_v_problems): seed, n 5, m 0.05, the sha256 of the
  frozen stage_d.toml (its tau1 table, its var-2048 pin, its SLO and sentinel, which Stage V shares), the sessions,
  the hour caps (mini 1.0, V-ADM 3.5, V-CLS 7.0 h; 12 h over every session; 2 h between formal sessions), the run
  overheads (B0 95 s, T 110 s), the non-trivial thresholds, the access-log join tolerance, the gateway's drain, the
  request path's files, the points, the queue settings, weighted-v1, the pacing and its recomputation rule, the
  datasets, the product units and the sentinel. freeze.stage_v = true needs a passed mini smoke of this harness on
  the frozen request path (freeze_evidence_problems), the frozen gateway_path_sha256 and every dataset pinned.
- Points (POINTS): three bases written separately (044 審查: an L client from U2K-O512 would have no custom flag):
  server_base, the stage_d_plan workload whose server (B0, or T) the boot starts; client_base, the workload
  stage_d_plan.bench_argv_for builds the client from; tau_base, the workload whose tau1 sets the timeout. Then C,
  the profile (a committed verified one, served flagless, or the provisional study profile) and its class(es), the
  queue settings ("product" 2 s / 300 s, or "slow" 120 s / 600 s), weighted-v1 or pacing (never both), the dataset
  (per_block: block r reads <dataset>-r<r>), out_pinned (L points: every row's own output length), the judging view,
  the product unit, the bench estimate and the /readyz limits the gateway must report (weighted-v1: its envelope).
- Units (UNITS, 044 產品對應): a unit holds only when each of its points PASSes. Skipping is the inherited 039 / 030
  rule, per point, for every point (044 實作補記六): once a run of a point is not ok, that point alone leaves the
  later boots; its unit's other points keep running, so each point's own hypothesis stays answerable, and the unit
  cannot hold. The V2 controls and the var pair belong to no unit: the predicted var FAIL never stops P-U2K; PACE-2K
  is vetoed instead (VETOES) when F-U2K-var PASSes and P-U2K-var does not.
- Client argv (bench_argv): the frozen function's output with only 044's three differences: --base-url port 8000 ->
  8080 where the frozen function gives 8000; L points only, --custom-output-len 128 -> -1 (vLLM 0.30 reads each row's
  output_tokens only then); --request-id-prefix <tag>- appended on every gateway run (its default is
  bench-<uuid8>-; the access log's client_request_id is <tag>-<idx>). The sentinel's client is stage_d_plan's.
- Gateway settings (gateway_settings): D-C's (039 implementation note 7: 127.0.0.1:8080 -> vLLM 8000, tokenizer
  local from the served snapshot, drain 5 s) plus the point's queue settings, written out even when they are the
  product defaults, and its class, or weighted-v1's [admission] instead of a class, and [prefill_pacing]. toml_dumps
  writes nested tables (run_stage_d._toml writes flat keys only); the plan pins each point's rendered bytes.
- Pacing (044 V2): pinned in stage_v.toml, because a WSL deploy carries no results/; pacing_problems recomputes it
  from the committed D5 file (the Mac tests and the report): the slope is the least-squares slope of the mean
  largest first-wave TTFT over the points C <= C_KV (2K 35, 8K 13; the file's own fit spans every C), the rate
  round_to x floor(prompt / slope / round_to) tok/s, the burst k prompts with k the largest count whose serial
  prefill k x slope stays within (1 - m) x the tier's S2 TTFT (042 D5's 5 and 4).
- Timeouts: ceil60(2 N tau1 + 60 s) with tau1 = stage_d_plan.tau1(stage_d.toml, tau_base) and N =
  core_b_plan.num_prompts(C). Hours: a point's estimate is its bench estimate plus its server's run overhead; a V
  boot starts only while the elapsed time plus its estimate stays within the cap less END_RESERVE_H, the end
  sentinel's reserve (cap_allows).
- The request path (gateway_path_files, gateway_path_sha256): the import closure of service/capserve/__main__.py,
  computed from the source with ast and never by importing (research never imports capserve), with the same
  algorithm as capserve.gateway_path; service/tests assert both ends agree on the checkout.
- Freeze evidence: the mini's session.json records harness_sha256 (HARNESS_FILES, the five Stage V modules),
  gateway_path_sha256, each point's profile_sha256 and each dataset's dataset_sha256; they must be the current
  harness's, the frozen request path's and the bytes stage_v.toml pins (044 實作補記六).

Usage (CPU only, Mac or WSL):
    python3 benchmarks/stage_v_plan.py --session V-ADM --out stage-v-plan.json [--formal]

Stdlib only, Python 3.10 compatible (tomllib only in the CLI).
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import capacity_plan as cp
import core_b_plan as bp
import stage_d_plan as sd
from run_core_a import sha256_json

SCHEMA_VERSION = 1
STAGE = "V"
KIND = "verify"                        # every session is D-C's kind: fixed points, n runs, one boot each (039)
ROOT = Path(__file__).resolve().parent.parent
STAGE_V_PATH = ROOT / "configs" / "stage_v.toml"
RESULTS = ("results", "stage_v")
PORT = sd.PORT                         # the research vLLM (8000)
GATEWAY_PORT = sd.GATEWAY_PORT         # capserve serve (8080)
SEED = bp.SEED
N_RUNS = 5
GUARD_BAND = cp.GUARD_BAND
STAGE_D_SHA256 = "be7909b1e98183c337182f9559953e2613cad96f06d03899815fc0af88cba2a5"   # the frozen stage_d.toml (039)
D_A1B_B0 = "service/profiles/verified-formal-20261023-d-a1b-b0.json"
D_A1B_T = "service/profiles/verified-formal-20261023-d-a1b-t.json"
STUDY_PROFILE = "service/profiles/provisional-stage-v.json"
VIEWS = ("client", "upstream")
JUDGED_VIEWS = {"client": ["client"], "upstream": ["upstream"], "worse": ["upstream", "client"]}   # V2: the worse


def _limits(concurrency: int, queue: int, prompt: int, out: int) -> dict:
    return {"max_concurrency": concurrency, "max_queue": queue, "max_prompt_tokens": prompt, "max_output_tokens": out}


_FIXED = {"admission": False, "pacing": "", "dataset": "", "per_block": False, "out_pinned": False}
POINTS = {
    # V-GW: Stage D's classes through the gateway at C_PROD (D-C's protocol), product queue settings
    "GW-U32K": {"study": "V-GW", "server_base": "U32K", "client_base": "U32K", "tau_base": "U32K", "C": 2,
                "profile": D_A1B_B0, "classes": ["uniform-32k"], "provisional": False, "queue": "product", **_FIXED,
                "view": "client", "unit": "VG-32K", "bench_est_s": 1462, "limits": _limits(2, 2, 32640, 128)},
    "GW-U2K-O512": {"study": "V-GW", "server_base": "U2K-O512", "client_base": "U2K-O512", "tau_base": "U2K-O512",
                    "C": 32, "profile": STUDY_PROFILE, "classes": ["v-uniform-2k-o512"], "provisional": True,
                    "queue": "product", **_FIXED, "view": "client", "unit": "V1-2K", "bench_est_s": 216,
                    "limits": _limits(32, 32, 1920, 512)},
    "GW-U8K-O512": {"study": "V-GW", "server_base": "U8K-O512", "client_base": "U8K-O512", "tau_base": "U8K-O512",
                    "C": 12, "profile": STUDY_PROFILE, "classes": ["v-uniform-8k-o512"], "provisional": True,
                    "queue": "product", **_FIXED, "view": "client", "unit": "V1-8K", "bench_est_s": 426,
                    "limits": _limits(12, 12, 8064, 512)},
    "GW-M8K-T": {"study": "V-GW", "server_base": "M8K-T", "client_base": "M8K-T", "tau_base": "M8K-T", "C": 4,
                 "profile": D_A1B_T, "classes": ["mixed-8k-t"], "provisional": False, "queue": "product",
                 **{**_FIXED, "dataset": "mixed-8192"}, "view": "client", "unit": "VG-M8K-T", "bench_est_s": 490,
                 "limits": _limits(4, 4, 32640, 128)},
    # V1: outputs uniform on [1, 512], pinned per row, through the gateway
    "L-U2K": {"study": "V1", "server_base": "U2K-O512", "client_base": "U2K-var", "tau_base": "U2K-O512", "C": 32,
              "profile": STUDY_PROFILE, "classes": ["v-uniform-2k-o512"], "provisional": True, "queue": "product",
              **{**_FIXED, "dataset": "v1-2048", "per_block": True, "out_pinned": True}, "view": "client",
              "unit": "V1-2K", "bench_est_s": 156, "limits": _limits(32, 32, 1920, 512)},
    "L-U8K": {"study": "V1", "server_base": "U8K-O512", "client_base": "U8K-var", "tau_base": "U8K-O512", "C": 8,
              "profile": STUDY_PROFILE, "classes": ["v-uniform-8k-o512-c8"], "provisional": True, "queue": "product",
              **{**_FIXED, "dataset": "v1-8192", "per_block": True, "out_pinned": True}, "view": "client",
              "unit": "V1-8K", "bench_est_s": 370, "limits": _limits(8, 8, 8064, 512)},
    # V2: prefill pacing on (P) and its controls off (F), "slow" queue settings, judged on the worse view
    "P-U2K": {"study": "V2", "server_base": "U2K-gw", "client_base": "U2K-gw", "tau_base": "U2K-gw", "C": 32,
              "profile": STUDY_PROFILE, "classes": ["v-uniform-2k"], "provisional": True, "queue": "slow",
              **{**_FIXED, "pacing": "2048"}, "view": "worse", "unit": "PACE-2K", "bench_est_s": 130,
              "limits": _limits(32, 32, 1920, 128)},
    "P-U8K": {"study": "V2", "server_base": "U8K-gw", "client_base": "U8K-gw", "tau_base": "U8K-gw", "C": 8,
              "profile": STUDY_PROFILE, "classes": ["v-uniform-8k"], "provisional": True, "queue": "slow",
              **{**_FIXED, "pacing": "8192"}, "view": "worse", "unit": "PACE-8K", "bench_est_s": 325,
              "limits": _limits(8, 8, 8064, 128)},
    "P-U2K-var": {"study": "V2", "server_base": "U2K-var", "client_base": "U2K-var", "tau_base": "U2K-var", "C": 32,
                  "profile": STUDY_PROFILE, "classes": ["v-uniform-2k"], "provisional": True, "queue": "slow",
                  **{**_FIXED, "pacing": "2048", "dataset": "var-2048"}, "view": "worse", "unit": "",
                  "bench_est_s": 80, "limits": _limits(32, 32, 1920, 128)},
    "F-U2K-var": {"study": "V2", "server_base": "U2K-var", "client_base": "U2K-var", "tau_base": "U2K-var", "C": 32,
                  "profile": STUDY_PROFILE, "classes": ["v-uniform-2k"], "provisional": True, "queue": "slow",
                  **{**_FIXED, "dataset": "var-2048"}, "view": "worse", "unit": "", "bench_est_s": 78,
                  "limits": _limits(32, 32, 1920, 128)},
    "F-U2K": {"study": "V2", "server_base": "U2K-gw", "client_base": "U2K-gw", "tau_base": "U2K-gw", "C": 32,
              "profile": STUDY_PROFILE, "classes": ["v-uniform-2k"], "provisional": True, "queue": "slow", **_FIXED,
              "view": "worse", "unit": "", "bench_est_s": 125, "limits": _limits(32, 32, 1920, 128)},
    # V4: weighted-v1 over v-uniform-2k / v-uniform-8k, pacing off, client 32, judged upstream
    "W-90": {"study": "V4", "server_base": "U2K-var", "client_base": "U2K-var", "tau_base": "U8K-gw", "C": 32,
             "profile": STUDY_PROFILE, "classes": ["v-uniform-2k", "v-uniform-8k"], "provisional": True,
             "queue": "slow", **{**_FIXED, "admission": True, "dataset": "v4-90", "per_block": True},
             "view": "upstream", "unit": "W1", "bench_est_s": 170, "limits": _limits(32, 32, 8064, 128)},
    "W-50": {"study": "V4", "server_base": "U2K-var", "client_base": "U2K-var", "tau_base": "U8K-gw", "C": 32,
             "profile": STUDY_PROFILE, "classes": ["v-uniform-2k", "v-uniform-8k"], "provisional": True,
             "queue": "slow", **{**_FIXED, "admission": True, "dataset": "v4-50", "per_block": True},
             "view": "upstream", "unit": "W1", "bench_est_s": 330, "limits": _limits(32, 32, 8064, 128)},
    # the mini's policy point: one 2K and eight 8K first (33 units > 32), formal parameters
    "W-mini": {"study": "V4", "server_base": "U2K-var", "client_base": "U2K-var", "tau_base": "U8K-gw", "C": 9,
               "profile": STUDY_PROFILE, "classes": ["v-uniform-2k", "v-uniform-8k"], "provisional": True,
               "queue": "slow", **{**_FIXED, "admission": True, "dataset": "v4-mini"}, "view": "upstream",
               "unit": "", "bench_est_s": 30, "limits": _limits(32, 32, 8064, 128)},
}
SESSIONS = {"mini": ["GW-U8K-O512", "L-U8K", "GW-M8K-T", "P-U2K", "F-U2K-var", "W-mini"],
            "V-ADM": ["P-U2K", "P-U8K", "P-U2K-var", "F-U2K-var", "F-U2K", "W-90", "W-50"],
            "V-CLS": ["GW-U32K", "GW-U2K-O512", "GW-U8K-O512", "GW-M8K-T", "L-U2K", "L-U8K"]}
UNITS = {"VG-32K": ["GW-U32K"], "VG-M8K-T": ["GW-M8K-T"], "V1-2K": ["L-U2K", "GW-U2K-O512"],
         "V1-8K": ["L-U8K", "GW-U8K-O512"], "PACE-2K": ["P-U2K"], "PACE-8K": ["P-U8K"], "W1": ["W-90", "W-50"]}
VETOES = {"PACE-2K": {"control": "F-U2K-var", "paced": "P-U2K-var"}}   # F-U2K-var PASS, P-U2K-var not: no PACE-2K
QUEUE = {"product": {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0},
         "slow": {"max_queue_wait_s": 120.0, "request_deadline_s": 600.0}}
WEIGHTED_V1 = {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
               "classes": [{"name": "v-uniform-2k", "units": 1, "max_concurrency": 32},
                           {"name": "v-uniform-8k", "units": 4, "max_concurrency": 8}]}
D5_ANALYSIS = "results/stage_d/d5-startup-wave/analysis.json"
PACING_TIERS = ("2048", "8192")
PACING_PARAMS = ("burst_tokens", "rate_tokens_per_s")                  # the gateway's [prefill_pacing]
PACING = {"source": D5_ANALYSIS, "round_to": 10,
          "2048": {"workload": "U2K", "prompt": 1920, "c_kv": 35, "burst_requests": 5, "burst_tokens": 9600,
                   "rate_tokens_per_s": 5920},
          "8192": {"workload": "U8K", "prompt": 8064, "c_kv": 13, "burst_requests": 4, "burst_tokens": 32256,
                   "rate_tokens_per_s": 5800}}
HOUR_CAPS_H = {"mini": 1.0, "V-ADM": 3.5, "V-CLS": 7.0}
TOTAL_CAP_H = 12
MIN_GAP_H = 2
END_RESERVE_H = 0.06                   # the end sentinel (103 s bench + 95 s overhead)
RUN_OVERHEAD_S = {"B0": 95, "T": 110}
NONTRIVIAL = {"u_w": 0.9, "n1": 0.9}
JOIN_TOLERANCE_MS = 50
GATEWAY_DRAIN_S = 5
SENTINEL = dict(sd.SENTINEL)           # stage_d.toml's: U2K at C 25 on the Core B server, direct
SENTINEL_BENCH_EST_S = 103
DATASETS = tuple([f"v1-{t}-r{r}" for t in (2048, 8192) for r in range(1, N_RUNS + 1)]
                 + [f"v4-{m}-r{r}" for m in (90, 50) for r in range(1, N_RUNS + 1)] + ["v1-mini-8192", "v4-mini"])
VAR_DATASET = "var-2048"               # D-C's, pinned in stage_d.toml
MIX_DATASET = "mixed-8192"             # 007's 8K mix, pinned in pins.toml
# The mini (044 mini smoke): n 2, N 20 (run_stage_d.MINI_N), the sentinel at C 2 (run_capacity.MINI_SENTINEL_C), each
# point's mini C, L-U8K on v1-mini-8192 (outputs 1 / 2 / 512 in rows 5-7), GW-U8K-O512's block 1 forced to time out
# so the skip rule drops its block 2 (044 實作補記六: with n = 2 only a block-1 timeout leaves a later run of the same
# point to skip; L-U8K keeps both runs for its output_lens pins). Its bench estimates are rough N = 20 values for its
# own cap check only; the dummy SLO, N, the forced timeout and the forced cap stop before the last planned V boot are
# the runner's.
MINI = {"n": 2, "N": 20, "sentinel_C": 2,
        "C": {"GW-U8K-O512": 4, "L-U8K": 4, "GW-M8K-T": 2, "P-U2K": 8, "F-U2K-var": 4, "W-mini": 9},
        "datasets": {"L-U8K": "v1-mini-8192"},
        "forced_timeout": {"GW-U8K-O512": [1]},
        "bench_est_s": {"GW-U8K-O512": 90, "L-U8K": 60, "GW-M8K-T": 60, "P-U2K": 20, "F-U2K-var": 15, "W-mini": 30,
                        "sentinel": 20}}
TOP_KEYS = {"schema_version", "freeze", "stage_v", "slo", "secondary_slo"}
FREEZE_KEYS = {"stage_v", "mini", "gateway_path_sha256"}
STAGE_V_KEYS = {"seed", "n", "guard_band", "stage_d_sha256", "sessions", "hour_caps_h", "total_cap_h", "min_gap_h",
                "run_overhead_s", "nontrivial", "join_tolerance_ms", "gateway_drain_s", "gateway_path_files", "points",
                "queue", "weighted_v1", "pacing", "datasets", "units", "sentinel"}
POINT_KEYS = {*POINTS["GW-U32K"], "profile_sha256"}
HARNESS_FILES = ("benchmarks/stage_v_plan.py", "benchmarks/stage_v_datasets.py", "benchmarks/stage_v_analysis.py",
                 "benchmarks/run_stage_v.py", "benchmarks/stage_v_report.py")
PACKAGE = "capserve"
PACKAGE_DIR = ("service", PACKAGE)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SESSION_ID = re.compile(r"smoke-v-[A-Za-z0-9._-]+")
_GATEWAY_FILE = re.compile(r"service/capserve/[A-Za-z0-9_]+\.py")
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


class PlanError(bp.PlanError):
    pass


def _sha(value) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


# ---------------------------------------------------------------- stage_v.toml (044)

def _point_problems(points) -> list:
    problems = []
    if not cp._keys(problems, "[stage_v.points]", points, set(POINTS)):
        return problems
    for p in sorted(points):
        x = points[p]
        if p not in POINTS or not cp._keys(problems, f"[stage_v.points.{p}]", x, POINT_KEYS):
            continue
        for key, value in POINTS[p].items():
            cp._expect(problems, f"points.{p}.{key}", x.get(key), value, "044")
        if not _sha(x.get("profile_sha256")):
            problems.append(f"points.{p}.profile_sha256 {x.get('profile_sha256')!r} is not a sha256")
    return problems


def _dataset_problems(datasets, pins: dict, stage_d_cfg: dict) -> list:
    problems = []
    if not cp._keys(problems, "[stage_v.datasets]", datasets, {*DATASETS, VAR_DATASET, MIX_DATASET}):
        return problems
    for key in DATASETS:
        if not (datasets.get(key) == "" or _sha(datasets.get(key))):
            problems.append(f"datasets.{key} {datasets.get(key)!r} is not a sha256 (or \"\" before the datasets)")
    var = (((stage_d_cfg or {}).get("stage_d") or {}).get("var_dataset_sha256") or {}).get("2048")
    if not (_sha(datasets.get(VAR_DATASET)) and datasets.get(VAR_DATASET) == var):
        problems.append(f"datasets.{VAR_DATASET} {datasets.get(VAR_DATASET)!r} is not stage_d.toml's "
                        f"var_dataset_sha256 2048 {var!r} (044 V2: D-C's sequence)")
    mix = ((pins.get("core_b") or {}).get("mix_dataset_sha256") or {}).get("8192")
    if not (_sha(datasets.get(MIX_DATASET)) and datasets.get(MIX_DATASET) == mix):
        problems.append(f"datasets.{MIX_DATASET} {datasets.get(MIX_DATASET)!r} is not pins core_b "
                        f"mix_dataset_sha256 8192 {mix!r} (044 V-GW: 007's 8K mix)")
    return problems


def stage_v_problems(cfg: dict, pins: dict, stage_d_cfg: dict) -> list:
    """stage_v.toml against the 044 rules and the frozen stage_d.toml it shares the SLO, the sentinel and var-2048
    with; [] when build_plan may use it."""
    problems = []
    if not cp._keys(problems, "stage_v.toml", cfg, TOP_KEYS):
        return problems
    cp._expect(problems, "schema_version", cfg.get("schema_version"), SCHEMA_VERSION, "stage_v.toml schema")
    freeze = cfg.get("freeze")
    if cp._keys(problems, "[freeze]", freeze, FREEZE_KEYS):
        if type(freeze.get("stage_v")) is not bool:
            problems.append(f"freeze.stage_v {freeze.get('stage_v')!r} is not a bool")
        if not isinstance(freeze.get("mini"), str):
            problems.append(f"freeze.mini {freeze.get('mini')!r} is not a string")
        if not (freeze.get("gateway_path_sha256") == "" or _sha(freeze.get("gateway_path_sha256"))):
            problems.append(f"freeze.gateway_path_sha256 {freeze.get('gateway_path_sha256')!r} is not a sha256 (or "
                            "\"\" before the freeze)")
    v = cfg.get("stage_v")
    sd_cfg = stage_d_cfg or {}
    if cp._keys(problems, "[stage_v]", v, STAGE_V_KEYS):
        want = {"seed": SEED, "n": N_RUNS, "guard_band": GUARD_BAND, "stage_d_sha256": STAGE_D_SHA256,
                "sessions": SESSIONS, "hour_caps_h": HOUR_CAPS_H, "total_cap_h": TOTAL_CAP_H, "min_gap_h": MIN_GAP_H,
                "run_overhead_s": RUN_OVERHEAD_S, "nontrivial": NONTRIVIAL, "join_tolerance_ms": JOIN_TOLERANCE_MS,
                "gateway_drain_s": GATEWAY_DRAIN_S, "queue": QUEUE, "weighted_v1": WEIGHTED_V1, "pacing": PACING,
                "units": UNITS, "sentinel": SENTINEL}
        for key, value in want.items():
            cp._expect(problems, key, v.get(key), value, "044")
        cp._expect(problems, "sentinel", v.get("sentinel"), (sd_cfg.get("stage_d") or {}).get("sentinel"),
                   "044: stage_d.toml's")
        files = v.get("gateway_path_files")
        if not (isinstance(files, list) and files and all(isinstance(f, str) and _GATEWAY_FILE.fullmatch(f)
                                                          for f in files) and files == sorted(set(files))):
            problems.append(f"gateway_path_files {files!r} must name the request path's files, sorted, once each "
                            "(044: the AST closure of capserve/__main__.py)")
        problems += _point_problems(v.get("points"))
        problems += _dataset_problems(v.get("datasets"), pins, sd_cfg)
    cp._expect(problems, "[slo]", cfg.get("slo"), cp.SLO_S2, "037 §3: S2, the user's Stage C choice")
    cp._expect(problems, "[slo]", cfg.get("slo"), sd_cfg.get("slo"), "044: stage_d.toml's")
    cp._expect(problems, "[secondary_slo]", cfg.get("secondary_slo"), cp.SECONDARY_SLO, "030 8.2: descriptive")
    cp._expect(problems, "[secondary_slo]", cfg.get("secondary_slo"), sd_cfg.get("secondary_slo"),
               "044: stage_d.toml's")
    if isinstance(freeze, dict) and freeze.get("stage_v") is True:
        if not (isinstance(freeze.get("mini"), str) and _SESSION_ID.fullmatch(freeze["mini"])):
            problems.append(f"freeze.stage_v = true needs freeze.mini naming a smoke-v-* session, not "
                            f"{freeze.get('mini')!r}")
        if not _sha(freeze.get("gateway_path_sha256")):
            problems.append("freeze.stage_v = true needs freeze.gateway_path_sha256, the frozen request path")
        datasets = v.get("datasets") if isinstance(v, dict) and isinstance(v.get("datasets"), dict) else {}
        missing = [k for k in DATASETS if not _sha(datasets.get(k))]
        if missing:
            problems.append(f"freeze.stage_v = true needs every dataset pinned, not {missing}")
    return problems


def profile_problems(cfg: dict, root=ROOT) -> list:
    """Each point's profile under root against its pinned sha256 (044: the bytes the point serves)."""
    problems = []
    for p, x in sorted(cfg["stage_v"]["points"].items()):
        path = Path(root) / x["profile"]
        try:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            problems.append(f"points.{p}: no readable profile {x['profile']}")
            continue
        if got != x["profile_sha256"]:
            problems.append(f"points.{p}: {x['profile']} sha256 {got} != pinned {x['profile_sha256']} (044)")
    return problems


def harness_sha256(root=ROOT) -> str:
    """sha256 over HARNESS_FILES (path and bytes): the harness a mini smoke ran."""
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def mini_dataset_keys() -> list:
    """The datasets the mini reads (its points' datasets, MINI's own for L-U8K), sorted."""
    return sorted({MINI["datasets"].get(p, POINTS[p]["dataset"]) for p in SESSIONS["mini"]} - {""})


def freeze_evidence_problems(cfg: dict, root) -> list:
    """Once freeze.stage_v is true: the named mini exists under root/results/stage_v as a passed non-formal mini smoke
    of the current harness (harness_sha256) that ran the frozen request path (gateway_path_sha256) and served and read
    the bytes stage_v.toml pins now: each mini point's profile_sha256 and each dataset it read (044 實作補記六). []
    before."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("stage_v") is not True:
        return []
    sid = freeze.get("mini")
    path = Path(root).joinpath(*RESULTS, str(sid), "session.json")
    try:
        s = json.loads(path.read_text())
    except (OSError, ValueError):
        return [f"freeze.mini: no readable {path}"]
    problems = []
    if (s.get("mode"), s.get("session"), s.get("formal"), (s.get("smoke") or {}).get("ok")) != ("mini", "mini", False,
                                                                                                True):
        problems.append(f"freeze.mini: {sid} is not a passed non-formal mini smoke")
    if s.get("harness_sha256") != harness_sha256():      # the code running now, wherever root is
        problems.append(f"freeze.mini: {sid} ran another harness (harness_sha256 {s.get('harness_sha256')!r})")
    if s.get("gateway_path_sha256") != freeze.get("gateway_path_sha256"):
        problems.append(f"freeze.mini: {sid} ran request path {s.get('gateway_path_sha256')!r}, not the frozen "
                        f"gateway_path_sha256 {freeze.get('gateway_path_sha256')!r}")
    v = cfg.get("stage_v") or {}
    pins = {p: ((v.get("points") or {}).get(p) or {}).get("profile_sha256") for p in SESSIONS["mini"]}
    if s.get("profile_sha256") != pins:
        problems.append(f"freeze.mini: {sid} served profile_sha256 {s.get('profile_sha256')!r}, not the pinned {pins} "
                        "(044 實作補記六: a profile changed after the mini voids its evidence)")
    data = {k: (v.get("datasets") or {}).get(k) for k in mini_dataset_keys()}
    if s.get("dataset_sha256") != data:
        problems.append(f"freeze.mini: {sid} read dataset_sha256 {s.get('dataset_sha256')!r}, not the pinned {data} "
                        "(044 實作補記六)")
    return problems


# ---------------------------------------------------------------- the request path (044 "gateway_path_files")

def _imports(path: Path, package_dir: Path) -> set:
    """The capserve modules the module at path imports (names, e.g. "config"), from its source."""
    found = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE:
                    found.add(parts[1] if len(parts) > 1 else "__init__")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            parts = (node.module or "").split(".")
            if parts[0] != PACKAGE:
                continue
            module = parts[1] if len(parts) > 1 else None
        elif node.level == 1:                    # from . import x / from .x import y (a flat package)
            module = node.module.split(".")[0] if node.module else None
        else:                                    # beyond the package: not capserve
            continue
        if module is not None:
            found.add(module)
        else:                                    # from capserve import a, b: modules of the package, or __init__ names
            found.add("__init__")
            found |= {a.name for a in node.names if (package_dir / f"{a.name}.py").is_file()}
    return found


def gateway_path_files(root=ROOT) -> list:
    """The checkout-relative paths (sorted) of the files `python -m capserve serve` can import from capserve under
    root: the import closure of capserve/__main__.py (and capserve/__init__.py, which python -m runs first), every
    import statement counted, inside functions too; other packages are not followed."""
    package_dir = Path(root).joinpath(*PACKAGE_DIR)
    seen, todo = set(), ["__init__", "__main__"]
    while todo:
        name = todo.pop()
        path = package_dir / f"{name}.py"
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        todo += sorted(_imports(path, package_dir) - seen)
    return sorted("/".join(PACKAGE_DIR) + f"/{name}.py" for name in seen)


def gateway_path_sha256(root=ROOT) -> str:
    """sha256 over gateway_path_files (path, NUL, bytes, NUL each, sorted): the request path Stage V freezes."""
    h = hashlib.sha256()
    for rel in gateway_path_files(root):
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def gateway_path_problems(cfg: dict, root=ROOT) -> list:
    """stage_v.toml's request path against root's: the listed files are its closure, and a frozen
    gateway_path_sha256 is its sha (044: checked when a session starts, and by the freeze test until the last V
    session; never by build_plan, so a report still rebuilds a plan after the request path moves on)."""
    files = gateway_path_files(root)
    problems = []
    if cfg["stage_v"]["gateway_path_files"] != files:
        problems.append(f"gateway_path_files {cfg['stage_v']['gateway_path_files']} != the request path {files} of "
                        f"{root} (044: the AST closure of capserve/__main__.py)")
    frozen = cfg["freeze"]["gateway_path_sha256"]
    if frozen and gateway_path_sha256(root) != frozen:
        problems.append(f"gateway_path_sha256 {gateway_path_sha256(root)} of {root} != the frozen {frozen} (044 凍結: "
                        "the request path changed; a new freeze, mini and session)")
    return problems


# ---------------------------------------------------------------- pacing (044 V2)

def _slope(points: list) -> Fraction:
    """The least-squares slope of y on x, exactly."""
    n = len(points)
    mx = Fraction(sum(x for x, _ in points), n)
    my = sum(y for _, y in points) / n
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)


def pacing_params(d5: dict, pins: dict) -> dict:
    """044 V2's pacing per tier from D5's analysis.json (the committed c1b first waves): the least-squares slope of
    the mean largest first-wave TTFT over the points C <= C_KV, rate = round_to x floor(prompt / slope / round_to)
    tok/s, burst = k prompts, k = floor((1 - m) x the tier's S2 TTFT / slope)."""
    out = {}
    for tier in PACING_TIERS:
        rule = PACING[tier]
        c_kv = cp.c_kv_for(pins, rule["prompt"] + bp.N_GEN)
        points = [(int(C), Fraction(str(x["mean_ms"]))) for C, x in d5["workloads"][rule["workload"]]["points"].items()
                  if int(C) <= c_kv]
        if len({C for C, _ in points}) < 2:
            raise PlanError(f"D5 {rule['workload']}: fewer than two points with C <= C_KV {c_kv}: no slope (044 V2)")
        slope = _slope(points)                                          # ms per request
        if slope <= 0:
            raise PlanError(f"D5 {rule['workload']}: slope {float(slope)} ms is not positive (044 V2)")
        step = PACING["round_to"]
        k = math.floor((1 - Fraction(str(GUARD_BAND))) * cp.SLO_S2["ttft_p95_ms"][tier] / slope)
        out[tier] = {"workload": rule["workload"], "prompt": rule["prompt"], "c_kv": c_kv, "burst_requests": k,
                     "burst_tokens": k * rule["prompt"],
                     "rate_tokens_per_s": step * math.floor(rule["prompt"] * 1000 / slope / step),
                     "slope_ms": float(slope), "points": len(points)}
    return out


def pacing_problems(cfg: dict, root, pins: dict) -> list:
    """stage_v.toml's pinned pacing against pacing_params of root's committed D5 file."""
    path = Path(root) / PACING["source"]
    try:
        got = pacing_params(json.loads(path.read_text()), pins)
    except (OSError, ValueError, KeyError, TypeError, PlanError) as exc:
        return [f"pacing: cannot recompute from {path}: {exc}"]
    problems = []
    for tier in PACING_TIERS:
        want = {k: got[tier][k] for k in PACING[tier]}
        pinned = cfg["stage_v"]["pacing"].get(tier)
        if pinned != want:
            problems.append(f"pacing.{tier} {pinned!r} != {want!r} recomputed from {PACING['source']} (044 V2)")
    return problems


# ---------------------------------------------------------------- hours (044 停止規則與時數上限)

def cap_allows(elapsed_s: float, est_s: int, cap_h: float) -> bool:
    """A V boot starts only if the session's elapsed time plus the point's estimate stays within its cap less
    END_RESERVE_H (the end sentinel always runs), exactly."""
    return Fraction(elapsed_s) + est_s <= (Fraction(str(cap_h)) - Fraction(str(END_RESERVE_H))) * 3600


def total_cap_allows(used_h: float, session_cap_h: float, total_cap_h=TOTAL_CAP_H) -> bool:
    """A V session starts only if the hours used by every V session so far plus its cap stay within the total."""
    return Fraction(str(used_h)) + Fraction(str(session_cap_h)) <= total_cap_h


def min_gap_ok(previous_finished_at: str, started_at: str, min_gap_h=MIN_GAP_H) -> bool:
    """At least min_gap_h hours between the previous formal V session's finished_at and this start (ISO times)."""
    gap = dt.datetime.fromisoformat(started_at) - dt.datetime.fromisoformat(previous_finished_at)
    return gap >= dt.timedelta(hours=min_gap_h)


# ---------------------------------------------------------------- tags, argv, gateway settings

def run_tag(phase: str, point: str, C: int, *, block: int | None = None, position: str | None = None) -> str:
    """derived/<tag>-requests.jsonl names: V-<point>-c<C>-r<block> for a Stage V point, and the sentinel's
    stage_d_plan.run_tag sentinel-<pos>-U2K-c<C>."""
    if phase == "V" and point in POINTS and type(block) is int and position is None:
        return f"V-{point}-c{C}-r{block}"
    if phase == "sentinel" and point == SENTINEL["workload"] and position in ("start", "end") and block is None:
        return sd.run_tag(phase, point, C, position=position)
    raise PlanError(f"no run tag for phase {phase!r} point {point!r} block {block!r} position {position!r}")


def bench_argv(plan: dict, pins: dict, point: str, *, block: int, result_dir: str, result_filename: str,
               datasets: dict | None = None, vllm_bin: str = "{vllm}") -> list:
    """The client of one run of point in block (datasets: {key: path} for the point's dataset): the frozen
    stage_d_plan.bench_argv_for of its client base with only 044's three differences (the gateway's port, an L
    point's --custom-output-len -1, --request-id-prefix <tag>-)."""
    x = plan["points"].get(point)
    if x is None:
        raise PlanError(f"{point!r} is not a point of the {plan['session']} plan")
    if not (type(block) is int and 1 <= block <= x["n"]):
        raise PlanError(f"{point}: block {block!r} is not in 1..{x['n']}")
    key, path = (x["datasets"] or {}).get(str(block)), None
    if key is not None:
        path = (datasets or {}).get(key)
        if not path:
            raise PlanError(f"{point} block {block} needs dataset {key} (--dataset {key}=<file>, 044)")
    argv = sd.bench_argv_for(pins, x["client_base"], C=x["C"], result_dir=result_dir, result_filename=result_filename,
                             dataset=path, vllm_bin=vllm_bin)
    i = argv.index("--base-url") + 1
    if argv[i] == f"http://127.0.0.1:{PORT}":
        argv[i] = f"http://127.0.0.1:{GATEWAY_PORT}"
    elif argv[i] != f"http://127.0.0.1:{GATEWAY_PORT}":
        raise PlanError(f"{point}: the frozen client targets {argv[i]}, neither vLLM nor the gateway")
    if x["out_pinned"]:
        j = argv.index("--custom-output-len") + 1
        if argv[j] != str(bp.N_GEN):
            raise PlanError(f"{point}: the frozen client's --custom-output-len is {argv[j]}, not {bp.N_GEN}")
        argv[j] = "-1"                                  # each row's pinned output_tokens (vLLM 0.30 CustomDataset)
    return argv + ["--request-id-prefix", run_tag("V", point, x["C"], block=block) + "-"]


def gateway_settings(plan: dict, pins: dict, point: str) -> dict:
    """The point's `capserve serve` settings: D-C's (039 implementation note 7) with its queue settings and its class,
    or weighted-v1's [admission] instead of a class, and its [prefill_pacing] (044)."""
    x = plan["points"][point]
    doc = {} if x["admission"] else {"active_class": x["classes"][0]}
    doc.update(backend_url=f"http://127.0.0.1:{PORT}", host="127.0.0.1", port=plan["gateway_port"],
               drain_timeout_s=plan["gateway_drain_s"], tokenizer="local", tokenizer_path=bp.snapshot_dir(pins),
               **x["queue_settings"])
    if x["admission"]:
        doc["admission"] = copy.deepcopy(x["admission_params"])
    if x["pacing"]:
        doc["prefill_pacing"] = dict(x["pacing_params"])
    return doc


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _toml_str(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04X}")
        elif 0xD800 <= o <= 0xDFFF:
            raise PlanError(f"cannot write a lone surrogate {o:#x} in a TOML string")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_key(key) -> str:
    if not isinstance(key, str):
        raise PlanError(f"TOML keys are strings, not {key!r}")
    return key if _BARE_KEY.fullmatch(key) else _toml_str(key)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanError(f"cannot write {value!r} as a TOML value")
        return repr(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise PlanError(f"cannot write {type(value).__name__} values as TOML")


def toml_dumps(doc: dict) -> str:
    """TOML text that tomllib reads back as doc: at each level its plain keys (scalars, and arrays whose tables are
    inline), then its tables under [dotted.headers], to any depth."""
    lines = []

    def table(t: dict, path: list) -> None:
        if path:
            lines.extend(([""] if lines else []) + ["[" + ".".join(_toml_key(k) for k in path) + "]"])
        lines.extend(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in t.items() if not isinstance(v, dict))
        for k, v in t.items():
            if isinstance(v, dict):
                table(v, path + [k])

    table(doc, [])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- plan

def _plan_point(cfg: dict, stage_d_cfg: dict, point: str, *, n: int, mini: bool) -> dict:
    v = cfg["stage_v"]
    x = copy.deepcopy(v["points"][point])
    C = MINI["C"][point] if mini else x["C"]
    N = MINI["N"] if mini else bp.num_prompts(C)
    server = sd.workload(x["server_base"])["server"]
    key, per_block = x["dataset"], x["per_block"]
    if mini and point in MINI["datasets"]:
        key, per_block = MINI["datasets"][point], False
    bench = MINI["bench_est_s"][point] if mini else x["bench_est_s"]
    overhead = v["run_overhead_s"][server]
    x.update(C=C, N=N, n=n, server=server, dataset=key, per_block=per_block,
             timeout_s=sd.timeout_s(sd.tau1(stage_d_cfg, x["tau_base"]), N), bench_est_s=bench,
             run_overhead_s=overhead, est_s=bench + overhead, judged_views=list(JUDGED_VIEWS[x["view"]]),
             queue_settings=dict(v["queue"][x["queue"]]),
             admission_params=copy.deepcopy(v["weighted_v1"]) if x["admission"] else None,
             pacing_params={k: v["pacing"][x["pacing"]][k] for k in PACING_PARAMS} if x["pacing"] else None,
             datasets={str(r): f"{key}-r{r}" if per_block else key for r in range(1, n + 1)} if key else None,
             forced_timeout_blocks=list(MINI["forced_timeout"].get(point, [])) if mini else [])
    return x


def build_plan(cfg: dict, pins: dict, stage_d_cfg: dict, *, session_name: str, pins_sha256: str, stage_v_sha256: str,
               stage_d_sha256: str, formal: bool = False) -> dict:
    """The plan of one Stage V session (mini, V-ADM or V-CLS); formal only after the freeze, the mini never."""
    problems = stage_v_problems(cfg, pins, stage_d_cfg)
    problems += [f"stage_d.toml: {p}" for p in sd.stage_d_problems(stage_d_cfg, pins)]
    if problems:
        raise PlanError("stage_v.toml: " + "; ".join(problems))
    v = cfg["stage_v"]
    if stage_d_sha256 != v["stage_d_sha256"]:
        raise PlanError(f"stage_d.toml sha256 {stage_d_sha256} != stage_d_sha256 {v['stage_d_sha256']} pinned in "
                        "stage_v.toml (044: the frozen stage_d.toml's tau1, var-2048, SLO and sentinel)")
    if session_name not in SESSIONS:
        raise PlanError(f"unknown session {session_name!r}; 044 defines {sorted(SESSIONS)}")
    mini = session_name == "mini"
    if formal and mini:
        raise PlanError("the mini smoke is never a formal session (044 mini)")
    if formal and cfg["freeze"]["stage_v"] is not True:
        raise PlanError("a formal plan needs freeze.stage_v = true (044 凍結)")
    n = MINI["n"] if mini else v["n"]
    points = {p: _plan_point(cfg, stage_d_cfg, p, n=n, mini=mini) for p in v["sessions"][session_name]}
    s = v["sentinel"]
    sc = MINI["sentinel_C"] if mini else s["C"]
    sn = MINI["N"] if mini else bp.num_prompts(sc)
    s_server = sd.workload(s["workload"])["server"]
    s_bench = MINI["bench_est_s"]["sentinel"] if mini else SENTINEL_BENCH_EST_S
    sentinel = {"workload": s["workload"], "L": sd.kv_len(s["workload"]), "mix": "uniform", "C": sc, "N": sn,
                "tolerance": s["tolerance"], "timeout_s": sd.timeout_s(sd.tau1(stage_d_cfg, s["workload"]), sn),
                "server": s_server, "bench_est_s": s_bench, "run_overhead_s": v["run_overhead_s"][s_server],
                "est_s": s_bench + v["run_overhead_s"][s_server]}
    units = {u: list(pts) for u, pts in v["units"].items() if set(pts) <= set(points)}
    used = sorted({k for x in points.values() for k in (x["datasets"] or {}).values()})
    sb = pins["stage_b"]
    plan = {"schema_version": SCHEMA_VERSION, "stage": STAGE, "session": session_name, "kind": KIND,
            "mini": copy.deepcopy(MINI) if mini else None, "seed": v["seed"], "n": n, "guard_band": v["guard_band"],
            "slo": cfg["slo"], "secondary_slo": cfg["secondary_slo"], "window": cp.WINDOW,
            "nontrivial": v["nontrivial"], "join_tolerance_ms": v["join_tolerance_ms"], "points": points,
            "block_orders": {str(r): cp.block_order(v["seed"], r, list(points)) for r in range(1, n + 1)},
            "units": units, "vetoes": {u: dict(x) for u, x in VETOES.items()
                                       if u in units and set(x.values()) <= set(points)},
            "sentinel": sentinel, "hour_cap_h": v["hour_caps_h"][session_name], "total_cap_h": v["total_cap_h"],
            "min_gap_h": v["min_gap_h"], "end_reserve_h": END_RESERVE_H,
            "estimate_s": sum(x["n"] * x["est_s"] for x in points.values()) + 2 * sentinel["est_s"],
            "gateway_port": GATEWAY_PORT, "gateway_drain_s": v["gateway_drain_s"],
            "gateway_path_files": v["gateway_path_files"], "gateway_path_sha256": cfg["freeze"]["gateway_path_sha256"],
            "datasets": {k: v["datasets"][k] for k in used}, "server_ready_timeout_s": cp.SERVER_READY_TIMEOUT_S,
            "expected_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"],
            "gpu_memory_utilization": sb["gpu_memory_utilization"], "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"],
            "long_prefill_token_threshold": pins["advanced"]["long_prefill_token_threshold"],
            "pins_sha256": pins_sha256, "stage_v_sha256": stage_v_sha256, "stage_d_sha256": stage_d_sha256}
    for p, x in points.items():
        x["gateway"] = gateway_settings(plan, pins, p)
        x["gateway_toml_sha256"] = hashlib.sha256(toml_dumps(x["gateway"]).encode()).hexdigest()
    return plan


def plan_sha256(plan: dict) -> str:
    return sha256_json(plan)


def load_stage_v_bytes() -> bytes:
    return STAGE_V_PATH.read_bytes()


def main(argv: list | None = None) -> int:
    import argparse
    import tomllib

    parser = argparse.ArgumentParser(description="Stage V plan (decision 044)")
    parser.add_argument("--session", required=True, choices=sorted(SESSIONS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--formal", action="store_true", help="refuse unless freeze.stage_v = true")
    args = parser.parse_args(argv)
    cfg_raw, sd_raw, pins_raw = load_stage_v_bytes(), sd.load_stage_d_bytes(), cp.load_pins_bytes()
    cfg, sd_cfg, pins = (tomllib.loads(raw.decode()) for raw in (cfg_raw, sd_raw, pins_raw))
    try:
        plan = build_plan(cfg, pins, sd_cfg, session_name=args.session,
                          pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                          stage_v_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                          stage_d_sha256=hashlib.sha256(sd_raw).hexdigest(), formal=args.formal)
    except PlanError as e:
        parser.error(str(e))
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({p: {k: x[k] for k in ("C", "N", "n", "timeout_s", "est_s")} for p, x in plan["points"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
