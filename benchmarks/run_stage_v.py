#!/usr/bin/env python3
"""Stage V session runner (decision 044; PREREGISTRATION §12). One process, under a GPU lock it already holds, runs one
session of stage_v_plan (the mini, V-ADM or V-CLS). Core, Stage C and Stage D modules are imported, never changed:

- the per-boot and per-run steps are run_stage_d's StageDHarness (itself run_capacity's _Harness: the lock check, 031's
  idle window and two-stage spill guard, env capture, run_adv.start_server and shutdown_group, the bench with its
  timeout and 1 Hz sampler, D-C's gateway stop); StageVHarness overrides only what 044 changes, because
  stage_d_plan.workload / run_tag and stage_d_analysis.spec raise on a Stage V point:
  - boot: the point's server_base (B0, or T for GW-M8K-T) through stage_d_plan.server_argv_for and
    run_stage_d.server_checks; every V boot runs the gateway, the sentinel never does; the boot records elapsed_s, the
    session time at its hour-cap check;
  - gateway_up: `python -m capserve serve` (a subprocess: research code never imports capserve) with the point's
    rendered settings (stage_v_plan.toml_dumps of plan["points"][p]["gateway"]) and its profile,
    --allow-provisional exactly on a provisional point (gateway_argv); its /readyz must be the point's frozen tuple
    (readyz_want: profile id and status, serving status, class or none (weighted-v1), limits, queue settings,
    admission and pacing parameters, gateway_path_sha256), else the boot fails before its first bench;
  - _row / client_argv: tags V-<point>-c<C>-r<b>, the plan's N and timeout (the mini's forced 5 s timeout), the client
    stage_v_plan.bench_argv (the frozen stage_d_plan.bench_argv_for with 044's three differences);
  - bench: run_stage_d.summarize for a fixed-output client; an L or W point's every input and output length is its
    block's dataset row's (044 (k), (l); stage_v_analysis.planned);
  - run_point: bench, the gateway still up, gateway_down (SIGTERM, SIGKILL after 30 s, under run_adv.DeferredSignals;
    its log reader joins, so the access log is complete), the access log joined to the run's N requests
    (stage_v_analysis.join_access_log on the log without the reader's stamps, access_log_text: 044 (h), (i)), the
    derived requests with the gateway's fields (stage_v_analysis.gateway_requests) in derived/<tag>-requests.jsonl,
    so the mini's run_capacity.mask_smoke keeps only idx / cls / out of them too;
  - decide: stage_v_analysis.run_outcome; the derived requests' checks, (j) and E (stage_v_analysis.run_valid) fold
    into the runner's verdict, so the row says invalid whenever the replay will.
- _v_machine (044 Sessions): sentinel; blocks r = 1..n in the plan's block orders, one boot and one V run per active
  point; before each V boot the hour cap (stage_v_plan.cap_allows on the elapsed time; the mini also stops before its
  last planned V boot, forced): a stop is session["cap_stop"] = {block, point, elapsed_s, forced} and every later run
  is not run (no row); once a run is not ok (invalid, fail or trivial), its point alone leaves the later boots (the
  inherited 039 / 030 rule, per point, for every unit: 044 實作補記六); sentinel (it runs after a cap stop too).
- Gateway faults (044: a gateway-caused invalid run stops the next V session until a decision log): a gateway that
  failed its boot checks, exited, whose log reader did not join, whose access log does not join, or whose run broke (j)
  is listed in session["gateway_faults"].

Refusals before any server starts (RunError): the output is not results/stage_v/formal-<id> (formal) or
results/stage_v/smoke-v-<id> (mini), or it exists; no held GPU lock; formal without freeze.stage_v = true; a mode that
does not run its plan (the mini runs only the mini plan); stage_v.toml breaks 044 or the plan is not
stage_v_plan.build_plan of these stage_v.toml / stage_d.toml / pins bytes; the request path is not the frozen one
(stage_v_plan.gateway_path_problems) or a point's profile not its pinned bytes (profile_problems); a dataset the
session reads missing or not its pinned sha256 (the mini's too: 044 pins them before it); the 12 h total over every
Stage V session's recorded hours (hours_used) plus this session's cap, or a formal-* / smoke-v-* directory without a
finished session.json (a session killed hard: its hours unknown); formal: a previous formal V session finished less
than 2 h ago, or one whose end is unknown (gap_problems), Stage B, inherited engine variables, the server env, the
freeze evidence (the passed mini of this harness on the frozen request path), and, after the session directory exists
(its session.json is then finished as aborted), the preflight's deployed clean tree and pinned vLLM. From its mkdir on
a session has a session.json: provisional (in_progress, finished_at null), finished on every exit Python sees (044
實作補記六).

Modes: --formal, or --mini (044 mini smoke before the freeze: the mini plan with run_capacity's dummy SLO and N = 20;
its verdict, mini_checks, goes to smoke.json and every latency is discarded, as run_capacity's: run_capacity.mask_smoke,
then its raw/*.gateway.log, whose sha256 the raw manifest keeps; 044 實作補記六).

Usage (WSL, ~/llm-serving-lab, the plan from benchmarks/stage_v_plan.py):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_stage_v.py --plan stage-v-plan.json \\
        --out results/stage_v/<session> --dataset var-2048=data/stage_d/var-2048.jsonl \\
        --dataset v4-90-r1=data/stage_v/v4-90-r1.jsonl ... --gpu-lock-held (--formal | --mini)
Exit: 0 a formal session that ran or a passed smoke, 1 a failed smoke, 2 refused or aborted.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
import re
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import coreb_smoke as smoke
import gpu_idle as gi
import run_adv as ra
import run_capacity as rc
import run_stage_d as rd
import stage_d_plan as sd
import stage_v_analysis as sva
import stage_v_plan as sv
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
MODES = ("formal", "mini")
PORT = sv.PORT
GATEWAY_PORT = sv.GATEWAY_PORT
ROOT = sv.ROOT
GATEWAY_READY_S = rd.GATEWAY_READY_S
MINI_SLO = rc.MINI_SLO
MINI_FORCED_TIMEOUT_S = rc.MINI_FORCED_TIMEOUT_S
SESSION_IDS = {"formal": re.compile(r"formal-[A-Za-z0-9._-]+"), "mini": re.compile(r"smoke-v-[A-Za-z0-9._-]+")}
LIMIT_KEYS = ("max_concurrency", "max_queue", "max_prompt_tokens", "max_output_tokens")
ADMISSION_KEYS = ("policy", "budget", "max_queue")
ADMISSION_CLASS_KEYS = ("name", "units", "max_concurrency")
QUEUE_KEYS = ("max_queue_wait_s", "request_deadline_s")
READYZ_KEYS = ("profile", "serving_status", "active_class", "limits", "queue", "admission", "prefill_pacing",
               "gateway_path_sha256")             # 044 boot: the /readyz tuple of a point
_STAMP = re.compile(r"^\d+ ", re.M)             # run_adv.pump_log: int(wall() * 1000) and a space before each line


class RunError(rd.RunError):
    pass


def output_problems(out_dir, *, mode: str, repo_root=ROOT) -> list:
    """[] when out_dir is results/stage_v/formal-<id> (formal) or results/stage_v/smoke-v-<id> (mini) under repo_root:
    the 2 h gap reads formal-*, the freeze names a smoke-v-* mini (044); so never anything under a runs/ directory
    (027 §3: product-test outputs are never research data)."""
    root = Path(repo_root).resolve().joinpath(*sv.RESULTS)
    out = Path(out_dir).resolve()
    if out == root or root not in out.parents:
        return [f"{out_dir} is not a session directory under results/stage_v/ (044: Stage V writes only there)"]
    parts = out.relative_to(root).parts
    if len(parts) != 1 or not SESSION_IDS[mode].fullmatch(parts[0]):
        want = "formal-<id>" if mode == "formal" else "smoke-v-<id>"
        return [f"{out_dir}: a {mode} Stage V session is results/stage_v/{want} (044: the 2 h gap reads formal-*, the "
                "freeze names a smoke-v-* mini)"]
    return []


# ---------------------------------------------------------------- the gateway (044 boot, (h))

def gateway_argv(python: str, config, profile, *, provisional: bool) -> list:
    """`python -m capserve serve` of one point: --allow-provisional exactly on a provisional point (044)."""
    return [python, "-m", "capserve", "serve", "--config", str(config), "--profile", str(profile)] + (
        ["--allow-provisional"] if provisional else [])


def gateway_env(base_env: dict) -> dict:
    """D-C's gateway env (039 implementation note 7): no CAPSERVE_* variable, the checkout's service and benchmarks on
    the path, the HF hub offline."""
    return {**{k: v for k, v in base_env.items() if not k.startswith("CAPSERVE_")},
            "PYTHONPATH": os.pathsep.join([str(ROOT / "service"), str(ROOT / "benchmarks")]),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}


def readyz_want(plan: dict, point: str, *, profile_id: str, gateway_path_sha256: str) -> dict:
    """The /readyz tuple a point's gateway must report (044 boot): its profile's id and status, the serving status,
    its class (weighted-v1: none), the limits, the queue settings, weighted-v1's and the pacing's parameters and the
    request path it runs."""
    x = plan["points"][point]
    status = "provisional" if x["provisional"] else "verified"
    adm = x["admission_params"]
    want = {"profile": {"id": profile_id, "status": status}, "serving_status": status,
            "active_class": None if x["admission"] else x["classes"][0], "limits": dict(x["limits"]),
            "queue": dict(x["queue_settings"]),
            "admission": None if adm is None else {**{k: adm[k] for k in ADMISSION_KEYS},
                                                   "classes": [{k: c[k] for k in ADMISSION_CLASS_KEYS}
                                                               for c in adm["classes"]]},
            "prefill_pacing": None if x["pacing_params"] is None else dict(x["pacing_params"]),
            "gateway_path_sha256": gateway_path_sha256}
    return {k: want[k] for k in READYZ_KEYS}


def _sub(d, keys) -> dict | None:
    return {k: d.get(k) for k in keys} if isinstance(d, dict) else None


def readyz_tuple(body) -> dict:
    """The part of a /readyz body 044 freezes, in readyz_want's shape (what the body lacks is None)."""
    b = body if isinstance(body, dict) else {}
    adm = b.get("admission")
    if isinstance(adm, dict):
        classes = adm.get("classes")
        adm = {**_sub(adm, ADMISSION_KEYS), "classes": [_sub(c, ADMISSION_CLASS_KEYS) for c in classes]
               if isinstance(classes, list) else classes}
    pacing = b.get("prefill_pacing")
    got = {"profile": _sub(b.get("profile"), ("id", "status")), "serving_status": b.get("serving_status"),
           "active_class": b.get("active_class"), "limits": _sub(b.get("limits"), LIMIT_KEYS),
           "queue": _sub(b.get("queue"), QUEUE_KEYS), "admission": adm,
           "prefill_pacing": dict(pacing) if isinstance(pacing, dict) else pacing,
           "gateway_path_sha256": b.get("gateway_path_sha256")}
    return {k: got[k] for k in READYZ_KEYS}


def access_log_text(raw: str) -> str:
    """The gateway's own output in its raw log: run_adv's log reader stamps every line with the wall clock in ms and a
    space (run_adv.pump_log); the access-log join reads the lines without it."""
    return _STAMP.sub("", raw)


# ---------------------------------------------------------------- per run (044 (k), (l))

def summarize(res: dict, *, plan: dict, row: dict) -> dict:
    """run_stage_d.summarize for the sentinel and a fixed-output client (uniform, 007's mix, var); an L or W point's:
    N completed, none failed, every input and output length its block's dataset row's (044 (k), (l))."""
    n = row["N"]
    if row["phase"] == "sentinel":
        return rd.summarize(res, x=sd.workload(row["workload"]), n=n)
    x = plan["points"][row["workload"]]
    if not (x["out_pinned"] or x["admission"]):
        return rd.summarize(res, x=sd.workload(x["client_base"]), n=n)
    want = sva.planned(plan, row["workload"], row["block"])
    reasons = []
    if res.get("completed") != n or res.get("failed"):
        reasons.append(f"completed={res.get('completed')} failed={res.get('failed')} (expected {n} / 0)")
    if res.get("input_lens") != [p for p, _ in want]:
        reasons.append(f"input_lens are not the prompt lengths of {x['datasets'][str(row['block'])]} (044 (k), (l))")
    if res.get("output_lens") != [o for _, o in want]:
        reasons.append(f"output_lens are not the output_tokens pinned in {x['datasets'][str(row['block'])]} (044 (k))")
    return {"ok": not reasons, "reasons": reasons, "num_prompts": n,
            **{k: res.get(k) for k in ("completed", "failed", "duration", "request_throughput", "output_throughput",
                                       "total_token_throughput")}}


# ---------------------------------------------------------------- hours and the gap (044 停止規則與時數上限)

class Unfinished(ValueError):
    """A session directory without a finished session.json: started, never recorded its end (044 實作補記六)."""


RECORD_IT = ("write a decision log on its hours and end (its last raw file's time, or started_at + its hour cap), "
             "record that finished_at in its session.json, then start the next session (044 實作補記六)")


def _session_dirs(repo_root, pattern: str = "*") -> list:
    """The session directories under results/stage_v matching pattern: every formal-* or smoke-v-* one (the runner
    creates it before anything can fail; a session.json may be missing after a hard kill) and any other directory
    holding a session.json; never prereg-sim or another directory of plain results."""
    root = Path(repo_root).joinpath(*sv.RESULTS)
    if not root.is_dir():
        return []
    return sorted(d for d in root.glob(pattern) if d.is_dir() and (
        any(rx.fullmatch(d.name) for rx in SESSION_IDS.values()) or (d / "session.json").exists()))


def _times(d: Path) -> tuple:
    """(started_at, finished_at, session) of a session directory; Unfinished (a ValueError) without a finished
    session.json."""
    path = d / "session.json"
    if not path.is_file():
        raise Unfinished(f"no session.json: a session that started and never finished (killed hard?); {RECORD_IT}")
    s = json.loads(path.read_text())
    if not isinstance(s, dict):
        raise Unfinished("its session.json is not a session record")
    if s.get("in_progress") is True or s.get("finished_at") is None:
        raise Unfinished(f"its session.json says it never finished (in_progress {s.get('in_progress')!r}, "
                         f"finished_at {s.get('finished_at')!r}: killed hard?); {RECORD_IT}")
    return dt.datetime.fromisoformat(s["started_at"]), dt.datetime.fromisoformat(s["finished_at"]), s


def hours_used(repo_root) -> tuple:
    """(hours as an exact Fraction, problems): finished_at - started_at summed over every Stage V session under
    results/stage_v (044 全期總上限: the mini, formal ones, reruns and aborted ones alike); a session directory
    without a finished session.json with both times is a problem (fail closed: 實作補記六)."""
    total, problems = dt.timedelta(), []
    for d in _session_dirs(repo_root):
        try:
            start, end, _ = _times(d)
            total += end - start
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{d.name}: cannot count its hours ({type(exc).__name__}: {exc}) (044 全期總上限)")
    return Fraction(total // dt.timedelta(microseconds=1), 3600 * 10 ** 6), problems


def gap_problems(repo_root, started_at: str, *, session_id: str | None = None, min_gap_h=sv.MIN_GAP_H) -> tuple:
    """(the previous formal V session {session_id, finished_at} or None, problems): among results/stage_v/formal-*
    other than session_id that started at or before started_at, the one that finished last must have finished at least
    min_gap_h hours before started_at (044: a formal session's lock window is >= 2 h after the previous one's); a
    formal-* directory without a finished session.json with both times is a problem (fail closed: 實作補記六)."""
    start, problems, prev = dt.datetime.fromisoformat(started_at), [], None
    for d in _session_dirs(repo_root, "formal-*"):
        sid = d.name
        if sid == session_id:
            continue
        try:
            begun, end, s = _times(d)
            if begun <= start and (prev is None or end > prev[1]):
                prev = (sid, end, s["finished_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{sid}: cannot tell when it finished ({type(exc).__name__}: {exc}), so not whether 2 h "
                            "have passed (044)")
    if prev is not None and not sv.min_gap_ok(prev[2], started_at, min_gap_h):
        problems.append(f"the formal V session {prev[0]} finished at {prev[2]}, less than {min_gap_h} h before "
                        f"{started_at} (044: >= {min_gap_h} h between formal V sessions)")
    return (None if prev is None else {"session_id": prev[0], "finished_at": prev[2]}), problems


# ---------------------------------------------------------------- the harness

class StageVHarness(rd.StageDHarness):
    """run_stage_d's per-boot / per-run steps with 044's points, gateway, access-log join and outcomes."""

    def __init__(self, *, t0: float, **kw):
        super().__init__(**kw)
        self.t0 = t0                          # the session's start on the monotonic clock (elapsed_s, the hour cap)

    def elapsed(self) -> float:
        return round(self.now() - self.t0, 3)

    def new_boot(self, phase: str, w: str, *, block=None, position=None, elapsed_s=None, attempt=None) -> dict:
        boot = {"boot": len(self.session["boots"]), "phase": phase, "workload": w, "block": block, "attempt": attempt,
                "position": position, "elapsed_s": elapsed_s, "reached_first_bench": None, "problems": []}
        label = f"sentinel-{position}-{w}" if phase == "sentinel" else f"{phase}-{w}-r{block}"
        boot["tag"] = f"b{boot['boot']:03d}-{label}"
        self.session["boots"].append(boot)
        return boot

    def _port_busy(self, tag: str, when: str) -> RunError:
        self.session["gateway_port_busy"] = {"boot": tag, "port": GATEWAY_PORT}
        return RunError(f"gateway_port_busy {when} {tag}: port {GATEWAY_PORT} is in use (a leftover gateway would "
                        "answer for this boot's); stopping, rerun under a new id (044, 039 D-C)")

    def _fault(self, boot_tag: str, reasons: list, run: str | None = None) -> None:
        self.session["gateway_faults"].append({"boot": boot_tag, "run": run, "reasons": list(reasons)})

    # ---------------------------------------------------------- boot
    def boot(self, boot: dict, next_point: Callable, planned: list | None = None) -> list:
        tag, w = boot["tag"], boot["workload"]
        via_gateway = boot["phase"] == "V"                # every Stage V point runs through the gateway
        base = self.plan["points"][w]["server_base"] if via_gateway else w
        if via_gateway and not self.port_free(GATEWAY_PORT):
            raise self._port_busy(tag, "before")
        idle = self.pre_boot(tag)
        self.snap(f"{tag}.before.json")
        argv, env = sd.server_argv_for(self.pins, base, port=PORT, vllm_bin=self.vllm_bin, base_env=self.base_env)
        record = self.session["runs"][tag] = {"argv": argv, "env": ra.recorded_server_env(env), "idle": idle,
                                              "server_base": base}
        proc, rows = None, []
        try:
            proc = self.popen(argv, env, self.raw / f"{tag}.server.log")
            ready, detail = self.wait_ready(proc, record)
            metrics = self.metrics_text(ready)
            facts, problems = rd.server_checks(rc._read(self.raw / f"{tag}.server.log"), metrics, ready=ready,
                                               plan=self.plan, w=base, detail=detail)
            record.update(facts)
            self.check_log(proc, "during the server start")
            if not problems and proc.poll() is not None:
                problems = [f"server exited (code {proc.poll()!r}) before its first bench (030 8.4)"]
            if not problems:
                entry = self.spill_check("ready", tag)
                if entry["spilled"]:
                    problems = [rc.spill_reason(entry, self.guard) + "; no bench runs on this boot"]
            if not problems and via_gateway:
                problems = self.gateway_up(boot, record)        # sets self.gw as soon as the process exists
                if problems:
                    self._fault(tag, problems)
            boot["problems"], boot["reached_first_bench"] = problems, not problems
            if problems:
                for C in planned or []:
                    rows.append(self.not_run(boot, C, problems))
            else:
                while (C := next_point(rows)) is not None:
                    rows.append(self.run_point(boot, proc, C))
        finally:
            try:
                if self.gw is not None:
                    self.gateway_down(self.gw, record, tag)
            finally:
                self.gw = None
                record["spill"] = self.guard.record()
                self.teardown(tag, proc, record)
        self.check_log(proc, "at teardown")
        self.snap(f"{tag}.after.json")
        return rows

    def gateway_up(self, boot: dict, record: dict) -> list:
        """Problems of the point's `python -m capserve serve` ([] when ready with the point's /readyz tuple); self.gw
        holds the process from popen on."""
        point, tag = boot["workload"], boot["tag"]
        x = self.plan["points"][point]
        text = sv.toml_dumps(x["gateway"])
        record["gateway"] = {"settings": x["gateway"], "profile": x["profile"],
                             "toml_sha256": hashlib.sha256(text.encode()).hexdigest()}
        if not self.port_free(GATEWAY_PORT):              # checked before the boot too; a race is still fatal
            raise self._port_busy(tag, "in")
        cfg_path = self.raw / f"{tag}.gateway.toml"
        cfg_path.write_text(text)
        argv = gateway_argv(self.gateway_python, cfg_path, ROOT / x["profile"], provisional=x["provisional"])
        record["gateway"]["argv"] = argv
        gw = self.gw = self.popen(argv, gateway_env(self.base_env), self.raw / f"{tag}.gateway.log")
        base = f"http://127.0.0.1:{GATEWAY_PORT}"
        t0, status, body = self.now(), 0, ""
        while self.now() - t0 < GATEWAY_READY_S and gw.poll() is None:
            status, body = self.http_get(f"{base}/readyz")
            if status == 200:
                break
            self.sleep(2)
        if status != 200:
            return [f"gateway not ready within {GATEWAY_READY_S} s (status {status}; exit {gw.poll()!r}) (044 boot)"]
        try:
            ready = json.loads(body)
        except ValueError:
            ready = {}
        got = record["gateway"]["readyz"] = readyz_tuple(ready)
        want = readyz_want(self.plan, point, profile_id=json.loads((ROOT / x["profile"]).read_text())["profile_id"],
                           gateway_path_sha256=self.session["gateway_path_sha256"])
        if got != want:
            return [f"gateway /readyz is not the point's frozen tuple{rc.dict_diff(got, want)} (044 boot)"]
        if gw.poll() is not None:
            return [f"the gateway exited (code {gw.poll()!r}) right after it was ready (044)"]
        return []

    # ---------------------------------------------------------- run
    def _row(self, boot: dict, C: int) -> dict:
        phase, w = boot["phase"], boot["workload"]
        row = {"phase": phase, "workload": w, "C": C, "block": boot["block"], "position": boot["position"],
               "boot": boot["boot"], "valid": False, "reasons": [], "timeout": False, "output_throughput": None,
               "metrics": None}
        if phase == "sentinel":
            s = self.plan["sentinel"]
            row.update(tag=sv.run_tag(phase, w, C, position=boot["position"]), N=s["N"], timeout_s=s["timeout_s"],
                       L=s["L"], mix=s["mix"])
        else:
            x = self.plan["points"][w]
            forced = boot["block"] in x["forced_timeout_blocks"]
            row.update(tag=sv.run_tag(phase, w, C, block=boot["block"]), N=x["N"],
                       timeout_s=MINI_FORCED_TIMEOUT_S if forced else x["timeout_s"],
                       dataset=(x["datasets"] or {}).get(str(boot["block"])))
        return row

    def client_argv(self, row: dict) -> list:
        kw = {"result_dir": str(self.raw), "result_filename": f"{row['tag']}.bench.json", "vllm_bin": self.vllm_bin}
        if row["phase"] == "sentinel":
            return sd.bench_argv_for(self.pins, row["workload"], C=row["C"], **kw)
        return sv.bench_argv(self.plan, self.pins, row["workload"], block=row["block"], datasets=self.datasets, **kw)

    def bench(self, row: dict, argv: list) -> tuple:
        tag, n = row["tag"], row["N"]
        row["argv"] = argv
        sampler = rc._rb().MetricsSampler(self.http_get, f"{self.base}/metrics", self.sample_interval)
        sampler.start()
        try:
            rc_, timed_out = self.run_bench(argv, self.raw / f"{tag}.bench.log", {**self.base_env, **bp.bench_env()},
                                            row["timeout_s"])
        finally:
            sampled = sampler.stop()
        row["metrics"] = {**sampled, "inflight_peak": rc.inflight_peak(sampler.samples)}
        row["healthy_after_bench"] = self.http_get(f"{self.base}/health")[0] == 200
        path, res, reasons = self.raw / f"{tag}.bench.json", None, []
        if rc_ == 0 and not timed_out and path.is_file():
            try:
                res = json.loads(path.read_text())
            except ValueError as exc:
                reasons.append(f"unreadable bench result ({exc})")
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {row['timeout_s']} s; its process group was killed "
                           "(030 8.4; 8.6 (f))")
        elif res is None and not reasons:
            reasons.append(f"vllm bench serve exit {rc_} (no result)")
        if res is not None:
            summary = summarize(res, plan=self.plan, row=row)
            reasons += summary.pop("reasons")
            row.update(summary=summary, output_throughput=summary.get("output_throughput"),
                       result=f"raw/{tag}.bench.json")
            reasons += rc.array_problems(res, N=n)
        return res, reasons

    def _after_bench(self, boot: dict, row: dict, *, read: bool) -> tuple:
        """(problems, {idx: access-log record}) of a V run after its bench (044 run_point): the gateway still up, then
        stopped (gateway_down: its log reader joins, so the log is complete), then its access log joined to the run's
        requests ((h), (i)) unless the bench gave no result."""
        tag, gw = boot["tag"], self.gw
        record = self.session["runs"][tag]
        problems = []
        if gw.poll() is not None:
            problems.append(f"the gateway exited (code {gw.poll()!r}) during the bench (044)")
        self.gateway_down(gw, record, tag)
        self.gw = None
        reader = getattr(gw, "log_reader", None)
        if reader is not None and not reader.join(0):
            problems.append("the gateway's log reader did not join: its access log may be incomplete (044 (h))")
        joined = {}
        if read:
            text = access_log_text(rc._read(self.raw / f"{tag}.gateway.log"))
            found, joined, warm = sva.join_access_log(text, plan=self.plan, point=row["workload"], block=row["block"])
            row["access_log"] = {"log": f"raw/{tag}.gateway.log", "problems": found, "joined": len(joined)}
            row["warmups"] = sva.warmup_spans(warm, joined, x=self.plan["points"][row["workload"]])   # (j): 實作補記六
            problems += found
        if problems:
            self._fault(tag, problems, run=row["tag"])
        return problems, joined

    def run_point(self, boot: dict, proc, C: int) -> dict:
        row = self._row(boot, C)
        v = row["phase"] == "V"
        if self.spilled_in is not None:
            row["reasons"] = [f"not run: the boot stopped after a spill in {self.spilled_in} (031; 030 8.6 (g))"]
            return self.decide(row, None)
        if proc.poll() is not None:
            row["reasons"] = [f"server exited (code {proc.poll()!r}) before this bench; not run (030 8.6 (b))"]
            return self.decide(row, None)
        if v and (self.gw is None or self.gw.poll() is not None):
            code = None if self.gw is None else self.gw.poll()
            row["reasons"] = [f"the gateway exited (code {code!r}) before this bench; not run (044)"]
            return self.decide(row, None)
        res, reasons = self.bench(row, self.client_argv(row))
        self.check_log(proc, "during the bench")
        reasons += self.spill_row(row)
        joined = {}
        if v:
            problems, joined = self._after_bench(boot, row, read=res is not None)
            reasons += [p for p in problems if p not in reasons]
        x = sd.workload(self.plan["points"][row["workload"]]["client_base"] if v else row["workload"])
        reqs = rd.derived_requests(res, x=x) if res is not None else None
        if reqs is not None:
            if v:                                         # their checks: decide (stage_v_analysis.run_valid)
                reqs = sva.gateway_requests(reqs, joined, plan=self.plan, point=row["workload"], block=row["block"])
            path = self.derived / f"{row['tag']}-requests.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in reqs))
            row["derived"] = path.relative_to(self.out).as_posix()
            self.session["derived_sha256"][row["tag"]] = rc._sha256_file(path)
            self.requests[row["tag"]] = reqs
            if not v:
                reasons += [p for p in ca.derived_problems(reqs, mix=row["mix"], C=C, N=row["N"]) if p not in reasons]
        row["reasons"], row["valid"] = reasons, not reasons
        return self.decide(row, reqs)

    def decide(self, row: dict, reqs) -> dict:
        """A V run's outcome with the replay's own function (stage_v_analysis.run_outcome); the analysis' checks beyond
        the bench and the access log (stage_v_analysis.run_valid: the derived requests, (j), E) are folded into the
        runner's verdict first."""
        if row["phase"] == "V":
            if row["valid"]:
                ok, why = sva.run_valid(row, reqs, self.plan)
                if not ok:
                    row["reasons"], row["valid"] = list(why), False
                    x = self.plan["points"][row["workload"]]
                    warm = row["warmups"] if not sva.warmup_problems(row, x) else ()
                    j = sva.inflight_problems(sva.in_vllm(reqs, C=x["C"], N=x["N"], warmups=warm), x=x) if reqs else []
                    if j:
                        self._fault(self.session["boots"][row["boot"]]["tag"], j, run=row["tag"])
            row["outcome"] = sva.run_outcome(row, reqs, self.plan)
        return self.emit(row)


# ---------------------------------------------------------------- the state machine (044 Sessions)

def _sentinel(h: StageVHarness, position: str) -> list:
    s = h.plan["sentinel"]
    return h.boot(h.new_boot("sentinel", s["workload"], position=position, elapsed_s=h.elapsed()), rc._fixed([s["C"]]),
                  [s["C"]])


def _v_machine(h: StageVHarness) -> None:
    """044: sentinel; blocks r = 1..n in the plan's block orders, one boot and one V run per active point, the hour cap
    before each V boot (the mini: a forced stop before its last planned one), a point whose run is not ok skips its
    later boots; sentinel."""
    plan = h.plan
    start = _sentinel(h, "start")
    if not start[0]["valid"]:
        raise RunError(f"the start sentinel is invalid ({'; '.join(start[0]['reasons'])}): the sentinel gate needs "
                       "both sentinels valid, so the session stops here (030 8.6; 044)")
    order, dropped = sva.v_order(plan), set()
    for k, (r, p) in enumerate(order):
        if p in dropped:
            continue
        x = plan["points"][p]
        elapsed = h.elapsed()
        allowed = sv.cap_allows(elapsed, x["est_s"], plan["hour_cap_h"])
        last = not any(q not in dropped for _, q in order[k + 1:])
        if not allowed or (plan["mini"] and last):
            h.session["cap_stop"] = {"block": r, "point": p, "elapsed_s": elapsed, "forced": allowed}
            break
        rows = h.boot(h.new_boot("V", p, block=r, elapsed_s=elapsed), rc._fixed([x["C"]]), [x["C"]])
        if rows[0]["outcome"] != "ok":
            dropped.add(p)                            # 044 實作補記六: the point alone (039 / 030), never its unit
    _sentinel(h, "end")


# ---------------------------------------------------------------- mini (044 mini smoke)

def build_mini_plan(plan: dict) -> dict:
    """The mini's session plan: stage_v_plan's mini plan (n 2, N 20, the sentinel at C 2, each point's mini C,
    v1-mini-8192, GW-U8K-O512's block 1 forced to time out, so the skip rule drops its block 2) with run_capacity's
    dummy SLO no valid run can miss; N = 20 is run_stage_d.mini_num_prompts' patch for the session."""
    p = copy.deepcopy(plan)
    p.update(slo=MINI_SLO, stage=f"{sv.STAGE}-mini")
    return p


def mini_checks(plan: dict, session: dict, rows: list, requests: dict) -> dict:
    """044 mini smoke's verdict: the replay is clean (order, skips, the forced cap stop); every run valid but the forced
    timeout, which hit; the skip rule walked: each later block's run of a forced point was skipped, and there is one
    (skip_walked, 實作補記六); the derived schema; every boot passed its checks, its gateway answered with the point's
    /readyz tuple and stopped cleanly; pacing made a first-wave request wait and weighted-v1 queued one; U_W and N1
    computed; the session records harness_sha256 and gateway_path_sha256; clean teardowns."""
    replay, sched = sva.schedule(plan, session["boots"], rows, requests, cap_stop=session.get("cap_stop"))
    forced = sorted([p, r] for p, x in plan["points"].items() for r in x["forced_timeout_blocks"])
    walk = [[later, p] for p, r in forced for later in range(r + 1, plan["n"] + 1)]
    bad = [r["tag"] for r in rows if not r.get("valid") and [r["workload"], r.get("block")] not in forced]
    hit = sorted([r["workload"], r.get("block")] for r in rows if r.get("timeout"))
    schema = all(set(q) == set(sva.DERIVED_KEYS if r["phase"] == "V" else rd.DERIVED_KEYS)
                 for r in rows for q in requests.get(r["tag"]) or [])
    pts = plan["points"]
    valid = [(pts[r["workload"]], requests[r["tag"]]) for r in rows if r["phase"] == "V" and r.get("valid")]
    conditions = [sva.nontrivial(sva.in_vllm(reqs, C=x["C"], N=x["N"]), x=x, plan=plan) for x, reqs in valid]
    gws = [session["runs"].get(b["tag"], {}).get("gateway") for b in session["boots"] if b["phase"] == "V"]
    checks = {
        "replay_clean": not replay, "other_runs_valid": not bad, "forced_timeouts_hit": hit == forced,
        "derived_schema": schema, "boots_passed_checks": all(not b["problems"] for b in session["boots"]),
        "gateway_ok": bool(gws) and all(g and g.get("readyz") and not g.get("stop_problems") for g in gws),
        "forced_cap_stop": (session.get("cap_stop") or {}).get("forced") is True,
        "prefill_waited": any(x["pacing"] and any((q["gw_prefill_wait_ms"] or 0) > 0 for q in reqs)
                              for x, reqs in valid),
        "queued": any(x["admission"] and any(q["gw_queue_wait_ms"] > 0 for q in reqs) for x, reqs in valid),
        "nontrivial_computable": any(c is not None for c in conditions)
        and all(c is None or c["value"] is not None for c in conditions),
        "recorded_hashes": all(sv._sha(session.get(k)) for k in ("harness_sha256", "gateway_path_sha256")),
        "clean_teardowns": not session.get("teardown_problems"),
        "skip_walked": bool(walk) and all(w in sched["skipped"] for w in walk)}
    return {"ok": all(checks.values()), "checks": checks, "replay_problems": replay, "invalid_runs": bad,
            "timeouts": hit, "skipped": sched["skipped"]}


# ---------------------------------------------------------------- session

def _refusals(plan, *, cfg, stage_d_cfg, pins, pins_sha256, stage_v_sha256, stage_d_sha256, out_dir, mode,
              lock_held_check, gpu_lock_held, base_env, repo_root) -> None:
    problems = output_problems(out_dir, mode=mode, repo_root=repo_root)
    if problems:
        raise RunError("; ".join(problems))
    if not gpu_lock_held:
        raise RunError(f"Stage V boots must hold the GPU lock: flock -n {smoke.GPU_LOCK} ... --gpu-lock-held "
                       "(MACHINE_POLICY)")
    if not lock_held_check():
        raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (MACHINE_POLICY)")
    formal = mode == "formal"
    if formal and ((cfg or {}).get("freeze") or {}).get("stage_v") is not True:
        raise RunError("formal Stage V sessions require freeze.stage_v = true in configs/stage_v.toml (044 凍結)")
    if (plan.get("session") == "mini") is not (mode == "mini"):
        raise RunError(f"--{mode} runs {'the mini plan only' if mode == 'mini' else 'V-ADM or V-CLS, never the mini'}, "
                       f"not the {plan.get('session')!r} plan (044)")
    try:
        want = sv.build_plan(cfg, pins, stage_d_cfg, session_name=plan.get("session"), pins_sha256=pins_sha256,
                             stage_v_sha256=stage_v_sha256, stage_d_sha256=stage_d_sha256, formal=formal)
        problems = [f"plan {k} differs from stage_v_plan.build_plan{rc.dict_diff(plan.get(k), want.get(k))}"
                    for k in sorted(set(plan) | set(want), key=str) if plan.get(k) != want.get(k)]
    except (bp.PlanError, OSError, KeyError, ValueError, TypeError) as exc:
        problems = [f"the plan cannot be rebuilt: {exc}"]
    if not problems:
        problems += [f"request path: {p}" for p in sv.gateway_path_problems(cfg, ROOT)]
        problems += [f"profile: {p}" for p in sv.profile_problems(cfg, ROOT)]
    if formal:
        sb = pins.get("stage_b") or {}
        problems += rc._rb().stage_b_problems(plan, pins, gdn_prefill_backend=sb.get("gdn_prefill_backend"),
                                              gpu_memory_utilization=sb.get("gpu_memory_utilization"))
        inherited = bp.inherited_unlisted(base_env)
        if inherited:
            problems.append(f"inherited engine variables {inherited}: a formal session inherits none (030 8.6)")
        problems += rc.server_env_problems(pins, base_env)
        problems += [f"freeze evidence: {p}" for p in sv.freeze_evidence_problems(cfg, repo_root)]
    if problems:
        raise RunError("; ".join(problems))


def _dataset_problems(plan: dict, datasets: dict) -> tuple:
    """({key: sha256}, problems): every dataset the session reads, present and its pinned sha256 (the mini's too: 044
    pins the datasets before it)."""
    sha, problems = {}, []
    for key in sorted(plan["datasets"]):
        path = datasets.get(key)
        if not path or not Path(path).is_file():
            problems.append(f"{key} needs --dataset {key}=<file> (044)")
            continue
        sha[key] = rc._sha256_file(path)
        if sha[key] != plan["datasets"][key]:
            problems.append(f"dataset {key}: sha256 {sha[key]} != pinned {plan['datasets'][key]!r} (044: pinned in "
                            "stage_v.toml before the mini)")
    return sha, problems


def run_session(plan: dict, *, cfg: dict, stage_d_cfg: dict, pins: dict, pins_sha256: str, stage_v_sha256: str,
                stage_d_sha256: str, out_dir, session_id: str, mode: str = "formal", popen: Callable,
                signal_group: Callable, group_alive: Callable, http_get: Callable, run_bench: Callable,
                gpu_sample: Callable, compute_apps: Callable, shared_usage: Callable,
                lock_held_check: Callable | None = None, gpu_lock_held: bool = False, vllm_bin: str = ".venv/bin/vllm",
                datasets: dict | None = None, gateway_python: str = sys.executable,
                port_free: Callable = rd.default_port_free, sample_interval: float | None = 1.0,
                toolchain_check: Callable = bp.toolchain_problems, capture_env: Callable | None = None, base_env=None,
                now: Callable = time.monotonic, sleep: Callable = time.sleep,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(), repo_root=ROOT) -> dict:
    """One Stage V session in `mode` ("formal" or "mini"); the session record (also session.json)."""
    if mode not in MODES:
        raise RunError(f"unknown mode {mode!r}; run_stage_v runs {MODES}")
    base_env = dict(os.environ if base_env is None else base_env)
    lock_held_check = lock_held_check or ra.held_gpu_lock
    _refusals(plan, cfg=cfg, stage_d_cfg=stage_d_cfg, pins=pins, pins_sha256=pins_sha256,
              stage_v_sha256=stage_v_sha256, stage_d_sha256=stage_d_sha256, out_dir=out_dir, mode=mode,
              lock_held_check=lock_held_check, gpu_lock_held=gpu_lock_held, base_env=base_env, repo_root=repo_root)
    formal = mode == "formal"
    session_plan = build_mini_plan(plan) if mode == "mini" else plan
    datasets = dict(datasets or {})
    data_sha, problems = _dataset_problems(session_plan, datasets)
    if problems:
        raise RunError("; ".join(problems))
    server_env = sd.server_argv_for(pins, sv.SENTINEL["workload"], port=PORT, vllm_bin=vllm_bin, base_env=base_env)[1]
    missing = toolchain_check(server_env)
    if missing:
        raise RunError("toolchain incomplete, not starting (014): " + "; ".join(missing))
    used, problems = hours_used(repo_root)
    if not problems and not sv.total_cap_allows(used, plan["hour_cap_h"], plan["total_cap_h"]):
        problems = [f"the Stage V sessions so far used {float(used):.3f} h; with this session's cap of "
                    f"{plan['hour_cap_h']} h that is over the {plan['total_cap_h']} h total: not starting, ask the "
                    "user first (044 全期總上限)"]
    if problems:
        raise RunError("; ".join(problems))
    started, t0 = clock(), now()
    previous = None
    if formal:
        previous, problems = gap_problems(repo_root, started)
        if problems:
            raise RunError("; ".join(problems))
    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten (rerun under a new id)")
    # 044 實作補記六: from the mkdir on, session.json exists: provisional (in_progress, finished_at null) until the
    # finally below finishes it on every exit Python sees, so a hard kill leaves a record hours_used / gap_problems
    # refuse instead of a directory they cannot see.
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": session_plan["stage"],
               "session": plan["session"], "kind": plan["kind"], "mode": mode, "formal": formal,
               "started_at": started, "finished_at": None, "in_progress": True, "boots": [], "runs": {},
               "out": str(out)}
    for d in ("raw", "env", "derived"):
        (out / d).mkdir(parents=True)
    rc._write_json(out / "session.json", session)
    h = fh = None
    try:
        try:
            rc._write_json(out / "plan.json", session_plan)
            pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True) if capture_env else {}
            repo = pre.get("repo") or {}
            vllm = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
            if formal:
                if vllm != pins["vllm"]["version"]:
                    raise RunError(f"vLLM {vllm!r} in the preflight != pinned {pins['vllm']['version']} (003; 030 "
                                   "8.6)")
                bad = env_problems(pre, rc.PLATFORM) + provenance_problems(pre, pre)
                if repo.get("source") != "DEPLOYED_COMMIT":
                    bad.append(f"repo source {repo.get('source')!r}: the code is not a deployed bundle (one "
                               "DEPLOYED_COMMIT)")
                if bad:
                    raise RunError("preflight: " + "; ".join(bad))
            sb = pins["stage_b"]
            session.update({
                "gpu_lock_held": gpu_lock_held, "plan_sha256": sha256_json(session_plan), "pins_sha256": pins_sha256,
                "stage_v_sha256": stage_v_sha256, "stage_d_sha256": stage_d_sha256,
                "plan_pins_sha256": plan.get("pins_sha256"), "plan_stage_v_sha256": plan.get("stage_v_sha256"),
                "plan_stage_d_sha256": plan.get("stage_d_sha256"), "hours_used_h": float(used),
                "previous_formal": previous, "preflight": "env/preflight.json" if capture_env else None,
                "repo_commit": repo.get("commit"), "repo_source": repo.get("source"), "vllm": vllm,
                "server": {"gdn_prefill_backend": sb["gdn_prefill_backend"],
                           "gpu_memory_utilization": sb["gpu_memory_utilization"], "port": PORT, "vllm_bin": vllm_bin},
                "datasets": {k: str(datasets[k]) for k in sorted(data_sha)}, "dataset_sha256": data_sha,
                "server_env": ra.recorded_server_env(server_env),
                "inherited_unlisted": bp.inherited_unlisted(base_env), "gpu_idle": gi.criterion(sb["v_total_mib"]),
                "derived_sha256": {}, "cap_stop": None, "gateway_faults": [], "result_dir": str(out / "raw"),
                "root": str(ROOT), "gateway_python": gateway_python, "harness_sha256": sv.harness_sha256(ROOT),
                "gateway_path_sha256": sv.gateway_path_sha256(ROOT), "capserve_sha256": rd.capserve_sha256(ROOT),
                "profile_sha256": {p: rc._sha256_file(ROOT / x["profile"]) for p, x in plan["points"].items()}})
            fh = open(out / "rows.jsonl", "w")
            h = StageVHarness(plan=session_plan, pins=pins, out=out, session=session, formal=formal, pre=pre,
                              rows_fh=fh, popen=popen, signal_group=signal_group, group_alive=group_alive,
                              http_get=http_get, run_bench=run_bench, gpu_sample=gpu_sample,
                              compute_apps=compute_apps, shared_usage=shared_usage, lock_held_check=lock_held_check,
                              capture_env=capture_env, base_env=base_env, vllm_bin=vllm_bin,
                              sample_interval=sample_interval, now=now, sleep=sleep, clock=clock, datasets=datasets,
                              gateway_python=gateway_python, port_free=port_free, t0=t0)
            with rd.mini_num_prompts(plan["mini"]["N"]) if mode == "mini" else contextlib.nullcontext():
                _v_machine(h)
            if mode == "mini":
                session["smoke"] = mini_checks(session_plan, session, h.rows, h.requests)
        finally:
            if fh is not None:
                fh.close()
    except BaseException as exc:
        session["aborted"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        flags = h.env_flags if h is not None else []
        session.update(finished_at=clock(), in_progress=False, outcomes=rc._outcomes(h.rows if h else [], session),
                       env_flags=flags, env_valid=h is not None and not flags)
        if not formal:
            if "smoke" not in session:
                session["smoke"] = {"ok": False, "aborted": session.get("aborted")}
            rc._write_json(out / "smoke.json", session["smoke"])
            rc.mask_smoke(out)
            for rec in session["runs"].values():
                rec.pop("startup_s", None)
        rows_path = out / "rows.jsonl"
        session["rows_sha256"] = rc._sha256_file(rows_path) if rows_path.is_file() else None
        write_raw_manifest(out)
        if not formal:                    # 044 實作補記六: the smoke's per-request gateway times go too; the raw
            logs = sorted((out / "raw").glob("*.gateway.log"))            # manifest above keeps their sha256
            for path in logs:
                path.unlink()
            session["gateway_logs_discarded"] = [path.relative_to(out).as_posix() for path in logs]
        rc._write_json(out / "session.json", session)
    return session


def main(argv: list | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="stage_v_plan.py output")
    parser.add_argument("--out", required=True, help="results/stage_v/formal-<id> or results/stage_v/smoke-v-<id>")
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--dataset", action="append", default=[], help="<key>=path, a key of [stage_v.datasets] (044)")
    parser.add_argument("--gpu-lock-held", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", dest="mode", action="store_const", const="formal")
    modes.add_argument("--mini", dest="mode", action="store_const", const="mini")
    args = parser.parse_args(argv)
    raw = {"stage_v": sv.load_stage_v_bytes(), "stage_d": sd.load_stage_d_bytes(), "pins": cp.load_pins_bytes()}
    cfg, sd_cfg, pins = (tomllib.loads(raw[k].decode()) for k in ("stage_v", "stage_d", "pins"))
    rb = rc._rb()
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    try:
        summary = run_session(
            json.loads(Path(args.plan).read_text()), cfg=cfg, stage_d_cfg=sd_cfg, pins=pins,
            pins_sha256=hashlib.sha256(raw["pins"]).hexdigest(),
            stage_v_sha256=hashlib.sha256(raw["stage_v"]).hexdigest(),
            stage_d_sha256=hashlib.sha256(raw["stage_d"]).hexdigest(), out_dir=args.out,
            session_id=Path(args.out).name, mode=args.mode, popen=ra.start_server, signal_group=smoke.signal_group,
            group_alive=smoke.group_alive, http_get=rb.default_http_get, run_bench=ra.default_run_bench,
            gpu_sample=gi.nvidia_sample, compute_apps=rc.compute_apps_bounded, shared_usage=gi.read_shared_usage,
            lock_held_check=ra.held_gpu_lock, gpu_lock_held=args.gpu_lock_held, vllm_bin=args.vllm_bin,
            capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py"),
            datasets=dict(x.split("=", 1) for x in args.dataset), sample_interval=1.0)
    except (RunError, rc.RunError, ra.RunError) as exc:
        print(f"run_stage_v: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": summary.get("mode"), "outcomes": summary.get("outcomes"),
                      "cap_stop": summary.get("cap_stop"), "smoke": summary.get("smoke")}))
    if args.mode == "formal":
        return 0
    return 0 if (summary.get("smoke") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
