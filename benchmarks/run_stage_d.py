#!/usr/bin/env python3
"""Stage D session runner (decision 039; PREREGISTRATION §9). One process, under a GPU lock it already holds, runs one
session of stage_d_plan (D-A1, D-A2, D-B or D-C). Core and Stage C modules are imported, never changed:

- the per-boot and per-run steps are run_capacity's (_Harness: the lock check, 031's idle window and two-stage spill
  guard, env capture, run_adv.start_server and shutdown_group, the bench with its timeout and 1 Hz sampler); this
  module's StageDHarness overrides only what 039 changes: the server (B0, or T with --long-prefill-token-threshold),
  its non-default args check, the client (stage_d_plan.bench_argv_for: 512-token outputs, var datasets, the open-loop
  rate, the gateway's port), the result check (each workload's output length and mix), the derived requests and their
  check (stage_d_analysis), the tags, and the D-C gateway;
- D-A runs run_capacity's own state machine (sentinel, Phase S with capacity_plan.search_next, confirm plan, Phase C
  with capacity_plan.phase_c_boots, sentinel) with stage_d_analysis.search_bits / phase_c_outcome; D-B runs the same
  machine (_search_machine, a copy with the search and confirm functions passed in) with 039's scan and confirm set
  (stage_d_plan.open_search_next / build_open_confirm_plan); its "C" is the grid index (rate, N and the run's own
  client seed come from the plan);
- D-C: sentinel; blocks r = 1..n in capacity_plan.block_order, one boot and one V run per active point; 030's skip
  rule: a point leaves later blocks after a valid FAIL or an invalid run; sentinel.
- D-C gateway (U2K-gw, U8K-gw): after the server's checks, `python -m capserve serve` from this checkout (a subprocess:
  research code never imports capserve) with the production settings that matter (127.0.0.1:8080 -> vLLM 8000,
  tokenizer local from the served snapshot, the class of the workload) and the verified profile of the plan. Port
  8080 must be free before the boot (a leftover gateway would answer for it): a busy port aborts the session
  (gateway_port_busy), as 031's gpu_not_idle does, before vLLM starts. Its /readyz must name that profile (id and
  status) and class with max_concurrency = the class's verified limit, else the boot fails before its first bench.
  A gateway that exits while serving makes the run invalid (and the boot's later runs not run). The process is the
  boot's to stop from the moment it exists (self.gw, set right after popen, so an interrupt during its readiness
  wait stops it too): SIGTERM, SIGKILL after 30 s, then the group must be gone, under run_adv.DeferredSignals, before
  the server's teardown. session.json records harness_sha256 (stage_d_plan.HARNESS_FILES; the freeze evidence). session.json records the sha256 of service/capserve and the profile (capserve_sha256): H-D7 measures
  that gateway.

Refusals before any server starts, as run_capacity: the output is not results/stage_d/<session> (or exists), no held
GPU lock, stage_d.toml breaks 039 or the plan is not stage_d_plan.build_plan of it (lambda_ref recomputed from the
committed c1b files), a missing dataset; formal: freeze.stage_d true with its evidence, Stage B measured, no
inherited engine variable, the server env the report rebuilds, the pinned datasets, a deployed clean tree and the
pinned vLLM.

Modes: --formal, or --mini (the 039 smoke before the freeze: the same machinery on a mini plan with a dummy SLO no
valid run can miss and N = 20; its verdict goes to smoke.json and every latency is discarded, as run_capacity's).

Usage (WSL, ~/llm-serving-lab, the plan from benchmarks/stage_d_plan.py):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_stage_d.py --plan stage-d-plan.json \\
        --out results/stage_d/<session> --dataset mixed:8192=data/core_b/mixed-8192.jsonl \\
        --dataset var:2048=data/stage_d/var-2048.jsonl --gpu-lock-held (--formal | --mini)
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
import signal
import sys
import time
from pathlib import Path
from typing import Callable

import capacity_plan as cp
import core_b_plan as bp
import coreb_smoke as smoke
import gpu_idle as gi
import run_adv as ra
import run_capacity as rc
import stage_d_analysis as sda
import stage_d_plan as sd
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
MODES = ("formal", "mini")
PORT = sd.PORT
ROOT = sd.ROOT
GATEWAY_READY_S = 120
GATEWAY_STOP_S = 30
GATEWAY_DRAIN_S = 5
GATEWAY_KILL_WAIT_S = 10
MINI_N = 20
MINI_SLO = rc.MINI_SLO
MINI_FORCED_TIMEOUT_S = rc.MINI_FORCED_TIMEOUT_S
# the mini plans (039 smoke): every kind of workload, server and client once, with a dummy SLO
MINI_WORKLOADS = {"closed": ["U2K-O512", "M8K-T"], "open": ["U2K-P"], "verify": ["U2K-var", "U2K-gw", "U8K-gw"]}
MINI_C_CAP = {"U2K-O512": 3, "M8K-T": 2}
MINI_OPEN_POINTS = (1, 7, 9)                  # the open mini runs 0.4, 1.1 and 1.5 lambda_ref (over lambda_ref too)
DERIVED_KEYS = (*rc.DERIVED_KEYS, "start_s")
MINI_FORCED = {"U2K-O512": (3,)}
MINI_VERIFY = {"C": 4, "n": 2}


class RunError(rc.RunError):
    pass


def output_problems(out_dir, *, repo_root=ROOT) -> list:
    root = Path(repo_root).resolve().joinpath(*sd.RESULTS)
    out = Path(out_dir).resolve()
    if out == root or root not in out.parents:
        return [f"{out_dir} is not a session directory under results/stage_d/ (039: Stage D writes only there)"]
    if rc.PRODUCT_RUNS in out.relative_to(root).parts:
        return [f"{out_dir} is inside a runs/ directory: product-test outputs are never research data (027 §3)"]
    return []


# ---------------------------------------------------------------- server checks (030 8.6 (b) (c) (d); 039 T)

def nondefault_problems(log_text: str, *, kv_bytes: int, threshold: int | None) -> tuple:
    """(watched non-default args, problems): max_num_batched_tokens 2048, kv_cache_memory_bytes K, and
    long_prefill_token_threshold exactly when the workload's server is T; no observation or G key."""
    got = ra.parse_nondefault(log_text)
    if got is None:
        return None, ["no non-default args line in the server log (030 8.6 (c))"]
    want = {"max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS, "kv_cache_memory_bytes": kv_bytes}
    if threshold is not None:
        want["long_prefill_token_threshold"] = threshold
    problems = []
    for k in ra.WATCHED_KEYS:
        if k in want and (k not in got or type(got[k]) is not type(want[k]) or got[k] != want[k]):
            problems.append(f"non-default args {k} = {got.get(k, '<absent>')!r}; this server needs {want[k]!r} "
                            "(030 8.6 (c); 039)")
        elif k not in want and k in got:
            problems.append(f"non-default args carry {k} = {got[k]!r}, a key this server must not set (039)")
    return got, problems


def server_checks(log_text: str, metrics_text: str, *, ready: bool, plan: dict, w: str, detail: str = "") -> tuple:
    facts = smoke.parse_server_log(log_text)
    problems = [] if ready else [f"server not ready{detail} (030 8.6 (b))"]
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN (030 8.6 (b); 003)")
    t = plan["long_prefill_token_threshold"] if sd.workload(w)["server"] == "T" else None
    nondefault, nd = nondefault_problems(log_text, kv_bytes=plan["kv_cache_memory_bytes"], threshold=t)
    cfg, blocks = ra.blocks_problems(metrics_text, expected_blocks=plan["expected_blocks"],
                                     block_size=plan["block_size"])
    return {"log_facts": facts, "nondefault": nondefault, "cache_config": cfg}, problems + nd + blocks


# ---------------------------------------------------------------- per run

def summarize(res: dict, *, x: dict, n: int) -> dict:
    """run_core_b.summarize_result's checks for a Stage D workload: N completed, none failed, every output the
    workload's length, the input lengths of its mix (uniform: the prompt; mixed: 007's counts; var: inside
    [64, tier - 128])."""
    reasons = []
    if res.get("completed") != n or res.get("failed"):
        reasons.append(f"completed={res.get('completed')} failed={res.get('failed')} (expected {n} / 0)")
    lens, outs = res.get("input_lens") or [], res.get("output_lens") or []
    if x["mix"] == "uniform":
        ok = lens == [x["prompt"]] * n
    elif x["mix"] == "mixed":
        short, long_ = bp.mixed_counts(n)
        ok = sorted(lens) == sorted([x["tier"] - bp.N_GEN] * short + [bp.LONG_INPUT] * long_)
    else:
        ok = len(lens) == n and all(sd.VAR_MIN_PROMPT <= v <= x["tier"] - bp.N_GEN for v in lens)
    if not ok:
        reasons.append(f"input_lens do not match the {x['mix']} mix of this workload ({len(lens)} values)")
    if outs != [x["out"]] * n:
        reasons.append(f"output_lens not all {x['out']}")
    return {"ok": not reasons, "reasons": reasons, "num_prompts": n,
            **{k: res.get(k) for k in ("completed", "failed", "duration", "request_throughput", "output_throughput",
                                       "total_token_throughput")}}


def derived_requests(res: dict, *, x: dict) -> list | None:
    """derived/<tag>-requests.jsonl rows (030 8.7 schema): cls "long" for a mixed workload's 32,640-token prompts,
    "short" otherwise (a var prompt is short at any length); never prompt or generated text."""
    cols = [res.get(k) for k in ("input_lens", "output_lens", "ttfts", "latencies")]
    if not all(isinstance(c, list) for c in cols) or len({len(c) for c in cols}) != 1:
        return None
    starts = res.get("start_times")
    ok_starts = isinstance(starts, list) and len(starts) == len(cols[0]) and starts and \
        all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in starts)
    t0 = min(starts) if ok_starts else None
    out = []
    for i, (n, o, t, la) in enumerate(zip(*cols)):
        cls = "long" if x["mix"] == "mixed" and n == bp.LONG_INPUT else "short"
        out.append({"idx": i, "cls": cls, "ttft_ms": None if t is None else t * 1000,
                    "latency_ms": None if la is None else la * 1000, "out": o,
                    "start_s": None if t0 is None else round(starts[i] - t0, 6)})   # send time (039 D-B order)
    return out


def default_port_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def capserve_sha256(root=ROOT, profile: str | None = None) -> str:
    """sha256 over service/capserve/*.py (and the gateway profile): the gateway H-D7 measures (039)."""
    root = Path(root)
    h = hashlib.sha256()
    files = sorted((root / "service" / "capserve").glob("*.py")) + ([root / profile] if profile else [])
    for path in files:
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def gateway_settings(plan: dict, pins: dict, w: str) -> dict:
    """The D-C gateway (039): the production settings that matter, on loopback in front of the research vLLM."""
    return {"active_class": sd.workload(w)["gateway_class"], "backend_url": f"http://127.0.0.1:{PORT}",
            "host": "127.0.0.1", "port": plan["verify"]["gateway_port"], "drain_timeout_s": GATEWAY_DRAIN_S,
            "tokenizer": "local", "tokenizer_path": bp.snapshot_dir(pins)}


def _toml(doc: dict) -> str:
    return "".join(f"{k} = {json.dumps(v)}\n" for k, v in doc.items())


class StageDHarness(rc._Harness):
    """run_capacity's per-boot / per-run steps with 039's servers, clients, checks and the D-C gateway."""

    def __init__(self, *, datasets: dict, gateway_python: str, port_free: Callable = default_port_free, **kw):
        super().__init__(mixed_datasets=None, **kw)
        self.kind = self.plan["kind"]
        self.datasets = datasets              # {"mixed:8192": path, "var:2048": path, ...}
        self.gateway_python = gateway_python
        self.port_free = port_free
        self.gw = None                        # the current boot's gateway (D-C)

    # ---------------------------------------------------------- boot
    def boot(self, boot: dict, next_point: Callable, planned: list | None = None) -> list | None:
        tag, w = boot["tag"], boot["workload"]
        if sd.workload(w)["via"] == "gateway":
            port = self.plan["verify"]["gateway_port"]
            if not self.port_free(port):
                self.session["gateway_port_busy"] = {"boot": tag, "port": port}
                raise RunError(f"gateway_port_busy before {tag}: port {port} is in use (a leftover gateway would "
                               "answer for this boot's); stopping before vLLM starts, rerun under a new id (039 D-C)")
        idle = self.pre_boot(tag)
        self.snap(f"{tag}.before.json")
        argv, env = sd.server_argv_for(self.pins, w, port=PORT, vllm_bin=self.vllm_bin, base_env=self.base_env)
        record = self.session["runs"][tag] = {"argv": argv, "env": ra.recorded_server_env(env), "idle": idle}
        proc = None
        rows = []
        try:
            proc = self.popen(argv, env, self.raw / f"{tag}.server.log")
            ready, detail = self.wait_ready(proc, record)
            metrics = self.metrics_text(ready)
            facts, problems = server_checks(rc._read(self.raw / f"{tag}.server.log"), metrics, ready=ready,
                                            plan=self.plan, w=w, detail=detail)
            record.update(facts)
            self.check_log(proc, "during the server start")
            if not problems and proc.poll() is not None:
                problems = [f"server exited (code {proc.poll()!r}) before its first bench (030 8.4)"]
            if not problems:
                entry = self.spill_check("ready", tag)
                if entry["spilled"]:
                    problems = [rc.spill_reason(entry, self.guard) + "; no bench runs on this boot"]
            if not problems and sd.workload(w)["via"] == "gateway":
                problems = self.gateway_up(boot, record)        # sets self.gw as soon as the process exists
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
        if boot["phase"] == "S" and not boot["reached_first_bench"]:
            return None
        return rows

    def gateway_up(self, boot: dict, record: dict) -> list:
        """Problems of `python -m capserve serve` with the plan's verified profile ([] when ready and serving the
        workload's class at its verified limit); self.gw holds the process from popen on."""
        w, tag = boot["workload"], boot["tag"]
        settings = gateway_settings(self.plan, self.pins, w)
        record["gateway"] = {"settings": settings, "profile": self.plan["verify"]["gateway_profile"]}
        if not self.port_free(settings["port"]):          # checked before the boot too; a race is still fatal
            self.session["gateway_port_busy"] = {"boot": tag, "port": settings["port"]}
            raise RunError(f"gateway_port_busy in {tag}: port {settings['port']} is in use (039 D-C)")
        cfg_path = self.raw / f"{tag}.gateway.toml"
        cfg_path.write_text(_toml(settings))
        profile = ROOT / self.plan["verify"]["gateway_profile"]
        argv = [self.gateway_python, "-m", "capserve", "serve", "--config", str(cfg_path), "--profile", str(profile)]
        env = {**{k: v for k, v in self.base_env.items() if not k.startswith("CAPSERVE_")},
               "PYTHONPATH": os.pathsep.join([str(ROOT / "service"), str(ROOT / "benchmarks")]),
               "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
        record["gateway"]["argv"] = argv
        gw = self.gw = self.popen(argv, env, self.raw / f"{tag}.gateway.log")
        base = f"http://127.0.0.1:{settings['port']}"
        t0, status, body = self.now(), 0, ""
        while self.now() - t0 < GATEWAY_READY_S and gw.poll() is None:
            status, body = self.http_get(f"{base}/readyz")
            if status == 200:
                break
            self.sleep(2)
        if status != 200:
            return [f"gateway not ready within {GATEWAY_READY_S} s (status {status}; exit {gw.poll()!r}) (039 D-C)"]
        try:
            ready = json.loads(body)
        except ValueError:
            ready = {}
        x = self.plan["workloads"][w]
        want_c = x.get("class_limit", x["C"])        # the class's verified limit: the point's C, except in the mini
        want_id = json.loads(profile.read_text())["profile_id"]
        got = ((ready.get("profile") or {}).get("id"), (ready.get("profile") or {}).get("status"),
               ready.get("active_class"), (ready.get("limits") or {}).get("max_concurrency"))
        record["gateway"]["readyz"] = {"profile": ready.get("profile"), "active_class": ready.get("active_class"),
                                       "limits": ready.get("limits")}
        want = (want_id, "verified", settings["active_class"], want_c)
        if got != want:
            return [f"gateway serves {got}, not {want} (039 D-C)"]
        if gw.poll() is not None:
            return [f"the gateway exited (code {gw.poll()!r}) right after it was ready (039 D-C)"]
        return []

    def gateway_down(self, gw, record: dict, tag: str) -> None:
        """SIGTERM the gateway's group, SIGKILL after GATEWAY_STOP_S, then the group must be gone; interrupts wait
        until it is (run_adv.DeferredSignals), so an ssh hang-up never leaves a gateway on 8080."""
        problems = []
        with ra.DeferredSignals():
            if gw.poll() is None or self.group_alive(gw):
                self.signal_group(gw, signal.SIGTERM)
                t0 = self.now()
                while (gw.poll() is None or self.group_alive(gw)) and self.now() - t0 < GATEWAY_STOP_S:
                    self.sleep(1)
                if gw.poll() is None or self.group_alive(gw):
                    self.signal_group(gw, signal.SIGKILL)
                    problems.append(f"gateway still running {GATEWAY_STOP_S} s after SIGTERM; killed")
                    t0 = self.now()
                    while self.group_alive(gw) and self.now() - t0 < GATEWAY_KILL_WAIT_S:
                        self.sleep(1)
                    if self.group_alive(gw):
                        problems.append("residual process in the gateway group after SIGKILL")
            reader = getattr(gw, "log_reader", None)
            if reader is not None and not reader.join(ra.LOG_JOIN_S):
                problems.append("gateway log reader still open after its stop")
        record["gateway"]["stop_problems"] = problems
        if problems:
            self.session.setdefault("teardown_problems", []).append({"boot": tag, "reasons": ["gateway: " + x
                                                                                              for x in problems]})

    # ---------------------------------------------------------- run
    def _x(self, w: str) -> dict:
        return sda.spec(self.plan, w)

    def _row(self, boot: dict, C: int) -> dict:
        w, phase = boot["workload"], boot["phase"]
        x = self._x(w)
        row = {"phase": phase, "workload": w, "C": C, "block": boot["block"], "position": boot["position"],
               "boot": boot["boot"], "tier": x["tier"], "mix": x["mix"], "out": x["out"], "valid": False,
               "reasons": [], "timeout": False, "output_throughput": None, "metrics": None}
        row["tag"] = sd.run_tag(phase, w, C, block=boot["block"], position=boot["position"], kind=self.kind)
        row["N"] = sda.run_n(self.plan, row)
        if phase == "sentinel":
            row["timeout_s"] = self.plan["sentinel"]["timeout_s"]
        elif self.kind == "closed":
            row["timeout_s"] = self.plan["workloads"][w]["timeouts_s"][str(C)]
        elif self.kind == "open":
            row["timeout_s"] = self.plan["workloads"][w]["points"][str(C)]["timeout_s"]
            row["rate"] = self.plan["workloads"][w]["points"][str(C)]["rate"]
        else:
            row["timeout_s"] = self.plan["workloads"][w]["timeout_s"]
        return row

    def decide(self, row: dict, reqs) -> dict:
        if row["phase"] == "S":
            bits = sda.search_bits(row, reqs, self.plan)
            row["search_pass"], row["within_band"] = bits["pass"], bits["within_band"]
        elif row["phase"] in ("C", "V"):                # V: D-C's skip rule reads it too (039)
            row["outcome"] = sda.phase_c_outcome(row, reqs, self.plan)
        return self.emit(row)

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
            summary = summarize(res, x=self._x(row["workload"]), n=n)
            reasons += summary.pop("reasons")
            row.update(summary=summary, output_throughput=summary.get("output_throughput"),
                       result=f"raw/{tag}.bench.json")
            reasons += rc.array_problems(res, N=n)
        return res, reasons

    def client_argv(self, row: dict) -> list:
        w = row["workload"]
        x = self._x(w)
        dataset = None
        if x["mix"] == "mixed":
            dataset = self.datasets.get(f"mixed:{x['tier']}")
        elif x["mix"] == "var":
            dataset = self.datasets.get(f"var:{x['tier']}")
        kw = {"result_dir": str(self.raw), "result_filename": f"{row['tag']}.bench.json", "dataset": dataset,
              "vllm_bin": self.vllm_bin}
        if self.kind == "open" and row["phase"] in ("S", "C"):
            return sd.bench_argv_for(self.pins, w, rate=row["rate"], N=row["N"],
                                     warmups=self.plan["open_loop"]["warmups"],
                                     seed=sd.open_seed(w, row["C"], phase=row["phase"], block=row["block"]), **kw)
        return sd.bench_argv_for(self.pins, w, C=row["C"], **kw)

    def run_point(self, boot: dict, proc, C: int) -> dict:
        row = self._row(boot, C)
        if self.spilled_in is not None:
            row["reasons"] = [f"not run: the boot stopped after a spill in {self.spilled_in} (031; 030 8.6 (g))"]
            return self.decide(row, None)
        if proc.poll() is not None:
            row["reasons"] = [f"server exited (code {proc.poll()!r}) before this bench; not run (030 8.6 (b))"]
            return self.decide(row, None)
        if self.gw is not None and self.gw.poll() is not None:
            row["reasons"] = [f"the gateway exited (code {self.gw.poll()!r}) before this bench; not run (039 D-C)"]
            return self.decide(row, None)
        res, reasons = self.bench(row, self.client_argv(row))
        self.check_log(proc, "during the bench")
        reasons += self.spill_row(row)
        if self.gw is not None and self.gw.poll() is not None:
            reasons.append(f"the gateway exited (code {self.gw.poll()!r}) during the bench (039 D-C)")
        reqs = derived_requests(res, x=self._x(row["workload"])) if res is not None else None
        if reqs is not None:
            path = self.derived / f"{row['tag']}-requests.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in reqs))
            row["derived"] = path.relative_to(self.out).as_posix()
            self.session["derived_sha256"][row["tag"]] = rc._sha256_file(path)
            self.requests[row["tag"]] = reqs
            n = row["N"]
            ordered = self.kind == "open" and row["phase"] in ("S", "C")
            reasons += [p for p in sda.derived_problems(reqs, x=self._x(row["workload"]), c=sda.window_c(self.plan, row, n),
                                                        n=n, ordered=ordered) if p not in reasons]
        row["reasons"], row["valid"] = reasons, not reasons
        return self.decide(row, reqs)


# ---------------------------------------------------------------- state machines

def _start_sentinel(h: StageDHarness) -> None:
    s = h.plan["sentinel"]
    start = h.boot(h.new_boot("sentinel", s["workload"], position="start"), rc._fixed([s["C"]]), [s["C"]])
    if not start[0]["valid"]:
        raise RunError(f"the start sentinel is invalid ({'; '.join(start[0]['reasons'])}): the sentinel gate needs "
                       "both sentinels valid, so the session stops here (030 8.6)")


def _end_sentinel(h: StageDHarness) -> None:
    s = h.plan["sentinel"]
    h.boot(h.new_boot("sentinel", s["workload"], position="end"), rc._fixed([s["C"]]), [s["C"]])


def _verify_machine(h: StageDHarness) -> None:
    """D-C (039): sentinel; stage_d_analysis.verify_order's boots, one V run each, 030's skip rule; sentinel."""
    plan = h.plan
    _start_sentinel(h)
    dropped = set()
    for r, w in sda.verify_order(plan):
        if w in dropped:
            continue
        C = plan["workloads"][w]["C"]
        rows = h.boot(h.new_boot("V", w, block=r), rc._fixed([C]), [C])
        if rows[0]["outcome"] != "ok":
            dropped.add(w)
    _end_sentinel(h)


def _search_machine(h: StageDHarness, write_session: Callable, *, search_next: Callable,
                    build_confirm: Callable) -> dict:
    """run_capacity._state_machine with the Phase S search and the confirm plan passed in (D-B: 039's scan and K)."""
    plan = h.plan
    _start_sentinel(h)
    searches, retry = {}, []

    def search(w: str, attempt: int):
        c_cap = plan["workloads"][w]["c_cap"]
        rows = h.boot(h.new_boot("S", w, attempt=attempt),
                      lambda done: search_next(c_cap, [(r["C"], r["search_pass"]) for r in done]))
        return None if rows is None else [{"C": r["C"], "pass": r["search_pass"], "within_band": r["within_band"]}
                                          for r in rows]

    for w in plan["workloads"]:
        hist = search(w, 1)
        if hist is None:
            retry.append(w)
        else:
            searches[w] = hist
    for w in cp.phase_s_retries(plan["workloads"], set(retry)):
        searches[w] = search(w, 2)
    confirm = build_confirm(plan, searches)
    rc._write_json(h.out / "confirm_plan.json", confirm)
    h.session.update(confirm_plan_sha256=cp.confirm_plan_sha256(confirm), confirm_plan_recorded_at=h.clock())
    write_session()
    outcomes = {w: [] for w in plan["workloads"]}
    for r in range(1, confirm["blocks"] + 1):
        new = []
        for w, points in cp.phase_c_boots(confirm, r, outcomes):
            for row in h.boot(h.new_boot("C", w, block=r), rc._fixed(points), points):
                new.append((w, (r, row["C"], row["outcome"])))
        for w, o in new:
            outcomes[w].append(o)
    _end_sentinel(h)
    return confirm


# ---------------------------------------------------------------- mini (039 smoke)

def build_mini_plan(plan: dict) -> dict:
    """The smoke's plan: the kind's MINI_WORKLOADS, a dummy SLO, N = MINI_N (closed loop and D-C through
    core_b_plan.num_prompts, patched for the session; open loop in the points), small caps, n_near 2 / n_low 1, the
    sentinel at C 2, and a forced 5 s timeout (U2K-O512 at C 3: a fail, then the skip rule). The open mini scans
    the grid points 0.4, 1.1 and 1.5 lambda_ref. D-C points (U2K-var and both gateways: uniform-8k's 8,064-token prompts
    sit exactly on its class bound) run at C 4 behind a gateway that still serves its class's verified limit
    (class_limit)."""
    p = copy.deepcopy(plan)
    p["slo"] = MINI_SLO
    p["stage"] = f"{sd.STAGE}-mini"
    p["n_low"] = 1
    p["sentinel"].update(C=rc.MINI_SENTINEL_C, N=MINI_N)
    keep = MINI_WORKLOADS[p["kind"]]
    workloads = {}
    for w in keep:
        x = copy.deepcopy(plan["workloads"].get(w) or {})
        if not x:
            x = {**sd.WORKLOADS[w], "kv_len": sd.kv_len(w), "c_kv": 0}
        if p["kind"] == "closed":
            cap = MINI_C_CAP[w]
            x.update(c_cap=cap, n_near=2, bracket=cp.bracket_points(cap))
            x["timeouts_s"] = {str(C): (MINI_FORCED_TIMEOUT_S if C in MINI_FORCED.get(w, ()) else
                                        x.get("timeouts_s", {}).get(str(C), 600)) for C in range(1, cap + 1)}
        elif p["kind"] == "open":
            cap = len(MINI_OPEN_POINTS)
            x.update(c_cap=cap, n_near=2, scan=list(range(1, cap + 1)),
                     points={str(k): {**x["points"][str(i)], "N": MINI_N}
                             for k, i in enumerate(MINI_OPEN_POINTS, start=1)})
        else:
            x.update(class_limit=x["C"], C=MINI_VERIFY["C"], n=MINI_VERIFY["n"])
        workloads[w] = x
    p["workloads"] = workloads
    return p


def mini_checks(plan: dict, session: dict, rows: list, requests: dict) -> dict:
    """The 039 smoke's verdict: the replay is clean, every run valid except the forced timeouts, the derived schema
    holds, the forced timeouts hit, the boots' checks passed (T included), the gateway answered (D-C), clean
    teardowns."""
    confirm = None
    if plan["kind"] != "verify":
        confirm = json.loads((Path(session["out"]) / "confirm_plan.json").read_text())
    replay = sda.replay_problems(plan, confirm, session["boots"], rows, requests,
                                 confirm_sha256=session.get("confirm_plan_sha256"))
    forced = {(w, C) for w, cs in MINI_FORCED.items() for C in cs if w in plan["workloads"]}
    bad = [r["tag"] for r in rows if not r.get("valid") and (r["workload"], r["C"]) not in forced]
    hit = sorted({(r["workload"], r["C"]) for r in rows if r.get("timeout")})
    schema = all(set(q) == set(DERIVED_KEYS) for reqs in requests.values() for q in reqs)
    boots_ok = all(not b["problems"] for b in session["boots"])
    gateway = None
    if plan["kind"] == "verify":
        gws = [rec.get("gateway") for rec in session["runs"].values() if rec.get("gateway")]
        gateway = bool(gws) and all((g.get("readyz") or {}).get("profile") and not g.get("stop_problems")
                                    for g in gws)
    checks = {"replay_clean": not replay, "other_runs_valid": not bad, "forced_timeouts_hit": hit == sorted(forced),
              "derived_schema": schema, "boots_passed_checks": boots_ok,
              "clean_teardowns": not session.get("teardown_problems")}
    if gateway is not None:
        checks["gateway_ok"] = gateway
    return {"ok": all(checks.values()), "kind": plan["kind"], "checks": checks, "replay_problems": replay,
            "invalid_runs": bad, "timeouts": [list(t) for t in hit]}


@contextlib.contextmanager
def mini_num_prompts(n: int = MINI_N):
    """run_capacity's own patch: core_b_plan.num_prompts returns n for the session, then is restored."""
    with rc.mini_num_prompts(n):
        yield


# ---------------------------------------------------------------- session

def expected_plan(cfg: dict, pins: dict, *, session_name: str, pins_sha256: str, stage_d_sha256: str, formal: bool,
                  repo_root=ROOT, uniform_plan: dict | None = None) -> dict:
    """stage_d_plan.build_plan of these bytes; C_KV's block pool from uniform_plan (the report passes the derived_from
    session's) or pins [stage_b] (the runner: build_plan refuses a pool other than the pins' anyway)."""
    sb = pins["stage_b"]
    pool = uniform_plan or {"num_gpu_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"]}
    return sd.build_plan(cfg, pins, session_name=session_name, uniform_plan=pool, pins_sha256=pins_sha256,
                         stage_d_sha256=stage_d_sha256, formal=formal)


def _refusals(plan, *, cfg, pins, pins_sha256, stage_d_sha256, out_dir, mode, lock_held_check, gpu_lock_held,
              base_env, repo_root) -> None:
    problems = output_problems(out_dir, repo_root=repo_root)
    if problems:
        raise RunError("; ".join(problems))
    if not gpu_lock_held:
        raise RunError(f"Stage D boots must hold the GPU lock: flock -n {smoke.GPU_LOCK} ... --gpu-lock-held "
                       "(MACHINE_POLICY)")
    if not lock_held_check():
        raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (MACHINE_POLICY)")
    formal = mode == "formal"
    if formal and ((cfg or {}).get("freeze") or {}).get("stage_d") is not True:
        raise RunError("formal Stage D sessions require freeze.stage_d = true in configs/stage_d.toml (039 凍結)")
    problems = [f"stage_d.toml: {p}" for p in sd.stage_d_problems(cfg, pins)]
    if not problems:
        try:
            want = expected_plan(cfg, pins, session_name=plan.get("session"), pins_sha256=pins_sha256,
                                 stage_d_sha256=stage_d_sha256, formal=formal, repo_root=repo_root)
            problems = [f"plan {k} differs from stage_d_plan.build_plan{rc.dict_diff(plan.get(k), want.get(k))}"
                        for k in sorted(set(plan) | set(want), key=str) if plan.get(k) != want.get(k)]
        except (sd.PlanError, OSError, KeyError, ValueError) as exc:
            problems = [f"the plan cannot be rebuilt: {exc}"]
    if formal:
        sb = pins.get("stage_b") or {}
        problems += rc._rb().stage_b_problems(plan, pins, gdn_prefill_backend=sb.get("gdn_prefill_backend"),
                                              gpu_memory_utilization=sb.get("gpu_memory_utilization"))
        inherited = bp.inherited_unlisted(base_env)
        if inherited:
            problems.append(f"inherited engine variables {inherited}: a formal session inherits none (030 8.6)")
        problems += rc.server_env_problems(pins, base_env)
        problems += [f"freeze evidence: {p}" for p in sd.freeze_evidence_problems(cfg, repo_root)]
    if problems:
        raise RunError("; ".join(problems))


def _dataset_problems(plan: dict, datasets: dict, *, formal: bool) -> tuple:
    """({key: sha256}, problems): every dataset this plan's workloads read, present, and pinned when formal."""
    need = sorted({f"{x['mix']}:{x['tier']}" for x in plan["workloads"].values() if x["mix"] in ("mixed", "var")})
    sha, problems = {}, []
    for key in need:
        path = datasets.get(key)
        if not path or not Path(path).is_file():
            problems.append(f"{key} needs --dataset {key}=<file> (039)")
            continue
        sha[key] = rc._sha256_file(path)
        mix, tier = key.split(":")
        pinned = (plan["mix_dataset_sha256"] if mix == "mixed" else plan["var_dataset_sha256"]).get(tier)
        if formal and sha[key] != pinned:
            problems.append(f"{key}: sha256 {sha[key]} != pinned {pinned!r} (039)")
    return sha, problems


def run_session(plan: dict, *, cfg: dict, pins: dict, pins_sha256: str, stage_d_sha256: str, out_dir, session_id: str,
                mode: str = "formal", popen: Callable, signal_group: Callable, group_alive: Callable,
                http_get: Callable, run_bench: Callable, gpu_sample: Callable, compute_apps: Callable,
                shared_usage: Callable, lock_held_check: Callable | None = None, gpu_lock_held: bool = False,
                vllm_bin: str = ".venv/bin/vllm", datasets: dict | None = None, gateway_python: str = sys.executable,
                port_free: Callable = default_port_free,
                sample_interval: float | None = 1.0, toolchain_check: Callable = bp.toolchain_problems,
                capture_env: Callable | None = None, base_env=None, now: Callable = time.monotonic,
                sleep: Callable = time.sleep, clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
                repo_root=ROOT) -> dict:
    """One Stage D session in `mode` ("formal" or "mini"); the session record (also session.json)."""
    if mode not in MODES:
        raise RunError(f"unknown mode {mode!r}; run_stage_d runs {MODES}")
    base_env = dict(os.environ if base_env is None else base_env)
    lock_held_check = lock_held_check or ra.held_gpu_lock
    _refusals(plan, cfg=cfg, pins=pins, pins_sha256=pins_sha256, stage_d_sha256=stage_d_sha256, out_dir=out_dir,
              mode=mode, lock_held_check=lock_held_check, gpu_lock_held=gpu_lock_held, base_env=base_env,
              repo_root=repo_root)
    formal = mode == "formal"
    session_plan = build_mini_plan(plan) if mode == "mini" else plan
    datasets = dict(datasets or {})
    data_sha, problems = _dataset_problems(session_plan, datasets, formal=formal)
    if problems:
        raise RunError("; ".join(problems))
    server_env = sd.server_argv_for(pins, "U2K", port=PORT, vllm_bin=vllm_bin, base_env=base_env)[1]
    missing = toolchain_check(server_env)
    if missing:
        raise RunError("toolchain incomplete, not starting (014): " + "; ".join(missing))
    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten (rerun under a new id)")
    for d in ("raw", "env", "derived"):
        (out / d).mkdir(parents=True)
    rc._write_json(out / "plan.json", session_plan)
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True) if capture_env else {}
    repo = pre.get("repo") or {}
    vllm = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
    if formal:
        if vllm != pins["vllm"]["version"]:
            raise RunError(f"vLLM {vllm!r} in the preflight != pinned {pins['vllm']['version']} (003; 030 8.6)")
        bad = env_problems(pre, rc.PLATFORM) + provenance_problems(pre, pre)
        if repo.get("source") != "DEPLOYED_COMMIT":
            bad.append(f"repo source {repo.get('source')!r}: the code is not a deployed bundle (one DEPLOYED_COMMIT)")
        if bad:
            raise RunError("preflight: " + "; ".join(bad))
    sb = pins["stage_b"]
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": session_plan["stage"],
               "session": plan["session"], "kind": plan["kind"], "mode": mode, "formal": formal,
               "gpu_lock_held": gpu_lock_held, "plan_sha256": sha256_json(session_plan), "pins_sha256": pins_sha256,
               "stage_d_sha256": stage_d_sha256, "plan_pins_sha256": plan.get("pins_sha256"),
               "plan_stage_d_sha256": plan.get("stage_d_sha256"), "started_at": clock(),
               "preflight": "env/preflight.json" if capture_env else None, "repo_commit": repo.get("commit"),
               "repo_source": repo.get("source"), "vllm": vllm,
               "server": {"gdn_prefill_backend": sb["gdn_prefill_backend"],
                          "gpu_memory_utilization": sb["gpu_memory_utilization"], "port": PORT, "vllm_bin": vllm_bin},
               "datasets": {k: str(v) for k, v in datasets.items()}, "dataset_sha256": data_sha,
               "server_env": ra.recorded_server_env(server_env), "inherited_unlisted": bp.inherited_unlisted(base_env),
               "gpu_idle": gi.criterion(sb["v_total_mib"]), "boots": [], "runs": {}, "derived_sha256": {},
               "out": str(out), "result_dir": str(out / "raw"), "gateway_python": gateway_python,
               "harness_sha256": sd.harness_sha256(ROOT),
               "capserve_sha256": capserve_sha256(ROOT, (plan.get("verify") or {}).get("gateway_profile"))
               if plan["kind"] == "verify" else None}
    fh = open(out / "rows.jsonl", "w")
    h = StageDHarness(plan=session_plan, pins=pins, out=out, session=session, formal=formal, pre=pre, rows_fh=fh,
                      popen=popen, signal_group=signal_group, group_alive=group_alive, http_get=http_get,
                      run_bench=run_bench, gpu_sample=gpu_sample, compute_apps=compute_apps,
                      shared_usage=shared_usage, lock_held_check=lock_held_check, capture_env=capture_env,
                      base_env=base_env, vllm_bin=vllm_bin, sample_interval=sample_interval, now=now, sleep=sleep,
                      clock=clock, datasets=datasets, gateway_python=gateway_python, port_free=port_free)

    def write_session() -> None:
        rc._write_json(out / "session.json", session)

    try:
        try:
            with mini_num_prompts() if mode == "mini" else contextlib.nullcontext():
                if session_plan["kind"] == "verify":
                    _verify_machine(h)
                elif session_plan["kind"] == "open":
                    _search_machine(h, write_session, search_next=sd.open_search_next,
                                    build_confirm=sd.build_open_confirm_plan)
                else:
                    rc._state_machine(h, write_session)
                if mode == "mini":
                    session["smoke"] = mini_checks(session_plan, session, h.rows, h.requests)
        finally:
            fh.close()
    except BaseException as exc:
        session["aborted"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.update(finished_at=clock(), outcomes=rc._outcomes(h.rows, session), env_flags=h.env_flags,
                       env_valid=not h.env_flags)
        if not formal:
            if "smoke" not in session:
                session["smoke"] = {"ok": False, "aborted": session.get("aborted")}
            rc._write_json(out / "smoke.json", session["smoke"])
            rc.mask_smoke(out)
            for rec in session["runs"].values():
                rec.pop("startup_s", None)
        session["rows_sha256"] = rc._sha256_file(out / "rows.jsonl")
        write_raw_manifest(out)
        write_session()
    return session


def main(argv: list | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="stage_d_plan.py output")
    parser.add_argument("--out", required=True, help="results/stage_d/<session>")
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--dataset", action="append", default=[], help="mixed:8192=path or var:<tier>=path (039)")
    parser.add_argument("--gpu-lock-held", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", dest="mode", action="store_const", const="formal")
    modes.add_argument("--mini", dest="mode", action="store_const", const="mini")
    args = parser.parse_args(argv)
    cfg_raw, pins_raw = sd.load_stage_d_bytes(), cp.load_pins_bytes()
    rb = rc._rb()
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    try:
        summary = run_session(
            json.loads(Path(args.plan).read_text()), cfg=tomllib.loads(cfg_raw.decode()),
            pins=tomllib.loads(pins_raw.decode()), pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
            stage_d_sha256=hashlib.sha256(cfg_raw).hexdigest(), out_dir=args.out, session_id=Path(args.out).name,
            mode=args.mode, popen=ra.start_server, signal_group=smoke.signal_group, group_alive=smoke.group_alive,
            http_get=rb.default_http_get, run_bench=ra.default_run_bench, gpu_sample=gi.nvidia_sample,
            compute_apps=rc.compute_apps_bounded, shared_usage=gi.read_shared_usage, lock_held_check=ra.held_gpu_lock,
            gpu_lock_held=args.gpu_lock_held, vllm_bin=args.vllm_bin,
            capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py"),
            datasets=dict(x.split("=", 1) for x in args.dataset), sample_interval=1.0)
    except (RunError, rc.RunError, ra.RunError) as exc:
        print(f"run_stage_d: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": summary.get("mode"), "outcomes": summary.get("outcomes"),
                      "smoke": summary.get("smoke")}))
    if args.mode == "formal":
        return 0
    return 0 if (summary.get("smoke") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
