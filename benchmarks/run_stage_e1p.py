#!/usr/bin/env python3
"""Stage E1-P runner (decision 041 v3 §1.6-1.7): the seed-swap probe's boots and benches under the GPU lock, with
028 / C0's stop rules (run_c0_probe's sampler, watched bench and constants, imported unchanged) and 024's observation
(run_adv's OTLP receiver, the iteration log, adv_decomp's alignment).

Per boot (stage_e1p_plan's plan): no new boot when elapsed + boot_margin_s > cap_s ("cap"); this process still holds
the lock ("lock"); the 031 idle window ("gpu_not_idle"; an nvidia-smi failure "gpu_check"); the spill guard's
pre-boot reading ("spill_unreadable"); env capture; the nvidia-smi sampler (>= 88 C "temperature", lost telemetry
"telemetry"); one OTLP receiver for the boot; the observed B0 server (stage_e1p_plan.server_argv_for); /health within
the startup timeout or what is left of the cap (a start cut at the cap stops "cap"); the start fails unless FLASH_ATTN,
exactly B0's watched non-default args (run_adv.manipulation_problems) and /metrics' 248 blocks of 528
(run_adv.blocks_problems); the ready spill reading. Then its benches in order: none whose 720 s timeout could pass the
cap ("cap"); /metrics before, the 1 Hz series, `vllm bench serve` with the bench's seed, N and request-id prefix
(stage_e1p_plan.bench_argv_for; its process group killed at the timeout or at once on the sampler's halt), /health,
/metrics after; the bench's main spans (cmpl-<prefix><i>-0, waited for up to span_wait_s), adv_decomp.align, the
derived requests (adv_decomp.derived_requests plus e2e, client start and latency) and the steps stamped in the bench
window; E1-D's frozen per-run quantities on the bench (stage_e1.f_spikes, and at Core's N stage_e1.bench_episodes:
L1-L5, send lags) into derived/<bench>-e1.json (descriptive: an error never costs the bench).
A bench is invalid on exit != 0, a timeout or a stop, completed != N or failed > 0, input_lens not all 8,064, an
unhealthy server, blocks != 248, a spill (031), or a main request without a span or without F (041 v3 §1.6: F is the
measure). A timeout, a stop or an unhealthy server ends the boot; its remaining benches are not_run rows. A boot
failed when its server did not start or >= 2 of its benches are invalid; the 2nd failed boot stops the session
("failed_boots"); 88 C or lost telemetry stops it at once, a flag set during the teardown included. Nothing reads a
latency to stop or extend the session (028 rule 7).
Sessions: refused without --gpu-lock-held verified, for a plan other than build_plan's under this config, a toolchain
problem, an output that is not a new results/stage_e/<id>; a formal session also for PYTORCH_* or inherited engine
variables in the caller's env, a non-WSL-CUDA host, another vLLM, env / provenance problems at the preflight, or when
results/stage_e already holds a valid formal E1-P session or two attempts (stage_e1p_plan.formal_attempts). A
session.json stub (status running) is written first and replaced atomically; status refused when no server ever
started, aborted after a harness failure, finished otherwise (a stop rule included). Smoke mode checks 041 v3 §1.7
(smoke_checks) and then masks the latencies (bench results, logs, spans, Prometheus scrapes, steps, series, the
derived latencies and e1 files, the rows' and boots' latency fields).

Usage (WSL, research checkout, tools/rtx/gpu_status.sh FREE):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_stage_e1p.py --plan plan.json \\
        --out results/stage_e/<session> --gpu-lock-held (--smoke | --formal)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import adv_decomp as ad  # noqa: E402
import adv_smoke  # noqa: E402
import core_b_plan as bp  # noqa: E402
import gpu_idle as gi  # noqa: E402
import run_adv as ra  # noqa: E402
import run_c0_probe as rcp  # noqa: E402
import run_capacity as rc  # noqa: E402
import stage_e1 as se1  # noqa: E402
import stage_e1p_plan as pp  # noqa: E402
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest  # noqa: E402

SCHEMA_VERSION = 1
Stop = rcp.Stop
DERIVED_KEYS = ("idx", "cls", "client_ttft_ms", "arrival", "Q_ms", "P_ms", "server_ttft_ms", "O_ms", "F_ms",
                "prompt_tokens", "completion_tokens", "e2e_ms", "start_time", "latency_ms")
# what a smoke keeps of a row (counts and flags; 041 v3 §1.7: the smoke's latencies are not looked at)
MASK_ROW_KEYS = ("boot", "bench", "position", "tag", "seed_label", "seed", "prefix", "L", "mix", "C", "N", "ok",
                 "reasons", "timeout", "stopped_hot", "stopped_telemetry", "not_run", "healthy_after_bench",
                 "num_gpu_blocks", "spans_main", "f_computable", "input_lens_8064", "n_steps_window",
                 "derived_schema_ok", "derived", "e1", "e1_error", "spill")
MASK_DERIVED_KEYS = ("idx", "cls")
SMOKE_LATENCY_RAW = ("*.bench.json", "*.bench.log", "*.spans.jsonl", "*.metrics.before.txt", "*.metrics.after.txt",
                     "*.steps.jsonl", "*.series.jsonl")


class RunError(RuntimeError):
    pass


# ---------------------------------------------------------------- per-bench quantities (pure)

def derived_rows(aligned: dict, res: dict) -> list:
    """derived/<bench>-requests.jsonl (numbers only): adv_decomp's rows plus the span e2e (server clock) and the
    client's start time and latency (client clock; never combined with the server's)."""
    st, la = res.get("start_times") or [], res.get("latencies") or []
    out = []
    for row, r in zip(ad.derived_requests(aligned), aligned["requests"]):
        i = row["idx"]
        out.append({**row, "e2e_ms": None if r.get("e2e") is None else r["e2e"] * 1000,
                    "start_time": st[i] if i < len(st) else None,
                    "latency_ms": None if i >= len(la) or la[i] is None else la[i] * 1000})
    return out


def span_problems(aligned: dict, reqs: list, N: int) -> list:
    """041 v3 §1.6: F is the measure, so every main request needs its span and its F."""
    out = []
    if aligned["missing"]:
        out.append(f"{len(aligned['missing'])} of {N} main requests have no span (F needs every one; 041 v3 §1.6)")
    if aligned["duplicates"] or aligned["unexpected"]:
        out.append(f"duplicate / unexpected main spans {(aligned['duplicates'] + aligned['unexpected'])[:4]}")
    if not aligned["missing"] and any(r["F_ms"] is None for r in reqs):
        out.append("F is not computable for every request (a span without its scheduling time)")
    return out


def bench_e1(reqs: list, steps: list, e1cfg: dict, *, full: bool) -> dict:
    """E1-D's frozen per-run quantities on one bench (stage_e1): the position-0 F spikes with the anchored step count
    and, at Core's N (full), the slip episodes with L1-L5 (L3 from the span e2e, L4 from the server arrivals) and the
    client's send lags."""
    out = {"f_spikes": se1.f_spikes(reqs, steps, e1cfg), "episodes": None, "l1_in_normal_groups": None,
           "n_dec_median": None}
    if full:
        arrivals = [r["arrival"] for r in reqs]
        st, lat = [r["start_time"] for r in reqs], [r["latency_ms"] for r in reqs]
        lags = None if None in st + lat else se1.send_lags_by_idx(st, [s + v / 1000 for s, v in zip(st, lat)])
        out.update(se1.bench_episodes([r["client_ttft_ms"] for r in reqs], lat_ms=[r["e2e_ms"] for r in reqs],
                                      times_s=arrivals, cfg=e1cfg, F_ms=[r["F_ms"] for r in reqs],
                                      Q_ms=[r["Q_ms"] for r in reqs], steps=steps, arrivals_s=arrivals,
                                      send_lags=lags))
    return out


def _not_run(bench: dict, reason: str) -> dict:
    return {**{k: bench[k] for k in ("boot", "bench", "tag", "seed_label", "seed", "prefix", "N")},
            "position": bench["bench"], "L": pp.L, "mix": pp.MIX, "C": pp.C, "ok": False,
            "reasons": [f"not run: {reason}"], "timeout": False, "stopped_hot": False, "stopped_telemetry": False,
            "not_run": True}


# ---------------------------------------------------------------- one session's boots

class E1PRun:
    """The boots and benches of one session (028 / C0's stop rules; 024's observation)."""

    def __init__(self, *, plan: dict, pins: dict, out: Path, session: dict, e1cfg: dict, formal: bool, pre: dict,
                 rows_fh, popen: Callable, signal_group: Callable, group_alive: Callable, http_get: Callable,
                 run_bench: Callable, gpu_sample: Callable, compute_apps: Callable, shared_usage: Callable,
                 start_smi: Callable, receiver_factory: Callable, lock_held_check: Callable, capture_env: Callable,
                 base_env: dict, vllm_bin: str, sample_interval, now: Callable, sleep: Callable, wall: Callable,
                 t_start: float, startup_timeout_s: float):
        self.plan, self.pins, self.out, self.session, self.e1cfg = plan, pins, out, session, e1cfg
        self.formal, self.pre, self.rows_fh = formal, pre, rows_fh
        self.raw, self.derived = out / "raw", out / "derived"
        self.popen, self.signal_group, self.group_alive, self.http_get = popen, signal_group, group_alive, http_get
        self.run_bench, self.gpu_sample, self.compute_apps = run_bench, gpu_sample, compute_apps
        self.shared_usage, self.start_smi, self.receiver_factory = shared_usage, start_smi, receiver_factory
        self.lock_held_check, self.capture_env = lock_held_check, capture_env
        self.base_env, self.vllm_bin, self.sample_interval = base_env, vllm_bin, sample_interval
        self.now, self.sleep, self.wall, self.t_start = now, sleep, wall, t_start
        self.startup_timeout_s, self.cap_s = startup_timeout_s, plan["cap_s"]
        self.base = f"http://127.0.0.1:{pp.PORT}"
        self.rows, self.env_flags = [], []

    def elapsed(self) -> float:
        return self.now() - self.t_start

    def emit(self, row: dict) -> dict:
        self.rows.append(row)
        self.rows_fh.write(json.dumps(row) + "\n")
        self.rows_fh.flush()
        return row

    def snap(self, name: str) -> None:
        rec = self.capture_env(out_path=self.out / "env" / name, gpu_lock_held=True)
        self.env_flags.extend({"env": name, "reason": why} for why in env_problems(rec, "wsl2-cuda"))
        if self.formal:
            self.env_flags.extend({"env": name, "reason": why} for why in provenance_problems(rec, self.pre))

    def check_log(self, proc, when: str) -> None:
        """A server log that could not be written is a harness failure: abort, never a result."""
        error = getattr(getattr(proc, "log_reader", None), "error", None)
        if error:
            raise RunError(f"the server log could not be written ({error}) {when}: a harness failure; session aborted")

    def halted(self, smi, where: str) -> Stop:
        if smi.hot.is_set():
            return Stop("temperature", f"GPU temperature >= {self.plan['temp_stop_c']} C {where}: torn down and "
                                       "stopped (028 stop rule 6)")
        return Stop("telemetry", f"nvidia-smi telemetry lost {where} ({smi.lost_reason}): the 88 C stop (028 stop "
                                 "rule 6) cannot be applied; torn down and stopped")

    # ---------------------------------------------------------- boot
    def pre_boot(self, b: int) -> tuple:
        """028 rules 2 / 3 with 031's idle window, then the spill guard's pre-boot reading: (idle report, guard)."""
        tag = rcp.boot_tag(b)
        if self.elapsed() + self.plan["boot_margin_s"] > self.cap_s:
            raise Stop("cap", f"elapsed {self.elapsed():.0f} s + {self.plan['boot_margin_s']} s > cap "
                              f"{self.cap_s:.0f} s: no new boot (028 stop rule 3)")
        if not self.lock_held_check():
            raise Stop("lock", f"this process does not hold the GPU lock any more; stopping before {tag}")
        rule = self.session["gpu_idle"]
        try:
            ok, report = gi.wait_idle(self.gpu_sample, self.compute_apps, bound_mib=rule["bound_mib"], now=self.now,
                                      sleep=self.sleep)
        except Exception as exc:  # noqa: BLE001 - nvidia-smi failed or hung: the GPU state is unknown
            raise Stop("gpu_check", f"nvidia-smi failed before {tag}: {type(exc).__name__}: {exc}") from exc
        if not ok:
            raise Stop("gpu_not_idle", f"the GPU was not idle in {report['attempt']} windows of "
                                       f"{rule['window_samples']} samples before {tag}: "
                                       f"{'; '.join(report['problems'])} (031; 028 stop rule 2)", {"idle": report})
        guard = gi.SpillGuard(self.shared_usage)
        try:
            guard.start()
        except gi.SharedUsageError as exc:
            raise Stop("spill_unreadable", f"the pre-boot shared usage before {tag} is unreadable ({exc}): the 031 "
                                           "spill guard cannot be applied; stopped", {"spill": guard.record()}) from exc
        return report, guard

    def checked_start(self, tag: str, proc, rec: dict, smi) -> None:
        """/health within the startup timeout and what is left of the cap, then the manipulation and block checks."""
        budget = min(self.startup_timeout_s, self.cap_s - self.elapsed())
        t0, ready = self.now(), False
        while self.now() - t0 < budget and proc.poll() is None and not smi.halt.is_set():
            if self.http_get(f"{self.base}/health")[0] == 200:
                ready = True
                break
            self.sleep(max(0, min(2, budget - (self.now() - t0))))
        rec["startup_s"] = round(self.now() - t0, 1)
        log = rc._read(self.raw / f"{tag}.server.log")
        facts, nondefault, manip = ra.manipulation_problems(log, "B0", self.pins)
        code, text = self.http_get(f"{self.base}/metrics") if ready else (0, "")
        cache, blocks = ra.blocks_problems(text if code == 200 else "", expected_blocks=self.plan["expected_blocks"],
                                           block_size=self.plan["block_size"])
        if ready:
            reasons = []
        elif smi.hot.is_set():
            reasons = [f"GPU temperature >= {self.plan['temp_stop_c']} C during the start"]
        elif smi.halt.is_set():
            reasons = [f"nvidia-smi telemetry lost during the start ({smi.lost_reason})"]
        elif budget < self.startup_timeout_s and proc.poll() is None:
            rec["cut_at_cap"] = True
            reasons = [f"server not ready within {budget:.0f} s: the start was cut at the cap {self.cap_s:.0f} s"]
        else:
            reasons = [f"server not ready within {self.startup_timeout_s} s (exit code {proc.poll()!r})"]
        reasons += manip + blocks
        rec.update(startup=rcp.startup_facts(log), log_facts=facts, nondefault=nondefault, cache_config=cache,
                   startup_ok=not reasons, reasons=reasons)

    def guarded(self, row: dict, guard) -> Stop | None:
        """031 after a bench: its spill record and verdict; an unreadable reading invalidates it and stops."""
        try:
            entry = guard.check(row["tag"])
        except gi.SharedUsageError as exc:
            row["ok"] = False
            row["reasons"].append(f"spill guard unreadable after the bench ({exc}; 031: fail closed)")
            return Stop("spill_unreadable", f"the shared usage after {row['tag']} is unreadable ({exc}): the 031 "
                                            "spill guard cannot be applied; torn down and stopped")
        row["spill"] = guard.row_record()
        if entry["spilled"]:
            row["ok"] = False
            row["reasons"].append(rc.spill_reason(entry, guard))
        return None

    def teardown(self, tag: str, proc, rec: dict, smi, receiver) -> None:
        """Server group down (SIGKILL escalation included) and its log joined, the receiver closed, the sampler
        stopped and its samples written; interrupts arriving meanwhile wait until all of that is done."""
        with ra.DeferredSignals() as deferred:
            problems = []
            try:
                if proc is not None:
                    problems += ra.shutdown_group(proc, deferred, signal_group=self.signal_group,
                                                  group_alive=self.group_alive, now=self.now, sleep=self.sleep)
                    reader = getattr(proc, "log_reader", None)
                    if reader is not None and not reader.join(ra.LOG_JOIN_S):
                        problems.append("server log reader still open after teardown")
            finally:
                try:
                    problems += receiver.close() or []
                finally:
                    errors = list(getattr(receiver, "errors", None) or [])
                    if errors:
                        rec["receiver_errors"] = errors
                    smi_problems = smi.stop() or []
                    samples = smi.samples()
                    (self.derived / f"{tag}-smi.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
                    rec.update(smi=f"derived/{tag}-smi.jsonl", smi_summary=rcp.smi_summary(samples),
                               smi_problems=smi_problems)
                    if problems:
                        rec["teardown_problems"] = problems
                        self.session.setdefault("teardown_problems", []).append({"boot": tag, "reasons": problems})

    def boot_once(self, boot: dict, idle: dict, guard) -> Stop | None:
        b, benches = boot["boot"], boot["benches"]
        tag = rcp.boot_tag(b)
        self.snap(f"{tag}.before.json")
        argv, env = pp.server_argv_for(self.pins, vllm_bin=self.vllm_bin, base_env=self.base_env)
        rec = self.session["boots"][tag] = {"argv": argv, "env": ra.recorded_server_env(env), "t_start": self.wall(),
                                            "elapsed_at_start_s": round(self.elapsed(), 1), "server_started": False,
                                            "startup_ok": False, "reasons": [], "idle": idle}
        smi = self.start_smi(self.raw / f"{tag}.smi.csv")
        receiver = self.receiver_factory(self.raw / f"{tag}.otlp", self.raw / f"{tag}.spans.jsonl", f"e1p-b{b}-")
        proc, stop, done, why = None, None, 0, None
        try:
            receiver.start()
            proc = self.popen(argv, env, self.raw / f"{tag}.server.log")
            rec["server_started"] = True
            self.checked_start(tag, proc, rec, smi)
            self.check_log(proc, "during the server start")
            if smi.halt.is_set():
                stop = self.halted(smi, f"during the start of {tag}")
            elif rec.get("cut_at_cap"):
                stop = Stop("cap", f"the start of {tag} was cut at the cap {self.cap_s:.0f} s: no bench")
            elif not rec["startup_ok"]:
                why = "the server start failed: " + "; ".join(rec["reasons"])
            else:
                try:
                    ready = guard.check("ready")
                except gi.SharedUsageError as exc:
                    stop = Stop("spill_unreadable", f"the shared usage after {tag} became ready is unreadable ({exc}): "
                                                    "the 031 spill guard cannot be applied; no bench, stopped")
                else:
                    if ready["spilled"]:
                        why = f"the boot stopped before its first bench: {rc.spill_reason(ready, guard)}"
                for bench in benches:
                    if stop is not None or why is not None:
                        break
                    if self.elapsed() + self.plan["timeout_s"] > self.cap_s:
                        stop = Stop("cap", f"elapsed {self.elapsed():.0f} s + the {self.plan['timeout_s']} s bench "
                                           f"timeout > cap {self.cap_s:.0f} s: no new bench (028 hard cap)")
                        break
                    row, end = self.bench_once(bench, tag, proc, smi, receiver)
                    if not smi.halt.is_set():
                        stop = self.guarded(row, guard)
                    else:                              # 88 C / lost telemetry: torn down at once, no reading, so
                        row["ok"] = False              # nothing shows the bench did not spill (031: fail closed)
                        row["reasons"].append("no 031 spill reading after the bench: the stop tore the server down at "
                                              "once (031: fail closed)")
                    self.emit(row)
                    done = bench["bench"]
                    if smi.halt.is_set():
                        stop = self.halted(smi, f"in {row['tag']}")
                        break
                    if stop is not None:
                        break
                    if (row.get("spill") or {}).get("spilled"):
                        why = f"the boot stopped after a spill in {row['tag']} (031)"
                        break
                    if end:
                        why = f"the boot ended after {row['tag']} ({'; '.join(row['reasons'])})"
                        break
        finally:
            rec["spill"] = guard.record()
            self.teardown(tag, proc, rec, smi, receiver)
        for bench in benches[done:]:
            self.emit(_not_run(bench, stop.reason if stop else why))
        if stop is None and smi.halt.is_set():  # 88 C or lost telemetry while the server went down: still a stop
            stop = self.halted(smi, f"at the teardown of {tag}")
        self.check_log(proc, "at teardown")
        receiver.raise_if_failed()
        self.snap(f"{tag}.after.json")
        invalid = sum(not r["ok"] for r in self.rows if r["boot"] == b)
        rec.update(t_end=self.wall(), n_valid=len(benches) - invalid, failed=not rec["startup_ok"] or invalid >= 2)
        return stop

    # ---------------------------------------------------------- bench
    def bench_once(self, bench: dict, boot_tag: str, proc, smi, receiver) -> tuple:
        """One bench and its row; (row, end the boot)."""
        tag, N, prefix = bench["tag"], bench["N"], bench["prefix"]
        row = {**{k: bench[k] for k in ("boot", "bench", "tag", "seed_label", "seed", "prefix", "N")},
               "position": bench["bench"], "L": pp.L, "mix": pp.MIX, "C": pp.C, "ok": False, "reasons": [],
               "timeout": False, "stopped_hot": False, "stopped_telemetry": False, "not_run": False}
        url = f"{self.base}/metrics"
        t_before = self.wall()
        code0, before = self.http_get(url)
        before = before if code0 == 200 else ""
        (self.raw / f"{tag}.metrics.before.txt").write_text(before)
        cache, blocks = ra.blocks_problems(before, expected_blocks=self.plan["expected_blocks"],
                                           block_size=self.plan["block_size"])
        argv = pp.bench_argv_for(self.pins, seed=bench["seed"], N=N, prefix=prefix, result_dir=str(self.raw),
                                 result_filename=f"{tag}.bench.json", vllm_bin=self.vllm_bin)
        row["argv"] = argv
        sampler = ra.TimeSeriesSampler(self.http_get, url, self.sample_interval, wall=self.wall)
        sampler.start()
        try:
            code, timed_out, stopped = self.run_bench(argv, self.raw / f"{tag}.bench.log",
                                                      {**self.base_env, **bp.bench_env()}, self.plan["timeout_s"],
                                                      smi.halt)
        finally:
            samples = sampler.stop()
        self.check_log(proc, "during the bench")
        healthy = self.http_get(f"{self.base}/health")[0] == 200
        code1, after = self.http_get(url)
        after = after if code1 == 200 else ""
        t_after = self.wall()
        row.update(t_before=t_before, t_after=t_after)
        (self.raw / f"{tag}.metrics.after.txt").write_text(after)
        ra._write_jsonl(self.raw / f"{tag}.series.jsonl", samples)
        path, res, reasons = self.raw / f"{tag}.bench.json", None, []
        if code == 0 and not timed_out and not stopped and path.is_file():
            try:
                res = json.loads(path.read_text())
            except ValueError as exc:
                reasons.append(f"unreadable bench result ({exc})")
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {self.plan['timeout_s']} s; its process group was killed "
                           "and the server torn down (028 stop rule 4)")
        elif stopped and smi.hot.is_set():
            row["stopped_hot"] = True
            reasons.append(f"stopped: GPU temperature >= {self.plan['temp_stop_c']} C (028 stop rule 6)")
        elif stopped:
            row["stopped_telemetry"] = True
            reasons.append(f"stopped: nvidia-smi telemetry lost ({smi.lost_reason}; 028 stop rule 6)")
        elif res is None and not reasons:
            reasons.append(f"vllm bench serve exit {code}")
        if res is not None:
            fields = rcp.bench_fields(res, N=N)
            reasons += fields.pop("reasons")
            fields.pop("ok")
            row.update(fields, result=f"raw/{tag}.bench.json")
            lens = res.get("input_lens") or []
            row["input_lens_8064"] = len(lens) == N and all(x == pp.L - bp.N_GEN for x in lens)
        got = rc.wait_for_prefix_spans(receiver, prefix, N, now=self.now, sleep=self.sleep,
                                       limit_s=self.plan["span_wait_s"]) if res is not None else 0
        receiver.raise_if_failed()
        row["spans_main"] = got
        steps = ad.parse_iteration_log(rc._read(self.raw / f"{boot_tag}.server.log").splitlines())
        window = ad.steps_in_window(steps, t_before, t_after)
        ra._write_jsonl(self.raw / f"{tag}.steps.jsonl", window)
        row["n_steps_window"] = len(window)
        if res is not None:
            aligned = ad.align(ad.llm_request_records(receiver.read_spans()), res, prefix=prefix, L=pp.L)
            reqs = derived_rows(aligned, res)
            dpath = self.derived / f"{tag}-requests.jsonl"
            dpath.write_text("".join(json.dumps(q) + "\n" for q in reqs))
            row["derived"] = dpath.relative_to(self.out).as_posix()
            row["derived_sha256"] = hashlib.sha256(dpath.read_bytes()).hexdigest()
            row["derived_schema_ok"] = len(reqs) == N and all(tuple(q) == DERIVED_KEYS for q in reqs)
            missing = span_problems(aligned, reqs, N)
            row["f_computable"] = not missing
            reasons += missing
            if not missing:
                try:
                    e1 = bench_e1(reqs, window, self.e1cfg, full=N == bp.num_prompts(pp.C))
                    epath = self.derived / f"{tag}-e1.json"
                    epath.write_text(json.dumps(e1, indent=1) + "\n")
                    row["e1"] = epath.relative_to(self.out).as_posix()
                    row["e1_sha256"] = hashlib.sha256(epath.read_bytes()).hexdigest()
                except Exception as exc:  # noqa: BLE001 - the descriptive layer never costs a measured bench
                    row["e1_error"] = f"{type(exc).__name__}: {exc}"
        if not healthy:
            reasons.append("server unhealthy after the bench (028 stop rule 5); the boot ends")
        reasons += blocks
        row.update(ok=res is not None and not reasons, reasons=reasons, healthy_after_bench=healthy,
                   num_gpu_blocks=(cache or {}).get("num_gpu_blocks"), series=ra.series_summary(samples),
                   scrape_delta=ra.scrape_deltas(before, after))
        return row, timed_out or stopped or not healthy


# ---------------------------------------------------------------- smoke

def smoke_checks(plan: dict, rows: list, session: dict) -> dict:
    """041 v3 §1.7's pass conditions over the smoke's rows (before the latencies are masked)."""
    want = [(x["tag"], x["seed_label"]) for boot in plan["boots"] for x in boot["benches"]]
    run = [r for r in rows if not r.get("not_run")]
    boots = list(session["boots"].values())
    smi = [b.get("smi_summary") or {} for b in boots]
    checks = {"covered": [(r["tag"], r["seed_label"]) for r in run] == want and len(rows) == len(want),
              "not_stopped": session.get("stopped") is None,
              "every_bench_valid": bool(run) and all(r["ok"] for r in rows),
              "startup": len(boots) == len(plan["boots"]) and all(b.get("startup_ok") is True for b in boots),
              "spans": all(r.get("spans_main") == r["N"] for r in run),
              "f_computable": all(r.get("f_computable") is True for r in run),
              "steps_in_window": all((r.get("n_steps_window") or 0) > 0 for r in run),
              "input_lens_8064": all(r.get("input_lens_8064") is True for r in run),
              "derived_schema": all(r.get("derived_schema_ok") is True and r.get("e1") and not r.get("e1_error")
                                    for r in run),
              "blocks": all(r.get("num_gpu_blocks") == plan["expected_blocks"] for r in run),
              "no_spill": not any((r.get("spill") or {}).get("spilled") for r in rows),
              "smi": bool(smi) and all(s.get("n_samples") and "temp_c" not in s.get("fields_missing", ["temp_c"])
                                       for s in smi) and not any(b.get("smi_problems") for b in boots),
              "teardown_clean": not session.get("teardown_problems")}
    return {"ok": all(checks.values()), "checks": checks, "covered": [list(x) for x in want]}


def mask_smoke(out: Path, session: dict) -> list:
    """The smoke's latencies leave the session (041 v3 §1.7: not looked at): bench results and logs, spans and OTLP
    bodies, Prometheus scrapes, steps and series leave raw/; the derived rows keep idx / cls, the e1 files go; rows keep
    MASK_ROW_KEYS; the boots lose their latency fields (adv_smoke._drop_latency) and their wall-clock window. Server
    logs and the nvidia-smi samples stay (raw/ is git-ignored)."""
    gone, raw, derived = [], out / "raw", out / "derived"
    for pattern in SMOKE_LATENCY_RAW:
        for p in sorted(raw.glob(pattern)):
            p.unlink()
            gone.append(p.relative_to(out).as_posix())
    for d in sorted(raw.glob("*.otlp")):
        shutil.rmtree(d)
        gone.append(d.relative_to(out).as_posix())
    for p in sorted(derived.glob("*-e1.json")):
        p.unlink()
        gone.append(p.relative_to(out).as_posix())
    for p in sorted(derived.glob("*-requests.jsonl")):
        keep = [{k: q.get(k) for k in MASK_DERIVED_KEYS} for q in map(json.loads, p.read_text().splitlines())]
        p.write_text("".join(json.dumps(q) + "\n" for q in keep))
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines() if x.strip()]
    (out / "rows.jsonl").write_text("".join(json.dumps({k: r[k] for k in MASK_ROW_KEYS if k in r}) + "\n"
                                            for r in rows))
    session["boots"] = {tag: {k: v for k, v in rec.items() if k not in ("t_start", "t_end")}   # the boot's window
                        for tag, rec in adv_smoke._drop_latency(session["boots"]).items()}
    return gone


# ---------------------------------------------------------------- session

def run_session(plan: dict, *, pins: dict, pins_sha256: str, out_dir, session_id: str, mode: str,
                expected_plan: dict | None, popen: Callable, signal_group: Callable, group_alive: Callable,
                http_get: Callable, run_bench: Callable, gpu_sample: Callable, compute_apps: Callable,
                shared_usage: Callable, start_smi: Callable, receiver_factory: Callable, lock_held_check: Callable,
                gpu_lock_held: bool, capture_env: Callable, t_start: float, vllm_bin: str = ".venv/bin/vllm",
                base_env=None, toolchain_check: Callable = bp.toolchain_problems, sample_interval: float | None = 1.0,
                now: Callable = time.monotonic, sleep: Callable = time.sleep, wall: Callable = time.time,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
                startup_timeout_s: float = rcp.STARTUP_TIMEOUT_S, repo_root=ROOT) -> dict:
    """One E1-P session. t_start: now() when this process started under the lock (the cap's origin)."""
    base_env = dict(os.environ if base_env is None else base_env)
    if mode not in ("formal", "smoke") or plan.get("stage") != pp.STAGE or plan.get("mode") != mode:
        raise RunError(f"not an E1-P {mode} plan")
    out = Path(out_dir)
    if out.resolve().parent != Path(repo_root).resolve().joinpath(*pp.RESULTS):
        raise RunError(f"{out_dir} is not results/stage_e/<session id> (041 v3 §0)")
    if not (gpu_lock_held and lock_held_check()):
        raise RunError("E1-P boots hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... --gpu-lock-held "
                       "(verified, not trusted; MACHINE_POLICY)")
    formal = mode == "formal"
    problems = []
    if plan.get("harness_sha256") != pp.harness_sha256() or expected_plan is None or plan != expected_plan:
        problems.append(f"a {mode} plan must be exactly stage_e1p_plan.build_plan's (041 v3 §1.7, 凍結)")
    e1cfg = pp.e1_config()
    if sha256_json(e1cfg) != plan.get("stage_e1_sha256"):
        problems.append("configs/stage_e1.toml is not the frozen E1-D config the plan names")
    if formal:
        if any(k.startswith("PYTORCH_") for k in base_env):
            problems.append("the caller's env carries PYTORCH_* variables")
        inherited = bp.inherited_unlisted(base_env)
        if inherited:
            problems.append(f"inherited engine variables {inherited}: a formal session inherits none (030 8.6)")
        prior = pp.formal_attempts(repo_root)
        if len(prior) >= 2 or any(a["valid"] for a in prior):
            problems.append(f"formal E1-P attempts {[a['dir'].name for a in prior]} already exist: one valid "
                            "session, or one void attempt and its rerun, at most (041 v3 §0)")
    _, env = pp.server_argv_for(pins, vllm_bin=vllm_bin, base_env=base_env)
    missing = toolchain_check(env)
    if missing:
        problems.append("toolchain incomplete (014): " + "; ".join(missing))
    if problems:
        raise RunError("; ".join(problems))
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    for sub in ("raw", "env", "derived"):
        (out / sub).mkdir(parents=True)
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": pp.STAGE, "mode": mode,
               "formal": formal, "purpose": "exploratory", "decision": "041 v3 §1.6", "status": "running",
               "plan_sha256": sha256_json(plan), "pins_sha256": pins_sha256,
               "stage_e1p_sha256": plan["stage_e1p_sha256"], "harness_sha256": pp.harness_sha256(),
               "gpu_idle": gi.criterion(pins["stage_b"]["v_total_mib"]), "started_at": clock(),
               "cap_s": plan["cap_s"], "elapsed_at_start_s": round(now() - t_start, 1), "boots": {},
               "failed_boots": 0, "stopped": None}

    def write_session():                                    # atomic: a torn write never hides the attempt
        tmp = out / "session.json.tmp"
        tmp.write_text(json.dumps(session, indent=1) + "\n")
        os.replace(tmp, out / "session.json")

    write_session()                                         # the stub: an interrupted session is still an attempt
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True)
    session["host"] = {"system": (pre.get("os") or {}).get("system"), "wsl": (pre.get("os") or {}).get("wsl"),
                       "nvidia": bool((pre.get("gpu") or {}).get("nvidia"))}
    session["versions"] = rcp.runtime_versions(pre, pins)
    got = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
    session["vllm_version"] = got
    bad = [] if got == pins["vllm"]["version"] else [f"vLLM {got!r}, not the pinned {pins['vllm']['version']}"]
    if formal:
        bad += env_problems(pre, "wsl2-cuda") + provenance_problems(pre, pre) + pp.host_problems(session["host"])
    if bad:
        session.update(status="refused", refused=bad)
        write_session()
        raise RunError(f"{mode} E1-P session refused before it starts: " + "; ".join(bad))
    fh = open(out / "rows.jsonl", "w")
    run = E1PRun(plan=plan, pins=pins, out=out, session=session, e1cfg=e1cfg, formal=formal, pre=pre, rows_fh=fh,
                 popen=popen, signal_group=signal_group, group_alive=group_alive, http_get=http_get,
                 run_bench=run_bench, gpu_sample=gpu_sample, compute_apps=compute_apps, shared_usage=shared_usage,
                 start_smi=start_smi, receiver_factory=receiver_factory, lock_held_check=lock_held_check,
                 capture_env=capture_env, base_env=pp.strip_env(base_env), vllm_bin=vllm_bin,
                 sample_interval=sample_interval, now=now, sleep=sleep, wall=wall, t_start=t_start,
                 startup_timeout_s=startup_timeout_s)
    b = 0
    try:
        for boot in plan["boots"]:
            b = boot["boot"]
            stop = run.boot_once(boot, *run.pre_boot(b))
            if stop is not None:
                raise stop
            if session["boots"][rcp.boot_tag(b)]["failed"]:
                session["failed_boots"] += 1
                if session["failed_boots"] >= plan["max_failed_boots"]:
                    raise Stop("failed_boots", f"{session['failed_boots']} failed boots (028 stop rule 5)")
    except Stop as s:
        session["stopped"] = {"rule": s.rule, "reason": s.reason, "boot": b, **({"detail": s.detail} if s.detail
                                                                                  else {})}
    except BaseException as exc:
        session["aborted"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        fh.close()
        started = any(r.get("server_started") for r in session["boots"].values())
        status = ("aborted" if "aborted" in session else "finished") if started else "refused"
        rows = run.rows
        session.update({"status": status, "finished_at": clock(), "elapsed_s": round(now() - t_start, 1),
                        "env_flags": run.env_flags, "env_valid": not run.env_flags,
                        "outcomes": {"boots": len(session["boots"]), "benches": len(rows),
                                     "valid": sum(bool(r["ok"]) for r in rows),
                                     "invalid": sum(not r["ok"] for r in rows),
                                     "not_run": sum(bool(r.get("not_run")) for r in rows),
                                     "failed_boots": session["failed_boots"]}})
        if mode == "smoke":
            session["smoke"] = smoke_checks(plan, rows, session) if status == "finished" else \
                {"ok": False, "status": status, "aborted": session.get("aborted")}
            (out / "smoke.json").write_text(json.dumps(session["smoke"], indent=1) + "\n")
            mask_smoke(out, session)
        session["rows_sha256"] = hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
        write_raw_manifest(out)
        write_session()
    return session


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI, WSL only
    t_start = time.monotonic()           # flock starts this process once it holds the lock: the cap counts from here
    import coreb_smoke as smoke
    import run_core_b as rb
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--gpu-lock-held", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", dest="mode", action="store_const", const="formal")
    modes.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    args = parser.parse_args(argv)
    raw = (ROOT / "configs" / "pins.toml").read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(pp.CONFIG.read_text())
    pins_sha = hashlib.sha256(raw).hexdigest()
    previous = {sig: signal.signal(sig, smoke._exit_on_signal) for sig in (signal.SIGHUP, signal.SIGTERM)}
    try:
        plan = json.loads(Path(args.plan).read_text())
        expected = pp.build_plan(pins, cfg, mode=args.mode, pins_sha256=pins_sha)
        s = run_session(plan, pins=pins, pins_sha256=pins_sha, out_dir=args.out, session_id=Path(args.out).name,
                        mode=args.mode, expected_plan=expected, popen=ra.start_server, signal_group=smoke.signal_group,
                        group_alive=smoke.group_alive, http_get=rb.default_http_get, run_bench=rcp.run_bench_watched,
                        gpu_sample=gi.nvidia_sample, compute_apps=rcp.compute_apps_bounded,
                        shared_usage=gi.read_shared_usage,
                        start_smi=lambda path: rcp.SmiSampler(path, temp_stop_c=plan["temp_stop_c"]).start(),
                        receiver_factory=ra.receiver_factory_for(pins), lock_held_check=ra.held_gpu_lock,
                        gpu_lock_held=args.gpu_lock_held,
                        capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py"), t_start=t_start,
                        vllm_bin=args.vllm_bin)
    except (RunError, pp.PlanError) as exc:
        print(f"run_stage_e1p: {exc}", file=sys.stderr)
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({"status": s["status"], "stopped": s["stopped"], "outcomes": s["outcomes"],
                      "smoke": (s.get("smoke") or {}).get("ok")}))
    return 0 if args.mode == "formal" or (s.get("smoke") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
