#!/usr/bin/env python3
"""Stage E2 runner (decision 041 v3 §2.3-2.10): one 024 observed boot and one bench per planned boot, under the GPU
lock, with 031's idle window and spill guard (run_capacity._Harness) and Advanced's observability (run_adv: the OTLP
receiver, the iteration log, 1 Hz series, scrapes, adv_decomp).

Per boot: pre_boot (lock held by this process, 031 idle window, spill guard) -> the condition's server
(stage_e2_plan.server_argv_for) with its OTLP receiver -> /health -> the manipulation checks (FLASH_ATTN, exactly the
condition's non-default args incl. enable_prefix_caching, /metrics' num_gpu_blocks / block_size, cache_config_info's
enable_prefix_caching) -> the ready spill reading -> one `vllm bench serve` (stage_e2_plan.bench_argv_for) with the
1 Hz sampler -> spans, steps, the chunk check (T, T256, TK25), the PC hit check (PC-on hits within the shared prefix,
PC-off none), decomposition and derived requests -> the spill reading after the bench -> teardown.
A row is valid by adv_analysis.invalid_reasons plus spill, the PC checks and the cache_config_info label; after an
invalid row the rest of that cell and mix is skipped (024); the formal session stops between boots at the cap.
Smoke mode runs the smoke plan under run_capacity.mini_num_prompts(20), records the measured block counts and masks
the latencies (run_capacity.mask_smoke).

Usage (WSL, research checkout):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_stage_e2.py --plan plan.json \\
        --out results/stage_e/<session> --dataset mixed:2048=data/core_b/mixed-2048.jsonl \\
        --dataset shared:2048=data/stage_e/shared-2048.jsonl --gpu-lock-held (--formal | --smoke)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import signal
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import adv_analysis as aa  # noqa: E402
import adv_decomp as ad  # noqa: E402
import core_b_plan as bp  # noqa: E402
import gpu_idle as gi  # noqa: E402
import run_adv as ra  # noqa: E402
import run_capacity as rc  # noqa: E402
import stage_e2_plan as e2p  # noqa: E402
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest  # noqa: E402

SCHEMA_VERSION = 1
PREFIX_HITS = "vllm:prefix_cache_hits_total"            # confirmed on vLLM 0.30.0 by the WSL source check (§2.10)
HIT_FLOOR = 0.95                                        # 041 v3 §2.8
DECOMP_COND = {"B0": "B0", "T": "T", "T256": "T", "K25": "K3", "TK25": "T"}   # adv_decomp's knobs; PC: none


class RunError(rc.RunError):
    pass


def boot_tag(boot: dict) -> str:
    return f"b{boot['block']}-{boot['cell'].replace('@', '-c')}-{boot['mix']}-r{boot['restart']}"


def invalid_reasons(row: dict) -> list:
    m = row.get("manip") or {}
    out = aa.invalid_reasons(row)
    if (row.get("spill") or {}).get("spilled"):
        out.append("spill (031)")
    if m.get("pc_label_ok") is False:
        out.append("cache_config_info enable_prefix_caching")
    if m.get("pc_hits_ok") is False:
        out.append("prefix cache hits")
    return out


def pc_hits_problems(cond: str, cell: str, delta_hits, *, N: int, warmup: int, shared: int) -> list:
    if delta_hits is None:
        return [f"no {PREFIX_HITS} delta"] if cell.startswith("PC") else []
    if cell == "PC-on":
        lo, hi = HIT_FLOOR * N * shared, (N + warmup + 1) * shared
        if not lo <= delta_hits <= hi:
            return [f"prefix cache hits {delta_hits:.0f} outside [{lo:.0f}, {hi:.0f}] (041 v3 §2.8)"]
    elif delta_hits != 0:
        return [f"prefix cache hits {delta_hits:.0f} with prefix caching off (041 v3 §2.8)"]
    return []


class E2Harness(rc._Harness):
    def __init__(self, *, cfg: dict, datasets: dict, receiver_factory: Callable, wall: Callable, **kw):
        super().__init__(mixed_datasets=None, **kw)
        self.cfg, self.datasets, self.receiver_factory, self.wall = cfg, datasets, receiver_factory, wall

    def _base_row(self, boot: dict, tag: str) -> dict:
        return {**{k: boot[k] for k in ("block", "cell", "cond", "C", "L", "mix", "restart", "N", "warmup", "prefix",
                                         "timeout_s")},
                "tag": tag, "ok": False, "timeout": False, "reasons": [], "manip": {}, "valid": False}

    def run_boot(self, boot: dict) -> dict:
        tag = boot_tag(boot)
        idle = self.pre_boot(tag)
        self.snap(f"{tag}.before.json")
        sb = self.pins["stage_b"]
        argv, env = e2p.server_argv_for(boot["cond"], self.pins, self.cfg, gdn_prefill_backend=sb["gdn_prefill_backend"],
                                        gpu_memory_utilization=sb["gpu_memory_utilization"], port=e2p.PORT,
                                        vllm_bin=self.vllm_bin, base_env=self.base_env)
        record = self.session["runs"][tag] = {"argv": argv, "env": ra.recorded_server_env(env), "idle": idle}
        row = self._base_row(boot, tag)
        receiver = self.receiver_factory(self.raw / f"{tag}.otlp", self.raw / f"{tag}.spans.jsonl", boot["prefix"])
        receiver.start()
        proc = None
        try:
            proc = self.popen(argv, env, self.raw / f"{tag}.server.log")
            ready, detail = self.wait_ready(proc, record)
            metrics = self.metrics_text(ready)
            facts, nondefault, manip = e2p.manipulation_problems(rc._read(self.raw / f"{tag}.server.log"),
                                                                 boot["cond"], self.pins, self.cfg)
            expected = self.plan["expected_blocks"][boot["cond"]]
            cache, blocks = ra.blocks_problems(metrics, expected_blocks=expected, block_size=self.plan["block_size"])
            if not expected and self.plan["mode"] == "smoke" and cache is not None:
                blocks = [] if cache.get("block_size") == self.plan["block_size"] else blocks   # the smoke measures it
            label = e2p.prefix_caching_label(metrics)
            label_ok = label is (boot["cond"] == "PC")
            record.update(log_facts=facts, nondefault=nondefault, cache_config=cache, prefix_caching_label=label)
            row["manip"].update(facts_ok=not manip, blocks_ok=not blocks, pc_label_ok=label_ok,
                                num_gpu_blocks=(cache or {}).get("num_gpu_blocks"),
                                mamba_cache_mode=facts.get("mamba_cache_mode"))
            stop = ([] if ready else [f"server not ready{detail}"]) + manip + blocks
            if ready and not label_ok:
                stop.append(f"cache_config_info enable_prefix_caching {label!r} for condition {boot['cond']}")
            self.check_log(proc, "during the server start")
            if not stop and proc.poll() is not None:
                stop = [f"server exited (code {proc.poll()!r}) before its bench"]
            if not stop:
                entry = self.spill_check("ready", tag)
                if entry["spilled"]:
                    stop = [rc.spill_reason(entry, self.guard) + "; no bench runs on this boot"]
            if stop:
                row["reasons"] = stop
            else:
                self.bench_and_measure(boot, row, record, receiver, cache, proc)
                row["reasons"] += self.spill_row(row)
        finally:
            record["spill"] = self.guard.record() if self.guard else None
            self.teardown(tag, proc, record, receiver)
        self.check_log(proc, "at teardown")
        receiver.raise_if_failed()
        self.snap(f"{tag}.after.json")
        row["invalid_reasons"] = invalid_reasons(row)
        row["valid"] = not row["invalid_reasons"]
        return self.emit(row)

    def bench_and_measure(self, boot: dict, row: dict, record: dict, receiver, cache: dict, proc) -> None:
        tag, cond, mix = row["tag"], boot["cond"], boot["mix"]
        url = f"{self.base}/metrics"
        t_before = self.wall()
        code0, before = self.http_get(url)
        before = before if code0 == 200 else ""
        (self.raw / f"{tag}.metrics.before.txt").write_text(before)
        argv = e2p.bench_argv_for(self.pins, mix=mix, C=boot["C"], prefix=boot["prefix"], result_dir=str(self.raw),
                                  result_filename=f"{tag}.bench.json", datasets=self.datasets, vllm_bin=self.vllm_bin)
        row["argv"] = argv
        sampler = ra.TimeSeriesSampler(self.http_get, url, self.sample_interval, wall=self.wall)
        sampler.start()
        try:
            code, timed_out = self.run_bench(argv, self.raw / f"{tag}.bench.log", {**self.base_env, **bp.bench_env()},
                                             boot["timeout_s"])
        finally:
            samples = sampler.stop()
        self.check_log(proc, "during the bench")
        row["healthy_after_bench"] = self.http_get(f"{self.base}/health")[0] == 200
        code1, after = self.http_get(url)
        after = after if code1 == 200 else ""
        t_after = self.wall()
        record.update(t_before=t_before, t_after=t_after)
        (self.raw / f"{tag}.metrics.after.txt").write_text(after)
        ra._write_jsonl(self.raw / f"{tag}.series.jsonl", samples)
        path = self.raw / f"{tag}.bench.json"
        res = json.loads(path.read_text()) if code == 0 and not timed_out and path.is_file() else None
        waited = ra.wait_for_spans(receiver, boot["N"], now=self.now, sleep=self.sleep,
                                   limit_s=self.plan.get("span_wait_s", ra.SPAN_WAIT_S))[1] if res is not None else 0.0
        receiver.raise_if_failed()
        row["spans"] = {**receiver.stats(), "waited_s": round(waited, 1)}
        reasons = []
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {boot['timeout_s']} s; its process group was killed")
        elif res is None:
            reasons.append(f"vllm bench serve exit {code}")
        if res is not None:
            fields = ra.bench_fields(res, L=boot["L"], C=boot["C"], mix="uniform" if mix == "shared" else mix)
            reasons = fields.pop("reasons") + reasons
            row.update(fields)
            row["result"] = f"raw/{tag}.bench.json"
        try:
            pre = ad.counter_delta(before, after, ad.PREEMPTIONS)
        except ad.DecompError:
            pre = None
        try:
            hits = ad.counter_delta(before, after, PREFIX_HITS)
        except ad.DecompError:
            hits = None
        series = ra.series_summary(samples)
        waiting = [s.get("vllm:num_requests_waiting") for s in samples if s.get("vllm:num_requests_waiting") is not None]
        row.update(preemptions=None if pre is None else int(pre), prefix_cache_hits=hits, kv_peak=series["kv_peak"],
                   series=series, one_hz=ad.one_hz_shares(samples, nb=cache["num_gpu_blocks"], L=boot["L"]),
                   beta=(sum(w > 0 for w in waiting) / len(waiting)) if waiting else None,
                   scrape_delta=ra.scrape_deltas(before, after))
        shared = self.cfg["e2"]["shared_prefix"]["min_shared"]
        hit_problems = pc_hits_problems(cond, boot["cell"], hits, N=boot["N"], warmup=boot["warmup"], shared=shared) \
            if mix == "shared" else []
        if mix == "shared":
            row["manip"]["pc_hits_ok"] = not hit_problems
        reasons += hit_problems
        steps = ad.parse_iteration_log(rc._read(self.raw / f"{tag}.server.log").splitlines())
        ra._write_jsonl(self.raw / f"{tag}.steps.jsonl", steps)
        window = ad.steps_in_window(steps, t_before, t_after)
        row["n_steps_window"] = len(window)
        threshold = self.plan["thresholds"].get(cond)
        if threshold and (window or res is not None):
            chunk = ra.chunk_problems(window, threshold)
            row["manip"]["chunk_ok"] = not chunk
            reasons += chunk
        if res is not None:
            spans = receiver.read_spans()
            try:
                if cond in DECOMP_COND:
                    row["decomp"] = ad.summarize_run(
                        spans=spans, detailed=res, steps=steps, prefix=boot["prefix"], L=boot["L"],
                        cond=DECOMP_COND[cond], num_gpu_blocks=cache["num_gpu_blocks"], metrics_before=before,
                        metrics_after=after, window=(t_before, t_after), block_size=self.plan["block_size"],
                        long_prefill_token_threshold=threshold or self.pins["advanced"]["long_prefill_token_threshold"])
                else:
                    row["decomp"] = "n/a (prefix caching changes the token accounting; 041 v3 §2.8)"
                records = [r for r in ad.llm_request_records(spans)
                           if r.get("arrival") is not None and t_before <= r["arrival"] <= t_after]
                ra._write_jsonl(self.derived / f"{tag}-requests.jsonl",
                                ad.derived_requests(ad.align(records, res, prefix=boot["prefix"], L=boot["L"])))
                row["derived"] = f"derived/{tag}-requests.jsonl"
            except Exception as exc:  # noqa: BLE001 - the descriptive layer never costs a measured Y
                row["decomp"] = None
                row["decomp_error"] = f"{type(exc).__name__}: {exc}"
        row["ok"] = bool(res is not None and row.get("ok") and not timed_out)
        row["reasons"] = reasons


def _skipped(boot: dict, why: str) -> dict:
    return {**{k: boot[k] for k in ("block", "cell", "cond", "C", "L", "mix", "restart", "N", "warmup", "prefix",
                                     "timeout_s")},
            "tag": boot_tag(boot), "ok": False, "skipped": True, "valid": False, "reasons": [why], "manip": {}}


def smoke_checks(plan: dict, rows: list) -> dict:
    by = {(r["cell"], r["mix"]): r for r in rows}
    checks = {"every_boot_valid": all(r["valid"] for r in rows),
              "pc_on_hits": (by.get(("PC-on", "shared")) or {}).get("manip", {}).get("pc_hits_ok") is True,
              "pc_off_no_hits": (by.get(("PC-off", "shared")) or {}).get("manip", {}).get("pc_hits_ok") is True,
              "chunk_checks": all(r["manip"].get("chunk_ok") is True for r in rows if r["cond"] in e2p.CHUNKED),
              "derived": all(r.get("derived") for r in rows), "no_spill": not any((r.get("spill") or {}).get("spilled")
                                                                                  for r in rows)}
    blocks = {r["cond"]: r["manip"].get("num_gpu_blocks") for r in rows}
    return {"ok": all(checks.values()), "checks": checks, "measured_blocks": blocks,
            "mamba_cache_mode": {r["cond"]: r["manip"].get("mamba_cache_mode") for r in rows},
            "memory_after_bench": {r["tag"]: (r.get("series") or {}).get("kv_peak") for r in rows}}


def run_session(plan: dict, *, pins: dict, cfg: dict, pins_sha256: str, stage_e2_sha256: str, out_dir,
                session_id: str, mode: str, popen: Callable, signal_group: Callable, group_alive: Callable,
                http_get: Callable, run_bench: Callable, gpu_sample: Callable, compute_apps: Callable,
                shared_usage: Callable, lock_held_check: Callable, gpu_lock_held: bool, capture_env: Callable,
                datasets: dict, receiver_factory: Callable, vllm_bin: str = ".venv/bin/vllm", base_env=None,
                expected_plan: dict | None = None, toolchain_check: Callable = bp.toolchain_problems,
                sample_interval: float = 1.0, now: Callable = time.monotonic, sleep: Callable = time.sleep,
                wall: Callable = time.time, clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
                repo_root=ROOT) -> dict:
    base_env = dict(os.environ if base_env is None else base_env)
    if mode not in ("formal", "smoke") or plan.get("stage") != e2p.STAGE or plan.get("mode") != mode:
        raise RunError(f"not an E2 {mode} plan")
    root_out = Path(repo_root).resolve().joinpath(*e2p.RESULTS)
    out = Path(out_dir)
    if out.resolve() == root_out or root_out not in out.resolve().parents:
        raise RunError(f"{out_dir} is not a session directory under results/stage_e/ (041)")
    if not (gpu_lock_held and lock_held_check()):
        raise RunError("E2 boots hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... --gpu-lock-held "
                       "(verified, not trusted; MACHINE_POLICY)")
    formal = mode == "formal"
    problems = []
    if formal:
        if plan.get("harness_sha256") != e2p.harness_sha256() or expected_plan is None or plan != expected_plan:
            problems.append("a formal plan must be exactly stage_e2_plan.build_plan's (041 凍結)")
        if any(k.startswith("PYTORCH_") for k in base_env):
            problems.append("the caller's env carries PYTORCH_* variables (041 v3 §2.3)")
        inherited = bp.inherited_unlisted(base_env)
        if inherited:
            problems.append(f"inherited engine variables {inherited}: a formal session inherits none (030 8.6)")
    used = sorted({b["dataset"] for b in plan["boots"] if b["dataset"]})
    dataset_sha = {}
    for key in used:
        path = datasets.get(key)
        if not path or not Path(path).is_file():
            problems.append(f"dataset {key} missing (--dataset {key}=<file>)")
            continue
        dataset_sha[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        want = plan["datasets_sha256"].get(key)
        if (formal or key == "mixed:2048") and dataset_sha[key] != want:
            problems.append(f"dataset {key} sha256 {dataset_sha[key]} is not the pinned {want}")
    missing = toolchain_check(e2p.ap.server_env_for(e2p.strip_env(base_env), pins))
    if missing:
        problems.append("toolchain incomplete (014): " + "; ".join(missing))
    if problems:
        raise RunError("; ".join(problems))
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    for sub in ("raw", "env", "derived"):
        (out / sub).mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True)
    host = {"system": (pre.get("os") or {}).get("system"), "wsl": (pre.get("os") or {}).get("wsl"),
            "nvidia": bool((pre.get("gpu") or {}).get("nvidia"))}
    if formal:
        bad = env_problems(pre, "wsl2-cuda") + provenance_problems(pre, pre) + e2p.host_problems(host)
        got = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
        if got != pins["vllm"]["version"]:
            bad.append(f"vLLM {got!r}, not the pinned {pins['vllm']['version']}")
        if bad:
            raise RunError("formal E2 session refused before it starts: " + "; ".join(bad))
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": e2p.STAGE, "mode": mode,
               "formal": formal, "plan_sha256": sha256_json(plan), "pins_sha256": pins_sha256,
               "stage_e2_sha256": stage_e2_sha256, "harness_sha256": e2p.harness_sha256(), "host": host,
               "datasets_sha256": dataset_sha, "gpu_idle": gi.criterion(pins["stage_b"]["v_total_mib"]),
               "started_at": clock(), "boots": [], "runs": {}}
    rows = []
    with open(out / "rows.jsonl", "w") as fh:
        h = E2Harness(plan={**plan, "server_ready_timeout_s": plan.get("server_ready_timeout_s", ra.STARTUP_TIMEOUT_S)},
                      pins=pins, out=out, session=session, formal=formal, pre=pre, rows_fh=fh, popen=popen,
                      signal_group=signal_group, group_alive=group_alive, http_get=http_get, run_bench=run_bench,
                      gpu_sample=gpu_sample, compute_apps=compute_apps, shared_usage=shared_usage,
                      lock_held_check=lock_held_check, capture_env=capture_env, base_env=e2p.strip_env(base_env),
                      vllm_bin=vllm_bin, sample_interval=sample_interval, now=now, sleep=sleep, clock=clock, cfg=cfg,
                      datasets=datasets, receiver_factory=receiver_factory, wall=wall)
        invalid_cells, t0 = {}, now()
        ctx = rc.mini_num_prompts(e2p.SMOKE_N) if mode == "smoke" else contextlib.nullcontext()
        try:
            with ctx:
                for boot in plan["boots"]:
                    cell = (boot["cell"], boot["mix"])
                    if plan.get("cap_s") and now() - t0 > plan["cap_s"]:
                        session["cap_reached"] = True
                        rows.append(h.emit(_skipped(boot, f"time cap {plan['cap_s']:.0f} s reached (041 v3 §2.9)")))
                    elif cell in invalid_cells:
                        rows.append(h.emit(_skipped(boot, f"cell {cell} had an invalid run ({invalid_cells[cell]}; "
                                                          "024)")))
                    else:
                        row = h.run_boot(boot)
                        rows.append(row)
                        if not row["valid"]:
                            invalid_cells[cell] = row["tag"]
        except BaseException as exc:
            session["aborted"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            session.update({"finished_at": clock(), "env_flags": h.env_flags, "env_valid": not h.env_flags,
                            "outcomes": {"boots": len(rows), "valid": sum(r["valid"] for r in rows),
                                         "skipped": sum(bool(r.get("skipped")) for r in rows)}})
            if mode == "smoke":
                session["smoke"] = smoke_checks(plan, rows) if "aborted" not in session else \
                    {"ok": False, "aborted": session["aborted"]}
                (out / "smoke.json").write_text(json.dumps(session["smoke"], indent=1) + "\n")
                rc.mask_smoke(out)
            session["rows_sha256"] = hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
            write_raw_manifest(out)
            (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI, WSL only
    import coreb_smoke as smoke
    import run_core_b as rb
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", action="append", default=[], help="mixed:2048=path / shared:2048=path")
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--gpu-lock-held", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", dest="mode", action="store_const", const="formal")
    modes.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), e2p.CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    plan = json.loads(Path(args.plan).read_text())
    try:
        expected = e2p.build_plan(pins, cfg, mode="formal", pins_sha256=hashlib.sha256(raw).hexdigest(),
                                  stage_e2_sha256=hashlib.sha256(cfg_raw).hexdigest()) if args.mode == "formal" else None
        s = run_session(plan, pins=pins, cfg=cfg, pins_sha256=hashlib.sha256(raw).hexdigest(),
                        stage_e2_sha256=hashlib.sha256(cfg_raw).hexdigest(), out_dir=args.out,
                        session_id=Path(args.out).name, mode=args.mode, popen=ra.start_server,
                        signal_group=smoke.signal_group, group_alive=smoke.group_alive, http_get=rb.default_http_get,
                        run_bench=ra.default_run_bench, gpu_sample=gi.nvidia_sample,
                        compute_apps=rc.compute_apps_bounded, shared_usage=gi.read_shared_usage,
                        lock_held_check=ra.held_gpu_lock, gpu_lock_held=args.gpu_lock_held,
                        capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py"),
                        datasets=dict(x.split("=", 1) for x in args.dataset), receiver_factory=ra.receiver_factory_for(pins),
                        vllm_bin=args.vllm_bin, expected_plan=expected)
    except (RunError, rc.RunError, ra.RunError, e2p.PlanError) as exc:
        print(f"run_stage_e2: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"outcomes": s["outcomes"], "smoke": (s.get("smoke") or {}).get("ok")}))
    return 0 if args.mode == "formal" or (s.get("smoke") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
