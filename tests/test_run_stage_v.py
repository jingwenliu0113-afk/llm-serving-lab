"""Tests for benchmarks/run_stage_v.py (decision 044): whole Stage V sessions against a fake WSL (vLLM from
test_run_stage_d's world; capserve serve with its --allow-provisional rule, its /readyz from the settings and the
profile it was given, and its access log; the clock, env capture and `vllm bench serve`; no GPU). Each bench's requests
and access-log lines come from test_stage_v_analysis's closed-loop FIFO gateway, and every session is replayed with
stage_v_analysis. Real execution happens in WSL under the GPU lock: the mini before the freeze, the two formal sessions
after it."""

import ast
import copy
import datetime as dt
import hashlib
import json
import shutil
import signal
import sys
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import run_capacity as rc
import run_stage_d as rd
import run_stage_v as rv
import stage_d_plan as sd
import stage_v_analysis as sva
import stage_v_plan as sv
import test_run_stage_d as td
import test_stage_v_analysis as tva

ROOT = Path(__file__).resolve().parent.parent
PINS = td.PINS
CFG = tva.CFG
SD_RAW = (ROOT / "configs" / "stage_d.toml").read_bytes()
SD_CFG = tomllib.loads(SD_RAW.decode())
SD_SHA = hashlib.sha256(SD_RAW).hexdigest()
PINS_SHA, SV_SHA = "ab" * 32, "cd" * 32
MINI_ID = "smoke-v-test"
T0 = dt.datetime(2026, 11, 2, 9, 0, tzinfo=dt.timezone.utc)
DEFAULT_PREFIX = "bench-1a2b3c4d-"          # vLLM 0.30's own request-id prefix without --request-id-prefix


def stamped(lines, ms=1761000000000):
    """Lines as run_adv's log reader writes them: the wall clock in ms and a space before each."""
    return "".join(f"{ms} {line}\n" for line in lines)


def readyz_body(argv):
    """What capserve's /readyz reports for `capserve serve --config F --profile P`: the profile's id and status, the
    active class's limits (weighted-v1: its envelope), the queue settings, the policies and this checkout's request
    path."""
    cfg = tomllib.loads(Path(argv[5]).read_text())
    prof = json.loads(Path(argv[7]).read_text())
    classes = {c["name"]: c for c in prof["classes"]}
    adm = cfg.get("admission")
    if adm:
        names = [c["name"] for c in adm["classes"]]
        limits = {"max_concurrency": max(c["max_concurrency"] for c in adm["classes"]), "max_queue": adm["max_queue"],
                  "max_pending": adm["budget"] + adm["max_queue"] + 16,
                  "max_prompt_tokens": max(classes[n]["max_prompt_tokens"] for n in names),
                  "max_output_tokens": classes[names[0]]["max_output_tokens"]}
        admission = {**adm, "classes": [{**a, "max_prompt_tokens": classes[a["name"]]["max_prompt_tokens"],
                                         "max_output_tokens": classes[a["name"]]["max_output_tokens"]}
                                        for a in adm["classes"]]}
    else:
        c = classes[cfg["active_class"]]
        limits = {k: c[k] for k in ("max_concurrency", "max_queue", "max_prompt_tokens", "max_output_tokens")}
        limits["max_pending"] = c["max_concurrency"] + c["max_queue"] + 16
        admission = None
    return {"ready": True, "not_ready_reason": None,
            "profile": {"id": prof["profile_id"], "status": prof["status"],
                        "provisional": prof["status"] != "verified"},
            "serving_status": prof["status"], "active_class": cfg.get("active_class"), "limits": limits,
            "queue": {"max_queue_wait_s": cfg.get("max_queue_wait_s", 2.0),
                      "request_deadline_s": cfg.get("request_deadline_s", 300.0)},
            "admission": admission, "admission_state": None if admission is None else {"units_in_flight": 0},
            "prefill_pacing": cfg.get("prefill_pacing"), "gateway_path_sha256": sv.gateway_path_sha256(),
            "gateway_validated": None}


class Jitter(dict):
    """Service times (s) by units for test_stage_v_analysis.closed_loop that vary a little from request to request, as
    a real vLLM's do: equal times would make a whole wave arrive at one instant (an empty T_W in the mini)."""

    def __init__(self):
        super().__init__({1: 1.0, 4: 1.5})
        self.k = 0

    def __getitem__(self, units):
        self.k += 1
        return super().__getitem__(units) * (1 + 0.013 * (self.k % 7))


class World(td.World):
    """test_run_stage_d's fake WSL plus 044's gateway: capserve serve exits (2) on a provisional profile without
    --allow-provisional; /readyz is readyz_body unless readyz(body, label) edits it; a V bench's requests are
    test_stage_v_analysis.make_reqs(plan, point, block, **kw(point, block)) and its access-log lines carry the bench's
    --request-id-prefix (vLLM's own default without one). The main run's access-log lines reach the log file only when
    the gateway stops (the reader drains the pipe at EOF): a log parsed before gateway_down misses them. dies: (point,
    block) whose gateway exits during the bench; slow(point, block) -> extra seconds a bench takes."""

    def __init__(self, tmp_path, plan, *, kw=None, readyz=None, dies=(), slow=None, **k):
        super().__init__(tmp_path, **k)
        self.plan, self.kw, self.readyz_edit, self.dies, self.slow = plan, kw or tva.ok_kw, readyz, set(dies), slow
        self.gw_log, self.buffered = None, []

    def clock(self):
        return (T0 + dt.timedelta(seconds=self.t)).isoformat()

    def popen(self, argv, env, log_path):
        if "capserve" not in argv:
            return super().popen(argv, env, log_path)
        self.events.append("gateway:start")
        self.gateway_argvs.append(argv)
        self.gateway_envs.append(env)
        self.gw, self.gw_log = td.Proc(88), Path(log_path)
        self.gw_log.write_text(stamped(["INFO:     Started server process [88]"]))
        if json.loads(Path(argv[7]).read_text())["status"] != "verified" and "--allow-provisional" not in argv:
            self.gw.returncode = 2                       # capserve serve refuses a provisional profile (exit 2)
        return self.gw

    def signal_group(self, proc, sig):
        if proc is self.gw and self.buffered:
            with self.gw_log.open("a") as f:
                f.write("".join(self.buffered))
            self.buffered = []
            self.events.append("gateway:log-drained")
        super().signal_group(proc, sig)

    def http_get(self, url):
        if ":8080/readyz" in url:
            if not (self.gw and self.gw.returncode is None):
                return 0, "refused"
            body = readyz_body(self.gateway_argvs[-1])
            if self.readyz_edit:
                self.readyz_edit(body, self.label)
            return 200, json.dumps(body)
        return super().http_get(url)

    def run_bench(self, argv, log_path, env, timeout_s):
        a = dict(zip(argv, argv[1:]))
        tag = a["--result-filename"].removesuffix(".bench.json")
        if not tag.startswith("V-"):
            return super().run_bench(argv, log_path, env, timeout_s)
        self.point = f"bench:{tag}"
        self.events.append(f"bench:{tag}")
        self.bench_argvs.append(argv)
        Path(log_path).write_text("log\n")
        if timeout_s < 10:                                   # every fake bench takes 10 s: a shorter timeout hits
            self.t += timeout_s
            return -9, True
        head, block = tag.rsplit("-r", 1)
        point, block = head[2:].rsplit("-c", 1)[0], int(block)
        x = self.plan["points"][point]
        reqs = tva.make_reqs(self.plan, point, block, **{"service": Jitter(), **self.kw(point, block)})
        base = sd.workload(x["client_base"])
        lens = []
        for (prompt, _), q in zip(sva.planned(self.plan, point, block), reqs):
            if base["mix"] == "mixed":
                lens.append(bp.LONG_INPUT if q["cls"] == "long" else base["tier"] - bp.N_GEN)
            else:
                lens.append(prompt if prompt is not None else 64 + (q["idx"] * 37) % 1857)
        n = len(reqs)
        res = {"completed": n, "failed": 0, "duration": 10.0, "request_throughput": n / 10.0,
               "output_throughput": 300.0, "total_token_throughput": 1e5, "input_lens": lens,
               "output_lens": [q["out"] for q in reqs], "ttfts": [q["ttft_ms"] / 1000 for q in reqs],
               "latencies": [q["latency_ms"] / 1000 for q in reqs],
               "start_times": [1000.0 + q["start_s"] for q in reqs]}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        prefix = a.get("--request-id-prefix", DEFAULT_PREFIX)

        def ids(entries):
            for i, e in enumerate(entries):
                e["client_request_id"] = f"{prefix}{i}"

        text = tva.log_text(self.plan, point, block, reqs, change=ids)
        self.buffered.append(stamped(text.splitlines()))
        if (point, block) in self.dies:
            self.gw.returncode = 1
        self.t += 10.0 + (self.slow(point, block) if self.slow else 0.0)
        return 0, False


def make_repo(tmp_path):
    """A repo root: Stage D's committed results (D5, for the report's pacing check) and every dataset as a small file
    named by its key (the fake bench reads none of them)."""
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "results" / "stage_d").symlink_to(ROOT / "results" / "stage_d")
    data = root / "data"
    data.mkdir()
    paths, sha = {}, {}
    for key in (*sv.DATASETS, sv.VAR_DATASET, sv.MIX_DATASET):
        p = data / f"{key}.jsonl"
        p.write_text(json.dumps({"dataset": key}) + "\n")
        paths[key], sha[key] = str(p), hashlib.sha256(p.read_bytes()).hexdigest()
    return root, paths, sha


def evidence(root, *, started="2026-11-01T09:00:00+00:00", finished="2026-11-01T09:40:00+00:00"):
    d = root.joinpath(*sv.RESULTS, MINI_ID)
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps({
        "mode": "mini", "session": "mini", "formal": False, "smoke": {"ok": True},
        "harness_sha256": sv.harness_sha256(), "gateway_path_sha256": sv.gateway_path_sha256(), "started_at": started,
        "finished_at": finished}))


def configs(root, sha, *, formal):
    """stage_v.toml / stage_d.toml / pins as the session ran them: every dataset pinned to the repo's files (the frozen
    stage_d.toml and pins pin var-2048 and mixed-8192 to them too); formal: frozen, with the mini's evidence."""
    c, d, p = copy.deepcopy(CFG), copy.deepcopy(SD_CFG), copy.deepcopy(PINS)
    c["stage_v"]["datasets"].update(sha)
    d["stage_d"]["var_dataset_sha256"] = {**d["stage_d"]["var_dataset_sha256"], "2048": sha[sv.VAR_DATASET]}
    d["stage_d"]["mix_dataset_sha256"] = {"8192": sha[sv.MIX_DATASET]}
    p["core_b"]["mix_dataset_sha256"] = {**p["core_b"]["mix_dataset_sha256"], "8192": sha[sv.MIX_DATASET]}
    if formal:
        c["freeze"].update(stage_v=True, mini=MINI_ID, gateway_path_sha256=sv.gateway_path_sha256())
        evidence(root)
    return c, d, p


def out_dir(root, name, mode):
    return root.joinpath(*sv.RESULTS, f"formal-t-{name.lower()}" if mode == "formal" else "smoke-v-t")


def session(tmp_path, name, *, mode="formal", world=None, cfg_over=None, plan_over=None, kw=None, repo=None, **k):
    root, paths, sha = repo or make_repo(tmp_path)
    c, d, p = configs(root, sha, formal=mode == "formal")
    k.setdefault("gpu_lock_held", True)
    k.setdefault("capture_env", td.Env())
    k.setdefault("base_env", td.BASE_ENV)
    k.setdefault("toolchain_check", lambda env: [])
    if cfg_over:
        cfg_over(c)
    plan = sv.build_plan(c, p, d, session_name=name, pins_sha256=PINS_SHA, stage_v_sha256=SV_SHA,
                         stage_d_sha256=SD_SHA, formal=mode == "formal" and c["freeze"]["stage_v"] is True)
    plan = json.loads(json.dumps(plan))                                      # as main reads stage_v_plan's file
    if plan_over:
        plan_over(plan)
    w = world(plan) if world else World(tmp_path, plan, kw=kw)
    out = k.pop("out", out_dir(root, name, mode))
    s = rv.run_session(plan, cfg=c, stage_d_cfg=d, pins=p, pins_sha256=PINS_SHA, stage_v_sha256=SV_SHA,
                       stage_d_sha256=SD_SHA, out_dir=out, session_id=out.name, mode=mode, popen=w.popen,
                       signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                       run_bench=w.run_bench, gpu_sample=w.gpu_sample, compute_apps=w.compute_apps,
                       shared_usage=w.shared_usage, lock_held_check=k.pop("lock_held_check", w.lock_check),
                       datasets=k.pop("datasets", paths), gateway_python="/venv/python",
                       port_free=k.pop("port_free", lambda port: True), sample_interval=None, now=w.now, sleep=w.sleep,
                       clock=k.pop("clock", w.clock), repo_root=root, **k)
    return out, s, w


def load(out):
    out = Path(out)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    reqs = {r["tag"]: [json.loads(x) for x in (out / r["derived"]).read_text().splitlines()]
            for r in rows if r.get("derived")}
    cap = {"plan": json.loads((out / "plan.json").read_text()),
           "session": json.loads((out / "session.json").read_text()), "rows": rows}
    return cap, reqs


def analysis(out):
    cap, reqs = load(out)
    return sva.analyze(cap, reqs), cap["rows"]


# ------------------------------------------------------------------ formal sessions --

@pytest.mark.parametrize("name", ["V-ADM", "V-CLS"])
def test_a_formal_session_runs_044s_protocol_and_replays_clean(tmp_path, name):
    out, s, w = session(tmp_path, name)
    a, rows = analysis(out)
    plan = json.loads((out / "plan.json").read_text())
    assert a["replay"] == {"ok": True, "problems": []} and a["sentinel"]["gate"]["ok"]
    assert all(r["valid"] for r in rows) and {r.get("outcome") for r in rows if r["phase"] == "V"} == {"ok"}
    assert {p: x["class"] for p, x in a["points"].items()} == dict.fromkeys(plan["points"], "PASS")
    assert all(x["holds"] for x in a["units"].values())
    v_boots = [b for b in s["boots"] if b["phase"] == "V"]
    assert len(v_boots) == 5 * len(plan["points"]) and len(s["boots"]) == len(v_boots) + 2
    assert s["formal"] and s["stage"] == "V" and s["session"] == name and s["cap_stop"] is None
    assert s["gateway_path_sha256"] == sv.gateway_path_sha256() == plan["gateway_path_sha256"]
    assert s["harness_sha256"] == sv.harness_sha256() and s["capserve_sha256"] == rd.capserve_sha256(ROOT)
    assert s["profile_sha256"] == {p: x["profile_sha256"] for p, x in plan["points"].items()}
    assert set(s["dataset_sha256"]) == set(plan["datasets"])
    elapsed = [b["elapsed_s"] for b in s["boots"]]
    assert elapsed == sorted(elapsed) and elapsed[0] >= 0 and all(isinstance(t, float) for t in elapsed)
    assert len(w.gateway_argvs) == len(v_boots) and len(w.server_argvs) == len(s["boots"])


def test_every_v_row_merges_the_gateways_fields_into_its_derived_file(tmp_path):
    out, s, w = session(tmp_path, "V-ADM")
    cap, reqs = load(out)
    for r in cap["rows"]:
        keys = set(sva.DERIVED_KEYS) if r["phase"] == "V" else set(rd.DERIVED_KEYS)
        assert all(set(q) == keys for q in reqs[r["tag"]]), r["tag"]
        assert s["derived_sha256"][r["tag"]] == hashlib.sha256((out / r["derived"]).read_bytes()).hexdigest()
    w90 = next(r for r in cap["rows"] if r["workload"] == "W-90")
    assert {q["units"] for q in reqs[w90["tag"]]} == {1, 4}
    assert {q["cls"] for q in reqs[w90["tag"]]} == {"short", "long"}
    assert not list(out.glob("derived/*-gateway.jsonl"))                     # one file per run: mask_smoke covers it
    access = w90["access_log"]
    assert access["problems"] == [] and access["joined"] == w90["N"] and access["log"].endswith(".gateway.log")


def test_the_client_is_the_plans_and_the_gateway_the_points(tmp_path):
    out, s, w = session(tmp_path, "V-CLS")
    plan = json.loads((out / "plan.json").read_text())
    rows = load(out)[0]["rows"]
    for r in rows:
        if r["phase"] != "V":
            continue
        want = sv.bench_argv(plan, PINS, r["workload"], block=r["block"], result_dir=str(out / "raw"),
                             result_filename=f"{r['tag']}.bench.json",
                             datasets={k: v for k, v in s["datasets"].items()}, vllm_bin=".venv/bin/vllm")
        assert r["argv"] == want and want[-2:] == ["--request-id-prefix", f"{r['tag']}-"]
        assert "http://127.0.0.1:8080" in want
        if r["workload"].startswith("L-"):
            assert want[want.index("--custom-output-len") + 1] == "-1"
    for argv, boot in zip(w.gateway_argvs, [b for b in s["boots"] if b["phase"] == "V"]):
        x = plan["points"][boot["workload"]]
        assert argv[:5] == ["/venv/python", "-m", "capserve", "serve", "--config"]
        assert argv[5] == str(out / "raw" / f"{boot['tag']}.gateway.toml") and argv[6] == "--profile"
        assert argv[7] == str(ROOT / x["profile"])
        assert ("--allow-provisional" in argv) is x["provisional"]               # only the provisional points
        assert tomllib.loads(Path(argv[5]).read_text()) == x["gateway"]
        assert hashlib.sha256(Path(argv[5]).read_bytes()).hexdigest() == x["gateway_toml_sha256"]
    verified = {b["workload"] for b in s["boots"]
                if b["phase"] == "V" and not plan["points"][b["workload"]]["provisional"]}
    assert verified == {"GW-U32K", "GW-M8K-T"}
    env = w.gateway_envs[0]
    assert not any(k.startswith("CAPSERVE_") for k in env) and env["HF_HUB_OFFLINE"] == "1"
    t_boots = [b for b in s["boots"] if b["phase"] == "V" and b["workload"] == "GW-M8K-T"]
    assert all(s["runs"][b["tag"]]["argv"][-2:] == ["--long-prefill-token-threshold", "512"] for b in t_boots)


def test_every_boot_records_the_readyz_tuple_it_checked(tmp_path):
    out, s, w = session(tmp_path, "V-ADM")
    plan = json.loads((out / "plan.json").read_text())
    for b in s["boots"]:
        rec = s["runs"][b["tag"]].get("gateway")
        if b["phase"] != "V":
            assert rec is None
            continue
        x = plan["points"][b["workload"]]
        profile_id = json.loads((ROOT / x["profile"]).read_text())["profile_id"]
        want = rv.readyz_want(plan, b["workload"], profile_id=profile_id, gateway_path_sha256=s["gateway_path_sha256"])
        assert rec["readyz"] == want and rec["stop_problems"] == []
        assert want["serving_status"] == "provisional" and want["queue"] == sv.QUEUE["slow"]
    w90 = rv.readyz_want(plan, "W-90", profile_id="provisional-stage-v", gateway_path_sha256="g" * 64)
    assert w90["active_class"] is None and w90["admission"] == sv.WEIGHTED_V1 and w90["prefill_pacing"] is None
    p = rv.readyz_want(plan, "P-U8K", profile_id="provisional-stage-v", gateway_path_sha256="g" * 64)
    assert p["prefill_pacing"] == {"burst_tokens": 32256, "rate_tokens_per_s": 5800} and p["admission"] is None


def test_the_gateway_stops_before_the_server_and_the_log_is_read_after_it(tmp_path):
    out, s, w = session(tmp_path, "V-CLS")
    ev = [e for e in w.events if e.startswith(("gateway", "server:signal", "bench:V"))]
    for i, e in enumerate(ev):
        if e == "gateway:start":
            assert ev[i + 1].startswith("bench:V") and ev[i + 2] == "gateway:log-drained"
            assert ev[i + 3] == f"gateway:signal:{int(signal.SIGTERM)}" and ev[i + 4].startswith("server:signal")
    a, rows = analysis(out)
    assert a["replay"]["ok"] and all(r["valid"] for r in rows)                # every main line joined


def test_a_v_run_without_its_request_id_prefix_cannot_join(tmp_path, monkeypatch):
    real = sv.bench_argv
    monkeypatch.setattr(sv, "bench_argv", lambda *a, **k: real(*a, **k)[:-2])
    out, s, w = session(tmp_path, "V-ADM")
    rows = load(out)[0]["rows"]
    v = [r for r in rows if r["phase"] == "V"]
    assert v and all(not r["valid"] and any("client_request_id" in x for x in r["reasons"]) for r in v)
    assert all(r["outcome"] == "invalid" for r in v)


# ------------------------------------------------------------------ the boot's gateway checks --

def test_a_readyz_that_differs_from_the_points_tuple_fails_the_boot(tmp_path):
    def edit(body, label):
        if label.startswith("V-P-U8K"):
            body["prefill_pacing"] = {"burst_tokens": 32256, "rate_tokens_per_s": 5810}

    out, s, w = session(tmp_path, "V-ADM", world=lambda plan: World(tmp_path, plan, readyz=edit))
    bad = [b for b in s["boots"] if b["workload"] == "P-U8K"]
    assert len(bad) == 1 and bad[0]["reached_first_bench"] is False
    assert "readyz" in bad[0]["problems"][0] and "prefill_pacing" in bad[0]["problems"][0]
    assert s["gateway_faults"] and s["gateway_faults"][0]["boot"] == bad[0]["tag"]
    a, rows = analysis(out)
    assert a["replay"]["ok"] and a["points"]["P-U8K"]["class"] == "NT"
    assert not [e for e in w.events if e.startswith("bench:V-P-U8K")]


@pytest.mark.parametrize("field, value", [
    ("profile", {"id": "verified-formal-20261017-c1b", "status": "verified", "provisional": False}),
    ("serving_status", "verified"), ("active_class", "v-uniform-2k-o512"),
    ("limits", {"max_concurrency": 33, "max_queue": 32, "max_prompt_tokens": 1920, "max_output_tokens": 128}),
    ("queue", {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0}), ("admission", sv.WEIGHTED_V1),
    ("prefill_pacing", None), ("gateway_path_sha256", "0" * 64)])
def test_each_part_of_the_readyz_tuple_is_checked(tmp_path, field, value):
    def edit(body, label):
        if label.startswith("V-P-U2K-r"):
            body[field] = value

    out, s, w = session(tmp_path, "V-ADM", world=lambda plan: World(tmp_path, plan, readyz=edit))
    bad = [b for b in s["boots"] if b["workload"] == "P-U2K"]
    assert len(bad) == 1 and bad[0]["reached_first_bench"] is False and field in bad[0]["problems"][0]


def test_a_provisional_point_without_the_flag_never_serves(tmp_path, monkeypatch):
    real = rv.gateway_argv
    monkeypatch.setattr(rv, "gateway_argv", lambda *a, **k: real(*a, **{**k, "provisional": False}))
    out, s, w = session(tmp_path, "V-CLS")
    plan = json.loads((out / "plan.json").read_text())
    for b in s["boots"]:
        if b["phase"] == "V":
            assert b["reached_first_bench"] is (not plan["points"][b["workload"]]["provisional"]), b["tag"]
    assert any("not ready" in b["problems"][0] for b in s["boots"] if b["problems"])


def test_a_busy_gateway_port_aborts_the_session_before_vllm_starts(tmp_path):
    with pytest.raises(rv.RunError, match="gateway_port_busy"):
        session(tmp_path, "V-ADM", port_free=lambda port: port != 8080)
    out = out_dir(tmp_path / "repo", "V-ADM", "formal")
    s = json.loads((out / "session.json").read_text())
    assert s["gateway_port_busy"]["port"] == 8080 and "gateway_port_busy" in s["aborted"]
    assert [b["phase"] for b in s["boots"]] == ["sentinel", "V"] and not s["runs"].get(s["boots"][1]["tag"])


def test_a_gateway_that_dies_during_a_bench_invalidates_the_run(tmp_path):
    out, s, w = session(tmp_path, "V-ADM", world=lambda plan: World(tmp_path, plan, dies={("W-50", 1)}))
    a, rows = analysis(out)
    bad = [r for r in rows if r["workload"] == "W-50"]
    assert len(bad) == 1 and not bad[0]["valid"] and any("gateway exited" in x for x in bad[0]["reasons"])
    assert a["replay"]["ok"] and a["points"]["W-50"]["class"] == "NT" and a["points"]["W-90"]["class"] == "NT"
    assert not [r for r in rows if r["workload"] in ("W-90", "W-50") and r["block"] > 1]   # the unit W1 is dead
    assert any(f.get("run") == bad[0]["tag"] for f in s["gateway_faults"])


def test_the_runners_verdict_folds_in_j_and_e(tmp_path):
    """044 (j) and E make a run invalid: the runner records them in the row, so the replay recomputes the same."""
    def kw(point, block):
        base = tva.ok_kw(point, block)
        return {**base, "prefill": {}} if (point, block) == ("P-U2K", 1) else base

    out, s, w = session(tmp_path, "V-ADM", kw=kw)
    a, rows = analysis(out)
    bad = next(r for r in rows if r["workload"] == "P-U2K")
    assert not bad["valid"] and bad["outcome"] == "invalid" and any(x.startswith("E:") for x in bad["reasons"])
    assert [r["block"] for r in rows if r["workload"] == "P-U2K"] == [1] and a["replay"]["ok"]


# ------------------------------------------------------------------ hours (044 停止規則與時數上限) --

def test_the_hour_cap_stops_new_v_boots_and_the_end_sentinel_still_runs(tmp_path):
    def slow(point, block):
        return 6.5 * 3600 if block == 2 else 0.0

    out, s, w = session(tmp_path, "V-CLS", world=lambda plan: World(tmp_path, plan, slow=slow))
    stop = s["cap_stop"]
    assert stop is not None and stop["forced"] is False and stop["block"] == 2
    assert s["boots"][-1]["phase"] == "sentinel" and s["boots"][-1]["position"] == "end"
    assert stop["elapsed_s"] <= s["boots"][-1]["elapsed_s"]
    a, rows = analysis(out)
    assert a["replay"]["ok"] and a["schedule"]["not_run_cap"][0] == [stop["block"], stop["point"]]
    plan = json.loads((out / "plan.json").read_text())
    assert not sv.cap_allows(stop["elapsed_s"], plan["points"][stop["point"]]["est_s"], 7.0)


def test_every_session_counts_toward_the_12_hour_total(tmp_path):
    root, paths, sha = make_repo(tmp_path)
    old = root.joinpath(*sv.RESULTS, "smoke-v-old")
    old.mkdir(parents=True)
    (old / "session.json").write_text(json.dumps({"started_at": "2026-10-30T00:00:00+00:00",
                                                  "finished_at": "2026-10-30T05:00:01+00:00"}))
    with pytest.raises(rv.RunError, match="12"):                             # 5 h + 1 s used + V-CLS's 7 h > 12 h
        session(tmp_path, "V-CLS", repo=(root, paths, sha))
    assert not out_dir(root, "V-CLS", "formal").exists()
    (old / "session.json").write_text(json.dumps({"started_at": "2026-10-30T00:00:00+00:00",
                                                  "finished_at": "2026-10-30T04:20:00+00:00"}))
    out, s, w = session(tmp_path, "V-CLS", repo=(root, paths, sha))        # 4 h 20 min + 40 min (the mini) + 7 h
    assert s["hours_used_h"] == pytest.approx(5.0)
    (old / "session.json").write_text(json.dumps({"started_at": "2026-10-30T00:00:00+00:00"}))
    with pytest.raises(rv.RunError, match="smoke-v-old"):
        session(tmp_path / "again", "V-ADM", repo=(root, paths, sha))


def test_formal_sessions_are_at_least_2_hours_apart(tmp_path):
    root, paths, sha = make_repo(tmp_path)
    prev = root.joinpath(*sv.RESULTS, "formal-t-prev")
    prev.mkdir(parents=True)

    def write(finished):
        (prev / "session.json").write_text(json.dumps({"session": "V-ADM", "formal": True,
                                                       "started_at": "2026-11-02T05:00:00+00:00",
                                                       "finished_at": finished}))

    write("2026-11-02T07:00:01+00:00")                                       # T0 is 09:00: 1 s short of 2 h
    with pytest.raises(rv.RunError, match="2 h"):
        session(tmp_path, "V-CLS", repo=(root, paths, sha))
    write("2026-11-02T07:00:00+00:00")
    out, s, w = session(tmp_path, "V-CLS", repo=(root, paths, sha))
    assert s["previous_formal"] == {"session_id": "formal-t-prev", "finished_at": "2026-11-02T07:00:00+00:00"}
    shutil.rmtree(out)                                                       # (it started at the same fake 09:00)
    late = root.joinpath(*sv.RESULTS, "formal-t-zz-late")                   # the one that finished last counts
    late.mkdir()
    (late / "session.json").write_text(json.dumps({"session": "V-ADM", "formal": True,
                                                   "started_at": "2026-11-02T06:00:00+00:00",
                                                   "finished_at": "2026-11-02T07:30:00+00:00"}))
    with pytest.raises(rv.RunError, match="formal-t-zz-late finished at 2026-11-02T07:30"):
        session(tmp_path / "late", "V-ADM", repo=(root, paths, sha))
    (late / "session.json").unlink()
    write(None)
    with pytest.raises(rv.RunError, match="formal-t-prev"):
        session(tmp_path / "again", "V-ADM", repo=(root, paths, sha))
    (prev / "session.json").write_text(json.dumps({"started_at": "2026-11-02T05:00:00",          # no time zone: its
                                                   "finished_at": "2026-11-02T06:00:00"}))      # hours count, its gap
    with pytest.raises(rv.RunError, match="formal-t-prev: cannot tell when it finished"):        # cannot be told
        session(tmp_path / "naive", "V-ADM", repo=(root, paths, sha))
    mini_root = make_repo(tmp_path / "m")                                     # the mini needs no gap
    gap = mini_root[0].joinpath(*sv.RESULTS, "formal-t-prev")
    gap.mkdir(parents=True)
    (gap / "session.json").write_text(json.dumps({"session": "V-ADM", "formal": True,
                                                  "started_at": "2026-11-02T08:00:00+00:00",
                                                  "finished_at": "2026-11-02T08:30:00+00:00"}))
    out, s, w = session(tmp_path / "m", "mini", mode="mini", repo=mini_root)
    assert s["mode"] == "mini"


# ------------------------------------------------------------------ refusals --

def test_refusals_before_anything_starts(tmp_path):
    with pytest.raises(rv.RunError, match="results/stage_v"):
        session(tmp_path / "a", "V-ADM", out=tmp_path / "elsewhere")
    with pytest.raises(rv.RunError, match="formal-"):
        session(tmp_path / "b", "V-ADM", out=tmp_path / "b" / "repo" / "results" / "stage_v" / "v-adm")
    with pytest.raises(rv.RunError, match="smoke-v-"):
        session(tmp_path / "c", "mini", mode="mini", out=tmp_path / "c" / "repo" / "results" / "stage_v" / "mini-1")
    with pytest.raises(rv.RunError, match="formal Stage V sessions require freeze.stage_v"):
        session(tmp_path / "d", "V-ADM", cfg_over=lambda c: c["freeze"].update(stage_v=False))
    with pytest.raises(rv.RunError, match="plan .* differs"):
        session(tmp_path / "e", "V-ADM", plan_over=lambda p: p["points"]["W-90"].update(C=31))
    with pytest.raises(rv.RunError, match="mini"):
        session(tmp_path / "f", "V-ADM", mode="mini")
    with pytest.raises(rv.RunError, match="must hold the GPU lock"):
        session(tmp_path / "g", "V-ADM", gpu_lock_held=False)
    with pytest.raises(rv.RunError, match="does not hold"):
        session(tmp_path / "h", "V-ADM", lock_held_check=lambda: False)
    with pytest.raises(rv.RunError, match="toolchain"):
        session(tmp_path / "i", "V-ADM", toolchain_check=lambda env: ["nvcc missing"])
    root = make_repo(tmp_path / "j")
    out_dir(root[0], "V-ADM", "formal").mkdir(parents=True)
    with pytest.raises(rv.RunError, match="exists"):
        session(tmp_path / "j", "V-ADM", repo=root)


def test_sha_mismatches_refuse_the_session(tmp_path):
    def moved(c):                                                            # the request path moved on since the freeze
        c["freeze"].update(gateway_path_sha256="0" * 64)
        ev = tmp_path / "a" / "repo" / "results" / "stage_v" / MINI_ID / "session.json"
        ev.write_text(json.dumps({**json.loads(ev.read_text()), "gateway_path_sha256": "0" * 64}))

    with pytest.raises(rv.RunError, match="^request path: gateway_path_sha256"):
        session(tmp_path / "a", "V-ADM", cfg_over=moved)
    with pytest.raises(rv.RunError, match="provisional-stage-v.json sha256"):
        session(tmp_path / "b", "V-ADM", cfg_over=lambda c: c["stage_v"]["points"]["P-U2K"].update(
            profile_sha256="0" * 64))
    root, paths, sha = make_repo(tmp_path / "c")
    Path(paths["v4-90-r3"]).write_text("changed\n")
    with pytest.raises(rv.RunError, match="v4-90-r3"):
        session(tmp_path / "c", "V-ADM", repo=(root, paths, sha))
    root, paths, sha = make_repo(tmp_path / "d")
    with pytest.raises(rv.RunError, match="v1-mini-8192"):                  # the mini's datasets are pinned too
        session(tmp_path / "d", "mini", mode="mini", repo=(root, paths, sha),
                cfg_over=lambda c: c["stage_v"]["datasets"].update({"v1-mini-8192": ""}))
    with pytest.raises(rv.RunError, match="needs --dataset"):
        session(tmp_path / "e", "V-ADM", datasets={})
    root, paths, sha = make_repo(tmp_path / "g")
    with pytest.raises(rv.RunError, match="needs --dataset var-2048"):
        session(tmp_path / "g", "V-ADM", repo=(root, paths, sha), datasets={**paths, "var-2048": str(root / "nope")})
    with pytest.raises(rv.RunError, match="freeze evidence"):
        session(tmp_path / "f", "V-ADM", cfg_over=lambda c: c["freeze"].update(mini="smoke-v-other"))


def test_a_formal_run_needs_a_deployed_clean_preflight(tmp_path):
    class Dirty(td.Env):
        def __call__(self, out_path, gpu_lock_held):
            rec = super().__call__(out_path, gpu_lock_held)
            rec["repo"]["source"] = "WORKTREE"
            Path(out_path).write_text(json.dumps(rec))
            return rec

    with pytest.raises(rv.RunError, match="DEPLOYED_COMMIT"):
        session(tmp_path, "V-ADM", capture_env=Dirty())


def test_the_runner_never_imports_capserve_and_imports_only_research_modules():
    tree = ast.parse((ROOT / "benchmarks" / "run_stage_v.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(n == "capserve" or n.startswith("capserve.") for n in names)
    assert names <= {"__future__", "argparse", "contextlib", "copy", "datetime", "hashlib", "json", "os", "re",
                     "signal", "sys", "time", "fractions", "pathlib", "typing", "tomllib", "capacity_analysis",
                     "capacity_plan", "core_b_plan", "coreb_smoke", "gpu_idle", "run_adv", "run_capacity",
                     "run_core_a", "run_stage_d", "stage_d_plan", "stage_v_analysis", "stage_v_plan"}


def test_the_stage_v_harness_overrides_what_044_names():
    for name in ("boot", "gateway_up", "_row", "client_argv", "bench", "run_point", "decide"):
        assert name in vars(rv.StageVHarness), name
    assert issubclass(rv.StageVHarness, rd.StageDHarness)


# ------------------------------------------------------------------ mini --

def test_the_mini_smoke_passes_and_discards_latencies(tmp_path):
    out, s, w = session(tmp_path, "mini", mode="mini")
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["ok"] is True, smoke
    assert s["mode"] == "mini" and s["formal"] is False and s["session"] == "mini" and s["stage"] == "V-mini"
    assert len(s["harness_sha256"]) == 64 and s["gateway_path_sha256"] == sv.gateway_path_sha256()
    assert smoke["timeouts"] == [["GW-U8K-O512", 2]]
    assert s["cap_stop"]["forced"] is True and s["cap_stop"]["block"] == 2
    plan = json.loads((out / "plan.json").read_text())
    assert plan["slo"] == rc.MINI_SLO and plan["points"]["L-U8K"]["datasets"]["1"] == "v1-mini-8192"
    assert not list((out / "raw").glob("*.bench.json"))
    for p in (out / "derived").glob("*-requests.jsonl"):
        assert all(set(json.loads(x)) == {"idx", "cls", "out"} for x in p.read_text().splitlines())
    l8 = [argv for argv in w.bench_argvs if "--custom-output-len" in argv and "v1-mini-8192" in " ".join(argv)]
    assert l8 and all(argv[argv.index("--num-prompts") + 1] == "20" for argv in l8)
    assert bp.num_prompts(32) == 320                                          # the N patch is undone
    checks = smoke["checks"]
    assert checks["prefill_waited"] and checks["queued"] and checks["nontrivial_computable"]
    assert checks["forced_cap_stop"] and checks["gateway_ok"] and checks["recorded_hashes"]
    freeze = {"freeze": {"stage_v": True, "mini": out.name, "gateway_path_sha256": sv.gateway_path_sha256()}}
    assert sv.freeze_evidence_problems(freeze, tmp_path / "repo") == []      # this mini is the freeze's evidence


def test_a_mini_whose_forced_timeout_never_hits_fails(tmp_path):
    def plan_over(p):
        p["points"]["GW-U8K-O512"]["forced_timeout_blocks"] = []

    with pytest.raises(rv.RunError, match="plan .* differs"):
        session(tmp_path / "a", "mini", mode="mini", plan_over=plan_over)
    out, s, w = session(tmp_path / "b", "mini", mode="mini",
                        world=lambda plan: World(tmp_path / "b", plan, dies={("L-U8K", 1)}))
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["ok"] is False and smoke["timeouts"] == [] and smoke["invalid_runs"]


def test_build_mini_plan_changes_only_the_slo_and_the_stage():
    plan = tva.MINI
    m = rv.build_mini_plan(plan)
    assert m["slo"] == rc.MINI_SLO and m["stage"] == "V-mini" and plan["slo"] == CFG["slo"]
    assert {k: v for k, v in m.items() if k not in ("slo", "stage")} == {k: v for k, v in plan.items()
                                                                          if k not in ("slo", "stage")}


# ------------------------------------------------------------------ helpers --

def test_access_log_text_drops_the_readers_stamps_only():
    raw = stamped(['{"a": 1}', "INFO:     Started", "   {not json"]) + "no stamp {x}\n"
    assert rv.access_log_text(raw) == '{"a": 1}\nINFO:     Started\n   {not json\nno stamp {x}\n'


def test_output_problems(tmp_path):
    root = tmp_path / "repo"
    base = root.joinpath(*sv.RESULTS)
    assert rv.output_problems(base / "formal-x", mode="formal", repo_root=root) == []
    assert rv.output_problems(base / "smoke-v-x", mode="mini", repo_root=root) == []
    assert rv.output_problems(base / "smoke-v-x", mode="formal", repo_root=root)
    assert rv.output_problems(base / "formal-x", mode="mini", repo_root=root)
    assert rv.output_problems(base / "formal-x" / "runs" / "a", mode="formal", repo_root=root)
    assert rv.output_problems(base / "formal-x" / "y", mode="formal", repo_root=root)
    assert rv.output_problems(base, mode="formal", repo_root=root)
    assert rv.output_problems(root / "results" / "stage_d" / "formal-x", mode="formal", repo_root=root)


# ------------------------------------------------------------------ more gateway faults --

class Fading(td.Proc):
    """A gateway process whose poll() reports exit code 1 from its n-th call on."""

    def __init__(self, pid, n):
        super().__init__(pid)
        self.calls, self.n = 0, n

    def poll(self):
        self.calls += 1
        if self.calls >= self.n and self.returncode is None:
            self.returncode = 1
        return self.returncode


def fading_world(tmp_path, point, n):
    class FadingWorld(World):
        def popen(self, argv, env, log_path):
            proc = super().popen(argv, env, log_path)
            if proc is self.gw and self.label.startswith(f"V-{point}-r1"):
                self.gw = Fading(88, n)
                return self.gw
            return proc

    return lambda plan: FadingWorld(tmp_path, plan)


def test_a_gateway_gone_right_after_ready_or_before_the_bench_runs_nothing(tmp_path):
    # poll calls: the readiness loop (1), right after /readyz (2), run_point before the bench (3)
    out, s, w = session(tmp_path / "a", "V-ADM", world=fading_world(tmp_path / "a", "F-U2K", 2))
    boot = next(b for b in s["boots"] if b["workload"] == "F-U2K")
    assert boot["reached_first_bench"] is False and "right after it was ready" in boot["problems"][0]
    out, s, w = session(tmp_path / "b", "V-ADM", world=fading_world(tmp_path / "b", "F-U2K", 3))
    rows = load(out)[0]["rows"]
    row = next(r for r in rows if r["workload"] == "F-U2K")
    assert "before this bench" in row["reasons"][0] and "argv" not in row and row["outcome"] == "invalid"
    assert not [e for e in w.events if e.startswith("bench:V-F-U2K-c")]


def test_a_gateway_log_reader_that_never_joins_invalidates_the_run(tmp_path):
    class Stuck(World):
        def popen(self, argv, env, log_path):
            proc = super().popen(argv, env, log_path)
            if proc is self.gw and self.label.startswith("V-W-90-r1"):
                proc.log_reader = type("R", (), {"error": None, "join": lambda self, t: False})()
            return proc

    out, s, w = session(tmp_path, "V-ADM", world=lambda plan: Stuck(tmp_path, plan))
    row = next(r for r in load(out)[0]["rows"] if r["workload"] == "W-90")
    assert not row["valid"] and any("log reader did not join" in x for x in row["reasons"])
    assert any(f["run"] == row["tag"] for f in s["gateway_faults"])


def test_a_run_that_breaks_j_is_invalid_and_a_gateway_fault(tmp_path):
    def kw(point, block):
        base = tva.ok_kw(point, block)
        return {**base, "override": {100: {"gw_released_s": 10 ** 4}}} if (point, block) == ("P-U8K", 1) else base

    out, s, w = session(tmp_path, "V-ADM", kw=kw)
    a, rows = analysis(out)
    row = next(r for r in rows if r["workload"] == "P-U8K")
    assert not row["valid"] and any("(044 (j))" in x for x in row["reasons"]) and a["replay"]["ok"]
    assert [f["reasons"] for f in s["gateway_faults"] if f["run"] == row["tag"]] == [
        [x for x in row["reasons"] if "(044 (j))" in x]]


def test_a_gateway_port_taken_while_the_server_starts_aborts_the_session(tmp_path):
    calls = []

    def port_free(port):
        calls.append(port)
        return len(calls) != 2                                          # free before the boot, taken at gateway_up

    with pytest.raises(rv.RunError, match="gateway_port_busy in"):
        session(tmp_path, "V-ADM", port_free=port_free)
    s = json.loads((out_dir(tmp_path / "repo", "V-ADM", "formal") / "session.json").read_text())
    assert s["gateway_port_busy"]["boot"] == s["boots"][1]["tag"] and calls == [8080, 8080]


def test_formal_environment_refusals(tmp_path, monkeypatch):
    with pytest.raises(rv.RunError, match="inherited engine variables"):
        session(tmp_path / "a", "V-ADM", base_env={**td.BASE_ENV, "VLLM_LOGGING_LEVEL": "DEBUG"})
    with pytest.raises(rv.RunError, match="OTEL"):
        session(tmp_path / "b", "V-ADM", base_env={**td.BASE_ENV, "OTEL_SDK_DISABLED": "true"})
    real = configs

    def unpinned(root, sha, *, formal):                                      # pins [stage_b] without its pin-memory A/B
        c, d, p = real(root, sha, formal=formal)
        p["stage_b"]["wsl2_pin_memory"] = "2"
        return c, d, p

    monkeypatch.setattr(sys.modules[__name__], "configs", unpinned)
    with pytest.raises(rv.RunError, match="wsl2_pin_memory"):
        session(tmp_path / "c", "V-ADM")
    monkeypatch.undo()

    class Vllm(td.Env):
        def __call__(self, out_path, gpu_lock_held):
            rec = super().__call__(out_path, gpu_lock_held)
            rec["runtime"]["vllm"]["vllm"] = "0.31.0"
            return rec

    with pytest.raises(rv.RunError, match="0.31.0"):
        session(tmp_path / "d", "V-ADM", capture_env=Vllm())


@pytest.mark.parametrize("edit, needle", [
    (lambda res: res.update(completed=res["completed"] - 1), "completed="),
    (lambda res: res["input_lens"].__setitem__(3, res["input_lens"][3] - 1), "input_lens"),
    (lambda res: res["output_lens"].__setitem__(3, res["output_lens"][3] + 1), "output_lens are not"),
])
def test_an_l_or_w_runs_bench_result_is_its_datasets_row_by_row(tmp_path, edit, needle):
    class Off(World):
        def run_bench(self, argv, log_path, env, timeout_s):
            out = super().run_bench(argv, log_path, env, timeout_s)
            a = dict(zip(argv, argv[1:]))
            if a["--result-filename"].startswith("V-L-U8K-c8-r1"):
                path = Path(a["--result-dir"], a["--result-filename"])
                res = json.loads(path.read_text())
                edit(res)
                path.write_text(json.dumps(res))
            return out

    out, s, w = session(tmp_path, "V-CLS", world=lambda plan: Off(tmp_path, plan))
    row = next(r for r in load(out)[0]["rows"] if r["workload"] == "L-U8K")
    assert not row["valid"] and any(needle in x for x in row["reasons"]), row["reasons"]


def test_an_invalid_start_sentinel_stops_the_session(tmp_path):
    class Bad(World):
        def run_bench(self, argv, log_path, env, timeout_s):
            out = super().run_bench(argv, log_path, env, timeout_s)
            a = dict(zip(argv, argv[1:]))
            if a["--result-filename"].startswith("sentinel-start"):
                path = Path(a["--result-dir"], a["--result-filename"])
                res = json.loads(path.read_text())
                res["ttfts"][3] = None                                    # the derived row has no number (030 8.6 (e))
                path.write_text(json.dumps(res))
            return out

    with pytest.raises(rv.RunError, match="start sentinel is invalid"):
        session(tmp_path, "V-ADM", world=lambda plan: Bad(tmp_path, plan))


def test_a_failing_run_kills_its_unit(tmp_path):
    def kw(point, block):
        base = tva.ok_kw(point, block)
        return {**base, "ttft": 9000.0} if (point, block) == ("GW-U2K-O512", 1) else base

    out, s, w = session(tmp_path, "V-CLS", kw=kw)
    a, rows = analysis(out)
    fail = next(r for r in rows if r["workload"] == "GW-U2K-O512")
    assert fail["valid"] and fail["outcome"] == "fail" and a["replay"]["ok"]
    order = json.loads((out / "plan.json").read_text())["block_orders"]["1"]
    l_first = order.index("L-U2K") < order.index("GW-U2K-O512")
    assert [r["block"] for r in rows if r["workload"] == "L-U2K"] == ([1] if l_first else [])
    assert a["points"]["GW-U2K-O512"]["class"] == "FAIL" and a["units"]["V1-2K"]["holds"] is False


def test_the_minis_forced_stop_comes_before_its_last_still_planned_v_boot(tmp_path):
    out, s, w = session(tmp_path, "mini", mode="mini",
                        world=lambda plan: World(tmp_path, plan, dies={("GW-M8K-T", 1)}))
    stop = s["cap_stop"]
    assert (stop["block"], stop["point"], stop["forced"]) == (2, "GW-U8K-O512", True)
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["checks"]["replay_clean"] and not smoke["ok"]               # its forced timeout never ran


def test_each_mini_check_can_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "mask_smoke", lambda out: [])                    # keep the latencies for this test
    out, s, w = session(tmp_path, "mini", mode="mini")
    cap, reqs = load(out)
    plan, session_rec, rows = cap["plan"], cap["session"], cap["rows"]
    assert rv.mini_checks(plan, session_rec, rows, reqs)["ok"] is True

    def tag_of(point, block=1):
        return next(r["tag"] for r in rows if r["workload"] == point and r["block"] == block)

    def first_valid(c):
        return next(r for r in c["rows"] if r["phase"] == "V" and r["valid"])

    def gw(c):
        b = next(b for b in c["session"]["boots"] if b["phase"] == "V")
        return c["session"]["runs"][b["tag"]]["gateway"]

    def empty_t_w(c):
        x = plan["points"]["P-U2K"]
        by = {q["idx"]: q for q in c["reqs"][tag_of("P-U2K")]}
        by[x["N"] - x["C"]]["gw_arrival_s"] = by[x["C"]]["gw_arrival_s"]

    cases = {
        "replay_clean": lambda c: first_valid(c).update(outcome="fail"),
        "other_runs_valid": lambda c: first_valid(c).update(valid=False),
        "forced_timeouts_hit": lambda c: next(r for r in c["rows"] if r.get("timeout")).update(timeout=False),
        "derived_schema": lambda c: c["reqs"][first_valid(c)["tag"]][0].pop("start_s"),
        "boots_passed_checks": lambda c: c["session"]["boots"][3].update(problems=["x"]),
        "gateway_ok": lambda c: gw(c).update(stop_problems=["x"]),
        "forced_cap_stop": lambda c: c["session"]["cap_stop"].update(forced=False),
        "prefill_waited": lambda c: [q.update(gw_prefill_wait_ms=0.0) for t, qs in c["reqs"].items() if "P-U2K" in t
                                     for q in qs],
        "queued": lambda c: [q.update(gw_queue_wait_ms=0.0) for t, qs in c["reqs"].items() if "W-mini" in t
                             for q in qs],
        "nontrivial_computable": empty_t_w,
        "recorded_hashes": lambda c: c["session"].update(gateway_path_sha256=None),
        "clean_teardowns": lambda c: c["session"].update(teardown_problems=[{"boot": "b", "reasons": ["x"]}]),
    }
    assert set(cases) == set(rv.mini_checks(plan, session_rec, rows, reqs)["checks"])
    for name, change in cases.items():
        c = copy.deepcopy({"session": session_rec, "rows": rows, "reqs": reqs})
        change(c)
        got = rv.mini_checks(plan, c["session"], c["rows"], c["reqs"])
        assert got["checks"][name] is False and got["ok"] is False, name


def test_an_interrupt_while_the_gateway_starts_still_stops_it(tmp_path):
    class Hangup(World):
        def http_get(self, url):
            if ":8080/readyz" in url:
                raise SystemExit(129)
            return super().http_get(url)

    world = []

    def make(plan):
        world.append(Hangup(tmp_path, plan))
        return world[0]

    with pytest.raises(SystemExit):
        session(tmp_path, "V-ADM", world=make)
    ev = [e for e in world[0].events if e.startswith(("gateway", "server:signal"))]
    i = ev.index("gateway:start")
    assert ev[i + 1] == f"gateway:signal:{int(signal.SIGTERM)}" and ev[i + 2].startswith("server:signal")
