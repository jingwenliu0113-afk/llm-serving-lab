"""Tests for benchmarks/run_stage_d.py (decision 039): whole Stage D sessions against a fake WSL (server, /metrics,
gateway, clock, env capture and `vllm bench serve`; no GPU), analysed with stage_d_analysis. Real execution happens in
WSL under the GPU lock: the three mini smokes before the freeze, the formal sessions after it."""

import copy
import hashlib
import json
import signal
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import gpu_idle as gi
import run_stage_d as rd
import stage_d_analysis as sda
import stage_d_plan as sd

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_d.toml").read_text())
C1B = ROOT / "results" / "capacity" / "formal-20261017-c1b"
PINS_SHA, SD_SHA = "ab" * 32, "cd" * 32
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "CAPSERVE_API_KEY": "never-passed-on"}
COMMIT = "97a948c911940249b3d04bff38727a1fcb8b1d64"


def truth(w, x, phase):
    """(ttft ms, tpot ms, long ttft ms) by workload and point: passes up to a limit, TPOT 120 ms above it."""
    limit = {"U32K": 2, "U8K-T": 5, "M8K-T": 3, "U2K-O512": 31, "U8K-O512": 9, "U2K-P": 5, "U8K-P": 6}.get(w, 10 ** 6)
    return (900.0, 50.0 if x <= limit else 120.0, 5000.0)


class Proc:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None
        self.log_reader = type("R", (), {"error": None, "join": lambda self, t: True})()

    def poll(self):
        return self.returncode


class World:
    """Fake WSL. bad_log[label] drops a key from the server's non-default args ("threshold": T's); readyz overrides
    the gateway's /readyz body; spill_at: shared-usage points that read +200 MiB."""

    def __init__(self, tmp_path, *, bad_log=None, readyz=None, spill_at=()):
        self.tmp, self.bad_log, self.readyz, self.spill_at = Path(tmp_path), bad_log or {}, readyz, set(spill_at)
        self.t, self.events, self.server_argvs, self.bench_argvs, self.gateway_argvs = 0.0, [], [], [], []
        self.gateway_envs, self.proc, self.gw, self.point, self.label = [], None, None, "pre", None

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def lock_check(self):
        return True

    def gpu_sample(self):
        return {"mem_used_mib": 567, "util_pct": 0, "pstate": "P8"}

    def compute_apps(self):
        return []

    def shared_usage(self):
        return {"luid": "luid_0", "dedicated_mib": 1000.0,
                "shared_mib": 64.4 + (0.0 if self.point == "pre" else 80.0) + (200.0 if self.point in self.spill_at
                                                                                 else 0.0)}

    def popen(self, argv, env, log_path):
        log_path = Path(log_path)
        if "capserve" in argv:
            self.events.append("gateway:start")
            self.gateway_argvs.append(argv)
            self.gateway_envs.append(env)
            self.gw = Proc(88)
            return self.gw
        label = log_path.name.split("-", 1)[1].removesuffix(".server.log")
        self.label, self.point = label, f"ready:{label}"
        self.events.append(f"server:{label}")
        self.server_argvs.append(argv)
        a = dict(zip(argv, argv[1:]))
        nd = {"model_tag": argv[2], "max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
              "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"])}
        if "--long-prefill-token-threshold" in a and self.bad_log.get(label) != "threshold":
            nd["long_prefill_token_threshold"] = int(a["--long-prefill-token-threshold"])
        log_path.write_text(f"1 (APIServer pid=1) INFO non-default args: {nd!r}\n"
                            "2 (EngineCore pid=2) INFO [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.\n")
        self.proc = Proc(77)
        return self.proc

    def signal_group(self, proc, sig):
        if proc is self.gw:
            self.events.append(f"gateway:signal:{int(sig)}")
        else:
            self.events.append(f"server:signal:{int(sig)}")
            self.point = "pre"
        proc.returncode = 0

    def group_alive(self, proc):
        return proc.returncode is None

    def http_get(self, url):
        if ":8080/readyz" in url:
            body = self.readyz or {"profile": {"id": "verified-formal-20261017-c1b", "status": "verified"},
                                   "active_class": self.gateway_class(), "limits": {"max_concurrency": self.limit()}}
            return (200, json.dumps(body)) if self.gw and self.gw.returncode is None else (0, "refused")
        alive = self.proc is not None and self.proc.returncode is None
        if url.endswith("/health"):
            return (200, "") if alive else (0, "refused")
        lab = 'engine="0",model_name="m"'
        return 200, (f'vllm:cache_config_info{{block_size="528",engine="0",num_gpu_blocks="248"}} 1.0\n'
                     f"vllm:num_requests_running{{{lab}}} 3.0\nvllm:num_requests_waiting{{{lab}}} 0.0\n")

    def gateway_class(self):
        cfg = json.loads(json.dumps(tomllib.loads(Path(self.gateway_argvs[-1][5]).read_text())))
        return cfg["active_class"]

    def limit(self):
        return {"uniform-2k": 32, "uniform-8k": 8}[self.gateway_class()]

    def run_bench(self, argv, log_path, env, timeout_s):
        a = dict(zip(argv, argv[1:]))
        tag = a["--result-filename"].removesuffix(".bench.json")
        self.point = f"bench:{tag}"
        self.events.append(f"bench:{tag}")
        self.bench_argvs.append(argv)
        Path(log_path).write_text("log\n")
        if timeout_s < 10:                                   # every fake bench takes 10 s: a shorter timeout hits
            self.t += timeout_s
            return -9, True
        parts = tag.split("-")
        phase = parts[0]
        if phase == "sentinel":
            w, x = parts[2], int(parts[3][1:])
        else:
            unit = next(p for p in parts[1:] if p[0] in "ci" and p[1:].isdigit())
            w, x = "-".join(parts[1:parts.index(unit)]), int(unit[1:])
        n = int(a["--num-prompts"])
        dataset = Path(a["--dataset-path"]).name if "--dataset-path" in a else ""
        mixed, var = dataset.startswith("mixed"), dataset.startswith("var")
        out = int(a.get("--random-output-len") or a.get("--custom-output-len"))
        prompt = int(a["--random-input-len"]) if "--random-input-len" in a else None
        ttft, tpot, long_ttft = truth(w, x, phase) if phase != "sentinel" else (800.0, 40.0, None)
        lens, ttfts, lats = [], [], []
        for i in range(n):
            long_ = mixed and i % 10 == 5
            t = long_ttft if long_ else ttft
            lens.append(bp.LONG_INPUT if long_ else (64 + (i * 37) % 1857) if var else
                        (8064 if mixed else prompt))
            ttfts.append(t / 1000)
            lats.append(t / 1000 + (out - 1) * tpot / 1000)
        res = {"completed": n, "failed": 0, "duration": 10.0, "request_throughput": n / 10.0,
               "output_throughput": 300.0, "total_token_throughput": 1e5, "input_lens": lens, "output_lens": [out] * n,
               "ttfts": ttfts, "latencies": lats, "start_times": [1000.0 + 0.05 * i for i in range(n)]}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        self.t += 10.0
        return 0, False


class Env:
    def __call__(self, out_path, gpu_lock_held):
        rec = {"runtime": {"vllm": {"vllm": "0.30.0", "torch": "2.13.0+cu130"}},
               "repo": {"commit": COMMIT, "dirty": False, "status": [], "source": "DEPLOYED_COMMIT"},
               "gpu": {"nvidia": [{"name": "NVIDIA GeForce RTX 5070 Ti", "driver_version": "617.42",
                                   "enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]},
               "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def repo(tmp_path):
    """A repo root with c1b (lambda_ref) and the datasets the fake bench never reads."""
    root = tmp_path / "repo"
    (root / "results" / "capacity").mkdir(parents=True)
    (root / "results" / "capacity" / "formal-20261017-c1b").symlink_to(C1B)
    data = root / "data"
    data.mkdir()
    paths, sha = {}, {}
    for key in ("mixed:8192", "var:2048", "var:8192"):
        p = data / f"{key.replace(':', '-')}.jsonl"
        p.write_text(f'{{"prompt": "{key}"}}\n')
        paths[key], sha[key] = str(p), hashlib.sha256(p.read_bytes()).hexdigest()
    return root, paths, sha


def frozen(root, sha):
    c, p = copy.deepcopy(CFG), copy.deepcopy(PINS)
    c["freeze"].update(stage_d=True, mini_closed="smoke-d-a", mini_open="smoke-d-b", mini_verify="smoke-d-c")
    c["stage_d"]["mix_dataset_sha256"] = {"8192": sha["mixed:8192"]}
    p["core_b"]["mix_dataset_sha256"] = {**p["core_b"]["mix_dataset_sha256"], "8192": sha["mixed:8192"]}
    c["stage_d"]["var_dataset_sha256"] = {"2048": sha["var:2048"], "8192": sha["var:8192"]}
    for key, kind in sd.FREEZE_EVIDENCE.items():
        d = root.joinpath(*sd.RESULTS, c["freeze"][key])
        d.mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({"mode": "mini", "kind": kind, "formal": False,
                                                    "smoke": {"ok": True}}))
    return c, p


def session(tmp_path, name, *, mode="formal", world=None, cfg_over=None, plan_over=None, **kw):
    root, paths, sha = repo(tmp_path)
    c, p = frozen(root, sha) if mode == "formal" else (copy.deepcopy(CFG), copy.deepcopy(PINS))
    if cfg_over:
        cfg_over(c)
    plan = rd.expected_plan(c, p, session_name=name, pins_sha256=PINS_SHA, stage_d_sha256=SD_SHA,
                            formal=mode == "formal" and c["freeze"]["stage_d"] is True, repo_root=root)
    if plan_over:
        plan_over(plan)
    w = world or World(tmp_path)
    out = kw.pop("out", root.joinpath(*sd.RESULTS, f"t-{name}"))
    s = rd.run_session(plan, cfg=c, pins=p, pins_sha256=PINS_SHA, stage_d_sha256=SD_SHA, out_dir=out,
                       session_id=out.name, mode=mode, popen=w.popen, signal_group=w.signal_group,
                       group_alive=w.group_alive, http_get=w.http_get, run_bench=w.run_bench,
                       gpu_sample=w.gpu_sample, compute_apps=w.compute_apps, shared_usage=w.shared_usage,
                       lock_held_check=w.lock_check, gpu_lock_held=True, datasets=paths, gateway_python="/venv/python",
                       port_free=kw.pop("port_free", lambda port: True),
                       toolchain_check=lambda env: [], capture_env=Env(), base_env=BASE_ENV, sample_interval=None,
                       now=w.now, sleep=w.sleep, repo_root=root, **kw)
    return out, s, w


def analysis(out):
    out = Path(out)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    reqs = {r["tag"]: [json.loads(x) for x in (out / r["derived"]).read_text().splitlines()]
            for r in rows if r.get("derived")}
    cap = {"plan": json.loads((out / "plan.json").read_text()), "session": json.loads((out / "session.json").read_text()),
           "rows": rows, "confirm_plan": json.loads((out / "confirm_plan.json").read_text())
           if (out / "confirm_plan.json").is_file() else None}
    return sda.analyze(cap, reqs), rows


# ------------------------------------------------------------------ D-A --

def test_a_closed_session_runs_030s_protocol_with_the_t_server(tmp_path):
    out, s, w = session(tmp_path, "D-A1")
    a, rows = analysis(out)
    assert a["replay"] == {"ok": True, "problems": []} and all(r["valid"] for r in rows)
    assert (a["workloads"]["U32K"]["c_prod"], a["workloads"]["U8K-T"]["c_prod"], a["workloads"]["M8K-T"]["c_prod"]) == (
        {"value": 2, "status": "EXACT"}, {"value": 5, "status": "EXACT"}, {"value": 3, "status": "EXACT"})
    t_flag = ["--long-prefill-token-threshold", "512"]
    labels = [b["workload"] for b in s["boots"]]
    t_boots = [argv[-2:] == t_flag for argv in w.server_argvs]
    assert t_boots == [sd.workload(x)["server"] == "T" for x in labels]
    assert s["formal"] and s["kind"] == "closed" and s["dataset_sha256"].keys() == {"mixed:8192"}
    assert a["hypotheses"]["H-D2b"]["U8K-T"]["verdict"] == "supported"


def test_a_t_server_without_its_threshold_fails_before_its_first_bench(tmp_path):
    world = World(tmp_path, bad_log={"S-U8K-T-a1": "threshold"})
    out, s, w = session(tmp_path, "D-A1", world=world)
    boot = next(b for b in s["boots"] if b["workload"] == "U8K-T" and b["attempt"] == 1)
    assert boot["reached_first_bench"] is False and "long_prefill_token_threshold" in boot["problems"][0]
    assert sda.analyze(*analysis_parts(out))["replay"]["ok"]                 # retried once at the end of Phase S


def analysis_parts(out):
    out = Path(out)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    reqs = {r["tag"]: [json.loads(x) for x in (out / r["derived"]).read_text().splitlines()]
            for r in rows if r.get("derived")}
    return ({"plan": json.loads((out / "plan.json").read_text()), "session": json.loads((out / "session.json").read_text()),
             "rows": rows, "confirm_plan": json.loads((out / "confirm_plan.json").read_text())}, reqs)


def test_512_token_runs_use_their_own_client_and_checks(tmp_path):
    out, s, w = session(tmp_path, "D-A2")
    a, rows = analysis(out)
    assert a["replay"]["ok"] and all(r["valid"] for r in rows)
    o512 = [argv for argv in w.bench_argvs if "--random-output-len" in argv
            and argv[argv.index("--random-output-len") + 1] == "512"]
    assert o512 and all(argv[argv.index("--random-input-len") + 1] in ("1920", "8064") for argv in o512)
    assert a["workloads"]["U2K-O512"]["c_prod"]["value"] == 31


# ------------------------------------------------------------------ D-B --

def test_an_open_session_sends_poisson_arrivals_at_the_grid_rates(tmp_path):
    out, s, w = session(tmp_path, "D-B")
    a, rows = analysis(out)
    assert a["replay"]["ok"] and all(r["valid"] for r in rows)
    runs = [r for r in rows if r["phase"] in ("S", "C")]
    assert all("rate" in r for r in runs)
    open_argvs = [argv for argv in w.bench_argvs if "--burstiness" in argv]
    assert len(open_argvs) == len(runs) and not any("--max-concurrency" in argv for argv in open_argvs)
    u = a["workloads"]["U2K-P"]
    assert u["c_prod"] == {"value": 5, "status": "EXACT"}
    assert u["rate_prod"] == pytest.approx(round(0.9 * sd.lambda_ref(C1B, "U2K")["lambda_ref"], 4))
    assert a["hypotheses"]["H-D4"]["U2K-P"]["verdict"] == "supported"


# ------------------------------------------------------------------ D-C --

def test_a_verify_session_boots_every_point_n_times_and_runs_the_gateway(tmp_path):
    out, s, w = session(tmp_path, "D-C")
    a, rows = analysis(out)
    assert a["replay"]["ok"] and len(s["boots"]) == 22 and all(r["valid"] for r in rows)
    assert s["capserve_sha256"] == rd.capserve_sha256(ROOT, "service/profiles/verified-formal-20261017-c1b.json")
    assert {k: v["class"] for k, v in a["workloads"].items()} == dict.fromkeys(
        ("U2K-var", "U8K-var", "U2K-gw", "U8K-gw"), "PASS")
    assert len(w.gateway_argvs) == 10
    argv = w.gateway_argvs[0]
    assert argv[:4] == ["/venv/python", "-m", "capserve", "serve"]
    assert argv[-1] == str(ROOT / "service" / "profiles" / "verified-formal-20261017-c1b.json")
    settings = tomllib.loads(Path(argv[5]).read_text())
    assert settings["tokenizer"] == "local" and settings["backend_url"] == "http://127.0.0.1:8000"
    assert settings["tokenizer_path"] == bp.snapshot_dir(PINS) and settings["port"] == 8080
    env = w.gateway_envs[0]
    assert "CAPSERVE_API_KEY" not in env and env["HF_HUB_OFFLINE"] == "1"
    # the gateway is stopped before the server's teardown, every time
    ev = [e for e in w.events if e.startswith(("gateway", "server:signal"))]
    for i, e in enumerate(ev):
        if e == "gateway:start":
            assert ev[i + 1] == f"gateway:signal:{int(signal.SIGTERM)}" and ev[i + 2].startswith("server:signal")
    gw_argvs = [x for x in w.bench_argvs if "http://127.0.0.1:8080" in x]
    assert len(gw_argvs) == 10 and all(argv[argv.index("--max-concurrency") + 1] in ("32", "8") for argv in gw_argvs)
    assert a["hypotheses"]["H-D7"]["U8K-gw"]["verdict"] == "supported"


def test_a_gateway_serving_another_class_fails_its_boot(tmp_path):
    world = World(tmp_path, readyz={"profile": {"id": "verified-formal-20261017-c1b", "status": "verified"},
                                    "active_class": "uniform-8k", "limits": {"max_concurrency": 8}})
    out, s, w = session(tmp_path, "D-C", world=world)
    bad = [b for b in s["boots"] if b["workload"] == "U2K-gw"]
    assert len(bad) == 1 and bad[0]["reached_first_bench"] is False and "gateway serves" in bad[0]["problems"][0]
    a, rows = analysis(out)
    assert a["replay"]["ok"] and a["workloads"]["U2K-gw"]["class"] == "NT"


# ------------------------------------------------------------------ mini --

@pytest.mark.parametrize("name, kind", [("D-A1", "closed"), ("D-B", "open"), ("D-C", "verify")])
def test_the_mini_smokes_pass_and_discard_latencies(tmp_path, name, kind):
    root = tmp_path / "repo"
    out, s, w = session(tmp_path, name, mode="mini", out=root.joinpath(*sd.RESULTS, f"smoke-d-{kind}"))
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["ok"] is True, smoke
    assert s["mode"] == "mini" and s["formal"] is False and s["kind"] == kind
    assert not list((out / "raw").glob("*.bench.json"))
    for p in (out / "derived").glob("*-requests.jsonl"):
        assert all(set(json.loads(x)) == {"idx", "cls", "out"} for x in p.read_text().splitlines())
    if kind == "closed":
        assert smoke["timeouts"] == [["U2K-O512", 3]]


# ------------------------------------------------------------------ refusals --

def test_refusals_before_anything_starts(tmp_path):
    with pytest.raises(rd.RunError, match="results/stage_d"):
        session(tmp_path / "a", "D-A1", out=tmp_path / "elsewhere")
    with pytest.raises(rd.RunError, match="freeze.stage_d"):
        session(tmp_path / "b", "D-A1", mode="formal", cfg_over=lambda c: c["freeze"].update(stage_d=False))
    with pytest.raises(rd.RunError, match="plan .* differs"):
        session(tmp_path / "c", "D-A1", plan_over=lambda p: p["workloads"]["U32K"].update(c_cap=7))
    with pytest.raises(rd.RunError, match="sha256"):
        session(tmp_path / "d", "D-C", cfg_over=lambda c: c["stage_d"].update(
            var_dataset_sha256={"2048": "e" * 64, "8192": "e" * 64}))


def test_the_runner_never_imports_capserve():
    text = (ROOT / "benchmarks" / "run_stage_d.py").read_text()
    assert "import capserve" not in text and "from capserve" not in text


def test_a_busy_gateway_port_fails_the_boot_without_starting_a_gateway(tmp_path):
    out, s, w = session(tmp_path, "D-C", port_free=lambda port: port != 8080)
    gw_boots = [b for b in s["boots"] if b["workload"].endswith("-gw")]
    assert gw_boots and all("port 8080 is in use" in b["problems"][0] for b in gw_boots)
    assert w.gateway_argvs == []


def test_a_gateway_that_dies_during_a_bench_invalidates_the_run(tmp_path):
    class Dying(World):
        def run_bench(self, argv, log_path, env, timeout_s):
            out = super().run_bench(argv, log_path, env, timeout_s)
            if "8080" in " ".join(argv) and self.gw is not None:
                self.gw.returncode = 1
            return out

    out, s, w = session(tmp_path, "D-C", world=Dying(tmp_path))
    a, rows = analysis(out)
    gw_rows = [r for r in rows if r["workload"].endswith("-gw")]
    assert a["replay"]["ok"] and gw_rows and all("gateway exited" in " ".join(r["reasons"]) for r in gw_rows)
    assert {r["workload"] for r in gw_rows} == {"U2K-gw", "U8K-gw"} and len(gw_rows) == 2     # skipped after


def test_open_loop_runs_send_their_own_seed(tmp_path):
    out, s, w = session(tmp_path, "D-B")
    seeds = [argv[argv.index("--seed") + 1] for argv in w.bench_argvs if "--burstiness" in argv]
    assert len(seeds) == len(set(seeds)) > 5


def test_the_d_c_gateway_uses_the_production_configs_settings():
    """039 D-C measures the production gateway: its tokenizer mode and class come from rtx-verified.toml's [gateway]
    and pinned profile (the class of each workload is the profile's), only the addresses are the research's."""
    prod = tomllib.loads((ROOT / "service" / "deploy" / "rtx-verified.toml").read_text())
    plan = rd.expected_plan(CFG, PINS, session_name="D-C", pins_sha256=PINS_SHA, stage_d_sha256=SD_SHA, formal=False)
    for w in ("U2K-gw", "U8K-gw"):
        s = rd.gateway_settings(plan, PINS, w)
        assert s["tokenizer"] == prod["gateway"]["tokenizer"] == "local"
        assert s["tokenizer_path"] == bp.snapshot_dir(PINS)
    assert plan["verify"]["gateway_profile"] == prod["deploy"]["production_profile"]
    profile = json.loads((ROOT / prod["deploy"]["production_profile"]).read_text())
    limits = {c["name"]: c["max_concurrency"] for c in profile["classes"]}
    assert (limits["uniform-2k"], limits["uniform-8k"]) == (sd.VERIFY["points"]["U2K-gw"], sd.VERIFY["points"]["U8K-gw"])
