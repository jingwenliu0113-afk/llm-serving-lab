"""Tests for benchmarks/run_capacity.py: the Stage C (C1) session runner (decision 030 8.4 / 8.6 / 8.7;
PREREGISTRATION §8), with a fake server, /metrics, OTLP receiver, clock, env capture and `vllm bench serve` (no GPU).
Real execution happens in WSL under the GPU lock: the FS-order and mini smokes before the freeze, the formal session
after it."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import run_adv as ra
import run_capacity as rc
from run_core_a import sha256_json

ROOT = Path(__file__).resolve().parent.parent
CORE_UNIFORM = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"
PINS_SHA, CAP_SHA = "ab" * 32, "cd" * 32
EPOCH = 1_760_000_000.0
COMMIT = "97a948c911940249b3d04bff38727a1fcb8b1d64"
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u"}
DERIVED_KEYS = {"idx", "cls", "ttft_ms", "latency_ms", "out"}


# ------------------------------------------------------------------ truth, plans --

def scenario(w, C, phase, block):
    """The S2 truth of test_capacity_analysis: U2K TPOT 3C ms (C^ 33), U8K TPOT 12C ms (C^ 8), M2K short TTFT 6 s from
    C 2 on (C^ 1), M8K long TTFT 21 s (infeasible); a start-up wave in idx < C."""
    if w == "U2K":
        return {"short": 900, "tpot": 3 * C, "burst": 8000}
    if w == "U8K":
        return {"short": 2500, "tpot": 12 * C, "burst": 11000}
    if w == "M2K":
        return {"short": 600 if C == 1 else 6000, "long": 6300, "tpot": 20}
    return {"short": 1500, "long": 21000, "tpot": 30}


@pytest.fixture(scope="module")
def cfg():
    return tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())


@pytest.fixture(scope="module")
def pins():
    return tomllib.loads((ROOT / "configs" / "pins.toml").read_text())


def make_plan(cfg, pins, *, formal=False):
    session, up = cp.load_uniform_session(CORE_UNIFORM)
    return cp.build_plan(cfg, pins, session=session, uniform_plan=up, pins_sha256=PINS_SHA, capacity_sha256=CAP_SHA,
                         formal=formal)


@pytest.fixture(scope="module")
def plan(cfg, pins):
    return make_plan(cfg, pins)


def datasets(tmp_path):
    """Stand-in mixed datasets (the fake bench never reads them) and their sha256."""
    d = Path(tmp_path) / "data"
    d.mkdir(parents=True, exist_ok=True)
    paths, sha = {}, {}
    for L in (2048, 8192):
        p = d / f"mixed-{L}.jsonl"
        p.write_text(f'{{"prompt": "x{L}", "output_tokens": 128}}\n')
        paths[L], sha[str(L)] = str(p), hashlib.sha256(p.read_bytes()).hexdigest()
    return paths, sha


EVIDENCE = {"fs_order_smoke": "smoke-fs-order-t", "mini_smoke": "smoke-mini-t", "c0_decision": "031"}


def write_evidence(root):
    """The freeze evidence (030 implementation note) under a test repo root: two passed smokes and C0's result log."""
    root = Path(root)
    files = {root / "results" / "capacity" / EVIDENCE[key] / "session.json":
             json.dumps({"mode": mode, "formal": False, "smoke": {"ok": True}})
             for key, mode in (("fs_order_smoke", "fs-order"), ("mini_smoke", "mini"))}
    files[root / "docs" / "decisions" / f"{EVIDENCE['c0_decision']}-c0-result.md"] = "# C0\n"
    for path, text in files.items():
        if not path.exists():                      # idempotent: never touch evidence that is already there
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def frozen(cfg, pins, sha, branch="A"):
    """capacity.toml after the 030 freeze (c0_branch set, freeze.capacity = true, evidence named), with the stand-in
    datasets pinned."""
    c, p = copy.deepcopy(cfg), copy.deepcopy(pins)
    c["freeze"]["capacity"] = True
    c["freeze"].update(EVIDENCE)
    c["capacity"]["c0_branch"] = branch
    c["capacity"]["mix_dataset_sha256"] = dict(sha)
    p["core_b"]["mix_dataset_sha256"] = dict(sha)
    return c, p


def formal_setup(tmp_path, cfg, pins):
    ds, sha = datasets(tmp_path)
    c, p = frozen(cfg, pins, sha)
    return c, p, make_plan(c, p, formal=True), ds


# ------------------------------------------------------------------ fake world --

def stamp(t, line):
    return f"{int(t * 1000)} {line}"


def nondefault_from_argv(argv):
    """What vLLM 0.30.0 logs as non-default args for a server argv (api_utils.py:286, a dict repr)."""
    a = dict(zip(argv, argv[1:]))
    d = {"model_tag": argv[2], "max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
         "attention_backend": "FLASH_ATTN", "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"])}
    if "--otlp-traces-endpoint" in a:
        d["otlp_traces_endpoint"] = a["--otlp-traces-endpoint"]
    if "--enable-logging-iteration-details" in argv:
        d["enable_logging_iteration_details"] = True
    return d


class FakeReader:
    def __init__(self):
        self.error = None

    def join(self, timeout):
        return True


class FakeProc:
    pid = 77

    def __init__(self):
        self.returncode = None
        self.log_reader = FakeReader()

    def poll(self):
        return self.returncode


def json_decode(body, gz):
    return json.loads(body)


class FakeReceiver(ra.OtlpReceiver):
    """run_adv's receiver logic (raw bodies, JSONL) without a socket."""

    def __init__(self, world, *a, **kw):
        super().__init__(*a, decode=json_decode, **kw)
        self.world = world

    def start(self):
        self.world.events.append("receiver:start")
        return self

    def close(self):
        self.world.events.append("receiver:close")
        return []


class FakeEnv:
    """Fake scripts/env_capture.py --vllm-python record of a deployed bundle."""

    def __init__(self, world=None, *, enforced=300.0, vllm="0.30.0", dirty=False, source="DEPLOYED_COMMIT",
                 commits=None):
        self.world, self.enforced, self.vllm, self.dirty, self.source = world, enforced, vllm, dirty, source
        self.commits, self.calls = commits or {}, []

    def __call__(self, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        if self.world is not None:
            self.world.events.append(f"env:{Path(out_path).name}")
        commit = self.commits.get(len(self.calls), COMMIT)
        rec = {"runtime": {"vllm": {"vllm": self.vllm, "torch": "2.13.0+cu130"}},
               "repo": {"commit": commit, "dirty": self.dirty, "status": [], "source": self.source},
               "gpu": {"nvidia": [{"name": "NVIDIA GeForce RTX 5070 Ti", "driver_version": "617.42",
                                   "enforced_power_limit_w": self.enforced, "power_default_limit_w": 300.0}]},
               "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def parse_tag(tag):
    """(phase, workload, C, block) of a capacity_plan.run_tag."""
    parts = tag.split("-")
    if parts[0] == "sentinel":
        return "sentinel", parts[2], int(parts[3][1:]), None
    if parts[0] == "S":
        return "S", parts[1], int(parts[2][1:]), None
    if parts[0] == "C":
        return "C", parts[1], int(parts[2][1:]), int(parts[3][1:])
    return "fs", "U2K", int(parts[-1][1:]), None


class World:
    """Fake WSL. boots[label] (label = the boot tag without its bNNN- number): "not_ready", "no_flash", "blocks",
    "obs_key", "mnbt", "no_args", "dies" (the server exits after its first bench), "exits_after_checks" (right after
    the boot checks read /metrics). bench[run tag]: "timeout", "exit", "failed", "short_arrays", "no_long_in_w",
    "timeout_with_result", "garbage" (an unreadable result file); a bench whose timeout is below its 10 s also times
    out. lock: a bool or {lock check number: bool}. arrivals(C, N, prefix) -> {idx: offset s} for the FS-order spans
    (default: dispatch order). thr_scale[run tag] scales that run's output throughput."""

    def __init__(self, tmp_path, *, truth=scenario, boots=None, bench=None, mem=400, apps=(), lock=True,
                 arrivals=None, thr_scale=None):
        self.tmp_path, self.truth, self.boots, self.bench = Path(tmp_path), truth, boots or {}, bench or {}
        self.thr_scale = thr_scale or {}
        self.mem, self.apps, self.lock, self.arrivals = mem, list(apps), lock, arrivals
        self.t, self.events, self.labels, self.benches, self.bench_argvs, self.bench_timeouts = 0.0, [], [], [], [], {}
        self.server_argvs, self.server_envs, self.receivers, self.lock_calls = [], [], [], 0
        self.proc = self.how = None
        self.blocks, self.benches_this_boot = 248, 0
        self.session_at_first_c = None
        self.bench_s = 10.0                                    # every fake bench takes 10 s

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def wall(self):
        return EPOCH + self.t

    def lock_check(self):
        self.lock_calls += 1
        self.events.append("lock")
        if isinstance(self.lock, dict):
            return self.lock.get(self.lock_calls, True)
        return self.lock

    def gpu_mem_used(self):
        self.events.append("mem")
        return self.mem

    def compute_apps(self):
        self.events.append("apps")
        return list(self.apps)

    def receiver_factory(self, raw_dir, spans_path, prefix):
        r = FakeReceiver(self, raw_dir, spans_path, prefix=prefix)
        self.receivers.append(r)
        return r

    def popen(self, argv, env, log_path):
        log_path = Path(log_path)
        label = log_path.name.split("-", 1)[1].removesuffix(".server.log")
        if label.startswith("C-") and self.session_at_first_c is None:
            out = log_path.parent.parent
            self.session_at_first_c = {"session": json.loads((out / "session.json").read_text()),
                                       "confirm_plan": json.loads((out / "confirm_plan.json").read_text())}
        self.events.append(f"popen:{label}")
        self.labels.append(label)
        self.server_argvs.append(argv)
        self.server_envs.append(env)
        how = self.how = self.boots.get(label)
        nd = nondefault_from_argv(argv)
        if how == "obs_key":
            nd["enable_logging_iteration_details"] = True
        if how == "mnbt":
            nd["max_num_batched_tokens"] = 8192
        lines = [] if how == "no_args" else [f"(APIServer pid=1) INFO non-default args: {nd!r}"]
        backend = "TRITON_ATTN" if how == "no_flash" else "FLASH_ATTN"
        lines += [f"(EngineCore pid=2) INFO [cuda.py:478] Using AttentionBackendEnum.{backend} backend.",
                  "(EngineCore pid=2) INFO GPU KV cache size: 130,944 tokens"]
        log_path.write_text("".join(stamp(self.wall(), x) + "\n" for x in lines))
        self.proc = FakeProc()
        if how == "not_ready":
            self.proc.returncode = 1
        self.blocks = 247 if how == "blocks" else 248
        self.benches_this_boot = 0
        return self.proc

    def signal_group(self, proc, sig):
        self.events.append(f"signal:{int(sig)}")
        proc.returncode = 0

    def group_alive(self, proc):
        return proc.returncode is None

    def metrics_text(self):
        lab = 'engine="0",model_name="m"'
        return "\n".join([
            f'vllm:cache_config_info{{block_size="528",engine="0",num_gpu_blocks="{self.blocks}"}} 1.0',
            f"vllm:kv_cache_usage_perc{{{lab}}} 0.5", f"vllm:num_requests_running{{{lab}}} 3.0",
            f"vllm:num_requests_waiting{{{lab}}} 1.0", f"vllm:num_preemptions_total{{{lab}}} 0.0", ""])

    def http_get(self, url):
        alive = self.proc is not None and self.proc.returncode is None
        if url.endswith("/health"):
            self.events.append("health")
            return (200, "") if alive else (0, "connection refused")
        if not alive:
            return 0, "connection refused"
        text = self.metrics_text()
        if self.how == "exits_after_checks" and self.benches_this_boot == 0:
            self.proc.returncode = 1                   # the server dies right after the boot checks read /metrics
        return 200, text

    def run_bench(self, argv, log_path, env, timeout_s):
        a = dict(zip(argv, argv[1:]))
        tag = a["--result-filename"].removesuffix(".bench.json")
        self.events.append(f"bench:{tag}")
        self.benches.append(tag)
        self.bench_argvs.append(argv)
        self.bench_timeouts[tag] = timeout_s
        Path(log_path).write_text("bench log\n")
        how = self.bench.get(tag, "ok")
        self.benches_this_boot += 1
        if self.how == "dies":
            self.proc.returncode = -9
        if how == "timeout" or timeout_s < self.bench_s:      # a bench longer than its timeout is killed
            self.t += timeout_s
            return -9, True
        if how == "exit":
            return 1, False
        if how == "garbage":
            Path(a["--result-dir"], a["--result-filename"]).write_text("{not json")
            return 0, False
        N, C = int(a["--num-prompts"]), int(a["--max-concurrency"])
        phase, w, _, block = parse_tag(tag)
        L = int(a["--random-input-len"]) + 128 if "--random-input-len" in a else cp.WORKLOAD_SPEC[w][0]
        mixed = "--dataset-path" in a
        spec = self.truth(w, C, phase, block)
        if mixed and how == "no_long_in_w":          # every long request in the first or the last C (outside W)
            long_idx = set([i for i in range(N) if i < C or i >= N - C][:N // 10])
            assert len(long_idx) == N // 10
        else:
            long_idx = {i for i in range(N) if mixed and i % 10 == 5}
        lens, ttfts, lats = [], [], []
        for i in range(N):
            t = spec.get("long", 0) if i in long_idx else spec["short"]
            if i < C and "burst" in spec:
                t = spec["burst"]
            lens.append(bp.LONG_INPUT if i in long_idx else L - bp.N_GEN)
            ttfts.append(t / 1000)
            lats.append(t / 1000 + 127 * spec["tpot"] / 1000)
        failed = 1 if how == "failed" else 0
        res = {"completed": N - failed, "failed": failed, "duration": 10.0, "request_throughput": N / 10,
               "output_throughput": N * 12.8 * self.thr_scale.get(tag, 1.0), "total_token_throughput": 1e5,
               "input_lens": lens,
               "output_lens": [bp.N_GEN] * N, "ttfts": ttfts, "latencies": lats[:-1] if how == "short_arrays" else lats,
               "generated_texts": ["secret prompt text"] * N}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        prefix = a.get("--request-id-prefix")
        if prefix is not None and self.receivers:
            offsets = self.arrivals(C, N, prefix) if self.arrivals else {i: 0.01 * i for i in range(N)}
            spans = [{"name": "llm_request", "start_time_unix_nano": int(round((EPOCH + self.t + off) * 1e9)),
                      "attributes": {"gen_ai.request.id": f"cmpl-{prefix}{i}-0", "gen_ai.usage.prompt_tokens": 1920,
                                     "gen_ai.latency.time_to_first_token": 0.5}}
                     for i, off in offsets.items()]
            spans.append({"name": "llm_request", "start_time_unix_nano": 1,
                          "attributes": {"gen_ai.request.id": "cmpl-warmup-0"}})
            assert self.receivers[-1].receive(json.dumps(spans).encode(), None) == 200
        self.t += self.bench_s
        return (-9, True) if how == "timeout_with_result" else (0, False)


def run(world, plan, cfg, pins, *, out=None, mode="formal", evidence=True, **kw):
    out = out or world.tmp_path / "repo" / "results" / "capacity" / "s1"
    kw.setdefault("repo_root", world.tmp_path / "repo")
    if evidence and (cfg.get("freeze") or {}).get("capacity") is True:
        write_evidence(kw["repo_root"])
    kw.setdefault("gpu_lock_held", True)
    kw.setdefault("lock_held_check", world.lock_check)
    kw.setdefault("toolchain_check", lambda env: [])
    kw.setdefault("capture_env", FakeEnv(world))
    kw.setdefault("sample_interval", None)
    kw.setdefault("base_env", BASE_ENV)
    kw.setdefault("receiver_factory", world.receiver_factory)
    if "mixed_datasets" not in kw:
        kw["mixed_datasets"] = datasets(world.tmp_path)[0]
    summary = rc.run_session(plan, cfg=cfg, pins=pins, pins_sha256=PINS_SHA, capacity_sha256=CAP_SHA, out_dir=out,
                             session_id=out.name, mode=mode, popen=world.popen, signal_group=world.signal_group,
                             group_alive=world.group_alive, http_get=world.http_get, run_bench=world.run_bench,
                             gpu_mem_used=world.gpu_mem_used, compute_apps=world.compute_apps, now=world.now,
                             sleep=world.sleep, **kw)
    return out, summary, read_rows(out)


def read_rows(out):
    path = Path(out) / "rows.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def load(out):
    """{session, plan, confirm_plan, rows} and the derived requests of a session directory."""
    out = Path(out)
    rows = read_rows(out)
    requests = {}
    for r in rows:
        p = out / "derived" / f"{r['tag']}-requests.jsonl"
        if p.is_file():
            requests[r["tag"]] = [json.loads(x) for x in p.read_text().splitlines()]
    cap = {"session": json.loads((out / "session.json").read_text()),
           "plan": json.loads((out / "plan.json").read_text()),
           "confirm_plan": json.loads((out / "confirm_plan.json").read_text()), "rows": rows}
    return cap, requests


def replay(cap, requests):
    return ca.replay_problems(cap["plan"], cap["confirm_plan"], cap["session"]["boots"], cap["rows"], requests,
                              confirm_sha256=cap["session"]["confirm_plan_sha256"])


def vs(e):
    return e["value"], e["status"]


@pytest.fixture(scope="module")
def formal_run(tmp_path_factory, cfg, pins):
    tmp = tmp_path_factory.mktemp("formal")
    c, p, plan, ds = formal_setup(tmp, cfg, pins)
    w = World(tmp)
    out, summary, rows = run(w, plan, c, p, mixed_datasets=ds)
    return {"world": w, "out": out, "summary": summary, "rows": rows, "cfg": c, "pins": p, "plan": plan, "ds": ds}


# ------------------------------------------------------------------ output location (030 8.7) --

@pytest.mark.parametrize("rel, ok", [
    ("results/capacity/formal-20261020-c1", True),
    ("results/capacity/smoke/20261017-mini", True),
    ("results/capacity", False),                                  # the directory itself, not a session
    ("results/core_b/formal-20261020-c1", False),
    ("results/core_a/x", False), ("results/quality/x", False), ("results/advanced/x", False),
    ("results/decision_table/x", False),
    ("results/capacity/../advanced/x", False),                    # escapes after normalisation
    ("runs/pt-20261017", False), ("results/capacity/runs/x", False),
    ("elsewhere/capacity/x", False),
])
def test_outputs_go_only_under_results_capacity(tmp_path, rel, ok):
    problems = rc.output_problems(tmp_path / rel, repo_root=tmp_path)
    assert (problems == []) is ok, problems


def test_output_check_follows_symlinks(tmp_path):
    (tmp_path / "results" / "capacity").mkdir(parents=True)
    (tmp_path / "results" / "core_b").mkdir()
    (tmp_path / "results" / "capacity" / "link").symlink_to(tmp_path / "results" / "core_b")
    assert rc.output_problems(tmp_path / "results" / "capacity" / "link" / "s", repo_root=tmp_path)


# ------------------------------------------------------------------ plan check --

def test_the_repo_plan_passes_the_plan_check(cfg, pins, plan):
    assert rc.plan_problems(plan, cfg, pins, pins_sha256=PINS_SHA, capacity_sha256=CAP_SHA, formal=False) == []


@pytest.mark.parametrize("change, match", [
    (lambda p: p["workloads"]["U8K"]["timeouts_s"].update({"8": 600}), "workloads"),
    (lambda p: p["workloads"]["U2K"].update(c_cap=128), "workloads"),
    (lambda p: p["slo"].update(tpot_p95_ms=200), "slo"),
    (lambda p: p.update(guard_band=0.0), "guard_band"),
    (lambda p: p["sentinel"].update(C=8), "sentinel"),
    (lambda p: p.update(pins_sha256="00" * 32), "pins_sha256"),
    (lambda p: p.update(capacity_sha256="00" * 32), "capacity_sha256"),
    (lambda p: p.update(expected_blocks=247), "expected_blocks"),
])
def test_plan_check_recomputes_every_value_from_capacity_toml_and_pins(cfg, pins, plan, change, match):
    p = copy.deepcopy(plan)
    change(p)
    problems = rc.plan_problems(p, cfg, pins, pins_sha256=PINS_SHA, capacity_sha256=CAP_SHA, formal=False)
    assert problems and any(match in x for x in problems), problems


def test_plan_check_refuses_a_config_that_breaks_030(cfg, pins, plan):
    c = copy.deepcopy(cfg)
    c["capacity"]["n_low"] = 2
    assert rc.plan_problems(plan, c, pins, pins_sha256=PINS_SHA, capacity_sha256=CAP_SHA, formal=False)


# ------------------------------------------------------------------ boot checks, 030 8.6 (b) (c) (d) --

def formal_server_log(pins, plan, **edit):
    argv, _ = cp.server_argv_for(pins)
    nd = {**nondefault_from_argv(argv), **edit}
    nd = {k: v for k, v in nd.items() if v is not None}
    return (f"1 (APIServer pid=1) INFO non-default args: {nd!r}\n"
            "2 (EngineCore pid=2) INFO [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.\n")


def metrics(blocks=248, block_size=528):
    return f'vllm:cache_config_info{{block_size="{block_size}",engine="0",num_gpu_blocks="{blocks}"}} 1.0\n'


def test_the_core_b_server_passes_the_boot_checks(pins, plan):
    rec, problems = rc.server_checks(formal_server_log(pins, plan), metrics(), ready=True, plan=plan)
    assert problems == []
    assert rec["nondefault"] == {"max_num_batched_tokens": 2048, "kv_cache_memory_bytes": 4294967296}
    assert rec["cache_config"] == {"num_gpu_blocks": 248, "block_size": 528}
    assert rec["log_facts"]["attention_backend"] == "FLASH_ATTN"


@pytest.mark.parametrize("log_edit, metrics_text, ready, match", [
    ({}, metrics(), False, "not ready"),                                                      # (b)
    ({"enable_logging_iteration_details": True}, metrics(), True, "enable_logging_iteration_details"),   # (c)
    ({"otlp_traces_endpoint": "http://127.0.0.1:4318/v1/traces"}, metrics(), True, "otlp_traces_endpoint"),
    ({"long_prefill_token_threshold": 512}, metrics(), True, "long_prefill_token_threshold"),  # T
    ({"scheduler_reserve_full_isl": False}, metrics(), True, "scheduler_reserve_full_isl"),   # G
    ({"max_num_batched_tokens": 8192}, metrics(), True, "max_num_batched_tokens"),
    ({"kv_cache_memory_bytes": 3221225472}, metrics(), True, "kv_cache_memory_bytes"),        # K3
    ({"kv_cache_memory_bytes": None}, metrics(), True, "kv_cache_memory_bytes"),
    ({}, metrics(blocks=247), True, "num_gpu_blocks"),                                         # (d)
    ({}, metrics(block_size=512), True, "num_gpu_blocks"),
    ({}, "", True, "cache_config_info"),
])
def test_boot_checks_b_c_d(pins, plan, log_edit, metrics_text, ready, match):
    _, problems = rc.server_checks(formal_server_log(pins, plan, **log_edit), metrics_text, ready=ready, plan=plan)
    assert problems and any(match in p for p in problems), problems


def test_boot_checks_need_flash_attn_and_the_args_line(pins, plan):
    log = formal_server_log(pins, plan).replace("FLASH_ATTN", "TRITON_ATTN")
    assert any("FLASH_ATTN" in p for p in rc.server_checks(log, metrics(), ready=True, plan=plan)[1])
    no_args = formal_server_log(pins, plan).splitlines()[1]
    assert any("non-default args" in p for p in rc.server_checks(no_args, metrics(), ready=True, plan=plan)[1])


# ------------------------------------------------------------------ per-run checks (a) (e), derived (8.7) --

def detailed(N=200, L=2048, mixed=False, ttft=0.9, tpot=0.03):
    lens = [bp.LONG_INPUT if mixed and i % 10 == 5 else L - 128 for i in range(N)]
    return {"completed": N, "failed": 0, "input_lens": lens, "output_lens": [128] * N, "ttfts": [ttft] * N,
            "latencies": [ttft + 127 * tpot] * N, "generated_texts": ["prompt text"] * N}


@pytest.mark.parametrize("key", ["ttfts", "latencies", "input_lens", "output_lens"])
def test_array_lengths_must_be_n(key):
    res = detailed()
    assert rc.array_problems(res, N=200) == []
    res[key] = res[key][:-1]
    assert any(key in p for p in rc.array_problems(res, N=200))
    del res[key]
    assert any(key in p for p in rc.array_problems(res, N=200))


def test_derived_requests_follow_the_030_schema_without_prompt_text():
    res = detailed(mixed=True, ttft=1.5, tpot=0.04)
    out = rc.derived_requests(res, L=2048)
    assert len(out) == 200 and all(set(q) == DERIVED_KEYS for q in out)
    assert [q["idx"] for q in out] == list(range(200))
    assert out[5]["cls"] == "long" and out[4]["cls"] == "short"
    assert out[0]["ttft_ms"] == pytest.approx(1500.0) and out[0]["latency_ms"] == pytest.approx(1500 + 127 * 40.0)
    assert out[0]["out"] == 128
    assert "prompt" not in json.dumps(out)
    assert ca.derived_problems(out, mix="mixed", C=8, N=200) == []


def test_derived_requests_mark_an_unknown_length_and_keep_none():
    res = detailed()
    res["input_lens"][3] = 1000
    res["ttfts"][4] = None
    out = rc.derived_requests(res, L=2048)
    assert out[3]["cls"] == "other" and out[4]["ttft_ms"] is None
    assert ca.derived_problems(out, mix="uniform", C=8, N=200)
    assert rc.derived_requests({**res, "latencies": res["latencies"][:-1]}, L=2048) is None


def test_inflight_peak_is_running_plus_waiting():
    samples = [{"vllm:num_requests_running": 3.0, "vllm:num_requests_waiting": 1.0},
               {"vllm:num_requests_running": 5.0, "vllm:num_requests_waiting": None}, {}]
    assert rc.inflight_peak(samples) == 5.0
    assert rc.inflight_peak([]) is None


# ------------------------------------------------------------------ the formal session (030 8.4) --

def test_formal_session_runs_sentinel_search_confirm_blocks_sentinel(formal_run):
    s, rows = formal_run["summary"], formal_run["rows"]
    assert s["formal"] is True and s["mode"] == "formal" and s["stage"] == "C1" and "aborted" not in s
    assert s["env_valid"] is True and s["env_flags"] == []
    phases = [b["phase"] for b in s["boots"]]
    assert phases[0] == phases[-1] == "sentinel"
    assert phases[1:5] == ["S"] * 4 and set(phases[5:-1]) == {"C"}
    assert [b["workload"] for b in s["boots"][1:5]] == list(cp.WORKLOADS)
    assert [b["position"] for b in s["boots"] if b["phase"] == "sentinel"] == ["start", "end"]
    assert all(b["reached_first_bench"] for b in s["boots"])
    assert [r["phase"] for r in rows][0] == "sentinel" and rows[-1]["phase"] == "sentinel"
    assert all(r["valid"] for r in rows), [r["reasons"] for r in rows if not r["valid"]][:3]


def test_formal_session_replays_cleanly_and_its_analysis_matches_the_truth(formal_run):
    cap, requests = load(formal_run["out"])
    assert replay(cap, requests) == []
    a = ca.analyze(cap, requests)
    assert a["replay"]["ok"] and a["sentinel"]["gate"]["ok"]
    wl = a["workloads"]
    assert wl["U2K"]["K"] == [16, 32, 33, 34] and vs(wl["U2K"]["c_slo"]) == (33, "EXACT")
    assert vs(wl["U8K"]["c_slo"]) == (8, "EXACT") and vs(wl["M2K"]["c_slo"]) == (1, "EXACT")
    assert vs(wl["M8K"]["c_slo"]) == (0, "INFEASIBLE")


def test_confirm_plan_and_its_sha256_are_recorded_before_phase_c(formal_run):
    first = formal_run["world"].session_at_first_c
    assert first is not None, "no Phase C boot"
    assert first["session"]["confirm_plan_sha256"] == cp.confirm_plan_sha256(first["confirm_plan"])
    assert "finished_at" not in first["session"]
    cap, _ = load(formal_run["out"])
    assert cap["session"]["confirm_plan_sha256"] == sha256_json(cap["confirm_plan"])


def test_session_files_and_derived_schema(formal_run):
    out, rows, s = formal_run["out"], formal_run["rows"], formal_run["summary"]
    for name in ("session.json", "plan.json", "confirm_plan.json", "rows.jsonl", "raw_manifest.json"):
        assert (out / name).is_file(), name
    assert (out / "env" / "preflight.json").is_file()
    for r in rows:
        p = out / "derived" / f"{r['tag']}-requests.jsonl"
        assert r["derived"] == f"derived/{r['tag']}-requests.jsonl" and p.is_file()
        reqs = [json.loads(x) for x in p.read_text().splitlines()]
        assert len(reqs) == r["N"] and all(set(q) == DERIVED_KEYS for q in reqs)
        assert s["derived_sha256"][r["tag"]] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert s["rows_sha256"] == hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
    assert "secret prompt text" not in "".join(p.read_text() for p in (out / "derived").iterdir())
    manifest = json.loads((out / "raw_manifest.json").read_text())["files"]
    assert any(k.endswith(".bench.json") for k in manifest) and any(k.endswith(".server.log") for k in manifest)
    assert json.loads((out / "plan.json").read_text()) == formal_run["plan"]
    assert s["plan_sha256"] == sha256_json(formal_run["plan"])


def test_rows_carry_the_identity_timeouts_metrics_and_decisions(formal_run):
    plan, rows = formal_run["plan"], formal_run["rows"]
    w = formal_run["world"]
    for r in rows:
        want_t = plan["sentinel"]["timeout_s"] if r["phase"] == "sentinel" else \
            plan["workloads"][r["workload"]]["timeouts_s"][str(r["C"])]
        assert r["timeout_s"] == want_t == w.bench_timeouts[r["tag"]]
        assert r["N"] == bp.num_prompts(r["C"]) and r["timeout"] is False
        assert r["tag"] == cp.run_tag(r["phase"], r["workload"], r["C"], block=r["block"], position=r["position"])
        assert r["metrics"]["kv_cache_usage_peak"] == 0.5 and r["metrics"]["inflight_peak"] == 4.0
        assert r["output_throughput"] == pytest.approx(r["N"] * 12.8)
        if r["phase"] == "S":
            assert isinstance(r["search_pass"], bool) and isinstance(r["within_band"], bool) and "outcome" not in r
        elif r["phase"] == "C":
            assert r["outcome"] in ("ok", "fail") and "search_pass" not in r


def test_server_and_bench_argv_are_the_core_b_ones_unchanged(formal_run):
    w, s, pins = formal_run["world"], formal_run["summary"], formal_run["pins"]
    want, env = cp.server_argv_for(pins, port=8000, vllm_bin=".venv/bin/vllm", base_env=BASE_ENV)
    assert all(argv == want for argv in w.server_argvs)
    assert not any(f in a for a in w.server_argvs for f in ("--otlp-traces-endpoint",
                                                              "--enable-logging-iteration-details"))
    assert all(rec["argv"] == want and rec["env"] == ra.recorded_server_env(env) for rec in s["runs"].values())
    for r, argv in zip(formal_run["rows"], w.bench_argvs):
        L = r["L"]
        assert argv == r["argv"] == cp.bench_argv_for(
            pins, workload=r["workload"], C=r["C"], result_dir=s["result_dir"],
            result_filename=f"{r['tag']}.bench.json", port=8000,
            mixed_dataset=formal_run["ds"][L] if r["mix"] == "mixed" else None, vllm_bin=".venv/bin/vllm")
        assert "--request-id-prefix" not in argv


def test_every_boot_runs_the_preflight_in_order(formal_run):
    ev = formal_run["world"].events
    pops = [i for i, e in enumerate(ev) if e.startswith("popen:")]
    for i in pops:
        before = ev[:i]
        last_env = max(j for j, e in enumerate(before) if e.startswith("env:") and e.endswith(".before.json"))
        last_apps = max(j for j, e in enumerate(before) if e == "apps")
        last_mem = max(j for j, e in enumerate(before) if e == "mem")
        last_lock = max(j for j, e in enumerate(before) if e == "lock")
        assert last_lock < last_mem < last_apps < last_env < i
    n = len(pops)
    assert sum(e.startswith("env:") and e.endswith(".after.json") for e in ev) == n
    assert formal_run["world"].lock_calls == n + 1          # session start + before every boot


# ------------------------------------------------------------------ refusals before anything starts --

@pytest.mark.parametrize("change, kw, match", [
    (None, {"gpu_lock_held": False}, "lock"),
    (lambda c, p, plan: c["freeze"].update(capacity=False), {}, "freeze.capacity"),
    (lambda c, p, plan: c["capacity"].update(c0_branch="D"), {}, "c0_branch"),
    (lambda c, p, plan: plan["workloads"]["U2K"].update(n_near=3), {}, "workloads"),
    (lambda c, p, plan: plan.update(pins_sha256="00" * 32), {}, "pins_sha256"),
    (lambda c, p, plan: plan.update(capacity_sha256="00" * 32), {}, "capacity_sha256"),
    (lambda c, p, plan: p["stage_b"].update(gpu_memory_utilization="0.85"), {}, "gpu_memory_utilization"),
    (None, {"base_env": {**BASE_ENV, "VLLM_USE_V1": "1"}}, "VLLM_USE_V1"),
    # core_b_plan.server_env keeps an inherited OTEL_* variable, and capacity_report refuses the recorded server env
    # afterwards: refused before the 7-12 h session instead
    (None, {"base_env": {**BASE_ENV, "OTEL_SERVICE_NAME": "x"}}, "OTEL_SERVICE_NAME"),
    (None, {"toolchain_check": lambda env: ["no ninja"]}, "toolchain"),
])
def test_formal_refusals(tmp_path, cfg, pins, change, kw, match):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    if change:
        change(c, p, plan)
    w = World(tmp_path)
    with pytest.raises(rc.RunError, match=match):
        run(w, plan, c, p, mixed_datasets=ds, **kw)
    assert w.labels == [] and not (tmp_path / "repo" / "results" / "capacity" / "s1").exists()


def test_formal_refuses_a_dataset_that_is_not_the_pinned_one(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    Path(ds[8192]).write_text("other\n")
    with pytest.raises(rc.RunError, match="sha256"):
        run(World(tmp_path), plan, c, p, mixed_datasets=ds)
    with pytest.raises(rc.RunError, match="--mixed-dataset"):
        run(World(tmp_path), plan, c, p, mixed_datasets={2048: ds[2048]})


@pytest.mark.parametrize("env, match", [
    (dict(vllm="0.31.0"), "vLLM"), (dict(dirty=True), "dirty"), (dict(enforced=250.0), "power limit"),
    (dict(source="git"), "DEPLOYED_COMMIT"),
])
def test_formal_preflight_refusals(tmp_path, cfg, pins, env, match):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path)
    with pytest.raises(rc.RunError, match=match):
        run(w, plan, c, p, mixed_datasets=ds, capture_env=FakeEnv(w, **env))
    assert w.labels == []


def test_the_lock_is_verified_not_trusted_at_the_session_start(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, lock=False)
    with pytest.raises(rc.RunError, match="does not hold"):
        run(w, plan, c, p, mixed_datasets=ds)
    assert w.labels == [] and not (tmp_path / "repo" / "results" / "capacity" / "s1").exists()


def test_non_formal_modes_also_need_the_held_lock(tmp_path, cfg, pins, plan):
    for mode in ("mini", "fs-order"):
        w = World(tmp_path / mode)
        with pytest.raises(rc.RunError, match="lock"):
            run(w, plan, cfg, pins, mode=mode, gpu_lock_held=False)
        assert w.labels == []


def test_refuses_an_output_outside_results_capacity_and_an_existing_session(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    for bad in (tmp_path / "repo" / "results" / "advanced" / "s1", tmp_path / "repo" / "runs" / "pt-1"):
        w = World(tmp_path)
        with pytest.raises(rc.RunError, match="results/capacity"):
            run(w, plan, c, p, mixed_datasets=ds, out=bad)
        assert not bad.exists() and w.labels == []
    out = tmp_path / "repo" / "results" / "capacity" / "s1"
    out.mkdir(parents=True)
    with pytest.raises(rc.RunError, match="never resumed"):
        run(World(tmp_path), plan, c, p, mixed_datasets=ds)


def test_formal_needs_the_formal_mode_flag_and_an_unknown_mode_is_refused(tmp_path, cfg, pins, plan):
    with pytest.raises(rc.RunError, match="mode"):
        run(World(tmp_path), plan, cfg, pins, mode="session")


# ------------------------------------------------------------------ during the session --

def test_the_lock_is_verified_again_before_every_boot(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, lock={4: False})              # start, sentinel, S-U2K: fine; before S-U8K: lost
    with pytest.raises(rc.RunError, match="any more"):
        run(w, plan, c, p, mixed_datasets=ds)
    s = json.loads((tmp_path / "repo" / "results" / "capacity" / "s1" / "session.json").read_text())
    assert "lock" in s["aborted"] and w.labels == ["sentinel-start-U2K", "S-U2K-a1"]
    assert all(e != "popen:S-U8K-a1" for e in w.events)


@pytest.mark.parametrize("kw, match", [({"mem": 4000}, "memory.used"), ({"apps": ["123, python, 9000"]}, "compute")])
def test_a_busy_gpu_before_a_boot_stops_the_session(tmp_path, cfg, pins, kw, match):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, **kw)
    with pytest.raises(rc.RunError, match=match):
        run(w, plan, c, p, mixed_datasets=ds)
    assert w.labels == []
    s = json.loads((tmp_path / "repo" / "results" / "capacity" / "s1" / "session.json").read_text())
    assert match in s["aborted"]


def test_an_invalid_start_sentinel_stops_the_session(tmp_path, cfg, pins):
    """8.6: the sentinel gate needs both sentinels valid, so a session whose start sentinel failed is never valid."""
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, bench={"sentinel-start-U2K-c25": "exit"})
    with pytest.raises(rc.RunError, match="start sentinel"):
        run(w, plan, c, p, mixed_datasets=ds)
    assert w.labels == ["sentinel-start-U2K"]
    rows = read_rows(tmp_path / "repo" / "results" / "capacity" / "s1")
    assert len(rows) == 1 and rows[0]["valid"] is False


def test_an_exception_mid_session_tears_the_server_down_and_records_the_abort(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path)
    real = w.run_bench

    def boom(argv, *a):
        if "S-U8K-c1.bench.json" in argv:
            raise KeyboardInterrupt
        return real(argv, *a)

    w.run_bench = boom
    with pytest.raises(KeyboardInterrupt):
        run(w, plan, c, p, mixed_datasets=ds)
    out = tmp_path / "repo" / "results" / "capacity" / "s1"
    s = json.loads((out / "session.json").read_text())
    assert s["aborted"].startswith("KeyboardInterrupt") and w.proc.returncode == 0
    last_popen = max(i for i, e in enumerate(w.events) if e.startswith("popen:"))
    assert w.events[last_popen] == "popen:S-U8K-a1" and "signal:2" in w.events[last_popen:]
    assert (out / "raw_manifest.json").is_file()


def test_env_flags_make_the_session_env_invalid(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path)
    env = FakeEnv(w, commits={5: "f" * 40})          # a redeploy between two captures
    _, s, _ = run(w, plan, c, p, mixed_datasets=ds, capture_env=env)
    assert s["env_valid"] is False and any("commit" in f["reason"] for f in s["env_flags"])


# ------------------------------------------------------------------ validity (030 8.6 (a)-(f)) and skip rules --

@pytest.mark.parametrize("bench, match", [
    ("failed", "completed"),                 # (a)
    ("short_arrays", "latencies"),           # (e) array lengths
    ("timeout", "timeout"),                  # (f)
    ("timeout_with_result", "timeout"),      # (f): a result file never rescues a timed-out bench
    ("exit", "exit"),
    ("garbage", "unreadable"),               # a broken result file is an invalid run, never a crash
])
def test_a_phase_c_run_failing_a_rule_is_invalid_and_drops_its_point(tmp_path, cfg, pins, bench, match):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, bench={"C-U8K-c8-r1": bench})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    bad = [r for r in rows if r["tag"] == "C-U8K-c8-r1"][0]
    assert bad["valid"] is False and any(match in x for x in bad["reasons"]), bad["reasons"]
    assert bad["outcome"] == "invalid" and bad["timeout"] is (bench.startswith("timeout"))
    later = sorted({r["C"] for r in rows if r["phase"] == "C" and r["workload"] == "U8K" and r["block"] >= 2})
    assert later == [4, 7]                    # 8 and every higher point leave later blocks (013: never replaced)
    cap, requests = load(out)
    assert replay(cap, requests) == []


def test_a_mixed_run_without_a_long_request_in_w_is_invalid(tmp_path, cfg, pins):
    """8.6 (e): M2K passing everywhere searches up to C 32 (320 requests, 32 long), whose long requests all sit in the
    first or the last C; the run is invalid and counts as a fail for navigation."""
    def truth(w, C, phase, block):
        return {"short": 600, "long": 6300, "tpot": 20} if w == "M2K" else scenario(w, C, phase, block)

    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, truth=truth, bench={"S-M2K-c32": "no_long_in_w"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    bad = [r for r in rows if r["tag"] == "S-M2K-c32"][0]
    assert bad["valid"] is False and bad["search_pass"] is False
    assert any("long request in W" in x for x in bad["reasons"]), bad["reasons"]
    cap, requests = load(out)
    assert replay(cap, requests) == [] and cap["confirm_plan"]["workloads"]["M2K"]["c_hat"] == 31


@pytest.mark.parametrize("how, match", [("not_ready", "not ready"), ("no_flash", "FLASH_ATTN"),
                                        ("obs_key", "enable_logging_iteration_details"), ("mnbt", "max_num_batched"),
                                        ("no_args", "non-default args"), ("blocks", "num_gpu_blocks")])
def test_a_failed_phase_c_boot_records_its_planned_runs_as_invalid(tmp_path, cfg, pins, how, match):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, boots={"C-M2K-r1": how})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    m2 = [r for r in rows if r["workload"] == "M2K" and r["phase"] == "C"]
    assert {r["block"] for r in m2} == {1} and len(m2) == 2
    assert all(not r["valid"] and any(match in x for x in r["reasons"]) for r in m2)
    assert not any(t.startswith("C-M2K") for t in w.benches)          # no bench on a failed boot
    boot = [b for b in s["boots"] if b["phase"] == "C" and b["workload"] == "M2K"][0]
    assert boot["reached_first_bench"] is False and boot["problems"]
    cap, requests = load(out)
    assert replay(cap, requests) == []


def test_a_server_that_dies_mid_boot_makes_its_later_runs_invalid_without_benching(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, boots={"C-U2K-r1": "dies"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    u2 = [r for r in rows if r["workload"] == "U2K" and r["phase"] == "C" and r["block"] == 1]
    assert len(u2) == 4 and u2[0]["valid"] is True
    assert all(not r["valid"] and any("exited" in x for x in r["reasons"]) for r in u2[1:])
    assert sum(t.startswith("C-U2K") and t.endswith("-r1") for t in w.benches) == 1
    cap, requests = load(out)
    assert replay(cap, requests) == []


def test_a_phase_s_boot_that_fails_before_its_first_bench_is_retried_once_at_the_end(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, boots={"S-U8K-a1": "not_ready"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    got = [(b["phase"], b["workload"], b["attempt"], b["reached_first_bench"]) for b in s["boots"][1:6]]
    assert got == [("S", "U2K", 1, True), ("S", "U8K", 1, False), ("S", "M2K", 1, True), ("S", "M8K", 1, True),
                   ("S", "U8K", 2, True)]
    assert not any(r["boot"] == 2 for r in rows)                     # a failed Phase S boot has no row
    cap, requests = load(out)
    assert replay(cap, requests) == [] and cap["confirm_plan"]["workloads"]["U8K"]["c_hat"] == 8


@pytest.mark.parametrize("label", ["S-U8K-a1", "C-M2K-r1"])
def test_a_server_that_exits_after_its_checks_has_failed_before_its_first_bench(tmp_path, cfg, pins, label):
    """030 8.4 step 4: a Phase S boot whose server is gone before the first bench is retried (no row, no C 1 fail);
    a Phase C boot records its planned runs as invalid without benching."""
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, boots={label: "exits_after_checks"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    boot = [b for b in s["boots"] if rc.boot_label(b) == label][0]
    assert boot["reached_first_bench"] is False and any("exited" in x for x in boot["problems"])
    assert not any(r["boot"] == boot["boot"] and r["valid"] for r in rows)
    assert not any(t.startswith(label[:5]) and label.startswith("C-") and t.endswith("-r1") for t in w.benches)
    cap, requests = load(out)
    assert replay(cap, requests) == []
    if label.startswith("S-"):
        assert not any(r["boot"] == boot["boot"] for r in rows)
        assert cap["confirm_plan"]["workloads"]["U8K"]["c_hat"] == 8          # the retry searched normally


def test_a_search_whose_boot_fails_twice_gets_no_points(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, boots={"S-U8K-a1": "blocks", "S-U8K-a2": "blocks"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    cap, requests = load(out)
    assert replay(cap, requests) == []
    assert cap["confirm_plan"]["workloads"]["U8K"]["points"] == [] and not any(
        r["workload"] == "U8K" and r["phase"] == "C" for r in rows)


def test_an_invalid_search_run_counts_as_a_fail_for_navigation(tmp_path, cfg, pins):
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    w = World(tmp_path, bench={"S-U2K-c16": "timeout"})
    out, s, rows = run(w, plan, c, p, mixed_datasets=ds)
    r16 = [r for r in rows if r["tag"] == "S-U2K-c16"][0]
    assert r16["valid"] is False and r16["search_pass"] is False and r16["timeout"] is True
    cap, requests = load(out)
    assert replay(cap, requests) == [] and cap["confirm_plan"]["workloads"]["U2K"]["c_hat"] == 15


def test_a_boot_with_no_active_point_is_skipped(tmp_path, cfg, pins):
    """U8K's lowest point (c_low 4) is invalid in block 1, so every U8K point leaves the later blocks and U8K has no
    boot after block 1; M2K's C 2 FAILs in block 1, so its later boots run C 1 only."""
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    out, s, rows = run(World(tmp_path, bench={"C-U8K-c4-r1": "exit"}), plan, c, p, mixed_datasets=ds)
    assert [b["block"] for b in s["boots"] if b["phase"] == "C" and b["workload"] == "U8K"] == [1]
    assert [b["block"] for b in s["boots"] if b["phase"] == "C" and b["workload"] == "M8K"] == [1, 2, 3, 4, 5]
    m2 = [(r["block"], r["C"]) for r in rows if r["phase"] == "C" and r["workload"] == "M2K"]
    assert sorted(m2) == [(1, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1)]
    cap, requests = load(out)
    assert replay(cap, requests) == []


# ------------------------------------------------------------------ mini smoke (030 8.6, before the freeze) --

@pytest.fixture(scope="module")
def mini_run(tmp_path_factory, cfg, pins, plan):
    tmp = tmp_path_factory.mktemp("mini")
    w = World(tmp)
    out, summary, rows = run(w, plan, cfg, pins, mode="mini", out=tmp / "repo" / "results" / "capacity" / "mini-1")
    return {"world": w, "out": out, "summary": summary, "rows": rows}


def test_mini_plan_is_a_small_replayable_plan_with_a_dummy_slo(plan):
    with rc.mini_num_prompts():
        mini = rc.build_mini_plan(plan)
        assert bp.num_prompts(25) == rc.MINI_N
    assert bp.num_prompts(25) == 250                                 # restored
    assert list(mini["workloads"]) == ["U2K", "M2K"] and mini["stage"] == "C1-mini"
    assert mini["slo"]["id"] == "MINI" and mini["secondary_slo"] == {}
    assert {w: (x["c_cap"], x["n_near"], x["bracket"]) for w, x in mini["workloads"].items()} == {
        "U2K": (6, 2, [1, 2, 4, 6]), "M2K": (2, 2, [1, 2])}
    assert mini["workloads"]["U2K"]["timeouts_s"]["6"] == rc.MINI_FORCED_TIMEOUT_S
    assert all(t > rc.MINI_FORCED_TIMEOUT_S for w, x in mini["workloads"].items() for C, t in x["timeouts_s"].items()
               if (w, C) != ("U2K", "6"))
    assert mini["sentinel"]["C"] == 2 and mini["sentinel"]["N"] == rc.MINI_N
    assert mini["mini"]["base_plan_sha256"] == sha256_json(plan)


def test_mini_plan_reaches_bisection_and_a_low_point_whatever_the_latencies(plan):
    """030 8.6: the mini smoke covers the state machine. Under the dummy SLO every valid run passes, so the forced
    timeout at U2K's cap decides the search: bracket 1, 2, 4, 6 (timeout), bisection at 5, C^ 5, K = {3 (c_low, n_low),
    4, 5, 6}; the timeout then drops 6 from block 2."""
    with rc.mini_num_prompts():
        mini = rc.build_mini_plan(plan)
    hist = []
    while (C := cp.search_next(mini["workloads"]["U2K"]["c_cap"], [(h["C"], h["pass"]) for h in hist])) is not None:
        ok = C not in rc.MINI_FORCED["U2K"]
        hist.append({"C": C, "pass": ok, "within_band": ok})
    assert [h["C"] for h in hist] == [1, 2, 4, 6, 5]
    m2 = [{"C": C, "pass": True, "within_band": True} for C in (1, 2)]
    conf = cp.build_confirm_plan(mini, {"U2K": hist, "M2K": m2})
    assert [(p["C"], p["role"], p["n"]) for p in conf["workloads"]["U2K"]["points"]] == [
        (3, "low", rc.MINI_N_LOW), (4, "near", 2), (5, "near", 2), (6, "near", 2)]
    assert [(p["C"], p["role"]) for p in conf["workloads"]["M2K"]["points"]] == [(1, "near"), (2, "near")]


def test_mini_session_exercises_the_state_machine_timeout_skip_and_replay(mini_run):
    s, rows, w = mini_run["summary"], mini_run["rows"], mini_run["world"]
    smoke = s["smoke"]
    assert s["mode"] == "mini" and s["formal"] is False and s["stage"] == "C1-mini"
    assert smoke["ok"] is True, smoke
    assert smoke["checks"]["replay"] == [] and smoke["checks"]["timeouts"] >= 2
    assert smoke["checks"]["skipped"] == [["U2K", 6]] and smoke["checks"]["unexpected_invalid"] == []
    assert smoke["checks"]["bisected"] == [["U2K", 5]] and smoke["checks"]["low_points"] == [["U2K", 3]]
    assert all(int(dict(zip(a, a[1:]))["--num-prompts"]) == rc.MINI_N for a in w.bench_argvs)
    assert bp.num_prompts(25) == 250
    u2 = [(r["phase"], r["C"], r["block"], r["valid"]) for r in rows
          if r["workload"] == "U2K" and r["phase"] != "sentinel"]
    assert [x[1] for x in u2 if x[0] == "S"] == [1, 2, 4, 6, 5]
    assert ("S", 6, None, False) in u2 and ("C", 6, 1, False) in u2 and ("C", 6, 2, False) not in u2
    assert ("C", 3, 1, True) in u2 and ("C", 3, 2, True) not in u2                 # c_low: n_low = 1
    assert json.loads((mini_run["out"] / "smoke.json").read_text()) == smoke


def test_mini_checks_need_a_bisection_step_and_a_low_point(tmp_path, cfg, pins, plan, monkeypatch):
    """A mini session that never bisects and so has no c_low (the earlier mini plan: U2K C_cap 2, the forced timeout
    at C 2, so F - P = 1 and C^ = 1) replays cleanly, times out and skips, but is not a passed smoke."""
    monkeypatch.setattr(rc, "MINI_C_CAP", {"U2K": 2, "M2K": 2})
    monkeypatch.setattr(rc, "MINI_FORCED", {"U2K": (2,)})
    _, s, _ = run(World(tmp_path), plan, cfg, pins, mode="mini")
    checks = s["smoke"]["checks"]
    assert checks["replay"] == [] and checks["timeouts"] >= 1 and checks["skipped"] == [["U2K", 2]]
    assert checks["unexpected_invalid"] == [] and checks["forced_missed"] == []
    assert checks["bisected"] == [] and checks["low_points"] == [] and s["smoke"]["ok"] is False


def test_mini_session_discards_its_latencies(mini_run):
    out, rows = mini_run["out"], mini_run["rows"]
    assert not list((out / "raw").glob("*.bench.json")) and not list((out / "raw").glob("*.bench.log"))
    assert list((out / "raw").glob("*.server.log"))
    for p in (out / "derived").iterdir():
        assert all(set(json.loads(x)) == {"idx", "cls", "out"} for x in p.read_text().splitlines())
    text = (out / "rows.jsonl").read_text() + (out / "session.json").read_text()
    for key in ("ttft", "latency", "throughput", "tpot", "metrics", "startup_s", "duration"):
        assert key not in text, key
    manifest = json.loads((out / "raw_manifest.json").read_text())["files"]
    assert not any(k.endswith((".bench.json", ".bench.log")) for k in manifest)
    s = json.loads((out / "session.json").read_text())
    assert s["rows_sha256"] == hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()     # after masking


def test_mini_restores_num_prompts_even_when_the_session_fails(tmp_path, cfg, pins, plan):
    w = World(tmp_path, mem=4000)
    with pytest.raises(rc.RunError):
        run(w, plan, cfg, pins, mode="mini")
    assert bp.num_prompts(25) == 250


def test_mini_checks_flag_an_unexpected_invalid_run(tmp_path, cfg, pins, plan):
    w = World(tmp_path, bench={"C-M2K-c1-r1": "exit"})
    _, s, _ = run(w, plan, cfg, pins, mode="mini")
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"]["unexpected_invalid"] == ["C-M2K-c1-r1"]


# ------------------------------------------------------------------ FS-order smoke (030 8.6, D3) --

@pytest.mark.parametrize("arrivals, ok, first, last", [
    (lambda C, N: {i: 0.01 * i for i in range(N)}, True, True, 0),
    (lambda C, N: {**{i: 0.01 * i for i in range(N)}, C - 1: 0.01 * C + 0.001}, False, False, 0),  # C-1 after C
    (lambda C, N: {**{i: 0.01 * i for i in range(N)}, **{N - C - k: 0.01 * N + k for k in range(1, 4)}}, True, True,
     3),
    (lambda C, N: {**{i: 0.01 * i for i in range(N)}, **{N - C - k: 0.01 * N + k for k in range(1, 5)}}, False, True,
     4),
])
def test_fs_order_check(arrivals, ok, first, last):
    got = rc.fs_order_check(arrivals(70, 700), C=70, N=700)
    assert (got["ok"], got["first_c_equal"], got["last_c_diff"]) == (ok, first, last)


def test_fs_order_check_needs_every_arrival():
    got = rc.fs_order_check({i: 0.01 * i for i in range(199)}, C=2, N=200)
    assert got["ok"] is False and got["missing"] == 1


def fs_run(tmp_path, cfg, pins, plan, **world_kw):
    w = World(tmp_path, **world_kw)
    out, s, rows = run(w, plan, cfg, pins, mode="fs-order", out=tmp_path / "repo" / "results" / "capacity" / "fs-1")
    return w, out, s, rows


def test_fs_order_boots_once_with_the_observation_flags_and_benches_c2_and_c70(tmp_path, cfg, pins, plan):
    w, out, s, rows = fs_run(tmp_path, cfg, pins, plan)
    assert w.labels == ["fs-U2K"] and s["mode"] == "fs-order" and s["formal"] is False
    argv = w.server_argvs[0]
    assert "--otlp-traces-endpoint" in argv and "--enable-logging-iteration-details" in argv
    want, _ = __import__("adv_plan").server_argv_for("B0", pins, gdn_prefill_backend="triton",
                                                     gpu_memory_utilization="0.91", port=8000,
                                                     vllm_bin=".venv/bin/vllm", base_env=BASE_ENV)
    assert argv == want
    assert [int(dict(zip(a, a[1:]))["--max-concurrency"]) for a in w.bench_argvs] == [2, 70]
    assert [int(dict(zip(a, a[1:]))["--num-prompts"]) for a in w.bench_argvs] == [200, 700]
    prefixes = [dict(zip(a, a[1:]))["--request-id-prefix"] for a in w.bench_argvs]
    assert len(set(prefixes)) == 2
    assert [w.bench_timeouts[t] for t in w.benches] == [plan["workloads"]["U2K"]["timeouts_s"][k] for k in ("2", "70")]
    smoke = s["smoke"]
    assert smoke["ok"] is True, smoke
    assert [(x["C"], x["first_c_equal"], x["last_c_diff"]) for x in smoke["benches"]] == [(2, True, 0), (70, True, 0)]
    assert json.loads((out / "smoke.json").read_text()) == smoke
    assert "receiver:start" in w.events and "receiver:close" in w.events


def test_fs_order_fails_when_a_late_request_arrives_in_the_first_wave(tmp_path, cfg, pins, plan):
    def arrivals(C, N, prefix):
        a = {i: 0.01 * i for i in range(N)}
        if C == 70:
            a[69], a[70] = a[70], a[69]
        return a

    _, _, s, _ = fs_run(tmp_path, cfg, pins, plan, arrivals=arrivals)
    assert s["smoke"]["ok"] is False
    assert [x["first_c_equal"] for x in s["smoke"]["benches"]] == [True, False]


def test_fs_order_fails_on_a_boot_problem_and_discards_latencies(tmp_path, cfg, pins, plan):
    w, out, s, _ = fs_run(tmp_path, cfg, pins, plan, boots={"fs-U2K": "blocks"})
    assert s["smoke"]["ok"] is False and w.benches == []
    w, out, s, _ = fs_run(tmp_path / "ok", cfg, pins, plan)
    raw = out / "raw"
    assert not list(raw.glob("*.bench.json")) and not list(raw.glob("*.spans.jsonl"))
    assert not list(raw.glob("*.otlp")) and not (out / "derived").exists()
    text = (out / "rows.jsonl").read_text() + (out / "session.json").read_text() + (out / "smoke.json").read_text()
    for key in ("ttft", "latency", "arrival", "throughput", "startup_s"):
        assert key not in text, key


# ------------------------------------------------------------------ CLI --

def test_main_needs_exactly_one_mode_and_the_lock_flag(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rc, "run_session", lambda plan, **kw: calls.append(kw) or {"outcomes": {}, "mode": kw["mode"],
                                                                                   "smoke": {"ok": True}})
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    base = ["--plan", str(plan), "--out", "results/capacity/x"]
    for argv in (base + ["--gpu-lock-held"], base + ["--gpu-lock-held", "--formal", "--mini"]):
        with pytest.raises(SystemExit):
            rc.main(argv)
    assert calls == []
    assert rc.main(base + ["--gpu-lock-held", "--mini", "--mixed-dataset", "2048=a.jsonl"]) == 0
    kw = calls[-1]
    assert kw["mode"] == "mini" and kw["gpu_lock_held"] is True and kw["mixed_datasets"] == {2048: "a.jsonl"}
    assert kw["lock_held_check"] is ra.held_gpu_lock and kw["popen"] is ra.start_server
    assert kw["run_bench"] is ra.default_run_bench and kw["gpu_mem_used"] is ra.nvidia_mem_used_bounded
    assert kw["compute_apps"] is rc.compute_apps_bounded and kw["sample_interval"] == 1.0
    assert kw["capacity_sha256"] == hashlib.sha256((ROOT / "configs" / "capacity.toml").read_bytes()).hexdigest()
    assert kw["pins_sha256"] == hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    rc.main(base + ["--gpu-lock-held", "--smoke-fs-order"])
    assert calls[-1]["mode"] == "fs-order"
    rc.main(base + ["--gpu-lock-held", "--formal"])
    assert calls[-1]["mode"] == "formal"


def test_main_returns_2_on_a_refusal(tmp_path, monkeypatch, capsys):
    def refuse(plan, **kw):
        raise rc.RunError("nope")

    monkeypatch.setattr(rc, "run_session", refuse)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    assert rc.main(["--plan", str(plan), "--out", "results/capacity/x", "--gpu-lock-held", "--formal"]) == 2
    assert "nope" in capsys.readouterr().err


def test_main_returns_2_when_run_adv_stops_the_session(tmp_path, monkeypatch, capsys):
    def receiver_failed(plan, **kw):
        raise ra.ReceiverError("the OTLP receiver could not store spans")

    monkeypatch.setattr(rc, "run_session", receiver_failed)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    assert rc.main(["--plan", str(plan), "--out", "results/capacity/x", "--gpu-lock-held", "--smoke-fs-order"]) == 2
    assert "receiver" in capsys.readouterr().err


def test_compute_apps_bounded_lists_processes_and_bounds_nvidia_smi(monkeypatch):
    calls = []

    class Done:
        def __init__(self, out):
            self.stdout = out

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return Done(outs.pop(0))

    outs = ["No running processes found\n", "123, python, 9000\n\n456, vllm, 100\n"]
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert rc.compute_apps_bounded() == []
    assert rc.compute_apps_bounded(timeout_s=3) == ["123, python, 9000", "456, vllm, 100"]
    assert calls[0][0][1] == "--query-compute-apps=pid,process_name,used_memory"
    assert calls[0][1]["timeout"] == ra.NVIDIA_SMI_TIMEOUT_S and calls[1][1]["timeout"] == 3
    assert calls[0][1]["check"] is True


def test_a_formal_session_needs_the_freeze_evidence(tmp_path, cfg, pins):
    """030 implementation note: no formal session without the passed FS-order / mini smokes and C0's result log."""
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    world = World(tmp_path)
    with pytest.raises(rc.RunError, match="freeze evidence"):
        run(world, plan, c, p, mixed_datasets=ds, evidence=False)
    assert not (tmp_path / "repo" / "results" / "capacity" / "s1").exists()
