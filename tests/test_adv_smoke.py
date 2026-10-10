"""Tests for benchmarks/adv_smoke.py: the decision 024 feasibility smokes FS1-FS5 (023: they only check that the
Advanced harness can run, never at a formal cell, masked: pass / fail and counts only).

Every boot goes through the runner (run_adv) as a one-boot, non-formal session; here a fake runner stands in for it
and follows the shared row / plan contract. Nothing touches a GPU, a server or the network.
"""

import copy
import gzip
import hashlib
import json
import re
import time
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import adv_smoke as sm
import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent
GIB = 2**30
STEP = 256 * 2**20
SHORT, LONG = 2048 - 128, bp.LONG_INPUT
LATENCY = re.compile(r"(?i)(ttft|tpot|itl|e2e|latency|_ms$|duration|throughput|elapsed|arrival|queue_time|^[QPOF]$)")


@pytest.fixture(scope="module")
def pins():
    return tomllib.loads((ROOT / "configs" / "pins.toml").read_text())


# ------------------------------------------------------------------ datasets --

def dataset_lines(n_blocks=6, long_pos=(4, 0, 7, 3, 5, 9), long_chars=(164000, 87000, 169000, 80000, 90000, 99000)):
    """Blocks of ten like the pinned mixed file: 9 short rows (~11k chars) + 1 long row (>= 79k chars)."""
    lines = []
    for b in range(n_blocks):
        for i in range(10):
            n = long_chars[b] if i == long_pos[b] else 9000 + 300 * i
            lines.append(json.dumps({"prompt": "x" * n, "output_tokens": 128}))
    return lines


def write_dataset(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_pins(pins, sha):
    p = copy.deepcopy(pins)
    p["core_b"]["mix_dataset_sha256"]["2048"] = sha
    return p


def test_long_rows_are_the_longest_row_of_each_block():
    assert sm.long_row_indices(dataset_lines(), 4) == [4, 10, 27, 33]


def test_long_row_detection_refuses_an_ambiguous_block():
    lines = dataset_lines()
    lines[12] = json.dumps({"prompt": "x" * 80000, "output_tokens": 128})  # second long-looking row in block 1
    with pytest.raises(sm.SmokeError, match="block 1"):
        sm.long_row_indices(lines, 2)


def test_prepare_datasets_writes_verbatim_subsets_under_raw(tmp_path, pins):
    lines = dataset_lines()
    src = tmp_path / "mixed-2048.jsonl"
    p = pinned_pins(pins, write_dataset(src, lines))
    ds = sm.prepare_datasets(src, tmp_path / "out", p)
    assert Path(ds["row4"]).parent == tmp_path / "out" / "raw" / "datasets"  # results/**/raw/ never enters git
    assert Path(ds["row4"]).read_text().splitlines() == [lines[4]]
    assert Path(ds["long4"]).read_text().splitlines() == [lines[i] for i in (4, 10, 27, 33)]
    assert ds["pinned"] == str(src)


def test_prepare_datasets_refuses_an_unpinned_file(tmp_path, pins):
    src = tmp_path / "mixed-2048.jsonl"
    write_dataset(src, dataset_lines())
    with pytest.raises(sm.SmokeError, match="sha256"):
        sm.prepare_datasets(src, tmp_path / "out", pins)


def test_prepare_datasets_requires_row_4_to_be_long(tmp_path, pins):
    lines = dataset_lines(long_pos=(3, 0, 7, 3, 5, 9))
    src = tmp_path / "mixed-2048.jsonl"
    p = pinned_pins(pins, write_dataset(src, lines))
    with pytest.raises(sm.SmokeError, match="index 4"):
        sm.prepare_datasets(src, tmp_path / "out", p)


# --------------------------------------------------------------------- specs --

DS = {"row4": "/d/row4.jsonl", "long4": "/d/long4.jsonl", "pinned": "/d/mixed-2048.jsonl"}


def by_name(specs):
    return {s["name"]: s for s in specs}


def test_fs1_boots_every_condition_once_with_a_tiny_bench(pins):
    s = sm.probe_specs("FS1", pins, DS)
    assert [x["cond"] for x in s] == list(ap.CONDITIONS)
    assert all((x["L"], x["mix"], x["C"], x["N"], x["warmup"]) == (2048, "uniform", 1, 1, 0) for x in s)
    assert all(x["expect_lens"] == [SHORT] for x in s)


def test_fs2_is_row_4_alone_at_c1_under_b0_and_t(pins):
    s = by_name(sm.probe_specs("FS2", pins, DS))
    assert set(s) == {"fs2-B0", "fs2-T"}
    for x in s.values():
        assert (x["L"], x["mix"], x["C"], x["N"], x["warmup"], x["dataset"]) == (2048, "mixed", 1, 1, 0, DS["row4"])
        assert x["expect_lens"] == [LONG]


def test_fs3_is_four_long_rows_at_c4_under_g_and_b0_with_a_5_minute_cap(pins):
    s = by_name(sm.probe_specs("FS3", pins, DS))
    assert set(s) == {"fs3-G", "fs3-B0"}
    for x in s.values():
        assert (x["mix"], x["C"], x["N"], x["dataset"], x["timeout_s"]) == ("mixed", 4, 4, DS["long4"], 300)
        assert x["expect_lens"] == [LONG] * 4
    assert 4 * (66 + 0) > 247  # 024: four long requests need 264 > 247 blocks


def test_fs4_runs_b0_uniform_and_mixed_and_tg_mixed_at_c2_n20(pins):
    s = by_name(sm.probe_specs("FS4", pins, DS))
    assert set(s) == {"fs4-B0-uniform", "fs4-B0-mixed", "fs4-TG-mixed"}
    assert all((x["C"], x["N"], x["warmup"]) == (2, 20, 2) for x in s.values())
    assert s["fs4-B0-mixed"]["dataset"] == DS["pinned"]  # the first 20 rows of the pinned file (--disable-shuffle)
    assert sorted(s["fs4-B0-mixed"]["expect_lens"]) == sorted([SHORT] * 18 + [LONG] * 2)
    assert s["fs4-B0-uniform"]["timeout_s"] == pins["advanced"]["timeouts_s"]["uniform_2048"]
    assert s["fs4-TG-mixed"]["timeout_s"] == pins["advanced"]["timeouts_s"]["mixed_2048"]


def test_fs5_is_b0_uniform_with_a_5_second_timeout(pins):
    (x,) = sm.probe_specs("FS5", pins, DS)
    assert (x["cond"], x["L"], x["mix"], x["C"], x["N"], x["timeout_s"]) == ("B0", 2048, "uniform", 2, 20, 5)


def test_no_smoke_runs_at_a_formal_cell(pins):
    for fs in sm.FS_NAMES:
        for x in sm.probe_specs(fs, pins, DS):
            formal_c = int(pins["advanced"]["c"][str(x["L"])])
            assert not (x["C"] == formal_c and x["N"] == bp.num_prompts(formal_c)), x["name"]


# ------------------------------------------------------------- plan and argv --

def test_smoke_plan_is_a_one_boot_contract_plan(pins):
    spec = by_name(sm.probe_specs("FS4", pins, DS))["fs4-B0-mixed"]
    plan = sm.smoke_plan(spec, pins, pins_sha256="abc")
    (boot,) = plan["boots"]
    # run_adv requires Core's N / warmup at the boot's C; the bench really runs smoke_counts (with_counts)
    assert boot == {"block": 1, "cond": "B0", "L": 2048, "mix": "mixed", "restart": 1, "C": 2, "N": 200, "warmup": 2,
                    "timeout_s": 600, "prefix": ap.request_id_prefix("B0", 2048, "mixed", 1)}
    assert plan["smoke_counts"] == {"N": 20, "warmup": 2}
    assert plan["c"] == {"2048": 2} and plan["timeouts_s"]["mixed_2048"] == 600
    assert plan["smoke"] == "FS4" and plan["pins_sha256"] == "abc"
    assert plan["conditions"] == list(ap.CONDITIONS)
    assert plan["expected_blocks"]["B0"] == 248
    assert plan["block_size"] == 528 and plan["kv_cache_memory_bytes"] == 4 * GIB
    fs5 = sm.smoke_plan(sm.probe_specs("FS5", pins, DS)[0], pins, pins_sha256="abc")
    assert fs5["boots"][0]["timeout_s"] == fs5["timeouts_s"]["uniform_2048"] == 5


def test_every_smoke_plan_passes_the_runners_plan_check(pins):
    import run_adv as ra

    for fs in sm.FS_NAMES:
        for spec in sm.probe_specs(fs, pins, DS):
            plan = sm.smoke_plan(spec, pins, pins_sha256="abc")
            assert ra.plan_problems(plan, pins, formal=False) == [], spec["name"]


def test_k3_expected_blocks_before_and_after_measurement(pins):
    pins = copy.deepcopy(pins)
    pins["advanced"]["num_gpu_blocks_k3"] = 0  # the draft state, whatever the repo pins are now
    assert sm.expected_blocks(pins)["K3"] == 186  # 3 GiB bracketed exactly by 248 blocks at 4 GiB
    measured = copy.deepcopy(pins)
    measured["advanced"]["num_gpu_blocks_k3"] = 185
    assert sm.expected_blocks(measured)["K3"] == 185
    assert sm.expected_blocks(measured, k3_bytes=3 * GIB + STEP)["K3"] == 201  # an override is predicted again
    assert sm.pins_for_smoke(pins, 3 * GIB + STEP)["advanced"]["kv_low_bytes"] == 3 * GIB + STEP
    assert pins["advanced"]["kv_low_bytes"] == 3 * GIB  # the caller's pins are never changed
    frozen = copy.deepcopy(pins)
    frozen["advanced"]["num_gpu_blocks_k3"] = 186
    stepped = sm.pins_for_smoke(frozen, 3 * GIB + STEP)  # a count measured at 3 GiB must not survive a step-up
    assert stepped["advanced"]["num_gpu_blocks_k3"] == 0 and sm.expected_blocks(stepped)["K3"] == 201
    assert sm.pins_for_smoke(frozen, 3 * GIB)["advanced"]["num_gpu_blocks_k3"] == 186  # same bytes: kept


def test_k3_override_must_be_a_256_mib_multiple(pins):
    with pytest.raises(sm.SmokeError):
        sm.pins_for_smoke(pins, 3 * GIB + 1)


def bench(pins, **kw):
    a = dict(cond="B0", L=2048, C=25, mix="uniform", restart=1, result_dir="/r", result_filename="b.json", port=8000)
    a.update(kw)
    return ap.bench_argv_for(pins, **a)


def test_with_counts_only_rewrites_n_c_and_warmup(pins):
    argv = bench(pins, C=2)
    got = sm.with_counts(argv, C=2, N=20, warmup=2)
    pairs = dict(zip(got, got[1:]))
    assert (pairs["--num-prompts"], pairs["--max-concurrency"], pairs["--num-warmups"]) == ("20", "2", "2")
    diff = [i for i, (a, b) in enumerate(zip(argv, got)) if a != b]
    assert len(got) == len(argv) and [argv[i - 1] for i in diff] == ["--num-prompts"]
    with pytest.raises(sm.SmokeError):
        sm.with_counts(["vllm", "bench", "serve"], C=1, N=1, warmup=0)


def test_flag_value_and_result_path(pins):
    argv = bench(pins, result_dir="/r/raw", result_filename="x.json")
    assert sm.flag_value(argv, "--request-id-prefix") == "adv-B0-2048-uniform-r1-"
    assert sm.result_path(argv) == Path("/r/raw/x.json")


# --------------------------------------------------------------------- K3 rule --

def test_k3_report_passes_at_186_blocks_and_proves_2_75_gib_fails(pins):
    r = sm.k3_report(186, kv_bytes=3 * GIB, pins=pins)
    assert r["rule_ok"] is True and r["problems"] == []
    assert r["c_kv"] == {"2048": 26, "8192": 9, "32768": bp.c_kv(186, 528, 32768)}
    assert (r["lower_bytes"], r["lower_blocks_max"], r["lower_fails"]) == (int(2.75 * GIB), 171, True)
    assert r["freeze_ready"] is True and r["next_kv_low_bytes"] is None


def test_k3_report_fails_at_170_blocks_and_names_the_next_step(pins):
    r = sm.k3_report(170, kv_bytes=3 * GIB, pins=pins)
    assert r["rule_ok"] is False and any("C_KV(2048) = 24 < C_knee 25" in p for p in r["problems"])
    assert r["next_kv_low_bytes"] == 3 * GIB + STEP and r["freeze_ready"] is False


def test_k3_report_without_a_measurement(pins):
    r = sm.k3_report(None, kv_bytes=3 * GIB, pins=pins)
    assert r["rule_ok"] is False and r["freeze_ready"] is False
    assert r["problems"] == ["num_gpu_blocks not read from /metrics"]


# ------------------------------------------------------------- iteration log --

def it_line(i, ctx_req, ctx_tok, gen_req=0, gen_tok=0, stamp=True, dummy=False, t0=1760100000000):
    s = (f"(EngineCore pid=1) INFO 10-10 22:00:00 [loggers.py:185] Iteration({i}): {ctx_req} context requests, "
         f"{ctx_tok} context tokens, {gen_req} generation requests, {gen_tok} generation tokens, "
         f"iteration elapsed time: 12.34 ms{' (dummy)' if dummy else ''}, GPU KV cache usage: 26.4%")
    return f"{t0 + 50 * i} {s}" if stamp else s


def long_prefill(t, first=1):
    """One 32640-token prompt at C = 1 with threshold t (0 = the 2048 budget), then 3 decode steps."""
    chunk = t or 2048
    lines, left, i = [], LONG, first
    while left > 0:
        lines.append(it_line(i, 1, min(chunk, left)))
        left -= min(chunk, left)
        i += 1
    lines += [it_line(i + k, 0, 0, 1, 1) for k in range(3)]
    return lines


def test_step_counts_b0_has_full_budget_steps():
    c = sm.step_counts(["noise", it_line(0, 0, 0, dummy=True)] + long_prefill(0))
    assert c["n_iteration_lines"] == c["n_steps"] == 16 + 3
    assert c["epoch_ms_stamps"] is True
    assert c["n_ctx_steps"] == 16 and c["n_ctx_budget_single"] == 15 and c["max_step_tokens"] == 2048
    assert c["n_ctx_over_threshold"] == 16 and c["n_chunk_violations"] == 16  # 15 x 2048 + 1920


def test_step_counts_t_keeps_every_chunk_at_512():
    c = sm.step_counts(long_prefill(512))
    assert c["n_ctx_steps"] == 64 and c["n_ctx_over_threshold"] == 0 and c["n_chunk_violations"] == 0
    assert c["n_ctx_budget_single"] == 0 and c["max_step_tokens"] == 512


def test_step_counts_sees_lines_without_a_runner_timestamp():
    c = sm.step_counts([it_line(1, 1, 2048, stamp=False), it_line(2, 1, 2048)])
    assert (c["n_iteration_lines"], c["n_steps"]) == (2, 1)
    assert sm.step_counts([])["max_step_tokens"] is None


# ------------------------------------------------------------- bench and mask --

def bench_json(lens, outs=None, failed=0):
    outs = outs if outs is not None else [128] * len(lens)
    return {"completed": len(lens) - failed, "failed": failed, "input_lens": lens, "output_lens": outs,
            "ttfts": [0.1] * len(lens), "mean_ttft_ms": 100.0, "duration": 9.0}


def test_count_bench_result_counts_without_values(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(bench_json([LONG, SHORT, LONG], outs=[128, 128, 127])))
    c = sm.count_bench_result(p, [LONG, LONG, LONG])
    assert c == {"exists": True, "n": 3, "completed": 3, "failed": 0, "n_input_ok": 2, "inputs_ok": False,
                 "n_output_ok": 2, "outputs_ok": False}
    assert sm.count_bench_result(tmp_path / "none.json", [LONG]) == {"exists": False}


def test_mask_row_keeps_only_counts_and_flags():
    row = {"cond": "B0", "L": 2048, "mix": "uniform", "restart": 1, "block": 1, "C": 2, "N": 20, "ok": False,
           "reasons": ["completed=20 failed=0 (expected 200 / 0)"], "timeout": False, "skipped": False,
           "completed": 20, "failed": 0, "num_prompts": 200, "p95_ttft_ms": 1.0, "short_class_p95_ttft_ms": 2.0,
           "long_class_p95_ttft_ms": 3.0, "output_throughput": 4.0, "duration": 5.0, "preemptions": 0, "kv_peak": 0.9,
           "manip": {"facts_ok": True, "blocks_ok": True, "chunk_ok": None, "num_gpu_blocks": 248},
           "decomp": {"valid": True, "checks": {"a": True, "b": True, "c": True, "d": True},
                      "tail": {"O": 1.0, "Q": 2.0, "P": 3.0}, "reasons": ["(b) |delta sum 0.1 - sum Q 0.1| > 1 ms"],
                      "q_shares": {"no-queue": 1.0}, "residual_bm_share": 0.0}}
    m = sm.mask_row(row)
    assert m["decomp"] == {"valid": True, "checks": {"a": True, "b": True, "c": True, "d": True}}
    assert m["manip"] == row["manip"] and m["kv_peak_present"] is True and m["completed"] == 20
    assert sm.latency_keys(m) == []
    assert sm.latency_keys(row)  # the scanner does see the unmasked fields
    assert sm.mask_row(None) is None


def test_redact_bench_log_cuts_the_result_section_and_progress():
    text = ("Starting initial single prompt test run...\nInitial test run completed.\n"
            " 45%|####      | 9/20 [00:03<00:04,  2.51it/s]\r100%|##########| 20/20 [00:07<00:00,  2.71it/s]\n"
            "============ Serving Benchmark Result ============\nMean TTFT (ms): 123.4\n")
    out = sm.redact_bench_log(text)
    assert "Serving Benchmark Result" not in out and "123.4" not in out and "it/s" not in out
    assert "Initial test run completed." in out and "[result section removed]" in out


# ---------------------------------------------------------------- the fake runner --

METRICS = ('# HELP x\nvllm:cache_config_info{{block_size="528",engine="0",num_gpu_blocks="{nb}"}} 1.0\n'
           'vllm:num_preemptions_total{{engine="0"}} {pre}\n'
           'vllm:time_to_first_token_seconds_sum{{engine="0"}} 1.5\n')


class World:
    """Fakes for what the smoke itself touches: /health, /metrics, nvidia-smi, /proc, sockets, the clock."""

    def __init__(self, nb=248, preempt_per_scrape=0, health_ok_calls=None, residual=(), ports_open=(), mem=(400,)):
        self.nb, self.preempt_per_scrape, self.health_ok_calls = nb, preempt_per_scrape, health_ok_calls
        self.residual = None if residual is None else list(residual)
        self.ports_open, self.mem = set(ports_open), list(mem)
        self.calls, self.scrapes, self.t = [], 0, 0.0

    def http_get(self, url, timeout=5):
        self.calls.append(url)
        if url.endswith("/health"):
            ok = self.health_ok_calls is None or sum(u.endswith("/health") for u in self.calls) <= self.health_ok_calls
            return (200, "") if ok else (0, "refused")
        self.scrapes += 1
        return 200, METRICS.format(nb=self.nb, pre=float(self.preempt_per_scrape * self.scrapes))

    def gpu_mem_used(self):
        return self.mem.pop(0) if len(self.mem) > 1 else self.mem[0]

    def procs_with(self, token):
        return None if self.residual is None else list(self.residual)

    def port_open(self, host, port):
        return port in self.ports_open

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeRunAdv:
    """Stands in for run_adv.run_session: one row per boot per the shared contract, bench JSON in raw/, spans and
    Prometheus scrapes in raw/, derived/<tag>-requests.jsonl with latency columns, a timestamped server log."""

    def __init__(self, world, *, steps=None, timeout=False, raise_after_bench=False, facts_ok=True, decomp_checks=None,
                 drop_fields=(), teardown_problems=(), scrape_edit=None, one_hz=None, series_samples=5):
        self.world, self.steps, self.timeout, self.raise_after_bench = world, steps, timeout, raise_after_bench
        self.facts_ok, self.drop_fields, self.teardown_problems = facts_ok, drop_fields, list(teardown_problems)
        self.decomp_checks = decomp_checks or {"a": True, "b": True, "c": True, "d": True}
        self.scrape_edit, self.series_samples = scrape_edit, series_samples
        self.one_hz = one_hz if one_hz is not None else {"kv_full_while_waiting": 0.0, "gate_zone": 0.0,
                                                         "n_samples": series_samples}
        self.plans, self.argvs = [], []

    def scrape_delta(self):
        """As run_adv.scrape_deltas: every histogram / counter of the two scrapes, errors empty."""
        import run_adv as ra

        d = {h: {"buckets": {"+Inf": 20.0}, "sum": 1.0, "count": 20.0, "mean": 0.05} for h in ra.HISTOGRAMS}
        d.update({c: 0.0 for c in ra.COUNTERS}, errors=[])
        if self.scrape_edit:
            self.scrape_edit(d)
        return d

    def run_bench(self, argv, log_path, env, timeout_s):
        self.argvs.append(argv)
        Path(log_path).write_text("Initial test run completed.\n 100%|##| 2/2 [00:01<00:00, 1.9it/s]\n"
                                  "============ Serving Benchmark Result ============\nMean TTFT (ms): 77.7\n")
        if self.timeout or timeout_s < 10:
            return {"rc": None, "timeout": True}
        n = int(sm.flag_value(argv, "--num-prompts"))
        if "--dataset-path" in argv:
            rows = Path(sm.flag_value(argv, "--dataset-path")).read_text().splitlines()[:n]
            lens = [LONG if len(json.loads(r)["prompt"]) > 50000 else SHORT for r in rows]
        else:
            lens = [int(sm.flag_value(argv, "--random-input-len"))] * n
        sm.result_path(argv).write_text(json.dumps(bench_json(lens)))
        return {"rc": 0, "timeout": False}

    def __call__(self, plan, *, pins, out_dir, session_id, mixed_datasets, wrap_http_get, wrap_run_bench):
        self.plans.append((plan, pins, mixed_datasets))
        http_get, run_bench = wrap_http_get(self.world.http_get), wrap_run_bench(self.run_bench)
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "raw").mkdir()
        (out / "derived").mkdir()
        (boot,) = plan["boots"]
        cond, L, mix = boot["cond"], boot["L"], boot["mix"]
        tag = f"{cond}-{L}-{mix}-r{boot['restart']}"
        assert http_get("http://127.0.0.1:8000/health")[0] == 200
        before = http_get("http://127.0.0.1:8000/metrics")[1]
        (out / "raw" / f"scrape-{tag}-before.prom").write_text(before)
        steps = self.steps if self.steps is not None else long_prefill(0)
        (out / "raw" / f"{tag}.server.log").write_text("\n".join(["1760100000000 INFO starting"] + steps) + "\n")
        argv = ap.bench_argv_for(pins, cond=cond, L=L, C=boot["C"], mix=mix, restart=boot["restart"],
                                 result_dir=str(out / "raw"), result_filename=f"bench-{tag}.json", port=8000,
                                 mixed_dataset=mixed_datasets.get(L) if mix == "mixed" else None)
        res = run_bench(argv, out / "raw" / f"bench-{tag}.log", {}, boot["timeout_s"])
        if self.raise_after_bench:
            raise RuntimeError("receiver thread died")
        (out / "raw" / f"{tag}.series.jsonl").write_text("".join(
            json.dumps({"t": 1.7e9 + i, "vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_waiting": 0.0}) + "\n"
            for i in range(self.series_samples)))
        (out / "raw" / f"spans-{tag}.jsonl.gz").write_bytes(
            gzip.compress(json.dumps({"attributes": {"gen_ai.latency.time_in_queue": 0.01}}).encode()))
        http_get("http://127.0.0.1:8000/metrics")
        row = {"cond": cond, "L": L, "mix": mix, "restart": boot["restart"], "block": boot["block"], "C": boot["C"],
               "N": boot["N"], "skipped": False, "timeout": res["timeout"]}
        bj = out / "raw" / f"bench-{tag}.json"
        if res["timeout"]:
            row.update(ok=False, reasons=["timeout"], manip=self.manip(cond), decomp=None, preemptions=None,
                       kv_peak=None, long_class_p95_ttft_ms=None)
        else:
            from run_core_b import summarize_result

            data = json.loads(bj.read_text())
            row.update(summarize_result(data, L=L, C=boot["C"], mix=mix))
            row.update(long_class_p95_ttft_ms=1.0, preemptions=self.world.preempt_per_scrape * 2 and 3, kv_peak=0.99,
                       manip=self.manip(cond), scrape_delta=self.scrape_delta(), one_hz=dict(self.one_hz),
                       series={"kv_peak": 0.99, "waiting_peak": 0.0, "running_peak": 2.0,
                               "n_samples": self.series_samples},
                       decomp={"valid": all(v is not False for v in self.decomp_checks.values()),
                               "checks": dict(self.decomp_checks), "tail": {"O": 1.0, "Q": 2.0, "P": 3.0},
                               "q_shares": {k: 0.0 for k in ("no-queue", "budget-long")},
                               "q_shares_sens": {"no-queue": 1.0}, "residual_bm_share": 0.0,
                               "unexplained_break_share": 0.0, "n_steps": 9, "reasons": ["(b) 0.1 s"]})
            with open(out / "derived" / f"{tag}-requests.jsonl", "w") as fh:
                for i in range(len(data["input_lens"])):
                    fh.write(json.dumps({"idx": i, "cls": "short", "client_ttft_ms": 1.0, "arrival": 1.7e9,
                                         "Q_ms": 1.0, "P_ms": 1.0, "prompt_tokens": data["input_lens"][i],
                                         "completion_tokens": 128}) + "\n")
        for k in self.drop_fields:
            row.pop(k, None)
        (out / "rows.jsonl").write_text(json.dumps(row) + "\n")
        session = {"session_id": session_id, "formal": False, "teardown_problems": self.teardown_problems,
                   "runs": {tag: {"argv": ["vllm", "serve"], "startup_s": 70.2, "t_before": 1000.0,
                                  "t_after": 1012.5, "log_facts": {"attention_backend": "FLASH_ATTN"}}}}
        (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
        return session

    def manip(self, cond):
        return {"facts_ok": self.facts_ok, "blocks_ok": True, "chunk_ok": True if cond in ("T", "TG") else None,
                "num_gpu_blocks": self.world.nb}


def probe(tmp_path, spec, world, runner, pins):
    return sm.run_probe(spec, pins=pins, out=tmp_path, session_runner=runner, http_get=world.http_get,
                        gpu_mem_used=world.gpu_mem_used, procs_with=world.procs_with, port_open=world.port_open,
                        now=world.now, sleep=world.sleep, pins_sha256="abc")


def mixed_ds(tmp_path, pins):
    src = tmp_path / "mixed-2048.jsonl"
    p = pinned_pins(pins, write_dataset(src, dataset_lines()))
    return sm.prepare_datasets(src, tmp_path / "smoke", p), p


def test_probe_rewrites_n_counts_then_masks(tmp_path, pins):
    ds, p = mixed_ds(tmp_path, pins)
    spec = by_name(sm.probe_specs("FS4", p, ds))["fs4-B0-mixed"]
    world = World()
    runner = FakeRunAdv(world)
    rep = probe(tmp_path / "smoke", spec, world, runner, p)
    argv = runner.argvs[0]
    assert sm.flag_value(argv, "--num-prompts") == "20" and sm.flag_value(argv, "--num-warmups") == "2"
    assert rep["bench"] == {"exists": True, "n": 20, "completed": 20, "failed": 0, "n_input_ok": 20,
                            "inputs_ok": True, "n_output_ok": 20, "outputs_ok": True}
    assert rep["health_after_bench"] is True and rep["residual_client_procs"] == 0
    assert rep["cache_config"] == {"num_gpu_blocks": 248, "block_size": 528}
    assert rep["field_problems"] == [] and rep["derived"] == {"files": 1, "rows": 20}
    assert rep["steps"]["n_steps"] == 19 and rep["post"] == {"server_port_open": False, "receiver_port_open": False,
                                                             "gpu_idle_after": True}
    d = tmp_path / "smoke" / "fs4-B0-mixed"
    assert not list(d.rglob("bench-*.json")) and not list(d.rglob("spans-*"))  # read, counted, deleted
    assert not list(d.rglob("*.prom"))  # Prometheus text carries TTFT / queue histograms
    assert "77.7" not in (d / "raw" / "bench-B0-2048-mixed-r1.log").read_text()
    row = json.loads((d / "rows.jsonl").read_text())
    assert sm.latency_keys(row) == [] and row["completed"] == 20
    for line in (d / "derived" / "B0-2048-mixed-r1-requests.jsonl").read_text().splitlines():
        assert set(json.loads(line)) == {"idx", "cls", "prompt_tokens", "completion_tokens"}
    assert rep["masked"]["unmasked_fields"] == [] and sm.latency_keys(rep) == []
    assert (d / "raw" / "B0-2048-mixed-r1.server.log").exists()  # raw/ stays local; only token counts are read


def test_probe_strips_the_bench_window_timings_from_session_json(tmp_path, pins):
    """session.json is committed: its runs lose startup_s and the t_before / t_after wall times that bracket the
    bench (t_after - t_before ~ its duration); the scan for unmasked fields covers session.json too."""
    spec = sm.probe_specs("FS1", pins, DS)[0]
    world = World()
    rep = probe(tmp_path, spec, world, FakeRunAdv(world, teardown_problems=["server had to be killed"]), pins)
    d = tmp_path / spec["name"]
    meta = json.loads((d / "session.json").read_text())
    (run,) = meta["runs"].values()
    assert set(run) == {"argv", "log_facts"} and run["log_facts"] == {"attention_backend": "FLASH_ATTN"}
    assert meta["teardown_problems"] == ["server had to be killed"] and meta["session_id"]
    assert rep["teardown_problems"] == ["server had to be killed"]  # read before masking
    assert "session.json" in rep["masked"]["rewritten"] and rep["masked"]["unmasked_fields"] == []
    assert sm.latency_keys(meta) == []


def test_mask_session_flags_timings_it_finds_in_session_json(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "session.json").write_text(json.dumps({"runs": {"x": {"startup_s": 1.0, "t_before": 2.0, "t_after": 3.0,
                                                               "duration": 4.0, "argv": []}}}, indent=1))
    masked = sm.mask_session(d, [])
    assert json.loads((d / "session.json").read_text()) == {"runs": {"x": {"argv": []}}}
    assert masked["rewritten"] == ["session.json"] and masked["unmasked_fields"] == []
    assert sm.latency_keys({"runs": {"x": {"t_before": 1, "t_after": 2, "startup_s": 3}}}) == [
        "runs.x.t_before", "runs.x.t_after", "runs.x.startup_s"]


def test_probe_records_a_runner_exception_and_still_masks(tmp_path, pins):
    spec = sm.probe_specs("FS1", pins, DS)[0]
    world = World()
    rep = probe(tmp_path, spec, world, FakeRunAdv(world, raise_after_bench=True), pins)
    assert any("harness error: RuntimeError: receiver thread died" in r for r in rep["reasons"])
    assert rep["bench"]["exists"] is True and not list((tmp_path / spec["name"]).rglob("bench-*.json"))


def test_probe_refuses_to_start_on_a_busy_gpu(tmp_path, pins):
    world = World(mem=(5000,))
    runner = FakeRunAdv(world)
    with pytest.raises(sm.SmokeError, match="GPU busy"):
        probe(tmp_path, sm.probe_specs("FS1", pins, DS)[0], world, runner, pins)
    assert runner.plans == []


def test_probe_waits_for_the_gpu_to_be_released(tmp_path, pins):
    world = World(mem=(400, 9000, 9000, 9000))  # before, then never back to idle after teardown
    rep = probe(tmp_path, sm.probe_specs("FS1", pins, DS)[0], world, FakeRunAdv(world), pins)
    assert rep["post"]["gpu_idle_after"] is False


def test_probe_passes_the_k3_override_and_dataset_to_the_runner(tmp_path, pins):
    spec = dict(sm.probe_specs("FS1", pins, DS)[-1])
    world = World(nb=201)
    runner = FakeRunAdv(world)
    sm.run_probe(spec, pins=pins, out=tmp_path, session_runner=runner, http_get=world.http_get,
                 gpu_mem_used=world.gpu_mem_used, procs_with=world.procs_with, port_open=world.port_open,
                 now=world.now, sleep=world.sleep, pins_sha256="abc", k3_bytes=3 * GIB + STEP)
    plan, run_pins, _ = runner.plans[0]
    assert run_pins["advanced"]["kv_low_bytes"] == 3 * GIB + STEP and plan["expected_blocks"]["K3"] == 201


# ------------------------------------------------------------------ FS verdicts --

def run_fs(tmp_path, fs, pins, world, runner, ds=DS):
    return [probe(tmp_path, s, world, runner, pins) for s in sm.probe_specs(fs, pins, ds)]


def test_fs1_reads_blocks_from_metrics_per_condition_and_reports_k3(tmp_path, pins):
    world = World(nb=186)  # every boot reads 186: right for K3 (3 GiB), wrong for the 4 GiB conditions
    reps = run_fs(tmp_path, "FS1", pins, world, FakeRunAdv(world))
    v = sm.evaluate("FS1", reps, pins=pins)
    assert v["k3"]["num_gpu_blocks"] == 186 and v["k3"]["rule_ok"] is True and v["k3"]["kv_low_bytes"] == 3 * GIB
    assert v["ok"] is False and v["reasons"]
    assert all(r.split(":")[0] in ("B0", "T", "G", "TG") and "num_gpu_blocks 186 != expected 248" in r
               for r in v["reasons"])


def fs1_reps(tmp_path, pins, **kw):
    world = World()
    reps = run_fs(tmp_path, "FS1", pins, world, FakeRunAdv(world, **kw))
    for r in reps:
        if r["cond"] == "K3":
            r["cache_config"] = {"num_gpu_blocks": 186, "block_size": 528}
    return reps


def test_fs1_ok_when_every_boot_checks_out(tmp_path, pins):
    v = sm.evaluate("FS1", fs1_reps(tmp_path, pins), pins=pins)
    assert v["ok"] is True and v["reasons"] == []


@pytest.mark.parametrize("kw,match", [
    ({"facts_ok": False}, "log facts"),
    ({"decomp_checks": {"a": False, "b": True, "c": True, "d": True}}, "spans"),
    ({"steps": [it_line(1, 1, 2048, stamp=False)]}, "timestamp"),
    ({"steps": []}, "iteration"),
])
def test_fs1_fails_on_each_boot_check(tmp_path, pins, kw, match):
    v = sm.evaluate("FS1", fs1_reps(tmp_path, pins, **kw), pins=pins)
    assert v["ok"] is False and any(match in r for r in v["reasons"])


def test_fs1_fails_when_t_misses_the_chunk_check(tmp_path, pins):
    reps = fs1_reps(tmp_path, pins)
    next(r for r in reps if r["cond"] == "T")["row"]["manip"]["chunk_ok"] = False
    v = sm.evaluate("FS1", reps, pins=pins)
    assert v["ok"] is False and any("T: chunk" in r for r in v["reasons"])


def test_fs1_fails_when_k3_misses_the_rule(tmp_path, pins):
    reps = fs1_reps(tmp_path, pins)
    next(r for r in reps if r["cond"] == "K3")["cache_config"] = {"num_gpu_blocks": 170, "block_size": 528}
    v = sm.evaluate("FS1", reps, pins=pins)
    assert v["ok"] is False and v["k3"]["next_kv_low_bytes"] == 3 * GIB + STEP


def test_fs2_checks_the_chunk_sizes(tmp_path, pins):
    ds, p = mixed_ds(tmp_path, pins)
    world = World()
    b0 = probe(tmp_path / "s", by_name(sm.probe_specs("FS2", p, ds))["fs2-B0"], world,
               FakeRunAdv(world, steps=long_prefill(0)), p)
    t = probe(tmp_path / "s", by_name(sm.probe_specs("FS2", p, ds))["fs2-T"], world,
              FakeRunAdv(world, steps=long_prefill(512)), p)
    assert sm.evaluate("FS2", [b0, t], pins=p) == {"ok": True, "reasons": []}
    bad_t = dict(t, steps=b0["steps"])
    v = sm.evaluate("FS2", [b0, bad_t], pins=p)
    assert v["ok"] is False and any("T:" in r and "512" in r for r in v["reasons"])
    bad_b0 = dict(b0, steps=t["steps"])
    v = sm.evaluate("FS2", [bad_b0, t], pins=p)
    assert any("B0:" in r and "2048" in r for r in v["reasons"])


def test_fs3_needs_preemptions_under_g_and_none_under_b0(tmp_path, pins):
    ds, p = mixed_ds(tmp_path, pins)
    specs = by_name(sm.probe_specs("FS3", p, ds))
    wg, wb = World(preempt_per_scrape=2), World()
    g = probe(tmp_path / "s", specs["fs3-G"], wg, FakeRunAdv(wg), p)
    b0 = probe(tmp_path / "s", specs["fs3-B0"], wb, FakeRunAdv(wb), p)
    assert g["preemptions"] == 3 and g["preemptions_scraped"] == 2 and b0["preemptions"] == 0
    assert sm.evaluate("FS3", [g, b0], pins=p) == {"ok": True, "reasons": []}
    v = sm.evaluate("FS3", [dict(g, preemptions=0), dict(b0, preemptions=1)], pins=p)
    assert any(r.startswith("G:") and "preemption" in r for r in v["reasons"])
    assert any(r.startswith("B0:") and "preemption" in r for r in v["reasons"])
    v = sm.evaluate("FS3", [dict(g, bench=dict(g["bench"], completed=3)), b0], pins=p)
    assert v["ok"] is False and any("G:" in r and "complete" in r for r in v["reasons"])


def test_fs4_checks_fields_validity_and_completion(tmp_path, pins):
    ds, p = mixed_ds(tmp_path, pins)
    world = World()
    reps = run_fs(tmp_path / "a", "FS4", p, world, FakeRunAdv(world), ds)
    assert sm.evaluate("FS4", reps, pins=p) == {"ok": True, "reasons": []}
    reps = run_fs(tmp_path / "b", "FS4", p, world, FakeRunAdv(world, drop_fields=("kv_peak",)), ds)
    assert any("kv_peak" in r for r in sm.evaluate("FS4", reps, pins=p)["reasons"])
    reps = run_fs(tmp_path / "c", "FS4", p, world,
                  FakeRunAdv(world, decomp_checks={"a": True, "b": False, "c": None, "d": True}), ds)
    v = sm.evaluate("FS4", reps, pins=p)
    assert any("(b)" in r for r in v["reasons"]) and not any("(c)" in r for r in v["reasons"])


def null_hist(name):
    return lambda d: d.update({name: None})


@pytest.mark.parametrize("kw, match", [
    ({"scrape_edit": null_hist("vllm:request_prefill_time_seconds")}, "vllm:request_prefill_time_seconds"),
    ({"scrape_edit": null_hist("vllm:time_to_first_token_seconds")}, "vllm:time_to_first_token_seconds"),
    ({"scrape_edit": null_hist("vllm:generation_tokens_total")}, "vllm:generation_tokens_total"),
    ({"scrape_edit": lambda d: d.update(errors=["vllm:request_decode_time_seconds bucket bounds differ"])},
     "scrape delta: 1 error"),
    ({"scrape_edit": lambda d: d.clear()}, "scrape delta"),
    ({"one_hz": {"kv_full_while_waiting": None, "gate_zone": None, "n_samples": 0}}, "1 Hz shares"),
    ({"one_hz": {"kv_full_while_waiting": 0.0, "gate_zone": None, "n_samples": 4}}, "1 Hz shares"),
    ({"series_samples": 2}, "1 Hz series"),
])
def test_fs4_requires_the_scrape_delta_and_the_1_hz_series(tmp_path, pins, kw, match):
    """024 FS4: rows, derived/, scrape delta, 1 Hz series and the manipulation fields must all be complete. A series
    of 2 samples is only the sampler's start and stop: its 1 Hz loop never ran."""
    ds, p = mixed_ds(tmp_path, pins)
    world = World()
    reps = run_fs(tmp_path / "x", "FS4", p, world, FakeRunAdv(world, **kw), ds)
    v = sm.evaluate("FS4", reps, pins=p)
    assert v["ok"] is False
    bad = [r for r in v["reasons"] if match in r]
    assert {r.split(":")[0] for r in bad} == {"fs4-B0-uniform", "fs4-B0-mixed"}, v["reasons"]  # TG only completes
    assert sm.latency_keys(reps) == []


def test_fs4_tg_only_needs_to_complete(tmp_path, pins):
    ds, p = mixed_ds(tmp_path, pins)
    world = World()
    reps = run_fs(tmp_path, "FS4", p, world, FakeRunAdv(world), ds)
    tg = next(r for r in reps if r["name"] == "fs4-TG-mixed")
    tg["row"]["decomp"] = None
    tg["field_problems"] = ["decomp"]
    assert sm.evaluate("FS4", reps, pins=p)["ok"] is True
    tg["bench"] = dict(tg["bench"], completed=19, failed=1)
    v = sm.evaluate("FS4", reps, pins=p)
    assert v["ok"] is False and any(r.startswith("fs4-TG-mixed") for r in v["reasons"])


def test_fs5_timeout_path(tmp_path, pins):
    world = World()
    (spec,) = sm.probe_specs("FS5", pins, DS)
    rep = probe(tmp_path / "a", spec, world, FakeRunAdv(world), pins)
    assert rep["row"]["ok"] is False and rep["row"]["timeout"] is True and rep["bench"] == {"exists": False}
    assert sm.evaluate("FS5", [rep], pins=pins) == {"ok": True, "reasons": []}
    cases = [(World(residual=[4242]), {}, "process group"), (World(residual=None), {}, "process group"),
             (World(health_ok_calls=1), {}, "healthy"),
             (World(ports_open={4318}), {}, "receiver"), (World(ports_open={8000}), {}, "server port"),
             (World(), {"teardown_problems": ["server had to be killed"]}, "teardown")]
    for i, (w, kw, match) in enumerate(cases):
        rep = probe(tmp_path / f"c{i}", spec, w, FakeRunAdv(w, **kw), pins)
        v = sm.evaluate("FS5", [rep], pins=pins)
        assert v["ok"] is False and any(match in r for r in v["reasons"]), (match, v)


def test_fs5_fails_when_the_row_is_not_a_timeout(tmp_path, pins):
    world = World()
    (spec,) = sm.probe_specs("FS5", pins, DS)
    rep = probe(tmp_path, dict(spec, timeout_s=300), world, FakeRunAdv(world), pins)
    v = sm.evaluate("FS5", [rep], pins=pins)
    assert v["ok"] is False and any("timeout" in r for r in v["reasons"])


# ---------------------------------------------------------------- whole smoke --

def run_all(tmp_path, pins, world, runner, fs=sm.FS_NAMES, **kw):
    src = tmp_path / "mixed-2048.jsonl"
    p = pinned_pins(pins, write_dataset(src, dataset_lines()))
    return sm.run_smoke(tmp_path / "smoke", fs=fs, pins=p, pins_sha256="abc", mixed_dataset=src, session_runner=runner,
                        http_get=world.http_get, gpu_mem_used=world.gpu_mem_used, procs_with=world.procs_with,
                        port_open=world.port_open, now=world.now, sleep=world.sleep, **kw), p


class PerSpecRunner(FakeRunAdv):
    """FS2 B0 / T chunking, FS3 G preemptions and the FS5 timeout come from the spec being run."""

    def __call__(self, plan, **kw):
        (boot,) = plan["boots"]
        self.steps = long_prefill(512 if boot["cond"] in ("T", "TG") else 0)
        self.world.nb = 186 if boot["cond"] == "K3" else 248
        self.world.preempt_per_scrape = 1 if (plan["smoke"], boot["cond"]) == ("FS3", "G") else 0
        return super().__call__(plan, **kw)


def test_run_smoke_end_to_end_with_fakes(tmp_path, pins):
    world = World()
    rep, _ = run_all(tmp_path, pins, world, PerSpecRunner(world))
    assert rep["ok"] is True, rep
    assert [rep["fs"][f]["ok"] for f in sm.FS_NAMES] == [True] * 5
    assert rep["gpu_lock_held"] is True and rep["formal"] is False
    saved = json.loads((tmp_path / "smoke" / "smoke.json").read_text())
    assert saved == rep and sm.latency_keys(saved) == []
    assert not list((tmp_path / "smoke").rglob("bench-*.json"))
    assert saved["fs"]["FS1"]["k3"]["num_gpu_blocks"] == 186


def test_run_smoke_subset_and_no_overwrite(tmp_path, pins):
    world = World()
    rep, _ = run_all(tmp_path, pins, world, PerSpecRunner(world), fs=("FS5",))
    assert list(rep["fs"]) == ["FS5"]
    with pytest.raises(sm.SmokeError, match="exists"):
        run_all(tmp_path, pins, world, PerSpecRunner(world), fs=("FS5",))


def test_run_smoke_stops_and_saves_when_the_gpu_turns_busy(tmp_path, pins):
    world = World(mem=[400] * 8 + [8000])
    runner = PerSpecRunner(world)
    with pytest.raises(sm.SmokeError, match="GPU busy"):
        run_all(tmp_path, pins, world, runner)
    saved = json.loads((tmp_path / "smoke" / "smoke.json").read_text())
    assert saved["ok"] is False and saved["aborted"].startswith("GPU busy")


# ------------------------------------------------------------------------ CLI --

def test_cli_requires_the_gpu_lock(tmp_path, capsys):
    assert sm.main(["--out", str(tmp_path / "s")]) == 2
    assert "--gpu-lock-held" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_cli_refuses_an_existing_session(tmp_path, capsys):
    (tmp_path / "s").mkdir()
    assert sm.main(["--out", str(tmp_path / "s"), "--gpu-lock-held"], lock_held_check=lambda: True) == 2
    assert "exists" in capsys.readouterr().err


def test_cli_verifies_the_lock_instead_of_trusting_the_flag(tmp_path, capsys):
    """--gpu-lock-held without flock -n: refused before anything is written."""
    assert sm.main(["--out", str(tmp_path / "s"), "--gpu-lock-held"], lock_held_check=lambda: False) == 2
    err = capsys.readouterr().err
    assert "does not hold" in err and "flock -n" in err
    assert not (tmp_path / "s").exists()


def test_cli_checks_the_lock_with_the_runners_probe_by_default(tmp_path, monkeypatch, capsys):
    import run_adv as ra

    calls = []
    monkeypatch.setattr(ra, "held_gpu_lock", lambda: calls.append(1) or False)
    assert sm.main(["--out", str(tmp_path / "s"), "--gpu-lock-held"]) == 2
    assert calls and "does not hold" in capsys.readouterr().err


def test_cli_refuses_an_unknown_fs(tmp_path):
    with pytest.raises(SystemExit):
        sm.main(["--out", str(tmp_path / "s"), "--gpu-lock-held", "--only", "FS9"])


# ------------------------------------------------- the real runner, faked world --

EPOCH = 1_760_000_000.0
SAMPLE_S = 0.002  # the runner's 1 Hz interval, shrunk so its sampler loop runs during the fake bench


class RealWorld:
    """A fake WSL for the real run_adv.run_session (as tests/test_run_adv.py fakes it): a server whose log,
    /metrics and spans follow its argv, a bench that honours the argv the smoke rewrote, a clock."""

    def __init__(self, timeout_below_s=10):
        self.t, self.cond, self.proc, self.log_path, self.receiver, self.iter = 0.0, None, None, None, None, 0
        self.timeout_below_s, self.started, self.bench_argvs = timeout_below_s, [], []
        self.c = dict.fromkeys(("q_count", "q_sum", "prompt", "pre"), 0.0)
        self.health_calls = self.metrics_calls = 0

    def wall(self):
        return EPOCH + self.t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def popen(self, argv, env, log_path):
        a = dict(zip(argv, argv[1:]))
        nd = {"model_tag": argv[2], "max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
              "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"]),
              "otlp_traces_endpoint": a["--otlp-traces-endpoint"], "enable_logging_iteration_details": True}
        if "--long-prefill-token-threshold" in a:
            nd["long_prefill_token_threshold"] = int(a["--long-prefill-token-threshold"])
        if "--no-scheduler-reserve-full-isl" in argv:
            nd["scheduler_reserve_full_isl"] = False
        k3 = a["--kv-cache-memory-bytes"] == str(3 * GIB)
        t, g = "long_prefill_token_threshold" in nd, "scheduler_reserve_full_isl" in nd
        self.cond = "K3" if k3 else {(0, 0): "B0", (1, 0): "T", (0, 1): "G", (1, 1): "TG"}[(t, g)]
        self.started.append(self.cond)
        lines = [f"(APIServer pid=1) INFO 10-10 12:00:00 [api_utils.py:286] non-default args: {nd!r}",
                 "(EngineCore pid=2) INFO 10-10 12:00:01 [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend."]
        self.log_path, self.iter = Path(log_path), 0
        self.log_path.write_text("".join(f"{int(self.wall() * 1000)} {x}\n" for x in lines))

        class Proc:
            pid, returncode = 77, None

            def poll(self):
                return self.returncode

        self.proc = Proc()
        return self.proc

    def signal_group(self, proc, sig):
        proc.returncode = 0

    def group_alive(self, proc):
        return proc.returncode is None

    def receiver_factory(self, raw_dir, spans_path, prefix):
        import run_adv as ra

        world = self

        class R(ra.OtlpReceiver):
            def start(self):
                world.receiver = self
                return self

            def close(self):
                pass

        return R(raw_dir, spans_path, prefix=prefix, decode=lambda body, gz: json.loads(body))

    def metrics(self):
        nb = 186 if self.cond == "K3" else 248
        c, lab = self.c, 'engine="0"'
        other = ("request_prefill_time_seconds", "time_to_first_token_seconds", "request_decode_time_seconds",
                 "request_inference_time_seconds", "iteration_tokens_total")
        hists = [line for h in other for line in (f'vllm:{h}_bucket{{{lab},le="+Inf"}} {c["q_count"]}',
                                                  f"vllm:{h}_sum{{{lab}}} 0.0", f"vllm:{h}_count{{{lab}}} {c['q_count']}")]
        return "\n".join([
            f'vllm:cache_config_info{{block_size="528",engine="0",num_gpu_blocks="{nb}"}} 1.0',
            f"vllm:kv_cache_usage_perc{{{lab}}} 0.5", f"vllm:num_requests_running{{{lab}}} 1.0",
            f"vllm:num_requests_waiting{{{lab}}} 0.0", f"vllm:num_preemptions_total{{{lab}}} {c['pre']}",
            f"vllm:prompt_tokens_total{{{lab}}} {c['prompt']}", f"vllm:generation_tokens_total{{{lab}}} 0.0",
            f'vllm:request_queue_time_seconds_bucket{{{lab},le="+Inf"}} {c["q_count"]}',
            f"vllm:request_queue_time_seconds_sum{{{lab}}} {c['q_sum']}",
            f"vllm:request_queue_time_seconds_count{{{lab}}} {c['q_count']}", *hists, ""])

    def http_get(self, url, timeout=5):
        if url.endswith("/health"):
            self.health_calls += 1
            return (200, "") if self.proc is not None and self.proc.returncode is None else (0, "refused")
        self.metrics_calls += 1
        return 200, self.metrics()

    def run_bench(self, argv, log_path, env, timeout_s):
        self.bench_argvs.append(argv)
        start, deadline = self.metrics_calls, time.monotonic() + 5
        while self.metrics_calls - start < 2 and time.monotonic() < deadline:
            time.sleep(0.005)  # the runner's 1 Hz loop (here SAMPLE_S) samples while the bench runs
        Path(log_path).write_text("Initial test run completed.\n== Serving Benchmark Result ==\nMean TTFT (ms): 5\n")
        if timeout_s < self.timeout_below_s:
            self.t += timeout_s
            return -9, True
        a = dict(zip(argv, argv[1:]))
        n, prefix = int(a["--num-prompts"]), a["--request-id-prefix"]
        if "--dataset-path" in a:
            rows = Path(a["--dataset-path"]).read_text().splitlines()[:n]
            lens = [LONG if len(json.loads(r)["prompt"]) > 50000 else SHORT for r in rows]
        else:
            lens = [int(a["--random-input-len"])] * n
        reqs = [("cmpl-readycheck-0", lens[0])] + [(f"cmpl-{prefix}{i}-0", x) for i, x in enumerate(lens)]
        chunk = 512 if self.cond in ("T", "TG") else 2048
        t0, lines, k = self.t, [], 0
        for _, tokens in reqs:
            left = tokens
            while left > 0:
                m = min(chunk, left)
                left -= m
                stamp = int((EPOCH + t0 + 0.3 + 0.0005 * k) * 1000)
                lines.append(f"{stamp} (APIServer pid=1) INFO [loggers.py:185] Engine 000: Iteration({self.iter}): "
                             f"1 context requests, {m} context tokens, 0 generation requests, 0 generation tokens, "
                             "iteration elapsed time: 10.00 ms, GPU KV cache usage: 50.0%")
                self.iter += 1
                k += 1
        with open(self.log_path, "a") as fh:
            fh.write("".join(x + "\n" for x in lines))
        spans = [{"name": "llm_request", "start_time_unix_nano": int(round((EPOCH + t0 + 0.2 + 0.01 * i) * 1e9)),
                  "attributes": {"gen_ai.request.id": rid, "gen_ai.usage.prompt_tokens": tok,
                                 "gen_ai.usage.completion_tokens": 128, "gen_ai.latency.time_in_queue": 0.0,
                                 "gen_ai.latency.time_in_model_prefill": 0.05,
                                 "gen_ai.latency.time_to_first_token": 0.05, "gen_ai.latency.e2e": 1.0}}
                 for i, (rid, tok) in enumerate(reqs)]
        assert self.receiver.receive(json.dumps(spans).encode(), None) == 200
        self.c["q_count"] += len(reqs)
        self.c["prompt"] += sum(t for _, t in reqs)
        if self.cond == "G" and set(lens) == {LONG} and n == 4:
            self.c["pre"] += 2  # over-commit: 4 x 66 > 247 blocks
        res = {"completed": n, "failed": 0, "num_prompts": n, "duration": 12.0, "request_throughput": 1.0,
               "output_throughput": 1.0, "total_token_throughput": 1.0, "input_lens": lens, "output_lens": [128] * n,
               "ttfts": [0.1 + 0.001 * i for i in range(n)], "latencies": [1.2 + 0.001 * i for i in range(n)],
               "errors": [""] * n, **{f"{s}_{m}_ms": 1.0 for s in ("mean", "median", "p50", "p95", "p99")
                                      for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")}}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        self.t += 12.0
        return 0, False

    def gpu_mem_used(self):
        return 400

    def backends(self):
        return {"popen": self.popen, "signal_group": self.signal_group, "group_alive": self.group_alive,
                "http_get": self.http_get, "run_bench": self.run_bench, "gpu_mem_used": self.gpu_mem_used,
                "gpu_lock_free": lambda: True, "lock_held_check": lambda: True,
                "receiver_factory": self.receiver_factory, "capture_env": None,
                "toolchain_check": lambda env: [], "base_env": {"PATH": "/usr/bin"}, "now": self.now,
                "sleep": self.sleep, "wall": self.wall, "sample_interval": SAMPLE_S}


def real_smoke(tmp_path, pins, world, fs=sm.FS_NAMES):
    src = tmp_path / "mixed-2048.jsonl"
    p = pinned_pins(pins, write_dataset(src, dataset_lines()))
    runner = sm.run_adv_session_runner(gdn_prefill_backend="triton", gpu_memory_utilization="0.91",
                                       vllm_bin=".venv/bin/vllm", **world.backends())
    return sm.run_smoke(tmp_path / "smoke", fs=fs, pins=p, pins_sha256="abc", mixed_dataset=src, session_runner=runner,
                        http_get=world.http_get, gpu_mem_used=world.gpu_mem_used, procs_with=lambda token: [],
                        port_open=lambda host, port: False, now=world.now, sleep=world.sleep)


def test_every_smoke_passes_through_the_real_runner(tmp_path, pins):
    """adv_smoke -> run_adv.run_session (real plan check, manipulation check, receiver, decomposition, rows) ->
    masked evidence; only the server, client and clock are fake."""
    world = RealWorld()
    rep = real_smoke(tmp_path, pins, world)
    assert {f: v["reasons"] for f, v in rep["fs"].items()} == {f: [] for f in sm.FS_NAMES}
    assert rep["ok"] is True and rep["fs"]["FS1"]["k3"]["freeze_ready"] is True
    assert world.started == ["B0", "T", "G", "TG", "K3", "B0", "T", "G", "B0", "B0", "B0", "TG", "B0"][:13]
    n_prompts = [int(sm.flag_value(a, "--num-prompts")) for a in world.bench_argvs]
    assert n_prompts == [1] * 5 + [1, 1] + [4, 4] + [20, 20, 20, 20]  # never Core's N at a formal C
    probes = {p["name"]: p for v in rep["fs"].values() for p in v["probes"]}
    assert probes["fs4-B0-mixed"]["row"]["N_bench"] == 20 and probes["fs4-B0-mixed"]["row"]["N"] == 200
    assert probes["fs5-B0-uniform"]["row"]["timeout"] is True
    smoke_dir = tmp_path / "smoke"
    assert not [p for p in smoke_dir.rglob("*") if p.is_file() and (p.name.endswith(".bench.json") or
                                                                     p.name.endswith(".spans.jsonl") or
                                                                     ".metrics." in p.name or p.suffix in (".pb",))]
    for rows in smoke_dir.rglob("rows.jsonl"):
        assert all(sm.latency_keys(json.loads(x)) == [] for x in rows.read_text().splitlines())
    for d in smoke_dir.rglob("derived/*.jsonl"):
        assert all(sm.latency_keys(json.loads(x)) == [] for x in d.read_text().splitlines())
    metas = list(smoke_dir.rglob("session.json"))
    assert len(metas) == 13
    for meta in metas:  # the runner's session.json loses startup_s / t_before / t_after
        m = json.loads(meta.read_text())
        assert sm.latency_keys(m) == [] and m["runs"] and all("argv" in r for r in m["runs"].values())
    assert sm.latency_keys(json.loads((smoke_dir / "smoke.json").read_text())) == []
    fs4 = [p for p in rep["fs"]["FS4"]["probes"] if p["cond"] == "B0"]
    assert all(p["observability"] == [] and p["series_samples"] > 2 for p in fs4)


# ------------------------------------------------------------ real helpers --

def test_port_open_on_loopback():
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert sm.port_open("127.0.0.1", port) is True
    finally:
        srv.close()
    assert sm.port_open("127.0.0.1", port) is False


@pytest.mark.skipif(Path("/proc").is_dir(), reason="Linux / WSL has /proc")
def test_procs_with_fails_safe_without_proc():
    assert sm.procs_with("adv-B0-2048-uniform-r1-") is None


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc (WSL / Linux)")
def test_procs_with_finds_a_live_client_by_its_prefix():
    import subprocess
    import sys

    token = "adv-smoke-test-prefix-r9-"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", token])
    try:
        assert proc.pid in sm.procs_with(token)
    finally:
        proc.kill()
        proc.wait()
    assert proc.pid not in sm.procs_with(token)


def test_cli_wires_the_real_runner_and_the_pins_bytes(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_runner(**kw):
        seen["runner"] = kw
        return "RUNNER"

    def fake_smoke(out, **kw):
        seen["smoke"] = (out, kw)
        return {"ok": True, "fs": {"FS5": {"ok": True, "reasons": []}}}

    import run_adv as ra

    monkeypatch.setattr(sm, "run_adv_session_runner", fake_runner)
    monkeypatch.setattr(sm, "run_smoke", fake_smoke)
    monkeypatch.setattr(sm.smoke.signal, "signal", lambda *a: None)  # main's SIGHUP / SIGTERM handlers stay out
    assert sm.main(["--out", str(tmp_path / "s"), "--gpu-lock-held", "--only", "FS5,FS1", "--k3-bytes",
                    str(3 * GIB + STEP)], lock_held_check=lambda: True) == 0
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    assert seen["runner"]["gdn_prefill_backend"] == pins["stage_b"]["gdn_prefill_backend"]
    assert seen["runner"]["gpu_memory_utilization"] == pins["stage_b"]["gpu_memory_utilization"]
    out, kw = seen["smoke"]
    assert out == tmp_path / "s" and kw["fs"] == ["FS5", "FS1"] and kw["session_runner"] == "RUNNER"
    assert kw["pins_sha256"] == hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    assert kw["k3_bytes"] == 3 * GIB + STEP and kw["server_port"] == 8000
    assert kw["procs_with"] is sm.procs_with and kw["port_open"] is sm.port_open
    assert kw["gpu_mem_used"] is ra.nvidia_mem_used_bounded  # nvidia-smi with a timeout
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_the_real_runner_wiring_uses_the_bounded_nvidia_smi_and_the_lock_probe(tmp_path, monkeypatch):
    import run_adv as ra

    seen = {}
    monkeypatch.setattr(ra, "run_session", lambda plan, **kw: seen.update(kw) or {})
    runner = sm.run_adv_session_runner(gdn_prefill_backend="triton", gpu_memory_utilization="0.91",
                                       vllm_bin=".venv/bin/vllm", root=ROOT)
    runner({"pins_sha256": "abc"}, pins={}, out_dir=tmp_path / "s", session_id="s", mixed_datasets={},
           wrap_http_get=lambda f: f, wrap_run_bench=lambda f: f)
    assert seen["gpu_mem_used"] is ra.nvidia_mem_used_bounded and seen["lock_held_check"] is ra.held_gpu_lock
    assert seen["gpu_lock_held"] is True and seen["formal"] is False
