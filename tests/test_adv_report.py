"""Tests for benchmarks/adv_report.py: the Advanced report CLI and every decision 024 A14 refusal
(PREREGISTRATION §7), written before any Advanced data. Fixtures are synthetic; no results/ value is read."""

import ast
import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import adv_analysis as aa
import adv_plan as ap
import adv_report as ar
import core_b_plan as bp
import run_adv as ra
from run_core_a import sha256_json

ROOT = Path(__file__).resolve().parent.parent
SERVER = {"gdn_prefill_backend": "triton", "gpu_memory_utilization": "0.91", "port": 8000,
          "vllm_bin": ".venv/bin/vllm"}
RESULT_DIR = "/home/u/llm-serving-lab/results/advanced/formal-20261011-adv/raw"
DATASETS = {"2048": "data/core_b/mixed-2048.jsonl", "8192": "data/core_b/mixed-8192.jsonl"}
# (uniform, mixed) short-class P95 TTFT per condition; restarts at -0.2 %, 0, +0.2 %
Y = {"B0": (100.0, 400.0), "T": (90.0, 150.0), "G": (100.0, 380.0), "TG": (90.0, 140.0), "K3": (100.0, 420.0)}


def frozen_pins_bytes() -> bytes:
    """configs/pins.toml frozen: freeze.advanced = true and the K3 count measured (186; smoke FS1 2026-10-11).
    Works whether the repo pins are still the draft or already frozen."""
    text = (ROOT / "configs" / "pins.toml").read_text()
    if text.count("\nadvanced = false") == 1:
        assert text.count("\nnum_gpu_blocks_k3 = 0 ") == 1
        text = (text.replace("\nadvanced = false", "\nadvanced = true", 1)
                .replace("\nnum_gpu_blocks_k3 = 0 ", "\nnum_gpu_blocks_k3 = 186 ", 1))
    assert text.count("\nadvanced = true") == 1 and text.count("\nnum_gpu_blocks_k3 = 186 ") == 1
    return text.encode()


def unfrozen_pins_bytes() -> bytes:
    """The draft pins (freeze.advanced = false, K3 not measured), derived from the frozen bytes."""
    return (frozen_pins_bytes().decode().replace("\nadvanced = true", "\nadvanced = false", 1)
            .replace("\nnum_gpu_blocks_k3 = 186 ", "\nnum_gpu_blocks_k3 = 0 ", 1).encode())


@pytest.fixture(scope="module")
def frozen():
    raw = frozen_pins_bytes()
    return tomllib.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def uniform_core(pins, knee=None, **meta):
    knee = knee or {2048: 25, 8192: 8}
    up = bp.build_uniform_plan(pins, num_gpu_blocks=248, block_size=528)
    rows = [{"L": L, "mix": "uniform", "restart": r, "C": C, "ok": True, "output_throughput": 100.0 * min(C, knee[L]),
             "short_class_p95_ttft_ms": 80.0}
            for L in (2048, 8192) for r in (1, 2, 3) for C in up["grids"][str(L)]]
    return {"session": {"session_id": ap.DERIVED_FROM, "formal": True, "env_valid": True, **meta}, "plan": up,
            "rows": rows}


def core_mixed(**meta):
    rows = [{"L": L, "mix": "mixed", "restart": r, "C": C, "ok": True, "short_class_p95_ttft_ms": 320.0}
            for L, C in ap.C_KNEE.items() for r in (1, 2, 3)]
    return {"session": {"session_id": "formal-20261010-coreb-mixed2", "formal": True, "env_valid": True, **meta},
            "plan": {"derived_from": ap.DERIVED_FROM}, "rows": rows}


def adv_session(pins, sha):
    """A formal Advanced session as run_adv records it: server argv / env per boot, bench argv per row."""
    u = uniform_core(pins)
    plan = ap.build_plan(pins, session=u["session"], uniform_plan=u["plan"], rows=u["rows"], pins_sha256=sha)
    runs, rows = {}, []
    for b in plan["boots"]:
        tag = ra.boot_tag(b)
        argv, env = ap.server_argv_for(b["cond"], pins, base_env={"PATH": "/usr/bin", "OTEL_SDK_DISABLED": "1"},
                                       **SERVER)
        runs[tag] = {"argv": argv, "env": ra.recorded_server_env(env)}
        mixed = b["mix"] == "mixed"
        y = Y[b["cond"]][mixed] * (1 + 0.002 * (b["restart"] - 2))
        rows.append({**{k: b[k] for k in ("cond", "L", "mix", "restart", "block", "C", "N", "prefix", "timeout_s")},
                     "tag": tag, "ok": True, "reasons": [], "timeout": False, "skipped": False,
                     "short_class_p95_ttft_ms": y, "long_class_p95_ttft_ms": 5000.0 if mixed else None,
                     "output_throughput": 1000.0, "total_token_throughput": 9000.0, "p95_tpot_ms": 30.0,
                     "preemptions": 0, "kv_peak": 0.9,
                     "manip": {"facts_ok": True, "blocks_ok": True,
                               "chunk_ok": True if b["cond"] in ("T", "TG") else None,
                               "num_gpu_blocks": plan["expected_blocks"][b["cond"]]},
                     "decomp": None,
                     "argv": ap.bench_argv_for(pins, cond=b["cond"], L=b["L"], C=b["C"], mix=b["mix"],
                                               restart=b["restart"], result_dir=RESULT_DIR,
                                               result_filename=f"{tag}.bench.json", port=SERVER["port"],
                                               mixed_dataset=DATASETS[str(b["L"])] if mixed else None,
                                               vllm_bin=SERVER["vllm_bin"])})
    session = {"schema_version": 1, "session_id": "formal-20261011-adv", "formal": True, "env_valid": True,
               "plan_sha256": sha256_json(plan), "pins_sha256": sha, "plan_pins_sha256": sha,
               "derived_from": ap.DERIVED_FROM, "server": dict(SERVER), "result_dir": RESULT_DIR,
               "mixed_datasets": dict(DATASETS),
               "mixed_dataset_sha256": dict(pins["core_b"]["mix_dataset_sha256"]), "runs": runs}
    return {"session": session, "plan": plan, "rows": rows}


def resign(adv):
    """Record the (modified) plan's sha256 as the one the session ran, isolating another refusal."""
    adv["session"]["plan_sha256"] = sha256_json(adv["plan"])
    return adv


def first(adv, cond="B0", mix="mixed"):
    return next(r for r in adv["rows"] if r["cond"] == cond and r["mix"] == mix)


# ------------------------------------------------------------------ accepted input --

def test_a_complete_formal_session_passes_every_refusal_and_is_analysed(frozen):
    pins, sha = frozen
    adv, uni = adv_session(pins, sha), uniform_core(pins)
    assert ar.refusal_problems(adv, uni, pins=pins, pins_sha256=sha) == []
    rep = ar.report(adv, uni, pins=pins, pins_sha256=sha)
    want = aa.analyze(copy.deepcopy(adv), uni)
    assert json.loads(json.dumps({k: v for k, v in rep.items() if k != "report"})) == json.loads(json.dumps(want))
    assert rep["report"] == {"session_id": "formal-20261011-adv", "uniform_session": ap.DERIVED_FROM,
                             "core_mixed_session": None, "plan_sha256": sha256_json(adv["plan"]),
                             "pins_sha256": sha, "mixed_dataset_sha256": pins["core_b"]["mix_dataset_sha256"]}
    assert set(rep["levels"]) == {"2048", "8192"} and len(rep["runs"]) == 60


def test_the_report_runs_adv_analysis(frozen, monkeypatch):
    pins, sha = frozen
    seen = []
    monkeypatch.setattr(aa, "analyze", lambda adv, uni, core_mixed=None: seen.append(core_mixed) or {"x": 1})
    cm = core_mixed()
    assert ar.report(adv_session(pins, sha), uniform_core(pins), pins=pins, pins_sha256=sha,
                     core_mixed=cm)["x"] == 1
    assert seen == [cm]


def test_skipped_and_invalid_runs_are_kept_and_reported(frozen):
    pins, sha = frozen
    adv = adv_session(pins, sha)
    t = [r for r in adv["rows"] if (r["cond"], r["L"], r["mix"]) == ("T", 2048, "mixed")]
    t[0].update(ok=False, timeout=True, reasons=["timeout"], short_class_p95_ttft_ms=None)
    for r in t[1:]:
        r.update(ok=False, skipped=True, reasons=["skipped"], short_class_p95_ttft_ms=None, argv=None)
        adv["session"]["runs"].pop(r["tag"])  # skipped boots never start a server
    rep = ar.report(adv, uniform_core(pins), pins=pins, pins_sha256=sha)
    bad = [r for r in rep["runs"] if not r["valid"]]
    assert len(bad) == 3 and rep["levels"]["2048"]["conditions"]["T"]["category"] in ("NT", None)


# ------------------------------------------------------------------ refusals (A14) --

def mutate_runs(adv, fn, cond="B0"):
    tag = first(adv, cond)["tag"]
    fn(adv["session"]["runs"][tag])


@pytest.mark.parametrize("change, match", [
    # session: formal, env-valid, complete
    (lambda a, u, p: a["session"].update(formal=False), "not formal"),
    (lambda a, u, p: a["session"].update(env_valid=False), "env-invalid"),
    (lambda a, u, p: a["session"].pop("env_valid"), "env-invalid"),
    (lambda a, u, p: a["session"].update(aborted="RunError: GPU busy"), "aborted"),
    (lambda a, u, p: a["rows"].pop(), "do not cover"),
    (lambda a, u, p: a["rows"].append(dict(a["rows"][0])), "do not cover"),
    # freeze and pins
    (lambda a, u, p: p["freeze"].update(advanced=False), "freeze.advanced"),
    (lambda a, u, p: p["advanced"].update(long_prefill_token_threshold=1024), "long_prefill_token_threshold"),
    # derived_from and C
    (lambda a, u, p: resign(a)["plan"].update(derived_from="formal-x") or resign(a), "derived_from"),
    (lambda a, u, p: u["session"].update(session_id="formal-x"), "uniform session"),
    (lambda a, u, p: u["session"].update(formal=False), "uniform session"),
    (lambda a, u, p: u["session"].update(env_valid=None), "uniform session"),
    (lambda a, u, p: u.update(uniform_core(p, knee={2048: 18, 8192: 8})), "C_knee"),
    (lambda a, u, p: (a["plan"].update(c={"2048": 24, "8192": 8}), resign(a)), "C_knee"),
    (lambda a, u, p: first(a).update(C=24), "C = 24"),
    # argv / env beyond the condition table
    (lambda a, u, p: mutate_runs(a, lambda r: r["argv"].append("--no-scheduler-reserve-full-isl")), "server argv"),
    (lambda a, u, p: mutate_runs(a, lambda r: r["argv"].remove("--enable-logging-iteration-details"), "T"),
     "server argv"),
    (lambda a, u, p: mutate_runs(a, lambda r: r["env"]["OTEL"].update(OTEL_SDK_DISABLED="true")), "server env"),
    (lambda a, u, p: mutate_runs(a, lambda r: r["env"]["engine_vars"].append("VLLM_LOGGING_LEVEL"), "K3"),
     "server env"),
    (lambda a, u, p: mutate_runs(a, lambda r: r["env"].update(VLLM_GDN_DECODE_KERNEL="flashinfer"), "G"),
     "server env"),
    (lambda a, u, p: a["session"]["runs"].pop(first(a)["tag"]), "no server record"),
    (lambda a, u, p: first(a, "TG").update(argv=first(a, "TG")["argv"] + ["--request-rate", "8"]), "bench argv"),
    (lambda a, u, p: first(a, "T", "uniform")["argv"].__setitem__(-1, "adv-B0-2048-uniform-r1-"), "bench argv"),
    (lambda a, u, p: a["session"]["server"].update(gpu_memory_utilization="0.85"), "gpu_memory_utilization"),
    (lambda a, u, p: a["session"]["server"].update(gdn_prefill_backend="flashinfer"), "gdn_prefill_backend"),
    # mixed dataset
    (lambda a, u, p: a["session"]["mixed_dataset_sha256"].update({"8192": "f" * 64}), "mixed dataset"),
    (lambda a, u, p: a["session"].pop("mixed_dataset_sha256"), "mixed dataset"),
    # plan / pins sha
    (lambda a, u, p: a["plan"].update(seed=1), "plan.json"),
    (lambda a, u, p: (a["plan"].update(pins_sha256="ee" * 32), resign(a)), "plan was built"),
    (lambda a, u, p: a["session"].update(pins_sha256="ee" * 32), "session ran"),
    # the plan itself (seeded 60-boot order, frozen blocks)
    (lambda a, u, p: (a["plan"].update(boots=a["plan"]["boots"][::-1]), resign(a)), "60-boot"),
    (lambda a, u, p: (a["plan"]["expected_blocks"].update(K3=0), resign(a)), "expected_blocks"),
])
def test_refusals(frozen, change, match):
    pins, sha = frozen
    p = copy.deepcopy(pins)
    adv, uni = adv_session(p, sha), uniform_core(p)
    change(adv, uni, p)
    problems = ar.refusal_problems(adv, uni, pins=p, pins_sha256=sha)
    assert any(match in x for x in problems), problems
    with pytest.raises(ar.ReportError, match=match.replace("(", r"\(")):
        ar.report(adv, uni, pins=p, pins_sha256=sha)


def test_refuses_pins_other_than_configs_pins_toml(frozen):
    pins, sha = frozen
    adv = adv_session(pins, sha)
    problems = ar.refusal_problems(adv, uniform_core(pins), pins=pins, pins_sha256="ee" * 32)
    assert any("plan was built" in x for x in problems) and any("session ran" in x for x in problems)


def test_refuses_a_core_mixed_session_that_is_not_formal_or_not_derived_from_the_uniform(frozen):
    pins, sha = frozen
    adv, uni = adv_session(pins, sha), uniform_core(pins)
    assert ar.refusal_problems(adv, uni, pins=pins, pins_sha256=sha, core_mixed=core_mixed()) == []
    bad = core_mixed(formal=False)
    assert any("Core mixed" in x for x in ar.refusal_problems(adv, uni, pins=pins, pins_sha256=sha, core_mixed=bad))
    other = core_mixed()
    other["plan"]["derived_from"] = "formal-x"
    assert any("Core mixed" in x for x in ar.refusal_problems(adv, uni, pins=pins, pins_sha256=sha,
                                                               core_mixed=other))


def test_the_runs_record_of_skipped_boots_is_not_required(frozen):
    pins, sha = frozen
    adv = adv_session(pins, sha)
    r = first(adv, "G")
    r.update(skipped=True, ok=False, argv=None, short_class_p95_ttft_ms=None)
    adv["session"]["runs"].pop(r["tag"])
    assert ar.refusal_problems(adv, uniform_core(pins), pins=pins, pins_sha256=sha) == []


def test_an_analysis_error_is_a_refusal(frozen, monkeypatch):
    pins, sha = frozen

    def boom(*a, **k):
        raise aa.AnalysisError("row outside the condition table")

    monkeypatch.setattr(aa, "analyze", boom)
    with pytest.raises(ar.ReportError, match="condition table"):
        ar.report(adv_session(pins, sha), uniform_core(pins), pins=pins, pins_sha256=sha)


# ------------------------------------------------------------------ CLI --

def write_session(d: Path, part: dict) -> Path:
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(part["session"], indent=1) + "\n")
    (d / "plan.json").write_text(json.dumps(part["plan"], indent=1) + "\n")
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part["rows"]))
    return d


def cli_setup(tmp_path, frozen, monkeypatch, adv_change=None):
    pins, sha = frozen
    monkeypatch.setattr(ar, "load_pins_bytes", frozen_pins_bytes)
    adv = adv_session(pins, sha)
    if adv_change:
        adv_change(adv)
    s = write_session(tmp_path / "adv", adv)
    u = write_session(tmp_path / "uniform", uniform_core(pins))
    return s, u, tmp_path / "out" / "analysis.json"


def test_cli_end_to_end_writes_analysis_json(tmp_path, frozen, monkeypatch, capsys):
    s, u, out = cli_setup(tmp_path, frozen, monkeypatch)
    out.parent.mkdir()
    m = write_session(tmp_path / "mixed", core_mixed())
    assert ar.main(["--session", str(s), "--uniform-session", str(u), "--core-mixed-session", str(m),
                    "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    pins, sha = frozen
    want = ar.report(ar.load_session(s), ar.load_session(u), pins=pins, pins_sha256=sha,
                     core_mixed=ar.load_session(m))
    assert rep == json.loads(json.dumps(want))
    assert rep["report"]["core_mixed_session"] == "formal-20261010-coreb-mixed2"
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["rows"]) == {"2048", "8192"} and printed["session"] == "formal-20261011-adv"


def with_run_descriptives(adv):
    """The per-run descriptive fields run_adv records (scrape_deltas, one_hz_shares, client_stats, decomp tail95),
    scaled by the restart (1, 2, 3) so each cell's mean / SE is known: mean = 2 x, SE = x / sqrt 3."""
    shares = dict.fromkeys(aa.Q_CATEGORIES, 0.0) | {"no-queue": 1.0}
    for r in adv["rows"]:
        k, mixed = r["restart"], r["mix"] == "mixed"
        r["scrape_delta"] = {"vllm:request_queue_time_seconds": {"buckets": {}, "sum": 2.0 * k, "count": 2.0,
                                                                 "mean": 1.0 * k},
                             "vllm:request_prefill_time_seconds": {"buckets": {}, "sum": 6.0 * k, "count": 2.0,
                                                                   "mean": 3.0 * k},
                             "errors": []}
        r["one_hz"] = {"kv_full_while_waiting": 0.1 * k, "gate_zone": 0.2 * k, "n_samples": 300}
        r["client"] = {"short_p50_ttft_ms": 50.0 * k, "short_p99_ttft_ms": 500.0 * k,
                       "long_p50_ttft_ms": 4000.0 * k if mixed else None,
                       "long_p95_ttft_ms": 5000.0 if mixed else None, "short_p95_tpot_ms": 25.0 * k}
        r["decomp"] = {"valid": True, "checks": {"a": True, "b": True, "c": True, "d": True}, "reasons": [],
                       "tail": {"O": 1.0, "Q": 10.0, "P": 5.0, "F": 0.0},
                       "tail95": {"O": 2.0 * k, "Q": 20.0 * k, "P": 6.0 * k, "F": 0.0},
                       "q_shares": shares, "q_shares_sens": shares, "residual_bm_share": 0.0,
                       "unexplained_break_share": 0.0}


def test_the_report_writes_the_a13_a8_descriptive_table(tmp_path, frozen, monkeypatch):
    """024: adv_report outputs analysis.json with the per-(condition, L, mix) descriptive table, including the
    histogram queue / prefill means, the 1 Hz shares, the >= P95 tail and the client descriptives (A13 / A8)."""
    s, u, out = cli_setup(tmp_path, frozen, monkeypatch, adv_change=with_run_descriptives)
    out.parent.mkdir()
    assert ar.main(["--session", str(s), "--uniform-session", str(u), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    se = 1 / 3 ** 0.5
    for L in ("2048", "8192"):
        cells = rep["levels"][L]["descriptive"]["cells"]
        assert set(cells) == {f"{c}-{m}" for c in aa.CONDITIONS for m in aa.MIXES}
        for name, cell in cells.items():
            mixed = name.endswith("-mixed")
            assert cell["hist"]["queue_time_mean_s"] == {"mean": pytest.approx(2.0), "se": pytest.approx(se), "n": 3}
            assert cell["hist"]["prefill_time_mean_s"]["mean"] == pytest.approx(6.0)
            assert cell["one_hz"]["kv_full_while_waiting"]["mean"] == pytest.approx(0.2)
            assert cell["one_hz"]["gate_zone"]["se"] == pytest.approx(0.2 * se)
            assert cell["client"]["short_p50_ttft_ms"]["mean"] == pytest.approx(100.0)
            assert cell["client"]["short_p99_ttft_ms"]["mean"] == pytest.approx(1000.0)
            assert cell["client"]["short_p95_tpot_ms"]["mean"] == pytest.approx(50.0)
            assert (cell["client"]["long_p50_ttft_ms"] or {}).get("mean") == (pytest.approx(8000.0) if mixed else None)
            assert cell["tail95"]["Q"] == {"mean": pytest.approx(40.0), "se": pytest.approx(20 * se), "n": 3}
        assert rep["levels"][L]["descriptive"]["tail95_delta"]["B0"]["Q"]["delta"] == pytest.approx(0.0)


def test_cli_summary_does_not_reduce_a_not_testable_h_c0_to_r0(tmp_path, frozen, monkeypatch, capsys):
    def b0_mixed_invalid(adv):
        for r in adv["rows"]:
            if (r["cond"], r["mix"], r["restart"]) == ("B0", "mixed", 1):
                r.update(ok=False, reasons=["vllm bench serve exit 1"])

    s, u, out = cli_setup(tmp_path, frozen, monkeypatch, adv_change=b0_mixed_invalid)
    out.parent.mkdir()
    assert ar.main(["--session", str(s), "--uniform-session", str(u), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == {"2048": "R0", "8192": "R0"}
    assert set(printed["conclusions"].values()) == {aa.CONCLUSION_H_C0_NT}
    assert printed["incomplete"] == {"2048": True, "8192": True}
    assert printed["cross_l"] is None and printed["cross_l_conclusion"] == aa.CROSS_L_NT


def test_cli_refuses_and_writes_nothing(tmp_path, frozen, monkeypatch, capsys):
    s, u, out = cli_setup(tmp_path, frozen, monkeypatch, adv_change=lambda a: a["session"].update(formal=False))
    out.parent.mkdir()
    assert ar.main(["--session", str(s), "--uniform-session", str(u), "--out", str(out)]) == 2
    assert not out.exists() and "refusing" in capsys.readouterr().err


def test_cli_refuses_unfrozen_pins(tmp_path, frozen, monkeypatch, capsys):
    """With draft pins (freeze.advanced = false), nothing is analysed."""
    s, u, out = cli_setup(tmp_path, frozen, monkeypatch)
    monkeypatch.setattr(ar, "load_pins_bytes", unfrozen_pins_bytes)
    out.parent.mkdir()
    assert ar.main(["--session", str(s), "--uniform-session", str(u), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert not out.exists() and "freeze.advanced" in err and "pins" in err


def test_load_pins_bytes_reads_configs_pins_toml():
    assert ar.load_pins_bytes() == (ROOT / "configs" / "pins.toml").read_bytes()


def test_stdlib_only_and_python_3_10_syntax():
    src = Path(ar.__file__).read_text()
    ast.parse(src, feature_version=(3, 10))
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not [ln for ln in top if "tomllib" in ln or "run_core_b" in ln]
