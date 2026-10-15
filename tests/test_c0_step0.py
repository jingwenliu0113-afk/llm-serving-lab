"""Tests for benchmarks/c0_step0.py: the 028 step-0 desk checks S1-S5 over raw files fetched read-only from WSL,
with synthetic Core bench JSONs, Advanced steps / logs and manifests (no real raw file is in git)."""

import hashlib
import json
from pathlib import Path

import pytest

import c0_analysis as ca
import c0_step0 as s0
import run_c0_probe as rp

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_791_657_760.0
PERIOD = 12.72


def cycle(C=8, N=200, slips=(), shift=340.0):
    """Closed-loop TTFTs (ms), as in test_c0_analysis: idx < C is the first wave, a slipped group is shifted on its
    positions 1..C-1."""
    base = [1485.0] + [2477.0] * (C - 2) + [2320.0]
    t = [base[i % C] for i in range(N)]
    for i in range(1, C):
        t[i] = 2800.0 + 1200.0 * i
    for k in slips:
        for p in range(1, C):
            t[C * k + p] += shift
    return t


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(session, raw):
    files = {f"raw/{p.name}": {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in sorted(raw.iterdir())}
    (session / "raw_manifest.json").write_text(json.dumps({"note": "x", "files": files}, indent=1) + "\n")


# ------------------------------------------------------------------ Core (S1, S2, S5) --

CORE_SLIPS = {(1, 8): (8,), (2, 8): (16, 23), (2, 9): (14,), (1, 4): (10, 20, 30, 40)}
CORE_STARTUP = ["(APIServer pid=1) INFO 10-09 13:00:00 [api_utils.py:286] non-default args: "
                "{'max_num_batched_tokens': 2048}",
                "(EngineCore pid=2) INFO 10-09 13:00:05 [gpu_model_runner.py:4100] Graph capturing finished in 5 secs, "
                "took 0.52 GiB"]
SLOW = "(EngineCore pid=2) WARNING 10-09 13:00:06 [utils.py:10] slow startup"


def make_core(tmp_path, *, missing=(), tamper=()):
    session = tmp_path / "results" / "core_b" / s0.CORE_SID
    raw = tmp_path / "fetched" / s0.CORE_SID
    session.mkdir(parents=True)
    raw.mkdir(parents=True)
    for r in (1, 2, 3):
        for C in s0.CORE_C:
            t = cycle(C=C, slips=CORE_SLIPS.get((r, C), ()))
            if (r, C) == (1, 7):  # elevated without slips: a whole position sits 400 ms higher
                t = [x + 400.0 if i >= C and i % C == 3 else x for i, x in enumerate(t)]
            res = {"completed": 200, "failed": 0, "ttfts": [x / 1000 for x in t], "input_lens": [8064] * 200,
                   "output_lens": [128] * 200}
            (raw / s0.core_bench_name(r, C)).write_text(json.dumps(res))
        lines = CORE_STARTUP + ([SLOW] if r == 2 else []) + [
            "(APIServer pid=1) INFO:     Application startup complete.", "(APIServer pid=1) WARNING after startup"]
        (raw / s0.core_log_name(r)).write_text("\n".join(lines) + "\n")
    write_manifest(session, raw)
    for r, C in missing:
        (raw / s0.core_bench_name(r, C)).unlink()
    for r, C in tamper:
        with open(raw / s0.core_bench_name(r, C), "a") as fh:
            fh.write(" ")
    return session, raw


def test_core_file_list_is_the_028_set():
    names = s0.core_files()
    assert len(names) == 15 + 3
    assert "bench-uniform-8192-r2-c9.json" in names and "server-uniform-8192-r3.log" in names
    assert s0.CORE_C == (4, 7, 8, 9, 13)


def test_verify_raw_reports_ok_missing_mismatch_and_unlisted(tmp_path):
    session, raw = make_core(tmp_path, missing={(3, 13)}, tamper={(3, 4)})
    got = s0.verify_raw(session, raw, s0.core_files() + ["nope.json"])
    assert got["bench-uniform-8192-r1-c8.json"] == "ok"
    assert got["bench-uniform-8192-r3-c13.json"] == "missing"
    assert got["bench-uniform-8192-r3-c4.json"] == "sha256 mismatch"
    assert got["nope.json"] == "not in manifest"
    assert set(s0.verify_raw(session, tmp_path / "absent", s0.core_files()).values()) == {"missing"}


def test_s1_classifies_every_core_run_and_the_prestated_replication(tmp_path):
    session, raw = make_core(tmp_path, missing={(3, 13)}, tamper={(3, 4)})
    status = s0.verify_raw(session, raw, s0.core_files())
    s1 = s0.s1_core(raw, status)
    runs = {(x["restart"], x["C"]): x for x in s1["runs"]}
    assert len(runs) == 15
    assert runs[(3, 13)]["status"] == "missing" and "episodes" not in runs[(3, 13)]
    assert runs[(3, 4)]["status"] == "sha256 mismatch" and "episodes" not in runs[(3, 4)]
    assert [(e["k"], e["position"], e["length"]) for e in runs[(2, 8)]["episodes"]] == [(16, 1, 7), (23, 1, 7)]
    assert [e["start_idx"] for e in runs[(2, 9)]["episodes"]] == [127]
    assert runs[(1, 8)]["mode"] == "high" and runs[(3, 8)]["mode"] == "low"
    rep = s1["replication"]
    # 028 implementation note: C = 8 only (groups 8, 16, 24 of 1..24 = 1/8); at C = 9 a start 9k+1 is never 1 (mod 64)
    # below N = 200, so C = 9 starts are reported descriptively, never against a chance rate
    assert rep["C"] == [8] and rep["mod"] == 64 and rep["residue"] == 1
    assert rep["episodes"] == 3 and rep["at"] == 2 and rep["share"] == pytest.approx(2 / 3) and rep["chance"] == 1 / 8
    assert rep["binomial_tail_p"] == pytest.approx(ca.binomial_tail(3, 2, 1 / 8))
    assert rep["start_idx"] == [65, 129, 185]
    assert s1["c9_start_idx"] == [127]


def test_s2_asks_whether_elevated_c4_c7_runs_are_slip_driven(tmp_path):
    session, raw = make_core(tmp_path, tamper={(3, 4)})
    s1 = s0.s1_core(raw, s0.verify_raw(session, raw, s0.core_files()))
    runs = {(x["restart"], x["C"]): x for x in s0.s2_core(s1)["runs"]}
    assert set(runs) == {(r, C) for r in (1, 2, 3) for C in (4, 7)}
    slip = runs[(1, 4)]
    assert slip["elevated"] and slip["n_episodes"] == 4 and slip["slip_driven"] is True
    assert slip["y_without_slipped_minus_m_a_ms"] <= ca.SLIP_MS < slip["y_minus_m_a_ms"]
    flat = runs[(1, 7)]
    assert flat["elevated"] and flat["n_episodes"] == 0 and flat["slip_driven"] is False
    assert flat["y_minus_m_a_ms"] == pytest.approx(400.0)
    assert runs[(2, 4)]["elevated"] is False and runs[(2, 4)]["slip_driven"] is False
    assert runs[(3, 4)]["status"] == "sha256 mismatch" and runs[(3, 4)]["slip_driven"] is None


def test_s5_compares_the_core_r2_startup_with_r1_and_r3(tmp_path):
    session, raw = make_core(tmp_path)
    s5 = s0.s5_core_startup(raw, s0.verify_raw(session, raw, s0.core_files()))
    assert s5["only_in_r2"]["warning"] == [rp.normalize_line(SLOW)]
    assert all(v == [] for k, v in s5["only_in_r2"].items() if k != "warning")
    assert all(v == [] for v in s5["missing_in_r2"].values())
    assert s5["status"] == {f"r{r}": "ok" for r in (1, 2, 3)}


# ------------------------------------------------------------------ Advanced (S3, S4) --

ADV_SLIPS = {"G-8192-uniform-r1": (8,)}
GC_LINE = "(APIServer pid=11) INFO 10-11 00:00:00 [gc.py:1] gc: collected 1234 objects"


def adv_requests(slips):
    t, out = cycle(slips=slips), []
    for idx in range(200):
        k, p = divmod(idx, 8)
        tk = T0 + PERIOD * k
        if idx < 8:
            arrival = T0 + 0.001 * idx
        elif p == 0:
            arrival = tk
        else:
            arrival = tk + (0.092 if k in slips else 0.417) + 1.41 * (p - 1)
        out.append({"idx": idx, "cls": "short", "client_ttft_ms": t[idx], "arrival": arrival,
                    "Q_ms": 349.0 if k in slips and p >= 1 else 0.0, "prompt_tokens": 8064, "completion_tokens": 128})
    return out


def adv_steps(slips):
    out = []
    for j in range(int(25 * PERIOD / 0.35)):
        t = T0 + 0.35 * j
        k = int((t - T0) // PERIOD)
        out.append({"t": t, "index": j, "ctx_req": 1, "ctx_tok": 2048, "gen_req": 7, "gen_tok": 7,
                    "elapsed_ms": 349.0 if k in slips else 345.0, "u": 50.0})
    return out


def adv_server_log(steps, reqs, slips, sizes):
    lines = [(T0 - 60, f"(EngineCore pid=12) INFO 10-11 00:00:00 [gpu_model_runner.py:4000] Capturing CUDA graphs "
                       f"(mixed prefill-decode, PIECEWISE): sizes {sizes}"),
             (T0 - 59, "(APIServer pid=11) INFO:     Application startup complete.")]
    lines += [(s["t"], f"(APIServer pid=11) INFO 10-11 00:00:00 [loggers.py:185] Engine 000: Iteration({s['index']}): "
                       "1 context requests, 2048 context tokens, 7 generation requests, 7 generation tokens, "
                       f"iteration elapsed time: {s['elapsed_ms']:.2f} ms, GPU KV cache usage: 50.0%") for s in steps]
    lines += [(T0 + 5 + 10 * n, "(APIServer pid=11) INFO 10-11 00:00:00 [loggers.py:200] Engine 000: Avg prompt "
                                "throughput: 1000.0 tokens/s") for n in range(32)]
    lines += [(reqs[8 * k + 1]["arrival"] + 0.2, GC_LINE) for k in slips]
    return "".join(f"{int(t * 1000)} {x}\n" for t, x in sorted(lines))


def make_adv(tmp_path, *, missing_steps=()):
    session = tmp_path / "results" / "advanced" / s0.ADV_SID
    raw = tmp_path / "fetched" / s0.ADV_SID
    (session / "derived").mkdir(parents=True)
    raw.mkdir(parents=True)
    for tag in s0.ADV_TAGS:
        slips = ADV_SLIPS.get(tag, ())
        reqs, steps = adv_requests(slips), adv_steps(slips)
        (session / "derived" / f"{tag}-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        (raw / f"{tag}.steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
        sizes = "[1, 2, 4, 8]" if slips else "[1, 2, 4]"
        (raw / f"{tag}.server.log").write_text(adv_server_log(steps, reqs, slips, sizes))
        bench = "progress 100%\n" + ("WARNING: client retried\n" if tag == "K3-8192-uniform-r2" else "")
        (raw / f"{tag}.bench.log").write_text(bench)
    write_manifest(session, raw)
    for tag in missing_steps:
        (raw / f"{tag}.steps.jsonl").unlink()
    return session, raw


def test_adv_file_list_covers_steps_and_logs_of_the_nine_runs():
    names = s0.adv_files()
    assert len(s0.ADV_TAGS) == 9 and len(names) == 27
    assert "K3-8192-uniform-r3.steps.jsonl" in names and "B0-8192-uniform-r2.bench.log" in names


def test_s3_compares_slip_group_steps_with_neighbours_and_the_extra_queue(tmp_path):
    session, raw = make_adv(tmp_path, missing_steps={"B0-8192-uniform-r1"})
    s3 = s0.s3_steps(session, raw, s0.verify_raw(session, raw, s0.adv_files()))
    assert s3["missing"] == ["B0-8192-uniform-r1"]
    (ep,) = s3["episodes"]
    assert ep["tag"] == "G-8192-uniform-r1" and ep["k"] == 8
    assert ep["slip_group"]["step_ms_median"] == 349.0 and ep["slip_group"]["n_steps"] > 30
    assert ep["neighbours"]["7"]["step_ms_median"] == 345.0 and ep["neighbours"]["9"]["step_ms_median"] == 345.0
    assert ep["extra_q_ms"] == pytest.approx(349.0) and ep["extra_q_over_step"] == pytest.approx(1.0)


def test_s4_lists_log_lines_near_slip_moments_and_startup_differences(tmp_path):
    session, raw = make_adv(tmp_path)
    s4 = s0.s4_logs(session, raw, s0.verify_raw(session, raw, s0.adv_files()))
    near = s4["near_slips"]
    assert near["window_s"] == s0.LOG_WINDOW_S and near["slip_groups"] == 1 and near["non_slip_groups"] == 9 * 24 - 1
    assert near["with"]["other"] == {"slip": 1, "non_slip": 0}
    assert near["with"]["iteration"] == {"slip": 1, "non_slip": 215}
    (line,) = [x for x in near["lines"] if "gc:" in x["line"]]
    assert line["tag"] == "G-8192-uniform-r1" and line["k"] == 8 and line["offset_s"] == pytest.approx(0.2, abs=2e-3)
    start = s4["startup"]
    assert start["high"] == ["G-8192-uniform-r1"] and len(start["low"]) == 8
    assert start["only_in_high"]["cuda_graph"] == [rp.normalize_line(
        "(EngineCore pid=12) INFO 10-11 00:00:00 [gpu_model_runner.py:4000] Capturing CUDA graphs "
        "(mixed prefill-decode, PIECEWISE): sizes [1, 2, 4, 8]")]
    assert len(start["only_in_low"]["cuda_graph"]) == 1 and start["only_in_high"]["warning"] == []
    assert s4["bench_logs"] == {"K3-8192-uniform-r2": ["WARNING: client retried"]}


def test_missing_raw_archive_is_recorded_and_the_checks_still_run(tmp_path):
    core, _ = make_core(tmp_path)
    adv, _ = make_adv(tmp_path)
    out = s0.run_step0(core_session=core, core_raw=tmp_path / "none" / "core", adv_session=adv,
                       adv_raw=tmp_path / "none" / "adv")
    assert set(out["files"]["core"].values()) == {"missing"} and set(out["files"]["adv"].values()) == {"missing"}
    assert out["S1"]["replication"]["episodes"] == 0 and out["S3"]["episodes"] == []
    assert len(out["S3"]["missing"]) == 9 and out["S4"]["startup"]["high"] == []
    assert out["decision"] == "028" and out["descriptive_only"] is True


def test_main_writes_step0_json_outside_phase_1_dirs(tmp_path, capsys):
    core, core_raw = make_core(tmp_path)
    adv, adv_raw = make_adv(tmp_path)
    args = ["--core-session", str(core), "--core-raw", str(core_raw), "--adv-session", str(adv),
            "--adv-raw", str(adv_raw)]
    out = tmp_path / "results" / "capacity" / "c0-step0"
    assert s0.main(args + ["--out", str(out)]) == 0
    got = json.loads((out / "step0.json").read_text())
    assert set(got) >= {"S1", "S2", "S3", "S4", "S5", "files"}
    assert got["S1"]["replication"]["at"] == 2
    for bad in (core / "step0", adv / "x", ROOT / "results" / "core_b" / "c0-step0",
                ROOT / "results" / "advanced" / "c0-step0"):
        assert s0.main(args + ["--out", str(bad)]) == 2
        assert not bad.exists()
    assert not (core / "analysis.json").exists() and not (adv / "step0.json").exists()


def test_list_files_prints_the_fetch_lists(capsys):
    assert s0.main(["--list-files", "core"]) == 0
    assert capsys.readouterr().out.split() == s0.core_files()
    assert s0.main(["--list-files", "adv"]) == 0
    assert capsys.readouterr().out.split() == s0.adv_files()
