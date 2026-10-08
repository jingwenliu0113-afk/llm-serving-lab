"""Tests for benchmarks/core_b_report.py: one analysis.json from a formal uniform session and
the formal mixed session derived from it (PREREGISTRATION §2 H-B1..H-B4; 007; 013)."""

import json

import pytest

import core_b_plan as bp
import core_b_report as br

GRIDS = {"2048": [1, 2, 4, 8, 11, 16, 22, 31, 32, 44, 64, 128, 256], "8192": [1, 2, 4, 6, 8, 11, 16, 32, 64],
         "32768": [1, 2, 3, 4, 8, 16]}
C_KV = {"2048": 22, "8192": 8, "32768": 2}
KNEE = {2048: 22, 8192: 8, 32768: 2}


def uniform_rows():
    rows = []
    for L, grid in GRIDS.items():
        L, k = int(L), KNEE[int(L)]
        for r in (1, 2, 3):
            for C in grid:
                eps = 1 + 0.001 * (r - 2)
                rows.append({"L": L, "mix": "uniform", "restart": r, "C": C, "ok": True,
                             "output_throughput": 100.0 * min(C, k) * eps,
                             "p95_ttft_ms": (100.0 if C <= k else 100.0 * C / k) * eps,
                             "p95_tpot_ms": (10 + C / 20) * eps, "short_class_p95_ttft_ms": (100.0 + r),
                             "metrics_sampled": {"kv_cache_usage_peak": min(1.0, C / (2 * k)), "preemptions": 0}})
    return rows


def mixed_rows(increase=1.3):
    return [{"L": L, "mix": "mixed", "restart": r, "C": KNEE[L], "ok": True, "output_throughput": 1.0,
             "p95_ttft_ms": 1.0, "p95_tpot_ms": 1.0, "short_class_p95_ttft_ms": (100.0 + r) * increase,
             "metrics_sampled": {"kv_cache_usage_peak": 0.9, "preemptions": 3}}
            for L in bp.MIXED_LEVELS for r in (1, 2, 3)]


def uniform_plan():
    return {"schema_version": 1, "seed": 20261008, "num_gpu_blocks": 160, "block_size": 528, "c_kv": C_KV,
            "grids": GRIDS, "runs": []}


def session(sid, plan, rows, formal=True):
    return {"session": {"session_id": sid, "formal": formal}, "plan": plan, "rows": rows}


def mixed_session(**kw):
    plan = bp.build_mixed_plan(uniform_plan(), KNEE, uniform_session="b-u")
    return session("b-m", plan, mixed_rows(**kw))


def test_report_computes_all_four_hypotheses():
    rep = br.report(session("b-u", uniform_plan(), uniform_rows()), mixed_session())
    assert set(rep["hypotheses"]) == {"H-B1", "H-B2", "H-B3", "H-B4"}
    assert rep["hypotheses"]["H-B1"]["verdict"] == "not refuted"
    assert rep["hypotheses"]["H-B2"]["verdict"] == "not refuted"
    b4 = rep["hypotheses"]["H-B4"]
    assert [u["L"] for u in b4["units"]] == [2048, 8192] and b4["verdict"] == "not refuted"
    assert b4["units"][0]["increase_pct"] == pytest.approx(30.0)
    cell = rep["cells"]["uniform-2048-22"]
    assert cell["n_valid"] == 3 and cell["output_throughput"] == pytest.approx(2200.0)
    assert rep["cells"]["mixed-8192-8"]["preemptions_total"] == 9


def test_h_b4_small_increase_refutes():
    rep = br.report(session("b-u", uniform_plan(), uniform_rows()), mixed_session(increase=1.1))
    assert rep["hypotheses"]["H-B4"]["verdict"] == "refuted"


def test_without_a_mixed_session_h_b4_is_not_testable():
    rep = br.report(session("b-u", uniform_plan(), uniform_rows()), None)
    assert all(u["status"] == "not testable" for u in rep["hypotheses"]["H-B4"]["units"])


@pytest.mark.parametrize("change, match", [
    (lambda u, m: u["session"].update(formal=False), "formal"),
    (lambda u, m: m["session"].update(formal=False), "formal"),
    (lambda u, m: m["plan"].update(derived_from="other"), "derived"),
    (lambda u, m: m["plan"]["c_knee"].update({"2048": 16}), "C_knee"),
])
def test_report_refuses_inconsistent_or_non_formal_sessions(change, match):
    u, m = session("b-u", uniform_plan(), uniform_rows()), mixed_session()
    change(u, m)
    with pytest.raises(br.ReportError, match=match):
        br.report(u, m)


def test_cli_writes_analysis_json(tmp_path):
    def write(d, s):
        d.mkdir()
        (d / "session.json").write_text(json.dumps(s["session"]))
        (d / "plan.json").write_text(json.dumps(s["plan"]))
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s["rows"]))

    write(tmp_path / "u", session("b-u", uniform_plan(), uniform_rows()))
    write(tmp_path / "m", mixed_session())
    out = tmp_path / "analysis.json"
    assert br.main(["--uniform", str(tmp_path / "u"), "--mixed", str(tmp_path / "m"), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["uniform_session"] == "b-u" and rep["mixed_session"] == "b-m"
    (tmp_path / "m" / "session.json").write_text(json.dumps({"session_id": "b-m", "formal": False}))
    assert br.main(["--uniform", str(tmp_path / "u"), "--mixed", str(tmp_path / "m"),
                    "--out", str(tmp_path / "x.json")]) == 2
    assert not (tmp_path / "x.json").exists()
