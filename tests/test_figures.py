"""Tests for scripts/svg_chart.py (stdlib SVG line charts) and benchmarks/core_a_figures.py."""

import xml.etree.ElementTree as ET

import core_a_figures as cf
import svg_chart as sc


def test_line_chart_is_valid_svg_with_series_and_labels():
    svg = sc.line_chart(
        title="Peak memory vs context", x_label="context (tokens)", y_label="MiB",
        series={"Q8_0": [(2048, 5000), (8192, 5200), (32768, 6000)], "Q4_K_M": [(2048, 3000), (32768, 4000)]},
        log2_x=True,
    )
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    lines = [e for e in root.iter() if e.tag.endswith("polyline")]
    assert len(lines) == 2
    text = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("text"))
    for label in ("Peak memory vs context", "context (tokens)", "MiB", "Q8_0", "Q4_K_M", "2048", "32768"):
        assert label in text


def test_line_chart_handles_missing_points_and_single_series():
    svg = sc.line_chart(title="t", x_label="x", y_label="y", series={"a": [(1, 1.0)]})
    ET.fromstring(svg)


def test_memory_figure_from_pilot_rows():
    summary = lambda dev, host: {"memory": {"devices": [{"name": "MTL0", "self": dev}], "host": {"self": host},  # noqa: E731
                                            "other": []}}
    rows = [{"block": "memory", "kind": "fit", "outcome": "ok", "size": "4B", "quant": q, "L": L,
             "summary": summary(base + L // 64, 300), "external_memory": {"peak_footprint_bytes": (base + 500) * 2**20}}
            for q, base in (("Q8_0", 4300), ("Q4_K_M", 2600)) for L in (2048, 8192, 32768)]
    series = cf.memory_series(rows, platform_label="M4 Pro (Metal)")
    assert set(series) == {"4B-Q8_0", "4B-Q4_K_M"}
    assert series["4B-Q8_0"] == [(2048, 4300 + 32 + 300), (8192, 4300 + 128 + 300), (32768, 4300 + 512 + 300)]
    svg = cf.memory_figure(rows, platform_label="M4 Pro (Metal)")
    assert "llama.cpp memory breakdown" in svg and "M4 Pro (Metal)" in svg


# ------------------------------------------------- latency / frontier / Core B figures --

def test_decode_and_prefill_series_from_process_level_cells():
    from test_core_a_analysis import grid, ideal_decode, ideal_ttft

    rows = grid(ideal_decode, ideal_ttft)
    dec = cf.speed_series(rows, metric="decode")
    assert set(dec) == {f"{s}-{q}" for s in ("4B", "9B") for q in ("Q8_0", "Q6_K", "Q4_K_M")}
    assert [L for L, _ in dec["4B-Q8_0"]] == [2048, 8192, 32768]
    assert dec["4B-Q8_0"][0][1] > dec["4B-Q8_0"][-1][1]  # decode slows down with context
    pre = cf.speed_series(rows, metric="prefill")
    assert pre["9B-Q4_K_M"][0][1] > 0
    svg = cf.speed_figure(rows, metric="decode", platform_label="RTX 5070 Ti (CUDA)")
    assert "decode" in svg and "RTX 5070 Ti (CUDA)" in svg


def test_frontier_points_pair_kld_with_decode_speed():
    from test_core_a_analysis import grid, ideal_decode, ideal_ttft

    core_a_rows = grid(ideal_decode, ideal_ttft)
    quality_rows = [{"kind": "kld", "size": s, "point": q, "outcome": "ok", "metrics": {"kld_mean": k}}
                    for (s, q), k in {("4B", "Q8_0"): 0.002, ("4B", "Q4_K_M"): 0.03, ("9B", "Q8_0"): 0.001}.items()]
    quality_rows.append({"kind": "kld", "size": "4B", "point": "Q8_0", "outcome": "ok", "variant": "fa_off",
                         "metrics": {"kld_mean": 9.9}})
    pts = cf.frontier_points(core_a_rows, quality_rows, L=2048)
    assert set(pts) == {"4B-Q8_0", "4B-Q4_K_M", "9B-Q8_0"}  # only configs with both measurements
    assert pts["4B-Q8_0"][0][1] == 0.002 and pts["4B-Q4_K_M"][0][0] > pts["4B-Q8_0"][0][0]
    svg = cf.frontier_figure(core_a_rows, quality_rows, platform_label="M4 Pro (Metal)")
    assert "KLD" in svg and "wsl2-cuda" in svg  # 020: the figure says where quality was measured


def test_frontier_cli_needs_a_formal_rtx_quality_session(tmp_path, capsys):
    import json

    a, q = tmp_path / "a", tmp_path / "q"
    for d, meta in ((a, {"mode": "formal", "env_valid": True, "sentinel": {"drift": False}, "seed": 20261008,
                         "platform": "macos-metal"}),
                    (q, {"mode": "formal", "env_valid": True, "platform": "macos-metal"})):
        d.mkdir()
        (d / "session.json").write_text(json.dumps(meta))
        (d / "rows.jsonl").write_text("")
    assert cf.main(["frontier", "--session", str(a), "--quality-session", str(q), "--platform-label", "x",
                    "--out", str(tmp_path / "f.svg")]) == 2
    assert "wsl2-cuda" in capsys.readouterr().err and not (tmp_path / "f.svg").exists()


def test_latency_figures_only_from_valid_formal_sessions(tmp_path, capsys):
    import json

    (tmp_path / "plan.json").write_text(json.dumps({"jobs": []}))
    (tmp_path / "rows.jsonl").write_text("")
    (tmp_path / "session.json").write_text(json.dumps({"mode": "pilot", "env_valid": True,
                                                       "sentinel": {"drift": False}}))
    assert cf.main(["decode", "--session", str(tmp_path), "--platform-label", "x",
                    "--out", str(tmp_path / "d.svg")]) == 2
    assert "formal" in capsys.readouterr().err and not (tmp_path / "d.svg").exists()


def test_core_b_series_and_figure():
    import core_b_figures as bf

    cells = {f"uniform-{L}-{C}": {"n_valid": 3, "output_throughput": 100.0 * min(C, k), "p95_ttft_ms": 50.0 * C,
                                  "p95_tpot_ms": 10 + C, "kv_cache_usage_peak_max": 0.5, "preemptions_total": 0}
             for L, k in ((2048, 22), (8192, 8)) for C in (1, 2, 4, 8, 16, 32)}
    cells["mixed-2048-22"] = {"n_valid": 3, "output_throughput": 1.0, "p95_ttft_ms": 1.0, "p95_tpot_ms": 1.0}
    analysis = {"cells": cells, "hypotheses": {"H-B1": {"units": [{"L": 2048, "c_knee": 22}, {"L": 8192, "c_knee": 8}]}}}
    s, dropped = bf.series(analysis, "p95_ttft_ms")
    assert set(s) == {"L=2048", "L=8192"} and s["L=2048"][0] == (1, 50.0) and len(s["L=2048"]) == 6 and dropped == []
    svg = bf.figure(analysis, "p95_ttft_ms", platform_label="RTX 5070 Ti")
    assert "P95 TTFT" in svg and "C_knee" in svg and "L=8192" in svg



def test_core_b_figure_plots_only_cells_with_three_valid_restarts():
    import core_b_figures as bf

    cells = {f"uniform-2048-{C}": {"n_valid": 3, "p95_ttft_ms": 10.0 * C} for C in (1, 2, 4)}
    cells["uniform-2048-8"] = {"n_valid": 2, "p95_ttft_ms": 999.0}
    analysis = {"cells": cells, "hypotheses": {"H-B1": {"units": [
        {"L": 2048, "status": "not testable", "missing_c": [8]}]}}}
    s, dropped = bf.series(analysis, "p95_ttft_ms")
    assert s["L=2048"] == [(1, 10.0), (2, 20.0), (4, 40.0)] and dropped == ["L=2048 C=8 (n=2)"]
    svg = bf.figure(analysis, "p95_ttft_ms", platform_label="RTX")
    assert "not testable" in svg and "C=8" in svg
