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
