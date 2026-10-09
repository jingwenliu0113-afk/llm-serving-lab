"""Tests for quality/quality_analysis.py: H-A5 (decision 005 / PREREGISTRATION §2) and the
descriptive downstream comparisons, written before any quality data exists."""

import pytest

import quality_analysis as qa


def test_kld_z_test_two_sided():
    t = qa.kld_test("4B", "Q8_0", (0.0014, 0.00001), "Q6_K", (0.0055, 0.00003))
    assert t["z"] == pytest.approx((0.0055 - 0.0014) / (0.00001 ** 2 + 0.00003 ** 2) ** 0.5)
    assert t["p"] < 1e-12 and t["direction_ok"] is True
    flat = qa.kld_test("4B", "Q8_0", (0.0014, 0.0005), "Q6_K", (0.0015, 0.0005))
    assert flat["p"] == pytest.approx(2 * (1 - qa.NormalDist().cdf(0.1 / (2 ** 0.5 * 0.5))), rel=1e-9)


def test_paired_bootstrap_is_deterministic_and_floored():
    a = [2.0 - 0.01 * (i % 7) for i in range(580)]  # 9B-Q4_K_M: always a bit lower
    b = [2.05] * 580
    r1 = qa.paired_nll_test(a, b, n_boot=2000, seed=20261008)
    r2 = qa.paired_nll_test(a, b, n_boot=2000, seed=20261008)
    assert r1 == r2
    assert r1["mean_delta"] < 0 and r1["direction_ok"] is True
    assert r1["p"] == pytest.approx(1 / 2000)  # floor at 1 / n_boot
    assert r1["ci95"][0] < r1["mean_delta"] < r1["ci95"][1] < 0


def test_paired_bootstrap_null_gives_large_p():
    a = [((i * 7919) % 13) / 100 for i in range(580)]
    b = list(reversed(a))
    assert qa.paired_nll_test(a, b, n_boot=2000, seed=1)["p"] > 0.2


def test_holm_step_down():
    res = qa.holm({"t1": 0.01, "t2": 0.04, "t3": 0.03, "t4": 0.005, "t5": 0.2}, alpha=0.05)
    # sorted: 0.005 (<= .01), 0.01 (<= .0125), 0.03 (> .01667 -> stop)
    assert res["t4"]["reject"] and res["t1"]["reject"]
    assert not res["t3"]["reject"] and not res["t2"]["reject"] and not res["t5"]["reject"]
    assert res["t4"]["p_adj"] == pytest.approx(0.025) and res["t1"]["p_adj"] == pytest.approx(0.04)
    assert res["t3"]["p_adj"] == pytest.approx(0.09)


def kld_rows(vals):
    return [{"kind": "kld", "size": s, "point": q, "outcome": "ok",
             "metrics": {"kld_mean": m, "kld_mean_se": se, "n_chunks": 580}} for (s, q), (m, se) in vals.items()]


GOOD_KLD = {("4B", "Q8_0"): (0.0020, 0.00002), ("4B", "Q6_K"): (0.0070, 0.00004), ("4B", "Q4_K_M"): (0.030, 0.0001),
            ("9B", "Q8_0"): (0.0012, 0.00001), ("9B", "Q6_K"): (0.0045, 0.00003), ("9B", "Q4_K_M"): (0.020, 0.0001)}


def ppl_rows(a, b):
    return [{"kind": "ppl", "size": "9B", "point": "Q4_K_M", "outcome": "ok", "metrics": {"chunk_nll": a}},
            {"kind": "ppl", "size": "4B", "point": "Q8_0", "outcome": "ok", "metrics": {"chunk_nll": b}}]


def test_h_a5_supported_only_if_all_five_survive_holm_in_direction():
    a = [2.0 - 0.01 * (i % 7) for i in range(580)]
    rows = kld_rows(GOOD_KLD) + ppl_rows(a, [2.05] * 580)
    res = qa.h_a5(rows, n_boot=2000, seed=20261008)
    assert len(res["tests"]) == 5 and res["verdict"] == "not refuted"
    bad = dict(GOOD_KLD)
    bad[("9B", "Q6_K")] = (0.0011, 0.00001)  # Q6_K below Q8_0: significant but wrong direction
    res = qa.h_a5(kld_rows(bad) + ppl_rows(a, [2.05] * 580), n_boot=2000, seed=20261008)
    assert res["verdict"] == "refuted"


def test_h_a5_missing_cell_is_not_testable_not_refuted():
    rows = kld_rows({k: v for k, v in GOOD_KLD.items() if k != ("9B", "Q4_K_M")})
    res = qa.h_a5(rows + ppl_rows([1.0] * 580, [1.1] * 580), n_boot=200, seed=1)
    assert res["verdict"] == "not testable"


def mc(task, size, point, correct):
    return {"kind": "mc", "task": task, "size": size, "point": point, "outcome": "ok",
            "metrics": {"correct": correct, "n_tasks": len(correct)}}


def test_downstream_pairs_ci_and_flip_rate():
    n = 400
    bf16 = [1] * 300 + [0] * 100
    q4 = [1] * 260 + [0] * 140  # 40 flips 1 -> 0
    rows = [mc("mmlu", "4B", "BF16", bf16), mc("mmlu", "4B", "Q4_K_M", q4)]
    out = qa.downstream(rows, n_boot=2000, seed=20261008)
    d = next(x for x in out if x["pair"] == ["BF16", "Q4_K_M"])
    assert d["delta_acc"] == pytest.approx(-40 / n) and d["flip_rate"] == pytest.approx(40 / n)
    assert d["resolved"] is True and d["ci95"][1] < 0
    same = qa.downstream([mc("arc", "4B", "BF16", bf16), mc("arc", "4B", "Q8_0", list(bf16))], n_boot=500, seed=1)
    assert same[0]["resolved"] is False and same[0]["flip_rate"] == 0


def test_quality_report_cli_writes_analysis(tmp_path):
    import json

    a = [2.0 - 0.01 * (i % 7) for i in range(580)]
    rows = kld_rows(GOOD_KLD) + ppl_rows(a, [2.05] * 580) + [
        mc("mmlu", "4B", "BF16", [1, 0] * 200), mc("mmlu", "4B", "Q8_0", [1, 0] * 200)]
    for i, r in enumerate(rows):
        r["job_id"] = f"j{i}"
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "session.json").write_text(json.dumps({"session_id": "q", "mode": "formal", "env_valid": True,
                                                       "platform": "wsl2-cuda"}))
    assert qa.main(["--session", str(tmp_path), "--n-boot", "200"]) == 0
    out = json.loads((tmp_path / "analysis.json").read_text())
    assert out["h_a5"]["verdict"] == "not refuted" and out["downstream"][0]["pair"] == ["BF16", "Q8_0"]
    assert out["session"] == "q" and out["env_valid"] is True and out["platform"] == "wsl2-cuda"


@pytest.mark.parametrize("meta, reason", [
    ({"session_id": "p", "mode": "pilot", "env_valid": True, "platform": "wsl2-cuda"}, "formal"),
    ({"session_id": "f", "mode": "formal", "env_valid": False, "platform": "wsl2-cuda"}, "env-invalid"),
    ({"session_id": "f", "mode": "formal", "platform": "wsl2-cuda"}, "env-invalid"),
    # 020: formal quality is measured on wsl2-cuda only; an old session without a platform is a Mac session
    ({"session_id": "f", "mode": "formal", "env_valid": True, "platform": "macos-metal"}, "020"),
    ({"session_id": "f", "mode": "formal", "env_valid": True}, "020"),
])
def test_quality_cli_refuses_pilot_and_env_invalid_sessions(tmp_path, capsys, meta, reason):
    """005 / 011: the pilot only reports b + c and N; an env-invalid session is rerun (012, 013)."""
    import json

    rows = kld_rows(GOOD_KLD) + ppl_rows([1.0] * 580, [1.1] * 580) + [
        mc("mmlu", "4B", "BF16", [1, 0] * 200), mc("mmlu", "4B", "Q8_0", [1, 0] * 200)]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "session.json").write_text(json.dumps(meta))
    assert qa.main(["--session", str(tmp_path), "--n-boot", "200"]) == 2
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "analysis.json").exists()
