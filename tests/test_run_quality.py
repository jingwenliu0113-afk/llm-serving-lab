"""Tests for quality/run_quality.py: the quality protocol of decision 005 as a
saved plan executed one llama-perplexity process per job, with KLD scratch
handled per docs/STORAGE_PLAN.md."""

import copy
import json
import stat
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

import run_quality as rq

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def plan(pins, **kw):
    args = dict(mode="pilot", n_thr=8, poll=0)
    args.update(kw)
    return rq.build_quality_plan(pins, **args)


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_job_order_per_size_follows_005(pins):
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    for mode, p, extra_kld, extra_mc in (("formal", frozen, 0, 0), ("pilot", pins, 2, 1)):
        jobs = plan(p, mode=mode)["jobs"]
        for size in [m["size"] for m in pins["core_a"]["models"]]:
            kinds = [j["kind"] for j in jobs if j["size"] == size]
            n = 6 + extra_kld
            assert kinds[:n] == ["kld_base", "kld_self"] + ["kld"] * (3 + extra_kld) + ["delete_kld"]
            assert Counter(kinds[n:])["mc"] == 4 * 3 + extra_mc and Counter(kinds[n:])["winogrande"] == 4


def test_ha5_pair_gets_per_chunk_ppl(pins):
    ppl = [(j["size"], j["point"]) for j in plan(pins)["jobs"] if j["kind"] == "ppl"]
    big, small = pins["core_a"]["models"][-1]["size"], pins["core_a"]["models"][0]["size"]
    q = pins["core_a"]["quants"]
    assert sorted(ppl) == sorted([(big, q[-1]), (small, q[0])])
    j = next(j for j in plan(pins)["jobs"] if j["kind"] == "ppl")
    assert arg(j["argv"], "--ppl-output-type") == "1"


def test_flags_match_005(pins):
    for j in plan(pins, n_thr=8, poll=50)["jobs"]:
        if j["kind"] == "delete_kld":
            continue
        a = j["argv"]
        assert a[0] == "{bin}/llama-perplexity"
        for flag, value in rq.QUALITY_FLAGS.items():
            want = "off" if (flag, j.get("variant")) == ("-fa", "fa_off") else value
            assert arg(a, flag) == want, (j["id"], flag)
        assert (arg(a, "-t"), arg(a, "--poll")) == ("8", "50")
        if j["kind"] in ("mc", "winogrande"):
            assert arg(a, "-np") == "16"


def test_kld_jobs_share_the_size_kld_file(pins):
    jobs = [j for j in plan(pins)["jobs"] if j["size"] == "4B"]
    base = next(j for j in jobs if j["kind"] == "kld_base")
    path = arg(base["argv"], "--kl-divergence-base")
    assert path == "{data}/kld/4B.kld"
    for j in jobs:
        if j["kind"] in ("kld", "kld_self"):
            assert arg(j["argv"], "--kl-divergence-base") == path and "--kl-divergence" in j["argv"]


def test_pilot_vs_formal_counts(pins):
    pilot = plan(pins)
    assert pilot["expected"]["n_chunks"] == 20
    assert all(arg(j["argv"], "--chunks") == "20" for j in pilot["jobs"] if j["kind"] in ("kld_base", "ppl"))
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    formal = plan(frozen, mode="formal")
    assert formal["expected"]["n_chunks"] == pins["quality"]["expected"]["wikitext_chunks"]
    assert all("--chunks" not in j["argv"] for j in formal["jobs"] if j["kind"] != "delete_kld")
    mmlu = next(j for j in formal["jobs"] if j.get("task") == "mmlu")
    assert mmlu["expect"]["n_tasks"] == pins["quality"]["expected"]["mmlu_tasks"]


def test_formal_requires_freeze(pins):
    with pytest.raises(rq.QualityError, match="freeze.mac"):
        plan(pins, mode="formal")


# --------------------------------------------------------------- runner --

FAKE_PPL = r'''
import os, sys
a = sys.argv
g = lambda f: a[a.index(f) + 1] if f in a else None
chunks = int(g("--chunks") or os.environ["FAKE_FULL_CHUNKS"])
if "--kl-divergence" in a:
    if os.environ.get("FAKE_KLD_FAIL") and "Q6_K" in g("-m"):
        print("kl_divergence: inconsistent vocabulary (1 vs 2)", file=sys.stderr); sys.exit(0)
    print(f"kl_divergence: computing over {chunks} chunks, n_ctx=512, batch_size=2048, n_seq=4")
    print("====== KL divergence statistics ======")
    print("Mean    KLD:   0.001400 ±   0.000010")
elif "--multiple-choice" in a or "--winogrande" in a:
    n = int(os.environ["FAKE_TASKS"])
    if "--winogrande" in a:
        for i in range(1, n + 1): print(f"{i}\t100.0000\t -1.0   -2.0  1  1")
        print(f"Final Winogrande score({n} tasks): 100.0000 +/- 0.0000", file=sys.stderr)
    else:
        for i in range(1, n + 1): print(f"{i}\t100.00000000")
        print("Final result: 100.0000 +/- 0.0000", file=sys.stderr)
else:
    if g("--kl-divergence-base"):
        open(g("--kl-divergence-base"), "wb").write(b"_logits_")
    print(f"perplexity: calculating perplexity over {chunks} chunks, n_ctx=512, batch_size=2048, n_seq=4", file=sys.stderr)
    for i in range(chunks): print(f"{i*512:8d}  9.0000  2.197225  0.100000")
    print("Final estimate: PPL = 9.0000 +/- 0.05000", file=sys.stderr)
'''


class Env:
    def __init__(self):
        self.calls = []

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        rec = {"runtime": {"llama_cpp": {"build_number": 11146, "commit": COMMIT}}, "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def run(tmp_path, monkeypatch, plan_obj, **kw):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "llama-perplexity"
    exe.write_text(f"#!{sys.executable}\n{FAKE_PPL}")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    data = tmp_path / "data"
    (data / "kld").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FAKE_FULL_CHUNKS", str(plan_obj["expected"]["n_chunks"]))
    monkeypatch.setenv("FAKE_TASKS", str(plan_obj["expected"]["pilot_tasks"]))
    kw.setdefault("path_exists", lambda p: False)
    kw.setdefault("disk_ok", lambda path, need_gib: {"ok": True})
    summary = rq.run_quality(plan_obj, bin_dir=bin_dir, models_dir=tmp_path / "models", data_dir=data,
                             out_dir=tmp_path / "out", capture_env=Env(), session_id="q1", **kw)
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    return summary, rows, data


def small(pins):
    p = copy.deepcopy(pins)
    p["core_a"]["models"] = p["core_a"]["models"][:1]
    return plan(p)


def test_run_all_ok_and_kld_deleted(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    assert summary["outcomes"]["error"] == 0, [r for r in rows if r["outcome"] != "ok"]
    deletions = [r for r in rows if r["kind"] == "delete_kld"]
    assert deletions and deletions[0]["deleted"] is True
    assert not (data / "kld" / "4B.kld").exists()
    mc = next(r for r in rows if r["kind"] == "mc")
    assert mc["metrics"]["n_tasks"] == 200 and sum(mc["metrics"]["correct"]) == 200


def test_failed_kld_keeps_the_file(tmp_path, monkeypatch, pins):
    monkeypatch.setenv("FAKE_KLD_FAIL", "1")
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    bad = [r for r in rows if r["kind"] == "kld" and r["outcome"] == "error"]
    assert bad and "inconsistent vocabulary" in bad[0]["reason"]
    deletion = next(r for r in rows if r["kind"] == "delete_kld")
    assert deletion["deleted"] is False
    assert (data / "kld" / "4B.kld").exists()


def test_disk_guard_failure_skips_kld_jobs(tmp_path, monkeypatch, pins):
    summary, rows, _ = run(tmp_path, monkeypatch, small(pins), disk_ok=lambda path, need_gib: {"ok": False})
    kld_rows = [r for r in rows if r["kind"] in ("kld_base", "kld_self", "kld")]
    assert all(r["outcome"] == "skipped" for r in kld_rows)
    assert any(r["outcome"] == "ok" for r in rows if r["kind"] == "mc")


def test_refuses_when_mac_lock_held_by_other(tmp_path, monkeypatch, pins):
    with pytest.raises(rq.QualityError, match="Mac heavy lock"):
        run(tmp_path, monkeypatch, small(pins), path_exists=lambda p: True)


def test_formal_requires_our_mac_lock(tmp_path, monkeypatch, pins):
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    frozen["core_a"]["models"] = frozen["core_a"]["models"][:1]
    with pytest.raises(rq.QualityError, match="with-mac-lock"):
        run(tmp_path, monkeypatch, plan(frozen, mode="formal"))


def test_quality_session_writes_a_raw_manifest(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    manifest = json.loads((tmp_path / "out" / "raw_manifest.json").read_text())
    assert sorted(manifest["files"]) == sorted(p.relative_to(tmp_path / "out").as_posix()
                                               for p in (tmp_path / "out" / "raw").iterdir())


def test_pilot_can_be_limited_to_available_sizes(pins):
    p = rq.build_quality_plan(pins, mode="pilot", n_thr=8, poll=0, sizes=["4B"])
    assert {j["size"] for j in p["jobs"]} == {"4B"}
    assert any(j["kind"] == "ppl" for j in p["jobs"])  # the 4B half of the H-A5 pair
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    with pytest.raises(rq.QualityError, match="all sizes"):
        rq.build_quality_plan(frozen, mode="formal", n_thr=8, poll=0, sizes=["4B"])


def test_quality_session_reports_env_validity(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    assert summary["env_valid"] is True and summary["env_flags"] == []
