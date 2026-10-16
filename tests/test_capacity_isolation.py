"""Decision 030 8.6 / 8.7 and 027 §3 regressions for the Stage C modules: they import only the standard library, the
Core and Advanced research modules and each other (never capserve, the C0 probe or the service), and a session plus
its report write nothing outside results/capacity/<session>/."""

import ast
import json
import os
import sys
from pathlib import Path

import pytest

import capacity_report as cr
from test_run_capacity import CORE_UNIFORM, World, formal_setup, run, write_evidence

ROOT = Path(__file__).resolve().parent.parent
# gpu_idle (decision 031: the GPU idle criterion and the spill guard of run validity (g)) is imported by run_capacity
# and capacity_report, so it is a Stage C module frozen with them (030 凍結), not a Phase 1 one
CAPACITY_MODULES = ("capacity_plan", "capacity_analysis", "run_capacity", "capacity_report", "gpu_idle")
# 030 8.6 凍結: the frozen Stage C modules only import Core and Advanced code (guarded by test_core_untouched /
# test_frozen_untouched), and one another
RESEARCH = {"core_b_plan", "core_b_analysis", "run_core_a", "run_core_b", "coreb_smoke", "adv_plan", "adv_decomp",
            "run_adv", *CAPACITY_MODULES}


def imports(path: Path) -> set:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("module", CAPACITY_MODULES)
def test_capacity_modules_import_only_stdlib_core_advanced_and_each_other(module):
    got = imports(ROOT / "benchmarks" / f"{module}.py")
    outside = sorted(m for m in got - RESEARCH if m not in sys.stdlib_module_names and m != "__future__")
    assert outside == [], f"{module} imports {outside}"
    assert "capserve" not in got and not any(m.startswith(("c0_", "run_c0")) for m in got)


def top_level_imports(path: Path) -> set:
    found = set()
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def research_dependencies(bench: Path) -> set:
    """The benchmarks/ modules the Stage C modules import (anywhere, e.g. run_capacity's lazy run_core_b), and what
    those import at module level, transitively; the Stage C modules themselves excluded."""
    local = {p.stem for p in bench.glob("*.py")}
    todo = [m for mod in CAPACITY_MODULES for m in imports(bench / f"{mod}.py") if m in local]
    seen = set()
    while todo:
        m = todo.pop()
        if m in seen or m in CAPACITY_MODULES:
            continue
        seen.add(m)
        todo += [x for x in top_level_imports(bench / f"{m}.py") if x in local]
    return seen


def test_every_research_module_stage_c_depends_on_is_byte_guarded():
    """030 8.6 凍結: the frozen Stage C modules import only Core and Advanced code guarded by test_core_untouched /
    test_frozen_untouched. A dependency outside both (run_core_a: sha256_json for the plan / confirm-plan sha256,
    env_problems / provenance_problems for session validity) could change Stage C after the freeze unnoticed."""
    from test_core_untouched import CORE_SHA256
    from test_frozen_untouched import FROZEN_SHA256

    deps = research_dependencies(ROOT / "benchmarks")
    assert {"run_core_a", "run_core_b", "core_b_plan", "run_adv"} <= deps
    guarded = {Path(p).stem for p in {**CORE_SHA256, **FROZEN_SHA256} if p.startswith("benchmarks/")}
    assert sorted(deps - guarded) == []


def test_the_import_check_catches_capserve_and_the_c0_probe(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("import json\nfrom capserve.profile import x\nimport run_c0_probe\n")
    got = imports(p)
    assert sorted(m for m in got - RESEARCH if m not in sys.stdlib_module_names) == ["capserve", "run_c0_probe"]


def tree(root: Path) -> dict:
    out = {}
    if root.exists():
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = Path(dirpath) / f
                out[p] = (p.stat().st_mtime_ns, p.stat().st_size)
    return out


def test_a_session_and_its_report_write_only_under_results_capacity(tmp_path, monkeypatch):
    import tomllib

    cfg = tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    c, p, plan, ds = formal_setup(tmp_path, cfg, pins)
    repo_results = {d: tree(ROOT / d) for d in ("results", "configs")}
    write_evidence(tmp_path / "repo")              # the freeze evidence exists before the session (030 impl. note)
    before = tree(tmp_path)
    out, _, _ = run(World(tmp_path), plan, c, p, mixed_datasets=ds)
    monkeypatch.setattr(cr, "ROOT", tmp_path / "repo")
    monkeypatch.setattr(cr, "load_configs", lambda: (c, p, "cd" * 32, "ab" * 32))
    assert cr.main(["--session", str(out), "--uniform-session", str(CORE_UNIFORM),
                    "--out", str(out / "analysis.json")]) == 0
    after = tree(tmp_path)
    changed = [f for f in after if before.get(f) != after[f]]
    assert changed and all(out in f.parents for f in changed), [str(f) for f in changed if out not in f.parents]
    assert json.loads((out / "analysis.json").read_text())["provenance"]["session_id"] == out.name
    assert {d: tree(ROOT / d) for d in ("results", "configs")} == repo_results     # the real repo: untouched
