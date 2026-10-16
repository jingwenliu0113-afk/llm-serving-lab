"""Decision 030 凍結 regression: after the Stage C freeze (2026-10-17, decision 032 for C0), the Stage C modules and
configs/capacity.toml stay byte-identical to the freeze commit.

- The modules are the ones that passed the mini and FS-order smokes (deployed f0eee6a; unchanged since).
- capacity.toml is the frozen one: the formal session records its sha256 and capserve.profile_gen refuses a profile
  when it differs from the freeze commit's (030 8.9: any rule change after the freeze needs its own decision log and a
  new session). This test catches such an edit before a 7-9 h GPU session runs on it.

Digests were computed once with `shasum -a 256` on the freeze commit's files and are hard-coded, so this test needs no
git.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE_C_SHA256 = {
    "benchmarks/capacity_plan.py": "d13c445842de1f6be75b353b4cbefbdaf66a1997f7a7393ab8a067781eb2b4de",
    "benchmarks/capacity_analysis.py": "7b70b9f2480b0d87889c4faf320b7f1d8f7abe3589c83c67913c9c02e6e1071e",
    "benchmarks/run_capacity.py": "6d4bac8bec0f645bba3ec273dd5793dc2940193d3293512c25170a2ab7446554",
    "benchmarks/capacity_report.py": "b6309b9c2acc1f533dd6d0d5cbf1daa985133a857b344cfd0a89c9177f3ce710",
    # decision 031: the idle criterion and the spill guard of run validity (g), imported by run_capacity / the report
    "benchmarks/gpu_idle.py": "b04872c7d7ee2592fc94cbd4e631c27def45dc1e041dbb0e6c0bbb189c3e3b53",
    "configs/capacity.toml": "b7c4a6c68bdd7b39857b55f2cb6f00feb0e0fc134334263a22bb5284da843e21",
}


def stage_c_drift(root: Path) -> list:
    return [path for path, want in sorted(STAGE_C_SHA256.items())
            if not (root / path).is_file() or hashlib.sha256((root / path).read_bytes()).hexdigest() != want]


def test_stage_c_modules_and_config_are_the_frozen_bytes():
    assert stage_c_drift(ROOT) == [], "Stage C changed after the freeze (030 8.9: needs a decision log and a new session)"


def test_every_stage_c_module_is_guarded():
    from test_capacity_isolation import CAPACITY_MODULES

    assert {f"benchmarks/{m}.py" for m in CAPACITY_MODULES} | {"configs/capacity.toml"} == set(STAGE_C_SHA256)


def test_drift_detects_an_edit(tmp_path):
    for path in STAGE_C_SHA256:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    assert stage_c_drift(tmp_path) == []
    toml = tmp_path / "configs" / "capacity.toml"
    toml.write_bytes(toml.read_bytes().replace(b"guard_band = 0.05", b"guard_band = 0.04"))
    (tmp_path / "benchmarks" / "gpu_idle.py").unlink()
    assert stage_c_drift(tmp_path) == ["benchmarks/gpu_idle.py", "configs/capacity.toml"]
