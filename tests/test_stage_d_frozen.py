"""Decision 039 凍結 regression: after the Stage D freeze (2026-10-23), the Stage D modules and configs/stage_d.toml
stay byte-identical to the freeze commit.

- The modules are the ones the three mini smokes ran (deployed 91e6e73, harness_sha256 2b85155d...), so the freeze
  evidence check (stage_d_plan.freeze_evidence_problems) and this test agree.
- stage_d.toml is the frozen one (freeze.stage_d = true and the three mini ids): any rule change after the freeze needs
  its own decision log and a new session. This test catches such an edit before a GPU session runs on it.

Digests were computed once with `shasum -a 256` on the freeze commit's files and are hard-coded, so this test needs no
git.
"""

import hashlib
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_d_plan as sd  # noqa: E402

STAGE_D_SHA256 = {
    "benchmarks/stage_d_plan.py": "dee3be8ff53e7c765591fc67ce3f532b16b86dd5714e6f17d76beab84c40262e",
    "benchmarks/stage_d_analysis.py": "7dda6f4ffb411711c93b4414a313a77a366b502aa8b24e62a28c5bb0de6423b3",
    "benchmarks/run_stage_d.py": "62be1cbc01b92bbc1de41b4b9d077e73b116978f8a23a4c20b0cd7fd33cb4a47",
    "benchmarks/stage_d_report.py": "caa73a0facc33b898222e3598796c05ccd3d7d8036140b738a64a0d58853fee6",
    "benchmarks/stage_d_datasets.py": "c015f61d01ff851804b9bbbfc8bfd6c2bc9a5121ae3d66fa11045311fa7129f7",
    "configs/stage_d.toml": "be7909b1e98183c337182f9559953e2613cad96f06d03899815fc0af88cba2a5",
}


def stage_d_drift(root: Path) -> list:
    return [path for path, want in sorted(STAGE_D_SHA256.items())
            if not (root / path).is_file() or hashlib.sha256((root / path).read_bytes()).hexdigest() != want]


def test_stage_d_modules_and_config_are_the_frozen_bytes():
    assert stage_d_drift(ROOT) == [], "Stage D changed after the freeze (039 凍結: needs a decision log and a new session)"


def test_every_stage_d_module_is_guarded():
    assert set(sd.HARNESS_FILES) | {"configs/stage_d.toml"} == set(STAGE_D_SHA256)


def test_the_frozen_config_names_the_passed_minis_of_this_harness():
    cfg = tomllib.loads((ROOT / "configs" / "stage_d.toml").read_text())
    assert cfg["freeze"]["stage_d"] is True
    assert sd.freeze_evidence_problems(cfg, ROOT) == []
    assert sd.harness_sha256().startswith("2b85155d")


def test_drift_detects_an_edit(tmp_path):
    for path in STAGE_D_SHA256:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    assert stage_d_drift(tmp_path) == []
    toml = tmp_path / "configs" / "stage_d.toml"
    toml.write_bytes(toml.read_bytes().replace(b"guard_band = 0.05", b"guard_band = 0.04"))
    (tmp_path / "benchmarks" / "stage_d_datasets.py").unlink()
    assert stage_d_drift(tmp_path) == ["benchmarks/stage_d_datasets.py", "configs/stage_d.toml"]
