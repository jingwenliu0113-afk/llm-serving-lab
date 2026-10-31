"""Decision 044 凍結 regression: after the Stage V freeze (2026-10-31), the five Stage V modules and
configs/stage_v.toml stay byte-identical to the freeze commit, and the request path stays the frozen one.

- The modules are the ones the mini smoke smoke-v-20261031 ran (deployed b792196, harness_sha256 a1e4a564...), so the
  freeze evidence check (stage_v_plan.freeze_evidence_problems) and this test agree.
- stage_v.toml is the frozen one (freeze.stage_v = true, the mini id and gateway_path_sha256): any rule change after
  the freeze needs its own decision log, a new mini and new sessions. This test catches such an edit before a GPU
  session runs on it.
- 044 凍結: from the freeze to the end of V-CLS, the files in the request path's closure (capserve/__main__.py's
  imports, stage_v_plan.gateway_path_files) do not change on main; gateway_path_problems checks them here.

Digests were computed once with `shasum -a 256` on the freeze commit's files and are hard-coded, so this test needs no
git.
"""

import hashlib
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_v_plan as sv  # noqa: E402

STAGE_V_SHA256 = {
    "benchmarks/stage_v_plan.py": "6224bddd699f13412ffea68626ed9601f85c8f9ffd934508b03ec310bd1d95e1",
    "benchmarks/stage_v_datasets.py": "c64146f0d16f034d0b2eb7999b67c7bb3e4f4462507cee9e957882379fdb84f0",
    "benchmarks/stage_v_analysis.py": "1ca55dcab4615b3e7a46d09edfdd4b4429264fae1b2daad2a8fcc7ea255fd2d5",
    "benchmarks/run_stage_v.py": "239acc139d90cf36ef14e7cd63cf00652e0e8492b963061417489feb014140b8",
    "benchmarks/stage_v_report.py": "a8bf4cfb29b2e191785f750812c14f4dbd770d949a9e093537189d1272c36da7",
    "configs/stage_v.toml": "fe1fb09415bc3ee56a5fa5187f52db5733cded86b87aca0dda54452fd060f02d",
}
GATEWAY_PATH_SHA256 = "679a7c661ec36cbca1e472ddcad80f590981ad6fdebb49ef8ef58e69fee5a259"


def cfg() -> dict:
    return tomllib.loads((ROOT / "configs" / "stage_v.toml").read_text())


def stage_v_drift(root: Path) -> list:
    return [path for path, want in sorted(STAGE_V_SHA256.items())
            if not (root / path).is_file() or hashlib.sha256((root / path).read_bytes()).hexdigest() != want]


def test_stage_v_modules_and_config_are_the_frozen_bytes():
    assert stage_v_drift(ROOT) == [], "Stage V changed after the freeze (044 凍結: needs a decision log and a new mini)"


def test_every_stage_v_module_is_guarded():
    assert set(sv.HARNESS_FILES) | {"configs/stage_v.toml"} == set(STAGE_V_SHA256)


def test_the_frozen_config_names_the_passed_mini_of_this_harness():
    c = cfg()
    assert c["freeze"] == {"stage_v": True, "mini": "smoke-v-20261031", "gateway_path_sha256": GATEWAY_PATH_SHA256}
    assert sv.freeze_evidence_problems(c, ROOT) == []
    assert sv.harness_sha256().startswith("a1e4a564")


def test_the_request_path_and_the_profiles_are_the_frozen_bytes():
    c = cfg()
    assert sv.gateway_path_problems(c, ROOT) == [], "044 凍結: the request path changed before V-CLS ended"
    assert sv.gateway_path_sha256(ROOT) == GATEWAY_PATH_SHA256
    assert sv.profile_problems(c, ROOT) == []


def test_drift_detects_an_edit(tmp_path):
    for path in STAGE_V_SHA256:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    assert stage_v_drift(tmp_path) == []
    toml = tmp_path / "configs" / "stage_v.toml"
    toml.write_bytes(toml.read_bytes().replace(b"guard_band = 0.05", b"guard_band = 0.04"))
    (tmp_path / "benchmarks" / "stage_v_report.py").unlink()
    assert stage_v_drift(tmp_path) == ["benchmarks/stage_v_report.py", "configs/stage_v.toml"]
