"""Decision 024 regression: the Advanced harness imports the Core modules and never changes them.

Each Core module's bytes must hash to what the file was at 68f6a65 (the commit that entered Advanced, before any
Advanced code). The digests below were computed once with
    git show 68f6a65:<path> | shasum -a 256
and are hard-coded, so this test needs no git and runs anywhere the checkout does.
"""

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE_ADVANCED_COMMIT = "68f6a65"
CORE_SHA256 = {
    "benchmarks/core_b_plan.py": "69b0f38af0e2b905819f8467dd220b7d44fee0128b3ff5466cbc7a7cdc513052",
    "benchmarks/run_core_b.py": "6091159cd16e84ae832a7d6fd39e0e72963299931dd213727189a6893353a318",
    "benchmarks/core_b_analysis.py": "5d94c5b9c806e860c111e73fa84e650a8a3b5281b402e53cc6378421f04b4ca6",
    "benchmarks/core_b_report.py": "736ec0c3234e3277389d6cb19ea6f27e6bfcaa2fe1352d4a6ae52e02310e259f",
    "benchmarks/coreb_smoke.py": "8ca331958bf6ceeabdaa138026e0429a383b11bad205f88699c54074d718b49a",
    "scripts/env_capture.py": "9508fdf9c44ccc6b798b52f15809ebf507df291d932cc0876d4d2b447471480f",
}


def core_drift(root: Path) -> list:
    """Core modules under root that are missing or whose bytes differ from the pre-Advanced commit."""
    return [path for path, want in sorted(CORE_SHA256.items())
            if not (root / path).is_file() or hashlib.sha256((root / path).read_bytes()).hexdigest() != want]


def test_core_modules_are_byte_identical_to_the_pre_advanced_commit():
    assert core_drift(ROOT) == [], f"Core modules changed since {PRE_ADVANCED_COMMIT} (024: never modify Core)"


def test_the_six_024_core_modules_are_covered():
    assert set(CORE_SHA256) == {"benchmarks/core_b_plan.py", "benchmarks/run_core_b.py",
                                "benchmarks/core_b_analysis.py", "benchmarks/core_b_report.py",
                                "benchmarks/coreb_smoke.py", "scripts/env_capture.py"}
    assert all(len(v) == 64 and int(v, 16) >= 0 for v in CORE_SHA256.values())


def test_drift_check_catches_a_one_byte_edit_and_a_missing_file(tmp_path):
    for path in CORE_SHA256:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    assert core_drift(tmp_path) == []
    edited = tmp_path / "benchmarks" / "run_core_b.py"
    edited.write_bytes(edited.read_bytes() + b"\n")
    (tmp_path / "scripts" / "env_capture.py").unlink()
    assert core_drift(tmp_path) == ["benchmarks/run_core_b.py", "scripts/env_capture.py"]
