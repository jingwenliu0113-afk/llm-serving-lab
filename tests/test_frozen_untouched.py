"""Decision 027 regression: Phase 2 (Stage C research, the serving service) never changes the code that produced the
committed Core / Advanced conclusions.

- Advanced: every Advanced module is byte-identical to the freeze commit 97a948c (024).
- Core analysis that test_core_untouched does not cover (Core A, quality, the deployment decision table): byte-identical
  to the commit 4cb751f that closed Phase 1 (026).

Digests were computed once with `git show <commit>:<path> | shasum -a 256` and are hard-coded, so this test needs no git.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_SHA256 = {
    # Advanced, at the freeze commit 97a948c
    "benchmarks/adv_plan.py": "0de4e462dc763895956bf856886130f669c0e0bf3b32e0e03fb445ffe647346f",
    "benchmarks/adv_decomp.py": "bad9bdbce3cf7e901ffc5295bb1acacef86db81f17287a0b30a3a381a4e58f77",
    "benchmarks/adv_analysis.py": "14a5dba205aaaadd5eb72611dc76d3b8c6b283c8bbb34801f6ed80076312b1b6",
    "benchmarks/adv_report.py": "fcacae10d06e7848eafdf8e6933a8a562e940174c6c4d65778c2b6b690dc8bf4",
    "benchmarks/adv_smoke.py": "471cba1de08b651155b4f969c605aa17c0999d2f9402b34be7b8599cecfef4c8",
    "benchmarks/run_adv.py": "08c89f82999be2c6d89a56b6ed0b5033b6e25ff3444c7df27e6bc5e5ccdb94f0",
    # Core A / quality / decision table, at 4cb751f (Phase 1 closed)
    "benchmarks/core_a_analysis.py": "434e32555805eafcf09a88c1006558dd0f7a34d9810520ed9d99c537243f3ba4",
    "benchmarks/core_a_report.py": "a70687d2705d35f35628fba5368c9cf0847f291eb41b178605b48280c247f2c2",
    "benchmarks/core_a_parse.py": "d2defe8604abbbd7721b60e852154b4e4746a32b1fd4fcabc5522814c80f5926",
    "quality/quality_analysis.py": "ad9b6d31c40fe8966fcdff2ae0bb1f35a77000ab44fbd36bf11d933e1a0957c8",
    "benchmarks/decision_table.py": "7e874d058671d30c538337f3e8543e39734a699f7b736ba399e753dd423127d1",
}


def frozen_drift(root: Path) -> list:
    """Frozen modules under root that are missing or whose bytes differ from their recorded commit."""
    return [path for path, want in sorted(FROZEN_SHA256.items())
            if not (root / path).is_file() or hashlib.sha256((root / path).read_bytes()).hexdigest() != want]


def test_frozen_modules_are_byte_identical():
    assert frozen_drift(ROOT) == [], "frozen Core / Advanced code changed (027: Phase 2 never modifies it)"


def test_drift_detects_an_edit_and_a_deletion(tmp_path):
    for path in FROZEN_SHA256:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    assert frozen_drift(tmp_path) == []
    (tmp_path / "benchmarks" / "adv_analysis.py").write_text("# edited\n")
    (tmp_path / "benchmarks" / "decision_table.py").unlink()
    assert frozen_drift(tmp_path) == ["benchmarks/adv_analysis.py", "benchmarks/decision_table.py"]
