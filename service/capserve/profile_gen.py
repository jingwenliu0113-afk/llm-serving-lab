"""python -m capserve.profile_gen --session <Stage C session> --out <profile.json>   (M0: a refusing stub)

A verified profile is the only kind that may become a product limit (027 §4), and only this generator may write one,
from a formal, env-valid Stage C session analysed by the frozen Stage C modules (decision 029 "產生器"). It has no
override options (021 R1).
- Until configs/capacity.toml has [freeze] capacity = true (a missing file or key is false; 030 implementation
  note) it always exits 2: PREREGISTRATION §8 is not frozen, so no verified capacity can exist.
- The generator itself (recompute and compare the analysis, map each L to a class, refuse untestable classes,
  max_queue = max_concurrency) is milestone M2; in M0 it exits 2 even after the freeze. It never writes a file.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

CAPACITY_TOML = Path(__file__).resolve().parents[2] / "configs" / "capacity.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capserve.profile_gen",
                                     description="verified capacity profile from a frozen Stage C analysis (M2)")
    parser.add_argument("--session", required=True, help="results/capacity/<formal session>")
    parser.add_argument("--out", required=True, help="where the verified profile would be written")
    return parser


def option_names() -> set:
    return {s for a in _parser()._actions for s in a.option_strings if s not in ("-h", "--help")}


def capacity_frozen(path: Path = CAPACITY_TOML) -> bool:
    """[freeze] capacity is exactly true; a missing file or key is false. tomllib.TOMLDecodeError propagates."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return False
    return (tomllib.loads(text).get("freeze") or {}).get("capacity") is True


def main(argv: list | None = None, *, capacity_toml: Path = CAPACITY_TOML) -> int:
    _parser().parse_args(argv)
    try:
        frozen = capacity_frozen(capacity_toml)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"profile_gen: refused: cannot read {capacity_toml}: {exc}", file=sys.stderr)
        return 2
    if not frozen:
        print(f"profile_gen: refused: [freeze] capacity is not true in {capacity_toml} (a missing file or key is "
              "false). PREREGISTRATION §8 is not frozen, so no verified capacity exists yet (027 §4, 029); "
              "develop against a provisional profile with --allow-provisional.", file=sys.stderr)
        return 2
    print("profile_gen: refused: the verified-profile generator is milestone M2 (029): it needs the frozen Stage C "
          "modules and a valid formal C1 analysis.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
