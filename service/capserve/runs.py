"""Product-test run directories (decisions 027 §5, 029 "產品負載測試"): the only place capserve writes results.

- A run lives at <service_root>/runs/<id>, id = pt-<UTC yyyymmddThhmmssZ>[-<scenario>] (lowercase). Ids starting
  formal- (the research's formal sessions) or anything else are refused; there is no --formal option anywhere.
- The resolved path (symlinks followed) must be inside <service_root>/runs, outside the research directory
  (~/llm-serving-lab) and outside this repo's results/; an existing run is never overwritten.
- meta.json says purpose "product-test", formal false, research_valid false and "not a capacity measurement", with the
  release sha and the profile id / status. Research analyses accept only formal sessions written by research runners,
  so a product run can never become research evidence.
- update_meta adds fields (outcome, problems) but never changes the fixed ones (FIXED, the release and the profile
  included: a run's provenance is set once), and keeps meta.json's mtime at the run's start, so RTX IT-6's
  `find ... -newer runs/<id>/meta.json` covers the whole run.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

SCHEMA_VERSION = 1
PURPOSE = "product-test"
NOTE = "not a capacity measurement"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIRNAME = "runs"
RUN_ID = re.compile(r"pt-\d{8}T\d{6}Z(-[a-z0-9][a-z0-9-]*)?")
SCENARIO = re.compile(r"[a-z0-9][a-z0-9-]*")
FIXED = ("schema_version", "run_id", "kind", "purpose", "formal", "research_valid", "note", "created_utc", "release",
         "profile")   # the run's provenance too: a provisional run never becomes verified afterwards (027 §4)


class RunError(ValueError):
    pass


def _utc(now_utc: dt.datetime) -> dt.datetime:
    return now_utc.astimezone(dt.timezone.utc)


def new_run_id(now_utc: dt.datetime, scenario: str | None = None) -> str:
    rid = "pt-" + _utc(now_utc).strftime("%Y%m%dT%H%M%SZ")
    if scenario is None:
        return rid
    if not SCENARIO.fullmatch(scenario):
        raise RunError(f"scenario {scenario!r} must be lowercase letters, digits and dashes")
    return f"{rid}-{scenario}"


def runs_root(service_root) -> Path:
    return Path(service_root) / RUNS_DIRNAME


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def run_path_problems(run_id: str, *, service_root, research_dir, repo_root=REPO_ROOT) -> list:
    """Every rule the run <service_root>/runs/<run_id> breaks (empty: allowed)."""
    problems = []
    if run_id.startswith("formal-"):
        problems.append(f"run id {run_id!r}: formal- ids belong to research sessions, never to product tests")
    elif not RUN_ID.fullmatch(run_id):
        problems.append(f"run id {run_id!r} is not pt-<UTC yyyymmddThhmmssZ>[-<scenario>]")
    root = runs_root(service_root).resolve()
    path = (runs_root(service_root) / run_id).resolve()
    if not _inside(path, root) or path == root:
        problems.append(f"{path} is not inside {root}")
    research = Path(research_dir).resolve()
    if _inside(path, research) or _inside(root, research):
        problems.append(f"{path} is inside the research directory {research} (never written by product tests)")
    results = (Path(repo_root) / "results").resolve()
    if _inside(path, results) or _inside(root, results):
        problems.append(f"{path} is under {results} (research results; product outputs never go there)")
    if path.exists():
        problems.append(f"{path} exists; a run is never overwritten")
    return problems


def _dumps(meta: dict) -> str:
    return json.dumps(meta, indent=1, sort_keys=True) + "\n"


def create_run(*, service_root, research_dir, run_id: str, kind: str, release: str | None, profile: dict | None,
               now_utc: dt.datetime, extra: dict | None = None, repo_root=REPO_ROOT) -> Path:
    """Create <service_root>/runs/<run_id>/meta.json, or RunError listing every rule broken (nothing written)."""
    problems = run_path_problems(run_id, service_root=service_root, research_dir=research_dir, repo_root=repo_root)
    if problems:
        raise RunError("; ".join(problems))
    clash = sorted(set(extra or {}) & set(FIXED))
    if clash:
        raise RunError(f"extra fields {clash} would replace fixed meta fields")
    path = runs_root(service_root) / run_id
    runs_root(service_root).mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RunError(f"{path} exists; a run is never overwritten") from exc
    meta = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind, "purpose": PURPOSE, "formal": False,
            "research_valid": False, "note": NOTE, "release": release, "profile": profile,
            "created_utc": _utc(now_utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **(extra or {})}
    with open(path / "meta.json", "x") as f:
        f.write(_dumps(meta))
    return path


def read_meta(run_dir) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text())


def update_meta(run_dir, **fields) -> dict:
    """Merge fields into meta.json (atomic replace), keeping the fixed fields and the file's start mtime."""
    changed = sorted(set(fields) & set(FIXED))
    if changed:
        raise RunError(f"meta fields {changed} are fixed at creation")
    path = Path(run_dir) / "meta.json"
    st = path.stat()
    meta = {**json.loads(path.read_text()), **fields}
    tmp = path.with_name("meta.json.tmp")
    tmp.write_text(_dumps(meta))
    os.replace(tmp, path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    return meta
