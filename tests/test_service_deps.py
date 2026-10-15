"""Decision 027 §3: the serving service uses only packages already hash-locked for the WSL vLLM venv.

The `service` dependency group (pyproject.toml, resolved in uv.lock for the Mac dev environment) must resolve to a
closure in which every package (except Windows-only markers) appears in configs/vllm.lock.txt at the same version, so
the service runs on the RTX in the research venv read-only, with no new install there.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def vllm_lock_versions(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\;]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def service_closure(lock: dict) -> dict:
    """name -> version for the transitive closure of the root project's `service` group (non-Windows deps only)."""
    by_name = {p["name"]: p for p in lock["package"]}
    root = by_name["llm-serving-lab"]
    todo = [d for d in root["dev-dependencies"]["service"] if "win32" not in d.get("marker", "")]
    seen = {}
    while todo:
        name = todo.pop()["name"]
        if name in seen:
            continue
        pkg = by_name[name]
        seen[name] = pkg["version"]
        todo.extend(d for d in pkg.get("dependencies", []) if "win32" not in d.get("marker", ""))
    return seen


def mismatches(closure: dict, pins: dict) -> list:
    return sorted(f"{n}=={v} (vllm lock: {pins.get(n, 'absent')})" for n, v in closure.items() if pins.get(n) != v)


def test_service_closure_matches_the_vllm_lock():
    closure = service_closure(tomllib.loads((ROOT / "uv.lock").read_text()))
    assert {"fastapi", "starlette", "uvicorn", "httpx", "prometheus-client", "pydantic"} <= set(closure)
    assert mismatches(closure, vllm_lock_versions((ROOT / "configs" / "vllm.lock.txt").read_text())) == []


def test_mismatch_detection():
    pins = vllm_lock_versions("fastapi==0.136.3 \\\n    --hash=sha256:00\nhttpx==0.28.1 \\\n")
    assert pins == {"fastapi": "0.136.3", "httpx": "0.28.1"}
    assert mismatches({"fastapi": "0.136.3", "httpx": "0.27.0", "rich": "1.0"}, pins) == [
        "httpx==0.27.0 (vllm lock: 0.28.1)", "rich==1.0 (vllm lock: absent)"]
