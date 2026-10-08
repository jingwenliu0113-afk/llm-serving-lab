#!/usr/bin/env python3
"""Produce the Core A GGUF artifacts for one size (decision 004 / 008).

    BF16 GGUF   = convert_hf_to_gguf.py <pinned snapshot> --outtype bf16 --no-mtp
    Q8_0/Q6_K/Q4_K_M = llama-quantize <BF16> <out> <Q>     (no imatrix)

Before anything runs: the Mac heavy lock must not be held by another job, the
llama.cpp build and source tree must match pins.toml [llama_cpp], the source
snapshot must contain the required small files, its shards must match the
pinned sha256, and disk_guard must pass. The lock is re-checked before every
step. Each step writes into models/gguf/.tmp/ and is moved into place only if
its sha256 agrees with any existing pin, so a re-run never clobbers a pinned
artifact; steps whose pinned output and recipe are already on disk are skipped.
Every output gets a recipe results/artifacts/<filename>.json (inputs incl. all
small source files, exact argv, toolchain, output hash) and its sha256/size in
pins.toml, so scratch tiers can be deleted and rebuilt bit-for-bit
(docs/STORAGE_PLAN.md).

Deleting the source safetensors is a separate command. It re-verifies every
derived GGUF on disk against pins and recipe, refuses unless pins.toml and the
recipes are committed, and appends the deletion to results/artifacts/deletions.jsonl.

Usage:
    uv run python scripts/produce_gguf.py produce --size 4B
    git add configs/pins.toml results/artifacts && git commit
    uv run python scripts/produce_gguf.py delete-source --size 4B   # then commit deletions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_guard  # noqa: E402
import env_capture  # noqa: E402

GiB = 1024**3
MAC_HEAVY_LOCK = "/tmp/ai-portfolio-mac-heavy.lock"
# BF16 GGUF ~ source shard bytes; Q8_0 + Q6_K + Q4_K_M ~ 1.24 x BF16 (decision 008 sizes).
NEED_FACTOR = 2.3
# Read by convert_hf_to_gguf.py; README.md feeds general.* metadata (gguf-py metadata.py),
# so both sizes must be converted from the same small-file set.
REQUIRED_SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")


class ProduceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _model(pins: dict, size: str) -> dict:
    for m in pins["core_a"]["models"]:
        if m["size"] == size:
            return m
    raise ProduceError(f"size {size!r} not in pins.toml core_a.models")


def source_dir(model: dict) -> Path:
    org, name = model["source_repo"].split("/")
    return Path("models") / "hf" / f"{org}__{name}@{model['source_revision'][:8]}"


def plan_steps(pins: dict, size: str, root: Path, staging: Path | None = None) -> list:
    """Commands for one size. Outputs go to `staging` (default models/gguf); the
    quantize input is always the BF16 GGUF in models/gguf."""
    m = _model(pins, size)
    gguf = root / "models" / "gguf"
    staging = gguf if staging is None else staging
    bf16 = gguf / m["reference"]["filename"]
    steps = [{
        "kind": "convert", "quant": None, "output": m["reference"]["filename"],
        "argv": [str(root / "third_party" / "convert-venv" / "bin" / "python"),
                 str(root / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"),
                 str(root / source_dir(m)), "--outtype", "bf16", "--no-mtp",
                 "--outfile", str(staging / bf16.name)],
    }]
    quantize = root / "third_party" / "llama.cpp" / "build-metal" / "bin" / "llama-quantize"
    for q in pins["core_a"]["quants"]:
        name = m["files"][q]["filename"]
        steps.append({"kind": "quantize", "quant": q, "output": name,
                      "argv": [str(quantize), str(bf16), str(staging / name), q]})
    return steps


def update_pins_text(text: str, filename: str, sha256: str, size: int) -> str:
    pattern = re.compile(
        r'(filename = "' + re.escape(filename) + r'"[^\n]*\n)sha256 = "([0-9a-f]*)"([^\n]*\n)size_bytes = (\d+)'
    )
    m = pattern.search(text)
    if m is None:
        raise ProduceError(f"{filename} not found in pins.toml")
    if m.group(2) and m.group(2) != sha256:
        raise ProduceError(f"{filename} already pinned with a different sha256 ({m.group(2)})")
    return text[:m.start()] + f'{m.group(1)}sha256 = "{sha256}"{m.group(3)}size_bytes = {size}' + text[m.end():]


def _default_lock_held_by_other() -> bool:
    return os.path.exists(MAC_HEAVY_LOCK) and os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") != "1"


def _default_disk_ok(path, need_gib: float) -> dict:
    return disk_guard.check(str(path), need_bytes=int(need_gib * GiB),
                            reserve_bytes=disk_guard.DEFAULT_RESERVE_GIB * GiB)


def _default_run(argv: list) -> int:
    return subprocess.run(argv).returncode


def _default_toolchain(root: Path) -> dict:
    """llama.cpp build (llama-quantize has no --version; llama-completion comes from the
    same build dir), the source tree the converter is read from, and the converter Python."""
    tp = root / "third_party"
    res = env_capture.default_run([str(tp / "llama.cpp/build-metal/bin/llama-completion"), "--version"])
    head = env_capture.default_run(["git", "-C", str(tp / "llama.cpp"), "rev-parse", "HEAD"])
    status = env_capture.default_run(["git", "-C", str(tp / "llama.cpp"), "status", "--porcelain"])
    py = env_capture.default_run([str(tp / "convert-venv/bin/python"), "--version"])
    quantize = tp / "llama.cpp/build-metal/bin/llama-quantize"
    return {
        "llama_cpp": env_capture.parse_llama_version((res.stderr + res.stdout) if res else ""),
        "llama_cpp_tree": {"head": head.stdout.strip() if head and head.returncode == 0 else None,
                           "dirty": bool(status.stdout.strip()) if status and status.returncode == 0 else None},
        "converter_python": (py.stdout + py.stderr).strip().removeprefix("Python ") if py else None,
        "llama_quantize_sha256": _sha256(quantize) if quantize.is_file() else None,
    }


def check_toolchain(tc: dict, pins: dict) -> None:
    want = pins["llama_cpp"]
    build = tc.get("llama_cpp") or {}
    tree = tc.get("llama_cpp_tree") or {}
    problems = []
    if build.get("build_number") != want["build_number"] or build.get("commit") != want["commit"]:
        problems.append(f"llama.cpp build {build.get('build_number')}/{build.get('commit')} "
                        f"!= pinned {want['build_number']}/{want['commit']}")
    if tree.get("head") != want["commit"]:
        problems.append(f"third_party/llama.cpp HEAD {tree.get('head')} != pinned {want['commit']}")
    if tree.get("dirty") is not False:
        problems.append(f"third_party/llama.cpp tree dirty={tree.get('dirty')} (converter is read from it)")
    if problems:
        raise ProduceError("toolchain does not match pins.toml [llama_cpp]: " + "; ".join(problems))


def _pin_entry(m: dict, filename: str) -> dict:
    for e in [m["reference"], *m["files"].values()]:
        if e["filename"] == filename:
            return e
    raise ProduceError(f"{filename} not in pins.toml")


def _already_done(out: Path, entry: dict, recipe_path: Path) -> bool:
    if not entry["sha256"] or not out.is_file() or not recipe_path.is_file():
        return False
    if json.loads(recipe_path.read_text()).get("output", {}).get("sha256") != entry["sha256"]:
        return False
    return _sha256(out) == entry["sha256"]


def produce(
    *,
    size: str,
    root: Path,
    run: Callable = _default_run,
    shard_sha256_override: dict | None = None,
    toolchain: dict | None = None,
    now: Callable = lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    lock_held_by_other: Callable = _default_lock_held_by_other,
    disk_ok: Callable = _default_disk_ok,
) -> dict:
    root = Path(root)
    pins_path = root / "configs" / "pins.toml"
    pins = tomllib.loads(pins_path.read_text())
    m = _model(pins, size)
    src = root / source_dir(m)

    def check_lock():
        if lock_held_by_other():
            raise ProduceError(f"Mac heavy lock is held by another job ({MAC_HEAVY_LOCK}); "
                               "retry later (MACHINE_POLICY)")

    check_lock()
    tc = dict(toolchain if toolchain is not None else _default_toolchain(root))
    check_toolchain(tc, pins)

    expected = shard_sha256_override or m["source_shards"]
    missing = [n for n in REQUIRED_SOURCE_FILES if not (src / n).is_file()]
    if missing:
        raise ProduceError(f"source snapshot {src} lacks {missing}; fetch them at the pinned revision first")
    small_files = {p.name: _sha256(p) for p in sorted(src.iterdir())
                   if p.is_file() and p.name not in expected and not p.name.endswith(".part")}
    for name, want in expected.items():
        path = src / name
        if not path.is_file() or _sha256(path) != want:
            raise ProduceError(f"source shard {name} missing or sha256 mismatch (pinned {want})")

    need_gib = NEED_FACTOR * sum((src / n).stat().st_size for n in expected) / GiB
    gguf_dir = root / "models" / "gguf"
    guard = disk_ok(gguf_dir, need_gib)
    if not guard.get("ok"):
        raise ProduceError(f"not enough disk for {need_gib:.1f} GiB (docs/STORAGE_PLAN.md): {guard}")

    staging = gguf_dir / ".tmp"
    staging.mkdir(parents=True, exist_ok=True)
    recipes = root / "results" / "artifacts"
    recipes.mkdir(parents=True, exist_ok=True)
    tc["converter_lock_sha256"] = _sha256(root / "configs" / "converter.lock.txt")
    tc["produce_python"] = platform.python_version()

    artifacts, bf16 = [], None
    for step in plan_steps(pins, size, root, staging=staging):
        out, tmp = gguf_dir / step["output"], staging / step["output"]
        recipe_path = recipes / f"{out.name}.json"
        entry = _pin_entry(m, out.name)
        if _already_done(out, entry, recipe_path):
            digest, nbytes, reused = entry["sha256"], out.stat().st_size, True
        else:
            check_lock()
            tmp.unlink(missing_ok=True)
            started = now()
            code = run(step["argv"])
            if code != 0 or not tmp.is_file():
                tmp.unlink(missing_ok=True)
                raise ProduceError(f"{step['kind']} failed (exit {code}): {' '.join(step['argv'])}")
            digest, nbytes, reused = _sha256(tmp), tmp.stat().st_size, False
            if entry["sha256"] and entry["sha256"] != digest:
                tmp.unlink()
                raise ProduceError(f"{out.name}: rebuilt bytes {digest} differ from the pinned sha256 "
                                   f"{entry['sha256']}; existing file left untouched")
        if step["kind"] == "convert":
            inputs = [{"repo": m["source_repo"], "revision": m["source_revision"], "shards": dict(expected),
                       "files": small_files}]
            bf16 = {"filename": out.name, "sha256": digest}
        else:
            inputs = [bf16]
        if not reused:
            os.replace(tmp, out)
            recipe = {
                "schema_version": 1, "kind": step["kind"], "size": size, "quant": step["quant"],
                "inputs": inputs, "argv": step["argv"], "toolchain": tc,
                "output": {"filename": out.name, "sha256": digest, "size_bytes": nbytes,
                           "path": f"models/gguf/{out.name}"},
                "started_at": started, "finished_at": now(), "disk_guard": guard,
            }
            recipe_path.write_text(json.dumps(recipe, indent=1) + "\n")
            pins_path.write_text(update_pins_text(pins_path.read_text(), out.name, digest, nbytes))
        artifacts.append({"filename": out.name, "quant": step["quant"], "sha256": digest,
                          "size_bytes": nbytes, "argv": step["argv"], "reused": reused})
    shutil.rmtree(staging, ignore_errors=True)
    return {"size": size, "artifacts": artifacts}


def _default_git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def delete_source(
    *,
    size: str,
    root: Path,
    git: Callable = _default_git,
    now: Callable = lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
) -> dict:
    """Delete the source safetensors only when STORAGE_PLAN's three conditions hold:
    every derived GGUF verified on disk against pins and recipe, pins + recipes
    committed, and the deletion recorded (results/artifacts/deletions.jsonl)."""
    root = Path(root)
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    m = _model(pins, size)
    verified = {}
    for e in [m["reference"], *m["files"].values()]:
        recipe_path = root / "results" / "artifacts" / f"{e['filename']}.json"
        if not e["sha256"] or not recipe_path.is_file():
            raise ProduceError(f"{e['filename']}: sha256 in pins and recipe are required before deleting the source")
        if json.loads(recipe_path.read_text()).get("output", {}).get("sha256") != e["sha256"]:
            raise ProduceError(f"{e['filename']}: recipe output sha256 != pins sha256")
        gguf = root / "models" / "gguf" / e["filename"]
        if not gguf.is_file():
            raise ProduceError(f"{e['filename']}: derived GGUF missing on disk")
        if gguf.stat().st_size != e["size_bytes"] or _sha256(gguf) != e["sha256"]:
            raise ProduceError(f"{e['filename']}: on-disk size/sha256 does not match pins")
        verified[e["filename"]] = e["sha256"]

    tracked = ["configs/pins.toml"] + [f"results/artifacts/{n}.json" for n in verified]
    status = git(root, "status", "--porcelain", "--", *tracked)
    if status.returncode != 0 or status.stdout.strip():
        raise ProduceError(f"commit configs/pins.toml and the recipes first (git status: {status.stdout.strip()!r})")
    for path in tracked:
        if git(root, "ls-files", "--error-unmatch", path).returncode != 0:
            raise ProduceError(f"{path} is not committed; commit pins and recipes before deleting the source")

    removed = []
    for name in m["source_shards"]:
        path = source_dir(m) / name
        if (root / path).exists():
            (root / path).unlink()
            removed.append(str(path))
    record = {"size": size, "removed": removed, "source": f"{m['source_repo']}@{m['source_revision']}",
              "source_shards": dict(m["source_shards"]), "verified_sha256": verified, "at": now()}
    with open(root / "results" / "artifacts" / "deletions.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return {"removed": removed, "record": record}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["produce", "delete-source"])
    parser.add_argument("--size", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        if args.cmd == "produce":
            print(json.dumps(produce(size=args.size, root=root), indent=1))
        else:
            print(json.dumps(delete_source(size=args.size, root=root), indent=1))
    except ProduceError as exc:
        print(f"produce_gguf: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
