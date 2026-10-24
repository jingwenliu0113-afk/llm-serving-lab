"""The gateway's request path (decision 044 "產品程式" item 7): the files `python -m capserve serve` can import from
capserve, and gateway_path_sha256, the hash Stage V freezes (the gateway is its manipulated variable) and /readyz
discloses.

- The files are the import closure of capserve/__main__.py computed from the source with ast, never by importing.
  Every import statement anywhere in a module counts, inside functions too (__main__ imports capserve.server only in
  main()): `import capserve.x`, `from capserve.x import ...`, `from capserve import x` (x a module of the package)
  and the relative forms add module x, which is followed in turn. capserve/__init__.py is always in it (python -m
  capserve runs it first). capserve is a flat package (service/tests assert it), so one level is the whole module
  name. Other packages are not followed: the frozen research modules profile.py imports are locked by the frozen-module
  tests, the environment by its freeze_sha256.
- gateway_path_sha256 = sha256 over the files in sorted order of their checkout-relative path
  (service/capserve/<name>.py): path, NUL, bytes, NUL for each, the idiom of run_stage_d.capserve_sha256 and
  stage_d_plan.harness_sha256. The research harness (stage_v_plan) computes the same value with its own copy of this
  algorithm, since research never imports capserve; service/tests assert both ends agree on this checkout.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PACKAGE = "capserve"
PACKAGE_DIR = ("service", PACKAGE)
ROOT = Path(__file__).resolve().parents[2]   # the checkout or release this gateway runs from
ENTRY = "__main__"
INIT = "__init__"


def _imports(path: Path, package_dir: Path) -> set:
    """The capserve modules the module at path imports (names, e.g. "config")."""
    found = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE:
                    found.add(parts[1] if len(parts) > 1 else INIT)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            parts = (node.module or "").split(".")
            if parts[0] != PACKAGE:
                continue
            module = parts[1] if len(parts) > 1 else None
        elif node.level == 1:   # from . import x / from .x import y (beyond the package: not capserve)
            module = node.module.split(".")[0] if node.module else None
        else:
            continue
        if module is not None:
            found.add(module)
        else:   # from capserve import a, b: each a module of the package, or a name of its __init__
            found.add(INIT)
            found |= {a.name for a in node.names if (package_dir / f"{a.name}.py").is_file()}
    return found


def gateway_path_files(root=ROOT) -> list:
    """The checkout-relative paths (sorted) of the request path's files under root."""
    package_dir = Path(root).joinpath(*PACKAGE_DIR)
    seen, todo = set(), [INIT, ENTRY]
    while todo:
        name = todo.pop()
        path = package_dir / f"{name}.py"
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        todo += sorted(_imports(path, package_dir) - seen)
    return sorted("/".join(PACKAGE_DIR) + f"/{name}.py" for name in seen)


def gateway_path_sha256(root=ROOT) -> str:
    h = hashlib.sha256()
    for rel in gateway_path_files(root):
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()
