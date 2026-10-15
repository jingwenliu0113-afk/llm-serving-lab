"""Decision 027 §3 / 029 regression: the research harness and analysis never import the serving service (capserve).

capserve may import research modules read-only (core_b_plan, run_adv, ...); the reverse direction is forbidden, so
product code can never change what a research runner or analysis computes. The check parses every .py file under
benchmarks/, scripts/ and quality/ (AST, so comments and strings do not count) for `import capserve...` and
`from capserve... import ...`.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIRS = ("benchmarks", "scripts", "quality")
PRODUCT_PACKAGE = "capserve"


def _is_product(module: str | None) -> bool:
    return module is not None and (module == PRODUCT_PACKAGE or module.startswith(PRODUCT_PACKAGE + "."))


def product_imports(root: Path) -> list:
    """'<relative path>:<line>' of every import of capserve in the research directories under root."""
    found = []
    for d in RESEARCH_DIRS:
        for path in sorted((root / d).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    hit = any(_is_product(alias.name) for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    hit = node.level == 0 and _is_product(node.module)
                else:
                    continue
                if hit:
                    found.append(f"{path.relative_to(root)}:{node.lineno}")
    return sorted(found)


def test_research_code_never_imports_capserve():
    assert product_imports(ROOT) == [], "research code imports the serving service (027 §3: never)"


def test_detection_catches_every_import_form_and_ignores_text(tmp_path):
    for d in RESEARCH_DIRS:
        (tmp_path / d).mkdir()
    (tmp_path / "benchmarks" / "a.py").write_text("import json\nimport capserve\n")
    (tmp_path / "scripts" / "b.py").write_text("from capserve.admission import AdmissionController\n")
    (tmp_path / "quality" / "c.py").write_text("import os, capserve.profile as cp\n")
    (tmp_path / "quality" / "sub").mkdir()
    (tmp_path / "quality" / "sub" / "d.py").write_text(
        "# import capserve\ns = 'from capserve import x'\nimport capserver\nfrom . import capserve\n")
    assert product_imports(tmp_path) == ["benchmarks/a.py:2", "quality/c.py:1", "scripts/b.py:1"]
