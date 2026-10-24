"""Tests for capserve/gateway_path.py: the gateway's request path (decision 044 "產品程式" item 7) is the import
closure of capserve/__main__.py, computed from the source by ast (never by importing; imports inside functions count),
with capserve/__init__.py; gateway_path_sha256 hashes those files (path and bytes) in sorted order."""

import hashlib
from pathlib import Path

import pytest

from capserve import gateway_path as gp

ROOT = Path(__file__).resolve().parents[2]
CAPSERVE = ROOT / "service" / "capserve"
# 044: 15 files plus the modules 044 adds to the request path (gateway_path itself); deploy, itest, loadtest, runs,
# procs and profile_gen are outside it
REQUEST_PATH = ["__init__", "__main__", "admission", "app", "config", "gateway_path", "guard", "keys", "metrics",
                "openai_api", "profile", "proxy", "retry", "server", "tokens", "upstream"]
OUTSIDE = ["deploy", "itest", "loadtest", "runs", "procs", "profile_gen"]


def test_the_request_path_is_main_s_import_closure_without_the_deploy_tools():
    files = gp.gateway_path_files()
    assert files == [f"service/capserve/{name}.py" for name in REQUEST_PATH]
    assert not {f"service/capserve/{name}.py" for name in OUTSIDE} & set(files)
    assert {p.stem for p in CAPSERVE.glob("*.py")} == set(REQUEST_PATH) | set(OUTSIDE)   # every module is placed


def test_capserve_is_a_flat_package():
    """The closure follows capserve.<module> one level deep: a subpackage would escape it."""
    assert [p for p in CAPSERVE.iterdir() if p.is_dir() and p.name != "__pycache__"] == []


def test_the_sha256_hashes_path_and_bytes_in_order():
    h = hashlib.sha256()
    for rel in gp.gateway_path_files(ROOT):
        h.update(rel.encode() + b"\0" + (ROOT / rel).read_bytes() + b"\0")
    assert gp.gateway_path_sha256(ROOT) == gp.gateway_path_sha256() == h.hexdigest()


def package(tmp_path, modules: dict) -> Path:
    pkg = tmp_path / "service" / "capserve"
    pkg.mkdir(parents=True)
    for name, text in modules.items():
        (pkg / f"{name}.py").write_text(text)
    return tmp_path


def test_every_import_form_counts_inside_functions_too_and_text_does_not(tmp_path):
    root = package(tmp_path, {
        "__init__": '"""doc"""\n',
        "__main__": ("import json\nfrom capserve.a import x\nimport capserve.b as bee\n\n"
                     "def main():\n    from capserve import c, NAME\n    from . import d\n    from .e import f\n"),
        "a": "x = 1\n", "b": "", "c": "import capserve\n", "d": "", "e": "f = 2\n",
        "g": "# import capserve.h\ns = 'from capserve import h'\n", "h": "",
        "i": "", "j": "import capserver\nimport other.capserve\n"})
    (root / "service" / "capserve" / "d.py").write_text("import capserve.g\n")
    assert gp.gateway_path_files(root) == [f"service/capserve/{n}.py"
                                           for n in ("__init__", "__main__", "a", "b", "c", "d", "e", "g")]


def test_the_closure_is_transitive_and_follows_init(tmp_path):
    root = package(tmp_path, {"__init__": "from capserve import k\n", "__main__": "import capserve.a\n",
                              "a": "from capserve.b import y\n", "b": "y = 1\n", "k": "", "unused": ""})
    assert gp.gateway_path_files(root) == [f"service/capserve/{n}.py" for n in ("__init__", "__main__", "a", "b", "k")]


def test_a_changed_byte_in_the_path_changes_the_sha_and_one_outside_does_not(tmp_path):
    root = package(tmp_path, {"__init__": "", "__main__": "import capserve.a\n", "a": "x = 1\n", "deploy": "y = 1\n"})
    before = gp.gateway_path_sha256(root)
    (root / "service" / "capserve" / "deploy.py").write_text("y = 2\n")
    assert gp.gateway_path_sha256(root) == before
    (root / "service" / "capserve" / "a.py").write_text("x = 2\n")
    assert gp.gateway_path_sha256(root) != before


def test_the_research_harness_computes_the_same_closure():
    """044: stage_v_plan computes gateway_path_files / gateway_path_sha256 with its own copy of the algorithm (research
    never imports capserve); on this checkout both ends agree. Skipped until the Stage V harness exists."""
    svp = pytest.importorskip("stage_v_plan")
    assert list(svp.gateway_path_files(ROOT)) == gp.gateway_path_files(ROOT)
    assert svp.gateway_path_sha256(ROOT) == gp.gateway_path_sha256(ROOT)
