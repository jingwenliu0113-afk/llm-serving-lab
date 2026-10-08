#!/usr/bin/env python3
"""Cross-check the HF-tokenizer constants of decision 005 against the pinned llama.cpp tokenizer.

The constants frozen in pins.toml were computed with the HF `tokenizers` package:
WikiText chunk count (580) and the multiple-choice structural exclusions (MMLU 3,
HellaSwag 0, ARC 0). llama-perplexity tokenizes with llama.cpp, so the pilot must
confirm both agree (005 pilot step 3). Mirrors of llama-perplexity's input handling:
- `-f` drops exactly one trailing newline; perplexity sets escape = false, so
  llama-tokenize runs with --no-escape;
- multiple choice tokenizes question + " " + answer with add_special (no BOS for Qwen3.5).

Multiple-choice strings are tokenized by llama-tokenize in batches joined by SEP.
A batch is accepted only if the HF tokenization of the joined text equals the
per-string HF tokens with SEP's tokens in between (so joining did not move a token
boundary) and llama.cpp produces exactly that sequence. Otherwise the batch is
bisected down to single strings, which llama-tokenize then tokenizes alone.

Usage (needs `tokenizers`; the converter venv has it):
    third_party/convert-venv/bin/python quality/tokenizer_crosscheck.py \\
        --llama-tokenize third_party/llama.cpp/build-metal/bin/llama-tokenize \\
        --gguf models/gguf/Qwen3.5-4B-Q4_K_M.gguf \\
        --tokenizer models/hf/Qwen__Qwen3.5-4B@851bf6e8/tokenizer.json \\
        --out quality/subsets/tokenizer-crosscheck.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_subset  # noqa: E402
import mmlu_exclusions  # noqa: E402

SEP = "7\n"  # a digit is its own pre-token and a newline cannot prefix the next word
BATCH = 1024


def perplexity_prompt(raw: str) -> str:
    """common/arg.cpp `-f`: read the file, pop one trailing newline."""
    return raw[:-1] if raw.endswith("\n") else raw


def check_wikitext(raw: str, *, hf: Callable, llama: Callable, n_ctx: int, expected_chunks: int) -> dict:
    text = perplexity_prompt(raw)
    ids_hf, ids_llama = hf(text), llama(text)
    chunks = len(ids_llama) // n_ctx  # perplexity.cpp: n_chunk_max = tokens.size() / n_ctx
    return {"n_tokens_hf": len(ids_hf), "n_tokens_llama": len(ids_llama), "identical": ids_hf == ids_llama,
            "chunks_llama": chunks, "expected_chunks": expected_chunks, "ok": chunks == expected_chunks}


def _mc_strings(data: bytes, indices: list) -> list:
    tasks = mc_subset.read_tasks(data)
    out = []
    for i in sorted(indices):
        t = mc_subset.parse_task(tasks[i])
        out.extend((i, t["question"] + " " + a) for a in t["answers"])
    return out


def check_mc(data: bytes, indices: list, *, hf: Callable, llama: Callable, committed_exclusions: list,
             batch: int = BATCH) -> dict:
    items = _mc_strings(data, indices)
    texts = [s for _, s in items]
    hf_ids = [hf(s) for s in texts]
    sep_ids = hf(SEP)
    llama_ids: list = [None] * len(texts)
    fallback = 0

    def solve(lo: int, hi: int) -> None:
        nonlocal fallback
        if hi - lo == 1:
            llama_ids[lo] = llama(texts[lo])
            fallback += 1
            return
        expected = []
        for k in range(lo, hi):
            expected.extend(hf_ids[k])
            if k + 1 < hi:
                expected.extend(sep_ids)
        joined = SEP.join(texts[lo:hi])
        if hf(joined) == expected and llama(joined) == expected:
            for k in range(lo, hi):
                llama_ids[k] = hf_ids[k]
            return
        mid = (lo + hi) // 2
        solve(lo, mid)
        solve(mid, hi)

    for start in range(0, len(texts), batch):
        solve(start, min(start + batch, len(texts)))

    differ = [items[k][0] for k in range(len(texts)) if llama_ids[k] != hf_ids[k]]

    def exclusions(ids: list) -> list:
        by_task: dict = {}
        for (task, _), seq in zip(items, ids):
            by_task.setdefault(task, []).append(seq)
        return sorted(t for t, seqs in by_task.items() if mmlu_exclusions.is_excluded(seqs))

    ex_hf, ex_llama = exclusions(hf_ids), exclusions(llama_ids)
    return {
        "n_tasks": len(set(indices)), "n_strings": len(texts),
        "n_identical": len(texts) - len(differ), "tasks_with_differences": sorted(set(differ))[:20],
        "replacement_chars": sum(s.count("�") for s in texts),
        "exclusions_hf": ex_hf, "exclusions_llama": ex_llama, "committed_exclusions": sorted(committed_exclusions),
        "per_string_fallback": fallback, "ok": ex_llama == sorted(committed_exclusions),
    }


def make_record(*, build: dict, gguf: dict, tokenizer_sha256: str, checks: dict) -> dict:
    return {
        "purpose": "decision 005 pilot step 3: HF-tokenizer constants vs the pinned llama.cpp tokenizer",
        "llama_cpp": build, "gguf": gguf, "hf_tokenizer_sha256": tokenizer_sha256,
        "mirrors": {"wikitext": "-f pops one trailing newline; escape = false",
                    "multiple_choice": "question + ' ' + answer, add_special=true, parse_special=false"},
        "checks": checks, "ok": all(c.get("ok") for c in checks.values()),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def llama_tokenizer(exe: str, gguf: str) -> Callable:
    def tokenize(text: str) -> list:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            path = fh.name
        try:
            res = subprocess.run([exe, "-m", gguf, "-f", path, "--ids", "--no-escape", "--log-disable"],
                                 capture_output=True, text=True, check=True)
        finally:
            Path(path).unlink()
        line = res.stdout.strip().splitlines()[-1]
        return json.loads(line)

    return tokenize


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-tokenize", required=True)
    parser.add_argument("--gguf", required=True, help="any pinned Core A GGUF (vocab only is read)")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json of the pinned snapshot")
    parser.add_argument("--data", default="data/datasets")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer  # lazy: only the CLI needs it

    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    q = pins["quality"]
    data = Path(args.data)
    for name in ("wiki_test_raw", "hellaswag", "mmlu", "arc"):
        path = data / Path(q["files"][name]["path"]).name
        if _sha256(path) != q["files"][name]["sha256"]:
            raise SystemExit(f"{path}: sha256 does not match pins.toml quality.files.{name}")

    tok = Tokenizer.from_file(args.tokenizer)
    hf = lambda t: tok.encode(t, add_special_tokens=False).ids  # noqa: E731
    llama = llama_tokenizer(args.llama_tokenize, args.gguf)
    version = subprocess.run([args.llama_tokenize, "--version"], capture_output=True, text=True)
    build_line = (version.stderr + version.stdout).strip().splitlines()[:1]

    excl = json.loads((root / q["subsets"]["mmlu"]["exclude_file"]).read_text())["excluded"]
    hs_idx = json.loads((root / q["subsets"]["hellaswag"]["index_file"]).read_text())["indices"]
    mmlu_idx = json.loads((root / q["subsets"]["mmlu"]["index_file"]).read_text())["indices"]
    arc = (data / "arc-combined-validation.bin").read_bytes()

    checks = {
        "wikitext": check_wikitext((data / "wiki.test.raw").read_text(encoding="utf-8"), hf=hf, llama=llama,
                                   n_ctx=512, expected_chunks=q["expected"]["wikitext_chunks"]),
        "mmlu": check_mc((data / "mmlu-test.bin").read_bytes(), mmlu_idx, hf=hf, llama=llama,
                         committed_exclusions=excl),
        "hellaswag": check_mc((data / "hellaswag-validation.bin").read_bytes(), hs_idx, hf=hf, llama=llama,
                              committed_exclusions=[]),
        "arc": check_mc(arc, list(range(len(mc_subset.read_tasks(arc)))), hf=hf, llama=llama,
                        committed_exclusions=[]),
    }
    rec = make_record(build={"version_line": build_line[0] if build_line else None},
                      gguf={"file": Path(args.gguf).name, "sha256": _sha256(Path(args.gguf))},
                      tokenizer_sha256=_sha256(Path(args.tokenizer)), checks=checks)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("ok",)} for k, v in checks.items()} | {"ok": rec["ok"]}))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
