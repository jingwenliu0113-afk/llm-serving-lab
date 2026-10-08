"""Tests for quality/tokenizer_crosscheck.py: the HF-tokenizer constants of decision 005
(WikiText chunk count, multiple-choice structural exclusions) checked against the
pinned llama.cpp tokenizer (llama-tokenize), which is what llama-perplexity uses."""

import json
import re
import struct

import mc_subset
import tokenizer_crosscheck as tc


def stable_toy(text):
    """Stand-in BPE: words (with one leading space) and single punctuation / newline chars."""
    return [sum(map(ord, p)) * 7 + len(p) for p in re.findall(r" ?[A-Za-z]+|[0-9]|\n|[^\sA-Za-z0-9]| ", text)]


def test_wikitext_mirrors_perplexity_file_reading():
    # llama-perplexity -f pops exactly one trailing newline before tokenizing.
    assert tc.perplexity_prompt("a b\n\n") == "a b\n"
    res = tc.check_wikitext("one two three four five six seven\n", hf=stable_toy, llama=stable_toy, n_ctx=2,
                            expected_chunks=3)
    assert res == {"n_tokens_hf": 7, "n_tokens_llama": 7, "identical": True, "chunks_llama": 3,
                   "expected_chunks": 3, "ok": True}


def test_wikitext_mismatch_is_reported():
    res = tc.check_wikitext("one two three\n", hf=stable_toy, llama=lambda t: stable_toy(t)[:-1], n_ctx=1,
                            expected_chunks=3)
    assert res["identical"] is False and res["ok"] is False and res["chunks_llama"] == 2


def _s(text):
    b = text.encode()
    return struct.pack("<I", len(b)) + b


def _task(t):
    n = len(t["answers"])
    mc1 = struct.pack("<I", n) + b"".join(_s(a) for a in t["answers"]) + struct.pack(f"<{n}i", *t["labels"])
    return _s(t["question"]) + mc1 + struct.pack("<I", 0)


def make_bin(tasks):
    return mc_subset.write_tasks([_task(t) for t in tasks])


TASKS = [
    {"question": "Q one", "answers": ["alpha", "beta"], "labels": [1, 0]},
    {"question": "Q two", "answers": ["x", "x y"], "labels": [0, 1]},  # "Q two x" is a prefix of "Q two x y"
    {"question": "Q three.", "answers": ["gamma.", "delta!"], "labels": [0, 1]},
]


def test_mc_split_recovers_per_string_tokens_and_exclusions():
    data = make_bin(TASKS)
    res = tc.check_mc(data, [0, 1, 2], hf=stable_toy, llama=stable_toy, committed_exclusions=[1])
    assert res["n_strings"] == 6 and res["n_identical"] == 6
    assert res["exclusions_hf"] == [1] and res["exclusions_llama"] == [1]
    assert res["ok"] is True and res["per_string_fallback"] == 0


def test_mc_detects_a_tokenizer_difference():
    data = make_bin(TASKS)

    def llama(text):  # merges "x y" into one token in llama.cpp only
        return stable_toy(text.replace("x y", "xy"))

    res = tc.check_mc(data, [0, 1, 2], hf=stable_toy, llama=llama, committed_exclusions=[1])
    assert res["ok"] is False and res["n_identical"] < res["n_strings"]


def test_mc_falls_back_per_string_when_joining_changes_boundaries():
    """If the separator merges with a string in the HF tokenization, that string is tokenized alone."""
    data = make_bin(TASKS)

    def merging(text):  # "." followed by the separator digit becomes one token
        return stable_toy(text.replace("." + tc.SEP, ".@" + tc.SEP[1:]))

    calls = []

    def llama(text):
        calls.append(text)
        return merging(text)

    res = tc.check_mc(data, [0, 1, 2], hf=merging, llama=llama, committed_exclusions=[1])
    assert res["ok"] is True and res["per_string_fallback"] >= 1
    assert any(c == "Q three. gamma." for c in calls)


def test_record_is_json_serialisable():
    rec = tc.make_record(build={"build_number": 11146, "commit": "7fe"}, gguf={"file": "x.gguf", "sha256": "ab"},
                         tokenizer_sha256="cd", checks={"wikitext": {"ok": True}})
    assert json.loads(json.dumps(rec))["ok"] is True
