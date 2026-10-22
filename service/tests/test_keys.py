"""Tests for capserve/keys.py: the per-key API keys file (names and sha256 only, never a key) and its quotas
(decision 038 §5): a requests-per-minute token bucket and a cap on a key's requests inside the gateway."""

import hashlib

import pytest

from capserve.keys import KeyFileError, QuotaDenied, load_keyring


def sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


KEYS = f"""
[[keys]]
name = "team-a"
sha256 = "{sha('key-a')}"
requests_per_minute = 2
max_in_flight = 1

[[keys]]
name = "team-b"
sha256 = "{sha('key-b')}"
"""


def ring(tmp_path, text=KEYS):
    path = tmp_path / "keys.toml"
    path.write_text(text)
    return load_keyring(path)


def test_keys_match_by_sha256_only(tmp_path):
    r = ring(tmp_path)
    assert (r.match(b"key-a"), r.match(b"key-b"), r.match(b"key-c"), r.match(b"")) == ("team-a", "team-b", None, None)
    assert r.names == ("team-a", "team-b")


ENTRY = '[[keys]]\nname = "{name}"\nsha256 = "{digest}"\n'


@pytest.mark.parametrize("text, match", [
    ("", "no keys"),
    ('[[keys]]\nname = "a"\nsha256 = "abc"\n', "sha256"),
    (ENTRY.format(name="a", digest=sha("x")) + ENTRY.format(name="a", digest=sha("y")), "duplicate"),
    (ENTRY.format(name="a", digest=sha("x")) + ENTRY.format(name="b", digest=sha("x")), "duplicate"),
    (ENTRY.format(name="a", digest=sha("x")) + 'key = "plaintext"\n', "key"),
    (ENTRY.format(name="a b", digest=sha("x")), "name"),
    (ENTRY.format(name="a", digest=sha("x")) + "requests_per_minute = 0\n", "requests_per_minute"),
    (ENTRY.format(name="a", digest=sha("x")) + "max_in_flight = 0\n", "max_in_flight"),
    ('[other]\nx = 1\n', "other"),
    ("not toml [", "cannot read"),
])
def test_bad_key_files_are_refused(tmp_path, text, match):
    with pytest.raises(KeyFileError, match=match):
        ring(tmp_path, text)


def test_a_missing_key_file_is_refused(tmp_path):
    with pytest.raises(KeyFileError, match="cannot read"):
        load_keyring(tmp_path / "absent.toml")


def test_the_rate_limit_is_a_token_bucket(tmp_path):
    r = ring(tmp_path)
    for now in (0.0, 0.0):
        r.admit("team-a", now)()                          # admitted, then released at once
    with pytest.raises(QuotaDenied) as exc:
        r.admit("team-a", 0.0)
    assert exc.value.limit == "requests_per_minute" and exc.value.retry_after_s == pytest.approx(30.0)
    with pytest.raises(QuotaDenied) as exc:
        r.admit("team-a", 15.0)
    assert exc.value.retry_after_s == pytest.approx(15.0)
    r.admit("team-a", 30.0)()                              # one token refilled after 60 / 2 s


def test_max_in_flight_counts_until_release_and_costs_no_token(tmp_path):
    r = ring(tmp_path)
    release = r.admit("team-a", 0.0)
    with pytest.raises(QuotaDenied) as exc:
        r.admit("team-a", 0.0)
    assert exc.value.limit == "max_in_flight" and exc.value.retry_after_s is None
    release()
    release()                                              # idempotent
    r.admit("team-a", 0.0)                                 # the denied request took no token: 1 of 2 left
    assert r.in_flight("team-a") == 1


def test_a_key_without_limits_is_unlimited(tmp_path):
    r = ring(tmp_path)
    releases = [r.admit("team-b", 0.0) for _ in range(1000)]
    assert r.in_flight("team-b") == 1000
    for release in releases:
        release()
    assert r.in_flight("team-b") == 0
