"""Per-key API keys and quotas (decision 038 §5).

The keys file (CAPSERVE_API_KEYS_FILE; refused inside the checkout or release by __main__) is TOML:

    [[keys]]
    name = "team-a"              # [A-Za-z0-9._-]{1,64}: the only thing logs and metrics ever show
    sha256 = "<64 hex>"          # sha256 of the key's UTF-8 bytes; the key itself is never stored anywhere
    requests_per_minute = 60     # optional: a token bucket of that size, refilled at that rate
    max_in_flight = 8            # optional: the key's requests inside the gateway, from auth to the response end

- match: sha256 of the bearer token, compared with every entry by hmac.compare_digest, never stopping early.
- admit (after auth and the 503 gate, before the body is read): max_in_flight first (a request it refuses costs no
  token), then the bucket (full at the key's first request). QuotaDenied carries the limit and, for the bucket, the
  seconds until the next token; the gateway answers 429 quota_exceeded. The returned release is idempotent and runs
  as a scope cleanup.
- Quotas only ever turn requests away: they never raise the concurrency the capacity profile allows.
"""

from __future__ import annotations

import hashlib
import hmac
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class KeyFileError(ValueError):
    pass


class QuotaDenied(Exception):
    def __init__(self, limit: str, retry_after_s: float | None):
        super().__init__(limit)
        self.limit = limit                    # "requests_per_minute" | "max_in_flight"
        self.retry_after_s = retry_after_s    # the bucket's wait; None for max_in_flight


class KeyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requests_per_minute: int | None = Field(None, ge=1)
    max_in_flight: int | None = Field(None, ge=1)


class _KeyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    keys: list[KeyEntry]


@dataclass
class _State:
    tokens: float = 0.0
    last: float | None = None
    in_flight: int = 0


class KeyRing:
    def __init__(self, entries: list):
        self._entries = list(entries)
        self._digests = [bytes.fromhex(e.sha256) for e in self._entries]
        self._by_name = {e.name: e for e in self._entries}
        self._state = {e.name: _State() for e in self._entries}

    @property
    def names(self) -> tuple:
        return tuple(e.name for e in self._entries)

    def match(self, token: bytes) -> str | None:
        digest = hashlib.sha256(token).digest()
        found = None
        for entry, want in zip(self._entries, self._digests):
            if hmac.compare_digest(digest, want):
                found = entry.name
        return found

    def in_flight(self, name: str) -> int:
        return self._state[name].in_flight

    def admit(self, name: str, now: float) -> Callable[[], None]:
        entry, st = self._by_name[name], self._state[name]
        if entry.max_in_flight is not None and st.in_flight >= entry.max_in_flight:
            raise QuotaDenied("max_in_flight", None)
        rpm = entry.requests_per_minute
        if rpm is not None:
            rate = rpm / 60.0
            st.tokens = float(rpm) if st.last is None else min(float(rpm), st.tokens + (now - st.last) * rate)
            st.last = now
            if st.tokens < 1.0:
                raise QuotaDenied("requests_per_minute", (1.0 - st.tokens) / rate)
            st.tokens -= 1.0
        st.in_flight += 1
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                st.in_flight -= 1

        return release


def load_keyring(path) -> KeyRing:
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise KeyFileError(f"{path}: cannot read the keys file: {exc}") from exc
    unknown = sorted(set(data) - {"keys"})
    if unknown:
        raise KeyFileError(f"{path}: unknown tables {unknown} (only [[keys]])")
    if not data.get("keys"):
        raise KeyFileError(f"{path}: no keys ([[keys]] with name and sha256)")
    try:
        doc = _KeyFile.model_validate(data)
    except ValidationError as exc:
        raise KeyFileError(f"{path}: {exc}") from exc
    for field in ("name", "sha256"):
        values = [getattr(e, field) for e in doc.keys]
        if len(set(values)) != len(values):
            raise KeyFileError(f"{path}: duplicate {field} among the keys")
    return KeyRing(doc.keys)
