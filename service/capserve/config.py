"""Gateway settings (decision 029): a TOML file validated by strict pydantic models (unknown keys are errors).

The environment supplies exactly three things, never stored in settings or logs:
- CAPSERVE_API_KEY: the optional bearer key (bytes; compared with hmac.compare_digest), or
- CAPSERVE_API_KEYS_FILE: per-key keys (sha256 only) with quotas (038 §5, capserve/keys.py); never both. Without
  either the gateway binds only a loopback address.
- CAPSERVE_ALLOW_PROVISIONAL=1: the same as --allow-provisional (027 §4).

Limits (max_concurrency, max_queue, token bounds) come from the capacity profile's active_class, never from here;
max_queue_wait_s and the pending slots are product policy, not research results (029). There is no workers option:
the gateway is one process, otherwise its in-process limits would multiply.
"""

from __future__ import annotations

import ipaddress
import tomllib
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from capserve.profile import WorkloadClass

API_KEY_ENV = "CAPSERVE_API_KEY"
KEYS_FILE_ENV = "CAPSERVE_API_KEYS_FILE"
ALLOW_PROVISIONAL_ENV = "CAPSERVE_ALLOW_PROVISIONAL"


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


REJECTION_REASONS = ("queue_full", "queue_timeout", "backend_unavailable", "draining")


class RetryAfter(_Strict):
    """Retry-After seconds per rejection reason. mode "adaptive" (038 §3, capserve/retry.py) estimates the 429s
    (queue_full, queue_timeout) from the observed service time within [min_s, max_s] and uses the fixed values until
    a request has completed and for the 503s; mode "fixed" is the MVP's fixed value per reason."""
    queue_full: int = Field(2, ge=1)
    queue_timeout: int = Field(2, ge=1)
    backend_unavailable: int = Field(5, ge=1)
    draining: int = Field(5, ge=1)
    mode: Literal["fixed", "adaptive"] = "adaptive"
    min_s: float = Field(1.0, gt=0)
    max_s: float = Field(60.0, gt=0)

    @model_validator(mode="after")
    def _bounds(self):
        if self.min_s > self.max_s:
            raise ValueError(f"retry_after_s.min_s {self.min_s} > max_s {self.max_s}")
        return self

    def fixed_values(self) -> dict:
        return {reason: getattr(self, reason) for reason in REJECTION_REASONS}


class TokenOverheads(_Strict):
    """Template tokens the byte fast path adds on top of one token per prompt byte (tokens.py). Deliberately
    generous until RTX IT-4 measures the real chat-template overhead; too large only costs a /tokenize call."""
    chat_overhead: int = Field(32, ge=0)     # chat template + generation prompt
    per_message: int = Field(16, ge=0)       # role markers and separators of one message
    completion_overhead: int = Field(1, ge=0)


class Settings(_Strict):
    active_class: str
    # 038 §4: "vllm" asks vLLM's /tokenize under a permit (029); "local" counts in the gateway from tokenizer_path,
    # the snapshot vLLM serves (the deploy sets it), before admission.
    tokenizer: Literal["vllm", "local"] = "vllm"
    tokenizer_path: str | None = None
    backend_url: str = "http://127.0.0.1:8100"
    host: str = "127.0.0.1"
    port: int = Field(8080, ge=0, le=65535)
    max_queue_wait_s: float = Field(2.0, gt=0)
    pre_admission_slots: int = Field(16, ge=0)
    request_deadline_s: float = Field(300.0, gt=0)
    max_body_bytes: int = Field(1 << 20, ge=1)
    body_read_timeout_s: float = Field(10.0, gt=0)
    drain_timeout_s: int = Field(60, ge=0)
    disconnect_poll_s: float = Field(0.25, gt=0)
    monitor_interval_s: float = Field(2.0, gt=0)
    monitor_timeout_s: float = Field(2.0, gt=0)
    monitor_fail_threshold: int = Field(3, ge=1)
    retry_after_s: RetryAfter = RetryAfter()
    tokens: TokenOverheads = TokenOverheads()

    @model_validator(mode="after")
    def _tokenizer(self):
        if (self.tokenizer == "local") != (self.tokenizer_path is not None):
            raise ValueError('tokenizer_path is required with tokenizer = "local" and allowed only with it')
        return self

    def max_pending(self, cls: WorkloadClass) -> int:
        """Requests the guard lets in before reading a body: running + queued + pre-admission slots."""
        return cls.max_concurrency + cls.max_queue + self.pre_admission_slots


def load_settings(path) -> Settings:
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: cannot read the settings: {exc}") from exc
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def api_key_from_env(env: Mapping[str, str]) -> bytes | None:
    key = env.get(API_KEY_ENV, "")
    return key.encode("utf-8") if key else None


def keys_file_from_env(env: Mapping[str, str]) -> str | None:
    path = env.get(KEYS_FILE_ENV, "")
    if path and env.get(API_KEY_ENV, ""):
        raise ConfigError(f"set either {API_KEY_ENV} (one key) or {KEYS_FILE_ENV} (per-key quotas), not both")
    return path or None


def allow_provisional(flag: bool, env: Mapping[str, str]) -> bool:
    return bool(flag) or env.get(ALLOW_PROVISIONAL_ENV) == "1"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind(host: str, auth) -> None:
    """Without a key (auth: the API key or the keyring, None when neither) the gateway is reachable only from this
    machine (029 Auth)."""
    if auth is None and not _is_loopback(host):
        raise ConfigError(f"refusing to bind {host!r} without {API_KEY_ENV} or {KEYS_FILE_ENV}: only loopback is "
                          "allowed without a key")
