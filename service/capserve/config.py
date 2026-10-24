"""Gateway settings (decision 029): a TOML file validated by strict pydantic models (unknown keys are errors).

The environment supplies exactly three things, never stored in settings or logs:
- CAPSERVE_API_KEY: the optional bearer key (bytes; compared with hmac.compare_digest), or
- CAPSERVE_API_KEYS_FILE: per-key keys (sha256 only) with quotas (038 §5, capserve/keys.py); never both. Without
  either the gateway binds only a loopback address.
- CAPSERVE_ALLOW_PROVISIONAL=1: the same as --allow-provisional (027 §4).

Limits (max_concurrency, max_queue, token bounds) come from the capacity profile's active_class, never from here;
max_queue_wait_s and the pending slots are product policy, not research results (029). There is no workers option:
the gateway is one process, otherwise its in-process limits would multiply.

Candidate policies (decision 044 "產品程式" items 2-5; 043 7.3's designs, which only Stage V validates), off unless set:
- [admission] (policy "weighted-v1", budget, max_queue, classes [{name, units, max_concurrency}]) replaces
  active_class (there is none): a prompt is admitted as the first named class it fits, under one budget of units and
  each class's cap (capserve.admission.PolicyController); max_pending = budget + max_queue + pre_admission_slots.
- [prefill_pacing] (burst_tokens, rate_tokens_per_s): a prefill token bucket in the active class's FIFO.
- Never both (never measured together: V4 ran with pacing off); either counts every prompt with tokenizer "local".
- served_classes is the one place a gateway's classes are resolved (capserve serve, create_app, deploy up): each by
  profile.serving_class (030 8.7, 043, whatever flag); the parameters must fit the classes (ConfigError: the caps are
  the classes' max_concurrency, units x cap = budget, strictly increasing prompt limits, one output bound; a burst at
  least the class's max_prompt_tokens, else a prompt could never be admitted); and a candidate policy is served only
  (ProfileError otherwise, no flag overrides it) on a provisional profile (research: it loads only with
  --allow-provisional) or on a verified §12 profile carrying the identical section (the same parameters and queue
  settings) whose units were validated on this gateway's request path (gateway_path_sha256). deploy (study=False)
  never serves one without such a section, a provisional profile's included.
"""

from __future__ import annotations

import ipaddress
import ssl
import tomllib
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from capserve.profile import (STAGE_V_SECTION, AdmissionParams, CapacityProfile, PacingParams, ProfileError,
                              WorkloadClass, admission_problems)

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


class TLS(_Strict):
    """038 §6: the gateway serves HTTPS with these files (the user's certificate and key, outside the checkout)."""
    certfile: str
    keyfile: str


def loopback_client_context(tls: TLS) -> ssl.SSLContext:
    """How the supervisor, capserve.loadtest and capserve.itest reach their own TLS gateway on 127.0.0.1 (038 §6):
    trust exactly the configured certificate (no system CAs; accepted as a trust anchor even when it is a CA-issued
    leaf, VERIFY_X509_PARTIAL_CHAIN) and skip the host-name check, since a public certificate never names 127.0.0.1."""
    ctx = ssl.create_default_context(cafile=tls.certfile)
    ctx.check_hostname = False
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


class Settings(_Strict):
    active_class: str | None = None   # required, unless [admission] (044) names the classes instead
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
    # 043 §7.2: how long uvicorn keeps an idle client connection open. Its own default, 5 s, is raced by a client pool
    # keeping idle connections as long or longer (vllm bench's aiohttp pool: keepalive_timeout 60 s, FIFO; httpx's
    # default 5 s): D-B's failures before the first token, 042. A pool whose idle limit is below 75 s (aiohttp's default
    # 15 s, httpx's 5 s, vllm bench's 60 s) lets an idle connection expire before the gateway closes it; a client
    # keeping idle connections longer or without a limit (Go's default 90 s, urllib3) can still race. Idle sockets
    # cost no request slot (the guard counts requests).
    keep_alive_s: int = Field(75, ge=1)
    disconnect_poll_s: float = Field(0.25, gt=0)
    monitor_interval_s: float = Field(2.0, gt=0)
    monitor_timeout_s: float = Field(2.0, gt=0)
    monitor_fail_threshold: int = Field(3, ge=1)
    retry_after_s: RetryAfter = RetryAfter()
    tokens: TokenOverheads = TokenOverheads()
    tls: TLS | None = None
    admission: AdmissionParams | None = None        # 044: weighted-v1 (candidate; off unless set)
    prefill_pacing: PacingParams | None = None      # 044: prefill pacing (candidate; off unless set)

    @model_validator(mode="after")
    def _tokenizer(self):
        if (self.tokenizer == "local") != (self.tokenizer_path is not None):
            raise ValueError('tokenizer_path is required with tokenizer = "local" and allowed only with it')
        return self

    @model_validator(mode="after")
    def _policies(self):
        if (self.active_class is None) == (self.admission is None):
            raise ValueError("set exactly one of active_class (one class, 029) and [admission] (044 weighted-v1, which "
                             "names its classes and has no active class)")
        if self.admission is not None and self.prefill_pacing is not None:
            raise ValueError("[admission] and [prefill_pacing] together were never measured (044: V4 ran with pacing "
                             "off)")
        if (self.admission is not None or self.prefill_pacing is not None) and self.tokenizer != "local":
            raise ValueError('a candidate policy counts every prompt in the gateway: it needs tokenizer = "local" '
                             "(044: weighted-v1 classifies by the count, pacing spends it)")
        return self

    def max_pending(self, cls: WorkloadClass | None = None) -> int:
        """Requests the guard lets in before reading a body: running + queued + pre-admission slots (044 weighted-v1:
        its budget + its max_queue + the slots)."""
        if self.admission is not None:
            return self.admission.budget + self.admission.max_queue + self.pre_admission_slots
        return cls.max_concurrency + cls.max_queue + self.pre_admission_slots


CANDIDATE_POLICIES = ("admission", "prefill_pacing")   # 044: validated only by Stage V


def candidate_policies(settings: Settings) -> list:
    return [name for name in CANDIDATE_POLICIES if getattr(settings, name) is not None]


def queue_settings(settings: Settings) -> dict:
    """The queue settings a candidate policy is validated with (044: max_queue_wait_s, request_deadline_s)."""
    return {"max_queue_wait_s": settings.max_queue_wait_s, "request_deadline_s": settings.request_deadline_s}


def _policy_config_problems(settings: Settings, classes: list) -> list:
    problems = []
    pacing = settings.prefill_pacing
    if pacing is not None and pacing.burst_tokens < classes[0].max_prompt_tokens:
        problems.append(f"[prefill_pacing] burst_tokens {pacing.burst_tokens} < class {classes[0].name}'s "
                        f"max_prompt_tokens {classes[0].max_prompt_tokens}: such a prompt could never be admitted "
                        "(044)")
    if settings.admission is not None:
        problems += admission_problems(settings.admission, classes)
    return problems


def _policy_serving_problems(settings: Settings, profile: CapacityProfile, classes: list, *, gateway_path_sha256: str,
                             study: bool) -> list:
    names = candidate_policies(settings)
    if not names:
        return []
    tables = " and ".join(f"[{n}]" for n in names)
    if not profile.verified:
        return [] if study else [f"{tables}: deploy never serves a candidate policy without a §12 section, a "
                                 f"provisional profile's ({profile.profile_id}) included (044)"]
    if profile.section != STAGE_V_SECTION:
        return [f"{tables}: verified profile {profile.profile_id} ({profile.section}) carries no Stage V section: a "
                "candidate policy serves only on a provisional profile with --allow-provisional (research) or a "
                "verified §12 profile carrying the identical section (044), whatever flag is given"]
    problems = []
    queue, validated_path = queue_settings(settings), profile.provenance.gateway_path_sha256
    if gateway_path_sha256 != validated_path:
        problems.append(f"this gateway's request path (gateway_path_sha256 {gateway_path_sha256}) is not the one "
                        f"profile {profile.profile_id}'s units were validated on ({validated_path}): its candidate "
                        "policies are refused (044)")
    if settings.prefill_pacing is not None:
        c = classes[0]
        want = None if c.pacing is None else {**c.pacing.model_dump(include=set(PacingParams.model_fields)),
                                              "queue": c.pacing.queue.model_dump()}
        got = {**settings.prefill_pacing.model_dump(), "queue": queue}
        if got != want:
            problems.append(f"[prefill_pacing] {settings.prefill_pacing.model_dump()} with queue {queue} is not class "
                            f"{c.name}'s validated pacing {want} (044: exactly these, or none)")
    if settings.admission is not None:
        adm = profile.admission
        want = None if adm is None else {**adm.model_dump(include=set(AdmissionParams.model_fields)),
                                         "queue": adm.queue.model_dump()}
        got = {**settings.admission.model_dump(), "queue": queue}
        if got != want:
            problems.append(f"[admission] {settings.admission.model_dump()} with queue {queue} is not profile "
                            f"{profile.profile_id}'s validated admission section {want} (044: exactly these, or none)")
        elif adm.gateway_path_sha256 != gateway_path_sha256:
            problems.append(f"the admission section was validated on request path {adm.gateway_path_sha256}, not this "
                            f"gateway's {gateway_path_sha256} (044)")
    return problems


def served_classes(settings: Settings, profile: CapacityProfile, *, gateway_path_sha256: str,
                   study: bool = True) -> list:
    """The classes a gateway with these settings serves, in admission order: [the active class] or weighted-v1's.
    ProfileError (030 8.7 / 043 / 044 refusals, whatever flag) or ConfigError (parameters that do not fit the
    classes); see the module docstring. study=False is deploy's call: never a candidate policy without §12."""
    names = [settings.active_class] if settings.admission is None else [a.name for a in settings.admission.classes]
    classes = [profile.serving_class(n) for n in names]
    if problems := _policy_serving_problems(settings, profile, classes, gateway_path_sha256=gateway_path_sha256,
                                            study=study):
        raise ProfileError("; ".join(problems))
    if problems := _policy_config_problems(settings, classes):
        raise ConfigError("; ".join(problems))
    return classes


def served_limits(settings: Settings, classes: list) -> dict:
    """/readyz's limits: the active class's, or weighted-v1's envelope (044: the most requests in flight, the largest
    cap since units x cap = budget; its max_queue; the largest prompt; the one output bound)."""
    adm = settings.admission
    if adm is None:
        c = classes[0]
        return {"max_concurrency": c.max_concurrency, "max_queue": c.max_queue, "max_pending": settings.max_pending(c),
                "max_prompt_tokens": c.max_prompt_tokens, "max_output_tokens": c.max_output_tokens}
    return {"max_concurrency": max(a.max_concurrency for a in adm.classes), "max_queue": adm.max_queue,
            "max_pending": settings.max_pending(), "max_prompt_tokens": max(c.max_prompt_tokens for c in classes),
            "max_output_tokens": classes[0].max_output_tokens}


def admission_view(settings: Settings, classes: list) -> dict | None:
    """/readyz's admission (044): weighted-v1's parameters with each class's token limits; None when off."""
    adm = settings.admission
    if adm is None:
        return None
    return {**adm.model_dump(), "classes": [{**a.model_dump(), "max_prompt_tokens": c.max_prompt_tokens,
                                             "max_output_tokens": c.max_output_tokens}
                                            for a, c in zip(adm.classes, classes)]}


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
