"""Tests for capserve/config.py: settings from TOML; the environment supplies only CAPSERVE_API_KEY and
CAPSERVE_ALLOW_PROVISIONAL (decision 029). Decision 044's candidate policies ([admission] weighted-v1, which replaces
the active class, and [prefill_pacing]) are off by default, never together, count with the local tokenizer, must fit
the classes they name, and serve only on a provisional profile (research) or a verified §12 profile carrying the
identical section with its queue settings and request path; deploy never serves one without such a section."""

import json
from pathlib import Path

import pytest

from capserve import config as cfg
from capserve.profile import CapacityProfile, ProfileError, WorkloadClass
from fakes.profiles import GATEWAY_PATH_SHA256, SLOW_QUEUE, admission_section, rehash, stage_v_copy, stage_v_doc
from fakes.profiles import v1_class

CLS = WorkloadClass(name="2k", max_prompt_tokens=1920, max_output_tokens=128, max_concurrency=4, max_queue=4,
                    source="t")


def test_defaults_from_a_minimal_toml(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('active_class = "2k"\n')
    s = cfg.load_settings(path)
    assert s.active_class == "2k"
    assert (s.host, s.port, s.backend_url) == ("127.0.0.1", 8080, "http://127.0.0.1:8100")
    assert (s.max_queue_wait_s, s.pre_admission_slots, s.request_deadline_s) == (2.0, 16, 300.0)
    assert (s.max_body_bytes, s.body_read_timeout_s, s.drain_timeout_s) == (1 << 20, 10.0, 60)
    assert (s.monitor_interval_s, s.monitor_timeout_s, s.monitor_fail_threshold) == (2.0, 2.0, 3)
    assert s.disconnect_poll_s == 0.25
    assert s.keep_alive_s == 75   # 043 §7.2: above vllm bench's aiohttp pool (60 s) and httpx's 5 s (uvicorn's is 5)
    assert s.retry_after_s.model_dump() == {"queue_full": 2, "queue_timeout": 2, "backend_unavailable": 5,
                                            "draining": 5, "mode": "adaptive", "min_s": 1.0, "max_s": 60.0}
    assert s.max_pending(CLS) == 4 + 4 + 16


def test_nested_tables_and_overrides(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('active_class = "2k"\nport = 9000\npre_admission_slots = 0\nrequest_deadline_s = 30\n'
                    'keep_alive_s = 120\n[retry_after_s]\nqueue_full = 7\n[tokens]\nchat_overhead = 40\n')
    s = cfg.load_settings(path)
    assert s.port == 9000 and s.retry_after_s.queue_full == 7 and s.retry_after_s.draining == 5
    assert s.keep_alive_s == 120
    assert s.tokens.chat_overhead == 40 and s.max_pending(CLS) == 8
    assert s.request_deadline_s == 30.0   # a TOML integer is a valid float


@pytest.mark.parametrize("text", [
    'active_class = "2k"\napi_key = "secret"\n',          # the key comes only from the environment
    'active_class = "2k"\nworkers = 2\n',                  # one process only (029 drain)
    'active_class = "2k"\nport = "8080"\n',
    'active_class = "2k"\n[retry_after_s]\nqueue_fll = 1\n',
    'port = 8080\n',
    'active_class = "2k"\nmax_queue_wait_s = 0\n',
    'active_class = "2k"\npre_admission_slots = -1\n',
    'active_class = "2k"\nkeep_alive_s = 0\n',            # every connection would close after its first response
    'active_class = "2k"\nkeep_alive_s = 75.5\n',         # uvicorn's timeout_keep_alive is whole seconds
    'active_class = "2k"\nnot toml',
])
def test_bad_settings_are_config_errors(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(text)
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings(path)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings(tmp_path / "absent.toml")


def test_api_key_only_from_the_environment_as_bytes():
    assert cfg.api_key_from_env({"CAPSERVE_API_KEY": "k-é"}) == "k-é".encode()
    assert cfg.api_key_from_env({"CAPSERVE_API_KEY": ""}) is None
    assert cfg.api_key_from_env({}) is None


def test_settings_never_hold_the_key(tmp_path):
    assert "api_key" not in cfg.Settings.model_fields


def test_allow_provisional_by_flag_or_env_exactly_1():
    assert cfg.allow_provisional(True, {}) is True
    assert cfg.allow_provisional(False, {"CAPSERVE_ALLOW_PROVISIONAL": "1"}) is True
    for value in ("0", "", "true", "yes", " 1"):
        assert cfg.allow_provisional(False, {"CAPSERVE_ALLOW_PROVISIONAL": value}) is False
    assert cfg.allow_provisional(False, {}) is False


def test_without_a_key_only_loopback_binds():
    for host in ("127.0.0.1", "::1", "localhost", "127.0.0.2"):
        cfg.check_bind(host, None)
    for host in ("0.0.0.0", "::", "192.168.1.5", "gpu-server"):
        with pytest.raises(cfg.ConfigError):
            cfg.check_bind(host, None)
        cfg.check_bind(host, b"k")


def test_the_local_tokenizer_needs_its_snapshot_path(tmp_path):
    """038 §4: tokenizer "local" counts prompts in the gateway from the served model's snapshot; "vllm" (default)
    asks vLLM's /tokenize. A path without local mode, or local mode without a path, is refused."""
    path = tmp_path / "c.toml"
    path.write_text('active_class = "2k"\n')
    assert (cfg.load_settings(path).tokenizer, cfg.load_settings(path).tokenizer_path) == ("vllm", None)
    path.write_text('active_class = "2k"\ntokenizer = "local"\ntokenizer_path = "/snap"\n')
    assert cfg.load_settings(path).tokenizer_path == "/snap"
    for text in ('tokenizer = "local"\n', 'tokenizer_path = "/snap"\n', 'tokenizer = "hf"\n'):
        path.write_text('active_class = "2k"\n' + text)
        with pytest.raises(cfg.ConfigError):
            cfg.load_settings(path)


def test_the_keys_file_comes_from_the_environment_and_excludes_the_single_key():
    """038 §5: CAPSERVE_API_KEYS_FILE (per-key quotas) or CAPSERVE_API_KEY (one key, no quota), never both."""
    assert cfg.keys_file_from_env({"CAPSERVE_API_KEYS_FILE": "/etc/k.toml"}) == "/etc/k.toml"
    assert cfg.keys_file_from_env({"CAPSERVE_API_KEYS_FILE": ""}) is None and cfg.keys_file_from_env({}) is None
    with pytest.raises(cfg.ConfigError, match="either"):
        cfg.keys_file_from_env({"CAPSERVE_API_KEYS_FILE": "/etc/k.toml", "CAPSERVE_API_KEY": "k"})


def test_tls_needs_both_files(tmp_path):
    """038 §6: [tls] certfile and keyfile, both; nothing else in the table."""
    path = tmp_path / "c.toml"
    path.write_text('active_class = "2k"\n[tls]\ncertfile = "/c.pem"\nkeyfile = "/k.pem"\n')
    s = cfg.load_settings(path)
    assert (s.tls.certfile, s.tls.keyfile) == ("/c.pem", "/k.pem")
    assert cfg.load_settings(tmp_path / "c.toml").tls is not None
    path.write_text('active_class = "2k"\n')
    assert cfg.load_settings(path).tls is None
    for table in ('certfile = "/c.pem"\n', 'certfile = "/c.pem"\nkeyfile = "/k.pem"\npassword = "x"\n'):
        path.write_text('active_class = "2k"\n[tls]\n' + table)
        with pytest.raises(cfg.ConfigError):
            cfg.load_settings(path)


def test_the_loopback_client_trusts_exactly_the_configured_certificate(tmp_path):
    import ssl
    from fakes.certs import self_signed
    cert, _ = self_signed(tmp_path)
    ctx = cfg.loopback_client_context(cfg.TLS(certfile=cert, keyfile="/unused"))
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is False
    assert ctx.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN



# ------------------------------------------------------------------ 044: candidate policies --

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
LOCAL = {"tokenizer": "local", "tokenizer_path": "/snap"}
ADMISSION = {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
             "classes": [{"name": "v-uniform-2k", "units": 1, "max_concurrency": 32},
                         {"name": "v-uniform-8k", "units": 4, "max_concurrency": 8}]}
PACING_2K = {"burst_tokens": 9600, "rate_tokens_per_s": 5920}
OTHER_PATH = "00" * 32


def settings(**over) -> cfg.Settings:
    return cfg.Settings.model_validate({"active_class": "v-uniform-2k", **LOCAL, **over})


def weighted(**over) -> cfg.Settings:
    return cfg.Settings.model_validate({**LOCAL, "admission": ADMISSION, **over})


def provisional(classes) -> CapacityProfile:
    doc = json.loads((PROFILES / "provisional-dev.json").read_text())
    doc.update(profile_id="provisional-v-test", classes=[
        {"name": n, "max_prompt_tokens": p, "max_output_tokens": o, "max_concurrency": c, "max_queue": c,
         "source": "test"} for n, p, o, c in classes])
    return CapacityProfile.model_validate(rehash(doc))


STUDY = provisional([("v-uniform-2k", 1920, 128, 32), ("v-uniform-8k", 8064, 128, 8)])
C1B = CapacityProfile.model_validate(json.loads((PROFILES / "verified-formal-20261017-c1b.json").read_text()))
V12 = CapacityProfile.model_validate(stage_v_doc())   # a §12 profile (the loader's rules: test_profile)


def test_candidate_policies_are_off_by_default(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('active_class = "2k"\n')
    s = cfg.load_settings(path)
    assert (s.admission, s.prefill_pacing, cfg.candidate_policies(s)) == (None, None, [])


def test_weighted_v1_replaces_the_active_class_and_sizes_the_pending_cap(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('tokenizer = "local"\ntokenizer_path = "/snap"\nmax_queue_wait_s = 120.0\n'
                    'request_deadline_s = 600.0\n[admission]\npolicy = "weighted-v1"\nbudget = 32\nmax_queue = 32\n'
                    '[[admission.classes]]\nname = "v-uniform-2k"\nunits = 1\nmax_concurrency = 32\n'
                    '[[admission.classes]]\nname = "v-uniform-8k"\nunits = 4\nmax_concurrency = 8\n')
    s = cfg.load_settings(path)
    assert s.active_class is None and s.admission.model_dump() == ADMISSION and cfg.candidate_policies(s) == [
        "admission"]
    assert s.max_pending() == 32 + 32 + 16 and s.admission.label == "weighted-v1:v-uniform-2k+v-uniform-8k"
    path.write_text('active_class = "2k"\ntokenizer = "local"\ntokenizer_path = "/snap"\n'
                    '[prefill_pacing]\nburst_tokens = 9600\nrate_tokens_per_s = 5920\n')
    s = cfg.load_settings(path)
    assert s.prefill_pacing.model_dump() == PACING_2K and cfg.candidate_policies(s) == ["prefill_pacing"]


@pytest.mark.parametrize("over, match", [
    ({"admission": ADMISSION}, "exactly one"),                                    # with an active class
    ({"active_class": None}, "exactly one"),                                      # neither
    ({"active_class": None, "admission": ADMISSION, "prefill_pacing": PACING_2K}, "never measured"),
    ({"tokenizer": "vllm", "tokenizer_path": None, "prefill_pacing": PACING_2K}, "local"),
    ({"active_class": None, "tokenizer": "vllm", "tokenizer_path": None, "admission": ADMISSION}, "local"),
    ({"active_class": None, "admission": {**ADMISSION, "classes": ADMISSION["classes"][:1]}}, "classes"),
    ({"active_class": None, "admission": {**ADMISSION, "policy": "weighted-v2"}}, "policy"),
    ({"active_class": None, "admission": {**ADMISSION, "order": "fifo"}}, "order"),
    ({"prefill_pacing": {**PACING_2K, "burst_tokens": 0}}, "burst_tokens"),
    ({"prefill_pacing": {**PACING_2K, "rate_tokens_per_s": 5920.5}}, "rate_tokens_per_s"),
])
def test_bad_candidate_policy_settings_are_refused(over, match):
    with pytest.raises(ValueError, match=match):
        settings(**over)


def test_without_a_candidate_policy_the_active_class_is_served_as_before():
    s = settings(active_class="uniform-2k", tokenizer="vllm", tokenizer_path=None)
    assert cfg.served_classes(s, C1B, gateway_path_sha256=OTHER_PATH) == [C1B.serving_class("uniform-2k")]
    assert cfg.served_classes(s, C1B, gateway_path_sha256=OTHER_PATH, study=False) == [C1B.workload_class("uniform-2k")]
    assert cfg.served_limits(s, [C1B.workload_class("uniform-2k")]) == {
        "max_concurrency": 32, "max_queue": 32, "max_pending": 32 + 32 + 16, "max_prompt_tokens": 1920,
        "max_output_tokens": 128}


def test_a_provisional_study_profile_serves_the_candidate_policies():
    """044 item 5: research (capserve serve --allow-provisional; a provisional profile loads only with it)."""
    s = settings(prefill_pacing=PACING_2K)
    assert [c.name for c in cfg.served_classes(s, STUDY, gateway_path_sha256=OTHER_PATH)] == ["v-uniform-2k"]
    s = weighted()
    got = cfg.served_classes(s, STUDY, gateway_path_sha256=OTHER_PATH)
    assert [c.name for c in got] == ["v-uniform-2k", "v-uniform-8k"]
    assert cfg.served_limits(s, got) == {"max_concurrency": 32, "max_queue": 32, "max_pending": 32 + 32 + 16,
                                         "max_prompt_tokens": 8064, "max_output_tokens": 128}


@pytest.mark.parametrize("s, profile, match", [
    (settings(prefill_pacing={**PACING_2K, "burst_tokens": 1919}), STUDY, "burst_tokens"),       # 044 item 3
    (weighted(admission={**ADMISSION, "classes": [{"name": "v-uniform-2k", "units": 1, "max_concurrency": 31},
                                                  ADMISSION["classes"][1]]}), STUDY, "max_concurrency"),
    (weighted(admission={**ADMISSION, "classes": [ADMISSION["classes"][0],
                                                  {"name": "v-uniform-8k", "units": 3, "max_concurrency": 8}]}),
     STUDY, "budget"),
    (weighted(admission={**ADMISSION, "classes": [ADMISSION["classes"][0],
                                                  {"name": "v-uniform-8k", "units": 2, "max_concurrency": 16}]}),
     STUDY, "the caps are the profile's limits"),                 # 2 x 16 = 32: only the cap rule refuses it
    (weighted(admission={**ADMISSION, "classes": ADMISSION["classes"][::-1]}), STUDY, "increasing"),
    (weighted(), provisional([("v-uniform-2k", 1920, 128, 32), ("v-uniform-8k", 8064, 512, 8)]), "max_output_tokens"),
    (weighted(), provisional([("v-uniform-2k", 1920, 128, 32), ("v-uniform-8k", 1920, 128, 8)]), "increasing"),
])
def test_a_candidate_policy_must_fit_the_classes_it_names(s, profile, match):
    with pytest.raises(cfg.ConfigError, match=match):
        cfg.served_classes(s, profile, gateway_path_sha256=OTHER_PATH)


def test_weighted_v1_names_only_classes_of_the_profile():
    s = weighted(admission={**ADMISSION, "classes": [ADMISSION["classes"][0],
                                                     {"name": "v-uniform-32k", "units": 4, "max_concurrency": 8}]})
    with pytest.raises(ProfileError, match="no class"):
        cfg.served_classes(s, STUDY, gateway_path_sha256=OTHER_PATH)


@pytest.mark.parametrize("s", [settings(active_class="uniform-2k", prefill_pacing=PACING_2K),
                               weighted(admission={**ADMISSION, "classes": [
                                   {"name": "uniform-2k", "units": 1, "max_concurrency": 32},
                                   {"name": "uniform-8k", "units": 4, "max_concurrency": 8}]})])
def test_a_verified_profile_without_a_section_12_never_serves_a_candidate_policy(s):
    """044 item 5: c1b (§8) carries no validated section, whatever flag is given."""
    with pytest.raises(ProfileError, match="044") as exc:
        cfg.served_classes(s, C1B, gateway_path_sha256=GATEWAY_PATH_SHA256)
    assert "§12" in str(exc.value)


@pytest.mark.parametrize("s", [settings(prefill_pacing=PACING_2K), weighted()])
def test_deploy_never_serves_a_candidate_policy_without_a_section_12(s):
    """044 item 5: a provisional profile's included (study=False is deploy's call)."""
    with pytest.raises(ProfileError, match="deploy"):
        cfg.served_classes(s, STUDY, gateway_path_sha256=OTHER_PATH, study=False)


PACED = dict(active_class="uniform-2k", prefill_pacing=PACING_2K, **SLOW_QUEUE)
W1 = dict(active_class=None, admission={**ADMISSION, "classes": [
    {"name": "uniform-2k", "units": 1, "max_concurrency": 32},
    {"name": "uniform-8k", "units": 4, "max_concurrency": 8}]}, **SLOW_QUEUE)


@pytest.mark.parametrize("over", [PACED, W1])
@pytest.mark.parametrize("study", [True, False])
def test_a_section_12_profile_serves_exactly_its_validated_section(over, study):
    s = settings(**over)
    got = cfg.served_classes(s, V12, gateway_path_sha256=GATEWAY_PATH_SHA256, study=study)
    assert [c.name for c in got] == (["uniform-2k"] if "prefill_pacing" in over else ["uniform-2k", "uniform-8k"])


@pytest.mark.parametrize("over, path, match", [
    ({**PACED, "prefill_pacing": {**PACING_2K, "burst_tokens": 9601}}, GATEWAY_PATH_SHA256, "prefill_pacing"),
    ({**PACED, "prefill_pacing": {**PACING_2K, "rate_tokens_per_s": 5930}}, GATEWAY_PATH_SHA256, "prefill_pacing"),
    ({**PACED, "max_queue_wait_s": 2.0}, GATEWAY_PATH_SHA256, "queue"),                  # the product's fast 429
    ({**PACED, "request_deadline_s": 300.0}, GATEWAY_PATH_SHA256, "queue"),
    (PACED, OTHER_PATH, "gateway_path_sha256"),                                           # another request path
    ({**W1, "admission": {**W1["admission"], "max_queue": 31}}, GATEWAY_PATH_SHA256, "admission"),
    ({**W1, "max_queue_wait_s": 2.0}, GATEWAY_PATH_SHA256, "queue"),
    (W1, OTHER_PATH, "gateway_path_sha256"),
])
def test_a_section_12_profile_refuses_any_other_policy_queue_or_request_path(over, path, match):
    with pytest.raises(ProfileError, match=match):
        cfg.served_classes(settings(**over), V12, gateway_path_sha256=path)


def test_a_section_12_profile_without_the_section_refuses_the_policy():
    plain = CapacityProfile.model_validate(stage_v_doc(classes=[stage_v_copy("U2K", pacing=False),
                                                                stage_v_copy("U8K", pacing=False)]))
    with pytest.raises(ProfileError, match="pacing"):
        cfg.served_classes(settings(**PACED), plain, gateway_path_sha256=GATEWAY_PATH_SHA256)
    paced = CapacityProfile.model_validate(stage_v_doc(admission=None))
    with pytest.raises(ProfileError, match="admission"):
        cfg.served_classes(settings(**W1), paced, gateway_path_sha256=GATEWAY_PATH_SHA256)
    v1 = CapacityProfile.model_validate(stage_v_doc(classes=[v1_class()], admission=None))
    s = settings(active_class="uniform-2k-o1-512")   # V1's class: no candidate policy, any request path
    assert [c.max_output_tokens for c in cfg.served_classes(s, v1, gateway_path_sha256=OTHER_PATH)] == [512]
    assert admission_section()["queue"] == SLOW_QUEUE
