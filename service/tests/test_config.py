"""Tests for capserve/config.py: settings from TOML; the environment supplies only CAPSERVE_API_KEY and
CAPSERVE_ALLOW_PROVISIONAL (decision 029)."""

import pytest

from capserve import config as cfg
from capserve.profile import WorkloadClass

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
    assert s.keep_alive_s == 75   # 043 §7.2: above aiohttp's 60 s and httpx's 5 s pools (uvicorn's own default is 5)
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
