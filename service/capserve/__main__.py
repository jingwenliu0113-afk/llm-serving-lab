"""python -m capserve serve --config F --profile P [--allow-provisional]  (decision 029)

Loads the settings (TOML), the API key (CAPSERVE_API_KEY, environment only) and the capacity profile, then serves
one process under uvicorn (server.run). A provisional profile is refused (exit 2) unless --allow-provisional or
CAPSERVE_ALLOW_PROVISIONAL=1 is given (027 §4); when allowed, the startup line, /readyz, /metrics and every response
header say provisional. A verified profile needs no flag; the loader checks its provenance against this checkout's
frozen configs (capserve.profile), and it is served only with the bytes of one of this checkout's
service/profiles/<name>.json files (the committed profiles the Mac test regenerates from the frozen analysis; deploy
passes a copy of one, runs/<id>/profile.json), never a hand-written file elsewhere. An unavailable active class of a
verified profile (decision 030 8.7: Stage C verified no number for it) exits 2 whatever flag is given; a verified 0
class (refuse it) starts, says so, and answers every completion request 503 class_not_serviceable. Without an API key
only a loopback host is accepted. Configuration errors exit 2.
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import sys
from pathlib import Path

from capserve.config import (ConfigError, allow_provisional, api_key_from_env, check_bind, keys_file_from_env,
                             load_settings)
from capserve.keys import KeyFileError, load_keyring
from capserve.profile import ProfileError, parse_profile
from capserve.tokens import LocalTokenizer, TokenizerLoadError

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"   # service/profiles of this checkout or release
CHECKOUT = Path(__file__).resolve().parents[2]                     # this checkout or release: no keys file in it


def committed_profile_problems(data: bytes, profiles_dir) -> list:
    """[] when data are the bytes of one of profiles_dir's visible <name>.json files (deploy's location rule)."""
    try:
        known = {p.read_bytes() for p in Path(profiles_dir).glob("*.json")
                 if p.suffix == ".json" and not p.name.startswith(".")}
    except OSError as exc:
        return [f"cannot read {profiles_dir}: {exc}"]
    if data in known:
        return []
    return [f"a verified profile is served only with the bytes of one of {profiles_dir}/<name>.json (service/profiles: "
            "the committed profiles the generator reproduces, 029), never a hand-written file elsewhere"]


def tls_problems(tls, checkout=CHECKOUT) -> list:
    """038 §6: the certificate and key are the user's, outside the checkout or release, readable and a pair."""
    problems = []
    for name in ("certfile", "keyfile"):
        path = Path(getattr(tls, name))
        if path.resolve().is_relative_to(checkout):
            problems.append(f"[tls] {name} {path} is inside {checkout} (the checkout or release)")
        elif not os.access(path, os.R_OK) or not path.is_file():
            problems.append(f"[tls] cannot read {name} {path}")
    if not problems:
        try:
            ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(tls.certfile, tls.keyfile)
        except (ssl.SSLError, OSError) as exc:
            problems.append(f"[tls] cannot load the certificate and key: {exc}")
    return problems


def main(argv: list | None = None, *, env=None, run=None, frozen=None, profiles_dir=PROFILES_DIR) -> int:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(prog="python -m capserve", description="capacity-aware gateway in front of vLLM")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="serve the OpenAI-compatible API under one capacity profile")
    serve.add_argument("--config", required=True, help="gateway settings (TOML)")
    serve.add_argument("--profile", required=True, help="capacity profile (JSON)")
    serve.add_argument("--allow-provisional", action="store_true",
                       help="allow a provisional profile (development / integration tests; not product limits)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if settings.tls is not None and (problems := tls_problems(settings.tls)):
            raise ConfigError("; ".join(problems))
        api_key = api_key_from_env(env)
        keyring = None
        if keys_path := keys_file_from_env(env):   # 038 §5: names and sha256 only, outside anything deployed or committed
            if Path(keys_path).resolve().is_relative_to(CHECKOUT):
                raise ConfigError(f"the keys file must not be inside {CHECKOUT} (the checkout or release)")
            keyring = load_keyring(keys_path)
        check_bind(settings.host, api_key or keyring)
        try:
            data = Path(args.profile).read_bytes()      # read once: these bytes are checked and served
        except OSError as exc:
            raise ProfileError(f"{args.profile}: cannot read the profile: {exc}") from exc
        profile = parse_profile(data, source=args.profile, frozen=frozen,
                                allow_provisional=allow_provisional(args.allow_provisional, env))
        if profile.verified and (problems := committed_profile_problems(data, profiles_dir)):
            raise ProfileError(f"{args.profile}: " + "; ".join(problems))
        cls = profile.serving_class(settings.active_class)   # an unavailable class: refused, no flag overrides it
        # 038 §4: a local tokenizer is loaded before serving; one that cannot load is a configuration error
        local = LocalTokenizer.load(settings.tokenizer_path) if settings.tokenizer == "local" else None
    except (ConfigError, ProfileError, TokenizerLoadError, KeyFileError) as exc:
        print(f"capserve: {exc}", file=sys.stderr)
        return 2
    if not profile.verified:
        print(f"capserve: serving PROVISIONAL profile {profile.profile_id} (class {cls.name}): its limits are not a "
              f"verified capacity ({profile.provisional_reason})", file=sys.stderr)
    if not cls.serviceable:
        print(f"capserve: class {cls.name} of verified profile {profile.profile_id} has a verified capacity of 0 "
              "(Stage C INFEASIBLE, 030 8.7): every completion request is answered 503 class_not_serviceable and "
              "/readyz is never ready", file=sys.stderr)
    if run is None:
        from capserve.server import run
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(settings, profile, api_key=api_key, local_tokenizer=local, keyring=keyring)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
