"""python -m capserve serve --config F --profile P [--allow-provisional]  (decision 029)

Loads the settings (TOML), the API key (CAPSERVE_API_KEY, environment only) and the capacity profile, then serves
one process under uvicorn (server.run). A provisional profile is refused (exit 2) unless --allow-provisional or
CAPSERVE_ALLOW_PROVISIONAL=1 is given (027 §4); when allowed, the startup line, /readyz, /metrics and every response
header say provisional. Without an API key only a loopback host is accepted. Configuration errors exit 2.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from capserve.config import ConfigError, allow_provisional, api_key_from_env, check_bind, load_settings
from capserve.profile import ProfileError, load_profile


def main(argv: list | None = None, *, env=None, run=None) -> int:
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
        api_key = api_key_from_env(env)
        check_bind(settings.host, api_key)
        profile = load_profile(args.profile, allow_provisional=allow_provisional(args.allow_provisional, env))
        cls = profile.workload_class(settings.active_class)
    except (ConfigError, ProfileError) as exc:
        print(f"capserve: {exc}", file=sys.stderr)
        return 2
    if not profile.verified:
        print(f"capserve: serving PROVISIONAL profile {profile.profile_id} (class {cls.name}): its limits are not a "
              f"verified capacity ({profile.provisional_reason})", file=sys.stderr)
    if run is None:
        from capserve.server import run
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(settings, profile, api_key=api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
