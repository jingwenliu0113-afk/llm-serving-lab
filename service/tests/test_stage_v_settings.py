"""Decision 044: the gateway settings the Stage V harness renders (stage_v_plan.gateway_settings through its nested TOML
writer) are settings capserve loads and serves on each point's profile, with the /readyz limits, queue settings and
candidate policy the plan expects, and the real gateway's /readyz body gives, through the runner's readyz_tuple, the
tuple the runner's readyz_want freezes for that point (實作補記六). Research never imports capserve, so this
cross-check lives on the product side (a service test may import the research harness)."""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from capserve import config as cfgmod
from capserve import gateway_path as gp
from capserve import profile as cp
from capserve.app import create_app
from capserve.tokens import LocalTokenizer
from fakes.fake_tokenizer import WordsHF

svp = pytest.importorskip("stage_v_plan")
rsv = pytest.importorskip("run_stage_v")

ROOT = Path(__file__).resolve().parents[2]


def build(session: str) -> dict:
    raw = {n: (ROOT / "configs" / f"{n}.toml").read_bytes() for n in ("stage_v", "stage_d", "pins")}
    cfg, sd_cfg, pins = (tomllib.loads(raw[n].decode()) for n in ("stage_v", "stage_d", "pins"))
    return svp.build_plan(cfg, pins, sd_cfg, session_name=session, pins_sha256=hashlib.sha256(raw["pins"]).hexdigest(),
                          stage_v_sha256=hashlib.sha256(raw["stage_v"]).hexdigest(),
                          stage_d_sha256=hashlib.sha256(raw["stage_d"]).hexdigest())


@pytest.mark.parametrize("session", ["mini", "V-ADM", "V-CLS"])
def test_every_points_settings_load_and_serve_on_its_profile(tmp_path, session):
    plan = build(session)
    for name, x in plan["points"].items():
        path = tmp_path / f"{name}.toml"
        path.write_text(svp.toml_dumps(x["gateway"]))
        settings = cfgmod.load_settings(path)
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        assert prof.verified is (not x["provisional"]), name
        classes = cfgmod.served_classes(settings, prof, gateway_path_sha256=gp.gateway_path_sha256(), study=True)
        assert [c.name for c in classes] == x["classes"], name
        limits = cfgmod.served_limits(settings, classes)
        assert {k: limits[k] for k in x["limits"]} == x["limits"], name
        assert cfgmod.queue_settings(settings) == x["queue_settings"], name
        want = (["admission"] if x["admission"] else []) + (["prefill_pacing"] if x["pacing"] else [])
        assert cfgmod.candidate_policies(settings) == want, name
        assert (settings.host, settings.port, settings.tokenizer) == ("127.0.0.1", 8080, "local")
        assert settings.backend_url == "http://127.0.0.1:8000" and settings.drain_timeout_s == 5


def test_no_candidate_policy_of_the_study_deploys():
    """044 item 5: the research settings serve a candidate policy only as a study (capserve serve
    --allow-provisional); deploy (study=False) refuses each of them."""
    plan = build("V-ADM")
    for name in ("P-U2K", "W-90"):
        x = plan["points"][name]
        settings = cfgmod.Settings.model_validate(x["gateway"])
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        with pytest.raises(cp.ProfileError, match="deploy never serves a candidate policy"):
            cfgmod.served_classes(settings, prof, gateway_path_sha256=gp.gateway_path_sha256(), study=False)


@pytest.mark.parametrize("session", ["mini", "V-ADM", "V-CLS"])
def test_the_real_gateways_readyz_gives_each_points_frozen_tuple(tmp_path, session):
    """044 實作補記六: for every point of every session, the gateway capserve builds from the point's rendered settings
    and profile (a local tokenizer injected: no snapshot on the Mac) reports, through run_stage_v.readyz_tuple, exactly
    run_stage_v.readyz_want; the runner's fakes (tests/test_run_stage_v) only model that body."""
    plan = build(session)
    sha = svp.gateway_path_sha256()
    assert sha == gp.gateway_path_sha256()
    for name, x in plan["points"].items():
        path = tmp_path / f"{name}.toml"
        path.write_text(svp.toml_dumps(x["gateway"]))
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        gateway = create_app(cfgmod.load_settings(path), prof, local_tokenizer=LocalTokenizer(WordsHF()))
        profile_id = json.loads((ROOT / x["profile"]).read_text())["profile_id"]
        want = rsv.readyz_want(plan, name, profile_id=profile_id, gateway_path_sha256=sha)
        assert rsv.readyz_tuple(gateway.readiness()) == want, name
        assert None not in (want["profile"]["id"], want["serving_status"], want["limits"], want["queue"]), name
