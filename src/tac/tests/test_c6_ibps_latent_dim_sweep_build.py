# SPDX-License-Identifier: MIT
"""Tests for C6 IBPS reactivation path (b) latent_dim sweep BUILD.

Lane: lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518
Per R3 SEAL council op-routable #2 + C6 IBPS empirical falsification 2026-05-17.

Test coverage:
- Architecture supports latent_dim ∈ {24, 48, 96, 192}
- Param count diagnostics per width
- 3 NEW recipes exist + parse + pass Catalog #270 dispatch optimization protocol
- Catalog #240 recipe-vs-trainer-state consistency
- Catalog #324 predicted_band post-training validation
- Catalog #220 distinguishing_feature_* fields declared per recipe
- Sister regression: baseline C6 still passes
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REMOTE_DRIVER = REPO_ROOT / "scripts" / "remote_lane_substrate_c6_e4_mdl_ibps.sh"

LATENT_DIMS = (48, 96, 192)
BASELINE_LATENT_DIM = 24

# Expected param breakdowns verified empirically 2026-05-18
EXPECTED_PARAM_BREAKDOWN = {
    24: {"encoder": 35664, "decoder": 78818, "latents": 14400, "total": 128882},
    48: {"encoder": 38784, "decoder": 88034, "latents": 28800, "total": 155618},
    96: {"encoder": 45024, "decoder": 106466, "latents": 57600, "total": 209090},
    192: {"encoder": 57504, "decoder": 143330, "latents": 115200, "total": 316034},
}


def _recipe_path(latent_dim: int) -> Path:
    return (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / f"substrate_c6_e4_mdl_ibps_latent{latent_dim}_modal_t4_smoke_dispatch.yaml"
    )


# ---------------------------------------------------------------------------
# Architecture coverage
# ---------------------------------------------------------------------------


def test_architecture_supports_latent_dim_48():
    """latent_dim=48 (2x baseline) instantiates + matches expected param count."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    cfg = MDLIBPSConfig(latent_dim=48, num_pairs=600)
    sub = MDLIBPSSubstrate(cfg)
    bd = sub.num_parameters_breakdown()
    assert bd == EXPECTED_PARAM_BREAKDOWN[48], (
        f"latent_48 param breakdown drift: got {bd}, expected {EXPECTED_PARAM_BREAKDOWN[48]}"
    )


def test_architecture_supports_latent_dim_96():
    """latent_dim=96 (4x baseline) instantiates + matches expected param count."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    cfg = MDLIBPSConfig(latent_dim=96, num_pairs=600)
    sub = MDLIBPSSubstrate(cfg)
    bd = sub.num_parameters_breakdown()
    assert bd == EXPECTED_PARAM_BREAKDOWN[96], (
        f"latent_96 param breakdown drift: got {bd}, expected {EXPECTED_PARAM_BREAKDOWN[96]}"
    )


def test_architecture_supports_latent_dim_192():
    """latent_dim=192 (8x baseline) instantiates + matches expected param count."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    cfg = MDLIBPSConfig(latent_dim=192, num_pairs=600)
    sub = MDLIBPSSubstrate(cfg)
    bd = sub.num_parameters_breakdown()
    assert bd == EXPECTED_PARAM_BREAKDOWN[192], (
        f"latent_192 param breakdown drift: got {bd}, expected {EXPECTED_PARAM_BREAKDOWN[192]}"
    )


def test_architecture_latent_dim_baseline_unchanged():
    """Sister regression: latent_dim=24 baseline param breakdown unchanged."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    cfg = MDLIBPSConfig(latent_dim=24, num_pairs=600)
    sub = MDLIBPSSubstrate(cfg)
    bd = sub.num_parameters_breakdown()
    assert bd == EXPECTED_PARAM_BREAKDOWN[24], (
        f"baseline latent_24 param breakdown drift: got {bd}, expected {EXPECTED_PARAM_BREAKDOWN[24]}"
    )


def test_architecture_param_count_monotonic():
    """Total param count strictly grows with latent_dim (sanity)."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    totals = []
    for ld in (24, 48, 96, 192):
        cfg = MDLIBPSConfig(latent_dim=ld, num_pairs=600)
        sub = MDLIBPSSubstrate(cfg)
        totals.append(sub.num_parameters_breakdown()["total"])
    assert totals == sorted(totals), f"param totals not monotonic: {totals}"
    # Strict increase
    for i in range(1, len(totals)):
        assert totals[i] > totals[i - 1], f"non-strict increase at index {i}: {totals}"


def test_architecture_forward_pass_per_width():
    """Forward pass succeeds at each width (smoke shape + dtype contract)."""
    import torch

    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )

    for ld in LATENT_DIMS:
        cfg = MDLIBPSConfig(
            latent_dim=ld,
            num_pairs=4,
            decoder_embed_dim=16,
            decoder_channels=(12, 10, 8, 6),
            decoder_num_upsample_blocks=4,
            output_height=48,
            output_width=64,
        )
        sub = MDLIBPSSubstrate(cfg)
        idx = torch.arange(4, dtype=torch.long)
        frames = torch.rand(4, 3, 48, 64)
        rgb_0, rgb_1, mu, logvar = sub(idx, frames_for_encoder=frames)
        assert rgb_0.shape == (4, 3, 48, 64), f"rgb_0 shape drift at ld={ld}: {rgb_0.shape}"
        assert rgb_1.shape == (4, 3, 48, 64), f"rgb_1 shape drift at ld={ld}: {rgb_1.shape}"
        assert mu.shape == (4, ld), f"mu shape drift at ld={ld}: {mu.shape}"
        assert logvar.shape == (4, ld), f"logvar shape drift at ld={ld}: {logvar.shape}"


# ---------------------------------------------------------------------------
# Recipe existence + parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_file_exists(latent_dim: int):
    """Each sweep recipe YAML exists at canonical path."""
    p = _recipe_path(latent_dim)
    assert p.is_file(), f"recipe missing: {p}"


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_yaml_parses(latent_dim: int):
    """Each sweep recipe parses as YAML + has canonical top-level fields."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"recipe {p} did not parse as dict"
    assert data["schema_version"] == 1
    assert data["name"] == f"substrate_c6_e4_mdl_ibps_latent{latent_dim}_modal_t4_smoke_dispatch"
    assert data["lane_id"] == "lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518"


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_env_overrides_threads_latent_dim(latent_dim: int):
    """env_overrides.C6_E4_MDL_IBPS_LATENT_DIM matches recipe name's latent_dim."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = data.get("env_overrides", {})
    actual = env.get("C6_E4_MDL_IBPS_LATENT_DIM")
    assert actual == str(latent_dim), (
        f"recipe {p.name}: env C6_E4_MDL_IBPS_LATENT_DIM={actual!r} "
        f"does not match expected {latent_dim!r}"
    )


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_env_overrides_thread_sweep_lane_and_recipe(latent_dim: int):
    """Sweep recipe env must override the shared baseline C6 driver identity."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = data.get("env_overrides", {})
    assert env.get("C6_E4_MDL_IBPS_LANE_ID") == (
        "lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518"
    )
    assert env.get("C6_E4_MDL_IBPS_RECIPE_PATH") == (
        f".omx/operator_authorize_recipes/"
        f"substrate_c6_e4_mdl_ibps_latent{latent_dim}_modal_t4_smoke_dispatch.yaml"
    )
    assert env.get("TAG") == f"substrate_c6_e4_mdl_ibps_latent{latent_dim}"


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_is_non_dispatchable_pending_operator_signoff(latent_dim: int):
    """Recipe is research_only=true + dispatch_enabled=false per Catalog #240."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data.get("research_only") is True, (
        f"recipe {p.name}: research_only must be True (current: {data.get('research_only')})"
    )
    assert data.get("dispatch_enabled") is False, (
        f"recipe {p.name}: dispatch_enabled must be False (current: {data.get('dispatch_enabled')})"
    )
    blockers = data.get("dispatch_blockers", [])
    assert len(blockers) >= 4, (
        f"recipe {p.name}: dispatch_blockers must have at least 4 entries; got {len(blockers)}"
    )


# ---------------------------------------------------------------------------
# Catalog #324 predicted_band post-training validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_predicted_band_validation_status_pending(latent_dim: int):
    """Per Catalog #324: predicted_band_validation_status must be pending_post_training."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    status = data.get("predicted_band_validation_status")
    assert status == "pending_post_training", (
        f"recipe {p.name}: predicted_band_validation_status must be "
        f"'pending_post_training' per Catalog #324 (got {status!r})"
    )
    assert data.get("predicted_band_reactivation_criteria"), (
        f"recipe {p.name}: predicted_band_reactivation_criteria required when pending"
    )
    # predicted_score_target should be null when pending
    assert data.get("predicted_score_target") is None, (
        f"recipe {p.name}: predicted_score_target should be None when pending_post_training "
        f"(got {data.get('predicted_score_target')!r})"
    )


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_passes_catalog_324_validator(latent_dim: int):
    """Each recipe passes the canonical Catalog #324 validator."""
    from tac.optimization.tier_c_density_post_training_validator import (
        validate_recipe_predicted_band,
    )

    p = _recipe_path(latent_dim)
    verdict = validate_recipe_predicted_band(p)
    assert verdict.is_valid, (
        f"recipe {p.name}: Catalog #324 validator rejected: "
        f"status={verdict.validation_status} blockers={verdict.blockers}"
    )


# ---------------------------------------------------------------------------
# Catalog #270 dispatch optimization protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_passes_catalog_270_dispatch_protocol(latent_dim: int):
    """Each recipe passes the Catalog #270 umbrella dispatch protocol (5/5+8/8+5/5)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from canonical_dispatch_optimization_protocol import (
        verify_dispatch_protocol_complete,
    )

    trainer = REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    recipe_name = f"substrate_c6_e4_mdl_ibps_latent{latent_dim}_modal_t4_smoke_dispatch"
    verdict = verify_dispatch_protocol_complete(
        trainer=trainer,
        recipe=recipe_name,
    )
    assert verdict.overall_pass, (
        f"recipe {recipe_name}: Catalog #270 dispatch protocol FAILED. "
        f"blockers={verdict.blockers}"
    )
    # All tiers pass
    assert all(verdict.tier1.pass_signals.values()), (
        f"recipe {recipe_name}: Tier 1 signal failures: "
        f"{[k for k, v in verdict.tier1.pass_signals.items() if not v]}"
    )
    assert all(verdict.tier2.pass_signals.values()), (
        f"recipe {recipe_name}: Tier 2 signal failures: "
        f"{[k for k, v in verdict.tier2.pass_signals.items() if not v]}"
    )
    assert all(verdict.tier3.pass_signals.values()), (
        f"recipe {recipe_name}: Tier 3 signal failures: "
        f"{[k for k, v in verdict.tier3.pass_signals.items() if not v]}"
    )


# ---------------------------------------------------------------------------
# Catalog #220 + #272 distinguishing feature integration contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_declares_distinguishing_feature_fields(latent_dim: int):
    """Per Catalog #220 + #272: distinguishing_feature_* fields declared."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "distinguishing_feature_name" in data, (
        f"recipe {p.name}: distinguishing_feature_name required per Catalog #272"
    )
    assert "distinguishing_bytes_path" in data, (
        f"recipe {p.name}: distinguishing_bytes_path required per Catalog #272"
    )
    assert "inflate_consumer_function" in data, (
        f"recipe {p.name}: inflate_consumer_function required per Catalog #272"
    )
    # The function MUST exist in the substrate package
    assert data["inflate_consumer_function"] == (
        "tac.substrates.c6_e4_mdl_ibps.inflate.inflate_one_video"
    ), f"recipe {p.name}: inflate_consumer_function should point to canonical inflate"


# ---------------------------------------------------------------------------
# Sister regression: baseline C6 unaffected
# ---------------------------------------------------------------------------


def test_baseline_c6_recipe_still_passes_catalog_270():
    """Sister regression: baseline C6 dispatch protocol unchanged."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from canonical_dispatch_optimization_protocol import (
        verify_dispatch_protocol_complete,
    )

    trainer = REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    verdict = verify_dispatch_protocol_complete(
        trainer=trainer,
        recipe="substrate_c6_e4_mdl_ibps_modal_t4_dispatch",
    )
    assert verdict.overall_pass, (
        f"BASELINE C6 dispatch protocol regression: {verdict.blockers}"
    )


def test_baseline_c6_recipe_still_passes_catalog_324():
    """Sister regression: baseline C6 still passes Catalog #324."""
    from tac.optimization.tier_c_density_post_training_validator import (
        validate_recipe_predicted_band,
    )

    p = (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
    )
    verdict = validate_recipe_predicted_band(p)
    assert verdict.is_valid, (
        f"BASELINE C6 Catalog #324 regression: {verdict.blockers}"
    )


# ---------------------------------------------------------------------------
# Recipe lane_id consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", LATENT_DIMS)
def test_recipe_lane_id_canonical(latent_dim: int):
    """All sweep recipes share canonical lane_id (umbrella sweep lane)."""
    import yaml

    p = _recipe_path(latent_dim)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data.get("lane_id") == (
        "lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518"
    ), f"recipe {p.name}: lane_id drift: {data.get('lane_id')!r}"


# ---------------------------------------------------------------------------
# Cost band ladder discipline (sequential dispatch order)
# ---------------------------------------------------------------------------


def test_recipe_cost_band_ladder_increasing():
    """Cost should increase with model size (latent_48 < latent_192)."""
    import yaml

    costs = []
    for ld in LATENT_DIMS:
        data = yaml.safe_load(_recipe_path(ld).read_text(encoding="utf-8"))
        cost = data.get("cost_band", {}).get("predicted_cost_usd")
        assert cost is not None, f"latent_{ld} missing cost_band.predicted_cost_usd"
        costs.append(cost)
    # latent_48 ≤ latent_96 ≤ latent_192 (monotonic non-decreasing)
    for i in range(1, len(costs)):
        assert costs[i] >= costs[i - 1], (
            f"cost band ladder regression: latent_{LATENT_DIMS[i]}=${costs[i]} "
            f"< latent_{LATENT_DIMS[i - 1]}=${costs[i - 1]}"
        )


def test_recipe_canary_dependency_chain_correct():
    """Sequential canary chain: baseline→48→96→192."""
    import yaml

    expected_canary = {
        48: "c6_e4_mdl_ibps",  # baseline IS the canary
        96: "c6_e4_mdl_ibps_latent48",  # latent_48 IS the canary for 96
        192: "c6_e4_mdl_ibps_latent96",  # latent_96 IS the canary for 192
    }
    for ld, expected in expected_canary.items():
        data = yaml.safe_load(_recipe_path(ld).read_text(encoding="utf-8"))
        actual = data.get("canary_dependency")
        assert actual == expected, (
            f"latent_{ld} canary_dependency drift: got {actual!r}, expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# Remote-driver dispatch contract
# ---------------------------------------------------------------------------


def test_c6_remote_driver_bash_syntax_clean() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_c6_remote_driver_threads_latent_sweep_lane_recipe_and_flags(
    tmp_path: Path,
) -> None:
    """Runtime proof: latent sweep env reaches claim verification, provenance, and argv."""

    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)

    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["C6_E4_MDL_IBPS_LANE_ID"],
            "instance_job_id": os.environ["C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )

    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )

    trainer = workspace / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
(out / "stats.json").write_text(
    json.dumps({
        "auth_eval_score_claim_valid": True,
        "auth_eval_score_axis": "contest_cuda",
        "auth_eval_exact_cuda_complete": True,
        "auth_eval_score": 9.99,
        "auth_eval_result_path": "fake/contest_auth_eval.json",
    }),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    lane_id = "lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(output_dir),
            "C6_E4_MDL_IBPS_OUTPUT_DIR": str(output_dir),
            "C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID": "c6-latent48-test-job",
            "C6_E4_MDL_IBPS_LANE_ID": lane_id,
            "C6_E4_MDL_IBPS_RECIPE_PATH": (
                ".omx/operator_authorize_recipes/"
                "substrate_c6_e4_mdl_ibps_latent48_modal_t4_smoke_dispatch.yaml"
            ),
            "TAG": "substrate_c6_e4_mdl_ibps_latent48",
            "C6_E4_MDL_IBPS_EPOCHS": "50",
            "C6_E4_MDL_IBPS_LATENT_DIM": "48",
            "C6_E4_MDL_IBPS_BETA_IB": "0.01",
            "C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16": "true",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, (
        "C6 remote lane script failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Stage 0 DONE: active dispatch claim verified" in result.stdout
    argv = json.loads((output_dir / "argv.json").read_text(encoding="utf-8"))
    expected_pairs = {
        "--epochs": "50",
        "--latent-dim": "48",
        "--beta-ib": "0.01",
    }
    for flag, value in expected_pairs.items():
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value
    assert "--enable-autocast-fp16" in argv

    provenance = json.loads((tmp_path / "logs" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["lane_id"] == lane_id
    assert provenance["recipe"].endswith(
        "substrate_c6_e4_mdl_ibps_latent48_modal_t4_smoke_dispatch.yaml"
    )
    assert provenance["tag"] == "substrate_c6_e4_mdl_ibps_latent48"
    assert provenance["latent_dim"] == "48"
