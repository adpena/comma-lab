# SPDX-License-Identifier: MIT
"""Tests for C6 IBPS reactivation path (a) β_ib sweep BUILD.

Lane: lane_c6_ibps_beta_ib_sweep_build_per_symposium_parallel_path_a_20260518
Per #848 C6 IBPS symposium parallel-path (a) MacKay MDL verdict + Contrarian
binding revision (parallel paths (a)+(b) under $5 envelope cap).

Test coverage:
- Architecture supports β_ib ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}
  (β_ib does NOT affect architecture params; latent_dim=24 unchanged)
- Trainer --beta-ib flag wired in TIER_1_OPERATOR_REQUIRED_FLAGS + argparse + downstream
- 5 sweep recipes exist + parse + pass Catalog #270 dispatch optimization protocol
- Catalog #240 recipe-vs-trainer-state consistency
- Catalog #324 predicted_band post-training validation
- Catalog #220 distinguishing_feature_* fields declared per recipe
- Sister regression: baseline C6 still passes; sister #846 latent_dim BUILD still passes
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REMOTE_DRIVER = REPO_ROOT / "scripts" / "remote_lane_substrate_c6_e4_mdl_ibps.sh"

# 5 β_ib sweep values, including the beta_ib=0.01 fixed-width control.
BETA_IB_SWEEP_VALUES = ("0p0001", "0p001", "0p01", "0p1", "1p0")
BETA_IB_NUMERIC = {
    "0p0001": 0.0001,
    "0p001": 0.001,
    "0p01": 0.01,
    "0p1": 0.1,
    "1p0": 1.0,
}
BASELINE_BETA_IB = 0.01
BASELINE_LATENT_DIM = 24

UMBRELLA_LANE_ID = (
    "lane_c6_ibps_beta_ib_sweep_build_per_symposium_parallel_path_a_20260518"
)


def _recipe_path(beta_ib_token: str) -> Path:
    return (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / f"substrate_c6_e4_mdl_ibps_beta_ib_{beta_ib_token}_modal_t4_smoke_dispatch.yaml"
    )


# ---------------------------------------------------------------------------
# Trainer + architecture support (zero LOC delta required)
# ---------------------------------------------------------------------------


def test_trainer_declares_beta_ib_in_tier1_operator_required_flags():
    """--beta-ib must be in TIER_1_OPERATOR_REQUIRED_FLAGS per Catalog #151."""
    trainer_src = (
        REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    ).read_text(encoding="utf-8")
    assert '"--beta-ib"' in trainer_src, (
        "Trainer does not declare --beta-ib in TIER_1_OPERATOR_REQUIRED_FLAGS"
    )
    assert '"env": "C6_E4_MDL_IBPS_BETA_IB"' in trainer_src, (
        "Trainer manifest does not declare canonical env C6_E4_MDL_IBPS_BETA_IB"
    )


def test_trainer_argparse_declares_beta_ib():
    """argparse declares --beta-ib flag (zero LOC delta needed)."""
    trainer_src = (
        REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    ).read_text(encoding="utf-8")
    assert 'p.add_argument("--beta-ib"' in trainer_src, (
        "argparse does not declare --beta-ib (per CLAUDE.md NEVER invent CLI flags)"
    )


def test_trainer_threads_beta_ib_to_mdlibpsconfig():
    """Trainer threads args.beta_ib to MDLIBPSConfig at _full_main."""
    trainer_src = (
        REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    ).read_text(encoding="utf-8")
    assert "beta_ib=args.beta_ib" in trainer_src, (
        "Trainer does not thread args.beta_ib to MDLIBPSConfig"
    )


def test_architecture_supports_arbitrary_beta_ib_values():
    """MDLIBPSConfig.beta_ib accepts arbitrary values (sweep range [0.0001, 1.0])."""
    from tac.substrates.c6_e4_mdl_ibps.architecture import MDLIBPSConfig

    for value in (0.0001, 0.001, 0.01, 0.1, 1.0):
        cfg = MDLIBPSConfig(latent_dim=24, num_pairs=4, beta_ib=value)
        assert cfg.beta_ib == value, f"MDLIBPSConfig.beta_ib drift at {value}"


def test_score_aware_loss_uses_beta_ib_as_lagrangian_multiplier():
    """score_aware_loss.py uses beta_ib as IB Lagrangian multiplier per MDL theory."""
    src = (
        REPO_ROOT
        / "src"
        / "tac"
        / "substrates"
        / "c6_e4_mdl_ibps"
        / "score_aware_loss.py"
    ).read_text(encoding="utf-8")
    # Canonical pattern: ib_term = self.weights.beta_ib * kl_mean
    assert "self.weights.beta_ib" in src, (
        "score_aware_loss.py does not reference beta_ib Lagrangian multiplier"
    )
    assert "kl_mean" in src, (
        "score_aware_loss.py does not reference KL mean"
    )


def test_driver_threads_beta_ib_env_to_cli_flag():
    """Driver threads beta_ib to trainer args, provenance, and terminal claim notes."""
    driver_src = REMOTE_DRIVER.read_text(encoding="utf-8")
    assert "$C6_E4_MDL_IBPS_BETA_IB" in driver_src, (
        "Driver does not thread $C6_E4_MDL_IBPS_BETA_IB env var"
    )
    assert "--beta-ib" in driver_src, (
        "Driver does not pass --beta-ib CLI flag to trainer"
    )
    assert '"beta_ib": "$C6_E4_MDL_IBPS_BETA_IB"' in driver_src, (
        "Driver provenance does not record beta_ib"
    )
    assert "beta_ib=$C6_E4_MDL_IBPS_BETA_IB" in driver_src, (
        "Driver terminal claim notes do not record beta_ib"
    )


# ---------------------------------------------------------------------------
# Recipe existence + parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_file_exists(beta_ib_token: str):
    """Each sweep recipe YAML exists at canonical path."""
    p = _recipe_path(beta_ib_token)
    assert p.is_file(), f"recipe missing: {p}"


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_yaml_parses(beta_ib_token: str):
    """Each sweep recipe parses as YAML + has canonical top-level fields."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"recipe {p} did not parse as dict"
    assert data["schema_version"] == 1
    assert (
        data["name"]
        == f"substrate_c6_e4_mdl_ibps_beta_ib_{beta_ib_token}_modal_t4_smoke_dispatch"
    )
    assert data["lane_id"] == UMBRELLA_LANE_ID


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_env_overrides_threads_beta_ib(beta_ib_token: str):
    """env_overrides.C6_E4_MDL_IBPS_BETA_IB matches recipe name's β_ib value."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = data.get("env_overrides", {})
    actual = env.get("C6_E4_MDL_IBPS_BETA_IB")
    expected = BETA_IB_NUMERIC[beta_ib_token]
    assert float(actual) == expected, (
        f"recipe {p.name}: env C6_E4_MDL_IBPS_BETA_IB={actual!r} "
        f"does not match expected {expected!r}"
    )


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_env_overrides_keeps_baseline_latent_dim(beta_ib_token: str):
    """β_ib sweep does NOT modify latent_dim; latent_dim=24 baseline preserved."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = data.get("env_overrides", {})
    actual_ld = env.get("C6_E4_MDL_IBPS_LATENT_DIM")
    assert str(actual_ld) == str(BASELINE_LATENT_DIM), (
        f"recipe {p.name}: env C6_E4_MDL_IBPS_LATENT_DIM={actual_ld!r} "
        f"must be baseline {BASELINE_LATENT_DIM} (β_ib sweep does not vary latent_dim)"
    )


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_env_overrides_thread_sweep_lane_and_recipe(beta_ib_token: str):
    """Sweep recipe env must override the shared baseline C6 driver identity."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = data.get("env_overrides", {})
    assert env.get("C6_E4_MDL_IBPS_LANE_ID") == UMBRELLA_LANE_ID
    assert env.get("C6_E4_MDL_IBPS_RECIPE_PATH") == (
        f".omx/operator_authorize_recipes/"
        f"substrate_c6_e4_mdl_ibps_beta_ib_{beta_ib_token}_modal_t4_smoke_dispatch.yaml"
    )
    assert env.get("TAG") == f"substrate_c6_e4_mdl_ibps_beta_ib_{beta_ib_token}"


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_is_non_dispatchable_pending_operator_signoff(beta_ib_token: str):
    """Recipe is research_only=true + dispatch_enabled=false per Catalog #240."""
    import yaml

    p = _recipe_path(beta_ib_token)
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


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_predicted_band_validation_status_pending(beta_ib_token: str):
    """Per Catalog #324: predicted_band_validation_status must be pending_post_training."""
    import yaml

    p = _recipe_path(beta_ib_token)
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


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_passes_catalog_324_validator(beta_ib_token: str):
    """Each recipe passes the canonical Catalog #324 validator."""
    from tac.optimization.tier_c_density_post_training_validator import (
        validate_recipe_predicted_band,
    )

    p = _recipe_path(beta_ib_token)
    verdict = validate_recipe_predicted_band(p)
    assert verdict.is_valid, (
        f"recipe {p.name}: Catalog #324 validator rejected: "
        f"status={verdict.validation_status} blockers={verdict.blockers}"
    )


# ---------------------------------------------------------------------------
# Catalog #270 dispatch optimization protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_passes_catalog_270_dispatch_protocol(beta_ib_token: str):
    """Each recipe passes the Catalog #270 umbrella dispatch protocol (5/5+8/8+5/5)."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from canonical_dispatch_optimization_protocol import (
        verify_dispatch_protocol_complete,
    )

    trainer = REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    recipe_name = (
        f"substrate_c6_e4_mdl_ibps_beta_ib_{beta_ib_token}_modal_t4_smoke_dispatch"
    )
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


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_declares_distinguishing_feature_fields(beta_ib_token: str):
    """Per Catalog #220 + #272: distinguishing_feature_* fields declared."""
    import yaml

    p = _recipe_path(beta_ib_token)
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
    # The function MUST exist in the substrate package (same as baseline + sister latent_dim sweep)
    assert data["inflate_consumer_function"] == (
        "tac.substrates.c6_e4_mdl_ibps.inflate.inflate_one_video"
    ), f"recipe {p.name}: inflate_consumer_function should point to canonical inflate"
    # Distinguishing feature name should encode the β_ib unwind
    name = data["distinguishing_feature_name"]
    assert beta_ib_token in name, (
        f"recipe {p.name}: distinguishing_feature_name={name!r} should include β_ib token {beta_ib_token}"
    )
    assert "lagrangian" in name.lower(), (
        f"recipe {p.name}: distinguishing_feature_name={name!r} should reference 'lagrangian' (the unwind hypothesis)"
    )


# ---------------------------------------------------------------------------
# Catalog #309 horizon_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_declares_horizon_class_frontier_pursuit(beta_ib_token: str):
    """Per Catalog #309: horizon_class declared as frontier_pursuit."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data.get("horizon_class") == "frontier_pursuit", (
        f"recipe {p.name}: horizon_class must be 'frontier_pursuit' per Catalog #309 "
        f"(IBPS class shift; β-sweep does not change substrate class)"
    )


# ---------------------------------------------------------------------------
# Sister regression: baseline C6 + sister latent_dim BUILD unaffected
# ---------------------------------------------------------------------------


def test_baseline_c6_recipe_still_passes_catalog_270():
    """Sister regression: baseline C6 dispatch protocol unchanged."""
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


def test_sister_latent_dim_sweep_recipes_still_pass_catalog_270():
    """Sister regression: sister #846 latent_dim sweep recipes still pass."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from canonical_dispatch_optimization_protocol import (
        verify_dispatch_protocol_complete,
    )

    trainer = REPO_ROOT / "experiments" / "train_substrate_c6_e4_mdl_ibps.py"
    for ld in (48, 96, 192):
        recipe_name = f"substrate_c6_e4_mdl_ibps_latent{ld}_modal_t4_smoke_dispatch"
        verdict = verify_dispatch_protocol_complete(
            trainer=trainer,
            recipe=recipe_name,
        )
        assert verdict.overall_pass, (
            f"SISTER latent_{ld} dispatch protocol regression: {verdict.blockers}"
        )


# ---------------------------------------------------------------------------
# Recipe lane_id consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_lane_id_canonical(beta_ib_token: str):
    """All sweep recipes share canonical lane_id (umbrella sweep lane)."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data.get("lane_id") == UMBRELLA_LANE_ID, (
        f"recipe {p.name}: lane_id drift: {data.get('lane_id')!r}"
    )


def test_recipe_lane_id_differs_from_sister_latent_dim_sweep():
    """β_ib sweep umbrella lane is DISTINCT from sister #846 latent_dim sweep umbrella."""
    import yaml

    sister_ld_path = (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / "substrate_c6_e4_mdl_ibps_latent48_modal_t4_smoke_dispatch.yaml"
    )
    sister_data = yaml.safe_load(sister_ld_path.read_text(encoding="utf-8"))
    sister_lane_id = sister_data.get("lane_id")
    assert sister_lane_id != UMBRELLA_LANE_ID, (
        f"β_ib sweep lane_id must DIFFER from sister latent_dim sweep lane_id "
        f"(both shared: {sister_lane_id!r}). Separate umbrella lanes enable parallel "
        f"dispatch tracking per Catalog #126 lane pre-registration."
    )


# ---------------------------------------------------------------------------
# Cost band ladder discipline (sequential dispatch order)
# ---------------------------------------------------------------------------


def test_recipe_cost_band_within_envelope():
    """Per-recipe cost should fit $5 envelope per #848 Contrarian binding revision."""
    import yaml

    costs = []
    for ib in BETA_IB_SWEEP_VALUES:
        data = yaml.safe_load(_recipe_path(ib).read_text(encoding="utf-8"))
        cost = data.get("cost_band", {}).get("predicted_cost_usd")
        assert cost is not None, f"beta_ib_{ib} missing cost_band.predicted_cost_usd"
        assert cost <= 1.00, f"beta_ib_{ib} cost ${cost} exceeds $1.00 per smoke ceiling"
        costs.append(cost)
    total = sum(costs)
    assert total <= 5.0, (
        f"Total β_ib sweep envelope ${total} exceeds $5.00 cap per #848 Contrarian "
        f"binding revision (parallel paths (a)+(b) MUST fit $5 envelope)"
    )


@pytest.mark.parametrize("beta_ib_token", BETA_IB_SWEEP_VALUES)
def test_recipe_canary_dependency_chain_baseline(beta_ib_token: str):
    """Every β_ib recipe depends on baseline C6 (which is the empirically falsified canary)."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    canary = data.get("canary_dependency")
    assert canary == "c6_e4_mdl_ibps", (
        f"recipe {p.name}: canary_dependency must be baseline 'c6_e4_mdl_ibps' "
        f"(the empirically falsified canary per #836); got {canary!r}"
    )


# ---------------------------------------------------------------------------
# Remote-driver dispatch contract (sister regression — driver unchanged)
# ---------------------------------------------------------------------------


def test_c6_remote_driver_bash_syntax_clean() -> None:
    """Sister regression: driver script bash syntax unchanged."""
    result = subprocess.run(
        ["bash", "-n", str(REMOTE_DRIVER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Distinguishing-feature naming discipline per β_ib value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta_ib_token,expected_substring",
    [
        ("0p0001", "0p0001"),
        ("0p001", "0p001"),
        ("0p01", "0p01"),
        ("0p1", "0p1"),
        ("1p0", "1p0"),
    ],
)
def test_recipe_distinguishing_feature_name_encodes_beta_ib_value(
    beta_ib_token: str, expected_substring: str
) -> None:
    """distinguishing_feature_name must encode β_ib value (each variant must be distinct)."""
    import yaml

    p = _recipe_path(beta_ib_token)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    name = data.get("distinguishing_feature_name", "")
    assert expected_substring in name, (
        f"recipe {p.name}: distinguishing_feature_name={name!r} must encode "
        f"β_ib value {expected_substring}"
    )


# ---------------------------------------------------------------------------
# Lane pre-registration regression (Catalog #126)
# ---------------------------------------------------------------------------


def test_umbrella_lane_pre_registered_in_lane_registry():
    """Per Catalog #126: umbrella lane MUST be pre-registered before work starts."""
    import json

    registry_path = REPO_ROOT / ".omx" / "state" / "lane_registry.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    lane_ids = [lane.get("id") or lane.get("lane_id") for lane in data.get("lanes", [])]
    assert UMBRELLA_LANE_ID in lane_ids, (
        f"Umbrella lane {UMBRELLA_LANE_ID} not pre-registered in lane_registry.json"
    )


# ---------------------------------------------------------------------------
# 2D ablation surface composition with sister #846
# ---------------------------------------------------------------------------


def test_combined_envelope_parallel_paths_a_plus_b_fits_5_dollar_cap():
    """Per #848 Contrarian binding revision: parallel paths (a)+(b) MUST fit $5 cap."""
    import yaml

    # Path (a) β_ib sweep total
    a_total = 0.0
    for ib in BETA_IB_SWEEP_VALUES:
        data = yaml.safe_load(_recipe_path(ib).read_text(encoding="utf-8"))
        a_total += data["cost_band"]["predicted_cost_usd"]

    # Path (b) latent_dim sweep total (sister #846)
    b_total = 0.0
    for ld in (48, 96, 192):
        p = (
            REPO_ROOT
            / ".omx"
            / "operator_authorize_recipes"
            / f"substrate_c6_e4_mdl_ibps_latent{ld}_modal_t4_smoke_dispatch.yaml"
        )
        if not p.is_file():
            pytest.skip(f"Sister latent_{ld} recipe not present at {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        b_total += data["cost_band"]["predicted_cost_usd"]

    combined = a_total + b_total
    assert combined <= 5.0, (
        f"Combined parallel envelope (path a β_ib ${a_total} + path b latent_dim ${b_total}) "
        f"= ${combined} exceeds $5.00 cap per #848 Contrarian binding revision"
    )
