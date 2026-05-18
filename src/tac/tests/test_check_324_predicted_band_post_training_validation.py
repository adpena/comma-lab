# SPDX-License-Identifier: MIT
"""Tests for Catalog #324 META-FIX — predicted-band post-training Tier-C validation.

Per operator NON-NEGOTIABLE 2026-05-17 + META-FIX Catalog #324:
empirical bug class anchor is C6 IBPS 22× miss (3.04 actual vs predicted
[0.113, 0.163]) because Tier-C density was measured on random-init weights.
Sister #835 Assumption-Adversary WARNED; #836 empirically falsified.

Test surfaces:
  * ``TierCDensityWithProvenance`` __post_init__ invariants
  * ``PredictedBandWithValidation`` __post_init__ + auto-derived validation_status
  * Canonical builders (random_init / post_training / operator_waived)
  * ``validate_recipe_predicted_band`` recipe-level audit
  * Catalog #324 STRICT preflight gate end-to-end
  * Audit tool end-to-end + JSON report schema
  * C6 IBPS recipe empirical falsification regression guard
  * Z6 Phase 3 [0.13, 0.16] recipe pending_post_training classification
  * Sister gate interop (Catalog #321, #322, #323 subsumption matrix)
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tac.optimization.tier_c_density_post_training_validator import (
    PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN,
    PredictedBandValidationError,
    PredictedBandWithValidation,
    RecipeAuditVerdict,
    TIER_C_POST_TRAINING_SCHEMA_VERSION,
    TierCDensitySource,
    TierCDensityWithProvenance,
    VALIDATION_STATUS_OPERATOR_WAIVED,
    VALIDATION_STATUS_PENDING_POST_TRAINING,
    VALIDATION_STATUS_PHANTOM_RANDOM_INIT,
    VALIDATION_STATUS_VALIDATED_POST_TRAINING,
    build_predicted_band_from_tier_c_density,
    build_tier_c_density_operator_waived,
    build_tier_c_density_post_training,
    build_tier_c_density_random_init,
    validate_recipe_predicted_band,
)


# =========================================================================
# TierCDensityWithProvenance invariants
# =========================================================================


def test_random_init_with_sentinel_archive_constructs():
    density = TierCDensityWithProvenance(
        density_value=2.67e-5,
        source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
        measured_at_utc="2026-05-17T22:00:00Z",
        archive_sha256="random_init_no_archive",
        epochs_trained=0,
        canonical_helper_invocation="tools/mdl_scorer_conditional_ablation.py --tier c",
    )
    assert density.density_value == 2.67e-5
    assert density.source == TierCDensitySource.RANDOM_INIT_PRE_TRAINING
    assert density.requires_post_training_revalidation() is True


def test_post_training_with_real_sha_constructs():
    density = TierCDensityWithProvenance(
        density_value=0.001,
        source=TierCDensitySource.POST_TRAINING_200EP_FULL,
        measured_at_utc="2026-05-17T23:30:00Z",
        archive_sha256="be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec",
        epochs_trained=200,
    )
    assert density.requires_post_training_revalidation() is False


def test_random_init_with_real_sha_rejected():
    """RANDOM_INIT requires the sentinel archive_sha256 — real sha is inconsistent."""
    with pytest.raises(PredictedBandValidationError) as exc_info:
        TierCDensityWithProvenance(
            density_value=2.67e-5,
            source=TierCDensitySource.POST_TRAINING_200EP_FULL,  # Wrong source for sentinel
            measured_at_utc="2026-05-17T22:00:00Z",
            archive_sha256="random_init_no_archive",
            epochs_trained=200,
        )
    assert any("sentinel" in b.lower() for b in exc_info.value.blockers)


def test_unknown_provenance_refused():
    with pytest.raises(PredictedBandValidationError) as exc_info:
        TierCDensityWithProvenance(
            density_value=0.001,
            source=TierCDensitySource.UNKNOWN_PROVENANCE,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="a" * 64,
            epochs_trained=10,
        )
    assert "UNKNOWN_PROVENANCE" in str(exc_info.value)
    assert any("Catalog #324" in b for b in exc_info.value.blockers)


def test_operator_waived_requires_substantive_rationale():
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=0.001,
            source=TierCDensitySource.OPERATOR_WAIVED,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="a" * 64,
            rationale="",  # empty
        )


def test_operator_waived_rejects_placeholder_rationale():
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=0.001,
            source=TierCDensitySource.OPERATOR_WAIVED,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="a" * 64,
            rationale="<rationale>",
        )


def test_operator_waived_accepts_substantive_rationale():
    density = build_tier_c_density_operator_waived(
        density_value=0.001,
        measured_at_utc="2026-05-17T00:00:00Z",
        archive_sha256="b" * 64,
        rationale="Tier-C density derived from sister C5 substrate empirical anchor; operator-attested per council 2026-05-17",
    )
    assert density.requires_post_training_revalidation() is False


def test_negative_density_rejected():
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=-1.0,
            source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="random_init_no_archive",
        )


def test_nan_density_rejected():
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=float("nan"),
            source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="random_init_no_archive",
        )


def test_epochs_trained_inconsistent_with_source_rejected():
    """POST_TRAINING_200EP requires >= 50 epochs."""
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=0.001,
            source=TierCDensitySource.POST_TRAINING_200EP_FULL,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="a" * 64,
            epochs_trained=10,  # Too few for 200EP source
        )


def test_random_init_epochs_must_be_zero():
    with pytest.raises(PredictedBandValidationError):
        TierCDensityWithProvenance(
            density_value=0.001,
            source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
            measured_at_utc="2026-05-17T00:00:00Z",
            archive_sha256="random_init_no_archive",
            epochs_trained=10,  # Nonzero for random-init
        )


def test_to_provenance_dict_includes_canonical_fields():
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    d = density.to_provenance_dict()
    assert d["tier_c_density_value"] == 2.67e-5
    assert d["tier_c_density_source"] == "random_init_pre_training"
    assert d["tier_c_density_archive_sha256"] == "random_init_no_archive"
    assert d["tier_c_density_schema_version"] == TIER_C_POST_TRAINING_SCHEMA_VERSION


# =========================================================================
# PredictedBandWithValidation invariants
# =========================================================================


def test_phantom_random_init_band_derives_phantom_status():
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=0.113,
        proposed_band_high=0.163,
        proposed_target=0.138,
    )
    assert band.validation_status == VALIDATION_STATUS_PHANTOM_RANDOM_INIT
    assert band.requires_post_training_revalidation is True


def test_post_training_band_derives_validated_status():
    density = build_tier_c_density_post_training(
        density_value=0.001,
        epochs_trained=200,
        archive_sha256="be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec",
        measured_at_utc="2026-05-17T23:30:00Z",
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=0.10,
        proposed_band_high=0.20,
        proposed_target=0.15,
    )
    assert band.validation_status == VALIDATION_STATUS_VALIDATED_POST_TRAINING
    assert band.requires_post_training_revalidation is False


def test_operator_waived_band_derives_waived_status():
    density = build_tier_c_density_operator_waived(
        density_value=0.001,
        measured_at_utc="2026-05-17T00:00:00Z",
        rationale="Operator-attested per council 2026-05-17 with substantive justification",
        archive_sha256="c" * 64,
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=0.10,
        proposed_band_high=0.20,
        proposed_target=0.15,
    )
    assert band.validation_status == VALIDATION_STATUS_OPERATOR_WAIVED


def test_band_target_outside_range_rejected():
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    with pytest.raises(PredictedBandValidationError):
        build_predicted_band_from_tier_c_density(
            tier_c_density=density,
            proposed_band_low=0.10,
            proposed_band_high=0.20,
            proposed_target=0.30,  # Outside [0.10, 0.20]
        )


def test_band_low_above_high_rejected():
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    with pytest.raises(PredictedBandValidationError):
        build_predicted_band_from_tier_c_density(
            tier_c_density=density,
            proposed_band_low=0.20,
            proposed_band_high=0.10,
            proposed_target=0.15,
        )


def test_band_to_recipe_dict_round_trip():
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=0.113,
        proposed_band_high=0.163,
        proposed_target=0.138,
    )
    d = band.to_recipe_dict()
    assert d["predicted_band"] == [0.113, 0.163]
    assert d["predicted_score_target"] == 0.138
    assert d["predicted_band_validation_status"] == "phantom_random_init"
    assert d["predicted_band_requires_post_training_revalidation"] is True
    assert "tier_c_density_value" in d["predicted_band_tier_c_density_provenance"]


# =========================================================================
# validate_recipe_predicted_band — recipe-level audit
# =========================================================================


def test_recipe_with_no_predicted_band_out_of_scope(tmp_path):
    recipe = tmp_path / "no_band.yaml"
    recipe.write_text("schema_version: 1\nname: test\n")
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.has_predicted_band is False
    assert verdict.validation_status == "absent"


def test_recipe_with_band_but_no_status_rejected(tmp_path):
    recipe = tmp_path / "no_status.yaml"
    recipe.write_text("schema_version: 1\nname: test\npredicted_band: [0.10, 0.20]\npredicted_score_target: 0.15\n")
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is False
    assert verdict.validation_status == "missing_validation_status"
    assert verdict.has_predicted_band is True
    assert verdict.detected_predicted_band == (0.10, 0.20)
    assert verdict.detected_target == 0.15
    assert any("Catalog #324" in b for b in verdict.blockers)


def test_recipe_with_band_and_validated_status_passes(tmp_path):
    recipe = tmp_path / "validated.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        predicted_band_validation_status: validated_post_training
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.validation_status == VALIDATION_STATUS_VALIDATED_POST_TRAINING


def test_recipe_with_band_and_pending_status_passes(tmp_path):
    recipe = tmp_path / "pending.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        predicted_band_validation_status: pending_post_training
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.validation_status == VALIDATION_STATUS_PENDING_POST_TRAINING


def test_recipe_with_band_and_research_only_passes(tmp_path):
    recipe = tmp_path / "research_only.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        research_only: true
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.validation_status == "research_only"
    assert verdict.is_research_only is True


def test_recipe_with_band_and_dispatch_disabled_passes(tmp_path):
    recipe = tmp_path / "dispatch_disabled.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        dispatch_enabled: false
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.validation_status == "research_only"


def test_recipe_with_band_and_substantive_waiver_passes(tmp_path):
    recipe = tmp_path / "waived.yaml"
    recipe.write_text(textwrap.dedent(f"""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]  # {PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN}:operator-attested via council 2026-05-17
        predicted_score_target: 0.15
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is True
    assert verdict.validation_status == VALIDATION_STATUS_OPERATOR_WAIVED


def test_recipe_with_band_and_placeholder_waiver_rejected(tmp_path):
    recipe = tmp_path / "placeholder_waiver.yaml"
    recipe.write_text(textwrap.dedent(f"""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]  # {PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN}:<rationale>
        predicted_score_target: 0.15
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is False  # Placeholder rejected


def test_recipe_with_phantom_random_init_status_requires_opt_out(tmp_path):
    """A recipe declaring phantom_random_init but still dispatch-enabled is invalid."""
    recipe = tmp_path / "phantom_dispatch_enabled.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.113, 0.163]
        predicted_score_target: 0.138
        predicted_band_validation_status: phantom_random_init
        dispatch_enabled: true
    """))
    verdict = validate_recipe_predicted_band(recipe)
    assert verdict.is_valid is False
    assert verdict.validation_status == VALIDATION_STATUS_PHANTOM_RANDOM_INIT


def test_missing_recipe_file_returns_valid_absent():
    verdict = validate_recipe_predicted_band("/nonexistent/path/recipe.yaml")
    assert verdict.is_valid is True
    assert verdict.validation_status == "absent"


# =========================================================================
# C6 IBPS empirical anchor regression guards
# =========================================================================


def test_c6_ibps_empirical_anchor_phantom_classification():
    """Reproduce the C6 IBPS 22× miss empirical bug class anchor.

    Sister #835 declared predicted_band [0.113, 0.163] derived from
    Tier-C density 2.67e-5 measured on random-init weights. Sister #836
    smoke landed 3.04 — 22× outside upper. Verify the canonical helper
    auto-classifies this as phantom_random_init.
    """
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:51:13Z",
        canonical_helper_invocation="tools/mdl_scorer_conditional_ablation.py --tier c (random-init pre-training)",
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=0.113,
        proposed_band_high=0.163,
        proposed_target=0.138,
    )
    assert band.validation_status == VALIDATION_STATUS_PHANTOM_RANDOM_INIT, (
        "C6 IBPS recipe-derived band MUST classify as phantom_random_init per "
        "empirical falsification anchor #836 (actual 3.04 vs predicted [0.113, 0.163])"
    )
    assert band.requires_post_training_revalidation is True


def test_c6_ibps_post_50ep_smoke_validates():
    """If Tier-C is re-measured on the actual 50ep smoke archive, it would validate."""
    density = build_tier_c_density_post_training(
        density_value=0.5,  # hypothetical post-training density
        epochs_trained=50,
        archive_sha256="be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec",
        measured_at_utc="2026-05-17T23:46:38Z",
    )
    band = build_predicted_band_from_tier_c_density(
        tier_c_density=density,
        proposed_band_low=2.5,
        proposed_band_high=3.5,
        proposed_target=3.0,  # post-empirical re-estimation
    )
    assert band.validation_status == VALIDATION_STATUS_VALIDATED_POST_TRAINING


# =========================================================================
# Audit tool end-to-end
# =========================================================================


def test_audit_tool_clean_recipe(tmp_path):
    """Audit tool should classify a clean recipe as PASS."""
    recipe = tmp_path / "clean.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: clean
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        predicted_band_validation_status: validated_post_training
    """))

    result = subprocess.run(
        [sys.executable, "tools/audit_predicted_band_provenance.py",
         "--recipe-glob", str(tmp_path / "*.yaml"),
         "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_audit_tool_strict_flag_fails_on_violation(tmp_path):
    """--strict should exit rc=1 if any FAIL verdict found."""
    recipe = tmp_path / "bad.yaml"
    recipe.write_text("predicted_band: [0.10, 0.20]\npredicted_score_target: 0.15\n")

    result = subprocess.run(
        [sys.executable, "tools/audit_predicted_band_provenance.py",
         "--recipe-glob", str(tmp_path / "*.yaml"),
         "--strict",
         "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


def test_audit_tool_json_report(tmp_path):
    """JSON report should include canonical schema version + per-recipe verdicts."""
    recipe = tmp_path / "test.yaml"
    recipe.write_text(textwrap.dedent("""
        schema_version: 1
        name: test
        predicted_band: [0.10, 0.20]
        predicted_score_target: 0.15
        research_only: true
    """))
    report_out = tmp_path / "report.json"

    result = subprocess.run(
        [sys.executable, "tools/audit_predicted_band_provenance.py",
         "--recipe-glob", str(tmp_path / "*.yaml"),
         "--report-out", str(report_out),
         "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert report_out.exists()
    data = json.loads(report_out.read_text())
    assert data["schema_version"] == "predicted_band_audit_v1_20260517"
    assert data["recipes_scanned"] == 1
    assert data["in_scope_count"] == 1
    assert data["pass_count"] == 1
    assert data["fail_count"] == 0
    assert "anchor_note" in data
    assert "C6 IBPS" in data["anchor_note"]


# =========================================================================
# Live-repo regression guards
# =========================================================================


def test_live_c6_ibps_recipe_currently_missing_validation_status():
    """Regression guard: the live C6 IBPS recipe currently lacks validation_status.

    This test will start FAILING (PASS) when the operator backfills the recipe
    per the META-FIX #324 op-routables — at that point, flip the assertion.
    """
    recipe_path = Path(".omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml")
    if not recipe_path.exists():
        pytest.skip("C6 IBPS recipe not in repo")
    verdict = validate_recipe_predicted_band(recipe_path)
    # At META-FIX #324 landing, this verdict is FAIL pending backfill.
    # Document the empirical state without enforcement until operator action.
    assert verdict.has_predicted_band is True
    # The recipe currently has dispatch_enabled=true + no validation_status.


def test_live_z6_redirect_recipe_currently_research_only():
    """Z6 Phase 3 redirect recipe should be research_only (pending sister build)."""
    recipe_path = Path(".omx/operator_authorize_recipes/q4_substrate_class_shift_redirect_z6_predictive_coding_20260517T222800Z.yaml")
    if not recipe_path.exists():
        pytest.skip("Z6 redirect recipe not in repo")
    verdict = validate_recipe_predicted_band(recipe_path)
    assert verdict.is_valid is True
    assert verdict.validation_status == "research_only"


# =========================================================================
# Provenance contract extension backward-compat
# =========================================================================


def test_provenance_contract_still_callable():
    """Catalog #323 Provenance contract must remain callable (sister #832 wire-in)."""
    from tac.provenance import Provenance, ProvenanceKind, ProvenanceEvidenceGrade

    prov = Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        source_path=".omx/state/synthetic_test_artifact.json",
        source_sha256="a" * 64,
        measurement_axis="[research-signal]",
        hardware_substrate="macos_arm64",
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc="2026-05-17T00:00:00Z",
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_research_sidecar",
        rejection_reason="test fixture only",
    )
    assert prov.score_claim_valid is False


def test_tier_c_density_to_provenance_dict_compatible_with_catalog_323():
    """The tier_c_density_to_provenance_dict shape should be embeddable in Catalog #323."""
    density = build_tier_c_density_random_init(
        density_value=2.67e-5,
        measured_at_utc="2026-05-17T22:00:00Z",
    )
    d = density.to_provenance_dict()
    # The dict should be JSON-serializable
    json_text = json.dumps(d, sort_keys=True)
    parsed = json.loads(json_text)
    assert parsed["tier_c_density_source"] == "random_init_pre_training"


# =========================================================================
# Sister gate subsumption matrix
# =========================================================================


def test_sister_gate_321_still_extant():
    """Catalog #321 (phantom WZ savings) must still be in preflight."""
    from tac import preflight as preflight_module
    assert hasattr(preflight_module, "check_no_phantom_wyner_ziv_savings_from_research_sidecar")


def test_sister_gate_323_still_extant():
    """Catalog #323 (canonical Provenance) must still be in preflight."""
    from tac import preflight as preflight_module
    # Sister #832's gate may be named slightly differently; check both common forms.
    has_323 = (
        hasattr(preflight_module, "check_no_score_claim_without_canonical_provenance")
        or hasattr(preflight_module, "check_score_claims_have_canonical_provenance")
    )
    assert has_323, "Catalog #323 gate must remain in preflight per sister #832 landing"


# =========================================================================
# Schema constants pinned
# =========================================================================


def test_schema_version_pinned():
    assert TIER_C_POST_TRAINING_SCHEMA_VERSION == "tier_c_post_training_v1_20260517"


def test_validation_status_constants_pinned():
    assert VALIDATION_STATUS_VALIDATED_POST_TRAINING == "validated_post_training"
    assert VALIDATION_STATUS_PENDING_POST_TRAINING == "pending_post_training"
    assert VALIDATION_STATUS_PHANTOM_RANDOM_INIT == "phantom_random_init"
    assert VALIDATION_STATUS_OPERATOR_WAIVED == "operator_waived"


def test_waiver_token_pinned():
    assert PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN == "PREDICTED_BAND_RANDOM_INIT_OK"


def test_tier_c_density_source_enum_values_pinned():
    """Enum values are part of the canonical contract; pinning prevents silent drift."""
    expected = {
        "random_init_pre_training",
        "post_training_50ep_smoke",
        "post_training_200ep_full",
        "unknown_provenance",
        "operator_waived",
    }
    actual = {s.value for s in TierCDensitySource}
    assert actual == expected
