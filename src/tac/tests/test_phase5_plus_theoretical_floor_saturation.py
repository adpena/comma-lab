"""Tests for Phase 5+ theoretical-floor saturation scaffold.

These tests cover the SKETCH-CONJECTURE-ONLY scaffold — they verify that the
scaffold refuses any GPU dispatch path and that all band claims are tagged
[conjecture] (NOT [predicted]).
"""
from __future__ import annotations

import pytest

from tac.phase5_plus import (
    PHASE5_PLUS_CONJECTURAL_BAND_TAG,
    PHASE5_PLUS_PROVENANCE,
    PHASE5_PLUS_VERSION,
    TheoreticalFloorSaturationConfig,
    TheoreticalFloorSaturationConjecture,
    phase5_plus_conjectural_band,
    phase5_plus_lagrangian_form,
)


def test_version_is_sketch_conjecture_only():
    assert "sketch" in PHASE5_PLUS_VERSION.lower()
    assert "conjecture-only" in PHASE5_PLUS_VERSION.lower()


def test_conjectural_band_tag_uses_conjecture_not_predicted():
    """CLAUDE.md ``forbidden_score_claims`` requires Phase 5+ to use
    [conjecture] not [predicted]. The distinction is structural.
    """
    assert PHASE5_PLUS_CONJECTURAL_BAND_TAG.startswith("[conjecture;")
    assert "[predicted;" not in PHASE5_PLUS_CONJECTURAL_BAND_TAG


def test_provenance_dispatch_readiness_blocks_all_dispatch():
    dr = PHASE5_PLUS_PROVENANCE["dispatch_readiness"]
    assert dr["DISPATCH_READY"] is False
    assert dr["REQUIRES_OPERATOR_APPROVAL"] is True
    assert dr["GATED_ON_PHASE3_ANCHOR"] is True
    assert dr["GPU_BUDGET_USD"] == 0.0
    assert dr["CATHEDRAL_AUTOPILOT_DISPATCH_QUEUE_INCLUDED"] is False


def test_provenance_lists_four_conjectural_mechanisms():
    mechs = PHASE5_PLUS_PROVENANCE["conjectural_mechanisms"]
    assert "score_aware_substrate_co_evolution" in mechs
    assert "closed_form_fisher_rao_geodesic" in mechs
    assert "cross_paradigm_composition_phase_1_plus_2_plus_3" in mechs
    assert "joint_source_coding_refinement_finite_block" in mechs


def test_config_defaults_match_coherence_council():
    cfg = TheoreticalFloorSaturationConfig()
    assert cfg.target_score_band_lower == 0.116
    assert cfg.target_score_band_upper == 0.131
    assert cfg.fisher_rao_geodesic_method == "closed_form"
    assert cfg.finite_block_correction_n_pairs == 600


def test_config_rejects_invalid_band():
    with pytest.raises(ValueError):
        TheoreticalFloorSaturationConfig(target_score_band_lower=0.5, target_score_band_upper=0.4)
    with pytest.raises(ValueError):
        TheoreticalFloorSaturationConfig(target_score_band_lower=-0.1)


def test_config_rejects_invalid_geodesic_method():
    with pytest.raises(ValueError, match="fisher_rao_geodesic_method"):
        TheoreticalFloorSaturationConfig(fisher_rao_geodesic_method="invalid")


def test_config_rejects_invalid_dual_init():
    with pytest.raises(ValueError):
        TheoreticalFloorSaturationConfig(substrate_evolution_dual_lambda_init=0)
    with pytest.raises(ValueError):
        TheoreticalFloorSaturationConfig(substrate_evolution_dual_rho_init=-1.0)


def test_config_rejects_zero_pairs_for_finite_block_correction():
    with pytest.raises(ValueError):
        TheoreticalFloorSaturationConfig(finite_block_correction_n_pairs=0)


def test_dispatch_raises_not_implemented():
    cfg = TheoreticalFloorSaturationConfig()
    conj = TheoreticalFloorSaturationConjecture(config=cfg)
    with pytest.raises(NotImplementedError, match="SKETCH-CONJECTURE-ONLY"):
        conj.dispatch()


def test_emit_conjecture_manifest_has_required_fields():
    cfg = TheoreticalFloorSaturationConfig()
    conj = TheoreticalFloorSaturationConjecture(config=cfg)
    manifest = conj.emit_conjecture_manifest()

    for required in (
        "phase",
        "lane_id",
        "config",
        "lagrangian_form",
        "conjectural_band",
        "council_memo_path",
        "dispatch_status",
        "dispatch_ready",
        "requires_operator_approval",
        "gpu_budget_usd",
        "cathedral_autopilot_dispatch_queue_included",
        "manifest_schema_version",
    ):
        assert required in manifest

    assert manifest["dispatch_ready"] is False
    assert manifest["gpu_budget_usd"] == 0.0
    assert manifest["cathedral_autopilot_dispatch_queue_included"] is False
    assert manifest["lane_id"] == "lane_phase5_theoretical_floor_saturation"


def test_lagrangian_form_invokes_phase3_plus_substrate_dual():
    form = phase5_plus_lagrangian_form()
    assert "L_phase3" in form["form"]
    assert "substrate" in form["form"]
    assert "Tishby 1999" in form["theorems_invoked"]
    assert "Boyd 2011" in form["theorems_invoked"]
    assert "Berger 1971" in form["theorems_invoked"]
    assert "Tao" in form["theorems_invoked"]  # Tao consult required


def test_conjectural_band_tag_is_conjecture():
    cfg = TheoreticalFloorSaturationConfig()
    band = phase5_plus_conjectural_band(cfg)
    assert band["tag"].startswith("[conjecture;")
    assert "[predicted;" not in band["tag"]
    assert band["lower_bound"] == 0.116
    assert band["upper_bound"] == 0.131
    assert band["median_conjecture"] == pytest.approx(0.1235, abs=1e-4)


def test_conjectural_band_documents_promotion_criteria():
    cfg = TheoreticalFloorSaturationConfig()
    band = phase5_plus_conjectural_band(cfg)
    assert "promotion_to_predicted_requires" in band
    assert "promotion_to_empirical_requires" in band
    # Phase 3 anchor is the gate for predicted
    assert any("Phase 3" in s for s in band["promotion_to_predicted_requires"])
    # Operator approval is the gate for empirical
    assert any("Operator approval" in s for s in band["promotion_to_empirical_requires"])
