# SPDX-License-Identifier: MIT
"""Tests for HPRC family-binding policy."""

from __future__ import annotations

from tac.substrates.hprc.archive import HprcSectionKind
from tac.substrates.hprc.lineage import (
    HPRC_FAMILY_BINDINGS,
    HPRC_OPTIMIZATION_LEVERS,
    HprcOptimizationLever,
    HprcRole,
    get_binding,
    hprc_campaign_manifest,
    primary_rate_collapse_candidates,
    residual_sidecar_candidates,
)


def test_hprc_bindings_are_unique_and_section_typed() -> None:
    ids = [binding.family_id for binding in HPRC_FAMILY_BINDINGS]
    assert len(ids) == len(set(ids))
    for binding in HPRC_FAMILY_BINDINGS:
        assert binding.roles
        assert binding.allowed_sections
        assert binding.promotion_gate
        assert binding.rate_axis_risk
        assert all(isinstance(section, HprcSectionKind) for section in binding.allowed_sections)


def test_z8_is_not_allowed_as_primary_payload_again() -> None:
    z8 = get_binding("z8_hpc_teacher_residual")

    assert HprcRole.TEACHER_ONLY in z8.roles
    assert HprcRole.BASE_RECEIVER not in z8.roles
    assert HprcSectionKind.DECODER_QW not in z8.allowed_sections
    assert HprcSectionKind.LATENTS_RC not in z8.allowed_sections
    assert "explicit" in z8.promotion_gate


def test_primary_and_residual_candidate_sets_are_separated() -> None:
    primaries = set(primary_rate_collapse_candidates())
    residuals = set(residual_sidecar_candidates())

    assert "pr95_hnerv_control" in primaries
    assert "rnerv_pact_nerv_base" in primaries
    assert "c3_cool_chic_overfit_codec" in primaries
    assert "z8_hpc_teacher_residual" not in primaries
    assert "z8_hpc_teacher_residual" in residuals
    assert "raft_motion_side_information" in residuals


def test_hprc_campaign_manifest_is_non_authoritative_and_rate_first() -> None:
    manifest = hprc_campaign_manifest()

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    rules = " ".join(manifest["rate_first_rules"])
    assert "not explicit per-pair fields" in rules
    assert "q=0.25" in rules
    assert "full SHA-256" in rules
    assert "valid semantic mutation" in rules
    assert "inflate runtime" in rules
    assert any(
        row["family_id"] == "clade_spade_semantic_conditioning"
        for row in manifest["bindings"]
    )
    levers = {row["lever"] for row in manifest["optimization_levers"]}
    assert HprcOptimizationLever.RECEIVER_WEIGHT_QUANTIZATION.value in levers
    assert HprcOptimizationLever.MOTION_COMPENSATED_SIDE_INFO.value in levers
    assert HprcOptimizationLever.FULL_VIDEO_BUNDLE_KKT_ALLOCATION.value in levers
    assert HprcOptimizationLever.INVENTED_RECEIVER_PARADIGM.value in levers


def test_hprc_optimizer_taxonomy_covers_core_byte_levers() -> None:
    levers = {row["lever"] for row in HPRC_OPTIMIZATION_LEVERS}

    required = {
        HprcOptimizationLever.RECEIVER_WEIGHT_QUANTIZATION.value,
        HprcOptimizationLever.LATENT_ENTROPY_CODING.value,
        HprcOptimizationLever.SCORER_WEIGHTED_ABLATION.value,
        HprcOptimizationLever.RESIDUAL_TOKEN_WATERFILL.value,
        HprcOptimizationLever.RANGE_ANS_ARITHMETIC_CODING.value,
        HprcOptimizationLever.NATIVE_RUST_ZIG_DECODE.value,
        HprcOptimizationLever.EXACT_REPLAY_ACCEPTANCE.value,
        HprcOptimizationLever.UNKNOWN_FUTURE_LEVER.value,
    }
    assert required.issubset(levers)
    assert all(row["gate"] for row in HPRC_OPTIMIZATION_LEVERS)
