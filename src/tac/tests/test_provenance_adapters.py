# SPDX-License-Identifier: MIT
"""Tests for tac.provenance.adapters — backward-compat shims."""

from __future__ import annotations

import pytest

from tac.provenance import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    contest_result_to_provenance,
    cost_band_anchor_to_provenance,
    council_record_to_provenance,
    deliverability_proof_to_provenance,
    master_gradient_plan_to_provenance,
    modal_call_id_ledger_event_to_provenance,
    substrate_composition_row_to_provenance,
    wyner_ziv_layer_result_to_provenance,
)

# -----------------------------------------------------------------------------
# contest_result_to_provenance
# -----------------------------------------------------------------------------


def test_contest_result_promotable_cuda():
    """Legacy ContestResult with archive_path + CUDA axis → PROMOTABLE."""
    legacy_dict = {
        "archive_sha256": "a" * 64,
        "archive_path": "submissions/a1/archive.zip",
        "archive_member_name": "0.bin",
        "axis": "contest-CUDA",
        "hardware_substrate": "t4",
        "captured_at_utc": "2026-05-17T22:00:00Z",
        "promotion_eligible": True,
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.artifact_kind == ProvenanceKind.CONTEST_ARCHIVE_MEMBER
    assert prov.evidence_grade == ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA
    assert prov.measurement_axis == "[contest-CUDA]"
    assert prov.hardware_substrate == "linux_x86_64_t4"
    assert prov.promotion_eligible


def test_contest_result_demotes_when_no_archive_path():
    """Without archive_path, even CUDA axis demotes per backward-compat."""
    legacy_dict = {
        "archive_sha256": "b" * 64,
        "axis": "contest-CUDA",
        "hardware_substrate": "t4",
    }
    prov = contest_result_to_provenance(legacy_dict)
    # Demoted because we can't satisfy CONTEST_ARCHIVE_MEMBER without zip_path
    assert prov.evidence_grade != ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA


def test_contest_result_advisory_macos():
    legacy_dict = {
        "archive_sha256": "c" * 64,
        "axis": "macOS-CPU advisory",
        "hardware_substrate": "darwin",
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY
    assert not prov.promotion_eligible


def test_contest_result_advisory_macos_mlx():
    legacy_dict = {
        "archive_sha256": "c" * 64,
        "axis": "macOS-MLX research-signal",
        "hardware_substrate": "mlx",
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL
    assert prov.measurement_axis == "[macOS-MLX research-signal]"
    assert prov.hardware_substrate == "macos_arm64_mlx"
    assert not prov.promotion_eligible


def test_contest_result_mps_proxy():
    legacy_dict = {
        "archive_sha256": "d" * 64,
        "axis": "MPS-PROXY",
        "hardware_substrate": "darwin",
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.evidence_grade == ProvenanceEvidenceGrade.MPS_PROXY
    assert not prov.promotion_eligible


def test_contest_result_missing_fields_returns_research():
    legacy_dict = {}  # totally empty
    prov = contest_result_to_provenance(legacy_dict)
    # Adapter should never crash; returns research-only at minimum
    assert prov.artifact_kind in (ProvenanceKind.DERIVED_AGGREGATE, ProvenanceKind.RESEARCH_SIDECAR)
    assert not prov.promotion_eligible


def test_contest_result_invalid_sha_replaced_with_placeholder():
    legacy_dict = {
        "archive_sha256": "not-a-real-sha",
        "axis": "contest-CUDA",
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.source_sha256 == "0" * 64


def test_contest_result_uppercase_sha_lowercased():
    legacy_dict = {
        "archive_sha256": "A" * 64,
        "axis": "research-signal",
    }
    prov = contest_result_to_provenance(legacy_dict)
    assert prov.source_sha256 == "a" * 64


def test_contest_result_object_attribute_access():
    """Adapter handles object with attribute access (dataclass-like)."""

    class FakeResult:
        archive_sha256 = "e" * 64
        axis = "contest-CPU"
        hardware_substrate = "modal_cpu"

    prov = contest_result_to_provenance(FakeResult())
    assert prov.measurement_axis == "[contest-CPU]"


# -----------------------------------------------------------------------------
# cost_band_anchor_to_provenance
# -----------------------------------------------------------------------------


def test_cost_band_anchor_without_nested_result():
    anchor = {"anchor_path": ".omx/state/cost_band/anchor_1.json"}
    prov = cost_band_anchor_to_provenance(anchor)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR


def test_cost_band_anchor_with_nested_contest_result():
    anchor = {
        "contest_result": {
            "archive_sha256": "f" * 64,
            "axis": "contest-CUDA",
            "hardware_substrate": "a100",
            "archive_path": "submissions/x/archive.zip",
        }
    }
    prov = cost_band_anchor_to_provenance(anchor)
    assert prov.artifact_kind == ProvenanceKind.CONTEST_ARCHIVE_MEMBER


# -----------------------------------------------------------------------------
# council_record_to_provenance
# -----------------------------------------------------------------------------


def test_council_record_basic():
    record = {
        "deliberation_id": "feedback_council_x_20260517",
        "topic": "x-topic",
    }
    prov = council_record_to_provenance(record)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR
    assert "feedback_council_x" in prov.source_path
    assert "x-topic" in prov.rejection_reason


# -----------------------------------------------------------------------------
# substrate_composition_row_to_provenance (Catalog #823 detection)
# -----------------------------------------------------------------------------


def test_composition_row_byte_identity_detected():
    """Sister-contended REDO_PIVOT_FIX_ALL surface — verify byte-identity flag."""
    row = {
        "pair_key": "lane_g_v3_renderer__x__siren_renderer",
        "candidate_a_sha256": "08f12d72" + "2" * 56,
        "candidate_b_sha256": "08f12d72" + "2" * 56,
        "verdict": "super_additive_FALSE_SIGNAL_BYTE_IDENTITY_ARTIFACT",
    }
    prov = substrate_composition_row_to_provenance(row)
    assert prov.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT


def test_composition_row_phantom_verdict_caught():
    row = {
        "pair_key": "test_x_other",
        "verdict": "phantom_score_artifact",
    }
    prov = substrate_composition_row_to_provenance(row)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR
    assert "phantom" in prov.rejection_reason.lower()


def test_composition_row_clean_returns_research_default():
    row = {"pair_key": "a_x_b", "verdict": "clean"}
    prov = substrate_composition_row_to_provenance(row)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR


# -----------------------------------------------------------------------------
# deliverability_proof_to_provenance
# -----------------------------------------------------------------------------


def test_deliverability_proof_basic():
    proof = {"archive_sha256": "7" * 64}
    prov = deliverability_proof_to_provenance(proof)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR


def test_deliverability_proof_tier_4_forbidden_demotes():
    proof = {"archive_sha256": "6" * 64, "tier": "TIER_4_FORBIDDEN"}
    prov = deliverability_proof_to_provenance(proof)
    assert "forbidden" in prov.rejection_reason.lower()


# -----------------------------------------------------------------------------
# wyner_ziv_layer_result_to_provenance
# -----------------------------------------------------------------------------


def test_wyner_ziv_layer_result_basic():
    result = {"layer_name": "test_layer"}
    prov = wyner_ziv_layer_result_to_provenance(result)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR
    assert "test_layer" in prov.source_path


# -----------------------------------------------------------------------------
# master_gradient_plan_to_provenance
# -----------------------------------------------------------------------------


def test_master_gradient_plan_with_archive():
    plan = {
        "plan_id": "fec6_plan",
        "archive_path": "submissions/fec6/archive.zip",
        "archive_sha256": "9" * 64,  # hex chars only
    }
    prov = master_gradient_plan_to_provenance(plan)
    assert prov.artifact_kind == ProvenanceKind.DERIVED_AGGREGATE
    # Adapter is conservative — non-promotable until canonical embed
    assert not prov.promotion_eligible


def test_master_gradient_plan_minimal():
    plan = {"plan_id": "x"}
    prov = master_gradient_plan_to_provenance(plan)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR


# -----------------------------------------------------------------------------
# modal_call_id_ledger_event_to_provenance
# -----------------------------------------------------------------------------


def test_modal_call_id_ledger_event_with_score():
    event = {
        "call_id": "fc-test123",
        "score": 0.2,
        "score_axis": "contest_cuda",
        "platform": "t4",
        "archive_sha256": "5" * 64,
        "written_at_utc": "2026-05-17T22:00:00Z",
    }
    prov = modal_call_id_ledger_event_to_provenance(event)
    assert prov.artifact_kind == ProvenanceKind.DERIVED_AGGREGATE
    # Conservative: not promotable until canonical embed
    assert not prov.promotion_eligible


def test_modal_call_id_ledger_event_without_score():
    event = {"call_id": "fc-pending"}
    prov = modal_call_id_ledger_event_to_provenance(event)
    assert prov.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR


# -----------------------------------------------------------------------------
# Adapter resilience
# -----------------------------------------------------------------------------


def test_adapter_returns_valid_provenance_for_garbage_input():
    """Adapter should NEVER raise; returns NULL sentinel on total failure."""
    # Try various malformed inputs
    for garbage in [{}, None, "string", 42, [], {"random": "stuff"}]:
        try:
            prov = contest_result_to_provenance(garbage)
            assert isinstance(prov, Provenance)
        except Exception:
            # If it does raise, the test fails (adapter must be resilient)
            # but accept TypeError on None as the only edge case
            if garbage is not None:
                pytest.fail(f"adapter raised on input {garbage!r}")
