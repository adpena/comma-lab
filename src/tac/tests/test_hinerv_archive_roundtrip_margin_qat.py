# SPDX-License-Identifier: MIT
"""Decision-table behavior of the ArchiveRoundtripMarginQAT aggregator receipt.

All survival rows here are synthetic FIXTURES (labelled, no MLX) used to verify
that the aggregator runs the mechanical four-surface decision table correctly
and stays false-authority.  The real rows come from the in-loop live smoke.
"""

from __future__ import annotations

import pytest

from tac.analysis.hinerv_archive_roundtrip_margin_qat import (
    HI_NERV_ARCHIVE_ROUNDTRIP_MARGIN_QAT_SCHEMA,
    build_hinerv_archive_roundtrip_margin_qat_receipt,
)

ACTION = "a" * 64
_FORBIDDEN = {
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
}


def _surface_row(surface: str, *, wrong: int, live: int, survived: bool, action_id: str = ACTION) -> dict:
    retention = (wrong / live) if live > 0 else None
    row = {
        "schema": "hi_nerv_target_region_birth_survival.v1",
        "surface": surface,
        "action_id": action_id,
        "survived": survived,
        "live_wrong_to_target": live,
        "surface_wrong_to_target_count": wrong,
        "scorer_effect_retention_ratio": retention,
        "target_margin_p10": None,
    }
    row[f"{surface}_wrong_to_target"] = wrong
    return row


def _build(fq, shadow, pb, **kw):
    return build_hinerv_archive_roundtrip_margin_qat_receipt(
        fakequant_survival=fq,
        archive_roundtrip_shadow_survival=shadow,
        parseback_survival=pb,
        **kw,
    )


def test_case_a_latents_fine_quantizer_mismatch_recommends_backend_qat() -> None:
    # fakequant high (0.90), shadow low (0.0001), parseback low (0.0001).
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=2, live=13488, survived=False),
        _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
    )
    assert receipt["schema"] == HI_NERV_ARCHIVE_ROUNDTRIP_MARGIN_QAT_SCHEMA
    assert receipt["interpretation_case"].startswith("A_")
    assert receipt["recommended_lowering"] == "backend_qat"
    assert receipt["first_failed_surface"] == "archive_roundtrip_shadow"
    assert receipt["first_failed_section"] == "latents_fine"
    assert receipt["fakequant_surface_state"] == "high"
    assert receipt["archive_roundtrip_shadow_surface_state"] == "low"
    # Custody: aggregator is false-authority, never promotable.
    assert receipt["promotion_eligible"] is False
    assert not (_FORBIDDEN & {k for k, v in receipt.items() if v is True})


def test_case_b_shadow_survives_blames_export_not_latents() -> None:
    # fakequant high, shadow high, parseback low -> NOT latents_fine.
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=12000, live=13488, survived=True),
        _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
    )
    assert receipt["interpretation_case"].startswith("B_")
    assert receipt["recommended_lowering"] == "audit_export_selection_or_decoder_sections"
    # The shadow survived, so the FIRST failed surface is parse-back, not shadow.
    assert receipt["first_failed_surface"] == "parseback_mlx"
    assert receipt["first_failed_section"] is None


def test_case_c_shadow_collapses_but_parseback_survives_is_reconcile() -> None:
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=2, live=13488, survived=False),
        _surface_row("parseback_mlx", wrong=12000, live=13488, survived=True),
    )
    assert receipt["interpretation_case"].startswith("C_")
    assert receipt["recommended_lowering"] == "compare_shadow_vs_parseback_decoded_tensors"
    # Earliest collapse in pipeline order is the shadow.
    assert receipt["first_failed_surface"] == "archive_roundtrip_shadow"


def test_case_d_all_survive_routes_to_gate() -> None:
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=12100, live=13488, survived=True),
        _surface_row("parseback_mlx", wrong=12000, live=13488, survived=True),
    )
    assert receipt["interpretation_case"].startswith("D_")
    assert receipt["recommended_lowering"] == "proceed_to_gate_and_lowering_race"
    assert receipt["first_failed_surface"] is None


def test_case_e_all_low_is_identity_or_stale_audit() -> None:
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=1, live=13488, survived=False),
        _surface_row("archive_roundtrip_shadow", wrong=1, live=13488, survived=False),
        _surface_row("parseback_mlx", wrong=1, live=13488, survived=False),
    )
    assert receipt["interpretation_case"].startswith("E_")
    assert receipt["recommended_lowering"] == "audit_action_identity_and_custody"
    # fakequant is the earliest surface and it is low.
    assert receipt["first_failed_surface"] == "fakequant_mlx"


def test_missing_shadow_row_is_incomplete_not_a_false_clear() -> None:
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        None,  # shadow not yet emitted (the in-loop smoke has not run)
        _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
    )
    assert receipt["interpretation_case"].startswith("U_")
    assert "archive_roundtrip_shadow_surface_row_missing" in receipt["blockers"]
    assert receipt["archive_roundtrip_shadow_surface_state"] == "unknown"


def test_action_id_mismatch_is_flagged_not_silently_aggregated() -> None:
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=2, live=13488, survived=False, action_id="b" * 64),
        _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
    )
    assert receipt["action_id_consistent_across_surfaces"] is False
    assert "archive_roundtrip_margin_qat_action_id_mismatch_across_surfaces" in receipt["blockers"]


def test_truthy_authority_in_surface_row_is_refused() -> None:
    bad = _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True)
    bad["promotion_eligible"] = True  # forged authority on a survival row
    with pytest.raises(ValueError, match="forbidden truthy authority"):
        _build(
            bad,
            _surface_row("archive_roundtrip_shadow", wrong=2, live=13488, survived=False),
            _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
        )


def test_real_artifact_retention_numbers_reproduce_case_a() -> None:
    # The exact 20260607 numbers: live 13488, fakequant 12183, parseback 2.
    # With the shadow at parse-back level, this MUST classify as Case A.
    receipt = _build(
        _surface_row("fakequant_mlx", wrong=12183, live=13488, survived=True),
        _surface_row("archive_roundtrip_shadow", wrong=2, live=13488, survived=False),
        _surface_row("parseback_mlx", wrong=2, live=13488, survived=False),
    )
    assert receipt["fakequant_retention"] == pytest.approx(12183 / 13488)
    assert receipt["parseback_retention"] == pytest.approx(2 / 13488)
    assert receipt["interpretation_case"].startswith("A_")
