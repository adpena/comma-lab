# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    ADMIT,
    CUT,
    DEMOTE,
    PROTECT,
    RETRAIN,
    ContestBytePrice,
    build_nerv_byte_price_plan,
)


def _price() -> ContestBytePrice:
    return ContestBytePrice(
        score_per_byte=0.001,
        original_video_bytes=25_000,
        source="fixture_contest_byte_price",
    )


def _custody(**extra: object) -> dict[str, object]:
    row = {
        "archive_sha256": "a" * 64,
        "axis_tag": "[contest-CUDA]",
        "receiver_proof_status": "satisfied",
        "full_video_coverage": True,
    }
    row.update(extra)
    return row


def test_existing_section_rate_price_math_cuts_only_when_total_delta_negative() -> None:
    plan = build_nerv_byte_price_plan(
        [
            _custody(
                row_id="cut_selector",
                section_id="selectors_rc",
                bytes=100,
                delta_nonrate_score=0.050,
            ),
            _custody(
                row_id="protect_decoder",
                section_id="decoder_qw",
                bytes=100,
                delta_nonrate_score=0.150,
            ),
        ],
        byte_price=_price(),
    )

    rows = {row["row_id"]: row for row in plan["decision_rows"]}
    assert rows["cut_selector"]["delta_rate_score"] == -0.100
    assert rows["cut_selector"]["delta_total_score"] == -0.050
    assert rows["cut_selector"]["decision"] == CUT
    assert rows["protect_decoder"]["delta_rate_score"] == -0.100
    assert rows["protect_decoder"]["delta_total_score"] == pytest.approx(0.050)
    assert rows["protect_decoder"]["decision"] == PROTECT
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_missing_custody_fails_closed_even_when_math_would_cut() -> None:
    plan = build_nerv_byte_price_plan(
        [
            {
                "row_id": "uncustodied",
                "section_id": "latents_rc",
                "bytes": 100,
                "delta_nonrate_score": 0.0,
                "axis_tag": "[contest-CUDA]",
                "receiver_proof_status": "satisfied",
                "full_video_coverage": True,
            }
        ],
        byte_price=_price(),
    )

    row = plan["decision_rows"][0]
    assert row["economic_decision"] == CUT
    assert row["decision"] == DEMOTE
    assert "missing_archive_sha256" in row["blockers"]
    assert row["delta_total_score"] == -0.100


def test_residual_admission_is_strictly_negative_total_only() -> None:
    plan = build_nerv_byte_price_plan(
        [
            _custody(
                row_id="residual_break_even",
                section_id="residual_rc",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.100,
            ),
            _custody(
                row_id="residual_win",
                section_id="residual_rc_win",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.101,
            ),
            _custody(
                row_id="residual_signal_not_enough",
                section_id="residual_rc_weak",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.050,
            ),
        ],
        byte_price=_price(),
    )

    rows = {row["row_id"]: row for row in plan["decision_rows"]}
    assert rows["residual_break_even"]["delta_total_score"] == 0.0
    assert rows["residual_break_even"]["decision"] == RETRAIN
    assert rows["residual_win"]["delta_total_score"] < 0.0
    assert rows["residual_win"]["decision"] == ADMIT
    assert rows["residual_signal_not_enough"]["decision"] == RETRAIN
    assert plan["admitted_section_ids"] == ["residual_rc_win"]


def test_full_video_coverage_missing_blocks_admission() -> None:
    plan = build_nerv_byte_price_plan(
        [
            _custody(
                row_id="sampled_residual",
                section_id="residual_rc",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.200,
                full_video_coverage=False,
            )
        ],
        byte_price=_price(),
    )

    row = plan["decision_rows"][0]
    assert row["economic_decision"] == ADMIT
    assert row["decision"] == DEMOTE
    assert "full_video_coverage_missing" in row["blockers"]
    assert plan["full_video_coverage"] is False


def test_requested_max_pairs_does_not_fake_full_video_coverage() -> None:
    plan = build_nerv_byte_price_plan(
        [
            _custody(
                row_id="requested_full_but_sampled",
                section_id="residual_rc",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.200,
                full_video_coverage=None,
                max_pairs=600,
                n_samples=6,
            ),
            _custody(
                row_id="actually_full",
                section_id="residual_rc_full",
                row_kind="new_residual_sidecar",
                bytes=100,
                delta_nonrate_score=-0.200,
                full_video_coverage=None,
                evaluated_pairs=600,
            ),
        ],
        byte_price=_price(),
    )

    rows = {row["row_id"]: row for row in plan["decision_rows"]}
    assert rows["requested_full_but_sampled"]["economic_decision"] == ADMIT
    assert rows["requested_full_but_sampled"]["decision"] == DEMOTE
    assert "full_video_coverage_missing" in rows["requested_full_but_sampled"]["blockers"]
    assert rows["actually_full"]["decision"] == ADMIT


def test_artifact_level_runtime_ready_and_full_video_scope_are_custody_context() -> None:
    plan = build_nerv_byte_price_plan(
        {
            "schema": "compact_receiver_section_value_profile.v1",
            "candidate_id": "receiver_proven_candidate",
            "axis_tag": "[contest-CUDA]",
            "archive_sha256": "b" * 64,
            "runtime_consumption_proof": {
                "runtime_consumption_proof_ready": True,
                "score_claim": False,
            },
            "scope_status": {"full_video": "executed"},
            "section_value_rows": [
                {
                    "row_id": "cut_receiver_proven_selector",
                    "section_id": "selectors_rc",
                    "bytes": 100,
                    "delta_nonrate_score": 0.050,
                }
            ],
        },
        byte_price=_price(),
    )

    row = plan["decision_rows"][0]
    assert row["receiver_proof_status"] == "runtime_consumption_proof_ready"
    assert row["full_video_coverage"] is True
    assert row["decision"] == CUT
    assert "receiver_proof_not_satisfied" not in row["blockers"]
    assert "full_video_coverage_missing" not in row["blockers"]


def test_legacy_mlx_section_profile_gets_explicit_advisory_axis_without_authority() -> None:
    plan = build_nerv_byte_price_plan(
        {
            "schema": "hprc_mlx_component_neutralization_profile.v1",
            "source_schema": "pact_nerv_selector_v4_section_value_profile.v1",
            "candidate_archive": {"sha256": "c" * 64},
            "scope_status": {"full_video": "executed"},
            "blockers": ["mlx_local_response_is_advisory_not_score_authority"],
            "section_value_rows": [
                {
                    "variant_id": "neutralize_latents_rc",
                    "neutralized_section": "latents_rc",
                    "archive_bytes_removed_vs_baseline": 10_000,
                    "delta_nonrate_score": 0.0,
                    "receiver_proof_status": "satisfied",
                }
            ],
        },
        byte_price=_price(),
    )

    row = plan["decision_rows"][0]
    assert row["axis_labels"] == ["[macOS-MLX research-signal]"]
    assert row["economic_decision"] == CUT
    assert row["decision"] == DEMOTE
    assert "axis_label_missing" not in row["blockers"]
    assert "advisory_or_proxy_axis_not_promotion_authority" in row["blockers"]
    assert plan["score_claim"] is False


def test_missing_original_video_bytes_fallback_fails_closed() -> None:
    missing_price = ContestBytePrice(
        score_per_byte=None,
        original_video_bytes=None,
        source="unresolved",
        blockers=("contest_byte_price_unavailable_original_video_bytes_missing",),
    )

    plan = build_nerv_byte_price_plan(
        [
            _custody(
                row_id="selector",
                section_id="selectors_rc",
                bytes=100,
                delta_nonrate_score=0.0,
            )
        ],
        byte_price=missing_price,
    )

    row = plan["decision_rows"][0]
    assert row["delta_rate_score"] is None
    assert row["delta_total_score"] is None
    assert row["decision"] == DEMOTE
    assert "contest_byte_price_unavailable_original_video_bytes_missing" in row["blockers"]
    assert "contest_byte_price_missing_fail_closed" in row["blockers"]
