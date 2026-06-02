# SPDX-License-Identifier: MIT
"""NO-FAKE tests for PACT-NeRV-VQ competitiveness gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.substrates.pact_nerv_vq.competitiveness_gate import (
    PactVqCompetitivenessGateError,
    build_pact_vq_competitiveness_gate,
    build_pact_vq_competitiveness_gate_from_paths,
)


def test_rate_only_codec_win_is_preserved_but_not_exact_spend(tmp_path: Path) -> None:
    source_response = _response(
        score=90.66354296056916,
        d_seg=0.5048259229958058,
        d_pose=161.237585550944,
        rate=0.02655045989679346,
        bytes_=39874,
    )
    best_response = _response(
        score=90.66201548013069,
        d_seg=0.5048259229958058,
        d_pose=161.237585550944,
        rate=0.0250229794583312,
        bytes_=37580,
    )
    source_profile = _profile_with_response(tmp_path, "source", source_response)
    best_profile = _profile_with_response(tmp_path, "best", best_response)

    gate = build_pact_vq_competitiveness_gate(
        codec_sweep_report=_codec_sweep(),
        source_replay_profile=source_profile,
        best_codec_replay_profile=best_profile,
    )

    assert gate["verdict"] == "PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION"
    assert gate["abandon_family"] is False
    assert gate["abandon_current_artifact"] is False
    assert gate["demote_for_full_stack_portfolio"] is False
    assert gate["exact_axis_blocked"] is True
    assert gate["preserve_rate_primitive"] is True
    assert gate["exact_spend_candidate"] is False
    assert gate["best_decoder_codec"] == "int2_mixed"
    assert gate["best_codec_receiver_proof_passed"] is True
    assert gate["deltas"]["score"] < 0
    assert gate["deltas"]["nonrate_score"] == pytest.approx(0.0)
    assert gate["rate_axis"]["rate_only_replay"] is True
    assert gate["rate_axis"]["bytes_saved_vs_source_replay"] == 2294
    assert "best_codec_improves_rate_only_distortion_unchanged" in gate["blockers"]
    assert "pact_vq_distortion_not_competitive_at_current_fit" in gate["blockers"]
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["promotion_eligible"] is False
    assert gate["score_claim"] is False
    assert "preserve_best_codec_as_rate_primitive_for_full_stack_portfolio" in gate[
        "recommended_next_actions"
    ]
    assert "optimize_order_dependent_full_stack_interaction_before_design_retirement" in gate[
        "recommended_next_actions"
    ]


def test_gate_accepts_low_distortion_local_candidate_but_keeps_false_authority(
    tmp_path: Path,
) -> None:
    source_profile = _profile_with_response(
        tmp_path,
        "source",
        _response(score=0.9, d_seg=0.008, d_pose=0.2, rate=0.03, bytes_=40_000),
    )
    best_profile = _profile_with_response(
        tmp_path,
        "best",
        _response(score=0.8, d_seg=0.007, d_pose=0.18, rate=0.02, bytes_=30_000),
    )

    gate = build_pact_vq_competitiveness_gate(
        codec_sweep_report=_codec_sweep(),
        source_replay_profile=source_profile,
        best_codec_replay_profile=best_profile,
    )

    assert gate["verdict"] == "LOCAL_CANDIDATE_FOR_EXACT_SPEND_TRIAGE"
    assert gate["exact_spend_candidate"] is True
    assert gate["abandon_current_artifact"] is False
    assert gate["exact_axis_blocked"] is False
    assert gate["preserve_rate_primitive"] is True
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["score_claim_valid"] is False
    assert "run_claimed_exact_cpu_cuda_pair_replay_before_any_score_claim" in gate[
        "recommended_next_actions"
    ]


def test_gate_refuses_authority_inputs(tmp_path: Path) -> None:
    source_profile = _profile_with_response(
        tmp_path,
        "source",
        _response(score=0.9, d_seg=0.008, d_pose=0.2, rate=0.03, bytes_=40_000),
    )
    best_profile = _profile_with_response(
        tmp_path,
        "best",
        _response(score=0.8, d_seg=0.007, d_pose=0.18, rate=0.02, bytes_=30_000),
    )
    sweep = _codec_sweep()
    sweep["ready_for_exact_eval_dispatch"] = True

    with pytest.raises(PactVqCompetitivenessGateError, match="authority"):
        build_pact_vq_competitiveness_gate(
            codec_sweep_report=sweep,
            source_replay_profile=source_profile,
            best_codec_replay_profile=best_profile,
        )


def test_gate_from_paths_reads_nested_baseline_response(tmp_path: Path) -> None:
    source_profile = _profile_with_response(
        tmp_path,
        "source",
        _response(score=90.0, d_seg=0.5, d_pose=160.0, rate=0.03, bytes_=40_000),
    )
    best_profile = _profile_with_response(
        tmp_path,
        "best",
        _response(score=89.99, d_seg=0.5, d_pose=160.0, rate=0.02, bytes_=30_000),
    )
    sweep_path = tmp_path / "sweep.json"
    source_path = tmp_path / "source_profile.json"
    best_path = tmp_path / "best_profile.json"
    sweep_path.write_text(json.dumps(_codec_sweep()), encoding="utf-8")
    source_path.write_text(json.dumps(source_profile), encoding="utf-8")
    best_path.write_text(json.dumps(best_profile), encoding="utf-8")

    gate = build_pact_vq_competitiveness_gate_from_paths(
        codec_sweep_report_path=sweep_path,
        source_replay_profile_path=source_path,
        best_codec_replay_profile_path=best_path,
    )

    assert gate["best_decoder_codec"] == "int2_mixed"
    assert gate["source"]["response_path"].endswith("source_response.json")
    assert gate["best_codec"]["response_path"].endswith("best_response.json")


def _codec_sweep() -> dict:
    return {
        "schema": "compact_decoder_codec_sweep.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "full_video_mlx_scorer_replay_not_attached",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "best_variant": {
            "decoder_codec": "int2_mixed",
            "archive_bytes": 37_641,
            "archive_sha256": "sha-int2",
            "receiver_proof_passed": True,
            "blockers": [
                "full_video_mlx_scorer_replay_not_attached",
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
        },
    }


def _profile_with_response(tmp_path: Path, name: str, response: dict) -> dict:
    response_path = tmp_path / f"{name}_response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return {
        "schema": "hprc_mlx_component_neutralization_profile.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "variant_rows": [
            {
                "variant_id": "baseline",
                "archive_zip_bytes": response["archive_size_bytes"],
                "mlx_response": response_path.as_posix(),
            }
        ],
    }


def _response(
    *,
    score: float,
    d_seg: float,
    d_pose: float,
    rate: float,
    bytes_: int,
) -> dict:
    return {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "avg_segnet_dist": d_seg,
        "avg_posenet_dist": d_pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": bytes_,
        "evidence_tag": "[macOS-MLX research-signal]",
        "score_axis": "[macOS-MLX research-signal]",
        "max_pairs": 600,
        "n_samples": 600,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
