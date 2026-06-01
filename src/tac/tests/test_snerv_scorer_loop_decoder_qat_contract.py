# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV scorer-loop decoder/QAT contracts."""

from __future__ import annotations

import pytest

from tac.analysis.snerv_scorer_loop_decoder_qat_contract import (
    SnervScorerLoopDecoderQatContractError,
    build_snerv_scorer_loop_decoder_qat_contract,
)


def test_failed_pose_gate_routes_to_trainer_implementation_not_training_claim() -> None:
    contract = build_snerv_scorer_loop_decoder_qat_contract(
        _failed_gate(),
        source_gate_path=".omx/research/snerv_pose_guarded_decoder_gate.json",
        source_gate_sha256="abc123",
        dispatch_hold_reason="pr101_cpu_pending_blocks_exact_cuda_dispatch",
    ).as_jsonable()

    assert contract["ready_for_scorer_loop_trainer_implementation"] is True
    assert contract["ready_for_local_training_smoke"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["accepted_rows_in_source_gate"] == 0
    assert contract["closed_form_scalar_weighting_no_go"] is True
    assert contract["acceptance_gate"]["max_d_pose_linf"] == 2.13907
    assert contract["acceptance_gate"]["max_d_seg_linf"] == 0.02
    assert contract["allowed_training_modes"][0]["mode_id"] == "decoder_weight_linf_waterfill_qat"
    assert (
        "snerv_scorer_loop_decoder_qat_trainer_cli_missing"
        in contract["blockers"]
    )
    assert (
        "pr101_cpu_pending_blocks_exact_cuda_dispatch"
        in contract["blockers"]
    )
    assert (
        "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py"
        in contract["next_code_artifacts"]
    )


def test_gate_with_accepted_candidate_does_not_spawn_new_training_contract() -> None:
    gate = _failed_gate()
    gate["verdict"] = "GO_LOCAL_CONTINUATION_ONLY"
    gate["accepted_rows"] = [{"label": "candidate"}]

    contract = build_snerv_scorer_loop_decoder_qat_contract(gate).as_jsonable()

    assert contract["ready_for_scorer_loop_trainer_implementation"] is False
    assert "pose_gate_has_accepted_candidate_or_nonterminal_verdict" in contract["blockers"]
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_source_gate_score_claim_fails_closed() -> None:
    gate = _failed_gate()
    gate["score_claim"] = True

    with pytest.raises(SnervScorerLoopDecoderQatContractError, match="false-authority"):
        build_snerv_scorer_loop_decoder_qat_contract(gate)


def test_missing_baseline_fails_closed() -> None:
    gate = _failed_gate()
    del gate["baseline_score_linf"]

    with pytest.raises(SnervScorerLoopDecoderQatContractError, match="baseline_score"):
        build_snerv_scorer_loop_decoder_qat_contract(gate)


def _failed_gate() -> dict:
    return {
        "schema": "snerv_pose_guarded_decoder_gate.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "verdict": "NO_GO_FOR_PROMOTION_OR_EXACT_EVAL",
        "next_action": "implement_scorer_loop_or_nonlinear_qat_decoder_before_more_sweeps",
        "baseline_label": "least_squares_baseline_existing",
        "baseline_archive_bytes": 33_754,
        "baseline_d_seg_linf": 0.022644,
        "baseline_d_pose_linf": 2.13907,
        "baseline_score_linf": 6.91189,
        "max_archive_bytes": 35_802,
        "seg_ceiling": 0.02,
        "accepted_rows": [],
        "closed_form_scalar_weighting_no_go": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
