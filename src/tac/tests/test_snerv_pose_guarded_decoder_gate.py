# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the SNeRV pose-guarded decoder gate."""

from __future__ import annotations

import pytest

from tac.analysis.snerv_pose_guarded_decoder_gate import (
    SnervPoseGuardedDecoderGateError,
    build_snerv_pose_guarded_decoder_gate,
)


def test_closed_form_seg_gain_pose_damage_is_no_go() -> None:
    payload = {
        "rows": [
            _row(
                label="least_squares_baseline_existing",
                mode="least_squares",
                archive=33_754,
                d_seg=0.022644,
                d_pose=2.13907,
                score=6.91189,
            ),
            _row(
                label="pose_gain_0.01",
                mode="score_weighted",
                component="pose",
                gain=0.01,
                archive=33_864,
                d_seg=0.01947,
                d_pose=3.39478,
                score=7.79656,
            ),
        ]
    }

    gate = build_snerv_pose_guarded_decoder_gate([payload]).as_jsonable()

    assert gate["verdict"] == "NO_GO_FOR_PROMOTION_OR_EXACT_EVAL"
    assert gate["accepted_rows"] == []
    assert gate["closed_form_scalar_weighting_no_go"] is True
    assert "closed_form_scalar_component_weighting_no_go" in gate["blockers"]
    assert gate["rows"][0]["passes_seg_gate"] is True
    assert gate["rows"][0]["passes_pose_guard"] is False
    assert "pose_guard_failed" in gate["rows"][0]["blockers"]
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["score_claim"] is False


def test_candidate_must_improve_score_and_hold_pose() -> None:
    payload = {
        "rows": [
            _row(
                label="least_squares_baseline_existing",
                mode="least_squares",
                archive=33_754,
                d_seg=0.022644,
                d_pose=2.13907,
                score=6.91189,
            ),
            _row(
                label="scorer_loop_candidate",
                mode="nonlinear_qat",
                archive=34_000,
                d_seg=0.019,
                d_pose=2.10,
                score=6.80,
            ),
        ]
    }

    gate = build_snerv_pose_guarded_decoder_gate([payload]).as_jsonable()

    assert gate["verdict"] == "GO_LOCAL_CONTINUATION_ONLY"
    assert gate["accepted_rows"][0]["label"] == "scorer_loop_candidate"
    assert gate["accepted_rows"][0]["accepted_for_local_continuation"] is True
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["promotion_eligible"] is False


def test_missing_baseline_fails_closed() -> None:
    payload = {
        "rows": [
            _row(
                label="score_weighted_only",
                mode="score_weighted",
                component="pose",
                gain=0.1,
                archive=33_900,
                d_seg=0.019,
                d_pose=3.0,
                score=7.5,
            )
        ]
    }

    with pytest.raises(SnervPoseGuardedDecoderGateError, match="baseline"):
        build_snerv_pose_guarded_decoder_gate([payload])


def _row(
    *,
    label: str,
    mode: str,
    archive: int,
    d_seg: float,
    d_pose: float,
    score: float,
    component: str | None = None,
    gain: float | None = None,
) -> dict:
    row = {
        "sweep_label": label,
        "hf_decoder_fit_mode": mode,
        "archive_bytes_total": archive,
        "receiver_archive_sha256": f"sha-{label}",
        "receiver_archive_replay_verified": True,
        "d_seg_mean_linf": d_seg,
        "d_pose_mean_linf": d_pose,
        "score_linf": score,
    }
    if component is not None:
        row["hf_decoder_saliency_component"] = component
    if gain is not None:
        row["hf_decoder_saliency_gain"] = gain
    return row
