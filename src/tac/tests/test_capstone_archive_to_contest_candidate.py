# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the capstone closure pipeline tool.

These tests verify ACTUAL behavior, not constants: that the contest rate term
uses the real archive.zip file size, that the drift correction subtracts the
bias and adds the guard (conservative >= point), that the submit rule is the
conservative projection vs the frontier, that the eval packet is ARMED only on
a real sub-frontier candidate and NEVER reports a fired/score-claim state, and
that the end-to-end pipeline on a real on-disk capstone archive produces a
d_seg that matches the trainer's reloaded-int8 advisory bit-for-bit (argmax is
stable across the camera round-trip) while the rate is recomputed from the ZIP.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.capstone_archive_to_contest_candidate import (
    DEFAULT_BIAS_LOCAL_MINUS_CONTEST,
    DEFAULT_GUARD_BAND,
    PREDICTED_AXIS_TAG,
    RATE_SCALE,
    UNCOMPRESSED_SIZE_BYTES,
    build_eval_packet,
    predict_contest_cpu,
    run_closure_pipeline,
)

REPO = Path(__file__).resolve().parents[3]
SMOKE_VQ = REPO / "experiments/results/capstone_vq_index_smoke_b20_n8/archive.zip"
GT_CACHE = REPO / "experiments/results/capstone_gt_targets_cache"
N8_CACHE = GT_CACHE / "gt_targets_n8.pt"


# ---------------------------------------------------------------------------
# Step 3: drift correction (pure, no archive needed)
# ---------------------------------------------------------------------------
def test_rate_denominator_is_contest_uncompressed_size():
    # NO-FAKE: the rate denominator must be the contest's uncompressed size, not
    # an invented constant. This is the value evaluate.py charges against.
    assert UNCOMPRESSED_SIZE_BYTES == 37_545_489
    assert RATE_SCALE == 25.0


def test_conservative_projection_subtracts_bias_adds_guard():
    pred = predict_contest_cpu(0.190000, 0.191099824)
    # point = local - bias (macOS reads high -> subtract bias)
    assert pred.point_projection == pytest.approx(0.190000 - DEFAULT_BIAS_LOCAL_MINUS_CONTEST)
    # conservative = point + guard (worst-case-low-confidence, lower is better)
    assert pred.conservative_projection == pytest.approx(
        pred.point_projection + DEFAULT_GUARD_BAND
    )
    # conservative must be >= point (it adds the guard back)
    assert pred.conservative_projection > pred.point_projection


def test_submit_rule_conservative_beats_frontier():
    # A local 0.189987 conservatively projects below T_1=0.19 (drift memo 3.2).
    pred = predict_contest_cpu(0.189987, 0.19)
    assert pred.conservative_beats_frontier is True
    # Because bias (1.05e-5) > guard (3e-6), a local AT the frontier conservatively
    # projects ~7.5e-6 BELOW it (drift memo 3.2: net required beat is only ~1.3e-5,
    # so the bias-minus-guard cushion makes "local == frontier" still qualify).
    pred_at = predict_contest_cpu(0.191099824, 0.191099824)
    assert pred_at.conservative_beats_frontier is True
    assert pred_at.margin_vs_frontier == pytest.approx(
        DEFAULT_BIAS_LOCAL_MINUS_CONTEST - DEFAULT_GUARD_BAND, abs=1e-9
    )
    # A local that is +2e-5 ABOVE the frontier does NOT qualify after the cushion.
    pred_over = predict_contest_cpu(0.191099824 + 2e-5, 0.191099824)
    assert pred_over.conservative_beats_frontier is False


def test_prediction_is_never_a_score_claim():
    pred = predict_contest_cpu(0.18, 0.19).as_dict()
    assert pred["axis"] == PREDICTED_AXIS_TAG
    assert pred["score_claim"] is False
    assert pred["promotion_eligible"] is False
    assert "prediction_only" in pred["authority"]


# ---------------------------------------------------------------------------
# Step 4: eval packet arming
# ---------------------------------------------------------------------------
def test_eval_packet_armed_only_when_sub_frontier(tmp_path):
    sub = predict_contest_cpu(0.18999, 0.191099824)  # beats frontier
    over = predict_contest_cpu(0.20000, 0.191099824)  # above frontier
    pkt_sub = build_eval_packet(
        archive_path=Path("a.zip"), archive_sha256="ab" * 32, archive_zip_bytes=100,
        prediction=sub, candidate_id="c", out_dir=tmp_path,
    )
    pkt_over = build_eval_packet(
        archive_path=Path("a.zip"), archive_sha256="ab" * 32, archive_zip_bytes=100,
        prediction=over, candidate_id="c", out_dir=tmp_path,
    )
    assert pkt_sub["armed"] is True
    assert pkt_sub["recommended_action"] == "dispatch_contest_cpu_exact_eval"
    assert pkt_over["armed"] is False
    assert pkt_over["recommended_action"] == "observe_only"


def test_eval_packet_never_fired_and_carries_exact_archive_bytes(tmp_path):
    pred = predict_contest_cpu(0.18, 0.19)
    sha = "cd" * 32
    pkt = build_eval_packet(
        archive_path=Path("/x/archive.zip"), archive_sha256=sha,
        archive_zip_bytes=123456, prediction=pred, candidate_id="cand", out_dir=tmp_path,
    )
    # NO-FAKE: the packet is ARMED but never fired; no score claim.
    assert pkt["fired"] is False
    assert pkt["score_claim"] is False
    # The eval command pins the EXACT archive sha256 (no drift-substitution).
    assert sha in pkt["modal_cpu_eval_command"]
    assert "--expected-archive-sha256" in pkt["modal_cpu_eval_command"]
    assert "modal_auth_eval_cpu.py" in pkt["modal_cpu_eval_command"]
    # Lane-claim precedes dispatch (HARVEST-OR-LOSE + cross-agent coordination).
    assert "claim_lane_dispatch.py claim" in pkt["lane_claim_command"]


# ---------------------------------------------------------------------------
# End-to-end on a REAL on-disk capstone archive (the contest-faithful local path)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (SMOKE_VQ.exists() and N8_CACHE.exists()),
    reason="requires the n8 vq_index smoke archive + n8 GT targets cache",
)
def test_end_to_end_real_archive_rate_uses_zip_and_dseg_is_stable(tmp_path):
    rec = run_closure_pipeline(
        archive_path=SMOKE_VQ,
        targets_cache=GT_CACHE,
        frontier_cpu_score=0.191099824,
        out_dir=tmp_path / "closure",
        candidate_id="test_n8",
    )
    local = rec["contest_faithful_local_score"]
    # [1] the ACTUAL numpy inflate wrote camera-resolution frames to disk.
    assert rec["inflate"]["frames_shape"][1:] == [874, 1164, 3]
    assert rec["num_pairs"] == 8
    # [2] rate term recomputed from the REAL archive.zip file size (not payload).
    zip_bytes = SMOKE_VQ.stat().st_size
    assert local["archive_zip_bytes"] == zip_bytes
    assert local["score_rate_contribution"] == pytest.approx(
        RATE_SCALE * zip_bytes / UNCOMPRESSED_SIZE_BYTES
    )
    # d_seg is a real argmax-disagreement rate in [0, 1]; pose>0 (pose enabled).
    assert 0.0 <= local["avg_segnet_dist"] <= 1.0
    assert local["avg_posenet_dist"] > 0.0
    # S = 100*d_seg + sqrt(10*d_pose) + 25*rate, recomputed from components.
    assert local["score_recomputed_from_components"] == pytest.approx(
        100.0 * local["avg_segnet_dist"]
        + math.sqrt(10.0 * local["avg_posenet_dist"])
        + local["score_rate_contribution"]
    )
    # NON-PROMOTABLE advisory axis.
    assert local["axis"] == "[macOS-CPU advisory]"
    assert local["score_claim"] is False
    # n8 != 600 -> the pipeline honestly flags it as NOT a faithful contest score.
    assert local["full_600_pair_faithful"] is False


@pytest.mark.skipif(
    not (SMOKE_VQ.exists() and N8_CACHE.exists()),
    reason="requires the n8 vq_index smoke archive + n8 GT targets cache",
)
def test_end_to_end_dseg_matches_trainer_reloaded_advisory(tmp_path):
    # The disk-frame d_seg must match the trainer's render-res reloaded-int8 d_seg
    # bit-for-bit: argmax classification is stable across the camera round-trip.
    import json

    rec = run_closure_pipeline(
        archive_path=SMOKE_VQ, targets_cache=GT_CACHE,
        frontier_cpu_score=0.191099824, out_dir=tmp_path / "closure",
        candidate_id="test_n8_match",
    )
    res = json.loads(
        (SMOKE_VQ.parent / "capstone_result.json").read_text()
    )
    trainer_dseg = res["reloaded_int8_advisory"]["reloaded_int8_d_seg"]
    assert rec["contest_faithful_local_score"]["avg_segnet_dist"] == pytest.approx(
        trainer_dseg, abs=1e-9
    )
