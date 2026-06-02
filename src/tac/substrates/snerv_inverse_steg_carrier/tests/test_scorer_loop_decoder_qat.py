# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV scorer-loop decoder/QAT helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    QuantizedDecoderStats,
    SnervDecoderEval,
    SnervPairEval,
    SnervScorerLoopDecoderQatError,
    _nes_pair_robust_objective,
    decoder_eval_pair_deltas,
    decoder_search_direction_labels,
    decoder_trial_passes_pose_guard,
    quantize_decoder_for_qat,
)


def test_quantize_decoder_for_qat_changes_off_grid_weights_and_preserves_shape() -> None:
    decoder = _decoder()

    quantized, stats = quantize_decoder_for_qat(decoder, bits=4)

    assert quantized.levels == decoder.levels
    assert stats.bits == 4
    assert stats.total_weights == 27
    assert stats.payload_bytes_fp32_receiver > 0
    assert len(stats.payload_sha256_fp32_receiver) == 64
    assert stats.max_abs_error > 0.0
    assert quantized.kernels[0]["LH"].shape == (3, 3)
    assert not np.array_equal(quantized.kernels[0]["LH"], decoder.kernels[0]["LH"])


def test_decoder_trial_pose_guard_refuses_score_gain_with_pose_damage() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(
        label="lower_score_pose_damage",
        score=6.8,
        d_pose=2.1,
        d_seg=0.01,
        replay=True,
    )

    assert decoder_trial_passes_pose_guard(candidate, current, pose_slack=0.0) is False
    assert decoder_trial_passes_pose_guard(candidate, current, pose_slack=0.2) is True


def test_decoder_trial_pose_guard_refuses_score_gain_with_seg_damage() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.01, replay=True)
    candidate = _eval(
        label="lower_score_seg_damage",
        score=6.8,
        d_pose=1.9,
        d_seg=0.011,
        replay=True,
    )

    assert decoder_trial_passes_pose_guard(candidate, current, pose_slack=0.0) is False


def test_decoder_trial_pose_guard_can_allow_explicit_seg_slack() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.01, replay=True)
    candidate = _eval(
        label="lower_score_pose_safe_seg_slack",
        score=6.8,
        d_pose=1.9,
        d_seg=0.011,
        replay=True,
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is False
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            seg_slack=0.001,
        )
        is True
    )


def test_decoder_trial_pose_guard_requires_receiver_replay() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(label="no_replay", score=6.0, d_pose=1.0, d_seg=0.01, replay=False)

    assert decoder_trial_passes_pose_guard(candidate, current) is False


def test_decoder_trial_pose_guard_can_require_pair_robust_score_improvement() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="aggregate_win_pair_cancellation",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.011,
                d_pose_linf=0.29,
                score_linf_without_rate=2.1,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.027,
                d_pose_linf=0.49,
                score_linf_without_rate=4.5,
            ),
        ),
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pair_guard_min_score_improved_fraction=1.0,
        )
        is False
    )


def test_decoder_trial_pose_guard_can_limit_pair_pose_worsening() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="aggregate_win_pose_cancellation",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.31,
                score_linf_without_rate=1.9,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.029,
                d_pose_linf=0.47,
                score_linf_without_rate=4.8,
            ),
        ),
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pair_guard_max_pose_worsened_fraction=0.0,
        )
        is False
    )


def test_decoder_eval_json_preserves_pair_local_detector_response() -> None:
    row = _eval(
        label="candidate",
        score=6.0,
        d_pose=0.5,
        d_seg=0.01,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.5,
                score_linf_without_rate=3.23606797749979,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.02,
                d_pose_linf=0.25,
                score_linf_without_rate=3.58113883008419,
            ),
        ),
    )

    payload = row.as_jsonable()

    assert payload["per_pair"] == [
        {
            "pair_index": 0,
            "d_seg_linf": 0.01,
            "d_pose_linf": 0.5,
            "score_linf_without_rate": 3.23606797749979,
        },
        {
            "pair_index": 1,
            "d_seg_linf": 0.02,
            "d_pose_linf": 0.25,
            "score_linf_without_rate": 3.58113883008419,
        },
    ]


def test_decoder_eval_pair_deltas_preserve_direction_per_pair() -> None:
    baseline = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.7,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="candidate",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.25,
                score_linf_without_rate=2.1,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.031,
                d_pose_linf=0.53,
                score_linf_without_rate=5.2,
            ),
        ),
    )

    deltas = decoder_eval_pair_deltas(baseline, candidate)

    assert [row.as_jsonable() for row in deltas] == [
        {
            "pair_index": 0,
            "d_seg_linf_delta": pytest.approx(-0.001),
            "d_pose_linf_delta": pytest.approx(-0.05),
            "score_linf_without_rate_delta": pytest.approx(-0.6),
        },
        {
            "pair_index": 1,
            "d_seg_linf_delta": pytest.approx(0.001),
            "d_pose_linf_delta": pytest.approx(0.03),
            "score_linf_without_rate_delta": pytest.approx(0.2),
        },
    ]


def test_decoder_eval_pair_deltas_fail_closed_on_pair_mismatch() -> None:
    baseline = _eval(label="baseline", score=7.0, d_pose=0.4, d_seg=0.02, replay=True)
    candidate = _eval(
        label="candidate",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=9,
                d_seg_linf=0.009,
                d_pose_linf=0.25,
                score_linf_without_rate=2.1,
            ),
        ),
    )

    with pytest.raises(SnervScorerLoopDecoderQatError, match="missing from baseline"):
        decoder_eval_pair_deltas(baseline, candidate)


def test_quantize_decoder_rejects_invalid_bit_width() -> None:
    with pytest.raises(SnervScorerLoopDecoderQatError, match="bits"):
        quantize_decoder_for_qat(_decoder(), bits=1)


def test_top_weight_coordinate_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=2,
        search_mode="top_weight_coordinate",
    )

    assert labels == ("coord_001", "coord_002")


def test_learned_random_subspace_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=3,
        search_mode="learned_random_subspace",
        seed=99,
    )

    assert labels == (
        "learned_subspace_001",
        "learned_subspace_002",
        "learned_subspace_003",
    )


def test_nes_pair_robust_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=3,
        search_mode="nes_pair_robust",
        seed=99,
    )

    assert labels == ("nes_probe_001", "nes_probe_002", "nes_probe_003")


def test_nes_pair_robust_objective_penalizes_pair_pose_damage() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.40,
        d_seg=0.020,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.010,
                d_pose_linf=0.20,
                score_linf_without_rate=4.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.020,
                d_pose_linf=0.40,
                score_linf_without_rate=3.0,
            ),
        ),
    )
    pair_robust = _eval(
        label="pair_robust",
        score=6.8,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.19,
                score_linf_without_rate=3.8,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.019,
                d_pose_linf=0.39,
                score_linf_without_rate=2.9,
            ),
        ),
    )
    pose_damaged = _eval(
        label="pose_damaged",
        score=6.7,
        d_pose=0.41,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.19,
                score_linf_without_rate=3.8,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.019,
                d_pose_linf=0.42,
                score_linf_without_rate=2.8,
            ),
        ),
    )

    assert _nes_pair_robust_objective(
        pair_robust,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        pair_guard_min_score_improved_fraction=1.0,
        pair_guard_max_pose_worsened_fraction=0.0,
    ) < _nes_pair_robust_objective(
        pose_damaged,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        pair_guard_min_score_improved_fraction=1.0,
        pair_guard_max_pose_worsened_fraction=0.0,
    )


def test_decoder_search_direction_labels_rejects_unknown_mode() -> None:
    with pytest.raises(SnervScorerLoopDecoderQatError, match="search_mode"):
        decoder_search_direction_labels(
            np.array([1.0, 2.0]),
            max_trials=1,
            search_mode="not_a_mode",
        )


def _decoder() -> HfGenerationDecoder:
    base = np.array(
        [
            [0.0103, -0.0217, 0.0331],
            [0.0449, -0.0555, 0.0662],
            [0.0777, -0.0888, 0.0999],
        ],
        dtype=np.float64,
    )
    return HfGenerationDecoder(
        kernels={
            0: {
                "LH": base,
                "HL": base * 0.5,
                "HH": -base * 0.25,
            }
        },
        levels=1,
    )


def _eval(
    *,
    label: str,
    score: float,
    d_pose: float,
    replay: bool,
    d_seg: float = 0.01,
    per_pair: tuple[SnervPairEval, ...] | None = None,
) -> SnervDecoderEval:
    if per_pair is None:
        per_pair = (
            SnervPairEval(
                pair_index=0,
                d_seg_linf=float(d_seg),
                d_pose_linf=float(d_pose),
                score_linf_without_rate=100.0 * float(d_seg)
                + float(np.sqrt(10.0 * max(float(d_pose), 0.0))),
            ),
        )
    return SnervDecoderEval(
        label=label,
        iteration=0,
        archive_bytes=1234,
        archive_sha256="0" * 64,
        d_seg_linf=d_seg,
        d_pose_linf=d_pose,
        score_linf=score,
        rate_term=0.001,
        receiver_archive_replay_verified=replay,
        accepted=False,
        blockers=() if replay else ("receiver_archive_replay_failed",),
        quantized_decoder=QuantizedDecoderStats(
            bits=8,
            scale_count=3,
            zero_scale_count=0,
            total_weights=27,
            zero_weight_fraction=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            payload_bytes_fp32_receiver=99,
            payload_sha256_fp32_receiver="1" * 64,
        ),
        per_pair=per_pair,
    )
