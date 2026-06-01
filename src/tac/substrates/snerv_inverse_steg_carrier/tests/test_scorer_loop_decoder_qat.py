# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV scorer-loop decoder/QAT helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    QuantizedDecoderStats,
    SnervDecoderEval,
    SnervScorerLoopDecoderQatError,
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


def test_decoder_trial_pose_guard_requires_receiver_replay() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(label="no_replay", score=6.0, d_pose=1.0, d_seg=0.01, replay=False)

    assert decoder_trial_passes_pose_guard(candidate, current) is False


def test_quantize_decoder_rejects_invalid_bit_width() -> None:
    with pytest.raises(SnervScorerLoopDecoderQatError, match="bits"):
        quantize_decoder_for_qat(_decoder(), bits=1)


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
) -> SnervDecoderEval:
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
    )
