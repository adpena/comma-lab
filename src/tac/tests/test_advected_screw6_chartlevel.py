# SPDX-License-Identifier: MIT
"""Focused tests for full-screw custody and chart coefficient packets."""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.predict_project_receiver import (
    ChartRGBCoefficientPacket,
    PredictProjectReceiverError,
    apply_chart_rgb_coefficients,
    counted_full_screw_xi_series,
    decode_chart_rgb_coefficients,
    encode_chart_rgb_coefficients,
    fit_chart_rgb_coefficients,
)


def test_full_screw_consumes_all_six_source_coordinates() -> None:
    poses = np.asarray(
        [
            [30.0, 0.1, -0.2, 0.01, -0.02, 0.03],
            [31.0, -0.2, 0.3, -0.04, 0.05, -0.06],
        ],
        dtype=np.float64,
    )
    xi, custody = counted_full_screw_xi_series(
        poses,
        translation_scale=0.16,
        rotation_scale=1.0,
        pitch_rad=-0.05,
        source_sha256="1" * 64,
    )
    assert xi.shape == (2, 6)
    assert np.count_nonzero(xi, axis=0).tolist() == [2, 2, 2, 2, 2, 2]
    assert custody["all_six_source_coordinates_consumed"] is True
    assert custody["additional_video_derived_bytes"] == 0
    assert np.linalg.norm(xi, axis=1).min() > 1.0


def test_full_screw_refuses_a_zeroed_rotation_scale() -> None:
    with pytest.raises(PredictProjectReceiverError, match="translation and rotation active"):
        counted_full_screw_xi_series(
            np.ones((2, 6), dtype=np.float64),
            translation_scale=0.16,
            rotation_scale=0.0,
            pitch_rad=0.0,
            source_sha256="2" * 64,
        )


def test_full_screw_refuses_placeholder_source_hash() -> None:
    with pytest.raises(PredictProjectReceiverError, match="SHA-256"):
        counted_full_screw_xi_series(
            np.ones((2, 6), dtype=np.float64),
            translation_scale=0.16,
            rotation_scale=1.0,
            pitch_rad=0.0,
            source_sha256="0" * 64,
        )


def test_chart_coefficient_packet_is_strict_and_receiver_consumed() -> None:
    baseline = np.full((4, 5, 3), 100, dtype=np.uint8)
    chart = np.asarray(
        [
            [0, 0, 1, 1, 2],
            [0, 3, 3, 1, 2],
            [4, 4, 3, 2, 2],
            [4, 0, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    offsets = np.asarray(
        [[1, 2, 3], [4, 5, 6], [-1, -2, -3], [7, 8, 9], [-4, -5, -6]],
        dtype=np.int16,
    )
    target = np.clip(baseline.astype(np.int16) + offsets[chart], 0, 255).astype(np.uint8)
    coefficients, scale = fit_chart_rgb_coefficients(baseline, target, chart)
    packet = ChartRGBCoefficientPacket(
        coefficients=coefficients[None],
        scales=np.asarray([scale], dtype="<f2"),
    )
    payload = encode_chart_rgb_coefficients(packet)
    decoded = decode_chart_rgb_coefficients(payload)
    assert encode_chart_rgb_coefficients(decoded) == payload
    assert np.array_equal(apply_chart_rgb_coefficients(baseline, chart, decoded, 0), target)
    with pytest.raises(PredictProjectReceiverError):
        decode_chart_rgb_coefficients(payload + b"trailing")
