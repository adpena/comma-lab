from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.seg_secant_rd_curve import (
    BREAK_EVEN_BYTES_PER_DSEG,
    CONTEST_PAIR_COUNT,
    SegSecantError,
    adjacent_seg_secants,
    default_operating_points,
    margin_ordered_abandonment,
    measure_parseback_payload,
    spatial_subsample_preimage_residual,
    summarize_per_class,
    truncate_preimage_residual_precision,
)


def test_default_grid_has_three_families_and_nine_unique_points() -> None:
    points = default_operating_points()
    assert len(points) == 9
    assert len({point.point_id for point in points}) == 9
    assert {point.family for point in points} == {
        "margin_abandonment",
        "precision_truncation",
        "spatial_subsample",
    }


def test_margin_abandonment_copies_complete_owned_blocks_only() -> None:
    source = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    predictor = np.full_like(source, 17)
    margins = np.array([[0.01, 0.20], [0.30, 0.02]], dtype=np.float64)
    rows = np.array([[0, 1], [4, 5]])
    cols = np.array([[0, 1], [4, 5]])

    result, telemetry = margin_ordered_abandonment(
        source,
        predictor,
        margins,
        row_indices=rows,
        col_indices=cols,
        threshold=0.1,
    )

    for row_offset in range(2):
        for col_offset in range(2):
            observed = result[rows[:, row_offset, None], cols[None, :, col_offset], :]
            expected = source[rows[:, row_offset, None], cols[None, :, col_offset], :]
            expected = expected.copy()
            expected[0, 0] = 17
            expected[1, 1] = 17
            np.testing.assert_array_equal(observed, expected)
    assert telemetry["abandoned_scorer_pixels"] == 2
    assert telemetry["abandoned_fraction"] == 0.5
    np.testing.assert_array_equal(source, np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))


def test_margin_abandonment_retains_threshold_equality() -> None:
    source = np.full((2, 2, 1), 9, dtype=np.uint8)
    predictor = np.full((2, 2, 1), 3, dtype=np.uint8)
    result, telemetry = margin_ordered_abandonment(
        source,
        predictor,
        np.array([[0.1]]),
        row_indices=np.array([[0, 1]]),
        col_indices=np.array([[0, 1]]),
        threshold=0.1,
    )
    np.testing.assert_array_equal(result, source)
    assert telemetry["abandoned_scorer_pixels"] == 0


def test_precision_truncation_is_signed_toward_zero_and_uint8() -> None:
    predictor = np.array([[[100], [100], [100], [100]]], dtype=np.uint8)
    source = np.array([[[107], [93], [104], [96]]], dtype=np.uint8)
    result, telemetry = truncate_preimage_residual_precision(
        source, predictor, drop_low_bits=2
    )
    np.testing.assert_array_equal(result.reshape(-1), np.array([104, 96, 104, 96]))
    assert result.dtype == np.uint8
    assert telemetry["changed_camera_values"] == 2


@pytest.mark.parametrize("bits", [-1, 8, 1.5, True])
def test_precision_truncation_refuses_invalid_depth(bits: object) -> None:
    frame = np.zeros((1, 1, 1), dtype=np.uint8)
    with pytest.raises(SegSecantError):
        truncate_preimage_residual_precision(frame, frame, drop_low_bits=bits)  # type: ignore[arg-type]


def test_spatial_subsample_reconstructs_affine_residual_exactly() -> None:
    yy, xx = np.mgrid[:5, :7]
    predictor = np.full((5, 7, 1), 100, dtype=np.uint8)
    residual = (2 * yy + 3 * xx)[..., None].astype(np.uint8)
    source = predictor + residual
    result, samples, telemetry = spatial_subsample_preimage_residual(
        source, predictor, sample_stride=2
    )
    np.testing.assert_array_equal(result, source)
    np.testing.assert_array_equal(samples, residual[::2, ::2])
    assert telemetry["sample_shape"] == [3, 4, 1]
    assert telemetry["clipped_camera_values"] == 0


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_spatial_subsample_refuses_invalid_stride(stride: object) -> None:
    frame = np.zeros((2, 2, 1), dtype=np.uint8)
    with pytest.raises(SegSecantError):
        spatial_subsample_preimage_residual(  # type: ignore[arg-type]
            frame, frame, sample_stride=stride
        )


def test_both_payload_codecs_parse_back_exactly() -> None:
    predictor = np.arange(24, dtype=np.int64).reshape(2, 4, 3)
    chosen = predictor + np.array(
        [
            [[0, 1, -1], [2, -2, 0], [3, 0, -3], [4, -4, 0]],
            [[5, 0, -5], [6, -6, 0], [7, 0, -7], [8, -8, 0]],
        ],
        dtype=np.int64,
    )
    measured = measure_parseback_payload(chosen, predictor)
    assert measured["brotli_q11"]["parseback_exact"] is True
    assert measured["zstd_19"]["parseback_exact"] is True
    assert measured["nonzero_values"] == 16
    assert measured["raw_bytes"] == chosen.size * 4


def test_per_class_summary_uses_truth_class_denominators() -> None:
    labels = np.array([[0, 0, 1], [1, 2, 2]])
    predicted = np.array([[0, 1, 1], [2, 2, 0]])
    result = summarize_per_class(labels, predicted, class_count=3)
    assert result["0"] == {"pixels": 2, "mismatches": 1, "d_seg_conditional": 0.5}
    assert result["1"] == {"pixels": 2, "mismatches": 1, "d_seg_conditional": 0.5}
    assert result["2"] == {"pixels": 2, "mismatches": 1, "d_seg_conditional": 0.5}


def test_secant_sign_accepts_distortion_only_above_byte_saving_break_even() -> None:
    points = [
        {
            "point_id": "source",
            "family": "reference",
            "d_seg": 0.0,
            "brotli_q11_bytes_per_pair": 1_000.0,
        },
        {
            "point_id": "cheap_to_prevent",
            "family": "margin",
            "d_seg": 1e-6,
            "brotli_q11_bytes_per_pair": 999.9,
        },
        {
            "point_id": "expensive_to_prevent",
            "family": "precision",
            "d_seg": 1e-6,
            "brotli_q11_bytes_per_pair": 999.7,
        },
    ]
    rows = adjacent_seg_secants(points, codec_key="brotli_q11_bytes_per_pair")
    by_family = {row["family"]: row for row in rows}
    assert BREAK_EVEN_BYTES_PER_DSEG == 150_181_956.0
    assert CONTEST_PAIR_COUNT == 600
    assert by_family["margin"]["n600_equivalent_bytes_saved_per_unit_d_seg"] == pytest.approx(
        60_000_000.0
    )
    assert by_family["margin"]["accept_higher_d_seg_improves_two_term_score"] is False
    assert by_family["precision"]["n600_equivalent_bytes_saved_per_unit_d_seg"] == pytest.approx(
        180_000_000.0
    )
    assert by_family["precision"]["accept_higher_d_seg_improves_two_term_score"] is True


def test_secants_skip_flat_and_retain_non_saving_dominated_rows() -> None:
    points = [
        {
            "point_id": "source",
            "family": "reference",
            "d_seg": 0.0,
            "zstd_19_bytes_per_pair": 100.0,
        },
        {
            "point_id": "flat",
            "family": "margin",
            "d_seg": 0.0,
            "zstd_19_bytes_per_pair": 90.0,
        },
        {
            "point_id": "more_bytes",
            "family": "precision",
            "d_seg": 0.1,
            "zstd_19_bytes_per_pair": 110.0,
        },
    ]
    rows = adjacent_seg_secants(points, codec_key="zstd_19_bytes_per_pair")
    assert len(rows) == 1
    assert rows[0]["family"] == "precision"
    assert rows[0]["bytes_saved_per_pair"] == -10.0
    assert rows[0]["byte_saving_direction"] == "higher_distortion_uses_more_bytes"
    assert rows[0]["accept_higher_d_seg_improves_two_term_score"] is False
    assert rows[0]["derived_two_term_delta_score"] > 0
