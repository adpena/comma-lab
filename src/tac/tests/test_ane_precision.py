"""Tests for :mod:`tac.ane_precision` -- the ddm_ane2 selective-precision algebra.

These run in the MAIN ``.venv``, which carries no ``coremltools``: that is the
point.  The split construction, the op-sequence identity guard, the per-axis
verdicts and the realized-hybrid geometry must all be checkable without an ANE,
so a reviewer can falsify the arithmetic without owning the hardware.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac.ane_precision import (
    POSE_D_POSE_T4_EXACT,
    POSE_PER_DIM_TOLERANCE,
    SPLIT_BACKENDS,
    AnePrecisionError,
    assert_op_sequence_stable,
    compute_op_names,
    crop_boxes,
    crop_boxes_with_cores,
    crop_pixel_fraction,
    dilate_bool,
    fixed_crop_boxes,
    group_ranges,
    hybrid_speedup,
    margin_band_mask,
    occupied_tiles,
    pose_drift_verdict,
    seg_flip_verdict,
    selective_fp32_names,
    selector_from_names,
    split_backend_name,
    split_fp16_names,
)

RECORDS = [
    ("const", "w0"),
    ("conv", "conv_0"),
    ("const", "w1"),
    ("batch_norm", "bn_0"),
    ("relu", "relu_0"),
    ("conv", "conv_1"),
    ("linear", "head"),
]


# ------------------------------------------------------------- op sequence


def test_compute_op_names_drops_consts_and_keeps_order():
    assert compute_op_names(RECORDS) == ("conv_0", "bn_0", "relu_0", "conv_1", "head")


def test_compute_op_names_refuses_duplicate_names():
    with pytest.raises(AnePrecisionError, match="duplicate MIL op name"):
        compute_op_names([("conv", "a"), ("relu", "a")])


def test_compute_op_names_refuses_an_all_const_program():
    with pytest.raises(AnePrecisionError, match="no compute ops"):
        compute_op_names([("const", "w0"), ("const", "w1")])


def test_assert_op_sequence_stable_accepts_identical_sequences():
    names = compute_op_names(RECORDS)
    assert_op_sequence_stable(names, list(names), context="rung")


def test_assert_op_sequence_stable_names_the_length_drift():
    with pytest.raises(AnePrecisionError, match="length drifted in rung"):
        assert_op_sequence_stable(("a", "b"), ("a",), context="rung")


def test_assert_op_sequence_stable_names_the_drifting_index():
    with pytest.raises(AnePrecisionError, match="at index 1: enumerated 'b'"):
        assert_op_sequence_stable(("a", "b"), ("a", "c"))


# ------------------------------------------------------------ constructions


def test_split_k_zero_is_the_all_fp16_endpoint():
    names = compute_op_names(RECORDS)
    assert split_fp16_names(names, 0) == frozenset(names)


def test_split_k_all_is_the_all_fp32_endpoint():
    names = compute_op_names(RECORDS)
    assert split_fp16_names(names, len(names)) == frozenset()


def test_split_takes_the_prefix_and_leaves_the_tail_fp32():
    names = compute_op_names(RECORDS)
    assert split_fp16_names(names, 2) == frozenset({"conv_0", "bn_0", "relu_0"})


def test_split_refuses_an_out_of_range_k():
    with pytest.raises(AnePrecisionError, match=r"outside \[0, 5\]"):
        split_fp16_names(compute_op_names(RECORDS), 6)


def test_group_ranges_cover_exactly_and_front_load_the_remainder():
    ranges = group_ranges(10, 3)
    assert ranges == ((0, 4), (4, 7), (7, 10))
    assert sum(hi - lo for lo, hi in ranges) == 10


def test_group_ranges_endpoints():
    assert group_ranges(5, 1) == ((0, 5),)
    assert group_ranges(3, 3) == ((0, 1), (1, 2), (2, 3))


def test_group_ranges_refuses_more_groups_than_ops():
    with pytest.raises(AnePrecisionError, match=r"groups must be in \[1, 3\]"):
        group_ranges(3, 4)


def test_selective_fp32_holds_out_exactly_the_named_ordinals():
    names = compute_op_names(RECORDS)
    fp16 = selective_fp32_names(names, [0, 4])
    assert fp16 == frozenset({"bn_0", "relu_0", "conv_1"})


def test_selective_fp32_refuses_an_out_of_range_ordinal():
    with pytest.raises(AnePrecisionError, match="outside"):
        selective_fp32_names(compute_op_names(RECORDS), [99])


def test_selector_returns_true_only_for_wanted_and_records_every_op():
    class Op:
        def __init__(self, op_type, name):
            self.op_type = op_type
            self.name = name

    selector = selector_from_names({"conv_0"})
    ops = [Op(t, n) for t, n in RECORDS]
    verdicts = [selector(op) for op in ops]
    assert verdicts == [False, True, False, False, False, False, False]
    assert selector.observed == RECORDS
    assert compute_op_names(selector.observed) == ("conv_0", "bn_0", "relu_0", "conv_1", "head")


def test_split_backend_names_are_the_registered_ones():
    for k in (1, 2, 4, 8, 16, 32, 64):
        assert split_backend_name(k) in SPLIT_BACKENDS
    with pytest.raises(AnePrecisionError):
        split_backend_name(-1)


# ----------------------------------------------------------------- verdicts


def test_seg_flip_verdict_reports_bit_exact_and_the_bar_multiple():
    row = seg_flip_verdict(0, 1000, 3.3e-5)
    assert row["bit_exact_argmax"] is True
    assert row["passes_authority_bar"] is True
    assert row["multiple_of_bar"] == 0.0


def test_seg_flip_verdict_reports_over_bar_with_per_pair_stats():
    per_pair = [0.0, 1e-4, 2e-4, 3e-4]
    row = seg_flip_verdict(60, 1_000_000, 3.3e-5, per_pair)
    assert row["flip_rate"] == pytest.approx(6e-5)
    assert row["passes_authority_bar"] is False
    assert row["multiple_of_bar"] == pytest.approx(6e-5 / 3.3e-5)
    assert row["pairs_with_any_flip"] == 3
    assert row["per_pair_max"] == pytest.approx(3e-4)


def test_seg_flip_verdict_refuses_an_empty_denominator():
    with pytest.raises(AnePrecisionError, match="total_px must be positive"):
        seg_flip_verdict(1, 0, 3.3e-5)


def test_pose_drift_verdict_passes_both_bars_when_the_backend_is_exact():
    ref = np.array([[31.0, 0.02, 0.01, 0.003, 0.004, 0.012]])
    row = pose_drift_verdict(ref, ref.copy())
    assert row["self_mse_median"] == 0.0
    assert row["passes_per_dim_tolerance"] is True
    assert row["readable_against_d_pose"] is True
    assert row["per_dim_tolerance"] == pytest.approx(math.sqrt(POSE_D_POSE_T4_EXACT))


def test_pose_drift_verdict_attributes_the_damage_to_the_large_magnitude_dim():
    ref = np.tile(np.array([[31.0, 0.02, 0.01, 0.003, 0.004, 0.012]]), (4, 1))
    got = ref.copy()
    got[:, 0] += 0.26  # ane1's measured fp16 drift on dim 0
    row = pose_drift_verdict(ref, got)
    assert row["per_dim_share_of_mse"][0] > 0.999
    assert row["passes_per_dim_tolerance"] is False
    assert row["max_dim_multiple_of_per_dim_tolerance"] > 90
    assert row["readable_against_d_pose"] is False


def test_pose_drift_verdict_reads_a_tolerable_backend_as_readable():
    ref = np.tile(np.array([[31.0, 0.02, 0.01, 0.003, 0.004, 0.012]]), (4, 1))
    got = ref.copy()
    got[:, 0] += 1e-5
    row = pose_drift_verdict(ref, got)
    assert row["passes_per_dim_tolerance"] is True
    assert row["readable_against_d_pose"] is True


def test_pose_drift_verdict_refuses_a_shape_mismatch():
    with pytest.raises(AnePrecisionError, match="shape mismatch"):
        pose_drift_verdict(np.zeros((2, 6)), np.zeros((3, 6)))


def test_pose_per_dim_tolerance_is_sqrt_of_the_exact_d_pose():
    assert pytest.approx(2.7875e-3, rel=1e-3) == POSE_PER_DIM_TOLERANCE


# --------------------------------------------------------- hybrid geometry


def test_margin_band_mask_selects_the_low_margin_pixels():
    margin = np.array([[0.1, 0.5], [0.9, 0.05]])
    assert margin_band_mask(margin, 0.4).tolist() == [[True, False], [False, True]]


def test_dilate_bool_with_zero_halo_is_the_identity():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    assert np.array_equal(dilate_bool(mask, 0), mask)


def test_dilate_bool_grows_a_single_pixel_into_a_square_and_clips_at_the_edge():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    grown = dilate_bool(mask, 1)
    assert grown[:2, :2].all()
    assert grown.sum() == 4  # clipped at both edges, not 9


def test_dilate_bool_on_an_empty_mask_stays_empty():
    assert not dilate_bool(np.zeros((3, 3), dtype=bool), 2).any()


def test_dilate_bool_refuses_a_non_2d_mask():
    with pytest.raises(AnePrecisionError, match="2-D mask"):
        dilate_bool(np.zeros((2, 2, 2), dtype=bool), 1)


def test_occupied_tiles_counts_only_lit_tiles():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[3, 3] = True
    assert occupied_tiles(mask, 2) == (2, 4)
    assert occupied_tiles(np.zeros((4, 4), dtype=bool), 2) == (0, 4)


def test_crop_boxes_are_tile_aligned_expanded_by_halo_and_clipped():
    mask = np.zeros((8, 8), dtype=bool)
    mask[5, 5] = True
    boxes = crop_boxes(mask, 4, 2)
    assert boxes == ((2, 8, 2, 8),)


def test_crop_boxes_returns_nothing_for_an_empty_band():
    assert crop_boxes(np.zeros((8, 8), dtype=bool), 4, 2) == ()


def test_crop_pixel_fraction_counts_overlapping_halos_twice():
    boxes = ((0, 4, 0, 4), (0, 4, 0, 4))
    assert crop_pixel_fraction(boxes, 4, 4) == pytest.approx(2.0)


def test_crop_pixel_fraction_refuses_a_bad_frame_shape():
    with pytest.raises(AnePrecisionError, match="bad frame shape"):
        crop_pixel_fraction(((0, 1, 0, 1),), 0, 4)


def test_hybrid_speedup_passes_and_fails_the_three_times_bar():
    fast = hybrid_speedup(ane_s=0.005, recompute_s=0.02, reference_s=0.31)
    assert fast["speedup"] == pytest.approx(12.4)
    assert fast["passes_speed_bar"] is True
    slow = hybrid_speedup(ane_s=0.005, recompute_s=1.09, reference_s=0.255)
    assert slow["passes_speed_bar"] is False
    assert slow["recompute_share_of_hybrid"] > 0.99


def test_hybrid_speedup_refuses_negative_legs():
    with pytest.raises(AnePrecisionError, match="recompute_s must be >= 0"):
        hybrid_speedup(ane_s=0.1, recompute_s=-1.0, reference_s=0.3)


def test_crop_boxes_with_cores_pairs_each_tile_with_its_expanded_box():
    mask = np.zeros((8, 8), dtype=bool)
    mask[5, 5] = True
    pairs = crop_boxes_with_cores(mask, 4, 2)
    assert pairs == (((4, 8, 4, 8), (2, 8, 2, 8)),)


def test_crop_boxes_with_cores_cores_tile_the_plane_without_overlap():
    mask = np.ones((8, 8), dtype=bool)
    pairs = crop_boxes_with_cores(mask, 4, 3)
    cores = [core for core, _ in pairs]
    seen = np.zeros((8, 8), dtype=int)
    for y0, y1, x0, x1 in cores:
        seen[y0:y1, x0:x1] += 1
    assert seen.max() == 1 and seen.min() == 1


def test_crop_boxes_with_cores_expanded_boxes_contain_their_cores():
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 15] = True
    mask[15, 0] = True
    for (cy0, cy1, cx0, cx1), (ey0, ey1, ex0, ex1) in crop_boxes_with_cores(mask, 4, 5):
        assert ey0 <= cy0 and ey1 >= cy1 and ex0 <= cx0 and ex1 >= cx1


def test_crop_boxes_with_cores_refuses_a_bad_tile():
    with pytest.raises(AnePrecisionError, match="tile must be positive"):
        crop_boxes_with_cores(np.ones((4, 4), dtype=bool), 0, 1)


def test_fixed_crop_boxes_are_all_the_same_size_and_in_bounds():
    mask = np.zeros((384, 512), dtype=bool)
    mask[0, 0] = True
    mask[383, 511] = True
    mask[200, 250] = True
    pairs = fixed_crop_boxes(mask, 64, 32)
    assert len(pairs) == 3
    for (cy0, cy1, cx0, cx1), (by0, by1, bx0, bx1) in pairs:
        assert (by1 - by0, bx1 - bx0) == (128, 128)
        assert by0 >= 0 and by1 <= 384 and bx0 >= 0 and bx1 <= 512
        assert by0 <= cy0 and by1 >= cy1 and bx0 <= cx0 and bx1 >= cx1


def test_fixed_crop_boxes_shift_inward_at_the_frame_edge_rather_than_clipping():
    mask = np.zeros((128, 128), dtype=bool)
    mask[0, 0] = True
    ((core, box),) = fixed_crop_boxes(mask, 64, 32)
    assert core == (0, 64, 0, 64)
    assert box == (0, 128, 0, 128)


def test_fixed_crop_boxes_refuse_a_box_larger_than_the_frame():
    with pytest.raises(AnePrecisionError, match="does not fit"):
        fixed_crop_boxes(np.ones((64, 64), dtype=bool), 64, 32)
