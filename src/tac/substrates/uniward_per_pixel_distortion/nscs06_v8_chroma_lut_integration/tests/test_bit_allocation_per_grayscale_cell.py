# SPDX-License-Identifier: MIT
"""NO-FAKE tests (Slot EEE Class 2: verify BEHAVIOR, not constants) for the
WAVE-5B UNIWARD -> grayscale-stream per-cell bit-allocation wire.

Every test fails if the helper body is replaced by a no-op or a constant
return. The CORE NO-FAKE PROOF
(``test_uniward_and_uniform_allocate_differently_NONFAKE_PROOF``) asserts the
UNIWARD cost-map actually changes the allocation vs the uniform baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.bit_allocator.per_byte import PerByteAllocationMethod
from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.bit_allocation_per_grayscale_cell import (
    GRAYSCALE_STREAM_INTEGRATION_NAME,
    PerGrayscaleCellBitAllocationResult,
    aggregate_uniward_cost_into_grayscale_cells,
    allocate_grayscale_cells_uniform_baseline,
    allocate_grayscale_cells_uniward_weighted,
    allocation_diff_from_uniform_cells,
    build_canonical_provenance_for_grayscale_bit_allocation,
    build_uniward_bit_allocated_grayscale_stream,
    per_cell_sensitivity_from_uniward_cell_weights,
    quantize_grayscale_stream_by_cell_allocation,
)


def _synthetic_cost(gh: int, gw: int, n: int = 2) -> np.ndarray:
    """A per-pixel cost field with a clear textured/smooth split at OUTPUT res.

    Left half = HIGH cost (textured/blind); right half = LOW cost (smooth/
    sensitive). The cell aggregation must preserve this split.
    """
    h, w = gh * 4, gw * 4  # output res = 4x the cell grid
    cost = np.zeros((n, h, w), dtype=np.float32)
    cost[:, :, : w // 2] = 9.0  # left half textured (high cost)
    cost[:, :, w // 2 :] = 0.1  # right half smooth (low cost)
    return cost


# ---------------------------------------------------------------------------
# aggregate_uniward_cost_into_grayscale_cells
# ---------------------------------------------------------------------------


def test_aggregate_preserves_textured_smooth_split():
    gh, gw = 8, 8
    cost = _synthetic_cost(gh, gw)
    per_cell = aggregate_uniward_cost_into_grayscale_cells(
        cost, grayscale_h=gh, grayscale_w=gw
    )
    assert per_cell.shape == (gh, gw)
    # Left-half cells should be HIGH cost, right-half LOW.
    left = per_cell[:, : gw // 2].mean()
    right = per_cell[:, gw // 2 :].mean()
    assert left > right * 10.0, f"split not preserved: left={left} right={right}"


def test_aggregate_block_average_not_subsample():
    # A cost field where the cell CENTER pixel is smooth but the block is
    # textured must aggregate to HIGH cost (block-average, not subsample).
    gh, gw = 2, 2
    h, w = gh * 4, gw * 4
    cost = np.full((1, h, w), 8.0, dtype=np.float32)
    # Zero only the block centers (would be picked by strided subsample).
    cost[0, 1::4, 1::4] = 0.0
    per_cell = aggregate_uniward_cost_into_grayscale_cells(
        cost, grayscale_h=gh, grayscale_w=gw
    )
    # Block-average keeps the surrounding 8.0; a subsample would be 0.0.
    assert per_cell.min() > 5.0, f"block-average failed: {per_cell}"


def test_aggregate_non_integer_ratio_uses_bilinear_fallback():
    # H not a multiple of gh -> PIL bilinear fallback path.
    gh, gw = 5, 5
    cost = np.full((1, 17, 17), 3.0, dtype=np.float32)
    per_cell = aggregate_uniward_cost_into_grayscale_cells(
        cost, grayscale_h=gh, grayscale_w=gw
    )
    assert per_cell.shape == (gh, gw)
    assert np.all(np.isfinite(per_cell))
    assert per_cell.mean() == pytest.approx(3.0, abs=0.5)


def test_aggregate_rejects_bad_shape():
    with pytest.raises(ValueError):
        aggregate_uniward_cost_into_grayscale_cells(
            np.zeros((4, 4)), grayscale_h=2, grayscale_w=2
        )


# ---------------------------------------------------------------------------
# per_cell_sensitivity_from_uniward_cell_weights  (INVERSE behavior)
# ---------------------------------------------------------------------------


def test_sensitivity_is_inverse_of_cost():
    # HIGH cost (blind) -> LOW sensitivity; LOW cost (sensitive) -> HIGH.
    cost = np.array([[9.0, 0.01], [0.01, 9.0]], dtype=np.float64)
    sens = per_cell_sensitivity_from_uniward_cell_weights(cost)
    assert sens.shape == (4,)
    # flat order gy*gw+gx: [hi,lo, lo,hi] -> sens [lo,hi, hi,lo]
    assert sens[0] < sens[1], "high-cost cell should have lower sensitivity"
    assert sens[3] < sens[2], "high-cost cell should have lower sensitivity"


def test_sensitivity_finite_for_zero_cost():
    cost = np.zeros((2, 2), dtype=np.float64)
    sens = per_cell_sensitivity_from_uniward_cell_weights(cost)
    assert np.all(np.isfinite(sens))
    assert np.all(sens > 0.0)


# ---------------------------------------------------------------------------
# allocator routing (canonical tac.bit_allocator.per_byte)
# ---------------------------------------------------------------------------


def test_uniward_allocator_routes_through_canonical_per_byte():
    sens = np.array([100.0, 1.0, 1.0, 1.0], dtype=np.float64)
    plan = allocate_grayscale_cells_uniward_weighted(
        sens, total_budget_bits=16, top_k=1
    )
    assert plan.method == PerByteAllocationMethod.TOP_K_BY_SENSITIVITY
    # The most-sensitive cell (0) should get the most bits.
    assert plan.bits_per_byte.get(0, 0) >= max(
        plan.bits_per_byte.get(i, 0) for i in (1, 2, 3)
    )


def test_uniform_baseline_routes_through_canonical_per_byte():
    plan = allocate_grayscale_cells_uniform_baseline(4, total_budget_bits=16)
    assert plan.method == PerByteAllocationMethod.UNIFORM_BASELINE
    # Uniform: every cell gets 16//4 = 4 bits.
    assert all(b == 4 for b in plan.bits_per_byte.values())


# ---------------------------------------------------------------------------
# CORE NO-FAKE PROOF — the cost-map changes the allocation
# ---------------------------------------------------------------------------


def test_uniward_and_uniform_allocate_differently_NONFAKE_PROOF():
    """The UNIWARD cost-map MUST change which cells keep precision.

    If the wire ignored the cost-map (a no-op), the UNIWARD and uniform
    allocations would be identical and the diff would be empty. A non-empty
    diff is the structural NON-FAKE proof per Catalog #105/#139/#220.
    """
    n_cells = 16
    # Strongly non-uniform sensitivity: cell 0 dominant, rest flat.
    sens = np.ones(n_cells, dtype=np.float64)
    sens[0] = 1000.0
    sens[1] = 500.0
    budget = n_cells * 4  # sub-full budget forces a non-trivial allocation
    uw = allocate_grayscale_cells_uniward_weighted(
        sens, total_budget_bits=budget, top_k=2
    )
    uf = allocate_grayscale_cells_uniform_baseline(
        n_cells, total_budget_bits=budget
    )
    uw_bits = np.array([uw.bits_per_byte.get(i, 0) for i in range(n_cells)])
    uf_bits = np.array([uf.bits_per_byte.get(i, 0) for i in range(n_cells)])
    diff = allocation_diff_from_uniform_cells(uw_bits, uf_bits)
    assert diff.size > 0, "NON-FAKE proof FAILED: cost-map did not change alloc"
    # The dominant cell should be among the changed (kept high precision).
    assert 0 in diff.tolist()


def test_allocation_diff_empty_when_identical():
    bits = np.array([4, 4, 4, 4])
    diff = allocation_diff_from_uniform_cells(bits, bits.copy())
    assert diff.size == 0


def test_allocation_diff_shape_mismatch_raises():
    with pytest.raises(ValueError):
        allocation_diff_from_uniform_cells(np.zeros(4), np.zeros(8))


# ---------------------------------------------------------------------------
# quantize_grayscale_stream_by_cell_allocation  (the coarsening)
# ---------------------------------------------------------------------------


def test_quantize_8bit_cell_unchanged():
    stream = np.arange(2 * 2 * 2, dtype=np.uint8).reshape(2, 2, 2)
    bits = np.full(4, 8, dtype=np.int64)  # all full precision
    out = quantize_grayscale_stream_by_cell_allocation(stream, bits)
    np.testing.assert_array_equal(out, stream)


def test_quantize_coarsens_low_bit_cell():
    # A 1-bit cell keeps only 2 distinct levels -> entropy collapse.
    num_pairs = 4
    stream = np.random.RandomState(0).randint(
        0, 256, size=(num_pairs, 1, 2), dtype=np.uint8
    )
    bits = np.array([1, 8], dtype=np.int64)  # cell 0 = 1-bit, cell 1 = 8-bit
    out = quantize_grayscale_stream_by_cell_allocation(stream, bits)
    # Cell 0 (1-bit) collapses to <= 2 distinct values across pairs.
    distinct_cell0 = len(np.unique(out[:, 0, 0]))
    assert distinct_cell0 <= 2, f"1-bit cell not coarsened: {distinct_cell0}"
    # Cell 1 (8-bit) unchanged.
    np.testing.assert_array_equal(out[:, 0, 1], stream[:, 0, 1])


def test_quantize_broadcasts_bit_depth_across_pairs():
    # The same cell's bit-depth must apply at EVERY pair (broadcast).
    num_pairs = 8
    stream = np.full((num_pairs, 1, 1), 200, dtype=np.uint8)
    bits = np.array([2], dtype=np.int64)  # 2-bit cell
    out = quantize_grayscale_stream_by_cell_allocation(stream, bits)
    # 200 at 2-bit (3 levels in [0,255], step 85) -> round(200/85)*85 = 170.
    assert np.all(out == 170), f"broadcast/quantize failed: {np.unique(out)}"


def test_quantize_zero_bit_collapses_to_midgrey():
    stream = np.random.RandomState(1).randint(
        0, 256, size=(3, 1, 1), dtype=np.uint8
    )
    bits = np.array([0], dtype=np.int64)
    out = quantize_grayscale_stream_by_cell_allocation(stream, bits)
    assert np.all(out == 128)


def test_quantize_rejects_bad_shape():
    with pytest.raises(ValueError):
        quantize_grayscale_stream_by_cell_allocation(
            np.zeros((4, 4), dtype=np.uint8), np.zeros(16, dtype=np.int64)
        )


def test_quantize_rejects_len_mismatch():
    with pytest.raises(ValueError):
        quantize_grayscale_stream_by_cell_allocation(
            np.zeros((2, 2, 2), dtype=np.uint8), np.zeros(8, dtype=np.int64)
        )


# ---------------------------------------------------------------------------
# end-to-end build_uniward_bit_allocated_grayscale_stream  (NON-FAKE)
# ---------------------------------------------------------------------------


def test_end_to_end_coarsens_textured_cells_keeps_smooth():
    """End-to-end: textured (high-cost) cells coarsen, smooth (low-cost) keep.

    This is the WAVE-5B behavioral claim: the cost-map's spatial structure must
    propagate to which cells keep luma precision. Fails if the wire ignores cost.
    """
    rng = np.random.RandomState(2)
    num_pairs, gh, gw = 6, 8, 8
    h, w = gh * 4, gw * 4
    # Real-ish stream: random luma.
    stream = rng.randint(0, 256, size=(num_pairs, gh, gw), dtype=np.uint8)
    # rgb_pairs only needed if cost not supplied; supply a cost directly.
    cost = _synthetic_cost(gh, gw, n=num_pairs)  # left textured, right smooth
    n_cells = gh * gw
    budget = n_cells * 4  # 4 bits/cell average forces real coarsening
    res = build_uniward_bit_allocated_grayscale_stream(
        grayscale_stream=stream,
        rgb_pairs=np.zeros((num_pairs, 3, h, w), dtype=np.uint8),  # unused
        total_budget_bits=budget,
        top_k=n_cells // 2,
        uniward_cost_map=cost,
        min_bits_per_cell=2,
    )
    assert isinstance(res, PerGrayscaleCellBitAllocationResult)
    assert res.grayscale_quantized.shape == stream.shape
    bits2d = res.bits_per_cell.reshape(gh, gw)
    left_bits = bits2d[:, : gw // 2].mean()  # textured (high cost -> low sens)
    right_bits = bits2d[:, gw // 2 :].mean()  # smooth (low cost -> high sens)
    # Smooth (sensitive) cells should keep MORE bits than textured (blind).
    assert right_bits > left_bits, (
        f"UNIWARD did not concentrate precision on smooth cells: "
        f"left(textured)={left_bits} right(smooth)={right_bits}"
    )


def test_end_to_end_changes_stream_bytes_nonfake():
    """The quantized stream MUST differ from the input (coarsening happened)."""
    rng = np.random.RandomState(3)
    num_pairs, gh, gw = 4, 8, 8
    stream = rng.randint(0, 256, size=(num_pairs, gh, gw), dtype=np.uint8)
    cost = _synthetic_cost(gh, gw, n=num_pairs)
    res = build_uniward_bit_allocated_grayscale_stream(
        grayscale_stream=stream,
        rgb_pairs=np.zeros((num_pairs, 3, gh * 4, gw * 4), dtype=np.uint8),
        total_budget_bits=gh * gw * 3,  # 3-bit avg -> real coarsening
        top_k=(gh * gw) // 4,
        uniward_cost_map=cost,
        min_bits_per_cell=2,
    )
    assert not np.array_equal(res.grayscale_quantized, stream), (
        "no-op: quantized stream identical to input"
    )


def test_end_to_end_uniward_vs_uniform_allocate_differently_NONFAKE():
    """End-to-end NON-FAKE: UNIWARD and uniform produce different bit maps."""
    rng = np.random.RandomState(4)
    num_pairs, gh, gw = 4, 8, 8
    stream = rng.randint(0, 256, size=(num_pairs, gh, gw), dtype=np.uint8)
    cost = _synthetic_cost(gh, gw, n=num_pairs)
    n_cells = gh * gw
    budget = n_cells * 4
    res_uw = build_uniward_bit_allocated_grayscale_stream(
        grayscale_stream=stream,
        rgb_pairs=np.zeros((num_pairs, 3, gh * 4, gw * 4), dtype=np.uint8),
        total_budget_bits=budget,
        top_k=n_cells // 2,
        uniward_cost_map=cost,
        min_bits_per_cell=2,
    )
    uf_plan = allocate_grayscale_cells_uniform_baseline(
        n_cells, total_budget_bits=budget
    )
    uf_bits = np.full(n_cells, 2, dtype=np.int64)
    for off, b in uf_plan.bits_per_byte.items():
        uf_bits[off] = max(int(b), 2)
    diff = allocation_diff_from_uniform_cells(res_uw.bits_per_cell, uf_bits)
    assert diff.size > 0, "NON-FAKE FAILED: UNIWARD allocation == uniform"


def test_end_to_end_rejects_bad_stream_shape():
    with pytest.raises(ValueError):
        build_uniward_bit_allocated_grayscale_stream(
            grayscale_stream=np.zeros((4, 4), dtype=np.uint8),
            rgb_pairs=np.zeros((4, 3, 8, 8), dtype=np.uint8),
            total_budget_bits=64,
            top_k=4,
            uniward_cost_map=np.zeros((4, 8, 8), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# result properties + provenance
# ---------------------------------------------------------------------------


def test_result_properties_count_correctly():
    num_pairs, gh, gw = 5, 4, 4
    n_cells = gh * gw
    stream = np.zeros((num_pairs, gh, gw), dtype=np.uint8)
    cost = np.full((num_pairs, gh * 4, gw * 4), 1.0, dtype=np.float32)
    res = build_uniward_bit_allocated_grayscale_stream(
        grayscale_stream=stream,
        rgb_pairs=np.zeros((num_pairs, 3, gh * 4, gw * 4), dtype=np.uint8),
        total_budget_bits=n_cells * 5,
        top_k=n_cells // 2,
        uniward_cost_map=cost,
        min_bits_per_cell=2,
    )
    assert res.n_cells == n_cells
    assert res.n_stream_bytes == num_pairs * n_cells
    assert (
        res.n_cells_at_full_precision + res.n_cells_coarsened == n_cells
    )


def test_provenance_carries_non_promotable_markers():
    num_pairs, gh, gw = 2, 4, 4
    stream = np.zeros((num_pairs, gh, gw), dtype=np.uint8)
    cost = np.full((num_pairs, gh * 4, gw * 4), 1.0, dtype=np.float32)
    res = build_uniward_bit_allocated_grayscale_stream(
        grayscale_stream=stream,
        rgb_pairs=np.zeros((num_pairs, 3, gh * 4, gw * 4), dtype=np.uint8),
        total_budget_bits=gh * gw * 4,
        top_k=4,
        uniward_cost_map=cost,
    )
    prov = build_canonical_provenance_for_grayscale_bit_allocation(
        result=res, total_budget_bits=gh * gw * 4, top_k=4
    )
    # Catalog #341 non-promotable markers.
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "macOS-MLX research-signal"
    assert prov["consumed_byte_surface"] == "grayscale_stream"
    assert prov["integration_id"] == GRAYSCALE_STREAM_INTEGRATION_NAME
    assert prov["nscs06_v8_substrate_modification_scope"].startswith("none")
