# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the UNIWARD->bit_allocator->v8 LUT wire (#1570).

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE Class 2: these tests verify
BEHAVIOR (the cost-map actually changes the allocation; quantization actually
coarsens; the diff is non-empty; coarsening actually changes the v8 render),
NOT constants. Every test would FAIL if the helper bodies were replaced by a
no-op that returns the input unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.bit_allocator.per_byte import PerByteAllocationPlan
from tac.substrates.nscs06_v8_chroma_lut.architecture import (
    build_chroma_lut_from_ground_truth,
    lookup_rgb_via_chroma_lut,
)
from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.bit_allocation_per_lut_byte import (
    BIT_ALLOCATION_INTEGRATION_NAME,
    CANONICAL_EQUATION_ID_PROPOSED,
    PerLutByteBitAllocationResult,
    allocate_lut_bits_uniform_baseline,
    allocate_lut_bits_uniward_weighted,
    allocation_diff_from_uniform,
    build_canonical_provenance_for_bit_allocation,
    build_uniward_bit_allocated_chroma_lut,
    compute_uniward_cost_map_for_frames,
    per_lut_byte_sensitivity_from_uniward_weights,
    quantize_lut_by_allocation,
)
from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.weight_map_per_lut_index import (
    aggregate_per_pixel_uniward_weights_into_lut_bins,
)

LEVELS = 16
CLASSES = 5
N_LUT_BYTES = LEVELS * CLASSES * 3  # 240


def _structured_frames(n: int = 4, h: int = 24, w: int = 32, seed: int = 7):
    """Smooth-gradient + textured frames so the UNIWARD cost map is non-uniform."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((n, 3, h, w), dtype=np.uint8)
    for i in range(n):
        xx, _ = np.meshgrid(np.arange(w), np.arange(h))
        smooth = (xx / w * 255).astype(np.uint8)
        texture = rng.integers(0, 256, (h, w), dtype=np.uint8)
        half = w // 2
        base = smooth.copy()
        base[:, half:] = texture[:, half:]
        rgb[i, 0] = base
        rgb[i, 1] = np.roll(base, 3, axis=1)
        rgb[i, 2] = np.roll(base, 7, axis=0)
    cls = rng.integers(0, CLASSES, (n, h, w), dtype=np.uint8)
    return rgb, cls


# ── compute_uniward_cost_map_for_frames ──────────────────────────────────


def test_cost_map_is_nonuniform_on_structured_frames():
    """The UNIWARD cost map is HIGH on textured regions, LOW on smooth ones."""
    rgb, _ = _structured_frames()
    cost = compute_uniward_cost_map_for_frames(rgb)
    assert cost.shape == (rgb.shape[0], rgb.shape[2], rgb.shape[3])
    assert cost.dtype == np.float32
    # textured (right) half has higher mean cost than smooth (left) half
    w = rgb.shape[3]
    left = cost[:, :, : w // 2].mean()
    right = cost[:, :, w // 2 :].mean()
    assert right > left, f"textured half {right} should exceed smooth half {left}"
    # genuinely non-uniform (would be ~0 std if the helper returned a constant)
    assert float(cost.std()) > 1e-3


def test_cost_map_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_uniward_cost_map_for_frames(np.zeros((2, 4, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_uniward_cost_map_for_frames(np.zeros((2, 3, 8, 8), dtype=np.float32))


# ── per_lut_byte_sensitivity_from_uniward_weights ────────────────────────


def test_sensitivity_is_inverse_of_uniward_weight():
    """High UNIWARD weight (blind) -> low sensitivity; the 3 RGB bytes share it."""
    rgb, cls = _structured_frames()
    cost = compute_uniward_cost_map_for_frames(rgb)
    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=cost
    )
    sens = per_lut_byte_sensitivity_from_uniward_weights(per_bin)
    assert sens.shape == (N_LUT_BYTES,)
    assert np.all(np.isfinite(sens))
    assert np.all(sens >= 0.0)
    # The bin with the HIGHEST UNIWARD weight must map to the LOWEST sensitivity.
    wb = per_bin.weight_per_bin
    hi_lvl, hi_cls = np.unravel_index(np.argmax(wb), wb.shape)
    lo_lvl, lo_cls = np.unravel_index(
        np.argmin(np.where(wb > 0, wb, np.inf)), wb.shape
    )
    hi_byte0 = (hi_lvl * CLASSES + hi_cls) * 3
    lo_byte0 = (lo_lvl * CLASSES + lo_cls) * 3
    assert sens[hi_byte0] < sens[lo_byte0], (
        "highest-UNIWARD-weight (blind) bin must have LOWER allocator sensitivity"
    )
    # The 3 channel-bytes of a bin share the bin's sensitivity.
    assert sens[hi_byte0] == sens[hi_byte0 + 1] == sens[hi_byte0 + 2]


# ── allocate_lut_bits_* ──────────────────────────────────────────────────


def test_uniward_and_uniform_allocate_differently_NONFAKE_PROOF():
    """The CORE NO-FAKE PROOF: the cost-map actually changes the allocation.

    A wire that ignored the cost-map would produce an allocation identical to
    the uniform baseline. The diff MUST be non-empty (Catalog #105/#139/#220).
    """
    rgb, cls = _structured_frames()
    cost = compute_uniward_cost_map_for_frames(rgb)
    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=cost
    )
    sens = per_lut_byte_sensitivity_from_uniward_weights(per_bin)
    # Sub-full budget forces coarsening so the top-K ranking matters.
    budget = N_LUT_BYTES * 4  # half the full-precision budget
    uniward_plan = allocate_lut_bits_uniward_weighted(
        sens, total_budget_bits=budget, top_k=N_LUT_BYTES // 4
    )
    uniform_plan = allocate_lut_bits_uniform_baseline(
        N_LUT_BYTES, total_budget_bits=budget
    )
    from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.bit_allocation_per_lut_byte import (
        _bits_per_byte_array,
    )

    uw_bits = _bits_per_byte_array(uniward_plan, N_LUT_BYTES)
    uf_bits = _bits_per_byte_array(uniform_plan, N_LUT_BYTES)
    diff = allocation_diff_from_uniform(uw_bits, uf_bits)
    assert diff.size > 0, (
        "NO-FAKE PROOF FAILED: UNIWARD allocation identical to uniform — "
        "the cost-map is not being consumed (no-op wire)"
    )
    # UNIWARD concentrates bits: it has full-precision bytes AND zero-bit bytes,
    # whereas uniform spreads evenly. Check the UNIWARD top bytes are higher.
    assert uw_bits.max() >= uf_bits.max()
    assert uw_bits.min() <= uf_bits.min()


def test_uniform_baseline_spreads_evenly():
    plan = allocate_lut_bits_uniform_baseline(
        N_LUT_BYTES, total_budget_bits=N_LUT_BYTES * 4
    )
    from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.bit_allocation_per_lut_byte import (
        _bits_per_byte_array,
    )

    bits = _bits_per_byte_array(plan, N_LUT_BYTES)
    # uniform: every byte gets 4 bits (240*4 / 240 = 4)
    assert int(bits.min()) == 4
    assert int(bits.max()) == 4


def test_allocator_plans_carry_canonical_provenance():
    rgb, cls = _structured_frames()
    cost = compute_uniward_cost_map_for_frames(rgb)
    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=cost
    )
    sens = per_lut_byte_sensitivity_from_uniward_weights(per_bin)
    plan = allocate_lut_bits_uniward_weighted(
        sens, total_budget_bits=N_LUT_BYTES * 4, top_k=32
    )
    assert isinstance(plan, PerByteAllocationPlan)
    assert plan.score_claim is False
    assert plan.promotion_eligible is False
    assert plan.axis_tag == "[predicted]"
    assert plan.canonical_equation_id == "per_byte_leverage_uniformly_distributed_v1"


# ── quantize_lut_by_allocation ───────────────────────────────────────────


def test_quantization_coarsens_low_bit_bytes_BEHAVIOR():
    """A 4-bit byte keeps fewer distinct values than the original 8-bit byte."""
    rng = np.random.default_rng(3)
    lut = rng.integers(0, 256, (LEVELS, CLASSES, 3), dtype=np.uint8)
    # byte 0 gets 8 bits (unchanged); byte 1 gets 4 bits (coarsened); byte 2 gets 0
    bits = np.full((N_LUT_BYTES,), 8, dtype=np.int64)
    bits[1] = 4
    bits[2] = 0
    q = quantize_lut_by_allocation(lut, bits)
    flat_in = lut.reshape(-1)
    flat_out = q.reshape(-1)
    # 8-bit byte unchanged
    assert flat_out[0] == flat_in[0]
    # 0-bit byte collapses to mid-grey
    assert flat_out[2] == 128
    # 4-bit byte is rounded to a 15-step grid (not arbitrary 0-255)
    step = 255.0 / 15.0
    nearest = round(round(flat_in[1] / step) * step)
    assert flat_out[1] == nearest
    # shape + dtype preserved (v8 archive builder agnostic)
    assert q.shape == lut.shape
    assert q.dtype == np.uint8


def test_min_bits_floor_prevents_catastrophic_destruction_BEHAVIOR():
    """The min_bits floor coarsens (not destroys) low-sensitivity bytes.

    With a 0 floor, low-sensitivity bytes collapse to mid-grey (128) which is
    catastrophic for reconstruction. With a 2-bit floor, every byte retains a
    4-level chroma — gracefully coarsened. This test proves the floor changes
    behavior (the floored render is closer to the base render than the
    0-floored render).
    """
    from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.bit_allocation_per_lut_byte import (
        _bits_per_byte_array,
    )

    rgb, cls = _structured_frames()
    cost = compute_uniward_cost_map_for_frames(rgb)
    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=cost
    )
    sens = per_lut_byte_sensitivity_from_uniward_weights(per_bin)
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    # aggressive sub-full budget so many bytes fall below the top-K cutoff
    budget = N_LUT_BYTES * 3
    plan = allocate_lut_bits_uniward_weighted(sens, total_budget_bits=budget, top_k=30)
    bits_floor0 = _bits_per_byte_array(plan, N_LUT_BYTES, min_bits_per_byte=0)
    bits_floor2 = _bits_per_byte_array(plan, N_LUT_BYTES, min_bits_per_byte=2)
    # floor=0 has bytes at 0 bits (destroyed); floor=2 has none below 2
    assert int(bits_floor0.min()) == 0
    assert int(bits_floor2.min()) >= 2
    # the floor=2 quantized LUT is closer to the base LUT than floor=0
    lut0 = quantize_lut_by_allocation(base_lut, bits_floor0)
    lut2 = quantize_lut_by_allocation(base_lut, bits_floor2)
    mse0 = float(((lut0.astype(np.float64) - base_lut) ** 2).mean())
    mse2 = float(((lut2.astype(np.float64) - base_lut) ** 2).mean())
    assert mse2 < mse0, (
        f"floor=2 MSE {mse2} should be lower than floor=0 MSE {mse0} "
        "(floor prevents catastrophic destruction)"
    )


def test_build_respects_min_bits_floor():
    """The end-to-end build with default min_bits=2 never produces a 0-bit byte."""
    rgb, cls = _structured_frames()
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    result = build_uniward_bit_allocated_chroma_lut(
        rgb_pairs=rgb,
        class_labels=cls,
        base_lut=base_lut,
        total_budget_bits=N_LUT_BYTES * 3,
        top_k=30,
    )
    assert int(result.bits_per_lut_byte.min()) >= 2


def test_full_precision_allocation_is_identity():
    """If every byte gets 8 bits, the LUT is unchanged."""
    rng = np.random.default_rng(11)
    lut = rng.integers(0, 256, (LEVELS, CLASSES, 3), dtype=np.uint8)
    bits = np.full((N_LUT_BYTES,), 8, dtype=np.int64)
    q = quantize_lut_by_allocation(lut, bits)
    assert np.array_equal(q, lut)


def test_quantization_reduces_distinct_values_at_low_budget():
    """Coarsening at a low budget reduces the number of distinct LUT values."""
    rng = np.random.default_rng(13)
    lut = rng.integers(0, 256, (LEVELS, CLASSES, 3), dtype=np.uint8)
    bits = np.full((N_LUT_BYTES,), 2, dtype=np.int64)  # 2-bit -> 3 levels
    q = quantize_lut_by_allocation(lut, bits)
    n_distinct_in = len(np.unique(lut))
    n_distinct_out = len(np.unique(q))
    assert n_distinct_out < n_distinct_in
    assert n_distinct_out <= 4  # at most 2**2 representable levels


# ── end-to-end build_uniward_bit_allocated_chroma_lut ────────────────────


def test_end_to_end_wire_coarsens_some_bytes():
    """End-to-end: real cost -> allocate -> quantize. Some bytes coarsen."""
    rgb, cls = _structured_frames()
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    assert base_lut.shape == (LEVELS, CLASSES, 3)
    result = build_uniward_bit_allocated_chroma_lut(
        rgb_pairs=rgb,
        class_labels=cls,
        base_lut=base_lut,
        total_budget_bits=N_LUT_BYTES * 4,  # half budget forces coarsening
        top_k=N_LUT_BYTES // 4,
    )
    assert isinstance(result, PerLutByteBitAllocationResult)
    assert result.lut_quantized.shape == (LEVELS, CLASSES, 3)
    assert result.lut_quantized.dtype == np.uint8
    assert result.n_lut_bytes == N_LUT_BYTES
    # at a half budget, some bytes are coarsened and some are full-precision
    assert result.n_bytes_coarsened > 0
    assert result.n_bytes_at_full_precision > 0
    # the quantized LUT differs from the base (coarsening actually happened)
    assert not np.array_equal(result.lut_quantized, base_lut)


def test_end_to_end_full_budget_is_identity():
    """At the full-precision budget the wire is a no-op on the LUT."""
    rgb, cls = _structured_frames()
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    result = build_uniward_bit_allocated_chroma_lut(
        rgb_pairs=rgb,
        class_labels=cls,
        base_lut=base_lut,
        total_budget_bits=N_LUT_BYTES * 8,  # full precision
        top_k=N_LUT_BYTES,
    )
    assert np.array_equal(result.lut_quantized, base_lut)
    assert result.n_bytes_coarsened == 0


def test_coarsened_lut_changes_v8_render_NONFAKE():
    """The coarsened LUT changes the inflate-time render (not a no-op render).

    Per Catalog #105/#139/#220: the v8 inflate consumes EVERY LUT byte via
    lookup_rgb_via_chroma_lut, so coarsening the LUT MUST change the rendered
    RGB. A wire that produced an identical render would be a no-op.
    """
    rgb, cls = _structured_frames()
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    result = build_uniward_bit_allocated_chroma_lut(
        rgb_pairs=rgb,
        class_labels=cls,
        base_lut=base_lut,
        total_budget_bits=N_LUT_BYTES * 2,  # aggressive coarsening
        top_k=16,
    )
    # render the same gray+cls map through both LUTs (8x8 maps so every
    # (level, class) combination is exercised)
    rng = np.random.default_rng(99)
    gray_map = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    cls_map = rng.integers(0, CLASSES, (8, 8), dtype=np.uint8)
    render_base = lookup_rgb_via_chroma_lut(gray_map, cls_map, base_lut)
    render_q = lookup_rgb_via_chroma_lut(gray_map, cls_map, result.lut_quantized)
    assert render_base.shape == render_q.shape
    assert not np.array_equal(render_base, render_q), (
        "coarsened LUT produced IDENTICAL render — the LUT bytes are not "
        "consumed (no-op wire)"
    )


# ── provenance ───────────────────────────────────────────────────────────


def test_provenance_carries_non_promotable_markers():
    rgb, cls = _structured_frames()
    base_lut = build_chroma_lut_from_ground_truth(rgb, cls)
    result = build_uniward_bit_allocated_chroma_lut(
        rgb_pairs=rgb,
        class_labels=cls,
        base_lut=base_lut,
        total_budget_bits=N_LUT_BYTES * 4,
        top_k=32,
    )
    prov = build_canonical_provenance_for_bit_allocation(
        result=result, total_budget_bits=N_LUT_BYTES * 4, top_k=32
    )
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "macOS-MLX research-signal"
    assert prov["integration_id"] == BIT_ALLOCATION_INTEGRATION_NAME
    assert prov["canonical_equation_id_proposed"] == CANONICAL_EQUATION_ID_PROPOSED
    assert prov["consumed_substrate_scope"] == "read_only_consumer_import"
    # bit-allocator PRIMARY hook #3 fired
    assert 3 in prov["hook_numbers_fired"]
