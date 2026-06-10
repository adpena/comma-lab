# SPDX-License-Identifier: MIT
"""Behavioral tests for the evaluator invisibility basis (task #47).

The CERTIFICATION test is the heart: a perturbation IN tier-1, pushed through the
REAL scorer preprocessing (``F.interpolate`` exactly as ``upstream/modules.py``
calls it), produces a BIT-IDENTICAL scorer input (residual == 0.0); an
out-of-basis perturbation differs; the uint8 clipping boundary is honored.

A FAKE basis would (a) claim invisibility a perturbation does not have, or (b)
let the closed-form matrix drift from ``F.interpolate``.  Both are caught here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tac.optimization.evaluator_invisibility_basis import (
    CAMERA_H,
    CAMERA_W,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
    EvaluatorInvisibilityBasis,
    EvaluatorInvisibilityBasisError,
    Frame0SegNetCorollary,
    Tier2MeasuredLowSensitivity,
    _resize_1d_matrix,
    build_evaluator_invisibility_basis,
    derive_resize_kernel,
    derive_tier1_resize_null_space,
)

try:
    from tac.substrates._shared.constants_provenance_manifest import MeasurementScope
    HAVE_SCOPE = True
except Exception:  # pragma: no cover
    HAVE_SCOPE = False


def _resize2d(x: torch.Tensor) -> torch.Tensor:
    """The contest scorer's shared first preprocessing op (upstream/modules.py)."""
    return F.interpolate(
        x, size=(SCORER_INPUT_H, SCORER_INPUT_W), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# 1. The closed-form derivation == F.interpolate (DERIVATION FIDELITY).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_in,n_out",
    [(CAMERA_H, SCORER_INPUT_H), (CAMERA_W, SCORER_INPUT_W), (16, 7), (100, 37)],
)
def test_closed_form_matrix_matches_f_interpolate(n_in, n_out):
    """The derived 1D matrix reproduces F.interpolate to fp64 roundoff — proving
    the null space is computed against the REAL operator, not an approximation."""
    M = np.ascontiguousarray(_resize_1d_matrix(n_in, n_out))
    x = torch.rand(1, 1, n_in, 1, dtype=torch.float64)
    y_torch = F.interpolate(x, size=(n_out, 1), mode="bilinear", align_corners=False)
    y_torch = y_torch[0, 0, :, 0].numpy()
    xv = np.ascontiguousarray(x[0, 0, :, 0].numpy())
    y_mat = M.dot(xv)
    assert np.abs(y_torch - y_mat).max() < 1e-12


def test_resize_1d_rejects_upsampling():
    with pytest.raises(EvaluatorInvisibilityBasisError):
        _resize_1d_matrix(7, 16)


def test_derive_resize_kernel_rejects_bad_axis():
    with pytest.raises(EvaluatorInvisibilityBasisError):
        derive_resize_kernel("z", 100, 50)


# ---------------------------------------------------------------------------
# 2. TIER-1 CERTIFICATION — the heart: in-basis -> bit-identical scorer input.
# ---------------------------------------------------------------------------
def test_tier1_certification_in_basis_is_bit_identical():
    """255-amplitude perturbation at EVERY zero-weight pixel (all 3 channels) ->
    the resized scorer input is BIT-IDENTICAL (residual == 0.0 exactly).

    This is amplitude-unlimited certified invisibility: the perturbation is at
    the uint8 maximum yet produces zero change at the scorer input."""
    t1 = derive_tier1_resize_null_space()
    torch.manual_seed(0xC0FFEE)
    base = torch.rand(1, 3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 255.0
    y0 = _resize2d(base)
    mask = t1.zero_weight_pixel_mask()
    rr, cc = np.where(mask)
    assert len(rr) > 0
    delta = torch.zeros_like(base)
    for ch in range(3):
        delta[0, ch, rr, cc] = 255.0
    y1 = _resize2d(base + delta)
    residual = (y1 - y0).abs().max().item()
    assert residual == 0.0  # EXACTLY bit-identical, not merely small.


def test_tier1_certification_negative_amplitude_also_invisible():
    """The invisibility is amplitude-unlimited: a -255 perturbation is equally
    invisible (the null direction has zero weight regardless of sign/magnitude)."""
    t1 = derive_tier1_resize_null_space()
    torch.manual_seed(7)
    base = torch.rand(1, 3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 255.0
    y0 = _resize2d(base)
    mask = t1.zero_weight_pixel_mask()
    rr, cc = np.where(mask)
    delta = torch.zeros_like(base)
    delta[0, 0, rr, cc] = -255.0
    delta[0, 1, rr, cc] = 1234.5  # absurd over-range amplitude
    y1 = _resize2d(base + delta)
    assert (y1 - y0).abs().max().item() == 0.0


def test_out_of_basis_perturbation_changes_scorer_input():
    """A perturbation at a NON-zero-weight pixel DOES change the scorer input —
    the basis discriminates (a FAKE basis would call everything invisible)."""
    t1 = derive_tier1_resize_null_space()
    torch.manual_seed(11)
    base = torch.rand(1, 3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 255.0
    y0 = _resize2d(base)
    zr, zc = set(t1.zero_weight_rows), set(t1.zero_weight_cols)
    nr = next(r for r in range(CAMERA_H) if r not in zr)
    nc = next(c for c in range(CAMERA_W) if c not in zc)
    delta = torch.zeros_like(base)
    delta[0, 0, nr, nc] = 255.0
    y1 = _resize2d(base + delta)
    assert (y1 - y0).abs().max().item() > 0.0


def test_single_zero_weight_pixel_invisible_each():
    """Every individual zero-weight pixel is invisible on its own (not only the
    union) — spot-check a sample so the basis is per-pixel certified."""
    t1 = derive_tier1_resize_null_space()
    torch.manual_seed(3)
    base = torch.rand(1, 1, CAMERA_H, CAMERA_W, dtype=torch.float64) * 255.0
    y0 = _resize2d(base.repeat(1, 3, 1, 1))[:, :1]
    sample_rows = list(t1.zero_weight_rows)[:5]
    sample_cols = list(t1.zero_weight_cols)[:5]
    for r in sample_rows:
        delta = torch.zeros_like(base)
        delta[0, 0, r, :] = 200.0
        y1 = _resize2d((base + delta).repeat(1, 3, 1, 1))[:, :1]
        assert (y1 - y0).abs().max().item() == 0.0
    for c in sample_cols:
        delta = torch.zeros_like(base)
        delta[0, 0, :, c] = 200.0
        y1 = _resize2d((base + delta).repeat(1, 3, 1, 1))[:, :1]
        assert (y1 - y0).abs().max().item() == 0.0


def test_clipping_boundary_honored_in_certification():
    """The certified invisibility holds when the perturbed value is uint8-clipped
    BEFORE resize (the real codec path clips to [0,255]).  A zero-weight pixel
    clipped to 255 still contributes zero weight -> still invisible."""
    t1 = derive_tier1_resize_null_space()
    torch.manual_seed(5)
    base = torch.rand(1, 3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 255.0
    y0 = _resize2d(base.clamp(0, 255))
    mask = t1.zero_weight_pixel_mask()
    rr, cc = np.where(mask)
    perturbed = base.clone()
    for ch in range(3):
        perturbed[0, ch, rr, cc] = 99999.0  # huge, will clip to 255
    perturbed = perturbed.clamp(0, 255)
    y1 = _resize2d(perturbed)
    assert (y1 - y0).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 3. Tier-1 derived numbers (exact, hardware-independent).
# ---------------------------------------------------------------------------
def test_tier1_derived_dimensions_are_canonical():
    t1 = derive_tier1_resize_null_space()
    assert t1.h_kernel.n_zero_weight == 106
    assert t1.w_kernel.n_zero_weight == 140
    assert t1.n_zero_weight_pixels_per_channel == 230_904
    # full-rank downsample: rank == scorer pixels
    assert t1.h_kernel.rank == SCORER_INPUT_H
    assert t1.w_kernel.rank == SCORER_INPUT_W
    assert t1.full_null_dim == CAMERA_H * CAMERA_W - SCORER_INPUT_H * SCORER_INPUT_W
    assert abs(t1.zero_weight_pixel_fraction - 0.226969) < 1e-5
    assert abs(t1.full_null_fraction - 0.80674) < 1e-4


def test_tier1_zero_weight_pixels_subset_of_full_null():
    """The axis-aligned zero-weight pixels (1a) are a strict subset of the full
    null space (1b)."""
    t1 = derive_tier1_resize_null_space()
    assert t1.n_zero_weight_pixels_per_channel < t1.full_null_dim


def test_is_pixel_invisible_query():
    t1 = derive_tier1_resize_null_space()
    r = t1.zero_weight_rows[0]
    assert t1.is_pixel_invisible(r, 5) is True
    c = t1.zero_weight_cols[0]
    assert t1.is_pixel_invisible(5, c) is True
    # a pixel whose row AND col are both non-zero-weight is NOT single-pixel inv.
    zr, zc = set(t1.zero_weight_rows), set(t1.zero_weight_cols)
    nr = next(r for r in range(CAMERA_H) if r not in zr)
    nc = next(c for c in range(CAMERA_W) if c not in zc)
    assert t1.is_pixel_invisible(nr, nc) is False


def test_is_pixel_invisible_bounds_check():
    t1 = derive_tier1_resize_null_space()
    with pytest.raises(EvaluatorInvisibilityBasisError):
        t1.is_pixel_invisible(-1, 0)
    with pytest.raises(EvaluatorInvisibilityBasisError):
        t1.is_pixel_invisible(0, CAMERA_W)


def test_zero_weight_mask_count_matches_derivation():
    t1 = derive_tier1_resize_null_space()
    mask = t1.zero_weight_pixel_mask()
    assert int(mask.sum()) == t1.n_zero_weight_pixels_per_channel


# ---------------------------------------------------------------------------
# 4. FRAME0 SegNet corollary (the trivial-by-construction tier-1 case).
# ---------------------------------------------------------------------------
def test_frame0_segnet_corollary_all_invisible():
    cor = Frame0SegNetCorollary(camera_pixels=CAMERA_H * CAMERA_W)
    assert cor.segnet_invisible_fraction == 1.0
    assert cor.segnet_invisible_directions == CAMERA_H * CAMERA_W * 3


def test_frame0_segnet_corollary_matches_upstream_slice_semantics():
    """SegNet reads x[:, -1, ...] (frame1); a frame0-only perturbation is exactly
    SegNet-invisible.  Verify against the real slice op."""
    torch.manual_seed(13)
    # (B, T=2, C=3, H, W) — small synthetic grid (the slice semantics are size
    # independent; this is a contract test, not a scorer-scale test).
    pair = torch.rand(1, 2, 3, 40, 60, dtype=torch.float64) * 255.0
    seg_in0 = pair[:, -1, ...]  # what SegNet.preprocess slices BEFORE resize
    pair2 = pair.clone()
    pair2[:, 0, ...] += 255.0  # perturb ALL of frame0
    seg_in1 = pair2[:, -1, ...]
    assert torch.equal(seg_in0, seg_in1)  # frame0 invisible to SegNet's input


# ---------------------------------------------------------------------------
# 5. The combined basis artifact + query API + JSONL round-trip.
# ---------------------------------------------------------------------------
def test_build_basis_tier1_present():
    basis = build_evaluator_invisibility_basis()
    assert basis.schema == "evaluator_invisibility_basis.v1"
    assert basis.tier1_resize.n_zero_weight_pixels_per_channel == 230_904
    assert basis.frame0_corollary.segnet_invisible_fraction == 1.0
    assert basis.tier1_free_byte_fraction_per_channel() > 0.22


def test_query_tier1_pixel_invisible_frame_roles():
    basis = build_evaluator_invisibility_basis()
    r = basis.tier1_resize.zero_weight_rows[0]
    assert basis.tier1_pixel_invisible("frame1", r, 5) is True
    assert basis.tier1_pixel_invisible("frame0", r, 5) is True
    assert basis.tier1_frame0_segnet_invisible() is True


def test_query_rejects_bad_frame_role_and_channel():
    basis = build_evaluator_invisibility_basis()
    with pytest.raises(EvaluatorInvisibilityBasisError):
        basis.tier1_pixel_invisible("frameX", 0, 0)
    with pytest.raises(EvaluatorInvisibilityBasisError):
        basis.tier1_pixel_invisible("frame1", 0, 0, channel=5)


def test_jsonl_round_trip_rederives_tier1():
    """JSONL persistence: tier-1 is RE-DERIVED from sizes on load (the certified
    basis is reproducible from sizes alone — the header summary is an audit echo).
    Round-trip preserves the derived dimensions exactly."""
    basis = build_evaluator_invisibility_basis(
        provenance={"subagent": "evaluator_null_space_compiler_20260609"}
    )
    lines = basis.to_jsonl_lines()
    assert json.loads(lines[0])["kind"] == "header"
    rebuilt = EvaluatorInvisibilityBasis.from_jsonl_lines(lines)
    assert (
        rebuilt.tier1_resize.n_zero_weight_pixels_per_channel
        == basis.tier1_resize.n_zero_weight_pixels_per_channel
    )
    assert rebuilt.tier1_resize.full_null_dim == basis.tier1_resize.full_null_dim
    assert rebuilt.provenance["subagent"] == "evaluator_null_space_compiler_20260609"


def test_from_jsonl_rejects_empty_and_non_header():
    with pytest.raises(EvaluatorInvisibilityBasisError):
        EvaluatorInvisibilityBasis.from_jsonl_lines([])
    with pytest.raises(EvaluatorInvisibilityBasisError):
        EvaluatorInvisibilityBasis.from_jsonl_lines([json.dumps({"kind": "x"})])


def test_header_marks_non_promotable():
    basis = build_evaluator_invisibility_basis()
    header = json.loads(basis.to_jsonl_lines()[0])
    assert header["evidence"]["score_claim"] is False
    assert header["evidence"]["promotable"] is False
    assert header["tier1_resize"]["evidence_grade"] == "mathematical-derivation"


# ---------------------------------------------------------------------------
# 6. TIER-2 measured rows — kept SEPARATE from tier 1; scoped (Catalog #385).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_SCOPE, reason="MeasurementScope unavailable")
def test_tier2_row_carries_measurement_scope():
    scope = MeasurementScope(
        pairs=600,
        frames=2,
        scorer_surfaces=("d_seg", "d_pose"),
        authority_tier="macos_cpu_advisory",
        artifact_path="/Volumes/VertigoDataTier/pact/atlas/cone_maps/pair_0.npz",
    )
    row = Tier2MeasuredLowSensitivity(
        pair_index=0,
        region_class=None,
        usable_budget_fraction=0.46,
        pose_null_fraction=0.80,
        pair_budget=129257.0,
        mean_radius_usable=1.5,
        pose_binds_fraction=0.73,
        fragile_fraction=0.014,
        cone_map_path="/Volumes/VertigoDataTier/pact/atlas/cone_maps/pair_0.npz",
        cone_map_sha256="deadbeef",
        measurement_scope=scope,
    )
    r = row.to_row()
    assert r["tier"] == 2
    assert r["evidence_grade"] == "[macOS-CPU advisory]"
    assert r["promotable"] is False
    assert r["measurement_scope_empty"] is False
    assert r["measurement_scope"]["authority_tier"] == "macos_cpu_advisory"


@pytest.mark.skipif(not HAVE_SCOPE, reason="MeasurementScope unavailable")
def test_tier2_empty_scope_flagged_fragile():
    scope = MeasurementScope()  # empty
    row = Tier2MeasuredLowSensitivity(
        pair_index=1, region_class=2, usable_budget_fraction=0.3,
        pose_null_fraction=0.5, pair_budget=10.0, mean_radius_usable=1.0,
        pose_binds_fraction=0.4, fragile_fraction=0.02,
        cone_map_path="/Volumes/VertigoDataTier/pact/x.npz",
        cone_map_sha256="ab", measurement_scope=scope,
    )
    assert row.measurement_scope_empty is True  # measured but no scope = fragile flag


def test_tier2_rejects_out_of_range_fraction():
    with pytest.raises(EvaluatorInvisibilityBasisError):
        Tier2MeasuredLowSensitivity(
            pair_index=0, region_class=None, usable_budget_fraction=1.5,
            pose_null_fraction=0.5, pair_budget=1.0, mean_radius_usable=1.0,
            pose_binds_fraction=0.4, fragile_fraction=0.02,
            cone_map_path="/Volumes/x.npz", cone_map_sha256="ab",
            measurement_scope=None,
        )


def test_tier2_rejects_tmp_cone_path():
    with pytest.raises(EvaluatorInvisibilityBasisError):
        Tier2MeasuredLowSensitivity(
            pair_index=0, region_class=None, usable_budget_fraction=0.3,
            pose_null_fraction=0.5, pair_budget=1.0, mean_radius_usable=1.0,
            pose_binds_fraction=0.4, fragile_fraction=0.02,
            cone_map_path="/tmp/x.npz", cone_map_sha256="ab",
            measurement_scope=None,
        )


@pytest.mark.skipif(not HAVE_SCOPE, reason="MeasurementScope unavailable")
def test_basis_keeps_tiers_separate_in_jsonl():
    """Tier 1 lives in the header; tier 2 rows are separate lines tagged tier=2.
    A consumer can never read a tier-2 measured budget as a tier-1 certified zero."""
    scope = MeasurementScope(pairs=8, authority_tier="macos_cpu_advisory")
    row = Tier2MeasuredLowSensitivity(
        pair_index=5, region_class=None, usable_budget_fraction=0.4,
        pose_null_fraction=0.6, pair_budget=100.0, mean_radius_usable=1.2,
        pose_binds_fraction=0.5, fragile_fraction=0.01,
        cone_map_path="/Volumes/VertigoDataTier/pact/p5.npz", cone_map_sha256="cd",
        measurement_scope=scope,
    )
    basis = build_evaluator_invisibility_basis(tier2_rows=[row])
    lines = basis.to_jsonl_lines()
    header = json.loads(lines[0])
    assert header["kind"] == "header"
    assert header["tier1_resize"]["tier"] == 1
    body = [json.loads(ln) for ln in lines[1:]]
    assert all(b["kind"] == "tier2_row" and b["tier"] == 2 for b in body)
    rebuilt = EvaluatorInvisibilityBasis.from_jsonl_lines(lines)
    assert len(rebuilt.tier2_rows) == 1
    assert rebuilt.tier2_by_pair(5)[0].usable_budget_fraction == 0.4


# ---------------------------------------------------------------------------
# 7. REAL-SCORER NO-FAKE proof: tier-1 invisibility survives the FULL
#    upstream preprocessing of BOTH heads (resize for SegNet; resize+yuv6 for
#    PoseNet), bit-identical, on a real-scale frame pair.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tier1_invisible_through_full_upstream_preprocess_both_heads():
    """The certification at the REAL preprocessing surface: perturb frame1 (and
    frame0) at the zero-weight pixels and confirm BOTH SegNet's resized input AND
    PoseNet's yuv6 input are bit-identical.  This is the actual contract: tier-1
    is invisible to the EXACT operators upstream/modules.py applies."""
    import sys
    from pathlib import Path

    up = Path(__file__).resolve().parents[3] / "upstream"
    if str(up) not in sys.path:
        sys.path.insert(0, str(up))
    from frame_utils import rgb_to_yuv6, segnet_model_input_size  # type: ignore

    t1 = derive_tier1_resize_null_space()
    mask = t1.zero_weight_pixel_mask()
    rr, cc = np.where(mask)
    torch.manual_seed(0xBA51C)
    # (C, H, W) camera frames, mid-range to exercise yuv6 too.
    f0 = torch.rand(3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 200 + 20
    f1 = torch.rand(3, CAMERA_H, CAMERA_W, dtype=torch.float64) * 200 + 20

    def seg_preprocess(frame_chw):
        x = frame_chw.unsqueeze(0)
        return F.interpolate(
            x, size=(segnet_model_input_size[1], segnet_model_input_size[0]),
            mode="bilinear",
        )

    def pose_preprocess(frame_chw):
        x = frame_chw.unsqueeze(0)
        x = F.interpolate(
            x, size=(segnet_model_input_size[1], segnet_model_input_size[0]),
            mode="bilinear",
        )
        return rgb_to_yuv6(x)

    seg0 = seg_preprocess(f1)  # SegNet sees frame1 only
    pose0_f0 = pose_preprocess(f0)
    pose0_f1 = pose_preprocess(f1)

    # Perturb frame1 at zero-weight pixels (huge amplitude) + frame0 anywhere.
    f1p = f1.clone()
    for ch in range(3):
        f1p[ch, rr, cc] = 255.0
    f0p = f0.clone()
    f0p[0] += 255.0  # frame0 arbitrary perturbation (SegNet-invisible)
    for ch in range(3):
        f0p[ch, rr, cc] = 0.0  # frame0 zero-weight pixels -> PoseNet-invisible too

    seg1 = seg_preprocess(f1p)
    pose1_f0 = pose_preprocess(f0p)
    pose1_f1 = pose_preprocess(f1p)

    # SegNet input bit-identical (frame1 perturbed only at zero-weight pixels).
    assert (seg1 - seg0).abs().max().item() == 0.0
    # PoseNet frame1 input bit-identical (zero-weight pixels are resize-null,
    # and yuv6 is a fixed function of the resized input).
    assert (pose1_f1 - pose0_f1).abs().max().item() == 0.0
    # Conversely: the +255 on frame0 ch0 (NOT at zero-weight) DOES change
    # PoseNet's frame0 input — proving frame0 is SegNet-free but NOT PoseNet-free
    # outside the resize zero-weight set (the corollary's exact boundary).
    assert (pose1_f0 - pose0_f0).abs().max().item() > 0.0
    # PoseNet frame0 input bit-identical at the zero-weight pixels:
    f0_zero_only = f0.clone()
    for ch in range(3):
        f0_zero_only[ch, rr, cc] = 255.0
    pose1_f0_zero = pose_preprocess(f0_zero_only)
    assert (pose1_f0_zero - pose0_f0).abs().max().item() == 0.0
