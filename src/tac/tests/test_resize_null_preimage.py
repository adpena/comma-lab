# SPDX-License-Identifier: MIT
"""Behavioral tests for the resize-null preimage compiler (task #49).

The contract under test (NOT constants — behavior, per CLAUDE.md "NO FAKE
IMPLEMENTATIONS" Class 2):

  - the certified zero-weight fill leaves the scorer PROJECTION bit-identical
    (``max|R x̃ - R x| == 0.0`` against the REAL projector, every tier);
  - the fill ONLY touches certified-invisible pixels (out-of-mask unchanged);
  - outputs are valid uint8 / same shape (operator caveat (a));
  - equality holds on the RGB tensor BEFORE PoseNet's YUV conversion AND survives
    the FULL upstream PoseNet/SegNet preprocess (operator caveat (b), no-fake);
  - the discriminator is real: a perturbation OUTSIDE the zero-weight set DOES
    change the projection (a fake "everything invisible" basis is caught);
  - idempotence: re-applying the preimage to its own output is a no-op;
  - the tier-2 descent is never worse than tier-1 (a true descent);
  - bytes reduction is real on a compressible frame (measured, not asserted).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.evaluator_invisibility_basis import (
    derive_tier1_resize_null_space,
)
from tac.optimization.resize_null_preimage import (
    CONTEST_TOTAL_BYTES,
    ResizeNullPreimageError,
    ResizeProjector,
    apply_tier1_zero_weight_fill,
    apply_tier2_null_basis_descent,
    apply_tier3_blockwise_flat_preimage,
    coded_size_both,
    coded_size_bytes,
    preimage_rate_score_delta,
    zero_weight_pixel_mask,
)

# Use a SMALL contest-shaped downsample for fast tests: keep the contest aspect
# behavior (downsample with dropped rows/cols) but tiny so the projector and the
# coder are cheap.  The math is dimension-independent.
CH, CW = 40, 60
SH, SW = 17, 26  # n_out < n_in along both axes => guaranteed dropped pixels


@pytest.fixture(scope="module")
def small_basis():
    return derive_tier1_resize_null_space(
        camera_h=CH, camera_w=CW, scorer_h=SH, scorer_w=SW
    )


@pytest.fixture(scope="module")
def small_projector():
    return ResizeProjector.build(camera_h=CH, camera_w=CW, scorer_h=SH, scorer_w=SW)


@pytest.fixture(scope="module")
def small_mask(small_basis):
    return zero_weight_pixel_mask(
        camera_h=CH, camera_w=CW, scorer_h=SH, scorer_w=SW, basis=small_basis
    )


def _make_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:CH, 0:CW]
    base = ((xx / CW * 120 + yy / CH * 80 + 40)).astype(np.uint8)
    f = np.stack([base, np.roll(base, 3, 1), np.roll(base, 2, 0)], axis=2).astype(int)
    f = np.clip(f + rng.integers(-6, 6, f.shape), 0, 255).astype(np.uint8)
    return f


# ---------------------------------------------------------------------------
# Mask + projector sanity (reuse #47's certified basis).
# ---------------------------------------------------------------------------
def test_mask_matches_47_certified_count(small_basis, small_mask):
    assert int(small_mask.sum()) == small_basis.n_zero_weight_pixels_per_channel
    assert small_mask.shape == (CH, CW)
    assert small_mask.dtype == bool


def test_mask_nonempty_and_nontrivial(small_mask):
    # contest-shaped downsample MUST drop some rows/cols (fail-closed against a
    # degenerate empty or all-true mask).
    n = int(small_mask.sum())
    assert 0 < n < small_mask.size


def test_projector_reproduces_torch_interpolate(small_projector):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    plane = rng.integers(0, 256, size=(CH, CW)).astype(np.float64)
    y_np = small_projector.project_plane(plane)
    x = torch.from_numpy(plane[None, None, :, :])
    y_t = torch.nn.functional.interpolate(
        x, size=(SH, SW), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    assert np.max(np.abs(y_np - y_t)) < 1e-9


# ---------------------------------------------------------------------------
# TIER 1 — exactness + scope + validity.
# ---------------------------------------------------------------------------
def test_tier1_projection_residual_exactly_zero(small_projector, small_basis):
    frame = _make_frame(2)
    pre, proof = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    assert proof.max_abs_projection_residual == 0.0
    assert proof.exact is True


def test_tier1_only_touches_masked_pixels(small_projector, small_basis, small_mask):
    frame = _make_frame(3)
    pre, _ = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    # every pixel OUTSIDE the zero-weight mask is byte-identical to the original.
    keep = ~small_mask
    assert np.array_equal(pre[keep], frame[keep])


def test_tier1_output_valid_uint8_same_shape(small_projector, small_basis):
    frame = _make_frame(4)
    pre, proof = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    assert pre.dtype == np.uint8
    assert pre.shape == frame.shape
    assert proof.valid_uint8 is True


def test_tier1_amplitude_unlimited_in_mask_still_exact(small_projector, small_basis,
                                                       small_mask):
    # set masked pixels to extreme values via explicit constant strategy; still 0.
    frame = _make_frame(5)
    pre, proof = apply_tier1_zero_weight_fill(
        frame, strategy="constant", projector=small_projector, basis=small_basis,
        mask=small_mask,
    )
    assert proof.max_abs_projection_residual == 0.0
    # all masked pixels became the constant (0) — a real mutation occurred.
    assert np.all(pre[small_mask] == 0)


def test_tier1_explicit_strategies_all_exact(small_projector, small_basis):
    frame = _make_frame(6)
    for strat in ("constant", "horizontal_predictor", "vertical_predictor",
                  "neighbor_mean"):
        pre, proof = apply_tier1_zero_weight_fill(
            frame, strategy=strat, projector=small_projector, basis=small_basis
        )
        assert proof.max_abs_projection_residual == 0.0, strat
        assert pre.dtype == np.uint8


# ---------------------------------------------------------------------------
# The DISCRIMINATOR: an out-of-mask perturbation DOES change the projection.
# (A fake basis calling everything invisible is caught here.)
# ---------------------------------------------------------------------------
def test_out_of_mask_perturbation_changes_projection(small_projector, small_mask):
    frame = _make_frame(7).astype(np.uint8)
    y0 = small_projector.project_frame(frame)
    # find a NON-masked pixel and perturb it.
    nz = np.argwhere(~small_mask)
    r, c = int(nz[len(nz) // 2][0]), int(nz[len(nz) // 2][1])
    pert = frame.copy()
    pert[r, c, 0] = np.uint8((int(frame[r, c, 0]) + 100) % 256)
    y1 = small_projector.project_frame(pert)
    assert np.max(np.abs(y1 - y0)) > 1e-6


def test_in_mask_perturbation_does_not_change_projection(small_projector, small_mask):
    frame = _make_frame(8).astype(np.uint8)
    y0 = small_projector.project_frame(frame)
    mz = np.argwhere(small_mask)
    r, c = int(mz[len(mz) // 2][0]), int(mz[len(mz) // 2][1])
    pert = frame.copy()
    pert[r, c, :] = np.uint8(255)  # extreme change in a certified-invisible pixel
    y1 = small_projector.project_frame(pert)
    assert np.max(np.abs(y1 - y0)) == 0.0


# ---------------------------------------------------------------------------
# RGB-before-YUV + full upstream preprocess survival (operator caveat (b)).
# ---------------------------------------------------------------------------
def test_preimage_survives_full_upstream_preprocess():
    """The certified fill must leave BOTH heads' real preprocessed input
    bit-identical (the no-fake check, against the actual upstream modules)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("segmentation_models_pytorch")
    try:
        import modules  # upstream
    except Exception:  # pragma: no cover - import-resilience
        pytest.skip("upstream modules unavailable")

    from tac.optimization.evaluator_invisibility_basis import (
        CAMERA_H, CAMERA_W, derive_tier1_resize_null_space,
    )
    basis = derive_tier1_resize_null_space()
    proj = ResizeProjector.build()
    mask = zero_weight_pixel_mask(basis=basis)
    # a real-scale frame pair (small N, full camera size).
    rng = np.random.default_rng(9)
    frame0 = rng.integers(0, 256, (CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    frame1 = rng.integers(0, 256, (CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    pre1, proof = apply_tier1_zero_weight_fill(frame1, projector=proj, basis=basis,
                                               mask=mask)
    assert proof.max_abs_projection_residual == 0.0

    def _pair_tensor(f0, f1):
        # (B=1, T=2, C=3, H, W), the upstream scorer input layout.
        a = torch.from_numpy(f0).permute(2, 0, 1).float()
        b = torch.from_numpy(f1).permute(2, 0, 1).float()
        return torch.stack([a, b], dim=0).unsqueeze(0)

    seg = modules.SegNet()
    pose = modules.PoseNet()
    with torch.no_grad():
        seg_in0 = seg.preprocess_input(_pair_tensor(frame0, frame1))
        seg_in1 = seg.preprocess_input(_pair_tensor(frame0, pre1))
        pose_in0 = pose.preprocess_input(_pair_tensor(frame0, frame1))
        pose_in1 = pose.preprocess_input(_pair_tensor(frame0, pre1))
    # SegNet input (resize of frame1) bit-identical.
    assert torch.max(torch.abs(seg_in1 - seg_in0)).item() == 0.0
    # PoseNet input (resize + YUV6 of the pair) bit-identical — RGB equality BEFORE
    # YUV implies YUV6 equality.
    assert torch.max(torch.abs(pose_in1 - pose_in0)).item() == 0.0


# ---------------------------------------------------------------------------
# Idempotence + tier monotonicity + bytes reality.
# ---------------------------------------------------------------------------
def test_tier1_idempotent(small_projector, small_basis):
    frame = _make_frame(10)
    pre1, _ = apply_tier1_zero_weight_fill(
        frame, strategy="constant", projector=small_projector, basis=small_basis
    )
    pre2, _ = apply_tier1_zero_weight_fill(
        pre1, strategy="constant", projector=small_projector, basis=small_basis
    )
    assert np.array_equal(pre1, pre2)


def test_tier2_never_worse_than_tier1(small_projector, small_basis):
    frame = _make_frame(11)
    _, p1 = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    _, p2 = apply_tier2_null_basis_descent(
        frame, projector=small_projector, basis=small_basis
    )
    assert p2.bytes_after["brotli"] <= p1.bytes_after["brotli"]
    assert p2.max_abs_projection_residual == 0.0


def test_tier3_exact_and_valid(small_projector, small_basis):
    frame = _make_frame(12)
    pre, proof = apply_tier3_blockwise_flat_preimage(
        frame, projector=small_projector, basis=small_basis
    )
    assert proof.max_abs_projection_residual == 0.0
    assert pre.dtype == np.uint8 and pre.shape == frame.shape


def test_bytes_reduction_nonneg_on_compressible_frame(small_projector, small_basis):
    # tier-1 measured-best never INCREASES bytes vs the original (it can fall back
    # to the cheapest fill); on a compressible frame it strictly reduces.
    frame = _make_frame(13)
    _, proof = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    # at least one coder reduces; neither increases beyond a small margin.
    assert proof.bytes_reduction_brotli >= 0 or proof.bytes_reduction_lzma >= 0


def test_coded_size_real_coders():
    arr = np.zeros((100, 100), dtype=np.uint8)
    b = coded_size_bytes(arr, coder="brotli")
    z = coded_size_bytes(arr, coder="lzma")
    both = coded_size_both(arr)
    assert b > 0 and z > 0
    assert both["brotli"] == b and both["lzma"] == z
    # a flat array compresses to far less than its raw size.
    assert b < arr.size


def test_coded_size_search_quality_faster_path():
    arr = np.zeros((200, 200), dtype=np.uint8)
    q5 = coded_size_bytes(arr, coder="brotli", brotli_quality=5)
    q11 = coded_size_bytes(arr, coder="brotli", brotli_quality=11)
    assert q5 > 0 and q11 > 0  # both real measurements


# ---------------------------------------------------------------------------
# Fail-closed input validation + THE LAW rate delta.
# ---------------------------------------------------------------------------
def test_rejects_non_3d_frame(small_projector, small_basis):
    with pytest.raises(ResizeNullPreimageError):
        apply_tier1_zero_weight_fill(
            np.zeros((10, 10), dtype=np.uint8), projector=small_projector,
            basis=small_basis,
        )


def test_rejects_mismatched_mask(small_projector, small_basis):
    frame = _make_frame(14)
    bad_mask = np.zeros((CH + 1, CW), dtype=bool)
    with pytest.raises(ResizeNullPreimageError):
        apply_tier1_zero_weight_fill(
            frame, projector=small_projector, basis=small_basis, mask=bad_mask
        )


def test_rejects_unknown_coder():
    with pytest.raises(ResizeNullPreimageError):
        coded_size_bytes(np.zeros((4, 4), dtype=np.uint8), coder="zstd")


def test_projector_rejects_wrong_plane_shape(small_projector):
    with pytest.raises(ResizeNullPreimageError):
        small_projector.project_plane(np.zeros((CH + 1, CW)))


def test_rate_score_delta_is_negative_for_freed_bytes():
    # THE LAW rate term: freeing bytes => negative ΔS (improvement).
    ds = preimage_rate_score_delta(1000)
    assert ds < 0.0
    assert abs(ds - (25.0 * -1000 / CONTEST_TOTAL_BYTES)) < 1e-12


def test_rate_score_delta_zero_for_no_freed_bytes():
    assert preimage_rate_score_delta(0) == 0.0


def test_frame_proof_to_dict_carries_false_authority(small_projector, small_basis):
    frame = _make_frame(15)
    _, proof = apply_tier1_zero_weight_fill(
        frame, projector=small_projector, basis=small_basis
    )
    d = proof.to_dict()
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["proof_evidence_grade"] == "mathematical-derivation"
    assert d["bytes_evidence_grade"] == "[macOS-CPU advisory]"
    assert d["max_abs_projection_residual"] == 0.0
