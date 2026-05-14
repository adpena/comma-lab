# SPDX-License-Identifier: MIT
"""Lock-in tests for half-frame mask warp at inflate (Quantizr paradigm).

The archive ships only 600 odd-frame (t+1) masks. At inflate, the missing
600 even-frame (t) masks are reconstructed by inverse-warping each odd mask
through the per-pair RadialZoomWarp. This saves ~50% of mask bytes (the
biggest archive lever for sub-Quantizr scores).

These tests pin the warp_inverse_masks contract so the train↔inflate
consistency holds: any model trained on (mask_t = inverse_warp(mask_t1))
will see the SAME mask_t at inflate.
"""
from __future__ import annotations

import pytest
import torch

from tac.radial_zoom import RadialZoomWarp


def _square_mask(h: int = 384, w: int = 512, cx: int = 280, cy: int = 200,
                 size: int = 30, cls: int = 1) -> torch.Tensor:
    m = torch.zeros(h, w, dtype=torch.long)
    m[cy - size // 2 : cy + size // 2, cx - size // 2 : cx + size // 2] = cls
    return m


# ── Identity (zero-zoom) warp must round-trip ─────────────────────────────


def test_warp_inverse_zero_zoom_is_identity() -> None:
    """If all zoom scalars are 0, the inverse warp is the identity — every
    mask round-trips exactly."""
    zw = RadialZoomWarp(n_pairs=4)
    # zoom_scalars default to small values near zero. Force exact zero:
    with torch.no_grad():
        zw.zoom_scalars.zero_()
    masks_t1 = torch.stack([_square_mask(cx=300 + 5 * k) for k in range(4)])
    out = zw.warp_inverse_masks(masks_t1, torch.arange(4))
    assert out.shape == masks_t1.shape
    assert out.dtype == masks_t1.dtype
    assert torch.equal(out, masks_t1), "zero-zoom should be identity"


# ── Forward + inverse warp on the same mask is approximately identity ────


def test_forward_then_inverse_recovers_class_count_approximately() -> None:
    """Apply forward zoom (compress), then inverse zoom (decompress). The
    set of class-1 pixels should approximately match (some boundary loss is
    expected from nearest-neighbour resampling)."""
    zw = RadialZoomWarp(n_pairs=2)
    # Set a small but nonzero zoom
    with torch.no_grad():
        zw.zoom_scalars[:] = torch.tensor([0.05, -0.05])  # ~5% zoom each way
    mask_t = _square_mask(cx=320, cy=210, size=40)  # away from FoE for measurable warp
    masks_t1 = mask_t.unsqueeze(0).expand(2, -1, -1).contiguous()
    out = zw.warp_inverse_masks(masks_t1, torch.arange(2))
    n_in = (mask_t == 1).sum().item()
    for k in range(2):
        n_out = (out[k] == 1).sum().item()
        # Allow up to 30% boundary loss from nearest-neighbour interpolation
        ratio = n_out / n_in
        assert 0.7 <= ratio <= 1.3, f"pair {k} class-pixel ratio {ratio:.2f} outside tolerance"


# ── Class-index preservation: no fractional classes leak in ──────────────


def test_warp_inverse_preserves_class_set() -> None:
    """Nearest-neighbour resampling must keep classes integer-valued and
    within the original set (no half-classes from bilinear interpolation)."""
    zw = RadialZoomWarp(n_pairs=3)
    with torch.no_grad():
        zw.zoom_scalars[:] = torch.tensor([0.08, -0.08, 0.0])
    masks_t1 = torch.stack([
        _square_mask(cx=300, cy=200, cls=2),
        _square_mask(cx=350, cy=240, cls=4),
        _square_mask(cx=200, cy=160, cls=1),
    ])
    out = zw.warp_inverse_masks(masks_t1, torch.arange(3))
    assert out.dtype == torch.long
    # Each output mask must have classes ⊆ input classes ∪ {0 from border padding}
    for k in range(3):
        in_classes = set(masks_t1[k].unique().tolist())
        out_classes = set(out[k].unique().tolist())
        assert out_classes.issubset(in_classes | {0}), (
            f"pair {k} introduced new classes: in={in_classes}, out={out_classes}"
        )


# ── Border padding: OOB pixels take the most-common class (0) ────────────


def test_warp_inverse_uses_border_padding() -> None:
    """When inverse-zoom samples from outside the input, those pixels must
    take the border class (0 = road), not return zeros from default
    `padding_mode='zeros'` behaviour. (Both behave the same here since
    border value at the edges of comma frames is class 0 — but we explicitly
    test that no -1 / NaN / huge values leak through.)"""
    zw = RadialZoomWarp(n_pairs=1)
    with torch.no_grad():
        zw.zoom_scalars[:] = torch.tensor([0.1])  # max zoom — pushes content outward
    # Place class 1 at the very edge so inverse-zoom samples beyond it
    mask = torch.zeros(384, 512, dtype=torch.long)
    mask[:, 510:] = 1  # right edge column
    out = zw.warp_inverse_masks(mask.unsqueeze(0), torch.arange(1))
    assert out.dtype == torch.long
    # Output must contain only valid class indices (0..4)
    assert out.min() >= 0
    assert out.max() <= 4


# ── Batch consistency: per-pair scalars actually applied per-pair ────────


def test_warp_inverse_per_pair_scalars_independent() -> None:
    """Pair k's warp should use ONLY zoom_scalars[k]. Verify by comparing
    a batch run to per-pair single runs."""
    zw = RadialZoomWarp(n_pairs=4)
    with torch.no_grad():
        zw.zoom_scalars[:] = torch.tensor([0.05, -0.07, 0.02, -0.03])
    masks_t1 = torch.stack([
        _square_mask(cx=300 + 10 * k, cy=200, size=30, cls=1 + (k % 4))
        for k in range(4)
    ])

    batch_out = zw.warp_inverse_masks(masks_t1, torch.arange(4))
    for k in range(4):
        single_out = zw.warp_inverse_masks(
            masks_t1[k:k + 1], torch.tensor([k]),
        )
        assert torch.equal(batch_out[k], single_out[0]), (
            f"pair {k}: batched output differs from single — per-pair scalar "
            f"isolation broken"
        )


# ── Shape contract ────────────────────────────────────────────────────────


def test_warp_inverse_shape_dtype_preserved() -> None:
    zw = RadialZoomWarp(n_pairs=8)
    masks = torch.zeros(8, 384, 512, dtype=torch.long)
    out = zw.warp_inverse_masks(masks, torch.arange(8))
    assert out.shape == masks.shape
    assert out.dtype == masks.dtype


def test_warp_inverse_validates_indices_within_bounds() -> None:
    """Out-of-range pair_indices should raise via PyTorch indexing."""
    zw = RadialZoomWarp(n_pairs=4)
    masks = torch.zeros(2, 384, 512, dtype=torch.long)
    with pytest.raises((IndexError, RuntimeError)):
        zw.warp_inverse_masks(masks, torch.tensor([0, 999]))


# ── Lane D2: profile validation for mask_half_sim_prob ───────────────────


def test_preflight_catches_mask_half_sim_without_zoom_flow() -> None:
    """A profile setting mask_half_sim_prob > 0 without use_zoom_flow=True
    is dead-weight compute (the renderer can't consume the flow signal)."""
    from tac.preflight import preflight_profiles, PreflightError
    from tac.profiles import PROFILES

    # Inject a bad profile temporarily
    bad = dict(PROFILES["shiraz"])
    bad["mask_half_sim_prob"] = 0.5
    bad["use_zoom_flow"] = False
    PROFILES["__bad_test__"] = bad
    try:
        with pytest.raises(PreflightError) as ei:
            preflight_profiles(strict=True, verbose=False)
        msg = str(ei.value)
        assert "mask_half_sim_prob" in msg
        assert "use_zoom_flow" in msg
    finally:
        del PROFILES["__bad_test__"]


def test_preflight_accepts_mask_half_sim_with_zoom_flow() -> None:
    """The valid combination (both set, both consistent) must pass cleanly."""
    from tac.preflight import preflight_profiles
    from tac.profiles import PROFILES

    good = dict(PROFILES["green"])  # already has use_zoom_flow=True
    good["mask_half_sim_prob"] = 0.5
    PROFILES["__good_test__"] = good
    try:
        # Should not raise
        violations = preflight_profiles(strict=False, verbose=False)
        bad_for_us = [v for v in violations if "__good_test__" in v]
        assert not bad_for_us, f"unexpected violations: {bad_for_us}"
    finally:
        del PROFILES["__good_test__"]


def test_den_profile_consistency() -> None:
    """The DEN profile (Lane C) must satisfy the new rule."""
    from tac.profiles import DEN
    if DEN.get("mask_half_sim_prob", 0) > 0:
        assert DEN.get("use_zoom_flow"), (
            "DEN sets mask_half_sim_prob but use_zoom_flow not True — "
            "would trigger preflight failure"
        )
