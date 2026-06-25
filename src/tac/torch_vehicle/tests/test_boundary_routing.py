# SPDX-License-Identifier: MIT
"""NO-FAKE synthetic-unit tests for the boundary capacity-router
(``tac.torch_vehicle.boundary_routing``).

These run on SYNTHETIC tensors ($0, non-contending: no scorer forward, no MPS,
no real checkpoint). They verify BEHAVIOR (not constants): the distance map
actually measures distance-to-the-class-1 boundary; the proximity key is high on
the band and low in the interior; the FiLM gate is identity-at-init AND
concentrates modulation near the boundary after the heads are perturbed; the
oriented PE is finite + correctly shaped; param count matches the documented
formula. Each ``test_*`` is named for the load-bearing claim it protects.

A complementary demo on REAL cached GT (``gt_targets_n6.pt``) lives at the bottom
of the module under ``demo_real_gt()`` (NOT a pytest test — gated on the cache
file existing, run manually / by the build report); it is a cheap ``torch.load``
of a tiny tensor blob, never a scorer forward.
"""

from __future__ import annotations

import torch

from tac.torch_vehicle.boundary_routing import (
    LANE_CLASS,
    BoundaryFiLM,
    boundary_distance_map,
    boundary_proximity_feature,
    directional_positional_encoding,
    local_boundary_tangent,
)


def _synthetic_seg(B: int = 2, H: int = 32, W: int = 48) -> torch.Tensor:
    """A deterministic GT-argmax with a horizontal lane STRIPE of class 1.

    Rows [H//2-1, H//2, H//2+1] are class 1 (lane); everything else alternates
    class 0 / class 2 so the only class-1 boundary is the stripe's top/bottom
    edges (known, hand-checkable geometry).
    """
    seg = torch.zeros((B, H, W), dtype=torch.int64)
    seg[:, ::2, :] = 0
    seg[:, 1::2, :] = 2
    lane_rows = [H // 2 - 1, H // 2, H // 2 + 1]
    for r in lane_rows:
        seg[:, r, :] = LANE_CLASS
    return seg


# ---------------------------------------------------------------------------
# boundary_distance_map
# ---------------------------------------------------------------------------
def test_distance_map_shape_dtype_finite_and_promotes_2d():
    seg = _synthetic_seg(B=3, H=16, W=20)
    dm = boundary_distance_map(seg)
    assert dm.shape == (3, 16, 20)
    assert dm.dtype == torch.float32
    assert torch.isfinite(dm).all()
    # 2D input is promoted to (1, H, W).
    dm2 = boundary_distance_map(seg[0])
    assert dm2.shape == (1, 16, 20)
    assert torch.isfinite(dm2).all()


def test_distance_map_is_small_on_band_and_grows_into_interior():
    """Unsigned distance (|signed level-set|) must be SMALL (~1px, the interface
    runs between pixels) on the lane-stripe edges and increase away from it on
    both sides."""
    H, W = 40, 24
    seg = _synthetic_seg(B=1, H=H, W=W)
    dm = boundary_distance_map(seg)[0]  # (H, W)
    # The pixels straddling the edge are ~1 px from the boundary curve; the
    # band minimum over the whole frame is small.
    assert float(dm.min()) <= 1.0 + 1e-4
    # Far from the stripe (top row) distance must be strictly larger than just
    # outside the stripe.
    top_edge = H // 2 - 1  # first lane row
    just_outside = float(dm[top_edge - 1, W // 2])
    far_outside = float(dm[0, W // 2])
    assert far_outside > just_outside > 0.0
    # Magnitude sanity: one row above the stripe is ~1-2 px from the boundary.
    assert 0.0 < just_outside <= 2.0


def test_signed_distance_is_negative_inside_lane_positive_outside():
    seg = _synthetic_seg(B=1, H=32, W=24)
    sdm = boundary_distance_map(seg, signed=True)[0]
    lane_mask = seg[0] == LANE_CLASS
    # The lane stripe is 3 px thick: its CENTER row is strictly inside -> negative.
    center = 32 // 2
    assert float(sdm[center, 12]) < 0.0
    assert lane_mask[center, 12]
    # A pixel far outside is positive.
    assert float(sdm[0, 12]) > 0.0


def test_distance_map_no_lane_pixels_is_finite_large_sentinel():
    seg = torch.full((1, 64, 64), 2, dtype=torch.int64)  # no class 1 anywhere
    dm = boundary_distance_map(seg)
    assert torch.isfinite(dm).all()
    # Sentinel = the frame diagonal (~90 px here) so proximity exp(-d/tau) ~ 0
    # everywhere (the whole frame is "interior" — no boundary to route to).
    prox = boundary_proximity_feature(dm, tau=4.0)
    assert float(prox.max()) < 1e-6


def test_distance_map_preserves_device_cpu():
    seg = _synthetic_seg(B=1, H=8, W=8)
    dm = boundary_distance_map(seg)
    assert dm.device == seg.device  # CPU here; the contract is "same device".


# ---------------------------------------------------------------------------
# boundary_proximity_feature
# ---------------------------------------------------------------------------
def test_proximity_is_high_on_band_and_decays_to_zero_interior():
    seg = _synthetic_seg(B=1, H=40, W=24)
    dm = boundary_distance_map(seg)
    prox = boundary_proximity_feature(dm, tau=4.0)[0]
    assert prox.shape == (40, 24)
    assert float(prox.min()) >= 0.0 and float(prox.max()) <= 1.0
    # HIGH (close to 1) on the band edges (distance ~1 px -> exp(-1/4)~0.78).
    assert float(prox.max()) > 0.7
    # Near 0 far in the interior (top row, ~19 px away -> exp(-18/4)~0.01).
    assert float(prox[0, 12]) < 0.05


def test_proximity_tau_controls_band_width():
    seg = _synthetic_seg(B=1, H=40, W=24)
    dm = boundary_distance_map(seg)
    wide = boundary_proximity_feature(dm, tau=10.0)
    narrow = boundary_proximity_feature(dm, tau=2.0)
    # Wider tau -> more pixels with appreciable proximity.
    assert int((wide > 0.5).sum()) > int((narrow > 0.5).sum())


def test_proximity_rejects_nonpositive_tau():
    dm = torch.zeros((1, 4, 4))
    for bad in (0.0, -1.0):
        try:
            boundary_proximity_feature(dm, tau=bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"tau={bad} should raise ValueError")


# ---------------------------------------------------------------------------
# BoundaryFiLM
# ---------------------------------------------------------------------------
def test_boundary_film_identity_at_init():
    """At init (zero-init heads) the gate is an EXACT no-op regardless of p."""
    torch.manual_seed(0)
    gate = BoundaryFiLM(channels=16, embed_dim=8)
    x = torch.randn(2, 16, 12, 18)
    p = torch.rand(2, 1, 12, 18)
    y = gate(x, p)
    assert torch.equal(y, x), "gate must be identity at init (basin-resume safety)"


def test_boundary_film_param_count_matches_formula():
    E, C = 8, 64
    gate = BoundaryFiLM(channels=C, embed_dim=E)
    # embed (1->E): E*(1+1); gamma+beta heads (E->C): 2 * C*(E+1).
    expected = (E * 2) + 2 * (C * (E + 1))
    assert gate.param_count() == expected
    # Sanity: the documented 1168 for E=8, C=64.
    assert gate.param_count() == 1168


def test_boundary_film_param_count_is_tiny_vs_conv_weights():
    """The added params must be << the ~83K conv weights (rate cost negligible)."""
    gate = BoundaryFiLM(channels=64, embed_dim=8)
    assert gate.param_count() < 0.02 * 83_000  # < 2% of decoder conv weights


def test_boundary_film_concentrates_modulation_near_boundary():
    """After perturbing the heads, the gate must change near-boundary pixels MORE
    than interior pixels (the WHOLE POINT: capacity routed to the band)."""
    torch.manual_seed(1)
    gate = BoundaryFiLM(channels=8, embed_dim=8)
    # Perturb the (zero-init) heads so gamma/beta are nonzero (simulate trained).
    with torch.no_grad():
        for p_ in gate.gamma_head.parameters():
            p_.add_(torch.randn_like(p_) * 0.5)
        for p_ in gate.beta_head.parameters():
            p_.add_(torch.randn_like(p_) * 0.5)
    seg = _synthetic_seg(B=1, H=48, W=24)
    dm = boundary_distance_map(seg)
    prox = boundary_proximity_feature(dm, tau=4.0)  # (1, 48, 24)
    x = torch.randn(1, 8, 48, 24)
    with torch.no_grad():
        y = gate(x, prox)
    delta = (y - x).abs().mean(dim=1)[0]  # (48, 24) per-pixel change magnitude
    band = prox[0] > 0.5
    interior = prox[0] < 0.01  # deep interior: prox ~ 0
    assert band.any() and interior.any()
    band_change = float(delta[band].mean())
    interior_change = float(delta[interior].mean())
    # The modulation is scaled per-pixel by p, so the change is proportional to
    # proximity: the band (p~1) is modulated MUCH more than the deep interior
    # (p~0). The load-bearing property is the RATIO (capacity routed to the band)
    # plus the per-pixel proportionality (interior change scales with its tiny p).
    assert band_change > 20 * (interior_change + 1e-8)
    # Per-pixel proportionality: mean interior change <= ~ mean interior p * a
    # bounded modulation magnitude (here interior p < 0.01, so change is small in
    # absolute terms relative to the band).
    mean_interior_p = float(prox[0][interior].mean())
    assert mean_interior_p < 0.01
    assert interior_change < band_change


def test_boundary_film_resizes_proximity_to_feature_resolution():
    """Decoder stages are lower-res than the 384x512 GT band; the gate must
    bilinear-resize the proximity key to the feature map size."""
    torch.manual_seed(2)
    gate = BoundaryFiLM(channels=4, embed_dim=4)
    x = torch.randn(1, 4, 6, 8)  # low-res decoder stage
    prox = torch.rand(1, 1, 32, 24)  # full-res-ish band key
    y = gate(x, prox)  # must not raise; identity at init
    assert y.shape == x.shape
    assert torch.equal(y, x)  # still identity at init even after resize


def test_boundary_film_accepts_3d_proximity():
    gate = BoundaryFiLM(channels=4, embed_dim=4)
    x = torch.randn(1, 4, 10, 10)
    prox_3d = torch.rand(1, 10, 10)  # (B, H, W) accepted
    y = gate(x, prox_3d)
    assert y.shape == x.shape


def test_boundary_film_rejects_channel_mismatch():
    gate = BoundaryFiLM(channels=8, embed_dim=4)
    x = torch.randn(1, 16, 4, 4)  # 16 != 8
    try:
        gate(x, torch.rand(1, 1, 4, 4))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("channel mismatch should raise ValueError")


# ---------------------------------------------------------------------------
# directional PE + tangent
# ---------------------------------------------------------------------------
def test_local_boundary_tangent_shape_unit_and_finite():
    seg = _synthetic_seg(B=2, H=32, W=24)
    sdm = boundary_distance_map(seg, signed=True)
    tan = local_boundary_tangent(sdm)
    assert tan.shape == (2, 32, 24, 2)
    assert torch.isfinite(tan).all()
    norms = torch.sqrt((tan**2).sum(dim=-1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tangent_of_horizontal_stripe_is_horizontal():
    """For a horizontal lane stripe the boundary runs left-right, so the tangent
    near the band should be ~(+/-1, 0) (horizontal)."""
    seg = _synthetic_seg(B=1, H=40, W=40)
    sdm = boundary_distance_map(seg, signed=True)
    tan = local_boundary_tangent(sdm)[0]  # (40, 40, 2)
    # Sample a pixel just above the stripe, mid-width: gradient is vertical ->
    # tangent horizontal -> |tx| ~ 1, |ty| ~ 0.
    r = 40 // 2 - 3
    tx, ty = float(tan[r, 20, 0]), float(tan[r, 20, 1])
    assert abs(abs(tx) - 1.0) < 1e-3
    assert abs(ty) < 1e-3


def test_directional_pe_shape_finite_and_bounded():
    B, H, W, n = 1, 16, 20, 5
    seg = _synthetic_seg(B=B, H=H, W=W)
    sdm = boundary_distance_map(seg, signed=True)
    tan = local_boundary_tangent(sdm)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    coords = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    enc = directional_positional_encoding(coords, tan, n_freqs=n)
    assert enc.shape == (B, H, W, 4 * n)
    assert torch.isfinite(enc).all()
    # sin/cos bounded in [-1, 1].
    assert float(enc.max()) <= 1.0 + 1e-6 and float(enc.min()) >= -1.0 - 1e-6


def test_directional_pe_is_anisotropic():
    """The across-edge channels must carry HIGHER frequency content than the
    along-edge channels (the directional inductive bias). We check that swapping
    freq_across/freq_along changes the encoding (it is not isotropic)."""
    H, W, n = 24, 24, 4
    seg = _synthetic_seg(B=1, H=H, W=W)
    sdm = boundary_distance_map(seg, signed=True)
    tan = local_boundary_tangent(sdm)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    coords = torch.stack([xs, ys], dim=-1).unsqueeze(0)
    aniso = directional_positional_encoding(
        coords, tan, n_freqs=n, freq_across=32.0, freq_along=4.0
    )
    swapped = directional_positional_encoding(
        coords, tan, n_freqs=n, freq_across=4.0, freq_along=32.0
    )
    assert not torch.allclose(aniso, swapped), "encoding must be direction-aware"


def test_directional_pe_rejects_bad_shapes():
    coords = torch.zeros(1, 4, 4, 2)
    tan_bad = torch.zeros(1, 4, 4, 3)  # last dim != 2
    try:
        directional_positional_encoding(coords, tan_bad)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("last-dim != 2 should raise ValueError")


# ---------------------------------------------------------------------------
# REAL cached-GT demo (NOT a pytest test; cheap torch.load, no scorer forward)
# ---------------------------------------------------------------------------
def demo_real_gt(cache_path: str | None = None) -> dict:
    """Demo ``boundary_distance_map`` on REAL cached GT seg targets if present.

    Returns a small dict of measured stats (or ``{"available": False}`` if the
    cache file is absent). This is a $0 ``torch.load`` of a tiny tensor blob —
    NO scorer forward, NO MPS, non-contending with the live training daemons.
    """
    from pathlib import Path

    default = "experiments/results/capstone_gt_targets_cache/gt_targets_n6.pt"
    path = Path(cache_path or default)
    if not path.exists():
        return {"available": False, "path": str(path)}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    seg = blob["seg"]  # (n, 384, 512) int64
    dm = boundary_distance_map(seg)
    prox = boundary_proximity_feature(dm, tau=4.0)
    lane_frac = float((seg == LANE_CLASS).float().mean())
    band_frac = float((prox > 0.5).float().mean())
    return {
        "available": True,
        "path": str(path),
        "seg_shape": tuple(seg.shape),
        "lane_class_fraction": round(lane_frac, 5),
        "boundary_band_fraction_tau4": round(band_frac, 5),
        "distance_finite": bool(torch.isfinite(dm).all()),
        "distance_max_px": round(float(dm.max()), 2),
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(demo_real_gt(), indent=2))
