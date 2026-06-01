# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the G3 HPRC synthesis adjoint A^T (Daubechies gate).

These tests verify the adjoint identity ``<A x, y> == <x, A^T y>`` to machine
precision AND verify that a NON-adjoint transform FAILS the test (the NO-FAKE
guard — a tautology would pass any transform). Every assertion exercises the
ACTUAL forward/adjoint pair, not metadata constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.analysis.hprc_synthesis_adjoint import (
    HprcDecodeGeometry,
    _adjoint_dotproduct,
    adjoint_dotproduct_bilinear,
    adjoint_dotproduct_latent,
    adjoint_dotproduct_nearest,
    adjoint_dotproduct_residual,
    bilinear_resize_adjoint,
    bilinear_resize_forward,
    geometry_from_compact_packet,
    nearest_resize_adjoint,
    nearest_resize_forward,
    push_pixel_saliency_to_latent,
    push_pixel_saliency_to_residual_grid,
)


# ---------------------------------------------------------------------------
# G3 exactness: <A x, y> == <x, A^T y> to machine precision.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src_h", "src_w", "dst_h", "dst_w"),
    [(3, 4, 24, 32), (6, 8, 48, 64), (12, 16, 874, 1164), (1, 1, 10, 10), (5, 7, 5, 7)],
)
def test_nearest_resize_adjoint_is_exact(src_h, src_w, dst_h, dst_w):
    r = adjoint_dotproduct_nearest(
        src_h=src_h, src_w=src_w, dst_h=dst_h, dst_w=dst_w, channels=3, seed=1
    )
    assert r.is_exact
    assert r.rel_residual < 1e-12, f"nearest adjoint not exact: {r.rel_residual}"


@pytest.mark.parametrize(
    ("src_h", "src_w", "dst_h", "dst_w"),
    [(12, 16, 874, 1164), (3, 4, 24, 32), (6, 8, 384, 512)],
)
def test_bilinear_resize_adjoint_is_exact(src_h, src_w, dst_h, dst_w):
    r = adjoint_dotproduct_bilinear(
        src_h=src_h, src_w=src_w, dst_h=dst_h, dst_w=dst_w, channels=3, frames=2, seed=2
    )
    assert r.is_exact
    assert r.rel_residual < 1e-11, f"bilinear adjoint not exact: {r.rel_residual}"


@pytest.mark.parametrize("output_mode", ["nearest", "bilinear"])
def test_composite_residual_decode_adjoint_is_exact(output_mode):
    """The FULL residual-decode composite A^T must be the exact transpose."""
    g = HprcDecodeGeometry(
        decoder_height=12,
        decoder_width=16,
        camera_height=874,
        camera_width=1164,
        residual_grid_h=3,
        residual_grid_w=4,
        channels=3,
        residual_scale=0.37,
        residual_gain=1.0,
        output_resize_mode=output_mode,
    )
    r = adjoint_dotproduct_residual(g, selector=200.0 / 255.0, frames=2, seed=3)
    assert r.is_exact
    assert r.rel_residual < 1e-12, f"composite residual adjoint not exact: {r.rel_residual}"
    # The two inner products must actually be equal (not both zero by accident).
    assert abs(r.lhs_inner) > 1.0
    assert abs(r.lhs_inner - r.rhs_inner) < 1e-9


def test_latent_decode_adjoint_is_exact():
    rng = np.random.default_rng(5)
    basis = rng.standard_normal((4, 12, 16, 3))
    r = adjoint_dotproduct_latent(basis=basis, latent_gain=1.0, frames=2, seed=6)
    assert r.is_exact
    assert r.rel_residual < 1e-11


# ---------------------------------------------------------------------------
# NO-FAKE guard: a NON-adjoint transform MUST FAIL the dot-product test.
# ---------------------------------------------------------------------------


def test_meanpool_is_not_the_adjoint_and_fails_the_test():
    """A mean-pool (divides by block size) is NOT the transpose of a gather.

    If the dot-product test passed this, it would be a tautology. It must FAIL.
    """

    def fake_meanpool_adjoint(y, src_h, src_w):
        out = nearest_resize_adjoint(y, src_h, src_w)
        counts = nearest_resize_adjoint(np.ones_like(y), src_h, src_w)
        return out / np.maximum(counts, 1.0)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4, 3))
    y = rng.standard_normal((24, 32, 3))
    r = _adjoint_dotproduct(
        operator="fake_meanpool",
        x=x,
        y=y,
        forward=lambda v: nearest_resize_forward(v, 24, 32),
        adjoint=lambda v: fake_meanpool_adjoint(v, 3, 4),
        tol=1e-9,
    )
    assert not r.is_exact, "BUG: a mean-pool passed the adjoint exactness test"
    assert r.rel_residual > 0.1, "mean-pool deviation should be large"


def test_forward_transpose_is_not_the_adjoint_for_bilinear():
    """Using the FORWARD bilinear (downsample) as the 'adjoint' must FAIL.

    A common bug: re-use the forward resize (down to source size) as if it were
    the transpose. For a non-square gather this is NOT the adjoint and must fail.
    """
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1, 6, 8, 3))
    y = rng.standard_normal((1, 48, 64, 3))
    r = _adjoint_dotproduct(
        operator="fake_forward_as_adjoint",
        x=x,
        y=y,
        forward=lambda v: bilinear_resize_forward(v, 48, 64),
        adjoint=lambda v: bilinear_resize_forward(v, 6, 8),  # WRONG: forward, not transpose
        tol=1e-9,
    )
    assert not r.is_exact, "BUG: forward-as-adjoint passed the exactness test"


def test_true_bilinear_adjoint_passes_where_forward_as_adjoint_fails():
    """Confirm the TRUE adjoint passes the exact case the fake one fails."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1, 6, 8, 3))
    y = rng.standard_normal((1, 48, 64, 3))
    r_true = _adjoint_dotproduct(
        operator="bilinear",
        x=x,
        y=y,
        forward=lambda v: bilinear_resize_forward(v, 48, 64),
        adjoint=lambda v: bilinear_resize_adjoint(v, 6, 8),
        tol=1e-9,
    )
    assert r_true.is_exact


# ---------------------------------------------------------------------------
# Adjoint mass-conservation: A^T accumulates (sum-pool), not averages.
# ---------------------------------------------------------------------------


def test_nearest_adjoint_conserves_total_mass_against_forward():
    """<A 1, y> == <1, A^T y>: total saliency mass is preserved by the adjoint."""
    rng = np.random.default_rng(9)
    y = rng.standard_normal((24, 32, 3))
    x_ones = np.ones((3, 4, 3))
    lhs = float(np.sum(nearest_resize_forward(x_ones, 24, 32) * y))
    rhs = float(np.sum(x_ones * nearest_resize_adjoint(y, 3, 4)))
    assert abs(lhs - rhs) < 1e-9


def test_residual_adjoint_output_shape_matches_token_grid():
    """A^T must land on the residual-token grid (frames, grid_h, grid_w[, C])."""
    g = HprcDecodeGeometry(
        decoder_height=12,
        decoder_width=16,
        camera_height=48,
        camera_width=64,
        residual_grid_h=3,
        residual_grid_w=4,
        channels=3,
        residual_scale=0.5,
        residual_gain=1.0,
        output_resize_mode="bilinear",
    )
    pixel = np.zeros((2, 48, 64))
    pixel[:, 10:14, 12:16] = 5.0
    collapsed = push_pixel_saliency_to_residual_grid(pixel, g, selector=1.0)
    assert collapsed.shape == (2, 3, 4)
    per_channel = push_pixel_saliency_to_residual_grid(
        pixel, g, selector=1.0, collapse_channels=False
    )
    assert per_channel.shape == (2, 3, 4, 3)
    # The hotspot pixel mass must land in a NON-zero token (not spread to zero).
    assert collapsed.sum() > 0.0


def test_latent_saliency_shape_is_per_dim():
    rng = np.random.default_rng(10)
    basis = rng.standard_normal((4, 12, 16, 3))
    pixel = rng.standard_normal((2, 12, 16, 3))
    lat = push_pixel_saliency_to_latent(pixel, basis=basis, latent_gain=1.0)
    assert lat.shape == (2, 4)


# ---------------------------------------------------------------------------
# Geometry extraction from a live HPRC packet (no idealized constants).
# ---------------------------------------------------------------------------


def test_geometry_from_compact_packet_reads_shipped_params():
    from tac.substrates.hprc.archive import parse_hprc_packet
    from tac.substrates.hprc.learned_receiver import (
        build_compact_receiver_packet_from_lowres_frames,
        decode_compact_receiver_packet,
    )

    rng = np.random.default_rng(12)
    frames = rng.integers(0, 256, size=(6, 12, 16, 3), dtype=np.uint8).astype(np.float32)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames, basis_count=3, residual_grid_h=3, residual_grid_w=4
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet))
    g = geometry_from_compact_packet(compact, camera_height=874, camera_width=1164)
    assert g.residual_grid_h == 3
    assert g.residual_grid_w == 4
    assert g.camera_height == 874
    assert g.camera_width == 1164
    assert g.output_resize_mode in {"nearest", "bilinear"}
    assert g.residual_scale > 0.0
    # The geometry's adjoint must be exact on the real packet's parameters.
    r = adjoint_dotproduct_residual(g, selector=1.0, frames=2, seed=99)
    assert r.is_exact


def test_real_packet_geometry_composite_adjoint_exact_bilinear():
    """End-to-end: a REAL HPRC packet's decode geometry yields an exact adjoint."""
    from tac.substrates.hprc.archive import parse_hprc_packet
    from tac.substrates.hprc.learned_receiver import (
        build_compact_receiver_packet_from_lowres_frames,
        decode_compact_receiver_packet,
    )

    rng = np.random.default_rng(33)
    frames = rng.integers(0, 256, size=(8, 24, 32, 3), dtype=np.uint8).astype(np.float32)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames, basis_count=3, residual_grid_h=6, residual_grid_w=8
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet))
    g = geometry_from_compact_packet(compact, camera_height=874, camera_width=1164)
    r = adjoint_dotproduct_residual(g, selector=1.0, frames=8, seed=44)
    assert r.is_exact
    assert r.rel_residual < 1e-11
