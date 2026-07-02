"""Tests for the AA-SDF observation-map render (footprint-integrated witness render).

Covers the NEW primitives: supersample coords (ss=1 byte-identical to the trainer grid),
box-downsample (numpy authority == torch area for integer factors; MLX parity >= 0.9997),
mip-NeRF IPE curvelet attenuation (monotone, bounded, sin/cos tiling), and the ss=1
byte-identical guarantee for the AA render-through-R drop-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.aa_sdf_observation_render import (
    SEG_H,
    SEG_W,
    AASDFRenderError,
    apply_ipe_attenuation,
    box_downsample_np,
    build_render_coords,
    build_supersampled_coords,
    ipe_curvelet_attenuation,
    ipe_footprint_sigma,
)


# ---------------------------------------------------------------------------
# coords / constants
# ---------------------------------------------------------------------------
def test_seg_constants_match_canonical_scorer_hw():
    from tac.local_acceleration.pr95_hnerv_mlx_training import SCORER_HW

    assert (SEG_H, SEG_W) == tuple(SCORER_HW)


def test_build_render_coords_bit_identical_to_trainer_helper():
    # ss=1 rendering must be byte-identical to the current point-sampled path.
    from experiments.train_witness_realized_through_R_mlx import _build_render_coords

    for h, w in [(4, 4), (192, 256), (384, 512)]:
        ours = build_render_coords(h, w)
        theirs = _build_render_coords(h, w)
        assert ours.shape == theirs.shape
        assert np.array_equal(ours, theirs), f"coord mismatch at {(h, w)}"


def test_supersampled_coords_ss1_is_identity():
    base = build_render_coords(8, 6)
    ss1 = build_supersampled_coords(8, 6, 1)
    assert np.array_equal(base, ss1)


def test_supersampled_coords_shape():
    c = build_supersampled_coords(8, 6, 3)
    assert c.shape == (8 * 3 * 6 * 3, 2)


def test_supersampled_coords_rejects_bad_ss():
    with pytest.raises(AASDFRenderError):
        build_supersampled_coords(8, 6, 0)


# ---------------------------------------------------------------------------
# box downsample
# ---------------------------------------------------------------------------
def test_box_downsample_ss1_identity():
    x = np.random.default_rng(0).standard_normal((2, 4, 6, 3)).astype(np.float32)
    assert np.array_equal(box_downsample_np(x, 1), x)


def test_box_downsample_block_mean_correct():
    # A hand 2x2 block: mean of the ss x ss cell.
    x = np.zeros((1, 2, 2, 1), np.float64)
    x[0, :, :, 0] = [[1.0, 3.0], [5.0, 7.0]]
    out = box_downsample_np(x, 2)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == pytest.approx(4.0)  # (1+3+5+7)/4


def test_box_downsample_matches_torch_area_integer_factor():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 12, 16, 3)).astype(np.float32) * 40.0 + 128.0
    ours = box_downsample_np(x, 4)  # -> (3, 3, 4, 3)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    area = torch.nn.functional.interpolate(xt, size=(3, 4), mode="area")
    theirs = area.permute(0, 2, 3, 1).contiguous().numpy()
    # torch area for integer factor == exact block mean; agree to fp tolerance.
    assert np.allclose(ours, theirs, atol=1e-4, rtol=1e-4)


def test_box_downsample_mlx_parity():
    mx = pytest.importorskip("mlx.core")
    from tac.boundary_math.aa_sdf_observation_render import box_downsample_mlx

    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 8, 8, 3)).astype(np.float32) * 30.0 + 120.0
    ref = box_downsample_np(x, 2)
    got = np.asarray(box_downsample_mlx(mx.array(x), 2))
    assert got.shape == ref.shape
    # numpy is the authority; MLX matches to fp tolerance. Parity (1 - rel L2) >= 0.9997.
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
    assert (1.0 - rel) >= 0.9997


def test_box_downsample_rejects_indivisible():
    x = np.zeros((1, 5, 6, 3), np.float32)
    with pytest.raises(AASDFRenderError):
        box_downsample_np(x, 2)


# ---------------------------------------------------------------------------
# mip-NeRF IPE attenuation
# ---------------------------------------------------------------------------
def test_ipe_footprint_sigma_scales():
    sx, sy = ipe_footprint_sigma(384, 512, footprint_scale=1.0)
    # finer grid (more columns) => smaller x pitch => smaller sx.
    sx2, _ = ipe_footprint_sigma(384, 1024, footprint_scale=1.0)
    assert sx2 < sx
    assert sx > 0 and sy > 0
    # scaling the footprint scales sigma linearly.
    sx_hi, _ = ipe_footprint_sigma(384, 512, footprint_scale=2.0)
    assert sx_hi == pytest.approx(2.0 * sx)


def test_ipe_attenuation_bounded_and_monotone():
    # Frequencies increasing in magnitude -> attenuation strictly decreasing, in (0, 1].
    B = np.array([[0.0, 1.0, 2.0, 8.0], [0.0, 1.0, 2.0, 8.0]], np.float64)
    sx, sy = ipe_footprint_sigma(64, 64, 1.0)
    att = ipe_curvelet_attenuation(B, sx, sy)
    assert att.shape == (4,)
    assert att[0] == pytest.approx(1.0)  # zero frequency: no attenuation
    assert np.all(att > 0.0) and np.all(att <= 1.0)
    assert np.all(np.diff(att) < 0.0)  # strictly decreasing with frequency


def test_apply_ipe_attenuation_tiles_sin_cos():
    # feats (P, 2*cols) = [sin | cos]; att applies to both halves identically.
    cols = 3
    feats = np.arange(2 * 2 * cols, dtype=np.float32).reshape(2, 2 * cols)
    att = np.array([0.5, 0.25, 0.1], np.float32)
    out = apply_ipe_attenuation(feats, att)
    tiled = np.concatenate([att, att])
    assert np.allclose(out, feats * tiled[None, :])


def test_apply_ipe_attenuation_rejects_mismatch():
    feats = np.zeros((4, 8), np.float32)  # cols = 4
    with pytest.raises(AASDFRenderError):
        apply_ipe_attenuation(feats, np.zeros(3, np.float32))


# ---------------------------------------------------------------------------
# ss=1 render byte-identity (the byte-identical default guarantee)
# ---------------------------------------------------------------------------
def test_render_aa_ss1_byte_identical_to_render_batch():
    mx = pytest.importorskip("mlx.core")
    from experiments.train_witness_realized_through_R_mlx import (
        render_batch_through_R_mlx,
    )
    from tac.boundary_math.aa_sdf_observation_render import (
        render_aa_batch_through_R_mlx,
    )

    # Tiny witness stub with a .call_batch matching the trainer contract (M, P, 3) in [0,255].
    class _StubWitness:
        def call_batch(self, coord_feats, code_indices):
            rng = mx.random.uniform(shape=(len(code_indices), coord_feats.shape[0], 3))
            return rng * 255.0

    render_h, render_w = 12, 16
    coord_feats = mx.array(build_render_coords(render_h, render_w))  # (P, 2) stand-in feats
    codes = mx.array([0, 1])
    mx.random.seed(7)
    ref = np.asarray(render_batch_through_R_mlx(_StubWitness(), coord_feats, codes, render_h, render_w))
    mx.random.seed(7)
    got = np.asarray(render_aa_batch_through_R_mlx(_StubWitness(), coord_feats, codes, render_h, render_w, 1))
    assert got.shape == ref.shape == (2, SEG_H, SEG_W, 3)
    assert np.array_equal(got, ref), "ss=1 AA render must be byte-identical to render_batch_through_R_mlx"
