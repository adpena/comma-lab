# SPDX-License-Identifier: MIT
"""Behavior tests for the PTNC PoseNet-Jacobian saliency field (task #61).

NO-FAKE (class 2 + class 8): the tests verify ACTUAL behavior of the measured-field transform — a
constant/uniform saliency yields the identity weight map (proving the Jacobian field is load-bearing,
not cosmetic); a concentrated saliency redistributes weight onto the pose-relevant pixels; the
fail-closed guard fires on an all-zero (severed-gradient) field. The measured-field path itself (the
real frozen-PoseNet backprop) is exercised by the on-scorer test below (slow; tiny field). If every
test here still passed with the weight transform replaced by a constant, the suite would be verifying
constants not behavior — the concentration + identity + monotonicity tests make that impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.posenet_jacobian_saliency import (
    PixelSaliencyField,
    PoseNetSaliencyError,
    downsample_field,
    identity_weight_map,
    saliency_to_weight_map,
)


def _field(saliency: np.ndarray, slot: int = 0) -> PixelSaliencyField:
    s = np.asarray(saliency, dtype=np.float32)
    smax = float(s.max())
    nz = float(np.mean(s > (1e-6 * max(smax, 1e-30))))
    return PixelSaliencyField(
        saliency=s, h=int(s.shape[0]), w=int(s.shape[1]), frame_slot=slot,
        compute_path="cpu_torch", nonzero_fraction=nz, max_value=smax,
    )


# --- identity / uniform control (proves the Jacobian field is load-bearing) -------------------------
def test_uniform_saliency_gives_uniform_weight():
    """A flat saliency field => uniform weight map == identity (PTNC degenerates to dense MSE)."""
    f = _field(np.full((16, 20), 3.0, dtype=np.float32))
    w = saliency_to_weight_map(f, floor=0.02, gamma=1.0, normalize=True)
    assert np.allclose(w, 1.0, atol=1e-5)
    ident = identity_weight_map(16, 20)
    assert np.allclose(w, ident, atol=1e-5)


def test_identity_weight_map_is_uniform_ones():
    w = identity_weight_map(8, 11)
    assert w.shape == (8, 11)
    assert np.all(w == 1.0)


# --- concentration (the PTNC mechanism: redistribute capacity to pose-relevant pixels) --------------
def test_concentrated_saliency_redistributes_weight():
    """A saliency spike at one pixel => that pixel gets much higher weight than the null pixels."""
    s = np.full((10, 10), 0.001, dtype=np.float32)
    s[5, 5] = 1.0  # the pose tube
    f = _field(s)
    w = saliency_to_weight_map(f, floor=0.02, gamma=1.0, normalize=True)
    assert w[5, 5] > 5.0 * w[0, 0]  # the tube weighs far more than the null
    # mean renormalised to ~1 so the loss magnitude is comparable to dense MSE.
    assert abs(float(w.mean()) - 1.0) < 1e-4


def test_gamma_sharpens_concentration():
    """gamma>1 concentrates weight more onto the high-saliency tube (mid-pixels lose weight)."""
    s = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    f = _field(s)
    w1 = saliency_to_weight_map(f, gamma=1.0, normalize=False)
    w3 = saliency_to_weight_map(f, gamma=3.0, normalize=False)
    # min (floor) and max (1.0) are identical across gamma; the SHARPENING shows in the interior:
    # gamma>1 pushes mid-saliency pixels DOWN toward the floor (less weight off the tube).
    assert float(np.median(w3)) < float(np.median(w1))
    # equivalently, the share of total weight held by the top decile grows with gamma.
    top1 = np.sort(w1.ravel())[::-1][: w1.size // 10].sum() / w1.sum()
    top3 = np.sort(w3.ravel())[::-1][: w3.size // 10].sum() / w3.sum()
    assert top3 > top1


def test_weight_is_monotone_in_saliency():
    """Higher saliency => weight never decreases (a pure monotone transform)."""
    s = np.array([[0.0, 0.1, 0.5], [0.7, 0.9, 1.0]], dtype=np.float32) + 1e-3
    f = _field(s)
    w = saliency_to_weight_map(f, gamma=1.0, normalize=False)
    flat_s = s.ravel()
    flat_w = w.ravel()
    order = np.argsort(flat_s)
    assert np.all(np.diff(flat_w[order]) >= -1e-6)


def test_floor_keeps_nonzero_weight_everywhere():
    """The floor keeps a small positive weight even where saliency is ~0 (numerical stability)."""
    s = np.zeros((6, 6), dtype=np.float32)
    s[0, 0] = 1.0
    f = _field(s)
    w = saliency_to_weight_map(f, floor=0.05, gamma=1.0, normalize=False)
    assert float(w.min()) >= 0.05 - 1e-6
    assert float(w.min()) > 0.0


# --- fail-closed (severed-gradient signature) -------------------------------------------------------
def test_all_zero_saliency_raises():
    """An identically-zero field is the severed-gradient signature => refuse to fake pose-null."""
    f = _field(np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(PoseNetSaliencyError):
        saliency_to_weight_map(f)


# --- summary / shape contracts ----------------------------------------------------------------------
def test_field_summary_contract():
    s = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    f = _field(s)
    summ = f.to_summary()
    assert summ["max"] == pytest.approx(3.0)
    assert summ["mean"] == pytest.approx(1.5)
    assert 0.0 <= summ["nonzero_fraction"] <= 1.0


def test_weight_map_shape_matches_field():
    f = _field(np.random.default_rng(0).random((12, 9)).astype(np.float32) + 1e-3)
    w = saliency_to_weight_map(f)
    assert w.shape == (12, 9)
    assert w.dtype == np.float32


def test_downsample_field_mean_pools():
    s = np.array([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    d = downsample_field(s, 2)
    assert d.shape == (1, 2)
    assert d[0, 0] == pytest.approx(1.0)
    assert d[0, 1] == pytest.approx(2.0)


def test_downsample_factor_one_is_identity():
    s = np.random.default_rng(1).random((4, 4)).astype(np.float32)
    assert np.array_equal(downsample_field(s, 1), s)


def test_normalize_preserves_relative_order():
    """Normalisation rescales but does not change which pixels weigh more."""
    s = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float32)
    f = _field(s)
    wn = saliency_to_weight_map(f, normalize=True)
    wu = saliency_to_weight_map(f, normalize=False)
    # same argsort order
    assert np.array_equal(np.argsort(wn.ravel()), np.argsort(wu.ravel()))


def test_concentration_metric_top20_fraction():
    """A realistic concentrated field puts a disproportionate weight share in the top pixels."""
    rng = np.random.default_rng(2)
    s = rng.random((40, 40)).astype(np.float32) ** 4  # heavy-tailed (concentrated)
    f = _field(s)
    w = saliency_to_weight_map(f, gamma=1.0, normalize=True)
    flat = np.sort(w.ravel())[::-1]
    top20 = flat[: int(0.2 * flat.size)].sum() / flat.sum()
    assert top20 > 0.20  # more than uniform share => redistribution happened


# --- on-scorer measured field (slow; the real frozen PoseNet Jacobian) ------------------------------
@pytest.mark.slow
def test_measured_field_is_real_and_nondegenerate():
    """The REAL frozen-PoseNet pixel-Jacobian: nonzero, concentrated, fail-closed otherwise.

    Exercises the actual differentiable-yuv6 backprop path on a small GT pair. Skips if the GT video /
    weights are unavailable in this environment.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    harness = root / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
    for p in (root, root / "src", root / "upstream", harness):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    pytest.importorskip("torch")
    pytest.importorskip("av")
    try:
        import render_and_score_lib as L  # type: ignore
        from modules import PoseNet, posenet_sd_path  # type: ignore
        from safetensors.torch import load_file

        from tac.boundary_math.posenet_jacobian_saliency import compute_posenet_pixel_saliency
        from tac.differentiable_eval_roundtrip import (
            patch_upstream_yuv6_globally,
            unpatch_upstream_yuv6,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"scorer/GT unavailable: {exc}")


    if not Path(posenet_sd_path).exists():
        pytest.skip("posenet weights unavailable")
    net = PoseNet().eval()
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for pp in net.parameters():
        pp.requires_grad_(False)
    try:
        gt = L.decode_gt_pairs([0])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GT decode unavailable: {exc}")
    g0 = gt[0][0].float().permute(2, 0, 1).contiguous()
    g1 = gt[0][1].float().permute(2, 0, 1).contiguous()

    tok = patch_upstream_yuv6_globally()
    try:
        field = compute_posenet_pixel_saliency(net, g0, g1, frame_slot=0)
    finally:
        unpatch_upstream_yuv6(tok)
    assert field.max_value > 0.0
    assert field.nonzero_fraction > 0.1  # PoseNet reads a substantial fraction of pixels
    # concentrated: max >> median (the pose tube is sparse).
    summ = field.to_summary()
    assert summ["max"] > 10.0 * summ["median"]
    w = saliency_to_weight_map(field, floor=0.02, gamma=1.0, normalize=True)
    assert w.max() > 3.0 * w.mean()  # real redistribution
