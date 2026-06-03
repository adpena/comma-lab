# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the canonical ``recon_pixel_weight`` channel.

The channel is the codex-named reusable building block (memo
``codex_findings_z8_pixel_driver_and_segnet_grid_premise_20260531T153038Z``):
an OPT-IN per-pixel reconstruction-loss weight (default OFF, byte-identical to
pre-channel runs per Catalog #290) that consumes a measured-SegNet-saliency map
and weights the recon MSE so the renderer spends capacity on score-relevant
pixels. These tests verify REAL behavior (Class-2 NO-FAKE per CLAUDE.md
"NO FAKE IMPLEMENTATIONS"):

  * default OFF reproduces the EXACT uniform recon (byte-identical guard);
  * the channel ACTUALLY changes the recon loss for a non-uniform map (the
    headline no-op guard — FAILS if the weighted branch reverts to uniform);
  * the ``"mean"`` normalization preserves the loss SCALE on a uniform map
    (so the recon_weight Lagrangian coefficient is comparable across A/B arms);
  * shape coercion ``(H,W)`` / ``(H,W,1)`` / ``(H,W,3)`` / ``(1,H,W,1)`` /
    ``(N,H,W,C)`` / ``(N,2,H,W,C)`` all work and mismatched / negative /
    non-finite maps are REFUSED;
  * gradient flows to the renderer through the weighted recon but NOT to the
    weight (the weight is ``mx.stop_gradient``).

MLX-bound; skips cleanly on non-Apple-Silicon CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates._shared.mlx_score_aware import (
    RendererBundle,
    score_aware_loss,
)
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)
from tac.substrates._shared.mlx_score_aware.loss import (
    _prepare_recon_pixel_weight,
    _weighted_recon,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")

_LANE = "lane_recon_pixel_weight_canonical_channel_20260531"


def _tiny_bundle(
    *,
    recon_pixel_weight=None,
    recon_pixel_weight_normalize: str = "mean",
    num_pairs: int = 4,
    seed: int = 0,
) -> RendererBundle:
    """A tiny pure-reconstruction Dreamer bundle (distill OFF) with random targets.

    Random non-zero targets so the recon term is genuinely non-trivial (a zeros
    target would make every weighting trivially equal). distill=0 keeps the test
    fast + isolates the recon channel.
    """
    import mlx.core as mx

    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_pairs=num_pairs,
        num_groups=2,
        num_categories=4,
        decoder_latent_dim=8,
        base_channels=4,
        eval_size=(384, 512),
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    rng = np.random.default_rng(seed)
    t0 = mx.array(rng.random((num_pairs, 384, 512, 3)).astype(np.float32))
    t1 = mx.array(rng.random((num_pairs, 384, 512, 3)).astype(np.float32))
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=0.0,
        recon_pixel_weight=recon_pixel_weight,
        recon_pixel_weight_normalize=recon_pixel_weight_normalize,
    )


# --------------------------------------------------------------------------- #
# default OFF = byte-identical to uniform recon
# --------------------------------------------------------------------------- #


@mlx_only
def test_default_off_is_byte_identical_uniform_recon() -> None:
    """recon_pixel_weight=None reproduces the EXACT uniform recon (Catalog #290)."""
    import mlx.core as mx

    bundle = _tiny_bundle(recon_pixel_weight=None)
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    # Recompute the canonical uniform recon by hand and require EXACT match.
    rgb_0, rgb_1 = _decode(bundle, idx)
    gt_0 = bundle.target_rgb_0[idx]
    gt_1 = bundle.target_rgb_1[idx]
    uniform = mx.mean((rgb_0 - gt_0) ** 2) + mx.mean((rgb_1 - gt_1) ** 2)
    mx.eval(uniform)
    assert float(parts["recon"].item()) == pytest.approx(float(uniform.item()), rel=0, abs=0)


def _decode(bundle, idx):
    from tac.substrates._shared.mlx_score_aware.loss import decode_frames_nhwc01

    return decode_frames_nhwc01(bundle, idx)


# --------------------------------------------------------------------------- #
# the channel ACTUALLY changes the loss (headline no-op guard)
# --------------------------------------------------------------------------- #


@mlx_only
def test_non_uniform_weight_changes_recon_loss() -> None:
    """A non-uniform weight map produces a DIFFERENT recon loss than uniform.

    This is the NO-FAKE headline guard: it FAILS if the weighted branch silently
    reverts to the uniform mean (a no-op channel). Computed on the SAME decoded
    frames so the ONLY difference is the weighting (the dreamer renderer is
    stochastic — comparing two decodes would let sampling noise masquerade as
    the channel's effect, defeating the no-op guard).
    """
    import mlx.core as mx

    rng = np.random.default_rng(7)
    # GT and a degraded "render": left half has LARGER error than right half so
    # a left-heavy weight raises the loss and a right-heavy weight lowers it
    # relative to uniform — the weighting must move the number BOTH directions.
    gt = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))
    err = np.zeros((4, 384, 512, 3), dtype=np.float32)
    err[:, :, :256, :] = 0.5  # big error on the LEFT half
    err[:, :, 256:, :] = 0.05  # small error on the RIGHT half
    rgb = mx.array((np.array(gt) + err).astype(np.float32))

    bundle = _tiny_bundle(recon_pixel_weight_normalize="mean")
    uniform = float(mx.mean((rgb - gt) ** 2).item())

    w_left = np.ones((1, 384, 512, 1), dtype=np.float32)
    w_left[:, :, :256, :] = 5.0
    w_left[:, :, 256:, :] = 0.2  # emphasize the high-error LEFT half
    w_right = np.ones((1, 384, 512, 1), dtype=np.float32)
    w_right[:, :, :256, :] = 0.2
    w_right[:, :, 256:, :] = 5.0  # emphasize the low-error RIGHT half

    recon_left = float(_weighted_recon(bundle, rgb, gt, mx.array(w_left)).item())
    recon_right = float(_weighted_recon(bundle, rgb, gt, mx.array(w_right)).item())
    # Not NaN, all positive.
    assert recon_left > 0.0 and recon_right > 0.0 and uniform > 0.0
    # The channel is a REAL re-weight: emphasizing the high-error region RAISES
    # the loss above uniform; emphasizing the low-error region LOWERS it.
    assert recon_left > uniform > recon_right, (
        f"recon_pixel_weight is a no-op or inverted: left={recon_left} uniform={uniform} right={recon_right}"
    )


@mlx_only
def test_constant_weight_with_mean_normalize_matches_uniform() -> None:
    """A CONSTANT weight under 'mean' normalization recovers the uniform recon.

    Sanity: w(x)=c constant => mean(c*sq)/mean(c) == mean(sq). Proves the
    normalization is a true scale-preserving re-distribution (not an arbitrary
    rescale). Compared on the SAME decode (the dreamer renderer is stochastic
    in forward_training, so two separate decodes diverge by sampling noise — we
    isolate the WEIGHTING by reducing one decode two ways via _weighted_recon).
    """
    import mlx.core as mx

    rng = np.random.default_rng(11)
    rgb_0 = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))
    rgb_1 = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))
    gt_0 = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))
    gt_1 = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))

    bundle = _tiny_bundle(recon_pixel_weight_normalize="mean")
    # Uniform reference (the default-OFF branch math).
    uniform = mx.mean((rgb_0 - gt_0) ** 2) + mx.mean((rgb_1 - gt_1) ** 2)
    # Constant weight via the channel helper, on the SAME frames.
    w = mx.full((1, 384, 512, 1), 3.7)
    weighted = _weighted_recon(bundle, rgb_0, gt_0, w) + _weighted_recon(bundle, rgb_1, gt_1, w)
    mx.eval(uniform, weighted)
    assert float(weighted.item()) == pytest.approx(float(uniform.item()), rel=1e-5)


@mlx_only
def test_none_normalize_applies_raw_scale() -> None:
    """recon_pixel_weight_normalize='none' applies the raw (unnormalized) map.

    Compared on the SAME decoded frames (stochastic-renderer-safe): raw weight
    2.0 everywhere scales the recon exactly 2x vs the uniform mean.
    """
    import mlx.core as mx

    rng = np.random.default_rng(3)
    rgb = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))
    gt = mx.array(rng.random((4, 384, 512, 3)).astype(np.float32))

    bundle_mean = _tiny_bundle(recon_pixel_weight_normalize="mean")
    bundle_none = _tiny_bundle(recon_pixel_weight_normalize="none")
    uniform = mx.mean((rgb - gt) ** 2)
    w = mx.full((1, 384, 512, 1), 2.0)
    raw = _weighted_recon(bundle_none, rgb, gt, w)
    # Sanity: 'mean' normalize on the same constant map == uniform.
    norm = _weighted_recon(bundle_mean, rgb, gt, w)
    mx.eval(uniform, raw, norm)
    assert float(raw.item()) == pytest.approx(2.0 * float(uniform.item()), rel=1e-5)
    assert float(norm.item()) == pytest.approx(float(uniform.item()), rel=1e-5)


# --------------------------------------------------------------------------- #
# shape coercion + validation
# --------------------------------------------------------------------------- #


@mlx_only
@pytest.mark.parametrize(
    "shape,expected_last",
    [((384, 512), 1), ((384, 512, 1), 1), ((384, 512, 3), 3), ((1, 384, 512, 1), 1)],
)
def test_prepare_weight_accepts_canonical_shapes(shape, expected_last) -> None:
    import mlx.core as mx

    bundle = _tiny_bundle(recon_pixel_weight=mx.ones(shape))
    prepared = _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))
    assert prepared.ndim == 4
    assert prepared.shape[0] == 1
    assert prepared.shape[1:3] == (384, 512)
    assert prepared.shape[3] == expected_last


@mlx_only
def test_prepare_weight_indexes_per_pair_and_frame_maps() -> None:
    import mlx.core as mx

    weight_np = np.ones((4, 2, 384, 512, 1), dtype=np.float32)
    weight_np[2, 1, :, :, 0] = 7.0
    weight_np[0, 1, :, :, 0] = 3.0
    bundle = _tiny_bundle(recon_pixel_weight=mx.array(weight_np), num_pairs=4)
    idx = mx.array([2, 0], dtype=mx.int32)

    prepared = _prepare_recon_pixel_weight(
        bundle,
        (2, 384, 512, 3),
        idx=idx,
        frame_index=1,
    )
    mx.eval(prepared)
    assert prepared.shape == (2, 384, 512, 1)
    assert float(prepared[0, 0, 0, 0].item()) == pytest.approx(7.0)
    assert float(prepared[1, 0, 0, 0].item()) == pytest.approx(3.0)


@mlx_only
def test_prepare_weight_indexes_per_pair_shared_frame_maps() -> None:
    import mlx.core as mx

    weight_np = np.ones((4, 384, 512, 1), dtype=np.float32)
    weight_np[3, :, :, 0] = 5.0
    bundle = _tiny_bundle(recon_pixel_weight=mx.array(weight_np), num_pairs=4)

    prepared = _prepare_recon_pixel_weight(
        bundle,
        (1, 384, 512, 3),
        idx=mx.array([3], dtype=mx.int32),
        frame_index=0,
    )
    mx.eval(prepared)
    assert prepared.shape == (1, 384, 512, 1)
    assert float(prepared[0, 0, 0, 0].item()) == pytest.approx(5.0)


@mlx_only
def test_prepare_weight_accepts_lazy_provider_batch_slices() -> None:
    import mlx.core as mx

    class Provider:
        def recon_pixel_weight_for_batch(self, *, idx, frame_shape, frame_index):
            b, h, w, _c = frame_shape
            scale = 3.0 if frame_index == 1 else 1.0
            values = mx.reshape(idx.astype(mx.float32) + scale, (int(b), 1, 1, 1))
            return mx.broadcast_to(values, (int(b), int(h), int(w), 1))

    bundle = _tiny_bundle(recon_pixel_weight=Provider(), num_pairs=4)
    prepared = _prepare_recon_pixel_weight(
        bundle,
        (2, 384, 512, 3),
        idx=mx.array([2, 0], dtype=mx.int32),
        frame_index=1,
    )
    mx.eval(prepared)

    assert prepared.shape == (2, 384, 512, 1)
    assert float(prepared[0, 0, 0, 0].item()) == pytest.approx(5.0)
    assert float(prepared[1, 0, 0, 0].item()) == pytest.approx(3.0)


@mlx_only
def test_prepare_weight_rejects_spatial_mismatch() -> None:
    import mlx.core as mx

    bundle = _tiny_bundle(recon_pixel_weight=mx.ones((100, 200)))
    with pytest.raises(MlxScoreAwareHarnessError, match="must match decoded frame"):
        _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))


@mlx_only
def test_prepare_weight_rejects_negative() -> None:
    import mlx.core as mx

    w = mx.ones((384, 512))
    w = w - 2.0  # all -1
    bundle = _tiny_bundle(recon_pixel_weight=w)
    with pytest.raises(MlxScoreAwareHarnessError, match="non-negative"):
        _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))


@mlx_only
def test_prepare_weight_rejects_non_finite() -> None:
    import mlx.core as mx

    w = np.ones((384, 512), dtype=np.float32)
    w[0, 0] = np.inf
    bundle = _tiny_bundle(recon_pixel_weight=mx.array(w))
    with pytest.raises(MlxScoreAwareHarnessError, match="finite"):
        _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))


@mlx_only
def test_prepare_weight_rejects_zero_total_mass() -> None:
    import mlx.core as mx

    bundle = _tiny_bundle(recon_pixel_weight=mx.zeros((384, 512)))
    with pytest.raises(MlxScoreAwareHarnessError, match="positive total mass"):
        _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))


@mlx_only
def test_prepare_weight_rejects_bad_ndim() -> None:
    import mlx.core as mx

    bundle = _tiny_bundle(recon_pixel_weight=mx.ones((1, 2, 2, 384, 512, 1)))
    with pytest.raises(MlxScoreAwareHarnessError):
        _prepare_recon_pixel_weight(bundle, (4, 384, 512, 3))


@mlx_only
def test_prepare_weight_rejects_dynamic_map_without_indices() -> None:
    import mlx.core as mx

    bundle = _tiny_bundle(
        recon_pixel_weight=mx.ones((4, 2, 384, 512, 1)),
        num_pairs=4,
    )
    with pytest.raises(MlxScoreAwareHarnessError, match="requires pair indices"):
        _prepare_recon_pixel_weight(bundle, (2, 384, 512, 3))


def test_bundle_rejects_bad_normalize_mode() -> None:
    """Validation does not require MLX (pure dataclass __post_init__)."""
    # A bundle with a bogus normalize mode is refused at construction.
    with pytest.raises(MlxScoreAwareHarnessError, match="recon_pixel_weight_normalize"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=1,
            forward_convention="call_b2chw_255",
            recon_pixel_weight_normalize="bogus",
        )


# --------------------------------------------------------------------------- #
# gradient flows to renderer, NOT to the weight
# --------------------------------------------------------------------------- #


@mlx_only
def test_gradient_flows_to_renderer_not_weight() -> None:
    """The weighted recon yields a finite, non-zero renderer gradient; the weight
    carries NO gradient (it is mx.stop_gradient)."""
    import mlx.core as mx
    import mlx.nn as mlx_nn

    w = np.ones((384, 512), dtype=np.float32)
    w[:, :256] = 4.0
    bundle = _tiny_bundle(recon_pixel_weight=mx.array(w), recon_pixel_weight_normalize="mean", seed=5)
    idx = mx.array([0, 1], dtype=mx.int32)

    # Canonical adapter pattern: nn.value_and_grad(model, closure) where the
    # closure reads bundle.model (per adapter.py train_step).
    def _loss_fn_inner():
        total, _ = score_aware_loss(bundle, idx)
        return total

    loss_and_grad_fn = mlx_nn.value_and_grad(bundle.model, _loss_fn_inner)
    val, grads = loss_and_grad_fn()
    mx.eval(val, grads)
    assert float(val.item()) == float(val.item())  # not NaN
    # At least one renderer parameter gradient is non-zero (the channel does
    # produce a real training signal).
    flat = []

    def _collect(d):
        for v in d.values():
            if isinstance(v, dict):
                _collect(v)
            elif hasattr(v, "shape"):
                flat.append(v)

    _collect(grads)
    assert flat, "no renderer gradients collected"
    total_abs = sum(float(mx.sum(mx.abs(g)).item()) for g in flat)
    assert total_abs > 0.0, "renderer gradient is identically zero (channel inert)"


@mlx_only
def test_score_loss_gradient_is_blocked_to_recon_pixel_weight() -> None:
    """The full score_aware_loss path must stop gradients into the weight map."""
    import mlx.core as mx

    class _FixedRenderer:
        def __call__(self, idx):
            b = idx.shape[0]
            # Spatially nonuniform output so removing stop_gradient would create
            # a nonzero d(loss)/d(weight) under mean normalization.
            x = mx.linspace(0.0, 255.0, 512)
            y = mx.linspace(0.0, 255.0, 384).reshape(384, 1)
            plane = ((x + y) / 2.0).astype(mx.float32)
            frame0 = mx.broadcast_to(plane.reshape(1, 1, 384, 512), (b, 3, 384, 512))
            frame1 = mx.broadcast_to((255.0 - plane).reshape(1, 1, 384, 512), (b, 3, 384, 512))
            return mx.stack([frame0, frame1], axis=1)

    rng = np.random.default_rng(19)
    targets0 = mx.array(rng.random((2, 384, 512, 3)).astype(np.float32))
    targets1 = mx.array(rng.random((2, 384, 512, 3)).astype(np.float32))
    idx = mx.array([0, 1], dtype=mx.int32)

    def _loss_for_weight(weight):
        bundle = RendererBundle(
            model=_FixedRenderer(),
            target_rgb_0=targets0,
            target_rgb_1=targets1,
            num_pairs=2,
            forward_convention="call_b2chw_255",
            distillation_weight=0.0,
            recon_pixel_weight=weight,
            recon_pixel_weight_normalize="mean",
        )
        total, _ = score_aware_loss(bundle, idx)
        return total

    grad = mx.grad(_loss_for_weight)(mx.ones((384, 512), dtype=mx.float32))
    mx.eval(grad)
    assert float(mx.sum(mx.abs(grad)).item()) == pytest.approx(0.0, abs=0.0)


@mlx_only
def test_weighted_recon_helper_matches_manual_math() -> None:
    """_weighted_recon == mean(w*sq)/mean(w) under 'mean' normalize."""
    import mlx.core as mx

    rng = np.random.default_rng(0)
    rgb = mx.array(rng.random((2, 8, 8, 3)).astype(np.float32))
    gt = mx.array(rng.random((2, 8, 8, 3)).astype(np.float32))
    w = mx.array(rng.random((1, 8, 8, 1)).astype(np.float32) + 0.1)
    bundle = _tiny_bundle(recon_pixel_weight_normalize="mean")
    got = _weighted_recon(bundle, rgb, gt, w)
    sq = (rgb - gt) ** 2
    expect = mx.mean(w * sq) / (mx.mean(w) + 1e-12)
    mx.eval(got, expect)
    assert float(got.item()) == pytest.approx(float(expect.item()), rel=1e-6)
