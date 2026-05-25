# SPDX-License-Identifier: MIT
"""ARCH-4a SegNet/EfficientNet-B2-UNet portable primitive tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tac.portable_primitives import (
    SEGNET_CLASSES,
    SEGNET_DECODER_CHANNELS,
    SEGNET_ENCODER_CHANNELS,
    SEGNET_ENCODER_STAGE_SPEC,
    SEGNET_MODEL_INPUT_HW,
    SEGNET_MODEL_INPUT_SIZE,
    PortableEfficientNetB2Backbone,
    PortableMBConvBlock,
    PortableSegmentationHead,
    PortableSegNet,
    PortableSqueezeExcite,
    PortableUnetDecoder,
    PortableUnetDecoderBlock,
    is_mlx_available,
    is_pytorch_available,
)
from tac.portable_primitives.tensor import from_numpy, to_numpy

REPO = Path(__file__).resolve().parents[4]
ATOL_FP32 = 5e-3
RTOL_FP32 = 5e-3


def _seeded_input(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_segnet_constants_match_upstream_shape_contract() -> None:
    assert SEGNET_CLASSES == 5
    assert SEGNET_MODEL_INPUT_SIZE == (512, 384)
    assert SEGNET_MODEL_INPUT_HW == (384, 512)
    assert SEGNET_ENCODER_CHANNELS == (3, 16, 24, 48, 120, 352)
    assert SEGNET_DECODER_CHANNELS == (256, 128, 64, 32, 16)


def test_segnet_decoder_shapes_match_live_smp_unet_when_available() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("timm")
    sys.path.insert(0, str(REPO / "upstream"))
    try:
        from modules import SegNet
    finally:
        try:
            sys.path.remove(str(REPO / "upstream"))
        except ValueError:
            pass

    model = SegNet()
    assert tuple(model.encoder.out_channels) == SEGNET_ENCODER_CHANNELS
    conv1_shapes = [
        (block.conv1[0].in_channels, block.conv1[0].out_channels)
        for block in model.decoder.blocks
    ]
    assert conv1_shapes == [(472, 256), (304, 128), (152, 64), (80, 32), (32, 16)]
    assert model.segmentation_head[0].in_channels == SEGNET_DECODER_CHANNELS[-1]
    assert model.segmentation_head[0].out_channels == SEGNET_CLASSES


pytestmark = pytest.mark.skipif(
    not (is_mlx_available() and is_pytorch_available()),
    reason="Both MLX + PyTorch must be installed for cross-backend equivalence tests",
)


def test_squeeze_excite_forward_equivalence() -> None:
    x_np = _seeded_input((2, 8, 6, 6), seed=10)
    se_mlx = PortableSqueezeExcite(8, squeeze_channels=2, backend="mlx", seed=3)
    se_pt = PortableSqueezeExcite(8, squeeze_channels=2, backend="pytorch", seed=3)

    y_mlx = to_numpy(se_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(se_pt(from_numpy(x_np, "pytorch")), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_mbconv_block_forward_equivalence_and_shape() -> None:
    x_np = _seeded_input((2, 8, 8, 8), seed=11)
    block_mlx = PortableMBConvBlock(8, 8, backend="mlx", expand_ratio=2, seed=4)
    block_pt = PortableMBConvBlock(8, 8, backend="pytorch", expand_ratio=2, seed=4)

    y_mlx = to_numpy(block_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(block_pt(from_numpy(x_np, "pytorch")), "pytorch")

    assert y_mlx.shape == x_np.shape
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_unet_decoder_block_forward_equivalence() -> None:
    x_np = _seeded_input((2, 16, 4, 4), seed=12)
    skip_np = _seeded_input((2, 8, 8, 8), seed=13)
    block_mlx = PortableUnetDecoderBlock(16, 8, 10, backend="mlx", seed=5)
    block_pt = PortableUnetDecoderBlock(16, 8, 10, backend="pytorch", seed=5)

    y_mlx = to_numpy(
        block_mlx(from_numpy(x_np, "mlx"), from_numpy(skip_np, "mlx")),
        "mlx",
    )
    y_pt = to_numpy(
        block_pt(from_numpy(x_np, "pytorch"), from_numpy(skip_np, "pytorch")),
        "pytorch",
    )

    assert y_mlx.shape == (2, 10, 8, 8)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_unet_decoder_and_segmentation_head_shape() -> None:
    features_np = [
        _seeded_input((1, 3, 64, 64), seed=20),
        _seeded_input((1, 16, 32, 32), seed=21),
        _seeded_input((1, 24, 16, 16), seed=22),
        _seeded_input((1, 48, 8, 8), seed=23),
        _seeded_input((1, 120, 4, 4), seed=24),
        _seeded_input((1, 352, 2, 2), seed=25),
    ]

    decoder = PortableUnetDecoder(backend="mlx", seed=6)
    head = PortableSegmentationHead(backend="mlx", seed=7)
    decoded = decoder([from_numpy(arr, "mlx") for arr in features_np])
    logits = head(decoded)

    logits_np = to_numpy(logits, "mlx")
    assert logits_np.shape == (1, SEGNET_CLASSES, 64, 64)


def test_efficientnet_b2_backbone_feature_contract_shape() -> None:
    x_np = _seeded_input((1, 3, 64, 64), seed=30)
    backbone = PortableEfficientNetB2Backbone(backend="mlx", seed=8)

    features = backbone(from_numpy(x_np, "mlx"))
    shapes = [to_numpy(feature, "mlx").shape for feature in features]

    assert sum(1 for spec in SEGNET_ENCODER_STAGE_SPEC if spec[6]) == 5
    assert shapes == [
        (1, 3, 64, 64),
        (1, 16, 32, 32),
        (1, 24, 16, 16),
        (1, 48, 8, 8),
        (1, 120, 4, 4),
        (1, 352, 2, 2),
    ]


def test_portable_segnet_wrapper_shape_and_distortion_identity() -> None:
    model = PortableSegNet(backend="mlx", seed=9)
    preprocessed = model.preprocess_input(
        from_numpy(_seeded_input((1, 2, 3, 32, 48), seed=31), "mlx")
    )
    logits = model.forward_3d(
        from_numpy(_seeded_input((1, 3, 64, 64), seed=32), "mlx")
    )
    distortion = model.compute_distortion(logits, logits)

    assert to_numpy(preprocessed, "mlx").shape == (1, 3, *SEGNET_MODEL_INPUT_HW)
    assert to_numpy(logits, "mlx").shape == (1, SEGNET_CLASSES, 64, 64)
    np.testing.assert_allclose(to_numpy(distortion, "mlx"), np.zeros((1,), np.float32))


# ---------------------------------------------------------------------------
# MLX-ARCH-4 FULL ASSEMBLY: extended canonical-input + cross-backend tests
# ---------------------------------------------------------------------------
#
# These extend the sister scaffold tests above with: (a) the canonical 384x512
# input full feature shape contract (vs the 64x64 sister scaffold test);
# (b) the SegNet 5D-input -> logits contract on BOTH backends; (c) the
# preprocess_input + compute_distortion contracts on PyTorch (the canonical
# contest-axis path per Catalog #1 + #192 + #317); (d) per-batch distortion
# shape.  Per Carmack MVP-first 5-step + ARCH-3 PortablePoseNet precedent:
# full upstream-weight numeric parity defers to ARCH-5 paired ground-truth.


def test_encoder_stage_spec_canonical_emit_count_and_channels() -> None:
    """7-stage spec emits 5 features matching SEGNET_ENCODER_CHANNELS[1:]."""
    assert len(SEGNET_ENCODER_STAGE_SPEC) == 7
    emits = sum(1 for spec in SEGNET_ENCODER_STAGE_SPEC if spec[6])
    assert emits == 5
    for spec in SEGNET_ENCODER_STAGE_SPEC:
        assert len(spec) == 7
    emitting_stages = [s for s in SEGNET_ENCODER_STAGE_SPEC if s[6]]
    emit_channels = tuple(s[2] for s in emitting_stages)
    assert emit_channels == SEGNET_ENCODER_CHANNELS[1:], (
        f"emit channel chain drift: {emit_channels} != {SEGNET_ENCODER_CHANNELS[1:]}"
    )


def test_encoder_backbone_canonical_384x512_input_matches_upstream_shapes() -> None:
    """At canonical 384x512 input, encoder returns exact upstream smp.Unet shapes."""
    x_np = _seeded_input((1, 3, 384, 512), seed=50)
    backbone_mlx = PortableEfficientNetB2Backbone(backend="mlx", seed=0)
    backbone_pt = PortableEfficientNetB2Backbone(backend="pytorch", seed=0)
    features_mlx = backbone_mlx(from_numpy(x_np, "mlx"))
    features_pt = backbone_pt(from_numpy(x_np, "pytorch"))

    expected = [
        (1, 3, 384, 512),
        (1, 16, 192, 256),
        (1, 24, 96, 128),
        (1, 48, 48, 64),
        (1, 120, 24, 32),
        (1, 352, 12, 16),
    ]
    mlx_shapes = [tuple(to_numpy(f, "mlx").shape) for f in features_mlx]
    pt_shapes = [tuple(f.shape) for f in features_pt]
    assert mlx_shapes == expected, f"MLX shape drift: {mlx_shapes} != {expected}"
    assert pt_shapes == expected, f"PT shape drift: {pt_shapes} != {expected}"


def test_segnet_full_forward_5d_logits_shape_pytorch() -> None:
    """SegNet 5D (B,T,3,H,W) -> (B,5,384,512) logits on PyTorch backend."""
    import torch

    x_np = _seeded_input((1, 2, 3, 256, 512), seed=60) * 50.0
    segnet = PortableSegNet(backend="pytorch", seed=0)
    logits = segnet(torch.from_numpy(x_np))
    assert tuple(logits.shape) == (1, SEGNET_CLASSES, 384, 512)


def test_segnet_full_forward_5d_logits_shape_mlx() -> None:
    """SegNet 5D (B,T,3,H,W) -> (B,5,384,512) logits on MLX backend (parity)."""
    x_5d_np = _seeded_input((1, 2, 3, 256, 512), seed=61) * 50.0
    segnet_mlx = PortableSegNet(backend="mlx", seed=0)
    logits = segnet_mlx(from_numpy(x_5d_np, "mlx"))
    assert tuple(to_numpy(logits, "mlx").shape) == (1, SEGNET_CLASSES, 384, 512)


def test_segnet_preprocess_input_slices_last_frame_pytorch() -> None:
    """preprocess_input on PyTorch mirrors upstream bilinear-to-(384,512)."""
    import torch

    segnet = PortableSegNet(backend="pytorch", seed=0)
    x_np = _seeded_input((2, 3, 3, 192, 256), seed=62)
    pre = segnet.preprocess_input(torch.from_numpy(x_np))
    assert tuple(pre.shape) == (2, 3, 384, 512)


def test_segnet_compute_distortion_pytorch_in_unit_interval() -> None:
    """PyTorch compute_distortion returns per-batch argmax-disagreement in [0,1]."""
    import torch

    segnet = PortableSegNet(backend="pytorch", seed=0)
    x_np = _seeded_input((1, 2, 3, 256, 512), seed=70) * 50.0
    out1 = segnet(torch.from_numpy(x_np))
    out2 = out1.clone()
    dist_zero = segnet.compute_distortion(out1, out2)
    assert tuple(dist_zero.shape) == (1,)
    assert float(dist_zero[0]) == 0.0
    x2_np = _seeded_input((1, 2, 3, 256, 512), seed=71) * 50.0
    out3 = segnet(torch.from_numpy(x2_np))
    dist_nonzero = segnet.compute_distortion(out1, out3)
    assert tuple(dist_nonzero.shape) == (1,)
    assert 0.0 <= float(dist_nonzero[0]) <= 1.0


def test_segnet_distortion_returns_per_batch_scalar() -> None:
    """compute_distortion returns one scalar per batch element (PyTorch)."""
    import torch

    segnet = PortableSegNet(backend="pytorch", seed=0)
    x_np = _seeded_input((3, 2, 3, 192, 256), seed=73) * 50.0
    x_np2 = _seeded_input((3, 2, 3, 192, 256), seed=74) * 50.0
    out1 = segnet(torch.from_numpy(x_np))
    out2 = segnet(torch.from_numpy(x_np2))
    dist = segnet.compute_distortion(out1, out2)
    assert tuple(dist.shape) == (3,)
    for v in dist:
        assert 0.0 <= float(v) <= 1.0


def test_segnet_canonical_input_hw_path_matches_upstream_contract_pytorch() -> None:
    """Canonical (B, T, 3, 384, 512) input flows through preprocess to logits."""
    import torch

    segnet = PortableSegNet(backend="pytorch", seed=0)
    x_np = _seeded_input((1, 2, 3, *SEGNET_MODEL_INPUT_HW), seed=72)
    logits = segnet(torch.from_numpy(x_np))
    assert tuple(logits.shape) == (1, SEGNET_CLASSES, *SEGNET_MODEL_INPUT_HW)
