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
    SEGNET_MODEL_INPUT_HW,
    SEGNET_MODEL_INPUT_SIZE,
    PortableMBConvBlock,
    PortableSegmentationHead,
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
