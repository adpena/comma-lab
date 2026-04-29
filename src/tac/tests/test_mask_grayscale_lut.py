"""Tests for the Selfcomp grayscale-LUT mask codec."""
from __future__ import annotations

import pytest
import torch

from tac.mask_grayscale_lut import (
    CLASS_TO_GRAY,
    LUT_DEFAULT_SIGMA,
    NUM_CLASSES,
    create_gaussian_softmax_lut,
    decode_grayscale_to_classes,
    encode_masks_grayscale,
)


def test_encode_all_classes() -> None:
    """encode_masks_grayscale produces the documented gray values per class."""
    class_ids = torch.tensor(
        [[[c for c in range(NUM_CLASSES)]]],
        dtype=torch.int64,
    )  # (1, 1, NUM_CLASSES)
    gray = encode_masks_grayscale(class_ids)
    assert gray.dtype == torch.uint8
    assert gray.shape == class_ids.shape
    expected = [CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)]
    assert gray[0, 0].tolist() == expected


def test_roundtrip() -> None:
    """encode -> decode is identity on legal class_ids."""
    g = torch.randint(0, NUM_CLASSES, (4, 8, 8), dtype=torch.int64)
    gray = encode_masks_grayscale(g)
    recovered = decode_grayscale_to_classes(gray)
    assert torch.equal(recovered, g)


def test_decode_noisy_still_correct() -> None:
    """Decoding survives ±10 noise on each pixel — half the gap between targets.

    Targets are [0, 64, 128, 192, 255] — the smallest gap is 63 (255-192).
    We jitter by up to ±10 gray levels, far below the half-gap, so nearest-
    neighbour decoding must still recover the original class.
    """
    g = torch.randint(0, NUM_CLASSES, (4, 8, 8), dtype=torch.int64)
    gray = encode_masks_grayscale(g).to(torch.int16)
    noise = torch.randint(-10, 11, gray.shape, dtype=torch.int16)
    noisy = (gray + noise).clamp(0, 255).to(torch.uint8)
    recovered = decode_grayscale_to_classes(noisy)
    assert torch.equal(recovered, g)


def test_lut_shape_sums_to_1() -> None:
    """LUT shape is (256, 5) and every row is a probability distribution."""
    lut = create_gaussian_softmax_lut(sigma=LUT_DEFAULT_SIGMA)
    assert lut.shape == (256, NUM_CLASSES)
    assert lut.dtype == torch.float32
    sums = lut.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_lut_argmax_matches_nearest_gray() -> None:
    """LUT argmax over gray rows matches the nearest-neighbour decoder.

    For every gray value v, the highest-probability class in lut[v] must be
    the closest CLASS_TO_GRAY target — otherwise the LUT silently disagrees
    with the contest decoder.
    """
    lut = create_gaussian_softmax_lut(sigma=LUT_DEFAULT_SIGMA)
    lut_argmax = lut.argmax(dim=1)
    gray_axis = torch.arange(256, dtype=torch.uint8)
    nn_classes = decode_grayscale_to_classes(gray_axis)
    assert torch.equal(lut_argmax.to(torch.int64), nn_classes)


def test_encode_rejects_out_of_range() -> None:
    """encode_masks_grayscale raises on class id >= NUM_CLASSES."""
    bad = torch.tensor([[[NUM_CLASSES]]], dtype=torch.int64)
    with pytest.raises(ValueError, match=r"class_ids must be in"):
        encode_masks_grayscale(bad)


def test_lut_rejects_zero_sigma() -> None:
    """create_gaussian_softmax_lut rejects sigma <= 0."""
    with pytest.raises(ValueError, match=r"sigma must be positive"):
        create_gaussian_softmax_lut(sigma=0.0)


def test_lut_accepts_custom_targets() -> None:
    """Custom learnable targets preserve LUT shape and argmax locations."""
    targets = torch.tensor([3.0, 252.0, 70.0, 185.0, 130.0])
    lut = create_gaussian_softmax_lut(sigma=LUT_DEFAULT_SIGMA, targets=targets)
    assert lut.shape == (256, NUM_CLASSES)
    assert torch.equal(lut[targets.long()].argmax(dim=1), torch.arange(NUM_CLASSES))


def test_lut_rejects_invalid_custom_targets() -> None:
    """Custom targets must be one finite in-range value per class."""
    with pytest.raises(ValueError, match=r"shape"):
        create_gaussian_softmax_lut(targets=torch.zeros(NUM_CLASSES + 1))
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        create_gaussian_softmax_lut(targets=torch.tensor([0.0, 255.0, -1.0, 192.0, 128.0]))
