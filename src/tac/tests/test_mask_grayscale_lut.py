# SPDX-License-Identifier: MIT
"""Tests for the Selfcomp grayscale-LUT mask codec."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tac.mask_grayscale_lut import (
    CLASS_TO_GRAY,
    LUT_DEFAULT_SIGMA,
    NUM_CLASSES,
    create_gaussian_softmax_lut,
    decode_grayscale_to_classes,
    encode_masks_grayscale,
    grayscale_to_probability_map,
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


def test_lut_matches_selfcomp_bell_softmax_formula() -> None:
    """Canonical LUT must match the public Selfcomp inflate formula exactly."""
    lut = create_gaussian_softmax_lut(sigma=LUT_DEFAULT_SIGMA)
    gray_axis = torch.arange(256, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor([CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)]).float()
    squared_diff = (gray_axis - targets.unsqueeze(0)) ** 2
    bell = torch.exp(-squared_diff / (2.0 * LUT_DEFAULT_SIGMA * LUT_DEFAULT_SIGMA))
    expected = torch.softmax(bell, dim=1)
    assert torch.allclose(lut, expected, atol=1e-7)
    # At exact class targets this analog prior is deliberately not one-hot.
    target_rows = lut[targets.long()]
    assert torch.all(target_rows.max(dim=1).values < 0.5)


def test_grayscale_to_probability_map_shapes() -> None:
    gray_2d = torch.tensor([[0, 64], [128, 255]], dtype=torch.uint8)
    probs_2d = grayscale_to_probability_map(gray_2d)
    assert probs_2d.shape == (NUM_CLASSES, 2, 2)
    assert torch.allclose(probs_2d.sum(dim=0), torch.ones(2, 2), atol=1e-6)

    gray_3d = gray_2d.unsqueeze(0).repeat(3, 1, 1)
    probs_3d = grayscale_to_probability_map(gray_3d)
    assert probs_3d.shape == (3, NUM_CLASSES, 2, 2)
    assert torch.allclose(probs_3d.sum(dim=1), torch.ones(3, 2, 2), atol=1e-6)

    gray_4d = gray_3d.view(1, 3, 2, 2)
    probs_4d = grayscale_to_probability_map(gray_4d)
    assert probs_4d.shape == (1, 3, NUM_CLASSES, 2, 2)
    assert torch.allclose(probs_4d.sum(dim=2), torch.ones(1, 3, 2, 2), atol=1e-6)


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


def _load_train_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "experiments" / filename
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_build_pair_tensors_uses_soft_lut(module) -> None:
    mask_classes = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 0], [1, 2]],
        ],
        dtype=torch.int64,
    )
    gt_frames = torch.zeros(2, 3, 4, 4)
    mask_pairs, gt_pairs = module._build_pair_tensors(mask_classes, gt_frames)
    expected = grayscale_to_probability_map(
        encode_masks_grayscale(mask_classes), sigma=15.0, channel_first=True
    ).view(1, 2, NUM_CLASSES, 2, 2)
    assert mask_pairs.shape == (1, 2, NUM_CLASSES, 2, 2)
    assert gt_pairs.shape == (1, 2, 3, 4, 4)
    assert torch.allclose(mask_pairs, expected, atol=1e-7)
    assert torch.all(mask_pairs.max(dim=2).values < 0.5)


def test_train_segmap_pair_tensors_use_soft_lut() -> None:
    module = _load_train_module("train_segmap.py", "train_segmap_under_test")
    _assert_build_pair_tensors_uses_soft_lut(module)


def test_train_segmap_film_canvas_pair_tensors_use_soft_lut() -> None:
    module = _load_train_module(
        "train_segmap_film_canvas.py", "train_segmap_film_canvas_under_test"
    )
    _assert_build_pair_tensors_uses_soft_lut(module)
