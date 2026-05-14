# SPDX-License-Identifier: MIT
"""Regression tests for Lane Omega Fisher profiler mask decoding."""
from __future__ import annotations

import torch

from experiments.profile_hessian_per_weight import _gray_mask_to_class_ids


def test_gray_mask_to_class_ids_decodes_av1_luma_values() -> None:
    gray = torch.tensor(
        [
            [0, 63, 126, 189, 252],
            [1, 62, 127, 191, 255],
        ],
        dtype=torch.uint8,
    )
    decoded = _gray_mask_to_class_ids(gray)
    assert decoded.dtype == torch.long
    assert decoded.tolist() == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]


def test_gray_mask_to_class_ids_clamps_codec_noise() -> None:
    gray = torch.tensor([0, 255, 250], dtype=torch.uint8)
    decoded = _gray_mask_to_class_ids(gray)
    assert decoded.min().item() >= 0
    assert decoded.max().item() <= 4
