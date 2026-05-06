"""Tests for Lane SH arithmetic SegMap inflate parity."""

from __future__ import annotations

import inspect

import pytest
import torch

from submissions.robust_current import inflate_segmap as standard
from submissions.robust_current import inflate_segmap_arithmetic as arith


def test_arithmetic_default_soft_lut_matches_standard_segmap_path() -> None:
    gray = torch.tensor(
        [
            [
                [0, 255, 64],
                [192, 128, 17],
            ]
        ],
        dtype=torch.uint8,
    )
    device = torch.device("cpu")

    got = arith._grayscale_to_mask_features(gray, device=device, mode="soft_lut")
    expected = standard._grayscale_to_mask_features(gray, device=device, mode="soft_lut")

    assert got.shape == (1, 5, 2, 3)
    assert torch.allclose(got, expected)
    assert torch.allclose(got.sum(dim=1), torch.ones(1, 2, 3))


def test_arithmetic_hard_onehot_mode_preserves_legacy_projection() -> None:
    gray = torch.tensor([[[0, 255, 64, 192, 128]]], dtype=torch.uint8)
    got = arith._grayscale_to_mask_features(
        gray, device=torch.device("cpu"), mode="onehot"
    )

    assert got.shape == (1, 5, 1, 5)
    assert torch.equal(got.argmax(dim=1), torch.tensor([[[0, 1, 2, 3, 4]]]))
    assert torch.all(got.sum(dim=1) == 1.0)


@pytest.mark.parametrize("alias", ["hard", "one_hot", "onehot", "hard_onehot"])
def test_arithmetic_grayscale_mode_aliases(alias: str) -> None:
    assert arith._normalize_grayscale_mode(alias) == "hard_onehot"


def test_arithmetic_rejects_unknown_grayscale_mode() -> None:
    with pytest.raises(RuntimeError, match="SEGMAP_GRAYSCALE_MODE"):
        arith._normalize_grayscale_mode("nearest")


def test_arithmetic_inflate_uses_bicubic_resize_for_standard_runtime_parity() -> None:
    source = inspect.getsource(arith.inflate)
    assert 'mode="bicubic"' in source
    assert 'mode="bilinear"' not in source
