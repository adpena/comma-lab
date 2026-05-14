# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[3]
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


def _load_train_renderer():
    spec = importlib.util.spec_from_file_location(
        "_train_renderer_half_frame_noise_test",
        TRAIN_RENDERER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_half_frame_training_noise_uses_pair_index_basis() -> None:
    module = _load_train_renderer()
    masks = torch.arange(3 * 2 * 2, dtype=torch.long).reshape(3, 2, 2)

    mask_t, mask_t1 = module.training_mask_pair_from_index(
        masks,
        4,
        pair_index_basis="half_frame_pair_index",
    )

    assert torch.equal(mask_t, masks[2].unsqueeze(0))
    assert torch.equal(mask_t1, masks[2].unsqueeze(0))


def test_half_frame_training_noise_fails_closed_on_bad_pair_index() -> None:
    module = _load_train_renderer()
    masks = torch.zeros((3, 2, 2), dtype=torch.long)

    with pytest.raises(IndexError, match="half-frame mask pair index"):
        module.training_mask_pair_from_index(
            masks,
            6,
            pair_index_basis="half_frame_pair_index",
        )
