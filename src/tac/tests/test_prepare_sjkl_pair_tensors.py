from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "experiments" / "prepare_sjkl_pair_tensors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_sjkl_pair_tensors", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_renderer_target_slot_selects_even_fake1_frames() -> None:
    module = _load_module()
    frames = torch.arange(4 * 2 * 3 * 3, dtype=torch.float32).reshape(4, 2, 3, 3)

    out = module.normalize_renderer_target_slot(frames, target_slot=0)

    assert out.shape == (2, 3, 2, 3)
    assert torch.equal(out[0], frames[0].permute(2, 0, 1))
    assert torch.equal(out[1], frames[2].permute(2, 0, 1))


def test_normalize_renderer_target_slot_slices_before_contiguous_storage() -> None:
    module = _load_module()
    frames = torch.zeros(120, 8, 8, 3, dtype=torch.float32)

    out = module.normalize_renderer_target_slot(frames, target_slot=0, max_pairs=4)

    assert out.shape == (4, 3, 8, 8)
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()


def test_normalize_renderer_target_slot_rejects_unsupported_slot() -> None:
    module = _load_module()
    frames = torch.zeros(4, 2, 3, 3)

    with pytest.raises(ValueError, match="target_slot=0"):
        module.normalize_renderer_target_slot(frames, target_slot=1)


def test_build_gt_pairs_btchw() -> None:
    module = _load_module()
    frames = [
        torch.full((2, 3, 3), value, dtype=torch.uint8)
        for value in (1, 2, 3, 4)
    ]

    pairs = module.build_gt_pairs_btchw(frames, n_pairs=2)

    assert pairs.shape == (2, 2, 3, 2, 3)
    assert pairs.dtype == torch.uint8
    assert int(pairs[0, 0].sum()) == int(frames[0].permute(2, 0, 1).sum())
    assert int(pairs[1, 1].sum()) == int(frames[3].permute(2, 0, 1).sum())
