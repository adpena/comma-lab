from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).with_name("run_hm1_frame_dim_curve.py")


def _module():
    spec = importlib.util.spec_from_file_location("ddm_hm1_frame_dim_curve_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stratified_selection_is_reproducible_random_n120() -> None:
    module = _module()
    first = module.stratified_frame_ids()
    second = module.stratified_frame_ids()
    assert first.tolist() == second.tolist()
    assert len(first) == 120
    assert len(set(first.tolist())) == 120
    assert first.tolist() != list(range(120))
    for stratum in range(10):
        assert sum(stratum * 60 <= int(value) < (stratum + 1) * 60 for value in first) == 12


def test_dimension_state_slices_only_frame_conditioning_inputs() -> None:
    module = _module()
    source = {
        "frame_embed.weight": torch.arange(48).reshape(6, 8),
        "frame_shift.weight": torch.arange(32).reshape(4, 8),
        "frame_scale.weight": torch.arange(32).reshape(4, 8) + 100,
        "frame_shift.bias": torch.arange(4),
        "conv_a.weight": torch.arange(12).reshape(3, 4),
    }
    result = module._dimension_state(source, (1, 4, 7))
    assert result["frame_embed.weight"].shape == (6, 3)
    assert result["frame_shift.weight"].shape == (4, 3)
    assert result["frame_scale.weight"].shape == (4, 3)
    assert result["frame_embed.weight"][0].tolist() == [1, 4, 7]
    assert torch.equal(result["frame_shift.bias"], source["frame_shift.bias"])
    assert torch.equal(result["conv_a.weight"], source["conv_a.weight"])
    for key in result:
        assert result[key].data_ptr() != source[key].data_ptr()
