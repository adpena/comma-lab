# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "measure_segnet_argmax_stable_interior.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("measure_segnet_argmax_stable_interior", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"SABOR tool optional dependency missing: {exc.name}")
    return module


def test_pair_hwc_to_tchw_preserves_upstream_pair_contract():
    tool = _load_tool_module()
    h, w = tool.camera_size[1], tool.camera_size[0]
    first = torch.zeros((h, w, 3), dtype=torch.uint8)
    last = torch.full((h, w, 3), 7, dtype=torch.uint8)

    pair = tool._pair_hwc_to_tchw([first, last])

    assert tuple(pair.shape) == (tool.seq_len, 3, h, w)
    assert pair.dtype == torch.float32
    assert pair[0, :, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert pair[1, :, 0, 0].tolist() == [7.0, 7.0, 7.0]


def test_pair_hwc_to_tchw_rejects_non_uint8_rgb_decode():
    tool = _load_tool_module()
    h, w = tool.camera_size[1], tool.camera_size[0]
    bad = torch.zeros((h, w, 3), dtype=torch.float32)

    with pytest.raises(TypeError, match="must be uint8"):
        tool._pair_hwc_to_tchw([bad, bad])


def test_segnet_logits_from_pair_uses_preprocess_input_5d_contract():
    tool = _load_tool_module()

    class FakeSegNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocess_shape = None
            self.forward_shape = None

        def preprocess_input(self, x):
            self.preprocess_shape = tuple(x.shape)
            h, w = tool.segnet_model_input_size[1], tool.segnet_model_input_size[0]
            return torch.ones((x.shape[0], 3, h, w), dtype=x.dtype)

        def forward(self, x):
            self.forward_shape = tuple(x.shape)
            return torch.zeros((x.shape[0], 5, x.shape[-2], x.shape[-1]), dtype=x.dtype)

    scorer = FakeSegNet()
    pair = torch.zeros((tool.seq_len, 3, 8, 10), dtype=torch.float32)

    logits = tool._segnet_logits_from_pair(scorer, pair)

    assert scorer.preprocess_shape == (1, tool.seq_len, 3, 8, 10)
    assert scorer.forward_shape == (1, 3, tool.segnet_model_input_size[1], tool.segnet_model_input_size[0])
    assert tuple(logits.shape) == (5, tool.segnet_model_input_size[1], tool.segnet_model_input_size[0])


def test_empirical_stability_reports_mean_and_all_samples_separately():
    tool = _load_tool_module()
    argmax = torch.tensor([[0, 0], [1, 1]], dtype=torch.uint8)
    stable_a = torch.tensor([[True, False], [True, True]])
    stable_b = torch.tensor([[True, True], [False, True]])

    rec = tool._empirical_stability_from_masks(argmax, [stable_a, stable_b], n_classes=3)

    assert rec["stable_fraction_mean_per_perturbation"] == pytest.approx(0.75)
    assert rec["stable_fraction_all_samples"] == pytest.approx(0.50)
    assert rec["per_class_stable_fraction_mean_per_perturbation"][0] == pytest.approx(0.75)
    assert rec["per_class_stable_fraction_mean_per_perturbation"][1] == pytest.approx(0.75)
    assert rec["per_class_stable_fraction_mean_per_perturbation"][2] != rec[
        "per_class_stable_fraction_mean_per_perturbation"
    ][2]

