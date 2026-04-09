from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_v2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ShiftModel(nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x + self.shift


class FakeLoss:
    def __call__(self, fx, gx):
        base = float(fx.mean().item() - gx.mean().item())
        loss = torch.tensor(abs(base), dtype=torch.float32)
        pose = abs(base) + 1.0
        seg = abs(base) + 2.0
        return loss, pose, seg


class TrainPostfilterV2Tests(unittest.TestCase):
    def test_evaluate_model_pairs_averages_loss_pose_and_seg(self) -> None:
        mod = load_module()
        pairs = [
            torch.zeros(1, 2, 2, 2, 3, dtype=torch.uint8),
            torch.ones(1, 2, 2, 2, 3, dtype=torch.uint8),
        ]
        gt_pairs = [
            torch.zeros(1, 2, 2, 2, 3, dtype=torch.uint8),
            torch.zeros(1, 2, 2, 2, 3, dtype=torch.uint8),
        ]

        loss, pose, seg = mod.evaluate_model_pairs(
            ShiftModel(1.0),
            pairs,
            gt_pairs,
            eval_indices=[0, 1],
            loss_fn=FakeLoss(),
            device="cpu",
        )

        self.assertAlmostEqual(loss, 1.5, places=5)
        self.assertAlmostEqual(pose, 2.5, places=5)
        self.assertAlmostEqual(seg, 3.5, places=5)

    def test_quantize_state_dict_like_saved_int8_zeroes_tiny_values(self) -> None:
        mod = load_module()
        state = {
            "conv.weight": torch.tensor([0.00001, 1.0], dtype=torch.float32),
            "conv.bias": torch.tensor([0.0, -0.25], dtype=torch.float32),
        }

        quantized = mod.quantize_state_dict_like_saved_int8(state)

        self.assertEqual(set(quantized.keys()), set(state.keys()))
        self.assertEqual(float(quantized["conv.weight"][0]), 0.0)
        self.assertNotEqual(float(quantized["conv.weight"][1]), 0.0)
        self.assertAlmostEqual(float(quantized["conv.bias"][0]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
