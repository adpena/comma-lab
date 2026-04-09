from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_saliency.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_saliency", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainPostfilterSaliencyTests(unittest.TestCase):
    def test_fake_quantize_weight_preserves_shape_and_is_stable_for_zero(self) -> None:
        mod = load_module()
        weight = torch.zeros(3, 3, 3, 3)
        quantized = mod.fake_quantize_weight(weight)
        self.assertEqual(tuple(quantized.shape), (3, 3, 3, 3))
        self.assertTrue(torch.equal(quantized, weight))

    def test_fake_quantize_weight_is_bounded_to_original_range(self) -> None:
        mod = load_module()
        weight = torch.tensor([[-2.0, -0.25, 0.0, 0.25, 2.0]], dtype=torch.float32)
        quantized = mod.fake_quantize_weight(weight)
        self.assertEqual(tuple(quantized.shape), (1, 5))
        self.assertLessEqual(float(quantized.abs().max()), 2.0)
        self.assertGreaterEqual(float(quantized.abs().max()), 1.9)

    def test_save_model_int8_supports_per_channel_quant_and_fp32_biases(self) -> None:
        mod = load_module()
        model = mod.PostFilter(hidden=4, kernel=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidate.pt"
            mod.save_model_int8(model, path, meta={"variant": "residual"}, per_channel=True)
            state = torch.load(path, map_location="cpu")

        self.assertEqual(tuple(state["conv1.weight.s"].shape), (4,))
        self.assertEqual(tuple(state["conv2.weight.s"].shape), (4,))
        self.assertEqual(tuple(state["conv3.weight.s"].shape), (3,))
        self.assertIn("conv1.bias", state)
        self.assertNotIn("conv1.bias.q", state)
        self.assertEqual(tuple(state["conv1.bias"].shape), (4,))


if __name__ == "__main__":
    unittest.main()
