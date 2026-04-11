from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_pixelshuffle_dilated.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_pixelshuffle_dilated", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainPostfilterPixelshuffleDilatedTests(unittest.TestCase):
    def test_model_exposes_pixelshuffle_and_dilation_behavior(self) -> None:
        mod = load_module()
        model = mod.PixelShuffleDilatedPostFilter(hidden=64, kernel=3)
        x = torch.rand(2, 3, 16, 16, dtype=torch.float32) * 255.0

        y = model(x)

        self.assertIsInstance(model.down, torch.nn.PixelUnshuffle)
        self.assertIsInstance(model.up, torch.nn.PixelShuffle)
        self.assertEqual(model.down.downscale_factor, 2)
        self.assertEqual(model.up.upscale_factor, 2)
        self.assertEqual(model.conv1.in_channels, 12)
        self.assertEqual(model.conv4.out_channels, 12)
        self.assertEqual(tuple(model.conv2.dilation), (2, 2))
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(y, x))

    def test_parser_defaults_to_h64_psd_run_name(self) -> None:
        mod = load_module()
        args = mod.build_arg_parser().parse_args([])

        self.assertEqual(args.hidden, 64)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.tag, "psd_h64_long1000")


if __name__ == "__main__":
    unittest.main()
