from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_dct.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_dct", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainPostfilterDctTests(unittest.TestCase):
    def test_mid_frequency_mask_excludes_dc(self) -> None:
        mod = load_module()
        mask = mod.build_mid_frequency_mask(8)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertGreater(mask.sum().item(), 0.0)

    def test_zero_init_filter_is_identity_even_when_padding_is_needed(self) -> None:
        mod = load_module()
        model = mod.BlockDCTMidbandFilter(block=8)
        x = torch.rand(2, 3, 11, 13) * 255.0
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(x, y, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
