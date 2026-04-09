from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_dilated_h64.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_dilated_h64", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainPostfilterDilatedH64Tests(unittest.TestCase):
    def test_dilated_wrapper_exposes_dilated_meta(self) -> None:
        mod = load_module()
        meta = mod.normalize_postfilter_meta(64, 3, 20.0)
        self.assertEqual(meta["variant"], "dilated")
        self.assertEqual(meta["hidden"], 64)
        self.assertEqual(meta["kernel"], 3)

    def test_qat_dilated_postfilter_uses_dilation_two(self) -> None:
        mod = load_module()
        model = mod.QATDilatedPostFilter(hidden=8, kernel=3)
        self.assertEqual(model.conv2.dilation, (2, 2))
        self.assertEqual(model.conv2.padding, (2, 2))


if __name__ == "__main__":
    unittest.main()
