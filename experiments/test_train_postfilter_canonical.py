from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_canonical.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_canonical", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainPostfilterCanonicalTests(unittest.TestCase):
    def test_select_validation_indices_is_deterministic_and_bounded(self) -> None:
        mod = load_module()
        indices_a = mod.select_validation_indices(600, 8, seed=1234)
        indices_b = mod.select_validation_indices(600, 8, seed=1234)

        self.assertEqual(indices_a, indices_b)
        self.assertEqual(len(indices_a), 75)
        self.assertTrue(all(0 <= idx < 600 for idx in indices_a))
        self.assertEqual(indices_a, sorted(indices_a))

    def test_select_validation_indices_disables_cleanly(self) -> None:
        mod = load_module()
        self.assertEqual(mod.select_validation_indices(600, 0), [])
        self.assertEqual(mod.select_validation_indices(1, 8), [])


if __name__ == "__main__":
    unittest.main()
