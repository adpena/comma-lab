from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_pairaware.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_pairaware", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainPostfilterPairAwareTests(unittest.TestCase):
    def test_pair_bhwc_to_6ch_preserves_previous_then_current_order(self) -> None:
        mod = load_module()
        prev = torch.full((1, 2, 2, 3), 10.0)
        curr = torch.full((1, 2, 2, 3), 20.0)
        pair = torch.stack([prev, curr], dim=1)

        pair_6ch = mod.pair_bhwc_to_6ch(pair)

        self.assertEqual(tuple(pair_6ch.shape), (1, 6, 2, 2))
        self.assertTrue(torch.all(pair_6ch[:, 0:3] == 10.0))
        self.assertTrue(torch.all(pair_6ch[:, 3:6] == 20.0))

    def test_apply_pairaware_postfilter_returns_corrected_current_frame(self) -> None:
        mod = load_module()
        model = mod.PairAwarePostFilter(hidden=4, kernel=3)
        prev = torch.full((1, 2, 3, 3), 7.0)
        curr = torch.full((1, 2, 3, 3), 42.0)
        pair = torch.stack([prev, curr], dim=1)

        corrected = mod.apply_pairaware_postfilter(model, pair)
        current = mod.current_frame_bchw_from_pair(pair)

        self.assertEqual(tuple(corrected.shape), (1, 3, 2, 3))
        self.assertTrue(torch.equal(corrected, current))

    def test_main_runs_a_tiny_dry_run(self) -> None:
        mod = load_module()

        summary = mod.main([
            "--hidden", "4",
            "--kernel", "3",
            "--height", "4",
            "--width", "5",
            "--batch-size", "2",
        ])

        self.assertEqual(summary["model"], "PairAwarePostFilter")
        self.assertEqual(summary["input_shape"], [2, 2, 4, 5, 3])
        self.assertEqual(summary["pair_6ch_shape"], [2, 6, 4, 5])
        self.assertEqual(summary["output_shape"], [2, 3, 4, 5])
        self.assertEqual(summary["max_abs_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
