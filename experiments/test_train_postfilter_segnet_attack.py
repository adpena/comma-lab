from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_segnet_attack.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_segnet_attack", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainPostfilterSegnetAttackTests(unittest.TestCase):
    def test_module_exposes_best_checkpoint_helper(self) -> None:
        mod = load_module()
        self.assertTrue(callable(mod.save_best_checkpoint))

    def test_evaluate_ema_model_runs_on_small_dummy_pairs(self) -> None:
        mod = load_module()

        class DummyPoseNet(torch.nn.Module):
            def __init__(self, pose_value: float) -> None:
                super().__init__()
                self.pose_value = pose_value

            def preprocess_input(self, x):
                return x

            def forward(self, x):
                b = x.shape[0]
                pose = torch.full((b, 6), self.pose_value, dtype=x.dtype, device=x.device)
                return {"pose": pose}

        class DummySegNet(torch.nn.Module):
            def __init__(self, seg_classes: int = 3) -> None:
                super().__init__()
                self.seg_classes = seg_classes

            def preprocess_input(self, x):
                return x

            def forward(self, x):
                b, t, c, h, w = x.shape
                seg = torch.zeros((b * t, self.seg_classes, h, w), dtype=x.dtype, device=x.device)
                seg[:, 0] = 1.0
                return seg

        model = mod.QATPostFilter(hidden=4, kernel=3)
        ema = mod.EMA(model, decay=0.9)
        comp_pair = torch.zeros(1, 2, 4, 4, 3)
        gt_pair = torch.zeros(1, 2, 4, 4, 3)
        score, pose, seg = mod.evaluate_ema_model(
            model=model,
            ema=ema,
            eval_indices=[0],
            comp_pairs=[comp_pair],
            gt_pairs=[gt_pair],
            posenet=DummyPoseNet(0.0),
            segnet=DummySegNet(),
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertGreaterEqual(pose, 0.0)
        self.assertGreaterEqual(seg, 0.0)


if __name__ == "__main__":
    unittest.main()
