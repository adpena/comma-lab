from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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
        self.assertTrue(callable(mod.save_final_artifacts))

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

    def test_save_final_artifacts_writes_final_and_best_metadata(self) -> None:
        mod = load_module()
        model = mod.QATPostFilter(hidden=4, kernel=3)
        meta = {"variant": "saliency_weighted", "hidden": 4, "kernel": 3, "alpha": 20.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            best_payload = {
                "epoch": 12,
                "scorer": 1.84,
                "fp32_path": str(output_dir / "postfilter_unit_best_fp32.pt"),
                "int8_path": str(output_dir / "postfilter_unit_best_int8.pt"),
                "int8_size": 12345,
                "meta": meta,
            }
            result = mod.save_final_artifacts(
                model=model,
                output_dir=output_dir,
                tag="unit",
                meta=meta,
                baseline_loss=1.4248,
                final_loss=1.2233,
                final_pose=0.047042,
                final_seg=0.005374,
                best_eval_payload=best_payload,
            )

            self.assertTrue((output_dir / "postfilter_unit_fp32.pt").exists())
            self.assertTrue((output_dir / "postfilter_unit_int8.pt").exists())
            self.assertTrue((output_dir / "postfilter_unit_final_meta.json").exists())
            self.assertTrue((output_dir / "postfilter_unit_best_meta.json").exists())

            final_meta = json.loads((output_dir / "postfilter_unit_final_meta.json").read_text())
            best_meta = json.loads((output_dir / "postfilter_unit_best_meta.json").read_text())

        self.assertEqual(result["final_meta_path"], str(output_dir / "postfilter_unit_final_meta.json"))
        self.assertEqual(final_meta["final_loss"], 1.2233)
        self.assertEqual(final_meta["best_eval"]["epoch"], 12)
        self.assertEqual(best_meta["epoch"], 12)
        self.assertEqual(best_meta["scorer"], 1.84)


if __name__ == "__main__":
    unittest.main()
