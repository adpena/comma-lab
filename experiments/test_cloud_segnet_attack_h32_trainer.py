from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "cloud_segnet_attack_h32_trainer.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "cloud_segnet_attack_h32_trainer", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CloudSegnetAttackH32TrainerTests(unittest.TestCase):
    def test_select_device_falls_back_to_cpu_for_unsupported_cuda_capability(self) -> None:
        original_is_available = torch.cuda.is_available
        original_get_capability = torch.cuda.get_device_capability
        original_mps = getattr(torch.backends, "mps", None)
        original_mps_is_available = getattr(original_mps, "is_available", None)
        try:
            torch.cuda.is_available = lambda: True
            torch.cuda.get_device_capability = lambda *_args, **_kwargs: (6, 0)
            if original_mps is not None:
                original_mps.is_available = lambda: False
            mod = load_module()
            self.assertEqual(str(mod.select_device()), "cpu")
        finally:
            torch.cuda.is_available = original_is_available
            torch.cuda.get_device_capability = original_get_capability
            if original_mps is not None and original_mps_is_available is not None:
                original_mps.is_available = original_mps_is_available

    def test_parser_is_fixed_h32_and_exposes_expected_flags(self) -> None:
        mod = load_module()
        parser = mod.build_arg_parser()
        args = parser.parse_args(["--epochs", "7", "--alpha", "13.5", "--tag", "demo"])

        self.assertFalse(any(action.dest == "hidden" for action in parser._actions))
        self.assertEqual(args.epochs, 7)
        self.assertEqual(args.alpha, 13.5)
        self.assertEqual(args.tag, "demo")

    def test_resolve_asset_uses_kaggle_dataset_fallback(self) -> None:
        mod = load_module()
        original_exists = mod.Path.exists
        try:
            mod.Path.exists = lambda self: str(self) == "/kaggle/input/comma-lab-private-assets/decode_base_archive.zip"
            asset = mod.resolve_asset("reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip")
        finally:
            mod.Path.exists = original_exists
        self.assertEqual(str(asset), "/kaggle/input/comma-lab-private-assets/decode_base_archive.zip")

    def test_normalize_postfilter_meta_records_fixed_h32_family(self) -> None:
        mod = load_module()
        meta = mod.normalize_postfilter_meta(alpha=20.0, kernel=3)

        self.assertEqual(meta["variant"], "cloud_segnet_attack_h32")
        self.assertEqual(meta["hidden"], 32)
        self.assertEqual(meta["kernel"], 3)
        self.assertEqual(meta["alpha"], 20.0)

    def test_save_final_artifacts_emits_durable_best_and_final_metadata(self) -> None:
        import torch

        mod = load_module()
        model = mod.PostFilter()
        meta = mod.normalize_postfilter_meta(alpha=20.0, kernel=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            best_payload = mod.save_best_checkpoint(
                model=model,
                ema=mod.EMA(model, decay=0.9),
                output_dir=output_dir,
                tag="unit",
                meta=meta,
                epoch=3,
                scorer=1.2345,
            )
            final_payload = mod.save_final_artifacts(
                model=model,
                output_dir=output_dir,
                tag="unit",
                meta=meta,
                baseline_loss=2.5,
                final_loss=1.5,
                final_pose=0.05,
                final_seg=0.01,
                best_eval_payload=best_payload,
            )

            best_meta_path = output_dir / "postfilter_unit_best_meta.json"
            final_meta_path = output_dir / "postfilter_unit_final_meta.json"

            self.assertTrue(best_meta_path.exists())
            self.assertTrue(final_meta_path.exists())

            best_meta = json.loads(best_meta_path.read_text())
            final_meta = json.loads(final_meta_path.read_text())

        self.assertEqual(best_payload["epoch"], 3)
        self.assertEqual(best_meta["scorer"], 1.2345)
        self.assertEqual(final_payload["best_meta_path"], str(best_meta_path))
        self.assertEqual(final_meta["final_loss"], 1.5)
        self.assertEqual(final_meta["best_eval"]["epoch"], 3)
        self.assertEqual(final_meta["best_eval"]["scorer"], 1.2345)


if __name__ == "__main__":
    unittest.main()
