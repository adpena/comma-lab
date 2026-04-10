from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


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
    def test_detect_device_falls_back_to_cpu_for_unsupported_cuda_capability(self) -> None:
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
            self.assertEqual(str(mod.detect_device()), "cpu")
        finally:
            torch.cuda.is_available = original_is_available
            torch.cuda.get_device_capability = original_get_capability
            if original_mps is not None and original_mps_is_available is not None:
                original_mps.is_available = original_mps_is_available

    def test_arg_parser_accepts_checkpoint_selection_flags(self) -> None:
        mod = load_module()
        args = mod.build_arg_parser().parse_args(["--checkpoint-select-int8", "--per-channel-int8"])
        self.assertTrue(args.checkpoint_select_int8)
        self.assertTrue(args.per_channel_int8)

    def test_default_tag_and_metadata_surface_are_dilated(self) -> None:
        mod = load_module()
        self.assertEqual(mod.make_default_tag(64, 20.0), "dilated_qat_ema_h64_a20")
        meta = mod.normalize_postfilter_meta(64, 3, 20.0)
        self.assertEqual(meta["variant"], "dilated")
        self.assertEqual(meta["hidden"], 64)
        self.assertEqual(meta["kernel"], 3)

    def test_qat_dilated_postfilter_uses_dilation_two(self) -> None:
        mod = load_module()
        model = mod.QATDilatedPostFilter(hidden=8, kernel=3)
        self.assertEqual(model.conv2.dilation, (2, 2))
        self.assertEqual(model.conv2.padding, (2, 2))

    def test_save_best_checkpoint_writes_durable_metadata(self) -> None:
        mod = load_module()
        model = mod.QATDilatedPostFilter(hidden=4, kernel=3)
        ema = mod.EMA(model, decay=0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            payload = mod.save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=out_dir,
                tag="unit_dilated",
                meta={"variant": "dilated", "hidden": 4, "kernel": 3, "alpha": 20.0},
                epoch=7,
                scorer=1.234,
                per_channel_int8=True,
            )

            meta_path = out_dir / "postfilter_unit_dilated_best_meta.json"
            self.assertTrue((out_dir / "postfilter_unit_dilated_best_fp32.pt").exists())
            self.assertTrue((out_dir / "postfilter_unit_dilated_best_int8.pt").exists())
            self.assertTrue(meta_path.exists())

            on_disk = json.loads(meta_path.read_text())
            self.assertEqual(on_disk["tag"], "unit_dilated")
            self.assertEqual(on_disk["epoch"], 7)
            self.assertAlmostEqual(on_disk["scorer"], 1.234)
            self.assertEqual(on_disk["meta"]["variant"], "dilated")
            self.assertEqual(payload["int8_size"], on_disk["int8_size"])
            self.assertEqual(tuple(torch.load(out_dir / "postfilter_unit_dilated_best_int8.pt", weights_only=True)["conv1.weight.s"].shape), (4,))


if __name__ == "__main__":
    unittest.main()
