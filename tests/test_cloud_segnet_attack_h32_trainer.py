from __future__ import annotations

import importlib.util
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

    def test_decode_archive_accepts_extracted_directory(self) -> None:
        mod = load_module()
        called = []

        def fake_decode_video(path: str, target_h: int = mod.CAMERA_SIZE[1], target_w: int = mod.CAMERA_SIZE[0]):
            called.append(path)
            return ["ok"]

        original_decode_video = mod.decode_video
        try:
            mod.decode_video = fake_decode_video
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                extracted = root / "decode_base_archive"
                extracted.mkdir()
                (extracted / "0.mkv").write_bytes(b"mkv")
                result = mod.decode_archive(str(extracted))
        finally:
            mod.decode_video = original_decode_video

        self.assertEqual(result, ["ok"])
        self.assertEqual(called, [str(extracted / "0.mkv")])


if __name__ == "__main__":
    unittest.main()
