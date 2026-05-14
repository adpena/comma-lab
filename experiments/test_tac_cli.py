# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = ROOT / "src" / "tac" / "cli.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tac.cli", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TacCliExperimentsTests(unittest.TestCase):
    def test_lossless_next_frame_sample_subcommand_defers_context_to_profile(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "sample.nfg"
            with mock.patch.object(
                mod,
                "encode_commavq_next_frame_sample",
                return_value={
                    "command": "lossless_next_frame_sample",
                    "token_path": str(token_path),
                    "encoded_path": str(output_path),
                    "profile": "gpt_next_frame_small",
                    "context_frames": 2,
                    "frame_count": 3,
                    "encoded_bytes": 64,
                    "compression_ratio": 12.0,
                    "exact_match": True,
                    "local_only": True,
                    "measured": False,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "next-frame-sample",
                        "--profile",
                        "gpt_next_frame_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            encoded_path=output_path,
            profile="gpt_next_frame_small",
            max_frames=32,
            context_frames=None,
            device="mps",
            dtype="auto",
            verify_decode=False,
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_next_frame_sample")
        self.assertTrue(result["local_only"])
        self.assertFalse(result["measured"])


if __name__ == "__main__":
    unittest.main()
