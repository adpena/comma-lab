from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessBaselineTests(unittest.TestCase):
    def test_render_lzma_decompress_script_roundtrips_payload_files(self) -> None:
        from tac.lossless.codecs import compress_token_records, render_lzma_decompress_script
        from tac.lossless.data import TokenRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "decoded"
            output_dir.mkdir()
            records = [
                TokenRecord("clip_a.npy", np.arange(128, dtype=np.int16).reshape(1, 8, 16)),
                TokenRecord("clip_b.npy", (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)),
            ]
            compress_token_records(profile="lzma_baseline", records=records, output_dir=root)
            decompress_path = root / "decompress.py"
            decompress_path.write_text(render_lzma_decompress_script())

            subprocess.run(
                [sys.executable, str(decompress_path)],
                cwd=root,
                check=True,
                env={**dict(), "OUTPUT_DIR": str(output_dir)},
            )

            self.assertTrue(np.array_equal(np.load(output_dir / "clip_a.npy"), records[0].tokens))
            self.assertTrue(np.array_equal(np.load(output_dir / "clip_b.npy"), records[1].tokens))

    def test_compress_token_records_preserves_file_names(self) -> None:
        from tac.lossless.codecs import compress_token_records
        from tac.lossless.data import TokenRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_dir = root / "payload"
            payload_dir.mkdir()
            records = [
                TokenRecord("clip_b.npy", (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)),
                TokenRecord("clip_a.npy", np.arange(128, dtype=np.int16).reshape(1, 8, 16)),
            ]

            archive_bytes = compress_token_records(profile="lzma_baseline", records=records, output_dir=payload_dir)

            self.assertGreater(archive_bytes, 0)
            self.assertTrue((payload_dir / "clip_a.npy").exists())
            self.assertTrue((payload_dir / "clip_b.npy").exists())

    def test_build_lzma_baseline_submission_uses_dataset_loader_and_packages_zip(self) -> None:
        from tac.lossless.codecs import build_lzma_baseline_submission

        calls = []

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            calls.append(
                {
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "data_files": data_files,
                }
            )
            return {
                "train": [
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = build_lzma_baseline_submission(
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
            )

            self.assertTrue(Path(result["archive_path"]).exists())
            self.assertTrue((root / "decompress.py").exists())
            self.assertTrue((root / "payload" / "clip_a.npy").exists())
            self.assertTrue((root / "payload" / "clip_b.npy").exists())

        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})

    def test_evaluate_lzma_baseline_submission_returns_measured_result(self) -> None:
        from tac.lossless.codecs import evaluate_lzma_baseline_submission

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {
                "train": [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = evaluate_lzma_baseline_submission(
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
            )

        self.assertEqual(result["command"], "lossless_baseline_evaluate")
        self.assertTrue(result["verification"]["exact_match"])
        self.assertGreater(result["compression"]["compression_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
