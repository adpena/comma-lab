# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessSubmissionTests(unittest.TestCase):
    def test_build_submission_zip_creates_challenge_layout_with_deterministic_order(self) -> None:
        from tac.lossless.submission import build_submission_zip

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_dir = root / "payload"
            payload_dir.mkdir()
            (payload_dir / "clip_b.bin").write_bytes(b"bbb")
            (payload_dir / "clip_a.bin").write_bytes(b"aaa")
            decompress_path = root / "decompress.py"
            decompress_path.write_text("print('ok')\n")
            output_path = root / "submission.zip"

            build_submission_zip(
                payload_dir=payload_dir,
                decompress_path=decompress_path,
                output_path=output_path,
            )

            with zipfile.ZipFile(output_path) as zf:
                self.assertEqual(
                    zf.namelist(),
                    ["clip_a.bin", "clip_b.bin", "decompress.py"],
                )
                self.assertEqual(zf.read("clip_a.bin"), b"aaa")
                self.assertEqual(zf.read("clip_b.bin"), b"bbb")
                self.assertEqual(zf.read("decompress.py"), b"print('ok')\n")

    def test_build_submission_zip_is_byte_stable_across_repeated_builds(self) -> None:
        from tac.lossless.submission import build_submission_zip

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_dir = root / "payload"
            payload_dir.mkdir()
            (payload_dir / "nested").mkdir()
            (payload_dir / "nested" / "0001.bin").write_bytes(b"frame-1")
            (payload_dir / "0000.bin").write_bytes(b"frame-0")
            decompress_path = root / "decompress.py"
            decompress_path.write_text("print('decode')\n")
            first_zip = root / "first.zip"
            second_zip = root / "second.zip"

            build_submission_zip(
                payload_dir=payload_dir,
                decompress_path=decompress_path,
                output_path=first_zip,
            )
            build_submission_zip(
                payload_dir=payload_dir,
                decompress_path=decompress_path,
                output_path=second_zip,
            )

            self.assertEqual(
                hashlib.sha256(first_zip.read_bytes()).hexdigest(),
                hashlib.sha256(second_zip.read_bytes()).hexdigest(),
            )

    def test_validate_submission_inputs_requires_decompress_file_named_exactly(self) -> None:
        from tac.lossless.submission import validate_submission_inputs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_dir = root / "payload"
            payload_dir.mkdir()
            wrong_name = root / "decode.py"
            wrong_name.write_text("print('wrong')\n")

            with self.assertRaisesRegex(ValueError, "decompress.py"):
                validate_submission_inputs(
                    payload_dir=payload_dir,
                    decompress_path=wrong_name,
                )

    def test_validate_submission_inputs_rejects_decompress_inside_payload_tree(self) -> None:
        from tac.lossless.submission import validate_submission_inputs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_dir = root / "payload"
            payload_dir.mkdir()
            embedded = payload_dir / "decompress.py"
            embedded.write_text("print('embedded')\n")

            with self.assertRaisesRegex(ValueError, "payload"):
                validate_submission_inputs(
                    payload_dir=payload_dir,
                    decompress_path=embedded,
                )


if __name__ == "__main__":
    unittest.main()
