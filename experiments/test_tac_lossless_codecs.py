from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessCodecsTests(unittest.TestCase):
    def test_lzma_roundtrip_restores_numpy_tokens(self) -> None:
        from tac.lossless.codecs import lzma_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.npy"
            compressed = root / "tokens.lzma"
            restored = root / "restored.npy"
            np.save(source, np.array([[1, 2], [3, 4]], dtype=np.int16))

            result = lzma_roundtrip_file(source_path=source, compressed_path=compressed, restored_path=restored)

            self.assertTrue(compressed.exists())
            self.assertTrue(restored.exists())
            self.assertEqual(result["archive_bytes"], compressed.stat().st_size)
            self.assertTrue(np.array_equal(np.load(restored), np.load(source)))

    def test_zpaq_roundtrip_fails_cleanly_when_binary_missing(self) -> None:
        from tac.lossless.codecs import compress_lossless_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            target = root / "tokens.zpaq"
            source.write_bytes(b"payload")

            with mock.patch("tac.lossless.codecs.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "zpaq"):
                    compress_lossless_file(
                        profile="zpaq_baseline",
                        input_path=source,
                        output_path=target,
                    )

    def test_zpaq_roundtrip_uses_subprocess_and_restores_file(self) -> None:
        from tac.lossless.codecs import compress_lossless_file, decompress_lossless_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            archive = root / "tokens.zpaq"
            restored = root / "restored.bin"
            source.write_bytes(b"zpaq payload" * 8)

            def fake_run(cmd, *, cwd=None, check=None, stdout=None, stderr=None):
                self.assertFalse(isinstance(cmd, str))
                if cmd[1] == "add":
                    self.assertEqual(cmd[3], "tokens.bin")
                    Path(cmd[2]).write_bytes(b"zpaq-archive")
                elif cmd[1] == "extract":
                    extract_dir = Path(cmd[cmd.index("-to") + 1])
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    (extract_dir / "tokens.bin").write_bytes(source.read_bytes())
                else:
                    raise AssertionError(f"unexpected command: {cmd}")

            with (
                mock.patch("tac.lossless.codecs.shutil.which", return_value="/usr/bin/zpaq"),
                mock.patch("tac.lossless.codecs.subprocess.run", side_effect=fake_run) as mocked_run,
            ):
                compression = compress_lossless_file(
                    profile="zpaq_baseline",
                    input_path=source,
                    output_path=archive,
                )
                restored_path = decompress_lossless_file(
                    profile="zpaq_baseline",
                    archive_path=archive,
                    output_path=restored,
                )
                restored_bytes = restored.read_bytes()
                source_bytes = source.read_bytes()

        self.assertEqual(mocked_run.call_count, 2)
        self.assertEqual(compression.method, "zpaq")
        self.assertEqual(compression.archive_bytes, len(b"zpaq-archive"))
        self.assertEqual(restored_path, restored)
        self.assertEqual(restored_bytes, source_bytes)


if __name__ == "__main__":
    unittest.main()
