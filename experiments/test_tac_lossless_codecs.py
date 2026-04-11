from __future__ import annotations

import builtins
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

    def test_zstd_dict_roundtrip_fails_cleanly_when_backend_missing(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            source.write_bytes(b"payload")

            with mock.patch("tac.lossless.codecs._require_zstd_backend", side_effect=RuntimeError("zstd")):
                with self.assertRaisesRegex(RuntimeError, "zstd"):
                    zstd_dict_roundtrip_file(
                        source_path=source,
                        compressed_path=compressed,
                        restored_path=restored,
                    )

    def test_zstd_dict_roundtrip_uses_backend_and_restores_file(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"zstd payload" * 8
            source.write_bytes(payload)

            class FakeBackend:
                def train_dictionary(self, samples, *, dict_size):
                    self.samples = list(samples)
                    self.dict_size = dict_size
                    return b"dict-bytes"

                def compress(self, data, *, dictionary):
                    assert dictionary == b"dict-bytes"
                    return b"zstd-archive:" + data

                def decompress(self, data, *, dictionary):
                    assert dictionary == b"dict-bytes"
                    assert data.startswith(b"zstd-archive:")
                    return data[len(b"zstd-archive:") :]

            backend = FakeBackend()

            with mock.patch("tac.lossless.codecs._require_zstd_backend", return_value=backend):
                result = zstd_dict_roundtrip_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    dict_size=1024,
                )

                self.assertEqual(result["archive_bytes"], compressed.stat().st_size)
                self.assertEqual(restored.read_bytes(), payload)

        self.assertEqual(backend.samples, [payload])
        self.assertEqual(backend.dict_size, 1024)
        self.assertEqual(result["method"], "zstd_dict")
        self.assertEqual(result["dictionary_bytes"], len(b"dict-bytes"))

    def test_benchmark_zstd_dict_file_reports_ratio(self) -> None:
        from tac.lossless.codecs import benchmark_zstd_dict_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"zstd payload" * 8
            source.write_bytes(payload)

            with mock.patch(
                "tac.lossless.codecs.zstd_dict_roundtrip_file",
                return_value={
                    "method": "zstd_dict",
                    "source_path": str(source),
                    "compressed_path": str(compressed),
                    "restored_path": str(restored),
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                    "archive_bytes": 50,
                    "original_bytes": len(payload),
                    "compression_rate": len(payload) / 50,
                },
            ) as mocked:
                result = benchmark_zstd_dict_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    sample_paths=[source, source],
                    dict_size=1024,
                )

        mocked.assert_called_once_with(
            source_path=source,
            compressed_path=compressed,
            restored_path=restored,
            dict_size=1024,
            sample_payloads=[payload, payload],
        )
        self.assertEqual(result["command"], "lossless_zstd_dict_benchmark")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["dictionary_bytes"], 123)
        self.assertEqual(result["archive_bytes"], 50)

    def test_zstd_dict_roundtrip_falls_back_to_cli_binary_when_python_backend_is_missing(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"zstd payload" * 8
            source.write_bytes(payload)

            original_import = builtins.__import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "zstandard":
                    raise ImportError("no module named zstandard")
                return original_import(name, globals, locals, fromlist, level)

            def fake_run(cmd, *, cwd=None, check=None, stdout=None, stderr=None):
                self.assertFalse(isinstance(cmd, str))
                if "--train" in cmd:
                    dict_path = Path(cmd[cmd.index("-o") + 1])
                    dict_path.write_bytes(b"dict-bytes")
                    return
                if "-d" in cmd:
                    output_path = Path(cmd[cmd.index("-o") + 1])
                    source_path = Path(cmd[cmd.index("-d") + 1])
                    archive = source_path.read_bytes()
                    self.assertTrue(archive.startswith(b"zstd-archive:"))
                    output_path.write_bytes(archive[len(b"zstd-archive:") :])
                    return
                output_path = Path(cmd[cmd.index("-o") + 1])
                source_path = Path(cmd[cmd.index("-o") - 1])
                output_path.write_bytes(b"zstd-archive:" + source_path.read_bytes())

            with (
                mock.patch("builtins.__import__", side_effect=fake_import),
                mock.patch("tac.lossless.codecs.shutil.which", return_value="/usr/bin/zstd"),
                mock.patch("tac.lossless.codecs.subprocess.run", side_effect=fake_run),
            ):
                result = zstd_dict_roundtrip_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    dict_size=1024,
                )

                self.assertEqual(result["archive_bytes"], compressed.stat().st_size)
                self.assertEqual(restored.read_bytes(), payload)

        self.assertEqual(result["method"], "zstd_dict")
        self.assertEqual(result["dictionary_bytes"], len(b"dict-bytes"))

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
