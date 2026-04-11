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
                    dict_size=64,
                    sample_payloads=[payload, payload],
                )

                self.assertEqual(result["archive_bytes"], compressed.stat().st_size)
                self.assertEqual(restored.read_bytes(), payload)

        self.assertEqual(backend.samples, [payload, payload])
        self.assertEqual(backend.dict_size, 64)
        self.assertEqual(result["method"], "zstd_dict")
        self.assertEqual(result["dictionary_bytes"], len(b"dict-bytes"))

    def test_zstd_dict_roundtrip_splits_large_samples_into_blocks_for_training(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"abcdefghijkl"
            source.write_bytes(payload)

            class FakeBackend:
                def train_dictionary(self, samples, *, dict_size):
                    self.samples = list(samples)
                    self.dict_size = dict_size
                    return b"dict-bytes"

                def compress(self, data, *, dictionary):
                    return b"zstd-archive:" + data

                def decompress(self, data, *, dictionary):
                    return data[len(b"zstd-archive:") :]

            backend = FakeBackend()

            with mock.patch("tac.lossless.codecs._require_zstd_backend", return_value=backend):
                zstd_dict_roundtrip_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    dict_size=4,
                    sample_payloads=[payload],
                    sample_block_bytes=4,
                )

        self.assertEqual(backend.samples, [b"abcd", b"efgh", b"ijkl"])
        self.assertEqual(backend.dict_size, 4)

    def test_zstd_dict_roundtrip_caps_blocked_training_samples_deterministically(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"abcdefghijkl"
            source.write_bytes(payload)

            class FakeBackend:
                def train_dictionary(self, samples, *, dict_size):
                    self.samples = list(samples)
                    self.dict_size = dict_size
                    return b"dict-bytes"

                def compress(self, data, *, dictionary):
                    return b"zstd-archive:" + data

                def decompress(self, data, *, dictionary):
                    return data[len(b"zstd-archive:") :]

            backend = FakeBackend()

            with mock.patch("tac.lossless.codecs._require_zstd_backend", return_value=backend):
                zstd_dict_roundtrip_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    dict_size=3,
                    sample_payloads=[payload],
                    sample_block_bytes=1,
                    max_training_samples=4,
                )

        self.assertEqual(backend.samples, [b"a", b"d", b"h", b"l"])
        self.assertEqual(backend.dict_size, 3)

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
                    sample_block_bytes=4096,
                    max_training_samples=256,
                )

        mocked.assert_called_once_with(
            source_path=source,
            compressed_path=compressed,
            restored_path=restored,
            dict_size=1024,
            sample_payloads=[payload, payload],
            sample_block_bytes=4096,
            max_training_samples=256,
        )
        self.assertEqual(result["command"], "lossless_zstd_dict_benchmark")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["dictionary_bytes"], 123)
        self.assertEqual(result["archive_bytes"], 50)

    def test_benchmark_zstd_dict_directory_reports_aggregate_ratio(self) -> None:
        from tac.lossless.codecs import benchmark_zstd_dict_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            compressed_root = root / "compressed"
            restored_root = root / "restored"
            sample_a = root / "sample_a.bin"
            sample_b = root / "sample_b.bin"
            source_root.mkdir(parents=True, exist_ok=True)
            sample_a.write_bytes(b"sample-a")
            sample_b.write_bytes(b"sample-b")
            first = source_root / "a.bin"
            second = source_root / "nested" / "b.bin"
            second.parent.mkdir(parents=True, exist_ok=True)
            first.write_bytes(b"aaa")
            second.write_bytes(b"bbbb")

            payloads = {
                first: {
                    "archive_bytes": 5,
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                },
                second: {
                    "archive_bytes": 7,
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                },
            }

            def fake_roundtrip(
                *,
                source_path,
                compressed_path,
                restored_path,
                dict_size,
                sample_payloads,
                sample_block_bytes=None,
                max_training_samples=None,
            ):
                source_path = Path(source_path)
                compressed_path = Path(compressed_path)
                restored_path = Path(restored_path)
                compressed_path.parent.mkdir(parents=True, exist_ok=True)
                restored_path.parent.mkdir(parents=True, exist_ok=True)
                compressed_path.write_bytes(b"x" * payloads[source_path]["archive_bytes"])
                restored_path.write_bytes(source_path.read_bytes())
                return {
                    "method": "zstd_dict",
                    "source_path": str(source_path),
                    "compressed_path": str(compressed_path),
                    "restored_path": str(restored_path),
                    "dictionary_bytes": payloads[source_path]["dictionary_bytes"],
                    "sample_count": payloads[source_path]["sample_count"],
                    "archive_bytes": payloads[source_path]["archive_bytes"],
                    "original_bytes": source_path.stat().st_size,
                    "compression_rate": source_path.stat().st_size / payloads[source_path]["archive_bytes"],
                }

            with mock.patch("tac.lossless.codecs.zstd_dict_roundtrip_file", side_effect=fake_roundtrip) as mocked:
                result = benchmark_zstd_dict_directory(
                    source_root=source_root,
                    compressed_root=compressed_root,
                    restored_root=restored_root,
                    sample_paths=[sample_a, sample_b],
                    dict_size=4096,
                    sample_block_bytes=2048,
                    max_training_samples=256,
                )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["command"], "lossless_zstd_dict_directory_benchmark")
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["dictionary_bytes"], 123)
        self.assertEqual(result["archive_bytes"], 12)
        self.assertEqual(result["original_bytes"], 7)
        self.assertAlmostEqual(result["compression_rate"], 7 / 12)

    def test_benchmark_zstd_dict_chunked_file_splits_source_and_verifies_roundtrip(self) -> None:
        from tac.lossless.codecs import benchmark_zstd_dict_chunked_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "stream.bin"
            compressed_root = root / "compressed"
            restored_root = root / "restored"
            sample_a = root / "sample_a.bin"
            sample_b = root / "sample_b.bin"
            payload = b"abcdefghij"
            source.write_bytes(payload)
            sample_a.write_bytes(b"sample-a")
            sample_b.write_bytes(b"sample-b")

            def fake_directory(
                *,
                source_root,
                compressed_root,
                restored_root,
                sample_paths,
                dict_size,
                sample_block_bytes=None,
                max_training_samples=None,
            ):
                source_root = Path(source_root)
                restored_root = Path(restored_root)
                chunks = sorted(path for path in source_root.iterdir() if path.is_file())
                self.assertEqual([chunk.name for chunk in chunks], ["000000.bin", "000001.bin", "000002.bin"])
                self.assertEqual([chunk.read_bytes() for chunk in chunks], [b"abcd", b"efgh", b"ij"])
                for chunk in chunks:
                    target = restored_root / chunk.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(chunk.read_bytes())
                return {
                    "command": "lossless_zstd_dict_directory_benchmark",
                    "method": "zstd_dict",
                    "source_root": str(source_root),
                    "compressed_root": str(compressed_root),
                    "restored_root": str(restored_root),
                    "dictionary_bytes": 123,
                    "sample_count": len(sample_paths),
                    "file_count": len(chunks),
                    "archive_bytes": 12,
                    "original_bytes": 10,
                    "compression_rate": 10 / 12,
                }

            with mock.patch("tac.lossless.codecs.benchmark_zstd_dict_directory", side_effect=fake_directory) as mocked:
                result = benchmark_zstd_dict_chunked_file(
                    source_path=source,
                    compressed_root=compressed_root,
                    restored_root=restored_root,
                    block_bytes=4,
                    sample_paths=[sample_a, sample_b],
                    dict_size=4096,
                    sample_block_bytes=2048,
                    max_training_samples=256,
                )

        mocked.assert_called_once()
        self.assertEqual(result["command"], "lossless_zstd_dict_chunked_benchmark")
        self.assertEqual(result["file_count"], 3)
        self.assertEqual(result["block_bytes"], 4)
        self.assertTrue(result["exact_match"])
        self.assertAlmostEqual(result["compression_rate"], 10 / 12)

    def test_zstd_dict_roundtrip_rejects_insufficient_sample_corpus(self) -> None:
        from tac.lossless.codecs import zstd_dict_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            payload = b"zstd payload"
            source.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "sample corpus"):
                zstd_dict_roundtrip_file(
                    source_path=source,
                    compressed_path=compressed,
                    restored_path=restored,
                    dict_size=1024,
                    sample_payloads=[payload],
                )

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
                    dict_size=64,
                    sample_payloads=[payload, payload],
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
                    # Accept both "tokens.bin" and "./tokens.bin"
                    self.assertIn(Path(cmd[3]).name, ["tokens.bin"])
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
