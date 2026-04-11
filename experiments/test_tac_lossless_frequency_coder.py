from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessFrequencyCoderTests(unittest.TestCase):
    def test_encode_decode_prev_symbol_stream_roundtrip_restores_uint16_tokens(self) -> None:
        from tac.lossless.frequency_coder import (
            decode_uint16_prev_symbol_stream,
            encode_uint16_prev_symbol_stream,
        )

        tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)

        encoded = encode_uint16_prev_symbol_stream(tokens)
        restored = decode_uint16_prev_symbol_stream(encoded.encoded_bytes)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(encoded.token_count, int(tokens.size))
        self.assertGreater(encoded.context_count, 0)
        self.assertEqual(encoded.header_bytes + encoded.payload_bytes, len(encoded.encoded_bytes))

    def test_encode_decode_prev_symbol_stream_supports_empty_stream(self) -> None:
        from tac.lossless.frequency_coder import (
            decode_uint16_prev_symbol_stream,
            encode_uint16_prev_symbol_stream,
        )

        tokens = np.array([], dtype=np.uint16)

        encoded = encode_uint16_prev_symbol_stream(tokens)
        restored = decode_uint16_prev_symbol_stream(encoded.encoded_bytes)

        self.assertEqual(encoded.token_count, 0)
        self.assertEqual(encoded.context_count, 0)
        self.assertEqual(restored.tolist(), [])

    def test_encode_decode_prev_symbol_file_roundtrip_restores_uint16_stream(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import (
            decode_uint16_prev_symbol_file,
            encode_uint16_prev_symbol_file,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            encoded_path = root / "tokens.tpc"
            restored_path = root / "restored.bin"
            tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)
            tokens.tofile(source)

            encoded = encode_uint16_prev_symbol_file(source, encoded_path)
            restored = decode_uint16_prev_symbol_file(encoded_path, restored_path)

            self.assertTrue(np.array_equal(np.fromfile(restored, dtype=np.uint16), tokens))
            self.assertEqual(encoded["token_count"], int(tokens.size))
            self.assertEqual(encoded["encoded_path"], str(encoded_path))

    def test_benchmark_prev_symbol_frequency_stream_reports_ratio(self) -> None:
        from tac.lossless.frequency_coder import benchmark_prev_symbol_frequency_stream

        tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)

        result = benchmark_prev_symbol_frequency_stream(tokens)

        self.assertEqual(result["command"], "lossless_prev_symbol_frequency_benchmark")
        self.assertEqual(result["token_count"], int(tokens.size))
        self.assertGreater(result["encoded_bytes"], 0)
        self.assertIn("compression_ratio", result)
        self.assertIn("context_count", result)

    def test_benchmark_prev_symbol_frequency_file_reports_ratio(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import benchmark_prev_symbol_frequency_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)
            tokens.tofile(source)

            result = benchmark_prev_symbol_frequency_file(source)

        self.assertEqual(result["command"], "lossless_prev_symbol_frequency_benchmark")
        self.assertEqual(result["token_count"], int(tokens.size))
        self.assertGreater(result["encoded_bytes"], 0)
        self.assertIn("compression_ratio", result)
        self.assertIn("context_count", result)

    def test_benchmark_uint16_frequency_file_reports_ratio(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import benchmark_uint16_frequency_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            encoded_path = root / "tokens.tfc"
            tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)
            tokens.tofile(source)

            result = benchmark_uint16_frequency_file(source, encoded_path)

        self.assertEqual(result["command"], "lossless_frequency_benchmark")
        self.assertEqual(result["token_count"], int(tokens.size))
        self.assertGreater(result["original_bytes"], 0)
        self.assertGreater(result["encoded_bytes"], 0)
        self.assertIn("compression_ratio", result)

    def test_encode_decode_frequency_file_roundtrip_restores_uint16_stream(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import decode_uint16_frequency_file, encode_uint16_frequency_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            encoded_path = root / "tokens.tfc"
            restored_path = root / "restored.bin"
            tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)
            tokens.tofile(source)

            encoded = encode_uint16_frequency_file(source, encoded_path)
            restored = decode_uint16_frequency_file(encoded_path, restored_path)

            self.assertTrue(np.array_equal(np.fromfile(restored, dtype=np.uint16), tokens))
            self.assertEqual(encoded["token_count"], int(tokens.size))
            self.assertEqual(encoded["encoded_path"], str(encoded_path))
            self.assertEqual(encoded["restored_dtype"], "uint16")

    def test_encode_decode_roundtrip_restores_uint16_tokens(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.array([9, 9, 3, 3, 3, 7, 7, 0, 65535, 65535, 12, 12, 12, 12], dtype=np.uint16)

        encoded = encode_uint16_frequency_stream(tokens)
        restored = decode_uint16_frequency_stream(encoded.encoded_bytes)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(encoded.token_count, int(tokens.size))
        self.assertEqual(encoded.unique_symbols, 6)
        self.assertEqual(encoded.header_bytes + encoded.payload_bytes, len(encoded.encoded_bytes))
        self.assertGreater(encoded.payload_bytes, 0)

    def test_encode_decode_roundtrip_supports_empty_stream(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.array([], dtype=np.uint16)

        encoded = encode_uint16_frequency_stream(tokens)
        restored = decode_uint16_frequency_stream(encoded.encoded_bytes)

        self.assertEqual(encoded.token_count, 0)
        self.assertEqual(encoded.unique_symbols, 0)
        self.assertEqual(encoded.payload_bytes, 0)
        self.assertEqual(restored.dtype, np.uint16)
        self.assertEqual(restored.tolist(), [])

    def test_encode_single_symbol_stream_uses_header_only(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.full((33,), 42, dtype=np.uint16)

        encoded = encode_uint16_frequency_stream(tokens)
        restored = decode_uint16_frequency_stream(encoded.encoded_bytes)

        self.assertEqual(encoded.unique_symbols, 1)
        self.assertEqual(encoded.payload_bytes, 0)
        self.assertTrue(np.array_equal(restored, tokens))

    def test_encoding_is_deterministic_for_equal_input(self) -> None:
        from tac.lossless.frequency_coder import encode_uint16_frequency_stream

        tokens = np.array([8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6], dtype=np.uint16)

        first = encode_uint16_frequency_stream(tokens)
        second = encode_uint16_frequency_stream(tokens)

        self.assertEqual(first.encoded_bytes, second.encoded_bytes)
        self.assertEqual(first.max_code_bits, second.max_code_bits)

    def test_header_serializes_only_sparse_symbol_frequencies(self) -> None:
        from tac.lossless.frequency_coder import encode_uint16_frequency_stream

        tokens = np.array([0, 0, 1, 1, 1024, 1024, 65535, 65535], dtype=np.uint16)

        encoded = encode_uint16_frequency_stream(tokens)

        self.assertEqual(encoded.unique_symbols, 4)
        self.assertLess(encoded.header_bytes, 64)

    def test_decode_rejects_truncated_payload(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.array([1, 2, 1, 2, 1, 2, 3, 3, 3, 0], dtype=np.uint16)
        encoded = encode_uint16_frequency_stream(tokens)

        with self.assertRaisesRegex(ValueError, "payload"):
            decode_uint16_frequency_stream(encoded.encoded_bytes[:-1])

    def test_decode_rejects_trailing_payload_bytes(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.array([7, 7, 7, 1, 1, 0, 0, 0, 0], dtype=np.uint16)
        encoded = encode_uint16_frequency_stream(tokens)

        with self.assertRaisesRegex(ValueError, "trailing"):
            decode_uint16_frequency_stream(encoded.encoded_bytes + b"\x00")

    def test_decode_rejects_empty_stream_with_declared_payload_bytes(self) -> None:
        from tac.lossless.frequency_coder import decode_uint16_frequency_stream

        with self.assertRaisesRegex(ValueError, "payload"):
            decode_uint16_frequency_stream(b"TFC1\x00\x00\x01\x01")

    def test_encoder_rejects_non_uint16_symbols(self) -> None:
        from tac.lossless.frequency_coder import encode_uint16_frequency_stream

        with self.assertRaisesRegex(ValueError, "uint16"):
            encode_uint16_frequency_stream([0, 1, 65536])

        with self.assertRaisesRegex(ValueError, "uint16"):
            encode_uint16_frequency_stream([0, -1, 1])

    def test_encode_frequency_file_rejects_odd_byte_stream(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import encode_uint16_frequency_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "broken.bin"
            source.write_bytes(b"\x01\x02\x03")

            with self.assertRaisesRegex(ValueError, "even number of bytes"):
                encode_uint16_frequency_file(source, root / "broken.tfc")

    def test_benchmark_frequency_file_rejects_odd_byte_stream(self) -> None:
        import tempfile

        from tac.lossless.frequency_coder import benchmark_uint16_frequency_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "broken.bin"
            source.write_bytes(b"\x01\x02\x03")

            with self.assertRaisesRegex(ValueError, "even number of bytes"):
                benchmark_uint16_frequency_file(source, root / "broken.tfc")

    def test_lossless_package_exports_frequency_coder(self) -> None:
        from tac.lossless import decode_uint16_frequency_stream, encode_uint16_frequency_stream

        tokens = np.array([4, 4, 4, 2, 2, 1], dtype=np.uint16)
        encoded = encode_uint16_frequency_stream(tokens)
        restored = decode_uint16_frequency_stream(encoded.encoded_bytes)

        self.assertTrue(np.array_equal(restored, tokens))


if __name__ == "__main__":
    unittest.main()
