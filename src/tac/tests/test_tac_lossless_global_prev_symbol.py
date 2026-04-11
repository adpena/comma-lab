from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessGlobalPrevSymbolTests(unittest.TestCase):
    def test_encode_decode_corpus_roundtrips_token_records(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import (
            decode_corpus_global_prev_symbol_position_major,
            encode_corpus_global_prev_symbol_position_major,
        )

        records = [
            TokenRecord("clip_b", (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)),
            TokenRecord("clip_a", np.arange(128, dtype=np.int16).reshape(1, 8, 16)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoded = encode_corpus_global_prev_symbol_position_major(records=records, output_dir=root / "encoded")
            decoded = decode_corpus_global_prev_symbol_position_major(
                encoded_dir=root / "encoded",
                output_dir=root / "decoded",
            )

            self.assertEqual(encoded["record_count"], 2)
            self.assertEqual(encoded["chunk_count"], 1)
            self.assertTrue((root / "encoded" / "manifest.json").exists())
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_a"), records[1].tokens))
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_b"), records[0].tokens))
            self.assertEqual(decoded["record_count"], 2)

    def test_encode_corpus_supports_multiple_chunks(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import encode_corpus_global_prev_symbol_position_major

        records = [
            TokenRecord(f"clip_{index}", np.full((1, 8, 16), index, dtype=np.int16))
            for index in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = encode_corpus_global_prev_symbol_position_major(
                records=records,
                output_dir=root / "encoded",
                chunk_count=2,
            )

            self.assertEqual(result["chunk_count"], 2)
            self.assertTrue((root / "encoded" / "chunk_000.tpc").exists())
            self.assertTrue((root / "encoded" / "chunk_001.tpc").exists())

    def test_encode_corpus_rejects_non_positive_chunk_count(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import encode_corpus_global_prev_symbol_position_major

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "chunk_count"):
                encode_corpus_global_prev_symbol_position_major(
                    records=records,
                    output_dir=Path(tmpdir) / "encoded",
                    chunk_count=0,
                )


if __name__ == "__main__":
    unittest.main()
