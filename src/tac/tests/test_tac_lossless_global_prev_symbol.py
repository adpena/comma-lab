# SPDX-License-Identifier: MIT
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

    def test_encode_corpus_can_preserve_input_record_order(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import (
            decode_corpus_global_prev_symbol_position_major,
            encode_corpus_global_prev_symbol_position_major,
        )

        records = [
            TokenRecord("clip_c", np.full((1, 8, 16), 3, dtype=np.int16)),
            TokenRecord("clip_a", np.full((1, 8, 16), 1, dtype=np.int16)),
            TokenRecord("clip_b", np.full((1, 8, 16), 2, dtype=np.int16)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoded = encode_corpus_global_prev_symbol_position_major(
                records=records,
                output_dir=root / "encoded",
                preserve_input_order=True,
            )
            decoded = decode_corpus_global_prev_symbol_position_major(
                encoded_dir=root / "encoded",
                output_dir=root / "decoded",
            )

            manifest = (root / "encoded" / "manifest.json").read_text()
            self.assertIn('"preserve_input_order": true', manifest)
            self.assertEqual(encoded["record_order"], "input")
            self.assertEqual(decoded["record_count"], 3)
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_a"), records[1].tokens))
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_b"), records[2].tokens))
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_c"), records[0].tokens))

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

    def test_encode_decode_corpus_supports_recursive_bisect_frame_order(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import (
            decode_corpus_global_prev_symbol_position_major,
            encode_corpus_global_prev_symbol_position_major,
        )

        records = [
            TokenRecord("clip_a", np.arange(8 * 8 * 16, dtype=np.int16).reshape(8, 8, 16)),
            TokenRecord("clip_b", (np.arange(8 * 8 * 16, dtype=np.int16) + 1000).reshape(8, 8, 16)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoded = encode_corpus_global_prev_symbol_position_major(
                records=records,
                output_dir=root / "encoded",
                frame_order="recursive_bisect",
            )
            decoded = decode_corpus_global_prev_symbol_position_major(
                encoded_dir=root / "encoded",
                output_dir=root / "decoded",
            )

            self.assertEqual(encoded["frame_order"], "recursive_bisect")
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_a"), records[0].tokens))
            self.assertTrue(np.array_equal(np.load(root / "decoded" / "clip_b"), records[1].tokens))
            self.assertEqual(decoded["frame_order"], "recursive_bisect")

    def test_encode_corpus_rejects_unknown_frame_order(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import encode_corpus_global_prev_symbol_position_major

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "frame_order"):
                encode_corpus_global_prev_symbol_position_major(
                    records=records,
                    output_dir=Path(tmpdir) / "encoded",
                    frame_order="unknown",
                )

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

    def test_order_token_records_rejects_unknown_strategy(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with self.assertRaisesRegex(ValueError, "strategy"):
            order_token_records(records, strategy="unknown")

    def test_order_token_records_similarity_strategy_preserves_set_and_returns_noncanonical_order(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [
            TokenRecord("clip_a", np.zeros((2, 8, 16), dtype=np.int16)),
            TokenRecord("clip_b", np.full((2, 8, 16), 100, dtype=np.int16)),
            TokenRecord("clip_c", np.full((2, 8, 16), 1, dtype=np.int16)),
        ]

        ordered = order_token_records(records, strategy="clip_greedy_nn")

        self.assertEqual({record.file_name for record in ordered}, {"clip_a", "clip_b", "clip_c"})
        self.assertEqual(ordered[0].file_name, "clip_c")

    def test_order_token_records_label_grouped_strategy_requires_label_map(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with self.assertRaisesRegex(ValueError, "label_map"):
            order_token_records(records, strategy="label_grouped_clip_greedy_nn")

    def test_order_token_records_label_grouped_strategy_honors_label_groups(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [
            TokenRecord("clip_b", np.full((2, 8, 16), 100, dtype=np.int16)),
            TokenRecord("clip_a", np.zeros((2, 8, 16), dtype=np.int16)),
            TokenRecord("clip_d", np.full((2, 8, 16), 200, dtype=np.int16)),
            TokenRecord("clip_c", np.full((2, 8, 16), 1, dtype=np.int16)),
        ]

        ordered = order_token_records(
            records,
            strategy="label_grouped_clip_greedy_nn",
            label_map={
                "clip_a": "urban",
                "clip_c": "urban",
                "clip_b": "highway",
                "clip_d": "highway",
            },
        )

        first_group = {record.file_name for record in ordered[:2]}
        second_group = {record.file_name for record in ordered[2:]}

        self.assertIn(first_group, ({"clip_a", "clip_c"}, {"clip_b", "clip_d"}))
        self.assertIn(second_group, ({"clip_a", "clip_c"}, {"clip_b", "clip_d"}))
        self.assertNotEqual(first_group, second_group)

    def test_order_token_records_label_lexicographic_clip_rank_requires_label_map(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with self.assertRaisesRegex(ValueError, "label_map"):
            order_token_records(records, strategy="label_lexicographic_clip_rank")

    def test_order_token_records_label_lexicographic_clip_rank_sorts_by_label_then_rank(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [
            TokenRecord("clip_d", np.full((2, 8, 16), 7, dtype=np.int16)),
            TokenRecord("clip_b", np.full((2, 8, 16), 3, dtype=np.int16)),
            TokenRecord("clip_c", np.full((2, 8, 16), 7, dtype=np.int16)),
            TokenRecord("clip_a", np.full((2, 8, 16), 1, dtype=np.int16)),
        ]

        ordered = order_token_records(
            records,
            strategy="label_lexicographic_clip_rank",
            label_map={
                "clip_a": [0, 1],
                "clip_b": [0, 1],
                "clip_c": [1, 0],
                "clip_d": [1, 0],
            },
        )

        self.assertEqual(
            [record.file_name for record in ordered],
            ["clip_a", "clip_b", "clip_c", "clip_d"],
        )

    def test_order_token_records_hybrid_thresh8_parent046_label_greedy_uses_dense_labels_parent_fallback_and_greedy_bucket_order(
        self,
    ) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        def make_record(file_name: str, value: int) -> TokenRecord:
            return TokenRecord(file_name, np.full((2, 8, 16), value, dtype=np.int16))

        dense_names = [f"dense_{index}" for index in range(9)]
        dense_values = {name: 40 + index for index, name in enumerate(dense_names)}
        same_parent_values = {
            "same_parent_0": 50,
            "same_parent_1": 51,
        }
        other_parent_values = {
            "other_parent_0": 0,
            "other_parent_1": 1,
            "other_parent_2": 2,
            "other_parent_3": 3,
        }

        records = [
            make_record("same_parent_1", same_parent_values["same_parent_1"]),
            make_record("dense_4", dense_values["dense_4"]),
            make_record("other_parent_2", other_parent_values["other_parent_2"]),
            make_record("dense_0", dense_values["dense_0"]),
            make_record("same_parent_0", same_parent_values["same_parent_0"]),
            make_record("dense_8", dense_values["dense_8"]),
            make_record("other_parent_0", other_parent_values["other_parent_0"]),
            make_record("dense_3", dense_values["dense_3"]),
            make_record("dense_6", dense_values["dense_6"]),
            make_record("other_parent_3", other_parent_values["other_parent_3"]),
            make_record("dense_2", dense_values["dense_2"]),
            make_record("other_parent_1", other_parent_values["other_parent_1"]),
            make_record("dense_5", dense_values["dense_5"]),
            make_record("dense_1", dense_values["dense_1"]),
            make_record("dense_7", dense_values["dense_7"]),
        ]

        label_map = {
            **{name: [1, 0, 0, 0, 1, 0, 1, 0] for name in dense_names},
            "same_parent_0": [1, 2, 0, 0, 1, 0, 1, 0],
            "same_parent_1": [1, 3, 0, 0, 1, 1, 1, 0],
            "other_parent_0": [0, 0, 0, 0, 0, 0, 0, 0],
            "other_parent_1": [0, 0, 0, 1, 0, 0, 0, 0],
            "other_parent_2": [0, 1, 0, 0, 0, 1, 0, 0],
            "other_parent_3": [0, 1, 0, 1, 0, 1, 0, 0],
        }

        ordered = order_token_records(
            records,
            strategy="hybrid_thresh8_parent046_label_greedy",
            label_map=label_map,
        )

        self.assertEqual(
            [record.file_name for record in ordered],
            [
                "dense_0",
                "dense_1",
                "dense_2",
                "dense_3",
                "dense_4",
                "dense_5",
                "dense_6",
                "dense_7",
                "dense_8",
                "same_parent_0",
                "same_parent_1",
                "other_parent_0",
                "other_parent_1",
                "other_parent_2",
                "other_parent_3",
            ],
        )

    def test_order_token_records_hybrid_thresh8_parent046_label_greedy_requires_label_map(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [TokenRecord("clip", np.arange(128, dtype=np.int16).reshape(1, 8, 16))]

        with self.assertRaisesRegex(ValueError, "label_map"):
            order_token_records(records, strategy="hybrid_thresh8_parent046_label_greedy")

    def test_order_token_records_explicit_strategy_replays_saved_order(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.global_prev_symbol import order_token_records

        records = [
            TokenRecord("clip_a", np.zeros((1, 8, 16), dtype=np.int16)),
            TokenRecord("clip_b", np.ones((1, 8, 16), dtype=np.int16)),
            TokenRecord("clip_c", np.full((1, 8, 16), 2, dtype=np.int16)),
        ]

        ordered = order_token_records(
            records,
            strategy="explicit",
            explicit_order=["clip_c", "clip_a", "clip_b"],
        )

        self.assertEqual([record.file_name for record in ordered], ["clip_c", "clip_a", "clip_b"])


if __name__ == "__main__":
    unittest.main()
