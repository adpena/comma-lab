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


class TacLosslessEvaluateTests(unittest.TestCase):
    def test_resolve_commavq_data_files_matches_public_challenge_split(self) -> None:
        from tac.lossless.data import resolve_commavq_data_files

        self.assertEqual(
            resolve_commavq_data_files("challenge"),
            {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]},
        )
        self.assertEqual(
            resolve_commavq_data_files(["data-0007.tar.gz"]),
            {"train": ["data-0007.tar.gz"]},
        )
        self.assertEqual(
            resolve_commavq_data_files([0, 1]),
            {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]},
        )
        self.assertEqual(
            resolve_commavq_data_files({"train": [2], "validation": ["data-0003.tar.gz"]}),
            {"train": ["data-0002.tar.gz"], "validation": ["data-0003.tar.gz"]},
        )

    def test_load_commavq_reference_records_uses_injected_loader_and_sorts_names(self) -> None:
        from tac.lossless.data import load_commavq_reference_records

        calls: list[dict[str, object]] = []

        def fake_loader(dataset_name: str, *, num_proc: int | None, data_files: dict[str, list[str]]):
            calls.append(
                {
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "data_files": data_files,
                }
            )
            return {
                "train": [
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": np.array([[2, 3]], dtype=np.int16)},
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.array([[0, 1]], dtype=np.int16)},
                ]
            }

        records = load_commavq_reference_records(
            split="challenge",
            dataset_loader=fake_loader,
            num_proc=4,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})
        self.assertEqual([record.file_name for record in records], ["clip_a.npy", "clip_b.npy"])
        self.assertTrue(np.array_equal(records[0].tokens, np.array([[0, 1]], dtype=np.int16)))

    def test_verify_exact_token_files_reports_missing_and_mismatch_files(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.evaluate import verify_exact_token_files

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "clip_a.npy", np.array([[1, 2], [3, 4]], dtype=np.int16))
            np.save(root / "clip_b.npy", np.array([[1, 2], [3, 9]], dtype=np.int16))

            records = [
                TokenRecord("clip_a.npy", np.array([[1, 2], [3, 4]], dtype=np.int16)),
                TokenRecord("clip_b.npy", np.array([[1, 2], [3, 4]], dtype=np.int16)),
                TokenRecord("clip_c.npy", np.array([[8, 8], [8, 8]], dtype=np.int16)),
            ]

            result = verify_exact_token_files(records, root)

        self.assertFalse(result.exact_match)
        self.assertEqual(result.checked_items, 3)
        self.assertEqual(result.mismatch_count, 2)

    def test_evaluate_lossless_archive_uses_file_compare_and_commavq_rate(self) -> None:
        from tac.lossless.data import TokenRecord, commavq_original_bytes
        from tac.lossless.evaluate import evaluate_lossless_archive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission.zip"
            archive_path.write_bytes(b"x" * 512)
            decompressed_root = root / "decompressed"
            decompressed_root.mkdir()

            records = [
                TokenRecord("clip_a.npy", np.array([[0, 1], [2, 3]], dtype=np.int16)),
                TokenRecord("clip_b.npy", np.array([[4, 5], [6, 7]], dtype=np.int16)),
            ]
            for record in records:
                np.save(decompressed_root / record.file_name, record.tokens)

            compression, verification = evaluate_lossless_archive(
                profile="lzma_baseline",
                reference_records=records,
                decompressed_root=decompressed_root,
                archive_path=archive_path,
                method="lzma",
            )

        self.assertTrue(verification.exact_match)
        self.assertEqual(verification.checked_items, 2)
        self.assertEqual(verification.mismatch_count, 0)
        self.assertEqual(compression.archive_bytes, 512)
        self.assertEqual(compression.original_bytes, commavq_original_bytes(2))
        self.assertAlmostEqual(compression.compression_rate, commavq_original_bytes(2) / 512)

    def test_evaluate_lossless_archive_rejects_non_exact_roundtrips(self) -> None:
        from tac.lossless.data import TokenRecord
        from tac.lossless.evaluate import evaluate_lossless_archive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission.zip"
            archive_path.write_bytes(b"x" * 256)
            decompressed_root = root / "decompressed"
            decompressed_root.mkdir()
            np.save(decompressed_root / "clip.npy", np.array([[1, 2], [3, 9]], dtype=np.int16))

            with self.assertRaisesRegex(ValueError, "exact"):
                evaluate_lossless_archive(
                    profile="lzma_baseline",
                    reference_records=[TokenRecord("clip.npy", np.array([[1, 2], [3, 4]], dtype=np.int16))],
                    decompressed_root=decompressed_root,
                    archive_path=archive_path,
                    method="lzma",
                )

    def test_evaluate_commavq_dataset_archive_loads_split_records_via_injected_loader(self) -> None:
        from tac.lossless.data import commavq_original_bytes
        from tac.lossless.evaluate import evaluate_commavq_dataset_archive

        calls: list[dict[str, object]] = []

        def fake_loader(dataset_name: str, *, num_proc: int | None, data_files: dict[str, list[str]]):
            calls.append(
                {
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "data_files": data_files,
                }
            )
            return {
                "train": [
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": np.array([[4, 5]], dtype=np.int16)},
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.array([[0, 1]], dtype=np.int16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission.zip"
            archive_path.write_bytes(b"x" * 400)
            decompressed_root = root / "decompressed"
            decompressed_root.mkdir()
            np.save(decompressed_root / "clip_a.npy", np.array([[0, 1]], dtype=np.int16))
            np.save(decompressed_root / "clip_b.npy", np.array([[4, 5]], dtype=np.int16))

            compression, verification = evaluate_commavq_dataset_archive(
                profile="lzma_baseline",
                archive_path=archive_path,
                decompressed_root=decompressed_root,
                method="lzma",
                split=[0, 1],
                dataset_loader=fake_loader,
                num_proc=2,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})
        self.assertTrue(verification.exact_match)
        self.assertEqual(compression.original_bytes, commavq_original_bytes(2))
        self.assertAlmostEqual(compression.compression_rate, commavq_original_bytes(2) / 400)


if __name__ == "__main__":
    unittest.main()
