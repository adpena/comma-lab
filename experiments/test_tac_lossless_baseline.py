from __future__ import annotations

import subprocess
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


class TacLosslessBaselineTests(unittest.TestCase):
    def test_build_prev_symbol_position_major_submission_uses_dataset_loader_and_packages_zip(self) -> None:
        from tac.lossless.codecs import _build_baseline_submission

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
                    {"json": {"file_name": "clip_a"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = _build_baseline_submission(
                profile="prev_symbol_position_major",
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
            )

            self.assertTrue(Path(result["archive_path"]).exists())
            self.assertTrue((root / "decompress.py").exists())
            self.assertTrue((root / "payload" / "clip_a.tpc").exists())
            self.assertTrue((root / "payload" / "clip_b.tpc").exists())
            self.assertFalse(result["local_only"])
            self.assertTrue(result["challenge_valid"])

        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})

    def test_render_prev_symbol_position_major_decompress_script_roundtrips_extensionless_names(self) -> None:
        from tac.lossless.arithmetic import flatten_tokens_for_gpt_arithmetic
        from tac.lossless.codecs import (
            render_prev_symbol_position_major_decompress_script,
            render_prev_symbol_position_major_runtime_module,
        )
        from tac.lossless.frequency_coder import encode_uint16_prev_symbol_stream

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "decoded"
            output_dir.mkdir()

            records = {
                "clip_a": np.arange(128, dtype=np.int16).reshape(1, 8, 16),
                "clip_b": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16),
            }
            for file_name, tokens in records.items():
                flat = flatten_tokens_for_gpt_arithmetic(tokens, layout="position_major").astype(np.uint16)
                encoded = encode_uint16_prev_symbol_stream(flat)
                (root / f"{file_name}.tpc").write_bytes(encoded.encoded_bytes)

            (root / "_lossless_prev_symbol_runtime.py").write_text(
                render_prev_symbol_position_major_runtime_module()
            )
            decompress_path = root / "decompress.py"
            decompress_path.write_text(render_prev_symbol_position_major_decompress_script())

            subprocess.run(
                [sys.executable, str(decompress_path)],
                cwd=root,
                check=True,
                env={**dict(), "OUTPUT_DIR": str(output_dir)},
            )

            self.assertTrue(np.array_equal(np.load(output_dir / "clip_a"), records["clip_a"]))
            self.assertTrue(np.array_equal(np.load(output_dir / "clip_b"), records["clip_b"]))

    def test_evaluate_prev_symbol_position_major_submission_returns_measured_result(self) -> None:
        from tac.lossless.codecs import evaluate_lossless_baseline_submission

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {
                "train": [
                    {"json": {"file_name": "clip_a"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = evaluate_lossless_baseline_submission(
                profile="prev_symbol_position_major",
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
            )

        self.assertEqual(result["command"], "lossless_baseline_evaluate")
        self.assertEqual(result["baseline"]["method"], "prev_symbol_position_major")
        self.assertTrue(result["challenge_valid"])
        self.assertFalse(result["local_only"])
        self.assertTrue(result["verification"]["exact_match"])
        self.assertGreater(result["compression"]["compression_rate"], 1.0)

    def test_build_lzma_baseline_submission_prefers_dataset_map_for_real_workloads(self) -> None:
        from tac.lossless.codecs import build_lzma_baseline_submission

        class FakeTrainSplit:
            def __init__(self) -> None:
                self.examples = [
                    {
                        "json": {"file_name": "clip_b.npy"},
                        "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16),
                    },
                    {
                        "json": {"file_name": "clip_a.npy"},
                        "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16),
                    },
                ]
                self.map_calls: list[dict[str, object]] = []

            def map(self, fn, *, desc=None, num_proc=None, load_from_cache_file=None, fn_kwargs=None):
                self.map_calls.append(
                    {
                        "desc": desc,
                        "num_proc": num_proc,
                        "load_from_cache_file": load_from_cache_file,
                    }
                )
                outputs = [fn(example, **(fn_kwargs or {})) for example in self.examples]
                return {"train": outputs}

            def __iter__(self):
                raise AssertionError("build_lzma_baseline_submission should use dataset.map when available")

        fake_split = FakeTrainSplit()
        calls = []

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            calls.append(
                {
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "data_files": data_files,
                }
            )
            return {"train": fake_split}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = build_lzma_baseline_submission(
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
                num_proc=3,
            )

            self.assertTrue(Path(result["archive_path"]).exists())
            self.assertTrue((root / "payload" / "clip_a.npy").exists())
            self.assertTrue((root / "payload" / "clip_b.npy").exists())

        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})
        self.assertEqual(len(fake_split.map_calls), 1)
        self.assertEqual(fake_split.map_calls[0]["desc"], "compress_example")
        self.assertEqual(fake_split.map_calls[0]["num_proc"], 3)
        self.assertFalse(fake_split.map_calls[0]["load_from_cache_file"])

    def test_build_lzma_baseline_submission_sums_columnar_map_output_without_row_iteration(self) -> None:
        from tac.lossless.codecs import build_lzma_baseline_submission

        class FakeMapResult:
            def __init__(self, archive_bytes: list[int]) -> None:
                self._archive_bytes = archive_bytes

            def __getitem__(self, key: str):
                if key != "archive_bytes":
                    raise KeyError(key)
                return list(self._archive_bytes)

            def __iter__(self):
                raise AssertionError("build_lzma_baseline_submission should prefer column access over row iteration")

        class FakeTrainSplit:
            def __init__(self) -> None:
                self.examples = [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
                self.num_rows = len(self.examples)

            def map(self, fn, *, desc=None, num_proc=None, load_from_cache_file=None, fn_kwargs=None):
                outputs = [fn(example, **(fn_kwargs or {})) for example in self.examples]
                return FakeMapResult([int(item["archive_bytes"]) for item in outputs])

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {"train": FakeTrainSplit()}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = build_lzma_baseline_submission(
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
                num_proc=1,
            )

        self.assertGreater(result["payload_bytes"], 0)

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

    def test_render_lzma_decompress_script_preserves_extensionless_commavq_names(self) -> None:
        from tac.lossless.codecs import compress_token_records, render_lzma_decompress_script
        from tac.lossless.data import TokenRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "decoded"
            output_dir.mkdir()
            records = [
                TokenRecord("clip_a", np.arange(128, dtype=np.int16).reshape(1, 8, 16)),
                TokenRecord("clip_b", (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)),
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

            self.assertTrue((output_dir / "clip_a").exists())
            self.assertTrue((output_dir / "clip_b").exists())
            self.assertTrue(np.array_equal(np.load(output_dir / "clip_a"), records[0].tokens))
            self.assertTrue(np.array_equal(np.load(output_dir / "clip_b"), records[1].tokens))

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

    def test_build_lzma_baseline_submission_reports_zip_bytes_as_archive_bytes(self) -> None:
        from tac.lossless.codecs import build_lzma_baseline_submission

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {
                "train": [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = build_lzma_baseline_submission(
                split=[0, 1],
                work_dir=root,
                dataset_loader=fake_loader,
                num_proc=1,
            )

            archive_path = Path(result["archive_path"])
            self.assertEqual(result["archive_bytes"], archive_path.stat().st_size)
            self.assertIn("payload_bytes", result)
            self.assertGreater(result["payload_bytes"], 0)

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

    def test_build_zpaq_baseline_submission_uses_dataset_loader_and_external_binary(self) -> None:
        from tac.lossless.codecs import build_zpaq_baseline_submission

        calls = []

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            calls.append({"dataset_name": dataset_name, "data_files": data_files})
            return {
                "train": [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        def fake_run(cmd, *, cwd=None, check=None, stdout=None, stderr=None):
            if cmd[1] != "add":
                raise AssertionError(f"unexpected command: {cmd}")
            Path(cmd[2]).write_bytes(b"zpaq-archive")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                mock.patch("tac.lossless.codecs.shutil.which", return_value="/usr/bin/zpaq"),
                mock.patch("tac.lossless.codecs.subprocess.run", side_effect=fake_run),
            ):
                result = build_zpaq_baseline_submission(
                    split=[0, 1],
                    work_dir=root,
                    dataset_loader=fake_loader,
                )

            self.assertTrue(Path(result["archive_path"]).exists())
            self.assertEqual(result["method"], "zpaq")
            self.assertTrue(result["local_only"])
            self.assertFalse(result["challenge_valid"])
            self.assertFalse(result["runtime_bundle_included"])
            self.assertIn("bundled", result["challenge_validity_reason"])
            self.assertTrue((root / "decompress.py").exists())
            self.assertTrue((root / "payload" / "clip_a.npy.zpaq").exists())
            self.assertTrue((root / "payload" / "clip_b.npy.zpaq").exists())

        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})

    def test_build_zpaq_baseline_submission_only_claims_challenge_valid_with_bundled_runtime(self) -> None:
        from tac.lossless.codecs import build_zpaq_baseline_submission

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {
                "train": [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                ]
            }

        def fake_run(cmd, *, cwd=None, check=None, stdout=None, stderr=None):
            if cmd[1] != "add":
                raise AssertionError(f"unexpected command: {cmd}")
            Path(cmd[2]).write_bytes(b"zpaq-archive")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundled_runtime = root / "zpaq-runtime"
            bundled_runtime.write_bytes(b"binary")
            with (
                mock.patch("tac.lossless.codecs.shutil.which", return_value="/usr/bin/zpaq"),
                mock.patch("tac.lossless.codecs.subprocess.run", side_effect=fake_run),
            ):
                result = build_zpaq_baseline_submission(
                    split=[0, 1],
                    work_dir=root,
                    dataset_loader=fake_loader,
                    runtime_bundle_path=bundled_runtime,
                )

            self.assertFalse(result["local_only"])
            self.assertTrue(result["challenge_valid"])
            self.assertTrue(result["runtime_bundle_included"])
            self.assertEqual(result["runtime_bundle_relpath"], "runtime/zpaq-runtime")
            self.assertTrue((root / "payload" / "runtime" / "zpaq-runtime").exists())

    def test_evaluate_zpaq_baseline_submission_returns_measured_result(self) -> None:
        from tac.lossless.codecs import evaluate_zpaq_baseline_submission

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            return {
                "train": [
                    {"json": {"file_name": "clip_a.npy"}, "token.npy": np.arange(128, dtype=np.int16).reshape(1, 8, 16)},
                    {"json": {"file_name": "clip_b.npy"}, "token.npy": (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)},
                ]
            }

        def fake_run(cmd, *, cwd=None, check=None, stdout=None, stderr=None, env=None):
            if cmd[1] == "add":
                Path(cmd[2]).write_bytes(b"zpaq-archive")
                return
            if cmd[1] == "extract":
                extract_dir = Path(cmd[cmd.index("-to") + 1])
                extract_dir.mkdir(parents=True, exist_ok=True)
                if "clip_a.npy.zpaq" in cmd[2]:
                    payload = np.arange(128, dtype=np.int16).reshape(1, 8, 16)
                    encoded = payload.reshape(-1, 128).T.ravel().tobytes()
                    (extract_dir / "clip_a.npy").write_bytes(encoded)
                else:
                    payload = (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16)
                    encoded = payload.reshape(-1, 128).T.ravel().tobytes()
                    (extract_dir / "clip_b.npy").write_bytes(encoded)
                return
            if Path(cmd[0]).name.startswith("python") and cmd[1].endswith("decompress.py"):
                output_dir = Path(env["OUTPUT_DIR"])
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / "clip_a.npy", np.arange(128, dtype=np.int16).reshape(1, 8, 16))
                np.save(output_dir / "clip_b.npy", (np.arange(128, dtype=np.int16) + 128).reshape(1, 8, 16))
                return
            raise AssertionError(f"unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                mock.patch("tac.lossless.codecs.shutil.which", return_value="/usr/bin/zpaq"),
                mock.patch("tac.lossless.codecs.subprocess.run", side_effect=fake_run),
            ):
                result = evaluate_zpaq_baseline_submission(
                    split=[0, 1],
                    work_dir=root,
                    dataset_loader=fake_loader,
                )

        self.assertEqual(result["command"], "lossless_baseline_evaluate")
        self.assertEqual(result["baseline"]["method"], "zpaq")
        self.assertTrue(result["local_only"])
        self.assertFalse(result["challenge_valid"])
        self.assertTrue(result["verification"]["exact_match"])
        self.assertGreater(result["compression"]["compression_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
