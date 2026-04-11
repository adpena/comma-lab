from __future__ import annotations

import importlib.util
import json
import tempfile
import os
import sys
import unittest
from pathlib import Path
from unittest import mock
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = ROOT / "src" / "tac" / "cli.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tac.cli", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TacCliTests(unittest.TestCase):
    def test_module_invocation_executes_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                **os.environ,
                "PYTHONPATH": str(SRC_ROOT),
            }
            result = subprocess.run(
                [sys.executable, "-m", "tac.cli", "lossless", "profiles"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lossless_profiles", result.stdout)

    def test_parser_exposes_lossy_and_lossless_subcommands(self) -> None:
        mod = load_module()
        parser = mod.build_parser()

        help_text = parser.format_help()

        self.assertIn("lossy", help_text)
        self.assertIn("lossless", help_text)

    def test_lossy_subcommand_parses_training_surface(self) -> None:
        mod = load_module()
        args = mod.build_parser().parse_args(
            [
                "lossy",
                "--tag",
                "demo",
                "--hidden",
                "32",
                "--variant",
                "dilated",
                "--epochs",
                "7",
            ]
        )

        self.assertEqual(args.command, "lossy")
        self.assertEqual(args.tag, "demo")
        self.assertEqual(args.hidden, 32)
        self.assertEqual(args.variant, "dilated")
        self.assertEqual(args.epochs, 7)

    def test_lossless_subcommand_exposes_named_profiles(self) -> None:
        mod = load_module()
        args = mod.build_parser().parse_args(["lossless", "profiles"])

        self.assertEqual(args.command, "lossless")
        self.assertEqual(args.lossless_command, "profiles")

    def test_lossless_plan_subcommand_builds_gpt_arithmetic_plan(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = mod.main(
                [
                    "lossless",
                    "plan",
                    "--profile",
                    "gpt_arithmetic_small",
                    "--work-dir",
                    str(root),
                    "--split",
                    "0",
                    "1",
                ]
            )

        self.assertEqual(result["command"], "lossless_plan")
        self.assertEqual(result["plan"]["profile"], "gpt_arithmetic_small")
        self.assertEqual(result["plan"]["method"], "gpt_arithmetic")
        self.assertEqual(result["plan"]["split"], ["0", "1"])

    def test_lossless_estimate_subcommand_reports_gpt_workload(self) -> None:
        mod = load_module()
        estimate_payload = {
            "profile": "gpt_arithmetic_small",
            "method": "gpt_arithmetic",
            "model": "small",
            "context_tokens": 256,
            "dataset_name": "commaai/commavq",
            "split": ["0", "1"],
            "work_dir": "/tmp/example",
            "status": "estimated",
            "measured": False,
            "example_count": 5000,
            "frames_per_example": 1200,
            "tokens_per_frame": 129,
            "flat_tokens_per_example": 154801,
            "total_flat_tokens": 774005000,
        }
        with mock.patch.object(
            mod,
            "estimate_gpt_arithmetic_workload",
            return_value=mod.GPTArithmeticEstimate(**estimate_payload),
        ) as mocked:
            result = mod.main(
                [
                    "lossless",
                    "estimate",
                    "--profile",
                    "gpt_arithmetic_small",
                    "--split",
                    "0",
                    "1",
                ]
            )

        mocked.assert_called_once_with("gpt_arithmetic_small", split=["0", "1"], work_dir=None)
        self.assertEqual(result["command"], "lossless_estimate")
        self.assertEqual(result["estimate"]["total_flat_tokens"], 774005000)

    def test_lossless_prepare_subcommand_materializes_gpt_token_stream(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "train.bin"
            with mock.patch.object(
                mod,
                "materialize_gpt_arithmetic_stream",
                return_value={
                    "command": "lossless_prepare",
                    "profile": "gpt_arithmetic_small",
                    "output_path": str(output),
                    "example_count": 5000,
                    "token_count": 774005000,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prepare",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--split",
                        "0",
                        "1",
                        "--output",
                        str(output),
                    ]
                )

        mocked.assert_called_once_with("gpt_arithmetic_small", split=["0", "1"], output_path=output)
        self.assertEqual(result["command"], "lossless_prepare")
        self.assertEqual(result["token_count"], 774005000)

    def test_lossless_package_subcommand_builds_submission_zip(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "payload"
            payload.mkdir()
            (payload / "tokens.bin").write_bytes(b"abc")
            decompress = root / "decompress.py"
            decompress.write_text("print('decode')\n")
            output = root / "submission.zip"

            result = mod.main(
                [
                    "lossless",
                    "package",
                    "--payload-dir",
                    str(payload),
                    "--decompress",
                    str(decompress),
                    "--output",
                    str(output),
                ]
            )

            self.assertTrue(Path(result["output"]).exists())

        self.assertEqual(result["command"], "lossless_package")

    def test_lossless_baseline_subcommand_routes_to_dataset_baseline_builder(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.object(
                mod,
                "evaluate_lossless_baseline_submission",
                return_value={
                    "command": "lossless_baseline_evaluate",
                    "compression": {"compression_rate": 3.4, "method": "lzma"},
                    "verification": {"exact_match": True},
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "baseline",
                        "--profile",
                        "lzma_baseline",
                        "--work-dir",
                        str(root),
                        "--split",
                        "0",
                        "1",
                    ]
                )

        mocked.assert_called_once_with(profile="lzma_baseline", split=["0", "1"], work_dir=root)
        self.assertEqual(result["command"], "lossless_baseline_evaluate")
        self.assertEqual(result["compression"]["compression_rate"], 3.4)

    def test_lossless_baseline_subcommand_routes_non_lzma_profiles_correctly(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.object(
                mod,
                "evaluate_lossless_baseline_submission",
                return_value={
                    "command": "lossless_baseline_evaluate",
                    "baseline": {"method": "zpaq"},
                    "compression": {"compression_rate": 2.3, "method": "zpaq"},
                    "verification": {"exact_match": True},
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "baseline",
                        "--profile",
                        "zpaq_baseline",
                        "--work-dir",
                        str(root),
                        "--split",
                        "0",
                        "1",
                    ]
                )

        mocked.assert_called_once_with(profile="zpaq_baseline", split=["0", "1"], work_dir=root)
        self.assertEqual(result["compression"]["method"], "zpaq")

    def test_lossless_compress_subcommand_runs_real_lzma_roundtrip(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = root / "original.bin"
            archive = root / "compressed.lzma"
            decompressed = root / "roundtrip.bin"
            original.write_bytes((b"hello lossless" * 16) + (b"\x00" * 64))

            result = mod.main(
                [
                    "lossless",
                    "compress",
                    "--profile",
                    "lzma_baseline",
                    "--input",
                    str(original),
                    "--output",
                    str(archive),
                    "--decompressed-output",
                    str(decompressed),
                ]
            )

            self.assertTrue(archive.exists())
            self.assertTrue(decompressed.exists())
            self.assertEqual(decompressed.read_bytes(), original.read_bytes())

        self.assertEqual(result["command"], "lossless_compress")
        self.assertEqual(result["compression"]["profile"], "lzma_baseline")
        self.assertEqual(result["compression"]["method"], "lzma")
        self.assertTrue(result["verification"]["exact_match"])

    def test_lossless_compress_subcommand_routes_zpaq_profile_to_codec_layer(self) -> None:
        mod = load_module()
        from tac.lossless.contracts import LosslessCompressionResult, LosslessVerificationResult

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = root / "original.bin"
            archive = root / "compressed.zpaq"
            decompressed = root / "roundtrip.bin"
            original.write_bytes(b"zpaq route test")

            def fake_compress_lossless_file(*, profile, input_path, output_path):
                Path(output_path).write_bytes(b"zpaq-arc")
                return LosslessCompressionResult(
                    profile="zpaq_baseline",
                    archive_path=str(output_path),
                    archive_bytes=8,
                    original_bytes=len(original.read_bytes()),
                    compression_rate=2.0,
                    method="zpaq",
                )

            def fake_decompress_lossless_file(*, profile, archive_path, output_path):
                Path(output_path).write_bytes(original.read_bytes())
                return Path(output_path)

            with (
                mock.patch.object(
                    mod,
                    "compress_lossless_file",
                    side_effect=fake_compress_lossless_file,
                ) as mocked_compress,
                mock.patch.object(
                    mod,
                    "decompress_lossless_file",
                    side_effect=fake_decompress_lossless_file,
                ) as mocked_decompress,
                mock.patch.object(
                    mod,
                    "evaluate_lossless_archive",
                    return_value=(
                        LosslessCompressionResult(
                            profile="zpaq_baseline",
                            archive_path=str(archive),
                            archive_bytes=8,
                            original_bytes=len(original.read_bytes()),
                            compression_rate=2.0,
                            method="zpaq",
                        ),
                        LosslessVerificationResult(
                            exact_match=True,
                            checked_items=len(original.read_bytes()),
                            mismatch_count=0,
                        ),
                    ),
                ) as mocked_eval,
            ):
                result = mod.main(
                    [
                        "lossless",
                        "compress",
                        "--profile",
                        "zpaq_baseline",
                        "--input",
                        str(original),
                        "--output",
                        str(archive),
                        "--decompressed-output",
                        str(decompressed),
                    ]
                )

        mocked_compress.assert_called_once()
        mocked_decompress.assert_called_once()
        mocked_eval.assert_called_once()
        self.assertEqual(result["compression"]["method"], "zpaq")
        self.assertTrue(result["verification"]["exact_match"])

    def test_lossless_evaluate_subcommand_reports_inverse_archive_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = root / "original.bin"
            decompressed = root / "decompressed.bin"
            archive = root / "submission.zip"
            original.write_bytes(b"abcd")
            decompressed.write_bytes(b"abcd")
            archive.write_bytes(b"xy")

            result = mod.main(
                [
                    "lossless",
                    "evaluate",
                    "--profile",
                    "lzma_baseline",
                    "--method",
                    "lzma",
                    "--original",
                    str(original),
                    "--decompressed",
                    str(decompressed),
                    "--archive",
                    str(archive),
                ]
            )

        self.assertEqual(result["compression"]["compression_rate"], 2.0)
        self.assertTrue(result["verification"]["exact_match"])

    def test_lossless_promote_updates_separate_lossless_state(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "result.json"
            reports = root / "reports"
            omx_state = root / ".omx" / "state"
            omx_research = root / ".omx" / "research"
            reports.mkdir(parents=True)
            omx_state.mkdir(parents=True)
            omx_research.mkdir(parents=True)
            archive_path = root / "submission.zip"
            archive_path.write_bytes(b"zip")
            (reports / "lossless_results.jsonl").write_text("")
            (reports / "lossless_timeline.jsonl").write_text("")
            (reports / "lossless_latest.md").write_text("stale\n")
            (omx_state / "lossless_focus.md").write_text("stale\n")
            (omx_state / "lossless_next_experiments.md").write_text("stale\n")
            (omx_research / "lossless_findings.md").write_text("stale\n")
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": 2,
                        "original_bytes": 4,
                        "compression_rate": 0.5,
                        "method": "lzma",
                    }
                )
            )

            result = mod.main(
                [
                    "lossless",
                    "promote",
                    "--result-json",
                    str(result_path),
                    "--repo-root",
                    str(root),
                ]
            )

            latest = (reports / "lossless_latest.md").read_text()
            ledger = (reports / "lossless_results.jsonl").read_text()

        self.assertEqual(result["command"], "lossless_promote")
        self.assertIn("0.5000", latest)
        self.assertIn("\"compression_rate\": 0.5", ledger)


if __name__ == "__main__":
    unittest.main()
