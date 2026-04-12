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


ROOT = Path(__file__).resolve().parents[3]
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

    def test_lossless_tiny_frame_predictor_summary_subcommand_reports_architecture_budget(self) -> None:
        mod = load_module()
        with mock.patch.object(
            mod,
            "summarize_tiny_frame_predictor",
            return_value={
                "command": "lossless_tiny_frame_predictor_summary",
                "profile": "tiny_frame_predictor_small",
                "parameter_count": 123456,
                "context_frames": 8,
                "positions": 128,
            },
        ) as mocked:
            result = mod.main(
                [
                    "lossless",
                    "tiny-frame-predictor-summary",
                    "--profile",
                    "tiny_frame_predictor_small",
                ]
            )

        mocked.assert_called_once()
        self.assertEqual(result["command"], "lossless_tiny_frame_predictor_summary")
        self.assertEqual(result["parameter_count"], 123456)

    def test_lossless_hybrid_select_subcommand_parses_inputs_and_metrics(self) -> None:
        mod = load_module()
        args = mod.build_parser().parse_args(
            [
                "lossless",
                "hybrid-select",
                "--input",
                "/tmp/candidate_a.json",
                "--input",
                "/tmp/candidate_b.json",
                "--output",
                "/tmp/selected.json",
                "--metric",
                "archive_bytes:min",
                "--metric",
                "encode_seconds:min",
            ]
        )

        self.assertEqual(args.command, "lossless")
        self.assertEqual(args.lossless_command, "hybrid-select")
        self.assertEqual(
            args.input,
            ["/tmp/candidate_a.json", "/tmp/candidate_b.json"],
        )
        self.assertEqual(args.metric, ["archive_bytes:min", "encode_seconds:min"])
        self.assertEqual(args.output, "/tmp/selected.json")

    def test_lossless_hybrid_select_subcommand_ranks_exact_candidate_json(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_a = root / "candidate_a.json"
            candidate_b = root / "candidate_b.json"
            output = root / "selected.json"
            candidate_a.write_text(
                json.dumps(
                    {
                        "strategy": "baseline",
                        "exact_match": True,
                        "archive_bytes": 620,
                        "encode_seconds": 4.0,
                    }
                ),
                encoding="utf-8",
            )
            candidate_b.write_text(
                json.dumps(
                    {
                        "strategy": "frontier",
                        "exact_match": True,
                        "archive_bytes": 600,
                        "encode_seconds": 5.0,
                    }
                ),
                encoding="utf-8",
            )

            result = mod.main(
                [
                    "lossless",
                    "hybrid-select",
                    "--input",
                    str(candidate_a),
                    "--input",
                    str(candidate_b),
                    "--output",
                    str(output),
                    "--metric",
                    "archive_bytes:min",
                    "--metric",
                    "encode_seconds:min",
                ]
            )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["command"], "lossless_hybrid_select")
        self.assertEqual(result["selected_input"], str(candidate_b))
        self.assertEqual(result["ranked_inputs"], [str(candidate_b), str(candidate_a)])
        self.assertEqual(result["selected_summary"]["strategy"], "frontier")
        self.assertEqual(written["selected_input"], str(candidate_b))
        self.assertEqual(written["ranked_inputs"], [str(candidate_b), str(candidate_a)])

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
                    "--layout",
                    "position_major",
                ]
            )

        self.assertEqual(result["command"], "lossless_plan")
        self.assertEqual(result["plan"]["profile"], "gpt_arithmetic_small")
        self.assertEqual(result["plan"]["method"], "gpt_arithmetic")
        self.assertEqual(result["plan"]["layout"], "position_major")
        self.assertEqual(result["plan"]["split"], ["0", "1"])

    def test_lossless_estimate_subcommand_defaults_split_to_challenge_alias(self) -> None:
        mod = load_module()
        estimate_payload = {
            "profile": "gpt_arithmetic_small",
            "method": "gpt_arithmetic",
            "model": "small",
            "context_tokens": 256,
            "dataset_name": "commaai/commavq",
            "layout": "frame_major",
            "split": ["challenge"],
            "work_dir": None,
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
                ]
            )

        mocked.assert_called_once_with("gpt_arithmetic_small", split=None, work_dir=None, layout="frame_major")
        self.assertEqual(result["estimate"]["split"], ["challenge"])

    def test_lossless_estimate_subcommand_reports_gpt_workload(self) -> None:
        mod = load_module()
        estimate_payload = {
            "profile": "gpt_arithmetic_small",
            "method": "gpt_arithmetic",
            "model": "small",
            "context_tokens": 256,
            "dataset_name": "commaai/commavq",
            "layout": "position_major",
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
                    "--layout",
                    "position_major",
                ]
            )

        mocked.assert_called_once_with("gpt_arithmetic_small", split=["0", "1"], work_dir=None, layout="position_major")
        self.assertEqual(result["command"], "lossless_estimate")
        self.assertEqual(result["estimate"]["layout"], "position_major")
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
                    "layout": "position_major",
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
                    "--layout",
                    "position_major",
                    "--output",
                    str(output),
                ]
            )

        mocked.assert_called_once_with("gpt_arithmetic_small", split=["0", "1"], output_path=output, layout="position_major")
        self.assertEqual(result["command"], "lossless_prepare")
        self.assertEqual(result["layout"], "position_major")
        self.assertEqual(result["token_count"], 774005000)

    def test_lossless_frequency_subcommand_writes_symbol_report(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            report_path = root / "freq.json"
            with mock.patch.object(
                mod,
                "write_symbol_frequency_report",
                return_value={
                    "command": "lossless_frequency_report",
                    "token_path": str(token_path),
                    "token_count": 10,
                    "unique_symbols": 3,
                    "empirical_bits_per_token": 1.2,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "frequency-report",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(report_path),
                    ]
                )

        mocked.assert_called_once_with(token_path=token_path, output_path=report_path)
        self.assertEqual(result["command"], "lossless_frequency_report")
        self.assertEqual(result["empirical_bits_per_token"], 1.2)

    def test_lossless_gpt_score_subcommand_reports_metrics(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "gpt_score.json"
            with mock.patch.object(
                mod,
                "score_commavq_gpt_sample",
                return_value={
                    "command": "lossless_gpt_score_sample",
                    "token_path": str(token_path),
                    "output_path": str(output_path),
                    "profile": "gpt_arithmetic_small",
                    "layout": "frame_major",
                    "device": "mps",
                    "dtype": "float32",
                    "scored_tokens": 32,
                    "bits_per_token": 1.25,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "gpt-score",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "mps",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            output_path=output_path,
            profile="gpt_arithmetic_small",
            max_scored_tokens=None,
            context_tokens=None,
            device="mps",
            dtype="auto",
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_gpt_score_sample")
        self.assertEqual(result["device"], "mps")
        self.assertEqual(result["bits_per_token"], 1.25)

    def test_lossless_gpt_score_subcommand_supports_context_override(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "gpt_score.json"
            with mock.patch.object(
                mod,
                "score_commavq_gpt_sample",
                return_value={
                    "command": "lossless_gpt_score_sample",
                    "token_path": str(token_path),
                    "output_path": str(output_path),
                    "profile": "gpt_arithmetic_small",
                    "layout": "frame_major",
                    "device": "cpu",
                    "dtype": "float32",
                    "context_tokens": 2580,
                    "scored_tokens": 64,
                    "bits_per_token": 2.0,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "gpt-score",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--context-tokens",
                        "2580",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            output_path=output_path,
            profile="gpt_arithmetic_small",
            max_scored_tokens=None,
            context_tokens=2580,
            device="auto",
            dtype="auto",
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["context_tokens"], 2580)

    def test_lossless_gpt_score_probe_subcommand_reports_fastest_backend(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "probe.json"
            with mock.patch.object(
                mod,
                "probe_commavq_gpt_devices",
                return_value={
                    "command": "lossless_gpt_score_probe",
                    "token_path": str(token_path),
                    "output_path": str(output_path),
                    "fastest_device": "cpu",
                    "best_bits_device": "cpu",
                    "context_tokens": 2580,
                    "results": [
                        {"device": "cpu", "seconds": 1.0, "bits_per_token": 9.2},
                        {"device": "mps", "seconds": 3.5, "bits_per_token": 9.2},
                    ],
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "gpt-score-probe",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "cpu",
                        "--device",
                        "mps",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            output_path=output_path,
            profile="gpt_arithmetic_small",
            max_scored_tokens=64,
            context_tokens=None,
            devices=("cpu", "mps"),
            dtype="auto",
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_gpt_score_probe")
        self.assertEqual(result["fastest_device"], "cpu")

    def test_lossless_gpt_arithmetic_sample_subcommand_reports_encoded_bytes(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "sample.gta"
            with mock.patch.object(
                mod,
                "encode_commavq_gpt_sample",
                return_value={
                    "command": "lossless_gpt_arithmetic_sample",
                    "token_path": str(token_path),
                    "encoded_path": str(output_path),
                    "device": "mps",
                    "token_count": 256,
                    "encoded_bytes": 64,
                    "compression_ratio": 8.0,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "gpt-arithmetic-sample",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "mps",
                        "--max-tokens",
                        "256",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            encoded_path=output_path,
            profile="gpt_arithmetic_small",
            max_tokens=256,
            context_tokens=None,
            device="mps",
            dtype="auto",
            verify_decode=False,
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_gpt_arithmetic_sample")
        self.assertEqual(result["compression_ratio"], 8.0)

    def test_lossless_gpt_arithmetic_global_sample_subcommand_reports_encoded_bytes(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "sample.gta"
            with mock.patch.object(
                mod,
                "encode_commavq_gpt_global_sample",
                return_value={
                    "command": "lossless_gpt_arithmetic_global_sample",
                    "token_path": str(token_path),
                    "encoded_path": str(output_path),
                    "device": "cpu",
                    "token_count": 512,
                    "encoded_bytes": 96,
                    "compression_ratio": 10.666666666666666,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "gpt-arithmetic-global-sample",
                        "--profile",
                        "gpt_arithmetic_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "cpu",
                        "--max-tokens",
                        "512",
                        "--verify-decode",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            encoded_path=output_path,
            profile="gpt_arithmetic_small",
            max_tokens=512,
            context_tokens=None,
            device="cpu",
            dtype="auto",
            verify_decode=True,
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_gpt_arithmetic_global_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_token_rgb_sample_subcommand_reports_rgb_cache(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.npy"
            output_path = root / "frames.npy"
            commavq_root = root / "commavq"
            with mock.patch.object(
                mod,
                "decode_commavq_token_file_to_rgb",
                return_value={
                    "command": "lossless_token_rgb_sample",
                    "token_path": str(token_path),
                    "output_path": str(output_path),
                    "frame_count": 16,
                    "frame_shape": [128, 256, 3],
                    "batch_size": 8,
                    "device": "mps",
                    "dtype": "float16",
                    "commavq_root": str(commavq_root),
                    "measured": False,
                    "local_only": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "token-rgb-sample",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--max-frames",
                        "16",
                        "--batch-size",
                        "8",
                        "--device",
                        "mps",
                        "--dtype",
                        "float16",
                        "--commavq-root",
                        str(commavq_root),
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            output_path=output_path,
            max_frames=16,
            batch_size=8,
            device="mps",
            dtype="float16",
            commavq_root=commavq_root,
            decoder_url=mod.OFFICIAL_DECODER_URL,
        )
        self.assertEqual(result["command"], "lossless_token_rgb_sample")
        self.assertEqual(result["frame_count"], 16)

    def test_lossless_pose_labels_sample_subcommand_reports_label_map(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "pose_labels.json"
            with mock.patch.object(
                mod,
                "build_pose_label_map_sample",
                return_value={
                    "command": "lossless_pose_labels_sample",
                    "output_path": str(output_path),
                    "record_count": 128,
                    "split": ["0", "1"],
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "pose-labels-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "128",
                        "--split",
                        "0",
                        "1",
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=["0", "1"],
            max_records=128,
        )
        self.assertEqual(result["command"], "lossless_pose_labels_sample")
        self.assertEqual(result["record_count"], 128)

    def test_lossless_rgb_labels_sample_subcommand_reports_label_map(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "rgb_labels.json"
            commavq_root = root / "commavq"
            with mock.patch.object(
                mod,
                "build_rgb_label_map_sample",
                return_value={
                    "command": "lossless_rgb_labels_sample",
                    "output_path": str(output_path),
                    "record_count": 64,
                    "split": ["0", "1"],
                    "max_keyframes": 6,
                    "local_only": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "rgb-labels-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--split",
                        "0",
                        "1",
                        "--max-keyframes",
                        "6",
                        "--batch-size",
                        "8",
                        "--device",
                        "mps",
                        "--dtype",
                        "float16",
                        "--commavq-root",
                        str(commavq_root),
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=["0", "1"],
            max_records=64,
            bridge_loader=mod.load_official_commavq_bridge,
            batch_size=8,
            max_keyframes=6,
            device="mps",
            dtype="float16",
            commavq_root=commavq_root,
            decoder_url=mod.OFFICIAL_DECODER_URL,
        )
        self.assertEqual(result["command"], "lossless_rgb_labels_sample")
        self.assertEqual(result["record_count"], 64)

    def test_lossless_next_frame_sample_subcommand_reports_encoded_bytes(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "sample.nfg"
            with mock.patch.object(
                mod,
                "encode_commavq_next_frame_sample",
                return_value={
                    "command": "lossless_next_frame_sample",
                    "token_path": str(token_path),
                    "encoded_path": str(output_path),
                    "device": "mps",
                    "frame_count": 32,
                    "encoded_bytes": 128,
                    "compression_ratio": 4.0,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "next-frame-sample",
                        "--profile",
                        "gpt_next_frame_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "mps",
                        "--max-frames",
                        "32",
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            encoded_path=output_path,
            profile="gpt_next_frame_small",
            max_frames=32,
            context_frames=None,
            device="mps",
            dtype="auto",
            verify_decode=False,
            cache_dir=None,
            model_url=None,
            gpt_module_path=None,
        )
        self.assertEqual(result["command"], "lossless_next_frame_sample")
        self.assertEqual(result["compression_ratio"], 4.0)

    def test_lossless_next_frame_sample_subcommand_supports_context_override_and_verify_decode(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            output_path = root / "sample.nfg"
            gpt_module_path = root / "gpt.py"
            with mock.patch.object(
                mod,
                "encode_commavq_next_frame_sample",
                return_value={
                    "command": "lossless_next_frame_sample",
                    "token_path": str(token_path),
                    "encoded_path": str(output_path),
                    "device": "cpu",
                    "frame_count": 8,
                    "encoded_bytes": 64,
                    "compression_ratio": 2.0,
                    "exact_match": True,
                    "local_only": True,
                    "measured": False,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "next-frame-sample",
                        "--profile",
                        "gpt_next_frame_small",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(output_path),
                        "--context-frames",
                        "2",
                        "--device",
                        "cpu",
                        "--dtype",
                        "float32",
                        "--verify-decode",
                        "--gpt-module-path",
                        str(gpt_module_path),
                    ]
                )

        mocked.assert_called_once_with(
            token_path=token_path,
            encoded_path=output_path,
            profile="gpt_next_frame_small",
            max_frames=32,
            context_frames=2,
            device="cpu",
            dtype="float32",
            verify_decode=True,
            cache_dir=None,
            model_url=None,
            gpt_module_path=str(gpt_module_path),
        )
        self.assertTrue(result["local_only"])
        self.assertFalse(result["measured"])
        self.assertTrue(result["exact_match"])

    def test_lossless_frequency_encode_subcommand_writes_coded_stream(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            encoded_path = root / "train.tfc"
            with mock.patch.object(
                mod,
                "encode_uint16_frequency_file",
                return_value={
                    "command": "lossless_frequency_encode",
                    "source_path": str(token_path),
                    "encoded_path": str(encoded_path),
                    "token_count": 10,
                    "unique_symbols": 3,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "frequency-encode",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(encoded_path),
                    ]
                )

        mocked.assert_called_once_with(token_path, encoded_path)
        self.assertEqual(result["command"], "lossless_frequency_encode")
        self.assertEqual(result["token_count"], 10)

    def test_lossless_global_prev_symbol_order_sample_subcommand_reports_exact_result(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "order.json"
            with mock.patch.object(
                mod,
                "benchmark_global_prev_symbol_record_order_sample",
                return_value={
                    "command": "lossless_global_prev_symbol_order_sample",
                    "strategy": "clip_recursive_pca",
                    "max_records": 64,
                    "output_path": str(output_path),
                    "archive_bytes": 123,
                    "original_bytes": 456,
                    "compression_ratio": 3.707317073170732,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "global-prev-symbol-order-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--strategy",
                        "clip_recursive_pca",
                        "--split",
                        "0",
                        "1",
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=["0", "1"],
            max_records=64,
            strategy="clip_recursive_pca",
            frame_order="canonical",
            labels_path=None,
            order_path=None,
        )
        self.assertEqual(result["command"], "lossless_global_prev_symbol_order_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_global_prev_symbol_order_sample_subcommand_accepts_label_file(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "order.json"
            labels_path = root / "labels.json"
            with mock.patch.object(
                mod,
                "benchmark_global_prev_symbol_record_order_sample",
                return_value={
                    "command": "lossless_global_prev_symbol_order_sample",
                    "strategy": "label_grouped_clip_greedy_nn",
                    "max_records": 64,
                    "output_path": str(output_path),
                    "archive_bytes": 123,
                    "original_bytes": 456,
                    "compression_ratio": 3.707317073170732,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "global-prev-symbol-order-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--strategy",
                        "label_grouped_clip_greedy_nn",
                        "--labels",
                        str(labels_path),
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=None,
            max_records=64,
            strategy="label_grouped_clip_greedy_nn",
            frame_order="canonical",
            labels_path=labels_path,
            order_path=None,
        )
        self.assertEqual(result["command"], "lossless_global_prev_symbol_order_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_global_prev_symbol_order_sample_subcommand_accepts_order_file(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "order.json"
            order_path = root / "saved-order.json"
            with mock.patch.object(
                mod,
                "benchmark_global_prev_symbol_record_order_sample",
                return_value={
                    "command": "lossless_global_prev_symbol_order_sample",
                    "strategy": "explicit",
                    "max_records": 64,
                    "output_path": str(output_path),
                    "archive_bytes": 123,
                    "original_bytes": 456,
                    "compression_ratio": 3.707317073170732,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "global-prev-symbol-order-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--strategy",
                        "explicit",
                        "--order-file",
                        str(order_path),
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=None,
            max_records=64,
            strategy="explicit",
            frame_order="canonical",
            labels_path=None,
            order_path=order_path,
        )
        self.assertEqual(result["command"], "lossless_global_prev_symbol_order_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_global_prev_symbol_order_sample_subcommand_accepts_label_lexicographic_strategy(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "order.json"
            labels_path = root / "labels.json"
            with mock.patch.object(
                mod,
                "benchmark_global_prev_symbol_record_order_sample",
                return_value={
                    "command": "lossless_global_prev_symbol_order_sample",
                    "strategy": "label_lexicographic_clip_rank",
                    "max_records": 64,
                    "output_path": str(output_path),
                    "archive_bytes": 123,
                    "original_bytes": 456,
                    "compression_ratio": 3.707317073170732,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "global-prev-symbol-order-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--strategy",
                        "label_lexicographic_clip_rank",
                        "--labels",
                        str(labels_path),
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=None,
            max_records=64,
            strategy="label_lexicographic_clip_rank",
            frame_order="canonical",
            labels_path=labels_path,
            order_path=None,
        )
        self.assertEqual(result["command"], "lossless_global_prev_symbol_order_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_global_prev_symbol_order_sample_subcommand_accepts_hybrid_thresh8_strategy(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "order.json"
            labels_path = root / "labels.json"
            with mock.patch.object(
                mod,
                "benchmark_global_prev_symbol_record_order_sample",
                return_value={
                    "command": "lossless_global_prev_symbol_order_sample",
                    "strategy": "hybrid_thresh8_parent046_label_greedy",
                    "max_records": 64,
                    "output_path": str(output_path),
                    "archive_bytes": 123,
                    "original_bytes": 456,
                    "compression_ratio": 3.707317073170732,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "global-prev-symbol-order-sample",
                        "--output",
                        str(output_path),
                        "--max-records",
                        "64",
                        "--strategy",
                        "hybrid_thresh8_parent046_label_greedy",
                        "--labels",
                        str(labels_path),
                    ]
                )

        mocked.assert_called_once_with(
            output_path=output_path,
            split=None,
            max_records=64,
            strategy="hybrid_thresh8_parent046_label_greedy",
            frame_order="canonical",
            labels_path=labels_path,
            order_path=None,
        )
        self.assertEqual(result["command"], "lossless_global_prev_symbol_order_sample")
        self.assertTrue(result["exact_match"])

    def test_lossless_prev_symbol_encode_subcommand_writes_coded_stream(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            encoded_path = root / "train.tpc"
            with mock.patch.object(
                mod,
                "encode_uint16_prev_symbol_file",
                return_value={
                    "command": "lossless_prev_symbol_encode",
                    "source_path": str(token_path),
                    "encoded_path": str(encoded_path),
                    "token_count": 10,
                    "context_count": 4,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-symbol-encode",
                        "--tokens",
                        str(token_path),
                        "--output",
                        str(encoded_path),
                    ]
                )

        mocked.assert_called_once_with(token_path, encoded_path)
        self.assertEqual(result["command"], "lossless_prev_symbol_encode")
        self.assertEqual(result["context_count"], 4)

    def test_lossless_prev_symbol_decode_subcommand_restores_coded_stream(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoded_path = root / "train.tpc"
            restored_path = root / "restored.bin"
            with mock.patch.object(
                mod,
                "decode_uint16_prev_symbol_file",
                return_value=str(restored_path),
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-symbol-decode",
                        "--encoded",
                        str(encoded_path),
                        "--output",
                        str(restored_path),
                    ]
                )

        mocked.assert_called_once_with(encoded_path, restored_path)
        self.assertEqual(result["command"], "lossless_prev_symbol_decode")
        self.assertEqual(result["restored_path"], str(restored_path))

    def test_lossless_prev_symbol_benchmark_subcommand_reports_conditional_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            with mock.patch.object(
                mod,
                "benchmark_prev_symbol_frequency_file",
                return_value={
                    "command": "lossless_prev_symbol_frequency_benchmark",
                    "token_count": 10,
                    "context_count": 4,
                    "encoded_bytes": 20,
                    "compression_ratio": 1.0,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-symbol-benchmark",
                        "--tokens",
                        str(token_path),
                    ]
                )

        mocked.assert_called_once_with(token_path, max_tokens=None)
        self.assertEqual(result["command"], "lossless_prev_symbol_frequency_benchmark")
        self.assertEqual(result["context_count"], 4)

    def test_lossless_prev_symbol_benchmark_subcommand_supports_max_tokens(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            with mock.patch.object(
                mod,
                "benchmark_prev_symbol_frequency_file",
                return_value={
                    "command": "lossless_prev_symbol_frequency_benchmark",
                    "token_count": 1,
                    "context_count": 1,
                    "encoded_bytes": 4,
                    "compression_ratio": 0.5,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-symbol-benchmark",
                        "--tokens",
                        str(token_path),
                        "--max-tokens",
                        "1",
                    ]
                )

        mocked.assert_called_once_with(token_path, max_tokens=1)
        self.assertEqual(result["token_count"], 1)

    def test_lossless_prev_pair_benchmark_subcommand_reports_conditional_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            with mock.patch.object(
                mod,
                "benchmark_prev_pair_frequency_file",
                return_value={
                    "command": "lossless_prev_pair_frequency_benchmark",
                    "token_count": 10,
                    "context_count": 5,
                    "encoded_bytes": 18,
                    "compression_ratio": 1.1,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-pair-benchmark",
                        "--tokens",
                        str(token_path),
                    ]
                )

        mocked.assert_called_once_with(token_path, max_tokens=None)
        self.assertEqual(result["command"], "lossless_prev_pair_frequency_benchmark")
        self.assertEqual(result["context_count"], 5)

    def test_lossless_prev_pair_benchmark_subcommand_supports_max_tokens(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "train.bin"
            with mock.patch.object(
                mod,
                "benchmark_prev_pair_frequency_file",
                return_value={
                    "command": "lossless_prev_pair_frequency_benchmark",
                    "token_count": 2,
                    "context_count": 1,
                    "encoded_bytes": 6,
                    "compression_ratio": 0.66,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "prev-pair-benchmark",
                        "--tokens",
                        str(token_path),
                        "--max-tokens",
                        "2",
                    ]
                )

        mocked.assert_called_once_with(token_path, max_tokens=2)
        self.assertEqual(result["token_count"], 2)

    def test_lossless_zstd_dict_benchmark_subcommand_reports_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.bin"
            compressed = root / "tokens.zst"
            restored = root / "restored.bin"
            sample_a = root / "sample_a.bin"
            sample_b = root / "sample_b.bin"
            with mock.patch.object(
                mod,
                "benchmark_zstd_dict_file",
                return_value={
                    "command": "lossless_zstd_dict_benchmark",
                    "method": "zstd_dict",
                    "source_path": str(source),
                    "compressed_path": str(compressed),
                    "restored_path": str(restored),
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                    "archive_bytes": 50,
                    "original_bytes": 96,
                    "compression_rate": 1.92,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "zstd-dict-benchmark",
                        "--source",
                        str(source),
                        "--compressed",
                        str(compressed),
                        "--restored",
                        str(restored),
                        "--sample",
                        str(sample_a),
                        "--sample",
                        str(sample_b),
                        "--dict-size",
                        "4096",
                        "--sample-block-bytes",
                        "2048",
                    ]
                )

        mocked.assert_called_once_with(
            source_path=source,
            compressed_path=compressed,
            restored_path=restored,
            sample_paths=[sample_a, sample_b],
            dict_size=4096,
            sample_block_bytes=2048,
            max_training_samples=None,
        )
        self.assertEqual(result["command"], "lossless_zstd_dict_benchmark")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["compression_rate"], 1.92)

    def test_lossless_zstd_dict_dir_benchmark_subcommand_reports_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            compressed_root = root / "compressed"
            restored_root = root / "restored"
            sample_a = root / "sample_a.bin"
            sample_b = root / "sample_b.bin"
            with mock.patch.object(
                mod,
                "benchmark_zstd_dict_directory",
                return_value={
                    "command": "lossless_zstd_dict_directory_benchmark",
                    "method": "zstd_dict",
                    "source_root": str(source_root),
                    "compressed_root": str(compressed_root),
                    "restored_root": str(restored_root),
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                    "file_count": 2,
                    "archive_bytes": 12,
                    "original_bytes": 7,
                    "compression_rate": 7 / 12,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "zstd-dict-dir-benchmark",
                        "--source-root",
                        str(source_root),
                        "--compressed-root",
                        str(compressed_root),
                        "--restored-root",
                        str(restored_root),
                        "--sample",
                        str(sample_a),
                        "--sample",
                        str(sample_b),
                        "--dict-size",
                        "4096",
                        "--sample-block-bytes",
                        "2048",
                    ]
                )

        mocked.assert_called_once_with(
            source_root=source_root,
            compressed_root=compressed_root,
            restored_root=restored_root,
            sample_paths=[sample_a, sample_b],
            dict_size=4096,
            sample_block_bytes=2048,
            max_training_samples=None,
        )
        self.assertEqual(result["command"], "lossless_zstd_dict_directory_benchmark")
        self.assertEqual(result["file_count"], 2)
        self.assertAlmostEqual(result["compression_rate"], 7 / 12)

    def test_lossless_zstd_dict_chunk_benchmark_subcommand_reports_ratio(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "stream.bin"
            compressed_root = root / "compressed"
            restored_root = root / "restored"
            sample_a = root / "sample_a.bin"
            sample_b = root / "sample_b.bin"
            with mock.patch.object(
                mod,
                "benchmark_zstd_dict_chunked_file",
                return_value={
                    "command": "lossless_zstd_dict_chunked_benchmark",
                    "method": "zstd_dict",
                    "source_path": str(source),
                    "compressed_root": str(compressed_root),
                    "restored_root": str(restored_root),
                    "dictionary_bytes": 123,
                    "sample_count": 2,
                    "file_count": 3,
                    "block_bytes": 4096,
                    "archive_bytes": 12,
                    "original_bytes": 10,
                    "compression_rate": 10 / 12,
                    "exact_match": True,
                },
            ) as mocked:
                result = mod.main(
                    [
                        "lossless",
                        "zstd-dict-chunk-benchmark",
                        "--source",
                        str(source),
                        "--compressed-root",
                        str(compressed_root),
                        "--restored-root",
                        str(restored_root),
                        "--block-bytes",
                        "4096",
                        "--sample",
                        str(sample_a),
                        "--sample",
                        str(sample_b),
                        "--dict-size",
                        "8192",
                        "--sample-block-bytes",
                        "2048",
                    ]
                )

        mocked.assert_called_once_with(
            source_path=source,
            compressed_root=compressed_root,
            restored_root=restored_root,
            block_bytes=4096,
            sample_paths=[sample_a, sample_b],
            dict_size=8192,
            sample_block_bytes=2048,
            max_training_samples=None,
        )
        self.assertEqual(result["command"], "lossless_zstd_dict_chunked_benchmark")
        self.assertEqual(result["file_count"], 3)
        self.assertTrue(result["exact_match"])
        self.assertAlmostEqual(result["compression_rate"], 10 / 12)

    def test_lossless_prev_symbol_benchmark_subcommand_rejects_odd_byte_stream(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "broken.bin"
            token_path.write_bytes(b"\x01\x02\x03")

            with self.assertRaisesRegex(ValueError, "even number of bytes"):
                mod.main(
                    [
                        "lossless",
                        "prev-symbol-benchmark",
                        "--tokens",
                        str(token_path),
                    ]
                )

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
                        "archive_bytes": archive_path.stat().st_size,
                        "original_bytes": 4,
                        "compression_rate": 4 / archive_path.stat().st_size,
                        "method": "lzma",
                        "record_count": 4,
                        "checked_items": 4,
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
        self.assertIn("1.3333", latest)
        self.assertIn("\"compression_rate\": 1.3333333333333333", ledger)


if __name__ == "__main__":
    unittest.main()
