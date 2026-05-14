# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comma_lab.task_codec import (
    ArchitectureRegistry,
    EvaluationRecord,
    FinalMetadata,
    ProxyEvaluationRecord,
    QuantizationMetadata,
    ResumeState,
    ScorerRegistry,
    register_default_architectures,
)


class TaskCodecCoreTests(unittest.TestCase):
    def test_scorer_registry_wraps_callable_and_metadata(self) -> None:
        registry = ScorerRegistry()

        def score_pair(left: int, right: int) -> int:
            return left + right

        scorer = registry.register(
            "toy_sum",
            score_pair,
            family="unit",
            description="simple adder",
            outputs=("score",),
        )

        self.assertEqual(scorer.name, "toy_sum")
        self.assertEqual(scorer.family, "unit")
        self.assertEqual(scorer.outputs, ("score",))
        self.assertEqual(scorer(2, 3), 5)
        self.assertEqual(registry.get("toy_sum")(7, 4), 11)
        self.assertEqual(registry.names(), ("toy_sum",))

    def test_architecture_registry_resolves_existing_variant_metadata(self) -> None:
        registry = ArchitectureRegistry()
        register_default_architectures(registry)

        config = registry.resolve_config(
            {
                "variant": "saliency_weighted",
                "hidden": "24",
                "kernel": "3",
                "alpha": "20.0",
            }
        )

        self.assertEqual(config.variant, "saliency_weighted")
        self.assertEqual(config.entrypoint, "experiments/train_postfilter_saliency.py")
        self.assertEqual(config.parameters["hidden"], 24)
        self.assertEqual(config.parameters["kernel"], 3)
        self.assertEqual(config.parameters["alpha"], 20.0)
        self.assertEqual(config.parameters["variant"], "saliency_weighted")

    def test_quantization_metadata_loads_best_meta_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            meta_path = root / "postfilter_lane_best_meta.json"
            int8_path = root / "postfilter_lane_best_int8.pt"
            fp32_path = root / "postfilter_lane_best_fp32.pt"
            int8_path.write_bytes(b"int8")
            fp32_path.write_bytes(b"fp32")
            meta_path.write_text(
                json.dumps(
                    {
                        "epoch": 12,
                        "scorer": 3.456,
                        "fp32_path": str(fp32_path),
                        "int8_path": str(int8_path),
                        "int8_size": 12345,
                        "meta": {
                            "variant": "film_conditioned",
                            "hidden": 32,
                            "kernel": 3,
                            "alpha": 20.0,
                        },
                    }
                )
            )

            record = QuantizationMetadata.from_path(meta_path)

        self.assertEqual(record.best_meta_path, meta_path)
        self.assertEqual(record.int8_path, int8_path)
        self.assertEqual(record.fp32_path, fp32_path)
        self.assertEqual(record.variant, "film_conditioned")
        self.assertEqual(record.hidden, 32)
        self.assertEqual(record.kernel, 3)
        self.assertEqual(record.alpha, 20.0)
        self.assertEqual(record.int8_size, 12345)
        self.assertEqual(record.scorer, 3.456)

    def test_quantization_metadata_finds_adjacent_best_meta_for_int8_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            int8_path = root / "postfilter_lane_best_int8.pt"
            meta_path = root / "postfilter_lane_best_meta.json"
            int8_path.write_bytes(b"int8")
            meta_path.write_text(
                json.dumps(
                    {
                        "epoch": 9,
                        "score": 4.25,
                        "int8_path": str(int8_path),
                        "int8_size": 2048,
                        "meta": {"variant": "residual", "hidden": 16, "kernel": 3},
                    }
                )
            )

            record = QuantizationMetadata.from_path(int8_path)

        self.assertEqual(record.best_meta_path, meta_path)
        self.assertEqual(record.int8_path, int8_path)
        self.assertEqual(record.variant, "residual")
        self.assertEqual(record.scorer, 4.25)
        self.assertEqual(record.hidden, 16)

    def test_resume_state_from_best_meta_loads_checkpoint_without_final_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_meta_path = root / "postfilter_lane_best_meta.json"
            best_fp32_path = root / "postfilter_lane_best_fp32.pt"
            best_int8_path = root / "postfilter_lane_best_int8.pt"
            best_fp32_path.write_bytes(b"fp32")
            best_int8_path.write_bytes(b"int8")
            best_meta_path.write_text(
                json.dumps(
                    {
                        "epoch": 12,
                        "scorer": 3.25,
                        "fp32_path": str(best_fp32_path),
                        "int8_path": str(best_int8_path),
                        "int8_size": 4096,
                        "meta": {"variant": "residual", "hidden": 16, "kernel": 3},
                    }
                )
            )

            state = ResumeState.from_path(best_meta_path)

        self.assertIsNone(state.final)
        self.assertIsNotNone(state.best_checkpoint)
        self.assertEqual(state.best_meta_path, best_meta_path)
        self.assertEqual(state.best_checkpoint.int8_path, best_int8_path)
        self.assertEqual(state.preferred_int8_path, best_int8_path)

    def test_resume_state_from_final_meta_loads_nested_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_meta_path = root / "postfilter_lane_final_meta.json"
            final_fp32_path = root / "postfilter_lane_fp32.pt"
            final_int8_path = root / "postfilter_lane_int8.pt"
            best_meta_path = root / "postfilter_lane_best_meta.json"
            best_fp32_path = root / "postfilter_lane_best_fp32.pt"
            best_int8_path = root / "postfilter_lane_best_int8.pt"
            final_fp32_path.write_bytes(b"final-fp32")
            final_int8_path.write_bytes(b"final-int8")
            best_fp32_path.write_bytes(b"best-fp32")
            best_int8_path.write_bytes(b"best-int8")
            best_payload = {
                "epoch": 7,
                "scorer": 1.99,
                "fp32_path": str(best_fp32_path),
                "int8_path": str(best_int8_path),
                "int8_size": 1024,
                "meta": {"variant": "film_conditioned", "hidden": 32, "kernel": 3, "alpha": 20.0},
            }
            final_meta_path.write_text(
                json.dumps(
                    {
                        "tag": "lane",
                        "fp32_path": str(final_fp32_path),
                        "int8_path": str(final_int8_path),
                        "int8_size": 2048,
                        "baseline_loss": 1.8,
                        "final_loss": 1.6,
                        "final_pose": 0.04,
                        "final_seg": 0.005,
                        "meta": {"variant": "film_conditioned", "hidden": 32, "kernel": 3, "alpha": 20.0},
                        "best_eval": best_payload,
                        "best_meta_path": str(best_meta_path),
                        "final_meta_path": str(final_meta_path),
                    }
                )
            )

            state = ResumeState.from_path(final_meta_path)

        self.assertIsNotNone(state.final)
        self.assertIsNotNone(state.best_checkpoint)
        self.assertEqual(state.final.best_meta_path, best_meta_path)
        self.assertEqual(state.final.final_meta_path, final_meta_path)
        self.assertEqual(state.best_checkpoint.int8_path, best_int8_path)
        self.assertEqual(state.preferred_int8_path, best_int8_path)
        self.assertIsInstance(state.final, FinalMetadata)

    def test_evaluation_record_loads_current_workflow_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "track": "robust_current",
                        "device": "cpu",
                        "report_path": "/tmp/report.txt",
                        "copied_report_path": None,
                        "current_workflow_archive_bytes": 700000,
                        "pose_distortion": 0.0123,
                        "seg_distortion": 0.0045,
                        "original_uncompressed_bytes": 10000000,
                        "current_workflow_rate": 0.07,
                        "current_workflow_score": 1.89,
                        "rule_faithful_bundle_bytes": 710000,
                        "rule_faithful_bundle_paths": ["postfilter_int8.pt"],
                        "rule_faithful_rate": 0.071,
                        "rule_faithful_score": 1.91,
                        "rule_faithful_status": "estimated",
                    }
                )
            )

            record = EvaluationRecord.from_path(summary_path)

        self.assertEqual(record.track, "robust_current")
        self.assertEqual(record.device, "cpu")
        self.assertEqual(record.current_workflow_score, 1.89)
        self.assertEqual(record.current_workflow_archive_bytes, 700000)
        self.assertEqual(record.source_path, summary_path)

    def test_proxy_record_loads_trailing_json_object_from_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proxy_lane.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[proxy-faithful] weights: /tmp/postfilter_lane_best_int8.pt",
                        "[proxy-faithful] archive: /tmp/archive.zip",
                        "[proxy-faithful] device: cpu",
                        "[proxy-faithful] Results:",
                        '{"pose_distortion": 0.01, "seg_distortion": 0.002, "current_workflow_rate": 0.05, '
                        '"current_workflow_score": 1.5, "current_workflow_archive_bytes": 54321, '
                        '"weights": "/tmp/postfilter_lane_best_int8.pt", "archive": "/tmp/archive.zip", "device": "cpu"}',
                    ]
                )
            )

            record = ProxyEvaluationRecord.from_path(log_path)

        self.assertEqual(record.weights_path, Path("/tmp/postfilter_lane_best_int8.pt"))
        self.assertEqual(record.archive_path, Path("/tmp/archive.zip"))
        self.assertEqual(record.current_workflow_score, 1.5)
        self.assertEqual(record.current_workflow_archive_bytes, 54321)
        self.assertEqual(record.source_path, log_path)

    def test_proxy_record_falls_back_to_text_log_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proxy_lane.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[proxy-faithful] weights: /tmp/postfilter_lane_best_int8.pt",
                        "[proxy-faithful] archive: /tmp/archive.zip",
                        "[proxy-faithful] device: cpu",
                        "  PoseNet distortion: 0.01000000",
                        "  SegNet distortion:  0.00200000",
                        "  Compression rate:   0.05000000",
                        "  Final score:        1.5000",
                    ]
                )
            )

            record = ProxyEvaluationRecord.from_path(log_path)

        self.assertEqual(record.weights_path, Path("/tmp/postfilter_lane_best_int8.pt"))
        self.assertEqual(record.archive_path, Path("/tmp/archive.zip"))
        self.assertEqual(record.device, "cpu")
        self.assertEqual(record.current_workflow_score, 1.5)
        self.assertEqual(record.current_workflow_rate, 0.05)


if __name__ == "__main__":
    unittest.main()
