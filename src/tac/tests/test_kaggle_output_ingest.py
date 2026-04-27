from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tac.deploy.kaggle.kaggle_output_ingest as mod


class KaggleOutputIngestTests(unittest.TestCase):
    def test_extract_training_signals_from_json_stream_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "kernel.log"
            rows = [
                {"stream_name": "stdout", "data": "[cloud-segnet-h32] best checkpoint -> epoch 40 score=1.8123 int8=16455 bytes\n"},
                {"stream_name": "stdout", "data": "Saved fp32: /kaggle/working/postfilter_x_fp32.pt\n"},
                {"stream_name": "stdout", "data": "Saved int8: /kaggle/working/postfilter_x_int8.pt (16455 bytes)\n"},
                {"stream_name": "stdout", "data": "Saved final meta: /kaggle/working/postfilter_x_final_meta.json\n"},
            ]
            log_path.write_text(json.dumps(rows))

            signals = mod.extract_training_signals(log_path)

        self.assertEqual(signals["best_checkpoint"]["epoch"], 40)
        self.assertEqual(signals["best_checkpoint"]["score"], 1.8123)
        self.assertEqual(signals["saved"]["fp32"], "/kaggle/working/postfilter_x_fp32.pt")
        self.assertEqual(signals["saved"]["int8_bytes"], 16455)

    def test_extract_proxy_result_from_plain_text_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proxy.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[proxy-faithful] Results:",
                        "  PoseNet distortion: 0.05168364",
                        "  SegNet distortion:  0.00543626",
                        "  Compression rate:   0.02301653",
                        "  Final score:        1.8400",
                    ]
                )
            )

            signals = mod.extract_training_signals(log_path)

        self.assertEqual(signals["proxy_result"]["current_workflow_score"], 1.84)
        self.assertEqual(signals["proxy_result"]["pose_distortion"], 0.05168364)

    def test_extract_failure_signature_from_json_stream_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "kernel.log"
            rows = [
                {"stream_name": "stderr", "data": "Traceback (most recent call last):\n"},
                {"stream_name": "stderr", "data": "  File \"/kaggle/src/script.py\", line 1, in <module>\n"},
                {"stream_name": "stderr", "data": "ImportError: tac is not importable and no bundled wheel was found\n"},
            ]
            log_path.write_text(json.dumps(rows))

            signals = mod.extract_training_signals(log_path)

        self.assertEqual(signals["failure"]["error_type"], "ImportError")
        self.assertIn("no bundled wheel", signals["failure"]["message"])

    def test_ingest_existing_log_copies_into_run_evidence_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "kaggle-run.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1",
                        "kernel_ref": "adpena/comma-lab-demo",
                    }
                )
            )
            downloaded = root / "downloaded"
            downloaded.mkdir()
            log_path = downloaded / "demo.log"
            log_path.write_text("hello\n")
            out_root = root / "reports"

            report = mod.ingest_downloaded_outputs(
                manifest_path=manifest_path,
                download_dir=downloaded,
                output_root=out_root,
            )

            evidence_dir = out_root / "kaggle-demo-v1"
            self.assertEqual(report["run_id"], "kaggle-demo-v1")
            self.assertTrue((evidence_dir / "demo.log").exists())

    def test_ingest_summary_surfaces_latest_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "kaggle-run.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1",
                        "kernel_ref": "adpena/comma-lab-demo",
                    }
                )
            )
            downloaded = root / "downloaded"
            downloaded.mkdir()
            log_path = downloaded / "demo.log"
            rows = [
                {"stream_name": "stderr", "data": "Traceback (most recent call last):\n"},
                {"stream_name": "stderr", "data": "ImportError: tac is not importable\n"},
            ]
            log_path.write_text(json.dumps(rows))
            out_root = root / "reports"

            report = mod.ingest_downloaded_outputs(
                manifest_path=manifest_path,
                download_dir=downloaded,
                output_root=out_root,
            )

        self.assertEqual(report["latest_failure"]["error_type"], "ImportError")
        self.assertIn("tac is not importable", report["latest_failure"]["message"])

    def test_ingest_summary_surfaces_checkpoint_meta_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "kaggle-run.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1",
                        "kernel_ref": "adpena/comma-lab-demo",
                    }
                )
            )
            downloaded = root / "downloaded"
            downloaded.mkdir()
            meta_path = downloaded / "postfilter_demo_best_meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "epoch": 30,
                        "scorer": 3.96,
                        "int8_size": 45785,
                        "meta": {"variant": "dilated", "hidden": 64},
                    }
                )
            )
            out_root = root / "reports"

            report = mod.ingest_downloaded_outputs(
                manifest_path=manifest_path,
                download_dir=downloaded,
                output_root=out_root,
            )

        self.assertEqual(report["latest_checkpoint"]["epoch"], 30)
        self.assertEqual(report["latest_checkpoint"]["scorer"], 3.96)
        self.assertEqual(report["latest_checkpoint"]["variant"], "dilated")

    def test_ingest_summary_finds_checkpoint_meta_in_nested_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "kaggle-run.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1",
                        "kernel_ref": "adpena/comma-lab-demo",
                    }
                )
            )
            downloaded = root / "downloaded"
            nested = downloaded / "postfilter_weights"
            nested.mkdir(parents=True)
            meta_path = nested / "postfilter_demo_best_meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "epoch": 42,
                        "scorer": 3.5,
                        "int8_size": 123,
                        "meta": {"variant": "dilated", "hidden": 64},
                    }
                )
            )
            out_root = root / "reports"

            report = mod.ingest_downloaded_outputs(
                manifest_path=manifest_path,
                download_dir=downloaded,
                output_root=out_root,
            )

        self.assertEqual(report["latest_checkpoint"]["epoch"], 42)


if __name__ == "__main__":
    unittest.main()
