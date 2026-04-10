from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import kaggle_output_ingest as mod


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


if __name__ == "__main__":
    unittest.main()
