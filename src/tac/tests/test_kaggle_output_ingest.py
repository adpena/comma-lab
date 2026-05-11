from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
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

    def test_download_policy_skips_rebuildable_raw_and_cache_noise(self) -> None:
        keep, reason = mod.should_download_output("run/eval/contest_auth_eval.json")
        self.assertTrue(keep)
        self.assertEqual(reason, "matched")

        keep, reason = mod.should_download_output("run/eval/inflated/0.raw")
        self.assertFalse(keep)
        self.assertEqual(reason, "matched_exclude_patterns")

        keep, reason = mod.should_download_output("src/tac/__pycache__/module.cpython-312.pyc")
        self.assertFalse(keep)
        self.assertEqual(reason, "matched_exclude_patterns")

    def test_download_policy_supports_narrow_include_patterns(self) -> None:
        keep, reason = mod.should_download_output(
            "workspace/src/tac/preflight.py",
            include_patterns=(r"^pr106_yshift_score_table/",),
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "not_matched_by_include_patterns")

        keep, reason = mod.should_download_output(
            "pr106_yshift_score_table/yshift_run/eval/contest_auth_eval.json",
            include_patterns=(r"^pr106_yshift_score_table/",),
        )
        self.assertTrue(keep)

    def test_download_output_items_writes_hash_manifest_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items = [
                SimpleNamespace(file_name="run/eval/contest_auth_eval.json", url="mem://eval"),
                SimpleNamespace(file_name="run/eval/inflated/0.raw", url="mem://raw"),
            ]

            downloaded, skipped, log_downloaded, files_seen, files_matched = mod._download_output_items(
                kernel_slug="demo",
                page_items=items,
                log_text="hello\n",
                download_dir=root,
                fetch_bytes=lambda url: {"mem://eval": b'{"score_claim": false}'}[url],
                include_patterns=None,
                exclude_patterns=mod.DEFAULT_EXCLUDE_OUTPUT_PATTERNS,
            )

            self.assertEqual(files_seen, 2)
            self.assertEqual(files_matched, 1)
            self.assertTrue(log_downloaded)
            self.assertEqual([row.file_name for row in downloaded], ["run/eval/contest_auth_eval.json", "demo.log"])
            self.assertEqual(skipped[0].file_name, "run/eval/inflated/0.raw")
            self.assertEqual(downloaded[0].bytes, len(b'{"score_claim": false}'))
            self.assertEqual(len(downloaded[0].sha256), 64)
            self.assertTrue((root / "run/eval/contest_auth_eval.json").exists())
            self.assertTrue((root / "demo.log").exists())

    def test_skipped_by_reason_is_compact_and_counted(self) -> None:
        counts = mod._skipped_by_reason(
            [
                mod.SkippedOutput(file_name="a.py", reason="not_matched_by_include_patterns"),
                mod.SkippedOutput(file_name="b.py", reason="not_matched_by_include_patterns"),
                mod.SkippedOutput(file_name="0.raw", reason="matched_exclude_patterns"),
            ]
        )

        self.assertEqual(
            counts,
            {
                "matched_exclude_patterns": 1,
                "not_matched_by_include_patterns": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
