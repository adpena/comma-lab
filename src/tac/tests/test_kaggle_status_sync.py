# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy" / "kaggle" / "kaggle_status_sync.py"


def load_module():
    spec = importlib.util.spec_from_file_location("kaggle_status_sync", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KaggleStatusSyncTests(unittest.TestCase):
    def test_parse_kaggle_status_text_recognizes_running_kernelworker_status(self) -> None:
        mod = load_module()
        payload = mod.parse_kaggle_status_text("KernelWorkerStatus.RUNNING")

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["phase"], "kernel_running")
        self.assertEqual(payload["kernel_status"], "KernelWorkerStatus.RUNNING")

    def test_parse_kaggle_status_text_marks_quota_blocked_messages(self) -> None:
        mod = load_module()
        payload = mod.parse_kaggle_status_text(
            "Push attempt blocked by Kaggle free-tier quota: maximum batch GPU session count of 2 reached."
        )

        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["phase"], "quota_blocked")
        self.assertIn("quota", payload["notes"])

    def test_parse_kaggle_status_text_marks_cancel_acknowledged(self) -> None:
        mod = load_module()
        payload = mod.parse_kaggle_status_text(
            'adpena/comma-lab-demo has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
        )

        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["phase"], "kernel_cancel_acknowledged")

    def test_sync_status_from_manifest_writes_status_payload(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "remote_jobs" / "kaggle-demo.json"
            status_path = tmp / "status" / "kaggle-demo.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1-20260410T000419Z",
                        "host": "kaggle",
                        "platform": "kaggle",
                        "status": "draft",
                        "phase": "draft",
                        "manifest_path": str(manifest_path),
                        "status_path": str(status_path),
                        "kernel_ref": "adpena/comma-lab-demo",
                        "notes": "demo manifest",
                        "written_at": "2026-04-10T00:04:19Z",
                    },
                    indent=2,
                )
            )

            payload = mod.sync_kaggle_status(
                manifest_path,
                status_text="KernelWorkerStatus.RUNNING",
                output_path=status_path,
            )

            written = json.loads(status_path.read_text())

        self.assertEqual(payload["slug"], "kaggle-demo")
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["phase"], "kernel_running")
        self.assertEqual(payload["kernel_ref"], "adpena/comma-lab-demo")
        self.assertEqual(written["status_path"], str(status_path))
        self.assertEqual(written["status"], "running")
        self.assertEqual(written["phase"], "kernel_running")
        self.assertIn("written_at", written)

    def test_sync_status_queries_kaggle_command_when_text_missing(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "remote_jobs" / "kaggle-demo.json"
            status_path = tmp / "status" / "kaggle-demo.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1-20260410T000419Z",
                        "host": "kaggle",
                        "platform": "kaggle",
                        "status": "draft",
                        "manifest_path": str(manifest_path),
                        "status_path": str(status_path),
                        "kernel_ref": "adpena/comma-lab-demo",
                    },
                    indent=2,
                )
            )

            calls: list[list[str]] = []

            def fake_run(command: list[str]) -> object:
                calls.append(command)

                class Result:
                    returncode = 0
                    stdout = "KernelWorkerStatus.COMPLETE\n"
                    stderr = ""

                return Result()

            original_which = mod.shutil.which
            mod.shutil.which = lambda name: "/usr/bin/kaggle" if name == "kaggle" else None
            try:
                payload = mod.sync_kaggle_status(
                    manifest_path,
                    output_path=status_path,
                    command_runner=fake_run,
                )
            finally:
                mod.shutil.which = original_which

        self.assertEqual(calls, [["kaggle", "kernels", "status", "adpena/comma-lab-demo"]])
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["phase"], "kernel_complete")

    def test_sync_status_falls_back_to_uv_wrapped_kaggle_cli(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "remote_jobs" / "kaggle-demo.json"
            status_path = tmp / "status" / "kaggle-demo.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1-20260410T000419Z",
                        "host": "kaggle",
                        "platform": "kaggle",
                        "status": "draft",
                        "manifest_path": str(manifest_path),
                        "status_path": str(status_path),
                        "kernel_ref": "adpena/comma-lab-demo",
                    },
                    indent=2,
                )
            )

            calls: list[list[str]] = []

            def fake_run(command: list[str]) -> object:
                calls.append(command)

                class Result:
                    returncode = 0
                    stdout = "KernelWorkerStatus.RUNNING\n"
                    stderr = ""

                return Result()

            original_which = mod.shutil.which
            mod.shutil.which = lambda name: None
            try:
                payload = mod.sync_kaggle_status(
                    manifest_path,
                    output_path=status_path,
                    command_runner=fake_run,
                )
            finally:
                mod.shutil.which = original_which

        self.assertEqual(
            calls,
            [["uv", "run", "--with", "kaggle", "kaggle", "kernels", "status", "adpena/comma-lab-demo"]],
        )
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["phase"], "kernel_running")

    def test_sync_status_treats_not_found_kernel_as_not_pushed(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "remote_jobs" / "kaggle-demo.json"
            status_path = tmp / "status" / "kaggle-demo.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "slug": "kaggle-demo",
                        "run_id": "kaggle-demo-v1-20260410T000419Z",
                        "host": "kaggle",
                        "platform": "kaggle",
                        "status": "draft",
                        "manifest_path": str(manifest_path),
                        "status_path": str(status_path),
                        "kernel_ref": "adpena/comma-lab-demo",
                    },
                    indent=2,
                )
            )

            def fake_run(_command: list[str]) -> object:
                class Result:
                    returncode = 1
                    stdout = ""
                    stderr = "404 Client Error: Not Found for url: https://api.kaggle.com/..."

                return Result()

            payload = mod.sync_kaggle_status(
                manifest_path,
                output_path=status_path,
                command_runner=fake_run,
            )

        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["phase"], "kernel_not_pushed")


if __name__ == "__main__":
    unittest.main()
