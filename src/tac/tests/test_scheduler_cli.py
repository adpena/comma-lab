# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.comma_lab import cli
from src.comma_lab.scheduler.registry import load_platform_registry
from src.comma_lab.scheduler.repository import collect_run_records
from src.comma_lab.scheduler.reporting import build_budget_report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def registry_payload() -> dict[str, object]:
    return {
        "version": 1,
        "platforms": [
            {
                "name": "local",
                "kind": "local",
                "result_devices": ["cpu", "mps"],
                "manifest_globs": [
                    ".omx/logs/remote_jobs/local-*.json",
                    ".omx/logs/remote_jobs/v2-*.json",
                ],
                "budget": {
                    "max_runs": 2,
                    "max_active_runs": 2,
                    "max_archive_bytes": 2000000,
                },
            },
            {
                "name": "remote-gpu",
                "kind": "remote",
                "manifest_globs": [
                    ".omx/logs/remote_jobs/remote-gpu-*.json",
                    "remote-runs/remote-gpu/*/manifest.json",
                ],
                "status_globs": [
                    ".omx/status/remote-gpu-*.json",
                    "remote-runs/remote-gpu/*/status.json",
                ],
                "ledger_paths": ["remote-runs/remote-gpu/_ledger.jsonl"],
                "budget": {
                    "max_active_runs": 1,
                    "max_failed_runs": 1,
                },
            },
            {
                "name": "kaggle",
                "kind": "remote",
                "result_devices": ["kaggle"],
                "manifest_globs": [
                    ".omx/logs/remote_jobs/kaggle-*.json",
                    "remote-runs/kaggle/*/manifest.json",
                ],
                "status_globs": [
                    ".omx/status/kaggle-*.json",
                    "remote-runs/kaggle/*/status.json",
                ],
                "ledger_paths": ["remote-runs/kaggle/_ledger.jsonl"],
                "budget": {"max_runs": 2, "max_active_runs": 1},
            },
            {
                "name": "modal",
                "kind": "remote",
                "result_devices": ["modal"],
                "manifest_globs": [
                    ".omx/logs/remote_jobs/modal-*.json",
                    "remote-runs/modal/*/manifest.json",
                ],
                "status_globs": [
                    ".omx/status/modal-*.json",
                    "remote-runs/modal/*/status.json",
                ],
                "ledger_paths": ["remote-runs/modal/_ledger.jsonl"],
                "budget": {"max_runs": 2, "max_active_runs": 1},
            },
            {
                "name": "coiled",
                "kind": "remote",
                "result_devices": ["coiled"],
                "manifest_globs": [
                    ".omx/logs/remote_jobs/coiled-*.json",
                    "remote-runs/coiled/*/manifest.json",
                ],
                "status_globs": [
                    ".omx/status/coiled-*.json",
                    "remote-runs/coiled/*/status.json",
                ],
                "ledger_paths": ["remote-runs/coiled/_ledger.jsonl"],
                "budget": {"max_runs": 2, "max_active_runs": 1},
            },
        ],
    }


class SchedulerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = Path(self.tmpdir.name)
        self.registry_path = self.repo_root / "configs" / "platforms.json"

        write_json(self.registry_path, registry_payload())

        write_jsonl(
            self.repo_root / "reports" / "results.jsonl",
            [
                {
                    "run_id": "robust_current-local-promoted-cpu-2026-04-09",
                    "track": "robust_current",
                    "device": "cpu",
                    "packaging_view": "current_workflow",
                    "current_workflow_score": 1.73,
                    "archive_bytes": 864167,
                    "ts_utc": "2026-04-09T15:07:00+00:00",
                    "artifacts": {"summary_json": "reports/raw/demo/summary.json"},
                },
                {
                    "run_id": "robust_current-kaggle-promoted-2026-04-09",
                    "track": "robust_current",
                    "device": "kaggle",
                    "packaging_view": "current_workflow",
                    "current_workflow_score": 1.71,
                    "archive_bytes": 864100,
                    "ts_utc": "2026-04-09T15:08:00+00:00",
                    "artifacts": {"summary_json": "reports/raw/demo/kaggle-summary.json"},
                },
                {
                    "run_id": "robust_current-modal-promoted-2026-04-09",
                    "track": "robust_current",
                    "device": "modal",
                    "packaging_view": "current_workflow",
                    "current_workflow_score": 1.70,
                    "archive_bytes": 864090,
                    "ts_utc": "2026-04-09T15:09:00+00:00",
                    "artifacts": {"summary_json": "reports/raw/demo/modal-summary.json"},
                },
                {
                    "run_id": "robust_current-coiled-promoted-2026-04-09",
                    "track": "robust_current",
                    "device": "coiled",
                    "packaging_view": "current_workflow",
                    "current_workflow_score": 1.69,
                    "archive_bytes": 864080,
                    "ts_utc": "2026-04-09T15:10:00+00:00",
                    "artifacts": {"summary_json": "reports/raw/demo/coiled-summary.json"},
                },
            ],
        )
        write_json(
            self.repo_root / "remote-runs" / "remote-gpu" / "lane-alpha" / "manifest.json",
            {
                "slug": "lane-alpha",
                "run_id": "20260409T120000Z",
                "host": "remote-gpu",
                "started_at_utc": "20260409T120000Z",
                "status": "starting",
            },
        )
        write_json(
            self.repo_root / "remote-runs" / "remote-gpu" / "lane-alpha" / "status.json",
            {
                "slug": "lane-alpha",
                "run_id": "20260409T120000Z",
                "status": "running",
                "pid": 1234,
            },
        )
        write_json(
            self.repo_root / "remote-runs" / "kaggle" / "lane-beta" / "status.json",
            {
                "slug": "kaggle-pairaware-h64",
                "run_id": "kgl-20260409T121000Z",
                "status": "running",
                "phase": "training",
            },
        )
        write_json(
            self.repo_root / "remote-runs" / "coiled" / "lane-gamma" / "status.json",
            {
                "slug": "coiled-quant-audit",
                "run_id": "coiled-20260409T122000Z",
                "status": "running",
                "phase": "quantization_audit",
            },
        )
        write_json(
            self.repo_root / ".omx" / "logs" / "remote_jobs" / "local-legacy-rerun.json",
            {
                "slug": "local-legacy-rerun",
                "host": "local-mps",
                "status": "running_managed_session",
                "session_id": 64668,
                "written_at": "2026-04-08T21:50:00-05:00",
            },
        )

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = cli.main(list(argv))
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_collect_run_records_reads_results_and_remote_status(self) -> None:
        registry = load_platform_registry(self.registry_path)

        records = collect_run_records(self.repo_root, registry)
        budget = build_budget_report(registry, records)

        self.assertEqual(len(records), 8)
        self.assertEqual(
            {record.platform for record in records},
            {"remote-gpu", "coiled", "kaggle", "local", "modal"},
        )
        self.assertEqual(budget.platforms["local"].usage.total_runs, 2)
        self.assertEqual(budget.platforms["local"].usage.archive_bytes, 864167)
        self.assertEqual(budget.platforms["local"].usage.active_runs, 1)
        self.assertEqual(budget.platforms["remote-gpu"].usage.active_runs, 1)
        self.assertEqual(budget.platforms["kaggle"].usage.active_runs, 1)
        self.assertEqual(budget.platforms["coiled"].usage.active_runs, 1)
        self.assertEqual(budget.platforms["modal"].usage.total_runs, 1)
        legacy = next(record for record in records if record.run_id == "local-legacy-rerun")
        self.assertEqual(legacy.platform, "local")
        self.assertEqual(legacy.status, "running_managed_session")

    def test_sched_status_json_reports_latest_results_and_active_runs(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "status",
            "--repo-root",
            str(self.repo_root),
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["result_count"], 4)
        self.assertEqual(payload["tracks"][0]["track"], "robust_current")
        self.assertEqual(payload["tracks"][0]["latest_result"]["score"], 1.69)
        self.assertEqual(payload["tracks"][0]["latest_result"]["platform"], "coiled")
        self.assertEqual(
            {item["platform"] for item in payload["active_runs"]},
            {"remote-gpu", "coiled", "kaggle", "local"},
        )

    def test_sched_results_json_honors_limit(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "results",
            "--repo-root",
            str(self.repo_root),
            "--limit",
            "2",
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][0]["run_id"], "robust_current-coiled-promoted-2026-04-09")
        self.assertEqual(payload["results"][0]["platform"], "coiled")
        self.assertEqual(payload["results"][1]["run_id"], "robust_current-modal-promoted-2026-04-09")
        self.assertEqual(payload["results"][1]["platform"], "modal")

    def test_sched_budget_json_reports_usage_against_registry(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "budget",
            "--repo-root",
            str(self.repo_root),
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        local = next(item for item in payload["platforms"] if item["name"] == "local")
        remote_gpu = next(item for item in payload["platforms"] if item["name"] == "remote-gpu")
        kaggle = next(item for item in payload["platforms"] if item["name"] == "kaggle")
        modal = next(item for item in payload["platforms"] if item["name"] == "modal")
        coiled = next(item for item in payload["platforms"] if item["name"] == "coiled")
        self.assertEqual(local["usage"]["total_runs"], 2)
        self.assertEqual(local["usage"]["archive_bytes"], 864167)
        self.assertFalse(local["over_budget"])
        self.assertEqual(local["usage"]["active_runs"], 1)
        self.assertEqual(remote_gpu["usage"]["active_runs"], 1)
        self.assertEqual(kaggle["usage"]["active_runs"], 1)
        self.assertEqual(modal["usage"]["total_runs"], 1)
        self.assertEqual(coiled["usage"]["active_runs"], 1)

    def test_sched_budget_requires_registry(self) -> None:
        self.registry_path.unlink()
        rc, stdout, stderr = self.run_cli(
            "sched",
            "budget",
            "--repo-root",
            str(self.repo_root),
        )

        self.assertEqual(rc, 1)
        self.assertEqual(stdout, "")
        self.assertIn("--registry", stderr)


if __name__ == "__main__":
    unittest.main()
