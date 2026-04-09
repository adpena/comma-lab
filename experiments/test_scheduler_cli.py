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


class SchedulerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = Path(self.tmpdir.name)
        self.registry_path = self.repo_root / "configs" / "platforms.json"

        write_json(
            self.registry_path,
            {
                "version": 1,
                "platforms": [
                    {
                        "name": "cpu",
                        "kind": "local",
                        "result_devices": ["cpu"],
                        "budget": {
                            "max_runs": 2,
                            "max_archive_bytes": 2000000,
                        },
                    },
                    {
                        "name": "bat00",
                        "kind": "remote",
                        "manifest_globs": ["remote-runs/*/manifest.json"],
                        "status_globs": ["remote-runs/*/status.json"],
                        "budget": {
                            "max_active_runs": 1,
                            "max_failed_runs": 1,
                        },
                    },
                ],
            },
        )

        write_jsonl(
            self.repo_root / "reports" / "results.jsonl",
            [
                {
                    "run_id": "robust_current-long1000-h64-promoted-cpu-2026-04-09",
                    "track": "robust_current",
                    "device": "cpu",
                    "packaging_view": "current_workflow",
                    "current_workflow_score": 1.73,
                    "archive_bytes": 864167,
                    "ts_utc": "2026-04-09T15:07:00+00:00",
                    "artifacts": {"summary_json": "reports/raw/demo/summary.json"},
                }
            ],
        )
        write_json(
            self.repo_root / "remote-runs" / "lane-alpha" / "manifest.json",
            {
                "slug": "lane-alpha",
                "run_id": "20260409T120000Z",
                "host": "bat00",
                "started_at_utc": "20260409T120000Z",
                "status": "starting",
            },
        )
        write_json(
            self.repo_root / "remote-runs" / "lane-alpha" / "status.json",
            {
                "slug": "lane-alpha",
                "run_id": "20260409T120000Z",
                "status": "running",
                "pid": 1234,
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

        self.assertEqual(len(records), 2)
        self.assertEqual({record.platform for record in records}, {"bat00", "cpu"})
        self.assertEqual(budget.platforms["cpu"].usage.total_runs, 1)
        self.assertEqual(budget.platforms["cpu"].usage.archive_bytes, 864167)
        self.assertEqual(budget.platforms["bat00"].usage.active_runs, 1)

    def test_sched_status_json_reports_latest_results_and_active_runs(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "status",
            "--repo-root",
            str(self.repo_root),
            "--registry",
            str(self.registry_path),
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["tracks"][0]["track"], "robust_current")
        self.assertEqual(payload["tracks"][0]["latest_result"]["score"], 1.73)
        self.assertEqual(payload["active_runs"][0]["platform"], "bat00")

    def test_sched_results_json_honors_limit(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "results",
            "--repo-root",
            str(self.repo_root),
            "--limit",
            "1",
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["run_id"], "robust_current-long1000-h64-promoted-cpu-2026-04-09")
        self.assertEqual(payload["results"][0]["archive_bytes"], 864167)

    def test_sched_budget_json_reports_usage_against_registry(self) -> None:
        rc, stdout, stderr = self.run_cli(
            "sched",
            "budget",
            "--repo-root",
            str(self.repo_root),
            "--registry",
            str(self.registry_path),
            "--json",
        )

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        cpu = next(item for item in payload["platforms"] if item["name"] == "cpu")
        bat00 = next(item for item in payload["platforms"] if item["name"] == "bat00")
        self.assertEqual(cpu["usage"]["total_runs"], 1)
        self.assertEqual(cpu["usage"]["archive_bytes"], 864167)
        self.assertFalse(cpu["over_budget"])
        self.assertEqual(bat00["usage"]["active_runs"], 1)

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
