from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.comma_lab import cli

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class StateCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = Path(self.tmpdir.name)
        write_json(
            self.repo_root / ".omx" / "state" / "promoted_result.json",
            {
                "run_id": "robust_current-dilated-h64-modal-cpu-2026-04-10",
                "track": "robust_current",
                "score": 1.33,
                "pose_distortion": 0.00218374,
                "seg_distortion": 0.00609921,
                "rate": 0.02301653,
                "archive_bytes": 864167,
                "authoritative_report_path": "reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt",
                "authoritative_report_copy_path": "reports/raw/robust_current-current_workflow-cpu-report.txt",
                "summary_path": "reports/raw/robust_current-current_workflow-cpu-summary.json",
                "artifact_path": "submissions/robust_current/postfilter_int8.pt",
                "variant": "dilated_h64",
                "platform": "modal_a10g",
                "epoch": 905,
                "promoted_at": "2026-04-10T21:30:00-05:00",
            },
        )
        report_path = (
            self.repo_root
            / "reports"
            / "raw"
            / "2026-04-10-dilated-h64-authoritative"
            / "robust_current-dilated-h64-authoritative-cpu-report.txt"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 1.33\n")
        (self.repo_root / "submissions" / "robust_current").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "submissions" / "robust_current" / "postfilter_int8.pt").write_bytes(b"x")
        write_json(
            self.repo_root / "reports" / "raw" / "robust_current-current_workflow-cpu-summary.json",
            {"current_workflow_score": 1.51},
        )
        (self.repo_root / "reports" / "latest.md").parent.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "reports" / "latest.md").write_text("1.51\n")
        write_json(
            self.repo_root / ".omx" / "logs" / "remote_jobs" / "local-modal-dilated-h64-authoritative-eval.json",
            {"slug": "local-modal-dilated-h64-authoritative-eval", "status": "running_managed_session", "session_id": 999999},
        )

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = cli.main(list(argv))
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_state_doctor_json_reports_drift(self) -> None:
        rc, stdout, stderr = self.run_cli("state", "doctor", "--repo-root", str(self.repo_root), "--json")

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        codes = {finding["code"] for finding in payload["findings"]}
        self.assertIn("canonical_summary_mismatch", codes)
        self.assertIn("latest_report_stale", codes)

    def test_state_sync_repairs_summary(self) -> None:
        rc, stdout, stderr = self.run_cli("state", "sync", "--repo-root", str(self.repo_root), "--json")

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertGreaterEqual(payload["changed_count"], 1)
        summary = json.loads(
            (self.repo_root / "reports" / "raw" / "robust_current-current_workflow-cpu-summary.json").read_text()
        )
        self.assertEqual(summary["current_workflow_score"], 1.33)

    def test_state_promote_validates_and_syncs(self) -> None:
        rc, stdout, stderr = self.run_cli("state", "promote", "--repo-root", str(self.repo_root), "--json")

        self.assertEqual(rc, 0, stderr)
        payload = json.loads(stdout)
        self.assertGreaterEqual(payload["changed_count"], 1)
        latest = (self.repo_root / "reports" / "latest.md").read_text()
        self.assertIn("1.33", latest)

    def test_module_invocation_status_still_works_without_pythonpath_hacks(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.comma_lab.cli", "status"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Repo root:", result.stdout)


if __name__ == "__main__":
    unittest.main()
