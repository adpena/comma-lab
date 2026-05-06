from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class StateSyncTests(unittest.TestCase):
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
        report = "\n".join(
            [
                "=== Evaluation config ===",
                "=== Evaluation results over 600 samples ===",
                "  Average PoseNet Distortion: 0.00218374",
                "  Average SegNet Distortion: 0.00609921",
                "  Submission file size: 864,167 bytes",
                "  Compression Rate: 0.02301653",
                "  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 1.33",
                "",
            ]
        )
        authoritative = (
            self.repo_root
            / "reports"
            / "raw"
            / "2026-04-10-dilated-h64-authoritative"
            / "robust_current-dilated-h64-authoritative-cpu-report.txt"
        )
        authoritative.parent.mkdir(parents=True, exist_ok=True)
        authoritative.write_text(report)

        write_json(
            self.repo_root / "reports" / "raw" / "robust_current-current_workflow-cpu-summary.json",
            {
                "track": "robust_current",
                "current_workflow_score": 1.51,
                "pose_distortion": 0.01229283,
                "seg_distortion": 0.00579903,
                "current_workflow_archive_bytes": 864167,
                "current_workflow_rate": 0.02301653,
            },
        )
        (self.repo_root / "reports" / "latest.md").parent.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "reports" / "latest.md").write_text("Track B floor is 1.51\n")
        (self.repo_root / "reports" / "results.jsonl").write_text("")
        (self.repo_root / "reports" / "timeline.jsonl").write_text("")
        current_focus = self.repo_root / ".omx" / "state" / "current_focus.md"
        current_focus.parent.mkdir(parents=True, exist_ok=True)
        current_focus.write_text("# Current Focus\n\nFloor: 1.51\n")
        next_experiments = self.repo_root / ".omx" / "state" / "next_experiments.md"
        next_experiments.parent.mkdir(parents=True, exist_ok=True)
        next_experiments.write_text("# Next\n\nPromoted floor: 1.51\n")
        findings = self.repo_root / ".omx" / "research" / "findings.md"
        findings.parent.mkdir(parents=True, exist_ok=True)
        findings.write_text("# Findings\n\nPromoted floor: 1.51\n")
        run_log = self.repo_root / ".ralph" / "run_log.md"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        run_log.write_text("# run log\n\nPromoted floor: 1.51\n")
        write_json(
            self.repo_root / ".omx" / "logs" / "remote_jobs" / "local-modal-dilated-h64-authoritative-eval.json",
            {
                "slug": "local-modal-dilated-h64-authoritative-eval",
                "host": "local-cpu",
                "status": "running_managed_session",
                "session_id": 999999,
                "submission_dir": "/tmp/robust_current_modal_dilated_eval",
            },
        )

    def test_doctor_detects_split_brain_and_stale_managed_session(self) -> None:
        from src.comma_lab import state_sync

        report = state_sync.doctor_repo(self.repo_root)

        codes = {finding.code for finding in report.findings}
        self.assertIn("canonical_summary_mismatch", codes)
        self.assertIn("latest_report_stale", codes)
        self.assertIn("stale_managed_session", codes)

    def test_doctor_reports_malformed_jsonl_without_crashing(self) -> None:
        from src.comma_lab import state_sync

        (self.repo_root / "reports" / "results.jsonl").write_text("{not json}\n")

        report = state_sync.doctor_repo(self.repo_root)

        malformed = [finding for finding in report.findings if finding.code == "malformed_jsonl"]
        self.assertEqual(len(malformed), 1)
        self.assertEqual(malformed[0].path, "reports/results.jsonl:1")
        self.assertIn("not valid JSON", malformed[0].message)

    def test_sync_repo_repairs_projections_and_marks_stale_managed_session(self) -> None:
        from src.comma_lab import state_sync

        result = state_sync.sync_repo(self.repo_root)

        self.assertTrue(result.changed_paths)
        summary = json.loads(
            (self.repo_root / "reports" / "raw" / "robust_current-current_workflow-cpu-summary.json").read_text()
        )
        self.assertEqual(summary["current_workflow_score"], 1.33)
        self.assertEqual(summary["pose_distortion"], 0.00218374)
        latest = (self.repo_root / "reports" / "latest.md").read_text()
        self.assertIn("1.33", latest)
        self.assertNotIn("1.51", latest)
        manifest = json.loads(
            (self.repo_root / ".omx" / "logs" / "remote_jobs" / "local-modal-dilated-h64-authoritative-eval.json").read_text()
        )
        self.assertEqual(manifest["status"], "stale")
        self.assertIn("process/session not found", manifest["notes"])

    def test_sync_repo_writes_promoted_ledgers(self) -> None:
        from src.comma_lab import state_sync

        state_sync.sync_repo(self.repo_root)

        results_lines = [
            json.loads(line)
            for line in (self.repo_root / "reports" / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
        timeline_lines = [
            json.loads(line)
            for line in (self.repo_root / "reports" / "timeline.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(results_lines[-1]["current_workflow_score"], 1.33)
        self.assertEqual(results_lines[-1]["run_id"], "robust_current-dilated-h64-modal-cpu-2026-04-10")
        self.assertEqual(timeline_lines[-1]["score"], 1.33)


if __name__ == "__main__":
    unittest.main()
