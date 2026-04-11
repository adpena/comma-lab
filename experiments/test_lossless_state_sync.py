from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CLI_MODULE_PATH = ROOT / "src" / "comma_lab" / "cli.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_cli_module():
    spec = importlib.util.spec_from_file_location("src.comma_lab.cli", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LosslessStateSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = Path(self.tmpdir.name)

        write_json(
            self.repo_root / ".omx" / "state" / "lossless_promoted_result.json",
            {
                "profile": "lzma_baseline",
                "archive_path": str(self.repo_root / "reports" / "lossless" / "submission.zip"),
                "archive_bytes": 120,
                "original_bytes": 480,
                "compression_rate": 4.0,
                "method": "lzma",
            },
        )
        archive = self.repo_root / "reports" / "lossless" / "submission.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"zip-bytes")
        (self.repo_root / "reports" / "lossless_latest.md").parent.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "reports" / "lossless_latest.md").write_text("stale latest\n")
        (self.repo_root / "reports" / "lossless_results.jsonl").write_text("")
        (self.repo_root / "reports" / "lossless_timeline.jsonl").write_text("")
        (self.repo_root / ".omx" / "state" / "lossless_focus.md").parent.mkdir(parents=True, exist_ok=True)
        (self.repo_root / ".omx" / "state" / "lossless_focus.md").write_text("stale focus\n")
        (self.repo_root / ".omx" / "state" / "lossless_next_experiments.md").write_text("stale next\n")
        (self.repo_root / ".omx" / "research" / "lossless_findings.md").parent.mkdir(parents=True, exist_ok=True)
        (self.repo_root / ".omx" / "research" / "lossless_findings.md").write_text("stale findings\n")

        # lossy surfaces must remain untouched
        (self.repo_root / "reports" / "latest.md").write_text("lossy latest\n")
        (self.repo_root / "reports" / "results.jsonl").write_text('{"score": 1.33}\n')
        (self.repo_root / "reports" / "timeline.jsonl").write_text('{"score": 1.33}\n')
        (self.repo_root / ".omx" / "state" / "current_focus.md").write_text("lossy focus\n")
        (self.repo_root / ".omx" / "research" / "findings.md").write_text("lossy findings\n")

    def test_doctor_detects_stale_lossless_surfaces(self) -> None:
        from src.comma_lab import lossless_state_sync

        report = lossless_state_sync.doctor_repo(self.repo_root)

        codes = {finding.code for finding in report.findings}
        self.assertEqual(
            codes,
            {
                "lossless_latest_stale",
                "lossless_results_stale",
                "lossless_timeline_stale",
                "lossless_focus_stale",
                "lossless_next_experiments_stale",
                "lossless_findings_stale",
            },
        )

    def test_sync_repo_repairs_lossless_surfaces_without_touching_lossy_state(self) -> None:
        from src.comma_lab import lossless_state_sync

        result = lossless_state_sync.sync_repo(self.repo_root)

        self.assertEqual(
            set(result.changed_paths),
            {
                "reports/lossless_latest.md",
                "reports/lossless_results.jsonl",
                "reports/lossless_timeline.jsonl",
                ".omx/state/lossless_focus.md",
                ".omx/state/lossless_next_experiments.md",
                ".omx/research/lossless_findings.md",
            },
        )
        latest = (self.repo_root / "reports" / "lossless_latest.md").read_text()
        focus = (self.repo_root / ".omx" / "state" / "lossless_focus.md").read_text()
        next_experiments = (self.repo_root / ".omx" / "state" / "lossless_next_experiments.md").read_text()
        findings = (self.repo_root / ".omx" / "research" / "lossless_findings.md").read_text()
        results_rows = [
            json.loads(line)
            for line in (self.repo_root / "reports" / "lossless_results.jsonl").read_text().splitlines()
            if line.strip()
        ]
        timeline_rows = [
            json.loads(line)
            for line in (self.repo_root / "reports" / "lossless_timeline.jsonl").read_text().splitlines()
            if line.strip()
        ]

        self.assertIn("4.0000", latest)
        self.assertIn("lzma_baseline", focus)
        self.assertIn("promoted baseline", next_experiments)
        self.assertIn("separate from the lossy promotion flow", findings)
        self.assertEqual(results_rows[-1]["compression_rate"], 4.0)
        self.assertEqual(results_rows[-1]["profile"], "lzma_baseline")
        self.assertEqual(timeline_rows[-1]["compression_rate"], 4.0)
        self.assertEqual(timeline_rows[-1]["event"], "promotion")

        self.assertEqual((self.repo_root / "reports" / "latest.md").read_text(), "lossy latest\n")
        self.assertEqual((self.repo_root / "reports" / "results.jsonl").read_text(), '{"score": 1.33}\n')
        self.assertEqual((self.repo_root / "reports" / "timeline.jsonl").read_text(), '{"score": 1.33}\n')
        self.assertEqual((self.repo_root / ".omx" / "state" / "current_focus.md").read_text(), "lossy focus\n")
        self.assertEqual((self.repo_root / ".omx" / "research" / "findings.md").read_text(), "lossy findings\n")

    def test_sync_repo_is_idempotent_after_repair(self) -> None:
        from src.comma_lab import lossless_state_sync

        first = lossless_state_sync.sync_repo(self.repo_root)
        second = lossless_state_sync.sync_repo(self.repo_root)

        self.assertTrue(first.changed_paths)
        self.assertEqual(second.changed_paths, ())

    def test_cli_exposes_separate_lossless_state_sync_surface(self) -> None:
        cli = load_cli_module()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            rc = cli.main(["lossless-state", "sync", "--repo-root", str(self.repo_root)])

        self.assertEqual(rc, 0)
        self.assertIn("lossless-state sync: changed 6 path(s)", stdout.getvalue())
        self.assertIn("4.0000", (self.repo_root / "reports" / "lossless_latest.md").read_text())

    def test_cli_exposes_lossless_state_promote_surface(self) -> None:
        cli = load_cli_module()
        source_record = self.repo_root / "incoming_lossless_result.json"
        promoted_archive = self.repo_root / "reports" / "lossless" / "submission-v2.zip"
        promoted_archive.parent.mkdir(parents=True, exist_ok=True)
        promoted_archive.write_bytes(b"zip-bytes")
        write_json(
            source_record,
            {
                "profile": "lzma_baseline",
                "archive_path": str(promoted_archive),
                "archive_bytes": 100,
                "original_bytes": 500,
                "compression_rate": 5.0,
                "method": "lzma",
            },
        )
        (self.repo_root / ".omx" / "state" / "lossless_promoted_result.json").unlink()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            rc = cli.main(
                [
                    "lossless-state",
                    "promote",
                    "--repo-root",
                    str(self.repo_root),
                    "--record",
                    str(source_record),
                ]
            )

        self.assertEqual(rc, 0)
        self.assertIn("lossless-state promote: changed", stdout.getvalue())
        canonical = json.loads((self.repo_root / ".omx" / "state" / "lossless_promoted_result.json").read_text())
        self.assertEqual(canonical["compression_rate"], 5.0)
        self.assertIn("5.0000", (self.repo_root / "reports" / "lossless_latest.md").read_text())

    def test_lossless_state_promote_requires_archive_to_exist(self) -> None:
        from src.comma_lab import lossless_state_sync

        record = self.repo_root / "incoming_lossless_result.json"
        write_json(
            record,
            {
                "profile": "lzma_baseline",
                "archive_path": str(self.repo_root / "missing.zip"),
                "archive_bytes": 100,
                "original_bytes": 500,
                "compression_rate": 5.0,
                "method": "lzma",
            },
        )

        with self.assertRaises(FileNotFoundError):
            lossless_state_sync.promote_record(self.repo_root, record_path=record)

    def test_lossless_state_promote_requires_archive_bytes_to_match_file(self) -> None:
        from src.comma_lab import lossless_state_sync

        archive = self.repo_root / "reports" / "lossless" / "submission-v3.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"zip-bytes")
        record = self.repo_root / "incoming_lossless_result.json"
        write_json(
            record,
            {
                "profile": "lzma_baseline",
                "archive_path": str(archive),
                "archive_bytes": archive.stat().st_size + 2,
                "original_bytes": 500,
                "compression_rate": 5.0,
                "method": "lzma",
            },
        )

        with self.assertRaisesRegex(ValueError, "archive_bytes"):
            lossless_state_sync.promote_record(self.repo_root, record_path=record)


if __name__ == "__main__":
    unittest.main()
