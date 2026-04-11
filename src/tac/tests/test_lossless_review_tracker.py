from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CLI_MODULE_PATH = ROOT / "src" / "comma_lab" / "cli.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("src.comma_lab.cli", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LosslessReviewTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.repo_root = Path(self.tmpdir.name)
        (self.repo_root / ".omx" / "state").mkdir(parents=True, exist_ok=True)
        tracker = {
            "version": 3,
            "last_scan": "2026-04-11T00:00:00+00:00",
            "entity_count": 4,
            "review_count": 2,
            "entities": {
                "src.tac.lossless.gpt_score::score_commavq_gpt_sample": {
                    "module": "src.tac.lossless.gpt_score",
                    "file_path": "src/tac/lossless/gpt_score.py",
                    "entity_type": "function",
                    "name": "score_commavq_gpt_sample",
                    "start_line": 1,
                    "end_line": 10,
                    "line_count": 10,
                    "complexity": 3,
                    "last_modified_commit": "abc123",
                    "last_modified_date": "2026-04-11T00:00:00+00:00",
                    "review_status": "reviewed",
                    "reviewed_by": "council_greenup",
                    "reviewed_at": "2026-04-11T01:00:00+00:00",
                    "review_pass": "pass1",
                    "notes": "",
                },
                "experiments.test_tac_lossless_gpt_score::TacLosslessGptScoreTests": {
                    "module": "experiments.test_tac_lossless_gpt_score",
                    "file_path": "experiments/test_tac_lossless_gpt_score.py",
                    "entity_type": "class",
                    "name": "TacLosslessGptScoreTests",
                    "start_line": 1,
                    "end_line": 40,
                    "line_count": 40,
                    "complexity": 2,
                    "last_modified_commit": "def456",
                    "last_modified_date": "2026-04-11T00:00:00+00:00",
                    "review_status": "unreviewed",
                    "reviewed_by": "",
                    "reviewed_at": "",
                    "review_pass": "",
                    "notes": "",
                },
                "src.tac.cli::_run_lossless": {
                    "module": "src.tac.cli",
                    "file_path": "src/tac/cli.py",
                    "entity_type": "function",
                    "name": "_run_lossless",
                    "start_line": 1,
                    "end_line": 50,
                    "line_count": 50,
                    "complexity": 10,
                    "last_modified_commit": "ghi789",
                    "last_modified_date": "2026-04-11T00:00:00+00:00",
                    "review_status": "needs_fix",
                    "reviewed_by": "council",
                    "reviewed_at": "2026-04-11T02:00:00+00:00",
                    "review_pass": "pass2",
                    "notes": "",
                },
                "src.tac.training::Trainer.fit_lazy": {
                    "module": "src.tac.training",
                    "file_path": "src/tac/training.py",
                    "entity_type": "method",
                    "name": "Trainer.fit_lazy",
                    "start_line": 1,
                    "end_line": 50,
                    "line_count": 50,
                    "complexity": 10,
                    "last_modified_commit": "zzz999",
                    "last_modified_date": "2026-04-11T00:00:00+00:00",
                    "review_status": "reviewed",
                    "reviewed_by": "council_greenup",
                    "reviewed_at": "2026-04-11T03:00:00+00:00",
                    "review_pass": "pass3",
                    "notes": "",
                },
            },
        }
        (self.repo_root / ".omx" / "state" / "review_tracker.json").write_text(json.dumps(tracker, indent=2) + "\n")

    def test_sync_repo_projects_lossless_subset_without_trampling_global_tracker(self) -> None:
        from src.comma_lab import lossless_review_tracker

        result = lossless_review_tracker.sync_repo(self.repo_root)
        payload = json.loads((self.repo_root / ".omx" / "state" / "lossless_review_tracker.json").read_text())

        self.assertEqual(result.changed_paths, (".omx/state/lossless_review_tracker.json",))
        self.assertEqual(payload["entity_count"], 3)
        self.assertEqual(payload["review_count"], 2)
        self.assertIn("src.tac.lossless.gpt_score::score_commavq_gpt_sample", payload["entities"])
        self.assertIn("experiments.test_tac_lossless_gpt_score::TacLosslessGptScoreTests", payload["entities"])
        self.assertIn("src.tac.cli::_run_lossless", payload["entities"])
        self.assertNotIn("src.tac.training::Trainer.fit_lazy", payload["entities"])

    def test_doctor_repo_reports_stale_projection(self) -> None:
        from src.comma_lab import lossless_review_tracker

        report = lossless_review_tracker.doctor_repo(self.repo_root)
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].code, "lossless_review_tracker_stale")

    def test_scan_repo_refreshes_global_tracker_before_projecting(self) -> None:
        from src.comma_lab import lossless_review_tracker

        with mock.patch("src.comma_lab.lossless_review_tracker.subprocess.run") as mocked:
            result = lossless_review_tracker.scan_repo(self.repo_root)

        self.assertEqual(result.changed_paths, (".omx/state/lossless_review_tracker.json",))
        mocked.assert_called_once()
        argv = mocked.call_args.args[0]
        self.assertEqual(argv[:3], [sys.executable, "tools/review_tracker.py", "scan"])

    def test_cli_exposes_lossless_review_sync_surface(self) -> None:
        cli = load_cli_module()
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            rc = cli.main(["lossless-review", "sync", "--repo-root", str(self.repo_root)])

        self.assertEqual(rc, 0)
        self.assertIn("lossless-review sync: changed 1 path(s)", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
