from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "reports" / "graphs" / "site"
INDEX = SITE / "index.html"
REPORT_HISTORY_HTML = SITE / "report_history.html"
REPORT_HISTORY_JSON = SITE / "report_history.json"
LOSSLESS_LATEST = SITE / "lossless_latest.md"


class BuildStaticSiteParityTests(unittest.TestCase):
    def test_history_viewer_artifacts_are_exported(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_report_history.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["python3", "reports/graphs/build_static_site.py"],
            cwd=ROOT,
            check=True,
        )

        self.assertTrue(REPORT_HISTORY_HTML.exists())
        self.assertTrue(REPORT_HISTORY_JSON.exists())
        self.assertTrue(LOSSLESS_LATEST.exists())

    def test_check_passes_after_rebuild(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_static_site.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["python3", "reports/graphs/build_static_site.py", "--check"],
            cwd=ROOT,
            check=True,
        )

    def test_check_fails_when_site_copy_drifts(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_static_site.py"],
            cwd=ROOT,
            check=True,
        )

        original = INDEX.read_text()
        try:
            INDEX.write_text(original + "\n<!-- drift -->\n")
            result = subprocess.run(
                ["python3", "reports/graphs/build_static_site.py", "--check"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Drift detected", result.stderr)
        finally:
            INDEX.write_text(original)

    def test_check_fails_when_promoted_state_drifts(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_static_site.py"],
            cwd=ROOT,
            check=True,
        )
        latest = ROOT / "reports" / "latest.md"
        original = latest.read_text()
        try:
            latest.write_text("# latest report\n\nstale 1.51\n")
            result = subprocess.run(
                ["python3", "reports/graphs/build_static_site.py", "--check"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Promoted state drift detected", result.stderr)
        finally:
            latest.write_text(original)


if __name__ == "__main__":
    unittest.main()
