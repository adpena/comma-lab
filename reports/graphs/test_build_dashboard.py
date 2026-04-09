from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "reports" / "graphs" / "index.html"


class BuildDashboardRegressionTests(unittest.TestCase):
    def test_delta_compare_uses_prior_floor_metrics(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_dashboard.py"],
            cwd=ROOT,
            check=True,
        )

        html = HTML_PATH.read_text()

        self.assertIn(
            "<tr><td>PoseNet distortion</td><td>0.04678315</td><td>0.03317023</td>",
            html,
        )
        self.assertIn(
            "<tr><td>SegNet distortion</td><td>0.00581610</td><td>0.00575544</td>",
            html,
        )
        self.assertIn("weighted ensemble h32 + MC 75/25", html)
        self.assertIn("long1000 QAT+EMA learned int8 post-filter h64", html)

    def test_homepage_uses_layered_current_floor_language(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_dashboard.py"],
            cwd=ROOT,
            check=True,
        )

        html = HTML_PATH.read_text()

        self.assertIn("Start here", html)
        self.assertIn("What the score means", html)
        self.assertIn("Why 1.73 beat 1.84", html)
        self.assertNotIn("Why 2.05 beat 2.08", html)
        self.assertNotIn("moved the floor to 2.05", html)

    def test_search_path_uses_scaled_trajectory_language(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_dashboard.py"],
            cwd=ROOT,
            check=True,
        )

        html = HTML_PATH.read_text()

        self.assertIn("Trajectory and branch points", html)
        self.assertIn("actual current_workflow score", html)
        self.assertIn("off-scale diagnostic spike", html)
        self.assertNotIn("Search path", html)

    def test_homepage_has_top_runs_explorer(self) -> None:
        subprocess.run(
            ["python3", "reports/graphs/build_dashboard.py"],
            cwd=ROOT,
            check=True,
        )

        html = HTML_PATH.read_text()

        self.assertIn("Top runs explorer", html)
        self.assertIn("Unofficial community leaderboard snapshot", html)
        self.assertIn("drag to move", html)
        self.assertIn("data-run-picker", html)
        self.assertIn("data-zoom-stage", html)


if __name__ == "__main__":
    unittest.main()
