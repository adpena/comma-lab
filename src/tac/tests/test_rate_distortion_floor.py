# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "rd_floor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rate_distortion_floor", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RateDistortionFloorTests(unittest.TestCase):
    def test_score_from_terms_matches_formula(self) -> None:
        mod = load_module()
        score = mod.score_from_terms(pose=0.04809216, seg=0.00576402, rate=0.02301653)
        self.assertAlmostEqual(score, 1.8453003615921668, places=9)

    def test_required_pose_for_target_decreases_with_lower_target(self) -> None:
        mod = load_module()
        loose = mod.required_pose_for_target(target=1.85, seg=0.00576402, rate=0.02301653)
        strict = mod.required_pose_for_target(target=1.80, seg=0.00576402, rate=0.02301653)
        self.assertLess(strict, loose)

    def test_pareto_frontier_keeps_only_monotone_score_improvements(self) -> None:
        mod = load_module()
        points = [
            mod.RunPoint("a", "a.json", 2.1, 900_000, 0.1, 0.01, 0.02),
            mod.RunPoint("b", "b.json", 2.2, 910_000, 0.1, 0.01, 0.02),
            mod.RunPoint("c", "c.json", 1.9, 920_000, 0.1, 0.01, 0.02),
            mod.RunPoint("d", "d.json", 1.95, 930_000, 0.1, 0.01, 0.02),
        ]
        frontier = mod.pareto_frontier(points)
        self.assertEqual([point.label for point in frontier], ["a", "c"])

    def test_load_summary_points_filters_non_matching_json(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "good").mkdir()
            (root / "bad").mkdir()
            (root / "good" / "summary.json").write_text(
                """
                {
                  "track": "robust_current",
                  "current_workflow_score": 1.85,
                  "current_workflow_archive_bytes": 864167,
                  "pose_distortion": 0.04809216,
                  "seg_distortion": 0.00576402,
                  "current_workflow_rate": 0.02301653
                }
                """
            )
            (root / "bad" / "summary.json").write_text("{\"hello\": \"world\"}")
            points = mod.load_summary_points(root)
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0].score, 1.85)


if __name__ == "__main__":
    unittest.main()
