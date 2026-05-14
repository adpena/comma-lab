# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessHybridSelectorTests(unittest.TestCase):
    def test_select_exact_candidate_prefers_ordered_metrics(self) -> None:
        from tac.lossless.hybrid_selector import SelectionMetric, select_exact_candidate

        candidates = [
            {"strategy": "slow_exact", "exact_match": True, "archive_bytes": 640, "encode_seconds": 3.2},
            {"strategy": "inexact_but_smaller", "exact_match": False, "archive_bytes": 512, "encode_seconds": 2.1},
            {"strategy": "fast_exact", "exact_match": True, "archive_bytes": 640, "encode_seconds": 1.8},
            {"strategy": "smallest_exact", "exact_match": True, "archive_bytes": 600, "encode_seconds": 4.1},
        ]

        winner = select_exact_candidate(
            candidates,
            metrics=(
                SelectionMetric("archive_bytes", "min"),
                SelectionMetric("encode_seconds", "min"),
            ),
        )

        self.assertEqual(winner["strategy"], "smallest_exact")

    def test_select_exact_candidate_supports_max_metrics(self) -> None:
        from tac.lossless.hybrid_selector import SelectionMetric, select_exact_candidate

        candidates = [
            {"strategy": "baseline", "exact_match": True, "compression_rate": 3.8, "archive_bytes": 620},
            {"strategy": "frontier", "exact_match": True, "compression_rate": 4.1, "archive_bytes": 650},
            {"strategy": "inexact", "exact_match": False, "compression_rate": 5.0, "archive_bytes": 400},
        ]

        winner = select_exact_candidate(
            candidates,
            metrics=(
                SelectionMetric("compression_rate", "max"),
                SelectionMetric("archive_bytes", "min"),
            ),
        )

        self.assertEqual(winner["strategy"], "frontier")

    def test_rank_exact_candidates_uses_deterministic_summary_fallback(self) -> None:
        from tac.lossless.hybrid_selector import SelectionMetric, rank_exact_candidates

        candidates = [
            {"strategy": "bravo", "exact_match": True, "archive_bytes": 700},
            {"strategy": "alpha", "exact_match": True, "archive_bytes": 700},
            {"strategy": "charlie", "exact_match": True, "archive_bytes": 710},
        ]

        ranked = rank_exact_candidates(
            candidates,
            metrics=(SelectionMetric("archive_bytes", "min"),),
        )
        reversed_ranked = rank_exact_candidates(
            list(reversed(candidates)),
            metrics=(SelectionMetric("archive_bytes", "min"),),
        )

        self.assertEqual([item["strategy"] for item in ranked], ["alpha", "bravo", "charlie"])
        self.assertEqual([item["strategy"] for item in reversed_ranked], ["alpha", "bravo", "charlie"])

    def test_select_exact_candidate_requires_at_least_one_exact_match(self) -> None:
        from tac.lossless.hybrid_selector import SelectionMetric, select_exact_candidate

        candidates = [
            {"strategy": "candidate_a", "exact_match": False, "archive_bytes": 700},
            {"strategy": "candidate_b", "exact_match": False, "archive_bytes": 680},
        ]

        with self.assertRaisesRegex(ValueError, "exact"):
            select_exact_candidate(
                candidates,
                metrics=(SelectionMetric("archive_bytes", "min"),),
            )

    def test_rank_exact_candidates_validates_required_metric_and_exact_flag(self) -> None:
        from tac.lossless.hybrid_selector import SelectionMetric, rank_exact_candidates

        with self.assertRaisesRegex(ValueError, "exact_match"):
            rank_exact_candidates(
                [{"strategy": "missing_exact", "archive_bytes": 700}],
                metrics=(SelectionMetric("archive_bytes", "min"),),
            )

        with self.assertRaisesRegex(ValueError, "archive_bytes"):
            rank_exact_candidates(
                [{"strategy": "missing_metric", "exact_match": True}],
                metrics=(SelectionMetric("archive_bytes", "min"),),
            )


if __name__ == "__main__":
    unittest.main()
