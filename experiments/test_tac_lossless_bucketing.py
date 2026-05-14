# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessBucketingTests(unittest.TestCase):
    def test_derive_token_segment_features_reports_simple_token_signals(self) -> None:
        from tac.lossless.bucketing import derive_token_segment_features

        segment = np.array([7, 7, 2, 9], dtype=np.uint16)

        features = derive_token_segment_features(segment, segment_index=3)

        self.assertEqual(features.segment_index, 3)
        self.assertEqual(features.token_count, 4)
        self.assertEqual(features.first_token, 7)
        self.assertEqual(features.last_token, 9)
        self.assertEqual(features.unique_token_count, 3)

    def test_derive_token_segment_features_supports_empty_segments(self) -> None:
        from tac.lossless.bucketing import derive_token_segment_features

        features = derive_token_segment_features([], segment_index=0)

        self.assertEqual(features.segment_index, 0)
        self.assertEqual(features.token_count, 0)
        self.assertIsNone(features.first_token)
        self.assertIsNone(features.last_token)
        self.assertEqual(features.unique_token_count, 0)

    def test_assign_segments_to_buckets_uses_sorted_bucket_keys(self) -> None:
        from tac.lossless.bucketing import assign_segments_to_buckets, derive_token_segment_features

        segments = [
            (9, 9),
            (1,),
            (1, 4),
            (9, 8),
        ]
        features = tuple(
            derive_token_segment_features(segment, segment_index=index) for index, segment in enumerate(segments)
        )

        assignments, bucket_keys = assign_segments_to_buckets(features, key_fields=("token_count", "first_token"))

        self.assertEqual(bucket_keys, ((1, 1), (2, 1), (2, 9)))
        self.assertEqual([assignment.bucket_id for assignment in assignments], [2, 0, 1, 2])
        self.assertEqual(assignments[0].bucket_key, assignments[3].bucket_key)

    def test_build_bucketing_plan_applies_and_restores_exact_segment_order(self) -> None:
        from tac.lossless.bucketing import (
            apply_bucketing_plan,
            build_bucketing_plan,
            restore_bucketed_segments,
        )

        segments = [
            (9, 9),
            (1,),
            (1, 4),
            (9, 8),
        ]

        plan = build_bucketing_plan(segments, key_fields=("token_count", "first_token"))
        ordered_segments = apply_bucketing_plan(segments, plan)
        restored_segments = restore_bucketed_segments(ordered_segments, plan)

        self.assertEqual(plan.ordered_indices, (1, 2, 0, 3))
        self.assertEqual(plan.restore_indices, (2, 0, 1, 3))
        self.assertEqual(list(ordered_segments), [(1,), (1, 4), (9, 9), (9, 8)])
        self.assertEqual(list(restored_segments), segments)


if __name__ == "__main__":
    unittest.main()
