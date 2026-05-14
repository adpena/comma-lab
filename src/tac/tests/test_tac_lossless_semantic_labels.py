# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessSemanticLabelsTests(unittest.TestCase):
    def test_pose_label_vector_quantizes_richer_nan_robust_motion_summary(self) -> None:
        from tac.lossless.semantic_labels import pose_label_vector

        pose = np.array(
            [
                [0.0, 0.02, 0.00, 0.000, 0.000, 0.00],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [8.0, 0.30, 0.05, 0.010, 0.000, 0.20],
                [16.0, 0.60, 0.10, 0.020, 0.000, 0.40],
            ],
            dtype=np.float64,
        )

        label = pose_label_vector(pose)

        self.assertEqual(len(label), 8)
        self.assertEqual(label[0], 2)  # forward motion bucket
        self.assertEqual(label[1], 1)  # lateral motion bucket
        self.assertEqual(label[2], 1)  # vertical motion bucket
        self.assertEqual(label[3], 2)  # turn magnitude bucket
        self.assertEqual(label[4], 3)  # stop/go regime: mixed stop and strong go
        self.assertEqual(label[5], 2)  # turning regime: strong right turn
        self.assertEqual(label[6], 2)  # motion variance bucket
        self.assertEqual(label[7], 3)  # jerk bucket

    def test_pose_label_vector_handles_stationary_all_nan_tail_without_noise(self) -> None:
        from tac.lossless.semantic_labels import pose_label_vector

        pose = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.01, 0.0, 0.0, 0.0, 0.01],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        label = pose_label_vector(pose)

        self.assertEqual(label, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_pose_label_vector_separates_turn_direction_regimes(self) -> None:
        from tac.lossless.semantic_labels import pose_label_vector

        left_turn = np.array(
            [
                [6.0, 0.2, 0.0, 0.0, 0.0, -0.30],
                [6.2, 0.2, 0.0, 0.0, 0.0, -0.35],
                [6.4, 0.2, 0.0, 0.0, 0.0, -0.40],
            ],
            dtype=np.float64,
        )
        right_turn = np.array(
            [
                [6.0, 0.2, 0.0, 0.0, 0.0, 0.30],
                [6.2, 0.2, 0.0, 0.0, 0.0, 0.35],
                [6.4, 0.2, 0.0, 0.0, 0.0, 0.40],
            ],
            dtype=np.float64,
        )

        left_label = pose_label_vector(left_turn)
        right_label = pose_label_vector(right_turn)

        self.assertEqual(left_label[5], 1)
        self.assertEqual(right_label[5], 2)
        self.assertEqual(left_label[:5], right_label[:5])
        self.assertEqual(left_label[6:], right_label[6:])

    def test_build_pose_label_map_sample_writes_real_name_json(self) -> None:
        from tac.lossless.semantic_labels import build_pose_label_map_sample

        examples = [
            {
                "json": {"file_name": "clip_b"},
                "pose.npy": np.full((2, 6), [12.0, 0.1, 0.0, 0.0, 0.0, 0.02], dtype=np.float64),
            },
            {
                "json": {"file_name": "clip_a"},
                "pose.npy": np.full((2, 6), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
        ]

        def fake_loader(_dataset_name: str, *, num_proc=None, data_files=None):
            return {"train": examples}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pose_labels.json"
            result = build_pose_label_map_sample(
                output_path=output_path,
                split=[0, 1],
                max_records=2,
                dataset_loader=fake_loader,
            )

            payload = json.loads(output_path.read_text())

        self.assertEqual(result["command"], "lossless_pose_labels_sample")
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(sorted(payload), ["clip_a", "clip_b"])
        self.assertEqual(payload["clip_a"], [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertGreater(payload["clip_b"][0], payload["clip_a"][0])
        self.assertEqual(len(payload["clip_b"]), 8)


if __name__ == "__main__":
    unittest.main()
