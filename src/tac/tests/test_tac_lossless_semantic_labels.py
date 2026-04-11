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
    def test_pose_label_vector_quantizes_nan_robust_motion_summary(self) -> None:
        from tac.lossless.semantic_labels import pose_label_vector

        pose = np.array(
            [
                [10.0, 0.10, 0.02, 0.001, 0.002, 0.03],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [14.0, 0.20, 0.04, 0.003, 0.004, 0.05],
            ],
            dtype=np.float64,
        )

        label = pose_label_vector(pose)

        self.assertEqual(len(label), 4)
        self.assertEqual(label[0], 3)  # forward speed bucket
        self.assertEqual(label[1], 0)  # lateral bucket
        self.assertEqual(label[2], 0)  # vertical bucket
        self.assertEqual(label[3], 0)  # angular bucket

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
        self.assertEqual(payload["clip_a"][0], 0)
        self.assertGreater(payload["clip_b"][0], payload["clip_a"][0])


if __name__ == "__main__":
    unittest.main()
