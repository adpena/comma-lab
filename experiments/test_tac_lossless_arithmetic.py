from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessArithmeticTests(unittest.TestCase):
    def test_flatten_tokens_for_gpt_arithmetic_adds_frame_bos_and_segment_eot(self) -> None:
        import numpy as np

        from tac.lossless.arithmetic import flatten_tokens_for_gpt_arithmetic

        tokens = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            dtype=np.int16,
        )

        flat = flatten_tokens_for_gpt_arithmetic(tokens, bos_token=1024, eot_token=1025)

        self.assertEqual(flat.tolist(), [1024, 1, 2, 3, 4, 1024, 5, 6, 7, 8, 1025])

    def test_build_gpt_arithmetic_plan_returns_typed_small_plan(self) -> None:
        from tac.lossless.arithmetic import build_gpt_arithmetic_plan

        plan = build_gpt_arithmetic_plan(
            "gpt_arithmetic_small",
            split=["0", "1"],
            work_dir="/tmp/tac-lossless-arithmetic",
        )

        self.assertEqual(plan.profile, "gpt_arithmetic_small")
        self.assertEqual(plan.method, "gpt_arithmetic")
        self.assertEqual(plan.model, "small")
        self.assertEqual(plan.context_tokens, 256)
        self.assertEqual(plan.status, "planned")
        self.assertFalse(plan.measured)
        self.assertEqual(plan.split, ("0", "1"))

    def test_build_gpt_arithmetic_plan_returns_typed_large_plan(self) -> None:
        from tac.lossless.arithmetic import build_gpt_arithmetic_plan

        plan = build_gpt_arithmetic_plan("gpt_arithmetic_large")

        self.assertEqual(plan.profile, "gpt_arithmetic_large")
        self.assertEqual(plan.model, "large")
        self.assertEqual(plan.context_tokens, 1024)
        self.assertEqual(plan.dataset_name, "commaai/commavq")

    def test_build_gpt_arithmetic_plan_rejects_non_arithmetic_profiles(self) -> None:
        from tac.lossless.arithmetic import build_gpt_arithmetic_plan

        with self.assertRaisesRegex(ValueError, "unsupported arithmetic method"):
            build_gpt_arithmetic_plan("lzma_baseline")

    def test_lossless_package_exports_arithmetic_scaffold(self) -> None:
        from tac.lossless import build_gpt_arithmetic_plan

        plan = build_gpt_arithmetic_plan("gpt_arithmetic_small")
        self.assertEqual(plan.model, "small")

    def test_estimate_gpt_arithmetic_workload_uses_dataset_loader_and_reports_counts(self) -> None:
        import numpy as np

        from tac.lossless.arithmetic import estimate_gpt_arithmetic_workload

        calls = []

        def fake_loader(dataset_name: str, *, num_proc=None, data_files=None):
            calls.append(
                {
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "data_files": data_files,
                }
            )
            return {
                "train": [
                    {"json": {"file_name": "clip_a"}, "token.npy": np.arange(16, dtype=np.int16).reshape(2, 2, 4)},
                    {"json": {"file_name": "clip_b"}, "token.npy": (np.arange(16, dtype=np.int16) + 16).reshape(2, 2, 4)},
                ]
            }

        estimate = estimate_gpt_arithmetic_workload(
            "gpt_arithmetic_small",
            split=["0", "1"],
            dataset_loader=fake_loader,
            num_proc=2,
        )

        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["data_files"], {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]})
        self.assertEqual(estimate.profile, "gpt_arithmetic_small")
        self.assertEqual(estimate.example_count, 2)
        self.assertEqual(estimate.frames_per_example, 2)
        self.assertEqual(estimate.tokens_per_frame, 9)
        self.assertEqual(estimate.flat_tokens_per_example, 19)
        self.assertEqual(estimate.total_flat_tokens, 38)
        self.assertFalse(estimate.measured)


if __name__ == "__main__":
    unittest.main()
