from __future__ import annotations

import unittest

import tac.deploy.kaggle.build_kaggle_kernels as mod


class BuildKaggleKernelsTests(unittest.TestCase):
    def test_dataset_backed_kernels_do_not_bundle_runtime_assets(self) -> None:
        specs = mod.kernel_specs()

        dilated = specs["dilated_h64_long1000"]
        segnet = specs["segnet_attack_fixed_h32"]

        expected_ref = mod.kaggle_dataset_ref()
        self.assertEqual(dilated.dataset_sources, (expected_ref,))
        self.assertEqual(segnet.dataset_sources, (expected_ref,))
        self.assertEqual(dilated.include_paths, ())
        self.assertEqual(segnet.include_paths, ())
        # launch_policy was stripped in round 28 (silently ignored by Kaggle server)
        self.assertIsNone(dilated.launch_policy)
        self.assertIsNone(segnet.launch_policy)


if __name__ == "__main__":
    unittest.main()
