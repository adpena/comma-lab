from __future__ import annotations

import unittest

from experiments import build_kaggle_kernels as mod


class BuildKaggleKernelsTests(unittest.TestCase):
    def test_dataset_backed_kernels_do_not_bundle_runtime_assets(self) -> None:
        specs = mod.kernel_specs()

        dilated = specs["dilated_h64_long1000"]
        segnet = specs["segnet_attack_fixed_h32"]

        self.assertEqual(dilated.dataset_sources, (mod.ASSET_DATASET_REF,))
        self.assertEqual(segnet.dataset_sources, (mod.ASSET_DATASET_REF,))
        self.assertEqual(dilated.include_paths, ())
        self.assertEqual(segnet.include_paths, ())


if __name__ == "__main__":
    unittest.main()
