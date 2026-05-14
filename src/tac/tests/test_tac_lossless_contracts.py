# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessContractsTests(unittest.TestCase):
    def test_tac_package_import_is_lightweight(self) -> None:
        import tac

        self.assertTrue(hasattr(tac, "__version__"))

    def test_lossless_result_models_are_typed(self) -> None:
        from tac.lossless.contracts import LosslessCompressionResult, LosslessVerificationResult

        result = LosslessCompressionResult(
            profile="lzma_baseline",
            archive_path="submission.zip",
            archive_bytes=100,
            original_bytes=400,
            compression_rate=4.0,
            method="lzma",
        )
        verify = LosslessVerificationResult(
            exact_match=True,
            checked_items=10,
            mismatch_count=0,
        )

        self.assertEqual(result.profile, "lzma_baseline")
        self.assertEqual(result.compression_rate, 4.0)
        self.assertTrue(verify.exact_match)
        self.assertEqual(verify.checked_items, 10)

    def test_lossless_profiles_include_expected_named_baselines(self) -> None:
        from tac.lossless.profiles import PROFILES

        self.assertIn("lzma_baseline", PROFILES)
        self.assertIn("zpaq_baseline", PROFILES)
        self.assertIn("gpt_arithmetic_small", PROFILES)
        self.assertIn("gpt_arithmetic_large", PROFILES)
        self.assertIn("neural_codec_smoke", PROFILES)


if __name__ == "__main__":
    unittest.main()
