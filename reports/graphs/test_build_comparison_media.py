from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "reports" / "graphs" / "build_comparison_media.py"


class BuildComparisonMediaRegressionTests(unittest.TestCase):
    def test_zoom_reference_is_not_hardcoded_to_floor_184(self) -> None:
        source = SOURCE_PATH.read_text()

        self.assertNotIn('variant["id"] == "floor_184"', source)


if __name__ == "__main__":
    unittest.main()
