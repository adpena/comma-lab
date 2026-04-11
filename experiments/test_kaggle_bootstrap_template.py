from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "tac" / "bootstrap_codegen.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tac_bootstrap_codegen", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KaggleBootstrapTemplateTests(unittest.TestCase):
    def test_render_includes_required_entrypoint_names(self) -> None:
        mod = load_module()
        rendered = mod.render_bootstrap(
            required_symbols=(
                "build_postfilter_meta",
                "resolve_cloud_archive_source",
                "save_best_checkpoint",
            ),
            dataset_hint="comma-lab-private-assets",
        )

        self.assertIn("def ensure_tac_importable()", rendered)
        self.assertIn("def tac_has_required_entrypoints", rendered)
        self.assertIn("def find_tac_wheel_candidates", rendered)
        self.assertIn("resolve_cloud_archive_source", rendered)
        self.assertIn("comma-lab-private-assets", rendered)


if __name__ == "__main__":
    unittest.main()
