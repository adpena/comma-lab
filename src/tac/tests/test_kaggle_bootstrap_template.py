from __future__ import annotations

import unittest

from tac.bootstrap_codegen import render_bootstrap


class KaggleBootstrapTemplateTests(unittest.TestCase):
    def _render(self, symbols=("build_postfilter_meta", "save_best_checkpoint")):
        return render_bootstrap(
            required_symbols=symbols,
            dataset_hint="comma-lab-private-assets",
        )

    def test_render_compiles(self) -> None:
        """Generated preamble must be syntactically valid Python."""
        rendered = self._render()
        compile(rendered, "<preamble>", "exec")

    def test_render_includes_bootstrap_stub(self) -> None:
        """Preamble must embed the two-stage _tac_bootstrap function."""
        rendered = self._render()
        self.assertIn("def _tac_bootstrap(", rendered)
        self.assertIn("_tac_bootstrap(", rendered)

    def test_render_includes_dataset_hint(self) -> None:
        rendered = self._render()
        self.assertIn("comma-lab-private-assets", rendered)

    def test_render_includes_required_symbols(self) -> None:
        rendered = self._render(("build_postfilter_meta", "resolve_cloud_archive_source"))
        self.assertIn("resolve_cloud_archive_source", rendered)
        self.assertIn("build_postfilter_meta", rendered)

    def test_render_checks_missing_entrypoints(self) -> None:
        """Must raise ImportError if required symbols absent from tac.entrypoints."""
        rendered = self._render()
        self.assertIn("_missing_ep", rendered)
        # Uses 'from tac import entrypoints' form
        self.assertIn("from tac import entrypoints", rendered)

    def test_render_defines_script_path(self) -> None:
        """SCRIPT_PATH must be defined so _tac_bootstrap can search the script dir."""
        rendered = self._render()
        self.assertIn("SCRIPT_PATH", rendered)
        self.assertIn("Path(__file__).resolve()", rendered)

    def test_render_uses_uv_or_pip(self) -> None:
        """BOOTSTRAP_STUB must contain uv-preferred install with pip fallback."""
        rendered = self._render()
        # Install command uses list form: ["pip", "install", ...]
        self.assertIn('"pip", "install"', rendered)
        self.assertIn("shutil.which", rendered)


if __name__ == "__main__":
    unittest.main()
