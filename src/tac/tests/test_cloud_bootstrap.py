"""Tests for tac.deploy.cloud_bootstrap — canonical tac wheel bootstrap module."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tac.deploy.cloud_bootstrap import (
    BOOTSTRAP_STUB,
    DEFAULT_DATASET_HINT,
    WHEEL_GLOBS,
    _is_importable,
    bootstrap,
    find_wheel,
)


# ---------------------------------------------------------------------------
# WHEEL_GLOBS + constants
# ---------------------------------------------------------------------------

class TestConstants(unittest.TestCase):
    def test_wheel_globs_has_both_names(self) -> None:
        self.assertIn("tac-*.whl", WHEEL_GLOBS)
        self.assertIn("comma_video_lab_ball_pack-*.whl", WHEEL_GLOBS)

    def test_tac_glob_first_for_priority(self) -> None:
        self.assertEqual(WHEEL_GLOBS[0], "tac-*.whl")

    def test_default_dataset_hint(self) -> None:
        self.assertEqual(DEFAULT_DATASET_HINT, "comma-lab-private-assets")


# ---------------------------------------------------------------------------
# find_wheel
# ---------------------------------------------------------------------------

class TestFindWheel(unittest.TestCase):
    def test_finds_tac_wheel_directly_in_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            result = find_wheel(root)
            self.assertEqual(result.name, "tac-1.0.0-py3-none-any.whl")

    def test_finds_legacy_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl").touch()
            result = find_wheel(root)
            self.assertEqual(result.name, "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl")

    def test_tac_wheel_takes_priority_over_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            (root / "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl").touch()
            result = find_wheel(root)
            self.assertTrue(result.name.startswith("tac-"))

    def test_finds_wheel_in_dataset_hint_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "comma-lab-private-assets"
            sub.mkdir()
            (sub / "tac-1.2.3-py3-none-any.whl").touch()
            result = find_wheel(root)
            self.assertEqual(result.name, "tac-1.2.3-py3-none-any.whl")

    def test_finds_latest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            (root / "tac-2.0.0-py3-none-any.whl").touch()
            result = find_wheel(root)
            self.assertEqual(result.name, "tac-2.0.0-py3-none-any.whl")

    def test_extra_roots_searched_before_input_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = root / "bundle"
            extra.mkdir()
            (extra / "tac-3.0.0-py3-none-any.whl").touch()
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            result = find_wheel(root, extra_roots=(extra,))
            self.assertEqual(result.name, "tac-3.0.0-py3-none-any.whl")

    def test_raises_import_error_with_instructions_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ImportError) as ctx:
                find_wheel(Path(tmp))
            msg = str(ctx.exception)
            self.assertIn("tac wheel not found", msg)
            self.assertIn("kaggle datasets version", msg)


# ---------------------------------------------------------------------------
# _is_importable
# ---------------------------------------------------------------------------

class TestIsImportable(unittest.TestCase):
    def test_stdlib_module_is_importable(self) -> None:
        self.assertTrue(_is_importable("json"))

    def test_nonexistent_module_is_not_importable(self) -> None:
        self.assertFalse(_is_importable("_tac_nonexistent_pkg_xyz123"))


# ---------------------------------------------------------------------------
# bootstrap — idempotency
# ---------------------------------------------------------------------------

class TestBootstrap(unittest.TestCase):
    def test_no_op_when_tac_already_importable(self) -> None:
        """bootstrap() must not call subprocess when tac is already installed."""
        with mock.patch("subprocess.check_call") as mock_sub:
            bootstrap()
        mock_sub.assert_not_called()

    def test_raises_with_verify_submodule_message_for_old_wheel(self) -> None:
        """Pre-v1.0.0 wheel: tac importable but required submodule missing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-0.9.0-py3-none-any.whl").touch()

            def fake_importable(module: str) -> bool:
                return module == "tac"

            with mock.patch("subprocess.check_call"), \
                 mock.patch("tac.deploy.cloud_bootstrap._is_importable",
                            side_effect=fake_importable):
                with self.assertRaises(ImportError) as ctx:
                    bootstrap(root, verify_submodule="tac.deploy.kaggle.runner")
            msg = str(ctx.exception)
            self.assertIn("pre-v1.0.0", msg)
            self.assertIn("kaggle datasets version", msg)
            self.assertIn("tac v1.0.0", msg)

    def test_raises_with_outdated_message_when_verify_module_fails(self) -> None:
        """Completely broken wheel: even tac core fails to import."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()

            with mock.patch("subprocess.check_call"), \
                 mock.patch("tac.deploy.cloud_bootstrap._is_importable",
                            return_value=False):
                with self.assertRaises(ImportError) as ctx:
                    bootstrap(root)
            msg = str(ctx.exception)
            self.assertIn("kaggle datasets version", msg)

    def test_entrypoint_symbols_verified(self) -> None:
        """Missing entrypoints must raise ImportError with rebuild instructions."""
        def fake_importable(module: str) -> bool:
            return True  # tac + submodule both "importable"

        class _FakeEP:
            build_postfilter_meta = None
            # save_best_checkpoint intentionally missing

        with mock.patch("tac.deploy.cloud_bootstrap._is_importable",
                        side_effect=fake_importable), \
             mock.patch.dict(sys.modules, {"tac.entrypoints": _FakeEP}), \
             mock.patch("subprocess.check_call"):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "tac-1.0.0-py3-none-any.whl").touch()
                # tac "importable" so skip install; then verify entrypoints
                # (re-trigger: patch _is_importable to return False for initial check)
                def ep_is_importable(m: str) -> bool:
                    return m != "tac"  # force install path
                with mock.patch("tac.deploy.cloud_bootstrap._is_importable",
                                side_effect=ep_is_importable):
                    with self.assertRaises(ImportError) as ctx:
                        bootstrap(
                            Path(tmp),
                            entrypoint_symbols=("build_postfilter_meta", "save_best_checkpoint"),
                        )
            # Error should mention missing entrypoints or rebuild instructions
            self.assertIn("kaggle datasets version", str(ctx.exception))


# ---------------------------------------------------------------------------
# BOOTSTRAP_STUB — structural checks
# ---------------------------------------------------------------------------

class TestBootstrapStub(unittest.TestCase):
    def test_stub_compiles(self) -> None:
        compile(BOOTSTRAP_STUB, "<BOOTSTRAP_STUB>", "exec")

    def test_stub_defines_tac_bootstrap_function(self) -> None:
        self.assertIn("def _tac_bootstrap(", BOOTSTRAP_STUB)

    def test_stub_handles_uv_and_pip(self) -> None:
        self.assertIn("shutil.which", BOOTSTRAP_STUB)
        # Install command uses list form: ["pip", "install", ...]
        self.assertIn('"pip", "install"', BOOTSTRAP_STUB)

    def test_stub_has_idempotency_check(self) -> None:
        # Uses __import__ via _imp() helper for dynamic idempotency check
        self.assertIn('_imp("tac")', BOOTSTRAP_STUB)

    def test_stub_has_kaggle_fallback_path(self) -> None:
        self.assertIn("/kaggle/input", BOOTSTRAP_STUB)

    def test_stub_has_modal_fallback_path(self) -> None:
        self.assertIn("/vol/input", BOOTSTRAP_STUB)

    def test_stub_has_lightning_fallback_path(self) -> None:
        self.assertIn("/teamspace", BOOTSTRAP_STUB)


if __name__ == "__main__":
    unittest.main()
