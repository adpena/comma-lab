"""Tests for tac.deploy.kaggle.runner.

Covers: find_tac_wheel, resolve_training_script, resolve_supervision_assets,
_strip_flags, build_kaggle_command (flag structure only), save_manifest.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.deploy.kaggle.runner import (  # noqa: E402
    WHEEL_GLOBS,
    _strip_flags,
    ensure_tac,
    find_tac_wheel,
    resolve_supervision_assets,
    resolve_training_script,
    save_manifest,
)


# ---------------------------------------------------------------------------
# find_tac_wheel
# ---------------------------------------------------------------------------

class TestFindTacWheel(unittest.TestCase):
    def test_finds_tac_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            result = find_tac_wheel(root)
            self.assertEqual(result.name, "tac-1.0.0-py3-none-any.whl")

    def test_finds_legacy_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl").touch()
            result = find_tac_wheel(root)
            self.assertEqual(result.name, "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl")

    def test_tac_wheel_takes_priority_over_legacy(self) -> None:
        """tac-*.whl is first in WHEEL_GLOBS so it should win."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            (root / "comma_video_lab_ball_pack-0.9.0-py3-none-any.whl").touch()
            result = find_tac_wheel(root)
            self.assertTrue(result.name.startswith("tac-"))

    def test_finds_wheel_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "comma-lab-private-assets"
            sub.mkdir()
            (sub / "tac-1.2.3-py3-none-any.whl").touch()
            result = find_tac_wheel(root)
            self.assertEqual(result.name, "tac-1.2.3-py3-none-any.whl")

    def test_returns_latest_version_when_multiple(self) -> None:
        """Sorted order: tac-1.0.0 < tac-2.0.0 → last candidate wins."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tac-1.0.0-py3-none-any.whl").touch()
            (root / "tac-2.0.0-py3-none-any.whl").touch()
            result = find_tac_wheel(root)
            self.assertEqual(result.name, "tac-2.0.0-py3-none-any.whl")

    def test_raises_import_error_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ImportError) as ctx:
                find_tac_wheel(Path(tmp))
            self.assertIn("tac wheel not found", str(ctx.exception))
            self.assertIn("kaggle datasets version", str(ctx.exception))

    def test_wheel_globs_covers_both_names(self) -> None:
        self.assertIn("tac-*.whl", WHEEL_GLOBS)
        self.assertIn("comma_video_lab_ball_pack-*.whl", WHEEL_GLOBS)
        # tac must come first — it takes priority
        self.assertEqual(WHEEL_GLOBS[0], "tac-*.whl")


# ---------------------------------------------------------------------------
# ensure_tac — post-install verification
# ---------------------------------------------------------------------------

class TestEnsureTac(unittest.TestCase):
    def test_skips_install_if_already_importable(self) -> None:
        """If tac.deploy.kaggle.runner is already importable, no pip install runs."""
        with mock.patch("subprocess.check_call") as mock_pip:
            ensure_tac(Path("/nonexistent"))
        mock_pip.assert_not_called()

    def test_raises_if_post_install_runner_missing(self) -> None:
        """Old wheel installs tac but lacks runner — must raise with upload instructions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "tac-0.9.0-py3-none-any.whl"
            wheel.touch()

            # Simulate: tac not importable → triggers install path
            # After install: tac importable but runner still missing
            import_call_count = [0]
            def fake_import(name: str, *_a: object, **_kw: object) -> None:
                import_call_count[0] += 1
                raise ImportError(f"no module named {name}")

            with mock.patch("subprocess.check_call"), \
                 mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(ImportError) as ctx:
                    ensure_tac(root)
            self.assertIn("pre-v1.0.0", str(ctx.exception))
            self.assertIn("kaggle datasets version", str(ctx.exception))
            self.assertIn("tac v1.0.0", str(ctx.exception))


# ---------------------------------------------------------------------------
# resolve_training_script
# ---------------------------------------------------------------------------

class TestResolveTrainingScript(unittest.TestCase):
    def test_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train_renderer_fridrich.py"
            script.touch()
            launcher = root / "kaggle_asym_warp_launcher.py"
            result = resolve_training_script(launcher)
            self.assertEqual(result, script)

    def test_nested_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "experiments" / "train_renderer_fridrich.py"
            nested.parent.mkdir()
            nested.touch()
            launcher = root / "kaggle_asym_warp_launcher.py"
            result = resolve_training_script(launcher)
            self.assertEqual(result, nested)

    def test_flat_takes_priority_over_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "train_renderer_fridrich.py"
            flat.touch()
            nested = root / "experiments" / "train_renderer_fridrich.py"
            nested.parent.mkdir()
            nested.touch()
            launcher = root / "kaggle_asym_warp_launcher.py"
            result = resolve_training_script(launcher)
            self.assertEqual(result, flat)

    def test_raises_when_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "kaggle_asym_warp_launcher.py"
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_training_script(launcher)
            self.assertIn("train_renderer_fridrich.py", str(ctx.exception))
            self.assertIn("Tried", str(ctx.exception))


# ---------------------------------------------------------------------------
# resolve_supervision_assets
# ---------------------------------------------------------------------------

class TestResolveSupervisionAssets(unittest.TestCase):
    def _make_asset_root(self, tmp: str, *, raft: bool = False, targets: bool = False) -> Path:
        root = Path(tmp)
        if raft:
            (root / "raft_flow.pt").touch()
        if targets:
            (root / "posenet_targets.bin").touch()
        return root

    def test_base_variant_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assets = resolve_supervision_assets("base", Path(tmp))
            self.assertEqual(assets, {})

    def test_raft_only_returns_raft_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_asset_root(tmp, raft=True)
            assets = resolve_supervision_assets("raft_only", root)
            self.assertIn("raft_flow", assets)
            self.assertNotIn("posenet_targets", assets)

    def test_supervised_returns_both_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_asset_root(tmp, raft=True, targets=True)
            assets = resolve_supervision_assets("supervised", root)
            self.assertIn("raft_flow", assets)
            self.assertIn("posenet_targets", assets)

    def test_raft_only_missing_raft_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_supervision_assets("raft_only", Path(tmp))
            msg = str(ctx.exception)
            self.assertIn("raft_flow.pt", msg)
            self.assertIn("modal volume get", msg)
            self.assertIn("comma-lab-results", msg)
            self.assertIn("kaggle datasets version", msg)

    def test_supervised_missing_targets_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_asset_root(tmp, raft=True)  # raft present, targets absent
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_supervision_assets("supervised", root)
            msg = str(ctx.exception)
            self.assertIn("posenet_targets.bin", msg)
            self.assertIn("modal volume get", msg)
            self.assertIn("comma-lab-results", msg)
            self.assertIn("kaggle datasets version", msg)


# ---------------------------------------------------------------------------
# _strip_flags
# ---------------------------------------------------------------------------

class TestStripFlags(unittest.TestCase):
    def test_strips_flag_and_value(self) -> None:
        flags = ["--device", "cuda", "--epochs", "100"]
        result = _strip_flags(flags, frozenset({"--device"}))
        self.assertEqual(result, ["--epochs", "100"])

    def test_strips_multiple_flags(self) -> None:
        flags = ["--a", "1", "--b", "2", "--c", "3"]
        result = _strip_flags(flags, frozenset({"--a", "--c"}))
        self.assertEqual(result, ["--b", "2"])

    def test_preserves_unlisted_flags(self) -> None:
        flags = ["--epochs", "100", "--lr", "0.001"]
        result = _strip_flags(flags, frozenset({"--device"}))
        self.assertEqual(result, flags)

    def test_empty_flags(self) -> None:
        result = _strip_flags([], frozenset({"--device"}))
        self.assertEqual(result, [])

    def test_consecutive_strip_targets(self) -> None:
        flags = ["--raft-flow-path", "/x/flow.pt", "--pose-targets-path", "/x/targets.bin"]
        result = _strip_flags(flags, frozenset({"--raft-flow-path", "--pose-targets-path"}))
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------

class TestSaveManifest(unittest.TestCase):
    def test_manifest_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp)
            cmd = [sys.executable, "train.py", "--epochs", "100"]
            path = save_manifest("supervised", cmd, working_dir=working_dir)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["variant"], "supervised")
            self.assertEqual(data["provider"], "kaggle")
            self.assertEqual(data["full_command"], cmd)
            self.assertIn("started_at", data)

    def test_manifest_filename_includes_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_manifest("raft_only", [], working_dir=Path(tmp))
            self.assertIn("raft_only", path.name)

    def test_script_path_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "train.py"
            path = save_manifest("base", [], working_dir=Path(tmp), script_path=script)
            data = json.loads(path.read_text())
            self.assertEqual(data["script"], str(script))


# ---------------------------------------------------------------------------
# build_kaggle_command — structure checks (no actual tac import needed for these)
# ---------------------------------------------------------------------------

class TestBuildKaggleCommand(unittest.TestCase):
    def _make_assets(self, tmp: str, variant: str) -> Path:
        root = Path(tmp)
        if variant in ("supervised", "raft_only"):
            (root / "raft_flow.pt").touch()
        if variant == "supervised":
            (root / "posenet_targets.bin").touch()
        return root

    def _mock_build_flags(self, variant: str = "base", resume_from: object = None, **_kw: object) -> list[str]:
        """Minimal stand-in for tac.deploy.deploy_config.build_flags.

        Returns bare flags only — no script path prepended.
        build_kaggle_command must NOT pass provider_script_path to build_flags;
        this mock enforces that contract by not accepting or returning a script path.
        """
        return [
            "--device", "cuda",                          # BASE_FLAGS device (will be stripped)
            "--max-hours", "5.5",                        # BASE_FLAGS max-hours (will be stripped)
            "--raft-flow-path", "/results/flow.pt",      # VARIANT_FLAGS (will be stripped)
            "--pose-targets-path", "/results/targets.bin",  # VARIANT_FLAGS (will be stripped)
            "--epochs", "20000",
        ]

    def _run_build(self, variant: str, tmp: str) -> list[str]:
        from tac.deploy.kaggle.runner import build_kaggle_command
        script = Path(tmp) / "train.py"
        script.touch()
        asset_root = self._make_assets(tmp, variant)
        with mock.patch("tac.deploy.deploy_config.build_flags", side_effect=self._mock_build_flags), \
             mock.patch("tac.deploy.deploy_config.ALL_VARIANTS", ["base", "supervised", "raft_only"]):
            return build_kaggle_command(variant, script, asset_root)

    def test_command_starts_with_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._run_build("base", tmp)
            self.assertEqual(cmd[0], sys.executable)

    def test_command_includes_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "train.py"
            script.touch()
            cmd = self._run_build("base", tmp)
            self.assertEqual(cmd[1], str(script))

    def test_modal_paths_stripped_kaggle_paths_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._run_build("base", tmp)
            joined = " ".join(cmd)
            self.assertNotIn("/results/", joined)
            self.assertIn("--device", joined)
            self.assertIn("cuda", joined)
            self.assertIn("--max-hours", joined)
            self.assertIn("8.5", joined)

    def test_supervised_injects_asset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._run_build("supervised", tmp)
            joined = " ".join(cmd)
            self.assertIn("--raft-flow-path", joined)
            self.assertIn("raft_flow.pt", joined)
            self.assertIn("--pose-targets-path", joined)
            self.assertIn("posenet_targets.bin", joined)

    def test_raft_only_injects_raft_not_posenet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._run_build("raft_only", tmp)
            joined = " ".join(cmd)
            self.assertIn("--raft-flow-path", joined)
            self.assertNotIn("--pose-targets-path", joined)

    def test_invalid_variant_raises(self) -> None:
        from tac.deploy.kaggle.runner import build_kaggle_command
        with self.assertRaises(ValueError) as ctx:
            with tempfile.TemporaryDirectory() as tmp:
                build_kaggle_command("nonexistent", Path(tmp) / "t.py", Path(tmp))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_build_flags_never_receives_provider_script_path(self) -> None:
        """build_flags must NOT be called with provider_script_path.

        If it were, build_flags would return ["python", script, ...flags...] and
        the command would double the script path: [sys.executable, script, "python", script, ...].
        """
        from tac.deploy.kaggle.runner import build_kaggle_command
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "train.py"
            script.touch()

            call_kwargs: dict = {}

            def capture_build_flags(**kwargs: object) -> list[str]:
                call_kwargs.update(kwargs)
                return ["--epochs", "20000"]

            with mock.patch("tac.deploy.deploy_config.build_flags", side_effect=capture_build_flags), \
                 mock.patch("tac.deploy.deploy_config.ALL_VARIANTS", ["base", "supervised", "raft_only"]):
                build_kaggle_command("base", script, Path(tmp))

            self.assertNotIn(
                "provider_script_path", call_kwargs,
                "build_flags was called with provider_script_path — this causes a double-script bug"
            )

    def test_command_has_no_double_script(self) -> None:
        """Final command must contain the script exactly once (no double-script from build_flags)."""
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._run_build("base", tmp)
            script = str(Path(tmp) / "train.py")
            occurrences = cmd.count(script)
            self.assertEqual(occurrences, 1, f"script appears {occurrences}x in command: {cmd}")


if __name__ == "__main__":
    unittest.main()
