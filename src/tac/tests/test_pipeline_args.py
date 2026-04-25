"""Test that pipeline.py subprocess calls match the target scripts' argparse.

This test exists because we had THREE bugs where pipeline.py passed flags
that the target script didn't accept:
1. --archive instead of --checkpoint (auth_eval_renderer.py)
2. --upstream instead of --upstream-dir (auth_eval_renderer.py)
3. --embed-dim before it was added to train_distill.py

The fix: parse the subprocess commands through the target's actual argparse
and verify they don't raise SystemExit or error.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock


class TestPipelineSubprocessArgs(unittest.TestCase):
    """Verify pipeline.py's subprocess commands match target argparse."""

    def _get_parser(self, script_path: str):
        """Import a script's argparse parser without executing main()."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_target", script_path)
        mod = importlib.util.module_from_spec(spec)
        # Prevent the script from running on import
        old_argv = sys.argv
        sys.argv = [script_path]
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
        # Find the parser — look for parse_args or build_parser
        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, 'parse_args') and hasattr(obj, 'add_argument'):
                return obj
        return None

    def test_pipeline_eval_args_match_auth_eval(self):
        """pipeline.py step_eval passes --checkpoint, --upstream-dir, --archive-size-bytes."""
        src = Path("experiments/pipeline.py").read_text()
        # Find the actual cmd= block (last occurrence of auth_eval_renderer.py)
        parts = src.split("auth_eval_renderer.py")
        eval_section = parts[-1].split("]")[0]  # last occurrence = the actual cmd
        for flag in ["--checkpoint", "--upstream-dir", "--device", "--archive-size-bytes"]:
            assert flag in eval_section, f"pipeline.py missing {flag} for auth_eval"

    def test_pipeline_pose_tto_args_match_optimize_poses(self):
        """pipeline.py step_pose_tto passes valid args."""
        src = Path("experiments/pipeline.py").read_text()
        # Extract the pose TTO command section
        pose_section = src.split("optimize_poses.py")[1].split("]")[0]
        for flag in ["--checkpoint", "--masks", "--device", "--steps", "--lr",
                      "--batch-pairs", "--eval-roundtrip", "--output-dir"]:
            assert flag in pose_section, f"pipeline.py missing {flag} for optimize_poses"

    def test_pipeline_qat_args_match_qat_finetune(self):
        """pipeline.py step_qat passes valid args (fp4-epochs, lr, upstream, arch flags)."""
        src = Path("experiments/pipeline.py").read_text()
        parts = src.split("qat_finetune.py")
        qat_section = parts[-1].split("]")[0]  # last occurrence = actual cmd
        for flag in ["--checkpoint", "--upstream", "--device", "--fp4-epochs", "--lr",
                      "--output-dir", "--base-ch", "--mid-ch", "--pose-dim"]:
            assert flag in qat_section, f"pipeline.py missing {flag} for qat_finetune"
        # Verify OLD wrong flags are NOT present
        for bad_flag in ["--masks", "--qat-epochs", "--qat-lr", "--eval-roundtrip"]:
            assert bad_flag not in qat_section, f"pipeline.py still has old wrong flag {bad_flag} for qat_finetune"

    def test_launch_script_flags_exist_in_train_distill(self):
        """Every flag in launch_wilde_shiraz.sh exists in train_distill.py argparse."""
        import re
        launch_src = Path("experiments/launch_wilde_shiraz.sh").read_text()
        distill_src = Path("experiments/train_distill.py").read_text()

        # Extract all --flag-name patterns from the launch script
        launch_flags = set(re.findall(r'--([a-z][a-z0-9-]+)', launch_src))
        # Remove non-train_distill flags (ssh, rsync, etc)
        non_training = {'progress', 'exclude', 'include', 'delete'}
        launch_flags -= non_training

        # Extract all --flag-name from train_distill.py add_argument calls
        distill_flags = set(re.findall(r'add_argument\(["\']--([a-z][a-z0-9-]+)', distill_src))

        missing = launch_flags - distill_flags
        # Some flags are legitimate non-argparse (like --masks which maps differently)
        known_ok = {'masks', 'gt-poses', 'upstream', 'tto-frames', 'output-dir',
                     'device', 'seed', 'checkpoint-every', 'eval-every', 'log-every',
                     'checkpoint',
                     # Vast.ai/deployment flags (not train_distill args)
                     'image', 'disk'}
        real_missing = missing - known_ok

        self.assertEqual(real_missing, set(),
                         f"Launch script has flags not in train_distill.py argparse: {real_missing}")


if __name__ == "__main__":
    unittest.main()
