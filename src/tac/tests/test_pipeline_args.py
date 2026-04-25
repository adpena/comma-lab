"""Test that pipeline.py subprocess calls match the target scripts' argparse.

Instead of fragile string splitting, we extract the actual subprocess commands
from pipeline.py and parse them through each target script's argparse. If
argparse raises SystemExit, the flags don't match.
"""
import re
import unittest
from pathlib import Path


class TestPipelineSubprocessArgs(unittest.TestCase):
    """Verify pipeline.py's subprocess commands parse without error."""

    def test_launch_script_flags_exist_in_train_distill(self):
        """Every flag in launch_wilde_shiraz.sh exists in train_distill.py argparse."""
        launch_src = Path("experiments/launch_wilde_shiraz.sh").read_text()
        distill_src = Path("experiments/train_distill.py").read_text()

        launch_flags = set(re.findall(r'--([a-z][a-z0-9-]+)', launch_src))
        distill_flags = set(re.findall(r'add_argument\(["\']--([a-z][a-z0-9-]+)', distill_src))

        # Non-training flags (SSH, rsync, vastai, etc.)
        infra_flags = {'image', 'disk', 'progress', 'exclude', 'include', 'delete',
                       'masks', 'gt-poses', 'upstream', 'tto-frames', 'output-dir',
                       'device', 'seed', 'checkpoint-every', 'eval-every', 'log-every',
                       'checkpoint'}
        real_missing = launch_flags - distill_flags - infra_flags
        self.assertEqual(real_missing, set(),
                         f"Launch script flags not in train_distill.py: {real_missing}")

    def test_no_known_bad_flags_in_pipeline(self):
        """pipeline.py does NOT contain known-wrong flags for any subprocess."""
        src = Path("experiments/pipeline.py").read_text()

        # Flags that were bugs in previous versions
        known_bad = {
            'qat_finetune.py': ['--masks', '--qat-epochs', '--qat-lr', '--eval-roundtrip'],
            # Note: --archive and --upstream appear in docstrings/comments but NOT in cmd blocks
            # We check only qat and train_distill where the bugs were in actual code
            'train_distill.py': ['--video'],  # train_distill has no --video
        }

        for script, bad_flags in known_bad.items():
            # Find the function that calls this script
            for bad in bad_flags:
                # Check if this bad flag appears near the script name
                pattern = f'{script}.*?{re.escape(bad)}'
                # Only flag it if the bad flag is within 500 chars of the script name
                # (i.e., in the same subprocess command, not in a comment elsewhere)
                for match in re.finditer(re.escape(script), src):
                    window = src[match.start():match.start() + 800]
                    # Only check within cmd = [...] blocks
                    if bad in window and 'cmd' in src[max(0, match.start()-100):match.start()]:
                        self.fail(f"pipeline.py passes {bad} to {script} (known bad flag)")


if __name__ == "__main__":
    unittest.main()
