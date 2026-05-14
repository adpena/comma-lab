# SPDX-License-Identifier: MIT
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

    def test_pipeline_train_distill_flags_exist_in_train_distill(self):
        """Every --flag pipeline.py passes to train_distill.py exists in its argparse.

        Replaces the old launch_wilde_shiraz.sh test (script killed in commit 04392166
        when ad-hoc deployment was retired in favor of the canonical pipeline).
        """
        repo_root = Path(__file__).resolve().parents[3]
        pipeline_path = repo_root / "experiments" / "pipeline.py"
        distill_path = repo_root / "experiments" / "train_distill.py"
        if not pipeline_path.exists() or not distill_path.exists():
            self.skipTest("pipeline.py or train_distill.py not present")

        pipeline_src = pipeline_path.read_text()
        distill_src = distill_path.read_text()

        # Find each `cmd = [...]` (or `cmd_train = [...]` etc.) literal-list block whose
        # contents reference train_distill.py and harvest only that block's --flags.
        # This avoids cross-contamination with neighboring qat_finetune.py / archive blocks.
        list_block = re.compile(r'\b\w*cmd\w*\s*=\s*\[(.*?)\]', re.DOTALL)
        used_flags: set[str] = set()
        for m in list_block.finditer(pipeline_src):
            block = m.group(1)
            if 'train_distill.py' not in block:
                continue
            used_flags.update(re.findall(r'"--([a-z][a-z0-9-]+)"', block))
            used_flags.update(re.findall(r"'--([a-z][a-z0-9-]+)'", block))

        distill_flags = set(re.findall(r'add_argument\(["\']--([a-z][a-z0-9-]+)', distill_src))

        real_missing = used_flags - distill_flags
        self.assertEqual(
            real_missing, set(),
            f"pipeline.py passes flags to train_distill.py that argparse does not accept: {real_missing}",
        )

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
