from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "feedback_loop_sweep.py"


def test_feedback_loop_sweep_refuses_retired_paid_dispatch_path(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--output-dir",
            str(tmp_path / "feedback"),
            "--allow-paid-dispatch",
            "--max-cycles",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "paid dispatch is retired" in proc.stderr
    assert "promote_optimizer_candidate_for_exact_eval.py" in proc.stderr
