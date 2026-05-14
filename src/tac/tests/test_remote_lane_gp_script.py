# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_gp_gaussian_process_pose.sh"


def _script_text() -> str:
    return SCRIPT.read_text()


def test_script_has_set_euo_pipefail() -> None:
    assert "set -euo pipefail" in _script_text()


def test_no_shell_zip() -> None:
    script = _script_text()
    executable_lines = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("#")
    )
    assert re.search(r"\bzip\b", executable_lines) is None
    assert "zipfile.ZipFile" in script


def test_no_mps_fallback() -> None:
    executable_lines = "\n".join(
        line for line in _script_text().splitlines() if not line.strip().startswith("#")
    )
    assert "mps" not in executable_lines.lower()


def test_stage_labels_present() -> None:
    script = _script_text()
    for stage in ("Stage 0", "Stage 1", "Stage 2", "Stage 3"):
        assert stage in script
