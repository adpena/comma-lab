from __future__ import annotations

import os
import time
import textwrap
from pathlib import Path

from tac import preflight


REPO = Path(__file__).resolve().parents[3]


def _write_remote_lane(root: Path, name: str, body: str) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / name
    path.write_text(textwrap.dedent(body).lstrip())
    # Scope of Check 43 is scripts added/modified after 2026-04-29.
    future_ts = time.mktime((2026, 4, 30, 12, 0, 0, 0, 0, -1))
    os.utime(path, (future_ts, future_ts))
    return path


def test_lane_methodology_doc_exists_and_mentions_controlled_baseline_lane() -> None:
    doc = REPO / "docs" / "lane_methodology.md"

    assert doc.exists()
    assert "controlled_baseline_lane" in doc.read_text()


def test_controlled_baseline_preflight_check_exists() -> None:
    assert hasattr(preflight, "check_remote_lane_scripts_have_controlled_baseline")


def test_new_remote_lane_without_controlled_baseline_warns(tmp_path: Path) -> None:
    _write_remote_lane(
        tmp_path,
        "remote_lane_future_without_control.sh",
        """
        #!/bin/bash
        set -euo pipefail
        echo run
        """,
    )

    warnings = preflight.check_remote_lane_scripts_have_controlled_baseline(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(warnings) == 1
    assert "controlled_baseline" in warnings[0]


def test_new_remote_lane_with_controlled_baseline_passes(tmp_path: Path) -> None:
    _write_remote_lane(
        tmp_path,
        "remote_lane_future_with_control.sh",
        """
        #!/bin/bash
        set -euo pipefail
        controlled_baseline="lane_future_minimal_change"
        echo "$controlled_baseline"
        """,
    )

    warnings = preflight.check_remote_lane_scripts_have_controlled_baseline(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert warnings == []
