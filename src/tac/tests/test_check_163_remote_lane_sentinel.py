# SPDX-License-Identifier: MIT
"""Tests for Catalog #163 — remote_lane sentinel sourcing rule.

Bug class: WWW4 dispatch (Modal A100 fc-01KREXK209TRX7ED5ZRVXHY1VT,
2026-05-12) crashed because ``scripts/remote_lane_substrate_sane_hnerv.sh``
sourced ``scripts/remote_archive_only_eval.sh`` without prepending
``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1``. The sourced main flow ran,
hit "FATAL: archive missing", and exited before the calling lane's
stages started.

These tests pin: positive (catches bare source), negative (accepts
sentinel + waiver + process-sub forms), waiver-respect, edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap,
)


def _make_repo_with(scripts: dict[str, str], tmp_path: Path) -> Path:
    """Create a fake repo with scripts/ dir containing the given files."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name, content in scripts.items():
        (scripts_dir / name).write_text(content)
    return tmp_path


def test_bare_source_without_sentinel_is_caught(tmp_path):
    """Bare ``source ... remote_archive_only_eval.sh`` is a violation."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(vlist) == 1
    assert "remote_lane_test.sh" in vlist[0]
    assert "sentinel" in vlist[0].lower()


def test_inline_sentinel_is_accepted(tmp_path):
    """``VAR=1 source ...`` inline assignment is OK."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 "
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_export_sentinel_in_preceding_lines_is_accepted(tmp_path):
    """``export VAR=1`` on a previous line is OK if within 5 lines."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "export REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1\n"
                "# some comment\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_sentinel_too_far_back_is_rejected(tmp_path):
    """Sentinel more than 5 lines back does NOT save the source line."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "export REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1\n"
                "echo 1\n"
                "echo 2\n"
                "echo 3\n"
                "echo 4\n"
                "echo 5\n"
                "echo 6\n"
                "echo 7\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(vlist) == 1


def test_process_substitution_is_skipped(tmp_path):
    """``source <(grep ...)`` extracts only function body, doesn't run main flow."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "source <(grep -A 30 '^bootstrap_runtime_deps()' "
                '"$WORKSPACE/scripts/remote_archive_only_eval.sh")\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_process_substitution_awk_form_is_skipped(tmp_path):
    """``source <(awk ...)`` is the same pattern, also skipped."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "source <(awk '/^bootstrap_runtime_deps\\(\\)/,/^}/' "
                '"$WORKSPACE/scripts/remote_archive_only_eval.sh")\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_child_shell_invocation_is_skipped(tmp_path):
    """``bash ... script.sh ...`` runs in child shell, main flow OK to exit."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'bash "$WORKSPACE/scripts/remote_archive_only_eval.sh" --archive foo\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_same_line_waiver_is_respected(tmp_path):
    """``# REMOTE_LANE_FULL_PIPELINE_OK:<reason>`` waiver bypasses the check."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"  '
                "# REMOTE_LANE_FULL_PIPELINE_OK:legacy lane runs full main flow\n"
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_dot_form_source_is_caught(tmp_path):
    """``. ./script.sh`` is equivalent to ``source ./script.sh``."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                '. "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(vlist) == 1


def test_dot_form_with_sentinel_is_accepted(tmp_path):
    """``VAR=1 . ./script.sh`` is OK."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 . "$WORKSPACE/scripts/'
                'remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_strict_mode_raises_on_violation(tmp_path):
    """``strict=True`` must raise PreflightError on any violation."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    with pytest.raises(PreflightError):
        check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
            repo_root=repo, strict=True, verbose=False
        )


def test_no_scripts_dir_returns_empty(tmp_path):
    """Empty repo (no scripts dir) returns 0 violations, no exception."""
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert vlist == []


def test_unrelated_script_not_scanned(tmp_path):
    """A non-``remote_lane_*`` script (e.g. ``setup.sh``) is not scanned."""
    repo = _make_repo_with(
        {
            "setup.sh": (
                "#!/usr/bin/env bash\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_other_source_targets_ignored(tmp_path):
    """``source other_file.sh`` (not the canonical bootstrap) is not flagged."""
    repo = _make_repo_with(
        {
            "remote_lane_test.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'source "$WORKSPACE/scripts/some_other_helper.sh"\n'
            )
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert vlist == []


def test_live_repo_count_is_zero():
    """Live repo state: WWW4's same-day fix already drove violations to 0.

    This pins the strict-flip at the moment of landing per CLAUDE.md
    "Strict-flip atomicity rule".
    """
    repo_root = Path(__file__).resolve().parents[3]
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert vlist == [], (
        "Live repo has remote_lane_*.sh files missing the WWW4 sentinel:\n  "
        + "\n  ".join(vlist)
    )


def test_multiple_violations_reported(tmp_path):
    """Multiple bad scripts produce multiple violations."""
    repo = _make_repo_with(
        {
            "remote_lane_a.sh": (
                "#!/usr/bin/env bash\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            ),
            "remote_lane_b.sh": (
                "#!/usr/bin/env bash\n"
                'source "$WORKSPACE/scripts/remote_archive_only_eval.sh"\n'
            ),
        },
        tmp_path,
    )
    vlist = check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(vlist) == 2
