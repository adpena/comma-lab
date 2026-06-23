# SPDX-License-Identifier: MIT
"""Catalog #389 — check_no_orphan_prone_daemon_launch guard tests.

Positive: catches the `nohup bash -c '... | tee L > /dev/null' & disown`
orphan signature. Negative: allows spawn_durable_daemon.py usage + short
backgrounded commands + run_in_background. Waiver-respect: same-line
`# ORPHAN_PRONE_DAEMON_LAUNCH_OK:<rationale>` (placeholder rejected). Plus the
live-repo regression guard (count must stay 0).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_389_line_is_orphan_prone,
    _check_389_rationale_is_placeholder,
    check_no_orphan_prone_daemon_launch,
)


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Line-level signature
# --------------------------------------------------------------------------
def test_line_matches_full_orphan_signature():
    line = "nohup bash -c 'python worker.py --watch | tee log.txt > /dev/null' & disown"
    assert _check_389_line_is_orphan_prone(line) is True


def test_line_matches_double_quoted_bash_c():
    line = 'nohup bash -c "python worker.py | tee log.txt" & disown'
    assert _check_389_line_is_orphan_prone(line) is True


def test_line_matches_trailing_ampersand_no_disown():
    line = "nohup bash -c 'python w.py | tee log.txt > /dev/null' &"
    assert _check_389_line_is_orphan_prone(line) is True


def test_line_no_nohup_does_not_match():
    line = "bash -c 'python w.py | tee log.txt' & disown"
    assert _check_389_line_is_orphan_prone(line) is False


def test_line_no_tee_does_not_match():
    line = "nohup bash -c 'python w.py > log.txt' & disown"
    assert _check_389_line_is_orphan_prone(line) is False


def test_line_no_background_does_not_match():
    line = "nohup bash -c 'python w.py | tee log.txt > /dev/null'"
    assert _check_389_line_is_orphan_prone(line) is False


def test_line_short_backgrounded_command_does_not_match():
    line = "python quick_oneshot.py --flag &"
    assert _check_389_line_is_orphan_prone(line) is False


def test_line_plain_nohup_no_bash_c_does_not_match():
    line = "nohup python worker.py | tee log.txt > /dev/null & disown"
    assert _check_389_line_is_orphan_prone(line) is False


# --------------------------------------------------------------------------
# Placeholder rejection
# --------------------------------------------------------------------------
def test_placeholder_rationales_rejected():
    assert _check_389_rationale_is_placeholder("") is True
    assert _check_389_rationale_is_placeholder("<rationale>") is True
    assert _check_389_rationale_is_placeholder("<reason>") is True
    assert _check_389_rationale_is_placeholder("  <reason>  ") is True


def test_real_rationale_not_placeholder():
    assert _check_389_rationale_is_placeholder("legacy historical recipe only") is False


# --------------------------------------------------------------------------
# File-level scanner: positive
# --------------------------------------------------------------------------
def test_scanner_catches_orphan_launch(tmp_path):
    _write(
        tmp_path,
        "tools/bad_daemon.sh",
        "#!/bin/bash\n"
        "nohup bash -c 'python serve.py --watch | tee out.log > /dev/null' & disown\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert len(v) == 1
    assert "tools/bad_daemon.sh:2" in v[0]
    assert "Catalog #389" in v[0]


def test_scanner_catches_in_python_file(tmp_path):
    _write(
        tmp_path,
        "scripts/launch.py",
        "import os\n"
        "os.system(\"nohup bash -c 'python r.py | tee r.log > /dev/null' & disown\")\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert len(v) == 1


def test_scanner_strict_raises(tmp_path):
    _write(
        tmp_path,
        "experiments/d.sh",
        "nohup bash -c 'python r.py --watch | tee r.log > /dev/null' & disown\n",
    )
    with pytest.raises(PreflightError):
        check_no_orphan_prone_daemon_launch(repo_root=tmp_path, strict=True)


# --------------------------------------------------------------------------
# File-level scanner: negative
# --------------------------------------------------------------------------
def test_scanner_allows_spawn_durable_daemon_usage(tmp_path):
    _write(
        tmp_path,
        "tools/good_launch.sh",
        "#!/bin/bash\n"
        ".venv/bin/python tools/spawn_durable_daemon.py --log d.log --label svc "
        "-- python serve.py --watch\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


def test_scanner_allows_short_backgrounded_command(tmp_path):
    _write(
        tmp_path,
        "scripts/quick.sh",
        "#!/bin/bash\npython oneshot.py --flag &\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


def test_scanner_allows_nohup_without_bash_c(tmp_path):
    _write(
        tmp_path,
        "scripts/n.sh",
        "nohup python worker.py | tee log.txt > /dev/null & disown\n",
    )
    # No bash -c wrapper -> worker IS the backgrounded process (no orphan risk);
    # not the targeted signature.
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


def test_scanner_skips_the_canonical_launcher_itself(tmp_path):
    # The canonical launcher documents the orphan signature in its docstring.
    _write(
        tmp_path,
        "tools/spawn_durable_daemon.py",
        "# docstring example: nohup bash -c '... | tee LOG > /dev/null' & disown\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


def test_scanner_skips_vendored_paths(tmp_path):
    _write(
        tmp_path,
        "experiments/results/public_pr99_intake_x/source/run.sh",
        "nohup bash -c 'python r.py | tee r.log > /dev/null' & disown\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


# --------------------------------------------------------------------------
# Waiver respect
# --------------------------------------------------------------------------
def test_scanner_respects_real_waiver(tmp_path):
    _write(
        tmp_path,
        "tools/legacy.sh",
        "nohup bash -c 'python r.py | tee r.log > /dev/null' & disown "
        "# ORPHAN_PRONE_DAEMON_LAUNCH_OK:legacy historical recipe, never re-run\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert v == []


def test_scanner_rejects_placeholder_waiver(tmp_path):
    _write(
        tmp_path,
        "tools/legacy.sh",
        "nohup bash -c 'python r.py | tee r.log > /dev/null' & disown "
        "# ORPHAN_PRONE_DAEMON_LAUNCH_OK:<rationale>\n",
    )
    v = check_no_orphan_prone_daemon_launch(repo_root=tmp_path)
    assert len(v) == 1


# --------------------------------------------------------------------------
# Live-repo regression guard
# --------------------------------------------------------------------------
def test_live_repo_count_is_zero():
    """The strict-flipped gate must stay at 0 in the real repo."""
    v = check_no_orphan_prone_daemon_launch()
    assert v == [], f"Catalog #389 live count drifted: {v[:3]}"
