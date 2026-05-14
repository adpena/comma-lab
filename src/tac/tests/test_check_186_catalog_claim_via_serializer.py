# SPDX-License-Identifier: MIT
"""Catalog #186 — `check_catalog_claim_committed_via_serializer` tests.

The check refuses bare ``tools/claim_catalog_number.py claim`` invocations
in subagent / fan-out contexts. Per CANON-1.E hardening, the canonical
shape MUST be::

    python tools/claim_catalog_number.py claim \\
        --commit-via-serializer --reason "<purpose>"

Bug class anchor: R7-1 recurrence 2026-05-13 08:16Z + 08:37Z. The 2026-05-08
fcntl lock landed in ``tools/claim_catalog_number.py`` was bypassed by
working-tree rollback. Pid 40139 claimed #183, pid 58394 re-claimed #183
21 minutes later. Both invocations were bare. The CANON-1.E
``--commit-via-serializer`` flag closes the race by making the increment
git-transactional; this gate refuses the bare invocation surface.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    _CHECK_186_BARE_WAIVER_TOKEN,
    _CHECK_186_CANONICAL_TOOL_RELPATH,
    _CHECK_186_FILE_LEVEL_WAIVER_TOKEN,
    PreflightError,
    _check_186_detect_bare_claim_lines,
    _check_186_is_scoped_path,
    check_catalog_claim_committed_via_serializer,
)

# ─────────────────────────────────────────────────────────────────────────
# Phase 1: scope-filter unit tests
# ─────────────────────────────────────────────────────────────────────────


def test_canonical_tool_excluded():
    """The canonical tool itself is always out of scope."""
    assert _check_186_is_scoped_path(_CHECK_186_CANONICAL_TOOL_RELPATH) is False


def test_test_files_excluded():
    """test_*.py files are out of scope (they exercise the CLI)."""
    assert _check_186_is_scoped_path("src/tac/tests/test_foo.py") is False
    assert _check_186_is_scoped_path("tools/test_helper.py") is False


def test_intake_clones_excluded():
    """Vendored intake clones are out of scope."""
    assert _check_186_is_scoped_path(
        "experiments/results/public_pr95_intake_codex/foo.py"
    ) is False


def test_build_artifacts_excluded():
    """experiments/results/ build artifacts are out of scope."""
    assert _check_186_is_scoped_path(
        "experiments/results/lane_foo_20260513/script.sh"
    ) is False


def test_reports_raw_excluded():
    """reports/raw/ (kaggle ingested snapshots) is out of scope."""
    assert _check_186_is_scoped_path("reports/raw/kaggle_pr106/foo.py") is False


def test_oss_export_excluded():
    """OSS export mirrors are out of scope."""
    assert _check_186_is_scoped_path(".omx/oss_export/foo.py") is False


def test_tools_python_in_scope():
    """tools/*.py files are in scope."""
    assert _check_186_is_scoped_path("tools/some_dispatcher.py") is True


def test_scripts_shell_in_scope():
    """scripts/*.sh files are in scope."""
    assert _check_186_is_scoped_path("scripts/remote_lane_foo.sh") is True


def test_src_tac_in_scope():
    """src/tac/*.py is in scope."""
    assert _check_186_is_scoped_path("src/tac/some_module.py") is True


def test_top_level_files_out_of_scope():
    """Top-level files (CLAUDE.md, etc.) are out of scope."""
    assert _check_186_is_scoped_path("CLAUDE.md") is False
    assert _check_186_is_scoped_path("README.md") is False


def test_unrelated_suffix_out_of_scope():
    """.md / .json / .yaml files are out of scope."""
    assert _check_186_is_scoped_path("tools/foo.md") is False
    assert _check_186_is_scoped_path("scripts/config.yaml") is False


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: line-detector unit tests
# ─────────────────────────────────────────────────────────────────────────


def test_detect_bare_python_subprocess_call():
    """Bare python subprocess.run([..., 'claim', ...]) is flagged."""
    text = (
        'import subprocess\n'
        'subprocess.run(["python", "tools/claim_catalog_number.py", '
        '"claim"], check=True)\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert len(flagged) == 1
    assert flagged[0][0] == 2  # line 2


def test_detect_bare_shell_invocation():
    """Bare shell `python tools/claim_catalog_number.py claim` is flagged."""
    text = (
        '#!/bin/bash\n'
        'NEXT=$(python tools/claim_catalog_number.py claim)\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert len(flagged) == 1
    assert flagged[0][0] == 2


def test_canonical_invocation_not_flagged():
    """Canonical --commit-via-serializer form is NOT flagged."""
    text = (
        'NEXT=$(python tools/claim_catalog_number.py claim '
        '--commit-via-serializer --reason "WAVE-X: foo")\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_canonical_python_subprocess_not_flagged():
    """Canonical subprocess invocation with --commit-via-serializer is OK."""
    text = (
        'subprocess.run([\n'
        '    "python", "tools/claim_catalog_number.py", "claim",\n'
        '    "--commit-via-serializer", "--reason", "FIX-WAVE-7"\n'
        '], check=True)\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_multiline_bare_python_subprocess_is_flagged():
    """Bare multi-line subprocess argv calls are the Catalog #186 bug class."""
    text = (
        'subprocess.run([\n'
        '    "python", "tools/claim_catalog_number.py",\n'
        '    "claim",\n'
        '    "--reason", "FIX-WAVE-7"\n'
        '], check=True)\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert len(flagged) == 1
    assert flagged[0][0] == 2
    assert "--commit-via-serializer" not in flagged[0][1]


def test_peek_subcommand_not_flagged():
    """`peek` subcommand is not the bug class (read-only)."""
    text = (
        'NEXT=$(python tools/claim_catalog_number.py peek)\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_set_subcommand_not_flagged():
    """`set` subcommand is operator-only init, not the bug class."""
    text = (
        'python tools/claim_catalog_number.py set --value 200\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_same_line_waiver_respected():
    """`# CATALOG_CLAIM_BARE_OK:<reason>` exempts a single line."""
    text = (
        'NEXT=$(python tools/claim_catalog_number.py claim)  '
        f'{_CHECK_186_BARE_WAIVER_TOKEN}operator-one-off-housekeeping\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_docstring_only_reference_not_flagged():
    """Docstring lines describing the CLI without invoking it are not flagged."""
    text = (
        '"""\n'
        'Usage:\n'
        '    The bare claim form would be python tools/claim_catalog_number.py claim\n'
        'but you should use --commit-via-serializer instead.\n'
        '"""\n'
    )
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


def test_comment_line_documentation_skipped():
    """Comment-only line that documents the CLI without invocation markers is skipped."""
    text = (
        '# The canonical form is `claim_catalog_number.py claim --commit-via-serializer`\n'
    )
    # This line carries --commit-via-serializer so it's not bare.
    flagged = _check_186_detect_bare_claim_lines(text)
    assert flagged == []


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: end-to-end gate behavior on synthetic repo
# ─────────────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal repo skeleton with the scan-dir tree."""
    for d in ("tools", "scripts", "src/tac", "experiments"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    # The canonical tool is always exempt; create a placeholder.
    (tmp_path / "tools/claim_catalog_number.py").write_text(
        "# Canonical tool - exempt from #186\n"
        "# python tools/claim_catalog_number.py claim\n"
    )
    return tmp_path


def test_clean_repo_no_violations(tmp_path):
    repo = _make_repo(tmp_path)
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_bare_claim_in_tools_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools/bad_dispatcher.py").write_text(
        "import subprocess\n"
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/bad_dispatcher.py" in violations[0]


def test_bare_claim_in_scripts_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "scripts/bad_remote.sh").write_text(
        "#!/bin/bash\n"
        "NEXT=$(python tools/claim_catalog_number.py claim)\n"
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "scripts/bad_remote.sh" in violations[0]


def test_canonical_form_not_flagged_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools/good_dispatcher.py").write_text(
        "import subprocess\n"
        'subprocess.run(["python", "tools/claim_catalog_number.py", '
        '"claim", "--commit-via-serializer", "--reason", "OK"], check=True)\n'
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_same_line_waiver_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "scripts/operator_helper.sh").write_text(
        "#!/bin/bash\n"
        "NEXT=$(python tools/claim_catalog_number.py claim)  "
        f"{_CHECK_186_BARE_WAIVER_TOKEN}operator-one-off\n"
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_file_level_waiver_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools/legacy_dispatcher.py").write_text(
        f"# {_CHECK_186_FILE_LEVEL_WAIVER_TOKEN}legacy-housekeeping-helper\n"
        "import subprocess\n"
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_intake_clone_excluded_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    intake_dir = repo / "experiments/results/public_pr95_intake_codex"
    intake_dir.mkdir(parents=True)
    (intake_dir / "bad_dispatcher.py").write_text(
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    # The intake clone is excluded - no violation despite the bare claim.
    assert violations == []


def test_test_files_excluded_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src/tac").mkdir(parents=True, exist_ok=True)
    (repo / "src/tac/tests").mkdir(parents=True, exist_ok=True)
    (repo / "src/tac/tests/test_claim.py").write_text(
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_strict_mode_raises(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools/bad.py").write_text(
        "import subprocess\n"
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    with pytest.raises(PreflightError):
        check_catalog_claim_committed_via_serializer(
            repo_root=repo, strict=True, verbose=False
        )


def test_multiple_violations_all_collected(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools/bad_one.py").write_text(
        "import subprocess\n"
        'subprocess.run(["python", "tools/claim_catalog_number.py", "claim"])\n'
    )
    (repo / "scripts/bad_two.sh").write_text(
        "#!/bin/bash\n"
        "python tools/claim_catalog_number.py claim\n"
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 2


def test_peek_not_flagged_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "scripts/peek_helper.sh").write_text(
        "#!/bin/bash\n"
        "CURRENT=$(python tools/claim_catalog_number.py peek)\n"
    )
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_strict_zero():
    """Sanity guard: the live repo MUST pass strict mode.

    Per CLAUDE.md Strict-flip atomicity rule, Catalog #186 lands STRICT
    in the same commit batch as the wire-in. This test confirms the
    in-repo live count is 0 at landing time.
    """
    repo = Path(__file__).resolve().parents[3]
    violations = check_catalog_claim_committed_via_serializer(
        repo_root=repo, strict=False, verbose=False
    )
    # The canonical tool itself + tests are excluded by scope filter; any
    # remaining production caller must use --commit-via-serializer or
    # carry an explicit waiver. Live count at landing: 0.
    assert violations == [], (
        "Catalog #186 live count drifted above 0:\n"
        + "\n".join(violations[:10])
    )
