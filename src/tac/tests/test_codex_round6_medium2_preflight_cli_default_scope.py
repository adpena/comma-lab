"""Tests for codex round 6 MEDIUM 2 fix: preflight CLI default `--scope`.

Catalog #145 — refuses any change to ``parser.add_argument("--scope",
default=...)`` that sets a non-``all`` default. The DX ratchet defaulted
to ``dev`` and silently weakened the green CLI; the fix restores ``all``.

Bug class: codex round 6 MEDIUM 2 (2026-05-09). Memory:
feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# CLI default scope is `all`
# --------------------------------------------------------------------------


def test_preflight_cli_scope_default_is_all():
    """The committed CLI must default to `all`."""
    preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
    text = preflight_path.read_text()
    tree = ast.parse(text)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "--scope"):
            continue
        # Find default kwarg
        for kw in node.keywords:
            if kw.arg == "default":
                assert isinstance(kw.value, ast.Constant)
                assert kw.value.value == "all", (
                    f"--scope default is {kw.value.value!r}, expected 'all' "
                    f"(catalog #145; see codex round 6 MEDIUM 2)"
                )
                found = True
                break
    assert found, "preflight CLI must define --scope flag with default kwarg"


def test_preflight_cli_dev_emits_warning():
    """When --scope dev is explicitly used, a warning must be emitted."""
    preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
    text = preflight_path.read_text()
    # The fix added a "WARNING: --scope dev" emission. Check it's present.
    assert "WARNING: --scope dev" in text, (
        "fix #145 must emit a warning when --scope dev is used"
    )


# --------------------------------------------------------------------------
# Preflight check #145 STRICT
# --------------------------------------------------------------------------


def test_preflight_check_145_passes_with_zero_violations():
    from tac.preflight import check_preflight_cli_default_scope_is_all

    violations = check_preflight_cli_default_scope_is_all(
        verbose=False, strict=False,
    )
    assert violations == [], (
        f"Catalog #145 preflight should be at 0; got {len(violations)}:\n  "
        + "\n  ".join(violations[:5])
    )


def test_preflight_check_145_fires_on_dev_default(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_all

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='dev', choices=('dev', 'release'))\n"
    )

    violations = check_preflight_cli_default_scope_is_all(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert len(violations) >= 1
    assert "dev" in str(violations) and "default" in str(violations)


def test_preflight_check_145_accepts_all_default(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_all

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='all')\n"
    )

    violations = check_preflight_cli_default_scope_is_all(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_accepts_same_line_waiver(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_all

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='dev')  # PREFLIGHT_CLI_SCOPE_DEFAULT_OK:test-fixture\n"
    )

    violations = check_preflight_cli_default_scope_is_all(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_accepts_multiline_waiver(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_all

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument(\n"
        "    '--scope',\n"
        "    default='dev',\n"
        "    help='dev only'  # PREFLIGHT_CLI_SCOPE_DEFAULT_OK:multiline-waiver\n"
        ")\n"
    )

    violations = check_preflight_cli_default_scope_is_all(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_strict_mode_raises():
    from tac.preflight import (
        PreflightError,
        check_preflight_cli_default_scope_is_all,
    )

    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td)
        target = fake_repo / "src" / "tac" / "preflight.py"
        target.parent.mkdir(parents=True)
        target.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--scope', default='release')\n"
        )

        with pytest.raises(PreflightError):
            check_preflight_cli_default_scope_is_all(
                repo_root=fake_repo, verbose=False, strict=True,
            )


def test_preflight_check_145_no_scope_flag_is_clean(tmp_path: Path):
    """A CLI with no --scope flag at all is not in scope."""
    from tac.preflight import check_preflight_cli_default_scope_is_all

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--profile', default='proven_baseline')\n"
    )

    violations = check_preflight_cli_default_scope_is_all(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []
