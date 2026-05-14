# SPDX-License-Identifier: MIT
"""Tests for Catalog #145: preflight CLI default `--scope`.

Catalog #145 — refuses any change to ``parser.add_argument("--scope",
default=...)`` that moves the routine CLI away from the bounded ``dev`` gate
or removes the explicit ``all``/``release`` full-surface choices.

Bug class: full-surface release/custody scans as the routine edit-loop
default, which violates the 30s DX budget.
"""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# CLI default scope is bounded `dev`
# --------------------------------------------------------------------------


def test_preflight_cli_scope_default_is_dev():
    """The committed CLI must default to bounded `dev`."""
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
                assert kw.value.value == "dev", (
                    f"--scope default is {kw.value.value!r}, expected 'dev' "
                    f"(catalog #145; bounded preflight default)"
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
        "Catalog #145 must emit a warning when --scope dev is explicitly used"
    )


# --------------------------------------------------------------------------
# Preflight check #145 STRICT
# --------------------------------------------------------------------------


def test_preflight_check_145_passes_with_zero_violations():
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        verbose=False, strict=False,
    )
    assert violations == [], (
        f"Catalog #145 preflight should be at 0; got {len(violations)}:\n  "
        + "\n  ".join(violations[:5])
    )


def test_preflight_check_145_fires_on_all_default(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='all', choices=('dev', 'release', 'all'))\n"
    )

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert len(violations) >= 1
    assert "all" in str(violations) and "default" in str(violations)


def test_preflight_check_145_accepts_dev_default_with_full_choices(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='dev', choices=('dev', 'release', 'all'))\n"
    )

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_fires_when_full_choices_missing(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='dev', choices=('dev',))\n"
    )

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert len(violations) >= 1
    assert "missing" in str(violations) and "release" in str(violations)


def test_preflight_check_145_accepts_same_line_waiver(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--scope', default='dev')  # PREFLIGHT_CLI_SCOPE_DEFAULT_OK:test-fixture\n"
    )

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_accepts_multiline_waiver(tmp_path: Path):
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

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

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_145_strict_mode_raises():
    from tac.preflight import (
        PreflightError,
        check_preflight_cli_default_scope_is_bounded_dev,
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
            check_preflight_cli_default_scope_is_bounded_dev(
                repo_root=fake_repo, verbose=False, strict=True,
            )


def test_preflight_check_145_no_scope_flag_is_clean(tmp_path: Path):
    """A CLI with no --scope flag at all is not in scope."""
    from tac.preflight import check_preflight_cli_default_scope_is_bounded_dev

    fake_repo = tmp_path
    target = fake_repo / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--profile', default='proven_baseline')\n"
    )

    violations = check_preflight_cli_default_scope_is_bounded_dev(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []
