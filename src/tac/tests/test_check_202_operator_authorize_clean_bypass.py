"""Tests for Catalog #202 whole-tree clean-check bypass attestation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_catalog_202_bypass_requires_paired_env_attestation,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTENT = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
_ATTESTATION = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"


def test_live_repo_catalog_202_violation_count_zero() -> None:
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=_REPO_ROOT,
    )
    assert violations == []


def test_py_callsite_intent_without_attestation_flagged(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_dispatcher.py"
    bad.write_text(
        "import os\n"
        f'os.environ["{_INTENT}"] = "1"\n'
        "print('no sentinel attestation')\n"
    )
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert _INTENT in violations[0]
    assert _ATTESTATION in violations[0]


def test_sh_callsite_intent_without_attestation_flagged(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "bad_wrapper.sh"
    bad.write_text(
        "#!/bin/bash\n"
        f"export {_INTENT}=1\n"
        "echo no-attestation\n"
    )
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert len(violations) == 1


def test_paired_env_vars_accepted(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    good = tmp_path / "scripts" / "good_wrapper.sh"
    good.write_text(
        "#!/bin/bash\n"
        f"export {_INTENT}=1\n"
        f"export {_ATTESTATION}=1\n"
    )
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert violations == []


def test_comment_only_mention_not_flagged(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "doc_only.py"
    ok.write_text(
        f'"""Mentions {_INTENT}=1 in documentation only."""\n'
        f"# export {_INTENT}=1\n"
        "print('ok')\n"
    )
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert violations == []


def test_sameline_waiver_accepted(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    waived = tmp_path / "tools" / "waived_dispatcher.py"
    waived.write_text(
        "import os\n"
        f'os.environ["{_INTENT}"] = "1"  '
        "# OPERATOR_AUTHORIZE_CLEAN_BYPASS_OK:fixture-has-independent-guard\n"
    )
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert violations == []


def test_out_of_window_attestation_not_accepted(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "far_apart.py"
    lines = [f'os.environ["{_INTENT}"] = "1"']
    lines.extend(f"# filler {idx}" for idx in range(35))
    lines.append(f'os.environ["{_ATTESTATION}"] = "1"')
    bad.write_text("\n".join(lines) + "\n")
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert len(violations) == 1


def test_canonical_operator_authorize_contract_checked(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    target = tmp_path / "tools" / "operator_authorize.py"
    target.write_text("print('missing catalog 202 helper')\n")
    violations = check_catalog_202_bypass_requires_paired_env_attestation(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert any("_whole_tree_clean_check_bypass_active" in item for item in violations)
    assert any("_dispatch_modal" in item for item in violations)


def test_strict_mode_raises(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad.py"
    bad.write_text(f'os.environ["{_INTENT}"] = "1"\n')
    with pytest.raises(PreflightError, match="Catalog #202"):
        check_catalog_202_bypass_requires_paired_env_attestation(
            strict=True,
            verbose=False,
            repo_root=tmp_path,
        )


def _run_helper_with_env(env_updates: dict[str, str | None]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in (_INTENT, _ATTESTATION):
        env.pop(key, None)
    for key, value in env_updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'tools'); "
                "import operator_authorize as oa; "
                "print(oa._whole_tree_clean_check_bypass_active())"
            ),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runtime_helper_no_env_returns_false() -> None:
    proc = _run_helper_with_env({})
    assert proc.returncode == 0
    assert "False" in proc.stdout


def test_runtime_helper_intent_without_attestation_exits_12() -> None:
    proc = _run_helper_with_env({_INTENT: "1"})
    assert proc.returncode == 12
    assert _ATTESTATION in proc.stderr


def test_runtime_helper_paired_env_returns_true() -> None:
    proc = _run_helper_with_env({_INTENT: "1", _ATTESTATION: "1"})
    assert proc.returncode == 0
    assert "True" in proc.stdout
    assert "OPERATOR-AUTHORIZE BYPASS" in proc.stderr


def test_runtime_helper_falsy_attestation_exits_12() -> None:
    proc = _run_helper_with_env({_INTENT: "1", _ATTESTATION: "0"})
    assert proc.returncode == 12
    assert "missing or falsy" in proc.stderr
