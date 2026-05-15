# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_paired_dispatch_uses_anchor_skip_flag,
    preflight_all,
)


def _write_caller(repo_root: Path, rel_path: str, text: str) -> Path:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_246_flags_paired_dispatch_callers_without_anchor_skip_flag(tmp_path: Path) -> None:
    _write_caller(
        tmp_path,
        "tools/caller.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run(['python', 'tools/dispatch_modal_paired_auth_eval.py', '--archive', 'a.zip'])",
                "",
            ]
        ),
    )

    violations = check_paired_dispatch_uses_anchor_skip_flag(repo_root=tmp_path)

    assert len(violations) == 1
    assert "tools/caller.py:2" in violations[0]
    assert "--skip-axis-if-promotable-anchor-exists" in violations[0]


def test_check_246_accepts_required_flag_in_multiline_argv(tmp_path: Path) -> None:
    _write_caller(
        tmp_path,
        "scripts/caller.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run([",
                "    'python',",
                "    'tools/dispatch_modal_paired_auth_eval.py',",
                "    '--archive',",
                "    'a.zip',",
                "    '--skip-axis-if-promotable-anchor-exists',",
                "])",
                "",
            ]
        ),
    )

    assert check_paired_dispatch_uses_anchor_skip_flag(repo_root=tmp_path) == []


def test_check_246_accepts_real_waiver_and_rejects_placeholder(tmp_path: Path) -> None:
    _write_caller(
        tmp_path,
        "experiments/ok.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run(['python', 'tools/dispatch_modal_paired_auth_eval.py'])",
                "# PAIRED_DISPATCH_FORCE_REFIRE_OK: deterministic cross-axis refire",
                "",
            ]
        ),
    )
    _write_caller(
        tmp_path,
        "experiments/bad.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run(['python', 'tools/dispatch_modal_paired_auth_eval.py'])",
                "# PAIRED_DISPATCH_FORCE_REFIRE_OK:<rationale>",
                "",
            ]
        ),
    )

    violations = check_paired_dispatch_uses_anchor_skip_flag(repo_root=tmp_path)

    assert len(violations) == 1
    assert "experiments/bad.py:2" in violations[0]
    assert "PLACEHOLDER waiver token" in violations[0]


def test_check_246_strict_raises_on_violation(tmp_path: Path) -> None:
    _write_caller(
        tmp_path,
        "src/tac/caller.py",
        "subprocess.run(['python', 'tools/dispatch_modal_paired_auth_eval.py'])\n",
    )

    with pytest.raises(PreflightError, match="Catalog #246"):
        check_paired_dispatch_uses_anchor_skip_flag(repo_root=tmp_path, strict=True)


def test_check_246_wired_into_preflight_all_warn_only() -> None:
    source = inspect.getsource(preflight_all)

    assert "check_paired_dispatch_uses_anchor_skip_flag" in source
    call_idx = source.index("check_paired_dispatch_uses_anchor_skip_flag")
    window = source[call_idx : call_idx + 160]
    assert "strict=False" in window
