"""Tests for preflight_shell_lane_arity (Check 72).

This is the structural fix for BUG CLASS A — invented CLI flags inside
remote_lane_*.sh shell-script invocations of experiments/*.py. The
2026-04-29 Lane MM/SA-v2/SC++-v2/SO-v2 incident burned ~$3 of Modal time
to four lane failures whose root cause was a regex scanner matching a
COMMENT containing `--hard` (in "NEVER git pull / git reset --hard").
The existing preflight_arity scans Python launchers + bash -c strings
but did NOT walk bare shell-script invocations.

These tests cover:
  * positive: real codebase scan returns 0 violations (live STRICT gate)
  * negative: a synthetic invented flag is detected
  * line-continuation: backslash-continued multiline invocations are parsed
  * pipeline-bound: flags AFTER a pipe / && do not pollute the invocation
  * non-target: flags to non-experiments/*.py invocations are ignored
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    ArityViolation,
    _collapse_shell_continuations,
    _scan_shell_lane_invocations,
    preflight_shell_lane_arity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Live-codebase gate (the STRICT promotion check) ──────────────────────────


def test_real_codebase_passes_strict() -> None:
    """All 84 scripts/remote_lane_*.sh files MUST pass STRICT.

    This is the gate that flips the check to STRICT in preflight_all().
    If this test breaks, a real BUG CLASS A regression has been introduced.
    """
    # Will raise ArityViolation on any violation.
    violations = preflight_shell_lane_arity(strict=True, verbose=False)
    assert violations == [], (
        f"Live-codebase shell-lane arity scan found "
        f"{len(violations)} violations:\n  "
        + "\n  ".join(violations)
    )


# ── Helpers for synthetic-script tests ───────────────────────────────────────


def _write_target(tmp_path: Path, name: str, flags: list[str]) -> Path:
    """Write a fake experiments/*.py with the given add_argument flags."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(exist_ok=True)
    body = "import argparse\np = argparse.ArgumentParser()\n"
    for f in flags:
        body += f'p.add_argument("{f}", type=str)\n'
    body += 'p.parse_args()\n'
    target = exp_dir / name
    target.write_text(body)
    return target


def _write_shell(tmp_path: Path, name: str, body: str) -> Path:
    """Write a fake scripts/remote_lane_<name>.sh."""
    sh_dir = tmp_path / "scripts"
    sh_dir.mkdir(exist_ok=True)
    sh = sh_dir / name
    sh.write_text(body)
    return sh


# ── Negative test: invented flag is caught ───────────────────────────────────


def test_invented_flag_is_detected(tmp_path: Path) -> None:
    """A shell script passing --invented-flag to a target that has no such
    argparse arg MUST produce a violation."""
    _write_target(tmp_path, "fake_train.py", ["--epochs", "--lr"])
    _write_shell(
        tmp_path,
        "remote_lane_test_synthetic.sh",
        '#!/bin/bash\n'
        '"$PYBIN" -u experiments/fake_train.py --epochs 100 --invented-flag VALUE\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_synthetic.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "'--invented-flag'" in violations[0]
    assert "experiments/fake_train.py" in violations[0]
    assert "BUG CLASS A" in violations[0]


def test_invented_flag_strict_raises(tmp_path: Path) -> None:
    """STRICT mode must raise ArityViolation when an invented flag is found."""
    _write_target(tmp_path, "fake_train.py", ["--epochs"])
    _write_shell(
        tmp_path,
        "remote_lane_test_strict.sh",
        '"$PYBIN" -u experiments/fake_train.py --epochs 100 --bogus 1\n',
    )
    with pytest.raises(ArityViolation, match="SHELL-LANE ARITY MISMATCH"):
        preflight_shell_lane_arity(
            repo_root=tmp_path,
            shell_files=["scripts/remote_lane_test_strict.sh"],
            strict=True,
            verbose=False,
        )


# ── Positive test: valid flags pass ──────────────────────────────────────────


def test_valid_invocation_passes(tmp_path: Path) -> None:
    """When every flag exists on the target, no violation is reported."""
    _write_target(tmp_path, "fake_train.py", ["--epochs", "--lr", "--batch-size"])
    _write_shell(
        tmp_path,
        "remote_lane_test_valid.sh",
        '"$PYBIN" -u experiments/fake_train.py '
        '--epochs 100 --lr 1e-3 --batch-size 8\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_valid.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


# ── Line-continuation test: multiline invocations are parsed ────────────────


def test_backslash_line_continuation_is_handled(tmp_path: Path) -> None:
    """Multi-line invocations via backslash continuation MUST be collapsed
    so that flags on continuation lines are still scanned."""
    _write_target(
        tmp_path,
        "fake_train.py",
        ["--epochs", "--lr", "--batch-size", "--device"],
    )
    _write_shell(
        tmp_path,
        "remote_lane_test_continuation.sh",
        '"$PYBIN" -u experiments/fake_train.py \\\n'
        '    --epochs 100 \\\n'
        '    --lr 1e-3 \\\n'
        '    --batch-size 8 \\\n'
        '    --device cuda\n',
    )
    # Should pass because all 4 flags exist on the target.
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_continuation.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_line_continuation_catches_invented_flag_on_continuation(
    tmp_path: Path,
) -> None:
    """If the invented flag is on a continuation line, line-collapse must
    still expose it to the scanner."""
    _write_target(tmp_path, "fake_train.py", ["--epochs", "--lr"])
    _write_shell(
        tmp_path,
        "remote_lane_test_continuation_invented.sh",
        '"$PYBIN" -u experiments/fake_train.py \\\n'
        '    --epochs 100 \\\n'
        '    --i-do-not-exist VALUE\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_continuation_invented.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "'--i-do-not-exist'" in violations[0]


# ── Pipeline test: flags after | / && / ; are not attributed to first cmd ──


def test_flags_after_pipe_are_not_attributed(tmp_path: Path) -> None:
    """`python a.py --good | tee --bad` — `--bad` belongs to tee, not a.py."""
    _write_target(tmp_path, "fake_train.py", ["--good"])
    _write_shell(
        tmp_path,
        "remote_lane_test_pipe.sh",
        '"$PYBIN" -u experiments/fake_train.py --good 1 2>&1 | tee --bad-flag log\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_pipe.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == [], (
        "Flags after a pipe should not be attributed to the python target. "
        f"Got: {violations}"
    )


def test_flags_after_double_amp_are_not_attributed(tmp_path: Path) -> None:
    """`python a.py --good && python b.py --bad` — only --good for a.py."""
    _write_target(tmp_path, "fake_a.py", ["--good"])
    _write_target(tmp_path, "fake_b.py", ["--bad"])
    _write_shell(
        tmp_path,
        "remote_lane_test_amp.sh",
        '"$PYBIN" -u experiments/fake_a.py --good 1 && '
        '"$PYBIN" -u experiments/fake_b.py --bad 2\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_amp.sh"],
        strict=False,
        verbose=False,
    )
    # Both invocations should be analyzed independently and pass.
    assert violations == []


# ── Negative-target test: non-experiments/*.py invocations are ignored ───────


def test_non_experiments_target_is_ignored(tmp_path: Path) -> None:
    """`python /opt/conda/bin/foo --invented` should NOT be checked."""
    _write_shell(
        tmp_path,
        "remote_lane_test_other.sh",
        'python /opt/conda/bin/foo --invented-flag\n'
        'pip install --upgrade pip\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_other.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_target_without_argparse_is_ignored(tmp_path: Path) -> None:
    """A target that has no add_argument calls (e.g. a hello-world script)
    is treated as out-of-scope, NOT as 'every flag is invented'."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "no_argparse.py").write_text('print("hello")\n')
    _write_shell(
        tmp_path,
        "remote_lane_test_noargs.sh",
        '"$PYBIN" -u experiments/no_argparse.py --whatever VALUE\n',
    )
    violations = preflight_shell_lane_arity(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_noargs.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


# ── Helper-function unit tests ───────────────────────────────────────────────


def test_collapse_shell_continuations_basic() -> None:
    """Trailing-backslash + newline + indent collapses; the leading space on
    the continuation source line is preserved (single-space separator after
    re-collapse). We assert the FUNCTIONAL property: the tokens are still
    parseable as separate words."""
    text = "echo a \\\n   b \\\n   c\n"
    out = _collapse_shell_continuations(text)
    # No more newlines (other than the trailing one).
    assert out.count("\n") == 1
    # Every token still appears in order, separated by whitespace.
    tokens = out.split()
    assert tokens == ["echo", "a", "b", "c"]


def test_collapse_shell_continuations_preserves_unrelated_newlines() -> None:
    text = "line1\nline2 \\\n   continued\nline3\n"
    out = _collapse_shell_continuations(text)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[0] == "line1"
    assert lines[1].split() == ["line2", "continued"]
    assert lines[2] == "line3"


def test_scan_shell_lane_invocations_extracts_target_and_flags(
    tmp_path: Path,
) -> None:
    sh = tmp_path / "remote_lane_x.sh"
    sh.write_text(
        '"$PYBIN" -u experiments/foo.py --a 1 --b 2 \\\n'
        '    --c 3\n'
    )
    out = _scan_shell_lane_invocations(sh)
    assert len(out) == 1
    lineno, target, flags = out[0]
    assert target == "experiments/foo.py"
    assert sorted(flags) == ["--a", "--b", "--c"]
    assert lineno >= 1
