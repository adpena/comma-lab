# SPDX-License-Identifier: MIT
"""Tests for preflight_t4_oom_training_guard (Check 73).

Closes BUG CLASS B (the 2026-04-29 Modal incident): a remote_lane_*.sh
invokes experiments/train_segmap.py (or train_renderer.py) without an
explicit --batch-size, and the unchunked train_epoch tries to allocate
7.03 GiB on a 14.56 GiB T4 — OOM in 126 s, $3 Modal time wasted across
Lanes SA-v2 / SC++-v2 / SO-v2.

These tests cover:
  * positive: real codebase scan returns 0 violations (live STRICT gate)
  * negative-missing: lane script with no --batch-size flag is detected
  * negative-too-large: lane script with --batch-size 64 is detected
  * positive-bounded: lane script with --batch-size 8 passes
  * positive-tier-hint: lane script with `export GPU_TIER_HINT=A10G` opts out
  * non-target ignored: a lane invoking only contest_auth_eval.py is skipped
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    preflight_t4_oom_training_guard,
)


# ── Live-codebase gate ───────────────────────────────────────────────────────


def test_real_codebase_passes_strict() -> None:
    """All scripts/remote_lane_*.sh files MUST pass STRICT today.

    Will raise PreflightError on any violation — this is the gate that
    flips Check 73 to STRICT in preflight_all().
    """
    violations = preflight_t4_oom_training_guard(strict=True, verbose=False)
    assert violations == [], (
        f"Live-codebase T4-OOM training-guard scan found "
        f"{len(violations)} violations:\n  "
        + "\n  ".join(violations)
    )


# ── Synthetic-script helpers ─────────────────────────────────────────────────


def _write_shell(tmp_path: Path, name: str, body: str) -> Path:
    sh_dir = tmp_path / "scripts"
    sh_dir.mkdir(exist_ok=True)
    sh = sh_dir / name
    sh.write_text(body)
    return sh


# ── Negative tests ──────────────────────────────────────────────────────────


def test_missing_batch_size_is_detected(tmp_path: Path) -> None:
    """A lane script invoking train_segmap.py without --batch-size must fail."""
    _write_shell(
        tmp_path,
        "remote_lane_test_missing.sh",
        '#!/bin/bash\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --epochs 600 --lr 1e-3 \\\n'
        '    --output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_missing.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "without --batch-size" in violations[0]
    assert "BUG CLASS B" in violations[0]


def test_oversized_batch_size_is_detected(tmp_path: Path) -> None:
    """--batch-size 64 > T4 cap (32) must fail."""
    _write_shell(
        tmp_path,
        "remote_lane_test_oversized.sh",
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --epochs 600 --batch-size 64 --lr 1e-3 \\\n'
        '    --output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_oversized.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "--batch-size 64" in violations[0]
    assert "T4 cap" in violations[0]


def test_strict_raises_on_violation(tmp_path: Path) -> None:
    """STRICT must raise PreflightError on any violation."""
    _write_shell(
        tmp_path,
        "remote_lane_test_strict.sh",
        '"$PYBIN" -u experiments/train_segmap.py --epochs 100 '
        '--output-dir /tmp/out\n',
    )
    with pytest.raises(PreflightError, match="T4-OOM TRAINING GUARD"):
        preflight_t4_oom_training_guard(
            repo_root=tmp_path,
            shell_files=["scripts/remote_lane_test_strict.sh"],
            strict=True,
            verbose=False,
        )


# ── Positive tests ──────────────────────────────────────────────────────────


def test_batch_size_at_cap_passes(tmp_path: Path) -> None:
    """--batch-size 8 (the SegMap convention) passes."""
    _write_shell(
        tmp_path,
        "remote_lane_test_bs8.sh",
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --epochs 600 --batch-size 8 --lr 1e-3 \\\n'
        '    --output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_bs8.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_batch_size_at_exact_cap_passes(tmp_path: Path) -> None:
    """--batch-size 32 (exactly at the cap) passes."""
    _write_shell(
        tmp_path,
        "remote_lane_test_bs32.sh",
        '"$PYBIN" -u experiments/train_segmap.py --batch-size 32 '
        '--epochs 100 --output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_bs32.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_gpu_tier_hint_opts_out(tmp_path: Path) -> None:
    """A lane that exports GPU_TIER_HINT skips the batch-size check entirely."""
    _write_shell(
        tmp_path,
        "remote_lane_test_tier_hint.sh",
        '#!/bin/bash\n'
        'export GPU_TIER_HINT=A10G\n'
        '# No --batch-size and no cap — but tier hint says "not T4".\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --epochs 600 --lr 1e-3 \\\n'
        '    --output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_tier_hint.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_non_training_target_is_skipped(tmp_path: Path) -> None:
    """A lane that only invokes contest_auth_eval.py (not a training target)
    is not subject to the T4 batch-size cap."""
    _write_shell(
        tmp_path,
        "remote_lane_test_eval_only.sh",
        '"$PYBIN" -u experiments/contest_auth_eval.py \\\n'
        '    --archive /tmp/archive.zip --device cuda\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_eval_only.sh"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_train_renderer_also_covered(tmp_path: Path) -> None:
    """experiments/train_renderer.py is a T4-sensitive target too."""
    _write_shell(
        tmp_path,
        "remote_lane_test_renderer.sh",
        '"$PYBIN" -u experiments/train_renderer.py --epochs 100 '
        '--output-dir /tmp/out\n',
    )
    violations = preflight_t4_oom_training_guard(
        repo_root=tmp_path,
        shell_files=["scripts/remote_lane_test_renderer.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "experiments/train_renderer.py" in violations[0]
