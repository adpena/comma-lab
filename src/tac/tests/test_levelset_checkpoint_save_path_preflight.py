"""Focused tests for Task #537's warn-only checkpoint save-path preflight."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_levelset_checkpoint_save_paths_preserve_periodic_full_state,
)


def test_live_levelset_trainer_has_distinct_periodic_full_state_paths():
    assert check_levelset_checkpoint_save_paths_preserve_periodic_full_state() == []


def test_overwriting_periodic_full_state_fixture_is_reported():
    source = """
def save(out_dir, resume_arrays):
    _atomic_savez(out_dir / "levelset_resume_state.npz", resume_arrays)
"""
    violations = check_levelset_checkpoint_save_paths_preserve_periodic_full_state(
        source_text=source)
    assert len(violations) == 1
    assert "distinct stage+epoch" in violations[0]


def test_clean_source_fixture_is_accepted():
    source = """
def names(stage, ep):
    return (f"levelset_periodic_ema_{stage}_ep{ep}.npz",
            f"levelset_periodic_resume_{stage}_ep{ep}.npz")

def save(out_dir, ema_arrays, resume_arrays, ema_periodic, resume_periodic):
    _atomic_savez(out_dir / "levelset_resume_state.npz", resume_arrays)
    _atomic_savez(out_dir / ema_periodic, ema_arrays)
    _atomic_savez(out_dir / resume_periodic, resume_arrays)
"""
    assert check_levelset_checkpoint_save_paths_preserve_periodic_full_state(
        source_text=source) == []


def test_strict_mode_raises_for_violation(tmp_path: Path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        '_atomic_savez(out_dir / "levelset_resume_state.npz", resume_arrays)\n',
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="checkpoint_save_paths"):
        check_levelset_checkpoint_save_paths_preserve_periodic_full_state(
            trainer_path=trainer, strict=True)
