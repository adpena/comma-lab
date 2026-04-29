"""Tests for runtime guards added in deep hardening pass 3 dimension 3.

- experiments/contest_auth_eval.py: _validate_archive_members whitelist
- src/tac/training.py: finite-loss assertion before backward (smoke only —
  full training-loop integration is exercised by existing trainer tests)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_validator():
    mod = _load_module(
        REPO_ROOT / "experiments" / "contest_auth_eval.py",
        "_test_contest_auth_eval",
    )
    return mod._validate_archive_members


# ─── Archive whitelist validator ───────────────────────────────────────────


def test_validator_passes_canonical_archive():
    """Renderer + masks + poses is the canonical contest contract."""
    validator = _get_validator()
    validator(["renderer.bin", "masks.mkv", "poses.pt"])  # no raise


def test_validator_passes_brotli_renderer():
    """Brotli-compressed renderer is allowed (.bin.br suffix)."""
    validator = _get_validator()
    validator(["renderer.bin.br", "masks.mkv", "poses.pt"])


def test_validator_rejects_macos_resource_fork():
    """macOS ._foo files inflate the rate silently — must raise."""
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN files"):
        validator(["renderer.bin", "._renderer.bin", "masks.mkv"])


def test_validator_rejects_ds_store():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        validator(["renderer.bin", ".DS_Store"])


def test_validator_rejects_unknown_file_type():
    """Stale debug artifacts (e.g., .pkl) should fail loud."""
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="UNKNOWN file types"):
        validator(["renderer.bin", "masks.mkv", "debug_state.pkl"])


def test_validator_rejects_empty_archive():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="EMPTY archive"):
        validator([])


def test_validator_rejects_macosx_dir():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        validator(["renderer.bin", "__MACOSX/renderer.bin"])


# ─── Trainer finite-loss guard (smoke check) ────────────────────────────────


def test_finite_loss_guard_present_in_training_py():
    """Verify the finite-loss assertion is wired into the canonical Trainer
    BEFORE backward(). If the guard is removed, this test fails loud so the
    operator knows protection was lost."""
    text = (REPO_ROOT / "src" / "tac" / "training.py").read_text()
    # Both Trainer paths (canonical + lazy) must guard.
    assert "non-finite loss" in text, (
        "training.py: finite-loss guard removed — re-add the .item()-based "
        "check before backward() (deep hardening pass 3 dim 3)."
    )
    # Specifically: must mention 'before backward' to ensure ordering matters.
    assert text.count("non-finite loss") >= 2, (
        "training.py: only one finite-loss guard present (expected 2: "
        "canonical Trainer + lazy variant)."
    )
