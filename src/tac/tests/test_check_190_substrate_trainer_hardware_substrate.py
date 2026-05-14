# SPDX-License-Identifier: MIT
"""Tests for Catalog #190 — check_substrate_trainer_does_not_hardcode_hardware_substrate.

SIREN PRE-DISPATCH AUDIT 2026-05-13 CRITICAL #1 self-protect.

Bug class: substrate trainers under ``experiments/train_substrate_*.py``
previously hardcoded ``hardware_substrate="linux_x86_64_t4"`` in the
``ContestResult`` passed to ``posterior_update_locked(...)``. Recipes
target A100 / 4090 / H100 / A10G / L40S, but the trainer silently
labeled every dispatch as T4 — violating CLAUDE.md "Forbidden
empirical-claim-without-evidence-tag (the docstring-overstatement trap)".

The canonical fix: detect substrate dynamically via
``tac.substrates._shared.trainer_skeleton.detect_hardware_substrate``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainer_does_not_hardcode_hardware_substrate,
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a fake repo with `experiments/train_substrate_*.py` populated."""
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Positive cases — gate should flag
# ─────────────────────────────────────────────────────────────────────────


def test_hardcoded_t4_is_violation(tmp_path):
    """Literal ``hardware_substrate="linux_x86_64_t4"`` is a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_foo.py":
            "result = ContestResult(\n"
            '    hardware_substrate="linux_x86_64_t4",\n'
            ")\n",
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1
    assert "train_substrate_foo.py" in vs[0]


def test_hardcoded_a100_is_violation(tmp_path):
    """Literal ``hardware_substrate="linux_x86_64_a100"`` is a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_bar.py":
            'hardware_substrate="linux_x86_64_a100",\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_hardcoded_4090_is_violation(tmp_path):
    """Literal ``hardware_substrate="linux_x86_64_4090"`` is a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_baz.py":
            'hardware_substrate="linux_x86_64_4090",\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_hardcoded_h100_is_violation(tmp_path):
    """Literal ``hardware_substrate="linux_x86_64_h100"`` is a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_qux.py":
            'hardware_substrate="linux_x86_64_h100",\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_hardcoded_with_trailing_comment_is_violation(tmp_path):
    """Literal with trailing # comment is still a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_quux.py":
            'hardware_substrate="linux_x86_64_t4",  # default; wrapper overrides\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_multiple_trainers_all_flagged(tmp_path):
    """Each violating trainer contributes one violation row."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_a.py":
            'hardware_substrate="linux_x86_64_t4",\n',
        "experiments/train_substrate_b.py":
            'hardware_substrate="linux_x86_64_a100",\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 2
    assert any("train_substrate_a.py" in v for v in vs)
    assert any("train_substrate_b.py" in v for v in vs)


# ─────────────────────────────────────────────────────────────────────────
# Negative cases — gate should accept
# ─────────────────────────────────────────────────────────────────────────


def test_dynamic_detection_is_accepted(tmp_path):
    """Variable reference is the canonical pattern; not a violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_clean.py":
            "_detected_substrate = _canon_detect_hardware_substrate(...)\n"
            "result = ContestResult(\n"
            "    hardware_substrate=_detected_substrate,\n"
            ")\n",
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_helper_call_is_accepted(tmp_path):
    """Direct call expression is also accepted."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_clean2.py":
            "result = ContestResult(\n"
            "    hardware_substrate=detect_hardware_substrate(axis='cuda'),\n"
            ")\n",
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_empty_corpus_no_violations(tmp_path):
    """No experiments dir → no violations."""
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert vs == []


def test_non_substrate_trainer_not_scanned(tmp_path):
    """Files not matching `train_substrate_*.py` are not scanned."""
    root = _make_corpus(tmp_path, {
        "experiments/train_other.py":
            'hardware_substrate="linux_x86_64_t4",\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_unrelated_string_not_flagged(tmp_path):
    """Unrelated mentions of `linux_x86_64_t4` (e.g. comments) are ignored."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_doc.py":
            '# This trainer historically used linux_x86_64_t4 as default\n'
            "print('contest substrate detection')\n",
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# Waiver mechanism
# ─────────────────────────────────────────────────────────────────────────


def test_same_line_waiver_respected(tmp_path):
    """Same-line ``# HARDWARE_SUBSTRATE_HARDCODE_OK:<reason>`` waives."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_waived.py":
            'hardware_substrate="linux_x86_64_t4",  '
            '# HARDWARE_SUBSTRATE_HARDCODE_OK:test-fixture-pins-known-substrate\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_waiver_without_reason_does_not_waive(tmp_path):
    """Bare ``# HARDWARE_SUBSTRATE_HARDCODE_OK:`` (no reason) does NOT waive."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_bare_waiver.py":
            'hardware_substrate="linux_x86_64_t4",  # HARDWARE_SUBSTRATE_HARDCODE_OK:\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


# ─────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on any violation."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_strict.py":
            'hardware_substrate="linux_x86_64_t4",\n',
    })
    with pytest.raises(PreflightError):
        check_substrate_trainer_does_not_hardcode_hardware_substrate(
            repo_root=root, strict=True, verbose=False,
        )


def test_strict_mode_silent_on_clean_corpus(tmp_path):
    """strict=True is silent on clean corpus."""
    root = _make_corpus(tmp_path, {
        "experiments/train_substrate_clean.py":
            'hardware_substrate=_detected_substrate,\n',
    })
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=root, strict=True, verbose=False,
    )
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_has_zero_violations():
    """STRICT @ 0 invariant: live repo has zero violations.

    This regression guard ensures the SIREN audit's bulk-fix is preserved.
    """
    vs = check_substrate_trainer_does_not_hardcode_hardware_substrate(
        repo_root=None, strict=False, verbose=False,
    )
    assert vs == [], (
        "Catalog #190 live violations re-introduced: "
        + "\n  ".join(v[:200] for v in vs[:5])
    )
