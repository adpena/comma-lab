# SPDX-License-Identifier: MIT
"""Tests for Catalog #197 — check_full_cpu_requires_explicit_advisory_waiver.

Operator approved 2026-05-13. Self-protection for the
device-or-die-gate-bypass-via-uncoupled-flags bug class introduced by the
time-traveler ``--full-cpu`` mode landing.

Coverage:

- Live repo regression guard (count = 0 at landing)
- Positive: violating trainer detected (missing waiver flag)
- Positive: violating trainer detected (missing validator call)
- Negative: canonical trainer with both coupling layers accepted
- Negative: trainer that does NOT declare --full-cpu accepted
- Same-line waiver opt-out accepted
- Strict mode raises ``PreflightError``
- Comment / docstring mentions of --full-cpu not flagged
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_full_cpu_requires_explicit_advisory_waiver,
)

# ---------------------------------------------------------------------------
# Live repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_violation_count_zero():
    """Catalog #197 must remain STRICT @ 0 on the live tree."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=repo_root,
    )
    assert violations == [], (
        f"Live count drifted above 0: {violations}"
    )


# ---------------------------------------------------------------------------
# Positive: violating trainer detected
# ---------------------------------------------------------------------------


def test_missing_waiver_flag_flagged(tmp_path):
    """A trainer declaring --full-cpu without --advisory-cpu-explicitly-waived is flagged."""
    (tmp_path / "experiments").mkdir()
    bad = tmp_path / "experiments" / "train_substrate_no_waiver.py"
    bad.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "missing the sister" in violations[0]
    assert "--advisory-cpu-explicitly-waived" in violations[0]


def test_missing_validator_call_flagged(tmp_path):
    """A trainer declaring both flags but no validator call is flagged."""
    (tmp_path / "experiments").mkdir()
    bad = tmp_path / "experiments" / "train_substrate_no_validator.py"
    bad.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")\n'
        'p.add_argument("--advisory-cpu-explicitly-waived", action="store_true")\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "validator" in violations[0]


# ---------------------------------------------------------------------------
# Negative: canonical trainer accepted
# ---------------------------------------------------------------------------


def test_canonical_trainer_accepted(tmp_path):
    """A canonical trainer with both flags + validator call passes."""
    (tmp_path / "experiments").mkdir()
    good = tmp_path / "experiments" / "train_substrate_canonical.py"
    good.write_text(
        'import argparse\n'
        '\n'
        'def _validate_full_cpu_flags(args):\n'
        '    pass\n'
        '\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")\n'
        'p.add_argument("--advisory-cpu-explicitly-waived", action="store_true")\n'
        '\n'
        'def main():\n'
        '    args = p.parse_args()\n'
        '    _validate_full_cpu_flags(args)\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_trainer_without_full_cpu_flag_accepted(tmp_path):
    """A trainer that does NOT declare --full-cpu is out of scope."""
    (tmp_path / "experiments").mkdir()
    irrelevant = tmp_path / "experiments" / "train_substrate_no_full_cpu.py"
    irrelevant.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--device", choices=("cuda","cpu"), default="cuda")\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


def test_alternate_validator_name_accepted(tmp_path):
    """The non-underscore validator name ``validate_full_cpu_flags`` is accepted."""
    (tmp_path / "experiments").mkdir()
    good = tmp_path / "experiments" / "train_substrate_alt_validator.py"
    good.write_text(
        'import argparse\n'
        '\n'
        'def validate_full_cpu_flags(args):\n'
        '    pass\n'
        '\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")\n'
        'p.add_argument("--advisory-cpu-explicitly-waived", action="store_true")\n'
        '\n'
        'def main():\n'
        '    validate_full_cpu_flags(p.parse_args())\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Same-line waiver
# ---------------------------------------------------------------------------


def test_same_line_waiver_accepted(tmp_path):
    """Same-line ``# FULL_CPU_COUPLED_FLAGS_OK:`` waiver opts out."""
    (tmp_path / "experiments").mkdir()
    waived = tmp_path / "experiments" / "train_substrate_waived.py"
    waived.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")  '
        '# FULL_CPU_COUPLED_FLAGS_OK:alternative-attestation-mechanism-via-env-var\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_strict_mode_raises(tmp_path):
    """``strict=True`` raises ``PreflightError`` when violations exist."""
    (tmp_path / "experiments").mkdir()
    bad = tmp_path / "experiments" / "train_substrate_evil.py"
    bad.write_text(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--full-cpu", action="store_true")\n'
    )
    with pytest.raises(PreflightError) as exc_info:
        check_full_cpu_requires_explicit_advisory_waiver(
            strict=True, verbose=False, repo_root=tmp_path,
        )
    assert "Catalog #197" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Comment / docstring mentions not flagged
# ---------------------------------------------------------------------------


def test_docstring_mention_not_flagged(tmp_path):
    """A docstring mention of ``--full-cpu`` (without argparse decl) is OK."""
    (tmp_path / "experiments").mkdir()
    docs_only = tmp_path / "experiments" / "train_substrate_docs_only.py"
    docs_only.write_text(
        '"""Mentions --full-cpu in docstring only; sister tool implements it."""\n'
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--device", choices=("cuda","cpu"))\n'
    )
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# No experiments/ directory
# ---------------------------------------------------------------------------


def test_no_experiments_dir(tmp_path):
    """Missing ``experiments/`` directory returns empty (no error)."""
    violations = check_full_cpu_requires_explicit_advisory_waiver(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert violations == []
