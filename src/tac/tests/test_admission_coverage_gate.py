"""Tests for the #254 self-protect coverage gate
``tac.preflight.check_heavy_witness_trainers_call_admission_guard``.

The gate scans ``experiments/train_*witness*realized_through_R*.py`` and warns for any heavy
witness trainer that does NOT call ``assert_governed_admission`` (the P0 machine-crash gate).
It is the sister STRICT-or-warn preflight self-protection for the admission-guard wire-in, so a
future un-wired witness trainer can never silently bypass the memory governor."""
from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    check_heavy_witness_trainers_call_admission_guard as _gate,
)

_WITNESS = "train_levelset_witness_realized_through_R_mlx.py"
_BODY_NO_GUARD = (
    "import argparse\n"
    "def main(argv=None):\n"
    "    ap = argparse.ArgumentParser()\n"
    "    args = ap.parse_args(argv)\n"
    "    return 0\n"
)
_BODY_WITH_GUARD = (
    "import argparse\n"
    "def main(argv=None):\n"
    "    ap = argparse.ArgumentParser()\n"
    "    args = ap.parse_args(argv)\n"
    "    from tac.admission_guard import assert_governed_admission\n"
    "    assert_governed_admission('train_levelset_witness_realized_through_R_mlx')\n"
    "    return 0\n"
)


def _mk(tmp_path, name, body):
    exp = tmp_path / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / name).write_text(body)


def test_positive_missing_guard_flags(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_NO_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "assert_governed_admission" in v[0]
    assert _WITNESS in v[0]


def test_negative_guard_present_clears(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_WITH_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_waiver_respected(tmp_path):
    body = _BODY_NO_GUARD + "# ADMISSION_GUARD_WAIVED: light entrypoint, never allocates heavy\n"
    _mk(tmp_path, _WITNESS, body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_out_of_scope_substrate_trainer_not_scanned(tmp_path):
    # a train_substrate_*.py (no 'witness...realized_through_R') is out of scope even without a guard.
    _mk(tmp_path, "train_substrate_foo.py", _BODY_NO_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_strict_raises_on_missing(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_NO_GUARD)
    with pytest.raises(PreflightError):
        _gate(repo_root=tmp_path, strict=True, verbose=False)


def test_real_repo_all_witness_trainers_wired():
    # every live witness trainer already calls the guard -> gate is at live-count 0.
    v = _gate(strict=False, verbose=False)
    assert v == [], f"unexpected un-wired witness trainer(s): {v}"
