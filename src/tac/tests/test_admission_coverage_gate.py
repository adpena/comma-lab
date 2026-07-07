"""Tests for the #254 self-protect coverage gate
``tac.preflight.check_heavy_witness_trainers_call_admission_guard``.

The gate scans ``experiments/train_*witness*realized_through_R*.py`` AND (since the 2026-07-06
memory-safety review) ``experiments/train_substrate_*.py`` and warns for any heavy trainer that
does NOT call ``assert_governed_admission`` (the P0 machine-crash gate). It is the sister
STRICT-or-warn preflight self-protection for the admission-guard wire-in, so a future un-wired
heavy trainer can never silently bypass the memory governor. The substrate family is warn-only
VISIBLE BACKLOG (~104 files at in-scoping) — a tracked queue, not a carve-out."""
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


def test_substrate_trainer_now_in_scope_flagged(tmp_path):
    # 2026-07-06 review: train_substrate_*.py is IN scope — an un-guarded substrate trainer is a
    # visible warn (the backlog), no longer a silent carve-out.
    _mk(tmp_path, "train_substrate_foo.py", _BODY_NO_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "train_substrate_foo.py" in v[0]
    assert "assert_governed_admission" in v[0]


def test_substrate_trainer_waiver_respected(tmp_path):
    body = _BODY_NO_GUARD + "# ADMISSION_GUARD_WAIVED: light entrypoint, never allocates heavy\n"
    _mk(tmp_path, "train_substrate_foo.py", body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_substrate_trainer_with_guard_clears(tmp_path):
    _mk(tmp_path, "train_substrate_foo.py", _BODY_WITH_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_strict_raises_on_missing(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_NO_GUARD)
    with pytest.raises(PreflightError):
        _gate(repo_root=tmp_path, strict=True, verbose=False)


def test_real_repo_all_witness_trainers_wired():
    # every live WITNESS trainer already calls the guard -> witness live-count 0. The substrate
    # family is the KNOWN warn-only backlog (visible queue, wired by a follow-up campaign) —
    # substrate violations are allowed here but nothing OUTSIDE that family may appear.
    v = _gate(strict=False, verbose=False)
    non_substrate = [x for x in v if not x.startswith("experiments/train_substrate_")]
    assert non_substrate == [], f"unexpected un-wired witness trainer(s): {non_substrate}"
