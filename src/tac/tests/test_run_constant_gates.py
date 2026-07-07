"""Tests for tac.run_constant_gates (task #340 — hardcoded run constants in consumers)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.run_constant_gates import (
    check_no_hardcoded_run_constants_in_consumers,
    scan_repo_for_hardcoded_run_constants,
)

_REPO = Path(__file__).resolve().parents[3]


def _mk_repo(tmp_path: Path, rel: str, text: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return tmp_path


def _findings(tmp_path: Path):
    return scan_repo_for_hardcoded_run_constants(tmp_path)


# ---------------------------------------------------------------- P1: stage CLI defaults
def test_p1_flags_int_default_tau(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=300)\n')
    f = _findings(tmp_path)
    assert len(f) == 1 and f[0].pattern == "P1"


def test_p1_flags_wrapped_default_l7(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--l7", type=int,\n                default=900)\n')
    f = _findings(tmp_path)
    assert len(f) == 1 and f[0].pattern == "P1"


def test_p1_default_none_not_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=None)\n')
    assert _findings(tmp_path) == []


def test_p1_float_tau_not_flagged(tmp_path):
    # A float --tau is a different quantity (e.g. a tolerance), not a stage epoch.
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=float, default=5e-4)\n')
    assert _findings(tmp_path) == []


def test_p1_non_literal_default_not_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=cfg.tau)\n')
    assert _findings(tmp_path) == []


def test_p1_window_does_not_bleed_into_next_add_argument(tmp_path):
    # Regression for the landing false-positive: a fixed default=None flag followed
    # by an unrelated int-default flag must NOT be flagged.
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=None,\n'
             '                help="OVERRIDE only")\n'
             'ap.add_argument("--rss-mb", type=int, default=2500)\n')
    assert _findings(tmp_path) == []


# ---------------------------------------------------------------- P2: literal in strings
def test_p2_flags_literal_hint_string(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'print("run: tools/dashboard_reload.py --tau 300 --l7 600")\n')
    f = _findings(tmp_path)
    assert len(f) == 1 and f[0].pattern == "P2"  # one line = one finding


def test_p2_trainer_flag_name_not_flagged(tmp_path):
    # "--l7-start-epoch 1001" is a TRAINER flag (the DSL compile target), not the
    # consumer override "--l7 <int>".
    _mk_repo(tmp_path, "tools/foo.py",
             'cmd = "--l7-start-epoch 1001"\n')
    assert _findings(tmp_path) == []


def test_p2_comment_line_not_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py", '# old default was --tau 300\n')
    assert _findings(tmp_path) == []


# ---------------------------------------------------------------- P3: resolution literals
def test_p3_resolution_in_display_tool_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/dashboard_foo.py", "CAMERA_H, CAMERA_W = 874, 1164\n")
    f = _findings(tmp_path)
    assert len(f) == 1 and f[0].pattern == "P3"


def test_p3_resolution_in_build_tool_not_flagged(tmp_path):
    # Build/measurement tools keep deliberate provenance pins — out of P3 scope.
    _mk_repo(tmp_path, "tools/build_foo.py", "CAMERA_H = 874\n")
    assert _findings(tmp_path) == []


def test_p3_larger_number_not_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/dashboard_foo.py", "x = 18744\n")
    assert _findings(tmp_path) == []


# ---------------------------------------------------------------- P4: stage-key literals
def test_p4_stage_key_literal_assignment_flagged(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py", "tau_start = 300\n")
    f = _findings(tmp_path)
    assert len(f) == 1 and f[0].pattern == "P4"


def test_p4_derive_first_fallback_not_flagged(tmp_path):
    # The accepted derive-with-fallback pattern (dashboard_trajectory_model).
    _mk_repo(tmp_path, "tools/foo.py",
             'tau = int(schedule.get("tau_start") or 300)\n')
    assert _findings(tmp_path) == []


# ---------------------------------------------------------------- waivers + exclusions
def test_waiver_with_real_rationale_respected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=300)  '
             '# RUN_CONSTANT_OK:historical replay tool pinned to the 20260601 run\n')
    assert _findings(tmp_path) == []


def test_placeholder_waiver_rejected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--tau", type=int, default=300)  # RUN_CONSTANT_OK:<rationale>\n')
    assert len(_findings(tmp_path)) == 1


def test_excluded_surfaces_not_scanned(tmp_path):
    bad = 'ap.add_argument("--tau", type=int, default=300)\n'
    _mk_repo(tmp_path, "tools/test_foo.py", bad)          # tests excluded
    _mk_repo(tmp_path, "src/tac/witness_dsl/x.py", bad)   # the DSL itself excluded
    _mk_repo(tmp_path, "src/tac/clip_profile.py", "H = 874\n")  # canonical home excluded
    _mk_repo(tmp_path, "experiments/train_foo.py", bad)   # trainers = DSL compile target
    assert _findings(tmp_path) == []


def test_strict_raises_with_rule_chain(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             'ap.add_argument("--l7", type=int, default=600)\n')
    with pytest.raises(RuntimeError) as ei:
        check_no_hardcoded_run_constants_in_consumers(strict=True, repo_root=tmp_path)
    msg = str(ei.value)
    assert "schedule_readback" in msg and "RUN_CONSTANT_OK" in msg


# ---------------------------------------------------------------- live-repo invariants
def test_live_repo_routed_files_are_clean():
    findings = scan_repo_for_hardcoded_run_constants(_REPO)
    routed = ("dashboard_reload.py", "dashboard_supervisor.py", "launch_witness_run.py")
    dirty = [f for f in findings if Path(f.path).name in routed]
    assert dirty == [], f"routed consumers regressed: {[f.describe() for f in dirty]}"


def test_live_repo_scan_runs_without_exception():
    findings = check_no_hardcoded_run_constants_in_consumers(strict=False, repo_root=_REPO)
    assert isinstance(findings, list)
