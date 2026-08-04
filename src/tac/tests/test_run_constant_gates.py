"""Tests for tac.run_constant_gates (task #340 — hardcoded run constants in consumers)."""
from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# THE RATCHET + ITS WIRE-IN (ddm_wt1, task #868).
# This module had ZERO consumers from 2026-07-07 until 2026-08-01: its own docstring
# deferred the wire-in behind a named blocker and nothing ever came back for it. These
# tests pin BOTH halves — that the ratchet can actually fail, and that something real
# calls it — because a gate nobody runs and a gate that cannot fail are the same defect
# wearing different clothes.
# ---------------------------------------------------------------------------


def test_ratchet_passes_at_the_measured_baseline() -> None:
    """POSITIVE control on the real repo: at HEAD the pinned baseline holds."""
    from tac.run_constant_gates import run_constant_ratchet

    ok, report = run_constant_ratchet(_REPO)
    assert ok, report
    assert "PASS" in report


def test_ratchet_actually_fails_when_the_baseline_is_tightened() -> None:
    """NEGATIVE control: the anti-vacuity property.

    A gate that returns True unconditionally prints the clean symbol forever and trains its
    readers to ignore it. Driving the pins to zero must produce a FAIL naming every finding,
    which is exactly what a NEW violation would look like against the real baseline.
    """
    from tac.run_constant_gates import run_constant_ratchet

    ok, report = run_constant_ratchet(
        _REPO, baseline={"hardcoded_run_constants": 0, "canonical_constant_copies": 0})
    assert not ok
    assert "FAIL" in report
    assert "NEW violation" in report


def test_ratchet_baseline_keys_match_the_checks_it_runs() -> None:
    """A pin whose key nobody reads is silently vacuous — the scope-emptiness genus."""
    from tac.run_constant_gates import RUN_CONSTANT_RATCHET_BASELINE, run_constant_ratchet

    _, report = run_constant_ratchet(_REPO)
    for key in RUN_CONSTANT_RATCHET_BASELINE:
        assert key in report, f"baseline key {key!r} is pinned but never reported"
    # ddm_gk1 2026-08-03: P6 joined the ratchet. This assertion is the reason the
    # key set is pinned at all — adding a pin without adding it here would leave a
    # scanner whose result nobody reads, so the pin is updated deliberately.
    assert set(RUN_CONSTANT_RATCHET_BASELINE) == {
        "hardcoded_run_constants",
        "canonical_constant_copies",
        "guarded_constant_frozen_literals",
    }


def test_ratchet_reports_a_drained_count_as_needing_a_re_pin() -> None:
    """Draining below baseline must SAY so, or the pins rot upward and the teeth go blunt."""
    from tac.run_constant_gates import run_constant_ratchet

    ok, report = run_constant_ratchet(
        _REPO, baseline={"hardcoded_run_constants": 999, "canonical_constant_copies": 999})
    assert ok
    assert "re-pin" in report


def test_gate_38_is_registered_in_all_lanes_preflight_and_calls_the_ratchet() -> None:
    """THE WIRE-IN ITSELF. Without this the module is orphaned again the moment someone
    reorders the gate list, and the fix would be invisible to every test."""
    import tools.all_lanes_preflight as alp

    ok, report = alp._run_run_constant_gates_gate()
    assert ok, report
    assert "ratchet" in report

    src = Path(alp.__file__).read_text(encoding="utf-8")
    assert "_run_run_constant_gates_gate," in src, "runner must be registered as a PreflightStep"
    # Regex, not an indentation-exact literal: a reformat must not silently un-test the wire-in.
    assert re.search(r'PreflightStep\(\s*"GATE",\s*38,', src), (
        "Gate #38 must be registered as a numbered PreflightStep")
    assert "Gate #38" in src, "the gate must be listed in the operator-facing docstring"


def test_failure_report_names_the_derivation_fix_not_just_the_count() -> None:
    """CLAUDE.md failure-message discipline: the output must be the documentation."""
    from tac.run_constant_gates import run_constant_ratchet

    doc = run_constant_ratchet.__doc__ or ""
    assert "Rule chain" in doc
    assert "waive same-line" in doc or "re-derivation trigger" in doc


# ---------------------------------------------------------------------------
# P6 — frozen literal at a site a GuardedConstant declares a LIVE derivation for
# (ddm_gk1, task #847, 2026-08-03).
#
# P6 is pinned at live count 0 BECAUSE the same landing migrated its only
# instance. A gate at zero with no test showing it fire is indistinguishable
# from a gate that CANNOT fire — the vacuity genus — so every test below exists
# to make the difference observable. `_p6_repo` builds a synthetic repo carrying
# the exact shape the canonical instance had before migration.
# ---------------------------------------------------------------------------
_P6_DECLARED_NAME = "margin_floor"      # from MARGIN_FLOOR.literal_site_names
_P6_DECLARED_VALUE = "0.1"              # from MARGIN_FLOOR.incumbent_literal


def _p6_findings(tmp_path: Path):
    from tac.run_constant_gates import scan_repo_for_guarded_constant_frozen_literals

    return scan_repo_for_guarded_constant_frozen_literals(tmp_path)


def test_p6_positive_control_frozen_function_default_is_detected(tmp_path):
    """POSITIVE CONTROL: the canonical instance's exact pre-migration shape.

    `margin_floor: float = 0.1` as a function default, in a file that never
    imports the registry, is the literal that froze the output of
    `tac.optimization.lane_guard:derive_margin_floor`.
    """
    _mk_repo(tmp_path, "src/tac/optimization/mod.py",
             f"def __init__(self, {_P6_DECLARED_NAME}: float = {_P6_DECLARED_VALUE}):\n"
             "    return None\n")
    f = _p6_findings(tmp_path)
    assert len(f) == 1, [x.describe() for x in f]
    assert f[0].site_name == _P6_DECLARED_NAME
    assert f[0].constant_id == "seg_margin_hinge_floor"
    assert "derive_margin_floor" in f[0].derivation


def test_p6_positive_control_plain_assignment_is_detected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py", f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n")
    assert len(_p6_findings(tmp_path)) == 1


def test_p6_positive_control_call_keyword_is_detected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             f"build(x, {_P6_DECLARED_NAME}={_P6_DECLARED_VALUE})\n")
    assert len(_p6_findings(tmp_path)) == 1


def test_p6_undeclared_name_is_not_flagged(tmp_path):
    """The gate is DECLARATION-DRIVEN: an undeclared constant is out of scope by
    design (ddm_gd5 refuted the auto-derived variant), and must not be guessed at."""
    _mk_repo(tmp_path, "tools/foo.py", f"some_other_threshold = {_P6_DECLARED_VALUE}\n")
    assert _p6_findings(tmp_path) == []


def test_p6_declared_name_with_a_different_value_is_not_flagged(tmp_path):
    """Both the name AND the exact value must match — this is what keeps the
    false-positive rate at zero on a 5,000-file scan."""
    _mk_repo(tmp_path, "tools/foo.py", f"{_P6_DECLARED_NAME} = 0.25\n")
    assert _p6_findings(tmp_path) == []


def test_p6_waiver_with_real_rationale_respected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}  "
             "# GUARDED_CONSTANT_OK:replay tool pinned to the 20260601 archive\n")
    assert _p6_findings(tmp_path) == []


def test_p6_placeholder_waiver_rejected(tmp_path):
    _mk_repo(tmp_path, "tools/foo.py",
             f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}  # GUARDED_CONSTANT_OK:<rationale>\n")
    assert len(_p6_findings(tmp_path)) == 1


def test_p6_importing_the_registry_does_NOT_launder_a_frozen_literal(tmp_path):
    """INVERTED 2026-08-03 (ddm_gk1) by sm1's GAUGE SELF-TEST — deliberately.

    This test previously asserted that a file which imports the registry is
    EXEMPT. Ask the gauge question: what does the gate read if the cure is
    applied and nothing else changes? Adding an import is the cosmetic cure, so
    the old exemption made the gate read "cured" while the frozen literal sat
    untouched -- it measured the knob, not the condition.

    The exemption was also redundant: the scan only flags ``ast.Constant``, so a
    GENUINELY migrated site (whose default is a Name) is already invisible --
    asserted by the sibling test below. Removing it additionally closes the
    recorded blind spot that ``run_constant_gates.py`` exempted ITSELF by
    importing REGISTRY.

    A file that imports the registry AND still hardcodes the value is exactly
    the case worth catching.
    """
    _mk_repo(tmp_path, "tools/foo.py",
             "from tac.witness_dsl.guarded_constant_registry import MARGIN_FLOOR_INCUMBENT\n"
             f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n")
    assert len(_p6_findings(tmp_path)) == 1


def test_p6_a_genuinely_migrated_site_is_invisible(tmp_path):
    """The REAL cure -- the value becomes a Name -- is what silences the gate."""
    _mk_repo(tmp_path, "tools/foo.py",
             "from tac.witness_dsl.guarded_constant_registry import MARGIN_FLOOR_INCUMBENT\n"
             f"def f({_P6_DECLARED_NAME}: float = MARGIN_FLOOR_INCUMBENT):\n"
             f"    return {_P6_DECLARED_NAME}\n")
    assert _p6_findings(tmp_path) == []


def test_p6_merely_mentioning_the_registry_does_not_exempt_a_file(tmp_path):
    """Regression guard for `mentions-it == guarded-by-it`.

    The exemption originally tested the raw file TEXT, so a comment naming the
    registry silenced the gate for that whole file. Exemption is now an AST
    import check: naming a guard is not being guarded by it.
    """
    _mk_repo(tmp_path, "tools/foo.py",
             "# see tac.witness_dsl.guarded_constant_registry for the declaration\n"
             f'DOC = "tac.witness_dsl.guarded_constant_registry"\n'
             f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n")
    assert len(_p6_findings(tmp_path)) == 1


def test_p6_strict_raises_with_the_rule_chain(tmp_path):
    from tac.run_constant_gates import check_no_frozen_literal_where_guarded_derivation_declared

    _mk_repo(tmp_path, "tools/foo.py", f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n")
    with pytest.raises(RuntimeError) as ei:
        check_no_frozen_literal_where_guarded_derivation_declared(
            strict=True, repo_root=tmp_path)
    msg = str(ei.value)
    assert "derive_margin_floor" in msg
    assert "GUARDED_CONSTANT_OK:" in msg
    assert "Fix:" in msg


# ------------------------------------------------------------------ P6 anti-vacuity
def test_p6_scope_reports_its_denominator_on_the_live_repo():
    """A zero count is only readable NEXT TO the scope that produced it."""
    from tac.run_constant_gates import scan_guarded_constant_frozen_literals_with_scope

    findings, scope = scan_guarded_constant_frozen_literals_with_scope(_REPO)
    assert findings == [], [f.describe() for f in findings]
    assert not scope.is_vacuous, scope.describe()
    assert scope.registry_import_ok
    assert scope.declared_site_names >= 1
    assert scope.files_scanned > 100, "the live scan must actually traverse the repo"


def test_p6_empty_scope_is_vacuous_not_pass(monkeypatch):
    """THE ANTI-VACUITY CONTROL. A broken registry import must NOT read as clean.

    Without this, `_guarded_literal_targets` failing degrades to zero targets ->
    zero findings -> "at baseline" -> PASS, forever, silently. That is precisely
    the defect class this gate was built to extinct, so it is asserted against
    the gate itself.
    """
    import tac.run_constant_gates as rcg

    monkeypatch.setattr(rcg, "_guarded_literal_targets", lambda: ({}, False))
    findings, scope = rcg.scan_guarded_constant_frozen_literals_with_scope(_REPO)
    assert findings == []
    assert scope.is_vacuous
    assert "VACUOUS" in scope.describe()

    ok, report = rcg.run_constant_ratchet(
        _REPO,
        baseline={"hardcoded_run_constants": 999, "canonical_constant_copies": 999,
                  "guarded_constant_frozen_literals": 0},
    )
    assert not ok, "an empty P6 scope must REFUSE, not pass at baseline"
    assert "VACUOUS" in report


def test_p6_ratchet_reports_the_scope_even_when_passing():
    """The denominator travels with every report, not only with failures."""
    from tac.run_constant_gates import run_constant_ratchet

    ok, report = run_constant_ratchet(_REPO)
    assert ok, report
    assert "guarded_constant_frozen_literals scope:" in report
    assert "file(s) scanned" in report


def test_p6_canonical_instance_stays_migrated():
    """Regression guard on the migration itself: the one site P6 was built for
    must keep consuming the registry, or the gate silently loses its instance."""
    src = (_REPO / "src/tac/optimization/direct_description_joint_descent.py").read_text(
        encoding="utf-8")
    assert "from tac.witness_dsl.guarded_constant_registry import" in src
    assert "margin_floor: float = _MARGIN_FLOOR_DEFAULT" in src


# ---------------------------------------------------------------------------
# ddm_gk1 2026-08-03 — the STAGED surface + the actionable-message contract.
#
# These are the tests the repo-wide P6 tests above do not cover: the gate is
# only real if it fires at COMMIT, and its message is only actionable if the
# symbol it names exists.
# ---------------------------------------------------------------------------
def test_p6_refusal_message_names_a_registry_symbol_that_actually_exists():
    """"Fix: consume X" is worthless if X does not exist.

    The first draft upper-cased the ``constant_id``, producing
    ``SEG_MARGIN_HINGE_FLOOR`` -- while the real attribute is ``MARGIN_FLOOR``.
    The name is now derived from the registry module's own namespace.
    """
    import ast as _ast

    from tac.run_constant_gates import _guarded_literal_targets, _p6_scan_one_file
    from tac.witness_dsl import guarded_constant_registry as reg

    targets, ok = _guarded_literal_targets()
    assert ok
    src = f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n"
    v = _p6_scan_one_file("probe.py", src, _ast.parse(src), targets)[0]
    assert hasattr(reg, v.registry_attr), (
        f"describe() points at {v.registry_attr!r}, absent from the registry"
    )
    msg = v.describe()
    assert f"guarded_constant_registry.{v.registry_attr}" in msg
    assert "derive_margin_floor" in msg          # the rule that fired
    assert "Fix:" in msg                          # the fix


def test_p6_only_lines_none_means_every_line_not_no_lines():
    """``None`` is 'diff unavailable'; ``set()`` is 'nothing added'.

    Conflating them would filter out every site and pass SILENTLY -- the
    vacuity-equals-pass genus inside the guard built to close it.
    """
    import ast as _ast

    from tac.run_constant_gates import _guarded_literal_targets, _p6_scan_one_file

    targets, _ = _guarded_literal_targets()
    src = f"{_P6_DECLARED_NAME} = {_P6_DECLARED_VALUE}\n"
    tree = _ast.parse(src)
    assert len(_p6_scan_one_file("p.py", src, tree, targets, only_lines=None)) == 1
    assert _p6_scan_one_file("p.py", src, tree, targets, only_lines=set()) == []
    assert len(_p6_scan_one_file("p.py", src, tree, targets, only_lines={1})) == 1


def test_p6_staged_scanner_names_unexamined_files_instead_of_swallowing_them():
    """A caller must never report clean over files it could not read."""
    from tac.run_constant_gates import scan_staged_for_guarded_constant_frozen_literals

    violations, unexamined = scan_staged_for_guarded_constant_frozen_literals(
        repo_root=_REPO, files=["src/tac/does_not_exist_ddm_gk1.py"]
    )
    assert violations == []
    assert any("unreadable" in u for u in unexamined), unexamined


def test_p6_is_wired_into_the_hook_that_actually_fires_not_only_preflight_all():
    """MEASURED (ddm_ss1): ``--no-codebase`` -- this hook's default -- examines 0
    of 27 preflight gates, which is why STRICT Catalog #307/#308 have never run
    at commit. A gate registered only in ``preflight_all()`` is decoration.

    Also asserts ORDER: the step must precede ``run_preflight()``, which
    early-returns on failure and would otherwise skip it.
    """
    import inspect
    import re as _re

    import tools.preflight_hook as hook

    assert hasattr(hook, "run_guarded_constant_frozen_literal_scan")
    main_src = inspect.getsource(hook.main)
    # Compare CALL SITES, not substring offsets: main()'s comments mention
    # `run_preflight()` in prose long before it is called, and a naive .index()
    # matched the prose (caught by this test's own first run).
    calls = _re.findall(r"^\s+rc = (\w+)\(", main_src, _re.M)
    assert "run_guarded_constant_frozen_literal_scan" in calls, calls
    assert calls.index("run_guarded_constant_frozen_literal_scan") < calls.index(
        "run_preflight"
    ), calls

