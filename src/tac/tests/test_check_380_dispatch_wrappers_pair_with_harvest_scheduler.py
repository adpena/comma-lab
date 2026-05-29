# SPDX-License-Identifier: MIT
"""Tests for Catalog #380 STRICT preflight gate (Surface B of canonical
2-landing pattern for silent-orphan-harvest bug class extinction).

Sister gate: Catalog #380 `check_dispatch_wrappers_pair_with_harvest_scheduler_invocation`.
Sister of Catalog #339 (post-spawn-registration fail-closed) + Catalog
#360 (pre-spawn-FATAL silent-no-spawn).

Per RECOVERY-AUDIT-V2 PHASE A + STAND-DOWN-REVIEW-AUDIT TOP-2 op-routables
2026-05-28: refuses dispatch wrappers under `tools/` + `scripts/` +
`experiments/` + `src/tac/` (excluding self-exempt) that contain
dispatch-trigger tokens (`.spawn(`, `modal_train_lane`, `modal run`) but
do NOT pair with canonical pairing tokens (`schedule_canonical_modal_harvest_cron`
/ `harvest_modal_calls`) AND lack same-line
`# HARVEST_SCHEDULER_PAIRED_OK:<rationale>` waiver.

Sister tests for:
- Surface A canonical helper at `tools/schedule_canonical_modal_harvest_cron.sh`
  (`src/tac/tests/test_schedule_canonical_modal_harvest_cron.py`)
- Sister Catalog #339 + #360 (canonical extinction at upstream surfaces).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_dispatch_wrappers_pair_with_harvest_scheduler_invocation,
    _check_380_body_has_pairing,
    _check_380_find_dispatch_trigger_lines,
    _check_380_line_has_waiver,
    _check_380_iter_scan_files,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- Helper unit tests -----------------------------------------------------


def test_body_with_pairing_token_accepted() -> None:
    """_check_380_body_has_pairing returns True when file references harvest_modal_calls."""
    body = "import x\n# ...\nfrom tac.something import harvest_modal_calls\n"
    assert _check_380_body_has_pairing(body) is True


def test_body_with_scheduler_token_accepted() -> None:
    """_check_380_body_has_pairing returns True when file references schedule_canonical_modal_harvest_cron."""
    body = "subprocess.run(['tools/schedule_canonical_modal_harvest_cron.sh', '--install'])\n"
    assert _check_380_body_has_pairing(body) is True


def test_body_without_pairing_token_rejected() -> None:
    """_check_380_body_has_pairing returns False when no canonical pairing tokens present."""
    body = "import x\nfn.spawn(arg=1)\n"
    assert _check_380_body_has_pairing(body) is False


# --- AST-aware Python dispatch detection ----------------------------------


def test_py_real_spawn_call_detected() -> None:
    """AST scan flags real .spawn(...) Call nodes."""
    body = "import x\ncall = fn.spawn(arg=1)\n"
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert len(hits) == 1
    assert hits[0][0] == 2


def test_py_docstring_mention_not_flagged() -> None:
    """Docstring mentions of .spawn() are NOT flagged (AST-aware)."""
    body = '"""Docstring mentioning .spawn() invocation pattern."""\nx = 1\n'
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert hits == []


def test_py_comment_mention_not_flagged() -> None:
    """Comment mentions of .spawn() are NOT flagged."""
    body = "# This comment mentions fn.spawn() and modal_train_lane.py\nx = 1\n"
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert hits == []


def test_py_string_literal_mention_not_flagged() -> None:
    """String-literal mentions of .spawn() are NOT flagged."""
    body = 'msg = "Modal .spawn() returned no function call id"\n'
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert hits == []


def test_py_subprocess_run_modal_train_lane_detected() -> None:
    """subprocess.run() with modal_train_lane.py first-arg is flagged."""
    body = (
        "import subprocess\n"
        "subprocess.run(['python', 'experiments/modal_train_lane.py', '--lane-id', 'x'])\n"
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert len(hits) == 1


def test_py_subprocess_popen_modal_train_lane_detected() -> None:
    """subprocess.Popen() with modal_train_lane.py first-arg is flagged."""
    body = (
        "import subprocess\n"
        "subprocess.Popen(['.venv/bin/python', 'experiments/modal_train_lane.py'])\n"
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert len(hits) == 1


def test_py_syntax_error_returns_empty() -> None:
    """Files with SyntaxError return empty trigger list (graceful fallback)."""
    body = "def broken(:\n    pass\n"
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=True)
    assert hits == []


# --- Shell dispatch detection ---------------------------------------------


def test_sh_modal_run_command_detected() -> None:
    """Real `modal run experiments/modal_train_lane.py` command is flagged."""
    body = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        ".venv/bin/modal run --detach experiments/modal_train_lane.py\n"
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=False)
    assert len(hits) == 1


def test_sh_python_modal_train_lane_command_detected() -> None:
    """Real `python experiments/modal_train_lane.py` command is flagged."""
    body = (
        "#!/usr/bin/env bash\n"
        "python experiments/modal_train_lane.py --lane-id foo\n"
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=False)
    assert len(hits) == 1


def test_sh_log_string_literal_mention_not_flagged() -> None:
    """Shell log/echo string-literal mention of modal_train_lane is NOT flagged."""
    body = (
        "#!/usr/bin/env bash\n"
        'log "modal_train_lane.py passes trainer_module_path=None"\n'
        'echo "experiments/modal_train_lane.py is the canonical dispatcher"\n'
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=False)
    assert hits == []


def test_sh_comment_mention_not_flagged() -> None:
    """Shell comment line mention of modal_train_lane is NOT flagged."""
    body = (
        "#!/usr/bin/env bash\n"
        "# This script wraps experiments/modal_train_lane.py for canary use.\n"
    )
    hits = _check_380_find_dispatch_trigger_lines(body, is_py=False)
    assert hits == []


# --- Waiver semantics -----------------------------------------------------


def test_waiver_with_substantive_rationale_accepted() -> None:
    """Same-line waiver with non-placeholder rationale (>= 4 chars) accepted."""
    line = (
        "fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:"
        "legacy_archive_smoke_path_paired_via_canonical_call_id_ledger_4layer_pattern"
    )
    assert _check_380_line_has_waiver(line) is True


def test_waiver_with_placeholder_rationale_rejected() -> None:
    """Same-line waiver with placeholder `<rationale>` literal rejected."""
    line = "fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:<rationale>"
    assert _check_380_line_has_waiver(line) is False


def test_waiver_with_placeholder_reason_rejected() -> None:
    """Same-line waiver with placeholder `<reason>` literal rejected."""
    line = "fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:<reason>"
    assert _check_380_line_has_waiver(line) is False


def test_waiver_with_short_rationale_rejected() -> None:
    """Same-line waiver with rationale < 4 chars rejected."""
    line = "fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:xy"
    assert _check_380_line_has_waiver(line) is False


def test_waiver_without_marker_rejected() -> None:
    """Line without the HARVEST_SCHEDULER_PAIRED_OK marker is not waived."""
    line = "fn.spawn(arg=1)  # some other comment"
    assert _check_380_line_has_waiver(line) is False


def test_waiver_with_empty_rationale_rejected() -> None:
    """Waiver marker with empty rationale rejected."""
    line = "fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:"
    assert _check_380_line_has_waiver(line) is False


# --- End-to-end gate behavior (synthetic) ---------------------------------


def test_synthetic_no_dispatch_passes(tmp_path: Path) -> None:
    """Synthetic repo with no dispatch wrappers produces 0 violations."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "noop.py").write_text("x = 1\n")
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_synthetic_dispatch_with_canonical_pairing_passes(tmp_path: Path) -> None:
    """Synthetic dispatch wrapper that references canonical pairing token passes."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "dispatcher.py").write_text(
        "# Pairs with canonical harvester via harvest_modal_calls.py\n"
        "import harvest_modal_calls\n"
        "call = fn.spawn(arg=1)\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_synthetic_dispatch_without_pairing_flagged(tmp_path: Path) -> None:
    """Synthetic dispatch wrapper without canonical pairing is flagged."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "bad_dispatcher.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "tools/bad_dispatcher.py" in violations[0]
    assert "Catalog #380" in violations[0]


def test_synthetic_dispatch_with_waiver_passes(tmp_path: Path) -> None:
    """Synthetic dispatch wrapper with same-line waiver passes."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "waived_dispatcher.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:"
        "legacy_archive_smoke_paired_via_external_harvest_actuator\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_synthetic_dispatch_with_placeholder_waiver_flagged(tmp_path: Path) -> None:
    """Synthetic dispatch wrapper with placeholder waiver still flagged."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "placeholder_dispatcher.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)  # HARVEST_SCHEDULER_PAIRED_OK:<rationale>\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


# --- Self-exempt cascade --------------------------------------------------


def test_canonical_helper_path_is_self_exempt(tmp_path: Path) -> None:
    """The canonical helper tools/harvest_modal_calls.py is self-exempt."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "harvest_modal_calls.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)\n"  # Would otherwise be flagged.
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_test_file_is_excluded(tmp_path: Path) -> None:
    """Test files (under tests/ or named test_*.py) are excluded from scan."""
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "tests" / "test_dispatcher.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_results_dir_is_excluded(tmp_path: Path) -> None:
    """experiments/results/ paths are excluded from scan."""
    (tmp_path / "experiments" / "results" / "lane_foo_modal").mkdir(parents=True)
    (tmp_path / "experiments" / "results" / "lane_foo_modal" / "build.py").write_text(
        "import x\n"
        "call = fn.spawn(arg=1)\n"
    )
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# --- Strict-mode behavior -------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on at least one violation."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "bad.py").write_text(
        "import x\ncall = fn.spawn(arg=1)\n"
    )
    with pytest.raises(PreflightError) as exc_info:
        check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #380" in str(exc_info.value) or "harvest_scheduler" in str(exc_info.value)


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode returns empty list (no raise) on clean repo."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "clean.py").write_text("x = 1\n")
    out = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=tmp_path, strict=True
    )
    assert out == []


# --- Live-repo regression guard (warn-only baseline) ---------------------


def test_live_repo_count_bounded_at_landing() -> None:
    """Live-repo regression guard: at landing the warn-only baseline is ~11.

    This test allows for slow expansion as new dispatch wrappers land
    (each new wrapper SHOULD either pair with the canonical scheduler or
    carry the waiver, but pre-existing wrappers may take time to backfill).

    Threshold set generously (<=25) per CLAUDE.md "Strict-flip atomicity
    rule" — the WARN-ONLY initial wire-in surfaces every existing
    offender at preflight time so the operator can backfill at
    operator-routable cap-windows. Strict-flip after backfill brings
    count to 0 (or to per-callsite waivers).
    """
    violations = check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(
        repo_root=REPO_ROOT, strict=False
    )
    # Allow up to 25 to accommodate slow backfill; landing baseline is ~11.
    assert len(violations) <= 25, (
        f"Catalog #380 live-repo count exceeded 25 (landing baseline ~11); "
        f"new dispatch wrappers must pair with canonical scheduler at "
        f"tools/schedule_canonical_modal_harvest_cron.sh OR carry "
        f"# HARVEST_SCHEDULER_PAIRED_OK:<rationale> waiver. Violations:\n  "
        + "\n  ".join(violations[:5])
    )


# --- Orchestrator wire-in regression guard --------------------------------


def test_orchestrator_callsite_is_warn_only_pending_backfill() -> None:
    """Orchestrator wires the gate WARN-ONLY (strict=False) pending backfill."""
    import inspect
    from tac.preflight import preflight_all

    source = inspect.getsource(preflight_all)
    # The orchestrator must invoke the gate with strict=False per
    # "Strict-flip atomicity rule" (live count 11 > 0 at landing).
    assert "check_dispatch_wrappers_pair_with_harvest_scheduler_invocation" in source


def test_orchestrator_callsite_uses_canonical_strict_false() -> None:
    """Orchestrator callsite uses strict=False (WARN-ONLY) as documented."""
    import inspect
    from tac.preflight import preflight_all

    source = inspect.getsource(preflight_all)
    # Find the line invoking the gate. The two surrounding lines should
    # contain `strict=False`.
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if "check_dispatch_wrappers_pair_with_harvest_scheduler_invocation(" in line:
            # The strict=False arg may be on the SAME line or the NEXT line.
            window = "\n".join(lines[i:i + 4])
            assert "strict=False" in window, (
                f"Catalog #380 orchestrator callsite should use strict=False per "
                f"Strict-flip atomicity rule; got:\n{window}"
            )
            return
    pytest.fail("Catalog #380 not found in preflight_all orchestrator")


# --- Catalog #185 sister regression guard (live count zero verification) -


def test_gate_function_callable_via_globals_per_catalog_185() -> None:
    """Catalog #185 META-meta-meta requires the gate be callable via globals()."""
    import tac.preflight as preflight_mod

    fn = getattr(preflight_mod, "check_dispatch_wrappers_pair_with_harvest_scheduler_invocation", None)
    assert fn is not None
    assert callable(fn)


# --- File-system scan helper test ----------------------------------------


def test_iter_scan_files_yields_under_canonical_dirs(tmp_path: Path) -> None:
    """_check_380_iter_scan_files yields .py + .sh under tools/scripts/experiments/src/tac."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "foo.py").write_text("x = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bar.sh").write_text("#!/bin/sh\n")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "baz.py").write_text("y = 2\n")
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "qux.py").write_text("z = 3\n")
    paths = list(_check_380_iter_scan_files(tmp_path))
    rels = {rel for rel, _ in paths}
    # Self-exempt paths excluded; src/tac/preflight.py is in the exempt list.
    assert "tools/foo.py" in rels
    assert "scripts/bar.sh" in rels
    assert "experiments/baz.py" in rels
    assert "src/tac/qux.py" in rels
