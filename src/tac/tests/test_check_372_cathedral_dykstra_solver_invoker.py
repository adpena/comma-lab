# SPDX-License-Identifier: MIT
"""Tests for Catalog #372 STRICT preflight gate (Dykstra Pareto solver invoker)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cathedral_autopilot_main_invokes_dykstra_pareto_solver,
)


def _write_target(repo_root: Path, body: str) -> Path:
    """Write a synthetic cathedral_autopilot_autonomous_loop.py with given main body."""
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    target = tools_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(body, encoding="utf-8")
    return target


def test_live_repo_zero_violations():
    """Live repo state must have 0 violations (canonical wire-in is present)."""
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(strict=False)
    assert violations == [], (
        f"Catalog #372 live repo regression: {len(violations)} violations: "
        + "\n".join(violations[:3])
    )


def test_no_invoker_flagged(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):
            print('no Dykstra invoker')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1
    assert "main() does NOT invoke" in violations[0]
    assert "invoke_dykstra_pareto_solver_on_candidates" in violations[0]


def test_invoke_dykstra_pareto_solver_token_accepted(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):
            invoke_dykstra_pareto_solver_on_candidates([], top_n=10)
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_solve_pareto_polytope_intersection_token_accepted(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):
            verdict = solve_pareto_polytope_intersection(polytope, initial_point={}, candidate_id='x')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_waiver_with_real_rationale_accepted(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):  # DYKSTRA_PARETO_SOLVER_INVOKER_WAIVED:test reason for waiver
            print('no invoker but waived')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_placeholder_rationale_rejected(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):  # DYKSTRA_PARETO_SOLVER_INVOKER_WAIVED:<rationale>
            print('no invoker; placeholder waiver should be rejected')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1


def test_reason_placeholder_rejected(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):  # DYKSTRA_PARETO_SOLVER_INVOKER_WAIVED:<reason>
            print('no invoker; <reason> placeholder rejected')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1


def test_short_rationale_rejected(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):  # DYKSTRA_PARETO_SOLVER_INVOKER_WAIVED:no
            print('rationale too short (<4 chars) rejected')
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1


def test_strict_raises_preflight_error_on_violation(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):
            print('no Dykstra invoker; strict mode should raise')
            return 0
    """)
    _write_target(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #372" in str(exc_info.value)
    assert "Meta-Lagrangian/Pareto solver" in str(exc_info.value)


def test_strict_silent_on_clean(tmp_path):
    body = textwrap.dedent("""\
        def main(argv=None):
            invoke_dykstra_pareto_solver_on_candidates([], top_n=10)
            return 0
    """)
    _write_target(tmp_path, body)
    # Should not raise.
    result = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=True
    )
    assert result == []


def test_missing_target_silent(tmp_path):
    # No tools/cathedral_autopilot_autonomous_loop.py at all.
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_missing_main_flagged(tmp_path):
    body = textwrap.dedent("""\
        def some_other_function():
            print('no main() function at all')
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1
    assert "missing def main" in violations[0] or "no top-level def main" in violations[0]


def test_attribute_call_form_accepted(tmp_path):
    """If the invoker is called via attribute access, the gate should still match."""
    body = textwrap.dedent("""\
        def main(argv=None):
            module.invoke_dykstra_pareto_solver_on_candidates([], top_n=10)
            return 0
    """)
    _write_target(tmp_path, body)
    violations = check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_orchestrator_wire_in_strict_true_regression():
    """Verify preflight_all() wires Catalog #372 with strict=True."""
    import inspect
    from tac import preflight
    source = inspect.getsource(preflight.preflight_all)
    assert "check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(" in source
    # Confirm strict=True is passed.
    assert (
        "check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(\n"
        "            strict=True, verbose=verbose" in source
    ) or (
        "check_cathedral_autopilot_main_invokes_dykstra_pareto_solver(strict=True" in source
    )


def test_catalog_185_sister_callable_regression():
    """Verify Catalog #185 sister gate can find Catalog #372 via globals."""
    from tac import preflight
    assert hasattr(preflight, "check_cathedral_autopilot_main_invokes_dykstra_pareto_solver")
    assert callable(preflight.check_cathedral_autopilot_main_invokes_dykstra_pareto_solver)


def test_acceptance_tokens_pinned():
    """Token set is the canonical contract."""
    from tac.preflight import _CHECK_372_ACCEPTANCE_TOKENS
    assert "invoke_dykstra_pareto_solver_on_candidates" in _CHECK_372_ACCEPTANCE_TOKENS
    assert "solve_pareto_polytope_intersection" in _CHECK_372_ACCEPTANCE_TOKENS


def test_waiver_token_pinned():
    from tac.preflight import _CHECK_372_WAIVER_TOKEN
    assert _CHECK_372_WAIVER_TOKEN == "DYKSTRA_PARETO_SOLVER_INVOKER_WAIVED"


def test_placeholder_rationales_pinned():
    from tac.preflight import _CHECK_372_PLACEHOLDER_RATIONALES
    assert "<rationale>" in _CHECK_372_PLACEHOLDER_RATIONALES
    assert "<reason>" in _CHECK_372_PLACEHOLDER_RATIONALES
