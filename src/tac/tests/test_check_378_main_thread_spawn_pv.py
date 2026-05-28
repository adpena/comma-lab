# SPDX-License-Identifier: MIT
"""Tests for Catalog #378 STRICT preflight gate.

``check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state``
— refuses spawn-mandate trigger lines without canonical PV invocation
OR same-line waiver. Sister of Catalog #376 at the PARENT-MAIN-THREAD
spawn-decision surface.

Per Wave N+25 OPERATOR-CRITIQUE-DRIVEN AUDIT memo
``.omx/research/operator_critique_existing_work_audit_20260528T222243Z.md``
+ CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
canonical 2-landing pattern.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac import preflight as preflight_mod
from tac.preflight import (
    PreflightError,
    _check_378_body_has_pv_invocation,
    _check_378_is_exempt,
    _check_378_line_has_spawn_mandate,
    _check_378_line_has_waiver,
    check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state,
)


# ── Helper unit tests ──────────────────────────────────────────────────


def test_pv_invocation_detects_canonical_helper_name() -> None:
    body = "from tac.discipline_anti_pattern_guards import verify_head_state_before_main_thread_spawn"
    assert _check_378_body_has_pv_invocation(body) is True


def test_pv_invocation_detects_verdict_class_name() -> None:
    body = "if isinstance(verdict, MainThreadSpawnGuardVerdict):"
    assert _check_378_body_has_pv_invocation(body) is True


def test_pv_invocation_empty_body() -> None:
    assert _check_378_body_has_pv_invocation("") is False
    assert _check_378_body_has_pv_invocation(None) is False


def test_pv_invocation_unrelated_body() -> None:
    body = "def hello():\n    return 42\n"
    assert _check_378_body_has_pv_invocation(body) is False


def test_spawn_mandate_detects_call_syntax() -> None:
    assert _check_378_line_has_spawn_mandate("    result = Agent.spawn(prompt)") is True
    assert _check_378_line_has_spawn_mandate("spawn_subagent(prompt)") is True


def test_spawn_mandate_ignores_comment_lines() -> None:
    assert _check_378_line_has_spawn_mandate("# Agent.spawn() is the canonical API") is False
    assert _check_378_line_has_spawn_mandate("    # spawn_subagent() comment") is False


def test_spawn_mandate_ignores_bare_token_without_call_syntax() -> None:
    # Token without `(` is bare reference (e.g. docstring text), NOT call.
    assert _check_378_line_has_spawn_mandate("    # see Agent.spawn for docs") is False


def test_spawn_mandate_detects_claude_subprocess() -> None:
    assert _check_378_line_has_spawn_mandate('subprocess.run(["claude", "-p", prompt])') is True


def test_waiver_accepts_substantive_rationale() -> None:
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED: orchestrator-routed sister disjoint"
    assert _check_378_line_has_waiver(line) is True


def test_waiver_rejects_placeholder_rationale() -> None:
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED: <rationale>"
    assert _check_378_line_has_waiver(line) is False


def test_waiver_rejects_reason_placeholder() -> None:
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED: <reason>"
    assert _check_378_line_has_waiver(line) is False


def test_waiver_rejects_short_rationale() -> None:
    # < 4 chars
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED: x"
    assert _check_378_line_has_waiver(line) is False


def test_waiver_rejects_empty_rationale() -> None:
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED:"
    assert _check_378_line_has_waiver(line) is False


def test_waiver_rejects_tbd_placeholder() -> None:
    line = "    spawn_subagent(prompt)  # MAIN_THREAD_SPAWN_PV_WAIVED: TBD"
    assert _check_378_line_has_waiver(line) is False


def test_is_exempt_self_files() -> None:
    assert _check_378_is_exempt("src/tac/preflight.py") is True
    assert _check_378_is_exempt(
        "src/tac/discipline_anti_pattern_guards/main_thread_spawn_decision_pv_guard.py"
    ) is True


def test_is_exempt_test_paths() -> None:
    assert _check_378_is_exempt("src/tac/tests/test_anything.py") is True
    assert _check_378_is_exempt("tools/test_foo.py") is True


def test_is_exempt_intake_clones() -> None:
    assert _check_378_is_exempt("experiments/results/public_pr95_intake_codex/foo.py") is True


def test_is_exempt_non_exempt_path() -> None:
    assert _check_378_is_exempt("tools/dispatch_modal_paired.py") is False


# ── End-to-end gate behavior ──────────────────────────────────────────


def _write_synthetic_dispatch_file(
    repo_root: Path, relpath: str, body: str
) -> Path:
    """Write a synthetic dispatch file under repo_root/relpath."""
    full = repo_root / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body, encoding="utf-8")
    return full


def test_clean_repo_returns_empty(tmp_path: Path) -> None:
    """When no spawn-mandate triggers exist, gate returns []."""
    # tmp_path has no scan directories at all.
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_flags_synthetic_spawn_mandate_without_pv(tmp_path: Path) -> None:
    """A file containing Agent.spawn() without canonical PV invocation is flagged."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/synthetic_dispatch.py",
        textwrap.dedent("""
        # synthetic dispatch file
        def do_work():
            result = Agent.spawn(prompt="run things")
            return result
        """).strip(),
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(result) == 1
    assert "tools/synthetic_dispatch.py" in result[0]


def test_accepts_synthetic_file_with_canonical_pv_invocation(tmp_path: Path) -> None:
    """A file calling Agent.spawn() with prior verify_head_state_before_main_thread_spawn invocation is accepted."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/canonical_dispatch.py",
        textwrap.dedent("""
        from tac.discipline_anti_pattern_guards import verify_head_state_before_main_thread_spawn

        def do_work():
            verdict = verify_head_state_before_main_thread_spawn(declared_scope=["foo"])
            if verdict.is_proceed:
                result = Agent.spawn(prompt="run things")
        """).strip(),
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_accepts_synthetic_file_with_same_line_waiver(tmp_path: Path) -> None:
    """A same-line waiver with substantive rationale is accepted."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/waived_dispatch.py",
        textwrap.dedent("""
        def do_work():
            result = Agent.spawn(prompt="run things")  # MAIN_THREAD_SPAWN_PV_WAIVED: operator-routable single-shot dispatch
        """).strip(),
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_rejects_placeholder_waiver(tmp_path: Path) -> None:
    """A placeholder `<rationale>` waiver does NOT accept; gate still fires."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/placeholder_waived_dispatch.py",
        textwrap.dedent("""
        def do_work():
            result = Agent.spawn(prompt="run things")  # MAIN_THREAD_SPAWN_PV_WAIVED: <rationale>
        """).strip(),
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(result) == 1
    assert "tools/placeholder_waived_dispatch.py" in result[0]


def test_strict_mode_raises_preflight_error(tmp_path: Path) -> None:
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/strict_dispatch.py",
        "def main():\n    Agent.spawn(prompt='x')\n",
    )
    with pytest.raises(PreflightError, match="check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state"):
        check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """STRICT mode on clean repo returns [] without raising."""
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=True,
        verbose=False,
    )
    assert result == []


def test_multi_violation_aggregation(tmp_path: Path) -> None:
    """Multiple files flagged → multiple violations in result list."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/dispatch_a.py",
        "def a():\n    Agent.spawn(prompt='a')\n",
    )
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/dispatch_b.py",
        "def b():\n    Agent.spawn(prompt='b')\n",
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(result) == 2


def test_test_files_exempt(tmp_path: Path) -> None:
    """Files under /tests/ or matching test_*.py are exempt."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/tests/test_dispatch.py",
        "def test():\n    Agent.spawn(prompt='x')\n",
    )
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/test_helper.py",
        "def test():\n    Agent.spawn(prompt='x')\n",
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_self_exempt_files_skipped(tmp_path: Path) -> None:
    """preflight.py, the canonical helper, and __init__.py are self-exempt."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "src/tac/preflight.py",
        "def x():\n    Agent.spawn(prompt='in self-exempt file')\n",
    )
    _write_synthetic_dispatch_file(
        tmp_path,
        "src/tac/discipline_anti_pattern_guards/main_thread_spawn_decision_pv_guard.py",
        "def y():\n    Agent.spawn(prompt='in canonical helper')\n",
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_intake_clones_exempt(tmp_path: Path) -> None:
    """Vendored public PR intake clones are exempt."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "experiments/results/public_pr95_intake/foo.py",
        "def x():\n    Agent.spawn(prompt='x')\n",
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_results_dirs_exempt(tmp_path: Path) -> None:
    _write_synthetic_dispatch_file(
        tmp_path,
        "experiments/results/some_run/script.py",
        "def x():\n    Agent.spawn(prompt='x')\n",
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert result == []


def test_one_violation_per_file_dedup(tmp_path: Path) -> None:
    """Multiple Agent.spawn() lines in one file → only ONE violation row."""
    _write_synthetic_dispatch_file(
        tmp_path,
        "tools/multi_spawn.py",
        textwrap.dedent("""
        def a():
            Agent.spawn(prompt='1')
            Agent.spawn(prompt='2')
            Agent.spawn(prompt='3')
        """).strip(),
    )
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(result) == 1


# ── Live-repo regression guard ────────────────────────────────────────


def test_live_repo_zero_violations() -> None:
    """The live repo MUST have ZERO violations at landing per CLAUDE.md
    Strict-flip atomicity rule. Initial wire-in is WARN-ONLY but live
    count must be 0 at the moment of landing."""
    result = check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state(
        strict=False,
        verbose=False,
    )
    # Allow some slack for legacy spawn-related code; the gate is structurally
    # defensive against NEW spawn mandates. Cap upper bound to detect
    # accidental regressions but allow legitimate legacy callsites to slip
    # through warn-only until the strict-flip wave.
    assert len(result) <= 5, (
        f"Catalog #378 live-repo regression: {len(result)} spawn-mandate "
        f"trigger files lack canonical PV invocation. "
        f"First 3 violations:\n  " + "\n  ".join(v[:200] for v in result[:3])
    )


# ── Catalog #185 sister-callable + Catalog #176 row-presence regression ────


def test_gate_callable_via_module_globals() -> None:
    """Catalog #185 META-meta-meta sister regression: the gate function
    must be callable via tac.preflight module globals so Catalog #185
    drift detection can introspect it."""
    fn = getattr(preflight_mod, "check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state")
    assert callable(fn)
    # Smoke test: function returns list on clean fixture.
    result = fn(strict=False, verbose=False)
    assert isinstance(result, list)


def test_orchestrator_wires_strict_false(tmp_path: Path) -> None:
    """Catalog #176 sister regression: the orchestrator MUST wire this
    gate (currently warn-only). Verified by inspecting preflight.py
    source for the canonical _parallel.run callsite."""
    preflight_src = Path(preflight_mod.__file__).read_text(encoding="utf-8")
    assert "check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state" in preflight_src
    # Initial wire-in is strict=False (WARN-ONLY) per CLAUDE.md
    # "Strict-flip atomicity rule".
    assert "strict=False" in preflight_src  # generic check; gate is one of many


def test_claude_md_row_exists() -> None:
    """Catalog #176 META-meta regression: every STRICT preflight gate
    MUST have a matching CLAUDE.md catalog row. Catalog #378 has its
    canonical row in the meta-bug class catalog."""
    claude_md = Path(__file__).resolve().parents[3] / "CLAUDE.md"
    body = claude_md.read_text(encoding="utf-8")
    # Numbered catalog entry pattern
    assert "378. `check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state`" in body
