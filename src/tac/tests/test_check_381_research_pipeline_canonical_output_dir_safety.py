# SPDX-License-Identifier: MIT
"""Tests for Catalog #381 — research-pipeline canonical output-dir safety gate.

Sister of canonical helper ``tac.research_pipeline_output_dir_safety`` per
the canonical 2-landing pattern; the canonical helper has its own
35-test coverage at
``src/tac/research_pipeline_output_dir_safety/tests/test_output_dir_safety.py``.
THESE tests cover the STRICT preflight gate side of the pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper,
)


def _write_synthetic_tool(
    repo_root: Path, rel: str, *, with_canonical_token: bool, with_waiver: bool = False,
    waiver_rationale: str = "explicit-operator-approved-overwrite",
) -> Path:
    p = repo_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if with_canonical_token:
        body = '''"""Tool."""
from tac.research_pipeline_output_dir_safety import enforce_research_pipeline_output_dir
def main():
    enforce_research_pipeline_output_dir(out, repo_root=REPO_ROOT)
'''
    elif with_waiver:
        body = f'''"""Tool."""  # RESEARCH_PIPELINE_OUTPUT_DIR_SAFETY_WAIVED:{waiver_rationale}
def main():
    pass
'''
    else:
        # Bare write-pattern with no canonical helper
        body = '''"""Tool."""
import json
def main():
    out = "/tmp/foo.json"
    json.dump({}, open(out, "w"))
'''
    p.write_text(body)
    return p


# ----- live-repo regression guard -----

def test_live_repo_passes_strict_gate() -> None:
    """At landing time, all 3 tools must carry the canonical helper invocation."""

    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False
        )
    )
    assert violations == [], (
        f"live repo regressed Catalog #381: {len(violations)} violation(s):\n"
        + "\n".join(violations[:5])
    )


# ----- synthetic violation surface -----

def test_synthetic_missing_canonical_helper_flagged(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
    )
    _write_synthetic_tool(
        repo,
        "tools/run_repair_autonomous_multi_archive_runner.py",
        with_canonical_token=True,
    )
    _write_synthetic_tool(
        repo,
        "tools/build_frontier_final_rate_attack_queue.py",
        with_canonical_token=True,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert len(violations) == 1
    assert "run_pr95_local_training_probe.py" in violations[0]


def test_all_3_tools_canonical_accepted(tmp_path: Path) -> None:
    repo = tmp_path
    for tool_rel in (
        "tools/run_pr95_local_training_probe.py",
        "tools/run_repair_autonomous_multi_archive_runner.py",
        "tools/build_frontier_final_rate_attack_queue.py",
    ):
        _write_synthetic_tool(repo, tool_rel, with_canonical_token=True)
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert violations == []


def test_missing_tool_silent(tmp_path: Path) -> None:
    """If a tool is not present (e.g. removed), gate silently skips."""

    repo = tmp_path
    # No tools created
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert violations == []


# ----- waiver semantics -----

def test_waiver_with_substantive_rationale_accepted(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
        with_waiver=True,
        waiver_rationale="legacy-tool-deferred-canonical-migration-pending-operator-review",
    )
    _write_synthetic_tool(
        repo,
        "tools/run_repair_autonomous_multi_archive_runner.py",
        with_canonical_token=True,
    )
    _write_synthetic_tool(
        repo,
        "tools/build_frontier_final_rate_attack_queue.py",
        with_canonical_token=True,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert violations == []


def test_placeholder_rationale_rejected_per_catalog_287(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
        with_waiver=True,
        waiver_rationale="<rationale>",
    )
    _write_synthetic_tool(
        repo,
        "tools/run_repair_autonomous_multi_archive_runner.py",
        with_canonical_token=True,
    )
    _write_synthetic_tool(
        repo,
        "tools/build_frontier_final_rate_attack_queue.py",
        with_canonical_token=True,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert len(violations) == 1


def test_short_rationale_rejected(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
        with_waiver=True,
        waiver_rationale="xy",  # <4 chars
    )
    _write_synthetic_tool(
        repo,
        "tools/run_repair_autonomous_multi_archive_runner.py",
        with_canonical_token=True,
    )
    _write_synthetic_tool(
        repo,
        "tools/build_frontier_final_rate_attack_queue.py",
        with_canonical_token=True,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert len(violations) == 1


# ----- strict mode -----

def test_strict_mode_raises_PreflightError(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=True, verbose=False, repo_root=repo
        )
    err = str(exc_info.value)
    assert "Catalog #381" not in err  # gate uses function name in message
    assert (
        "check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper"
        in err
    )


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    repo = tmp_path
    # No tools created -> silent
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=True, verbose=False, repo_root=repo
        )
    )
    assert violations == []


# ----- diagnostic message quality -----

def test_violation_message_includes_canonical_anti_pattern_id(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert any(
        "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1"
        in v
        for v in violations
    )


def test_violation_message_includes_77_anchor_count(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert any("77 in-place field mutations" in v for v in violations)


def test_violation_message_includes_operator_routable_unwind(tmp_path: Path) -> None:
    repo = tmp_path
    _write_synthetic_tool(
        repo,
        "tools/run_pr95_local_training_probe.py",
        with_canonical_token=False,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert any("Operator-routable unwind" in v for v in violations)


# ----- helper detection coverage -----

def test_validate_research_pipeline_output_dir_token_accepted(tmp_path: Path) -> None:
    """Validator-only callsite (not enforce) is also acceptable per canonical set."""

    repo = tmp_path
    body = '''"""Tool."""
from tac.research_pipeline_output_dir_safety import validate_research_pipeline_output_dir
def main():
    v = validate_research_pipeline_output_dir(out, repo_root=REPO_ROOT)
'''
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "run_pr95_local_training_probe.py").write_text(body)
    _write_synthetic_tool(
        repo,
        "tools/run_repair_autonomous_multi_archive_runner.py",
        with_canonical_token=True,
    )
    _write_synthetic_tool(
        repo,
        "tools/build_frontier_final_rate_attack_queue.py",
        with_canonical_token=True,
    )
    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False, repo_root=repo
        )
    )
    assert violations == []


# ----- orchestrator + Catalog #185 sister-callable -----

def test_orchestrator_wires_strict_true() -> None:
    """Catalog #176 META-meta: callsite in preflight_all is strict=True."""

    import inspect

    from tac import preflight

    source = inspect.getsource(preflight.preflight_all)
    # The wire-in is strict=True (live count 0 at landing per canonical 2-landing pattern)
    assert (
        "check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper"
        in source
    )


def test_catalog_185_sister_callable_via_globals() -> None:
    """Catalog #185 META-meta-meta: gate function must be callable via globals."""

    from tac import preflight

    assert hasattr(
        preflight,
        "check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper",
    )
    func = getattr(
        preflight,
        "check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper",
    )
    assert callable(func)
    # Default args (strict=False) should not raise on clean repo
    violations = func(strict=False, verbose=False)
    assert isinstance(violations, list)


def test_signature_kwargs_only() -> None:
    """Function uses kwargs-only contract for compositional safety."""

    import inspect

    from tac.preflight import (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper,
    )

    sig = inspect.signature(
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper
    )
    # All params should be kwargs-only (after the *,)
    for name, param in sig.parameters.items():
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ), f"param {name!r} kind={param.kind} (expected KEYWORD_ONLY)"
