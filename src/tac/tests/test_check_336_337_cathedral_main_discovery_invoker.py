# SPDX-License-Identifier: MIT
"""Tests for Catalog #336 + #337 — cathedral main() invokes discovery + master-gradient rerank.

R11-H1-1-PLUS-H1-6-FIX-WAVE landing 2026-05-19 per Cable H1 R11 adversarial
review (commit 725699560). Sister of Catalog #335 (upstream auto-discovery
surface) at the downstream INVOCATION surface.

The Assumption-Adversary R11 verdict: "Convention-over-configuration
auto-discovery is sufficient to extinct the orphan-signal class" is
CARGO-CULTED-EMPIRICALLY-FALSIFIED. The auto-discovery loop is a tested
helper without a runtime invoker. THIS FIX-WAVE lands the invoker
callsite in main() + the two STRICT preflight gates (Catalog #336 + #337)
that prevent regression.

Test coverage:
- Live-repo regression guard for both gates (0 violations at landing)
- Synthetic violation: main() without invoker call
- Synthetic acceptance: main() with each acceptance token
- Waiver mechanism with placeholder rejection
- Strict-mode behavior (raises vs silent)
- Orchestrator wire-in (preflight_all) regression guard
- Catalog #185 sister regression (gate function callable via globals)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cathedral_autopilot_main_invokes_discover_and_register_consumers,
    check_rerank_candidates_via_master_gradient_invokes_consumers,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard (the gates' primary structural invariant)
# ---------------------------------------------------------------------------


def test_live_repo_336_passes_clean() -> None:
    """Live repo Catalog #336: invoker callsite present (R11 H1-1 fix landed)."""
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers()
    assert violations == [], (
        f"Live repo Catalog #336 violations:\n  - "
        + "\n  - ".join(v[:200] for v in violations[:3])
    )


def test_live_repo_337_passes_clean() -> None:
    """Live repo Catalog #337: master-gradient rerank invoker callsite present."""
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers()
    assert violations == [], (
        f"Live repo Catalog #337 violations:\n  - "
        + "\n  - ".join(v[:200] for v in violations[:3])
    )


def test_live_repo_336_strict_does_not_raise() -> None:
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        strict=True
    )
    assert violations == []


def test_live_repo_337_strict_does_not_raise() -> None:
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        strict=True
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _write_synthetic_target(
    repo: Path,
    main_body: str,
    *,
    waiver_on_main_line: str = "",
) -> Path:
    """Build a synthetic tools/cathedral_autopilot_autonomous_loop.py file."""
    target = repo / "tools" / "cathedral_autopilot_autonomous_loop.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    main_line = "def main(argv=None):"
    if waiver_on_main_line:
        main_line = f"{main_line}  # {waiver_on_main_line}"
    indented_body = "\n".join("    " + ln for ln in main_body.splitlines())
    src = (
        "# SPDX-License-Identifier: MIT\n"
        '"""Synthetic cathedral autopilot stub for Catalog #336/#337 testing."""\n'
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def discover_and_register_consumers(*args, **kwargs):\n"
        "    return []\n"
        "\n"
        "\n"
        "def discover_compliant_consumer_modules(*args, **kwargs):\n"
        "    return []\n"
        "\n"
        "\n"
        "def rerank_candidates_via_master_gradient(*args, **kwargs):\n"
        "    return []\n"
        "\n"
        "\n"
        "def invoke_cathedral_consumers_on_candidates(*args, **kwargs):\n"
        '    return {"schema": "test"}\n'
        "\n"
        "\n"
        f"{main_line}\n"
        f"{indented_body}\n"
        "    return 0\n"
    )
    target.write_text(src, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Catalog #336 — discovery invoker tests
# ---------------------------------------------------------------------------


def test_336_no_invoker_call_flagged(tmp_path: Path) -> None:
    """main() with no acceptance token call must be flagged."""
    _write_synthetic_target(tmp_path, "x = 1\nprint(x)")
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "does NOT invoke any of" in violations[0]


def test_336_accepts_invoke_helper(tmp_path: Path) -> None:
    """main() calling invoke_cathedral_consumers_on_candidates must pass."""
    _write_synthetic_target(
        tmp_path,
        "result = invoke_cathedral_consumers_on_candidates([])",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_336_accepts_discover_and_register(tmp_path: Path) -> None:
    """main() calling discover_and_register_consumers directly must pass."""
    _write_synthetic_target(
        tmp_path,
        "regs = discover_and_register_consumers()",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_336_accepts_discover_compliant_consumer_modules(tmp_path: Path) -> None:
    """main() calling discover_compliant_consumer_modules must pass."""
    _write_synthetic_target(
        tmp_path,
        "mods = discover_compliant_consumer_modules()",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_336_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Same-line waiver with non-placeholder rationale must short-circuit."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="CATHEDRAL_MAIN_DISCOVERY_INVOKER_WAIVED:test fixture path",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_336_waiver_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Placeholder rationale rejected (the gate's docstring cannot self-waive)."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="CATHEDRAL_MAIN_DISCOVERY_INVOKER_WAIVED:<rationale>",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_336_strict_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode must raise PreflightError on violation."""
    _write_synthetic_target(tmp_path, "pass")
    with pytest.raises(PreflightError, match="Catalog #336"):
        check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
            repo_root=tmp_path, strict=True
        )


def test_336_strict_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode must NOT raise when clean."""
    _write_synthetic_target(
        tmp_path,
        "x = invoke_cathedral_consumers_on_candidates([])",
    )
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_336_missing_target_file_silent(tmp_path: Path) -> None:
    """Target file missing → silent skip (no violation, no raise)."""
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_336_main_function_missing_flagged(tmp_path: Path) -> None:
    """Target file present but no def main() → violation."""
    target = tmp_path / "tools" / "cathedral_autopilot_autonomous_loop.py"
    target.parent.mkdir(parents=True)
    target.write_text("# no main function here\nx = 1\n")
    violations = check_cathedral_autopilot_main_invokes_discover_and_register_consumers(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "missing def main" in violations[0] or "no top-level def main" in violations[0]


# ---------------------------------------------------------------------------
# Catalog #337 — master-gradient rerank invoker tests
# ---------------------------------------------------------------------------


def test_337_no_invoker_call_flagged(tmp_path: Path) -> None:
    """main() with no master-gradient acceptance token call must be flagged."""
    _write_synthetic_target(tmp_path, "x = 1\nprint(x)")
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "does NOT invoke any of" in violations[0]


def test_337_accepts_invoke_helper(tmp_path: Path) -> None:
    """main() calling invoke_cathedral_consumers_on_candidates must pass.

    (The helper internally invokes rerank_candidates_via_master_gradient.)
    """
    _write_synthetic_target(
        tmp_path,
        "result = invoke_cathedral_consumers_on_candidates([])",
    )
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_337_accepts_direct_rerank(tmp_path: Path) -> None:
    """main() calling rerank_candidates_via_master_gradient directly must pass."""
    _write_synthetic_target(
        tmp_path,
        "ranked = rerank_candidates_via_master_gradient([])",
    )
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_337_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Same-line waiver with non-placeholder rationale must short-circuit."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="RERANK_MASTER_GRADIENT_INVOKER_WAIVED:test fixture path",
    )
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        repo_root=tmp_path
    )
    assert violations == []


def test_337_strict_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode must raise PreflightError on violation."""
    _write_synthetic_target(tmp_path, "pass")
    with pytest.raises(PreflightError, match="Catalog #337"):
        check_rerank_candidates_via_master_gradient_invokes_consumers(
            repo_root=tmp_path, strict=True
        )


def test_337_missing_target_file_silent(tmp_path: Path) -> None:
    violations = check_rerank_candidates_via_master_gradient_invokes_consumers(
        repo_root=tmp_path
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard (Catalog #176 sister)
# ---------------------------------------------------------------------------


def test_336_orchestrator_callsite_wired() -> None:
    """preflight_all wires #336 (WARN-ONLY at landing per Strict-flip atomicity)."""
    preflight_source = Path(__file__).parent.parent / "preflight.py"
    text = preflight_source.read_text(encoding="utf-8")
    assert "check_cathedral_autopilot_main_invokes_discover_and_register_consumers" in text
    # Confirm it appears in the orchestrator with strict=False (warn-only at landing).
    assert (
        "check_cathedral_autopilot_main_invokes_discover_and_register_consumers(\n"
        "            strict=False"
    ) in text, (
        "Catalog #336 must be wired into preflight_all with strict=False at landing"
    )


def test_337_orchestrator_callsite_wired() -> None:
    """preflight_all wires #337 (WARN-ONLY at landing per Strict-flip atomicity)."""
    preflight_source = Path(__file__).parent.parent / "preflight.py"
    text = preflight_source.read_text(encoding="utf-8")
    assert "check_rerank_candidates_via_master_gradient_invokes_consumers" in text
    assert (
        "check_rerank_candidates_via_master_gradient_invokes_consumers(\n"
        "            strict=False"
    ) in text


# ---------------------------------------------------------------------------
# Catalog #185 sister regression — gate functions callable via globals
# ---------------------------------------------------------------------------


def test_336_337_callable_via_globals() -> None:
    """Both gates discoverable in tac.preflight module globals."""
    import tac.preflight as preflight_mod
    assert callable(getattr(
        preflight_mod,
        "check_cathedral_autopilot_main_invokes_discover_and_register_consumers",
        None,
    ))
    assert callable(getattr(
        preflight_mod,
        "check_rerank_candidates_via_master_gradient_invokes_consumers",
        None,
    ))


# ---------------------------------------------------------------------------
# Helper function tests: invoke_cathedral_consumers_on_candidates
# ---------------------------------------------------------------------------


def test_invoke_helper_returns_canonical_schema() -> None:
    """The R11 helper returns the canonical schema + observability-only fields."""
    # Import via sys.path manipulation since tools/ is not a package.
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tools_dir = str(repo_root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import cathedral_autopilot_autonomous_loop as loop

    # Empty candidate list — no consumers invoked but schema present.
    result = loop.invoke_cathedral_consumers_on_candidates(
        [], top_n=5, repo_root=repo_root, panel_axis="contest_cpu",
    )
    assert result["schema"] == "cathedral_consumer_invocation_v1_20260519"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["evidence_grade"] == "[predicted, cathedral consumer invocation]"
    assert isinstance(result["consumer_names"], list)
    assert isinstance(result["invocations"], list)
    assert isinstance(result["master_gradient_annotations"], list)
    # Live repo should have discovered some consumers.
    assert result["consumer_count"] > 0


def test_invoke_helper_with_synthetic_candidate() -> None:
    """Helper invokes consume_candidate per discovered consumer per candidate."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tools_dir = str(repo_root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import cathedral_autopilot_autonomous_loop as loop

    # Build a minimal CandidateRow.
    candidate = loop.CandidateRow(
        candidate_id="test_invoker_smoke",
        family="test_family",
        predicted_score_delta=-0.001,
        estimated_dispatch_cost_usd=0.10,
        expected_information_gain=0.5,
        archive_sha256="0" * 64,
    )
    result = loop.invoke_cathedral_consumers_on_candidates(
        [candidate], top_n=1, repo_root=repo_root, panel_axis="contest_cpu",
    )
    assert result["candidates_invoked"] == 1
    # Each discovered consumer × candidate produces one invocation row.
    assert len(result["invocations"]) == result["consumer_count"]
    # Each invocation must have observability-only fields.
    for inv in result["invocations"]:
        # Either succeeded (has these keys) or surfaced error.
        if "error" not in inv:
            assert "predicted_delta_adjustment" in inv
            assert "axis_tag" in inv
            assert inv.get("promotable") is False, (
                "consumer contributions MUST be observability-only "
                "per Catalog #287/#323 — no promotion authority"
            )
    # Master-gradient annotations also generated.
    assert result["master_gradient_rerank_invoked"] is True
    assert len(result["master_gradient_annotations"]) >= 1


def test_invoke_helper_include_master_gradient_false() -> None:
    """Helper allows opt-out of master-gradient rerank."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tools_dir = str(repo_root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import cathedral_autopilot_autonomous_loop as loop

    result = loop.invoke_cathedral_consumers_on_candidates(
        [], top_n=5, repo_root=repo_root, panel_axis="contest_cpu",
        include_master_gradient_rerank=False,
    )
    assert result["master_gradient_rerank_invoked"] is False
    assert result["master_gradient_annotations"] == []


def test_invoke_helper_bounds_top_n() -> None:
    """Helper caps to top_n candidates."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tools_dir = str(repo_root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import cathedral_autopilot_autonomous_loop as loop

    # Build 5 candidates, cap at 2.
    candidates = [
        loop.CandidateRow(
            candidate_id=f"test_{i}",
            family="test_family",
            predicted_score_delta=-0.001 * i,
            estimated_dispatch_cost_usd=0.10,
            expected_information_gain=0.5,
            archive_sha256=f"{i:064d}",
        )
        for i in range(5)
    ]
    result = loop.invoke_cathedral_consumers_on_candidates(
        candidates, top_n=2, repo_root=repo_root,
    )
    assert result["candidates_invoked"] == 2
    # Per consumer x 2 candidates = consumer_count * 2 invocations.
    assert len(result["invocations"]) == result["consumer_count"] * 2
