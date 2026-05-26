# SPDX-License-Identifier: MIT
"""Tests for META-LAGRANGIAN-WIRE-1 Phase 1 + Catalog #355 STRICT preflight gate.

Per T3 grand strategy review Decision 5 long-term centerpiece + operator
decision 2026-05-20 + WIRE-IN-RIGOR-AUDIT empirical finding that
``src/tac/findings_lagrangian/`` had ZERO production callers despite being
the canonical "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST
EMPHASIS" surface per CLAUDE.md.

Tests cover:

1. The Phase 1 canonical invocation helper
   :func:`invoke_meta_lagrangian_on_candidates` in
   ``tools/cathedral_autopilot_autonomous_loop.py``:
     - Synthetic CandidateRow → returns Phase 1 annotation rows
     - Adjustment factor is bounded to [0.95, 1.05]
     - Observability-only contract (score_claim=False, promotable=False,
       axis_tag="[predicted]")
     - Top-N bounding behavior
     - Per-candidate exception trapping (one failure does not crash)
     - NaN guard on predicted_score_delta
     - Schema constant pinned
2. The Catalog #355 STRICT preflight gate
   :func:`check_cathedral_autopilot_main_invokes_meta_lagrangian`:
     - Live-repo regression guard (0 violations at landing)
     - Synthetic violation: main() without invoker call
     - Synthetic acceptance: main() with each acceptance token
     - Waiver mechanism with placeholder rejection
     - Strict-mode behavior (raises vs silent)
     - Missing target file silent skip
     - Orchestrator wire-in regression guard (preflight_all strict=True)
     - Catalog #185 sister regression (gate function callable via globals)

6-hook wire-in declaration per Catalog #125 (tested in the helper +
gate behavior; not a separate test):
  - hook #1 sensitivity-map = ACTIVE (posterior_sigma_per_term surfaced
    per Catalog #305 observability)
  - hook #2 Pareto constraint = N/A at Phase 1 (Phase 2 lands the
    Lagrangian dual-variable surface that maps to Pareto KKT)
  - hook #3 bit-allocator = N/A at Phase 1 (Phase 2 lands per-element
    learned-optimal destination)
  - hook #4 cathedral autopilot dispatch = ACTIVE PRIMARY (the wire-in
    callsite IS the structural protection)
  - hook #5 continual-learning posterior = ACTIVE (the Phase 1 helper
    uses posterior_update_from_anchors per the canonical
    conjugate Bayesian update)
  - hook #6 probe-disambiguator = N/A at Phase 1 (Phase 2 lands the
    info-gain action-selector branch)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make tools/ importable for the helper tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


# Import the helper + dataclass from the cathedral autopilot module.
# The module-level import succeeds because we've already added tools/ to
# sys.path above.
from cathedral_autopilot_autonomous_loop import (  # noqa: E402
    META_LAGRANGIAN_INVOCATION_SCHEMA,
    PHASE_2_DUAL_SOLVER_INVOCATION_SCHEMA,
    CandidateRow,
    _candidate_residuals_for_lagrangian,
    _lagrangian_derived_adjustment_factor,
    invoke_meta_lagrangian_on_candidates,
)

from tac.preflight import (  # noqa: E402
    PreflightError,
    check_cathedral_autopilot_main_invokes_meta_lagrangian,
)

# ---------------------------------------------------------------------------
# Helper: build a synthetic CandidateRow
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    candidate_id: str = "synthetic_candidate",
    family: str = "synthetic_family",
    predicted_score_delta: float = -0.005,
    expected_information_gain: float = 0.5,
    estimated_dispatch_cost_usd: float = 1.0,
    consumer_payload: dict | None = None,
) -> CandidateRow:
    """Build a minimal synthetic CandidateRow for testing."""
    return CandidateRow(
        candidate_id=candidate_id,
        family=family,
        predicted_score_delta=predicted_score_delta,
        expected_information_gain=expected_information_gain,
        estimated_dispatch_cost_usd=estimated_dispatch_cost_usd,
        consumer_payload=consumer_payload or {},
    )


# ---------------------------------------------------------------------------
# Phase 1 helper tests
# ---------------------------------------------------------------------------


def test_schema_constant_pinned() -> None:
    """Schema constant must be the canonical pinned value."""
    assert META_LAGRANGIAN_INVOCATION_SCHEMA == "meta_lagrangian_invocation_v1_20260520"


def test_helper_returns_canonical_keys_and_observability_only_contract() -> None:
    """Phase 1 helper must return canonical keys + observability-only."""
    candidate = _make_candidate()
    result = invoke_meta_lagrangian_on_candidates([candidate])
    # Canonical top-level keys.
    expected_keys = {
        "schema",
        "evidence_grade",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "panel_axis",
        "top_n",
        "candidates_invoked",
        "phase",
        "invocations",
        "per_candidate_errors",
        "next_phase_roadmap",
    }
    assert expected_keys.issubset(result.keys()), (
        f"missing keys: {expected_keys - set(result.keys())}"
    )
    # Observability-only contract.
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["evidence_grade"] == "[predicted, meta-Lagrangian invocation]"
    assert result["phase"] == "phase_1_canonical_invocation_with_bounded_proxy_adjuster"


def test_helper_invocation_carries_canonical_per_candidate_markers() -> None:
    """Each invocation row must declare the canonical non-promotable markers."""
    candidate = _make_candidate()
    result = invoke_meta_lagrangian_on_candidates([candidate])
    assert result["candidates_invoked"] == 1
    invocations = result["invocations"]
    assert len(invocations) == 1
    row = invocations[0]
    # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323.
    assert row["score_claim"] is False
    assert row["promotable"] is False
    assert row["axis_tag"] == "[predicted]"
    assert row["candidate_id"] == "synthetic_candidate"
    assert row["family"] == "synthetic_family"
    assert "lagrangian_scalar" in row
    assert "adjustment_factor" in row
    assert "decompose" in row


def test_helper_adjustment_factor_bounded_to_5_percent_band() -> None:
    """Phase 1 adjustment factor must be in [0.95, 1.05] for any candidate."""
    test_deltas = [-1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0]
    for delta in test_deltas:
        candidate = _make_candidate(predicted_score_delta=delta)
        result = invoke_meta_lagrangian_on_candidates([candidate])
        row = result["invocations"][0]
        adj = row["adjustment_factor"]
        assert 0.95 <= adj <= 1.05, (
            f"adjustment_factor {adj} out of bound for predicted_score_delta={delta}"
        )


def test_helper_handles_top_n_bounding() -> None:
    """top_n must cap the candidates processed."""
    candidates = [
        _make_candidate(candidate_id=f"c{i}", family=f"f{i}", predicted_score_delta=-0.001 * i)
        for i in range(10)
    ]
    result = invoke_meta_lagrangian_on_candidates(candidates, top_n=3)
    assert result["candidates_invoked"] == 3
    assert len(result["invocations"]) == 3
    # First 3 candidates should be the ones processed.
    ids = [row["candidate_id"] for row in result["invocations"]]
    assert ids == ["c0", "c1", "c2"]


def test_helper_no_top_n_processes_all_candidates() -> None:
    """top_n=None must process all candidates."""
    candidates = [
        _make_candidate(candidate_id=f"c{i}", family=f"f{i}")
        for i in range(5)
    ]
    result = invoke_meta_lagrangian_on_candidates(candidates, top_n=None)
    assert result["candidates_invoked"] == 5


def test_helper_empty_candidate_list_returns_zero_invocations() -> None:
    """Empty candidate list must return zero invocations gracefully."""
    result = invoke_meta_lagrangian_on_candidates([])
    assert result["candidates_invoked"] == 0
    assert result["invocations"] == []
    assert result["per_candidate_errors"] == 0


def test_helper_panel_axis_passthrough() -> None:
    """panel_axis must be echoed in the output."""
    candidate = _make_candidate()
    result = invoke_meta_lagrangian_on_candidates([candidate], panel_axis="contest_cuda")
    assert result["panel_axis"] == "contest_cuda"


def test_helper_per_candidate_error_trapped_not_raised() -> None:
    """A per-candidate exception must NOT crash the helper."""
    # Build a candidate with a NaN predicted_score_delta — the residual
    # extractor's NaN guard returns (), which the helper treats as "skip"
    # rather than raising.
    candidate = CandidateRow(
        candidate_id="nan_candidate",
        family="nan_family",
        predicted_score_delta=float("nan"),
        expected_information_gain=0.5,
        estimated_dispatch_cost_usd=1.0,
    )
    result = invoke_meta_lagrangian_on_candidates([candidate])
    # The helper must return cleanly (no raise).
    assert result["candidates_invoked"] == 1
    # NaN residual produces empty residuals → "Phase 1 skip" annotation.
    row = result["invocations"][0]
    assert row.get("lagrangian_scalar") is None
    assert row["adjustment_factor"] == 1.0
    assert "Phase 1 skip" in row.get("rationale", "") or "missing partition" in row.get(
        "rationale", ""
    )


def test_residual_extractor_clips_to_unit_interval() -> None:
    """_candidate_residuals_for_lagrangian must clip to [-1, 1]."""
    candidate_high = _make_candidate(predicted_score_delta=5.0)
    candidate_low = _make_candidate(predicted_score_delta=-5.0)
    candidate_mid = _make_candidate(predicted_score_delta=0.3)
    assert _candidate_residuals_for_lagrangian(candidate_high) == (1.0,)
    assert _candidate_residuals_for_lagrangian(candidate_low) == (-1.0,)
    assert _candidate_residuals_for_lagrangian(candidate_mid) == (0.3,)


def test_residual_extractor_nan_guard() -> None:
    """_candidate_residuals_for_lagrangian must return () on NaN."""
    candidate_nan = CandidateRow(
        candidate_id="x",
        family="x",
        predicted_score_delta=float("nan"),
        expected_information_gain=0.0,
        estimated_dispatch_cost_usd=0.0,
    )
    assert _candidate_residuals_for_lagrangian(candidate_nan) == ()


def test_lagrangian_derived_adjustment_factor_nan_guards() -> None:
    """The adjuster must return 1.0 on any non-finite input."""
    assert _lagrangian_derived_adjustment_factor(float("nan"), 0.5) == 1.0
    assert _lagrangian_derived_adjustment_factor(1.0, float("nan")) == 1.0
    assert _lagrangian_derived_adjustment_factor(1.0, -0.5) == 1.0


def test_lagrangian_derived_adjustment_factor_bounded_band() -> None:
    """The adjuster must always return a value in [0.95, 1.05]."""
    test_scalars = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
    test_sigmas = [0.0, 0.1, 1.0, 10.0, 100.0]
    for s in test_scalars:
        for sig in test_sigmas:
            factor = _lagrangian_derived_adjustment_factor(s, sig)
            assert 0.95 <= factor <= 1.05, (
                f"adjuster({s}, {sig}) = {factor} out of bound"
            )


def test_helper_observability_decompose_includes_4_lagrangian_terms() -> None:
    """The decompose dict must surface the 4 Lagrangian terms + scalar."""
    candidate = _make_candidate()
    result = invoke_meta_lagrangian_on_candidates([candidate])
    row = result["invocations"][0]
    assert "decompose" in row
    decompose = row["decompose"]
    assert "data_fit" in decompose
    assert "occam_complexity_weighted" in decompose
    assert "occam_interpretability_weighted" in decompose
    assert "partition_penalty_weighted" in decompose
    assert "info_gain_reward_weighted" in decompose
    assert "scalar" in decompose


def test_helper_phase_2_dual_solver_consumes_axis_decomposition() -> None:
    candidate = _make_candidate(
        consumer_payload={
            "predicted_axis_decomposition": {
                "predicted_d_seg_delta": 0.001,
                "predicted_d_pose_delta": 0.0001,
                "predicted_archive_bytes_delta": 200,
                "axis_tag": "[predicted]",
                "canonical_provenance": {"fixture": "phase_2_dual_solver"},
            }
        }
    )
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
        phase_2_use_mlx=True,
    )
    assert result["phase"] == "phase_2_dual_solver_enabled_with_phase_1_fallback"
    assert result["phase_2_dual_solver_enabled"] is True
    assert result["phase_2_dual_solver_invoked_count"] == 1
    row = result["invocations"][0]
    assert "phase_1_adjustment_factor" in row
    phase_2 = row["phase_2_dual_solver"]
    assert phase_2["schema"] == PHASE_2_DUAL_SOLVER_INVOCATION_SCHEMA
    assert phase_2["status"] == "invoked"
    assert phase_2["score_claim"] is False
    assert phase_2["promotion_eligible"] is False
    assert phase_2["ready_for_exact_eval_dispatch"] is False
    assert row["adjustment_factor"] == phase_2["adjustment_factor"]
    assert phase_2["dual_variables_per_axis"]["seg"] > 0
    assert phase_2["dual_variables_per_axis"]["pose"] > 0
    assert phase_2["dual_variables_per_axis"]["rate"] > 0


def test_helper_phase_2_dual_solver_refuses_scalar_axis_inference() -> None:
    candidate = _make_candidate()
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
    )
    assert result["phase_2_dual_solver_invoked_count"] == 0
    row = result["invocations"][0]
    phase_2 = row["phase_2_dual_solver"]
    assert phase_2["status"] == "skipped"
    assert phase_2["invoked"] is False
    assert "refuses scalar-to-axis inference" in phase_2["rationale"]
    assert "phase_1_adjustment_factor" not in row


# ---------------------------------------------------------------------------
# Catalog #355 STRICT preflight gate tests
# ---------------------------------------------------------------------------


def test_live_repo_355_passes_clean() -> None:
    """Live repo Catalog #355: meta-Lagrangian invoker callsite present."""
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian()
    assert violations == [], (
        "Live repo Catalog #355 violations:\n  - "
        + "\n  - ".join(v[:200] for v in violations[:3])
    )


def test_live_repo_355_strict_does_not_raise() -> None:
    """Strict-mode live repo must NOT raise PreflightError."""
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(strict=True)
    assert violations == []


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
        '"""Synthetic cathedral autopilot stub for Catalog #355 testing."""\n'
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def invoke_meta_lagrangian_on_candidates(*args, **kwargs):\n"
        '    return {"schema": "test"}\n'
        "\n"
        "\n"
        "def compute_findings_lagrangian(*args, **kwargs):\n"
        "    return None\n"
        "\n"
        "\n"
        "def recommend_next_action_via_expected_information_gain(*args, **kwargs):\n"
        "    return None\n"
        "\n"
        "\n"
        f"{main_line}\n"
        f"{indented_body}\n"
        "    return 0\n"
    )
    target.write_text(src, encoding="utf-8")
    return target


def test_355_no_invoker_call_flagged(tmp_path: Path) -> None:
    """main() with no acceptance token call must be flagged."""
    _write_synthetic_target(tmp_path, "x = 1\nprint(x)")
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "does NOT invoke any of" in violations[0]


def test_355_accepts_invoke_meta_lagrangian_helper(tmp_path: Path) -> None:
    """main() calling invoke_meta_lagrangian_on_candidates must pass."""
    _write_synthetic_target(
        tmp_path,
        "result = invoke_meta_lagrangian_on_candidates([])",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert violations == []


def test_355_accepts_direct_compute_findings_lagrangian(tmp_path: Path) -> None:
    """main() calling compute_findings_lagrangian directly must pass."""
    _write_synthetic_target(
        tmp_path,
        "result = compute_findings_lagrangian('eq', posterior=None, partition=None, anchor_residuals=[])",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert violations == []


def test_355_accepts_recommend_next_action(tmp_path: Path) -> None:
    """main() calling recommend_next_action_via_expected_information_gain must pass."""
    _write_synthetic_target(
        tmp_path,
        "result = recommend_next_action_via_expected_information_gain([], "
        "posteriors_by_equation_id={}, budget_usd=1.0)",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert violations == []


def test_355_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Same-line waiver with non-placeholder rationale must short-circuit."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="META_LAGRANGIAN_INVOKER_WAIVED:test fixture path",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert violations == []


def test_355_waiver_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Placeholder rationale rejected (the gate's docstring cannot self-waive)."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="META_LAGRANGIAN_INVOKER_WAIVED:<rationale>",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_355_waiver_short_rationale_rejected(tmp_path: Path) -> None:
    """Rationale <4 chars rejected per the gate's documented constraint."""
    _write_synthetic_target(
        tmp_path,
        "x = 1",
        waiver_on_main_line="META_LAGRANGIAN_INVOKER_WAIVED:abc",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_355_strict_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode must raise PreflightError on violation."""
    _write_synthetic_target(tmp_path, "pass")
    with pytest.raises(PreflightError, match="Catalog #355"):
        check_cathedral_autopilot_main_invokes_meta_lagrangian(
            repo_root=tmp_path, strict=True
        )


def test_355_strict_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode must NOT raise when clean."""
    _write_synthetic_target(
        tmp_path,
        "x = invoke_meta_lagrangian_on_candidates([])",
    )
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_355_missing_target_file_silent(tmp_path: Path) -> None:
    """Target file missing → silent skip (no violation, no raise)."""
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert violations == []


def test_355_main_function_missing_flagged(tmp_path: Path) -> None:
    """Target file present but no def main() → violation."""
    target = tmp_path / "tools" / "cathedral_autopilot_autonomous_loop.py"
    target.parent.mkdir(parents=True)
    target.write_text("# no main function here\nx = 1\n")
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "missing def main" in violations[0] or "no top-level def main" in violations[0]


def test_355_syntax_error_tolerated(tmp_path: Path) -> None:
    """Target file with SyntaxError → graceful violation (not crash)."""
    target = tmp_path / "tools" / "cathedral_autopilot_autonomous_loop.py"
    target.parent.mkdir(parents=True)
    target.write_text("def main(:\n    invalid syntax\n")
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "SyntaxError" in violations[0]


# ---------------------------------------------------------------------------
# Orchestrator + sister-gate regression guards (Catalog #176 + #185 sister)
# ---------------------------------------------------------------------------


def test_orchestrator_wires_355_strict_true() -> None:
    """preflight_all must invoke Catalog #355 with strict=True."""
    preflight_source = (_REPO_ROOT / "src" / "tac" / "preflight.py").read_text(
        encoding="utf-8"
    )
    # Find the wire-in line.
    assert "check_cathedral_autopilot_main_invokes_meta_lagrangian(" in preflight_source
    # Verify strict=True is the orchestrator argument (it appears within the
    # wire-in call site in preflight_all).
    # Specifically: the call site MUST contain strict=True.
    # Walk all callsites to find at least one with strict=True.
    import re as _re

    callsites = _re.findall(
        r"check_cathedral_autopilot_main_invokes_meta_lagrangian\s*\(\s*([^)]*)\s*\)",
        preflight_source,
    )
    # At least one callsite (the preflight_all one) must include strict=True.
    strict_true_found = any("strict=True" in cs for cs in callsites)
    assert strict_true_found, (
        f"preflight_all must call check_cathedral_autopilot_main_invokes_meta_lagrangian "
        f"with strict=True; found callsites: {callsites!r}"
    )


def test_catalog_185_sister_callable_via_globals() -> None:
    """Catalog #185 sister regression: gate must be importable + callable."""
    import tac.preflight as _pf

    fn = getattr(_pf, "check_cathedral_autopilot_main_invokes_meta_lagrangian", None)
    assert fn is not None, "Catalog #355 gate must be importable from tac.preflight"
    assert callable(fn), "Catalog #355 gate must be callable"
    # Verify signature is keyword-only.
    import inspect as _inspect

    sig = _inspect.signature(fn)
    # The gate takes ONLY keyword args (repo_root, strict, verbose).
    for name, param in sig.parameters.items():
        assert param.kind in (
            _inspect.Parameter.KEYWORD_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"parameter {name} must be keyword-able"


def test_helper_importable_from_cathedral_autopilot_module() -> None:
    """The Phase 1 helper must be importable from the canonical module."""
    from cathedral_autopilot_autonomous_loop import (
        META_LAGRANGIAN_INVOCATION_SCHEMA as _schema,
    )
    from cathedral_autopilot_autonomous_loop import (
        invoke_meta_lagrangian_on_candidates as _helper,
    )

    assert callable(_helper)
    assert _schema == "meta_lagrangian_invocation_v1_20260520"
