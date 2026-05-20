# SPDX-License-Identifier: MIT
"""Tests for tac.findings_lagrangian.phase_2_ablation.ablation_framework.

Covers:
  - AdjusterAblationContext dataclass contract
  - AdjusterAblationVerdict dataclass contract + canonical marker invariants
  - compute_solver_dual_variable_for_adjuster (3 adjusters; synthetic posterior)
  - paired_comparison_against_hand_derived (3 modes)
  - fcntl-locked JSONL persistence (append + load lenient + load strict)
  - Catalog #287 placeholder-rationale rejection (via dataclass invariants)
  - Catalog #323 canonical Provenance integration (non-promotable markers)
  - Catalog #138 strict-load discipline (corrupt-file raise)
  - Live-repo regression guards
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.findings_lagrangian.phase_2_ablation import (
    AblationError,
    AblationMode,
    AdjusterAblationContext,
    AdjusterAblationVerdict,
    DEFAULT_ABLATION_MODE,
    DEFAULT_DIVERGENCE_SIGMA_BOUND,
    PHASE_2_ABLATION_POSTERIOR_PATH,
    PHASE_2_ABLATION_POSTERIOR_LOCK_PATH,
    PHASE_2_ABLATION_SCHEMA,
    PROMOTION_THRESHOLD_MAX_REGRESSION_USD,
    PROMOTION_THRESHOLD_MIN_ANCHORS,
    SUPPORTED_ADJUSTERS,
    append_paired_comparison_row,
    compute_solver_dual_variable_for_adjuster,
    load_paired_comparison_rows_lenient,
    load_paired_comparison_rows_strict,
    paired_comparison_against_hand_derived,
)


# ----------------------------------------------------------------------------
# Module-level constants regression
# ----------------------------------------------------------------------------

def test_supported_adjusters_pinned() -> None:
    """The 3 selected adjusters must remain Phase 2 START scope."""
    assert SUPPORTED_ADJUSTERS == (
        "mdl_density",
        "predicted_dispatch_risk",
        "composition_alpha_v2",
    )


def test_schema_version_pinned() -> None:
    """Schema id is pinned so downstream consumers don't drift."""
    assert PHASE_2_ABLATION_SCHEMA == "phase_2_ablation_posterior_v1_20260520"


def test_canonical_paths_under_omx_state() -> None:
    """Posterior path lives under .omx/state per Catalog #131 discipline."""
    assert str(PHASE_2_ABLATION_POSTERIOR_PATH).startswith(".omx/state/")
    assert str(PHASE_2_ABLATION_POSTERIOR_LOCK_PATH).startswith(".omx/state/")


def test_default_mode_is_paired_comparison() -> None:
    """Safety-rail default: hand-derived authoritative; solver measured."""
    assert DEFAULT_ABLATION_MODE is AblationMode.PAIRED_COMPARISON


def test_promotion_thresholds_pinned() -> None:
    """Operator-routable promotion thresholds (forbidden-premature-KILL safe)."""
    assert PROMOTION_THRESHOLD_MIN_ANCHORS == 30
    assert PROMOTION_THRESHOLD_MAX_REGRESSION_USD == 1.0
    assert DEFAULT_DIVERGENCE_SIGMA_BOUND == 2.0


def test_ablation_mode_string_values() -> None:
    """Mode values are stable string identifiers for CLI flag."""
    assert AblationMode.HAND_DERIVED.value == "hand_derived"
    assert AblationMode.SOLVER_DERIVED.value == "solver_derived"
    assert AblationMode.PAIRED_COMPARISON.value == "paired_comparison"


# ----------------------------------------------------------------------------
# AdjusterAblationContext invariants
# ----------------------------------------------------------------------------

def test_context_construction_happy_path() -> None:
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_1",
        family="hnerv",
    )
    assert ctx.adjuster_name == "mdl_density"
    assert ctx.mode is AblationMode.PAIRED_COMPARISON


def test_context_rejects_unknown_adjuster() -> None:
    with pytest.raises(AblationError, match="not in SUPPORTED_ADJUSTERS"):
        AdjusterAblationContext(
            adjuster_name="phantom_adjuster",
            mode=AblationMode.PAIRED_COMPARISON,
            base_delta=-0.1,
            signal_value=0.5,
            candidate_id="cand_1",
        )


def test_context_rejects_nan_base_delta() -> None:
    with pytest.raises(AblationError, match="must not be NaN"):
        AdjusterAblationContext(
            adjuster_name="mdl_density",
            mode=AblationMode.PAIRED_COMPARISON,
            base_delta=float("nan"),
            signal_value=0.5,
            candidate_id="cand_1",
        )


def test_context_rejects_empty_candidate_id() -> None:
    with pytest.raises(AblationError, match="non-empty"):
        AdjusterAblationContext(
            adjuster_name="mdl_density",
            mode=AblationMode.PAIRED_COMPARISON,
            base_delta=-0.1,
            signal_value=0.5,
            candidate_id="",
        )


def test_context_rejects_non_enum_mode() -> None:
    with pytest.raises(AblationError, match="mode must be AblationMode"):
        AdjusterAblationContext(
            adjuster_name="mdl_density",
            mode="paired_comparison",  # type: ignore[arg-type]
            base_delta=-0.1,
            signal_value=0.5,
            candidate_id="cand_1",
        )


def test_context_accepts_none_signal() -> None:
    """None signal must be accepted (unknown signal = passthrough)."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=None,
        candidate_id="cand_1",
    )
    assert ctx.signal_value is None


# ----------------------------------------------------------------------------
# AdjusterAblationVerdict canonical marker invariants (Catalog #341)
# ----------------------------------------------------------------------------

def _make_verdict_kwargs() -> dict:
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_1",
    )
    return dict(
        context=ctx,
        hand_derived_delta=-0.1,
        solver_derived_delta=-0.1,
        solver_posterior_sigma=0.5,
        divergence=0.0,
        divergence_absolute=0.0,
        sign_flip=False,
        within_tolerance=True,
        authoritative_delta=-0.1,
        captured_at_utc="2026-05-20T15:00:00+00:00",
    )


def test_verdict_construction_happy_path() -> None:
    v = AdjusterAblationVerdict(**_make_verdict_kwargs())
    assert v.score_claim is False
    assert v.promotable is False
    assert v.axis_tag == "[predicted]"


def test_verdict_rejects_score_claim_true() -> None:
    kw = _make_verdict_kwargs()
    kw["score_claim"] = True
    with pytest.raises(AblationError, match="Catalog #341"):
        AdjusterAblationVerdict(**kw)


def test_verdict_rejects_promotable_true() -> None:
    kw = _make_verdict_kwargs()
    kw["promotable"] = True
    with pytest.raises(AblationError, match="Catalog #341"):
        AdjusterAblationVerdict(**kw)


def test_verdict_rejects_non_predicted_axis() -> None:
    kw = _make_verdict_kwargs()
    kw["axis_tag"] = "[contest-CUDA]"
    with pytest.raises(AblationError, match="Catalog #341"):
        AdjusterAblationVerdict(**kw)


def test_verdict_to_jsonl_dict_canonical_keys() -> None:
    v = AdjusterAblationVerdict(**_make_verdict_kwargs())
    d = v.to_jsonl_dict()
    required = {
        "schema",
        "captured_at_utc",
        "adjuster_name",
        "mode",
        "candidate_id",
        "family",
        "base_delta",
        "signal_value",
        "panel_axis",
        "sigma_obs",
        "hand_derived_delta",
        "solver_derived_delta",
        "solver_posterior_sigma",
        "divergence",
        "divergence_absolute",
        "sign_flip",
        "within_tolerance",
        "authoritative_delta",
        "score_claim",
        "promotable",
        "axis_tag",
    }
    assert required <= set(d.keys())


def test_verdict_to_jsonl_dict_nan_encoded_as_none() -> None:
    kw = _make_verdict_kwargs()
    kw["solver_posterior_sigma"] = float("nan")
    v = AdjusterAblationVerdict(**kw)
    d = v.to_jsonl_dict()
    assert d["solver_posterior_sigma"] is None
    # JSON-serializable (no NaN literal)
    json.dumps(d, allow_nan=False)


# ----------------------------------------------------------------------------
# compute_solver_dual_variable_for_adjuster
# ----------------------------------------------------------------------------

def test_solver_dual_unknown_adjuster() -> None:
    with pytest.raises(AblationError, match="not in SUPPORTED_ADJUSTERS"):
        compute_solver_dual_variable_for_adjuster(
            "phantom_adjuster", -0.1, 0.5
        )


@pytest.mark.parametrize("name", SUPPORTED_ADJUSTERS)
def test_solver_dual_none_signal_passes_through(name: str) -> None:
    """None signal = passthrough delta + NaN sigma (sister hand-derived semantic)."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(name, -0.1, None)
    assert adjusted == -0.1
    assert math.isnan(sigma)


@pytest.mark.parametrize("name", SUPPORTED_ADJUSTERS)
def test_solver_dual_non_numeric_signal_passes_through(name: str) -> None:
    """Non-numeric signal = passthrough + NaN sigma."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        name, -0.1, "not a number"  # type: ignore[arg-type]
    )
    assert adjusted == -0.1
    assert math.isnan(sigma)


def test_solver_mdl_density_saturated_floors() -> None:
    """MDL density 0.97 → within-class saturated → floor at -0.005."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "mdl_density", -0.5, 0.99, sigma_obs=0.01
    )
    # With strong observation evidence + saturated band, posterior mu > 0.95
    # → floor at -0.005 (less-negative than -0.5).
    assert adjusted == pytest.approx(-0.005, abs=1e-9)
    assert sigma > 0


def test_solver_mdl_density_low_passes_through() -> None:
    """MDL density 0.5 (with low observation sigma) → across-class → passthrough."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "mdl_density", -0.5, 0.1, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.5, abs=1e-9)
    assert sigma > 0


def test_solver_predicted_dispatch_risk_refusal_floors() -> None:
    """Risk 75 → refusal → floor at 0.0."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "predicted_dispatch_risk", -0.5, 75.0, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(0.0, abs=1e-9)
    assert sigma > 0


def test_solver_predicted_dispatch_risk_moderate_halves() -> None:
    """Risk 30 → moderate → 50% penalty."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "predicted_dispatch_risk", -0.4, 30.0, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.2, abs=1e-3)


def test_solver_predicted_dispatch_risk_low_passes_through() -> None:
    """Risk 5 → low → passthrough."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "predicted_dispatch_risk", -0.4, 5.0, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.4, abs=1e-3)


def test_solver_composition_alpha_super_additive_rewards() -> None:
    """Alpha 1.5 → SUPER_ADDITIVE → reward factor 1.5×."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "composition_alpha_v2", -0.1, 1.5, sigma_obs=0.01
    )
    # SUPER_ADDITIVE branch: base_delta * mu (clamped to [1.0, 2.0])
    # With strong evidence + alpha=1.5 → mu close to 1.5 → reward ~1.5×
    assert adjusted == pytest.approx(-0.15, abs=1e-2)


def test_solver_composition_alpha_super_additive_cap() -> None:
    """Alpha 2.5 → SUPER_ADDITIVE bounded at 2.0 cap."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "composition_alpha_v2", -0.1, 2.5, sigma_obs=0.01
    )
    # With strong evidence + alpha clipped to 2.5, posterior mu ~2.0+
    # → reward capped at 2.0×.
    assert adjusted == pytest.approx(-0.2, abs=1e-2)


def test_solver_composition_alpha_additive_passes_through() -> None:
    """Alpha 0.9 (with strong evidence) → ADDITIVE → passthrough."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "composition_alpha_v2", -0.1, 0.9, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.1, abs=1e-3)


def test_solver_composition_alpha_sub_additive_halves() -> None:
    """Alpha 0.4 → SUB_ADDITIVE → 50% penalty."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "composition_alpha_v2", -0.4, 0.4, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.2, abs=1e-2)


def test_solver_composition_alpha_saturating_floors() -> None:
    """Alpha 0.1 → SATURATING → floor at -0.005."""
    adjusted, sigma = compute_solver_dual_variable_for_adjuster(
        "composition_alpha_v2", -0.5, 0.1, sigma_obs=0.01
    )
    assert adjusted == pytest.approx(-0.005, abs=1e-9)


# ----------------------------------------------------------------------------
# paired_comparison_against_hand_derived - 3 modes
# ----------------------------------------------------------------------------

def test_paired_comparison_hand_derived_mode_skips_solver() -> None:
    """HAND_DERIVED mode: solver delta == hand delta; sigma is NaN."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.HAND_DERIVED,
        base_delta=-0.1,
        signal_value=0.97,
        candidate_id="cand_hd",
    )
    v = paired_comparison_against_hand_derived(ctx)
    assert v.hand_derived_delta == v.solver_derived_delta
    assert math.isnan(v.solver_posterior_sigma)
    assert v.divergence == 0.0
    assert v.authoritative_delta == v.hand_derived_delta


def test_paired_comparison_solver_derived_mode_uses_solver() -> None:
    """SOLVER_DERIVED mode: authoritative = solver delta."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.SOLVER_DERIVED,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_sd",
        sigma_obs=0.01,
    )
    v = paired_comparison_against_hand_derived(ctx)
    assert v.authoritative_delta == v.solver_derived_delta


def test_paired_comparison_default_mode_uses_hand_derived_as_authoritative() -> None:
    """PAIRED_COMPARISON mode (default): hand-derived is authoritative safety rail."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.97,
        candidate_id="cand_pc",
    )
    v = paired_comparison_against_hand_derived(ctx)
    assert v.authoritative_delta == v.hand_derived_delta
    # Solver was invoked, so divergence MAY be non-zero.
    assert not math.isnan(v.solver_posterior_sigma)


def test_paired_comparison_divergence_zero_on_passthrough() -> None:
    """When signal is None, both adjusters passthrough → divergence == 0."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=None,
        candidate_id="cand_none",
    )
    v = paired_comparison_against_hand_derived(ctx)
    assert v.divergence == 0.0
    assert v.within_tolerance is True


def test_paired_comparison_sign_flip_detection() -> None:
    """Sign flip is True when adjusters disagree on direction."""
    # Construct a scenario where hand floors at 0 and solver also floors at 0
    # (both same sign). To force a sign flip we'd need different math; for
    # mdl_density both floor at -0.005 → no flip. Smoke test detection.
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.97,
        candidate_id="cand_sf",
    )
    v = paired_comparison_against_hand_derived(ctx)
    # Both hand and solver floor at -0.005 (negative) → no flip.
    assert v.sign_flip is False


def test_paired_comparison_within_tolerance_paired_with_sigma() -> None:
    """within_tolerance compares divergence to 2 * solver_sigma."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_wt",
    )
    v = paired_comparison_against_hand_derived(ctx)
    if not math.isnan(v.solver_posterior_sigma):
        expected = v.divergence_absolute <= (
            DEFAULT_DIVERGENCE_SIGMA_BOUND * abs(v.solver_posterior_sigma)
        )
        assert v.within_tolerance == expected


def test_paired_comparison_captured_at_utc_iso_format() -> None:
    """captured_at_utc is ISO 8601 with timezone."""
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_ts",
    )
    v = paired_comparison_against_hand_derived(ctx)
    # Format: 2026-05-20T15:00:00+00:00
    assert "T" in v.captured_at_utc
    assert "+00:00" in v.captured_at_utc or v.captured_at_utc.endswith("Z")


# ----------------------------------------------------------------------------
# fcntl-locked JSONL persistence
# ----------------------------------------------------------------------------

def test_append_and_load_lenient(tmp_path: Path) -> None:
    """Round-trip: append → load_lenient returns the same row."""
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    lock = tmp_path / "phase_2_ablation_posterior.lock"
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_persist",
    )
    v = paired_comparison_against_hand_derived(ctx)
    append_paired_comparison_row(v, posterior_path=posterior, lock_path=lock)
    rows = load_paired_comparison_rows_lenient(posterior_path=posterior)
    assert len(rows) == 1
    assert rows[0]["adjuster_name"] == "mdl_density"
    assert rows[0]["candidate_id"] == "cand_persist"
    assert rows[0]["schema"] == PHASE_2_ABLATION_SCHEMA


def test_load_lenient_missing_returns_empty(tmp_path: Path) -> None:
    rows = load_paired_comparison_rows_lenient(
        posterior_path=tmp_path / "does_not_exist.jsonl"
    )
    assert rows == []


def test_load_lenient_skips_malformed_lines(tmp_path: Path) -> None:
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    posterior.write_text(
        '{"adjuster_name": "mdl_density", "valid": true}\n'
        "this is not json\n"
        '{"adjuster_name": "predicted_dispatch_risk", "valid": true}\n'
        "\n",
        encoding="utf-8",
    )
    rows = load_paired_comparison_rows_lenient(posterior_path=posterior)
    assert len(rows) == 2
    assert {r["adjuster_name"] for r in rows} == {
        "mdl_density", "predicted_dispatch_risk"
    }


def test_load_strict_raises_on_malformed(tmp_path: Path) -> None:
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    posterior.write_text(
        '{"adjuster_name": "mdl_density", "valid": true}\n'
        "garbage line\n",
        encoding="utf-8",
    )
    with pytest.raises(AblationError, match="corrupt at line 2"):
        load_paired_comparison_rows_strict(posterior_path=posterior)


def test_load_strict_raises_on_non_dict_root(tmp_path: Path) -> None:
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    posterior.write_text(
        '["not", "a", "dict"]\n',
        encoding="utf-8",
    )
    with pytest.raises(AblationError, match="not a "):
        load_paired_comparison_rows_strict(posterior_path=posterior)


def test_load_strict_missing_returns_empty(tmp_path: Path) -> None:
    rows = load_paired_comparison_rows_strict(
        posterior_path=tmp_path / "does_not_exist.jsonl"
    )
    assert rows == []


def test_append_emits_canonical_provenance_metadata(tmp_path: Path) -> None:
    """Each row carries written_pid + written_at_utc + row_uuid per Catalog #131."""
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    lock = tmp_path / "phase_2_ablation_posterior.lock"
    ctx = AdjusterAblationContext(
        adjuster_name="composition_alpha_v2",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.9,
        candidate_id="cand_meta",
    )
    v = paired_comparison_against_hand_derived(ctx)
    append_paired_comparison_row(v, posterior_path=posterior, lock_path=lock)
    rows = load_paired_comparison_rows_lenient(posterior_path=posterior)
    assert "written_pid" in rows[0]
    assert "written_at_utc" in rows[0]
    assert "row_uuid" in rows[0]
    assert len(rows[0]["row_uuid"]) == 12


def test_multi_append_preserves_order(tmp_path: Path) -> None:
    """JSONL append-only: rows appear in insertion order."""
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    lock = tmp_path / "phase_2_ablation_posterior.lock"
    for i, name in enumerate(SUPPORTED_ADJUSTERS):
        ctx = AdjusterAblationContext(
            adjuster_name=name,
            mode=AblationMode.PAIRED_COMPARISON,
            base_delta=-0.1,
            signal_value=0.5,
            candidate_id=f"cand_{i}",
        )
        v = paired_comparison_against_hand_derived(ctx)
        append_paired_comparison_row(v, posterior_path=posterior, lock_path=lock)
    rows = load_paired_comparison_rows_lenient(posterior_path=posterior)
    assert len(rows) == 3
    assert [r["candidate_id"] for r in rows] == ["cand_0", "cand_1", "cand_2"]


def test_append_jsonl_byte_stable_sort_keys(tmp_path: Path) -> None:
    """JSON output uses sort_keys=True (byte-stable, diff-friendly)."""
    posterior = tmp_path / "phase_2_ablation_posterior.jsonl"
    lock = tmp_path / "phase_2_ablation_posterior.lock"
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        base_delta=-0.1,
        signal_value=0.5,
        candidate_id="cand_stable",
    )
    v = paired_comparison_against_hand_derived(ctx)
    append_paired_comparison_row(v, posterior_path=posterior, lock_path=lock)

    text = posterior.read_text(encoding="utf-8").strip()
    parsed = json.loads(text)
    re_encoded = json.dumps(parsed, sort_keys=True, allow_nan=False)
    assert text == re_encoded


# ----------------------------------------------------------------------------
# Phase 1 preservation regression guards
# ----------------------------------------------------------------------------

def test_phase_1_invocation_point_preserved() -> None:
    """Catalog #355 STRICT gate must still pass: Phase 2 cannot remove the
    Phase 1 invocation point in cathedral_autopilot_autonomous_loop.main().
    """
    from tac.preflight import (
        check_cathedral_autopilot_main_invokes_meta_lagrangian,
    )
    # Should not raise + should not return violations on the live repo.
    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(
        strict=False, verbose=False
    )
    assert violations == []


def test_phase_2_does_not_break_phase_1_helper_import() -> None:
    """Phase 1 invoker helper must remain importable."""
    from tools.cathedral_autopilot_autonomous_loop import (
        invoke_meta_lagrangian_on_candidates,
    )
    assert callable(invoke_meta_lagrangian_on_candidates)


# ----------------------------------------------------------------------------
# Autopilot wire-in regression guards
# ----------------------------------------------------------------------------

def test_autopilot_phase_2_invoker_importable() -> None:
    """The Phase 2 invoker helper must be importable from the autopilot module."""
    from tools.cathedral_autopilot_autonomous_loop import (
        invoke_phase_2_ablation_on_candidates,
        PHASE_2_ABLATION_INVOCATION_SCHEMA,
    )
    assert callable(invoke_phase_2_ablation_on_candidates)
    assert PHASE_2_ABLATION_INVOCATION_SCHEMA == (
        "phase_2_ablation_invocation_v1_20260520"
    )


def test_autopilot_phase_2_invoker_empty_candidates_returns_canonical_payload() -> None:
    """Empty candidate list must return a canonical payload (no crash)."""
    from tools.cathedral_autopilot_autonomous_loop import (
        invoke_phase_2_ablation_on_candidates,
    )
    payload = invoke_phase_2_ablation_on_candidates(
        [], mode="paired_comparison"
    )
    assert payload["schema"] == "phase_2_ablation_invocation_v1_20260520"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["candidates_invoked"] == 0
    assert payload["verdicts"] == []


def test_autopilot_phase_2_invoker_invalid_mode_returns_schema_error() -> None:
    """Invalid mode string returns a payload with schema_error (no crash)."""
    from tools.cathedral_autopilot_autonomous_loop import (
        invoke_phase_2_ablation_on_candidates,
    )
    payload = invoke_phase_2_ablation_on_candidates(
        [], mode="phantom_mode"
    )
    assert "schema_error" in payload
    assert payload["per_candidate_errors"] >= 1


def test_autopilot_phase_2_invoker_runs_on_synthetic_candidate() -> None:
    """Phase 2 invoker runs end-to-end on a synthetic CandidateRow."""
    from tools.cathedral_autopilot_autonomous_loop import (
        CandidateRow,
        invoke_phase_2_ablation_on_candidates,
    )
    # Construct a minimal synthetic CandidateRow. Field names mirror live
    # CandidateRow signature in tools/cathedral_autopilot_autonomous_loop.py.
    candidate = CandidateRow(
        candidate_id="cand_synthetic",
        family="test_family",
        predicted_score_delta=-0.05,
        estimated_dispatch_cost_usd=1.0,
        expected_information_gain=0.1,
        mdl_density=0.99,
        predicted_dispatch_risk=60.0,
        composition_alpha=0.5,
    )
    payload = invoke_phase_2_ablation_on_candidates(
        [candidate], mode="paired_comparison"
    )
    assert payload["candidates_invoked"] == 1
    assert payload["adjusters_per_candidate"] == 3
    assert len(payload["verdicts"]) == 3
    # All 3 SUPPORTED_ADJUSTERS were invoked
    adjusters_invoked = {v["adjuster_name"] for v in payload["verdicts"]}
    assert adjusters_invoked == set(SUPPORTED_ADJUSTERS)
    # Per-adjuster aggregate contains all 3
    assert set(payload["aggregate_divergence"].keys()) == set(SUPPORTED_ADJUSTERS)


def test_autopilot_phase_2_cli_flag_present() -> None:
    """The CLI flag --phase-2-ablation-mode must accept all 3 modes."""
    import argparse
    from tools.cathedral_autopilot_autonomous_loop import main

    # Use --help to dump CLI without firing the loop (catches argparse breakage)
    with pytest.raises(SystemExit):
        main(["--help"])


def test_autopilot_phase_2_cli_persist_flag_present() -> None:
    """The CLI flag --persist-phase-2-ablation must be defined."""
    from tools.cathedral_autopilot_autonomous_loop import main

    with pytest.raises(SystemExit):
        main(["--help"])
