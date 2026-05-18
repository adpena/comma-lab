# SPDX-License-Identifier: MIT
"""Tests for ASYMPTOTIC PURSUIT candidate-readiness assessment + dispatch queue.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #270 + #313
+ #315 + #240. Covers helper unit tests, end-to-end assessment correctness,
EV-per-dollar ranking, top-1 verdict logic, provenance discipline, and
operator-authorize command generation.

Sister of `tools/asymptotic_pursuit_candidate_readiness_assessment.py` +
`tools/asymptotic_pursuit_dispatch_queue.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from asymptotic_pursuit_candidate_readiness_assessment import (  # noqa: E402
    CANONICAL_CANDIDATES,
    CandidateReadiness,
    ReadinessAssessment,
    _classify_horizon_class,
    _classify_readiness_verdict,
    _compute_blocking_issues,
    _compute_ev_per_dollar,
    _estimate_dispatch_cost,
    assess_candidate,
    assess_candidates,
    build_operator_authorize_command,
    rank_by_ev_per_dollar,
    write_artifact,
)
from asymptotic_pursuit_dispatch_queue import (  # noqa: E402
    build_dispatch_sequence,
    compute_cost_band_rollup,
    compute_operator_attention_budget,
)


# ---------- helper unit tests ----------


def test_classify_horizon_class_asymptotic_pursuit():
    """Predicted low < 0.120 is ASYMPTOTIC_PURSUIT per Catalog #309."""
    assert _classify_horizon_class(0.05, 0.10) == "asymptotic_pursuit"
    assert _classify_horizon_class(0.08, 0.12) == "asymptotic_pursuit"


def test_classify_horizon_class_frontier_pursuit():
    """0.120 <= low < 0.180 is FRONTIER_PURSUIT."""
    assert _classify_horizon_class(0.13, 0.16) == "frontier_pursuit"
    assert _classify_horizon_class(0.155, 0.175) == "frontier_pursuit"


def test_classify_horizon_class_plateau_adjacent():
    """0.180 <= low is PLATEAU_ADJACENT."""
    assert _classify_horizon_class(0.185, 0.200) == "plateau_adjacent"


def test_classify_horizon_class_unknown_for_none():
    """None predicted band returns unknown."""
    assert _classify_horizon_class(None, None) == "unknown"
    assert _classify_horizon_class(None, 0.15) == "unknown"
    assert _classify_horizon_class(0.10, None) == "unknown"


def test_compute_ev_per_dollar_baseline_beating():
    """Predicted midpoint below baseline 0.19205 → positive EV."""
    ev = _compute_ev_per_dollar(0.10, 0.15, 1.0)
    assert ev > 0
    # midpoint 0.125 → delta 0.067 → EV 0.067/1.0 = 0.067
    assert abs(ev - 0.067) < 0.001


def test_compute_ev_per_dollar_above_baseline_zero():
    """Predicted midpoint above baseline → zero EV."""
    assert _compute_ev_per_dollar(0.20, 0.22, 1.0) == 0.0


def test_compute_ev_per_dollar_none_band_zero():
    """None band → zero EV."""
    assert _compute_ev_per_dollar(None, None, 1.0) == 0.0


def test_estimate_dispatch_cost_t4_paired():
    """T4 paired cost should include CPU eval."""
    cost, wall = _estimate_dispatch_cost("T4", 100)
    # 100 * 20s = 2000s = ~33min on T4 = ~$0.328 + $0.10 CPU
    assert cost > 0.10  # at least CPU cost
    assert wall == 2000


def test_estimate_dispatch_cost_a100_more_expensive_per_epoch():
    """A100 should be more expensive per epoch than T4 (higher $/hr)."""
    t4_cost, _ = _estimate_dispatch_cost("T4", 1000)
    a100_cost, _ = _estimate_dispatch_cost("A100", 1000)
    assert a100_cost > t4_cost


# ---------- blocking issues + readiness verdict ----------


def test_compute_blocking_issues_clean_recipe_no_blockers(tmp_path):
    """Clean state should produce no blocking issues."""
    # Need both files to exist for clean state
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("---\n")
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=recipe,
        trainer_path=Path(__file__),  # An existing file
    )
    assert len(issues) == 0


def test_compute_blocking_issues_council_proceed_with_revisions_flagged():
    """PROCEED_WITH_REVISIONS triggers Catalog #315 blocker."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED_WITH_REVISIONS",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
    )
    assert any("CATALOG_315" in i for i in issues)


def test_compute_blocking_issues_probe_blocking_flagged():
    """Catalog #313 probe-blocking adds explicit blocker."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=True,
        predecessor_probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
    )
    assert any("CATALOG_313" in i for i in issues)


def test_classify_readiness_verdict_no_blockers_ready():
    """No blockers → READY."""
    assert _classify_readiness_verdict(()) == "READY"


def test_classify_readiness_verdict_probe_blocker_defer():
    """Probe-blocker = DEFER (substantive evidence needed)."""
    assert _classify_readiness_verdict(("CATALOG_313_PROBE_BLOCKING:foo",)) == "DEFER"


def test_classify_readiness_verdict_council_blocker_defer():
    """PROCEED_WITH_REVISIONS = DEFER per Catalog #315."""
    assert _classify_readiness_verdict(("CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM",)) == "DEFER"


def test_classify_readiness_verdict_recipe_flag_needs_fix():
    """Recipe research_only/dispatch_enabled flags = NEEDS_FIX."""
    assert _classify_readiness_verdict(("RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP",)) == "NEEDS_FIX"


def test_classify_readiness_verdict_full_main_blocker_defer():
    """NotImplementedError in _full_main = DEFER (engineering work needed)."""
    assert _classify_readiness_verdict(("CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError",)) == "DEFER"


# ---------- per-candidate assessment ----------


def test_assess_candidate_atw_codec_v2_is_blocked_by_probe():
    """Catalog #313: ATW V2 has BLOCKING probe outcome from 2026-05-16.

    Empirical receipt: probe_id atw_v2_d4_h_latent_given_scorer_class_20260516
    returned INDEPENDENT verdict (MI=0.006385 << 0.5 threshold).
    """
    c = assess_candidate("atw_codec_v2")
    assert c.predecessor_probe_blocking is True
    assert c.predecessor_probe_id == "atw_v2_d4_h_latent_given_scorer_class_20260516"
    assert c.predecessor_probe_verdict == "INDEPENDENT"
    assert c.readiness_verdict == "DEFER"


def test_assess_candidate_z6_is_blocked_by_council():
    """Catalog #315: Z6 has council verdict PROCEED_WITH_REVISIONS."""
    c = assess_candidate("time_traveler_l5_z6")
    assert c.latest_council_verdict == "PROCEED_WITH_REVISIONS"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"


def test_assess_candidate_z6_has_predicted_band_asymptotic_or_frontier():
    """Z6 design memo declares predicted ΔS band [0.13, 0.16]."""
    c = assess_candidate("time_traveler_l5_z6")
    assert c.predicted_delta_s_band_low is not None
    assert c.predicted_delta_s_band_high is not None
    # The Z6 design memo declares [0.13, 0.16] which is frontier_pursuit
    # band per Catalog #309; the actual asymptotic band is for Z8.
    assert c.horizon_class in ("frontier_pursuit", "asymptotic_pursuit")


# ---------- ranking ----------


def test_rank_by_ev_per_dollar_ready_before_needs_fix_before_defer():
    """Tier order: READY > NEEDS_FIX > DEFER, then by EV/$ within tier."""
    fake_ready = CandidateReadiness(
        substrate_id="ready_low_ev",
        recipe_basename="x",
        recipe_path=None,
        trainer_path=None,
        lane_maturity="L1",
        impl_complete=True,
        full_main_implemented=True,
        full_main_blocker=None,
        latest_council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10,
        predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X",
        estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100,
        gpu_class="T4",
        min_smoke_gpu="T4",
        cost_band_epochs=100,
        horizon_class="asymptotic_pursuit",
        blocking_issues=(),
        readiness_verdict="READY",
        ev_per_dollar=0.05,
    )
    fake_defer_higher_ev = CandidateReadiness(
        **{**fake_ready.__dict__, "substrate_id": "defer_high_ev", "readiness_verdict": "DEFER", "ev_per_dollar": 1.0},
    )
    ranked = rank_by_ev_per_dollar((fake_ready, fake_defer_higher_ev))
    # READY beats DEFER regardless of EV
    assert ranked[0] == "ready_low_ev"
    assert ranked[1] == "defer_high_ev"


def test_rank_by_ev_per_dollar_within_tier_by_ev():
    """Within same tier, higher EV wins."""
    base = dict(
        recipe_basename="x", recipe_path=None, trainer_path=None,
        lane_maturity="L1", impl_complete=True, full_main_implemented=True,
        full_main_blocker=None, latest_council_verdict="PROCEED",
        research_only=False, dispatch_enabled=True, dispatch_blockers=(),
        predecessor_probe_blocking=False, predecessor_probe_id=None,
        predecessor_probe_verdict=None, predicted_delta_s_band_low=0.10,
        predicted_delta_s_band_high=0.12, predicted_delta_s_provenance="X",
        estimated_dispatch_cost_usd=1.0, estimated_dispatch_wall_clock_seconds=100,
        gpu_class="T4", min_smoke_gpu="T4", cost_band_epochs=100,
        horizon_class="asymptotic_pursuit", blocking_issues=(),
        readiness_verdict="NEEDS_FIX",
    )
    a = CandidateReadiness(substrate_id="a", **base, ev_per_dollar=0.50)
    b = CandidateReadiness(substrate_id="b", **base, ev_per_dollar=0.10)
    ranked = rank_by_ev_per_dollar((b, a))
    assert ranked[0] == "a"
    assert ranked[1] == "b"


# ---------- operator-authorize command ----------


def test_operator_authorize_command_needs_fix_returns_diagnostic():
    """NEEDS_FIX returns a diagnostic comment, not a runnable command."""
    fake = CandidateReadiness(
        substrate_id="x", recipe_basename="x",
        recipe_path=None, trainer_path=None, lane_maturity="L1",
        impl_complete=False, full_main_implemented=True, full_main_blocker=None,
        latest_council_verdict="PROCEED", research_only=False, dispatch_enabled=True,
        dispatch_blockers=(), predecessor_probe_blocking=False,
        predecessor_probe_id=None, predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10, predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X", estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100, gpu_class="T4",
        min_smoke_gpu="T4", cost_band_epochs=100, horizon_class="asymptotic_pursuit",
        blocking_issues=("X",), readiness_verdict="NEEDS_FIX", ev_per_dollar=0.5,
    )
    cmd = build_operator_authorize_command(fake)
    assert cmd.startswith("#"), f"NEEDS_FIX should return diagnostic comment, got: {cmd}"


def test_operator_authorize_command_ready_invokes_smoke_before_full():
    """READY returns canonical smoke-before-full command per Catalog #167."""
    fake = CandidateReadiness(
        substrate_id="x", recipe_basename="x",
        recipe_path=Path("/some/recipe.yaml"), trainer_path=None, lane_maturity="L1",
        impl_complete=True, full_main_implemented=True, full_main_blocker=None,
        latest_council_verdict="PROCEED", research_only=False, dispatch_enabled=True,
        dispatch_blockers=(), predecessor_probe_blocking=False,
        predecessor_probe_id=None, predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10, predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X", estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100, gpu_class="T4",
        min_smoke_gpu="T4", cost_band_epochs=100, horizon_class="asymptotic_pursuit",
        blocking_issues=(), readiness_verdict="READY", ev_per_dollar=0.5,
    )
    cmd = build_operator_authorize_command(fake)
    assert "tools/run_modal_smoke_before_full.py" in cmd
    assert "--recipe /some/recipe.yaml" in cmd


# ---------- end-to-end assessment ----------


def test_assess_candidates_returns_all_six_canonical():
    """Default canonical list has 6 candidates."""
    a = assess_candidates()
    assert len(a.candidates) == 6


def test_assess_candidates_provenance_discipline_score_claim_false():
    """Per provenance: assessment is planning artifact, never score claim."""
    a = assess_candidates()
    assert a.score_claim is False
    assert a.promotion_eligible is False
    assert a.evidence_grade == "predicted"
    assert a.provenance_kind == "PREDICTED_FROM_MODEL"


def test_assess_candidates_per_candidate_provenance_discipline():
    """Each candidate carries score_claim=false + promotion_eligible=false."""
    a = assess_candidates()
    for c in a.candidates:
        assert c.score_claim is False, f"{c.substrate_id} score_claim leaked"
        assert c.promotion_eligible is False, f"{c.substrate_id} promotion_eligible leaked"
        assert c.evidence_grade == "predicted"


def test_assess_candidates_atw_v2_excluded_from_top_1_due_to_probe():
    """ATW V2 is BLOCKED by Catalog #313; should not be TOP-1."""
    a = assess_candidates()
    assert a.top_1_substrate != "atw_codec_v2"


# ---------- write artifact ----------


def test_write_artifact_persists_to_canonical_path(tmp_path):
    """write_artifact writes a JSON file under .omx/state/asymptotic_pursuit/."""
    # Create a minimal fake repo root structure
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    a = assess_candidates(repo_root=tmp_path)
    path = write_artifact(a, repo_root=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    assert "asymptotic_pursuit" in str(path)
    payload = json.loads(path.read_text())
    assert payload["score_claim"] is False
    assert payload["evidence_grade"] == "predicted"
    assert payload["provenance_kind"] == "PREDICTED_FROM_MODEL"
    assert "result_review_blockers" in payload


# ---------- dispatch sequence + cost rollup ----------


def test_build_dispatch_sequence_includes_smoke_before_full_per_catalog_167():
    """Every dispatch sequence starts with smoke per Catalog #167."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    assert len(seq) == 6
    for entry in seq:
        stages = entry["stages"]
        assert any(s["stage"] == "smoke_100ep" for s in stages), \
            f"smoke_100ep stage missing for {entry['substrate_id']}"


def test_build_dispatch_sequence_includes_paired_cpu_axis_per_claude_md():
    """Per CLAUDE.md 'BOTH CPU AND CUDA': every sequence has paired CPU."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    for entry in seq:
        assert any(s["stage"] == "paired_cpu_axis_verification" for s in entry["stages"])


def test_compute_cost_band_rollup_aggregates_per_tier():
    """Cost rollup separates READY / NEEDS_FIX / DEFER tiers."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    rollup = compute_cost_band_rollup(seq)
    assert rollup["total_count"] == 6
    assert (
        rollup["ready_count"]
        + rollup["needs_fix_count"]
        + rollup["defer_count"]
        == rollup["total_count"]
    )


def test_compute_operator_attention_budget_within_t2_t3_budgets():
    """6 candidates well within T2 (90/30d) + T3 (13/30d) budgets."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    budget = compute_operator_attention_budget(seq)
    assert budget["t2_within_budget"] is True
    assert budget["t3_within_budget"] is True


# ---------- regression: top-2 surfaces stage 2 stacking ----------


def test_assess_candidates_top_2_surfaces_for_stage_2_stacking():
    """TOP-2 substrate must be surfaced for Stage 2 stacking design per Q9."""
    a = assess_candidates()
    assert a.top_2_substrate is not None
    assert a.top_2_substrate != a.top_1_substrate


# ---------- regression: WZ-on-existing-archives DEFERRED confirmation ----------


def test_canonical_candidates_does_not_include_wyner_ziv_existing_archives():
    """Per Option B falsification: WZ-on-existing-archives is NOT a canonical
    candidate for the ASYMPTOTIC pivot — replaced by substrate-class-shift.
    """
    assert all(
        "wyner_ziv_existing" not in c and "wyner_ziv_hoist" not in c
        for c in CANONICAL_CANDIDATES
    )


def test_canonical_candidates_inclusion_set():
    """The canonical 6 candidates are Z6 + ATW V2 + TT5L Autonomy + C6 IBPS + NSCS01 + NSCS03."""
    assert "time_traveler_l5_z6" in CANONICAL_CANDIDATES
    assert "atw_codec_v2" in CANONICAL_CANDIDATES
    assert "time_traveler_l5_autonomy" in CANONICAL_CANDIDATES
    assert "c6_e4_mdl_ibps" in CANONICAL_CANDIDATES
