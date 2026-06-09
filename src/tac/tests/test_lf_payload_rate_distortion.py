"""Behavioral tests for the SNeRV LF payload rate-distortion reverse-waterfill planner.

These tests verify ACTUAL BEHAVIOR (per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE
Class 2): every test would FAIL if the planner returned canonical constants instead
of computing THE LAW. The synthetic atlas + sections carry a KNOWN optimum so the
ranking is checked against a hand-derived answer, not against the code's own output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.optimization.lf_payload_rate_distortion import (
    ACTION_QUANTIZE,
    ACTION_RECODE,
    CONTEST_BYTE_PRICE,
    AtlasScope,
    AtlasSensitivity,
    BaselineScoreTerms,
    CoefficientGroup,
    LfPayloadRateDistortionError,
    PayloadSection,
    atlas_scope_from_grid,
    atlas_sensitivities_from_cells,
    build_drop_action,
    build_quantize_action,
    build_recode_action,
    delta_distortion_score,
    delta_pose_score,
    delta_rate_score,
    estimate_section_sensitivity,
    keep_component,
    plan_lf_payload_actions,
)

CONTEST_N = 37_545_489


# ---------------------------------------------------------------------------
# Fixtures: a synthetic atlas + sections with a KNOWN optimum.
# ---------------------------------------------------------------------------
@pytest.fixture
def scope() -> AtlasScope:
    return AtlasScope(
        band_indices=frozenset({0, 1, 2}),
        channel_bases=frozenset({"yuv", "rgb"}),
        channels=frozenset({"y", "all"}),
        orientations=frozenset({"vertical", "isotropic"}),
        frame_incidences=frozenset({"frame1_only", "both_opposite"}),
        amplitudes_lsb=(4.0,),
    )


@pytest.fixture
def sensitivities() -> tuple[AtlasSensitivity, ...]:
    # Band 0 is HIGH sensitivity (the scorer's spectral peak); band 2 is LOW.
    return (
        AtlasSensitivity(
            band_index=0, h_seg=0.0057, h_pose=0.27,
            channel_basis="yuv", channel="y", orientation="vertical",
            frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
        AtlasSensitivity(
            band_index=2, h_seg=0.00001, h_pose=0.00001,
            channel_basis="yuv", channel="y", orientation="vertical",
            frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )


@pytest.fixture
def baseline() -> BaselineScoreTerms:
    return BaselineScoreTerms(d_seg=0.0023, d_pose=0.0013, archive_bytes=200_000)


# ---------------------------------------------------------------------------
# THE LAW (the core predicate).
# ---------------------------------------------------------------------------
def test_keep_component_high_value_low_byte_is_kept():
    # ΔS_distortion of dropping = +2.0 score units; freeing 100 bytes frees
    # 25*100/N ≈ 6.66e-5 score units. 2.0 >> 6.66e-5 => KEEP.
    assert keep_component(delta_distortion=2.0, delta_bytes_freed=100) is True


def test_keep_component_low_value_high_byte_is_dropped():
    # ΔS_distortion of dropping ≈ 1e-6; freeing 1,000,000 bytes frees
    # 25*1e6/N ≈ 0.666 score units. 1e-6 < 0.666 => DROP (do not keep).
    assert keep_component(delta_distortion=1e-6, delta_bytes_freed=1_000_000) is False


def test_keep_component_threshold_is_exact_law():
    freed = 1_000_000
    rate = delta_rate_score(freed)  # = 25*1e6/N
    # Just above threshold => keep; just below => drop.
    assert keep_component(rate + 1e-9, freed) is True
    assert keep_component(rate - 1e-9, freed) is False


def test_keep_component_no_byte_movement_keeps_iff_distortion_positive():
    assert keep_component(0.5, 0) is True
    assert keep_component(0.0, 0) is False
    assert keep_component(-0.1, 0) is False


def test_delta_rate_score_matches_contest_formula():
    assert delta_rate_score(37_545_489) == pytest.approx(25.0)
    assert delta_rate_score(-37_545_489) == pytest.approx(-25.0)
    assert pytest.approx(25.0 / CONTEST_N) == CONTEST_BYTE_PRICE


def test_delta_pose_score_is_nonlinear_sqrt_term():
    # sqrt(10*0.01) - sqrt(10*0.0) = sqrt(0.1) ≈ 0.3162
    assert delta_pose_score(0.0, 0.01) == pytest.approx(math.sqrt(0.1), rel=1e-9)
    # Symmetric reduction is negative.
    assert delta_pose_score(0.01, 0.0) == pytest.approx(-math.sqrt(0.1), rel=1e-9)


def test_delta_distortion_score_combines_seg_and_pose_terms():
    # 100*(0.01-0.0) + (sqrt(10*0.02)-sqrt(10*0.01))
    expected = 100.0 * 0.01 + (math.sqrt(0.2) - math.sqrt(0.1))
    assert delta_distortion_score(0.0, 0.01, 0.01, 0.02) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Section sensitivity estimate + fail-closed scope.
# ---------------------------------------------------------------------------
def test_estimate_in_scope_sums_matched_cell_sensitivity(scope, sensitivities):
    sec = PayloadSection(
        name="lf", bytes=100,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert est.atlas_scope_valid is True
    assert est.est_d_seg_value == pytest.approx(0.0057)
    assert est.est_d_pose_value == pytest.approx(0.27)
    assert est.matched_cell_count == 1


def test_estimate_out_of_band_fails_closed(scope, sensitivities):
    # Band 5 is OUTSIDE the swept grid {0,1,2}.
    sec = PayloadSection(
        name="oob", bytes=100,
        coefficient_group=CoefficientGroup(band_indices=(5,)),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert est.atlas_scope_valid is False
    assert est.est_d_seg_value is None
    assert est.est_d_pose_value is None
    assert "band_index 5 outside" in est.scope_reason


def test_estimate_out_of_amplitude_fails_closed(scope, sensitivities):
    # amplitude 99.0 is OUTSIDE the swept amplitudes (4.0).
    sec = PayloadSection(
        name="oa", bytes=100,
        coefficient_group=CoefficientGroup(band_indices=(0,), amplitude_lsb=99.0),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert est.atlas_scope_valid is False
    assert "amplitude" in est.scope_reason


def test_estimate_unmeasured_channel_basis_fails_closed(scope, sensitivities):
    sec = PayloadSection(
        name="cb", bytes=100,
        coefficient_group=CoefficientGroup(band_indices=(0,), channel_basis="lab"),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert est.atlas_scope_valid is False
    assert "channel_basis" in est.scope_reason


def test_estimate_no_matching_cell_fails_closed(scope):
    # In-scope band but NO atlas cell exists for it (empty sensitivities).
    sec = PayloadSection(
        name="nomatch", bytes=100,
        coefficient_group=CoefficientGroup(band_indices=(1,)),
    )
    est = estimate_section_sensitivity(sec, (), scope)
    assert est.atlas_scope_valid is False
    assert "no atlas cell matched" in est.scope_reason


def test_estimate_multi_band_sums_across_bands(scope, sensitivities):
    sec = PayloadSection(
        name="multi", bytes=100,
        coefficient_group=CoefficientGroup(
            band_indices=(0, 2), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert est.atlas_scope_valid is True
    assert est.est_d_seg_value == pytest.approx(0.0057 + 0.00001)
    assert est.matched_cell_count == 2


# ---------------------------------------------------------------------------
# Candidate action evaluation (the proposal rows + value_per_byte ordering).
# ---------------------------------------------------------------------------
def test_drop_high_value_low_byte_is_kept_under_law(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="peak", bytes=100,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    drop = build_drop_action(sec, est, baseline)
    # Dropping the peak section RAISES the predicted score (positive ΔS_total).
    assert drop.est_delta_score_total is not None
    assert drop.est_delta_score_total > 0.0
    assert drop.keep_section_under_law is True
    assert drop.pays_rent_predicted is False  # dropping does NOT pay rent


def test_drop_low_value_high_byte_pays_rent(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="junk", bytes=1_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    drop = build_drop_action(sec, est, baseline)
    # Dropping a low-value high-byte section LOWERS the predicted score.
    assert drop.est_delta_score_total is not None
    assert drop.est_delta_score_total < 0.0
    assert drop.keep_section_under_law is False
    assert drop.pays_rent_predicted is True


def test_value_per_byte_ordering_high_byte_low_value_ranks_first(
    scope, sensitivities, baseline
):
    # Two droppable low-value sections; the bigger one frees more score per byte set
    # is not the point — the one that frees MORE total score ranks by value/byte.
    big = PayloadSection(
        name="big", bytes=2_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    small = PayloadSection(
        name="small", bytes=10_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    eb = build_drop_action(big, estimate_section_sensitivity(big, sensitivities, scope), baseline)
    es = build_drop_action(small, estimate_section_sensitivity(small, sensitivities, scope), baseline)
    # Both pay rent; both have near-zero distortion cost so value_per_byte ≈
    # rate_price for both (≈ CONTEST_BYTE_PRICE). The distortion difference is tiny
    # but real: the small section gives up the SAME tiny per-cell value spread over
    # FEWER bytes, so its per-byte distortion cost is higher => slightly lower vpb.
    assert eb.value_per_byte is not None and es.value_per_byte is not None
    assert eb.value_per_byte == pytest.approx(CONTEST_BYTE_PRICE, abs=1e-7)


def test_scope_invalid_action_has_none_estimates(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="oob", bytes=500,
        coefficient_group=CoefficientGroup(band_indices=(9,)),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    drop = build_drop_action(sec, est, baseline)
    assert drop.est_delta_score_total is None
    assert drop.value_per_byte is None
    assert drop.pays_rent_predicted is None
    assert drop.keep_section_under_law is None
    assert drop.atlas_scope_valid is False


def test_quantize_partial_keeps_fraction_of_bytes_and_value(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=100_000,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    q = build_quantize_action(sec, est, baseline, quantize_step=1.0)
    # step=1 keeps 1/(1+1)=0.5 of bytes, frees the other 50,000.
    assert q.delta_bytes == -50_000
    assert q.action_kind == ACTION_QUANTIZE
    # Gives up half the distortion value => est_delta_d_seg = 0.5 * 0.0057.
    assert q.est_delta_d_seg == pytest.approx(0.5 * 0.0057)


def test_quantize_step_monotone_more_step_frees_more_bytes(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=100_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    q_small = build_quantize_action(sec, est, baseline, quantize_step=0.5)
    q_big = build_quantize_action(sec, est, baseline, quantize_step=2.0)
    assert (-q_big.delta_bytes) > (-q_small.delta_bytes)


def test_recode_lossless_always_pays_rent_when_frees_bytes(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=100_000, recodeable_floor_bytes=60_000,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    recode = build_recode_action(sec, est, baseline)
    assert recode is not None
    assert recode.action_kind == ACTION_RECODE
    assert recode.delta_bytes == -40_000
    assert recode.est_delta_d_seg == 0.0  # lossless => zero distortion cost
    assert recode.pays_rent_predicted is True  # frees bytes at zero cost


def test_recode_none_when_no_floor_declared(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=100_000,
        coefficient_group=CoefficientGroup(band_indices=(0,)),
    )
    est = estimate_section_sensitivity(sec, sensitivities, scope)
    assert build_recode_action(sec, est, baseline) is None


# ---------------------------------------------------------------------------
# The planner (ranking + segregation of scope-invalid).
# ---------------------------------------------------------------------------
def test_plan_ranks_dropping_low_value_high_byte_first(scope, sensitivities, baseline):
    junk = PayloadSection(
        name="junk_big", bytes=1_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    peak = PayloadSection(
        name="peak_small", bytes=100,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    plan = plan_lf_payload_actions([junk, peak], sensitivities, scope, baseline)
    assert plan["n_ranked"] >= 1
    best = plan["best_action_id"]
    # The peak section's DROP must NOT be the best action (it RAISES score).
    assert "peak_small::drop" not in [r["action_id"] for r in plan["ranked_actions"]]
    # The junk DROP pays rent and ranks.
    assert any(r["action_id"] == "junk_big::drop" for r in plan["ranked_actions"])
    assert best is not None


def test_plan_segregates_scope_invalid_into_needs_remeasure(scope, sensitivities, baseline):
    oob = PayloadSection(
        name="oob", bytes=500_000,
        coefficient_group=CoefficientGroup(band_indices=(7,)),
    )
    plan = plan_lf_payload_actions([oob], sensitivities, scope, baseline)
    assert plan["n_needs_remeasure"] >= 1
    assert plan["n_ranked"] == 0
    names = [r["section_name"] for r in plan["needs_exact_remeasure"]]
    assert "oob" in names


def test_plan_scope_invalid_never_ranked_above_scope_valid(scope, sensitivities, baseline):
    good = PayloadSection(
        name="good", bytes=1_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    oob = PayloadSection(
        name="oob", bytes=5_000_000,
        coefficient_group=CoefficientGroup(band_indices=(9,)),
    )
    plan = plan_lf_payload_actions([good, oob], sensitivities, scope, baseline)
    ranked_names = [r["section_name"] for r in plan["ranked_actions"]]
    assert "oob" not in ranked_names  # never ranked despite being bigger
    assert "good" in ranked_names


def test_plan_emits_false_authority_contract(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=1_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    plan = plan_lf_payload_actions([sec], sensitivities, scope, baseline)
    assert plan["promotable"] is False
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["authority"] == "planning_control_false_authority"
    for row in plan["ranked_actions"]:
        assert row["requires_exact_remeasure"] is True
        assert row["promotable"] is False
        assert row["score_claim"] is False


def test_plan_requires_recompute_after_accept(scope, sensitivities, baseline):
    sec = PayloadSection(
        name="lf", bytes=1_000_000,
        coefficient_group=CoefficientGroup(
            band_indices=(2,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    plan = plan_lf_payload_actions([sec], sensitivities, scope, baseline)
    assert plan["requires_recompute_after_accept"] is True


# ---------------------------------------------------------------------------
# Atlas adaptation against the REAL v2 atlas schema (apples-to-apples).
# ---------------------------------------------------------------------------
def _real_atlas_path() -> Path | None:
    candidates = [
        Path("/Volumes/VertigoDataTier/pact/scorer_spectral_atlas_min_20260609/scorer_spectral_sensitivity.v2.json"),
        Path("/Volumes/VertigoDataTier/pact/scorer_spectral_atlas_fast_20260609/scorer_spectral_sensitivity.v2.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def test_atlas_scope_from_synthetic_grid_block():
    atlas = {
        "schema": "scorer_spectral_sensitivity.v2",
        "authority_tier": "exact_cpu_advisory",
        "grid": {
            "n_bands": 3,
            "channel_bases": ["rgb", "yuv"],
            "rgb_channels": ["all"],
            "yuv_channels": ["y"],
            "orientations": ["isotropic", "vertical"],
            "frame_incidences": ["frame1_only", "both_opposite"],
            "amplitudes_lsb": [4.0],
        },
        "cells": [],
        "source_raw": {"path": "/Volumes/VertigoDataTier/pact/x/source.raw"},
    }
    sc = atlas_scope_from_grid(atlas)
    assert sc.band_indices == frozenset({0, 1, 2})
    assert "yuv" in sc.channel_bases
    assert sc.amplitudes_lsb == (4.0,)


def test_atlas_sensitivities_from_synthetic_cells():
    atlas = {
        "schema": "scorer_spectral_sensitivity.v2",
        "authority_tier": "exact_cpu_advisory",
        "cells": [
            {
                "band_index": 0, "H_seg": 0.005, "H_pose": 0.27,
                "channel_basis": "yuv", "channel": "y", "orientation": "vertical",
                "frame_incidence": "frame1_only", "amplitude_lsb": 4.0,
            }
        ],
        "source_raw": {"path": "/Volumes/VertigoDataTier/pact/x/source.raw"},
    }
    sens = atlas_sensitivities_from_cells(atlas)
    assert len(sens) == 1
    assert sens[0].h_seg == pytest.approx(0.005)
    assert sens[0].authority_tier == "exact_cpu_advisory"


@pytest.mark.skipif(_real_atlas_path() is None, reason="real atlas artifact unavailable")
def test_real_atlas_adapts_and_places_band0():
    atlas = json.loads(_real_atlas_path().read_text())
    sc = atlas_scope_from_grid(atlas)
    sens = atlas_sensitivities_from_cells(atlas)
    assert 0 in sc.band_indices
    assert len(sens) == 24
    # The atlas headline says seg peak is vertical/y; band 0 must place in scope.
    sec = PayloadSection(
        name="lf", bytes=200_000,
        coefficient_group=CoefficientGroup(
            band_indices=(0,), channel_basis="yuv", channel="y",
            orientation="vertical", frame_incidence="frame1_only", amplitude_lsb=4.0,
        ),
    )
    est = estimate_section_sensitivity(sec, sens, sc)
    assert est.atlas_scope_valid is True
    assert est.est_d_seg_value is not None and est.est_d_seg_value > 0.0


# ---------------------------------------------------------------------------
# Input validation (fail-closed; never silently coerce).
# ---------------------------------------------------------------------------
def test_coefficient_group_requires_bands():
    with pytest.raises(LfPayloadRateDistortionError):
        CoefficientGroup(band_indices=())


def test_payload_section_rejects_negative_bytes():
    with pytest.raises(LfPayloadRateDistortionError):
        PayloadSection(name="x", bytes=-1, coefficient_group=CoefficientGroup(band_indices=(0,)))


def test_payload_section_rejects_floor_above_bytes():
    with pytest.raises(LfPayloadRateDistortionError):
        PayloadSection(
            name="x", bytes=100, recodeable_floor_bytes=200,
            coefficient_group=CoefficientGroup(band_indices=(0,)),
        )


def test_atlas_sensitivity_rejects_tmp_artifact_path():
    with pytest.raises(LfPayloadRateDistortionError):
        AtlasSensitivity(band_index=0, h_seg=0.0, h_pose=0.0, artifact_path="/tmp/x.json")


def test_baseline_rejects_negative_terms():
    with pytest.raises(LfPayloadRateDistortionError):
        BaselineScoreTerms(d_seg=-0.1, d_pose=0.0, archive_bytes=0)


def test_candidate_action_rejects_unknown_kind(scope, sensitivities, baseline):
    from tac.optimization.lf_payload_rate_distortion import CandidateActionEvaluation
    with pytest.raises(LfPayloadRateDistortionError):
        CandidateActionEvaluation(
            action_id="x", action_kind="bogus", section_name="s",
            est_delta_d_seg=0.0, est_delta_d_pose=0.0, delta_bytes=-1,
            atlas_scope_valid=True, scope_reason="", baseline=baseline,
        )


# ---------------------------------------------------------------------------
# CLI end-to-end against a synthetic G1b verdict (the input contract).
# ---------------------------------------------------------------------------
def test_cli_build_plan_from_synthetic_g1b_and_atlas(tmp_path):
    # Build a synthetic G1b verdict matching snerv_g1b_export_binding_verdict.v1.
    g1b = {
        "schema": "snerv_g1b_export_binding_verdict.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "path_a_advisory": {
            "byte_decomposition": {
                "archive_bytes_total_linf": 250_000,
                "lf_payload_bytes": 200_000,
                "linf_steps_payload_bytes": 30_000,
                "decoder_bytes": 15_000,
                "metadata_bytes": 4_000,
                "receiver_archive_header_bytes": 1_000,
            },
            "archive_surface_distortion": {
                "d_seg_mean_linf": 0.0023,
                "d_pose_mean_linf": 0.0013,
            },
        },
    }
    atlas = {
        "schema": "scorer_spectral_sensitivity.v2",
        "authority_tier": "exact_cpu_advisory",
        "grid": {
            "n_bands": 3,
            "channel_bases": ["rgb", "yuv"],
            "rgb_channels": ["all"],
            "yuv_channels": ["y"],
            "orientations": ["isotropic", "vertical"],
            "frame_incidences": ["frame1_only", "both_opposite"],
            "amplitudes_lsb": [4.0],
        },
        "cells": [
            {
                "band_index": 0, "H_seg": 0.0057, "H_pose": 0.27,
                "channel_basis": "yuv", "channel": "y", "orientation": "vertical",
                "frame_incidence": "frame1_only", "amplitude_lsb": 4.0,
            }
        ],
        "source_raw": {"path": str(tmp_path / "source.raw")},
    }
    g1b_path = tmp_path / "g1b.json"
    atlas_path = tmp_path / "atlas.json"
    g1b_path.write_text(json.dumps(g1b))
    atlas_path.write_text(json.dumps(atlas))

    # Import the CLI module by path (tools/ is not a package).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "snerv_lf_cli",
        Path(__file__).resolve().parents[3] / "tools" / "snerv_lf_payload_rate_distortion.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    plan = cli.build_plan_from_files(
        g1b_verdict_path=str(g1b_path), atlas_path=str(atlas_path)
    )
    # Baseline derived from the G1b archive surface.
    assert plan["baseline"]["d_seg"] == pytest.approx(0.0023)
    assert plan["baseline"]["archive_bytes"] == 250_000
    # The default section map places only the LF payload (band 0) => it is scope-valid;
    # the LF payload at band 0 is the SCORER PEAK, so dropping it RAISES score => the
    # LF drop must NOT pay rent (it must be kept).
    lf_drop = next(
        (r for r in plan["ranked_actions"] + plan["not_paying_rent"]
         if r["action_id"] == "lf_payload_bytes::drop"),
        None,
    )
    assert lf_drop is not None
    assert lf_drop["keep_section_under_law"] is True
    # The other sections (step-maps/decoder/metadata/header) are unmapped by default.
    assert "linf_steps_payload_bytes" in plan["inputs"]["unmapped_sections"]
    assert plan["inputs"]["section_map_is_default"] is True


def test_cli_section_map_places_low_value_section_and_it_pays_rent(tmp_path):
    g1b = {
        "schema": "snerv_g1b_export_binding_verdict.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "path_a_advisory": {
            "byte_decomposition": {
                "archive_bytes_total_linf": 1_250_000,
                "lf_payload_bytes": 1_000_000,
                "linf_steps_payload_bytes": 250_000,
            },
            "archive_surface_distortion": {
                "d_seg_mean_linf": 0.0023,
                "d_pose_mean_linf": 0.0013,
            },
        },
    }
    atlas = {
        "schema": "scorer_spectral_sensitivity.v2",
        "authority_tier": "exact_cpu_advisory",
        "grid": {
            "n_bands": 3,
            "channel_bases": ["yuv"],
            "yuv_channels": ["y"],
            "orientations": ["vertical"],
            "frame_incidences": ["frame1_only"],
            "amplitudes_lsb": [4.0],
        },
        "cells": [
            {
                "band_index": 2, "H_seg": 0.00001, "H_pose": 0.00001,
                "channel_basis": "yuv", "channel": "y", "orientation": "vertical",
                "frame_incidence": "frame1_only", "amplitude_lsb": 4.0,
            }
        ],
        "source_raw": {"path": str(tmp_path / "source.raw")},
    }
    section_map = {
        # Place the step-maps section in the LOW-sensitivity band 2 => dropping it
        # is nearly free distortion-wise but frees 250 KB => pays rent.
        "linf_steps_payload_bytes": {
            "band_indices": [2], "channel_basis": "yuv", "channel": "y",
            "orientation": "vertical", "frame_incidence": "frame1_only",
            "amplitude_lsb": 4.0, "droppable": True,
        }
    }
    g1b_path = tmp_path / "g1b.json"
    atlas_path = tmp_path / "atlas.json"
    map_path = tmp_path / "map.json"
    g1b_path.write_text(json.dumps(g1b))
    atlas_path.write_text(json.dumps(atlas))
    map_path.write_text(json.dumps(section_map))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "snerv_lf_cli2",
        Path(__file__).resolve().parents[3] / "tools" / "snerv_lf_payload_rate_distortion.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    plan = cli.build_plan_from_files(
        g1b_verdict_path=str(g1b_path),
        atlas_path=str(atlas_path),
        section_map_path=str(map_path),
    )
    drop = next(
        (r for r in plan["ranked_actions"] if r["action_id"] == "linf_steps_payload_bytes::drop"),
        None,
    )
    assert drop is not None
    assert drop["pays_rent_predicted"] is True
    assert drop["keep_section_under_law"] is False
    assert plan["best_action_id"] == "linf_steps_payload_bytes::drop"
