# SPDX-License-Identifier: MIT
"""Focused tests for the G2 real-seed input-domain audit."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import warp_real_luma_frame0 as g1_warp
from tac.boundary_math.analytic_lane_render_band import LaneBandRenderConfig
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.optimization.predictor_upgrade_xi_chart import StaticCharts
from tac.optimization.resize_full_kernel import FullResizeKernel
from tools.measure_realization_g2_lattice import (
    CONTEXTUAL_STAGE_SCHEMA,
    INTERIOR_STAGE_SCHEMA,
    SOURCE_CONTROL_STAGE_SCHEMA,
    RealizationAuditError,
    _apply_local_chart_delta,
    _effective_chart_direction_count,
    _exception_parseback,
    _head_patch_144,
    _level_comparison,
    _prepare_chart_branch,
    _rank4_chart_directions,
    _select_chart_centerline_intercept,
    apply_contextual_rgb_exceptions,
    apply_dying_write_exceptions,
    audit_prefix,
    contextual_advected_rgb_plane,
    contextual_banded_projection,
    derive_bidirectional_amplitude_ladder,
    encode_contextual_rgb_exceptions,
    encode_dying_write_exceptions,
    interior_rgb_plane,
    openpilot_frame0_class_prior,
    parse_frozen_scorer_palette,
    protect_seed_class_sites,
    serialize_frozen_scorer_palette,
    summarize_chart_symbol_receiver_rows,
    summarize_contextual_prefix,
    summarize_interior_prefix,
    summarize_secant_prefix,
    summarize_source_control_prefix,
)


def _chart_symbol_stage(pair: int, *, admitted: bool) -> dict:
    predicates = {
        "semantic_cells_to_rgb_exact": admitted,
        "pose_within_tube": True,
        "zero_or_counted_bytes": True,
        "receiver_derived_rgb": True,
        "factor2_uint8_exact": True,
        "double_decode_identical": True,
        "rate_above_lambda": admitted,
    }
    return {
        "pair_index": pair,
        "selected_candidate": {
            "admitted": admitted,
            "admission_predicates": predicates,
            "packet": {"bytes": 20},
            "deltas": {"delta_d_seg": -0.01 if admitted else 0.0, "delta_d_pose": 0.0},
        },
    }


def test_g2g_chart_symbol_summary_preserves_hard_predicates_and_scope() -> None:
    summary = summarize_chart_symbol_receiver_rows(
        [_chart_symbol_stage(0, admitted=True), _chart_symbol_stage(34, admitted=False)],
        candidate_scope="four chart-only rescues",
    )
    assert summary["pair_indices"] == [0, 34]
    assert summary["admitted_pair_indices"] == [0]
    assert summary["first_admitted_realization_correction"] is True
    assert summary["selected_candidate_packet_bytes"] == 40
    assert summary["admission_predicate_failure_histogram"] == {
        "rate_above_lambda": 1,
        "semantic_cells_to_rgb_exact": 1,
    }
    assert summary["candidate_scope"] == "four chart-only rescues"
    assert summary["score_claim"] is False


def test_g2g_chart_symbol_summary_refuses_duplicate_or_unsorted_pairs() -> None:
    with pytest.raises(RealizationAuditError, match="sorted and unique"):
        summarize_chart_symbol_receiver_rows(
            [_chart_symbol_stage(34, admitted=False), _chart_symbol_stage(0, admitted=False)],
            candidate_scope="malformed",
        )


def _stage(pair: int) -> dict:
    return {
        "schema": "predict_project_pair_stage.v0",
        "pair_index": pair,
        "hard_oracle": {
            "schema": "predict_project_hard_oracle_pair.v0",
            "cell_exact": True,
            "pose_within_tube": True,
            "uint8_factor2_exact": False,
            "d_seg": 0.25,
            "d_pose": 0.0,
        },
    }


def test_audit_prefix_keeps_label_cell_exact_separate_from_rgb_lattice_exactness():
    constraints = [
        {"time": 0, "cell_id": 1, "stratum": "boundary_codim1"},
        {"time": 1, "cell_id": 0, "stratum": "critical_event"},
        {"time": 20, "cell_id": 4, "stratum": "cell_interior"},
    ]
    row = audit_prefix(prefix=16, stage_rows=[_stage(i) for i in range(16)], constraints=constraints)
    assert row["label_only_pair_count"] == 16
    assert row["rgb_projection_pair_count"] == 0
    assert row["uint8_factor2_exact_fraction"] is None
    assert row["declared_constraint_count"] == 2
    assert row["plane_level_cache_replay"]["cell_exact_pairs"] == 16
    assert row["status"] == "BLOCKED_INPUT_DOMAIN_LABEL_FIELD_IS_NOT_RGB_PLANE"


def test_audit_prefix_refuses_unaudited_rgb_fields_or_changed_settled_blocker():
    rows = [_stage(i) for i in range(16)]
    rows[0]["hard_oracle"]["projected_rgb_sha256"] = "f" * 64
    with pytest.raises(RealizationAuditError, match="unaudited RGB"):
        audit_prefix(prefix=16, stage_rows=rows, constraints=[])

    rows = [_stage(i) for i in range(16)]
    rows[0]["hard_oracle"]["uint8_factor2_exact"] = True
    with pytest.raises(RealizationAuditError, match="no longer matches"):
        audit_prefix(prefix=16, stage_rows=rows, constraints=[])


def _source_control_stage(pair: int, *, survives: bool) -> dict:
    return {
        "schema": SOURCE_CONTROL_STAGE_SCHEMA,
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "additional_seed_bytes": 1_179_648,
        "hard_oracle": {
            "d_seg_realized_vs_frozen_target": 0.001,
            "d_seg_description_vs_frozen_target": 0.343,
            "d_seg_realized_argmax_vs_description": 0.342,
            "realized_argmax_equals_description": False,
            "d_pose_realized_vs_frozen_target": 0.0001,
            "d_pose_realized_outside_declared_tube": 0.0,
            "pose_within_declared_tube": True,
        },
        "declared_write_survival": [{"class_id": 1, "stratum": "boundary_codim1", "survives": survives}],
        "timings_seconds": {
            "source_cache_load": 0.1,
            "seed_cell_decode": 0.2,
            "source_plane_projection": 0.3,
            "lattice_double_decode": 0.4,
            "native_cpu_torch_hard_oracle": 0.5,
            "total": 1.5,
        },
    }


def test_source_plane_prefix_proves_lattice_but_refuses_zero_byte_semantic_promotion():
    rows = [_source_control_stage(pair, survives=pair % 2 == 0) for pair in range(16)]
    summary = summarize_source_control_prefix(16, rows)
    assert summary["uint8_factor2_exact_fraction"] == 1.0
    assert summary["double_decode_identical_pair_count"] == 16
    assert summary["semantic_cells_to_rgb_exact_pair_count"] == 0
    assert summary["additional_seed_bytes_total"] == 16 * 1_179_648
    assert summary["zero_added_seed_byte_target_met"] is False
    assert summary["by_class"] == [
        {
            "class_id": 1,
            "declared_writes": 16,
            "surviving_writes": 8,
            "dying_writes": 8,
            "survival_fraction": 0.5,
        }
    ]
    assert summary["status"] == "MEASURED_SOURCE_RGB_CONTROL_NOT_SEED_RECEIVER"


def test_source_plane_prefix_refuses_noncontiguous_or_noncanonical_ladder():
    rows = [_source_control_stage(pair, survives=True) for pair in range(16)]
    rows[5]["pair_index"] = 6
    with pytest.raises(RealizationAuditError, match="contiguous"):
        summarize_source_control_prefix(16, rows)


def test_zero_byte_interior_rungs_are_deterministic_and_behaviorally_distinct():
    yy, xx = np.indices((384, 512))
    cells = ((yy // 37 + xx // 41) % 5).astype(np.uint8)
    r1 = interior_rgb_plane(cells, "R1_FIXED_MAGNITUDE")
    r2 = interior_rgb_plane(cells, "R2_MAX_MARGIN")
    r3a = interior_rgb_plane(cells, "R3_HOPFIELD_MEMORY_PROX")
    r3b = interior_rgb_plane(cells, "R3_HOPFIELD_MEMORY_PROX")
    assert r1.shape == (384, 512, 3) and r1.dtype == np.uint8
    assert not np.array_equal(r1, r2)
    assert not np.array_equal(r2, r3a)
    assert not np.array_equal(r1, r3a)
    assert np.array_equal(r3a, r3b)


def _site_seed() -> dict:
    return {
        "ground_chart": {
            "coordinate_quantum": {"numerator": 1, "denominator": 256},
            "cells": [
                {
                    "class_id": class_id,
                    "site_y_q": (20 + class_id * 30) * 256,
                    "site_x_q": (40 + class_id * 50) * 256,
                }
                for class_id in range(5)
            ],
        },
        "movable_tracks": [],
    }


def test_frozen_scorer_palette_is_counted_canonical_packet():
    payload = serialize_frozen_scorer_palette()
    palette = parse_frozen_scorer_palette(payload)
    assert len(payload) == 21
    assert palette.shape == (5, 3)
    assert serialize_frozen_scorer_palette() == payload
    with pytest.raises(RealizationAuditError, match="packet mismatch"):
        parse_frozen_scorer_palette(payload[:-1])


def test_openpilot_frame0_prior_reuses_charts_and_protects_every_class_site():
    seed = _site_seed()
    ru = np.ones((384, 512), dtype=np.uint8)
    ru[:120] = 2
    hood = np.zeros((384, 512), dtype=np.bool_)
    hood[300:] = True
    charts = StaticCharts(ru, hood, ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)))
    lane = np.zeros((384, 512), dtype=np.bool_)
    lane[180:280, 250:256] = True
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=(384, 512), pitch=-0.05)
    prior, sites = openpilot_frame0_class_prior(
        seed=seed,
        static_charts=charts,
        lane_mask=lane,
        geom=geom,
    )
    assert prior.dtype == np.uint8 and prior.shape == (384, 512)
    assert len(sites) == 5
    for site in sites:
        assert prior[site["y"], site["x"]] == site["class_id"]
    protected_again, sites_again = protect_seed_class_sites(prior, seed)
    assert np.array_equal(protected_again, prior)
    assert sites_again == sites


def test_r4_exception_stream_contains_only_dying_ordinals_and_roundtrips():
    constraints = [
        {"y": 3, "x": 4, "cell_id": 1, "stratum": "boundary_codim1"},
        {"y": 7, "x": 8, "cell_id": 0, "stratum": "critical_event"},
        {"y": 9, "x": 10, "cell_id": 1, "stratum": "boundary_codim1"},
    ]
    survival = [{"survives": False}, {"survives": True}, {"survives": False}]
    source = np.zeros((384, 512, 3), dtype=np.uint8)
    source[3, 4] = (11, 12, 13)
    source[9, 10] = (21, 22, 23)
    payload = encode_dying_write_exceptions(constraints, survival, source)
    assert len(payload) == 2 + 2 * 5
    base = np.full_like(source, 127)
    decoded, ordinals = apply_dying_write_exceptions(base, constraints, payload)
    assert ordinals == [0, 2]
    assert decoded[3, 4].tolist() == [11, 12, 13]
    assert decoded[9, 10].tolist() == [21, 22, 23]
    assert decoded[7, 8].tolist() == [127, 127, 127]


def test_contextual_exception_stream_uses_seed_ordinals_and_roundtrips_exactly():
    constraints = [
        {"y": 3, "x": 4, "cell_id": 1, "stratum": "boundary_codim1"},
        {"y": 7, "x": 8, "cell_id": 0, "stratum": "critical_event"},
        {"y": 9, "x": 10, "cell_id": 4, "stratum": "cell_interior"},
    ]
    base = np.full((384, 512, 3), 127, dtype=np.uint8)
    projected = base.copy()
    projected[3, 4] = (51, 255, 204)
    projected[9, 10] = (0, 255, 153)
    payload, ordinals = encode_contextual_rgb_exceptions(base, projected, constraints)
    decoded, decoded_ordinals = apply_contextual_rgb_exceptions(base, constraints, payload)
    assert ordinals == decoded_ordinals == [0, 2]
    assert np.array_equal(decoded, projected)
    assert encode_contextual_rgb_exceptions(base, decoded, constraints)[0] == payload

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(RealizationAuditError, match="checksum"):
        apply_contextual_rgb_exceptions(base, constraints, bytes(corrupted))


def test_contextual_banded_projection_changes_only_violated_sites_toward_prototype():
    constraints = [
        {"y": 3, "x": 4, "cell_id": 1, "stratum": "boundary_codim1"},
        {"y": 7, "x": 8, "cell_id": 0, "stratum": "critical_event"},
    ]
    base = np.full((384, 512, 3), 127, dtype=np.uint8)
    projected = contextual_banded_projection(base, constraints, [0], 32)
    assert projected[3, 4].tolist() == [95, 159, 159]
    assert projected[7, 8].tolist() == [127, 127, 127]
    changed = np.any(projected != base, axis=2)
    assert int(np.count_nonzero(changed)) == 1


def test_contextual_advected_rgb_identity_is_byte_exact_for_zero_xi():
    yy, xx = np.indices((384, 512))
    plane = np.stack((yy % 256, xx % 256, (yy + xx) % 256), axis=-1).astype(np.uint8)
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=(384, 512), pitch=-0.05)
    warped = contextual_advected_rgb_plane(plane, np.zeros(6), geom)
    assert np.array_equal(warped, plane)


def _contextual_stage(pair: int, *, survives: bool) -> dict:
    return {
        "schema": CONTEXTUAL_STAGE_SCHEMA,
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "hard_oracle": {
            "d_seg_realized_vs_frozen_target": 0.2,
            "d_seg_description_vs_frozen_target": 0.3,
            "d_seg_realized_argmax_vs_description": 0.4,
            "realized_argmax_equals_description": survives,
            "all_declared_writes_survive": survives,
            "d_pose_realized_vs_frozen_target": 0.01,
            "d_pose_realized_outside_declared_tube": 0.02,
            "pose_within_declared_tube": survives,
        },
        "declared_write_survival": [
            {
                "ordinal": 0,
                "class_id": 1,
                "stratum": "boundary_codim1",
                "survives": survives,
                "target_logit_margin": 0.5 if survives else -0.5,
                "margin_bucket": "positive_le_1" if survives else "nonpositive",
            }
        ],
        "exception_stream": {"bytes": 19, "record_count": 1},
        "projection": {"selected_band": 32},
    }


def test_contextual_prefix_keeps_whole_description_write_pose_and_bytes_separate():
    rows = [_contextual_stage(pair, survives=pair % 2 == 0) for pair in range(16)]
    summary = summarize_contextual_prefix(16, rows, bootstrap_bytes=100_000)
    assert summary["semantic_cells_to_rgb_exact_pair_count"] == 8
    assert summary["all_declared_writes_survive_pair_count"] == 8
    assert summary["declared_write_survival_fraction"] == 0.5
    assert summary["pose_within_declared_tube_pair_count"] == 8
    assert summary["margin_survival_contingency"] == {
        "positive_survives": 8,
        "positive_total": 8,
        "nonpositive_survives": 0,
        "nonpositive_total": 8,
    }
    assert summary["byte_accounting"]["contextual_total_bytes"] == 78_969 + 100_000 + 16 * 19
    assert summary["byte_accounting"]["fits_target_box"] is True


def test_g2e_rank4_chart_local_rounding_and_557_double_decode_are_deterministic():
    jacobian = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    first = _rank4_chart_directions(jacobian)
    second = _rank4_chart_directions(jacobian)
    assert first.shape == (6, 4)
    assert np.array_equal(first, second)
    constraints = [
        {"y": 3, "x": 4, "cell_id": 1, "stratum": "boundary_codim1"},
        {"y": 7, "x": 8, "cell_id": 0, "stratum": "critical_event"},
    ]
    baseline = np.full((384, 512, 3), 127, dtype=np.uint8)
    candidate, actual, saturation = _apply_local_chart_delta(baseline, constraints, 8.0 * first[:, 0])
    assert saturation == 0
    assert np.max(np.abs(actual)) <= 8.0
    payload, parsed_once, ordinals = _exception_parseback(baseline, constraints, candidate)
    _, parsed_twice, ordinals_twice = _exception_parseback(baseline, constraints, candidate)
    assert payload
    assert ordinals == ordinals_twice
    assert np.array_equal(parsed_once, parsed_twice)


def test_g2e_rank4_chart_zero_pads_single_write_three_rgb_chart() -> None:
    jacobian = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float64)
    first = _rank4_chart_directions(jacobian)
    second = _rank4_chart_directions(jacobian)
    assert first.shape == (3, 4)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first[:, :3]), axis=0) == pytest.approx(np.ones(3))
    assert np.array_equal(first[:, 3], np.zeros(3))
    for vector in first[:, :3].T:
        response_sum = float(np.sum(jacobian @ vector))
        pivot = int(np.argmax(np.abs(vector)))
        assert response_sum >= 0.0
        if response_sum == 0.0:
            assert vector[pivot] > 0.0


def test_g2f_amplitude_ladder_is_derived_from_exact_r_gain_and_g2e_extent() -> None:
    prior = {"secant_observations": [{"signed_amplitude": (4.0, -4.0, 8.0, -8.0)[index % 4]} for index in range(64)]}
    amplitudes, custody = derive_bidirectional_amplitude_ladder(
        FullResizeKernel.build(),
        prior,
    )
    assert amplitudes == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert custody["lsb_lawref_id"] == "witness_realization_lsb_regime_v1"
    assert custody["r_operator_lawref_id"] == "separable_resize_full_kernel_direct_sum_v1"
    assert custody["exact_r_induced_linf_gain"] == pytest.approx(1.0)
    assert custody["constant_guessed"] is False


def test_g2f_effective_direction_count_refuses_interior_zero_holes() -> None:
    assert _effective_chart_direction_count(np.eye(4)) == 4
    zero_padded = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    assert _effective_chart_direction_count(zero_padded) == 3
    with pytest.raises(RealizationAuditError, match="contiguous rank prefix"):
        _effective_chart_direction_count(np.asarray([[1.0, 0.0, 1.0, 0.0]]))


def test_g2f_chart_intercept_move_is_coherent_normalized_and_deterministic() -> None:
    lines = [
        LaneLine(
            centerline_coeffs=np.asarray([0.0]),
            halfwidth_coeffs=np.asarray([0.0, halfwidth]),
            forward_range=(0.0, 10_000.0),
        )
        for halfwidth in (1.0, 3.0)
    ]
    config = LaneBandRenderConfig(dash_gate=False)
    line_index, coefficient_index, gain, baseline_coverage = _select_chart_centerline_intercept(lines, config)
    assert line_index == 1
    assert coefficient_index == 0
    assert gain > 0.0
    base = np.full((384, 512, 3), 127, dtype=np.uint8)
    palette = np.asarray(((100, 100, 100), (200, 180, 160)), dtype=np.uint8)
    first = _prepare_chart_branch(
        base1=base,
        lines=lines,
        config=config,
        palette=palette,
        line_index=line_index,
        coefficient_gain_pixels_per_unit=gain,
        signed_amplitude=2.0,
        baseline_coverage=baseline_coverage,
    )
    second = _prepare_chart_branch(
        base1=base,
        lines=lines,
        config=config,
        palette=palette,
        line_index=line_index,
        coefficient_gain_pixels_per_unit=gain,
        signed_amplitude=2.0,
        baseline_coverage=baseline_coverage,
    )
    probe, delta, saturation, custody = first
    assert np.array_equal(probe, second[0])
    assert np.array_equal(delta, second[1])
    assert custody == second[3]
    assert saturation == 0
    assert custody.coefficient_delta == pytest.approx(2.0 / gain)
    assert custody.max_centerline_displacement_pixels == pytest.approx(2.0)
    assert custody.changed_pixel_count > 1
    assert custody.changed_rgb_value_count >= custody.changed_pixel_count
    assert np.max(np.abs(delta)) > 0


def test_g2f_level_comparison_preserves_both_levels_and_rescues() -> None:
    comparison = _level_comparison(
        [False, True, False, True],
        [True, True, False, False],
    )
    assert comparison["chart_only_rescued_pair_indices"] == [0]
    assert comparison["pixel_only_pair_indices"] == [3]
    assert comparison["both_selected_pair_indices"] == [1]
    assert comparison["neither_selected_pair_indices"] == [2]


def test_g2e_head_patch_is_exact_144d_zero_padded_input() -> None:
    feature = np.arange(16 * 4 * 5, dtype=np.float64).reshape(16, 4, 5)
    interior = _head_patch_144(feature, 2, 3)
    corner = np.asarray(_head_patch_144(feature, 0, 0)).reshape(16, 3, 3)
    assert len(interior) == 144
    assert np.all(corner[:, 0, :] == 0.0)
    assert np.all(corner[:, :, 0] == 0.0)


def _secant_stage(pair: int, *, admitted: bool) -> dict:
    return {
        "schema": "realization_g2e_secant_pair.v1",
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "pair_solve": {
            "status": "ADMITTED_RECEIVER_CLOSED" if admitted else "TRUST_REGION_REFUSED",
            "admitted": admitted,
        },
        "correction_packet": {"bytes": 5 if admitted else 0},
        "hard_oracle": {
            "realized_argmax_equals_description": admitted,
            "all_declared_writes_survive": admitted,
            "d_seg_realized_vs_frozen_target": 0.2,
            "d_pose_realized_vs_frozen_target": 0.01,
            "d_pose_realized_outside_declared_tube": 0.02,
            "pose_within_declared_tube": admitted,
        },
        "declared_write_survival": [
            {
                "class_id": 1,
                "stratum": "boundary_codim1",
                "margin_bucket": "positive_le_1" if admitted else "nonpositive",
                "survives": admitted,
            }
        ],
    }


def test_g2e_prefix_preserves_refusals_and_rate_pose_decomposition() -> None:
    rows = [_secant_stage(pair, admitted=pair % 2 == 0) for pair in range(16)]
    summary = summarize_secant_prefix(16, rows, base_bytes=200_000)
    assert summary["whole_description_exact_pair_count"] == 8
    assert summary["uint8_factor2_exact_pair_count"] == 16
    assert summary["double_decode_identical_pair_count"] == 16
    assert summary["admitted_pair_count"] == 8
    assert summary["solve_status_histogram"] == {
        "ADMITTED_RECEIVER_CLOSED": 8,
        "TRUST_REGION_REFUSED": 8,
    }
    assert summary["byte_accounting"]["correction_bytes"] == 40
    assert summary["mean_d_pose_declared_tube_debt"] == pytest.approx(0.02)
    admissibility = summary["predict_project_realization_admissibility_v1"]
    assert admissibility["accepted"] is False
    assert admissibility["predicates"]["factor2_uint8_exact"] is True
    assert admissibility["predicates"]["double_decode_identical"] is True
    assert admissibility["predicates"]["semantic_cells_to_rgb_exact"] is False
    assert admissibility["predicates"]["zero_added_seed_bytes"] is False
    assert admissibility["predicates"]["receiver_derived_rgb"] is False


def _interior_stage(pair: int, *, survives: bool, rung: str = "R2_MAX_MARGIN") -> dict:
    return {
        "schema": INTERIOR_STAGE_SCHEMA,
        "rung_id": rung,
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "additional_seed_bytes": 0,
        "hard_oracle": {
            "d_seg_realized_vs_frozen_target": 0.2,
            "d_seg_description_vs_frozen_target": 0.343,
            "d_seg_realized_argmax_vs_description": 0.4,
            "realized_argmax_equals_description": False,
            "all_declared_writes_survive": survives,
            "d_pose_realized_vs_frozen_target": 0.01,
            "d_pose_realized_outside_declared_tube": 0.02,
            "pose_within_declared_tube": False,
        },
        "declared_write_survival": [
            {
                "ordinal": 0,
                "class_id": 1,
                "stratum": "boundary_codim1",
                "survives": survives,
                "target_logit_margin": 0.5 if survives else -0.5,
                "margin_bucket": "positive_le_1" if survives else "nonpositive",
            }
        ],
        "exception_stream": {
            "record_count": 0,
            "records": [],
        },
        "timings_seconds": {
            "source_cache_load_shared": 0.1,
            "seed_cell_decode_shared": 0.1,
            "rgb_plane_decode": 0.1,
            "lattice_double_decode": 0.2,
            "native_cpu_torch_hard_oracle": 0.3,
            "rung_total": 0.7,
            "pair_total_to_stage": 0.7,
        },
    }


def test_interior_prefix_reports_pair_and_write_exactness_separately():
    rows = [_interior_stage(pair, survives=pair % 2 == 0) for pair in range(16)]
    summary = summarize_interior_prefix(16, rows, rung_id="R2_MAX_MARGIN")
    assert summary["uint8_factor2_exact_fraction"] == 1.0
    assert summary["semantic_cells_to_rgb_exact_pair_count"] == 0
    assert summary["all_declared_writes_survive_pair_count"] == 8
    assert summary["declared_write_survival_fraction"] == 0.5
    assert summary["by_margin_bucket"] == [
        {
            "margin_bucket": "nonpositive",
            "declared_writes": 8,
            "surviving_writes": 0,
            "dying_writes": 8,
            "survival_fraction": 0.0,
        },
        {
            "margin_bucket": "positive_le_1",
            "declared_writes": 8,
            "surviving_writes": 8,
            "dying_writes": 0,
            "survival_fraction": 1.0,
        },
    ]
