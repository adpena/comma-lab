# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_continuous_paint_ceiling import (
    ContinuousPaintError,
    apply_global_channel_statistics,
    decompose_mechanisms,
    encode_global_channel_statistics,
    encode_stratum_spectrum_coefficients,
    fit_global_channel_statistics,
    measure_fitted_geometry_sdwl1,
    render_analytic_coverage_blend,
    render_hard_camera_placement,
    render_stratum_spectrum_match,
    scorer_native_divergence_rows,
    signed_distance_fields,
    solve_stratum_spectrum_coefficients,
    split_curve_provenance,
    stage_transition,
    stratum_spectrum_components,
    stratum_spectrum_normal_equations,
)
from tac.through_r.resolution_chain import CAMERA_HW, SEG_HW
from tools.measure_ddm_pt1_continuous_paint_ceiling import (
    PT1Config,
    _depth_of_first_divergence,
    execute,
    prepare,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / ".omx/research/configs/ddm_pt1_continuous_paint_ceiling_20260723.json"
)
PALETTE = np.asarray(
    (
        (11, 3, 9),
        (51, 255, 204),
        (0, 153, 0),
        (107, 0, 114),
        (63, 72, 63),
    ),
    dtype=np.uint8,
)


def _five_stripe_labels(pairs: int = 1) -> np.ndarray:
    labels = np.empty((pairs, *SEG_HW), dtype=np.uint8)
    width = SEG_HW[1] // 5
    for class_id in range(5):
        start = class_id * width
        stop = SEG_HW[1] if class_id == 4 else (class_id + 1) * width
        labels[:, :, start:stop] = class_id
    return labels


def test_signed_distance_fit_recovers_source_partition() -> None:
    labels = _five_stripe_labels()
    fields = signed_distance_fields(labels)
    assert fields.shape == (*labels.shape, 5)
    assert np.array_equal(fields.argmax(axis=-1), labels)
    assert np.isfinite(fields).all()


def test_signed_distance_fit_composes_canonical_subpixel_tie_localizer() -> None:
    labels = _five_stripe_labels()
    margins = np.ones(labels.shape, dtype=np.float32)
    margins[:, :, 102] = 3.0
    fields = signed_distance_fields(labels, margins=margins)
    difference_left = fields[0, 0, 101, 0] - fields[0, 0, 101, 1]
    difference_right = fields[0, 0, 102, 0] - fields[0, 0, 102, 1]
    zero_crossing = difference_left / (difference_left - difference_right)
    assert zero_crossing == pytest.approx(0.25)
    assert np.array_equal(fields.argmax(axis=-1), labels)


def test_primary_hard_camera_arm_emits_only_full_prototypes() -> None:
    fields = np.full((1, *CAMERA_HW, 5), -2.0, dtype=np.float32)
    fields[:, :, : CAMERA_HW[1] // 2, 0] = 2.0
    fields[:, :, CAMERA_HW[1] // 2 :, 1] = 2.0
    camera = render_hard_camera_placement(fields, PALETTE)
    unique = {tuple(row) for row in np.unique(camera.reshape(-1, 3), axis=0)}
    assert unique == {tuple(PALETTE[0]), tuple(PALETTE[1])}
    assert camera.dtype == np.uint8


def test_secondary_analytic_arm_is_separate_and_can_emit_blends() -> None:
    fields = np.full((1, *CAMERA_HW, 5), -8.0, dtype=np.float32)
    fields[..., 0] = 0.0
    fields[..., 1] = 0.0
    analytic = render_analytic_coverage_blend(fields, PALETTE, softness=1.0)
    hard = render_hard_camera_placement(fields, PALETTE)
    expected = np.rint((PALETTE[0].astype(float) + PALETTE[1]) / 2).astype(
        np.uint8
    )
    assert np.array_equal(analytic[0, 0, 0], expected)
    assert not np.array_equal(analytic, hard)


def test_curve_provenance_is_an_exact_partition() -> None:
    labels = _five_stripe_labels()
    described = labels == 0
    split = split_curve_provenance(
        target_labels=labels,
        described_curve_mask=described,
        dilation=1,
    )
    assert not np.any(
        split["already_described_curve_sites"]
        & split["freshly_fitted_curve_sites"]
    )
    assert np.array_equal(
        split["already_described_curve_sites"]
        | split["freshly_fitted_curve_sites"],
        split["target_boundary_sites"],
    )
    assert split["already_described_curve_sites"].any()
    assert split["freshly_fitted_curve_sites"].any()


def test_mechanism_decomposition_is_disjoint_and_scoped() -> None:
    target = np.zeros((1, 2, 4), dtype=np.uint8)
    baseline = np.asarray([[[1, 1, 1, 1], [1, 1, 0, 0]]], dtype=np.uint8)
    primary = np.asarray([[[0, 1, 1, 1], [0, 1, 0, 0]]], dtype=np.uint8)
    statistics = np.asarray(
        [[[1, 0, 1, 1], [1, 1, 0, 0]]],
        dtype=np.uint8,
    )
    texture = np.asarray([[[1, 1, 0, 1], [1, 1, 0, 0]]], dtype=np.uint8)
    band = np.asarray(
        [[[True, True, True, False], [False, False, False, False]]],
        dtype=bool,
    )
    row = decompose_mechanisms(
        target=target,
        baseline=baseline,
        primary_hard=primary,
        statistics_control=baseline,
        statistics_matched=statistics,
        texture_probe=texture,
        boundary_band=band,
    )
    assert row.sub_cell_placement == 1
    assert row.bn_se_amplitude_statistics == 1
    assert row.texture_prior_or_region_erf == 1
    assert row.class_interaction == 1
    assert row.corrected_total_primary == 2
    assert row.corrected_total_statistics_only == 1
    assert row.corrected_total_secondary_only == 1


def test_global_statistics_payload_is_exact_and_geometry_preserving() -> None:
    source = np.asarray(
        [[[[10, 20, 30], [30, 40, 50]], [[50, 60, 70], [70, 80, 90]]]],
        dtype=np.uint8,
    )
    target = source // 2 + 20
    scale, offset = fit_global_channel_statistics(source, target)
    matched = apply_global_channel_statistics(source, scale, offset)
    payload = encode_global_channel_statistics(scale, offset)
    assert matched.shape == source.shape
    assert np.array_equal(matched, target)
    assert len(payload) == 30
    assert payload.startswith(b"PT1AS1")


def test_stratum_spectrum_match_is_deterministic_and_parse_backed() -> None:
    labels = _five_stripe_labels()
    components = stratum_spectrum_components(seed=0)
    coefficients = np.zeros(components.shape[:3], dtype=np.float32)
    coefficients[0, 0, 0] = 4.0
    first = render_stratum_spectrum_match(
        labels,
        PALETTE,
        coefficients,
        seed=0,
        components=components,
    )
    second = render_stratum_spectrum_match(
        labels,
        PALETTE,
        coefficients,
        seed=0,
        components=components,
    )
    assert np.array_equal(first, second)
    assert first.shape == (1, *SEG_HW, 3)
    assert np.any(first[labels == 0] != PALETTE[0])
    assert len(encode_stratum_spectrum_coefficients(coefficients)) == 186


def test_stratum_spectrum_fit_is_conditional_on_flat_paint() -> None:
    labels = _five_stripe_labels()
    components = stratum_spectrum_components(seed=0)
    target = PALETTE[labels]
    gram, rhs = stratum_spectrum_normal_equations(
        labels,
        target,
        PALETTE,
        seed=0,
        components=components,
    )
    coefficients = solve_stratum_spectrum_coefficients(gram, rhs)
    assert np.array_equal(coefficients, np.zeros_like(coefficients))


def test_scorer_native_profile_has_required_fields_and_trajectory() -> None:
    truth = {
        "layer_a": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "layer_b": np.zeros((2, 3), dtype=np.float32),
    }
    candidate = {
        "layer_a": np.ones((2, 2, 2, 2), dtype=np.float32),
        "layer_b": np.ones((2, 3), dtype=np.float32),
    }
    margins = np.zeros((2, *SEG_HW), dtype=np.float32)
    rows = scorer_native_divergence_rows(
        candidate=candidate,
        target=truth,
        margins=margins,
    )
    assert [row["layer"] for row in rows] == ["layer_a", "layer_b"]
    assert rows[0]["channel_group"] == "all_2ch"
    assert rows[0]["spatial_band"] == "2x2"
    assert rows[0]["fisher_weighted_delta"] is not None
    assert rows[0]["trajectory_delta_norm_relative"] == 0.0


def test_depth_of_first_divergence_distinguishes_stem_and_late() -> None:
    rows = [
        {
            "layer": "stem",
            "delta_norm_relative": 0.0,
            "trajectory_delta_norm_relative": 0.0,
            "cross_batch_trajectory_delta_norm_relative": 0.0,
        },
        {
            "layer": "head",
            "delta_norm_relative": 0.1,
            "trajectory_delta_norm_relative": 0.2,
            "cross_batch_trajectory_delta_norm_relative": 0.3,
        },
    ]
    depth = _depth_of_first_divergence(rows, final_argmax_errors=1)
    assert depth["static"]["first_divergent_layer"] == "head"
    assert depth["static"]["status"] == "LATE_DIVERGENCE_FEATURE_RELAY_CANDIDATE"
    rows[0]["trajectory_delta_norm_relative"] = 0.1
    stem = _depth_of_first_divergence(rows, final_argmax_errors=1)
    assert stem["within_batch_feature_trajectory"]["status"] == (
        "STEM_DIVERGENCE_NO_DOWNSTREAM_ONLY_CORRECTION"
    )


def test_stage_transition_closes_and_rejects_bad_owner_shape() -> None:
    target = np.zeros((1, 2, 2), dtype=np.uint8)
    before = np.asarray([[[0, 1], [1, 0]]], dtype=np.uint8)
    after = np.asarray([[[1, 0], [1, 0]]], dtype=np.uint8)
    row = stage_transition(before=before, after=after, target=target)
    assert row["errors_after"] == (
        row["errors_before"]
        + row["errors_introduced"]
        - row["errors_corrected"]
    )
    with pytest.raises(ContinuousPaintError, match="owner_mask"):
        stage_transition(
            before=before,
            after=after,
            target=target,
            owner_mask=np.ones((2, 2), dtype=bool),
        )


def test_fitted_geometry_is_charged_by_exact_sdwl1_parseback() -> None:
    labels = _five_stripe_labels()
    margins = np.ones(labels.shape, dtype=np.float32)
    poses = np.zeros((1, 6), dtype=np.float64)
    debt = measure_fitted_geometry_sdwl1(labels, margins, poses)
    assert debt.bytes > 0
    assert debt.exact_parseback is True
    assert len(debt.sha256) == 64
    assert debt.described_scalar_facts == 76


def test_checked_in_config_refuses_execution() -> None:
    config = PT1Config.model_validate_json(CONFIG_PATH.read_bytes())
    assert config.execution_allowed is False
    with pytest.raises(ContinuousPaintError, match="execution_allowed=false"):
        execute(config, Path("unused.json"), ["tool", "--config", str(CONFIG_PATH)])


def test_authorized_successor_still_requires_independent_survival_wall() -> None:
    config = PT1Config.model_validate_json(CONFIG_PATH.read_bytes())
    successor = config.model_copy(update={"execution_allowed": True})
    with pytest.raises(
        ContinuousPaintError,
        match="independent SHA-bound survival-wall receipt",
    ):
        execute(
            successor,
            Path("unused.json"),
            ["tool", "--config", str(CONFIG_PATH)],
        )


def test_prepare_emits_no_measurement_or_zero_byte_fitted_claim(
    tmp_path: Path,
) -> None:
    config = PT1Config.model_validate_json(CONFIG_PATH.read_bytes())
    output = tmp_path / "prepared.json"
    prepare(
        config,
        output,
        ["tool", "--config", str(CONFIG_PATH), "--output", str(output), "--prepare"],
    )
    row = json.loads(output.read_text())
    assert row["status"] == "PREPARED_NOT_EXECUTED"
    assert row["verdict"] == "NO_VERDICT_EXECUTION_FORBIDDEN"
    assert row["description_cost_policy"]["fresh_fitted_curve_delta_bytes"] is None
    assert len(row["mechanism_arms"]) == 3
    assert all(arm["outcome"] is None for arm in row["mechanism_arms"])
    assert all(arm["first_rung"] for arm in row["mechanism_arms"])
    assert row["diagnostic_variants"][0]["outcome"] is None
    assert row["falsifiers"]["mechanism_primary"]["status"] == (
        "PENDING_INDEPENDENT_WALL_RECEIPT"
    )
