# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math import warp_real_luma_frame0 as g1_warp
from tac.boundary_math.analytic_lane_render_band import LaneBandRenderConfig, serialize_lane_band_rd
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.optimization import predictor_upgrade_xi_chart as predictor_module
from tac.optimization.predict_project_schema import (
    PredictProjectSchemaError,
    attach_generic_predictor_policy,
    build_minimal_constraint_seed,
    parse_constraint_seed,
    serialize_constraint_seed,
)
from tac.optimization.predictor_upgrade_xi_chart import (
    LaneCoefficientAddress,
    LaneCoefficientDelta,
    PredictorUpgradeError,
    ProjectedChartChoice,
    StaticCharts,
    advect_categorical_field,
    apply_lane_coefficient_deltas,
    chart_section_receipt,
    decode_lane_chart_with_symbols,
    decode_lane_coefficient_deltas,
    encode_lane_coefficient_deltas,
    lane_coefficient_packet_size,
    load_lane_chart,
    miss_cause,
    parse_static_charts,
    predict_cell_field,
    predictor_policy,
    projected_greedy_with_swap_search,
    relative_adjacent_xi,
    serialize_static_charts,
    summarize_joint_chart_prefix_rows,
    train_static_charts,
    validate_resume_config,
)


def labels() -> np.ndarray:
    rows = np.zeros((64, 8, 10), dtype=np.uint8)
    rows[:, :, 4:] = 2
    rows[:, 3:5, 4:6] = 1
    rows[:, 2, 2] = 3
    rows[:, 6:, 2:8] = 4
    return rows


def seed() -> dict:
    return build_minimal_constraint_seed(
        bytes([0, 1, 2, 3, 4, 0]), scorer_height=2, scorer_width=3, camera_height=4, camera_width=6
    )


def policy() -> dict:
    payload = serialize_static_charts(train_static_charts(labels()))
    return predictor_policy(
        chart_section_receipt(payload),
        {
            "lane_chart_raw_bytes": 159386,
            "lane_chart_brotli_bytes": 41303,
            "lane_chart_sha256": "d2b2a62eeb6ebe45cbf908dafa7e081eabddaca0f424faac970b41eea650d810",
            "lane_chart_zlib9_bytes_diagnostic_only": 47546,
            "lane_chart_zlib9_sha256_diagnostic_only": "ef16824eea59415e71435b94c450c2d554e0db08c0981fae6b392ab08170d287",
            "lane_chart_status": "executed_in_task578_measurement_external_counted_custody",
            "executed": True,
            "execution_scope": "task578_measurement_only",
            "receiver_closed": False,
        },
    )


def test_legacy_schema_roundtrip_remains_byte_identical() -> None:
    encoded = serialize_constraint_seed(seed())
    assert serialize_constraint_seed(parse_constraint_seed(encoded)) == encoded
    assert "generic_predictor_policy" not in parse_constraint_seed(encoded)["receiver"]


def test_new_policy_roundtrip_is_explicit_and_unknowns_fail_closed() -> None:
    value = attach_generic_predictor_policy(seed(), policy())
    encoded = serialize_constraint_seed(value)
    assert serialize_constraint_seed(parse_constraint_seed(encoded)) == encoded
    assert parse_constraint_seed(encoded)["receiver"]["generic_predictor_policy"]["policy_id"].endswith(".v2")
    bad = copy.deepcopy(policy())
    bad["unknown"] = True
    with pytest.raises(PredictProjectSchemaError, match="fields mismatch"):
        attach_generic_predictor_policy(seed(), bad)
    bad = copy.deepcopy(policy())
    bad["policy_id"] = "unknown"
    with pytest.raises(PredictProjectSchemaError, match="not admitted"):
        attach_generic_predictor_policy(seed(), bad)
    bad = copy.deepcopy(policy())
    bad["motion_custody"]["g1_receipt_sha256"] = "1" * 64
    with pytest.raises(PredictProjectSchemaError, match="G1 receipt custody mismatch"):
        attach_generic_predictor_policy(seed(), bad)


def test_static_chart_payload_is_canonical_and_contains_no_full_target_sequence() -> None:
    source = labels()
    payload = serialize_static_charts(train_static_charts(source))
    assert serialize_static_charts(parse_static_charts(payload)) == payload
    assert len(payload) < source.nbytes
    assert source.tobytes() not in payload


def test_chart_symbol_packet_is_counted_canonical_and_receiver_applied() -> None:
    pairs = [
        [
            LaneLine(
                centerline_coeffs=np.asarray([0.0, 0.25 + pair], dtype=np.float64),
                halfwidth_coeffs=np.asarray([0.0, 2.0], dtype=np.float64),
                forward_range=(0.0, 50.0),
            )
        ]
        for pair in range(2)
    ]
    config = LaneBandRenderConfig(dash_gate=False)
    lane_payload = serialize_lane_band_rd(pairs, config)
    symbol = LaneCoefficientDelta(
        pair_index=1,
        line_index=0,
        coefficient_index=1,
        coefficient_delta=0.03123456789,
    )
    payload = encode_lane_coefficient_deltas((symbol,))
    assert len(payload) == 20
    assert decode_lane_coefficient_deltas(payload) == (symbol,)
    corrected, decoded_config, receipt = decode_lane_chart_with_symbols(lane_payload, payload)
    baseline, _ = predictor_module.deserialize_lane_band_any(lane_payload)
    assert decoded_config.dash_gate is False
    assert corrected[1][0].centerline_coeffs[1] == pytest.approx(
        baseline[1][0].centerline_coeffs[1] + float(np.float32(0.03123456789)), rel=0.0, abs=1e-12
    )
    assert corrected[0][0].centerline_coeffs.tolist() == baseline[0][0].centerline_coeffs.tolist()
    assert receipt["chart_symbol_bytes"] == len(payload)
    assert receipt["double_decode_byte_identical"] is True
    assert receipt["rule_118"]["target_cells_or_scorer_state_in_packet"] is False
    assert encode_lane_coefficient_deltas(()) == b""
    assert decode_lane_coefficient_deltas(b"") == ()


def test_chart_symbol_packet_and_addresses_fail_closed() -> None:
    first = LaneCoefficientDelta(0, 0, 0, 0.25)
    second = LaneCoefficientDelta(1, 0, 0, -0.25)
    payload = encode_lane_coefficient_deltas((first, second))
    for malformed in (payload[:-1], payload + b"x", payload[:12] + bytes([payload[12] ^ 1]) + payload[13:]):
        with pytest.raises(PredictorUpgradeError):
            decode_lane_coefficient_deltas(malformed)
    with pytest.raises(PredictorUpgradeError, match="sorted"):
        encode_lane_coefficient_deltas((second, first))
    with pytest.raises(PredictorUpgradeError, match="address"):
        apply_lane_coefficient_deltas(
            [[LaneLine(np.asarray([0.0]), np.asarray([1.0]))]],
            (LaneCoefficientDelta(0, 0, 1, 0.25),),
        )


def test_chart_symbol_multirow_packet_size_parseback_and_application() -> None:
    pairs = [
        [
            LaneLine(
                centerline_coeffs=np.asarray([pair, line, 0.5, -0.25], dtype=np.float64),
                halfwidth_coeffs=np.asarray([0.0, 2.0], dtype=np.float64),
            )
            for line in range(2)
        ]
        for pair in range(2)
    ]
    symbols = (
        LaneCoefficientDelta(0, 0, 0, 0.125),
        LaneCoefficientDelta(0, 1, 2, -0.5),
        LaneCoefficientDelta(1, 0, 3, 1.25),
    )
    payload = encode_lane_coefficient_deltas(symbols)
    assert len(payload) == lane_coefficient_packet_size(3) == 12 + 8 * 3
    assert encode_lane_coefficient_deltas(decode_lane_coefficient_deltas(payload)) == payload
    corrected = apply_lane_coefficient_deltas(pairs, decode_lane_coefficient_deltas(payload))
    assert corrected[0][0].centerline_coeffs[0] == pytest.approx(0.125)
    assert corrected[0][1].centerline_coeffs[2] == pytest.approx(0.0)
    assert corrected[1][0].centerline_coeffs[3] == pytest.approx(1.0)
    assert corrected[1][1].centerline_coeffs.tolist() == pairs[1][1].centerline_coeffs.tolist()

    header = struct.Struct(">5sBHI")
    body = payload[header.size :]
    row_size = 8
    reordered = body[row_size : 2 * row_size] + body[:row_size] + body[2 * row_size :]
    magic, version, count, _ = header.unpack_from(payload)
    noncanonical = header.pack(magic, version, count, zlib.crc32(reordered) & 0xFFFFFFFF) + reordered
    with pytest.raises(PredictorUpgradeError, match="sorted and address-unique"):
        decode_lane_coefficient_deltas(noncanonical)

    duplicate = body[:row_size] + body[:row_size] + body[2 * row_size :]
    nonunique = header.pack(magic, version, count, zlib.crc32(duplicate) & 0xFFFFFFFF) + duplicate
    with pytest.raises(PredictorUpgradeError, match="sorted and address-unique"):
        decode_lane_coefficient_deltas(nonunique)


def test_projected_greedy_search_deterministically_adds_polishes_and_swaps() -> None:
    a = LaneCoefficientAddress(0, 0)
    b = LaneCoefficientAddress(0, 1)
    c = LaneCoefficientAddress(1, 0)

    def objective(choices: tuple[ProjectedChartChoice, ...]) -> tuple[int, int, float]:
        state = {(choice.address, choice.amplitude) for choice in choices}
        lookup = {
            frozenset(): (4, 4, 4.0),
            frozenset({(a, 1.0)}): (3, 4, 4.0),
            frozenset({(a, 1.0), (b, 1.0)}): (2, 3, 3.0),
            frozenset({(a, 2.0), (b, 1.0)}): (2, 2, 2.0),
            frozenset({(a, 2.0), (c, 1.0)}): (1, 1, 1.0),
            frozenset({(a, 2.0), (b, 1.0), (c, 1.0)}): (0, 0, 0.0),
        }
        return lookup.get(frozenset(state), (4, 4, 9.0))

    one = projected_greedy_with_swap_search(
        addresses=(a, b, c),
        amplitudes=(1.0, 2.0),
        objective=objective,
        polish_pass_limit=3,
        swap_pass_limit=1,
    )
    two = projected_greedy_with_swap_search(
        addresses=(a, b, c),
        amplitudes=(2.0, 1.0),
        objective=objective,
        polish_pass_limit=3,
        swap_pass_limit=1,
    )
    assert one == two
    assert one.status == "MODEL_ADMITTED_CANDIDATE"
    assert [prefix.cardinality for prefix in one.prefixes] == [1, 2, 3]
    assert one.prefixes[0].choices == (ProjectedChartChoice(a, 1.0),)
    assert one.prefixes[1].choices == (ProjectedChartChoice(a, 2.0), ProjectedChartChoice(c, 1.0))
    assert one.final_constraint_key == (0, 0, 0.0)
    assert "not a global-optimum proof" in one.search_completeness


def test_projected_greedy_search_refuses_duplicates_and_reports_exhaustion() -> None:
    address = LaneCoefficientAddress(0, 0)
    with pytest.raises(PredictorUpgradeError, match="sorted, and unique"):
        projected_greedy_with_swap_search(
            addresses=(address, address),
            amplitudes=(1.0,),
            objective=lambda choices: (1, 1, float(len(choices))),
        )
    result = projected_greedy_with_swap_search(
        addresses=(address,),
        amplitudes=(1.0,),
        objective=lambda choices: (1, 1, 1.0 if not choices else 0.5),
    )
    assert result.status == "EXHAUSTED_ALL_ADDRESSES"
    assert result.prefixes[0].constraint_key == (1, 1, 0.5)


def test_joint_chart_prefix_summary_uses_only_complete_hard_predicates() -> None:
    names = (
        "semantic_exact",
        "pose_tube",
        "factor2_uint8_exact",
        "double_decode",
        "receiver_RGB",
        "counted_bytes",
        "rate_above_lambda",
    )
    rows = []
    for k in range(1, 3):
        predicates = dict.fromkeys(names, True)
        if k != 2:
            predicates["semantic_exact"] = False
        rows.append(
            {
                "k": k,
                "packet_bytes": lane_coefficient_packet_size(k),
                "semantic_mismatch_count": 0 if k == 2 else 1,
                "pose_tube_debt": 0.0,
                "d_seg": 0.1 / k,
                "d_pose": 0.01 / k,
                "incremental_delta_d_seg": -0.01,
                "incremental_delta_d_pose": -0.001,
                "cumulative_delta_d_seg": -0.01 * k,
                "cumulative_delta_d_pose": -0.001 * k,
                "delta_bytes": 20 if k == 1 else 8,
                "incremental_score_units_recovered": 0.02,
                "cumulative_score_units_recovered": 0.02 * k,
                "incremental_marginal_score_units_per_byte": 0.001,
                "cumulative_average_score_units_per_byte": 0.001,
                "admission_predicates": predicates,
                "admitted": k == 2,
                "rate_stop": False,
                "terminal_stop_reason": "FIRST_ADMITTED_MIN_K_STOP" if k == 2 else None,
            }
        )
    summary = summarize_joint_chart_prefix_rows(rows)
    assert summary["measured_prefix_count"] == 2
    assert summary["admitted_prefix_count"] == 1
    assert summary["minimum_admitted_k"] == 2
    assert summary["minimum_admitted_packet_bytes"] == 28
    assert summary["measured_k_curve"][-1]["terminal_stop_reason"] == "FIRST_ADMITTED_MIN_K_STOP"
    assert summary["rate_break_even_stop"] is None
    assert summary["model_prediction_is_hard_predicate"] is False


def test_joint_chart_prefix_summary_preserves_terminal_rate_stop() -> None:
    predicates = {
        "semantic_exact": False,
        "pose_tube": True,
        "factor2_uint8_exact": True,
        "double_decode": True,
        "receiver_RGB": True,
        "counted_bytes": True,
        "rate_above_lambda": False,
    }
    summary = summarize_joint_chart_prefix_rows(
        [
            {
                "k": 1,
                "packet_bytes": 20,
                "delta_bytes": 20,
                "semantic_mismatch_count": 1,
                "pose_tube_debt": 0.0,
                "d_seg": 0.1,
                "d_pose": 0.01,
                "incremental_delta_d_seg": 0.0,
                "incremental_delta_d_pose": 0.0,
                "cumulative_delta_d_seg": 0.0,
                "cumulative_delta_d_pose": 0.0,
                "incremental_score_units_recovered": 0.0,
                "cumulative_score_units_recovered": 0.0,
                "incremental_marginal_score_units_per_byte": 0.0,
                "cumulative_average_score_units_per_byte": 0.0,
                "admission_predicates": predicates,
                "admitted": False,
                "rate_stop": True,
                "terminal_stop_reason": "RATE_BREAK_EVEN_STOP",
            }
        ]
    )
    assert summary["status"] == "RATE_BREAK_EVEN_STOP"
    assert summary["rate_break_even_stop"]["k"] == 1
    assert summary["rate_break_even_stop"]["delta_bytes"] == 20


def test_predictor_is_deterministic_causal_and_does_not_mutate_prior() -> None:
    charts = train_static_charts(labels())
    prior = labels()[0].copy()
    original = prior.copy()
    xi = np.zeros(6, dtype=np.float64)
    lane = np.zeros(prior.shape, dtype=np.bool_)
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=prior.shape, pitch=0.0)
    one = predict_cell_field(
        pair_index=1,
        prior_decoded_field=prior,
        charts=charts,
        relative_xi=xi,
        worldsheet_geom=geom,
        lane_mask=lane,
    )
    two = predict_cell_field(
        pair_index=1,
        prior_decoded_field=prior,
        charts=charts,
        relative_xi=xi,
        worldsheet_geom=geom,
        lane_mask=lane,
    )
    assert one.dtype == np.uint8
    assert one.tobytes() == two.tobytes()
    assert np.array_equal(prior, original)
    with pytest.raises(PredictorUpgradeError, match="prior decoded"):
        predict_cell_field(
            pair_index=1,
            prior_decoded_field=None,
            charts=charts,
            relative_xi=xi,
            worldsheet_geom=geom,
            lane_mask=lane,
        )


def test_nontrivial_advection_calls_canonical_worldsheet_and_is_not_flat_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    h, w = 32, 48
    yy, xx = np.indices((h, w))
    prior = ((yy // 4 + xx // 5) % 5).astype(np.uint8)
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=(h, w), pitch=-0.05)
    pose = np.asarray([30.0, 2.0, 18.0, 0.0, 0.0, 0.0], dtype=np.float64)
    xi = g1_warp.xi_from_pose_calibration(pose, s_t=-0.00143, s_r=0.0, pitch=-0.05)
    original = g1_warp.warp_frame0_native_numpy
    calls = 0

    def wrapped(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(g1_warp, "warp_frame0_native_numpy", wrapped)
    projected = advect_categorical_field(prior, xi, geom)
    dx, dy = int(np.rint(xi[0] * 256.0)), int(np.rint(xi[2] * 256.0))
    flat = np.zeros_like(prior)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    flat[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = prior[sy0:sy1, sx0:sx1]
    assert calls == 1
    assert not np.array_equal(projected, flat)


def test_g1_proxy_twist_uses_pose_t_directly_without_redifferencing() -> None:
    poses = np.asarray(
        [
            [200.0, 30.0, -40.0, 0.2, -0.1, 0.3],
            [20.0, 2.0, 5.0, 0.01, 0.02, 0.03],
        ]
    )
    twists, custody = relative_adjacent_xi(poses, s_t=-0.00143, s_r=0.0, pitch_rad=-0.05)
    expected = g1_warp.xi_from_pose_calibration(poses[1], s_t=-0.00143, s_r=0.0, pitch=-0.05)
    assert np.array_equal(twists[0], np.zeros(6))
    assert np.allclose(twists[1], expected)
    assert custody["absolute_trajectory_fabricated"] is False
    assert custody["already_relative_twist_redifferenced"] is False


def test_reused_hungarian_track_update_is_applied_without_target_raster() -> None:
    charts = train_static_charts(labels())
    prior = np.zeros((8, 10), dtype=np.uint8)
    track = {
        "cell_id": 3,
        "knots": [
            {"time": 0, "y_q": 3 * 256, "x_q": 2 * 256, "height_q": 2 * 256, "width_q": 2 * 256},
            {"time": 2, "y_q": 3 * 256, "x_q": 2 * 256, "height_q": 2 * 256, "width_q": 2 * 256},
        ],
    }
    predicted = predict_cell_field(
        pair_index=1,
        prior_decoded_field=prior,
        charts=charts,
        relative_xi=np.zeros(6),
        worldsheet_geom=g1_warp.GroundHomographyGeom.eon(native_hw=prior.shape, pitch=0.0),
        movable_tracks=[track],
        lane_mask=np.zeros(prior.shape, dtype=np.bool_),
    )
    assert np.count_nonzero(predicted == 3) == 4


def test_lane_executes_before_mycar_and_movable_with_measured_edge_admission() -> None:
    ru = np.zeros((4, 4), dtype=np.uint8)
    hood = np.zeros((4, 4), dtype=np.bool_)
    hood[0, 0] = True
    charts = StaticCharts(ru, hood, ((0, 1), (1, 3), (1, 4)))
    lane = np.ones((4, 4), dtype=np.bool_)
    track = {
        "cell_id": 3,
        "knots": [
            {"time": 0, "y_q": 2 * 256, "x_q": 2 * 256, "height_q": 2 * 256, "width_q": 2 * 256},
            {"time": 2, "y_q": 2 * 256, "x_q": 2 * 256, "height_q": 2 * 256, "width_q": 2 * 256},
        ],
    }
    predicted, trace = predict_cell_field(
        pair_index=1,
        prior_decoded_field=ru.copy(),
        charts=charts,
        relative_xi=np.zeros(6),
        worldsheet_geom=g1_warp.GroundHomographyGeom.eon(native_hw=ru.shape, pitch=0.0),
        lane_mask=lane,
        movable_tracks=[track],
        return_trace=True,
    )
    assert np.all(trace["lane"] == 1)
    assert trace["mycar"][0, 0] == 4
    assert np.all(predicted[1:3, 1:3] == 3)


def test_miss_causes_follow_executed_sequential_policy() -> None:
    target = np.asarray([[1, 1, 0, 2, 4, 3, 0]], dtype=np.uint8)
    predicted = np.asarray([[0, 4, 2, 0, 0, 0, 2]], dtype=np.uint8)
    ru = np.asarray([[0, 0, 0, 2, 0, 0, 1]], dtype=np.uint8)
    charts = StaticCharts(ru, np.zeros_like(ru, dtype=np.bool_), ())
    lane = np.asarray([[False, True, False, False, False, False, False]], dtype=np.bool_)
    trace = {
        "road_undrivable": np.asarray([[0, 0, 2, 2, 0, 0, 0]], dtype=np.uint8),
        "lane": np.asarray([[0, 1, 2, 2, 0, 0, 0]], dtype=np.uint8),
        "mycar": np.asarray([[0, 4, 2, 2, 0, 0, 0]], dtype=np.uint8),
    }
    strata = np.asarray([[0, 0, 0, 0, 0, 2, 3]], dtype=np.uint8)
    assert miss_cause(target, predicted, strata, charts, lane, trace).tolist() == [[1, 2, 1, 2, 1, 3, 4]]


def test_pair0_is_chart_only_and_unknown_class_ids_are_rejected() -> None:
    charts = train_static_charts(labels())
    out = predict_cell_field(
        pair_index=0,
        prior_decoded_field=None,
        charts=charts,
        relative_xi=np.zeros(6),
        worldsheet_geom=g1_warp.GroundHomographyGeom.eon(native_hw=charts.hood.shape),
        lane_mask=np.zeros(charts.hood.shape, dtype=np.bool_),
    )
    assert set(np.unique(out)).issubset({0, 2, 4})
    bad = labels()
    bad[0, 0, 0] = 7
    with pytest.raises(PredictorUpgradeError, match="canonical class IDs"):
        train_static_charts(bad)


def test_policy_counts_actual_payload_and_external_lane_once() -> None:
    value = policy()
    assert value["counted_sections"][0]["raw_bytes"] > 0
    assert value["external_counted_custody"] == {
        "lane_chart_raw_bytes": 159386,
        "lane_chart_brotli_bytes": 41303,
        "lane_chart_sha256": "d2b2a62eeb6ebe45cbf908dafa7e081eabddaca0f424faac970b41eea650d810",
        "lane_chart_zlib9_bytes_diagnostic_only": 47546,
        "lane_chart_zlib9_sha256_diagnostic_only": "ef16824eea59415e71435b94c450c2d554e0db08c0981fae6b392ab08170d287",
        "lane_chart_status": "executed_in_task578_measurement_external_counted_custody",
        "executed": True,
        "execution_scope": "task578_measurement_only",
        "receiver_closed": False,
    }


def test_exact_lane_packet_decodes_and_validates_custody() -> None:
    path = Path(
        "/Volumes/VertigoDataTier/pact/evidence/boundary_inverse_20260721/run_20260721T052100Z_threshold0p5/coherent_slot_none_dash.lbnd2"
    )
    if not path.is_file():
        pytest.skip("Task #595 SSD packet is not mounted; measurement CLI still fails closed")
    pairs, config, custody = load_lane_chart(path)
    assert len(pairs) == 600 and config.lane_cls == 1
    assert custody["executed"] is True and custody["lane_chart_brotli_bytes"] == 41303
    assert custody["lane_chart_zlib9_bytes_diagnostic_only"] == 47546
    assert custody["receiver_closed"] is False


def test_missing_lane_packet_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PredictorUpgradeError, match="raw/brotli/SHA custody mismatch"):
        load_lane_chart(tmp_path / "absent.lbnd2")


def test_resume_refuses_source_or_config_drift() -> None:
    validate_resume_config({"config_sha256": "a" * 64}, "a" * 64)
    with pytest.raises(PredictorUpgradeError, match="source/config drift"):
        validate_resume_config({"config_sha256": "a" * 64}, "b" * 64)


def test_source_and_motion_custody_change_run_config_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    seeds = {name: tmp_path / f"{name}.ppcs" for name in ("loose", "knee", "tight")}
    lane = tmp_path / "lane.lbnd2"
    lane_custody = {"lane_chart_sha256": "c" * 64}
    motion = {"pitch_custody": {"resolved_value": -0.05}}
    monkeypatch.setattr(predictor_module, "sha256_file", lambda _path: "a" * 64)
    baseline = predictor_module._source_config_hash(
        cache=cache,
        n_pairs=64,
        chunk_size=16,
        predecessor_seeds=seeds,
        lane_chart=lane,
        lane_custody=lane_custody,
        motion_custody=motion,
    )
    changed_motion = predictor_module._source_config_hash(
        cache=cache,
        n_pairs=64,
        chunk_size=16,
        predecessor_seeds=seeds,
        lane_chart=lane,
        lane_custody=lane_custody,
        motion_custody={"pitch_custody": {"resolved_value": -0.04}},
    )
    module_path = Path(predictor_module.__file__)
    monkeypatch.setattr(
        predictor_module,
        "sha256_file",
        lambda path: "b" * 64 if Path(path) == module_path else "a" * 64,
    )
    changed_source = predictor_module._source_config_hash(
        cache=cache,
        n_pairs=64,
        chunk_size=16,
        predecessor_seeds=seeds,
        lane_chart=lane,
        lane_custody=lane_custody,
        motion_custody=motion,
    )
    assert baseline != changed_motion
    assert baseline != changed_source
