# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tools.probe_ddm_kinetic_laguerre_at_tolerance import (
    CODEC_IDS,
    DDMKineticLaguerreAtToleranceProbeV1,
    MetricSpec,
    ProbeError,
    ProgramState,
    _fit_independent_program,
    _fit_kinetic_program,
    _fit_pack_kinetic_waterfill,
    _kernel_contract,
    _program_parseback_receipt,
    _quantize_checked,
    _regular_triangulation_edges,
    _stable_pose_advection,
    coder_race,
    decode_envelope,
    decode_program,
    derive_class_weights,
    derive_site_schedule,
    encode_raw_program,
    extract_site_states,
    measure_label_program,
    pack_program,
    parse_raw_program,
    power_assign_ckdtree,
    power_assign_numpy_fp32,
    unpack_program,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / ".omx/research/configs/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723.json"


def _synthetic_labels(n: int = 6, shape: tuple[int, int] = (18, 22)) -> np.ndarray:
    rows, cols = np.indices(shape)
    out = np.empty((n, *shape), np.uint8)
    for frame in range(n):
        out[frame] = (cols + frame // 2 >= shape[1] // 2).astype(np.uint8) + 2 * (rows >= 2 * shape[0] // 3).astype(
            np.uint8
        )
        out[frame, :2, :3] = 4
        out[frame, shape[0] // 2, 2:-2:4] = 3
    return out


def test_live_config_is_typed_and_authority_bound() -> None:
    config = DDMKineticLaguerreAtToleranceProbeV1.model_validate_json(CONFIG.read_bytes())
    assert config.execution_allowed is True
    assert config.score_claim is False
    assert config.representation.site_counts == (64, 128, 256, 512)
    assert len(config.typed_config_hash()) == 64

    payload = json.loads(CONFIG.read_text())
    payload["execution_allowed"] = False
    with pytest.raises(ValidationError):
        DDMKineticLaguerreAtToleranceProbeV1.model_validate(payload)


@pytest.mark.parametrize("codec", tuple(CODEC_IDS))
def test_all_real_coders_roundtrip_canonical_program(codec: str) -> None:
    program = ProgramState(
        {"schema": "test.v1", "mode": "bounded"},
        {
            "a": np.arange(-30, 30, dtype="<i2").reshape(10, 6),
            "b": np.array([0, 1, 1, 2, 3, 5, 8], dtype="<u2"),
        },
    )
    raw = encode_raw_program(program)
    packed = pack_program(raw, codec)
    assert unpack_program(packed) == raw
    parsed = parse_raw_program(unpack_program(packed))
    assert np.array_equal(parsed.arrays["a"], program.arrays["a"])
    assert np.array_equal(parsed.arrays["b"], program.arrays["b"])


def test_coder_race_reports_exact_double_parseback() -> None:
    raw = encode_raw_program(
        ProgramState(
            {"schema": "test.v1"},
            {"delta": np.zeros((64, 3), dtype="<i2")},
        )
    )
    payload, receipt = coder_race(raw, tuple(CODEC_IDS))
    assert receipt["selected_bytes"] == len(payload)
    assert len(receipt["rows"]) == 3
    assert all(row["double_parseback_identity"] for row in receipt["rows"])


@pytest.mark.parametrize(
    "metric",
    (
        MetricSpec("isotropic_power_control"),
        MetricSpec("shared_chart_anisotropic_spd", 1.2, 1 / 1.2),
        MetricSpec("projective_depth_stratified", horizon_row=3.0),
    ),
)
def test_ckdtree_kernel_is_bit_identical_to_numpy_fp32(metric: MetricSpec) -> None:
    shape = (9, 11)
    sites = np.array(
        [[1, 1], [1, 9], [7, 1], [7, 9], [4, 5], [2, 5], [6, 5]],
        np.float32,
    )
    classes = np.array([0, 1, 2, 3, 4, 0, 1], np.uint8)
    weights = np.array([0.5, 1.25, 0.0, 2.0, 0.75], np.float32)
    rows, cols = np.indices(shape, dtype=np.float32)
    pixels = np.column_stack((rows.ravel(), cols.ravel()))
    expected = power_assign_numpy_fp32(sites, weights, classes, pixels, metric, shape)
    actual = power_assign_ckdtree(sites, weights, classes, pixels, metric, shape)
    assert np.array_equal(actual, expected)


def test_kernel_contract_covers_all_metrics() -> None:
    receipt = _kernel_contract()
    assert receipt["all_bit_identical"] is True
    assert len(receipt["rows"]) == 3


def test_regular_triangulation_degeneracy_is_deterministic() -> None:
    sites = np.array(
        [[2, 2], [2, 8], [8, 2], [8, 8], [5, 5], [5, 5]],
        np.float32,
    )
    classes = np.array([0, 1, 2, 3, 4, 4], np.uint8)
    weights = np.zeros(5, np.float32)
    metric = MetricSpec("isotropic_power_control")
    first = _regular_triangulation_edges(sites, weights, classes, metric, (11, 11))
    second = _regular_triangulation_edges(sites, weights, classes, metric, (11, 11))
    assert first == second


def test_rank_deficient_pose_regression_is_finite_and_quantizable() -> None:
    xi = np.ones((8, 6), np.float64)
    xi[:, 0] += np.linspace(0.0, 1e-12, len(xi))
    target = np.column_stack((np.linspace(10, 20, len(xi)), np.linspace(30, 15, len(xi))))
    advection = _stable_pose_advection(xi, target)
    assert np.isfinite(advection).all()
    quantized = _quantize_checked(advection, 8.0, "<i2", "test_advection")
    assert quantized.dtype == np.dtype("<i2")


def test_full_length_pose_regression_is_warning_clean() -> None:
    rng = np.random.default_rng(1234)
    xi = rng.normal(size=(600, 6))
    target = rng.uniform(0, 512, size=(600, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        advection = _stable_pose_advection(xi, target)
    assert np.isfinite(advection).all()


def test_quantization_fails_closed_on_nonfinite_or_overflow() -> None:
    with pytest.raises(ProbeError, match="non-finite"):
        _quantize_checked(np.array([np.inf]), 1.0, "<i2", "bad")
    with pytest.raises(ProbeError, match="exceeds"):
        _quantize_checked(np.array([1e9]), 1.0, "<i2", "bad")


def test_site_extraction_is_nested_deterministic_and_class_complete() -> None:
    labels = _synthetic_labels()
    first_schedule, first_receipt = derive_site_schedule(labels, 64)
    second_schedule, second_receipt = derive_site_schedule(labels, 64)
    first = extract_site_states(labels, first_schedule)
    second = extract_site_states(labels, second_schedule)
    assert np.array_equal(first_schedule, second_schedule)
    assert np.array_equal(first, second)
    assert set(first_schedule[:5]) == set(range(5))
    assert first_receipt == second_receipt
    assert first.shape == (6, 64, 2)


def test_independent_program_double_decode_and_measurement() -> None:
    labels = _synthetic_labels()
    schedule, _ = derive_site_schedule(labels, 64)
    sites = extract_site_states(labels, schedule)
    metric = MetricSpec("isotropic_power_control")
    weights = derive_class_weights(sites, schedule, metric, labels.shape[1:])
    palette = np.arange(15, dtype=np.uint8).reshape(5, 3)
    program = _fit_independent_program(
        sites,
        weights,
        schedule,
        metric,
        palette,
        degree=1,
        pair_ids=range(len(labels)),
    )
    raw = encode_raw_program(program)
    payload, _ = coder_race(raw, tuple(CODEC_IDS))
    decoded, receipt = _program_parseback_receipt(payload)
    assert receipt["double_decode_identity"] is True
    assert np.array_equal(decoded.site_classes, schedule)
    measurement = measure_label_program(decoded, labels, stop_after_errors=None)
    assert measurement["complete"] is True
    assert measurement["sites"] == labels.size
    assert measurement["errors"] is not None


def test_kinetic_program_is_xi_driven_and_event_charged() -> None:
    labels = _synthetic_labels(n=8)
    schedule, _ = derive_site_schedule(labels, 64)
    sites = extract_site_states(labels, schedule)
    metric = MetricSpec("shared_chart_anisotropic_spd", 1.1, 1 / 1.1)
    weights = derive_class_weights(sites, schedule, metric, labels.shape[1:])
    xi = np.column_stack(
        [
            np.linspace(-1, 1, len(labels)),
            np.linspace(0, 0.5, len(labels)),
            np.zeros((len(labels), 4)),
        ]
    )
    palette = np.arange(15, dtype=np.uint8).reshape(5, 3)
    program = _fit_kinetic_program(
        sites,
        weights,
        xi,
        schedule,
        metric,
        palette,
        degree=2,
        segment_count=2,
        pair_ids=range(len(labels)),
        image_shape=labels.shape[1:],
        include_events=True,
    )
    assert np.count_nonzero(program.arrays["xi_advection_q"]) > 0
    raw = encode_raw_program(program)
    payload, _ = coder_race(raw, tuple(CODEC_IDS))
    decoded = decode_envelope(payload)
    assert decoded.metadata["xi_driven"] is True
    assert decoded.metadata["segment_count"] == 2
    assert decoded.sites.shape == sites.shape
    assert decoded.class_weights.shape == weights.shape
    assert decoded.event_rows.ndim == 2
    assert decoded.event_rows.shape[1] == 4


def test_temporal_waterfill_charges_events_and_stays_inside_home() -> None:
    labels = _synthetic_labels(n=8)
    schedule, _ = derive_site_schedule(labels, 64)
    sites = extract_site_states(labels, schedule)
    metric = MetricSpec("isotropic_power_control")
    weights = derive_class_weights(sites, schedule, metric, labels.shape[1:])
    xi = np.column_stack([np.linspace(-1, 1, len(labels)), np.zeros((len(labels), 5))])
    payload, receipt = _fit_pack_kinetic_waterfill(
        sites=sites,
        weights=weights,
        xi=xi,
        site_classes=schedule,
        metric=metric,
        palette=np.zeros((5, 3), np.uint8),
        degree=1,
        pair_ids=range(len(labels)),
        image_shape=labels.shape[1:],
        home_bytes=10_000,
        coders=tuple(CODEC_IDS),
    )
    decoded = decode_envelope(payload)
    assert len(payload) <= 10_000
    assert receipt["within_predictor_home"] is True
    assert receipt["minimum_frames_per_segment"] == 8
    assert receipt["maximum_segment_count"] == 1
    assert decoded.metadata["regular_triangulation_event_count"] == len(decoded.event_rows)


def test_measurement_early_stop_is_an_exact_lower_bound() -> None:
    labels = _synthetic_labels()
    schedule, _ = derive_site_schedule(labels, 64)
    sites = extract_site_states(labels, schedule)
    metric = MetricSpec("isotropic_power_control")
    weights = np.zeros((len(labels), 5), np.float32)
    program = _fit_independent_program(
        sites,
        weights,
        schedule,
        metric,
        np.zeros((5, 3), np.uint8),
        degree=1,
        pair_ids=range(len(labels)),
    )
    decoded = decode_program(program)
    measurement = measure_label_program(decoded, labels, stop_after_errors=0)
    assert measurement["complete"] is False
    assert measurement["errors"] is None
    assert measurement["errors_lower_bound"] > 0
    assert measurement["evaluated_frames"] < len(labels)
