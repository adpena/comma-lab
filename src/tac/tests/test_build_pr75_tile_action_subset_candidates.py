from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr75_tile_action_subset_candidates.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_pr75_tile_action_subset_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p5_encoder_remaps_subset_dictionary_to_packed_runtime_records() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 92, 0.2, 0.1, 0.0),
        builder.ActionRecord(1, 36, 109, 93, 0.1, 0.0, 0.1),
    ]

    encoded = builder._encode_actions(records, "p5")

    assert encoded.wire_format == "p5"
    assert encoded.encoded_action_codec == "brotli_custom_dict_packed24_pair10_tile8_action6"
    assert len(encoded.encoded_action_stream) == 6
    assert [entry["source_action_id"] for entry in encoded.dictionary_entries] == [92, 93]
    assert encoded.raw_runtime_records == (
        (33).to_bytes(2, "little") + bytes([109, 0])
        + (36).to_bytes(2, "little") + bytes([109, 1])
    )
    assert encoded.action_dict_raw.startswith(b"TAD1")


def test_p6_encoder_delta_varint_packs_fixed_dictionary_records() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 92, 0.2, 0.1, 0.0),
        builder.ActionRecord(1, 36, 109, 93, 0.1, 0.0, 0.1),
    ]

    encoded = builder._encode_actions(records, "p6")
    payload = builder._build_payload(
        {
            "mask_br": b"m",
            "model_br": b"r",
            "pose_br": b"p",
        },
        encoded,
        len(records),
    )

    assert encoded.wire_format == "p6"
    assert encoded.encoded_action_codec == "brotli_delta_varint_pair_tile_action"
    assert encoded.encoded_action_stream == b"\x21\x6d\x5c\x03\x6d\x5d"
    assert encoded.raw_runtime_records == (
        (33).to_bytes(2, "little") + bytes([109, 92])
        + (36).to_bytes(2, "little") + bytes([109, 93])
    )
    assert payload.startswith(b"P6")


def test_p6_encoder_rejects_custom_dictionary_and_decreasing_pairs() -> None:
    builder = _load_builder()
    custom = builder.ActionRecord(
        0,
        33,
        109,
        92,
        0.2,
        0.1,
        0.0,
        custom_delta_rgb=(1.0, 2.0, 3.0),
    )
    with pytest.raises(ValueError, match="custom action deltas require p4 or p5"):
        builder._encode_actions([custom], "p6")

    records = [
        builder.ActionRecord(0, 36, 109, 92, 0.2, 0.1, 0.0),
        builder.ActionRecord(1, 33, 109, 93, 0.1, 0.0, 0.1),
    ]
    with pytest.raises(ValueError, match="nondecreasing pairs"):
        builder._encode_actions(records, "p6")


def test_top_policy_supports_amplitude_shift_suffix() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 5, 0.2, 0.1, 0.0),
        builder.ActionRecord(1, 36, 109, 7, 0.1, 0.0, 0.1),
    ]

    selected = builder._select_policy(records, "top2_ampminus1")

    assert [record.action_id for record in selected] == [3, 5]


def test_top_drop_add_policy_uses_one_based_rank_positions() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 10, 109, 5, 0.5, 0.5, 0.0),
        builder.ActionRecord(1, 11, 109, 7, 0.4, 0.4, 0.0),
        builder.ActionRecord(2, 12, 109, 9, 0.3, 0.3, 0.0),
        builder.ActionRecord(3, 13, 109, 11, 0.2, 0.2, 0.0),
    ]

    selected = builder._select_policy(records, "top3_drop2_add4")

    assert [record.index for record in selected] == [0, 2, 3]


def test_beam_pose_policy_can_skip_high_combined_pose_harm_record() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 9, 0.20, 0.20, -0.30),
        builder.ActionRecord(1, 36, 109, 7, 0.10, 0.01, 0.20),
        builder.ActionRecord(2, 46, 108, 86, 0.08, 0.08, 0.00),
    ]

    selected = builder._select_policy(records, "beam_pose2_top3")

    assert [record.index for record in selected] == [1, 2]


def test_ampfit_uses_per_record_pose_direction_and_records_source_action() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 9, 0.2, 0.1, -0.1),
        builder.ActionRecord(1, 36, 109, 7, 0.1, 0.1, 0.1),
    ]

    selected = builder._select_policy(records, "top2_ampfit")

    assert [(record.action_id, record.source_action_id) for record in selected] == [
        (7, 9),
        (9, 7),
    ]
    assert builder._selection_guard_summary(selected)["changed_action_id_record_count"] == 2


def test_signed_boost_skips_exact_noop_residuals() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 4, 0.2, 0.1, 0.1),
        builder.ActionRecord(1, 36, 109, 0, 0.1, 0.1, 0.1),
    ]

    selected = builder._select_policy(records, "top2_signedboost1")

    assert [(record.index, record.action_id, record.transform) for record in selected] == [
        (0, 4, "identity"),
        (0, 0, "signed_same_amp0_residual"),
        (1, 0, "identity"),
    ]
    summary = builder._selection_guard_summary(selected)
    assert summary["pair_tile_duplicate_group_count"] == 1
    assert summary["exact_duplicate_record_count"] == 0
    assert sum(record.delta_combined for record in builder._unique_source_records(selected)) == pytest.approx(0.3)


def test_custom_action_deltas_require_custom_dictionary_wire_format() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 4, 0.2, 0.1, 0.1),
    ]
    selected = builder._select_policy(records, "top1_customboost125")

    with pytest.raises(ValueError, match="custom action deltas require p4 or p5"):
        builder._encode_actions(selected, "p3")

    encoded = builder._encode_actions(selected, "p5")

    assert encoded.dictionary_entries[0]["custom"] is True
    assert encoded.raw_runtime_records == (33).to_bytes(2, "little") + bytes([109, 0])


def test_wild_direction_mean_collapses_amplitudes_into_custom_dictionary() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 33, 109, 0, 0.2, 0.1, 0.1),
        builder.ActionRecord(1, 36, 109, 4, 0.1, 0.1, 0.0),
    ]

    selected = builder._select_policy(records, "top2_wilddirmean")
    encoded = builder._encode_actions(selected, "p5")

    assert [record.custom_delta_rgb for record in selected] == [
        (4.0, 4.0, 4.0),
        (4.0, 4.0, 4.0),
    ]
    assert len(encoded.dictionary_entries) == 1
    assert encoded.dictionary_entries[0]["custom"] is True


def _trace_sample(seg: float, pose: float) -> dict[str, float]:
    return {
        "score_seg_contribution_exact": seg,
        "score_pose_contribution_first_order": pose,
        "score_combined_contribution_first_order": seg + pose,
    }


def test_calibrated_lagrangian_policy_skips_bad_nonprefix_record() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 10, 109, 5, 0.9, 0.9, 0.0),
        builder.ActionRecord(1, 11, 109, 7, 0.1, 0.1, 0.0),
        builder.ActionRecord(2, 12, 109, 9, 0.1, 0.1, 0.0),
    ]
    base_trace = {
        10: _trace_sample(1.0, 1.0),
        11: _trace_sample(1.0, 1.0),
        12: _trace_sample(1.0, 1.0),
    }
    calibration = {
        "identity": builder.CalibrationTrace(
            transform="identity",
            max_rank=3,
            path=Path("identity_component_trace.json"),
            by_pair={
                10: _trace_sample(1.1, 1.1),
                11: _trace_sample(0.99, 0.99),
                12: _trace_sample(0.98, 0.98),
            },
        )
    }

    selected = builder._select_policy(
        records,
        "lag_eval_top3",
        base_trace=base_trace,
        calibration_traces=calibration,
    )

    assert [record.index for record in selected] == [1, 2]
    assert [record.calibration_rank for record in selected] == [2, 3]
    assert selected[0].delta_combined == pytest.approx(0.02)


def test_calibrated_lagrangian_policy_can_choose_amplitude_variant() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 10, 109, 9, 0.9, 0.9, 0.0),
        builder.ActionRecord(1, 11, 109, 7, 0.1, 0.1, 0.0),
    ]
    base_trace = {
        10: _trace_sample(1.0, 1.0),
        11: _trace_sample(1.0, 1.0),
    }
    calibration = {
        "identity": builder.CalibrationTrace(
            transform="identity",
            max_rank=2,
            path=Path("identity_component_trace.json"),
            by_pair={
                10: _trace_sample(1.05, 1.05),
                11: _trace_sample(0.99, 0.99),
            },
        ),
        "ampminus1": builder.CalibrationTrace(
            transform="ampminus1",
            max_rank=1,
            path=Path("ampminus1_component_trace.json"),
            by_pair={
                10: _trace_sample(0.98, 0.98),
                11: _trace_sample(1.00, 1.00),
            },
        ),
    }

    selected = builder._select_policy(
        records,
        "lag_eval_top2",
        base_trace=base_trace,
        calibration_traces=calibration,
    )

    assert [(record.index, record.action_id, record.transform) for record in selected] == [
        (0, 7, "amp_shift_-1"),
        (1, 7, "identity"),
    ]
    assert selected[0].calibration_source.startswith("ampminus1:top1:")


def test_calibrated_lagrangian_accepts_multiple_same_transform_traces() -> None:
    builder = _load_builder()
    records = [
        builder.ActionRecord(0, 10, 109, 9, 0.9, 0.9, 0.0),
        builder.ActionRecord(1, 11, 109, 7, 0.1, 0.1, 0.0),
    ]
    base_trace = {
        10: _trace_sample(1.0, 1.0),
        11: _trace_sample(1.0, 1.0),
    }
    calibration = [
        builder.CalibrationTrace(
            transform="identity",
            max_rank=2,
            path=Path("identity_weak_component_trace.json"),
            by_pair={
                10: _trace_sample(1.1, 1.1),
                11: _trace_sample(0.99, 0.99),
            },
        ),
        builder.CalibrationTrace(
            transform="identity",
            max_rank=2,
            path=Path("identity_strong_component_trace.json"),
            by_pair={
                10: _trace_sample(0.98, 0.98),
                11: _trace_sample(1.00, 1.00),
            },
        ),
    ]

    selected = builder._select_policy(
        records,
        "lag_eval_top2",
        base_trace=base_trace,
        calibration_traces=calibration,
    )

    assert [record.index for record in selected] == [0, 1]
    assert "identity_strong_component_trace" in selected[0].calibration_source
