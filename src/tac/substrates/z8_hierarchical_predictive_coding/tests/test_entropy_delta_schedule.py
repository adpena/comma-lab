# SPDX-License-Identifier: MIT
"""Tests for Z8 per-subband entropy-delta schedule selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    build_entropy_delta_materializer_work_order,
    build_entropy_delta_schedule_from_headroom_report,
    coerce_entropy_detail_quantization_steps,
    parse_aggregate_subband_key,
)


def test_parse_aggregate_subband_key() -> None:
    assert parse_aggregate_subband_key("L0_hh") == (0, "hh")
    assert parse_aggregate_subband_key("L12_lh") == (12, "lh")
    with pytest.raises(ValueError):
        parse_aggregate_subband_key("frame_0_details:0:hh")
    with pytest.raises(ValueError):
        parse_aggregate_subband_key("L0_bad")


def test_build_entropy_delta_schedule_selects_min_bytes_within_mse_budget() -> None:
    report = {
        "tool": "z8_detail_coeff_entropy_headroom_report",
        "archive_path": "archive/0.bin",
        "archive_total_bytes": 1000,
        "wavelet_blob_bytes": 900,
        "pairs_measured": 2,
        "per_subband": [
            {
                "subband": "L0_hh",
                "quant_sweep": [
                    {
                        "quant_step": 0.00390625,
                        "distortion_mse": 1.0e-8,
                        "live_codec_method": "zigzag_u16_byteplane",
                        "live_codec_brotli_bytes_per_coeff": 0.4,
                    },
                    {
                        "quant_step": 0.03125,
                        "distortion_mse": 8.0e-6,
                        "live_codec_method": "qi16_constriction_range",
                        "live_codec_brotli_bytes_per_coeff": 0.2,
                    },
                    {
                        "quant_step": 0.25,
                        "distortion_mse": 1.0e-3,
                        "live_codec_method": "qi16_zero_rle",
                        "live_codec_brotli_bytes_per_coeff": 0.01,
                    },
                ],
            },
            {
                "subband": "L1_lh",
                "quant_sweep": [
                    {
                        "quant_step": 0.00390625,
                        "distortion_mse": 2.0e-8,
                        "live_codec_method": "zigzag_u16_byteplane",
                        "live_codec_brotli_bytes_per_coeff": 0.7,
                    },
                    {
                        "quant_step": 0.0625,
                        "distortion_mse": 2.0e-4,
                        "live_codec_method": "qi16_zero_rle",
                        "live_codec_brotli_bytes_per_coeff": 0.02,
                    },
                ],
            },
        ],
    }

    schedule = build_entropy_delta_schedule_from_headroom_report(report, max_subband_mse=1.0e-5)

    assert schedule["ready_for_materializer"] is True
    assert schedule["score_claim"] is False
    assert schedule["entropy_detail_quantization_steps"] == {
        "frame_0_details:0:hh": 0.03125,
        "frame_1_details:0:hh": 0.03125,
        "frame_0_details:1:lh": 0.00390625,
        "frame_1_details:1:lh": 0.00390625,
    }
    reasons = {row["aggregate_subband"]: row["selection_reason"] for row in schedule["chosen_subbands"]}
    assert reasons == {
        "L0_hh": "within_max_subband_mse_min_bytes",
        "L1_lh": "within_max_subband_mse_min_bytes",
    }


def test_build_entropy_delta_schedule_falls_back_to_min_distortion_when_needed() -> None:
    report = {
        "per_subband": [
            {
                "subband": "L0_hl",
                "quant_sweep": [
                    {
                        "quant_step": 0.125,
                        "distortion_mse": 2.0e-4,
                        "live_codec_method": "qi16_zero_rle",
                        "live_codec_brotli_bytes_per_coeff": 0.05,
                    },
                    {
                        "quant_step": 0.00390625,
                        "distortion_mse": 5.0e-6,
                        "live_codec_method": "zigzag_u16_byteplane",
                        "live_codec_brotli_bytes_per_coeff": 0.5,
                    },
                ],
            }
        ],
    }

    schedule = build_entropy_delta_schedule_from_headroom_report(report, max_subband_mse=1.0e-8)

    assert schedule["entropy_detail_quantization_steps"] == {
        "frame_0_details:0:hl": 0.00390625,
        "frame_1_details:0:hl": 0.00390625,
    }
    assert schedule["chosen_subbands"][0]["selection_reason"] == "no_admissible_step_min_distortion_fallback"


def test_entropy_delta_schedule_blocks_partial_headroom_coverage_by_default() -> None:
    report = {
        "pairs_measured": 6,
        "total_pairs_in_archive": 600,
        "per_subband": [
            {
                "subband": "L0_hh",
                "quant_sweep": [
                    {
                        "quant_step": 0.03125,
                        "distortion_mse": 1.0e-7,
                        "live_codec_method": "qi16_constriction_range",
                        "live_codec_brotli_bytes_per_coeff": 0.2,
                    }
                ],
            }
        ],
    }

    schedule = build_entropy_delta_schedule_from_headroom_report(report, max_subband_mse=1.0e-6)

    assert schedule["ready_for_materializer"] is False
    assert schedule["blockers"] == ["partial_headroom_coverage:6/600"]
    with pytest.raises(ValueError, match="not ready_for_materializer"):
        coerce_entropy_detail_quantization_steps(schedule)


def test_entropy_delta_schedule_can_explicitly_allow_partial_advisory_coverage() -> None:
    report = {
        "pairs_measured": 6,
        "total_pairs_in_archive": 600,
        "per_subband": [
            {
                "subband": "L0_hh",
                "quant_sweep": [
                    {
                        "quant_step": 0.03125,
                        "distortion_mse": 1.0e-7,
                        "live_codec_method": "qi16_constriction_range",
                        "live_codec_brotli_bytes_per_coeff": 0.2,
                    }
                ],
            }
        ],
    }

    schedule = build_entropy_delta_schedule_from_headroom_report(
        report,
        max_subband_mse=1.0e-6,
        require_full_archive_coverage=False,
    )

    assert schedule["ready_for_materializer"] is True
    assert coerce_entropy_detail_quantization_steps(schedule) == {
        "frame_0_details:0:hh": 0.03125,
        "frame_1_details:0:hh": 0.03125,
    }


def test_entropy_delta_schedule_work_order_executes_ready_schedule(tmp_path: Path) -> None:
    archive_bin = tmp_path / "0.bin"
    archive_bin.write_bytes(b"z8")
    schedule = {
        "schema": "z8_entropy_delta_schedule.v2",
        "ready_for_materializer": True,
        "schedule_sha256": "schedule-sha",
        "source_report_sha256": "report-sha",
        "source_archive_sha256": "archive-sha",
        "source_archive_path": archive_bin.as_posix(),
        "blockers": [],
        "entropy_detail_quantization_steps": {
            "frame_0_details:0:hh": 0.03125,
            "frame_1_details:0:hh": 0.03125,
        },
    }

    work_order = build_entropy_delta_materializer_work_order(
        schedule,
        schedule_json_path="runs/z8/schedule.json",
        output_dir="runs/z8/materialized",
        emit_receiver_proof=True,
        run_inflate_runtime_benchmark=True,
    )

    assert work_order["schema"] == "z8_entropy_delta_materializer_work_order.v1"
    assert work_order["ready_for_materializer_execution"] is True
    assert work_order["score_claim"] is False
    assert work_order["source_archive_bin"] == archive_bin.as_posix()
    assert work_order["step_count"] == 2
    command = work_order["materializer_command"]
    assert command[:2] == [
        ".venv/bin/python",
        "tools/materialize_z8_joint_p18_p19_deadzone_candidate.py",
    ]
    assert "--no-mutate-coefficients" in command
    assert "--entropy-code-quantized-details" in command
    assert "--emit-receiver-proof" in command
    assert "--run-inflate-runtime-benchmark" in command
    assert work_order["exact_axis_blocker"] == "contest_cpu_cuda_eval_not_executed"


def test_entropy_delta_schedule_work_order_blocks_not_ready_schedule() -> None:
    schedule = {
        "schema": "z8_entropy_delta_schedule.v2",
        "ready_for_materializer": False,
        "blockers": ["partial_headroom_coverage:6/600"],
        "entropy_detail_quantization_steps": {
            "frame_0_details:0:hh": 0.03125,
        },
    }

    work_order = build_entropy_delta_materializer_work_order(
        schedule,
        schedule_json_path="runs/z8/schedule.json",
        output_dir="runs/z8/materialized",
        archive_bin="runs/z8/0.bin",
    )

    assert work_order["ready_for_materializer_execution"] is False
    assert work_order["materializer_command"] is None
    assert work_order["blockers"]
    assert work_order["score_claim"] is False


def test_entropy_delta_schedule_work_order_blocks_missing_archive_bin() -> None:
    schedule = {
        "schema": "z8_entropy_delta_schedule.v2",
        "ready_for_materializer": True,
        "blockers": [],
        "entropy_detail_quantization_steps": {
            "frame_0_details:0:hh": 0.03125,
        },
    }

    work_order = build_entropy_delta_materializer_work_order(
        schedule,
        schedule_json_path="runs/z8/schedule.json",
        output_dir="runs/z8/materialized",
        archive_bin="runs/z8/missing.bin",
    )

    assert work_order["ready_for_materializer_execution"] is False
    assert work_order["materializer_command"] is None
    assert work_order["blockers"] == ["source_archive_bin_missing_on_disk:runs/z8/missing.bin"]
