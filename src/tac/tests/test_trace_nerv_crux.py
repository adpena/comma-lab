# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "trace_nerv_crux.py"


def _run_trace(tmp_path: Path, payload: dict[str, object], *extra: str) -> list[dict]:
    artifact = tmp_path / "training_artifact.json"
    out = tmp_path / "trace_rows.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--training-artifact",
            str(artifact),
            "--out",
            str(out),
            *extra,
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    return rows


def _by_metric(rows: list[dict]) -> dict[str, dict]:
    return {str(row["metric"]): row for row in rows}


def _receiver_surface_trace(**overrides: float) -> dict[str, float]:
    trace = {
        "receiver_surface_loss_delta": -0.01,
        "receiver_surface_float_rgb_delta_linf": 0.02,
        "receiver_surface_uint8_changed_pixels": 7.0,
        "receiver_surface_uint8_delta_abs_max": 2.0,
        "receiver_surface_segnet_input_delta_linf": 0.015,
        "receiver_surface_worst_region_margin_p50_delta": 0.0002,
        "receiver_surface_argmax_flipped_pixels": 3.0,
        "receiver_surface_argmax_changed_count_region": 5.0,
        "receiver_surface_target_hard_won_count": 2.0,
        "receiver_surface_target_hard_lost_count": 0.0,
        "receiver_surface_net_target_support_delta": 2.0,
        "receiver_surface_wrong_to_target_count": 2.0,
        "receiver_surface_target_to_wrong_count": 0.0,
        "receiver_surface_wrong_to_wrong_count": 3.0,
        "receiver_surface_posenet_input_delta_linf": 0.03,
        "receiver_surface_pose_output_delta": -0.05,
        "receiver_surface_fakequant_argmax_flipped_pixels": 3.0,
        "receiver_surface_fakequant_margin_delta": 0.0002,
        "receiver_surface_fakequant_pose_output_delta": 0.05,
        "receiver_surface_fakequant_survival": 1.0,
        "receiver_surface_parseback_argmax_flipped_pixels": 3.0,
        "receiver_surface_parseback_pose_output_delta": 0.05,
        "receiver_surface_parseback_survival": 1.0,
        "receiver_surface_inflated_argmax_flipped_pixels": 3.0,
        "receiver_surface_inflated_pose_output_delta": 0.05,
        "receiver_surface_inflate_survival": 1.0,
    }
    trace.update(overrides)
    return trace


def test_trace_nerv_crux_emits_contest_unit_rows_without_score_claim(tmp_path: Path) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 50.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.2,
                "loss_part_pose_direct_live_raw_mse": 0.00196,
                "loss_part_pose_direct_live_score_term": 0.14,
                "loss_part_pose_direct_live_score_marginal_wrt_raw_mse": 35.714285714285715,
                "loss_part_pose_direct_live_yuv6_pair_std": 0.22,
                "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std": 0.08,
                "train_time_archive_bytes": 150_000,
            },
            "receiver_surface_trace": _receiver_surface_trace(),
        },
    )

    metrics = _by_metric(rows)
    assert metrics["score_weighted_total_unsolved_argmax_mass"]["score_units"] == pytest.approx(50.0)
    assert metrics["pose_direct_live_score_term"]["score_units"] == pytest.approx(0.14)
    assert metrics["pose_direct_live_score_marginal_wrt_raw_mse"]["value"] == pytest.approx(
        5.0 / ((10.0 * 0.00196) ** 0.5)
    )
    assert metrics["archive_rate_score"]["score_units"] == pytest.approx(25.0 * 150_000 / 37_545_489)
    assert metrics["receiver_surface_uint8_changed_pixels"]["value"] == pytest.approx(7.0)
    assert metrics["receiver_surface_uint8_delta_abs_max"]["value"] == pytest.approx(2.0)
    assert metrics["receiver_surface_parseback_argmax_flipped_pixels"]["value"] == pytest.approx(3.0)
    assert metrics["receiver_surface_fakequant_pose_output_delta"]["value"] == pytest.approx(0.05)
    assert metrics["receiver_surface_fakequant_survival"]["value"] == pytest.approx(1.0)
    assert metrics["receiver_surface_parseback_pose_output_delta"]["value"] == pytest.approx(0.05)
    assert metrics["receiver_surface_parseback_survival"]["value"] == pytest.approx(1.0)
    assert metrics["receiver_surface_inflated_pose_output_delta"]["value"] == pytest.approx(0.05)
    assert metrics["receiver_surface_inflate_survival"]["value"] == pytest.approx(1.0)
    assert metrics["receiver_surface_target_hard_won_count"]["value"] == pytest.approx(2.0)
    assert metrics["receiver_surface_net_target_support_delta"]["value"] == pytest.approx(2.0)
    assert {row["authority"] for row in rows} == {"macos_mlx_false_authority_no_score_claim"}
    for row in rows:
        assert "score_claim" not in row
        assert "promotion_eligible" not in row


def test_trace_nerv_crux_blocks_missing_required_direct_live_posenet(tmp_path: Path) -> None:
    rows = _run_trace(
        tmp_path,
        {"final_loss_components": {"train_time_archive_bytes": 1_000}},
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_direct_live_posenet_path" in blockers


def test_trace_nerv_crux_blocks_missing_direct_live_segnet_path(tmp_path: Path) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            }
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_direct_live_segnet_path" in blockers


def test_trace_nerv_crux_blocks_segnet_path_without_target_region_debt(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            }
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_direct_live_segnet_target_region_debt" in blockers


def test_trace_nerv_crux_blocks_missing_receiver_surface_trace(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            }
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_receiver_surface_trace" in blockers


def test_trace_nerv_crux_allows_missing_receiver_surface_for_forensic_read(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            }
        },
        "--allow-missing-receiver-surface-trace",
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_receiver_surface_trace" not in blockers


def test_trace_nerv_crux_blocks_loss_improvement_without_uint8_motion(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": _receiver_surface_trace(
                receiver_surface_loss_delta=-0.25,
                receiver_surface_uint8_changed_pixels=0.0,
            ),
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_loss_improved_without_uint8_motion" in blockers


def test_trace_nerv_crux_blocks_uint8_motion_without_argmax_or_margin_motion(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": _receiver_surface_trace(
                receiver_surface_uint8_changed_pixels=5.0,
                receiver_surface_worst_region_margin_p50_delta=0.0,
                receiver_surface_argmax_flipped_pixels=0.0,
            ),
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_uint8_motion_without_argmax_or_margin_motion" in blockers


def test_trace_nerv_crux_refuses_pair_local_smoke_as_receiver_trace(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "substrate_artifact_metadata": {
                "substrate_supplied_score_aware_training": {
                    "output_head_target_bias_init": {
                        "scorer_domain_bootstrap": {
                            "max_accepted_frame1_delta_abs": 0.48111245036125183,
                            "max_accepted_frame1_receiver_uint8_changed_count": 556860,
                        }
                    }
                }
            },
        },
    )

    metrics = _by_metric(rows)
    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert metrics["receiver_surface_trace_present"]["value"] == pytest.approx(0.0)
    assert metrics["receiver_surface_float_rgb_delta_linf"]["value"] is None
    assert metrics["receiver_surface_uint8_changed_pixels"]["value"] is None
    assert "missing_receiver_surface_trace" in blockers


def test_trace_nerv_crux_blocks_argmax_motion_without_fakequant_survival(
    tmp_path: Path,
) -> None:
    trace = _receiver_surface_trace()
    trace.pop("receiver_surface_fakequant_argmax_flipped_pixels")
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": trace,
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_fakequant_survival_missing" in blockers


def test_trace_nerv_crux_blocks_argmax_churn_without_target_support_breakdown(
    tmp_path: Path,
) -> None:
    trace = _receiver_surface_trace()
    trace.pop("receiver_surface_target_hard_won_count")
    trace.pop("receiver_surface_net_target_support_delta")
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": trace,
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_argmax_motion_without_target_support_breakdown" in blockers


def test_trace_nerv_crux_blocks_fakequant_motion_without_parseback_survival(
    tmp_path: Path,
) -> None:
    trace = _receiver_surface_trace()
    trace.pop("receiver_surface_parseback_argmax_flipped_pixels")
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": trace,
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_parseback_survival_missing" in blockers


def test_trace_nerv_crux_blocks_parseback_motion_without_inflate_survival(
    tmp_path: Path,
) -> None:
    trace = _receiver_surface_trace()
    trace.pop("receiver_surface_inflated_argmax_flipped_pixels")
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": trace,
        },
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "receiver_surface_inflate_survival_missing" in blockers


def test_trace_nerv_crux_ignores_legacy_receiver_surface_aliases(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                "loss_part_pose_direct_live_raw_mse": 0.004,
                "train_time_archive_bytes": 1_000,
            },
            "receiver_surface_trace": {
                "loss_delta": -0.01,
                "float_rgb_delta_linf": 0.02,
                "uint8_changed_pixels": 7.0,
                "argmax_flipped_pixels": 3.0,
            },
        },
    )

    metrics = _by_metric(rows)
    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert metrics["receiver_surface_trace_present"]["value"] == pytest.approx(0.0)
    assert metrics["receiver_surface_uint8_changed_pixels"]["value"] is None
    assert metrics["receiver_surface_argmax_flipped_pixels"]["value"] is None
    assert "missing_receiver_surface_trace" in blockers


def test_trace_nerv_crux_reads_nested_readiness_metrics_and_derives_pose_terms(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "direct_live_segnet_gate": {
                "metrics": {
                    "score_weighted_total_unsolved_argmax_mass": 12.5,
                    "segnet_direct_live_candidate_target_class_min_ratio": 0.4,
                }
            },
            "direct_live_posenet_gate": {
                "metrics": {
                    "pose_direct_live_raw_mse": 0.004,
                    "pose_direct_live_yuv6_pair_std": 0.22,
                }
            },
            "metrics": {"archive_bytes": 1_024},
            "receiver_surface_trace": _receiver_surface_trace(),
        },
    )

    metrics = _by_metric(rows)
    assert metrics["score_weighted_total_unsolved_argmax_mass"]["score_units"] == pytest.approx(12.5)
    assert metrics["candidate_target_class_min_ratio"]["value"] == pytest.approx(0.4)
    assert metrics["pose_direct_live_score_term"]["score_units"] == pytest.approx((10.0 * 0.004 + 1.0e-12) ** 0.5)
    assert metrics["pose_direct_live_score_marginal_wrt_raw_mse"]["value"] == pytest.approx(
        5.0 / ((10.0 * 0.004 + 1.0e-12) ** 0.5)
    )
    assert metrics["archive_rate_score"]["score_units"] == pytest.approx(25.0 * 1_024 / 37_545_489)
    assert not [row.get("blocker") for row in rows if row.get("blocker")]


def test_trace_nerv_crux_reads_runner_per_epoch_metrics_and_top_level_rate(
    tmp_path: Path,
) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "archive_bytes": 216_000,
            "per_epoch_metrics": [
                {
                    "loss_components": {
                        "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 99.0,
                        "loss_part_pose_direct_live_raw_mse": 0.04,
                    }
                },
                {
                    "loss_components": {
                        "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 57.648216247558594,
                        "loss_part_segnet_direct_live_argmax_disagreement": 0.5764821767807007,
                        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                        "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
                        "loss_part_pose_direct_live_raw_mse": 180.729,
                        "loss_part_pose_direct_live_yuv6_pair_std": 0.12,
                        "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std": 0.03,
                    }
                },
            ],
            "receiver_surface_trace": _receiver_surface_trace(),
        },
    )

    metrics = _by_metric(rows)
    assert metrics["score_weighted_total_unsolved_argmax_mass"]["score_units"] == pytest.approx(57.648216247558594)
    assert metrics["argmax_disagreement"]["value"] == pytest.approx(0.5764821767807007)
    assert metrics["pose_direct_live_score_term"]["score_units"] == pytest.approx((10.0 * 180.729 + 1.0e-12) ** 0.5)
    assert metrics["pose_direct_live_score_marginal_wrt_raw_mse"]["value"] == pytest.approx(
        5.0 / ((10.0 * 180.729 + 1.0e-12) ** 0.5)
    )
    assert metrics["archive_bytes"]["value"] == pytest.approx(216_000.0)
    assert metrics["archive_rate_score"]["score_units"] == pytest.approx(25.0 * 216_000 / 37_545_489)
    assert not [row.get("blocker") for row in rows if row.get("blocker")]


def test_trace_nerv_crux_uses_telemetry_jsonl_fallback(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "loss_components": {
                            "loss_part_pose_direct_live_raw_mse": 100.0,
                        }
                    }
                ),
                json.dumps(
                    {
                        "loss_components": {
                            "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                            "loss_part_pose_direct_live_raw_mse": 0.0025,
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _run_trace(
        tmp_path,
        {
            "archive_bytes": 10_000,
            "telemetry_path": telemetry.as_posix(),
            "receiver_surface_trace": _receiver_surface_trace(),
        },
    )

    metrics = _by_metric(rows)
    assert metrics["score_weighted_total_unsolved_argmax_mass"]["score_units"] == pytest.approx(12.5)
    assert metrics["pose_direct_live_raw_mse"]["value"] == pytest.approx(0.0025)
    assert not [row.get("blocker") for row in rows if row.get("blocker")]


def test_trace_nerv_crux_reads_sibling_birth_survival_retention(
    tmp_path: Path,
) -> None:
    action_id = "a" * 64
    artifact = tmp_path / "training_artifact.json"
    out = tmp_path / "trace_rows.json"
    artifact.write_text(
        json.dumps(
            {
                "final_loss_components": {
                    "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 10.0,
                    "loss_part_pose_direct_live_raw_mse": 1.0,
                    "train_time_archive_bytes": 16_325,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hi_nerv_birth_action_effects.jsonl").write_text(
        json.dumps(
            {
                "schema": "tac.action_effect.v1",
                "action_id": action_id,
                "authority": "batch_local_live_mlx",
                "wrong_to_target": 13_488,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "hi_nerv_birth_fakequant_survival.json").write_text(
        json.dumps(
            {
                "schema": "hi_nerv_target_region_birth_survival.v1",
                "action_id": action_id,
                "surface": "fakequant_mlx",
                "survived": True,
                "wrong_to_target_count": 12_183,
                "fakequant_target_margin_certificate": {
                    "schema": "tac.target_margin_certificate.v1",
                    "action_id": action_id,
                    "surface": "fakequant_mlx",
                    "target_margin_floor": 0.0,
                    "target_margin_floor_satisfied": True,
                    "target_margin_min": -0.02,
                    "target_margin_p10": 0.08,
                    "target_margin_mean": 0.57,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hi_nerv_selected_birth_parseback_survival.json").write_text(
        json.dumps(
            {
                "schema": "hi_nerv_target_region_birth_survival.v1",
                "action_id": action_id,
                "surface": "parseback_mlx",
                "survived": True,
                "wrong_to_target_count": 2,
                "parseback_target_margin_certificate": {
                    "schema": "tac.target_margin_certificate.v1",
                    "action_id": action_id,
                    "surface": "parseback_mlx",
                    "target_margin_floor": 0.0,
                    "target_margin_floor_satisfied": False,
                    "target_margin_min": -2.41,
                    "target_margin_p10": -1.33,
                    "target_margin_mean": -0.22,
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--training-artifact",
            str(artifact),
            "--out",
            str(out),
            "--allow-missing-direct-live-segnet",
            "--allow-missing-direct-live-posenet",
            "--allow-missing-receiver-surface-trace",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    rows = json.loads(out.read_text(encoding="utf-8"))
    metrics = _by_metric(rows)
    assert metrics["live_wrong_to_target_count"]["value"] == pytest.approx(13_488.0)
    assert metrics["fakequant_wrong_to_target_count"]["value"] == pytest.approx(12_183.0)
    assert metrics["fakequant_wrong_to_target_retention_ratio"]["value"] == pytest.approx(
        12_183 / 13_488
    )
    assert metrics["parseback_wrong_to_target_count"]["value"] == pytest.approx(2.0)
    assert metrics["parseback_wrong_to_target_retention_ratio"]["value"] == pytest.approx(
        2 / 13_488
    )
    assert metrics["fakequant_target_margin_p10"]["value"] == pytest.approx(0.08)
    assert metrics["fakequant_target_margin_floor_satisfied"]["value"] == pytest.approx(1.0)
    assert metrics["parseback_target_margin_p10"]["value"] == pytest.approx(-1.33)
    assert metrics["parseback_target_margin_floor_satisfied"]["value"] == pytest.approx(0.0)
    assert metrics["parseback_scorer_effect_survived"]["value"] == pytest.approx(0.0)
    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "hinerv_birth_parseback_margin_floor_failed" in blockers
    assert "hinerv_birth_parseback_scorer_effect_collapse" in blockers
