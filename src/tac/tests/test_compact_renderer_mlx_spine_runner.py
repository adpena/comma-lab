# SPDX-License-Identifier: MIT
"""Tests for the MLX-first compact renderer spine runner."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import signal
import struct
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.analysis.nerv_modelsize_budget import enumerate_snerv_modelsize_candidates
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_compact_renderer_mlx_spine_runner as runner_mod  # noqa: E402
from tools.run_compact_renderer_mlx_spine_runner import (  # noqa: E402
    COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
    CompactRendererMlxSpineRunnerError,
    _hi_nerv_pose_trusted_birth_payload_blockers,
    _parse_args,
    _pr95_long_campaign_prelaunch_blockers,
    _require_scorer_upstream_dir_for_distillation,
    _resolve_execute_modelsize_candidate,
    _resolve_source_video_path,
    _validate_hi_nerv_frontier_training_config,
    adapt_pr95_mlx_report_to_spine,
    adapt_pr95_stage8_report_to_spine,
    build_plan_only_report,
    execute_hi_nerv_mlx_scoreaware_and_adapt,
    execute_pact_nerv_selector_v4_mlx_smoke_and_adapt,
    execute_pr95_hnerv_mlx_scoreaware_and_adapt,
    execute_snerv_inverse_steg_advisory_and_adapt,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    import av as _av  # noqa: F401

    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


def test_hinerv_runner_short_smoke_readiness_consumes_strict_launch_actuators() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    helper = source[
        source.index("def _write_hi_nerv_runner_short_scorer_smoke_readiness(") : source.index(
            "def _attach_hi_nerv_short_scorer_smoke_readiness("
        )
    ]
    call = source[
        source.index("short_scorer_smoke_readiness = _write_hi_nerv_runner_short_scorer_smoke_readiness(") : source.index(
            "short_scorer_smoke_readiness_summary ="
        )
    ]

    assert "require_section_byte_dual_ascent" in helper
    assert "require_pose_direct_live_distillation" in helper
    assert "decoder_weight_waterfill_plan_metadata" in helper
    assert "output_head_target_bias_init_metadata" in helper
    assert "_reload_hi_nerv_training_artifact_for_readiness(" in helper
    assert "_substrate_score_aware_training_from_artifact(" in helper
    assert "readiness_artifact_dict" in helper
    assert "require_section_byte_dual_ascent=launch_hard_byte_ceiling is not None" in call
    assert "require_pose_direct_live_distillation=True" in call
    assert "decoder_weight_waterfill_plan_metadata=(" in call
    assert '"segnet_direct_live_target_mass_floor_weight": float(' in call
    assert "segnet_direct_live_target_mass_floor_weight" in call
    assert '"segnet_direct_live_target_min_ratio_floor_weight": float(' in call
    assert "segnet_direct_live_target_min_ratio_floor_weight" in call


def test_pose_trusted_birth_payload_validator_rejects_metadata_only_acceptance() -> None:
    blockers = _hi_nerv_pose_trusted_birth_payload_blockers(
        {
            "schema": "hi_nerv_target_region_birth.v1",
            "accepted": True,
            "accepted_step_count": 1,
            "action_id": "a" * 64,
        }
    )

    assert "hi_nerv_pose_trusted_birth_receipt_missing" in blockers
    assert "hi_nerv_pose_trusted_birth_target_hard_won_missing" in blockers
    assert "hi_nerv_pose_trusted_birth_pose_cap_telemetry_missing" in blockers
    assert "hi_nerv_pose_trusted_birth_exact_nonrate_not_improved" in blockers
    assert "hi_nerv_pose_trusted_birth_candidate_frontier_missing" in blockers


def test_pose_trusted_birth_payload_validator_accepts_receiver_closed_receipt() -> None:
    blockers = _hi_nerv_pose_trusted_birth_payload_blockers(
        {
            "schema": "hi_nerv_target_region_birth.v1",
            "accepted": True,
            "action_id": "a" * 64,
            "receipt": {
                "schema": "hi_nerv_target_region_birth_receipt.v1",
                "surface": "live_mlx",
                "action_id": "a" * 64,
                "accepted_step_count": 1,
                "updated_parameter_names": ["head_rgb_1.weight"],
                "argmax_transitions": {
                    "target_hard_won_count": 3,
                    "target_hard_lost_count": 0,
                    "net_target_support_delta": 3,
                },
                "pose_guard": {
                    "available": True,
                    "pose_input_contest_resolution": True,
                    "max_accepted_pose_output_delta_l2": 0.025,
                    "max_pose_output_delta_l2": 0.05,
                },
                "exact_nonrate": {
                    "pose_term_available": True,
                    "delta_score_nonrate": -0.1,
                },
                "candidate_frontier_telemetry": {
                    "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
                    "candidate_attempt_count": 2,
                },
                "pose_compensation": {
                    "composite_accepted": True,
                    "frame1_receiver_uint8_unchanged_by_compensation": True,
                    "compensation_updated_parameter_names": ["head_rgb_0.bias"],
                },
            },
        }
    )

    assert blockers == []


def test_pose_trusted_birth_payload_validator_rejects_already_won_spill() -> None:
    blockers = _hi_nerv_pose_trusted_birth_payload_blockers(
        {
            "schema": "hi_nerv_target_region_birth.v1",
            "accepted": True,
            "action_id": "a" * 64,
            "receipt": {
                "schema": "hi_nerv_target_region_birth_receipt.v1",
                "surface": "live_mlx",
                "action_id": "a" * 64,
                "accepted_step_count": 1,
                "updated_parameter_names": ["head_rgb_1.weight"],
                "argmax_transitions": {
                    "target_hard_won_count": 5,
                    "target_hard_lost_count": 1,
                    "target_to_wrong_count": 1,
                    "net_target_support_delta": 4,
                },
                "pose_guard": {
                    "available": True,
                    "pose_input_contest_resolution": True,
                    "max_accepted_pose_output_delta_l2": 0.025,
                    "max_pose_output_delta_l2": 0.05,
                },
                "exact_nonrate": {
                    "pose_term_available": True,
                    "delta_score_nonrate": -0.1,
                },
                "candidate_frontier_telemetry": {
                    "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
                    "candidate_attempt_count": 2,
                    "final_already_won_lost_count": 1,
                },
            },
        }
    )

    assert "hi_nerv_pose_trusted_birth_target_hard_lost" in blockers
    assert "hi_nerv_pose_trusted_birth_target_to_wrong" in blockers
    assert "hi_nerv_pose_trusted_birth_already_won_lost" in blockers


def test_hinerv_runner_calls_pose_trusted_birth_validator_after_actuator() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    body = source[
        source.index("target_region_birth_payload = dict(") : source.index(
            "except Exception as exc:",
            source.index("target_region_birth_payload = dict("),
        )
    ]

    assert "_hi_nerv_pose_trusted_birth_payload_blockers(" in body
    assert 'target_region_birth_payload["accepted"] = False' in body
    assert 'target_region_birth_payload["pose_trusted_validated"]' in body


def test_hinerv_runner_binds_target_support_floor_to_loss_and_contract() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    body = source[
        source.index("def execute_hi_nerv_mlx_scoreaware_and_adapt(") : source.index(
            "def _compact_score_aware_training_telemetry_contract("
        )
    ]
    main_hi_branch = source[
        source.index('elif args.execute_family == "hi_nerv":') : source.index(
            "elif args.execute_family in PLANNER_GATED_FAMILIES:"
        )
    ]
    execute_to_smoke_call = body[
        body.index("artifact = _run_hi_nerv_mlx_scoreaware_smoke(") : body.index("artifact_dict = artifact.as_dict()")
    ]
    compact_body = "".join(body.split())
    compact_main_hi_branch = "".join(main_hi_branch.split())
    compact_execute_to_smoke_call = "".join(execute_to_smoke_call.split())

    assert "segnet_direct_live_target_mass_floor_weight=float(segnet_direct_live_target_mass_floor_weight)," in compact_body
    assert "segnet_direct_live_target_min_ratio_floor_weight=float(segnet_direct_live_target_min_ratio_floor_weight)," in compact_body
    assert "segnet_direct_live_target_mass_floor_weight=float(segnet_direct_live_target_mass_floor_weight),segnet_direct_live_target_min_ratio_floor_weight=float(" in compact_body
    assert "segnet_direct_live_target_mass_floor_weight=(args.segnet_direct_live_target_mass_floor_weight)," in compact_main_hi_branch
    assert "segnet_direct_live_target_min_ratio_floor_weight=(args.segnet_direct_live_target_min_ratio_floor_weight)," in compact_main_hi_branch
    assert "posenet_yuv6_geometry_tether_stage_weight=(args.posenet_yuv6_geometry_tether_stage_weight)," in compact_main_hi_branch
    assert "segnet_direct_live_target_mass_floor_weight=float(segnet_direct_live_target_mass_floor_weight)," in compact_execute_to_smoke_call
    assert "segnet_direct_live_target_min_ratio_floor_weight=float(segnet_direct_live_target_min_ratio_floor_weight)," in compact_execute_to_smoke_call


def test_hinerv_runner_binds_archive_parseback_selection_to_receiver_gate() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    body = source[
        source.index("def execute_hi_nerv_mlx_scoreaware_and_adapt(") : source.index(
            "def _compact_score_aware_training_telemetry_contract("
        )
    ]
    execute_to_smoke_call = body[
        body.index("artifact = _run_hi_nerv_mlx_scoreaware_smoke(") : body.index("artifact_dict = artifact.as_dict()")
    ]
    smoke_body = source[
        source.index("def _run_hi_nerv_mlx_scoreaware_smoke(") : source.index(
            "def _write_hi_nerv_runner_source_pair_reference_cache("
        )
    ]
    compact_body = "".join(body.split())
    compact_execute_to_smoke_call = "".join(execute_to_smoke_call.split())
    compact_smoke_body = "".join(smoke_body.split())

    assert "archive_selection_quality_scope=_hi_nerv_effective_receiver_cache_quality_max_pairs(" in compact_body
    assert "post_export_receiver_cache_quality_gate" in execute_to_smoke_call
    assert "archive_selection_replay_required=bool(" in compact_execute_to_smoke_call
    assert 'archive_selection_quality_scope["effective_max_pairs"]' in compact_execute_to_smoke_call
    assert "build_hi_nerv_archive_replay_components" in smoke_body
    assert '"archive_replay_components_fn"' in smoke_body
    assert "archive_selection_replay_required=bool(" in compact_smoke_body
    assert "archive_selection_replay_batch_size=(" in compact_smoke_body


def test_hinerv_receiver_replay_archive_selection_prefers_archive_backed_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_archive = tmp_path / "live" / "archive.zip"
    ema_archive = tmp_path / "ema" / "archive.zip"
    live_archive.parent.mkdir()
    ema_archive.parent.mkdir()
    live_archive.write_bytes(b"live")
    ema_archive.write_bytes(b"ema")
    calls: list[Path] = []

    def fake_receiver_quality(**kwargs):
        archive = Path(kwargs["archive_zip_path"])
        calls.append(archive)
        score = 12.0 if archive == live_archive else 25.0
        report_dir = Path(kwargs["output_dir"]) / "post_export_receiver_cache_quality"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "hi_nerv_receiver_cache_quality_report.json"
        report = {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": report_path.as_posix(),
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "quality_gate_passed": False,
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_probe": {
                "fit_gate_passed": False,
                "canonical_score": score,
                "avg_segnet_dist": score / 100.0,
                "avg_posenet_dist": score,
            },
            "blockers": ["test_false_authority"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_post_export_receiver_cache_quality",
        fake_receiver_quality,
    )

    manifest = runner_mod._write_hi_nerv_runner_receiver_replay_archive_selection(
        requested=True,
        archive_resolution={
            "candidates": [
                {
                    "candidate_kind": "ema",
                    "archive_path": ema_archive.as_posix(),
                    "archive_sha256": runner_mod._sha256_file(ema_archive),
                    "archive_bytes": ema_archive.stat().st_size,
                    "selected_for_training_artifact": True,
                    "diagnostic_only": False,
                },
                {
                    "candidate_kind": "live",
                    "archive_path": live_archive.as_posix(),
                    "archive_sha256": runner_mod._sha256_file(live_archive),
                    "archive_bytes": live_archive.stat().st_size,
                    "selected_for_training_artifact": False,
                    "diagnostic_only": True,
                },
            ],
        },
        source_video_path=tmp_path / "source.mkv",
        output_dir=tmp_path / "out",
        reference_cache_dir=None,
        max_pairs=4,
        batch_pairs=1,
        min_segnet_std=1.0,
        min_segnet_dynamic_range=16.0,
        max_segnet_mae_vs_reference_for_fit_gate=64.0,
        segnet_argmax_probe=True,
        segnet_argmax_batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=0.25,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.400001,
        repo_root=tmp_path,
        mlx_scorer_response_probe=True,
        mlx_scorer_response_upstream_dir=tmp_path / "upstream",
        mlx_scorer_response_device_type="cpu",
        mlx_scorer_response_batch_pairs=1,
    )

    assert manifest is not None
    assert calls == [ema_archive, live_archive]
    assert manifest["selected_candidate_kind"] == "live"
    assert manifest["selected_archive_path"] == live_archive.as_posix()
    assert manifest["selected_receiver_replay_local_canonical_score"] == 12.0
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["score_claim"] is False


def test_hinerv_receiver_surface_trace_attaches_real_replay_evidence(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "receiver_report.json"
    selected_report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "report_path": report_path.as_posix(),
        "archive_sha256": "a" * 64,
        "segnet_argmax_probe": {
            "sample_pairs": 2,
            "segnet_argmax_disagreement_rate": 0.25,
        },
        "mlx_scorer_response_probe_required": True,
        "mlx_scorer_response_probe": {
            "fit_gate_passed": True,
            "avg_segnet_dist": 0.01,
            "avg_posenet_dist": 0.02,
        },
        "blockers": [],
    }
    trace = runner_mod._hi_nerv_receiver_surface_trace_from_replay_evidence(
        receiver_replay_archive_selection={
            "selected_receiver_replay_report": selected_report,
        },
        post_export_receiver_cache_quality=None,
        local_cpu_replay_summary={
            "schema": "local_submission_replay.v1",
            "evaluation_passed": True,
            "axis_tag": "[macOS-CPU advisory]",
            "seg_distortion": 0.011,
            "pose_distortion": 0.021,
            "local_score_estimate": 0.123,
        },
    )

    assert trace is not None
    assert trace["schema"] == "nerv_receiver_surface_trace.v1"
    assert trace["archive_parseback_source"] == "receiver_replay_archive_selection"
    assert trace["receiver_surface_parseback_argmax_flipped_pixels"] == pytest.approx(
        0.25 * 2 * runner_mod.HI_NERV_SEGNET_SCORER_FRAME_PIXELS
    )
    assert trace["receiver_surface_parseback_segnet_distortion"] == pytest.approx(0.01)
    assert trace["receiver_surface_parseback_posenet_distortion"] == pytest.approx(0.02)
    assert trace["receiver_surface_inflate_evaluate_passed"] is True
    assert trace["receiver_surface_inflate_axis_tag"] == "[macOS-CPU advisory]"
    assert trace["receiver_surface_inflate_segnet_distortion"] == pytest.approx(0.011)
    assert trace["receiver_surface_inflate_posenet_distortion"] == pytest.approx(0.021)
    assert "receiver_surface_inflated_argmax_flipped_pixels" not in trace

    artifact_path = tmp_path / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_training_artifact.v1",
                "receiver_surface_trace": {
                    "receiver_surface_uint8_changed_pixels": 17.0,
                },
                "substrate_artifact_metadata": {
                    "receiver_surface_trace": {
                        "receiver_surface_fakequant_argmax_flipped_pixels": 3.0,
                    },
                    "score_aware_training": {
                        "receiver_surface_trace": {
                            "receiver_surface_argmax_flipped_pixels": 5.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    artifact_dict = {
        "receiver_surface_trace": {
            "receiver_surface_loss_delta": -0.2,
        },
        "substrate_artifact_metadata": {
            "score_aware_training": {
                "receiver_surface_trace": {
                    "receiver_surface_worst_region_margin_p50_delta": 0.125,
                },
            },
        },
    }

    runner_mod._attach_hi_nerv_runner_receiver_surface_trace(
        artifact_dict=artifact_dict,
        output_dir=tmp_path,
        receiver_surface_trace=trace,
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["receiver_surface_trace"]["receiver_surface_uint8_changed_pixels"] == (17.0)
    assert payload["receiver_surface_trace"]["receiver_surface_fakequant_argmax_flipped_pixels"] == 3.0
    assert payload["receiver_surface_trace"]["receiver_surface_argmax_flipped_pixels"] == 5.0
    assert (
        payload["receiver_surface_trace"]["receiver_surface_parseback_argmax_flipped_pixels"]
        == trace["receiver_surface_parseback_argmax_flipped_pixels"]
    )
    assert artifact_dict["receiver_surface_trace"]["receiver_surface_loss_delta"] == (-0.2)
    assert artifact_dict["receiver_surface_trace"]["receiver_surface_worst_region_margin_p50_delta"] == 0.125


def test_hinerv_receiver_replay_parseback_servo_lift_accepts_replay_surface(
    tmp_path: Path,
) -> None:
    selection = _receiver_replay_selection_manifest(
        old_summary={
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_avg_segnet_dist": 0.030,
            "mlx_scorer_response_avg_posenet_dist": 0.040,
            "segnet_argmax_disagreement_rate": 0.10,
            "segnet_argmax_sample_pairs": 2,
            "fakequant_survival": True,
        },
        new_summary={
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_avg_segnet_dist": 0.020,
            "mlx_scorer_response_avg_posenet_dist": 0.030,
            "segnet_argmax_disagreement_rate": 0.25,
            "segnet_argmax_sample_pairs": 2,
            "receiver_surface_uint8_changed_pixels": 128,
            "fakequant_survival": True,
        },
    )

    lift = runner_mod._write_hi_nerv_receiver_replay_parseback_servo_lift(
        receiver_replay_archive_selection=selection,
        receiver_surface_trace=None,
        local_cpu_replay_summary={"evaluation_passed": True},
        output_dir=tmp_path,
    )

    assert lift is not None
    assert Path(lift["artifact_path"]).is_file()
    servo = lift["parseback_servo_lift"]
    assert servo["servo_lift_accepted"] is True
    assert servo["action_effect"]["action_effect_admitted"] is True
    assert servo["action_effect"]["state_custody"]["source_archive_sha256"] == "a" * 64
    assert servo["action_effect"]["state_custody"]["archive_sha256"] == "b" * 64
    assert servo["action_effect"]["delta_score_total"] < 0.0
    assert lift["score_claim"] is False


def test_hinerv_receiver_replay_parseback_servo_lift_blocks_missing_surfaces() -> None:
    selection = _receiver_replay_selection_manifest(
        old_summary={
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_avg_segnet_dist": 0.030,
            "mlx_scorer_response_avg_posenet_dist": 0.040,
            "segnet_argmax_disagreement_rate": 0.10,
            "segnet_argmax_sample_pairs": 2,
        },
        new_summary={
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_avg_segnet_dist": 0.020,
            "mlx_scorer_response_avg_posenet_dist": 0.030,
            "segnet_argmax_disagreement_rate": 0.25,
            "segnet_argmax_sample_pairs": 2,
        },
    )

    lift = runner_mod._hi_nerv_receiver_replay_parseback_servo_lift(
        receiver_replay_archive_selection=selection,
        receiver_surface_trace=None,
        local_cpu_replay_summary=None,
    )

    assert lift is not None
    servo = lift["parseback_servo_lift"]
    assert servo["servo_lift_accepted"] is False
    assert "action_effect_fakequant_survival_missing" in servo["blockers"]
    assert "servo_lift_uint8_receiver_contact_missing" in servo["blockers"]
    assert "servo_lift_inflate_survival_missing" in servo["blockers"]


def test_hinerv_archive_selection_birth_survival_candidate_row_is_not_gate_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "ema" / "archive.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"unit archive")
    calls: list[dict[str, object]] = []

    import tac.substrates.hi_nerv.birth_survival as survival_mod

    def fake_measure_birth_parseback_survival_from_report(**kwargs):
        calls.append(dict(kwargs))
        return {
            "schema": "hi_nerv_target_region_birth_survival.v1",
            "surface": "parseback_mlx",
            "action_id": "a" * 64,
            "survived": True,
            "region_hard_won_count": 1,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        survival_mod,
        "measure_birth_parseback_survival_from_report",
        fake_measure_birth_parseback_survival_from_report,
    )

    row = runner_mod._write_hi_nerv_runner_birth_parseback_survival_for_archive(
        archive_path=archive,
        output_dir=archive.parent,
        live_birth_payload={"action_id": "a" * 64, "accepted": True},
        scorer_teacher=object(),
        target_labels=np.zeros((1, 2, 2), dtype=np.int32),
        pair_indices=np.array([0], dtype=np.int64),
        candidate_kind="ema",
        canonical_for_launch_gate=False,
    )

    assert calls
    assert row is not None
    assert row["schema"] == "hi_nerv_target_region_birth_survival_candidate.v1"
    assert row["source_schema"] == "hi_nerv_target_region_birth_survival.v1"
    assert row["canonical_launch_gate_schema"] is False
    persisted = json.loads(Path(row["artifact_path"]).read_text(encoding="utf-8"))
    assert persisted["schema"] == "hi_nerv_target_region_birth_survival_candidate.v1"


def test_hinerv_selected_birth_parseback_survival_promotes_only_selected_candidate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "ema" / "archive.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"unit archive")
    candidate_row = {
        "schema": "hi_nerv_target_region_birth_survival_candidate.v1",
        "source_schema": "hi_nerv_target_region_birth_survival.v1",
        "surface": "parseback_mlx",
        "action_id": "b" * 64,
        "survived": True,
        "candidate_kind": "ema",
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    candidate_path = archive.parent / "hi_nerv_birth_parseback_survival_ema.json"
    candidate_path.write_text(json.dumps(candidate_row), encoding="utf-8")
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    artifact_path = training_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps({"substrate_artifact_metadata": {"score_aware_training": {}}}),
        encoding="utf-8",
    )
    artifact_dict = {"substrate_artifact_metadata": {"score_aware_training": {}}}

    promoted = runner_mod._promote_hi_nerv_selected_birth_parseback_survival(
        archive_resolution={
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "candidate_kind": "ema",
        },
        output_dir=training_dir,
        artifact_dict=artifact_dict,
    )

    assert promoted is not None
    assert promoted["schema"] == "hi_nerv_target_region_birth_survival.v1"
    assert promoted["canonical_launch_gate_schema"] is True
    assert promoted["selected_archive_path"] == archive.as_posix()
    assert Path(promoted["artifact_path"]).name == "hi_nerv_selected_birth_parseback_survival.json"
    assert artifact_dict["selected_birth_parseback_survival"]["action_id"] == "b" * 64
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    persisted_training = persisted["substrate_artifact_metadata"]["score_aware_training"]
    assert persisted_training["selected_birth_parseback_survival"]["survived"] is True


def test_hinerv_live_birth_survival_rows_fail_closed_on_missing_inputs(tmp_path: Path) -> None:
    row = runner_mod._write_hi_nerv_runner_live_birth_survival_rows(
        model=object(),
        output_dir=tmp_path,
        live_birth_payload={"action_id": "c" * 64},
        scorer_teacher=None,
        target_labels=np.zeros((1, 2, 2), dtype=np.int32),
        pair_indices=np.array([0], dtype=np.int64),
    )

    assert row["schema"] == "hi_nerv_runner_live_birth_survival_bundle.v1"
    assert row["blockers"] == ["birth_survival_live_scorer_teacher_missing"]
    fake = json.loads((tmp_path / "hi_nerv_birth_fakequant_survival.json").read_text(encoding="utf-8"))
    hyst = json.loads((tmp_path / "hi_nerv_birth_hysteresis.json").read_text(encoding="utf-8"))
    assert fake["schema"] == "hi_nerv_target_region_birth_survival_blocked.v1"
    assert hyst["schema"] == "hi_nerv_target_region_birth_hysteresis_blocked.v1"
    assert fake["score_claim"] is False
    assert hyst["ready_for_exact_eval_dispatch"] is False


def test_hinerv_live_birth_survival_rows_fail_closed_on_rejected_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.hi_nerv.birth_survival as survival_mod

    def fail_measurement(*_args, **_kwargs):
        raise AssertionError("rejected births must not be remeasured as survival")

    monkeypatch.setattr(survival_mod, "measure_birth_survival", fail_measurement)
    monkeypatch.setattr(survival_mod, "measure_birth_hysteresis", fail_measurement)

    row = runner_mod._write_hi_nerv_runner_live_birth_survival_rows(
        model=object(),
        output_dir=tmp_path,
        live_birth_payload={
            "schema": "hi_nerv_target_region_birth_actuator.v1",
            "action_id": "e" * 64,
            "accepted": False,
            "blockers": ["hinerv_target_region_birth_no_accepted_step"],
            "surface": "direct_live_mlx",
        },
        scorer_teacher=object(),
        target_labels=np.zeros((1, 2, 2), dtype=np.int32),
        pair_indices=np.array([0], dtype=np.int64),
    )

    assert row["schema"] == "hi_nerv_runner_live_birth_survival_bundle.v1"
    assert row["birth_accepted"] is False
    assert row["blockers"][0] == "birth_survival_live_birth_not_accepted"
    metadata_row = runner_mod._strip_substrate_metadata_authority_fields(row)
    assert "score_claim" not in metadata_row
    assert "promotion_eligible" not in metadata_row
    assert "ready_for_exact_eval_dispatch" not in metadata_row
    fake = json.loads((tmp_path / "hi_nerv_birth_fakequant_survival.json").read_text(encoding="utf-8"))
    hyst = json.loads((tmp_path / "hi_nerv_birth_hysteresis.json").read_text(encoding="utf-8"))
    assert fake["schema"] == "hi_nerv_target_region_birth_survival_blocked.v1"
    assert hyst["schema"] == "hi_nerv_target_region_birth_hysteresis_blocked.v1"
    assert fake["accepted"] is False
    assert fake["source_blockers"] == ["hinerv_target_region_birth_no_accepted_step"]
    assert hyst["blocker"] == "birth_survival_live_birth_not_accepted"


def test_hinerv_archive_birth_parseback_survival_fails_closed_on_rejected_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\0" * 18)

    import tac.substrates.hi_nerv.birth_survival as survival_mod

    def fail_measurement(*_args, **_kwargs):
        raise AssertionError("rejected births must not be parseback remeasured")

    monkeypatch.setattr(
        survival_mod,
        "measure_birth_parseback_survival_from_report",
        fail_measurement,
    )

    row = runner_mod._write_hi_nerv_runner_birth_parseback_survival_for_archive(
        archive_path=archive,
        output_dir=tmp_path,
        live_birth_payload={
            "schema": "hi_nerv_target_region_birth_actuator.v1",
            "action_id": "f" * 64,
            "accepted": False,
            "blockers": ["hinerv_target_region_birth_no_accepted_step"],
            "surface": "direct_live_mlx",
        },
        scorer_teacher=object(),
        target_labels=np.zeros((1, 2, 2), dtype=np.int32),
        pair_indices=np.array([0], dtype=np.int64),
        candidate_kind="ema",
    )

    assert row is not None
    assert row["schema"] == "hi_nerv_target_region_birth_survival_blocked.v1"
    assert row["blocker"] == "birth_survival_parseback_live_birth_not_accepted"
    assert row["accepted"] is False
    assert row["source_blockers"] == ["hinerv_target_region_birth_no_accepted_step"]
    persisted = json.loads((tmp_path / "hi_nerv_birth_parseback_survival_ema.json").read_text(encoding="utf-8"))
    assert persisted["action_id"] == "f" * 64
    assert persisted["score_claim"] is False


def test_hinerv_live_birth_hysteresis_probe_restores_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")

    class DummyModel:
        def __init__(self) -> None:
            self.weight = mx.array([1.0], dtype=mx.float32)

        def parameters(self):
            return {"weight": self.weight}

        def update(self, params):
            self.weight = params["weight"]

    model = DummyModel()

    import tac.substrates.hi_nerv.birth_survival as survival_mod

    def fake_measure_birth_survival(*_args, **_kwargs):
        return {
            "schema": "hi_nerv_target_region_birth_survival.v1",
            "surface": "fakequant_mlx",
            "action_id": "d" * 64,
            "survived": True,
            "blockers": [],
        }

    def fake_measure_birth_hysteresis(model_arg, **_kwargs):
        model_arg.update({"weight": mx.array([9.0], dtype=mx.float32)})
        mx.eval(model_arg.parameters())
        return {
            "schema": "hi_nerv_target_region_birth_hysteresis.v1",
            "surface": "live_mlx_continued",
            "action_id": "d" * 64,
            "passed": True,
        }

    monkeypatch.setattr(survival_mod, "measure_birth_survival", fake_measure_birth_survival)
    monkeypatch.setattr(survival_mod, "measure_birth_hysteresis", fake_measure_birth_hysteresis)

    row = runner_mod._write_hi_nerv_runner_live_birth_survival_rows(
        model=model,
        output_dir=tmp_path,
        live_birth_payload={"action_id": "d" * 64, "accepted": True},
        scorer_teacher=object(),
        target_labels=np.zeros((1, 2, 2), dtype=np.int32),
        pair_indices=np.array([0], dtype=np.int64),
        target_rgb_0=np.zeros((1, 2, 2, 3), dtype=np.float32),
        target_rgb_1=np.zeros((1, 2, 2, 3), dtype=np.float32),
    )

    assert row["fakequant_survived"] is True
    assert row["hysteresis_passed"] is True
    assert float(np.asarray(model.weight)[0]) == pytest.approx(1.0)
    fake = json.loads((tmp_path / "hi_nerv_birth_fakequant_survival.json").read_text(encoding="utf-8"))
    hyst = json.loads((tmp_path / "hi_nerv_birth_hysteresis.json").read_text(encoding="utf-8"))
    assert fake["producer"] == "hi_nerv_runner_live_birth_survival"
    assert hyst["producer"] == "hi_nerv_runner_live_birth_survival"


def _receiver_replay_selection_manifest(
    *,
    old_summary: dict[str, object],
    new_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "hi_nerv_receiver_replay_archive_selection.v1",
        "manifest_path": "/tmp/receiver_replay_archive_selection.json",
        "selected_archive_path": "/tmp/live.zip",
        "rows": [
            {
                "candidate_kind": "ema",
                "archive_path": "/tmp/ema.zip",
                "archive_sha256": "a" * 64,
                "archive_bytes": 1000,
                "selected_by_training_proxy": True,
                "receiver_replay_summary": old_summary,
            },
            {
                "candidate_kind": "live",
                "archive_path": "/tmp/live.zip",
                "archive_sha256": "b" * 64,
                "archive_bytes": 1001,
                "selected_by_training_proxy": False,
                "receiver_replay_summary": new_summary,
            },
        ],
    }


def test_hinerv_archive_resolution_dedup_preserves_manifest_candidate_kind(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "training"
    selection_dir = training_dir / "ema_archive_selection"
    ema_archive = selection_dir / "ema" / "archive.zip"
    live_archive = selection_dir / "live" / "archive.zip"
    ema_archive.parent.mkdir(parents=True)
    live_archive.parent.mkdir(parents=True)
    ema_archive.write_bytes(b"ema")
    live_archive.write_bytes(b"live")
    manifest_path = selection_dir / "ema_archive_selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "long_training_ema_archive_selection.v1",
                "selected_archive_path": ema_archive.as_posix(),
                "selected_candidate_kind": "ema",
                "rows": [
                    {
                        "candidate_kind": "ema",
                        "status": "exported",
                        "archive_path": ema_archive.as_posix(),
                    },
                    {
                        "candidate_kind": "live",
                        "status": "exported",
                        "archive_path": live_archive.as_posix(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    resolution = runner_mod._hi_nerv_runner_archive_resolution_from_artifact(
        artifact_dict={
            "archive_path": ema_archive.as_posix(),
            "archive_selection_manifest_path": manifest_path.as_posix(),
        },
        training_dir=training_dir,
        repo_root=tmp_path,
    )

    assert resolution["archive_path"] == ema_archive.as_posix()
    by_path = {candidate["archive_path"]: candidate for candidate in resolution["candidates"]}
    assert by_path[ema_archive.as_posix()]["candidate_kind"] == "ema"
    assert by_path[ema_archive.as_posix()]["selected_for_training_artifact"] is True
    assert by_path[live_archive.as_posix()]["candidate_kind"] == "live"


def _assert_compact_runner_rerun_provenance(
    payload: dict[str, object],
    *,
    family: str,
    output_dir: Path,
) -> None:
    provenance = payload["runner_invocation_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["schema"] == "compact_runner_invocation_provenance.v1"
    assert payload["original_argv"] == provenance["original_argv"]
    assert payload["direct_smoke_rerun_argv"] == provenance["same_output_dir_rerun"]["argv"]
    assert "--execute-family" in provenance["original_arg_tokens"]
    assert family in provenance["original_arg_tokens"]
    rerun_argv = provenance["same_output_dir_rerun"]["argv"]
    assert isinstance(rerun_argv, list)
    assert "--overwrite" in rerun_argv
    assert "--output-dir" in rerun_argv
    assert rerun_argv[rerun_argv.index("--output-dir") + 1] == output_dir.as_posix()
    assert provenance["score_claim"] is False
    assert provenance["promotion_eligible"] is False
    assert provenance["rank_or_kill_eligible"] is False
    assert provenance["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    storage_tier = provenance["output_storage_tier"]
    assert isinstance(storage_tier, dict)
    assert storage_tier["large_artifacts_under_output_dir"] is True
    assert storage_tier["score_claim"] is False


def _fake_hinerv_output_head_contrast_init_payload(
    pair_indices,
    *,
    min_output_std: float,
    max_gain: float,
) -> dict[str, object]:
    try:
        pair_index_values = [int(value) for value in pair_indices.tolist()]
    except AttributeError:
        pair_index_values = [int(value) for value in np.asarray(pair_indices).reshape(-1).tolist()]
    return {
        "schema": "hi_nerv_output_head_target_contrast_init.v1",
        "enabled": True,
        "source_pair_indices": pair_index_values,
        "min_output_std": float(min_output_std),
        "max_gain": float(max_gain),
        "runtime_sidecar_bytes": 0,
        "archive_charged_decoder_tensors": [
            "head_rgb_0.weight",
            "head_rgb_1.weight",
        ],
    }


def _fake_hinerv_scorer_domain_bootstrap_payload(
    pair_indices,
    *,
    steps: int,
    learning_rate: float,
    rgb_weight: float,
    yuv6_weight: float,
    temporal_delta_weight: float,
    contrast_floor_weight: float,
    rgb_std_min_ratio: float,
    yuv6_temporal_std_min_ratio: float,
    weight_decay: float,
    grad_clip_max_norm: float | None,
) -> dict[str, object]:
    try:
        pair_index_values = [int(value) for value in pair_indices.tolist()]
    except AttributeError:
        pair_index_values = [int(value) for value in np.asarray(pair_indices).reshape(-1).tolist()]
    return {
        "schema": "hi_nerv_scorer_domain_bootstrap.v1",
        "enabled": True,
        "method": "unit_fake_archive_charged_bootstrap",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "rgb_weight": float(rgb_weight),
        "yuv6_weight": float(yuv6_weight),
        "temporal_delta_weight": float(temporal_delta_weight),
        "contrast_floor_weight": float(contrast_floor_weight),
        "rgb_std_min_ratio": float(rgb_std_min_ratio),
        "yuv6_temporal_std_min_ratio": float(yuv6_temporal_std_min_ratio),
        "weight_decay": float(weight_decay),
        "grad_clip_max_norm": (None if grad_clip_max_norm is None else float(grad_clip_max_norm)),
        "source_pair_indices": pair_index_values,
        "accepted_step_count": int(steps),
        "rejected_step_count": 0,
        "runtime_sidecar_bytes": 0,
        "archive_charged_decoder_tensors": [
            "decoder.bootstrap.weight",
        ],
        "human_visual_fidelity_objective": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _passing_snerv_scorer_tether_smoke_report(*, steps: int = 2) -> dict[str, object]:
    return {
        "schema": "snerv_scorer_tether_smoke.v1",
        "operation": "unit_stub_snerv_pr95_scorer_tether_dual_ascent_smoke",
        "steps": int(steps),
        "passed": True,
        "blockers": [],
        "metric_summary": {
            "step_count": int(steps),
            "final": {
                "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.25,
            },
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_hi_nerv_frontier_gate_counts_direct_live_segnet_as_real_teacher() -> None:
    gate = _validate_hi_nerv_frontier_training_config(
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_class_balanced_ce_weight=1.0,
        pose_distillation_weight=0.0,
        allow_segnet_only_research=True,
        allow_unscored_research_smoke=False,
        score_aware_training_plan={},
    )

    assert gate["launch_allowed"] is True
    assert gate["real_segnet_teacher_attached"] is True
    assert gate["real_posenet_teacher_attached"] is False
    assert gate["direct_live_escape_controls_attached"] is True
    assert gate["segnet_only_research_allowed"] is True
    assert "hi_nerv_real_segnet_teacher_missing" not in gate["blockers"]
    assert "hi_nerv_real_posenet_teacher_missing" in gate["blockers"]
    assert "hi_nerv_direct_live_escape_controls_missing" not in gate["blockers"]


def test_hi_nerv_frontier_gate_counts_direct_live_posenet_as_real_teacher() -> None:
    gate = _validate_hi_nerv_frontier_training_config(
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_class_balanced_ce_weight=1.0,
        pose_distillation_weight=0.0,
        pose_direct_live_distillation_weight=0.5,
        allow_segnet_only_research=False,
        allow_unscored_research_smoke=False,
        score_aware_training_plan={},
    )

    assert gate["launch_allowed"] is True
    assert gate["frontier_targeting"] is True
    assert gate["real_segnet_teacher_attached"] is True
    assert gate["real_posenet_teacher_attached"] is True
    assert gate["pose_direct_live_attached"] is True
    assert gate["pose_direct_live_distillation_weight"] == pytest.approx(0.5)
    assert "hi_nerv_real_posenet_teacher_missing" not in gate["blockers"]


def test_hi_nerv_frontier_gate_counts_target_support_floor_as_escape_control() -> None:
    gate = _validate_hi_nerv_frontier_training_config(
        segnet_distillation_weight=0.0,
        segnet_direct_live_target_mass_floor_weight=0.25,
        segnet_direct_live_target_min_ratio_floor_weight=0.125,
        pose_distillation_weight=1.0,
        allow_segnet_only_research=False,
        allow_unscored_research_smoke=False,
        score_aware_training_plan={},
    )

    assert gate["launch_allowed"] is True
    assert gate["real_segnet_teacher_attached"] is True
    assert gate["direct_live_escape_controls_attached"] is True
    assert gate["direct_live_escape_controls"]["target_mass_floor"] == pytest.approx(0.25)
    assert gate["direct_live_escape_controls"]["target_min_ratio_floor"] == pytest.approx(0.125)
    assert "hi_nerv_direct_live_escape_controls_missing" not in gate["blockers"]


def test_hi_nerv_frontier_gate_rejects_direct_live_without_escape_controls() -> None:
    gate = _validate_hi_nerv_frontier_training_config(
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pose_distillation_weight=1.0,
        allow_segnet_only_research=False,
        allow_unscored_research_smoke=False,
        score_aware_training_plan={},
    )

    assert gate["base_launch_allowed"] is True
    assert gate["launch_allowed"] is False
    assert gate["direct_live_escape_controls_attached"] is False
    assert "hi_nerv_direct_live_escape_controls_missing" in gate["blockers"]


def test_hi_nerv_pr95_prelaunch_allows_explicit_segnet_only_research_probe() -> None:
    plan = {
        "long_campaign_prelaunch_gate": {
            "launch_allowed": False,
            "blockers": ["hi_nerv_real_posenet_teacher_missing"],
        }
    }

    strict = _pr95_long_campaign_prelaunch_blockers(
        plan,
        epochs=8,
        allow_segnet_only_research=False,
        allow_unscored_research_smoke=False,
    )
    waived = _pr95_long_campaign_prelaunch_blockers(
        plan,
        epochs=8,
        allow_segnet_only_research=True,
        allow_unscored_research_smoke=False,
    )

    assert strict == [
        "pr95_long_campaign_prelaunch_gate_failed",
        "hi_nerv_real_posenet_teacher_missing",
    ]
    assert waived == []


@pytest.fixture(autouse=True)
def _stub_snerv_scorer_tether_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_mod,
        "run_snerv_scorer_tether_smoke",
        lambda *, steps=2: _passing_snerv_scorer_tether_smoke_report(steps=int(steps)),
    )


def _fake_snerv_receiver_reconstruction_profile(
    *,
    profile_id: str,
    reference_kind: str,
    mse: float = 1.25,
    max_abs: float = 2.0,
) -> dict[str, object]:
    return {
        "schema": "snerv_receiver_frame_reconstruction_profile.v1",
        "profile_id": profile_id,
        "reference_kind": reference_kind,
        "packet_source": "unit_test_selected_packet",
        "receiver_decoded_selected_packet": True,
        "shape_matches": True,
        "receiver_frames_finite": True,
        "mse_nchw255": float(mse),
        "mae_nchw255": 0.5,
        "rmse_nchw255": float(mse) ** 0.5,
        "max_abs_nchw255": float(max_abs),
        "worst_pairs_by_mse": [
            {
                "rank": 0,
                "pair_idx": 0,
                "source_pair_idx": 0,
                "mse_nchw255": float(mse),
                "mae_nchw255": 0.5,
                "max_abs_nchw255": float(max_abs),
            }
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_synthetic_pr95_archive(path: Path, *, pairs: int = 600) -> Path:
    chunks = []
    for payload in (f'{{"pairs":{pairs}}}'.encode(), b"decoder", b"latents"):
        chunks.append(struct.pack("<I", len(payload)))
        chunks.append(payload)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"".join(chunks))
    return path


def _write_hinerv_receiver_proof(
    path: Path,
    *,
    archive_bytes: int,
    archive_sha256: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_mlx_generated_receiver_proof.v1",
                "runtime_consumption_proof_ready": True,
                "receiver_archive_replay_verified": True,
                "archive_bytes": int(archive_bytes),
                "archive_sha256": archive_sha256,
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_hinerv_live_section_byte_metrics_callback_refreshes_current_hiv1_packet(
    monkeypatch,
) -> None:
    from tac.substrates.hi_nerv import archive as hinerv_archive
    from tac.substrates.hi_nerv import archive_candidate as hinerv_archive_candidate

    pack_calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.export_calls = 0

        def export_state_dict(self) -> dict[str, np.ndarray]:
            self.export_calls += 1
            return {"decoder.weight": np.asarray([self.export_calls], dtype=np.float32)}

    def fake_pack_archive_from_exported_state_dict(**kwargs):
        pack_calls.append(dict(kwargs))
        return f"hiv1-packet-{len(pack_calls)}".encode("ascii")

    def fake_build_archive_section_telemetry(packet: bytes) -> dict[str, object]:
        packet_index = int(packet.decode("ascii").rsplit("-", 1)[-1])
        return {
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "inner_payload_bytes": 300 + packet_index,
            "sections": [
                {
                    "name": "decoder_state",
                    "role": "decoder",
                    "bytes": 200 + packet_index,
                },
                {
                    "name": "latents_coarse",
                    "role": "latent",
                    "bytes": 40 + packet_index,
                },
            ],
        }

    monkeypatch.setattr(
        hinerv_archive_candidate,
        "pack_archive_from_exported_state_dict",
        fake_pack_archive_from_exported_state_dict,
    )
    monkeypatch.setattr(
        hinerv_archive,
        "build_archive_section_telemetry",
        fake_build_archive_section_telemetry,
    )
    callback, metadata = runner_mod._build_hi_nerv_live_train_time_section_byte_metrics_callback(
        cfg=SimpleNamespace(num_pairs=2),
        decoder_codec="int4_mixed",
        latent_codec="int16_brotli_q11",
        decoder_weight_waterfill_plan={"schema": "unit_waterfill"},
        train_time_section_byte_control={
            "metrics_payload": {
                "archive_bytes": 999,
                "section_bytes": {"decoder_state": 500},
            }
        },
        optimizer_controls={"section_byte_refresh_every_steps": 2},
    )
    assert callback is not None

    model = FakeModel()
    first = callback(model, None, {})
    second = callback(model, None, {})
    third = callback(model, None, {})

    assert first is not None
    assert second is not None
    assert third is not None
    assert first["source"] == "live_current_hiv1_packet"
    assert first["archive_bytes"] == 301
    assert first["section_bytes"] == {
        "decoder_state": 201,
        "latents_coarse": 41,
    }
    assert first["live_profile"] == {
        "refresh_call": 1,
        "refresh_every_steps": 2,
    }
    assert [key for key, value in first.items() if isinstance(value, (int, float, bool))] == ["archive_bytes"]
    assert second["live_profile"]["cache_hit"] is True
    assert second["live_profile"]["callback_call"] == 2
    assert third["archive_bytes"] == 302
    assert model.export_calls == 2
    assert len(pack_calls) == 2
    assert pack_calls[0]["decoder_codec"] == "int4_mixed"
    assert pack_calls[0]["latent_codec"] == "int16_brotli_q11"
    assert pack_calls[0]["decoder_weight_waterfill_plan"] == {"schema": "unit_waterfill"}
    assert metadata["successful_refresh_count"] == 2
    assert metadata["fallback_count"] == 0
    assert metadata["last_live_archive_bytes"] == 302
    assert metadata["last_live_section_bytes"] == {
        "decoder_state": 202,
        "latents_coarse": 42,
    }


def test_hinerv_runner_receiver_cache_quality_uses_explicit_reference_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    reference = tmp_path / "reference_cache"
    reference.mkdir()
    captured: dict[str, object] = {}

    from tac.substrates.hi_nerv import receiver_cache_quality

    def fake_write_hi_nerv_receiver_cache_quality_report(**kwargs):
        captured.update(kwargs)
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": (tmp_path / "report.json").as_posix(),
            "archive_path": Path(kwargs["archive_zip_path"]).as_posix(),
            "archive_sha256": "a" * 64,
            "candidate_cache_dir": (tmp_path / "candidate_cache").as_posix(),
            "reference_cache_dir": Path(kwargs["reference_cache_dir"]).as_posix(),
            "quality_gate_path": (tmp_path / "gate.json").as_posix(),
            "quality_gate_passed": True,
            "quality_gate": {
                "verdict": "CACHE_QUALITY_GATE_PASSED",
                "stats": {
                    "candidate_segnet_last_rgb": {"std": 12.0},
                    "candidate_posenet_yuv6_pair": {"std": 3.0},
                },
                "distance_to_reference": {"segnet_last_rgb_mae": 1.5},
            },
            "distortion_crux_probe_path": (tmp_path / "crux.json").as_posix(),
            "distortion_crux_probe": {
                "schema": "nerv_scorer_input_distortion_crux.v1",
                "fit_gate_passed": True,
                "aggregate": {
                    "dominant_domain_top_k": "posenet_yuv6_pair",
                    "posenet_yuv6_pair_mae_255": {"mean": 2.25},
                },
                "hard_pair_coverage": {
                    "schema": "nerv_hard_pair_coverage_evidence.v1",
                    "score_axis_hard_pair_coverage": False,
                    "coverage_valid_for_distortion": False,
                    "prioritized_pair_indices": [2, 0],
                    "hard_pair_count": 2,
                },
            },
            "mlx_scorer_response_probe_path": (tmp_path / "mlx_scorer_response_probe.json").as_posix(),
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_probe": {
                "schema": "hi_nerv_receiver_cache_mlx_scorer_response_probe.v1",
                "fit_gate_passed": True,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.02,
                "blockers": ["hi_nerv_receiver_cache_mlx_scorer_response_probe_is_false_authority"],
            },
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        }

    monkeypatch.setattr(
        receiver_cache_quality,
        "write_hi_nerv_receiver_cache_quality_report",
        fake_write_hi_nerv_receiver_cache_quality_report,
    )

    report = runner_mod._write_hi_nerv_runner_post_export_receiver_cache_quality(
        requested=True,
        archive_zip_path=archive,
        source_video_path=tmp_path / "unused_source.mkv",
        output_dir=tmp_path / "training",
        reference_cache_dir=reference,
        max_pairs=3,
        batch_pairs=2,
        min_segnet_std=1.25,
        min_segnet_dynamic_range=7.5,
        max_segnet_mae_vs_reference_for_fit_gate=22.0,
        segnet_argmax_probe=True,
        segnet_argmax_batch_frames=3,
        max_segnet_argmax_disagreement_for_fit_gate=0.125,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.625,
        repo_root=REPO_ROOT,
    )

    assert Path(captured["archive_zip_path"]) == archive
    assert Path(captured["reference_cache_dir"]) == reference
    assert captured["max_pairs"] == 3
    assert captured["batch_pairs"] == 2
    assert captured["sample_pairs"] == 3
    assert captured["min_segnet_std"] == pytest.approx(1.25)
    assert captured["min_segnet_dynamic_range"] == pytest.approx(7.5)
    assert captured["max_segnet_mae_vs_reference_for_fit_gate"] == pytest.approx(22.0)
    assert Path(captured["segnet_argmax_probe_upstream_dir"]) == REPO_ROOT / "upstream"
    assert captured["segnet_argmax_probe_batch_frames"] == 3
    assert captured["max_segnet_argmax_disagreement_for_fit_gate"] == pytest.approx(0.125)
    assert captured["min_segnet_argmax_occupied_class_fraction_for_fit_gate"] == pytest.approx(0.625)
    assert captured["require_mlx_scorer_response_probe"] is True
    assert Path(captured["mlx_scorer_response_upstream_dir"]) == REPO_ROOT / "upstream"
    assert captured["mlx_scorer_response_device_type"] == "cpu"
    assert captured["mlx_scorer_response_batch_pairs"] == 1
    assert captured["max_mlx_scorer_response_posenet_dist_for_fit_gate"] == pytest.approx(0.01)
    assert captured["max_mlx_scorer_response_segnet_dist_for_fit_gate"] == pytest.approx(0.25)
    summary = runner_mod._hi_nerv_receiver_cache_quality_summary(report)
    assert summary["quality_gate_passed"] is True
    assert summary["candidate_posenet_yuv6_pair_stats"] == {"std": 3.0}
    assert summary["distortion_crux_probe_passed"] is True
    assert summary["distortion_crux_dominant_domain"] == "posenet_yuv6_pair"
    assert summary["mlx_scorer_response_probe_required"] is True
    assert summary["mlx_scorer_response_probe_passed"] is True
    assert summary["mlx_scorer_response_avg_posenet_dist"] == pytest.approx(0.001)
    assert summary["mlx_scorer_response_avg_segnet_dist"] == pytest.approx(0.02)
    assert summary["hard_pair_coverage"]["prioritized_pair_indices"] == [2, 0]
    assert runner_mod._hi_nerv_receiver_cache_quality_routable_hard_pair_coverage(report) is None


def test_hinerv_runner_receiver_cache_quality_exposes_routable_crux_pairs() -> None:
    report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "distortion_crux_probe": {
            "schema": "nerv_scorer_input_distortion_crux.v1",
            "fit_gate_passed": False,
            "hard_pair_coverage": {
                "schema": "nerv_hard_pair_coverage_evidence.v1",
                "score_axis_hard_pair_coverage": True,
                "coverage_valid_for_distortion": True,
                "representative_distortion_evidence": True,
                "prioritized_pair_indices": [17, 4, 17],
                "hard_pair_count": 2,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
        "blockers": ["nerv_distortion_crux_posenet_yuv6_pair_mae_too_high"],
    }

    coverage = runner_mod._hi_nerv_receiver_cache_quality_routable_hard_pair_coverage(report)

    assert coverage is not None
    assert coverage["prioritized_pair_indices"] == [17, 4, 17]
    assert coverage["score_claim"] is False


def test_hinerv_runner_receiver_cache_quality_blocks_invalid_mlx_response_batch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    reference = tmp_path / "reference_cache"
    reference.mkdir()

    report = runner_mod._write_hi_nerv_runner_post_export_receiver_cache_quality(
        requested=True,
        archive_zip_path=archive,
        source_video_path=tmp_path / "unused_source.mkv",
        output_dir=tmp_path / "training",
        reference_cache_dir=reference,
        max_pairs=1,
        batch_pairs=1,
        min_segnet_std=1.0,
        min_segnet_dynamic_range=16.0,
        max_segnet_mae_vs_reference_for_fit_gate=64.0,
        segnet_argmax_probe=True,
        segnet_argmax_batch_frames=4,
        max_segnet_argmax_disagreement_for_fit_gate=0.25,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.400001,
        repo_root=REPO_ROOT,
        mlx_scorer_response_batch_pairs=0,
    )

    assert report["quality_gate_passed"] is False
    assert "hi_nerv_receiver_cache_quality_mlx_scorer_response_batch_pairs_invalid" in report["blockers"]


def test_hinerv_runner_receiver_cache_quality_builds_source_reference_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    source = tmp_path / "source.mkv"
    source.write_bytes(b"unit source video")
    captured_reference: dict[str, object] = {}
    captured_quality: dict[str, object] = {}

    from tac.local_acceleration import mlx_preprocess
    from tac.substrates.hi_nerv import receiver_cache_quality

    def fake_write_scorer_input_cache_from_video_file(
        video_path,
        output_dir,
        *,
        max_pairs,
        batch_pairs,
        **_kwargs,
    ):
        captured_reference.update(
            {
                "video_path": Path(video_path),
                "output_dir": Path(output_dir),
                "max_pairs": int(max_pairs),
                "batch_pairs": int(batch_pairs),
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "manifest.json").write_text("{}", encoding="utf-8")
        return {"pair_count": int(max_pairs)}

    def fake_write_hi_nerv_receiver_cache_quality_report(**kwargs):
        captured_quality.update(kwargs)
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": (tmp_path / "report.json").as_posix(),
            "archive_path": Path(kwargs["archive_zip_path"]).as_posix(),
            "archive_sha256": "b" * 64,
            "candidate_cache_dir": (tmp_path / "candidate_cache").as_posix(),
            "reference_cache_dir": Path(kwargs["reference_cache_dir"]).as_posix(),
            "quality_gate_path": (tmp_path / "gate.json").as_posix(),
            "quality_gate_passed": False,
            "quality_gate": {
                "verdict": "CACHE_QUALITY_GATE_FAILED",
                "stats": {
                    "candidate_segnet_last_rgb": {"std": 0.1},
                    "candidate_posenet_yuv6_pair": {"std": 0.0},
                },
                "distance_to_reference": {"segnet_last_rgb_mae": 99.0},
            },
            "blockers": [
                "hi_nerv_receiver_cache_quality_is_false_authority",
                "candidate_posenet_yuv6_pair_low_dynamic_range",
            ],
        }

    monkeypatch.setattr(
        mlx_preprocess,
        "write_scorer_input_cache_from_video_file",
        fake_write_scorer_input_cache_from_video_file,
    )
    monkeypatch.setattr(
        receiver_cache_quality,
        "write_hi_nerv_receiver_cache_quality_report",
        fake_write_hi_nerv_receiver_cache_quality_report,
    )

    report = runner_mod._write_hi_nerv_runner_post_export_receiver_cache_quality(
        requested=True,
        archive_zip_path=archive,
        source_video_path=source,
        output_dir=tmp_path / "training",
        reference_cache_dir=None,
        max_pairs=4,
        batch_pairs=2,
        min_segnet_std=1.0,
        min_segnet_dynamic_range=16.0,
        max_segnet_mae_vs_reference_for_fit_gate=64.0,
        segnet_argmax_probe=False,
        segnet_argmax_batch_frames=4,
        max_segnet_argmax_disagreement_for_fit_gate=0.25,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.400001,
        repo_root=REPO_ROOT,
    )

    reference_dir = tmp_path / "training" / "post_export_receiver_cache_quality" / "source_video_reference_cache"
    assert captured_reference["video_path"] == source
    assert captured_reference["output_dir"] == reference_dir
    assert captured_reference["max_pairs"] == 4
    assert captured_reference["batch_pairs"] == 2
    assert Path(captured_quality["reference_cache_dir"]) == reference_dir
    assert captured_quality["segnet_argmax_probe_upstream_dir"] is None
    assert captured_quality["require_mlx_scorer_response_probe"] is True
    assert Path(captured_quality["mlx_scorer_response_upstream_dir"]) == (REPO_ROOT / "upstream")
    assert captured_quality["mlx_scorer_response_batch_pairs"] == 1
    assert report["quality_gate_passed"] is False
    assert "candidate_posenet_yuv6_pair_low_dynamic_range" in report["blockers"]


def test_hinerv_runner_receiver_cache_quality_attaches_to_training_artifact(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    artifact_path = training_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps({"substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}}}),
        encoding="utf-8",
    )
    artifact_dict = {"substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}}}
    report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "report_path": (tmp_path / "report.json").as_posix(),
        "archive_path": (tmp_path / "archive.zip").as_posix(),
        "quality_gate_passed": False,
        "quality_gate": {"verdict": "CACHE_QUALITY_GATE_FAILED", "stats": {}},
        "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
    }

    runner_mod._attach_hi_nerv_post_export_receiver_cache_quality(
        artifact_dict=artifact_dict,
        output_dir=training_dir,
        report=report,
    )

    metadata = artifact_dict["substrate_artifact_metadata"]
    assert metadata["post_export_receiver_cache_quality"]["quality_gate_passed"] is False
    assert (
        metadata["score_aware_training"]["post_export_receiver_cache_quality"]["quality_gate_verdict"]
        == "CACHE_QUALITY_GATE_FAILED"
    )
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    persisted_metadata = persisted["substrate_artifact_metadata"]
    assert persisted_metadata["post_export_receiver_cache_quality"]["quality_gate_passed"] is False


def test_hinerv_runner_crux_trace_attaches_to_training_artifact(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "hi_nerv_mlx_training"
    training_dir.mkdir()
    artifact_path = training_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "archive_bytes": 178_000,
                "receiver_surface_trace": {
                    "schema": "nerv_receiver_surface_trace.v1",
                    "receiver_surface_trace_present": True,
                    "receiver_surface_uint8_changed_pixels": 3,
                    "receiver_surface_argmax_flipped_pixels": 2,
                    "receiver_surface_target_hard_won_count": 2,
                    "receiver_surface_net_target_support_delta": 2,
                    "receiver_surface_fakequant_argmax_flipped_pixels": 2,
                    "receiver_surface_parseback_argmax_flipped_pixels": 2,
                    "receiver_surface_inflated_argmax_flipped_pixels": 2,
                    "receiver_surface_posenet_input_delta_linf": 0.03,
                    "receiver_surface_pose_output_delta": 0.01,
                    "receiver_surface_fakequant_pose_output_delta": 0.01,
                    "receiver_surface_parseback_pose_output_delta": 0.01,
                    "receiver_surface_inflated_pose_output_delta": 0.01,
                },
                "per_epoch_metrics": [
                    {
                        "loss_components": {
                            "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 12.5,
                            "loss_part_pose_direct_live_raw_mse": 0.0025,
                        }
                    }
                ],
                "substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}},
            }
        ),
        encoding="utf-8",
    )
    artifact_dict = {
        "receiver_surface_trace": {
            "schema": "nerv_receiver_surface_trace.v1",
            "receiver_surface_trace_present": True,
            "receiver_surface_uint8_changed_pixels": 3,
            "receiver_surface_argmax_flipped_pixels": 2,
            "receiver_surface_target_hard_won_count": 2,
            "receiver_surface_net_target_support_delta": 2,
            "receiver_surface_fakequant_argmax_flipped_pixels": 2,
            "receiver_surface_parseback_argmax_flipped_pixels": 2,
            "receiver_surface_inflated_argmax_flipped_pixels": 2,
            "receiver_surface_posenet_input_delta_linf": 0.03,
            "receiver_surface_pose_output_delta": 0.01,
            "receiver_surface_fakequant_pose_output_delta": 0.01,
            "receiver_surface_parseback_pose_output_delta": 0.01,
            "receiver_surface_inflated_pose_output_delta": 0.01,
        },
        "substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}},
    }

    report = runner_mod._write_hi_nerv_runner_crux_trace(
        artifact_dict=artifact_dict,
        output_dir=training_dir,
    )

    trace_path = training_dir / "nerv_crux_trace_rows.json"
    assert report["written"] is True
    assert report["path"] == trace_path.as_posix()
    assert report["blockers"] == []
    assert trace_path.is_file()
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    trace_metadata = persisted["substrate_artifact_metadata"]["nerv_crux_trace"]
    assert trace_metadata["path"] == trace_path.as_posix()
    assert trace_metadata["row_count"] > 0
    assert (
        artifact_dict["substrate_artifact_metadata"]["score_aware_training"]["nerv_crux_trace"]["path"]
        == trace_path.as_posix()
    )


def test_snerv_runner_crux_trace_attaches_to_score_aware_long_training(
    tmp_path: Path,
) -> None:
    long_training_dir = tmp_path / "snerv_score_aware_long_training" / "long_training"
    long_training_dir.mkdir(parents=True)
    artifact_path = long_training_dir / "training_artifact.json"
    telemetry_path = long_training_dir / "telemetry.jsonl"
    telemetry_path.write_text("", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "archive_bytes": 99_000,
                "receiver_surface_trace": {
                    "schema": "nerv_receiver_surface_trace.v1",
                    "receiver_surface_trace_present": True,
                    "receiver_surface_uint8_changed_pixels": 4,
                    "receiver_surface_argmax_flipped_pixels": 2,
                    "receiver_surface_target_hard_won_count": 2,
                    "receiver_surface_net_target_support_delta": 2,
                    "receiver_surface_fakequant_argmax_flipped_pixels": 2,
                    "receiver_surface_parseback_argmax_flipped_pixels": 2,
                    "receiver_surface_inflated_argmax_flipped_pixels": 2,
                    "receiver_surface_posenet_input_delta_linf": 0.02,
                    "receiver_surface_pose_output_delta": 0.01,
                    "receiver_surface_fakequant_pose_output_delta": 0.01,
                    "receiver_surface_parseback_pose_output_delta": 0.01,
                    "receiver_surface_inflated_pose_output_delta": 0.01,
                },
                "per_epoch_metrics": [
                    {
                        "loss_components": {
                            "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 7.25,
                            "loss_part_pose_direct_live_raw_mse": 0.0036,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = {}
    score_aware_long_training = {"telemetry_path": telemetry_path.as_posix()}

    report = runner_mod._write_snerv_runner_crux_trace(
        artifact=artifact,
        score_aware_long_training=score_aware_long_training,
    )

    trace_path = long_training_dir / "nerv_crux_trace_rows.json"
    assert report["written"] is True
    assert report["path"] == trace_path.as_posix()
    assert report["blockers"] == []
    assert trace_path.is_file()
    assert score_aware_long_training["nerv_crux_trace"]["path"] == trace_path.as_posix()
    assert artifact["score_aware_long_training"]["nerv_crux_trace"]["row_count"] > 0


def test_hinerv_runner_archive_resolution_uses_emitted_overcap_ema_archives(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "hi_nerv_mlx_training"
    selection_dir = training_dir / "ema_archive_selection"
    (selection_dir / "live").mkdir(parents=True, exist_ok=True)
    (selection_dir / "ema").mkdir(parents=True, exist_ok=True)
    live_archive = _write_synthetic_pr95_archive(
        selection_dir / "live" / "archive.zip",
        pairs=2,
    )
    ema_archive = _write_synthetic_pr95_archive(
        selection_dir / "ema" / "archive.zip",
        pairs=2,
    )
    # Make live strictly cheaper so the diagnostic resolver's byte-first choice
    # is deterministic when both selector rows failed the same hard cap.
    with zipfile.ZipFile(ema_archive, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("padding.bin", b"x" * 16)
    manifest = selection_dir / "ema_archive_selection.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "long_training_ema_archive_selection.v1",
                "selected_archive_path": None,
                "selected_archive_sha256": None,
                "selected_archive_bytes": None,
                "rows": [
                    {
                        "schema": "long_training_archive_selection_candidate.v1",
                        "candidate_kind": "live",
                        "status": "failed",
                        "failure": ("ValueError:HiNeRV archive exceeds hard_byte_ceiling: 291749 > 285000"),
                    },
                    {
                        "schema": "long_training_archive_selection_candidate.v1",
                        "candidate_kind": "ema",
                        "status": "failed",
                        "failure": ("ValueError:HiNeRV archive exceeds hard_byte_ceiling: 291798 > 285000"),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = {
        "archive_path": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "archive_selection_manifest_path": manifest.as_posix(),
        "substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}},
    }
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "training_artifact.json").write_text(
        json.dumps(artifact, sort_keys=True),
        encoding="utf-8",
    )

    resolution = runner_mod._hi_nerv_runner_archive_resolution_from_artifact(
        artifact_dict=artifact,
        training_dir=training_dir,
        repo_root=REPO_ROOT,
    )
    runner_mod._attach_hi_nerv_runner_archive_resolution(
        artifact_dict=artifact,
        output_dir=training_dir,
        archive_resolution=resolution,
    )

    assert resolution["archive_path"] == live_archive.as_posix()
    assert resolution["archive_bytes"] == live_archive.stat().st_size
    assert resolution["archive_sha256"] == runner_mod._sha256_file(live_archive)
    assert resolution["diagnostic_only"] is True
    assert resolution["byte_cap_rejected"] is True
    assert resolution["candidate_kind"] == "live"
    assert "hi_nerv_archive_selection_no_selected_archive" in resolution["blockers"]
    assert "hi_nerv_archive_selection_rejected_over_hard_byte_ceiling" in resolution["blockers"]
    persisted = json.loads((training_dir / "training_artifact.json").read_text(encoding="utf-8"))
    persisted_resolution = persisted["substrate_artifact_metadata"]["archive_resolution"]
    assert persisted_resolution["archive_path"] == live_archive.as_posix()
    assert persisted_resolution["diagnostic_only"] is True


def test_hinerv_runner_short_scorer_smoke_readiness_attaches_to_training_artifact(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    artifact_path = training_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps({"substrate_artifact_metadata": {"score_aware_training": {"schema": "unit"}}}),
        encoding="utf-8",
    )
    artifact_dict = {
        "per_epoch_metrics": [
            {
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.12,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "loss_part_segnet_direct_live_target_mass_floor_loss": 0.05,
                    "loss_part_segnet_direct_live_target_min_ratio_floor_loss": 0.04,
                    "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 0.0,
                    "loss_part_scorer_input_contrast_floor": 0.01,
                    "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
                    "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
                    "loss_part_scorer_input_shape_tether": 0.02,
                    "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
                    "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
                    "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
                    "loss_part_posenet_temporal_signal_floor": 0.03,
                    "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
                    "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
                    "loss_part_pose_score_term": 0.2,
                    "loss_part_pose_distill_raw_mse": 0.004,
                    "loss_part_pose_score_marginal_wrt_raw_mse": 25.0,
                    "loss_part_pose_distill_score_marginal_wrt_raw_mse": 25.0,
                    "dual_ascent_active": 1.0,
                    "dual_ascent_constraint_count": 4.0,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_distill": 0.12,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_distill": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_distill": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_distill": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_distill": 0.1,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_argmax_disagreement": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_argmax_disagreement": 0.1,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.05,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_mass_floor": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_mass_floor": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_mass_floor": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_mass_floor": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_target_mass_floor": 0.1,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.1,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.04,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_min_ratio_floor": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_min_ratio_floor": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.1,
                    "dual_ascent_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.03,
                    "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 1.0,
                    "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.4,
                    "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.1,
                }
            }
        ],
        "substrate_artifact_metadata": {
            "score_aware_training": {
                "schema": "unit",
                "scorer_domain_bootstrap": {
                    "schema": "hi_nerv_scorer_domain_bootstrap.v1",
                    "enabled": True,
                    "accepted_step_count": 1.0,
                    "segnet_hard_birth_bootstrap": {
                        "schema": "hi_nerv_scorer_domain_bootstrap_live_segnet_hard_birth.v1",
                        "enabled": True,
                    },
                    "metrics_after": {
                        "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio": 0.25,
                        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass": 0.0,
                    },
                },
            }
        },
    }
    post_export_quality = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "report_path": (tmp_path / "quality.json").as_posix(),
        "archive_path": (tmp_path / "archive.zip").as_posix(),
        "archive_sha256": "a" * 64,
        "archive_bytes": 12345,
        "zip_member": "x",
        "candidate_cache_dir": (tmp_path / "cache").as_posix(),
        "candidate_cache_manifest_path": (tmp_path / "cache" / "manifest.json").as_posix(),
        "candidate_cache_manifest_sha256": "c" * 64,
        "cache_manifest_summary": {
            "source_kind": "hi_nerv_direct_receiver_render",
            "raw_sha256": "b" * 64,
            "pair_count": 1,
            "array_sha256": {},
        },
        "direct_receiver_cache_report": {
            "schema": "hi_nerv_direct_receiver_cache_report.v1",
            "source_family": "hi_nerv",
            "archive_sha256": "a" * 64,
            "zip_member": "x",
            "archive_magic": "HIV1",
            "cached_pair_count": 1,
            "direct_render_raw_sha256": "b" * 64,
            "identity_audit_sha256": "d" * 64,
            "candidate_cache_identity_mode": ("hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"),
        },
        "quality_gate_path": (tmp_path / "gate.json").as_posix(),
        "quality_gate_passed": True,
        "quality_gate": {
            "verdict": "CACHE_QUALITY_GATE_PASSED",
            "stats": {
                "candidate_segnet_last_rgb": {"std": 12.0},
                "candidate_posenet_yuv6_pair": {"std": 3.0},
                "candidate_posenet_yuv6_temporal_signal": {
                    "mean_abs": 1.0,
                    "std": 1.0,
                },
            },
        },
        "scorer_input_distribution_gate": {
            "schema": "hi_nerv_receiver_cache_scorer_input_distribution_gate.v1",
            "fit_gate_passed": True,
            "blockers": [],
        },
        "segnet_argmax_probe_path": (tmp_path / "argmax.json").as_posix(),
        "segnet_argmax_probe": {
            "fit_gate_passed": True,
            "segnet_argmax_disagreement_rate": 0.02,
            "candidate_occupied_class_fraction": 0.8,
            "candidate_target_class_coverage_fraction": 0.8,
            "candidate_target_class_min_ratio": 0.25,
            "candidate_target_material_class_covered_count": 4.0,
            "target_material_class_count": 5.0,
            "reference_occupied_class_fraction": 0.9,
            "blockers": ["hi_nerv_receiver_cache_segnet_argmax_probe_is_false_authority"],
        },
        "mlx_scorer_response_probe_path": (tmp_path / "mlx_scorer_response_probe.json").as_posix(),
        "mlx_scorer_response_probe_required": True,
        "mlx_scorer_response_probe": {
            "schema": "hi_nerv_receiver_cache_mlx_scorer_response_probe.v1",
            "fit_gate_passed": True,
            "avg_posenet_dist": 0.002,
            "avg_segnet_dist": 0.03,
            "blockers": ["hi_nerv_receiver_cache_mlx_scorer_response_probe_is_false_authority"],
        },
        "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
    }

    report = runner_mod._write_hi_nerv_runner_short_scorer_smoke_readiness(
        output_dir=training_dir,
        artifact_dict=artifact_dict,
        train_time_controls={
            "segnet_direct_live_distillation_weight": 0.4,
            "segnet_direct_live_target_mass_floor_weight": 0.4,
            "segnet_direct_live_target_min_ratio_floor_weight": 0.4,
            "scorer_input_contrast_floor_weight": 0.5,
            "scorer_input_contrast_floor_segnet_min_std_ratio": 0.6,
            "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio": 0.6,
            "scorer_input_shape_tether_weight": 0.25,
            "posenet_temporal_signal_floor_weight": 0.25,
        },
        post_export_quality=post_export_quality,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_unscored_research_smoke=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
        require_section_byte_dual_ascent=False,
        require_pose_direct_live_distillation=False,
        decoder_weight_waterfill_plan_metadata=None,
        output_head_target_bias_init_metadata={
            "schema": "hi_nerv_output_head_target_bias_init.v1",
            "enabled": True,
            "contrast_init": {
                "schema": "hi_nerv_output_head_target_contrast_init.v1",
                "enabled": True,
            },
        },
    )

    assert report["short_scorer_teacher_smoke_ready"] is True
    assert report["ready_for_long_run"] is True
    assert report["actionable_blockers"] == []
    assert report["scorer_domain_hard_birth_bootstrap_gate"]["after_min_ratio_cleared"] is True
    assert Path(report["report_path"]).is_file()
    metadata = artifact_dict["substrate_artifact_metadata"]
    summary = metadata["short_scorer_teacher_smoke_readiness"]
    admission = metadata["short_scorer_teacher_smoke_long_run_admission"]
    assert summary["ready_for_long_run"] is True
    assert admission["long_run_admission_passed"] is True
    assert admission["admission_blockers"] == []
    assert (
        metadata["score_aware_training"]["short_scorer_teacher_smoke_readiness"]["short_scorer_teacher_smoke_ready"]
        is True
    )
    assert (
        metadata["score_aware_training"]["short_scorer_teacher_smoke_long_run_admission"]["long_run_admission_passed"]
        is True
    )
    assert summary["scorer_input_shape_tether_gate"]["enabled"] is True
    assert summary["posenet_temporal_signal_floor_gate"]["enabled"] is True
    assert summary["direct_live_segnet_gate"]["subcontrol_weights"][
        "segnet_direct_live_target_mass_floor_weight"
    ] == pytest.approx(0.4)
    assert summary["direct_live_segnet_gate"]["subcontrol_weights"][
        "segnet_direct_live_target_min_ratio_floor_weight"
    ] == pytest.approx(0.4)
    assert summary["direct_live_segnet_gate"]["metrics"][
        "loss_part_segnet_direct_live_target_mass_floor_loss"
    ] == pytest.approx(0.05)
    assert summary["direct_live_segnet_gate"]["metrics"][
        "loss_part_segnet_direct_live_target_min_ratio_floor_loss"
    ] == pytest.approx(0.04)
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    persisted_summary = persisted["substrate_artifact_metadata"]["short_scorer_teacher_smoke_readiness"]
    persisted_admission = persisted["substrate_artifact_metadata"]["short_scorer_teacher_smoke_long_run_admission"]
    assert persisted_summary["ready_for_long_run"] is True
    assert persisted_admission["long_run_admission_passed"] is True


def test_hinerv_short_scorer_readiness_reloads_durable_final_metrics(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    loss_components = {
        "loss_part_segnet_direct_live_distill": 0.12,
        "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
        "dual_ascent_active": 1.0,
        "dual_ascent_constraint_count": 2.0,
    }
    for constraint in (
        "hi_nerv_segnet_direct_live_distill",
        "hi_nerv_segnet_direct_live_argmax_disagreement",
    ):
        loss_components.update(
            {
                f"dual_ascent_metric__{constraint}": 0.12,
                f"dual_ascent_missing_metric__{constraint}": 0.0,
                f"dual_ascent_lambda__{constraint}": 0.04,
                f"dual_ascent_update_count__{constraint}": 1.0,
                f"dual_ascent_weight_applied__{constraint}": 1.0,
                f"dual_ascent_effective_loss_weight__{constraint}": 0.5,
                f"dual_ascent_violation__{constraint}": 0.1,
            }
        )
    (training_dir / "training_artifact.json").write_text(
        json.dumps(
            {
                "per_epoch_metrics": [{"loss_components": loss_components}],
                "substrate_artifact_metadata": {
                    "score_aware_training": {"schema": "unit"},
                    "substrate_supplied_score_aware_training": {
                        "decoder_fake_quant_forward": {
                            "per_tensor_waterfill_enabled": True,
                            "per_tensor_waterfill_group_count": 2,
                            "per_tensor_waterfill_bits_by_name": {
                                "head_rgb_0.bias": 8,
                                "convnext_blocks.0.norm.bias": 4,
                            },
                        }
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = runner_mod._write_hi_nerv_runner_short_scorer_smoke_readiness(
        output_dir=training_dir,
        artifact_dict={
            "per_epoch_metrics": [],
            "substrate_artifact_metadata": {"score_aware_training": {"schema": "stale_in_memory_summary"}},
        },
        train_time_controls={
            "segnet_direct_live_distillation_weight": 0.4,
            "pose_direct_live_distillation_weight": 0.0,
            "scorer_input_contrast_floor_weight": 0.0,
            "scorer_input_shape_tether_weight": 0.0,
            "posenet_temporal_signal_floor_weight": 0.0,
        },
        post_export_quality=None,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_unscored_research_smoke=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
        require_section_byte_dual_ascent=False,
        require_pose_direct_live_distillation=False,
        decoder_weight_waterfill_plan_metadata={
            "attached": True,
            "row_count": 2,
        },
    )

    assert report["final_loss_components_present"] is True
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_telemetry" not in report["actionable_blockers"]
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" not in report["actionable_blockers"]
    assert "hi_nerv_short_smoke_direct_live_dual_ascent_weight_not_applied" not in report["actionable_blockers"]
    assert "hi_nerv_short_smoke_decoder_waterfill_fake_quant_not_bound" not in report["actionable_blockers"]
    assert report["decoder_weight_waterfill_actuation_gate"]["train_time_fake_quant_bound"] is True
    assert report["decoder_weight_waterfill_actuation_gate"]["fake_quant_targeted_tensor_count"] == pytest.approx(2.0)


def test_hinerv_runner_short_scorer_smoke_readiness_failure_marks_training_artifact(
    tmp_path: Path,
) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    artifact_path = training_dir / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "substrate_artifact_metadata": {
                    "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                    "score_aware_training": {"schema": "unit"},
                }
            }
        ),
        encoding="utf-8",
    )
    artifact_dict = {
        "per_epoch_metrics": [
            {
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.12,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_scorer_input_contrast_floor": 0.01,
                    "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
                    "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
                    "loss_part_scorer_input_shape_tether": 0.02,
                    "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
                    "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
                    "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
                    "loss_part_posenet_temporal_signal_floor": 0.03,
                    "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
                    "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
                }
            }
        ],
        "substrate_artifact_metadata": {
            "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            "score_aware_training": {"schema": "unit"},
        },
    }
    post_export_quality = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "report_path": (tmp_path / "quality.json").as_posix(),
        "quality_gate_passed": True,
        "quality_gate": {"verdict": "CACHE_QUALITY_GATE_PASSED", "stats": {}},
        "scorer_input_distribution_gate": {"fit_gate_passed": True, "blockers": []},
        "segnet_argmax_probe": {
            "fit_gate_passed": True,
            "segnet_argmax_disagreement_rate": 0.02,
            "candidate_occupied_class_fraction": 0.8,
            "candidate_target_class_coverage_fraction": 0.8,
            "candidate_target_material_class_covered_count": 4.0,
            "target_material_class_count": 5.0,
            "reference_occupied_class_fraction": 0.9,
            "blockers": [],
        },
        "mlx_scorer_response_probe_required": True,
        "mlx_scorer_response_probe": {
            "fit_gate_passed": True,
            "avg_posenet_dist": 0.002,
            "avg_segnet_dist": 0.03,
            "blockers": [],
        },
        "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
    }

    report = runner_mod._write_hi_nerv_runner_short_scorer_smoke_readiness(
        output_dir=training_dir,
        artifact_dict=artifact_dict,
        train_time_controls={
            "segnet_direct_live_distillation_weight": 0.4,
            "scorer_input_contrast_floor_weight": 0.5,
            "scorer_input_contrast_floor_segnet_min_std_ratio": 0.6,
            "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio": 0.6,
            "scorer_input_shape_tether_weight": 0.25,
            "posenet_temporal_signal_floor_weight": 0.25,
        },
        post_export_quality=post_export_quality,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_unscored_research_smoke=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
        require_section_byte_dual_ascent=False,
        require_pose_direct_live_distillation=False,
        decoder_weight_waterfill_plan_metadata=None,
    )

    assert report["ready_for_long_run"] is False
    metadata = artifact_dict["substrate_artifact_metadata"]
    admission = metadata["short_scorer_teacher_smoke_long_run_admission"]
    assert admission["long_run_admission_passed"] is False
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in admission["admission_blockers"]
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in admission["admission_blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in metadata["blockers"]
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in metadata["blockers"]
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    persisted_metadata = persisted["substrate_artifact_metadata"]
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in persisted_metadata["blockers"]


def test_hinerv_runner_receiver_cache_quality_forwards_prioritized_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    source = tmp_path / "source.mkv"
    source.write_bytes(b"unit source video")
    captured_reference: dict[str, object] = {}
    captured_quality: dict[str, object] = {}

    from tac.substrates.hi_nerv import receiver_cache_quality

    def fake_write_reference_cache(**kwargs):
        captured_reference.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {"pair_count": len(kwargs["pair_indices"])}

    def fake_write_hi_nerv_receiver_cache_quality_report(**kwargs):
        captured_quality.update(kwargs)
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": (tmp_path / "report.json").as_posix(),
            "archive_path": Path(kwargs["archive_zip_path"]).as_posix(),
            "archive_sha256": "c" * 64,
            "candidate_cache_dir": (tmp_path / "candidate_cache").as_posix(),
            "reference_cache_dir": Path(kwargs["reference_cache_dir"]).as_posix(),
            "quality_gate_passed": False,
            "quality_gate": {"verdict": "CACHE_QUALITY_GATE_FAILED", "stats": {}},
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        }

    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_source_pair_reference_cache",
        fake_write_reference_cache,
    )
    monkeypatch.setattr(
        receiver_cache_quality,
        "write_hi_nerv_receiver_cache_quality_report",
        fake_write_hi_nerv_receiver_cache_quality_report,
    )

    report = runner_mod._write_hi_nerv_runner_post_export_receiver_cache_quality(
        requested=True,
        archive_zip_path=archive,
        source_video_path=source,
        output_dir=tmp_path / "training",
        reference_cache_dir=None,
        max_pairs=2,
        batch_pairs=1,
        pair_indices=(3, 1, 3),
        min_segnet_std=1.0,
        min_segnet_dynamic_range=16.0,
        max_segnet_mae_vs_reference_for_fit_gate=64.0,
        segnet_argmax_probe=False,
        segnet_argmax_batch_frames=4,
        max_segnet_argmax_disagreement_for_fit_gate=0.25,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.400001,
        repo_root=REPO_ROOT,
    )

    assert report["quality_gate_passed"] is False
    assert captured_reference["pair_indices"] == (3, 1)
    assert captured_quality["pair_indices"] == (3, 1)
    assert captured_quality["sample_pairs"] == 2


def test_hinerv_runner_receiver_cache_quality_records_default_pair_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    source = tmp_path / "source.mkv"
    source.write_bytes(b"unit source video")
    captured_reference: dict[str, object] = {}
    captured_quality: dict[str, object] = {}

    from tac.local_acceleration import mlx_preprocess
    from tac.substrates.hi_nerv import receiver_cache_quality

    def fake_write_scorer_input_cache_from_video_file(
        video_path,
        output_dir,
        *,
        max_pairs,
        batch_pairs,
        **_kwargs,
    ):
        captured_reference.update(
            {
                "video_path": Path(video_path),
                "output_dir": Path(output_dir),
                "max_pairs": int(max_pairs),
                "batch_pairs": int(batch_pairs),
            }
        )
        cache_dir = Path(output_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {"pair_count": int(max_pairs)}

    def fake_write_hi_nerv_receiver_cache_quality_report(**kwargs):
        captured_quality.update(kwargs)
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": (tmp_path / "report.json").as_posix(),
            "archive_path": Path(kwargs["archive_zip_path"]).as_posix(),
            "archive_sha256": "c" * 64,
            "candidate_cache_dir": (tmp_path / "candidate_cache").as_posix(),
            "reference_cache_dir": Path(kwargs["reference_cache_dir"]).as_posix(),
            "quality_gate_passed": False,
            "quality_gate": {"verdict": "CACHE_QUALITY_GATE_FAILED", "stats": {}},
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        }

    monkeypatch.setattr(
        mlx_preprocess,
        "write_scorer_input_cache_from_video_file",
        fake_write_scorer_input_cache_from_video_file,
    )
    monkeypatch.setattr(
        receiver_cache_quality,
        "write_hi_nerv_receiver_cache_quality_report",
        fake_write_hi_nerv_receiver_cache_quality_report,
    )

    report = runner_mod._write_hi_nerv_runner_post_export_receiver_cache_quality(
        requested=True,
        archive_zip_path=archive,
        source_video_path=source,
        output_dir=tmp_path / "training",
        reference_cache_dir=None,
        max_pairs=2,
        batch_pairs=1,
        pair_indices=(),
        min_segnet_std=1.0,
        min_segnet_dynamic_range=16.0,
        max_segnet_mae_vs_reference_for_fit_gate=64.0,
        segnet_argmax_probe=False,
        segnet_argmax_batch_frames=4,
        max_segnet_argmax_disagreement_for_fit_gate=0.25,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.400001,
        repo_root=REPO_ROOT,
    )

    assert report["runner_receiver_cache_quality_pair_indices"] == [0, 1]
    assert captured_reference["video_path"] == source
    assert captured_reference["max_pairs"] == 2
    assert captured_reference["batch_pairs"] == 1
    assert captured_quality["pair_indices"] == (0, 1)
    assert captured_quality["sample_pairs"] == 2


def test_hinerv_receiver_cache_summary_preserves_segnet_class_occupancy(
    tmp_path: Path,
) -> None:
    report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "report_path": (tmp_path / "report.json").as_posix(),
        "archive_path": (tmp_path / "archive.zip").as_posix(),
        "quality_gate_passed": False,
        "quality_gate": {"verdict": "CACHE_QUALITY_GATE_FAILED", "stats": {}},
        "runner_receiver_cache_quality_max_pairs": 2,
        "runner_requested_receiver_cache_quality_max_pairs": 1,
        "runner_effective_receiver_cache_quality_max_pairs": 2,
        "runner_quality_floor_receiver_cache_quality_min_pairs": 2,
        "runner_train_scope_receiver_cache_quality_min_pairs": 2,
        "segnet_argmax_probe": {
            "fit_gate_passed": False,
            "sample_pairs": 1,
            "segnet_argmax_disagreement_rate": 0.5,
            "boundary_argmax_disagreement_rate": 0.9,
            "candidate_argmax_histogram": [0, 0, 196536, 0, 72],
            "reference_argmax_histogram": [44132, 1401, 96943, 3564, 50568],
        },
        "blockers": ["candidate_segnet_argmax_disagreement_too_high"],
    }

    summary = runner_mod._hi_nerv_receiver_cache_quality_summary(report)

    assert summary["segnet_candidate_argmax_histogram"] == [0, 0, 196536, 0, 72]
    assert summary["segnet_reference_argmax_histogram"] == [
        44132,
        1401,
        96943,
        3564,
        50568,
    ]
    assert summary["segnet_candidate_occupied_class_fraction"] == pytest.approx(0.2)
    assert summary["segnet_candidate_any_occupied_class_fraction"] == pytest.approx(0.4)
    assert summary["segnet_candidate_target_class_coverage_fraction"] == pytest.approx(0.2)
    assert summary["segnet_candidate_target_any_class_coverage_fraction"] == pytest.approx(0.4)
    assert summary["segnet_candidate_target_class_min_ratio"] == pytest.approx(0.0)
    assert summary["segnet_target_material_class_count"] == pytest.approx(5.0)
    assert summary["segnet_candidate_target_material_class_covered_count"] == pytest.approx(1.0)
    assert summary["segnet_candidate_target_class_covered"] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert summary["segnet_argmax_occupancy_min_class_pixel_count"] == pytest.approx(197.0)
    assert summary["segnet_argmax_target_coverage_min_class_pixel_count"] == pytest.approx(197.0)
    assert summary["segnet_reference_occupied_class_fraction"] == pytest.approx(1.0)
    assert summary["segnet_reference_any_occupied_class_fraction"] == pytest.approx(1.0)
    assert summary["segnet_argmax_sample_pairs"] == 1
    assert summary["runner_receiver_cache_quality_max_pairs"] == 2
    assert summary["runner_requested_receiver_cache_quality_max_pairs"] == 1
    assert summary["runner_effective_receiver_cache_quality_max_pairs"] == 2
    assert summary["runner_quality_floor_receiver_cache_quality_min_pairs"] == 2
    assert summary["runner_train_scope_receiver_cache_quality_min_pairs"] == 2


def test_hinerv_train_receiver_class_escape_contract_blocks_lost_escape() -> None:
    contract = runner_mod._hi_nerv_train_receiver_class_escape_contract(
        training_telemetry_contract={
            "segnet_direct_live_max_candidate_occupied_class_fraction": 0.8,
        },
        receiver_cache_quality_summary={
            "segnet_candidate_occupied_class_fraction": 0.2,
            "segnet_reference_occupied_class_fraction": 1.0,
        },
    )

    assert contract["passed"] is False
    assert contract["train_direct_live_max_candidate_occupied_class_fraction"] == 0.8
    assert contract["receiver_candidate_occupied_class_fraction"] == 0.2
    assert "hi_nerv_train_time_class_escape_not_receiver_export_preserved" in contract["blockers"]
    assert "hi_nerv_receiver_export_segnet_argmax_class_collapse" in contract["blockers"]


def test_hinerv_train_receiver_class_escape_contract_blocks_lost_target_coverage() -> None:
    contract = runner_mod._hi_nerv_train_receiver_class_escape_contract(
        training_telemetry_contract={
            "segnet_direct_live_max_candidate_occupied_class_fraction": 0.8,
            "segnet_direct_live_max_candidate_target_class_coverage_fraction": 0.8,
        },
        receiver_cache_quality_summary={
            "segnet_candidate_occupied_class_fraction": 0.8,
            "segnet_reference_occupied_class_fraction": 1.0,
            "segnet_candidate_target_class_coverage_fraction": 0.4,
            "segnet_target_material_class_count": 5.0,
        },
    )

    assert contract["passed"] is False
    assert contract["train_direct_live_max_candidate_target_class_coverage_fraction"] == 0.8
    assert contract["receiver_candidate_target_class_coverage_fraction"] == 0.4
    assert "hi_nerv_train_time_target_class_coverage_not_receiver_export_preserved" in contract["blockers"]
    assert "hi_nerv_receiver_export_segnet_target_class_coverage_collapse" in contract["blockers"]


def test_hinerv_train_receiver_class_escape_contract_blocks_scope_mismatch_claim() -> None:
    contract = runner_mod._hi_nerv_train_receiver_class_escape_contract(
        training_telemetry_contract={
            "row_count": 2,
            "train_source_pair_indices_observed": [0, 1],
            "train_source_pair_count_observed": 2,
            "segnet_direct_live_max_candidate_occupied_class_fraction": 0.8,
            "segnet_direct_live_max_candidate_target_class_coverage_fraction": 0.8,
        },
        receiver_cache_quality_summary={
            "segnet_argmax_sample_pairs": 1,
            "runner_receiver_cache_quality_pair_indices": [0],
            "segnet_candidate_occupied_class_fraction": 0.4,
            "segnet_reference_occupied_class_fraction": 1.0,
            "segnet_candidate_target_class_coverage_fraction": 0.4,
            "segnet_target_material_class_count": 5.0,
        },
    )

    assert contract["passed"] is False
    assert contract["train_receiver_scope_aligned"] is False
    assert contract["train_telemetry_row_count"] == 2
    assert contract["train_source_pair_indices_observed"] == [0, 1]
    assert contract["train_source_pair_count_observed"] == 2
    assert contract["receiver_segnet_argmax_sample_pairs"] == 1
    assert contract["receiver_pair_indices"] == [0]
    assert "hi_nerv_train_receiver_class_escape_pair_scope_mismatch" in contract["blockers"]
    assert "hi_nerv_train_time_class_escape_not_receiver_export_preserved" not in contract["blockers"]
    assert "hi_nerv_train_time_target_class_coverage_not_receiver_export_preserved" not in contract["blockers"]
    assert "hi_nerv_receiver_export_segnet_target_class_coverage_collapse" in contract["blockers"]


def test_hinerv_train_receiver_class_escape_contract_preserves_aligned_scope_loss() -> None:
    contract = runner_mod._hi_nerv_train_receiver_class_escape_contract(
        training_telemetry_contract={
            "row_count": 2,
            "train_source_pair_indices_observed": [0, 1],
            "train_source_pair_count_observed": 2,
            "segnet_direct_live_max_candidate_occupied_class_fraction": 0.8,
            "segnet_direct_live_max_candidate_target_class_coverage_fraction": 0.8,
        },
        receiver_cache_quality_summary={
            "segnet_argmax_sample_pairs": 2,
            "runner_receiver_cache_quality_pair_indices": [0, 1],
            "segnet_candidate_occupied_class_fraction": 0.4,
            "segnet_reference_occupied_class_fraction": 1.0,
            "segnet_candidate_target_class_coverage_fraction": 0.4,
            "segnet_target_material_class_count": 5.0,
        },
    )

    assert contract["passed"] is False
    assert contract["train_receiver_scope_aligned"] is True
    assert contract["train_source_pair_indices_observed"] == [0, 1]
    assert contract["receiver_pair_indices"] == [0, 1]
    assert "hi_nerv_train_receiver_class_escape_pair_scope_mismatch" not in contract["blockers"]
    assert "hi_nerv_train_time_class_escape_not_receiver_export_preserved" in contract["blockers"]
    assert "hi_nerv_train_time_target_class_coverage_not_receiver_export_preserved" in contract["blockers"]


def test_hinerv_effective_receiver_cache_quality_max_pairs_tracks_train_scope() -> None:
    assert runner_mod._hi_nerv_effective_receiver_cache_quality_max_pairs(
        requested_max_pairs=1,
        num_pairs=2,
        train_batch_pairs=2,
    ) == {
        "requested_max_pairs": 1,
        "quality_floor_min_pairs": 2,
        "train_scope_min_pairs": 2,
        "effective_max_pairs": 2,
    }
    assert (
        runner_mod._hi_nerv_effective_receiver_cache_quality_max_pairs(
            requested_max_pairs=4,
            num_pairs=2,
            train_batch_pairs=2,
        )["effective_max_pairs"]
        == 4
    )
    assert runner_mod._hi_nerv_effective_receiver_cache_quality_max_pairs(
        requested_max_pairs=1,
        num_pairs=1,
        train_batch_pairs=4,
    ) == {
        "requested_max_pairs": 1,
        "quality_floor_min_pairs": 1,
        "train_scope_min_pairs": 1,
        "effective_max_pairs": 1,
    }
    assert runner_mod._hi_nerv_effective_receiver_cache_quality_max_pairs(
        requested_max_pairs=1,
        num_pairs=600,
        train_batch_pairs=4,
    ) == {
        "requested_max_pairs": 1,
        "quality_floor_min_pairs": 16,
        "train_scope_min_pairs": 16,
        "effective_max_pairs": 16,
    }


def _snerv_official_skip_candidate(mode: str) -> dict:
    rows = enumerate_snerv_modelsize_candidates(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        wavelet="haar",
        levels=(1,),
        bits_per_coeffs=(1.5,),
        step_map_bits_per_coeffs=(0.5,),
        decoder_codecs=("int8_symmetric",),
        snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        official_modelsize_mparams=(0.05,),
        emb_sizes=(0,),
        patch_radius=1,
        mfu_scales=(1, 2, 4),
        hfr_gain=0.0,
        temporal_context=0,
        temporal_modes=("official_haar_dwt1d_lowpass",),
        official_skip_high_modes=(mode,),
    )
    official_rows = [row for row in rows if row.capacity_source == "official_snerv_modelsize"]
    assert len(official_rows) == 1
    return official_rows[0].as_dict()


def _write_snerv_binary_profile_receiver_feedback(
    tmp_path: Path,
    *,
    candidate: dict,
    archive_bytes: int,
    archive_sha256: str,
) -> Path:
    run_root = tmp_path / "snerv_run"
    package = run_root / "snerv_mlx_native_export" / "native_train_export" / "snerv_mlx_native_archive_bound_package"
    archive = package / "archive.zip"
    packet = package.parent / "snerv_mlx_native_packet.snar"
    proof = package / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
    profile_dir = tmp_path / "binary_profile"
    startup = run_root / "compact_renderer_mlx_spine_runner_startup.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"synthetic archive bytes")
    packet.write_bytes(b"SNAR1 synthetic packet bytes")
    actual_archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "execute_family": "snerv",
                "modelsize_candidate": candidate,
                "modelsize_candidate_id": "auto",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                "archive_bytes": int(archive_bytes),
                "archive_path": archive.as_posix(),
                "archive_sha256": actual_archive_sha,
                "runtime_consumption_proof_ready": True,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_dir / "snerv_binary_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "snerv_binary_profile.v1",
                "charged_archive_bytes": int(archive_bytes),
                "input_kind": "contest_archive_zip",
                "input_path": archive.as_posix(),
                "input_sha256": actual_archive_sha,
                "snar1_metadata": {"n_pairs": int(candidate["num_pairs"])},
                "snar1_packet_bytes": packet.stat().st_size,
                "snar1_packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return profile


def test_scorer_error_pair_curriculum_from_xray_rows_builds_weights(
    tmp_path: Path,
) -> None:
    report = tmp_path / "xray.json"
    report.write_text(
        json.dumps(
            {
                "schema": "mlx_prefilter_error_anatomy.v1",
                "rows": [
                    {
                        "pair_idx": 5,
                        "component_score_no_rate": 2.0,
                        "seg_score_contribution": 1.25,
                        "pose_score_contribution": 0.75,
                    },
                    {
                        "pair_idx": 9,
                        "component_score_no_rate": 4.0,
                        "seg_score_contribution": 0.0,
                        "pose_score_contribution": 4.0,
                    },
                    {"pair_idx": 11, "component_score_no_rate": -1.0},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        scorer_error_pair_curriculum_json=report,
        auto_scorer_error_pair_curriculum=False,
        mlx_profile=(),
        scorer_error_pair_curriculum_default_weight=0.5,
        scorer_error_pair_curriculum_gain=3.0,
        scorer_error_pair_curriculum_field="component_score_no_rate",
        scorer_error_pair_curriculum_top_k=0,
        repo_root=REPO_ROOT,
    )

    weights, metadata = runner_mod._scorer_error_pair_curriculum_from_args(
        args,
        output_dir=tmp_path / "out",
    )

    assert weights == {9: pytest.approx(3.5), 5: pytest.approx(2.0)}
    assert metadata["enabled"] is True
    assert metadata["field"] == "component_score_no_rate"
    assert metadata["weighted_pair_count"] == 2
    geometry = metadata["distortion_geometry"]
    assert geometry["full_lagrangian_is_final_arbiter"] is True
    assert geometry["segnet_domain"] == "last_frame_spatial_argmax_boundary_geometry"
    assert metadata["score_claim"] is False


def test_scorer_error_pair_curriculum_honors_axis_field_and_top_k(
    tmp_path: Path,
) -> None:
    report = tmp_path / "xray.jsonl"
    report.write_text(
        "\n".join(
            [
                json.dumps({"pair_idx": 1, "pose_score_contribution": 2.0}),
                json.dumps({"pair_idx": 2, "pose_score_contribution": 8.0}),
                json.dumps({"pair_idx": 3, "pose_score_contribution": 4.0}),
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        scorer_error_pair_curriculum_json=report,
        auto_scorer_error_pair_curriculum=False,
        mlx_profile=(),
        scorer_error_pair_curriculum_default_weight=1.0,
        scorer_error_pair_curriculum_gain=2.0,
        scorer_error_pair_curriculum_field="pose_score_contribution",
        scorer_error_pair_curriculum_top_k=1,
        repo_root=REPO_ROOT,
    )

    weights, metadata = runner_mod._scorer_error_pair_curriculum_from_args(
        args,
        output_dir=tmp_path / "out",
    )

    assert weights == {2: pytest.approx(3.0)}
    assert metadata["field"] == "pose_score_contribution"
    assert metadata["top_k"] == 1
    assert metadata["top_weighted_pairs"][0]["pair_idx"] == 2


def test_scorer_error_pair_curriculum_consumes_direct_vjp_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "direct_full_scorer_vjp_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "direct_full_scorer_vjp_bundle.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "exact_reduction_contract": {
                    "single_update_after_all_shards_reduce": True,
                    "budget_spend_allowed_before_full_reduction": False,
                    "full_reduction_complete": True,
                },
                "gradient_quality_blockers": [],
                "shards": [
                    {
                        "backend": "mlx",
                        "device_type": "gpu",
                        "pair_start": 10,
                        "pair_end": 13,
                        "loss_contribution": 0.125,
                        "gradient_quality_blockers": [],
                        "posenet_yuv6_pair_grad": {
                            "per_pair_l2": [0.25, 7.5, 0.75],
                        },
                        "segnet_last_rgb_grad": {
                            "per_pair_l2": [1.75, 0.5, 0.25],
                        },
                        "top_pairs_by_grad_l2": [
                            {
                                "pair_idx": 10,
                                "combined_grad_l2": 2.0,
                                "pose_grad_l2": 0.25,
                                "seg_grad_l2": 1.75,
                            },
                            {
                                "pair_idx": 11,
                                "combined_grad_l2": 8.0,
                                "pose_grad_l2": 7.5,
                                "seg_grad_l2": 0.5,
                            },
                        ],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        scorer_error_pair_curriculum_json=bundle,
        auto_scorer_error_pair_curriculum=False,
        mlx_profile=(),
        scorer_error_pair_curriculum_default_weight=0.25,
        scorer_error_pair_curriculum_gain=4.0,
        scorer_error_pair_curriculum_field="pose_grad_l2",
        scorer_error_pair_curriculum_top_k=0,
        repo_root=REPO_ROOT,
    )

    weights, metadata = runner_mod._scorer_error_pair_curriculum_from_args(
        args,
        output_dir=tmp_path / "out",
    )

    assert weights == {
        11: pytest.approx(4.25),
        12: pytest.approx(0.25 + 4.0 * (0.75 / 7.5)),
        10: pytest.approx(0.25 + 4.0 * (0.25 / 7.5)),
    }
    assert metadata["enabled"] is True
    assert metadata["field"] == "pose_grad_l2"
    assert metadata["input_row_count"] == 3
    assert metadata["distortion_geometry"]["selected_axis_field"] == "pose_grad_l2"
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False


def test_scorer_error_pair_curriculum_rejects_partial_direct_vjp_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "partial_direct_full_scorer_vjp_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "direct_full_scorer_vjp_bundle.v1",
                "exact_reduction_contract": {
                    "full_reduction_complete": False,
                    "budget_spend_allowed_before_full_reduction": False,
                },
                "gradient_quality_blockers": [],
                "shards": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(CompactRendererMlxSpineRunnerError, match="exact full-video reduction"):
        runner_mod._load_scorer_error_pair_rows(bundle)


def test_scorer_error_pair_curriculum_rejects_quality_blocked_direct_vjp_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "blocked_direct_full_scorer_vjp_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "direct_full_scorer_vjp_bundle.v1",
                "exact_reduction_contract": {
                    "full_reduction_complete": True,
                    "budget_spend_allowed_before_full_reduction": False,
                },
                "gradient_quality_blockers": ["segnet_last_rgb_gradient_abs_max_exceeds_sanity_limit"],
                "shards": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(CompactRendererMlxSpineRunnerError, match="gradient quality blockers"):
        runner_mod._load_scorer_error_pair_rows(bundle)


def test_scorer_coupled_rd_metadata_prices_cooperative_receiver_axes() -> None:
    metadata = runner_mod._scorer_coupled_rd_metadata()

    assert metadata["schema"] == "contest_scorer_coupled_rd_allocation_facts.v2"
    assert metadata["authority"] == "planning_metadata_only_not_score_authority"
    assert metadata["reference_uncompressed_total_bytes"] == 37_545_489
    assert metadata["fixed_marginal_byte_price"] == "25/uncompressed_total"
    assert metadata["fixed_marginal_byte_price_score_per_byte"] == pytest.approx(25.0 / 37_545_489)
    assert metadata["cooperative_receiver"]["human_visual_fidelity_objective"] is False
    assert metadata["cooperative_receiver"]["projection_domains"] == [
        "segnet_last_frame_rgb_384x512_argmax5",
        "posenet_pair_yuv6_384x512_pose6",
        "archive_zip_bytes",
    ]
    assert metadata["segnet_domain"]["output_math"] == ("five_class_argmax_disagreement_rate")
    assert metadata["segnet_domain"]["score_derivative"] == pytest.approx(100.0)
    assert metadata["segnet_domain"]["frame_score_weights"] == {
        "frame_0": 0.0,
        "frame_1": 100.0,
    }
    assert metadata["posenet_domain"]["score_derivative_formula"] == ("5/sqrt(10*d_pose)")
    assert metadata["posenet_domain"]["score_derivative_operating_point_dependent"] is True
    assert metadata["frame_pair_asymmetry"]["frame_0"]["segnet_score_weight"] == 0.0
    assert metadata["frame_pair_asymmetry"]["frame_1"]["segnet_score_weight"] == (pytest.approx(100.0))
    assert metadata["atom_admission_rule"]["delta_convention"] == ("candidate_minus_current")
    assert metadata["atom_admission_rule"]["admit_when"] == ("linearized_score_delta < 0")


def _synthetic_snerv_packet(*, pairs: int = 2) -> bytes:
    plane_count = int(pairs) * 2 * 3
    lf_planes = [(np.arange(48, dtype=np.int64).reshape(6, 8) + idx) % 17 for idx in range(plane_count)]
    step_maps = [np.full((6, 8), 1.0 + idx * 0.01, dtype=np.float32) for idx in range(plane_count)]
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=np.zeros(plane_count, dtype=np.float32),
        ),
        lf_payload=encode_lf_quant_payload(lf_planes),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=2)),
        step_map_packet=encode_step_maps(step_maps, bins=4).packet,
        metadata={
            "n_pairs": int(pairs),
            "frames_per_pair": 2,
            "channels": 3,
            "height": 12,
            "width": 16,
            "lf_plane_count": plane_count,
            "levels": 2,
            "wavelet": "db2",
        },
    )
    return archive.packet


def test_snerv_native_export_attachment_threads_mlx_prefilter_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod

    captured: dict[str, object] = {}

    def fake_train_export_snerv_mlx_native(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        report = out / "snerv_mlx_native_train_export.json"
        packet = out / "snerv_mlx_native_packet.snar"
        archive = out / "archive.zip"
        proof = out / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        profile = out / "local_mlx_prefilter" / "local_mlx_prefilter_profile.json"
        progress = out / "local_mlx_prefilter" / "local_mlx_prefilter_progress.jsonl"
        packet.write_bytes(b"SNAR1 fake packet")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", packet.read_bytes())
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_passed": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        profile.parent.mkdir()
        profile_payload = {
            "schema": "hprc_mlx_component_profile.v1",
            "n_samples": 600,
            "num_pairs": 600,
            "scorer_batch_pairs": 4,
            "scope_status": {"full_video": "complete"},
            "blockers": ["mlx_profile_batch_pairs_not_singleton"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        profile.write_text(json.dumps(profile_payload, sort_keys=True), encoding="utf-8")
        progress.write_text(
            json.dumps({"schema": "mlx_renderer_prefilter_progress.v1"}) + "\n",
            encoding="utf-8",
        )
        artifact = {
            "schema": "snerv_mlx_native_train_export.v1",
            "num_pairs": 600,
            "source_pair_indices": [],
            "report_path": report.as_posix(),
            "packet_path": packet.as_posix(),
            "packet_bytes": packet.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "snerv_official_tub_source_fixture_binding": {
                "schema": "snerv_official_tub_source_fixture_binding.v1",
                "component_id": "tub",
                "source_fixture_replay_bound": True,
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed": True,
                "full_tub_source_forward_parity_proven": False,
                "source_forward_replay_authority": False,
                "preserved_source_parity_blockers": [
                    "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"
                ],
                "blockers": ["snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "snerv_official_tub_source_fixture_replay_bound": True,
            "snerv_official_tub_source_fixture_replay_passed": True,
            "snerv_official_tub_source_forward_fixture_bound": True,
            "official_source_parity_blockers": [
                "snerv_official_bootstrap_stores_haar_ll_as_mfu_skip_high",
                "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing",
            ],
            "receiver_target_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_source_targets",
                    reference_kind="source_targets_nchw255",
                    mse=0.75,
                    max_abs=1.5,
                )
            ),
            "receiver_export_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_export_reference",
                    reference_kind="export_reference_nchw255",
                    mse=0.0,
                    max_abs=0.0,
                )
            ),
            "native_mlx_training_executed": True,
            "native_mlx_training_kind": "score_aware_long_training",
            "native_mlx_hf_decoder_training": {
                "requested_steps": 1,
                "attempted": True,
                "executed": True,
                "accepted": True,
                "blockers": [],
            },
            "score_aware_long_training_executed": True,
            "score_aware_long_training": {
                "executed": True,
                "epochs": 8,
                "optimizer": "pact_muon_adamw",
                "scorer_input_distribution_guard_bound": True,
                "scorer_input_contrast_floor_bound": True,
                "scorer_input_shape_tether_bound": True,
                "training_telemetry_contract": {
                    "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
                    "telemetry_exists": True,
                    "row_count": 1,
                    "passed": True,
                    "expected_scorer_input_guard_metric": True,
                    "scorer_input_guard_metric_observed": True,
                    "scorer_input_guard_dual_metric_observed": True,
                    "expected_scorer_input_contrast_floor_metric": True,
                    "scorer_input_contrast_floor_metric_observed": True,
                    "scorer_input_contrast_floor_segnet_ratio_metric_observed": True,
                    "scorer_input_contrast_floor_posenet_ratio_metric_observed": True,
                    "expected_scorer_input_shape_tether_metric": True,
                    "scorer_input_shape_tether_metric_observed": True,
                    "scorer_input_shape_tether_segnet_metric_observed": True,
                    "scorer_input_shape_tether_posenet_pair_metric_observed": True,
                    "scorer_input_shape_tether_posenet_delta_metric_observed": True,
                    "blockers": [],
                },
            },
            "scorer_loop_qat": {
                "executed": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "accepted_improvement": True,
                "emitted_packet_uses_scorer_loop_best_decoder": True,
            },
            "local_mlx_prefilter_profile": profile_payload,
            "local_mlx_prefilter_profile_path": profile.as_posix(),
            "local_mlx_prefilter_progress_path": progress.as_posix(),
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        report.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        native_mod,
        "train_export_snerv_mlx_native",
        fake_train_export_snerv_mlx_native,
    )

    stage_weights = {
        "recon": 0.5,
        "distill": 1.25,
        "pose_distill": 0.75,
        "scorer_input_guard": 0.25,
        "scorer_input_contrast_floor": 0.375,
        "scorer_input_shape_tether": 0.625,
        "segnet_direct_live_distill": 0.125,
    }
    out = runner_mod._run_snerv_native_mlx_export_attachment(
        requested=True,
        output_dir=tmp_path / "snerv_native_attachment",
        num_pairs=600,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        scorer_upstream_dir=REPO_ROOT / "upstream",
        modelsize_candidate={"candidate_id": "snerv-prefilter-test"},
        prioritized_pair_indices=(3, 5),
        scorer_error_pair_sampling_weights={5: 2.0},
        scorer_error_pair_curriculum={
            "schema": "test_scorer_error_pair_curriculum.v1",
            "enabled": True,
            "field": "component_score_no_rate",
            "default_weight": 1.0,
        },
        repo_root=REPO_ROOT,
        allow_overwrite=False,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=12,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=2,
        scorer_loop_qat_search_mode="learned_random_subspace",
        scorer_loop_qat_qat_bits=4,
        scorer_loop_qat_decoder_payload_codec="int4_symmetric",
        scorer_loop_qat_lf_payload_codec="int4_symmetric",
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
        scorer_loop_qat_pair_guard_min_score_improved_fraction=0.875,
        scorer_loop_qat_pair_guard_max_pose_worsened_fraction=0.125,
        scorer_loop_qat_device="gpu",
        scorer_loop_qat_perturb_scale=0.03125,
        scorer_loop_qat_byte_pressure_multiplier=2.0,
        scorer_loop_qat_section_value_pressure_multiplier=1.5,
        scorer_loop_qat_max_archive_byte_growth=9,
        scorer_loop_qat_byte_growth_admission_mode="rate_paid",
        scorer_loop_qat_pose_slack=0.004,
        scorer_loop_qat_seg_slack=0.005,
        scorer_loop_qat_seed=99,
        recon_pixel_weight_path=None,
        recon_pixel_weight_manifest_path=None,
        recon_pixel_weight_normalize="mean",
        native_mlx_decoder_train_steps=0,
        native_mlx_decoder_train_lr=1e-5,
        native_mlx_decoder_train_ridge=1e-6,
        native_mlx_decoder_train_optimizer="closed_form",
        official_trained_checkpoint_state_dict_path=(tmp_path / "official_state_dict_slice.npz"),
        score_aware_long_training_epochs=8,
        score_aware_long_training_lr=1e-3,
        score_aware_long_training_batch_pairs=2,
        score_aware_long_training_optimizer="pact_muon_adamw",
        score_aware_long_training_grad_clip_max_norm=None,
        score_aware_long_training_weight_decay=0.01,
        score_aware_long_training_eval_roundtrip_ste=True,
        score_aware_long_training_scorer_tether_smoke_steps=3,
        score_aware_long_training_section_byte_refresh_every_steps=25,
        score_aware_long_training_scorer_input_contrast_floor_weight=0.11,
        score_aware_long_training_scorer_input_shape_tether_weight=0.12,
        score_aware_long_training_loss_weights=stage_weights,
        score_aware_long_training_pose_warmup_epochs=2,
        score_aware_long_training_scorer_input_shape_warmup_epochs=1,
        score_aware_long_training_segnet_direct_live_escape_warmup_epochs=3,
        score_aware_long_training_scorer_space_step_guard_enabled=True,
        score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.31,
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.32,
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.33,
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.34,
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=0.035,
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio=3.25,
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae=0.14,
        score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.15,
        score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=2.25,
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.16,
        score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term=1.75,
        score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term=0.055,
        score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening=0.025,
        score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening=0.026,
        score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=0.027,
        score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=0.028,
        score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening=0.029,
        score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=1.25,
        score_aware_long_training_scorer_space_step_guard_backtracking_steps=4,
        score_aware_long_training_scorer_space_step_guard_backtracking_shrink=0.4,
        checkpoint_retention_keep_last_n=2,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=0.5,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_student_live_calibration_weight=0.9,
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_base_loss_weight=0.5,
        segnet_direct_live_class_histogram_weight=0.75,
        segnet_direct_live_class_balanced_hinge_weight=0.375,
        segnet_direct_live_class_balanced_ce_weight=0.625,
        segnet_direct_live_class_balanced_squared_hinge_weight=0.875,
        segnet_direct_live_class_region_recon_weight=0.9375,
        segnet_tau_boundary=1.25,
        segnet_hinge_margin=0.5,
        distillation_device="mps",
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0001,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=512,
        score_aware_long_training_pr95_faithful_curriculum=True,
        score_aware_long_training_pr95_muon_policy="faithful_stage8_only",
        write_mlx_prefilter_profile=True,
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=7,
    )

    assert captured["write_mlx_prefilter_profile"] is True
    assert captured["mlx_prefilter_scorer_device"] == "gpu"
    assert captured["mlx_prefilter_scorer_batch_pairs"] == 4
    assert captured["mlx_prefilter_progress_every"] == 7
    assert captured["prioritized_pair_indices"] == (3, 5)
    assert captured["scorer_error_pair_sampling_weights"] == {5: 2.0}
    assert captured["scorer_error_pair_curriculum"]["field"] == ("component_score_no_rate")
    assert captured["score_aware_long_training_optimizer"] == "pact_muon_adamw"
    assert captured["official_trained_checkpoint_state_dict_path"] == (tmp_path / "official_state_dict_slice.npz")
    assert captured["scorer_loop_qat_pair_guard_min_score_improved_fraction"] == (pytest.approx(0.875))
    assert captured["scorer_loop_qat_pair_guard_max_pose_worsened_fraction"] == (pytest.approx(0.125))
    assert captured["scorer_loop_qat_perturb_scale"] == pytest.approx(0.03125)
    assert captured["scorer_loop_qat_byte_pressure_multiplier"] == pytest.approx(2.0)
    assert captured["scorer_loop_qat_section_value_pressure_multiplier"] == (pytest.approx(1.5))
    assert captured["scorer_loop_qat_max_archive_byte_growth"] == 9
    assert captured["scorer_loop_qat_byte_growth_admission_mode"] == "rate_paid"
    assert captured["scorer_loop_qat_pose_slack"] == pytest.approx(0.004)
    assert captured["scorer_loop_qat_seg_slack"] == pytest.approx(0.005)
    assert captured["scorer_loop_qat_seed"] == 99
    assert captured["score_aware_long_training_pr95_faithful_curriculum"] is True
    assert captured["score_aware_long_training_loss_weights"] == stage_weights
    assert captured["score_aware_long_training_pose_warmup_epochs"] == 2
    assert captured["score_aware_long_training_scorer_input_shape_warmup_epochs"] == 1
    assert captured["score_aware_long_training_segnet_direct_live_escape_warmup_epochs"] == 3
    assert captured["score_aware_long_training_scorer_input_contrast_floor_weight"] == (pytest.approx(0.11))
    assert captured["score_aware_long_training_scorer_input_shape_tether_weight"] == (pytest.approx(0.12))
    assert captured["score_aware_long_training_scorer_space_step_guard_enabled"] is True
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"
    ] == pytest.approx(0.31)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction"
    ] == pytest.approx(0.32)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"
    ] == pytest.approx(0.33)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio"
    ] == pytest.approx(0.34)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop"
    ] == pytest.approx(0.035)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio"
    ] == pytest.approx(3.25)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae"
    ] == pytest.approx(0.14)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"
    ] == pytest.approx(0.15)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"
    ] == pytest.approx(2.25)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement"
    ] == pytest.approx(0.16)
    assert captured["score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term"] == pytest.approx(1.75)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term"
    ] == pytest.approx(0.055)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening"
    ] == pytest.approx(0.025)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening"
    ] == pytest.approx(0.026)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening"
    ] == pytest.approx(0.027)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening"
    ] == pytest.approx(0.028)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening"
    ] == pytest.approx(0.029)
    assert captured[
        "score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"
    ] == pytest.approx(1.25)
    assert captured["score_aware_long_training_scorer_space_step_guard_backtracking_steps"] == 4
    assert captured["score_aware_long_training_scorer_space_step_guard_backtracking_shrink"] == pytest.approx(0.4)
    assert captured["segnet_student_live_calibration_weight"] == pytest.approx(0.9)
    assert captured["segnet_direct_live_distillation_weight"] == pytest.approx(0.25)
    assert captured["segnet_direct_live_base_loss_weight"] == pytest.approx(0.5)
    assert captured["segnet_direct_live_class_histogram_weight"] == pytest.approx(0.75)
    assert captured["segnet_direct_live_class_balanced_hinge_weight"] == pytest.approx(0.375)
    assert captured["segnet_direct_live_class_balanced_ce_weight"] == pytest.approx(0.625)
    assert captured["segnet_direct_live_class_balanced_squared_hinge_weight"] == pytest.approx(0.875)
    assert captured["segnet_direct_live_class_region_recon_weight"] == pytest.approx(0.9375)
    assert out["executed"] is True
    assert out["local_mlx_prefilter_profile_path"].endswith("local_mlx_prefilter_profile.json")
    assert out["local_mlx_prefilter_progress_path"].endswith("local_mlx_prefilter_progress.jsonl")
    assert out["local_mlx_prefilter_profile"]["scorer_batch_pairs"] == 4
    assert out["scorer_error_pair_curriculum"]["consumed_by_native_mlx_train_export"] is True
    assert out["scorer_error_pair_curriculum"]["weighted_pair_count"] == 1
    assert out["receiver_reconstruction_verified"] is True
    assert out["snerv_official_tub_source_fixture_replay_bound"] is True
    assert out["snerv_official_tub_source_fixture_replay_passed"] is True
    assert out["snerv_official_tub_source_forward_fixture_bound"] is True
    assert out["snerv_official_tub_source_fixture_binding"]["source_fixture_replay_bound"] is True
    assert (
        "snerv_official_tub_batched_temporal_context_source_forward_replay_missing"
        not in out["official_source_parity_blockers"]
    )
    assert (
        "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing"
        in out["official_source_parity_blockers"]
    )
    assert out["receiver_reconstruction"]["target_mse_nchw255"] == pytest.approx(0.75)
    assert out["receiver_reconstruction"]["export_mse_nchw255"] == pytest.approx(0.0)
    assert out["snerv_scorer_tether_smoke_gate"]["required"] is True
    assert out["snerv_scorer_tether_smoke_gate"]["executed"] is True
    assert out["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert out["snerv_scorer_tether_smoke_gate"]["steps"] == 3
    assert Path(out["snerv_scorer_tether_smoke_gate"]["gate_path"]).is_file()
    assert out["score_aware_long_training_telemetry_contract"]["passed"] is True
    assert out["native_mlx_full600_campaign_ready"] is True
    assert out["score_claim"] is False


def test_snerv_native_export_attachment_blocks_failed_training_telemetry_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod

    def fake_train_export_snerv_mlx_native(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        packet = out / "snerv_mlx_native_packet.snar"
        archive = out / "archive.zip"
        proof = out / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        packet.write_bytes(b"SNAR1 fake packet")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", packet.read_bytes())
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_passed": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "schema": "snerv_mlx_native_train_export.v1",
            "num_pairs": 600,
            "packet_path": packet.as_posix(),
            "packet_bytes": packet.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_target_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_source_targets",
                    reference_kind="source_targets_nchw255",
                    mse=0.75,
                    max_abs=1.5,
                )
            ),
            "receiver_export_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_export_reference",
                    reference_kind="export_reference_nchw255",
                    mse=0.0,
                    max_abs=0.0,
                )
            ),
            "native_mlx_training_executed": True,
            "native_mlx_training_kind": "score_aware_long_training",
            "score_aware_long_training_executed": True,
            "score_aware_long_training": {
                "executed": True,
                "epochs": 8,
                "training_telemetry_contract": {
                    "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
                    "telemetry_exists": True,
                    "row_count": 1,
                    "passed": False,
                    "blockers": ["unit_snerv_training_control_missing"],
                },
            },
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        native_mod,
        "train_export_snerv_mlx_native",
        fake_train_export_snerv_mlx_native,
    )

    out = runner_mod._run_snerv_native_mlx_export_attachment(
        requested=True,
        output_dir=tmp_path / "snerv_native_attachment_bad_telemetry",
        num_pairs=600,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        scorer_upstream_dir=REPO_ROOT / "upstream",
        modelsize_candidate={"candidate_id": "snerv-bad-telemetry-test"},
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights={},
        scorer_error_pair_curriculum={},
        repo_root=REPO_ROOT,
        allow_overwrite=False,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=12,
        run_scorer_loop_qat=False,
        scorer_loop_qat_max_trials=0,
        scorer_loop_qat_search_mode="learned_random_subspace",
        scorer_loop_qat_qat_bits=4,
        scorer_loop_qat_decoder_payload_codec="int4_symmetric",
        scorer_loop_qat_lf_payload_codec="int4_symmetric",
        scorer_loop_qat_component_guard_mode="score_primary",
        scorer_loop_qat_pair_guard_min_score_improved_fraction=1.0,
        scorer_loop_qat_pair_guard_max_pose_worsened_fraction=0.0,
        scorer_loop_qat_device="gpu",
        scorer_loop_qat_perturb_scale=0.02,
        scorer_loop_qat_byte_pressure_multiplier=1.0,
        scorer_loop_qat_section_value_pressure_multiplier=1.0,
        scorer_loop_qat_max_archive_byte_growth=None,
        scorer_loop_qat_byte_growth_admission_mode="hard_cap",
        scorer_loop_qat_pose_slack=0.0,
        scorer_loop_qat_seg_slack=0.0,
        scorer_loop_qat_seed=0,
        recon_pixel_weight_path=None,
        recon_pixel_weight_manifest_path=None,
        recon_pixel_weight_normalize="mean",
        native_mlx_decoder_train_steps=0,
        native_mlx_decoder_train_lr=1e-5,
        native_mlx_decoder_train_ridge=1e-6,
        native_mlx_decoder_train_optimizer="closed_form",
        score_aware_long_training_epochs=8,
        score_aware_long_training_lr=1e-3,
        score_aware_long_training_batch_pairs=2,
        score_aware_long_training_optimizer="pact_muon_adamw",
        score_aware_long_training_grad_clip_max_norm=None,
        score_aware_long_training_weight_decay=0.01,
        score_aware_long_training_eval_roundtrip_ste=True,
        score_aware_long_training_section_byte_refresh_every_steps=25,
        checkpoint_retention_keep_last_n=2,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=0.5,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.25,
        segnet_hinge_margin=0.5,
        distillation_device="mps",
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0001,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=512,
        score_aware_long_training_pr95_faithful_curriculum=True,
        score_aware_long_training_pr95_muon_policy="faithful_stage8_only",
        write_mlx_prefilter_profile=False,
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=0,
    )

    assert out["receiver_reconstruction_verified"] is True
    assert out["score_aware_long_training_telemetry_contract"]["passed"] is False
    assert "unit_snerv_training_control_missing" in out["blockers"]
    assert "snerv_score_aware_long_training_telemetry_contract_failed" in out["blockers"]
    assert out["native_mlx_full600_campaign_ready"] is False


def test_snerv_native_export_attachment_refuses_long_training_when_tether_smoke_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod

    def failing_smoke(*, steps: int = 2) -> dict[str, object]:
        return {
            "schema": "snerv_scorer_tether_smoke.v1",
            "steps": int(steps),
            "passed": False,
            "blockers": ["snerv_posenet_yuv6_pair_distill_dual_lambda_inactive"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fail_native_export(**_kwargs):
        raise AssertionError("native SNeRV export must not launch after failed tether smoke")

    monkeypatch.setattr(runner_mod, "run_snerv_scorer_tether_smoke", failing_smoke)
    monkeypatch.setattr(native_mod, "train_export_snerv_mlx_native", fail_native_export)

    out = runner_mod._run_snerv_native_mlx_export_attachment(
        requested=True,
        output_dir=tmp_path / "snerv_native_attachment_failed_tether",
        num_pairs=2,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        scorer_upstream_dir=REPO_ROOT / "upstream",
        modelsize_candidate={"candidate_id": "snerv-failed-tether-test"},
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights={},
        scorer_error_pair_curriculum={},
        repo_root=REPO_ROOT,
        allow_overwrite=False,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=12,
        run_scorer_loop_qat=False,
        scorer_loop_qat_max_trials=0,
        scorer_loop_qat_search_mode="learned_random_subspace",
        scorer_loop_qat_qat_bits=4,
        scorer_loop_qat_decoder_payload_codec="int4_symmetric",
        scorer_loop_qat_lf_payload_codec="int4_symmetric",
        scorer_loop_qat_component_guard_mode="score_primary",
        scorer_loop_qat_pair_guard_min_score_improved_fraction=1.0,
        scorer_loop_qat_pair_guard_max_pose_worsened_fraction=0.0,
        scorer_loop_qat_device="gpu",
        scorer_loop_qat_perturb_scale=0.02,
        scorer_loop_qat_byte_pressure_multiplier=1.0,
        scorer_loop_qat_section_value_pressure_multiplier=1.0,
        scorer_loop_qat_max_archive_byte_growth=None,
        scorer_loop_qat_byte_growth_admission_mode="hard_cap",
        scorer_loop_qat_pose_slack=0.0,
        scorer_loop_qat_seg_slack=0.0,
        scorer_loop_qat_seed=0,
        recon_pixel_weight_path=None,
        recon_pixel_weight_manifest_path=None,
        recon_pixel_weight_normalize="mean",
        native_mlx_decoder_train_steps=0,
        native_mlx_decoder_train_lr=1e-5,
        native_mlx_decoder_train_ridge=1e-6,
        native_mlx_decoder_train_optimizer="closed_form",
        score_aware_long_training_epochs=8,
        score_aware_long_training_lr=1e-3,
        score_aware_long_training_batch_pairs=2,
        score_aware_long_training_optimizer="pact_muon_adamw",
        score_aware_long_training_grad_clip_max_norm=None,
        score_aware_long_training_weight_decay=0.01,
        score_aware_long_training_eval_roundtrip_ste=True,
        score_aware_long_training_scorer_tether_smoke_steps=5,
        score_aware_long_training_section_byte_refresh_every_steps=25,
        checkpoint_retention_keep_last_n=2,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=0.5,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.25,
        segnet_hinge_margin=0.5,
        distillation_device="mps",
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0001,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=512,
        score_aware_long_training_pr95_faithful_curriculum=True,
        score_aware_long_training_pr95_muon_policy="faithful_stage8_only",
        write_mlx_prefilter_profile=False,
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=0,
    )

    assert out["executed"] is False
    assert out["native_mlx_training_executed"] is False
    assert out["score_aware_long_training_executed"] is False
    assert "snerv_scorer_tether_smoke_failed_before_long_training" in out["blockers"]
    assert "snerv_posenet_yuv6_pair_distill_dual_lambda_inactive" in out["blockers"]
    gate = out["snerv_scorer_tether_smoke_gate"]
    assert gate["required"] is True
    assert gate["executed"] is True
    assert gate["passed"] is False
    assert gate["steps"] == 5
    assert Path(gate["gate_path"]).is_file()


def test_write_decoder_weight_saliency_artifact_for_waterfill(tmp_path: Path) -> None:
    artifact = {
        "substrate_artifact_metadata": {
            "decoder_weight_gradient_saliency": {
                "schema": "mlx_decoder_weight_gradient_saliency.v1",
                "row_count": 1,
                "rows": [
                    {
                        "group_name": "decoder.blocks.0.weight",
                        "saliency": 3.25,
                        "sample_count": 2,
                        "numel": 16,
                    }
                ],
                "saliency_by_name": {"decoder.blocks.0.weight": 3.25},
                "blockers": [],
                "authority": "macos_mlx_research_signal_false_authority",
            }
        }
    }

    out = runner_mod._write_decoder_weight_saliency_artifact(
        artifact_dict=artifact,
        output_dir=tmp_path / "hi_nerv_mlx_training",
        family="hi_nerv",
    )

    path = Path(out["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert out["written"] is True
    assert out["row_count"] == 1
    assert out["sha256"] == runner_mod._sha256_file(path)
    assert payload["artifact_schema"] == ("compact_runner_decoder_weight_saliency_artifact.v1")
    assert payload["schema"] == "mlx_decoder_weight_gradient_saliency.v1"
    assert payload["family"] == "hi_nerv"
    assert payload["rows"][0]["group_name"] == "decoder.blocks.0.weight"


def test_hinerv_runner_materializes_waterfill_from_trained_ladder(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        **{
            "blocks.0.weight": np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
            "latents_coarse": np.asarray([999.0], dtype=np.float32),
        },
    )
    manifest_path = tmp_path / "state_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": state_path.as_posix(),
                "artifact_sha256": runner_mod._sha256_file(state_path),
                "tensor_count": 2,
                "consumption_recommended": True,
                "blockers": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ladder_path = tmp_path / "trained_ladder.json"
    ladder_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "family": "hi_nerv",
                "axis_tag": "[planning/control]",
                "num_pairs": 600,
                "archive_rows": [
                    {
                        "row_id": "tiny",
                        "archive_bytes": 1234,
                        "archive_sha256": "a" * 64,
                        "state_npz_manifest_path": manifest_path.as_posix(),
                        "runtime_consumption_proof_ready": True,
                        "receiver_cache_quality_gate_passed": True,
                        "receiver_cache_quality_blockers": [],
                        "decoder_codec": "int8_mixed",
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    saliency = {
        "schema": "mlx_decoder_weight_gradient_saliency.v1",
        "row_count": 1,
        "saliency_by_name": {"blocks.0.weight": 1.25},
        "rows": [
            {
                "group_name": "blocks.0.weight",
                "saliency": 1.25,
                "numel": 3,
            }
        ],
        "blockers": [],
        "authority": "macos_mlx_research_signal_false_authority",
    }
    saliency_path = tmp_path / "decoder_weight_gradient_saliency.json"
    saliency_path.write_text(json.dumps(saliency, sort_keys=True), encoding="utf-8")

    out = runner_mod._write_hi_nerv_decoder_weight_waterfill_from_trained_ladder(
        output_dir=tmp_path / "waterfill",
        trained_archive_byte_oracle={
            "schema": "hi_nerv_trained_archive_byte_oracle.v1",
            "candidate_id": "candidate_a",
            "receiver_closed_modelsize_ladder_path": ladder_path.as_posix(),
        },
        decoder_weight_saliency_artifact={
            "written": True,
            "path": saliency_path.as_posix(),
        },
        action_bits=(0, 2, 32),
    )

    assert out["written"] is True
    assert out["active"] is True
    assert out["candidate_plan_count"] == 1
    bundle_path = Path(out["path"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema"] == "hinerv_archive_ladder_waterfill.v1"
    plan_path = Path(out["candidate_plan_paths"][0]["path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == runner_mod.NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
    assert plan["_source_waterfill_bundle_path"] == bundle_path.as_posix()
    assert plan["candidate_id"] == "candidate_a:tiny"
    assert plan["score_claim"] is False
    assert out["score_claim"] is False


def test_hinerv_runner_materializes_waterfill_from_receiver_closed_ladder(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "live"
    export_dir.mkdir()
    archive = export_dir / "archive.zip"
    archive.write_bytes(b"trained-hinerv")
    state_path = export_dir / "hi_nerv_mlx_exported_state.npz"
    np.savez(
        state_path,
        **{
            "blocks.0.weight": np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
            "latents_coarse": np.asarray([999.0], dtype=np.float32),
        },
    )
    manifest_path = export_dir / "hi_nerv_mlx_exported_state_npz_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": state_path.as_posix(),
                "artifact_sha256": runner_mod._sha256_file(state_path),
                "tensor_count": 2,
                "consumption_recommended": True,
                "blockers": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ladder_path = tmp_path / "receiver_closed_ladder.json"
    ladder_path.write_text(
        json.dumps(
            {
                "schema": "nerv_receiver_closed_modelsize_ladder.v1",
                "carrier_id": "hi_nerv",
                "axis_tag": "[planning/control]",
                "normalized_rows": [
                    {
                        "row_id": "receiver_smoke",
                        "archive_bytes": archive.stat().st_size,
                        "archive_path": archive.as_posix(),
                        "archive_sha256": runner_mod._sha256_file(archive),
                        "receiver_proof_passed": False,
                        "blockers": ["receiver_closed_byte_proof_missing"],
                        "source": {
                            "row_id": "receiver_smoke",
                            "candidate_id": "receiver_smoke",
                            "archive_bytes": archive.stat().st_size,
                            "archive_path": archive.as_posix(),
                            "archive_sha256": runner_mod._sha256_file(archive),
                            "num_pairs": 1,
                            "receiver_proof_passed": False,
                            "receiver_archive_replay_verified": False,
                            "post_export_receiver_cache_quality": {
                                "quality_gate_passed": False,
                                "quality_gate_verdict": "CACHE_INPUTS_LOCAL_ONLY",
                                "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
                            },
                            "blockers": ["hi_nerv_trained_archive_byte_oracle_partial_pair_scope"],
                        },
                    }
                ],
                "blockers": ["receiver_closed_modelsize_ladder_not_ready"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    saliency_path = tmp_path / "decoder_weight_gradient_saliency.json"
    saliency_path.write_text(
        json.dumps(
            {
                "schema": "mlx_decoder_weight_gradient_saliency.v1",
                "row_count": 1,
                "saliency_by_name": {"blocks.0.weight": 1.25},
                "rows": [
                    {
                        "group_name": "blocks.0.weight",
                        "saliency": 1.25,
                        "numel": 3,
                    }
                ],
                "blockers": [],
                "authority": "macos_mlx_research_signal_false_authority",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = runner_mod._write_hi_nerv_decoder_weight_waterfill_from_trained_ladder(
        output_dir=tmp_path / "waterfill",
        trained_archive_byte_oracle={
            "schema": "hi_nerv_trained_archive_byte_oracle.v1",
            "candidate_id": "receiver_smoke",
            "receiver_closed_modelsize_ladder_path": ladder_path.as_posix(),
        },
        decoder_weight_saliency_artifact={
            "written": True,
            "path": saliency_path.as_posix(),
        },
        action_bits=(0, 2, 32),
    )

    assert out["written"] is True
    assert out["active"] is True
    assert out["candidate_plan_count"] == 1
    bundle = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert bundle["source_schema"] == "nerv_receiver_closed_modelsize_ladder.v1"
    row = bundle["rows"][0]
    assert row["state_npz_manifest_path"] == manifest_path.as_posix()
    assert row["waterfill_plan"]["candidate_id"] == "receiver_smoke:receiver_smoke"
    assert "full_video_coverage_missing" in row["blockers"]
    assert "receiver_closed_byte_proof_missing" in row["blockers"]
    assert "hi_nerv_receiver_cache_quality_is_false_authority" in row["blockers"]
    assert out["score_claim"] is False


def test_hinerv_runner_waterfill_from_trained_ladder_fails_closed_without_saliency(
    tmp_path: Path,
) -> None:
    out = runner_mod._write_hi_nerv_decoder_weight_waterfill_from_trained_ladder(
        output_dir=tmp_path / "waterfill",
        trained_archive_byte_oracle={
            "schema": "hi_nerv_trained_archive_byte_oracle.v1",
            "candidate_id": "candidate_a",
            "receiver_closed_modelsize_ladder_path": (tmp_path / "missing_ladder.json").as_posix(),
        },
        decoder_weight_saliency_artifact={"written": False},
    )

    assert out["written"] is False
    assert "hi_nerv_decoder_weight_saliency_artifact_missing_for_waterfill" in out["blockers"]
    assert "hi_nerv_trained_receiver_closed_modelsize_ladder_missing_for_waterfill" in out["blockers"]


def test_compact_scoreaware_stage_loss_weights_feed_curriculum() -> None:
    weights = runner_mod._compact_scoreaware_stage_loss_weights(
        recon=0.25,
        segnet=2.0,
        pose=1.5,
    )
    stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="unit_hi_nerv",
        epochs=9,
        loss_weights=weights,
    )

    assert weights["recon"] == pytest.approx(0.25)
    assert weights["distill"] == pytest.approx(2.0)
    assert weights["pose_distill"] == pytest.approx(1.5)
    assert weights["scorer_input_guard"] == pytest.approx(1.0)
    assert weights["scorer_input_contrast_floor"] == pytest.approx(1.0)
    assert weights["scorer_input_shape_tether"] == pytest.approx(1.0)
    assert weights["segnet_direct_live_distill"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_class_histogram"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_class_balanced_hinge"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_class_balanced_ce"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_class_balanced_squared_hinge"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_class_region_recon"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_target_mass_floor"] == pytest.approx(2.0)
    assert len(stages) == 1
    assert stages[0].start_epoch == 0
    assert stages[0].end_epoch == 9
    assert dict(stages[0].loss_weights) == weights
    warmup_stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="unit_hi_nerv",
        epochs=9,
        loss_weights=weights,
        pose_distillation_warmup_epochs=3,
    )
    assert len(warmup_stages) == 2
    assert warmup_stages[0].start_epoch == 0
    assert warmup_stages[0].end_epoch == 3
    assert dict(warmup_stages[0].loss_weights) == {
        **weights,
        "pose_distill": 0.0,
        "pose_direct_live_distill": 0.0,
    }
    assert warmup_stages[1].start_epoch == 3
    assert warmup_stages[1].end_epoch == 9
    assert dict(warmup_stages[1].loss_weights) == weights
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="finite"):
        runner_mod._compact_scoreaware_stage_loss_weights(
            recon=-0.1,
            segnet=1.0,
            pose=1.0,
        )
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="smaller than epochs"):
        runner_mod._compact_scoreaware_curriculum_stages(
            substrate_id="unit_hi_nerv",
            epochs=3,
            loss_weights=weights,
            pose_distillation_warmup_epochs=3,
        )


def test_compact_scoreaware_direct_live_escape_warmup_amplifies_class_atoms() -> None:
    weights = runner_mod._compact_scoreaware_stage_loss_weights(
        recon=0.25,
        segnet=2.0,
        pose=1.5,
    )
    stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="unit_hi_nerv",
        epochs=8,
        loss_weights=weights,
        segnet_direct_live_escape_warmup_epochs=3,
        segnet_direct_live_escape_class_multiplier=3.0,
    )

    assert len(stages) == 2
    assert stages[0].start_epoch == 0
    assert stages[0].end_epoch == 3
    assert stages[0].loss_weights["segnet_direct_live_base_loss"] == pytest.approx(0.0)
    assert stages[0].loss_weights["segnet_direct_live_distill"] == pytest.approx(2.0)
    assert stages[0].loss_weights["segnet_direct_live_class_histogram"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_class_balanced_hinge"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_class_balanced_ce"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_class_balanced_squared_hinge"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_class_region_recon"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_rare_class_logit"] == pytest.approx(6.0)
    assert stages[0].loss_weights["segnet_direct_live_target_mass_floor"] == pytest.approx(6.0)
    assert stages[1].start_epoch == 3
    assert stages[1].end_epoch == 8
    assert dict(stages[1].loss_weights) == weights
    with pytest.raises(
        CompactRendererMlxSpineRunnerError,
        match="segnet_direct_live_escape_class_multiplier",
    ):
        runner_mod._compact_scoreaware_curriculum_stages(
            substrate_id="unit_hi_nerv",
            epochs=8,
            loss_weights=weights,
            segnet_direct_live_escape_warmup_epochs=3,
            segnet_direct_live_escape_class_multiplier=0.0,
        )


def test_compact_scoreaware_shape_and_pose_warmups_compose() -> None:
    weights = runner_mod._compact_scoreaware_stage_loss_weights(
        recon=1.0,
        segnet=2.0,
        pose=3.0,
        scorer_input_shape_tether=4.0,
        segnet_direct_live=5.0,
    )
    weights["segnet_direct_live_base_loss"] = 6.0

    stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="unit_hi_nerv",
        epochs=8,
        loss_weights=weights,
        scorer_input_shape_warmup_epochs=3,
        pose_distillation_warmup_epochs=5,
        segnet_direct_live_escape_warmup_epochs=2,
    )

    assert [(stage.start_epoch, stage.end_epoch) for stage in stages] == [
        (0, 2),
        (2, 3),
        (3, 5),
        (5, 8),
    ]
    assert stages[0].loss_weights == {
        **weights,
        "pose_distill": 0.0,
        "pose_direct_live_distill": 0.0,
        "segnet_direct_live_distill": 0.0,
        "segnet_direct_live_class_histogram": 0.0,
        "segnet_direct_live_class_balanced_hinge": 0.0,
        "segnet_direct_live_class_balanced_ce": 0.0,
        "segnet_direct_live_class_balanced_squared_hinge": 0.0,
        "segnet_direct_live_class_region_recon": 0.0,
        "segnet_direct_live_rare_class_logit": 0.0,
        "segnet_direct_live_target_mass_floor": 0.0,
        "segnet_direct_live_target_min_ratio_floor": 0.0,
        "segnet_direct_live_base_loss": 0.0,
    }
    assert stages[1].loss_weights == {
        **weights,
        "pose_distill": 0.0,
        "pose_direct_live_distill": 0.0,
        "segnet_direct_live_distill": 0.0,
        "segnet_direct_live_class_histogram": 0.0,
        "segnet_direct_live_class_balanced_hinge": 0.0,
        "segnet_direct_live_class_balanced_ce": 0.0,
        "segnet_direct_live_class_balanced_squared_hinge": 0.0,
        "segnet_direct_live_class_region_recon": 0.0,
        "segnet_direct_live_rare_class_logit": 0.0,
        "segnet_direct_live_target_mass_floor": 0.0,
        "segnet_direct_live_target_min_ratio_floor": 0.0,
    }
    assert stages[2].loss_weights == {
        **weights,
        "pose_distill": 0.0,
        "pose_direct_live_distill": 0.0,
    }
    assert stages[3].loss_weights == weights


def test_compact_family_interrupted_report_preserves_false_authority_evidence(
    tmp_path: Path,
) -> None:
    out = tmp_path / "snerv_run"
    train_dir = out / "snerv_mlx_native_export"
    train_dir.mkdir(parents=True)
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    telemetry = train_dir / "telemetry.jsonl"
    startup.write_text('{"schema":"compact_carrier_startup_marker.v1"}\n')
    telemetry.write_text('{"epoch":0,"loss":1.0}\n')
    source_video = tmp_path / "0.mkv"
    source_video.write_bytes(b"fake video")
    args = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--planner-row-id",
            "snerv::candidate::optimizer",
            "--output-dir",
            out.as_posix(),
            "--source-video-path",
            source_video.as_posix(),
            "--allow-manual-compact-family-launch",
        ]
    )

    report = runner_mod._write_compact_family_interrupted_report(
        output_dir=out,
        args=args,
        source_video_path=source_video,
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        modelsize_candidate={"candidate_id": "snerv_test", "nominal_bytes": 123},
        signum=signal.SIGTERM,
        reason="unit_test",
    )

    report_path = Path(report["report_path"])
    payload = json.loads(report_path.read_text())
    assert payload["mode"] == "interrupted_compact_family_run"
    assert payload["signal_name"] == "SIGTERM"
    assert payload["training_executed"] is False
    assert payload["training_started"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "snerv_training_interrupted_before_export" in payload["blockers"]
    evidence_paths = {row["path"] for row in payload["evidence_files"]}
    assert startup.as_posix() in evidence_paths
    assert telemetry.as_posix() in evidence_paths
    assert all("sha256" in row for row in payload["evidence_files"])
    assert Path(payload["candidate_feedback"]["row_path"]).is_file()
    assert payload["candidate_feedback"]["row"]["schema"] == "nerv_candidate_feedback_row.v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_recover_interrupted_report_from_startup_marker_summarizes_telemetry(
    tmp_path: Path,
) -> None:
    out = tmp_path / "hi_nerv_run"
    train_dir = out / "hi_nerv_mlx_training"
    train_dir.mkdir(parents=True)
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "pid": 123,
                "execute_family": "hi_nerv",
                "planner_row_id": "hi_nerv::hinerv_np600::adamw",
                "modelsize_candidate_id": "hinerv_np600",
                "modelsize_candidate": {"candidate_id": "hinerv_np600"},
                "output_dir": out.as_posix(),
                "source_video_path": "/Volumes/VertigoDataTier/pact/upstream/videos/0.mkv",
                "hard_byte_ceilings": [178_000, 285_000],
                "command_args": {"execute_family": "hi_nerv"},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry = train_dir / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 0, "loss": 9.0}, sort_keys=True),
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-03T06:19:19Z",
                        "epoch": 26805,
                        "learning_rate": 2.7e-5,
                        "loss": 25.21,
                        "loss_components": {
                            "loss_part_distill": 5.64,
                            "loss_part_pose_distill": 2.52,
                            "loss_part_pr95_c1a_entropy": 6.68,
                            "pr95_stage_index": 8.0,
                            "pr95_stage_uses_muon": 1.0,
                        },
                        "per_axis_decomposition": {
                            "pose": 2.28,
                            "seg": 5.65,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_meta = checkpoint_dir / "epoch026805_20260603T061919Z.meta.json"
    checkpoint_live = checkpoint_dir / "epoch026805_20260603T061919Z.live.state.npsd"
    checkpoint_ema = checkpoint_dir / "epoch026805_20260603T061919Z.ema_shadow.state.npsd"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "schema_version": "long_training_canonical_checkpoint.v1",
                "global_epoch": 26805,
                "live_state_path": checkpoint_live.as_posix(),
                "ema_shadow_state_path": checkpoint_ema.as_posix(),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_live.write_bytes(b"live-state")
    checkpoint_ema.write_bytes(b"ema-state")

    report = runner_mod._write_compact_family_interrupted_report_from_startup_marker(
        output_dir=out,
        reason="unit_test_recovery",
    )

    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    telemetry_summary = payload["telemetry_summary"]
    assert payload["mode"] == "recovered_interrupted_compact_family_run"
    assert payload["recovered"] is True
    assert payload["execute_family"] == "hi_nerv"
    assert payload["training_started"] is True
    assert payload["training_executed"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "hi_nerv_training_interrupted_before_export" in payload["blockers"]
    assert telemetry_summary["row_count"] == 2
    assert telemetry_summary["last_epoch"] == 26805
    assert telemetry_summary["last_loss_components"]["pr95_stage_index"] == 8.0
    assert telemetry_summary["last_loss_components"]["pr95_stage_uses_muon"] == 1.0
    assert telemetry_summary["last_per_axis_decomposition"]["pose"] == 2.28
    evidence_paths = {row["path"] for row in payload["evidence_files"]}
    assert startup.as_posix() in evidence_paths
    assert telemetry.as_posix() in evidence_paths
    assert checkpoint_meta.as_posix() in evidence_paths
    assert checkpoint_live.as_posix() in evidence_paths
    assert checkpoint_ema.as_posix() in evidence_paths
    assert Path(payload["candidate_feedback"]["row_path"]).is_file()
    assert payload["candidate_feedback"]["row"]["schema"] == "nerv_candidate_feedback_row.v1"


def test_recover_interrupted_report_summarizes_snerv_nested_long_training_telemetry(
    tmp_path: Path,
) -> None:
    out = tmp_path / "snerv_run"
    train_dir = (
        out / "snerv_mlx_native_export" / "native_train_export" / "snerv_score_aware_long_training" / "long_training"
    )
    train_dir.mkdir(parents=True)
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "pid": 123,
                "execute_family": "snerv",
                "planner_row_id": "snerv::candidate::pact_muon_adamw",
                "modelsize_candidate_id": "snerv_candidate",
                "modelsize_candidate": {
                    "candidate_id": "snerv_candidate",
                    "num_pairs": 600,
                },
                "output_dir": out.as_posix(),
                "source_video_path": "/Volumes/VertigoDataTier/pact/upstream/videos/0.mkv",
                "hard_byte_ceilings": [178_000],
                "command_args": {"execute_family": "snerv", "num_pairs": 16},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry = train_dir / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 0, "loss": 195.0}, sort_keys=True),
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-05T17:30:58Z",
                        "epoch": 1,
                        "learning_rate": 1e-3,
                        "loss": 713.17,
                        "loss_components": {
                            "dual_ascent_metric__snerv_posenet_yuv6_pair_distill": 100.7159,
                            "dual_ascent_metric__snerv_scorer_input_distribution_guard": 3.585,
                            "dual_ascent_metric__snerv_segnet_last_frame_distill": 0.2474,
                            "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 6.0,
                            "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.365,
                            "dual_ascent_missing_metric__snerv_coder_qat_quant_residual": 0.0,
                            "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                            "dual_ascent_missing_metric__snerv_scorer_input_distribution_guard": 0.0,
                            "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                            "loss_part_coder_qat_quant_residual": 0.03918,
                            "loss_part_pr95_stage_forced_extra_qat_active": 1.0,
                            "loss_part_pr95_stage_scorer_input_distribution_guard": 3.585,
                            "loss_part_weighted_coder_qat_quant_residual": 0.000064,
                            "train_time_section_bytes__decoder_payload": 13_518.0,
                            "train_time_section_bytes__lf_payload": 388.0,
                        },
                        "per_axis_decomposition": {
                            "pose": 399.86,
                            "seg": 6.15,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = runner_mod._write_compact_family_interrupted_report_from_startup_marker(
        output_dir=out,
        reason="unit_test_snerv_recovery",
    )

    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    telemetry_summary = payload["telemetry_summary"]
    loss_components = telemetry_summary["last_loss_components"]
    assert payload["mode"] == "recovered_interrupted_compact_family_run"
    assert payload["execute_family"] == "snerv"
    assert payload["training_executed"] is True
    assert "snerv_training_interrupted_before_export" in payload["blockers"]
    assert telemetry_summary["row_count"] == 2
    assert telemetry_summary["last_epoch"] == 1
    assert telemetry_summary["path"] == telemetry.as_posix()
    assert payload["score_aware_long_training_telemetry_contract"]["passed"] is True
    assert loss_components["loss_part_pr95_stage_forced_extra_qat_active"] == 1.0
    assert loss_components["loss_part_coder_qat_quant_residual"] == 0.03918
    assert loss_components["dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill"] == 0.0
    assert loss_components["train_time_section_bytes__decoder_payload"] == 13_518.0
    evidence_paths = {row["path"] for row in payload["evidence_files"]}
    assert startup.as_posix() in evidence_paths
    assert telemetry.as_posix() in evidence_paths
    assert Path(payload["candidate_feedback"]["row_path"]).is_file()
    assert payload["candidate_feedback"]["row"]["schema"] == "nerv_candidate_feedback_row.v1"
    feedback_row = payload["candidate_feedback"]["row"]
    assert feedback_row["candidate_id"] == "snerv_candidate"
    assert feedback_row["feedback_kind"] == "score_aware_training_telemetry"
    assert feedback_row["candidate_num_pairs"] == 600
    assert feedback_row["measured_num_pairs"] == 16
    assert feedback_row["scope_matches_candidate"] is False
    assert feedback_row["feedback_scope"] == "bounded_score_aware_training_telemetry"
    assert feedback_row["snerv_scorer_domain_tether_health"]["passed"] is True
    assert feedback_row["snerv_scorer_domain_tether_passed"] is True
    assert feedback_row["snerv_scorer_input_distribution_guard_proof"]["passed"] is True
    assert feedback_row["snerv_renderer_nondegenerate_proof"]["passed"] is False


@pytest.mark.parametrize("family", ("hi_nerv", "snerv"))
def test_recover_interrupted_report_preserves_startup_rerun_provenance(
    tmp_path: Path,
    family: str,
) -> None:
    out = tmp_path / f"{family}_run"
    out.mkdir()
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    original_argv = [
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        family,
        "--output-dir",
        out.as_posix(),
        "--num-pairs",
        "16",
        "--epochs",
        "1",
    ]
    rerun_argv = [
        sys.executable,
        (REPO_ROOT / "tools/run_compact_renderer_mlx_spine_runner.py").as_posix(),
        "--execute-family",
        family,
        "--output-dir",
        out.as_posix(),
        "--num-pairs",
        "16",
        "--epochs",
        "1",
        "--overwrite",
    ]
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "pid": 456,
                "execute_family": family,
                "planner_row_id": f"{family}::unit::rerun",
                "modelsize_candidate_id": f"{family}_candidate",
                "modelsize_candidate": {"candidate_id": f"{family}_candidate"},
                "output_dir": out.as_posix(),
                "source_video_path": ("/Volumes/VertigoDataTier/pact/upstream/videos/0.mkv"),
                "hard_byte_ceilings": [178_000],
                "command_args": {"execute_family": family, "num_pairs": 16},
                "original_argv": original_argv,
                "direct_smoke_rerun_argv": rerun_argv,
                "runner_invocation_provenance": {
                    "schema": "compact_runner_invocation_provenance.v1",
                    "source": "startup_marker",
                    "original_argv": original_argv,
                    "original_arg_tokens": original_argv[1:],
                    "same_output_dir_rerun": {
                        "argv": rerun_argv,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = runner_mod._write_compact_family_interrupted_report_from_startup_marker(
        output_dir=out,
        reason="unit_test_rerun_recovery",
    )

    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    provenance = payload["runner_invocation_provenance"]
    assert payload["mode"] == "recovered_interrupted_compact_family_run"
    assert payload["execute_family"] == family
    assert payload["original_argv"] == original_argv
    assert payload["direct_smoke_rerun_argv"] == rerun_argv
    assert provenance["source"] == "startup_marker_recovery"
    assert provenance["same_output_dir_rerun"]["argv"] == rerun_argv
    assert provenance["recovered_from_startup_marker"] is True
    assert provenance["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert f"{family}_training_interrupted_before_export" in payload["blockers"]


def test_hinerv_training_telemetry_contract_accepts_nested_control_metrics(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.25,
                    "loss_part_pose_distill": 2.5,
                    "loss_part_pose_score_term": 5.0,
                    "loss_part_pr95_stage_seg_surrogate": 1.25,
                    "loss_part_pr95_stage_pose_surrogate": 5.0,
                    "loss_part_pr95_stage_scorer_input_distribution_guard": 0.125,
                    "loss_part_pr95_stage_scorer_input_shape_tether": 0.25,
                    "loss_part_pr95_stage_scorer_input_shape_tether_segnet_last_rgb": 0.0625,
                    "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_pair": 0.09375,
                    "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.09375,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether": 0.1875,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether_pair": 0.09375,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether_temporal_delta": 0.09375,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor": 0.125,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_std_ratio": 0.3125,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_abs_ratio": 0.28125,
                    "loss_part_pr95_stage_segnet_direct_live_distill": 0.03125,
                    "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement": 0.75,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
                    "loss_part_segnet_direct_live_class_histogram_loss": 0.021,
                    "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.022,
                    "loss_part_segnet_direct_live_class_balanced_ce_loss": 0.023,
                    "loss_part_segnet_direct_live_class_balanced_squared_hinge_loss": 0.024,
                    "loss_part_segnet_direct_live_class_region_recon_loss": 0.025,
                    "loss_part_segnet_direct_live_target_mass_floor_loss": 0.026,
                    "loss_part_pr95_stage_segnet_direct_live_target_mass_floor_loss": 0.026,
                    "loss_part_segnet_direct_live_target_min_ratio_floor_loss": 0.027,
                    "loss_part_pr95_stage_segnet_direct_live_target_min_ratio_floor_loss": 0.027,
                    "loss_part_weighted_pr95_stage_segnet_direct_live_distill": 0.015625,
                    "segnet_student_live_calibration_active": 1.0,
                    "loss_part_segnet_student_live_calibration": 0.0625,
                    "loss_part_weighted_segnet_student_live_calibration": 0.0625,
                    "train_time_archive_rate_score": 0.01,
                    "train_time_section_rate_score__decoder_payload": 0.002,
                    "dual_ascent_missing_metric__hi_nerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_scorer_input_distribution_guard": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_histogram": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_missing_fraction_histogram": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_ce": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_missing_fraction_ce": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_squared_hinge": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_region_recon": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_last_frame_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_posenet_yuv6_pair_distill": 0.25,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_histogram": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_missing_fraction_histogram": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_ce": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_missing_fraction_ce": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_squared_hinge": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_region_recon": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_mass_floor": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_floor": 0.03125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate": 0.03125,
                    "dual_ascent_lambda__hi_nerv_archive_total_bytes": 0.375,
                    "dual_ascent_lambda__hi_nerv_decoder_payload_section_bytes": 0.5,
                    "dual_ascent_weight_applied__hi_nerv_archive_total_bytes": 1.0,
                    "dual_ascent_weight_applied__hi_nerv_decoder_payload_section_bytes": 1.0,
                    "gradient_multiplier_requested_control_count": 1.0,
                    "gradient_multiplier_applied_leaf_count": 1.0,
                    "gradient_multiplier_missing_requested_count": 0.0,
                    "gradient_multiplier_requested_but_unapplied": 0.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=1.0,
        segnet_direct_live_distillation_weight=0.5,
        segnet_direct_live_class_histogram_weight=0.25,
        segnet_direct_live_class_balanced_hinge_weight=0.5,
        segnet_direct_live_class_balanced_ce_weight=0.25,
        segnet_direct_live_class_balanced_squared_hinge_weight=0.25,
        segnet_direct_live_class_region_recon_weight=0.25,
        segnet_direct_live_target_mass_floor_weight=0.25,
        segnet_direct_live_target_min_ratio_floor_weight=0.25,
        pr95_faithful_curriculum_enabled=True,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=2.0,
        scorer_input_shape_tether_weight=1.0,
        posenet_yuv6_geometry_tether_weight=1.0,
        posenet_temporal_signal_floor_weight=1.0,
        gradient_multiplier_controls_requested=True,
    )

    assert contract["passed"] is True
    assert contract["blockers"] == []
    assert contract["segnet_dual_metric_observed"] is True
    assert contract["posenet_dual_metric_observed"] is True
    assert contract["segnet_dual_lambda_active_observed"] is True
    assert contract["posenet_dual_lambda_active_observed"] is True
    assert contract["scorer_input_guard_dual_metric_observed"] is True
    assert contract["archive_rate_metric_observed"] is True
    assert contract["archive_byte_dual_lambda_active_observed"] is True
    assert contract["archive_byte_dual_weight_applied_observed"] is True
    assert contract["section_rate_metric_observed"] is True
    assert contract["section_byte_dual_lambda_active_observed"] is True
    assert contract["section_byte_dual_weight_applied_observed"] is True
    assert contract["section_byte_dual_zero_base_masked_observed"] is False
    assert contract["expected_gradient_multiplier_controls"] is True
    assert contract["gradient_multiplier_requested_observed"] is True
    assert contract["gradient_multiplier_applied_observed"] is True
    assert contract["gradient_multiplier_missing_requested_observed"] is False
    assert contract["gradient_multiplier_noop_observed"] is False
    assert contract["scorer_input_guard_metric_observed"] is True
    assert contract["expected_scorer_input_shape_tether_metric"] is True
    assert contract["scorer_input_shape_tether_metric_observed"] is True
    assert contract["scorer_input_shape_tether_segnet_metric_observed"] is True
    assert contract["scorer_input_shape_tether_posenet_pair_metric_observed"] is True
    assert contract["scorer_input_shape_tether_posenet_delta_metric_observed"] is True
    assert contract["expected_posenet_yuv6_geometry_tether_metric"] is True
    assert contract["posenet_yuv6_geometry_tether_metric_observed"] is True
    assert contract["posenet_yuv6_geometry_tether_pair_metric_observed"] is True
    assert contract["posenet_yuv6_geometry_tether_delta_metric_observed"] is True
    assert contract["expected_posenet_temporal_signal_floor_metric"] is True
    assert contract["posenet_temporal_signal_floor_metric_observed"] is True
    assert contract["posenet_temporal_signal_floor_std_ratio_metric_observed"] is True
    assert contract["posenet_temporal_signal_floor_mean_abs_ratio_metric_observed"] is True
    assert contract["segnet_live_calibration_active_observed"] is True
    assert contract["segnet_live_calibration_loss_observed"] is True
    assert contract["expected_segnet_direct_live_distillation"] is True
    assert contract["expected_segnet_direct_live_subcontrols"] == {
        "class_balanced_ce": True,
        "class_balanced_hinge": True,
        "class_balanced_squared_hinge": True,
        "class_histogram": True,
        "class_region_recon": True,
        "rare_class_logit": False,
        "target_mass_floor": True,
        "target_min_ratio_floor": True,
    }
    assert contract["segnet_direct_live_distillation_loss_observed"] is True
    assert contract["segnet_direct_live_argmax_metric_observed"] is True
    assert contract["segnet_direct_live_class_occupancy_metric_observed"] is True
    assert contract["segnet_direct_live_class_histogram_metric_observed"] is True
    assert contract["segnet_direct_live_class_balanced_hinge_metric_observed"] is True
    assert contract["segnet_direct_live_class_balanced_ce_metric_observed"] is True
    assert contract["segnet_direct_live_class_balanced_squared_hinge_metric_observed"] is True
    assert contract["segnet_direct_live_class_region_recon_metric_observed"] is True
    assert contract["segnet_direct_live_target_mass_floor_metric_observed"] is True
    assert contract["segnet_direct_live_target_min_ratio_floor_metric_observed"] is True
    assert contract["segnet_direct_live_dual_metric_observed"] is True
    assert contract["segnet_direct_live_dual_lambda_active_observed"] is True
    assert contract["missing_segnet_direct_live_dual_metric_suffixes"] == []
    assert contract["missing_segnet_direct_live_dual_activation_suffixes"] == []
    assert contract["segnet_direct_live_max_candidate_occupied_class_fraction"] == pytest.approx(0.6)
    assert contract["segnet_direct_live_target_class_coverage_metric_observed"] is True
    assert contract["segnet_direct_live_max_candidate_target_class_coverage_fraction"] == pytest.approx(1.0)
    assert contract["segnet_direct_live_target_class_min_ratio_metric_observed"] is True
    assert contract["segnet_direct_live_max_candidate_target_class_min_ratio"] == pytest.approx(0.25)


def test_hinerv_training_telemetry_contract_tracks_source_pairs(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "batch_observability": {
                    "train_batch": {
                        "actual_batch_size": 2,
                        "source_pair_indices": [1, 0],
                    },
                },
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.03125,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["train_source_pair_indices_observed"] == [0, 1]
    assert contract["train_source_pair_count_observed"] == 2
    assert contract["train_max_actual_batch_size_observed"] == 2


def test_hinerv_training_telemetry_contract_uses_support_ladder_effective_controls(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 3,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 1.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.3,
                    "loss_part_segnet_direct_live_target_mass_floor_loss": 2.0,
                    "loss_part_segnet_direct_live_rare_class_logit_loss": 3.0,
                    "loss_part_segnet_direct_live_class_balanced_hinge_loss": 4.0,
                    "active_loss_weight__segnet_direct_live_target_mass_floor": 2.0,
                    "active_loss_weight__segnet_direct_live_rare_class_logit": 1.0,
                    "active_loss_weight__segnet_direct_live_class_balanced_hinge": 0.5,
                    "active_loss_weight_positive__segnet_direct_live_target_mass_floor": 1.0,
                    "active_loss_weight_positive__segnet_direct_live_rare_class_logit": 1.0,
                    "active_loss_weight_positive__segnet_direct_live_class_balanced_hinge": 1.0,
                    "scorer_support_ladder_enabled": 1.0,
                    "scorer_support_ladder_active": 1.0,
                    "scorer_support_ladder_component_active__segnet_direct_live_target_mass_floor": 1.0,
                    "scorer_support_ladder_component_active__segnet_direct_live_rare_class_logit": 1.0,
                    "scorer_support_ladder_component_active__segnet_direct_live_class_balanced_hinge": 1.0,
                    "scorer_support_ladder_component_weight__segnet_direct_live_target_mass_floor": 2.0,
                    "scorer_support_ladder_component_weight__segnet_direct_live_rare_class_logit": 1.0,
                    "scorer_support_ladder_component_weight__segnet_direct_live_class_balanced_hinge": 0.5,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_rare_class_logit": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_mass_floor": 0.5,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.5,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit": 0.5,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_rare_class_logit": 0.5,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.5,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        scorer_support_ladder_enabled=True,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["blockers"] == []
    assert contract["expected_scorer_support_ladder"] is True
    assert contract["scorer_support_ladder_enabled_observed"] is True
    assert contract["scorer_support_ladder_active_observed"] is True
    assert contract["expected_segnet_direct_live_subcontrols"]["target_mass_floor"] is True
    assert contract["expected_segnet_direct_live_subcontrols"]["rare_class_logit"] is True
    assert contract["expected_segnet_direct_live_subcontrols"]["class_balanced_hinge"] is True
    assert contract["segnet_direct_live_effective_subcontrol_weights"]["target_mass_floor"] == pytest.approx(2.0)
    assert contract["segnet_direct_live_effective_subcontrol_weights"]["rare_class_logit"] == pytest.approx(1.0)
    assert contract["segnet_direct_live_effective_subcontrol_weights"]["class_balanced_hinge"] == pytest.approx(0.5)
    assert "segnet_direct_live_target_min_ratio_mass_floor" in contract["expected_segnet_direct_live_dual_suffixes"]
    assert (
        "segnet_direct_live_target_min_ratio_rare_class_logit" in contract["expected_segnet_direct_live_dual_suffixes"]
    )


def test_hinerv_training_telemetry_contract_rejects_empty_active_support_ladder(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 1,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 1.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.3,
                    "scorer_support_ladder_enabled": 1.0,
                    "scorer_support_ladder_active": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        scorer_support_ladder_enabled=True,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["scorer_support_ladder_active_observed"] is True
    assert "hi_nerv_score_aware_training_scorer_support_ladder_effective_subcontrol_missing" in contract["blockers"]


def test_hinerv_training_telemetry_contract_allows_under_budget_byte_duals(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "train_time_archive_rate_score": 0.01,
                    "train_time_section_rate_score__decoder_payload": 0.002,
                    "dual_ascent_violation__hi_nerv_archive_total_bytes": -0.03,
                    "dual_ascent_update_count__hi_nerv_archive_total_bytes": 1.0,
                    "dual_ascent_violation__hi_nerv_decoder_payload_section_bytes": -0.02,
                    "dual_ascent_update_count__hi_nerv_decoder_payload_section_bytes": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["archive_byte_dual_positive_violation_observed"] is False
    assert contract["archive_byte_dual_update_observed"] is True
    assert contract["section_byte_dual_positive_violation_observed"] is False
    assert contract["section_byte_dual_update_observed"] is True
    assert contract["archive_byte_dual_lambda_active_observed"] is False
    assert contract["section_byte_dual_lambda_active_observed"] is False
    assert contract["blockers"] == []


def test_hinerv_training_telemetry_contract_rejects_section_rate_without_section_lambda(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "train_time_archive_rate_score": 0.01,
                    "train_time_section_rate_score__decoder_payload": 0.002,
                    "dual_ascent_lambda__hi_nerv_archive_total_bytes": 0.375,
                    "dual_ascent_violation__hi_nerv_decoder_payload_section_bytes": 0.004,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["section_rate_metric_observed"] is True
    assert contract["section_byte_dual_positive_violation_observed"] is True
    assert contract["section_byte_dual_lambda_active_observed"] is False
    assert "hi_nerv_score_aware_training_section_byte_dual_lambda_never_active" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_lambda_without_weight_application(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "train_time_archive_rate_score": 0.01,
                    "train_time_section_rate_score__decoder_payload": 0.002,
                    "dual_ascent_lambda__hi_nerv_archive_total_bytes": 0.375,
                    "dual_ascent_violation__hi_nerv_archive_total_bytes": 0.005,
                    "dual_ascent_lambda__hi_nerv_decoder_payload_section_bytes": 0.5,
                    "dual_ascent_violation__hi_nerv_decoder_payload_section_bytes": 0.004,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["archive_byte_dual_lambda_active_observed"] is True
    assert contract["archive_byte_dual_weight_applied_observed"] is False
    assert contract["section_byte_dual_lambda_active_observed"] is True
    assert contract["section_byte_dual_weight_applied_observed"] is False
    assert "hi_nerv_score_aware_training_archive_byte_dual_weight_never_applied" in contract["blockers"]
    assert "hi_nerv_score_aware_training_section_byte_dual_weight_never_applied" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_stale_gradient_multiplier(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "gradient_multiplier_requested_control_count": 1.0,
                    "gradient_multiplier_applied_leaf_count": 0.0,
                    "gradient_multiplier_missing_requested_count": 1.0,
                    "gradient_multiplier_requested_but_unapplied": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        gradient_multiplier_controls_requested=True,
    )

    assert contract["passed"] is False
    assert contract["gradient_multiplier_requested_observed"] is True
    assert contract["gradient_multiplier_applied_observed"] is False
    assert contract["gradient_multiplier_missing_requested_observed"] is True
    assert contract["gradient_multiplier_noop_observed"] is True
    assert "hi_nerv_score_aware_training_gradient_multiplier_never_applied" in contract["blockers"]
    assert "hi_nerv_score_aware_training_gradient_multiplier_missing_requested_leaf" in contract["blockers"]
    assert "hi_nerv_score_aware_training_gradient_multiplier_requested_but_unapplied" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_missing_direct_live_metrics(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.25,
                    "loss_part_pose_distill": 2.5,
                    "loss_part_pose_score_term": 5.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill": 0.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert "hi_nerv_score_aware_training_direct_live_segnet_loss_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_direct_live_segnet_argmax_metric_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_direct_live_segnet_class_occupancy_metric_missing" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_missing_direct_live_subcontrol_metric(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_class_region_recon_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["expected_segnet_direct_live_distillation"] is True
    assert contract["expected_segnet_direct_live_subcontrols"]["class_region_recon"] is True
    assert contract["segnet_direct_live_class_region_recon_metric_observed"] is False
    assert "hi_nerv_score_aware_training_direct_live_segnet_class_region_recon_metric_missing" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_missing_shape_tether_metrics(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_scorer_input_shape_tether": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_shape_tether_weight=1.0,
    )

    assert contract["passed"] is False
    assert contract["scorer_input_shape_tether_metric_observed"] is True
    assert "hi_nerv_score_aware_training_scorer_input_shape_tether_segnet_metric_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_scorer_input_shape_tether_posenet_pair_metric_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_scorer_input_shape_tether_posenet_delta_metric_missing" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_missing_temporal_floor_ratios(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_posenet_temporal_signal_floor": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        posenet_temporal_signal_floor_weight=1.0,
    )

    assert contract["passed"] is False
    assert contract["posenet_temporal_signal_floor_metric_observed"] is True
    assert contract["posenet_temporal_signal_floor_std_ratio_metric_observed"] is False
    assert contract["posenet_temporal_signal_floor_mean_abs_ratio_metric_observed"] is False
    assert "hi_nerv_score_aware_training_posenet_temporal_signal_floor_std_ratio_metric_missing" in contract["blockers"]
    assert (
        "hi_nerv_score_aware_training_posenet_temporal_signal_floor_mean_abs_ratio_metric_missing"
        in contract["blockers"]
    )


def test_hinerv_training_telemetry_contract_rejects_missing_geometry_tether_parts(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_posenet_yuv6_geometry_tether": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        posenet_yuv6_geometry_tether_weight=1.0,
    )

    assert contract["passed"] is False
    assert contract["posenet_yuv6_geometry_tether_metric_observed"] is True
    assert contract["posenet_yuv6_geometry_tether_pair_metric_observed"] is False
    assert contract["posenet_yuv6_geometry_tether_delta_metric_observed"] is False
    assert "hi_nerv_score_aware_training_posenet_yuv6_geometry_tether_pair_metric_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_posenet_yuv6_geometry_tether_delta_metric_missing" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_direct_live_class_collapse(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 2.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["segnet_direct_live_max_candidate_occupied_class_fraction"] == pytest.approx(0.4)
    assert "hi_nerv_score_aware_training_direct_live_segnet_candidate_argmax_collapsed" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_direct_live_target_class_collapse(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 2.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["segnet_direct_live_max_candidate_occupied_class_fraction"] == pytest.approx(0.8)
    assert contract["segnet_direct_live_max_candidate_target_class_coverage_fraction"] == pytest.approx(0.4)
    assert "hi_nerv_score_aware_training_direct_live_segnet_candidate_argmax_collapsed" not in contract["blockers"]
    assert "hi_nerv_score_aware_training_direct_live_segnet_target_class_coverage_collapsed" in contract["blockers"]


def test_hinerv_training_telemetry_contract_rejects_direct_live_target_mass_collapse(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 2.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["segnet_direct_live_max_candidate_target_class_coverage_fraction"] == pytest.approx(1.0)
    assert contract["segnet_direct_live_max_candidate_target_class_min_ratio"] == pytest.approx(0.05)
    assert "hi_nerv_score_aware_training_direct_live_segnet_target_class_mass_collapsed" in contract["blockers"]


def test_hinerv_training_telemetry_contract_uses_configured_class_floor(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 2.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
                    "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.125,
                    "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        min_segnet_direct_live_occupied_class_fraction_for_fit_gate=0.4,
    )

    assert contract["passed"] is True
    assert contract["min_segnet_direct_live_occupied_class_fraction_for_fit_gate"] == pytest.approx(0.4)
    assert "hi_nerv_score_aware_training_direct_live_segnet_candidate_argmax_collapsed" not in contract["blockers"]


def test_hinerv_training_telemetry_contract_requires_direct_live_posenet_dual(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_pose_direct_live_distill": 2.0,
                    "loss_part_pose_direct_live_raw_mse": 4.0,
                    "loss_part_pose_direct_live_score_term": 2.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    blocked = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_direct_live_distillation_weight=0.5,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )
    assert blocked["passed"] is False
    assert "hi_nerv_score_aware_training_direct_live_posenet_dual_metric_never_observed" in blocked["blockers"]
    assert "hi_nerv_score_aware_training_direct_live_posenet_dual_lambda_never_active" in blocked["blockers"]

    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_pose_direct_live_distill": 2.0,
                    "loss_part_pose_direct_live_raw_mse": 4.0,
                    "loss_part_pose_direct_live_score_term": 2.0,
                    "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_update_count__hi_nerv_posenet_yuv6_pair_distill": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    passed = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_direct_live_distillation_weight=0.5,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )
    assert passed["passed"] is True
    assert passed["posenet_direct_live_dual_metric_observed"] is True
    assert passed["posenet_direct_live_dual_lambda_active_observed"] is True
    assert passed["missing_posenet_direct_live_dual_metric_suffixes"] == []
    assert passed["missing_posenet_direct_live_dual_activation_suffixes"] == []


def test_hinerv_training_telemetry_contract_rejects_pr95_alias_staleness(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_pr95_stage_seg_surrogate": 9.0,
                    "loss_part_pr95_stage_pose_surrogate": 7.0,
                    "dual_ascent_missing_metric__hi_nerv_segnet_last_frame_distill": 1.0,
                    "dual_ascent_missing_metric__hi_nerv_posenet_yuv6_pair_distill": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = runner_mod._compact_score_aware_training_telemetry_contract(
        telemetry,
        family="hi_nerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=1.0,
        pr95_faithful_curriculum_enabled=True,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=2.0,
    )

    assert contract["passed"] is False
    assert "hi_nerv_score_aware_training_pr95_seg_alias_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_pr95_pose_alias_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_dual_segnet_metric_never_observed" in contract["blockers"]
    assert "hi_nerv_score_aware_training_dual_scorer_input_guard_metric_never_observed" in contract["blockers"]
    assert "hi_nerv_score_aware_training_section_rate_metric_missing" in contract["blockers"]
    assert "hi_nerv_score_aware_training_live_segnet_calibration_never_active" in contract["blockers"]
    assert "hi_nerv_score_aware_training_live_segnet_calibration_loss_missing" in contract["blockers"]


def test_write_decoder_weight_saliency_artifact_missing_fails_closed(
    tmp_path: Path,
) -> None:
    out = runner_mod._write_decoder_weight_saliency_artifact(
        artifact_dict={},
        output_dir=tmp_path / "hi_nerv_mlx_training",
        family="hi_nerv",
    )

    assert out["written"] is False
    assert out["reason"] == "decoder_weight_gradient_saliency_missing"
    assert out["authority"] == "macos_mlx_research_signal_false_authority"


def _write_mlx_prefilter_profile(
    path: Path,
    *,
    pairs: int,
    batch_pairs: int,
    score: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                "max_pairs": int(pairs),
                "num_pairs": int(pairs),
                "n_samples": int(pairs),
                "scorer_batch_pairs": int(batch_pairs),
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": float(score)},
                "mlx_response_summary": {
                    "batch_pairs": int(batch_pairs),
                    "max_pairs": int(pairs),
                    "n_samples": int(pairs),
                },
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_verified_joint_recon_weight(
    root: Path,
    *,
    pairs: int,
    name: str,
) -> tuple[Path, Path]:
    out = root / "experiments" / "results" / name
    out.mkdir(parents=True)
    weight = np.ones((pairs, 2, 384, 512, 1), dtype=np.float32)
    weight_path = out / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(weight_path, weight=weight)
    sha = runner_mod._sha256_file(weight_path)
    manifest_path = out / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": sha,
                "config": {
                    "num_pairs": pairs,
                    "scorer_hw": [384, 512],
                },
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                    "gradient_health": {
                        "schema": "joint_recon_pixel_weight_gradient_health.v1",
                        "status": "pass_finite",
                        "component_count": 1,
                        "components_with_nonfinite": 0,
                        "total_nonfinite_values": 0,
                        "consumption_recommended": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return weight_path, manifest_path


def test_planner_row_launch_gate_rejects_manual_hinerv_without_row_id(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="",
        allow_manual_compact_family_launch=False,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert blockers == [
        "hi_nerv_planner_row_id_missing",
        ("top_priority_compact_carrier_launch_must_come_from_nerv_long_training_campaign_plan"),
    ]

    report = runner_mod._write_planner_row_launch_refusal(
        output_dir=tmp_path,
        args=args,
        blockers=blockers,
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        repo_root=REPO_ROOT,
    )
    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert payload["mode"] == "compact_carrier_planner_row_launch_refused"
    assert payload["trainer_launch_allowed"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["planner_launch_contract"]["planner_row_id"] is None
    assert payload["planner_launch_contract"]["allow_manual_compact_family_launch"] is False
    assert payload["blockers"] == blockers


def _write_planner_row_queue_artifact(
    path: Path,
    *,
    schema: str = "experiment_queue.v1",
    family: str = "hi_nerv",
    row_id: str = "hi_nerv::candidate::adamw",
    status: str = "queued",
    blocked: bool = False,
    runnable_contract: bool = True,
    command_extra: list[str] | None = None,
) -> Path:
    payload = {
        "schema": schema,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [
            {
                "id": "unit_planner_row",
                "family": family,
                "status": status,
                "blocked": blocked,
                "launch_authority_contract": {
                    "schema": "nerv_long_training_queue_launch_authority_contract.v1",
                    "queue_status_is_local_mlx_plan": True,
                    "queue_status_is_runnable_plan": runnable_contract,
                    "queue_launch_blockers": [] if runnable_contract else ["unit_not_runnable"],
                    "queue_status_is_receiver_proof": False,
                    "queue_status_is_cpu_replay_proof": False,
                    "queue_status_is_exact_eval_authority": False,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "run_mlx_first_campaign_row",
                        "command": [
                            "python",
                            "tools/run_compact_renderer_mlx_spine_runner.py",
                            "--execute-family",
                            family,
                            "--planner-row-id",
                            row_id,
                            *(command_extra or []),
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_snerv_lf_hf_replacement_queue_artifact(
    path: Path,
    *,
    row_id: str = "snerv_lf_hf_replace_unit",
    status: str = "local_bounded_smoke_ready_no_authority",
    blocked: bool = False,
    runnable_contract: bool = True,
    solution_family: str = "official_tub_lf_hf_decoder_replacement",
    include_bounded_training_contract: bool = True,
    bounded_training_contract_bound: bool = True,
    include_command: bool = True,
    include_unblock_command: bool = False,
    unblock_contract_runnable: bool = True,
    command_extra: list[str] | None = None,
) -> Path:
    bounded_training_contract = {
        "schema": "snerv_lf_hf_bounded_training_binding_contract.v1",
        "solution_family": solution_family,
        "runner_actuator_required": True,
        "runner_actuator_bound": bounded_training_contract_bound,
        "runner_actuator": {
            "kind": "bounded_snerv_training_smoke",
            "runner": "tools/run_compact_renderer_mlx_spine_runner.py",
            "consumes_queue_artifact": True,
        }
        if bounded_training_contract_bound
        else None,
        "blockers": []
        if bounded_training_contract_bound
        else ["snerv_lf_conditioned_hf_bounded_training_binding_missing"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    command = [
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "snerv",
        "--planner-row-id",
        row_id,
        "--num-pairs",
        "16",
        "--epochs",
        "128",
        "--modelsize-candidate-id",
        "snerv_lf_hf_unit_candidate",
        "--skip-snerv-native-mlx-archive-export",
        *(command_extra or []),
    ]
    row = {
        "schema": "snerv_lf_hf_replacement_candidate_row.v1",
        "queue_row_id": row_id,
        "row_id": row_id,
        "family": "snerv",
        "candidate_class": "learned_lf_hf_replacement",
        "solution_family": solution_family,
        "status": status,
        "blocked": blocked,
        "launch_authority_contract": {
            "schema": "nerv_long_training_queue_launch_authority_contract.v1",
            "queue_status_is_local_mlx_plan": True,
            "queue_status_is_runnable_plan": runnable_contract,
            "queue_launch_blockers": [] if runnable_contract else ["unit_not_runnable"],
            "queue_status_is_receiver_proof": False,
            "queue_status_is_cpu_replay_proof": False,
            "queue_status_is_exact_eval_authority": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if include_command:
        row["command_argv"] = command
    if include_unblock_command:
        row["unblock_command_argv"] = command
        row["unblock_launch_authority_contract"] = {
            "schema": "snerv_lf_hf_queue_unblock_launch_contract.v1",
            "queue_unblock_status_is_local_mlx_plan": True,
            "queue_unblock_status_is_runnable_plan": unblock_contract_runnable,
            "queue_unblock_kind": "snerv_renderer_nondegenerate_smoke",
            "queue_unblock_blockers": []
            if unblock_contract_runnable
            else ["snerv_lf_hf_queue_unblock_command_not_runnable"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    if include_bounded_training_contract:
        row["bounded_training_binding_contract"] = bounded_training_contract
    payload = {
        "schema": "snerv_lf_hf_replacement_queue.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "queue_rows": [row],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_planner_row_launch_gate_requires_queue_artifact_for_planner_row() -> None:
    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "hi_nerv_planner_row_queue_artifact_missing" in blockers
    assert "planner_row_queue_artifact_required_for_planner_row_launch" in blockers
    assert "bounded_planner_row_timing_smoke_waiver_missing" in blockers


def test_planner_row_launch_gate_allows_queued_runnable_artifact(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(tmp_path / "queue.json")
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=600,
        epochs=29650,
        repo_root=REPO_ROOT,
    )

    assert runner_mod._planner_row_launch_blockers(args) == []
    guard = runner_mod._planner_row_launch_guard(args)
    record = guard["queue_artifact_status"]["artifact_records"][0]
    assert record["bytes"] == queue_path.stat().st_size
    assert record["sha256"] == runner_mod._sha256_file(queue_path)


def test_planner_row_launch_gate_accepts_snerv_lf_hf_replacement_queue_rows(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        repo_root=REPO_ROOT,
    )

    assert runner_mod._planner_row_launch_blockers(args) == []
    guard = runner_mod._planner_row_launch_guard(args)
    match = guard["queue_artifact_status"]["matched_runnable_records"][0]
    assert match["context"] == "queue_rows[0]"
    assert match["row_status_runnable"] is True
    assert match["launch_contract_runnable"] is True
    assert match["score_claim"] is False
    assert match["promotion_eligible"] is False
    assert match["ready_for_exact_eval_dispatch"] is False


def test_planner_row_launch_gate_checks_snerv_lf_hf_solution_family(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        solution_family="lf_conditioned_hf_residual_generator",
        command_extra=[
            "--snerv-lf-hf-solution-family",
            "lf_conditioned_hf_residual_generator",
            "--segnet-direct-live-target-mass-floor-weight",
            "0.50",
            "--segnet-direct-live-target-min-ratio-floor-weight",
            "0.5000",
        ],
    )

    matching_args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        snerv_lf_hf_solution_family="lf_conditioned_hf_residual_generator",
        segnet_direct_live_target_mass_floor_weight=0.5,
        segnet_direct_live_target_min_ratio_floor_weight=0.5,
        repo_root=REPO_ROOT,
    )
    assert runner_mod._planner_row_launch_blockers(matching_args) == []

    mismatched_args = SimpleNamespace(
        **{
            **matching_args.__dict__,
            "snerv_lf_hf_solution_family": "official_tub_lf_hf_decoder_replacement",
        }
    )
    blockers = runner_mod._planner_row_launch_blockers(mismatched_args)
    assert "planner_row_command_mismatch:--snerv-lf-hf-solution-family" in blockers
    assert "snerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers


def test_planner_row_launch_gate_rejects_snerv_lf_hf_runnable_without_command(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        include_command=False,
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        repo_root=REPO_ROOT,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert "planner_row_command_missing" in blockers
    assert "snerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers


def test_planner_row_launch_gate_accepts_snerv_lf_hf_blocked_unblock_command(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        status="blocked_until_prerequisite_evidence",
        blocked=True,
        runnable_contract=False,
        bounded_training_contract_bound=False,
        include_command=False,
        include_unblock_command=True,
        command_extra=["--segnet-direct-live-escape-class-multiplier", "16"],
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        segnet_direct_live_escape_class_multiplier=16.0,
        repo_root=REPO_ROOT,
    )

    assert runner_mod._planner_row_launch_blockers(args) == []
    guard = runner_mod._planner_row_launch_guard(args)
    match = guard["queue_artifact_status"]["matched_runnable_records"][0]
    assert match["context"] == "queue_rows[0].unblock_command_argv"
    assert match["planner_row_command_mode"] == "unblock"
    assert match["blocked"] is True
    assert match["row_status_runnable"] is True
    assert match["launch_contract_runnable"] is True


def test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_unblock_command(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        status="blocked_until_prerequisite_evidence",
        blocked=True,
        runnable_contract=False,
        bounded_training_contract_bound=False,
        include_command=False,
        include_unblock_command=True,
        unblock_contract_runnable=False,
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        repo_root=REPO_ROOT,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert "snerv_lf_hf_queue_unblock_command_not_runnable" in blockers
    assert "snerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers


def test_planner_row_launch_gate_rejects_snerv_lf_hf_missing_training_contract(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        include_bounded_training_contract=False,
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        repo_root=REPO_ROOT,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert "snerv_lf_hf_bounded_training_binding_contract_missing" in blockers
    assert "snerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers
    guard = runner_mod._planner_row_launch_guard(args)
    match = guard["queue_artifact_status"]["matched_records"][0]
    assert match["row_status_runnable"] is True
    assert match["launch_contract_runnable"] is False


def test_planner_row_launch_gate_rejects_snerv_lf_hf_unbound_training_contract(
    tmp_path: Path,
) -> None:
    queue_path = _write_snerv_lf_hf_replacement_queue_artifact(
        tmp_path / "snerv_lf_hf_queue.json",
        solution_family="lf_conditioned_hf_residual_generator",
        bounded_training_contract_bound=False,
    )
    args = SimpleNamespace(
        execute_family="snerv",
        planner_row_id="snerv_lf_hf_replace_unit",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=16,
        epochs=128,
        modelsize_candidate_id="snerv_lf_hf_unit_candidate",
        repo_root=REPO_ROOT,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert "snerv_lf_hf_bounded_training_binding_contract_not_bound" in blockers
    assert "snerv_lf_conditioned_hf_bounded_training_binding_missing" in blockers
    assert "snerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers


def test_planner_row_launch_gate_rejects_fake_queue_schema(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        schema="fake_queue.v1",
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_queue_artifact_schema_not_allowed" in blockers
    assert "hi_nerv_planner_row_id_not_found_in_queue_artifact" in blockers


def test_planner_row_launch_gate_rejects_stale_command_controls(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "queued-candidate",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=16,
            modelsize_candidate_id="different-candidate",
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--epochs" in blockers
    assert "planner_row_command_mismatch:--modelsize-candidate-id" in blockers


def test_planner_row_launch_gate_rejects_stale_hinerv_scorer_controls(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "queued-candidate",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-histogram-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-hinge-weight",
            "0.5",
            "--segnet-direct-live-class-balanced-ce-weight",
            "0.25",
            "--scorer-input-distribution-guard-weight",
            "2",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.6",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.6",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            modelsize_candidate_id="queued-candidate",
            segnet_distillation_objective="kl_t2",
            segnet_direct_live_distillation_weight=0.0,
            segnet_direct_live_class_histogram_weight=0.0,
            segnet_direct_live_class_balanced_hinge_weight=0.0,
            segnet_direct_live_class_balanced_ce_weight=0.0,
            scorer_input_distribution_guard_weight=0.0,
            scorer_input_contrast_floor_weight=0.0,
            scorer_input_contrast_floor_segnet_min_std_ratio=0.5,
            scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.5,
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--segnet-distillation-objective" in blockers
    assert "planner_row_command_mismatch:--segnet-direct-live-distillation-weight" in blockers
    assert "planner_row_command_mismatch:--segnet-direct-live-class-histogram-weight" in blockers
    assert "planner_row_command_mismatch:--segnet-direct-live-class-balanced-hinge-weight" in blockers
    assert "planner_row_command_mismatch:--segnet-direct-live-class-balanced-ce-weight" in blockers
    assert "planner_row_command_mismatch:--scorer-input-distribution-guard-weight" in blockers
    assert "planner_row_command_mismatch:--scorer-input-contrast-floor-weight" in blockers
    assert "planner_row_command_mismatch:--scorer-input-contrast-floor-segnet-min-std-ratio" in blockers
    assert "planner_row_command_mismatch:--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio" in blockers


def test_planner_row_launch_gate_tracks_snerv_official_modelsize_controls(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        family="snerv",
        row_id="snerv::candidate::native_rate_aware_training",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "auto",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-modelsize-control-profile",
            "official_cli_default",
            "--snerv-official-enc-strds",
            "1,2,2",
            "--snerv-official-dec-strds",
            "2,2,1",
            "--snerv-official-skip-high-mode",
            "shared_mean",
            "--snerv-official-trained-checkpoint-state-dict-path",
            "official_state_queued.npz",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::native_rate_aware_training",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            modelsize_candidate_id="auto",
            snerv_official_modelsize_mparams=[0.07],
            snerv_modelsize_control_profile="contest_receiver_profile",
            snerv_official_enc_strds=(1, 3, 2),
            snerv_official_dec_strds=(2, 2, 1),
            snerv_official_skip_high_mode="full",
            snerv_official_trained_checkpoint_state_dict_path=Path("official_state_local.npz"),
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--snerv-official-modelsize-mparams" in blockers
    assert "planner_row_command_mismatch:--snerv-modelsize-control-profile" in blockers
    assert "planner_row_command_mismatch:--snerv-official-enc-strds" in blockers
    assert "planner_row_command_mismatch:--snerv-official-skip-high-mode" in blockers
    assert "planner_row_command_mismatch:--snerv-official-trained-checkpoint-state-dict-path" in blockers
    assert "planner_row_command_mismatch:--snerv-official-dec-strds" not in blockers


def test_planner_row_launch_gate_accepts_numeric_equivalent_snerv_hfr_gain(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        family="snerv",
        row_id="snerv::candidate::native_rate_aware_training",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "candidate",
            "--snerv-hfr-gain",
            "0",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::native_rate_aware_training",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            modelsize_candidate_id="candidate",
            snerv_hfr_gain=0.0,
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--snerv-hfr-gain" not in blockers


def test_planner_row_launch_gate_rejects_nonrunnable_queue_artifact(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        status="disabled",
        blocked=True,
        runnable_contract=False,
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "hi_nerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers
    assert "bounded_planner_row_timing_smoke_waiver_missing" in blockers


def test_planner_row_launch_gate_allows_only_bounded_timing_smoke_waiver() -> None:
    assert (
        runner_mod._planner_row_launch_blockers(
            SimpleNamespace(
                execute_family="snerv",
                planner_row_id="snerv::candidate::adamw",
                planner_row_queue_artifact=[],
                allow_bounded_planner_row_timing_smoke_waiver=True,
                allow_manual_compact_family_launch=False,
                num_pairs=2,
                epochs=1,
                repo_root=REPO_ROOT,
            )
        )
        == []
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::adamw",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=True,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )
    assert "snerv_bounded_timing_smoke_waiver_exceeds_limits" in blockers


def test_snerv_bounded_timing_smoke_skips_scorer_loop_qat_by_default() -> None:
    args = SimpleNamespace(
        coder_aware_qat=True,
        snerv_scorer_loop_qat=False,
        snerv_bounded_smoke_allow_scorer_loop_qat=False,
    )

    effective, policy = runner_mod._snerv_scorer_loop_qat_request_policy(
        args,
        bounded_timing_smoke_waiver_consumed=True,
    )

    assert effective is False
    assert policy["requested"] is True
    assert policy["effective_requested"] is False
    assert policy["blockers"] == ["snerv_bounded_timing_smoke_scorer_loop_qat_skipped"]
    assert "packet_compression" in policy["skip_reason"]


def test_snerv_bounded_timing_smoke_can_opt_into_scorer_loop_qat() -> None:
    args = SimpleNamespace(
        coder_aware_qat=False,
        snerv_scorer_loop_qat=True,
        snerv_bounded_smoke_allow_scorer_loop_qat=True,
    )

    effective, policy = runner_mod._snerv_scorer_loop_qat_request_policy(
        args,
        bounded_timing_smoke_waiver_consumed=True,
    )

    assert effective is True
    assert policy["requested"] is True
    assert policy["effective_requested"] is True
    assert policy["blockers"] == []


def test_snerv_native_scorer_loop_qat_status_prefers_nested_export_truth() -> None:
    status = runner_mod._snerv_native_scorer_loop_qat_status(
        {
            "scorer_loop_qat_attached": True,
            "scorer_loop_qat_receiver_contract_satisfied": True,
            "scorer_loop_qat_ready_for_pose_guard_gate": True,
            "scorer_loop_qat_accepted_improvement": True,
            "scorer_loop_qat_best_materialized": True,
            "scorer_loop_qat": {
                "executed": False,
                "receiver_contract_satisfied": False,
                "ready_for_pose_guard_gate": False,
                "accepted_improvement": False,
                "best_packet_materialized": True,
                "emitted_packet_uses_scorer_loop_best_decoder": False,
            },
        }
    )

    assert status == {
        "attached": False,
        "receiver_contract_satisfied": False,
        "ready_for_pose_guard_gate": False,
        "accepted_improvement": False,
        "best_materialized": False,
    }


def test_planner_row_launch_gate_allows_explicit_manual_without_row() -> None:
    assert (
        runner_mod._planner_row_launch_blockers(
            SimpleNamespace(
                execute_family="snerv",
                planner_row_id="",
                planner_row_queue_artifact=[],
                allow_bounded_planner_row_timing_smoke_waiver=False,
                allow_manual_compact_family_launch=True,
                num_pairs=600,
                epochs=29650,
                repo_root=REPO_ROOT,
            )
        )
        == []
    )


def test_planner_row_launch_gate_allows_non_required_family() -> None:
    assert (
        runner_mod._planner_row_launch_blockers(
            SimpleNamespace(
                execute_family="pr95_hnerv",
                planner_row_id="",
                planner_row_queue_artifact=[],
                allow_bounded_planner_row_timing_smoke_waiver=False,
                allow_manual_compact_family_launch=False,
                num_pairs=600,
                epochs=29650,
                repo_root=REPO_ROOT,
            )
        )
        == []
    )


def test_compact_family_startup_marker_records_mlx_custody(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weight_path = tmp_path.parent / "joint_p18_p19_recon_pixel_weight.npz"
    weight_path.write_bytes(b"joint-weight")

    def fake_discover_joint_recon_pixel_weight_path(*, repo_root, num_pairs):
        return weight_path, {
            "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
            "num_pairs": int(num_pairs),
        }

    monkeypatch.setattr(
        runner_mod,
        "_discover_joint_recon_pixel_weight_path",
        fake_discover_joint_recon_pixel_weight_path,
    )
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        modelsize_candidate_id="candidate",
        target_modelsize_mparams=[],
        hinerv_target_modelsize_mparams=[],
        snerv_official_modelsize_mparams=[],
        auto_joint_recon_pixel_weight=True,
        distillation_device="mps",
        requested_distillation_device="gpu",
        mlx_prefilter_scorer_device="gpu",
        mlx_prefilter_scorer_batch_pairs=8,
        mlx_prefilter_progress_every=10,
        num_pairs=600,
        output_dir=tmp_path,
        repo_root=Path("/repo"),
    )

    path = runner_mod._write_compact_family_startup_marker(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000, 216_000),
        modelsize_candidate={"candidate_id": "candidate", "num_pairs": 600},
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "compact_carrier_startup_marker.v1"
    assert payload["execute_family"] == "hi_nerv"
    assert payload["planner_row_id"] == "hi_nerv::candidate::adamw"
    assert payload["distillation_device"] == "mps"
    assert payload["requested_distillation_device"] == "gpu"
    assert payload["mlx_prefilter_scorer_device"] == "gpu"
    assert payload["mlx_prefilter_scorer_batch_pairs"] == 8
    assert payload["auto_joint_recon_pixel_weight_path"] == weight_path.as_posix()
    assert payload["auto_joint_recon_pixel_weight_sha256"] == runner_mod._sha256_file(weight_path)
    assert payload["auto_joint_recon_pixel_weight_error"] is None
    assert payload["modelsize_target_binding"]["schema"] == ("compact_startup_modelsize_target_binding.v1")
    assert payload["modelsize_target_binding"]["inverse_target_requested"] is False
    assert payload["modelsize_target_binding"]["selected_from_inverse_target"] is False
    assert payload["byte_cap_binding"]["schema"] == ("compact_startup_byte_cap_binding.v1")
    assert payload["byte_cap_binding"]["hard_byte_cap_requested"] is True
    assert payload["byte_cap_binding"]["tightest_hard_byte_ceiling"] == 178_000
    assert payload["byte_cap_binding"]["authority_surface"] == ("measured_archive_zip_bytes_after_receiver_export")
    assert "byte_cap_requires_measured_archive_zip_export" in payload["byte_cap_binding"]["blockers"]
    assert payload["campaign_identity"]["auto_joint_recon_pixel_weight_path"] == (weight_path.as_posix())
    assert payload["score_claim"] is False
    assert payload["frontier_score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["false_authority_flags"] == [
        "macos_mlx_research_signal_until_archive_receiver_and_exact_eval",
        "startup_marker_before_trained_export_or_full_video_replay",
    ]
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            tmp_path,
            allow_startup_marker_only=True,
        )
        is False
    )

    (tmp_path / "foreign_artifact.json").write_text("{}", encoding="utf-8")
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            tmp_path,
            allow_startup_marker_only=True,
        )
        is True
    )


def test_compact_family_startup_marker_records_inverse_modelsize_target(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::targeted::adamw",
        modelsize_candidate_id="auto",
        target_modelsize_mparams=[0.178],
        hinerv_target_modelsize_mparams=[],
        snerv_official_modelsize_mparams=[],
        auto_joint_recon_pixel_weight=False,
        distillation_device="gpu",
        requested_distillation_device="gpu",
        mlx_prefilter_scorer_device="gpu",
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=50,
        num_pairs=600,
        output_dir=tmp_path,
        repo_root=Path("/repo"),
    )

    path = runner_mod._write_compact_family_startup_marker(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "candidate_id": "hinerv_np600_target",
            "capacity_source": "local_hinerv_target_modelsize",
            "target_modelsize_mparams": 0.178,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 164_000,
            "byte_headroom": 14_000,
            "nominal_under_ceiling": True,
            "modelsize_control_contract": {
                "control_semantics": "local_receiver_visible_grid_search_nearest_target",
                "shared_target_modelsize_mparams_consumed_as": ("nearest_local_param_count_target"),
                "modelsize_mparams_is_official_upstream_flag": False,
                "modelsize_mparams_caps_archive_zip_bytes": False,
                "archive_bytes_authority_required": True,
            },
        },
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    binding = payload["modelsize_target_binding"]
    assert binding["inverse_target_requested"] is True
    assert binding["selected_from_inverse_target"] is True
    assert binding["effective_requested_targets_for_family"] == [0.178]
    assert binding["selected_capacity_source"] == "local_hinerv_target_modelsize"
    assert binding["control_semantics"] == ("local_receiver_visible_grid_search_nearest_target")
    assert binding["blockers"] == []
    byte_binding = payload["byte_cap_binding"]
    assert byte_binding["hard_byte_cap_requested"] is True
    assert byte_binding["tightest_hard_byte_ceiling"] == 178_000
    assert byte_binding["selected_candidate_hard_byte_ceiling"] == 178_000
    assert byte_binding["selected_candidate_nominal_total_payload_bytes"] == 164_000
    assert byte_binding["nominal_headroom_against_tightest_hard_ceiling"] == 14_000
    assert byte_binding["selected_candidate_nominal_under_tightest_hard_ceiling"] is True
    assert byte_binding["selected_candidate_nominal_under_own_hard_ceiling"] is True
    assert byte_binding["calibrated_predicted_archive_bytes"] is None
    assert byte_binding["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert byte_binding["archive_bytes_authority_required"] is True
    assert byte_binding["blockers"] == ["byte_cap_requires_measured_archive_zip_export"]


def test_startup_byte_cap_binding_allows_looser_selected_sweep_ceiling() -> None:
    binding = runner_mod._startup_byte_cap_binding(
        family="hi_nerv",
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        modelsize_candidate={
            "candidate_id": "hinerv_285k_candidate",
            "hard_byte_ceiling": 285_000,
            "nominal_total_payload_bytes": 240_000,
            "byte_headroom": 45_000,
            "nominal_under_ceiling": True,
            "modelsize_control_contract": {
                "archive_bytes_authority_required": True,
                "modelsize_mparams_caps_archive_zip_bytes": False,
            },
        },
    )

    assert binding["tightest_hard_byte_ceiling"] == 178_000
    assert binding["selected_candidate_hard_byte_ceiling"] == 285_000
    assert binding["selected_candidate_nominal_under_tightest_hard_ceiling"] is False
    assert binding["selected_candidate_nominal_under_own_hard_ceiling"] is True
    assert binding["tightest_hard_ceiling_is_blocking"] is False
    assert binding["blockers"] == ["byte_cap_requires_measured_archive_zip_export"]


def test_compact_family_interrupted_report_preserves_false_authority_custody(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir()
    telemetry.write_text('{"epoch":0,"loss":1.0}\n', encoding="utf-8")
    startup = tmp_path / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    startup.write_text(
        '{"schema":"compact_carrier_startup_marker.v1"}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        modelsize_candidate_id="candidate",
        allow_duplicate_campaign=False,
        output_dir=tmp_path,
        overwrite=False,
        repo_root=Path("/repo"),
        num_pairs=600,
        auto_joint_recon_pixel_weight=False,
        distillation_device="mps",
        mlx_prefilter_scorer_device="gpu",
    )

    report = runner_mod._write_compact_family_interrupted_report(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={"candidate_id": "candidate", "num_pairs": 600},
        signum=15,
        reason="unit_test_signal",
    )

    assert report["schema"] == COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA
    assert report["mode"] == "interrupted_compact_family_run"
    assert report["signal_name"] == "SIGTERM"
    assert report["interruption_reason"] == "unit_test_signal"
    assert report["execute_family"] == "hi_nerv"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "hi_nerv_training_interrupted_before_export" in report["blockers"]
    evidence_by_name = {Path(row["path"]).name: row for row in report["evidence_files"]}
    assert "telemetry.jsonl" in evidence_by_name
    assert evidence_by_name["telemetry.jsonl"]["sha256"] == runner_mod._sha256_file(telemetry)
    report_path = tmp_path / "compact_renderer_mlx_spine_runner_report.json"
    assert report_path.is_file()


def test_hi_nerv_source_faithfulness_classifies_local_adaptation() -> None:
    report = runner_mod._hi_nerv_source_faithfulness_report(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=False,
            use_convnext_blocks=False,
        ),
        decoder_codec="portfolio_auto",
    )

    assert report["schema"] == "hi_nerv_source_faithfulness.v1"
    assert report["classification"] == "local_hiv1_adaptation_not_official_hinerv"
    assert report["source_faithful_official_hinerv"] is False
    assert report["local_hiv1_adaptation"] is True
    assert "hinerv_official_hierarchical_feature_grid_not_enabled" in report["blockers"]
    assert "hinerv_pr95_pr101_latent_delta_brotli_codec_missing" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hi_nerv_source_faithfulness_separates_official_controls_from_pr95_gaps() -> None:
    report = runner_mod._hi_nerv_source_faithfulness_report(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=True,
            use_convnext_blocks=True,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=3,
        ),
        decoder_codec="portfolio_auto",
    )

    assert report["classification"] == ("official_hinerv_control_candidate_source_parity_bound_pr95_better_gaps")
    assert report["official_hinerv_control"] is True
    assert report["source_faithful_official_hinerv"] is False
    assert report["official_source_parity_proof_required"] is True
    assert report["official_source_parity_proof_attached"] is True
    assert report["local_hiv1_adaptation"] is True
    assert report["official_hinerv_blockers"] == []
    assert report["source_parity_blockers"] == []
    assert report["source_parity_binding"]["required_for_long_training_ready"] is True
    assert (
        report["source_parity_binding"]["feature_statuses"]["hi_nerv_official_patch_index_path"]
        == "implemented_or_bound"
    )
    assert "hinerv_pr95_pixelshuffle_bilinear_skip_refine_path_missing" in report["pr95_better_blockers"]
    assert report["local_grid_levels"] == 2
    assert report["convnext_kernel_size"] == 3
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hi_nerv_source_faithfulness_metadata_strips_authority_keys() -> None:
    metadata = runner_mod._hi_nerv_source_faithfulness_metadata(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=True,
            use_convnext_blocks=True,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=3,
        ),
        decoder_codec="portfolio_auto",
    )

    assert metadata["schema"] == "hi_nerv_source_faithfulness.v1"
    assert metadata["official_hinerv_control"] is True
    assert metadata["source_faithful_official_hinerv"] is False
    assert metadata["official_source_parity_proof_required"] is True
    for forbidden in (
        "score_claim",
        "frontier_score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        assert forbidden not in metadata


def test_hinerv_refuses_non_official_control_candidate_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("local HiV1-style candidate must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_local_control_refusal",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "local-hiv1-unit-candidate",
            "latent_dim": 4,
            "embed_dim": 4,
            "decoder_channel": 4,
            "decoder_codec": "int4_mixed",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 100_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": False,
            "use_convnext_blocks": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_official_control_launch_refused"
    assert out["training_executed"] is False
    assert out["trainer_launch_allowed"] is False
    assert out["hi_nerv_source_faithfulness"]["official_hinerv_control"] is False
    assert out["hi_nerv_control_precedence"]["schema"] == ("hi_nerv_launch_control_precedence.v1")
    assert out["hi_nerv_control_precedence"]["more_finely_grained_child_rules_take_priority"] is True
    assert out["hi_nerv_control_precedence"]["parent_rules_remain_required_guardrails"] is True
    assert (
        "hinerv_official_hierarchical_feature_grid_not_enabled"
        in out["hi_nerv_control_precedence"]["source_base_blockers"]
    )
    assert "hinerv_official_control_required_for_top_priority_launch" in out["blockers"]
    assert "hinerv_official_hierarchical_feature_grid_not_enabled" in out["blockers"]
    assert "hinerv_official_convnext_blocks_not_enabled" in out["blockers"]


def test_startup_marker_only_output_dir_is_not_dirty(tmp_path: Path) -> None:
    out = tmp_path / "candidate"
    out.mkdir()
    marker = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    marker.write_text("{}", encoding="utf-8")

    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is False
    )

    (out / "hi_nerv_mlx_training").mkdir()
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is False
    )

    (out / "hi_nerv_mlx_training" / "telemetry.jsonl").write_text(
        "{}",
        encoding="utf-8",
    )
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is True
    )


def test_torch_scorer_device_alias_resolves_gpu_to_concrete_backend() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )

    assert (
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_torch,
        )
        == "mps"
    )

    fake_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    assert (
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_cuda,
        )
        == "cuda"
    )
    assert runner_mod._resolve_torch_scorer_device_alias("metal") == "mps"


def test_torch_scorer_device_alias_fails_closed_without_gpu() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        ),
    )

    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match=r"neither torch\.cuda nor torch\.backends\.mps",
    ):
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_torch,
        )


def test_mlx_prefilter_scorer_device_alias_uses_mlx_device_dialect() -> None:
    assert (
        runner_mod._resolve_mlx_prefilter_scorer_device_alias(
            None,
            fallback_device="mps",
        )
        == "gpu"
    )
    assert (
        runner_mod._resolve_mlx_prefilter_scorer_device_alias(
            "metal",
            fallback_device="cpu",
        )
        == "gpu"
    )
    assert (
        runner_mod._resolve_mlx_prefilter_scorer_device_alias(
            "mps",
            fallback_device="cpu",
        )
        == "gpu"
    )
    assert (
        runner_mod._resolve_mlx_prefilter_scorer_device_alias(
            None,
            fallback_device="cpu",
        )
        == "cpu"
    )

    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="mlx prefilter scorer device",
    ):
        runner_mod._resolve_mlx_prefilter_scorer_device_alias("vulkan")


def test_hinerv_execute_allows_runner_startup_marker_only_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "hinerv_marker_only"
    output_dir.mkdir()
    marker = output_dir / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    marker.write_text("{}", encoding="utf-8")
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=output_dir,
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        distillation_device="mps",
        requested_distillation_device="gpu",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs
    assert captured_train_kwargs["distillation_device"] == "mps"
    assert captured_train_kwargs["requested_distillation_device"] == "gpu"
    assert captured_train_kwargs["mlx_prefilter_scorer_device"] == "gpu"
    assert captured_train_kwargs["prioritized_pair_indices"] == ()
    assert captured_train_kwargs["use_hierarchical_feature_grid"] is True
    assert captured_train_kwargs["use_convnext_blocks"] is True
    assert out["score_aware_training"]["local_mlx_prefilter"]["scorer_device"] == ("gpu")
    assert out["execute_family"] == "hi_nerv"
    assert out["training_executed"] is True
    assert marker.is_file()
    embedded_plan = out["nerv_long_training_campaign_plan"]
    assert embedded_plan["planner_row_queue_artifact_path"].endswith("/compact_renderer_mlx_spine_runner_report.json")
    embedded_command = embedded_plan["campaign_rows"][0]["command_argv"]
    assert "--planner-row-queue-artifact" in embedded_command


def test_hinerv_execute_forwards_prioritized_pair_indices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=4)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "decoder_weight_gradient_saliency_artifact": None,
                }
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_priority",
        num_pairs=4,
        epochs=1,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        checkpoint_interval_epochs=7,
        checkpoint_dir=tmp_path / "external_checkpoints",
        resume_from_checkpoint=tmp_path / "external_checkpoints/epoch000006.meta.json",
        prioritized_pair_indices=(3, 1, 3),
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["prioritized_pair_indices"] == (3, 1)
    assert captured_train_kwargs["use_hierarchical_feature_grid"] is True
    assert captured_train_kwargs["use_convnext_blocks"] is True
    assert captured_train_kwargs["checkpoint_interval_epochs"] == 7
    assert captured_train_kwargs["checkpoint_dir"] == (tmp_path / "external_checkpoints").resolve(strict=False)
    assert captured_train_kwargs["resume_from_checkpoint"] == (
        tmp_path / "external_checkpoints/epoch000006.meta.json"
    ).resolve(strict=False)
    prioritized = out["score_aware_training"]["prioritized_pair_training"]
    assert prioritized["enabled"] is True
    assert prioritized["pair_indices"] == [3, 1]
    assert prioritized["pair_index_domain"] == "source_video_pair_indices_0_to_model_num_pairs_minus_1"
    assert prioritized["model_num_pairs"] == 4
    assert prioritized["hydrated_target_pair_count"] == 4
    assert prioritized["source_pair_indices"] == [3, 1]
    assert prioritized["local_target_rows_are_compact"] is False
    assert prioritized["arbitrary_source_pair_hydration"] is False
    assert prioritized["target_hydration_pair_indices_consumed"] is False
    assert prioritized["target_hydration_pair_indices_consumed_by_renderer_bundle"] is False
    assert prioritized["sampling_scope"] == ("full_video_target_hydration_with_prioritized_sampling")
    assert prioritized["requires_num_pairs_covering_pair_ids"] is False
    assert prioritized["score_claim"] is False
    assert prioritized["promotion_eligible"] is False
    assert out["score_claim"] is False


def test_hinerv_execute_forwards_explicit_pr95_curriculum_total_epochs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_pr95_total_epochs",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        hi_nerv_pr95_curriculum_total_epochs=29_650,
        repo_root=REPO_ROOT,
    )

    optimizer_policy = captured_train_kwargs["hi_nerv_optimizer_policy"]
    assert captured_train_kwargs["pr95_curriculum_total_epochs"] == 29_650
    assert optimizer_policy["resolved_policy"] == "pr95_curriculum"
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is True
    assert optimizer_policy["pr95_curriculum_total_epochs"] == 29_650
    assert optimizer_policy["pr95_curriculum_total_epochs_requested"] == 29_650
    assert optimizer_policy["pr95_curriculum_total_epochs_defaulted"] is False
    assert optimizer_policy["pr95_stage_source_weight_amplification_requested"] is False
    assert optimizer_policy["pr95_stage_source_weight_amplification_enabled"] is True
    assert optimizer_policy["pr95_stage_source_weight_amplification_defaulted"] is True
    assert captured_train_kwargs["pr95_stage_source_weight_amplification_enabled"] is True
    assert out["hi_nerv_optimizer_policy"]["pr95_curriculum_total_epochs"] == 29_650
    assert out["hi_nerv_optimizer_policy"]["pr95_stage_source_weight_amplification_enabled"] is True
    assert out["score_aware_training"]["pr95_curriculum_total_epochs"] == 29_650
    assert out["score_aware_training"]["pr95_curriculum_total_epochs_consumed"] is True
    assert out["score_aware_training"]["pr95_stage_source_weight_amplification_enabled"] is True


def test_parse_prioritized_pair_indices_arg() -> None:
    assert runner_mod._parse_nonnegative_int_csv("") == ()
    assert runner_mod._parse_nonnegative_int_csv("3,1,3,0") == (3, 1, 0)
    assert runner_mod._normalize_nonnegative_int_sequence(None) == ()
    assert runner_mod._normalize_nonnegative_int_sequence("3,1,3") == (3, 1)
    with pytest.raises(CompactRendererMlxSpineRunnerError):
        runner_mod._parse_nonnegative_int_csv("2,-1")


def test_parse_prioritized_pair_indices_file_arg(tmp_path: Path) -> None:
    pair_file = tmp_path / "feedback.json"
    pair_file.write_text(
        '{"sample_generalization_gate":{"hard_pair_coverage":{"prioritized_pair_indices":[8,2,8]}}}',
        encoding="utf-8",
    )
    args = runner_mod._parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "10",
            "--prioritized-pair-indices",
            "3,2",
            "--prioritized-pair-indices-file",
            str(pair_file),
        ]
    )

    assert runner_mod._prioritized_pair_indices_from_args(args) == (3, 2, 8)


def test_hinerv_priority_hydration_contract_explicit_sparse_mode() -> None:
    contract = runner_mod._hi_nerv_priority_target_hydration_contract(
        prioritized_pair_indices=(7, 2),
        model_num_pairs=10,
        hydrated_target_pair_count=2,
        sparse_target_hydration=True,
    )

    assert contract["enabled"] is True
    assert contract["sampling_scope"] == "sparse_source_pair_target_hydration"
    assert contract["pair_index_domain"] == ("source_video_pair_indices_0_to_model_num_pairs_minus_1")
    assert contract["local_target_rows_are_compact"] is True
    assert contract["arbitrary_source_pair_hydration"] is True
    assert contract["target_hydration_pair_indices_consumed"] is True
    assert contract["requires_num_pairs_covering_pair_ids"] is True


def test_parse_prioritized_pair_indices_allows_source_ids_above_num_pairs() -> None:
    args = runner_mod._parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "4",
            "--prioritized-pair-indices",
            "3,4",
        ]
    )

    assert runner_mod._prioritized_pair_indices_from_args(args) == (3, 4)


def test_hinerv_execute_rejects_out_of_range_prioritized_pairs(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="out-of-range"):
        execute_hi_nerv_mlx_scoreaware_and_adapt(
            output_dir=tmp_path / "hinerv_out_of_range_priority",
            num_pairs=4,
            epochs=1,
            batch_pair_indices_per_step=2,
            learning_rate=1e-3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            hard_byte_ceilings=(178_000,),
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            segnet_distillation_weight=1.0,
            pose_distillation_weight=1.0,
            prioritized_pair_indices=(3, 4),
            repo_root=REPO_ROOT,
        )


def test_hinerv_private_smoke_refuses_positive_hard_birth_without_segnet_argmax(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    bootstrap_called = False
    train_called = False

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def configure_decoder_fake_quant_forward(self, **_kwargs):
            return None

        def initialize_output_head_bias_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            epsilon,
        ):
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(self, *_args, **_kwargs):
            nonlocal bootstrap_called
            bootstrap_called = True
            raise AssertionError("hard-birth without SegNet argmax must refuse before bootstrap")

        def num_parameters(self):
            return 123

    def fake_decode_mlx_targets(
        _video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def fail_train(**_kwargs):
        nonlocal train_called
        train_called = True
        raise AssertionError("hard-birth guard must refuse before training")

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fail_train,
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )

    with pytest.raises(
        CompactRendererMlxSpineRunnerError,
        match="requires real SegNet teacher argmax labels",
    ):
        runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=tmp_path / "hard_birth_without_segnet_argmax",
            num_pairs=1,
            epochs=1,
            batch_pair_indices_per_step=1,
            learning_rate=1e-3,
            source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            use_hierarchical_feature_grid=False,
            use_convnext_blocks=False,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=7,
            mid_injection_block_index=1,
            fine_injection_block_index=4,
            decoder_codec="portfolio_auto",
            ema_decay=0.9,
            segnet_distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_distillation_loss="mse",
            pose_distillation_huber_delta=1.0,
            recon_loss_stage_weight=1.0,
            segnet_loss_stage_weight=1.0,
            pose_loss_stage_weight=1.0,
            scorer_input_guard_stage_weight=1.0,
            scorer_input_contrast_floor_stage_weight=None,
            scorer_input_shape_tether_stage_weight=None,
            segnet_direct_live_stage_weight=None,
            segnet_distillation_objective="kl_t2",
            distillation_temperature=2.0,
            segnet_tau_boundary=1.0,
            segnet_hinge_margin=1.0,
            scorer_domain_bootstrap_segnet_margin_weight=0.0,
            scorer_domain_bootstrap_segnet_hard_birth_weight=2.0,
            distillation_device="cpu",
            requested_distillation_device=None,
            allow_segnet_only_research=False,
            coder_aware_qat=False,
            coder_qat_quant_bits=8,
            coder_qat_quant_residual_weight=0.0,
            coder_qat_magnitude_weight=0.0,
            coder_qat_delta_weight=0.0,
            coder_qat_c1a_entropy_weight=0.0,
            coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
            coder_qat_c1a_sample_size=(runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE),
            recon_pixel_weight_path=None,
            decoder_weight_waterfill_plan=None,
            recon_pixel_weight_auto_discovery=None,
            auto_segnet_boundary_recon_weight=False,
            recon_pixel_weight_tau=1.0,
            recon_pixel_weight_normalize="mean",
            mlx_prefilter_scorer_device=None,
            mlx_prefilter_scorer_batch_pairs=1,
            mlx_prefilter_progress_every=50,
            telemetry_flush_interval_epochs=1,
            checkpoint_interval_epochs=1,
            checkpoint_dir=None,
            resume_from_checkpoint=None,
            optimizer_kind="adamw",
            hi_nerv_optimizer_policy={},
            optimizer_controls={},
            prioritized_pair_indices=(),
            random_seed=0,
            scorer_upstream_dir=REPO_ROOT / "upstream",
            repo_root=REPO_ROOT,
        )

    assert bootstrap_called is False
    assert train_called is False


def test_hinerv_private_smoke_defaults_to_full_target_hydration_for_hard_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    captured: dict[str, object] = {}

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fake_quant = {}

        def configure_decoder_fake_quant_forward(self, **kwargs):
            self.fake_quant = dict(kwargs)

        def initialize_output_head_bias_from_targets(
            self,
            target_rgb_0,
            target_rgb_1,
            *,
            epsilon,
        ):
            captured["output_head_target_bias_init_call"] = {
                "target0_shape": tuple(target_rgb_0.shape),
                "target1_shape": tuple(target_rgb_1.shape),
                "epsilon": float(epsilon),
            }
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.bias",
                    "head_rgb_1.bias",
                ],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            target_rgb_0,
            target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            captured["output_head_target_contrast_init_call"] = {
                "target0_shape": tuple(target_rgb_0.shape),
                "target1_shape": tuple(target_rgb_1.shape),
                "pair_indices": _fake_hinerv_output_head_contrast_init_payload(
                    pair_indices,
                    min_output_std=min_output_std,
                    max_gain=max_gain,
                )["source_pair_indices"],
                "min_output_std": float(min_output_std),
                "max_gain": float(max_gain),
            }
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(
            self,
            target_rgb_0,
            target_rgb_1,
            *,
            pair_indices,
            steps,
            learning_rate,
            rgb_weight,
            yuv6_weight,
            temporal_delta_weight,
            contrast_floor_weight,
            rgb_std_min_ratio,
            yuv6_temporal_std_min_ratio,
            weight_decay,
            grad_clip_max_norm,
            target_segnet_argmax_1=None,
            scorer_teacher=None,
            segnet_margin_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_min_ratio_floor=0.02,
            pair_local_smoke_artifact_dir=None,
        ):
            captured["scorer_domain_bootstrap_call"] = {
                "target0_shape": tuple(target_rgb_0.shape),
                "target1_shape": tuple(target_rgb_1.shape),
                "steps": int(steps),
                "scorer_teacher_present": scorer_teacher is not None,
                "segnet_margin_bootstrap_weight": float(segnet_margin_bootstrap_weight),
                "segnet_hard_birth_bootstrap_weight": float(segnet_hard_birth_bootstrap_weight),
                "segnet_hard_birth_bootstrap_min_ratio_floor": float(segnet_hard_birth_bootstrap_min_ratio_floor),
                "pair_local_smoke_artifact_dir": (
                    None if pair_local_smoke_artifact_dir is None else Path(pair_local_smoke_artifact_dir).as_posix()
                ),
            }
            return _fake_hinerv_scorer_domain_bootstrap_payload(
                pair_indices,
                steps=steps,
                learning_rate=learning_rate,
                rgb_weight=rgb_weight,
                yuv6_weight=yuv6_weight,
                temporal_delta_weight=temporal_delta_weight,
                contrast_floor_weight=contrast_floor_weight,
                rgb_std_min_ratio=rgb_std_min_ratio,
                yuv6_temporal_std_min_ratio=yuv6_temporal_std_min_ratio,
                weight_decay=weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            )

        def fit_target_region_birth_from_segnet(
            self,
            *,
            scorer_teacher,
            target_rgb_0,
            target_rgb_1,
            pair_indices,
            target_segnet_argmax_1,
            max_steps,
            learning_rate,
            target_min_region_ratio,
            pose_teacher,
            require_pose_trust,
            lambda_support_preserve,
            lambda_outside_argmax_preserve,
            lambda_already_won_hard_preserve,
            lambda_already_won_rgb_preserve,
            already_won_margin_floor,
            lambda_pose_trust_preserve,
            lambda_pose_target,
            grad_clip_max_norm,
        ):
            captured["target_region_birth_call"] = {
                "scorer_teacher_present": scorer_teacher is not None,
                "pose_teacher_present": pose_teacher is not None,
                "require_pose_trust": bool(require_pose_trust),
                "target0_shape": tuple(target_rgb_0.shape),
                "target1_shape": tuple(target_rgb_1.shape),
                "pair_indices": tuple(int(value) for value in pair_indices),
                "target_segnet_argmax_shape": tuple(target_segnet_argmax_1.shape),
                "max_steps": int(max_steps),
                "learning_rate": float(learning_rate),
                "target_min_region_ratio": float(target_min_region_ratio),
                "lambda_support_preserve": float(lambda_support_preserve),
                "lambda_outside_argmax_preserve": float(lambda_outside_argmax_preserve),
                "lambda_already_won_hard_preserve": float(lambda_already_won_hard_preserve),
                "lambda_already_won_rgb_preserve": float(lambda_already_won_rgb_preserve),
                "already_won_margin_floor": float(already_won_margin_floor),
                "lambda_pose_trust_preserve": float(lambda_pose_trust_preserve),
                "lambda_pose_target": float(lambda_pose_target),
                "grad_clip_max_norm": grad_clip_max_norm,
            }
            return {
                "schema": "hi_nerv_target_region_birth.v1",
                "actuator_id": "fit_target_region_birth_from_segnet",
                "accepted": True,
                "accepted_step_count": 1,
                "action_id": "a" * 64,
                "target_hard_won_count": 5.0,
                "target_hard_lost_count": 0.0,
                "net_target_support_delta": 5.0,
                "receipt": {
                    "schema": "hi_nerv_target_region_birth_receipt.v1",
                    "surface": "live_mlx",
                    "action_id": "a" * 64,
                    "accepted_step_count": 1,
                    "updated_parameter_names": ["head_rgb_1.weight"],
                    "argmax_transitions": {
                        "target_hard_won_count": 5.0,
                        "target_hard_lost_count": 0.0,
                        "target_to_wrong_count": 0.0,
                        "net_target_support_delta": 5.0,
                    },
                    "pose_guard": {
                        "available": True,
                        "pose_input_contest_resolution": True,
                        "max_accepted_pose_output_delta_l2": 0.025,
                        "max_pose_output_delta_l2": 0.05,
                    },
                    "exact_nonrate": {
                        "pose_term_available": True,
                        "delta_score_nonrate": -0.1,
                    },
                    "candidate_frontier_telemetry": {
                        "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
                        "candidate_attempt_count": 2,
                    },
                },
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": ["decoder.target_region_birth.delta"],
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        def num_parameters(self):
            return 123

    class FakeArtifact:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def as_dict(self) -> dict[str, object]:
            return dict(self._payload)

    class FakeSegNetTeacher:
        num_classes = 8
        frame_count = 10
        live_segnet_adapter = object()

        def teacher_argmax_for_indices(self, idx):
            import mlx.core as mx

            return mx.zeros((int(idx.shape[0]), 384, 512), dtype=mx.int32)

        def teacher_logits_for_frames_nhwc01(self, frames):
            return frames

    class FakePoseNetTeacher:
        pose_dims = 12
        live_posenet_adapter = object()

        def teacher_pose_for_yuv6_pair_nhwc(self, pairs):
            return pairs

    def fake_decode_mlx_targets(
        video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        captured["decode_call"] = {
            "video_path": Path(video_path),
            "num_pairs": int(num_pairs),
            "output_height": int(output_height),
            "output_width": int(output_width),
            "pair_indices": tuple(pair_indices or ()),
        }
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def fake_run_mlx_score_aware_full_main(**kwargs):
        bundle = kwargs["bundle"]
        captured["bundle_num_pairs"] = int(bundle.num_pairs)
        captured["bundle_source_pair_indices"] = tuple(bundle.source_pair_indices or ())
        captured["model_num_pairs"] = int(bundle.model.cfg.num_pairs)
        captured["bundle_scorer_input_distribution_guard_weight"] = float(bundle.scorer_input_distribution_guard_weight)
        captured["bundle_scorer_input_distribution_guard_saturation_margin"] = float(
            bundle.scorer_input_distribution_guard_saturation_margin
        )
        captured["bundle_scorer_input_distribution_guard_temperature"] = float(
            bundle.scorer_input_distribution_guard_temperature
        )
        captured["bundle_scorer_input_contrast_floor_weight"] = float(bundle.scorer_input_contrast_floor_weight)
        captured["bundle_scorer_input_contrast_floor_segnet_min_std_ratio"] = float(
            bundle.scorer_input_contrast_floor_segnet_min_std_ratio
        )
        captured["bundle_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"] = float(
            bundle.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
        )
        captured["bundle_posenet_yuv6_geometry_tether_weight"] = float(bundle.posenet_yuv6_geometry_tether_weight)
        captured["bundle_posenet_temporal_signal_floor_weight"] = float(bundle.posenet_temporal_signal_floor_weight)
        captured["bundle_posenet_temporal_signal_min_std_ratio"] = float(bundle.posenet_temporal_signal_min_std_ratio)
        captured["bundle_posenet_temporal_signal_min_mean_abs_ratio"] = float(
            bundle.posenet_temporal_signal_min_mean_abs_ratio
        )
        captured["bundle_segnet_direct_live_rare_class_logit_weight"] = float(
            bundle.segnet_direct_live_rare_class_logit_weight
        )
        captured["bundle_segnet_direct_live_class_balanced_hinge_weight"] = float(
            bundle.segnet_direct_live_class_balanced_hinge_weight
        )
        captured["bundle_segnet_direct_live_class_balanced_ce_weight"] = float(
            bundle.segnet_direct_live_class_balanced_ce_weight
        )
        captured["bundle_segnet_direct_live_class_balanced_squared_hinge_weight"] = float(
            bundle.segnet_direct_live_class_balanced_squared_hinge_weight
        )
        captured["bundle_segnet_direct_live_class_region_recon_weight"] = float(
            bundle.segnet_direct_live_class_region_recon_weight
        )
        captured["bundle_segnet_direct_live_target_mass_floor_weight"] = float(
            bundle.segnet_direct_live_target_mass_floor_weight
        )
        captured["bundle_segnet_direct_live_target_min_ratio_floor_weight"] = float(
            bundle.segnet_direct_live_target_min_ratio_floor_weight
        )
        captured["run_prioritized_pair_indices"] = tuple(kwargs["prioritized_pair_indices"])
        captured["scorer_space_step_guard_enabled"] = bool(kwargs["scorer_space_step_guard_enabled"])
        captured["scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"] = float(
            kwargs["scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"]
        )
        captured["scorer_space_step_guard_min_post_segnet_occupied_class_fraction"] = float(
            kwargs["scorer_space_step_guard_min_post_segnet_occupied_class_fraction"]
        )
        captured["scorer_space_step_guard_max_post_segnet_contrast_ratio"] = float(
            kwargs["scorer_space_step_guard_max_post_segnet_contrast_ratio"]
        )
        captured["scorer_space_step_guard_max_post_segnet_distribution_mae"] = float(
            kwargs["scorer_space_step_guard_max_post_segnet_distribution_mae"]
        )
        captured["scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"] = float(
            kwargs["scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"]
        )
        captured["scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"] = float(
            kwargs["scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"]
        )
        captured["scorer_space_step_guard_backtracking_steps"] = int(
            kwargs["scorer_space_step_guard_backtracking_steps"]
        )
        captured["scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"] = float(
            kwargs["scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"]
        )
        captured["scorer_space_step_guard_backtracking_shrink"] = float(
            kwargs["scorer_space_step_guard_backtracking_shrink"]
        )
        captured["scorer_support_ladder_enabled"] = bool(kwargs["scorer_support_ladder_enabled"])
        captured["scorer_support_ladder_target_coverage_floor"] = float(
            kwargs["scorer_support_ladder_target_coverage_floor"]
        )
        captured["scorer_support_ladder_target_min_ratio_floor"] = float(
            kwargs["scorer_support_ladder_target_min_ratio_floor"]
        )
        captured["scorer_support_ladder_patience_steps"] = int(kwargs["scorer_support_ladder_patience_steps"])
        captured["scorer_support_ladder_growth_factor"] = float(kwargs["scorer_support_ladder_growth_factor"])
        captured["scorer_support_ladder_max_multiplier"] = float(kwargs["scorer_support_ladder_max_multiplier"])
        captured["scorer_support_ladder_base_loss_max_when_active"] = float(
            kwargs["scorer_support_ladder_base_loss_max_when_active"]
        )
        captured["curriculum_stage_loss_weights"] = [dict(stage.loss_weights) for stage in kwargs["curriculum_stages"]]
        captured["run_pr95_curriculum_total_epochs"] = int(kwargs["pr95_curriculum_total_epochs"])
        captured["checkpoint_selection_metric_key"] = str(kwargs["checkpoint_selection_metric_key"])
        captured["checkpoint_selection_metric_required"] = bool(kwargs["checkpoint_selection_metric_required"])
        dual_config = kwargs["train_time_dual_ascent_config"]
        captured["dual_ascent_constraints"] = {row["constraint_id"]: dict(row) for row in dual_config["constraints"]}
        archive = tmp_path / "fake_hinerv_sparse_hydration_archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=10)
        return FakeArtifact(
            {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": runner_mod._sha256_file(archive),
                "substrate_artifact_metadata": dict(bundle.substrate_artifact_metadata),
            }
        )

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "build_mlx_segnet_pair_teacher",
        lambda *args, **kwargs: FakeSegNetTeacher(),
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "build_mlx_posenet_pair_teacher",
        lambda *args, **kwargs: FakePoseNetTeacher(),
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )

    artifact = runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
        output_dir=tmp_path / "private_smoke_full_priority_hydration",
        num_pairs=10,
        epochs=1,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        use_hierarchical_feature_grid=False,
        use_convnext_blocks=False,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=1,
        fine_injection_block_index=4,
        decoder_codec="portfolio_auto",
        hi_nerv_latent_codec="int16_brotli_q11",
        ema_decay=0.9,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        pose_direct_live_distillation_weight=0.0,
        pose_trust_required=True,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        recon_loss_stage_weight=1.0,
        segnet_loss_stage_weight=1.0,
        pose_loss_stage_weight=1.0,
        scorer_input_guard_stage_weight=1.0,
        scorer_input_contrast_floor_stage_weight=None,
        scorer_input_shape_tether_stage_weight=None,
        posenet_yuv6_geometry_tether_stage_weight=0.75,
        posenet_temporal_signal_floor_stage_weight=0.625,
        segnet_direct_live_stage_weight=1.0,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        scorer_input_distribution_guard_weight=0.25,
        scorer_input_distribution_guard_saturation_margin=0.03,
        scorer_input_distribution_guard_temperature=0.02,
        scorer_input_contrast_floor_weight=0.75,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.6,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.4,
        posenet_yuv6_geometry_tether_weight=0.95,
        posenet_temporal_signal_floor_weight=0.85,
        posenet_temporal_signal_min_std_ratio=0.35,
        posenet_temporal_signal_min_mean_abs_ratio=0.45,
        scorer_domain_bootstrap_segnet_margin_weight=1.5,
        scorer_domain_bootstrap_segnet_hard_birth_weight=3.0,
        scorer_domain_bootstrap_segnet_hard_birth_min_ratio_floor=0.07,
        scorer_domain_bootstrap_segnet_hard_birth_support_preserve_weight=73.0,
        scorer_domain_bootstrap_segnet_hard_birth_outside_argmax_preserve_weight=19.0,
        scorer_domain_bootstrap_segnet_hard_birth_already_won_hard_preserve_weight=211.0,
        scorer_domain_bootstrap_segnet_hard_birth_already_won_rgb_preserve_weight=223.0,
        scorer_domain_bootstrap_segnet_hard_birth_already_won_margin_floor=1.25,
        scorer_domain_bootstrap_segnet_hard_birth_pose_trust_preserve_weight=37.0,
        scorer_domain_bootstrap_segnet_hard_birth_pose_target_weight=41.0,
        segnet_direct_live_class_balanced_ce_weight=0.25,
        segnet_direct_live_target_mass_floor_weight=0.4,
        segnet_direct_live_target_min_ratio_floor_weight=0.4,
        distillation_device="cpu",
        requested_distillation_device=None,
        allow_segnet_only_research=False,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.0,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
        coder_qat_c1a_sample_size=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
        recon_pixel_weight_path=None,
        decoder_weight_waterfill_plan=None,
        recon_pixel_weight_auto_discovery=None,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=1.0,
        recon_pixel_weight_normalize="mean",
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=50,
        telemetry_flush_interval_epochs=1,
        checkpoint_interval_epochs=1,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        optimizer_kind="adamw",
        hi_nerv_optimizer_policy={},
        optimizer_controls={},
        prioritized_pair_indices=(7, 2),
        random_seed=0,
        scorer_upstream_dir=REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
    )

    decode_call = captured["decode_call"]
    assert decode_call["num_pairs"] == 10
    assert decode_call["pair_indices"] == ()
    assert captured["bundle_num_pairs"] == 10
    assert captured["bundle_source_pair_indices"] == ()
    assert captured["model_num_pairs"] == 10
    assert captured["bundle_scorer_input_distribution_guard_weight"] == pytest.approx(0.25)
    guard_dual = captured["dual_ascent_constraints"]["hi_nerv_scorer_input_distribution_guard"]
    assert guard_dual["metric_name"] == "loss_part_scorer_input_distribution_guard"
    assert guard_dual["loss_weight_key"] == "scorer_input_guard"
    assert guard_dual["weight_scale"] == pytest.approx(0.25)
    assert captured["bundle_scorer_input_distribution_guard_saturation_margin"] == pytest.approx(0.03)
    assert captured["bundle_scorer_input_distribution_guard_temperature"] == pytest.approx(0.02)
    assert captured["bundle_scorer_input_contrast_floor_weight"] == pytest.approx(0.75)
    assert captured["bundle_scorer_input_contrast_floor_segnet_min_std_ratio"] == pytest.approx(0.6)
    assert captured["bundle_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"] == pytest.approx(0.4)
    assert captured["bundle_posenet_yuv6_geometry_tether_weight"] == pytest.approx(0.95)
    assert captured["bundle_posenet_temporal_signal_floor_weight"] == pytest.approx(0.85)
    assert captured["bundle_posenet_temporal_signal_min_std_ratio"] == pytest.approx(0.35)
    assert captured["bundle_posenet_temporal_signal_min_mean_abs_ratio"] == pytest.approx(0.45)
    assert captured["bundle_segnet_direct_live_rare_class_logit_weight"] == (pytest.approx(0.0))
    assert captured["bundle_segnet_direct_live_class_balanced_hinge_weight"] == (pytest.approx(0.0))
    assert captured["bundle_segnet_direct_live_class_balanced_ce_weight"] == (pytest.approx(0.25))
    assert captured["bundle_segnet_direct_live_class_balanced_squared_hinge_weight"] == pytest.approx(0.0)
    assert captured["bundle_segnet_direct_live_class_region_recon_weight"] == (pytest.approx(0.0))
    assert captured["bundle_segnet_direct_live_target_mass_floor_weight"] == (pytest.approx(0.4))
    assert captured["bundle_segnet_direct_live_target_min_ratio_floor_weight"] == pytest.approx(0.4)
    bootstrap_call = captured["scorer_domain_bootstrap_call"]
    assert bootstrap_call["scorer_teacher_present"] is True
    assert bootstrap_call["segnet_margin_bootstrap_weight"] == pytest.approx(1.5)
    assert bootstrap_call["segnet_hard_birth_bootstrap_weight"] == pytest.approx(3.0)
    assert bootstrap_call["segnet_hard_birth_bootstrap_min_ratio_floor"] == pytest.approx(0.07)
    bootstrap_metadata = artifact.as_dict()["substrate_artifact_metadata"]["score_aware_training"][
        "output_head_target_bias_init"
    ]["scorer_domain_bootstrap"]
    assert bootstrap_metadata["segnet_hard_birth_bootstrap_requested_weight"] == pytest.approx(3.0)
    assert bootstrap_metadata["segnet_hard_birth_bootstrap_effective_weight"] == pytest.approx(3.0)
    assert bootstrap_metadata["segnet_hard_birth_bootstrap_request_consumed"] is True
    assert bootstrap_metadata["actuators_enabled_effective"] == {
        "segnet_teacher": True,
        "pose_teacher": True,
        "hard_birth": True,
    }
    assert bootstrap_metadata["exact_posenet_target_pose"]["enabled"] is True
    target_region_birth = bootstrap_metadata["target_region_birth_actuator"]
    assert target_region_birth["accepted"] is True
    assert target_region_birth["net_target_support_delta"] == pytest.approx(5.0)
    target_birth_call = captured["target_region_birth_call"]
    assert target_birth_call["scorer_teacher_present"] is True
    assert target_birth_call["pose_teacher_present"] is True
    assert target_birth_call["require_pose_trust"] is True
    assert target_birth_call["target0_shape"] == (8, 384, 512, 3)
    assert target_birth_call["target1_shape"] == (8, 384, 512, 3)
    assert target_birth_call["pair_indices"] == tuple(range(8))
    assert target_birth_call["target_segnet_argmax_shape"] == (8, 384, 512)
    assert target_birth_call["max_steps"] == 32
    assert target_birth_call["target_min_region_ratio"] == pytest.approx(0.07)
    assert target_birth_call["lambda_support_preserve"] == pytest.approx(73.0)
    assert target_birth_call["lambda_outside_argmax_preserve"] == pytest.approx(19.0)
    assert target_birth_call["lambda_already_won_hard_preserve"] == pytest.approx(211.0)
    assert target_birth_call["lambda_already_won_rgb_preserve"] == pytest.approx(223.0)
    assert target_birth_call["already_won_margin_floor"] == pytest.approx(1.25)
    assert target_birth_call["lambda_pose_trust_preserve"] == pytest.approx(37.0)
    assert target_birth_call["lambda_pose_target"] == pytest.approx(41.0)
    assert (
        bootstrap_call["pair_local_smoke_artifact_dir"]
        == (tmp_path / "private_smoke_full_priority_hydration" / "hi_nerv_pair_local_actuator_smoke").as_posix()
    )
    temporal_dual = captured["dual_ascent_constraints"]["hi_nerv_posenet_temporal_signal_floor"]
    assert temporal_dual["metric_name"] == ("loss_part_posenet_temporal_signal_floor")
    assert temporal_dual["loss_weight_key"] == "posenet_temporal_signal_floor"
    assert temporal_dual["weight_scale"] == pytest.approx(0.85)
    geometry_dual = captured["dual_ascent_constraints"]["hi_nerv_posenet_yuv6_geometry_tether"]
    assert geometry_dual["metric_name"] == ("loss_part_posenet_yuv6_geometry_tether")
    assert geometry_dual["loss_weight_key"] == "posenet_yuv6_geometry_tether"
    assert geometry_dual["weight_scale"] == pytest.approx(0.95)
    assert captured["curriculum_stage_loss_weights"]
    assert all(
        weights["posenet_yuv6_geometry_tether"] == pytest.approx(0.75)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["posenet_temporal_signal_floor"] == pytest.approx(0.625)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_target_mass_floor"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_class_balanced_ce"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_target_min_ratio_floor"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_rare_class_logit"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_class_balanced_hinge"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_class_balanced_squared_hinge"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert all(
        weights["segnet_direct_live_class_region_recon"] == pytest.approx(1.0)
        for weights in captured["curriculum_stage_loss_weights"]
    )
    assert captured["scorer_space_step_guard_enabled"] is True
    assert captured["scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"] == pytest.approx(0.4)
    assert captured["scorer_space_step_guard_min_post_segnet_occupied_class_fraction"] == pytest.approx(0.4)
    assert captured["scorer_space_step_guard_max_post_segnet_contrast_ratio"] == (pytest.approx(4.25))
    assert captured["scorer_space_step_guard_max_post_segnet_distribution_mae"] == (
        pytest.approx(runner_mod.HI_NERV_SEGNET_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD)
    )
    assert captured["scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"] == (
        pytest.approx(runner_mod.HI_NERV_POSENET_YUV6_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD)
    )
    assert captured["scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"] == (
        pytest.approx(runner_mod.HI_NERV_POSENET_YUV6_CONTRAST_RATIO_MAX_FOR_STEP_GUARD)
    )
    assert captured["scorer_space_step_guard_backtracking_steps"] == 6
    assert captured["scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"] == pytest.approx(
        runner_mod.HI_NERV_BOOTSTRAP_DIRECT_NONRATE_SCORE_MAX_WORSENING_FOR_STEP_GUARD
    )
    assert captured["scorer_space_step_guard_backtracking_shrink"] == pytest.approx(0.5)
    assert captured["scorer_support_ladder_enabled"] is True
    assert captured["scorer_support_ladder_target_coverage_floor"] == pytest.approx(
        runner_mod.HI_NERV_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE
    )
    assert captured["scorer_support_ladder_target_min_ratio_floor"] == pytest.approx(
        runner_mod.HI_NERV_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE
    )
    assert captured["scorer_support_ladder_patience_steps"] == 1
    assert captured["scorer_support_ladder_growth_factor"] == pytest.approx(2.0)
    assert captured["scorer_support_ladder_max_multiplier"] == pytest.approx(16.0)
    assert captured["scorer_support_ladder_base_loss_max_when_active"] == pytest.approx(0.25)
    assert captured["run_prioritized_pair_indices"] == (7, 2)
    assert captured["run_pr95_curriculum_total_epochs"] == 8
    assert captured["checkpoint_selection_metric_key"] == ("loss_part_segnet_direct_live_escape_selection")
    assert captured["checkpoint_selection_metric_required"] is True
    metadata = artifact.as_dict()["substrate_artifact_metadata"]
    training = metadata["score_aware_training"]
    assert metadata["model_num_pairs"] == 10
    assert metadata["hydrated_target_pair_count"] == 10
    assert metadata["source_pair_indices"] == []
    assert metadata["sparse_prioritized_target_hydration"] is False
    priority = training["prioritized_pair_training"]
    step_guard = training["scorer_space_step_guard"]
    support_ladder = training["segnet_direct_live_class_support_ladder"]
    assert step_guard["enabled"] is True
    assert step_guard["bound_to_shared_mlx_adapter"] is True
    assert step_guard["min_post_segnet_occupied_class_fraction"] == pytest.approx(0.4)
    assert step_guard["max_post_segnet_contrast_ratio"] == pytest.approx(4.25)
    assert step_guard["max_post_segnet_distribution_mae"] == pytest.approx(
        runner_mod.HI_NERV_SEGNET_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD
    )
    assert support_ladder["enabled"] is True
    assert support_ladder["bound_to_shared_mlx_adapter"] is True
    assert support_ladder["target_coverage_floor"] == pytest.approx(
        runner_mod.HI_NERV_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE
    )
    assert support_ladder["target_min_ratio_floor"] == pytest.approx(
        runner_mod.HI_NERV_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE
    )
    assert support_ladder["stage_order"][0] == "target_mass_floor_plus_rare_class_logit"
    assert step_guard["max_post_posenet_yuv6_distribution_mae"] == pytest.approx(
        runner_mod.HI_NERV_POSENET_YUV6_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD
    )
    assert step_guard["max_post_posenet_yuv6_contrast_ratio"] == pytest.approx(
        runner_mod.HI_NERV_POSENET_YUV6_CONTRAST_RATIO_MAX_FOR_STEP_GUARD
    )
    assert step_guard["backtracking_steps"] == 6
    assert step_guard["backtracking_shrink"] == pytest.approx(0.5)
    assert priority["sampling_scope"] == ("full_video_target_hydration_with_prioritized_sampling")
    assert priority["target_hydration_pair_indices_consumed"] is False
    assert priority["target_hydration_pair_indices_consumed_by_renderer_bundle"] is False
    assert training["pr95_curriculum_total_epochs"] is None
    assert training["pr95_curriculum_total_epochs_consumed"] is False
    guard = training["scorer_input_distribution_guard"]
    assert guard["enabled"] is True
    assert guard["bound_to_renderer_bundle"] is True
    assert guard["weight"] == pytest.approx(0.25)
    assert guard["saturation_margin"] == pytest.approx(0.03)
    assert guard["temperature"] == pytest.approx(0.02)
    contrast = training["scorer_input_contrast_floor"]
    assert contrast["enabled"] is True
    assert contrast["bound_to_renderer_bundle"] is True
    assert contrast["weight"] == pytest.approx(0.75)
    assert contrast["segnet_last_rgb_min_std_ratio"] == pytest.approx(0.6)
    assert contrast["posenet_yuv6_pair_min_std_ratio"] == pytest.approx(0.4)
    assert contrast["domains"] == {
        "segnet": "last_frame_rgb_after_eval_roundtrip",
        "posenet": "two_frame_pr95_yuv6_after_eval_roundtrip",
    }
    temporal = training["posenet_temporal_signal_floor"]
    assert temporal["enabled"] is True
    assert temporal["bound_to_renderer_bundle"] is True
    assert temporal["weight"] == pytest.approx(0.85)
    assert temporal["min_std_ratio"] == pytest.approx(0.35)
    assert temporal["min_mean_abs_ratio"] == pytest.approx(0.45)
    init_call = captured["output_head_target_bias_init_call"]
    assert init_call["target0_shape"] == (10, 384, 512, 3)
    assert init_call["target1_shape"] == (10, 384, 512, 3)
    assert init_call["epsilon"] == pytest.approx(1.0 / 1024.0)
    output_init = training["output_head_target_bias_init"]
    assert output_init["enabled"] is True
    assert output_init["bound_to_renderer_model"] is True
    assert output_init["runtime_sidecar_bytes"] == 0
    assert output_init["archive_charged_decoder_tensors"] == [
        "decoder.bootstrap.weight",
        "decoder.target_region_birth.delta",
        "head_rgb_0.bias",
        "head_rgb_0.weight",
        "head_rgb_1.bias",
        "head_rgb_1.weight",
    ]
    contrast_init = output_init["contrast_init"]
    assert contrast_init["enabled"] is True
    assert contrast_init["source_pair_indices"] == list(range(8))
    assert contrast_init["pair_index_semantics"] == "identity_local_rows_are_source_pairs"
    assert contrast_init["max_pairs"] == 8
    contrast_call = captured["output_head_target_contrast_init_call"]
    assert contrast_call["target0_shape"] == (8, 384, 512, 3)
    assert contrast_call["target1_shape"] == (8, 384, 512, 3)
    assert contrast_call["pair_indices"] == list(range(8))
    assert contrast_call["min_output_std"] == pytest.approx(1.0e-6)
    assert contrast_call["max_gain"] == pytest.approx(4096.0)


def test_hinerv_private_smoke_forwards_explicit_pr95_curriculum_total_epochs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import archive_candidate as hinerv_archive_candidate
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    captured: dict[str, object] = {}

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def configure_decoder_fake_quant_forward(self, **_kwargs):
            return None

        def initialize_output_head_bias_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            epsilon,
        ):
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.bias",
                    "head_rgb_1.bias",
                ],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            steps,
            learning_rate,
            rgb_weight,
            yuv6_weight,
            temporal_delta_weight,
            contrast_floor_weight,
            rgb_std_min_ratio,
            yuv6_temporal_std_min_ratio,
            weight_decay,
            grad_clip_max_norm,
            target_segnet_argmax_1=None,
            scorer_teacher=None,
            segnet_margin_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_min_ratio_floor=0.02,
            pair_local_smoke_artifact_dir=None,
        ):
            return _fake_hinerv_scorer_domain_bootstrap_payload(
                pair_indices,
                steps=steps,
                learning_rate=learning_rate,
                rgb_weight=rgb_weight,
                yuv6_weight=yuv6_weight,
                temporal_delta_weight=temporal_delta_weight,
                contrast_floor_weight=contrast_floor_weight,
                rgb_std_min_ratio=rgb_std_min_ratio,
                yuv6_temporal_std_min_ratio=yuv6_temporal_std_min_ratio,
                weight_decay=weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            )

        def num_parameters(self):
            return 123

    class FakeArtifact:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def as_dict(self) -> dict[str, object]:
            return dict(self._payload)

    def fake_decode_mlx_targets(
        _video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        captured["decode_pair_indices"] = tuple(pair_indices or ())
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def fake_export_hi_nerv_mlx_archive(
        _model_obj,
        archive_output_dir,
        **kwargs,
    ):
        captured["export_hard_byte_ceiling"] = kwargs.get("hard_byte_ceiling")
        Path(archive_output_dir).mkdir(parents=True, exist_ok=True)
        archive = Path(archive_output_dir) / "fake_hinerv_export_archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return archive, runner_mod._sha256_file(archive), archive.stat().st_size

    class FakeSegNetTeacher:
        num_classes = 3
        frame_count = 10
        upstream_segnet_safetensors_sha256 = "0" * 64

        def teacher_argmax_for_indices(self, indices):
            import mlx.core as mx

            return mx.zeros((int(indices.shape[0]), 384, 512), dtype=mx.int32)

        def teacher_logits_for_frames_nhwc01(self, frames):
            import mlx.core as mx

            batch, height, width, _channels = frames.shape
            zeros = mx.zeros((batch, height, width, 1), dtype=mx.float32)
            ones = mx.ones((batch, height, width, 1), dtype=mx.float32)
            return mx.concatenate([ones, zeros, zeros], axis=-1)

    class FakePoseNetTeacher:
        pose_dims = 6
        upstream_posenet_safetensors_sha256 = "1" * 64

        def teacher_pose_for_yuv6_pair_nhwc(self, yuv6_pair):
            import mlx.core as mx

            return mx.zeros((int(yuv6_pair.shape[0]), self.pose_dims), dtype=mx.float32)

    def fake_build_hi_nerv_archive_replay_components(
        archive_path,
        batch,
        *,
        target_rgb_0,
        target_rgb_1,
        scorer_teacher,
        pose_scorer_teacher,
        candidate_kind,
    ):
        captured["archive_replay_archive_path"] = Path(archive_path).name
        captured["archive_replay_batch"] = dict(batch)
        captured["archive_replay_target_shape"] = tuple(target_rgb_0.shape)
        captured["archive_replay_target1_shape"] = tuple(target_rgb_1.shape)
        captured["archive_replay_scorer_teacher"] = scorer_teacher
        captured["archive_replay_pose_scorer_teacher"] = pose_scorer_teacher
        captured["archive_replay_candidate_kind"] = str(candidate_kind)
        return {
            "archive_replay_pair_count": 1.0,
            "parseback_rgb_pair_mse": 0.0,
        }

    def fake_write_mlx_renderer_prefilter_profile(**kwargs):
        captured["prefilter_profile_scorer_device"] = kwargs["scorer_device"]
        captured["prefilter_profile_scorer_batch_pairs"] = int(
            kwargs["scorer_batch_pairs"]
        )
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema": "mlx_renderer_prefilter_profile.v1",
                    "run_id": kwargs["run_id"],
                    "archive_bytes": int(kwargs["archive_bytes"]),
                    "archive_sha256": str(kwargs["archive_sha256"]),
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )

    def fake_run_mlx_score_aware_full_main(**kwargs):
        captured["run_pr95_curriculum_total_epochs"] = int(kwargs["pr95_curriculum_total_epochs"])
        captured["run_pr95_enabled"] = bool(kwargs["pr95_faithful_curriculum_enabled"])
        captured["archive_selection_replay_required"] = bool(kwargs["archive_selection_replay_required"])
        captured["archive_selection_replay_batch_size"] = int(kwargs["archive_selection_replay_batch_size"])
        replay_components = kwargs["bundle"].archive_replay_components_fn(
            tmp_path / "parseback.zip",
            {"local_pair_indices": np.array([0], dtype=np.int64)},
            "ema",
        )
        captured["archive_replay_components"] = dict(replay_components)
        exported = kwargs["bundle"].export_archive_fn(
            kwargs["bundle"].model,
            tmp_path / "fake_hinerv_export",
        )
        captured["export_callback_result"] = exported
        archive = tmp_path / "fake_hinerv_pr95_total_archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return FakeArtifact(
            {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": runner_mod._sha256_file(archive),
                "substrate_artifact_metadata": dict(kwargs["bundle"].substrate_artifact_metadata),
            }
        )

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "build_mlx_segnet_pair_teacher",
        lambda *_args, **_kwargs: FakeSegNetTeacher(),
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "build_mlx_posenet_pair_teacher",
        lambda *_args, **_kwargs: FakePoseNetTeacher(),
    )
    monkeypatch.setattr(
        hinerv_archive_candidate,
        "export_hi_nerv_mlx_archive",
        fake_export_hi_nerv_mlx_archive,
    )
    monkeypatch.setattr(
        hinerv_archive_candidate,
        "build_hi_nerv_archive_replay_components",
        fake_build_hi_nerv_archive_replay_components,
    )
    import tac.local_acceleration.mlx_renderer_prefilter_profile as prefilter_mod

    monkeypatch.setattr(
        prefilter_mod,
        "write_mlx_renderer_prefilter_profile",
        fake_write_mlx_renderer_prefilter_profile,
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )

    artifact = runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
        output_dir=tmp_path / "private_smoke_pr95_total",
        num_pairs=2,
        epochs=3,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=1,
        fine_injection_block_index=4,
        decoder_codec="portfolio_auto",
        hi_nerv_latent_codec="int16_brotli_q11",
        ema_decay=0.9,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_direct_live_distillation_weight=0.25,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        recon_loss_stage_weight=1.0,
        segnet_loss_stage_weight=1.0,
        pose_loss_stage_weight=1.0,
        scorer_input_guard_stage_weight=1.0,
        scorer_input_contrast_floor_stage_weight=None,
        scorer_input_shape_tether_stage_weight=None,
        segnet_direct_live_stage_weight=None,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        scorer_domain_bootstrap_segnet_hard_birth_weight=0.0,
        distillation_device="cpu",
        requested_distillation_device=None,
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0001,
        coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
        coder_qat_c1a_sample_size=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
        recon_pixel_weight_path=None,
        decoder_weight_waterfill_plan=None,
        recon_pixel_weight_auto_discovery=None,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=1.0,
        recon_pixel_weight_normalize="mean",
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=50,
        telemetry_flush_interval_epochs=1,
        checkpoint_interval_epochs=1,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        optimizer_kind="adamw",
        hi_nerv_optimizer_policy={
            "pr95_faithful_curriculum_enabled": True,
            "pr95_muon_policy": "faithful_stage8_only",
        },
        pr95_curriculum_total_epochs=29_650,
        optimizer_controls={},
        prioritized_pair_indices=(),
        archive_selection_replay_required=True,
        archive_selection_replay_batch_size=2,
        random_seed=0,
        scorer_upstream_dir=REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-private-smoke-candidate",
            "num_pairs": 2,
            "latent_dim": 4,
            "embed_dim": 4,
            "decoder_channel": 4,
            "decoder_codec": "portfolio_auto",
            "hi_nerv_latent_codec": "int16_brotli_q11",
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 120_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "local_grid_levels": 2,
            "local_grid_channels": 4,
            "convnext_mlp_ratio": 2,
            "convnext_kernel_size": 7,
            "mid_injection_block_index": 1,
            "fine_injection_block_index": 4,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "hi_nerv",
                "control_semantics": "local_receiver_visible_grid_search_nearest_target",
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        archive_section_telemetry={
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "archive_zip_bytes": 333,
            "sections": [
                {"name": "decoder_state", "role": "decoder", "bytes": 210},
            ],
        },
    )

    metadata = artifact.as_dict()["substrate_artifact_metadata"]
    training = metadata["score_aware_training"]
    consumption = metadata["modelsize_candidate_consumption"]
    assert consumption["attached"] is True
    assert consumption["candidate_id"] == "hinerv-private-smoke-candidate"
    assert consumption["consumed_by_runner_config"] is True
    assert consumption["consumed_by_decoder_codec"] is True
    assert consumption["consumed_by_archive_export_hard_byte_ceiling"] is True
    assert consumption["archive_export_hard_byte_ceiling_measurement_bypass"] is False
    assert consumption["hard_byte_ceiling_consumed_by_train_time_dual_ascent"] is True
    assert consumption["train_time_section_byte_control_active"] is True
    assert consumption["train_time_section_byte_control_blockers"] == []
    assert consumption["hard_byte_ceiling_train_time_dual_ascent_blockers"] == []
    assert consumption["hard_byte_ceiling"] == 178_000
    assert captured["export_hard_byte_ceiling"] == 178_000
    assert captured["archive_selection_replay_required"] is True
    assert captured["archive_selection_replay_batch_size"] == 2
    assert captured["archive_replay_archive_path"] == "parseback.zip"
    assert captured["archive_replay_batch"]["local_pair_indices"].tolist() == [0]
    assert captured["archive_replay_target_shape"] == (2, 384, 512, 3)
    assert captured["archive_replay_target1_shape"] == (2, 384, 512, 3)
    assert captured["archive_replay_scorer_teacher"] is None
    assert captured["archive_replay_pose_scorer_teacher"] is not None
    assert int(captured["archive_replay_pose_scorer_teacher"].pose_dims) == 6
    assert captured["archive_replay_candidate_kind"] == "ema"
    assert captured["archive_replay_components"]["parseback_rgb_pair_mse"] == 0.0
    assert metadata["config"]["latent_dim_mid"] == 4
    assert metadata["config"]["embed_dim"] == 4
    assert metadata["config"]["decoder_channels"] == [4, 4, 4, 4, 4, 4, 4]
    assert captured["run_pr95_enabled"] is True
    assert captured["run_pr95_curriculum_total_epochs"] == 29_650
    assert training["pr95_faithful_curriculum_enabled"] is True
    assert training["pr95_curriculum_total_epochs"] == 29_650
    assert training["pr95_curriculum_total_epochs_consumed"] is True


def test_hinerv_private_smoke_refuses_inert_hard_byte_ceiling_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    train_called = False

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def configure_decoder_fake_quant_forward(self, **_kwargs):
            return None

        def initialize_output_head_bias_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            epsilon,
        ):
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            steps,
            learning_rate,
            rgb_weight,
            yuv6_weight,
            temporal_delta_weight,
            contrast_floor_weight,
            rgb_std_min_ratio,
            yuv6_temporal_std_min_ratio,
            weight_decay,
            grad_clip_max_norm,
            target_segnet_argmax_1=None,
            scorer_teacher=None,
            segnet_margin_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_min_ratio_floor=0.02,
            pair_local_smoke_artifact_dir=None,
        ):
            return _fake_hinerv_scorer_domain_bootstrap_payload(
                pair_indices,
                steps=steps,
                learning_rate=learning_rate,
                rgb_weight=rgb_weight,
                yuv6_weight=yuv6_weight,
                temporal_delta_weight=temporal_delta_weight,
                contrast_floor_weight=contrast_floor_weight,
                rgb_std_min_ratio=rgb_std_min_ratio,
                yuv6_temporal_std_min_ratio=yuv6_temporal_std_min_ratio,
                weight_decay=weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            )

        def num_parameters(self):
            return 123

    def fake_decode_mlx_targets(
        _video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def fail_train(**_kwargs):
        nonlocal train_called
        train_called = True
        raise AssertionError("inactive hard-byte ceiling must refuse before training")

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fail_train,
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )

    with pytest.raises(
        CompactRendererMlxSpineRunnerError,
        match="train-time section byte controller is inactive",
    ):
        runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=tmp_path / "private_smoke_inert_byte_cap",
            num_pairs=2,
            epochs=3,
            batch_pair_indices_per_step=1,
            learning_rate=1e-3,
            source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            use_hierarchical_feature_grid=True,
            use_convnext_blocks=True,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=7,
            mid_injection_block_index=1,
            fine_injection_block_index=4,
            decoder_codec="portfolio_auto",
            hi_nerv_latent_codec="int16_brotli_q11",
            hard_byte_ceiling=178_000,
            ema_decay=0.9,
            segnet_distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_distillation_loss="mse",
            pose_distillation_huber_delta=1.0,
            recon_loss_stage_weight=1.0,
            segnet_loss_stage_weight=1.0,
            pose_loss_stage_weight=1.0,
            scorer_input_guard_stage_weight=1.0,
            scorer_input_contrast_floor_stage_weight=None,
            scorer_input_shape_tether_stage_weight=None,
            segnet_direct_live_stage_weight=None,
            segnet_distillation_objective="kl_t2",
            distillation_temperature=2.0,
            segnet_tau_boundary=1.0,
            segnet_hinge_margin=1.0,
            scorer_domain_bootstrap_segnet_hard_birth_weight=0.0,
            distillation_device="cpu",
            requested_distillation_device=None,
            allow_segnet_only_research=False,
            coder_aware_qat=False,
            coder_qat_quant_bits=8,
            coder_qat_quant_residual_weight=0.0,
            coder_qat_magnitude_weight=0.0,
            coder_qat_delta_weight=0.0,
            coder_qat_c1a_entropy_weight=0.0,
            coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
            coder_qat_c1a_sample_size=(runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE),
            recon_pixel_weight_path=None,
            decoder_weight_waterfill_plan=None,
            recon_pixel_weight_auto_discovery=None,
            auto_segnet_boundary_recon_weight=False,
            recon_pixel_weight_tau=1.0,
            recon_pixel_weight_normalize="mean",
            mlx_prefilter_scorer_device=None,
            mlx_prefilter_scorer_batch_pairs=1,
            mlx_prefilter_progress_every=50,
            telemetry_flush_interval_epochs=1,
            checkpoint_interval_epochs=1,
            checkpoint_dir=None,
            resume_from_checkpoint=None,
            optimizer_kind="adamw",
            hi_nerv_optimizer_policy={},
            optimizer_controls={},
            prioritized_pair_indices=(),
            random_seed=0,
            scorer_upstream_dir=REPO_ROOT / "upstream",
            repo_root=REPO_ROOT,
        )

    assert train_called is False


def test_hinerv_private_smoke_rejects_out_of_range_prioritized_pairs(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="out-of-range"):
        runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=tmp_path / "private_smoke_out_of_range_priority",
            num_pairs=4,
            epochs=1,
            batch_pair_indices_per_step=2,
            learning_rate=1e-3,
            source_video_path=tmp_path / "not_loaded_when_pair_guard_fails.mkv",
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            use_hierarchical_feature_grid=False,
            use_convnext_blocks=False,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=7,
            mid_injection_block_index=1,
            fine_injection_block_index=4,
            decoder_codec="portfolio_auto",
            ema_decay=0.9,
            segnet_distillation_weight=1.0,
            pose_distillation_weight=1.0,
            pose_distillation_loss="mse",
            pose_distillation_huber_delta=1.0,
            recon_loss_stage_weight=1.0,
            segnet_loss_stage_weight=1.0,
            pose_loss_stage_weight=1.0,
            scorer_input_guard_stage_weight=1.0,
            scorer_input_contrast_floor_stage_weight=None,
            scorer_input_shape_tether_stage_weight=None,
            segnet_direct_live_stage_weight=None,
            segnet_distillation_objective="kl_t2",
            distillation_temperature=2.0,
            segnet_tau_boundary=1.0,
            segnet_hinge_margin=1.0,
            distillation_device="cpu",
            requested_distillation_device=None,
            allow_segnet_only_research=False,
            coder_aware_qat=False,
            coder_qat_quant_bits=8,
            coder_qat_quant_residual_weight=0.0,
            coder_qat_magnitude_weight=0.0,
            coder_qat_delta_weight=0.0,
            coder_qat_c1a_entropy_weight=0.0,
            coder_qat_c1a_sigma=0.0,
            coder_qat_c1a_sample_size=0,
            recon_pixel_weight_path=None,
            decoder_weight_waterfill_plan=None,
            recon_pixel_weight_auto_discovery=None,
            auto_segnet_boundary_recon_weight=False,
            recon_pixel_weight_tau=1.0,
            recon_pixel_weight_normalize="mean",
            mlx_prefilter_scorer_device=None,
            mlx_prefilter_scorer_batch_pairs=1,
            mlx_prefilter_progress_every=50,
            telemetry_flush_interval_epochs=1,
            checkpoint_interval_epochs=1,
            checkpoint_dir=None,
            resume_from_checkpoint=None,
            optimizer_kind="adamw",
            hi_nerv_optimizer_policy={},
            optimizer_controls={},
            prioritized_pair_indices=(4,),
            random_seed=0,
            scorer_upstream_dir=REPO_ROOT / "upstream",
            repo_root=REPO_ROOT,
        )


def test_adapt_pr95_mlx_report_emits_spine_acquisition_and_runner(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "stage08.pt"
    latents = tmp_path / "stage08.latents.npy"
    manifest = tmp_path / "stage08.pt.export_manifest.json"
    weights.write_bytes(b"trained-mlx-weights")
    latents.write_bytes(b"trained-mlx-latents")
    manifest.write_text(
        json.dumps({"schema": "pr95_mlx_long_training_pytorch_export_manifest.v1"}),
        encoding="utf-8",
    )
    report_path = tmp_path / "pr95_mlx_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "pr95_mlx_long_training_plan.v1",
                "mode": "executed_smoke",
                "evidence_grade": "[macOS-MLX research-signal]",
                "source_video_frame_count": 4,
                "max_frames": 4,
                "checkpoint_artifacts": [
                    {
                        "stage_index": 8,
                        "global_epoch": 8,
                        "pytorch_state_dict_path": weights.as_posix(),
                        "latents_path": latents.as_posix(),
                        "pytorch_export_manifest_path": manifest.as_posix(),
                        "pytorch_export_succeeded": True,
                        "trained_latents_exported": True,
                        "evidence_grade": "[macOS-MLX research-signal]",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    followup_path = tmp_path / "hprc_queue_followup_report.json"
    followup_path.write_text(
        json.dumps(
            {
                "schema": "hprc_queue_followup_report.v1",
                "training_result_path": "prior_hprc_rate_collapse.json",
                "archive": {
                    "archive_zip_sha256": "f" * 64,
                    "archive_zip_bytes": 217365,
                },
                "planner_learning_signals": [
                    {
                        "schema": "hprc_planner_learning_signal.v1",
                        "signal_id": ("hprc_rate_feasible_but_resolution_distortion_bound"),
                        "status": "route_to_pose_geometry_or_predictive_redesign",
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = adapt_pr95_mlx_report_to_spine(
        pr95_mlx_report_path=report_path,
        output_dir=tmp_path / "out",
        hard_byte_ceilings=(178_000,),
        hprc_queue_followup_report_paths=(followup_path,),
        repo_root=REPO_ROOT,
    )

    assert out["schema"] == COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["spine_adapter_report_path"]).is_file()
    acquisition = json.loads(Path(out["acquisition_report_path"]).read_text())
    assert acquisition["rows"][0]["family"] == "pr95_hnerv"
    assert acquisition["rows"][0]["coverage"]["declared_pairs"] == 2
    assert acquisition["rows"][0]["coverage"]["valid_for_base_comparison"] is False
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    assert runner["selected_runner_rows"] == []
    assert "no_full_coverage_compact_base_candidate" in runner["blockers"]
    assert "implementation_readiness_blocked_fake_or_incomplete_candidate" in runner["blockers"]
    assert runner["hprc_queue_followup_signal_rows"][0]["signal_id"] == (
        "hprc_rate_feasible_but_resolution_distortion_bound"
    )
    assert out["hprc_queue_followup_report_paths"] == [followup_path.as_posix()]
    assert "mlx_local_report_is_advisory_not_score_authority" in out["blockers"]


def test_plan_only_report_keeps_all_compact_families_false_authority(
    tmp_path: Path,
) -> None:
    report = build_plan_only_report(
        output_dir=tmp_path / "plan",
        hard_byte_ceilings=(178_000, 216_000),
        repo_root=REPO_ROOT,
    )

    assert report["nerv_oss_flag_audit"]["schema"] == "nerv_oss_flag_audit.v1"
    assert "--modelsize" in report["nerv_oss_flag_audit"]["hnerv_high_ev_flags"]
    assert "--modelsize" in report["nerv_oss_flag_audit"]["snerv_high_ev_flags"]
    assert "--quant-level" in report["nerv_oss_flag_audit"]["hinerv_high_ev_flags"]
    assert report["hinerv_modelsize_budget"]["schema"] == "nerv_modelsize_budget.v1"
    assert report["hinerv_modelsize_budget"]["selected_candidate_count"] > 0
    assert report["hinerv_modelsize_budget"]["score_claim"] is False
    assert report["snerv_modelsize_budget"]["schema"] == "snerv_modelsize_budget.v1"
    assert report["snerv_modelsize_budget"]["selected_candidate_count"] > 0
    assert report["snerv_modelsize_budget"]["score_claim"] is False
    campaign_plan = report["nerv_long_training_campaign_plan"]
    assert campaign_plan["schema"] == "nerv_long_training_campaign_plan.v1"
    assert campaign_plan["score_claim"] is False
    assert campaign_plan["ready_for_exact_eval_dispatch"] is False
    assert campaign_plan["family_counts"]["hi_nerv"] > 0
    assert campaign_plan["family_counts"]["snerv"] > 0
    launchable_local_rows = [row for row in campaign_plan["campaign_rows"] if row["local_mlx_launch_command_ready"]]
    assert campaign_plan["launchable_local_row_count"] == len(launchable_local_rows)
    assert launchable_local_rows == []
    assert all(row["command_argv"] for row in campaign_plan["campaign_rows"])
    assert any(
        row["family"] == "hi_nerv" and "requires_verified_joint_p18_p19_recon_pixel_weight_artifact" in row["blockers"]
        for row in campaign_plan["campaign_rows"]
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"]["queue_status_is_runnable_plan"]
        is row["local_mlx_launch_command_ready"]
        for row in campaign_plan["campaign_rows"]
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"]["queue_status_is_receiver_proof"] is False
        and row["experiment_queue_entry"]["launch_authority_contract"]["queue_status_is_cpu_replay_proof"] is False
        and row["experiment_queue_entry"]["launch_authority_contract"]["queue_status_is_exact_eval_authority"] is False
        for row in campaign_plan["campaign_rows"]
    )
    assert campaign_plan["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert campaign_plan["experiment_queue_experiment_count"] == campaign_plan["campaign_row_count"]
    snerv_campaign_rows = [row for row in campaign_plan["campaign_rows"] if row["family"] == "snerv"]
    assert snerv_campaign_rows
    assert all(
        (not row["local_mlx_launch_command_ready"]) or row["hard_byte_ceiling_satisfied_for_long_training"]
        for row in snerv_campaign_rows
    )
    assert all(
        row["score_lowering_gate"]["command_materialized"] is row["local_mlx_launch_command_ready"]
        and row["score_lowering_gate"]["local_mlx_executable"] is row["local_mlx_launch_command_ready"]
        and row["score_lowering_gate"]["prelaunch_allowed"] is row["local_mlx_launch_command_ready"]
        and row["score_lowering_gate"]["promotion_prelaunch_allowed"] is False
        and row["score_lowering_gate"]["cpu_replay_ready"] is False
        and row["score_lowering_gate"]["exact_gate_ready"] is False
        and row["score_lowering_gate"]["score_claim"] is False
        and row["score_lowering_gate"]["score_claim_valid"] is False
        and row["score_lowering_gate"]["promotion_eligible"] is False
        and row["score_lowering_gate"]["ready_for_exact_eval_dispatch"] is False
        for row in snerv_campaign_rows
    )
    assert any(
        "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training" in row["blockers"]
        for row in snerv_campaign_rows
    )
    assert any(
        "snerv_byte_closed_archive_export_missing" in row["score_lowering_gate"]["promotion_blockers"]
        for row in snerv_campaign_rows
    )
    assert report["nerv_stack_synergy_audit"]["schema"] == ("nerv_stack_synergy_audit.v1")
    assert report["nerv_stack_synergy_audit"]["score_claim"] is False
    assert {row["stack_id"] for row in report["nerv_stack_synergy_audit"]["stacks"]} == {
        "hi_nerv",
        "snerv",
    }

    families = {row["family"]: row for row in report["target_family_rows"]}
    assert "pr95_hnerv" in families
    assert "hi_nerv" in families
    assert "snerv" in families
    assert "rnerv" in families
    assert "sr_nerv" in families
    assert "boostnerv" in families
    assert "pvq_nerv" in families
    assert "rt_vq_nerv" in families
    assert "pact_nerv_selector_v4" in families
    assert families["pr95_hnerv"]["status"] == ("executable_mlx_archive_export_control_arm")
    assert "not a PR95-faithful reproduction" in families["pr95_hnerv"]["execution_scope"]
    assert families["pact_nerv_vq"]["status"] == "executable_mlx_backend_available"
    assert families["pact_nerv_selector_v4"]["status"] == ("executable_mlx_backend_available")
    assert families["pact_nerv_selector_v4"]["section_value_profiler"] == (
        "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
    )
    assert "pact_nerv_selector_v4" in families["pact_nerv_selector_v4"]["trainer_entrypoint"]
    assert families["pvq_nerv"]["status"] == "executable_via_pact_nerv_vq_adapter"
    assert families["hi_nerv"]["status"] == ("mlx_archive_export_adapter_available_distortion_fit_actuator_pending")
    assert families["hi_nerv"]["trainer_entrypoint"].endswith("--execute-family hi_nerv")
    assert families["hi_nerv"]["archive_exporter"] == (
        "tac.substrates.hi_nerv.archive_candidate.export_hi_nerv_mlx_archive"
    )
    assert families["hi_nerv"]["stack_role"] == "primary_carrier"
    assert "super-small-rate-by-design" in families["hi_nerv"]["rate_axis_evidence"]
    assert "cheap bytes alone cannot promote" in families["hi_nerv"]["distortion_fit_blocker"]
    hinerv_plan = families["hi_nerv"]["score_aware_carrier_training_plan"]
    assert hinerv_plan["planner_action"] == ("run_receiver_closed_modelsize_ladder_before_score_aware_training")
    assert hinerv_plan["carrier_fit_status"] == "unusable"
    assert hinerv_plan["allocator_target_surface"] == "decoder_weights"
    assert hinerv_plan["score_claim"] is False
    assert hinerv_plan["ready_for_exact_eval_dispatch"] is False
    assert "carrier_fit_unusable_d_seg" in hinerv_plan["dispatch_blockers"]
    assert "latent_posthoc_allocator_demoted_low_leverage" in hinerv_plan["dispatch_blockers"]
    assert families["snerv"]["status"] == ("executable_cpu_advisory_plus_mlx_native_export_adapter_available")
    assert families["snerv"]["trainer_kind"] == (
        "mlx_native_target_hydration_receiver_export_available_scoreaware_long_training_missing"
    )
    assert families["snerv"]["next_action"].startswith(
        "bind_learned_mlx_scoreaware_decoder_training_to_snerv_native_export"
    )
    assert (
        "optional MLX-native target-hydration/export/receiver-proof attachment"
        in (families["snerv"]["execution_scope"])
    )
    assert families["snerv"]["archive_exporter"] == (
        "tac.substrates.snerv_inverse_steg_carrier.archive_candidate.export_snerv_archive_bound_candidate_package"
    )
    assert families["snerv"]["stack_role"] == "primary_carrier"
    assert families["snerv"]["score_aware_carrier_training_plan"]["score_aware_training_ready"] is False
    assert (
        "missing_training_stack:real_segnet_teacher"
        in families["snerv"]["score_aware_carrier_training_plan"]["dispatch_blockers"]
    )
    assert "sr_nerv_lowres_encode_superresolve_resolution_deadzone" in families["snerv"]["allowed_enhancers"]
    assert "ffnerv_flow_pose_channel" in families["snerv"]["allowed_enhancers"]
    assert families["sr_nerv"]["stack_role"] == ("resolution_axis_enhancer_or_design_knob")
    assert families["sr_nerv"]["enhancer_priority"] > families["boostnerv"]["carrier_priority"]
    assert families["rnerv"]["status"] == "migration_required"
    assert families["rnerv"]["stack_role"] == "enhancer_or_search_prior"
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_plan_only_report_routes_backend_rows_by_real_executability(
    tmp_path: Path,
) -> None:
    report = build_plan_only_report(
        output_dir=tmp_path / "plan",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    rows = {(row["family"], row["hard_byte_ceiling"]): row for row in report["compact_base_campaign_rows"]}
    assert rows[("pact_nerv_vq", 178_000)]["route_status"] == ("queued_for_mlx_training_archive_export_receiver_proof")
    assert rows[("pr95_hnerv", 178_000)]["route_status"] == ("queued_for_mlx_training_archive_export_receiver_proof")
    assert rows[("pvq_nerv", 178_000)]["canonical_family"] == "pact_nerv_vq"
    assert rows[("pvq_nerv", 178_000)]["route_status"] == ("queued_for_mlx_training_archive_export_receiver_proof")
    assert rows[("pact_nerv_selector_v4", 178_000)]["route_status"] == (
        "queued_for_mlx_training_archive_export_receiver_proof"
    )
    assert rows[("pact_nerv_selector_v4", 178_000)]["section_value_profiler"] == (
        "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
    )
    assert rows[("hi_nerv", 178_000)]["route_status"] == (
        "queued_for_mlx_archive_adapter_smoke_scoreaware_training_pending"
    )
    assert rows[("hi_nerv", 178_000)]["stack_role"] == "primary_carrier"
    assert "export_hi_nerv_mlx_archive" in rows[("hi_nerv", 178_000)]["archive_exporter"]
    modelsize_budget = rows[("hi_nerv", 178_000)]["modelsize_budget"]
    assert modelsize_budget["schema"] == "nerv_modelsize_budget.v1"
    assert modelsize_budget["hard_byte_ceilings"] == [178_000]
    assert modelsize_budget["selected_candidate_count"] > 0
    assert modelsize_budget["ready_for_exact_eval_dispatch"] is False
    snerv_budget = rows[("snerv", 178_000)]["modelsize_budget"]
    assert snerv_budget["schema"] == "snerv_modelsize_budget.v1"
    assert snerv_budget["hard_byte_ceilings"] == [178_000]
    assert snerv_budget["selected_candidate_count"] > 0
    assert snerv_budget["ready_for_exact_eval_dispatch"] is False
    campaign_plan = rows[("hi_nerv", 178_000)]["score_aware_carrier_training_plan"]
    assert campaign_plan["planner_action"] == ("run_receiver_closed_modelsize_ladder_before_score_aware_training")
    assert campaign_plan["linf_latent_posthoc_status"] == "demoted"
    assert campaign_plan["promotion_eligible"] is False
    assert rows[("snerv", 178_000)]["route_status"] == ("migration_required_before_runner_execution")
    assert rows[("snerv", 178_000)]["stack_role"] == "primary_carrier"
    assert rows[("sr_nerv", 178_000)]["route_status"] == ("migration_required_before_runner_execution")
    assert rows[("sr_nerv", 178_000)]["stack_role"] == ("resolution_axis_enhancer_or_design_knob")
    assert "scorer_mirror_check" in rows[("sr_nerv", 178_000)]["next_action"]
    assert rows[("boostnerv", 178_000)]["route_status"] == ("migration_required_before_runner_execution")
    assert rows[("boostnerv", 178_000)]["stack_role"] == "enhancer_bolt_on"
    assert rows[("rnerv", 178_000)]["trainer_entrypoint"] is None
    assert rows[("rnerv", 178_000)]["stack_role"] == "enhancer_or_search_prior"
    assert rows[("pact_nerv_vq", 178_000)]["score_claim"] is False
    assert rows[("pact_nerv_vq", 178_000)]["ready_for_exact_eval_dispatch"] is False
    assert rows[("pact_nerv_selector_v4", 178_000)]["score_claim"] is False
    assert rows[("pact_nerv_selector_v4", 178_000)]["ready_for_exact_eval_dispatch"] is False


def test_execute_modelsize_candidate_auto_uses_tightest_viable_byte_ceiling() -> None:
    hi_feedback = [
        {
            "family": "hi_nerv",
            "row_id": "hi_receiver_closed_calibration",
            "nominal_total_payload_bytes": 100_000,
            "measured_archive_bytes": 90_000,
            "receiver_proof_passed": True,
        }
    ]
    sn_feedback = [
        {
            "family": "snerv",
            "row_id": "sn_receiver_closed_calibration",
            "nominal_total_payload_bytes": 100_000,
            "measured_archive_bytes": 90_000,
            "receiver_proof_passed": True,
        }
    ]
    hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000, 285_000),
        byte_cap_feedback_rows=hi_feedback,
    )
    sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000, 285_000),
        byte_cap_feedback_rows=sn_feedback,
    )

    assert hi is not None
    assert hi["family"] == "hi_nerv"
    assert hi["hard_byte_ceiling"] == 178_000
    assert hi["byte_cap_controller"]["calibration_observation_count"] == 1
    assert hi["byte_cap_controller"]["predicted_under_hard_byte_ceiling"] is True
    assert runner_mod._byte_cap_controller_predicts_under_ceiling(hi)
    assert hi["use_hierarchical_feature_grid"] is True
    assert hi["use_convnext_blocks"] is True
    hi_precedence = hi["modelsize_control_contract"]["control_precedence"]
    assert hi_precedence["more_finely_grained_child_rules_take_priority"] is True
    assert hi_precedence["official_controls_are_base_constraints_not_rate_optimizer_overrides"] is True
    target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        hinerv_target_modelsize_mparams=(0.03,),
        byte_cap_feedback_rows=hi_feedback,
    )
    assert target_hi is not None
    assert target_hi["family"] == "hi_nerv"
    assert target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert target_hi["use_hierarchical_feature_grid"] is True
    assert target_hi["use_convnext_blocks"] is True
    assert target_hi["target_modelsize_mparams"] == 0.03
    target_hi_precedence = target_hi["modelsize_control_contract"]["control_precedence"]
    assert target_hi_precedence["highest_specificity_active_layer"] == ("pact_target_modelsize_child_rule")
    assert target_hi_precedence["more_finely_grained_child_rules_take_priority"] is True
    assert target_hi["modelsize_error_mparams"] == pytest.approx(abs(target_hi["modelsize_mparams"] - 0.03))
    assert target_hi["candidate_id"].endswith("_tgtmp0p03")
    reparsed_target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=target_hi["candidate_id"],
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        byte_cap_feedback_rows=hi_feedback,
    )
    assert reparsed_target_hi == target_hi
    assert target_hi["ready_for_exact_eval_dispatch"] is False
    assert sn is not None
    assert sn["family"] == "snerv"
    assert sn["hard_byte_ceiling"] == 285_000
    assert sn["nominal_under_ceiling"] is True
    official_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(216_000,),
        snerv_official_modelsize_mparams=(0.05,),
        byte_cap_feedback_rows=sn_feedback,
    )
    assert official_sn is not None
    assert official_sn["family"] == "snerv"
    assert official_sn["capacity_source"] == "official_snerv_modelsize"
    assert official_sn["modelsize_mparams"] == 0.05
    assert official_sn["official_modelsize_solution"]["fc_dim"] == (official_sn["fc_dim"])
    assert official_sn["ready_for_exact_eval_dispatch"] is False
    spectra_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(216_000,),
        snerv_official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
        byte_cap_feedback_rows=sn_feedback,
    )
    assert spectra_sn is not None
    assert spectra_sn["capacity_source"] == "official_snerv_modelsize"
    assert spectra_sn["snerv_model_size_adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert "_adspectra_oms0p05_" in spectra_sn["candidate_id"]
    reparsed_spectra_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=spectra_sn["candidate_id"],
        hard_byte_ceilings=(178_000,),
    )
    assert reparsed_spectra_sn is not None
    assert reparsed_spectra_sn["snerv_model_size_adapter"] == (SNERV_SPECTRA_PRESERVING_ADAPTER)
    assert reparsed_spectra_sn["candidate_id"] == spectra_sn["candidate_id"]
    official_primitives_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(2_000_000_000,),
        snerv_official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
        snerv_official_skip_high_modes=("full",),
        byte_cap_feedback_rows=sn_feedback,
    )
    assert official_primitives_sn is not None
    assert official_primitives_sn["capacity_source"] == "official_snerv_modelsize"
    assert official_primitives_sn["levels"] == 1
    assert official_primitives_sn["snerv_model_size_adapter"] == (SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER)
    assert "_haar_lv1_" in official_primitives_sn["candidate_id"]
    assert "_adofficial_oms0p05_" in official_primitives_sn["candidate_id"]
    assert runner_mod._snerv_auto_skip_high_modes_for_resolution(
        explicit_mode=None,
        model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
    ) == ("scalar_mean", "channel_mean", "shared_mean")
    official_primitives_auto_skip_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        snerv_official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
        snerv_official_skip_high_modes=runner_mod._snerv_auto_skip_high_modes_for_resolution(
            explicit_mode=None,
            model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
        ),
        byte_cap_feedback_rows=sn_feedback,
    )
    assert official_primitives_auto_skip_sn is not None
    assert official_primitives_auto_skip_sn["official_skip_high_mode"] in {
        "channel_mean",
        "scalar_mean",
    }
    assert official_primitives_auto_skip_sn["nominal_under_ceiling"] is True
    assert (
        official_primitives_auto_skip_sn["nominal_total_payload_bytes"]
        < (official_primitives_sn["nominal_total_payload_bytes"])
    )
    assert runner_mod._snerv_auto_skip_high_modes_for_resolution(
        explicit_mode="shared_mean",
        model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
    ) == ("shared_mean",)
    reparsed_official_primitives_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=official_primitives_sn["candidate_id"],
        hard_byte_ceilings=(178_000,),
    )
    assert reparsed_official_primitives_sn is not None
    assert reparsed_official_primitives_sn["snerv_model_size_adapter"] == (
        SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    )
    assert reparsed_official_primitives_sn["candidate_id"] == (official_primitives_sn["candidate_id"])
    assert reparsed_official_primitives_sn["official_skip_high_mode_token_missing"] is True
    target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(36_000,),
        hinerv_target_modelsize_mparams=(0.02,),
        byte_cap_feedback_rows=hi_feedback,
    )
    assert target_hi is not None
    assert target_hi["family"] == "hi_nerv"
    assert target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert target_hi["use_hierarchical_feature_grid"] is True
    assert target_hi["use_convnext_blocks"] is True
    assert target_hi["target_modelsize_mparams"] == 0.02
    assert target_hi["modelsize_error_mparams"] is not None
    assert "_tgtmp0p02" in target_hi["candidate_id"]
    assert target_hi["ready_for_exact_eval_dispatch"] is False
    shared_target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        target_modelsize_mparams=(0.03,),
        byte_cap_feedback_rows=hi_feedback,
    )
    assert shared_target_hi is not None
    assert shared_target_hi["family"] == "hi_nerv"
    assert shared_target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert shared_target_hi["use_hierarchical_feature_grid"] is True
    assert shared_target_hi["use_convnext_blocks"] is True
    assert shared_target_hi["target_modelsize_mparams"] == 0.03
    assert shared_target_hi["modelsize_error_mparams"] == pytest.approx(
        abs(shared_target_hi["modelsize_mparams"] - 0.03)
    )
    assert shared_target_hi["candidate_id"].endswith("_tgtmp0p03")
    shared_hi_contract = shared_target_hi["modelsize_control_contract"]
    assert shared_hi_contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert shared_hi_contract["control_semantics"] == ("local_receiver_visible_grid_search_nearest_target")
    assert shared_hi_contract["shared_target_modelsize_mparams_consumed_as"] == ("nearest_local_param_count_target")
    assert shared_hi_contract["modelsize_mparams_is_official_upstream_flag"] is False
    assert shared_hi_contract["archive_bytes_authority_required"] is True
    shared_target_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(216_000,),
        target_modelsize_mparams=(0.05,),
        byte_cap_feedback_rows=sn_feedback,
    )
    assert shared_target_sn is not None
    assert shared_target_sn["family"] == "snerv"
    assert shared_target_sn["capacity_source"] == "official_snerv_modelsize"
    assert shared_target_sn["modelsize_mparams"] == 0.05
    assert shared_target_sn["official_modelsize_solution"]["fc_dim"] == (shared_target_sn["fc_dim"])
    assert shared_target_sn["ready_for_exact_eval_dispatch"] is False
    shared_sn_contract = shared_target_sn["modelsize_control_contract"]
    assert shared_sn_contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert shared_sn_contract["control_semantics"] == ("official_snerv_modelsize_quadratic_fc_dim_solve")
    assert shared_sn_contract["shared_target_modelsize_mparams_consumed_as"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert shared_sn_contract["modelsize_mparams_is_official_upstream_flag"] is True
    assert shared_sn_contract["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert shared_sn_contract["mutates_receiver_visible_fc_dim"] is True
    assert shared_sn_contract["archive_bytes_authority_required"] is True
    explicit_shared_target_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=shared_target_sn["candidate_id"],
        hard_byte_ceilings=(178_000,),
    )
    assert explicit_shared_target_sn is not None
    assert explicit_shared_target_sn["candidate_id"] == shared_target_sn["candidate_id"]
    assert explicit_shared_target_sn["capacity_source"] == "official_snerv_modelsize"
    assert explicit_shared_target_sn["modelsize_mparams"] == 0.05
    assert explicit_shared_target_sn["official_modelsize_solution"]["fc_dim"] == (explicit_shared_target_sn["fc_dim"])
    explicit = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=hi["candidate_id"],
        hard_byte_ceilings=(178_000, 285_000),
        byte_cap_feedback_rows=hi_feedback,
    )
    assert explicit == hi
    assert (
        _resolve_execute_modelsize_candidate(
            family="hi_nerv",
            candidate_id="manual",
            hard_byte_ceilings=(178_000,),
        )
        is None
    )
    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id="missing-candidate",
            hard_byte_ceilings=(178_000,),
        )


def test_snerv_modelsize_resolution_uses_effective_spectra_adapter() -> None:
    args = SimpleNamespace(
        snerv_spectra_preserving_adapter=True,
        snerv_model_size_adapter=None,
    )
    assert runner_mod._effective_snerv_modelsize_adapter_for_resolution(args) == (SNERV_SPECTRA_PRESERVING_ADAPTER)
    candidate = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(216_000,),
        snerv_official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter=runner_mod._effective_snerv_modelsize_adapter_for_resolution(args),
    )
    assert candidate is not None
    assert candidate["snerv_model_size_adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER

    conflict_args = SimpleNamespace(
        snerv_spectra_preserving_adapter=True,
        snerv_model_size_adapter="snerv_fc_dim_emb_size_adapter_v1",
    )
    with pytest.raises(SystemExit):
        runner_mod._effective_snerv_modelsize_adapter_for_resolution(conflict_args)


def test_snerv_execution_fc_dim_resolver_consumes_official_solution() -> None:
    fc_dim = runner_mod._resolve_snerv_execution_fc_dim(
        {
            "candidate_id": "snerv_solution",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "official_modelsize_solution": {"fc_dim": 11},
        },
        cli_override=None,
        fallback=9,
    )

    assert fc_dim == 11
    assert runner_mod._resolve_snerv_execution_fc_dim_with_source(
        {
            "candidate_id": "snerv_solution",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "official_modelsize_solution": {"fc_dim": 11},
        },
        cli_override=17,
        fallback=9,
    ) == (11, "official_modelsize_solution")


def test_snerv_execution_fc_dim_resolver_recomputes_official_formula() -> None:
    fc_dim = runner_mod._resolve_snerv_execution_fc_dim(
        {
            "candidate_id": "snerv_formula",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "num_pairs": 600,
            "carrier_hw": [384, 512],
            "enc_strds": [5, 4, 2, 2, 2],
            "dec_strds": [5, 4, 2, 2, 2],
        },
        cli_override=None,
        fallback=9,
    )

    assert fc_dim == 11
    assert runner_mod._resolve_snerv_execution_fc_dim_with_source(
        {
            "candidate_id": "snerv_formula",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "num_pairs": 600,
            "carrier_hw": [384, 512],
            "enc_strds": [5, 4, 2, 2, 2],
            "dec_strds": [5, 4, 2, 2, 2],
        },
        cli_override=17,
        fallback=9,
    ) == (11, "official_modelsize_formula")


def test_snerv_execution_fc_dim_resolver_rejects_fake_modelsize_fallback() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="modelsize_mparams requires official_modelsize_solution",
    ):
        runner_mod._resolve_snerv_execution_fc_dim(
            {
                "candidate_id": "snerv_missing_formula_controls",
                "modelsize_mparams": 0.05,
            },
            cli_override=7,
            fallback=9,
        )


def test_snerv_execution_fc_dim_resolver_keeps_manual_fallback_without_modelsize() -> None:
    fc_dim = runner_mod._resolve_snerv_execution_fc_dim(
        {"candidate_id": "manual_no_modelsize"},
        cli_override=7,
        fallback=9,
    )

    assert fc_dim == 7
    assert runner_mod._resolve_snerv_execution_fc_dim_with_source(
        {"candidate_id": "manual_no_modelsize"},
        cli_override=7,
        fallback=9,
    ) == (7, "manual_cli_override")
    assert runner_mod._resolve_snerv_execution_fc_dim_with_source(
        {"candidate_id": "manual_no_modelsize"},
        cli_override=None,
        fallback=9,
    ) == (9, "fallback_default_missing_official_modelsize_inputs")


def test_byte_cap_controller_uses_measured_archive_feedback() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "small",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 60,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            },
            {
                "candidate_id": "large",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 80,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            },
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "measured_export",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 60,
                "measured_archive_bytes": 90,
                "decoder_codec": "int4_scale_bundled",
                "receiver_proof_passed": True,
            }
        ],
    )

    by_id = {row["candidate_id"]: row for row in attached}
    assert by_id["small"]["byte_cap_controller"]["predicted_archive_bytes"] == 90
    assert by_id["small"]["byte_cap_controller"]["predicted_under_hard_byte_ceiling"] is True
    assert by_id["large"]["byte_cap_controller"]["predicted_archive_bytes"] == 120
    assert by_id["large"]["byte_cap_controller"]["predicted_under_hard_byte_ceiling"] is False
    assert runner_mod._byte_cap_controller_predicts_under_ceiling(by_id["small"])
    assert not runner_mod._byte_cap_controller_predicts_under_ceiling(by_id["large"])


def test_byte_cap_controller_exploits_measured_compression_headroom() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "larger",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 140,
                "decoder_codec": "portfolio_auto",
                "nominal_under_ceiling": False,
            }
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "compressed_export",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 100,
                "measured_archive_bytes": 50,
                "decoder_codec": "portfolio_auto",
                "receiver_proof_ready": True,
            }
        ],
    )

    controller = attached[0]["byte_cap_controller"]
    assert controller["prediction_rule"] == (
        "max_observed_archive_to_nominal_ratio_or_additive_overhead_plus_loo_residual_guard"
    )
    assert controller["calibration_residual_guard_bytes"] == 0
    assert controller["predicted_archive_bytes"] == 90
    assert controller["predicted_under_hard_byte_ceiling"] is True


def test_byte_cap_controller_uses_leave_one_out_residual_guard() -> None:
    feedback_rows = [
        {
            "row_id": "early_a",
            "family": "hi_nerv",
            "nominal_total_payload_bytes": 177_554,
            "measured_archive_bytes": 214_187,
            "decoder_codec": "int7_mixed",
            "receiver_proof_ready": True,
        },
        {
            "row_id": "early_b",
            "family": "hi_nerv",
            "nominal_total_payload_bytes": 177_554,
            "measured_archive_bytes": 215_512,
            "decoder_codec": "int7_mixed",
            "receiver_proof_ready": True,
        },
        {
            "row_id": "final_overcap",
            "family": "hi_nerv",
            "nominal_total_payload_bytes": 138_998,
            "measured_archive_bytes": 178_479,
            "decoder_codec": "int7_mixed",
            "receiver_proof_ready": True,
        },
    ]

    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "too_tight",
                "family": "hi_nerv",
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 137_188,
                "decoder_codec": "int7_mixed",
                "nominal_under_ceiling": True,
            },
            {
                "candidate_id": "guarded",
                "family": "hi_nerv",
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 136_900,
                "decoder_codec": "int7_mixed",
                "nominal_under_ceiling": True,
            },
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=feedback_rows,
    )

    by_id = {row["candidate_id"]: row for row in attached}
    tight = by_id["too_tight"]["byte_cap_controller"]
    guarded = by_id["guarded"]["byte_cap_controller"]
    assert tight["calibration_residual_guard_bytes"] == 1_523
    assert tight["predicted_archive_bytes"] == 178_192
    assert tight["predicted_under_hard_byte_ceiling"] is False
    assert guarded["calibration_residual_guard_bytes"] == 1_523
    assert guarded["predicted_archive_bytes"] == 177_904
    assert guarded["predicted_under_hard_byte_ceiling"] is True


def test_byte_cap_controller_ignores_unproven_feedback_rows() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "candidate",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 80,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            }
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "optimistic_unproven",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 100,
                "measured_archive_bytes": 50,
                "decoder_codec": "int4_scale_bundled",
                "receiver_proof_passed": False,
            }
        ],
    )

    controller = attached[0]["byte_cap_controller"]
    assert controller["predicted_archive_bytes"] == 80
    assert controller["predicted_under_hard_byte_ceiling"] is None
    assert controller["prediction_rule"] == "nominal_payload_bytes_uncalibrated"
    assert controller["prediction_authority"] == "uncalibrated_nominal_payload_diagnostic"
    assert controller["calibrated_archive_prediction"] is False
    assert controller["calibration_observation_count"] == 0
    assert "byte_cap_controller_measured_archive_feedback_missing" in controller["blockers"]
    assert not runner_mod._byte_cap_controller_predicts_under_ceiling(attached[0])


def test_execute_modelsize_auto_uses_byte_cap_feedback_to_avoid_overcap() -> None:
    near_cap = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
    )
    assert near_cap is not None
    assert near_cap["modelsize_auto_selection_requires_measured_archive_feedback"] is True
    assert near_cap["modelsize_auto_selection_warning"] == (
        "hi_nerv_uncalibrated_near_cap_candidate_requires_archive_measurement"
    )
    assert near_cap["modelsize_auto_selection_predicted_archive_bytes"] == 178_223
    assert near_cap["modelsize_auto_selection_hard_byte_ceiling"] == 178_000
    assert near_cap["score_claim"] is False
    assert near_cap["ready_for_exact_eval_dispatch"] is False
    launch_args = runner_mod._parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--planner-row-id",
            "unit_hi_near_cap_auto",
            "--allow-manual-compact-family-launch",
            "--num-pairs",
            "600",
            "--epochs",
            "8",
        ]
    )
    assert (
        runner_mod._uncalibrated_near_cap_modelsize_launch_blocker(
            args=launch_args,
            modelsize_candidate=near_cap,
        )
        == "hi_nerv_modelsize_auto_near_cap_requires_measured_archive_feedback_before_long_training"
    )
    probe_args = runner_mod._parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--planner-row-id",
            "unit_hi_near_cap_probe",
            "--allow-manual-compact-family-launch",
            "--allow-unscored-research-smoke",
            "--allow-bounded-planner-row-timing-smoke-waiver",
            "--num-pairs",
            "16",
            "--epochs",
            "1",
        ]
    )
    assert (
        runner_mod._uncalibrated_near_cap_modelsize_launch_blocker(
            args=probe_args,
            modelsize_candidate=near_cap,
        )
        is None
    )

    nominal = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        hinerv_target_modelsize_mparams=(0.12,),
        byte_cap_feedback_rows=[
            {
                "family": "hi_nerv",
                "row_id": "target_export",
                "decoder_codec": "portfolio_auto",
                "nominal_total_payload_bytes": 100_000,
                "measured_archive_bytes": 120_000,
                "receiver_proof_passed": True,
            }
        ],
    )
    assert nominal is not None
    assert nominal["byte_cap_controller"]["calibration_observation_count"] == 1

    calibrated = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        byte_cap_feedback_rows=[
            {
                "family": "hi_nerv",
                "row_id": "prior_export",
                "decoder_codec": nominal["decoder_codec"],
                "nominal_total_payload_bytes": nominal["nominal_total_payload_bytes"],
                "measured_archive_bytes": nominal["nominal_total_payload_bytes"] + 20_000,
                "receiver_proof_passed": True,
            }
        ],
    )

    assert calibrated is not None
    assert calibrated["hard_byte_ceiling"] == 178_000
    controller = calibrated["byte_cap_controller"]
    assert controller["calibration_observation_count"] == 1
    assert controller["predicted_under_hard_byte_ceiling"] is True
    assert controller["predicted_archive_bytes"] <= 178_000
    assert runner_mod._byte_cap_controller_predicts_under_ceiling(calibrated)


def test_execute_modelsize_auto_fails_closed_when_calibrated_feedback_has_no_under_cap_candidate() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="hi_nerv_modelsize_auto_no_calibrated_candidate_under_hard_byte_ceiling",
    ):
        _resolve_execute_modelsize_candidate(
            family="hi_nerv",
            candidate_id="auto",
            hard_byte_ceilings=(178_000,),
            byte_cap_feedback_rows=[
                {
                    "family": "hi_nerv",
                    "row_id": "oversized_export",
                    "decoder_codec": "portfolio_auto",
                    "nominal_total_payload_bytes": 1,
                    "measured_archive_bytes": 1_000_000,
                    "receiver_proof_passed": True,
                }
            ],
        )


def test_execute_snerv_modelsize_auto_fails_closed_when_no_nominal_candidate_under_cap() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="snerv_modelsize_auto_no_candidate_under_hard_byte_ceiling",
    ):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id="auto",
            hard_byte_ceilings=(1_000,),
            snerv_official_modelsize_mparams=(0.05,),
            snerv_model_size_adapter=("snerv_official_mfu_hfr_tub_primitives_adapter"),
            snerv_temporal_modes=("official_haar_dwt1d_lowpass",),
        )


def test_explicit_snerv_modelsize_candidate_rejects_calibrated_over_cap_feedback() -> None:
    rows = [
        row.as_dict()
        for row in enumerate_snerv_modelsize_candidates(
            hard_byte_ceilings=(216_000,),
            num_pairs=600,
            official_modelsize_mparams=(0.05,),
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            temporal_modes=("official_haar_dwt1d_lowpass",),
            official_skip_high_modes=("scalar_mean",),
        )
    ]
    candidate = next(row for row in rows if row["official_modelsize_solution"])
    feedback = {
        "family": "snerv",
        "candidate_id": candidate["candidate_id"],
        "decoder_payload_codec": candidate["decoder_payload_codec"],
        "nominal_total_payload_bytes": candidate["nominal_total_payload_bytes"],
        "measured_archive_bytes": 300_000,
        "receiver_proof_passed": True,
    }

    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="snerv_modelsize_explicit_candidate_over_hard_byte_ceiling",
    ):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id=candidate["candidate_id"],
            hard_byte_ceilings=(216_000,),
            snerv_official_modelsize_mparams=(0.05,),
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            snerv_temporal_modes=("official_haar_dwt1d_lowpass",),
            snerv_official_skip_high_modes=("scalar_mean",),
            byte_cap_feedback_rows=[feedback],
        )


def test_explicit_snerv_modelsize_candidate_accepts_calibrated_under_cap_feedback() -> None:
    rows = [
        row.as_dict()
        for row in enumerate_snerv_modelsize_candidates(
            hard_byte_ceilings=(216_000,),
            num_pairs=600,
            official_modelsize_mparams=(0.05,),
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            temporal_modes=("official_haar_dwt1d_lowpass",),
            official_skip_high_modes=("scalar_mean",),
        )
    ]
    candidate = next(row for row in rows if row["official_modelsize_solution"])
    selected = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=candidate["candidate_id"],
        hard_byte_ceilings=(216_000,),
        snerv_official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        snerv_temporal_modes=("official_haar_dwt1d_lowpass",),
        snerv_official_skip_high_modes=("scalar_mean",),
        byte_cap_feedback_rows=[
            {
                "family": "snerv",
                "candidate_id": candidate["candidate_id"],
                "decoder_payload_codec": candidate["decoder_payload_codec"],
                "nominal_total_payload_bytes": candidate["nominal_total_payload_bytes"],
                "measured_archive_bytes": 200_000,
                "receiver_proof_passed": True,
            }
        ],
    )

    assert selected is not None
    assert selected["candidate_id"] == candidate["candidate_id"]
    assert selected["byte_cap_controller"]["predicted_archive_bytes"] == 200_000
    assert selected["byte_cap_controller"]["predicted_under_hard_byte_ceiling"] is True


def test_snerv_modelsize_candidates_include_official_skip_high_mode_in_identity() -> None:
    rows = enumerate_snerv_modelsize_candidates(
        hard_byte_ceilings=(36_000,),
        num_pairs=600,
        levels=(2,),
        bits_per_coeffs=(1.5,),
        step_map_bits_per_coeffs=(0.5,),
        decoder_codecs=("int2_symmetric",),
        official_modelsize_mparams=(0.05,),
        snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        temporal_modes=("official_haar_dwt1d_lowpass",),
        official_skip_high_modes=(
            "full",
            "shared_mean",
            "channel_mean",
            "scalar_mean",
        ),
    )

    keyed = {row.official_skip_high_mode: row.as_dict() for row in rows}
    assert set(keyed) == {"full", "shared_mean", "channel_mean", "scalar_mean"}
    assert {row["bits_per_coeff"] for row in keyed.values()} == {1.0}
    assert {row["step_map_bits_per_coeff"] for row in keyed.values()} == {1.0}
    assert "_sksharedmean_" not in keyed["full"]["candidate_id"]
    assert "_sksharedmean_" in keyed["shared_mean"]["candidate_id"]
    assert "_skchannelmean_" in keyed["channel_mean"]["candidate_id"]
    assert "_skscalarmean_" in keyed["scalar_mean"]["candidate_id"]
    assert keyed["shared_mean"]["snerv_official_skip_high_mode"] == "shared_mean"
    assert keyed["channel_mean"]["snerv_official_skip_high_mode"] == "channel_mean"
    assert keyed["scalar_mean"]["snerv_official_skip_high_mode"] == "scalar_mean"
    assert keyed["shared_mean"]["candidate_id"] != keyed["full"]["candidate_id"]


def test_modelsize_byte_cap_feedback_loader_accepts_checkpoint_exports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv-row",
                "archive_bytes": 123_456,
                "decoder_codec": "int4_mixed",
                "receiver_proof_ready": True,
                "modelsize_candidate": {
                    "candidate_id": "hinerv-row",
                    "family": "hi_nerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 111_111,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([path])

    assert rows == [
        {
            "family": "hi_nerv",
            "row_id": "hinerv-row",
            "candidate_id": "hinerv-row",
            "measured_archive_bytes": 123_456,
            "nominal_total_payload_bytes": 111_111,
            "hard_byte_ceiling": 178_000,
            "decoder_codec": "int4_mixed",
            "source_bound_controls": {
                "family": "hi_nerv",
                "hard_byte_ceiling": 178_000,
            },
            "receiver_closed": True,
            "receiver_closed_status": "inline_receiver_closed",
            "source_path": path.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_preserves_nested_required_nominal_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hinerv_export.json"
    candidate = {
        "candidate_id": "hinerv-row",
        "family": "hi_nerv",
        "num_pairs": 600,
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 138_998,
        "decoder_codec": "int7_mixed",
    }
    path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv-row",
                "archive_bytes": 178_479,
                "decoder_codec": "int7_mixed",
                "receiver_proof_ready": True,
                "modelsize_candidate": candidate,
                "modelsize_byte_cap_feedback_row": {
                    "schema": "nerv_modelsize_byte_cap_feedback_row.v1",
                    "family": "hi_nerv",
                    "candidate_id": "hinerv-row",
                    "measured_archive_bytes": 178_479,
                    "nominal_total_payload_bytes": 138_998,
                    "hard_byte_ceiling": 178_000,
                    "decoder_codec": "int7_mixed",
                    "modelsize_candidate": candidate,
                    "calibrated_archive_overrun_bytes": 479,
                    "required_nominal_payload_bytes_max": 138_624,
                    "hard_byte_ceiling_measurement_bypass_enabled": True,
                    "receiver_closed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([path])

    assert rows == [
        {
            "family": "hi_nerv",
            "row_id": "hinerv-row",
            "candidate_id": "hinerv-row",
            "measured_archive_bytes": 178_479,
            "nominal_total_payload_bytes": 138_998,
            "hard_byte_ceiling": 178_000,
            "decoder_codec": "int7_mixed",
            "calibrated_archive_overrun_bytes": 479,
            "required_nominal_payload_bytes_max": 138_624,
            "hard_byte_ceiling_measurement_bypass_enabled": True,
            "source_bound_controls": {
                "family": "hi_nerv",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_codec": "int7_mixed",
            },
            "receiver_closed": True,
            "receiver_closed_status": "inline_receiver_closed",
            "source_path": path.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_prefers_nested_candidate_nominal_over_packet_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snerv_export.json"
    state_slice = tmp_path / "official_trained_checkpoint_state_dict_slice.npz"
    state_slice.write_bytes(b"npz state slice")
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "archive_bytes": 444_036,
                "packet_bytes": 2_347_396,
                "decoder_codec": "int8_symmetric",
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "hard_byte_ceiling_measurement_bypass_enabled": True,
                "hard_byte_ceiling_checked_after_export": True,
                "calibrated_archive_overrun_bytes": 228_036,
                "required_nominal_payload_bytes_max": 91_875,
                "official_checkpoint_export_binding": {
                    "schema": "snerv_official_checkpoint_export_binding.v1",
                    "official_trained_checkpoint_state_dict_slice_present": True,
                    "official_trained_checkpoint_state_dict_slice_path": (state_slice.name),
                    "official_trained_checkpoint_state_dict_slice_bytes": (state_slice.stat().st_size),
                    "official_trained_checkpoint_state_dict_slice_sha256": "a" * 64,
                    "official_trained_checkpoint_state_dict_slice_member_count": 2,
                    "official_trained_checkpoint_state_dict_slice_member_names": [
                        "decoder.layers.0.weight",
                        "encoder.layers.0.weight",
                    ],
                    "official_trained_checkpoint_state_dict_slice_runner_arg": (
                        "--snerv-official-trained-checkpoint-state-dict-path"
                    ),
                },
                "modelsize_candidate": {
                    "candidate_id": "snerv-row",
                    "family": "snerv",
                    "hard_byte_ceiling": 216_000,
                    "nominal_total_payload_bytes": 188_854,
                    "decoder_payload_codec": "int8_symmetric",
                },
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([path])

    assert rows == [
        {
            "family": "snerv",
            "candidate_id": "snerv-row",
            "measured_archive_bytes": 444_036,
            "measured_payload_bytes": 2_347_396,
            "nominal_total_payload_bytes": 188_854,
            "hard_byte_ceiling": 216_000,
            "decoder_codec": "int8_symmetric",
            "calibrated_archive_overrun_bytes": 228_036,
            "required_nominal_payload_bytes_max": 91_875,
            "hard_byte_ceiling_measurement_bypass_enabled": True,
            "hard_byte_ceiling_checked_after_export": True,
            "snerv_official_trained_checkpoint_state_dict_slice_path": (state_slice.resolve(strict=False).as_posix()),
            "snerv_official_trained_checkpoint_state_dict_path": (state_slice.resolve(strict=False).as_posix()),
            "snerv_official_trained_checkpoint_state_dict_slice_present": True,
            "snerv_official_trained_checkpoint_state_dict_slice_file_present": True,
            "snerv_official_trained_checkpoint_state_dict_slice_runner_arg": (
                "--snerv-official-trained-checkpoint-state-dict-path"
            ),
            "snerv_official_trained_checkpoint_state_dict_slice_bytes": (state_slice.stat().st_size),
            "snerv_official_trained_checkpoint_state_dict_slice_sha256": "a" * 64,
            "snerv_official_trained_checkpoint_state_dict_slice_member_count": 2,
            "snerv_official_trained_checkpoint_state_dict_slice_member_names": [
                "decoder.layers.0.weight",
                "encoder.layers.0.weight",
            ],
            "source_bound_controls": {
                "family": "snerv",
                "hard_byte_ceiling": 216_000,
                "decoder_payload_codec": "int8_symmetric",
            },
            "receiver_closed": True,
            "receiver_closed_status": "inline_receiver_closed",
            "receiver_contract_satisfied": True,
            "receiver_proof_passed": True,
            "source_path": path.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_rejects_contract_only_checkpoint_export(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snerv_contract_only_export.json"
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "archive_bytes": 91_445,
                "packet_bytes": 188_000,
                "receiver_contract_satisfied": True,
                "modelsize_candidate": {
                    "candidate_id": "snerv-contract-only",
                    "family": "snerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 120_000,
                },
            }
        ),
        encoding="utf-8",
    )

    assert runner_mod._load_modelsize_byte_cap_feedback_rows([path]) == []


def test_modelsize_byte_cap_feedback_loader_reads_startup_candidate_fallback(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "startup.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "modelsize_candidate": {
                    "candidate_id": "hinerv-fallback",
                    "family": "hi_nerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 117_000,
                    "decoder_codec": "int7_mixed",
                },
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "hinerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv-fallback",
                "archive_bytes": 122_074,
                "receiver_proof_ready": True,
                "startup_json_path": startup.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([export])

    assert rows[0]["candidate_id"] == "hinerv-fallback"
    assert rows[0]["measured_archive_bytes"] == 122_074
    assert rows[0]["nominal_total_payload_bytes"] == 117_000
    assert rows[0]["hard_byte_ceiling"] == 178_000
    assert rows[0]["decoder_codec"] == "int7_mixed"
    assert rows[0]["receiver_closed"] is True


def test_modelsize_byte_cap_feedback_loader_accepts_snerv_binary_profile_with_receiver_proof(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="c" * 64,
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([profile])
    archive_path = (
        profile.parent.parent
        / "snerv_run"
        / "snerv_mlx_native_export"
        / "native_train_export"
        / "snerv_mlx_native_archive_bound_package"
        / "archive.zip"
    ).resolve(strict=False)
    packet_path = (
        profile.parent.parent
        / "snerv_run"
        / "snerv_mlx_native_export"
        / "native_train_export"
        / "snerv_mlx_native_packet.snar"
    ).resolve(strict=False)

    assert rows == [
        {
            "family": "snerv",
            "candidate_id": scalar["candidate_id"],
            "measured_archive_bytes": 91_445,
            "measured_payload_bytes": packet_path.stat().st_size,
            "nominal_total_payload_bytes": scalar["nominal_total_payload_bytes"],
            "hard_byte_ceiling": 178_000,
            "decoder_codec": "int8_symmetric",
            "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "archive_path": archive_path.as_posix(),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "source_bound_controls": runner_mod._byte_cap_candidate_match_controls(scalar),
            "receiver_closed": True,
            "receiver_closed_status": "associated_receiver_proof",
            "receiver_proof_path": (
                profile.parent.parent
                / "snerv_run"
                / "snerv_mlx_native_export"
                / "native_train_export"
                / "snerv_mlx_native_archive_bound_package"
                / "receiver_proof"
                / "snerv_inverse_steg_receiver_proof.json"
            )
            .resolve(strict=False)
            .as_posix(),
            "source_path": profile.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_rejects_partial_snerv_binary_profile(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="2" * 64,
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["snar1_metadata"]["n_pairs"] = 2
    profile.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    assert runner_mod._load_modelsize_byte_cap_feedback_rows([profile]) == []


def test_byte_cap_controller_keeps_snerv_skip_mode_feedback_candidate_scoped(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    full = _snerv_official_skip_candidate("full")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="d" * 64,
    )
    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([profile])

    attached = runner_mod._attach_byte_cap_controller_predictions(
        [scalar, full],
        family="snerv",
        byte_cap_feedback_rows=rows,
    )

    by_id = {row["candidate_id"]: row for row in attached}
    scalar_controller = by_id[scalar["candidate_id"]]["byte_cap_controller"]
    full_controller = by_id[full["candidate_id"]]["byte_cap_controller"]
    assert scalar_controller["calibration_observation_count"] == 1
    assert scalar_controller["predicted_archive_bytes"] == 91_445
    assert full_controller["calibration_observation_count"] == 0
    assert "byte_cap_controller_measured_archive_feedback_missing" in full_controller["blockers"]


def test_snerv_official_modelsize_controls_require_candidate_resolution() -> None:
    manual = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--modelsize-candidate-id",
            "manual",
            "--snerv-fc-dim",
            "13",
        ]
    )
    blocked = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--modelsize-candidate-id",
            "manual",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-official-enc-strds",
            "1,2,2",
        ]
    )

    assert runner_mod._snerv_official_modelsize_candidate_resolution_blockers(manual) == []
    blockers = runner_mod._snerv_official_modelsize_candidate_resolution_blockers(blocked)
    assert (
        "snerv_official_modelsize_control_requires_candidate_resolution:--snerv-official-modelsize-mparams"
    ) in blockers
    assert ("snerv_official_modelsize_control_requires_candidate_resolution:--snerv-official-enc-strds") in blockers
    with pytest.raises(SystemExit, match="SNeRV official modelsize controls require"):
        runner_mod.main(
            [
                "--execute-family",
                "snerv",
                "--modelsize-candidate-id",
                "manual",
                "--snerv-official-modelsize-mparams",
                "0.05",
            ]
        )


def test_hinerv_parser_exposes_receiver_bound_latent_codecs() -> None:
    for codec in (
        "auto",
        "int16_hi_ac_brotli_q11",
        "int8_brotli_q11",
        "int4_packed_brotli_q11",
        "int2_packed_brotli_q11",
    ):
        args = _parse_args(
            [
                "--execute-family",
                "hi_nerv",
                "--hi-nerv-latent-codec",
                codec,
            ]
        )

        assert args.execute_family == "hi_nerv"
        assert args.hi_nerv_latent_codec == codec


def test_hinerv_auto_latent_codec_prices_latents_before_decoder_precision() -> None:
    assert (
        runner_mod._resolve_hi_nerv_latent_codec_policy(
            requested="auto",
            hard_byte_ceiling=178_493,
        )
        == "int8_brotli_q11"
    )
    assert (
        runner_mod._resolve_hi_nerv_latent_codec_policy(
            requested="auto",
            hard_byte_ceiling=216_000,
        )
        == "int16_hi_ac_brotli_q11"
    )
    assert (
        runner_mod._resolve_hi_nerv_latent_codec_policy(
            requested="int16_raw",
            hard_byte_ceiling=178_493,
        )
        == "int16_raw"
    )


def test_hinerv_execute_resolves_auto_latent_codec_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_auto_latent_policy",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_493,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-auto-latent-policy",
            "latent_dim": 4,
            "embed_dim": 4,
            "decoder_channel": 4,
            "decoder_codec": "portfolio_auto",
            "hi_nerv_latent_codec": "auto",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_493,
            "nominal_total_payload_bytes": 170_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        post_export_receiver_cache_quality_gate=False,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["hi_nerv_latent_codec"] == "int8_brotli_q11"
    selection = out["modelsize_candidate_selection"]
    assert selection["requested_hi_nerv_latent_codec"] == "auto"
    assert selection["launch_hi_nerv_latent_codec"] == "int8_brotli_q11"
    score_training = out["score_aware_training"]
    assert score_training["requested_hi_nerv_latent_codec"] == "auto"
    assert score_training["hi_nerv_latent_codec"] == "int8_brotli_q11"
    assert score_training["hi_nerv_latent_codec_policy"] == {
        "schema": "compact_hi_nerv_latent_codec_policy.v1",
        "requested": "auto",
        "resolved": "int8_brotli_q11",
        "hard_byte_ceiling": 178_493,
        "policy": "protect_decoder_precision_under_178493_by_pricing_latents",
    }


def test_execute_modelsize_candidate_resolves_self_describing_queue_ids() -> None:
    hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000",
        hard_byte_ceilings=(178_000,),
    )
    hi_official = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=("hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_int4_mixed_ceil36000"),
        hard_byte_ceilings=(178_000,),
    )
    hi_target = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=("hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_int4_mixed_ceil36000_tgtmp0p02"),
        hard_byte_ceilings=(178_000,),
    )
    sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000",
        hard_byte_ceilings=(178_000,),
    )
    sn_rich = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p1_mfu1-3_hfr0p25_t1_adbase_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_spectra = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_mfu1-5_hfr0p375_t2_adspectra_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_temporal = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
            "mfu1-5_hfr0p375_t2_tmhaar1_adspectra_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_shared_mean = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
            "mfu1-5_hfr0p375_t2_adspectra_sksharedmean_"
            "int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_official_requested_bits = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_"
            "mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_"
            "skchannelmean_int8_symmetric_ceil178000"
        ),
        hard_byte_ceilings=(178_000,),
    )

    assert hi is not None
    assert hi["candidate_id"] == ("hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000")
    assert hi_target is not None
    assert hi_target["capacity_source"] == "local_hinerv_target_modelsize"
    assert hi_target["target_modelsize_mparams"] == 0.02
    assert hi_target["candidate_id"].endswith("_tgtmp0p02")
    assert hi["num_pairs"] == 600
    assert hi["hard_byte_ceiling"] == 36_000
    assert hi["decoder_codec"] == "int4_mixed"
    assert hi["use_hierarchical_feature_grid"] is False
    assert hi["mid_injection_block_index"] == 1
    assert hi["fine_injection_block_index"] == 4
    assert hi_official is not None
    assert hi_official["candidate_id"] == ("hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_int4_mixed_ceil36000")
    assert hi_official["use_hierarchical_feature_grid"] is True
    assert hi_official["use_convnext_blocks"] is True
    assert hi_official["local_grid_levels"] == 2
    assert hi_official["local_grid_channels"] == 4
    assert hi_official["convnext_mlp_ratio"] == 2
    assert hi_official["convnext_kernel_size"] == 7
    assert hi_official["mid_injection_block_index"] == 1
    assert hi_official["fine_injection_block_index"] == 4
    assert hi_official["hard_byte_ceiling"] == 36_000
    assert sn is not None
    assert sn["candidate_id"] == ("snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000")
    assert sn["legacy_candidate_id"] is True
    assert sn["num_pairs"] == 600
    assert sn["hard_byte_ceiling"] == 36_000
    assert sn["decoder_payload_codec"] == "int2_symmetric"
    assert sn_rich is not None
    assert sn_rich["candidate_id"] == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p1_mfu1-3_hfr0p25_t1_adbase_int2_symmetric_ceil36000"
    )
    assert sn_rich["wavelet"] == "haar"
    assert sn_rich["fc_dim"] == 11
    assert sn_rich["emb_size"] == 2
    assert sn_rich["patch_radius"] == 1
    assert sn_rich["mfu_scales"] == [1, 3]
    assert sn_rich["hfr_gain"] == 0.25
    assert sn_rich["temporal_context"] == 1
    assert sn_rich["snerv_model_size_adapter"] == ("snerv_fc_dim_emb_size_adapter_v1")
    assert sn_spectra is not None
    assert sn_spectra["candidate_id"] == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_mfu1-5_hfr0p375_t2_adspectra_int2_symmetric_ceil36000"
    )
    assert sn_spectra["patch_radius"] == 3
    assert sn_spectra["mfu_scales"] == [1, 5]
    assert sn_spectra["hfr_gain"] == pytest.approx(0.375)
    assert sn_spectra["temporal_context"] == 2
    assert sn_spectra["snerv_model_size_adapter"] == (SNERV_SPECTRA_PRESERVING_ADAPTER)
    assert sn_temporal is not None
    assert sn_temporal["temporal_context"] == 2
    assert sn_temporal["temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert sn_temporal["candidate_id"].find("_tmhaar1_") >= 0
    assert sn_shared_mean is not None
    assert sn_shared_mean["official_skip_high_mode"] == "shared_mean"
    assert sn_shared_mean["snerv_official_skip_high_mode"] == "shared_mean"
    assert sn_shared_mean["candidate_id"] == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
        "mfu1-5_hfr0p375_t2_adspectra_sksharedmean_"
        "int2_symmetric_ceil36000"
    )
    assert sn_official_requested_bits is not None
    assert sn_official_requested_bits["candidate_id"] == (
        "snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_"
        "mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_"
        "skchannelmean_int8_symmetric_ceil178000"
    )
    assert sn_official_requested_bits["canonical_candidate_id"] == (
        "snerv_np600_haar_lv1_lfb1_stepb1_fc11e0_p1_"
        "mfu1-2-4_hfr0_t0_tmhaar1_adofficial_oms0p05_"
        "skchannelmean_int8_symmetric_ceil178000"
    )
    assert sn_official_requested_bits["legacy_requested_lf_bit_candidate_id"] is True
    assert sn_official_requested_bits["bits_per_coeff"] == pytest.approx(1.5)
    assert sn_official_requested_bits["step_map_bits_per_coeff"] == pytest.approx(0.5)
    assert sn_official_requested_bits["requested_bits_per_coeff_from_candidate_id"] == (pytest.approx(1.5))
    assert sn_official_requested_bits["requested_step_map_bits_per_coeff_from_candidate_id"] == pytest.approx(0.5)
    assert sn_official_requested_bits["fc_dim"] == 11
    assert sn_official_requested_bits["emb_size"] == 0
    assert sn_official_requested_bits["snerv_model_size_adapter"] == (SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER)
    assert sn_official_requested_bits["temporal_mode"] == ("official_haar_dwt1d_lowpass")
    assert sn_official_requested_bits["official_skip_high_mode"] == "channel_mean"
    with pytest.raises(CompactRendererMlxSpineRunnerError):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id=(
                "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_mfu1-5_hfr0p3750_t2_adspectra_int2_symmetric_ceil36000"
            ),
            hard_byte_ceilings=(178_000,),
        )


def test_pose_instability_epoch_monitor_requires_sustained_bad_pose() -> None:
    monitor = runner_mod._PoseInstabilityEpochMonitor(
        min_epoch=2,
        consecutive_bad_epochs=2,
        pose_loss_threshold=100.0,
        pose_axis_threshold=100.0,
    )
    ok = SimpleNamespace(
        epoch=2,
        loss_components={"loss_part_pose_distill": 99.0},
        per_axis_decomposition={"pose": 99.0},
    )
    bad_a = SimpleNamespace(
        epoch=3,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 10.0},
    )
    bad_b = SimpleNamespace(
        epoch=4,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )

    monitor(ok)
    monitor(bad_a)
    with pytest.raises(runner_mod.LongTrainingStopRequested):
        monitor(bad_b)


def test_pose_instability_epoch_monitor_prefers_pose_score_term_units() -> None:
    monitor = runner_mod._PoseInstabilityEpochMonitor(
        min_epoch=0,
        consecutive_bad_epochs=1,
        pose_loss_threshold=100.0,
        pose_axis_threshold=100.0,
    )
    raw_mse_only_would_be_bad = SimpleNamespace(
        epoch=2,
        loss_components={
            "loss_part_pose_distill": 10_000.0,
            "loss_part_pose_score_term": 99.0,
        },
        per_axis_decomposition={"pose": 99.0},
    )

    monitor(raw_mse_only_would_be_bad)

    assert monitor.bad_epoch_count == 0
    assert monitor.as_dict()["pose_loss_metric_name"] == "loss_part_pose_score_term"


def test_pose_instability_epoch_monitor_uses_resume_local_epoch_window() -> None:
    monitor = runner_mod._PoseInstabilityEpochMonitor(
        start_epoch=9_000,
        min_epoch=2,
        consecutive_bad_epochs=2,
        pose_loss_threshold=100.0,
        pose_axis_threshold=100.0,
    )
    first_bad_resume_epoch = SimpleNamespace(
        epoch=9_000,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )
    second_bad_resume_epoch = SimpleNamespace(
        epoch=9_001,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )
    third_bad_resume_epoch = SimpleNamespace(
        epoch=9_002,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )
    fourth_bad_resume_epoch = SimpleNamespace(
        epoch=9_003,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )

    monitor(first_bad_resume_epoch)
    monitor(second_bad_resume_epoch)
    assert monitor.bad_epoch_count == 0
    monitor(third_bad_resume_epoch)
    assert monitor.bad_epoch_count == 1
    with pytest.raises(
        runner_mod.LongTrainingStopRequested,
        match=r"local_epoch=3:.*start_epoch=9000",
    ):
        monitor(fourth_bad_resume_epoch)


def test_pose_instability_epoch_monitor_logs_but_does_not_stop_hard_pair_refit() -> None:
    monitor = runner_mod._PoseInstabilityEpochMonitor(
        min_epoch=0,
        consecutive_bad_epochs=1,
        pose_loss_threshold=100.0,
        pose_axis_threshold=100.0,
        hard_pair_sampling_active=True,
    )
    bad = SimpleNamespace(
        epoch=9_000,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )

    monitor(bad)

    assert monitor.bad_epoch_count == 1
    assert monitor.as_dict()["hard_pair_axis_stop_enabled"] is False
    assert "stop_suppressed_for_hard_pair_sampling" in monitor.last_reason


def test_resume_start_epoch_for_pose_monitor_reads_checkpoint_meta(tmp_path: Path) -> None:
    meta = tmp_path / "epoch008999.meta.json"
    meta.write_text(json.dumps({"global_epoch": 8_999}), encoding="utf-8")

    assert runner_mod._resume_start_epoch_for_pose_monitor(meta) == 9_000
    assert runner_mod._resume_start_epoch_for_pose_monitor(None) == 0
    assert runner_mod._resume_start_epoch_for_pose_monitor(tmp_path / "missing.json") == 0


def test_active_campaign_lock_identity_excludes_output_dir(tmp_path: Path) -> None:
    weight = tmp_path / "weights.npz"
    np.savez_compressed(weight, weight=np.ones((1,), dtype=np.float32))
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args_a = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--output-dir",
            str(tmp_path / "a"),
            "--num-pairs",
            "600",
            "--epochs",
            "8",
            "--recon-pixel-weight-path",
            str(weight),
        ]
    )
    args_b = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--output-dir",
            str(tmp_path / "b"),
            "--num-pairs",
            "600",
            "--epochs",
            "8",
            "--recon-pixel-weight-path",
            str(weight),
        ]
    )

    payload_a = runner_mod._active_campaign_lock_payload(
        args_a,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )
    payload_b = runner_mod._active_campaign_lock_payload(
        args_b,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )

    assert payload_a == payload_b
    assert payload_a["recon_pixel_weight_sha256"]
    assert runner_mod._campaign_lock_digest(payload_a) == (runner_mod._campaign_lock_digest(payload_b))


def test_active_campaign_lock_refuses_duplicate_active_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "600",
            "--epochs",
            "8",
        ]
    )
    monkeypatch.setattr(runner_mod, "_active_family_campaign_processes", lambda **_: [])

    lock_path = runner_mod._acquire_active_campaign_lock(
        output_dir=tmp_path / "a",
        args=args,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )
    assert lock_path is not None
    assert lock_path.is_file()
    manifest = json.loads(lock_path.read_text(encoding="utf-8"))
    assert manifest["family"] == "hi_nerv"
    assert manifest["planner_row_id"] is None
    assert manifest["modelsize_candidate_id"] == "auto"
    assert manifest["score_claim"] is False
    try:
        with pytest.raises(SystemExit, match="duplicate active"):
            runner_mod._acquire_active_campaign_lock(
                output_dir=tmp_path / "b",
                args=args,
                source_video_path=source,
                hard_byte_ceilings=(178_000,),
            )
    finally:
        runner_mod._release_active_campaign_lock(lock_path, os.getpid())
    assert not lock_path.exists()


def test_active_campaign_lock_refuses_same_family_snerv_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "32",
            "--epochs",
            "1",
        ]
    )
    monkeypatch.setattr(
        runner_mod,
        "_active_process_table_rows",
        lambda: [
            {
                "pid": 424242,
                "ppid": 1,
                "elapsed": "03:55",
                "command": (
                    ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py "
                    "--n-pairs 600 --packet-out /Volumes/VertigoDataTier/pact/x.snar"
                ),
            }
        ],
    )
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)

    with pytest.raises(SystemExit, match="active same-family"):
        runner_mod._acquire_active_campaign_lock(
            output_dir=tmp_path / "snerv_run",
            args=args,
            source_video_path=source,
            hard_byte_ceilings=(178_000,),
        )

    refusal_paths = sorted(
        (tmp_path / ".active_compact_renderer_campaign_locks").glob("family_process_refusal_snerv_*.json")
    )
    assert len(refusal_paths) == 1
    payload = json.loads(refusal_paths[0].read_text(encoding="utf-8"))
    assert payload["schema"] == runner_mod.ACTIVE_FAMILY_PROCESS_REFUSAL_SCHEMA
    assert payload["family"] == "snerv"
    assert payload["active_processes"][0]["pid"] == 424242
    assert payload["score_claim"] is False


def test_active_family_process_detection_ignores_current_process_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid in {10, 11, 12})
    rows = [
        {
            "pid": 10,
            "ppid": 9,
            "elapsed": "00:01",
            "command": ("/bin/zsh -lc python tools/run_compact_renderer_mlx_spine_runner.py --execute-family snerv"),
        },
        {
            "pid": 11,
            "ppid": 10,
            "elapsed": "00:01",
            "command": (
                "/Users/adpena/Projects/pact/.venv/bin/python "
                "tools/run_compact_renderer_mlx_spine_runner.py --execute-family snerv"
            ),
        },
        {
            "pid": 12,
            "ppid": 1,
            "elapsed": "03:55",
            "command": (".venv/bin/python tools/run_snerv_inverse_steg_advisory.py --n-pairs 600"),
        },
    ]

    matches = runner_mod._active_family_campaign_processes(
        family="snerv",
        current_pid=11,
        process_rows=rows,
    )

    assert [row["pid"] for row in matches] == [12]


def test_active_family_process_detection_ignores_hinerv_pytest_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)
    rows = [
        {
            "pid": 424242,
            "ppid": 1,
            "elapsed": "01:25",
            "command": (
                ".venv/bin/python -m pytest -q "
                "src/tac/tests/test_hinerv_archive_size_ladder.py "
                "src/tac/tests/test_compact_renderer_mlx_spine_runner.py"
            ),
        },
    ]

    matches = runner_mod._active_family_campaign_processes(
        family="hi_nerv",
        current_pid=1,
        process_rows=rows,
    )

    assert matches == []


def test_active_campaign_lock_allow_duplicate_skips_family_process_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "32",
            "--epochs",
            "1",
            "--allow-duplicate-campaign",
        ]
    )
    monkeypatch.setattr(
        runner_mod,
        "_active_process_table_rows",
        lambda: [
            {
                "pid": 424242,
                "ppid": 1,
                "elapsed": "03:55",
                "command": ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py",
            }
        ],
    )
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)

    lock_path = runner_mod._acquire_active_campaign_lock(
        output_dir=tmp_path / "snerv_run",
        args=args,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )

    assert lock_path is None
    assert not (tmp_path / ".active_compact_renderer_campaign_locks").exists()


@pytest.mark.parametrize(
    "function_name",
    ("_run_pact_nerv_vq_mlx_smoke", "_run_pact_nerv_selector_v4_mlx_smoke"),
)
def test_pact_compact_runners_forward_optimizer_and_shared_qat_metadata(
    function_name: str,
) -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = [
        node
        for node in ast.walk(target_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    assert len(calls) == 1
    kw_names = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert "pr95_faithful_curriculum_enabled" in kw_names
    assert "pr95_curriculum_total_epochs" in kw_names
    assert "grad_clip_max_norm" in kw_names
    assert "weight_decay" in kw_names
    assert "optimizer_kind" in kw_names
    assert "warmup_epochs" in kw_names
    assert "warmup_steps_per_epoch" in kw_names
    assert "cosine_decay_enabled" in kw_names
    assert "cosine_decay_total_epochs" in kw_names
    assert "cosine_decay_min_lr_ratio" in kw_names
    target_source = ast.get_source_segment(source, target_fn) or ""
    assert '"optimizer_policy": strip_candidate_curriculum_authority_fields' in (target_source)
    assert '"optimizer_controls": strip_candidate_curriculum_authority_fields' in (target_source)
    assert '"coder_aware_qat": coder_qat_metadata_row' in target_source


def test_hinerv_runner_forwards_train_time_dual_ascent_to_shared_harness() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_hi_nerv_mlx_scoreaware_smoke"
    )
    calls = [
        node
        for node in ast.walk(target_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    assert len(calls) == 1
    kw_names = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert "train_time_dual_ascent_config" in kw_names
    assert "gradient_multiplier_by_name" in kw_names
    assert "bias_gradient_multiplier" in kw_names
    assert "output_head_bias_gradient_multiplier" in kw_names
    assert "scorer_space_step_guard_enabled" in kw_names
    assert "scorer_space_step_guard_min_pre_segnet_occupied_class_fraction" in kw_names
    assert "scorer_space_step_guard_min_post_segnet_occupied_class_fraction" in kw_names
    assert "scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction" in kw_names
    assert "scorer_space_step_guard_min_post_segnet_target_class_min_ratio" in kw_names
    assert "scorer_space_step_guard_max_post_segnet_target_class_ratio_drop" in kw_names
    assert "scorer_space_step_guard_max_post_segnet_contrast_ratio" in kw_names
    assert "scorer_space_step_guard_max_post_segnet_distribution_mae" in kw_names
    assert "scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae" in kw_names
    assert "scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio" in kw_names
    assert "scorer_space_step_guard_max_pose_score_term_relative_worsening" in kw_names
    assert "scorer_space_step_guard_max_pose_score_term_absolute_worsening" in kw_names
    assert "scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening" in kw_names
    assert "scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening" in kw_names
    assert "scorer_space_step_guard_max_direct_nonrate_score_worsening" in kw_names
    assert "scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening" in kw_names
    assert "scorer_space_step_guard_backtracking_steps" in kw_names
    assert "scorer_space_step_guard_backtracking_shrink" in kw_names
    assert "checkpoint_selection_metric_key" in kw_names
    assert "checkpoint_selection_metric_mode" in kw_names
    assert "checkpoint_selection_metric_required" in kw_names
    target_source = ast.get_source_segment(source, target_fn) or ""
    assert "segnet_direct_live_class_histogram_weight" in target_source
    assert "segnet_direct_live_class_balanced_hinge_weight" in target_source
    assert "build_default_nerv_train_time_dual_ascent_config" in target_source
    assert "build_hinerv_archive_section_qat_weight_policy" in target_source
    assert "joint_scorer_checkpoint_selection_active" in target_source
    assert "direct_live_class_escape_checkpoint_selection_active" in target_source
    assert "shape_tether_checkpoint_selection_active" in target_source
    assert "not direct_live_class_escape_checkpoint_selection_active" in target_source
    assert "loss_part_joint_scorer_proxy_nonrate" in target_source
    assert "loss_part_scorer_input_shape_tether" in target_source
    assert "posenet_yuv6_geometry_tether_weight" in target_source
    assert "posenet_yuv6_geometry_tether_stage_weight" in target_source
    assert "posenet_temporal_signal_floor_weight" in target_source
    assert "posenet_temporal_signal_floor_stage_weight" in target_source
    assert "loss_part_segnet_direct_live_escape_selection" in target_source
    assert "loss_part_segnet_direct_live_argmax_disagreement" in target_source
    assert "checkpoint_selection_metric_required" in target_source
    assert "archive_section_qat_policy = " in target_source
    assert "latent_qat_cfg = CoderAwareQATConfig" in target_source
    assert 'terms[f"latent_qat_{suffix}"] = value' in target_source
    assert '"archive_section_qat_weight_policy": (' in target_source
    assert '"train_time_dual_ascent": strip_candidate_curriculum_authority_fields' in (target_source)


def test_snerv_native_attachment_forwards_train_time_dual_ascent_to_shared_harness() -> None:
    from tac.substrates.snerv_inverse_steg_carrier import (
        mlx_native_train_export as snerv_export,
    )

    source = Path(snerv_export.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_score_aware_long_training_attachment"
    )
    calls = [
        node
        for node in ast.walk(target_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    assert len(calls) == 1
    kw_names = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert "train_time_dual_ascent_config" in kw_names
    target_source = ast.get_source_segment(source, target_fn) or ""
    assert "build_default_nerv_train_time_dual_ascent_config" in target_source
    assert "scorer_input_distribution_guard_weight=guard_weight" in target_source
    assert "posenet_yuv6_geometry_tether_weight=geometry_tether_weight" in (target_source)
    assert "score_aware_long_training_posenet_temporal_signal_floor_weight" in (target_source)
    assert '"train_time_dual_ascent": (' in target_source


def test_snerv_required_control_contract_rejects_missing_temporal_floor_binding() -> None:
    contract = runner_mod._snerv_score_aware_long_training_required_control_contract(
        executed=True,
        score_aware_long_training={
            "posenet_temporal_signal_floor_bound": False,
        },
        training_telemetry_contract={
            "expected_posenet_temporal_signal_floor_metric": True,
            "posenet_temporal_signal_floor_metric_observed": True,
            "posenet_temporal_signal_floor_std_ratio_metric_observed": True,
            "posenet_temporal_signal_floor_mean_abs_ratio_metric_observed": True,
        },
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_contrast_floor_weight=0.0,
        scorer_input_shape_tether_weight=0.0,
        posenet_yuv6_geometry_tether_weight=0.0,
        posenet_temporal_signal_floor_weight=1.0,
    )

    assert contract["passed"] is False
    assert contract["controls"]["posenet_temporal_signal_floor"]["required"] is True
    assert contract["controls"]["posenet_temporal_signal_floor"]["bound"] is False
    assert (
        "snerv_score_aware_long_training_posenet_temporal_signal_floor_required_control_not_bound"
        in contract["blockers"]
    )


def test_snerv_required_control_contract_rejects_missing_geometry_binding() -> None:
    contract = runner_mod._snerv_score_aware_long_training_required_control_contract(
        executed=True,
        score_aware_long_training={
            "posenet_yuv6_geometry_tether_bound": False,
        },
        training_telemetry_contract={
            "expected_posenet_yuv6_geometry_tether_metric": True,
            "posenet_yuv6_geometry_tether_metric_observed": True,
            "posenet_yuv6_geometry_tether_pair_metric_observed": True,
            "posenet_yuv6_geometry_tether_delta_metric_observed": True,
        },
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_contrast_floor_weight=0.0,
        scorer_input_shape_tether_weight=0.0,
        posenet_yuv6_geometry_tether_weight=1.0,
        posenet_temporal_signal_floor_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["controls"]["posenet_yuv6_geometry_tether"]["required"] is True
    assert contract["controls"]["posenet_yuv6_geometry_tether"]["bound"] is False
    assert (
        "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_required_control_not_bound"
        in contract["blockers"]
    )


def test_mlx_optimizer_controls_default_to_pact_muon_adamw() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind=None,
        requested_weight_decay=None,
        grad_clip_max_norm=1.0,
        warmup_epochs=0,
        warmup_steps_per_epoch=1,
        cosine_decay_enabled=False,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=1e-2,
        run_epochs=128,
    )

    assert controls["optimizer_kind"] == runner_mod.DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert controls["weight_decay_defaulted"] is True
    assert controls["grad_clip_max_norm"] == pytest.approx(1.0)
    assert controls["borrowed_pr95_partition_rule"] is True
    assert controls["score_claim"] is False
    assert controls["ready_for_exact_eval_dispatch"] is False


def test_compact_runner_parser_defaults_to_shared_optimizer_kind() -> None:
    args = _parse_args(["--execute-family", "hi_nerv", "--num-pairs", "1"])

    assert args.optimizer_kind == runner_mod.DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert args.segnet_distillation_objective == runner_mod.HI_NERV_DEFAULT_SEGNET_DISTILLATION_OBJECTIVE
    assert args.segnet_distillation_objective == "boundary_argmax_hinge"
    assert args.gradient_multiplier_by_name == []
    assert args.bias_gradient_multiplier is None
    assert args.output_head_bias_gradient_multiplier == pytest.approx(1.0)
    assert args.scorer_space_step_guard_enabled is True
    assert args.scorer_space_step_guard_min_pre_segnet_occupied_class_fraction == pytest.approx(0.4)
    assert args.scorer_space_step_guard_min_post_segnet_occupied_class_fraction == pytest.approx(
        runner_mod.HI_NERV_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE
    )

    assert args.scorer_space_step_guard_max_post_segnet_contrast_ratio == (pytest.approx(4.25))
    assert args.scorer_space_step_guard_max_post_segnet_distribution_mae == (
        pytest.approx(runner_mod.HI_NERV_SEGNET_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD)
    )
    assert args.scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae == (
        pytest.approx(runner_mod.HI_NERV_POSENET_YUV6_DISTRIBUTION_MAE_MAX_FOR_STEP_GUARD)
    )
    assert args.scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio == (
        pytest.approx(runner_mod.HI_NERV_POSENET_YUV6_CONTRAST_RATIO_MAX_FOR_STEP_GUARD)
    )
    assert args.scorer_space_step_guard_max_post_segnet_argmax_disagreement == (pytest.approx(0.5))
    assert args.scorer_space_step_guard_max_post_pose_score_term is None
    assert args.scorer_space_step_guard_max_post_pose_direct_live_score_term is None
    assert args.scorer_space_step_guard_max_pose_score_term_relative_worsening == pytest.approx(
        runner_mod.HI_NERV_POSE_SCORE_TERM_MAX_RELATIVE_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_max_pose_score_term_absolute_worsening == pytest.approx(
        runner_mod.HI_NERV_POSE_SCORE_TERM_MAX_ABSOLUTE_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening == pytest.approx(
        runner_mod.HI_NERV_POSE_SCORE_TERM_MAX_RELATIVE_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening == pytest.approx(
        runner_mod.HI_NERV_POSE_SCORE_TERM_MAX_ABSOLUTE_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_max_direct_nonrate_score_worsening == pytest.approx(
        runner_mod.HI_NERV_DIRECT_NONRATE_SCORE_MAX_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening == pytest.approx(
        runner_mod.HI_NERV_BOOTSTRAP_DIRECT_NONRATE_SCORE_MAX_WORSENING_FOR_STEP_GUARD
    )
    assert args.scorer_space_step_guard_backtracking_steps == 6
    assert args.scorer_space_step_guard_backtracking_shrink == pytest.approx(0.5)
    assert args.scorer_support_ladder_enabled is True
    assert args.scorer_support_ladder_patience_steps == 1
    assert args.scorer_support_ladder_growth_factor == pytest.approx(2.0)
    assert args.scorer_support_ladder_max_multiplier == pytest.approx(16.0)
    assert args.scorer_support_ladder_base_loss_max_when_active == pytest.approx(0.25)
    assert args.checkpoint_interval_epochs == runner_mod.DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS
    assert args.checkpoint_dir is None
    assert args.resume_from_checkpoint is None
    assert args.hi_nerv_pr95_curriculum_total_epochs is None


def test_snerv_score_aware_parser_defaults_eval_roundtrip_on() -> None:
    default_args = _parse_args(["--execute-family", "snerv", "--num-pairs", "1"])
    ablation_args = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "1",
            "--no-snerv-score-aware-long-training-eval-roundtrip-ste",
        ]
    )

    assert default_args.snerv_score_aware_long_training_eval_roundtrip_ste is True
    assert ablation_args.snerv_score_aware_long_training_eval_roundtrip_ste is False


def test_hinerv_main_forwards_direct_live_pose_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_execute_hi_nerv(**kwargs):
        captured.update(kwargs)
        report_path = Path(kwargs["output_dir"]) / "compact_renderer_mlx_spine_runner_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": runner_mod.COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
            "mode": "unit_hi_nerv_forwarded",
            "report_path": report_path.as_posix(),
            "archive_path": str(tmp_path / "archive.zip"),
            "archive_bytes": 123,
            "archive_sha256": "a" * 64,
            "blockers": ["unit_false_authority"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        runner_mod,
        "execute_hi_nerv_mlx_scoreaware_and_adapt",
        fake_execute_hi_nerv,
    )

    rc = runner_mod.main(
        [
            "--execute-family",
            "hi_nerv",
            "--output-dir",
            str(tmp_path / "out"),
            "--source-video-path",
            str(REPO_ROOT / "upstream/videos/0.mkv"),
            "--num-pairs",
            "1",
            "--epochs",
            "1",
            "--batch-pairs",
            "1",
            "--planner-row-id",
            "unit_hinerv_direct_live_pose_forwarding",
            "--segnet-distillation-weight",
            "0.25",
            "--pose-distillation-weight",
            "0.125",
            "--pose-direct-live-distillation-weight",
            "0.0625",
            "--pose-distillation-warmup-epochs",
            "1",
            "--segnet-direct-live-escape-warmup-epochs",
            "1",
            "--segnet-direct-live-escape-class-multiplier",
            "3.5",
            "--allow-unscored-research-smoke",
            "--allow-bounded-planner-row-timing-smoke-waiver",
            "--allow-duplicate-campaign",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert captured["pose_distillation_weight"] == pytest.approx(0.125)
    assert captured["pose_direct_live_distillation_weight"] == pytest.approx(0.0625)
    assert captured["pose_distillation_warmup_epochs"] == 1
    assert captured["segnet_direct_live_escape_warmup_epochs"] == 1
    assert captured["segnet_direct_live_escape_class_multiplier"] == pytest.approx(3.5)
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "unit_hi_nerv_forwarded"
    report_payload = json.loads(
        (tmp_path / "out" / "compact_renderer_mlx_spine_runner_report.json").read_text(encoding="utf-8")
    )
    _assert_compact_runner_rerun_provenance(
        report_payload,
        family="hi_nerv",
        output_dir=tmp_path / "out",
    )


def test_compact_runner_parser_accepts_hi_nerv_pr95_curriculum_total_epochs() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--epochs",
            "100",
            "--gradient-multiplier-by-name",
            "head_rgb_1.weight=4",
            "--gradient-multiplier-by-name",
            "blocks.6.conv.weight=2.5",
            "--bias-gradient-multiplier",
            "0.25",
            "--output-head-bias-gradient-multiplier",
            "0.5",
            "--hi-nerv-pr95-curriculum-total-epochs",
            "29650",
        ]
    )

    assert args.epochs == 100
    assert dict(args.gradient_multiplier_by_name) == {
        "head_rgb_1.weight": pytest.approx(4.0),
        "blocks.6.conv.weight": pytest.approx(2.5),
    }
    assert args.bias_gradient_multiplier == pytest.approx(0.25)
    assert args.output_head_bias_gradient_multiplier == pytest.approx(0.5)
    assert args.hi_nerv_pr95_curriculum_total_epochs == 29_650


def test_compact_runner_checkpoint_controls_parse_and_validate() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "1",
            "--checkpoint-interval-epochs",
            "17",
            "--checkpoint-dir",
            "ssd/checkpoints",
            "--resume-from-checkpoint",
            "ssd/checkpoints/epoch000016.meta.json",
        ]
    )

    assert args.checkpoint_interval_epochs == 17
    assert args.checkpoint_retention_keep_last_n == runner_mod.DEFAULT_COMPACT_FAMILY_CHECKPOINT_RETENTION_KEEP_LAST_N
    assert args.checkpoint_retention_keep_best_n == runner_mod.DEFAULT_COMPACT_FAMILY_CHECKPOINT_RETENTION_KEEP_BEST_N
    assert args.checkpoint_dir == Path("ssd/checkpoints")
    assert args.resume_from_checkpoint == Path("ssd/checkpoints/epoch000016.meta.json")
    assert runner_mod._resolve_checkpoint_interval_epochs(17, epochs=100) == 17
    assert runner_mod._resolve_checkpoint_retention_keep_last_n(-1) is None
    assert runner_mod._resolve_checkpoint_retention_keep_last_n(3) == 3
    assert runner_mod._resolve_checkpoint_retention_keep_best_n(2) == 2
    assert runner_mod._resolve_checkpoint_retention_keep_every_n_epochs(None) is None
    assert runner_mod._resolve_checkpoint_retention_keep_every_n_epochs(1000) == 1000
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="positive integer"):
        runner_mod._resolve_checkpoint_interval_epochs(True, epochs=100)
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="> 0"):
        runner_mod._resolve_checkpoint_interval_epochs(0, epochs=100)
    with pytest.raises(CompactRendererMlxSpineRunnerError, match=">= -1"):
        runner_mod._resolve_checkpoint_retention_keep_last_n(-2)
    with pytest.raises(CompactRendererMlxSpineRunnerError, match=">= 0"):
        runner_mod._resolve_checkpoint_retention_keep_best_n(-1)
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="> 0"):
        runner_mod._resolve_checkpoint_retention_keep_every_n_epochs(0)


def test_mlx_optimizer_controls_reject_weight_decay_for_no_decay_kind() -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="weight-decay"):
        runner_mod._resolve_mlx_score_aware_optimizer_controls(
            optimizer_kind="adam",
            requested_weight_decay=1.0e-4,
            grad_clip_max_norm=1.0,
            warmup_epochs=0,
            warmup_steps_per_epoch=1,
            cosine_decay_enabled=False,
            cosine_decay_total_epochs=None,
            cosine_decay_min_lr_ratio=1e-2,
            run_epochs=128,
        )


def test_mlx_optimizer_controls_default_cosine_total_to_run_epochs() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind="pact_muon_adamw",
        requested_weight_decay=2.0e-4,
        grad_clip_max_norm=0.5,
        warmup_epochs=5,
        warmup_steps_per_epoch=7,
        cosine_decay_enabled=True,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=5.0e-2,
        run_epochs=128,
    )

    assert controls["weight_decay_effective"] == pytest.approx(2.0e-4)
    assert controls["warmup_epochs"] == 5
    assert controls["warmup_steps_per_epoch"] == 7
    assert controls["cosine_decay_enabled"] is True
    assert controls["cosine_decay_total_epochs"] == 128
    assert controls["cosine_decay_total_epochs_defaulted_to_run_epochs"] is True
    assert controls["cosine_decay_min_lr_ratio"] == pytest.approx(5.0e-2)


@pytest.mark.parametrize(
    "execute_fn",
    (
        runner_mod.execute_pact_nerv_vq_mlx_smoke_and_adapt,
        runner_mod.execute_pact_nerv_selector_v4_mlx_smoke_and_adapt,
    ),
)
def test_pact_compact_execute_refuses_unsupported_weight_decay(
    tmp_path: Path,
    execute_fn,
) -> None:
    with pytest.raises(
        CompactRendererMlxSpineRunnerError,
        match="optimizer-weight-decay is only supported",
    ):
        execute_fn(
            output_dir=tmp_path / execute_fn.__name__,
            num_pairs=1,
            epochs=1,
            batch_pair_indices_per_step=1,
            learning_rate=1e-3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            optimizer_kind="adam",
            optimizer_weight_decay=1.0e-4,
            repo_root=REPO_ROOT,
        )


def test_default_source_video_resolves_from_external_upstream(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    upstream = tmp_path / "canonical_upstream"
    video = upstream / "videos" / "0.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    repo_root.mkdir()

    resolved = _resolve_source_video_path(
        runner_mod.DEFAULT_SOURCE_VIDEO_PATH,
        base=repo_root,
        upstream_dir=upstream,
    )

    assert resolved == video.resolve(strict=False)


def test_default_scorer_upstream_resolves_from_canonical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    repo_root.mkdir()
    canonical_upstream = tmp_path / "canonical_upstream"
    (canonical_upstream / "models").mkdir(parents=True)
    (canonical_upstream / "modules.py").write_text("# upstream\n", encoding="utf-8")
    (canonical_upstream / "models" / "posenet.safetensors").write_bytes(b"pose")
    (canonical_upstream / "models" / "segnet.safetensors").write_bytes(b"seg")
    monkeypatch.setattr(
        runner_mod,
        "DEFAULT_UPSTREAM_DIR",
        repo_root / "upstream",
    )
    monkeypatch.setattr(
        runner_mod,
        "CANONICAL_UPSTREAM_FALLBACK_DIR",
        canonical_upstream,
    )

    resolved = runner_mod._resolve_scorer_upstream_dir(
        repo_root,
        runner_mod.DEFAULT_UPSTREAM_DIR,
    )

    assert resolved == canonical_upstream.resolve(strict=False)


def test_explicit_scorer_upstream_does_not_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    repo_root.mkdir()
    explicit_upstream = tmp_path / "explicit_upstream"
    (explicit_upstream / "models").mkdir(parents=True)
    (explicit_upstream / "modules.py").write_text("# incomplete\n", encoding="utf-8")
    canonical_upstream = tmp_path / "canonical_upstream"
    (canonical_upstream / "models").mkdir(parents=True)
    (canonical_upstream / "modules.py").write_text("# upstream\n", encoding="utf-8")
    (canonical_upstream / "models" / "posenet.safetensors").write_bytes(b"pose")
    (canonical_upstream / "models" / "segnet.safetensors").write_bytes(b"seg")
    monkeypatch.setattr(
        runner_mod,
        "DEFAULT_UPSTREAM_DIR",
        repo_root / "upstream",
    )
    monkeypatch.setattr(
        runner_mod,
        "CANONICAL_UPSTREAM_FALLBACK_DIR",
        canonical_upstream,
    )

    resolved = runner_mod._resolve_scorer_upstream_dir(repo_root, explicit_upstream)

    assert resolved == explicit_upstream.resolve(strict=False)


def test_auto_joint_recon_pixel_weight_discovers_verified_pair_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_verified_joint_recon_weight(root, pairs=32, name="weight_32")
    weight_600, manifest_600 = _write_verified_joint_recon_weight(
        root,
        pairs=600,
        name="weight_600",
    )
    monkeypatch.setattr(runner_mod, "DEFAULT_SSD_ROOTS", ())

    discovered, metadata = runner_mod._discover_joint_recon_pixel_weight_path(
        repo_root=root,
        num_pairs=600,
    )

    assert discovered == weight_600.resolve(strict=False)
    assert metadata["selected_manifest_path"] == manifest_600.as_posix()
    assert metadata["selected_weight_sha256"] == runner_mod._sha256_file(weight_600)
    assert metadata["candidate_count"] == 1
    assert metadata["score_claim"] is False


def test_auto_joint_recon_pixel_weight_refuses_missing_pair_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_verified_joint_recon_weight(root, pairs=32, name="weight_32")
    monkeypatch.setattr(runner_mod, "DEFAULT_SSD_ROOTS", ())

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError, match="600"):
        runner_mod._discover_joint_recon_pixel_weight_path(
            repo_root=root,
            num_pairs=600,
        )


def test_pact_vq_execute_parser_exposes_real_scorer_binding_flags() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pact_nerv_vq",
            "--upstream-dir",
            "canonical_upstream",
            "--segnet-distillation-weight",
            "0.25",
            "--pose-distillation-weight",
            "0.75",
            "--pose-distillation-loss",
            "huber",
            "--pose-distillation-huber-delta",
            "2.5",
            "--pose-distillation-warmup-epochs",
            "7",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--distillation-temperature",
            "1.5",
            "--segnet-student-live-calibration-weight",
            "0.625",
            "--segnet-direct-live-base-loss-weight",
            "0.125",
            "--segnet-tau-boundary",
            "0.8",
            "--segnet-hinge-margin",
            "1.25",
            "--distillation-device",
            "cpu",
            "--compact-decoder-codec",
            "int8_scale_bundled",
            "--coder-aware-qat",
            "--coder-qat-quant-bits",
            "4",
            "--coder-qat-quant-residual-weight",
            "0.001",
            "--coder-qat-magnitude-weight",
            "0.0001",
            "--coder-qat-delta-weight",
            "0.0002",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hprc-queue-followup-report",
            "hprc_queue_followup_report.json",
        ]
    )

    assert args.segnet_distillation_weight == 0.25
    assert args.upstream_dir == Path("canonical_upstream")
    assert args.pose_distillation_weight == 0.75
    assert args.pose_distillation_loss == "huber"
    assert args.pose_distillation_huber_delta == 2.5
    assert args.pose_distillation_warmup_epochs == 7
    assert args.segnet_distillation_objective == "boundary_argmax_hinge"
    assert args.distillation_temperature == 1.5
    assert args.segnet_student_live_calibration_weight == pytest.approx(0.625)
    assert args.segnet_direct_live_base_loss_weight == pytest.approx(0.125)
    assert args.segnet_tau_boundary == 0.8
    assert args.segnet_hinge_margin == 1.25
    assert args.distillation_device == "cpu"
    assert args.compact_decoder_codec == "int8_scale_bundled"
    assert args.coder_aware_qat is True
    assert args.coder_qat_quant_bits == 4
    assert args.coder_qat_quant_residual_weight == 0.001
    assert args.coder_qat_magnitude_weight == 0.0001
    assert args.coder_qat_delta_weight == 0.0002
    assert args.coder_qat_c1a_entropy_weight == 0.0003
    assert args.coder_qat_c1a_sigma == 0.35
    assert args.coder_qat_c1a_sample_size == 64
    assert args.hprc_queue_followup_report == [Path("hprc_queue_followup_report.json")]
    assert args.allow_segnet_only_research is False


def test_execute_parser_accepts_all_pixel_argmax_hinge_bootstrap_objective() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--segnet-distillation-objective",
            "argmax_hinge",
        ]
    )

    assert args.segnet_distillation_objective == "argmax_hinge"


def test_real_scorer_distillation_requires_active_teacher_upstream_files(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "upstream"
    (incomplete / "models").mkdir(parents=True)
    (incomplete / "modules.py").write_text("# synthetic upstream\n", encoding="utf-8")

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        _require_scorer_upstream_dir_for_distillation(
            upstream_dir=incomplete,
            segnet_distillation_weight=0.1,
            pose_distillation_weight=0.0,
        )

    msg = str(exc.value)
    assert "--upstream-dir" in msg
    assert "segnet.safetensors" in msg
    assert "posenet.safetensors" not in msg

    (incomplete / "models" / "segnet.safetensors").write_bytes(b"seg")
    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        _require_scorer_upstream_dir_for_distillation(
            upstream_dir=incomplete,
            segnet_distillation_weight=0.0,
            pose_distillation_weight=0.1,
        )

    msg = str(exc.value)
    assert "posenet.safetensors" in msg
    assert "segnet.safetensors" not in msg


def test_selector_v4_execute_parser_exposes_real_family_controls() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pact_nerv_selector_v4",
            "--compact-selector-palette-size",
            "32",
            "--num-pairs",
            "600",
            "--segnet-distillation-weight",
            "0.2",
            "--pose-distillation-weight",
            "0.4",
            "--mlx-profile",
            "selector_section_value_profile.json",
        ]
    )

    assert args.execute_family == "pact_nerv_selector_v4"
    assert args.compact_selector_palette_size == 32
    assert args.num_pairs == 600
    assert args.segnet_distillation_weight == 0.2
    assert args.pose_distillation_weight == 0.4
    assert args.mlx_profile == [Path("selector_section_value_profile.json")]


def test_pr95_hnerv_execute_parser_exposes_public_archive_seed() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pr95_hnerv",
            "--pr95-source-archive",
            "public_pr95/archive.zip",
            "--num-pairs",
            "600",
            "--epochs",
            "2",
            "--run-receiver-proof",
            "--keep-receiver-proof-output",
            "--receiver-proof-timeout-seconds",
            "17",
        ]
    )

    assert args.execute_family == "pr95_hnerv"
    assert args.pr95_source_archive == Path("public_pr95/archive.zip")
    assert args.num_pairs == 600
    assert args.epochs == 2
    assert args.run_receiver_proof is True
    assert args.keep_receiver_proof_output is True
    assert args.receiver_proof_timeout_seconds == 17


def test_hinerv_snerv_execute_parser_accepts_planner_gated_families() -> None:
    hi = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "32",
            "--run-local-cpu-replay",
            "--keep-local-replay-inflated",
            "--retain-failed-local-replay-scratch",
            "--recon-pixel-weight-path",
            "weights.npz",
            "--auto-joint-recon-pixel-weight",
            "--auto-segnet-boundary-recon-weight",
            "--recon-pixel-weight-tau",
            "0.75",
            "--recon-pixel-weight-normalize",
            "none",
            "--recon-loss-stage-weight",
            "0.25",
            "--segnet-loss-stage-weight",
            "2.0",
            "--pose-loss-stage-weight",
            "1.5",
            "--scorer-input-guard-stage-weight",
            "0.75",
            "--scorer-input-contrast-floor-stage-weight",
            "1.25",
            "--scorer-input-shape-tether-stage-weight",
            "2.25",
            "--segnet-direct-live-stage-weight",
            "0.5",
            "--segnet-direct-live-target-mass-floor-weight",
            "0.1875",
            "--pose-distillation-warmup-epochs",
            "2",
            "--scorer-input-shape-warmup-epochs",
            "3",
            "--segnet-direct-live-escape-warmup-epochs",
            "1",
            "--segnet-direct-live-escape-class-multiplier",
            "2.75",
            "--scorer-input-distribution-guard-weight",
            "0.125",
            "--scorer-input-distribution-guard-saturation-margin",
            "0.03125",
            "--scorer-input-distribution-guard-temperature",
            "0.015625",
            "--scorer-input-contrast-floor-weight",
            "0.375",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.625",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.75",
            "--mlx-prefilter-scorer-batch-pairs",
            "8",
            "--mlx-prefilter-scorer-device",
            "gpu",
            "--mlx-prefilter-progress-every",
            "10",
            "--telemetry-flush-interval-epochs",
            "1",
            "--run-post-export-materializers",
            "--post-export-materializer-max-steps",
            "3",
            "--post-export-materializer-max-parallel",
            "2",
            "--post-export-materializer-max-experiments",
            "1",
            "--modelsize-candidate-id",
            "manual",
            "--planner-row-id",
            "hi_nerv::manual::lion",
            "--optimizer-kind",
            "lion",
            "--archive-section-telemetry-json",
            "hinerv_sections.json",
            "--hi-nerv-pr95-source-weight-amplification",
        ]
    )
    sn = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "128",
            "--coder-aware-qat",
            "--coder-qat-quant-bits",
            "4",
            "--snerv-scorer-loop-max-trials",
            "5",
            "--snerv-scorer-loop-search-mode",
            "learned_random_subspace",
            "--snerv-scorer-loop-byte-pressure-multiplier",
            "1.25",
            "--snerv-scorer-loop-section-value-pressure-multiplier",
            "1.75",
            "--snerv-scorer-loop-max-archive-byte-growth",
            "11",
            "--snerv-scorer-loop-byte-growth-admission-mode",
            "rate_paid",
            "--snerv-scorer-loop-lf-payload-codec",
            "auto",
            "--snerv-scorer-loop-pose-slack",
            "0.001",
            "--snerv-scorer-loop-seg-slack",
            "0.002",
            "--snerv-scorer-loop-pair-stride",
            "3",
            "--snerv-scorer-loop-start-pair",
            "7",
            "--snerv-spectra-preserving-adapter",
            "--snerv-mfu-scales",
            "1,3",
            "--snerv-model-size-adapter",
            "snerv_manual_unit_adapter",
            "--snerv-fc-dim",
            "13",
            "--snerv-emb-size",
            "5",
            "--snerv-patch-radius",
            "2",
            "--snerv-hfr-gain",
            "0.25",
            "--snerv-official-skip-high-mode",
            "shared_mean",
            "--snerv-temporal-context",
            "4",
            "--planner-row-id",
            "snerv::manual::native_rate_aware_training",
            "--skip-snerv-native-mlx-export",
            "--skip-snerv-native-mlx-archive-export",
            "--snerv-native-mlx-receiver-proof-timeout",
            "123",
            "--snerv-native-mlx-decoder-train-steps",
            "7",
            "--snerv-native-mlx-decoder-train-lr",
            "0.0003",
            "--snerv-native-mlx-decoder-train-ridge",
            "0.000004",
            "--snerv-native-mlx-decoder-train-optimizer",
            "lion",
            "--snerv-official-trained-checkpoint-state-dict-path",
            "official_state.npz",
            "--snerv-score-aware-long-training-epochs",
            "17",
            "--snerv-score-aware-long-training-lr",
            "0.0025",
            "--snerv-score-aware-long-training-batch-pairs",
            "6",
            "--snerv-score-aware-long-training-section-byte-refresh-every-steps",
            "9",
            "--snerv-score-aware-long-training-optimizer",
            "lion",
            "--snerv-score-aware-long-training-grad-clip-max-norm",
            "0.75",
            "--snerv-score-aware-long-training-weight-decay",
            "-1",
            "--snerv-score-aware-long-training-eval-roundtrip-ste",
            "--snerv-score-aware-long-training-scorer-tether-smoke-steps",
            "4",
            "--segnet-direct-live-escape-class-multiplier",
            "3.25",
            "--segnet-direct-live-target-mass-floor-weight",
            "0.3125",
            "--scorer-input-distribution-guard-weight",
            "0.5",
            "--scorer-input-distribution-guard-saturation-margin",
            "0.04",
            "--scorer-input-distribution-guard-temperature",
            "0.02",
            "--scorer-input-contrast-floor-weight",
            "0.875",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.55",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.45",
            "--snerv-score-aware-long-training-pr95-muon-policy",
            "every_stage",
        ]
    )

    assert hi.execute_family == "hi_nerv"
    assert hi.num_pairs == 32
    assert hi.run_local_cpu_replay is True
    assert hi.keep_local_replay_inflated is True
    assert hi.retain_failed_local_replay_scratch is True
    assert hi.recon_pixel_weight_path == Path("weights.npz")
    assert hi.auto_joint_recon_pixel_weight is True
    assert hi.auto_segnet_boundary_recon_weight is True
    assert hi.recon_pixel_weight_tau == 0.75
    assert hi.recon_pixel_weight_normalize == "none"
    assert hi.recon_loss_stage_weight == 0.25
    assert hi.segnet_loss_stage_weight == 2.0
    assert hi.pose_loss_stage_weight == 1.5
    assert hi.scorer_input_guard_stage_weight == pytest.approx(0.75)
    assert hi.scorer_input_contrast_floor_stage_weight == pytest.approx(1.25)
    assert hi.scorer_input_shape_tether_stage_weight == pytest.approx(2.25)
    assert hi.segnet_direct_live_stage_weight == pytest.approx(0.5)
    assert hi.segnet_direct_live_target_mass_floor_weight == pytest.approx(0.1875)
    assert hi.pose_distillation_warmup_epochs == 2
    assert hi.scorer_input_shape_warmup_epochs == 3
    assert hi.segnet_direct_live_escape_warmup_epochs == 1
    assert hi.segnet_direct_live_escape_class_multiplier == pytest.approx(2.75)
    assert hi.scorer_input_distribution_guard_weight == pytest.approx(0.125)
    assert hi.scorer_input_distribution_guard_saturation_margin == pytest.approx(0.03125)
    assert hi.scorer_input_distribution_guard_temperature == pytest.approx(0.015625)
    assert hi.scorer_input_contrast_floor_weight == pytest.approx(0.375)
    assert hi.scorer_input_contrast_floor_segnet_min_std_ratio == pytest.approx(0.625)
    assert hi.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio == pytest.approx(0.75)
    assert hi.mlx_prefilter_scorer_batch_pairs == 8
    assert hi.mlx_prefilter_scorer_device == "gpu"
    assert hi.mlx_prefilter_progress_every == 10
    assert hi.telemetry_flush_interval_epochs == 1
    assert hi.run_post_export_materializers is True
    assert hi.post_export_materializer_max_steps == 3
    assert hi.post_export_materializer_max_parallel == 2
    assert hi.post_export_materializer_max_experiments == 1
    assert hi.modelsize_candidate_id == "manual"
    assert hi.planner_row_id == "hi_nerv::manual::lion"
    assert hi.optimizer_kind == "lion"
    assert hi.archive_section_telemetry_json == Path("hinerv_sections.json")
    assert hi.hi_nerv_pr95_source_weight_amplification is True
    assert sn.execute_family == "snerv"
    assert sn.num_pairs == 128
    assert sn.coder_aware_qat is True
    assert sn.coder_qat_quant_bits == 4
    assert sn.snerv_scorer_loop_max_trials == 5
    assert sn.snerv_scorer_loop_search_mode == "learned_random_subspace"
    assert sn.snerv_scorer_loop_byte_pressure_multiplier == 1.25
    assert sn.snerv_scorer_loop_section_value_pressure_multiplier == 1.75
    assert sn.snerv_scorer_loop_max_archive_byte_growth == 11
    assert sn.snerv_scorer_loop_byte_growth_admission_mode == "rate_paid"
    assert sn.snerv_scorer_loop_lf_payload_codec == "auto"
    assert sn.snerv_scorer_loop_pose_slack == 0.001
    assert sn.snerv_scorer_loop_seg_slack == 0.002
    assert sn.snerv_scorer_loop_pair_stride == 3
    assert sn.snerv_scorer_loop_start_pair == 7
    assert sn.snerv_scorer_loop_pair_guard_min_score_improved_fraction == 1.0
    assert sn.snerv_scorer_loop_pair_guard_max_pose_worsened_fraction == 0.0
    assert sn.snerv_scorer_loop_component_guard_mode == "pose_seg_hard"
    assert sn.snerv_spectra_preserving_adapter is True
    assert sn.snerv_mfu_scales == "1,3"
    assert sn.snerv_model_size_adapter == "snerv_manual_unit_adapter"
    assert sn.snerv_fc_dim == 13
    assert sn.snerv_emb_size == 5
    assert sn.snerv_patch_radius == 2
    assert sn.snerv_hfr_gain == 0.25
    assert sn.snerv_official_skip_high_mode == "shared_mean"
    assert sn.snerv_temporal_context == 4
    assert sn.planner_row_id == "snerv::manual::native_rate_aware_training"
    assert sn.skip_snerv_native_mlx_export is True
    assert sn.skip_snerv_native_mlx_archive_export is True
    assert sn.snerv_native_mlx_receiver_proof_timeout == 123
    assert sn.snerv_native_mlx_decoder_train_steps == 7
    assert sn.snerv_native_mlx_decoder_train_lr == 0.0003
    assert sn.snerv_native_mlx_decoder_train_ridge == 0.000004
    assert sn.snerv_native_mlx_decoder_train_optimizer == "lion"
    assert sn.snerv_official_trained_checkpoint_state_dict_path == Path("official_state.npz")
    assert sn.snerv_score_aware_long_training_epochs == 17
    assert sn.snerv_score_aware_long_training_lr == 0.0025
    assert sn.snerv_score_aware_long_training_batch_pairs == 6
    assert sn.snerv_score_aware_long_training_section_byte_refresh_every_steps == 9
    assert sn.snerv_score_aware_long_training_optimizer == "lion"
    assert sn.snerv_score_aware_long_training_grad_clip_max_norm == 0.75
    assert sn.snerv_score_aware_long_training_weight_decay == -1.0
    assert sn.snerv_score_aware_long_training_eval_roundtrip_ste is True
    assert sn.snerv_score_aware_long_training_scorer_tether_smoke_steps == 4
    assert sn.segnet_direct_live_escape_class_multiplier == pytest.approx(3.25)
    assert sn.segnet_direct_live_target_mass_floor_weight == pytest.approx(0.3125)
    assert sn.scorer_input_distribution_guard_weight == pytest.approx(0.5)
    assert sn.scorer_input_distribution_guard_saturation_margin == pytest.approx(0.04)
    assert sn.scorer_input_distribution_guard_temperature == pytest.approx(0.02)
    assert sn.scorer_input_contrast_floor_weight == pytest.approx(0.875)
    assert sn.scorer_input_contrast_floor_segnet_min_std_ratio == pytest.approx(0.55)
    assert sn.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio == pytest.approx(0.45)
    assert sn.snerv_score_aware_long_training_pr95_faithful_curriculum is True
    assert sn.snerv_score_aware_long_training_pr95_muon_policy == "every_stage"


def test_top_priority_family_cli_refuses_missing_planner_row(
    tmp_path: Path,
) -> None:
    out = tmp_path / "hinerv_missing_planner"
    rc = runner_mod.main(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "32",
            "--epochs",
            "29650",
            "--output-dir",
            out.as_posix(),
            "--repo-root",
            REPO_ROOT.as_posix(),
        ]
    )

    assert rc == 0
    report_path = out / "compact_renderer_mlx_spine_runner_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "compact_carrier_planner_row_launch_refused"
    assert report["execute_family"] == "hi_nerv"
    assert report["training_executed"] is False
    assert report["trainer_launch_allowed"] is False
    assert "hi_nerv_planner_row_id_missing" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_main_from_snerv_advisory_forwards_cleanup_scratch_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(json.dumps({"schema": "unit.snerv"}), encoding="utf-8")
    source_video = tmp_path / "0.mkv"
    source_video.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def fake_adapt_snerv_advisory_report_to_spine(**kwargs):
        calls.append(dict(kwargs))
        out = Path(kwargs["output_dir"])
        runner_report = out / "compact_renderer_mlx_spine_runner_report.json"
        runner_report.parent.mkdir(parents=True, exist_ok=True)
        runner_report.write_text("{}", encoding="utf-8")
        return {
            "mode": "adapted_snerv_advisory_report_to_spine",
            "report_path": runner_report.as_posix(),
            "blockers": ["unit_not_exact_ready"],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(runner_mod, "_acquire_active_campaign_lock", lambda **_: None)
    monkeypatch.setattr(
        runner_mod,
        "adapt_snerv_advisory_report_to_spine",
        fake_adapt_snerv_advisory_report_to_spine,
    )

    base_argv = [
        "--from-snerv-advisory-report",
        report_path.as_posix(),
        "--source-video-path",
        source_video.as_posix(),
        "--upstream-dir",
        (tmp_path / "upstream").as_posix(),
        "--repo-root",
        tmp_path.as_posix(),
    ]

    assert (
        runner_mod.main(
            [
                *base_argv,
                "--output-dir",
                (tmp_path / "default_cleanup").as_posix(),
            ]
        )
        == 0
    )
    assert calls[-1]["cleanup_failed_local_replay_scratch"] is True
    assert json.loads(capsys.readouterr().out)["ready_for_exact_eval_dispatch"] is False

    assert (
        runner_mod.main(
            [
                *base_argv,
                "--output-dir",
                (tmp_path / "retain_cleanup").as_posix(),
                "--retain-failed-local-replay-scratch",
            ]
        )
        == 0
    )
    assert calls[-1]["cleanup_failed_local_replay_scratch"] is False


def test_main_execute_snerv_forwards_direct_live_segnet_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_video = tmp_path / "0.mkv"
    source_video.write_bytes(b"video")
    captured: dict[str, object] = {}

    def fake_execute_snerv_inverse_steg_advisory_and_adapt(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "compact_renderer_mlx_spine_runner_report.json"
        report = {
            "schema": COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
            "mode": "unit_snerv_execute",
            "report_path": report_path.as_posix(),
            "blockers": ["unit_false_authority"],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        return report

    monkeypatch.setattr(runner_mod, "_acquire_active_campaign_lock", lambda **_: None)
    monkeypatch.setattr(
        runner_mod,
        "execute_snerv_inverse_steg_advisory_and_adapt",
        fake_execute_snerv_inverse_steg_advisory_and_adapt,
    )

    rc = runner_mod.main(
        [
            "--execute-family",
            "snerv",
            "--output-dir",
            (tmp_path / "snerv_direct_live").as_posix(),
            "--source-video-path",
            source_video.as_posix(),
            "--repo-root",
            tmp_path.as_posix(),
            "--modelsize-candidate-id",
            "manual",
            "--allow-manual-compact-family-launch",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-histogram-weight",
            "0.5",
            "--segnet-direct-live-class-balanced-hinge-weight",
            "0.75",
            "--segnet-direct-live-class-balanced-ce-weight",
            "0.625",
            "--segnet-direct-live-class-balanced-squared-hinge-weight",
            "0.875",
            "--segnet-direct-live-class-region-recon-weight",
            "0.9375",
            "--recon-loss-stage-weight",
            "0.5",
            "--segnet-loss-stage-weight",
            "1.5",
            "--pose-loss-stage-weight",
            "0.25",
            "--scorer-input-guard-stage-weight",
            "0.375",
            "--scorer-input-contrast-floor-stage-weight",
            "0.625",
            "--scorer-input-shape-tether-stage-weight",
            "0.875",
            "--posenet-yuv6-geometry-tether-stage-weight",
            "0.9375",
            "--segnet-direct-live-stage-weight",
            "0.125",
            "--pose-distillation-warmup-epochs",
            "2",
            "--scorer-input-shape-warmup-epochs",
            "3",
            "--segnet-direct-live-escape-warmup-epochs",
            "1",
            "--segnet-direct-live-escape-class-multiplier",
            "4.0",
            "--allow-segnet-only-research",
            "--scorer-space-step-guard-min-pre-segnet-occupied-class-fraction",
            "0.21",
            "--scorer-space-step-guard-min-post-segnet-occupied-class-fraction",
            "0.22",
            "--scorer-space-step-guard-min-post-segnet-target-class-coverage-fraction",
            "0.23",
            "--scorer-space-step-guard-max-post-segnet-contrast-ratio",
            "2.4",
            "--scorer-space-step-guard-max-post-segnet-distribution-mae",
            "0.24",
            "--scorer-space-step-guard-max-post-posenet-yuv6-distribution-mae",
            "0.25",
            "--scorer-space-step-guard-max-post-posenet-yuv6-contrast-ratio",
            "2.5",
            "--scorer-space-step-guard-max-post-segnet-argmax-disagreement",
            "0.26",
            "--scorer-space-step-guard-max-post-pose-score-term",
            "1.1",
            "--scorer-space-step-guard-max-post-pose-direct-live-score-term",
            "0.051",
            "--scorer-space-step-guard-max-pose-score-term-relative-worsening",
            "0.031",
            "--scorer-space-step-guard-max-pose-score-term-absolute-worsening",
            "0.032",
            "--scorer-space-step-guard-max-pose-direct-live-score-term-relative-worsening",
            "0.033",
            "--scorer-space-step-guard-max-pose-direct-live-score-term-absolute-worsening",
            "0.034",
            "--scorer-space-step-guard-max-direct-nonrate-score-worsening",
            "0.035",
            "--scorer-space-step-guard-max-bootstrap-direct-nonrate-score-worsening",
            "2.0",
            "--scorer-space-step-guard-backtracking-steps",
            "5",
            "--scorer-space-step-guard-backtracking-shrink",
            "0.45",
        ]
    )

    assert rc == 0
    assert captured["segnet_direct_live_distillation_weight"] == pytest.approx(0.25)
    assert captured["segnet_direct_live_class_histogram_weight"] == pytest.approx(0.5)
    assert captured["segnet_direct_live_class_balanced_hinge_weight"] == pytest.approx(0.75)
    assert captured["segnet_direct_live_class_balanced_ce_weight"] == pytest.approx(0.625)
    assert captured["segnet_direct_live_class_balanced_squared_hinge_weight"] == pytest.approx(0.875)
    assert captured["segnet_direct_live_class_region_recon_weight"] == pytest.approx(0.9375)
    assert captured["recon_loss_stage_weight"] == pytest.approx(0.5)
    assert captured["segnet_loss_stage_weight"] == pytest.approx(1.5)
    assert captured["pose_loss_stage_weight"] == pytest.approx(0.25)
    assert captured["scorer_input_guard_stage_weight"] == pytest.approx(0.375)
    assert captured["scorer_input_contrast_floor_stage_weight"] == pytest.approx(0.625)
    assert captured["scorer_input_shape_tether_stage_weight"] == pytest.approx(0.875)
    assert captured["posenet_yuv6_geometry_tether_stage_weight"] == pytest.approx(0.9375)
    assert captured["segnet_direct_live_stage_weight"] == pytest.approx(0.125)
    assert captured["pose_distillation_warmup_epochs"] == 2
    assert captured["scorer_input_shape_warmup_epochs"] == 3
    assert captured["segnet_direct_live_escape_warmup_epochs"] == 1
    assert captured["segnet_direct_live_escape_class_multiplier"] == pytest.approx(4.0)
    assert captured["scorer_space_step_guard_enabled"] is True
    assert captured["scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"] == pytest.approx(0.21)
    assert captured["scorer_space_step_guard_min_post_segnet_occupied_class_fraction"] == pytest.approx(0.22)
    assert captured["scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"] == pytest.approx(0.23)
    assert captured["scorer_space_step_guard_max_post_segnet_contrast_ratio"] == (pytest.approx(2.4))
    assert captured["scorer_space_step_guard_max_post_segnet_distribution_mae"] == (pytest.approx(0.24))
    assert captured["scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"] == pytest.approx(0.25)
    assert captured["scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"] == pytest.approx(2.5)
    assert captured["scorer_space_step_guard_max_post_segnet_argmax_disagreement"] == (pytest.approx(0.26))
    assert captured["scorer_space_step_guard_max_post_pose_score_term"] == (pytest.approx(1.1))
    assert captured["scorer_space_step_guard_max_post_pose_direct_live_score_term"] == (pytest.approx(0.051))
    assert captured["scorer_space_step_guard_max_pose_score_term_relative_worsening"] == pytest.approx(0.031)
    assert captured["scorer_space_step_guard_max_pose_score_term_absolute_worsening"] == pytest.approx(0.032)
    assert captured["scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening"] == pytest.approx(
        0.033
    )
    assert captured["scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening"] == pytest.approx(
        0.034
    )
    assert captured["scorer_space_step_guard_max_direct_nonrate_score_worsening"] == (pytest.approx(0.035))
    assert captured["scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"] == pytest.approx(2.0)
    assert captured["scorer_space_step_guard_backtracking_steps"] == 5
    assert captured["scorer_space_step_guard_backtracking_shrink"] == pytest.approx(0.45)
    assert json.loads(capsys.readouterr().out)["ready_for_exact_eval_dispatch"] is False
    report_payload = json.loads(
        (tmp_path / "snerv_direct_live" / "compact_renderer_mlx_spine_runner_report.json").read_text(encoding="utf-8")
    )
    _assert_compact_runner_rerun_provenance(
        report_payload,
        family="snerv",
        output_dir=tmp_path / "snerv_direct_live",
    )


def test_post_export_materializer_executor_runs_output_scoped_queue(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "queue_artifact.json"
    queue_path = tmp_path / "experiment_queue.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_post_export_materializer",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "exp",
                "lane_id": "unit",
                "metadata": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "write_artifact",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({artifact.as_posix()!r}).write_text("
                                "json.dumps({'schema':'post-export-unit.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": artifact.as_posix(),
                                "key": "schema",
                                "equals": "post-export-unit.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    queue_path.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    queue_root = tmp_path / "queue_root"
    plan = {
        "schema": "compact_carrier_post_export_materializer_plan.v1",
        "compiled": True,
        "queue_id": "unit_post_export_materializer",
        "experiment_queue_path": queue_path.as_posix(),
        "queue_output_dir": queue_root.as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    result = runner_mod._execute_carrier_post_export_materializer_plan(
        plan=plan,
        requested=True,
        max_steps=1,
        max_parallel=1,
        repo_root=REPO_ROOT,
    )

    assert result["schema"] == "compact_carrier_post_export_materializer_execution.v1"
    assert result["requested"] is True
    assert result["executed"] is True
    assert result["blockers"] == []
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["worker"]["success_count"] == 1
    assert result["worker"]["failure_count"] == 0
    assert result["worker"]["steps_started"] == 1
    assert Path(result["state_path"]).is_file()
    assert Path(result["log_root"]).is_dir()
    assert Path(result["execution_path"]).is_file()
    assert artifact.is_file()
    persisted = json.loads(Path(result["execution_path"]).read_text(encoding="utf-8"))
    assert persisted["state_path"] == result["state_path"]
    assert persisted["worker"]["success_count"] == 1


def test_post_export_materializer_executor_can_focus_one_chain(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    other = tmp_path / "other.json"
    queue_path = tmp_path / "experiment_queue.json"
    queue_root = tmp_path / "queue_root"
    handoff_dir = queue_root / "archive_zip_repack_v1_exact_eval_handoff"
    source_queue = handoff_dir / "source_queue.json"
    harvest_report = handoff_dir / "harvest_report.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_post_export_chain_focus",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "archive_zip_repack_chain",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({first.as_posix()!r}).write_text("
                                "json.dumps({'schema':'first.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": first.as_posix(),
                                "key": "schema",
                                "equals": "first.v1",
                            }
                        ],
                    },
                    {
                        "id": "harvest_materializer_chains",
                        "requires": ["materialize_local_proof_chain"],
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({handoff_dir.as_posix()!r}).mkdir("
                                "parents=True, exist_ok=True); "
                                f"pathlib.Path({second.as_posix()!r}).write_text("
                                "json.dumps({'schema':'second.v1'}), "
                                "encoding='utf-8'); "
                                f"pathlib.Path({source_queue.as_posix()!r}).write_text("
                                "json.dumps({'schema':'optimizer_candidate_queue_v1'}), "
                                "encoding='utf-8'); "
                                f"pathlib.Path({harvest_report.as_posix()!r}).write_text("
                                "json.dumps({'schema':'materializer_chain_harvest_report.v1', "
                                "'score_claim': False, "
                                "'ready_for_exact_eval_dispatch': False}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": second.as_posix(),
                                "key": "schema",
                                "equals": "second.v1",
                            }
                        ],
                    },
                ],
            },
            {
                "id": "packet_member_zip_header_elide_chain",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({other.as_posix()!r}).write_text("
                                "json.dumps({'schema':'other.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": other.as_posix(),
                                "key": "schema",
                                "equals": "other.v1",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    queue_path.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    plan = {
        "schema": "compact_carrier_post_export_materializer_plan.v1",
        "compiled": True,
        "queue_id": "unit_post_export_chain_focus",
        "experiment_queue_path": queue_path.as_posix(),
        "experiment_queue_state_path": (queue_root / "experiment_queue.sqlite").as_posix(),
        "queue_output_dir": queue_root.as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    result = runner_mod._execute_carrier_post_export_materializer_plan(
        plan=plan,
        requested=True,
        max_steps=2,
        max_parallel=1,
        max_experiments=1,
        repo_root=REPO_ROOT,
    )

    assert result["blockers"] == []
    assert result["max_experiments"] == 1
    assert result["worker"]["steps_started"] == 2
    assert result["worker"]["started_experiment_ids"] == ["archive_zip_repack_chain"]
    assert [row["ready_step"]["step_id"] for row in result["worker"]["step_results"]] == [
        "materialize_local_proof_chain",
        "harvest_materializer_chains",
    ]
    assert first.is_file()
    assert second.is_file()
    assert not other.exists()
    handoff_summary = result["handoff_summary"]
    assert handoff_summary["handoff_count"] == 1
    handoff = handoff_summary["rows"][0]
    assert handoff["target_kind"] == "archive_zip_repack_v1"
    assert handoff["source_queue_path"] == source_queue.as_posix()
    assert handoff["harvest_report_path"] == harvest_report.as_posix()
    assert handoff["source_queue_schema"] == "optimizer_candidate_queue_v1"
    assert handoff["harvest_report_schema"] == "materializer_chain_harvest_report.v1"
    assert handoff["ready_for_exact_eval_dispatch"] is False


def test_post_export_materializer_sweep_feedback_keeps_byte_saving_atoms(
    tmp_path: Path,
) -> None:
    positive = tmp_path / "archive_zip_repack_v1" / "sweep.json"
    positive.parent.mkdir(parents=True)
    positive.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "target_kind": "archive_zip_repack_v1",
                "observation_count": 1,
                "rate_positive_count": 1,
                "rate_nonpositive_count": 0,
                "max_saved_bytes": 979,
                "total_positive_saved_bytes": 979,
                "planner_feedback": {
                    "recommended_acquisition_rule": ("rank_rate_positive_materializer_after_inflate_parity")
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    zero = tmp_path / "packet_member_zip_header_elide_v1" / "sweep.json"
    zero.parent.mkdir(parents=True)
    zero.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "target_kind": "packet_member_zip_header_elide_v1",
                "observation_count": 1,
                "rate_positive_count": 0,
                "rate_nonpositive_count": 1,
                "max_saved_bytes": 0,
                "total_positive_saved_bytes": 0,
                "planner_feedback": {
                    "recommended_acquisition_rule": (
                        "demote_packet_member_zip_header_elide_v1_for_matching_archive_class"
                    )
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    summary = runner_mod._post_export_materializer_sweep_feedback_summary(tmp_path)

    assert summary["schema"] == ("compact_carrier_post_export_sweep_feedback_summary.v1")
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["byte_saving_sweep_count"] == 1
    assert summary["zero_save_sweep_count"] == 1
    assert summary["total_positive_saved_bytes"] == 979
    assert summary["retain_target_kinds"] == ["archive_zip_repack_v1"]
    assert summary["zero_save_target_kinds"] == ["packet_member_zip_header_elide_v1"]
    assert summary["recommended_global_rule"] == ("retain_and_order_byte_saving_atoms_before_demoting_full_lane")
    dispositions = {row["target_kind"]: row["full_stack_chain_disposition"] for row in summary["rows"]}
    assert dispositions["archive_zip_repack_v1"] == ("retain_byte_saving_atom_for_ordered_chain_solver")
    assert dispositions["packet_member_zip_header_elide_v1"] == ("demote_only_matching_zero_save_archive_class")


def test_recon_pixel_weight_loader_records_file_custody(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_weight.npz"
    np.savez(
        weight_path,
        weight=np.ones((384, 512, 1), dtype=np.float32),
    )

    weight, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        normalize="mean",
    )

    assert weight.shape == (384, 512, 1)
    assert metadata["schema"] == "compact_recon_pixel_weight.v1"
    assert metadata["enabled"] is True
    assert metadata["source_kind"] == "file"
    assert metadata["path"] == weight_path.as_posix()
    assert metadata["sha256"] == runner_mod._sha256_file(weight_path)
    assert metadata["npz_key"] == "weight"
    assert metadata["normalize"] == "mean"
    assert metadata["scorer_terms"] == {
        "p18_segnet": "caller_supplied",
        "p19_posenet": "caller_supplied",
    }
    assert metadata["stats"]["shape"] == [384, 512, 1]
    assert metadata["stats"]["nonzero_fraction"] == 1.0
    assert metadata["producer_manifest"]["status"] == ("not_found_unverified_manual_or_legacy_weight")
    assert metadata["producer_manifest"]["consumption_certified"] is False
    assert metadata["authority"] == "false_macos_mlx_research_signal"


def test_recon_pixel_weight_loader_accepts_pair_frame_map(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_pair_frame_weight.npz"
    weight = np.ones((3, 2, 384, 512, 1), dtype=np.float32)
    weight[1, 0, :, :, 0] = 2.0
    np.savez_compressed(weight_path, weight=weight)

    loaded, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        expected_pairs=3,
        normalize="mean",
    )

    assert loaded.shape == (3, 2, 384, 512, 1)
    assert metadata["schema"] == "compact_recon_pixel_weight.v1"
    assert metadata["enabled"] is True
    assert metadata["source_kind"] == "file"
    assert metadata["expected_pairs"] == 3
    assert metadata["stats"]["shape"] == [3, 2, 384, 512, 1]
    assert metadata["sha256"] == runner_mod._sha256_file(weight_path)


def test_recon_pixel_weight_loader_carries_verified_gradient_manifest(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((2, 2, 384, 512, 1), dtype=np.float32),
    )
    health = {
        "schema": "joint_recon_pixel_weight_gradient_health.v1",
        "surface_generation_backend": "torch_exact_cpu_scorer_vjp.v1",
        "component_count": 14,
        "components_with_nonfinite": 0,
        "total_nonfinite_values": 0,
        "sanitized_components": [],
        "status": "pass_finite",
        "consumption_recommended": True,
    }
    (tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json").write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": runner_mod._sha256_file(weight_path),
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "surface_generation_backend": "torch_exact_cpu_scorer_vjp.v1",
                    "gradient_health": health,
                    "blockers": [],
                    "training_consumption_recommended": True,
                },
            }
        ),
        encoding="utf-8",
    )

    _, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        expected_pairs=2,
        normalize="mean",
    )

    producer = metadata["producer_manifest"]
    assert producer["status"] == "verified_finite_gradient_manifest"
    assert producer["consumption_certified"] is True
    assert producer["weight_sha256"] == runner_mod._sha256_file(weight_path)
    assert producer["gradient_health"] == health


def test_recon_pixel_weight_loader_carries_hard_region_manifest_false_authority(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "receiver_replay_hard_region_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((1, 2, 384, 512, 1), dtype=np.float32),
    )
    source_identity = {
        "candidate_argmax_sha256": "c" * 64,
        "reference_argmax_sha256": "r" * 64,
    }
    (tmp_path / "receiver_replay_hard_region_recon_pixel_weight_manifest.json").write_text(
        json.dumps(
            {
                "schema": "receiver_replay_hard_region_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": runner_mod._sha256_file(weight_path),
                "weight_array_sha256": "a" * 64,
                "metadata": {
                    "schema": "receiver_replay_hard_region_recon_pixel_weight.v1",
                    "source_report_schema": "receiver_replay_scorer_hard_regions.v1",
                    "source_report_label": "guarded_hinerv_receiver_replay",
                    "evidence_grade": "local_receiver_replay_scorer_hard_region_false_authority",
                    "evidence_tag": "[local receiver-replay hard-region signal]",
                    "target_frame_index": 1,
                    "applied_hard_region_records": 5,
                    "applied_component_bboxes": 4,
                    "source_report_identity": source_identity,
                },
                "consumption": {
                    "training_arg": "--recon-pixel-weight-path",
                    "training_consumption_recommended": True,
                    "auto_discovery_eligible": False,
                    "reason": "explicit_hard_region_smoke",
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    _, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        expected_pairs=1,
        normalize="mean",
    )

    producer = metadata["producer_manifest"]
    assert producer["status"] == "receiver_replay_hard_region_manifest_false_authority"
    assert producer["consumption_certified"] is False
    assert producer["training_consumption_recommended"] is True
    assert producer["auto_discovery_eligible"] is False
    assert producer["applied_hard_region_records"] == 5
    assert producer["target_frame_index"] == 1
    assert producer["source_report_identity"] == source_identity


def test_recon_pixel_weight_loader_refuses_stale_manifest_without_gradient_health(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((2, 2, 384, 512, 1), dtype=np.float32),
    )
    (tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json").write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": runner_mod._sha256_file(weight_path),
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            expected_pairs=2,
            normalize="mean",
        )

    assert "missing gradient_health" in str(exc.value)


def test_recon_pixel_weight_loader_rejects_pair_count_mismatch(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_pair_frame_weight.npy"
    np.save(weight_path, np.ones((2, 2, 384, 512, 1), dtype=np.float32))

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            expected_pairs=3,
            normalize="mean",
        )

    assert "pair count" in str(exc.value)


def test_recon_pixel_weight_loader_fails_closed_on_bad_shape(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "wrong_shape.npy"
    np.save(weight_path, np.ones((192, 256), dtype=np.float32))

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            normalize="mean",
        )

    assert "spatial shape" in str(exc.value)


def test_hinerv_full_coverage_execute_runs_local_cpu_replay_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    staged_calls: list[dict[str, object]] = []

    def fake_stage_local_replay_submission(**kwargs):
        staged_calls.append(kwargs)
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        return FakeReplaySummary()

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_post_export_receiver_cache_quality",
        lambda **_kwargs: {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": str(tmp_path / "receiver_quality.json"),
            "quality_gate_passed": True,
            "quality_gate": {"verdict": "passed", "stats": {}, "blockers": []},
            "segnet_argmax_probe": {
                "fit_gate_passed": True,
                "segnet_argmax_disagreement_rate": 0.0,
                "candidate_argmax_histogram": [1, 1, 1, 1, 1],
                "reference_argmax_histogram": [1, 1, 1, 1, 1],
            },
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    mlx_profile_path = tmp_path / "full_video_mlx_profile.json"
    mlx_profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 0.1},
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(mlx_profile_path,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=7,
        optimizer_kind="lion",
        upstream_dir=tmp_path / "canonical_upstream",
        repo_root=REPO_ROOT,
    )

    assert Path(captured_train_kwargs["scorer_upstream_dir"]) == (tmp_path / "canonical_upstream")
    assert captured_train_kwargs["mlx_prefilter_scorer_batch_pairs"] == 4
    assert captured_train_kwargs["mlx_prefilter_progress_every"] == 7
    assert captured_train_kwargs["optimizer_kind"] == "lion"
    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "lion"
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert optimizer_controls["weight_decay_defaulted"] is True
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(1.0)
    optimizer_policy = captured_train_kwargs["hi_nerv_optimizer_policy"]
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert out["score_aware_training"]["optimizer_policy"] == optimizer_policy
    assert out["score_aware_training"]["optimizer_controls"] == optimizer_controls
    assert out["score_aware_training"]["local_mlx_prefilter"] == {
        "schema": "compact_hi_nerv_local_mlx_prefilter_config.v1",
        "scorer_device": "cpu",
        "scorer_batch_pairs": 4,
        "progress_every": 7,
        "singleton_required_for_local_cpu_replay_unlock": True,
        "gpu_profiles_are_prefilter_only": False,
        "batched_profiles_are_prefilter_only": True,
        "authority": "macos_mlx_research_signal_false_authority",
    }
    assert staged_calls
    assert replay_calls
    assert out["local_cpu_replay_gate"]["executed"] is True
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is True
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is True
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_mlx_prefilter_passed"] is True
    assert out["mlx_prefilter_coverage"]["blockers"] == []
    assert out["local_cpu_replay_summary"]["axis_tag"] == "[macOS-CPU advisory]"
    assert out["local_cpu_replay_summary"]["score_claim"] is False
    assert out["local_cpu_replay_summary_paths"]
    assert Path(out["local_cpu_replay_summary_paths"][0]).is_file()
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert post_export["queue_launch_executed"] is False
    assert post_export["experiment_count"] > 0
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert post_export["archive_record"]["source_runtime_dir"].endswith("/hi_nerv_mlx_training/submission")
    assert post_export["archive_record"]["source_inflate_sh_path"].endswith(
        "/hi_nerv_mlx_training/submission/inflate.sh"
    )
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith("/hi_nerv_mlx_training/submission")
    assert first_context["packet_member_merge_source_runtime_dir"].endswith("/hi_nerv_mlx_training/submission")
    queue = json.loads(Path(post_export["experiment_queue_path"]).read_text())
    assert queue["schema"] == "experiment_queue.v1"
    assert post_export["experiment_queue_state_path"].endswith("/experiment_queue.sqlite")
    harvest_step = queue["experiments"][0]["steps"][1]
    state_arg_index = harvest_step["command"].index("--state") + 1
    assert harvest_step["command"][state_arg_index] == post_export["experiment_queue_state_path"]
    assert post_export["score_claim"] is False
    assert post_export["ready_for_exact_eval_dispatch"] is False
    post_export_execution = out["post_export_materializer_execution"]
    assert post_export_execution["requested"] is False
    assert post_export_execution["executed"] is False
    assert post_export_execution["mode"] == "compile_only_execution_not_requested"
    assert Path(post_export_execution["execution_path"]).is_file()
    assert "local_cpu_replay_not_executed" not in out["blockers"]
    assert "local_cpu_replay_not_run_partial_pair_coverage" not in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]


def test_hinerv_long_campaign_refuses_when_pr95_prelaunch_gate_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("long campaign should refuse before MLX training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_refusal",
        num_pairs=600,
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_pr95_binding_prelaunch_refused"
    assert out["training_executed"] is False
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "hi_nerv_modelsize_archive_budget_missing" in out["blockers"]
    gate = out["candidate_curriculum_plan"]["long_campaign_prelaunch_gate"]
    assert gate["launch_allowed"] is False
    assert "receiver_proof" in gate["post_run_requirements_excluded"]
    assert out["candidate_feedback"]["row"]["long_campaign_prelaunch_launch_allowed"] is False
    assert Path(out["report_path"]).is_file()


def test_hinerv_long_native_optimizer_probe_bypasses_pr95_prelaunch_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_native_adamw_long_probe",
        num_pairs=2,
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        optimizer_kind="adamw",
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_hi_nerv_mlx_scoreaware_and_exported"
    assert out["training_executed"] is True
    assert captured_train_kwargs["optimizer_kind"] == "adamw"
    policy = captured_train_kwargs["hi_nerv_optimizer_policy"]
    assert policy["resolved_policy"] == "native_optimizer"
    assert policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert "pr95_long_campaign_prelaunch_gate_failed" not in out["blockers"]


def test_hinerv_trained_archive_byte_oracle_writes_receiver_closed_ladder(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"hi-nerv-byte-closed-candidate")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_tiny",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 40_000,
        },
        hard_byte_ceilings=(178_000,),
        num_pairs=2,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[macOS-CPU advisory]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": False},
        repo_root=REPO_ROOT,
    )

    oracle_path = Path(oracle["path"])
    ladder_path = Path(oracle["receiver_closed_modelsize_ladder_path"])
    assert oracle_path.is_file()
    assert ladder_path.is_file()
    assert oracle["schema"] == "hi_nerv_trained_archive_byte_oracle.v1"
    assert oracle["row"]["receiver_proof_passed"] is True
    assert oracle["row"]["measured_archive_bytes"] == archive.stat().st_size
    assert oracle["row"]["archive_sha256"] == archive_sha
    assert oracle["measured_byte_cap_report"]["archive_zip_bytes"] == archive.stat().st_size
    assert (
        oracle["measured_byte_cap_report"]["delta_bytes_vs_tightest_hard_byte_ceiling"]
        == archive.stat().st_size - 178_000
    )
    assert oracle["row"]["measured_archive_bytes_under_tightest_hard_ceiling"] is True
    assert "hi_nerv_trained_archive_byte_oracle_partial_pair_scope" in oracle["blockers"]
    assert "hi_nerv_local_cpu_replay_not_contest_auth_axis" in oracle["blockers"]
    assert oracle["receiver_closed_modelsize_ladder"]["schema"] == ("nerv_receiver_closed_modelsize_ladder.v1")
    assert oracle["score_claim"] is False
    assert oracle["ready_for_exact_eval_dispatch"] is False


def test_hinerv_trained_archive_byte_oracle_blocks_measured_over_cap(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"x" * 64)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_overcap",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 32,
            "nominal_total_payload_bytes": 16,
        },
        hard_byte_ceilings=(32,),
        num_pairs=600,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[contest-CPU]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": True},
        repo_root=REPO_ROOT,
    )

    report = oracle["measured_byte_cap_report"]
    assert report["archive_zip_bytes"] == 64
    assert report["delta_bytes_vs_tightest_hard_byte_ceiling"] == 32
    assert report["under_tightest_hard_byte_ceiling"] is False
    assert "measured_archive_bytes_exceed_tightest_hard_ceiling" in report["blockers"]
    assert "measured_archive_bytes_exceed_tightest_hard_ceiling" in oracle["blockers"]
    assert oracle["feedback_ready"] is False


def test_hinerv_trained_archive_byte_oracle_blocks_receiver_class_collapse(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"x" * 64)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_collapsed",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 16,
        },
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[contest-CPU]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": True},
        post_export_receiver_cache_quality_summary={
            "schema": "hi_nerv_receiver_cache_quality_summary.v1",
            "quality_gate_passed": False,
            "segnet_candidate_occupied_class_fraction": 0.2,
            "segnet_reference_occupied_class_fraction": 1.0,
            "blockers": ["hi_nerv_receiver_cache_segnet_argmax_class_collapse"],
        },
        repo_root=REPO_ROOT,
    )

    assert oracle["measured_archive_bytes"] == archive.stat().st_size
    assert oracle["row"]["post_export_receiver_cache_quality_feedback_ready"] is False
    assert oracle["feedback_ready"] is False
    assert "hi_nerv_trained_archive_byte_oracle_receiver_argmax_class_collapse" in oracle["blockers"]
    assert "hi_nerv_receiver_cache_segnet_argmax_class_collapse" in oracle["blockers"]


def test_hinerv_trained_archive_byte_oracle_blocks_numeric_receiver_class_collapse(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"x" * 64)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_numeric_collapsed",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 16,
        },
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[contest-CPU]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": True},
        post_export_receiver_cache_quality_summary={
            "schema": "hi_nerv_receiver_cache_quality_summary.v1",
            "quality_gate_passed": True,
            "segnet_candidate_occupied_class_fraction": 0.2,
            "segnet_reference_occupied_class_fraction": 1.0,
            "blockers": [],
        },
        repo_root=REPO_ROOT,
    )

    assert oracle["row"]["post_export_receiver_cache_quality_feedback_ready"] is False
    assert oracle["feedback_ready"] is False
    assert "hi_nerv_trained_archive_byte_oracle_receiver_argmax_class_collapse_numeric" in oracle["blockers"]


def test_hinerv_long_mlx_training_readiness_signals_summarize_actuators() -> None:
    signals = runner_mod._hi_nerv_long_mlx_training_readiness_signals(
        config_gate={
            "frontier_targeting": True,
            "real_segnet_teacher_attached": True,
            "real_posenet_teacher_attached": True,
            "pose_direct_live_attached": True,
        },
        training_telemetry_contract={
            "passed": True,
            "telemetry_path": "/tmp/unit_telemetry.jsonl",
            "row_count": 4,
            "archive_byte_dual_lambda_active_observed": True,
            "archive_byte_dual_weight_applied_observed": True,
            "section_rate_metric_observed": True,
            "section_byte_dual_lambda_active_observed": True,
            "section_byte_dual_weight_applied_observed": True,
            "posenet_direct_live_score_term_metric_observed": True,
            "segnet_direct_live_argmax_metric_observed": True,
            "scorer_input_contrast_floor_metric_observed": True,
            "posenet_temporal_signal_floor_metric_observed": True,
        },
        trained_archive_byte_oracle={
            "feedback_ready": True,
            "measured_archive_bytes": 123_456,
            "archive_sha256": "a" * 64,
            "measured_byte_cap_report": {
                "under_tightest_hard_byte_ceiling": True,
                "delta_bytes_vs_tightest_hard_byte_ceiling": -54_321,
            },
            "row": {"archive_bytes": 123_456},
            "blockers": [],
        },
        decoder_weight_saliency_artifact={
            "written": True,
            "row_count": 7,
            "path": "/tmp/saliency.json",
        },
        decoder_weight_waterfill_from_trained_ladder={
            "active": True,
            "candidate_plan_count": 2,
            "path": "/tmp/waterfill.json",
            "blockers": [],
        },
        short_scorer_smoke_readiness={
            "ready_for_long_run": True,
            "short_scorer_teacher_smoke_ready": True,
            "report_path": "/tmp/readiness.json",
            "actionable_blockers": [],
        },
        post_export_receiver_cache_quality_summary={
            "quality_gate_passed": True,
            "runner_effective_receiver_cache_quality_max_pairs": 16,
            "runner_quality_floor_receiver_cache_quality_min_pairs": 16,
            "segnet_argmax_disagreement_rate": 0.02,
            "mlx_scorer_response_probe_passed": True,
        },
        optimizer_policy={
            "effective_optimizer_label": "pr95_8stage_muon_adamw_every_stage",
            "pr95_faithful_curriculum_enabled": True,
            "native_optimizer_active": False,
            "optimizer_kind": "pact_muon_adamw",
        },
        optimizer_controls={"weight_decay_effective": 1.0e-4},
    )

    assert signals["schema"] == ("compact_hi_nerv_long_mlx_training_readiness_signals.v1")
    assert signals["ready_for_controlled_long_mlx_training"] is True
    assert signals["ready_for_promotion_replay"] is True
    assert signals["control_blockers"] == []
    assert signals["archive"]["measured_archive_bytes"] == 123_456
    assert signals["telemetry"]["section_byte_dual_lambda_active_observed"] is True
    assert signals["decoder_weight_saliency"]["row_count"] == 7
    assert signals["decoder_weight_waterfill"]["candidate_plan_count"] == 2
    assert signals["receiver_cache_quality"]["effective_max_pairs"] == 16
    assert signals["score_claim"] is False
    assert signals["ready_for_exact_eval_dispatch"] is False


def test_measured_archive_byte_cap_report_allows_selected_looser_ceiling() -> None:
    report = runner_mod._measured_archive_byte_cap_report(
        archive_bytes=240_000,
        archive_sha256="a" * 64,
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        candidate={
            "candidate_id": "hinerv_285k_candidate",
            "hard_byte_ceiling": 285_000,
        },
    )

    assert report["tightest_hard_byte_ceiling"] == 178_000
    assert report["selected_candidate_hard_byte_ceiling"] == 285_000
    assert report["under_tightest_hard_byte_ceiling"] is False
    assert report["under_selected_candidate_hard_byte_ceiling"] is True
    assert report["tightest_hard_ceiling_is_blocking"] is False
    assert report["blockers"] == []


def test_hinerv_refuses_unscored_launch_but_consumes_modelsize_ladder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("unscored HiNeRV launch must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    ladder = tmp_path / "hinerv_receiver_closed_ladder.json"
    ladder.write_text(
        json.dumps(
            {
                "schema": "nerv_receiver_closed_modelsize_ladder.v1",
                "carrier_id": "hi_nerv",
                "modelsize_budget_rows": [
                    {
                        "row_id": "tiny",
                        "archive_bytes": 40_000,
                        "archive_sha256": "a" * 64,
                        "nonrate_score": 95.0,
                        "modelsize_mparams": 0.02,
                        "fc_dim": 8,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "receiver_archive_replay_verified": True,
                        "receiver_proof_path": "proofs/tiny.json",
                        "receiver_proof_sha256": "b" * 64,
                        "axis_tag": "[macOS-CPU advisory]",
                        "num_pairs": 600,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    {
                        "row_id": "small",
                        "archive_bytes": 80_000,
                        "archive_sha256": "c" * 64,
                        "nonrate_score": 80.0,
                        "modelsize_mparams": 0.04,
                        "fc_dim": 16,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "receiver_archive_replay_verified": True,
                        "receiver_proof_path": "proofs/small.json",
                        "receiver_proof_sha256": "d" * 64,
                        "axis_tag": "[macOS-CPU advisory]",
                        "num_pairs": 600,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(ladder,),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert "hi_nerv_real_segnet_teacher_missing" in out["blockers"]
    assert "hi_nerv_real_posenet_teacher_missing" in out["blockers"]
    assert out["modelsize_budget_evidence"]["row_count"] == 2
    assert out["modelsize_budget_evidence"]["sources"][0]["rows_added"] == 2
    plan = out["score_aware_carrier_training_plan"]
    assert plan["planner_action"] == "run_receiver_closed_modelsize_ladder_before_score_aware_training"
    assert plan["modelsize_budget_receiver_closed_ready"] is False
    assert "modelsize_budget:source_bound_modelsize_or_fc_dim_missing" in plan["dispatch_blockers"]
    assert "modelsize_budget:modelsize_control_contract_missing_or_invalid" in plan["dispatch_blockers"]
    assert plan["evidence_summary"]["receiver_closed_selected_modelsize_archive_bytes"] == 80_000
    assert out["score_aware_training_config_gate"]["launch_allowed"] is False
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["report_path"]).is_file()


def test_hinerv_refusal_filters_modelsize_rows_without_nonrate_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("invalid modelsize evidence must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    bad_ladder = tmp_path / "hinerv_waterfill_without_nonrate.json"
    bad_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "archive_bytes": 134_938,
                        "archive_sha256": "a" * 64,
                        "waterfill_summary": {"group_count": 24},
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_bad_modelsize_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(bad_ladder,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["rows_seen"] == 1
    assert source["rows_added"] == 0
    assert source["rows_rejected"] == 1
    assert "modelsize_budget_json_rows_rejected" in source["blockers"]
    rejected = set(source["rejected_rows"][0]["blockers"])
    assert {
        "receiver_closed_modelsize_ladder_schema_required",
        "receiver_closed_byte_proof_missing",
        "receiver_proof_path_missing",
        "receiver_proof_sha256_missing_or_invalid",
        "receiver_proof_axis_tag_missing",
        "receiver_proof_full_sample_count_missing",
        "source_bound_modelsize_or_fc_dim_missing",
        "modelsize_budget_row_missing_nonrate_score",
    }.issubset(rejected)
    assert "hi_nerv_real_segnet_teacher_missing" in out["blockers"]
    assert "hi_nerv_real_posenet_teacher_missing" in out["blockers"]
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_hinerv_refusal_rejects_raw_modelsize_budget_selected_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("raw planning artifact must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    raw_budget = tmp_path / "nerv_modelsize_budget.json"
    raw_budget.write_text(
        json.dumps(
            {
                "schema": "nerv_modelsize_budget.v1",
                "family": "hi_nerv",
                "selected_candidates": [
                    {
                        "candidate_id": "hinerv_np600_ld4_ed8_dc8_int4_mixed_ceil178000",
                        "archive_bytes": 72_000,
                        "nonrate_score": 0.21,
                        "modelsize_mparams": 0.032,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "selected_candidate_count": 1,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_raw_modelsize_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(raw_budget,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["source_schema"] == "nerv_modelsize_budget.v1"
    assert source["authority"] == ("planning_artifact_only_not_receiver_closed_ladder_evidence")
    assert source["rows_seen"] == 1
    assert source["rows_added"] == 0
    assert source["rows_rejected"] == 1
    assert source["score_claim"] is False
    assert source["promotion_eligible"] is False
    assert source["ready_for_exact_eval_dispatch"] is False
    assert "raw_nerv_modelsize_budget_artifact_not_receiver_closed_ladder" in source["blockers"]
    assert "receiver_closed_modelsize_ladder_schema_required" in source["blockers"]
    assert source["rejected_rows"][0]["blockers"] == [
        "selected_candidates_are_planning_rows_not_receiver_closed_ladder",
        "receiver_closed_byte_proof_missing",
        "measured_receiver_archive_bytes_missing",
    ]
    plan = out["score_aware_carrier_training_plan"]
    assert plan["modelsize_budget_receiver_closed_ready"] is False
    assert "receiver_closed_modelsize_budget_ladder_missing" in plan["dispatch_blockers"]
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_hinerv_refusal_rejects_canonical_score_only_modelsize_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("score-only modelsize evidence must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    score_only_ladder = tmp_path / "hinerv_score_only_ladder.json"
    score_only_ladder.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_modelsize_budget_plan.v2",
                "modelsize_budget_rows": [
                    {
                        "row_id": "score_only",
                        "archive_bytes": 120_000,
                        "canonical_score": 0.5,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_score_only_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(score_only_ladder,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["rows_seen"] == 1
    assert source["rows_rejected"] == 1
    rejected = set(source["rejected_rows"][0]["blockers"])
    assert {
        "receiver_closed_modelsize_ladder_schema_required",
        "modelsize_budget_row_missing_nonrate_score",
        "receiver_proof_path_missing",
        "receiver_proof_sha256_missing_or_invalid",
        "archive_sha256_missing_or_invalid",
        "receiver_proof_axis_tag_missing",
        "receiver_proof_full_sample_count_missing",
        "source_bound_modelsize_or_fc_dim_missing",
    }.issubset(rejected)


def test_hinerv_auto_mlx_prefilter_profile_unlocks_local_cpu_replay_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (out / "local_mlx_prefilter_profile.json").write_text(
            json.dumps(
                {
                    "schema": "hprc_mlx_component_neutralization_profile.v1",
                    "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                    "max_pairs": 600,
                    "num_pairs": 600,
                    "n_samples": 600,
                    "scorer_batch_pairs": 1,
                    "scope_status": {"full_video": "executed"},
                    "score_components": {"canonical_score": 0.1},
                    "mlx_response_summary": {
                        "batch_pairs": 1,
                        "max_pairs": 600,
                        "n_samples": 600,
                    },
                    "section_value_rows": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_stage_local_replay_submission(**kwargs):
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        return FakeReplaySummary()

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_post_export_receiver_cache_quality",
        lambda **_kwargs: {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": str(tmp_path / "receiver_quality.json"),
            "quality_gate_passed": True,
            "quality_gate": {"verdict": "passed", "stats": {}, "blockers": []},
            "segnet_argmax_probe": {
                "fit_gate_passed": True,
                "segnet_argmax_disagreement_rate": 0.0,
                "candidate_argmax_histogram": [1, 1, 1, 1, 1],
                "reference_argmax_histogram": [1, 1, 1, 1, 1],
            },
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls
    assert out["auto_mlx_prefilter_profile_path"].endswith("local_mlx_prefilter_profile.json")
    assert out["mlx_profile_paths"] == [out["auto_mlx_prefilter_profile_path"]]
    assert out["local_cpu_replay_gate"]["executed"] is True
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is True
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in out["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" not in out["blockers"]
    assert (
        "hi_nerv_full_video_local_prefilter_missing"
        not in out["candidate_feedback"]["row"]["pr95_stack_binding_blockers"]
    )
    assert (
        "hi_nerv_local_cpu_replay_gate_missing" not in out["candidate_feedback"]["row"]["pr95_stack_binding_blockers"]
    )


def test_hinerv_execute_emits_trained_archive_byte_oracle_feedback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = {
        "candidate_id": "unit_hinerv_candidate",
        "family": "hi_nerv",
        "num_pairs": 600,
        "latent_dim": 4,
        "embed_dim": 8,
        "decoder_channel": 8,
        "decoder_codec": "int4_mixed",
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 70_000,
        "modelsize_mparams": 0.031,
        "use_hierarchical_feature_grid": True,
        "use_convnext_blocks": True,
    }

    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        archive_sha = runner_mod._sha256_file(archive)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        _write_hinerv_receiver_proof(
            out / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json",
            archive_bytes=archive.stat().st_size,
            archive_sha256=archive_sha,
        )
        (out / "local_mlx_prefilter_profile.json").write_text(
            json.dumps(
                {
                    "schema": "hprc_mlx_component_neutralization_profile.v1",
                    "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                    "max_pairs": 600,
                    "num_pairs": 600,
                    "n_samples": 600,
                    "scorer_batch_pairs": 1,
                    "scope_status": {"full_video": "executed"},
                    "score_components": {"canonical_score": 0.1},
                    "mlx_response_summary": {
                        "batch_pairs": 1,
                        "max_pairs": 600,
                        "n_samples": 600,
                    },
                    "section_value_rows": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        }

    def fake_stage_local_replay_submission(**kwargs):
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_post_export_receiver_cache_quality",
        lambda **_kwargs: {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": str(tmp_path / "byte_oracle_receiver_quality.json"),
            "quality_gate_passed": True,
            "quality_gate": {"verdict": "passed", "stats": {}, "blockers": []},
            "segnet_argmax_probe": {
                "fit_gate_passed": True,
                "segnet_argmax_disagreement_rate": 0.0,
                "candidate_argmax_histogram": [1, 1, 1, 1, 1],
                "reference_argmax_histogram": [1, 1, 1, 1, 1],
            },
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        lambda **_kwargs: FakeReplaySummary(),
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_byte_oracle",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        modelsize_candidate=candidate,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    oracle = out["trained_archive_byte_oracle"]
    assert oracle["schema"] == "hi_nerv_trained_archive_byte_oracle.v1"
    assert oracle["measured_archive_bytes"] == out["archive_bytes"]
    assert oracle["row"]["receiver_proof_passed"] is True
    assert oracle["row"]["archive_bytes"] == out["archive_bytes"]
    assert oracle["row"]["nominal_total_payload_bytes"] == 70_000
    assert Path(oracle["path"]).is_file()
    assert Path(oracle["receiver_closed_modelsize_ladder_path"]).is_file()
    score_training = out["score_aware_training"]
    assert score_training["archive_path"] == out["archive_path"]
    assert score_training["archive_bytes"] == out["archive_bytes"]
    assert score_training["archive_sha256"] == out["archive_sha256"]
    assert score_training["archive_resolution"]["archive_path"] == out["archive_path"]
    byte_feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert byte_feedback["byte_feedback_source"] == ("hi_nerv_trained_archive_byte_oracle")
    assert byte_feedback["measured_archive_bytes"] == out["archive_bytes"]
    assert byte_feedback["trained_archive_byte_oracle_path"] == oracle["path"]
    assert out["candidate_feedback"]["row"]["measured_archive_bytes"] == out["archive_bytes"]
    readiness = out["hi_nerv_long_mlx_training_readiness_signals"]
    assert readiness["schema"] == ("compact_hi_nerv_long_mlx_training_readiness_signals.v1")
    assert readiness["archive"]["measured_archive_bytes"] == out["archive_bytes"]
    assert readiness["teachers"]["real_segnet_teacher_attached"] is True
    assert readiness["teachers"]["real_posenet_teacher_attached"] is True
    assert out["score_aware_training"]["long_mlx_training_readiness_signals"] == (readiness)
    assert readiness["score_claim"] is False
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert "hi_nerv_local_cpu_replay_not_contest_auth_axis" in out["blockers"]


def test_hinerv_auto_joint_recon_weight_flows_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight_path, _manifest_path = _write_verified_joint_recon_weight(
        tmp_path / "repo",
        pairs=2,
        name="weight_2",
    )
    discovery = {
        "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
        "status": "selected_verified_joint_p18_p19_weight",
        "num_pairs": 2,
        "selected_weight_path": weight_path.as_posix(),
        "selected_weight_sha256": runner_mod._sha256_file(weight_path),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    captured: dict[str, object] = {}

    def fake_discover(**kwargs):
        assert kwargs["num_pairs"] == 2
        return weight_path, discovery

    def fake_train(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "pose_direct_live_distillation": {
                        "schema": "mlx_score_aware_pose_direct_live_distillation.v1",
                        "enabled": False,
                        "weight": 0.0,
                        "authority": "macos_mlx_research_signal_false_authority",
                    },
                    "training_telemetry_contract": {
                        "schema": "compact_score_aware_training_telemetry_contract.v1",
                        "passed": True,
                        "blockers": [],
                        "authority": "macos_mlx_research_signal_false_authority",
                    },
                    "recon_pixel_weight": {
                        "schema": "compact_recon_pixel_weight.v1",
                        "enabled": True,
                        "source_kind": "auto_discovered_joint_p18_p19_file",
                        "path": weight_path.as_posix(),
                        "auto_discovery": discovery,
                        "authority": "false_macos_mlx_research_signal",
                    },
                }
            },
        }

    monkeypatch.setattr(runner_mod, "_discover_joint_recon_pixel_weight_path", fake_discover)
    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_auto_joint",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        auto_joint_recon_pixel_weight=True,
        recon_loss_stage_weight=0.25,
        segnet_loss_stage_weight=2.0,
        pose_loss_stage_weight=1.5,
        mlx_prefilter_scorer_device="gpu",
        repo_root=REPO_ROOT,
    )

    assert captured["recon_pixel_weight_path"] == weight_path
    assert captured["recon_pixel_weight_auto_discovery"] == discovery
    assert captured["auto_segnet_boundary_recon_weight"] is False
    assert captured["recon_loss_stage_weight"] == 0.25
    assert captured["segnet_loss_stage_weight"] == 2.0
    assert captured["pose_loss_stage_weight"] == 1.5
    assert captured["scorer_input_guard_stage_weight"] == 1.0
    assert captured["scorer_input_contrast_floor_stage_weight"] is None
    assert captured["scorer_input_shape_tether_stage_weight"] is None
    assert captured["segnet_direct_live_stage_weight"] is None
    assert captured["mlx_prefilter_scorer_device"] == "gpu"
    assert out["score_aware_training"]["stage_loss_weights"] == {
        "distill": 2.0,
        "pose_direct_live_distill": 1.5,
        "pose_distill": 1.5,
        "recon": 0.25,
        "scorer_input_contrast_floor": 1.0,
        "scorer_input_guard": 1.0,
        "scorer_input_shape_tether": 1.0,
        "posenet_yuv6_geometry_tether": 1.0,
        "posenet_temporal_signal_floor": 1.0,
        "segnet_direct_live_distill": 2.0,
        "segnet_direct_live_class_histogram": 2.0,
        "segnet_direct_live_class_balanced_hinge": 2.0,
        "segnet_direct_live_class_balanced_ce": 2.0,
        "segnet_direct_live_class_balanced_squared_hinge": 2.0,
        "segnet_direct_live_class_region_recon": 2.0,
        "segnet_direct_live_rare_class_logit": 2.0,
        "segnet_direct_live_target_mass_floor": 2.0,
        "segnet_direct_live_target_min_ratio_floor": 2.0,
    }
    assert out["score_aware_training"]["recon_pixel_weight"]["source_kind"] == ("auto_discovered_joint_p18_p19_file")
    assert out["score_aware_training"]["pose_direct_live_distillation"]["enabled"] is False
    assert out["score_aware_training"]["training_telemetry_contract"]["passed"] is True
    assert "hinerv_candidate_curriculum_recon_pixel_weight_missing" not in out["blockers"]


def test_hinerv_sampled_mlx_profile_does_not_unlock_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("sampled MLX profile must not run default CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    sampled_profile = tmp_path / "sampled_mlx_profile.json"
    sampled_profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "mlx_response_summary": {"max_pairs": 128, "n_samples": 128},
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "sampled_prefix_requires_full_video_rerun"},
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(sampled_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is False
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out["blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" in out["blockers"]
    assert "sampled_mlx_prefilter_requires_full_video_rerun" in out["blockers"]


def test_hinerv_full_video_bad_mlx_score_does_not_unlock_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("bad MLX prefilter score must not run default CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    bad_profile = tmp_path / "bad_full_video_mlx_profile.json"
    bad_profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 90.0},
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(bad_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["best_full_video_mlx_score"] == 90.0
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in out["blockers"]
    assert "mlx_prefilter_score_not_below_local_replay_threshold" in out["blockers"]
    assert "mlx_score_above_hard_demote_threshold" in out["blockers"]


def test_hinerv_batched_full_video_mlx_prefilter_feeds_acquisition_not_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        _write_mlx_prefilter_profile(
            out / "local_mlx_prefilter_profile.json",
            pairs=600,
            batch_pairs=8,
            score=91.0,
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("batched MLX acquisition profile must not run CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_batched_gpu_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_profile_paths"] == []
    assert out["mlx_prefilter_coverage"]["blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in out["blockers"]
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" not in out["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" in out["blockers"]
    feedback_blockers = out["candidate_feedback"]["row"]["pr95_stack_binding_blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in feedback_blockers
    assert "hi_nerv_local_cpu_replay_gate_missing" in feedback_blockers
    feedback_row = out["candidate_feedback"]["row"]
    assert feedback_row["mlx_prefilter_has_full_video"] is True
    assert feedback_row["mlx_prefilter_local_replay_passed"] is False
    assert feedback_row["mlx_prefilter_blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert feedback_row["local_cpu_replay_gate_has_full_video_mlx_prefilter"] is True
    assert feedback_row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"] is False
    assert feedback_row["local_cpu_replay_gate_executed"] is False


def test_hinerv_full_coverage_waits_for_mlx_prefilter_before_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is False
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out["blockers"]


def test_hinerv_execute_threads_coder_qat_and_reads_verified_waterfill_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}
    weight_path = tmp_path / "joint_p18_p19_weight.npy"
    np.save(weight_path, np.ones((384, 512), dtype=np.float32))
    waterfill_plan_path = tmp_path / "hinerv_decoder_waterfill.json"
    waterfill_plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv-unit-candidate",
        "group_count": 1,
        "full_video_coverage": True,
        "receiver_proof_status": "runtime_consumption_proof_ready",
        "rows": [
            {
                "group_name": "head_rgb_1.bias",
                "shape": [3],
                "numel": 3,
                "selected_bits": 0,
                "selected_action": "zero_rle",
                "blockers": [],
            }
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    waterfill_plan_path.write_text(
        json.dumps(waterfill_plan, sort_keys=True),
        encoding="utf-8",
    )

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "schema": "mlx_score_aware_training_objective.v1",
                    "segnet_distillation_weight": 0.0,
                    "pose_distillation_weight": 0.0,
                },
                "substrate_supplied_score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "coder_aware_qat": {
                        "schema": "coder_aware_decoder_qat.v1",
                        "enabled": True,
                        "quant_bits": 4,
                        "quant_residual_weight": 0.001,
                        "magnitude_weight": 0.0001,
                        "delta_weight": 0.0002,
                        "c1a_entropy_weight": 0.0003,
                        "c1a_sigma": 0.35,
                        "c1a_sample_size": 64,
                        "c1a_source": (
                            "PR95 cat_entropy_v2 soft categorical entropy adapted to selected decoder weights"
                        ),
                        "authority": "false_macos_mlx_research_signal",
                    },
                    "recon_pixel_weight": {
                        "schema": "compact_recon_pixel_weight.v1",
                        "enabled": True,
                        "source_kind": "file",
                        "path": Path(kwargs["recon_pixel_weight_path"]).as_posix(),
                        "sha256": runner_mod._sha256_file(Path(kwargs["recon_pixel_weight_path"])),
                        "npz_key": None,
                        "normalize": kwargs["recon_pixel_weight_normalize"],
                        "scorer_terms": {
                            "p18_segnet": "caller_supplied",
                            "p19_posenet": "caller_supplied",
                        },
                        "authority": "false_macos_mlx_research_signal",
                    },
                },
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_qat",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        decoder_codec="int2_scale_bundled",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-unit-candidate",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "hi_nerv_latent_codec": "int16_brotli_q11",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "hi_nerv",
                "control_semantics": ("local_receiver_visible_grid_search_nearest_target"),
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        decoder_weight_waterfill_plan_json=waterfill_plan_path,
        recon_pixel_weight_path=weight_path,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=0.5,
        recon_pixel_weight_normalize="mean",
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["candidate_curriculum_plan"]["coder_pressure"]["quant_bits"] == 4
    assert captured_train_kwargs["latent_dim"] == 12
    assert captured_train_kwargs["embed_dim"] == 16
    assert captured_train_kwargs["decoder_channel"] == 6
    assert captured_train_kwargs["decoder_codec"] == "portfolio_auto"
    assert captured_train_kwargs["hi_nerv_latent_codec"] == "int16_brotli_q11"
    assert captured_train_kwargs["hard_byte_ceiling"] == 178_000
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
    assert captured_train_kwargs["coder_qat_c1a_entropy_weight"] == 0.0003
    assert captured_train_kwargs["coder_qat_c1a_sigma"] == 0.35
    assert captured_train_kwargs["coder_qat_c1a_sample_size"] == 64
    captured_waterfill = captured_train_kwargs["decoder_weight_waterfill_plan"]
    assert captured_waterfill["schema"] == waterfill_plan["schema"]
    assert captured_waterfill["rows"] == waterfill_plan["rows"]
    launch_custody = captured_waterfill["compact_runner_launch_custody"]
    assert launch_custody["schema"] == ("compact_hi_nerv_decoder_weight_waterfill_launch_custody.v1")
    assert launch_custody["path"] == waterfill_plan_path.as_posix()
    assert launch_custody["sha256"] == runner_mod._sha256_file(waterfill_plan_path)
    assert launch_custody["source_schema"] == "nerv_decoder_weight_waterfill.v1"
    assert launch_custody["score_claim"] is False
    assert captured_waterfill["receiver_proof_status"] == "runtime_consumption_proof_ready"
    assert captured_waterfill["full_video_coverage"] is True
    assert captured_waterfill["blockers"] == []
    assert captured_train_kwargs["recon_pixel_weight_path"] == weight_path
    assert captured_train_kwargs["auto_segnet_boundary_recon_weight"] is False
    assert captured_train_kwargs["recon_pixel_weight_tau"] == 0.5
    assert captured_train_kwargs["recon_pixel_weight_normalize"] == "mean"
    coder_qat = out["score_aware_training"]["coder_aware_qat"]
    expected_coder_qat = {
        "schema": "coder_aware_decoder_qat.v1",
        "enabled": True,
        "quant_bits": 4,
        "quant_residual_weight": 0.001,
        "magnitude_weight": 0.0001,
        "delta_weight": 0.0002,
        "c1a_entropy_weight": 0.0003,
        "c1a_sigma": 0.35,
        "c1a_sample_size": 64,
        "c1a_source": ("PR95 cat_entropy_v2 soft categorical entropy adapted to selected decoder weights"),
        "authority": "false_macos_mlx_research_signal",
    }
    for key, expected in expected_coder_qat.items():
        assert coder_qat[key] == expected
    selection = out["modelsize_candidate_selection"]
    assert selection["selection_mode"] == "planner_candidate"
    assert selection["candidate"]["candidate_id"] == "hinerv-unit-candidate"
    assert selection["modelsize_control_contract"]["family"] == "hi_nerv"
    assert selection["modelsize_control_contract"]["control_semantics"] == (
        "local_receiver_visible_grid_search_nearest_target"
    )
    assert selection["modelsize_control_contract"]["archive_bytes_authority_required"] is True
    assert selection["modelsize_control_contract"]["control_precedence"]["child_rules_override_parent_defaults"] is True
    precedence = out["hi_nerv_control_precedence"]
    assert precedence["more_finely_grained_child_rules_take_priority"] is True
    assert precedence["pact_controls_take_priority_inside_source_faithful_subset"] is True
    assert precedence["highest_specificity_active_layer"] == ("promotion_and_exact_eval_gates")
    assert (
        precedence["modelsize_control_precedence"]["highest_specificity_active_layer"]
        == "pact_receiver_visible_modelsize_child_rule"
    )
    assert selection["candidate_curriculum_plan"]["coder_pressure"]["enabled"] is True
    assert selection["candidate_curriculum_plan"]["coder_pressure"]["quant_bits"] == 4
    assert selection["launch_latent_dim"] == 12
    assert selection["launch_embed_dim"] == 16
    assert selection["launch_decoder_channel"] == 6
    assert selection["requested_launch_decoder_codec"] == "int4_mixed"
    assert selection["launch_decoder_codec"] == "portfolio_auto"
    assert selection["launch_decoder_codec_policy"]["portfolio_auto_intervened"] is True
    assert selection["launch_hi_nerv_latent_codec"] == "int16_brotli_q11"
    assert out["score_aware_training"]["requested_launch_decoder_codec"] == ("int4_mixed")
    assert out["score_aware_training"]["decoder_codec"] == "portfolio_auto"
    assert out["score_aware_training"]["hi_nerv_latent_codec"] == ("int16_brotli_q11")
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is True
    assert waterfill["path"] == waterfill_plan_path.as_posix()
    assert waterfill["sha256"] == runner_mod._sha256_file(waterfill_plan_path)
    assert waterfill["launch_custody"] == launch_custody
    assert waterfill["source_schema"] == "nerv_decoder_weight_waterfill.v1"
    assert waterfill["row_count"] == 1
    assert waterfill["source_blockers"] == []
    assert waterfill["active"] is True
    assert waterfill["validated"] is True
    assert waterfill["validated_rows"] == [
        {
            "group_name": "head_rgb_1.bias",
            "expected_shape": [3],
            "expected_numel": 3,
            "shape_checked": True,
            "numel_checked": True,
        }
    ]
    assert waterfill["score_claim"] is False
    feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert feedback["measured_archive_bytes"] == out["archive_bytes"]
    assert "partial_pair_byte_feedback_only" in out["candidate_curriculum_plan"]["blockers"]
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in out["candidate_curriculum_plan"]["blockers"]
    candidate_feedback = out["candidate_feedback"]
    assert Path(candidate_feedback["row_path"]).is_file()
    assert Path(candidate_feedback["ledger_path"]).is_file()
    assert candidate_feedback["row"]["candidate_id"] == "hinerv-unit-candidate"
    assert candidate_feedback["row"]["candidate_num_pairs"] == 600
    assert candidate_feedback["row"]["measured_num_pairs"] == 2
    assert candidate_feedback["row"]["feedback_ready"] is False
    assert candidate_feedback["row"]["feedback_scope"] == "partial_pair_advisory"
    assert "partial_pair_byte_feedback_only" in (candidate_feedback["row"]["blockers"])
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in (candidate_feedback["row"]["blockers"])
    assert "hi_nerv_trained_archive_byte_oracle_partial_pair_scope" in (feedback["byte_oracle_blockers"])
    assert candidate_feedback["score_claim"] is False
    assert out["score_aware_training"]["recon_pixel_weight"]["source_kind"] == "file"
    assert out["score_aware_training"]["recon_pixel_weight"]["path"] == (weight_path.as_posix())
    assert out["score_aware_training"]["recon_pixel_weight"]["scorer_terms"] == {
        "p18_segnet": "caller_supplied",
        "p19_posenet": "caller_supplied",
    }


def test_hinerv_execute_uses_receiver_survival_portfolio_for_scorer_active_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "schema": "mlx_score_aware_training_objective.v1",
                    "training_telemetry_contract": {
                        "schema": "compact_score_aware_training_telemetry_contract.v1",
                        "passed": True,
                        "row_count": 1,
                        "segnet_direct_live_max_candidate_occupied_class_fraction": 0.8,
                        "segnet_direct_live_max_candidate_target_class_coverage_fraction": 1.0,
                        "segnet_direct_live_max_candidate_target_class_min_ratio": 0.25,
                    },
                },
            },
        }

    def fake_receiver_quality(**_kwargs):
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "quality_gate_passed": True,
            "quality_gate": {"verdict": "CACHE_QUALITY_GATE_PASSED"},
            "segnet_argmax_probe": {
                "candidate_occupied_class_fraction": 0.8,
                "reference_occupied_class_fraction": 0.8,
                "candidate_target_class_coverage_fraction": 1.0,
                "candidate_target_class_min_ratio": 0.25,
                "target_material_class_count": 5,
                "segnet_argmax_disagreement_rate": 0.05,
                "sample_pairs": 2,
            },
            "blockers": [],
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "_write_hi_nerv_runner_post_export_receiver_cache_quality",
        fake_receiver_quality,
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_direct_live_codec_policy",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-direct-live-codec-policy",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "hi_nerv_latent_codec": "int16_brotli_q11",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "hi_nerv",
                "control_semantics": ("local_receiver_visible_grid_search_nearest_target"),
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_objective="boundary_argmax_hinge",
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_class_balanced_ce_weight=1.0,
        pose_direct_live_distillation_weight=0.25,
        hi_nerv_optimizer_policy="native_optimizer",
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["decoder_codec"] == "portfolio_auto"
    selection = out["modelsize_candidate_selection"]
    assert selection["requested_launch_decoder_codec"] == "int4_mixed"
    assert selection["launch_decoder_codec"] == "portfolio_auto"
    policy = selection["launch_decoder_codec_policy"]
    assert policy["portfolio_auto_intervened"] is True
    assert policy["policy"] == ("score_preserving_receiver_survival_portfolio_under_byte_cap")
    assert out["score_aware_training"]["requested_launch_decoder_codec"] == ("int4_mixed")
    assert out["score_aware_training"]["launch_decoder_codec"] == "portfolio_auto"
    assert out["score_aware_training"]["launch_decoder_codec_policy"] == policy


def test_hinerv_execute_threads_archive_section_telemetry_into_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 2048,
        "section_payload_bytes": 1900,
        "sections": [
            {"name": "decoder_state", "role": "decoder", "bytes": 1200},
            {"name": "latents_coarse", "role": "latent", "bytes": 80},
            {"name": "latents_mid", "role": "latent", "bytes": 120},
            {"name": "latents_fine", "role": "latent", "bytes": 160},
            {"name": "hiv1_header", "role": "header", "bytes": 32},
        ],
    }
    telemetry_path = tmp_path / "hinerv_archive_section_telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "substrate_supplied_score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "archive_section_telemetry_attachment": kwargs["archive_section_telemetry_metadata"],
                    "archive_section_qat_weight_policy": {
                        "schema": "hi_nerv_archive_section_qat_weight_policy.v1",
                        "active": True,
                        "decoder_section_bytes": 1200,
                        "latent_section_bytes": 360,
                        "extra_loss_weights": {
                            "coder_qat_quant_residual": 0.002,
                            "latent_qat_quant_residual": 0.0015,
                        },
                        "blockers": [],
                    },
                    "latent_coder_aware_qat": {
                        "schema": "latent_coder_aware_qat.v1",
                        "enabled": True,
                        "quant_bits": 8,
                    },
                },
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_archive_section_qat",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        archive_section_telemetry_json=telemetry_path,
        post_export_receiver_cache_quality_gate=False,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["archive_section_telemetry"] == telemetry
    attachment = captured_train_kwargs["archive_section_telemetry_metadata"]
    assert attachment["attached"] is True
    assert attachment["validated"] is True
    assert attachment["path"] == telemetry_path.as_posix()
    assert attachment["sha256"] == runner_mod._sha256_file(telemetry_path)
    assert attachment["source_schema"] == "hinerv_archive_section_telemetry.v1"
    assert attachment["archive_zip_bytes"] == 2048
    assert attachment["section_payload_bytes"] == 1900
    assert attachment["section_count"] == 5
    report_attachment = out["score_aware_training"]["archive_section_telemetry_attachment"]
    assert report_attachment["validated"] is True
    assert report_attachment["path"] == telemetry_path.as_posix()
    policy = out["score_aware_training"]["archive_section_qat_weight_policy"]
    assert policy["active"] is True
    assert policy["decoder_section_bytes"] == 1200
    assert policy["latent_section_bytes"] == 360
    assert "latent_qat_quant_residual" in policy["extra_loss_weights"]
    latent_qat = out["score_aware_training"]["latent_coder_aware_qat"]
    assert latent_qat["enabled"] is True
    assert out["score_claim"] is False


@pytest.mark.parametrize(
    ("case_name", "payload", "expected_blocker"),
    [
        ("missing_file", None, "hi_nerv_archive_section_telemetry_json_missing"),
        (
            "schema_mismatch",
            {
                "schema": "wrong_schema.v1",
                "profile_ready": True,
                "sections": [{"name": "decoder_state", "role": "decoder", "bytes": 1}],
            },
            "hi_nerv_archive_section_telemetry_schema_mismatch",
        ),
        (
            "profile_not_ready",
            {
                "schema": "hinerv_archive_section_telemetry.v1",
                "profile_ready": False,
                "sections": [{"name": "decoder_state", "role": "decoder", "bytes": 1}],
            },
            "hi_nerv_archive_section_telemetry_not_profile_ready",
        ),
        (
            "sections_missing",
            {
                "schema": "hinerv_archive_section_telemetry.v1",
                "profile_ready": True,
            },
            "hi_nerv_archive_section_telemetry_sections_missing",
        ),
        (
            "decoder_missing",
            {
                "schema": "hinerv_archive_section_telemetry.v1",
                "profile_ready": True,
                "sections": [{"name": "latents_mid", "role": "latent", "bytes": 16}],
            },
            "hi_nerv_archive_section_telemetry_decoder_state_missing",
        ),
    ],
)
def test_hinerv_execute_refuses_bad_archive_section_telemetry_before_training(
    tmp_path: Path,
    monkeypatch,
    case_name: str,
    payload: dict[str, object] | None,
    expected_blocker: str,
) -> None:
    telemetry_path = tmp_path / f"{case_name}.json"
    if payload is not None:
        telemetry_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def fail_train(**_kwargs):
        raise AssertionError("trainer must not run with invalid telemetry")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / f"hinerv_bad_telemetry_{case_name}",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        archive_section_telemetry_json=telemetry_path,
        post_export_receiver_cache_quality_gate=False,
        repo_root=REPO_ROOT,
    )

    assert out["training_executed"] is False
    assert out["trainer_launch_allowed"] is False
    assert out["mode"] == "hi_nerv_archive_section_telemetry_launch_refused"
    assert expected_blocker in out["blockers"]
    assert "hi_nerv_training_not_launched" in out["blockers"]
    attachment = out["score_aware_training"]["archive_section_telemetry_attachment"]
    assert attachment["attached"] is True
    assert attachment["validated"] is False
    assert expected_blocker in attachment["blockers"]


def test_hinerv_execute_does_not_top_level_block_when_section_telemetry_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "substrate_supplied_score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "archive_section_qat_weight_policy": {
                        "schema": "hi_nerv_archive_section_qat_weight_policy.v1",
                        "attached": False,
                        "active": False,
                        "blockers": ["hinerv_archive_section_telemetry_not_attached"],
                    },
                },
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_no_section_telemetry",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        post_export_receiver_cache_quality_gate=False,
        repo_root=REPO_ROOT,
    )

    assert "hinerv_archive_section_telemetry_not_attached" not in out["blockers"]
    assert out["score_aware_training"]["archive_section_telemetry_attachment"]["attached"] is False


def test_hinerv_execute_failure_preserves_archive_section_telemetry_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 256,
        "sections": [{"name": "decoder_state", "role": "decoder", "bytes": 128}],
    }
    telemetry_path = tmp_path / "valid_hinerv_sections.json"
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")

    def fail_train(**_kwargs):
        raise RuntimeError("synthetic trainer failure")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_failure_section_telemetry",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        archive_section_telemetry_json=telemetry_path,
        post_export_receiver_cache_quality_gate=False,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_mlx_scoreaware_failed"
    assert "synthetic trainer failure" in out["failure"]
    attachment = out["score_aware_training"]["archive_section_telemetry_attachment"]
    assert attachment["attached"] is True
    assert attachment["validated"] is True
    assert attachment["path"] == telemetry_path.as_posix()
    assert "hi_nerv_mlx_scoreaware_or_export_failed" in out["blockers"]


def _hinerv_waterfill_modelsize_candidate(
    candidate_id: str = "hinerv-unit-candidate",
) -> dict[str, object]:
    return {
        "schema": "hinerv_modelsize_candidate.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "latent_dim": 12,
        "embed_dim": 16,
        "decoder_channel": 6,
        "mid_injection_block_index": 2,
        "fine_injection_block_index": 5,
        "decoder_codec": "int4_mixed",
        "num_pairs": 600,
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 160_000,
        "nominal_under_ceiling": True,
        "use_hierarchical_feature_grid": True,
        "use_convnext_blocks": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_hinerv_full600_modelsize_candidate_can_run_partial_timing_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    captured: dict[str, object] = {}
    decoder_weight_waterfill_plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv-unit-candidate",
        "rows": [
            {"group_name": "head_rgb_1.bias", "selected_bits": 4},
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fake_quant = {}

        def configure_decoder_fake_quant_forward(self, **kwargs):
            self.fake_quant = dict(kwargs)

        def initialize_output_head_bias_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            epsilon,
        ):
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.bias",
                    "head_rgb_1.bias",
                ],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            steps,
            learning_rate,
            rgb_weight,
            yuv6_weight,
            temporal_delta_weight,
            contrast_floor_weight,
            rgb_std_min_ratio,
            yuv6_temporal_std_min_ratio,
            weight_decay,
            grad_clip_max_norm,
            target_segnet_argmax_1=None,
            scorer_teacher=None,
            segnet_margin_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_min_ratio_floor=0.02,
            pair_local_smoke_artifact_dir=None,
        ):
            return _fake_hinerv_scorer_domain_bootstrap_payload(
                pair_indices,
                steps=steps,
                learning_rate=learning_rate,
                rgb_weight=rgb_weight,
                yuv6_weight=yuv6_weight,
                temporal_delta_weight=temporal_delta_weight,
                contrast_floor_weight=contrast_floor_weight,
                rgb_std_min_ratio=rgb_std_min_ratio,
                yuv6_temporal_std_min_ratio=yuv6_temporal_std_min_ratio,
                weight_decay=weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            )

        def num_parameters(self):
            return 65050

    class FakeArtifact:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def as_dict(self) -> dict[str, object]:
            return dict(self._payload)

    def fake_decode_mlx_targets(
        video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        captured["decode_num_pairs"] = int(num_pairs)
        captured["decode_pair_indices"] = tuple(pair_indices or ())
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def fake_run_mlx_score_aware_full_main(**kwargs):
        bundle = kwargs["bundle"]
        captured["bundle_num_pairs"] = int(bundle.num_pairs)
        captured["model_num_pairs"] = int(bundle.model.cfg.num_pairs)
        captured["model_fake_quant"] = dict(bundle.model.fake_quant)
        captured["gradient_multiplier_by_name"] = dict(kwargs["gradient_multiplier_by_name"])
        captured["metadata"] = dict(bundle.substrate_artifact_metadata)
        return FakeArtifact({"substrate_artifact_metadata": captured["metadata"]})

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )

    artifact = runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
        output_dir=tmp_path / "hinerv_partial_full600_candidate",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
        latent_dim=12,
        embed_dim=16,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=2,
        fine_injection_block_index=5,
        decoder_codec="int4_mixed",
        hi_nerv_latent_codec="int16_brotli_q11",
        hard_byte_ceiling=178_000,
        ema_decay=0.9,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        recon_loss_stage_weight=1.0,
        segnet_loss_stage_weight=1.0,
        pose_loss_stage_weight=1.0,
        scorer_input_guard_stage_weight=1.0,
        scorer_input_contrast_floor_stage_weight=None,
        scorer_input_shape_tether_stage_weight=None,
        segnet_direct_live_stage_weight=None,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        scorer_domain_bootstrap_segnet_hard_birth_weight=0.0,
        distillation_device="cpu",
        requested_distillation_device=None,
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0001,
        coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
        coder_qat_c1a_sample_size=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
        recon_pixel_weight_path=None,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        recon_pixel_weight_auto_discovery=None,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=1.0,
        recon_pixel_weight_normalize="mean",
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=50,
        telemetry_flush_interval_epochs=1,
        checkpoint_interval_epochs=1,
        checkpoint_retention_keep_last_n=1,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        optimizer_kind="pact_muon_adamw",
        hi_nerv_optimizer_policy={
            "pr95_faithful_curriculum_enabled": True,
            "pr95_muon_policy": "every_stage",
            "native_optimizer_active": True,
        },
        optimizer_controls={},
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights=None,
        scorer_error_pair_curriculum=None,
        random_seed=0,
        scorer_upstream_dir=REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
        pr95_curriculum_total_epochs=80,
        modelsize_candidate=_hinerv_waterfill_modelsize_candidate(),
        archive_section_telemetry={
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "archive_zip_bytes": 4096,
            "sections": [
                {"name": "decoder_state", "role": "decoder", "bytes": 3072},
                {"name": "latents_coarse", "role": "latent", "bytes": 128},
                {"name": "latents_mid", "role": "latent", "bytes": 256},
            ],
        },
    )

    assert artifact.as_dict()["substrate_artifact_metadata"] == captured["metadata"]
    assert captured["decode_num_pairs"] == 2
    assert captured["decode_pair_indices"] == ()
    assert captured["bundle_num_pairs"] == 2
    assert captured["model_num_pairs"] == 600
    assert captured["model_fake_quant"]["enabled"] is True
    assert captured["model_fake_quant"]["quant_bits"] == 4
    assert captured["model_fake_quant"]["per_tensor_bits"] == {"head_rgb_1.bias": 4}
    assert captured["model_fake_quant"]["stage_controlled"] is True
    assert captured["gradient_multiplier_by_name"] == {"head_rgb_1.bias": pytest.approx(1.414214)}
    metadata = captured["metadata"]
    assert metadata["num_pairs"] == 2
    assert metadata["training_num_pairs"] == 2
    assert metadata["model_num_pairs"] == 600
    receiver_proof_policy = metadata["receiver_proof_export_policy"]
    assert receiver_proof_policy["enabled"] is False
    assert receiver_proof_policy["training_num_pairs"] == 2
    assert receiver_proof_policy["model_num_pairs"] == 600
    assert receiver_proof_policy["partial_candidate_smoke_skips_full_receiver_proof"] is True
    consumption = metadata["modelsize_candidate_consumption"]
    assert consumption["candidate_num_pairs"] == 600
    assert consumption["training_num_pairs"] == 2
    assert consumption["model_num_pairs"] == 600
    assert consumption["partial_pair_training_against_full_candidate"] is True
    fake_quant = metadata["score_aware_training"]["decoder_fake_quant_forward"]
    assert fake_quant["enabled"] is True
    assert fake_quant["configured_enabled"] is True
    assert fake_quant["initial_forward_active"] is False
    assert fake_quant["stage_controlled"] is True
    assert fake_quant["stage_control_source"] == "pr95_faithful_stage_verdict.qat_active"
    assert fake_quant["per_tensor_waterfill_enabled"] is True
    assert fake_quant["per_tensor_waterfill_bits_by_name"] == {"head_rgb_1.bias": 4}
    gradient_policy = metadata["score_aware_training"]["gradient_multipliers"][
        "decoder_weight_waterfill_gradient_policy"
    ]
    assert gradient_policy["active_after_merge"] is True
    assert gradient_policy["merged_multiplier_by_name"] == {"head_rgb_1.bias": pytest.approx(1.414214)}
    assert "score_claim" not in gradient_policy
    assert "promotion_eligible" not in gradient_policy
    assert "ready_for_exact_eval_dispatch" not in gradient_policy
    assert gradient_policy["authority"] == "macos_mlx_research_signal_false_authority"


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX required (Apple Silicon)")
def test_hinerv_private_smoke_generates_startup_section_telemetry_for_qat_terms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mlx.core as mx

    from tac.substrates._shared import mlx_score_aware as mlx_score_aware_pkg
    from tac.substrates.hi_nerv import archive as hinerv_archive
    from tac.substrates.hi_nerv import archive_candidate as hinerv_archive_candidate
    from tac.substrates.hi_nerv import mlx_renderer as hinerv_mlx_renderer

    captured: dict[str, object] = {}
    live_pack_calls: list[dict[str, object]] = []

    class FakeHinervModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fake_quant = {}
            self._params = {
                "decoder": {
                    "head": {
                        "weight": mx.array([0.0, 0.5, 1.0], dtype=mx.float32),
                    },
                },
                "latents_coarse": mx.array([0.25, 0.5], dtype=mx.float32),
                "latents_mid": mx.array([0.125, 0.75], dtype=mx.float32),
            }

        def configure_decoder_fake_quant_forward(self, **kwargs):
            self.fake_quant = dict(kwargs)

        def initialize_output_head_bias_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            epsilon,
        ):
            return {
                "schema": "hi_nerv_output_head_target_bias_init.v1",
                "enabled": True,
                "epsilon": float(epsilon),
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.bias",
                    "head_rgb_1.bias",
                ],
            }

        def initialize_output_head_contrast_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            min_output_std,
            max_gain,
        ):
            return _fake_hinerv_output_head_contrast_init_payload(
                pair_indices,
                min_output_std=min_output_std,
                max_gain=max_gain,
            )

        def fit_scorer_domain_bootstrap_from_targets(
            self,
            _target_rgb_0,
            _target_rgb_1,
            *,
            pair_indices,
            steps,
            learning_rate,
            rgb_weight,
            yuv6_weight,
            temporal_delta_weight,
            contrast_floor_weight,
            rgb_std_min_ratio,
            yuv6_temporal_std_min_ratio,
            weight_decay,
            grad_clip_max_norm,
            target_segnet_argmax_1=None,
            scorer_teacher=None,
            segnet_margin_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_weight=0.0,
            segnet_hard_birth_bootstrap_min_ratio_floor=0.02,
            pair_local_smoke_artifact_dir=None,
        ):
            return _fake_hinerv_scorer_domain_bootstrap_payload(
                pair_indices,
                steps=steps,
                learning_rate=learning_rate,
                rgb_weight=rgb_weight,
                yuv6_weight=yuv6_weight,
                temporal_delta_weight=temporal_delta_weight,
                contrast_floor_weight=contrast_floor_weight,
                rgb_std_min_ratio=rgb_std_min_ratio,
                yuv6_temporal_std_min_ratio=yuv6_temporal_std_min_ratio,
                weight_decay=weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            )

        def num_parameters(self):
            return 7

        def parameters(self):
            return self._params

        def export_state_dict(self) -> dict[str, np.ndarray]:
            return {
                "decoder.weight": np.asarray([1.0, 2.0], dtype=np.float32),
                "latents_coarse": np.asarray([0.25, 0.5], dtype=np.float32),
            }

    class FakeArtifact:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def as_dict(self) -> dict[str, object]:
            return dict(self._payload)

    def fake_decode_mlx_targets(
        video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices=None,
    ):
        shape = (int(num_pairs), int(output_height), int(output_width), 3)
        return mx.zeros(shape, dtype=mx.float32), mx.zeros(shape, dtype=mx.float32)

    def fake_run_mlx_score_aware_full_main(**kwargs):
        bundle = kwargs["bundle"]
        captured["extra_loss_weights"] = dict(bundle.extra_loss_weights)
        terms = bundle.extra_loss_terms(
            bundle.model,
            mx.array([0], dtype=mx.int32),
        )
        mx.eval(*terms.values())
        captured["extra_loss_term_keys"] = sorted(terms)
        captured["term_values"] = {key: float(value.item()) for key, value in terms.items()}
        captured["dual_loss_weight_keys"] = sorted(
            row["loss_weight_key"] for row in kwargs["train_time_dual_ascent_config"]["constraints"]
        )
        captured["dual_metric_names"] = sorted(
            row["metric_name"] for row in kwargs["train_time_dual_ascent_config"]["constraints"]
        )
        assert bundle.train_time_section_byte_metrics is not None
        captured["section_byte_metrics"] = bundle.train_time_section_byte_metrics(
            bundle.model,
            mx.array([0], dtype=mx.int32),
            dict(bundle.extra_loss_weights),
        )
        captured["metadata"] = dict(bundle.substrate_artifact_metadata)
        return FakeArtifact({"substrate_artifact_metadata": captured["metadata"]})

    def fake_pack_archive_from_exported_state_dict(**kwargs):
        live_pack_calls.append(dict(kwargs))
        return b"live-hiv1-section-test"

    def fake_build_archive_section_telemetry(packet: bytes) -> dict[str, object]:
        assert packet == b"live-hiv1-section-test"
        return {
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "inner_payload_bytes": 333,
            "sections": [
                {"name": "decoder_state", "role": "decoder", "bytes": 210},
                {"name": "latents_coarse", "role": "latent", "bytes": 41},
                {"name": "latents_mid", "role": "latent", "bytes": 42},
            ],
        }

    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "decode_mlx_targets",
        fake_decode_mlx_targets,
    )
    monkeypatch.setattr(
        mlx_score_aware_pkg,
        "run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )
    monkeypatch.setattr(
        hinerv_mlx_renderer,
        "HinervSubstrateMLX",
        FakeHinervModel,
    )
    monkeypatch.setattr(
        hinerv_archive_candidate,
        "pack_archive_from_exported_state_dict",
        fake_pack_archive_from_exported_state_dict,
    )
    monkeypatch.setattr(
        hinerv_archive,
        "build_archive_section_telemetry",
        fake_build_archive_section_telemetry,
    )

    artifact = runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
        output_dir=tmp_path / "hinerv_section_qat_consumption",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=tmp_path / "not_read_by_fake_decoder.mkv",
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=2,
        fine_injection_block_index=5,
        decoder_codec="int4_mixed",
        hi_nerv_latent_codec="int16_brotli_q11",
        hard_byte_ceiling=178_000,
        ema_decay=0.9,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        recon_loss_stage_weight=1.0,
        segnet_loss_stage_weight=1.0,
        pose_loss_stage_weight=1.0,
        scorer_input_guard_stage_weight=1.0,
        scorer_input_contrast_floor_stage_weight=None,
        scorer_input_shape_tether_stage_weight=None,
        segnet_direct_live_stage_weight=None,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        scorer_domain_bootstrap_segnet_hard_birth_weight=0.0,
        distillation_device="cpu",
        requested_distillation_device=None,
        allow_segnet_only_research=False,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SIGMA,
        coder_qat_c1a_sample_size=runner_mod.DEFAULT_PACT_CODER_QAT_C1A_SAMPLE_SIZE,
        recon_pixel_weight_path=None,
        decoder_weight_waterfill_plan=None,
        recon_pixel_weight_auto_discovery=None,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=1.0,
        recon_pixel_weight_normalize="mean",
        mlx_prefilter_scorer_device=None,
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=50,
        telemetry_flush_interval_epochs=1,
        checkpoint_interval_epochs=1,
        checkpoint_retention_keep_last_n=1,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        optimizer_kind="pact_muon_adamw",
        hi_nerv_optimizer_policy={},
        optimizer_controls={"section_byte_refresh_every_steps": 1},
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights=None,
        scorer_error_pair_curriculum=None,
        random_seed=0,
        scorer_upstream_dir=REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
    )

    assert artifact.as_dict()["substrate_artifact_metadata"] == captured["metadata"]
    weights = captured["extra_loss_weights"]
    assert weights["coder_qat_quant_residual"] > 0.001
    assert weights["latent_qat_quant_residual"] > 0.001
    assert "coder_qat_quant_residual" in captured["extra_loss_term_keys"]
    assert "latent_qat_quant_residual" in captured["extra_loss_term_keys"]
    assert "coder_qat_quant_residual" in captured["dual_loss_weight_keys"]
    assert "latent_qat_quant_residual" in captured["dual_loss_weight_keys"]
    assert "train_time_section_rate_score__decoder_state" in captured["dual_metric_names"]
    assert "train_time_section_rate_score__latents_coarse" in captured["dual_metric_names"]
    assert "train_time_section_rate_score__latents_mid" in captured["dual_metric_names"]
    section_metrics = captured["section_byte_metrics"]
    assert section_metrics["source"] == "live_current_hiv1_packet"
    assert section_metrics["archive_bytes"] == 333
    assert section_metrics["section_bytes"] == {
        "decoder_state": 210,
        "latents_coarse": 41,
        "latents_mid": 42,
    }
    assert len(live_pack_calls) == 2
    assert all(call["decoder_codec"] == "int4_mixed" for call in live_pack_calls)
    assert all(call["latent_codec"] == "int16_brotli_q11" for call in live_pack_calls)
    metadata = captured["metadata"]["score_aware_training"]
    assert metadata["archive_section_qat_weight_policy"]["active"] is True
    section_control = metadata["train_time_section_byte_control"]
    assert section_control["active"] is True
    consumption = captured["metadata"]["modelsize_candidate_consumption"]
    assert consumption["hard_byte_ceiling_consumed_by_train_time_dual_ascent"] is True
    assert consumption["train_time_section_byte_control_active"] is True
    assert consumption["train_time_section_byte_control_blockers"] == []
    assert consumption["hard_byte_ceiling_train_time_dual_ascent_blockers"] == []
    assert section_control["section_byte_budgets"] == {
        "decoder_state": 127_576,
        "latents_coarse": 24_907,
        "latents_mid": 25_515,
    }
    assert section_control["section_byte_loss_weight_key_map"] == {
        "decoder_state": "coder_qat_quant_residual",
        "latents_coarse": "latent_qat_quant_residual",
        "latents_mid": "latent_qat_quant_residual",
    }
    assert metadata["latent_coder_aware_qat"]["enabled"] is True
    live_metadata = metadata["live_train_time_section_byte_metrics"]
    assert live_metadata["enabled"] is True
    assert live_metadata["refresh_every_steps"] == 1
    assert live_metadata["successful_refresh_count"] == 1
    assert live_metadata["last_live_archive_bytes"] == 333
    assert live_metadata["last_live_section_bytes"] == {
        "decoder_state": 210,
        "latents_coarse": 41,
        "latents_mid": 42,
    }
    attachment = metadata["archive_section_telemetry_attachment"]
    assert attachment["generated_from_initial_model"] is True
    assert attachment["validated"] is True
    assert attachment["startup_generation"]["source_schema"] == ("hinerv_archive_section_telemetry.v1")


def _write_hinerv_waterfill_plan(
    path: Path,
    *,
    candidate_id: str = "hinerv-unit-candidate",
    group_name: str = "head_rgb_1.bias",
    shape: list[int] | None = None,
    numel: int | None = None,
) -> Path:
    row: dict[str, object] = {
        "group_name": group_name,
        "selected_bits": 0,
        "selected_action": "zero_rle",
    }
    if shape is not None:
        row["shape"] = shape
    if numel is not None:
        row["numel"] = numel
    path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": candidate_id,
                "group_count": 1,
                "rows": [row],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _execute_hinerv_waterfill_validation_probe(
    tmp_path: Path,
    monkeypatch,
    *,
    plan_path: Path,
    candidate_id: str = "hinerv-unit-candidate",
) -> tuple[dict[str, object], dict[str, object]]:
    captured_train_kwargs: dict[str, object] = {}

    def fail_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        raise AssertionError("invalid waterfill plan must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / f"hinerv_waterfill_refusal_{candidate_id}",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        modelsize_candidate=_hinerv_waterfill_modelsize_candidate(candidate_id),
        decoder_weight_waterfill_plan_json=plan_path,
        repo_root=REPO_ROOT,
    )
    return out, captured_train_kwargs


def test_hinerv_waterfill_plan_candidate_mismatch_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "candidate_mismatch_waterfill.json",
        candidate_id="different-hinerv-candidate",
        shape=[3],
        numel=3,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    assert out["mode"] == "hi_nerv_decoder_weight_waterfill_plan_launch_refused"
    assert out["training_executed"] is False
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert waterfill["validated"] is False
    assert "decoder_weight_waterfill_candidate_id_mismatch:different-hinerv-candidate" in (out["blockers"])
    assert waterfill["candidate_match"]["matched"] is False
    assert waterfill["score_claim"] is False


def test_hinerv_waterfill_plan_unknown_group_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "unknown_group_waterfill.json",
        group_name="missing_decoder_group.weight",
        shape=[1],
        numel=1,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert "decoder_weight_waterfill_group_missing:missing_decoder_group.weight" in (out["blockers"])


def test_hinerv_waterfill_plan_shape_mismatch_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "shape_mismatch_waterfill.json",
        group_name="head_rgb_1.bias",
        shape=[4],
        numel=3,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert "decoder_weight_waterfill_shape_mismatch:head_rgb_1.bias" in out["blockers"]
    assert waterfill["validated_rows"][0]["declared_shape"] == [4]


def test_hinerv_waterfill_plan_projects_feature_grid_time_axis_for_smoke() -> None:
    candidate = _hinerv_waterfill_modelsize_candidate()
    full_shapes = runner_mod._hi_nerv_expected_decoder_state_shapes(
        num_pairs=600,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=2,
        fine_injection_block_index=5,
    )
    smoke_shapes = runner_mod._hi_nerv_expected_decoder_state_shapes(
        num_pairs=1,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=2,
        fine_injection_block_index=5,
    )
    group_name = "feature_grids.0.grids.0"
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv-unit-candidate",
        "rows": [
            {
                "group_name": group_name,
                "shape": list(full_shapes[group_name]),
                "numel": int(math.prod(full_shapes[group_name])),
                "selected_bits": 4,
                "selected_action": "int4",
            }
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    projected_plan, metadata = runner_mod._validate_hi_nerv_decoder_weight_waterfill_plan_attachment(
        plan=plan,
        metadata={
            "schema": "compact_hi_nerv_decoder_weight_waterfill_plan_attachment.v1",
            "attached": True,
            "blockers": [],
        },
        candidate=candidate,
        num_pairs=1,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=7,
        mid_injection_block_index=2,
        fine_injection_block_index=5,
    )

    assert projected_plan is not None
    assert metadata["attached"] is True
    assert metadata["active"] is True
    assert metadata["validated"] is True
    assert metadata["blockers"] == []
    assert metadata["train_time_fake_quant_bound"] is True
    assert metadata["fake_quant_forward"]["configured"] is True
    assert metadata["fake_quant_forward"]["targeted_tensor_count"] == 1
    assert metadata["fake_quant_forward"]["per_tensor_waterfill_bits_by_name"] == {group_name: 4}
    projection = metadata["smoke_projection"]
    assert projection["active"] is True
    assert projection["projected_row_count"] == 1
    assert projection["projection_scope"] == "hi_nerv_feature_grid_time_axis_only"
    row = projected_plan["rows"][0]
    assert row["shape"] == list(smoke_shapes[group_name])
    assert row["numel"] == int(math.prod(smoke_shapes[group_name]))
    assert row["selected_bits"] == 4
    assert row["compact_runner_smoke_projected_from_shape"] == list(full_shapes[group_name])
    assert projected_plan["compact_runner_smoke_projection"]["score_claim"] is False


def test_hinerv_waterfill_plan_compiles_train_time_fake_quant_bits() -> None:
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {"group_name": "head_rgb_1.weight", "selected_bits": 0},
            {"group_name": "blocks.0.conv.weight", "selected_bits": 4},
            {"group_name": "blocks.1.conv.weight", "selected_bits": 6},
            {"group_name": "blocks.2.conv.weight", "selected_bits": 7},
            {"group_name": "latent_embed.bias", "selected_bits": 32},
        ],
    }

    assert runner_mod._decoder_weight_waterfill_fake_quant_bits_by_name(plan) == {
        "head_rgb_1.weight": 0,
        "blocks.0.conv.weight": 4,
        "blocks.1.conv.weight": 6,
        "blocks.2.conv.weight": 7,
        "latent_embed.bias": 32,
    }

    invalid = {**plan, "rows": [{"group_name": "x", "selected_bits": 3}]}
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="selected_bits",
    ):
        runner_mod._decoder_weight_waterfill_fake_quant_bits_by_name(invalid)


def test_hinerv_waterfill_plan_compiles_gradient_multiplier_policy() -> None:
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {"group_name": "head_rgb_1.bias", "selected_bits": 0},
            {"group_name": "blocks.0.conv.weight", "selected_bits": 4},
            {"group_name": "blocks.1.conv.weight", "selected_bits": 8},
            {"group_name": "latent_embed.bias", "selected_bits": 32},
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    multipliers, policy = runner_mod._decoder_weight_waterfill_gradient_multiplier_by_name(plan)

    assert multipliers == {
        "head_rgb_1.bias": pytest.approx(2.828427),
        "blocks.0.conv.weight": pytest.approx(1.414214),
        "blocks.1.conv.weight": pytest.approx(1.05),
    }
    assert "latent_embed.bias" not in multipliers
    assert policy["active"] is True
    assert policy["targeted_tensor_count"] == 3
    assert policy["fake_quant_targeted_tensor_count"] == 3
    assert policy["blockers"] == []
    assert policy["score_claim"] is False


def test_hinerv_waterfill_gradient_multiplier_policy_explicit_override() -> None:
    waterfill_policy = {
        "schema": "compact_hi_nerv_decoder_weight_waterfill_gradient_policy.v1",
        "active": True,
        "multiplier_by_name": {"head_rgb_1.bias": 1.414214},
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    merged, policy = runner_mod._merge_decoder_waterfill_gradient_multiplier_controls(
        explicit_gradient_multiplier_by_name={
            "head_rgb_1.bias": 0.25,
            "head_rgb_0.bias": 0.5,
        },
        waterfill_gradient_multiplier_by_name={
            "head_rgb_1.bias": 1.414214,
            "blocks.0.conv.weight": 2.0,
        },
        waterfill_policy=waterfill_policy,
    )

    assert merged == {
        "head_rgb_1.bias": 0.25,
        "head_rgb_0.bias": 0.5,
        "blocks.0.conv.weight": 2.0,
    }
    assert policy["explicit_override_names"] == ["head_rgb_1.bias"]
    assert policy["explicit_overrides_take_precedence"] is True
    assert policy["merged_control_count"] == 3
    assert policy["active_after_merge"] is True


def test_hinerv_pose_distillation_warmup_compiles_real_curriculum_stages() -> None:
    weights = runner_mod._compact_scoreaware_stage_loss_weights(
        recon=1.0,
        segnet=2.0,
        pose=3.0,
    )
    stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="compact_runner_hi_nerv_mlx",
        epochs=4,
        loss_weights=weights,
        pose_distillation_warmup_epochs=2,
    )

    assert len(stages) == 2
    assert stages[0].start_epoch == 0
    assert stages[0].end_epoch == 2
    assert stages[0].loss_weights == {
        **weights,
        "pose_distill": 0.0,
        "pose_direct_live_distill": 0.0,
    }
    assert stages[1].start_epoch == 2
    assert stages[1].end_epoch == 4
    assert stages[1].loss_weights == weights

    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="smaller than epochs",
    ):
        runner_mod._compact_scoreaware_curriculum_stages(
            substrate_id="compact_runner_hi_nerv_mlx",
            epochs=2,
            loss_weights={"recon": 1.0, "distill": 1.0, "pose_distill": 1.0},
            pose_distillation_warmup_epochs=2,
        )


def test_hinerv_modelsize_launch_auto_binds_joint_scorer_pressure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_auto_score_pressure",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-auto-score-pressure",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.5,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        optimizer_kind="adafactor",
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["optimizer_controls"]["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["optimizer_controls"]["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert captured_train_kwargs["segnet_distillation_weight"] == 1.0
    assert captured_train_kwargs["pose_distillation_weight"] == 1.0
    assert captured_train_kwargs["mid_injection_block_index"] == 2
    assert captured_train_kwargs["fine_injection_block_index"] == 5
    assert captured_train_kwargs["pose_distillation_loss"] == "huber"
    assert captured_train_kwargs["pose_distillation_huber_delta"] == 2.5
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["hi_nerv_optimizer_policy"]["resolved_policy"] == ("native_optimizer")
    assert captured_train_kwargs["hi_nerv_optimizer_policy"]["optimizer_kind_consumed_by_native_mlx"] is True
    plan = captured_train_kwargs["candidate_curriculum_plan"]
    assert "hinerv_candidate_curriculum_requires_real_segnet_teacher" not in plan["blockers"]
    assert "hinerv_candidate_curriculum_requires_real_posenet_teacher" not in plan["blockers"]
    binding = out["hi_nerv_modelsize_launch_pressure"]
    assert binding["source"] == "modelsize_candidate_minimum_joint_scorer_pressure"
    assert {row["field"] for row in binding["mutations"]} == {
        "segnet_distillation_weight",
        "pose_distillation_weight",
    }
    assert out["score_aware_training"]["requested_segnet_distillation_weight"] == 0.0
    assert out["score_aware_training"]["requested_pose_distillation_weight"] == 0.0
    assert out["score_aware_training"]["segnet_distillation_weight"] == 1.0
    assert out["score_aware_training"]["pose_distillation_weight"] == 1.0
    assert out["score_aware_training"]["optimizer_kind"] == "adafactor"
    assert out["score_aware_training"]["optimizer_controls"]["optimizer_kind"] == ("adafactor")
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == ("native_optimizer")
    assert out["score_aware_training_config_gate"]["frontier_targeting"] is True
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" not in out["blockers"]
    assert "hinerv_candidate_curriculum_requires_real_segnet_teacher" not in out["blockers"]
    assert "hinerv_candidate_curriculum_requires_real_posenet_teacher" not in out["blockers"]


def test_hinerv_direct_live_segnet_plus_pose_counts_as_joint_teacher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_direct_live_joint_teacher",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=1.0,
        segnet_direct_live_base_loss_weight=0.0,
        segnet_distillation_objective="argmax_hinge",
        pose_distillation_weight=0.01,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=0.05,
        scorer_input_shape_tether_weight=0.25,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["segnet_distillation_weight"] == 0.0
    assert captured_train_kwargs["segnet_direct_live_distillation_weight"] == 1.0
    assert captured_train_kwargs["segnet_direct_live_base_loss_weight"] == 0.0
    assert captured_train_kwargs["pose_distillation_weight"] == pytest.approx(0.01)
    gate = out["score_aware_training_config_gate"]
    assert gate["frontier_targeting"] is True
    assert gate["real_segnet_teacher_attached"] is True
    assert gate["real_posenet_teacher_attached"] is True
    assert gate["direct_live_escape_controls_attached"] is True
    assert gate["direct_live_escape_controls"]["shape_tether"] == pytest.approx(0.25)
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" not in out["blockers"]


def test_hinerv_execute_forwards_target_mass_floor_to_live_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_target_mass_floor_forwarding",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_target_mass_floor_weight=0.75,
        segnet_direct_live_stage_weight=2.5,
        pose_direct_live_distillation_weight=0.5,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["segnet_direct_live_target_mass_floor_weight"] == (pytest.approx(0.75))
    assert captured_train_kwargs["segnet_direct_live_stage_weight"] == pytest.approx(2.5)
    assert out["score_aware_training_config_gate"]["launch_allowed"] is True
    assert out["score_aware_training_config_gate"]["direct_live_escape_controls"]["target_mass_floor"] == pytest.approx(
        0.75
    )
    assert out["candidate_curriculum_plan"]["scorer_pressure"]["segnet_direct_live_subcontrol_weights"][
        "target_mass_floor"
    ] == pytest.approx(0.75)


def test_hinerv_modelsize_launch_preserves_explicit_segnet_only_research(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_segnet_only_modelsize_probe",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-segnet-only-research",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=1.0,
        pose_distillation_weight=0.0,
        allow_segnet_only_research=True,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["segnet_distillation_weight"] == 1.0
    assert captured_train_kwargs["pose_distillation_weight"] == 0.0
    assert captured_train_kwargs["allow_segnet_only_research"] is True
    binding = out["hi_nerv_modelsize_launch_pressure"]
    assert binding["source"] == "caller_supplied_segnet_only_research_modelsize_pressure"
    assert binding["mutations"] == []
    assert binding["allow_segnet_only_research"] is True
    assert out["score_aware_training"]["requested_pose_distillation_weight"] == 0.0
    assert out["score_aware_training"]["pose_distillation_weight"] == 0.0
    assert out["score_aware_training_config_gate"]["launch_allowed"] is True
    assert out["score_aware_training_config_gate"]["frontier_targeting"] is False
    assert out["score_aware_training_config_gate"]["segnet_only_research_allowed"] is True
    assert "hi_nerv_segnet_only_research_not_frontier_targeting" in out["blockers"]
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" in out["blockers"]


def test_hinerv_optimizer_policy_refuses_and_avoids_silent_optimizer_swallowing() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="non-PR95-compatible --optimizer-kind would be ignored",
    ):
        runner_mod._resolve_hi_nerv_optimizer_policy(
            requested_policy="pr95_curriculum",
            epochs=29_650,
            optimizer_kind="lion",
        )

    native = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="lion",
    )
    assert native["resolved_policy"] == "native_optimizer"
    assert native["optimizer_kind_consumed_by_native_mlx"] is True

    native_adamw = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="adamw",
    )
    assert native_adamw["resolved_policy"] == "native_optimizer"
    assert native_adamw["optimizer_kind_consumed_by_native_mlx"] is True
    assert native_adamw["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert native_adamw["pr95_faithful_curriculum_enabled"] is False

    pr95 = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="adamw",
        pr95_curriculum_total_epochs=29_650,
    )
    assert pr95["resolved_policy"] == "pr95_curriculum"
    assert pr95["pr95_faithful_curriculum_enabled"] is True
    assert pr95["pr95_muon_policy"] == "faithful_stage8_only"


def test_hinerv_auto_policy_binds_long_pact_muon_adamw_to_pr95_every_stage() -> None:
    default_policy = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="pact_muon_adamw",
    )

    assert default_policy["resolved_policy"] == "pr95_curriculum"
    assert default_policy["optimizer_kind"] == "pact_muon_adamw"
    assert default_policy["pr95_muon_policy"] == "every_stage"
    assert default_policy["effective_optimizer_label"] == "pr95_8stage_muon_adamw_every_stage"
    assert default_policy["optimizer_kind_consumed_by_native_mlx"] is False
    assert default_policy["optimizer_kind_consumed_by_pr95_curriculum"] is True
    assert default_policy["pr95_faithful_curriculum_enabled"] is True


def test_hinerv_auto_policy_keeps_short_pact_muon_adamw_smokes_native() -> None:
    smoke_policy = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=2,
        optimizer_kind="pact_muon_adamw",
    )

    assert smoke_policy["resolved_policy"] == "native_optimizer"
    assert smoke_policy["optimizer_kind"] == "pact_muon_adamw"
    assert smoke_policy["pr95_muon_policy"] is None
    assert smoke_policy["effective_optimizer_label"] == "pact_muon_adamw"
    assert smoke_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert smoke_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert smoke_policy["pr95_faithful_curriculum_enabled"] is False


def test_hinerv_optimizer_controls_default_to_pact_muon_adamw() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind="pact_muon_adamw",
        requested_weight_decay=None,
        grad_clip_max_norm=1.0,
        warmup_epochs=0,
        warmup_steps_per_epoch=1,
        cosine_decay_enabled=False,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=1e-2,
        run_epochs=29_650,
    )

    assert controls["optimizer_kind"] == "pact_muon_adamw"
    assert controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert controls["weight_decay_defaulted"] is True
    assert controls["borrowed_pr95_partition_rule"] is True
    assert controls["original_pact_default_optimizer"] is True


def test_hinerv_optimizer_controls_refuse_silent_decay_drop() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="optimizer-weight-decay is only supported",
    ):
        runner_mod._resolve_mlx_score_aware_optimizer_controls(
            optimizer_kind="adamax",
            requested_weight_decay=1.0e-4,
            grad_clip_max_norm=1.0,
            warmup_epochs=0,
            warmup_steps_per_epoch=1,
            cosine_decay_enabled=False,
            cosine_decay_total_epochs=None,
            cosine_decay_min_lr_ratio=1e-2,
            run_epochs=64,
        )


@pytest.mark.skipif(
    not _MLX_AVAILABLE or not _AV_AVAILABLE,
    reason="MLX and PyAV runtime video decode are required for HiNeRV adapter smoke",
)
def test_hinerv_execute_runs_training_archive_and_receiver_proof(
    tmp_path: Path,
) -> None:
    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=1,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        allow_unscored_research_smoke=True,
        coder_aware_qat=True,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_c1a_entropy_weight=0.0001,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_hi_nerv_mlx_scoreaware_and_exported"
    assert out["execute_family"] == "hi_nerv"
    assert out["training_executed"] is True
    assert out["adapter_smoke_only"] is False
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["archive_path"]).is_file()
    assert out["archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert len(out["archive_sha256"]) == 64
    assert out["scorer_upstream_snapshot"]["modules_py_exists"] is True
    assert out["scorer_upstream_snapshot"]["posenet_safetensors_exists"] is True
    assert out["scorer_upstream_snapshot"]["segnet_safetensors_exists"] is True
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    proof = json.loads(Path(out["receiver_proof_report_paths"][0]).read_text())
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_contract_satisfied"] is True
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" in out["blockers"]
    readiness = out["short_scorer_teacher_smoke_readiness"]
    assert Path(readiness["report_path"]).is_file()
    assert readiness["ready_for_long_run"] is False
    assert "hi_nerv_short_smoke_unscored_research_smoke_enabled" in readiness["actionable_blockers"]
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in out["blockers"]
    assert out["score_aware_training"]["optimizer_kind"] == "pact_muon_adamw"
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == ("native_optimizer")
    assert out["score_aware_training"]["optimizer_controls"]["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert "hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs" not in out["blockers"]
    assert "hi_nerv_receiver_proof_missing" not in out["blockers"]
    assert "local_cpu_replay_not_run_partial_pair_coverage" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert Path(out["report_path"]).is_file()


def test_snerv_execution_writes_archive_bound_report_and_reusable_hooks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    captured_advisory_kwargs: dict[str, object] = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_advisory_kwargs.update(kwargs)
        assert kwargs["n_pairs"] == 2

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=2,
            source_pair_indices=(7, 2),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet=str(kwargs["wavelet"]),
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        package_path = package_dir / "archive_bound_candidate_adapter_package.json"
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        package_path.write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_gate",
        num_pairs=2,
        epochs=3,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-unit-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "mfu_scales": (1, 5),
            "hfr_gain": 0.375,
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "snerv",
                "control_semantics": ("manual_receiver_visible_fc_dim_feature_basis"),
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        prioritized_pair_indices=(7, 2, 7),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_snerv_archive_bound_advisory_and_exported"
    assert out["execute_family"] == "snerv"
    assert out["training_executed"] is False
    assert captured_advisory_kwargs["levels"] == 2
    assert captured_advisory_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_advisory_kwargs["step_map_coder_mode"] == "waterfill"
    assert captured_advisory_kwargs["step_map_waterfill_bits_per_coeff"] == 0.5
    assert captured_advisory_kwargs["decoder_payload_codec"] == "int2_symmetric"
    assert captured_advisory_kwargs["snerv_mfu_scales"] == (1, 5)
    assert captured_advisory_kwargs["snerv_hfr_gain"] == 0.375
    assert captured_advisory_kwargs["pair_indices"] == (7, 2)
    selection = out["modelsize_candidate_selection"]
    assert selection["selection_mode"] == "planner_candidate"
    assert selection["candidate"]["candidate_id"] == "snerv-unit-candidate"
    assert selection["modelsize_control_contract"]["family"] == "snerv"
    assert selection["modelsize_control_contract"]["control_semantics"] == (
        "manual_receiver_visible_fc_dim_feature_basis"
    )
    assert selection["modelsize_control_contract"]["archive_bytes_authority_required"] is True
    assert selection["launch_levels"] == 2
    assert selection["launch_bits_per_coeff"] == 1.5
    assert selection["launch_decoder_payload_codec"] == "int2_symmetric"
    assert selection["candidate_curriculum_plan"]["receiver_grammar_controls"]["step_map_coder_mode"] == "waterfill"
    assert Path(out["archive_path"]).is_file()
    assert out["archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert Path(out["receiver_archive_packet_path"]).read_bytes() == packet
    assert Path(out["advisory_report_path"]).is_file()
    assert Path(out["runtime_package_path"]).is_file()
    assert Path(out["trained_ladder_row_payload_path"]).is_file()
    assert out["trained_ladder_row_payload"]["schema"] == ("nerv_trained_ladder_row_payload.v1")
    assert out["trained_ladder_row_payload"]["status"] == ("trained_ladder_row_blocked")
    assert out["trained_ladder_row_payload"]["archive_path_kind"] == ("contest_archive_zip")
    assert "sample_pair_count_below_full600" in out["trained_ladder_row_payload"]["blockers"]
    assert out["receiver_proof_report_paths"]
    planner = out["score_aware_carrier_training_plan"]
    assert planner["score_aware_training_ready"] is False
    native_contract = out["snerv_mlx_native_adapter_contract"]
    assert native_contract["schema"] == "snerv_mlx_native_adapter_contract.v1"
    assert native_contract["surfaces_ready"] is True
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in (native_contract["blockers"])
    assert out["score_aware_training"]["status"] == ("executed_cpu_advisory_mlx_native_training_missing")
    assert out["score_aware_training"]["target_bits_per_coeff"] == 1.5
    assert out["score_aware_training"]["step_map_coder_mode"] == "waterfill"
    assert out["score_aware_training"]["decoder_payload_codec"] == "int2_symmetric"
    assert out["score_aware_training"]["source_pair_indices"] == [7, 2]
    assert out["score_aware_training"]["prioritized_pair_training"]["enabled"] is True
    assert out["score_aware_training"]["prioritized_pair_training"]["pair_indices"] == [
        7,
        2,
    ]
    assert out["score_aware_training"]["prioritized_pair_training"]["consumed_by_cpu_advisory"] is True
    assert out["score_aware_training"]["prioritized_pair_training"]["consumed_by_mlx_native_export"] is False
    assert out["score_aware_training"]["prioritized_pair_training"]["score_claim"] is False
    assert out["score_aware_training"]["prioritized_pair_training"]["promotion_eligible"] is False
    assert out["score_aware_training"]["prioritized_pair_training"]["ready_for_exact_eval_dispatch"] is False
    feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert feedback["measured_payload_bytes"] == len(packet)
    assert feedback["measured_archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert "partial_pair_byte_feedback_only" in out["blockers"]
    candidate_feedback = out["candidate_feedback"]
    assert Path(candidate_feedback["row_path"]).is_file()
    assert Path(candidate_feedback["ledger_path"]).is_file()
    assert candidate_feedback["row"]["candidate_id"] == "snerv-unit-candidate"
    assert candidate_feedback["row"]["candidate_num_pairs"] == 600
    assert candidate_feedback["row"]["measured_num_pairs"] == 2
    assert candidate_feedback["row"]["feedback_ready"] is False
    assert candidate_feedback["score_claim"] is False
    binary_profile = out["snerv_binary_profile"]
    assert binary_profile["profile_written"] is True
    assert Path(binary_profile["profile_path"]).is_file()
    assert binary_profile["snar1_packet_bytes"] == len(packet)
    assert binary_profile["lf_payload_bytes"] > 0
    assert binary_profile["score_claim"] is False
    assert out["score_aware_training"]["beats_frontier_rate"] is True
    assert out["reusable_optimization_followups"]["applies_after_byte_closed_export"] is True
    assert "final_rate_attack_and_repair_materializers" in out["reusable_optimization_followups"]["required_hooks"]
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert post_export["queue_launch_executed"] is False
    assert post_export["experiment_count"] > 0
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert post_export["archive_record"]["source_runtime_dir"].endswith("/snerv_archive_bound_package/submission")
    assert post_export["archive_record"]["source_inflate_sh_path"].endswith(
        "/snerv_archive_bound_package/submission/inflate.sh"
    )
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith("/snerv_archive_bound_package/submission")
    assert first_context["packet_member_merge_source_runtime_dir"].endswith("/snerv_archive_bound_package/submission")
    queue = json.loads(Path(post_export["experiment_queue_path"]).read_text())
    harvest_step = queue["experiments"][0]["steps"][1]
    state_arg_index = harvest_step["command"].index("--state") + 1
    assert harvest_step["command"][state_arg_index] == post_export["experiment_queue_state_path"]
    post_export_execution = out["post_export_materializer_execution"]
    assert post_export_execution["requested"] is False
    assert post_export_execution["executed"] is False
    assert Path(post_export_execution["execution_path"]).is_file()
    assert (
        out["reusable_optimization_followups"]["post_export_experiment_queue_path"]
        == post_export["experiment_queue_path"]
    )
    assert out["candidate_curriculum_plan"]["training_plan"]["receiver_proof_attached"] is True
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out["blockers"]
    assert out["snerv_mlx_native_adapter_contract"]["surfaces_ready"] is True
    assert "snerv_receiver_proof_missing" not in out["blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]


def test_execute_snerv_attaches_native_mlx_export_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    native_calls: list[dict[str, object]] = []

    def fake_run_snerv_advisory(**kwargs):
        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=int(kwargs["n_pairs"]),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=kwargs["snerv_fc_dim"],
            snerv_emb_size=kwargs["snerv_emb_size"],
            snerv_patch_radius=kwargs["snerv_patch_radius"],
            snerv_model_size_adapter=kwargs["snerv_model_size_adapter"],
            snerv_mfu_scales=kwargs["snerv_mfu_scales"],
            snerv_hfr_gain=kwargs["snerv_hfr_gain"],
            snerv_temporal_context=kwargs["snerv_temporal_context"],
            snerv_temporal_mode=kwargs["snerv_temporal_mode"],
            decoder_feature_count=(int(kwargs["snerv_fc_dim"]) + int(kwargs["snerv_emb_size"])),
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "runtime_consumption_proof_passed": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {"candidate_rows": [row]},
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_train_export_snerv_mlx_native(**kwargs):
        native_calls.append(dict(kwargs))
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        packet_path = out / "snerv_mlx_native_packet.snar"
        packet_path.write_bytes(b"native-snar")
        archive = out / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", b"native-snar")
        proof = out / "receiver_proof.json"
        proof.write_text(
            json.dumps({"receiver_contract_satisfied": True}),
            encoding="utf-8",
        )
        report = out / "snerv_mlx_native_train_export.json"
        long_training_executed = int(kwargs.get("score_aware_long_training_epochs") or 0) > 0
        real_teachers_bound = bool(
            float(kwargs.get("segnet_distillation_weight") or 0.0) > 0.0
            and float(kwargs.get("pose_distillation_weight") or 0.0) > 0.0
        )
        scorer_input_distribution_guard_bound = bool(
            float(kwargs.get("score_aware_long_training_scorer_input_distribution_guard_weight") or 0.0) > 0.0
        )
        modelsize_candidate = dict(kwargs.get("modelsize_candidate") or {})
        omit_top_level_long_training_controls = bool(
            modelsize_candidate.get("test_omit_top_level_score_aware_long_training_controls")
        )
        hard_byte_ceiling = int(modelsize_candidate.get("hard_byte_ceiling") or 0)
        byte_cap_control = {
            "schema": "snerv_mlx_native_hard_byte_ceiling_control.v1",
            "attached": hard_byte_ceiling > 0,
            "hard_byte_ceiling": hard_byte_ceiling or None,
            "packet_bytes": packet_path.stat().st_size,
            "archive_bytes": archive.stat().st_size,
            "under_hard_byte_ceiling": (archive.stat().st_size <= hard_byte_ceiling if hard_byte_ceiling > 0 else None),
            "delta_bytes_vs_hard_byte_ceiling": (
                archive.stat().st_size - hard_byte_ceiling if hard_byte_ceiling > 0 else None
            ),
            "enforced": hard_byte_ceiling > 0,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        payload = {
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": report.as_posix(),
            "packet_path": packet_path.as_posix(),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet_path),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "byte_cap_control": byte_cap_control,
            "step_map_packet_schema": "snerv_step_map_coder.adaptive.v1",
            "step_map_coder_mode": "waterfill_mlx_native_uniform_importance_bridge",
            "step_map_waterfill_bits_per_coeff": 0.5,
            "step_map_coder_groups": [
                {
                    "group_name": "unit_step_maps",
                    "bits_per_coeff": 0.5,
                    "coeff_count": 64,
                }
            ],
            "runtime_submission_dir": (out / "submission").as_posix(),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_target_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_source_targets",
                    reference_kind="source_targets_nchw255",
                    mse=2.5,
                    max_abs=4.0,
                )
            ),
            "receiver_export_reconstruction_profile": (
                _fake_snerv_receiver_reconstruction_profile(
                    profile_id="selected_packet_vs_export_reference",
                    reference_kind="export_reference_nchw255",
                    mse=0.125,
                    max_abs=0.75,
                )
            ),
            "native_mlx_training_executed": int(kwargs.get("native_mlx_decoder_train_steps") or 0) > 0
            or long_training_executed,
            "native_mlx_training_kind": (
                "snerv_mlx_score_aware_haar_renderer"
                if long_training_executed
                else "full_batch_hf_decoder_gradient_descent"
            ),
            "score_aware_long_training_executed": False,
            "score_aware_long_training_real_teachers_bound": real_teachers_bound,
            "score_aware_long_training_has_real_segnet_teacher": real_teachers_bound,
            "score_aware_long_training_has_real_posenet_teacher": real_teachers_bound,
            "score_aware_long_training_scorer_input_distribution_guard_bound": False,
            "score_aware_long_training_kind": (
                "snerv_mlx_score_aware_haar_renderer" if long_training_executed else "none"
            ),
            "score_aware_long_training_telemetry_contract_passed": (long_training_executed),
            "score_aware_long_training_control_bound": long_training_executed,
            "score_aware_long_training": {
                "schema": "snerv_mlx_score_aware_long_training_attachment.v1",
                "executed": long_training_executed,
                "scorer_input_distribution_guard_bound": (scorer_input_distribution_guard_bound),
                "scorer_input_contrast_floor_bound": bool(
                    float(kwargs.get("score_aware_long_training_scorer_input_contrast_floor_weight") or 0.0) > 0.0
                ),
                "training_telemetry_contract": {
                    "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
                    "passed": long_training_executed,
                    "expected_scorer_input_guard_metric": long_training_executed,
                    "scorer_input_guard_metric_observed": long_training_executed,
                    "scorer_input_guard_dual_metric_observed": (long_training_executed),
                    "expected_scorer_input_contrast_floor_metric": (long_training_executed),
                    "scorer_input_contrast_floor_metric_observed": (long_training_executed),
                    "scorer_input_contrast_floor_segnet_ratio_metric_observed": (long_training_executed),
                    "scorer_input_contrast_floor_posenet_ratio_metric_observed": (long_training_executed),
                    "blockers": ([] if long_training_executed else ["snerv_score_aware_long_training_not_executed"]),
                },
                "requested_epochs": int(kwargs.get("score_aware_long_training_epochs") or 0),
                "learning_rate": float(kwargs.get("score_aware_long_training_lr") or 0.0),
                "batch_pairs": int(kwargs.get("score_aware_long_training_batch_pairs") or 0),
                "optimizer_kind": str(kwargs.get("score_aware_long_training_optimizer") or ""),
                "eval_roundtrip_ste_enabled": bool(kwargs.get("score_aware_long_training_eval_roundtrip_ste")),
                "coder_aware_qat_bound": True,
                "pr95_faithful_curriculum_enabled": bool(
                    kwargs.get("score_aware_long_training_pr95_faithful_curriculum")
                ),
                "pr95_muon_policy": str(kwargs.get("score_aware_long_training_pr95_muon_policy") or ""),
                "scorer_input_distribution_guard": {
                    "schema": "snerv_mlx_score_aware_scorer_input_distribution_guard.v1",
                    "enabled": bool(
                        float(kwargs.get("score_aware_long_training_scorer_input_distribution_guard_weight") or 0.0)
                        > 0.0
                    ),
                    "weight": float(
                        kwargs.get("score_aware_long_training_scorer_input_distribution_guard_weight") or 0.0
                    ),
                    "saturation_margin": float(
                        kwargs.get("score_aware_long_training_scorer_input_distribution_guard_saturation_margin") or 0.0
                    ),
                    "temperature": float(
                        kwargs.get("score_aware_long_training_scorer_input_distribution_guard_temperature") or 0.0
                    ),
                },
                "scorer_input_contrast_floor": {
                    "schema": "snerv_mlx_score_aware_scorer_input_contrast_floor.v1",
                    "enabled": bool(
                        float(kwargs.get("score_aware_long_training_scorer_input_contrast_floor_weight") or 0.0) > 0.0
                    ),
                    "weight": float(kwargs.get("score_aware_long_training_scorer_input_contrast_floor_weight") or 0.0),
                    "segnet_last_rgb_min_std_ratio": float(
                        kwargs.get("score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio") or 0.0
                    ),
                    "posenet_yuv6_pair_min_std_ratio": float(
                        kwargs.get("score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio")
                        or 0.0
                    ),
                },
                "has_real_segnet_teacher": real_teachers_bound,
                "has_real_posenet_teacher": real_teachers_bound,
                "teacher_binding": {
                    "schema": "snerv_mlx_real_scorer_teacher_binding.v1",
                    "requested": real_teachers_bound,
                    "segnet_distillation_weight": float(kwargs.get("segnet_distillation_weight") or 0.0),
                    "pose_distillation_weight": float(kwargs.get("pose_distillation_weight") or 0.0),
                    "pose_distillation_loss": str(kwargs.get("pose_distillation_loss") or "mse"),
                    "pose_distillation_huber_delta": float(kwargs.get("pose_distillation_huber_delta") or 1.0),
                    "has_real_segnet_teacher": real_teachers_bound,
                    "has_real_posenet_teacher": real_teachers_bound,
                    "allow_segnet_only_research": bool(kwargs.get("allow_segnet_only_research")),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "native_mlx_hf_decoder_training": {
                "schema": "snerv_native_mlx_hf_decoder_training.v1",
                "requested_steps": int(kwargs.get("native_mlx_decoder_train_steps") or 0),
                "learning_rate": float(kwargs.get("native_mlx_decoder_train_lr") or 0.0),
                "ridge": float(kwargs.get("native_mlx_decoder_train_ridge") or 0.0),
                "executed": int(kwargs.get("native_mlx_decoder_train_steps") or 0) > 0,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "scorer_loop_qat": {
                "executed": bool(kwargs.get("run_scorer_loop_qat")),
                "receiver_contract_satisfied": bool(kwargs.get("run_scorer_loop_qat")),
                "ready_for_pose_guard_gate": bool(kwargs.get("run_scorer_loop_qat")),
                "accepted_improvement": bool(kwargs.get("run_scorer_loop_qat")),
                "emitted_packet_uses_scorer_loop_best_decoder": False,
            },
            "num_pairs": int(kwargs["num_pairs"]),
            "source_pair_indices": [
                int(value) for value in (kwargs.get("pair_indices") or tuple(range(int(kwargs["num_pairs"]))))
            ],
            "blockers": [
                *([] if long_training_executed else ["snerv_mlx_score_aware_long_training_not_executed"]),
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if omit_top_level_long_training_controls:
            for key in (
                "score_aware_long_training_executed",
                "score_aware_long_training_real_teachers_bound",
                "score_aware_long_training_has_real_segnet_teacher",
                "score_aware_long_training_has_real_posenet_teacher",
                "score_aware_long_training_scorer_input_distribution_guard_bound",
                "score_aware_long_training_telemetry_contract_passed",
                "score_aware_long_training_control_bound",
            ):
                payload.pop(key, None)
        submission = out / "submission"
        submission.mkdir(exist_ok=True)
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        if kwargs.get("write_mlx_prefilter_profile"):
            profile = _write_mlx_prefilter_profile(
                out / "local_mlx_prefilter_profile.json",
                pairs=int(kwargs["num_pairs"]),
                batch_pairs=int(kwargs.get("mlx_prefilter_scorer_batch_pairs") or 1),
                score=0.1,
            )
            payload["local_mlx_prefilter_profile"] = {
                "schema": "snerv_mlx_native_prefilter_profile.v1",
                "written": True,
                "profile_path": profile.as_posix(),
                "profile_sha256": runner_mod._sha256_file(profile),
                "blockers": ["mlx_local_replay_not_contest_auth_axis"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            payload["local_mlx_prefilter_profile_path"] = profile.as_posix()
            payload["local_mlx_prefilter_progress_path"] = (out / "local_mlx_prefilter_progress.jsonl").as_posix()
        report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    class FakeQatResult:
        def as_jsonable(self) -> dict[str, object]:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 2,
                "scorer_loop_evaluations": 1,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": len(packet),
                    "archive_sha256": "3" * 64,
                    "score_linf": 12.0,
                },
                "best": {
                    "archive_bytes": len(packet) - 1,
                    "archive_sha256": "4" * 64,
                    "score_linf": 11.0,
                },
                "blockers": ["local_smoke_only_not_full_600_pairs"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    monkeypatch.setattr(
        native_mod,
        "train_export_snerv_mlx_native",
        fake_train_export_snerv_mlx_native,
    )
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )
    recon_weight_path = tmp_path / "joint_recon_weight.npy"
    np.save(recon_weight_path, np.ones((384, 512), dtype=np.float32))
    recon_weight_manifest_path = tmp_path / "joint_recon_weight_manifest.json"
    recon_weight_manifest_path.write_text("{}", encoding="utf-8")
    recon_weight_discovery = {
        "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
        "status": "selected_verified_joint_p18_p19_weight",
        "num_pairs": 2,
        "selected_manifest_path": recon_weight_manifest_path.as_posix(),
        "selected_manifest_sha256": runner_mod._sha256_file(recon_weight_manifest_path),
        "selected_weight_path": recon_weight_path.as_posix(),
        "selected_weight_sha256": runner_mod._sha256_file(recon_weight_path),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    def fake_discover_joint_recon_pixel_weight_path(**kwargs):
        assert kwargs["num_pairs"] == 2
        return recon_weight_path, recon_weight_discovery

    monkeypatch.setattr(
        runner_mod,
        "_discover_joint_recon_pixel_weight_path",
        fake_discover_joint_recon_pixel_weight_path,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_native_gate",
        num_pairs=2,
        epochs=3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-native-smoke",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "modelsize_mparams": 0.05,
            "official_modelsize_solution": {
                "schema": "official_snerv_modelsize_to_fc_dim.v1",
                "modelsize_mparams": 0.05,
                "fc_dim": 11,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "test_omit_top_level_score_aware_long_training_controls": True,
        },
        run_native_mlx_export=True,
        run_scorer_loop_qat=True,
        snerv_scorer_loop_max_trials=1,
        snerv_scorer_loop_search_mode="top_weight_coordinate",
        snerv_scorer_loop_qat_bits=4,
        snerv_scorer_loop_component_guard_mode="pose_seg_hard",
        auto_joint_recon_pixel_weight=True,
        recon_pixel_weight_normalize="none",
        snerv_native_mlx_decoder_train_steps=11,
        snerv_native_mlx_decoder_train_lr=0.004,
        snerv_native_mlx_decoder_train_ridge=0.0003,
        snerv_native_mlx_decoder_train_optimizer="adam",
        snerv_score_aware_long_training_epochs=13,
        snerv_score_aware_long_training_lr=0.002,
        snerv_score_aware_long_training_batch_pairs=2,
        snerv_score_aware_long_training_section_byte_refresh_every_steps=7,
        snerv_score_aware_long_training_optimizer="lion",
        snerv_score_aware_long_training_grad_clip_max_norm=0.5,
        snerv_score_aware_long_training_weight_decay=None,
        snerv_score_aware_long_training_eval_roundtrip_ste=True,
        scorer_input_distribution_guard_weight=0.375,
        scorer_input_distribution_guard_saturation_margin=0.03125,
        scorer_input_distribution_guard_temperature=0.015625,
        scorer_input_contrast_floor_weight=0.875,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.55,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.45,
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.41,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.42,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.43,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.44,
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=0.045,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=3.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.46,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.47,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=2.75,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.48,
        scorer_space_step_guard_max_post_pose_score_term=1.25,
        scorer_space_step_guard_max_post_pose_direct_live_score_term=0.052,
        scorer_space_step_guard_max_pose_score_term_relative_worsening=0.041,
        scorer_space_step_guard_max_pose_score_term_absolute_worsening=0.042,
        scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=0.043,
        scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=0.044,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.045,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=2.5,
        scorer_space_step_guard_backtracking_steps=7,
        scorer_space_step_guard_backtracking_shrink=0.55,
        snerv_official_skip_high_mode_override="shared_mean",
        segnet_distillation_weight=0.025,
        pose_distillation_weight=0.0025,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.25,
        segnet_distillation_objective="boundary_decision_tckd",
        distillation_temperature=3.0,
        segnet_student_live_calibration_weight=0.625,
        segnet_tau_boundary=0.75,
        segnet_hinge_margin=1.25,
        prioritized_pair_indices=(7, 2, 7),
        repo_root=REPO_ROOT,
    )

    assert native_calls
    assert native_calls[0]["num_pairs"] == 2
    assert native_calls[0]["pair_indices"] is None
    assert native_calls[0]["prioritized_pair_indices"] == (7, 2)
    assert native_calls[0]["run_scorer_loop_qat"] is True
    assert native_calls[0]["scorer_loop_qat_max_trials"] == 1
    assert native_calls[0]["scorer_loop_qat_search_mode"] == "top_weight_coordinate"
    assert native_calls[0]["scorer_loop_qat_qat_bits"] == 4
    assert native_calls[0]["scorer_loop_qat_component_guard_mode"] == "pose_seg_hard"
    assert native_calls[0]["scorer_loop_qat_pair_guard_min_score_improved_fraction"] == pytest.approx(1.0)
    assert native_calls[0]["scorer_loop_qat_pair_guard_max_pose_worsened_fraction"] == pytest.approx(0.0)
    assert native_calls[0]["scorer_loop_qat_decoder_payload_codec"] == ("int2_symmetric")
    assert native_calls[0]["scorer_loop_qat_lf_payload_codec"] == "portfolio_auto"
    assert native_calls[0]["native_mlx_decoder_train_steps"] == 11
    assert native_calls[0]["native_mlx_decoder_train_lr"] == pytest.approx(0.004)
    assert native_calls[0]["native_mlx_decoder_train_ridge"] == pytest.approx(0.0003)
    assert native_calls[0]["native_mlx_decoder_train_optimizer"] == "adam"
    assert native_calls[0]["score_aware_long_training_epochs"] == 13
    assert native_calls[0]["score_aware_long_training_lr"] == pytest.approx(0.002)
    assert native_calls[0]["score_aware_long_training_batch_pairs"] == 2
    assert native_calls[0]["score_aware_long_training_section_byte_refresh_every_steps"] == 7
    assert native_calls[0]["score_aware_long_training_optimizer"] == "lion"
    assert native_calls[0]["score_aware_long_training_grad_clip_max_norm"] == 0.5
    assert native_calls[0]["score_aware_long_training_weight_decay"] is None
    assert native_calls[0]["score_aware_long_training_eval_roundtrip_ste"] is True
    assert native_calls[0]["score_aware_long_training_scorer_input_distribution_guard_weight"] == pytest.approx(0.375)
    assert native_calls[0][
        "score_aware_long_training_scorer_input_distribution_guard_saturation_margin"
    ] == pytest.approx(0.03125)
    assert native_calls[0]["score_aware_long_training_scorer_input_distribution_guard_temperature"] == pytest.approx(
        0.015625
    )
    assert native_calls[0]["score_aware_long_training_scorer_input_contrast_floor_weight"] == pytest.approx(0.875)
    assert native_calls[0][
        "score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio"
    ] == pytest.approx(0.55)
    assert native_calls[0][
        "score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"
    ] == pytest.approx(0.45)
    assert native_calls[0]["score_aware_long_training_scorer_space_step_guard_enabled"] is True
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction"
    ] == pytest.approx(0.41)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction"
    ] == pytest.approx(0.42)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"
    ] == pytest.approx(0.43)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio"
    ] == pytest.approx(0.44)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop"
    ] == pytest.approx(0.045)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio"
    ] == pytest.approx(3.5)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae"
    ] == pytest.approx(0.46)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"
    ] == pytest.approx(0.47)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"
    ] == pytest.approx(2.75)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement"
    ] == pytest.approx(0.48)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term"
    ] == pytest.approx(1.25)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term"
    ] == pytest.approx(0.052)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening"
    ] == pytest.approx(0.041)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening"
    ] == pytest.approx(0.042)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening"
    ] == pytest.approx(0.043)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening"
    ] == pytest.approx(0.044)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening"
    ] == pytest.approx(0.045)
    assert native_calls[0][
        "score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening"
    ] == pytest.approx(2.5)
    assert native_calls[0]["score_aware_long_training_scorer_space_step_guard_backtracking_steps"] == 7
    assert native_calls[0]["score_aware_long_training_scorer_space_step_guard_backtracking_shrink"] == pytest.approx(
        0.55
    )
    assert native_calls[0]["segnet_distillation_weight"] == pytest.approx(0.025)
    assert native_calls[0]["pose_distillation_weight"] == pytest.approx(0.0025)
    assert native_calls[0]["pose_distillation_loss"] == "huber"
    assert native_calls[0]["pose_distillation_huber_delta"] == pytest.approx(2.25)
    assert native_calls[0]["segnet_distillation_objective"] == ("boundary_decision_tckd")
    assert native_calls[0]["distillation_temperature"] == pytest.approx(3.0)
    assert native_calls[0]["segnet_student_live_calibration_weight"] == pytest.approx(0.625)
    assert native_calls[0]["segnet_tau_boundary"] == pytest.approx(0.75)
    assert native_calls[0]["segnet_hinge_margin"] == pytest.approx(1.25)
    assert native_calls[0]["distillation_device"] == "cpu"
    assert native_calls[0]["allow_segnet_only_research"] is False
    assert native_calls[0]["modelsize_candidate"]["snerv_score_aware_long_training_epochs"] == 13
    assert (
        native_calls[0]["modelsize_candidate"]["snerv_score_aware_long_training_section_byte_refresh_every_steps"] == 7
    )
    assert native_calls[0]["modelsize_candidate"]["fc_dim"] == 11
    assert native_calls[0]["modelsize_candidate"]["fc_dim_source"] == ("official_modelsize_solution")
    assert native_calls[0]["modelsize_candidate"]["official_modelsize_solution"]["fc_dim"] == 11
    assert native_calls[0]["modelsize_candidate"]["official_skip_high_mode"] == ("shared_mean")
    assert native_calls[0]["modelsize_candidate"]["snerv_official_skip_high_mode"] == "shared_mean"
    assert native_calls[0]["modelsize_candidate"]["snerv_segnet_distillation_weight"] == pytest.approx(0.025)
    assert native_calls[0]["modelsize_candidate"]["snerv_pose_distillation_weight"] == pytest.approx(0.0025)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_distribution_guard_weight"
    ] == pytest.approx(0.375)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_distribution_guard_saturation_margin"
    ] == pytest.approx(0.03125)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_distribution_guard_temperature"
    ] == pytest.approx(0.015625)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_contrast_floor_weight"
    ] == pytest.approx(0.875)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio"
    ] == pytest.approx(0.55)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"
    ] == pytest.approx(0.45)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"
    ] == pytest.approx(0.43)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio"
    ] == pytest.approx(0.44)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop"
    ] == pytest.approx(0.045)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"
    ] == pytest.approx(0.47)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term"
    ] == pytest.approx(0.052)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening"
    ] == pytest.approx(0.044)
    assert (
        native_calls[0]["modelsize_candidate"][
            "snerv_score_aware_long_training_scorer_space_step_guard_backtracking_steps"
        ]
        == 7
    )
    assert Path(native_calls[0]["recon_pixel_weight_path"]) == recon_weight_path
    assert Path(native_calls[0]["recon_pixel_weight_manifest_path"]) == recon_weight_manifest_path
    assert native_calls[0]["recon_pixel_weight_normalize"] == "none"
    assert native_calls[0]["write_mlx_prefilter_profile"] is False
    assert native_calls[0]["mlx_prefilter_scorer_device"] == "cpu"
    assert native_calls[0]["mlx_prefilter_scorer_batch_pairs"] == 1
    native = out["snerv_mlx_native_export"]
    assert native["executed"] is True
    assert native["source_pair_indices"] == [0, 1]
    assert native["prioritized_pair_training"]["enabled"] is True
    assert native["prioritized_pair_training"]["pair_indices"] == [7, 2]
    assert native["prioritized_pair_training"]["sampling_scope"] == (
        "score_aware_training_batches_not_target_hydration"
    )
    assert native["prioritized_pair_training"]["consumed_by_native_mlx_train_export"] is True
    assert native["receiver_proof_passed"] is True
    assert native["receiver_contract_satisfied"] is True
    assert native["receiver_reconstruction_verified"] is True
    assert native["receiver_reconstruction"]["target_mse_nchw255"] == pytest.approx(2.5)
    assert native["receiver_target_reconstruction_mse_nchw255"] == pytest.approx(2.5)
    assert native["receiver_export_reconstruction_mse_nchw255"] == pytest.approx(0.125)
    assert native["byte_cap_control"]["schema"] == ("snerv_mlx_native_hard_byte_ceiling_control.v1")
    assert native["byte_cap_control"]["hard_byte_ceiling"] == 178_000
    assert native["byte_cap_control"]["archive_bytes"] == native["archive_bytes"]
    assert native["byte_cap_control"]["under_hard_byte_ceiling"] is True
    assert native["step_map_packet_schema"] == "snerv_step_map_coder.adaptive.v1"
    assert native["step_map_coder_mode"] == ("waterfill_mlx_native_uniform_importance_bridge")
    assert native["step_map_coder_groups"]
    assert native["native_mlx_training_executed"] is True
    assert native["native_mlx_training_kind"] == ("snerv_mlx_score_aware_haar_renderer")
    assert native["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert native["snerv_scorer_tether_smoke_gate"]["required"] is True
    assert native["score_aware_long_training_executed"] is True
    assert native["score_aware_long_training_telemetry_contract_passed"] is True
    assert native["score_aware_long_training_control_bound"] is True
    assert native["score_aware_long_training_real_teachers_bound"] is True
    assert native["score_aware_long_training_has_real_segnet_teacher"] is True
    assert native["score_aware_long_training_has_real_posenet_teacher"] is True
    assert native["score_aware_long_training_scorer_input_distribution_guard_bound"] is True
    assert native["score_aware_long_training_coder_qat_bound"] is True
    assert native["score_aware_long_training_pr95_curriculum_bound"] is True
    assert native["score_aware_long_training_pr95_muon_policy"] == ("faithful_stage8_only")
    assert native["native_mlx_hf_decoder_training"]["requested_steps"] == 11
    assert native["native_mlx_hf_decoder_training"]["learning_rate"] == pytest.approx(0.004)
    assert native["native_mlx_hf_decoder_training"]["ridge"] == pytest.approx(0.0003)
    assert native["scorer_loop_qat_attached"] is True
    assert native["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert native["scorer_loop_qat_ready_for_pose_guard_gate"] is True
    assert native["scorer_loop_qat_accepted_improvement"] is True
    assert native["scorer_loop_qat_best_materialized"] is False
    assert Path(native["artifact_report_path"]).is_file()
    recon_weight = out["snerv_recon_pixel_weight"]
    assert recon_weight["requested"] is True
    assert recon_weight["manifest_path"] == recon_weight_manifest_path.as_posix()
    assert recon_weight["auto_discovery"] == recon_weight_discovery
    assert recon_weight["enabled"] is False
    assert recon_weight["native_export_consumed"] is False
    assert recon_weight["primary_archive_consumed"] is False
    assert recon_weight["primary_archive_source"] == "snerv_native_mlx_export_direct"
    assert out["score_aware_training"]["mlx_native_train_export_attached"] is True
    assert out["score_aware_training"]["mlx_native_receiver_proof_passed"] is True
    assert out["score_aware_training"]["scorer_tether_smoke_gate"]["passed"] is True
    assert out["score_aware_training"]["mlx_native_receiver_reconstruction_verified"] is True
    assert out["score_aware_training"]["mlx_native_receiver_target_reconstruction_mse_nchw255"] == pytest.approx(2.5)
    assert out["score_aware_training"]["mlx_native_full600_export_verified"] is False
    top_prioritized = out["score_aware_training"]["prioritized_pair_training"]
    assert top_prioritized["consumed_by_mlx_native_export"] is True
    assert top_prioritized["mlx_native_export_blocker"] is None
    assert (
        out["score_aware_training"]["mlx_native_file_backed_export_evidence"]["file_backed_export_proof_passed"] is True
    )
    assert (
        out["score_aware_training"]["mlx_native_file_backed_export_evidence"][
            "required_pair_file_backed_export_proof_passed"
        ]
        is False
    )
    plan = out["candidate_curriculum_plan"]
    assert plan["training_plan"]["native_mlx_train_export_attached"] is True
    assert plan["training_plan"]["native_mlx_receiver_proof_passed"] is True
    assert plan["training_plan"]["native_mlx_file_backed_export_proof_passed"] is True
    assert plan["training_plan"]["native_mlx_required_full600_file_backed_export_proof_passed"] is False
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["native_mlx_real_segnet_teacher_bound"] is True
    assert plan["training_plan"]["native_mlx_real_posenet_teacher_bound"] is True
    assert plan["training_plan"]["native_mlx_joint_real_teachers_bound"] is True
    assert plan["training_plan"]["native_mlx_eval_roundtrip_ste_bound"] is True
    assert plan["training_plan"]["native_mlx_differentiable_pose_preprocess_bound"] is True
    assert plan["training_plan"]["native_mlx_long_training_bound"] is True
    assert plan["receiver_grammar_controls"]["step_map_coder_mode"] == "waterfill"
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" not in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" not in plan["blockers"]
    assert "snerv_qat_forward_missing" not in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" not in plan["blockers"]
    assert "snerv_candidate_curriculum_requires_waterfill_step_maps" not in plan["blockers"]
    assert plan["pr95_stack_binding"]["complete"] is False
    assert "snerv_native_scorer_loop_best_packet_not_materialized" in plan["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in plan["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out["blockers"]
    assert "snerv_mlx_native_receiver_proof_missing_or_failed" not in out["blockers"]
    assert "snerv_mlx_native_export_partial_pair_coverage" in out["blockers"]
    assert "snerv_mlx_native_arbitrary_pair_hydration_not_implemented" not in out["blockers"]
    assert "snerv_mlx_native_prioritized_pair_hydration_not_consumed" not in out["blockers"]
    assert "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" not in plan["blockers"]
    assert "snerv_mlx_native_longer_staged_training_not_executed" not in out["blockers"]


def test_snerv_native_attachment_preserves_official_binding_for_curriculum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod
    from tac.analysis.nerv_candidate_curriculum import (
        build_snerv_candidate_curriculum_plan,
    )

    def fake_train_export_snerv_mlx_native(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        packet_path = out / "snerv_mlx_native_packet.snar"
        packet_path.write_bytes(b"official-snar2-packet")
        archive_path = out / "archive.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", packet_path.read_bytes())
        report_path = out / "snerv_mlx_native_train_export.json"
        payload = {
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": report_path.as_posix(),
            "packet_path": packet_path.as_posix(),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet_path),
            "packet_source": "mlx_target_hydration_numpy_closed_form_decoder_fit",
            "archive_path": archive_path.as_posix(),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive_path),
            "num_pairs": 2,
            "source_pair_indices": [0, 1],
            "native_mlx_training_executed": False,
            "native_mlx_hf_decoder_training": {
                "schema": "snerv_native_mlx_hf_decoder_training.v1",
                "requested_steps": 0,
                "executed": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_aware_long_training": {
                "schema": "snerv_mlx_score_aware_long_training_attachment.v1",
                "executed": False,
                "blockers": ["snerv_mlx_score_aware_long_training_not_executed"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "receiver_proof_path": (out / "receiver_proof.json").as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
            "snerv_official_mfu_hfr_tub_export_bound": True,
            "snerv_official_mfu_hfr_tub_export_bound_semantics": ("receiver_payload_bound_not_source_forward_parity"),
            "snerv_official_mfu_hfr_tub_receiver_payload_bound": True,
            "snerv_official_mfu_hfr_tub_frame_producing_export": True,
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound": False,
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority": False,
            "snerv_official_mfu_hfr_tub_export_blockers": [
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
            ],
            "official_primitive_binding": {
                "schema": "snerv_official_mfu_hfr_tub_export_binding.v3",
                "official_receiver_payload_bound": True,
                "selected_packet_frame_producing_official_export": True,
                "selected_packet_official_payload_runtime_decode_authority": True,
                "blockers": [
                    "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                    "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "selected_official_authority": {
                "schema": "snerv_selected_packet_official_payload_authority.v1",
                "official_decoder_payload_selected": True,
                "frame_producing_official_export": True,
                "frame_decode_succeeded": True,
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "blockers": [
                "snerv_mlx_score_aware_long_training_not_executed",
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        Path(payload["receiver_proof_path"]).write_text(
            json.dumps({"receiver_contract_satisfied": True}),
            encoding="utf-8",
        )
        report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        native_mod,
        "train_export_snerv_mlx_native",
        fake_train_export_snerv_mlx_native,
    )

    attachment = runner_mod._run_snerv_native_mlx_export_attachment(
        requested=True,
        output_dir=tmp_path / "official_attachment",
        num_pairs=2,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        scorer_upstream_dir=REPO_ROOT / "upstream",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "official-binding-attachment",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 1.0,
            "decoder_payload_codec": "int8_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights={},
        scorer_error_pair_curriculum={},
        repo_root=REPO_ROOT,
        allow_overwrite=False,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=1,
        run_scorer_loop_qat=False,
        scorer_loop_qat_max_trials=0,
        scorer_loop_qat_search_mode="top_weight_coordinate",
        scorer_loop_qat_qat_bits=4,
        scorer_loop_qat_decoder_payload_codec="int8_symmetric",
        scorer_loop_qat_lf_payload_codec="portfolio_auto",
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
        scorer_loop_qat_pair_guard_min_score_improved_fraction=1.0,
        scorer_loop_qat_pair_guard_max_pose_worsened_fraction=0.0,
        scorer_loop_qat_device="cpu",
        scorer_loop_qat_perturb_scale=0.0,
        scorer_loop_qat_byte_pressure_multiplier=1.0,
        scorer_loop_qat_section_value_pressure_multiplier=1.0,
        scorer_loop_qat_max_archive_byte_growth=0,
        scorer_loop_qat_byte_growth_admission_mode="rate_paid",
        scorer_loop_qat_pose_slack=0.0,
        scorer_loop_qat_seg_slack=0.0,
        scorer_loop_qat_seed=1337,
        recon_pixel_weight_path=None,
        recon_pixel_weight_manifest_path=None,
        recon_pixel_weight_normalize="mean",
        native_mlx_decoder_train_steps=0,
        native_mlx_decoder_train_lr=1.0e-5,
        native_mlx_decoder_train_ridge=1.0e-6,
        native_mlx_decoder_train_optimizer="pact_guarded_adamw",
        score_aware_long_training_epochs=0,
        score_aware_long_training_lr=1.0e-3,
        score_aware_long_training_batch_pairs=2,
        score_aware_long_training_section_byte_refresh_every_steps=25,
        score_aware_long_training_optimizer="pact_muon_adamw",
        score_aware_long_training_grad_clip_max_norm=1.0,
        score_aware_long_training_weight_decay=1.0e-4,
        score_aware_long_training_eval_roundtrip_ste=False,
        score_aware_long_training_scorer_tether_smoke_steps=2,
        score_aware_long_training_scorer_input_distribution_guard_weight=0.0,
        score_aware_long_training_scorer_input_distribution_guard_saturation_margin=0.02,
        score_aware_long_training_scorer_input_distribution_guard_temperature=0.01,
        score_aware_long_training_scorer_input_contrast_floor_weight=0.0,
        score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio=0.5,
        score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.5,
        score_aware_long_training_scorer_input_shape_tether_weight=0.0,
        score_aware_long_training_posenet_temporal_signal_floor_weight=0.0,
        score_aware_long_training_posenet_temporal_signal_min_std_ratio=0.25,
        score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio=0.25,
        score_aware_long_training_loss_weights={},
        score_aware_long_training_pose_warmup_epochs=0,
        score_aware_long_training_scorer_input_shape_warmup_epochs=0,
        score_aware_long_training_segnet_direct_live_escape_warmup_epochs=0,
        checkpoint_retention_keep_last_n=None,
        checkpoint_retention_keep_best_n=0,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_student_live_calibration_weight=1.0,
        segnet_direct_live_distillation_weight=0.0,
        pose_direct_live_distillation_weight=0.0,
        segnet_direct_live_base_loss_weight=1.0,
        segnet_direct_live_class_histogram_weight=0.0,
        segnet_direct_live_class_balanced_hinge_weight=0.0,
        segnet_direct_live_class_balanced_ce_weight=0.0,
        segnet_direct_live_class_balanced_squared_hinge_weight=0.0,
        segnet_direct_live_class_region_recon_weight=0.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        distillation_device="cpu",
        allow_segnet_only_research=False,
        coder_aware_qat=False,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.0,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=512,
        score_aware_long_training_pr95_faithful_curriculum=False,
        score_aware_long_training_pr95_muon_policy="every_stage",
        write_mlx_prefilter_profile=False,
        mlx_prefilter_scorer_device="cpu",
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=50,
    )

    assert attachment["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert attachment["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert attachment["snerv_official_mfu_hfr_tub_frame_producing_export"] is True
    assert attachment["packet_source"] == "mlx_target_hydration_numpy_closed_form_decoder_fit"
    assert attachment["official_primitive_binding"]["official_receiver_payload_bound"] is True

    plan = build_snerv_candidate_curriculum_plan(
        candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "official-binding-attachment",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 1.0,
            "decoder_payload_codec": "int8_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        requested_epochs=0,
        num_pairs=2,
        step_map_coder_mode="waterfill",
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_artifact_evidence=attachment,
        measured_num_pairs=2,
    )
    official_split = plan["official_source_forward_authority_split"]
    assert official_split["export_bound"] is True
    assert official_split["receiver_payload_bound"] is True
    assert official_split["frame_producing_export"] is True
    assert "snerv_official_mfu_hfr_tub_export_not_bound" not in plan["blockers"]
    assert "snerv_official_mfu_hfr_tub_receiver_payload_not_bound" not in plan["blockers"]
    assert "snerv_official_mfu_hfr_tub_frame_producing_export_missing" not in plan["blockers"]
    assert "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority" in plan["blockers"]


def test_snerv_coder_aware_qat_executes_receiver_priced_scorer_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    captured_advisory_kwargs: dict[str, object] = {}
    captured_qat_kwargs: dict[str, object] = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_advisory_kwargs.update(kwargs)

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=int(kwargs["n_pairs"]),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            snerv_model_size_adapter=kwargs["snerv_model_size_adapter"],
            snerv_mfu_scales=kwargs["snerv_mfu_scales"],
            snerv_hfr_gain=kwargs["snerv_hfr_gain"],
            snerv_temporal_context=0,
            snerv_temporal_mode=kwargs["snerv_temporal_mode"],
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_run_snerv_scorer_loop_decoder_qat(**kwargs):
        captured_qat_kwargs.update(kwargs)
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                SimpleNamespace(
                    as_jsonable=lambda: {
                        "label": "fake_progress_eval",
                        "archive_bytes": 123,
                        "score_linf": 4.5,
                    }
                )
            )

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "improvement_score_delta": -0.01,
                "improvement_d_pose_delta": 0.0,
                "improvement_d_seg_delta": -0.001,
                "scorer_loop_evaluations": 3,
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 2,
                    "min_score_improved_fraction": 0.5,
                    "max_pose_worsened_fraction": 0.25,
                    "pose_slack": 0.001,
                    "score_improved_fraction": 1.0,
                    "pose_worsened_fraction": 0.0,
                    "permissive_guard": False,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(as_jsonable=as_jsonable)

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat",
        fake_run_snerv_scorer_loop_decoder_qat,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_qat_gate",
        num_pairs=2,
        epochs=3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-q-smoke",
            "wavelet": "haar",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "fc_dim": 11,
            "emb_size": 2,
            "patch_radius": 1,
            "temporal_context": 1,
            "temporal_mode": "official_haar_dwt1d_lowpass",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        snerv_spectra_preserving_adapter=True,
        snerv_mfu_scales=(1, 3),
        snerv_hfr_gain=0.25,
        run_scorer_loop_qat=True,
        snerv_scorer_loop_qat_bits=4,
        snerv_scorer_loop_max_trials=5,
        snerv_scorer_loop_search_mode="learned_random_subspace",
        snerv_scorer_loop_step_map_bins=8,
        snerv_scorer_loop_lf_payload_codec="auto",
        snerv_scorer_loop_perturb_scale=0.03,
        snerv_scorer_loop_byte_pressure_multiplier=1.25,
        snerv_scorer_loop_section_value_pressure_multiplier=1.75,
        snerv_scorer_loop_max_archive_byte_growth=77,
        snerv_scorer_loop_byte_growth_admission_mode="rate_paid",
        snerv_scorer_loop_pose_slack=0.001,
        snerv_scorer_loop_seg_slack=0.002,
        snerv_scorer_loop_pair_stride=3,
        snerv_scorer_loop_start_pair=7,
        snerv_scorer_loop_pair_guard_min_score_improved_fraction=0.5,
        snerv_scorer_loop_pair_guard_max_pose_worsened_fraction=0.25,
        snerv_scorer_loop_component_guard_mode="pose_hard",
        random_seed=123,
        repo_root=REPO_ROOT,
    )

    assert captured_advisory_kwargs["snerv_model_size_adapter"] == (SNERV_SPECTRA_PRESERVING_ADAPTER)
    assert captured_advisory_kwargs["wavelet"] == "haar"
    assert captured_advisory_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_advisory_kwargs["snerv_fc_dim"] == 11
    assert captured_advisory_kwargs["snerv_emb_size"] == 2
    assert captured_advisory_kwargs["snerv_temporal_context"] == 1
    assert captured_advisory_kwargs["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert captured_advisory_kwargs["snerv_mfu_scales"] == (1, 3)
    assert captured_advisory_kwargs["snerv_hfr_gain"] == 0.25
    assert captured_qat_kwargs["n_pairs"] == 2
    assert captured_qat_kwargs["levels"] == 2
    assert captured_qat_kwargs["wavelet"] == "haar"
    assert captured_qat_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_qat_kwargs["snerv_spectra_preserving_adapter"] is True
    assert captured_qat_kwargs["snerv_model_size_adapter"] == (SNERV_SPECTRA_PRESERVING_ADAPTER)
    assert captured_qat_kwargs["snerv_fc_dim"] == 11
    assert captured_qat_kwargs["snerv_emb_size"] == 2
    assert captured_qat_kwargs["snerv_temporal_context"] == 1
    assert captured_qat_kwargs["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert captured_qat_kwargs["snerv_mfu_scales"] == (1, 3)
    assert captured_qat_kwargs["snerv_hfr_gain"] == 0.25
    assert captured_qat_kwargs["qat_bits"] == 4
    assert captured_qat_kwargs["decoder_payload_codec"] == "int2_symmetric"
    assert captured_qat_kwargs["lf_payload_codec"] == "auto"
    assert captured_qat_kwargs["max_trials"] == 5
    assert captured_qat_kwargs["search_mode"] == "learned_random_subspace"
    assert captured_qat_kwargs["step_map_bins"] == 8
    assert captured_qat_kwargs["perturb_scale"] == pytest.approx(0.03)
    assert captured_qat_kwargs["byte_pressure_multiplier"] == 1.25
    assert captured_qat_kwargs["section_value_pressure_multiplier"] == pytest.approx(1.75)
    assert captured_qat_kwargs["max_archive_byte_growth"] == 77
    assert captured_qat_kwargs["byte_growth_admission_mode"] == "rate_paid"
    assert captured_qat_kwargs["pose_slack"] == 0.001
    assert captured_qat_kwargs["seg_slack"] == 0.002
    assert captured_qat_kwargs["pair_stride"] == 3
    assert captured_qat_kwargs["start_pair"] == 7
    assert captured_qat_kwargs["pair_guard_min_score_improved_fraction"] == 0.5
    assert captured_qat_kwargs["pair_guard_max_pose_worsened_fraction"] == 0.25
    assert captured_qat_kwargs["component_guard_mode"] == "pose_hard"
    assert captured_qat_kwargs["seed"] == 123
    assert callable(captured_qat_kwargs["progress_callback"])

    qat = out["snerv_scorer_loop_qat"]
    assert qat["executed"] is True
    assert qat["component_guard_mode"] == "pose_hard"
    assert qat["pair_robust_admission"]["schema"] == ("snerv_pair_robust_admission.v1")
    assert qat["pair_robust_admission"]["passed"] is True
    assert qat["lf_payload_codec"] == "auto"
    assert qat["perturb_scale"] == pytest.approx(0.03)
    assert qat["byte_pressure_multiplier"] == pytest.approx(1.25)
    assert qat["section_value_pressure_multiplier"] == pytest.approx(1.75)
    assert qat["max_archive_byte_growth"] == 77
    assert qat["byte_growth_admission_mode"] == "rate_paid"
    assert qat["pose_slack"] == pytest.approx(0.001)
    assert qat["seg_slack"] == pytest.approx(0.002)
    assert qat["seed"] == 123
    assert qat["accepted_improvement"] is True
    assert qat["receiver_contract_satisfied"] is True
    assert qat["ready_for_pose_guard_gate"] is True
    assert Path(qat["result_path"]).is_file()
    progress_path = Path(qat["progress_jsonl_path"])
    assert progress_path.is_file()
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["schema"] == "snerv_scorer_loop_decoder_qat_progress.v1"
    assert progress_payload["score_claim"] is False
    assert progress_payload["row"]["label"] == "fake_progress_eval"
    assert len(qat["progress_jsonl_sha256"]) == 64
    assert out["score_aware_training"]["scorer_loop_qat"]["executed"] is True
    assert out["score_aware_training"]["scorer_loop_component_guard_mode"] == "pose_hard"
    assert out["score_aware_training"]["status"] == (
        "executed_cpu_advisory_plus_receiver_priced_scorer_loop_qat_mlx_native_training_missing"
    )
    plan = out["candidate_curriculum_plan"]
    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert plan["training_plan"]["receiver_proof_attached"] is True
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_qat_forward_missing" in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" in plan["blockers"]
    assert "snerv_receiver_proof_missing" not in plan["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out["blockers"]
    assert "snerv_mlx_native_longer_staged_training_not_executed" in out["blockers"]
    assert "snerv_longer_staged_score_aware_training_not_executed" not in out["blockers"]


def test_snerv_runner_refuses_conflicting_candidate_adapter_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="modelsize candidate adapter conflicts",
    ):
        execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=tmp_path / "snerv_conflict",
            num_pairs=2,
            epochs=3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            modelsize_candidate={
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv-conflict",
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int2_symmetric",
                "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
                "fc_dim": 11,
                "emb_size": 2,
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 150_000,
                "nominal_under_ceiling": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            snerv_spectra_preserving_adapter=True,
            repo_root=REPO_ROOT,
        )


def test_snerv_runner_refuses_conflicting_skip_high_mode_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="official_skip_high_mode conflicts",
    ):
        execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=tmp_path / "snerv_skip_conflict",
            num_pairs=2,
            epochs=3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            modelsize_candidate={
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv-skip-conflict",
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int2_symmetric",
                "snerv_model_size_adapter": (SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER),
                "official_skip_high_mode": "full",
                "fc_dim": 11,
                "emb_size": 2,
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 150_000,
                "nominal_under_ceiling": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            snerv_official_skip_high_mode_override="shared_mean",
            repo_root=REPO_ROOT,
        )


def test_snerv_runner_refuses_implicit_full_skip_high_under_byte_ceiling(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="official_skip_high_mode='full' is diagnostic-only",
    ):
        execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=tmp_path / "snerv_full_skip_high_byte_ceiling",
            num_pairs=2,
            epochs=3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            modelsize_candidate={
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv-full-skip-high-byte-ceiling",
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int2_symmetric",
                "snerv_model_size_adapter": (SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER),
                "official_skip_high_mode": "full",
                "fc_dim": 11,
                "emb_size": 2,
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 150_000,
                "nominal_under_ceiling": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            repo_root=REPO_ROOT,
        )


def test_snerv_batched_full_video_mlx_prefilter_feeds_acquisition_not_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=600)

    def fake_run_snerv_advisory(**kwargs):
        assert kwargs["n_pairs"] == 600

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=600,
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=91.0,
            score_l2=92.0,
            d_seg_mean_linf=0.5,
            d_pose_mean_linf=160.0,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    batched_profile = _write_mlx_prefilter_profile(
        tmp_path / "snerv_batched_mlx_full600.json",
        pairs=600,
        batch_pairs=8,
        score=91.0,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_batched_gpu_gate",
        num_pairs=600,
        epochs=3,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        mlx_profile_paths=(batched_profile,),
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-full600-batched-profile",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        repo_root=REPO_ROOT,
    )

    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_profile_paths"] == []
    assert out["mlx_prefilter_coverage"]["blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" not in out["blockers"]
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in out["blockers"]
    feedback_blockers = out["candidate_feedback"]["row"]["pr95_stack_binding_blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" not in feedback_blockers
    feedback_row = out["candidate_feedback"]["row"]
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in feedback_row["blockers"]
    assert feedback_row["mlx_prefilter_has_full_video"] is True
    assert feedback_row["mlx_prefilter_local_replay_passed"] is False
    assert feedback_row["mlx_prefilter_blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert feedback_row["local_cpu_replay_gate_has_full_video_mlx_prefilter"] is True
    assert feedback_row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"] is False
    assert feedback_row["local_cpu_replay_gate_executed"] is False


def test_snerv_native_auto_mlx_prefilter_runs_cpu_replay_on_native_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=600)
    native_dir = tmp_path / "native_export"
    native_dir.mkdir()
    native_archive = native_dir / "archive.zip"
    with zipfile.ZipFile(native_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"native-snar")
    native_submission = native_dir / "submission"
    native_submission.mkdir()
    (native_submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    native_packet = native_dir / "snerv_mlx_native_packet.snar"
    native_packet.write_bytes(b"native-snar")
    native_proof = native_dir / "receiver_proof.json"
    native_proof.write_text(
        json.dumps(
            {
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "blockers": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    native_prefilter = _write_mlx_prefilter_profile(
        native_dir / "local_mlx_prefilter_profile.json",
        pairs=600,
        batch_pairs=1,
        score=0.1,
    )

    def fake_native_attachment(**kwargs):
        assert kwargs["write_mlx_prefilter_profile"] is True
        assert kwargs["mlx_prefilter_scorer_device"] == "cpu"
        assert kwargs["mlx_prefilter_scorer_batch_pairs"] == 1
        return {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "executed": True,
            "requested": True,
            "num_pairs": 600,
            "source_pair_indices": list(range(600)),
            "artifact_report_path": (native_dir / "report.json").as_posix(),
            "packet_path": native_packet.as_posix(),
            "packet_bytes": native_packet.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(native_packet),
            "archive_path": native_archive.as_posix(),
            "archive_bytes": native_archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(native_archive),
            "runtime_submission_dir": native_submission.as_posix(),
            "receiver_proof_path": native_proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_training_executed": True,
            "native_mlx_training_kind": "snerv_mlx_score_aware_haar_renderer",
            "score_aware_long_training_executed": True,
            "score_aware_long_training_real_teachers_bound": True,
            "score_aware_long_training_has_real_segnet_teacher": True,
            "score_aware_long_training_has_real_posenet_teacher": True,
            "local_mlx_prefilter_profile": {
                "schema": "snerv_mlx_native_prefilter_profile.v1",
                "written": True,
                "profile_path": native_prefilter.as_posix(),
                "profile_sha256": runner_mod._sha256_file(native_prefilter),
                "blockers": ["mlx_local_replay_not_contest_auth_axis"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "local_mlx_prefilter_profile_path": native_prefilter.as_posix(),
            "local_mlx_prefilter_progress_path": (native_dir / "local_mlx_prefilter_progress.jsonl").as_posix(),
            "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_run_snerv_advisory(**kwargs):
        assert kwargs["n_pairs"] == 600

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=600,
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="haar",
            score_linf=91.0,
            score_l2=92.0,
            d_seg_mean_linf=0.5,
            d_pose_mean_linf=160.0,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        advisory_archive = package_dir / "archive.zip"
        with zipfile.ZipFile(advisory_archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        row = {
            "candidate_archive_path": advisory_archive.as_posix(),
            "candidate_archive_bytes": advisory_archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(advisory_archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        return {
            "archive_bound_candidate_adapter_package": {"candidate_rows": [row]},
            "receiver_proof": {
                "proof_path": (package_dir / "proof.json").as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    staged_calls: list[dict[str, object]] = []

    def fake_stage_local_replay_submission(**kwargs):
        staged_calls.append(kwargs)
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 0.2,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    monkeypatch.setattr(
        runner_mod,
        "_run_snerv_native_mlx_export_attachment",
        fake_native_attachment,
    )
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        lambda **_kwargs: FakeReplaySummary(),
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_native_full600_gate",
        num_pairs=600,
        epochs=3,
        hard_byte_ceilings=(178_000,),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-full600-native-auto-prefilter",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        run_native_mlx_export=True,
        snerv_score_aware_long_training_epochs=1,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        mlx_prefilter_scorer_device="cpu",
        mlx_prefilter_scorer_batch_pairs=1,
        mlx_prefilter_progress_every=3,
        repo_root=REPO_ROOT,
    )

    assert out["auto_mlx_prefilter_profile_path"] == native_prefilter.as_posix()
    assert out["mlx_profile_paths"] == [native_prefilter.as_posix()]
    assert out["local_cpu_replay_gate"]["executed"] is True
    assert out["local_cpu_replay_gate"]["archive_source"] == ("snerv_mlx_native_export_archive")
    assert out["local_cpu_replay_gate"]["archive_path"] == native_archive.as_posix()
    assert out["local_cpu_replay_gate"]["runtime_submission_dir"] == (native_submission.as_posix())
    assert staged_calls
    assert Path(staged_calls[0]["archive_zip_path"]) == native_archive
    assert Path(staged_calls[0]["runtime_submission_dir"]) == native_submission
    feedback = out["candidate_feedback"]["row"]
    assert feedback["mlx_prefilter_has_full_video"] is True
    assert feedback["mlx_prefilter_local_replay_passed"] is True
    assert feedback["local_cpu_replay_gate_executed"] is True


def test_snerv_long_campaign_refuses_when_pr95_prelaunch_gate_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod

    def fail_run_snerv_advisory(**_kwargs):
        raise AssertionError("long campaign should refuse before SNeRV advisory")

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fail_run_snerv_advisory)

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_refusal",
        num_pairs=600,
        epochs=8,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-long-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "snerv_pr95_binding_prelaunch_refused"
    assert out["training_executed"] is False
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in out["blockers"]
    assert "snerv_real_segnet_teacher_missing" in out["blockers"]
    gate = out["candidate_curriculum_plan"]["long_campaign_prelaunch_gate"]
    assert gate["launch_allowed"] is False
    assert "local_cpu_replay_gate" in gate["post_run_requirements_excluded"]
    assert out["candidate_feedback"]["row"]["candidate_id"] == ("snerv-long-candidate")
    assert out["candidate_feedback"]["row"]["long_campaign_prelaunch_launch_allowed"] is False
    assert Path(out["report_path"]).is_file()


def test_snerv_native_export_bypasses_pr95_prelaunch_only_for_local_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod

    class FakeAdvisory:
        receiver_archive_packet = b"SNAR1-unit"
        archive_bytes_total = 10
        levels = 2
        wavelet = "haar"
        score_linf = 1.0
        score_l2 = 1.1
        d_seg_mean_linf = 0.01
        d_pose_mean_linf = 0.02
        beats_frontier_rate = True
        receiver_archive_replay_verified = True

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "archive_bytes_total": 10,
                "levels": 2,
                "wavelet": "haar",
                "score_linf": 1.0,
                "score_l2": 1.1,
                "d_seg_mean_linf": 0.01,
                "d_pose_mean_linf": 0.02,
                "beats_frontier_rate": True,
                "receiver_archive_replay_verified": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    def fake_run_snerv_advisory(**_kwargs):
        return FakeAdvisory()

    def fake_package(**_kwargs):
        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"zip")
        return {
            "schema": "snerv_inverse_steg_archive_bound_adapter_package.v1",
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [
                    {
                        "candidate_archive_path": archive.as_posix(),
                        "candidate_archive_bytes": archive.stat().st_size,
                        "candidate_archive_sha256": "a" * 64,
                    }
                ],
            },
            "receiver_proof": {
                "proof_path": (tmp_path / "proof.json").as_posix(),
                "receiver_contract_satisfied": True,
            },
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": "a" * 64,
            "submission_dir": (tmp_path / "submission").as_posix(),
            "receiver_proof_path": (tmp_path / "proof.json").as_posix(),
            "receiver_contract_satisfied": True,
            "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_native_export(**_kwargs):
        report = tmp_path / "native_report.json"
        packet = tmp_path / "native_packet.snar"
        archive = tmp_path / "native_archive.zip"
        proof = tmp_path / "native_receiver_proof.json"
        report.write_text(
            '{"schema":"snerv_mlx_native_train_export.v1"}\n',
            encoding="utf-8",
        )
        packet.write_bytes(b"native packet")
        archive.write_bytes(b"native archive")
        proof.write_text(
            json.dumps(
                {
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_passed": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "schema": "snerv_mlx_native_train_export.v1",
            "num_pairs": 600,
            "artifact_report_path": report.as_posix(),
            "packet_path": packet.as_posix(),
            "packet_bytes": packet.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": True,
            "scorer_loop_qat": {
                "requested": False,
                "executed": False,
                "blockers": ["snerv_scorer_loop_qat_not_requested"],
            },
            "blockers": ["snerv_mlx_score_aware_long_training_not_executed"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_package,
    )
    monkeypatch.setattr(native_mod, "train_export_snerv_mlx_native", fake_native_export)

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_native_proof",
        num_pairs=600,
        epochs=8,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-long-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        run_native_mlx_export=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_snerv_archive_bound_advisory_and_exported"
    assert out["execute_family"] == "snerv"
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in out["blockers"]
    assert out["candidate_curriculum_plan"]["training_plan"]["native_mlx_long_training_bound"] is False
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in out["candidate_curriculum_plan"]["blockers"]
    )
    native = out["snerv_mlx_native_export"]
    assert native["native_mlx_full600_export_proof_ready"] is True
    assert native["native_mlx_full600_campaign_ready"] is False
    assert "snerv_mlx_native_export_closed_form_not_training" in native["blockers"]
    assert "snerv_mlx_native_full600_not_campaign_ready_without_learned_training" in native["blockers"]
    assert out["score_aware_training"]["mlx_native_training_required_next"] is True
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_adapt_snerv_advisory_report_consumes_existing_runtime_package(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    archive = _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    archive_sha = runner_mod._sha256_file(archive)
    archive_bytes = archive.stat().st_size
    proof_path = package_dir / "snerv_inverse_steg_receiver_proof.json"
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "runtime_package_dir": package_dir.as_posix(),
                "runtime_package": {
                    "archive_bound_candidate_adapter_package": {
                        "candidate_rows": [
                            {
                                "candidate_archive_path": archive.as_posix(),
                                "candidate_archive_sha256": archive_sha,
                                "candidate_archive_bytes": archive_bytes,
                                "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
                                "score_claim": False,
                                "promotion_eligible": False,
                                "ready_for_exact_eval_dispatch": False,
                            }
                        ]
                    },
                    "receiver_proof": proof_payload,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "adapted_snerv_advisory_report_to_spine"
    assert out["source_snerv_advisory_report_sha256"] == runner_mod._sha256_file(report_path)
    post_export = out["post_export_materializer_plan"]
    assert post_export["compiled"] is True
    assert post_export["archive_record"]["source_runtime_dir"].endswith("/snerv_package/submission")
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith("/snerv_package/submission")
    assert (
        "--source-runtime-dir"
        in json.loads(Path(post_export["materializer_work_queue_path"]).read_text())["rows"][0]["command"]
    )
    assert out["post_export_materializer_execution"]["requested"] is False
    assert "paired_contest_cpu_cuda_auth_eval_missing" in out["blockers"]
    assert out["source_parity_contract"]["schema"] == ("nerv_source_parity_contract.v1")
    assert out["source_parity_required_for_long_training_ready"] is True
    assert out["source_parity_blockers"] == []
    assert "source_parity:snerv_official_mfu_hfr_tub_parity_missing" in out["source_parity_nonblocking_gaps"]
    legacy_contract = out["legacy_advisory_ingest_contract"]
    assert legacy_contract["schema"] == "snerv_legacy_advisory_ingest_contract.v1"
    assert legacy_contract["source_parity_consumed"] is True
    assert legacy_contract["legacy_advisory_is_not_score_authority"] is True
    assert legacy_contract["score_claim"] is False
    assert legacy_contract["ready_for_exact_eval_dispatch"] is False


def test_adapt_snerv_advisory_report_uses_package_dir_archive_fallback(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    proof_path = package_dir / "snerv_inverse_steg_receiver_proof.json"
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "runtime_package_dir": package_dir.as_posix(),
                "runtime_package": {
                    "archive_bound_candidate_adapter_package": {"candidate_rows": []},
                    "receiver_proof": proof_payload,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    post_export = out["post_export_materializer_plan"]
    assert post_export["compiled"] is True
    assert post_export["archive_record"]["absolute_path"].endswith("/snerv_package/archive.zip")
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out["blockers"]


def test_adapt_snerv_advisory_report_accepts_archive_package_manifest(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    archive = _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    archive_sha = runner_mod._sha256_file(archive)
    archive_bytes = archive.stat().st_size
    proof_path = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
    proof_path.parent.mkdir()
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    package_manifest = package_dir / "archive_bound_candidate_adapter_package.json"
    package_manifest.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_archive_bound_candidate_package.v1",
                "archive_bound_candidate_adapter_package": {
                    "candidate_rows": [
                        {
                            "candidate_archive_path": archive.as_posix(),
                            "candidate_archive_sha256": archive_sha,
                            "candidate_archive_bytes": archive_bytes,
                            "runtime_adapter_manifest": {"n_pairs": 600},
                            "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
                            "score_claim": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        }
                    ]
                },
                "receiver_proof": proof_payload,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=package_manifest,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    assert out["source_snerv_advisory_report_sha256"] == runner_mod._sha256_file(package_manifest)
    assert out["runtime_package_dir"].endswith("/snerv_package")
    assert out["archive_sha256"] == archive_sha
    assert out["receiver_proof_report_paths"] == [proof_path.as_posix()]
    assert "snerv_packet_not_full_600_pairs" not in out["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in out["blockers"]
    assert out["post_export_materializer_plan"]["archive_record"]["source_runtime_dir"].endswith(
        "/snerv_package/submission"
    )


def test_pr95_stage8_execute_parser_exposes_source_lane_controls() -> None:
    args = _parse_args(
        [
            "--execute-pr95-stage8-source",
            "--stage8-epochs",
            "25",
            "--stage8-eval-every",
            "5",
            "--stage8-batch-size",
            "4",
            "--stage8-device",
            "cpu",
            "--stage8-muon-weight-decay",
            "0.0007",
            "--stage8-target-cache-path",
            "targets.pt",
            "--stage8-no-build-target-cache",
        ]
    )

    assert args.execute_pr95_stage8_source is True
    assert args.stage8_epochs == 25
    assert args.stage8_eval_every == 5
    assert args.stage8_batch_size == 4
    assert args.stage8_device == "cpu"
    assert args.stage8_muon_weight_decay == 0.0007
    assert args.stage8_target_cache_path == Path("targets.pt")
    assert args.stage8_no_build_target_cache is True


def test_adapt_pr95_stage8_report_emits_spine_runner_and_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = _write_synthetic_pr95_archive(tmp_path / "stage8_archive.zip")
    stage8_report_path = tmp_path / "pr95_stage8_report.json"
    stage8_report = {
        "schema": "pr95_stage8_from_public_archive_lane.v1",
        "mode": "execute",
        "report_path": stage8_report_path.as_posix(),
        "source_archive_zip": archive.as_posix(),
        "candidate_archive_zip_path": archive.as_posix(),
        "candidate_archive_zip_bytes": archive.stat().st_size,
        "candidate_archive_zip_sha256": runner_mod._sha256_file(archive),
        "local_training_result": {"raw_result": {"public_stage8_train_stage_called": True}},
        "exact_gate": {
            "schema": "exact_gate_blocker.v1",
            "blockers": ["contest_cpu_cuda_exact_eval_missing"],
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    stage8_report_path.write_text(
        json.dumps(stage8_report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        proof_path = out / "pr95_hnerv_receiver_proof.json"
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "report_path": proof_path.as_posix(),
            "archive_zip_path": Path(kwargs["archive_zip"]).as_posix(),
            "archive_zip_sha256": runner_mod._sha256_file(Path(kwargs["archive_zip"])),
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_output_kind": "contest_raw_rgb_interleaved",
            "receiver_output_bytes": 1,
            "receiver_proof_valid": True,
            "blockers": [],
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        proof_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = adapt_pr95_stage8_report_to_spine(
        pr95_stage8_report_path=stage8_report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "adapted_pr95_stage8_public_archive_report"
    assert out["score_claim"] is False
    assert out["stage8_source_faithfulness"]["public_stage8_train_stage_called"] is True
    assert out["stage8_source_faithfulness"]["source_faithful_training_complete"] is True
    assert out["receiver_proof_report_paths"]
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    assert "contest_cpu_cuda_exact_eval_missing" in out["blockers"]
    assert "runtime_consumption_smoke_is_not_score_authority" in out["blockers"]


def test_adapt_pr95_stage8_report_reuses_embedded_package_proof(
    tmp_path: Path,
) -> None:
    archive = _write_synthetic_pr95_archive(tmp_path / "stage8_archive.zip")
    proof_path = tmp_path / "embedded_receiver_proof.json"
    proof = {
        "schema": "pr95_mlx_pytorch_package_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "archive_path": archive.as_posix(),
        "archive_sha256": runner_mod._sha256_file(archive),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(
        json.dumps(proof, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stage8_report_path = tmp_path / "pr95_stage8_report.json"
    stage8_report_path.write_text(
        json.dumps(
            {
                "schema": "pr95_stage8_from_public_archive_lane.v1",
                "mode": "execute",
                "candidate_archive_zip_path": archive.as_posix(),
                "candidate_archive_zip_bytes": archive.stat().st_size,
                "candidate_archive_zip_sha256": runner_mod._sha256_file(archive),
                "package_report": {
                    "archive_bound_candidate_receiver_proof": proof,
                },
                "local_training_result": {"raw_result": {"public_stage8_train_stage_called": False}},
                "exact_gate": {"blockers": []},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = adapt_pr95_stage8_report_to_spine(
        pr95_stage8_report_path=stage8_report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["receiver_proof_report_paths"] == [proof_path.as_posix()]
    assert "receiver_proof_not_executed" not in out["blockers"]
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True


def test_pr95_hnerv_execute_arm_emits_runner_and_fail_closed_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "pr95_public_archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = Path(kwargs["archive_zip"])
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "proof_path": (out / "pr95_hnerv_receiver_proof.json").as_posix(),
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_output_kind": "contest_raw_rgb_interleaved",
            "receiver_output_bytes": 1,
            "receiver_proof_valid": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(report["proof_path"]).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner_mod,
        "_run_pr95_hnerv_mlx_scoreaware_smoke",
        fake_train,
    )
    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        source_archive_zip=tmp_path / "synthetic_source_pr95.zip",
        hard_byte_ceilings=(178_000,),
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    assert Path(out["acquisition_report_path"]).is_file()
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["family"] == "pr95_hnerv"
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert out["post_export_materializer_execution"]["requested"] is False
    assert out["post_export_materializer_execution"]["executed"] is False
    assert out["control_arm_scope"]["source_faithful_pr95_reproduction"] is False
    assert "pr95_hnerv_mlx_archive_export_control_arm_not_pr95_faithful_reproduction" in out["blockers"]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in out["blockers"]


def test_pr95_hnerv_execute_arm_uses_real_scorer_binding_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "pr95_public_archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "schema": "mlx_score_aware_training_objective.v1",
                    "segnet_distillation_weight": 0.05,
                    "pose_distillation_weight": 0.0005,
                    "has_real_segnet_teacher": True,
                    "has_real_posenet_teacher": True,
                    "allow_mock_scorer_teacher": False,
                    "allow_segnet_only_research": False,
                }
            },
        }

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = Path(kwargs["archive_zip"])
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "proof_path": (out / "pr95_hnerv_receiver_proof.json").as_posix(),
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_passed": True,
            "receiver_proof_valid": True,
            "blockers": [],
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(report["proof_path"]).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner_mod,
        "_run_pr95_hnerv_mlx_scoreaware_smoke",
        fake_train,
    )
    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        source_archive_zip=tmp_path / "synthetic_source_pr95.zip",
        hard_byte_ceilings=(178_000,),
        segnet_distillation_weight=0.05,
        pose_distillation_weight=0.0005,
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert "pr95_segnet_posenet_network_loss_not_wired_to_mlx" not in out["blockers"]
    assert "pr95_source_video_rgb_yuv6_preprocess_loss_is_not_full_scorer_loss" not in out["blockers"]
    assert "pr95_hnerv_default_scorer_distillation_weights_are_zero_unless_cli_overridden" not in out["blockers"]
    assert "pr95_mlx_scoreaware_teacher_distillation_is_advisory_not_exact_contest_loss" in out["blockers"]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in out["blockers"]


def test_vq_execute_forwards_optimizer_controls_and_shared_qat_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "family": "pact_nerv_vq",
            },
        }

    monkeypatch.setattr(runner_mod, "_run_pact_nerv_vq_mlx_smoke", fake_train)

    out = runner_mod.execute_pact_nerv_vq_mlx_smoke_and_adapt(
        output_dir=tmp_path / "vq_run",
        num_pairs=2,
        epochs=5,
        batch_pair_indices_per_step=1,
        learning_rate=1e-4,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        latent_dim=8,
        embed_dim=8,
        codebook_size=16,
        decoder_channel=8,
        decoder_codec="int4_scale_bundled",
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        optimizer_kind="lion",
        optimizer_grad_clip_max_norm=0.75,
        optimizer_weight_decay=2.0e-4,
        optimizer_warmup_epochs=2,
        optimizer_warmup_steps_per_epoch=3,
        optimizer_cosine_decay_enabled=True,
        optimizer_cosine_decay_total_epochs=5,
        optimizer_cosine_decay_min_lr_ratio=0.025,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "lion"
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(0.75)
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(2.0e-4)
    assert optimizer_controls["warmup_epochs"] == 2
    assert optimizer_controls["warmup_steps_per_epoch"] == 3
    assert optimizer_controls["cosine_decay_enabled"] is True
    assert optimizer_controls["cosine_decay_total_epochs"] == 5
    assert optimizer_controls["cosine_decay_min_lr_ratio"] == pytest.approx(0.025)
    optimizer_policy = captured_train_kwargs["optimizer_policy"]
    assert optimizer_policy["schema"] == "compact_pact_native_optimizer_policy.v1"
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert captured_train_kwargs["optimizer_kind"] == "lion"
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4

    report_training = out["score_aware_training"]
    assert report_training["optimizer_policy"]["schema"] == optimizer_policy["schema"]
    assert report_training["optimizer_policy"]["resolved_policy"] == (optimizer_policy["resolved_policy"])
    assert report_training["optimizer_policy"]["optimizer_kind_consumed_by_native_mlx"] is True
    assert report_training["optimizer_policy"]["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert report_training["optimizer_controls"]["optimizer_kind"] == (optimizer_controls["optimizer_kind"])
    assert report_training["optimizer_controls"]["grad_clip_max_norm"] == (optimizer_controls["grad_clip_max_norm"])
    assert (
        report_training["optimizer_controls"]["weight_decay_effective"]
        == (optimizer_controls["weight_decay_effective"])
    )
    assert report_training["optimizer_kind"] == "lion"
    assert report_training["native_optimizer_active"] is True
    assert report_training["effective_weight_decay"] == pytest.approx(2.0e-4)
    qat = report_training["coder_aware_qat"]
    assert qat["schema"] == "coder_aware_decoder_qat.v1"
    assert qat["enabled"] is True
    assert qat["quant_bits"] == 4
    assert qat["authority"] == "false_macos_mlx_research_signal"
    assert qat["authority_status"] == ("advisory_training_loss_only_not_archive_or_score_authority")
    assert "quantizer_geometry" in qat
    assert "include_substrings" in qat
    assert "exclude_substrings" in qat
    assert "pact_nerv_vq_spine_projection_manifest_missing" in out["blockers"]
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_selector_v4_execute_arm_emits_runner_and_fail_closed_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive)
        archive_sha = runner_mod._sha256_file(archive)
        archive_bytes = archive.stat().st_size
        receiver_dir = out / "receiver_proof"
        receiver_dir.mkdir(parents=True, exist_ok=True)
        receiver_proof = {
            "schema": "pact_nerv_selector_v4_mlx_generated_receiver_proof.v1",
            "proof_path": (receiver_dir / "pact_nerv_selector_v4_mlx_receiver_proof.json").as_posix(),
            "archive_path": archive.as_posix(),
            "archive_zip_path": archive.as_posix(),
            "archive_sha256": archive_sha,
            "archive_zip_sha256": archive_sha,
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_proof_valid": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(receiver_proof["proof_path"]).write_text(
            json.dumps(receiver_proof, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        projection = {
            "schema": "hprc_representation_spine_projection.v1",
            "family": "pact_nerv",
            "hprc_bin_bytes": 100,
            "manifest": {
                "representation_spine": {
                    "schema": "hprc_representation_spine_manifest.v1",
                    "family": "pact_nerv",
                    "hprc_bin_bytes": 100,
                    "source": {
                        "kind": "pact_nerv_selector_v4_export_payload",
                        "archive_zip_path": archive.as_posix(),
                        "archive_zip_sha256": archive_sha,
                        "bytes": archive_bytes,
                        "sha256": archive_sha,
                    },
                    "manifest_extra": {
                        "emitted_by": "export_pact_nerv_selector_v4_mlx_archive",
                        "num_pairs": 600,
                        "num_frames": 1200,
                        "coverage_source": "test_selector_v4_execute_arm",
                        "selector_codec": "run_length_varint_selector",
                    },
                    "sections": [
                        {
                            "name": "decoder_qw",
                            "role": "charged_decoder_or_program_weights",
                            "bytes": 60,
                            "sha256": "a" * 64,
                        },
                        {
                            "name": "latents_rc",
                            "role": "charged_per_pair_latents",
                            "bytes": 40,
                            "sha256": "b" * 64,
                        },
                    ],
                }
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        manifest = out / "hprc_representation_spine_pact_nerv_selector_v4_manifest.json"
        runner_mod._write_json(manifest, projection)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "substrate_artifact_metadata": {
                "family": "pact_nerv_selector_v4",
                "selector_codec": "run_length_varint_selector",
            },
        }

    monkeypatch.setattr(
        runner_mod,
        "_run_pact_nerv_selector_v4_mlx_smoke",
        fake_train,
    )
    projection_manifest_path = (
        tmp_path
        / "run"
        / "pact_nerv_selector_v4_mlx_training"
        / "hprc_representation_spine_pact_nerv_selector_v4_manifest.json"
    )
    mlx_profile_path = tmp_path / "selector_section_value_profile.json"
    mlx_profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "family": "pact_nerv",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "projection_manifest_path": projection_manifest_path.as_posix(),
                "scope_status": {
                    "full_video": True,
                    "axis": "[macOS-MLX research-signal]",
                },
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "archive_bytes_removed_vs_baseline": 60,
                        "delta_nonrate_score": 0.25,
                        "delta_total_mlx_score_advisory": 0.25,
                        "family": "pact_nerv",
                        "projection_manifest_path": (projection_manifest_path.as_posix()),
                        "marginal_status": "measured_full_video_mlx_advisory",
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=8,
        embed_dim=8,
        selector_palette_size=16,
        decoder_channel=8,
        decoder_codec="int4_scale_bundled",
        mlx_profile_paths=(mlx_profile_path,),
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        optimizer_kind="adafactor",
        optimizer_grad_clip_max_norm=0.5,
        optimizer_weight_decay=3.0e-4,
        optimizer_warmup_epochs=1,
        optimizer_warmup_steps_per_epoch=2,
        optimizer_cosine_decay_enabled=True,
        optimizer_cosine_decay_total_epochs=3,
        optimizer_cosine_decay_min_lr_ratio=0.05,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_pact_nerv_selector_v4_mlx_smoke_and_exported"
    assert out["execute_family"] == "pact_nerv_selector_v4"
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["selector_v4_archive_surface"]["selector_codec"] == ("run_length_varint_selector")
    assert out["score_aware_training"]["scorer_coupled_rd"]["fixed_marginal_byte_price"] == "25/uncompressed_total"
    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "adafactor"
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(0.5)
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(3.0e-4)
    assert optimizer_controls["warmup_epochs"] == 1
    assert optimizer_controls["warmup_steps_per_epoch"] == 2
    assert optimizer_controls["cosine_decay_enabled"] is True
    assert optimizer_controls["cosine_decay_total_epochs"] == 3
    assert optimizer_controls["cosine_decay_min_lr_ratio"] == pytest.approx(0.05)
    optimizer_policy = captured_train_kwargs["optimizer_policy"]
    assert optimizer_policy["schema"] == "compact_pact_native_optimizer_policy.v1"
    assert optimizer_policy["family"] == "pact_nerv_selector_v4"
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert out["score_aware_training"]["optimizer_policy"]["schema"] == (optimizer_policy["schema"])
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == (optimizer_policy["resolved_policy"])
    assert out["score_aware_training"]["optimizer_policy"]["optimizer_kind_consumed_by_native_mlx"] is True
    assert out["score_aware_training"]["optimizer_policy"]["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert out["score_aware_training"]["optimizer_controls"]["optimizer_kind"] == (optimizer_controls["optimizer_kind"])
    assert (
        out["score_aware_training"]["optimizer_controls"]["grad_clip_max_norm"]
        == optimizer_controls["grad_clip_max_norm"]
    )
    assert (
        out["score_aware_training"]["optimizer_controls"]["weight_decay_effective"]
        == optimizer_controls["weight_decay_effective"]
    )
    assert out["score_aware_training"]["optimizer_kind"] == "adafactor"
    assert out["score_aware_training"]["native_optimizer_active"] is True
    assert out["score_aware_training"]["effective_weight_decay"] == pytest.approx(3.0e-4)
    qat = out["score_aware_training"]["coder_aware_qat"]
    assert qat["schema"] == "coder_aware_decoder_qat.v1"
    assert qat["enabled"] is True
    assert qat["quant_bits"] == 4
    assert qat["quant_residual_weight"] == pytest.approx(0.001)
    assert qat["magnitude_weight"] == pytest.approx(0.0001)
    assert qat["delta_weight"] == pytest.approx(0.0002)
    assert qat["c1a_entropy_weight"] == pytest.approx(0.0003)
    assert qat["c1a_sigma"] == pytest.approx(0.35)
    assert qat["c1a_sample_size"] == 64
    assert qat["authority"] == "false_macos_mlx_research_signal"
    assert qat["authority_status"] == ("advisory_training_loss_only_not_archive_or_score_authority")
    assert "quantizer_geometry" in qat
    assert "include_substrings" in qat
    assert "exclude_substrings" in qat
    assert out["score_aware_training"]["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
    assert captured_train_kwargs["coder_qat_c1a_entropy_weight"] == 0.0003
    assert captured_train_kwargs["coder_qat_c1a_sigma"] == 0.35
    assert captured_train_kwargs["coder_qat_c1a_sample_size"] == 64
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    assert out["mlx_profile_paths"] == [mlx_profile_path.as_posix()]
    assert Path(out["acquisition_report_path"]).is_file()
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    assert runner["mlx_profile_paths"] == [mlx_profile_path.as_posix()]
    row = runner["selected_runner_rows"][0]
    assert row["family"] == "pact_nerv"
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    value_rows = {item["section_name"]: item for item in runner["section_value_rows"]}
    assert value_rows["decoder_qw"]["evidence_status"] == "measured_mlx_advisory"
    assert value_rows["decoder_qw"]["admission_status"] == ("admit_section_bytes_for_receiver_proof")
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert out["post_export_materializer_execution"]["requested"] is False
    assert out["post_export_materializer_execution"]["executed"] is False
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert "pact_nerv_selector_v4_spine_projection_manifest_missing" not in out["blockers"]


def test_selector_v4_execute_arm_threads_render_quality_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**_kwargs):
        return {
            "archive_path": None,
            "archive_bytes": None,
            "archive_sha256": None,
            "substrate_artifact_metadata": {
                "selector_v4_render_quality": {
                    "schema": "pact_nerv_selector_v4_mlx_render_quality.v1",
                    "verdict": "RENDER_OUTPUT_DEGENERATE_BLOCK_ARCHIVE_PROFILE",
                    "export_blocked_recommended": True,
                    "blockers": [
                        "selector_v4_render_segnet_last_frame_std_too_low",
                    ],
                },
            },
        }

    monkeypatch.setattr(
        runner_mod,
        "_run_pact_nerv_selector_v4_mlx_smoke",
        fake_train,
    )

    out = execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["execute_family"] == "pact_nerv_selector_v4"
    assert out["selector_v4_render_quality"]["export_blocked_recommended"] is True
    assert "selector_v4_render_segnet_last_frame_std_too_low" in out["blockers"]
    assert "pact_nerv_selector_v4_render_quality_gate_failed" in out["blockers"]
    assert "pact_nerv_selector_v4_archive_export_missing" in out["blockers"]
