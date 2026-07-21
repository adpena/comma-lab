# SPDX-License-Identifier: MIT
"""Focused tests for deterministic PREDICT-to-PROJECT receiver primitives."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tools.measure_predict_project_receiver as measurement_tool
from tac.canonical_equations.predict_project_receiver_20260721 import registration_gate
from tac.optimization.predict_project_receiver import (
    ACTION_LEVEL_RUNGS,
    ATTRIBUTION_REUSE_BINDINGS,
    BOUNDARY_INVERSE_ACTION_POLICY,
    CANONICAL_LAW_RESOLUTION_CUSTODY,
    CANONICAL_LAW_RESOLUTION_SHA256,
    GLOBAL_WATERFILL_LAMBDA_STAR,
    GLOBAL_WATERFILL_STREAMS,
    M1_ANCHORS,
    M1_FLIP_COUNT,
    M1_RECEIPT_COMMIT,
    M1_RECEIPT_PATH,
    M1_RECEIPT_SHA256,
    M1_SCORE_DENOMINATOR,
    PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
    S3_TRAINER_REUSE,
    SEGNET_CENTERED_HEAD_RANK,
    SEGNET_HEAD_RANK_EQUATION_ID,
    TEMPORAL_JITTER_AMORTIZATION_RATIO,
    LinearConstraint,
    PredictProjectReceiverError,
    camera_uint8_identity_sha256,
    double_decode_hash,
    extract_constraint_violations,
    global_joint_waterfill,
    hard_oracle_custody_sha256,
    measured_waterfill_adapter,
    plane_cache_key,
    predict_cell_field,
    project_linear_intersection,
    projected_plane_array_sha256,
    quantize_uint8_feasible,
    realize_inverse_r_camera_uint8,
    realize_projected_rgb_plane_camera_uint8,
    receiver_composition_metadata,
    stratify_predictor_quality,
    validate_action_level_ladder_evidence,
    validate_attribution_edit_telemetry,
    validate_flip_attribution_receipt,
    validate_global_joint_waterfill_evidence,
    validate_ladder_edit_request,
    validate_ladder_edit_response,
    validate_learned_tail_race_evidence,
    validate_per_flip_sellback_evidence,
    validate_pose_tube_knee_evidence,
    verify_pose_tightening_choice,
)
from tac.optimization.predict_project_schema import (
    PredictProjectSchemaError,
    build_minimal_constraint_seed,
    canonical_json_bytes,
    serialize_constraint_seed,
    validate_constraint_seed,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tools.measure_predict_project_receiver import MeasurementError, run_measurement


def seed() -> dict:
    return build_minimal_constraint_seed(
        bytes([0, 1, 2, 3, 4, 0]),
        scorer_height=2,
        scorer_width=3,
        camera_height=4,
        camera_width=6,
    )


def test_static_single_object_predictor_is_deterministic_and_double_decodes():
    value = seed()
    expected = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    assert np.array_equal(predict_cell_field(value, 0), expected)
    result = double_decode_hash(lambda: predict_cell_field(value, 217))
    assert result.byte_identical is True
    assert result.first_sha256 == result.second_sha256


def test_declared_phase_carrier_fails_closed_without_existing_carrier_callback():
    value = seed()
    value["boundary_jitter"]["selected_rung"] = "R1"
    value["boundary_jitter"]["r1"]["phase_carrier_id"] = "phase_carrier_425"
    with pytest.raises(PredictProjectReceiverError, match="R1 requires"):
        predict_cell_field(value, 0)
    output = predict_cell_field(
        value,
        0,
        phase_carrier=lambda **kwargs: kwargs["base_field"],
    )
    assert output.shape == (2, 3)


def test_r2_causal_response_requires_typed_appearance_and_generic_callback():
    value = seed()
    samples = [1, -1]
    value["boundary_jitter"]["selected_rung"] = "R2"
    value["boundary_jitter"]["r2"]["appearance_phase_chart"]["samples_q"] = samples
    value["boundary_jitter"]["r2"]["appearance_phase_chart"]["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(samples)
    ).hexdigest()
    value["boundary_jitter"]["r2"]["xi_response"]["response_model_id"] = "generic-causal-response.v1"
    with pytest.raises(PredictProjectReceiverError, match="R2 requires"):
        predict_cell_field(value, 0)
    output = predict_cell_field(value, 0, response_surface=lambda **kwargs: kwargs["base_field"])
    assert output.shape == (2, 3)
    assert "noise" not in canonical_json_bytes(value["boundary_jitter"]).decode("ascii")


def test_violation_extractor_emits_only_mismatches_and_b3_is_exact():
    predicted = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    desired = np.array([[0, 4], [2, 0]], dtype=np.uint8)
    strata = np.array([["cell_interior", "boundary_codim1"], ["movable_track", "critical_event"]])
    rows = extract_constraint_violations(predicted, desired, time=9, strata=strata)
    assert [(row["y"], row["x"], row["cell_id"]) for row in rows] == [(0, 1, 4), (1, 1, 0)]
    report = stratify_predictor_quality(predicted, desired, strata=strata)
    assert report["overall"] == {
        "already_satisfied": 2,
        "total": 4,
        "fraction": 0.5,
        "violations": 2,
    }
    assert report["by_stratum"]["critical_event"]["violations"] == 1
    assert report["evidence_source"] == "declared_constraint_fixture"
    assert report["source_ground_truth_claim"] is False
    with pytest.raises(PredictProjectReceiverError, match="frame0 is pose-only"):
        extract_constraint_violations(predicted, desired, time=9, frame_index=0)


def test_dykstra_projects_box_and_halfspaces_in_stable_order():
    constraints = (
        LinearConstraint("b", np.array([0.0, 1.0]), 2.0),
        LinearConstraint("a", np.array([1.0, 0.0]), 3.0),
        LinearConstraint("sum", np.array([1.0, 1.0]), 4.0),
    )
    result = project_linear_intersection(
        np.array([10.0, 10.0]), constraints, lower=np.zeros(2), upper=np.full(2, 255.0)
    )
    assert result.converged is True
    assert np.max(result.point - np.array([3.0, 2.0])) <= 1e-8
    assert float(np.sum(result.point)) <= 4.0 + 1e-8


def test_projection_fails_closed_on_nonconvergence():
    constraints = (LinearConstraint("x<=0", np.array([1.0]), 0.0),)
    with pytest.raises(PredictProjectReceiverError, match="did not converge"):
        project_linear_intersection(np.array([10.0]), constraints, iteration_cap=1, tolerance=0.0)


def test_nearest_uint8_lattice_is_feasible_and_lexicographic_on_ties():
    constraints = (
        LinearConstraint("sum<=5", np.array([1.0, 1.0]), 5.0),
        LinearConstraint("sum>=5", np.array([-1.0, -1.0]), -5.0),
    )
    result = quantize_uint8_feasible(np.array([2.5, 2.5]), constraints)
    assert tuple(result) == (2, 3)
    assert result.dtype == np.uint8


def test_independent_two_curve_waterfill_is_non_authoritative():
    assert measured_waterfill_adapter([], [])["status"] == "INCONCLUSIVE_DEPRECATED_INDEPENDENT_CURVES"
    result = measured_waterfill_adapter(
        [{"bytes": 0, "distortion": 0.2}, {"bytes": 10, "distortion": 0.1}],
        [{"bytes": 0, "distortion": 0.01}, {"bytes": 10, "distortion": 0.005}],
    )
    assert result["status"] == "INCONCLUSIVE_DEPRECATED_INDEPENDENT_CURVES"
    assert result["authority"] is False
    assert global_joint_waterfill(None)["status"] == "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP"


def test_receiver_metadata_reuses_existing_abis_and_never_serializes_kernel():
    kernel = FullResizeKernel.build(camera_h=4, camera_w=6, scorer_h=2, scorer_w=3)
    metadata = receiver_composition_metadata(kernel)
    assert metadata["kernel_serialized"] is False
    assert metadata["receiver_search_invocations"] == 0
    assert "solve_interval_frame" in metadata["interval_solver"]
    assert metadata["full_kernel_nullity_per_channel"] == 18
    assert metadata["full_kernel_callable"] is True
    assert metadata["inverse_r_output"] == "camera_resolution_rgb_uint8"


def test_inverse_r_realization_is_camera_uint8_integer_exact_and_nonserialized():
    kernel = FullResizeKernel.build(camera_h=4, camera_w=6, scorer_h=2, scorer_w=3)
    camera = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    numerators, denominator = kernel.operator.apply_numerators(camera)
    result = realize_inverse_r_camera_uint8(
        numerators,
        denominator,
        np.zeros((2, 3, 3), dtype=np.float64),
        predictor=camera,
        kernel=kernel,
    )
    assert result["integer_parseback_exact"] is True
    assert result["full_kernel_callable"] is True
    assert result["full_kernel_serialized"] is False
    assert result["camera_uint8_sha256"] == camera_uint8_identity_sha256(result["frame"])


def test_projected_rgb_lattice_stage_is_exact_zero_seed_and_scorer_free():
    kernel = FullResizeKernel.build(camera_h=4, camera_w=6, scorer_h=2, scorer_w=3)
    cells = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    rgb = np.array(
        [
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            [[41, 42, 43], [51, 52, 53], [61, 62, 63]],
        ],
        dtype=np.uint8,
    )
    custody = {
        "schema": PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
        "source_kind": "decoder_derived_from_seed",
        "generator_id": "fixture_generic_rgb_projection.v1",
        "seed_sha256": hashlib.sha256(b"seed").hexdigest(),
        "projected_rgb_sha256": projected_plane_array_sha256(rgb),
        "projected_cells_sha256": projected_plane_array_sha256(cells),
        "additional_seed_bytes": 0,
        "decoder_scorer_invocations": 0,
    }
    result = realize_projected_rgb_plane_camera_uint8(rgb, cells, custody, kernel=kernel)
    assert result["integer_parseback_exact"] is True
    assert result["factor2_verification"]["certified_exact"] is True
    assert result["additional_seed_bytes"] == 0
    assert result["decoder_scorer_invocations"] == 0
    assert np.array_equal(kernel.operator.apply(result["frame"]), rgb.astype(np.float64))


def test_projected_rgb_lattice_stage_refuses_label_plane_and_uncounted_encoder_input():
    kernel = FullResizeKernel.build(camera_h=4, camera_w=6, scorer_h=2, scorer_w=3)
    cells = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    custody = {
        "schema": PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
        "source_kind": "decoder_derived_from_seed",
        "generator_id": "fixture_generic_rgb_projection.v1",
        "seed_sha256": hashlib.sha256(b"seed").hexdigest(),
        "projected_rgb_sha256": projected_plane_array_sha256(cells),
        "projected_cells_sha256": projected_plane_array_sha256(cells),
        "additional_seed_bytes": 0,
        "decoder_scorer_invocations": 0,
    }
    with pytest.raises(PredictProjectReceiverError, match="2D class-ID field"):
        realize_projected_rgb_plane_camera_uint8(cells, cells, custody, kernel=kernel)

    rgb = np.repeat(cells[:, :, None], 3, axis=2)
    custody.update(
        {
            "source_kind": "encoder_supplied_counted",
            "projected_rgb_sha256": projected_plane_array_sha256(rgb),
        }
    )
    with pytest.raises(PredictProjectReceiverError, match="cannot claim zero"):
        realize_projected_rgb_plane_camera_uint8(rgb, cells, custody, kernel=kernel)

    custody.update(
        {
            "source_kind": "decoder_derived_from_seed",
            "additional_seed_bytes": 0,
            "decoder_scorer_invocations": 0.0,
        }
    )
    with pytest.raises(PredictProjectReceiverError, match="cannot invoke a scorer"):
        realize_projected_rgb_plane_camera_uint8(rgb, cells, custody, kernel=kernel)


def test_plane_cache_key_is_deterministic_and_custody_sensitive():
    payload = serialize_constraint_seed(seed())
    first = plane_cache_key(payload, 3, 1, "dykstra")
    assert first == plane_cache_key(payload, 3, 1, "dykstra")
    assert first != plane_cache_key(payload, 4, 1, "dykstra")


def test_measurement_cli_is_resumable_and_keeps_b2_incomplete_without_oracle(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    output = tmp_path / "measurement"
    command = [
        sys.executable,
        str(repo / "tools" / "measure_predict_project_receiver.py"),
        "--seed",
        str(seed_path),
        "--output-dir",
        str(output),
        "--pair-end",
        "2",
        "--chunk-size",
        "1",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    first = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    second = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    assert json.loads(first.stdout) == json.loads(second.stdout)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["b2"]["measurement_status"] == "INCOMPLETE_NO_HARD_ORACLE"
    assert receipt["authority"]["score_claim"] is False
    assert receipt["b3"]["authority"] == "NON_AUTHORITATIVE_DECLARED_CONSTRAINT_FIXTURE"
    assert receipt["b4"]["status"] == "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP"
    assert receipt["b4"]["per_flip_sellback_status"] == "INCONCLUSIVE_NO_MEASURED_PER_FLIP_SELLBACK"
    assert receipt["b4"]["action_level_ladder_status"] == "INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER"
    assert receipt["b4"]["attribution_edit_telemetry_status"] == ("INCONCLUSIVE_NO_MEASURED_ATTRIBUTION_EDIT_TELEMETRY")
    assert receipt["b4"]["learned_tail_race_status"] == ("INCONCLUSIVE_NO_MEASURED_LEARNED_TAIL_THREE_WAY_RACE")
    assert receipt["b4"]["pose_tube_knee_status"] == "INCONCLUSIVE_NO_MEASURED_POSE_TUBE_KNEE"
    assert receipt["m1_anchors"] == M1_ANCHORS
    assert receipt["b1"]["current_invocation_plane_cache_hits"] == 0
    assert receipt["b1"]["current_invocation_plane_cache_misses"] == 0
    assert receipt["b1"]["resumed_stage_rows"] == 2
    assert receipt["implementation_status"] == "BUILT_INTERFACE_MEASUREMENT_BLOCKED"
    assert receipt["canonical_law_resolution"] == CANONICAL_LAW_RESOLUTION_CUSTODY
    assert receipt["canonical_law_resolution_sha256"] == CANONICAL_LAW_RESOLUTION_SHA256
    assert receipt["gates"]["MS_vineyard_native_rasterizer"] == {
        "status": "MS_SCHEMA_BUILT_NATIVE_RASTERIZER_BLOCKED",
        "blocker": "MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED",
    }
    assert {
        "G1_pose_blind_constraint_tightening",
        "G2_camera_resolution_inverse_r",
        "G3_frame_asymmetry",
        "G4_cross_host_byte_identity",
        "G5_named_section_container",
    }.issubset(receipt["gates"])
    assert receipt["gates"]["global_joint_waterfill"]["blocker"] is not None
    assert receipt["gates"]["per_flip_sellback"]["blocker"] is not None
    assert receipt["gates"]["action_level_ladder"]["blocker"] is not None
    assert receipt["gates"]["attribution_edit_telemetry"]["blocker"] is not None
    assert receipt["gates"]["learned_tail_race"]["blocker"] is not None
    assert receipt["gates"]["pose_tube_knee"]["blocker"] is not None
    assert receipt["b5"]["causal_jitter_ladder"]["exceptions_term"] == "causal_sparse_exceptions"
    assert receipt["b5"]["temporal_jitter_law"]["lawref_resolved_ratio"] == (TEMPORAL_JITTER_AMORTIZATION_RATIO)
    assert len(list((output / "stages").glob("pair_*.json"))) == 2
    assert receipt["config"]["implementation_sources"]["resume_policy"] == (
        "EXACT_SOURCE_AND_CONFIG_ONLY_OLD_CODE_STAGES_REFUSED"
    )
    assert set(receipt["config"]["implementation_sources"]["files"]) == set(
        measurement_tool.IMPLEMENTATION_SOURCE_PATHS
    )


def test_measurement_resume_refuses_config_and_implementation_source_drift(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    output = tmp_path / "source-bound-measurement"
    run_measurement(seed_path, output, pair_end=1, chunk_size=1)
    stage_path = output / "stages" / "pair_0000.json"
    preserved_stage = stage_path.read_bytes()

    with pytest.raises(MeasurementError, match="stage config drift"):
        run_measurement(seed_path, output, pair_end=1, chunk_size=1, workers=2)

    drifted = copy.deepcopy(measurement_tool._implementation_source_custody())
    drifted["files"]["src/tac/optimization/predict_project_receiver.py"] = _digest("changed-source")
    drifted["aggregate_sha256"] = hashlib.sha256(canonical_json_bytes(drifted["files"])).hexdigest()
    monkeypatch.setattr(measurement_tool, "_implementation_source_custody", lambda: drifted)
    with pytest.raises(MeasurementError, match="stage implementation-source drift"):
        run_measurement(seed_path, output, pair_end=1, chunk_size=1)
    assert stage_path.read_bytes() == preserved_stage

    monkeypatch.undo()
    old_output = tmp_path / "simulated-old-code-measurement"
    run_measurement(seed_path, old_output, pair_end=1, chunk_size=1)
    old_stage_path = old_output / "stages" / "pair_0000.json"
    old_stage = json.loads(old_stage_path.read_text(encoding="utf-8"))
    old_stage.pop("implementation_sources_sha256")
    old_stage_path.write_text(json.dumps(old_stage, sort_keys=True), encoding="utf-8")
    with pytest.raises(MeasurementError, match="stage implementation-source drift"):
        run_measurement(seed_path, old_output, pair_end=1, chunk_size=1)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _adapter_custody(callback, seed_value: dict, *, cache_label: str = "stable-cache") -> dict:
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema": "predict_project_hard_oracle_custody.v0",
        "seed": 1234,
        "batch_size": 16,
        "measurement_axis": "[macOS-CPU advisory]",
        "scorer": {
            "implementation_id": "test.cpu_torch.scorer",
            "version": "fixture-v1",
            "source_sha256": _digest("scorer-source"),
            "segnet_weights_sha256": _digest("segnet-weights"),
            "posenet_weights_sha256": _digest("posenet-weights"),
        },
        "inputs": {
            "source_sha256": hashlib.sha256(serialize_constraint_seed(seed_value)).hexdigest(),
            "cache_sha256": _digest(cache_label),
            "evaluated_input_sha256": _digest("stable-evaluated-corpus"),
        },
        "adapter": {
            "identity": f"{callback.__module__}:{callback.__qualname__}",
            "source_sha256": source_sha256,
        },
    }


def valid_hard_oracle_without_sweep(**kwargs):
    predicted = kwargs["predicted"]
    return {
        "schema": "predict_project_hard_oracle_pair.v0",
        "pair_index": kwargs["pair_index"],
        "d_seg": float(kwargs["pair_index"]),
        "d_pose": float(kwargs["pair_index"]) / 10.0,
        "cell_exact": True,
        "pose_within_tube": True,
        "uint8_factor2_exact": True,
        "stage_seconds": {"projection": 0.0, "realization": 0.0, "verification": 0.0},
        "custody": _adapter_custody(valid_hard_oracle_without_sweep, kwargs["seed"]),
        "desired_cells": predicted.copy(),
    }


def represented_surface_hard_oracle(**kwargs):
    assert not np.array_equal(kwargs["predicted"], kwargs["represented"])
    row = valid_hard_oracle_without_sweep(**kwargs)
    row["custody"] = _adapter_custody(represented_surface_hard_oracle, kwargs["seed"])
    row["cell_exact"] = bool(kwargs["represented"][0, 0] == 4)
    return row


def fake_legacy_hard_oracle(**kwargs):
    return {
        "schema": "predict_project_hard_oracle_pair.v0",
        "pair_index": kwargs["pair_index"],
        "d_seg": 0.0,
        "d_pose": 0.0,
        "cell_exact": True,
        "pose_within_tube": True,
        "uint8_factor2_exact": True,
        "stage_seconds": {"projection": 0.0, "realization": 0.0, "verification": 0.0},
        "hard_oracle": "cpu_torch_seed1234_batch16",
        "desired_cells": None,
    }


def mixed_custody_hard_oracle(**kwargs):
    row = valid_hard_oracle_without_sweep(**kwargs)
    row["custody"] = _adapter_custody(
        mixed_custody_hard_oracle,
        kwargs["seed"],
        cache_label=f"cache-{kwargs['pair_index']}",
    )
    return row


def test_fake_callback_cannot_manufacture_legacy_custody_or_promotability(tmp_path):
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    with pytest.raises(MeasurementError, match="fields mismatch"):
        run_measurement(seed_path, tmp_path / "fake", pair_end=1, hard_oracle=fake_legacy_hard_oracle)
    assert not (tmp_path / "fake" / "receipt.json").exists()
    assert registration_gate(None)["registration_allowed"] is False


def test_mixed_hard_oracle_custody_is_rejected(tmp_path):
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    with pytest.raises(MeasurementError, match="mixed hard-oracle custody"):
        run_measurement(seed_path, tmp_path / "mixed", pair_end=2, hard_oracle=mixed_custody_hard_oracle)
    assert not (tmp_path / "mixed" / "receipt.json").exists()


def test_hard_oracle_prefix_has_structured_custody_and_never_builds_cross_pair_b4(tmp_path):
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    receipt = run_measurement(
        seed_path,
        tmp_path / "valid",
        pair_end=2,
        hard_oracle=valid_hard_oracle_without_sweep,
    )
    assert receipt["b2"]["measurement_status"] == "MEASURED_PREFIX"
    assert receipt["b2"]["custody_byte_identical_across_rows"] is True
    assert receipt["b2"]["contest_authority"] is False
    assert receipt["b2"]["promotion_eligible"] is False
    assert registration_gate(receipt["b2"])["registration_allowed"] is False
    assert receipt["b4"]["status"] == "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP"
    assert receipt["b4"]["independent_curve_authority"] is False
    assert receipt["b3"]["authority"] == "MEASURED_REAL_DESIRED_CELLS_NON_SOURCE_GROUND_TRUTH"
    assert receipt["b3"]["source_ground_truth_quality_claim"] is False


def test_hard_oracle_receives_projected_represented_field_and_hashes_both_surfaces(tmp_path):
    constrained = seed()
    constrained["constraint_seeds"] = [
        {
            "time": 0,
            "frame_index": 1,
            "obligation": "seg_and_pose",
            "y": 0,
            "x": 0,
            "cell_id": 4,
            "predictor_status": "violated",
            "stratum": "cell_interior",
            "pose_tube": None,
            "pose_tightening_id": None,
            "projector": None,
        }
    ]
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(constrained))
    receipt = run_measurement(
        seed_path,
        tmp_path / "represented",
        pair_end=1,
        hard_oracle=represented_surface_hard_oracle,
    )
    row = receipt["b2"]
    stage = json.loads((tmp_path / "represented/stages/pair_0000.json").read_text())
    assert row["cell_exact"] is True
    assert stage["hard_oracle"]["pair_input_sha256"] != stage["hard_oracle"]["represented_input_sha256"]


def _m1_receipt_binding() -> dict:
    return {"commit": M1_RECEIPT_COMMIT, "path": M1_RECEIPT_PATH, "sha256": M1_RECEIPT_SHA256}


def _synthetic_action_level_ladder(per_flip: dict, joint_decode: dict) -> dict:
    """Build a schema-only five-rung fixture; no rung row is a measurement."""

    joint_sha = hashlib.sha256(canonical_json_bytes(joint_decode)).hexdigest()
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for flip in per_flip["flips"]:
        grouped.setdefault((flip["decision"], flip["stratum"], flip["price_stratum"]), []).append(flip)
    families = []
    for family_index, ((decision, receiver_stratum, price_stratum), rows) in enumerate(sorted(grouped.items())):
        members = [row["flip_id"] for row in rows]
        isolated = len(members) == 1
        rung_bits = [40, 32, 24, 16, 8] if isolated else [8, 16, 24, 32, 40]
        pixels = sum(row["positive_pixel_count"] for row in rows)
        positive_score = pixels * 100 / M1_SCORE_DENOMINATOR
        rungs = []
        for rung_index, (rung_name, coded_bits) in enumerate(zip(ACTION_LEVEL_RUNGS, rung_bits, strict=True)):
            valid = rung_name != "L5_pixel_write" or isolated
            coded_bytes = coded_bits // 8
            rungs.append(
                {
                    "rung": rung_name,
                    "actuator_id": f"fixture.actuator.{rung_index}",
                    "measurement_status": "MEASURED_SAME_JOINT_DECODE_THROUGH_R",
                    "valid": valid,
                    "invalid_reason": None if valid else "L5_NON_SINGLETON_FORBIDDEN",
                    "joint_decode_sha256": joint_sha,
                    "archive_sha256": _digest(f"action-archive-{family_index}-{rung_index}"),
                    "decoded_output_sha256": _digest(f"action-output-{family_index}-{rung_index}"),
                    "context_sha256": _digest(f"action-context-{family_index}-{rung_index}"),
                    "coded_stream_sha256": _digest(f"action-stream-{family_index}-{rung_index}"),
                    "coded_bits": coded_bits,
                    "coded_bytes": coded_bytes,
                    "positive_flip_count": len(members),
                    "positive_pixel_count": pixels,
                    "positive_score_benefit": positive_score,
                    "effective_bytes_per_positive_flip": coded_bytes / len(members),
                    "erf_collateral_flip_ids": [],
                    "erf_collateral_flip_count": 0,
                    "erf_collateral_positive_pixel_count": 0,
                    "erf_collateral_d_seg": 0.0,
                    "net_score_benefit": positive_score,
                    "net_score_per_coded_byte": positive_score / coded_bytes,
                }
            )
        families.append(
            {
                "family_id": f"family:{family_index:02d}",
                "receiver_stratum": receiver_stratum,
                "price_stratum": price_stratum,
                "member_flip_ids": members,
                "family_membership_sha256": hashlib.sha256(canonical_json_bytes(members)).hexdigest(),
                "decision": decision,
                "isolated_singleton": isolated,
                "rungs": rungs,
                "selected_rung": "L5_pixel_write" if isolated else "L1_geometry_chart",
            }
        )

    distribution = []
    for receiver_stratum in ("cell_interior", "boundary_codim1", "movable_track", "critical_event"):
        for price_stratum in ("road_lane", "other_edge", "non_edge", "tight_margin"):
            matching = [
                family
                for family in families
                if family["receiver_stratum"] == receiver_stratum and family["price_stratum"] == price_stratum
            ]
            flip_count = sum(len(family["member_flip_ids"]) for family in matching)
            kept = [family for family in matching if family["decision"] == "keep"]
            eaten = [family for family in matching if family["decision"] == "eat"]
            selected = {
                family["family_id"]: next(row for row in family["rungs"] if row["rung"] == family["selected_rung"])
                for family in matching
            }
            distribution.append(
                {
                    "receiver_stratum": receiver_stratum,
                    "price_stratum": price_stratum,
                    "flip_count": flip_count,
                    "kept_flip_count": sum(len(family["member_flip_ids"]) for family in kept),
                    "eaten_flip_count": sum(len(family["member_flip_ids"]) for family in eaten),
                    "chosen_rung_flip_counts": {
                        rung: sum(
                            len(family["member_flip_ids"]) for family in matching if family["selected_rung"] == rung
                        )
                        for rung in ACTION_LEVEL_RUNGS
                    },
                    "chosen_rung_family_counts": {
                        rung: sum(family["selected_rung"] == rung for family in matching) for rung in ACTION_LEVEL_RUNGS
                    },
                    "admitted_coded_bits": sum(selected[family["family_id"]]["coded_bits"] for family in kept),
                    "admitted_coded_bytes": sum(selected[family["family_id"]]["coded_bytes"] for family in kept),
                    "eaten_avoided_coded_bits": sum(selected[family["family_id"]]["coded_bits"] for family in eaten),
                    "eaten_avoided_coded_bytes": sum(selected[family["family_id"]]["coded_bytes"] for family in eaten),
                    "erf_collateral_flip_count": 0,
                    "erf_collateral_positive_pixel_count": 0,
                    "erf_collateral_d_seg": 0.0,
                }
            )
    selected_rows = [
        next(row for row in family["rungs"] if row["rung"] == family["selected_rung"]) for family in families
    ]
    admitted_rows = [row for family, row in zip(families, selected_rows, strict=True) if family["decision"] == "keep"]
    eaten_rows = [row for family, row in zip(families, selected_rows, strict=True) if family["decision"] == "eat"]
    return {
        "schema": "predict_project_action_level_ladder.v0",
        "measurement_status": "MEASURED_SAME_JOINT_DECODE_ACTION_LEVEL_LADDER",
        "scope": "m1_all_17926_flips_all_five_rungs",
        "m1_receipt": _m1_receipt_binding(),
        "joint_decode_sha256": joint_sha,
        "boundary_inverse_policy": copy.deepcopy(BOUNDARY_INVERSE_ACTION_POLICY),
        "ladder_policy": {
            "rungs": list(ACTION_LEVEL_RUNGS),
            "selection": "minimum_coded_bytes_per_net_score_then_canonical_rung",
            "per_flip_ledger_role": "currency_only_not_actuator",
            "l5_policy": "isolated_singletons_only",
            "erf_policy": "charge_exact_through_r_collateral_d_seg",
            "same_joint_decode_required": True,
        },
        "families": families,
        "chosen_rung_distribution": distribution,
        "totals": {
            "family_count": len(families),
            "flip_count": M1_FLIP_COUNT,
            "kept_flip_count": len(per_flip["fixed_point"]["kept_flip_ids"]),
            "eaten_flip_count": len(per_flip["fixed_point"]["eaten_flip_ids"]),
            "candidate_selected_coded_bits": sum(row["coded_bits"] for row in selected_rows),
            "candidate_selected_coded_bytes": sum(row["coded_bytes"] for row in selected_rows),
            "admitted_coded_bits": sum(row["coded_bits"] for row in admitted_rows),
            "admitted_coded_bytes": sum(row["coded_bytes"] for row in admitted_rows),
            "eaten_avoided_coded_bits": sum(row["coded_bits"] for row in eaten_rows),
            "eaten_avoided_coded_bytes": sum(row["coded_bytes"] for row in eaten_rows),
            "admitted_positive_score_benefit": sum(row["positive_score_benefit"] for row in admitted_rows),
            "admitted_erf_collateral_flip_ids": [],
            "admitted_erf_collateral_flip_count": 0,
            "admitted_erf_collateral_positive_pixel_count": 0,
            "admitted_erf_collateral_d_seg": 0.0,
            "admitted_net_score_benefit": sum(row["net_score_benefit"] for row in admitted_rows),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def _synthetic_per_flip_sellback(joint_decode: dict) -> dict:
    """Build a schema-only fixture; this is not measured sellback evidence."""

    receiver_strata = ("cell_interior", "boundary_codim1", "movable_track", "critical_event")
    price_strata = ("road_lane", "other_edge", "non_edge", "tight_margin")
    flip_ids = [f"flip:{index:05d}" for index in range(M1_FLIP_COUNT)]
    kept_ids = flip_ids[1:]
    flips = []
    score_value = 100 / M1_SCORE_DENOMINATOR
    for index, flip_id in enumerate(flip_ids):
        coded_bits = 16 if index == 0 else 8
        flips.append(
            {
                "flip_id": flip_id,
                "stratum": receiver_strata[index % len(receiver_strata)],
                "price_stratum": price_strata[index % len(price_strata)],
                "positive_pixel_count": 1,
                "coded_bits": coded_bits,
                "derived_score_value": score_value,
                "score_per_coded_byte": score_value / (coded_bits / 8),
                "decision": "eat" if index == 0 else "keep",
                "coded_bits_by_iteration": (
                    [{"iteration_index": 0, "coded_bits": coded_bits}]
                    if index == 0
                    else [
                        {"iteration_index": 0, "coded_bits": coded_bits},
                        {"iteration_index": 1, "coded_bits": coded_bits},
                    ]
                ),
            }
        )
    ledger = []
    for price_stratum in price_strata:
        rows = [flip for flip in flips if flip["price_stratum"] == price_stratum]
        kept = [flip for flip in rows if flip["decision"] == "keep"]
        eaten = [flip for flip in rows if flip["decision"] == "eat"]
        kept_bits = sum(flip["coded_bits"] for flip in kept)
        eaten_bits = sum(flip["coded_bits"] for flip in eaten)
        eaten_pixels = sum(flip["positive_pixel_count"] for flip in eaten)
        ledger.append(
            {
                "price_stratum": price_stratum,
                "kept_flip_count": len(kept),
                "eaten_flip_count": len(eaten),
                "kept_coded_bits": kept_bits,
                "eaten_coded_bits": eaten_bits,
                "kept_coded_bytes": kept_bits / 8,
                "eaten_coded_bytes": eaten_bits / 8,
                "eaten_positive_pixel_count": eaten_pixels,
                "exact_d_seg_conceded": eaten_pixels / M1_SCORE_DENOMINATOR,
            }
        )
    result = {
        "schema": "predict_project_per_flip_sellback.v0",
        "measurement_status": "MEASURED_ITERATIVE_RECODE_FIXED_POINT",
        "scope": "m1_all_17926_flips_same_joint_decode",
        "m1_receipt": _m1_receipt_binding(),
        "joint_decode": dict(joint_decode),
        "context_model": {
            "model_id": "#557",
            "source_sha256": _digest("context-source"),
            "model_content_sha256": _digest("context-model"),
            "coded_bits_are_exact_integers": True,
        },
        "survival_histogram": {"surface_id": "r1b7", "content_sha256": _digest("r1b7-histogram")},
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "score_formula": {
            "numerator": 100,
            "denominator": M1_SCORE_DENOMINATOR,
            "expression": "positive_pixel_count*100/(600*512*384)",
        },
        "flip_count": M1_FLIP_COUNT,
        "stratum_price_anchors_bytes_per_event": {
            "road_lane": 2.48,
            "other_edge": 2.20,
            "non_edge": 4.42,
            "tight_margin": 3.36,
        },
        "flips": flips,
        "iterations": [
            {
                "iteration_index": 0,
                "input_kept_flip_ids": flip_ids,
                "output_kept_flip_ids": kept_ids,
                "context_sha256": _digest("context-0"),
                "coded_stream_sha256": _digest("stream-0"),
                "total_coded_bits": 16 + (M1_FLIP_COUNT - 1) * 8,
                "stable": False,
            },
            {
                "iteration_index": 1,
                "input_kept_flip_ids": kept_ids,
                "output_kept_flip_ids": kept_ids,
                "context_sha256": _digest("context-1"),
                "coded_stream_sha256": _digest("stream-1"),
                "total_coded_bits": (M1_FLIP_COUNT - 1) * 8,
                "stable": True,
            },
        ],
        "fixed_point": {
            "iteration_index": 1,
            "kept_flip_ids": kept_ids,
            "eaten_flip_ids": ["flip:00000"],
            "context_sha256": _digest("context-1"),
            "coded_stream_sha256": _digest("stream-1"),
            "stable": True,
        },
        "per_stratum_ledger": ledger,
        "action_level_ladder": None,
        "nonmonotone_context_observed": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    result["action_level_ladder"] = _synthetic_action_level_ladder(result, joint_decode)
    return result


def _synthetic_pose_tube_knee(joint_decode: dict) -> dict:
    """Build a schema-only fixture; this is not a measured Pose knee."""

    specs = [("pose-0", 0, 0.000102, 1000), ("pose-1", 1, 0.000103, 700), ("pose-2", 2, 0.000104, 600)]
    points = []
    previous = None
    for point_id, relaxation, d_pose, archive_bytes in specs:
        if previous is None:
            byte_savings = 0
            score_delta = 0.0
            marginal = None
        else:
            byte_savings = previous["archive_bytes"] - archive_bytes
            score_delta = (10 * d_pose) ** 0.5 - (10 * previous["d_pose"]) ** 0.5
            marginal = score_delta / byte_savings
        point = {
            "point_id": point_id,
            "tube_relaxation_q": relaxation,
            "d_pose": d_pose,
            "archive_bytes": archive_bytes,
            "byte_savings_from_previous": byte_savings,
            "nonlinear_sqrt_score_delta": score_delta,
            "marginal_score_per_byte": marginal,
            "archive_sha256": _digest(f"pose-archive-{relaxation}"),
            "decoded_output_sha256": _digest(f"pose-output-{relaxation}"),
        }
        points.append(point)
        previous = point
    return {
        "schema": "predict_project_pose_tube_knee.v0",
        "measurement_status": "MEASURED_SAME_JOINT_DECODE_POSE_TUBE_SWEEP",
        "scope": "full_n600_increasing_pose_tube_relaxation",
        "m1_receipt": _m1_receipt_binding(),
        "joint_decode": dict(joint_decode),
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "pose_score_term": "sqrt(10*d_pose)",
        "points": points,
        "selected_crossing": {
            "selected_point_id": "pose-1",
            "next_rejected_point_id": "pose-2",
            "selection_policy": "last_marginal_le_lambda_before_first_gt_lambda",
        },
        "kkt_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _synthetic_attribution_edit_telemetry(per_flip: dict, action_ladder: dict) -> dict:
    joint_sha = hashlib.sha256(canonical_json_bytes(per_flip["joint_decode"])).hexdigest()
    family_by_flip = {flip_id: family for family in action_ladder["families"] for flip_id in family["member_flip_ids"]}
    bindings = [
        {
            "flip_id": flip["flip_id"],
            "family_id": family_by_flip[flip["flip_id"]]["family_id"],
            "receipt_sha256": _digest(f"attribution:{flip['flip_id']}"),
        }
        for flip in per_flip["flips"]
    ]
    edits = []
    for family in action_ladder["families"]:
        for rung in family["rungs"]:
            request = {
                "schema": "predict_project_ladder_edit_request.v0",
                "request_id": f"edit:{family['family_id']}:{rung['rung']}",
                "family_id": family["family_id"],
                "rung": rung["rung"],
                "procedure_id": rung["actuator_id"],
                "parameters_sha256": _digest(f"parameters:{family['family_id']}:{rung['rung']}"),
                "before_archive_sha256": _digest(f"before-archive:{family['family_id']}:{rung['rung']}"),
                "joint_decode_sha256": joint_sha,
                "family_membership_sha256": family["family_membership_sha256"],
            }
            response = {
                "schema": "predict_project_ladder_edit_response.v0",
                "measurement_status": "MEASURED_DETERMINISTIC_REDECODE_THROUGH_R",
                "request_id": request["request_id"],
                "deterministic_redecode": True,
                "joint_decode_sha256": joint_sha,
                "before_archive_sha256": request["before_archive_sha256"],
                "after_archive_sha256": rung["archive_sha256"],
                "before_output_sha256": _digest(f"before-output:{family['family_id']}:{rung['rung']}"),
                "after_output_sha256": rung["decoded_output_sha256"],
                "delta_score": -rung["net_score_benefit"],
                "delta_bytes": rung["coded_bytes"],
                "erf_collateral_flip_ids": rung["erf_collateral_flip_ids"],
                "erf_collateral_flip_count": rung["erf_collateral_flip_count"],
                "erf_collateral_positive_pixel_count": rung["erf_collateral_positive_pixel_count"],
                "erf_collateral_d_seg": rung["erf_collateral_d_seg"],
                "binding_proof_sha256": _digest(f"binding:{family['family_id']}:{rung['rung']}"),
                "artifact_contract_sha256": _digest(f"artifact:{family['family_id']}:{rung['rung']}"),
                "score_claim": False,
                "promotion_eligible": False,
            }
            edits.append(
                {
                    "family_id": family["family_id"],
                    "rung": rung["rung"],
                    "action_rung_sha256": hashlib.sha256(canonical_json_bytes(rung)).hexdigest(),
                    "request": request,
                    "response": response,
                    "receipt_sha256": hashlib.sha256(
                        canonical_json_bytes({"request": request, "response": response})
                    ).hexdigest(),
                }
            )
    return {
        "schema": "predict_project_attribution_edit_telemetry.v0",
        "measurement_status": "MEASURED_EXACT_ATTRIBUTION_AND_LADDER_EDITS",
        "scope": "m1_all_17926_flips_all_five_rungs",
        "joint_decode_sha256": joint_sha,
        "reuse_bindings": copy.deepcopy(ATTRIBUTION_REUSE_BINDINGS),
        "action_ladder_sha256": hashlib.sha256(canonical_json_bytes(action_ladder)).hexdigest(),
        "flip_attribution_bindings": bindings,
        "edit_receipts": edits,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _synthetic_learned_tail_race(custody: dict, joint_decode: dict) -> dict:
    races = []
    for index, stream in enumerate(GLOBAL_WATERFILL_STREAMS):
        literal_bytes = 30 + index
        generator_bytes = 1
        pixels = 1
        eaten_score = 100 * (pixels / M1_SCORE_DENOMINATOR)
        common_score = 0.0
        specs = [
            {
                "option": "literal_exceptions",
                "exact_bytes": literal_bytes,
                "delta_score": common_score,
                "breakdown": {"literal_exception_bytes": literal_bytes},
            },
            {
                "option": "learned_generator",
                "exact_bytes": generator_bytes,
                "delta_score": common_score,
                "breakdown": {
                    "counted_weight_bytes": generator_bytes,
                    "instance_seed_bytes": 0,
                    "own_exception_bytes": 0,
                    "weights_sha256": _digest(f"weights:{stream}"),
                    "instance_seed_sha256": _digest(f"seed:{stream}"),
                    "own_exceptions_sha256": _digest(f"exceptions:{stream}"),
                },
            },
            {
                "option": "eaten_flip",
                "exact_bytes": 0,
                "delta_score": eaten_score,
                "breakdown": {
                    "flip_ids": ["flip:00000"],
                    "positive_pixel_count": pixels,
                    "exact_d_seg": pixels / M1_SCORE_DENOMINATOR,
                    "score_cost": eaten_score,
                },
            },
        ]
        for alternative in specs:
            alternative["lagrangian_cost"] = (
                alternative["delta_score"] + GLOBAL_WATERFILL_LAMBDA_STAR * alternative["exact_bytes"]
            )
        winner = min(specs, key=lambda alternative: alternative["lagrangian_cost"])["option"]
        races.append(
            {
                "stream": stream,
                "equal_realized_fidelity_sha256": _digest(f"fidelity:{stream}"),
                "hard_oracle_output_sha256": _digest(f"oracle:{stream}"),
                "alternatives": specs,
                "winner": winner,
            }
        )
    admitted = [race["stream"] for race in races if race["winner"] == "learned_generator"]
    return {
        "schema": "predict_project_learned_tail_three_way_race.v0",
        "measurement_status": "MEASURED_EQUAL_FIDELITY_THREE_WAY_RACE",
        "scope": "same_joint_decode_all_global_streams",
        "streams": list(GLOBAL_WATERFILL_STREAMS),
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "custody_sha256": hard_oracle_custody_sha256(custody),
        "joint_decode_sha256": hashlib.sha256(canonical_json_bytes(joint_decode)).hexdigest(),
        "trainer_reuse": copy.deepcopy(S3_TRAINER_REUSE),
        "rule118": {
            "generic_generator_compute_is_free": True,
            "generator_weights_are_counted_payload": True,
            "instance_seeds_are_counted_payload": True,
            "own_exceptions_are_counted_payload": True,
            "training_or_launch_performed": False,
        },
        "stream_races": races,
        "admitted_streams": admitted,
        "learned_default": "ABSENT_UNLESS_GENERATOR_STRICTLY_WINS",
        "score_claim": False,
        "promotion_eligible": False,
    }


def _global_waterfill_evidence() -> dict:
    value = seed()
    custody = _adapter_custody(valid_hard_oracle_without_sweep, value)
    joint_decode = {
        "decoder_id": "test.joint.decoder",
        "decoder_source_sha256": _digest("decoder"),
        "base_archive_sha256": _digest("base-archive"),
        "source_cache_sha256": _digest("source-cache"),
    }
    sellback = _synthetic_per_flip_sellback(joint_decode)
    kept_flip_ids = sellback["fixed_point"]["kept_flip_ids"]
    empty_flips = {stream: [] for stream in GLOBAL_WATERFILL_STREAMS}
    point0_flips = {stream: [] for stream in GLOBAL_WATERFILL_STREAMS}
    point0_flips["eat_flip"] = kept_flip_ids
    points = [
        {
            "point_id": "p0",
            "ordered_streams": list(GLOBAL_WATERFILL_STREAMS),
            "settings_sha256": _digest("settings-0"),
            "archive_sha256": _digest("archive-0"),
            "decoded_output_sha256": _digest("decode-0"),
            "delta_score": -0.001,
            "delta_bytes": 100,
            "stream_flip_ids": point0_flips,
            "credited_flip_ids": kept_flip_ids,
        },
        {
            "point_id": "p1",
            "ordered_streams": list(reversed(GLOBAL_WATERFILL_STREAMS)),
            "settings_sha256": _digest("settings-1"),
            "archive_sha256": _digest("archive-1"),
            "decoded_output_sha256": _digest("decode-1"),
            "delta_score": -0.0005,
            "delta_bytes": 50,
            "stream_flip_ids": empty_flips,
            "credited_flip_ids": [],
        },
    ]
    curves = [
        {
            "stream": stream,
            "points": [
                {"point_id": "p1", "delta_score": -0.0005, "delta_bytes": 50},
                {"point_id": "p0", "delta_score": -0.001, "delta_bytes": 100},
            ],
        }
        for stream in GLOBAL_WATERFILL_STREAMS
    ]
    action_ladder = copy.deepcopy(sellback["action_level_ladder"])
    return {
        "schema": "predict_project_global_joint_waterfill_evidence.v0",
        "measurement_status": "MEASURED_GLOBAL_JOINT_SWEEP",
        "scope": "aggregate_same_joint_decode_full_600",
        "pair_range": [0, 600],
        "pair_count": 600,
        "same_joint_decode": True,
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "m1_anchors": dict(M1_ANCHORS),
        "streams": list(GLOBAL_WATERFILL_STREAMS),
        "custody": custody,
        "joint_decode": joint_decode,
        "overlap_credit_policy": "global_flip_id_union_once.v1",
        "composition_policy": "ordered_commutator_aware.v1",
        "interaction_definition": "delta_joint_minus_sum_of_singles.v1",
        "points": points,
        "per_stream_marginal_curves": curves,
        "global_allocation": {
            "selected_point_id": "p0",
            "order": list(GLOBAL_WATERFILL_STREAMS),
            "admitted_streams": list(GLOBAL_WATERFILL_STREAMS),
            "admitted_flip_ids": kept_flip_ids,
        },
        "per_flip_sellback": sellback,
        "action_level_ladder": action_ladder,
        "attribution_edit_telemetry": _synthetic_attribution_edit_telemetry(sellback, action_ladder),
        "learned_tail_race": _synthetic_learned_tail_race(custody, joint_decode),
        "pose_tube_knee": _synthetic_pose_tube_knee(joint_decode),
        "eaten_flip_decomposition": {
            "eaten_flip_ids": ["flip:00000"],
            "flip_count": 1,
            "coded_bits": 16,
            "coded_bytes": 2.0,
            "by_stratum": [
                {
                    "stratum": "cell_interior",
                    "flip_ids": ["flip:00000"],
                    "flip_count": 1,
                    "coded_bits": 16,
                    "coded_bytes": 2.0,
                    "d_seg_cost": 1 / M1_SCORE_DENOMINATOR,
                },
                {
                    "stratum": "boundary_codim1",
                    "flip_ids": [],
                    "flip_count": 0,
                    "coded_bits": 0,
                    "coded_bytes": 0.0,
                    "d_seg_cost": 0.0,
                },
                {
                    "stratum": "movable_track",
                    "flip_ids": [],
                    "flip_count": 0,
                    "coded_bits": 0,
                    "coded_bytes": 0.0,
                    "d_seg_cost": 0.0,
                },
                {
                    "stratum": "critical_event",
                    "flip_ids": [],
                    "flip_count": 0,
                    "coded_bits": 0,
                    "coded_bytes": 0.0,
                    "d_seg_cost": 0.0,
                },
            ],
            "total_d_seg_cost": 1 / M1_SCORE_DENOMINATOR,
        },
        "pairwise_interaction_matrix": [[0.0 for _ in GLOBAL_WATERFILL_STREAMS] for _ in GLOBAL_WATERFILL_STREAMS],
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_global_joint_waterfill_requires_same_decode_and_credits_overlap_once():
    assert global_joint_waterfill({})["status"] == "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP"
    evidence = _global_waterfill_evidence()
    validated = validate_global_joint_waterfill_evidence(evidence)
    assert len(validated["points"][0]["credited_flip_ids"]) == M1_FLIP_COUNT - 1
    result = global_joint_waterfill(evidence)
    assert result["status"] == "MEASURED_GLOBAL_JOINT_SWEEP"
    assert result["per_flip_sellback_status"] == "MEASURED_ITERATIVE_RECODE_FIXED_POINT"
    assert result["pose_tube_knee_status"] == "MEASURED_SAME_JOINT_DECODE_POSE_TUBE_SWEEP"
    assert result["independent_curve_authority"] is False


def test_synthetic_per_flip_sellback_fixture_validates_exact_census_and_receipt_binding():
    evidence = _global_waterfill_evidence()["per_flip_sellback"]
    validated = validate_per_flip_sellback_evidence(evidence)
    assert len(validated["flips"]) == 17_926
    assert validated["m1_receipt"]["commit"] == M1_RECEIPT_COMMIT
    assert validated["m1_receipt"]["sha256"] == M1_RECEIPT_SHA256
    assert validated["nonmonotone_context_observed"] is False
    assert validated["action_level_ladder"]["measurement_status"] == ("MEASURED_SAME_JOINT_DECODE_ACTION_LEVEL_LADDER")
    assert sum(row["flip_count"] for row in validated["action_level_ladder"]["chosen_rung_distribution"]) == 17_926
    assert validated["action_level_ladder"]["boundary_inverse_policy"] == BOUNDARY_INVERSE_ACTION_POLICY
    assert (
        validate_action_level_ladder_evidence(validated["action_level_ladder"], validated)["totals"]["flip_count"]
        == M1_FLIP_COUNT
    )


def test_action_level_ladder_rejects_missing_rung_illegal_l5_wrong_choice_and_bad_distribution():
    evidence = _synthetic_per_flip_sellback(
        {
            "decoder_id": "test.joint.decoder",
            "decoder_source_sha256": _digest("decoder"),
            "base_archive_sha256": _digest("base-archive"),
            "source_cache_sha256": _digest("source-cache"),
        }
    )

    missing_rung = copy.deepcopy(evidence)
    missing_rung["action_level_ladder"]["families"][0]["rungs"].pop()
    with pytest.raises(PredictProjectReceiverError, match="all five rungs"):
        validate_per_flip_sellback_evidence(missing_rung)

    non_singleton = next(
        family for family in evidence["action_level_ladder"]["families"] if not family["isolated_singleton"]
    )
    illegal_l5 = copy.deepcopy(evidence)
    illegal_family = next(
        family
        for family in illegal_l5["action_level_ladder"]["families"]
        if family["family_id"] == non_singleton["family_id"]
    )
    illegal_l5_row = next(row for row in illegal_family["rungs"] if row["rung"] == "L5_pixel_write")
    illegal_l5_row["valid"] = True
    illegal_l5_row["invalid_reason"] = None
    with pytest.raises(PredictProjectReceiverError, match="L5 is valid only"):
        validate_per_flip_sellback_evidence(illegal_l5)

    wrong_choice = copy.deepcopy(evidence)
    wrong_choice["action_level_ladder"]["families"][0]["selected_rung"] = "L2_channel"
    with pytest.raises(PredictProjectReceiverError, match="deterministically cheapest"):
        validate_per_flip_sellback_evidence(wrong_choice)

    bad_distribution = copy.deepcopy(evidence)
    bad_distribution["action_level_ladder"]["chosen_rung_distribution"][0]["flip_count"] += 1
    with pytest.raises(PredictProjectReceiverError, match="chosen-rung distribution"):
        validate_per_flip_sellback_evidence(bad_distribution)


def test_action_level_ladder_rejects_collateral_and_global_reconciliation_drift():
    global_evidence = _global_waterfill_evidence()
    evidence = global_evidence["per_flip_sellback"]
    bad_collateral = copy.deepcopy(evidence)
    family = bad_collateral["action_level_ladder"]["families"][0]
    rung = family["rungs"][0]
    rung["erf_collateral_flip_ids"] = ["flip:00001"]
    rung["erf_collateral_flip_count"] = 1
    rung["erf_collateral_positive_pixel_count"] = 1
    with pytest.raises(PredictProjectReceiverError, match="collateral d_seg"):
        validate_per_flip_sellback_evidence(bad_collateral)

    global_mismatch = copy.deepcopy(global_evidence)
    global_mismatch["action_level_ladder"]["families"][0]["selected_rung"] = "L2_channel"
    with pytest.raises(PredictProjectReceiverError, match="deterministically cheapest"):
        validate_global_joint_waterfill_evidence(global_mismatch)


def test_full_flip_attribution_and_edit_telemetry_bind_exact_causal_chain_and_rungs():
    global_evidence = _global_waterfill_evidence()
    telemetry = global_evidence["attribution_edit_telemetry"]
    validated = validate_attribution_edit_telemetry(
        telemetry,
        global_evidence["per_flip_sellback"],
        global_evidence["action_level_ladder"],
    )
    assert len(validated["flip_attribution_bindings"]) == M1_FLIP_COUNT
    edit = validated["edit_receipts"][0]
    assert validate_ladder_edit_request(edit["request"])["rung"] == edit["rung"]
    assert validate_ladder_edit_response(edit["request"], edit["response"])["deterministic_redecode"] is True

    flip_id = validated["flip_attribution_bindings"][0]["flip_id"]
    family_id = validated["flip_attribution_bindings"][0]["family_id"]
    receipt = {
        "schema": "predict_project_flip_attribution.v0",
        "measurement_status": "MEASURED_EXACT_ATTRIBUTION_SAME_JOINT_DECODE",
        "flip_id": flip_id,
        "family_id": family_id,
        "joint_decode_sha256": telemetry["joint_decode_sha256"],
        "reuse_bindings": copy.deepcopy(ATTRIBUTION_REUSE_BINDINGS),
        "chain_order": ["chart_coefficient", "channel", "rank4_hyperplane", "regional_values", "pixels"],
        "causal_chain": {
            "chart_coefficient": {
                "coefficient_ids": ["chart:0"],
                "before_sha256": _digest("chart-before"),
                "after_sha256": _digest("chart-after"),
            },
            "channel": {
                "carrier_id": "carrier:0",
                "channel_ids": [0],
                "before_sha256": _digest("channel-before"),
                "after_sha256": _digest("channel-after"),
            },
            "rank4_hyperplane": {
                "law_id": SEGNET_HEAD_RANK_EQUATION_ID,
                "feature_q_before": [0] * SEGNET_CENTERED_HEAD_RANK,
                "feature_q_after": [1] + [0] * (SEGNET_CENTERED_HEAD_RANK - 1),
                "delta_w_q": [1] + [0] * (SEGNET_CENTERED_HEAD_RANK - 1),
                "signed_distance_q_before": -1,
                "signed_distance_q_after": 1,
            },
            "regional_values": {
                "region_id": "region:0",
                "before_sha256": _digest("region-before"),
                "after_sha256": _digest("region-after"),
            },
            "pixels": {
                "pixel_ids": ["pixel:0:0"],
                "realized_flip_ids": [flip_id],
                "before_sha256": _digest("pixels-before"),
                "after_sha256": _digest("pixels-after"),
            },
        },
        "artifact_contract_sha256": _digest("artifact-contract"),
        "score_claim": False,
        "promotion_eligible": False,
    }
    assert validate_flip_attribution_receipt(receipt)["flip_id"] == flip_id
    broken_chain = copy.deepcopy(receipt)
    broken_chain["causal_chain"].pop("channel")
    with pytest.raises(PredictProjectReceiverError, match="causal chain"):
        validate_flip_attribution_receipt(broken_chain)

    drift = copy.deepcopy(telemetry)
    drift["edit_receipts"][0]["response"]["after_output_sha256"] = _digest("wrong-output")
    with pytest.raises(PredictProjectReceiverError, match="metrics/custody"):
        validate_attribution_edit_telemetry(
            drift,
            global_evidence["per_flip_sellback"],
            global_evidence["action_level_ladder"],
        )


def test_learned_tail_race_requires_all_streams_equal_fidelity_and_strict_unique_winner():
    evidence = _global_waterfill_evidence()
    custody_sha = hard_oracle_custody_sha256(evidence["custody"])
    joint_sha = hashlib.sha256(canonical_json_bytes(evidence["joint_decode"])).hexdigest()
    race = evidence["learned_tail_race"]
    validated = validate_learned_tail_race_evidence(race, custody_sha256=custody_sha, joint_decode_sha256=joint_sha)
    assert validated["admitted_streams"] == list(GLOBAL_WATERFILL_STREAMS)

    omitted = copy.deepcopy(race)
    omitted["stream_races"].pop()
    with pytest.raises(PredictProjectReceiverError, match="cover every canonical stream"):
        validate_learned_tail_race_evidence(omitted, custody_sha256=custody_sha, joint_decode_sha256=joint_sha)

    unequal = copy.deepcopy(race)
    unequal["stream_races"][0]["alternatives"][1]["delta_score"] = 1.0
    unequal["stream_races"][0]["alternatives"][1]["lagrangian_cost"] = (
        1.0 + GLOBAL_WATERFILL_LAMBDA_STAR * unequal["stream_races"][0]["alternatives"][1]["exact_bytes"]
    )
    with pytest.raises(PredictProjectReceiverError, match="equal realized fidelity"):
        validate_learned_tail_race_evidence(unequal, custody_sha256=custody_sha, joint_decode_sha256=joint_sha)

    tie = copy.deepcopy(race)
    literal = tie["stream_races"][0]["alternatives"][0]
    generator = tie["stream_races"][0]["alternatives"][1]
    literal["exact_bytes"] = generator["exact_bytes"]
    literal["breakdown"]["literal_exception_bytes"] = generator["exact_bytes"]
    literal["lagrangian_cost"] = generator["lagrangian_cost"]
    with pytest.raises(PredictProjectReceiverError, match="unique deterministic winner"):
        validate_learned_tail_race_evidence(tie, custody_sha256=custody_sha, joint_decode_sha256=joint_sha)

    fixed_point_drift = _global_waterfill_evidence()
    eaten = fixed_point_drift["learned_tail_race"]["stream_races"][0]["alternatives"][2]["breakdown"]
    eaten["flip_ids"] = ["flip:not-in-fixed-point"]
    with pytest.raises(PredictProjectReceiverError, match="eaten alternative disagrees"):
        validate_global_joint_waterfill_evidence(fixed_point_drift)


def test_per_flip_sellback_rejects_threshold_chain_census_ledger_and_overlap_defects():
    joint_decode = _global_waterfill_evidence()["joint_decode"]

    above_threshold_eaten = _synthetic_per_flip_sellback(joint_decode)
    flip = above_threshold_eaten["flips"][0]
    flip["coded_bits"] = 8
    flip["coded_bits_by_iteration"][0]["coded_bits"] = 8
    flip["score_per_coded_byte"] = flip["derived_score_value"] / 1.0
    above_threshold_eaten["iterations"][0]["total_coded_bits"] -= 8
    with pytest.raises(PredictProjectReceiverError, match="lambda threshold"):
        validate_per_flip_sellback_evidence(above_threshold_eaten)

    below_threshold_kept = _synthetic_per_flip_sellback(joint_decode)
    flip = below_threshold_kept["flips"][1]
    flip["coded_bits"] = 16
    for history in flip["coded_bits_by_iteration"]:
        history["coded_bits"] = 16
    flip["score_per_coded_byte"] = flip["derived_score_value"] / 2.0
    below_threshold_kept["iterations"][0]["total_coded_bits"] += 8
    below_threshold_kept["iterations"][1]["total_coded_bits"] += 8
    with pytest.raises(PredictProjectReceiverError, match="lambda threshold"):
        validate_per_flip_sellback_evidence(below_threshold_kept)

    broken_chain = _synthetic_per_flip_sellback(joint_decode)
    broken_chain["iterations"][1]["input_kept_flip_ids"] = broken_chain["iterations"][1]["input_kept_flip_ids"][1:]
    broken_chain["iterations"][1]["output_kept_flip_ids"] = broken_chain["iterations"][1]["output_kept_flip_ids"][1:]
    with pytest.raises(PredictProjectReceiverError, match="chain is broken"):
        validate_per_flip_sellback_evidence(broken_chain)

    missing_flip = _synthetic_per_flip_sellback(joint_decode)
    missing_flip["flips"].pop()
    with pytest.raises(PredictProjectReceiverError, match="all 17926 flips"):
        validate_per_flip_sellback_evidence(missing_flip)

    inconsistent_ledger = _synthetic_per_flip_sellback(joint_decode)
    inconsistent_ledger["per_stratum_ledger"][0]["eaten_flip_count"] += 1
    with pytest.raises(PredictProjectReceiverError, match="counts/bytes/d_seg"):
        validate_per_flip_sellback_evidence(inconsistent_ledger)

    overlapping_fixed_point = _synthetic_per_flip_sellback(joint_decode)
    overlapping_fixed_point["fixed_point"]["kept_flip_ids"] = ["flip:00000"] + overlapping_fixed_point["fixed_point"][
        "kept_flip_ids"
    ]
    with pytest.raises(PredictProjectReceiverError, match="fixed point"):
        validate_per_flip_sellback_evidence(overlapping_fixed_point)


def test_per_flip_sellback_records_measured_nonmonotone_context_without_rejecting_it():
    joint_decode = _global_waterfill_evidence()["joint_decode"]
    evidence = _synthetic_per_flip_sellback(joint_decode)
    flip = evidence["flips"][1]
    flip["coded_bits"] = 9
    flip["coded_bits_by_iteration"][1]["coded_bits"] = 9
    flip["score_per_coded_byte"] = flip["derived_score_value"] / (9 / 8)
    evidence["iterations"][1]["total_coded_bits"] += 1
    evidence["per_stratum_ledger"][1]["kept_coded_bits"] += 1
    evidence["per_stratum_ledger"][1]["kept_coded_bytes"] += 1 / 8
    evidence["nonmonotone_context_observed"] = True
    assert validate_per_flip_sellback_evidence(evidence)["nonmonotone_context_observed"] is True


def test_synthetic_pose_tube_knee_derives_nonlinear_crossing_and_rejects_fabrication():
    joint_decode = _global_waterfill_evidence()["joint_decode"]
    evidence = _synthetic_pose_tube_knee(joint_decode)
    validated = validate_pose_tube_knee_evidence(evidence)
    assert validated["selected_crossing"]["selected_point_id"] == "pose-1"
    assert validated["kkt_claim"] is False

    inconsistent = _synthetic_pose_tube_knee(joint_decode)
    inconsistent["points"][1]["nonlinear_sqrt_score_delta"] = 0.0
    with pytest.raises(PredictProjectReceiverError, match="sqrt score delta"):
        validate_pose_tube_knee_evidence(inconsistent)

    fabricated_crossing = _synthetic_pose_tube_knee(joint_decode)
    fabricated_crossing["selected_crossing"]["selected_point_id"] = "pose-0"
    with pytest.raises(PredictProjectReceiverError, match="selected crossing"):
        validate_pose_tube_knee_evidence(fabricated_crossing)


def test_independent_or_cross_pair_waterfill_evidence_cannot_pass():
    evidence = _global_waterfill_evidence()
    evidence["scope"] = "cross_pair_independent_curves"
    evidence["same_joint_decode"] = False
    with pytest.raises(PredictProjectReceiverError, match="scope/policy"):
        validate_global_joint_waterfill_evidence(evidence)
    evidence = _global_waterfill_evidence()
    evidence["points"][0]["credited_flip_ids"] = ["flip:shared", "flip:shared"]
    with pytest.raises(PredictProjectReceiverError, match="credited exactly once"):
        validate_global_joint_waterfill_evidence(evidence)


def test_global_waterfill_rejects_omitted_stream_inconsistent_allocation_and_missing_eaten_flips():
    evidence = _global_waterfill_evidence()
    evidence["per_stream_marginal_curves"].pop()
    with pytest.raises(PredictProjectReceiverError, match="cover all canonical streams"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    evidence["global_allocation"]["selected_point_id"] = "p1"
    with pytest.raises(PredictProjectReceiverError, match="Lagrangian minimum"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    del evidence["eaten_flip_decomposition"]
    with pytest.raises(PredictProjectReceiverError, match="fields mismatch"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    evidence["eaten_flip_decomposition"] = copy.deepcopy(evidence["eaten_flip_decomposition"])
    evidence["eaten_flip_decomposition"]["by_stratum"].pop()
    with pytest.raises(PredictProjectReceiverError, match="cover all canonical strata"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    evidence["eaten_flip_decomposition"]["by_stratum"][0]["d_seg_cost"] = 0.0
    with pytest.raises(PredictProjectReceiverError, match="counts/bytes/dseg"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    del evidence["per_flip_sellback"]
    with pytest.raises(PredictProjectReceiverError, match="fields mismatch"):
        validate_global_joint_waterfill_evidence(evidence)

    evidence = _global_waterfill_evidence()
    del evidence["pose_tube_knee"]
    with pytest.raises(PredictProjectReceiverError, match="fields mismatch"):
        validate_global_joint_waterfill_evidence(evidence)


def test_global_waterfill_cannot_enter_receipt_without_full_hard_oracle(tmp_path):
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(seed()))
    with pytest.raises(MeasurementError, match="real full-600 hard-oracle"):
        run_measurement(
            seed_path,
            tmp_path / "global-without-hard-oracle",
            pair_end=1,
            global_waterfill_evidence=_global_waterfill_evidence(),
        )
    assert not (tmp_path / "global-without-hard-oracle" / "receipt.json").exists()


def test_proved_pose_tightening_is_shippable_and_unproved_tightening_rejected():
    value = seed()
    tightening = {
        "schema": "predict_project_pose_tightening.v0",
        "tightening_id": 0,
        "time": 0,
        "frame_index": 0,
        "pixel_coordinates": [{"y": 0, "x": 0, "channel": 0}],
        "lower_u8": [0],
        "upper_u8": [255],
        "linear_constraints": [{"constraint_id": "pixel-cap", "coefficients_q": [1], "upper_q": 255}],
        "pose_tube": {"lower_q": [-1], "upper_q": [1]},
        "universal_within_box_and_constraints_implies_pose_tube": True,
    }
    assertion = {key: tightening[key] for key in tightening if key != "schema"}
    tightening["proof"] = {
        "schema": "predict_project_pose_tightening_proof.v0",
        "status": "PROVED_BY_HARD_ORACLE",
        "method": "fixture-exhaustive-integer-box",
        "assertion_sha256": hashlib.sha256(canonical_json_bytes(assertion)).hexdigest(),
        "custody": _adapter_custody(valid_hard_oracle_without_sweep, value),
    }
    value["pose_tightening"].append(tightening)
    value["constraint_seeds"].append(
        {
            "time": 0,
            "frame_index": 0,
            "obligation": "pose_only",
            "y": None,
            "x": None,
            "cell_id": None,
            "predictor_status": None,
            "stratum": None,
            "pose_tube": {"lower_q": [-1], "upper_q": [1]},
            "pose_tightening_id": 0,
            "projector": None,
        }
    )
    validate_constraint_seed(value)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    result = verify_pose_tightening_choice(value, frame, time=0, frame_index=0)
    assert result["verified_tightenings"][0]["universal_pose_tube_admission"] is True
    assert result["decoder_scorer_invocations"] == 0
    value["pose_tightening"][0]["universal_within_box_and_constraints_implies_pose_tube"] = False
    with pytest.raises(PredictProjectSchemaError, match="universal tube assertion"):
        verify_pose_tightening_choice(value, frame, time=0, frame_index=0)


def test_b5_per_frame_represents_declared_constraint_cell(tmp_path):
    constrained = build_minimal_constraint_seed(
        bytes([0, 1, 2, 3, 4, 0]),
        scorer_height=2,
        scorer_width=3,
        camera_height=4,
        camera_width=6,
        constraint_seeds=[
            {
                "time": 0,
                "frame_index": 1,
                "y": 0,
                "x": 0,
                "cell_id": 4,
                "predictor_status": "violated",
                "stratum": "boundary_codim1",
                "pose_tube": None,
                "projector": None,
            }
        ],
    )
    seed_path = tmp_path / "seed.ppcs"
    seed_path.write_bytes(serialize_constraint_seed(constrained))
    receipt = run_measurement(seed_path, tmp_path / "b5", pair_end=1)
    represented = predict_cell_field(constrained, 0)
    represented[0, 0] = 4
    rows = [{"pair_index": 0, "cell_field_hex": represented.tobytes().hex()}]
    expected_sha256 = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    assert receipt["b5"]["per_frame"]["representation"] == "desired_from_declared_constraints"
    assert receipt["b5"]["per_frame"]["applied_constraint_cells"] == 1
    assert receipt["b5"]["per_frame"]["rows_sha256"] == expected_sha256
