from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tac.scorer_surrogate import whole_teacher_distilled_student as student


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _repo_file_custody(path: str) -> dict[str, object]:
    payload = f"TEST-ONLY repository custody for {path}".encode()
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _test_sidecar_custody(path: str) -> dict[str, object]:
    payload = f"TEST-ONLY sidecar custody for {path}".encode()
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _teacher_source_custody() -> dict[str, object]:
    r_operator: dict[str, object] = {
        "identity": student.R_OPERATOR_IDENTITY,
        "source": _repo_file_custody("src/tac/cuda_levelset_training.py"),
        "camera_hw": [874, 1164],
        "output_hw": [384, 512],
        "up_interpolation": "bicubic",
        "quantization": "round_ste_then_clamp_0_255",
        "down_interpolation": "bilinear",
        "align_corners": False,
        "input_surface": "float32_rgb_nhwc_0_255",
        "output_surface": "post_r_float32_rgb_nhwc_0_255",
    }
    r_operator["r_operator_sha256"] = _semantic_sha256(r_operator)

    scorer: dict[str, object] = {
        "architecture_identity": student.SEGNET_ARCHITECTURE_IDENTITY,
        "architecture_source": _repo_file_custody("upstream/modules.py"),
        "weights": _repo_file_custody("upstream/models/segnet.safetensors"),
        "frozen": True,
        "class_count": 5,
        "input_surface": "post_r_float32_rgb_nchw_0_255_384x512",
        "logit_surface": "float32_logits_nchw_5x384x512",
        "preprocess_identity": (
            "same_state_last_frame_then_bilinear_384x512_align_corners_false"
        ),
    }
    scorer["scorer_sha256"] = _semantic_sha256(scorer)

    objective: dict[str, object] = {
        "identity": student.SCALAR_OBJECTIVE_IDENTITY,
        "target": "centered_logit_decision_quotient_4d",
        "quotient_basis_version": "orthonormal_helmert_5x4_v1",
        "quotient_basis_sha256": student.HELMERT_BASIS_SHA256,
        "logit_lift": "zero_sum_helmert_lift_4_to_5",
        "loss": "cross_entropy",
        "reduction": "mean_over_batch_height_width",
        "label_semantics": "same_replay_integer_class_ids_0_to_4",
        "class_count": 5,
        "costate_surface": student.COSTATE_SURFACE_IDENTITY,
        "costate_units": (
            "d_mean_ce_per_post_r_rgb_code_value_where_one_unit_equals_one_of_255"
        ),
        "teacher_value_artifact": "teacher_quotient4",
        "teacher_costate_artifact": "teacher_input_costate",
        "same_scalar_for_value_and_costate": True,
        "post_r_input_surface_sha256": student._post_r_input_surface_sha256(
            r_operator["r_operator_sha256"]
        ),
        "r_operator_sha256": r_operator["r_operator_sha256"],
        "scorer_sha256": scorer["scorer_sha256"],
    }
    objective["scalar_objective_sha256"] = _semantic_sha256(objective)

    checkpoints = []
    checkpoint_paths = (
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz",
    )
    for (name, epoch), path in zip(student.CHECKPOINTS, checkpoint_paths, strict=True):
        checkpoints.append(
            {
                "checkpoint_name": name,
                "checkpoint_epoch": epoch,
                "source": _repo_file_custody(path),
            }
        )
    replay: dict[str, object] = {
        "source_kind": student.SOURCE_KIND,
        "renderer_identity": (
            "tools.dash_comb_probe_n600.Renderer_plus_task455_differentiable_chart"
        ),
        "renderer_source": _repo_file_custody("tools/dash_comb_probe_n600.py"),
        "replay_harness_source": _repo_file_custody(
            "tools/probe_frozen_replay_convex_head.py"
        ),
        "renderer_config": _test_sidecar_custody("custody/renderer_config.json"),
        "source_manifest": _test_sidecar_custody("custody/source_manifest.json"),
        "upstream_source_manifest": _test_sidecar_custody(
            "custody/upstream_source_manifest.json"
        ),
        "checkpoint_custody": checkpoints,
        "assignment_sha256": student._expected_assignment_sha256(),
        "r_operator_sha256": r_operator["r_operator_sha256"],
        "scorer_sha256": scorer["scorer_sha256"],
    }
    replay["replay_source_sha256"] = _semantic_sha256(replay)

    generation: dict[str, object] = {
        "axis": student.CACHE_GENERATION_AXIS,
        "teacher_backend": "torch",
        "teacher_device": "cpu",
        "rendered_state_count": 600,
        "mps_used": False,
        "synthetic_used": False,
        "source_video_substitute_used": False,
        "command_sha256": hashlib.sha256(b"TEST-ONLY generation argv").hexdigest(),
        "environment_sha256": hashlib.sha256(b"TEST-ONLY generation env").hexdigest(),
        "receipt": _test_sidecar_custody("custody/generation_receipt.json"),
    }
    generation["generation_sha256"] = _semantic_sha256(generation)

    custody: dict[str, object] = {
        "schema": student.TEACHER_SOURCE_CUSTODY_SCHEMA,
        "r_operator": r_operator,
        "frozen_segnet": scorer,
        "scalar_objective": objective,
        "replay_source": replay,
        "generation": generation,
    }
    custody["custody_sha256"] = _semantic_sha256(custody)
    return custody


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    values = logits.astype(np.float64)
    maximum = values.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(values - maximum).sum(axis=1)) + maximum[:, 0]
    selected = np.take_along_axis(values, labels[:, None], axis=1)[:, 0]
    return float(np.mean(logsumexp - selected))


def _manifest() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for pair_index in range(student.N600):
        checkpoint_index = pair_index % len(student.CHECKPOINTS)
        checkpoint_name, checkpoint_epoch = student.CHECKPOINTS[checkpoint_index]
        artifacts: dict[str, object] = {}
        specs = {
            "rendered_frame": ("<f4", student.FRAME_SHAPE),
            "teacher_quotient4": ("<f4", student.QUOTIENT_SHAPE),
            "teacher_input_costate": ("<f4", student.COSTATE_SHAPE),
            "labels": ("<i8", (384, 512)),
        }
        for name, (dtype, shape) in specs.items():
            artifacts[name] = {
                "path": f"pair_{pair_index:04d}/{name}.npy",
                "bytes": 1,
                "sha256": hashlib.sha256(
                    f"TEST-ONLY:{pair_index}:{name}".encode()
                ).hexdigest(),
                "dtype": dtype,
                "shape": list(shape),
            }
        rows.append(
            {
                "assignment_id": f"{checkpoint_name}:pair-{pair_index:04d}",
                "pair_index": pair_index,
                "checkpoint_index": checkpoint_index,
                "checkpoint_name": checkpoint_name,
                "checkpoint_epoch": checkpoint_epoch,
                "split": "heldout" if pair_index % 5 == 0 else "train",
                "source_kind": student.SOURCE_KIND,
                "artifacts": artifacts,
            }
        )
    return {
        "schema": student.CACHE_SCHEMA,
        "source_kind": student.SOURCE_KIND,
        "cohort_count": student.N600,
        "train_count": student.TRAIN_COUNT,
        "heldout_count": student.HELDOUT_COUNT,
        "checkpoint_epochs": [epoch for _name, epoch in student.CHECKPOINTS],
        "teacher_source_custody": _teacher_source_custody(),
        "rows": rows,
    }


def test_helmert_quotient_preserves_gauge_softmax_argmax_and_ce() -> None:
    rng = np.random.default_rng(455)
    logits = rng.standard_normal((2, 5, 3, 4), dtype=np.float32)
    gauge = rng.standard_normal((2, 1, 3, 4), dtype=np.float32) * np.float32(7.0)
    labels = rng.integers(0, 5, size=(2, 3, 4), dtype=np.int64)

    basis = student.HELMERT_BASIS_5X4.astype(np.float64)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        basis @ basis.T,
        np.eye(5) - np.ones((5, 5)) / 5.0,
        rtol=0.0,
        atol=2.0e-7,
    )

    quotient = student.quotient4_from_logits5_numpy(logits)
    shifted = student.quotient4_from_logits5_numpy(logits + gauge)
    np.testing.assert_allclose(quotient, shifted, rtol=2.0e-6, atol=2.0e-6)
    lifted = student.logits5_from_quotient4_numpy(quotient)
    centered = logits - logits.mean(axis=1, keepdims=True, dtype=np.float32)
    np.testing.assert_allclose(lifted, centered, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(lifted.sum(axis=1), 0.0, rtol=0.0, atol=5.0e-7)
    np.testing.assert_array_equal(logits.argmax(axis=1), lifted.argmax(axis=1))
    assert _cross_entropy(logits, labels) == pytest.approx(_cross_entropy(lifted, labels), abs=2e-7)


def test_deterministic_forward_layout_and_explicit_parameter_serialization() -> None:
    architecture = student.architecture_for_size("tiny")
    first_parameters = student.initialize_student_parameters(architecture, seed=455)
    second_parameters = student.initialize_student_parameters(architecture, seed=455)
    assert student.parameter_layout_sha256(architecture) == student.parameter_layout_sha256(
        architecture
    )
    for name in first_parameters:
        np.testing.assert_array_equal(first_parameters[name], second_parameters[name])

    frame = np.random.default_rng(9).uniform(0.0, 255.0, size=(1, 3, 5, 7)).astype(np.float32)
    first = student.student_forward_numpy(frame, architecture, first_parameters)
    second = student.student_forward_numpy(frame.copy(), architecture, second_parameters)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 4, 5, 7)

    blob_a = student.serialize_student_parameters(architecture, first_parameters)
    blob_b = student.serialize_student_parameters(architecture, second_parameters)
    assert blob_a == blob_b
    restored_architecture, restored_parameters = student.deserialize_student_parameters(blob_a)
    assert restored_architecture == architecture
    for name in first_parameters:
        np.testing.assert_array_equal(restored_parameters[name], first_parameters[name])

    corrupted = bytearray(blob_a)
    corrupted[-1] ^= 1
    with pytest.raises(student.StudentContractError, match="hash drifted"):
        student.deserialize_student_parameters(bytes(corrupted))


def test_forward_aggregation_exposes_single_bad_worst_pair() -> None:
    rng = np.random.default_rng(3)
    teacher = rng.standard_normal((1, 4, 3, 4), dtype=np.float32)
    rows = [
        student.forward_pair_metrics("good-a", teacher, teacher),
        student.forward_pair_metrics("bad", teacher, -teacher),
        student.forward_pair_metrics("good-b", teacher, teacher + np.float32(1.0e-5)),
    ]
    summary = student.aggregate_forward_pair_metrics(rows)
    assert summary["pair_count"] == 3
    assert summary["worst_relative_l2_assignment_id"] == "bad"
    assert summary["worst_cosine_assignment_id"] == "bad"
    assert summary["worst_argmax_disagreement_assignment_id"] == "bad"
    assert summary["worst_relative_l2"] == pytest.approx(2.0)


def test_cache_structure_requires_exact_sealed_n600_raw_replay_custody(tmp_path) -> None:
    manifest = _manifest()
    rows = student.validate_n600_cache_manifest_structure(manifest)
    assert len(rows) == 600
    assert len([row for row in rows if row.split == "train"]) == 480
    assert len([row for row in rows if row.split == "heldout"]) == 120
    assert {row.checkpoint_epoch for row in rows} == {150, 251, 275}

    wrong_schema = copy.deepcopy(manifest)
    wrong_schema["schema"] = "source_video_logits.v1"
    with pytest.raises(student.StudentContractError, match="unknown"):
        student.validate_n600_cache_manifest_structure(wrong_schema)

    n599 = copy.deepcopy(manifest)
    n599["rows"] = n599["rows"][:-1]  # type: ignore[index]
    with pytest.raises(student.StudentContractError, match="length 600"):
        student.validate_n600_cache_manifest_structure(n599)

    duplicate = copy.deepcopy(manifest)
    duplicate["rows"][1]["pair_index"] = 0  # type: ignore[index]
    with pytest.raises(student.StudentContractError, match="duplicated"):
        student.validate_n600_cache_manifest_structure(duplicate)

    missing_costate = copy.deepcopy(manifest)
    del missing_costate["rows"][0]["artifacts"]["teacher_input_costate"]  # type: ignore[index]
    with pytest.raises(student.StudentContractError, match="required raw tensor"):
        student.validate_n600_cache_manifest_structure(missing_costate)

    source_substitution = copy.deepcopy(manifest)
    source_substitution["source_kind"] = "source_video"
    with pytest.raises(student.StudentContractError, match="substitution"):
        student.validate_n600_cache_manifest_structure(source_substitution)

    missing_semantic_custody = copy.deepcopy(manifest)
    del missing_semantic_custody["teacher_source_custody"]
    with pytest.raises(student.StudentContractError, match="top-level keys"):
        student.validate_n600_cache_manifest_structure(missing_semantic_custody)

    rotated_quotient_basis = copy.deepcopy(manifest)
    rotated_quotient_basis["teacher_source_custody"]["scalar_objective"][  # type: ignore[index]
        "quotient_basis_sha256"
    ] = hashlib.sha256(b"rotated Helmert basis").hexdigest()
    with pytest.raises(student.StudentContractError, match="basis"):
        student.validate_n600_cache_manifest_structure(rotated_quotient_basis)

    wrong_reduction = copy.deepcopy(manifest)
    wrong_reduction["teacher_source_custody"]["scalar_objective"][  # type: ignore[index]
        "reduction"
    ] = "sum"
    with pytest.raises(student.StudentContractError, match="objective/reduction"):
        student.validate_n600_cache_manifest_structure(wrong_reduction)

    pre_r_normalized_costate = copy.deepcopy(manifest)
    pre_r_normalized_costate["teacher_source_custody"]["scalar_objective"][  # type: ignore[index]
        "costate_surface"
    ] = "gradient_wrt_pre_r_normalized_rgb_0_1"
    with pytest.raises(student.StudentContractError, match="costate surface"):
        student.validate_n600_cache_manifest_structure(pre_r_normalized_costate)

    scorer_substitution = copy.deepcopy(manifest)
    scorer_substitution["teacher_source_custody"]["frozen_segnet"][  # type: ignore[index]
        "architecture_identity"
    ] = "different_scorer"
    with pytest.raises(student.StudentContractError, match="SegNet architecture"):
        student.validate_n600_cache_manifest_structure(scorer_substitution)

    r_substitution = copy.deepcopy(manifest)
    r_substitution["teacher_source_custody"]["r_operator"][  # type: ignore[index]
        "identity"
    ] = "pre_r_proxy"
    with pytest.raises(student.StudentContractError, match="actual R"):
        student.validate_n600_cache_manifest_structure(r_substitution)

    placeholder_scorer_hash = copy.deepcopy(manifest)
    placeholder_scorer_hash["teacher_source_custody"]["frozen_segnet"][  # type: ignore[index]
        "scorer_sha256"
    ] = "a" * 64
    with pytest.raises(student.StudentContractError, match="placeholder"):
        student.validate_n600_cache_manifest_structure(placeholder_scorer_hash)

    sha_drift = copy.deepcopy(manifest)
    sha_drift["rows"][0]["artifacts"]["rendered_frame"]["sha256"] = "not-a-sha"  # type: ignore[index]
    with pytest.raises(student.StudentContractError, match="sha256"):
        student.validate_n600_cache_manifest_structure(sha_drift)

    # Structure-only metadata is never silently promoted to fit authority.
    with pytest.raises(student.StudentContractError, match="byte count drifted"):
        student.validate_n600_cache_manifest(manifest, bundle_root=tmp_path)


def test_economics_distinguishes_student_k_from_optional_inner_k2() -> None:
    k20 = student.surrogate_economics(
        tier="training_gradient",
        student_anchor_cadence=20,
        student_step_ms=1.0,
        exact_teacher_step_ms=3009.069611,
        anchor_update_ms=0.5,
        tier_gate_passed=True,
    )
    assert k20["component_pays"] is True
    assert k20["inclusive_95_kill_feasible"] is False
    assert k20["charged_ms_per_step"] > 0.05 * 3009.069611

    k21 = student.surrogate_economics(
        tier="training_gradient",
        student_anchor_cadence=21,
        student_step_ms=1.0,
        exact_teacher_step_ms=3009.069611,
        anchor_update_ms=0.0,
        tier_gate_passed=True,
    )
    assert k21["inclusive_95_kill_feasible"] is True

    composition = student.cadence_composition(
        student_anchor_cadence=40,
        inner_costate_reuse_cadence=2,
    )
    assert composition["student_anchor_cadence"] == 40
    assert composition["inner_costate_reuse_cadence"] == 2
    assert composition["student_cadence_capped_by_inner_controller"] is False
    assert composition["inherited_speed_claim"] is False
    with pytest.raises(student.StudentContractError, match="None, 1, or 2"):
        student.cadence_composition(
            student_anchor_cadence=40,
            inner_costate_reuse_cadence=3,
        )

    advisory_gate_failed = student.surrogate_economics(
        tier="forward_advisory",
        student_anchor_cadence=8,
        student_step_ms=1.0,
        exact_teacher_step_ms=100.0,
        anchor_update_ms=0.0,
        tier_gate_passed=False,
    )
    assert advisory_gate_failed["component_pays"] is True
    assert advisory_gate_failed["pays"] is False


def test_teacher_timing_receipt_is_byte_and_semantic_custody_bound(tmp_path) -> None:
    fake_cache = _TestOnlyN4Cache(tmp_path, seed=455)
    backend = _TestOnlyFakeMlxBackend()
    raw_path = tmp_path / "teacher_timing_raw.json"
    raw = {
        "schema": "whole_teacher_distilled_student_teacher_timing_raw.v1",
        "measurement_axis": student.MEASUREMENT_AXIS,
        "hardware_fingerprint_sha256": backend.hardware_fingerprint_sha256,
        "teacher_source_custody_sha256": (
            fake_cache.teacher_source_custody.custody_sha256
        ),
        "r_operator_sha256": fake_cache.teacher_source_custody.r_operator_sha256,
        "scorer_sha256": fake_cache.teacher_source_custody.scorer_sha256,
        "scalar_objective_sha256": (
            fake_cache.teacher_source_custody.scalar_objective_sha256
        ),
        "post_r_input_surface_sha256": (
            fake_cache.teacher_source_custody.post_r_input_surface_sha256
        ),
        "warmup_excluded": True,
        "summary_statistic": "median_ms",
        "rows": [
            {
                "sample_index": index,
                "exact_teacher_forward_ms": forward_ms,
                "exact_teacher_forward_input_vjp_ms": forward_vjp_ms,
            }
            for index, (forward_ms, forward_vjp_ms) in enumerate(
                ((9.0, 45.0), (10.0, 50.0), (11.0, 55.0))
            )
        ],
    }
    raw_path.write_text(json.dumps(raw, sort_keys=True))
    raw_custody = {
        "path": raw_path.name,
        "bytes": raw_path.stat().st_size,
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema": "whole_teacher_distilled_student_teacher_timing_receipt.v1",
        "measurement_axis": student.MEASUREMENT_AXIS,
        "hardware_fingerprint_sha256": backend.hardware_fingerprint_sha256,
        "teacher_source_custody_sha256": (
            fake_cache.teacher_source_custody.custody_sha256
        ),
        "r_operator_sha256": fake_cache.teacher_source_custody.r_operator_sha256,
        "scorer_sha256": fake_cache.teacher_source_custody.scorer_sha256,
        "scalar_objective_sha256": (
            fake_cache.teacher_source_custody.scalar_objective_sha256
        ),
        "post_r_input_surface_sha256": (
            fake_cache.teacher_source_custody.post_r_input_surface_sha256
        ),
        "warmup_excluded": True,
        "observation_count": 3,
        "exact_teacher_forward_ms": 10.0,
        "exact_teacher_forward_input_vjp_ms": 50.0,
        "raw_observations": raw_custody,
    }
    receipt_path = tmp_path / "teacher_timing_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    manifest = {
        "charged_timing_inputs": {
            "schema": "whole_teacher_distilled_student_charged_timing_inputs.v2",
            "receipt": {
                "path": receipt_path.name,
                "bytes": receipt_path.stat().st_size,
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            },
        }
    }

    verified = student._teacher_timing_inputs_from_manifest(
        manifest,
        bundle_root=tmp_path,
        teacher_source_custody=fake_cache.teacher_source_custody,
        backend_impl=backend,
    )
    assert verified["status"] == "CONTENT_BOUND_MATCHED_AXIS_RECEIPT_VERIFIED"
    assert verified["teacher_timing_receipt_sha256"] == manifest[
        "charged_timing_inputs"
    ]["receipt"]["sha256"]

    receipt["exact_teacher_forward_ms"] = 12.0
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    manifest["charged_timing_inputs"]["receipt"] = {
        "path": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    with pytest.raises(student.StudentContractError, match="raw median observations"):
        student._teacher_timing_inputs_from_manifest(
            manifest,
            bundle_root=tmp_path,
            teacher_source_custody=fake_cache.teacher_source_custody,
            backend_impl=backend,
        )

    receipt["exact_teacher_forward_ms"] = 10.0
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    manifest["charged_timing_inputs"]["receipt"] = {
        "path": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }

    raw_path.write_text("tampered")
    with pytest.raises(
        student.StudentContractError, match=r"observations (byte count|sha256) drifted"
    ):
        student._teacher_timing_inputs_from_manifest(
            manifest,
            bundle_root=tmp_path,
            teacher_source_custody=fake_cache.teacher_source_custody,
            backend_impl=backend,
        )


def test_fit_driver_surface_fails_closed_before_any_backend_or_teacher_fallback(tmp_path) -> None:
    with pytest.raises(student.StudentContractError, match="BLOCKED-DATA-CUSTODY"):
        student.fit_measure_cached_student(
            tmp_path / "absent_manifest.json",
            tmp_path,
            "tiny",
            455,
            1,
            lambda _stage, _step, _payload: None,
            None,
            tmp_path,
            "mlx",
            "numpy_fp32",
        )
    assert student.FIT_DRIVER_STATUS == "EXECUTABLE_CACHED_ONLY_MLX_AUTOGRAD_V1"

    duplicate_key_manifest = tmp_path / "duplicate_key_manifest.json"
    duplicate_key_manifest.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(student.StudentContractError, match="BLOCKED-DATA-CUSTODY"):
        student.fit_measure_cached_student(
            duplicate_key_manifest,
            tmp_path,
            "tiny",
            455,
            1,
            lambda _stage, _step, _payload: None,
            None,
            tmp_path,
            "mlx",
            "numpy_fp32",
        )


class _TestOnlyN4Cache:
    """Synthetic n4 structural fixture; never measurement or evidence authority."""

    def __init__(self, root, *, seed: int) -> None:
        self.bundle_root = root
        self.manifest_sha256 = hashlib.sha256(b"TEST-ONLY n4 manifest").hexdigest()
        self.teacher_source_custody = SimpleNamespace(
            custody_sha256=hashlib.sha256(b"TEST-ONLY semantic custody").hexdigest(),
            scalar_objective_sha256=hashlib.sha256(
                b"TEST-ONLY scalar objective"
            ).hexdigest(),
            r_operator_sha256=hashlib.sha256(b"TEST-ONLY R operator").hexdigest(),
            scorer_sha256=hashlib.sha256(b"TEST-ONLY scorer").hexdigest(),
            helmert_basis_sha256=student.HELMERT_BASIS_SHA256,
            post_r_input_surface_sha256=hashlib.sha256(
                b"TEST-ONLY post-R input surface"
            ).hexdigest(),
        )
        architecture = student.architecture_for_size("tiny")
        parameters = student.initialize_student_parameters(architecture, seed=seed)
        rng = np.random.default_rng(991)
        self.rows = tuple(
            SimpleNamespace(
                assignment_id=f"TEST-ONLY-pair-{pair_index}",
                pair_index=pair_index,
                checkpoint_name="TEST-ONLY",
                checkpoint_epoch=0,
                split="heldout" if pair_index == 3 else "train",
            )
            for pair_index in range(4)
        )
        self._arrays: dict[int, dict[str, np.ndarray]] = {}
        for pair_index in range(4):
            frame = rng.uniform(0.0, 255.0, size=(1, 3, 4, 5)).astype(np.float32)
            labels = np.zeros((4, 5), dtype=np.int64)
            labels[:, 2:] = 1 + pair_index % 4
            self._arrays[pair_index] = {
                "rendered_frame": frame,
                "teacher_quotient4": student.student_forward_numpy(
                    frame, architecture, parameters
                ),
                "teacher_input_costate": student.student_ce_input_vjp_numpy(
                    frame, architecture, parameters, labels
                ),
                "labels": labels,
            }

    @property
    def train_rows(self):
        return tuple(row for row in self.rows if row.split == "train")

    @property
    def heldout_rows(self):
        return tuple(row for row in self.rows if row.split == "heldout")

    def load_row(self, pair_index: int) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._arrays[pair_index].items()}


class _TestOnlyFakeMlxBackend:
    """NumPy test double for driver control flow; never labeled as measured MLX."""

    name = "mlx"

    def __init__(
        self,
        *,
        advisory_output_multiplier: float = 1.0,
        advisory_repeat_drift_after_call: int | None = None,
    ) -> None:
        self.train_boundaries: list[np.ndarray] = []
        self.warmup_calls = 0
        self.train_calls = 0
        self.measurement_axis = student.MEASUREMENT_AXIS
        self.hardware_descriptor = {"TEST-ONLY": "fake MLX control-flow backend"}
        self.hardware_fingerprint_sha256 = hashlib.sha256(
            b"TEST-ONLY fake MLX hardware"
        ).hexdigest()
        self.advisory_output_multiplier = np.float32(advisory_output_multiplier)
        self.advisory_repeat_drift_after_call = advisory_repeat_drift_after_call
        self.predict_calls = 0

    @staticmethod
    def parameters_from_numpy(parameters):
        return {name: np.asarray(value, dtype=np.float32).copy() for name, value in parameters.items()}

    @staticmethod
    def zeros_like(parameters):
        return {name: np.zeros_like(value) for name, value in parameters.items()}

    @staticmethod
    def tree_to_numpy(tree):
        return {name: np.asarray(value, dtype=np.float32).copy() for name, value in tree.items()}

    def warmup_train(self, _parameters, *, row, boundary_mask, policy) -> None:
        assert set(row) == {
            "rendered_frame",
            "teacher_quotient4",
            "teacher_input_costate",
            "labels",
        }
        assert policy.architecture.size == "tiny"
        assert np.any(boundary_mask) and not np.all(boundary_mask)
        self.warmup_calls += 1

    def train_step(
        self,
        parameters,
        first_moment,
        second_moment,
        *,
        optimizer_step,
        row,
        boundary_mask,
        policy,
    ):
        del row, policy
        self.train_boundaries.append(boundary_mask.copy())
        self.train_calls += 1
        loss = 1.0 / (optimizer_step + 1)
        return (
            self.parameters_from_numpy(parameters),
            self.parameters_from_numpy(first_moment),
            self.parameters_from_numpy(second_moment),
            loss,
            2.0,
        )

    def predict_and_vjp(self, parameters, *, frame, labels, architecture):
        self.predict_calls += 1
        quotient = student.student_forward_numpy(frame, architecture, parameters)
        input_vjp = student.student_ce_input_vjp_numpy(
            frame, architecture, parameters, labels
        )
        if (
            self.advisory_repeat_drift_after_call is not None
            and self.predict_calls > self.advisory_repeat_drift_after_call
        ):
            # This is only a control-flow fault injector.  The repeat must
            # not be promoted simply because the NumPy-authority stream is
            # stable while the separately required MLX advisory stream drifts.
            quotient = quotient + np.float32(0.25)
            input_vjp = input_vjp + np.float32(0.25)
        return (
            quotient * self.advisory_output_multiplier,
            input_vjp * self.advisory_output_multiplier,
            0.25,
            0.75,
        )


def _copy_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "arrays": {name: np.asarray(value).copy() for name, value in payload["arrays"].items()},
        "metadata": copy.deepcopy(payload["metadata"]),
    }


def test_cached_only_fit_driver_preserves_stages_and_resumes_bit_close(
    tmp_path, monkeypatch
) -> None:
    """TEST-ONLY n4 fixture proves control flow, not n600 student fidelity."""

    manifest_path = tmp_path / "TEST_ONLY_manifest.json"
    manifest_path.write_text("{}")
    fake_cache = _TestOnlyN4Cache(tmp_path, seed=455)
    # Deliberately invert advisory MLX outputs.  NumPy-fp32 primary teacher
    # fidelity must remain perfect while the separate parity gate fails.
    baseline_backend = _TestOnlyFakeMlxBackend(advisory_output_multiplier=-1.0)
    monkeypatch.setattr(student, "N600", 4)
    monkeypatch.setattr(student, "TRAIN_COUNT", 3)
    monkeypatch.setattr(student, "HELDOUT_COUNT", 1)
    monkeypatch.setattr(student, "FIT_CHECKPOINT_INTERVAL_STEPS", 1)
    monkeypatch.setattr(student, "validate_n600_cache_manifest", lambda *_args, **_kwargs: fake_cache)
    monkeypatch.setattr(student, "_load_mlx_backend", lambda: baseline_backend)
    checkpoints: list[tuple[str, int, dict[str, Any]]] = []

    def capture(stage: str, step: int, payload: dict[str, Any]) -> None:
        checkpoints.append((stage, step, _copy_checkpoint_payload(payload)))

    baseline = student.fit_measure_cached_student(
        manifest_path,
        tmp_path,
        "tiny",
        455,
        1,
        capture,
        None,
        tmp_path,
        "mlx",
        "numpy_fp32",
    )
    stages = [stage for stage, _step, _payload in checkpoints]
    assert stages == [
        "initialized",
        "fit_progress",
        "fit_progress",
        "fit_epoch_0001",
        "best_fit",
        "heldout_measurement",
        "completion",
    ]
    assert baseline["n_pairs"] == 4
    assert baseline["train_pairs"] == 3
    assert baseline["heldout_pairs"] == 1
    assert baseline["teacher_calls"] == 0
    assert baseline["backend"] == "mlx"
    assert baseline["measurement_axis"] == (
        "[n600 macOS-MLX advisory; NumPy-fp32 reference; no score authority]"
    )
    assert baseline["fit_steps"] == 3
    assert baseline["vjp_fidelity_decisive_full_vector"]["n600"]["pair_count"] == 4
    assert baseline["vjp_fidelity_boundary_diagnostic"]["n600"]["pair_count"] == 4
    assert baseline["authority"]["boundary_input_vjp_is_diagnostic_only"] is True
    assert baseline["authority"]["primary_teacher_fidelity_numerical_authority"] == (
        "numpy_fp32"
    )
    assert baseline["authority"]["mlx_outputs_used_for_primary_teacher_gate"] is False
    assert baseline["forward_fidelity"]["n600"]["worst_relative_l2"] == 0.0
    assert baseline["forward_fidelity"]["n600"]["worst_cosine"] == pytest.approx(1.0)
    assert baseline["vjp_fidelity_decisive_full_vector"]["n600"][
        "worst_relative_l2"
    ] == 0.0
    assert baseline["vjp_fidelity_decisive_full_vector"]["n600"][
        "worst_cosine"
    ] == pytest.approx(1.0)
    assert baseline["framework_parity"]["mlx_vs_numpy_forward_n600"][
        "worst_cosine"
    ] == pytest.approx(-1.0)
    assert baseline["framework_parity"]["mlx_vs_numpy_input_vjp_n600"][
        "worst_cosine"
    ] == pytest.approx(-1.0)
    for pair_row in baseline["per_pair"]:
        assert pair_row["forward_numpy_fp32_authority"]["cosine"] == pytest.approx(1.0)
        assert pair_row["full_input_vjp_numpy_fp32_decisive"]["cosine"] == pytest.approx(1.0)
        assert pair_row["mlx_vs_numpy_forward"]["cosine"] == pytest.approx(-1.0)
        assert pair_row["mlx_vs_numpy_input_vjp"]["cosine"] == pytest.approx(-1.0)
        assert "forward" not in pair_row
        assert "full_input_vjp_decisive" not in pair_row
    assert baseline["deterministic_repeat"]["pair_count"] == 4
    assert baseline["deterministic_repeat"]["numerical_authority"] == "numpy_fp32"
    assert baseline["deterministic_repeat"]["forward_equal"] is True
    assert baseline["deterministic_repeat"]["input_vjp_equal"] is True
    assert baseline["deterministic_repeat"]["combined_equal"] is True
    assert baseline["deterministic_repeat"]["authority_verified"] is True
    assert baseline["deterministic_repeat"]["mlx_advisory"]["forward_equal"] is True
    assert baseline["deterministic_repeat"]["mlx_advisory"]["input_vjp_equal"] is True
    assert baseline["deterministic_repeat"]["mlx_advisory"]["combined_equal"] is True
    assert baseline["deterministic_repeat"]["mlx_advisory"]["advisory_verified"] is True
    assert baseline["deterministic_repeat"]["all_required_streams_equal"] is True
    assert baseline["deterministic_repeat_verified"] is True
    assert baseline["deterministic_repeat"]["charged_student_timing_includes_repeat"] is False
    assert baseline["charged_timings"]["fully_charged_economics_ready"] is False
    assert baseline["source_custody"]["quotient_basis_sha256"] == student.HELMERT_BASIS_SHA256
    assert baseline["source_custody"]["post_r_input_surface_sha256"] == (
        fake_cache.teacher_source_custody.post_r_input_surface_sha256
    )
    assert baseline_backend.train_calls == 3
    assert all(np.any(mask) and not np.all(mask) for mask in baseline_backend.train_boundaries)

    # One warmup plus four primary rows precedes the advisory repeat.  A
    # second-pass advisory drift must leave the NumPy authority proof intact
    # yet fail the combined mandatory repeat gate, rather than being hidden by
    # a stable MLX first pass or treated as a second charged timing pass.
    advisory_drift_backend = _TestOnlyFakeMlxBackend(
        advisory_output_multiplier=-1.0,
        advisory_repeat_drift_after_call=5,
    )
    monkeypatch.setattr(student, "_load_mlx_backend", lambda: advisory_drift_backend)
    advisory_drift = student.fit_measure_cached_student(
        manifest_path,
        tmp_path,
        "tiny",
        455,
        1,
        lambda _stage, _step, _payload: None,
        None,
        tmp_path,
        "mlx",
        "numpy_fp32",
    )
    assert advisory_drift["deterministic_repeat"]["authority_verified"] is True
    assert advisory_drift["deterministic_repeat"]["mlx_advisory"]["advisory_verified"] is False
    assert advisory_drift["deterministic_repeat"]["all_required_streams_equal"] is False
    assert advisory_drift["deterministic_repeat_verified"] is False
    assert advisory_drift["deterministic_repeat"]["mlx_advisory"][
        "charged_student_timing_includes_repeat"
    ] is False

    progress_state = next(
        payload for stage, step, payload in checkpoints if stage == "fit_progress" and step == 1
    )
    resumed_backend = _TestOnlyFakeMlxBackend(advisory_output_multiplier=-1.0)
    monkeypatch.setattr(student, "_load_mlx_backend", lambda: resumed_backend)
    resumed_checkpoints: list[tuple[str, int, dict[str, Any]]] = []

    def capture_resumed(stage: str, step: int, payload: dict[str, Any]) -> None:
        resumed_checkpoints.append((stage, step, _copy_checkpoint_payload(payload)))

    resumed = student.fit_measure_cached_student(
        manifest_path,
        tmp_path,
        "tiny",
        455,
        1,
        capture_resumed,
        progress_state,
        tmp_path,
        "mlx",
        "numpy_fp32",
    )
    assert resumed_backend.train_calls == 2
    assert resumed["best_parameters_sha256"] == baseline["best_parameters_sha256"]
    assert resumed["best_objective"] == baseline["best_objective"]
    assert resumed["fit_loss"] == baseline["fit_loss"]
    assert resumed["forward_fidelity"] == baseline["forward_fidelity"]
    assert resumed["vjp_fidelity_decisive_full_vector"] == baseline[
        "vjp_fidelity_decisive_full_vector"
    ]

    tampered = _copy_checkpoint_payload(progress_state)
    parameter_key = next(name for name in tampered["arrays"] if name.startswith("parameter__"))
    tampered["arrays"][parameter_key].flat[0] += np.float32(1.0)
    with pytest.raises(student.StudentContractError, match="state array hash drifted"):
        student.fit_measure_cached_student(
            manifest_path,
            tmp_path,
            "tiny",
            455,
            1,
            capture_resumed,
            tampered,
            tmp_path,
            "mlx",
            "numpy_fp32",
        )

    completion_state = next(
        payload for stage, _step, payload in checkpoints if stage == "completion"
    )
    terminal_backend = _TestOnlyFakeMlxBackend()
    monkeypatch.setattr(student, "_load_mlx_backend", lambda: terminal_backend)
    terminal = student.fit_measure_cached_student(
        manifest_path,
        tmp_path,
        "tiny",
        455,
        1,
        capture_resumed,
        completion_state,
        tmp_path,
        "mlx",
        "numpy_fp32",
    )
    assert terminal == baseline
    assert terminal_backend.warmup_calls == 0
    assert terminal_backend.train_calls == 0
