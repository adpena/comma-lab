# SPDX-License-Identifier: MIT
"""Fail-closed, argv-inert policy for Task #538 shared-resize coupling.

This module does not activate a trainer lever.  It validates the advisory
measurement receipt and exposes the future completeness-compiler gate without
claiming that v10 consumes either surface today.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "shared_resize_joint_coupling_measurement.v2"
POLICY_SCHEMA = "shared_resize_joint_coupling_policy.v2"
COMPLETENESS_SCHEMA = "inverse_solve_completeness_manifest.v1"
AXIS = "[macOS-CPU advisory]"
N_PAIRS_TOTAL = 600
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

# The authority numbers ten factors; factor 3 has two independently owned
# leaves.  Keep both the ten primary IDs and the eleven ordered leaves so no
# consumer merges frame-0 freedom with ker(A)/MDL or shared-A with camera-A.
INVERSE_SOLVE_FACTOR_IDS: tuple[str, ...] = (
    "1_decode_frames",
    "2_uint8_lattice",
    "3_shared_resize",
    "4_rgb_yuv6_chroma",
    "5_conv_residual",
    "6_rank4_seg_head",
    "7_six_vector_pose_head",
    "8_frame0_seg_freedom",
    "9_kerA_MDL",
    "10_joint_score_waterfill",
)
INVERSE_SOLVE_FACTOR_LEAF_IDS: tuple[str, ...] = (
    "1_decode_frames",
    "2_uint8_lattice",
    "3a_camera_resize_A",
    "3b_shared_A_coupling",
    "4_rgb_yuv6_chroma",
    "5_conv_residual",
    "6_rank4_seg_head",
    "7_six_vector_pose_head",
    "8_frame0_seg_freedom",
    "9_kerA_MDL",
    "10_joint_score_waterfill",
)
_FACTOR_STATES = frozenset({"have", "missing", "folded"})
_EVIDENCE_LABELS = {
    "smooth_gram": "B1_LOCAL_DERIVED",
    "finite_response": "B32_DUPLICATE_LAST_SUBSET_ADVISORY",
    "contest_score": "NOT_MEASURED",
}


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"{key} must be explicitly false")


def deterministic_stride_sample_ids(
    n_total: int, n_sample: int, seed: int
) -> tuple[int, ...]:
    """Integer-only mirror of the measurement tool's cyclic stride sampler."""

    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (n_total, n_sample, seed)):
        raise ValueError("stride sample inputs must be integers")
    if n_total <= 0 or not (1 <= n_sample <= n_total):
        raise ValueError("require 1 <= n_sample <= n_total")
    centers = [((2 * index + 1) * n_total) // (2 * n_sample) for index in range(n_sample)]
    selected = tuple(sorted((center + seed % n_total) % n_total for center in centers))
    if len(set(selected)) != n_sample:
        raise ValueError("deterministic stride sampler produced duplicate pair IDs")
    return selected


def _finite_matrix_2x2(value: Any, *, label: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2x2 sequence")
    rows: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ValueError(f"{label} must be a 2x2 sequence")
        pair = (float(row[0]), float(row[1]))
        if not all(math.isfinite(v) for v in pair):
            raise ValueError(f"{label} contains a non-finite value")
        rows.append(pair)
    return rows[0], rows[1]


def _validate_gram(value: Any, *, label: str) -> None:
    matrix = _finite_matrix_2x2(value, label=label)
    scale = max(1.0, *(abs(v) for row in matrix for v in row))
    if abs(matrix[0][1] - matrix[1][0]) > 1e-8 * scale:
        raise ValueError(f"{label} must be symmetric")
    if matrix[0][0] < 0.0 or matrix[1][1] < 0.0:
        raise ValueError(f"{label} diagonal must be non-negative")
    if matrix[0][0] * matrix[1][1] + 1e-10 * scale < matrix[0][1] ** 2:
        raise ValueError(f"{label} must be positive semidefinite")


def _finite_score_payload(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a score mapping")
    result: dict[str, float] = {}
    for key in ("d_seg", "d_pose"):
        raw = value.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"{label}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label}.{key} must be finite and non-negative")
        if key == "d_seg" and number > 1.0:
            raise ValueError(f"{label}.d_seg must be in [0,1]")
        result[key] = number
    return result


def _cross_effect(delta: float) -> str:
    if delta < 0.0:
        return "MEASURED_HELP"
    if delta > 0.0:
        return "MEASURED_HARM"
    return "MEASURED_NEUTRAL"


def _recompute_actual_response(row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical recomputation of every field derived from exact arm scores."""

    baseline = _finite_score_payload(row.get("baseline"), label="baseline")
    raw_arms = row.get("arm_scores")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("arm_scores must be a mapping")
    arm_names = ("seg_plus", "seg_minus", "pose_plus", "pose_minus", "joint_plus")
    arms = {
        name: _finite_score_payload(raw_arms.get(name), label=f"arm_scores.{name}")
        for name in arm_names
    }
    one_sided = [
        [
            arms["seg_plus"][metric] - baseline[metric],
            arms["pose_plus"][metric] - baseline[metric],
        ]
        for metric in ("d_seg", "d_pose")
    ]
    central = [
        [
            (arms["seg_plus"][metric] - arms["seg_minus"][metric]) / 2.0,
            (arms["pose_plus"][metric] - arms["pose_minus"][metric]) / 2.0,
        ]
        for metric in ("d_seg", "d_pose")
    ]
    joint_delta = {
        metric: arms["joint_plus"][metric] - baseline[metric]
        for metric in ("d_seg", "d_pose")
    }
    baseline_score = 100.0 * baseline["d_seg"] + math.sqrt(10.0 * baseline["d_pose"])
    joint_score = (
        100.0 * arms["joint_plus"]["d_seg"]
        + math.sqrt(10.0 * arms["joint_plus"]["d_pose"])
    )
    seg_informative = one_sided[0][0] < 0.0
    pose_informative = one_sided[1][1] < 0.0
    joint_informative = joint_score < baseline_score
    classification = {
        "seg_direction": {
            "target_metric": "d_seg",
            "target_delta": one_sided[0][0],
            "quality": "MEASURED_HELP" if seg_informative else "UNINFORMATIVE_DIRECTION",
            "cross_d_pose_effect": (
                _cross_effect(one_sided[1][0])
                if seg_informative
                else "UNINFORMATIVE_DIRECTION"
            ),
        },
        "pose_direction": {
            "target_metric": "d_pose",
            "target_delta": one_sided[1][1],
            "quality": "MEASURED_HELP" if pose_informative else "UNINFORMATIVE_DIRECTION",
            "cross_d_seg_effect": (
                _cross_effect(one_sided[0][1])
                if pose_informative
                else "UNINFORMATIVE_DIRECTION"
            ),
        },
        "joint_direction": {
            "target_metric": "100*d_seg+sqrt(10*d_pose)",
            "target_delta": joint_score - baseline_score,
            "quality": "MEASURED_HELP" if joint_informative else "UNINFORMATIVE_DIRECTION",
        },
    }
    return {
        "one_sided_response_2x2": one_sided,
        "central_secant_response_2x2": central,
        "joint_plus_delta": joint_delta,
        "measured_direction_classification": classification,
        "cross_response_ratios": {
            "pose_change_per_abs_seg_change_under_seg_direction": (
                one_sided[1][0] / max(abs(one_sided[0][0]), 1e-30)
            ),
            "seg_change_per_abs_pose_change_under_pose_direction": (
                one_sided[0][1] / max(abs(one_sided[1][1]), 1e-30)
            ),
        },
    }


def _require_close(actual: Any, expected: float, *, label: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ValueError(f"{label} must be numeric")
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError(f"{label} is inconsistent with exact arm-score recomputation")


def _require_matrix_close(actual: Any, expected: list[list[float]], *, label: str) -> None:
    matrix = _finite_matrix_2x2(actual, label=label)
    for row in range(2):
        for column in range(2):
            _require_close(
                matrix[row][column], expected[row][column], label=f"{label}[{row}][{column}]"
            )


@dataclass(frozen=True)
class SharedResizeJointCouplingPolicy:
    """Sealed advisory policy; compile output is intentionally empty."""

    schema: str = POLICY_SCHEMA
    measurement_schema: str = SCHEMA
    n_pairs_total: int = N_PAIRS_TOTAL
    camera_hw: tuple[int, int] = CAMERA_HW
    scorer_hw: tuple[int, int] = SCORER_HW
    shared_A_identity: str = "torch_bilinear_align_corners_false_874x1164_to_384x512"
    live_trainer_argv: tuple[str, ...] = ()
    default_state: str = "off_pending_real_n600_subset_measurement"
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    pointer_moved: bool = False
    paid_dispatch: bool = False
    trainer_activation: bool = False
    live_v10_integration: bool = False

    def validate(self) -> None:
        if self.schema != POLICY_SCHEMA or self.measurement_schema != SCHEMA:
            raise ValueError("coupling policy schema mismatch")
        if self.n_pairs_total != N_PAIRS_TOTAL:
            raise ValueError("coupling policy requires n_pairs_total == 600")
        if self.camera_hw != CAMERA_HW or self.scorer_hw != SCORER_HW:
            raise ValueError("shared-A input/output geometry is sealed")
        if self.live_trainer_argv:
            raise ValueError("Task #538 policy is argv-inert")
        for key in (
            "score_claim",
            "promotion_eligible",
            "pointer_moved",
            "paid_dispatch",
            "trainer_activation",
            "live_v10_integration",
        ):
            if getattr(self, key) is not False:
                raise ValueError(f"{key} escalation is forbidden")
        if self.research_only is not True:
            raise ValueError("Task #538 policy must remain research_only")

    def compile(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def compile_shared_resize_joint_coupling_policy() -> dict[str, Any]:
    """Compile the sealed policy.  It can never emit trainer argv."""

    return SharedResizeJointCouplingPolicy().compile()


def validate_measurement_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate custody and finite coupling fields without granting authority."""

    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be a mapping")
    if receipt.get("schema") != SCHEMA or receipt.get("axis") != AXIS:
        raise ValueError("receipt schema/axis mismatch")
    if int(receipt.get("n_pairs_total", -1)) != N_PAIRS_TOTAL:
        raise ValueError("receipt requires exact n_pairs_total == 600")
    sample = receipt.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError("receipt sample custody is missing")
    n_sample = sample.get("n_of_600")
    pair_ids = sample.get("pair_ids")
    if (
        not isinstance(n_sample, int)
        or isinstance(n_sample, bool)
        or n_sample < 1
        or not isinstance(pair_ids, list)
        or len(pair_ids) != n_sample
    ):
        raise ValueError("sample n_of_600/pair_ids mismatch")
    if len(set(pair_ids)) != n_sample or any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= N_PAIRS_TOTAL
        for i in pair_ids
    ):
        raise ValueError("sample pair_ids must be unique integers in [0,600)")
    evidence_status = receipt.get("evidence_status")
    expected_status = "LIVENESS_ONLY_NOT_A_MEASUREMENT_VERDICT" if n_sample == 1 else "MEASURED_ADVISORY_SUBSET"
    if evidence_status != expected_status:
        raise ValueError("receipt evidence_status does not match sample size")
    if 1 < n_sample < 8:
        raise ValueError("n=2..7 is neither allowed liveness nor a measurement verdict")
    seed = sample.get("seed")
    if (
        sample.get("method") != "deterministic_cyclic_stride"
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("sample method/seed custody is missing")
    if tuple(pair_ids) != deterministic_stride_sample_ids(N_PAIRS_TOTAL, n_sample, seed):
        raise ValueError("sample pair_ids do not match the deterministic stride contract")

    custody = receipt.get("input_custody")
    if not isinstance(custody, Mapping):
        raise ValueError("input custody is missing")
    for key in ("checkpoint_sha256", "gt_cache_sha256", "segnet_sha256", "posenet_sha256"):
        if not isinstance(custody.get(key), str) or _SHA256_RE.fullmatch(custody[key]) is None:
            raise ValueError(f"input_custody.{key} is not a SHA-256")
    for key in ("checkpoint_path", "gt_cache_path", "segnet_path", "posenet_path"):
        if not isinstance(custody.get(key), str) or not custody[key].strip():
            raise ValueError(f"input_custody.{key} is missing")
    checkpoint_payload = custody.get("checkpoint_payload")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("input_custody.checkpoint_payload is missing")
    if checkpoint_payload.get("carrier_absent") is not True:
        raise ValueError("checkpoint carrier absence is not certified")
    if checkpoint_payload.get("base_inr_only") is not True:
        raise ValueError("checkpoint is not certified as base-INR-only")
    if checkpoint_payload.get("detected_carrier_keys") != []:
        raise ValueError("checkpoint contains a carrier payload")
    if (
        not isinstance(checkpoint_payload.get("checkpoint_key_count"), int)
        or checkpoint_payload["checkpoint_key_count"] < 1
        or _SHA256_RE.fullmatch(
            str(checkpoint_payload.get("checkpoint_key_manifest_sha256", ""))
        )
        is None
    ):
        raise ValueError("checkpoint key-manifest custody is missing")
    shared_A = receipt.get("shared_A")
    if not isinstance(shared_A, Mapping):
        raise ValueError("shared_A custody is missing")
    if shared_A.get("seg_pose_operator_identical") is not True:
        raise ValueError("shared A identity was not verified")
    if tuple(shared_A.get("camera_hw", ())) != CAMERA_HW or tuple(shared_A.get("scorer_hw", ())) != SCORER_HW:
        raise ValueError("shared A geometry differs from the sealed geometry")

    execution = receipt.get("execution_custody")
    if not isinstance(execution, Mapping):
        raise ValueError("execution custody is missing")
    if _GIT_OID_RE.fullmatch(str(execution.get("git_head", ""))) is None:
        raise ValueError("execution_custody.git_head is not a git SHA")
    if (
        not isinstance(execution.get("argv"), list)
        or not execution["argv"]
        or any(not isinstance(token, str) or not token for token in execution["argv"])
    ):
        raise ValueError("execution_custody.argv is missing")
    config = execution.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("execution_custody.config is missing")
    if config.get("n_sample") != n_sample or config.get("seed") != seed:
        raise ValueError("execution config does not bind the sampled subset")
    inputs = execution.get("input_bytes")
    if not isinstance(inputs, Mapping):
        raise ValueError("execution_custody.input_bytes is missing")
    required_bytes = (
        "checkpoint",
        "gt_cache",
        "segnet",
        "posenet",
        "modules_py",
        "frame_utils_py",
        "evaluate_py",
    )
    if any(
        not isinstance(inputs.get(key), int)
        or isinstance(inputs[key], bool)
        or inputs[key] <= 0
        for key in required_bytes
    ):
        raise ValueError("execution custody requires positive byte sizes for every authority input")
    sources = execution.get("upstream_source_sha256")
    if not isinstance(sources, Mapping) or any(
        _SHA256_RE.fullmatch(str(sources.get(key, ""))) is None
        for key in ("modules_py", "frame_utils_py", "evaluate_py")
    ):
        raise ValueError("execution custody lacks upstream source SHA-256 values")
    source_paths = execution.get("upstream_source_paths")
    if not isinstance(source_paths, Mapping) or any(
        not isinstance(source_paths.get(key), str) or not source_paths[key].strip()
        for key in ("modules_py", "frame_utils_py", "evaluate_py")
    ):
        raise ValueError("execution custody lacks upstream source paths")

    targets = receipt.get("gt_target_custody")
    if not isinstance(targets, Mapping):
        raise ValueError("rederived GT target custody is missing")
    if targets.get("targets_used") != "rederived_from_gt_frames":
        raise ValueError("measurement must use rederived GT targets")
    if targets.get("scorer_batch_size") != 32:
        raise ValueError("GT targets require exact scorer batch size 32")
    if targets.get("last_batch_padding") != "duplicate-last then discard padded outputs":
        raise ValueError("GT target padding geometry is not sealed")
    comparison = targets.get("cache_vs_rederived")
    if not isinstance(comparison, Mapping):
        raise ValueError("cache-vs-rederived target metrics are missing")
    for key in (
        "seg_mismatched_pixels",
        "seg_total_pixels",
        "pose_mismatched_elements",
        "pose_total_elements",
    ):
        if (
            not isinstance(comparison.get(key), int)
            or isinstance(comparison[key], bool)
            or comparison[key] < 0
        ):
            raise ValueError(f"gt_target_custody.cache_vs_rederived.{key} is invalid")
    for key in ("seg_mismatch_fraction", "pose_max_abs", "pose_mse"):
        value = comparison.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"gt_target_custody.cache_vs_rederived.{key} is invalid")
    if comparison["seg_total_pixels"] != n_sample * SCORER_HW[0] * SCORER_HW[1]:
        raise ValueError("GT Seg target comparison has the wrong sampled pixel count")
    if comparison["pose_total_elements"] != n_sample * 6:
        raise ValueError("GT Pose target comparison has the wrong sampled element count")
    if comparison["seg_mismatched_pixels"] > comparison["seg_total_pixels"]:
        raise ValueError("GT Seg mismatch count exceeds total pixels")
    expected_fraction = comparison["seg_mismatched_pixels"] / comparison["seg_total_pixels"]
    if not math.isclose(
        float(comparison["seg_mismatch_fraction"]), expected_fraction, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("GT Seg mismatch fraction does not match its counts")
    if comparison["pose_mismatched_elements"] > comparison["pose_total_elements"]:
        raise ValueError("GT Pose mismatch count exceeds total elements")

    if receipt.get("evidence_labels") != _EVIDENCE_LABELS:
        raise ValueError("receipt evidence labels are incomplete or authority-escalating")

    smooth = receipt.get("smooth_coupling")
    if not isinstance(smooth, Mapping):
        raise ValueError("smooth coupling data is missing")
    if smooth.get("evidence_label") != _EVIDENCE_LABELS["smooth_gram"]:
        raise ValueError("smooth coupling must remain labeled as a derived surrogate")
    if smooth.get("primary_surface") != "shared_frame1":
        raise ValueError("shared_frame1 must be the primary smooth coupling surface")
    for surface_name in ("shared_frame1", "full_pair_context"):
        surface = smooth.get(surface_name)
        if not isinstance(surface, Mapping):
            raise ValueError(f"smooth coupling surface {surface_name} is missing")
        _validate_gram(surface.get("raw_gram_2x2"), label=f"{surface_name}.raw_gram_2x2")
        _validate_gram(
            surface.get("score_priced_gram_2x2"),
            label=f"{surface_name}.score_priced_gram_2x2",
        )
    actual = receipt.get("actual_response")
    if not isinstance(actual, Mapping) or not isinstance(actual.get("by_support_fraction"), list):
        raise ValueError("actual response data is missing")
    if actual.get("scorer_batch_size") != 32:
        raise ValueError("actual response requires exact scorer batch size 32")
    if actual.get("last_batch_padding") != "duplicate-last then discard padded outputs":
        raise ValueError("actual response padding geometry is not sealed")
    if actual.get("evidence_label") != _EVIDENCE_LABELS["finite_response"]:
        raise ValueError("actual response must remain labeled as measured advisory evidence")
    if actual.get("native_or_full_n600_comparable") is not False:
        raise ValueError("subset finite response must be explicitly non-comparable to native/full-n600")
    if actual.get("perturbation_surface") != "shared_frame1_only":
        raise ValueError("finite response must perturb the primary shared-frame1 surface")
    if not actual["by_support_fraction"]:
        raise ValueError("actual response must contain at least one support fraction")
    for index, row in enumerate(actual["by_support_fraction"]):
        if not isinstance(row, Mapping):
            raise ValueError(f"actual response row {index} is not a mapping")
        fraction = float(row.get("support_fraction", 0.0))
        if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise ValueError(f"actual response row {index} has invalid support fraction")
        recomputed = _recompute_actual_response(row)
        _require_matrix_close(
            row.get("one_sided_response_2x2"),
            recomputed["one_sided_response_2x2"],
            label="one_sided_response_2x2",
        )
        _require_matrix_close(
            row.get("central_secant_response_2x2"),
            recomputed["central_secant_response_2x2"],
            label="central_secant_response_2x2",
        )
        stored_joint = row.get("joint_plus_delta")
        if not isinstance(stored_joint, Mapping):
            raise ValueError(f"actual response row {index} lacks joint_plus_delta")
        for metric, expected in recomputed["joint_plus_delta"].items():
            _require_close(
                stored_joint.get(metric), expected, label=f"joint_plus_delta.{metric}"
            )
        stored_ratios = row.get("cross_response_ratios")
        if not isinstance(stored_ratios, Mapping):
            raise ValueError(f"actual response row {index} lacks cross_response_ratios")
        for ratio, expected in recomputed["cross_response_ratios"].items():
            _require_close(
                stored_ratios.get(ratio), expected, label=f"cross_response_ratios.{ratio}"
            )
        classification = row.get("measured_direction_classification")
        if not isinstance(classification, Mapping):
            raise ValueError(f"actual response row {index} lacks measured direction classification")
        for direction in ("seg_direction", "pose_direction", "joint_direction"):
            detail = classification.get(direction)
            expected_detail = recomputed["measured_direction_classification"][direction]
            if not isinstance(detail, Mapping):
                raise ValueError(f"actual response row {index} lacks {direction} classification")
            for key, expected in expected_detail.items():
                if key == "target_delta":
                    _require_close(detail.get(key), expected, label=f"{direction}.{key}")
                elif detail.get(key) != expected:
                    raise ValueError(
                        f"actual response row {index} {direction}.{key} is inconsistent "
                        "with exact arm-score recomputation"
                    )
        realized = row.get("realized_lsb_counts")
        if not isinstance(realized, Mapping):
            raise ValueError(f"actual response row {index} lacks realized LSB counts")
        for arm in ("seg_plus", "seg_minus", "pose_plus", "pose_minus", "joint_plus"):
            counts = realized.get(arm)
            if not isinstance(counts, Mapping):
                raise ValueError(f"actual response row {index} lacks counts for {arm}")
            requested = counts.get("nonzero_requested")
            changed = counts.get("realized_changed")
            clipped = counts.get("boundary_clipped")
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (requested, changed, clipped)
            ):
                raise ValueError(f"actual response row {index} has invalid counts for {arm}")
            if changed + clipped != requested:
                raise ValueError(f"actual response row {index} counts do not conserve {arm} requests")
        for direction, arm in (
            ("seg_direction", "seg_plus"),
            ("pose_direction", "pose_plus"),
            ("joint_direction", "joint_plus"),
        ):
            if (
                recomputed["measured_direction_classification"][direction]["quality"]
                == "MEASURED_HELP"
                and realized[arm]["realized_changed"] <= 0
            ):
                raise ValueError(
                    f"actual response row {index} claims MEASURED_HELP for {direction} "
                    "without any realized changed pixels"
                )

    for key in (
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
        "paid_dispatch",
        "trainer_activation",
        "sacred_c2_mutated",
    ):
        _require_false(receipt, key)
    if receipt.get("research_only") is not True:
        raise ValueError("receipt must remain research_only")
    verdict_scope = receipt.get("verdict_scope")
    if not isinstance(verdict_scope, str) or not verdict_scope.strip():
        raise ValueError("receipt verdict_scope is missing")
    return dict(receipt)


def compile_inverse_solve_completeness_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile the future v10 completeness gate; exact ordered leaves are mandatory."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("completeness rows must be a sequence")
    observed = tuple(row.get("factor_id") for row in rows)
    if observed != INVERSE_SOLVE_FACTOR_LEAF_IDS:
        raise ValueError(
            "completeness factor leaves/order mismatch; expected "
            f"{INVERSE_SOLVE_FACTOR_LEAF_IDS!r}"
        )
    compiled: list[dict[str, str]] = []
    for row in rows:
        term = row.get("term")
        owner = row.get("owning_task")
        state = row.get("state")
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"{row.get('factor_id')} term is missing")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError(f"{row.get('factor_id')} owning_task is missing")
        if state not in _FACTOR_STATES:
            raise ValueError(f"{row.get('factor_id')} state must be one of {sorted(_FACTOR_STATES)}")
        compiled.append(
            {
                "factor_id": str(row["factor_id"]),
                "term": term.strip(),
                "owning_task": owner.strip(),
                "state": str(state),
            }
        )
    return {
        "schema": COMPLETENESS_SCHEMA,
        "numbered_factor_count": 10,
        "leaf_count": 11,
        "factor_ids": list(INVERSE_SOLVE_FACTOR_IDS),
        "ordered_leaf_rows": compiled,
        # This is deliberately a disposition inventory only.  Presence/state
        # cannot certify derivation, build, compile, consume, resume,
        # interaction, measurement, or adoption evidence.
        "disposition_rows_complete": all(row["state"] in {"have", "folded"} for row in compiled),
        "complete_by_construction": False,
        "compiler_gate_status": "DISPOSITION_ONLY_FUTURE_V10_GATE_NOT_LIVE_INTEGRATED",
        "live_v10_integration": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "AXIS",
    "CAMERA_HW",
    "COMPLETENESS_SCHEMA",
    "INVERSE_SOLVE_FACTOR_IDS",
    "INVERSE_SOLVE_FACTOR_LEAF_IDS",
    "N_PAIRS_TOTAL",
    "POLICY_SCHEMA",
    "SCHEMA",
    "SCORER_HW",
    "SharedResizeJointCouplingPolicy",
    "compile_inverse_solve_completeness_manifest",
    "compile_shared_resize_joint_coupling_policy",
    "deterministic_stride_sample_ids",
    "validate_measurement_receipt",
]
