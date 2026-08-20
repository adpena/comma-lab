# SPDX-License-Identifier: MIT
"""Deterministic NumPy PREDICT-to-PROJECT receiver primitives.

This module is scorer agnostic.  It predicts cells from the single-object seed,
extracts only violated constraints, projects finite linear cell/tube systems,
and composes the existing factor-2 interval, support-fill, and full-kernel
surfaces.  Hard scorer callbacks belong in the measurement CLI only.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.r1b4_section_receiver import (
    RECEIVER_SCHEMA as R1B4_RECEIVER_SCHEMA,
)
from tac.canonical_equations.day_consolidation_laws_20260720 import breakeven_bytes
from tac.canonical_equations.evaluators import (
    LAWREF_BUILTIN_EVALUATORS,
    get_evaluator,
    has_evaluator,
    populate_lawref_evaluators,
    register_evaluator,
)
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    EQUATION_ID as TEMPORAL_JITTER_EQUATION_ID,
)
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    amortization_ratio,
    build_partition_temporal_transport_amortization_v1,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    EQUATION_ID as SEGNET_HEAD_RANK_EQUATION_ID,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    build_segnet_head_rank4_linear_flipdist_v1,
    head_difference_rank,
)
from tac.optimization.joint_seg_pose_rate import solve_interval_frame
from tac.optimization.predict_project_schema import (
    CLASS_COUNT,
    STRATA,
    canonical_json_bytes,
    derive_morse_smale_raster,
    serialize_constraint_seed,
    validate_constraint_seed,
)
from tac.optimization.resize_full_kernel import (
    FULL_RESIZE_KERNEL_SCHEMA,
    FullResizeKernel,
)
from tac.optimization.uint8_lattice_feasibility import (
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
    lawref_to_declaration,
    resolve,
)
from tac.witness_dsl.v10_production_receiver import (
    ARITHMETIC_ID as V10_ARITHMETIC_ID,
)
from tac.witness_dsl.v10_production_receiver import (
    RECEIVER_CONTRACT_ID as V10_RECEIVER_CONTRACT_ID,
)
from tac.witness_dsl.v10_production_receiver import (
    TIE_POLICY_ID as V10_TIE_POLICY_ID,
)

REALIZATION_BREAKEVEN_EQUATION_ID: Final = "realization_breakeven_bytes_v1"
RATE_LAW_LADDER_EQUATION_ID: Final = "rate_law_ladder_v1"
RATE_LAW_LADDER_REUSE_SOURCE: Final = "src/tac/canonical_equations/rate_law_ladder_20260713.py"
PDE_318_REUSE_ID: Final = "#318"
ADVECTED_MOTION_BASE_SCHEMA: Final = "predict_project_advected_motion_base.v1"
COUNTED_PLANAR_XI_SCHEMA: Final = "predict_project_counted_planar_xi.v1"
COUNTED_FULL_SCREW_XI_SCHEMA: Final = "predict_project_counted_full_screw_xi.v1"
CHART_RGB_COEFFICIENT_SCHEMA: Final = "predict_project_chart_rgb_coefficients.v1"
CHART_RGB_COEFFICIENT_MAGIC: Final = b"PCR1"
CHART_RGB_COEFFICIENT_VERSION: Final = 1
_CHART_RGB_PREFIX: Final = struct.Struct("<4sHII")
_CHART_RGB_CRC: Final = struct.Struct("<I")
ADVECTED_MOTION_LAWREFS: Final = (
    "ego_motion_cumulative_se3_bspline_v1",
    "lane_band_ego_factorization_source_reparam_v1",
    "lane_band_source_reparam_measured_resolution_v1",
)


def _evaluate_realization_breakeven(inputs: Mapping[str, Any]) -> float:
    if set(inputs) != {"realized_recovery_s"}:
        raise ValueError("realization breakeven LawRef inputs mismatch")
    return breakeven_bytes(float(inputs["realized_recovery_s"]))


def _evaluate_temporal_jitter_ratio(inputs: Mapping[str, Any]) -> float:
    if set(inputs) != {"naive_bytes_total", "trajectory_bytes_total"}:
        raise ValueError("temporal jitter LawRef inputs mismatch")
    return amortization_ratio(
        float(inputs["naive_bytes_total"]),
        float(inputs["trajectory_bytes_total"]),
    )


def _evaluate_segnet_centered_rank(inputs: Mapping[str, Any]) -> int:
    keys = sorted(inputs)
    expected = [f"singval_{index}" for index in range(len(inputs))]
    if keys != expected:
        raise ValueError("SegNet centered-rank LawRef inputs mismatch")
    return head_difference_rank([float(inputs[key]) for key in expected])


def _register_lawref_adapter(equation_id: str, evaluator: Callable[[Mapping[str, Any]], Any]) -> None:
    """Install one process-local adapter, or defer to the canonical builtin.

    `populate_lawref_evaluators()` installs the canonical registry first, so once
    an id graduates into `LAWREF_BUILTIN_EVALUATORS` this module's local adapter
    is a redundant duplicate rather than a conflict.  Refusing that case made the
    whole module unimportable: `realization_breakeven_bytes_v1` graduated in
    81337cd93c and every `import tac.optimization.predict_project_receiver`
    afterwards raised at module scope (line 287 resolves the laws at import),
    taking the predictor chain and its test modules down with it.  Both
    implementations compute `realized_recovery_s / (25.0 / 37_545_489.0)`; the
    builtin adds a finiteness guard, so it is the strictly stronger authority.
    A genuine conflict — two non-builtin adapters for one id — still refuses.
    """

    populate_lawref_evaluators()
    if has_evaluator(equation_id):
        existing = get_evaluator(equation_id)
        if existing is evaluator or existing is LAWREF_BUILTIN_EVALUATORS.get(equation_id):
            return
        raise RuntimeError(f"conflicting in-process LawRef evaluator for {equation_id}")
    register_evaluator(equation_id, evaluator)


def _timestamp_free_resolution(ref: LawRef, resolved: Any) -> dict[str, Any]:
    return {
        "equation_id": resolved.equation_id,
        "ladder_class": resolved.ladder_class,
        "lawref_declaration": lawref_to_declaration(ref),
        "lawref_resolved_value": resolved.value,
        "inputs": [record.to_dict() for record in resolved.resolved_inputs],
        "fallback_used": resolved.fallback_used,
        "warnings": list(resolved.warnings),
    }


def _resolve_predict_project_canonical_laws() -> dict[str, Any]:
    """Resolve all numeric canonical laws consumed by Task #597, without wall-clock fields."""

    adapters = {
        REALIZATION_BREAKEVEN_EQUATION_ID: _evaluate_realization_breakeven,
        TEMPORAL_JITTER_EQUATION_ID: _evaluate_temporal_jitter_ratio,
        SEGNET_HEAD_RANK_EQUATION_ID: _evaluate_segnet_centered_rank,
    }
    for equation_id, evaluator in adapters.items():
        _register_lawref_adapter(equation_id, evaluator)

    unit_score_recovery = InputRef.literal(
        1.0,
        provenance=(
            "unit score recovery chosen by Task #597 solely to derive the score-per-byte "
            "exchange rate as reciprocal breakeven bytes"
        ),
    )
    breakeven_ref = LawRef(
        equation_id=REALIZATION_BREAKEVEN_EQUATION_ID,
        inputs={"realized_recovery_s": unit_score_recovery},
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    breakeven_resolution = resolve(breakeven_ref)
    breakeven_value = float(breakeven_resolution.value)
    if not math.isfinite(breakeven_value) or breakeven_value <= 0:
        raise RuntimeError("canonical realization breakeven must resolve to finite positive bytes")
    lambda_star = 1.0 / breakeven_value

    temporal_equation = build_partition_temporal_transport_amortization_v1()
    if temporal_equation.equation_id != TEMPORAL_JITTER_EQUATION_ID:
        raise RuntimeError("temporal jitter builder returned an unexpected canonical equation ID")
    temporal_anchor = temporal_equation.empirical_anchors[0]
    temporal_rate = temporal_anchor.empirical_output["rate_zlib9_proxy"]
    temporal_ref = LawRef(
        equation_id=TEMPORAL_JITTER_EQUATION_ID,
        inputs={
            "naive_bytes_total": InputRef.literal(
                temporal_rate["naive_bytes_total_600_frames"],
                provenance=(
                    f"{temporal_anchor.anchor_id} canonical builder empirical output "
                    "rate_zlib9_proxy/naive_bytes_total_600_frames"
                ),
            ),
            "trajectory_bytes_total": InputRef.literal(
                temporal_rate["trajectory_bytes_total"]["screw"],
                provenance=(
                    f"{temporal_anchor.anchor_id} canonical builder empirical output "
                    "rate_zlib9_proxy/trajectory_bytes_total/screw"
                ),
            ),
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )
    temporal_resolution = resolve(temporal_ref)
    temporal_ratio = float(temporal_resolution.value)
    if not math.isfinite(temporal_ratio) or temporal_ratio <= 0:
        raise RuntimeError("canonical temporal jitter ratio must resolve finite and positive")

    rank_equation = build_segnet_head_rank4_linear_flipdist_v1()
    if rank_equation.equation_id != SEGNET_HEAD_RANK_EQUATION_ID:
        raise RuntimeError("SegNet rank builder returned an unexpected canonical equation ID")
    rank_anchor = rank_equation.empirical_anchors[0]
    singvals = rank_anchor.empirical_output["singvals"]
    rank_ref = LawRef(
        equation_id=SEGNET_HEAD_RANK_EQUATION_ID,
        inputs={
            f"singval_{index}": InputRef.literal(
                value,
                provenance=(f"{rank_anchor.anchor_id} canonical builder empirical output singvals/{index}"),
            )
            for index, value in enumerate(singvals)
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )
    rank_resolution = resolve(rank_ref)
    centered_rank = int(rank_resolution.value)
    if centered_rank <= 0 or centered_rank != rank_anchor.empirical_output["centered_head_rank"]:
        raise RuntimeError("canonical SegNet centered-rank resolution disagrees with its anchor")

    breakeven_custody = _timestamp_free_resolution(breakeven_ref, breakeven_resolution)
    breakeven_custody.update(
        {
            "canonical_function": ("tac.canonical_equations.day_consolidation_laws_20260720:breakeven_bytes"),
            "derived_operation": "reciprocal_of_breakeven_bytes_for_unit_score_recovery",
            "resolved_value": lambda_star,
            "units": "score_units_per_byte",
        }
    )
    temporal_custody = _timestamp_free_resolution(temporal_ref, temporal_resolution)
    temporal_custody.update(
        {
            "canonical_builder": (
                "tac.canonical_equations.partition_temporal_transport_amortization_20260715:"
                "build_partition_temporal_transport_amortization_v1"
            ),
            "canonical_function": (
                "tac.canonical_equations.partition_temporal_transport_amortization_20260715:amortization_ratio"
            ),
            "anchor_id": temporal_anchor.anchor_id,
            "resolved_value": temporal_ratio,
            "units": "dimensionless",
        }
    )
    rank_custody = _timestamp_free_resolution(rank_ref, rank_resolution)
    rank_custody.update(
        {
            "canonical_builder": (
                "tac.canonical_equations.segnet_head_rank4_flipdist_20260715:build_segnet_head_rank4_linear_flipdist_v1"
            ),
            "canonical_function": ("tac.canonical_equations.segnet_head_rank4_flipdist_20260715:head_difference_rank"),
            "anchor_id": rank_anchor.anchor_id,
            "resolved_value": centered_rank,
            "units": "centered_head_dimension",
        }
    )
    return {
        "schema": "predict_project_canonical_law_resolution_custody.v0",
        "numeric_laws": {
            "global_waterfill_lambda": breakeven_custody,
            "temporal_jitter_amortization_ratio": temporal_custody,
            "segnet_centered_head_rank": rank_custody,
        },
        "availability_only_no_numeric_consumption": {
            RATE_LAW_LADDER_EQUATION_ID: {
                "status": "REUSED_IDENTITY_ONLY_NO_NUMERIC_VALUE_CONSUMED",
                "source_path": RATE_LAW_LADDER_REUSE_SOURCE,
            },
            PDE_318_REUSE_ID: {
                "status": "REUSED_IDENTITY_ONLY_NO_NUMERIC_VALUE_CONSUMED",
                "source_path": "src/tac/preflight.py",
                "symbol": "check_master_gradient_raw_byte_authority_not_landed",
            },
        },
        "persistent_registry_mutated": False,
        "contains_timestamp": False,
    }


CANONICAL_LAW_RESOLUTION_CUSTODY: Final = _resolve_predict_project_canonical_laws()
CANONICAL_LAW_RESOLUTION_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(CANONICAL_LAW_RESOLUTION_CUSTODY)
).hexdigest()
GLOBAL_WATERFILL_LAMBDA_STAR: Final = CANONICAL_LAW_RESOLUTION_CUSTODY["numeric_laws"]["global_waterfill_lambda"][
    "resolved_value"
]
TEMPORAL_JITTER_AMORTIZATION_RATIO: Final = CANONICAL_LAW_RESOLUTION_CUSTODY["numeric_laws"][
    "temporal_jitter_amortization_ratio"
]["resolved_value"]
SEGNET_CENTERED_HEAD_RANK: Final = CANONICAL_LAW_RESOLUTION_CUSTODY["numeric_laws"]["segnet_centered_head_rank"][
    "resolved_value"
]

TIE_POLICY_ID: Final = "native-cpu-torch-f32-first-max-class-index.v1"
PROJECTION_SCHEMA: Final = "predict_project_linear_projection.v0"
HARD_ORACLE_CUSTODY_SCHEMA: Final = "predict_project_hard_oracle_custody.v0"
LOCAL_HARD_ORACLE_AXIS: Final = "[macOS-CPU advisory]"
GLOBAL_WATERFILL_SCHEMA: Final = "predict_project_global_joint_waterfill_evidence.v0"
GLOBAL_WATERFILL_STREAMS: Final = (
    "chart",
    "sites",
    "jitter",
    "events",
    "pose_tightening",
    "response",
    "eat_flip",
)
ACTION_LEVEL_RUNGS: Final = (
    "L1_geometry_chart",
    "L2_channel",
    "L3_hyperplane_feature",
    "L4_regional_plane",
    "L5_pixel_write",
)
ACTION_LEVEL_LADDER_SCHEMA: Final = "predict_project_action_level_ladder.v0"
FLIP_ATTRIBUTION_SCHEMA: Final = "predict_project_flip_attribution.v0"
LADDER_EDIT_REQUEST_SCHEMA: Final = "predict_project_ladder_edit_request.v0"
LADDER_EDIT_RESPONSE_SCHEMA: Final = "predict_project_ladder_edit_response.v0"
ATTRIBUTION_EDIT_TELEMETRY_SCHEMA: Final = "predict_project_attribution_edit_telemetry.v0"
LEARNED_TAIL_RACE_SCHEMA: Final = "predict_project_learned_tail_three_way_race.v0"
ATTRIBUTION_REUSE_BINDINGS: Final = {
    "exact_attribution": {"task": "#350", "source_path": "tools/witness_exact_ab.py"},
    "binding_proof": {
        "task": "#404",
        "source_path": "src/tac/witness_control/telemetry_binding.py",
    },
    "artifact_contract": {"task": "#420", "source_path": "src/tac/witness_run_artifacts.py"},
}
S3_TRAINER_REUSE: Final = {
    "trainer_id": "S3-integer-plane-banded-trainer",
    "source_path": "src/tac/boundary_math/integer_plane_banded_trainer.py",
    "launch_or_training_performed": False,
}
M1_RECEIPT_COMMIT: Final = "9bd01e2232f6898c2564ab8bb7254609c1ebf645"
M1_RECEIPT_PATH: Final = ".omx/research/rep_mine_solved_binary_20260721T045500Z.json"
M1_RECEIPT_SHA256: Final = "265302908fd7c4789891ab0d3b0f8aacaf9f178ea8e40f8737ed5f4fcd55b368"
M1_FLIP_COUNT: Final = 17_926
M1_SCORE_DENOMINATOR: Final = 600 * 512 * 384
M1_ANCHORS: Final = {
    "kernel_energy_percent": 45.1668,
    "kernel_energy_is_byte_savings": False,
    "gauge_energy_percent": 31.1071,
    "old_52_percent_gauge_rejected": True,
    "gauge_surface": "teacher_logits_distinct_from_camera_array_bytes",
    "gauge_additive_with_kernel_or_context": False,
    "centered_rank": SEGNET_CENTERED_HEAD_RANK,
    "quotient_cell_constant_explained_percent": 83.1564,
    "digital_cell_count": 21_304,
    "digital_cells_are_classical_ms_certificate": False,
    "context_bits_per_cell": 0.0150856545,
    "context_ideal_bytes": 222_447.0271,
    "context_estimate_optimistic": True,
    "context_excludes": ["Pose", "model", "header"],
    "context_is_lower_bound": False,
    "target_bytes": 216_222,
    "flip_count": M1_FLIP_COUNT,
    "r2b_isolated_fraction_of_lambda_percent": 6.806,
    "r2b_positive_interaction_measured": False,
    "r2b_admission": "EATEN_ABSENT_POSITIVE_INTERACTION",
}
PER_FLIP_PRICE_STRATA: Final = ("road_lane", "other_edge", "non_edge", "tight_margin")
M1_STRATUM_PRICE_ANCHORS_BYTES_PER_EVENT: Final = {
    "road_lane": 2.48,
    "other_edge": 2.20,
    "non_edge": 4.42,
    "tight_margin": 3.36,
}
GLOBAL_EATEN_FLIP_STRATA: Final = (
    "cell_interior",
    "boundary_codim1",
    "movable_track",
    "critical_event",
)
BOUNDARY_INVERSE_RECEIPT_COMMIT: Final = "e2f679755fea09e4c55a12592db1bc615373c6a8"
BOUNDARY_INVERSE_RECEIPT_PATH: Final = ".omx/research/boundary_inverse_custody_20260721T052100Z.json"
BOUNDARY_INVERSE_RECEIPT_SHA256: Final = "2c7c091c61d1676c80b5db1772a29d3b2f73934398966c8566b6175abc4021e3"
BOUNDARY_INVERSE_ACTION_POLICY: Final = {
    "receipt_commit": BOUNDARY_INVERSE_RECEIPT_COMMIT,
    "receipt_path": BOUNDARY_INVERSE_RECEIPT_PATH,
    "receipt_sha256": BOUNDARY_INVERSE_RECEIPT_SHA256,
    "axis": "[macOS-CPU advisory]",
    "evidence_scope": "mask_fidelity_only_no_through_r_score",
    "lossless_spatial_phase_atom_budget": False,
    "spatial_phase_atom_status": "FORMULATION_SCOPED_NEGATIVE_EIGHT_BIN_ONLY",
    "temporal_phase_status": "LIVE_R1_R2_CAUSAL_PATH",
    "l4_candidate": {
        "variant": "generic_2d_k4",
        "finite_sidecar_bytes": 586,
        "atom_count": 4,
        "mask_f1_delta": 0.0046822634384159,
        "remaining_false_negative_mask_pixels": 313_271,
        "through_r_priced": False,
        "admitted": False,
    },
}


class PredictProjectReceiverError(ValueError):
    """Fail-closed predictor, projection, lattice, or custody failure."""


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """A closed half-space ``normal @ x <= upper`` with stable ID."""

    constraint_id: str
    normal: np.ndarray
    upper: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if (
            not isinstance(self.constraint_id, str)
            or not self.constraint_id
            or normal.ndim != 1
            or normal.size == 0
            or not np.isfinite(normal).all()
            or np.linalg.norm(normal) == 0
            or isinstance(self.upper, bool)
            or not math.isfinite(float(self.upper))
        ):
            raise PredictProjectReceiverError("invalid finite half-space constraint")
        normal = np.array(normal, copy=True)
        normal.setflags(write=False)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "upper", float(self.upper))


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    point: np.ndarray
    iterations: int
    converged: bool
    max_violation: float
    algorithm: str
    cycle_detected: bool


@dataclass(frozen=True, slots=True)
class DoubleDecodeResult:
    first_sha256: str
    second_sha256: str
    byte_identical: bool


def _require_sha256(value: Any, label: str) -> str:
    if (
        not (
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(character in "0123456789abcdef" for character in value)
        )
        or value == "0" * 64
    ):
        raise PredictProjectReceiverError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or value.casefold() in {"unknown", "none", "null", "n/a", "na"}
    ):
        raise PredictProjectReceiverError(f"{label} must be a nonempty canonical string")
    return value


def validate_hard_oracle_custody(value: Any) -> dict[str, Any]:
    """Validate exact local hard-oracle custody without contest promotion."""

    expected = {"schema", "seed", "batch_size", "measurement_axis", "scorer", "inputs", "adapter"}
    if not isinstance(value, dict) or set(value) != expected:
        raise PredictProjectReceiverError("hard-oracle custody fields mismatch")
    if value["schema"] != HARD_ORACLE_CUSTODY_SCHEMA:
        raise PredictProjectReceiverError("hard-oracle custody schema mismatch")
    if value["seed"] != 1234 or isinstance(value["seed"], bool):
        raise PredictProjectReceiverError("hard-oracle custody seed must be exact 1234")
    if value["batch_size"] != 16 or isinstance(value["batch_size"], bool):
        raise PredictProjectReceiverError("hard-oracle custody batch_size must be exact 16")
    if value["measurement_axis"] != LOCAL_HARD_ORACLE_AXIS:
        raise PredictProjectReceiverError(
            "Task #597 local custody admits macOS-CPU advisory only; contest authority requires governed attestation"
        )
    scorer = value["scorer"]
    if not isinstance(scorer, dict) or set(scorer) != {
        "implementation_id",
        "version",
        "source_sha256",
        "segnet_weights_sha256",
        "posenet_weights_sha256",
    }:
        raise PredictProjectReceiverError("scorer custody fields mismatch")
    _require_identity(scorer["implementation_id"], "scorer implementation_id")
    _require_identity(scorer["version"], "scorer version")
    for key in ("source_sha256", "segnet_weights_sha256", "posenet_weights_sha256"):
        _require_sha256(scorer[key], f"scorer {key}")
    inputs = value["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "source_sha256",
        "cache_sha256",
        "evaluated_input_sha256",
    }:
        raise PredictProjectReceiverError("evaluated-input custody fields mismatch")
    for key in ("source_sha256", "cache_sha256", "evaluated_input_sha256"):
        _require_sha256(inputs[key], f"inputs {key}")
    adapter = value["adapter"]
    if not isinstance(adapter, dict) or set(adapter) != {"identity", "source_sha256"}:
        raise PredictProjectReceiverError("adapter custody fields mismatch")
    _require_identity(adapter["identity"], "adapter identity")
    _require_sha256(adapter["source_sha256"], "adapter source_sha256")
    return value.copy()


def hard_oracle_custody_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(validate_hard_oracle_custody(value))).hexdigest()


def _float_vector(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b" or raw.ndim != 1 or raw.size == 0:
        raise PredictProjectReceiverError(f"{label} must be a nonempty real vector")
    out = raw.astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise PredictProjectReceiverError(f"{label} must be finite")
    return out


def project_box(point: Any, lower: Any, upper: Any) -> np.ndarray:
    x = _float_vector(point, "point")
    lo = _float_vector(lower, "lower")
    hi = _float_vector(upper, "upper")
    if lo.shape != x.shape or hi.shape != x.shape or np.any(lo > hi):
        raise PredictProjectReceiverError("box bounds have incompatible shape or order")
    return np.minimum(np.maximum(x, lo), hi)


def project_halfspace(point: Any, constraint: LinearConstraint) -> np.ndarray:
    x = _float_vector(point, "point")
    if constraint.normal.shape != x.shape:
        raise PredictProjectReceiverError("half-space dimension mismatch")
    excess = float(np.dot(constraint.normal, x) - constraint.upper)
    if excess <= 0:
        return x.copy()
    return x - (excess / float(np.dot(constraint.normal, constraint.normal))) * constraint.normal


def max_linear_violation(
    point: Any,
    constraints: Sequence[LinearConstraint],
    *,
    lower: Any | None = None,
    upper: Any | None = None,
) -> float:
    x = _float_vector(point, "point")
    violations = [max(0.0, float(np.dot(row.normal, x) - row.upper)) for row in constraints]
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise PredictProjectReceiverError("both box bounds are required")
        lo, hi = _float_vector(lower, "lower"), _float_vector(upper, "upper")
        if lo.shape != x.shape or hi.shape != x.shape:
            raise PredictProjectReceiverError("box dimension mismatch")
        violations.extend((float(np.max(np.maximum(lo - x, 0))), float(np.max(np.maximum(x - hi, 0)))))
    return max(violations, default=0.0)


def project_linear_intersection(
    point: Any,
    constraints: Sequence[LinearConstraint],
    *,
    lower: Any | None = None,
    upper: Any | None = None,
    algorithm: Literal["dykstra", "pocs"] = "dykstra",
    tolerance: float = 1e-9,
    iteration_cap: int = 10_000,
) -> ProjectionResult:
    """Project onto a finite box/half-space intersection in stable order."""

    x = _float_vector(point, "point").copy()
    if not math.isfinite(tolerance) or tolerance < 0 or iteration_cap < 1:
        raise PredictProjectReceiverError("projection tolerance/cap is invalid")
    ordered = tuple(sorted(constraints, key=lambda row: row.constraint_id))
    if len({row.constraint_id for row in ordered}) != len(ordered):
        raise PredictProjectReceiverError("constraint IDs must be unique")
    if any(row.normal.shape != x.shape for row in ordered):
        raise PredictProjectReceiverError("constraint dimension mismatch")
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise PredictProjectReceiverError("both box bounds are required")
        lo, hi = _float_vector(lower, "lower"), _float_vector(upper, "upper")
        if lo.shape != x.shape or hi.shape != x.shape or np.any(lo > hi):
            raise PredictProjectReceiverError("invalid projection box")

    projectors: list[Callable[[np.ndarray], np.ndarray]] = []
    if lo is not None and hi is not None:
        projectors.append(lambda value: project_box(value, lo, hi))
    projectors.extend(lambda value, row=row: project_halfspace(value, row) for row in ordered)
    if not projectors:
        return ProjectionResult(x, 0, True, 0.0, algorithm, False)
    corrections = [np.zeros_like(x) for _ in projectors]
    seen: dict[bytes, int] = {}
    for iteration in range(1, iteration_cap + 1):
        before_cycle = x.copy()
        for index, projector in enumerate(projectors):
            if algorithm == "dykstra":
                proposal = x + corrections[index]
                projected = projector(proposal)
                corrections[index] = proposal - projected
            elif algorithm == "pocs":
                projected = projector(x)
            else:
                raise PredictProjectReceiverError(f"unsupported projection algorithm {algorithm!r}")
            x = projected
        violation = (
            max_linear_violation(x, ordered, lower=lo, upper=hi) if lo is not None else max_linear_violation(x, ordered)
        )
        step = float(np.max(np.abs(x - before_cycle)))
        if violation <= tolerance and step <= tolerance:
            return ProjectionResult(x, iteration, True, violation, algorithm, False)
        state = np.ascontiguousarray(x, dtype="<f8").tobytes()
        digest = hashlib.sha256(state).digest()
        if digest in seen and step > tolerance:
            raise PredictProjectReceiverError(
                f"projection cycle detected between iterations {seen[digest]} and {iteration}"
            )
        seen[digest] = iteration
    violation = (
        max_linear_violation(x, ordered, lower=lo, upper=hi) if lo is not None else max_linear_violation(x, ordered)
    )
    raise PredictProjectReceiverError(
        f"projection did not converge in {iteration_cap} iterations; max_violation={violation:.17g}"
    )


def quantize_uint8_feasible(
    point: Any,
    constraints: Sequence[LinearConstraint],
    *,
    lower: Any | None = None,
    upper: Any | None = None,
    tolerance: float = 1e-12,
    max_candidates: int = 1_000_000,
) -> np.ndarray:
    """Return the nearest feasible uint8 lattice point with lexicographic ties.

    Candidate rank-tuples are traversed by exact squared distance.  The first
    feasible tuple is therefore globally nearest within the bounded uint8 box;
    the value tuple in the heap key makes equal-distance ties lexicographic.
    """

    x = _float_vector(point, "point")
    if x.size > 12:
        raise PredictProjectReceiverError("exact nearest-lattice search is capped at 12 dimensions")
    if max_candidates < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise PredictProjectReceiverError("invalid lattice search budget/tolerance")
    lo = np.zeros(x.size, dtype=np.float64) if lower is None else _float_vector(lower, "lower")
    hi = np.full(x.size, 255.0, dtype=np.float64) if upper is None else _float_vector(upper, "upper")
    if lo.shape != x.shape or hi.shape != x.shape or np.any(lo > hi):
        raise PredictProjectReceiverError("invalid lattice bounds")
    integer_lows = np.maximum(0, np.ceil(lo - tolerance).astype(np.int64))
    integer_highs = np.minimum(255, np.floor(hi + tolerance).astype(np.int64))
    if np.any(integer_lows > integer_highs):
        raise PredictProjectReceiverError("box contains no uint8 lattice point")
    orders: list[tuple[int, ...]] = []
    for index in range(x.size):
        values = range(int(integer_lows[index]), int(integer_highs[index]) + 1)
        orders.append(tuple(sorted(values, key=lambda value: ((value - x[index]) ** 2, value))))

    def key(ranks: tuple[int, ...]) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
        values = tuple(orders[index][rank] for index, rank in enumerate(ranks))
        distance = math.fsum((value - x[index]) ** 2 for index, value in enumerate(values))
        return distance, values, ranks

    start = (0,) * x.size
    heap = [key(start)]
    visited = {start}
    examined = 0
    while heap and examined < max_candidates:
        _distance, values, ranks = heapq.heappop(heap)
        examined += 1
        candidate = np.asarray(values, dtype=np.float64)
        if max_linear_violation(candidate, constraints, lower=lo, upper=hi) <= tolerance:
            return candidate.astype(np.uint8)
        for axis in range(x.size):
            if ranks[axis] + 1 >= len(orders[axis]):
                continue
            neighbor = list(ranks)
            neighbor[axis] += 1
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                heapq.heappush(heap, key(neighbor_tuple))
    raise PredictProjectReceiverError(f"no feasible uint8 lattice point found within {max_candidates} exact candidates")


def _catmull_rom(values: Sequence[int], times: Sequence[int], time: int) -> float:
    if time <= times[0]:
        return float(values[0])
    if time >= times[-1]:
        return float(values[-1])
    segment = next(index for index in range(len(times) - 1) if times[index] <= time <= times[index + 1])
    p0 = float(values[max(0, segment - 1)])
    p1 = float(values[segment])
    p2 = float(values[segment + 1])
    p3 = float(values[min(len(values) - 1, segment + 2)])
    u = (time - times[segment]) / (times[segment + 1] - times[segment])
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u
    )


def trajectory_at(seed: Mapping[str, Any], time: int) -> tuple[float, float, float]:
    validated = validate_constraint_seed(seed)
    if isinstance(time, bool) or not isinstance(time, int) or not 0 <= time < 600:
        raise PredictProjectReceiverError("trajectory time is outside [0,600)")
    controls = validated["trajectory"]["controls"]
    times = [row["time"] for row in controls]
    values = [[row[key] for row in controls] for key in ("tx_q", "ty_q", "yaw_q")]
    residual = next((row for row in validated["trajectory"]["ar_residuals"] if row["time"] == time), None)
    deltas = (0, 0, 0) if residual is None else (residual["dtx_q"], residual["dty_q"], residual["dyaw_q"])
    quanta = validated["units"]
    translation = quanta["trajectory_translation"]["numerator"] / quanta["trajectory_translation"]["denominator"]
    rotation = quanta["trajectory_rotation"]["numerator"] / quanta["trajectory_rotation"]["denominator"]
    tx = (_catmull_rom(values[0], times, time) + deltas[0]) * translation
    ty = (_catmull_rom(values[1], times, time) + deltas[1]) * translation
    yaw = (_catmull_rom(values[2], times, time) + deltas[2]) * rotation
    return tx, ty, yaw


def _trajectory_at_validated(seed: Mapping[str, Any], time: int) -> tuple[float, float, float]:
    """Evaluate the canonical PPCS planar trajectory after one outer validation."""

    if isinstance(time, bool) or not isinstance(time, int) or not 0 <= time < 600:
        raise PredictProjectReceiverError("trajectory time is outside [0,600)")
    controls = seed["trajectory"]["controls"]
    times = [row["time"] for row in controls]
    values = [[row[key] for row in controls] for key in ("tx_q", "ty_q", "yaw_q")]
    residual = next((row for row in seed["trajectory"]["ar_residuals"] if row["time"] == time), None)
    deltas = (0, 0, 0) if residual is None else (residual["dtx_q"], residual["dty_q"], residual["dyaw_q"])
    quanta = seed["units"]
    translation = quanta["trajectory_translation"]["numerator"] / quanta["trajectory_translation"]["denominator"]
    rotation = quanta["trajectory_rotation"]["numerator"] / quanta["trajectory_rotation"]["denominator"]
    return (
        (_catmull_rom(values[0], times, time) + deltas[0]) * translation,
        (_catmull_rom(values[1], times, time) + deltas[1]) * translation,
        (_catmull_rom(values[2], times, time) + deltas[2]) * rotation,
    )


def counted_planar_xi_series(
    seed: Mapping[str, Any], *, pair_start: int = 0, pair_end: int = 600
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the already-counted planar trajectory as translation-first twists.

    PPCS names the three stored values ``(tx, ty, yaw)`` because its chart
    interpreter works in image coordinates.  The producer is the canonical
    :class:`XiEgoTrajectory` planar seam, where those values are ``(dy, ds,
    dpsi)``.  Its fixed SE(3) embedding is therefore
    ``xi=(dy,0,ds,0,dpsi,0)``.  No target pose, scorer, or additional payload is
    consulted here.
    """

    validated = validate_constraint_seed(seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (pair_start, pair_end)):
        raise PredictProjectReceiverError("xi pair range must use exact integers")
    if not 0 <= pair_start < pair_end <= 600:
        raise PredictProjectReceiverError("xi pair range is outside [0,600]")
    planar = np.asarray(
        [_trajectory_at_validated(validated, pair) for pair in range(pair_start, pair_end)],
        dtype=np.float64,
    )
    xi = np.zeros((pair_end - pair_start, 6), dtype=np.float64)
    xi[:, 0] = planar[:, 0]
    xi[:, 2] = planar[:, 1]
    xi[:, 4] = planar[:, 2]
    trajectory_bytes = len(canonical_json_bytes(validated["trajectory"]))
    custody = {
        "schema": COUNTED_PLANAR_XI_SCHEMA,
        "source_section": "PPCS.trajectory",
        "source_representation": validated["trajectory"]["representation"],
        "source_section_sha256": hashlib.sha256(canonical_json_bytes(validated["trajectory"])).hexdigest(),
        "source_section_raw_bytes": trajectory_bytes,
        "pair_range": [pair_start, pair_end],
        "embedding": "(tx=dy,ty=ds,yaw=dpsi)->xi=(dy,0,ds,0,dpsi,0)",
        "translation_first": True,
        "additional_video_derived_bytes": 0,
        "decoder_scorer_invocations": 0,
        "lawref_equation_ids": list(ADVECTED_MOTION_LAWREFS),
    }
    return xi, custody


def counted_full_screw_xi_series(
    stored_pose6: np.ndarray,
    *,
    translation_scale: float,
    rotation_scale: float,
    pitch_rad: float,
    pair_start: int = 0,
    pair_end: int | None = None,
    source_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map an already-counted six-coordinate pose stream to complete SE(3) twists.

    Unlike :func:`counted_planar_xi_series`, this path does not zero any source
    coordinate.  ``xi_from_pose_calibration`` maps all three translation and all
    three rotation coordinates through the NumPy ``tac.lie`` exp/log authority.
    The scorer and target are deliberately absent from this decoder primitive.
    """

    from tac.boundary_math import warp_real_luma_frame0 as g1_warp

    poses = np.asarray(stored_pose6, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6 or poses.shape[0] <= 0:
        raise PredictProjectReceiverError("stored full-screw pose must have shape (pairs,6)")
    if not np.all(np.isfinite(poses)):
        raise PredictProjectReceiverError("stored full-screw pose must be finite")
    stop = poses.shape[0] if pair_end is None else pair_end
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (pair_start, stop)):
        raise PredictProjectReceiverError("full-screw pair range must use exact integers")
    if not 0 <= pair_start < stop <= poses.shape[0]:
        raise PredictProjectReceiverError("full-screw pair range is outside stored pose coverage")
    for name, value in (
        ("translation_scale", translation_scale),
        ("rotation_scale", rotation_scale),
        ("pitch_rad", pitch_rad),
    ):
        if not math.isfinite(float(value)):
            raise PredictProjectReceiverError(f"{name} must be finite")
    if translation_scale == 0.0 or rotation_scale == 0.0:
        raise PredictProjectReceiverError("full-screw scales must keep translation and rotation active")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or source_sha256 == "0" * 64
    ):
        raise PredictProjectReceiverError("full-screw source SHA-256 is malformed")

    selected = poses[pair_start:stop]
    xi = np.stack(
        [
            g1_warp.xi_from_pose_calibration(
                pose,
                float(translation_scale),
                float(rotation_scale),
                float(pitch_rad),
                whole_ground=True,
            )
            for pose in selected
        ],
        axis=0,
    )
    if xi.shape != selected.shape or not np.all(np.isfinite(xi)):
        raise PredictProjectReceiverError("full-screw tac.lie conversion produced invalid twists")
    source_nonzero = np.count_nonzero(selected, axis=0)
    xi_nonzero = np.count_nonzero(xi, axis=0)
    if np.any(source_nonzero == 0) or np.any(xi_nonzero == 0):
        raise PredictProjectReceiverError("full-screw coverage requires every stored/xi coordinate to be active")
    custody = {
        "schema": COUNTED_FULL_SCREW_XI_SCHEMA,
        "source_section": "gt_n600.gt_poses already-counted six-coordinate pose sidecar",
        "source_sha256": source_sha256,
        "pair_range": [pair_start, stop],
        "source_coordinate_order": ["forward", "lateral", "vertical", "rot0", "rot1", "rot2"],
        "xi_coordinate_order": ["rho_x", "rho_y", "rho_z", "omega_x", "omega_y", "omega_z"],
        "mapping": "xi=log_se3(make_T(exp_so3(s_r*[pose3,pose4,pose5]),s_t*[pose2,pose1,pose0]))",
        "translation_scale": float(translation_scale),
        "rotation_scale": float(rotation_scale),
        "pitch_rad_ground_geometry": float(pitch_rad),
        "source_nonzero_count_by_coordinate": source_nonzero.astype(int).tolist(),
        "xi_nonzero_count_by_coordinate": xi_nonzero.astype(int).tolist(),
        "all_six_source_coordinates_consumed": True,
        "translation_first": True,
        "additional_video_derived_bytes": 0,
        "decoder_scorer_invocations": 0,
        "lawref_equation_ids": [
            "ego_motion_cumulative_se3_bspline_v1",
            "xi_advected_prior_per_class_chart_reconciliation_v1",
            *ADVECTED_MOTION_LAWREFS[1:],
        ],
    }
    return xi, custody


@dataclass(frozen=True)
class ChartRGBCoefficientPacket:
    """Counted per-pair RGB offsets interpreted in the decoded scene chart."""

    coefficients: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients)
        scales = np.asarray(self.scales)
        if coefficients.dtype != np.int8 or coefficients.ndim != 3:
            raise PredictProjectReceiverError("chart coefficients must be pair x class x RGB int8")
        if coefficients.shape[0] <= 0 or coefficients.shape[1:] != (CLASS_COUNT, 3):
            raise PredictProjectReceiverError("chart coefficients must cover positive pairs and five RGB classes")
        if scales.ndim != 1 or scales.shape[0] != coefficients.shape[0]:
            raise PredictProjectReceiverError("chart coefficient scales must contain one value per pair")
        canonical_scales = scales.astype("<f2", copy=True)
        if not np.all(np.isfinite(canonical_scales)) or np.any(canonical_scales <= 0):
            raise PredictProjectReceiverError("chart coefficient scales must be positive finite float16")
        canonical_coefficients = coefficients.astype(np.int8, copy=True)
        canonical_coefficients.setflags(write=False)
        canonical_scales.setflags(write=False)
        object.__setattr__(self, "coefficients", canonical_coefficients)
        object.__setattr__(self, "scales", canonical_scales)

    @property
    def pair_count(self) -> int:
        return int(self.coefficients.shape[0])


def encode_chart_rgb_coefficients(packet: ChartRGBCoefficientPacket) -> bytes:
    """Serialize a strict receiver-open chart-coefficient payload."""

    coefficient_bytes = packet.coefficients.tobytes(order="C")
    scale_bytes = packet.scales.astype("<f2", copy=False).tobytes(order="C")
    body = coefficient_bytes + scale_bytes
    header = {
        "schema": CHART_RGB_COEFFICIENT_SCHEMA,
        "version": CHART_RGB_COEFFICIENT_VERSION,
        "pair_count": packet.pair_count,
        "class_count": CLASS_COUNT,
        "channels": 3,
        "coefficient_dtype": "int8",
        "scale_dtype": "float16_le",
        "coefficient_bytes": len(coefficient_bytes),
        "scale_bytes": len(scale_bytes),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "receiver_basis": "decoded_scene_chart_class_indicator",
    }
    header_bytes = canonical_json_bytes(header)
    prefix = _CHART_RGB_PREFIX.pack(
        CHART_RGB_COEFFICIENT_MAGIC,
        CHART_RGB_COEFFICIENT_VERSION,
        len(header_bytes),
        len(body),
    )
    checksum = _CHART_RGB_CRC.pack(zlib.crc32(header_bytes + body) & 0xFFFFFFFF)
    return prefix + header_bytes + body + checksum


def decode_chart_rgb_coefficients(payload: bytes) -> ChartRGBCoefficientPacket:
    """Parse a chart-coefficient payload, refusing drift and trailing bytes."""

    if not isinstance(payload, bytes) or len(payload) < _CHART_RGB_PREFIX.size + _CHART_RGB_CRC.size:
        raise PredictProjectReceiverError("chart coefficient payload is truncated or not bytes")
    magic, version, header_size, body_size = _CHART_RGB_PREFIX.unpack_from(payload)
    expected_size = _CHART_RGB_PREFIX.size + header_size + body_size + _CHART_RGB_CRC.size
    if (
        magic != CHART_RGB_COEFFICIENT_MAGIC
        or version != CHART_RGB_COEFFICIENT_VERSION
        or len(payload) != expected_size
    ):
        raise PredictProjectReceiverError("chart coefficient magic/version/length mismatch")
    header_start = _CHART_RGB_PREFIX.size
    body_start = header_start + header_size
    body_end = body_start + body_size
    header_bytes = payload[header_start:body_start]
    body = payload[body_start:body_end]
    (stored_crc,) = _CHART_RGB_CRC.unpack(payload[body_end:])
    if stored_crc != (zlib.crc32(header_bytes + body) & 0xFFFFFFFF):
        raise PredictProjectReceiverError("chart coefficient CRC mismatch")
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictProjectReceiverError("chart coefficient header is malformed") from exc
    required = {
        "schema",
        "version",
        "pair_count",
        "class_count",
        "channels",
        "coefficient_dtype",
        "scale_dtype",
        "coefficient_bytes",
        "scale_bytes",
        "body_sha256",
        "receiver_basis",
    }
    if not isinstance(header, dict) or set(header) != required or canonical_json_bytes(header) != header_bytes:
        raise PredictProjectReceiverError("chart coefficient header is not canonical")
    pair_count = header["pair_count"]
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count <= 0:
        raise PredictProjectReceiverError("chart coefficient pair_count is invalid")
    coefficient_size = pair_count * CLASS_COUNT * 3
    scale_size = pair_count * 2
    if (
        header
        != {
            **header,
            "schema": CHART_RGB_COEFFICIENT_SCHEMA,
            "version": CHART_RGB_COEFFICIENT_VERSION,
            "class_count": CLASS_COUNT,
            "channels": 3,
            "coefficient_dtype": "int8",
            "scale_dtype": "float16_le",
            "coefficient_bytes": coefficient_size,
            "scale_bytes": scale_size,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "receiver_basis": "decoded_scene_chart_class_indicator",
        }
        or len(body) != coefficient_size + scale_size
    ):
        raise PredictProjectReceiverError("chart coefficient header/body custody mismatch")
    coefficients = np.frombuffer(body[:coefficient_size], dtype=np.int8).reshape(pair_count, CLASS_COUNT, 3).copy()
    scales = np.frombuffer(body[coefficient_size:], dtype="<f2").copy()
    return ChartRGBCoefficientPacket(coefficients=coefficients, scales=scales)


def fit_chart_rgb_coefficients(
    baseline_scorer_plane: np.ndarray,
    target_scorer_plane: np.ndarray,
    scene_chart: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit one robust RGB-offset coefficient per decoded chart class."""

    baseline = np.asarray(baseline_scorer_plane)
    target = np.asarray(target_scorer_plane)
    chart = np.asarray(scene_chart)
    if (
        baseline.dtype != np.uint8
        or target.dtype != np.uint8
        or baseline.shape != target.shape
        or baseline.ndim != 3
        or baseline.shape[2] != 3
        or chart.dtype != np.uint8
        or chart.shape != baseline.shape[:2]
        or np.any(chart >= CLASS_COUNT)
    ):
        raise PredictProjectReceiverError("chart coefficient fit inputs have invalid geometry/dtype")
    residual = target.astype(np.int16) - baseline.astype(np.int16)
    values = np.zeros((CLASS_COUNT, 3), dtype=np.float64)
    for class_id in range(CLASS_COUNT):
        selected = residual[chart == class_id]
        if selected.size:
            values[class_id] = np.median(selected, axis=0)
    max_abs = float(np.max(np.abs(values)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    scale16 = float(np.float16(scale))
    if not math.isfinite(scale16) or scale16 <= 0.0:
        raise PredictProjectReceiverError("chart coefficient quantization scale is invalid")
    coefficients = np.clip(np.rint(values / scale16), -127.0, 127.0).astype(np.int8)
    return coefficients, scale16


def apply_chart_rgb_coefficients(
    baseline_scorer_plane: np.ndarray,
    scene_chart: np.ndarray,
    packet: ChartRGBCoefficientPacket,
    pair_index: int,
) -> np.ndarray:
    """Apply one counted coefficient row through its decoded scene-chart basis."""

    baseline = np.asarray(baseline_scorer_plane)
    chart = np.asarray(scene_chart)
    if (
        baseline.dtype != np.uint8
        or baseline.ndim != 3
        or baseline.shape[2] != 3
        or chart.dtype != np.uint8
        or chart.shape != baseline.shape[:2]
        or np.any(chart >= CLASS_COUNT)
    ):
        raise PredictProjectReceiverError("chart coefficient receiver inputs have invalid geometry/dtype")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or not 0 <= pair_index < packet.pair_count:
        raise PredictProjectReceiverError("chart coefficient pair_index is out of range")
    offsets = packet.coefficients[pair_index].astype(np.float64) * float(packet.scales[pair_index])
    corrected = baseline.astype(np.float64) + offsets[chart]
    return np.clip(np.rint(corrected), 0.0, 255.0).astype(np.uint8)


def advect_motion_base(
    frame0_base: np.ndarray,
    frame0_cells: np.ndarray,
    xi: np.ndarray,
    geom: Any,
    *,
    ground_class_ids: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Build ``frame1_base = W_xi(frame0_base)`` and transport its scene chart.

    RGB ground texture and the one-hot categorical chart share the same
    plane-induced homography.  The transported chart supplies the explicit
    ground/off-ground stratification for later PROJECT/exception accounting;
    it is not converted into RGB by an invented palette.  The generic warp is
    scorer-free and executes through the NumPy ``tac.lie`` authority.
    """

    from tac.boundary_math import warp_real_luma_frame0 as g1_warp

    rgb = np.asarray(frame0_base)
    cells = np.asarray(frame0_cells)
    twist = np.asarray(xi, dtype=np.float64)
    expected_hw = tuple(int(value) for value in geom.native_hw)
    if rgb.dtype != np.uint8 or rgb.shape != (*expected_hw, 3):
        raise PredictProjectReceiverError("advected frame0 base must be uint8 HWC at worldsheet geometry")
    if cells.dtype != np.uint8 or cells.shape != expected_hw or np.any(cells >= CLASS_COUNT):
        raise PredictProjectReceiverError("advected frame0 cells must be canonical uint8 class IDs")
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise PredictProjectReceiverError("advected motion requires one finite translation-first xi")
    classes = tuple(int(value) for value in ground_class_ids)
    if not classes or len(set(classes)) != len(classes) or any(not 0 <= value < CLASS_COUNT for value in classes):
        raise PredictProjectReceiverError("ground class IDs must be unique canonical classes")

    warped_rgb_fp = g1_warp.warp_frame0_native_numpy(rgb, twist, geom)
    warped_rgb = np.clip(np.rint(warped_rgb_fp), 0, 255).astype(np.uint8)
    one_hot = (cells[..., None] == np.arange(CLASS_COUNT, dtype=np.uint8)).astype(np.float64)
    warped_one_hot = g1_warp.warp_frame0_native_numpy(one_hot, twist, geom)
    frame1_cells = np.argmax(warped_one_hot, axis=-1).astype(np.uint8)
    ground = np.isin(frame1_cells, classes)
    # The plane-induced homography is geometrically valid only on ground
    # strata.  Keep the source-base value elsewhere so foreground/sky pixels
    # are not falsely assigned ground depth.  The transported scene chart,
    # rather than an encoder-side target mask, selects the receiver branch.
    frame1_base = np.where(ground[..., None], warped_rgb, rgb).astype(np.uint8)
    return {
        "schema": ADVECTED_MOTION_BASE_SCHEMA,
        "frame1_base": frame1_base,
        "frame1_cells": frame1_cells,
        "ground_mask": ground,
        "offground_mask": ~ground,
        "xi_l2": float(np.linalg.norm(twist)),
        "ground_pixels": int(np.count_nonzero(ground)),
        "offground_pixels": int(np.count_nonzero(~ground)),
        "rgb_transport": "transported_ground_chart ? H-warped-RGB : persisted-source-RGB",
        "ground_rgb_transport": "H=K(R(xi)-t(xi)nT/d)K^-1; inverse_bilinear_persist; uint8_round",
        "offground_rgb_transport": "persist_source_base_same_pixel",
        "scene_chart_transport": "one_hot_same_H_then_first_argmax",
        "additional_video_derived_bytes": 0,
        "decoder_scorer_invocations": 0,
        "lawref_equation_ids": list(ADVECTED_MOTION_LAWREFS),
        "frame1_base_sha256": projected_plane_array_sha256(frame1_base),
        "frame1_cells_sha256": projected_plane_array_sha256(frame1_cells),
    }


def _nearest_shift(grid: np.ndarray, tx: float, ty: float, yaw: float) -> np.ndarray:
    height, width = grid.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    dx, dy = xx - cx - tx, yy - cy - ty
    source_x = np.rint(cosine * dx - sine * dy + cx).astype(np.int64)
    source_y = np.rint(sine * dx + cosine * dy + cy).astype(np.int64)
    source_x = np.clip(source_x, 0, width - 1)
    source_y = np.clip(source_y, 0, height - 1)
    return grid[source_y, source_x]


def _interpolate_track_knot(knots: Sequence[Mapping[str, int]], time: int, key: str) -> float:
    if time <= knots[0]["time"]:
        return float(knots[0][key])
    if time >= knots[-1]["time"]:
        return float(knots[-1][key])
    right = next(index for index, knot in enumerate(knots) if knot["time"] >= time)
    left = right - 1
    alpha = (time - knots[left]["time"]) / (knots[right]["time"] - knots[left]["time"])
    return (1.0 - alpha) * knots[left][key] + alpha * knots[right][key]


def _inactive_tracks(seed: Mapping[str, Any], time: int) -> set[int]:
    track_ids = {track["track_id"] for track in seed["movable_tracks"]}
    born_later = {
        event["object_id"] for event in seed["events"] if event["kind"] == "birth" and event["object_id"] in track_ids
    }
    inactive: set[int] = set(born_later)
    for event in seed["events"]:
        if event["time"] > time:
            break
        if event["kind"] in {"death", "occlusion"}:
            inactive.add(event["object_id"])
        elif event["kind"] == "birth":
            inactive.discard(event["object_id"])
        elif event["kind"] == "split":
            inactive.add(event["object_id"])
            inactive.difference_update(event["related_ids"])
        elif event["kind"] == "merge":
            inactive.update(event["related_ids"])
            inactive.discard(event["object_id"])
    return inactive


def derive_ground_chart_raster(seed: Mapping[str, Any]) -> np.ndarray:
    """Derive the site-based fixture; MS-native bulk rasterization is blocked.

    Critical points and arcs do not define bulk cell interiors here.  The arcs
    are consumed only by causal boundary offsets in :func:`predict_cell_field`.
    """

    validated = validate_constraint_seed(seed)
    geometry = validated["ground_chart"]["geometry"]
    payload = derive_morse_smale_raster(validated["ground_chart"])
    return np.frombuffer(payload, dtype=np.uint8).reshape(geometry["scorer_height"], geometry["scorer_width"]).copy()


def _apply_causal_offsets(
    out: np.ndarray,
    offsets: Sequence[Mapping[str, Any]],
    arc_samples: Mapping[int, Mapping[int, Mapping[str, Any]]],
    arc_class_ids: Mapping[int, int],
    *,
    time: int,
    coordinate_scale: float,
    scale: float,
    normal_scale_quantum: float,
) -> None:
    for offset in offsets:
        if offset["time"] != time:
            continue
        point = arc_samples[offset["arc_id"]].get(offset["arc_index"])
        if point is None:
            raise PredictProjectReceiverError("causal offset references an unknown separatrix sample")
        normal_scale = offset["offset_q"] * scale * normal_scale_quantum
        y = round(point["ground_y_q"] * coordinate_scale + normal_scale * point["normal_y_q"])
        x = round(point["ground_x_q"] * coordinate_scale + normal_scale * point["normal_x_q"])
        if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
            out[y, x] = arc_class_ids[offset["arc_id"]]


def predict_cell_field(
    seed: Mapping[str, Any],
    time: int,
    *,
    phase_carrier: Callable[..., np.ndarray] | None = None,
    response_surface: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Predict one scorer-cell field from the declared single spacetime object."""

    validated = validate_constraint_seed(seed)
    chart = derive_ground_chart_raster(validated)
    tx, ty, yaw = trajectory_at(validated, time)
    out = _nearest_shift(chart, tx, ty, yaw)
    inactive = _inactive_tracks(validated, time)
    track_quantum = validated["units"]["track_coordinate"]
    track_scale = track_quantum["numerator"] / track_quantum["denominator"]
    for track in validated["movable_tracks"]:
        if track["track_id"] in inactive:
            continue
        knots = track["knots"]
        if time < knots[0]["time"] or time > knots[-1]["time"]:
            continue
        center_y = _interpolate_track_knot(knots, time, "y_q") * track_scale
        center_x = _interpolate_track_knot(knots, time, "x_q") * track_scale
        height = max(1, round(_interpolate_track_knot(knots, time, "height_q") * track_scale))
        width = max(1, round(_interpolate_track_knot(knots, time, "width_q") * track_scale))
        y0 = max(0, round(center_y - height / 2))
        y1 = min(out.shape[0], y0 + height)
        x0 = max(0, round(center_x - width / 2))
        x1 = min(out.shape[1], x0 + width)
        out[y0:y1, x0:x1] = track["cell_id"]
    arcs = validated["ground_chart"]["separatrix_arcs"]
    arc_samples = {arc["arc_id"]: {point["arc_index"]: point for point in arc["samples"]} for arc in arcs}
    cells_by_id = {cell["cell_id"]: cell for cell in validated["ground_chart"]["cells"]}
    arc_class_ids = {arc["arc_id"]: cells_by_id[arc["right_cell_id"]]["class_id"] for arc in arcs}
    chart_quantum = validated["ground_chart"]["coordinate_quantum"]
    coordinate_scale = chart_quantum["numerator"] / chart_quantum["denominator"]
    normal_quantum = validated["units"]["normal_vector"]
    normal_scale_quantum = normal_quantum["numerator"] / normal_quantum["denominator"]
    ladder = validated["boundary_jitter"]
    rung = ladder["selected_rung"]
    if rung == "R0":
        quantum = ladder["r0"]["offset_quantum"]
        _apply_causal_offsets(
            out,
            ladder["r0"]["offsets"],
            arc_samples,
            arc_class_ids,
            time=time,
            coordinate_scale=coordinate_scale,
            scale=quantum["numerator"] / quantum["denominator"],
            normal_scale_quantum=normal_scale_quantum,
        )
    elif rung == "R1":
        if phase_carrier is None:
            raise PredictProjectReceiverError("selected R1 requires the existing phase-carrier callback")
        quantum = ladder["r0"]["offset_quantum"]
        _apply_causal_offsets(
            out,
            ladder["r1"]["residuals"],
            arc_samples,
            arc_class_ids,
            time=time,
            coordinate_scale=coordinate_scale,
            scale=quantum["numerator"] / quantum["denominator"],
            normal_scale_quantum=normal_scale_quantum,
        )
        resets = tuple(
            event for event in validated["events"] if event["time"] == time and event["kind"] == "phase_reset"
        )
        out = np.asarray(
            phase_carrier(
                seed=validated,
                time=time,
                base_field=out.copy(),
                phase_carrier_id=ladder["r1"]["phase_carrier_id"],
                response_parameters=ladder["r1"]["phase_response_parameters"],
                phase_reset_events=resets,
            )
        )
        if out.dtype != np.uint8 or out.shape != chart.shape:
            raise PredictProjectReceiverError("phase-carrier callback must return a scorer-shaped uint8 cell field")
    elif rung == "R2":
        if response_surface is None:
            raise PredictProjectReceiverError("selected R2 requires the declared generic response-surface callback")
        out = np.asarray(
            response_surface(
                seed=validated,
                time=time,
                base_field=out.copy(),
                trajectory=trajectory_at(validated, time),
                appearance_phase_chart=ladder["r2"]["appearance_phase_chart"],
                response=ladder["r2"]["xi_response"],
            )
        )
        if out.dtype != np.uint8 or out.shape != chart.shape:
            raise PredictProjectReceiverError("R2 response callback must return a scorer-shaped uint8 cell field")
        quantum = ladder["r0"]["offset_quantum"]
        _apply_causal_offsets(
            out,
            ladder["r2"]["exceptions"],
            arc_samples,
            arc_class_ids,
            time=time,
            coordinate_scale=coordinate_scale,
            scale=quantum["numerator"] / quantum["denominator"],
            normal_scale_quantum=normal_scale_quantum,
        )
    if np.any(out >= CLASS_COUNT):
        raise PredictProjectReceiverError("predictor emitted a cell outside the frozen alphabet")
    return np.ascontiguousarray(out, dtype=np.uint8)


def extract_constraint_violations(
    predicted: np.ndarray,
    desired: np.ndarray,
    *,
    time: int,
    frame_index: int = 1,
    strata: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Emit canonical seeds for mismatches only; satisfied cells never serialize."""

    pred, target = np.asarray(predicted), np.asarray(desired)
    if pred.dtype != np.uint8 or target.dtype != np.uint8 or pred.ndim != 2 or pred.shape != target.shape:
        raise PredictProjectReceiverError("predicted/desired cell fields must be equal-shape uint8 2-D arrays")
    if np.any(pred >= CLASS_COUNT) or np.any(target >= CLASS_COUNT):
        raise PredictProjectReceiverError("cell field contains an out-of-range class")
    if isinstance(time, bool) or not isinstance(time, int) or not 0 <= time < 600 or frame_index not in (0, 1):
        raise PredictProjectReceiverError("invalid violation time/frame")
    if frame_index == 0:
        raise PredictProjectReceiverError("frame0 is pose-only; a Seg violation obligation is forbidden")
    if strata is None:
        stratum_values = np.full(pred.shape, "cell_interior", dtype=object)
    else:
        stratum_values = np.asarray(strata, dtype=object)
        if stratum_values.shape != pred.shape or any(value not in STRATA for value in stratum_values.flat):
            raise PredictProjectReceiverError("stratum field is invalid")
    rows: list[dict[str, Any]] = []
    for y, x in np.argwhere(pred != target):
        rows.append(
            {
                "time": time,
                "frame_index": frame_index,
                "obligation": "seg_and_pose",
                "y": int(y),
                "x": int(x),
                "cell_id": int(target[y, x]),
                "predictor_status": "violated",
                "stratum": str(stratum_values[y, x]),
                "pose_tube": None,
                "pose_tightening_id": None,
                "projector": None,
            }
        )
    return rows


def stratify_predictor_quality(
    predicted: np.ndarray,
    desired: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    evidence_source: Literal["declared_constraint_fixture", "hard_oracle_real_desired_cells"] = (
        "declared_constraint_fixture"
    ),
    desired_cells_sha256: str | None = None,
) -> dict[str, Any]:
    pred, target = np.asarray(predicted), np.asarray(desired)
    if pred.shape != target.shape or pred.dtype != np.uint8 or target.dtype != np.uint8:
        raise PredictProjectReceiverError("quality fields must be equal-shape uint8 arrays")
    stratum_values = (
        np.full(pred.shape, "cell_interior", dtype=object) if strata is None else np.asarray(strata, dtype=object)
    )
    if stratum_values.shape != pred.shape or any(value not in STRATA for value in stratum_values.flat):
        raise PredictProjectReceiverError("quality stratum field is invalid")
    if evidence_source not in {"declared_constraint_fixture", "hard_oracle_real_desired_cells"}:
        raise PredictProjectReceiverError("quality evidence source is invalid")
    if evidence_source == "hard_oracle_real_desired_cells":
        _require_sha256(desired_cells_sha256, "desired_cells_sha256")
    elif desired_cells_sha256 is not None:
        raise PredictProjectReceiverError("fixture quality cannot claim a desired-cell source hash")

    def rows_for(keys: np.ndarray) -> dict[str, dict[str, int | float]]:
        result: dict[str, dict[str, int | float]] = {}
        for key in sorted(set(keys.reshape(-1).tolist()), key=str):
            mask = keys == key
            denominator = int(np.count_nonzero(mask))
            numerator = int(np.count_nonzero((pred == target) & mask))
            result[str(key)] = {
                "already_satisfied": numerator,
                "total": denominator,
                "fraction": numerator / denominator if denominator else 0.0,
                "violations": denominator - numerator,
            }
        return result

    total = int(pred.size)
    satisfied = int(np.count_nonzero(pred == target))
    return {
        "schema": "predict_project_b3_stratification.v0",
        "overall": {
            "already_satisfied": satisfied,
            "total": total,
            "fraction": satisfied / total if total else 0.0,
            "violations": total - satisfied,
        },
        "by_class": rows_for(target),
        "by_stratum": rows_for(stratum_values),
        "evidence_source": evidence_source,
        "desired_cells_sha256": desired_cells_sha256,
        "quality_authority": (
            "MEASURED_REAL_DESIRED_CELLS_NON_SOURCE_GROUND_TRUTH"
            if evidence_source == "hard_oracle_real_desired_cells"
            else "NON_AUTHORITATIVE_DECLARED_CONSTRAINT_FIXTURE"
        ),
        "source_ground_truth_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def pose_tightening_for(seed: Mapping[str, Any], time: int, frame_index: int) -> tuple[Mapping[str, Any], ...]:
    """Return only universally proved, scorer-free pixel constraints."""

    validated = validate_constraint_seed(seed)
    if isinstance(time, bool) or not isinstance(time, int) or not 0 <= time < 600 or frame_index not in (0, 1):
        raise PredictProjectReceiverError("invalid pose-tightening time/frame")
    return tuple(
        row for row in validated["pose_tightening"] if row["time"] == time and row["frame_index"] == frame_index
    )


def verify_pose_tightening_choice(
    seed: Mapping[str, Any],
    camera_frame: np.ndarray,
    *,
    time: int,
    frame_index: int,
) -> dict[str, Any]:
    """Verify a decoder choice against proved shippable pixel constraints.

    This does not evaluate PoseNet. The hard-oracle proof establishes that the
    admitted integer polytope is universally inside the declared Pose tube.
    """

    validated = validate_constraint_seed(seed)
    geometry = validated["ground_chart"]["geometry"]
    frame = np.asarray(camera_frame)
    expected_shape = (geometry["camera_height"], geometry["camera_width"], 3)
    if frame.dtype != np.uint8 or frame.shape != expected_shape:
        raise PredictProjectReceiverError("pose tightening requires camera-resolution RGB uint8")
    rows = pose_tightening_for(validated, time, frame_index)
    results: list[dict[str, Any]] = []
    for row in rows:
        values = np.asarray(
            [frame[item["y"], item["x"], item["channel"]] for item in row["pixel_coordinates"]],
            dtype=np.int64,
        )
        lower = np.asarray(row["lower_u8"], dtype=np.int64)
        upper = np.asarray(row["upper_u8"], dtype=np.int64)
        if np.any(values < lower) or np.any(values > upper):
            raise PredictProjectReceiverError("decoder choice violates proved pose-tightening box")
        for constraint in row["linear_constraints"]:
            coefficients = np.asarray(constraint["coefficients_q"], dtype=np.int64)
            if int(np.dot(coefficients, values)) > constraint["upper_q"]:
                raise PredictProjectReceiverError("decoder choice violates proved pose-tightening half-space")
        results.append(
            {
                "tightening_id": row["tightening_id"],
                "proof_assertion_sha256": row["proof"]["assertion_sha256"],
                "universal_pose_tube_admission": True,
            }
        )
    return {
        "schema": "predict_project_pose_tightening_decode_verification.v0",
        "time": time,
        "frame_index": frame_index,
        "verified_tightenings": results,
        "decoder_scorer_invocations": 0,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PredictProjectReceiverError(f"{label} must be finite")
    return float(value)


def validate_m1_anchor_binding(value: Any) -> dict[str, Any]:
    """Bind Task #597 evidence to the exact measured M1 anchors and caveats."""

    if not isinstance(value, dict) or value != M1_ANCHORS:
        raise PredictProjectReceiverError("M1 anchor binding mismatch")
    return value.copy()


def _validate_m1_receipt_binding(value: Any) -> dict[str, Any]:
    expected = {
        "commit": M1_RECEIPT_COMMIT,
        "path": M1_RECEIPT_PATH,
        "sha256": M1_RECEIPT_SHA256,
    }
    if not isinstance(value, dict) or value != expected:
        raise PredictProjectReceiverError("M1 receipt commit/SHA binding mismatch")
    return value.copy()


def _validate_joint_decode_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "decoder_id",
        "decoder_source_sha256",
        "base_archive_sha256",
        "source_cache_sha256",
    }:
        raise PredictProjectReceiverError(f"{label} fields mismatch")
    _require_identity(value["decoder_id"], f"{label} decoder_id")
    for key in ("decoder_source_sha256", "base_archive_sha256", "source_cache_sha256"):
        _require_sha256(value[key], f"{label} {key}")
    return value.copy()


def validate_flip_attribution_receipt(value: Any) -> dict[str, Any]:
    """Validate one #350/#404/#420 full causal attribution receipt."""

    required = {
        "schema",
        "measurement_status",
        "flip_id",
        "family_id",
        "joint_decode_sha256",
        "reuse_bindings",
        "chain_order",
        "causal_chain",
        "artifact_contract_sha256",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("flip attribution fields mismatch")
    chain_order = ["chart_coefficient", "channel", "rank4_hyperplane", "regional_values", "pixels"]
    if (
        value["schema"] != FLIP_ATTRIBUTION_SCHEMA
        or value["measurement_status"] != "MEASURED_EXACT_ATTRIBUTION_SAME_JOINT_DECODE"
        or value["reuse_bindings"] != ATTRIBUTION_REUSE_BINDINGS
        or value["chain_order"] != chain_order
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("flip attribution scope/reuse/order mismatch")
    _require_identity(value["flip_id"], "attribution flip_id")
    _require_identity(value["family_id"], "attribution family_id")
    _require_sha256(value["joint_decode_sha256"], "attribution joint decode")
    _require_sha256(value["artifact_contract_sha256"], "attribution artifact contract")
    chain = value["causal_chain"]
    if not isinstance(chain, dict) or set(chain) != set(chain_order):
        raise PredictProjectReceiverError("attribution causal chain is incomplete")
    chart, channel = chain["chart_coefficient"], chain["channel"]
    hyperplane, regional, pixels = chain["rank4_hyperplane"], chain["regional_values"], chain["pixels"]
    if not isinstance(chart, dict) or set(chart) != {"coefficient_ids", "before_sha256", "after_sha256"}:
        raise PredictProjectReceiverError("chart-coefficient attribution fields mismatch")
    coefficient_ids = chart["coefficient_ids"]
    if not isinstance(coefficient_ids, list) or not coefficient_ids or coefficient_ids != sorted(set(coefficient_ids)):
        raise PredictProjectReceiverError("chart coefficient IDs must be sorted and unique")
    for identity in coefficient_ids:
        _require_identity(identity, "chart coefficient ID")
    if not isinstance(channel, dict) or set(channel) != {"carrier_id", "channel_ids", "before_sha256", "after_sha256"}:
        raise PredictProjectReceiverError("channel attribution fields mismatch")
    _require_identity(channel["carrier_id"], "channel carrier ID")
    channel_ids = channel["channel_ids"]
    if (
        not isinstance(channel_ids, list)
        or not channel_ids
        or channel_ids != sorted(set(channel_ids))
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in channel_ids)
    ):
        raise PredictProjectReceiverError("channel IDs must be sorted nonnegative integers")
    hyperplane_fields = {
        "law_id",
        "feature_q_before",
        "feature_q_after",
        "delta_w_q",
        "signed_distance_q_before",
        "signed_distance_q_after",
    }
    if not isinstance(hyperplane, dict) or set(hyperplane) != hyperplane_fields:
        raise PredictProjectReceiverError("rank4 hyperplane attribution fields mismatch")
    if hyperplane["law_id"] != SEGNET_HEAD_RANK_EQUATION_ID:
        raise PredictProjectReceiverError("rank4 attribution must use the canonical law")
    for key in ("feature_q_before", "feature_q_after", "delta_w_q"):
        vector = hyperplane[key]
        if (
            not isinstance(vector, list)
            or len(vector) != SEGNET_CENTERED_HEAD_RANK
            or any(isinstance(item, bool) or not isinstance(item, int) for item in vector)
        ):
            raise PredictProjectReceiverError(
                "rank4 attribution vectors must match the LawRef-resolved centered-head rank"
            )
    before, after = hyperplane["signed_distance_q_before"], hyperplane["signed_distance_q_after"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (before, after)) or not (
        (before <= 0 < after) or (after <= 0 < before)
    ):
        raise PredictProjectReceiverError("rank4 attribution must prove a decision-hyperplane crossing")
    if not isinstance(regional, dict) or set(regional) != {"region_id", "before_sha256", "after_sha256"}:
        raise PredictProjectReceiverError("regional-value attribution fields mismatch")
    _require_identity(regional["region_id"], "regional attribution ID")
    if not isinstance(pixels, dict) or set(pixels) != {
        "pixel_ids",
        "realized_flip_ids",
        "before_sha256",
        "after_sha256",
    }:
        raise PredictProjectReceiverError("pixel attribution fields mismatch")
    for key in ("pixel_ids", "realized_flip_ids"):
        identities = pixels[key]
        if not isinstance(identities, list) or not identities or identities != sorted(set(identities)):
            raise PredictProjectReceiverError("pixel/flip attribution IDs must be sorted and unique")
        for identity in identities:
            _require_identity(identity, "pixel attribution identity")
    if value["flip_id"] not in pixels["realized_flip_ids"]:
        raise PredictProjectReceiverError("attribution pixel stage must contain the bound flip")
    for stage in (chart, channel, regional, pixels):
        _require_sha256(stage["before_sha256"], "attribution before hash")
        _require_sha256(stage["after_sha256"], "attribution after hash")
    return value.copy()


def validate_ladder_edit_request(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "request_id",
        "family_id",
        "rung",
        "procedure_id",
        "parameters_sha256",
        "before_archive_sha256",
        "joint_decode_sha256",
        "family_membership_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("ladder edit request fields mismatch")
    if value["schema"] != LADDER_EDIT_REQUEST_SCHEMA or value["rung"] not in ACTION_LEVEL_RUNGS:
        raise PredictProjectReceiverError("ladder edit request schema/rung mismatch")
    for key in ("request_id", "family_id", "procedure_id"):
        _require_identity(value[key], f"ladder edit {key}")
    for key in ("parameters_sha256", "before_archive_sha256", "joint_decode_sha256", "family_membership_sha256"):
        _require_sha256(value[key], f"ladder edit {key}")
    return value.copy()


def validate_ladder_edit_response(request: Mapping[str, Any], value: Any) -> dict[str, Any]:
    request = validate_ladder_edit_request(dict(request))
    required = {
        "schema",
        "measurement_status",
        "request_id",
        "deterministic_redecode",
        "joint_decode_sha256",
        "before_archive_sha256",
        "after_archive_sha256",
        "before_output_sha256",
        "after_output_sha256",
        "delta_score",
        "delta_bytes",
        "erf_collateral_flip_ids",
        "erf_collateral_flip_count",
        "erf_collateral_positive_pixel_count",
        "erf_collateral_d_seg",
        "binding_proof_sha256",
        "artifact_contract_sha256",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("ladder edit response fields mismatch")
    if (
        value["schema"] != LADDER_EDIT_RESPONSE_SCHEMA
        or value["measurement_status"] != "MEASURED_DETERMINISTIC_REDECODE_THROUGH_R"
        or value["request_id"] != request["request_id"]
        or value["deterministic_redecode"] is not True
        or value["joint_decode_sha256"] != request["joint_decode_sha256"]
        or value["before_archive_sha256"] != request["before_archive_sha256"]
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("ladder edit response custody/policy mismatch")
    for key in (
        "after_archive_sha256",
        "before_output_sha256",
        "after_output_sha256",
        "binding_proof_sha256",
        "artifact_contract_sha256",
    ):
        _require_sha256(value[key], f"ladder edit response {key}")
    _finite_number(value["delta_score"], "ladder edit delta_score")
    if isinstance(value["delta_bytes"], bool) or not isinstance(value["delta_bytes"], int):
        raise PredictProjectReceiverError("ladder edit delta_bytes must be exact")
    collateral = value["erf_collateral_flip_ids"]
    if not isinstance(collateral, list) or collateral != sorted(set(collateral)):
        raise PredictProjectReceiverError("ladder edit collateral IDs must be sorted and unique")
    if value["erf_collateral_flip_count"] != len(collateral):
        raise PredictProjectReceiverError("ladder edit collateral count mismatch")
    pixels = value["erf_collateral_positive_pixel_count"]
    if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0:
        raise PredictProjectReceiverError("ladder edit collateral pixels must be exact nonnegative integers")
    if _finite_number(value["erf_collateral_d_seg"], "ladder edit collateral d_seg") != pixels / M1_SCORE_DENOMINATOR:
        raise PredictProjectReceiverError("ladder edit collateral d_seg mismatch")
    return value.copy()


def validate_per_flip_sellback_evidence(value: Any) -> dict[str, Any]:
    """Validate an exact M1-bound iterative per-flip sellback fixed point."""

    required = {
        "schema",
        "measurement_status",
        "scope",
        "m1_receipt",
        "joint_decode",
        "context_model",
        "survival_histogram",
        "lambda_star",
        "score_formula",
        "flip_count",
        "stratum_price_anchors_bytes_per_event",
        "flips",
        "iterations",
        "fixed_point",
        "per_stratum_ledger",
        "action_level_ladder",
        "nonmonotone_context_observed",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("per-flip sellback evidence fields mismatch")
    if (
        value["schema"] != "predict_project_per_flip_sellback.v0"
        or value["measurement_status"] != "MEASURED_ITERATIVE_RECODE_FIXED_POINT"
        or value["scope"] != "m1_all_17926_flips_same_joint_decode"
        or value["flip_count"] != M1_FLIP_COUNT
        or isinstance(value["flip_count"], bool)
        or value["stratum_price_anchors_bytes_per_event"] != M1_STRATUM_PRICE_ANCHORS_BYTES_PER_EVENT
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("per-flip sellback scope/authority mismatch")
    if _finite_number(value["lambda_star"], "per-flip lambda_star") != GLOBAL_WATERFILL_LAMBDA_STAR:
        raise PredictProjectReceiverError("per-flip lambda* mismatch")
    receipt = _validate_m1_receipt_binding(value["m1_receipt"])
    joint_decode = _validate_joint_decode_binding(value["joint_decode"], "per-flip joint decode")
    context_model = value["context_model"]
    if not isinstance(context_model, dict) or set(context_model) != {
        "model_id",
        "source_sha256",
        "model_content_sha256",
        "coded_bits_are_exact_integers",
    }:
        raise PredictProjectReceiverError("per-flip context-model custody fields mismatch")
    if context_model["model_id"] != "#557" or context_model["coded_bits_are_exact_integers"] is not True:
        raise PredictProjectReceiverError("per-flip context model must be exact #557 integer-bit custody")
    _require_sha256(context_model["source_sha256"], "context model source")
    _require_sha256(context_model["model_content_sha256"], "context model content")
    survival = value["survival_histogram"]
    if not isinstance(survival, dict) or set(survival) != {"surface_id", "content_sha256"}:
        raise PredictProjectReceiverError("per-flip survival custody fields mismatch")
    if survival["surface_id"] != "r1b7":
        raise PredictProjectReceiverError("per-flip survival must bind r1b7")
    _require_sha256(survival["content_sha256"], "r1b7 histogram content")
    if value["score_formula"] != {
        "numerator": 100,
        "denominator": M1_SCORE_DENOMINATOR,
        "expression": "positive_pixel_count*100/(600*512*384)",
    }:
        raise PredictProjectReceiverError("per-flip score formula mismatch")

    flips = value["flips"]
    if not isinstance(flips, list) or len(flips) != M1_FLIP_COUNT:
        raise PredictProjectReceiverError("per-flip evidence must contain all 17926 flips")
    flip_ids: list[str] = []
    normalized_flips: list[dict[str, Any]] = []
    flip_by_id: dict[str, dict[str, Any]] = {}
    costs_by_iteration: dict[int, dict[str, int]] = {}
    for flip in flips:
        if not isinstance(flip, dict) or set(flip) != {
            "flip_id",
            "stratum",
            "price_stratum",
            "positive_pixel_count",
            "coded_bits",
            "derived_score_value",
            "score_per_coded_byte",
            "decision",
            "coded_bits_by_iteration",
        }:
            raise PredictProjectReceiverError("per-flip row fields mismatch")
        flip_id = _require_identity(flip["flip_id"], "per-flip ID")
        if flip["stratum"] not in GLOBAL_EATEN_FLIP_STRATA or flip["price_stratum"] not in PER_FLIP_PRICE_STRATA:
            raise PredictProjectReceiverError("per-flip stratum is invalid")
        positive_pixels = flip["positive_pixel_count"]
        coded_bits = flip["coded_bits"]
        if (
            isinstance(positive_pixels, bool)
            or not isinstance(positive_pixels, int)
            or positive_pixels <= 0
            or isinstance(coded_bits, bool)
            or not isinstance(coded_bits, int)
            or coded_bits <= 0
        ):
            raise PredictProjectReceiverError("per-flip pixels/bits must be positive exact integers")
        score_value = positive_pixels * 100 / M1_SCORE_DENOMINATOR
        score_per_byte = score_value / (coded_bits / 8)
        if _finite_number(flip["derived_score_value"], "per-flip score value") != score_value:
            raise PredictProjectReceiverError("per-flip derived score value mismatch")
        if _finite_number(flip["score_per_coded_byte"], "per-flip score per byte") != score_per_byte:
            raise PredictProjectReceiverError("per-flip score-per-byte mismatch")
        if flip["decision"] not in {"keep", "eat"}:
            raise PredictProjectReceiverError("per-flip decision must be keep or eat")
        histories = flip["coded_bits_by_iteration"]
        if not isinstance(histories, list) or not histories:
            raise PredictProjectReceiverError("per-flip coded-bit history is missing")
        history_indices: list[int] = []
        for history in histories:
            if not isinstance(history, dict) or set(history) != {"iteration_index", "coded_bits"}:
                raise PredictProjectReceiverError("per-flip coded-bit history fields mismatch")
            iteration_index = history["iteration_index"]
            history_bits = history["coded_bits"]
            if (
                isinstance(iteration_index, bool)
                or not isinstance(iteration_index, int)
                or iteration_index < 0
                or isinstance(history_bits, bool)
                or not isinstance(history_bits, int)
                or history_bits <= 0
            ):
                raise PredictProjectReceiverError("per-flip coded-bit history must be positive integer data")
            history_indices.append(iteration_index)
            costs_by_iteration.setdefault(iteration_index, {})[flip_id] = history_bits
        if history_indices != sorted(set(history_indices)) or histories[-1]["coded_bits"] != coded_bits:
            raise PredictProjectReceiverError("per-flip coded-bit history is noncanonical or disagrees with final bits")
        flip_ids.append(flip_id)
        normalized = {**flip, "derived_score_value": score_value, "score_per_coded_byte": score_per_byte}
        normalized_flips.append(normalized)
        flip_by_id[flip_id] = normalized
    if flip_ids != sorted(set(flip_ids)) or len(flip_by_id) != M1_FLIP_COUNT:
        raise PredictProjectReceiverError("per-flip IDs must be all 17926 sorted and unique")

    iterations = value["iterations"]
    if not isinstance(iterations, list) or not iterations:
        raise PredictProjectReceiverError("per-flip iterative recode chain is missing")
    expected_indices = list(range(len(iterations)))
    prior_input: list[str] | None = None
    prior_costs: dict[str, int] | None = None
    nonmonotone_observed = False
    normalized_iterations: list[dict[str, Any]] = []
    for expected_index, iteration in zip(expected_indices, iterations, strict=True):
        if not isinstance(iteration, dict) or set(iteration) != {
            "iteration_index",
            "input_kept_flip_ids",
            "output_kept_flip_ids",
            "context_sha256",
            "coded_stream_sha256",
            "total_coded_bits",
            "stable",
        }:
            raise PredictProjectReceiverError("per-flip recode iteration fields mismatch")
        if iteration["iteration_index"] != expected_index or isinstance(iteration["iteration_index"], bool):
            raise PredictProjectReceiverError("per-flip recode indices must be contiguous")
        input_ids = iteration["input_kept_flip_ids"]
        output_ids = iteration["output_kept_flip_ids"]
        if (
            not isinstance(input_ids, list)
            or input_ids != sorted(set(input_ids))
            or not isinstance(output_ids, list)
            or output_ids != sorted(set(output_ids))
            or not set(output_ids).issubset(input_ids)
        ):
            raise PredictProjectReceiverError("per-flip recode kept sets are malformed")
        if expected_index == 0 and input_ids != flip_ids:
            raise PredictProjectReceiverError("per-flip first iteration must price all 17926 flips")
        if prior_input is not None and input_ids != prior_input:
            raise PredictProjectReceiverError("per-flip recode iteration chain is broken")
        _require_sha256(iteration["context_sha256"], "per-flip iteration context")
        _require_sha256(iteration["coded_stream_sha256"], "per-flip iteration coded stream")
        costs = costs_by_iteration.get(expected_index, {})
        if set(costs) != set(input_ids):
            raise PredictProjectReceiverError("per-flip iteration costs do not cover the input kept set")
        total_bits = sum(costs.values())
        if iteration["total_coded_bits"] != total_bits or isinstance(iteration["total_coded_bits"], bool):
            raise PredictProjectReceiverError("per-flip iteration coded-bit total is inconsistent")
        expected_output = sorted(
            flip_id
            for flip_id in input_ids
            if flip_by_id[flip_id]["derived_score_value"] / (costs[flip_id] / 8) >= GLOBAL_WATERFILL_LAMBDA_STAR
        )
        if output_ids != expected_output:
            raise PredictProjectReceiverError("per-flip keep/eat decision violates the lambda threshold")
        stable = input_ids == output_ids
        if iteration["stable"] is not stable or (stable and expected_index != len(iterations) - 1):
            raise PredictProjectReceiverError("per-flip stable marker is inconsistent or premature")
        if prior_costs is not None:
            nonmonotone_observed |= any(costs[flip_id] > prior_costs[flip_id] for flip_id in input_ids)
        prior_input = output_ids
        prior_costs = costs
        normalized_iterations.append(dict(iteration))
    if iterations[-1]["stable"] is not True:
        raise PredictProjectReceiverError("per-flip recode chain lacks a final stable fixed point")

    fixed = value["fixed_point"]
    if not isinstance(fixed, dict) or set(fixed) != {
        "iteration_index",
        "kept_flip_ids",
        "eaten_flip_ids",
        "context_sha256",
        "coded_stream_sha256",
        "stable",
    }:
        raise PredictProjectReceiverError("per-flip fixed-point fields mismatch")
    kept_ids = fixed["kept_flip_ids"]
    eaten_ids = fixed["eaten_flip_ids"]
    last = iterations[-1]
    if (
        fixed["iteration_index"] != last["iteration_index"]
        or kept_ids != last["output_kept_flip_ids"]
        or not isinstance(eaten_ids, list)
        or eaten_ids != sorted(set(flip_ids) - set(kept_ids))
        or set(kept_ids) & set(eaten_ids)
        or fixed["context_sha256"] != last["context_sha256"]
        or fixed["coded_stream_sha256"] != last["coded_stream_sha256"]
        or fixed["stable"] is not True
    ):
        raise PredictProjectReceiverError("per-flip fixed point is inconsistent with the recode chain")
    kept_set = set(kept_ids)
    for flip in normalized_flips:
        expected_decision = "keep" if flip["flip_id"] in kept_set else "eat"
        if flip["decision"] != expected_decision:
            raise PredictProjectReceiverError("per-flip final decision disagrees with the fixed point")
    if value["nonmonotone_context_observed"] is not nonmonotone_observed:
        raise PredictProjectReceiverError("per-flip context monotonicity observation is inconsistent")

    ledger = value["per_stratum_ledger"]
    if not isinstance(ledger, list) or len(ledger) != len(PER_FLIP_PRICE_STRATA):
        raise PredictProjectReceiverError("per-flip stratum ledger must cover all price strata")
    normalized_ledger: list[dict[str, Any]] = []
    for expected_stratum, row in zip(PER_FLIP_PRICE_STRATA, ledger, strict=True):
        required_row = {
            "price_stratum",
            "kept_flip_count",
            "eaten_flip_count",
            "kept_coded_bits",
            "eaten_coded_bits",
            "kept_coded_bytes",
            "eaten_coded_bytes",
            "eaten_positive_pixel_count",
            "exact_d_seg_conceded",
        }
        if not isinstance(row, dict) or set(row) != required_row or row["price_stratum"] != expected_stratum:
            raise PredictProjectReceiverError("per-flip stratum ledger fields/order mismatch")
        stratum_flips = [flip for flip in normalized_flips if flip["price_stratum"] == expected_stratum]
        kept = [flip for flip in stratum_flips if flip["decision"] == "keep"]
        eaten_rows = [flip for flip in stratum_flips if flip["decision"] == "eat"]
        kept_bits = sum(flip["coded_bits"] for flip in kept)
        eaten_bits = sum(flip["coded_bits"] for flip in eaten_rows)
        eaten_pixels = sum(flip["positive_pixel_count"] for flip in eaten_rows)
        expected = {
            "price_stratum": expected_stratum,
            "kept_flip_count": len(kept),
            "eaten_flip_count": len(eaten_rows),
            "kept_coded_bits": kept_bits,
            "eaten_coded_bits": eaten_bits,
            "kept_coded_bytes": kept_bits / 8,
            "eaten_coded_bytes": eaten_bits / 8,
            "eaten_positive_pixel_count": eaten_pixels,
            "exact_d_seg_conceded": eaten_pixels / M1_SCORE_DENOMINATOR,
        }
        if row != expected:
            raise PredictProjectReceiverError("per-flip stratum counts/bytes/d_seg are inconsistent")
        normalized_ledger.append(expected)
    normalized_sellback = {
        **value,
        "m1_receipt": receipt,
        "joint_decode": joint_decode,
        "flips": normalized_flips,
        "iterations": normalized_iterations,
        "per_stratum_ledger": normalized_ledger,
        "nonmonotone_context_observed": nonmonotone_observed,
    }
    normalized_sellback["action_level_ladder"] = validate_action_level_ladder_evidence(
        value["action_level_ladder"], normalized_sellback
    )
    return normalized_sellback


def validate_action_level_ladder_evidence(
    value: Any,
    per_flip_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate equal-benefit L1--L5 actuator pricing for every flip family.

    The per-flip sellback remains the value ledger.  This object proves which
    causal actuator is cheapest for each family after through-R ERF collateral;
    it cannot turn pixel prices into an actuator measurement by declaration.
    """

    required = {
        "schema",
        "measurement_status",
        "scope",
        "m1_receipt",
        "joint_decode_sha256",
        "boundary_inverse_policy",
        "ladder_policy",
        "families",
        "chosen_rung_distribution",
        "totals",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("action-level ladder evidence fields mismatch")
    expected_policy = {
        "rungs": list(ACTION_LEVEL_RUNGS),
        "selection": "minimum_coded_bytes_per_net_score_then_canonical_rung",
        "per_flip_ledger_role": "currency_only_not_actuator",
        "l5_policy": "isolated_singletons_only",
        "erf_policy": "charge_exact_through_r_collateral_d_seg",
        "same_joint_decode_required": True,
    }
    if (
        value["schema"] != ACTION_LEVEL_LADDER_SCHEMA
        or value["measurement_status"] != "MEASURED_SAME_JOINT_DECODE_ACTION_LEVEL_LADDER"
        or value["scope"] != "m1_all_17926_flips_all_five_rungs"
        or value["boundary_inverse_policy"] != BOUNDARY_INVERSE_ACTION_POLICY
        or value["ladder_policy"] != expected_policy
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("action-level ladder scope/policy mismatch")
    _validate_m1_receipt_binding(value["m1_receipt"])
    joint_decode = per_flip_evidence.get("joint_decode")
    expected_joint_sha = hashlib.sha256(canonical_json_bytes(joint_decode)).hexdigest()
    if value["joint_decode_sha256"] != expected_joint_sha:
        raise PredictProjectReceiverError("action-level ladder joint-decode hash mismatch")

    flips = per_flip_evidence.get("flips")
    fixed = per_flip_evidence.get("fixed_point")
    if not isinstance(flips, list) or not isinstance(fixed, Mapping):
        raise PredictProjectReceiverError("action-level ladder requires validated per-flip evidence")
    flip_by_id = {flip["flip_id"]: flip for flip in flips}
    all_flip_ids = sorted(flip_by_id)
    kept_ids = fixed.get("kept_flip_ids")
    eaten_ids = fixed.get("eaten_flip_ids")
    if (
        not isinstance(kept_ids, list)
        or not isinstance(eaten_ids, list)
        or kept_ids != sorted(set(kept_ids))
        or eaten_ids != sorted(set(eaten_ids))
        or sorted(kept_ids + eaten_ids) != all_flip_ids
        or set(kept_ids) & set(eaten_ids)
    ):
        raise PredictProjectReceiverError("action-level ladder kept/eaten fixed point is malformed")
    kept_set = set(kept_ids)
    eaten_set = set(eaten_ids)

    families = value["families"]
    if not isinstance(families, list) or not families:
        raise PredictProjectReceiverError("action-level ladder families are missing")
    family_ids: list[str] = []
    member_union: set[str] = set()
    normalized_families: list[dict[str, Any]] = []
    distribution_rows = {
        (receiver_stratum, price_stratum): {
            "receiver_stratum": receiver_stratum,
            "price_stratum": price_stratum,
            "flip_count": 0,
            "kept_flip_count": 0,
            "eaten_flip_count": 0,
            "chosen_rung_flip_counts": dict.fromkeys(ACTION_LEVEL_RUNGS, 0),
            "chosen_rung_family_counts": dict.fromkeys(ACTION_LEVEL_RUNGS, 0),
            "admitted_coded_bits": 0,
            "admitted_coded_bytes": 0,
            "eaten_avoided_coded_bits": 0,
            "eaten_avoided_coded_bytes": 0,
            "erf_collateral_flip_count": 0,
            "erf_collateral_positive_pixel_count": 0,
            "erf_collateral_d_seg": 0.0,
        }
        for receiver_stratum in GLOBAL_EATEN_FLIP_STRATA
        for price_stratum in PER_FLIP_PRICE_STRATA
    }
    selected_collateral_ids: set[str] = set()
    candidate_bits = 0
    candidate_bytes = 0
    admitted_bits = 0
    admitted_bytes = 0
    eaten_avoided_bits = 0
    eaten_avoided_bytes = 0
    admitted_positive_score: list[float] = []
    admitted_collateral_d_seg: list[float] = []
    admitted_collateral_pixels = 0
    admitted_net_score: list[float] = []

    for family in families:
        family_required = {
            "family_id",
            "receiver_stratum",
            "price_stratum",
            "member_flip_ids",
            "family_membership_sha256",
            "decision",
            "isolated_singleton",
            "rungs",
            "selected_rung",
        }
        if not isinstance(family, dict) or set(family) != family_required:
            raise PredictProjectReceiverError("action-level flip-family fields mismatch")
        family_id = _require_identity(family["family_id"], "action-level family_id")
        family_ids.append(family_id)
        members = family["member_flip_ids"]
        if (
            not isinstance(members, list)
            or not members
            or members != sorted(set(members))
            or not set(members).issubset(flip_by_id)
            or member_union & set(members)
        ):
            raise PredictProjectReceiverError("action-level family membership is malformed or overlapping")
        member_union.update(members)
        if family["family_membership_sha256"] != hashlib.sha256(canonical_json_bytes(members)).hexdigest():
            raise PredictProjectReceiverError("action-level family membership hash mismatch")
        receiver_stratum = family["receiver_stratum"]
        price_stratum = family["price_stratum"]
        if receiver_stratum not in GLOBAL_EATEN_FLIP_STRATA or price_stratum not in PER_FLIP_PRICE_STRATA:
            raise PredictProjectReceiverError("action-level family stratum is invalid")
        if any(
            flip_by_id[flip_id]["stratum"] != receiver_stratum or flip_by_id[flip_id]["price_stratum"] != price_stratum
            for flip_id in members
        ):
            raise PredictProjectReceiverError("action-level family disagrees with per-flip strata")
        decision = family["decision"]
        expected_decision = "keep" if set(members).issubset(kept_set) else "eat"
        if (
            decision not in {"keep", "eat"}
            or (set(members) & kept_set and set(members) & eaten_set)
            or decision != expected_decision
        ):
            raise PredictProjectReceiverError("action-level family must be homogeneous at the fixed point")
        isolated_singleton = family["isolated_singleton"]
        if not isinstance(isolated_singleton, bool) or (isolated_singleton and len(members) != 1):
            raise PredictProjectReceiverError("action-level singleton declaration is invalid")

        rungs = family["rungs"]
        if not isinstance(rungs, list) or len(rungs) != len(ACTION_LEVEL_RUNGS):
            raise PredictProjectReceiverError("action-level family must price all five rungs")
        normalized_rungs: list[dict[str, Any]] = []
        valid_rungs: list[dict[str, Any]] = []
        family_pixels = sum(flip_by_id[flip_id]["positive_pixel_count"] for flip_id in members)
        positive_score = family_pixels * 100 / M1_SCORE_DENOMINATOR
        for expected_rung, rung in zip(ACTION_LEVEL_RUNGS, rungs, strict=True):
            rung_required = {
                "rung",
                "actuator_id",
                "measurement_status",
                "valid",
                "invalid_reason",
                "joint_decode_sha256",
                "archive_sha256",
                "decoded_output_sha256",
                "context_sha256",
                "coded_stream_sha256",
                "coded_bits",
                "coded_bytes",
                "positive_flip_count",
                "positive_pixel_count",
                "positive_score_benefit",
                "effective_bytes_per_positive_flip",
                "erf_collateral_flip_ids",
                "erf_collateral_flip_count",
                "erf_collateral_positive_pixel_count",
                "erf_collateral_d_seg",
                "net_score_benefit",
                "net_score_per_coded_byte",
            }
            if not isinstance(rung, dict) or set(rung) != rung_required or rung["rung"] != expected_rung:
                raise PredictProjectReceiverError("action-level rung fields/order mismatch")
            _require_identity(rung["actuator_id"], "action-level actuator_id")
            if (
                rung["measurement_status"] != "MEASURED_SAME_JOINT_DECODE_THROUGH_R"
                or rung["joint_decode_sha256"] != expected_joint_sha
            ):
                raise PredictProjectReceiverError("action-level rung lacks same-joint-decode measurement custody")
            for key in ("archive_sha256", "decoded_output_sha256", "context_sha256", "coded_stream_sha256"):
                _require_sha256(rung[key], f"action-level {key}")
            coded_bits = rung["coded_bits"]
            coded_bytes = rung["coded_bytes"]
            if (
                isinstance(coded_bits, bool)
                or not isinstance(coded_bits, int)
                or coded_bits <= 0
                or isinstance(coded_bytes, bool)
                or not isinstance(coded_bytes, int)
                or coded_bytes != (coded_bits + 7) // 8
            ):
                raise PredictProjectReceiverError("action-level coded bits/bytes are not exact")
            if rung["positive_flip_count"] != len(members) or rung["positive_pixel_count"] != family_pixels:
                raise PredictProjectReceiverError("action-level positive flip benefit is inconsistent")
            if _finite_number(rung["positive_score_benefit"], "action positive score") != positive_score:
                raise PredictProjectReceiverError("action-level positive score benefit mismatch")
            expected_bytes_per_flip = coded_bytes / len(members)
            if (
                _finite_number(rung["effective_bytes_per_positive_flip"], "action bytes per flip")
                != expected_bytes_per_flip
            ):
                raise PredictProjectReceiverError("action-level bytes-per-flip mismatch")
            collateral_ids = rung["erf_collateral_flip_ids"]
            if (
                not isinstance(collateral_ids, list)
                or collateral_ids != sorted(set(collateral_ids))
                or set(collateral_ids) & set(members)
                or rung["erf_collateral_flip_count"] != len(collateral_ids)
            ):
                raise PredictProjectReceiverError("action-level ERF collateral IDs/count are inconsistent")
            for collateral_id in collateral_ids:
                _require_identity(collateral_id, "action-level ERF collateral flip ID")
            collateral_pixels = rung["erf_collateral_positive_pixel_count"]
            if (
                isinstance(collateral_pixels, bool)
                or not isinstance(collateral_pixels, int)
                or collateral_pixels < 0
                or bool(collateral_ids) != (collateral_pixels > 0)
            ):
                raise PredictProjectReceiverError("action-level ERF collateral pixel count is inconsistent")
            collateral_d_seg = collateral_pixels / M1_SCORE_DENOMINATOR
            if _finite_number(rung["erf_collateral_d_seg"], "action collateral d_seg") != collateral_d_seg:
                raise PredictProjectReceiverError("action-level ERF collateral d_seg mismatch")
            net_score = positive_score - 100 * collateral_d_seg
            if _finite_number(rung["net_score_benefit"], "action net score") != net_score:
                raise PredictProjectReceiverError("action-level net score benefit mismatch")
            expected_score_per_byte = net_score / coded_bytes
            if _finite_number(rung["net_score_per_coded_byte"], "action net score per byte") != expected_score_per_byte:
                raise PredictProjectReceiverError("action-level net score-per-byte mismatch")
            valid = rung["valid"]
            if not isinstance(valid, bool):
                raise PredictProjectReceiverError("action-level rung validity must be boolean")
            if expected_rung == "L5_pixel_write" and not isolated_singleton:
                if valid or rung["invalid_reason"] != "L5_NON_SINGLETON_FORBIDDEN":
                    raise PredictProjectReceiverError("L5 is valid only for isolated singletons")
            elif net_score <= 0:
                if valid or rung["invalid_reason"] != "NONPOSITIVE_NET_THROUGH_R_BENEFIT":
                    raise PredictProjectReceiverError("action rung with nonpositive net benefit must be invalid")
            elif valid is not True or rung["invalid_reason"] is not None:
                raise PredictProjectReceiverError("positive-net action rung must be valid")
            normalized = {
                **rung,
                "positive_score_benefit": positive_score,
                "effective_bytes_per_positive_flip": expected_bytes_per_flip,
                "erf_collateral_d_seg": collateral_d_seg,
                "net_score_benefit": net_score,
                "net_score_per_coded_byte": expected_score_per_byte,
            }
            normalized_rungs.append(normalized)
            if valid:
                valid_rungs.append(normalized)
        if not valid_rungs:
            raise PredictProjectReceiverError("action-level family has no valid rung")
        selected = min(
            valid_rungs,
            key=lambda rung: (
                rung["coded_bytes"] / rung["net_score_benefit"],
                ACTION_LEVEL_RUNGS.index(rung["rung"]),
            ),
        )
        if family["selected_rung"] != selected["rung"]:
            raise PredictProjectReceiverError("action-level selected rung is not deterministically cheapest")

        distribution = distribution_rows[(receiver_stratum, price_stratum)]
        distribution["flip_count"] += len(members)
        distribution["kept_flip_count" if decision == "keep" else "eaten_flip_count"] += len(members)
        distribution["chosen_rung_flip_counts"][selected["rung"]] += len(members)
        distribution["chosen_rung_family_counts"][selected["rung"]] += 1
        candidate_bits += selected["coded_bits"]
        candidate_bytes += selected["coded_bytes"]
        if decision == "keep":
            overlap = selected_collateral_ids & set(selected["erf_collateral_flip_ids"])
            if overlap:
                raise PredictProjectReceiverError("selected ERF collateral flip IDs must be credited once")
            selected_collateral_ids.update(selected["erf_collateral_flip_ids"])
            admitted_bits += selected["coded_bits"]
            admitted_bytes += selected["coded_bytes"]
            admitted_positive_score.append(selected["positive_score_benefit"])
            admitted_collateral_d_seg.append(selected["erf_collateral_d_seg"])
            admitted_collateral_pixels += selected["erf_collateral_positive_pixel_count"]
            admitted_net_score.append(selected["net_score_benefit"])
            distribution["admitted_coded_bits"] += selected["coded_bits"]
            distribution["admitted_coded_bytes"] += selected["coded_bytes"]
            distribution["erf_collateral_flip_count"] += selected["erf_collateral_flip_count"]
            distribution["erf_collateral_positive_pixel_count"] += selected["erf_collateral_positive_pixel_count"]
            distribution["erf_collateral_d_seg"] += selected["erf_collateral_d_seg"]
        else:
            eaten_avoided_bits += selected["coded_bits"]
            eaten_avoided_bytes += selected["coded_bytes"]
            distribution["eaten_avoided_coded_bits"] += selected["coded_bits"]
            distribution["eaten_avoided_coded_bytes"] += selected["coded_bytes"]
        normalized_families.append({**family, "rungs": normalized_rungs})

    if family_ids != sorted(set(family_ids)) or member_union != set(all_flip_ids):
        raise PredictProjectReceiverError("action-level families must canonically partition all 17926 flips")
    expected_distribution = list(distribution_rows.values())
    if value["chosen_rung_distribution"] != expected_distribution:
        raise PredictProjectReceiverError("action-level chosen-rung distribution is inconsistent")
    expected_totals = {
        "family_count": len(families),
        "flip_count": len(all_flip_ids),
        "kept_flip_count": len(kept_ids),
        "eaten_flip_count": len(eaten_ids),
        "candidate_selected_coded_bits": candidate_bits,
        "candidate_selected_coded_bytes": candidate_bytes,
        "admitted_coded_bits": admitted_bits,
        "admitted_coded_bytes": admitted_bytes,
        "eaten_avoided_coded_bits": eaten_avoided_bits,
        "eaten_avoided_coded_bytes": eaten_avoided_bytes,
        "admitted_positive_score_benefit": math.fsum(admitted_positive_score),
        "admitted_erf_collateral_flip_ids": sorted(selected_collateral_ids),
        "admitted_erf_collateral_flip_count": len(selected_collateral_ids),
        "admitted_erf_collateral_positive_pixel_count": admitted_collateral_pixels,
        "admitted_erf_collateral_d_seg": math.fsum(admitted_collateral_d_seg),
        "admitted_net_score_benefit": math.fsum(admitted_net_score),
    }
    if value["totals"] != expected_totals:
        raise PredictProjectReceiverError("action-level ladder totals are inconsistent")
    return {
        **value,
        "families": normalized_families,
        "chosen_rung_distribution": expected_distribution,
        "totals": expected_totals,
    }


def validate_attribution_edit_telemetry(
    value: Any,
    per_flip_evidence: Mapping[str, Any],
    action_ladder: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate attribution hashes and every measured L1--L5 edit response."""

    required = {
        "schema",
        "measurement_status",
        "scope",
        "joint_decode_sha256",
        "reuse_bindings",
        "action_ladder_sha256",
        "flip_attribution_bindings",
        "edit_receipts",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("attribution/edit telemetry fields mismatch")
    expected_joint_sha = hashlib.sha256(canonical_json_bytes(per_flip_evidence["joint_decode"])).hexdigest()
    if (
        value["schema"] != ATTRIBUTION_EDIT_TELEMETRY_SCHEMA
        or value["measurement_status"] != "MEASURED_EXACT_ATTRIBUTION_AND_LADDER_EDITS"
        or value["scope"] != "m1_all_17926_flips_all_five_rungs"
        or value["joint_decode_sha256"] != expected_joint_sha
        or value["reuse_bindings"] != ATTRIBUTION_REUSE_BINDINGS
        or value["action_ladder_sha256"] != hashlib.sha256(canonical_json_bytes(action_ladder)).hexdigest()
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("attribution/edit telemetry custody or ladder binding mismatch")
    families = action_ladder["families"]
    family_by_id = {family["family_id"]: family for family in families}
    expected_family_by_flip = {
        flip_id: family["family_id"] for family in families for flip_id in family["member_flip_ids"]
    }
    bindings = value["flip_attribution_bindings"]
    if not isinstance(bindings, list) or len(bindings) != M1_FLIP_COUNT:
        raise PredictProjectReceiverError("attribution bindings must cover all 17926 flips")
    normalized_bindings: list[dict[str, Any]] = []
    seen_flip_ids: list[str] = []
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"flip_id", "family_id", "receipt_sha256"}:
            raise PredictProjectReceiverError("attribution binding fields mismatch")
        flip_id = _require_identity(binding["flip_id"], "attribution binding flip_id")
        family_id = _require_identity(binding["family_id"], "attribution binding family_id")
        if expected_family_by_flip.get(flip_id) != family_id:
            raise PredictProjectReceiverError("attribution binding disagrees with action-family membership")
        _require_sha256(binding["receipt_sha256"], "attribution receipt")
        seen_flip_ids.append(flip_id)
        normalized_bindings.append(dict(binding))
    if seen_flip_ids != sorted(expected_family_by_flip):
        raise PredictProjectReceiverError("attribution bindings are missing, duplicate, or noncanonical")

    edits = value["edit_receipts"]
    expected_edit_count = len(families) * len(ACTION_LEVEL_RUNGS)
    if not isinstance(edits, list) or len(edits) != expected_edit_count:
        raise PredictProjectReceiverError("edit telemetry must cover every family/rung")
    normalized_edits: list[dict[str, Any]] = []
    expected_keys = [(family["family_id"], rung_name) for family in families for rung_name in ACTION_LEVEL_RUNGS]
    actual_keys: list[tuple[str, str]] = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {
            "family_id",
            "rung",
            "action_rung_sha256",
            "request",
            "response",
            "receipt_sha256",
        }:
            raise PredictProjectReceiverError("edit telemetry receipt fields mismatch")
        family_id = _require_identity(edit["family_id"], "edit telemetry family_id")
        rung_name = edit["rung"]
        if family_id not in family_by_id or rung_name not in ACTION_LEVEL_RUNGS:
            raise PredictProjectReceiverError("edit telemetry references an unknown family/rung")
        family = family_by_id[family_id]
        rung = next(row for row in family["rungs"] if row["rung"] == rung_name)
        rung_sha = hashlib.sha256(canonical_json_bytes(rung)).hexdigest()
        if edit["action_rung_sha256"] != rung_sha:
            raise PredictProjectReceiverError("edit telemetry is not byte-identical to the action rung")
        request = validate_ladder_edit_request(edit["request"])
        response = validate_ladder_edit_response(request, edit["response"])
        if (
            request["family_id"] != family_id
            or request["rung"] != rung_name
            or request["joint_decode_sha256"] != expected_joint_sha
            or request["family_membership_sha256"] != family["family_membership_sha256"]
            or response["after_archive_sha256"] != rung["archive_sha256"]
            or response["after_output_sha256"] != rung["decoded_output_sha256"]
            or response["delta_bytes"] != rung["coded_bytes"]
            or response["delta_score"] != -rung["net_score_benefit"]
            or response["erf_collateral_flip_ids"] != rung["erf_collateral_flip_ids"]
            or response["erf_collateral_flip_count"] != rung["erf_collateral_flip_count"]
            or response["erf_collateral_positive_pixel_count"] != rung["erf_collateral_positive_pixel_count"]
            or response["erf_collateral_d_seg"] != rung["erf_collateral_d_seg"]
        ):
            raise PredictProjectReceiverError("edit telemetry metrics/custody disagree with the action rung")
        receipt_sha = hashlib.sha256(canonical_json_bytes({"request": request, "response": response})).hexdigest()
        if edit["receipt_sha256"] != receipt_sha:
            raise PredictProjectReceiverError("edit telemetry receipt hash mismatch")
        actual_keys.append((family_id, rung_name))
        normalized_edits.append({**edit, "request": request, "response": response})
    if actual_keys != expected_keys:
        raise PredictProjectReceiverError("edit telemetry family/rung order is noncanonical")
    return {**value, "flip_attribution_bindings": normalized_bindings, "edit_receipts": normalized_edits}


def validate_pose_tube_knee_evidence(value: Any) -> dict[str, Any]:
    """Validate a same-joint-decode measured Pose relaxation crossing."""

    required = {
        "schema",
        "measurement_status",
        "scope",
        "m1_receipt",
        "joint_decode",
        "lambda_star",
        "pose_score_term",
        "points",
        "selected_crossing",
        "kkt_claim",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("pose-tube knee evidence fields mismatch")
    if (
        value["schema"] != "predict_project_pose_tube_knee.v0"
        or value["measurement_status"] != "MEASURED_SAME_JOINT_DECODE_POSE_TUBE_SWEEP"
        or value["scope"] != "full_n600_increasing_pose_tube_relaxation"
        or value["pose_score_term"] != "sqrt(10*d_pose)"
        or value["kkt_claim"] is not False
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("pose-tube knee scope/authority mismatch")
    if _finite_number(value["lambda_star"], "pose-knee lambda_star") != GLOBAL_WATERFILL_LAMBDA_STAR:
        raise PredictProjectReceiverError("pose-tube knee lambda* mismatch")
    receipt = _validate_m1_receipt_binding(value["m1_receipt"])
    joint_decode = _validate_joint_decode_binding(value["joint_decode"], "pose-knee joint decode")
    points = value["points"]
    if not isinstance(points, list) or len(points) < 3:
        raise PredictProjectReceiverError("pose-tube knee requires base, admitted, and rejected points")
    normalized_points: list[dict[str, Any]] = []
    relaxation_keys: list[tuple[int, str]] = []
    previous: dict[str, Any] | None = None
    for point in points:
        if not isinstance(point, dict) or set(point) != {
            "point_id",
            "tube_relaxation_q",
            "d_pose",
            "archive_bytes",
            "byte_savings_from_previous",
            "nonlinear_sqrt_score_delta",
            "marginal_score_per_byte",
            "archive_sha256",
            "decoded_output_sha256",
        }:
            raise PredictProjectReceiverError("pose-tube knee point fields mismatch")
        point_id = _require_identity(point["point_id"], "pose-knee point_id")
        relaxation = point["tube_relaxation_q"]
        archive_bytes = point["archive_bytes"]
        if (
            isinstance(relaxation, bool)
            or not isinstance(relaxation, int)
            or relaxation < 0
            or isinstance(archive_bytes, bool)
            or not isinstance(archive_bytes, int)
            or archive_bytes < 0
        ):
            raise PredictProjectReceiverError("pose-knee relaxation/bytes must be nonnegative exact integers")
        d_pose = _finite_number(point["d_pose"], "pose-knee d_pose")
        if d_pose < 0:
            raise PredictProjectReceiverError("pose-knee d_pose must be nonnegative")
        _require_sha256(point["archive_sha256"], "pose-knee archive")
        _require_sha256(point["decoded_output_sha256"], "pose-knee decoded output")
        if previous is None:
            expected_savings = 0
            expected_delta = 0.0
            expected_marginal = None
        else:
            if d_pose <= previous["d_pose"] or archive_bytes >= previous["archive_bytes"]:
                raise PredictProjectReceiverError("pose-knee sweep must increase d_pose and reduce bytes")
            expected_savings = previous["archive_bytes"] - archive_bytes
            expected_delta = math.sqrt(10 * d_pose) - math.sqrt(10 * previous["d_pose"])
            expected_marginal = expected_delta / expected_savings
        if point["byte_savings_from_previous"] != expected_savings:
            raise PredictProjectReceiverError("pose-knee byte savings mismatch")
        if _finite_number(point["nonlinear_sqrt_score_delta"], "pose-knee score delta") != expected_delta:
            raise PredictProjectReceiverError("pose-knee nonlinear sqrt score delta mismatch")
        if expected_marginal is None:
            if point["marginal_score_per_byte"] is not None:
                raise PredictProjectReceiverError("pose-knee base point cannot have a marginal")
        elif _finite_number(point["marginal_score_per_byte"], "pose-knee marginal") != expected_marginal:
            raise PredictProjectReceiverError("pose-knee marginal score-per-byte mismatch")
        normalized = {
            **point,
            "d_pose": d_pose,
            "nonlinear_sqrt_score_delta": expected_delta,
            "marginal_score_per_byte": expected_marginal,
        }
        normalized_points.append(normalized)
        relaxation_keys.append((relaxation, point_id))
        previous = normalized
    if relaxation_keys != sorted(set(relaxation_keys)):
        raise PredictProjectReceiverError("pose-knee points must be canonically ordered and unique")
    crossing = value["selected_crossing"]
    if not isinstance(crossing, dict) or set(crossing) != {
        "selected_point_id",
        "next_rejected_point_id",
        "selection_policy",
    }:
        raise PredictProjectReceiverError("pose-knee selected crossing fields mismatch")
    selected_index = 0
    for index in range(1, len(normalized_points)):
        marginal = normalized_points[index]["marginal_score_per_byte"]
        if marginal is not None and marginal <= GLOBAL_WATERFILL_LAMBDA_STAR:
            selected_index = index
            continue
        break
    if selected_index + 1 >= len(normalized_points):
        raise PredictProjectReceiverError("pose-knee sweep has no measured lambda crossing")
    if any(
        point["marginal_score_per_byte"] is None or point["marginal_score_per_byte"] <= GLOBAL_WATERFILL_LAMBDA_STAR
        for point in normalized_points[selected_index + 1 :]
    ):
        raise PredictProjectReceiverError("pose-knee sweep does not have one stable measured lambda crossing")
    if crossing != {
        "selected_point_id": normalized_points[selected_index]["point_id"],
        "next_rejected_point_id": normalized_points[selected_index + 1]["point_id"],
        "selection_policy": "last_marginal_le_lambda_before_first_gt_lambda",
    }:
        raise PredictProjectReceiverError("pose-knee selected crossing is inconsistent with measured marginals")
    return {**value, "m1_receipt": receipt, "joint_decode": joint_decode, "points": normalized_points}


def validate_learned_tail_race_evidence(
    value: Any,
    *,
    custody_sha256: str,
    joint_decode_sha256: str,
) -> dict[str, Any]:
    """Validate literal/generator/eaten races without performing training."""

    required = {
        "schema",
        "measurement_status",
        "scope",
        "streams",
        "lambda_star",
        "custody_sha256",
        "joint_decode_sha256",
        "trainer_reuse",
        "rule118",
        "stream_races",
        "admitted_streams",
        "learned_default",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("learned-tail race fields mismatch")
    if (
        value["schema"] != LEARNED_TAIL_RACE_SCHEMA
        or value["measurement_status"] != "MEASURED_EQUAL_FIDELITY_THREE_WAY_RACE"
        or value["scope"] != "same_joint_decode_all_global_streams"
        or value["streams"] != list(GLOBAL_WATERFILL_STREAMS)
        or _finite_number(value["lambda_star"], "learned-tail lambda") != GLOBAL_WATERFILL_LAMBDA_STAR
        or value["custody_sha256"] != custody_sha256
        or value["joint_decode_sha256"] != joint_decode_sha256
        or value["trainer_reuse"] != S3_TRAINER_REUSE
        or value["rule118"]
        != {
            "generic_generator_compute_is_free": True,
            "generator_weights_are_counted_payload": True,
            "instance_seeds_are_counted_payload": True,
            "own_exceptions_are_counted_payload": True,
            "training_or_launch_performed": False,
        }
        or value["learned_default"] != "ABSENT_UNLESS_GENERATOR_STRICTLY_WINS"
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("learned-tail race custody/policy mismatch")
    races = value["stream_races"]
    if not isinstance(races, list) or len(races) != len(GLOBAL_WATERFILL_STREAMS):
        raise PredictProjectReceiverError("learned-tail races must cover every canonical stream")
    normalized_races: list[dict[str, Any]] = []
    admitted: list[str] = []
    for expected_stream, race in zip(GLOBAL_WATERFILL_STREAMS, races, strict=True):
        if not isinstance(race, dict) or set(race) != {
            "stream",
            "equal_realized_fidelity_sha256",
            "hard_oracle_output_sha256",
            "alternatives",
            "winner",
        }:
            raise PredictProjectReceiverError("learned-tail stream race fields mismatch")
        if race["stream"] != expected_stream:
            raise PredictProjectReceiverError("learned-tail stream race order is noncanonical")
        _require_sha256(race["equal_realized_fidelity_sha256"], "learned-tail equal-fidelity hash")
        _require_sha256(race["hard_oracle_output_sha256"], "learned-tail hard-oracle output")
        alternatives = race["alternatives"]
        if not isinstance(alternatives, list) or len(alternatives) != 3:
            raise PredictProjectReceiverError("learned-tail race requires exactly three alternatives")
        normalized_alternatives: list[dict[str, Any]] = []
        for expected_option, alternative in zip(
            ("literal_exceptions", "learned_generator", "eaten_flip"), alternatives, strict=True
        ):
            if (
                not isinstance(alternative, dict)
                or set(alternative)
                != {
                    "option",
                    "exact_bytes",
                    "delta_score",
                    "lagrangian_cost",
                    "breakdown",
                }
                or alternative["option"] != expected_option
            ):
                raise PredictProjectReceiverError("learned-tail alternatives are missing or noncanonical")
            exact_bytes = alternative["exact_bytes"]
            if isinstance(exact_bytes, bool) or not isinstance(exact_bytes, int) or exact_bytes < 0:
                raise PredictProjectReceiverError("learned-tail exact bytes must be nonnegative integers")
            delta_score = _finite_number(alternative["delta_score"], "learned-tail delta score")
            expected_cost = delta_score + GLOBAL_WATERFILL_LAMBDA_STAR * exact_bytes
            if _finite_number(alternative["lagrangian_cost"], "learned-tail Lagrangian") != expected_cost:
                raise PredictProjectReceiverError("learned-tail Lagrangian cost mismatch")
            breakdown = alternative["breakdown"]
            if expected_option == "literal_exceptions":
                if not isinstance(breakdown, dict) or breakdown != {"literal_exception_bytes": exact_bytes}:
                    raise PredictProjectReceiverError("literal-exception byte breakdown mismatch")
            elif expected_option == "learned_generator":
                if not isinstance(breakdown, dict) or set(breakdown) != {
                    "counted_weight_bytes",
                    "instance_seed_bytes",
                    "own_exception_bytes",
                    "weights_sha256",
                    "instance_seed_sha256",
                    "own_exceptions_sha256",
                }:
                    raise PredictProjectReceiverError("learned-generator counted payload breakdown mismatch")
                byte_parts = [
                    breakdown[key] for key in ("counted_weight_bytes", "instance_seed_bytes", "own_exception_bytes")
                ]
                if (
                    any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in byte_parts)
                    or sum(byte_parts) != exact_bytes
                ):
                    raise PredictProjectReceiverError("learned-generator counted bytes do not sum exactly")
                for key in ("weights_sha256", "instance_seed_sha256", "own_exceptions_sha256"):
                    _require_sha256(breakdown[key], f"learned generator {key}")
            else:
                if not isinstance(breakdown, dict) or set(breakdown) != {
                    "flip_ids",
                    "positive_pixel_count",
                    "exact_d_seg",
                    "score_cost",
                }:
                    raise PredictProjectReceiverError("eaten-flip learned-tail breakdown mismatch")
                flip_ids = breakdown["flip_ids"]
                pixels = breakdown["positive_pixel_count"]
                if not isinstance(flip_ids, list) or flip_ids != sorted(set(flip_ids)):
                    raise PredictProjectReceiverError("learned-tail eaten flip IDs must be sorted and unique")
                if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0:
                    raise PredictProjectReceiverError("learned-tail eaten pixel count must be exact")
                d_seg = pixels / M1_SCORE_DENOMINATOR
                if breakdown["exact_d_seg"] != d_seg or breakdown["score_cost"] != 100 * d_seg:
                    raise PredictProjectReceiverError("learned-tail eaten score cost mismatch")
                if exact_bytes != 0 or delta_score != breakdown["score_cost"]:
                    raise PredictProjectReceiverError("eaten-flip race alternative bytes/score mismatch")
            normalized_alternatives.append(
                {**alternative, "delta_score": delta_score, "lagrangian_cost": expected_cost}
            )
        if normalized_alternatives[0]["delta_score"] != normalized_alternatives[1]["delta_score"]:
            raise PredictProjectReceiverError("literal and learned alternatives lack equal realized fidelity")
        costs = [alternative["lagrangian_cost"] for alternative in normalized_alternatives]
        minimum = min(costs)
        if costs.count(minimum) != 1:
            raise PredictProjectReceiverError("learned-tail race has no unique deterministic winner")
        winner = normalized_alternatives[costs.index(minimum)]["option"]
        if race["winner"] != winner:
            raise PredictProjectReceiverError("learned-tail declared winner is not the strict Lagrangian minimum")
        if winner == "learned_generator":
            admitted.append(expected_stream)
        normalized_races.append({**race, "alternatives": normalized_alternatives, "winner": winner})
    if value["admitted_streams"] != admitted:
        raise PredictProjectReceiverError("learned-tail admitted streams do not match strict winners")
    return {**value, "stream_races": normalized_races, "admitted_streams": admitted}


def validate_global_joint_waterfill_evidence(value: Any) -> dict[str, Any]:
    """Validate one global, same-decode, interaction-aware measured sweep."""

    required = {
        "schema",
        "measurement_status",
        "scope",
        "pair_range",
        "pair_count",
        "same_joint_decode",
        "lambda_star",
        "m1_anchors",
        "streams",
        "custody",
        "joint_decode",
        "overlap_credit_policy",
        "composition_policy",
        "interaction_definition",
        "points",
        "per_stream_marginal_curves",
        "global_allocation",
        "per_flip_sellback",
        "action_level_ladder",
        "attribution_edit_telemetry",
        "learned_tail_race",
        "pose_tube_knee",
        "eaten_flip_decomposition",
        "pairwise_interaction_matrix",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PredictProjectReceiverError("global joint-waterfill evidence fields mismatch")
    if (
        value["schema"] != GLOBAL_WATERFILL_SCHEMA
        or value["measurement_status"] != "MEASURED_GLOBAL_JOINT_SWEEP"
        or value["scope"] != "aggregate_same_joint_decode_full_600"
        or value["pair_range"] != [0, 600]
        or value["pair_count"] != 600
        or isinstance(value["pair_count"], bool)
        or value["same_joint_decode"] is not True
        or value["streams"] != list(GLOBAL_WATERFILL_STREAMS)
        or value["overlap_credit_policy"] != "global_flip_id_union_once.v1"
        or value["composition_policy"] != "ordered_commutator_aware.v1"
        or value["interaction_definition"] != "delta_joint_minus_sum_of_singles.v1"
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise PredictProjectReceiverError("global joint-waterfill scope/policy is not authoritative")
    if _finite_number(value["lambda_star"], "lambda_star") != GLOBAL_WATERFILL_LAMBDA_STAR:
        raise PredictProjectReceiverError("global joint-waterfill lambda* mismatch")
    m1_anchors = validate_m1_anchor_binding(value["m1_anchors"])
    custody = validate_hard_oracle_custody(value["custody"])
    decode = _validate_joint_decode_binding(value["joint_decode"], "joint-decode custody")
    sellback = validate_per_flip_sellback_evidence(value["per_flip_sellback"])
    action_ladder = validate_action_level_ladder_evidence(value["action_level_ladder"], sellback)
    attribution_telemetry = validate_attribution_edit_telemetry(
        value["attribution_edit_telemetry"], sellback, action_ladder
    )
    learned_tail_race = validate_learned_tail_race_evidence(
        value["learned_tail_race"],
        custody_sha256=hard_oracle_custody_sha256(custody),
        joint_decode_sha256=hashlib.sha256(canonical_json_bytes(decode)).hexdigest(),
    )
    pose_knee = validate_pose_tube_knee_evidence(value["pose_tube_knee"])
    if canonical_json_bytes(action_ladder) != canonical_json_bytes(sellback["action_level_ladder"]):
        raise PredictProjectReceiverError("global action ladder must equal the per-flip action ladder")
    if canonical_json_bytes(sellback["joint_decode"]) != canonical_json_bytes(decode):
        raise PredictProjectReceiverError("per-flip sellback must use the same joint decode")
    if canonical_json_bytes(pose_knee["joint_decode"]) != canonical_json_bytes(decode):
        raise PredictProjectReceiverError("pose-tube knee must use the same joint decode")

    points = value["points"]
    if not isinstance(points, list) or len(points) < 2:
        raise PredictProjectReceiverError("global joint-waterfill requires at least two measured compositions")
    point_ids: list[str] = []
    normalized_points: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {
            "point_id",
            "ordered_streams",
            "settings_sha256",
            "archive_sha256",
            "decoded_output_sha256",
            "delta_score",
            "delta_bytes",
            "stream_flip_ids",
            "credited_flip_ids",
        }:
            raise PredictProjectReceiverError("global joint-waterfill point fields mismatch")
        point_id = _require_identity(point["point_id"], "waterfill point_id")
        point_ids.append(point_id)
        order = point["ordered_streams"]
        if (
            not isinstance(order, list)
            or not order
            or len(order) != len(set(order))
            or not set(order).issubset(GLOBAL_WATERFILL_STREAMS)
        ):
            raise PredictProjectReceiverError("waterfill composition order is invalid")
        for key in ("settings_sha256", "archive_sha256", "decoded_output_sha256"):
            _require_sha256(point[key], f"waterfill point {key}")
        delta_score = _finite_number(point["delta_score"], "waterfill delta_score")
        delta_bytes = point["delta_bytes"]
        if isinstance(delta_bytes, bool) or not isinstance(delta_bytes, int):
            raise PredictProjectReceiverError("waterfill delta_bytes must be an exact integer")
        flip_map = point["stream_flip_ids"]
        if not isinstance(flip_map, dict) or set(flip_map) != set(GLOBAL_WATERFILL_STREAMS):
            raise PredictProjectReceiverError("waterfill stream flip map must cover the canonical streams")
        union: set[str] = set()
        for stream in GLOBAL_WATERFILL_STREAMS:
            flips = flip_map[stream]
            if not isinstance(flips, list) or flips != sorted(set(flips)):
                raise PredictProjectReceiverError("stream flip IDs must be sorted and unique")
            for flip_id in flips:
                union.add(_require_identity(flip_id, "flip_id"))
        if point["credited_flip_ids"] != sorted(union):
            raise PredictProjectReceiverError("overlapping flip IDs must be credited exactly once")
        normalized_points.append(
            {
                **point,
                "delta_score": delta_score,
                "delta_bytes": delta_bytes,
                "lagrangian_delta": delta_score + GLOBAL_WATERFILL_LAMBDA_STAR * delta_bytes,
            }
        )
    if point_ids != sorted(set(point_ids)):
        raise PredictProjectReceiverError("waterfill points must have sorted unique IDs")
    point_by_id = {point["point_id"]: point for point in normalized_points}
    appearing_streams = {stream for point in normalized_points for stream in point["ordered_streams"]}
    if appearing_streams != set(GLOBAL_WATERFILL_STREAMS):
        raise PredictProjectReceiverError("every canonical stream must appear in measured points")

    curves = value["per_stream_marginal_curves"]
    if not isinstance(curves, list) or len(curves) != len(GLOBAL_WATERFILL_STREAMS):
        raise PredictProjectReceiverError("per-stream marginal curves must cover all canonical streams")
    normalized_curves: list[dict[str, Any]] = []
    for expected_stream, curve in zip(GLOBAL_WATERFILL_STREAMS, curves, strict=True):
        if not isinstance(curve, dict) or set(curve) != {"stream", "points"}:
            raise PredictProjectReceiverError("per-stream marginal curve fields mismatch")
        if curve["stream"] != expected_stream:
            raise PredictProjectReceiverError("per-stream marginal curves are missing or noncanonical")
        curve_points = curve["points"]
        if not isinstance(curve_points, list) or len(curve_points) < 2:
            raise PredictProjectReceiverError("each marginal curve requires at least two measured points")
        normalized_curve_points: list[dict[str, Any]] = []
        curve_keys: list[tuple[int, str]] = []
        seen_curve_ids: set[str] = set()
        for curve_point in curve_points:
            if not isinstance(curve_point, dict) or set(curve_point) != {
                "point_id",
                "delta_score",
                "delta_bytes",
            }:
                raise PredictProjectReceiverError("marginal curve point fields mismatch")
            point_id = _require_identity(curve_point["point_id"], "marginal curve point_id")
            if point_id in seen_curve_ids or point_id not in point_by_id:
                raise PredictProjectReceiverError("marginal curve references duplicate or unknown point")
            measured = point_by_id[point_id]
            if expected_stream not in measured["ordered_streams"]:
                raise PredictProjectReceiverError("marginal curve references a point without its stream")
            delta_score = _finite_number(curve_point["delta_score"], "marginal curve delta_score")
            delta_bytes = curve_point["delta_bytes"]
            if isinstance(delta_bytes, bool) or not isinstance(delta_bytes, int):
                raise PredictProjectReceiverError("marginal curve delta_bytes must be an exact integer")
            if delta_score != measured["delta_score"] or delta_bytes != measured["delta_bytes"]:
                raise PredictProjectReceiverError("marginal curve deltas must exactly match measured points")
            seen_curve_ids.add(point_id)
            curve_keys.append((delta_bytes, point_id))
            normalized_curve_points.append(
                {"point_id": point_id, "delta_score": delta_score, "delta_bytes": delta_bytes}
            )
        if curve_keys != sorted(curve_keys):
            raise PredictProjectReceiverError("marginal curve points must use delta-bytes/point-ID order")
        normalized_curves.append({"stream": expected_stream, "points": normalized_curve_points})

    allocation = value["global_allocation"]
    if not isinstance(allocation, dict) or set(allocation) != {
        "selected_point_id",
        "order",
        "admitted_streams",
        "admitted_flip_ids",
    }:
        raise PredictProjectReceiverError("global allocation fields mismatch")
    selected_point_id = _require_identity(allocation["selected_point_id"], "selected allocation point")
    if selected_point_id not in point_by_id:
        raise PredictProjectReceiverError("global allocation references an unknown point")
    selected = point_by_id[selected_point_id]
    best = min(normalized_points, key=lambda point: (point["lagrangian_delta"], point["point_id"]))
    if selected_point_id != best["point_id"]:
        raise PredictProjectReceiverError("selected allocation is not the Lagrangian minimum at lambda*")
    if allocation["order"] != selected["ordered_streams"]:
        raise PredictProjectReceiverError("global allocation order does not match the selected point")
    canonical_admitted = [stream for stream in GLOBAL_WATERFILL_STREAMS if stream in selected["ordered_streams"]]
    if allocation["admitted_streams"] != canonical_admitted:
        raise PredictProjectReceiverError("global allocation admitted streams are inconsistent")
    if allocation["admitted_flip_ids"] != selected["credited_flip_ids"]:
        raise PredictProjectReceiverError("global allocation admitted flip IDs are inconsistent")
    fixed_point = sellback["fixed_point"]
    if allocation["admitted_flip_ids"] != fixed_point["kept_flip_ids"]:
        raise PredictProjectReceiverError("global admitted flips disagree with the per-flip fixed point")
    if not set(learned_tail_race["admitted_streams"]).issubset(allocation["admitted_streams"]):
        raise PredictProjectReceiverError("learned-tail generator winner is not in the global admitted streams")
    flip_by_id = {flip["flip_id"]: flip for flip in sellback["flips"]}
    fixed_eaten_ids = set(fixed_point["eaten_flip_ids"])
    for race in learned_tail_race["stream_races"]:
        eaten_option = next(
            alternative for alternative in race["alternatives"] if alternative["option"] == "eaten_flip"
        )
        eaten_breakdown = eaten_option["breakdown"]
        eaten_ids = eaten_breakdown["flip_ids"]
        if not set(eaten_ids).issubset(fixed_eaten_ids):
            raise PredictProjectReceiverError("learned-tail eaten alternative disagrees with the fixed point")
        expected_pixels = sum(flip_by_id[flip_id]["positive_pixel_count"] for flip_id in eaten_ids)
        if eaten_breakdown["positive_pixel_count"] != expected_pixels:
            raise PredictProjectReceiverError("learned-tail eaten pixels disagree with the per-flip ledger")

    eaten = value["eaten_flip_decomposition"]
    if not isinstance(eaten, dict) or set(eaten) != {
        "eaten_flip_ids",
        "flip_count",
        "coded_bits",
        "coded_bytes",
        "by_stratum",
        "total_d_seg_cost",
    }:
        raise PredictProjectReceiverError("eaten-flip decomposition fields mismatch")
    strata = eaten["by_stratum"]
    if not isinstance(strata, list) or len(strata) != len(GLOBAL_EATEN_FLIP_STRATA):
        raise PredictProjectReceiverError("eaten-flip decomposition must cover all canonical strata")
    eaten_union: set[str] = set()
    d_seg_costs: list[float] = []
    normalized_strata: list[dict[str, Any]] = []
    eaten_bits_total = 0
    eaten_pixels_total = 0
    per_flip_by_id = {flip["flip_id"]: flip for flip in sellback["flips"]}
    for expected_stratum, stratum_row in zip(GLOBAL_EATEN_FLIP_STRATA, strata, strict=True):
        if not isinstance(stratum_row, dict) or set(stratum_row) != {
            "stratum",
            "flip_ids",
            "flip_count",
            "coded_bits",
            "coded_bytes",
            "d_seg_cost",
        }:
            raise PredictProjectReceiverError("eaten-flip stratum fields mismatch")
        if stratum_row["stratum"] != expected_stratum:
            raise PredictProjectReceiverError("eaten-flip strata are missing or noncanonical")
        flip_ids = stratum_row["flip_ids"]
        if not isinstance(flip_ids, list) or flip_ids != sorted(set(flip_ids)):
            raise PredictProjectReceiverError("eaten flip IDs must be sorted and unique per stratum")
        for flip_id in flip_ids:
            normalized_id = _require_identity(flip_id, "eaten flip ID")
            if normalized_id in eaten_union:
                raise PredictProjectReceiverError("eaten flip IDs must occur in exactly one stratum")
            eaten_union.add(normalized_id)
        if any(
            flip_id not in per_flip_by_id or per_flip_by_id[flip_id]["stratum"] != expected_stratum
            for flip_id in flip_ids
        ):
            raise PredictProjectReceiverError("global eaten flip is unknown or assigned to the wrong stratum")
        expected_rows = [per_flip_by_id[flip_id] for flip_id in flip_ids]
        coded_bits = sum(flip["coded_bits"] for flip in expected_rows)
        positive_pixels = sum(flip["positive_pixel_count"] for flip in expected_rows)
        d_seg_cost = positive_pixels / M1_SCORE_DENOMINATOR
        expected_summary = {
            "stratum": expected_stratum,
            "flip_ids": flip_ids,
            "flip_count": len(flip_ids),
            "coded_bits": coded_bits,
            "coded_bytes": coded_bits / 8,
            "d_seg_cost": d_seg_cost,
        }
        if stratum_row != expected_summary:
            raise PredictProjectReceiverError("global eaten per-stratum counts/bytes/dseg are inconsistent")
        d_seg_costs.append(d_seg_cost)
        eaten_bits_total += coded_bits
        eaten_pixels_total += positive_pixels
        normalized_strata.append(expected_summary)
    if eaten["eaten_flip_ids"] != sorted(eaten_union) or eaten["eaten_flip_ids"] != fixed_point["eaten_flip_ids"]:
        raise PredictProjectReceiverError("eaten flip set must equal its stratum decomposition")
    total_d_seg_cost = _finite_number(eaten["total_d_seg_cost"], "total eaten flip d_seg cost")
    if (
        eaten["flip_count"] != len(eaten_union)
        or isinstance(eaten["flip_count"], bool)
        or eaten["coded_bits"] != eaten_bits_total
        or isinstance(eaten["coded_bits"], bool)
        or eaten["coded_bytes"] != eaten_bits_total / 8
        or total_d_seg_cost != eaten_pixels_total / M1_SCORE_DENOMINATOR
        or total_d_seg_cost != math.fsum(d_seg_costs)
    ):
        raise PredictProjectReceiverError("global eaten totals for counts/bytes/dseg are inconsistent")
    if set(allocation["admitted_flip_ids"]) & eaten_union:
        raise PredictProjectReceiverError("admitted and eaten flip IDs must be disjoint")

    matrix = value["pairwise_interaction_matrix"]
    size = len(GLOBAL_WATERFILL_STREAMS)
    if not isinstance(matrix, list) or len(matrix) != size:
        raise PredictProjectReceiverError("pairwise interaction matrix has wrong geometry")
    normalized_matrix: list[list[float]] = []
    for row_index, matrix_row in enumerate(matrix):
        if not isinstance(matrix_row, list) or len(matrix_row) != size:
            raise PredictProjectReceiverError("pairwise interaction matrix has wrong geometry")
        normalized_row = [_finite_number(item, "pairwise interaction") for item in matrix_row]
        if normalized_row[row_index] != 0.0:
            raise PredictProjectReceiverError("pairwise interaction diagonal must be zero")
        normalized_matrix.append(normalized_row)
    for row_index in range(size):
        for column_index in range(size):
            if normalized_matrix[row_index][column_index] != normalized_matrix[column_index][row_index]:
                raise PredictProjectReceiverError("pairwise interaction matrix must be symmetric")
    return {
        **value,
        "m1_anchors": m1_anchors,
        "custody": custody,
        "joint_decode": decode,
        "points": normalized_points,
        "per_stream_marginal_curves": normalized_curves,
        "per_flip_sellback": sellback,
        "action_level_ladder": action_ladder,
        "attribution_edit_telemetry": attribution_telemetry,
        "learned_tail_race": learned_tail_race,
        "pose_tube_knee": pose_knee,
        "eaten_flip_decomposition": {
            "eaten_flip_ids": eaten["eaten_flip_ids"],
            "flip_count": eaten["flip_count"],
            "coded_bits": eaten["coded_bits"],
            "coded_bytes": eaten["coded_bytes"],
            "by_stratum": normalized_strata,
            "total_d_seg_cost": total_d_seg_cost,
        },
        "pairwise_interaction_matrix": normalized_matrix,
    }


def global_joint_waterfill(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return fail-closed B4 evidence or the best measured joint composition."""

    if evidence is None or evidence == {}:
        return {
            "schema": "predict_project_b4_global_joint_waterfill.v0",
            "status": "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP",
            "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
            "streams": list(GLOBAL_WATERFILL_STREAMS),
            "per_flip_sellback_status": "INCONCLUSIVE_NO_MEASURED_PER_FLIP_SELLBACK",
            "action_level_ladder_status": "INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER",
            "attribution_edit_telemetry_status": "INCONCLUSIVE_NO_MEASURED_ATTRIBUTION_EDIT_TELEMETRY",
            "learned_tail_race_status": "INCONCLUSIVE_NO_MEASURED_LEARNED_TAIL_THREE_WAY_RACE",
            "pose_tube_knee_status": "INCONCLUSIVE_NO_MEASURED_POSE_TUBE_KNEE",
            "boundary_inverse_policy": dict(BOUNDARY_INVERSE_ACTION_POLICY),
            "independent_curve_authority": False,
            "score_claim": False,
            "promotion_eligible": False,
        }
    validated = validate_global_joint_waterfill_evidence(dict(evidence))
    best = next(
        point
        for point in validated["points"]
        if point["point_id"] == validated["global_allocation"]["selected_point_id"]
    )
    return {
        "schema": "predict_project_b4_global_joint_waterfill.v0",
        "status": "MEASURED_GLOBAL_JOINT_SWEEP",
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "streams": list(GLOBAL_WATERFILL_STREAMS),
        "selected_point_id": best["point_id"],
        "selected_lagrangian_delta": best["lagrangian_delta"],
        "evidence_sha256": hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
        "custody_sha256": hard_oracle_custody_sha256(validated["custody"]),
        "pairwise_interaction_matrix": validated["pairwise_interaction_matrix"],
        "global_allocation": validated["global_allocation"],
        "eaten_flip_decomposition": validated["eaten_flip_decomposition"],
        "per_flip_sellback_status": validated["per_flip_sellback"]["measurement_status"],
        "action_level_ladder_status": validated["action_level_ladder"]["measurement_status"],
        "attribution_edit_telemetry_status": validated["attribution_edit_telemetry"]["measurement_status"],
        "learned_tail_race_status": validated["learned_tail_race"]["measurement_status"],
        "pose_tube_knee_status": validated["pose_tube_knee"]["measurement_status"],
        "boundary_inverse_policy": validated["action_level_ladder"]["boundary_inverse_policy"],
        "per_flip_nonmonotone_context_observed": validated["per_flip_sellback"]["nonmonotone_context_observed"],
        "independent_curve_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def measured_waterfill_adapter(
    seg_curve: Sequence[Mapping[str, float]], pose_curve: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    """Retained only as an explicit non-authoritative compatibility refusal."""

    del seg_curve, pose_curve
    return {
        "status": "INCONCLUSIVE_DEPRECATED_INDEPENDENT_CURVES",
        "authority": False,
        "required_api": GLOBAL_WATERFILL_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
    }


def factor2_project_with_existing_solver(
    source_numerators: np.ndarray,
    common_denominator: int,
    rgb_band: np.ndarray,
    *,
    predictor: np.ndarray,
    kernel: FullResizeKernel | None = None,
    max_nodes_per_block: int = 4096,
) -> dict[str, Any]:
    """Compose #549 interval solve with #580 full-kernel metadata.

    ``ker(A)`` is never serialized here.  The optional kernel must own the
    exact operator supplied to the interval solver.
    """

    full_kernel = FullResizeKernel.build() if kernel is None else kernel
    result = solve_interval_frame(
        full_kernel.operator,
        source_numerators,
        common_denominator,
        rgb_band,
        predictor=predictor,
        max_nodes_per_block=max_nodes_per_block,
    )
    return {
        "frame": result.frame,
        "chosen_numerators": result.chosen_numerators,
        "binding_map": result.binding_map,
        "telemetry": result.telemetry,
        "composition": receiver_composition_metadata(full_kernel),
    }


def camera_uint8_identity_sha256(frame: np.ndarray) -> str:
    """Hash camera RGB bytes in the sealed cross-host order."""

    value = np.asarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise PredictProjectReceiverError("camera identity requires HxWx3 uint8")
    header = struct.pack(">II", value.shape[0], value.shape[1]) + b"C_order_y_x_channel_u8.v1\x00"
    return hashlib.sha256(header + np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


PROJECTED_RGB_PLANE_CUSTODY_SCHEMA: Final = "predict_project_rgb_plane_custody.v0"
PROJECTED_RGB_SOURCE_KINDS: Final = frozenset({"decoder_derived_from_seed", "encoder_supplied_counted"})


def _array_identity_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def validate_projected_rgb_plane_custody(
    value: Any,
    *,
    projected_rgb_plane: np.ndarray,
    projected_cells: np.ndarray,
) -> dict[str, Any]:
    """Validate the exact handoff from cell projection to RGB realization.

    A projected class-ID field is not an RGB scorer plane.  This contract keeps
    that distinction structural: the caller must bind both arrays by hash and
    say whether the RGB values are decoder-derived or counted encoder input.
    """

    expected = {
        "schema",
        "source_kind",
        "generator_id",
        "seed_sha256",
        "projected_rgb_sha256",
        "projected_cells_sha256",
        "additional_seed_bytes",
        "decoder_scorer_invocations",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PredictProjectReceiverError("projected RGB-plane custody fields mismatch")
    if value["schema"] != PROJECTED_RGB_PLANE_CUSTODY_SCHEMA:
        raise PredictProjectReceiverError("projected RGB-plane custody schema mismatch")
    if value["source_kind"] not in PROJECTED_RGB_SOURCE_KINDS:
        raise PredictProjectReceiverError("projected RGB-plane source kind is unsupported")
    _require_identity(value["generator_id"], "projected RGB generator_id")
    _require_sha256(value["seed_sha256"], "projected RGB seed_sha256")
    _require_sha256(value["projected_rgb_sha256"], "projected RGB content hash")
    _require_sha256(value["projected_cells_sha256"], "projected cell content hash")
    additional_seed_bytes = value["additional_seed_bytes"]
    if (
        isinstance(additional_seed_bytes, bool)
        or not isinstance(additional_seed_bytes, int)
        or additional_seed_bytes < 0
    ):
        raise PredictProjectReceiverError("additional projected-RGB seed bytes must be exact and nonnegative")
    if value["source_kind"] == "encoder_supplied_counted" and additional_seed_bytes == 0:
        raise PredictProjectReceiverError("encoder-supplied projected RGB cannot claim zero added bytes")
    decoder_scorer_invocations = value["decoder_scorer_invocations"]
    if (
        isinstance(decoder_scorer_invocations, bool)
        or not isinstance(decoder_scorer_invocations, int)
        or decoder_scorer_invocations != 0
    ):
        raise PredictProjectReceiverError("decoder projected-RGB construction cannot invoke a scorer")
    if value["projected_rgb_sha256"] != _array_identity_sha256(projected_rgb_plane):
        raise PredictProjectReceiverError("projected RGB-plane content hash mismatch")
    if value["projected_cells_sha256"] != _array_identity_sha256(projected_cells):
        raise PredictProjectReceiverError("projected cell-field content hash mismatch")
    return dict(value)


def projected_plane_array_sha256(value: np.ndarray) -> str:
    """Public fixed-order hash for a projected cell or RGB plane."""

    array = np.asarray(value)
    if array.dtype.kind == "O" or array.ndim not in (2, 3) or array.size == 0:
        raise PredictProjectReceiverError("projected-plane identity requires a nonempty 2D or 3D array")
    return _array_identity_sha256(array)


def realize_projected_rgb_plane_camera_uint8(
    projected_rgb_plane: np.ndarray,
    projected_cells: np.ndarray,
    projection_custody: Mapping[str, Any],
    *,
    kernel: FullResizeKernel | None = None,
) -> dict[str, Any]:
    """Realize an already-derived RGB scorer plane on the exact uint8 lattice.

    This is the composed G2 decoder stage.  It is intentionally unable to turn
    class IDs into RGB: that missing map is a realization input, not something
    the factor-2 integer solver may invent.  Hard SegNet/PoseNet verification is
    an encode-side caller responsibility and is not claimed by this function.
    """

    full_kernel = FullResizeKernel.build() if kernel is None else kernel
    rgb = np.asarray(projected_rgb_plane)
    cells = np.asarray(projected_cells)
    expected_rgb_shape = (full_kernel.scorer_h, full_kernel.scorer_w, 3)
    expected_cell_shape = (full_kernel.scorer_h, full_kernel.scorer_w)
    if rgb.dtype != np.uint8 or rgb.shape != expected_rgb_shape:
        raise PredictProjectReceiverError(
            f"projected RGB plane must be uint8 with shape {expected_rgb_shape}; "
            "a 2D class-ID field is not a realizable RGB plane"
        )
    if cells.dtype != np.uint8 or cells.shape != expected_cell_shape:
        raise PredictProjectReceiverError(f"projected cells must be uint8 with shape {expected_cell_shape}")
    if np.any(cells >= CLASS_COUNT):
        raise PredictProjectReceiverError("projected cells contain an out-of-range class ID")
    custody = validate_projected_rgb_plane_custody(
        projection_custody,
        projected_rgb_plane=rgb,
        projected_cells=cells,
    )
    frame = realize_factor2_uint8_scorer_plane(full_kernel.operator, rgb)
    verification = verify_factor2_uint8_scorer_plane(full_kernel.operator, frame, rgb)
    if not verification.certified_exact or not verification.numerator_exact:
        raise PredictProjectReceiverError("projected RGB-plane lattice realization failed exact verification")
    return {
        "schema": "predict_project_rgb_lattice_realization.v0",
        "frame": frame,
        "camera_shape": list(frame.shape),
        "camera_uint8_sha256": camera_uint8_identity_sha256(frame),
        "projected_rgb_sha256": custody["projected_rgb_sha256"],
        "projected_cells_sha256": custody["projected_cells_sha256"],
        "projection_custody": custody,
        "factor2_verification": {
            "scorer_values": verification.scorer_values,
            "owned_camera_values": verification.owned_camera_values,
            "unowned_camera_values": verification.unowned_camera_values,
            "numerator_equal_values": verification.numerator_equal_values,
            "canonical_equal_values": verification.canonical_equal_values,
            "denominator": verification.denominator,
            "numerator_exact": verification.numerator_exact,
            "certified_exact": verification.certified_exact,
        },
        "integer_parseback_exact": True,
        "additional_seed_bytes": custody["additional_seed_bytes"],
        "decoder_scorer_invocations": 0,
        "hard_argmax_status": "OWED_ENCODE_SIDE_HARD_ORACLE",
        "pose_status": "OWED_ENCODE_SIDE_HARD_ORACLE",
        "full_kernel_callable": True,
        "full_kernel_serialized": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def realize_inverse_r_camera_uint8(
    source_numerators: np.ndarray,
    common_denominator: int,
    rgb_band: np.ndarray,
    *,
    predictor: np.ndarray,
    kernel: FullResizeKernel | None = None,
    max_nodes_per_block: int = 4096,
) -> dict[str, Any]:
    """Run the existing exact inverse-R solve and seal camera uint8 identity."""

    full_kernel = FullResizeKernel.build() if kernel is None else kernel
    result = factor2_project_with_existing_solver(
        source_numerators,
        common_denominator,
        rgb_band,
        predictor=predictor,
        kernel=full_kernel,
        max_nodes_per_block=max_nodes_per_block,
    )
    frame = np.asarray(result["frame"])
    expected_shape = (full_kernel.camera_h, full_kernel.camera_w, 3)
    if frame.dtype != np.uint8 or frame.shape != expected_shape:
        raise PredictProjectReceiverError("inverse-R realization did not produce camera-resolution RGB uint8")
    realized_numerators, realized_denominator = full_kernel.operator.apply_numerators(frame)
    if realized_denominator != common_denominator or not np.array_equal(
        realized_numerators, result["chosen_numerators"]
    ):
        raise PredictProjectReceiverError("inverse-R integer realization failed exact parse-back")
    return {
        **result,
        "schema": "predict_project_inverse_r_camera_uint8.v0",
        "camera_shape": list(frame.shape),
        "camera_uint8_sha256": camera_uint8_identity_sha256(frame),
        "cross_host_order": "C_order_y_x_channel_u8.v1",
        "integer_parseback_exact": True,
        "full_kernel_callable": True,
        "full_kernel_serialized": False,
        "decoder_scorer_invocations": 0,
        "score_claim": False,
        "promotion_eligible": False,
    }


def receiver_composition_metadata(kernel: FullResizeKernel | None = None) -> dict[str, Any]:
    full_kernel = FullResizeKernel.build() if kernel is None else kernel
    return {
        "schema": "predict_project_receiver_composition.v0",
        "r1b4_receiver_schema": R1B4_RECEIVER_SCHEMA,
        "v10_receiver_contract_id": V10_RECEIVER_CONTRACT_ID,
        "tie_policy_id": V10_TIE_POLICY_ID,
        "arithmetic_id": V10_ARITHMETIC_ID,
        "interval_solver": "tac.optimization.joint_seg_pose_rate.solve_interval_frame",
        "full_kernel_schema": FULL_RESIZE_KERNEL_SCHEMA,
        "full_kernel_nullity_per_channel": full_kernel.coverage().full_nullity,
        "support_fill_hook": "tac.optimization.resize_null_preimage.apply_tier1_zero_weight_fill",
        "plane_cache_convention": "sha256(seed_bytes,pair_index,frame_index,projection_policy)",
        "worker_env": "INFLATE_WORKERS",
        "phase_carrier_hook": "existing phase/jitter carrier callback; required when declared",
        "kernel_serialized": False,
        "full_kernel_callable": True,
        "inverse_r_output": "camera_resolution_rgb_uint8",
        "cross_host_identity": "C_order_y_x_channel_u8.v1",
        "frame_roles": {"frame0": "pose_only", "frame1": "seg_and_pose"},
        "section_container": "predict_project_named_sections.v1",
        "receiver_search_invocations": 0,
        "score_claim": False,
        "promotion_eligible": False,
    }


def plane_cache_key(seed_bytes: bytes, pair_index: int, frame_index: int, projection_policy: str) -> str:
    if not isinstance(seed_bytes, bytes) or isinstance(pair_index, bool) or not 0 <= pair_index < 600:
        raise PredictProjectReceiverError("invalid plane-cache custody")
    if frame_index not in (0, 1) or not isinstance(projection_policy, str) or not projection_policy:
        raise PredictProjectReceiverError("invalid plane-cache frame/policy")
    digest = hashlib.sha256()
    for value in (
        seed_bytes,
        pair_index.to_bytes(2, "big"),
        frame_index.to_bytes(1, "big"),
        projection_policy.encode("ascii"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _decode_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise PredictProjectReceiverError("decode output cannot contain object dtype")
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode("ascii")
    return header + np.ascontiguousarray(array).tobytes()


def double_decode_hash(decode: Callable[[], Any]) -> DoubleDecodeResult:
    first = _decode_bytes(decode())
    second = _decode_bytes(decode())
    first_hash = hashlib.sha256(first).hexdigest()
    second_hash = hashlib.sha256(second).hexdigest()
    return DoubleDecodeResult(first_hash, second_hash, first == second and first_hash == second_hash)


def component_byte_accounting(seed: Mapping[str, Any]) -> dict[str, int]:
    """Canonical raw component bytes for the B5 decomposition."""

    from tac.optimization.predict_project_schema import canonical_json_bytes

    validated = validate_constraint_seed(seed)
    chart = validated["ground_chart"]
    jitter = validated["boundary_jitter"]
    components = {
        "chart": {key: value for key, value in chart.items() if key != "cells"},
        "sites": chart["cells"],
        "trajectory": validated["trajectory"],
        "bulk": {
            "schema": validated["schema"],
            "container": validated["container"],
            "grammar": validated["grammar"],
            "units": validated["units"],
            "receiver": validated["receiver"],
            "authority": validated["authority"],
        },
        "jitter": jitter["r0"],
        "response": {"selected_rung": jitter["selected_rung"], "r1": jitter["r1"], "r2": jitter["r2"]},
        "tracks": validated["movable_tracks"],
        "events": validated["events"],
        "pose_tightening": validated["pose_tightening"],
        "eat_flip": [],
        "constraints": validated["constraint_seeds"],
    }
    sizes = {name: len(canonical_json_bytes(value)) for name, value in components.items()}
    wire_bytes = len(serialize_constraint_seed(validated))
    sizes["container_and_section_overhead"] = wire_bytes - sum(sizes.values())
    if sizes["container_and_section_overhead"] < 0:
        raise PredictProjectReceiverError("component accounting overlaps named sections")
    sizes["total_components"] = wire_bytes
    return sizes


__all__ = [
    "ACTION_LEVEL_LADDER_SCHEMA",
    "ACTION_LEVEL_RUNGS",
    "ADVECTED_MOTION_BASE_SCHEMA",
    "ADVECTED_MOTION_LAWREFS",
    "ATTRIBUTION_EDIT_TELEMETRY_SCHEMA",
    "ATTRIBUTION_REUSE_BINDINGS",
    "BOUNDARY_INVERSE_ACTION_POLICY",
    "BOUNDARY_INVERSE_RECEIPT_COMMIT",
    "BOUNDARY_INVERSE_RECEIPT_PATH",
    "BOUNDARY_INVERSE_RECEIPT_SHA256",
    "CANONICAL_LAW_RESOLUTION_CUSTODY",
    "CANONICAL_LAW_RESOLUTION_SHA256",
    "CHART_RGB_COEFFICIENT_SCHEMA",
    "COUNTED_FULL_SCREW_XI_SCHEMA",
    "COUNTED_PLANAR_XI_SCHEMA",
    "GLOBAL_WATERFILL_LAMBDA_STAR",
    "GLOBAL_WATERFILL_STREAMS",
    "M1_ANCHORS",
    "M1_FLIP_COUNT",
    "M1_RECEIPT_COMMIT",
    "M1_RECEIPT_PATH",
    "M1_RECEIPT_SHA256",
    "M1_SCORE_DENOMINATOR",
    "PROJECTED_RGB_PLANE_CUSTODY_SCHEMA",
    "PROJECTED_RGB_SOURCE_KINDS",
    "REALIZATION_BREAKEVEN_EQUATION_ID",
    "S3_TRAINER_REUSE",
    "SEGNET_CENTERED_HEAD_RANK",
    "SEGNET_HEAD_RANK_EQUATION_ID",
    "TEMPORAL_JITTER_AMORTIZATION_RATIO",
    "TEMPORAL_JITTER_EQUATION_ID",
    "ChartRGBCoefficientPacket",
    "DoubleDecodeResult",
    "LinearConstraint",
    "PredictProjectReceiverError",
    "ProjectionResult",
    "advect_motion_base",
    "apply_chart_rgb_coefficients",
    "camera_uint8_identity_sha256",
    "component_byte_accounting",
    "counted_full_screw_xi_series",
    "counted_planar_xi_series",
    "decode_chart_rgb_coefficients",
    "derive_ground_chart_raster",
    "double_decode_hash",
    "encode_chart_rgb_coefficients",
    "extract_constraint_violations",
    "factor2_project_with_existing_solver",
    "fit_chart_rgb_coefficients",
    "global_joint_waterfill",
    "hard_oracle_custody_sha256",
    "max_linear_violation",
    "measured_waterfill_adapter",
    "plane_cache_key",
    "pose_tightening_for",
    "predict_cell_field",
    "project_box",
    "project_halfspace",
    "project_linear_intersection",
    "projected_plane_array_sha256",
    "quantize_uint8_feasible",
    "realize_inverse_r_camera_uint8",
    "realize_projected_rgb_plane_camera_uint8",
    "receiver_composition_metadata",
    "stratify_predictor_quality",
    "trajectory_at",
    "validate_action_level_ladder_evidence",
    "validate_attribution_edit_telemetry",
    "validate_flip_attribution_receipt",
    "validate_global_joint_waterfill_evidence",
    "validate_hard_oracle_custody",
    "validate_ladder_edit_request",
    "validate_ladder_edit_response",
    "validate_learned_tail_race_evidence",
    "validate_m1_anchor_binding",
    "validate_per_flip_sellback_evidence",
    "validate_pose_tube_knee_evidence",
    "validate_projected_rgb_plane_custody",
    "verify_pose_tightening_choice",
]
