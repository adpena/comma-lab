# SPDX-License-Identifier: MIT
"""Fresh n600 pose-trajectory compiler/selector for the exact G88 successor.

G91 is deliberately a finite-treatment compiler, not an unconstrained inverse
solver.  It derives a new SE(3) ``xi`` trajectory from the currently sealed
source PoseNet targets, optionally projects the centered trajectory onto a
low-dimensional population factorization, quantizes it once, and races the
existing exact-EOF XIP2 coders.  Every coder must decode to the same int16
table.  The selected XIP2 is then bound to the exact G88 base member and priced
through both outer ZIP encodings.

The scorer-native selector operates on *measured per-pair* frozen-PoseNet
losses for two actual decoded alternatives:

* PASS: incoming G88/G85 P0 with exact corrected Y1;
* XIP2: ``warp(exact corrected Y1, xi[p])`` with exact corrected Y1.

Because PoseNet evaluates each pair independently, a population mode vector is
exactly priced by selecting the corresponding measured loss for every pair.
The nonlinear population objective is applied only after taking the mean:

``sqrt(10 * mean(d_pose[p])) + 25 * exact_outer_bytes / 37_545_489``.

This permits an exact sparse-exception search in either G88 default direction
without a proxy loss.  Geometry, SVD, and coder sizes are proposal/factorability
surfaces only; they never substitute for the frozen-scorer measurements.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import xi_from_pose_calibration
from tac.boundary_math.xi_pose_coder import (
    dequantize_xi,
    parse_xi_payload,
    quantize_xi,
    serialize_xi_payload,
)
from tac.witness_dsl import taskspace_g88_population_conditional_y0_pvsa_v1 as g88

PAIR_COUNT: Final = 600
POSE_DIM: Final = 6
RATE_DENOMINATOR_BYTES: Final = 37_545_489
CODER_UNIVERSE: Final = (
    "none",
    "delta_ar",
    "spline_residual",
    "delta_res",
)
COMPILER_ID: Final = "tac.g91.fresh_n600_se3_xip2_finite_treatment_compiler_selector.v1"
TRAJECTORY_POLICY_ID: Final = "SOURCE_GT_POSE_GLOBAL_CALIBRATION_CENTERED_SVD_OPTIONAL_THEN_EXACT_XIP2_V1"
SELECTOR_POLICY_ID: Final = "ACTUAL_FROZEN_POSENET_PAIR_LOSSES_NONLINEAR_POPULATION_OBJECTIVE_EXACT_OUTER_BYTES_V1"
AUTHORITY_BLOCKER: Final = "G91_PUBLIC_RUNTIME_AND_UPSTREAM_EXACT_N600_EVAL_OWED"
FRESH_TARGET_CUSTODY_QUALIFIER: Final = (
    "SOURCE_GT_SCORER_TARGET_CUSTODY_LOCAL_CPU_RESEARCH_SIGNAL_NOT_PROMOTION_AUTHORITY"
)
G16_SETTLED_NEGATIVE_AUTHORITY: Final = (
    "SPEC_G16_QUANTIZED_XIP2_INVERSE_SOLVER_20260726_AFFINE_XI_TO_POSENET_R2_NEGATIVE_0P215"
)
INVERSE_CONTROL_SOLVED: Final = False
DIRECT_POSE_TARGET_AS_WARP_CONTROL_ADMISSIBLE: Final = False
ONLY_ADMISSIBLE_PROMOTION_PATH: Final = (
    "G95_POSENET_IN_LOOP_INVERSE_SOLVE_OVER_DECODER_CONTROLS_THEN_FACTOR_AND_QUANTIZE"
)


class G91PoseTrajectoryError(ValueError):
    """A G91 typed treatment, exact byte, or measured-selection contract failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G91PoseTrajectoryError(f"{label} must be canonical lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise G91PoseTrajectoryError(f"{label} must be one finite real scalar")
    return float(value)


def _immutable(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != dtype:
        raise G91PoseTrajectoryError(f"array dtype must be exactly {dtype}")
    result = np.ascontiguousarray(raw).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class G91TrajectoryTreatmentV1:
    """One preregistered geometry/factorization/quantization treatment."""

    treatment_id: str
    s_t: float
    s_r: float
    pitch: float
    centered_rank: int
    q_levels: int

    def __post_init__(self) -> None:
        if (
            type(self.treatment_id) is not str
            or not self.treatment_id
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in self.treatment_id)
        ):
            raise G91PoseTrajectoryError("treatment_id must be nonempty lowercase slug text")
        for label in ("s_t", "s_r", "pitch"):
            value = _finite(getattr(self, label), label)
            canonical = float(np.float32(value))
            if not math.isfinite(canonical):
                raise G91PoseTrajectoryError(f"{label} escaped finite fp32")
            object.__setattr__(self, label, canonical)
        if type(self.centered_rank) is not int or not 1 <= self.centered_rank <= POSE_DIM:
            raise G91PoseTrajectoryError("centered_rank must be an exact integer in [1,6]")
        if type(self.q_levels) is not int or not 1 <= self.q_levels <= 32767:
            raise G91PoseTrajectoryError("q_levels must be an exact integer in [1,32767]")


@dataclass(frozen=True, slots=True)
class G91TrajectoryV1:
    """One freshly derived float trajectory and its expanded exact XIP2 grid."""

    treatment: G91TrajectoryTreatmentV1
    calibrated_xi: np.ndarray
    factorized_xi: np.ndarray
    q_codes: np.ndarray
    scales: np.ndarray
    factorability: dict[str, Any]
    source_target_sha256: str

    def __post_init__(self) -> None:
        if type(self.treatment) is not G91TrajectoryTreatmentV1:
            raise G91PoseTrajectoryError("trajectory lost typed treatment")
        for value, dtype, label in (
            (self.calibrated_xi, np.dtype(np.float64), "calibrated_xi"),
            (self.factorized_xi, np.dtype(np.float64), "factorized_xi"),
            (self.q_codes, np.dtype(np.int16), "q_codes"),
            (self.scales, np.dtype(np.float32), "scales"),
        ):
            raw = np.asarray(value)
            expected = (PAIR_COUNT, POSE_DIM) if label != "scales" else (POSE_DIM,)
            if raw.dtype != dtype or raw.shape != expected or not np.all(np.isfinite(raw)):
                raise G91PoseTrajectoryError(f"{label} changed exact finite dtype/shape ABI")
        _require_sha256(self.source_target_sha256, "source_target_sha256")
        if not isinstance(self.factorability, dict):
            raise G91PoseTrajectoryError("factorability must be one concrete mapping")
        object.__setattr__(
            self,
            "calibrated_xi",
            _immutable(self.calibrated_xi, np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "factorized_xi",
            _immutable(self.factorized_xi, np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "q_codes",
            _immutable(self.q_codes, np.dtype(np.int16)),
        )
        object.__setattr__(
            self,
            "scales",
            _immutable(self.scales, np.dtype(np.float32)),
        )

    @property
    def decoded_xi(self) -> np.ndarray:
        return dequantize_xi(self.q_codes, self.scales)


@dataclass(frozen=True, slots=True)
class G91CoderRaceRowV1:
    """One exact XIP2 coder and G88/outer byte price for the same q table."""

    coder: str
    xip2_payload: bytes = field(repr=False)
    xip2_sha256: str
    operand_bytes: int
    operand_sha256: str
    successor_member_bytes: int
    successor_member_sha256: str
    stored_outer_bytes: int
    stored_outer_sha256: str
    deflated_outer_bytes: int
    deflated_outer_sha256: str
    selected_outer_bytes: int
    selected_outer_sha256: str
    selected_outer_encoding: str
    parsed_operand: g88.PopulationConditionalOperandV1 = field(repr=False, compare=False)
    archive_build: g88.PopulationConditionalPVSAArchiveBuildV1 = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.coder not in CODER_UNIVERSE:
            raise G91PoseTrajectoryError("coder row escaped the closed XIP2 coder universe")
        if type(self.xip2_payload) is not bytes or not self.xip2_payload.startswith(b"XIP2"):
            raise G91PoseTrajectoryError("coder row lost exact XIP2 payload")
        for label in (
            "xip2_sha256",
            "operand_sha256",
            "successor_member_sha256",
            "stored_outer_sha256",
            "deflated_outer_sha256",
            "selected_outer_sha256",
        ):
            _require_sha256(getattr(self, label), label)
        if _sha256(self.xip2_payload) != self.xip2_sha256:
            raise G91PoseTrajectoryError("coder row XIP2 bytes differ from receipt hash")
        if type(self.parsed_operand) is not g88.PopulationConditionalOperandV1:
            raise G91PoseTrajectoryError("coder row lost strict parsed G88 operand")
        if type(self.archive_build) is not g88.PopulationConditionalPVSAArchiveBuildV1:
            raise G91PoseTrajectoryError("coder row lost exact outer archive build")
        operand_bytes = self.parsed_operand.to_bytes()
        selected_member = self.archive_build.selected
        outer = self.archive_build.outer_build
        exact_custody = {
            "operand_bytes": len(operand_bytes),
            "operand_sha256": _sha256(operand_bytes),
            "successor_member_bytes": len(selected_member.member_bytes),
            "successor_member_sha256": _sha256(selected_member.member_bytes),
            "stored_outer_bytes": outer.stored.archive_nbytes,
            "stored_outer_sha256": outer.stored.archive_sha256,
            "deflated_outer_bytes": outer.deflated.archive_nbytes,
            "deflated_outer_sha256": outer.deflated.archive_sha256,
            "selected_outer_bytes": outer.selected.archive_nbytes,
            "selected_outer_sha256": outer.selected.archive_sha256,
            "selected_outer_encoding": outer.selected.encoding.value,
        }
        if any(getattr(self, label) != expected for label, expected in exact_custody.items()):
            raise G91PoseTrajectoryError("coder row receipt differs from exact parsed archive custody")
        if (
            self.parsed_operand.xip2_payload != self.xip2_payload
            or self.parsed_operand != selected_member.conditional_operand
            or operand_bytes != selected_member.conditional_operand_bytes
            or outer.selected.member_bytes != selected_member.member_bytes
            or outer.stored.member_bytes != selected_member.member_bytes
            or outer.deflated.member_bytes != selected_member.member_bytes
            or _sha256(outer.stored.archive_bytes) != self.stored_outer_sha256
            or _sha256(outer.deflated.archive_bytes) != self.deflated_outer_sha256
            or _sha256(outer.selected.archive_bytes) != self.selected_outer_sha256
        ):
            raise G91PoseTrajectoryError("coder row nested operand/member/outer bytes differ from exact build")

    def receipt(self) -> dict[str, Any]:
        return {
            "coder": self.coder,
            "xip2_bytes": len(self.xip2_payload),
            "xip2_sha256": self.xip2_sha256,
            "operand_bytes": self.operand_bytes,
            "operand_sha256": self.operand_sha256,
            "successor_member_bytes": self.successor_member_bytes,
            "successor_member_sha256": self.successor_member_sha256,
            "outer_store_bytes": self.stored_outer_bytes,
            "outer_store_sha256": self.stored_outer_sha256,
            "outer_deflate_bytes": self.deflated_outer_bytes,
            "outer_deflate_sha256": self.deflated_outer_sha256,
            "selected_outer_bytes": self.selected_outer_bytes,
            "selected_outer_sha256": self.selected_outer_sha256,
            "selected_outer_encoding": self.selected_outer_encoding,
        }


@dataclass(frozen=True, slots=True)
class G91ModeSelectionRowV1:
    """One exact sparse mode allocation priced on measured per-pair losses."""

    default_mode: Literal["PASS_P0", "XIP2_SE3_FRAME0_WARP"]
    xip2_pair_ids: tuple[int, ...]
    pass_pair_ids: tuple[int, ...]
    coder_row: G91CoderRaceRowV1
    mean_d_pose: float
    pose_score_term: float
    rate_score_term: float
    selector_objective: float
    control_rows: int

    def __post_init__(self) -> None:
        if self.default_mode not in {"PASS_P0", "XIP2_SE3_FRAME0_WARP"}:
            raise G91PoseTrajectoryError("selection row default mode escaped closed universe")
        complete = tuple(range(PAIR_COUNT))
        if tuple(sorted((*self.xip2_pair_ids, *self.pass_pair_ids))) != complete or set(self.xip2_pair_ids) & set(
            self.pass_pair_ids
        ):
            raise G91PoseTrajectoryError("selection row modes do not partition exact n600")
        for label in ("mean_d_pose", "pose_score_term", "rate_score_term", "selector_objective"):
            value = _finite(getattr(self, label), label)
            if value < 0.0:
                raise G91PoseTrajectoryError(f"{label} must be nonnegative")
        if self.control_rows != len(self.coder_row.parsed_operand.controls):
            raise G91PoseTrajectoryError("selection control count differs from parsed exact operand")

    def receipt(self) -> dict[str, Any]:
        return {
            "default_mode": self.default_mode,
            "xip2_pair_count": len(self.xip2_pair_ids),
            "xip2_pair_ids": list(self.xip2_pair_ids),
            "pass_pair_count": len(self.pass_pair_ids),
            "pass_pair_ids": list(self.pass_pair_ids),
            "control_rows": self.control_rows,
            "mean_d_pose": self.mean_d_pose,
            "pose_score_term": self.pose_score_term,
            "rate_score_term": self.rate_score_term,
            "selector_objective_without_invariant_seg": self.selector_objective,
            "exact_bytes": self.coder_row.receipt(),
        }


def derive_fresh_trajectory(
    source_gt_poses: np.ndarray,
    treatment: G91TrajectoryTreatmentV1,
) -> G91TrajectoryV1:
    """Derive a fresh n600 trajectory; no historical XIP2 bytes are accepted."""

    poses = np.asarray(source_gt_poses)
    if poses.dtype != np.float64 or poses.shape != (PAIR_COUNT, POSE_DIM) or not np.all(np.isfinite(poses)):
        raise G91PoseTrajectoryError("source_gt_poses must be exact finite float64 [600,6]")
    calibrated = np.stack(
        [
            xi_from_pose_calibration(
                poses[pair_id],
                treatment.s_t,
                treatment.s_r,
                treatment.pitch,
                whole_ground=True,
            )
            for pair_id in range(PAIR_COUNT)
        ],
        axis=0,
    ).astype(np.float64)
    factorized, factorability = centered_rank_projection(
        calibrated,
        rank=treatment.centered_rank,
    )
    q_codes, scales = quantize_xi(factorized, q_levels=treatment.q_levels)
    factorability = {
        **factorability,
        "float64_expanded_bytes": int(calibrated.nbytes),
        "factorized_float64_expanded_bytes": int(factorized.nbytes),
        "int16_expanded_bytes": int(q_codes.nbytes + scales.nbytes),
        "int16_table_sha256": _array_sha256(q_codes),
        "scales_sha256": _array_sha256(scales),
        "zlib9_int16_table_bytes": len(zlib.compress(q_codes.tobytes(), level=9)),
        "quantization_rmse": float(np.sqrt(np.mean((dequantize_xi(q_codes, scales) - factorized) ** 2))),
    }
    return G91TrajectoryV1(
        treatment=treatment,
        calibrated_xi=calibrated,
        factorized_xi=factorized,
        q_codes=q_codes,
        scales=scales,
        factorability=factorability,
        source_target_sha256=_array_sha256(poses),
    )


def centered_rank_projection(
    xi: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the centered population trajectory onto a deterministic SVD rank."""

    values = np.asarray(xi, dtype=np.float64)
    if values.shape != (PAIR_COUNT, POSE_DIM) or not np.all(np.isfinite(values)):
        raise G91PoseTrajectoryError("xi must be exact finite [600,6]")
    if type(rank) is not int or not 1 <= rank <= POSE_DIM:
        raise G91PoseTrajectoryError("rank must be an exact integer in [1,6]")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)

    # This matrix has only six columns.  Explicit rank-one accumulation avoids
    # a macOS Accelerate warning surface seen for exactly rank-deficient inputs
    # while remaining algebraically identical to ``(U*S) @ Vt``.
    def reconstruct(candidate_rank: int) -> np.ndarray:
        result = np.broadcast_to(mean, values.shape).copy()
        for component in range(candidate_rank):
            result += singular[component] * np.outer(u[:, component], vt[component])
        return result

    projected = reconstruct(rank)
    total_energy = float(np.sum(centered * centered))
    rows: list[dict[str, Any]] = []
    for candidate_rank in range(1, POSE_DIM + 1):
        candidate = reconstruct(candidate_rank)
        squared_error = float(np.sum((values - candidate) ** 2))
        rows.append(
            {
                "rank": candidate_rank,
                "relative_frobenius_error": (0.0 if total_energy == 0.0 else math.sqrt(squared_error / total_energy)),
                "explained_energy_fraction": (1.0 if total_energy == 0.0 else 1.0 - squared_error / total_energy),
            }
        )
    return np.ascontiguousarray(projected, dtype=np.float64), {
        "centered_singular_values": singular.tolist(),
        "selected_centered_rank": rank,
        "rank_curve": rows,
        "mean_sha256": _array_sha256(mean),
        "selected_projection_sha256": _array_sha256(projected),
    }


def _controls_for_partition(
    *,
    default_mode: Literal["PASS_P0", "XIP2_SE3_FRAME0_WARP"],
    xip2_pair_ids: tuple[int, ...],
) -> tuple[g88.ConditionalY0ControlV1, ...]:
    active = set(xip2_pair_ids)
    if len(active) != len(xip2_pair_ids) or any(
        type(value) is not int or not 0 <= value < PAIR_COUNT for value in active
    ):
        raise G91PoseTrajectoryError("xip2_pair_ids must be unique exact n600 pair IDs")
    if not active:
        raise G91PoseTrajectoryError("G91 candidate must execute XIP2 on at least one pair")
    if default_mode == "XIP2_SE3_FRAME0_WARP":
        return tuple(
            g88.ConditionalY0ControlV1(
                source_pair_id=pair_id,
                mode=g88.ConditionalY0ModeV1.PASS_P0,
            )
            for pair_id in range(PAIR_COUNT)
            if pair_id not in active
        )
    if default_mode == "PASS_P0":
        return tuple(
            g88.ConditionalY0ControlV1(
                source_pair_id=pair_id,
                mode=g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
            )
            for pair_id in sorted(active)
        )
    raise G91PoseTrajectoryError("default_mode escaped the exact G88 default universe")


def compile_coder_race(
    *,
    trajectory: G91TrajectoryV1,
    base_pvsa_member_bytes: bytes,
    semantic_p_sha256: str,
    default_mode: Literal["PASS_P0", "XIP2_SE3_FRAME0_WARP"],
    xip2_pair_ids: tuple[int, ...],
    coders: tuple[str, ...] = CODER_UNIVERSE,
    maximum_member_bytes: int = 2 << 20,
    maximum_section_bytes: int = 1 << 20,
) -> tuple[G91CoderRaceRowV1, ...]:
    """Compile every requested coder, enforcing equal q/scales and strict G88 parseback."""

    if type(base_pvsa_member_bytes) is not bytes or not base_pvsa_member_bytes:
        raise G91PoseTrajectoryError("base_pvsa_member_bytes must be exact nonempty bytes")
    semantic_sha = _require_sha256(semantic_p_sha256, "semantic_p_sha256")
    if type(coders) is not tuple or not coders or len(set(coders)) != len(coders):
        raise G91PoseTrajectoryError("coders must be one nonempty unique tuple")
    controls = _controls_for_partition(
        default_mode=default_mode,
        xip2_pair_ids=xip2_pair_ids,
    )
    mode = (
        g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP
        if default_mode == "XIP2_SE3_FRAME0_WARP"
        else g88.ConditionalY0ModeV1.PASS_P0
    )
    rows: list[G91CoderRaceRowV1] = []
    for coder in coders:
        if coder not in CODER_UNIVERSE:
            raise G91PoseTrajectoryError(f"unknown XIP2 coder {coder!r}")
        xip2 = serialize_xi_payload(
            trajectory.q_codes,
            trajectory.scales,
            coder=coder,
        )
        parsed_q, parsed_scales = parse_xi_payload(xip2)
        if not np.array_equal(parsed_q, trajectory.q_codes) or not np.array_equal(
            parsed_scales,
            trajectory.scales,
        ):
            raise G91PoseTrajectoryError("XIP2 coder changed the selected exact quantized trajectory")
        operand = g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha256(base_pvsa_member_bytes),
            semantic_p_sha256=semantic_sha,
            controls=controls,
            default_mode=mode,
            xip2_payload=xip2,
            pitch=trajectory.treatment.pitch,
        )
        operand_bytes = operand.to_bytes()
        build = g88.build_population_conditional_pvsa_archive(
            base_pvsa_member_bytes=base_pvsa_member_bytes,
            conditional_operand_bytes=operand_bytes,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
        )
        parsed = build.selected.conditional_operand
        if (
            parsed.to_bytes() != operand_bytes
            or parsed.transport is None
            or not np.array_equal(parsed.transport.q_codes, trajectory.q_codes)
            or not np.array_equal(parsed.transport.scales, trajectory.scales)
        ):
            raise G91PoseTrajectoryError("strict outer parseback changed exact G91 XIP2 custody")
        outer = build.outer_build
        rows.append(
            G91CoderRaceRowV1(
                coder=coder,
                xip2_payload=xip2,
                xip2_sha256=_sha256(xip2),
                operand_bytes=len(operand_bytes),
                operand_sha256=_sha256(operand_bytes),
                successor_member_bytes=len(build.selected.member_bytes),
                successor_member_sha256=build.selected.member_sha256,
                stored_outer_bytes=outer.stored.archive_nbytes,
                stored_outer_sha256=outer.stored.archive_sha256,
                deflated_outer_bytes=outer.deflated.archive_nbytes,
                deflated_outer_sha256=outer.deflated.archive_sha256,
                selected_outer_bytes=outer.selected.archive_nbytes,
                selected_outer_sha256=outer.selected.archive_sha256,
                selected_outer_encoding=outer.selected.encoding.value,
                parsed_operand=parsed,
                archive_build=build,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.selected_outer_bytes,
                len(row.xip2_payload),
                row.coder,
            ),
        )
    )


def sparse_exception_counts(pair_count: int) -> tuple[int, ...]:
    """Enumerate every nonzero prefix count through the supplied population.

    Outer DEFLATE pricing is non-additive and nonmonotone in the control count,
    so no logarithmic K grid is admissible.  This exhausts every K in the
    finite benefit-sorted-prefix family, including prefixes extending beyond
    the strictly positive local-benefit boundary.  It does *not* claim to
    exhaust the combinatorial universe of all subsets at fixed K.
    """

    if type(pair_count) is not int or not 0 <= pair_count <= PAIR_COUNT:
        raise G91PoseTrajectoryError("pair_count must be an exact integer in [0,600]")
    if pair_count == 0:
        return ()
    return tuple(range(1, pair_count + 1))


def compile_measured_mode_selection(
    *,
    trajectory: G91TrajectoryV1,
    base_pvsa_member_bytes: bytes,
    semantic_p_sha256: str,
    pass_d_pose: np.ndarray,
    xip2_d_pose: np.ndarray,
    coders: tuple[str, ...] = CODER_UNIVERSE,
    maximum_member_bytes: int = 2 << 20,
    maximum_section_bytes: int = 1 << 20,
) -> tuple[G91ModeSelectionRowV1, ...]:
    """Price exact sparse mode allocations from actual per-pair scorer losses."""

    pass_loss = np.asarray(pass_d_pose)
    xip2_loss = np.asarray(xip2_d_pose)
    for value, label in ((pass_loss, "pass_d_pose"), (xip2_loss, "xip2_d_pose")):
        if (
            value.dtype != np.float64
            or value.shape != (PAIR_COUNT,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0.0)
        ):
            raise G91PoseTrajectoryError(f"{label} must be exact finite nonnegative float64 [600]")

    rows: list[G91ModeSelectionRowV1] = []
    pair_ids = np.arange(PAIR_COUNT, dtype=np.int64)

    # Global XIP2 plus every stable measured-benefit-ordered PASS prefix.
    # K=600 is represented by the opposite default direction below because an
    # executable G88 operand must retain at least one XIP2 pair.
    pass_gain = xip2_loss - pass_loss
    pass_order = tuple(int(value) for value in np.lexsort((pair_ids, -pass_gain)))
    for count in range(PAIR_COUNT):
        pass_ids = tuple(sorted(pass_order[:count]))
        pass_set = set(pass_ids)
        xip2_ids = tuple(pair_id for pair_id in range(PAIR_COUNT) if pair_id not in pass_set)
        coder_row = compile_coder_race(
            trajectory=trajectory,
            base_pvsa_member_bytes=base_pvsa_member_bytes,
            semantic_p_sha256=semantic_p_sha256,
            default_mode="XIP2_SE3_FRAME0_WARP",
            xip2_pair_ids=xip2_ids,
            coders=coders,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
        )[0]
        chosen = xip2_loss.copy()
        chosen[np.asarray(pass_ids, dtype=np.int64)] = pass_loss[np.asarray(pass_ids, dtype=np.int64)]
        rows.append(
            _selection_row(
                default_mode="XIP2_SE3_FRAME0_WARP",
                xip2_pair_ids=xip2_ids,
                pass_pair_ids=pass_ids,
                coder_row=coder_row,
                chosen_d_pose=chosen,
            )
        )

    # Global PASS plus every stable measured-benefit-ordered XIP2 prefix.
    xip2_gain = pass_loss - xip2_loss
    xip2_order = tuple(int(value) for value in np.lexsort((pair_ids, -xip2_gain)))
    for count in sparse_exception_counts(PAIR_COUNT):
        xip2_ids = tuple(sorted(xip2_order[:count]))
        xip2_set = set(xip2_ids)
        pass_ids = tuple(pair_id for pair_id in range(PAIR_COUNT) if pair_id not in xip2_set)
        coder_row = compile_coder_race(
            trajectory=trajectory,
            base_pvsa_member_bytes=base_pvsa_member_bytes,
            semantic_p_sha256=semantic_p_sha256,
            default_mode="PASS_P0",
            xip2_pair_ids=xip2_ids,
            coders=coders,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
        )[0]
        chosen = pass_loss.copy()
        chosen[np.asarray(xip2_ids, dtype=np.int64)] = xip2_loss[np.asarray(xip2_ids, dtype=np.int64)]
        rows.append(
            _selection_row(
                default_mode="PASS_P0",
                xip2_pair_ids=xip2_ids,
                pass_pair_ids=pass_ids,
                coder_row=coder_row,
                chosen_d_pose=chosen,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.selector_objective,
                row.coder_row.selected_outer_bytes,
                row.control_rows,
                row.default_mode,
            ),
        )
    )


def _selection_row(
    *,
    default_mode: Literal["PASS_P0", "XIP2_SE3_FRAME0_WARP"],
    xip2_pair_ids: tuple[int, ...],
    pass_pair_ids: tuple[int, ...],
    coder_row: G91CoderRaceRowV1,
    chosen_d_pose: np.ndarray,
) -> G91ModeSelectionRowV1:
    mean_d_pose = float(np.mean(chosen_d_pose))
    pose_term, rate_term, objective = exact_prefix_objective(
        mean_d_pose=mean_d_pose,
        exact_outer_bytes=coder_row.selected_outer_bytes,
    )
    return G91ModeSelectionRowV1(
        default_mode=default_mode,
        xip2_pair_ids=xip2_pair_ids,
        pass_pair_ids=pass_pair_ids,
        coder_row=coder_row,
        mean_d_pose=mean_d_pose,
        pose_score_term=pose_term,
        rate_score_term=rate_term,
        selector_objective=objective,
        control_rows=len(coder_row.parsed_operand.controls),
    )


def exact_prefix_objective(
    *,
    mean_d_pose: float,
    exact_outer_bytes: int,
) -> tuple[float, float, float]:
    """Return the scorer-native pose, exact-rate, and combined prefix terms."""

    canonical_d_pose = _finite(mean_d_pose, "mean_d_pose")
    if canonical_d_pose < 0.0:
        raise G91PoseTrajectoryError("mean_d_pose must be nonnegative")
    if type(exact_outer_bytes) is not int or exact_outer_bytes <= 0:
        raise G91PoseTrajectoryError("exact_outer_bytes must be one positive exact integer")
    pose_term = math.sqrt(10.0 * canonical_d_pose)
    rate_term = 25.0 * exact_outer_bytes / RATE_DENOMINATOR_BYTES
    return pose_term, rate_term, pose_term + rate_term


def select_measured_mode_or_base(
    *,
    rows: tuple[G91ModeSelectionRowV1, ...],
    pass_d_pose: np.ndarray,
    base_outer_bytes: int,
) -> G91ModeSelectionRowV1 | None:
    """Select a strictly improving executable row or preserve exact base K=0.

    The unchanged base is not encoded as a fake empty-XIP2 operand.  A tie or
    regression returns ``None``, which is the typed ``NO_ACTIVE_G91_CANDIDATE``
    decision and requires callers to retain the exact incoming archive.
    """

    if not rows or any(type(row) is not G91ModeSelectionRowV1 for row in rows):
        raise G91PoseTrajectoryError("rows must be a nonempty exact G91 selection-row tuple")
    pass_loss = np.asarray(pass_d_pose)
    if (
        pass_loss.dtype != np.float64
        or pass_loss.shape != (PAIR_COUNT,)
        or not np.all(np.isfinite(pass_loss))
        or np.any(pass_loss < 0.0)
    ):
        raise G91PoseTrajectoryError("pass_d_pose must be exact finite nonnegative float64 [600]")
    base_objective = exact_prefix_objective(
        mean_d_pose=float(pass_loss.mean()),
        exact_outer_bytes=base_outer_bytes,
    )[2]
    best = min(
        rows,
        key=lambda row: (
            row.selector_objective,
            row.coder_row.selected_outer_bytes,
            row.control_rows,
            row.default_mode,
        ),
    )
    return best if best.selector_objective < base_objective else None


__all__ = [
    "AUTHORITY_BLOCKER",
    "CODER_UNIVERSE",
    "COMPILER_ID",
    "DIRECT_POSE_TARGET_AS_WARP_CONTROL_ADMISSIBLE",
    "FRESH_TARGET_CUSTODY_QUALIFIER",
    "G16_SETTLED_NEGATIVE_AUTHORITY",
    "INVERSE_CONTROL_SOLVED",
    "ONLY_ADMISSIBLE_PROMOTION_PATH",
    "PAIR_COUNT",
    "RATE_DENOMINATOR_BYTES",
    "SELECTOR_POLICY_ID",
    "TRAJECTORY_POLICY_ID",
    "G91CoderRaceRowV1",
    "G91ModeSelectionRowV1",
    "G91PoseTrajectoryError",
    "G91TrajectoryTreatmentV1",
    "G91TrajectoryV1",
    "centered_rank_projection",
    "compile_coder_race",
    "compile_measured_mode_selection",
    "derive_fresh_trajectory",
    "exact_prefix_objective",
    "select_measured_mode_or_base",
    "sparse_exception_counts",
]
