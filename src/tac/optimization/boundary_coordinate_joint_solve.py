# SPDX-License-Identifier: MIT
"""Counted localized boundary carrier with joint uint8 hard-oracle admission.

The generic curvelet/shearlet frame is regenerated at the receiver.  Atom
selection, quantized per-pair RGB coefficients, and scales are video-derived
and therefore serialized in the counted packet.  Candidate admission places
the exact factor-2 uint8 preimage and a fresh hard oracle *inside* the solve;
this module intentionally has no post-hoc repair path.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Final

import numpy as np

from tac.boundary_math.compact_shearlet_frame import (
    CompactShearletConfig,
    compact_shearlet_feats,
)
from tac.boundary_math.shared_receiver_admission import RATE_PRICE_PER_BYTE
from tac.boundary_math.windowed_curvelet_frame import (
    WindowedCurveletConfig,
    windowed_curvelet_feats,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    HardOracleEvaluation,
    RepairStatus,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

PACKET_SCHEMA: Final = "boundary_coordinate_packet.v1"
PACKET_MAGIC: Final = b"BGJ1"
PACKET_VERSION: Final = 1
_PACKET_PREFIX: Final = struct.Struct("<4sIII")
_PACKET_CRC: Final = struct.Struct("<I")
_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "family",
        "frame_config",
        "scorer_height",
        "scorer_width",
        "pair_count",
        "selected_feature_count",
        "channels",
        "atom_index_dtype",
        "coefficient_dtype",
        "scale_dtype",
        "atom_index_bytes",
        "coefficient_bytes",
        "scale_bytes",
        "body_sha256",
    }
)
ERM_REPLICAS: Final = 4
ERM_PROPOSALS_PER_REPLICA: Final = 16


class BoundaryJointSolveError(ValueError):
    """Fail-closed malformed packet, solve input, or evidence error."""


class FrameFamily(StrEnum):
    WINDOWED_CURVELET = "windowed_curvelet"
    COMPACT_SHEARLET = "compact_shearlet"


class QPSolveStatus(StrEnum):
    SOLVED = "SOLVED"
    STALLED_UNKNOWN = "STALLED_UNKNOWN"
    CYCLE_DETECTED_UNKNOWN = "CYCLE_DETECTED_UNKNOWN"
    BUDGET_EXHAUSTED_UNKNOWN = "BUDGET_EXHAUSTED_UNKNOWN"


class JointSolveStatus(StrEnum):
    FEASIBLE_HARD_ACCEPT = "FEASIBLE_HARD_ACCEPT"
    HARD_REJECTED_UNKNOWN = "HARD_REJECTED_UNKNOWN"
    STALLED_UNKNOWN = "STALLED_UNKNOWN"
    CYCLE_DETECTED_UNKNOWN = "CYCLE_DETECTED_UNKNOWN"
    BUDGET_EXHAUSTED_UNKNOWN = "BUDGET_EXHAUSTED_UNKNOWN"


class ERMStatus(StrEnum):
    HARD_ACCEPT = "HARD_ACCEPT"
    NO_HARD_ACCEPT = "NO_HARD_ACCEPT"
    DEGENERATE_ENERGY_SPREAD = "DEGENERATE_ENERGY_SPREAD"


def _immutable_copy(value: np.ndarray, *, dtype: np.dtype[Any] | str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BoundaryJointSolveError(f"{name} must be a positive integer")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BoundaryJointSolveError("packet header must be canonical JSON") from exc


def _validated_config(
    family: FrameFamily, config: Mapping[str, Any]
) -> WindowedCurveletConfig | CompactShearletConfig:
    if not isinstance(config, Mapping):
        raise BoundaryJointSolveError("frame_config must be a mapping")
    try:
        if family is FrameFamily.WINDOWED_CURVELET:
            typed: WindowedCurveletConfig | CompactShearletConfig = WindowedCurveletConfig(
                **dict(config)
            )
        elif family is FrameFamily.COMPACT_SHEARLET:
            typed = CompactShearletConfig(**dict(config))
        else:  # pragma: no cover - exhaustive enum guard
            raise BoundaryJointSolveError(f"unsupported frame family: {family}")
    except (TypeError, ValueError) as exc:
        raise BoundaryJointSolveError("invalid localized-frame configuration") from exc
    if dict(config) != asdict(typed):
        raise BoundaryJointSolveError(
            "frame_config must spell every canonical generic frame field exactly"
        )
    return typed


@dataclass(frozen=True)
class BoundaryCoordinatePacket:
    """Strict counted payload for a generic localized receiver frame."""

    family: FrameFamily
    frame_config: Mapping[str, Any]
    scorer_height: int
    scorer_width: int
    atom_indices: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        try:
            family = FrameFamily(self.family)
        except ValueError as exc:
            raise BoundaryJointSolveError(
                "family must be genuine windowed_curvelet or compact_shearlet"
            ) from exc
        config = _validated_config(family, self.frame_config)
        height = _positive_integer(self.scorer_height, "scorer_height")
        width = _positive_integer(self.scorer_width, "scorer_width")
        raw_indices = np.asarray(self.atom_indices)
        raw_coefficients = np.asarray(self.coefficients)
        raw_scales = np.asarray(self.scales)
        if raw_indices.dtype.kind not in ("i", "u") or raw_indices.ndim != 1:
            raise BoundaryJointSolveError("atom_indices must be a one-dimensional integer array")
        if raw_indices.size == 0 or np.any(raw_indices < 0):
            raise BoundaryJointSolveError("atom_indices must be nonempty and non-negative")
        indices = raw_indices.astype("<u4", copy=True)
        if len({int(v) for v in indices}) != int(indices.size):
            raise BoundaryJointSolveError("atom_indices must be unique")
        if raw_coefficients.dtype != np.int8 or raw_coefficients.ndim != 3:
            raise BoundaryJointSolveError("coefficients must be pair x feature x RGB int8")
        if raw_coefficients.shape[1:] != (indices.size, 3):
            raise BoundaryJointSolveError("coefficient feature/channel shape mismatch")
        if raw_coefficients.shape[0] <= 0:
            raise BoundaryJointSolveError("coefficients must cover at least one pair")
        if raw_scales.ndim != 1 or raw_scales.shape[0] != raw_coefficients.shape[0]:
            raise BoundaryJointSolveError("scales must contain one value per pair")
        scales64 = raw_scales.astype(np.float64, copy=False)
        if not np.all(np.isfinite(scales64)) or np.any(scales64 <= 0.0):
            raise BoundaryJointSolveError("scales must be finite and strictly positive")
        canonical_scales = raw_scales.astype("<f2", copy=True)
        if not np.all(np.isfinite(canonical_scales)) or np.any(canonical_scales <= 0):
            raise BoundaryJointSolveError("scales must remain positive finite float16 values")
        feature_count = _frame_feature_count(family, config)
        if int(np.max(indices)) >= feature_count:
            raise BoundaryJointSolveError(
                f"atom index exceeds the generic frame's {feature_count} features"
            )
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "frame_config", asdict(config))
        object.__setattr__(self, "scorer_height", height)
        object.__setattr__(self, "scorer_width", width)
        object.__setattr__(self, "atom_indices", _immutable_copy(indices, dtype="<u4"))
        object.__setattr__(
            self, "coefficients", _immutable_copy(raw_coefficients, dtype=np.int8)
        )
        object.__setattr__(self, "scales", _immutable_copy(canonical_scales, dtype="<f2"))

    @property
    def pair_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def selected_feature_count(self) -> int:
        return int(self.atom_indices.size)


def _frame_feature_count(
    family: FrameFamily,
    config: WindowedCurveletConfig | CompactShearletConfig,
) -> int:
    probe = np.zeros((1, 2), dtype=np.float32)
    if family is FrameFamily.WINDOWED_CURVELET:
        return int(windowed_curvelet_feats(probe, config).shape[1])
    return int(compact_shearlet_feats(probe, config).shape[1])


def encode_boundary_packet(packet: BoundaryCoordinatePacket) -> bytes:
    """Serialize with fixed endian dtypes, canonical JSON, body hash, and CRC."""

    atom_bytes = packet.atom_indices.astype("<u4", copy=False).tobytes(order="C")
    coefficient_bytes = packet.coefficients.tobytes(order="C")
    scale_bytes = packet.scales.astype("<f2", copy=False).tobytes(order="C")
    body = atom_bytes + coefficient_bytes + scale_bytes
    header = {
        "schema": PACKET_SCHEMA,
        "version": PACKET_VERSION,
        "family": packet.family.value,
        "frame_config": dict(packet.frame_config),
        "scorer_height": packet.scorer_height,
        "scorer_width": packet.scorer_width,
        "pair_count": packet.pair_count,
        "selected_feature_count": packet.selected_feature_count,
        "channels": 3,
        "atom_index_dtype": "uint32_le",
        "coefficient_dtype": "int8",
        "scale_dtype": "float16_le",
        "atom_index_bytes": len(atom_bytes),
        "coefficient_bytes": len(coefficient_bytes),
        "scale_bytes": len(scale_bytes),
        "body_sha256": sha256(body).hexdigest(),
    }
    header_bytes = _canonical_json(header)
    prefix = _PACKET_PREFIX.pack(
        PACKET_MAGIC, PACKET_VERSION, len(header_bytes), len(body)
    )
    checksum = _PACKET_CRC.pack(zlib.crc32(header_bytes + body) & 0xFFFFFFFF)
    return prefix + header_bytes + body + checksum


def decode_boundary_packet(payload: bytes) -> BoundaryCoordinatePacket:
    """Strictly parse a counted packet; trailing data and metadata drift refuse."""

    if not isinstance(payload, bytes) or len(payload) < _PACKET_PREFIX.size + _PACKET_CRC.size:
        raise BoundaryJointSolveError("packet is truncated or not bytes")
    magic, version, header_size, body_size = _PACKET_PREFIX.unpack_from(payload)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise BoundaryJointSolveError("packet magic/version mismatch")
    expected_size = _PACKET_PREFIX.size + header_size + body_size + _PACKET_CRC.size
    if len(payload) != expected_size:
        raise BoundaryJointSolveError("packet length mismatch or trailing bytes")
    header_start = _PACKET_PREFIX.size
    body_start = header_start + header_size
    body_end = body_start + body_size
    header_bytes = payload[header_start:body_start]
    body = payload[body_start:body_end]
    (stored_crc,) = _PACKET_CRC.unpack(payload[body_end:])
    if stored_crc != (zlib.crc32(header_bytes + body) & 0xFFFFFFFF):
        raise BoundaryJointSolveError("packet CRC mismatch")
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryJointSolveError("packet header is not ASCII JSON") from exc
    if not isinstance(header, dict) or frozenset(header) != _HEADER_FIELDS:
        raise BoundaryJointSolveError("packet header fields mismatch")
    if _canonical_json(header) != header_bytes:
        raise BoundaryJointSolveError("packet header is not canonically encoded")
    if header["schema"] != PACKET_SCHEMA or header["version"] != PACKET_VERSION:
        raise BoundaryJointSolveError("packet schema/version mismatch")
    if header["channels"] != 3:
        raise BoundaryJointSolveError("packet must carry exactly RGB channels")
    if (
        header["atom_index_dtype"] != "uint32_le"
        or header["coefficient_dtype"] != "int8"
        or header["scale_dtype"] != "float16_le"
    ):
        raise BoundaryJointSolveError("packet dtype contract mismatch")
    if header["body_sha256"] != sha256(body).hexdigest():
        raise BoundaryJointSolveError("packet body SHA-256 mismatch")
    pair_count = _positive_integer(header["pair_count"], "pair_count")
    selected = _positive_integer(
        header["selected_feature_count"], "selected_feature_count"
    )
    expected_parts = (selected * 4, pair_count * selected * 3, pair_count * 2)
    observed_parts = (
        header["atom_index_bytes"],
        header["coefficient_bytes"],
        header["scale_bytes"],
    )
    if observed_parts != expected_parts or sum(expected_parts) != len(body):
        raise BoundaryJointSolveError("packet body geometry mismatch")
    atom_end = expected_parts[0]
    coefficient_end = atom_end + expected_parts[1]
    indices = np.frombuffer(body[:atom_end], dtype="<u4").copy()
    coefficients = np.frombuffer(body[atom_end:coefficient_end], dtype=np.int8).reshape(
        pair_count, selected, 3
    ).copy()
    scales = np.frombuffer(body[coefficient_end:], dtype="<f2").copy()
    try:
        family = FrameFamily(header["family"])
    except ValueError as exc:
        raise BoundaryJointSolveError("packet frame family is not localized") from exc
    return BoundaryCoordinatePacket(
        family=family,
        frame_config=header["frame_config"],
        scorer_height=_positive_integer(header["scorer_height"], "scorer_height"),
        scorer_width=_positive_integer(header["scorer_width"], "scorer_width"),
        atom_indices=indices,
        coefficients=coefficients,
        scales=scales,
    )


def _normalized_coordinates(height: int, width: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)


def selected_frame_features(packet: BoundaryCoordinatePacket) -> np.ndarray:
    """Regenerate the generic localized frame and select counted columns."""

    config = _validated_config(packet.family, packet.frame_config)
    coords = _normalized_coordinates(packet.scorer_height, packet.scorer_width)
    if packet.family is FrameFamily.WINDOWED_CURVELET:
        full = windowed_curvelet_feats(coords, config)
    else:
        full = compact_shearlet_feats(coords, config)
    return np.asarray(full[:, packet.atom_indices], dtype=np.float64)


def rgb_direction_matrix(features: np.ndarray) -> np.ndarray:
    """Lift P x D localized atoms to interleaved RGB coordinates, P*3 x D*3."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise BoundaryJointSolveError("features must be a nonempty P x D matrix")
    if not np.all(np.isfinite(array)):
        raise BoundaryJointSolveError("features must be finite")
    pixels, features_count = array.shape
    result = np.zeros((pixels * 3, features_count * 3), dtype=np.float64)
    for channel in range(3):
        result[channel::3, channel::3] = array
    return result


def apply_boundary_packet(
    baseline_scorer_plane: np.ndarray,
    packet: BoundaryCoordinatePacket,
    pair_index: int,
) -> np.ndarray:
    """Decode one packet coordinate to an exact uint8 scorer-plane target."""

    baseline = np.asarray(baseline_scorer_plane)
    if baseline.dtype != np.uint8 or baseline.shape != (
        packet.scorer_height,
        packet.scorer_width,
        3,
    ):
        raise BoundaryJointSolveError("baseline scorer plane must match packet HxWx3 uint8")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise BoundaryJointSolveError("pair_index must be an integer")
    if not 0 <= pair_index < packet.pair_count:
        raise BoundaryJointSolveError("pair_index is out of range")
    features = selected_frame_features(packet)
    coefficients = packet.coefficients[pair_index].astype(np.float64)
    delta = features @ coefficients
    delta *= float(packet.scales[pair_index])
    target = baseline.astype(np.float64) + np.rint(delta).reshape(
        packet.scorer_height, packet.scorer_width, 3
    )
    return np.clip(target, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class ActiveSetQPResult:
    coefficients: np.ndarray
    status: QPSolveStatus
    active_rows: tuple[int, ...]
    iterations: int
    max_primal_violation: float
    min_active_multiplier: float | None
    stationarity_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", _immutable_copy(self.coefficients, dtype=np.float64)
        )


def solve_corrected_active_set_qp(
    first_order_jacobian: np.ndarray,
    secant_jacobian: np.ndarray,
    debt: np.ndarray,
    fisher_diagonal: np.ndarray,
    *,
    max_iterations: int = 128,
    tolerance: float = 1e-9,
) -> ActiveSetQPResult:
    """Solve weighted least norm under explicit first-order + secant constraints.

    The secant matrix is mandatory and shape-checked.  It may contain measured
    zeros, but it may not be omitted or silently synthesized by this function.
    """

    first = np.asarray(first_order_jacobian, dtype=np.float64)
    secant = np.asarray(secant_jacobian, dtype=np.float64)
    rhs = np.asarray(debt, dtype=np.float64)
    qdiag = np.asarray(fisher_diagonal, dtype=np.float64)
    if first.ndim != 2 or first.shape != secant.shape or first.shape[0] == 0:
        raise BoundaryJointSolveError("first-order and secant Jacobians must share nonempty MxD shape")
    if rhs.shape != (first.shape[0],) or qdiag.shape != (first.shape[1],):
        raise BoundaryJointSolveError("debt/Fisher geometry does not match Jacobians")
    if not all(np.all(np.isfinite(v)) for v in (first, secant, rhs, qdiag)):
        raise BoundaryJointSolveError("QP inputs must be finite")
    if np.any(qdiag <= 0.0):
        raise BoundaryJointSolveError("Fisher diagonal must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise BoundaryJointSolveError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise BoundaryJointSolveError("tolerance must be finite and positive")

    jacobian = first + secant
    inverse_q = 1.0 / qdiag
    # NumPy/Accelerate on some macOS hosts emits spurious floating-point
    # warnings from padded SIMD lanes in small matmul kernels even when the
    # returned matrix is finite.  Suppress only around the kernel and validate
    # every authoritative value immediately afterward.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = (jacobian * inverse_q[None, :]) @ jacobian.T
    if not np.all(np.isfinite(gram)):
        raise BoundaryJointSolveError("corrected QP Gram matrix is non-finite")
    multipliers = np.zeros(rhs.size, dtype=np.float64)
    passive: list[int] = []
    seen: set[tuple[int, ...]] = set()

    # Dual active-set solve:
    #   max_{lambda>=0} b^T lambda - 1/2 lambda^T A Q^-1 A^T lambda.
    # Its KKT map gives c=Q^-1 A^T lambda and A c-b>=0.  This is the
    # Lawson-Hanson active-set method specialized to the positive orthant.
    for iteration in range(1, max_iterations + 1):
        dual_ascent = rhs - gram @ multipliers
        inactive = [index for index in range(rhs.size) if index not in passive]
        worst = (
            max(inactive, key=lambda index: (dual_ascent[index], -index))
            if inactive
            else None
        )
        passive_residual = (
            float(np.max(np.abs(dual_ascent[passive]))) if passive else 0.0
        )
        if (
            (worst is None or float(dual_ascent[worst]) <= tolerance)
            and passive_residual <= tolerance
        ):
            coefficients = inverse_q * (jacobian.T @ multipliers)
            max_violation = max(0.0, float(np.max(rhs - jacobian @ coefficients)))
            stationarity = qdiag * coefficients - jacobian.T @ multipliers
            return ActiveSetQPResult(
                coefficients,
                QPSolveStatus.SOLVED,
                tuple(passive),
                iteration,
                max_violation,
                float(np.min(multipliers[passive])) if passive else None,
                float(np.max(np.abs(stationarity))),
            )
        if worst is None:
            coefficients = inverse_q * (jacobian.T @ multipliers)
            return ActiveSetQPResult(
                coefficients,
                QPSolveStatus.STALLED_UNKNOWN,
                tuple(passive),
                iteration,
                max(0.0, float(np.max(rhs - jacobian @ coefficients))),
                float(np.min(multipliers[passive])) if passive else None,
                math.inf,
            )
        passive.append(worst)

        while passive:
            passive_gram = gram[np.ix_(passive, passive)]
            try:
                solution = np.linalg.solve(passive_gram, rhs[passive])
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(passive_gram, rhs[passive], rcond=None)[0]
            trial = np.zeros_like(multipliers)
            trial[passive] = solution
            nonpositive = [index for index in passive if trial[index] <= tolerance]
            if not nonpositive:
                multipliers = trial
                break
            fractions = [
                multipliers[index]
                / (multipliers[index] - trial[index])
                for index in nonpositive
                if multipliers[index] - trial[index] > tolerance
            ]
            alpha = min(fractions, default=0.0)
            multipliers += alpha * (trial - multipliers)
            remove = [index for index in passive if multipliers[index] <= tolerance]
            if not remove:
                remove = [min(nonpositive)]
            for index in remove:
                multipliers[index] = 0.0
                passive.remove(index)

        signature = tuple(sorted(passive))
        if signature in seen:
            coefficients = inverse_q * (jacobian.T @ multipliers)
            return ActiveSetQPResult(
                coefficients,
                QPSolveStatus.CYCLE_DETECTED_UNKNOWN,
                tuple(passive),
                iteration,
                max(0.0, float(np.max(rhs - jacobian @ coefficients))),
                float(np.min(multipliers[passive])) if passive else None,
                math.inf,
            )
        seen.add(signature)

    coefficients = inverse_q * (jacobian.T @ multipliers)
    final_violation = float(np.max(rhs - jacobian @ coefficients))
    return ActiveSetQPResult(
        coefficients,
        QPSolveStatus.BUDGET_EXHAUSTED_UNKNOWN,
        tuple(passive),
        max_iterations,
        max(0.0, final_violation),
        float(np.min(multipliers[passive])) if passive else None,
        math.inf,
    )


@dataclass(frozen=True)
class RealizedBoundaryCandidate:
    coefficients: np.ndarray
    scorer_target: np.ndarray
    camera_frame: np.ndarray
    exact_verification: Factor2ExactVerification
    hard_evaluation: HardOracleEvaluation

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", _immutable_copy(self.coefficients, dtype=np.float64)
        )
        object.__setattr__(self, "scorer_target", _immutable_copy(self.scorer_target, dtype=np.uint8))
        object.__setattr__(self, "camera_frame", _immutable_copy(self.camera_frame, dtype=np.uint8))


def _render_and_evaluate(
    coefficients: np.ndarray,
    baseline_scorer_plane: np.ndarray,
    direction_matrix: np.ndarray,
    operator: DisjointResizeOperator,
    hard_oracle: Callable[[np.ndarray], HardOracleEvaluation],
) -> RealizedBoundaryCandidate:
    baseline = np.asarray(baseline_scorer_plane)
    directions = np.asarray(direction_matrix, dtype=np.float64)
    coeff = np.asarray(coefficients, dtype=np.float64)
    if baseline.dtype != np.uint8 or baseline.ndim != 3 or baseline.shape[2] != 3:
        raise BoundaryJointSolveError("baseline_scorer_plane must be HxWx3 uint8")
    if (baseline.shape[0], baseline.shape[1]) != (operator.scorer_h, operator.scorer_w):
        raise BoundaryJointSolveError("baseline scorer geometry does not match resize operator")
    if directions.shape != (baseline.size, coeff.size):
        raise BoundaryJointSolveError("direction matrix must map coefficients to flattened RGB")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(coeff)):
        raise BoundaryJointSolveError("candidate directions/coefficients must be finite")
    delta = np.rint(directions @ coeff).reshape(baseline.shape)
    target = np.clip(baseline.astype(np.float64) + delta, 0.0, 255.0).astype(np.uint8)
    camera = realize_factor2_uint8_scorer_plane(operator, target)
    verification = verify_factor2_uint8_scorer_plane(operator, camera, target)
    if not verification.certified_exact:
        raise BoundaryJointSolveError("factor-2 uint8 realization did not certify exact")
    evaluation = hard_oracle(camera.copy())
    if not isinstance(evaluation, HardOracleEvaluation):
        raise BoundaryJointSolveError("hard oracle must return HardOracleEvaluation")
    return RealizedBoundaryCandidate(coeff, target, camera, verification, evaluation)


@dataclass(frozen=True)
class JointBoundarySolveResult:
    status: JointSolveStatus
    qp: ActiveSetQPResult
    baseline_evaluation: HardOracleEvaluation
    candidate: RealizedBoundaryCandidate | None


def solve_joint_boundary_candidate(
    *,
    baseline_scorer_plane: np.ndarray,
    direction_matrix: np.ndarray,
    operator: DisjointResizeOperator,
    first_order_jacobian: np.ndarray,
    secant_jacobian: np.ndarray,
    debt: np.ndarray,
    fisher_diagonal: np.ndarray,
    hard_oracle: Callable[[np.ndarray], HardOracleEvaluation],
    max_qp_iterations: int = 128,
) -> JointBoundarySolveResult:
    """Run corrected QP, exact uint8 realization, and fresh hard admission once."""

    zero = np.zeros(np.asarray(direction_matrix).shape[1], dtype=np.float64)
    baseline_candidate = _render_and_evaluate(
        zero, baseline_scorer_plane, direction_matrix, operator, hard_oracle
    )
    qp = solve_corrected_active_set_qp(
        first_order_jacobian,
        secant_jacobian,
        debt,
        fisher_diagonal,
        max_iterations=max_qp_iterations,
    )
    if qp.status is not QPSolveStatus.SOLVED:
        return JointBoundarySolveResult(
            JointSolveStatus(qp.status.value), qp, baseline_candidate.hard_evaluation, None
        )
    candidate = _render_and_evaluate(
        qp.coefficients, baseline_scorer_plane, direction_matrix, operator, hard_oracle
    )
    if candidate.hard_evaluation.key[0] == 0:
        status = JointSolveStatus.FEASIBLE_HARD_ACCEPT
    elif candidate.hard_evaluation.key < baseline_candidate.hard_evaluation.key:
        status = JointSolveStatus.STALLED_UNKNOWN
    else:
        status = JointSolveStatus.HARD_REJECTED_UNKNOWN
    return JointBoundarySolveResult(
        status, qp, baseline_candidate.hard_evaluation, candidate
    )


@dataclass(frozen=True)
class ERMFallbackResult:
    status: ERMStatus
    candidate: RealizedBoundaryCandidate | None
    cheap_evaluations: int
    hard_terminal_evaluations: int
    replica_energy_spreads: tuple[float, ...]


def run_exact_erm_fallback(
    *,
    unknown_status: RepairStatus | JointSolveStatus,
    seed_coefficients: np.ndarray,
    baseline_scorer_plane: np.ndarray,
    direction_matrix: np.ndarray,
    operator: DisjointResizeOperator,
    hard_oracle: Callable[[np.ndarray], HardOracleEvaluation],
    cheap_energy: Callable[[np.ndarray], float],
    seed: int,
    proposal_scale: float = 1.0,
    degeneracy_tolerance: float = 1e-12,
) -> ERMFallbackResult:
    """Equal-budget 4x16 ERM probe with fresh hard terminal authority.

    Only solver-unknown states are routable.  Cheap Fisher/margin energy ranks
    proposals but cannot admit one; each nondegenerate replica contributes one
    independently rendered uint8 candidate to the hard oracle.
    """

    allowed = {
        RepairStatus.STALLED_UNKNOWN.value,
        RepairStatus.CYCLE_DETECTED_UNKNOWN.value,
        RepairStatus.BUDGET_EXHAUSTED_UNKNOWN.value,
        JointSolveStatus.STALLED_UNKNOWN.value,
        JointSolveStatus.CYCLE_DETECTED_UNKNOWN.value,
        JointSolveStatus.BUDGET_EXHAUSTED_UNKNOWN.value,
    }
    status_value = (
        unknown_status.value
        if isinstance(unknown_status, (RepairStatus, JointSolveStatus))
        else None
    )
    if status_value not in allowed:
        raise BoundaryJointSolveError("ERM fallback is allowed only for explicit unknown states")
    seed_coeff = np.asarray(seed_coefficients, dtype=np.float64)
    if seed_coeff.ndim != 1 or seed_coeff.size == 0 or not np.all(np.isfinite(seed_coeff)):
        raise BoundaryJointSolveError("seed_coefficients must be a finite nonempty vector")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BoundaryJointSolveError("seed must be an integer")
    if not math.isfinite(proposal_scale) or proposal_scale <= 0.0:
        raise BoundaryJointSolveError("proposal_scale must be finite and positive")
    if not math.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
        raise BoundaryJointSolveError("degeneracy_tolerance must be finite and non-negative")

    baseline_candidate = _render_and_evaluate(
        np.zeros_like(seed_coeff),
        baseline_scorer_plane,
        direction_matrix,
        operator,
        hard_oracle,
    )
    generator = np.random.default_rng(seed)
    terminal_proposals: list[np.ndarray] = []
    spreads: list[float] = []
    for _replica in range(ERM_REPLICAS):
        proposals: list[np.ndarray] = []
        energies: list[float] = []
        for _proposal in range(ERM_PROPOSALS_PER_REPLICA):
            perturbation = generator.integers(-2, 3, size=seed_coeff.size).astype(np.float64)
            proposed = seed_coeff + proposal_scale * perturbation
            energy = cheap_energy(proposed.copy())
            if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
                raise BoundaryJointSolveError("cheap energy must return a finite scalar")
            proposals.append(proposed)
            energies.append(float(energy))
        spread = float(max(energies) - min(energies))
        spreads.append(spread)
        winner = int(np.argmin(np.asarray(energies, dtype=np.float64)))
        terminal_proposals.append(proposals[winner])

    cheap_count = ERM_REPLICAS * ERM_PROPOSALS_PER_REPLICA
    if any(spread <= degeneracy_tolerance for spread in spreads):
        return ERMFallbackResult(
            ERMStatus.DEGENERATE_ENERGY_SPREAD, None, cheap_count, 0, tuple(spreads)
        )
    terminal_candidates = [
        _render_and_evaluate(
            proposal,
            baseline_scorer_plane,
            direction_matrix,
            operator,
            hard_oracle,
        )
        for proposal in terminal_proposals
    ]
    hard_accepted = [
        candidate
        for candidate in terminal_candidates
        if candidate.hard_evaluation.key[0] == 0
        and candidate.hard_evaluation.key < baseline_candidate.hard_evaluation.key
    ]
    if not hard_accepted:
        return ERMFallbackResult(
            ERMStatus.NO_HARD_ACCEPT,
            None,
            cheap_count,
            len(terminal_candidates),
            tuple(spreads),
        )
    best = min(hard_accepted, key=lambda candidate: candidate.hard_evaluation.key)
    return ERMFallbackResult(
        ERMStatus.HARD_ACCEPT,
        best,
        cheap_count,
        len(terminal_candidates),
        tuple(spreads),
    )


@dataclass(frozen=True)
class MeasuredCoordinate:
    coordinate_id: str
    boundary_radius: int
    score_gain: float
    charged_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_id, str) or not self.coordinate_id:
            raise BoundaryJointSolveError("coordinate_id must be nonempty")
        if isinstance(self.boundary_radius, bool) or not isinstance(self.boundary_radius, int) or self.boundary_radius <= 0:
            raise BoundaryJointSolveError("boundary_radius must be a positive integer")
        if not math.isfinite(self.score_gain) or self.score_gain < 0.0:
            raise BoundaryJointSolveError("score_gain must be finite and non-negative")
        if isinstance(self.charged_bytes, bool) or not isinstance(self.charged_bytes, int) or self.charged_bytes <= 0:
            raise BoundaryJointSolveError("charged_bytes must be a positive integer")

    @property
    def score_per_byte(self) -> float:
        return self.score_gain / self.charged_bytes


@dataclass(frozen=True)
class WaterfillResult:
    selected_ids: tuple[str, ...]
    spent_bytes: int
    score_gain: float
    first_rejected_id: str | None
    rate_price_per_byte: float


def select_measured_boundary_coordinates(
    coordinates: Sequence[MeasuredCoordinate],
    *,
    byte_budget: int,
    rate_price_per_byte: float = RATE_PRICE_PER_BYTE,
) -> WaterfillResult:
    """Radius-one-first strict reverse-waterfill over measured coordinate rows."""

    if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget < 0:
        raise BoundaryJointSolveError("byte_budget must be a non-negative integer")
    if not math.isfinite(rate_price_per_byte) or rate_price_per_byte <= 0.0:
        raise BoundaryJointSolveError("rate price must be finite and positive")
    rows = list(coordinates)
    if len({row.coordinate_id for row in rows}) != len(rows):
        raise BoundaryJointSolveError("coordinate_id values must be unique")
    indexed_rows = list(enumerate(rows))
    phases = (
        sorted(
            (item for item in indexed_rows if item[1].boundary_radius == 1),
            key=lambda item: (-item[1].score_per_byte, item[0]),
        ),
        sorted(
            (item for item in indexed_rows if item[1].boundary_radius != 1),
            key=lambda item: (-item[1].score_per_byte, item[0]),
        ),
    )
    selected: list[str] = []
    spent = 0
    score_gain = 0.0
    first_rejected: str | None = None
    for phase in phases:
        for _ordinal, row in phase:
            if row.score_per_byte <= rate_price_per_byte:
                if first_rejected is None:
                    first_rejected = row.coordinate_id
                break
            if spent + row.charged_bytes > byte_budget:
                if first_rejected is None:
                    first_rejected = row.coordinate_id
                break
            selected.append(row.coordinate_id)
            spent += row.charged_bytes
            score_gain += row.score_gain
    return WaterfillResult(
        tuple(selected), spent, score_gain, first_rejected, float(rate_price_per_byte)
    )


__all__ = [
    "BoundaryCoordinatePacket",
    "BoundaryJointSolveError",
    "ERMFallbackResult",
    "ERMStatus",
    "FrameFamily",
    "JointBoundarySolveResult",
    "JointSolveStatus",
    "MeasuredCoordinate",
    "QPSolveStatus",
    "WaterfillResult",
    "apply_boundary_packet",
    "decode_boundary_packet",
    "encode_boundary_packet",
    "rgb_direction_matrix",
    "run_exact_erm_fallback",
    "select_measured_boundary_coordinates",
    "selected_frame_features",
    "solve_corrected_active_set_qp",
    "solve_joint_boundary_candidate",
]
