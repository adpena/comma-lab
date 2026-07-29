# SPDX-License-Identifier: MIT
"""Terminal six-equation pose solve on frozen composed uint8 frames.

The solver owns no PoseNet, basis, or learned video data.  The caller supplies
the low-frequency basis renderer, the exact packet replacement callback, and
the scorer callback.  Every finite-difference and line-search verdict executes
the receiver path:

    render basis -> zero-mean/unit-RMS normalize -> add -> clamp -> round uint8
    -> canonical packet parse-back -> realized scorer callback

Only frame 0 may change.  Frame 1 is copied byte-for-byte from the frozen
composed parent and asserted after every realization.  A bounded stale
rehearsal can exercise the mechanism, but only a marked full-n600 joint verdict
with matching typed custody may produce a governed handoff.  This module never
promotes a score or moves the frontier pointer.  Production mode additionally
requires an atomic disk resume directory with immutable verdict, iteration,
and completed ledgers.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

FULL_N600_POSE_AUTHORITY_MARKER = "FULL_N600_TERMINAL_POSE_JOINT_ACTION"
STALE_POSE_REHEARSAL_AUTHORITY_MARKER = "STALE_POSE_MECHANISM_ONLY"
FULL_N600_SAMPLE_COUNT = 600

_PACKET_MAGIC = b"TPGNPKT1"
_PACKET_VERSION = 1
_PACKET_HEADER = struct.Struct(">8sBQIHII32s")
_PACKET_DIGEST_PREFIX = struct.Struct(">BQIHII")
_MAX_SELECTOR_BYTES = 255
_MAX_PACKET_PAIRS = 600
_MAX_PACKET_RANK = 64
_MAX_AMPLITUDE_Q8 = 65_535


class TerminalPoseError(ValueError):
    """Fail-closed malformed geometry, packet, callback, or authority result."""


class PoseAuthorityMode(StrEnum):
    PRODUCTION_FULL_N600 = "PRODUCTION_FULL_N600"
    STALE_REHEARSAL = "STALE_REHEARSAL"


class ContestAxis(StrEnum):
    CONTEST_CPU = "contest-CPU"
    CONTEST_CUDA = "contest-CUDA"


class CandidateArtifactScope(StrEnum):
    FULL_OUTER_ARCHIVE = "FULL_OUTER_ARCHIVE"
    TERMINAL_SECTION_ONLY = "TERMINAL_SECTION_ONLY"


def _sha256_text(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TerminalPoseError(f"{name} must be a lowercase SHA-256")
    return value


def _stripped_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TerminalPoseError(f"{name} must be a nonempty stripped string")
    return value


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class TerminalPoseCandidateArtifact:
    """Exact candidate bytes presented to the hard evaluator."""

    outer_archive: bytes
    terminal_packet: bytes
    scope: CandidateArtifactScope

    def __post_init__(self) -> None:
        if not isinstance(self.outer_archive, bytes) or not self.outer_archive:
            raise TerminalPoseError("artifact.outer_archive must be nonempty bytes")
        if not isinstance(self.terminal_packet, bytes) or not self.terminal_packet:
            raise TerminalPoseError("artifact.terminal_packet must be nonempty bytes")
        if not isinstance(self.scope, CandidateArtifactScope):
            raise TerminalPoseError("artifact.scope must be CandidateArtifactScope")

    @property
    def archive_sha256(self) -> str:
        return sha256(self.outer_archive).hexdigest()

    @property
    def archive_bytes(self) -> int:
        return len(self.outer_archive)

    @property
    def terminal_packet_sha256(self) -> str:
        return sha256(self.terminal_packet).hexdigest()

    @property
    def binding_sha256(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "archive_bytes": self.archive_bytes,
                    "archive_sha256": self.archive_sha256,
                    "scope": self.scope.value,
                    "terminal_packet_bytes": len(self.terminal_packet),
                    "terminal_packet_sha256": self.terminal_packet_sha256,
                }
            )
        ).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "binding_sha256": self.binding_sha256,
            "scope": self.scope.value,
            "terminal_packet_bytes": len(self.terminal_packet),
            "terminal_packet_sha256": self.terminal_packet_sha256,
        }


@dataclass(frozen=True)
class ProductionPoseCustodyV1:
    """Typed full-n600 scorer and runtime custody bound to every verdict."""

    parent_archive_sha256: str
    receiver_sha256: str
    evaluator_sha256: str
    upstream_sha256: str
    contest_axis: ContestAxis
    command: tuple[str, ...]
    hardware: str

    def __post_init__(self) -> None:
        for name in (
            "parent_archive_sha256",
            "receiver_sha256",
            "evaluator_sha256",
            "upstream_sha256",
        ):
            object.__setattr__(self, name, _sha256_text(getattr(self, name), name))
        if not isinstance(self.contest_axis, ContestAxis):
            raise TerminalPoseError("custody.contest_axis must be ContestAxis")
        if (
            not isinstance(self.command, tuple)
            or not self.command
            or any(not isinstance(part, str) or not part or part.strip() != part for part in self.command)
        ):
            raise TerminalPoseError("custody.command must be a nonempty tuple of argv strings")
        object.__setattr__(self, "hardware", _stripped_text(self.hardware, "custody.hardware"))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "ddm_terminal_pose_production_custody.v1",
            "parent_archive_sha256": self.parent_archive_sha256,
            "receiver_sha256": self.receiver_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "upstream_sha256": self.upstream_sha256,
            "contest_axis": self.contest_axis.value,
            "command": list(self.command),
            "hardware": self.hardware,
        }

    @property
    def digest(self) -> str:
        return sha256(_canonical_json(self.to_payload())).hexdigest()


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TerminalPoseError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise TerminalPoseError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise TerminalPoseError(f"{name} must be <= {maximum}")
    return result


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TerminalPoseError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise TerminalPoseError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise TerminalPoseError(f"{name} must be >= {minimum}")
    return result


def _selector(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TerminalPoseError("basis_selector must be a nonempty stripped string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TerminalPoseError("basis_selector must be ASCII") from exc
    if len(encoded) > _MAX_SELECTOR_BYTES:
        raise TerminalPoseError("basis_selector exceeds packet limit")
    return value


def _immutable_array(values: object, dtype: np.dtype, name: str) -> np.ndarray:
    raw = np.asarray(values)
    contiguous = np.ascontiguousarray(raw, dtype=dtype)
    if not np.all(np.isfinite(contiguous)) and contiguous.dtype.kind == "f":
        raise TerminalPoseError(f"{name} must be finite")
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


def _pose6(values: object, name: str) -> np.ndarray:
    result = _immutable_array(values, np.dtype(np.float64), name)
    if result.shape != (6,):
        raise TerminalPoseError(f"{name} must have shape (6,)")
    return result


def _coefficients(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim not in (1, 2) or raw.size == 0 or raw.dtype.kind not in ("i", "u"):
        raise TerminalPoseError(f"{name} must be a nonempty integer vector or matrix")
    if np.any(raw < np.iinfo(np.int16).min) or np.any(raw > np.iinfo(np.int16).max):
        raise TerminalPoseError(f"{name} exceeds int16")
    return _immutable_array(raw, np.dtype(np.int16), name)


@dataclass(frozen=True)
class TerminalPosePacketV1:
    """Counted payload: generic basis identity plus per-pair integer codes."""

    seed: int
    basis_selector: str
    amplitude_q8: int
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, "packet.seed", minimum=0, maximum=2**64 - 1))
        object.__setattr__(self, "basis_selector", _selector(self.basis_selector))
        object.__setattr__(
            self,
            "amplitude_q8",
            _integer(
                self.amplitude_q8,
                "packet.amplitude_q8",
                minimum=1,
                maximum=_MAX_AMPLITUDE_Q8,
            ),
        )
        coefficients = _coefficients(self.coefficients, "packet.coefficients")
        if coefficients.ndim != 2:
            raise TerminalPoseError("packet.coefficients must have shape [pairs, rank]")
        if not 1 <= coefficients.shape[0] <= _MAX_PACKET_PAIRS:
            raise TerminalPoseError("packet pair count exceeds receiver contract")
        if not 1 <= coefficients.shape[1] <= _MAX_PACKET_RANK:
            raise TerminalPoseError("packet rank exceeds receiver contract")
        object.__setattr__(self, "coefficients", coefficients)


def serialize_terminal_pose_packet(packet: TerminalPosePacketV1) -> bytes:
    """Serialize one canonical, checksummed terminal-pose payload."""

    if not isinstance(packet, TerminalPosePacketV1):
        raise TerminalPoseError("packet must be TerminalPosePacketV1")
    selector = packet.basis_selector.encode("ascii")
    coefficient_bytes = np.asarray(packet.coefficients, dtype=">i2").tobytes()
    body = selector + coefficient_bytes
    digest_material = (
        _PACKET_DIGEST_PREFIX.pack(
            _PACKET_VERSION,
            packet.seed,
            packet.amplitude_q8,
            len(selector),
            packet.coefficients.shape[0],
            packet.coefficients.shape[1],
        )
        + body
    )
    header = _PACKET_HEADER.pack(
        _PACKET_MAGIC,
        _PACKET_VERSION,
        packet.seed,
        packet.amplitude_q8,
        len(selector),
        packet.coefficients.shape[0],
        packet.coefficients.shape[1],
        sha256(digest_material).digest(),
    )
    return header + body


def parse_terminal_pose_packet(payload: bytes) -> TerminalPosePacketV1:
    """Strict parse: checksum, exact length, dimensions, and no trailing bytes."""

    if not isinstance(payload, bytes):
        raise TerminalPoseError("terminal pose payload must be bytes")
    if len(payload) < _PACKET_HEADER.size:
        raise TerminalPoseError("terminal pose payload is truncated")
    (
        magic,
        version,
        seed,
        amplitude_q8,
        selector_size,
        pair_count,
        rank,
        digest,
    ) = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    if magic != _PACKET_MAGIC or version != _PACKET_VERSION:
        raise TerminalPoseError("terminal pose packet magic/version differs")
    if not 1 <= selector_size <= _MAX_SELECTOR_BYTES:
        raise TerminalPoseError("terminal pose selector length differs")
    if not 1 <= pair_count <= _MAX_PACKET_PAIRS:
        raise TerminalPoseError("terminal pose pair count differs")
    if not 1 <= rank <= _MAX_PACKET_RANK:
        raise TerminalPoseError("terminal pose rank differs")
    expected = _PACKET_HEADER.size + selector_size + 2 * pair_count * rank
    if len(payload) != expected:
        raise TerminalPoseError("terminal pose payload length/trailing bytes differ")
    body = payload[_PACKET_HEADER.size :]
    digest_material = (
        _PACKET_DIGEST_PREFIX.pack(
            version,
            seed,
            amplitude_q8,
            selector_size,
            pair_count,
            rank,
        )
        + body
    )
    if sha256(digest_material).digest() != digest:
        raise TerminalPoseError("terminal pose payload checksum differs")
    selector_bytes = body[:selector_size]
    try:
        selector = selector_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TerminalPoseError("terminal pose basis selector is not ASCII") from exc
    coefficient_bytes = body[selector_size:]
    coefficients = np.frombuffer(coefficient_bytes, dtype=">i2").astype(np.int16).reshape(pair_count, rank)
    packet = TerminalPosePacketV1(
        seed=int(seed),
        basis_selector=selector,
        amplitude_q8=int(amplitude_q8),
        coefficients=coefficients,
    )
    if serialize_terminal_pose_packet(packet) != payload:
        raise TerminalPoseError("terminal pose packet is not canonical")
    return packet


@dataclass(frozen=True)
class PoseJointEvaluation:
    """Exact scorer callback result on one realized population candidate."""

    pose6: np.ndarray
    d_seg: float
    d_pose: float
    archive_bytes: int
    archive_sha256: str
    sample_count: int
    authority_marker: str
    custody_digest: str | None
    realized: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose6", _pose6(self.pose6, "evaluation.pose6"))
        object.__setattr__(
            self,
            "d_seg",
            _finite(self.d_seg, "evaluation.d_seg", minimum=0.0),
        )
        object.__setattr__(
            self,
            "d_pose",
            _finite(self.d_pose, "evaluation.d_pose", minimum=0.0),
        )
        object.__setattr__(
            self,
            "archive_bytes",
            _integer(self.archive_bytes, "evaluation.archive_bytes", minimum=1),
        )
        object.__setattr__(
            self,
            "archive_sha256",
            _sha256_text(self.archive_sha256, "evaluation.archive_sha256"),
        )
        object.__setattr__(
            self,
            "sample_count",
            _integer(self.sample_count, "evaluation.sample_count", minimum=1),
        )
        if (
            not isinstance(self.authority_marker, str)
            or not self.authority_marker
            or self.authority_marker.strip() != self.authority_marker
        ):
            raise TerminalPoseError("evaluation.authority_marker must be a nonempty stripped string")
        if type(self.realized) is not bool:
            raise TerminalPoseError("evaluation.realized must be bool")
        if self.custody_digest is not None:
            object.__setattr__(
                self,
                "custody_digest",
                _sha256_text(self.custody_digest, "evaluation.custody_digest"),
            )

    @property
    def full_n600(self) -> bool:
        return (
            self.realized
            and self.sample_count == FULL_N600_SAMPLE_COUNT
            and self.authority_marker == FULL_N600_POSE_AUTHORITY_MARKER
        )

    @property
    def joint_action(self) -> float:
        """Exact contest action recomputed from hard callback components."""

        return 100.0 * self.d_seg + math.sqrt(10.0 * self.d_pose) + 25.0 * self.archive_bytes / 37_545_489.0

    def to_payload(self) -> dict[str, object]:
        return {
            "pose6": self.pose6.tolist(),
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "sample_count": self.sample_count,
            "authority_marker": self.authority_marker,
            "custody_digest": self.custody_digest,
            "realized": self.realized,
            "joint_action": self.joint_action,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PoseJointEvaluation:
        return cls(
            pose6=np.asarray(payload["pose6"], dtype=np.float64),
            d_seg=payload["d_seg"],
            d_pose=payload["d_pose"],
            archive_bytes=payload["archive_bytes"],
            archive_sha256=payload["archive_sha256"],
            sample_count=payload["sample_count"],
            authority_marker=payload["authority_marker"],
            custody_digest=payload.get("custody_digest"),
            realized=payload["realized"],
        )


@dataclass(frozen=True)
class TerminalPoseGNConfig:
    relinearizations: int = 3
    damping: float = 1.0e-3
    amplitude_q8: int = 256
    line_search: tuple[float, ...] = (1.0, 0.5, 0.25)
    coefficient_limit: int = np.iinfo(np.int16).max
    authority_mode: PoseAuthorityMode = PoseAuthorityMode.STALE_REHEARSAL
    production_custody: ProductionPoseCustodyV1 | None = None
    resume_path: str | None = None

    def __post_init__(self) -> None:
        relinearizations = _integer(self.relinearizations, "config.relinearizations", minimum=2, maximum=3)
        object.__setattr__(self, "relinearizations", relinearizations)
        object.__setattr__(self, "damping", _finite(self.damping, "config.damping", minimum=0.0))
        if self.damping == 0.0:
            raise TerminalPoseError("config.damping must be positive")
        object.__setattr__(
            self,
            "amplitude_q8",
            _integer(
                self.amplitude_q8,
                "config.amplitude_q8",
                minimum=1,
                maximum=_MAX_AMPLITUDE_Q8,
            ),
        )
        if not isinstance(self.line_search, tuple) or not self.line_search:
            raise TerminalPoseError("config.line_search must be a nonempty tuple")
        line_search = tuple(_finite(value, "config.line_search value", minimum=0.0) for value in self.line_search)
        if any(value == 0.0 or value > 1.0 for value in line_search):
            raise TerminalPoseError("line-search values must be in (0, 1]")
        if tuple(sorted(set(line_search), reverse=True)) != line_search:
            raise TerminalPoseError("line-search values must be unique and descending")
        object.__setattr__(self, "line_search", line_search)
        object.__setattr__(
            self,
            "coefficient_limit",
            _integer(
                self.coefficient_limit,
                "config.coefficient_limit",
                minimum=1,
                maximum=np.iinfo(np.int16).max,
            ),
        )
        if not isinstance(self.authority_mode, PoseAuthorityMode):
            raise TerminalPoseError("config.authority_mode must be PoseAuthorityMode")
        if self.production_custody is not None and not isinstance(self.production_custody, ProductionPoseCustodyV1):
            raise TerminalPoseError("config.production_custody must be ProductionPoseCustodyV1")
        if self.resume_path is not None:
            resume = _stripped_text(self.resume_path, "config.resume_path")
            if not Path(resume).is_absolute():
                raise TerminalPoseError("config.resume_path must be absolute")
            object.__setattr__(self, "resume_path", resume)
        if self.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600:
            if self.production_custody is None:
                raise TerminalPoseError("production terminal pose requires typed custody")
            if self.resume_path is None:
                raise TerminalPoseError("production terminal pose requires an atomic resume path")


def _frozen_pair(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[0] != 2 or raw.shape[-1] != 3:
        raise TerminalPoseError("parent_pair must be uint8 [2,H,W,3]")
    if raw.shape[1] < 1 or raw.shape[2] < 1:
        raise TerminalPoseError("parent_pair spatial shape must be nonempty")
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.uint8).reshape(raw.shape)


def _rendered_basis(values: object, frame0_shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.asarray(values)
    expected_tail = tuple(frame0_shape)
    if (
        raw.dtype.kind not in ("i", "u", "f")
        or raw.ndim != 4
        or raw.shape[1:] != expected_tail
        or not 1 <= raw.shape[0] <= _MAX_PACKET_RANK
    ):
        raise TerminalPoseError("rendered basis must have shape [rank,H,W,3] matching frame 0")
    result = _immutable_array(raw, np.dtype(np.float64), "rendered_basis")
    if not np.all(np.isfinite(result)):
        raise TerminalPoseError("rendered basis must be finite")
    return result


def normalize_terminal_pose_basis(rendered_basis: np.ndarray) -> np.ndarray:
    """Decode-side zero-mean/unit-RMS normalization, independently per field."""

    raw = np.asarray(rendered_basis, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[-1] != 3 or raw.shape[0] == 0:
        raise TerminalPoseError("basis normalization requires [rank,H,W,3]")
    normalized = np.empty_like(raw)
    for index in range(raw.shape[0]):
        centered = raw[index] - float(np.mean(raw[index], dtype=np.float64))
        rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
        if not math.isfinite(rms) or rms <= 1.0e-12:
            raise TerminalPoseError("basis field has no normalizable amplitude")
        normalized[index] = centered / rms
    return _immutable_array(normalized, np.dtype(np.float64), "normalized_basis")


def realize_terminal_pose_pair(
    parent_pair: np.ndarray,
    rendered_basis: np.ndarray,
    quantized_coefficients: np.ndarray,
    *,
    amplitude_q8: int,
) -> np.ndarray:
    """Run the actual deterministic receiver path and freeze frame 1."""

    parent = _frozen_pair(parent_pair)
    raw_basis = _rendered_basis(rendered_basis, tuple(parent.shape[1:]))
    codes = _coefficients(quantized_coefficients, "quantized_coefficients")
    if codes.ndim != 1 or codes.shape[0] != raw_basis.shape[0]:
        raise TerminalPoseError("coefficient rank differs from rendered basis")
    amplitude_q8 = _integer(
        amplitude_q8,
        "amplitude_q8",
        minimum=1,
        maximum=_MAX_AMPLITUDE_Q8,
    )
    scale = amplitude_q8 / 256.0
    # Normalize inside every realized verdict, but accumulate one field at a
    # time so a full [rank,H,W,3] normalized copy is not required at n600
    # camera resolution.
    residual = np.zeros(parent.shape[1:], dtype=np.float64)
    for index, code in enumerate(np.asarray(codes, dtype=np.int64)):
        field = np.asarray(raw_basis[index], dtype=np.float64)
        centered = field - float(np.mean(field, dtype=np.float64))
        rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
        if not math.isfinite(rms) or rms <= 1.0e-12:
            raise TerminalPoseError("basis field has no normalizable amplitude")
        residual += float(code) * centered / rms
    frame0 = np.rint(np.clip(np.asarray(parent[0], dtype=np.float64) + scale * residual, 0.0, 255.0)).astype(np.uint8)
    result = np.stack((frame0, np.asarray(parent[1])), axis=0)
    if not np.array_equal(result[1], parent[1]):
        raise TerminalPoseError("terminal pose receiver changed frozen frame 1")
    return result


@dataclass(frozen=True)
class TerminalPoseStepTrace:
    iteration: int
    finite_difference_evaluations: int
    pose_mse_before: float
    pose_mse_after: float
    joint_action_before: float
    joint_action_after: float
    admitted: bool
    line_search_alpha: float | None
    coefficients_before: tuple[int, ...]
    coefficients_after: tuple[int, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "finite_difference_evaluations": self.finite_difference_evaluations,
            "pose_mse_before": self.pose_mse_before,
            "pose_mse_after": self.pose_mse_after,
            "joint_action_before": self.joint_action_before,
            "joint_action_after": self.joint_action_after,
            "admitted": self.admitted,
            "line_search_alpha": self.line_search_alpha,
            "coefficients_before": list(self.coefficients_before),
            "coefficients_after": list(self.coefficients_after),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TerminalPoseStepTrace:
        return cls(
            iteration=int(payload["iteration"]),
            finite_difference_evaluations=int(payload["finite_difference_evaluations"]),
            pose_mse_before=float(payload["pose_mse_before"]),
            pose_mse_after=float(payload["pose_mse_after"]),
            joint_action_before=float(payload["joint_action_before"]),
            joint_action_after=float(payload["joint_action_after"]),
            admitted=bool(payload["admitted"]),
            line_search_alpha=(None if payload["line_search_alpha"] is None else float(payload["line_search_alpha"])),
            coefficients_before=tuple(int(v) for v in payload["coefficients_before"]),
            coefficients_after=tuple(int(v) for v in payload["coefficients_after"]),
        )


@dataclass(frozen=True)
class TerminalPoseGNResult:
    authority_mode: PoseAuthorityMode
    initial_coefficients: np.ndarray
    final_coefficients: np.ndarray
    initial_evaluation: PoseJointEvaluation
    final_evaluation: PoseJointEvaluation
    target_pose6: np.ndarray
    steps: tuple[TerminalPoseStepTrace, ...]
    pose_mse_initial: float
    pose_mse_final: float
    strict_realized_improvement: bool
    governed_handoff_eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_coefficients",
            _coefficients(self.initial_coefficients, "result.initial_coefficients"),
        )
        object.__setattr__(
            self,
            "final_coefficients",
            _coefficients(self.final_coefficients, "result.final_coefficients"),
        )
        object.__setattr__(self, "target_pose6", _pose6(self.target_pose6, "result.target_pose6"))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "ddm_terminal_pose_gn.v2",
            "authority_mode": self.authority_mode.value,
            "initial_coefficients": self.initial_coefficients.tolist(),
            "final_coefficients": self.final_coefficients.tolist(),
            "target_pose6": self.target_pose6.tolist(),
            "pose_mse_initial": self.pose_mse_initial,
            "pose_mse_final": self.pose_mse_final,
            "joint_action_initial": self.initial_evaluation.joint_action,
            "joint_action_final": self.final_evaluation.joint_action,
            "joint_components_initial": {
                "d_seg": self.initial_evaluation.d_seg,
                "d_pose": self.initial_evaluation.d_pose,
                "archive_bytes": self.initial_evaluation.archive_bytes,
                "archive_sha256": self.initial_evaluation.archive_sha256,
                "custody_digest": self.initial_evaluation.custody_digest,
            },
            "joint_components_final": {
                "d_seg": self.final_evaluation.d_seg,
                "d_pose": self.final_evaluation.d_pose,
                "archive_bytes": self.final_evaluation.archive_bytes,
                "archive_sha256": self.final_evaluation.archive_sha256,
                "custody_digest": self.final_evaluation.custody_digest,
            },
            "steps": [step.to_payload() for step in self.steps],
            "strict_realized_improvement": self.strict_realized_improvement,
            "governed_handoff_eligible": self.governed_handoff_eligible,
            "production_accepted": False,
            "promotion_allowed": False,
            "score_claim": False,
            "pointer_moved": False,
        }


def _checked_ledger_payload(path: Path, *, expected_binding_sha256: str) -> dict[str, Any]:
    try:
        wrapper = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalPoseError(f"resume ledger is unreadable: {path}") from exc
    if set(wrapper) != {
        "binding_sha256",
        "payload",
        "payload_sha256",
        "schema",
    }:
        raise TerminalPoseError(f"resume ledger keys differ: {path}")
    if wrapper["schema"] != "ddm_terminal_pose_atomic_ledger.v1":
        raise TerminalPoseError(f"resume ledger schema differs: {path}")
    if wrapper["binding_sha256"] != expected_binding_sha256:
        raise TerminalPoseError(f"resume ledger binding differs: {path}")
    payload = wrapper["payload"]
    if sha256(_canonical_json(payload)).hexdigest() != wrapper["payload_sha256"]:
        raise TerminalPoseError(f"resume ledger checksum differs: {path}")
    if not isinstance(payload, dict):
        raise TerminalPoseError(f"resume ledger payload must be an object: {path}")
    return payload


def _atomic_checked_ledger_write(path: Path, *, binding_sha256: str, payload: dict[str, Any]) -> None:
    """Write once atomically; an existing path must be byte-logically identical."""

    binding_sha256 = _sha256_text(binding_sha256, "ledger.binding_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _checked_ledger_payload(path, expected_binding_sha256=binding_sha256)
        if existing != payload:
            raise TerminalPoseError(f"immutable resume ledger already differs: {path}")
        return
    wrapper = {
        "schema": "ddm_terminal_pose_atomic_ledger.v1",
        "binding_sha256": binding_sha256,
        "payload": payload,
        "payload_sha256": sha256(_canonical_json(payload)).hexdigest(),
    }
    encoded = _canonical_json(wrapper) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resume_result_payload(result: TerminalPoseGNResult) -> dict[str, Any]:
    return {
        "schema": "ddm_terminal_pose_completed_result.v1",
        "completed": True,
        "authority_mode": result.authority_mode.value,
        "initial_coefficients": result.initial_coefficients.tolist(),
        "final_coefficients": result.final_coefficients.tolist(),
        "initial_evaluation": result.initial_evaluation.to_payload(),
        "final_evaluation": result.final_evaluation.to_payload(),
        "target_pose6": result.target_pose6.tolist(),
        "steps": [step.to_payload() for step in result.steps],
        "pose_mse_initial": result.pose_mse_initial,
        "pose_mse_final": result.pose_mse_final,
        "strict_realized_improvement": result.strict_realized_improvement,
        "governed_handoff_eligible": result.governed_handoff_eligible,
    }


def _result_from_resume_payload(payload: dict[str, Any]) -> TerminalPoseGNResult:
    if payload.get("schema") != "ddm_terminal_pose_completed_result.v1" or payload.get("completed") is not True:
        raise TerminalPoseError("completed resume payload marker differs")
    return TerminalPoseGNResult(
        authority_mode=PoseAuthorityMode(payload["authority_mode"]),
        initial_coefficients=np.asarray(payload["initial_coefficients"], dtype=np.int16),
        final_coefficients=np.asarray(payload["final_coefficients"], dtype=np.int16),
        initial_evaluation=PoseJointEvaluation.from_payload(payload["initial_evaluation"]),
        final_evaluation=PoseJointEvaluation.from_payload(payload["final_evaluation"]),
        target_pose6=np.asarray(payload["target_pose6"], dtype=np.float64),
        steps=tuple(TerminalPoseStepTrace.from_payload(step) for step in payload["steps"]),
        pose_mse_initial=float(payload["pose_mse_initial"]),
        pose_mse_final=float(payload["pose_mse_final"]),
        strict_realized_improvement=bool(payload["strict_realized_improvement"]),
        governed_handoff_eligible=bool(payload["governed_handoff_eligible"]),
    )


def _mse(pose: np.ndarray, target: np.ndarray) -> float:
    difference = np.asarray(pose, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(difference * difference, dtype=np.float64))


def solve_terminal_pose_gn(
    parent_pair: np.ndarray,
    target_pose6: np.ndarray,
    render_basis: Callable[[int, str, tuple[int, int, int]], np.ndarray],
    artifact_for_coefficients: Callable[[np.ndarray], TerminalPoseCandidateArtifact],
    score_realized_candidate: Callable[[np.ndarray, TerminalPoseCandidateArtifact], PoseJointEvaluation],
    *,
    seed: int,
    basis_selector: str,
    config: TerminalPoseGNConfig,
    initial_coefficients: np.ndarray | None = None,
    pair_index: int = 0,
) -> TerminalPoseGNResult:
    """Solve the six target equations with quantized realized GN verdicts."""

    if not isinstance(config, TerminalPoseGNConfig):
        raise TerminalPoseError("config must be TerminalPoseGNConfig")
    if not callable(render_basis) or not callable(artifact_for_coefficients) or not callable(score_realized_candidate):
        raise TerminalPoseError("render, artifact, and scorer callbacks must be callable")
    seed = _integer(seed, "seed", minimum=0, maximum=2**64 - 1)
    basis_selector = _selector(basis_selector)
    parent = _frozen_pair(parent_pair)
    target = _pose6(target_pose6, "target_pose6")
    rendered = _rendered_basis(
        render_basis(seed, basis_selector, tuple(parent.shape[1:])),
        tuple(parent.shape[1:]),
    )
    rank = rendered.shape[0]
    pair_index = _integer(pair_index, "pair_index", minimum=0, maximum=_MAX_PACKET_PAIRS - 1)
    if initial_coefficients is None:
        current_codes = np.zeros(rank, dtype=np.int16)
    else:
        current_codes = np.asarray(_coefficients(initial_coefficients, "initial_coefficients"), dtype=np.int16).copy()
        if current_codes.ndim != 1 or current_codes.shape != (rank,):
            raise TerminalPoseError("initial coefficient rank differs from basis")
    initial_codes = current_codes.copy()

    custody_digest = None if config.production_custody is None else config.production_custody.digest
    binding_payload = {
        "schema": "ddm_terminal_pose_solver_binding.v1",
        "parent_pair_sha256": sha256(parent.tobytes()).hexdigest(),
        "target_pose6": target.tolist(),
        "rendered_basis_sha256": sha256(rendered.tobytes()).hexdigest(),
        "seed": seed,
        "basis_selector": basis_selector,
        "amplitude_q8": config.amplitude_q8,
        "rank": rank,
        "pair_index": pair_index,
        "initial_coefficients": initial_codes.tolist(),
        "relinearizations": config.relinearizations,
        "damping": config.damping,
        "line_search": list(config.line_search),
        "coefficient_limit": config.coefficient_limit,
        "authority_mode": config.authority_mode.value,
        "production_custody_digest": custody_digest,
    }
    solver_binding_sha256 = sha256(_canonical_json(binding_payload)).hexdigest()
    resume_root = None if config.resume_path is None else Path(config.resume_path)
    if config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600:
        if resume_root is None:
            raise AssertionError("production resume validation was bypassed")
        _atomic_checked_ledger_write(
            resume_root / "manifest.json",
            binding_sha256=solver_binding_sha256,
            payload=binding_payload,
        )
        completed_path = resume_root / "completed.json"
        if completed_path.exists():
            completed = _checked_ledger_payload(completed_path, expected_binding_sha256=solver_binding_sha256)
            result = _result_from_resume_payload(completed)
            if result.authority_mode is not config.authority_mode or not np.array_equal(result.target_pose6, target):
                raise TerminalPoseError("completed resume result differs from solver")
            return result

    def evaluate(codes: np.ndarray) -> tuple[PoseJointEvaluation, float]:
        codes = np.asarray(_coefficients(codes, "candidate coefficients"), dtype=np.int16)
        if codes.ndim != 1 or codes.shape != (rank,):
            raise TerminalPoseError("candidate coefficient rank differs")
        realized_pair = realize_terminal_pose_pair(
            parent,
            rendered,
            codes,
            amplitude_q8=config.amplitude_q8,
        )
        if not np.array_equal(realized_pair[1], parent[1]):
            raise TerminalPoseError("realized candidate changed frozen frame 1")
        artifact = artifact_for_coefficients(codes.copy())
        if not isinstance(artifact, TerminalPoseCandidateArtifact):
            raise TerminalPoseError("artifact callback must return TerminalPoseCandidateArtifact")
        if (
            config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600
            and artifact.scope is not CandidateArtifactScope.FULL_OUTER_ARCHIVE
        ):
            raise TerminalPoseError("production terminal pose requires a full outer archive artifact")
        parsed = parse_terminal_pose_packet(artifact.terminal_packet)
        if parsed.seed != seed or parsed.basis_selector != basis_selector or parsed.amplitude_q8 != config.amplitude_q8:
            raise TerminalPoseError("packet seed/basis selector/amplitude differs from solver")
        if pair_index >= parsed.coefficients.shape[0]:
            raise TerminalPoseError("packet does not contain requested pair index")
        if parsed.coefficients.shape[1] != rank or not np.array_equal(parsed.coefficients[pair_index], codes):
            raise TerminalPoseError("packet parse-back coefficients differ from candidate")
        verdict_key_payload = {
            "solver_binding_sha256": solver_binding_sha256,
            "artifact_binding_sha256": artifact.binding_sha256,
            "coefficients": codes.tolist(),
        }
        verdict_key = sha256(_canonical_json(verdict_key_payload)).hexdigest()
        verdict_path = None if resume_root is None else resume_root / "verdicts" / f"{verdict_key}.json"
        if (
            config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600
            and verdict_path is not None
            and verdict_path.exists()
        ):
            cached = _checked_ledger_payload(verdict_path, expected_binding_sha256=solver_binding_sha256)
            if cached.get("artifact") != artifact.to_payload() or cached.get("coefficients") != codes.tolist():
                raise TerminalPoseError("cached verdict artifact binding differs")
            evaluation = PoseJointEvaluation.from_payload(cached["evaluation"])
            cached_mse = float(cached["pose_mse"])
            if cached_mse != _mse(evaluation.pose6, target):
                raise TerminalPoseError("cached verdict pose MSE differs")
        else:
            evaluation = score_realized_candidate(realized_pair, artifact)
            cached_mse = _mse(evaluation.pose6, target) if isinstance(evaluation, PoseJointEvaluation) else math.nan
        if not isinstance(evaluation, PoseJointEvaluation):
            raise TerminalPoseError("scorer callback must return PoseJointEvaluation")
        if not evaluation.realized:
            raise TerminalPoseError("scorer callback returned a non-realized evaluation")
        if evaluation.archive_bytes != artifact.archive_bytes or evaluation.archive_sha256 != artifact.archive_sha256:
            raise TerminalPoseError("scorer archive bytes/SHA differ from candidate artifact")
        if config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600 and (
            not evaluation.full_n600 or evaluation.custody_digest != custody_digest
        ):
            raise TerminalPoseError("production terminal pose requires matching full-n600 custody")
        pose_mse = _mse(evaluation.pose6, target)
        if cached_mse != pose_mse:
            raise TerminalPoseError("verdict pose MSE differs after validation")
        if (
            config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600
            and verdict_path is not None
            and not verdict_path.exists()
        ):
            _atomic_checked_ledger_write(
                verdict_path,
                binding_sha256=solver_binding_sha256,
                payload={
                    "artifact": artifact.to_payload(),
                    "coefficients": codes.tolist(),
                    "evaluation": evaluation.to_payload(),
                    "pose_mse": pose_mse,
                },
            )
        return evaluation, pose_mse

    current_evaluation, current_mse = evaluate(current_codes)
    initial_evaluation = current_evaluation
    initial_mse = current_mse
    traces: list[TerminalPoseStepTrace] = []
    start_iteration = 0
    if config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600 and resume_root is not None:
        for iteration in range(config.relinearizations):
            iteration_path = resume_root / "iterations" / f"iteration_{iteration:03d}.json"
            if not iteration_path.exists():
                break
            cached_iteration = _checked_ledger_payload(iteration_path, expected_binding_sha256=solver_binding_sha256)
            if int(cached_iteration["iteration"]) != iteration:
                raise TerminalPoseError("cached iteration ordinal differs")
            trace = TerminalPoseStepTrace.from_payload(cached_iteration["trace"])
            if trace.iteration != iteration:
                raise TerminalPoseError("cached iteration trace differs")
            current_codes = np.asarray(
                _coefficients(
                    cached_iteration["current_coefficients"],
                    "cached current coefficients",
                ),
                dtype=np.int16,
            ).copy()
            if current_codes.shape != (rank,):
                raise TerminalPoseError("cached iteration coefficient rank differs")
            current_evaluation = PoseJointEvaluation.from_payload(cached_iteration["current_evaluation"])
            current_mse = float(cached_iteration["current_pose_mse"])
            if (
                not current_evaluation.full_n600
                or current_evaluation.custody_digest != custody_digest
                or current_mse != _mse(current_evaluation.pose6, target)
            ):
                raise TerminalPoseError("cached iteration authority differs")
            traces.append(trace)
            start_iteration = iteration + 1

    for iteration in range(start_iteration, config.relinearizations):
        before_codes = current_codes.copy()
        before_evaluation = current_evaluation
        before_mse = current_mse
        jacobian = np.empty((6, rank), dtype=np.float64)
        fd_evaluations = 0
        for coordinate in range(rank):
            plus = current_codes.astype(np.int64)
            minus = current_codes.astype(np.int64)
            plus[coordinate] = min(config.coefficient_limit, int(plus[coordinate]) + 1)
            minus[coordinate] = max(-config.coefficient_limit, int(minus[coordinate]) - 1)
            denominator = int(plus[coordinate] - minus[coordinate])
            if denominator == 0:
                raise TerminalPoseError("finite-difference coefficient is pinned")
            plus_evaluation, _ = evaluate(plus.astype(np.int16))
            minus_evaluation, _ = evaluate(minus.astype(np.int16))
            fd_evaluations += 2
            jacobian[:, coordinate] = (plus_evaluation.pose6 - minus_evaluation.pose6) / float(denominator)

        residual = target - current_evaluation.pose6
        normal = jacobian.T @ jacobian + config.damping * np.eye(rank)
        rhs = jacobian.T @ residual
        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        if not np.all(np.isfinite(delta)):
            raise TerminalPoseError("GN solve produced nonfinite coefficients")

        best: tuple[float, np.ndarray, PoseJointEvaluation, float] | None = None
        seen: set[bytes] = {current_codes.tobytes()}
        for alpha in config.line_search:
            candidate_i64 = np.rint(current_codes.astype(np.float64) + alpha * delta).astype(np.int64)
            candidate_i64 = np.clip(
                candidate_i64,
                -config.coefficient_limit,
                config.coefficient_limit,
            )
            candidate_codes = candidate_i64.astype(np.int16)
            digest = candidate_codes.tobytes()
            if digest in seen:
                continue
            seen.add(digest)
            candidate_evaluation, candidate_mse = evaluate(candidate_codes)
            if (
                candidate_mse < before_mse
                and candidate_evaluation.joint_action < before_evaluation.joint_action
                and (best is None or candidate_evaluation.joint_action < best[2].joint_action)
            ):
                best = (
                    alpha,
                    candidate_codes.copy(),
                    candidate_evaluation,
                    candidate_mse,
                )

        if best is None:
            admitted = False
            alpha_used = None
        else:
            admitted = True
            alpha_used, current_codes, current_evaluation, current_mse = best
        trace = TerminalPoseStepTrace(
            iteration=iteration,
            finite_difference_evaluations=fd_evaluations,
            pose_mse_before=before_mse,
            pose_mse_after=current_mse,
            joint_action_before=before_evaluation.joint_action,
            joint_action_after=current_evaluation.joint_action,
            admitted=admitted,
            line_search_alpha=alpha_used,
            coefficients_before=tuple(int(value) for value in before_codes),
            coefficients_after=tuple(int(value) for value in current_codes),
        )
        traces.append(trace)
        if config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600 and resume_root is not None:
            _atomic_checked_ledger_write(
                resume_root / "iterations" / f"iteration_{iteration:03d}.json",
                binding_sha256=solver_binding_sha256,
                payload={
                    "iteration": iteration,
                    "trace": trace.to_payload(),
                    "current_coefficients": current_codes.tolist(),
                    "current_evaluation": current_evaluation.to_payload(),
                    "current_pose_mse": current_mse,
                },
            )

    strict_improvement = current_mse < initial_mse and current_evaluation.joint_action < initial_evaluation.joint_action
    governed_handoff_eligible = (
        strict_improvement
        and config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600
        and current_evaluation.full_n600
        and current_evaluation.custody_digest == custody_digest
    )
    result = TerminalPoseGNResult(
        authority_mode=config.authority_mode,
        initial_coefficients=initial_codes,
        final_coefficients=current_codes,
        initial_evaluation=initial_evaluation,
        final_evaluation=current_evaluation,
        target_pose6=target,
        steps=tuple(traces),
        pose_mse_initial=initial_mse,
        pose_mse_final=current_mse,
        strict_realized_improvement=strict_improvement,
        governed_handoff_eligible=governed_handoff_eligible,
    )
    if config.authority_mode is PoseAuthorityMode.PRODUCTION_FULL_N600 and resume_root is not None:
        _atomic_checked_ledger_write(
            resume_root / "completed.json",
            binding_sha256=solver_binding_sha256,
            payload=_resume_result_payload(result),
        )
    return result


__all__ = [
    "FULL_N600_POSE_AUTHORITY_MARKER",
    "STALE_POSE_REHEARSAL_AUTHORITY_MARKER",
    "CandidateArtifactScope",
    "ContestAxis",
    "PoseAuthorityMode",
    "PoseJointEvaluation",
    "ProductionPoseCustodyV1",
    "TerminalPoseCandidateArtifact",
    "TerminalPoseError",
    "TerminalPoseGNConfig",
    "TerminalPoseGNResult",
    "TerminalPosePacketV1",
    "TerminalPoseStepTrace",
    "normalize_terminal_pose_basis",
    "parse_terminal_pose_packet",
    "realize_terminal_pose_pair",
    "serialize_terminal_pose_packet",
    "solve_terminal_pose_gn",
]
