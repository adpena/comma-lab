# SPDX-License-Identifier: MIT
"""Counted DM4-to-J5 application operator.

DM4's scorer-recursive rows are proposal descriptors, not archive mutations.
This module maps one descriptor into the existing integer J5 grammar by
enumerating exact receiver secants, applying the measured MS4d pair/bucket
curvature, and re-canonicalizing the selected effect through the #580
``range(A)`` projector before commit.

The local curvature is authoritative only in the MS4d post-R penultimate-head
quotient.  Its Newton step is therefore used as a proposal-ordering model; the
only acceptance authority remains exact receiver parse-back plus frozen-scorer
n600 replay.  V16/v17 have no transferable J5-coordinate validity curve, so
this operator admits no hand-set shrink/grow constants: the conservative smoke
radius is exactly one smallest receiver quantum per coordinate, with no reuse.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from tac.boundary_math.range_a_projection import apply_projection
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import realize_solve_camera

if TYPE_CHECKING:
    from tac.optimization.ddm_dm4_j5_adapter import DM4J5ProposalV1

SCHEMA: Final = "ddm_dm4_j5_counted_application.v1"
CONFIG_SCHEMA: Final = "ddm_dm4_j5_counted_application_config.v1"
MS4D_SCHEMA: Final = "ddm_seg_metric_custody.direct_scorer_intrinsic.v2"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
VALIDITY_GAP: Final = "J5_BUCKET_VALIDITY_RADIUS_CURVE_ABSENT_NO_SHRINK_GROW_TRANSFER"
HORIZON_GAP: Final = "J5_NCDE_REENTRY_TIME_CUSTODY_ABSENT_USING_CANONICAL_WINDOW_12"
RANGE_GAUGE_POLICY: Final = (
    "fp64_P_range(A)_then_nearest_unused_integer_J5_secant_with_stable_coordinate_tie_break"
)


class DDMCountedApplicationError(ValueError):
    """Fail-closed malformed custody, geometry, or application state."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _bound_bytes(path: Path, digest: str, byte_count: int | None = None) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise DDMCountedApplicationError(f"bound regular file is unavailable: {path}")
    payload = path.read_bytes()
    if byte_count is not None and len(payload) != int(byte_count):
        raise DDMCountedApplicationError(
            f"bound file byte count differs: {path}: {len(payload)} != {byte_count}"
        )
    actual = _sha256(payload)
    if actual != digest:
        raise DDMCountedApplicationError(
            f"bound file SHA-256 differs: {path}: {actual} != {digest}"
        )
    return payload


@dataclass(frozen=True, slots=True)
class BoundArtifactV1:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, name: str) -> BoundArtifactV1:
        if not isinstance(payload, Mapping):
            raise DDMCountedApplicationError(f"{name} binding must be a mapping")
        path = payload.get("path")
        digest = payload.get("sha256")
        byte_count = payload.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise DDMCountedApplicationError(f"{name} binding is malformed")
        return cls(path=path, sha256=digest, bytes=byte_count)

    def resolve(self, repo_root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else repo_root / path

    def read(self, repo_root: Path) -> bytes:
        return _bound_bytes(self.resolve(repo_root), self.sha256, self.bytes)

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class DDMCountedApplicationConfigV1:
    """Hash-bound wrapper around the immutable J8e ticket and J8f inputs."""

    config_path: str
    lane_id: str
    run_id: str
    output_root: str
    torch_threads: int
    smoke_horizon: int
    source_bindings: Mapping[str, BoundArtifactV1]
    execution_allowed: bool

    @classmethod
    def from_path(cls, path: str | Path) -> DDMCountedApplicationConfigV1:
        config_path = Path(path).resolve()
        raw = config_path.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != CONFIG_SCHEMA:
            raise DDMCountedApplicationError("counted-application config schema differs")
        required = {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "pointer": f"{POINTER} UNMOVED",
            "evidence_axis": EVIDENCE_AXIS,
            "main_landing_review_required": True,
            "torch_threads": 4,
            "smoke_horizon": 12,
            "horizon_derivation": HORIZON_GAP,
            "validity_policy": VALIDITY_GAP,
            "range_gauge_policy": RANGE_GAUGE_POLICY,
        }
        drift = {
            key: (payload.get(key), expected)
            for key, expected in required.items()
            if payload.get(key) != expected
        }
        if drift:
            raise DDMCountedApplicationError(
                f"counted-application false-authority/dynamic contract differs: {drift}"
            )
        if payload.get("execution_allowed") not in {False, True}:
            raise DDMCountedApplicationError("execution_allowed must be an exact boolean")
        bindings_payload = payload.get("source_bindings")
        required_bindings = {
            "authority",
            "j8e_ticket",
            "dm4_config",
            "dm4_receipt",
            "ms4d_direct_metric",
            "step4_ticket",
            "step4_checkpoint",
            "step4_verdict",
            "v17_validity_law",
            "v17_validity_receipt",
            "ncde_observer",
            "ncde_event_wiring",
            "range_a_projector",
            "operator_source",
            "runner_source",
        }
        if not isinstance(bindings_payload, Mapping) or set(bindings_payload) != required_bindings:
            raise DDMCountedApplicationError("counted-application source-binding set differs")
        bindings = {
            name: BoundArtifactV1.from_payload(value, name=name)
            for name, value in bindings_payload.items()
        }
        lane_id = payload.get("lane_id")
        run_id = payload.get("run_id")
        output_root = payload.get("output_root")
        if not all(isinstance(value, str) and value for value in (lane_id, run_id, output_root)):
            raise DDMCountedApplicationError("counted-application lane/run/output identity differs")
        config = cls(
            config_path=str(config_path),
            lane_id=str(lane_id),
            run_id=str(run_id),
            output_root=str(output_root),
            torch_threads=4,
            smoke_horizon=12,
            source_bindings=bindings,
            execution_allowed=bool(payload["execution_allowed"]),
        )
        # Config bytes are intentionally not self-bound. Every executable and
        # upstream evidence surface is, and the receipt records config SHA.
        if not raw:
            raise DDMCountedApplicationError("counted-application config is empty")
        return config

    @property
    def repo_root(self) -> Path:
        return Path(self.config_path).parents[3]

    def validate_all_bindings(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for name, binding in sorted(self.source_bindings.items()):
            payload = binding.read(self.repo_root)
            rows[name] = {
                **binding.to_payload(),
                "resolved_path": str(binding.resolve(self.repo_root)),
                "validated": True,
                "observed_sha256": _sha256(payload),
            }
        return rows

    def typed_hash(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "schema": CONFIG_SCHEMA,
                    "lane_id": self.lane_id,
                    "run_id": self.run_id,
                    "output_root": self.output_root,
                    "torch_threads": self.torch_threads,
                    "smoke_horizon": self.smoke_horizon,
                    "source_bindings": {
                        key: value.to_payload()
                        for key, value in sorted(self.source_bindings.items())
                    },
                    "execution_allowed": self.execution_allowed,
                    "validity_policy": VALIDITY_GAP,
                    "horizon_derivation": HORIZON_GAP,
                    "range_gauge_policy": RANGE_GAUGE_POLICY,
                    "score_claim": False,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class MS4DPairMetricV1:
    pair_id: int
    bucket_id: str
    hessian: np.ndarray
    adjoint: np.ndarray
    rank4_pair_normal: np.ndarray
    source_sha256: str
    support_count: int

    def __post_init__(self) -> None:
        hessian = np.asarray(self.hessian, dtype=np.float64)
        adjoint = np.asarray(self.adjoint, dtype=np.float64)
        normal = np.asarray(self.rank4_pair_normal, dtype=np.float64)
        if (
            hessian.shape != (4, 4)
            or adjoint.shape != (4,)
            or normal.shape != (4,)
            or not np.all(np.isfinite(hessian))
            or not np.all(np.isfinite(adjoint))
            or not np.all(np.isfinite(normal))
            or not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12)
            or float(np.linalg.eigvalsh(hessian).min()) < -1e-9
            or float(np.linalg.norm(normal)) <= 0.0
            or self.support_count <= 0
        ):
            raise DDMCountedApplicationError("MS4d pair metric geometry differs")
        object.__setattr__(self, "hessian", np.ascontiguousarray(hessian))
        object.__setattr__(self, "adjoint", np.ascontiguousarray(adjoint))
        object.__setattr__(self, "rank4_pair_normal", np.ascontiguousarray(normal))

    def newton_step(self) -> np.ndarray:
        """Return the minimum-norm preconditioned step ``-H^+ g``.

        The pseudoinverse cutoff is derived from machine precision and matrix
        dimension; it is not a tuned learning-rate or damping constant.
        """

        cutoff = np.finfo(np.float64).eps * max(self.hessian.shape)
        return np.ascontiguousarray(
            -np.linalg.pinv(self.hessian, rcond=cutoff) @ self.adjoint,
            dtype=np.float64,
        )

    def to_payload(self) -> dict[str, Any]:
        spectrum = np.linalg.eigvalsh(self.hessian)
        step = self.newton_step()
        return {
            "pair_id": self.pair_id,
            "bucket_id": self.bucket_id,
            "source_sha256": self.source_sha256,
            "support_count": self.support_count,
            "coordinate_domain": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
            "hessian": self.hessian.tolist(),
            "adjoint": self.adjoint.tolist(),
            "rank4_pair_normal": self.rank4_pair_normal.tolist(),
            "eigenvalues_ascending": spectrum.tolist(),
            "newton_step": step.tolist(),
            "newton_step_l2": float(np.linalg.norm(step)),
            "step_rule": "minimum_norm_-pinv(H)g_machine_epsilon_rank_cutoff",
            "global_learning_rate_used": False,
        }


def load_ms4d_pair_metric(
    *,
    path: str | Path,
    sha256: str,
    pair_id: int,
    bucket_id: str,
) -> MS4DPairMetricV1:
    raw = _bound_bytes(Path(path), sha256)
    payload = json.loads(raw)
    blocks = payload.get("direct_blocks")
    if payload.get("schema") != MS4D_SCHEMA or not isinstance(blocks, list):
        raise DDMCountedApplicationError("MS4d direct metric schema differs")
    matches = [
        row
        for row in blocks
        if isinstance(row, Mapping)
        and row.get("pair_id") == int(pair_id)
        and row.get("bucket_id") == bucket_id
    ]
    if len(matches) != 1:
        raise DDMCountedApplicationError(
            f"MS4d direct metric join differs for pair={pair_id} bucket={bucket_id}: {len(matches)}"
        )
    row = matches[0]
    if (
        row.get("metric_mode") != "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT"
        or row.get("secant_status") != "NOT_APPLICABLE_DIRECT_SCORER_INTRINSIC_NO_ACTUATOR"
        or row.get("support_status") != "MEASURED_EXACT_PF2_EVENT_INDEX"
    ):
        raise DDMCountedApplicationError("MS4d direct metric authority boundary differs")
    return MS4DPairMetricV1(
        pair_id=int(pair_id),
        bucket_id=bucket_id,
        hessian=np.asarray(row["composite_r_model_hessian"], dtype=np.float64),
        adjoint=np.asarray(row["composite_r_adjoint_readback"], dtype=np.float64),
        rank4_pair_normal=np.asarray(row["rank4_pair_normal"], dtype=np.float64),
        source_sha256=sha256,
        support_count=int(row["support_count"]),
    )


@dataclass(frozen=True, slots=True)
class SparseJ5CoordinateEffectV1:
    """One exact `+1/-1` receiver secant represented sparsely in camera bytes."""

    coordinate_index: int
    coordinate_name: str
    direction: int
    pair_id: int
    flat_indices: np.ndarray
    values: np.ndarray
    camera_shape: tuple[int, int, int, int]
    archive_bytes: int
    archive_sha256: str
    archive_byte_delta: int
    changed_channel_values: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.flat_indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.int16)
        size = int(np.prod(self.camera_shape, dtype=np.int64))
        if (
            self.coordinate_index < 0
            or not self.coordinate_name
            or self.direction not in {-1, 1}
            or not 0 <= self.pair_id < 600
            or len(self.camera_shape) != 4
            or tuple(self.camera_shape[1:]) != (874, 1164, 3)
            or indices.ndim != 1
            or values.ndim != 1
            or indices.size == 0
            or indices.size != values.size
            or np.any(indices < 0)
            or np.any(indices >= size)
            or np.any(np.diff(indices) <= 0)
            or np.any(values == 0)
            or self.changed_channel_values != int(indices.size)
            or self.archive_bytes <= 0
            or len(self.archive_sha256) != 64
        ):
            raise DDMCountedApplicationError("sparse J5 coordinate effect differs")
        object.__setattr__(self, "flat_indices", np.ascontiguousarray(indices))
        object.__setattr__(self, "values", np.ascontiguousarray(values))

    def norm_sq(self) -> float:
        values = self.values.astype(np.float64)
        return float(np.dot(values, values))

    def dot_dense(self, dense: np.ndarray) -> float:
        value = np.asarray(dense, dtype=np.float64)
        if value.shape != self.camera_shape or not np.all(np.isfinite(value)):
            raise DDMCountedApplicationError("dense projection target geometry differs")
        return float(np.dot(value.reshape(-1)[self.flat_indices], self.values.astype(np.float64)))

    def dense(self) -> np.ndarray:
        output = np.zeros(int(np.prod(self.camera_shape)), dtype=np.float64)
        output[self.flat_indices] = self.values
        return output.reshape(self.camera_shape)

    def to_payload(self) -> dict[str, Any]:
        return {
            "coordinate_index": self.coordinate_index,
            "coordinate_name": self.coordinate_name,
            "direction": self.direction,
            "pair_id": self.pair_id,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "archive_byte_delta": self.archive_byte_delta,
            "changed_channel_values": self.changed_channel_values,
            "realized_uint8_delta_l1": int(
                np.abs(self.values.astype(np.int64)).sum(dtype=np.int64)
            ),
            "realized_uint8_delta_l2": math.sqrt(self.norm_sq()),
            "delta_sha256_int64_indices_int16_values": _sha256(
                self.flat_indices.astype("<i8", copy=False).tobytes()
                + self.values.astype("<i2", copy=False).tobytes()
            ),
        }


def _receiver_reemit_exact(archive: bytes) -> bool:
    from tac.optimization.direct_description_joint_descent import lift_v15_archive

    return lift_v15_archive(archive).exact_reemit() == archive


def enumerate_j5_coordinate_effects(
    *,
    lift: Any,
    theta: np.ndarray,
    pair_id: int,
    include_lane_programs: bool,
) -> tuple[bytes, np.ndarray, tuple[SparseJ5CoordinateEffectV1, ...], dict[str, Any]]:
    """Compile every pair-effective smallest-lattice J5 secant.

    Candidate arrays are never retained: only sparse realized uint8 deltas and
    exact archive custody survive the loop.
    """

    from tac.optimization.direct_description_joint_descent import (
        compile_parameterized_archive,
        parameter_group_indices,
        realize_parameter_theta,
        receive_joint_descent_archive,
    )

    pair_id = int(pair_id)
    realized = realize_parameter_theta(lift, theta)
    base_archive, base_realized = compile_parameterized_archive(
        lift,
        realized,
        include_lane_programs=include_lane_programs,
    )
    if not _receiver_reemit_exact(base_archive):
        raise DDMCountedApplicationError("base J5 archive does not parse/re-emit exactly")
    base_camera = receive_joint_descent_archive(
        base_archive, verify_member_effects=False
    ).render_camera_pairs((pair_id,))
    if base_camera.shape != (1, 2, 874, 1164, 3) or base_camera.dtype != np.uint8:
        raise DDMCountedApplicationError("base J5 camera geometry differs")
    pair_camera = np.ascontiguousarray(base_camera[0])
    groups = parameter_group_indices(lift)
    selected: list[int] = []
    for track_index, track in enumerate(lift.g1.tracks):
        if any(
            int(lift.g1.knots[knot_index].pair_index) == pair_id
            for knot_index in track.knot_indices
        ):
            selected.extend((2 * track_index, 2 * track_index + 1))
    if include_lane_programs:
        selected.extend(groups["lane_program"])
    selected.extend(groups["shared_template_dof"])
    selected = sorted(set(selected))
    effects: list[SparseJ5CoordinateEffectV1] = []
    refusals: list[dict[str, Any]] = []
    for coordinate_index in selected:
        for direction in (-1, 1):
            probe = base_realized.copy()
            probe[coordinate_index] += np.float32(direction)
            try:
                archive, probe_realized = compile_parameterized_archive(
                    lift,
                    probe,
                    include_lane_programs=include_lane_programs,
                )
                if int(probe_realized[coordinate_index] - base_realized[coordinate_index]) != direction:
                    raise DDMCountedApplicationError("J5 coordinate realization did not move one quantum")
                if archive == base_archive or not _receiver_reemit_exact(archive):
                    raise DDMCountedApplicationError(
                        "J5 coordinate archive is byte-identical or fails exact parse-back"
                    )
                camera = receive_joint_descent_archive(
                    archive, verify_member_effects=False
                ).render_camera_pairs((pair_id,))[0]
            except Exception as exc:  # preserve exact per-coordinate refusal
                refusals.append(
                    {
                        "coordinate_index": coordinate_index,
                        "coordinate_name": lift.parameter_names[coordinate_index],
                        "direction": direction,
                        "reason": str(exc),
                    }
                )
                continue
            delta = camera.astype(np.int16) - pair_camera.astype(np.int16)
            indices = np.flatnonzero(delta.reshape(-1))
            if indices.size == 0:
                refusals.append(
                    {
                        "coordinate_index": coordinate_index,
                        "coordinate_name": lift.parameter_names[coordinate_index],
                        "direction": direction,
                        "reason": "PAIR_LOCAL_REALIZED_UINT8_DELTA_ZERO",
                    }
                )
                continue
            effects.append(
                SparseJ5CoordinateEffectV1(
                    coordinate_index=coordinate_index,
                    coordinate_name=str(lift.parameter_names[coordinate_index]),
                    direction=direction,
                    pair_id=pair_id,
                    flat_indices=indices,
                    values=delta.reshape(-1)[indices],
                    camera_shape=tuple(int(value) for value in delta.shape),
                    archive_bytes=len(archive),
                    archive_sha256=_sha256(archive),
                    archive_byte_delta=len(archive) - len(base_archive),
                    changed_channel_values=int(indices.size),
                )
            )
    if not effects:
        raise DDMCountedApplicationError(
            f"no receiver-effective J5 coordinate exists at pair {pair_id}"
        )
    ordered = tuple(
        sorted(
            effects,
            key=lambda row: (row.coordinate_name, row.coordinate_index, row.direction),
        )
    )
    receipt = {
        "schema": "ddm_j5_pair_coordinate_effect_inventory.v1",
        "pair_id": pair_id,
        "coordinate_count_total": len(lift.parameter_names),
        "candidate_coordinate_count": len(selected),
        "realized_direction_count": len(ordered),
        "refusal_count": len(refusals),
        "refusals": refusals,
        "base_archive_bytes": len(base_archive),
        "base_archive_sha256": _sha256(base_archive),
        "base_camera_sha256": _sha256(pair_camera.tobytes()),
        "include_lane_programs": bool(include_lane_programs),
        "inventory_manifest_sha256": _sha256(
            _canonical_bytes([effect.to_payload() for effect in ordered])
        ),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return base_archive, pair_camera, ordered, receipt


def descriptor_camera_delta(
    *,
    proposal: DM4J5ProposalV1,
    predictor_planes: np.ndarray,
    target_planes: np.ndarray,
    kernel: FullResizeKernel | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Realize the exact DM4 descriptor direction in camera uint8 coordinates."""

    base = np.asarray(predictor_planes)
    target = np.asarray(target_planes)
    if (
        base.shape != (2, 384, 512, 3)
        or target.shape != base.shape
        or base.dtype != np.uint8
        or target.dtype != np.uint8
    ):
        raise DDMCountedApplicationError("DM4 descriptor planes differ from exact pair geometry")
    candidate = proposal.candidate
    if candidate.get("mechanism") != "scorer_recursive_target":
        raise DDMCountedApplicationError("DM4 descriptor mechanism differs")
    indices = np.asarray(
        proposal.support_footprint.get("stem_block_indices", ()), dtype=np.int64
    )
    block_h, block_w = 192, 256
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= block_h * block_w)
    ):
        raise DDMCountedApplicationError("DM4 descriptor stem support differs")
    observed = _sha256(indices.astype("<u4", copy=False).tobytes())
    if observed != proposal.support_footprint.get(
        "stem_block_indices_sha256_uint32le"
    ):
        raise DDMCountedApplicationError("DM4 descriptor stem-support SHA differs")
    blocks = np.zeros((block_h, block_w), dtype=bool)
    blocks.reshape(-1)[indices] = True
    mask = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
    candidate_id = str(candidate["candidate_id"])
    quantum: int | None = None
    if "_q" in candidate_id:
        try:
            quantum = int(candidate_id.rsplit("_q", 1)[1])
        except ValueError as exc:
            raise DDMCountedApplicationError("DM4 descriptor quantum suffix differs") from exc
        if quantum <= 0 or quantum > 255:
            raise DDMCountedApplicationError("DM4 descriptor quantum is outside uint8")
    plane = base[1].copy()
    delta = target[1].astype(np.int16) - base[1].astype(np.int16)
    if quantum is not None:
        delta = np.clip(delta, -quantum, quantum)
    updated = plane.astype(np.int16)
    updated[mask] += delta[mask]
    plane = np.clip(updated, 0, 255).astype(np.uint8)
    resize_kernel = kernel or FullResizeKernel.build()
    base_camera = np.stack(
        [realize_solve_camera(base[index], resize_kernel) for index in range(2)]
    )
    candidate_camera = base_camera.copy()
    candidate_camera[1] = realize_solve_camera(plane, resize_kernel)
    camera_delta = candidate_camera.astype(np.int16) - base_camera.astype(np.int16)
    if not np.any(camera_delta):
        raise DDMCountedApplicationError("DM4 descriptor realizes a zero camera delta")
    receipt = {
        "proposal_id": proposal.proposal_id,
        "candidate_id": candidate_id,
        "pair_id": int(proposal.aimed_cell["pair_id"]),
        "bucket_id": str(proposal.aimed_cell["bucket_id"]),
        "stem_block_count": int(indices.size),
        "stem_support_sha256_uint32le": observed,
        "quantum": quantum,
        "descriptor_camera_delta_sha256_int16le": _sha256(
            camera_delta.astype("<i2", copy=False).tobytes()
        ),
        "changed_channel_values": int(np.count_nonzero(camera_delta)),
        "l1_uint8_delta": int(
            np.abs(camera_delta.astype(np.int64)).sum(dtype=np.int64)
        ),
        "l2_uint8_delta": float(
            np.sqrt(np.square(camera_delta, dtype=np.float64).sum(dtype=np.float64))
        ),
        "application_stage": (
            "stored scorer-recursive stem support -> exact target/quantum scorer plane "
            "-> canonical factor2 preimage -> camera uint8 direction"
        ),
    }
    return np.ascontiguousarray(camera_delta), receipt


def _candidate_model_row(
    *,
    effect: SparseJ5CoordinateEffectV1,
    target: np.ndarray,
    metric: MS4DPairMetricV1,
) -> dict[str, Any]:
    target64 = np.asarray(target, dtype=np.float64)
    target_norm_sq = float(np.vdot(target64, target64).real)
    if target_norm_sq <= 0.0:
        raise DDMCountedApplicationError("application projection target is zero")
    dot = effect.dot_dense(target64)
    alpha = dot / target_norm_sq
    newton = metric.newton_step()
    head_step = alpha * newton
    quadratic = float(
        np.dot(metric.adjoint, head_step)
        + 0.5 * np.dot(head_step, metric.hessian @ head_step)
    )
    effect_norm_sq = effect.norm_sq()
    residual_sq = target_norm_sq + effect_norm_sq - 2.0 * dot
    cosine = dot / math.sqrt(target_norm_sq * effect_norm_sq)
    return {
        "effect": effect,
        "projection_fraction_of_dm4_newton_step": alpha,
        "head_step": head_step,
        "predicted_head_quadratic_delta": quadratic,
        "camera_projection_residual_fraction": max(0.0, residual_sq) / target_norm_sq,
        "camera_cosine": float(np.clip(cosine, -1.0, 1.0)),
    }


def _model_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    effect = row["effect"]
    alpha = float(row["projection_fraction_of_dm4_newton_step"])
    quadratic = float(row["predicted_head_quadratic_delta"])
    return (
        alpha <= 0.0,
        quadratic >= 0.0,
        quadratic,
        float(row["camera_projection_residual_fraction"]),
        abs(alpha - 1.0),
        effect.archive_bytes,
        effect.coordinate_name,
        effect.direction,
    )


def _effect_distance_to_dense(
    effect: SparseJ5CoordinateEffectV1, target: np.ndarray
) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    target_norm_sq = float(np.vdot(target64, target64).real)
    return max(
        0.0,
        target_norm_sq + effect.norm_sq() - 2.0 * effect.dot_dense(target64),
    )


def select_counted_application(
    *,
    proposal: DM4J5ProposalV1,
    descriptor_delta: np.ndarray,
    effects: Sequence[SparseJ5CoordinateEffectV1],
    metric: MS4DPairMetricV1,
    used_raw_coordinates: set[int],
    used_projected_coordinates: set[int],
    projector: Callable[..., np.ndarray] = apply_projection,
) -> dict[str, Any]:
    """Select raw and #580-hygienic exact J5 integer secants.

    The projection happens after the raw acceptance model and before the
    coordinate is committed. The projected continuous effect is mapped back to
    the closest unused integer receiver secant, which supplies a deterministic
    gauge representative and preserves counted-byte closure.
    """

    target = np.asarray(descriptor_delta, dtype=np.float64)
    if target.shape != (2, 874, 1164, 3) or not np.all(np.isfinite(target)):
        raise DDMCountedApplicationError("descriptor camera delta geometry differs")
    pair_id = int(proposal.aimed_cell["pair_id"])
    bucket_id = str(proposal.aimed_cell["bucket_id"])
    if metric.pair_id != pair_id or metric.bucket_id != bucket_id:
        raise DDMCountedApplicationError("proposal/MS4d pair-bucket join differs")
    raw_rows = [
        _candidate_model_row(effect=effect, target=target, metric=metric)
        for effect in effects
        if effect.pair_id == pair_id
        and effect.coordinate_index not in used_raw_coordinates
    ]
    if not raw_rows:
        raise DDMCountedApplicationError("no unused raw J5 coordinate remains")
    raw = min(raw_rows, key=_model_rank)
    raw_effect: SparseJ5CoordinateEffectV1 = raw["effect"]
    raw_dense = raw_effect.dense()
    projected = np.asarray(
        projector(raw_dense, out_dtype=np.float64, compute_dtype=np.float64),
        dtype=np.float64,
    )
    if projected.shape != raw_dense.shape or not np.all(np.isfinite(projected)):
        raise DDMCountedApplicationError("#580 projected effect geometry differs")
    raw_energy = float(np.vdot(raw_dense, raw_dense).real)
    projected_energy = float(np.vdot(projected, projected).real)
    rejected = raw_dense - projected
    rejected_energy = float(np.vdot(rejected, rejected).real)
    if raw_energy <= 0.0:
        raise DDMCountedApplicationError("raw selected effect has zero energy")
    projected_rows: list[dict[str, Any]] = []
    for effect in effects:
        if (
            effect.pair_id != pair_id
            or effect.coordinate_index in used_projected_coordinates
        ):
            continue
        model = _candidate_model_row(effect=effect, target=target, metric=metric)
        projected_rows.append(
            {
                **model,
                "range_projected_distance_sq": _effect_distance_to_dense(
                    effect, projected
                ),
            }
        )
    if not projected_rows:
        raise DDMCountedApplicationError("no unused projected J5 coordinate remains")
    # Score-invisible hygiene is the primary post-acceptance constraint.
    # Curvature sign and stable coordinate identity then break equally close
    # integer representatives; no tunable scalar mixes the two metrics.
    projected_row = min(
        projected_rows,
        key=lambda row: (
            float(row["range_projected_distance_sq"]),
            float(row["predicted_head_quadratic_delta"]) >= 0.0,
            float(row["predicted_head_quadratic_delta"]),
            row["effect"].coordinate_name,
            row["effect"].direction,
        ),
    )
    projected_effect: SparseJ5CoordinateEffectV1 = projected_row["effect"]
    return {
        "schema": SCHEMA,
        "proposal": proposal.to_payload(),
        "ms4d_metric": metric.to_payload(),
        "raw_application": {
            **raw_effect.to_payload(),
            "projection_fraction_of_dm4_newton_step": float(
                raw["projection_fraction_of_dm4_newton_step"]
            ),
            "predicted_head_quadratic_delta": float(
                raw["predicted_head_quadratic_delta"]
            ),
            "camera_projection_residual_fraction": float(
                raw["camera_projection_residual_fraction"]
            ),
            "camera_cosine": float(raw["camera_cosine"]),
        },
        "range_gauge_projection": {
            "policy": RANGE_GAUGE_POLICY,
            "source": "tac.boundary_math.range_a_projection.apply_projection",
            "raw_effect_energy": raw_energy,
            "range_effect_energy": projected_energy,
            "rejected_null_gauge_energy": rejected_energy,
            "rejected_null_gauge_energy_fraction": rejected_energy / raw_energy,
            "orthogonality_residual": abs(
                raw_energy - projected_energy - rejected_energy
            ),
            "projected_effect_sha256_float64le": _sha256(
                projected.astype("<f8", copy=False).tobytes()
            ),
            "parameter_gauge_representative": (
                "minimum_integer_range_projection_residual_then_stable_coordinate_id"
            ),
        },
        "projected_application": {
            **projected_effect.to_payload(),
            "range_projected_distance_sq": float(
                projected_row["range_projected_distance_sq"]
            ),
            "projection_fraction_of_dm4_newton_step": float(
                projected_row["projection_fraction_of_dm4_newton_step"]
            ),
            "predicted_head_quadratic_delta": float(
                projected_row["predicted_head_quadratic_delta"]
            ),
            "camera_projection_residual_fraction": float(
                projected_row["camera_projection_residual_fraction"]
            ),
        },
        "trust_region": {
            "validity_gap": VALIDITY_GAP,
            "coordinate_quantum": 1,
            "coordinate_reuse_allowed": False,
            "shrink_factor": None,
            "grow_factor": None,
            "global_learning_rate": None,
            "conservative_bound": (
                "smallest receiver lattice; one quantum per coordinate from Step-4"
            ),
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def apply_coordinate_choices(
    theta: np.ndarray,
    receipts: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> np.ndarray:
    """Apply a receipt sequence to a Step-4 theta with no coordinate reuse."""

    if arm not in {"raw_application", "projected_application"}:
        raise DDMCountedApplicationError("application arm differs")
    output = np.asarray(theta, dtype=np.float32).copy()
    used: set[int] = set()
    for receipt in receipts:
        row = receipt.get(arm)
        if not isinstance(row, Mapping):
            raise DDMCountedApplicationError(f"receipt lacks {arm}")
        index = int(row["coordinate_index"])
        direction = int(row["direction"])
        if index in used or direction not in {-1, 1} or not 0 <= index < output.size:
            raise DDMCountedApplicationError("counted application violates one-quantum/no-reuse")
        output[index] += np.float32(direction)
        used.add(index)
    return output


def exact_joint_delta(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Price one exact n600 candidate against the Step-4 advisory reference."""

    required = ("d_seg", "d_pose", "archive_bytes")
    if any(key not in reference or key not in candidate for key in required):
        raise DDMCountedApplicationError("exact verdict lacks joint score components")
    reference_seg = float(reference["d_seg"])
    reference_pose = float(reference["d_pose"])
    candidate_seg = float(candidate["d_seg"])
    candidate_pose = float(candidate["d_pose"])
    reference_bytes = int(reference["archive_bytes"])
    candidate_bytes = int(candidate["archive_bytes"])
    if (
        min(reference_seg, reference_pose, candidate_seg, candidate_pose) < 0.0
        or min(reference_bytes, candidate_bytes) <= 0
    ):
        raise DDMCountedApplicationError("exact verdict score component differs")
    seg_term = 100.0 * (candidate_seg - reference_seg)
    pose_term = math.sqrt(10.0 * candidate_pose) - math.sqrt(10.0 * reference_pose)
    rate_term = 25.0 * (candidate_bytes - reference_bytes) / 37_545_489.0
    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "joint_delta": seg_term + pose_term + rate_term,
        "candidate_minus_reference_archive_bytes": candidate_bytes - reference_bytes,
        "acceptance_authority": "strict_exact_n600_joint_delta_lt_zero",
    }


__all__ = [
    "CONFIG_SCHEMA",
    "EVIDENCE_AXIS",
    "HORIZON_GAP",
    "POINTER",
    "RANGE_GAUGE_POLICY",
    "SCHEMA",
    "VALIDITY_GAP",
    "BoundArtifactV1",
    "DDMCountedApplicationConfigV1",
    "DDMCountedApplicationError",
    "MS4DPairMetricV1",
    "SparseJ5CoordinateEffectV1",
    "apply_coordinate_choices",
    "descriptor_camera_delta",
    "enumerate_j5_coordinate_effects",
    "exact_joint_delta",
    "load_ms4d_pair_metric",
    "select_counted_application",
]
