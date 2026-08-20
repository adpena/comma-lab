# SPDX-License-Identifier: MIT
"""Deterministic, source-only bounded-inverse fitter for the frozen R10 receiver.

This module is encoder-side research code.  It consumes exact uint8 base/source
pairs, solves the nine physical R10 sections, and emits only the canonical
packet consumed by :mod:`taskspace_r10_feature_texture_relay`.  It deliberately
contains no frozen evaluator import and assigns no scientific score authority
to its RGB fitting objective.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.dash_phase_carrier import (
    DashPhaseConfig,
    decode_dash_phase_carrier,
    encode_dash_phase_carrier,
)
from tac.boundary_math.stratified_depth_warp import affine_extra_flow, stratified_warp_uint8
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom, warp_frame0_uint8_numpy
from tac.boundary_math.xi_pose_coder import decode_xi_payload, quantize_xi, serialize_xi_payload
from tac.witness_dsl import taskspace_r10_feature_texture_relay as _r10
from tac.witness_dsl.taskspace_r10_feature_texture_relay import (
    R10BaseFeatureRecordV1,
    R10IdentityV1,
    R10PacketV1,
    R10PhysicalSpanV1,
    R10PullbackPolygonV1,
    R10RelayMode,
    R10Section,
    R10ShootingKnotV1,
    R10StratifiedFlowV1,
    R10TextureRecordV1,
    build_r10_decode_receipt,
    build_r10_selected_solution_adapter,
    packet_physical_spans,
    pair_population_sha256,
    parse_r10_packet,
    realization_sha256,
    serialize_r10_packet,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import G17PhysicalCodingGroupV1

G32_SCHEMA: Final = "taskspace_r10_n600_bounded_inverse_fitter.v2"
G32_MANIFEST_SCHEMA: Final = "taskspace_r10_counted_state_manifest.v1"
G32_REPAIR_SCHEMA: Final = "taskspace_r10_generic_bounded_repair.v1"
G32_ADAPTER_SCHEMA: Final = "taskspace_r10_g23_physical_group_adapter.v1"
G32_WRAPPER_MEMBER: Final = "r10.packet"
G32_WRAPPER_OWNER_IDS: Final = tuple(f"R10_{name}" for name in _r10.R10_CONSTRAINTS)
G32_AUTHORITY_BLOCKER: Final = "FULL_N600_PUBLIC_RECEIVER_SCORER_AND_COMPLETE_ZIP_PRICING_OWED"
G32_PUBLIC_VIDEO_BLOCKER: Final = "G29_PUBLIC_VIDEO_MUX_AND_UPSTREAM_REPLAY_OWED"
G32_TERMINAL_DESCENT_BLOCKER: Final = "TERMINAL_JOINT_DESCENT_NOT_ADMITTED_BY_ANALYTIC_LANDING"
G32_STREAM_DOMAIN: Final = b"R10_REALIZED_PAIR_V1\x00"
G32_SECTION_ORDER: Final = tuple(section.name for section in R10Section)

_FORBIDDEN_DECODER_STATE_TOKENS: Final = (
    "teacher",
    "ground_truth",
    "gt_argmax",
    "pose6_target",
    "evaluator_tensor",
    "target_frame",
    "source_frame",
    "hidden_threshold",
    "exception_table",
    "selected_threshold",
    "source_selected",
    "video_specific",
    "per_pair_exception",
)


class R10BoundedInverseError(ValueError):
    """A custody, fitting, counted-state, or deterministic ABI invariant failed."""


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_finite_numbers(value: Any, *, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise R10BoundedInverseError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_numbers(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _require_finite_numbers(child, path=f"{path}[{index}]")


def _strict_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise R10BoundedInverseError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_loads(payload: bytes | str) -> Any:
    """Parse JSON while refusing duplicate keys and non-finite constants."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload

    def reject_constant(value: str) -> None:
        raise R10BoundedInverseError(f"non-finite JSON constant: {value}")

    return json.loads(
        text,
        object_pairs_hook=_strict_object_pairs_hook,
        parse_constant=reject_constant,
    )


@dataclass(frozen=True, slots=True)
class R10BoundedInverseConfigV1:
    """Finite generic search schedule and execution bounds."""

    seed: int
    pair_count: int
    height: int
    width: int
    chunk_pairs: int = 1
    sample_stride: int = 8
    pitch_candidates_q20: tuple[int, ...] = (-20972, -10486, 0, 10486, 20972)
    xi_steps: tuple[float, ...] = (0.01, 0.0025, 0.000625)
    texture_frequencies_q8: tuple[int, ...] = (32, 64, 128, 256)
    texture_phases_q10: tuple[int, ...] = (0, 128, 256, 384, 512, 640, 768, 896)
    xi_q_levels: int = 4096
    support_radius_px: int = 8
    flow_translation_q8: tuple[int, ...] = (-512, -256, 0, 256, 512)
    decoder_iteration_budget: int = 0
    decoder_workspace_bytes: int = 0
    execute_reviewed: bool = False

    def __post_init__(self) -> None:
        for name in ("seed", "pair_count", "height", "width", "chunk_pairs", "sample_stride"):
            if type(getattr(self, name)) is not int:
                raise R10BoundedInverseError(f"{name} must be an exact int")
        if not (1 <= self.pair_count <= 600):
            raise R10BoundedInverseError("pair_count must be in [1,600]")
        if self.height < 8 or self.width < 8:
            raise R10BoundedInverseError("receiver dimensions must each be at least 8")
        if not (1 <= self.chunk_pairs <= self.pair_count):
            raise R10BoundedInverseError("chunk_pairs must be in [1,pair_count]")
        if not (1 <= self.sample_stride <= min(self.height, self.width)):
            raise R10BoundedInverseError("sample_stride is outside the receiver geometry")
        if not self.pitch_candidates_q20 or tuple(sorted(set(self.pitch_candidates_q20))) != self.pitch_candidates_q20:
            raise R10BoundedInverseError("pitch lattice must be a nonempty sorted unique tuple")
        if any(type(value) is not int or not (-(1 << 31) <= value < (1 << 31)) for value in self.pitch_candidates_q20):
            raise R10BoundedInverseError("pitch lattice contains a non-int32 coordinate")
        if not self.xi_steps or any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0 for value in self.xi_steps
        ):
            raise R10BoundedInverseError("xi_steps must be finite positive floats")
        if tuple(sorted(self.xi_steps, reverse=True)) != self.xi_steps:
            raise R10BoundedInverseError("xi_steps must be coarse-to-fine")
        if not self.texture_frequencies_q8 or any(
            type(value) is not int or not (1 <= value <= 65535) for value in self.texture_frequencies_q8
        ):
            raise R10BoundedInverseError("texture frequencies must be nonzero uint16 values")
        if not self.texture_phases_q10 or any(
            type(value) is not int or not (0 <= value < 1024) for value in self.texture_phases_q10
        ):
            raise R10BoundedInverseError("texture phases must be Q10 values")
        if not (1 <= self.xi_q_levels <= 32767):
            raise R10BoundedInverseError("xi_q_levels must be in [1,32767]")
        if not (2 <= self.support_radius_px <= min(self.height, self.width) // 2):
            raise R10BoundedInverseError("support_radius_px does not fit receiver geometry")
        if 0 not in self.flow_translation_q8 or any(
            type(value) is not int or not (-32768 <= value <= 32767) for value in self.flow_translation_q8
        ):
            raise R10BoundedInverseError("flow translation lattice must contain zero and fit int16")
        if self.decoder_iteration_budget != 0 or self.decoder_workspace_bytes != 0:
            raise R10BoundedInverseError("this landing exposes repair compatibility but keeps repair inactive")
        if type(self.execute_reviewed) is not bool:
            raise R10BoundedInverseError("execute_reviewed must be an exact bool")
        if self.pair_count == 600 and not self.execute_reviewed:
            raise R10BoundedInverseError("n600 requires reviewed execution plus live governed admission")


@dataclass(frozen=True, slots=True)
class R10SectionFitV1:
    section: str
    solver_family: str
    packet_offset: int
    byte_length: int
    section_sha256: str
    objective_before: int | None
    objective_after: int | None
    realized_changed_values: int | None
    realized_changed_pixels: int | None
    realized_l1_rgb_delta: int | None
    realized_max_abs_rgb_delta: int | None
    interaction_parents: tuple[str, ...]
    solver_iterations: int
    wall_seconds: float | None
    peak_working_bytes: int | None
    sufficient_statistics: Mapping[str, Any]
    d_seg: float | None = None
    d_pose: float | None = None
    complete_candidate_zip_bytes: int | None = None
    score_units_per_byte: float | None = None
    authority_blocker: str = G32_AUTHORITY_BLOCKER

    def __post_init__(self) -> None:
        if self.section not in G32_SECTION_ORDER:
            raise R10BoundedInverseError(f"unknown R10 section: {self.section}")
        if self.byte_length <= 0 or self.packet_offset < 0:
            raise R10BoundedInverseError("section fit lacks a positive exact packet span")
        if len(self.section_sha256) != 64:
            raise R10BoundedInverseError("section fit lacks a SHA-256 identity")
        if any(
            value is not None
            for value in (self.d_seg, self.d_pose, self.complete_candidate_zip_bytes, self.score_units_per_byte)
        ):
            raise R10BoundedInverseError("bounded RGB fitting cannot populate scientific authority fields")
        _require_finite_numbers(asdict(self))


@dataclass(frozen=True, slots=True)
class R10VideoSpecificClaimV1:
    claim_id: str
    section: str
    packet_offset: int
    byte_length: int
    span_sha256: str
    active: bool
    video_specific: bool
    derivation: str


@dataclass(frozen=True, slots=True)
class R10CountedStateManifestV1:
    schema: str
    packet_sha256: str
    packet_bytes: int
    wrapper_zip_sha256: str
    wrapper_zip_bytes: int
    wrapper_member_name: str
    wrapper_member_sha256: str
    member_data_archive_offset: int
    physical_sections: tuple[R10PhysicalSpanV1, ...]
    claims: tuple[R10VideoSpecificClaimV1, ...]
    generic_decoder_mechanisms: tuple[str, ...]
    encoder_only_evidence: tuple[str, ...]
    forbidden_payloads_present: tuple[str, ...]
    complete_packet_coverage: bool


@dataclass(frozen=True, slots=True)
class R10GenericRepairContractV1:
    schema: str = G32_REPAIR_SCHEMA
    operation_id: str = "r10.generic_bounded_repair.inactive_identity.v1"
    active: bool = False
    iteration_budget: int = 0
    workspace_bytes: int = 0
    video_specific_defaults: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class R10CorrectionPlacementComparisonV1:
    section: str
    stored_packet_bytes: int
    stored_wrapper_zip_bytes: int
    exact_generic_regenerator_available: bool
    bounded_generic_repair_available: bool
    delete_arm_realized_changed_values: int | None
    repair_arm_realized_changed_values: int | None
    verdict: str
    complete_candidate_zip_bytes: int | None = None
    d_seg: float | None = None
    d_pose: float | None = None


@dataclass(frozen=True, slots=True)
class R10ResidualInventoryV1:
    completed_analytic_stages: tuple[str, ...]
    bounded_source_objective_interaction_passes: int
    bounded_action_family_exhausted: bool
    unenumerated_action_families: tuple[str, ...]
    full_n600_receiver_measured_exhaustion: bool
    learned_residual_admitted: bool
    terminal_joint_descent_admitted: bool
    blocker: str


@dataclass(frozen=True, slots=True)
class R10ArchiveSectionSpanV1:
    section: str
    packet_offset: int
    member_offset: int
    archive_offset: int
    byte_length: int
    span_sha256: str


@dataclass(frozen=True, slots=True)
class R10ContinuationEquivalenceIdentityV1:
    schema: str
    source_sha256: str
    pair_population_sha256: str
    base_realization_sha256: str
    packet_sha256: str
    packet_bytes: int
    receiver_operation: str
    receiver_source_closure_sha256: str
    output_shape: tuple[int, int, int, int, int]
    output_dtype: str
    canonical_pair_order: tuple[int, ...]

    @property
    def identity_sha256(self) -> str:
        return _sha256(_canonical_json(asdict(self)))


@dataclass(frozen=True, slots=True)
class R10G23PhysicalGroupAdapterV1:
    schema: str
    physical_group: G17PhysicalCodingGroupV1 = field(repr=False)
    packet_sha256: str
    packet_bytes: int
    member_data_archive_offset: int
    section_spans: tuple[R10ArchiveSectionSpanV1, ...]
    g27_adapter_sha256: str
    g27_receiver_operation: str
    continuation_equivalence_sha256: str
    constraint_operand_spans: tuple[Mapping[str, Any], ...]
    blocker: str = G32_PUBLIC_VIDEO_BLOCKER


@dataclass(frozen=True, slots=True)
class R10BoundedInverseResultV1:
    schema: str
    packet: R10PacketV1
    packet_bytes: bytes = field(repr=False)
    wrapper_zip_bytes: bytes = field(repr=False)
    fitted_xi: np.ndarray = field(repr=False)
    section_fits: tuple[R10SectionFitV1, ...]
    counted_state_manifest: R10CountedStateManifestV1
    physical_group_adapter: R10G23PhysicalGroupAdapterV1
    continuation_equivalence: R10ContinuationEquivalenceIdentityV1
    repair_contract: R10GenericRepairContractV1
    correction_placement: tuple[R10CorrectionPlacementComparisonV1, ...]
    residual_inventory: R10ResidualInventoryV1
    bounded_output: np.ndarray | None = field(default=None, repr=False)
    bounded_decode_receipt: Mapping[str, Any] | None = None
    scratch_paths: tuple[str, ...] = ()
    pointer_delta: bool = False


def _validate_pair_arrays(base_pairs: np.ndarray, source_pairs: np.ndarray, config: R10BoundedInverseConfigV1) -> None:
    expected = (config.pair_count, 2, config.height, config.width, 3)
    if getattr(base_pairs, "dtype", None) != np.uint8 or tuple(base_pairs.shape) != expected:
        raise R10BoundedInverseError(f"base_pairs must be uint8 {expected}")
    if getattr(source_pairs, "dtype", None) != np.uint8 or tuple(source_pairs.shape) != expected:
        raise R10BoundedInverseError(f"source_pairs must be uint8 {expected}")
    if np.shares_memory(base_pairs, source_pairs):
        raise R10BoundedInverseError("base and encoder-only source buffers must not alias")


def _allocate_intermediate(
    name: str,
    config: R10BoundedInverseConfigV1,
    scratch_directory: str | Path | None,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | type[np.generic] = np.uint8,
) -> tuple[np.ndarray, str | None]:
    requested_shape = shape or (config.pair_count, 2, config.height, config.width, 3)
    if config.pair_count <= 24 and scratch_directory is None:
        return np.empty(requested_shape, dtype=dtype), None
    if scratch_directory is None:
        raise R10BoundedInverseError("populations above 24 pairs require explicit SSD scratch_directory")
    root = Path(scratch_directory).resolve()
    if not root.is_dir():
        raise R10BoundedInverseError(f"scratch_directory is not an existing directory: {root}")
    path = root / f"{name}.raw"
    if path.exists():
        expected_bytes = int(np.prod(requested_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_size != expected_bytes:
            raise R10BoundedInverseError(f"retained scratch has wrong identity or byte extent: {path}")
        array = np.memmap(path, mode="r+", dtype=dtype, shape=requested_shape, order="C")
    else:
        array = np.memmap(path, mode="w+", dtype=dtype, shape=requested_shape, order="C")
    return array, str(path)


def streaming_realization_sha256(
    raw_path: str | Path,
    *,
    pair_count: int,
    height: int,
    width: int,
    read_bytes: int = 16 << 20,
) -> str:
    """Hash an exact contiguous pair raw without materializing its population."""

    path = Path(raw_path)
    expected = pair_count * 2 * height * width * 3
    if path.stat().st_size != expected:
        raise R10BoundedInverseError(f"raw realization bytes differ: {path.stat().st_size} != {expected}")
    digest = hashlib.sha256()
    digest.update(G32_STREAM_DOMAIN)
    digest.update(struct.pack("<5I", pair_count, 2, height, width, 3))
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(read_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sse(left: np.ndarray, right: np.ndarray, *, stride: int = 1) -> int:
    a = np.asarray(left)[::stride, ::stride].astype(np.int64)
    b = np.asarray(right)[::stride, ::stride].astype(np.int64)
    delta = a - b
    return int(np.sum(delta * delta, dtype=np.int64))


def _changed_stats(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int]:
    delta = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return (
        int(np.count_nonzero(delta)),
        int(np.count_nonzero(np.any(delta != 0, axis=-1))),
        int(delta.astype(np.int64).sum()),
        int(delta.max(initial=0)),
    )


def _fit_pitch_and_xi(
    base_pairs: np.ndarray,
    source_pairs: np.ndarray,
    config: R10BoundedInverseConfigV1,
    *,
    fixed_pitch_q20: int | None = None,
    resume_pair_ranges: Sequence[Mapping[str, Any]] = (),
    range_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Finite native-geometry pitch/XIP2 solve with a full-resolution control.

    The frozen homography is defined on the native receiver raster.  Sampling
    is therefore applied to the residual *after* the native warp; constructing
    a fresh geometry on a decimated raster is a different, noncommuting
    operation and is forbidden here.
    """

    stride = config.sample_stride
    base_native = [np.asarray(base_pairs[p, 0]) for p in range(config.pair_count)]
    source_native = [np.asarray(source_pairs[p, 0]) for p in range(config.pair_count)]
    lattice: list[dict[str, int]] = []
    zero = np.zeros(6, dtype=np.float64)
    if fixed_pitch_q20 is None:
        for pitch_candidate_q20 in config.pitch_candidates_q20:
            pitch_candidate = pitch_candidate_q20 / float(1 << 20)
            total = 0
            for base, target in zip(base_native, source_native, strict=True):
                geom = GroundHomographyGeom.eon(native_hw=base.shape[:2], pitch=pitch_candidate)
                total += _sse(warp_frame0_uint8_numpy(base, zero, geom), target, stride=stride)
            lattice.append({"pitch_q20": pitch_candidate_q20, "objective": total})
        chosen = min(lattice, key=lambda row: (row["objective"], abs(row["pitch_q20"]), row["pitch_q20"]))
        pitch_q20 = int(chosen["pitch_q20"])
    else:
        if fixed_pitch_q20 not in config.pitch_candidates_q20:
            raise R10BoundedInverseError("resumed pitch is outside the configured finite lattice")
        pitch_q20 = fixed_pitch_q20
    pitch = pitch_q20 / float(1 << 20)
    xi = np.zeros((config.pair_count, 6), dtype=np.float64)
    pair_rows: list[dict[str, Any]] = []
    iterations = 0
    for range_payload in resume_pair_ranges:
        if range_payload.get("pitch_q20") != pitch_q20 or set(range_payload) != {"pitch_q20", "pair_solves"}:
            raise R10BoundedInverseError("resumed XIP2 range is not bound to the retained pitch")
        rows = range_payload["pair_solves"]
        if not isinstance(rows, list):
            raise R10BoundedInverseError("resumed XIP2 range pair_solves is not a list")
        for row in rows:
            pair_index = len(pair_rows)
            if not isinstance(row, dict) or row.get("pair_index") != pair_index:
                raise R10BoundedInverseError("resumed XIP2 range coordinates are not contiguous")
            values = np.asarray(row.get("xi"), dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise R10BoundedInverseError("resumed XIP2 range contains invalid coordinates")
            xi[pair_index] = values
            pair_rows.append(dict(row))
            iterations += int(row.get("candidate_evaluations", 0))
    for first_pair in range(len(pair_rows), config.pair_count, config.chunk_pairs):
        stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
        new_rows: list[dict[str, Any]] = []
        for pair_index in range(first_pair, stop_pair):
            base = base_native[pair_index]
            target = source_native[pair_index]
            geom = GroundHomographyGeom.eon(native_hw=base.shape[:2], pitch=pitch)

            def objective(
                value: np.ndarray,
                base_frame: np.ndarray = base,
                geometry: GroundHomographyGeom = geom,
                target_frame: np.ndarray = target,
            ) -> int:
                return _sse(
                    warp_frame0_uint8_numpy(base_frame, value, geometry),
                    target_frame,
                    stride=stride,
                )

            current = np.zeros(6, dtype=np.float64)
            current_obj = objective(current)
            initial = current_obj
            accepted = 0
            pair_iterations = 0
            for step in config.xi_steps:
                for coordinate in range(6):
                    candidates: list[tuple[int, float, np.ndarray]] = []
                    for sign in (-1.0, 1.0):
                        trial = current.copy()
                        trial[coordinate] += sign * step
                        candidates.append((objective(trial), abs(trial[coordinate]), trial))
                        iterations += 1
                        pair_iterations += 1
                    best_obj, _magnitude, best = min(
                        candidates,
                        key=lambda row: (row[0], row[1], tuple(row[2])),
                    )
                    if best_obj < current_obj:
                        current = best
                        current_obj = best_obj
                        accepted += 1
            xi[pair_index] = current
            row = {
                "pair_index": pair_index,
                "initial_objective": initial,
                "selected_objective": current_obj,
                "accepted_steps": accepted,
                "candidate_evaluations": pair_iterations,
                "xi": [float(value) for value in current],
            }
            pair_rows.append(row)
            new_rows.append(row)
        if range_callback is not None:
            range_callback(
                first_pair,
                stop_pair,
                {"pitch_q20": pitch_q20, "pair_solves": new_rows},
            )
    full_resolution_controls: list[dict[str, Any]] = []
    rejected_pairs_full_resolution: set[int] = set()
    for _control_pass in range(config.pair_count + 1):
        q, scales = quantize_xi(xi, q_levels=config.xi_q_levels)
        payload = serialize_xi_payload(q, scales, coder="delta_res")
        decoded = decode_xi_payload(payload)
        worsening: list[int] = []
        pass_rows: list[dict[str, Any]] = []
        for pair_index, (base, target) in enumerate(zip(base_native, source_native, strict=True)):
            geom = GroundHomographyGeom.eon(native_hw=base.shape[:2], pitch=pitch)
            zero_objective = _sse(warp_frame0_uint8_numpy(base, zero, geom), target)
            fitted_objective = _sse(warp_frame0_uint8_numpy(base, decoded[pair_index], geom), target)
            retained = fitted_objective <= zero_objective
            pass_rows.append(
                {
                    "pair_index": pair_index,
                    "zero_objective_full_resolution": zero_objective,
                    "fitted_objective_full_resolution": fitted_objective,
                    "retained": retained,
                }
            )
            if not retained and np.any(decoded[pair_index] != 0.0):
                worsening.append(pair_index)
        full_resolution_controls = pass_rows
        if not worsening:
            break
        rejected_pairs_full_resolution.update(worsening)
        xi[worsening] = 0.0
    else:  # pragma: no cover - each pass zeros at least one previously nonzero row
        raise R10BoundedInverseError("full-resolution XIP2 control failed to converge")
    quantized_objective = 0
    quantized_objective_full_resolution = 0
    for pair_index, (base, target) in enumerate(zip(base_native, source_native, strict=True)):
        geom = GroundHomographyGeom.eon(native_hw=base.shape[:2], pitch=pitch)
        realized = warp_frame0_uint8_numpy(base, decoded[pair_index], geom)
        quantized_objective += _sse(realized, target, stride=stride)
        quantized_objective_full_resolution += _sse(realized, target)
    return (
        pitch_q20,
        decoded,
        {
            "pitch_lattice": lattice,
            "pair_solves": pair_rows,
            "quantized_objective": quantized_objective,
            "quantized_objective_full_resolution": quantized_objective_full_resolution,
            "full_resolution_controls": full_resolution_controls,
            "rejected_pairs_full_resolution": sorted(rejected_pairs_full_resolution),
            "iterations": iterations,
            "xip2_payload": payload,
        },
    )


def _rank_one_fixed_point(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u, singular, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    if singular[0] == 0.0:
        return np.zeros(4, dtype=np.int64), np.array((1, 1, 1), dtype=np.int64), 0.0
    left = u[:, 0] * math.sqrt(float(singular[0]))
    right = vh[0] * math.sqrt(float(singular[0]))
    pivot = int(np.argmax(np.abs(right)))
    if right[pivot] < 0.0:
        left = -left
        right = -right
    max_left = float(np.max(np.abs(left), initial=0.0))
    max_right = float(np.max(np.abs(right), initial=0.0))
    scale_low = max_right / 30000.0 if max_right else 1.0
    scale_high = 30000.0 / max_left if max_left else 1.0
    scale = min(max(1.0, scale_low), max(1.0, scale_high))
    left_i = np.clip(np.rint(left * scale), -32768, 32767).astype(np.int64)
    right_i = np.clip(np.rint(right / scale), -32768, 32767).astype(np.int64)
    if not np.any(right_i):
        right_i[pivot] = 1
    approximation = np.outer(left_i, right_i)
    residual = float(np.sum((matrix - approximation) ** 2))
    return left_i, right_i, residual


def _record_from_values(values: Sequence[int]) -> R10BaseFeatureRecordV1:
    clipped = tuple(int(np.clip(value, -32768, 32767)) for value in values)
    weights = clipped[4:7]
    if not any(weights):
        weights = (1, 1, 1)
    return R10BaseFeatureRecordV1(*clipped[:4], weights)


def _fit_base_feature_record(
    pair: np.ndarray,
    target: np.ndarray,
    *,
    sample_stride: int,
) -> tuple[R10BaseFeatureRecordV1, dict[str, Any]]:
    columns: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for role in (0, 1):
        frame = np.asarray(pair[role], dtype=np.uint8)
        luma, laplacian, feature = _r10._feature_maps(frame)
        sample = (slice(None, None, sample_stride), slice(None, None, sample_stride))
        design = np.stack(
            (
                np.ones_like(luma[sample]),
                luma[sample] - 128,
                laplacian[sample],
                feature[sample],
            ),
            axis=-1,
        ).reshape(-1, 4)
        residual = (target[role][sample].astype(np.float64) - frame[sample].astype(np.float64)).reshape(-1, 3)
        columns.append(design.astype(np.float64))
        responses.append(residual)
        cached.append((frame, luma, laplacian, feature, target[role]))
    design = np.concatenate(columns, axis=0)
    response = np.concatenate(responses, axis=0)
    beta, _residuals, rank, singular_values = np.linalg.lstsq(design, response, rcond=None)
    left, weights, rank_one_residual = _rank_one_fixed_point(beta * 65536.0)
    current = np.concatenate((left, weights)).astype(np.int64)

    def sampled_objective(values: np.ndarray) -> int:
        record = _record_from_values(values)
        total = 0
        for frame, _luma, _lap, _feature, truth in cached:
            predicted = _r10._apply_feature_relay(
                frame,
                record,
                None,
                (),
                np.zeros(6, dtype=np.int64),
                frame_role=0,
            )
            total += _sse(predicted, truth, stride=sample_stride)
        return total

    initial = sampled_objective(np.array((0, 0, 0, 0, 1, 1, 1), dtype=np.int64))
    current_obj = sampled_objective(current)
    iterations = 0
    for step in (64, 16, 4, 1):
        for coordinate in range(7):
            choices: list[tuple[int, tuple[int, ...], np.ndarray]] = []
            for delta in (-step, step):
                trial = current.copy()
                trial[coordinate] = np.clip(trial[coordinate] + delta, -32768, 32767)
                if coordinate >= 4 and not np.any(trial[4:]):
                    continue
                choices.append((sampled_objective(trial), tuple(int(v) for v in trial), trial))
                iterations += 1
            if choices:
                best_obj, _lex, best = min(choices, key=lambda row: (row[0], row[1]))
                if best_obj < current_obj:
                    current = best
                    current_obj = best_obj
    record = _record_from_values(current)
    return record, {
        "objective_before": initial,
        "objective_after": current_obj,
        "design_rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
        "rank_one_residual": rank_one_residual,
        "iterations": iterations,
        "saturated_parameters": int(np.count_nonzero((current == -32768) | (current == 32767))),
    }


def _warp_pair_frame0(pair: np.ndarray, xi: np.ndarray, pitch: float) -> np.ndarray:
    out = np.array(pair, copy=True)
    geom = GroundHomographyGeom.eon(native_hw=pair.shape[1:3], pitch=pitch)
    out[0] = warp_frame0_uint8_numpy(out[0], xi, geom)
    return out


def _source_residual_support_labels(
    current_pairs: np.ndarray,
    source_pairs: np.ndarray,
    config: R10BoundedInverseConfigV1,
    *,
    output: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[dict[str, int], ...]]:
    labels = np.zeros((config.pair_count, config.height, config.width), dtype=np.uint8) if output is None else output
    if labels.dtype != np.uint8 or labels.shape != (config.pair_count, config.height, config.width):
        raise R10BoundedInverseError("support label output has the wrong uint8 population shape")
    labels[...] = 0
    rows: list[dict[str, int]] = []
    radius = config.support_radius_px
    for pair_index in range(config.pair_count):
        delta = np.abs(
            source_pairs[pair_index, 1].astype(np.int16) - current_pairs[pair_index, 1].astype(np.int16)
        ).sum(axis=-1, dtype=np.int64)
        flat_index = int(np.argmax(delta))
        row, col = np.unravel_index(flat_index, delta.shape)
        row = int(np.clip(row, radius + 1, config.height - radius - 2))
        col = int(np.clip(col, radius + 1, config.width - radius - 2))
        r0, r1 = row - radius, row + radius + 1
        c0, c1 = col - radius, col + radius + 1
        labels[pair_index, r0:r1, c0:c1] = 1
        rows.append(
            {
                "pair_index": pair_index,
                "center_row": row,
                "center_col": col,
                "support_pixels": int((r1 - r0) * (c1 - c0)),
            }
        )
    return labels, tuple(rows)


def _fit_texture_record(
    pair: np.ndarray,
    target: np.ndarray,
    base_record: R10BaseFeatureRecordV1,
    dashes: Sequence[Any],
    config: R10BoundedInverseConfigV1,
) -> tuple[R10TextureRecordV1, dict[str, Any]]:
    best: tuple[int, bytes, R10TextureRecordV1] | None = None
    iterations = 0
    for frequency in config.texture_frequencies_q8:
        for phase in config.texture_phases_q10:
            carriers = [
                _r10._texture_carrier(
                    config.height,
                    config.width,
                    dashes,
                    frequency_q8=frequency,
                    phase_q10=phase,
                    frame_role=role,
                )
                for role in (0, 1)
            ]
            numerator = 0.0
            denominator = 0.0
            weights = np.asarray(base_record.channel_weights_q8, dtype=np.float64)
            for role in (0, 1):
                base_only = _r10._apply_feature_relay(
                    pair[role],
                    base_record,
                    None,
                    (),
                    np.zeros(6, dtype=np.int64),
                    frame_role=role,
                )
                residual = target[role].astype(np.float64) - base_only.astype(np.float64)
                carrier = carriers[role][:: config.sample_stride, :: config.sample_stride].astype(np.float64)
                direction = carrier[..., None] * weights[None, None, :] / 65536.0
                sampled_residual = residual[:: config.sample_stride, :: config.sample_stride]
                numerator += float(np.sum(direction * sampled_residual))
                denominator += float(np.sum(direction * direction))
            amplitude = 0 if denominator == 0.0 else int(np.clip(np.rint(numerator / denominator), -32768, 32767))
            for candidate_amplitude in sorted({0, amplitude - 1, amplitude, amplitude + 1}):
                candidate_amplitude = int(np.clip(candidate_amplitude, -32768, 32767))
                record = R10TextureRecordV1(candidate_amplitude, frequency, phase, 256)
                total = 0
                for role in (0, 1):
                    output = _r10._apply_feature_relay(
                        pair[role],
                        base_record,
                        record,
                        dashes,
                        np.zeros(6, dtype=np.int64),
                        frame_role=role,
                    )
                    total += _sse(output, target[role], stride=config.sample_stride)
                key_bytes = struct.pack("<hHHh", record.amplitude_q8, frequency, phase, 256)
                candidate = (total, key_bytes, record)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                iterations += 1
    if best is None:
        raise R10BoundedInverseError("finite texture lattice produced no candidate")
    return best[2], {"objective_after": best[0], "iterations": iterations}


def _polygon_from_support(row: Mapping[str, int], height: int, width: int, radius: int) -> R10PullbackPolygonV1:
    center_row = row["center_row"]
    center_col = row["center_col"]
    r0, r1 = max(0, center_row - radius), min(height - 1, center_row + radius)
    c0, c1 = max(0, center_col - radius), min(width - 1, center_col + radius)

    def q15(value: int, extent: int) -> int:
        return int(np.clip(np.rint(value * 32767.0 / max(1, extent - 1)), 0, 32767))

    vertices = (
        (q15(c0, width), q15(r0, height)),
        (q15(c1, width), q15(r0, height)),
        (q15(c1, width), q15(r1, height)),
        (q15(c0, width), q15(r1, height)),
    )
    return R10PullbackPolygonV1(row["pair_index"], vertices)


def _fit_flow_record(
    frame: np.ndarray,
    target: np.ndarray,
    xi: np.ndarray,
    pitch: float,
    polygon: R10PullbackPolygonV1,
    config: R10BoundedInverseConfigV1,
) -> tuple[R10StratifiedFlowV1, dict[str, Any]]:
    geom = GroundHomographyGeom.eon(native_hw=frame.shape[:2], pitch=pitch)
    mask = _r10._polygon_mask(config.height, config.width, polygon)
    best: tuple[int, tuple[int, int], R10StratifiedFlowV1] | None = None
    iterations = 0
    for e_q8 in config.flow_translation_q8:
        for f_q8 in config.flow_translation_q8:
            affine = (0, 0, 0, 0, e_q8, f_q8)
            extra = affine_extra_flow(frame.shape[:2], np.asarray(affine, dtype=np.float64) / 256.0)
            output = stratified_warp_uint8(frame, xi, geom, offplane_mask=mask, extra_flow=extra)
            objective = _sse(output, target, stride=config.sample_stride)
            record = R10StratifiedFlowV1(polygon.pair_index, affine)
            candidate = (objective, (e_q8, f_q8), record)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            iterations += 1
    if best is None:
        raise R10BoundedInverseError("finite stratified-flow lattice produced no candidate")
    return best[2], {"objective_after": best[0], "iterations": iterations}


def _realize_pair_components(
    base_pair: np.ndarray,
    *,
    xi: np.ndarray,
    pitch: float,
    polygon: R10PullbackPolygonV1,
    flow: R10StratifiedFlowV1,
    base_record: R10BaseFeatureRecordV1,
    texture: R10TextureRecordV1,
    dashes: Sequence[Any],
    knot_delta: np.ndarray,
) -> np.ndarray:
    """Execute one pair through the exact frozen receiver primitives."""

    output = np.array(base_pair, copy=True)
    geom = GroundHomographyGeom.eon(native_hw=base_pair.shape[1:3], pitch=pitch)
    mask = _r10._polygon_mask(base_pair.shape[1], base_pair.shape[2], polygon)
    coefficients = np.asarray(flow.affine_q8, dtype=np.float64) / 256.0
    extra = affine_extra_flow(base_pair.shape[1:3], coefficients)
    output[0] = stratified_warp_uint8(
        output[0],
        xi,
        geom,
        offplane_mask=mask,
        extra_flow=extra,
    )
    for role in (0, 1):
        output[role] = _r10._apply_feature_relay(
            output[role],
            base_record,
            texture,
            dashes,
            knot_delta,
            frame_role=role,
        )
    return output


def _refit_joint_flow_record(
    base_pair: np.ndarray,
    target_pair: np.ndarray,
    *,
    xi: np.ndarray,
    pitch: float,
    polygon: R10PullbackPolygonV1,
    base_record: R10BaseFeatureRecordV1,
    texture: R10TextureRecordV1,
    dashes: Sequence[Any],
    knot_delta: np.ndarray,
    config: R10BoundedInverseConfigV1,
) -> tuple[R10StratifiedFlowV1, dict[str, Any]]:
    best: tuple[int, tuple[int, int], R10StratifiedFlowV1] | None = None
    iterations = 0
    for e_q8 in config.flow_translation_q8:
        for f_q8 in config.flow_translation_q8:
            record = R10StratifiedFlowV1(polygon.pair_index, (0, 0, 0, 0, e_q8, f_q8))
            output = _realize_pair_components(
                base_pair,
                xi=xi,
                pitch=pitch,
                polygon=polygon,
                flow=record,
                base_record=base_record,
                texture=texture,
                dashes=dashes,
                knot_delta=knot_delta,
            )
            objective = _sse(output[0], target_pair[0], stride=config.sample_stride)
            candidate = (objective, (e_q8, f_q8), record)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            iterations += 1
    if best is None:
        raise R10BoundedInverseError("joint finite stratified-flow lattice produced no candidate")
    return best[2], {"objective_after": best[0], "iterations": iterations}


def _deterministic_store_zip(member: bytes) -> tuple[bytes, int]:
    stream = io.BytesIO()
    info = zipfile.ZipInfo(G32_WRAPPER_MEMBER, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        archive.writestr(info, member)
    payload = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        reopened = archive.read(G32_WRAPPER_MEMBER)
        member_info = archive.getinfo(G32_WRAPPER_MEMBER)
    if reopened != member:
        raise R10BoundedInverseError("deterministic wrapper ZIP failed reopen identity")
    header = struct.unpack_from("<4s5H3I2H", payload, member_info.header_offset)
    if header[0] != b"PK\x03\x04":
        raise R10BoundedInverseError("wrapper member lacks canonical local ZIP header")
    data_offset = member_info.header_offset + 30 + header[-2] + header[-1]
    if payload[data_offset : data_offset + len(member)] != member:
        raise R10BoundedInverseError("wrapper member data range differs from exact packet")
    return payload, data_offset


def _manifest_claims(packet_bytes: bytes, spans: tuple[R10PhysicalSpanV1, ...]) -> tuple[R10VideoSpecificClaimV1, ...]:
    first = min(span.byte_offset for span in spans)
    claims = [
        R10VideoSpecificClaimV1(
            claim_id="PACKET_HEADER_AND_DIRECTORY",
            section="PACKET_HEADER_AND_DIRECTORY",
            packet_offset=0,
            byte_length=first,
            span_sha256=_sha256(packet_bytes[:first]),
            active=True,
            video_specific=True,
            derivation="canonical identity geometry and physical directory",
        )
    ]
    for span in spans:
        claims.append(
            R10VideoSpecificClaimV1(
                claim_id=f"COUNTED_{span.section}",
                section=span.section,
                packet_offset=span.byte_offset,
                byte_length=span.byte_length,
                span_sha256=span.range_sha256,
                active=True,
                video_specific=True,
                derivation="deterministic source-only finite inverse solve",
            )
        )
    return tuple(claims)


def audit_counted_state_manifest(manifest: R10CountedStateManifestV1, packet_bytes: bytes) -> None:
    """Fail closed on hidden code-as-data, span drift, overlap, or dead packet bytes."""

    if type(manifest) is not R10CountedStateManifestV1 or type(packet_bytes) is not bytes:
        raise R10BoundedInverseError("counted-state audit requires exact typed manifest and bytes")
    if manifest.schema != G32_MANIFEST_SCHEMA or manifest.packet_sha256 != _sha256(packet_bytes):
        raise R10BoundedInverseError("counted-state manifest packet identity drifted")
    if manifest.packet_bytes != len(packet_bytes) or manifest.forbidden_payloads_present:
        raise R10BoundedInverseError("counted-state manifest length or forbidden-payload gate failed")
    parse_r10_packet(packet_bytes)
    spans = packet_physical_spans(packet_bytes)
    if tuple(span.section for span in spans) != G32_SECTION_ORDER:
        raise R10BoundedInverseError("packet does not contain all nine physical sections in order")
    if spans != manifest.physical_sections:
        raise R10BoundedInverseError("physical section manifest differs from strict packet parse")
    cursor = 0
    seen_sections: set[str] = set()
    for claim in sorted(manifest.claims, key=lambda value: value.packet_offset):
        lowered = f"{claim.claim_id} {claim.derivation}".lower()
        if any(token in lowered for token in _FORBIDDEN_DECODER_STATE_TOKENS):
            raise R10BoundedInverseError(f"forbidden decoder-state claim: {claim.claim_id}")
        if claim.active and claim.video_specific and claim.byte_length <= 0:
            raise R10BoundedInverseError(f"active video-specific claim is uncounted: {claim.claim_id}")
        if claim.packet_offset != cursor:
            raise R10BoundedInverseError("counted claims overlap or leave uncovered packet bytes")
        stop = claim.packet_offset + claim.byte_length
        if stop > len(packet_bytes) or _sha256(packet_bytes[claim.packet_offset : stop]) != claim.span_sha256:
            raise R10BoundedInverseError(f"counted claim span drifted: {claim.claim_id}")
        cursor = stop
        if claim.section in G32_SECTION_ORDER:
            seen_sections.add(claim.section)
    if cursor != len(packet_bytes) or seen_sections != set(G32_SECTION_ORDER):
        raise R10BoundedInverseError("counted claims do not exactly cover packet and all sections")
    for mechanism in manifest.generic_decoder_mechanisms:
        lowered = mechanism.lower()
        if any(token in lowered for token in _FORBIDDEN_DECODER_STATE_TOKENS):
            raise R10BoundedInverseError(f"generic decoder mechanism contains hidden video state: {mechanism}")
    if not manifest.complete_packet_coverage:
        raise R10BoundedInverseError("manifest refuses to attest its mechanically verified coverage")


def apply_generic_bounded_repair(
    realized_pairs: np.ndarray,
    packet_bytes: bytes,
    iteration_budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the deliberately inactive generic post-R10 compatibility ABI."""

    parse_r10_packet(packet_bytes)
    pairs = np.asarray(realized_pairs)
    if pairs.dtype != np.uint8 or pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise R10BoundedInverseError("repair ABI requires uint8 [P,2,H,W,3]")
    if type(iteration_budget) is not int or iteration_budget != 0:
        raise R10BoundedInverseError("no active repair mode is admitted; iteration_budget must be zero")
    output = np.array(pairs, copy=True)
    return output, {
        "schema": G32_REPAIR_SCHEMA,
        "operation_id": "r10.generic_bounded_repair.inactive_identity.v1",
        "active": False,
        "iteration_budget": 0,
        "workspace_bytes": 0,
        "input_sha256": realization_sha256(pairs),
        "output_sha256": realization_sha256(output),
        "changed_values": 0,
        "video_specific_defaults": [],
        "authority": "compatibility_only_no_repair_evidence",
    }


def _scope_receiver_receipt_to_bounded_family(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Replace stale aspirational maximum/unbounded labels with measured scope.

    The frozen G27 receiver receipt predates G35 and exposes an aspirational
    ``maximum_inverse_solve_hook``.  G35 has executed only a finite action
    family, so retaining that label in a current receipt would be a false
    claim even though the underlying uint8 replay evidence is valid.
    """

    scoped = strict_json_loads(_canonical_json(receipt))
    if not isinstance(scoped, dict):
        raise R10BoundedInverseError("receiver receipt root is not an object")
    contract = scoped.get("asymmetric_codec_contract")
    if not isinstance(contract, dict):
        raise R10BoundedInverseError("receiver receipt lacks its asymmetric codec contract")
    contract["encoder_compiler"] = "finite_bounded_source_rgb_inverse_fit_and_selection"
    inherited = scoped.pop("maximum_inverse_solve_hook", None)
    if not isinstance(inherited, dict):
        raise R10BoundedInverseError("receiver receipt lacks the inherited inverse-solve hook")
    scoped["bounded_inverse_family_hook"] = {
        "status": "BOUNDED_ACTION_FAMILY_EXECUTED_MORE_FAMILIES_UNENUMERATED",
        "reason": "R10_GLOBAL_ACTION_UNIVERSE_AND_SCORER_NATIVE_SELECTION_OWED",
        "learned_state_policy": inherited.get("learned_state_policy"),
    }
    debt = scoped.get("strict_proxy_debt")
    if not isinstance(debt, list):
        raise R10BoundedInverseError("receiver receipt strict proxy debt is not a list")
    scoped["strict_proxy_debt"] = [
        (
            "R10_GLOBAL_ACTION_UNIVERSE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED"
            if row == "R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED"
            else row
        )
        for row in debt
    ]
    encoded = _canonical_json(scoped).lower()
    for forbidden in (b"maximum_inverse", b"maximum inverse", b"unbounded_encode"):
        if forbidden in encoded:
            raise R10BoundedInverseError("bounded receiver receipt retained a false maximum/unbounded label")
    return scoped


def _build_g23_adapter(packet_bytes: bytes, wrapper: bytes, data_offset: int) -> R10G23PhysicalGroupAdapterV1:
    g27_adapter = build_r10_selected_solution_adapter(packet_bytes)
    group = G17PhysicalCodingGroupV1(
        group_id=f"g32.r10.{_sha256(packet_bytes)[:16]}",
        exact_archive_bytes=wrapper,
        member_name=G32_WRAPPER_MEMBER,
        exact_member_bytes=packet_bytes,
        archive_offset=0,
        exact_range_bytes=wrapper,
        coder_owner="G35_R10_BOUNDED_INVERSE_FITTER",
        container_owner="G32_DETERMINISTIC_STORE_ZIP",
        receiver_consumer="G27_R10_PUBLIC_RECEIVER",
        receiver_operation=_r10.R10_RECEIVER_OPERATION,
        logical_owner_ids=G32_WRAPPER_OWNER_IDS,
    )
    spans = tuple(
        R10ArchiveSectionSpanV1(
            section=span.section,
            packet_offset=span.byte_offset,
            member_offset=span.byte_offset,
            archive_offset=data_offset + span.byte_offset,
            byte_length=span.byte_length,
            span_sha256=span.range_sha256,
        )
        for span in packet_physical_spans(packet_bytes)
    )
    constraint_spans = tuple(
        {
            "constraint": span.constraint,
            "section": span.section,
            "packet_offset": span.packet_offset,
            "member_offset": span.packet_offset,
            "archive_offset": data_offset + span.packet_offset,
            "byte_length": span.byte_length,
            "operand_sha256": span.operand_sha256,
            "physical_coding_group_id": group.group_id,
        }
        for span in g27_adapter.operand_spans
    )
    adapter_identity = _sha256(
        _canonical_json(
            {
                "packet_sha256": g27_adapter.packet_sha256,
                "receiver_operation": g27_adapter.receiver_operation,
                "operand_spans": [asdict(span) for span in g27_adapter.operand_spans],
            }
        )
    )
    packet = parse_r10_packet(packet_bytes)
    continuation = R10ContinuationEquivalenceIdentityV1(
        schema="taskspace_r10_continuation_equivalence_identity.v1",
        source_sha256=packet.identity.source_sha256,
        pair_population_sha256=packet.identity.pair_population_sha256,
        base_realization_sha256=packet.identity.base_realization_sha256,
        packet_sha256=_sha256(packet_bytes),
        packet_bytes=len(packet_bytes),
        receiver_operation=_r10.R10_RECEIVER_OPERATION,
        receiver_source_closure_sha256=str(g27_adapter.public_inflate_receiver_abi["closure_sha256"]),
        output_shape=(packet.n_pairs, 2, packet.height, packet.width, 3),
        output_dtype="uint8",
        canonical_pair_order=packet.pair_indices,
    )
    return R10G23PhysicalGroupAdapterV1(
        schema=G32_ADAPTER_SCHEMA,
        physical_group=group,
        packet_sha256=_sha256(packet_bytes),
        packet_bytes=len(packet_bytes),
        member_data_archive_offset=data_offset,
        section_spans=spans,
        g27_adapter_sha256=adapter_identity,
        g27_receiver_operation=_r10.R10_RECEIVER_OPERATION,
        continuation_equivalence_sha256=continuation.identity_sha256,
        constraint_operand_spans=constraint_spans,
    )


def compile_r10_bounded_inverse(
    base_pairs: np.ndarray,
    source_pairs: np.ndarray,
    *,
    source_sha256: str,
    config: R10BoundedInverseConfigV1,
    base_realization_sha256: str | None = None,
    execute_bounded_decode: bool = True,
    scratch_directory: str | Path | None = None,
    resume_stages: Mapping[str, Mapping[str, Any]] | None = None,
    stage_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    governance_callback: Callable[[], Mapping[str, Any]] | None = None,
    resume_ranges: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    range_callback: Callable[[str, int, int, Mapping[str, Any]], None] | None = None,
) -> R10BoundedInverseResultV1:
    """Fit, serialize, audit, and optionally execute one bounded-family R10 packet."""

    _validate_pair_arrays(base_pairs, source_pairs, config)
    if config.pair_count == 600:
        if governance_callback is None:
            raise R10BoundedInverseError("n600 requires a live canonical governance callback")
        governance = governance_callback()
        if (
            not isinstance(governance, Mapping)
            or governance.get("required") is not True
            or governance.get("verified") is not True
        ):
            raise R10BoundedInverseError("n600 governance callback did not return a verified canonical claim")
    if type(source_sha256) is not str or len(source_sha256) != 64 or source_sha256.lower() != source_sha256:
        raise R10BoundedInverseError("source_sha256 must be canonical lowercase SHA-256")
    resumed = resume_stages or {}
    resumed_ranges: dict[str, Sequence[Mapping[str, Any]]] = {
        stage: tuple(records) for stage, records in (resume_ranges or {}).items()
    }

    def range_payloads(stage: str) -> tuple[Mapping[str, Any], ...]:
        records = resumed_ranges.get(stage, ())
        expected_first = 0
        payloads: list[Mapping[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {"first_pair", "stop_pair", "payload"}:
                raise R10BoundedInverseError(f"resumed {stage} range record has a noncanonical key set")
            first_pair = record["first_pair"]
            stop_pair = record["stop_pair"]
            if (
                type(first_pair) is not int
                or type(stop_pair) is not int
                or first_pair != expected_first
                or stop_pair != min(config.pair_count, first_pair + config.chunk_pairs)
            ):
                raise R10BoundedInverseError(f"resumed {stage} range coordinates are not a canonical prefix")
            payload = record["payload"]
            if not isinstance(payload, Mapping):
                raise R10BoundedInverseError(f"resumed {stage} range payload is not an object")
            payloads.append(payload)
            expected_first = stop_pair
        return tuple(payloads)

    def publish_range(stage: str, first_pair: int, stop_pair: int, payload: Mapping[str, Any]) -> None:
        if config.pair_count == 600:
            assert governance_callback is not None
            governance = governance_callback()
            if governance.get("required") is not True or governance.get("verified") is not True:
                raise R10BoundedInverseError(f"n600 governance lapsed before {stage} range")
        _require_finite_numbers(payload, path=f"{stage}[{first_pair}:{stop_pair}]")
        if range_callback is not None:
            range_callback(stage, first_pair, stop_pair, payload)

    def publish(stage: str, payload: Mapping[str, Any]) -> None:
        if config.pair_count == 600:
            assert governance_callback is not None
            governance = governance_callback()
            if governance.get("required") is not True or governance.get("verified") is not True:
                raise R10BoundedInverseError(f"n600 governance lapsed before stage {stage}")
        _require_finite_numbers(payload, path=stage)
        if stage_callback is not None:
            stage_callback(stage, payload)

    pair_state = {
        "pair_indices": list(range(config.pair_count)),
        "source_frame_ids": [[2 * value, 2 * value + 1] for value in range(config.pair_count)],
    }
    publish("020_pair_index", pair_state)
    geometry_state = resumed.get("030_geometry")
    xip2_state = resumed.get("040_xip2")
    if geometry_state is not None and xip2_state is not None:
        pitch_q20 = int(geometry_state["pitch_q20"])
        if pitch_q20 not in config.pitch_candidates_q20:
            raise R10BoundedInverseError("resumed pitch is outside the configured finite lattice")
        xi = np.asarray(xip2_state["xi"], dtype=np.float64)
        if xi.shape != (config.pair_count, 6) or not np.isfinite(xi).all():
            raise R10BoundedInverseError("resumed XIP2 coordinates have wrong shape or non-finite values")
        xip2_payload = bytes.fromhex(str(xip2_state["xip2_payload_hex"]))
        decoded_xi = decode_xi_payload(xip2_payload)
        if not np.array_equal(decoded_xi, xi):
            raise R10BoundedInverseError("resumed XIP2 coordinates differ from exact counted payload")
        inverse_trace = {
            "pitch_lattice": list(geometry_state["pitch_lattice"]),
            "pair_solves": list(xip2_state["pair_solves"]),
            "quantized_objective": int(xip2_state["quantized_objective"]),
            "quantized_objective_full_resolution": int(xip2_state["quantized_objective_full_resolution"]),
            "full_resolution_controls": list(xip2_state["full_resolution_controls"]),
            "rejected_pairs_full_resolution": list(xip2_state["rejected_pairs_full_resolution"]),
            "iterations": int(xip2_state["iterations"]),
        }
    elif geometry_state is not None:
        pitch_q20 = int(geometry_state["pitch_q20"])
        pitch_lattice = list(geometry_state["pitch_lattice"])
        if pitch_q20 not in config.pitch_candidates_q20:
            raise R10BoundedInverseError("resumed pitch is outside the configured finite lattice")
        pitch_q20, xi, inverse_trace = _fit_pitch_and_xi(
            base_pairs,
            source_pairs,
            config,
            fixed_pitch_q20=pitch_q20,
            resume_pair_ranges=range_payloads("040_xip2"),
            range_callback=lambda first, stop, payload: publish_range("040_xip2", first, stop, payload),
        )
        inverse_trace["pitch_lattice"] = pitch_lattice
        xip2_payload = inverse_trace.pop("xip2_payload")
    elif xip2_state is not None:
        raise R10BoundedInverseError("XIP2 resume checkpoint requires its predecessor geometry checkpoint")
    else:
        pitch_q20, xi, inverse_trace = _fit_pitch_and_xi(
            base_pairs,
            source_pairs,
            config,
            resume_pair_ranges=range_payloads("040_xip2"),
            range_callback=lambda first, stop, payload: publish_range("040_xip2", first, stop, payload),
        )
        xip2_payload = inverse_trace.pop("xip2_payload")
    publish(
        "030_geometry",
        {"pitch_q20": pitch_q20, "pitch_lattice": inverse_trace["pitch_lattice"]},
    )
    publish(
        "040_xip2",
        {
            "xi": xi.tolist(),
            "xip2_payload_hex": xip2_payload.hex(),
            "pair_solves": inverse_trace["pair_solves"],
            "quantized_objective": inverse_trace["quantized_objective"],
            "quantized_objective_full_resolution": inverse_trace["quantized_objective_full_resolution"],
            "full_resolution_controls": inverse_trace["full_resolution_controls"],
            "rejected_pairs_full_resolution": inverse_trace["rejected_pairs_full_resolution"],
            "iterations": inverse_trace["iterations"],
        },
    )
    pitch = pitch_q20 / float(1 << 20)
    scratch_paths: list[str] = []
    warped, scratch = _allocate_intermediate("040_warped", config, scratch_directory)
    if scratch is not None:
        scratch_paths.append(scratch)
    for pair_index in range(config.pair_count):
        warped[pair_index] = _warp_pair_frame0(base_pairs[pair_index], xi[pair_index], pitch)

    feature_records: list[R10BaseFeatureRecordV1] = []
    feature_rows: list[dict[str, Any]] = []
    feature_realized, scratch = _allocate_intermediate("050_feature_realized", config, scratch_directory)
    if scratch is not None:
        scratch_paths.append(scratch)
    feature_state = resumed.get("050_base_feature")
    if feature_state is not None:
        for row in feature_state["records"]:
            feature_records.append(
                R10BaseFeatureRecordV1(
                    int(row["luma_bias_q8"]),
                    int(row["contrast_q8"]),
                    int(row["edge_gain_q8"]),
                    int(row["feature_gain_q8"]),
                    tuple(int(value) for value in row["channel_weights_q8"]),
                )
            )
        feature_rows = [dict(row) for row in feature_state["fit_rows"]]
        if len(feature_records) != config.pair_count or len(feature_rows) != config.pair_count:
            raise R10BoundedInverseError("resumed BASE_FEATURE population is incomplete")
    else:
        for payload in range_payloads("050_base_feature"):
            if set(payload) != {"records", "fit_rows"}:
                raise R10BoundedInverseError("resumed BASE_FEATURE range payload has a noncanonical key set")
            range_records = payload["records"]
            range_rows = payload["fit_rows"]
            if (
                not isinstance(range_records, list)
                or not isinstance(range_rows, list)
                or len(range_records) != len(range_rows)
            ):
                raise R10BoundedInverseError("resumed BASE_FEATURE range population is malformed")
            for record_row, fit_row in zip(range_records, range_rows, strict=True):
                pair_index = len(feature_records)
                if fit_row.get("pair_index") != pair_index:
                    raise R10BoundedInverseError("resumed BASE_FEATURE range coordinates drifted")
                feature_records.append(
                    R10BaseFeatureRecordV1(
                        int(record_row["luma_bias_q8"]),
                        int(record_row["contrast_q8"]),
                        int(record_row["edge_gain_q8"]),
                        int(record_row["feature_gain_q8"]),
                        tuple(int(value) for value in record_row["channel_weights_q8"]),
                    )
                )
                feature_rows.append(dict(fit_row))
        for first_pair in range(len(feature_records), config.pair_count, config.chunk_pairs):
            stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
            new_records: list[R10BaseFeatureRecordV1] = []
            new_rows: list[dict[str, Any]] = []
            for pair_index in range(first_pair, stop_pair):
                record, row = _fit_base_feature_record(
                    warped[pair_index],
                    source_pairs[pair_index],
                    sample_stride=config.sample_stride,
                )
                feature_records.append(record)
                fit_row = {"pair_index": pair_index, **row}
                feature_rows.append(fit_row)
                new_records.append(record)
                new_rows.append(fit_row)
            publish_range(
                "050_base_feature",
                first_pair,
                stop_pair,
                {"records": [asdict(row) for row in new_records], "fit_rows": new_rows},
            )
    if len(feature_records) != config.pair_count or len(feature_rows) != config.pair_count:
        raise R10BoundedInverseError("BASE_FEATURE range state did not close the population")
    if any(row.get("pair_index") != pair_index for pair_index, row in enumerate(feature_rows)):
        raise R10BoundedInverseError("BASE_FEATURE range fit rows are not canonical")
    for pair_index, record in enumerate(feature_records):
        for role in (0, 1):
            feature_realized[pair_index, role] = _r10._apply_feature_relay(
                warped[pair_index, role],
                record,
                None,
                (),
                np.zeros(6, dtype=np.int64),
                frame_role=role,
            )
    publish(
        "050_base_feature",
        {"records": [asdict(row) for row in feature_records], "fit_rows": feature_rows},
    )

    label_output, scratch = _allocate_intermediate(
        "060_support_labels",
        config,
        scratch_directory,
        shape=(config.pair_count, config.height, config.width),
    )
    if scratch is not None:
        scratch_paths.append(scratch)
    support_labels, support_rows = _source_residual_support_labels(
        feature_realized,
        source_pairs,
        config,
        output=label_output,
    )
    dash_cfg = DashPhaseConfig(
        lane_class=1,
        min_area=3,
        border_px=1,
        match_radius_px=float(max(config.height, config.width)),
        q_px=1.0,
        tilt_bins=16,
        dormant_max_frames=30,
        gap_xi="interp",
        s_t=0.0,
        s_r=0.0,
        pitch=pitch,
        include_xi=False,
    )
    texture_state = resumed.get("060_texture")
    if texture_state is None:
        dash1_payload, dash_report, decoded_dashes = encode_dash_phase_carrier(
            support_labels,
            xi,
            dash_cfg,
            geom=GroundHomographyGeom.eon(native_hw=(config.height, config.width), pitch=pitch),
        )
        dash_report_dict = asdict(dash_report)
    else:
        dash1_payload = bytes.fromhex(str(texture_state["dash1_payload_hex"]))
        decoded_dashes = decode_dash_phase_carrier(
            dash1_payload,
            xi_twists_external=xi,
            geom=GroundHomographyGeom.eon(native_hw=(config.height, config.width), pitch=pitch),
        )
        dash_report_dict = dict(texture_state["dash_report"])
    replayed_dashes = decode_dash_phase_carrier(
        dash1_payload,
        xi_twists_external=xi,
        geom=GroundHomographyGeom.eon(native_hw=(config.height, config.width), pitch=pitch),
    )
    if tuple(tuple(frame) for frame in replayed_dashes) != tuple(tuple(frame) for frame in decoded_dashes):
        raise R10BoundedInverseError("DASH1 strict external-XIP2 replay drifted")

    textures: list[R10TextureRecordV1] = []
    texture_rows: list[dict[str, Any]] = []
    textured, scratch = _allocate_intermediate("070_textured", config, scratch_directory)
    if scratch is not None:
        scratch_paths.append(scratch)
    if texture_state is not None:
        textures = [
            R10TextureRecordV1(
                int(row["amplitude_q8"]),
                int(row["frequency_q8"]),
                int(row["phase_q10"]),
                int(row["texture_gain_q8"]),
            )
            for row in texture_state["records"]
        ]
        texture_rows = [dict(row) for row in texture_state["fit_rows"]]
        if len(textures) != config.pair_count or len(texture_rows) != config.pair_count:
            raise R10BoundedInverseError("resumed TEXTURE population is incomplete")
    else:
        dash1_sha256 = _sha256(dash1_payload)
        for payload in range_payloads("060_texture"):
            if set(payload) != {"dash1_sha256", "records", "fit_rows"} or payload["dash1_sha256"] != dash1_sha256:
                raise R10BoundedInverseError("resumed TEXTURE range is not bound to the retained DASH1 payload")
            range_records = payload["records"]
            range_rows = payload["fit_rows"]
            if (
                not isinstance(range_records, list)
                or not isinstance(range_rows, list)
                or len(range_records) != len(range_rows)
            ):
                raise R10BoundedInverseError("resumed TEXTURE range population is malformed")
            for record_row, fit_row in zip(range_records, range_rows, strict=True):
                pair_index = len(textures)
                if fit_row.get("pair_index") != pair_index:
                    raise R10BoundedInverseError("resumed TEXTURE range coordinates drifted")
                textures.append(
                    R10TextureRecordV1(
                        int(record_row["amplitude_q8"]),
                        int(record_row["frequency_q8"]),
                        int(record_row["phase_q10"]),
                        int(record_row["texture_gain_q8"]),
                    )
                )
                texture_rows.append(dict(fit_row))
        for first_pair in range(len(textures), config.pair_count, config.chunk_pairs):
            stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
            new_records: list[R10TextureRecordV1] = []
            new_rows: list[dict[str, Any]] = []
            for pair_index in range(first_pair, stop_pair):
                record, row = _fit_texture_record(
                    warped[pair_index],
                    source_pairs[pair_index],
                    feature_records[pair_index],
                    decoded_dashes[pair_index],
                    config,
                )
                textures.append(record)
                fit_row = {"pair_index": pair_index, **row}
                texture_rows.append(fit_row)
                new_records.append(record)
                new_rows.append(fit_row)
            publish_range(
                "060_texture",
                first_pair,
                stop_pair,
                {
                    "dash1_sha256": dash1_sha256,
                    "records": [asdict(row) for row in new_records],
                    "fit_rows": new_rows,
                },
            )
    if len(textures) != config.pair_count or len(texture_rows) != config.pair_count:
        raise R10BoundedInverseError("TEXTURE range state did not close the population")
    if any(row.get("pair_index") != pair_index for pair_index, row in enumerate(texture_rows)):
        raise R10BoundedInverseError("TEXTURE range fit rows are not canonical")
    for pair_index, record in enumerate(textures):
        for role in (0, 1):
            textured[pair_index, role] = _r10._apply_feature_relay(
                warped[pair_index, role],
                feature_records[pair_index],
                record,
                decoded_dashes[pair_index],
                np.zeros(6, dtype=np.int64),
                frame_role=role,
            )
    publish(
        "060_texture",
        {
            "records": [asdict(row) for row in textures],
            "fit_rows": texture_rows,
            "dash1_payload_hex": dash1_payload.hex(),
            "dash_report": dash_report_dict,
        },
    )

    knots: list[R10ShootingKnotV1] = []
    knot_rows: list[dict[str, Any]] = []
    knot_state = resumed.get("070_shooting_knot")
    if knot_state is not None:
        knots = [
            R10ShootingKnotV1(int(row["pair_index"]), *(int(value) for value in row["deltas"]))
            for row in knot_state["records"]
        ]
        knot_rows = [dict(row) for row in knot_state["fit_rows"]]
        if len(knots) != config.pair_count or len(knot_rows) != config.pair_count:
            raise R10BoundedInverseError("resumed SHOOTING_KNOT population is incomplete")
    else:
        for payload in range_payloads("070_shooting_knot"):
            if set(payload) != {"records", "fit_rows"}:
                raise R10BoundedInverseError("resumed SHOOTING_KNOT range payload has a noncanonical key set")
            range_records = payload["records"]
            range_rows = payload["fit_rows"]
            if (
                not isinstance(range_records, list)
                or not isinstance(range_rows, list)
                or len(range_records) != len(range_rows)
            ):
                raise R10BoundedInverseError("resumed SHOOTING_KNOT range population is malformed")
            for record_row, fit_row in zip(range_records, range_rows, strict=True):
                pair_index = len(knots)
                if fit_row.get("pair_index") != pair_index or record_row.get("pair_index") != pair_index:
                    raise R10BoundedInverseError("resumed SHOOTING_KNOT range coordinates drifted")
                knots.append(
                    R10ShootingKnotV1(
                        pair_index,
                        *(int(value) for value in record_row["deltas"]),
                    )
                )
                knot_rows.append(dict(fit_row))
        for first_pair in range(len(knots), config.pair_count, config.chunk_pairs):
            stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
            new_records: list[R10ShootingKnotV1] = []
            new_rows: list[dict[str, Any]] = []
            for pair_index in range(first_pair, stop_pair):
                residual = source_pairs[pair_index].astype(np.int32) - textured[pair_index].astype(np.int32)
                weights = np.asarray(feature_records[pair_index].channel_weights_q8, dtype=np.float64)
                denominator = float(np.sum(weights * weights))
                projected = (
                    0
                    if denominator == 0.0
                    else int(
                        np.clip(
                            np.rint(float(np.mean(residual, axis=(0, 1, 2)) @ weights) * 65536.0 / denominator),
                            -32768,
                            32767,
                        )
                    )
                )
                knot = R10ShootingKnotV1(pair_index, projected, 0, 0, 0, 0, 0)
                fit_row = {"pair_index": pair_index, "luma_bias_delta_q8": projected}
                knots.append(knot)
                knot_rows.append(fit_row)
                new_records.append(knot)
                new_rows.append(fit_row)
            publish_range(
                "070_shooting_knot",
                first_pair,
                stop_pair,
                {
                    "records": [{"pair_index": row.pair_index, "deltas": list(row.deltas)} for row in new_records],
                    "fit_rows": new_rows,
                },
            )
    publish(
        "070_shooting_knot",
        {
            "records": [{"pair_index": row.pair_index, "deltas": list(row.deltas)} for row in knots],
            "fit_rows": knot_rows,
        },
    )
    publish(
        "080_dash1",
        {
            "dash1_payload_sha256": _sha256(dash1_payload),
            "dash1_payload_bytes": len(dash1_payload),
            "external_xip2_sha256": _sha256(xip2_payload),
            "include_xi": False,
        },
    )

    polygon_state = resumed.get("090_pullback_polygon")
    if polygon_state is None:
        polygons = tuple(
            _polygon_from_support(row, config.height, config.width, config.support_radius_px) for row in support_rows
        )
    else:
        polygons = tuple(
            R10PullbackPolygonV1(
                int(row["pair_index"]),
                tuple(tuple(int(value) for value in point) for point in row["vertices_q15"]),
            )
            for row in polygon_state["records"]
        )
        if len(polygons) != config.pair_count:
            raise R10BoundedInverseError("resumed PULLBACK_POLYGON population is incomplete")
    publish("090_pullback_polygon", {"records": [asdict(row) for row in polygons]})
    flows: list[R10StratifiedFlowV1] = []
    flow_rows: list[dict[str, Any]] = []
    flow_state = resumed.get("100_stratified_flow")
    if flow_state is not None:
        flows = [
            R10StratifiedFlowV1(
                int(row["pair_index"]),
                tuple(int(value) for value in row["affine_q8"]),
            )
            for row in flow_state["records"]
        ]
        flow_rows = [dict(row) for row in flow_state["fit_rows"]]
        if len(flows) != config.pair_count or len(flow_rows) != config.pair_count:
            raise R10BoundedInverseError("resumed STRATIFIED_FLOW population is incomplete")
    else:
        for payload in range_payloads("100_stratified_flow"):
            if set(payload) != {"records", "fit_rows"}:
                raise R10BoundedInverseError("resumed STRATIFIED_FLOW range payload has a noncanonical key set")
            range_records = payload["records"]
            range_rows = payload["fit_rows"]
            if (
                not isinstance(range_records, list)
                or not isinstance(range_rows, list)
                or len(range_records) != len(range_rows)
            ):
                raise R10BoundedInverseError("resumed STRATIFIED_FLOW range population is malformed")
            for record_row, fit_row in zip(range_records, range_rows, strict=True):
                pair_index = len(flows)
                if fit_row.get("pair_index") != pair_index or record_row.get("pair_index") != pair_index:
                    raise R10BoundedInverseError("resumed STRATIFIED_FLOW range coordinates drifted")
                flows.append(
                    R10StratifiedFlowV1(
                        pair_index,
                        tuple(int(value) for value in record_row["affine_q8"]),
                    )
                )
                flow_rows.append(dict(fit_row))
        for first_pair in range(len(flows), config.pair_count, config.chunk_pairs):
            stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
            new_records: list[R10StratifiedFlowV1] = []
            new_rows: list[dict[str, Any]] = []
            for pair_index in range(first_pair, stop_pair):
                polygon = polygons[pair_index]
                flow, row = _fit_flow_record(
                    base_pairs[pair_index, 0],
                    source_pairs[pair_index, 0],
                    xi[pair_index],
                    pitch,
                    polygon,
                    config,
                )
                fit_row = {"pair_index": pair_index, **row}
                flows.append(flow)
                flow_rows.append(fit_row)
                new_records.append(flow)
                new_rows.append(fit_row)
            publish_range(
                "100_stratified_flow",
                first_pair,
                stop_pair,
                {"records": [asdict(row) for row in new_records], "fit_rows": new_rows},
            )
    publish(
        "100_stratified_flow",
        {"records": [asdict(row) for row in flows], "fit_rows": flow_rows},
    )
    initial_corner: dict[str, Any] = {
        "feature_records": tuple(feature_records),
        "feature_rows": tuple(feature_rows),
        "textures": tuple(textures),
        "texture_rows": tuple(texture_rows),
        "knots": tuple(knots),
        "knot_rows": tuple(knot_rows),
        "dash1_payload": dash1_payload,
        "dash_report": dash_report_dict,
        "decoded_dashes": tuple(tuple(frame) for frame in decoded_dashes),
        "support_rows": tuple(support_rows),
        "polygons": tuple(polygons),
        "flows": tuple(flows),
        "flow_rows": tuple(flow_rows),
    }

    joint_state = resumed.get("110_joint_refit")
    if joint_state is not None and not any(stage.startswith("110_joint_") for stage in resumed_ranges):
        if set(joint_state) != {
            "interaction_passes",
            "block_order",
            "packet_sha256",
            "packet_bytes",
            "packet_hex",
            "trace",
        }:
            raise R10BoundedInverseError("resumed JOINT_REFIT stage has a noncanonical key set")
        retained_packet_bytes = bytes.fromhex(str(joint_state["packet_hex"]))
        if joint_state["packet_sha256"] != _sha256(retained_packet_bytes) or joint_state["packet_bytes"] != len(
            retained_packet_bytes
        ):
            raise R10BoundedInverseError("resumed JOINT_REFIT packet identity drifted")
        retained_packet = parse_r10_packet(retained_packet_bytes)
        if (
            retained_packet.n_pairs != config.pair_count
            or retained_packet.pitch_q20 != pitch_q20
            or retained_packet.xip2_payload != xip2_payload
        ):
            raise R10BoundedInverseError("resumed JOINT_REFIT packet coordinates differ from its predecessors")
        retained_trace = joint_state["trace"]
        retained_feature_rows = retained_trace["base_feature"]
        retained_texture_rows = retained_trace["texture_dash1"]["texture_rows"]
        retained_knot_rows = retained_trace["shooting_knot"]
        retained_flow_rows = retained_trace["pullback_polygon_stratified_flow"]["flow_rows"]
        retained_polygons = retained_packet.pullback_polygons
        retained_polygon_sha256 = _sha256(_canonical_json([asdict(row) for row in retained_polygons]))

        def retained_ranges(records: Sequence[Any], rows: Sequence[Mapping[str, Any]], **extra: Any):
            ranges: list[Mapping[str, Any]] = []
            for first_pair in range(0, config.pair_count, config.chunk_pairs):
                stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
                ranges.append(
                    {
                        "first_pair": first_pair,
                        "stop_pair": stop_pair,
                        "payload": {
                            **extra,
                            "records": [asdict(row) for row in records[first_pair:stop_pair]],
                            "fit_rows": [dict(row) for row in rows[first_pair:stop_pair]],
                        },
                    }
                )
            return tuple(ranges)

        resumed_ranges["110_joint_base_feature"] = retained_ranges(
            retained_packet.base_features,
            retained_feature_rows,
        )
        resumed_ranges["110_joint_texture"] = retained_ranges(
            retained_packet.textures,
            retained_texture_rows,
            dash1_sha256=_sha256(retained_packet.dash1_payload),
        )
        resumed_ranges["110_joint_shooting_knot"] = tuple(
            {
                "first_pair": first_pair,
                "stop_pair": stop_pair,
                "payload": {
                    "records": [
                        {"pair_index": row.pair_index, "deltas": list(row.deltas)}
                        for row in retained_packet.shooting_knots[first_pair:stop_pair]
                    ],
                    "fit_rows": [dict(row) for row in retained_knot_rows[first_pair:stop_pair]],
                },
            }
            for first_pair in range(0, config.pair_count, config.chunk_pairs)
            for stop_pair in (min(config.pair_count, first_pair + config.chunk_pairs),)
        )
        resumed_ranges["110_joint_stratified_flow"] = retained_ranges(
            retained_packet.stratified_flows,
            retained_flow_rows,
            polygon_sha256=retained_polygon_sha256,
        )

    # One real post-interaction block-coordinate pass.  GEOMETRY+XIP2 retain
    # their finite global inverse solution; every downstream block is re-fit on
    # frames that already include the initially selected polygon/flow physics.
    joint_refit_trace: dict[str, Any] = {
        "geometry_xip2": {
            "revalidated_finite_solution": True,
            "pitch_q20": pitch_q20,
            "xip2_sha256": _sha256(xip2_payload),
        }
    }
    for pair_index in range(config.pair_count):
        geom = GroundHomographyGeom.eon(native_hw=(config.height, config.width), pitch=pitch)
        mask = _r10._polygon_mask(config.height, config.width, polygons[pair_index])
        extra = affine_extra_flow(
            (config.height, config.width),
            np.asarray(flows[pair_index].affine_q8, dtype=np.float64) / 256.0,
        )
        warped[pair_index] = base_pairs[pair_index]
        warped[pair_index, 0] = stratified_warp_uint8(
            base_pairs[pair_index, 0],
            xi[pair_index],
            geom,
            offplane_mask=mask,
            extra_flow=extra,
        )

    refit_feature_records: list[R10BaseFeatureRecordV1] = []
    refit_feature_rows: list[dict[str, Any]] = []
    for payload in range_payloads("110_joint_base_feature"):
        if set(payload) != {"records", "fit_rows"}:
            raise R10BoundedInverseError("resumed joint BASE_FEATURE range has a noncanonical key set")
        for record_row, fit_row in zip(payload["records"], payload["fit_rows"], strict=True):
            pair_index = len(refit_feature_records)
            if fit_row.get("pair_index") != pair_index:
                raise R10BoundedInverseError("resumed joint BASE_FEATURE coordinates drifted")
            refit_feature_records.append(
                R10BaseFeatureRecordV1(
                    int(record_row["luma_bias_q8"]),
                    int(record_row["contrast_q8"]),
                    int(record_row["edge_gain_q8"]),
                    int(record_row["feature_gain_q8"]),
                    tuple(int(value) for value in record_row["channel_weights_q8"]),
                )
            )
            refit_feature_rows.append(dict(fit_row))
    for first_pair in range(len(refit_feature_records), config.pair_count, config.chunk_pairs):
        stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
        new_records: list[R10BaseFeatureRecordV1] = []
        new_rows: list[dict[str, Any]] = []
        for pair_index in range(first_pair, stop_pair):
            record, row = _fit_base_feature_record(
                warped[pair_index],
                source_pairs[pair_index],
                sample_stride=config.sample_stride,
            )
            fit_row = {"pair_index": pair_index, **row}
            refit_feature_records.append(record)
            refit_feature_rows.append(fit_row)
            new_records.append(record)
            new_rows.append(fit_row)
        publish_range(
            "110_joint_base_feature",
            first_pair,
            stop_pair,
            {"records": [asdict(row) for row in new_records], "fit_rows": new_rows},
        )
    for pair_index, record in enumerate(refit_feature_records):
        for role in (0, 1):
            feature_realized[pair_index, role] = _r10._apply_feature_relay(
                warped[pair_index, role],
                record,
                None,
                (),
                np.zeros(6, dtype=np.int64),
                frame_role=role,
            )
    feature_records = refit_feature_records
    feature_rows = refit_feature_rows
    joint_refit_trace["base_feature"] = refit_feature_rows

    support_labels, support_rows = _source_residual_support_labels(
        feature_realized,
        source_pairs,
        config,
        output=label_output,
    )
    dash1_payload, dash_report, decoded_dashes = encode_dash_phase_carrier(
        support_labels,
        xi,
        dash_cfg,
        geom=GroundHomographyGeom.eon(native_hw=(config.height, config.width), pitch=pitch),
    )
    dash_report_dict = asdict(dash_report)
    refit_textures: list[R10TextureRecordV1] = []
    refit_texture_rows: list[dict[str, Any]] = []
    joint_dash_sha256 = _sha256(dash1_payload)
    for payload in range_payloads("110_joint_texture"):
        if set(payload) != {"dash1_sha256", "records", "fit_rows"} or payload["dash1_sha256"] != joint_dash_sha256:
            raise R10BoundedInverseError("resumed joint TEXTURE range is not bound to its DASH1 payload")
        for record_row, fit_row in zip(payload["records"], payload["fit_rows"], strict=True):
            pair_index = len(refit_textures)
            if fit_row.get("pair_index") != pair_index:
                raise R10BoundedInverseError("resumed joint TEXTURE coordinates drifted")
            refit_textures.append(
                R10TextureRecordV1(
                    int(record_row["amplitude_q8"]),
                    int(record_row["frequency_q8"]),
                    int(record_row["phase_q10"]),
                    int(record_row["texture_gain_q8"]),
                )
            )
            refit_texture_rows.append(dict(fit_row))
    for first_pair in range(len(refit_textures), config.pair_count, config.chunk_pairs):
        stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
        new_records: list[R10TextureRecordV1] = []
        new_rows: list[dict[str, Any]] = []
        for pair_index in range(first_pair, stop_pair):
            record, row = _fit_texture_record(
                warped[pair_index],
                source_pairs[pair_index],
                feature_records[pair_index],
                decoded_dashes[pair_index],
                config,
            )
            fit_row = {"pair_index": pair_index, **row}
            refit_textures.append(record)
            refit_texture_rows.append(fit_row)
            new_records.append(record)
            new_rows.append(fit_row)
        publish_range(
            "110_joint_texture",
            first_pair,
            stop_pair,
            {
                "dash1_sha256": joint_dash_sha256,
                "records": [asdict(row) for row in new_records],
                "fit_rows": new_rows,
            },
        )
    for pair_index, record in enumerate(refit_textures):
        for role in (0, 1):
            textured[pair_index, role] = _r10._apply_feature_relay(
                warped[pair_index, role],
                feature_records[pair_index],
                record,
                decoded_dashes[pair_index],
                np.zeros(6, dtype=np.int64),
                frame_role=role,
            )
    textures = refit_textures
    texture_rows = refit_texture_rows
    joint_refit_trace["texture_dash1"] = {
        "texture_rows": refit_texture_rows,
        "dash1_sha256": _sha256(dash1_payload),
        "support_rows": list(support_rows),
    }

    refit_knots: list[R10ShootingKnotV1] = []
    refit_knot_rows: list[dict[str, Any]] = []
    for payload in range_payloads("110_joint_shooting_knot"):
        if set(payload) != {"records", "fit_rows"}:
            raise R10BoundedInverseError("resumed joint SHOOTING_KNOT range has a noncanonical key set")
        for record_row, fit_row in zip(payload["records"], payload["fit_rows"], strict=True):
            pair_index = len(refit_knots)
            if fit_row.get("pair_index") != pair_index or record_row.get("pair_index") != pair_index:
                raise R10BoundedInverseError("resumed joint SHOOTING_KNOT coordinates drifted")
            refit_knots.append(
                R10ShootingKnotV1(
                    pair_index,
                    *(int(value) for value in record_row["deltas"]),
                )
            )
            refit_knot_rows.append(dict(fit_row))
    for first_pair in range(len(refit_knots), config.pair_count, config.chunk_pairs):
        stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
        new_records: list[R10ShootingKnotV1] = []
        new_rows: list[dict[str, Any]] = []
        for pair_index in range(first_pair, stop_pair):
            residual = source_pairs[pair_index].astype(np.int32) - textured[pair_index].astype(np.int32)
            weights = np.asarray(feature_records[pair_index].channel_weights_q8, dtype=np.float64)
            denominator = float(np.sum(weights * weights))
            projected = (
                0
                if denominator == 0.0
                else int(
                    np.clip(
                        np.rint(float(np.mean(residual, axis=(0, 1, 2)) @ weights) * 65536.0 / denominator),
                        -32768,
                        32767,
                    )
                )
            )
            knot = R10ShootingKnotV1(pair_index, projected, 0, 0, 0, 0, 0)
            fit_row = {"pair_index": pair_index, "luma_bias_delta_q8": projected}
            refit_knots.append(knot)
            refit_knot_rows.append(fit_row)
            new_records.append(knot)
            new_rows.append(fit_row)
        publish_range(
            "110_joint_shooting_knot",
            first_pair,
            stop_pair,
            {
                "records": [{"pair_index": row.pair_index, "deltas": list(row.deltas)} for row in new_records],
                "fit_rows": new_rows,
            },
        )
    knots = refit_knots
    knot_rows = refit_knot_rows
    joint_refit_trace["shooting_knot"] = refit_knot_rows

    _polygon_labels, polygon_support_rows = _source_residual_support_labels(
        textured,
        source_pairs,
        config,
        output=label_output,
    )
    polygons = tuple(
        _polygon_from_support(row, config.height, config.width, config.support_radius_px)
        for row in polygon_support_rows
    )
    refit_flows: list[R10StratifiedFlowV1] = []
    refit_flow_rows: list[dict[str, Any]] = []
    polygon_sha256 = _sha256(_canonical_json([asdict(row) for row in polygons]))
    for payload in range_payloads("110_joint_stratified_flow"):
        if set(payload) != {"polygon_sha256", "records", "fit_rows"} or payload["polygon_sha256"] != polygon_sha256:
            raise R10BoundedInverseError("resumed joint STRATIFIED_FLOW range is not bound to its polygons")
        for record_row, fit_row in zip(payload["records"], payload["fit_rows"], strict=True):
            pair_index = len(refit_flows)
            if fit_row.get("pair_index") != pair_index or record_row.get("pair_index") != pair_index:
                raise R10BoundedInverseError("resumed joint STRATIFIED_FLOW coordinates drifted")
            refit_flows.append(
                R10StratifiedFlowV1(
                    pair_index,
                    tuple(int(value) for value in record_row["affine_q8"]),
                )
            )
            refit_flow_rows.append(dict(fit_row))
    for first_pair in range(len(refit_flows), config.pair_count, config.chunk_pairs):
        stop_pair = min(config.pair_count, first_pair + config.chunk_pairs)
        new_records: list[R10StratifiedFlowV1] = []
        new_rows: list[dict[str, Any]] = []
        for pair_index in range(first_pair, stop_pair):
            polygon = polygons[pair_index]
            knot_delta = _r10._shooting_delta(tuple(knots), pair_index)
            flow, row = _refit_joint_flow_record(
                base_pairs[pair_index],
                source_pairs[pair_index],
                xi=xi[pair_index],
                pitch=pitch,
                polygon=polygon,
                base_record=feature_records[pair_index],
                texture=textures[pair_index],
                dashes=decoded_dashes[pair_index],
                knot_delta=knot_delta,
                config=config,
            )
            fit_row = {"pair_index": pair_index, **row}
            refit_flows.append(flow)
            refit_flow_rows.append(fit_row)
            new_records.append(flow)
            new_rows.append(fit_row)
        publish_range(
            "110_joint_stratified_flow",
            first_pair,
            stop_pair,
            {
                "polygon_sha256": polygon_sha256,
                "records": [asdict(row) for row in new_records],
                "fit_rows": new_rows,
            },
        )
    flows = refit_flows
    flow_rows = refit_flow_rows
    joint_refit_trace["pullback_polygon_stratified_flow"] = {
        "polygons": [asdict(row) for row in polygons],
        "flow_rows": refit_flow_rows,
    }

    refit_corner: dict[str, Any] = {
        "feature_records": tuple(feature_records),
        "feature_rows": tuple(feature_rows),
        "textures": tuple(textures),
        "texture_rows": tuple(texture_rows),
        "knots": tuple(knots),
        "knot_rows": tuple(knot_rows),
        "dash1_payload": dash1_payload,
        "dash_report": dash_report_dict,
        "decoded_dashes": tuple(tuple(frame) for frame in decoded_dashes),
        "support_rows": tuple(support_rows),
        "polygons": tuple(polygons),
        "flows": tuple(flows),
        "flow_rows": tuple(flow_rows),
    }

    def corner_key_and_objective(corner: Mapping[str, Any]) -> tuple[int, bytes]:
        objective = 0
        for pair_index in range(config.pair_count):
            realized = _realize_pair_components(
                base_pairs[pair_index],
                xi=xi[pair_index],
                pitch=pitch,
                polygon=corner["polygons"][pair_index],
                flow=corner["flows"][pair_index],
                base_record=corner["feature_records"][pair_index],
                texture=corner["textures"][pair_index],
                dashes=corner["decoded_dashes"][pair_index],
                knot_delta=_r10._shooting_delta(corner["knots"], pair_index),
            )
            objective += _sse(realized[0], source_pairs[pair_index, 0])
            objective += _sse(realized[1], source_pairs[pair_index, 1])
        key = _canonical_json(
            {
                "feature_records": [asdict(row) for row in corner["feature_records"]],
                "textures": [asdict(row) for row in corner["textures"]],
                "knots": [asdict(row) for row in corner["knots"]],
                "dash1_sha256": _sha256(corner["dash1_payload"]),
                "polygons": [asdict(row) for row in corner["polygons"]],
                "flows": [asdict(row) for row in corner["flows"]],
            }
        )
        return objective, key

    corner_rows = [
        ("INITIAL_SEQUENTIAL", initial_corner, *corner_key_and_objective(initial_corner)),
        ("POST_INTERACTION_REFIT", refit_corner, *corner_key_and_objective(refit_corner)),
    ]
    selected_corner_name, selected_corner, selected_corner_objective, _selected_key = min(
        corner_rows,
        key=lambda row: (row[2], row[3], row[0]),
    )
    feature_records = list(selected_corner["feature_records"])
    feature_rows = list(selected_corner["feature_rows"])
    textures = list(selected_corner["textures"])
    texture_rows = list(selected_corner["texture_rows"])
    knots = list(selected_corner["knots"])
    knot_rows = list(selected_corner["knot_rows"])
    dash1_payload = selected_corner["dash1_payload"]
    dash_report_dict = selected_corner["dash_report"]
    decoded_dashes = selected_corner["decoded_dashes"]
    support_rows = selected_corner["support_rows"]
    polygons = selected_corner["polygons"]
    flows = list(selected_corner["flows"])
    flow_rows = list(selected_corner["flow_rows"])
    joint_refit_trace["corner_selection"] = {
        "selection_rule": "minimum exact full-resolution source RGB integer objective then canonical operands",
        "corners": [
            {"name": name, "objective": objective, "operand_key_sha256": _sha256(key)}
            for name, _corner, objective, key in corner_rows
        ],
        "selected": selected_corner_name,
        "selected_objective": selected_corner_objective,
    }

    population = pair_population_sha256(source_sha256, tuple(range(config.pair_count)))
    if base_realization_sha256 is None:
        if config.pair_count > 24:
            raise R10BoundedInverseError("large population requires a caller-supplied streaming base identity")
        base_identity = realization_sha256(base_pairs)
    else:
        base_identity = base_realization_sha256
    identity = R10IdentityV1(source_sha256, population, base_identity)
    packet = R10PacketV1(
        mode=R10RelayMode.JOINT,
        identity=identity,
        pair_indices=tuple(range(config.pair_count)),
        height=config.height,
        width=config.width,
        pitch_q20=pitch_q20,
        base_features=tuple(feature_records),
        xip2_payload=xip2_payload,
        textures=tuple(textures),
        shooting_knots=tuple(knots),
        dash1_payload=dash1_payload,
        pullback_polygons=polygons,
        stratified_flows=tuple(flows),
    )
    packet_bytes = serialize_r10_packet(packet)
    if serialize_r10_packet(parse_r10_packet(packet_bytes)) != packet_bytes:
        raise R10BoundedInverseError("R10 strict parse/re-emission changed packet bytes")
    spans = packet_physical_spans(packet_bytes)
    if tuple(span.section for span in spans) != G32_SECTION_ORDER:
        raise R10BoundedInverseError("bounded-family result lacks one or more physical R10 sections")
    publish(
        "110_joint_refit",
        {
            "interaction_passes": 1,
            "block_order": [
                "GEOMETRY+XIP2",
                "BASE_FEATURE",
                "TEXTURE+DASH1",
                "SHOOTING_KNOT",
                "PULLBACK_POLYGON+STRATIFIED_FLOW",
            ],
            "packet_sha256": _sha256(packet_bytes),
            "packet_bytes": len(packet_bytes),
            "packet_hex": packet_bytes.hex(),
            "trace": joint_refit_trace,
        },
    )
    wrapper, data_offset = _deterministic_store_zip(packet_bytes)
    claims = _manifest_claims(packet_bytes, spans)
    manifest = R10CountedStateManifestV1(
        schema=G32_MANIFEST_SCHEMA,
        packet_sha256=_sha256(packet_bytes),
        packet_bytes=len(packet_bytes),
        wrapper_zip_sha256=_sha256(wrapper),
        wrapper_zip_bytes=len(wrapper),
        wrapper_member_name=G32_WRAPPER_MEMBER,
        wrapper_member_sha256=_sha256(packet_bytes),
        member_data_archive_offset=data_offset,
        physical_sections=spans,
        claims=claims,
        generic_decoder_mechanisms=(
            "strict R10 parser and canonical section dispatch",
            "fixed integer luma edge feature arithmetic",
            "fixed SE3 homography and bilinear sampler",
            "fixed DASH chronology decoder and texture rasterizer",
            "fixed polygon rasterizer and stratified affine warp",
            "fixed uint8 round clip",
            "inactive identity repair ABI",
        ),
        encoder_only_evidence=(
            "source RGB frames",
            "selected base RGB frames",
            "finite search lattices and source-domain objectives",
            "rank-one normal equations and singular values",
            "residual support construction and interaction traces",
        ),
        forbidden_payloads_present=(),
        complete_packet_coverage=True,
    )
    audit_counted_state_manifest(manifest, packet_bytes)
    adapter = _build_g23_adapter(packet_bytes, wrapper, data_offset)
    g27_adapter = build_r10_selected_solution_adapter(packet_bytes)
    continuation = R10ContinuationEquivalenceIdentityV1(
        schema="taskspace_r10_continuation_equivalence_identity.v1",
        source_sha256=identity.source_sha256,
        pair_population_sha256=identity.pair_population_sha256,
        base_realization_sha256=identity.base_realization_sha256,
        packet_sha256=_sha256(packet_bytes),
        packet_bytes=len(packet_bytes),
        receiver_operation=_r10.R10_RECEIVER_OPERATION,
        receiver_source_closure_sha256=str(g27_adapter.public_inflate_receiver_abi["closure_sha256"]),
        output_shape=(config.pair_count, 2, config.height, config.width, 3),
        output_dtype="uint8",
        canonical_pair_order=packet.pair_indices,
    )
    if continuation.identity_sha256 != adapter.continuation_equivalence_sha256:
        raise R10BoundedInverseError("G23 adapter continuation-equivalence identity drifted")
    publish(
        "120_packet_adapter",
        {
            "packet_sha256": _sha256(packet_bytes),
            "wrapper_zip_sha256": _sha256(wrapper),
            "wrapper_zip_bytes": len(wrapper),
            "g23_group_id": adapter.physical_group.group_id,
            "g27_adapter_sha256": adapter.g27_adapter_sha256,
            "continuation_equivalence_sha256": adapter.continuation_equivalence_sha256,
        },
    )

    output: np.ndarray | None = None
    receipt: Mapping[str, Any] | None = None
    stats_accumulator = [0, 0, 0, 0]
    local_output = np.empty_like(base_pairs) if execute_bounded_decode else None
    for pair_index in range(config.pair_count):
        realized = _realize_pair_components(
            base_pairs[pair_index],
            xi=xi[pair_index],
            pitch=pitch,
            polygon=polygons[pair_index],
            flow=flows[pair_index],
            base_record=feature_records[pair_index],
            texture=textures[pair_index],
            dashes=decoded_dashes[pair_index],
            knot_delta=_r10._shooting_delta(tuple(knots), pair_index),
        )
        pair_stats = _changed_stats(base_pairs[pair_index], realized)
        stats_accumulator[0] += pair_stats[0]
        stats_accumulator[1] += pair_stats[1]
        stats_accumulator[2] += pair_stats[2]
        stats_accumulator[3] = max(stats_accumulator[3], pair_stats[3])
        if local_output is not None:
            local_output[pair_index] = realized
    stats = tuple(stats_accumulator)
    if execute_bounded_decode:
        if config.pair_count > 24:
            raise R10BoundedInverseError("bounded decode receipt is limited to <=24 pairs")
        output, receipt = build_r10_decode_receipt(
            packet_bytes,
            base_pairs,
            expected_source_sha256=source_sha256,
            expected_pair_population_sha256=population,
            include_mutation_canaries=True,
        )
        receipt = _scope_receiver_receipt_to_bounded_family(receipt)
        if local_output is None or not np.array_equal(local_output, output):
            raise R10BoundedInverseError("chunk-local joint realization differs from frozen G27 receiver")
        if _changed_stats(base_pairs, output) != stats:
            raise R10BoundedInverseError("streaming realized-change telemetry differs from bounded receiver")
        repaired, repair_receipt = apply_generic_bounded_repair(output, packet_bytes, 0)
        if not np.array_equal(repaired, output) or repair_receipt["changed_values"] != 0:
            raise R10BoundedInverseError("inactive repair ABI changed receiver output")

    trace_by_section: dict[str, Mapping[str, Any]] = {
        "PAIR_INDEX": {"coordinates": list(range(config.pair_count))},
        "GEOMETRY": {"pitch_lattice": inverse_trace["pitch_lattice"]},
        "BASE_FEATURE": {"pairs": feature_rows},
        "TEXTURE": {"pairs": texture_rows},
        "SHOOTING_KNOT": {"pairs": knot_rows},
        "XIP2": {
            "pair_solves": inverse_trace["pair_solves"],
            "quantized_objective": inverse_trace["quantized_objective"],
            "quantized_objective_full_resolution": inverse_trace["quantized_objective_full_resolution"],
            "full_resolution_controls": inverse_trace["full_resolution_controls"],
            "rejected_pairs_full_resolution": inverse_trace["rejected_pairs_full_resolution"],
        },
        "DASH1": {"support": list(support_rows), "report": dash_report_dict},
        "PULLBACK_POLYGON": {"records": len(polygons)},
        "STRATIFIED_FLOW": {"pairs": flow_rows},
    }
    solver_by_section = {
        "PAIR_INDEX": "canonical ordered population compiler",
        "GEOMETRY": "finite shared Q20 pitch lattice",
        "BASE_FEATURE": "least-squares rank-one fixed-point inverse",
        "TEXTURE": "finite frequency-phase plus analytic amplitude inverse",
        "SHOOTING_KNOT": "projected endpoint correction",
        "XIP2": "coarse-to-fine six-coordinate finite inverse",
        "DASH1": "source-residual connected-support chronology coder",
        "PULLBACK_POLYGON": "counted residual-support pullback box",
        "STRATIFIED_FLOW": "finite affine translation inverse",
    }
    parents = {
        "PAIR_INDEX": (),
        "GEOMETRY": ("PAIR_INDEX",),
        "BASE_FEATURE": ("GEOMETRY", "XIP2"),
        "TEXTURE": ("BASE_FEATURE", "DASH1"),
        "SHOOTING_KNOT": ("BASE_FEATURE", "TEXTURE"),
        "XIP2": ("GEOMETRY",),
        "DASH1": ("XIP2", "BASE_FEATURE"),
        "PULLBACK_POLYGON": ("BASE_FEATURE",),
        "STRATIFIED_FLOW": ("GEOMETRY", "XIP2", "PULLBACK_POLYGON"),
    }
    section_fits = tuple(
        R10SectionFitV1(
            section=span.section,
            solver_family=solver_by_section[span.section],
            packet_offset=span.byte_offset,
            byte_length=span.byte_length,
            section_sha256=span.range_sha256,
            objective_before=None,
            objective_after=None,
            realized_changed_values=None,
            realized_changed_pixels=None,
            realized_l1_rgb_delta=None,
            realized_max_abs_rgb_delta=None,
            interaction_parents=parents[span.section],
            solver_iterations=int(
                len(inverse_trace["pitch_lattice"]) * config.pair_count
                if span.section == "GEOMETRY"
                else inverse_trace["iterations"]
                if span.section == "XIP2"
                else sum(int(row.get("iterations", 0)) for row in trace_by_section[span.section].get("pairs", []))
            ),
            wall_seconds=None,
            peak_working_bytes=None,
            sufficient_statistics=trace_by_section[span.section],
        )
        for span in spans
    )
    correction_placement = tuple(
        R10CorrectionPlacementComparisonV1(
            section=span.section,
            stored_packet_bytes=span.byte_length,
            stored_wrapper_zip_bytes=len(wrapper),
            exact_generic_regenerator_available=False,
            bounded_generic_repair_available=False,
            delete_arm_realized_changed_values=None,
            repair_arm_realized_changed_values=None,
            verdict="STORED_COUNTED_STATE_REQUIRED",
        )
        for span in spans
        if span.section not in {"PAIR_INDEX"}
    )
    inventory = R10ResidualInventoryV1(
        completed_analytic_stages=(*G32_SECTION_ORDER, "JOINT_PACKET_REPLAY"),
        bounded_source_objective_interaction_passes=1,
        bounded_action_family_exhausted=False,
        unenumerated_action_families=(
            "globally_coupled_multi_section_search",
            "continuous_non_lattice_geometry_and_flow",
            "nonlinear_scorer_native_actions",
            "learned_residual_and_terminal_joint_descent",
            "alternative_receiver_program_factorizations",
        ),
        full_n600_receiver_measured_exhaustion=False,
        learned_residual_admitted=False,
        terminal_joint_descent_admitted=False,
        blocker=G32_TERMINAL_DESCENT_BLOCKER,
    )
    return R10BoundedInverseResultV1(
        schema=G32_SCHEMA,
        packet=packet,
        packet_bytes=packet_bytes,
        wrapper_zip_bytes=wrapper,
        fitted_xi=xi,
        section_fits=section_fits,
        counted_state_manifest=manifest,
        physical_group_adapter=adapter,
        continuation_equivalence=continuation,
        repair_contract=R10GenericRepairContractV1(),
        correction_placement=correction_placement,
        residual_inventory=inventory,
        bounded_output=output,
        bounded_decode_receipt=receipt,
        scratch_paths=tuple(scratch_paths),
    )


def result_receipt_dict(result: R10BoundedInverseResultV1) -> dict[str, Any]:
    """Return a canonical JSON-safe receipt without retaining large byte arrays."""

    if type(result) is not R10BoundedInverseResultV1:
        raise R10BoundedInverseError("result receipt requires exact typed result")
    group = result.physical_group_adapter.physical_group
    value = {
        "schema": result.schema,
        "packet": {
            "bytes": len(result.packet_bytes),
            "sha256": _sha256(result.packet_bytes),
            "physical_sections": [asdict(span) for span in packet_physical_spans(result.packet_bytes)],
        },
        "wrapper_zip": {
            "bytes": len(result.wrapper_zip_bytes),
            "sha256": _sha256(result.wrapper_zip_bytes),
            "member": G32_WRAPPER_MEMBER,
            "member_data_archive_offset": result.counted_state_manifest.member_data_archive_offset,
        },
        "section_fits": [asdict(row) for row in result.section_fits],
        "counted_state": {
            **asdict(result.counted_state_manifest),
            "physical_sections": [asdict(span) for span in result.counted_state_manifest.physical_sections],
        },
        "g23_physical_group": {
            "schema": result.physical_group_adapter.schema,
            "group_id": group.group_id,
            "archive_sha256": group.archive_sha256,
            "archive_bytes": len(group.exact_archive_bytes),
            "member_sha256": group.member_sha256,
            "member_bytes": len(group.exact_member_bytes),
            "range_sha256": group.range_sha256,
            "range_bytes": group.byte_length,
            "receiver_operation": group.receiver_operation,
            "logical_owner_ids": list(group.logical_owner_ids),
            "section_spans": [asdict(span) for span in result.physical_group_adapter.section_spans],
            "constraint_operand_spans": list(result.physical_group_adapter.constraint_operand_spans),
            "g27_adapter_sha256": result.physical_group_adapter.g27_adapter_sha256,
            "continuation_equivalence_sha256": result.physical_group_adapter.continuation_equivalence_sha256,
            "blocker": result.physical_group_adapter.blocker,
        },
        "continuation_equivalence": {
            **asdict(result.continuation_equivalence),
            "identity_sha256": result.continuation_equivalence.identity_sha256,
            "bounded_output_realization_sha256": (
                None if result.bounded_output is None else realization_sha256(result.bounded_output)
            ),
            "later_public_endpoint_requirement": (
                "same exact base realization plus packet plus receiver closure must reproduce the endpoint uint8 bytes"
            ),
        },
        "repair_contract": asdict(result.repair_contract),
        "correction_placement": [asdict(row) for row in result.correction_placement],
        "residual_inventory": asdict(result.residual_inventory),
        "bounded_decode_receipt": result.bounded_decode_receipt,
        "scratch_paths": list(result.scratch_paths),
        "pointer_delta": result.pointer_delta,
        "authority": {
            "research_only": True,
            "d_seg": None,
            "d_pose": None,
            "complete_candidate_zip_bytes": None,
            "contest_score": None,
            "blocker": G32_AUTHORITY_BLOCKER,
        },
    }
    _require_finite_numbers(value)
    return value


__all__ = [
    "G32_AUTHORITY_BLOCKER",
    "G32_PUBLIC_VIDEO_BLOCKER",
    "G32_SCHEMA",
    "G32_SECTION_ORDER",
    "R10ArchiveSectionSpanV1",
    "R10BoundedInverseConfigV1",
    "R10BoundedInverseError",
    "R10BoundedInverseResultV1",
    "R10ContinuationEquivalenceIdentityV1",
    "R10CorrectionPlacementComparisonV1",
    "R10CountedStateManifestV1",
    "R10G23PhysicalGroupAdapterV1",
    "R10GenericRepairContractV1",
    "R10ResidualInventoryV1",
    "R10SectionFitV1",
    "R10VideoSpecificClaimV1",
    "apply_generic_bounded_repair",
    "audit_counted_state_manifest",
    "compile_r10_bounded_inverse",
    "result_receipt_dict",
    "streaming_realization_sha256",
    "strict_json_loads",
]
