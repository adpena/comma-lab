# SPDX-License-Identifier: MIT
"""Scorer-free solve-minus-predict differential-operator instrumentation.

This module deliberately stops at the exact resize and rank-four head surfaces.
It never imports a scorer.  In particular, the camera-space costate produced here
is a factorized surrogate, not a frozen-SegNet input gradient: the inner encoder
Jacobian is absent and is recorded as a fail-closed blocker on every costate row.
"""

from __future__ import annotations

import json
import lzma
import os
import platform
import re
import shutil
import struct
import sys
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from tac.analysis.hprc_synthesis_adjoint import bilinear_resize_adjoint
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS
from tac.lie import _se3_numpy as se3_numpy
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.uint8_lattice_feasibility import realize_factor2_uint8_scorer_plane
from tac.witness_control.factorized_adjoint import CLASS_NAMES, exact_class_operator

CLASS_ORDER = tuple(CLASS_NAMES)
POINTER = "0.1910828242 [contest-CPU]"
AXIS = "[macOS-CPU frozen-scorer advisory]"
INNER_JACOBIAN_BLOCKER = "INNER_ENCODER_JACOBIAN_NOT_MEASURED_PRIMARY_PATH"
REACHABILITY_SEMANTICS = "canonical_tensor_primitive_basis_bounded_reachability_lower_bound"
FLIP_DISTANCE_SEMANTICS = "abs_margin_over_largest_target_rival_head_norm_cheapest_hyperplane_lower_bound"
RATE_STOP_THRESHOLD = 25.0 / 37_545_489.0
ZLIB_LEVEL = 1
LZMA_PRESET = 0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T", bound="StrictFrozenModel")


class SolveDiffMiningError(ValueError):
    """Fail-closed contract or custody error."""


def _validate_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not np.isfinite(value):
            raise SolveDiffMiningError(f"{path} contains a nonfinite value")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
        raise SolveDiffMiningError(f"{path} contains a nonfinite array value")


def _validate_sha(value: str, name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise SolveDiffMiningError(f"{name} must be a lowercase 64-hex SHA-256")


def _validate_path(value: str, name: str) -> None:
    if not value or "\x00" in value:
        raise SolveDiffMiningError(f"{name} must be a nonempty safe path")


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    @model_validator(mode="after")
    def _finite(self: _T) -> _T:
        _validate_finite_tree(self.model_dump(mode="python", by_alias=True))
        return self


class SolveDiffMiningConfigV1(StrictFrozenModel):
    schema_: Literal["SolveDiffMiningConfigV1"] = Field(
        default="SolveDiffMiningConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    input_mode: Literal["production", "synthetic_fixture"] = "production"
    seed: Literal[1234] = 1234
    pair_start: StrictInt = Field(default=0, ge=0, le=599)
    pair_count: StrictInt = Field(default=600, ge=1, le=600)
    chunk_size: StrictInt = Field(default=12, ge=1, le=12)
    solved_planes_receipt_path: StrictStr
    solved_planes_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    start_receipt_path: StrictStr
    start_receipt_sha256: StrictStr
    gt_cache_path: StrictStr
    gt_cache_sha256: StrictStr
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    margin_thresholds: tuple[float, ...] = (0.25, 1.0, 4.0)
    tolerance_retained_energy: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    temporal_window: StrictInt = Field(default=4, ge=2, le=64)
    ridge: float = Field(default=1e-6, ge=0.0, le=1e3)
    support_threshold: float = Field(default=0.5, ge=0.0, le=255.0)
    candidate_operators: tuple[StrictStr, ...] = (
        "xi_transport",
        "rank4_head_chart",
        "compact_parabolic_shearlet",
        "irreducible_residual",
    )
    spatial_chart: Literal["compact_parabolic_shearlet"] = "compact_parabolic_shearlet"
    coder_policy: Literal["min_bytes_tie_zlib"] = "min_bytes_tie_zlib"
    storage_roots: tuple[StrictStr, ...] = (
        "/Volumes/VertigoDataTier/pact",
        "/Volumes/APDataStore/pact",
    )
    required_free_bytes: StrictInt = Field(default=67_108_864, ge=1)
    hard_pair_panels: StrictInt = Field(default=3, ge=3, le=5)
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    archive_emitted: Literal[False] = False
    pointer_moved: Literal[False] = False
    pointer: Literal[POINTER] = POINTER
    evidence_axis: Literal[AXIS] = AXIS

    @field_validator(
        "class_order",
        "margin_thresholds",
        "tolerance_retained_energy",
        "candidate_operators",
        "storage_roots",
        mode="before",
    )
    @classmethod
    def _json_array_to_tuple(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise SolveDiffMiningError("sequence config fields must be JSON arrays")
        return tuple(value)

    @model_validator(mode="after")
    def _contract(self) -> SolveDiffMiningConfigV1:
        for name in (
            "solved_planes_receipt_sha256",
            "predictor_archive_sha256",
            "start_receipt_sha256",
            "gt_cache_sha256",
        ):
            _validate_sha(getattr(self, name), name)
        for name in (
            "solved_planes_receipt_path",
            "predictor_archive_path",
            "start_receipt_path",
            "gt_cache_path",
        ):
            _validate_path(getattr(self, name), name)
        if self.class_order != CLASS_ORDER:
            raise SolveDiffMiningError(f"class_order must be canonical {CLASS_ORDER!r}")
        if self.pair_start + self.pair_count > 600:
            raise SolveDiffMiningError("pair window escapes [0,600)")
        if self.input_mode == "production" and (self.pair_start, self.pair_count) != (0, 600):
            raise SolveDiffMiningError("production config must bind the exact full n600 window")
        margins = tuple(float(x) for x in self.margin_thresholds)
        if not margins or margins != tuple(sorted(set(margins))) or margins[0] <= 0.0:
            raise SolveDiffMiningError("margin_thresholds must be positive, sorted, and unique")
        rungs = tuple(float(x) for x in self.tolerance_retained_energy)
        if not rungs or rungs != tuple(sorted(set(rungs), reverse=True)):
            raise SolveDiffMiningError("tolerance rungs must be unique and nonincreasing")
        if rungs[0] != 1.0 or rungs[-1] != 0.0 or any(x < 0.0 or x > 1.0 for x in rungs):
            raise SolveDiffMiningError("tolerance rungs must span retained energy 1.0 to 0.0")
        required = {
            "xi_transport",
            "rank4_head_chart",
            "compact_parabolic_shearlet",
            "irreducible_residual",
        }
        if not required.issubset(self.candidate_operators):
            raise SolveDiffMiningError("candidate operator inventory is incomplete")
        if any("fourier" in name.lower() or "genuine_curvelet" in name.lower() for name in self.candidate_operators):
            raise SolveDiffMiningError("Fourier and genuine-curvelet candidate claims are forbidden")
        if not self.storage_roots or any(not Path(root).is_absolute() for root in self.storage_roots):
            raise SolveDiffMiningError("storage_roots must be nonempty absolute paths")
        return self

    def typed_config_hash(self) -> str:
        return sha256(canonical_json_bytes(self)).hexdigest()


class SolveDiffPairRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_pair.v1"] = Field(
        default="solve_diff_pair.v1", alias="schema", serialization_alias="schema"
    )
    pair_id: StrictInt = Field(ge=0, le=599)
    frame_index: StrictInt = Field(ge=0, le=1)
    total_energy: float = Field(ge=0.0)
    range_energy: float = Field(ge=0.0)
    ker_energy: float = Field(ge=0.0)
    range_fraction: float = Field(ge=0.0, le=1.0)
    ker_fraction: float = Field(ge=0.0, le=1.0)
    reconstruction_max_abs: float = Field(ge=0.0)
    orthogonality_abs: float = Field(ge=0.0)
    zero_energy_policy: Literal["range_fraction=ker_fraction=0"]
    reachability_semantics: Literal[REACHABILITY_SEMANTICS]
    reachable_basis_lower_bound: StrictInt = Field(ge=0)
    full_basis_directions: StrictInt = Field(ge=0)
    coder_bytes: dict[StrictStr, StrictInt]
    selected_coder: Literal["zlib", "lzma"]
    score_claim: Literal[False] = False


class SolveDiffStratumRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_stratum.v1"] = Field(
        default="solve_diff_stratum.v1", alias="schema", serialization_alias="schema"
    )
    pair_id: StrictInt = Field(ge=0, le=599)
    class_name: Literal["Road", "Lane", "Undrivable", "Movable", "MyCar"]
    topology_stratum: Literal["persistent", "birth", "death", "absent"]
    margin_stratum: StrictStr
    pixel_count: StrictInt = Field(ge=0)
    scorer_delta_energy: float = Field(ge=0.0)
    head_coordinate_energy: float = Field(ge=0.0)
    cheapest_rival_class: Literal["Road", "Lane", "Undrivable", "Movable", "MyCar"]
    head_normal_norm: float = Field(gt=0.0)
    flip_distance_mean: float = Field(ge=0.0)
    flip_distance_histogram: dict[StrictStr, StrictInt]
    flip_distance_semantics: Literal[FLIP_DISTANCE_SEMANTICS]
    information_fraction: float = Field(ge=0.0, le=1.0)
    coder_bytes: dict[StrictStr, StrictInt]
    selected_coder: Literal["zlib", "lzma"]
    start_attribution_status: Literal["MEASURED_RECEIPT_STRATUM", "NOT_IDENTIFIABLE_FROM_RECEIPT"]
    endpoint_label: Literal["EXISTING_C1_SOLVED_PAIR_SCORER_PLANES"]


class SolveDiffWindowRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_window.v1"] = Field(
        default="solve_diff_window.v1", alias="schema", serialization_alias="schema"
    )
    window_start: StrictInt = Field(ge=0)
    window_stop: StrictInt = Field(gt=0)
    stratum_name: Literal["all_endpoint", "Road", "Lane", "Undrivable", "Movable", "MyCar"]
    xi_mean: tuple[float, float, float, float, float, float]
    heldout_explained_squared_energy: float
    persistence_explained_squared_energy: float
    birth_frame_innovation_energy: float = Field(ge=0.0)
    movable_post_birth_predictable_energy: float = Field(ge=0.0)
    per_frame_residual_energy: float = Field(ge=0.0)
    information_fraction: float = Field(ge=0.0, le=1.0)
    coder_bytes: dict[StrictStr, StrictInt]
    selected_coder: Literal["zlib", "lzma"]
    convention: Literal["translation_first_(rho,omega)"] = "translation_first_(rho,omega)"


class SolveDiffCostateRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_costate.v1"] = Field(
        default="solve_diff_costate.v1", alias="schema", serialization_alias="schema"
    )
    pair_id: StrictInt = Field(ge=0, le=599)
    class_pair: StrictStr
    coefficient_family: Literal["compact_parabolic_shearlet", "rank4_head_chart"]
    coefficient_index: StrictInt = Field(ge=0)
    value: float
    head_linearization: Literal["EXACT_RANK4_5CLASS_QUOTIENT"]
    resize_adjoint: Literal["EXACT_BILINEAR_RESIZE_TRANSPOSE"]
    inner_encoder_jacobian: Literal["ABSENT"]
    blocker_status: Literal[INNER_JACOBIAN_BLOCKER]
    source_coordinates: Literal["CACHED_TARGET_LABEL_MARGIN_AND_ENDPOINT_RESIDUAL"]
    exact_frozen_segnet_input_gradient: Literal[False] = False


class SolveDiffToleranceRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_tolerance.v1"] = Field(
        default="solve_diff_tolerance.v1", alias="schema", serialization_alias="schema"
    )
    pair_id: StrictInt = Field(ge=0, le=599)
    frame_index: StrictInt = Field(ge=0, le=1)
    label: Literal["DERIVED_TOLERANCE_LADDER"]
    retained_energy_fraction: float = Field(ge=0.0, le=1.0)
    derived_energy: float = Field(ge=0.0)
    coder_bytes: dict[StrictStr, StrictInt]
    selected_coder: Literal["zlib", "lzma"]
    evaluator_measurement: Literal[False] = False


class SolveDiffStartReceiptRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_start_receipt.v1"] = Field(
        default="solve_diff_start_receipt.v1", alias="schema", serialization_alias="schema"
    )
    ladder_index: StrictInt = Field(ge=0)
    effective_added_budget_bytes: StrictInt = Field(ge=0)
    archive_bytes: StrictInt = Field(ge=0)
    d_seg: float = Field(ge=0.0)
    d_pose: float = Field(ge=0.0)
    stratum_family: StrictStr
    stratum_name: StrictStr
    stratum_d_seg: float = Field(ge=0.0)
    attribution_status: Literal[
        "MEASURED_RECEIPT_GLOBAL",
        "MEASURED_RECEIPT_AGGREGATE_STRATUM",
        "NOT_IDENTIFIABLE_FROM_RECEIPT",
    ]
    evidence_axis: Literal[AXIS] = AXIS
    score_claim: Literal[False] = False


class SolveDiffTemporalFeatureRowV1(StrictFrozenModel):
    schema_: Literal["solve_diff_temporal_feature.v1"] = Field(
        default="solve_diff_temporal_feature.v1",
        alias="schema",
        serialization_alias="schema",
    )
    pair_id: StrictInt = Field(ge=0, le=599)
    pose_twist: tuple[float, float, float, float, float, float]
    compact_endpoint_features: tuple[float, ...] = Field(min_length=1, max_length=256)
    stratum_endpoint_features: dict[StrictStr, tuple[float, ...]]
    movable_support_features: tuple[float, ...] = Field(min_length=1, max_length=256)
    residual_panel_features: tuple[float, ...] = Field(min_length=256, max_length=256)
    endpoint_energy: float = Field(ge=0.0)


class SolveDiffSummaryV1(StrictFrozenModel):
    schema_: Literal["solve_diff_summary.v1"] = Field(
        default="solve_diff_summary.v1", alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    pair_rows: StrictInt = Field(ge=0)
    stratum_rows: StrictInt = Field(ge=0)
    window_rows: StrictInt = Field(ge=0)
    costate_rows: StrictInt = Field(ge=0)
    tolerance_rows: StrictInt = Field(ge=0)
    start_receipt_rows: StrictInt = Field(ge=0)
    completed_stages: StrictInt = Field(ge=0)
    processed_pair_ids: tuple[StrictInt, ...]
    chart_paths: tuple[StrictStr, ...]
    kkt_status: Literal["BLOCKED_NO_RECEIVER_DELTA_DSEG"]
    blocker_status: Literal[INNER_JACOBIAN_BLOCKER]
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    archive_emitted: Literal[False] = False
    pointer_moved: Literal[False] = False
    pointer: Literal[POINTER] = POINTER
    evidence_axis: Literal[AXIS] = AXIS


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON, rejecting NaN and Infinity."""
    payload = _jsonable(value)
    _validate_finite_tree(payload)
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise SolveDiffMiningError(f"value is not canonical-JSON serializable: {exc}") from exc


def canonical_jsonl_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def sha256_file(path: str | Path, *, block_bytes: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_sha256_checked_bytes(path: str | Path, expected_sha256: str) -> bytes:
    _validate_sha(expected_sha256, "expected_sha256")
    target = Path(path)
    if not target.is_file():
        raise SolveDiffMiningError(f"custody path is not a file: {target}")
    data = target.read_bytes()
    actual = sha256(data).hexdigest()
    if actual != expected_sha256:
        raise SolveDiffMiningError(f"SHA-256 mismatch for {target}: {actual} != {expected_sha256}")
    return data


def load_checked_uint8_chunk(
    path: str | Path, expected_sha256: str, *, pair_count: int, height: int, width: int
) -> np.ndarray:
    data = load_sha256_checked_bytes(path, expected_sha256)
    expected = pair_count * height * width * 3
    if len(data) != expected:
        raise SolveDiffMiningError(f"chunk byte count {len(data)} != expected {expected}")
    return np.frombuffer(data, dtype=np.uint8).reshape(pair_count, height, width, 3)


def mmap_stored_npy_member(npz_path: str | Path, member: str) -> np.memmap:
    """Memory-map an uncompressed NPY member directly inside an NPZ file."""
    path = Path(npz_path)
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member if member.endswith(".npy") else f"{member}.npy")
        except KeyError as exc:
            raise SolveDiffMiningError(f"NPZ member is absent: {member}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise SolveDiffMiningError(f"NPZ member {info.filename} is compressed and cannot be mmap-read")
        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise SolveDiffMiningError("malformed NPZ local member header")
            name_len, extra_len = struct.unpack_from("<HH", local, 26)
            npy_start = info.header_offset + 30 + name_len + extra_len
            handle.seek(npy_start)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise SolveDiffMiningError(f"unsupported NPY version {version}")
            data_offset = handle.tell()
    order = "F" if fortran else "C"
    return np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=shape, order=order)


def realize_solve_camera(target_plane: np.ndarray, kernel: FullResizeKernel) -> np.ndarray:
    target = np.asarray(target_plane)
    if target.dtype != np.uint8 or target.shape != (kernel.scorer_h, kernel.scorer_w, 3):
        raise SolveDiffMiningError("target plane must be scorer-shaped uint8 HWC RGB")
    return realize_factor2_uint8_scorer_plane(kernel.operator, target)


def range_kernel_energy_split(delta: np.ndarray, kernel: FullResizeKernel) -> dict[str, Any]:
    raw = np.asarray(delta)
    if raw.dtype != np.float64 or raw.shape != (kernel.camera_h, kernel.camera_w, 3):
        raise SolveDiffMiningError("delta must be camera-shaped float64 RGB")
    if not np.all(np.isfinite(raw)):
        raise SolveDiffMiningError("delta must be finite")
    range_part = kernel.project_range(raw, dtype=np.float64)
    ker_part = raw - range_part
    total = float(np.vdot(raw, raw).real)
    range_energy = float(np.vdot(range_part, range_part).real)
    ker_energy = float(np.vdot(ker_part, ker_part).real)
    if total == 0.0:
        range_fraction = ker_fraction = 0.0
    else:
        range_fraction, ker_fraction = range_energy / total, ker_energy / total
    return {
        "range": range_part,
        "ker": ker_part,
        "total_energy": total,
        "range_energy": range_energy,
        "ker_energy": ker_energy,
        "range_fraction": range_fraction,
        "ker_fraction": ker_fraction,
        "reconstruction_max_abs": float(np.max(np.abs(raw - range_part - ker_part))),
        "orthogonality_abs": float(abs(np.vdot(range_part, ker_part))),
        "zero_energy_policy": "range_fraction=ker_fraction=0",
    }


def uint8_reachability_accounting(frame: np.ndarray, kernel: FullResizeKernel) -> dict[str, Any]:
    raw = np.asarray(frame)
    if raw.dtype != np.uint8 or raw.shape != (kernel.camera_h, kernel.camera_w, 3):
        raise SolveDiffMiningError("reachability requires camera-shaped uint8 RGB")
    result = kernel.uint8_reachability(raw).to_dict()
    if result["semantics"] != REACHABILITY_SEMANTICS:
        raise SolveDiffMiningError("upstream reachability semantics changed")
    return result


def canonical_class_masks(labels: np.ndarray, class_order: Sequence[str] = CLASS_ORDER) -> dict[str, np.ndarray]:
    raw = np.asarray(labels)
    if raw.dtype.kind not in "iu" or raw.ndim != 2:
        raise SolveDiffMiningError("labels must be a 2D integer array")
    if tuple(class_order) != CLASS_ORDER or np.any(raw < 0) or np.any(raw >= len(CLASS_ORDER)):
        raise SolveDiffMiningError("labels/class order violate the canonical five-class contract")
    return {name: raw == index for index, name in enumerate(CLASS_ORDER)}


def canonical_topology_masks(previous: np.ndarray, current: np.ndarray, class_id: int) -> dict[str, np.ndarray]:
    prev = np.asarray(previous)
    cur = np.asarray(current)
    if prev.shape != cur.shape or prev.ndim != 2 or prev.dtype.kind not in "iu" or cur.dtype.kind not in "iu":
        raise SolveDiffMiningError("topology labels must be same-shaped 2D integer arrays")
    if not 0 <= int(class_id) < len(CLASS_ORDER):
        raise SolveDiffMiningError("class_id is outside the canonical order")
    p, c = prev == class_id, cur == class_id
    return {"persistent": p & c, "birth": ~p & c, "death": p & ~c, "absent": ~p & ~c}


def canonical_margin_masks(margins: np.ndarray, thresholds: Sequence[float]) -> dict[str, np.ndarray]:
    raw = np.asarray(margins, dtype=np.float64)
    cuts = tuple(float(x) for x in thresholds)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise SolveDiffMiningError("margins must be a finite 2D field")
    if not cuts or cuts != tuple(sorted(set(cuts))) or cuts[0] <= 0.0:
        raise SolveDiffMiningError("margin thresholds must be positive, sorted, and unique")
    absolute = np.abs(raw)
    out: dict[str, np.ndarray] = {}
    lower = 0.0
    for upper in cuts:
        out[f"[{lower:g},{upper:g})"] = (absolute >= lower) & (absolute < upper)
        lower = upper
    out[f"[{lower:g},inf)"] = absolute >= lower
    return out


_FLIP_DISTANCE_EDGES = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def cheapest_target_hyperplane(class_id: int) -> tuple[str, float]:
    if not 0 <= int(class_id) < len(CLASS_ORDER):
        raise SolveDiffMiningError("class_id escapes the canonical five-class head")
    target = CLASS_ORDER[int(class_id)]
    candidates: list[tuple[float, str]] = []
    for rival in CLASS_ORDER:
        if rival == target:
            continue
        left, right = sorted((CLASS_ORDER.index(target), CLASS_ORDER.index(rival)))
        key = f"{CLASS_ORDER[left]}-{CLASS_ORDER[right]}"
        candidates.append((float(HEAD_PAIR_NORMS[key]), rival))
    norm, rival = max(candidates, key=lambda item: (item[0], item[1]))
    return rival, norm


def flip_distance_histogram(values: np.ndarray) -> dict[str, int]:
    distance = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(distance)) or np.any(distance < 0.0):
        raise SolveDiffMiningError("flip distances must be finite and nonnegative")
    out: dict[str, int] = {}
    lower = 0.0
    for upper in _FLIP_DISTANCE_EDGES:
        out[f"[{lower:g},{upper:g})"] = int(np.count_nonzero((distance >= lower) & (distance < upper)))
        lower = upper
    out[f"[{lower:g},inf)"] = int(np.count_nonzero(distance >= lower))
    return out


def rank4_head_accounting(class_field: np.ndarray, class_order: Sequence[str] = CLASS_ORDER) -> dict[str, Any]:
    field = np.asarray(class_field, dtype=np.float64)
    if field.ndim < 1 or field.shape[-1] != 5 or tuple(class_order) != CLASS_ORDER:
        raise SolveDiffMiningError("class_field must end in the canonical five-class order")
    if not np.all(np.isfinite(field)):
        raise SolveDiffMiningError("class_field must be finite")
    operator = exact_class_operator().astype(np.float64, copy=False)
    rank = int(np.linalg.matrix_rank(operator, tol=1e-10))
    gauge = float(np.max(np.abs(operator @ np.ones(5, dtype=np.float64))))
    if rank != 4 or gauge > 1e-10:
        raise SolveDiffMiningError("canonical head operator lost rank-four gauge-null structure")
    coordinates = np.einsum("...c,cd->...d", field, operator)
    return {
        "operator": operator,
        "coordinates": coordinates,
        "rank": rank,
        "gauge_null_linf": gauge,
        "head_linearization": "EXACT_RANK4_5CLASS_QUOTIENT",
        "inner_encoder_jacobian": "ABSENT",
        "blocker_status": INNER_JACOBIAN_BLOCKER,
    }


def exact_resize_adjoint_pullback(class_coordinate_field: np.ndarray, camera_h: int, camera_w: int) -> np.ndarray:
    field = np.asarray(class_coordinate_field, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 5 or not np.all(np.isfinite(field)):
        raise SolveDiffMiningError("class-coordinate field must be finite scorer HWC5")
    return bilinear_resize_adjoint(field[None], int(camera_h), int(camera_w))[0]


def coded_byte_counts(payload: bytes | np.ndarray) -> dict[str, Any]:
    if isinstance(payload, np.ndarray):
        raw = np.ascontiguousarray(payload).tobytes()
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise SolveDiffMiningError("coder payload must be bytes or ndarray")
    # These are actual deterministic coder outputs, not entropy estimates.  The
    # fast, named presets keep the full-P stratum ledger operationally bounded;
    # downstream promotion work may reprice selected operators at stronger
    # presets without changing the information decomposition.
    sizes = {
        "zlib": len(zlib.compress(raw, level=ZLIB_LEVEL)),
        "lzma": len(lzma.compress(raw, preset=LZMA_PRESET)),
    }
    selected = min(sizes, key=lambda name: (sizes[name], name != "zlib", name))
    return {"raw": len(raw), **sizes, "selected": selected, "selected_bytes": sizes[selected]}


def compact_parabolic_shearlet_coefficients(field: np.ndarray, *, block: int = 8) -> np.ndarray:
    """A compact parabolic directional chart; no Fourier/frame claim is made."""
    raw = np.asarray(field, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)) or block < 2:
        raise SolveDiffMiningError("compact chart needs a finite 2D field and block >= 2")
    rows: list[list[float]] = []
    for top in range(0, raw.shape[0], block):
        for left in range(0, raw.shape[1], block):
            patch = raw[top : top + block, left : left + block]
            gy, gx = np.gradient(patch) if min(patch.shape) > 1 else (np.zeros_like(patch), np.zeros_like(patch))
            rows.append(
                [
                    float(np.mean(patch)),
                    float(np.mean(gx)),
                    float(np.mean(gy)),
                    float(np.mean(gx + gy)),
                    float(np.mean(gx - gy)),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def xi_features_from_twists(pose_twists: np.ndarray) -> np.ndarray:
    twists = np.asarray(pose_twists, dtype=np.float64)
    if twists.ndim != 2 or twists.shape[1] != 6 or len(twists) < 2 or not np.all(np.isfinite(twists)):
        raise SolveDiffMiningError("pose twists must be finite (N,6), N>=2")
    transforms = se3_numpy.exp_se3(twists)
    relative = se3_numpy.compose(se3_numpy.inverse(transforms[:-1]), transforms[1:])
    features = se3_numpy.log_se3(relative)
    if features.shape != (len(twists) - 1, 6) or not np.all(np.isfinite(features)):
        raise SolveDiffMiningError("SE(3) feature production failed")
    return features


def _explained_energy(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.vdot(y, y).real)
    if denominator == 0.0:
        return 1.0 if float(np.vdot(p, p).real) == 0.0 else 0.0
    return 1.0 - float(np.vdot(y - p, y - p).real) / denominator


def leave_one_window_out_transport(
    xi_features: np.ndarray,
    targets: np.ndarray,
    *,
    window: int,
    ridge: float,
    initial_target: np.ndarray | None = None,
) -> tuple[dict[str, Any], ...]:
    x = np.asarray(xi_features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or y.shape[0] != x.shape[0] or x.shape[0] < 2:
        raise SolveDiffMiningError("transport features/targets have incompatible leading dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or window < 2 or ridge < 0.0:
        raise SolveDiffMiningError("transport inputs must be finite with valid window/ridge")
    flat = y.reshape(len(y), -1)
    if initial_target is None:
        initial = np.zeros_like(flat[:1])
    else:
        initial_raw = np.asarray(initial_target, dtype=np.float64)
        if initial_raw.shape != y.shape[1:] or not np.all(np.isfinite(initial_raw)):
            raise SolveDiffMiningError("initial transport target must match one target row")
        initial = initial_raw.reshape(1, -1)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(x), window):
        stop = min(start + window, len(x))
        held = np.arange(start, stop)
        train = np.setdiff1d(np.arange(len(x)), held)
        if len(train) == 0:
            prediction = np.zeros_like(flat[held])
        else:
            train_x = np.ascontiguousarray(x[train])
            train_y = np.ascontiguousarray(flat[train])
            gram = np.einsum("ni,nj->ij", train_x, train_x, optimize=False) + ridge * np.eye(6)
            rhs = np.einsum("ni,nj->ij", train_x, train_y, optimize=False)
            if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(rhs)):
                raise SolveDiffMiningError("transport normal equations became nonfinite")
            beta = np.linalg.solve(gram, rhs)
            if not np.all(np.isfinite(beta)):
                raise SolveDiffMiningError("transport solve became nonfinite")
            prediction = x[held] @ beta
        persistence = np.vstack((initial, flat[:-1]))[held]
        rows.append(
            {
                "window_start": int(start),
                "window_stop": int(stop),
                "xi_mean": tuple(float(v) for v in np.mean(x[held], axis=0)),
                "heldout_explained_squared_energy": _explained_energy(flat[held], prediction),
                "persistence_explained_squared_energy": _explained_energy(flat[held], persistence),
                "prediction": prediction.reshape((len(held), *y.shape[1:])),
                "residual": (flat[held] - prediction).reshape((len(held), *y.shape[1:])),
            }
        )
    return tuple(rows)


def partition_movable_innovations(
    masks: np.ndarray,
    predictions: np.ndarray,
    *,
    support_threshold: float = 0.5,
    previous_support: np.ndarray | None = None,
) -> dict[str, float]:
    actual = np.asarray(masks, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim < 2 or not np.all(np.isfinite(predicted)):
        raise SolveDiffMiningError("Movable masks/predictions must be finite and same-shaped")
    active, matched = actual > support_threshold, predicted > support_threshold
    if previous_support is None:
        previous_first = np.zeros_like(active[:1])
    else:
        prior = np.asarray(previous_support, dtype=np.float64)
        if prior.shape != actual.shape[1:] or not np.all(np.isfinite(prior)):
            raise SolveDiffMiningError("previous Movable support must match one temporal feature row")
        previous_first = (prior > support_threshold)[None]
    previous = np.vstack((previous_first, active[:-1]))
    birth = active & ~previous
    post_birth = active & previous
    birth_innovation = birth & ~matched
    post_predictable = post_birth & matched
    residual = active & ~matched & ~birth
    return {
        "birth_frame_innovation_energy": float(np.count_nonzero(birth_innovation)),
        "movable_post_birth_predictable_energy": float(np.count_nonzero(post_predictable)),
        "per_frame_residual_energy": float(np.count_nonzero(residual)),
    }


def derived_tolerance_ladder(delta: np.ndarray, retained_energy: Sequence[float]) -> tuple[dict[str, Any], ...]:
    raw = np.asarray(delta, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise SolveDiffMiningError("tolerance delta must be finite")
    rungs = tuple(float(x) for x in retained_energy)
    if not rungs or rungs != tuple(sorted(set(rungs), reverse=True)) or any(x < 0.0 or x > 1.0 for x in rungs):
        raise SolveDiffMiningError("retained-energy rungs must be unique, nonincreasing, and in [0,1]")
    total = float(np.vdot(raw, raw).real)
    return tuple(
        {
            "label": "DERIVED_TOLERANCE_LADDER",
            "retained_energy_fraction": rung,
            "derived_energy": total * rung,
            "field": raw * np.sqrt(rung),
            "evaluator_measurement": False,
        }
        for rung in rungs
    )


def rank_candidate_operators(components: Mapping[str, bytes | np.ndarray]) -> tuple[dict[str, Any], ...]:
    rows = []
    for name in ("xi_transport", "rank4_head_chart", "compact_parabolic_shearlet", "irreducible_residual"):
        if name not in components:
            raise SolveDiffMiningError(f"missing operator component {name}")
        coded = coded_byte_counts(components[name])
        rows.append(
            {
                "operator": name,
                "measured_coded_bytes": coded["selected_bytes"],
                "selected_coder": coded["selected"],
                "registered_rate_stop_law": RATE_STOP_THRESHOLD,
                "reachable_seg_debt": None,
                "kkt_admission": "BLOCKED_NO_RECEIVER_DELTA_DSEG",
            }
        )
    return tuple(rows)


def write_atomic_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def write_canonical_jsonl(path: str | Path, rows: Iterable[Any]) -> str:
    payload = b"".join(canonical_jsonl_line(row) for row in rows)
    write_atomic_bytes(path, payload)
    return sha256(payload).hexdigest()


def write_once_stage_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> Literal["WRITTEN", "RESUME_SKIP"]:
    payload = canonical_json_bytes(checkpoint) + b"\n"
    target = Path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise SolveDiffMiningError(f"refusing to overwrite unequal stage checkpoint: {target}")
        return "RESUME_SKIP"
    write_atomic_bytes(target, payload)
    return "WRITTEN"


def iter_costate_rows(path: str | Path) -> Iterator[SolveDiffCostateRowV1]:
    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield SolveDiffCostateRowV1.model_validate_json(line)
            except Exception as exc:
                raise SolveDiffMiningError(f"invalid costate row at line {line_number}: {exc}") from exc


def storage_preflight(config: SolveDiffMiningConfigV1) -> dict[str, Any]:
    observations = []
    selected = None
    for raw in config.storage_roots:
        root = Path(raw)
        if root.exists():
            free = shutil.disk_usage(root).free
            observations.append({"path": str(root), "free_bytes": free})
            if selected is None and free >= config.required_free_bytes:
                selected = str(root)
    if selected is None and config.input_mode == "production":
        raise SolveDiffMiningError("STORAGE_PREFLIGHT_BLOCKED_NO_TIER_WITH_REQUIRED_FREE_BYTES")
    return {
        "status": "PASS" if selected is not None else "FIXTURE_NO_BULK_SCRATCH_REQUIRED",
        "selected_bulk_root": selected,
        "required_free_bytes": config.required_free_bytes,
        "observations": observations,
    }


def _relative_repo_path(path: str) -> Path:
    return Path(path)


@dataclass(frozen=True)
class ProductionInputContext:
    solved_receipt: Mapping[str, Any]
    receiver: Any
    labels: np.memmap
    margins: np.memmap
    poses: np.memmap


@dataclass(frozen=True)
class MiningInputChunk:
    pair_ids: tuple[int, ...]
    solved_planes: np.ndarray
    predictor_planes: np.ndarray
    labels: np.ndarray
    previous_labels: np.ndarray
    margins: np.ndarray
    poses: np.ndarray
    source_hashes: Mapping[str, str]


def _open_production_inputs(config: SolveDiffMiningConfigV1) -> ProductionInputContext:
    receipt = json.loads(
        load_sha256_checked_bytes(
            _relative_repo_path(config.solved_planes_receipt_path),
            config.solved_planes_receipt_sha256,
        )
    )
    if receipt.get("schema") != "direct_description_full_precision_target_planes.v1" or receipt.get("pairs") != 600:
        raise SolveDiffMiningError("solved-plane receipt schema/pair custody mismatch")
    chunks = receipt.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 50:
        raise SolveDiffMiningError("solved-plane receipt must contain the frozen 50 chunks")
    archive_bytes = load_sha256_checked_bytes(
        _relative_repo_path(config.predictor_archive_path), config.predictor_archive_sha256
    )
    if len(archive_bytes) != 102_105:
        raise SolveDiffMiningError("predictor START archive must be the frozen 102105 bytes")
    from tac.optimization.direct_description_carrier_compose import (
        receive_carrier_compose_archive,
    )

    receiver = receive_carrier_compose_archive(archive_bytes)
    if sha256_file(config.gt_cache_path) != config.gt_cache_sha256:
        raise SolveDiffMiningError("GT cache SHA-256 mismatch")
    labels = mmap_stored_npy_member(config.gt_cache_path, "lstars")
    margins = mmap_stored_npy_member(config.gt_cache_path, "margins")
    poses = mmap_stored_npy_member(config.gt_cache_path, "gt_poses")
    if labels.shape != (600, 384, 512) or labels.dtype.kind not in "iu":
        raise SolveDiffMiningError("GT label member changed shape/dtype")
    if margins.shape != labels.shape or margins.dtype.kind != "f":
        raise SolveDiffMiningError("GT margin member changed shape/dtype")
    if poses.shape != (600, 6) or poses.dtype.kind != "f":
        raise SolveDiffMiningError("GT pose member changed shape/dtype")
    return ProductionInputContext(receipt, receiver, labels, margins, poses)


def _load_production_inputs(
    context: ProductionInputContext,
    config: SolveDiffMiningConfigV1,
    pair_ids: Sequence[int],
    kernel: FullResizeKernel,
) -> MiningInputChunk:
    """Load exactly one bounded pair chunk; never retain or return n600 RGB."""
    ids = tuple(int(value) for value in pair_ids)
    if not ids or len(ids) > config.chunk_size or len(ids) > 12:
        raise SolveDiffMiningError("production input load must contain 1..chunk_size<=12 pairs")
    if ids != tuple(range(ids[0], ids[0] + len(ids))):
        raise SolveDiffMiningError("production pair chunks must be contiguous")
    receiver = context.receiver
    local_ids = tuple(pair_id - receiver.predictor.source_pair_start for pair_id in ids)
    if any(index < 0 or index >= receiver.predictor.z.n_pairs for index in local_ids):
        raise SolveDiffMiningError("requested pair falls outside predictor receiver custody")
    predicted = receiver.render_pairs(local_ids)
    expected_planes = (len(ids), 2, kernel.scorer_h, kernel.scorer_w, 3)
    if predicted.shape != expected_planes or predicted.dtype != np.uint8:
        raise SolveDiffMiningError(
            f"predictor must return receiver-closed scorer planes {expected_planes}, got "
            f"{predicted.shape}/{predicted.dtype}"
        )
    solved_by_pair: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    chunk_hashes: dict[str, str] = {}
    for source_chunk in context.solved_receipt["chunks"]:
        source_ids = tuple(int(value) for value in source_chunk.get("pair_ids", ()))
        if len(source_ids) > 12:
            raise SolveDiffMiningError("receipt chunk exceeds 12-pair bound")
        selected = set(ids).intersection(source_ids)
        if not selected:
            continue
        y0 = load_checked_uint8_chunk(
            source_chunk["y0"]["path"],
            source_chunk["y0"]["sha256"],
            pair_count=len(source_ids),
            height=kernel.scorer_h,
            width=kernel.scorer_w,
        )
        y1 = load_checked_uint8_chunk(
            source_chunk["y1"]["path"],
            source_chunk["y1"]["sha256"],
            pair_count=len(source_ids),
            height=kernel.scorer_h,
            width=kernel.scorer_w,
        )
        chunk_index = int(source_chunk["chunk_index"])
        chunk_hashes[f"solved_chunk_{chunk_index:04d}_y0"] = source_chunk["y0"]["sha256"]
        chunk_hashes[f"solved_chunk_{chunk_index:04d}_y1"] = source_chunk["y1"]["sha256"]
        for pair_id in selected:
            index = source_ids.index(pair_id)
            solved_by_pair[pair_id] = (y0[index], y1[index])
    if set(solved_by_pair) != set(ids):
        raise SolveDiffMiningError("solved-plane chunks do not cover requested pairs exactly")
    solved = np.asarray([solved_by_pair[pair_id] for pair_id in ids], dtype=np.uint8)
    previous_ids = tuple(max(0, pair_id - 1) for pair_id in ids)
    return MiningInputChunk(
        pair_ids=ids,
        solved_planes=solved,
        predictor_planes=np.ascontiguousarray(predicted),
        labels=np.asarray(context.labels[list(ids)]),
        previous_labels=np.asarray(context.labels[list(previous_ids)]),
        margins=np.asarray(context.margins[list(ids)]),
        poses=np.asarray(context.poses[list(ids)], dtype=np.float64),
        source_hashes=chunk_hashes,
    )


def _fixture_inputs(pair_ids: Sequence[int], kernel: FullResizeKernel) -> MiningInputChunk:
    ids = tuple(int(value) for value in pair_ids)
    if not ids or len(ids) > 12:
        raise SolveDiffMiningError("fixture input load must contain 1..12 pairs")
    solved = np.empty((len(ids), 2, kernel.scorer_h, kernel.scorer_w, 3), dtype=np.uint8)
    predicted = np.empty((len(ids), 2, kernel.scorer_h, kernel.scorer_w, 3), dtype=np.uint8)
    labels = np.empty((len(ids), kernel.scorer_h, kernel.scorer_w), dtype=np.int64)
    previous = np.empty_like(labels)
    margins = np.empty_like(labels, dtype=np.float32)
    poses = np.empty((len(ids), 6), dtype=np.float64)
    yy, xx = np.indices((kernel.scorer_h, kernel.scorer_w))
    for local, pair_id in enumerate(ids):
        rng = np.random.default_rng(1234 + pair_id)
        for frame_index in (0, 1):
            plane = rng.integers(24, 232, size=(kernel.scorer_h, kernel.scorer_w, 3), dtype=np.uint8)
            solved[local, frame_index] = plane
            offset = ((pair_id + frame_index) % 5) - 2
            predicted[local, frame_index] = np.clip(plane.astype(np.int16) - offset, 0, 255).astype(np.uint8)
        labels[local] = (yy + xx + pair_id) % len(CLASS_ORDER)
        previous[local] = (yy + xx + max(0, pair_id - 1)) % len(CLASS_ORDER)
        margins[local] = np.linspace(-2.0, 2.0, kernel.scorer_h * kernel.scorer_w, dtype=np.float32).reshape(
            kernel.scorer_h, kernel.scorer_w
        )
        poses[local] = (0.01 * pair_id, 0.0, 0.0, 0.0, 0.0, 0.001 * pair_id)
    return MiningInputChunk(
        ids,
        solved,
        predicted,
        labels,
        previous,
        margins,
        poses,
        {"synthetic_fixture": sha256(canonical_json_bytes(ids)).hexdigest()},
    )


def _grid_mean_features(field: np.ndarray, *, bins: int = 4) -> np.ndarray:
    raw = np.asarray(field, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise SolveDiffMiningError("compact feature field must be finite 2D")
    rows = np.array_split(np.arange(raw.shape[0]), min(bins, raw.shape[0]))
    cols = np.array_split(np.arange(raw.shape[1]), min(bins, raw.shape[1]))
    return np.asarray([float(np.mean(raw[np.ix_(r, c)])) for r in rows for c in cols])


def _grid_presence_features(field: np.ndarray, *, bins: int = 4) -> np.ndarray:
    raw = np.asarray(field)
    if raw.ndim != 2:
        raise SolveDiffMiningError("presence feature field must be 2D")
    rows = np.array_split(np.arange(raw.shape[0]), min(bins, raw.shape[0]))
    cols = np.array_split(np.arange(raw.shape[1]), min(bins, raw.shape[1]))
    return np.asarray(
        [float(np.any(raw[np.ix_(r, c)])) for r in rows for c in cols],
        dtype=np.float64,
    )


def _residual_panel_features(field: np.ndarray) -> np.ndarray:
    raw = np.asarray(field, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise SolveDiffMiningError("residual panel field must be finite 2D")
    rows = np.rint(np.linspace(0, raw.shape[0] - 1, 16)).astype(np.int64)
    cols = np.rint(np.linspace(0, raw.shape[1] - 1, 16)).astype(np.int64)
    return raw[np.ix_(rows, cols)].reshape(-1)


def compact_endpoint_features(scorer_delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(scorer_delta, dtype=np.float64)
    if delta.ndim != 3 or delta.shape[-1] != 3:
        raise SolveDiffMiningError("endpoint feature input must be HWC3")
    scalar = np.linalg.norm(delta, axis=-1)
    grid = _grid_mean_features(scalar)
    endpoint = np.concatenate(
        (
            grid,
            np.asarray(
                [
                    np.mean(scalar),
                    np.std(scalar),
                    np.quantile(scalar, 0.25),
                    np.quantile(scalar, 0.5),
                    np.quantile(scalar, 0.75),
                    np.sqrt(np.mean(scalar * scalar)),
                ],
                dtype=np.float64,
            ),
        )
    )
    return endpoint


def compact_temporal_features(scorer_delta: np.ndarray, movable_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(scorer_delta, dtype=np.float64)
    movable = np.asarray(movable_mask)
    if delta.ndim != 3 or delta.shape[-1] != 3 or movable.shape != delta.shape[:2]:
        raise SolveDiffMiningError("temporal feature inputs have incompatible shapes")
    endpoint = compact_endpoint_features(delta)
    # Preserve small island births: mean occupancy followed by a 0.5 support
    # threshold erases objects occupying less than half a coarse cell.  The
    # temporal target is therefore explicit cell presence (0/1).
    return endpoint, _grid_presence_features(movable)


def xi_autocorrelation_rows(xi_features: np.ndarray, *, max_lag: int = 16) -> tuple[dict[str, Any], ...]:
    xi = np.asarray(xi_features, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 6 or not np.all(np.isfinite(xi)) or max_lag < 1:
        raise SolveDiffMiningError("xi autocorrelation requires finite (N,6) features and max_lag>=1")
    rows: list[dict[str, Any]] = []
    components = ("rho_x", "rho_y", "rho_z", "omega_x", "omega_y", "omega_z")
    for lag in range(1, min(max_lag, len(xi) - 1) + 1):
        for component, name in enumerate(components):
            left = xi[:-lag, component]
            right = xi[lag:, component]
            left_centered = left - np.mean(left)
            right_centered = right - np.mean(right)
            denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
            correlation = float(np.dot(left_centered, right_centered) / denominator) if denominator else 0.0
            rows.append({"lag": lag, "component": name, "autocorrelation": correlation})
    return tuple(rows)


def _costate_rows(
    pair_id: int,
    scorer_delta: np.ndarray,
    labels: np.ndarray,
    margins: np.ndarray,
    kernel: FullResizeKernel,
) -> list[SolveDiffCostateRowV1]:
    """Build honest factorized class/resize costates from cached endpoint data."""
    delta = np.asarray(scorer_delta, dtype=np.float64)
    target_labels = np.asarray(labels)
    target_margins = np.asarray(margins, dtype=np.float64)
    if target_labels.shape != delta.shape[:2] or target_margins.shape != delta.shape[:2]:
        raise SolveDiffMiningError("costate labels/margins must match scorer residual geometry")
    residual = np.mean(delta, axis=-1)
    operator = exact_class_operator().astype(np.float64, copy=False)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    quotient = eigenvectors[:, eigenvalues > 1e-10]
    if quotient.shape != (5, 4):
        raise SolveDiffMiningError("canonical class operator quotient is no longer rank four")
    rows: list[SolveDiffCostateRowV1] = []
    common = {
        "head_linearization": "EXACT_RANK4_5CLASS_QUOTIENT",
        "resize_adjoint": "EXACT_BILINEAR_RESIZE_TRANSPOSE",
        "inner_encoder_jacobian": "ABSENT",
        "blocker_status": INNER_JACOBIAN_BLOCKER,
        "source_coordinates": "CACHED_TARGET_LABEL_MARGIN_AND_ENDPOINT_RESIDUAL",
    }
    for target_id, target_name in enumerate(CLASS_ORDER):
        target_mask = target_labels == target_id
        if not np.any(target_mask):
            continue
        # Canonical categorical Fisher information in margin coordinates.  Clip
        # only to keep cosh finite; sech^2 is already numerically zero there.
        fisher_weight = 0.5 / np.cosh(np.clip(target_margins * 0.5, -20.0, 20.0)) ** 2
        amplitude = np.where(target_mask, residual * fisher_weight, 0.0)
        for rival_id, rival_name in enumerate(CLASS_ORDER):
            if rival_id == target_id:
                continue
            class_field = np.zeros((*delta.shape[:2], 5), dtype=np.float64)
            class_field[..., target_id] = amplitude
            class_field[..., rival_id] = -amplitude
            head_coordinates = rank4_head_accounting(class_field)["coordinates"]
            camera_costate = exact_resize_adjoint_pullback(head_coordinates, kernel.camera_h, kernel.camera_w)
            rank4_coefficients = np.mean(camera_costate, axis=(0, 1)) @ quotient
            family = f"{target_name}-{rival_name}"
            for index, value in enumerate(rank4_coefficients):
                rows.append(
                    SolveDiffCostateRowV1(
                        pair_id=pair_id,
                        class_pair=family,
                        coefficient_family="rank4_head_chart",
                        coefficient_index=index,
                        value=float(value),
                        **common,
                    )
                )
            chart = compact_parabolic_shearlet_coefficients(
                np.linalg.norm(camera_costate, axis=-1),
                block=max(2, min(kernel.camera_h, kernel.camera_w) // 4),
            )
            for index, value in enumerate(chart.reshape(-1)[:5]):
                rows.append(
                    SolveDiffCostateRowV1(
                        pair_id=pair_id,
                        class_pair=family,
                        coefficient_family="compact_parabolic_shearlet",
                        coefficient_index=index,
                        value=float(value),
                        **common,
                    )
                )
    return rows


_STAGE_MODELS: dict[str, type[StrictFrozenModel]] = {
    "pair_rows.jsonl": SolveDiffPairRowV1,
    "stratum_rows.jsonl": SolveDiffStratumRowV1,
    "costate_rows.jsonl": SolveDiffCostateRowV1,
    "tolerance_rows.jsonl": SolveDiffToleranceRowV1,
    "temporal_features.jsonl": SolveDiffTemporalFeatureRowV1,
}


def _write_once_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise SolveDiffMiningError(f"refusing to overwrite unequal stage member: {path}")
    else:
        write_atomic_bytes(path, payload)
    return sha256(payload).hexdigest()


def _stage_payload(rows: Sequence[StrictFrozenModel]) -> bytes:
    return b"".join(canonical_jsonl_line(row) for row in rows)


def _parse_typed_jsonl(path: Path, model: type[_T]) -> list[_T]:
    rows: list[_T] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise SolveDiffMiningError(f"invalid {model.__name__} row at {path}:{line_number}: {exc}") from exc
    return rows


def _validate_and_load_stage(
    stage_dir: Path,
    *,
    config: SolveDiffMiningConfigV1,
    pair_ids: tuple[int, ...],
    expected_stage_module_sha256: str,
) -> dict[str, list[StrictFrozenModel]]:
    checkpoint_path = stage_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise SolveDiffMiningError(f"resume stage lacks checkpoint: {stage_dir}")
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SolveDiffMiningError(f"invalid resume checkpoint {checkpoint_path}: {exc}") from exc
    if (
        checkpoint.get("schema") != "solve_diff_stage_checkpoint.v1"
        or checkpoint.get("config_sha256") != config.typed_config_hash()
        or checkpoint.get("instrument_module_sha256") != expected_stage_module_sha256
        or checkpoint.get("pair_ids") != list(pair_ids)
        or checkpoint.get("pair_start") != pair_ids[0]
        or checkpoint.get("pair_stop") != pair_ids[-1] + 1
    ):
        raise SolveDiffMiningError("resume checkpoint custody mismatch")
    members = checkpoint.get("members")
    if not isinstance(members, dict) or set(members) != set(_STAGE_MODELS):
        raise SolveDiffMiningError("resume checkpoint stage-member inventory mismatch")
    sources = checkpoint.get("source_hashes")
    expected_sources = {
        "solved_planes_receipt": config.solved_planes_receipt_sha256,
        "predictor_archive": config.predictor_archive_sha256,
        "start_receipt": config.start_receipt_sha256,
        "gt_cache": config.gt_cache_sha256,
    }
    if not isinstance(sources, dict) or any(sources.get(name) != digest for name, digest in expected_sources.items()):
        raise SolveDiffMiningError("resume checkpoint source-hash custody mismatch")
    for name, digest in sources.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise SolveDiffMiningError("resume checkpoint source hashes must be strings")
        _validate_sha(digest, f"checkpoint.source_hashes.{name}")
    loaded: dict[str, list[StrictFrozenModel]] = {}
    for name, model in _STAGE_MODELS.items():
        path = stage_dir / name
        metadata = members[name]
        if not path.is_file():
            raise SolveDiffMiningError(f"resume stage member is absent: {path}")
        payload = path.read_bytes()
        if metadata.get("sha256") != sha256(payload).hexdigest() or metadata.get("bytes") != len(payload):
            raise SolveDiffMiningError(f"resume stage member digest mismatch: {path}")
        rows = _parse_typed_jsonl(path, model)
        if metadata.get("rows") != len(rows):
            raise SolveDiffMiningError(f"resume stage member row-count mismatch: {path}")
        row_pair_ids = [row.pair_id for row in rows if hasattr(row, "pair_id")]
        if any(pair_id not in pair_ids for pair_id in row_pair_ids):
            raise SolveDiffMiningError(f"resume stage member pair coverage mismatch: {path}")
        loaded[name] = list(rows)
    if checkpoint.get("row_count") != sum(len(rows) for rows in loaded.values()):
        raise SolveDiffMiningError("resume checkpoint aggregate row-count mismatch")
    temporal_ids = tuple(
        row.pair_id for row in loaded["temporal_features.jsonl"] if isinstance(row, SolveDiffTemporalFeatureRowV1)
    )
    if temporal_ids != pair_ids:
        raise SolveDiffMiningError("resume checkpoint temporal pair coverage mismatch")
    return loaded


def _write_stage(
    stage_dir: Path,
    *,
    config: SolveDiffMiningConfigV1,
    pair_ids: tuple[int, ...],
    rows: Mapping[str, Sequence[StrictFrozenModel]],
    source_hashes: Mapping[str, str],
) -> dict[str, list[StrictFrozenModel]]:
    if set(rows) != set(_STAGE_MODELS):
        raise SolveDiffMiningError("stage writer received an incomplete member inventory")
    stage_dir.mkdir(parents=True, exist_ok=True)
    members: dict[str, dict[str, Any]] = {}
    normalized: dict[str, list[StrictFrozenModel]] = {}
    for name, model in _STAGE_MODELS.items():
        typed_rows = list(rows[name])
        if any(not isinstance(row, model) for row in typed_rows):
            raise SolveDiffMiningError(f"stage member {name} contains a wrong typed row")
        payload = _stage_payload(typed_rows)
        digest = _write_once_bytes(stage_dir / name, payload)
        members[name] = {"sha256": digest, "bytes": len(payload), "rows": len(typed_rows)}
        normalized[name] = typed_rows
    checkpoint = {
        "schema": "solve_diff_stage_checkpoint.v1",
        "config_sha256": config.typed_config_hash(),
        "instrument_module_sha256": sha256_file(Path(__file__).resolve()),
        "pair_ids": list(pair_ids),
        "pair_start": pair_ids[0],
        "pair_stop": pair_ids[-1] + 1,
        "pair_count": len(pair_ids),
        "row_count": sum(member["rows"] for member in members.values()),
        "members": members,
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    write_once_stage_checkpoint(stage_dir / "checkpoint.json", checkpoint)
    return normalized


def _load_start_receipt_rows(
    config: SolveDiffMiningConfigV1,
) -> tuple[list[SolveDiffStartReceiptRowV1], Mapping[str, Any] | None]:
    if config.input_mode == "synthetic_fixture":
        return [], None
    raw = load_sha256_checked_bytes(config.start_receipt_path, config.start_receipt_sha256)
    receipt = json.loads(raw)
    if receipt.get("schema") != "direct_description_v12_obligation_drain_receipt.v1":
        raise SolveDiffMiningError("START receipt schema changed")
    expected_authority = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
    }
    if any(receipt.get(name) != value for name, value in expected_authority.items()):
        raise SolveDiffMiningError("START receipt false-authority custody changed")
    ladder = receipt.get("ladder")
    if not isinstance(ladder, list) or not ladder:
        raise SolveDiffMiningError("START receipt measured budget ladder is absent")
    first_archive = ladder[0].get("archive", {})
    if (
        first_archive.get("bytes") != 102_105
        or first_archive.get("sha256") != config.predictor_archive_sha256
        or first_archive.get("receiver_closed") is not True
    ):
        raise SolveDiffMiningError("START receipt predictor archive custody mismatch")
    rows: list[SolveDiffStartReceiptRowV1] = []
    for ladder_index, rung in enumerate(ladder):
        bridge = rung.get("bridge", {})
        segmentation = bridge.get("segmentation", {})
        pose = bridge.get("pose", {})
        shared = {
            "ladder_index": ladder_index,
            "effective_added_budget_bytes": int(rung["effective_added_budget_bytes"]),
            "archive_bytes": int(rung["archive"]["bytes"]),
            "d_seg": float(segmentation["d_seg"]),
            "d_pose": float(pose["d_pose"]),
        }
        rows.append(
            SolveDiffStartReceiptRowV1(
                **shared,
                stratum_family="global",
                stratum_name="global",
                stratum_d_seg=float(segmentation["d_seg"]),
                attribution_status="MEASURED_RECEIPT_GLOBAL",
            )
        )
        strata = segmentation.get("strata")
        if not isinstance(strata, dict):
            raise SolveDiffMiningError("START receipt measured stratum ladder is malformed")
        for family, family_rows in sorted(strata.items()):
            if not isinstance(family_rows, dict):
                raise SolveDiffMiningError("START receipt stratum family is malformed")
            for name, values in sorted(family_rows.items()):
                rows.append(
                    SolveDiffStartReceiptRowV1(
                        **shared,
                        stratum_family=str(family),
                        stratum_name=str(name),
                        stratum_d_seg=float(values["d_seg"]),
                        attribution_status="MEASURED_RECEIPT_AGGREGATE_STRATUM",
                    )
                )
    return rows, receipt


def _process_chunk(
    chunk: MiningInputChunk,
    config: SolveDiffMiningConfigV1,
    kernel: FullResizeKernel,
) -> dict[str, list[StrictFrozenModel]]:
    pair_rows: list[SolveDiffPairRowV1] = []
    stratum_rows: list[SolveDiffStratumRowV1] = []
    costate_rows: list[SolveDiffCostateRowV1] = []
    tolerance_rows: list[SolveDiffToleranceRowV1] = []
    temporal_rows: list[SolveDiffTemporalFeatureRowV1] = []
    for local, pair_id in enumerate(chunk.pair_ids):
        endpoint_feature: np.ndarray | None = None
        movable_feature: np.ndarray | None = None
        stratum_features: dict[str, np.ndarray] | None = None
        endpoint_energy = 0.0
        for frame_index in (0, 1):
            solved_camera = realize_solve_camera(chunk.solved_planes[local, frame_index], kernel)
            predictor_camera = realize_solve_camera(chunk.predictor_planes[local, frame_index], kernel)
            delta_camera = solved_camera.astype(np.float64) - predictor_camera.astype(np.float64)
            energy = range_kernel_energy_split(delta_camera, kernel)
            reachability = uint8_reachability_accounting(predictor_camera, kernel)
            coded = coded_byte_counts(np.rint(delta_camera).astype(np.int16))
            pair_rows.append(
                SolveDiffPairRowV1(
                    pair_id=pair_id,
                    frame_index=frame_index,
                    total_energy=energy["total_energy"],
                    range_energy=energy["range_energy"],
                    ker_energy=energy["ker_energy"],
                    range_fraction=energy["range_fraction"],
                    ker_fraction=energy["ker_fraction"],
                    reconstruction_max_abs=energy["reconstruction_max_abs"],
                    orthogonality_abs=energy["orthogonality_abs"],
                    zero_energy_policy=energy["zero_energy_policy"],
                    reachability_semantics=REACHABILITY_SEMANTICS,
                    reachable_basis_lower_bound=reachability["feasible_basis_directions_lower_bound"],
                    full_basis_directions=reachability["full_basis_directions"],
                    coder_bytes={"zlib": coded["zlib"], "lzma": coded["lzma"]},
                    selected_coder=coded["selected"],
                )
            )
            scorer_delta = kernel.operator.apply(delta_camera)
            for rung in derived_tolerance_ladder(scorer_delta, config.tolerance_retained_energy):
                rung_coded = coded_byte_counts(np.asarray(rung["field"], dtype=np.float32))
                tolerance_rows.append(
                    SolveDiffToleranceRowV1(
                        pair_id=pair_id,
                        frame_index=frame_index,
                        label="DERIVED_TOLERANCE_LADDER",
                        retained_energy_fraction=rung["retained_energy_fraction"],
                        derived_energy=rung["derived_energy"],
                        coder_bytes={
                            "zlib": rung_coded["zlib"],
                            "lzma": rung_coded["lzma"],
                        },
                        selected_coder=rung_coded["selected"],
                    )
                )
            if frame_index != 1:
                continue
            labels = chunk.labels[local]
            margins = chunk.margins[local]
            costate_rows.extend(_costate_rows(pair_id, scorer_delta, labels, margins, kernel))
            margin_masks = canonical_margin_masks(margins, config.margin_thresholds)
            scalar_energy = np.sum(scorer_delta * scorer_delta, axis=-1)
            total_scalar_energy = float(np.sum(scalar_energy))
            class_field = np.zeros((*labels.shape, 5), dtype=np.float64)
            residual_scalar = np.mean(scorer_delta, axis=-1)
            fisher_weight = 0.5 / np.cosh(np.clip(margins.astype(np.float64) * 0.5, -20.0, 20.0)) ** 2
            for class_id in range(5):
                class_field[..., class_id] = np.where(
                    labels == class_id,
                    residual_scalar * fisher_weight,
                    0.0,
                )
            head_coordinates = rank4_head_accounting(class_field)["coordinates"]
            head_energy = np.sum(head_coordinates * head_coordinates, axis=-1)
            for class_id, class_name in enumerate(CLASS_ORDER):
                topology = canonical_topology_masks(chunk.previous_labels[local], labels, class_id)
                cheapest_rival, head_normal_norm = cheapest_target_hyperplane(class_id)
                flip_distance = np.abs(margins.astype(np.float64)) / head_normal_norm
                for topology_name, topology_mask in topology.items():
                    for margin_name, margin_mask in margin_masks.items():
                        mask = topology_mask & margin_mask
                        stratum_energy = float(np.sum(scalar_energy[mask]))
                        stratum_flip_distance = flip_distance[mask]
                        stratum_coded = coded_byte_counts(np.asarray(scorer_delta[mask], dtype=np.float32))
                        stratum_rows.append(
                            SolveDiffStratumRowV1(
                                pair_id=pair_id,
                                class_name=class_name,
                                topology_stratum=topology_name,
                                margin_stratum=margin_name,
                                pixel_count=int(np.count_nonzero(mask)),
                                scorer_delta_energy=stratum_energy,
                                head_coordinate_energy=float(np.sum(head_energy[mask])),
                                cheapest_rival_class=cheapest_rival,
                                head_normal_norm=head_normal_norm,
                                flip_distance_mean=(
                                    float(np.mean(stratum_flip_distance)) if stratum_flip_distance.size else 0.0
                                ),
                                flip_distance_histogram=flip_distance_histogram(stratum_flip_distance),
                                flip_distance_semantics=FLIP_DISTANCE_SEMANTICS,
                                information_fraction=(
                                    stratum_energy / total_scalar_energy if total_scalar_energy else 0.0
                                ),
                                coder_bytes={
                                    "zlib": stratum_coded["zlib"],
                                    "lzma": stratum_coded["lzma"],
                                },
                                selected_coder=stratum_coded["selected"],
                                start_attribution_status="NOT_IDENTIFIABLE_FROM_RECEIPT",
                                endpoint_label="EXISTING_C1_SOLVED_PAIR_SCORER_PLANES",
                            )
                        )
            endpoint_feature, movable_feature = compact_temporal_features(
                scorer_delta, labels == CLASS_ORDER.index("Movable")
            )
            stratum_features = {
                class_name: compact_endpoint_features(np.where((labels == class_id)[..., None], scorer_delta, 0.0))
                for class_id, class_name in enumerate(CLASS_ORDER)
            }
            endpoint_energy = total_scalar_energy
        if endpoint_feature is None or movable_feature is None or stratum_features is None:
            raise SolveDiffMiningError("pair stage did not produce frame-1 compact features")
        temporal_rows.append(
            SolveDiffTemporalFeatureRowV1(
                pair_id=pair_id,
                pose_twist=tuple(float(value) for value in chunk.poses[local]),
                compact_endpoint_features=tuple(float(value) for value in endpoint_feature),
                stratum_endpoint_features={
                    name: tuple(float(value) for value in values) for name, values in stratum_features.items()
                },
                movable_support_features=tuple(float(value) for value in movable_feature),
                residual_panel_features=tuple(
                    float(value) for value in _residual_panel_features(np.linalg.norm(scorer_delta, axis=-1))
                ),
                endpoint_energy=endpoint_energy,
            )
        )
    return {
        "pair_rows.jsonl": list(pair_rows),
        "stratum_rows.jsonl": list(stratum_rows),
        "costate_rows.jsonl": list(costate_rows),
        "tolerance_rows.jsonl": list(tolerance_rows),
        "temporal_features.jsonl": list(temporal_rows),
    }


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)


def _rgb_png(pixels: np.ndarray) -> bytes:
    image = np.asarray(pixels)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise SolveDiffMiningError("PNG pixels must be uint8 HWC3")
    height, width = image.shape[:2]
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _bar_chart_png(values: Sequence[float], *, width: int = 480, height: int = 240) -> bytes:
    numbers = np.asarray(tuple(values), dtype=np.float64)
    if numbers.ndim != 1 or not len(numbers) or not np.all(np.isfinite(numbers)):
        numbers = np.zeros(1, dtype=np.float64)
    magnitude = np.abs(numbers)
    scale = float(np.max(magnitude)) or 1.0
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    count = len(numbers)
    bar_width = max(1, width // max(1, count * 2))
    for index, value in enumerate(magnitude):
        x0 = int((index + 0.25) * width / count)
        x1 = min(width, x0 + bar_width)
        bar_height = int((height - 16) * float(value) / scale)
        pixels[height - bar_height : height, x0:x1] = (40, 105, 180)
    return _rgb_png(pixels)


def _hard_pair_panels_png(panels: Sequence[Sequence[float]], *, scale: int = 8) -> bytes:
    if not panels:
        return _bar_chart_png([0.0])
    rendered: list[np.ndarray] = []
    for panel in panels:
        field = np.asarray(panel, dtype=np.float64)
        if field.shape != (256,) or not np.all(np.isfinite(field)):
            raise SolveDiffMiningError("hard-pair panel must contain 256 finite residual cells")
        field = field.reshape(16, 16)
        maximum = float(np.max(field)) or 1.0
        normalized = np.clip(field / maximum, 0.0, 1.0)
        rgb = np.stack(
            (
                np.rint(255.0 * normalized),
                np.rint(160.0 * np.sqrt(normalized)),
                np.rint(255.0 * (1.0 - normalized)),
            ),
            axis=-1,
        ).astype(np.uint8)
        rendered.append(np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1))
    gap = np.full((16 * scale, 4, 3), 255, dtype=np.uint8)
    mosaic = rendered[0]
    for panel in rendered[1:]:
        mosaic = np.concatenate((mosaic, gap, panel), axis=1)
    return _rgb_png(mosaic)


def _chart_html(title: str, rows: Sequence[Mapping[str, Any]]) -> bytes:
    body = canonical_json_bytes(list(rows)).decode("utf-8")
    return (
        "<!doctype html><meta charset=utf-8><title>"
        + title
        + "</title><h1>"
        + title
        + "</h1><p>research_only=true; score_claim=false</p><pre id=data>"
        + body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>\n"
    ).encode("utf-8")


def _write_charts(
    root: Path,
    *,
    stratum_rows: Sequence[SolveDiffStratumRowV1],
    window_rows: Sequence[SolveDiffWindowRowV1],
    pair_rows: Sequence[SolveDiffPairRowV1],
    costate_rows: Sequence[SolveDiffCostateRowV1],
    temporal_rows: Sequence[SolveDiffTemporalFeatureRowV1],
    xi_autocorr_rows: Sequence[Mapping[str, Any]],
    hard_pair_panels: int,
) -> tuple[str, ...]:
    chart_dir = root / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    energy_groups: dict[str, float] = {}
    flip_distance_counts: dict[str, int] = {}
    for row in stratum_rows:
        key = f"{row.class_name}->{row.cheapest_rival_class}|{row.topology_stratum}|{row.margin_stratum}"
        energy_groups[key] = energy_groups.get(key, 0.0) + row.scorer_delta_energy
        if row.topology_stratum in {"persistent", "birth"}:
            for key, value in row.flip_distance_histogram.items():
                flip_distance_counts[key] = flip_distance_counts.get(key, 0) + value
    pair_energy: dict[int, float] = {}
    for row in pair_rows:
        pair_energy[row.pair_id] = pair_energy.get(row.pair_id, 0.0) + row.total_energy
    hard_pairs = sorted(pair_energy.items(), key=lambda item: (-item[1], item[0]))[
        : min(hard_pair_panels, len(pair_energy))
    ]
    panels_by_pair = {row.pair_id: row.residual_panel_features for row in temporal_rows}
    rank4 = [row for row in costate_rows if row.coefficient_family == "rank4_head_chart"]
    shearlet = [row for row in costate_rows if row.coefficient_family == "compact_parabolic_shearlet"]
    coded_rows = (
        {
            "operator": "xi_transport",
            "measured_coded_bytes": sum(row.coder_bytes[row.selected_coder] for row in window_rows),
            "byte_semantics": "sum_of_per_stratum_window_residual_streams",
        },
        {
            "operator": "rank4_head_chart",
            "measured_coded_bytes": coded_byte_counts(np.asarray([row.value for row in rank4], dtype=np.float64))[
                "selected_bytes"
            ],
            "byte_semantics": "coded_float64_coefficient_vector",
        },
        {
            "operator": "compact_parabolic_shearlet",
            "measured_coded_bytes": coded_byte_counts(np.asarray([row.value for row in shearlet], dtype=np.float64))[
                "selected_bytes"
            ],
            "byte_semantics": "coded_float64_coefficient_vector",
        },
        {
            "operator": "irreducible_residual",
            "measured_coded_bytes": sum(row.coder_bytes[row.selected_coder] for row in pair_rows),
            "byte_semantics": "sum_of_per_frame_camera_delta_int16_streams",
        },
    )
    coded_rows = tuple(
        {
            **row,
            "registered_rate_stop_law": RATE_STOP_THRESHOLD,
            "reachable_seg_debt": None,
            "kkt_admission": "BLOCKED_NO_RECEIVER_DELTA_DSEG",
        }
        for row in coded_rows
    )
    charts: dict[str, tuple[list[float], list[Mapping[str, Any]]]] = {
        "energy_by_hyperplane": (
            list(energy_groups.values()),
            [{"stratum": key, "energy": value} for key, value in sorted(energy_groups.items())],
        ),
        "xi_autocorr": (
            [float(row["autocorrelation"]) for row in xi_autocorr_rows],
            list(xi_autocorr_rows),
        ),
        "coded_byte_waterfall": (
            [float(row["measured_coded_bytes"]) for row in coded_rows],
            list(coded_rows),
        ),
        "flip_distance_histogram": (
            [float(value) for _, value in flip_distance_counts.items()],
            [
                {
                    "flip_distance_bin": key,
                    "pixel_count": value,
                    "semantics": FLIP_DISTANCE_SEMANTICS,
                }
                for key, value in flip_distance_counts.items()
            ],
        ),
        "hard_pair_panels": (
            [value for _, value in hard_pairs],
            [
                {
                    "panel_index": index,
                    "pair_id": pair_id,
                    "measured_endpoint_residual_energy": value,
                    "raw_frame_dump": False,
                }
                for index, (pair_id, value) in enumerate(hard_pairs)
            ],
        ),
    }
    paths: list[str] = []
    for name, (values, rows) in charts.items():
        png_path = chart_dir / f"{name}.png"
        html_path = chart_dir / f"{name}.html"
        png = (
            _hard_pair_panels_png([panels_by_pair[pair_id] for pair_id, _ in hard_pairs])
            if name == "hard_pair_panels"
            else _bar_chart_png(values)
        )
        write_atomic_bytes(png_path, png)
        write_atomic_bytes(html_path, _chart_html(name.replace("_", " "), rows))
        paths.extend((str(png_path), str(html_path)))
    return tuple(paths)


def run_mining_pass(
    config: SolveDiffMiningConfigV1,
    output_root: str | Path,
    *,
    pair_limit: int | None = None,
    resume: bool = False,
    resume_stage_module_sha256: str | None = None,
    argv: Sequence[str] | None = None,
    config_path: str | Path | None = None,
) -> SolveDiffSummaryV1:
    """Run chunk-bounded scorer-free mining with real write-once resume."""
    current_module_sha256 = sha256_file(Path(__file__).resolve())
    if resume_stage_module_sha256 is not None:
        _validate_sha(resume_stage_module_sha256, "resume_stage_module_sha256")
        if not resume:
            raise SolveDiffMiningError("resume_stage_module_sha256 requires resume=True")
    stage_producer_module_sha256 = resume_stage_module_sha256 or current_module_sha256
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    preflight = storage_preflight(config)
    limit = config.pair_count if pair_limit is None else int(pair_limit)
    if limit < 1 or limit > config.pair_count:
        raise SolveDiffMiningError("pair_limit must be in [1, config.pair_count]")
    pair_ids = tuple(range(config.pair_start, config.pair_start + limit))
    kernel = (
        FullResizeKernel.build(camera_h=8, camera_w=10, scorer_h=4, scorer_w=5)
        if config.input_mode == "synthetic_fixture"
        else FullResizeKernel.build()
    )
    start_rows, _start_receipt = _load_start_receipt_rows(config)
    context: ProductionInputContext | None = None
    combined: dict[str, list[StrictFrozenModel]] = {name: [] for name in _STAGE_MODELS}
    completed_stages = 0
    for offset in range(0, len(pair_ids), config.chunk_size):
        stage_ids = pair_ids[offset : offset + config.chunk_size]
        stage_dir = root / "stages" / f"pairs_{stage_ids[0]:04d}_{stage_ids[-1] + 1:04d}"
        checkpoint_path = stage_dir / "checkpoint.json"
        if checkpoint_path.exists():
            if not resume:
                raise SolveDiffMiningError(f"completed stage already exists; rerun with --resume: {stage_dir}")
            stage_rows = _validate_and_load_stage(
                stage_dir,
                config=config,
                pair_ids=stage_ids,
                expected_stage_module_sha256=stage_producer_module_sha256,
            )
        else:
            if resume_stage_module_sha256 is not None:
                raise SolveDiffMiningError(
                    "resume stage module override cannot create a missing stage; "
                    f"mixed-producer run refused at {stage_dir}"
                )
            if config.input_mode == "synthetic_fixture":
                chunk = _fixture_inputs(stage_ids, kernel)
            else:
                if context is None:
                    context = _open_production_inputs(config)
                chunk = _load_production_inputs(context, config, stage_ids, kernel)
            processed = _process_chunk(chunk, config, kernel)
            source_hashes = {
                "solved_planes_receipt": config.solved_planes_receipt_sha256,
                "predictor_archive": config.predictor_archive_sha256,
                "start_receipt": config.start_receipt_sha256,
                "gt_cache": config.gt_cache_sha256,
                **chunk.source_hashes,
            }
            stage_rows = _write_stage(
                stage_dir,
                config=config,
                pair_ids=stage_ids,
                rows=processed,
                source_hashes=source_hashes,
            )
            del chunk, processed
        for name in _STAGE_MODELS:
            combined[name].extend(stage_rows[name])
        completed_stages += 1

    pair_rows = [row for row in combined["pair_rows.jsonl"] if isinstance(row, SolveDiffPairRowV1)]
    stratum_rows = [row for row in combined["stratum_rows.jsonl"] if isinstance(row, SolveDiffStratumRowV1)]
    costate_rows = [row for row in combined["costate_rows.jsonl"] if isinstance(row, SolveDiffCostateRowV1)]
    tolerance_rows = [row for row in combined["tolerance_rows.jsonl"] if isinstance(row, SolveDiffToleranceRowV1)]
    temporal_rows = [
        row for row in combined["temporal_features.jsonl"] if isinstance(row, SolveDiffTemporalFeatureRowV1)
    ]
    if tuple(row.pair_id for row in temporal_rows) != pair_ids:
        raise SolveDiffMiningError("combined stage pair coverage is not exact and ordered")

    window_rows: list[SolveDiffWindowRowV1] = []
    xi_autocorr: tuple[dict[str, Any], ...] = ()
    if len(temporal_rows) >= 3:
        twists = np.asarray([row.pose_twist for row in temporal_rows], dtype=np.float64)
        xi = xi_features_from_twists(twists)
        xi_autocorr = xi_autocorrelation_rows(xi)
        targets_all_by_stratum = {
            "all_endpoint": np.asarray([row.compact_endpoint_features for row in temporal_rows], dtype=np.float64),
            **{
                class_name: np.asarray(
                    [row.stratum_endpoint_features[class_name] for row in temporal_rows],
                    dtype=np.float64,
                )
                for class_name in CLASS_ORDER
            },
        }
        targets_by_stratum = {name: targets[1:] for name, targets in targets_all_by_stratum.items()}
        movable_all = np.asarray([row.movable_support_features for row in temporal_rows], dtype=np.float64)
        movable = movable_all[1:]
        movable_transport = leave_one_window_out_transport(
            xi,
            movable,
            window=config.temporal_window,
            ridge=config.ridge,
            initial_target=movable_all[0],
        )
        for stratum_name, targets in targets_by_stratum.items():
            endpoint_transport = leave_one_window_out_transport(
                xi,
                targets,
                window=config.temporal_window,
                ridge=config.ridge,
                initial_target=targets_all_by_stratum[stratum_name][0],
            )
            for index, endpoint_row in enumerate(endpoint_transport):
                start = endpoint_row["window_start"]
                stop = endpoint_row["window_stop"]
                partition = (
                    partition_movable_innovations(
                        movable[start:stop],
                        movable_transport[index]["prediction"],
                        support_threshold=config.support_threshold,
                        previous_support=movable_all[start],
                    )
                    if stratum_name == "Movable"
                    else {
                        "birth_frame_innovation_energy": 0.0,
                        "movable_post_birth_predictable_energy": 0.0,
                        "per_frame_residual_energy": 0.0,
                    }
                )
                residual_coded = coded_byte_counts(np.asarray(endpoint_row["residual"], dtype=np.float32))
                information_fraction = float(np.clip(endpoint_row["heldout_explained_squared_energy"], 0.0, 1.0))
                window_rows.append(
                    SolveDiffWindowRowV1(
                        window_start=start,
                        window_stop=stop,
                        stratum_name=stratum_name,
                        xi_mean=endpoint_row["xi_mean"],
                        heldout_explained_squared_energy=endpoint_row["heldout_explained_squared_energy"],
                        persistence_explained_squared_energy=endpoint_row["persistence_explained_squared_energy"],
                        information_fraction=information_fraction,
                        coder_bytes={
                            "zlib": residual_coded["zlib"],
                            "lzma": residual_coded["lzma"],
                        },
                        selected_coder=residual_coded["selected"],
                        **partition,
                    )
                )

    outputs: dict[str, tuple[Path, Sequence[Any]]] = {
        "pair": (root / "pair_rows.jsonl", pair_rows),
        "stratum": (root / "stratum_rows.jsonl", stratum_rows),
        "window": (root / "window_rows.jsonl", window_rows),
        "costate": (root / "costate_rows.jsonl", costate_rows),
        "tolerance": (root / "tolerance_rows.jsonl", tolerance_rows),
        "start_receipt": (root / "start_receipt_rows.jsonl", start_rows),
        "temporal_features": (root / "temporal_features.jsonl", temporal_rows),
    }
    output_hashes = {name: write_canonical_jsonl(path, rows) for name, (path, rows) in outputs.items()}
    chart_paths = _write_charts(
        root,
        stratum_rows=stratum_rows,
        window_rows=window_rows,
        pair_rows=pair_rows,
        costate_rows=costate_rows,
        temporal_rows=temporal_rows,
        xi_autocorr_rows=xi_autocorr,
        hard_pair_panels=config.hard_pair_panels,
    )
    summary = SolveDiffSummaryV1(
        run_id=config.run_id,
        pair_rows=len(pair_rows),
        stratum_rows=len(stratum_rows),
        window_rows=len(window_rows),
        costate_rows=len(costate_rows),
        tolerance_rows=len(tolerance_rows),
        start_receipt_rows=len(start_rows),
        completed_stages=completed_stages,
        processed_pair_ids=pair_ids,
        chart_paths=chart_paths,
        kkt_status="BLOCKED_NO_RECEIVER_DELTA_DSEG",
        blocker_status=INNER_JACOBIAN_BLOCKER,
    )
    write_atomic_bytes(root / "summary.json", canonical_json_bytes(summary) + b"\n")
    module_path = Path(__file__).resolve()
    tool_path = module_path.parents[3] / "tools" / "measure_ddm_solve_diff_operator.py"
    implementation_hashes = {"module": current_module_sha256}
    if tool_path.is_file():
        implementation_hashes["tool"] = sha256_file(tool_path)
    if config_path is not None:
        path = Path(config_path)
        implementation_hashes["config_file"] = sha256_file(path)
    receipt = {
        "schema": "solve_diff_mining_receipt.v1",
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "argv": list(argv if argv is not None else sys.argv),
        "git_sha": _git_sha(),
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "host_axis": AXIS,
        "coder_contract": {
            "semantics": "MEASURED_PAYLOAD_BYTES_NOT_ENTROPY_ESTIMATE",
            "zlib_level": ZLIB_LEVEL,
            "lzma_preset": LZMA_PRESET,
        },
        "source_hashes": {
            "solved_planes_receipt": config.solved_planes_receipt_sha256,
            "predictor_archive": config.predictor_archive_sha256,
            "start_receipt": config.start_receipt_sha256,
            "gt_cache": config.gt_cache_sha256,
        },
        "implementation_hashes": implementation_hashes,
        "stage_producer_module_sha256": stage_producer_module_sha256,
        "finalizer_module_sha256": current_module_sha256,
        "stage_module_override_used": resume_stage_module_sha256 is not None,
        "output_jsonl_sha256": output_hashes,
        "output_member_sha256": {
            **{path.name: sha256_file(path) for path, _rows in outputs.values()},
            **{Path(path).name: sha256_file(path) for path in chart_paths},
            "summary.json": sha256_file(root / "summary.json"),
        },
        "storage_preflight": preflight,
        "summary": summary.model_dump(mode="json", by_alias=True),
        "inner_encoder_jacobian": "ABSENT",
        "blocker_status": INNER_JACOBIAN_BLOCKER,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
    }
    write_atomic_bytes(root / "receipt.json", canonical_json_bytes(receipt) + b"\n")
    retained_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "cleanup_manifest.json")
    cleanup = {
        "schema": "solve_diff_cleanup_manifest.v1",
        "deleted": [],
        "retained_outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "reason": "durable operator-facing telemetry or resumable stage evidence",
            }
            for path in retained_paths
        ],
        "rebuild_config_sha256": config.typed_config_hash(),
        "rebuild_argv": list(argv if argv is not None else sys.argv),
        "certified_success_scratch_only": True,
        "scratch_created": [],
        "scratch_cleanup_status": "NO_SCRATCH_CREATED",
    }
    write_atomic_bytes(root / "cleanup_manifest.json", canonical_json_bytes(cleanup) + b"\n")
    return summary


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    dot_git = root / ".git"
    if dot_git.is_file():
        marker = dot_git.read_text().strip()
        if not marker.startswith("gitdir: "):
            return None
        git_dir = Path(marker[8:])
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
    else:
        git_dir = dot_git
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text().strip()
    if value.startswith("ref: "):
        ref_name = value[5:]
        search_roots = [git_dir]
        commondir_path = git_dir / "commondir"
        if commondir_path.is_file():
            common = Path(commondir_path.read_text().strip())
            if not common.is_absolute():
                common = (git_dir / common).resolve()
            search_roots.append(common)
        for search_root in search_roots:
            ref = search_root / ref_name
            if ref.is_file():
                return ref.read_text().strip()
            packed = search_root / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line and not line.startswith(("#", "^")):
                        digest, name = line.split(" ", 1)
                        if name == ref_name:
                            return digest
        return None
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


__all__ = [
    "AXIS",
    "CLASS_ORDER",
    "INNER_JACOBIAN_BLOCKER",
    "POINTER",
    "SolveDiffCostateRowV1",
    "SolveDiffMiningConfigV1",
    "SolveDiffMiningError",
    "SolveDiffPairRowV1",
    "SolveDiffStartReceiptRowV1",
    "SolveDiffStratumRowV1",
    "SolveDiffSummaryV1",
    "SolveDiffTemporalFeatureRowV1",
    "SolveDiffToleranceRowV1",
    "SolveDiffWindowRowV1",
    "canonical_class_masks",
    "canonical_json_bytes",
    "canonical_jsonl_line",
    "canonical_margin_masks",
    "canonical_topology_masks",
    "coded_byte_counts",
    "compact_parabolic_shearlet_coefficients",
    "compact_temporal_features",
    "derived_tolerance_ladder",
    "exact_resize_adjoint_pullback",
    "iter_costate_rows",
    "leave_one_window_out_transport",
    "load_checked_uint8_chunk",
    "load_sha256_checked_bytes",
    "mmap_stored_npy_member",
    "partition_movable_innovations",
    "range_kernel_energy_split",
    "rank4_head_accounting",
    "rank_candidate_operators",
    "realize_solve_camera",
    "run_mining_pass",
    "sha256_file",
    "storage_preflight",
    "uint8_reachability_accounting",
    "write_canonical_jsonl",
    "write_once_stage_checkpoint",
    "xi_features_from_twists",
]
