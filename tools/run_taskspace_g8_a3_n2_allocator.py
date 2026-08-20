#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the reviewed real-n2 atomic G8 plus conditional-A experiment.

The module deliberately separates a pure, resumable orchestration core from
the real ep725/scorer backend.  Importing it, printing the reviewed command, or
running focused tests does not decode the source and does not load Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
UPSTREAM = REPO / "upstream"
TOOLS = REPO / "tools"
for _search_path in (SRC, UPSTREAM, TOOLS):
    if str(_search_path) not in sys.path:
        sys.path.insert(0, str(_search_path))

from tac.score_geometry import (  # noqa: E402
    CONTEST_REFERENCE_BYTES,
    RATE_COEFFICIENT,
    contest_score,
)

SCHEMA: Final = "tac.taskspace_g8_a3_n2_allocator.v1"
MANIFEST_SCHEMA: Final = "tac.taskspace_g8_a3_n2_allocator_manifest.v1"
CHECKPOINT_SCHEMA: Final = "tac.taskspace_g8_a3_n2_allocator_checkpoint.v1"
BLOCKER_SCHEMA: Final = "tac.taskspace_g8_a3_n2_allocator_blocker.v1"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
LANE_ID: Final = "lane_g14_taskspace_g8_a3_n2_allocator_20260726"
SEED: Final = 1234
PAIR_COUNT: Final = 2
FULL_G8_PREFIXES: Final = (1, 4, 16, 64, 256, 1024, 4096)
CHEAP_G8_PREFIXES: Final = (1, 4, 16, 64)
DEFAULT_A_PREFIXES: Final = (0, 1, 4, 16, 64)
DEFAULT_PALETTE_BOUNDS: Final = (2, 4)
G10_FROZEN_IMPLEMENTATION: Final = {
    "src/tac/witness_dsl/taskspace_pass_semantic_g.py": (
        "701bbab6561e2b8bfee1a4469f0aeb000347219b4a20fc9ffd2550d143171b38"
    ),
    "src/tac/witness_dsl/taskspace_pass_conditional_a.py": (
        "282f620c3c4712c35fc6b1a75b8f48b5de2ce250c145b06b0fb5b4f551a8050f"
    ),
    "src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py": (
        "aa1cc628764de0311a23db791635ec884f24d27dc126ceff28e03134e8f51ced"
    ),
}
G12_FROZEN_IMPLEMENTATION: Final = {
    "src/tac/witness_dsl/taskspace_same_class_realization_encoder.py": (
        "842035c29647f26aa620a1c19a26943bd8a307702563a283215afffcf234285a"
    ),
}
# Bounded behavior closure for the real G14 path.  This deliberately includes
# direct scorer/resize/archive dependencies and the receiver's parse/dispatch
# modules, not only the three headline G10 files.  Provider-specific future A
# implementations remain separately bound by their append-only extension
# contract.
REAL_PATH_IMPLEMENTATION_CLOSURE: Final = (
    "tools/run_taskspace_g8_a3_n2_allocator.py",
    "tools/materialize_taskspace_pga_n2_receipt.py",
    "tools/measure_taskspace_pga_n2_macos_cpu.py",
    "upstream/modules.py",
    "upstream/frame_utils.py",
    "src/tac/__init__.py",
    "src/tac/score_geometry.py",
    "src/tac/canonical_frontier_pointer.py",
    "src/tac/contest_compliance.py",
    "src/tac/exact_eval_custody.py",
    "src/tac/boundary_math/power_diagram_witness.py",
    "src/tac/optimization/direct_description_carrier_compose.py",
    "src/tac/optimization/direct_description_minimizer.py",
    "src/tac/optimization/uint8_lattice_feasibility.py",
    "src/tac/through_r/resolution_chain.py",
    "src/tac/witness_dsl/__init__.py",
    "src/tac/witness_dsl/bounded_target_g_encoder.py",
    "src/tac/witness_dsl/coupled_preimage_program.py",
    "src/tac/witness_dsl/dynamic_frontier_target.py",
    "src/tac/witness_dsl/ep725_levelset_predictor_adapter.py",
    "src/tac/witness_dsl/factorized_v9_predictor.py",
    "src/tac/witness_dsl/generative_taskspace_correction.py",
    "src/tac/witness_dsl/predictor_bound_residual.py",
    "src/tac/witness_dsl/predictor_preserving_coupled_preimage.py",
    "src/tac/witness_dsl/predictor_preserving_taskspace_overlay.py",
    "src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py",
    "src/tac/witness_dsl/taskspace_outer_archive_codec.py",
    "src/tac/witness_dsl/taskspace_pass_conditional_a.py",
    "src/tac/witness_dsl/taskspace_pass_semantic_g.py",
    "src/tac/witness_dsl/taskspace_post_g8_conditional_a.py",
    "src/tac/witness_dsl/taskspace_predictor_state_v2.py",
    "src/tac/witness_dsl/taskspace_predictor_v2_consumer_seam.py",
    "src/tac/witness_dsl/taskspace_same_class_realization_encoder.py",
    "src/tac/witness_dsl/taskspace_same_class_realization_repair.py",
    "src/tac/witness_dsl/taskspace_whole_archive_allocator.py",
)
_SAFE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
# Closed composite: proposal_id (192) + ':' + program_id (192) + ':' +
# canonical lowercase SHA-256 (64).  This is a stage-source bound only; the
# general wire/object identifier contract above remains unchanged.
_STAGE_SOURCE_ID_MAX_LENGTH: Final = 192 + 1 + 192 + 1 + 64
_SAFE_STAGE_SOURCE_ID_RE: Final = re.compile(rf"[A-Za-z0-9][A-Za-z0-9_.:-]{{0,{_STAGE_SOURCE_ID_MAX_LENGTH - 1}}}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class TaskspaceG8A3N2AllocatorError(RuntimeError):
    """Configuration, custody, composition, measurement, or resume failed."""


class G10ProductionCompositionUnavailable(TaskspaceG8A3N2AllocatorError):
    """The exact frozen G10 PASS/G8/A production surface is unavailable."""

    blocker_code: Final = "G10_PRODUCTION_COMPOSITION_UNAVAILABLE"


class AModeV1(StrEnum):
    PASS_A_V1 = "PASS_A_V1"
    TARGET_CONSTANT_RGB_V1 = "TARGET_CONSTANT_RGB_V1"
    POST_G8_Y1_SUPPORT_COPY_V1 = "POST_G8_Y1_SUPPORT_COPY_V1"
    COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1 = "COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1"
    COUNTED_XIP2_V1 = "COUNTED_XIP2_V1"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TaskspaceG8A3N2AllocatorError("record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_id(value: object, field: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must be bounded safe ASCII")
    return value


def _require_stage_source_identity(value: object, field: str) -> str:
    if type(value) is not str or _SAFE_STAGE_SOURCE_ID_RE.fullmatch(value) is None:
        raise TaskspaceG8A3N2AllocatorError(
            f"{field} must be bounded safe ASCII (1..{_STAGE_SOURCE_ID_MAX_LENGTH} characters)"
        )
    return value


def _require_exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must be an exact nonnegative integer")
    return value


def _require_finite_distance(value: object, field: str, *, bounded_one: bool) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must be one finite nonnegative float")
    if bounded_one and value > 1.0:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must not exceed one")
    return value


def _strict_positive_csv(value: str, *, field: str, allow_zero: bool = False) -> tuple[int, ...]:
    if type(value) is not str or not value or value.strip() != value:
        raise argparse.ArgumentTypeError(f"{field} must be a canonical comma-separated integer list")
    pieces = value.split(",")
    if any(not piece or not piece.isascii() or not piece.isdecimal() for piece in pieces):
        raise argparse.ArgumentTypeError(f"{field} must contain only base-10 integer tokens")
    parsed = tuple(int(piece) for piece in pieces)
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(f"{field} values must be >= {minimum}")
    if parsed != tuple(sorted(set(parsed))):
        raise argparse.ArgumentTypeError(f"{field} values must be sorted and unique")
    return parsed


def parse_palette_bounds(value: str) -> tuple[int, ...]:
    parsed = _strict_positive_csv(value, field="palette bounds")
    if any(item < 2 for item in parsed):
        raise argparse.ArgumentTypeError("palette bounds must be >= 2")
    return parsed


def parse_a_prefixes(value: str) -> tuple[int, ...]:
    parsed = _strict_positive_csv(value, field="A prefixes", allow_zero=True)
    if not parsed or parsed[0] != 0 or any(item not in DEFAULT_A_PREFIXES for item in parsed):
        raise argparse.ArgumentTypeError("A prefixes must be a sorted subset of 0,1,4,16,64 including 0")
    return parsed


def resolve_g8_prefixes(value: str, *, debt_count: int) -> tuple[int, ...]:
    """Resolve reviewed G8 prefixes and always preserve the exact debt endpoint."""

    if type(debt_count) is not int or debt_count < 1:
        raise TaskspaceG8A3N2AllocatorError("G8 acquisition requires positive exact debt_count")
    if value == "cheap":
        requested = CHEAP_G8_PREFIXES
    elif value == "full":
        requested = FULL_G8_PREFIXES
    else:
        try:
            requested = _strict_positive_csv(value, field="G8 prefixes")
        except argparse.ArgumentTypeError as exc:
            raise TaskspaceG8A3N2AllocatorError(str(exc)) from exc
        if any(item not in FULL_G8_PREFIXES for item in requested):
            raise TaskspaceG8A3N2AllocatorError("explicit G8 prefixes must be a subset of 1,4,16,64,256,1024,4096")
    return tuple(sorted({*(item for item in requested if item <= debt_count), debt_count}))


@dataclass(frozen=True, slots=True)
class G8BranchV1:
    proposal_id: str
    program_sha256: str
    family: str
    prefix_order: str
    palette_bound_per_class: int | None
    prefix_cell_count: int
    program: object

    def __post_init__(self) -> None:
        _require_id(self.proposal_id, "G8 proposal_id")
        _require_sha256(self.program_sha256, "G8 program_sha256")
        _require_id(self.family, "G8 family")
        _require_id(self.prefix_order, "G8 prefix_order")
        if self.palette_bound_per_class is not None and (
            type(self.palette_bound_per_class) is not int or self.palette_bound_per_class < 1
        ):
            raise TaskspaceG8A3N2AllocatorError("G8 palette bound must be positive or null")
        if type(self.prefix_cell_count) is not int or self.prefix_cell_count < 1:
            raise TaskspaceG8A3N2AllocatorError("G8 prefix_cell_count must be positive")

    @property
    def series_key(self) -> tuple[str, str, int | None]:
        return self.family, self.prefix_order, self.palette_bound_per_class

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "program_sha256": self.program_sha256,
            "family": self.family,
            "prefix_order": self.prefix_order,
            "palette_bound_per_class": self.palette_bound_per_class,
            "prefix_cell_count": self.prefix_cell_count,
        }


@dataclass(frozen=True, slots=True)
class AProgramV1:
    program_id: str
    program_sha256: str
    mode: AModeV1
    row_count: int
    acquisition_y1_sha256: str
    program: object
    ranking_sha256: str

    def __post_init__(self) -> None:
        _require_id(self.program_id, "A program_id")
        _require_sha256(self.program_sha256, "A program_sha256")
        if type(self.mode) is not AModeV1:
            raise TaskspaceG8A3N2AllocatorError("A mode escaped its closed universe")
        _require_exact_nonnegative_int(self.row_count, "A row_count")
        _require_sha256(self.acquisition_y1_sha256, "A acquisition_y1_sha256")
        _require_sha256(self.ranking_sha256, "A ranking_sha256")
        zero_row_modes = {
            AModeV1.PASS_A_V1,
            AModeV1.COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1,
        }
        if (self.mode in zero_row_modes) != (self.row_count == 0):
            raise TaskspaceG8A3N2AllocatorError("only PASS-A and counted empty-body global copy may contain zero rows")

    def as_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "program_sha256": self.program_sha256,
            "mode": self.mode.value,
            "row_count": self.row_count,
            "acquisition_y1_sha256": self.acquisition_y1_sha256,
            "ranking_sha256": self.ranking_sha256,
        }


def _float32_pair_mean(values: tuple[float, ...], *, field: str, bounded_one: bool) -> float:
    if type(values) is not tuple or len(values) != PAIR_COUNT:
        raise TaskspaceG8A3N2AllocatorError(f"{field} must contain exactly the n2 pair values")
    for index, value in enumerate(values):
        _require_finite_distance(value, f"{field}[{index}]", bounded_one=bounded_one)
    return float(np.mean(np.asarray(values, dtype=np.float32), dtype=np.float32))


@dataclass(frozen=True, slots=True)
class BoundedScorerEvidenceV1:
    """Dense-free scorer evidence sufficient to audit one n2 aggregate."""

    measurement_receipt_sha256: str
    candidate_forward_receipt_sha256: str
    candidate_pose6_sha256: str
    per_pair_d_seg: tuple[float, ...]
    per_pair_d_pose: tuple[float, ...]
    sample_count: int
    frozen_scorer_sha256: str
    target_forward_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "measurement_receipt_sha256",
            "candidate_forward_receipt_sha256",
            "candidate_pose6_sha256",
            "frozen_scorer_sha256",
            "target_forward_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if type(self.sample_count) is not int or self.sample_count != PAIR_COUNT:
            raise TaskspaceG8A3N2AllocatorError("scorer evidence sample_count must be exact n2")
        _float32_pair_mean(self.per_pair_d_seg, field="per_pair_d_seg", bounded_one=True)
        _float32_pair_mean(self.per_pair_d_pose, field="per_pair_d_pose", bounded_one=False)

    @property
    def aggregate_d_seg(self) -> float:
        return _float32_pair_mean(self.per_pair_d_seg, field="per_pair_d_seg", bounded_one=True)

    @property
    def aggregate_d_pose(self) -> float:
        return _float32_pair_mean(self.per_pair_d_pose, field="per_pair_d_pose", bounded_one=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "measurement_receipt_sha256": self.measurement_receipt_sha256,
            "candidate_forward_receipt_sha256": self.candidate_forward_receipt_sha256,
            "candidate_pose6_sha256": self.candidate_pose6_sha256,
            "per_pair_d_seg": list(self.per_pair_d_seg),
            "per_pair_d_pose": list(self.per_pair_d_pose),
            "sample_count": self.sample_count,
            "frozen_scorer_sha256": self.frozen_scorer_sha256,
            "target_forward_receipt_sha256": self.target_forward_receipt_sha256,
            "aggregate_reduction": "torch_float32_mean.v1",
            "dense_frames_logits_rgb_serialized": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> BoundedScorerEvidenceV1:
        if type(value) is not dict or set(value) != {
            "measurement_receipt_sha256",
            "candidate_forward_receipt_sha256",
            "candidate_pose6_sha256",
            "per_pair_d_seg",
            "per_pair_d_pose",
            "sample_count",
            "frozen_scorer_sha256",
            "target_forward_receipt_sha256",
            "aggregate_reduction",
            "dense_frames_logits_rgb_serialized",
        }:
            raise TaskspaceG8A3N2AllocatorError("bounded scorer evidence schema changed")
        if (
            value.get("aggregate_reduction") != "torch_float32_mean.v1"
            or value.get("dense_frames_logits_rgb_serialized") is not False
        ):
            raise TaskspaceG8A3N2AllocatorError("bounded scorer evidence truth/reduction changed")
        per_seg = value.get("per_pair_d_seg")
        per_pose = value.get("per_pair_d_pose")
        if type(per_seg) is not list or type(per_pose) is not list:
            raise TaskspaceG8A3N2AllocatorError("bounded scorer per-pair values changed exact type")
        try:
            return cls(
                measurement_receipt_sha256=value["measurement_receipt_sha256"],
                candidate_forward_receipt_sha256=value["candidate_forward_receipt_sha256"],
                candidate_pose6_sha256=value["candidate_pose6_sha256"],
                per_pair_d_seg=tuple(per_seg),
                per_pair_d_pose=tuple(per_pose),
                sample_count=value["sample_count"],
                frozen_scorer_sha256=value["frozen_scorer_sha256"],
                target_forward_receipt_sha256=value["target_forward_receipt_sha256"],
            )
        except KeyError as exc:
            raise TaskspaceG8A3N2AllocatorError("bounded scorer evidence omitted a field") from exc


@dataclass(frozen=True, slots=True)
class WholeObjectMeasurementV1:
    """One exact selected whole archive plus receiver/scorer identities."""

    measurement_id: str
    baseline_bundle_sha256: str
    bundle_sha256: str
    g8_program_sha256: str | None
    a_program_sha256: str
    a_packet_sha256: str
    a_source_binding_sha256: str
    a_mode: AModeV1
    a_row_count: int
    raw_section_bytes: int
    member_bytes: int
    member_sha256: str
    stored_archive_bytes: int
    stored_archive_sha256: str
    deflated_archive_bytes: int
    deflated_archive_sha256: str
    selected_encoding: Literal["STORE", "DEFLATE"]
    selected_archive_bytes: int
    selected_archive_sha256: str
    selected_archive_payload: bytes
    decoded_output_sha256: str
    receiver_receipt_sha256: str
    camera_y1_sha256: str
    candidate_seg_labels_sha256: str
    scorer_evidence: BoundedScorerEvidenceV1
    d_seg: float
    d_pose: float

    def __post_init__(self) -> None:
        _require_id(self.measurement_id, "measurement_id")
        for field in (
            "baseline_bundle_sha256",
            "bundle_sha256",
            "a_program_sha256",
            "a_packet_sha256",
            "a_source_binding_sha256",
            "member_sha256",
            "stored_archive_sha256",
            "deflated_archive_sha256",
            "selected_archive_sha256",
            "decoded_output_sha256",
            "receiver_receipt_sha256",
            "camera_y1_sha256",
            "candidate_seg_labels_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.g8_program_sha256 is not None:
            _require_sha256(self.g8_program_sha256, "g8_program_sha256")
        if type(self.a_mode) is not AModeV1:
            raise TaskspaceG8A3N2AllocatorError("measurement A mode escaped its closed universe")
        for field in (
            "a_row_count",
            "raw_section_bytes",
            "member_bytes",
            "stored_archive_bytes",
            "deflated_archive_bytes",
            "selected_archive_bytes",
        ):
            _require_exact_nonnegative_int(getattr(self, field), field)
        if self.member_bytes < 1 or self.selected_archive_bytes < 1:
            raise TaskspaceG8A3N2AllocatorError("measurement member/archive must be nonempty")
        _require_finite_distance(self.d_seg, "d_seg", bounded_one=True)
        _require_finite_distance(self.d_pose, "d_pose", bounded_one=False)
        if type(self.scorer_evidence) is not BoundedScorerEvidenceV1:
            raise TaskspaceG8A3N2AllocatorError("measurement scorer evidence changed exact type")
        if self.scorer_evidence.aggregate_d_seg != self.d_seg or self.scorer_evidence.aggregate_d_pose != self.d_pose:
            raise TaskspaceG8A3N2AllocatorError("measurement aggregate differs from exact per-pair scorer evidence")
        if type(self.selected_archive_payload) is not bytes or not self.selected_archive_payload:
            raise TaskspaceG8A3N2AllocatorError("selected archive payload must be nonempty exact bytes")
        if (
            len(self.selected_archive_payload) != self.selected_archive_bytes
            or _sha256(self.selected_archive_payload) != self.selected_archive_sha256
        ):
            raise TaskspaceG8A3N2AllocatorError("selected archive payload differs from its exact custody")
        expected_bytes = min(self.stored_archive_bytes, self.deflated_archive_bytes)
        expected_encoding = "STORE" if self.stored_archive_bytes <= self.deflated_archive_bytes else "DEFLATE"
        expected_sha = self.stored_archive_sha256 if expected_encoding == "STORE" else self.deflated_archive_sha256
        if (
            self.selected_archive_bytes != expected_bytes
            or self.selected_encoding != expected_encoding
            or self.selected_archive_sha256 != expected_sha
        ):
            raise TaskspaceG8A3N2AllocatorError("selected archive is not the canonical STORE/DEFLATE minimum")
        zero_row_modes = {
            AModeV1.PASS_A_V1,
            AModeV1.COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1,
        }
        if (self.a_mode in zero_row_modes) != (self.a_row_count == 0):
            raise TaskspaceG8A3N2AllocatorError("measurement A row count/mode mismatch")

    @property
    def score(self) -> float:
        return contest_score(self.d_seg, self.d_pose, self.selected_archive_bytes)

    def as_dict(self, *, archive_path: str | None = None) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "baseline_bundle_sha256": self.baseline_bundle_sha256,
            "bundle_sha256": self.bundle_sha256,
            "g8_program_sha256": self.g8_program_sha256,
            "a_program_sha256": self.a_program_sha256,
            "a_packet_sha256": self.a_packet_sha256,
            "a_source_binding_sha256": self.a_source_binding_sha256,
            "a_mode": self.a_mode.value,
            "a_row_count": self.a_row_count,
            "raw_section_bytes": self.raw_section_bytes,
            "member_bytes": self.member_bytes,
            "member_sha256": self.member_sha256,
            "stored_archive_bytes": self.stored_archive_bytes,
            "stored_archive_sha256": self.stored_archive_sha256,
            "deflated_archive_bytes": self.deflated_archive_bytes,
            "deflated_archive_sha256": self.deflated_archive_sha256,
            "selected_encoding": self.selected_encoding,
            "selected_archive_bytes": self.selected_archive_bytes,
            "selected_archive_sha256": self.selected_archive_sha256,
            "selected_archive_path": archive_path,
            "decoded_output_sha256": self.decoded_output_sha256,
            "receiver_receipt_sha256": self.receiver_receipt_sha256,
            "camera_y1_sha256": self.camera_y1_sha256,
            "candidate_seg_labels_sha256": self.candidate_seg_labels_sha256,
            "scorer_evidence": self.scorer_evidence.as_dict(),
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "derived_component_total": self.score,
        }

    @classmethod
    def validate_serialized_summary(cls, value: object) -> None:
        """Validate a dense-free row even when archive bytes are held elsewhere."""

        expected = (set(cls.__dataclass_fields__) - {"selected_archive_payload"}) | {
            "selected_archive_path",
            "derived_component_total",
        }
        if type(value) is not dict or set(value) != expected:
            raise TaskspaceG8A3N2AllocatorError("measurement summary fields changed")
        _require_id(value.get("measurement_id"), "measurement_id")
        for field in (
            "baseline_bundle_sha256",
            "bundle_sha256",
            "a_program_sha256",
            "a_packet_sha256",
            "a_source_binding_sha256",
            "member_sha256",
            "stored_archive_sha256",
            "deflated_archive_sha256",
            "selected_archive_sha256",
            "decoded_output_sha256",
            "receiver_receipt_sha256",
            "camera_y1_sha256",
            "candidate_seg_labels_sha256",
        ):
            _require_sha256(value.get(field), field)
        g8_sha = value.get("g8_program_sha256")
        if g8_sha is not None:
            _require_sha256(g8_sha, "g8_program_sha256")
        try:
            a_mode = AModeV1(value.get("a_mode"))
        except (TypeError, ValueError) as exc:
            raise TaskspaceG8A3N2AllocatorError("measurement summary A mode is invalid") from exc
        for field in (
            "a_row_count",
            "raw_section_bytes",
            "member_bytes",
            "stored_archive_bytes",
            "deflated_archive_bytes",
            "selected_archive_bytes",
        ):
            _require_exact_nonnegative_int(value.get(field), field)
        if value["member_bytes"] < 1 or value["selected_archive_bytes"] < 1:
            raise TaskspaceG8A3N2AllocatorError("measurement summary member/archive must be nonempty")
        d_seg = _require_finite_distance(value.get("d_seg"), "d_seg", bounded_one=True)
        d_pose = _require_finite_distance(value.get("d_pose"), "d_pose", bounded_one=False)
        evidence = BoundedScorerEvidenceV1.from_dict(value.get("scorer_evidence"))
        if evidence.aggregate_d_seg != d_seg or evidence.aggregate_d_pose != d_pose:
            raise TaskspaceG8A3N2AllocatorError("measurement summary aggregate differs from per-pair evidence")
        expected_bytes = min(value["stored_archive_bytes"], value["deflated_archive_bytes"])
        expected_encoding = "STORE" if value["stored_archive_bytes"] <= value["deflated_archive_bytes"] else "DEFLATE"
        expected_sha = (
            value["stored_archive_sha256"] if expected_encoding == "STORE" else value["deflated_archive_sha256"]
        )
        if (
            value.get("selected_archive_bytes") != expected_bytes
            or value.get("selected_encoding") != expected_encoding
            or value.get("selected_archive_sha256") != expected_sha
        ):
            raise TaskspaceG8A3N2AllocatorError("measurement summary selection differs from STORE/DEFLATE minimum")
        zero_row_modes = {
            AModeV1.PASS_A_V1,
            AModeV1.COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1,
        }
        if (a_mode in zero_row_modes) != (value["a_row_count"] == 0):
            raise TaskspaceG8A3N2AllocatorError("measurement summary A row count/mode mismatch")
        archive_path = value.get("selected_archive_path")
        if archive_path is not None and (
            type(archive_path) is not str or Path(archive_path).is_absolute() or ".." in Path(archive_path).parts
        ):
            raise TaskspaceG8A3N2AllocatorError("measurement summary archive path escaped custody")
        derived = value.get("derived_component_total")
        if type(derived) is not float or derived != contest_score(d_seg, d_pose, value["selected_archive_bytes"]):
            raise TaskspaceG8A3N2AllocatorError("measurement summary derived score changed")

    @classmethod
    def from_dict_and_archive(cls, value: dict[str, Any], archive: bytes) -> WholeObjectMeasurementV1:
        cls.validate_serialized_summary(value)
        expected = set(cls.__dataclass_fields__) - {"selected_archive_payload"}
        serialized = set(value) - {"selected_archive_path", "derived_component_total"}
        if serialized != expected:
            raise TaskspaceG8A3N2AllocatorError("measurement checkpoint fields changed")
        arguments = dict(value)
        arguments.pop("selected_archive_path", None)
        derived = arguments.pop("derived_component_total", None)
        try:
            arguments["a_mode"] = AModeV1(arguments["a_mode"])
        except (KeyError, ValueError) as exc:
            raise TaskspaceG8A3N2AllocatorError("measurement checkpoint A mode is invalid") from exc
        arguments["scorer_evidence"] = BoundedScorerEvidenceV1.from_dict(arguments.get("scorer_evidence"))
        arguments["selected_archive_payload"] = archive
        measurement = cls(**arguments)
        if type(derived) is not float or derived != measurement.score:
            raise TaskspaceG8A3N2AllocatorError("measurement checkpoint derived score changed")
        return measurement


@dataclass(frozen=True, slots=True)
class PairwiseTransitionV1:
    before_measurement_id: str
    after_measurement_id: str
    delta_d_seg: float
    delta_d_pose: float
    delta_selected_archive_bytes: int
    seg_term_delta: float
    pose_term_delta: float
    rate_term_delta: float
    distortion_term_delta: float
    exact_score_delta: float
    exact_score_improvement: float
    finite_byte_ceiling_real: float
    greatest_strict_integer_byte_delta: int
    improves_score: bool

    @classmethod
    def between(
        cls,
        before: WholeObjectMeasurementV1,
        after: WholeObjectMeasurementV1,
    ) -> PairwiseTransitionV1:
        if before.baseline_bundle_sha256 != after.baseline_bundle_sha256:
            raise TaskspaceG8A3N2AllocatorError("pairwise transition crossed baseline bundle custody")
        delta_seg = after.d_seg - before.d_seg
        delta_pose = after.d_pose - before.d_pose
        delta_bytes = after.selected_archive_bytes - before.selected_archive_bytes
        seg_term = 100.0 * delta_seg
        pose_term = math.sqrt(10.0 * after.d_pose) - math.sqrt(10.0 * before.d_pose)
        rate_term = RATE_COEFFICIENT * delta_bytes / CONTEST_REFERENCE_BYTES
        distortion = seg_term + pose_term
        exact = distortion + rate_term
        ceiling = -(CONTEST_REFERENCE_BYTES / RATE_COEFFICIENT) * distortion
        strict_integer = math.ceil(ceiling) - 1
        if not all(math.isfinite(value) for value in (seg_term, pose_term, rate_term, distortion, exact, ceiling)):
            raise TaskspaceG8A3N2AllocatorError("pairwise transition contains nonfinite arithmetic")
        return cls(
            before_measurement_id=before.measurement_id,
            after_measurement_id=after.measurement_id,
            delta_d_seg=delta_seg,
            delta_d_pose=delta_pose,
            delta_selected_archive_bytes=delta_bytes,
            seg_term_delta=seg_term,
            pose_term_delta=pose_term,
            rate_term_delta=rate_term,
            distortion_term_delta=distortion,
            exact_score_delta=exact,
            exact_score_improvement=-exact,
            finite_byte_ceiling_real=ceiling,
            greatest_strict_integer_byte_delta=strict_integer,
            improves_score=exact < 0.0,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def interaction_score(
    *,
    g8_a: WholeObjectMeasurementV1,
    g8_pass: WholeObjectMeasurementV1,
    g0_a: WholeObjectMeasurementV1,
    g0_pass: WholeObjectMeasurementV1,
) -> float:
    baseline_ids = {row.baseline_bundle_sha256 for row in (g8_a, g8_pass, g0_a, g0_pass)}
    if len(baseline_ids) != 1:
        raise TaskspaceG8A3N2AllocatorError("interaction crossed baseline bundle custody")
    return g8_a.score - g8_pass.score - g0_a.score + g0_pass.score


def assert_a_variant_invariants(
    pass_row: WholeObjectMeasurementV1,
    variants: Sequence[WholeObjectMeasurementV1],
) -> None:
    if pass_row.a_mode is not AModeV1.PASS_A_V1:
        raise TaskspaceG8A3N2AllocatorError("A invariant anchor must be one PASS-A row")
    for row in variants:
        if row.g8_program_sha256 != pass_row.g8_program_sha256:
            raise TaskspaceG8A3N2AllocatorError("A invariant comparison crossed a G8 branch")
        equalities = {
            "baseline_bundle_sha256": (
                row.baseline_bundle_sha256,
                pass_row.baseline_bundle_sha256,
            ),
            "a_source_binding_sha256": (
                row.a_source_binding_sha256,
                pass_row.a_source_binding_sha256,
            ),
            "camera_y1_sha256": (row.camera_y1_sha256, pass_row.camera_y1_sha256),
            "candidate_seg_labels_sha256": (
                row.candidate_seg_labels_sha256,
                pass_row.candidate_seg_labels_sha256,
            ),
            "per_pair_d_seg": (
                row.scorer_evidence.per_pair_d_seg,
                pass_row.scorer_evidence.per_pair_d_seg,
            ),
            "frozen_scorer_sha256": (
                row.scorer_evidence.frozen_scorer_sha256,
                pass_row.scorer_evidence.frozen_scorer_sha256,
            ),
            "target_forward_receipt_sha256": (
                row.scorer_evidence.target_forward_receipt_sha256,
                pass_row.scorer_evidence.target_forward_receipt_sha256,
            ),
            "d_seg": (row.d_seg, pass_row.d_seg),
        }
        drift = [field for field, (observed, expected) in equalities.items() if observed != expected]
        if drift:
            raise TaskspaceG8A3N2AllocatorError(f"A variant changed post-G8 Y1/Seg invariant: {sorted(drift)}")


def dedupe_g8_branches(
    branches: Iterable[G8BranchV1],
    *,
    allowed_prefixes: tuple[int, ...],
) -> tuple[tuple[G8BranchV1, ...], dict[str, list[str]]]:
    allowed = set(allowed_prefixes)
    seen: dict[str, G8BranchV1] = {}
    aliases: dict[str, list[str]] = {}
    for branch in branches:
        if type(branch) is not G8BranchV1:
            raise TaskspaceG8A3N2AllocatorError("G8 acquisition returned a noncanonical branch")
        if branch.prefix_cell_count not in allowed:
            continue
        aliases.setdefault(branch.program_sha256, []).append(branch.proposal_id)
        seen.setdefault(branch.program_sha256, branch)
    unique = tuple(seen.values())
    if not unique:
        raise TaskspaceG8A3N2AllocatorError("reviewed G8 prefix filter removed every program")
    return unique, aliases


def nondominated_g8_indices(rows: Sequence[WholeObjectMeasurementV1]) -> tuple[int, ...]:
    if not rows:
        raise TaskspaceG8A3N2AllocatorError("G8 screening requires at least one measured row")
    nondominated: list[int] = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            weak = (
                other.d_seg <= row.d_seg
                and other.d_pose <= row.d_pose
                and other.selected_archive_bytes <= row.selected_archive_bytes
            )
            strict = (
                other.d_seg < row.d_seg
                or other.d_pose < row.d_pose
                or other.selected_archive_bytes < row.selected_archive_bytes
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            nondominated.append(index)
    return tuple(nondominated)


def retained_g8_branch_ids(
    branches: Sequence[G8BranchV1],
    rows: Sequence[WholeObjectMeasurementV1],
) -> tuple[str, ...]:
    if len(branches) != len(rows) or not branches:
        raise TaskspaceG8A3N2AllocatorError("G8 branch/measurement screening rows do not align")
    nondominated = nondominated_g8_indices(rows)
    keep = set(nondominated)
    # Conditional A can reverse the scalar PASS-A ordering across G8
    # families/orders/palettes.  Preserve every currently nondominated
    # surface, then add its local geometric prefix context.  Selecting only
    # the scalar winner here would suppress exactly the interaction this
    # coupled experiment exists to measure.
    for anchor_index in nondominated:
        anchor = branches[anchor_index]
        same_series = sorted(
            (
                (branch.prefix_cell_count, index)
                for index, branch in enumerate(branches)
                if branch.series_key == anchor.series_key
            ),
            key=lambda item: (item[0], branches[item[1]].proposal_id),
        )
        position = next(position for position, (_prefix, index) in enumerate(same_series) if index == anchor_index)
        if position > 0:
            keep.add(same_series[position - 1][1])
        if position + 1 < len(same_series):
            keep.add(same_series[position + 1][1])
    return tuple(branches[index].proposal_id for index in sorted(keep))


def require_pass_a_first(programs: Sequence[AProgramV1]) -> tuple[AProgramV1, ...]:
    ordered = tuple(programs)
    if not ordered or ordered[0].mode is not AModeV1.PASS_A_V1 or ordered[0].row_count != 0:
        raise TaskspaceG8A3N2AllocatorError("conditional-A schedule must begin with versioned PASS-A")
    if sum(program.mode is AModeV1.PASS_A_V1 for program in ordered) != 1:
        raise TaskspaceG8A3N2AllocatorError("conditional-A schedule must contain exactly one PASS-A")
    return ordered


def g12_branches_from_acquisition(acquisition: object) -> tuple[G8BranchV1, ...]:
    """Adapt the frozen G12 acquisition without importing it at module load."""

    proposals = getattr(acquisition, "proposals", None)
    if type(proposals) is not tuple or not proposals:
        raise TaskspaceG8A3N2AllocatorError("G12 acquisition omitted its complete proposal tuple")
    branches: list[G8BranchV1] = []
    families: set[str] = set()
    orders: set[str] = set()
    for proposal in proposals:
        receipt = getattr(proposal, "receipt", None)
        program = getattr(proposal, "program", None)
        try:
            family = receipt.family.value
            prefix_order = receipt.prefix_order.value
            branch = G8BranchV1(
                proposal_id=proposal.proposal_id,
                program_sha256=receipt.program_sha256,
                family=family,
                prefix_order=prefix_order,
                palette_bound_per_class=receipt.palette_bound_per_class,
                prefix_cell_count=receipt.requested_prefix_cell_count,
                program=program,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise TaskspaceG8A3N2AllocatorError("G12 proposal shape changed") from exc
        branches.append(branch)
        families.add(family)
        orders.add(prefix_order)
    expected_families = {
        "CLASS_SHARED_TARGET_MEDOID_V1",
        "CLASS_BOUNDED_TARGET_MEDOIDS_V1",
        "TARGET_PIXEL_RGB_ORACLE_CONTROL_V1",
    }
    expected_orders = {"CANONICAL_ADDRESS_V1", "TARGET_RGB_SSE_DESCENDING_V1"}
    if families != expected_families or orders != expected_orders:
        raise TaskspaceG8A3N2AllocatorError("G12 acquisition omitted a required family or order")
    return tuple(branches)


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspaceG8A3N2AllocatorError(f"custody path is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspaceG8A3N2AllocatorError(f"cannot read custody path: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after or len(payload) != before.st_size:
        raise TaskspaceG8A3N2AllocatorError(f"custody path changed while reading: {path}")
    return payload


def _strict_json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TaskspaceG8A3N2AllocatorError(f"{field} must have exactly one terminal newline")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceG8A3N2AllocatorError(f"{field} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload[:-1].decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspaceG8A3N2AllocatorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceG8A3N2AllocatorError(f"{field} is not strict ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) + b"\n" != payload:
        raise TaskspaceG8A3N2AllocatorError(f"{field} is not canonical on parse-back")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _recover_or_clean_atomic_partials(path: Path, payload: bytes) -> None:
    """Recover complete matching scratch or remove only certified prefixes."""

    pattern = f".{path.name}.partial.*"
    changed = False
    for partial in sorted(path.parent.glob(pattern)):
        try:
            metadata = partial.stat(follow_symlinks=False)
        except OSError as exc:
            raise TaskspaceG8A3N2AllocatorError(f"cannot inspect atomic scratch: {partial}") from exc
        if not stat.S_ISREG(metadata.st_mode) or partial.is_symlink():
            raise TaskspaceG8A3N2AllocatorError(f"atomic scratch is not a safe regular file: {partial}")
        observed = _read_stable_regular(partial)
        if observed == payload:
            if not path.exists():
                try:
                    os.link(partial, path)
                except FileExistsError:
                    pass
            if _read_stable_regular(path) != payload:
                raise TaskspaceG8A3N2AllocatorError(f"recovered atomic scratch conflicts with destination: {partial}")
            partial.unlink()
            changed = True
            continue
        if len(observed) < len(payload) and payload.startswith(observed):
            # This is a certified interrupted prefix of the exact requested
            # write.  It was never published and is safe scratch to discard.
            partial.unlink()
            changed = True
            continue
        raise TaskspaceG8A3N2AllocatorError(
            f"atomic scratch is neither recoverable nor a certified payload prefix: {partial}"
        )
    if changed:
        _fsync_directory(path.parent)


def _atomic_write_once_or_equal(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Publish exact bytes atomically without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _recover_or_clean_atomic_partials(path, payload)
    if path.exists():
        if _read_stable_regular(path) != payload:
            raise TaskspaceG8A3N2AllocatorError(f"refusing to overwrite different checkpoint: {path}")
        return
    temporary = path.parent / f".{path.name}.partial.{os.getpid()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, mode)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _read_stable_regular(path) != payload:
                    raise TaskspaceG8A3N2AllocatorError(f"checkpoint race produced different bytes: {path}") from None
            _fsync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    except FileExistsError as exc:
        raise TaskspaceG8A3N2AllocatorError(f"stale partial checkpoint exists: {temporary}") from exc


def _safe_run_directory(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == Path("/") or absolute == Path("/tmp") or Path("/tmp") in absolute.parents:
        raise TaskspaceG8A3N2AllocatorError("durable run directory must not be /, /tmp, or below /tmp")
    if absolute.exists() and absolute.is_symlink():
        raise TaskspaceG8A3N2AllocatorError("run directory must not be a symlink")
    return absolute


@dataclass(slots=True)
class AtomicRunStore:
    run_dir: Path
    manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_dir: Path,
        *,
        stable_contract: dict[str, Any],
        pointer_start: dict[str, Any],
    ) -> AtomicRunStore:
        root = _safe_run_directory(run_dir)
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise TaskspaceG8A3N2AllocatorError(
                "run directory already exists; use --resume-from for an existing run"
            ) from exc
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "lane_id": LANE_ID,
            "stable_contract": stable_contract,
            "pointer_start": pointer_start,
            "pointer_start_sha256": _sha256(_canonical_json(pointer_start)),
        }
        _atomic_write_once_or_equal(root / "manifest.json", _canonical_json(manifest) + b"\n")
        return cls(root, manifest)

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        *,
        expected_stable_contract: dict[str, Any],
    ) -> AtomicRunStore:
        root = _safe_run_directory(run_dir)
        manifest = _strict_json_object(_read_stable_regular(root / "manifest.json"), field="run manifest")
        if (
            set(manifest)
            != {
                "schema",
                "lane_id",
                "stable_contract",
                "pointer_start",
                "pointer_start_sha256",
            }
            or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("lane_id") != LANE_ID
        ):
            raise TaskspaceG8A3N2AllocatorError("run manifest schema/lane changed")
        if manifest.get("stable_contract") != expected_stable_contract:
            raise TaskspaceG8A3N2AllocatorError("resume config/implementation/input custody changed")
        pointer_start = manifest.get("pointer_start")
        if type(pointer_start) is not dict or manifest.get("pointer_start_sha256") != _sha256(
            _canonical_json(pointer_start)
        ):
            raise TaskspaceG8A3N2AllocatorError("stored pointer start snapshot is internally inconsistent")
        return cls(root, manifest)

    def append_pointer_snapshot(self, snapshot: dict[str, Any], *, label: str) -> Path:
        _require_id(label, "pointer snapshot label")
        digest = _sha256(_canonical_json(snapshot))
        path = self.run_dir / "pointers" / f"{label}.{digest}.json"
        _atomic_write_once_or_equal(path, _canonical_json(snapshot) + b"\n")
        return path

    def register_extension_contract(self, extension_id: str, contract: dict[str, Any]) -> Path:
        """Append a frozen provider contract without rewriting the core manifest."""

        _require_id(extension_id, "extension_id")
        if type(contract) is not dict or not contract:
            raise TaskspaceG8A3N2AllocatorError("extension contract must be one nonempty object")
        payload = (
            _canonical_json(
                {
                    "schema": "tac.taskspace_g8_a3_n2_allocator_extension_contract.v1",
                    "extension_id": extension_id,
                    "core_manifest_sha256": _sha256(_canonical_json(self.manifest)),
                    "contract": contract,
                }
            )
            + b"\n"
        )
        path = self.run_dir / "extensions" / f"{extension_id}.json"
        _atomic_write_once_or_equal(path, payload)
        return path

    def checkpoint_json(self, stage_name: str, body: dict[str, Any]) -> Path:
        _require_id(stage_name, "stage_name")
        path = self.run_dir / "checkpoints" / f"{stage_name}.json"
        envelope = {
            "schema": CHECKPOINT_SCHEMA,
            "stage_name": stage_name,
            "manifest_sha256": _sha256(_canonical_json(self.manifest)),
            "body": body,
        }
        _atomic_write_once_or_equal(path, _canonical_json(envelope) + b"\n")
        return path

    def load_checkpoint_json(self, stage_name: str) -> dict[str, Any] | None:
        _require_id(stage_name, "stage_name")
        path = self.run_dir / "checkpoints" / f"{stage_name}.json"
        if not path.exists():
            return None
        envelope = _strict_json_object(_read_stable_regular(path), field=f"checkpoint {stage_name}")
        expected = {
            "schema": CHECKPOINT_SCHEMA,
            "stage_name": stage_name,
            "manifest_sha256": _sha256(_canonical_json(self.manifest)),
        }
        if type(envelope) is not dict or any(envelope.get(key) != value for key, value in expected.items()):
            raise TaskspaceG8A3N2AllocatorError(f"checkpoint {stage_name} changed custody")
        body = envelope.get("body")
        if type(body) is not dict:
            raise TaskspaceG8A3N2AllocatorError(f"checkpoint {stage_name} body is not one object")
        return body

    def checkpoint_measurement(
        self,
        stage_name: str,
        measurement: WholeObjectMeasurementV1,
    ) -> Path:
        branch_dir = self.run_dir / "branches" / stage_name
        archive_path = branch_dir / "selected.not_a_candidate.zip"
        _atomic_write_once_or_equal(archive_path, measurement.selected_archive_payload)
        relative_archive = os.fspath(archive_path.relative_to(self.run_dir))
        self.checkpoint_json(stage_name, measurement.as_dict(archive_path=relative_archive))
        return archive_path

    def load_measurement(self, stage_name: str) -> WholeObjectMeasurementV1 | None:
        body = self.load_checkpoint_json(stage_name)
        if body is None:
            return None
        archive_path_raw = body.get("selected_archive_path")
        if (
            type(archive_path_raw) is not str
            or Path(archive_path_raw).is_absolute()
            or ".." in Path(archive_path_raw).parts
        ):
            raise TaskspaceG8A3N2AllocatorError("checkpoint archive path escaped the run directory")
        archive = _read_stable_regular(self.run_dir / archive_path_raw)
        return WholeObjectMeasurementV1.from_dict_and_archive(body, archive)


class CoupledBackendV1(Protocol):
    """Injected real/fake backend; all methods are deterministic and source-bound."""

    def measure_primary_baseline(self) -> WholeObjectMeasurementV1: ...

    def measure_exact_semantic_control(self) -> WholeObjectMeasurementV1: ...

    def acquire_g8(self) -> tuple[object, dict[str, Any]]: ...

    def freeze_g8_acquisition(
        self,
        acquisition: object,
        branches: tuple[G8BranchV1, ...],
    ) -> dict[str, Any]: ...

    def restore_g8_acquisition(
        self,
        payload: dict[str, Any],
    ) -> tuple[G8BranchV1, ...]: ...

    def measure_g8_pass_a(self, branch: G8BranchV1) -> WholeObjectMeasurementV1: ...

    def acquire_g0_a(self, row_counts: tuple[int, ...]) -> tuple[AProgramV1, ...]: ...

    def acquire_a(self, branch: G8BranchV1, row_counts: tuple[int, ...]) -> tuple[AProgramV1, ...]: ...

    def measure_g8_a(self, branch: G8BranchV1, program: AProgramV1) -> WholeObjectMeasurementV1: ...

    def measure_g0_a(self, program: AProgramV1) -> WholeObjectMeasurementV1: ...


AdditionalAProviderV1 = Callable[[G8BranchV1 | None, tuple[int, ...]], Sequence[AProgramV1]]


def merge_a_programs(
    native: Sequence[AProgramV1],
    additional: Sequence[AProgramV1],
) -> tuple[AProgramV1, ...]:
    """Merge an append-only counted-A extension without disturbing PASS-first."""

    base = require_pass_a_first(native)
    extras = tuple(additional)
    if any(type(program) is not AProgramV1 for program in extras):
        raise TaskspaceG8A3N2AllocatorError("additional-A provider returned a noncanonical descriptor")
    if any(program.mode is AModeV1.PASS_A_V1 for program in extras):
        raise TaskspaceG8A3N2AllocatorError("additional-A provider must not inject another PASS-A")
    combined = [base[0]]
    seen = {base[0].program_sha256}
    for program in (*base[1:], *extras):
        if program.program_sha256 in seen:
            continue
        seen.add(program.program_sha256)
        combined.append(program)
    return require_pass_a_first(tuple(combined))


def _checkpoint_or_measure(
    store: AtomicRunStore,
    stage_name: str,
    callback: Callable[[], WholeObjectMeasurementV1],
) -> WholeObjectMeasurementV1:
    existing = store.load_measurement(stage_name)
    if existing is not None:
        return existing
    measured = callback()
    if type(measured) is not WholeObjectMeasurementV1:
        raise TaskspaceG8A3N2AllocatorError("backend returned noncanonical whole-object measurement")
    store.checkpoint_measurement(stage_name, measured)
    return measured


def _stage_token(prefix: str, identifier: str) -> str:
    _require_id(prefix, "stage prefix")
    _require_stage_source_identity(identifier, "stage source identity")
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)
    return f"{prefix}.{normalized[:96]}.{_sha256(identifier.encode('ascii'))[:12]}"


def run_coupled_experiment(
    *,
    store: AtomicRunStore,
    backend: CoupledBackendV1,
    g8_prefix_mode: str,
    a_prefixes: tuple[int, ...],
    additional_a_provider: AdditionalAProviderV1 | None = None,
    additional_a_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run/resume singleton G8 screening and retained conditional-A follow-up."""

    if (additional_a_provider is None) != (additional_a_contract is None):
        raise TaskspaceG8A3N2AllocatorError(
            "additional-A provider and its implementation/config contract must be supplied together"
        )
    if additional_a_contract is not None:
        store.register_extension_contract("additional_counted_a_v1", additional_a_contract)

    baseline = _checkpoint_or_measure(store, "stage_010.primary_baseline", backend.measure_primary_baseline)
    if baseline.g8_program_sha256 is not None or baseline.a_mode is not AModeV1.PASS_A_V1:
        raise TaskspaceG8A3N2AllocatorError("primary baseline must be no-G8 plus versioned PASS-A")
    exact_control = _checkpoint_or_measure(
        store,
        "stage_020.exact_semantic_control",
        backend.measure_exact_semantic_control,
    )
    if exact_control.measurement_id == baseline.measurement_id:
        raise TaskspaceG8A3N2AllocatorError("exact semantic control was conflated with PASS baseline")

    g0_additional: Sequence[AProgramV1] = ()
    if additional_a_provider is not None:
        g0_additional = additional_a_provider(None, a_prefixes)
    g0_programs = merge_a_programs(backend.acquire_g0_a(a_prefixes), g0_additional)
    # This write-once equality gate deliberately precedes every G0-A
    # measurement.  Resume may recompute an encoder-only ranking, but it may
    # never measure from descriptors that differ from the frozen acquisition.
    store.checkpoint_json(
        "stage_024.g0_a_programs",
        {
            "acquisition_kind": "deterministically_recomputed_then_write_once_equality_checked.v1",
            "programs": [program.as_dict() for program in g0_programs],
        },
    )
    if (
        g0_programs[0].program_sha256 != baseline.a_program_sha256
        or g0_programs[0].acquisition_y1_sha256 != baseline.camera_y1_sha256
    ):
        raise TaskspaceG8A3N2AllocatorError("G0 acquisition PASS-A differs from exact no-G8 baseline")
    g0_optimal_rows = [baseline]
    for program in g0_programs[1:]:
        if program.acquisition_y1_sha256 != baseline.camera_y1_sha256:
            raise TaskspaceG8A3N2AllocatorError("G0-optimal A program was not ranked on no-G8 Y1")
        stage = _stage_token(
            "stage_025.g0_optimal_a",
            f"{program.program_id}:{program.program_sha256}",
        )
        row = _checkpoint_or_measure(
            store,
            stage,
            lambda program=program: backend.measure_g0_a(program),
        )
        if row.g8_program_sha256 is not None or row.a_program_sha256 != program.program_sha256:
            raise TaskspaceG8A3N2AllocatorError("G0-optimal measurement changed G/A program custody")
        g0_optimal_rows.append(row)
    assert_a_variant_invariants(baseline, g0_optimal_rows)
    store.checkpoint_json(
        "stage_026.g0_a_acquisition",
        {
            "acquisition_y1_sha256": baseline.camera_y1_sha256,
            "programs": [program.as_dict() for program in g0_programs],
            "measurements": [row.as_dict() for row in g0_optimal_rows],
            "transitions_vs_g0_pass": [
                PairwiseTransitionV1.between(baseline, row).as_dict() for row in g0_optimal_rows[1:]
            ],
        },
    )

    g12_checkpoint = store.load_checkpoint_json("stage_030.g12_acquisition")
    if g12_checkpoint is None:
        acquisition, acquisition_summary = backend.acquire_g8()
        if type(acquisition_summary) is not dict:
            raise TaskspaceG8A3N2AllocatorError("G12 acquisition summary must be one dense-free object")
        debt_count = acquisition_summary.get("realization_debt_cell_count")
        if type(debt_count) is not int or debt_count < 1:
            raise TaskspaceG8A3N2AllocatorError("G12 acquisition summary omitted positive realization debt")
        allowed_prefixes = resolve_g8_prefixes(g8_prefix_mode, debt_count=debt_count)
        all_branches = g12_branches_from_acquisition(acquisition)
        branches, aliases = dedupe_g8_branches(all_branches, allowed_prefixes=allowed_prefixes)
        g12_checkpoint = {
            "summary": acquisition_summary,
            "allowed_prefixes": list(allowed_prefixes),
            "proposal_count": len(all_branches),
            "unique_screened_program_count": len(branches),
            "aliases_by_program_sha256": aliases,
            "screened_branches": [branch.as_dict() for branch in branches],
            "backend_restore": backend.freeze_g8_acquisition(acquisition, branches),
        }
        store.checkpoint_json("stage_030.g12_acquisition", g12_checkpoint)
    else:
        acquisition_summary = g12_checkpoint.get("summary")
        if type(acquisition_summary) is not dict:
            raise TaskspaceG8A3N2AllocatorError("restored G12 summary changed exact type")
        debt_count = acquisition_summary.get("realization_debt_cell_count")
        if type(debt_count) is not int or debt_count < 1:
            raise TaskspaceG8A3N2AllocatorError("restored G12 summary omitted exact debt")
        allowed_prefixes = resolve_g8_prefixes(g8_prefix_mode, debt_count=debt_count)
        if g12_checkpoint.get("allowed_prefixes") != list(allowed_prefixes):
            raise TaskspaceG8A3N2AllocatorError("restored G12 prefix universe differs from reviewed config")
        restore_payload = g12_checkpoint.get("backend_restore")
        if type(restore_payload) is not dict:
            raise TaskspaceG8A3N2AllocatorError("G12 checkpoint omitted typed backend restore payload")
        branches = backend.restore_g8_acquisition(restore_payload)
        expected_descriptors = g12_checkpoint.get("screened_branches")
        if [branch.as_dict() for branch in branches] != expected_descriptors:
            raise TaskspaceG8A3N2AllocatorError("restored G8 programs differ from checkpoint descriptors")
        aliases = g12_checkpoint.get("aliases_by_program_sha256")
        if type(aliases) is not dict:
            raise TaskspaceG8A3N2AllocatorError("restored G12 alias map changed exact type")
        proposal_count = g12_checkpoint.get("proposal_count")
        if type(proposal_count) is not int or proposal_count < len(branches):
            raise TaskspaceG8A3N2AllocatorError("restored G12 proposal count is invalid")
        all_branches = (None,) * proposal_count

    pass_rows: list[WholeObjectMeasurementV1] = []
    for branch in branches:
        stage = _stage_token("stage_100.g8_pass_a", branch.proposal_id)
        row = _checkpoint_or_measure(store, stage, lambda branch=branch: backend.measure_g8_pass_a(branch))
        if row.baseline_bundle_sha256 != baseline.baseline_bundle_sha256:
            raise TaskspaceG8A3N2AllocatorError("G8 singleton did not start from the immutable baseline bundle")
        if row.g8_program_sha256 != branch.program_sha256 or row.a_mode is not AModeV1.PASS_A_V1:
            raise TaskspaceG8A3N2AllocatorError("G8 screening row is not its branch plus PASS-A")
        pass_rows.append(row)

    retained_ids = retained_g8_branch_ids(branches, pass_rows)
    nondominated = nondominated_g8_indices(pass_rows)
    store.checkpoint_json(
        "stage_200.g8_screen",
        {
            "same_baseline_bundle_sha256": baseline.baseline_bundle_sha256,
            "nondominated_proposal_ids": [branches[index].proposal_id for index in nondominated],
            "retained_for_a_proposal_ids": list(retained_ids),
            "transitions_vs_g0_pass": [PairwiseTransitionV1.between(baseline, row).as_dict() for row in pass_rows],
        },
    )

    treatments: list[dict[str, Any]] = []
    retained = [branch for branch in branches if branch.proposal_id in set(retained_ids)]
    row_by_branch = dict(zip((branch.proposal_id for branch in branches), pass_rows, strict=True))
    for branch in retained:
        pass_row = row_by_branch[branch.proposal_id]
        native_programs = backend.acquire_a(branch, a_prefixes)
        additional_programs: Sequence[AProgramV1] = ()
        if additional_a_provider is not None:
            additional_programs = additional_a_provider(branch, a_prefixes)
        programs = merge_a_programs(native_programs, additional_programs)
        # The branch acquisition gate is keyed by the G8 program, not merely
        # by an A digest, so two source surfaces can never alias on resume.
        store.checkpoint_json(
            _stage_token("stage_250.branch_a_programs", branch.proposal_id),
            {
                "acquisition_kind": "deterministically_recomputed_then_write_once_equality_checked.v1",
                "g8_branch": branch.as_dict(),
                "programs": [program.as_dict() for program in programs],
            },
        )
        if programs[0].acquisition_y1_sha256 != pass_row.camera_y1_sha256:
            raise TaskspaceG8A3N2AllocatorError("PASS-A acquisition is not bound to the branch post-G8 Y1")
        branch_variants = [pass_row]
        for program in programs:
            if program.mode is AModeV1.PASS_A_V1:
                if program.program_sha256 != pass_row.a_program_sha256:
                    raise TaskspaceG8A3N2AllocatorError("backend PASS-A program differs from measured branch")
                continue
            if program.acquisition_y1_sha256 != pass_row.camera_y1_sha256:
                raise TaskspaceG8A3N2AllocatorError("A program was not acquired on this branch's post-G8 Y1")
            g8_stage = _stage_token(
                "stage_300.g8_a",
                f"{branch.proposal_id}:{program.program_id}",
            )
            g8_a = _checkpoint_or_measure(
                store,
                g8_stage,
                lambda branch=branch, program=program: backend.measure_g8_a(branch, program),
            )
            g0_stage = _stage_token(
                "stage_310.g0_a_control",
                f"{branch.proposal_id}:{program.program_id}:{program.program_sha256}",
            )
            g0_a = _checkpoint_or_measure(
                store,
                g0_stage,
                lambda program=program: backend.measure_g0_a(program),
            )
            if g8_a.a_program_sha256 != program.program_sha256:
                raise TaskspaceG8A3N2AllocatorError("G8+A measurement used a different canonical A program")
            if g0_a.a_program_sha256 != program.program_sha256:
                raise TaskspaceG8A3N2AllocatorError("matched G0 control changed the canonical A program")
            if g0_a.a_source_binding_sha256 != baseline.a_source_binding_sha256:
                raise TaskspaceG8A3N2AllocatorError("matched G0 control did not rebind A to actual no-G8 source")
            if g0_a.a_source_binding_sha256 == g8_a.a_source_binding_sha256:
                raise TaskspaceG8A3N2AllocatorError("matched G0 control reused post-G8 source-bound A bytes")
            branch_variants.append(g8_a)
            treatments.append(
                {
                    "g8_branch": branch.as_dict(),
                    "a_program": program.as_dict(),
                    "g8_a_measurement": g8_a.as_dict(),
                    "g0_a_measurement": g0_a.as_dict(),
                    "transition_vs_g8_pass": PairwiseTransitionV1.between(pass_row, g8_a).as_dict(),
                    "transition_vs_g0_pass": PairwiseTransitionV1.between(baseline, g0_a).as_dict(),
                    "interaction_I": interaction_score(
                        g8_a=g8_a,
                        g8_pass=pass_row,
                        g0_a=g0_a,
                        g0_pass=baseline,
                    ),
                }
            )
        assert_a_variant_invariants(pass_row, branch_variants)

    all_rows = [*g0_optimal_rows, *pass_rows]
    for treatment in treatments:
        # Reopen the exact persisted row instead of trusting a summary-only dict.
        identifier = f"{treatment['g8_branch']['proposal_id']}:{treatment['a_program']['program_id']}"
        row = store.load_measurement(_stage_token("stage_300.g8_a", identifier))
        assert row is not None
        all_rows.append(row)
        control_identifier = (
            f"{treatment['g8_branch']['proposal_id']}:"
            f"{treatment['a_program']['program_id']}:"
            f"{treatment['a_program']['program_sha256']}"
        )
        control = store.load_measurement(_stage_token("stage_310.g0_a_control", control_identifier))
        assert control is not None
        all_rows.append(control)
    best = min(all_rows, key=lambda row: (row.score, row.measurement_id))
    final_archive = store.run_dir / "final.selected.not_a_candidate.zip"
    _atomic_write_once_or_equal(final_archive, best.selected_archive_payload)
    result = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "axis": AXIS,
        "scope": "real ep725 n2 coupled G8 plus conditional-A advisory allocation",
        "baseline": baseline.as_dict(),
        "exact_semantic_g_control": exact_control.as_dict(),
        "diagnostic_controls": {
            "production_selection_excluded_measurement_ids": [exact_control.measurement_id],
            "reason": "exact semantic G chronology control is outside the PASS/G8 production chain",
        },
        "g0_a_acquisition": {
            "acquisition_y1_sha256": baseline.camera_y1_sha256,
            "programs": [program.as_dict() for program in g0_programs],
            "measurements": [row.as_dict() for row in g0_optimal_rows],
        },
        "g12": {
            "summary": acquisition_summary,
            "allowed_prefixes": list(allowed_prefixes),
            "proposal_count": len(all_branches),
            "unique_screened_program_count": len(branches),
            "aliases_by_program_sha256": aliases,
        },
        "g8_screen": {
            "branches": [branch.as_dict() for branch in branches],
            "measurements": [row.as_dict() for row in pass_rows],
            "nondominated_proposal_ids": [branches[index].proposal_id for index in nondominated],
            "retained_for_a_proposal_ids": list(retained_ids),
        },
        "conditional_a_treatments": treatments,
        "selected_research_row": best.as_dict(archive_path=os.fspath(final_archive.relative_to(store.run_dir))),
        "truth": {
            "research_only": True,
            "n2_only": True,
            "macos_cpu_advisory": True,
            "n600_evaluation": False,
            "authoritative_contest_cpu_evaluation": False,
            "authoritative_contest_cuda_evaluation": False,
            "score_claim": False,
            "candidate_archive_eligible": False,
            "promotion_eligible": False,
            "originality_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "pointer_moved": False,
            "public_archive_payload_reused": False,
            "dense_frames_persisted": False,
        },
    }
    return result


def parse_final_receipt(payload: bytes) -> dict[str, Any]:
    value = _strict_json_object(payload, field="G14 final receipt")
    expected_keys = {
        "schema",
        "lane_id",
        "axis",
        "scope",
        "manifest_sha256",
        "pointer_start",
        "pointer_latest",
        "latest_pointer_comparison",
        "pointer_observation_paths",
        "stable_contract",
        "baseline",
        "exact_semantic_g_control",
        "diagnostic_controls",
        "g0_a_acquisition",
        "g12",
        "g8_screen",
        "conditional_a_treatments",
        "selected_research_row",
        "truth",
    }
    if set(value) != expected_keys or value.get("schema") != SCHEMA:
        raise TaskspaceG8A3N2AllocatorError("G14 final receipt top-level schema changed")
    if value.get("lane_id") != LANE_ID or value.get("axis") != AXIS:
        raise TaskspaceG8A3N2AllocatorError("G14 final receipt lane/axis changed")
    comparison = value.get("latest_pointer_comparison")
    if comparison != {
        "comparable": False,
        "classification": "noncomparable_n2_omitted_runtime_research_lower_bound",
        "pairwise_common_basis_deltas_valid": True,
    }:
        raise TaskspaceG8A3N2AllocatorError("latest pointer comparison became false authority")
    truth = value.get("truth")
    expected_truth = {
        "research_only": True,
        "n2_only": True,
        "macos_cpu_advisory": True,
        "n600_evaluation": False,
        "authoritative_contest_cpu_evaluation": False,
        "authoritative_contest_cuda_evaluation": False,
        "score_claim": False,
        "candidate_archive_eligible": False,
        "promotion_eligible": False,
        "originality_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "public_archive_payload_reused": False,
        "dense_frames_persisted": False,
    }
    if truth != expected_truth:
        raise TaskspaceG8A3N2AllocatorError("G14 final receipt authority labels became permissive")
    selected = value.get("selected_research_row")
    if type(selected) is not dict or not str(selected.get("selected_archive_path", "")).endswith(
        ".not_a_candidate.zip"
    ):
        raise TaskspaceG8A3N2AllocatorError("G14 final archive is not explicitly non-candidate")
    summaries: list[object] = [
        value.get("baseline"),
        value.get("exact_semantic_g_control"),
        selected,
    ]
    g0 = value.get("g0_a_acquisition")
    g8 = value.get("g8_screen")
    treatments = value.get("conditional_a_treatments")
    if (
        type(g0) is not dict
        or type(g0.get("measurements")) is not list
        or type(g8) is not dict
        or type(g8.get("measurements")) is not list
        or type(treatments) is not list
    ):
        raise TaskspaceG8A3N2AllocatorError("G14 final receipt measurement collections changed")
    summaries.extend(g0["measurements"])
    summaries.extend(g8["measurements"])
    for treatment in treatments:
        if type(treatment) is not dict:
            raise TaskspaceG8A3N2AllocatorError("G14 conditional-A treatment changed exact type")
        summaries.extend((treatment.get("g8_a_measurement"), treatment.get("g0_a_measurement")))
    for summary in summaries:
        WholeObjectMeasurementV1.validate_serialized_summary(summary)
    return value


def finalize_receipt(
    *,
    store: AtomicRunStore,
    experiment: dict[str, Any],
    pointer_latest: dict[str, Any],
) -> bytes:
    latest_path = store.append_pointer_snapshot(pointer_latest, label="final")
    pointer_paths = sorted(
        os.fspath(path.relative_to(store.run_dir)) for path in (store.run_dir / "pointers").glob("*.json")
    )
    value = {
        **experiment,
        "manifest_sha256": _sha256(_canonical_json(store.manifest)),
        "pointer_start": store.manifest["pointer_start"],
        "pointer_latest": pointer_latest,
        "latest_pointer_comparison": {
            "comparable": False,
            "classification": "noncomparable_n2_omitted_runtime_research_lower_bound",
            "pairwise_common_basis_deltas_valid": True,
        },
        "pointer_observation_paths": pointer_paths,
        "stable_contract": store.manifest["stable_contract"],
    }
    del latest_path
    payload = _canonical_json(value) + b"\n"
    parse_final_receipt(payload)
    _atomic_write_once_or_equal(store.run_dir / "final_receipt.json", payload)
    return payload


def _g10_blocker(reason: str) -> bytes:
    return (
        _canonical_json(
            {
                "schema": BLOCKER_SCHEMA,
                "blocker_code": G10ProductionCompositionUnavailable.blocker_code,
                "reason": reason,
                "before_source_decode": True,
                "before_scorer_load": True,
                "fallback_used": False,
                "research_only": True,
            }
        )
        + b"\n"
    )


def require_g10_production_surface() -> dict[str, object]:
    """Resolve the complete production surface or fail before real work."""

    for relative, expected_sha256 in G10_FROZEN_IMPLEMENTATION.items():
        observed = _sha256(_read_stable_regular(REPO / relative))
        if observed != expected_sha256:
            raise G10ProductionCompositionUnavailable(f"frozen G10 implementation custody drifted: {relative}")
    try:
        from tac.witness_dsl import taskspace_monolithic_pga_receiver as receiver
        from tac.witness_dsl import taskspace_pass_conditional_a as conditional_a
        from tac.witness_dsl import taskspace_pass_semantic_g as pass_g
    except ImportError as exc:
        raise G10ProductionCompositionUnavailable("G10 production modules are unavailable") from exc
    required = {
        "compile_pass_semantic_g_envelope": getattr(pass_g, "compile_pass_semantic_g_envelope", None),
        "compile_pass_conditional_a": getattr(conditional_a, "compile_pass_conditional_a", None),
        "receive_from_causal_surface": getattr(
            receiver,
            "receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface",
            None,
        ),
    }
    missing = sorted(name for name, value in required.items() if not callable(value))
    if missing:
        raise G10ProductionCompositionUnavailable(f"G10 production callables missing: {missing}")
    return required


def _git_head() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskspaceG8A3N2AllocatorError("git HEAD is unavailable") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise TaskspaceG8A3N2AllocatorError("git HEAD is not canonical lowercase SHA-1")
    return head


def _implementation_custody() -> list[dict[str, Any]]:
    paths = REAL_PATH_IMPLEMENTATION_CLOSURE
    if len(paths) != len(set(paths)):
        raise TaskspaceG8A3N2AllocatorError("real-path implementation closure repeats a path")
    return [
        {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
        for relative in paths
        for payload in (_read_stable_regular(REPO / relative),)
    ]


def _a_program_identity(program: object) -> str:
    try:
        mode = program.mode.value
        constant_rows = [
            {
                "source_pair_id": row.source_pair_id,
                "scorer_row": row.scorer_row,
                "scorer_col": row.scorer_col,
                "rgb_u8": list(row.rgb_u8),
            }
            for row in program.constant_rgb_cells
        ]
        copy_rows = [
            {
                "source_pair_id": row.source_pair_id,
                "scorer_row": row.scorer_row,
                "scorer_col": row.scorer_col,
            }
            for row in program.corrected_y1_copy_cells
        ]
    except AttributeError as exc:
        raise TaskspaceG8A3N2AllocatorError("A program changed canonical high-level shape") from exc
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.g14_high_level_a_program_identity.v1",
                "mode": mode,
                "constant_rgb_cells": constant_rows,
                "corrected_y1_copy_cells": copy_rows,
            }
        )
    )


def _a_programs_for_surface(
    *,
    source_pair_ids: tuple[int, ...],
    camera_p0: np.ndarray,
    conditional_y1: np.ndarray,
    target_y0: np.ndarray,
    row_counts: tuple[int, ...],
    source_tag: str,
) -> tuple[AProgramV1, ...]:
    """Acquire canonical A programs on this exact conditional-Y1 surface."""

    from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
    from tac.witness_dsl.predictor_preserving_coupled_preimage import (
        CorrectedY1SupportCopyCellV1,
        PredictorPreservingA3Mode,
        PredictorPreservingA3ProgramV1,
        SparseConstantRGBCellV1,
    )

    expected_shape = (len(source_pair_ids), 874, 1164, 3)
    for field, value in (
        ("camera_p0", camera_p0),
        ("conditional_y1", conditional_y1),
        ("target_y0", target_y0),
    ):
        if value.dtype != np.uint8 or value.shape != expected_shape:
            raise TaskspaceG8A3N2AllocatorError(f"{field} changed exact n2 camera ABI")
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )

    def numerators(frames: np.ndarray) -> tuple[np.ndarray, int]:
        rows: list[np.ndarray] = []
        denominator: int | None = None
        for frame in frames:
            current, current_denominator = operator.apply_numerators(frame)
            if denominator is not None and denominator != current_denominator:
                raise TaskspaceG8A3N2AllocatorError("frozen scorer resize denominator changed")
            denominator = current_denominator
            rows.append(np.ascontiguousarray(current, dtype=np.int64))
        assert denominator is not None
        return np.stack(rows), denominator

    p0_num, denominator = numerators(camera_p0)
    y1_num, y1_denominator = numerators(conditional_y1)
    target_num, target_denominator = numerators(target_y0)
    if denominator != y1_denominator or denominator != target_denominator:
        raise TaskspaceG8A3N2AllocatorError("P0/Y1/target resize denominators differ")
    base_delta = p0_num - target_num
    base_sse = np.sum(base_delta * base_delta, axis=-1, dtype=np.int64)
    target_rgb = np.floor_divide(target_num + denominator // 2, denominator)
    target_rgb = np.clip(target_rgb, 0, 255).astype(np.uint8)
    pass_program = PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1)
    y1_sha = _array_sha256(conditional_y1)
    ranking_common = {
        "schema": "tac.g14_post_surface_a_ranking.v1",
        "source_pair_ids": list(source_pair_ids),
        "camera_p0_numerators_sha256": _array_sha256(p0_num),
        "conditional_y1_numerators_sha256": _array_sha256(y1_num),
        "target_y0_numerators_sha256": _array_sha256(target_num),
        "resize_denominator": denominator,
        "acquisition_y1_sha256": y1_sha,
    }
    programs = [
        AProgramV1(
            program_id=f"{source_tag}:pass",
            program_sha256=_a_program_identity(pass_program),
            mode=AModeV1.PASS_A_V1,
            row_count=0,
            acquisition_y1_sha256=y1_sha,
            program=pass_program,
            ranking_sha256=_sha256(_canonical_json({**ranking_common, "mode": "PASS_A_V1"})),
        )
    ]
    candidates = (
        (
            AModeV1.TARGET_CONSTANT_RGB_V1,
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            target_rgb.astype(np.int64) * denominator,
        ),
        (
            AModeV1.POST_G8_Y1_SUPPORT_COPY_V1,
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
            y1_num,
        ),
    )
    for mode, wire_mode, candidate_num in candidates:
        candidate_delta = candidate_num - target_num
        candidate_sse = np.sum(candidate_delta * candidate_delta, axis=-1, dtype=np.int64)
        reduction = base_sse - candidate_sse
        positive = np.flatnonzero(reduction.reshape(-1) > 0)
        local_pair, scorer_row, scorer_col = np.unravel_index(
            positive,
            (len(source_pair_ids), 384, 512),
        )
        gains = reduction.reshape(-1)[positive]
        order = np.lexsort((scorer_col, scorer_row, local_pair, -gains))
        ranking_sha = _sha256(
            _canonical_json(
                {
                    **ranking_common,
                    "mode": mode.value,
                    "available_positive_rows": len(order),
                    "ordered_address_sha256": _array_sha256(
                        np.ascontiguousarray(
                            np.stack((local_pair[order], scorer_row[order], scorer_col[order]), axis=1),
                            dtype=np.int64,
                        )
                    ),
                }
            )
        )
        for count in row_counts:
            if count == 0 or count > len(order):
                continue
            chosen = order[:count]
            if mode is AModeV1.TARGET_CONSTANT_RGB_V1:
                rows = tuple(
                    sorted(
                        (
                            SparseConstantRGBCellV1(
                                source_pair_ids[int(local_pair[index])],
                                int(scorer_row[index]),
                                int(scorer_col[index]),
                                tuple(
                                    int(value)
                                    for value in target_rgb[
                                        int(local_pair[index]),
                                        int(scorer_row[index]),
                                        int(scorer_col[index]),
                                    ]
                                ),
                            )
                            for index in chosen
                        ),
                        key=lambda row: row.address,
                    )
                )
                program = PredictorPreservingA3ProgramV1(wire_mode, constant_rgb_cells=rows)
            else:
                copy_rows = tuple(
                    sorted(
                        (
                            CorrectedY1SupportCopyCellV1(
                                source_pair_ids[int(local_pair[index])],
                                int(scorer_row[index]),
                                int(scorer_col[index]),
                            )
                            for index in chosen
                        ),
                        key=lambda row: row.address,
                    )
                )
                program = PredictorPreservingA3ProgramV1(wire_mode, corrected_y1_copy_cells=copy_rows)
            identity = _a_program_identity(program)
            programs.append(
                AProgramV1(
                    program_id=f"{source_tag}:{mode.value.lower()}:{count}:{identity[:12]}",
                    program_sha256=identity,
                    mode=mode,
                    row_count=count,
                    acquisition_y1_sha256=y1_sha,
                    program=program,
                    ranking_sha256=ranking_sha,
                )
            )
    return tuple(programs)


def _g8_program_identity(program: object) -> str:
    """Recompute G12's public program identity without importing a private helper."""

    try:
        runs = [
            {
                "source_pair_id": run.source_pair_id,
                "scorer_row": run.scorer_row,
                "scorer_col_start": run.scorer_col_start,
                "scorer_col_stop": run.scorer_col_stop,
                "semantic_class": run.semantic_class,
                "semantic_role": run.semantic_role.value,
                "rgb_u8": list(run.rgb_u8),
            }
            for run in program.runs
        ]
    except AttributeError as exc:
        raise TaskspaceG8A3N2AllocatorError("G8 program changed canonical run shape") from exc
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.same_class_realization_repair_program_identity.v1",
                "runs": runs,
            }
        )
    )


def _exact_scorer_plane_rgb(camera_frames: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact rational R image and its reviewed uint8 projection."""

    from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

    frames = np.asarray(camera_frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (874, 1164, 3):
        raise TaskspaceG8A3N2AllocatorError("exact R input changed camera uint8 ABI")
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    numerator_rows: list[np.ndarray] = []
    denominator: int | None = None
    for frame in frames:
        current, current_denominator = operator.apply_numerators(frame)
        if denominator is not None and current_denominator != denominator:
            raise TaskspaceG8A3N2AllocatorError("exact R denominator changed across n2")
        denominator = current_denominator
        numerator_rows.append(np.ascontiguousarray(current, dtype=np.int64))
    assert denominator is not None
    numerators = np.stack(numerator_rows)
    if denominator < 1 or np.any(numerators < 0):
        raise TaskspaceG8A3N2AllocatorError("exact R escaped its nonnegative rational domain")
    rounded = np.floor_divide(numerators + denominator // 2, denominator)
    if np.any(rounded > 255):
        raise TaskspaceG8A3N2AllocatorError("exact R uint8 projection escaped its channel domain")
    projected = np.ascontiguousarray(rounded, dtype=np.uint8)
    projected.setflags(write=False)
    return projected, {
        "schema": "tac.g14_exact_r_scorer_plane_rgb_custody.v1",
        "camera_frames_sha256": _array_sha256(frames),
        "rational_numerators_i64_sha256": _array_sha256(numerators),
        "rational_denominator": denominator,
        "rounding": "nonnegative_nearest_ties_up.v1",
        "projected_rgb_u8_sha256": _array_sha256(projected),
        "shape": list(projected.shape),
        "dense_rgb_serialized": False,
    }


@dataclass(frozen=True, slots=True)
class RealG14ContextV1:
    source_member: bytes
    source_runtime: bytes
    causal_surface: object
    predictor_surface: object
    pass_g: object
    pass_a: object
    primary_bundle: object
    exact_g: object
    exact_a: object
    exact_control_bundle: object
    target_frames: np.ndarray
    target_labels: np.ndarray
    target_poses: np.ndarray
    g8_target_custody: object
    custody: dict[str, Any]


def _build_real_g14_context(*, timeout_seconds: float) -> RealG14ContextV1:
    """Decode P once, load the real n2 target, and compile both frozen controls."""

    import materialize_taskspace_pga_n2_receipt as materializer
    import measure_taskspace_pga_n2_macos_cpu as measurement

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.witness_dsl.bounded_target_g_encoder import (
        FrozenTargetSliceCustodyV1,
        compile_bounded_target_g_v2,
    )
    from tac.witness_dsl.ep725_levelset_predictor_adapter import (
        EP725_RUNTIME_BYTES,
        EP725_RUNTIME_SHA256,
        decode_ep725_counted_member_ephemeral_surface,
        inspect_ep725_source,
    )
    from tac.witness_dsl.predictor_preserving_coupled_preimage import (
        PredictorCameraPairSurfaceV1,
        PredictorPreservingA3Mode,
        PredictorPreservingA3ProgramV1,
        compile_predictor_preserving_a3,
    )
    from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
        overlay_g_on_predictor_camera_y1,
    )
    from tac.witness_dsl.taskspace_pass_conditional_a import compile_pass_conditional_a
    from tac.witness_dsl.taskspace_pass_semantic_g import compile_pass_semantic_g_envelope
    from tac.witness_dsl.taskspace_same_class_realization_repair import (
        EncoderOnlyExactTargetLabelCustodyV1,
    )
    from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

    source = inspect_ep725_source()
    if len(source.runtime) != EP725_RUNTIME_BYTES or _sha256(source.runtime) != EP725_RUNTIME_SHA256:
        raise TaskspaceG8A3N2AllocatorError("explicit ep725 runtime custody changed")
    causal = decode_ep725_counted_member_ephemeral_surface(
        source.member,
        shipped_runtime=source.runtime,
        pair_count=PAIR_COUNT,
        timeout_seconds=timeout_seconds,
    )
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(causal.ephemeral_surface)
    pass_g = compile_pass_semantic_g_envelope(
        predictor_section_payload=source.member,
        predictor_surface=predictor_surface,
    )
    pass_program = PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1)
    pass_a = compile_pass_conditional_a(
        pass_program,
        predictor_surface=predictor_surface,
        pass_g=pass_g.decoded,
    )
    primary_bundle = TaskspaceSectionBundleV1(
        predictor_packet=source.member,
        generative_correction_packet=pass_g.envelope,
        coupled_preimage_packet=pass_a.packet,
    )

    target_frames, target_labels, target_poses, target_custody = measurement._load_target()
    target_cache_path, materializer_target_custody = materializer._load_target_cache_path()
    labels_again = np.ascontiguousarray(
        open_stored_npy_memmap(target_cache_path, "lstars")[:PAIR_COUNT],
        dtype=np.uint8,
    )
    if not np.array_equal(labels_again, target_labels):
        raise TaskspaceG8A3N2AllocatorError("materializer and scorer target-label views differ")
    legacy_target_custody = FrozenTargetSliceCustodyV1(
        cache_sha256=materializer.TARGET_CACHE_SHA256,
        member_name="lstars",
        source_pair_ids=causal.predictor_state.source_pair_ids,
        target_labels_sha256=_array_sha256(target_labels),
    )
    realization_profile, realization_custody = materializer._load_realization_profile()
    exact_g = compile_bounded_target_g_v2(
        causal.predictor_state,
        target_labels,
        target_custody=legacy_target_custody,
        realization_profile=realization_profile,
    )
    if not np.array_equal(exact_g.compiled.decoded.labels, target_labels):
        raise TaskspaceG8A3N2AllocatorError("exact-semantic diagnostic G changed target labels")
    exact_overlay = overlay_g_on_predictor_camera_y1(
        causal.frame1_camera,
        causal.predictor_state.labels,
        exact_g.compiled.decoded,
    )
    exact_a = compile_predictor_preserving_a3(
        pass_program,
        predictor_surface=predictor_surface,
        decoded_g=exact_g.compiled.decoded,
        corrected_y1_overlay=exact_overlay,
    )
    exact_control_bundle = TaskspaceSectionBundleV1(
        predictor_packet=source.member,
        generative_correction_packet=exact_g.compiled.packet,
        coupled_preimage_packet=exact_a.packet,
    )
    g8_target_custody = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256=measurement.TARGET_CACHE_SHA256,
        source_member_name="lstars[:2]",
        source_member_sha256=_array_sha256(target_labels),
        source_pair_ids=causal.predictor_state.source_pair_ids,
        target_labels_sha256=_array_sha256(target_labels),
        target_labels=target_labels,
    )
    return RealG14ContextV1(
        source_member=source.member,
        source_runtime=source.runtime,
        causal_surface=causal,
        predictor_surface=predictor_surface,
        pass_g=pass_g,
        pass_a=pass_a,
        primary_bundle=primary_bundle,
        exact_g=exact_g,
        exact_a=exact_a,
        exact_control_bundle=exact_control_bundle,
        target_frames=target_frames,
        target_labels=target_labels,
        target_poses=target_poses,
        g8_target_custody=g8_target_custody,
        custody={
            "directory_owned_counted_p": {
                "bytes": len(source.member),
                "sha256": _sha256(source.member),
            },
            "explicit_runtime": {
                "bytes": len(source.runtime),
                "sha256": _sha256(source.runtime),
            },
            "causal_p_receipt_sha256": causal.causal_receipt.receipt_sha256,
            "pass_semantic_g_envelope": {
                "mode": pass_g.mode.value,
                "bytes": len(pass_g.envelope),
                "sha256": _sha256(pass_g.envelope),
                "nonempty": True,
                "source_bound": True,
            },
            "pass_conditional_a": {
                "bytes": len(pass_a.packet),
                "sha256": _sha256(pass_a.packet),
                "source_binding_sha256": pass_a.source_binding.binding_sha256,
            },
            "target": target_custody,
            "materializer_target": materializer_target_custody,
            "realization_profile": realization_custody,
            "dense_encoder_evidence_serialized": False,
        },
    )


@dataclass(slots=True)
class _CachedDecodedFramesV1:
    frames: np.ndarray
    output_sha256: str
    output_nbytes: int


class _EphemeralDecodedFrameCacheV1:
    """Bounded in-memory join between G7 receiver and scorer callbacks."""

    def __init__(self) -> None:
        self._request_to_output: dict[tuple[str, str], str] = {}
        self._frames: dict[str, _CachedDecodedFramesV1] = {}

    def record(self, request: object, frames: np.ndarray) -> _CachedDecodedFramesV1:
        immutable = np.ascontiguousarray(frames, dtype=np.uint8).copy()
        immutable.setflags(write=False)
        output_sha256 = _array_sha256(immutable)
        cached = self._frames.get(output_sha256)
        if cached is None:
            cached = _CachedDecodedFramesV1(immutable, output_sha256, immutable.nbytes)
            self._frames[output_sha256] = cached
        elif not np.array_equal(cached.frames, immutable):
            raise TaskspaceG8A3N2AllocatorError("decoded-output SHA collision")
        key = (request.stage_id, request.archive_sha256)
        prior = self._request_to_output.get(key)
        if prior is not None and prior != output_sha256:
            raise TaskspaceG8A3N2AllocatorError("receiver replay changed decoded output")
        self._request_to_output[key] = output_sha256
        return cached

    def consume(self, request: object) -> np.ndarray:
        key = (request.stage_id, request.archive_sha256)
        output_sha256 = self._request_to_output.get(key)
        if output_sha256 != request.decoded_output_sha256:
            raise TaskspaceG8A3N2AllocatorError("scorer request is not joined to receiver output")
        cached = self._frames.get(output_sha256)
        if cached is None or cached.output_nbytes != request.decoded_output_nbytes:
            raise TaskspaceG8A3N2AllocatorError("decoded-output cache byte identity changed")
        result = cached.frames
        for stage_key in tuple(self._request_to_output):
            if stage_key[0] == request.stage_id:
                del self._request_to_output[stage_key]
        live = set(self._request_to_output.values())
        for digest in tuple(self._frames):
            if digest not in live:
                del self._frames[digest]
        return result


class RealCachedCausalPReceiverV1:
    """G10 receiver callback with explicit per-archive counted-P identity."""

    def __init__(self, context: RealG14ContextV1) -> None:
        from tac.witness_dsl.taskspace_monolithic_pga_receiver import (
            receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface,
        )

        self._context = context
        self._receive = receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface
        self.cache = _EphemeralDecodedFrameCacheV1()
        self.calls = 0

    def __call__(self, request: object) -> object:
        from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceReceiverReceiptV1

        decoded = self._receive(
            request.archive_bytes,
            causal_surface=self._context.causal_surface,
            expected_encoding=request.encoding,
            expected_archive_sha256=request.archive_sha256,
            expected_member_sha256=request.member_sha256,
        )
        receipt = decoded.receipt
        if (
            receipt.sections[0].payload_sha256 != _sha256(self._context.source_member)
            or receipt.sections[0].byte_length != len(self._context.source_member)
            or receipt.predictor_causal_decode_receipt.counted_member_sha256 != receipt.sections[0].payload_sha256
            or receipt.exact_counted_p_reopened_and_matched is not True
        ):
            raise TaskspaceG8A3N2AllocatorError("cached causal P was not revalidated against this archive")
        cached = self.cache.record(request, decoded.chronological_camera_frames)
        self.calls += 1
        return TaskspaceReceiverReceiptV1(
            stage_id=request.stage_id,
            encoding=request.encoding,
            archive_sha256=request.archive_sha256,
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=cached.output_sha256,
            decoded_output_nbytes=cached.output_nbytes,
            receiver_receipt_payload=receipt.to_receipt_bytes(),
        )


@dataclass(slots=True)
class _ScorerObservationV1:
    camera_y1_sha256: str
    candidate_labels: np.ndarray
    candidate_labels_sha256: str
    candidate_pose6_sha256: str
    candidate_scorer_rgb: np.ndarray
    candidate_scorer_rgb_custody: dict[str, Any]
    candidate_forward_receipt: dict[str, Any]
    candidate_forward_receipt_sha256: str
    per_pair_d_seg: tuple[float, ...]
    per_pair_d_pose: tuple[float, ...]
    d_seg: float
    d_pose: float


class FrozenN2ScorerSessionV1:
    """One deterministic CPU scorer/model load shared by every atomic branch."""

    def __init__(self, context: RealG14ContextV1, receiver: RealCachedCausalPReceiverV1) -> None:
        import measure_taskspace_pga_n2_macos_cpu as measurement
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise TaskspaceG8A3N2AllocatorError("Torch interop threads were already fixed above one") from None
        torch.use_deterministic_algorithms(True)
        self._torch = torch
        self._context = context
        self._receiver = receiver
        self.runtime_environment_custody = _runtime_environment_custody()
        self.runtime_environment_custody_sha256 = _sha256(_canonical_json(self.runtime_environment_custody))
        self._scorer_custody = measurement._verify_small_scorer_custody()
        self.frozen_scorer_sha256 = _sha256(_canonical_json(self._scorer_custody))
        self._model = DistortionNet().eval().to(torch.device("cpu"))
        self._model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        target_tensor = torch.from_numpy(np.ascontiguousarray(context.target_frames).copy())
        with torch.inference_mode():
            target_pose_a, target_seg_a = self._model(target_tensor)
            target_pose_b, target_seg_b = self._model(target_tensor)
        if any(not torch.equal(target_pose_a[key], target_pose_b[key]) for key in target_pose_a) or not torch.equal(
            target_seg_a,
            target_seg_b,
        ):
            raise TaskspaceG8A3N2AllocatorError("frozen scorer changed on target double forward")
        self._target_pose6 = target_pose_a["pose"][..., :6]
        self._target_argmax = target_seg_a.argmax(dim=1)
        target_argmax = np.ascontiguousarray(self._target_argmax.cpu().numpy(), dtype=np.uint8)
        if not np.array_equal(target_argmax, context.target_labels):
            raise TaskspaceG8A3N2AllocatorError("fresh SegNet target differs from frozen labels")
        target_pose = np.ascontiguousarray(self._target_pose6.cpu().numpy())
        pose_cache_max_abs = float(np.max(np.abs(target_pose.astype(np.float64) - context.target_poses)))
        pose_cache_scale = max(1.0, float(np.max(np.abs(context.target_poses))))
        pose_cache_atol = 2.0 * float(np.finfo(np.float32).eps) * pose_cache_scale
        if pose_cache_max_abs > pose_cache_atol:
            raise TaskspaceG8A3N2AllocatorError("fresh PoseNet target differs beyond two scaled fp32 eps")
        self.target_scorer_rgb, self.target_scorer_rgb_custody = _exact_scorer_plane_rgb(context.target_frames[:, 1])
        self.target_forward_receipt = {
            "schema": "tac.g14_frozen_target_forward.v1",
            "axis": AXIS,
            "frozen_scorer_sha256": self.frozen_scorer_sha256,
            "labels_u8_sha256": _array_sha256(target_argmax),
            "pose6_f32_sha256": _array_sha256(target_pose),
            "pose_cache_max_abs": pose_cache_max_abs,
            "pose_cache_atol": pose_cache_atol,
            "scorer_plane_rgb": self.target_scorer_rgb_custody,
            "double_forward_exact": True,
            "runtime_environment_custody_sha256": self.runtime_environment_custody_sha256,
            "torch_module_version": str(torch.__version__),
            "numpy_module_version": np.__version__,
            "seed": SEED,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        self.target_forward_receipt_sha256 = _sha256(_canonical_json(self.target_forward_receipt))
        self._observations: dict[tuple[str, str], _ScorerObservationV1] = {}
        self.calls = 0
        self.distinct_candidate_double_forwards = 0

    def _observe(self, request: object, frames: np.ndarray) -> _ScorerObservationV1:
        torch = self._torch
        candidate_tensor = torch.from_numpy(np.ascontiguousarray(frames).copy())
        with torch.inference_mode():
            pose_a, seg_a = self._model(candidate_tensor)
            pose_b, seg_b = self._model(candidate_tensor)
        if any(not torch.equal(pose_a[key], pose_b[key]) for key in pose_a) or not torch.equal(seg_a, seg_b):
            raise TaskspaceG8A3N2AllocatorError("frozen scorer changed on candidate double forward")
        candidate_pose6 = pose_a["pose"][..., :6]
        candidate_argmax = seg_a.argmax(dim=1)
        per_pair_pose = (candidate_pose6 - self._target_pose6).pow(2).mean(dim=1)
        per_pair_seg = (candidate_argmax != self._target_argmax).float().mean(dim=(1, 2))
        d_pose = float(per_pair_pose.mean().item())
        d_seg = float(per_pair_seg.mean().item())
        per_pair_d_seg_values = tuple(float(value) for value in per_pair_seg.cpu().tolist())
        per_pair_d_pose_values = tuple(float(value) for value in per_pair_pose.cpu().tolist())
        if not math.isfinite(d_pose) or not math.isfinite(d_seg) or not 0.0 <= d_seg <= 1.0:
            raise TaskspaceG8A3N2AllocatorError("frozen scorer emitted invalid component distances")
        if (
            _float32_pair_mean(per_pair_d_seg_values, field="per_pair_d_seg", bounded_one=True) != d_seg
            or _float32_pair_mean(per_pair_d_pose_values, field="per_pair_d_pose", bounded_one=False) != d_pose
        ):
            raise TaskspaceG8A3N2AllocatorError("frozen scorer aggregate differs from its per-pair evidence")
        labels = np.ascontiguousarray(candidate_argmax.cpu().numpy(), dtype=np.uint8)
        labels.setflags(write=False)
        pose = np.ascontiguousarray(candidate_pose6.cpu().numpy())
        scorer_rgb, scorer_rgb_custody = _exact_scorer_plane_rgb(frames[:, 1])
        forward = {
            "schema": "tac.g14_frozen_candidate_forward.v1",
            "axis": AXIS,
            "archive_sha256": request.archive_sha256,
            "member_sha256": request.member_sha256,
            "decoded_output_sha256": request.decoded_output_sha256,
            "camera_y1_sha256": _array_sha256(frames[:, 1]),
            "candidate_labels_u8_sha256": _array_sha256(labels),
            "candidate_pose6_f32_sha256": _array_sha256(pose),
            "scorer_plane_rgb": scorer_rgb_custody,
            "per_pair_d_seg": list(per_pair_d_seg_values),
            "per_pair_d_pose": list(per_pair_d_pose_values),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "frozen_scorer_sha256": self.frozen_scorer_sha256,
            "target_forward_receipt_sha256": self.target_forward_receipt_sha256,
            "runtime_environment_custody_sha256": self.runtime_environment_custody_sha256,
            "double_forward_exact": True,
            "research_only": True,
        }
        self.distinct_candidate_double_forwards += 1
        return _ScorerObservationV1(
            camera_y1_sha256=forward["camera_y1_sha256"],
            candidate_labels=labels,
            candidate_labels_sha256=forward["candidate_labels_u8_sha256"],
            candidate_pose6_sha256=forward["candidate_pose6_f32_sha256"],
            candidate_scorer_rgb=scorer_rgb,
            candidate_scorer_rgb_custody=scorer_rgb_custody,
            candidate_forward_receipt=forward,
            candidate_forward_receipt_sha256=_sha256(_canonical_json(forward)),
            per_pair_d_seg=per_pair_d_seg_values,
            per_pair_d_pose=per_pair_d_pose_values,
            d_seg=d_seg,
            d_pose=d_pose,
        )

    def __call__(self, request: object) -> object:
        from tac.witness_dsl.taskspace_whole_archive_allocator import (
            TaskspaceRealizedMeasurementReceiptV1,
        )

        frames = self._receiver.cache.consume(request)
        key = (request.archive_sha256, request.decoded_output_sha256)
        observation = self._observations.get(key)
        cache_hit = observation is not None
        if observation is None:
            observation = self._observe(request, frames)
            self._observations[key] = observation
        elif observation.camera_y1_sha256 != _array_sha256(frames[:, 1]):
            raise TaskspaceG8A3N2AllocatorError("cached scorer observation differs from decoded Y1")
        receipt = {
            "schema": "tac.g14_frozen_measurement_callback.v1",
            "axis": AXIS,
            "stage_id": request.stage_id,
            "archive_sha256": request.archive_sha256,
            "member_sha256": request.member_sha256,
            "decoded_output_sha256": request.decoded_output_sha256,
            "receiver_receipt_sha256": request.receiver_receipt_sha256,
            "candidate_forward_receipt_sha256": observation.candidate_forward_receipt_sha256,
            "candidate_forward_cache_hit": cache_hit,
            "d_seg": observation.d_seg,
            "d_pose": observation.d_pose,
            "research_only": True,
        }
        self.calls += 1
        return TaskspaceRealizedMeasurementReceiptV1(
            stage_id=request.stage_id,
            archive_sha256=request.archive_sha256,
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=request.decoded_output_sha256,
            decoded_output_nbytes=request.decoded_output_nbytes,
            receiver_receipt_sha256=request.receiver_receipt_sha256,
            d_seg=observation.d_seg,
            d_pose=observation.d_pose,
            measurement_receipt_payload=_canonical_json(receipt),
        )

    def observation_for_state(self, state: object) -> _ScorerObservationV1:
        key = (
            state.measurement_receipt.archive_sha256,
            state.measurement_receipt.decoded_output_sha256,
        )
        observation = self._observations.get(key)
        if observation is None:
            raise TaskspaceG8A3N2AllocatorError("whole-object state lost its scorer observation")
        return observation


class RealTaskspaceG8A3BackendV1:
    """Real ep725/G10/G12/G7 backend; every treatment is one singleton."""

    def __init__(self, *, timeout_seconds: float, palette_bounds: tuple[int, ...]) -> None:
        self.context = _build_real_g14_context(timeout_seconds=timeout_seconds)
        self.palette_bounds = palette_bounds
        self.receiver = RealCachedCausalPReceiverV1(self.context)
        self.scorer = FrozenN2ScorerSessionV1(self.context, self.receiver)
        self._primary: WholeObjectMeasurementV1 | None = None
        self._exact_control: WholeObjectMeasurementV1 | None = None
        self._compiled_g8: dict[str, object] = {}

    def _allocate_state(self, bundle: object, *, label: str) -> object:
        from tac.witness_dsl.dynamic_frontier_target import (
            load_dynamic_frontier_target,
            verify_dynamic_frontier_target_snapshot,
        )
        from tac.witness_dsl.taskspace_whole_archive_allocator import (
            TaskspaceWholeArchiveProposalV1,
            allocate_taskspace_whole_archive,
        )

        pointer = load_dynamic_frontier_target(repo_root=REPO)
        verify_dynamic_frontier_target_snapshot(pointer)
        if bundle.bundle_sha256 == self.context.primary_bundle.bundle_sha256:
            proposals: tuple[object, ...] = ()
        else:
            proposal = TaskspaceWholeArchiveProposalV1(
                proposal_id=f"g14_{_sha256(label.encode('ascii'))[:20]}",
                transform=lambda _before, exact=bundle: exact,
            )
            proposals = (proposal,)
        allocation = allocate_taskspace_whole_archive(
            self.context.primary_bundle,
            proposals,
            frontier_snapshot=pointer,
            receiver_callback=self.receiver,
            measurement_callback=self.scorer,
        )
        verify_dynamic_frontier_target_snapshot(pointer)
        state = allocation.baseline_state if not proposals else allocation.proposal_audits[0].trial_state
        if state.bundle.bundle_sha256 != bundle.bundle_sha256:
            raise TaskspaceG8A3N2AllocatorError("singleton allocator returned a different trial bundle")
        return state

    def _row_from_state(
        self,
        state: object,
        *,
        label: str,
        g8_program_sha256: str | None,
        a_program_sha256: str,
        a_mode: AModeV1,
        a_row_count: int,
        compiled_a: object,
    ) -> WholeObjectMeasurementV1:
        from tac.witness_dsl.taskspace_outer_archive_codec import OuterArchiveEncoding

        observation = self.scorer.observation_for_state(state)
        build = state.archive_build
        selected_encoding: Literal["STORE", "DEFLATE"] = (
            "STORE" if build.selected.encoding is OuterArchiveEncoding.STORED else "DEFLATE"
        )
        if build.selected.archive_bytes not in {
            build.stored.archive_bytes,
            build.deflated.archive_bytes,
        }:
            raise TaskspaceG8A3N2AllocatorError("selected archive escaped STORE/DEFLATE builds")
        return WholeObjectMeasurementV1(
            measurement_id=f"g14:{label}:{state.bundle.bundle_sha256[:16]}",
            baseline_bundle_sha256=self.context.primary_bundle.bundle_sha256,
            bundle_sha256=state.bundle.bundle_sha256,
            g8_program_sha256=g8_program_sha256,
            a_program_sha256=a_program_sha256,
            a_packet_sha256=_sha256(compiled_a.packet),
            a_source_binding_sha256=compiled_a.source_binding.binding_sha256,
            a_mode=a_mode,
            a_row_count=a_row_count,
            raw_section_bytes=state.raw_section_nbytes,
            member_bytes=state.member_nbytes,
            member_sha256=build.selected.member_sha256,
            stored_archive_bytes=build.stored.archive_nbytes,
            stored_archive_sha256=build.stored.archive_sha256,
            deflated_archive_bytes=build.deflated.archive_nbytes,
            deflated_archive_sha256=build.deflated.archive_sha256,
            selected_encoding=selected_encoding,
            selected_archive_bytes=build.selected.archive_nbytes,
            selected_archive_sha256=build.selected.archive_sha256,
            selected_archive_payload=build.selected.archive_bytes,
            decoded_output_sha256=state.measurement_receipt.decoded_output_sha256,
            receiver_receipt_sha256=state.selected_receiver_receipt.receiver_receipt_sha256,
            camera_y1_sha256=observation.camera_y1_sha256,
            candidate_seg_labels_sha256=observation.candidate_labels_sha256,
            scorer_evidence=BoundedScorerEvidenceV1(
                measurement_receipt_sha256=state.measurement_receipt.measurement_receipt_sha256,
                candidate_forward_receipt_sha256=observation.candidate_forward_receipt_sha256,
                candidate_pose6_sha256=observation.candidate_pose6_sha256,
                per_pair_d_seg=observation.per_pair_d_seg,
                per_pair_d_pose=observation.per_pair_d_pose,
                sample_count=PAIR_COUNT,
                frozen_scorer_sha256=self.scorer.frozen_scorer_sha256,
                target_forward_receipt_sha256=self.scorer.target_forward_receipt_sha256,
            ),
            d_seg=state.measurement_receipt.d_seg,
            d_pose=state.measurement_receipt.d_pose,
        )

    def _measure_bundle(
        self,
        bundle: object,
        *,
        label: str,
        g8_program_sha256: str | None,
        a_program_sha256: str,
        a_mode: AModeV1,
        a_row_count: int,
        compiled_a: object,
    ) -> WholeObjectMeasurementV1:
        state = self._allocate_state(bundle, label=label)
        return self._row_from_state(
            state,
            label=label,
            g8_program_sha256=g8_program_sha256,
            a_program_sha256=a_program_sha256,
            a_mode=a_mode,
            a_row_count=a_row_count,
            compiled_a=compiled_a,
        )

    def measure_primary_baseline(self) -> WholeObjectMeasurementV1:
        if self._primary is None:
            pass_identity = _a_program_identity(self.context.pass_a.program)
            self._primary = self._measure_bundle(
                self.context.primary_bundle,
                label="g0_pass",
                g8_program_sha256=None,
                a_program_sha256=pass_identity,
                a_mode=AModeV1.PASS_A_V1,
                a_row_count=0,
                compiled_a=self.context.pass_a,
            )
        return self._primary

    def measure_exact_semantic_control(self) -> WholeObjectMeasurementV1:
        if self._exact_control is None:
            self._exact_control = self._measure_bundle(
                self.context.exact_control_bundle,
                label="exact_semantic_g_diagnostic",
                g8_program_sha256=None,
                a_program_sha256=_a_program_identity(self.context.exact_a.program),
                a_mode=AModeV1.PASS_A_V1,
                a_row_count=0,
                compiled_a=self.context.exact_a,
            )
        return self._exact_control

    def acquire_g8(self) -> tuple[object, dict[str, Any]]:
        from tac.witness_dsl.taskspace_same_class_realization_encoder import (
            N2_STAGE_ABLATION_PRIORITY_AXIS,
            N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256,
            N2_STAGE_ABLATION_PRIORITY_SCOPE,
            EncoderOnlySameClassRealizationEvidenceV1,
            SameClassRealizationAcquisitionPlanV1,
            SameClassRealizationBaseInterpretationV1,
            acquire_same_class_realization_repair_programs,
        )

        baseline = self.measure_primary_baseline()
        baseline_state = self._allocate_state(self.context.primary_bundle, label="g12_evidence_baseline")
        observation = self.scorer.observation_for_state(baseline_state)
        if (
            baseline.selected_archive_sha256 != baseline_state.archive_build.selected.archive_sha256
            or baseline.camera_y1_sha256 != observation.camera_y1_sha256
            or baseline.candidate_seg_labels_sha256 != observation.candidate_labels_sha256
        ):
            raise TaskspaceG8A3N2AllocatorError("G12 evidence baseline differs from persisted production baseline")
        pass_decoded = self.context.pass_g.decoded
        evidence = EncoderOnlySameClassRealizationEvidenceV1(
            source_pair_ids=self.context.causal_surface.predictor_state.source_pair_ids,
            base_interpretation=(SameClassRealizationBaseInterpretationV1.PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1),
            base_semantic_binding_sha256=pass_decoded.repair_surface.g_semantic_binding_sha256,
            base_p_section_sha256=_sha256(self.context.source_member),
            base_p_section_bytes=len(self.context.source_member),
            base_g_section_sha256=None,
            base_g_section_bytes=None,
            base_camera_y1_sha256=_array_sha256(pass_decoded.conditional_camera_y1),
            frozen_scorer_sha256=self.scorer.frozen_scorer_sha256,
            candidate_forward_receipt_sha256=observation.candidate_forward_receipt_sha256,
            target_forward_receipt_sha256=self.scorer.target_forward_receipt_sha256,
            priority_ordering_receipt_sha256=N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256,
            priority_ordering_axis=N2_STAGE_ABLATION_PRIORITY_AXIS,
            priority_ordering_scope=N2_STAGE_ABLATION_PRIORITY_SCOPE,
            current_semantic_labels_sha256=_array_sha256(pass_decoded.semantic_labels),
            target_labels_sha256=_array_sha256(self.context.target_labels),
            realized_labels_sha256=observation.candidate_labels_sha256,
            candidate_scorer_rgb_sha256=_array_sha256(observation.candidate_scorer_rgb),
            target_scorer_rgb_sha256=_array_sha256(self.scorer.target_scorer_rgb),
            current_semantic_labels=pass_decoded.semantic_labels,
            target_labels=self.context.target_labels,
            realized_labels=observation.candidate_labels,
            candidate_scorer_rgb=observation.candidate_scorer_rgb,
            target_scorer_rgb=self.scorer.target_scorer_rgb,
        )
        telemetry = evidence.cell_partition_telemetry
        acquisition = acquire_same_class_realization_repair_programs(
            evidence,
            plan=SameClassRealizationAcquisitionPlanV1(
                palette_sizes_per_class=self.palette_bounds,
                geometric_prefix_ratios=(4,),
            ),
        )
        summary = {
            "schema": "tac.g14_real_g12_acquisition_summary.v1",
            "base_interpretation": evidence.base_interpretation.value,
            "evidence_binding_sha256": evidence.binding_sha256,
            "base_semantic_binding_sha256": evidence.base_semantic_binding_sha256,
            "base_p_section_sha256": evidence.base_p_section_sha256,
            "base_camera_y1_sha256": evidence.base_camera_y1_sha256,
            "pass_semantic_g_envelope_sha256": _sha256(self.context.pass_g.envelope),
            "candidate_forward_receipt_sha256": observation.candidate_forward_receipt_sha256,
            "target_forward_receipt_sha256": self.scorer.target_forward_receipt_sha256,
            "frozen_scorer_sha256": self.scorer.frozen_scorer_sha256,
            "candidate_scorer_plane_rgb": observation.candidate_scorer_rgb_custody,
            "target_scorer_plane_rgb": self.scorer.target_scorer_rgb_custody,
            "canonical_four_way_z_t_h_partition": telemetry.as_dict(),
            "closed_cell_count": telemetry.closed_cell_count,
            "realization_debt_cell_count": telemetry.realization_debt_cell_count,
            "topology_debt_cell_count": telemetry.topology_debt_cell_count,
            "fortunate_semantic_mismatch_cell_count": telemetry.fortunate_semantic_mismatch_cell_count,
            "prefix_cell_counts": list(acquisition.prefix_cell_counts),
            "proposal_count": len(acquisition.proposals),
            "unique_program_count": len(acquisition.unique_program_proposals),
            "duplicate_program_count": acquisition.duplicate_program_count,
            "palette_bounds": list(self.palette_bounds),
            "prefix_ratios": [4],
            "dense_labels_or_rgb_serialized": False,
            "scorer_invoked_for_h": True,
            "through_exact_r_rgb_custody": True,
            "research_only": True,
        }
        return acquisition, summary

    def freeze_g8_acquisition(
        self,
        acquisition: object,
        branches: tuple[G8BranchV1, ...],
    ) -> dict[str, Any]:
        proposal_by_id = {proposal.proposal_id: proposal for proposal in acquisition.proposals}
        rows: list[dict[str, Any]] = []
        for branch in branches:
            proposal = proposal_by_id.get(branch.proposal_id)
            if proposal is None or proposal.receipt.program_sha256 != branch.program_sha256:
                raise TaskspaceG8A3N2AllocatorError("screened G8 branch escaped its G12 acquisition")
            rows.append(
                {
                    "descriptor": branch.as_dict(),
                    "proposal_receipt_ascii": proposal.receipt.to_receipt_bytes().decode("ascii"),
                    "runs": [
                        {
                            "source_pair_id": run.source_pair_id,
                            "scorer_row": run.scorer_row,
                            "scorer_col_start": run.scorer_col_start,
                            "scorer_col_stop": run.scorer_col_stop,
                            "semantic_class": run.semantic_class,
                            "semantic_role": run.semantic_role.value,
                            "rgb_u8": list(run.rgb_u8),
                        }
                        for run in proposal.program.runs
                    ],
                }
            )
        return {
            "schema": "tac.g14_g12_typed_restore.v1",
            "g12_implementation": [
                {"path": path, "sha256": expected} for path, expected in G12_FROZEN_IMPLEMENTATION.items()
            ],
            "branches": rows,
            "scorer_backed_acquisition_must_not_recur_on_resume": True,
        }

    def restore_g8_acquisition(self, payload: dict[str, Any]) -> tuple[G8BranchV1, ...]:
        from tac.witness_dsl.taskspace_same_class_realization_encoder import (
            parse_same_class_realization_proposal_receipt,
        )
        from tac.witness_dsl.taskspace_same_class_realization_repair import (
            SameClassRealizationRepairProgramV1,
            SameClassRealizationRepairRunV1,
            SameClassSemanticRoleV1,
        )

        expected_implementation = [
            {"path": path, "sha256": expected} for path, expected in G12_FROZEN_IMPLEMENTATION.items()
        ]
        if (
            set(payload)
            != {
                "schema",
                "g12_implementation",
                "branches",
                "scorer_backed_acquisition_must_not_recur_on_resume",
            }
            or payload.get("schema") != "tac.g14_g12_typed_restore.v1"
        ):
            raise TaskspaceG8A3N2AllocatorError("G12 typed restore schema changed")
        if (
            payload.get("g12_implementation") != expected_implementation
            or payload.get("scorer_backed_acquisition_must_not_recur_on_resume") is not True
        ):
            raise TaskspaceG8A3N2AllocatorError("G12 typed restore implementation/truth changed")
        for relative, expected in G12_FROZEN_IMPLEMENTATION.items():
            if _sha256(_read_stable_regular(REPO / relative)) != expected:
                raise TaskspaceG8A3N2AllocatorError("G12 implementation drifted before typed restore")
        raw_rows = payload.get("branches")
        if type(raw_rows) is not list or not raw_rows:
            raise TaskspaceG8A3N2AllocatorError("G12 typed restore omitted branch programs")
        branches: list[G8BranchV1] = []
        for raw in raw_rows:
            if type(raw) is not dict or set(raw) != {"descriptor", "proposal_receipt_ascii", "runs"}:
                raise TaskspaceG8A3N2AllocatorError("G12 typed restore row changed")
            descriptor = raw["descriptor"]
            receipt_ascii = raw["proposal_receipt_ascii"]
            run_rows = raw["runs"]
            if type(descriptor) is not dict or type(receipt_ascii) is not str or type(run_rows) is not list:
                raise TaskspaceG8A3N2AllocatorError("G12 typed restore row has invalid exact types")
            try:
                receipt = parse_same_class_realization_proposal_receipt(receipt_ascii.encode("ascii"))
                runs = tuple(
                    SameClassRealizationRepairRunV1(
                        source_pair_id=row["source_pair_id"],
                        scorer_row=row["scorer_row"],
                        scorer_col_start=row["scorer_col_start"],
                        scorer_col_stop=row["scorer_col_stop"],
                        semantic_class=row["semantic_class"],
                        semantic_role=SameClassSemanticRoleV1(row["semantic_role"]),
                        rgb_u8=tuple(row["rgb_u8"]),
                    )
                    for row in run_rows
                )
                program = SameClassRealizationRepairProgramV1(runs=runs)
            except (KeyError, TypeError, ValueError, UnicodeEncodeError) as exc:
                raise TaskspaceG8A3N2AllocatorError("G12 typed repair program restore failed") from exc
            if _g8_program_identity(program) != receipt.program_sha256:
                raise TaskspaceG8A3N2AllocatorError("restored G8 program differs from frozen G12 receipt")
            expected_descriptor = {
                "proposal_id": receipt.proposal_id,
                "program_sha256": receipt.program_sha256,
                "family": receipt.family.value,
                "prefix_order": receipt.prefix_order.value,
                "palette_bound_per_class": receipt.palette_bound_per_class,
                "prefix_cell_count": receipt.requested_prefix_cell_count,
            }
            if descriptor != expected_descriptor:
                raise TaskspaceG8A3N2AllocatorError("restored G8 descriptor differs from parsed receipt")
            branches.append(
                G8BranchV1(
                    **expected_descriptor,
                    program=program,
                )
            )
        if len({branch.program_sha256 for branch in branches}) != len(branches):
            raise TaskspaceG8A3N2AllocatorError("typed G12 restore repeated a deduplicated program")
        return tuple(branches)

    def _compile_g8(self, branch: G8BranchV1) -> object:
        from tac.witness_dsl.taskspace_pass_semantic_g import compile_pass_semantic_g_envelope

        if _g8_program_identity(branch.program) != branch.program_sha256:
            raise TaskspaceG8A3N2AllocatorError("G8 branch program differs from its acquisition digest")
        compiled = self._compiled_g8.get(branch.program_sha256)
        if compiled is None:
            compiled = compile_pass_semantic_g_envelope(
                predictor_section_payload=self.context.source_member,
                predictor_surface=self.context.predictor_surface,
                repair_program=branch.program,
                target_custody=self.context.g8_target_custody,
            )
            if compiled.mode.value != "PASS_THEN_G8_V1":
                raise TaskspaceG8A3N2AllocatorError("G8 compile did not produce explicit PASS_THEN_G8")
            self._compiled_g8[branch.program_sha256] = compiled
        return compiled

    def acquire_g0_a(self, row_counts: tuple[int, ...]) -> tuple[AProgramV1, ...]:
        return _a_programs_for_surface(
            source_pair_ids=self.context.causal_surface.predictor_state.source_pair_ids,
            camera_p0=self.context.predictor_surface.camera_p0,
            conditional_y1=self.context.pass_g.decoded.conditional_camera_y1,
            target_y0=self.context.target_frames[:, 0],
            row_counts=row_counts,
            source_tag="g0",
        )

    def acquire_a(self, branch: G8BranchV1, row_counts: tuple[int, ...]) -> tuple[AProgramV1, ...]:
        compiled_g = self._compile_g8(branch)
        return _a_programs_for_surface(
            source_pair_ids=self.context.causal_surface.predictor_state.source_pair_ids,
            camera_p0=self.context.predictor_surface.camera_p0,
            conditional_y1=compiled_g.decoded.conditional_camera_y1,
            target_y0=self.context.target_frames[:, 0],
            row_counts=row_counts,
            source_tag=f"g8{branch.program_sha256[:12]}",
        )

    def _compile_a(self, pass_g: object, program: AProgramV1) -> object:
        from tac.witness_dsl.predictor_preserving_coupled_preimage import PredictorPreservingA3ProgramV1
        from tac.witness_dsl.taskspace_pass_conditional_a import compile_pass_conditional_a

        if type(program.program) is not PredictorPreservingA3ProgramV1:
            raise TaskspaceG8A3N2AllocatorError(
                "additional counted-A provider requires a separately reviewed compiler extension; emulation forbidden"
            )
        if _a_program_identity(program.program) != program.program_sha256:
            raise TaskspaceG8A3N2AllocatorError("high-level A program differs from its descriptor digest")
        return compile_pass_conditional_a(
            program.program,
            predictor_surface=self.context.predictor_surface,
            pass_g=pass_g.decoded,
        )

    def measure_g8_pass_a(self, branch: G8BranchV1) -> WholeObjectMeasurementV1:
        from tac.witness_dsl.predictor_preserving_coupled_preimage import (
            PredictorPreservingA3Mode,
            PredictorPreservingA3ProgramV1,
        )
        from tac.witness_dsl.taskspace_pass_conditional_a import compile_pass_conditional_a
        from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

        compiled_g = self._compile_g8(branch)
        pass_program = PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1)
        compiled_a = compile_pass_conditional_a(
            pass_program,
            predictor_surface=self.context.predictor_surface,
            pass_g=compiled_g.decoded,
        )
        bundle = TaskspaceSectionBundleV1(
            predictor_packet=self.context.source_member,
            generative_correction_packet=compiled_g.envelope,
            coupled_preimage_packet=compiled_a.packet,
        )
        return self._measure_bundle(
            bundle,
            label=f"g8_pass_{branch.program_sha256[:16]}",
            g8_program_sha256=branch.program_sha256,
            a_program_sha256=_a_program_identity(pass_program),
            a_mode=AModeV1.PASS_A_V1,
            a_row_count=0,
            compiled_a=compiled_a,
        )

    def measure_g8_a(self, branch: G8BranchV1, program: AProgramV1) -> WholeObjectMeasurementV1:
        from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

        compiled_g = self._compile_g8(branch)
        compiled_a = self._compile_a(compiled_g, program)
        bundle = TaskspaceSectionBundleV1(
            predictor_packet=self.context.source_member,
            generative_correction_packet=compiled_g.envelope,
            coupled_preimage_packet=compiled_a.packet,
        )
        return self._measure_bundle(
            bundle,
            label=f"g8a_{branch.program_sha256[:10]}_{program.program_sha256[:10]}",
            g8_program_sha256=branch.program_sha256,
            a_program_sha256=program.program_sha256,
            a_mode=program.mode,
            a_row_count=program.row_count,
            compiled_a=compiled_a,
        )

    def measure_g0_a(self, program: AProgramV1) -> WholeObjectMeasurementV1:
        from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

        compiled_a = self._compile_a(self.context.pass_g, program)
        bundle = TaskspaceSectionBundleV1(
            predictor_packet=self.context.source_member,
            generative_correction_packet=self.context.pass_g.envelope,
            coupled_preimage_packet=compiled_a.packet,
        )
        return self._measure_bundle(
            bundle,
            label=f"g0a_{program.program_sha256[:16]}",
            g8_program_sha256=None,
            a_program_sha256=program.program_sha256,
            a_mode=program.mode,
            a_row_count=program.row_count,
            compiled_a=compiled_a,
        )


def _reviewed_command(args: argparse.Namespace) -> list[str]:
    command = [
        os.fspath(REPO / ".venv/bin/python"),
        os.fspath(Path(__file__).resolve()),
        "--execute-reviewed",
        "--g8-prefixes",
        args.g8_prefixes,
        "--a-prefixes",
        ",".join(str(value) for value in args.a_prefixes),
        "--palette-bounds",
        ",".join(str(value) for value in args.palette_bounds),
        "--seed",
        str(args.seed),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.resume_from is not None:
        command.extend(("--resume-from", os.fspath(args.resume_from)))
    else:
        run_dir = args.run_dir or (REPO / ".omx/runs/taskspace_g8_a3_n2_allocator" / "REVIEWED_RUN_DIR_REQUIRED")
        command.extend(("--run-dir", os.fspath(run_dir)))
    return command


def _pointer_dict(snapshot: object) -> dict[str, Any]:
    try:
        value = asdict(snapshot)
    except TypeError as exc:
        raise TaskspaceG8A3N2AllocatorError("dynamic pointer snapshot changed typed shape") from exc
    if type(value) is not dict:
        raise TaskspaceG8A3N2AllocatorError("dynamic pointer snapshot did not serialize as one object")
    return value


def _real_input_custody() -> dict[str, Any]:
    """Reopen immutable inputs without decoding P or loading Torch."""

    import measure_taskspace_pga_n2_macos_cpu as measurement

    from tac.witness_dsl.ep725_levelset_predictor_adapter import inspect_ep725_source

    source = inspect_ep725_source()
    try:
        target_metadata = measurement.TARGET_CACHE_PATH.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspaceG8A3N2AllocatorError("frozen target cache metadata is unavailable") from exc
    if not stat.S_ISREG(target_metadata.st_mode) or target_metadata.st_size != measurement.TARGET_CACHE_BYTES:
        raise TaskspaceG8A3N2AllocatorError("frozen target cache size/type custody changed")
    spine = _read_stable_regular(measurement.SPINE_PATH)
    if _sha256(spine) != measurement.SPINE_SHA256:
        raise TaskspaceG8A3N2AllocatorError("constructive target spine custody changed")
    return {
        "ep725": {
            "archive_bytes": len(source.archive),
            "archive_sha256": source.archive_sha256,
            "counted_member_bytes": len(source.member),
            "counted_member_sha256": source.member_sha256,
            "runtime_bytes": len(source.runtime),
            "runtime_sha256": source.runtime_sha256,
            "manifest_bytes": len(source.manifest_bytes),
            "manifest_sha256": source.manifest_sha256,
        },
        "target_cache": {
            "path": os.fspath(measurement.TARGET_CACHE_PATH),
            "bytes": target_metadata.st_size,
            "pinned_sha256": measurement.TARGET_CACHE_SHA256,
            "full_rehash_this_runner": False,
            "n2_member_sha256": {
                "labels": measurement.TARGET_LABELS_U8_SHA256,
                "poses": measurement.TARGET_POSES_F64_SHA256,
                "f0": measurement.TARGET_F0_U8_SHA256,
                "f1": measurement.TARGET_F1_U8_SHA256,
            },
            "constructive_spine_sha256": measurement.SPINE_SHA256,
        },
        "frozen_scorer": measurement._verify_small_scorer_custody(),
    }


def _runtime_environment_custody() -> dict[str, Any]:
    """Bind exact runtime/package identities without importing Torch."""

    import importlib.metadata
    import platform
    import zlib

    distributions: dict[str, str] = {}
    for logical_name, distribution_name in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("safetensors", "safetensors"),
    ):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TaskspaceG8A3N2AllocatorError(
                f"required runtime distribution is unavailable: {distribution_name}"
            ) from exc
        if type(version) is not str or not version or not version.isascii():
            raise TaskspaceG8A3N2AllocatorError(f"runtime distribution version is noncanonical: {distribution_name}")
        distributions[logical_name] = version
    if distributions["numpy"] != np.__version__:
        raise TaskspaceG8A3N2AllocatorError("NumPy module/distribution versions differ")
    cache_tag = sys.implementation.cache_tag
    if type(cache_tag) is not str or not cache_tag:
        raise TaskspaceG8A3N2AllocatorError("Python cache tag is unavailable")
    return {
        "schema": "tac.g14_runtime_environment_custody.v1",
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "full_version": sys.version,
            "cache_tag": cache_tag,
            "executable": os.path.realpath(sys.executable),
            "compiler": platform.python_compiler(),
        },
        "numpy_module_version": np.__version__,
        "distributions": distributions,
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "zlib": {
            "compile_version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
    }


def _stable_contract(args: argparse.Namespace, *, git_head_at_start: str) -> dict[str, Any]:
    _require_id(git_head_at_start, "git_head_at_start")
    return {
        "schema": "tac.taskspace_g8_a3_n2_allocator_stable_contract.v1",
        "lane_id": LANE_ID,
        "axis": AXIS,
        "git_head_at_start": git_head_at_start,
        "config": {
            "pair_count": PAIR_COUNT,
            "seed": args.seed,
            "g8_prefixes": args.g8_prefixes,
            "a_prefixes": list(args.a_prefixes),
            "palette_bounds": list(args.palette_bounds),
            "timeout_seconds": args.timeout_seconds,
        },
        "implementation_custody": _implementation_custody(),
        "frozen_g10": G10_FROZEN_IMPLEMENTATION,
        "frozen_g12": G12_FROZEN_IMPLEMENTATION,
        "runtime_environment_custody": _runtime_environment_custody(),
        "input_custody": _real_input_custody(),
        "execution_contract": {
            "scorer_model_load_count": 1,
            "target_double_forward": True,
            "distinct_candidate_double_forward": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "cached_causal_p_with_per_archive_identity": True,
            "dense_frames_ephemeral": True,
            "branch_checkpoint_before_next_branch": True,
            "g12_acquisition_restored_not_recomputed_after_stage_030": True,
            "a_acquisition_recomputed_but_write_once_equality_gated_before_measurement": True,
        },
        "lineage": {
            "public_archive_payload_reused": False,
            "video_derived_values_counted": True,
            "generic_receiver_code_counted_as_payload": False,
            "exact_semantic_g_is_diagnostic_only": True,
        },
    }


def _resume_start_git_head(run_dir: Path) -> str:
    root = _safe_run_directory(run_dir)
    manifest = _strict_json_object(_read_stable_regular(root / "manifest.json"), field="run manifest preflight")
    try:
        value = manifest["stable_contract"]["git_head_at_start"]
    except (KeyError, TypeError) as exc:
        raise TaskspaceG8A3N2AllocatorError("resume manifest omitted original git provenance") from exc
    if type(value) is not str or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise TaskspaceG8A3N2AllocatorError("resume manifest original git provenance is invalid")
    return value


def _run_reviewed_real(args: argparse.Namespace) -> bytes:
    """Run or resume the reviewed real-n2 experiment under immutable custody."""

    from tac.witness_dsl.dynamic_frontier_target import (
        load_dynamic_frontier_target,
        verify_dynamic_frontier_target_snapshot,
    )

    require_g10_production_surface()
    for relative, expected in G12_FROZEN_IMPLEMENTATION.items():
        if _sha256(_read_stable_regular(REPO / relative)) != expected:
            raise TaskspaceG8A3N2AllocatorError(f"frozen G12 implementation custody drifted: {relative}")
    start_pointer = load_dynamic_frontier_target(repo_root=REPO)
    verify_dynamic_frontier_target_snapshot(start_pointer)
    pointer_start = _pointer_dict(start_pointer)
    if args.resume_from is None:
        stable_contract = _stable_contract(args, git_head_at_start=_git_head())
        assert args.run_dir is not None
        store = AtomicRunStore.create(
            args.run_dir,
            stable_contract=stable_contract,
            pointer_start=pointer_start,
        )
        store.append_pointer_snapshot(pointer_start, label="start")
    else:
        git_head_at_start = _resume_start_git_head(args.resume_from)
        stable_contract = _stable_contract(args, git_head_at_start=git_head_at_start)
        store = AtomicRunStore.resume(
            args.resume_from,
            expected_stable_contract=stable_contract,
        )
        store.append_pointer_snapshot(pointer_start, label="resume")
        final_path = store.run_dir / "final_receipt.json"
        if final_path.exists():
            final_payload = _read_stable_regular(final_path)
            parsed = parse_final_receipt(final_payload)
            if (
                parsed.get("manifest_sha256") != _sha256(_canonical_json(store.manifest))
                or parsed.get("stable_contract") != stable_contract
            ):
                raise TaskspaceG8A3N2AllocatorError("completed resume receipt differs from manifest custody")
            return final_payload

    backend = RealTaskspaceG8A3BackendV1(
        timeout_seconds=args.timeout_seconds,
        palette_bounds=args.palette_bounds,
    )
    experiment = run_coupled_experiment(
        store=store,
        backend=backend,
        g8_prefix_mode=args.g8_prefixes,
        a_prefixes=args.a_prefixes,
    )
    latest_pointer = load_dynamic_frontier_target(repo_root=REPO)
    verify_dynamic_frontier_target_snapshot(latest_pointer)
    return finalize_receipt(
        store=store,
        experiment=experiment,
        pointer_latest=_pointer_dict(latest_pointer),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--print-reviewed-command", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--g8-prefixes", default="cheap")
    parser.add_argument("--a-prefixes", type=parse_a_prefixes, default=DEFAULT_A_PREFIXES)
    parser.add_argument("--palette-bounds", type=parse_palette_bounds, default=DEFAULT_PALETTE_BOUNDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.seed != SEED:
        parser.error(f"the reviewed deterministic seed is fixed at {SEED}")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be finite and positive")
    if args.resume_from is not None and args.run_dir is not None:
        parser.error("use exactly one of --run-dir or --resume-from")
    if args.print_reviewed_command:
        print(_canonical_json({"reviewed_command": _reviewed_command(args), "executed": False}).decode("ascii"))
        return 0
    if not args.execute_reviewed:
        parser.error("real execution defaults to refusal; use --print-reviewed-command for root review")
    if args.run_dir is None and args.resume_from is None:
        parser.error("reviewed execution requires --run-dir or --resume-from")
    try:
        receipt = _run_reviewed_real(args)
    except G10ProductionCompositionUnavailable as exc:
        sys.stderr.buffer.write(_g10_blocker(str(exc)))
        return 3
    sys.stdout.buffer.write(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
