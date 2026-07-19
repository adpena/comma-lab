# SPDX-License-Identifier: MIT
"""Deterministic, counted V10 cold-program compiler and reference receiver.

This module proves local structure only.  It does not load a scorer, launch a
trainer, claim a score, or promote an archive.  Every admitted video-derived
byte is owned by one frozen semantic route and is reopened by the public
receiver before an authoritative consumption receipt can exist.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import struct
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
from tac.witness_dsl.lawref import lawref_to_declaration, resolve
from tac.witness_dsl.typed_config import TypedWitnessConfig
from tac.witness_dsl.v10_production_receiver import (
    RECEIVER_CONTRACT_ID as FACTOR2_RECEIVER_CONTRACT_ID,
)

MAGIC = b"TACV10P\x00"
VERSION = 2
PREFIX = struct.Struct(">8sHI")
PROGRAM_SCHEMA = "v10_counted_program.v2"
PARSER_PROOF_SCHEMA = "v10_contiguous_byte_proof.v2"
RECEIPT_SCHEMA = "v10_receiver_consumption.v2"
CHECKPOINT_SCHEMA = "v10_receiver_checkpoint.v2"
COMPLETENESS_SCHEMA = "inverse_solve_completeness_manifest.v2"
ROUTE_REGISTRY_SCHEMA = "v10_route_registry.v2"
HANDLER_REGISTRY_SCHEMA = "v10_handler_registry.v2"
HANDLER_REGISTRY_VERSION = 2
FROZEN_FACTOR_IDS = ("1", "2", "3a", "3b", "4", "5", "6", "7", "8", "9", "10")
IMPLEMENTED_FACTOR_IDS = ("1", "2", "3a", "3b", "4", "5", "6", "7", "8", "9")
MISSING_FACTOR_IDS = ("10",)
QUOTIENT_BASE_FACTOR_IDS = ("1", "2", "3a", "3b", "4", "6", "7", "8", "9")

_SECTION_FIELDS = (
    "section_id",
    "factor_ids",
    "producer_id",
    "consumer_id",
    "encoding",
    "video_derived",
    "byte_length",
    "sha256",
    "apply_order",
    "owned_parameter_groups",
    "frozen_parameter_groups",
    "class_ids",
    "cell_ids",
    "depends_on",
    "quotient_base_factor_ids",
)
_HEADER_FIELDS = (
    "schema",
    "version",
    "typed_config_hash",
    "argv_sha256",
    "section_count",
    "implemented_factor_ids",
    "missing_factor_ids",
    "route_registry_schema",
    "route_registry_sha256",
    "handler_registry_schema",
    "handler_registry_version",
    "handler_registry_sha256",
    "sections",
)
_CHECKPOINT_FIELDS = (
    "schema",
    "program_sha256",
    "typed_config_hash",
    "next_section_index",
    "consumed_section_ids",
    "state_bytes_b64",
    "state_sha256",
)
_COMPLETENESS_FIELDS = (
    "factor_id",
    "term_id",
    "owner_task",
    "disposition",
    "derivation_ref",
    "build_sha",
    "compiled_config_hash",
    "consumer_id",
    "resume_schema_and_replay_ref",
    "measurement_receipt_sha256",
    "authority_axis",
    "interaction_receipts",
    "adoption_or_scoped_exclusion",
    "strict_certificate",
)
_FORBIDDEN_STATE_TOKENS = frozenset(
    {
        "forkhead",
        "forkheadsolve",
        "forkema",
        "forkemaclearance",
        "forkstate",
        "optimizerfork",
        "optimizerstate",
        "emafork",
        "emastate",
        "resumefrom",
        "resumelr",
        "resumelrwarmup",
    }
)


class V10Refusal(ValueError):
    """A fail-closed V10 structural refusal."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V10Refusal("value is not canonical-JSON encodable") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, field_name: str) -> str:
    if not _is_sha256(value):
        raise V10Refusal(f"{field_name} must be a lowercase sha256")
    return value


def _require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V10Refusal(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_exact_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise V10Refusal(f"{field_name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise V10Refusal(f"{field_name} must be >= {minimum}")
    return value


def _require_string_tuple(value: Any, field_name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise V10Refusal(f"{field_name} must be tuple[str, ...]")
    if nonempty and not value:
        raise V10Refusal(f"{field_name} cannot be empty")
    for item in value:
        _require_identifier(item, field_name)
    if len(set(value)) != len(value):
        raise V10Refusal(f"{field_name} contains duplicates")
    return value


def _require_string_list(value: Any, field_name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise V10Refusal(f"{field_name} must be list[str]")
    if nonempty and not value:
        raise V10Refusal(f"{field_name} cannot be empty")
    for item in value:
        _require_identifier(item, field_name)
    if len(set(value)) != len(value):
        raise V10Refusal(f"{field_name} contains duplicates")
    return value


def canonical_semantic_payload(value: Mapping[str, Any]) -> bytes:
    """Encode one non-empty semantic section body in the only admitted form."""

    if not isinstance(value, Mapping) or not value:
        raise V10Refusal("semantic payload must be a non-empty mapping")
    return _canonical_json(dict(value))


class InstructionKind(str, Enum):
    COUNTED_GENERATOR = "CountedGenerator"
    FACTOR2_INTEGER_SCORER_PLANE = "Factor2IntegerScorerPlane"
    FRAME0_POSE_SIX_CARRIER = "Frame0PoseSixCarrier"
    INIT_HEAD_SOLVE = "InitHeadSolve"
    SHARED_RESIZE_PREIMAGE = "SharedResizePreimage"
    RGB_YUV6_PROJECTION = "RgbYuv6Projection"
    BLIND_FILL_RATE_GRAMMAR = "BlindFillRateGrammar"
    QUOTIENT_RESIDUAL_T = "QuotientResidualT"
    FORK_HEAD_SOLVE = "ForkHeadSolve"
    FORK_EMA_CLEARANCE = "ForkEmaClearance"
    RESUME_LR_WARMUP = "ResumeLRWarmup"


@dataclass(frozen=True)
class RouteSpec:
    kind: InstructionKind
    section_id: str
    factor_ids: tuple[str, ...]
    producer_id: str
    consumer_id: str
    encoding: str
    owned_parameter_group: str
    semantic_fields: tuple[str, ...]

    def manifest_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "section_id": self.section_id,
            "factor_ids": list(self.factor_ids),
            "producer_id": self.producer_id,
            "consumer_id": self.consumer_id,
            "encoding": self.encoding,
            "video_derived": True,
            "owned_parameter_group": self.owned_parameter_group,
            "semantic_fields": list(self.semantic_fields),
        }


FROZEN_ROUTES = (
    RouteSpec(
        InstructionKind.COUNTED_GENERATOR,
        "counted_generator",
        ("1",),
        "typed_v10_compiler",
        "receiver.counted_generator",
        "counted_generator_v2",
        "generator_seed.parameters",
        ("frame0_rgb", "frame1_rgb", "seed_bytes"),
    ),
    RouteSpec(
        InstructionKind.FACTOR2_INTEGER_SCORER_PLANE,
        "factor2_integer_scorer_plane",
        ("2",),
        "production_archive_builder",
        "receiver.factor2_integer_scorer_plane",
        "factor2_integer_scorer_plane_v1",
        "factor2.scorer_plane",
        ("y_uint8", "camera_shape", "scorer_shape", "receiver_contract_id"),
    ),
    RouteSpec(
        InstructionKind.FRAME0_POSE_SIX_CARRIER,
        "frame0_pose_six_carrier",
        ("7", "8"),
        "frame0_pose_six_compiler",
        "receiver.frame0_pose_six_carrier",
        "frame0_pose_six_carrier_v1",
        "frame0_pose_six.parameters",
        ("frame0_delta", "pose_six"),
    ),
    RouteSpec(
        InstructionKind.INIT_HEAD_SOLVE,
        "init_head_solve",
        ("6",),
        "init_head_solver",
        "receiver.init_head_solve",
        "init_head_solve_v2",
        "init_head.parameters",
        ("head_bias",),
    ),
    RouteSpec(
        InstructionKind.SHARED_RESIZE_PREIMAGE,
        "shared_resize_preimage",
        ("3a", "3b"),
        "shared_resize_preimage_solver",
        "receiver.shared_resize_preimage",
        "shared_resize_preimage_v1",
        "shared_resize_preimage.parameters",
        ("fanout", "weights"),
    ),
    RouteSpec(
        InstructionKind.RGB_YUV6_PROJECTION,
        "rgb_yuv6_projection",
        ("4",),
        "rgb_yuv6_projector",
        "receiver.rgb_yuv6_projection",
        "rgb_yuv6_projection_v1",
        "rgb_yuv6_projection.parameters",
        ("rgb_bias", "BT601_YUV6"),
    ),
    RouteSpec(
        InstructionKind.BLIND_FILL_RATE_GRAMMAR,
        "blind_fill_rate_grammar",
        ("9",),
        "blind_fill_rate_compiler",
        "receiver.blind_fill_rate_grammar",
        "blind_fill_rate_grammar_v1",
        "blind_fill_rate.parameters",
        ("blind_indices", "fill_value", "rate_tokens"),
    ),
    RouteSpec(
        InstructionKind.QUOTIENT_RESIDUAL_T,
        "quotient_residual_T",
        ("5",),
        "quotient_residual_trainer",
        "receiver.quotient_residual_T",
        "quotient_residual_t_v2",
        "T.road.cell_0",
        ("updates",),
    ),
)
_ROUTE_BY_ENCODING = MappingProxyType({route.encoding: route for route in FROZEN_ROUTES})
_ROUTE_BY_FACTOR = MappingProxyType(
    {factor_id: route for route in FROZEN_ROUTES for factor_id in route.factor_ids}
)
_FORBIDDEN_ENCODING_KINDS = MappingProxyType(
    {
        "fork_head_solve_v1": InstructionKind.FORK_HEAD_SOLVE,
        "fork_ema_clearance_v1": InstructionKind.FORK_EMA_CLEARANCE,
        "resume_lr_warmup_v1": InstructionKind.RESUME_LR_WARMUP,
    }
)
ROUTE_REGISTRY_SHA256 = _sha256(
    _canonical_json(
        {
            "schema": ROUTE_REGISTRY_SCHEMA,
            "routes": [route.manifest_row() for route in FROZEN_ROUTES],
        }
    )
)
@dataclass(frozen=True)
class Section:
    section_id: str
    factor_ids: tuple[str, ...]
    producer_id: str
    consumer_id: str
    encoding: str
    video_derived: bool
    payload: bytes
    apply_order: int
    owned_parameter_groups: tuple[str, ...] = ()
    frozen_parameter_groups: tuple[str, ...] = ()
    class_ids: tuple[str, ...] = ()
    cell_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    quotient_base_factor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("section_id", "producer_id", "consumer_id", "encoding"):
            _require_identifier(getattr(self, field_name), field_name)
        if not isinstance(self.video_derived, bool):
            raise V10Refusal(f"section {self.section_id}: video_derived must be bool")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise V10Refusal(f"section {self.section_id}: payload must be non-empty bytes")
        _require_exact_int(self.apply_order, f"section {self.section_id}: apply_order", minimum=0)
        _require_string_tuple(self.factor_ids, f"section {self.section_id}: factor_ids", nonempty=True)
        for field_name in (
            "owned_parameter_groups",
            "frozen_parameter_groups",
            "class_ids",
            "cell_ids",
            "depends_on",
            "quotient_base_factor_ids",
        ):
            _require_string_tuple(
                getattr(self, field_name), f"section {self.section_id}: {field_name}"
            )

    def manifest_row(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "factor_ids": list(self.factor_ids),
            "producer_id": self.producer_id,
            "consumer_id": self.consumer_id,
            "encoding": self.encoding,
            "video_derived": self.video_derived,
            "byte_length": len(self.payload),
            "sha256": _sha256(self.payload),
            "apply_order": self.apply_order,
            "owned_parameter_groups": list(self.owned_parameter_groups),
            "frozen_parameter_groups": list(self.frozen_parameter_groups),
            "class_ids": list(self.class_ids),
            "cell_ids": list(self.cell_ids),
            "depends_on": list(self.depends_on),
            "quotient_base_factor_ids": list(self.quotient_base_factor_ids),
        }


@dataclass(frozen=True)
class ParsedSection:
    metadata: Mapping[str, Any]
    payload: bytes
    start: int
    end: int

    @property
    def section_id(self) -> str:
        return str(self.metadata["section_id"])


@dataclass(frozen=True)
class ParsedProgram:
    program_bytes: bytes
    program_sha256: str
    typed_config_hash: str
    argv_sha256: str
    header: Mapping[str, Any]
    header_bytes: bytes
    sections: tuple[ParsedSection, ...]
    parser_proof: Mapping[str, Any]


def _decode_semantic(payload: bytes, expected_fields: set[str]) -> dict[str, Any]:
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V10Refusal("semantic payload is not valid JSON") from exc
    if not isinstance(doc, dict) or set(doc) != expected_fields:
        raise V10Refusal(
            f"semantic payload keys must be exactly {tuple(sorted(expected_fields))!r}"
        )
    if _canonical_json(doc) != payload:
        raise V10Refusal("semantic payload is not canonical JSON")
    return doc


def _validate_sections(sections: Sequence[Section]) -> tuple[Section, ...]:
    if isinstance(sections, (bytes, bytearray, str)):
        raise V10Refusal("sections must be a sequence of Section objects")
    ordered = tuple(sections)
    if len(ordered) != len(FROZEN_ROUTES):
        raise V10Refusal("cold V10 requires the exact frozen instruction sequence")
    if any(not isinstance(section, Section) for section in ordered):
        raise V10Refusal("every V10 section must be a Section object")

    seen_factors: set[str] = set()
    seen_groups: set[str] = set()
    prior_ids: list[str] = []
    prior_groups: list[str] = []
    for index, (section, route) in enumerate(zip(ordered, FROZEN_ROUTES, strict=True)):
        forbidden = _FORBIDDEN_ENCODING_KINDS.get(section.encoding)
        if forbidden is not None:
            raise V10Refusal(f"cold V10 forbids typed instruction {forbidden.value}")
        if section.encoding not in _ROUTE_BY_ENCODING:
            raise V10Refusal(f"unknown typed instruction encoding {section.encoding!r}")
        exact_route_fields = {
            "section_id": (section.section_id, route.section_id),
            "factor_ids": (section.factor_ids, route.factor_ids),
            "producer_id": (section.producer_id, route.producer_id),
            "consumer_id": (section.consumer_id, route.consumer_id),
            "encoding": (section.encoding, route.encoding),
            "video_derived": (section.video_derived, True),
        }
        drift = [
            name for name, (actual, expected) in exact_route_fields.items() if actual != expected
        ]
        if drift:
            raise V10Refusal(
                f"instruction {route.kind.value} route metadata drift: {', '.join(drift)}"
            )
        if section.apply_order != index:
            raise V10Refusal("section apply_order must be exactly contiguous in wire order")
        if section.owned_parameter_groups != (route.owned_parameter_group,):
            raise V10Refusal(
                f"instruction {route.kind.value} requires its exact disjoint parameter group"
            )
        if section.owned_parameter_groups[0] in seen_groups:
            raise V10Refusal("parameter group has more than one instruction owner")
        if section.depends_on != tuple(prior_ids):
            qualifier = "quotient residual T" if index == len(FROZEN_ROUTES) - 1 else route.kind.value
            raise V10Refusal(f"{qualifier} must depend on every predecessor section")
        if section.frozen_parameter_groups != tuple(prior_groups):
            qualifier = "quotient residual T" if index == len(FROZEN_ROUTES) - 1 else route.kind.value
            raise V10Refusal(f"{qualifier} must freeze every predecessor parameter group")
        for factor_id in section.factor_ids:
            if factor_id in seen_factors:
                raise V10Refusal(f"duplicate factor ownership: {factor_id}")
            seen_factors.add(factor_id)
        if route.kind is InstructionKind.QUOTIENT_RESIDUAL_T:
            if index != len(FROZEN_ROUTES) - 1:
                raise V10Refusal("quotient residual T must be terminal")
            if not section.class_ids or not section.cell_ids:
                raise V10Refusal("quotient residual T requires explicit class_ids and cell_ids")
            if section.quotient_base_factor_ids != QUOTIENT_BASE_FACTOR_IDS:
                raise V10Refusal("quotient residual T has the wrong exact quotient base")
        elif section.class_ids or section.cell_ids or section.quotient_base_factor_ids:
            raise V10Refusal(
                "only quotient residual T may declare class/cell/base routing metadata"
            )
        _decode_semantic(section.payload, set(route.semantic_fields) - {"BT601_YUV6"})
        seen_groups.add(route.owned_parameter_group)
        prior_groups.append(route.owned_parameter_group)
        prior_ids.append(route.section_id)

    if tuple(factor for route in FROZEN_ROUTES for factor in route.factor_ids) != (
        "1",
        "2",
        "7",
        "8",
        "6",
        "3a",
        "3b",
        "4",
        "9",
        "5",
    ):
        raise AssertionError("frozen V10 route order changed without updating its factor seal")
    if seen_factors != set(IMPLEMENTED_FACTOR_IDS):
        raise V10Refusal("implemented section factor set differs from the frozen seal")
    if seen_factors & set(MISSING_FACTOR_IDS):
        raise V10Refusal("missing factor 10 cannot own a paid section")
    return ordered


def build_payload_program(
    sections: Sequence[Section], *, typed_config_hash: str, argv_sha256: str
) -> bytes:
    """Build one canonical counted V10 program with a frozen factor seal."""

    ordered = _validate_sections(sections)
    _require_sha256(typed_config_hash, "typed_config_hash")
    _require_sha256(argv_sha256, "argv_sha256")
    header = {
        "schema": PROGRAM_SCHEMA,
        "version": VERSION,
        "typed_config_hash": typed_config_hash,
        "argv_sha256": argv_sha256,
        "section_count": len(ordered),
        "implemented_factor_ids": list(IMPLEMENTED_FACTOR_IDS),
        "missing_factor_ids": list(MISSING_FACTOR_IDS),
        "route_registry_schema": ROUTE_REGISTRY_SCHEMA,
        "route_registry_sha256": ROUTE_REGISTRY_SHA256,
        "handler_registry_schema": HANDLER_REGISTRY_SCHEMA,
        "handler_registry_version": HANDLER_REGISTRY_VERSION,
        "handler_registry_sha256": HANDLER_REGISTRY_SHA256,
        "sections": [section.manifest_row() for section in ordered],
    }
    header_bytes = _canonical_json(header)
    return PREFIX.pack(MAGIC, VERSION, len(header_bytes)) + header_bytes + b"".join(
        section.payload for section in ordered
    )


def _validate_header(header: Any, header_bytes: bytes) -> list[dict[str, Any]]:
    if not isinstance(header, dict) or _canonical_json(header) != header_bytes:
        raise V10Refusal("V10 header is not canonical JSON")
    if set(header) != set(_HEADER_FIELDS):
        raise V10Refusal("V10 header has an unknown or missing field")
    if header["schema"] != PROGRAM_SCHEMA:
        raise V10Refusal("V10 header schema mismatch")
    if type(header["version"]) is not int or header["version"] != VERSION:
        raise V10Refusal("V10 header version must be the exact integer registry version")
    _require_sha256(header["typed_config_hash"], "V10 header typed_config_hash")
    _require_sha256(header["argv_sha256"], "V10 header argv_sha256")
    if header["implemented_factor_ids"] != list(IMPLEMENTED_FACTOR_IDS):
        raise V10Refusal("V10 implemented-factor seal mismatch")
    if header["missing_factor_ids"] != list(MISSING_FACTOR_IDS):
        raise V10Refusal("V10 missing-factor seal mismatch")
    expected_registry = {
        "route_registry_schema": ROUTE_REGISTRY_SCHEMA,
        "route_registry_sha256": ROUTE_REGISTRY_SHA256,
        "handler_registry_schema": HANDLER_REGISTRY_SCHEMA,
        "handler_registry_version": HANDLER_REGISTRY_VERSION,
        "handler_registry_sha256": HANDLER_REGISTRY_SHA256,
    }
    for field_name, expected in expected_registry.items():
        actual = header[field_name]
        if field_name == "handler_registry_version" and type(actual) is not int:
            raise V10Refusal("handler registry version must be an exact integer")
        if actual != expected:
            raise V10Refusal(f"V10 authoritative registry seal mismatch: {field_name}")
    rows = header["sections"]
    count = header["section_count"]
    if type(count) is not int or count != len(FROZEN_ROUTES):
        raise V10Refusal("V10 section_count must equal the frozen route count")
    if not isinstance(rows, list) or len(rows) != count:
        raise V10Refusal("V10 section count mismatch")
    return rows


def _row_to_section(row: Any, index: int, payload: bytes) -> Section:
    if not isinstance(row, dict) or set(row) != set(_SECTION_FIELDS):
        raise V10Refusal(f"section row {index} has wrong schema")
    for field_name in ("section_id", "producer_id", "consumer_id", "encoding"):
        _require_identifier(row[field_name], f"section row {index}: {field_name}")
    for field_name in (
        "factor_ids",
        "owned_parameter_groups",
        "frozen_parameter_groups",
        "class_ids",
        "cell_ids",
        "depends_on",
        "quotient_base_factor_ids",
    ):
        _require_string_list(row[field_name], f"section row {index}: {field_name}")
    if row["video_derived"] is not True:
        raise V10Refusal(f"section row {index}: video_derived must be authoritative true")
    if type(row["apply_order"]) is not int or row["apply_order"] != index:
        raise V10Refusal("section order/apply_order mismatch")
    _require_exact_int(row["byte_length"], f"section row {index}: byte_length", minimum=1)
    _require_sha256(row["sha256"], f"section row {index}: sha256")
    try:
        section = Section(
            section_id=row["section_id"],
            factor_ids=tuple(row["factor_ids"]),
            producer_id=row["producer_id"],
            consumer_id=row["consumer_id"],
            encoding=row["encoding"],
            video_derived=row["video_derived"],
            payload=payload,
            apply_order=row["apply_order"],
            owned_parameter_groups=tuple(row["owned_parameter_groups"]),
            frozen_parameter_groups=tuple(row["frozen_parameter_groups"]),
            class_ids=tuple(row["class_ids"]),
            cell_ids=tuple(row["cell_ids"]),
            depends_on=tuple(row["depends_on"]),
            quotient_base_factor_ids=tuple(row["quotient_base_factor_ids"]),
        )
    except (TypeError, ValueError) as exc:
        raise V10Refusal(f"section row {index} has invalid typed fields") from exc
    if section.manifest_row() != row:
        raise V10Refusal(f"section row {index} disagrees with reopened payload bytes")
    return section


def parse_payload_program(program_bytes: bytes) -> ParsedProgram:
    """Reopen and prove the exact wire partition, routes, and factor seal."""

    if not isinstance(program_bytes, bytes):
        raise V10Refusal("program must be canonical bytes")
    if len(program_bytes) < PREFIX.size:
        raise V10Refusal("truncated V10 prefix")
    magic, prefix_version, header_length = PREFIX.unpack_from(program_bytes)
    if magic != MAGIC or prefix_version != VERSION:
        raise V10Refusal("wrong V10 magic or prefix version")
    if header_length <= 0:
        raise V10Refusal("V10 header length must be positive")
    header_start = PREFIX.size
    header_end = header_start + header_length
    if header_end > len(program_bytes):
        raise V10Refusal("truncated V10 header")
    header_bytes = program_bytes[header_start:header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V10Refusal("invalid V10 JSON header") from exc
    rows = _validate_header(header, header_bytes)

    cursor = header_end
    parsed_sections: list[ParsedSection] = []
    reconstructed: list[Section] = []
    ranges: list[dict[str, Any]] = []
    factor_ranges: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(_SECTION_FIELDS):
            raise V10Refusal(f"section row {index} has wrong schema")
        length = row["byte_length"]
        _require_exact_int(length, f"section row {index}: byte_length", minimum=1)
        _require_sha256(row["sha256"], f"section row {index}: sha256")
        end = cursor + length
        if end > len(program_bytes):
            raise V10Refusal(f"truncated section at row {index}")
        payload = program_bytes[cursor:end]
        if _sha256(payload) != row["sha256"]:
            raise V10Refusal(f"section hash mismatch at row {index}")
        section = _row_to_section(row, index, payload)
        parsed = ParsedSection(MappingProxyType(dict(row)), payload, cursor, end)
        parsed_sections.append(parsed)
        reconstructed.append(section)
        range_row = {
            "section_id": section.section_id,
            "start": cursor,
            "end": end,
            "byte_length": length,
            "sha256": row["sha256"],
        }
        ranges.append(range_row)
        for factor_id in section.factor_ids:
            factor_ranges[factor_id] = dict(range_row)
        cursor = end
    if cursor != len(program_bytes):
        raise V10Refusal("trailing or unowned V10 bytes")
    _validate_sections(tuple(reconstructed))
    if set(factor_ranges) != set(IMPLEMENTED_FACTOR_IDS):
        raise V10Refusal("parsed factor ranges differ from implemented-factor seal")
    if set(factor_ranges) & set(MISSING_FACTOR_IDS):
        raise V10Refusal("missing factors unexpectedly own wire ranges")

    payload_bytes = sum(item["byte_length"] for item in ranges)
    proof = {
        "schema": PARSER_PROOF_SCHEMA,
        "program_bytes": len(program_bytes),
        "prefix_bytes": PREFIX.size,
        "header_bytes": len(header_bytes),
        "payload_start": header_end,
        "payload_end": cursor,
        "payload_bytes": payload_bytes,
        "counted_video_derived_payload_bytes": payload_bytes,
        "unique_payload_range_count": len(ranges),
        "partition_sum_bytes": PREFIX.size + len(header_bytes) + payload_bytes,
        "ranges": ranges,
        "factor_ranges": factor_ranges,
        "contiguous": all(
            right["start"] == left["end"] for left, right in zip(ranges, ranges[1:])
        ),
        "no_gaps": ranges[0]["start"] == header_end
        and all(right["start"] == left["end"] for left, right in zip(ranges, ranges[1:])),
        "no_overlaps": all(
            right["start"] >= left["end"] for left, right in zip(ranges, ranges[1:])
        ),
        "no_trailing_bytes": ranges[-1]["end"] == len(program_bytes),
        "all_program_bytes_counted": PREFIX.size + len(header_bytes) + payload_bytes
        == len(program_bytes),
        "no_unowned_wire_bytes": True,
        "implemented_factor_ids": list(IMPLEMENTED_FACTOR_IDS),
        "missing_factor_ids": list(MISSING_FACTOR_IDS),
    }
    return ParsedProgram(
        program_bytes=program_bytes,
        program_sha256=_sha256(program_bytes),
        typed_config_hash=header["typed_config_hash"],
        argv_sha256=header["argv_sha256"],
        header=MappingProxyType(dict(header)),
        header_bytes=header_bytes,
        sections=tuple(parsed_sections),
        parser_proof=MappingProxyType(proof),
    )


@dataclass(frozen=True)
class HandlerResult:
    state: Mapping[str, Any]
    semantic_fields: tuple[str, ...]
    consumption_count: int
    consumed_payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, Mapping):
            raise V10Refusal("handler state must be a mapping")
        _require_string_tuple(self.semantic_fields, "handler semantic_fields", nonempty=True)
        if type(self.consumption_count) is not int:
            raise V10Refusal("handler consumption_count must be an exact integer")
        _require_sha256(self.consumed_payload_sha256, "handler consumed_payload_sha256")


Handler = Callable[[Mapping[str, Any], Mapping[str, Any], bytes], HandlerResult]


def _list_of_ints(
    value: Any,
    field_name: str,
    *,
    nonempty: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise V10Refusal(f"{field_name} must be a non-empty list[int]")
    for item in value:
        if type(item) is not int:
            raise V10Refusal(f"{field_name} must contain exact integers")
        if minimum is not None and item < minimum:
            raise V10Refusal(f"{field_name} contains a value below {minimum}")
        if maximum is not None and item > maximum:
            raise V10Refusal(f"{field_name} contains a value above {maximum}")
    return list(value)


def _state_frame(state: Mapping[str, Any], name: str) -> list[int]:
    value = state.get(name)
    return _list_of_ints(value, name, nonempty=True, minimum=0, maximum=255)


def _handled(
    state: Mapping[str, Any], semantic_fields: tuple[str, ...], payload: bytes
) -> HandlerResult:
    return HandlerResult(dict(state), semantic_fields, 1, _sha256(payload))


def _clip_byte(value: int) -> int:
    return min(255, max(0, value))


def _bt601_triplet(red: int, green: int, blue: int) -> tuple[int, int, int]:
    y_value = _clip_byte((77 * red + 150 * green + 29 * blue + 128) >> 8)
    u_value = _clip_byte(((-43 * red - 85 * green + 128 * blue + 128) >> 8) + 128)
    v_value = _clip_byte(((128 * red - 107 * green - 21 * blue + 128) >> 8) + 128)
    return y_value, u_value, v_value


def _yuv6(frame0: list[int], frame1: list[int]) -> list[int]:
    if len(frame0) != len(frame1) or len(frame0) % 3:
        raise V10Refusal("BT.601/YUV6 requires equal RGB frames with triplet alignment")
    output: list[int] = []
    for index in range(0, len(frame0), 3):
        output.extend(_bt601_triplet(*frame0[index : index + 3]))
        output.extend(_bt601_triplet(*frame1[index : index + 3]))
    return output


def _generator_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    if state:
        raise V10Refusal("CountedGenerator must initialize an empty receiver state")
    doc = _decode_semantic(payload, {"frame0_rgb", "frame1_rgb", "seed_bytes"})
    frame0 = _list_of_ints(doc["frame0_rgb"], "frame0_rgb", nonempty=True, minimum=0, maximum=255)
    frame1 = _list_of_ints(doc["frame1_rgb"], "frame1_rgb", nonempty=True, minimum=0, maximum=255)
    seed = _list_of_ints(doc["seed_bytes"], "seed_bytes", nonempty=True, minimum=0, maximum=255)
    if len(frame0) != len(frame1) or len(frame0) % 3:
        raise V10Refusal("CountedGenerator frames must have equal RGB-triplet lengths")
    if len(seed) > len(frame0) + 1:
        raise V10Refusal(
            "CountedGenerator seed_bytes must each reach at least one generated sample"
        )
    generated0 = [value ^ seed[index % len(seed)] for index, value in enumerate(frame0)]
    generated1 = [value ^ seed[(index + 1) % len(seed)] for index, value in enumerate(frame1)]
    output = {
        "frame0_rgb": generated0,
        "frame1_rgb": generated1,
        "pose_six": [0, 0, 0, 0, 0, 0],
        "shared_resize_preimage": {},
        "yuv6": [],
        "rate_tokens": [],
        "class_cells": {},
    }
    return _handled(output, ("frame0_rgb", "frame1_rgb", "seed_bytes"), payload)


def _factor2_integer_scorer_plane_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(
        payload,
        {"y_uint8", "camera_shape", "scorer_shape", "receiver_contract_id"},
    )
    if doc["receiver_contract_id"] != FACTOR2_RECEIVER_CONTRACT_ID:
        raise V10Refusal("factor-2 receiver contract id drift")
    camera_shape = _list_of_ints(
        doc["camera_shape"], "camera_shape", nonempty=True, minimum=1
    )
    scorer_shape = _list_of_ints(
        doc["scorer_shape"], "scorer_shape", nonempty=True, minimum=1
    )
    if len(camera_shape) != 3 or len(scorer_shape) != 3:
        raise V10Refusal("factor-2 camera/scorer shapes must be exact HWC triplets")
    if camera_shape[2] != 3 or scorer_shape[2] != 3:
        raise V10Refusal("factor-2 structural route requires exactly three RGB channels")
    if (
        camera_shape[0] > 4096
        or camera_shape[1] > 4096
        or scorer_shape[0] > 2048
        or scorer_shape[1] > 2048
    ):
        raise V10Refusal("factor-2 camera/scorer geometry exceeds production bounds")
    y_values = _list_of_ints(
        doc["y_uint8"], "y_uint8", nonempty=True, minimum=0, maximum=255
    )
    expected_y_values = scorer_shape[0] * scorer_shape[1] * scorer_shape[2]
    if len(y_values) != expected_y_values:
        raise V10Refusal("factor-2 y_uint8 length differs from scorer geometry")
    try:
        operator = DisjointResizeOperator.build(
            camera_h=camera_shape[0],
            camera_w=camera_shape[1],
            scorer_h=scorer_shape[0],
            scorer_w=scorer_shape[1],
        )
        y_plane = np.asarray(y_values, dtype=np.uint8).reshape(scorer_shape)
        frame = realize_factor2_uint8_scorer_plane(operator, y_plane)
        proof = verify_factor2_uint8_scorer_plane(operator, frame, y_plane)
    except Uint8LatticeError as exc:
        raise V10Refusal("factor-2 production geometry or realization refused") from exc
    if not proof.certified_exact:
        raise V10Refusal("factor-2 structural realization failed exact verification")
    output = dict(state)
    structural_frame = _state_frame(state, "frame1_rgb")
    realized_hash = _sha256(frame.tobytes(order="C"))
    realized_digest = bytes.fromhex(realized_hash)
    output["frame1_rgb"] = [
        realized_digest[index % len(realized_digest)]
        for index in range(len(structural_frame))
    ]
    output["factor2_realization"] = {
        "receiver_contract_id": FACTOR2_RECEIVER_CONTRACT_ID,
        "camera_shape": camera_shape,
        "scorer_shape": scorer_shape,
        "y_sha256": _sha256(y_plane.tobytes(order="C")),
        "frame_sha256": realized_hash,
        "denominator": proof.denominator,
        "numerator_values_verified": proof.numerator_equal_values,
        "canonical_values_verified": proof.canonical_equal_values,
        "certified_exact": True,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    return _handled(
        output,
        ("y_uint8", "camera_shape", "scorer_shape", "receiver_contract_id"),
        payload,
    )


def _frame0_pose_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"frame0_delta", "pose_six"})
    frame0 = _state_frame(state, "frame0_rgb")
    delta = _list_of_ints(doc["frame0_delta"], "frame0_delta", nonempty=True, minimum=-255, maximum=255)
    pose = _list_of_ints(
        doc["pose_six"], "pose_six", nonempty=True, minimum=-255, maximum=255
    )
    if len(delta) != len(frame0) or len(pose) != 6:
        raise V10Refusal("Frame0PoseSixCarrier requires a full frame0 delta and exactly six pose values")
    output = dict(state)
    output["frame0_rgb"] = [
        _clip_byte(value + shift + pose[index % len(pose)])
        for index, (value, shift) in enumerate(zip(frame0, delta))
    ]
    output["pose_six"] = pose
    return _handled(output, ("frame0_delta", "pose_six"), payload)


def _init_head_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"head_bias"})
    frame1 = _state_frame(state, "frame1_rgb")
    bias = _list_of_ints(doc["head_bias"], "head_bias", nonempty=True, minimum=-255, maximum=255)
    if len(bias) != len(frame1):
        raise V10Refusal("InitHeadSolve head_bias must match frame1 length")
    output = dict(state)
    output["frame1_rgb"] = [
        _clip_byte(value + shift) for value, shift in zip(frame1, bias)
    ]
    return _handled(output, ("head_bias",), payload)


def _shared_resize_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"fanout", "weights"})
    frame0 = _state_frame(state, "frame0_rgb")
    frame1 = _state_frame(state, "frame1_rgb")
    fanout = _list_of_ints(doc["fanout"], "fanout", nonempty=True, minimum=0)
    weights = _list_of_ints(doc["weights"], "weights", nonempty=True, minimum=1, maximum=255)
    if len(fanout) != len(weights) or any(index >= len(frame0) for index in fanout):
        raise V10Refusal("SharedResizePreimage requires aligned in-range fanout/weights")
    denominator = sum(weights)
    shared = {
        "fanout": fanout,
        "weights": weights,
        "frame0": sum(frame0[index] * weight for index, weight in zip(fanout, weights))
        // denominator,
        "frame1": sum(frame1[index] * weight for index, weight in zip(fanout, weights))
        // denominator,
    }
    output = dict(state)
    updated0 = list(frame0)
    updated1 = list(frame1)
    for index in fanout:
        updated0[index] = shared["frame0"]
        updated1[index] = shared["frame1"]
    output["frame0_rgb"] = updated0
    output["frame1_rgb"] = updated1
    output["shared_resize_preimage"] = shared
    return _handled(output, ("fanout", "weights"), payload)


def _rgb_yuv6_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"rgb_bias"})
    bias = _list_of_ints(doc["rgb_bias"], "rgb_bias", nonempty=True, minimum=-255, maximum=255)
    if len(bias) != 3:
        raise V10Refusal("RgbYuv6Projection rgb_bias must have three channels")
    frame0 = _state_frame(state, "frame0_rgb")
    frame1 = _state_frame(state, "frame1_rgb")
    adjusted0 = [_clip_byte(value + bias[index % 3]) for index, value in enumerate(frame0)]
    adjusted1 = [_clip_byte(value + bias[index % 3]) for index, value in enumerate(frame1)]
    output = dict(state)
    output["frame0_rgb"] = adjusted0
    output["frame1_rgb"] = adjusted1
    output["yuv6"] = _yuv6(adjusted0, adjusted1)
    output["rgb_yuv6_reference"] = "BT.601 integer YUV6 v1"
    return _handled(output, ("rgb_bias", "BT601_YUV6"), payload)


def _blind_fill_handler(
    state: Mapping[str, Any], _metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"blind_indices", "fill_value", "rate_tokens"})
    frame0 = _state_frame(state, "frame0_rgb")
    frame1 = _state_frame(state, "frame1_rgb")
    indices = _list_of_ints(doc["blind_indices"], "blind_indices", nonempty=True, minimum=0)
    if len(set(indices)) != len(indices) or any(index >= len(frame1) for index in indices):
        raise V10Refusal("BlindFillRateGrammar indices must be unique and in range")
    fill_value = doc["fill_value"]
    if type(fill_value) is not int or not 0 <= fill_value <= 255:
        raise V10Refusal("BlindFillRateGrammar fill_value must be an exact byte")
    rate_tokens = _list_of_ints(
        doc["rate_tokens"], "rate_tokens", nonempty=True, minimum=0, maximum=255
    )
    rate_adjustment = sum(
        (position + 1) * token for position, token in enumerate(rate_tokens)
    )
    if fill_value + rate_adjustment > 255:
        raise V10Refusal(
            "BlindFillRateGrammar reference carrier exceeds its exact uint8 range"
        )
    carried_fill = fill_value + rate_adjustment
    updated = list(frame1)
    for index in indices:
        updated[index] = carried_fill
    output = dict(state)
    output["frame1_rgb"] = updated
    output["yuv6"] = _yuv6(frame0, updated)
    output["rate_tokens"] = rate_tokens
    return _handled(output, ("blind_indices", "fill_value", "rate_tokens"), payload)


def _quotient_handler(
    state: Mapping[str, Any], metadata: Mapping[str, Any], payload: bytes
) -> HandlerResult:
    doc = _decode_semantic(payload, {"updates"})
    updates = doc["updates"]
    if not isinstance(updates, list) or not updates:
        raise V10Refusal("QuotientResidualT requires non-empty updates")
    frame0 = _state_frame(state, "frame0_rgb")
    frame1 = _state_frame(state, "frame1_rgb")
    class_ids = set(metadata["class_ids"])
    cell_ids = set(metadata["cell_ids"])
    routed = dict(state.get("class_cells", {}))
    seen_targets: set[tuple[str, int]] = set()
    for update in updates:
        if not isinstance(update, dict) or set(update) != {
            "class_id",
            "cell_id",
            "frame",
            "index",
            "delta",
        }:
            raise V10Refusal("invalid QuotientResidualT update schema")
        class_id = update["class_id"]
        cell_id = update["cell_id"]
        frame_name = update["frame"]
        index = update["index"]
        delta = update["delta"]
        if not isinstance(class_id, str) or not isinstance(cell_id, str):
            raise V10Refusal("QuotientResidualT class_id/cell_id must be strings")
        if class_id not in class_ids or cell_id not in cell_ids:
            raise V10Refusal("QuotientResidualT update uses an undeclared class/cell route")
        if frame_name not in {"frame0", "frame1"}:
            raise V10Refusal("QuotientResidualT frame must be frame0 or frame1")
        frame = frame0 if frame_name == "frame0" else frame1
        if type(index) is not int or not 0 <= index < len(frame):
            raise V10Refusal("QuotientResidualT update index is out of range")
        if type(delta) is not int or not -255 <= delta <= 255:
            raise V10Refusal("QuotientResidualT delta must be an exact bounded integer")
        if delta == 0:
            raise V10Refusal("QuotientResidualT refuses a paid zero residual")
        target = (frame_name, index)
        if target in seen_targets:
            raise V10Refusal("QuotientResidualT requires one residual owner per frame index")
        seen_targets.add(target)
        frame[index] = _clip_byte(frame[index] + delta)
        route_key = f"{class_id}:{cell_id}:{frame_name}"
        routed[route_key] = routed.get(route_key, 0) + delta
    output = dict(state)
    output["frame0_rgb"] = frame0
    output["frame1_rgb"] = frame1
    output["yuv6"] = _yuv6(frame0, frame1)
    output["class_cells"] = routed
    return _handled(output, ("updates",), payload)


_SEALED_HANDLER_REGISTRY: Mapping[str, Handler] = MappingProxyType(
    {
        "counted_generator_v2": _generator_handler,
        "factor2_integer_scorer_plane_v1": _factor2_integer_scorer_plane_handler,
        "frame0_pose_six_carrier_v1": _frame0_pose_handler,
        "init_head_solve_v2": _init_head_handler,
        "shared_resize_preimage_v1": _shared_resize_handler,
        "rgb_yuv6_projection_v1": _rgb_yuv6_handler,
        "blind_fill_rate_grammar_v1": _blind_fill_handler,
        "quotient_residual_t_v2": _quotient_handler,
    }
)


def _implementation_source_sha256(value: Any) -> str:
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise RuntimeError("V10 receiver implementation source is unavailable") from exc
    return _sha256(source)


_HANDLER_SHARED_SEMANTIC_COMPONENTS = MappingProxyType(
    {
        "HandlerResult": HandlerResult,
        "_canonical_json": _canonical_json,
        "_sha256": _sha256,
        "_decode_semantic": _decode_semantic,
        "_list_of_ints": _list_of_ints,
        "_state_frame": _state_frame,
        "_handled": _handled,
        "_clip_byte": _clip_byte,
        "_bt601_triplet": _bt601_triplet,
        "_yuv6": _yuv6,
        "DisjointResizeOperator": DisjointResizeOperator,
        "Uint8LatticeError": Uint8LatticeError,
        "realize_factor2_uint8_scorer_plane": realize_factor2_uint8_scorer_plane,
        "verify_factor2_uint8_scorer_plane": verify_factor2_uint8_scorer_plane,
    }
)
HANDLER_SHARED_SEMANTICS_SHA256 = _sha256(
    _canonical_json(
        {
            name: _implementation_source_sha256(component)
            for name, component in _HANDLER_SHARED_SEMANTIC_COMPONENTS.items()
        }
    )
)
HANDLER_IMPLEMENTATION_SHA256S: Mapping[str, str] = MappingProxyType(
    {
        encoding: _sha256(
            _canonical_json(
                {
                    "handler_source_sha256": _implementation_source_sha256(handler),
                    "shared_semantics_sha256": HANDLER_SHARED_SEMANTICS_SHA256,
                }
            )
        )
        for encoding, handler in _SEALED_HANDLER_REGISTRY.items()
    }
)
HANDLER_REGISTRY_SHA256 = _sha256(
    _canonical_json(
        {
            "schema": HANDLER_REGISTRY_SCHEMA,
            "version": HANDLER_REGISTRY_VERSION,
            "shared_semantics_sha256": HANDLER_SHARED_SEMANTICS_SHA256,
            "handlers": [
                {
                    "encoding": route.encoding,
                    "kind": route.kind.value,
                    "semantic_fields": list(route.semantic_fields),
                    "implementation_sha256": HANDLER_IMPLEMENTATION_SHA256S[
                        route.encoding
                    ],
                }
                for route in FROZEN_ROUTES
            ],
        }
    )
)
FROZEN_HANDLER_REGISTRY = _SEALED_HANDLER_REGISTRY
DEFAULT_HANDLERS = FROZEN_HANDLER_REGISTRY


@dataclass(frozen=True)
class ReceiverCheckpoint:
    program_sha256: str
    typed_config_hash: str
    next_section_index: int
    consumed_section_ids: tuple[str, ...]
    state_bytes: bytes
    state_sha256: str
    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_SCHEMA:
            raise V10Refusal("receiver checkpoint schema mismatch")
        _require_sha256(self.program_sha256, "receiver checkpoint program_sha256")
        _require_sha256(self.typed_config_hash, "receiver checkpoint typed_config_hash")
        _require_exact_int(
            self.next_section_index, "receiver checkpoint next_section_index", minimum=0
        )
        _require_string_tuple(
            self.consumed_section_ids, "receiver checkpoint consumed_section_ids"
        )
        if not isinstance(self.state_bytes, bytes) or not self.state_bytes:
            raise V10Refusal("receiver checkpoint state missing")
        _require_sha256(self.state_sha256, "receiver checkpoint state_sha256")
        if _sha256(self.state_bytes) != self.state_sha256:
            raise V10Refusal("receiver checkpoint state hash mismatch")
        try:
            state = json.loads(self.state_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V10Refusal("receiver checkpoint state is not JSON") from exc
        if not isinstance(state, dict) or _canonical_json(state) != self.state_bytes:
            raise V10Refusal("receiver checkpoint state is not a canonical mapping")

    def to_bytes(self) -> bytes:
        return _canonical_json(
            {
                "schema": self.schema,
                "program_sha256": self.program_sha256,
                "typed_config_hash": self.typed_config_hash,
                "next_section_index": self.next_section_index,
                "consumed_section_ids": list(self.consumed_section_ids),
                "state_bytes_b64": base64.b64encode(self.state_bytes).decode("ascii"),
                "state_sha256": self.state_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ReceiverCheckpoint":
        if not isinstance(payload, bytes) or not payload:
            raise V10Refusal("receiver checkpoint must be canonical bytes")
        try:
            doc = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V10Refusal("invalid receiver checkpoint JSON") from exc
        if not isinstance(doc, dict) or set(doc) != set(_CHECKPOINT_FIELDS):
            raise V10Refusal("receiver checkpoint has an unknown or missing field")
        if _canonical_json(doc) != payload:
            raise V10Refusal("receiver checkpoint is noncanonical")
        if doc["schema"] != CHECKPOINT_SCHEMA:
            raise V10Refusal("receiver checkpoint schema mismatch")
        _require_sha256(doc["program_sha256"], "receiver checkpoint program_sha256")
        _require_sha256(doc["typed_config_hash"], "receiver checkpoint typed_config_hash")
        _require_exact_int(
            doc["next_section_index"], "receiver checkpoint next_section_index", minimum=0
        )
        consumed = _require_string_list(
            doc["consumed_section_ids"], "receiver checkpoint consumed_section_ids"
        )
        if not isinstance(doc["state_bytes_b64"], str):
            raise V10Refusal("receiver checkpoint state_bytes_b64 must be a string")
        try:
            state_bytes = base64.b64decode(doc["state_bytes_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise V10Refusal("receiver checkpoint state_bytes_b64 is invalid") from exc
        if base64.b64encode(state_bytes).decode("ascii") != doc["state_bytes_b64"]:
            raise V10Refusal("receiver checkpoint state_bytes_b64 is noncanonical")
        return cls(
            program_sha256=doc["program_sha256"],
            typed_config_hash=doc["typed_config_hash"],
            next_section_index=doc["next_section_index"],
            consumed_section_ids=tuple(consumed),
            state_bytes=state_bytes,
            state_sha256=doc["state_sha256"],
        )


@dataclass(frozen=True)
class ReceiverResult:
    completed: bool
    output_bytes: bytes | None
    output_sha256: str | None
    receipts: tuple[Mapping[str, Any], ...]
    checkpoint: ReceiverCheckpoint


def _receive_reopened(
    program_bytes: bytes,
    *,
    handlers: Mapping[str, Handler],
    checkpoint: bytes | None,
    stop_after: int | None,
    authoritative_handlers: Mapping[str, Handler] = _SEALED_HANDLER_REGISTRY,
) -> ReceiverResult:
    parsed = parse_payload_program(program_bytes)
    section_ids = tuple(section.section_id for section in parsed.sections)
    if checkpoint is None:
        start_index = 0
        state: Mapping[str, Any] = {}
        consumed: tuple[str, ...] = ()
        prefix_receipts: tuple[Mapping[str, Any], ...] = ()
    else:
        if not isinstance(checkpoint, bytes):
            raise V10Refusal("receiver checkpoint input must be canonical reopened bytes")
        restored = ReceiverCheckpoint.from_bytes(checkpoint)
        if (
            restored.program_sha256 != parsed.program_sha256
            or restored.typed_config_hash != parsed.typed_config_hash
        ):
            raise V10Refusal("receiver checkpoint program/config drift")
        if restored.next_section_index != len(restored.consumed_section_ids):
            raise V10Refusal("receiver checkpoint index is not the consumed-prefix length")
        if restored.consumed_section_ids != section_ids[: restored.next_section_index]:
            raise V10Refusal("receiver checkpoint consumed IDs are not the exact prefix")
        if restored.next_section_index >= len(parsed.sections):
            raise V10Refusal("receiver checkpoint is complete or out of range")
        prefix = _receive_reopened(
            program_bytes,
            handlers=authoritative_handlers,
            checkpoint=None,
            stop_after=restored.next_section_index,
            authoritative_handlers=authoritative_handlers,
        )
        if restored.state_bytes != prefix.checkpoint.state_bytes:
            raise V10Refusal(
                "receiver checkpoint state differs from deterministic consumed-prefix replay"
            )
        state = json.loads(restored.state_bytes.decode("utf-8"))
        start_index = restored.next_section_index
        consumed = restored.consumed_section_ids
        prefix_receipts = prefix.receipts

    target = len(parsed.sections) if stop_after is None else stop_after
    if type(target) is not int or not start_index <= target <= len(parsed.sections):
        raise V10Refusal("stop_after must be an exact integer in the remaining section range")
    receipts: list[Mapping[str, Any]] = list(prefix_receipts)
    for section in parsed.sections[start_index:target]:
        encoding = str(section.metadata["encoding"])
        handler = handlers.get(encoding)
        if handler is None:
            raise V10Refusal(f"missing handler for frozen encoding {encoding}")
        before = _canonical_json(state)
        result = handler(state, section.metadata, section.payload)
        if not isinstance(result, HandlerResult):
            raise V10Refusal(f"handler {encoding} returned no typed consumption result")
        if type(result.consumption_count) is not int or result.consumption_count != 1:
            raise V10Refusal(f"section {section.section_id} must be consumed exactly once")
        canonical_handler = authoritative_handlers[encoding]
        if handler is not canonical_handler:
            raise V10Refusal("custom handler cannot issue an authoritative receiver receipt")
        if result.consumed_payload_sha256 != _sha256(section.payload):
            raise V10Refusal("handler consumption receipt is not bound to reopened payload bytes")
        route = _ROUTE_BY_ENCODING[encoding]
        if result.semantic_fields != route.semantic_fields:
            raise V10Refusal("handler semantic-field receipt differs from frozen registry")
        state = dict(result.state)
        after = _canonical_json(state)
        if after == before:
            raise V10Refusal(
                f"section {section.section_id} was counted but made no semantic state change"
            )
        decoded_before = _canonical_json(
            {
                "frame0_rgb": json.loads(before).get("frame0_rgb"),
                "frame1_rgb": json.loads(before).get("frame1_rgb"),
            }
        )
        decoded_after = _canonical_json(
            {
                "frame0_rgb": state.get("frame0_rgb"),
                "frame1_rgb": state.get("frame1_rgb"),
            }
        )
        if decoded_after == decoded_before:
            raise V10Refusal(
                f"section {section.section_id} was counted but left decoded frames unchanged"
            )
        receipts.append(
            MappingProxyType(
                {
                    "schema": RECEIPT_SCHEMA,
                    "handler_registry_schema": HANDLER_REGISTRY_SCHEMA,
                    "handler_registry_version": HANDLER_REGISTRY_VERSION,
                    "handler_registry_sha256": HANDLER_REGISTRY_SHA256,
                    "handler_shared_semantics_sha256": (
                        HANDLER_SHARED_SEMANTICS_SHA256
                    ),
                    "handler_implementation_sha256": (
                        HANDLER_IMPLEMENTATION_SHA256S[encoding]
                    ),
                    "authoritative_handler": True,
                    "section_id": section.section_id,
                    "factor_ids": list(section.metadata["factor_ids"]),
                    "consumer_id": section.metadata["consumer_id"],
                    "encoding": encoding,
                    "start": section.start,
                    "end": section.end,
                    "byte_length": len(section.payload),
                    "sha256": _sha256(section.payload),
                    "consumed_payload_sha256": result.consumed_payload_sha256,
                    "consumption_count": 1,
                    "semantic_fields": list(result.semantic_fields),
                    "state_before_sha256": _sha256(before),
                    "state_after_sha256": _sha256(after),
                    "decoded_frames_before_sha256": _sha256(decoded_before),
                    "decoded_frames_after_sha256": _sha256(decoded_after),
                }
            )
        )
        consumed += (section.section_id,)

    state_bytes = _canonical_json(state)
    checkpoint_result = ReceiverCheckpoint(
        program_sha256=parsed.program_sha256,
        typed_config_hash=parsed.typed_config_hash,
        next_section_index=target,
        consumed_section_ids=consumed,
        state_bytes=state_bytes,
        state_sha256=_sha256(state_bytes),
    )
    complete = target == len(parsed.sections)
    return ReceiverResult(
        completed=complete,
        output_bytes=state_bytes if complete else None,
        output_sha256=_sha256(state_bytes) if complete else None,
        receipts=tuple(receipts),
        checkpoint=checkpoint_result,
    )


def receive_payload_program(
    program_bytes: bytes,
    *,
    checkpoint: bytes | None = None,
    stop_after: int | None = None,
) -> ReceiverResult:
    """Public receiver: reopen canonical bytes and use only the frozen registry."""

    if not isinstance(program_bytes, bytes):
        raise V10Refusal("public receiver requires canonical program bytes, not ParsedProgram")
    return _receive_reopened(
        program_bytes,
        handlers=_SEALED_HANDLER_REGISTRY,
        checkpoint=checkpoint,
        stop_after=stop_after,
        authoritative_handlers=_SEALED_HANDLER_REGISTRY,
    )


def _receive_payload_program_for_test(
    program_bytes: bytes,
    *,
    handlers: Mapping[str, Handler],
    checkpoint: bytes | None = None,
    stop_after: int | None = None,
) -> ReceiverResult:
    """Private negative-test seam; custom handlers can never mint authority."""

    if not isinstance(program_bytes, bytes) or not isinstance(handlers, Mapping):
        raise V10Refusal("negative-test receiver requires bytes and a handler mapping")
    return _receive_reopened(
        program_bytes,
        handlers=handlers,
        checkpoint=checkpoint,
        stop_after=stop_after,
        authoritative_handlers=_SEALED_HANDLER_REGISTRY,
    )


@dataclass(frozen=True)
class EvidenceArtifact:
    payload: bytes
    sha256: str
    schema: str
    verdict: str
    authority_axis: str
    factor_id: str
    producer_id: str
    consumer_id: str
    compiled_config_hash: str
    program_sha256: str
    covered_section_id: str
    covered_section_sha256: str
    receiver_receipt_sha256: str

    @classmethod
    def create(
        cls,
        body: Mapping[str, Any],
        *,
        factor_id: str,
        producer_id: str,
        consumer_id: str,
        compiled_config_hash: str,
        program_sha256: str,
        covered_section_id: str,
        covered_section_sha256: str,
        receiver_receipt: Mapping[str, Any],
        schema: str = "v10_factor_interaction_receipt.v2",
        verdict: str = "PASS",
        authority_axis: str = "local-CPU structural/non-score",
    ) -> "EvidenceArtifact":
        if not isinstance(body, Mapping) or not body:
            raise V10Refusal("evidence body must be a non-empty semantic mapping")
        receipt_sha = _sha256(_canonical_json(dict(receiver_receipt)))
        doc = {
            "schema": schema,
            "verdict": verdict,
            "authority_axis": authority_axis,
            "factor_id": factor_id,
            "producer_id": producer_id,
            "consumer_id": consumer_id,
            "compiled_config_hash": compiled_config_hash,
            "program_sha256": program_sha256,
            "covered_section_id": covered_section_id,
            "covered_section_sha256": covered_section_sha256,
            "receiver_receipt_sha256": receipt_sha,
            "body": dict(body),
        }
        payload = _canonical_json(doc)
        return cls(
            payload=payload,
            sha256=_sha256(payload),
            schema=schema,
            verdict=verdict,
            authority_axis=authority_axis,
            factor_id=factor_id,
            producer_id=producer_id,
            consumer_id=consumer_id,
            compiled_config_hash=compiled_config_hash,
            program_sha256=program_sha256,
            covered_section_id=covered_section_id,
            covered_section_sha256=covered_section_sha256,
            receiver_receipt_sha256=receipt_sha,
        )

    def validate(
        self,
        *,
        factor_id: str,
        config_hash: str,
        program_hash: str,
        section: ParsedSection,
        receiver_receipt: Mapping[str, Any],
    ) -> None:
        if not isinstance(self.payload, bytes) or not self.payload or _sha256(self.payload) != self.sha256:
            raise V10Refusal(f"factor {factor_id}: empty or stale evidence bytes")
        try:
            doc = json.loads(self.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V10Refusal(f"factor {factor_id}: invalid evidence JSON") from exc
        expected_fields = {
            "schema",
            "verdict",
            "authority_axis",
            "factor_id",
            "producer_id",
            "consumer_id",
            "compiled_config_hash",
            "program_sha256",
            "covered_section_id",
            "covered_section_sha256",
            "receiver_receipt_sha256",
            "body",
        }
        if not isinstance(doc, dict) or set(doc) != expected_fields or _canonical_json(doc) != self.payload:
            raise V10Refusal(f"factor {factor_id}: evidence schema or canonical bytes are invalid")
        if self.schema != "v10_factor_interaction_receipt.v2" or doc["schema"] != self.schema:
            raise V10Refusal(f"factor {factor_id}: wrong interaction-receipt schema")
        if not isinstance(doc["body"], dict) or not doc["body"]:
            raise V10Refusal(f"factor {factor_id}: evidence has no semantic body")
        for field_name in expected_fields - {"body"}:
            if doc[field_name] != getattr(self, field_name):
                raise V10Refusal(f"factor {factor_id}: evidence object disagrees with reopened bytes")
        route = _ROUTE_BY_FACTOR[factor_id]
        if self.verdict != "PASS" or self.authority_axis != "local-CPU structural/non-score":
            raise V10Refusal(f"factor {factor_id}: evidence verdict or authority axis is adverse")
        if (
            self.factor_id != factor_id
            or self.producer_id != route.producer_id
            or self.consumer_id != route.consumer_id
        ):
            raise V10Refusal(f"factor {factor_id}: evidence route mismatch")
        if self.compiled_config_hash != config_hash or self.program_sha256 != program_hash:
            raise V10Refusal(f"factor {factor_id}: evidence config/program hash drift")
        if (
            self.covered_section_id != section.section_id
            or self.covered_section_sha256 != section.metadata["sha256"]
            or factor_id not in section.metadata["factor_ids"]
        ):
            raise V10Refusal(f"factor {factor_id}: evidence section coverage drift")
        if self.receiver_receipt_sha256 != _sha256(_canonical_json(dict(receiver_receipt))):
            raise V10Refusal(f"factor {factor_id}: evidence is not bound to the receiver receipt")
        if (
            receiver_receipt.get("authoritative_handler") is not True
            or receiver_receipt.get("section_id") != section.section_id
            or receiver_receipt.get("sha256") != section.metadata["sha256"]
            or receiver_receipt.get("consumption_count") != 1
            or receiver_receipt.get("handler_registry_sha256") != HANDLER_REGISTRY_SHA256
            or receiver_receipt.get("handler_shared_semantics_sha256")
            != HANDLER_SHARED_SEMANTICS_SHA256
            or receiver_receipt.get("handler_implementation_sha256")
            != HANDLER_IMPLEMENTATION_SHA256S[str(section.metadata["encoding"])]
        ):
            raise V10Refusal(f"factor {factor_id}: receiver receipt lacks authoritative coverage")


@dataclass(frozen=True)
class CompletenessRow:
    factor_id: str
    term_id: str
    owner_task: str
    disposition: str
    derivation_ref: str
    build_sha: str
    compiled_config_hash: str
    consumer_id: str
    resume_schema_and_replay_ref: str
    measurement_receipt_sha256: str | None
    authority_axis: str
    interaction_receipts: tuple[EvidenceArtifact, ...]
    adoption_or_scoped_exclusion: str
    strict_certificate: str

    def manifest_row(self) -> dict[str, Any]:
        row = {field_name: getattr(self, field_name) for field_name in _COMPLETENESS_FIELDS}
        row["interaction_receipts"] = [receipt.sha256 for receipt in self.interaction_receipts]
        return row


def _validate_completeness(
    rows: tuple[CompletenessRow, ...],
    *,
    config_hash: str,
    program_hash: str,
    sections: Sequence[ParsedSection],
    receiver_receipts: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    if len(rows) != len(FROZEN_FACTOR_IDS) or any(
        not isinstance(row, CompletenessRow) for row in rows
    ):
        raise V10Refusal("completeness manifest requires exactly 11 typed rows")
    if tuple(row.factor_id for row in rows) != FROZEN_FACTOR_IDS:
        raise V10Refusal("completeness rows must use the frozen 11-factor order")
    section_by_factor = {
        factor_id: section
        for section in sections
        for factor_id in section.metadata["factor_ids"]
    }
    receipt_by_section = {receipt["section_id"]: receipt for receipt in receiver_receipts}

    for row in rows:
        factor_id = row.factor_id
        for field_name in (
            "factor_id",
            "term_id",
            "owner_task",
            "disposition",
            "derivation_ref",
            "build_sha",
            "compiled_config_hash",
            "consumer_id",
            "resume_schema_and_replay_ref",
            "authority_axis",
            "adoption_or_scoped_exclusion",
            "strict_certificate",
        ):
            _require_identifier(getattr(row, field_name), f"factor {factor_id}: {field_name}")
        if not isinstance(row.interaction_receipts, tuple) or any(
            not isinstance(receipt, EvidenceArtifact) for receipt in row.interaction_receipts
        ):
            raise V10Refusal(f"factor {factor_id}: interaction_receipts must be typed artifacts")
        if row.build_sha != "UNCOMMITTED_LOCAL" and not (
            len(row.build_sha) in {40, 64}
            and all(character in "0123456789abcdef" for character in row.build_sha)
        ):
            raise V10Refusal(f"factor {factor_id}: build_sha is not a commit/content SHA")
        if row.compiled_config_hash != config_hash:
            raise V10Refusal(f"factor {factor_id}: completeness config hash drift")
        if row.authority_axis != "local-CPU structural/non-score":
            raise V10Refusal(f"factor {factor_id}: wrong authority axis")
        if row.resume_schema_and_replay_ref != (
            f"{CHECKPOINT_SCHEMA}:local-bit-identical-prefix-replay"
        ):
            raise V10Refusal(f"factor {factor_id}: wrong local resume schema")
        scoped = row.adoption_or_scoped_exclusion.lower()
        if "no launch" not in scoped or "no score" not in scoped:
            raise V10Refusal(f"factor {factor_id}: scope must state no launch and no score")
        if row.strict_certificate == "COMPLETE":
            raise V10Refusal(f"factor {factor_id}: local compiler cannot authorize COMPLETE")

        if factor_id in MISSING_FACTOR_IDS:
            if factor_id in section_by_factor:
                raise V10Refusal(f"factor {factor_id}: MISSING factor unexpectedly has a section")
            if (
                row.disposition != "MISSING"
                or row.consumer_id != "BLOCKED"
                or row.strict_certificate != "MISSING"
                or row.measurement_receipt_sha256 is not None
                or row.interaction_receipts
            ):
                raise V10Refusal(
                    f"factor {factor_id}: MISSING requires BLOCKED consumer and no receipts"
                )
            continue

        route = _ROUTE_BY_FACTOR[factor_id]
        if row.disposition not in {"HAVE", "FOLDED"}:
            raise V10Refusal(f"factor {factor_id}: implemented factor cannot be MISSING")
        if row.consumer_id != route.consumer_id or row.strict_certificate != "PARTIAL":
            raise V10Refusal(f"factor {factor_id}: implemented factor route/certificate drift")
        section = section_by_factor.get(factor_id)
        if section is None or section.metadata["consumer_id"] != route.consumer_id:
            raise V10Refusal(f"factor {factor_id}: actual consumer section is absent")
        runtime_receipt = receipt_by_section.get(section.section_id)
        if runtime_receipt is None:
            raise V10Refusal(f"factor {factor_id}: consumer has no runtime receipt")
        if row.disposition == "FOLDED" and (
            not row.interaction_receipts or row.measurement_receipt_sha256 is None
        ):
            raise V10Refusal(f"factor {factor_id}: FOLDED requires a reopened interaction receipt")
        for artifact in row.interaction_receipts:
            artifact.validate(
                factor_id=factor_id,
                config_hash=config_hash,
                program_hash=program_hash,
                section=section,
                receiver_receipt=runtime_receipt,
            )
        if row.measurement_receipt_sha256 is not None:
            _require_sha256(
                row.measurement_receipt_sha256,
                f"factor {factor_id}: measurement_receipt_sha256",
            )
            if row.measurement_receipt_sha256 not in {
                artifact.sha256 for artifact in row.interaction_receipts
            }:
                raise V10Refusal(f"factor {factor_id}: receipt SHA is not reopened evidence")
    return tuple(row.manifest_row() for row in rows), False


@dataclass(frozen=True)
class CompileResult:
    schema: str
    typed_config_hash: str
    trainer_argv: tuple[str, ...]
    argv_sha256: str
    parser_verified_arguments: Mapping[str, Any]
    resolved_lawref_manifest: Mapping[str, Any]
    dsl_program_manifest: Mapping[str, Any]
    dsl_compile_hash: str
    dsl_compile_provenance: Mapping[str, Any]
    dsl_bijection_complete: bool
    dsl_bijection_violations: tuple[str, ...]
    payload_program_bytes: bytes
    payload_program_sha256: str
    program_byte_count: int
    payload_byte_count: int
    counted_video_derived_bytes: int
    parser_proof: Mapping[str, Any]
    receiver_receipts: tuple[Mapping[str, Any], ...]
    receiver_output_bytes: bytes
    receiver_output_sha256: str
    resume_schema: str
    resume_replay_equal: bool
    completeness_schema: str
    completeness_rows: tuple[Mapping[str, Any], ...]
    implemented_factor_ids: tuple[str, ...]
    missing_factor_ids: tuple[str, ...]
    launch_ready: bool
    score_claim: bool = False
    promotion_eligible: bool = False


def _canonicalize_namespace(namespace: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, tuple):
            output[key] = list(value)
        else:
            output[key] = value
    return output


def _normalized_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _scan_forbidden_state(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise V10Refusal(f"cold V10 state key at {path} must be a string")
            if _normalized_token(key) in _FORBIDDEN_STATE_TOKENS:
                raise V10Refusal(f"cold V10 forbids fork/resume state token {key!r} at {path}")
            _scan_forbidden_state(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_forbidden_state(item, f"{path}[{index}]")
    elif isinstance(value, str) and _normalized_token(value) in _FORBIDDEN_STATE_TOKENS:
        raise V10Refusal(f"cold V10 forbids fork/resume state value {value!r} at {path}")


def _validate_cold(config: TypedWitnessConfig, sections: tuple[Section, ...]) -> None:
    program = config.to_program()
    if config.num_pairs != 600 or program.num_pairs != 600:
        raise V10Refusal("exact cold V10 compiler requires num_pairs == 600")
    if program.flag_dict().get("--verdict-pairs") != 0:
        raise V10Refusal("exact cold V10 requires typed --verdict-pairs 0")
    if program.flag_dict().get("--seed") != config.seed:
        raise V10Refusal("exact cold V10 requires typed --seed equal to TypedWitnessConfig.seed")
    if program.resume_from is not None:
        raise V10Refusal("cold V10 forbids non-null resume_from")
    _scan_forbidden_state(config.model_dump(mode="json"), "typed_config")
    for section in sections:
        forbidden = _FORBIDDEN_ENCODING_KINDS.get(section.encoding)
        if forbidden is not None:
            raise V10Refusal(f"cold V10 forbids typed instruction {forbidden.value}")
        try:
            body = json.loads(section.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V10Refusal(f"section {section.section_id}: payload is not JSON") from exc
        _scan_forbidden_state(section.manifest_row(), f"section.{section.section_id}.metadata")
        _scan_forbidden_state(body, f"section.{section.section_id}.payload")


@lru_cache(maxsize=4)
def _canonical_strict_bijection_audit(repo_root_text: str) -> tuple[bool, tuple[str, ...]]:
    from tac.v9_provenance_gates import (
        V9ProvenanceGateError,
        check_config_flag_provenance_bijection_complete,
    )

    repo_root = Path(repo_root_text)
    try:
        violations = check_config_flag_provenance_bijection_complete(
            repo_root=repo_root, strict=True, verbose=False
        )
    except V9ProvenanceGateError as exc:
        lines = tuple(
            line.strip()
            for line in str(exc).splitlines()[1:]
            if line.strip()
        )
        if not lines:
            lines = ("canonical strict #332 audit refused without structured violations",)
        return False, lines
    return not violations, tuple(str(item) for item in violations)


def compile_cold_v10(
    config: TypedWitnessConfig,
    sections: Sequence[Section],
    completeness_rows: Sequence[CompletenessRow],
    *,
    target_config_tags: Mapping[str, str] | None = None,
    repo_root: Path | str | None = None,
) -> CompileResult:
    """Compile a typed cold program into a non-authorizing local certificate."""

    if not isinstance(config, TypedWitnessConfig):
        raise V10Refusal("compile_cold_v10 requires TypedWitnessConfig")
    frozen_sections = tuple(sections)
    frozen_rows = tuple(completeness_rows)
    _validate_cold(config, frozen_sections)
    _validate_sections(frozen_sections)
    violations = config.validate_program()
    if violations:
        raise V10Refusal("typed WitnessProgram validation failed: " + " | ".join(violations))
    program = config.to_program()
    argv, canonical_manifest = program.compile_trainer_argv_with_constants(
        dict(target_config_tags or {}), repo_root=repo_root
    )

    resolved_manifest: dict[str, Any] = {
        key: dict(value) for key, value in canonical_manifest.items()
    }
    for lever in config.levers:
        for flag, lawref in lever.lawrefs.items():
            declaration = lawref_to_declaration(lawref)
            declared = lever.lawref_declarations.get(flag)
            if declared is not None and declared != declaration:
                raise V10Refusal(f"{flag}: live LawRef differs from its typed declaration")
            resolved = resolve(lawref, target_config_tags, repo_root=repo_root)
            record = resolved.to_dict()
            record.pop("resolved_at", None)
            if program.flag_dict().get(flag) != resolved.value:
                raise V10Refusal(f"{flag}: resolved LawRef value differs from typed DSL override")
            resolved_manifest[flag] = record

    argv_tuple = tuple(str(token) for token in argv)
    argv_sha = _sha256("\x00".join(argv_tuple).encode("utf-8"))
    try:
        parsed_args = build_real_trainer_parser().parse_args(list(argv_tuple[2:]))
    except SystemExit as exc:
        raise V10Refusal("compiled trainer argv refused by the real parser") from exc
    parser_arguments = _canonicalize_namespace(parsed_args)
    if parser_arguments.get("num_pairs") != 600 or parser_arguments.get("verdict_pairs") != 0:
        raise V10Refusal("real trainer parser lost exact n600/full-verdict settings")
    config_hash = config.typed_config_hash()

    try:
        from tac.v9_provenance_gates import (
            build_dsl_compile_provenance_document,
            canonicalize_resolved_argv,
        )

        dsl_compile_provenance = build_dsl_compile_provenance_document(
            program_name=config.name,
            typed_config=config,
            compiler_manifest=resolved_manifest,
            repo_root=Path(repo_root).resolve() if repo_root is not None else None,
        )
    except Exception as exc:
        raise V10Refusal("canonical #332 DSL self-recompile/hash check failed") from exc
    if dsl_compile_provenance.get("resolved_argv") != list(canonicalize_resolved_argv(argv_tuple)):
        raise V10Refusal("canonical #332 self-recompile argv differs from V10 argv")
    dsl_compile_hash = dsl_compile_provenance.get("dsl_compile_hash")
    _require_sha256(dsl_compile_hash, "canonical #332 dsl_compile_hash")
    stable_provenance = json.loads(json.dumps(dsl_compile_provenance))
    context = stable_provenance.setdefault("non_authoritative_context", {})
    if isinstance(context, dict):
        context["compiled_at_utc"] = "excluded-from-deterministic-certificate"

    raw_manifest = config.program_manifest()
    stable_manifest = {key: value for key, value in raw_manifest.items() if key != "compiled_at_utc"}
    stable_manifest["compiled_at_utc"] = "excluded-from-deterministic-certificate"
    stable_manifest["resolved_argv_sha256"] = argv_sha

    payload_program = build_payload_program(
        frozen_sections,
        typed_config_hash=config_hash,
        argv_sha256=argv_sha,
    )
    parsed_program = parse_payload_program(payload_program)
    if parsed_program.typed_config_hash != config_hash or parsed_program.argv_sha256 != argv_sha:
        raise V10Refusal("program header lost typed-config/argv binding")
    receiver = receive_payload_program(payload_program)
    if not receiver.completed or receiver.output_bytes is None or receiver.output_sha256 is None:
        raise V10Refusal("full receiver did not consume every section")
    if len(receiver.receipts) != len(parsed_program.sections):
        raise V10Refusal("receiver receipt count differs from section count")

    split = len(parsed_program.sections) // 2
    interrupted = receive_payload_program(payload_program, stop_after=split)
    resumed = receive_payload_program(
        payload_program, checkpoint=interrupted.checkpoint.to_bytes()
    )
    if resumed.output_bytes != receiver.output_bytes:
        raise V10Refusal("resume replay differs from uninterrupted receiver output")

    completeness_manifest, launch_ready = _validate_completeness(
        frozen_rows,
        config_hash=config_hash,
        program_hash=parsed_program.program_sha256,
        sections=parsed_program.sections,
        receiver_receipts=receiver.receipts,
    )
    active_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    bijection_complete, bijection_violations = _canonical_strict_bijection_audit(str(active_root))
    if launch_ready:
        raise V10Refusal("local V10 structural compiler cannot become launch-ready")
    return CompileResult(
        schema="v10_compiler_result.v2",
        typed_config_hash=config_hash,
        trainer_argv=argv_tuple,
        argv_sha256=argv_sha,
        parser_verified_arguments=parser_arguments,
        resolved_lawref_manifest=resolved_manifest,
        dsl_program_manifest=stable_manifest,
        dsl_compile_hash=dsl_compile_hash,
        dsl_compile_provenance=stable_provenance,
        dsl_bijection_complete=bijection_complete,
        dsl_bijection_violations=bijection_violations,
        payload_program_bytes=payload_program,
        payload_program_sha256=parsed_program.program_sha256,
        program_byte_count=parsed_program.parser_proof["program_bytes"],
        payload_byte_count=parsed_program.parser_proof["payload_bytes"],
        counted_video_derived_bytes=parsed_program.parser_proof[
            "counted_video_derived_payload_bytes"
        ],
        parser_proof=parsed_program.parser_proof,
        receiver_receipts=receiver.receipts,
        receiver_output_bytes=receiver.output_bytes,
        receiver_output_sha256=receiver.output_sha256,
        resume_schema=CHECKPOINT_SCHEMA,
        resume_replay_equal=True,
        completeness_schema=COMPLETENESS_SCHEMA,
        completeness_rows=completeness_manifest,
        implemented_factor_ids=IMPLEMENTED_FACTOR_IDS,
        missing_factor_ids=MISSING_FACTOR_IDS,
        launch_ready=False,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "COMPLETENESS_SCHEMA",
    "DEFAULT_HANDLERS",
    "FACTOR2_RECEIVER_CONTRACT_ID",
    "FROZEN_FACTOR_IDS",
    "FROZEN_HANDLER_REGISTRY",
    "FROZEN_ROUTES",
    "HANDLER_IMPLEMENTATION_SHA256S",
    "HANDLER_REGISTRY_SCHEMA",
    "HANDLER_REGISTRY_SHA256",
    "HANDLER_SHARED_SEMANTICS_SHA256",
    "IMPLEMENTED_FACTOR_IDS",
    "MAGIC",
    "MISSING_FACTOR_IDS",
    "PARSER_PROOF_SCHEMA",
    "PREFIX",
    "PROGRAM_SCHEMA",
    "QUOTIENT_BASE_FACTOR_IDS",
    "RECEIPT_SCHEMA",
    "ROUTE_REGISTRY_SCHEMA",
    "ROUTE_REGISTRY_SHA256",
    "CompileResult",
    "CompletenessRow",
    "EvidenceArtifact",
    "HandlerResult",
    "InstructionKind",
    "ParsedProgram",
    "ParsedSection",
    "ReceiverCheckpoint",
    "ReceiverResult",
    "RouteSpec",
    "Section",
    "V10Refusal",
    "build_payload_program",
    "canonical_semantic_payload",
    "compile_cold_v10",
    "parse_payload_program",
    "receive_payload_program",
]
