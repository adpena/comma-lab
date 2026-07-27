# SPDX-License-Identifier: MIT
"""Canonical selected-solution compiler, placement, lifecycle, and byte VM.

This module is the single definitions owner for the G17/G21 compiler surface.
``taskspace_g17_compiler_placement`` is intentionally only a compatibility
adapter that re-exports these exact Python objects.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Final, Literal, Protocol, TypeAlias

import numpy as np

from tac.contest_score import compute_contest_score, pose_term, rate_term, seg_term
from tac.witness_dsl.taskspace_g17_forward_observation import (
    G17CandidateForwardObservationV1,
    parse_g17_candidate_forward_receipt,
)


class G17CompilerPlacementError(ValueError):
    """A placement, ownership, lifecycle, or VM invariant failed closed."""


class G17CompilerBlockerCodeV1(StrEnum):
    G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED = "G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED"
    G17_UNSUPPORTED_TOPOLOGY_OR_CONSTRAINT_VM_OPERATION = "G17_UNSUPPORTED_TOPOLOGY_OR_CONSTRAINT_VM_OPERATION"
    G17_EXACT_CONTEST_AUTHORITY_ADAPTER_OWED = "G17_EXACT_CONTEST_AUTHORITY_ADAPTER_OWED"
    G17_AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE_OWED = "G17_AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE_OWED"
    G17_REAL_COMPOSED_COUNTERFACTUAL_EVIDENCE_OWED = "G17_REAL_COMPOSED_COUNTERFACTUAL_EVIDENCE_OWED"


class G17CompilerBlocker(G17CompilerPlacementError):
    def __init__(self, code: G17CompilerBlockerCodeV1, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class G17SemanticStreamRoleV1(StrEnum):
    SKELETON = "SKELETON"
    CONNECTION = "CONNECTION"
    FIBER = "FIBER"
    GAUGE = "GAUGE"
    RESIDUAL = "RESIDUAL"


class G17ScientificRoleV1(StrEnum):
    """The seven scientific factor roles; never aliases the five stream roles."""

    TOPOLOGY_WORLDSHEET = "topology_worldsheet"
    BULK_BOUNDARY = "bulk_boundary"
    LANE_CHART = "lane_chart"
    MOVABLE_MYCAR = "movable_mycar"
    CELL_VALUE_PREIMAGE = "cell_value_preimage"
    POSE_TRANSPORT_FRAME0 = "pose_transport_frame0"
    IRREDUCIBLE_QUOTIENT = "irreducible_quotient"


class G17LifecyclePhaseV1(StrEnum):
    SOURCE_TRUTH = "SOURCE_TRUTH"
    OBLIGATION_IR = "OBLIGATION_IR"
    REALIZED_PAIR = "REALIZED_PAIR"
    ARCHIVE_ARTIFACT = "ARCHIVE_ARTIFACT"
    AUTH_EVAL_CLOSURE = "AUTH_EVAL_CLOSURE"
    DECODE_RECEIPT = "DECODE_RECEIPT"
    SCORE_RECEIPT = "SCORE_RECEIPT"


class G17EvaluatorRecursionStageV1(StrEnum):
    L1_PROGRAM = "L1_program"
    L2_CHART = "L2_chart"
    L3_RASTER = "L3_raster"
    L4_SCORER_FEATURE = "L4_scorer_feature"
    L5_VERDICT = "L5_verdict"


class G17RecursionNamespaceV1(StrEnum):
    TS1_INFORMATION_HOME = "TS1_INFORMATION_HOME"
    SN1_ARTIFACT_LAYER = "SN1_ARTIFACT_LAYER"
    SN1_DERIVATION_RECURSION = "SN1_DERIVATION_RECURSION"
    LP1_PRICING_STRATUM = "LP1_PRICING_STRATUM"


_RECURSION_STAGES_BY_NAMESPACE: Final[dict[G17RecursionNamespaceV1, frozenset[str]]] = {
    G17RecursionNamespaceV1.TS1_INFORMATION_HOME: frozenset(item.value for item in G17EvaluatorRecursionStageV1),
    G17RecursionNamespaceV1.SN1_ARTIFACT_LAYER: frozenset(
        {"L1_PROGRAM", "L2_RECEIVER_R", "L3_SCORER_FEATURE", "L4_SCORER_DECISION", "L5_VERDICT"}
    ),
    G17RecursionNamespaceV1.SN1_DERIVATION_RECURSION: frozenset(
        {"L0_SCORE_SIGNATURE", "L1_TERM_NATIVE_GEOMETRY", "L2_TEMPORAL_COMPOSITION"}
    ),
    G17RecursionNamespaceV1.LP1_PRICING_STRATUM: frozenset(
        {"L1_program", "L2_chart_grammar", "L3_RGB_realization", "L4_scorer_feature"}
    ),
}


@dataclass(frozen=True, slots=True)
class G17RecursionCoordinateV1:
    namespace: G17RecursionNamespaceV1
    stage: str

    def __post_init__(self) -> None:
        if type(self.namespace) is not G17RecursionNamespaceV1:
            raise G17CompilerPlacementError("recursion namespace is not typed")
        if type(self.stage) is not str or not self.stage or not self.stage.isascii():
            raise G17CompilerPlacementError("recursion stage must be nonempty ASCII")
        if self.stage not in _RECURSION_STAGES_BY_NAMESPACE[self.namespace]:
            raise G17CompilerPlacementError("recursion stage does not belong to its versioned namespace")


class G17PlacementClassV1(StrEnum):
    GENERIC_DECODER_FREE = "GENERIC_DECODER_FREE"
    COUNTED_VIDEO_STATISTIC = "COUNTED_VIDEO_STATISTIC"
    ENCODER_ONLY_EVIDENCE = "ENCODER_ONLY_EVIDENCE"
    COUNTED_PACKAGED_EXECUTABLE = "COUNTED_PACKAGED_EXECUTABLE"


# Compatibility name.  It is an alias to the canonical placement axis, not a
# parallel enum that could be cross-cast or drift independently.
G17PhysicalByteHomeV1 = G17PlacementClassV1


class G17LogicalValueTypeV1(StrEnum):
    SEMANTIC_TOPOLOGY = "SEMANTIC_TOPOLOGY"
    REALIZATION_GAUGE = "REALIZATION_GAUGE"
    CHRONOLOGICAL_POSE_PREIMAGE = "CHRONOLOGICAL_POSE_PREIMAGE"
    POPULATION_SHARING = "POPULATION_SHARING"
    ENTROPY_CONTEXT = "ENTROPY_CONTEXT"
    ANALYTIC_RESIDUAL_OWNERSHIP = "ANALYTIC_RESIDUAL_OWNERSHIP"
    LEARNED_RESIDUAL_OWNERSHIP = "LEARNED_RESIDUAL_OWNERSHIP"
    ENCODER_ONLY_TEACHER_ORACLE_EVIDENCE = "ENCODER_ONLY_TEACHER_ORACLE_EVIDENCE"
    FORWARD_OBSERVATION = "FORWARD_OBSERVATION"
    TERMINAL_ENVELOPE = "TERMINAL_ENVELOPE"
    GENERIC_VM_INTERPRETER = "GENERIC_VM_INTERPRETER"
    COUNTED_VM_BYTECODE = "COUNTED_VM_BYTECODE"
    COUNTED_VM_OPERAND = "COUNTED_VM_OPERAND"
    PACKAGED_EXECUTABLE = "PACKAGED_EXECUTABLE"


class G17ArtifactClassV1(StrEnum):
    GENERIC_DETERMINISTIC_MECHANISM = "GENERIC_DETERMINISTIC_MECHANISM"
    IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC = "IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC"
    ENCODER_ONLY_EVIDENCE = "ENCODER_ONLY_EVIDENCE"
    PACKAGED_EXECUTABLE_OR_TABLE = "PACKAGED_EXECUTABLE_OR_TABLE"


class G17LogicalOwnershipKindV1(StrEnum):
    SEMANTIC_TOPOLOGY = "SEMANTIC_TOPOLOGY"
    REALIZATION_GAUGE = "REALIZATION_GAUGE"
    CHRONOLOGICAL_POSE = "CHRONOLOGICAL_POSE"
    POPULATION_SHARED = "POPULATION_SHARED"
    ENTROPY_CONTEXT = "ENTROPY_CONTEXT"
    ANALYTIC_RESIDUAL = "ANALYTIC_RESIDUAL"
    LEARNED_RESIDUAL = "LEARNED_RESIDUAL"
    TERMINAL_PROTOCOL = "TERMINAL_PROTOCOL"
    GENERIC_DECODER = "GENERIC_DECODER"
    COUNTED_INSTRUCTION = "COUNTED_INSTRUCTION"
    COUNTED_OPERAND = "COUNTED_OPERAND"
    ENCODER_EVIDENCE = "ENCODER_EVIDENCE"
    PACKAGED_EXECUTABLE = "PACKAGED_EXECUTABLE"


class G17ProofKindV1(StrEnum):
    ARCHIVE_DECODE_EQUALITY = "ARCHIVE_DECODE_EQUALITY"
    AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE = "AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE"
    SCORE_OBSERVATION = "SCORE_OBSERVATION"
    BEATS_CURRENT_FRONTIER = "BEATS_CURRENT_FRONTIER"


class G17ProofDependencyDomainV1(StrEnum):
    ARCHIVE_BYTES = "ARCHIVE_BYTES"
    MEMBER_CONTAINER_MAPPING = "MEMBER_CONTAINER_MAPPING"
    RECEIVER_IMPLEMENTATION = "RECEIVER_IMPLEMENTATION"
    RECEIVER_RUNTIME = "RECEIVER_RUNTIME"
    PAIR_ORDER = "PAIR_ORDER"
    DECODER_EQUALITY_ALGORITHM = "DECODER_EQUALITY_ALGORITHM"
    PUBLIC_ENTRYPOINT_CHAIN = "PUBLIC_ENTRYPOINT_CHAIN"
    RUNTIME_FILE_CLOSURE = "RUNTIME_FILE_CLOSURE"
    AUTH_EVAL_EXECUTION_RECEIPT = "AUTH_EVAL_EXECUTION_RECEIPT"
    DECODE_RECEIPT = "DECODE_RECEIPT"
    FROZEN_SCORER = "FROZEN_SCORER"
    SCORER_RUNTIME = "SCORER_RUNTIME"
    AXIS_AND_SAMPLE_SCOPE = "AXIS_AND_SAMPLE_SCOPE"
    SCORE_RECEIPT = "SCORE_RECEIPT"
    SEMANTIC_COMPETITIVE_TARGET = "SEMANTIC_COMPETITIVE_TARGET"
    POINTER_SNAPSHOT = "POINTER_SNAPSHOT"


class G17AuthorityClassV1(StrEnum):
    RESEARCH_ADVISORY = "RESEARCH_ADVISORY"
    CONTEST_CPU = "CONTEST_CPU"
    CONTEST_CUDA = "CONTEST_CUDA"
    OFFICIAL_EXTERNAL_TARGET = "OFFICIAL_EXTERNAL_TARGET"


class G17EffectObservationKindV1(StrEnum):
    ENDPOINT = "ENDPOINT"
    INDIVISIBLE_HYPEREDGE = "INDIVISIBLE_HYPEREDGE"


class G17EffectSupportV1(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class G17TerminalCompilerPassV1(StrEnum):
    """Required terminal ordering: solve first, train only the quotient, link last."""

    MAXIMAL_INVERSE_SOLVE = "MAXIMAL_INVERSE_SOLVE"
    MINIMAL_IRREDUCIBLE_JOINT_DESCENT = "MINIMAL_IRREDUCIBLE_JOINT_DESCENT"
    TERMINAL_LINK = "TERMINAL_LINK"


class G17RuntimeFileScopeV1(StrEnum):
    SUBMISSION_PUBLIC_ENTRYPOINT = "SUBMISSION_PUBLIC_ENTRYPOINT"
    SUBMISSION_RUNTIME_DEPENDENCY = "SUBMISSION_RUNTIME_DEPENDENCY"
    EVALUATOR_PUBLIC_ENTRYPOINT = "EVALUATOR_PUBLIC_ENTRYPOINT"
    EVALUATOR_RUNTIME_DEPENDENCY = "EVALUATOR_RUNTIME_DEPENDENCY"
    SYSTEM_RUNTIME_DEPENDENCY = "SYSTEM_RUNTIME_DEPENDENCY"


class G17RuntimeDependencyMechanismV1(StrEnum):
    PROCESS_EXEC = "PROCESS_EXEC"
    PYTHON_IMPORT = "PYTHON_IMPORT"
    DYNAMIC_LOAD = "DYNAMIC_LOAD"
    RUNTIME_FILE_READ = "RUNTIME_FILE_READ"


_G17_MAXIMALLY_INVERSE_SOLVED_ROLES: Final = tuple(
    role for role in G17ScientificRoleV1 if role is not G17ScientificRoleV1.IRREDUCIBLE_QUOTIENT
)
_G17_IRREDUCIBLE_ONLY_ROLE: Final = (G17ScientificRoleV1.IRREDUCIBLE_QUOTIENT,)
_G17_TERMINAL_COMPILER_PASS_ORDER: Final = (
    G17TerminalCompilerPassV1.MAXIMAL_INVERSE_SOLVE,
    G17TerminalCompilerPassV1.MINIMAL_IRREDUCIBLE_JOINT_DESCENT,
    G17TerminalCompilerPassV1.TERMINAL_LINK,
)


@dataclass(frozen=True, slots=True)
class G17TerminalCompilerScheduleV1:
    """Typed linker schedule; no earlier pass may train a reducible role."""

    inverse_solved_scientific_roles: tuple[G17ScientificRoleV1, ...]
    joint_descent_trainable_roles: tuple[G17ScientificRoleV1, ...]
    pass_order: tuple[G17TerminalCompilerPassV1, ...]

    def __post_init__(self) -> None:
        if self.inverse_solved_scientific_roles != _G17_MAXIMALLY_INVERSE_SOLVED_ROLES:
            raise G17CompilerPlacementError(
                "terminal compiler must maximally inverse-solve every reducible scientific role"
            )
        if self.joint_descent_trainable_roles != _G17_IRREDUCIBLE_ONLY_ROLE:
            raise G17CompilerPlacementError(
                "terminal joint descent may train only the irreducible quotient representation"
            )
        if self.pass_order != _G17_TERMINAL_COMPILER_PASS_ORDER:
            raise G17CompilerPlacementError(
                "joint descent must be the minimal final optimization pass immediately before terminal link"
            )

    @classmethod
    def canonical(cls) -> G17TerminalCompilerScheduleV1:
        return cls(
            inverse_solved_scientific_roles=_G17_MAXIMALLY_INVERSE_SOLVED_ROLES,
            joint_descent_trainable_roles=_G17_IRREDUCIBLE_ONLY_ROLE,
            pass_order=_G17_TERMINAL_COMPILER_PASS_ORDER,
        )


_FORBIDDEN_PAYLOAD_CLASSES: Final = frozenset(
    {
        "TARGET_FRAME",
        "DENSE_LABEL_GRID",
        "SCORER_RGB",
        "SCORER_NUMERATOR",
        "SCORER_DENOMINATOR",
        "SCORER_WEIGHT",
        "POSE6_TARGET_TABLE",
        "TEACHER_STATE",
        "ORACLE_STATE",
        "PUBLIC_CANDIDATE_PAYLOAD",
        "EXPLICIT_DENSE_PREIMAGE",
        "SOLVE_TRACE",
    }
)


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _exact_bytes(value: object, *, name: str, nonempty: bool = True) -> bytes:
    if type(value) is not bytes or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise G17CompilerPlacementError(f"{name} must be {qualifier}immutable bytes")
    return value


def _ascii(value: object, *, name: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise G17CompilerPlacementError(f"{name} must be nonempty ASCII")
    return value


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
        raise G17CompilerPlacementError("value is not finite canonical ASCII JSON") from exc


def _decode_canonical_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    """Decode one exact canonical JSON object, rejecting duplicate keys."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17CompilerPlacementError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    raw = _exact_bytes(payload, name=name)
    try:
        decoded = json.loads(raw.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17CompilerPlacementError(f"{name} is not canonical ASCII JSON") from exc
    if type(decoded) is not dict or _canonical_json(decoded) != raw:
        raise G17CompilerPlacementError(f"{name} changed under canonical parse/re-emit")
    return decoded


def _b64(payload: bytes) -> str:
    return b64encode(_exact_bytes(payload, name="base64 source", nonempty=False)).decode("ascii")


def _unb64(value: object, *, name: str, nonempty: bool = True) -> bytes:
    if type(value) is not str or not value.isascii():
        raise G17CompilerPlacementError(f"{name} must be canonical base64 text")
    try:
        payload = b64decode(value, validate=True)
    except (Base64Error, ValueError) as exc:
        raise G17CompilerPlacementError(f"{name} is not valid base64") from exc
    if b64encode(payload).decode("ascii") != value or (nonempty and not payload):
        raise G17CompilerPlacementError(f"{name} is noncanonical or empty")
    return payload


def _float_hex(value: float, *, name: str) -> str:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise G17CompilerPlacementError(f"{name} must be a finite nonnegative binary64")
    return value.hex()


def _parse_float_hex(value: object, *, name: str) -> float:
    if type(value) is not str or not value.isascii():
        raise G17CompilerPlacementError(f"{name} must be exact float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise G17CompilerPlacementError(f"{name} is not valid float.hex text") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed.hex() != value:
        raise G17CompilerPlacementError(f"{name} is noncanonical, negative, or non-finite")
    return parsed


def _relative_runtime_path(value: object, *, name: str) -> str:
    path = _ascii(value, name=name)
    parts = path.split("/")
    if (
        path.startswith("/")
        or path.endswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise G17CompilerPlacementError(f"{name} must be a normalized relative POSIX path")
    return path


class G17StrictReemittableV1(Protocol):
    schema: str

    def to_receipt_bytes(self) -> bytes: ...


G17StrictReceiptParserV1: TypeAlias = Callable[[bytes], G17StrictReemittableV1]


@dataclass(frozen=True, slots=True)
class G17ReopenedEvidencePacketV1:
    """Exact retained evidence bytes that really reopen under a strict parser."""

    exact_packet_bytes: bytes = field(repr=False)
    strict_parser: G17StrictReceiptParserV1 = field(repr=False, compare=False)
    expected_schema: str

    def __post_init__(self) -> None:
        payload = _exact_bytes(self.exact_packet_bytes, name="evidence packet")
        _ascii(self.expected_schema, name="evidence schema")
        if not callable(self.strict_parser):
            raise G17CompilerPlacementError("evidence packet requires a strict public parser")
        try:
            parsed = self.strict_parser(payload)
        except Exception as exc:
            raise G17CompilerPlacementError("strict parser refused retained evidence bytes") from exc
        if getattr(parsed, "schema", None) != self.expected_schema:
            raise G17CompilerPlacementError("reopened evidence schema changed")
        if not hasattr(parsed, "to_receipt_bytes") or parsed.to_receipt_bytes() != payload:
            raise G17CompilerPlacementError("evidence packet failed strict parse/re-emit identity")

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.exact_packet_bytes)

    def reopen(self) -> G17StrictReemittableV1:
        parsed = self.strict_parser(self.exact_packet_bytes)
        if parsed.to_receipt_bytes() != self.exact_packet_bytes:
            raise G17CompilerPlacementError("retained evidence drifted after construction")
        return parsed


@dataclass(frozen=True, slots=True)
class G17RuntimeDependencyFileV1:
    """One byte-owned file observed in the public evaluator process closure."""

    relative_path: str
    exact_file_bytes: bytes = field(repr=False)
    custody_owner: str
    scope: G17RuntimeFileScopeV1

    def __post_init__(self) -> None:
        _relative_runtime_path(self.relative_path, name="runtime file path")
        _exact_bytes(self.exact_file_bytes, name="runtime file")
        _ascii(self.custody_owner, name="runtime file custody owner")
        if type(self.scope) is not G17RuntimeFileScopeV1:
            raise G17CompilerPlacementError("runtime file scope is not typed")

    @property
    def content_sha256(self) -> str:
        return _sha256(self.exact_file_bytes)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"G17-RUNTIME-DEPENDENCY-FILE-V1\0"
            + self.relative_path.encode("ascii")
            + b"\0"
            + self.custody_owner.encode("ascii")
            + b"\0"
            + self.scope.value.encode("ascii")
            + b"\0"
            + self.exact_file_bytes
        )


@dataclass(frozen=True, slots=True)
class G17RuntimeDependencyEdgeV1:
    """Observed directed dependency: importer/launcher consumes dependency."""

    importer_path: str
    dependency_path: str
    mechanism: G17RuntimeDependencyMechanismV1

    def __post_init__(self) -> None:
        _relative_runtime_path(self.importer_path, name="runtime importer path")
        _relative_runtime_path(self.dependency_path, name="runtime dependency path")
        if self.importer_path == self.dependency_path:
            raise G17CompilerPlacementError("runtime dependency edge is self-referential")
        if type(self.mechanism) is not G17RuntimeDependencyMechanismV1:
            raise G17CompilerPlacementError("runtime dependency mechanism is not typed")


@dataclass(frozen=True, slots=True)
class G17FunctionalQuotientIdentityV1:
    """Function/evaluator-cell identity, intentionally independent of spelling."""

    exact_decoded_output_bytes: bytes = field(repr=False)
    exact_evaluator_cell_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _exact_bytes(self.exact_decoded_output_bytes, name="functional decoded output")
        _exact_bytes(self.exact_evaluator_cell_bytes, name="functional evaluator cell")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"G17-FUNCTIONAL-QUOTIENT-IDENTITY-V1\0"
            + len(self.exact_decoded_output_bytes).to_bytes(8, "big")
            + self.exact_decoded_output_bytes
            + self.exact_evaluator_cell_bytes
        )


@dataclass(frozen=True, slots=True)
class G17ParameterSpellingIdentityV1:
    """Exact counted spelling; repacks/requantization necessarily change it."""

    exact_parameter_bytes: bytes = field(repr=False)
    spelling_format: str

    def __post_init__(self) -> None:
        _exact_bytes(self.exact_parameter_bytes, name="parameter spelling")
        _ascii(self.spelling_format, name="spelling format")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"G17-PARAMETER-SPELLING-IDENTITY-V1\0"
            + self.spelling_format.encode("ascii")
            + b"\0"
            + self.exact_parameter_bytes
        )


@dataclass(frozen=True, slots=True)
class _TypedLogicalBytesV1:
    exact_bytes: bytes = field(repr=False)
    logical_type: G17LogicalValueTypeV1 = field(init=False)

    def __post_init__(self) -> None:
        _exact_bytes(self.exact_bytes, name=type(self).__name__)

    @property
    def content_sha256(self) -> str:
        return _sha256(self.exact_bytes)

    @property
    def identity_sha256(self) -> str:
        domain = f"G17-LOGICAL-{self.logical_type.value}-V1\0".encode("ascii")
        return _sha256(domain + self.exact_bytes)


@dataclass(frozen=True, slots=True)
class G17SemanticTopologyV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.SEMANTIC_TOPOLOGY, init=False)


@dataclass(frozen=True, slots=True)
class G17RealizationGaugeV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.REALIZATION_GAUGE, init=False)


@dataclass(frozen=True, slots=True)
class G17ChronologicalPosePreimageV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(
        default=G17LogicalValueTypeV1.CHRONOLOGICAL_POSE_PREIMAGE,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class G17PopulationSharingV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.POPULATION_SHARING, init=False)


@dataclass(frozen=True, slots=True)
class G17EntropyContextV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.ENTROPY_CONTEXT, init=False)


@dataclass(frozen=True, slots=True)
class G17AnalyticResidualOwnershipV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(
        default=G17LogicalValueTypeV1.ANALYTIC_RESIDUAL_OWNERSHIP,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class G17LearnedResidualOwnershipV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(
        default=G17LogicalValueTypeV1.LEARNED_RESIDUAL_OWNERSHIP,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class G17EncoderOnlyTeacherOracleEvidenceV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(
        default=G17LogicalValueTypeV1.ENCODER_ONLY_TEACHER_ORACLE_EVIDENCE,
        init=False,
    )


@dataclass(frozen=True, slots=True)
class G17ForwardObservationLogicalV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.FORWARD_OBSERVATION, init=False)


@dataclass(frozen=True, slots=True)
class G17TerminalEnvelopeLogicalV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.TERMINAL_ENVELOPE, init=False)


@dataclass(frozen=True, slots=True)
class G17GenericVMInterpreterV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.GENERIC_VM_INTERPRETER, init=False)


@dataclass(frozen=True, slots=True)
class G17CountedVMBytecodeV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.COUNTED_VM_BYTECODE, init=False)


@dataclass(frozen=True, slots=True)
class G17CountedVMOperandV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.COUNTED_VM_OPERAND, init=False)


@dataclass(frozen=True, slots=True)
class G17PackagedExecutableV1(_TypedLogicalBytesV1):
    logical_type: G17LogicalValueTypeV1 = field(default=G17LogicalValueTypeV1.PACKAGED_EXECUTABLE, init=False)


G17TypedLogicalValueV1 = (
    G17SemanticTopologyV1
    | G17RealizationGaugeV1
    | G17ChronologicalPosePreimageV1
    | G17PopulationSharingV1
    | G17EntropyContextV1
    | G17AnalyticResidualOwnershipV1
    | G17LearnedResidualOwnershipV1
    | G17EncoderOnlyTeacherOracleEvidenceV1
    | G17ForwardObservationLogicalV1
    | G17TerminalEnvelopeLogicalV1
    | G17GenericVMInterpreterV1
    | G17CountedVMBytecodeV1
    | G17CountedVMOperandV1
    | G17PackagedExecutableV1
)


_TYPED_LOGICAL_CLASSES: Final = (
    G17SemanticTopologyV1,
    G17RealizationGaugeV1,
    G17ChronologicalPosePreimageV1,
    G17PopulationSharingV1,
    G17EntropyContextV1,
    G17AnalyticResidualOwnershipV1,
    G17LearnedResidualOwnershipV1,
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17ForwardObservationLogicalV1,
    G17TerminalEnvelopeLogicalV1,
    G17GenericVMInterpreterV1,
    G17CountedVMBytecodeV1,
    G17CountedVMOperandV1,
    G17PackagedExecutableV1,
)

_OWNERSHIP_KIND_BY_CLASS: Final = {
    G17SemanticTopologyV1: G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
    G17RealizationGaugeV1: G17LogicalOwnershipKindV1.REALIZATION_GAUGE,
    G17ChronologicalPosePreimageV1: G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE,
    G17PopulationSharingV1: G17LogicalOwnershipKindV1.POPULATION_SHARED,
    G17EntropyContextV1: G17LogicalOwnershipKindV1.ENTROPY_CONTEXT,
    G17AnalyticResidualOwnershipV1: G17LogicalOwnershipKindV1.ANALYTIC_RESIDUAL,
    G17LearnedResidualOwnershipV1: G17LogicalOwnershipKindV1.LEARNED_RESIDUAL,
    G17EncoderOnlyTeacherOracleEvidenceV1: G17LogicalOwnershipKindV1.ENCODER_EVIDENCE,
    G17ForwardObservationLogicalV1: G17LogicalOwnershipKindV1.ENCODER_EVIDENCE,
    G17TerminalEnvelopeLogicalV1: G17LogicalOwnershipKindV1.TERMINAL_PROTOCOL,
    G17GenericVMInterpreterV1: G17LogicalOwnershipKindV1.GENERIC_DECODER,
    G17CountedVMBytecodeV1: G17LogicalOwnershipKindV1.COUNTED_INSTRUCTION,
    G17CountedVMOperandV1: G17LogicalOwnershipKindV1.COUNTED_OPERAND,
    G17PackagedExecutableV1: G17LogicalOwnershipKindV1.PACKAGED_EXECUTABLE,
}


@dataclass(frozen=True, slots=True)
class G17LogicalOwnershipV1:
    owner_id: str
    ownership_kind: G17LogicalOwnershipKindV1
    value: G17TypedLogicalValueV1 = field(repr=False)
    functional_identity: G17FunctionalQuotientIdentityV1 | None = None
    parameter_spelling: G17ParameterSpellingIdentityV1 | None = None

    def __post_init__(self) -> None:
        _ascii(self.owner_id, name="logical owner ID")
        if type(self.value) not in _TYPED_LOGICAL_CLASSES:
            raise G17CompilerPlacementError("logical ownership requires one explicit non-aliasing value type")
        if type(self.ownership_kind) is not G17LogicalOwnershipKindV1:
            raise G17CompilerPlacementError("logical ownership kind is not typed")
        if _OWNERSHIP_KIND_BY_CLASS[type(self.value)] is not self.ownership_kind:
            raise G17CompilerPlacementError("logical ownership kind cross-casts another value type")
        if (
            self.functional_identity is not None
            and type(self.functional_identity) is not G17FunctionalQuotientIdentityV1
        ):
            raise G17CompilerPlacementError("functional identity is not its exact type")
        if self.parameter_spelling is not None and type(self.parameter_spelling) is not G17ParameterSpellingIdentityV1:
            raise G17CompilerPlacementError("parameter spelling is not its exact type")
        if (
            self.ownership_kind
            in {
                G17LogicalOwnershipKindV1.COUNTED_INSTRUCTION,
                G17LogicalOwnershipKindV1.COUNTED_OPERAND,
                G17LogicalOwnershipKindV1.PACKAGED_EXECUTABLE,
            }
            and self.parameter_spelling is None
        ):
            raise G17CompilerPlacementError("counted instruction/operand/executable requires exact spelling identity")

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"G17-LOGICAL-OWNERSHIP-V1\0")
        digest.update(self.owner_id.encode("ascii") + b"\0")
        digest.update(self.ownership_kind.value.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(self.value.identity_sha256))
        if self.functional_identity is not None:
            digest.update(bytes.fromhex(self.functional_identity.identity_sha256))
        if self.parameter_spelling is not None:
            digest.update(bytes.fromhex(self.parameter_spelling.identity_sha256))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class G17PhysicalCodingGroupV1:
    """One exact charged range, possibly jointly coding several logical owners."""

    group_id: str
    exact_archive_bytes: bytes = field(repr=False)
    member_name: str
    exact_member_bytes: bytes = field(repr=False)
    archive_offset: int
    exact_range_bytes: bytes = field(repr=False)
    coder_owner: str
    container_owner: str
    receiver_consumer: str
    receiver_operation: str
    logical_owner_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _ascii(self.group_id, name="physical coding group ID")
        archive = _exact_bytes(self.exact_archive_bytes, name="coding-group archive")
        member = _exact_bytes(self.exact_member_bytes, name="coding-group member")
        encoded = _exact_bytes(self.exact_range_bytes, name="coding-group range")
        for name in ("member_name", "coder_owner", "container_owner", "receiver_consumer", "receiver_operation"):
            _ascii(getattr(self, name), name=name)
        if type(self.archive_offset) is not int or self.archive_offset < 0:
            raise G17CompilerPlacementError("coding-group archive offset is invalid")
        stop = self.archive_offset + len(encoded)
        if stop > len(archive) or archive[self.archive_offset : stop] != encoded:
            raise G17CompilerPlacementError("coding-group exact charged range differs from retained archive bytes")
        if (
            type(self.logical_owner_ids) is not tuple
            or not self.logical_owner_ids
            or len(self.logical_owner_ids) != len(set(self.logical_owner_ids))
        ):
            raise G17CompilerPlacementError("coding group requires unique logical owners")
        for owner_id in self.logical_owner_ids:
            _ascii(owner_id, name="coding-group logical owner")
        try:
            with zipfile.ZipFile(io.BytesIO(archive), "r") as opened:
                names = opened.namelist()
                if names.count(self.member_name) != 1:
                    raise G17CompilerPlacementError("archive member mapping is missing or duplicated")
                if opened.read(self.member_name) != member:
                    raise G17CompilerPlacementError("retained member bytes differ from actual archive member")
        except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
            raise G17CompilerPlacementError("coding-group archive failed actual ZIP reopen") from exc

    @property
    def byte_length(self) -> int:
        return len(self.exact_range_bytes)

    @property
    def archive_sha256(self) -> str:
        return _sha256(self.exact_archive_bytes)

    @property
    def member_sha256(self) -> str:
        return _sha256(self.exact_member_bytes)

    @property
    def range_sha256(self) -> str:
        return _sha256(self.exact_range_bytes)


@dataclass(frozen=True, slots=True)
class G17CompilerPlacementRecordV1:
    logical_owner: G17LogicalOwnershipV1
    scientific_role: G17ScientificRoleV1
    semantic_role: G17SemanticStreamRoleV1
    recursion_coordinate: G17RecursionCoordinateV1
    placement_class: G17PlacementClassV1
    artifact_class: G17ArtifactClassV1
    payload_class: str
    physical_coding_group_id: str | None
    video_specific_derivation: bool
    packaged_inside_archive: bool = False
    target_selected_constant: bool = False

    def __post_init__(self) -> None:
        if type(self.logical_owner) is not G17LogicalOwnershipV1:
            raise G17CompilerPlacementError("placement requires typed logical ownership")
        if type(self.scientific_role) is not G17ScientificRoleV1:
            raise G17CompilerPlacementError("scientific role is not typed")
        if type(self.semantic_role) is not G17SemanticStreamRoleV1:
            raise G17CompilerPlacementError("semantic stream role is not typed")
        if type(self.recursion_coordinate) is not G17RecursionCoordinateV1:
            raise G17CompilerPlacementError("recursion coordinate is not typed")
        if type(self.placement_class) is not G17PlacementClassV1:
            raise G17CompilerPlacementError("placement class is not typed")
        if type(self.artifact_class) is not G17ArtifactClassV1:
            raise G17CompilerPlacementError("artifact class is not typed")
        _ascii(self.payload_class, name="payload class")
        counted = self.placement_class in {
            G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
            G17PlacementClassV1.COUNTED_PACKAGED_EXECUTABLE,
        }
        if counted != (self.physical_coding_group_id is not None):
            raise G17CompilerPlacementError("counted placement must name exactly one physical coding group")
        if self.physical_coding_group_id is not None:
            _ascii(self.physical_coding_group_id, name="physical coding group foreign key")
        if (
            self.payload_class in _FORBIDDEN_PAYLOAD_CLASSES
            and self.placement_class is not G17PlacementClassV1.ENCODER_ONLY_EVIDENCE
        ):
            raise G17CompilerPlacementError("teacher/oracle/scorer/public payload escaped encoder-only evidence")
        if self.target_selected_constant and self.placement_class is G17PlacementClassV1.GENERIC_DECODER_FREE:
            raise G17CompilerPlacementError("target-selected constant was falsely claimed generic/free")
        if self.packaged_inside_archive and self.placement_class is G17PlacementClassV1.GENERIC_DECODER_FREE:
            raise G17CompilerPlacementError("physically packaged executable/table was falsely claimed free")
        expected_home = {
            G17ArtifactClassV1.GENERIC_DETERMINISTIC_MECHANISM: G17PlacementClassV1.GENERIC_DECODER_FREE,
            G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC: G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
            G17ArtifactClassV1.ENCODER_ONLY_EVIDENCE: G17PlacementClassV1.ENCODER_ONLY_EVIDENCE,
            G17ArtifactClassV1.PACKAGED_EXECUTABLE_OR_TABLE: G17PlacementClassV1.COUNTED_PACKAGED_EXECUTABLE,
        }[self.artifact_class]
        if self.placement_class is not expected_home:
            raise G17CompilerPlacementError("artifact class and placement class disagree")
        if self.artifact_class is G17ArtifactClassV1.GENERIC_DETERMINISTIC_MECHANISM and self.video_specific_derivation:
            raise G17CompilerPlacementError("video-specific mechanism was falsely classified generic")
        if (
            self.artifact_class is G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC
            and not self.video_specific_derivation
        ):
            raise G17CompilerPlacementError("counted video statistic omitted video-specific derivation truth")

    @property
    def object_identity_sha256(self) -> str:
        return self.logical_owner.identity_sha256


@dataclass(frozen=True, slots=True)
class G17CompilerPlacementManifestV1:
    records: tuple[G17CompilerPlacementRecordV1, ...]
    coding_groups: tuple[G17PhysicalCodingGroupV1, ...]
    expected_object_identities: tuple[str, ...]
    exact_archive_bytes: bytes = field(repr=False)
    member_name: str
    exact_member_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.records) is not tuple
            or not self.records
            or any(type(row) is not G17CompilerPlacementRecordV1 for row in self.records)
        ):
            raise G17CompilerPlacementError("placement manifest requires typed incidence rows")
        if type(self.coding_groups) is not tuple or any(
            type(group) is not G17PhysicalCodingGroupV1 for group in self.coding_groups
        ):
            raise G17CompilerPlacementError("placement manifest coding groups are not typed")
        archive = _exact_bytes(self.exact_archive_bytes, name="placement archive")
        member = _exact_bytes(self.exact_member_bytes, name="placement member")
        _ascii(self.member_name, name="placement member name")
        expected_ids = set(self.expected_object_identities)
        if len(expected_ids) != len(self.expected_object_identities):
            raise G17CompilerPlacementError("expected ownership coverage contains aliases")
        owner_instances: dict[str, G17LogicalOwnershipV1] = {}
        incidences: set[tuple[str, str, str, str, str | None]] = set()
        for row in self.records:
            owner_id = row.logical_owner.owner_id
            prior = owner_instances.setdefault(owner_id, row.logical_owner)
            if prior is not row.logical_owner:
                raise G17CompilerPlacementError("one logical owner ID refers to multiple objects")
            key = (
                owner_id,
                row.scientific_role.value,
                row.semantic_role.value,
                f"{row.recursion_coordinate.namespace.value}:{row.recursion_coordinate.stage}",
                row.physical_coding_group_id,
            )
            if key in incidences:
                raise G17CompilerPlacementError("duplicate scientific/semantic/recursion incidence")
            incidences.add(key)
        actual_ids = {owner.identity_sha256 for owner in owner_instances.values()}
        if actual_ids != expected_ids:
            raise G17CompilerPlacementError("placement manifest has missing, extra, or unclassified logical owners")
        groups = {group.group_id: group for group in self.coding_groups}
        if len(groups) != len(self.coding_groups):
            raise G17CompilerPlacementError("physical coding group IDs are duplicated")
        referenced_by_group: dict[str, set[str]] = {group_id: set() for group_id in groups}
        for row in self.records:
            if row.physical_coding_group_id is not None:
                if row.physical_coding_group_id not in groups:
                    raise G17CompilerPlacementError("placement row references an absent coding group")
                referenced_by_group[row.physical_coding_group_id].add(row.logical_owner.owner_id)
        ranges: list[tuple[int, int, str]] = []
        for group in self.coding_groups:
            if (
                group.exact_archive_bytes != archive
                or group.member_name != self.member_name
                or group.exact_member_bytes != member
            ):
                raise G17CompilerPlacementError("coding group belongs to another archive/member object")
            if referenced_by_group[group.group_id] != set(group.logical_owner_ids):
                raise G17CompilerPlacementError("coding-group many-to-many logical incidence is dead or incomplete")
            ranges.append((group.archive_offset, group.archive_offset + group.byte_length, group.group_id))
        ordered_ranges = sorted(ranges)
        cursor = 0
        for start, stop, _ in ordered_ranges:
            if start != cursor:
                kind = "overlap" if start < cursor else "gap"
                raise G17CompilerPlacementError(f"physical coding groups contain a {kind}")
            cursor = stop
        if cursor != len(archive):
            raise G17CompilerPlacementError("physical coding groups do not cover the exact counted archive")

    @property
    def manifest_sha256(self) -> str:
        digest = hashlib.sha256(b"G17-COMPILER-PLACEMENT-MANIFEST-V1\0")
        digest.update(bytes.fromhex(_sha256(self.exact_archive_bytes)))
        digest.update(self.member_name.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(_sha256(self.exact_member_bytes)))
        for row in sorted(
            self.records,
            key=lambda item: (
                item.logical_owner.owner_id,
                item.scientific_role.value,
                item.semantic_role.value,
                item.recursion_coordinate.namespace.value,
                item.recursion_coordinate.stage,
                item.physical_coding_group_id or "",
            ),
        ):
            digest.update(bytes.fromhex(row.logical_owner.identity_sha256))
            digest.update(row.scientific_role.value.encode("ascii") + b"\0")
            digest.update(row.semantic_role.value.encode("ascii") + b"\0")
            digest.update(row.recursion_coordinate.namespace.value.encode("ascii") + b"\0")
            digest.update(row.recursion_coordinate.stage.encode("ascii") + b"\0")
            digest.update(row.placement_class.value.encode("ascii") + b"\0")
            digest.update(row.artifact_class.value.encode("ascii") + b"\0")
            digest.update((row.physical_coding_group_id or "").encode("ascii") + b"\0")
        for group in sorted(self.coding_groups, key=lambda item: item.group_id):
            digest.update(group.group_id.encode("ascii") + b"\0")
            digest.update(group.archive_offset.to_bytes(8, "big"))
            digest.update(group.byte_length.to_bytes(8, "big"))
            digest.update(bytes.fromhex(group.range_sha256))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class G17PairPopulationV1:
    global_pair_ids: tuple[int, ...]
    source_pair_ids: tuple[int, ...]
    v9_pair_coordinates: tuple[int, ...]
    pbr_pair_coordinates: tuple[int, ...]
    obligation_ir_coordinates: tuple[int, ...]
    v10_local_coordinates: tuple[int, ...]

    def __post_init__(self) -> None:
        columns = (
            self.global_pair_ids,
            self.source_pair_ids,
            self.v9_pair_coordinates,
            self.pbr_pair_coordinates,
            self.obligation_ir_coordinates,
            self.v10_local_coordinates,
        )
        if any(type(column) is not tuple or not column for column in columns):
            raise G17CompilerPlacementError("pair population coordinate columns must be nonempty tuples")
        count = len(self.source_pair_ids)
        if any(len(column) != count for column in columns):
            raise G17CompilerPlacementError("pair population coordinate columns have different cardinality")
        if any(any(type(value) is not int or not 0 <= value <= 0xFFFF for value in column) for column in columns):
            raise G17CompilerPlacementError("pair population coordinates must be exact uint16 values")
        if any(len(set(column)) != count for column in columns):
            raise G17CompilerPlacementError("pair population map is not bijective")
        if self.global_pair_ids != tuple(range(self.global_pair_ids[0], self.global_pair_ids[0] + count)):
            raise G17CompilerPlacementError("global pair IDs are not canonical contiguous order")
        if self.source_pair_ids != self.global_pair_ids or self.v9_pair_coordinates != self.source_pair_ids:
            raise G17CompilerPlacementError("global/source/V9 coordinate order drifted")
        canonical_local = tuple(range(count))
        if (
            self.pbr_pair_coordinates != canonical_local
            or self.obligation_ir_coordinates != canonical_local
            or self.v10_local_coordinates != canonical_local
        ):
            raise G17CompilerPlacementError("PBR/IR/V10 local mappings are not canonical bijections")

    @property
    def binding_sha256(self) -> str:
        digest = hashlib.sha256(b"G17-PAIR-POPULATION-V1\0")
        for name, column in (
            ("global", self.global_pair_ids),
            ("source", self.source_pair_ids),
            ("v9", self.v9_pair_coordinates),
            ("pbr", self.pbr_pair_coordinates),
            ("ir", self.obligation_ir_coordinates),
            ("v10", self.v10_local_coordinates),
        ):
            digest.update(name.encode("ascii") + b"\0")
            digest.update(struct.pack(">" + "H" * len(column), *column))
        return digest.hexdigest()


class G17ObligationCoverageModeV1(StrEnum):
    COMPLETE = "COMPLETE"
    SPARSE_OWNED = "SPARSE_OWNED"


@dataclass(frozen=True, slots=True)
class G17ObligationCoordinateV1:
    pair_id: int
    coordinate_id: str

    def __post_init__(self) -> None:
        if type(self.pair_id) is not int or not 0 <= self.pair_id <= 0xFFFF:
            raise G17CompilerPlacementError("obligation pair ID is invalid")
        _ascii(self.coordinate_id, name="obligation coordinate ID")

    @property
    def identity(self) -> tuple[int, str]:
        return (self.pair_id, self.coordinate_id)


@dataclass(frozen=True, slots=True)
class G17SparseObligationOwnerV1:
    obligation: G17ObligationCoordinateV1
    physical_coding_group_id: str
    receiver_consumer: str
    receiver_operation: str

    def __post_init__(self) -> None:
        if type(self.obligation) is not G17ObligationCoordinateV1:
            raise G17CompilerPlacementError("sparse owner requires an exact obligation coordinate")
        for name in ("physical_coding_group_id", "receiver_consumer", "receiver_operation"):
            _ascii(getattr(self, name), name=f"sparse owner {name}")


@dataclass(frozen=True, slots=True)
class G17ObligationCoverageV1:
    population: G17PairPopulationV1
    mode: G17ObligationCoverageModeV1
    obligation_universe: tuple[G17ObligationCoordinateV1, ...]
    predictor_owned: tuple[G17ObligationCoordinateV1, ...]
    sparse_owned: tuple[G17SparseObligationOwnerV1, ...]

    def __post_init__(self) -> None:
        if type(self.population) is not G17PairPopulationV1 or type(self.mode) is not G17ObligationCoverageModeV1:
            raise G17CompilerPlacementError("obligation coverage requires exact population/mode types")
        if type(self.obligation_universe) is not tuple or not self.obligation_universe:
            raise G17CompilerPlacementError("obligation universe must be explicit and nonempty")
        if any(type(item) is not G17ObligationCoordinateV1 for item in self.obligation_universe):
            raise G17CompilerPlacementError("obligation universe contains an untyped coordinate")
        universe = tuple(item.identity for item in self.obligation_universe)
        if len(universe) != len(set(universe)) or any(
            item.pair_id not in self.population.source_pair_ids for item in self.obligation_universe
        ):
            raise G17CompilerPlacementError("obligation universe is duplicate or outside PairPopulation")
        if any(type(item) is not G17ObligationCoordinateV1 for item in self.predictor_owned):
            raise G17CompilerPlacementError("predictor-owned coverage contains an untyped coordinate")
        if any(type(item) is not G17SparseObligationOwnerV1 for item in self.sparse_owned):
            raise G17CompilerPlacementError("sparse-owned coverage contains an untyped receiver owner")
        predictor_ids = tuple(item.identity for item in self.predictor_owned)
        sparse_ids = tuple(item.obligation.identity for item in self.sparse_owned)
        if len(predictor_ids) != len(set(predictor_ids)) or len(sparse_ids) != len(set(sparse_ids)):
            raise G17CompilerPlacementError("one obligation has duplicate ownership")
        if set(predictor_ids) & set(sparse_ids):
            raise G17CompilerPlacementError("predictor and sparse owners overlap")
        if set(predictor_ids) | set(sparse_ids) != set(universe):
            raise G17CompilerPlacementError("obligation coverage has a missing or foreign coordinate")
        if self.mode is G17ObligationCoverageModeV1.COMPLETE and sparse_ids:
            raise G17CompilerPlacementError("complete coverage cannot carry residual sparse owners")
        if self.mode is G17ObligationCoverageModeV1.SPARSE_OWNED and not sparse_ids:
            raise G17CompilerPlacementError("sparse-owned coverage must bind at least one live receiver owner")


class G17PoseOwnershipV1(StrEnum):
    V9_POSE6 = "V9_POSE6"
    FRAME_ZERO_RESIDUAL = "FRAME_ZERO_RESIDUAL"
    REVERSE_CAUSAL_FRAME0_FROM_EXACT_Y1 = "REVERSE_CAUSAL_FRAME0_FROM_EXACT_Y1"


@dataclass(frozen=True, slots=True)
class G17PosePreimageOwnershipV1:
    population: G17PairPopulationV1
    ownership_by_pair: tuple[G17PoseOwnershipV1, ...]
    physical_coding_group_id_by_pair: tuple[str, ...]
    receiver_operation_by_pair: tuple[str, ...]
    explicit_preimage_packet: G17ReopenedEvidencePacketV1 | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.population) is not G17PairPopulationV1:
            raise G17CompilerPlacementError("pose ownership requires exact population")
        if len(self.ownership_by_pair) != len(self.population.source_pair_ids) or any(
            type(value) is not G17PoseOwnershipV1 for value in self.ownership_by_pair
        ):
            raise G17CompilerPlacementError("pose ownership must choose exactly one owner per pair")
        if len(self.physical_coding_group_id_by_pair) != len(self.ownership_by_pair) or len(
            self.receiver_operation_by_pair
        ) != len(self.ownership_by_pair):
            raise G17CompilerPlacementError("Pose ownership lacks a counted group/receiver operation per pair")
        for group_id, operation in zip(
            self.physical_coding_group_id_by_pair,
            self.receiver_operation_by_pair,
            strict=True,
        ):
            _ascii(group_id, name="Pose physical coding group")
            _ascii(operation, name="Pose receiver operation")
        requires_preimage = any(owner is not G17PoseOwnershipV1.V9_POSE6 for owner in self.ownership_by_pair)
        if requires_preimage != (self.explicit_preimage_packet is not None):
            raise G17CompilerPlacementError(
                "frame-zero/reverse-causal ownership requires exact strict-parsed preimage bytes"
            )
        if self.explicit_preimage_packet is not None:
            if type(self.explicit_preimage_packet) is not G17ReopenedEvidencePacketV1:
                raise G17CompilerPlacementError("explicit preimage evidence is opaque rather than strict-parsed")
            self.explicit_preimage_packet.reopen()

    @property
    def ownership_receipt_sha256(self) -> str:
        payload = {
            "population_sha256": self.population.binding_sha256,
            "owners": [item.value for item in self.ownership_by_pair],
            "coding_groups": list(self.physical_coding_group_id_by_pair),
            "receiver_operations": list(self.receiver_operation_by_pair),
            "explicit_preimage_packet_sha256": None
            if self.explicit_preimage_packet is None
            else self.explicit_preimage_packet.packet_sha256,
        }
        return _sha256(_canonical_json(payload))


class G17R10ConstraintV1(StrEnum):
    AMPLITUDE = "AMPLITUDE"
    FREQUENCY = "FREQUENCY"
    PHASE = "PHASE"
    CONTRAST = "CONTRAST"
    CHANNEL_ENERGY = "CHANNEL_ENERGY"
    TEXTURE = "TEXTURE"
    MULTIPLE_SHOOTING = "MULTIPLE_SHOOTING"
    FROZEN_FEATURE = "FROZEN_FEATURE"


class G17FrameRoleV1(StrEnum):
    Y0 = "Y0"
    Y1 = "Y1"
    CHRONOLOGICAL_PAIR = "CHRONOLOGICAL_PAIR"


@dataclass(frozen=True, slots=True)
class G17R10ConstraintCoordinateV1:
    constraint: G17R10ConstraintV1
    population: G17PairPopulationV1
    frame_role: G17FrameRoleV1
    scientific_role: G17ScientificRoleV1
    semantic_role: G17SemanticStreamRoleV1
    exact_value_bytes: bytes = field(repr=False)
    support: tuple[G17ObligationCoordinateV1, ...]
    tolerance: float
    exact_frozen_block_bytes: bytes = field(repr=False)
    exact_chronology_receipt_bytes: bytes = field(repr=False)
    generic_receiver_operation: str
    physical_coding_group_id: str
    counted_operand_offset: int
    counted_operand_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.constraint) is not G17R10ConstraintV1:
            raise G17CompilerPlacementError("R10 constraint discriminator is not typed")
        if type(self.population) is not G17PairPopulationV1:
            raise G17CompilerPlacementError("R10 constraint lacks exact PairPopulation")
        if type(self.frame_role) is not G17FrameRoleV1:
            raise G17CompilerPlacementError("R10 frame role is not typed")
        if (
            type(self.scientific_role) is not G17ScientificRoleV1
            or type(self.semantic_role) is not G17SemanticStreamRoleV1
        ):
            raise G17CompilerPlacementError("R10 scientific/semantic incidence is not typed")
        _exact_bytes(self.exact_value_bytes, name=f"R10 {self.constraint.value} value")
        _exact_bytes(self.exact_frozen_block_bytes, name="R10 frozen block identity")
        _exact_bytes(self.exact_chronology_receipt_bytes, name="R10 chronology receipt")
        operand = _exact_bytes(self.counted_operand_bytes, name="R10 counted operand")
        if (
            type(self.support) is not tuple
            or not self.support
            or any(type(item) is not G17ObligationCoordinateV1 for item in self.support)
        ):
            raise G17CompilerPlacementError("R10 support must be explicit typed coordinates")
        if len({item.identity for item in self.support}) != len(self.support) or any(
            item.pair_id not in self.population.source_pair_ids for item in self.support
        ):
            raise G17CompilerPlacementError("R10 support is duplicate or outside PairPopulation")
        if type(self.tolerance) is not float or not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise G17CompilerPlacementError("R10 tolerance must be finite nonnegative float")
        _ascii(self.generic_receiver_operation, name="R10 generic receiver operation")
        _ascii(self.physical_coding_group_id, name="R10 physical coding group")
        if type(self.counted_operand_offset) is not int or self.counted_operand_offset < 0:
            raise G17CompilerPlacementError("R10 counted operand offset is invalid")
        if self.counted_operand_offset + len(operand) > 1 << 31:
            raise G17CompilerPlacementError("R10 counted operand span exceeds receiver bound")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "constraint": self.constraint.value,
                    "population_sha256": self.population.binding_sha256,
                    "frame_role": self.frame_role.value,
                    "scientific_role": self.scientific_role.value,
                    "semantic_role": self.semantic_role.value,
                    "value_sha256": _sha256(self.exact_value_bytes),
                    "support": [list(item.identity) for item in self.support],
                    "tolerance": self.tolerance,
                    "frozen_block_sha256": _sha256(self.exact_frozen_block_bytes),
                    "chronology_receipt_sha256": _sha256(self.exact_chronology_receipt_bytes),
                    "generic_receiver_operation": self.generic_receiver_operation,
                    "physical_coding_group_id": self.physical_coding_group_id,
                    "counted_operand_offset": self.counted_operand_offset,
                    "counted_operand_bytes": len(self.counted_operand_bytes),
                    "counted_operand_sha256": _sha256(self.counted_operand_bytes),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class G17R10ProsodyFeatureRelayV1:
    constraint_coordinates: tuple[G17R10ConstraintCoordinateV1, ...]

    def __post_init__(self) -> None:
        if type(self.constraint_coordinates) is not tuple or any(
            type(item) is not G17R10ConstraintCoordinateV1 for item in self.constraint_coordinates
        ):
            raise G17CompilerPlacementError("R10 relay requires exact typed constraint coordinates")
        if tuple(item.constraint for item in self.constraint_coordinates) != tuple(G17R10ConstraintV1):
            raise G17CompilerPlacementError("R10 relay must cover every constraint once in canonical order")
        population = self.constraint_coordinates[0].population
        if any(item.population is not population for item in self.constraint_coordinates):
            raise G17CompilerPlacementError("R10 relay coordinates retain different PairPopulation objects")

    def require_receiver_consumption(
        self,
        evidence: G17CompilerPlacementManifestV1 | tuple[G17R10ConstraintV1, ...],
    ) -> None:
        """Validate operand spans, then refuse because V1 has no R10 physics op.

        A tuple of enum names is deliberately accepted only as an argument type
        so legacy caller-attestation probes receive the named typed blocker.
        """

        if type(evidence) is tuple:
            raise G17CompilerBlocker(
                G17CompilerBlockerCodeV1.G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED,
                "repeating constraint names is not receiver execution evidence",
            )
        if type(evidence) is not G17CompilerPlacementManifestV1:
            raise G17CompilerPlacementError("R10 closure requires the exact placement manifest")
        groups = {group.group_id: group for group in evidence.coding_groups}
        for row in self.constraint_coordinates:
            group = groups.get(row.physical_coding_group_id)
            if group is None:
                raise G17CompilerPlacementError("R10 operand names an absent physical coding group")
            stop = row.counted_operand_offset + len(row.counted_operand_bytes)
            if (
                stop > len(group.exact_member_bytes)
                or group.exact_member_bytes[row.counted_operand_offset : stop] != row.counted_operand_bytes
            ):
                raise G17CompilerPlacementError("R10 operand span differs from exact charged bytes")
            if group.receiver_operation != row.generic_receiver_operation:
                raise G17CompilerPlacementError("R10 constraint and coding group name different receiver operations")
        raise G17CompilerBlocker(
            G17CompilerBlockerCodeV1.G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED,
            "V1 byte VM has no topology/feature-relay/multiple-shooting physics operation",
        )


class G17VMOpcodeV1(IntEnum):
    PUSH_INPUT_SECTION = 1
    PUSH_LITERAL = 2
    CONCAT = 3
    SLICE = 4
    XOR = 5
    REPEAT = 6
    EMIT_SECTION = 7
    ASSERT_SHA256 = 8


@dataclass(frozen=True, slots=True)
class G17VMOperandV1:
    exact_bytes: bytes

    def __post_init__(self) -> None:
        _exact_bytes(self.exact_bytes, name="VM operand", nonempty=False)


@dataclass(frozen=True, slots=True)
class G17DeterministicReconstructionProgramV1:
    """Counted video-specific bytecode and operands for the generic stack VM."""

    bytecode: bytes
    operands: tuple[G17VMOperandV1, ...]

    def __post_init__(self) -> None:
        _exact_bytes(self.bytecode, name="VM bytecode")
        if type(self.operands) is not tuple or any(type(item) is not G17VMOperandV1 for item in self.operands):
            raise G17CompilerPlacementError("VM operands must be one exact typed tuple")
        for raw in self.bytecode:
            try:
                G17VMOpcodeV1(raw)
            except ValueError as exc:
                raise G17CompilerBlocker(
                    G17CompilerBlockerCodeV1.G17_UNSUPPORTED_TOPOLOGY_OR_CONSTRAINT_VM_OPERATION,
                    f"unsupported VM opcode {raw}; no topology/constraint/R10 no-op fallback exists",
                ) from exc

    @property
    def counted_payload_sha256(self) -> str:
        digest = hashlib.sha256(b"G17-COUNTED-VM-PROGRAM-V1\0")
        digest.update(self.bytecode)
        for operand in self.operands:
            digest.update(len(operand.exact_bytes).to_bytes(4, "big"))
            digest.update(operand.exact_bytes)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class G17VMExecutionReceiptV1:
    program_sha256: str
    input_root_sha256: str
    output_root_sha256: str
    emitted_sections: tuple[tuple[str, str, int], ...]
    deterministic_double_execution: Literal[True] = True


def _vm_execute_once(
    program: G17DeterministicReconstructionProgramV1,
    *,
    input_sections: dict[str, bytes],
) -> tuple[dict[str, bytes], G17VMExecutionReceiptV1]:
    if type(program) is not G17DeterministicReconstructionProgramV1:
        raise G17CompilerPlacementError("VM execution requires exact program type")
    if type(input_sections) is not dict or any(
        type(key) is not str or not key or type(value) is not bytes for key, value in input_sections.items()
    ):
        raise G17CompilerPlacementError("VM input sections must be exact ASCII-name to bytes mapping")
    stack: list[bytes] = []
    emitted: dict[str, bytes] = {}
    operand_index = 0

    def take_operand() -> bytes:
        nonlocal operand_index
        if operand_index >= len(program.operands):
            raise G17CompilerPlacementError("VM bytecode consumed beyond counted operand stream")
        value = program.operands[operand_index].exact_bytes
        operand_index += 1
        return value

    for raw_opcode in program.bytecode:
        opcode = G17VMOpcodeV1(raw_opcode)
        if opcode is G17VMOpcodeV1.PUSH_INPUT_SECTION:
            selector = take_operand()
            try:
                name = selector.decode("ascii")
            except UnicodeDecodeError as exc:
                raise G17CompilerPlacementError("VM input selector is not ASCII") from exc
            if name not in input_sections:
                raise G17CompilerPlacementError("VM selected an absent exact input section")
            stack.append(input_sections[name])
        elif opcode is G17VMOpcodeV1.PUSH_LITERAL:
            stack.append(take_operand())
        elif opcode is G17VMOpcodeV1.CONCAT:
            count_raw = take_operand()
            if len(count_raw) != 2:
                raise G17CompilerPlacementError("VM CONCAT count must be exact uint16 bytes")
            (count,) = struct.unpack(">H", count_raw)
            if count < 1 or count > len(stack):
                raise G17CompilerPlacementError("VM CONCAT stack arity is invalid")
            items = stack[-count:]
            del stack[-count:]
            stack.append(b"".join(items))
        elif opcode is G17VMOpcodeV1.SLICE:
            bounds = take_operand()
            if len(bounds) != 8 or not stack:
                raise G17CompilerPlacementError("VM SLICE requires stack value and exact uint32 bounds")
            start, stop = struct.unpack(">II", bounds)
            if start > stop or stop > len(stack[-1]):
                raise G17CompilerPlacementError("VM SLICE bounds escaped the exact source bytes")
            stack[-1] = stack[-1][start:stop]
        elif opcode is G17VMOpcodeV1.XOR:
            if len(stack) < 2:
                raise G17CompilerPlacementError("VM XOR requires two stack values")
            right = stack.pop()
            left = stack.pop()
            if len(left) != len(right):
                raise G17CompilerPlacementError("VM XOR operands must have equal byte length")
            stack.append(bytes(a ^ b for a, b in zip(left, right, strict=True)))
        elif opcode is G17VMOpcodeV1.REPEAT:
            count_raw = take_operand()
            if len(count_raw) != 4 or not stack:
                raise G17CompilerPlacementError("VM REPEAT requires stack value and exact uint32 count")
            (count,) = struct.unpack(">I", count_raw)
            if count > 1_000_000 or len(stack[-1]) * count > 1 << 31:
                raise G17CompilerPlacementError("VM REPEAT exceeds deterministic receiver bounds")
            stack[-1] *= count
        elif opcode is G17VMOpcodeV1.EMIT_SECTION:
            selector = take_operand()
            if not stack:
                raise G17CompilerPlacementError("VM EMIT requires one stack value")
            try:
                name = selector.decode("ascii")
            except UnicodeDecodeError as exc:
                raise G17CompilerPlacementError("VM output selector is not ASCII") from exc
            if not name or name in emitted:
                raise G17CompilerPlacementError("VM output section name is empty or duplicated")
            emitted[name] = stack.pop()
        elif opcode is G17VMOpcodeV1.ASSERT_SHA256:
            expected = take_operand()
            if len(expected) != 32 or not stack or hashlib.sha256(stack[-1]).digest() != expected:
                raise G17CompilerPlacementError("VM exact SHA-256 assertion failed")
        else:  # pragma: no cover - enum construction makes this unreachable
            raise G17CompilerBlocker(
                G17CompilerBlockerCodeV1.G17_UNSUPPORTED_TOPOLOGY_OR_CONSTRAINT_VM_OPERATION,
                f"unsupported VM opcode {raw_opcode}",
            )
    if operand_index != len(program.operands) or stack:
        raise G17CompilerPlacementError("VM left unconsumed operands or stack values")
    input_digest = hashlib.sha256(b"G17-VM-INPUT-ROOT-V1\0")
    for name, payload in sorted(input_sections.items()):
        input_digest.update(name.encode("ascii") + b"\0" + bytes.fromhex(_sha256(payload)))
    output_digest = hashlib.sha256(b"G17-VM-OUTPUT-ROOT-V1\0")
    rows: list[tuple[str, str, int]] = []
    for name, payload in emitted.items():
        digest = _sha256(payload)
        rows.append((name, digest, len(payload)))
        output_digest.update(name.encode("ascii") + b"\0" + bytes.fromhex(digest))
    return emitted, G17VMExecutionReceiptV1(
        program_sha256=program.counted_payload_sha256,
        input_root_sha256=input_digest.hexdigest(),
        output_root_sha256=output_digest.hexdigest(),
        emitted_sections=tuple(rows),
    )


def execute_g17_reconstruction_vm(
    program: G17DeterministicReconstructionProgramV1,
    *,
    input_sections: dict[str, bytes],
) -> tuple[dict[str, bytes], G17VMExecutionReceiptV1]:
    first_output, first_receipt = _vm_execute_once(program, input_sections=input_sections)
    second_output, second_receipt = _vm_execute_once(program, input_sections=input_sections)
    if first_output != second_output or first_receipt != second_receipt:
        raise G17CompilerPlacementError("generic reconstruction VM failed deterministic double execution")
    return first_output, first_receipt


@dataclass(frozen=True, slots=True)
class G17ProofDependencyV1:
    domain: G17ProofDependencyDomainV1
    exact_dependency_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.domain) is not G17ProofDependencyDomainV1:
            raise G17CompilerPlacementError("proof dependency domain is not typed")
        _exact_bytes(self.exact_dependency_bytes, name=f"proof dependency {self.domain.value}")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"G17-PROOF-DEPENDENCY-V1\0" + self.domain.value.encode("ascii") + b"\0" + self.exact_dependency_bytes
        )


_PROOF_DEPENDENCY_DOMAINS: Final = {
    G17ProofKindV1.ARCHIVE_DECODE_EQUALITY: frozenset(
        {
            G17ProofDependencyDomainV1.ARCHIVE_BYTES,
            G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING,
            G17ProofDependencyDomainV1.RECEIVER_IMPLEMENTATION,
            G17ProofDependencyDomainV1.RECEIVER_RUNTIME,
            G17ProofDependencyDomainV1.PAIR_ORDER,
            G17ProofDependencyDomainV1.DECODER_EQUALITY_ALGORITHM,
        }
    ),
    G17ProofKindV1.AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE: frozenset(
        {
            G17ProofDependencyDomainV1.ARCHIVE_BYTES,
            G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING,
            G17ProofDependencyDomainV1.PUBLIC_ENTRYPOINT_CHAIN,
            G17ProofDependencyDomainV1.RUNTIME_FILE_CLOSURE,
            G17ProofDependencyDomainV1.AUTH_EVAL_EXECUTION_RECEIPT,
        }
    ),
    G17ProofKindV1.SCORE_OBSERVATION: frozenset(
        {
            G17ProofDependencyDomainV1.ARCHIVE_BYTES,
            G17ProofDependencyDomainV1.DECODE_RECEIPT,
            G17ProofDependencyDomainV1.FROZEN_SCORER,
            G17ProofDependencyDomainV1.SCORER_RUNTIME,
            G17ProofDependencyDomainV1.AXIS_AND_SAMPLE_SCOPE,
        }
    ),
    G17ProofKindV1.BEATS_CURRENT_FRONTIER: frozenset(
        {
            G17ProofDependencyDomainV1.SCORE_RECEIPT,
            G17ProofDependencyDomainV1.SEMANTIC_COMPETITIVE_TARGET,
            G17ProofDependencyDomainV1.POINTER_SNAPSHOT,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class G17ProofDependencySetV1:
    proof_kind: G17ProofKindV1
    dependencies: tuple[G17ProofDependencyV1, ...]
    declared_external_reads: tuple[G17ProofDependencyDomainV1, ...]

    def __post_init__(self) -> None:
        if type(self.proof_kind) is not G17ProofKindV1:
            raise G17CompilerPlacementError("proof kind is not typed")
        if type(self.dependencies) is not tuple or any(
            type(item) is not G17ProofDependencyV1 for item in self.dependencies
        ):
            raise G17CompilerPlacementError("proof dependency set contains untyped evidence")
        domains = tuple(item.domain for item in self.dependencies)
        if len(domains) != len(set(domains)):
            raise G17CompilerPlacementError("proof dependency domain is duplicated")
        if set(domains) != _PROOF_DEPENDENCY_DOMAINS[self.proof_kind]:
            raise G17CompilerPlacementError("proof dependencies are missing, extra, or globally overjoined")
        if domains != tuple(sorted(domains, key=lambda item: item.value)):
            raise G17CompilerPlacementError("proof dependencies are not in canonical domain order")
        if type(self.declared_external_reads) is not tuple or any(
            type(item) is not G17ProofDependencyDomainV1 for item in self.declared_external_reads
        ):
            raise G17CompilerPlacementError("external reads must be explicit dependency domains")
        if len(self.declared_external_reads) != len(set(self.declared_external_reads)) or not set(
            self.declared_external_reads
        ).issubset(set(domains)):
            raise G17CompilerPlacementError("proof producer has an undeclared or duplicate external read")
        if (
            self.proof_kind is not G17ProofKindV1.BEATS_CURRENT_FRONTIER
            and G17ProofDependencyDomainV1.POINTER_SNAPSHOT in domains
        ):
            raise G17CompilerPlacementError("pointer identity leaked into an invariant artifact proof")

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"G17-PROOF-DEPENDENCY-SET-V1\0")
        digest.update(self.proof_kind.value.encode("ascii") + b"\0")
        for item in self.dependencies:
            digest.update(bytes.fromhex(item.identity_sha256))
        for item in self.declared_external_reads:
            digest.update(item.value.encode("ascii") + b"\0")
        return digest.hexdigest()

    def require_observed_external_reads(
        self,
        observed: tuple[G17ProofDependencyDomainV1, ...],
    ) -> None:
        if observed != self.declared_external_reads:
            raise G17CompilerPlacementError("proof producer performed an undeclared external read")


@dataclass(frozen=True, slots=True)
class G17CompetitiveTargetIdentityV1:
    competition_namespace: str
    metric_namespace: str
    selection_policy: str
    exact_evaluator_rules_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("competition_namespace", "metric_namespace", "selection_policy"):
            _ascii(getattr(self, name), name=f"competitive target {name}")
        _exact_bytes(self.exact_evaluator_rules_bytes, name="competitive evaluator rules")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"G17-SEMANTIC-COMPETITIVE-TARGET-V1\0"
            + self.competition_namespace.encode("ascii")
            + b"\0"
            + self.metric_namespace.encode("ascii")
            + b"\0"
            + self.selection_policy.encode("ascii")
            + b"\0"
            + self.exact_evaluator_rules_bytes
        )


@dataclass(frozen=True, slots=True)
class G17PointerSnapshotV1:
    """Volatile admission state, deliberately absent from decode/score proofs."""

    exact_pointer_bytes: bytes = field(repr=False)
    effective_frontier_score: float = field(init=False)

    def __post_init__(self) -> None:
        payload = _exact_bytes(self.exact_pointer_bytes, name="pointer snapshot")
        try:
            value = json.loads(payload.decode("utf-8"))
            score = value["effective_frontier"]["score"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise G17CompilerPlacementError("pointer snapshot lacks effective_frontier.score") from exc
        if type(score) not in {float, int} or not math.isfinite(float(score)):
            raise G17CompilerPlacementError("pointer effective frontier is not finite")
        object.__setattr__(self, "effective_frontier_score", float(score))

    @property
    def snapshot_sha256(self) -> str:
        return _sha256(self.exact_pointer_bytes)


@dataclass(frozen=True, slots=True)
class G17ResearchAuthorityEvidenceV1:
    evidence_receipt: G17ReopenedEvidencePacketV1
    sample_count: int
    axis_label: str
    exact_hardware_identity_bytes: bytes = field(repr=False)
    exact_scorer_identity_bytes: bytes = field(repr=False)
    exact_runtime_identity_bytes: bytes = field(repr=False)
    authority_class: G17AuthorityClassV1 = field(
        default=G17AuthorityClassV1.RESEARCH_ADVISORY,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self.evidence_receipt) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("research authority requires strict-reopened evidence")
        self.evidence_receipt.reopen()
        if type(self.sample_count) is not int or not 1 <= self.sample_count <= 600:
            raise G17CompilerPlacementError("research authority sample count is invalid")
        _ascii(self.axis_label, name="research authority axis")
        for name in (
            "exact_hardware_identity_bytes",
            "exact_scorer_identity_bytes",
            "exact_runtime_identity_bytes",
        ):
            _exact_bytes(getattr(self, name), name=name)

    @property
    def promotion_capable(self) -> Literal[False]:
        return False

    @property
    def authority_sha256(self) -> str:
        return _sha256(
            b"G17-RESEARCH-AUTHORITY-V1\0"
            + bytes.fromhex(self.evidence_receipt.packet_sha256)
            + self.sample_count.to_bytes(2, "big")
            + self.axis_label.encode("ascii")
            + b"\0"
            + self.exact_hardware_identity_bytes
            + self.exact_scorer_identity_bytes
            + self.exact_runtime_identity_bytes
        )


_AUTHORITY_SEAL: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class _G17ExactContestAuthorityEvidenceV1:
    authority_class: G17AuthorityClassV1
    evaluator_receipt: G17ReopenedEvidencePacketV1
    auth_eval_closure: C0BAuthEvalClosureV1
    archive_sha256: str
    decode_receipt_sha256: str
    sample_count: Literal[600]
    exact_hardware_identity_bytes: bytes = field(repr=False)
    exact_scorer_identity_bytes: bytes = field(repr=False)
    exact_runtime_identity_bytes: bytes = field(repr=False)
    exact_evaluation_command_bytes: bytes = field(repr=False)
    exact_evaluation_log_bytes: bytes = field(repr=False)
    total_score: float

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("contest authority has no public constructor")

    def _sealed_init(
        self,
        *,
        seal: object,
        authority_class: G17AuthorityClassV1,
        evaluator_receipt: G17ReopenedEvidencePacketV1,
        auth_eval_closure: C0BAuthEvalClosureV1,
        archive_sha256: str,
        decode_receipt_sha256: str,
        exact_hardware_identity_bytes: bytes,
        exact_scorer_identity_bytes: bytes,
        exact_runtime_identity_bytes: bytes,
        exact_evaluation_command_bytes: bytes,
        exact_evaluation_log_bytes: bytes,
        total_score: float,
    ) -> None:
        if seal is not _AUTHORITY_SEAL:
            raise G17CompilerPlacementError("contest authority is sealed to a reviewed evaluator adapter")
        if authority_class not in {G17AuthorityClassV1.CONTEST_CPU, G17AuthorityClassV1.CONTEST_CUDA}:
            raise G17CompilerPlacementError("contest authority class is invalid")
        if type(evaluator_receipt) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("contest authority lacks exact evaluator receipt")
        evaluator_receipt.reopen()
        if type(auth_eval_closure) is not C0BAuthEvalClosureV1:
            raise G17CompilerPlacementError("contest authority lacks sealed public AuthEvalClosure")
        if (
            evaluator_receipt.exact_packet_bytes
            != auth_eval_closure.public_evaluator_execution_receipt.exact_packet_bytes
        ):
            raise G17CompilerPlacementError("contest authority evaluator receipt differs from AuthEvalClosure")
        for digest in (archive_sha256, decode_receipt_sha256):
            if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise G17CompilerPlacementError("contest authority digest is invalid")
        if (
            archive_sha256 != auth_eval_closure.evaluated_archive_sha256
            or decode_receipt_sha256 != auth_eval_closure.public_decode_receipt_sha256
        ):
            raise G17CompilerPlacementError("contest authority refers to another AuthEvalClosure object")
        for value, name in (
            (exact_hardware_identity_bytes, "contest hardware"),
            (exact_scorer_identity_bytes, "contest scorer"),
            (exact_runtime_identity_bytes, "contest runtime"),
            (exact_evaluation_command_bytes, "contest evaluator command"),
            (exact_evaluation_log_bytes, "contest evaluator log"),
        ):
            _exact_bytes(value, name=name)
        if exact_evaluation_command_bytes != auth_eval_closure.exact_argv_bytes:
            raise G17CompilerPlacementError("contest evaluator command differs from AuthEvalClosure argv")
        if type(total_score) is not float or not math.isfinite(total_score) or total_score < 0.0:
            raise G17CompilerPlacementError("contest total score is invalid")
        for name, value in (
            ("authority_class", authority_class),
            ("evaluator_receipt", evaluator_receipt),
            ("auth_eval_closure", auth_eval_closure),
            ("archive_sha256", archive_sha256),
            ("decode_receipt_sha256", decode_receipt_sha256),
            ("sample_count", 600),
            ("exact_hardware_identity_bytes", exact_hardware_identity_bytes),
            ("exact_scorer_identity_bytes", exact_scorer_identity_bytes),
            ("exact_runtime_identity_bytes", exact_runtime_identity_bytes),
            ("exact_evaluation_command_bytes", exact_evaluation_command_bytes),
            ("exact_evaluation_log_bytes", exact_evaluation_log_bytes),
            ("total_score", total_score),
        ):
            object.__setattr__(self, name, value)

    @property
    def promotion_capable(self) -> Literal[True]:
        return True


class G17ContestCPUAuthorityEvidenceV1(_G17ExactContestAuthorityEvidenceV1):
    """Sealed; no public constructor exists until a reviewed exact-eval adapter lands."""


class G17ContestCUDAAuthorityEvidenceV1(_G17ExactContestAuthorityEvidenceV1):
    """Sealed and independent of the CPU authority variant."""


G17AuthorityEvidenceV1: TypeAlias = (
    G17ResearchAuthorityEvidenceV1 | G17ContestCPUAuthorityEvidenceV1 | G17ContestCUDAAuthorityEvidenceV1
)


def require_g17_exact_contest_authority_adapter() -> None:
    raise G17CompilerBlocker(
        G17CompilerBlockerCodeV1.G17_EXACT_CONTEST_AUTHORITY_ADAPTER_OWED,
        "reviewed exact contest evaluator receipt adapter is not landed; contest authority cannot be constructed",
    )


# C0B lifecycle: every edge retains its exact typed parent object.  Archive,
# decode, and score stages have sealed constructors that actually reopen or
# execute their object; hash-only construction is impossible.


@dataclass(frozen=True, slots=True)
class C0BSourceTruthV1:
    source_bytes: bytes = field(repr=False)
    evaluator_source_bytes: bytes = field(repr=False)
    evaluator_weights_bytes: bytes = field(repr=False)
    evaluator_runtime_bytes: bytes = field(repr=False)
    resize_r_implementation_bytes: bytes = field(repr=False)
    population: G17PairPopulationV1
    target_evidence: G17ReopenedEvidencePacketV1
    originality_declaration_bytes: bytes = field(repr=False)
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.SOURCE_TRUTH, init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_bytes",
            "evaluator_source_bytes",
            "evaluator_weights_bytes",
            "evaluator_runtime_bytes",
            "resize_r_implementation_bytes",
            "originality_declaration_bytes",
        ):
            _exact_bytes(getattr(self, name), name=f"C0B {name}")
        if type(self.population) is not G17PairPopulationV1:
            raise G17CompilerPlacementError("SourceTruth requires exact PairPopulation")
        if type(self.target_evidence) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("SourceTruth target evidence is not strict-reopened")
        self.target_evidence.reopen()

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"C0B-SOURCE-TRUTH-V1\0")
        for value in (
            self.source_bytes,
            self.evaluator_source_bytes,
            self.evaluator_weights_bytes,
            self.evaluator_runtime_bytes,
            self.resize_r_implementation_bytes,
            self.originality_declaration_bytes,
        ):
            digest.update(len(value).to_bytes(8, "big") + value)
        digest.update(bytes.fromhex(self.population.binding_sha256))
        digest.update(bytes.fromhex(self.target_evidence.packet_sha256))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class C0BObligationIRV1:
    source_truth: C0BSourceTruthV1
    obligation_ir_packet: G17ReopenedEvidencePacketV1
    population: G17PairPopulationV1
    coverage: G17ObligationCoverageV1
    pose_preimage_ownership: G17PosePreimageOwnershipV1
    r10_relay: G17R10ProsodyFeatureRelayV1 | None
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.OBLIGATION_IR, init=False)

    def __post_init__(self) -> None:
        if type(self.source_truth) is not C0BSourceTruthV1:
            raise G17CompilerPlacementError("C0B obligation IR requires exact SourceTruth parent")
        if type(self.obligation_ir_packet) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("C0B obligation IR packet is not strict-reopened")
        self.obligation_ir_packet.reopen()
        if self.population is not self.source_truth.population:
            raise G17CompilerPlacementError("C0B obligation IR retains another PairPopulation object")
        if self.coverage.population is not self.population:
            raise G17CompilerPlacementError("C0B obligation coverage retains another population object")
        if self.pose_preimage_ownership.population is not self.population:
            raise G17CompilerPlacementError("C0B Pose ownership retains another population object")
        if self.r10_relay is not None:
            if type(self.r10_relay) is not G17R10ProsodyFeatureRelayV1:
                raise G17CompilerPlacementError("C0B R10 relay is not typed")
            if self.r10_relay.constraint_coordinates[0].population is not self.population:
                raise G17CompilerPlacementError("C0B R10 relay retains another population object")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-OBLIGATION-IR-V1\0"
            + bytes.fromhex(self.source_truth.identity_sha256)
            + bytes.fromhex(self.obligation_ir_packet.packet_sha256)
            + bytes.fromhex(self.population.binding_sha256)
            + bytes.fromhex(self.pose_preimage_ownership.ownership_receipt_sha256)
        )


@dataclass(frozen=True, slots=True)
class C0BRealizedPairV1:
    obligation_ir: C0BObligationIRV1
    camera_y0: np.ndarray = field(repr=False)
    camera_y1: np.ndarray = field(repr=False)
    resize_r_proof: G17ReopenedEvidencePacketV1
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.REALIZED_PAIR, init=False)

    def __post_init__(self) -> None:
        if type(self.obligation_ir) is not C0BObligationIRV1:
            raise G17CompilerPlacementError("C0B realized pair requires exact ObligationIR parent")
        if type(self.resize_r_proof) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("C0B realization lacks strict-reopened R/resize proof")
        self.resize_r_proof.reopen()
        expected_pairs = len(self.obligation_ir.population.source_pair_ids)
        arrays: list[np.ndarray] = []
        for name in ("camera_y0", "camera_y1"):
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != np.uint8 or value.shape[0] != expected_pairs:
                raise G17CompilerPlacementError(f"C0B {name} is not a uint8 population array")
            copied = np.ascontiguousarray(value).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
            arrays.append(copied)
        if arrays[0].shape != arrays[1].shape:
            raise G17CompilerPlacementError("C0B Y0/Y1 shapes differ")

    @property
    def chronology_bytes(self) -> bytes:
        chronology = np.stack((self.camera_y0, self.camera_y1), axis=1)
        return memoryview(np.ascontiguousarray(chronology)).cast("B").tobytes()

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-REALIZED-PAIR-V1\0"
            + bytes.fromhex(self.obligation_ir.identity_sha256)
            + self.chronology_bytes
            + bytes.fromhex(self.resize_r_proof.packet_sha256)
        )


@dataclass(frozen=True, slots=True, init=False)
class C0BArchiveArtifactV1:
    realized_pair: C0BRealizedPairV1
    member_name: str
    member_bytes: bytes = field(repr=False)
    archive_bytes: bytes = field(repr=False)
    member_map: tuple[tuple[str, str, int], ...]
    placement_manifest: G17CompilerPlacementManifestV1
    decoder_program_bytes: bytes = field(repr=False)
    decoder_runtime_bytes: bytes = field(repr=False)
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.ARCHIVE_ARTIFACT, init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("ArchiveArtifact must be built by from_exact_zip")

    @classmethod
    def from_exact_zip(
        cls,
        *,
        realized_pair: C0BRealizedPairV1,
        archive_bytes: bytes,
        member_name: str,
        expected_member_bytes: bytes,
        placement_manifest: G17CompilerPlacementManifestV1,
        decoder_program_bytes: bytes,
        decoder_runtime_bytes: bytes,
    ) -> C0BArchiveArtifactV1:
        if type(realized_pair) is not C0BRealizedPairV1:
            raise G17CompilerPlacementError("C0B archive requires exact RealizedPair parent")
        archive = _exact_bytes(archive_bytes, name="C0B archive")
        expected_member = _exact_bytes(expected_member_bytes, name="C0B member")
        _ascii(member_name, name="C0B member name")
        _exact_bytes(decoder_program_bytes, name="C0B decoder program")
        _exact_bytes(decoder_runtime_bytes, name="C0B decoder runtime")
        if type(placement_manifest) is not G17CompilerPlacementManifestV1:
            raise G17CompilerPlacementError("C0B archive lacks exact placement manifest")
        try:
            with zipfile.ZipFile(io.BytesIO(archive), "r") as opened:
                names = opened.namelist()
                if len(names) != len(set(names)) or names.count(member_name) != 1:
                    raise G17CompilerPlacementError("C0B archive member map is duplicate or missing")
                member_map = tuple(
                    (name, _sha256(payload), len(payload)) for name in names for payload in (opened.read(name),)
                )
                actual_member = opened.read(member_name)
        except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
            raise G17CompilerPlacementError("C0B archive failed actual ZIP parse") from exc
        if actual_member != expected_member:
            raise G17CompilerPlacementError("C0B member is not contained byte-exactly in archive")
        if (
            placement_manifest.exact_archive_bytes != archive
            or placement_manifest.member_name != member_name
            or placement_manifest.exact_member_bytes != actual_member
        ):
            raise G17CompilerPlacementError("C0B placement belongs to another archive/member")
        sparse_groups = {item.physical_coding_group_id for item in realized_pair.obligation_ir.coverage.sparse_owned}
        pose_groups = set(realized_pair.obligation_ir.pose_preimage_ownership.physical_coding_group_id_by_pair)
        manifest_groups = {item.group_id for item in placement_manifest.coding_groups}
        if not sparse_groups.issubset(manifest_groups) or not pose_groups.issubset(manifest_groups):
            raise G17CompilerPlacementError("C0B archive dropped a sparse/Pose physical owner")
        instance = cls.__new__(cls)
        for name, value in (
            ("realized_pair", realized_pair),
            ("member_name", member_name),
            ("member_bytes", actual_member),
            ("archive_bytes", archive),
            ("member_map", member_map),
            ("placement_manifest", placement_manifest),
            ("decoder_program_bytes", decoder_program_bytes),
            ("decoder_runtime_bytes", decoder_runtime_bytes),
            ("phase", G17LifecyclePhaseV1.ARCHIVE_ARTIFACT),
        ):
            object.__setattr__(instance, name, value)
        return instance

    @property
    def archive_sha256(self) -> str:
        return _sha256(self.archive_bytes)

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-ARCHIVE-ARTIFACT-V1\0"
            + bytes.fromhex(self.realized_pair.identity_sha256)
            + self.archive_bytes
            + self.member_name.encode("ascii")
            + b"\0"
            + bytes.fromhex(self.placement_manifest.manifest_sha256)
        )


_AUTH_EVAL_CLOSURE_SEAL: Final = object()
_AUTH_EVAL_SHELL_PATH: Final = "upstream/evaluate.sh"
_AUTH_EVAL_EVALUATOR_PATH: Final = "upstream/evaluate.py"
_AUTH_EVAL_FRAME_UTILS_PATH: Final = "upstream/frame_utils.py"
_AUTH_EVAL_MODULES_PATH: Final = "upstream/modules.py"
_AUTH_EVAL_INFLATE_SH_PATH: Final = "inflate.sh"
_AUTH_EVAL_INFLATE_PY_PATH: Final = "inflate.py"


def _validate_auth_eval_runtime_graph_v1(
    runtime_files: tuple[G17RuntimeDependencyFileV1, ...],
    dependency_edges: tuple[G17RuntimeDependencyEdgeV1, ...],
    observed_runtime_paths: tuple[str, ...],
) -> dict[str, G17RuntimeDependencyFileV1]:
    """Validate the real evaluate.sh-rooted public execution graph.

    Kept separate from the sealed lifecycle constructor so the graph itself is
    regression-testable without fabricating an ArchiveArtifact or authority
    receipt.  System executables and native/package dependencies may extend the
    graph, but every retained node must remain reachable from evaluate.sh.
    """

    if (
        type(runtime_files) is not tuple
        or not runtime_files
        or any(type(item) is not G17RuntimeDependencyFileV1 for item in runtime_files)
    ):
        raise G17CompilerPlacementError("AuthEvalClosure runtime files are not an exact typed tuple")
    if runtime_files != tuple(sorted(runtime_files, key=lambda item: item.relative_path)):
        raise G17CompilerPlacementError("AuthEvalClosure runtime files are not in canonical path order")
    file_by_path = {item.relative_path: item for item in runtime_files}
    if len(file_by_path) != len(runtime_files):
        raise G17CompilerPlacementError("AuthEvalClosure runtime file paths are duplicated")
    required_scopes = {
        _AUTH_EVAL_SHELL_PATH: G17RuntimeFileScopeV1.EVALUATOR_PUBLIC_ENTRYPOINT,
        _AUTH_EVAL_EVALUATOR_PATH: G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        _AUTH_EVAL_FRAME_UTILS_PATH: G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        _AUTH_EVAL_MODULES_PATH: G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        _AUTH_EVAL_INFLATE_SH_PATH: G17RuntimeFileScopeV1.SUBMISSION_PUBLIC_ENTRYPOINT,
        _AUTH_EVAL_INFLATE_PY_PATH: G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
    }
    if any(
        file_by_path.get(path) is None or file_by_path[path].scope is not scope
        for path, scope in required_scopes.items()
    ):
        raise G17CompilerPlacementError(
            "AuthEvalClosure lacks exact evaluate.sh/evaluate.py/scorer-support/inflate public-entrypoint custody"
        )
    if type(dependency_edges) is not tuple or any(
        type(item) is not G17RuntimeDependencyEdgeV1 for item in dependency_edges
    ):
        raise G17CompilerPlacementError("AuthEvalClosure dependency edges are not an exact typed tuple")

    def edge_key(item: G17RuntimeDependencyEdgeV1) -> tuple[str, str, str]:
        return (item.importer_path, item.dependency_path, item.mechanism.value)

    if dependency_edges != tuple(sorted(dependency_edges, key=edge_key)):
        raise G17CompilerPlacementError("AuthEvalClosure dependency edges are not canonical")
    edge_keys = tuple(edge_key(item) for item in dependency_edges)
    if len(edge_keys) != len(set(edge_keys)):
        raise G17CompilerPlacementError("AuthEvalClosure dependency edges are duplicated")
    file_paths = set(file_by_path)
    if any(edge.importer_path not in file_paths or edge.dependency_path not in file_paths for edge in dependency_edges):
        raise G17CompilerPlacementError("AuthEvalClosure dependency edge names an unowned runtime file")
    required_edges = {
        (
            _AUTH_EVAL_SHELL_PATH,
            _AUTH_EVAL_INFLATE_SH_PATH,
            G17RuntimeDependencyMechanismV1.PROCESS_EXEC.value,
        ),
        (
            _AUTH_EVAL_SHELL_PATH,
            _AUTH_EVAL_EVALUATOR_PATH,
            G17RuntimeDependencyMechanismV1.PROCESS_EXEC.value,
        ),
        (
            _AUTH_EVAL_INFLATE_SH_PATH,
            _AUTH_EVAL_INFLATE_PY_PATH,
            G17RuntimeDependencyMechanismV1.PROCESS_EXEC.value,
        ),
        (
            _AUTH_EVAL_EVALUATOR_PATH,
            _AUTH_EVAL_FRAME_UTILS_PATH,
            G17RuntimeDependencyMechanismV1.PYTHON_IMPORT.value,
        ),
        (
            _AUTH_EVAL_EVALUATOR_PATH,
            _AUTH_EVAL_MODULES_PATH,
            G17RuntimeDependencyMechanismV1.PYTHON_IMPORT.value,
        ),
    }
    false_evaluator_launcher_edge = (
        _AUTH_EVAL_EVALUATOR_PATH,
        _AUTH_EVAL_INFLATE_SH_PATH,
        G17RuntimeDependencyMechanismV1.PROCESS_EXEC.value,
    )
    if false_evaluator_launcher_edge in edge_keys:
        raise G17CompilerPlacementError(
            "AuthEvalClosure fabricated evaluate.py launching inflate.sh; evaluate.sh is the real common parent"
        )
    if not required_edges.issubset(set(edge_keys)):
        raise G17CompilerPlacementError("AuthEvalClosure does not execute the required public entrypoint chain")
    reachable = {_AUTH_EVAL_SHELL_PATH}
    while True:
        expanded = reachable | {edge.dependency_path for edge in dependency_edges if edge.importer_path in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    if reachable != file_paths:
        raise G17CompilerPlacementError("AuthEvalClosure contains an unreachable or unowned runtime dependency")
    expected_paths = tuple(sorted(file_paths))
    if observed_runtime_paths != expected_paths:
        raise G17CompilerPlacementError(
            "AuthEvalClosure observed paths do not exactly cover recursively owned runtime files"
        )
    for path in observed_runtime_paths:
        _relative_runtime_path(path, name="observed runtime path")
    return file_by_path


@dataclass(frozen=True, slots=True, init=False)
class C0BAuthEvalClosureV1:
    """Sealed public-entrypoint closure for the exact archive under evaluation.

    A private in-process receiver callback cannot construct this object.  A
    reviewed adapter must observe the real ``upstream/evaluate.sh`` root launch
    both the exact ``inflate.sh``/``inflate.py`` pair and
    ``upstream/evaluate.py``.  The latter must in turn import the frozen scorer
    support modules.  Every recursively consumed runtime file is retained as
    byte-owned dependency evidence; a fabricated ``evaluate.py -> inflate.sh``
    edge is explicitly not the official execution graph.
    """

    archive_artifact: C0BArchiveArtifactV1
    runtime_files: tuple[G17RuntimeDependencyFileV1, ...]
    dependency_edges: tuple[G17RuntimeDependencyEdgeV1, ...]
    observed_runtime_paths: tuple[str, ...]
    dependency_discovery_receipt: G17ReopenedEvidencePacketV1
    public_evaluator_execution_receipt: G17ReopenedEvidencePacketV1
    executed_public_entrypoint_path: Literal["upstream/evaluate.sh"]
    evaluated_archive_sha256: str
    public_decode_receipt_sha256: str
    exact_argv_bytes: bytes = field(repr=False)
    exact_environment_bytes: bytes = field(repr=False)
    proof_dependencies: G17ProofDependencySetV1
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.AUTH_EVAL_CLOSURE, init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("AuthEvalClosure has no public constructor")

    def _sealed_init(
        self,
        *,
        seal: object,
        archive_artifact: C0BArchiveArtifactV1,
        runtime_files: tuple[G17RuntimeDependencyFileV1, ...],
        dependency_edges: tuple[G17RuntimeDependencyEdgeV1, ...],
        observed_runtime_paths: tuple[str, ...],
        dependency_discovery_receipt: G17ReopenedEvidencePacketV1,
        public_evaluator_execution_receipt: G17ReopenedEvidencePacketV1,
        executed_public_entrypoint_path: str,
        evaluated_archive_sha256: str,
        public_decode_receipt_sha256: str,
        exact_argv_bytes: bytes,
        exact_environment_bytes: bytes,
    ) -> None:
        if seal is not _AUTH_EVAL_CLOSURE_SEAL:
            raise G17CompilerPlacementError("AuthEvalClosure is sealed to a reviewed public evaluator adapter")
        if type(archive_artifact) is not C0BArchiveArtifactV1:
            raise G17CompilerPlacementError("AuthEvalClosure requires the exact ArchiveArtifact parent")
        _validate_auth_eval_runtime_graph_v1(runtime_files, dependency_edges, observed_runtime_paths)
        for receipt, name in (
            (dependency_discovery_receipt, "dependency discovery"),
            (public_evaluator_execution_receipt, "public evaluator execution"),
        ):
            if type(receipt) is not G17ReopenedEvidencePacketV1:
                raise G17CompilerPlacementError(f"AuthEvalClosure {name} receipt is not strict-reopened")
            receipt.reopen()
        if executed_public_entrypoint_path != _AUTH_EVAL_SHELL_PATH:
            raise G17CompilerPlacementError("AuthEvalClosure did not execute upstream/evaluate.sh as public entrypoint")
        for digest, name in (
            (evaluated_archive_sha256, "evaluated archive"),
            (public_decode_receipt_sha256, "public decode receipt"),
        ):
            if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise G17CompilerPlacementError(f"AuthEvalClosure {name} digest is invalid")
        if evaluated_archive_sha256 != archive_artifact.archive_sha256:
            raise G17CompilerPlacementError("AuthEvalClosure evaluated another archive object")
        _exact_bytes(exact_argv_bytes, name="AuthEvalClosure argv")
        _exact_bytes(exact_environment_bytes, name="AuthEvalClosure environment")

        entrypoint_chain_bytes = _canonical_json(
            {
                "executed_public_entrypoint_path": executed_public_entrypoint_path,
                "edges": [
                    {
                        "dependency_path": edge.dependency_path,
                        "importer_path": edge.importer_path,
                        "mechanism": edge.mechanism.value,
                    }
                    for edge in dependency_edges
                ],
            }
        )
        runtime_file_closure_bytes = _canonical_json(
            {
                "files": [
                    {
                        "bytes": len(item.exact_file_bytes),
                        "custody_owner": item.custody_owner,
                        "path": item.relative_path,
                        "scope": item.scope.value,
                        "sha256": item.content_sha256,
                    }
                    for item in runtime_files
                ],
                "observed_runtime_paths": list(observed_runtime_paths),
            }
        )
        execution_receipt_bytes = _canonical_json(
            {
                "archive_sha256": evaluated_archive_sha256,
                "argv_sha256": _sha256(exact_argv_bytes),
                "dependency_discovery_receipt_sha256": dependency_discovery_receipt.packet_sha256,
                "environment_sha256": _sha256(exact_environment_bytes),
                "public_decode_receipt_sha256": public_decode_receipt_sha256,
                "public_evaluator_execution_receipt_sha256": public_evaluator_execution_receipt.packet_sha256,
            }
        )
        dependencies = G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1.AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE,
            dependencies=tuple(
                sorted(
                    (
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.ARCHIVE_BYTES,
                            archive_artifact.archive_bytes,
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING,
                            _canonical_json(archive_artifact.member_map),
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.PUBLIC_ENTRYPOINT_CHAIN,
                            entrypoint_chain_bytes,
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.RUNTIME_FILE_CLOSURE,
                            runtime_file_closure_bytes,
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.AUTH_EVAL_EXECUTION_RECEIPT,
                            execution_receipt_bytes,
                        ),
                    ),
                    key=lambda item: item.domain.value,
                )
            ),
            declared_external_reads=tuple(
                sorted(
                    (
                        G17ProofDependencyDomainV1.AUTH_EVAL_EXECUTION_RECEIPT,
                        G17ProofDependencyDomainV1.PUBLIC_ENTRYPOINT_CHAIN,
                        G17ProofDependencyDomainV1.RUNTIME_FILE_CLOSURE,
                    ),
                    key=lambda item: item.value,
                )
            ),
        )
        for name, value in (
            ("archive_artifact", archive_artifact),
            ("runtime_files", runtime_files),
            ("dependency_edges", dependency_edges),
            ("observed_runtime_paths", observed_runtime_paths),
            ("dependency_discovery_receipt", dependency_discovery_receipt),
            ("public_evaluator_execution_receipt", public_evaluator_execution_receipt),
            ("executed_public_entrypoint_path", executed_public_entrypoint_path),
            ("evaluated_archive_sha256", evaluated_archive_sha256),
            ("public_decode_receipt_sha256", public_decode_receipt_sha256),
            ("exact_argv_bytes", exact_argv_bytes),
            ("exact_environment_bytes", exact_environment_bytes),
            ("proof_dependencies", dependencies),
            ("phase", G17LifecyclePhaseV1.AUTH_EVAL_CLOSURE),
        ):
            object.__setattr__(self, name, value)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-AUTH-EVAL-CLOSURE-V1\0"
            + bytes.fromhex(self.archive_artifact.identity_sha256)
            + bytes.fromhex(self.dependency_discovery_receipt.packet_sha256)
            + bytes.fromhex(self.public_evaluator_execution_receipt.packet_sha256)
            + bytes.fromhex(self.public_decode_receipt_sha256)
            + bytes.fromhex(self.proof_dependencies.identity_sha256)
        )


def require_g17_auth_eval_public_entrypoint_closure() -> None:
    raise G17CompilerBlocker(
        G17CompilerBlockerCodeV1.G17_AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE_OWED,
        "reviewed public evaluator adapter is not landed: exact archive must run through "
        "upstream/evaluate.sh -> {inflate.sh -> inflate.py, evaluate.py -> scorer support} "
        "with recursive byte-owned runtime dependency closure",
    )


@dataclass(frozen=True, slots=True)
class G17ReceiverExecutionResultV1:
    decoded_output_bytes: bytes = field(repr=False)
    receiver_receipt: G17ReopenedEvidencePacketV1
    emitted_pair_order: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_bytes(self.decoded_output_bytes, name="receiver decoded output")
        if type(self.receiver_receipt) is not G17ReopenedEvidencePacketV1:
            raise G17CompilerPlacementError("receiver result lacks strict-reopened receipt")
        self.receiver_receipt.reopen()
        if type(self.emitted_pair_order) is not tuple or any(type(item) is not int for item in self.emitted_pair_order):
            raise G17CompilerPlacementError("receiver pair order is not an exact tuple")


G17ArchiveReceiverV1: TypeAlias = Callable[[bytes], G17ReceiverExecutionResultV1]


@dataclass(frozen=True, slots=True, init=False)
class C0BDecodeReceiptV1:
    archive_artifact: C0BArchiveArtifactV1
    decoded_output_bytes: bytes = field(repr=False)
    receiver_receipt: G17ReopenedEvidencePacketV1
    receiver_implementation_bytes: bytes = field(repr=False)
    receiver_runtime_bytes: bytes = field(repr=False)
    receiver_asset_bytes: bytes = field(repr=False)
    group_mutation_results: tuple[tuple[str, Literal["CHANGED", "REFUSED"]], ...]
    proof_dependencies: G17ProofDependencySetV1
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.DECODE_RECEIPT, init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("DecodeReceipt must be built by from_receiver")

    @classmethod
    def from_receiver(
        cls,
        *,
        archive_artifact: C0BArchiveArtifactV1,
        receiver: G17ArchiveReceiverV1,
        receiver_implementation_bytes: bytes,
        receiver_runtime_bytes: bytes,
        receiver_asset_bytes: bytes,
        equality_algorithm_bytes: bytes,
    ) -> C0BDecodeReceiptV1:
        if type(archive_artifact) is not C0BArchiveArtifactV1:
            raise G17CompilerPlacementError("C0B decode requires exact ArchiveArtifact parent")
        if not callable(receiver):
            raise G17CompilerPlacementError("C0B decode requires a real receiver callback")
        for value, name in (
            (receiver_implementation_bytes, "receiver implementation"),
            (receiver_runtime_bytes, "receiver runtime"),
            (receiver_asset_bytes, "receiver assets"),
            (equality_algorithm_bytes, "decoder equality algorithm"),
        ):
            _exact_bytes(value, name=name)
        first = receiver(archive_artifact.archive_bytes)
        second = receiver(archive_artifact.archive_bytes)
        if type(first) is not G17ReceiverExecutionResultV1 or type(second) is not G17ReceiverExecutionResultV1:
            raise G17CompilerPlacementError("receiver callback returned an untyped result")
        if (
            first.decoded_output_bytes != second.decoded_output_bytes
            or first.receiver_receipt.exact_packet_bytes != second.receiver_receipt.exact_packet_bytes
            or first.emitted_pair_order != second.emitted_pair_order
        ):
            raise G17CompilerPlacementError("receiver failed deterministic double execution")
        population_order = archive_artifact.realized_pair.obligation_ir.population.source_pair_ids
        if first.emitted_pair_order != population_order:
            raise G17CompilerPlacementError("receiver omitted, duplicated, or reordered population pairs")
        if first.decoded_output_bytes != archive_artifact.realized_pair.chronology_bytes:
            raise G17CompilerPlacementError("decoded bytes differ from retained RealizedPair")
        mutation_results: list[tuple[str, Literal["CHANGED", "REFUSED"]]] = []
        for group in archive_artifact.placement_manifest.coding_groups:
            mutable = bytearray(archive_artifact.archive_bytes)
            mutable[group.archive_offset] ^= 1
            try:
                changed = receiver(bytes(mutable))
            except Exception:
                mutation_results.append((group.group_id, "REFUSED"))
            else:
                if type(changed) is not G17ReceiverExecutionResultV1:
                    raise G17CompilerPlacementError("mutated receiver returned an untyped result")
                if (
                    changed.decoded_output_bytes == first.decoded_output_bytes
                    and changed.receiver_receipt.exact_packet_bytes == first.receiver_receipt.exact_packet_bytes
                ):
                    raise G17CompilerPlacementError("counted physical coding group is decoder-dead")
                mutation_results.append((group.group_id, "CHANGED"))
        pair_order_bytes = struct.pack(">" + "H" * len(population_order), *population_order)
        dependencies = G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1.ARCHIVE_DECODE_EQUALITY,
            dependencies=tuple(
                sorted(
                    (
                        G17ProofDependencyV1(G17ProofDependencyDomainV1.ARCHIVE_BYTES, archive_artifact.archive_bytes),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING,
                            _canonical_json(archive_artifact.member_map),
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.RECEIVER_IMPLEMENTATION,
                            receiver_implementation_bytes,
                        ),
                        G17ProofDependencyV1(G17ProofDependencyDomainV1.RECEIVER_RUNTIME, receiver_runtime_bytes),
                        G17ProofDependencyV1(G17ProofDependencyDomainV1.PAIR_ORDER, pair_order_bytes),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.DECODER_EQUALITY_ALGORITHM,
                            equality_algorithm_bytes,
                        ),
                    ),
                    key=lambda item: item.domain.value,
                )
            ),
            declared_external_reads=(),
        )
        instance = cls.__new__(cls)
        for name, value in (
            ("archive_artifact", archive_artifact),
            ("decoded_output_bytes", first.decoded_output_bytes),
            ("receiver_receipt", first.receiver_receipt),
            ("receiver_implementation_bytes", receiver_implementation_bytes),
            ("receiver_runtime_bytes", receiver_runtime_bytes),
            ("receiver_asset_bytes", receiver_asset_bytes),
            ("group_mutation_results", tuple(mutation_results)),
            ("proof_dependencies", dependencies),
            ("phase", G17LifecyclePhaseV1.DECODE_RECEIPT),
        ):
            object.__setattr__(instance, name, value)
        return instance

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-DECODE-RECEIPT-V1\0"
            + bytes.fromhex(self.archive_artifact.identity_sha256)
            + self.decoded_output_bytes
            + bytes.fromhex(self.receiver_receipt.packet_sha256)
            + bytes.fromhex(self.proof_dependencies.identity_sha256)
        )


class C0BScoreAxisV1(StrEnum):
    CONTEST_CPU = "contest-CPU"
    CONTEST_CUDA = "contest-CUDA"
    MACOS_CPU_ADVISORY = "macOS-CPU-advisory"


@dataclass(frozen=True, slots=True)
class G17ScorerExecutionResultV1:
    observation: G17CandidateForwardObservationV1
    authority: G17ResearchAuthorityEvidenceV1

    def __post_init__(self) -> None:
        if type(self.observation) is not G17CandidateForwardObservationV1:
            raise G17CompilerPlacementError("scorer execution result lacks typed observation")
        if type(self.authority) is not G17ResearchAuthorityEvidenceV1:
            raise G17CompilerPlacementError("local scorer callback cannot mint contest authority")


G17ResearchScorerCallbackV1: TypeAlias = Callable[[C0BDecodeReceiptV1], G17ScorerExecutionResultV1]


@dataclass(frozen=True, slots=True, init=False)
class C0BScoreReceiptV1:
    decode_receipt: C0BDecodeReceiptV1
    authority: G17AuthorityEvidenceV1
    observation: G17CandidateForwardObservationV1
    d_seg_term: float
    d_pose_term: float
    rate_term: float
    total_score: float
    proof_dependencies: G17ProofDependencySetV1
    phase: G17LifecyclePhaseV1 = field(default=G17LifecyclePhaseV1.SCORE_RECEIPT, init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("ScoreReceipt must be built by an invoked scorer/evaluator adapter")

    @classmethod
    def from_research_scorer(
        cls,
        *,
        decode_receipt: C0BDecodeReceiptV1,
        scorer: G17ResearchScorerCallbackV1,
    ) -> C0BScoreReceiptV1:
        if type(decode_receipt) is not C0BDecodeReceiptV1:
            raise G17CompilerPlacementError("C0B score requires exact DecodeReceipt parent")
        if not callable(scorer):
            raise G17CompilerPlacementError("C0B score requires an invoked scorer callback")
        first = scorer(decode_receipt)
        second = scorer(decode_receipt)
        if type(first) is not G17ScorerExecutionResultV1 or type(second) is not G17ScorerExecutionResultV1:
            raise G17CompilerPlacementError("scorer callback returned an untyped result")
        if (
            first.observation.receipt.to_receipt_bytes() != second.observation.receipt.to_receipt_bytes()
            or first.authority.authority_sha256 != second.authority.authority_sha256
        ):
            raise G17CompilerPlacementError("scorer callback failed deterministic double execution")
        observation = first.observation
        authority = first.authority
        if observation.decoded_output_bytes != decode_receipt.decoded_output_bytes:
            raise G17CompilerPlacementError("scorer observation belongs to another decoded output")
        pair_count = len(decode_receipt.archive_artifact.realized_pair.obligation_ir.population.source_pair_ids)
        if authority.sample_count != pair_count:
            raise G17CompilerPlacementError("scorer authority sample scope differs from decoded population")
        if authority.evidence_receipt.exact_packet_bytes != observation.receipt.to_receipt_bytes():
            raise G17CompilerPlacementError("research authority does not reopen this candidate observation")
        if _sha256(authority.exact_scorer_identity_bytes) != observation.receipt.frozen_scorer_sha256:
            raise G17CompilerPlacementError("scorer authority identity differs from observation")
        if _sha256(authority.exact_runtime_identity_bytes) != observation.receipt.scorer_runtime_environment_sha256:
            raise G17CompilerPlacementError("scorer runtime authority differs from observation")
        d_seg_term = seg_term(observation.receipt.aggregate_d_seg)
        d_pose_term = pose_term(observation.receipt.aggregate_d_pose)
        archive_nbytes = len(decode_receipt.archive_artifact.archive_bytes)
        rate_score_term = rate_term(archive_nbytes)
        total = compute_contest_score(
            observation.receipt.aggregate_d_seg,
            observation.receipt.aggregate_d_pose,
            archive_nbytes,
        )
        scope_bytes = _canonical_json(
            {
                "authority_class": authority.authority_class.value,
                "axis_label": authority.axis_label,
                "sample_count": authority.sample_count,
                "hardware_sha256": _sha256(authority.exact_hardware_identity_bytes),
            }
        )
        dependencies = G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1.SCORE_OBSERVATION,
            dependencies=tuple(
                sorted(
                    (
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.ARCHIVE_BYTES,
                            decode_receipt.archive_artifact.archive_bytes,
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.DECODE_RECEIPT,
                            bytes.fromhex(decode_receipt.identity_sha256),
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.FROZEN_SCORER,
                            authority.exact_scorer_identity_bytes,
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.SCORER_RUNTIME,
                            authority.exact_runtime_identity_bytes,
                        ),
                        G17ProofDependencyV1(G17ProofDependencyDomainV1.AXIS_AND_SAMPLE_SCOPE, scope_bytes),
                    ),
                    key=lambda item: item.domain.value,
                )
            ),
            declared_external_reads=(),
        )
        instance = cls.__new__(cls)
        for name, value in (
            ("decode_receipt", decode_receipt),
            ("authority", authority),
            ("observation", observation),
            ("d_seg_term", d_seg_term),
            ("d_pose_term", d_pose_term),
            ("rate_term", rate_score_term),
            ("total_score", total),
            ("proof_dependencies", dependencies),
            ("phase", G17LifecyclePhaseV1.SCORE_RECEIPT),
        ):
            object.__setattr__(instance, name, value)
        return instance

    @property
    def axis(self) -> C0BScoreAxisV1:
        if type(self.authority) is G17ResearchAuthorityEvidenceV1:
            return C0BScoreAxisV1.MACOS_CPU_ADVISORY
        if type(self.authority) is G17ContestCPUAuthorityEvidenceV1:
            return C0BScoreAxisV1.CONTEST_CPU
        if type(self.authority) is G17ContestCUDAAuthorityEvidenceV1:
            return C0BScoreAxisV1.CONTEST_CUDA
        raise G17CompilerPlacementError("score authority variant is not sealed")

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            b"C0B-SCORE-RECEIPT-V1\0"
            + bytes.fromhex(self.decode_receipt.identity_sha256)
            + self.axis.value.encode("ascii")
            + b"\0"
            + struct.pack(">dddd", self.d_seg_term, self.d_pose_term, self.rate_term, self.total_score)
            + bytes.fromhex(self.proof_dependencies.identity_sha256)
        )


@dataclass(frozen=True, slots=True)
class G17WholeObjectStateV1:
    """One integrated archive/decode/score state; never a per-axis payoff row."""

    score_receipt: C0BScoreReceiptV1
    competitive_target: G17CompetitiveTargetIdentityV1

    def __post_init__(self) -> None:
        if type(self.score_receipt) is not C0BScoreReceiptV1:
            raise G17CompilerPlacementError("whole-object state requires exact composed score receipt")
        if type(self.competitive_target) is not G17CompetitiveTargetIdentityV1:
            raise G17CompilerPlacementError("whole-object state requires semantic CompetitiveTargetIdentity")

    @property
    def state_sha256(self) -> str:
        return _sha256(
            b"G17-WHOLE-OBJECT-STATE-V1\0"
            + bytes.fromhex(self.score_receipt.identity_sha256)
            + bytes.fromhex(self.competitive_target.identity_sha256)
        )


G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA_V1: Final = "tac.g17_whole_object_state_receipt.v1"
_G17_WHOLE_OBJECT_STATE_RECEIPT_ROLE_V1: Final = "G17_STRICT_WHOLE_OBJECT_STATE_EVIDENCE"
_G17_SCORE_OPERATION_ORDER_V1: Final = "100*d_seg+sqrt(10*d_pose)+25*(archive_nbytes/37545489)"
_G17_RECEIPT_SEAL: Final = object()

_G17_LOGICAL_CLASS_BY_TYPE: Final = {
    G17LogicalValueTypeV1.SEMANTIC_TOPOLOGY.value: G17SemanticTopologyV1,
    G17LogicalValueTypeV1.REALIZATION_GAUGE.value: G17RealizationGaugeV1,
    G17LogicalValueTypeV1.CHRONOLOGICAL_POSE_PREIMAGE.value: G17ChronologicalPosePreimageV1,
    G17LogicalValueTypeV1.POPULATION_SHARING.value: G17PopulationSharingV1,
    G17LogicalValueTypeV1.ENTROPY_CONTEXT.value: G17EntropyContextV1,
    G17LogicalValueTypeV1.ANALYTIC_RESIDUAL_OWNERSHIP.value: G17AnalyticResidualOwnershipV1,
    G17LogicalValueTypeV1.LEARNED_RESIDUAL_OWNERSHIP.value: G17LearnedResidualOwnershipV1,
    G17LogicalValueTypeV1.ENCODER_ONLY_TEACHER_ORACLE_EVIDENCE.value: G17EncoderOnlyTeacherOracleEvidenceV1,
    G17LogicalValueTypeV1.FORWARD_OBSERVATION.value: G17ForwardObservationLogicalV1,
    G17LogicalValueTypeV1.TERMINAL_ENVELOPE.value: G17TerminalEnvelopeLogicalV1,
    G17LogicalValueTypeV1.GENERIC_VM_INTERPRETER.value: G17GenericVMInterpreterV1,
    G17LogicalValueTypeV1.COUNTED_VM_BYTECODE.value: G17CountedVMBytecodeV1,
    G17LogicalValueTypeV1.COUNTED_VM_OPERAND.value: G17CountedVMOperandV1,
    G17LogicalValueTypeV1.PACKAGED_EXECUTABLE.value: G17PackagedExecutableV1,
}


@dataclass(frozen=True, slots=True)
class _G17CanonicalEvidenceDocumentV1:
    schema: str
    exact_bytes: bytes = field(repr=False)

    def to_receipt_bytes(self) -> bytes:
        return self.exact_bytes


def _parse_canonical_evidence_document(payload: bytes, *, expected_schema: str) -> _G17CanonicalEvidenceDocumentV1:
    decoded = _decode_canonical_json_object(payload, name=f"{expected_schema} evidence")
    if decoded.get("schema") != expected_schema:
        raise G17CompilerPlacementError("nested evidence schema changed")
    return _G17CanonicalEvidenceDocumentV1(schema=expected_schema, exact_bytes=payload)


def _strict_nested_document_payload(packet: G17ReopenedEvidencePacketV1) -> dict[str, str]:
    if type(packet) is not G17ReopenedEvidencePacketV1:
        raise G17CompilerPlacementError("nested evidence packet is not strict-reopened")
    reopened = packet.reopen()
    payload = reopened.to_receipt_bytes()
    _parse_canonical_evidence_document(payload, expected_schema=packet.expected_schema)
    return {"schema": packet.expected_schema, "exact_bytes_b64": _b64(payload)}


def _parse_nested_document(value: object, *, name: str) -> G17ReopenedEvidencePacketV1:
    if type(value) is not dict or set(value) != {"schema", "exact_bytes_b64"}:
        raise G17CompilerPlacementError(f"{name} nested evidence fields are not exact")
    schema = _ascii(value["schema"], name=f"{name} schema")
    payload = _unb64(value["exact_bytes_b64"], name=f"{name} bytes")

    def parser(raw: bytes) -> _G17CanonicalEvidenceDocumentV1:
        return _parse_canonical_evidence_document(raw, expected_schema=schema)

    return G17ReopenedEvidencePacketV1(
        exact_packet_bytes=payload,
        strict_parser=parser,
        expected_schema=schema,
    )


def _logical_owner_payload(owner: G17LogicalOwnershipV1) -> dict[str, Any]:
    functional = owner.functional_identity
    spelling = owner.parameter_spelling
    return {
        "owner_id": owner.owner_id,
        "ownership_kind": owner.ownership_kind.value,
        "logical_type": owner.value.logical_type.value,
        "exact_value_b64": _b64(owner.value.exact_bytes),
        "functional_identity": None
        if functional is None
        else {
            "decoded_output_b64": _b64(functional.exact_decoded_output_bytes),
            "evaluator_cell_b64": _b64(functional.exact_evaluator_cell_bytes),
        },
        "parameter_spelling": None
        if spelling is None
        else {
            "exact_bytes_b64": _b64(spelling.exact_parameter_bytes),
            "format": spelling.spelling_format,
        },
        "identity_sha256": owner.identity_sha256,
    }


def _parse_logical_owner_payload(value: object) -> G17LogicalOwnershipV1:
    expected = {
        "owner_id",
        "ownership_kind",
        "logical_type",
        "exact_value_b64",
        "functional_identity",
        "parameter_spelling",
        "identity_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise G17CompilerPlacementError("logical owner receipt fields are not exact")
    logical_type = _ascii(value["logical_type"], name="logical owner value type")
    value_class = _G17_LOGICAL_CLASS_BY_TYPE.get(logical_type)
    if value_class is None:
        raise G17CompilerPlacementError("logical owner value type is unknown")
    logical_value = value_class(_unb64(value["exact_value_b64"], name="logical owner value"))
    functional_value = value["functional_identity"]
    if functional_value is None:
        functional = None
    else:
        if type(functional_value) is not dict or set(functional_value) != {
            "decoded_output_b64",
            "evaluator_cell_b64",
        }:
            raise G17CompilerPlacementError("functional identity fields are not exact")
        functional = G17FunctionalQuotientIdentityV1(
            _unb64(functional_value["decoded_output_b64"], name="functional decoded output"),
            _unb64(functional_value["evaluator_cell_b64"], name="functional evaluator cell"),
        )
    spelling_value = value["parameter_spelling"]
    if spelling_value is None:
        spelling = None
    else:
        if type(spelling_value) is not dict or set(spelling_value) != {"exact_bytes_b64", "format"}:
            raise G17CompilerPlacementError("parameter spelling fields are not exact")
        spelling = G17ParameterSpellingIdentityV1(
            _unb64(spelling_value["exact_bytes_b64"], name="parameter spelling"),
            _ascii(spelling_value["format"], name="parameter spelling format"),
        )
    try:
        ownership_kind = G17LogicalOwnershipKindV1(value["ownership_kind"])
    except (TypeError, ValueError) as exc:
        raise G17CompilerPlacementError("logical ownership kind is invalid") from exc
    owner = G17LogicalOwnershipV1(
        owner_id=_ascii(value["owner_id"], name="logical owner ID"),
        ownership_kind=ownership_kind,
        value=logical_value,
        functional_identity=functional,
        parameter_spelling=spelling,
    )
    if value["identity_sha256"] != owner.identity_sha256:
        raise G17CompilerPlacementError("logical owner identity did not recompute")
    return owner


def _placement_payload(manifest: G17CompilerPlacementManifestV1) -> dict[str, Any]:
    records = []
    for row in manifest.records:
        records.append(
            {
                "logical_owner": _logical_owner_payload(row.logical_owner),
                "scientific_role": row.scientific_role.value,
                "semantic_role": row.semantic_role.value,
                "recursion_namespace": row.recursion_coordinate.namespace.value,
                "recursion_stage": row.recursion_coordinate.stage,
                "placement_class": row.placement_class.value,
                "artifact_class": row.artifact_class.value,
                "payload_class": row.payload_class,
                "physical_coding_group_id": row.physical_coding_group_id,
                "video_specific_derivation": row.video_specific_derivation,
                "packaged_inside_archive": row.packaged_inside_archive,
                "target_selected_constant": row.target_selected_constant,
            }
        )
    groups = [
        {
            "group_id": group.group_id,
            "member_name": group.member_name,
            "archive_offset": group.archive_offset,
            "range_nbytes": group.byte_length,
            "range_sha256": group.range_sha256,
            "coder_owner": group.coder_owner,
            "container_owner": group.container_owner,
            "receiver_consumer": group.receiver_consumer,
            "receiver_operation": group.receiver_operation,
            "logical_owner_ids": list(group.logical_owner_ids),
        }
        for group in manifest.coding_groups
    ]
    return {
        "records": records,
        "coding_groups": groups,
        "expected_object_identities": list(manifest.expected_object_identities),
        "manifest_sha256": manifest.manifest_sha256,
    }


def _parse_placement_payload(
    value: object,
    *,
    archive_bytes: bytes,
    member_name: str,
    member_bytes: bytes,
) -> G17CompilerPlacementManifestV1:
    if type(value) is not dict or set(value) != {
        "records",
        "coding_groups",
        "expected_object_identities",
        "manifest_sha256",
    }:
        raise G17CompilerPlacementError("placement receipt fields are not exact")
    if type(value["records"]) is not list or not value["records"]:
        raise G17CompilerPlacementError("placement receipt has no typed records")
    owner_cache: dict[str, tuple[bytes, G17LogicalOwnershipV1]] = {}
    records: list[G17CompilerPlacementRecordV1] = []
    record_fields = {
        "logical_owner",
        "scientific_role",
        "semantic_role",
        "recursion_namespace",
        "recursion_stage",
        "placement_class",
        "artifact_class",
        "payload_class",
        "physical_coding_group_id",
        "video_specific_derivation",
        "packaged_inside_archive",
        "target_selected_constant",
    }
    for row in value["records"]:
        if type(row) is not dict or set(row) != record_fields:
            raise G17CompilerPlacementError("placement record fields are not exact")
        owner_bytes = _canonical_json(row["logical_owner"])
        parsed_owner = _parse_logical_owner_payload(row["logical_owner"])
        prior = owner_cache.setdefault(parsed_owner.owner_id, (owner_bytes, parsed_owner))
        if prior[0] != owner_bytes:
            raise G17CompilerPlacementError("one logical owner ID has different receipt semantics")
        owner = prior[1]
        try:
            record = G17CompilerPlacementRecordV1(
                logical_owner=owner,
                scientific_role=G17ScientificRoleV1(row["scientific_role"]),
                semantic_role=G17SemanticStreamRoleV1(row["semantic_role"]),
                recursion_coordinate=G17RecursionCoordinateV1(
                    G17RecursionNamespaceV1(row["recursion_namespace"]),
                    row["recursion_stage"],
                ),
                placement_class=G17PlacementClassV1(row["placement_class"]),
                artifact_class=G17ArtifactClassV1(row["artifact_class"]),
                payload_class=row["payload_class"],
                physical_coding_group_id=row["physical_coding_group_id"],
                video_specific_derivation=row["video_specific_derivation"],
                packaged_inside_archive=row["packaged_inside_archive"],
                target_selected_constant=row["target_selected_constant"],
            )
        except (TypeError, ValueError) as exc:
            raise G17CompilerPlacementError("placement record typed semantics are invalid") from exc
        records.append(record)
    if type(value["coding_groups"]) is not list:
        raise G17CompilerPlacementError("placement coding groups must be a list")
    group_fields = {
        "group_id",
        "member_name",
        "archive_offset",
        "range_nbytes",
        "range_sha256",
        "coder_owner",
        "container_owner",
        "receiver_consumer",
        "receiver_operation",
        "logical_owner_ids",
    }
    groups: list[G17PhysicalCodingGroupV1] = []
    for row in value["coding_groups"]:
        if type(row) is not dict or set(row) != group_fields:
            raise G17CompilerPlacementError("coding-group receipt fields are not exact")
        if type(row["archive_offset"]) is not int or type(row["range_nbytes"]) is not int:
            raise G17CompilerPlacementError("coding-group span types are invalid")
        start = row["archive_offset"]
        stop = start + row["range_nbytes"]
        if start < 0 or stop > len(archive_bytes):
            raise G17CompilerPlacementError("coding-group span is outside archive")
        exact_range = archive_bytes[start:stop]
        if _sha256(exact_range) != row["range_sha256"]:
            raise G17CompilerPlacementError("coding-group range identity did not recompute")
        if type(row["logical_owner_ids"]) is not list:
            raise G17CompilerPlacementError("coding-group logical owners are untyped")
        groups.append(
            G17PhysicalCodingGroupV1(
                group_id=row["group_id"],
                exact_archive_bytes=archive_bytes,
                member_name=row["member_name"],
                exact_member_bytes=member_bytes,
                archive_offset=start,
                exact_range_bytes=exact_range,
                coder_owner=row["coder_owner"],
                container_owner=row["container_owner"],
                receiver_consumer=row["receiver_consumer"],
                receiver_operation=row["receiver_operation"],
                logical_owner_ids=tuple(row["logical_owner_ids"]),
            )
        )
    if type(value["expected_object_identities"]) is not list:
        raise G17CompilerPlacementError("expected ownership identities must be a list")
    manifest = G17CompilerPlacementManifestV1(
        records=tuple(records),
        coding_groups=tuple(groups),
        expected_object_identities=tuple(value["expected_object_identities"]),
        exact_archive_bytes=archive_bytes,
        member_name=member_name,
        exact_member_bytes=member_bytes,
    )
    if value["manifest_sha256"] != manifest.manifest_sha256:
        raise G17CompilerPlacementError("placement manifest identity did not recompute")
    return manifest


def _population_payload(population: G17PairPopulationV1) -> dict[str, list[int]]:
    return {
        "global_pair_ids": list(population.global_pair_ids),
        "source_pair_ids": list(population.source_pair_ids),
        "v9_pair_coordinates": list(population.v9_pair_coordinates),
        "pbr_pair_coordinates": list(population.pbr_pair_coordinates),
        "obligation_ir_coordinates": list(population.obligation_ir_coordinates),
        "v10_local_coordinates": list(population.v10_local_coordinates),
    }


def _parse_population_payload(value: object) -> G17PairPopulationV1:
    names = {
        "global_pair_ids",
        "source_pair_ids",
        "v9_pair_coordinates",
        "pbr_pair_coordinates",
        "obligation_ir_coordinates",
        "v10_local_coordinates",
    }
    if type(value) is not dict or set(value) != names or any(type(value[name]) is not list for name in names):
        raise G17CompilerPlacementError("pair-population receipt fields are not exact")
    return G17PairPopulationV1(**{name: tuple(value[name]) for name in names})


def _obligation_payload(coverage: G17ObligationCoverageV1) -> dict[str, Any]:
    return {
        "mode": coverage.mode.value,
        "obligation_universe": [[item.pair_id, item.coordinate_id] for item in coverage.obligation_universe],
        "predictor_owned": [[item.pair_id, item.coordinate_id] for item in coverage.predictor_owned],
        "sparse_owned": [
            {
                "obligation": [item.obligation.pair_id, item.obligation.coordinate_id],
                "physical_coding_group_id": item.physical_coding_group_id,
                "receiver_consumer": item.receiver_consumer,
                "receiver_operation": item.receiver_operation,
            }
            for item in coverage.sparse_owned
        ],
    }


def _coordinate(value: object, *, name: str) -> G17ObligationCoordinateV1:
    if type(value) is not list or len(value) != 2:
        raise G17CompilerPlacementError(f"{name} coordinate is not [pair_id,coordinate_id]")
    return G17ObligationCoordinateV1(value[0], value[1])


def _parse_obligation_payload(value: object, *, population: G17PairPopulationV1) -> G17ObligationCoverageV1:
    if type(value) is not dict or set(value) != {
        "mode",
        "obligation_universe",
        "predictor_owned",
        "sparse_owned",
    }:
        raise G17CompilerPlacementError("obligation receipt fields are not exact")
    if any(type(value[name]) is not list for name in ("obligation_universe", "predictor_owned", "sparse_owned")):
        raise G17CompilerPlacementError("obligation receipt collections are not lists")
    universe = tuple(_coordinate(item, name="universe") for item in value["obligation_universe"])
    predictor = tuple(_coordinate(item, name="predictor") for item in value["predictor_owned"])
    sparse: list[G17SparseObligationOwnerV1] = []
    for item in value["sparse_owned"]:
        if type(item) is not dict or set(item) != {
            "obligation",
            "physical_coding_group_id",
            "receiver_consumer",
            "receiver_operation",
        }:
            raise G17CompilerPlacementError("sparse ownership receipt fields are not exact")
        sparse.append(
            G17SparseObligationOwnerV1(
                obligation=_coordinate(item["obligation"], name="sparse"),
                physical_coding_group_id=item["physical_coding_group_id"],
                receiver_consumer=item["receiver_consumer"],
                receiver_operation=item["receiver_operation"],
            )
        )
    try:
        mode = G17ObligationCoverageModeV1(value["mode"])
    except (TypeError, ValueError) as exc:
        raise G17CompilerPlacementError("obligation coverage mode is invalid") from exc
    return G17ObligationCoverageV1(population, mode, universe, predictor, tuple(sparse))


def _pose_ownership_payload(ownership: G17PosePreimageOwnershipV1) -> dict[str, Any]:
    return {
        "ownership_by_pair": [item.value for item in ownership.ownership_by_pair],
        "physical_coding_group_id_by_pair": list(ownership.physical_coding_group_id_by_pair),
        "receiver_operation_by_pair": list(ownership.receiver_operation_by_pair),
        "explicit_preimage_packet": None
        if ownership.explicit_preimage_packet is None
        else _strict_nested_document_payload(ownership.explicit_preimage_packet),
        "ownership_receipt_sha256": ownership.ownership_receipt_sha256,
    }


def _parse_pose_ownership_payload(value: object, *, population: G17PairPopulationV1) -> G17PosePreimageOwnershipV1:
    if type(value) is not dict or set(value) != {
        "ownership_by_pair",
        "physical_coding_group_id_by_pair",
        "receiver_operation_by_pair",
        "explicit_preimage_packet",
        "ownership_receipt_sha256",
    }:
        raise G17CompilerPlacementError("Pose ownership receipt fields are not exact")
    if any(
        type(value[name]) is not list
        for name in (
            "ownership_by_pair",
            "physical_coding_group_id_by_pair",
            "receiver_operation_by_pair",
        )
    ):
        raise G17CompilerPlacementError("Pose ownership collections are not lists")
    try:
        owners = tuple(G17PoseOwnershipV1(item) for item in value["ownership_by_pair"])
    except (TypeError, ValueError) as exc:
        raise G17CompilerPlacementError("Pose ownership discriminator is invalid") from exc
    packet = None
    if value["explicit_preimage_packet"] is not None:
        packet = _parse_nested_document(value["explicit_preimage_packet"], name="Pose preimage")
    ownership = G17PosePreimageOwnershipV1(
        population=population,
        ownership_by_pair=owners,
        physical_coding_group_id_by_pair=tuple(value["physical_coding_group_id_by_pair"]),
        receiver_operation_by_pair=tuple(value["receiver_operation_by_pair"]),
        explicit_preimage_packet=packet,
    )
    if value["ownership_receipt_sha256"] != ownership.ownership_receipt_sha256:
        raise G17CompilerPlacementError("Pose ownership identity did not recompute")
    return ownership


def _r10_payload(relay: G17R10ProsodyFeatureRelayV1 | None) -> list[dict[str, Any]] | None:
    if relay is None:
        return None
    return [
        {
            "constraint": row.constraint.value,
            "frame_role": row.frame_role.value,
            "scientific_role": row.scientific_role.value,
            "semantic_role": row.semantic_role.value,
            "exact_value_b64": _b64(row.exact_value_bytes),
            "support": [[item.pair_id, item.coordinate_id] for item in row.support],
            "tolerance_hex": _float_hex(row.tolerance, name="R10 tolerance"),
            "exact_frozen_block_b64": _b64(row.exact_frozen_block_bytes),
            "exact_chronology_receipt_b64": _b64(row.exact_chronology_receipt_bytes),
            "generic_receiver_operation": row.generic_receiver_operation,
            "physical_coding_group_id": row.physical_coding_group_id,
            "counted_operand_offset": row.counted_operand_offset,
            "counted_operand_b64": _b64(row.counted_operand_bytes),
            "receipt_sha256": row.receipt_sha256,
        }
        for row in relay.constraint_coordinates
    ]


def _parse_r10_payload(
    value: object,
    *,
    population: G17PairPopulationV1,
) -> G17R10ProsodyFeatureRelayV1 | None:
    if value is None:
        return None
    if type(value) is not list:
        raise G17CompilerPlacementError("R10 receipt must be null or a list")
    fields_expected = {
        "constraint",
        "frame_role",
        "scientific_role",
        "semantic_role",
        "exact_value_b64",
        "support",
        "tolerance_hex",
        "exact_frozen_block_b64",
        "exact_chronology_receipt_b64",
        "generic_receiver_operation",
        "physical_coding_group_id",
        "counted_operand_offset",
        "counted_operand_b64",
        "receipt_sha256",
    }
    rows: list[G17R10ConstraintCoordinateV1] = []
    for item in value:
        if type(item) is not dict or set(item) != fields_expected or type(item["support"]) is not list:
            raise G17CompilerPlacementError("R10 coordinate receipt fields are not exact")
        try:
            row = G17R10ConstraintCoordinateV1(
                constraint=G17R10ConstraintV1(item["constraint"]),
                population=population,
                frame_role=G17FrameRoleV1(item["frame_role"]),
                scientific_role=G17ScientificRoleV1(item["scientific_role"]),
                semantic_role=G17SemanticStreamRoleV1(item["semantic_role"]),
                exact_value_bytes=_unb64(item["exact_value_b64"], name="R10 value"),
                support=tuple(_coordinate(row_value, name="R10 support") for row_value in item["support"]),
                tolerance=_parse_float_hex(item["tolerance_hex"], name="R10 tolerance"),
                exact_frozen_block_bytes=_unb64(item["exact_frozen_block_b64"], name="R10 frozen block"),
                exact_chronology_receipt_bytes=_unb64(
                    item["exact_chronology_receipt_b64"], name="R10 chronology receipt"
                ),
                generic_receiver_operation=item["generic_receiver_operation"],
                physical_coding_group_id=item["physical_coding_group_id"],
                counted_operand_offset=item["counted_operand_offset"],
                counted_operand_bytes=_unb64(item["counted_operand_b64"], name="R10 counted operand"),
            )
        except (TypeError, ValueError) as exc:
            raise G17CompilerPlacementError("R10 coordinate typed semantics are invalid") from exc
        if item["receipt_sha256"] != row.receipt_sha256:
            raise G17CompilerPlacementError("R10 coordinate identity did not recompute")
        rows.append(row)
    return G17R10ProsodyFeatureRelayV1(tuple(rows))


def _proof_payload(proof: G17ProofDependencySetV1) -> dict[str, Any]:
    return {
        "proof_kind": proof.proof_kind.value,
        "dependencies": [
            {"domain": item.domain.value, "exact_bytes_b64": _b64(item.exact_dependency_bytes)}
            for item in proof.dependencies
        ],
        "declared_external_reads": [item.value for item in proof.declared_external_reads],
        "identity_sha256": proof.identity_sha256,
    }


def _parse_proof_payload(value: object) -> G17ProofDependencySetV1:
    if type(value) is not dict or set(value) != {
        "proof_kind",
        "dependencies",
        "declared_external_reads",
        "identity_sha256",
    }:
        raise G17CompilerPlacementError("proof receipt fields are not exact")
    if type(value["dependencies"]) is not list or type(value["declared_external_reads"]) is not list:
        raise G17CompilerPlacementError("proof receipt collections are not lists")
    dependencies: list[G17ProofDependencyV1] = []
    for item in value["dependencies"]:
        if type(item) is not dict or set(item) != {"domain", "exact_bytes_b64"}:
            raise G17CompilerPlacementError("proof dependency receipt fields are not exact")
        try:
            domain = G17ProofDependencyDomainV1(item["domain"])
        except (TypeError, ValueError) as exc:
            raise G17CompilerPlacementError("proof dependency domain is invalid") from exc
        dependencies.append(G17ProofDependencyV1(domain, _unb64(item["exact_bytes_b64"], name=domain.value)))
    try:
        proof = G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1(value["proof_kind"]),
            dependencies=tuple(dependencies),
            declared_external_reads=tuple(
                G17ProofDependencyDomainV1(item) for item in value["declared_external_reads"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise G17CompilerPlacementError("proof dependency set typed semantics are invalid") from exc
    if value["identity_sha256"] != proof.identity_sha256:
        raise G17CompilerPlacementError("proof dependency identity did not recompute")
    return proof


def _member_map(archive_bytes: bytes) -> tuple[tuple[str, str, int], ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as opened:
            names = opened.namelist()
            if len(names) != len(set(names)):
                raise G17CompilerPlacementError("archive contains duplicate member names")
            return tuple((name, _sha256(payload), len(payload)) for name in names for payload in (opened.read(name),))
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise G17CompilerPlacementError("whole-state receipt archive failed actual ZIP reopen") from exc


def _strict_state_identity(identities: dict[str, str]) -> str:
    return _sha256(b"G17-STRICT-WHOLE-OBJECT-STATE-RECEIPT-V1\0" + _canonical_json(identities))


@dataclass(frozen=True, slots=True, init=False)
class G17WholeObjectStateReceiptV1:
    """Strict research-evidence receipt for one G17 whole-object state.

    Consumers MUST parse the exact receipt bytes.  The Python instance alone is
    not an authority capability.  This receipt deliberately identifies G17's
    private decoded chronology and records that no G29 public-RGB bridge has
    been proven.
    """

    schema: Literal["tac.g17_whole_object_state_receipt.v1"]
    exact_receipt_bytes: bytes = field(repr=False)
    archive_bytes: bytes = field(repr=False)
    decoded_output_bytes: bytes = field(repr=False)
    placement_manifest: G17CompilerPlacementManifestV1
    population: G17PairPopulationV1
    observation_receipt_bytes: bytes = field(repr=False)
    authority_class: G17AuthorityClassV1
    axis: C0BScoreAxisV1
    sample_count: int
    aggregate_d_seg: float
    aggregate_d_pose: float
    archive_nbytes: int
    total_score: float
    competitive_target_identity_sha256: str
    continuation_basis_sha256: str
    state_identity_sha256: str
    public_rgb_bridge_proven: Literal[False]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("whole-object receipt must be built or strict-parsed")

    @classmethod
    def _from_verified(cls, *, seal: object, **values: object) -> G17WholeObjectStateReceiptV1:
        if seal is not _G17_RECEIPT_SEAL:
            raise G17CompilerPlacementError("whole-object receipt construction is sealed")
        instance = cls.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance

    def to_receipt_bytes(self) -> bytes:
        return self.exact_receipt_bytes

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.exact_receipt_bytes)


def _strict_score_proof(
    score_receipt: C0BScoreReceiptV1,
    *,
    archive_bytes: bytes,
    strict_decode_identity_sha256: str,
) -> G17ProofDependencySetV1:
    source = {item.domain: item.exact_dependency_bytes for item in score_receipt.proof_dependencies.dependencies}
    if source[G17ProofDependencyDomainV1.ARCHIVE_BYTES] != archive_bytes:
        raise G17CompilerPlacementError("score proof archive dependency differs from exact archive")
    if source[G17ProofDependencyDomainV1.FROZEN_SCORER] != score_receipt.authority.exact_scorer_identity_bytes:
        raise G17CompilerPlacementError("score proof scorer dependency differs from authority")
    if source[G17ProofDependencyDomainV1.SCORER_RUNTIME] != score_receipt.authority.exact_runtime_identity_bytes:
        raise G17CompilerPlacementError("score proof runtime dependency differs from authority")
    return G17ProofDependencySetV1(
        proof_kind=G17ProofKindV1.SCORE_OBSERVATION,
        dependencies=tuple(
            sorted(
                (
                    G17ProofDependencyV1(G17ProofDependencyDomainV1.ARCHIVE_BYTES, archive_bytes),
                    G17ProofDependencyV1(
                        G17ProofDependencyDomainV1.DECODE_RECEIPT,
                        bytes.fromhex(strict_decode_identity_sha256),
                    ),
                    G17ProofDependencyV1(
                        G17ProofDependencyDomainV1.FROZEN_SCORER,
                        score_receipt.authority.exact_scorer_identity_bytes,
                    ),
                    G17ProofDependencyV1(
                        G17ProofDependencyDomainV1.SCORER_RUNTIME,
                        score_receipt.authority.exact_runtime_identity_bytes,
                    ),
                    G17ProofDependencyV1(
                        G17ProofDependencyDomainV1.AXIS_AND_SAMPLE_SCOPE,
                        source[G17ProofDependencyDomainV1.AXIS_AND_SAMPLE_SCOPE],
                    ),
                ),
                key=lambda item: item.domain.value,
            )
        ),
        declared_external_reads=(),
    )


def build_g17_whole_object_state_receipt(
    state: G17WholeObjectStateV1,
) -> G17WholeObjectStateReceiptV1:
    """Derive strict durable evidence from a real, internally coherent G17 state."""

    if type(state) is not G17WholeObjectStateV1:
        raise G17CompilerPlacementError("whole-object receipt requires the exact G17 state type")
    score = state.score_receipt
    if type(score) is not C0BScoreReceiptV1 or type(score.authority) is not G17ResearchAuthorityEvidenceV1:
        raise G17CompilerPlacementError(
            "V1 strict whole-object receipts support research authority only; contest authority adapter is owed"
        )
    decode = score.decode_receipt
    if type(decode) is not C0BDecodeReceiptV1:
        raise G17CompilerPlacementError("whole-object state lacks the exact decode receipt parent")
    artifact = decode.archive_artifact
    if type(artifact) is not C0BArchiveArtifactV1:
        raise G17CompilerPlacementError("whole-object state lacks the exact archive parent")
    archive_bytes = artifact.archive_bytes
    member_bytes = artifact.member_bytes
    observation = score.observation
    if type(observation) is not G17CandidateForwardObservationV1:
        raise G17CompilerPlacementError("whole-object state lacks a typed candidate observation")
    if (
        observation.archive_bytes != archive_bytes
        or observation.member_bytes != member_bytes
        or observation.decoded_output_bytes != decode.decoded_output_bytes
        or observation.receiver_receipt_bytes != decode.receiver_receipt.exact_packet_bytes
    ):
        raise G17CompilerPlacementError("whole-object observation differs from archive/decode parents")
    parsed_observation = parse_g17_candidate_forward_receipt(observation.receipt.to_receipt_bytes())
    if parsed_observation != observation.receipt:
        raise G17CompilerPlacementError("candidate observation receipt did not strict-reopen")
    expected_terms = (
        seg_term(parsed_observation.aggregate_d_seg),
        pose_term(parsed_observation.aggregate_d_pose),
        rate_term(len(archive_bytes)),
        compute_contest_score(
            parsed_observation.aggregate_d_seg, parsed_observation.aggregate_d_pose, len(archive_bytes)
        ),
    )
    if expected_terms != (score.d_seg_term, score.d_pose_term, score.rate_term, score.total_score):
        raise G17CompilerPlacementError("whole-object score terms do not match exact upstream operation order")
    if score.authority.sample_count != len(parsed_observation.source_pair_ids):
        raise G17CompilerPlacementError("whole-object authority sample count differs from observation")
    if score.authority.evidence_receipt.exact_packet_bytes != observation.receipt.to_receipt_bytes():
        raise G17CompilerPlacementError("whole-object authority evidence differs from observation receipt")
    if score.axis is not C0BScoreAxisV1.MACOS_CPU_ADVISORY:
        raise G17CompilerPlacementError("research whole-object receipt has a non-advisory score axis")
    # Reconstruct the placement object now, before serializing it, to catch
    # object.__setattr__ mutation of a frozen parent graph.
    placement_payload = _placement_payload(artifact.placement_manifest)
    reparsed_manifest = _parse_placement_payload(
        placement_payload,
        archive_bytes=archive_bytes,
        member_name=artifact.member_name,
        member_bytes=member_bytes,
    )
    if reparsed_manifest.manifest_sha256 != artifact.placement_manifest.manifest_sha256:
        raise G17CompilerPlacementError("whole-object placement changed under strict reconstruction")
    population = artifact.realized_pair.obligation_ir.population
    coverage = artifact.realized_pair.obligation_ir.coverage
    pose_ownership = artifact.realized_pair.obligation_ir.pose_preimage_ownership
    r10_relay = artifact.realized_pair.obligation_ir.r10_relay
    continuation_core = {
        "population": _population_payload(population),
        "obligation_coverage": _obligation_payload(coverage),
        "pose_ownership": _pose_ownership_payload(pose_ownership),
        "r10_relay": _r10_payload(r10_relay),
        "decoder_program_b64": _b64(artifact.decoder_program_bytes),
        "decoder_runtime_b64": _b64(artifact.decoder_runtime_bytes),
        "placement_manifest_sha256": reparsed_manifest.manifest_sha256,
    }
    continuation_basis_sha256 = _sha256(b"G17-CONTINUATION-BASIS-V1\0" + _canonical_json(continuation_core))
    archive_representation_identity = _sha256(
        b"G17-STRICT-ARCHIVE-REPRESENTATION-V1\0"
        + archive_bytes
        + artifact.member_name.encode("ascii")
        + b"\0"
        + member_bytes
        + bytes.fromhex(reparsed_manifest.manifest_sha256)
        + artifact.decoder_program_bytes
        + artifact.decoder_runtime_bytes
    )
    decode_proof = decode.proof_dependencies
    # Exact public/private distinction: this is G17 chronology, not G29 raw RGB.
    strict_decode_identity = _sha256(
        b"G17-STRICT-PRIVATE-DECODE-V1\0"
        + bytes.fromhex(archive_representation_identity)
        + decode.decoded_output_bytes
        + decode.receiver_receipt.exact_packet_bytes
        + bytes.fromhex(decode_proof.identity_sha256)
    )
    strict_score_proof = _strict_score_proof(
        score,
        archive_bytes=archive_bytes,
        strict_decode_identity_sha256=strict_decode_identity,
    )
    target = state.competitive_target
    if type(target) is not G17CompetitiveTargetIdentityV1:
        raise G17CompilerPlacementError("whole-object state lacks a typed competitive target")
    target_payload = {
        "competition_namespace": target.competition_namespace,
        "metric_namespace": target.metric_namespace,
        "selection_policy": target.selection_policy,
        "exact_evaluator_rules_b64": _b64(target.exact_evaluator_rules_bytes),
        "identity_sha256": target.identity_sha256,
    }
    realized_labels = np.asarray(observation.realized_seg_labels)
    target_labels = np.asarray(observation.target.seg_labels)
    seg_mismatch_count = int(np.count_nonzero(realized_labels != target_labels))
    seg_element_count = int(realized_labels.size)
    realized_pose = np.asarray(observation.realized_pose6, dtype=np.float64)
    target_pose = np.asarray(observation.target.pose6, dtype=np.float64)
    pose_squared_error_sum = float(np.sum(np.square(realized_pose - target_pose), dtype=np.float64))
    pose_element_count = int(realized_pose.size)
    if seg_mismatch_count / seg_element_count != parsed_observation.aggregate_d_seg:
        raise G17CompilerPlacementError("segmentation sufficient statistics do not recompute aggregate")
    if pose_squared_error_sum / pose_element_count != parsed_observation.aggregate_d_pose:
        raise G17CompilerPlacementError("Pose sufficient statistics do not recompute aggregate")
    authority_identity = score.authority.authority_sha256
    ownership_identity = _sha256(
        b"G17-STRICT-OWNERSHIP-SET-V1\0" + _canonical_json(sorted(reparsed_manifest.expected_object_identities))
    )
    score_identity = _sha256(
        b"G17-STRICT-SCORE-V1\0"
        + bytes.fromhex(strict_decode_identity)
        + struct.pack(">dddd", *expected_terms)
        + bytes.fromhex(strict_score_proof.identity_sha256)
        + bytes.fromhex(authority_identity)
    )
    identity_inputs = {
        "archive_representation_identity_sha256": archive_representation_identity,
        "authority_identity_sha256": authority_identity,
        "competitive_target_identity_sha256": target.identity_sha256,
        "continuation_basis_sha256": continuation_basis_sha256,
        "decode_proof_dependency_identity_sha256": decode_proof.identity_sha256,
        "observation_receipt_sha256": observation.receipt.receipt_sha256,
        "ownership_identity_sha256": ownership_identity,
        "placement_manifest_sha256": reparsed_manifest.manifest_sha256,
        "private_decode_identity_sha256": strict_decode_identity,
        "score_identity_sha256": score_identity,
        "score_proof_dependency_identity_sha256": strict_score_proof.identity_sha256,
    }
    state_identity = _strict_state_identity(identity_inputs)
    archive_map = _member_map(archive_bytes)
    payload = {
        "schema": G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA_V1,
        "role": _G17_WHOLE_OBJECT_STATE_RECEIPT_ROLE_V1,
        "research_only": True,
        "candidate_payload_allowed": False,
        "public_rgb_bridge_proven": False,
        "archive": {
            "exact_archive_b64": _b64(archive_bytes),
            "archive_sha256": _sha256(archive_bytes),
            "archive_nbytes": len(archive_bytes),
            "member_name": artifact.member_name,
            "exact_member_b64": _b64(member_bytes),
            "member_sha256": _sha256(member_bytes),
            "member_map": [list(item) for item in archive_map],
        },
        "placement": placement_payload,
        "continuation": {**continuation_core, "continuation_basis_sha256": continuation_basis_sha256},
        "decode": {
            "private_intermediate_only": True,
            "exact_decoded_output_b64": _b64(decode.decoded_output_bytes),
            "decoded_output_sha256": _sha256(decode.decoded_output_bytes),
            "receiver_receipt": _strict_nested_document_payload(decode.receiver_receipt),
            "private_decode_identity_sha256": strict_decode_identity,
        },
        "observation": {
            "exact_candidate_receipt_b64": _b64(observation.receipt.to_receipt_bytes()),
            "seg_mismatch_count": seg_mismatch_count,
            "seg_element_count": seg_element_count,
            "pose_squared_error_sum_hex": _float_hex(pose_squared_error_sum, name="Pose squared-error sum"),
            "pose_element_count": pose_element_count,
        },
        "score": {
            "aggregate_d_seg_hex": _float_hex(parsed_observation.aggregate_d_seg, name="aggregate d_seg"),
            "aggregate_d_pose_hex": _float_hex(parsed_observation.aggregate_d_pose, name="aggregate d_pose"),
            "d_seg_term_hex": _float_hex(score.d_seg_term, name="d_seg term"),
            "d_pose_term_hex": _float_hex(score.d_pose_term, name="d_pose term"),
            "rate_term_hex": _float_hex(score.rate_term, name="rate term"),
            "total_score_hex": _float_hex(score.total_score, name="total score"),
            "operation_order": _G17_SCORE_OPERATION_ORDER_V1,
            "score_identity_sha256": score_identity,
        },
        "authority": {
            "authority_class": score.authority.authority_class.value,
            "axis": score.axis.value,
            "sample_count": score.authority.sample_count,
            "axis_label": score.authority.axis_label,
            "evidence_receipt": _strict_nested_document_payload(score.authority.evidence_receipt),
            "exact_hardware_identity_b64": _b64(score.authority.exact_hardware_identity_bytes),
            "exact_scorer_identity_b64": _b64(score.authority.exact_scorer_identity_bytes),
            "exact_runtime_identity_b64": _b64(score.authority.exact_runtime_identity_bytes),
            "authority_identity_sha256": authority_identity,
        },
        "proof_dependencies": {
            "decode": _proof_payload(decode_proof),
            "score": _proof_payload(strict_score_proof),
        },
        "competitive_target": target_payload,
        "identities": {**identity_inputs, "state_identity_sha256": state_identity},
    }
    return parse_g17_whole_object_state_receipt(_canonical_json(payload))


def parse_g17_whole_object_state_receipt(payload: bytes) -> G17WholeObjectStateReceiptV1:
    """Strictly reopen, rederive, and re-emit a G17 whole-object receipt."""

    decoded = _decode_canonical_json_object(payload, name="G17 whole-object state receipt")
    expected_top = {
        "schema",
        "role",
        "research_only",
        "candidate_payload_allowed",
        "public_rgb_bridge_proven",
        "archive",
        "placement",
        "continuation",
        "decode",
        "observation",
        "score",
        "authority",
        "proof_dependencies",
        "competitive_target",
        "identities",
    }
    if set(decoded) != expected_top:
        raise G17CompilerPlacementError("whole-object receipt field set is not exact")
    if (
        decoded["schema"] != G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA_V1
        or decoded["role"] != _G17_WHOLE_OBJECT_STATE_RECEIPT_ROLE_V1
        or decoded["research_only"] is not True
        or decoded["candidate_payload_allowed"] is not False
        or decoded["public_rgb_bridge_proven"] is not False
    ):
        raise G17CompilerPlacementError("whole-object receipt role or authority boundary changed")
    archive = decoded["archive"]
    if type(archive) is not dict or set(archive) != {
        "exact_archive_b64",
        "archive_sha256",
        "archive_nbytes",
        "member_name",
        "exact_member_b64",
        "member_sha256",
        "member_map",
    }:
        raise G17CompilerPlacementError("whole-object archive fields are not exact")
    archive_bytes = _unb64(archive["exact_archive_b64"], name="whole-object archive")
    member_bytes = _unb64(archive["exact_member_b64"], name="whole-object member")
    member_name = _ascii(archive["member_name"], name="whole-object member name")
    if (
        type(archive["archive_nbytes"]) is not int
        or archive["archive_nbytes"] != len(archive_bytes)
        or archive["archive_sha256"] != _sha256(archive_bytes)
        or archive["member_sha256"] != _sha256(member_bytes)
    ):
        raise G17CompilerPlacementError("whole-object archive identity or length did not recompute")
    actual_map = _member_map(archive_bytes)
    if type(archive["member_map"]) is not list or archive["member_map"] != [list(item) for item in actual_map]:
        raise G17CompilerPlacementError("whole-object archive member map did not recompute")
    actual_by_name = {name: (digest, length) for name, digest, length in actual_map}
    if actual_by_name.get(member_name) != (_sha256(member_bytes), len(member_bytes)):
        raise G17CompilerPlacementError("whole-object selected member differs from archive bytes")
    manifest = _parse_placement_payload(
        decoded["placement"],
        archive_bytes=archive_bytes,
        member_name=member_name,
        member_bytes=member_bytes,
    )
    continuation = decoded["continuation"]
    continuation_fields = {
        "population",
        "obligation_coverage",
        "pose_ownership",
        "r10_relay",
        "decoder_program_b64",
        "decoder_runtime_b64",
        "placement_manifest_sha256",
        "continuation_basis_sha256",
    }
    if type(continuation) is not dict or set(continuation) != continuation_fields:
        raise G17CompilerPlacementError("continuation receipt fields are not exact")
    population = _parse_population_payload(continuation["population"])
    _parse_obligation_payload(continuation["obligation_coverage"], population=population)
    _parse_pose_ownership_payload(continuation["pose_ownership"], population=population)
    _parse_r10_payload(continuation["r10_relay"], population=population)
    _unb64(continuation["decoder_program_b64"], name="decoder program")
    _unb64(continuation["decoder_runtime_b64"], name="decoder runtime")
    if continuation["placement_manifest_sha256"] != manifest.manifest_sha256:
        raise G17CompilerPlacementError("continuation names another placement manifest")
    continuation_core = {name: continuation[name] for name in continuation_fields - {"continuation_basis_sha256"}}
    continuation_identity = _sha256(b"G17-CONTINUATION-BASIS-V1\0" + _canonical_json(continuation_core))
    if continuation["continuation_basis_sha256"] != continuation_identity:
        raise G17CompilerPlacementError("continuation basis identity did not recompute")
    decode = decoded["decode"]
    if (
        type(decode) is not dict
        or set(decode)
        != {
            "private_intermediate_only",
            "exact_decoded_output_b64",
            "decoded_output_sha256",
            "receiver_receipt",
            "private_decode_identity_sha256",
        }
        or decode["private_intermediate_only"] is not True
    ):
        raise G17CompilerPlacementError("private decode receipt fields are not exact")
    decoded_output_bytes = _unb64(decode["exact_decoded_output_b64"], name="private decoded output")
    receiver_receipt = _parse_nested_document(decode["receiver_receipt"], name="receiver receipt")
    if decode["decoded_output_sha256"] != _sha256(decoded_output_bytes):
        raise G17CompilerPlacementError("private decoded-output identity did not recompute")
    observation = decoded["observation"]
    if type(observation) is not dict or set(observation) != {
        "exact_candidate_receipt_b64",
        "seg_mismatch_count",
        "seg_element_count",
        "pose_squared_error_sum_hex",
        "pose_element_count",
    }:
        raise G17CompilerPlacementError("observation receipt fields are not exact")
    observation_bytes = _unb64(observation["exact_candidate_receipt_b64"], name="candidate observation receipt")
    try:
        observation_receipt = parse_g17_candidate_forward_receipt(observation_bytes)
    except Exception as exc:
        raise G17CompilerPlacementError("candidate observation failed its strict public parser") from exc
    if (
        observation_receipt.archive_sha256 != _sha256(archive_bytes)
        or observation_receipt.member_sha256 != _sha256(member_bytes)
        or observation_receipt.receiver_receipt_sha256 != receiver_receipt.packet_sha256
        or observation_receipt.decoded_output_sha256 != _sha256(decoded_output_bytes)
    ):
        raise G17CompilerPlacementError("candidate observation names another archive/decode object")
    for name in ("seg_mismatch_count", "seg_element_count", "pose_element_count"):
        if type(observation[name]) is not int:
            raise G17CompilerPlacementError(f"{name} must be an exact integer")
    if (
        observation["seg_element_count"] <= 0
        or not 0 <= observation["seg_mismatch_count"] <= observation["seg_element_count"]
        or observation["pose_element_count"] <= 0
    ):
        raise G17CompilerPlacementError("observation sufficient-statistic counts are invalid")
    recomputed_d_seg = observation["seg_mismatch_count"] / observation["seg_element_count"]
    pose_sse = _parse_float_hex(observation["pose_squared_error_sum_hex"], name="Pose squared-error sum")
    recomputed_d_pose = pose_sse / observation["pose_element_count"]
    score = decoded["score"]
    if (
        type(score) is not dict
        or set(score)
        != {
            "aggregate_d_seg_hex",
            "aggregate_d_pose_hex",
            "d_seg_term_hex",
            "d_pose_term_hex",
            "rate_term_hex",
            "total_score_hex",
            "operation_order",
            "score_identity_sha256",
        }
        or score["operation_order"] != _G17_SCORE_OPERATION_ORDER_V1
    ):
        raise G17CompilerPlacementError("score receipt fields or operation order changed")
    aggregate_d_seg = _parse_float_hex(score["aggregate_d_seg_hex"], name="aggregate d_seg")
    aggregate_d_pose = _parse_float_hex(score["aggregate_d_pose_hex"], name="aggregate d_pose")
    if (
        aggregate_d_seg != recomputed_d_seg
        or aggregate_d_pose != recomputed_d_pose
        or aggregate_d_seg != observation_receipt.aggregate_d_seg
        or aggregate_d_pose != observation_receipt.aggregate_d_pose
    ):
        raise G17CompilerPlacementError("observation aggregates did not recompute from typed sufficient statistics")
    expected_terms = (
        seg_term(aggregate_d_seg),
        pose_term(aggregate_d_pose),
        rate_term(len(archive_bytes)),
        compute_contest_score(aggregate_d_seg, aggregate_d_pose, len(archive_bytes)),
    )
    recorded_terms = tuple(
        _parse_float_hex(score[name], name=name)
        for name in ("d_seg_term_hex", "d_pose_term_hex", "rate_term_hex", "total_score_hex")
    )
    if recorded_terms != expected_terms:
        raise G17CompilerPlacementError("score components do not match exact upstream operation order")
    authority = decoded["authority"]
    authority_fields = {
        "authority_class",
        "axis",
        "sample_count",
        "axis_label",
        "evidence_receipt",
        "exact_hardware_identity_b64",
        "exact_scorer_identity_b64",
        "exact_runtime_identity_b64",
        "authority_identity_sha256",
    }
    if type(authority) is not dict or set(authority) != authority_fields:
        raise G17CompilerPlacementError("authority receipt fields are not exact")
    if (
        authority["authority_class"] != G17AuthorityClassV1.RESEARCH_ADVISORY.value
        or authority["axis"] != C0BScoreAxisV1.MACOS_CPU_ADVISORY.value
        or type(authority["sample_count"]) is not int
        or authority["sample_count"] != len(observation_receipt.source_pair_ids)
    ):
        raise G17CompilerPlacementError("authority class, axis, or sample scope is invalid")
    evidence_receipt = _parse_nested_document(authority["evidence_receipt"], name="authority evidence")
    if evidence_receipt.exact_packet_bytes != observation_bytes:
        raise G17CompilerPlacementError("authority evidence differs from candidate observation")
    hardware_bytes = _unb64(authority["exact_hardware_identity_b64"], name="authority hardware")
    scorer_bytes = _unb64(authority["exact_scorer_identity_b64"], name="authority scorer")
    runtime_bytes = _unb64(authority["exact_runtime_identity_b64"], name="authority runtime")
    axis_label = _ascii(authority["axis_label"], name="authority axis label")
    if (
        _sha256(scorer_bytes) != observation_receipt.frozen_scorer_sha256
        or _sha256(runtime_bytes) != observation_receipt.scorer_runtime_environment_sha256
    ):
        raise G17CompilerPlacementError("authority scorer/runtime bytes differ from observation custody")
    authority_identity = _sha256(
        b"G17-RESEARCH-AUTHORITY-V1\0"
        + bytes.fromhex(evidence_receipt.packet_sha256)
        + authority["sample_count"].to_bytes(2, "big")
        + axis_label.encode("ascii")
        + b"\0"
        + hardware_bytes
        + scorer_bytes
        + runtime_bytes
    )
    if authority["authority_identity_sha256"] != authority_identity:
        raise G17CompilerPlacementError("authority identity did not recompute")
    proofs = decoded["proof_dependencies"]
    if type(proofs) is not dict or set(proofs) != {"decode", "score"}:
        raise G17CompilerPlacementError("proof dependency receipt fields are not exact")
    decode_proof = _parse_proof_payload(proofs["decode"])
    decode_dependencies = {item.domain: item.exact_dependency_bytes for item in decode_proof.dependencies}
    if (
        decode_dependencies[G17ProofDependencyDomainV1.ARCHIVE_BYTES] != archive_bytes
        or decode_dependencies[G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING] != _canonical_json(actual_map)
        or decode_dependencies[G17ProofDependencyDomainV1.PAIR_ORDER]
        != struct.pack(">" + "H" * len(population.source_pair_ids), *population.source_pair_ids)
    ):
        raise G17CompilerPlacementError("decode proof dependencies do not describe this archive/population")
    archive_representation_identity = _sha256(
        b"G17-STRICT-ARCHIVE-REPRESENTATION-V1\0"
        + archive_bytes
        + member_name.encode("ascii")
        + b"\0"
        + member_bytes
        + bytes.fromhex(manifest.manifest_sha256)
        + _unb64(continuation["decoder_program_b64"], name="decoder program")
        + _unb64(continuation["decoder_runtime_b64"], name="decoder runtime")
    )
    private_decode_identity = _sha256(
        b"G17-STRICT-PRIVATE-DECODE-V1\0"
        + bytes.fromhex(archive_representation_identity)
        + decoded_output_bytes
        + receiver_receipt.exact_packet_bytes
        + bytes.fromhex(decode_proof.identity_sha256)
    )
    if decode["private_decode_identity_sha256"] != private_decode_identity:
        raise G17CompilerPlacementError("private decode identity did not recompute")
    score_proof = _parse_proof_payload(proofs["score"])
    score_dependencies = {item.domain: item.exact_dependency_bytes for item in score_proof.dependencies}
    scope_expected = _canonical_json(
        {
            "authority_class": authority["authority_class"],
            "axis_label": axis_label,
            "sample_count": authority["sample_count"],
            "hardware_sha256": _sha256(hardware_bytes),
        }
    )
    if (
        score_dependencies[G17ProofDependencyDomainV1.ARCHIVE_BYTES] != archive_bytes
        or score_dependencies[G17ProofDependencyDomainV1.DECODE_RECEIPT] != bytes.fromhex(private_decode_identity)
        or score_dependencies[G17ProofDependencyDomainV1.FROZEN_SCORER] != scorer_bytes
        or score_dependencies[G17ProofDependencyDomainV1.SCORER_RUNTIME] != runtime_bytes
        or score_dependencies[G17ProofDependencyDomainV1.AXIS_AND_SAMPLE_SCOPE] != scope_expected
    ):
        raise G17CompilerPlacementError("score proof dependencies do not describe this state")
    target = decoded["competitive_target"]
    if type(target) is not dict or set(target) != {
        "competition_namespace",
        "metric_namespace",
        "selection_policy",
        "exact_evaluator_rules_b64",
        "identity_sha256",
    }:
        raise G17CompilerPlacementError("competitive-target receipt fields are not exact")
    competitive_target = G17CompetitiveTargetIdentityV1(
        competition_namespace=target["competition_namespace"],
        metric_namespace=target["metric_namespace"],
        selection_policy=target["selection_policy"],
        exact_evaluator_rules_bytes=_unb64(target["exact_evaluator_rules_b64"], name="competitive evaluator rules"),
    )
    if target["identity_sha256"] != competitive_target.identity_sha256:
        raise G17CompilerPlacementError("competitive-target identity did not recompute")
    ownership_identity = _sha256(
        b"G17-STRICT-OWNERSHIP-SET-V1\0" + _canonical_json(sorted(manifest.expected_object_identities))
    )
    score_identity = _sha256(
        b"G17-STRICT-SCORE-V1\0"
        + bytes.fromhex(private_decode_identity)
        + struct.pack(">dddd", *expected_terms)
        + bytes.fromhex(score_proof.identity_sha256)
        + bytes.fromhex(authority_identity)
    )
    if score["score_identity_sha256"] != score_identity:
        raise G17CompilerPlacementError("strict score identity did not recompute")
    identities = decoded["identities"]
    expected_identity_fields = {
        "archive_representation_identity_sha256",
        "authority_identity_sha256",
        "competitive_target_identity_sha256",
        "continuation_basis_sha256",
        "decode_proof_dependency_identity_sha256",
        "observation_receipt_sha256",
        "ownership_identity_sha256",
        "placement_manifest_sha256",
        "private_decode_identity_sha256",
        "score_identity_sha256",
        "score_proof_dependency_identity_sha256",
        "state_identity_sha256",
    }
    if type(identities) is not dict or set(identities) != expected_identity_fields:
        raise G17CompilerPlacementError("whole-object identity field set is not exact")
    recomputed_identity_inputs = {
        "archive_representation_identity_sha256": archive_representation_identity,
        "authority_identity_sha256": authority_identity,
        "competitive_target_identity_sha256": competitive_target.identity_sha256,
        "continuation_basis_sha256": continuation_identity,
        "decode_proof_dependency_identity_sha256": decode_proof.identity_sha256,
        "observation_receipt_sha256": _sha256(observation_bytes),
        "ownership_identity_sha256": ownership_identity,
        "placement_manifest_sha256": manifest.manifest_sha256,
        "private_decode_identity_sha256": private_decode_identity,
        "score_identity_sha256": score_identity,
        "score_proof_dependency_identity_sha256": score_proof.identity_sha256,
    }
    if any(identities[name] != value for name, value in recomputed_identity_inputs.items()):
        raise G17CompilerPlacementError("one or more whole-object identities did not recompute")
    state_identity = _strict_state_identity(recomputed_identity_inputs)
    if identities["state_identity_sha256"] != state_identity:
        raise G17CompilerPlacementError("whole-object state identity did not recompute")
    instance = G17WholeObjectStateReceiptV1._from_verified(
        seal=_G17_RECEIPT_SEAL,
        schema=G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA_V1,
        exact_receipt_bytes=payload,
        archive_bytes=archive_bytes,
        decoded_output_bytes=decoded_output_bytes,
        placement_manifest=manifest,
        population=population,
        observation_receipt_bytes=observation_bytes,
        authority_class=G17AuthorityClassV1.RESEARCH_ADVISORY,
        axis=C0BScoreAxisV1.MACOS_CPU_ADVISORY,
        sample_count=authority["sample_count"],
        aggregate_d_seg=aggregate_d_seg,
        aggregate_d_pose=aggregate_d_pose,
        archive_nbytes=len(archive_bytes),
        total_score=expected_terms[3],
        competitive_target_identity_sha256=competitive_target.identity_sha256,
        continuation_basis_sha256=continuation_identity,
        state_identity_sha256=state_identity,
        public_rgb_bridge_proven=False,
    )
    if instance.to_receipt_bytes() != payload:
        raise G17CompilerPlacementError("whole-object receipt changed under parse/re-emit")
    return instance


class G17CallableActionModeV1(StrEnum):
    PRUNE_DELETE = "PRUNE_DELETE"
    MERGE_SHARE = "MERGE_SHARE"
    REPLACE = "REPLACE"
    MIGRATE = "MIGRATE"
    REQUANTIZE_STORAGE = "REQUANTIZE_STORAGE"
    ANALYTICIZE = "ANALYTICIZE"
    GAUGE_SELECT = "GAUGE_SELECT"
    LEARNED_ESCAPE = "LEARNED_ESCAPE"
    MACRO = "MACRO"


G17WholeObjectCompilerCallbackV1: TypeAlias = Callable[[G17WholeObjectStateV1], G17WholeObjectStateV1]


@dataclass(frozen=True, slots=True)
class G17CallableActionV1:
    action_id: str
    mode: G17CallableActionModeV1
    exact_implementation_bytes: bytes = field(repr=False)
    compiler_callback: G17WholeObjectCompilerCallbackV1 = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _ascii(self.action_id, name="callable action ID")
        if type(self.mode) is not G17CallableActionModeV1:
            raise G17CompilerPlacementError("callable action mode is not typed")
        _exact_bytes(self.exact_implementation_bytes, name="callable action implementation")
        if not callable(self.compiler_callback):
            raise G17CompilerPlacementError("action alternative must be callable")

    def invoke(self, parent: G17WholeObjectStateV1) -> G17WholeObjectStateV1:
        if type(parent) is not G17WholeObjectStateV1:
            raise G17CompilerPlacementError("callable action parent is not a whole-object state")
        result = self.compiler_callback(parent)
        if type(result) is not G17WholeObjectStateV1:
            raise G17CompilerBlocker(
                G17CompilerBlockerCodeV1.G17_REAL_COMPOSED_COUNTERFACTUAL_EVIDENCE_OWED,
                "callable action did not return exact composed archive/decode/score evidence",
            )
        if result.competitive_target is not parent.competitive_target:
            raise G17CompilerPlacementError("action changed semantic CompetitiveTargetIdentity")
        if result.score_receipt.decode_receipt.archive_artifact.archive_bytes == (
            parent.score_receipt.decode_receipt.archive_artifact.archive_bytes
        ):
            raise G17CompilerPlacementError("action returned the unchanged physical archive")
        return result


@dataclass(frozen=True, slots=True)
class G17MeasuredCounterfactualCornerV1:
    active_action_ids: tuple[str, ...]
    whole_object_state: G17WholeObjectStateV1

    def __post_init__(self) -> None:
        if type(self.active_action_ids) is not tuple or not self.active_action_ids:
            raise G17CompilerPlacementError("counterfactual corner must name active actions")
        for action_id in self.active_action_ids:
            _ascii(action_id, name="counterfactual action ID")
        if len(self.active_action_ids) != len(set(self.active_action_ids)):
            raise G17CompilerPlacementError("counterfactual corner repeats an action")
        if type(self.whole_object_state) is not G17WholeObjectStateV1:
            raise G17CompilerPlacementError("counterfactual corner lacks exact whole-object evidence")


def _ordered_nonempty_subsets(action_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(action_ids[index] for index in range(len(action_ids)) if mask & (1 << index))
        for mask in range(1, 1 << len(action_ids))
    )


@dataclass(frozen=True, slots=True)
class G17EffectObservationV1:
    observation_id: str
    kind: G17EffectObservationKindV1
    ordered_action_ids: tuple[str, ...]
    exact_parent: G17WholeObjectStateV1
    measured_corners: tuple[G17MeasuredCounterfactualCornerV1, ...]
    support: G17EffectSupportV1
    terminal_effect_consumer: str
    causal_owner_set: tuple[str, ...]
    verdict_scope: str
    d_seg_interaction_residual: float | None = field(init=False)
    d_pose_interaction_residual: float | None = field(init=False)
    archive_bytes_interaction_residual: int | None = field(init=False)
    joint_score_interaction_residual: float | None = field(init=False)

    def __post_init__(self) -> None:
        _ascii(self.observation_id, name="effect observation ID")
        if type(self.kind) is not G17EffectObservationKindV1 or type(self.support) is not G17EffectSupportV1:
            raise G17CompilerPlacementError("effect kind/support is not typed")
        if type(self.ordered_action_ids) is not tuple or any(
            type(item) is not str or not item or not item.isascii() for item in self.ordered_action_ids
        ):
            raise G17CompilerPlacementError("effect action IDs are invalid")
        if len(self.ordered_action_ids) != len(set(self.ordered_action_ids)):
            raise G17CompilerPlacementError("effect action order repeats an action")
        if type(self.exact_parent) is not G17WholeObjectStateV1:
            raise G17CompilerPlacementError("effect lacks exact parent whole-object state")
        if type(self.measured_corners) is not tuple or any(
            type(item) is not G17MeasuredCounterfactualCornerV1 for item in self.measured_corners
        ):
            raise G17CompilerPlacementError("effect corners are not exact typed states")
        if self.kind is G17EffectObservationKindV1.ENDPOINT and len(self.ordered_action_ids) != 1:
            raise G17CompilerPlacementError("endpoint effect must contain exactly one action")
        if self.kind is G17EffectObservationKindV1.INDIVISIBLE_HYPEREDGE and len(self.ordered_action_ids) < 2:
            raise G17CompilerPlacementError("interaction hyperedge arity must be at least two")
        expected_subsets = _ordered_nonempty_subsets(self.ordered_action_ids)
        observed_subsets = tuple(item.active_action_ids for item in self.measured_corners)
        if len(observed_subsets) != len(set(observed_subsets)) or any(
            item not in expected_subsets for item in observed_subsets
        ):
            raise G17CompilerPlacementError("effect corner lattice is duplicate or foreign")
        if self.support is G17EffectSupportV1.COMPLETE and observed_subsets != expected_subsets:
            raise G17CompilerPlacementError("complete effect omitted or reordered a counterfactual corner")
        if self.support is G17EffectSupportV1.PARTIAL and (
            not observed_subsets or set(observed_subsets) == set(expected_subsets)
        ):
            raise G17CompilerPlacementError("partial effect must retain real but incomplete corner support")
        parent_target = self.exact_parent.competitive_target
        parent_authority = self.exact_parent.score_receipt.authority.authority_class
        for corner in self.measured_corners:
            if corner.whole_object_state.competitive_target is not parent_target:
                raise G17CompilerPlacementError("effect corner changed semantic target identity")
            if corner.whole_object_state.score_receipt.authority.authority_class is not parent_authority:
                raise G17CompilerPlacementError("effect corner launders or mixes authority")
        _ascii(self.terminal_effect_consumer, name="terminal effect consumer")
        if type(self.causal_owner_set) is not tuple or not self.causal_owner_set:
            raise G17CompilerPlacementError("effect causal owner set must be explicit")
        for owner in self.causal_owner_set:
            _ascii(owner, name="effect causal owner")
        if len(self.causal_owner_set) != len(set(self.causal_owner_set)):
            raise G17CompilerPlacementError("effect causal owner set contains aliases")
        _ascii(self.verdict_scope, name="effect verdict scope")
        residuals: tuple[float | int | None, ...]
        if (
            self.kind is G17EffectObservationKindV1.INDIVISIBLE_HYPEREDGE
            and self.support is G17EffectSupportV1.COMPLETE
        ):
            state_by_subset = {corner.active_action_ids: corner.whole_object_state for corner in self.measured_corners}
            empty_state = self.exact_parent

            def mobius(value: Callable[[G17WholeObjectStateV1], float]) -> float:
                total = ((-1.0) ** len(self.ordered_action_ids)) * value(empty_state)
                for subset, state in state_by_subset.items():
                    total += ((-1.0) ** (len(self.ordered_action_ids) - len(subset))) * value(state)
                return float(total)

            residuals = (
                mobius(lambda state: state.score_receipt.observation.receipt.aggregate_d_seg),
                mobius(lambda state: state.score_receipt.observation.receipt.aggregate_d_pose),
                round(
                    mobius(lambda state: float(len(state.score_receipt.decode_receipt.archive_artifact.archive_bytes)))
                ),
                mobius(lambda state: state.score_receipt.total_score),
            )
        else:
            residuals = (None, None, None, None)
        for name, value in zip(
            (
                "d_seg_interaction_residual",
                "d_pose_interaction_residual",
                "archive_bytes_interaction_residual",
                "joint_score_interaction_residual",
            ),
            residuals,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def credit_decomposition(self) -> Literal["FORBIDDEN"]:
        return "FORBIDDEN"

    @property
    def promotion_capable(self) -> Literal[False]:
        return False

    def require_complete_composed_evidence(self) -> None:
        if self.support is not G17EffectSupportV1.COMPLETE:
            raise G17CompilerBlocker(
                G17CompilerBlockerCodeV1.G17_REAL_COMPOSED_COUNTERFACTUAL_EVIDENCE_OWED,
                "partial interaction cannot arbitrate or promote a whole-object state",
            )


def arbitrate_g17_whole_object_states(
    states: Sequence[G17WholeObjectStateV1],
) -> G17WholeObjectStateV1:
    """Choose only from exact composed evidence; no static/per-role payoff."""

    if not states or any(type(item) is not G17WholeObjectStateV1 for item in states):
        raise G17CompilerPlacementError("arbitration requires exact whole-object states")
    target = states[0].competitive_target
    authority = states[0].score_receipt.authority
    for state in states:
        if state.competitive_target is not target:
            raise G17CompilerPlacementError("arbitration mixed semantic target identities")
        if type(state.score_receipt.authority) is not type(authority):
            raise G17CompilerPlacementError("arbitration mixed incomparable authority variants")
        if type(authority) is G17ResearchAuthorityEvidenceV1 and (
            state.score_receipt.authority.sample_count != authority.sample_count
            or state.score_receipt.authority.axis_label != authority.axis_label
            or state.score_receipt.observation.receipt.frozen_scorer_sha256
            != states[0].score_receipt.observation.receipt.frozen_scorer_sha256
        ):
            raise G17CompilerPlacementError("arbitration mixed scorer/axis/sample scope")
    return min(states, key=lambda item: (item.score_receipt.total_score, item.state_sha256))


@dataclass(frozen=True, slots=True, init=False)
class G17BeatsCurrentFrontierV1:
    score_receipt: C0BScoreReceiptV1
    competitive_target: G17CompetitiveTargetIdentityV1
    pointer_snapshot: G17PointerSnapshotV1
    proof_dependencies: G17ProofDependencySetV1

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise G17CompilerPlacementError("frontier admission must be built by compare")

    @classmethod
    def compare(
        cls,
        *,
        score_receipt: C0BScoreReceiptV1,
        competitive_target: G17CompetitiveTargetIdentityV1,
        pointer_snapshot: G17PointerSnapshotV1,
    ) -> G17BeatsCurrentFrontierV1:
        if type(score_receipt) is not C0BScoreReceiptV1:
            raise G17CompilerPlacementError("frontier comparison requires exact ScoreReceipt")
        if type(score_receipt.authority) not in {
            G17ContestCPUAuthorityEvidenceV1,
            G17ContestCUDAAuthorityEvidenceV1,
        }:
            raise G17CompilerPlacementError("research/advisory score cannot support frontier admission")
        if (
            type(competitive_target) is not G17CompetitiveTargetIdentityV1
            or type(pointer_snapshot) is not G17PointerSnapshotV1
        ):
            raise G17CompilerPlacementError("frontier comparison lacks semantic target or pointer snapshot")
        if not score_receipt.total_score < pointer_snapshot.effective_frontier_score:
            raise G17CompilerPlacementError("score does not beat the refreshed effective frontier")
        dependencies = G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1.BEATS_CURRENT_FRONTIER,
            dependencies=tuple(
                sorted(
                    (
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.SCORE_RECEIPT,
                            bytes.fromhex(score_receipt.identity_sha256),
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.SEMANTIC_COMPETITIVE_TARGET,
                            bytes.fromhex(competitive_target.identity_sha256),
                        ),
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.POINTER_SNAPSHOT,
                            pointer_snapshot.exact_pointer_bytes,
                        ),
                    ),
                    key=lambda item: item.domain.value,
                )
            ),
            declared_external_reads=(G17ProofDependencyDomainV1.POINTER_SNAPSHOT,),
        )
        instance = cls.__new__(cls)
        for name, value in (
            ("score_receipt", score_receipt),
            ("competitive_target", competitive_target),
            ("pointer_snapshot", pointer_snapshot),
            ("proof_dependencies", dependencies),
        ):
            object.__setattr__(instance, name, value)
        return instance

    def rebase(self, pointer_snapshot: G17PointerSnapshotV1) -> G17BeatsCurrentFrontierV1:
        return type(self).compare(
            score_receipt=self.score_receipt,
            competitive_target=self.competitive_target,
            pointer_snapshot=pointer_snapshot,
        )


__all__ = [
    "G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA_V1",
    "C0BArchiveArtifactV1",
    "C0BAuthEvalClosureV1",
    "C0BDecodeReceiptV1",
    "C0BObligationIRV1",
    "C0BRealizedPairV1",
    "C0BScoreAxisV1",
    "C0BScoreReceiptV1",
    "C0BSourceTruthV1",
    "G17AnalyticResidualOwnershipV1",
    "G17ArtifactClassV1",
    "G17AuthorityClassV1",
    "G17BeatsCurrentFrontierV1",
    "G17CallableActionModeV1",
    "G17CallableActionV1",
    "G17ChronologicalPosePreimageV1",
    "G17CompetitiveTargetIdentityV1",
    "G17CompilerBlocker",
    "G17CompilerBlockerCodeV1",
    "G17CompilerPlacementError",
    "G17CompilerPlacementManifestV1",
    "G17CompilerPlacementRecordV1",
    "G17ContestCPUAuthorityEvidenceV1",
    "G17ContestCUDAAuthorityEvidenceV1",
    "G17CountedVMBytecodeV1",
    "G17CountedVMOperandV1",
    "G17DeterministicReconstructionProgramV1",
    "G17EffectObservationKindV1",
    "G17EffectObservationV1",
    "G17EffectSupportV1",
    "G17EncoderOnlyTeacherOracleEvidenceV1",
    "G17EntropyContextV1",
    "G17EvaluatorRecursionStageV1",
    "G17ForwardObservationLogicalV1",
    "G17FrameRoleV1",
    "G17FunctionalQuotientIdentityV1",
    "G17GenericVMInterpreterV1",
    "G17LearnedResidualOwnershipV1",
    "G17LifecyclePhaseV1",
    "G17LogicalOwnershipKindV1",
    "G17LogicalOwnershipV1",
    "G17LogicalValueTypeV1",
    "G17MeasuredCounterfactualCornerV1",
    "G17ObligationCoordinateV1",
    "G17ObligationCoverageModeV1",
    "G17ObligationCoverageV1",
    "G17PackagedExecutableV1",
    "G17PairPopulationV1",
    "G17ParameterSpellingIdentityV1",
    "G17PhysicalByteHomeV1",
    "G17PhysicalCodingGroupV1",
    "G17PlacementClassV1",
    "G17PointerSnapshotV1",
    "G17PopulationSharingV1",
    "G17PoseOwnershipV1",
    "G17PosePreimageOwnershipV1",
    "G17ProofDependencyDomainV1",
    "G17ProofDependencySetV1",
    "G17ProofDependencyV1",
    "G17ProofKindV1",
    "G17R10ConstraintCoordinateV1",
    "G17R10ConstraintV1",
    "G17R10ProsodyFeatureRelayV1",
    "G17RealizationGaugeV1",
    "G17ReceiverExecutionResultV1",
    "G17RecursionCoordinateV1",
    "G17RecursionNamespaceV1",
    "G17ReopenedEvidencePacketV1",
    "G17ResearchAuthorityEvidenceV1",
    "G17RuntimeDependencyEdgeV1",
    "G17RuntimeDependencyFileV1",
    "G17RuntimeDependencyMechanismV1",
    "G17RuntimeFileScopeV1",
    "G17ScientificRoleV1",
    "G17ScorerExecutionResultV1",
    "G17SemanticStreamRoleV1",
    "G17SemanticTopologyV1",
    "G17SparseObligationOwnerV1",
    "G17TerminalCompilerPassV1",
    "G17TerminalCompilerScheduleV1",
    "G17TerminalEnvelopeLogicalV1",
    "G17VMExecutionReceiptV1",
    "G17VMOpcodeV1",
    "G17VMOperandV1",
    "G17WholeObjectStateReceiptV1",
    "G17WholeObjectStateV1",
    "arbitrate_g17_whole_object_states",
    "build_g17_whole_object_state_receipt",
    "execute_g17_reconstruction_vm",
    "parse_g17_whole_object_state_receipt",
    "require_g17_auth_eval_public_entrypoint_closure",
    "require_g17_exact_contest_authority_adapter",
]
