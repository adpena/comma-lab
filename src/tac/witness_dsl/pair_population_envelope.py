# SPDX-License-Identifier: MIT
"""Typed pair coordinates and compact-program joins for the C0B codec seam.

This module is deliberately a structural compile boundary, not a codec or a
score claim.  It closes five identity gaps while keeping exact scorer, ground
truth, teacher, obligation-IR, and oracle artifacts encoder-only.  Counted
own-lineage dense planes or camera preimages are structurally legal; their
typed provenance and exact bytes remain visible instead of being rejected by
name or hidden behind an originality claim.

The counted preimage object is an exact compact *generator program*.  Its bytes
are retained and freshly reopened by a caller-supplied real receiver.  The
receiver must yield generated Y0/Y1 values so this module can derive their
identities and join them to an encoder-only ``ExplicitV10PreimageCompileResult``.
A hash-and-size reference to the dense V10 packet is never accepted as a
substitute for those compact program bytes.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import CodeType
from typing import Any, Final

import numpy as np

from tac.witness_dsl.coupled_witness_state import (
    CoupledWitnessState,
    CoupledWitnessStateError,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
)
from tac.witness_dsl.evaluator_obligation_ir import (
    EvaluatorObligationIR,
    ExplicitV10PreimageCompileResult,
)
from tac.witness_dsl.factorized_v9_predictor import (
    FactorizedV9SemanticReceiver,
    receive_factorized_v9_predictor,
)
from tac.witness_dsl.generative_taskspace_correction import (
    PACKET_SCHEMA as GENERATIVE_CORRECTION_PACKET_SCHEMA,
)
from tac.witness_dsl.generative_taskspace_correction import (
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
)
from tac.witness_dsl.progressive_geometry_residual import (
    ProgressiveGeometryResidualError,
    decode_progressive_geometry_residual,
)
from tac.witness_dsl.progressive_v9_entropy_measurement import (
    apply_progressive_v9_entropy_measurement,
)

PAIR_DOMAIN_INDEX_SCHEMA: Final = "tac.pair_domain_index.v1"
PAIR_COORDINATE_ROW_SCHEMA: Final = "tac.pair_coordinate_row.v1"
PAIR_POPULATION_SCHEMA: Final = "tac.pair_population.v1"
PAIR_POPULATION_ENVELOPE_SCHEMA: Final = "tac.pair_population.envelope.v1"
PAIR_REFERENCE_SCHEMA: Final = "tac.encoder_only_pair_reference.v1"
IR_COVERAGE_SCHEMA: Final = "tac.ir_candidate_semantic_coverage.v1"
IR_COVERAGE_ENVELOPE_SCHEMA: Final = "tac.ir_candidate_semantic_coverage.envelope.v1"
COMPACT_PROGRAM_SCHEMA: Final = "tac.compact_obligation_generator_program.v1"
COMPACT_PROGRAM_ENVELOPE_SCHEMA: Final = "tac.compact_obligation_generator_program.envelope.v1"
POSE_OWNERSHIP_SCHEMA: Final = "tac.exclusive_pose_ownership.v1"
JOIN_ENVELOPE_SCHEMA: Final = "tac.pair_population_compact_program_join.v1"
JOIN_ENVELOPE_WIRE_SCHEMA: Final = "tac.pair_population_compact_program_join.envelope.v1"
RECEIVER_BINDING_SCOPE: Final = "direct_python_function_source_and_code_nontransitive.v1"

_PAIR_REFERENCE_REOPEN_PROOF: Final = object()
_IR_COVERAGE_DERIVATION_PROOF: Final = object()
_COMPACT_PROGRAM_REOPEN_PROOF: Final = object()


class PairPopulationEnvelopeError(ValueError):
    """Malformed coordinate, orphaned foreign key, or forbidden payload."""


class PairDomain(StrEnum):
    """Local coordinate domains joined by one global source-pair identity."""

    V9 = "v9"
    PBR = "pbr"
    IR = "ir"
    V10 = "v10"


PAIR_DOMAIN_ORDER: Final = tuple(PairDomain)


class IRCoveragePolicy(StrEnum):
    COMPLETE = "complete"
    SPARSE_OWNED = "sparse_owned"


class SparseDebtOwner(StrEnum):
    FRAME1_PREIMAGE = "frame1_preimage"
    TERMINAL_QUOTIENT = "terminal_quotient"


class CompactProgramRole(StrEnum):
    Y0_Y1_OBLIGATION_GENERATOR = "compact_y0_y1_obligation_generator"
    FRAME0_POSE_RESIDUAL = "frame0_pose_residual"


class CountedSectionRole(StrEnum):
    GENERATIVE_CORRECTION = "generative_correction"
    FRAME1_PREIMAGE = "frame1_preimage"
    FRAME0_FROM_EXACT_Y1 = "frame0_from_exact_y1"
    FRAME0_POSE_RESIDUAL = "frame0_pose_residual"
    TERMINAL_QUOTIENT = "terminal_quotient"


class Frame0PoseMode(StrEnum):
    RESIDUAL_BEYOND_V9_POSE6 = "residual_beyond_v9_pose6_conditioned_on_exact_frame1"


class CountedPayloadLineage(StrEnum):
    """Custody lineage, deliberately distinct from a novelty claim."""

    ORIGINAL_OWN = "original_own_lineage"
    BORROWED_DISCLOSED = "borrowed_disclosed_lineage"


class CountedArtifactClass(StrEnum):
    """Typed payload class; exact forbidden encoder/scorer/GT classes fail closed."""

    GENERATOR_PARAMETERS = "generator_parameters"
    FRAME_PREIMAGE_PARAMETERS = "frame_preimage_parameters"
    COUPLED_PREIMAGE_PARAMETERS = "coupled_preimage_parameters"
    DENSE_REALIZED_Y = "dense_realized_y"
    CAMERA_PREIMAGE = "camera_preimage"
    POSE_RESIDUAL_PARAMETERS = "pose_residual_parameters"
    TERMINAL_QUOTIENT = "terminal_quotient"
    SCORER_MODEL = "scorer_model"
    GROUND_TRUTH = "ground_truth"
    GT_ARGMAX_TABLE = "gt_argmax_table"
    ENCODER_ONLY_TEACHER = "encoder_only_teacher"
    ENCODER_ONLY_OBLIGATION_IR = "encoder_only_obligation_ir"
    ENCODER_ONLY_ORACLE_EVIDENCE = "encoder_only_oracle_evidence"
    ENCODER_ONLY_EXPLICIT_PREIMAGE = "encoder_only_explicit_preimage"


_FORBIDDEN_COUNTED_ARTIFACT_CLASSES: Final = frozenset(
    {
        CountedArtifactClass.SCORER_MODEL,
        CountedArtifactClass.GROUND_TRUTH,
        CountedArtifactClass.GT_ARGMAX_TABLE,
        CountedArtifactClass.ENCODER_ONLY_TEACHER,
        CountedArtifactClass.ENCODER_ONLY_OBLIGATION_IR,
        CountedArtifactClass.ENCODER_ONLY_ORACLE_EVIDENCE,
        CountedArtifactClass.ENCODER_ONLY_EXPLICIT_PREIMAGE,
    }
)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise PairPopulationEnvelopeError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise PairPopulationEnvelopeError(
            f"{label} fields differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PairPopulationEnvelopeError(f"{label} must be a nonnegative exact integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise PairPopulationEnvelopeError(f"{label} must be positive")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PairPopulationEnvelopeError(f"{label} must be nonempty trimmed text")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairPopulationEnvelopeError(f"{label} must be lowercase SHA-256 hex")
    return value


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_tuple_of_ids(value: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(value)
    if not result:
        raise PairPopulationEnvelopeError(f"{label} must be nonempty")
    if any(type(item) is not int or item < 0 for item in result):
        raise PairPopulationEnvelopeError(f"{label} must contain nonnegative exact integers")
    if len(set(result)) != len(result):
        raise PairPopulationEnvelopeError(f"{label} must not contain duplicate source-pair IDs")
    return result


def _require_canonical_frozen_pair_order(
    ir_ids: tuple[int, ...],
    v10_ids: tuple[int, ...],
    *,
    pair_count: int,
) -> None:
    expected = tuple(range(pair_count))
    if ir_ids != expected or v10_ids != expected:
        raise PairPopulationEnvelopeError(
            "IR/V10 local pair indexes must equal the frozen canonical contiguous pair order"
        )


def _decode_envelope(payload: bytes, *, schema: str, label: str) -> Mapping[str, Any]:
    try:
        value = decode_canonical_json(payload)
    except CoupledWitnessStateError as exc:
        raise PairPopulationEnvelopeError(f"{label} is not canonical JSON") from exc
    _exact_keys(value, {"schema", "body", "body_sha256"}, label)
    if value["schema"] != schema:
        raise PairPopulationEnvelopeError(f"{label} schema differs")
    if canonical_sha256(value["body"]) != value["body_sha256"]:
        raise PairPopulationEnvelopeError(f"{label} body hash differs")
    return value["body"]


def _array_identity_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return canonical_sha256(
        {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "byte_length": int(contiguous.nbytes),
            "bytes_sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        }
    )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class PairDomainIndex:
    """Exact local-index -> global-source mapping for one coordinate domain."""

    domain: PairDomain
    local_to_source_pair_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, PairDomain):
            raise PairPopulationEnvelopeError("pair domain is invalid")
        if type(self.local_to_source_pair_ids) is not tuple:
            raise PairPopulationEnvelopeError("local_to_source_pair_ids must be an exact tuple")
        _strict_tuple_of_ids(self.local_to_source_pair_ids, f"{self.domain.value} pair index")

    @classmethod
    def contiguous(cls, domain: PairDomain, *, source_start: int, pair_count: int) -> PairDomainIndex:
        start = _nonnegative_int(source_start, "source_start")
        count = _positive_int(pair_count, "pair_count")
        return cls(domain, tuple(range(start, start + count)))

    def local_pair_id(self, source_pair_id: int) -> int:
        source = _nonnegative_int(source_pair_id, "source_pair_id")
        try:
            return self.local_to_source_pair_ids.index(source)
        except ValueError as exc:
            raise PairPopulationEnvelopeError(
                f"source pair {source} is outside the {self.domain.value} local index"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_DOMAIN_INDEX_SCHEMA,
            "domain": self.domain.value,
            "local_to_source_pair_ids": list(self.local_to_source_pair_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairDomainIndex:
        _exact_keys(value, {"schema", "domain", "local_to_source_pair_ids"}, "pair domain index")
        if value["schema"] != PAIR_DOMAIN_INDEX_SCHEMA or not isinstance(value["local_to_source_pair_ids"], list):
            raise PairPopulationEnvelopeError("pair domain index wire type differs")
        try:
            domain = PairDomain(value["domain"])
        except (TypeError, ValueError) as exc:
            raise PairPopulationEnvelopeError("pair domain index domain differs") from exc
        return cls(domain, tuple(value["local_to_source_pair_ids"]))


@dataclass(frozen=True, order=True, slots=True)
class PairCoordinateRow:
    source_pair_id: int
    v9_local_pair_id: int
    pbr_local_pair_id: int
    ir_local_pair_id: int
    v10_local_pair_id: int

    def __post_init__(self) -> None:
        for label, value in (
            ("source_pair_id", self.source_pair_id),
            ("v9_local_pair_id", self.v9_local_pair_id),
            ("pbr_local_pair_id", self.pbr_local_pair_id),
            ("ir_local_pair_id", self.ir_local_pair_id),
            ("v10_local_pair_id", self.v10_local_pair_id),
        ):
            _nonnegative_int(value, label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_COORDINATE_ROW_SCHEMA,
            "source_pair_id": self.source_pair_id,
            "v9_local_pair_id": self.v9_local_pair_id,
            "pbr_local_pair_id": self.pbr_local_pair_id,
            "ir_local_pair_id": self.ir_local_pair_id,
            "v10_local_pair_id": self.v10_local_pair_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairCoordinateRow:
        expected = {
            "schema",
            "source_pair_id",
            "v9_local_pair_id",
            "pbr_local_pair_id",
            "ir_local_pair_id",
            "v10_local_pair_id",
        }
        _exact_keys(value, expected, "pair coordinate row")
        if value["schema"] != PAIR_COORDINATE_ROW_SCHEMA:
            raise PairPopulationEnvelopeError("pair coordinate row schema differs")
        return cls(
            source_pair_id=value["source_pair_id"],
            v9_local_pair_id=value["v9_local_pair_id"],
            pbr_local_pair_id=value["pbr_local_pair_id"],
            ir_local_pair_id=value["ir_local_pair_id"],
            v10_local_pair_id=value["v10_local_pair_id"],
        )


@dataclass(frozen=True, slots=True)
class PairPopulation:
    """One hashed population joining global IDs to all four local domains."""

    source_pair_ids: tuple[int, ...]
    domain_indexes: tuple[PairDomainIndex, ...]

    def __post_init__(self) -> None:
        if type(self.source_pair_ids) is not tuple or type(self.domain_indexes) is not tuple:
            raise PairPopulationEnvelopeError("pair population fields must be exact tuples")
        _strict_tuple_of_ids(self.source_pair_ids, "pair population")
        if tuple(index.domain for index in self.domain_indexes) != PAIR_DOMAIN_ORDER:
            raise PairPopulationEnvelopeError("pair domain indexes must contain V9/PBR/IR/V10 in canonical order")
        for source_pair_id in self.source_pair_ids:
            for index in self.domain_indexes:
                index.local_pair_id(source_pair_id)

    @classmethod
    def derive(
        cls,
        *,
        source_pair_ids: Sequence[int],
        v9_local_to_source_pair_ids: Sequence[int],
        pbr_local_to_source_pair_ids: Sequence[int],
        ir_local_to_source_pair_ids: Sequence[int],
        v10_local_to_source_pair_ids: Sequence[int],
    ) -> PairPopulation:
        return cls(
            source_pair_ids=tuple(source_pair_ids),
            domain_indexes=(
                PairDomainIndex(PairDomain.V9, tuple(v9_local_to_source_pair_ids)),
                PairDomainIndex(PairDomain.PBR, tuple(pbr_local_to_source_pair_ids)),
                PairDomainIndex(PairDomain.IR, tuple(ir_local_to_source_pair_ids)),
                PairDomainIndex(PairDomain.V10, tuple(v10_local_to_source_pair_ids)),
            ),
        )

    def domain_index(self, domain: PairDomain) -> PairDomainIndex:
        return self.domain_indexes[PAIR_DOMAIN_ORDER.index(domain)]

    @property
    def rows(self) -> tuple[PairCoordinateRow, ...]:
        indexes = {index.domain: index for index in self.domain_indexes}
        return tuple(
            PairCoordinateRow(
                source_pair_id=source,
                v9_local_pair_id=indexes[PairDomain.V9].local_pair_id(source),
                pbr_local_pair_id=indexes[PairDomain.PBR].local_pair_id(source),
                ir_local_pair_id=indexes[PairDomain.IR].local_pair_id(source),
                v10_local_pair_id=indexes[PairDomain.V10].local_pair_id(source),
            )
            for source in self.source_pair_ids
        )

    @property
    def population_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_POPULATION_SCHEMA,
            "source_pair_ids": list(self.source_pair_ids),
            "domain_indexes": [index.as_dict() for index in self.domain_indexes],
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {"schema": PAIR_POPULATION_ENVELOPE_SCHEMA, "body": body, "body_sha256": canonical_sha256(body)}
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairPopulation:
        _exact_keys(value, {"schema", "source_pair_ids", "domain_indexes", "rows"}, "pair population")
        if value["schema"] != PAIR_POPULATION_SCHEMA:
            raise PairPopulationEnvelopeError("pair population schema differs")
        if not all(isinstance(value[key], list) for key in ("source_pair_ids", "domain_indexes", "rows")):
            raise PairPopulationEnvelopeError("pair population arrays differ")
        result = cls(
            source_pair_ids=tuple(value["source_pair_ids"]),
            domain_indexes=tuple(PairDomainIndex.from_dict(item) for item in value["domain_indexes"]),
        )
        serialized_rows = tuple(PairCoordinateRow.from_dict(item) for item in value["rows"])
        if serialized_rows != result.rows:
            raise PairPopulationEnvelopeError("serialized pair coordinates are not derived from domain indexes")
        return result

    @classmethod
    def from_bytes(cls, payload: bytes) -> PairPopulation:
        body = _decode_envelope(payload, schema=PAIR_POPULATION_ENVELOPE_SCHEMA, label="pair population envelope")
        result = cls.from_dict(body)
        if result.population_sha256 != canonical_sha256(body):
            raise PairPopulationEnvelopeError("pair population reconstructed identity differs")
        return result


@dataclass(frozen=True, slots=True)
class EncoderOnlyPairReference:
    """Pair IDs derived by reopening an exact encoder-only artifact."""

    role: str
    artifact_schema: str
    artifact_sha256: str
    artifact_bytes: int
    source_pair_ids: tuple[int, ...]
    semantic_sha256: str | None = None
    candidate_payload_allowed: bool = False
    _reopen_proof: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reopen_proof is not _PAIR_REFERENCE_REOPEN_PROOF:
            raise PairPopulationEnvelopeError("pair reference must be derived by reopening exact artifact bytes")
        _text(self.role, "pair-reference role")
        _text(self.artifact_schema, "pair-reference artifact_schema")
        _sha256(self.artifact_sha256, "pair-reference artifact_sha256")
        _positive_int(self.artifact_bytes, "pair-reference artifact_bytes")
        if type(self.source_pair_ids) is not tuple:
            raise PairPopulationEnvelopeError("pair-reference source IDs must be an exact tuple")
        _strict_tuple_of_ids(self.source_pair_ids, "pair-reference source IDs")
        if self.semantic_sha256 is not None:
            _sha256(self.semantic_sha256, "pair-reference semantic_sha256")
        if type(self.candidate_payload_allowed) is not bool or self.candidate_payload_allowed:
            raise PairPopulationEnvelopeError("encoder-only pair references are forbidden candidate payloads")

    @property
    def pair_ids_sha256(self) -> str:
        return canonical_sha256(list(self.source_pair_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_REFERENCE_SCHEMA,
            "role": self.role,
            "artifact_schema": self.artifact_schema,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "source_pair_ids": list(self.source_pair_ids),
            "pair_ids_sha256": self.pair_ids_sha256,
            "semantic_sha256": self.semantic_sha256,
            "candidate_payload_allowed": self.candidate_payload_allowed,
        }


def reopen_pbr2_pair_reference(payload: bytes) -> EncoderOnlyPairReference:
    """Derive the target window from a strictly parsed PBR2 teacher packet."""

    if not isinstance(payload, bytes) or not payload:
        raise PairPopulationEnvelopeError("PBR2 reference requires exact nonempty bytes")
    try:
        packet = decode_progressive_geometry_residual(payload)
    except ProgressiveGeometryResidualError as exc:
        raise PairPopulationEnvelopeError("PBR2 pair reference did not reopen") from exc
    header = packet.header
    if (
        header.get("schema") != "tac.progressive_geometry_residual.v3"
        or header.get("pbr2_is_target_derived") is not True
        or header.get("candidate_archive_admissible") is not False
        or header.get("score_claim") is not False
    ):
        raise PairPopulationEnvelopeError("PBR2 pair reference lost its encoder-only teacher boundary")
    start = _nonnegative_int(header.get("source_pair_start"), "PBR2 source_pair_start")
    stop = _positive_int(header.get("source_pair_stop_exclusive"), "PBR2 source_pair_stop_exclusive")
    if stop <= start:
        raise PairPopulationEnvelopeError("PBR2 source-pair window is empty")
    return EncoderOnlyPairReference(
        role="pbr2_target_window",
        artifact_schema=header["schema"],
        artifact_sha256=_payload_sha256(payload),
        artifact_bytes=len(payload),
        source_pair_ids=tuple(range(start, stop)),
        semantic_sha256=_sha256(header.get("target_semantic_sha256"), "PBR2 target semantic SHA-256"),
        _reopen_proof=_PAIR_REFERENCE_REOPEN_PROOF,
    )


def reopen_typed_pair_reference(
    payload: bytes,
    *,
    role: str,
    reopener: Callable[[bytes], object],
) -> EncoderOnlyPairReference:
    """Derive exact pair IDs from a reopened typed config, never an ID string."""

    if not isinstance(payload, bytes) or not payload or not callable(reopener):
        raise PairPopulationEnvelopeError("typed pair reference requires exact bytes and a callable reopener")
    try:
        payload_body = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairPopulationEnvelopeError("typed pair artifact bytes are not an exact JSON object") from exc
    if not isinstance(payload_body, Mapping):
        raise PairPopulationEnvelopeError("typed pair artifact bytes are not an exact JSON object")
    reopened = reopener(payload)
    pair_ids = reopened.get("pair_ids") if isinstance(reopened, Mapping) else getattr(reopened, "pair_ids", None)
    if not isinstance(pair_ids, (tuple, list)):
        raise PairPopulationEnvelopeError("reopened typed pair artifact exposes no exact pair_ids")
    if hasattr(reopened, "model_dump"):
        body = reopened.model_dump(mode="json", by_alias=True)  # type: ignore[attr-defined]
    elif isinstance(reopened, Mapping):
        body = dict(reopened)
    else:
        raise PairPopulationEnvelopeError("reopened typed pair artifact exposes no canonical body")
    if body != payload_body:
        raise PairPopulationEnvelopeError("reopened typed pair object differs from the exact artifact body")
    if body.get("pair_ids") != list(pair_ids):
        raise PairPopulationEnvelopeError("reopened typed pair IDs differ from its canonical body")
    schema = body.get("schema")
    return EncoderOnlyPairReference(
        role=role,
        artifact_schema=_text(schema, "typed pair artifact schema"),
        artifact_sha256=_payload_sha256(payload),
        artifact_bytes=len(payload),
        source_pair_ids=tuple(pair_ids),
        semantic_sha256=canonical_sha256(body),
        _reopen_proof=_PAIR_REFERENCE_REOPEN_PROOF,
    )


@dataclass(frozen=True, slots=True)
class ReopenedObjectJoin:
    """Actual typed objects whose identities are derived, never caller strings."""

    state: CoupledWitnessState
    predictor: FactorizedV9SemanticReceiver
    obligation_ir: EvaluatorObligationIR
    explicit_preimage: ExplicitV10PreimageCompileResult
    pair_population: PairPopulation
    _predictor_binding: Mapping[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, CoupledWitnessState):
            raise PairPopulationEnvelopeError("state join requires a reopened CoupledWitnessState")
        if not isinstance(self.predictor, FactorizedV9SemanticReceiver):
            raise PairPopulationEnvelopeError("predictor join requires a reopened V9 receiver")
        if not isinstance(self.obligation_ir, EvaluatorObligationIR):
            raise PairPopulationEnvelopeError("IR join requires a reopened EvaluatorObligationIR")
        if not isinstance(self.explicit_preimage, ExplicitV10PreimageCompileResult):
            raise PairPopulationEnvelopeError("preimage join requires a reopened explicit result")
        binding = self.predictor.semantic_binding()
        object.__setattr__(self, "_predictor_binding", binding)
        ir = self.obligation_ir
        result = self.explicit_preimage
        if ir.frozen_space != self.state.frozen_space:
            raise PairPopulationEnvelopeError("IR frozen space differs from the reopened coupled state")
        if ir.coupled_state_sha256 != self.state.state_sha256:
            raise PairPopulationEnvelopeError("IR coupled-state identity was not derived from the reopened state")
        if ir.predictor_state_sha256 != binding["predictor_program_sha256"]:
            raise PairPopulationEnvelopeError("IR predictor identity differs from the reopened program")
        if ir.predictor_semantic_sha256 != binding["predictor_semantic_sha256"]:
            raise PairPopulationEnvelopeError("IR predictor semantics differ from fresh V9 decode")
        expected_result_keys = {
            "evaluator_obligation_ir_sha256": ir.ir_sha256,
            "frozen_space_identity_sha256": self.state.frozen_space.identity_sha256,
            "coupled_state_sha256": self.state.state_sha256,
            "predictor_state_sha256": binding["predictor_program_sha256"],
            "predictor_semantic_sha256": binding["predictor_semantic_sha256"],
        }
        for key, expected in expected_result_keys.items():
            if getattr(result, key) != expected:
                raise PairPopulationEnvelopeError(f"explicit preimage {key} differs from reopened objects")
        v9_ids = self.pair_population.domain_index(PairDomain.V9).local_to_source_pair_ids
        ir_ids = self.pair_population.domain_index(PairDomain.IR).local_to_source_pair_ids
        v10_ids = self.pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids
        if v9_ids != self.predictor.source_pair_ids:
            raise PairPopulationEnvelopeError("V9 local/source mapping differs from the reopened predictor window")
        if result.pair_count != self.state.frozen_space.pair_count:
            raise PairPopulationEnvelopeError("explicit preimage pair count differs from frozen space")
        _require_canonical_frozen_pair_order(
            ir_ids,
            v10_ids,
            pair_count=self.state.frozen_space.pair_count,
        )
        result_pair_ids = tuple(receipt.pair_id for receipt in result.pair_receipts)
        if result_pair_ids != ir_ids or result_pair_ids != v10_ids:
            raise PairPopulationEnvelopeError(
                "explicit V10 receipts, IR indexes, and V10 indexes differ from canonical frozen order"
            )
        if ir_ids != v10_ids:
            raise PairPopulationEnvelopeError("IR/V10 local pair indexes do not describe the same result population")

    @classmethod
    def reopen(
        cls,
        *,
        state_bytes: bytes,
        predictor_program_bytes: bytes,
        obligation_ir_bytes: bytes,
        explicit_preimage_result_bytes: bytes,
        pair_population: PairPopulation,
        repository_root: Path | None = None,
    ) -> ReopenedObjectJoin:
        """Fresh-open all identity-bearing inputs before deriving their joins."""

        try:
            state = CoupledWitnessState.from_bytes(state_bytes)
            ir = EvaluatorObligationIR.from_bytes(obligation_ir_bytes)
            result = ExplicitV10PreimageCompileResult.from_bytes(explicit_preimage_result_bytes)
        except (CoupledWitnessStateError, ValueError) as exc:
            raise PairPopulationEnvelopeError("state/IR/preimage input did not reopen") from exc
        predictor = receive_factorized_v9_predictor(predictor_program_bytes, repository_root=repository_root)
        return cls(state, predictor, ir, result, pair_population)

    @property
    def predictor_pose6_sha256(self) -> str:
        return _sha256(self._predictor_binding["temporal_pose6_sha256"], "predictor Pose6 SHA-256")

    def generative_predictor_state(self) -> PredictorSemanticStateV1:
        """Reopen V9 again and derive the exact public state consumed by G."""

        fresh = receive_factorized_v9_predictor(self.predictor.program)
        labels = fresh.decode_all_semantics()
        pose6_codes = np.ascontiguousarray(fresh.receiver.pose6_codes)
        raw_pose6_sha256 = hashlib.sha256(memoryview(pose6_codes).cast("B")).hexdigest()
        result = PredictorSemanticStateV1(
            predictor_program_sha256=fresh.program_sha256,
            predictor_renderer_sha256=fresh.source_manifest_sha256,
            source_pair_ids=fresh.source_pair_ids,
            labels=labels,
            pose6_codes=pose6_codes,
        )
        if (
            result.predictor_program_sha256 != self._predictor_binding["predictor_program_sha256"]
            or result.predictor_renderer_sha256 != self._predictor_binding["predictor_renderer_sha256"]
            or result.labels_sha256 != self._predictor_binding["predictor_semantic_sha256"]
            or raw_pose6_sha256 != self.predictor_pose6_sha256
        ):
            raise PairPopulationEnvelopeError("fresh generative predictor state differs from reopened V9 identities")
        return result

    def identity_dict(self) -> dict[str, Any]:
        return {
            "coupled_state_sha256": self.state.state_sha256,
            "frozen_space_identity_sha256": self.state.frozen_space.identity_sha256,
            "predictor_program_sha256": self._predictor_binding["predictor_program_sha256"],
            "predictor_renderer_sha256": self._predictor_binding["predictor_renderer_sha256"],
            "predictor_semantic_sha256": self._predictor_binding["predictor_semantic_sha256"],
            "predictor_pose6_sha256": self.predictor_pose6_sha256,
            "obligation_ir_sha256": self.obligation_ir.ir_sha256,
            "explicit_preimage_result_sha256": self.explicit_preimage.result_sha256,
            "encoder_only_dense_y0_identity_sha256": self.explicit_preimage.scorer_y0_identity_sha256,
            "encoder_only_dense_y1_identity_sha256": self.explicit_preimage.scorer_y1_identity_sha256,
            "encoder_only_dense_v10_packet_sha256": self.explicit_preimage.receiver_packet_sha256,
            "encoder_only_dense_v10_packet_bytes": self.explicit_preimage.receiver_packet_bytes,
            "pair_population_sha256": self.pair_population.population_sha256,
        }


@dataclass(frozen=True, slots=True)
class DecodedCandidateSemantics:
    """Fresh candidate decode returned by a real G/P receiver adapter."""

    source_pair_ids: tuple[int, ...]
    labels: np.ndarray
    decoder_contract_id: str

    def __post_init__(self) -> None:
        if type(self.source_pair_ids) is not tuple:
            raise PairPopulationEnvelopeError("decoded candidate source IDs must be an exact tuple")
        _strict_tuple_of_ids(self.source_pair_ids, "decoded candidate source IDs")
        _text(self.decoder_contract_id, "candidate semantic decoder contract")
        labels = np.asarray(self.labels)
        if labels.dtype != np.uint8 or labels.ndim != 3 or labels.shape[0] != len(self.source_pair_ids):
            raise PairPopulationEnvelopeError("decoded candidate semantics must be uint8 [pairs,height,width]")
        if labels.size == 0 or int(labels.max()) > 4:
            raise PairPopulationEnvelopeError("decoded candidate semantics escape the five-class universe")
        object.__setattr__(self, "labels", _immutable_array(labels))


@dataclass(frozen=True, order=True, slots=True)
class SparseObligationOwnership:
    source_pair_id: int
    row: int
    col: int
    owner: SparseDebtOwner
    owner_section_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_pair_id, "sparse owner source_pair_id")
        _nonnegative_int(self.row, "sparse owner row")
        _nonnegative_int(self.col, "sparse owner col")
        if not isinstance(self.owner, SparseDebtOwner):
            raise PairPopulationEnvelopeError("sparse obligation owner is invalid")
        _sha256(self.owner_section_sha256, "sparse obligation owner section SHA-256")

    def as_dict(self, pair_population: PairPopulation) -> dict[str, Any]:
        return {
            "source_pair_id": self.source_pair_id,
            "ir_local_pair_id": pair_population.domain_index(PairDomain.IR).local_pair_id(self.source_pair_id),
            "row": self.row,
            "col": self.col,
            "owner": self.owner.value,
            "owner_section_sha256": self.owner_section_sha256,
        }


@dataclass(frozen=True, slots=True)
class IRCoverageReceipt:
    """Encoder-only proof that every scoped IR row is matched or owned."""

    pair_population_sha256: str
    obligation_ir_sha256: str
    predictor_program_sha256: str
    pbr2_packet_sha256: str
    pbr2_target_semantic_sha256: str
    candidate_program_sha256: str
    candidate_program_bytes: int
    compact_program_sha256: str
    candidate_decoder_contract_id: str
    candidate_semantic_identity_sha256: str
    policy: IRCoveragePolicy
    obligation_count: int
    matched_count: int
    sparse_ownership: tuple[SparseObligationOwnership, ...]
    encoder_only: bool = True
    teacher_truth_serialized: bool = False
    candidate_payload_allowed: bool = False
    _derivation_proof: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._derivation_proof is not _IR_COVERAGE_DERIVATION_PROOF:
            raise PairPopulationEnvelopeError("IR coverage must be derived from freshly decoded candidate semantics")
        for label, value in (
            ("pair population", self.pair_population_sha256),
            ("obligation IR", self.obligation_ir_sha256),
            ("predictor program", self.predictor_program_sha256),
            ("PBR2 packet", self.pbr2_packet_sha256),
            ("PBR2 target semantic", self.pbr2_target_semantic_sha256),
            ("candidate program", self.candidate_program_sha256),
            ("compact program", self.compact_program_sha256),
            ("candidate semantic", self.candidate_semantic_identity_sha256),
        ):
            _sha256(value, f"{label} SHA-256")
        _positive_int(self.candidate_program_bytes, "candidate_program_bytes")
        _text(self.candidate_decoder_contract_id, "candidate_decoder_contract_id")
        obligations = _positive_int(self.obligation_count, "obligation_count")
        matched = _nonnegative_int(self.matched_count, "matched_count")
        if not isinstance(self.policy, IRCoveragePolicy):
            raise PairPopulationEnvelopeError("IR coverage policy is invalid")
        if type(self.sparse_ownership) is not tuple or any(
            not isinstance(item, SparseObligationOwnership) for item in self.sparse_ownership
        ):
            raise PairPopulationEnvelopeError("IR sparse ownership must be an exact typed tuple")
        if matched + len(self.sparse_ownership) != obligations:
            raise PairPopulationEnvelopeError("IR coverage counts leave an orphaned obligation")
        if self.policy is IRCoveragePolicy.COMPLETE and self.sparse_ownership:
            raise PairPopulationEnvelopeError("complete IR coverage cannot carry sparse owners")
        if self.policy is IRCoveragePolicy.SPARSE_OWNED and not self.sparse_ownership:
            raise PairPopulationEnvelopeError("sparse-owned IR coverage must own at least one unmatched row")
        if tuple(sorted(self.sparse_ownership)) != self.sparse_ownership:
            raise PairPopulationEnvelopeError("sparse ownership must use canonical source/row/col/owner order")
        if len(set(self.sparse_ownership)) != len(self.sparse_ownership):
            raise PairPopulationEnvelopeError("sparse ownership contains duplicate rows")
        if (
            type(self.encoder_only) is not bool
            or not self.encoder_only
            or type(self.teacher_truth_serialized) is not bool
            or self.teacher_truth_serialized
            or type(self.candidate_payload_allowed) is not bool
            or self.candidate_payload_allowed
        ):
            raise PairPopulationEnvelopeError("IR coverage is encoder-only and cannot serialize teacher truth")

    def as_dict(self, pair_population: PairPopulation) -> dict[str, Any]:
        return {
            "schema": IR_COVERAGE_SCHEMA,
            "pair_population_sha256": self.pair_population_sha256,
            "obligation_ir_sha256": self.obligation_ir_sha256,
            "predictor_program_sha256": self.predictor_program_sha256,
            "pbr2_packet_sha256": self.pbr2_packet_sha256,
            "pbr2_target_semantic_sha256": self.pbr2_target_semantic_sha256,
            "candidate_program_sha256": self.candidate_program_sha256,
            "candidate_program_bytes": self.candidate_program_bytes,
            "compact_program_sha256": self.compact_program_sha256,
            "candidate_decoder_contract_id": self.candidate_decoder_contract_id,
            "candidate_semantic_identity_sha256": self.candidate_semantic_identity_sha256,
            "policy": self.policy.value,
            "obligation_count": self.obligation_count,
            "matched_count": self.matched_count,
            "sparse_ownership": [item.as_dict(pair_population) for item in self.sparse_ownership],
            "encoder_only": self.encoder_only,
            "teacher_truth_serialized": self.teacher_truth_serialized,
            "candidate_payload_allowed": self.candidate_payload_allowed,
        }


def derive_ir_coverage(
    join: ReopenedObjectJoin,
    *,
    pbr2_packet: bytes,
    compact_program: CompactObligationGeneratorProgram,
    candidate_program: bytes,
    policy: IRCoveragePolicy,
    sparse_ownership: Sequence[SparseObligationOwnership] = (),
) -> IRCoverageReceipt:
    """Reopen teacher/candidate bytes and derive complete-or-owned coverage."""

    if not isinstance(join, ReopenedObjectJoin):
        raise PairPopulationEnvelopeError("IR coverage requires a reopened object join")
    if not isinstance(compact_program, CompactObligationGeneratorProgram):
        raise PairPopulationEnvelopeError("IR coverage requires a freshly reopened compact counted program")
    if not isinstance(candidate_program, bytes) or not candidate_program:
        raise PairPopulationEnvelopeError("candidate semantic program must be exact nonempty bytes")
    if candidate_program.startswith((b"PBR1", b"PBR2")):
        raise PairPopulationEnvelopeError("exact PBR teacher bytes are forbidden as a candidate semantic program")
    if not isinstance(policy, IRCoveragePolicy):
        raise PairPopulationEnvelopeError("IR coverage requires a typed policy")
    pbr_reference = reopen_pbr2_pair_reference(pbr2_packet)
    population = join.pair_population
    pbr_ids = population.domain_index(PairDomain.PBR).local_to_source_pair_ids
    if pbr_reference.source_pair_ids != pbr_ids or population.source_pair_ids != pbr_ids:
        raise PairPopulationEnvelopeError("PBR2 target window differs from the active pair population")
    if population.domain_index(PairDomain.V9).local_to_source_pair_ids != pbr_ids:
        raise PairPopulationEnvelopeError("PBR2 teacher and V9 predictor do not share an exact source order")
    try:
        target = apply_progressive_v9_entropy_measurement(join.predictor.program, pbr2_packet)
    except (ValueError, ProgressiveGeometryResidualError) as exc:
        raise PairPopulationEnvelopeError("PBR2 target semantics did not reopen against the exact predictor") from exc
    generative_section = compact_program.section(CountedSectionRole.GENERATIVE_CORRECTION)
    exact_section_bytes = compact_program.program_bytes[generative_section.offset : generative_section.stop]
    if exact_section_bytes != candidate_program:
        raise PairPopulationEnvelopeError("candidate G bytes differ from the counted compact-program section")
    predictor_state = join.generative_predictor_state()
    try:
        decoded = apply_generative_taskspace_correction(candidate_program, predictor_state=predictor_state)
    except GenerativeTaskspaceCorrectionError as exc:
        raise PairPopulationEnvelopeError("candidate G did not reopen through its typed receiver") from exc
    if predictor_state.source_pair_ids != population.source_pair_ids:
        raise PairPopulationEnvelopeError("candidate semantic source order differs from PairPopulation")
    labels = decoded.labels
    expected_shape = (
        len(population.source_pair_ids),
        join.state.frozen_space.scorer_height,
        join.state.frozen_space.scorer_width,
    )
    if target.shape != expected_shape or labels.shape != expected_shape:
        raise PairPopulationEnvelopeError("teacher/candidate semantic geometry differs from frozen scorer space")

    source_to_population = {source: index for index, source in enumerate(population.source_pair_ids)}
    ir_to_source = population.domain_index(PairDomain.IR).local_to_source_pair_ids
    mismatches: list[tuple[int, int, int]] = []
    obligation_count = 0
    for cell in join.obligation_ir.frame1_cells:
        source = ir_to_source[cell.pair_id]
        if source not in source_to_population:
            continue
        local = source_to_population[source]
        if int(target[local, cell.row, cell.col]) != cell.winner_class_id:
            raise PairPopulationEnvelopeError("IR winner obligation differs from freshly recovered PBR semantics")
        obligation_count += 1
        if int(labels[local, cell.row, cell.col]) != cell.winner_class_id:
            mismatches.append((source, cell.row, cell.col))
    if obligation_count == 0:
        raise PairPopulationEnvelopeError("active PairPopulation has no IR obligations")

    if any(not isinstance(item, SparseObligationOwnership) for item in sparse_ownership):
        raise PairPopulationEnvelopeError("sparse ownership contains an invalid row")
    owners = tuple(sorted(sparse_ownership))
    section_role_by_owner = {
        SparseDebtOwner.FRAME1_PREIMAGE: CountedSectionRole.FRAME1_PREIMAGE,
        SparseDebtOwner.TERMINAL_QUOTIENT: CountedSectionRole.TERMINAL_QUOTIENT,
    }
    for owner in owners:
        expected_section = compact_program.section(section_role_by_owner[owner.owner])
        if owner.owner_section_sha256 != expected_section.payload_sha256:
            raise PairPopulationEnvelopeError("sparse owner is not bound to its reopened counted section")
    owner_addresses = [(item.source_pair_id, item.row, item.col) for item in owners]
    if policy is IRCoveragePolicy.COMPLETE:
        if mismatches or owners:
            raise PairPopulationEnvelopeError("complete IR coverage has unmatched or redundantly owned obligations")
    elif owner_addresses != sorted(mismatches):
        raise PairPopulationEnvelopeError("sparse owners do not exactly cover every unmatched IR obligation")
    return IRCoverageReceipt(
        pair_population_sha256=population.population_sha256,
        obligation_ir_sha256=join.obligation_ir.ir_sha256,
        predictor_program_sha256=join.predictor.program_sha256,
        pbr2_packet_sha256=pbr_reference.artifact_sha256,
        pbr2_target_semantic_sha256=pbr_reference.semantic_sha256 or "",
        candidate_program_sha256=_payload_sha256(candidate_program),
        candidate_program_bytes=len(candidate_program),
        compact_program_sha256=compact_program.program_sha256,
        candidate_decoder_contract_id=GENERATIVE_CORRECTION_PACKET_SCHEMA,
        candidate_semantic_identity_sha256=_array_identity_sha256(labels),
        policy=policy,
        obligation_count=obligation_count,
        matched_count=obligation_count - len(mismatches),
        sparse_ownership=owners,
        _derivation_proof=_IR_COVERAGE_DERIVATION_PROOF,
    )


@dataclass(frozen=True, slots=True)
class CountedPayloadProvenance:
    """Producer-declared section custody without novelty or score authority.

    The declaration is retained and byte-bound, but a production typed parser
    has not yet independently classified arbitrary A/T section contents.  It
    therefore cannot prove encoder-truth absence, legality, or originality.
    """

    artifact_class: CountedArtifactClass
    lineage: CountedPayloadLineage
    producer_id: str
    producer_source_sha256: str
    derivation_input_sha256: str
    video_derived: bool
    originality_claimed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_class, CountedArtifactClass):
            raise PairPopulationEnvelopeError("counted payload artifact class is invalid")
        if self.artifact_class in _FORBIDDEN_COUNTED_ARTIFACT_CLASSES:
            raise PairPopulationEnvelopeError(
                f"exact {self.artifact_class.value} artifact is forbidden in counted candidate bytes"
            )
        if not isinstance(self.lineage, CountedPayloadLineage):
            raise PairPopulationEnvelopeError("counted payload lineage is invalid")
        _text(self.producer_id, "counted payload producer_id")
        _sha256(self.producer_source_sha256, "counted payload producer source SHA-256")
        _sha256(self.derivation_input_sha256, "counted payload derivation input SHA-256")
        if type(self.video_derived) is not bool:
            raise PairPopulationEnvelopeError("counted payload video_derived must be an exact boolean")
        if type(self.originality_claimed) is not bool or self.originality_claimed:
            raise PairPopulationEnvelopeError(
                "section lineage is custody only; this structural receipt cannot claim originality"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_class": self.artifact_class.value,
            "lineage": self.lineage.value,
            "producer_id": self.producer_id,
            "producer_source_sha256": self.producer_source_sha256,
            "derivation_input_sha256": self.derivation_input_sha256,
            "video_derived": self.video_derived,
            "originality_claimed": self.originality_claimed,
            "classification_authority": "producer_declared_unverified_until_production_typed_parseback",
        }


@dataclass(frozen=True, slots=True)
class CountedProgramSection:
    """One exact, contiguous counted section derived from retained bytes."""

    role: CountedSectionRole
    offset: int
    byte_length: int
    payload_sha256: str
    provenance: CountedPayloadProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.role, CountedSectionRole):
            raise PairPopulationEnvelopeError("counted section role is invalid")
        _nonnegative_int(self.offset, "counted section offset")
        _positive_int(self.byte_length, "counted section byte_length")
        _sha256(self.payload_sha256, "counted section payload SHA-256")
        if not isinstance(self.provenance, CountedPayloadProvenance):
            raise PairPopulationEnvelopeError("counted section requires typed payload provenance")

    @property
    def stop(self) -> int:
        return self.offset + self.byte_length

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "offset": self.offset,
            "byte_length": self.byte_length,
            "payload_sha256": self.payload_sha256,
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RoleCounterfactualProgram:
    """One complete valid program differing in exactly one typed role section."""

    role: CountedSectionRole
    program_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, CountedSectionRole):
            raise PairPopulationEnvelopeError("counterfactual program role is invalid")
        if not isinstance(self.program_bytes, bytes) or not self.program_bytes:
            raise PairPopulationEnvelopeError("counterfactual program must retain exact nonempty bytes")

    @property
    def program_sha256(self) -> str:
        return _payload_sha256(self.program_bytes)


@dataclass(frozen=True, slots=True)
class SectionCausalityReceipt:
    """Receiver-realized proof for one valid same-role program counterfactual."""

    role: CountedSectionRole
    baseline_program_sha256: str
    counterfactual_program_sha256: str
    baseline_section_sha256: str
    counterfactual_section_sha256: str
    unchanged_non_target_sections_sha256: str
    y0_changed: bool
    y1_changed: bool
    receiver_exception_free: bool = True
    target_section_only_changed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.role, CountedSectionRole):
            raise PairPopulationEnvelopeError("section-causality role is invalid")
        for label, value in (
            ("baseline program", self.baseline_program_sha256),
            ("counterfactual program", self.counterfactual_program_sha256),
            ("baseline section", self.baseline_section_sha256),
            ("counterfactual section", self.counterfactual_section_sha256),
            ("unchanged non-target sections", self.unchanged_non_target_sections_sha256),
        ):
            _sha256(value, f"{label} SHA-256")
        if self.baseline_program_sha256 == self.counterfactual_program_sha256:
            raise PairPopulationEnvelopeError("counterfactual full program bytes did not change")
        if self.baseline_section_sha256 == self.counterfactual_section_sha256:
            raise PairPopulationEnvelopeError("counterfactual target section bytes did not change")
        for label, value in (
            ("y0_changed", self.y0_changed),
            ("y1_changed", self.y1_changed),
            ("receiver_exception_free", self.receiver_exception_free),
            ("target_section_only_changed", self.target_section_only_changed),
        ):
            if type(value) is not bool:
                raise PairPopulationEnvelopeError(f"section-causality {label} must be an exact boolean")
        if not self.receiver_exception_free or not self.target_section_only_changed:
            raise PairPopulationEnvelopeError("section causality requires a valid target-only counterfactual")
        if self.role in (CountedSectionRole.GENERATIVE_CORRECTION, CountedSectionRole.FRAME1_PREIMAGE):
            if not self.y1_changed:
                raise PairPopulationEnvelopeError(f"counted {self.role.value} counterfactual did not change Y1")
        elif self.role in (
            CountedSectionRole.FRAME0_FROM_EXACT_Y1,
            CountedSectionRole.FRAME0_POSE_RESIDUAL,
        ):
            if not self.y0_changed or self.y1_changed:
                raise PairPopulationEnvelopeError("counted frame0 counterfactual must change Y0 with Y1 fixed")
        elif not (self.y0_changed or self.y1_changed):
            raise PairPopulationEnvelopeError("counted terminal quotient counterfactual changed neither Y plane")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "baseline_program_sha256": self.baseline_program_sha256,
            "counterfactual_program_sha256": self.counterfactual_program_sha256,
            "baseline_section_sha256": self.baseline_section_sha256,
            "counterfactual_section_sha256": self.counterfactual_section_sha256,
            "unchanged_non_target_sections_sha256": self.unchanged_non_target_sections_sha256,
            "y0_changed": self.y0_changed,
            "y1_changed": self.y1_changed,
            "receiver_exception_free": self.receiver_exception_free,
            "target_section_only_changed": self.target_section_only_changed,
        }


@dataclass(frozen=True, slots=True)
class ReceiverSourceBinding:
    """Direct Python reference-function identity, explicitly non-transitive."""

    source_sha256: str
    source_bytes: int
    module: str
    qualname: str
    code_sha256: str
    callable_sha256: str
    binding_scope: str = RECEIVER_BINDING_SCOPE
    python_reference_mode_only: bool = True
    closure_defaults_globals_and_transitive_imports_bound: bool = False
    standalone_archive_runtime_authority: bool = False

    def __post_init__(self) -> None:
        _sha256(self.source_sha256, "receiver source SHA-256")
        _positive_int(self.source_bytes, "receiver source bytes")
        _text(self.module, "receiver module")
        _text(self.qualname, "receiver qualname")
        _sha256(self.code_sha256, "receiver code SHA-256")
        _sha256(self.callable_sha256, "receiver callable SHA-256")
        if self.binding_scope != RECEIVER_BINDING_SCOPE:
            raise PairPopulationEnvelopeError("receiver binding scope differs")
        if (
            self.python_reference_mode_only is not True
            or self.closure_defaults_globals_and_transitive_imports_bound is not False
            or self.standalone_archive_runtime_authority is not False
        ):
            raise PairPopulationEnvelopeError("receiver binding overclaims executable runtime custody")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "module": self.module,
            "qualname": self.qualname,
            "code_sha256": self.code_sha256,
            "callable_sha256": self.callable_sha256,
            "binding_scope": self.binding_scope,
            "python_reference_mode_only": self.python_reference_mode_only,
            "closure_defaults_globals_and_transitive_imports_bound": (
                self.closure_defaults_globals_and_transitive_imports_bound
            ),
            "standalone_archive_runtime_authority": self.standalone_archive_runtime_authority,
        }


@dataclass(frozen=True, slots=True)
class CompactGeneratorBatch:
    """One bounded generated output batch; dense values are not serialized."""

    source_pair_ids: tuple[int, ...]
    scorer_y0: np.ndarray
    scorer_y1: np.ndarray

    def __post_init__(self) -> None:
        if type(self.source_pair_ids) is not tuple:
            raise PairPopulationEnvelopeError("compact generator batch IDs must be an exact tuple")
        _strict_tuple_of_ids(self.source_pair_ids, "compact generator batch IDs")
        y0 = np.asarray(self.scorer_y0)
        y1 = np.asarray(self.scorer_y1)
        if y0.dtype != np.uint8 or y1.dtype != np.uint8 or y0.shape != y1.shape:
            raise PairPopulationEnvelopeError("compact generator outputs must be equal-shape uint8 Y0/Y1")
        if y0.ndim != 4 or y0.shape[0] != len(self.source_pair_ids) or y0.shape[-1] != 3:
            raise PairPopulationEnvelopeError("compact generator outputs must be [pairs,height,width,3]")
        object.__setattr__(self, "scorer_y0", _immutable_array(y0))
        object.__setattr__(self, "scorer_y1", _immutable_array(y1))


@dataclass(frozen=True, slots=True)
class CompactGeneratorDecode:
    """Typed Python-reference receiver output for a compact generator program.

    Payload-presence booleans below are receiver declarations.  Until the
    production A/T grammars independently parse every byte inside a standalone
    archive runtime, they are not proof that forbidden encoder/scorer material
    is absent.
    """

    decoder_contract_id: str
    batches: tuple[CompactGeneratorBatch, ...]
    program_roles: tuple[CompactProgramRole, ...]
    predictor_pose6_sha256: str
    conditioned_frame1_identity_sha256: str
    consumed_program_sha256: str
    section_manifest: tuple[CountedProgramSection, ...]
    role_counterfactuals: tuple[RoleCounterfactualProgram, ...]
    frame0_pose_mode: Frame0PoseMode = Frame0PoseMode.RESIDUAL_BEYOND_V9_POSE6
    absolute_pose6_values_present: bool = False
    stored_dense_y_present: bool = False
    stored_camera_preimages_present: bool = False
    encoder_only_artifacts_present: bool = False
    scorer_artifacts_present: bool = False
    ground_truth_artifacts_present: bool = False
    payload_presence_declarations_verified_by_production_parsers: bool = False

    def __post_init__(self) -> None:
        _text(self.decoder_contract_id, "compact generator decoder contract")
        if type(self.batches) is not tuple or not self.batches:
            raise PairPopulationEnvelopeError("compact generator decode requires nonempty bounded batches")
        if any(not isinstance(batch, CompactGeneratorBatch) for batch in self.batches):
            raise PairPopulationEnvelopeError("compact generator decode contains an invalid batch")
        expected_roles = tuple(CompactProgramRole)
        if type(self.program_roles) is not tuple or any(
            not isinstance(role, CompactProgramRole) for role in self.program_roles
        ):
            raise PairPopulationEnvelopeError("compact program roles must be an exact typed tuple")
        if self.program_roles != expected_roles:
            raise PairPopulationEnvelopeError(
                "compact program must exclusively own generator and frame0 residual roles"
            )
        _sha256(self.predictor_pose6_sha256, "compact generator predictor Pose6 SHA-256")
        _sha256(self.conditioned_frame1_identity_sha256, "compact generator conditioned frame1 SHA-256")
        _sha256(self.consumed_program_sha256, "compact generator consumed-program SHA-256")
        if type(self.section_manifest) is not tuple or any(
            not isinstance(section, CountedProgramSection) for section in self.section_manifest
        ):
            raise PairPopulationEnvelopeError("compact generator section manifest must be an exact typed tuple")
        if type(self.role_counterfactuals) is not tuple or any(
            not isinstance(item, RoleCounterfactualProgram) for item in self.role_counterfactuals
        ):
            raise PairPopulationEnvelopeError("compact role counterfactuals must be an exact typed tuple")
        if tuple(item.role for item in self.role_counterfactuals) != tuple(
            section.role for section in self.section_manifest
        ):
            raise PairPopulationEnvelopeError(
                "compact receiver must provide one canonical full-program counterfactual per counted role"
            )
        if self.frame0_pose_mode is not Frame0PoseMode.RESIDUAL_BEYOND_V9_POSE6:
            raise PairPopulationEnvelopeError("frame0 program may own only the residual beyond V9 Pose6")
        flags = (
            self.absolute_pose6_values_present,
            self.stored_dense_y_present,
            self.stored_camera_preimages_present,
            self.encoder_only_artifacts_present,
            self.scorer_artifacts_present,
            self.ground_truth_artifacts_present,
            self.payload_presence_declarations_verified_by_production_parsers,
        )
        if any(type(value) is not bool for value in flags):
            raise PairPopulationEnvelopeError("compact payload-presence fields must be exact booleans")
        if any(
            (
                self.absolute_pose6_values_present,
                self.encoder_only_artifacts_present,
                self.scorer_artifacts_present,
                self.ground_truth_artifacts_present,
            )
        ):
            raise PairPopulationEnvelopeError(
                "compact program contains duplicated pose or an exact encoder/scorer/ground-truth artifact"
            )
        if self.payload_presence_declarations_verified_by_production_parsers:
            raise PairPopulationEnvelopeError(
                "Python reference receiver cannot claim production-parser payload verification"
            )
        has_dense = any(
            section.provenance.artifact_class is CountedArtifactClass.DENSE_REALIZED_Y
            for section in self.section_manifest
        )
        has_camera_preimage = any(
            section.provenance.artifact_class is CountedArtifactClass.CAMERA_PREIMAGE
            for section in self.section_manifest
        )
        if self.stored_dense_y_present != has_dense or self.stored_camera_preimages_present != has_camera_preimage:
            raise PairPopulationEnvelopeError("dense/preimage presence flags differ from typed counted provenance")


def _generated_plane_identities(
    decoded: CompactGeneratorDecode,
    *,
    expected_source_ids: tuple[int, ...],
    scorer_height: int,
    scorer_width: int,
) -> tuple[str, str]:
    source_ids: list[int] = []
    y0_digest = hashlib.sha256()
    y1_digest = hashlib.sha256()
    byte_length = 0
    for batch in decoded.batches:
        expected_shape = (len(batch.source_pair_ids), scorer_height, scorer_width, 3)
        if batch.scorer_y0.shape != expected_shape:
            raise PairPopulationEnvelopeError("compact generator batch geometry differs from explicit admission")
        source_ids.extend(batch.source_pair_ids)
        y0_digest.update(np.ascontiguousarray(batch.scorer_y0).view(np.uint8))
        y1_digest.update(np.ascontiguousarray(batch.scorer_y1).view(np.uint8))
        byte_length += int(batch.scorer_y0.nbytes)
    if tuple(source_ids) != expected_source_ids:
        raise PairPopulationEnvelopeError("compact generator batches do not cover exact V10 source order")
    shape = [len(expected_source_ids), scorer_height, scorer_width, 3]
    y0_identity = canonical_sha256(
        {"dtype": "uint8", "shape": shape, "byte_length": byte_length, "bytes_sha256": y0_digest.hexdigest()}
    )
    y1_identity = canonical_sha256(
        {"dtype": "uint8", "shape": shape, "byte_length": byte_length, "bytes_sha256": y1_digest.hexdigest()}
    )
    return y0_identity, y1_identity


def _validate_section_manifest(
    program: bytes,
    sections: tuple[CountedProgramSection, ...],
) -> None:
    legacy_roles = (
        CountedSectionRole.GENERATIVE_CORRECTION,
        CountedSectionRole.FRAME1_PREIMAGE,
        CountedSectionRole.FRAME0_POSE_RESIDUAL,
    )
    roles = tuple(section.role for section in sections)
    allowed_roles = (legacy_roles, (*legacy_roles, CountedSectionRole.TERMINAL_QUOTIENT))
    if roles not in allowed_roles:
        raise PairPopulationEnvelopeError(
            "counted section roles/order differ from legacy G/frame1/frame0-pose[/terminal]"
        )
    _validate_counted_section_bytes(program, sections)


def _validate_reverse_causal_section_manifest(
    program: bytes,
    sections: tuple[CountedProgramSection, ...],
) -> None:
    reverse_causal_roles = (
        CountedSectionRole.GENERATIVE_CORRECTION,
        CountedSectionRole.FRAME0_FROM_EXACT_Y1,
    )
    roles = tuple(section.role for section in sections)
    allowed_roles = (
        reverse_causal_roles,
        (*reverse_causal_roles, CountedSectionRole.TERMINAL_QUOTIENT),
    )
    if roles not in allowed_roles:
        raise PairPopulationEnvelopeError(
            "reverse-causal counted section roles/order differ from G/frame0-from-exact-Y1[/terminal]"
        )
    _validate_counted_section_bytes(program, sections)


def _validate_counted_section_bytes(
    program: bytes,
    sections: tuple[CountedProgramSection, ...],
) -> None:
    cursor = 0
    for section in sections:
        if section.offset != cursor or section.stop > len(program):
            raise PairPopulationEnvelopeError("counted section manifest has a gap, overlap, or out-of-range span")
        if _payload_sha256(program[section.offset : section.stop]) != section.payload_sha256:
            raise PairPopulationEnvelopeError("counted section hash was not derived from retained program bytes")
        cursor = section.stop
    if cursor != len(program):
        raise PairPopulationEnvelopeError("counted section manifest leaves unconsumed program bytes")


def validate_counted_program_sections(
    program: bytes,
    sections: tuple[CountedProgramSection, ...],
) -> None:
    """Validate the legacy Pose6-owned G/frame1/frame0-pose byte partition."""

    if not isinstance(program, bytes) or not program:
        raise PairPopulationEnvelopeError("counted program must be exact nonempty bytes")
    if type(sections) is not tuple:
        raise PairPopulationEnvelopeError("counted section manifest must be an exact tuple")
    _validate_section_manifest(program, sections)


def validate_reverse_causal_counted_program_sections(
    program: bytes,
    sections: tuple[CountedProgramSection, ...],
) -> None:
    """Validate the disjoint reverse-causal G/frame0-from-exact-Y1 partition.

    ``FRAME0_FROM_EXACT_Y1`` is a coupled preimage conditioned on an already
    exact ``Y1``.  It is neither the legacy frame1 actuator nor the residual
    beyond V9 Pose6, so this grammar must not enter the legacy Compact seal.
    """

    if not isinstance(program, bytes) or not program:
        raise PairPopulationEnvelopeError("reverse-causal counted program must be exact nonempty bytes")
    if type(sections) is not tuple:
        raise PairPopulationEnvelopeError("reverse-causal counted section manifest must be an exact tuple")
    _validate_reverse_causal_section_manifest(program, sections)


def _decoded_behavior_identity(
    decoded: CompactGeneratorDecode,
    *,
    expected_source_ids: tuple[int, ...],
    scorer_height: int,
    scorer_width: int,
) -> tuple[Any, ...]:
    if not isinstance(decoded, CompactGeneratorDecode):
        raise PairPopulationEnvelopeError("compact receiver returned no typed generated outputs")
    y0_identity, y1_identity = _generated_plane_identities(
        decoded,
        expected_source_ids=expected_source_ids,
        scorer_height=scorer_height,
        scorer_width=scorer_width,
    )
    return (
        y0_identity,
        y1_identity,
        decoded.predictor_pose6_sha256,
        decoded.conditioned_frame1_identity_sha256,
        decoded.program_roles,
        decoded.frame0_pose_mode,
        decoded.absolute_pose6_values_present,
        decoded.stored_dense_y_present,
        decoded.stored_camera_preimages_present,
        decoded.encoder_only_artifacts_present,
        decoded.scorer_artifacts_present,
        decoded.ground_truth_artifacts_present,
        decoded.payload_presence_declarations_verified_by_production_parsers,
    )


def _portable_code_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code": _portable_code_body(value)}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if type(value) is complex:
        return {"complex_hex": [value.real.hex(), value.imag.hex()]}
    if type(value) is bytes:
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if type(value) is tuple:
        return {"tuple": [_portable_code_constant(item) for item in value]}
    if type(value) is frozenset:
        items = [_portable_code_constant(item) for item in value]
        items.sort(key=canonical_json_bytes)
        return {"frozenset": items}
    raise PairPopulationEnvelopeError(
        f"compact receiver code contains unsupported constant type {type(value).__name__}"
    )


def _portable_code_body(code: CodeType) -> dict[str, Any]:
    return {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_sha256": _payload_sha256(code.co_code),
        "exceptiontable_sha256": _payload_sha256(code.co_exceptiontable),
        "constants": [_portable_code_constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _source_code_object(source: bytes, source_path: Path, qualname: str) -> CodeType:
    try:
        root = compile(source, str(source_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise PairPopulationEnvelopeError("compact receiver source artifact does not compile") from exc
    matches: list[CodeType] = []

    def visit(code: CodeType) -> None:
        if code.co_qualname == qualname:
            matches.append(code)
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                visit(constant)

    visit(root)
    if len(matches) != 1:
        raise PairPopulationEnvelopeError("compact receiver code is not uniquely resolvable from its source")
    return matches[0]


def _bind_receiver_source(
    receiver: Callable[[bytes], CompactGeneratorDecode],
    receiver_source_bytes: bytes,
) -> ReceiverSourceBinding:
    """Bind direct Python function/code identity, not its runtime closure."""

    if not inspect.isfunction(receiver):
        raise PairPopulationEnvelopeError("compact reference receiver must be a source-bound Python function")
    source_name = inspect.getsourcefile(receiver)
    if not isinstance(source_name, str) or not source_name:
        raise PairPopulationEnvelopeError("compact receiver has no inspectable source artifact")
    source_path = Path(source_name).resolve()
    try:
        before = source_path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PairPopulationEnvelopeError("compact receiver source must be a regular file")
        reopened_source = source_path.read_bytes()
        after = source_path.stat()
    except OSError as exc:
        raise PairPopulationEnvelopeError("compact receiver source artifact cannot be reopened") from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity or len(reopened_source) != after.st_size:
        raise PairPopulationEnvelopeError("compact receiver source mutated during reopen")
    if reopened_source != receiver_source_bytes:
        raise PairPopulationEnvelopeError("compact receiver callable is not bound to the supplied source artifact")
    source_sha256 = _payload_sha256(reopened_source)
    expected_code = _source_code_object(reopened_source, source_path, receiver.__qualname__)
    executed_code_body = _portable_code_body(receiver.__code__)
    if canonical_json_bytes(executed_code_body) != canonical_json_bytes(_portable_code_body(expected_code)):
        raise PairPopulationEnvelopeError("executed compact receiver code differs from its exact source artifact")
    code_sha256 = canonical_sha256(executed_code_body)
    callable_sha256 = canonical_sha256(
        {
            "schema": "tac.compact_generator_receiver_callable.v1",
            "module": receiver.__module__,
            "qualname": receiver.__qualname__,
            "source_sha256": source_sha256,
            "code_sha256": code_sha256,
        }
    )
    return ReceiverSourceBinding(
        source_sha256=source_sha256,
        source_bytes=len(reopened_source),
        module=receiver.__module__,
        qualname=receiver.__qualname__,
        code_sha256=code_sha256,
        callable_sha256=callable_sha256,
    )


def _execute_source_bound_receiver(
    receiver: Callable[[bytes], CompactGeneratorDecode],
    receiver_source_bytes: bytes,
    binding: ReceiverSourceBinding,
    program: bytes,
    *,
    phase: str,
) -> CompactGeneratorDecode:
    """Execute only while the callable remains byte/code-identical to its source."""

    if _bind_receiver_source(receiver, receiver_source_bytes) != binding:
        raise PairPopulationEnvelopeError(f"{phase} receiver binding changed before execution")
    try:
        decoded = receiver(program)
    except Exception as exc:
        if _bind_receiver_source(receiver, receiver_source_bytes) != binding:
            raise PairPopulationEnvelopeError(f"{phase} receiver binding changed while raising") from exc
        raise PairPopulationEnvelopeError(
            f"{phase} receiver raised ({exc}); the program is not a valid counterfactual"
        ) from exc
    if _bind_receiver_source(receiver, receiver_source_bytes) != binding:
        raise PairPopulationEnvelopeError(f"{phase} receiver binding changed during execution")
    return decoded


def _decoded_replay_identity(
    decoded: CompactGeneratorDecode,
    *,
    expected_source_ids: tuple[int, ...],
    scorer_height: int,
    scorer_width: int,
) -> tuple[Any, ...]:
    return (
        *_decoded_behavior_identity(
            decoded,
            expected_source_ids=expected_source_ids,
            scorer_height=scorer_height,
            scorer_width=scorer_width,
        ),
        decoded.decoder_contract_id,
        decoded.consumed_program_sha256,
        decoded.section_manifest,
        decoded.role_counterfactuals,
    )


def _require_section_causality(
    *,
    program: bytes,
    decoded: CompactGeneratorDecode,
    receiver: Callable[[bytes], CompactGeneratorDecode],
    receiver_source_bytes: bytes,
    receiver_binding: ReceiverSourceBinding,
    predictor_state: PredictorSemanticStateV1,
    expected_source_ids: tuple[int, ...],
    scorer_height: int,
    scorer_width: int,
) -> tuple[SectionCausalityReceipt, ...]:
    baseline = _generated_plane_identities(
        decoded,
        expected_source_ids=expected_source_ids,
        scorer_height=scorer_height,
        scorer_width=scorer_width,
    )
    baseline_sections = {section.role: program[section.offset : section.stop] for section in decoded.section_manifest}
    receipts: list[SectionCausalityReceipt] = []

    for counterfactual in decoded.role_counterfactuals:
        variant = _execute_source_bound_receiver(
            receiver,
            receiver_source_bytes,
            receiver_binding,
            counterfactual.program_bytes,
            phase=f"counted {counterfactual.role.value} counterfactual",
        )
        if not isinstance(variant, CompactGeneratorDecode):
            raise PairPopulationEnvelopeError("counterfactual receiver returned no typed generated outputs")
        if variant.consumed_program_sha256 != counterfactual.program_sha256:
            raise PairPopulationEnvelopeError("counterfactual receiver did not bind exact program bytes")
        _validate_section_manifest(counterfactual.program_bytes, variant.section_manifest)
        if tuple(section.role for section in variant.section_manifest) != tuple(baseline_sections):
            raise PairPopulationEnvelopeError("counterfactual changed the canonical counted role ordering")
        variant_sections = {
            section.role: counterfactual.program_bytes[section.offset : section.stop]
            for section in variant.section_manifest
        }
        variant_manifest = {section.role: section for section in variant.section_manifest}
        baseline_manifest = {section.role: section for section in decoded.section_manifest}
        target = counterfactual.role
        if variant_sections[target] == baseline_sections[target]:
            raise PairPopulationEnvelopeError(
                f"counted {target.value} counterfactual did not change its target section"
            )
        for role, payload in baseline_sections.items():
            if role is not target and variant_sections[role] != payload:
                raise PairPopulationEnvelopeError(
                    f"counted {target.value} counterfactual changed non-target {role.value} bytes"
                )
            if variant_manifest[role].provenance != baseline_manifest[role].provenance:
                raise PairPopulationEnvelopeError(
                    f"counted {target.value} counterfactual changed typed provenance for {role.value}"
                )
        if target is CountedSectionRole.GENERATIVE_CORRECTION:
            try:
                apply_generative_taskspace_correction(variant_sections[target], predictor_state=predictor_state)
            except GenerativeTaskspaceCorrectionError as exc:
                raise PairPopulationEnvelopeError(
                    "counterfactual G section did not reopen through its typed receiver"
                ) from exc
        y0_identity, y1_identity = _generated_plane_identities(
            variant,
            expected_source_ids=expected_source_ids,
            scorer_height=scorer_height,
            scorer_width=scorer_width,
        )
        if (
            variant.decoder_contract_id != decoded.decoder_contract_id
            or variant.predictor_pose6_sha256 != decoded.predictor_pose6_sha256
            or variant.conditioned_frame1_identity_sha256 != y1_identity
            or variant.program_roles != decoded.program_roles
            or variant.frame0_pose_mode is not decoded.frame0_pose_mode
        ):
            raise PairPopulationEnvelopeError("counterfactual receiver returned invalid role or identity custody")
        y0_changed = y0_identity != baseline[0]
        y1_changed = y1_identity != baseline[1]
        unchanged_non_target_sections_sha256 = canonical_sha256(
            [
                {
                    "role": role.value,
                    "payload_sha256": _payload_sha256(payload),
                    "byte_length": len(payload),
                }
                for role, payload in baseline_sections.items()
                if role is not target
            ]
        )
        receipts.append(
            SectionCausalityReceipt(
                role=target,
                baseline_program_sha256=_payload_sha256(program),
                counterfactual_program_sha256=counterfactual.program_sha256,
                baseline_section_sha256=_payload_sha256(baseline_sections[target]),
                counterfactual_section_sha256=_payload_sha256(variant_sections[target]),
                unchanged_non_target_sections_sha256=unchanged_non_target_sections_sha256,
                y0_changed=y0_changed,
                y1_changed=y1_changed,
            )
        )
    return tuple(receipts)


@dataclass(frozen=True, slots=True)
class CompactObligationGeneratorProgram:
    """Exact counted bytes plus typed provenance and a fresh receiver reopen."""

    program_bytes: bytes
    decoder_contract_id: str
    pair_population_sha256: str
    explicit_preimage_result_sha256: str
    generated_y0_identity_sha256: str
    generated_y1_identity_sha256: str
    predictor_pose6_sha256: str
    conditioned_frame1_identity_sha256: str
    receiver_binding: ReceiverSourceBinding
    receiver_state_sha256: str
    section_manifest: tuple[CountedProgramSection, ...]
    section_causality: tuple[SectionCausalityReceipt, ...]
    program_roles: tuple[CompactProgramRole, ...]
    frame0_pose_mode: Frame0PoseMode
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    _reopen_proof: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reopen_proof is not _COMPACT_PROGRAM_REOPEN_PROOF:
            raise PairPopulationEnvelopeError("compact generator must be sealed by a fresh receiver reopen")
        if not isinstance(self.program_bytes, bytes) or not self.program_bytes:
            raise PairPopulationEnvelopeError("compact generator must retain exact nonempty program bytes")
        _text(self.decoder_contract_id, "compact generator decoder_contract_id")
        for label, value in (
            ("pair population", self.pair_population_sha256),
            ("explicit preimage result", self.explicit_preimage_result_sha256),
            ("generated Y0", self.generated_y0_identity_sha256),
            ("generated Y1", self.generated_y1_identity_sha256),
            ("predictor Pose6", self.predictor_pose6_sha256),
            ("conditioned frame1", self.conditioned_frame1_identity_sha256),
            ("receiver state", self.receiver_state_sha256),
        ):
            _sha256(value, f"{label} SHA-256")
        if not isinstance(self.receiver_binding, ReceiverSourceBinding):
            raise PairPopulationEnvelopeError("compact generator requires a typed receiver-source binding")
        if type(self.program_roles) is not tuple or any(
            not isinstance(role, CompactProgramRole) for role in self.program_roles
        ):
            raise PairPopulationEnvelopeError("compact generator roles must be exact typed owners")
        _validate_section_manifest(self.program_bytes, self.section_manifest)
        if type(self.section_causality) is not tuple or any(
            not isinstance(item, SectionCausalityReceipt) for item in self.section_causality
        ):
            raise PairPopulationEnvelopeError("compact generator section causality must be an exact typed tuple")
        if tuple(item.role for item in self.section_causality) != tuple(
            section.role for section in self.section_manifest
        ):
            raise PairPopulationEnvelopeError("compact generator lacks canonical causality for every counted role")
        if any(item.baseline_program_sha256 != self.program_sha256 for item in self.section_causality):
            raise PairPopulationEnvelopeError("section causality is not bound to the retained compact program")
        if self.program_roles != tuple(CompactProgramRole):
            raise PairPopulationEnvelopeError("compact generator role ownership differs")
        if self.frame0_pose_mode is not Frame0PoseMode.RESIDUAL_BEYOND_V9_POSE6:
            raise PairPopulationEnvelopeError("compact frame0 program is not residual-only")
        if (
            type(self.research_only) is not bool
            or not self.research_only
            or type(self.score_claim) is not bool
            or self.score_claim
            or type(self.promotion_eligible) is not bool
            or self.promotion_eligible
        ):
            raise PairPopulationEnvelopeError("compact structural program cannot claim score or promotion authority")

    @property
    def program_sha256(self) -> str:
        return _payload_sha256(self.program_bytes)

    @property
    def receiver_source_sha256(self) -> str:
        return self.receiver_binding.source_sha256

    @property
    def receiver_callable_sha256(self) -> str:
        return self.receiver_binding.callable_sha256

    def section(self, role: CountedSectionRole) -> CountedProgramSection:
        for section in self.section_manifest:
            if section.role is role:
                return section
        raise PairPopulationEnvelopeError(f"counted {role.value} section is absent")

    @classmethod
    def seal(
        cls,
        program_bytes: bytes,
        *,
        join: ReopenedObjectJoin,
        receiver: Callable[[bytes], CompactGeneratorDecode],
        receiver_source_bytes: bytes,
    ) -> CompactObligationGeneratorProgram:
        """Fresh-reopen exact compact bytes and match generated planes to admission."""

        if (
            not isinstance(program_bytes, bytes)
            or not program_bytes
            or not callable(receiver)
            or not isinstance(receiver_source_bytes, bytes)
            or not receiver_source_bytes
        ):
            raise PairPopulationEnvelopeError(
                "compact program seal requires exact program/receiver bytes and a callable"
            )
        receiver_binding = _bind_receiver_source(receiver, receiver_source_bytes)
        explicit = join.explicit_preimage
        if _payload_sha256(program_bytes) == explicit.receiver_packet_sha256:
            raise PairPopulationEnvelopeError("hash-only dense V10 packet substitution is forbidden")
        decoded = _execute_source_bound_receiver(
            receiver,
            receiver_source_bytes,
            receiver_binding,
            program_bytes,
            phase="baseline compact program",
        )
        if not isinstance(decoded, CompactGeneratorDecode):
            raise PairPopulationEnvelopeError("compact receiver returned no typed generated outputs")
        if decoded.consumed_program_sha256 != _payload_sha256(program_bytes):
            raise PairPopulationEnvelopeError("compact receiver did not bind the exact retained program bytes")
        _validate_section_manifest(program_bytes, decoded.section_manifest)
        replayed = _execute_source_bound_receiver(
            receiver,
            receiver_source_bytes,
            receiver_binding,
            program_bytes,
            phase="deterministic compact replay",
        )
        if _decoded_replay_identity(
            replayed,
            expected_source_ids=join.pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids,
            scorer_height=explicit.scorer_height,
            scorer_width=explicit.scorer_width,
        ) != _decoded_replay_identity(
            decoded,
            expected_source_ids=join.pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids,
            scorer_height=explicit.scorer_height,
            scorer_width=explicit.scorer_width,
        ):
            raise PairPopulationEnvelopeError("compact receiver is nondeterministic on identical program bytes")
        if any(section.payload_sha256 == explicit.receiver_packet_sha256 for section in decoded.section_manifest):
            raise PairPopulationEnvelopeError("dense V10 packet section substitution is forbidden")
        generative_section = decoded.section_manifest[0]
        generative_packet = program_bytes[generative_section.offset : generative_section.stop]
        predictor_state = join.generative_predictor_state()
        try:
            apply_generative_taskspace_correction(
                generative_packet,
                predictor_state=predictor_state,
            )
        except GenerativeTaskspaceCorrectionError as exc:
            raise PairPopulationEnvelopeError("counted G section did not reopen through the typed receiver") from exc
        y0_identity, y1_identity = _generated_plane_identities(
            decoded,
            expected_source_ids=join.pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids,
            scorer_height=explicit.scorer_height,
            scorer_width=explicit.scorer_width,
        )
        if y0_identity != explicit.scorer_y0_identity_sha256 or y1_identity != explicit.scorer_y1_identity_sha256:
            raise PairPopulationEnvelopeError(
                "compact generator outputs differ from encoder-only admitted Y identities"
            )
        if decoded.predictor_pose6_sha256 != join.predictor_pose6_sha256:
            raise PairPopulationEnvelopeError("compact frame0 residual is not based on freshly derived V9 Pose6")
        if decoded.conditioned_frame1_identity_sha256 != y1_identity:
            raise PairPopulationEnvelopeError("compact frame0 residual is not conditioned on exact generated frame1")
        section_causality = _require_section_causality(
            program=program_bytes,
            decoded=decoded,
            receiver=receiver,
            receiver_source_bytes=receiver_source_bytes,
            receiver_binding=receiver_binding,
            predictor_state=predictor_state,
            expected_source_ids=join.pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids,
            scorer_height=explicit.scorer_height,
            scorer_width=explicit.scorer_width,
        )
        receiver_state_sha256 = canonical_sha256(
            {
                "schema": "tac.compact_generator_receiver_state.v1",
                "derived_object_identities": join.identity_dict(),
                "decoder_contract_id": decoded.decoder_contract_id,
                "receiver_binding": receiver_binding.as_dict(),
                "program_sha256": _payload_sha256(program_bytes),
                "section_manifest": [section.as_dict() for section in decoded.section_manifest],
                "section_causality": [item.as_dict() for item in section_causality],
            }
        )
        return cls(
            program_bytes=program_bytes,
            decoder_contract_id=decoded.decoder_contract_id,
            pair_population_sha256=join.pair_population.population_sha256,
            explicit_preimage_result_sha256=explicit.result_sha256,
            generated_y0_identity_sha256=y0_identity,
            generated_y1_identity_sha256=y1_identity,
            predictor_pose6_sha256=join.predictor_pose6_sha256,
            conditioned_frame1_identity_sha256=y1_identity,
            receiver_binding=receiver_binding,
            receiver_state_sha256=receiver_state_sha256,
            section_manifest=decoded.section_manifest,
            section_causality=section_causality,
            program_roles=decoded.program_roles,
            frame0_pose_mode=decoded.frame0_pose_mode,
            _reopen_proof=_COMPACT_PROGRAM_REOPEN_PROOF,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPACT_PROGRAM_SCHEMA,
            "program_base64": base64.b64encode(self.program_bytes).decode("ascii"),
            "program_bytes": len(self.program_bytes),
            "program_sha256": self.program_sha256,
            "decoder_contract_id": self.decoder_contract_id,
            "pair_population_sha256": self.pair_population_sha256,
            "explicit_preimage_result_sha256": self.explicit_preimage_result_sha256,
            "generated_y0_identity_sha256": self.generated_y0_identity_sha256,
            "generated_y1_identity_sha256": self.generated_y1_identity_sha256,
            "predictor_pose6_sha256": self.predictor_pose6_sha256,
            "conditioned_frame1_identity_sha256": self.conditioned_frame1_identity_sha256,
            "receiver_source_sha256": self.receiver_source_sha256,
            "receiver_callable_sha256": self.receiver_callable_sha256,
            "receiver_binding": self.receiver_binding.as_dict(),
            "receiver_state_sha256": self.receiver_state_sha256,
            "section_manifest": [section.as_dict() for section in self.section_manifest],
            "section_causality": [item.as_dict() for item in self.section_causality],
            "program_roles": [role.value for role in self.program_roles],
            "frame0_pose_mode": self.frame0_pose_mode.value,
            "stored_dense_y_present": any(
                section.provenance.artifact_class is CountedArtifactClass.DENSE_REALIZED_Y
                for section in self.section_manifest
            ),
            "stored_camera_preimages_present": any(
                section.provenance.artifact_class is CountedArtifactClass.CAMERA_PREIMAGE
                for section in self.section_manifest
            ),
            "payload_presence_declarations": {
                "encoder_only_artifacts_present": False,
                "scorer_artifacts_present": False,
                "ground_truth_artifacts_present": False,
                "authority": "python_reference_receiver_declaration_only",
            },
            "payload_presence_declarations_verified_by_production_parsers": False,
            "standalone_archive_runtime_closure_bound": False,
            "originality_claimed": False,
            "retained_exact_program_bytes": True,
            "fresh_receiver_reopen_required": True,
            "candidate_payload_eligible": False,
            "candidate_payload_blocker": (
                "production typed frame1/frame0 section parsers, standalone runtime closure, and exact archive replay owed"
            ),
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {"schema": COMPACT_PROGRAM_ENVELOPE_SCHEMA, "body": body, "body_sha256": canonical_sha256(body)}
        )

    @classmethod
    def reopen(
        cls,
        payload: bytes,
        *,
        join: ReopenedObjectJoin,
        receiver: Callable[[bytes], CompactGeneratorDecode],
        receiver_source_bytes: bytes,
    ) -> CompactObligationGeneratorProgram:
        """Recover retained program bytes and require another fresh receiver pass."""

        body = _decode_envelope(payload, schema=COMPACT_PROGRAM_ENVELOPE_SCHEMA, label="compact program envelope")
        encoded = body.get("program_base64")
        if not isinstance(encoded, str):
            raise PairPopulationEnvelopeError("compact program bytes are absent")
        try:
            program = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise PairPopulationEnvelopeError("compact program base64 is invalid") from exc
        if base64.b64encode(program).decode("ascii") != encoded:
            raise PairPopulationEnvelopeError("compact program base64 is not canonical")
        result = cls.seal(
            program,
            join=join,
            receiver=receiver,
            receiver_source_bytes=receiver_source_bytes,
        )
        if result.as_dict() != body:
            raise PairPopulationEnvelopeError("compact program receipt differs after fresh reopen")
        return result


@dataclass(frozen=True, slots=True)
class ExclusivePoseOwnership:
    predictor_pose6_sha256: str
    frame0_residual_program_sha256: str
    conditioned_frame1_identity_sha256: str
    pair_population_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("predictor Pose6", self.predictor_pose6_sha256),
            ("frame0 residual program", self.frame0_residual_program_sha256),
            ("conditioned frame1", self.conditioned_frame1_identity_sha256),
            ("pair population", self.pair_population_sha256),
        ):
            _sha256(value, f"{label} SHA-256")

    @classmethod
    def bind(
        cls,
        join: ReopenedObjectJoin,
        compact_program: CompactObligationGeneratorProgram,
    ) -> ExclusivePoseOwnership:
        if compact_program.predictor_pose6_sha256 != join.predictor_pose6_sha256:
            raise PairPopulationEnvelopeError("pose ownership predictor identity differs")
        return cls(
            predictor_pose6_sha256=join.predictor_pose6_sha256,
            frame0_residual_program_sha256=compact_program.section(
                CountedSectionRole.FRAME0_POSE_RESIDUAL
            ).payload_sha256,
            conditioned_frame1_identity_sha256=compact_program.conditioned_frame1_identity_sha256,
            pair_population_sha256=join.pair_population.population_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": POSE_OWNERSHIP_SCHEMA,
            "v9_owner": {
                "domain": "absolute_pose6_predictor",
                "identity_sha256": self.predictor_pose6_sha256,
            },
            "frame0_owner": {
                "domain": Frame0PoseMode.RESIDUAL_BEYOND_V9_POSE6.value,
                "program_sha256": self.frame0_residual_program_sha256,
                "conditioned_frame1_identity_sha256": self.conditioned_frame1_identity_sha256,
            },
            "pair_population_sha256": self.pair_population_sha256,
            "absolute_pose_owner_count": 1,
            "frame0_residual_owner_count": 1,
            "duplicate_pose_payload_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class PairPopulationEnvelope:
    """Research-only structural receipt joining the five formerly hidden edges."""

    reopened: ReopenedObjectJoin
    pbr2_reference: EncoderOnlyPairReference
    v19c_typed_pairs_reference: EncoderOnlyPairReference
    ir_coverage: IRCoverageReceipt
    compact_program: CompactObligationGeneratorProgram
    pose_ownership: ExclusivePoseOwnership
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        population = self.reopened.pair_population
        if self.pbr2_reference.role != "pbr2_target_window":
            raise PairPopulationEnvelopeError("PBR2 pair reference role differs")
        if self.v19c_typed_pairs_reference.role != "v19c_typed_pair_ids":
            raise PairPopulationEnvelopeError("V19c typed-pair reference role differs")
        if self.pbr2_reference.source_pair_ids != population.domain_index(PairDomain.PBR).local_to_source_pair_ids:
            raise PairPopulationEnvelopeError("PBR2 reference differs from PairPopulation PBR index")
        ir_sources = set(population.domain_index(PairDomain.IR).local_to_source_pair_ids)
        v10_sources = set(population.domain_index(PairDomain.V10).local_to_source_pair_ids)
        if not set(self.v19c_typed_pairs_reference.source_pair_ids).issubset(ir_sources & v10_sources):
            raise PairPopulationEnvelopeError("V19c typed pair IDs escape the IR/V10 source population")
        identities = self.reopened.identity_dict()
        if (
            self.ir_coverage.pair_population_sha256 != population.population_sha256
            or self.ir_coverage.obligation_ir_sha256 != identities["obligation_ir_sha256"]
            or self.ir_coverage.predictor_program_sha256 != identities["predictor_program_sha256"]
            or self.ir_coverage.pbr2_packet_sha256 != self.pbr2_reference.artifact_sha256
            or self.ir_coverage.pbr2_target_semantic_sha256 != self.pbr2_reference.semantic_sha256
        ):
            raise PairPopulationEnvelopeError("IR coverage foreign keys differ from reopened objects")
        if (
            self.compact_program.pair_population_sha256 != population.population_sha256
            or self.compact_program.explicit_preimage_result_sha256 != identities["explicit_preimage_result_sha256"]
            or self.compact_program.generated_y0_identity_sha256 != identities["encoder_only_dense_y0_identity_sha256"]
            or self.compact_program.generated_y1_identity_sha256 != identities["encoder_only_dense_y1_identity_sha256"]
        ):
            raise PairPopulationEnvelopeError("compact program foreign keys differ from encoder-only admission")
        generative_section = self.compact_program.section(CountedSectionRole.GENERATIVE_CORRECTION)
        if (
            self.ir_coverage.compact_program_sha256 != self.compact_program.program_sha256
            or self.ir_coverage.candidate_program_sha256 != generative_section.payload_sha256
            or self.ir_coverage.candidate_program_bytes != generative_section.byte_length
        ):
            raise PairPopulationEnvelopeError("IR coverage is not bound to the counted G section")
        expected_receiver_state_sha256 = canonical_sha256(
            {
                "schema": "tac.compact_generator_receiver_state.v1",
                "derived_object_identities": identities,
                "decoder_contract_id": self.compact_program.decoder_contract_id,
                "receiver_binding": self.compact_program.receiver_binding.as_dict(),
                "program_sha256": self.compact_program.program_sha256,
                "section_manifest": [section.as_dict() for section in self.compact_program.section_manifest],
                "section_causality": [item.as_dict() for item in self.compact_program.section_causality],
            }
        )
        if self.compact_program.receiver_state_sha256 != expected_receiver_state_sha256:
            raise PairPopulationEnvelopeError("compact receiver state differs from reopened objects")
        if self.pose_ownership != ExclusivePoseOwnership.bind(self.reopened, self.compact_program):
            raise PairPopulationEnvelopeError("pose ownership is not exclusively derived from V9 and compact program")
        if (
            type(self.research_only) is not bool
            or not self.research_only
            or type(self.score_claim) is not bool
            or self.score_claim
            or type(self.promotion_eligible) is not bool
            or self.promotion_eligible
        ):
            raise PairPopulationEnvelopeError("pair-population join cannot claim score or promotion authority")

    @property
    def envelope_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": JOIN_ENVELOPE_SCHEMA,
            "pair_population": self.reopened.pair_population.as_dict(),
            "derived_object_identities": self.reopened.identity_dict(),
            "encoder_only_pair_references": {
                "pbr2_target_window": self.pbr2_reference.as_dict(),
                "v19c_typed_pair_ids": self.v19c_typed_pairs_reference.as_dict(),
            },
            "encoder_only_ir_coverage": self.ir_coverage.as_dict(self.reopened.pair_population),
            "counted_compact_obligation_generator": self.compact_program.as_dict(),
            "exclusive_pose_ownership": self.pose_ownership.as_dict(),
            "payload_firewall": {
                "reference_receiver_declares_untyped_dense_y_serialized": False,
                "reference_receiver_declares_untyped_camera_preimages_serialized": False,
                "reference_receiver_declares_pbr_teacher_serialized": False,
                "reference_receiver_declares_gt_or_target_semantics_serialized": False,
                "reference_receiver_declares_scorer_artifacts_serialized": False,
                "reference_receiver_declares_obligation_ir_serialized": False,
                "reference_receiver_declares_oracle_evidence_serialized": False,
                "hash_only_dense_packet_substitution_allowed": False,
                "untyped_sections_allowed_for_candidate": False,
                "frame1_preimage_typed_parser_closed": False,
                "frame0_pose_residual_typed_parser_closed": False,
                "declarations_independently_verified_by_production_parsers": False,
                "standalone_archive_runtime_closure_bound": False,
            },
            "candidate_payload_eligible": False,
            "candidate_payload_blocker": (
                "production typed frame1/frame0 grammars, standalone runtime closure, and exact archive replay owed"
            ),
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {"schema": JOIN_ENVELOPE_WIRE_SCHEMA, "body": body, "body_sha256": canonical_sha256(body)}
        )

    def validate_serialized(self, payload: bytes) -> None:
        body = _decode_envelope(payload, schema=JOIN_ENVELOPE_WIRE_SCHEMA, label="pair-population join envelope")
        if body != self.as_dict():
            raise PairPopulationEnvelopeError("serialized join differs from live reopened objects")


__all__ = [
    "COMPACT_PROGRAM_SCHEMA",
    "IR_COVERAGE_SCHEMA",
    "PAIR_DOMAIN_ORDER",
    "PAIR_POPULATION_SCHEMA",
    "CompactGeneratorBatch",
    "CompactGeneratorDecode",
    "CompactObligationGeneratorProgram",
    "CompactProgramRole",
    "CountedArtifactClass",
    "CountedPayloadLineage",
    "CountedPayloadProvenance",
    "CountedProgramSection",
    "CountedSectionRole",
    "DecodedCandidateSemantics",
    "EncoderOnlyPairReference",
    "ExclusivePoseOwnership",
    "Frame0PoseMode",
    "IRCoveragePolicy",
    "IRCoverageReceipt",
    "PairCoordinateRow",
    "PairDomain",
    "PairDomainIndex",
    "PairPopulation",
    "PairPopulationEnvelope",
    "PairPopulationEnvelopeError",
    "ReceiverSourceBinding",
    "ReopenedObjectJoin",
    "RoleCounterfactualProgram",
    "SectionCausalityReceipt",
    "SparseDebtOwner",
    "SparseObligationOwnership",
    "derive_ir_coverage",
    "reopen_pbr2_pair_reference",
    "reopen_typed_pair_reference",
    "validate_counted_program_sections",
    "validate_reverse_causal_counted_program_sections",
]
