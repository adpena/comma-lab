# SPDX-License-Identifier: MIT
"""Canonical scientific state for the original V9-to-V10 witness codec.

The project has many useful packet grammars, action receipts, and optimizer
states.  None of those is the scientific object being compressed.  This module
defines that object and keeps three identities deliberately separate:

``CoupledWitnessState``
    The source-derived candidate task-space sufficient statistic: V9 geometry
    and transport, V10 cell/preimage obligations, the joint frame-0 pose fibre,
    and (only when needed) the terminal quotient.  Global minimality is not
    claimed.

``WitnessCompileConfig``
    Receiver, precision, coder, and container choices.  These choices may be
    raced without pretending that the scientific state changed.

``CodecObjectManifest``
    The C0 metadata foreign-key join between state and compile policy.  Exact
    archive, decode, and score identities require downstream edge receipts.

All serialized forms are canonical JSON envelopes with an internal SHA-256.
This is an executable identity contract, not a score or promotion claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

COUPLED_WITNESS_STATE_SCHEMA = "tac.coupled_witness_state.v1"
COUPLED_WITNESS_STATE_ENVELOPE_SCHEMA = "tac.coupled_witness_state.envelope.v1"
FROZEN_SPACE_IDENTITY_SCHEMA = "tac.frozen_contest_space_identity.v1"
CONTENT_ADDRESS_SCHEMA = "tac.content_address.v1"
SCIENTIFIC_STREAM_SCHEMA = "tac.coupled_witness_stream.v1"
SCIENTIFIC_STREAM_DEPENDENCY_SCHEMA = "tac.coupled_witness_stream_dependency.v1"
STATE_PATCH_SCHEMA = "tac.coupled_witness_state_patch.v1"
STATE_PATCH_ENVELOPE_SCHEMA = "tac.coupled_witness_state_patch.envelope.v1"
STATE_TRANSITION_RECEIPT_SCHEMA = "tac.coupled_witness_transition_receipt.v1"
STATE_TRANSITION_RECEIPT_ENVELOPE_SCHEMA = (
    "tac.coupled_witness_transition_receipt.envelope.v1"
)
COMPILE_STREAM_POLICY_SCHEMA = "tac.witness_compile_stream_policy.v1"
WITNESS_COMPILE_CONFIG_SCHEMA = "tac.witness_compile_config.v1"
WITNESS_COMPILE_CONFIG_ENVELOPE_SCHEMA = "tac.witness_compile_config.envelope.v1"
CODEC_OBJECT_MANIFEST_SCHEMA = "tac.codec_object.v1"
CODEC_OBJECT_ENVELOPE_SCHEMA = "tac.codec_object.envelope.v1"

SOURCE_DERIVED_LINEAGE = "source-video-derived-our-original-build"
DECODER_PAYLOAD_POLICY = "counted-source-derived-statistics-only-no-scorer-no-gt-table"


class CoupledWitnessStateError(ValueError):
    """Fail-closed malformed state, transition, compile, or object identity."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted JSON spelling for an identity-bearing value."""

    def require_string_mapping_keys(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CoupledWitnessStateError(
                        f"canonical JSON mapping key at {path} must be a string"
                    )
                require_string_mapping_keys(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                require_string_mapping_keys(child, f"{path}[{index}]")

    require_string_mapping_keys(value, "root")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CoupledWitnessStateError("value is not canonical-JSON encodable") from exc


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise CoupledWitnessStateError("SHA-256 payload must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoupledWitnessStateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_canonical_json(payload: bytes) -> Any:
    """Decode canonical JSON while rejecting duplicate keys and alternate bytes."""

    if not isinstance(payload, bytes):
        raise CoupledWitnessStateError("serialized payload must be exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoupledWitnessStateError("serialized payload is not strict ASCII JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise CoupledWitnessStateError("serialized JSON is not canonical")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise CoupledWitnessStateError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CoupledWitnessStateError(f"{name} fields differ: missing={missing}, extra={extra}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or value.strip() != value or not value:
        raise CoupledWitnessStateError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _text(value, name)
    if text != text.lower():
        raise CoupledWitnessStateError(f"{name} must use lowercase SHA-256 hex")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CoupledWitnessStateError(f"{name} must be a SHA-256 hex string")
    return text


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoupledWitnessStateError(f"{name} must be a non-negative exact integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise CoupledWitnessStateError(f"{name} must be positive")
    return result


@dataclass(frozen=True, order=True)
class ContentAddress:
    """Path-independent identity of exact bytes consumed by the codec."""

    artifact_id: str
    artifact_schema: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.artifact_schema, "artifact_schema")
        _sha256(self.sha256, "sha256")
        _nonnegative_int(self.byte_length, "byte_length")

    @classmethod
    def from_payload(
        cls,
        *,
        artifact_id: str,
        artifact_schema: str,
        payload: bytes,
    ) -> ContentAddress:
        if not isinstance(payload, bytes):
            raise CoupledWitnessStateError("content payload must be exact bytes")
        return cls(artifact_id, artifact_schema, sha256_bytes(payload), len(payload))

    def verify_payload(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise CoupledWitnessStateError("content payload must be exact bytes")
        if len(payload) != self.byte_length or sha256_bytes(payload) != self.sha256:
            raise CoupledWitnessStateError(f"payload differs from content address {self.artifact_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTENT_ADDRESS_SCHEMA,
            "artifact_id": self.artifact_id,
            "artifact_schema": self.artifact_schema,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContentAddress:
        expected = {"schema", "artifact_id", "artifact_schema", "sha256", "byte_length"}
        _exact_keys(value, expected, "content address")
        if value["schema"] != CONTENT_ADDRESS_SCHEMA:
            raise CoupledWitnessStateError("content-address schema differs")
        return cls(
            artifact_id=value["artifact_id"],
            artifact_schema=value["artifact_schema"],
            sha256=value["sha256"],
            byte_length=value["byte_length"],
        )


@dataclass(frozen=True)
class FrozenSpaceIdentity:
    """Frozen source/evaluator cells whose candidate statistic will be encoded."""

    source_video: ContentAddress
    evaluator_artifacts: tuple[ContentAddress, ...]
    pair_count: int
    pair_order_id: str
    pair_order_sha256: str
    scorer_height: int
    scorer_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_video, ContentAddress):
            raise CoupledWitnessStateError("source_video must be a ContentAddress")
        if not isinstance(self.evaluator_artifacts, tuple) or not self.evaluator_artifacts:
            raise CoupledWitnessStateError("evaluator_artifacts must be a non-empty tuple")
        if any(not isinstance(item, ContentAddress) for item in self.evaluator_artifacts):
            raise CoupledWitnessStateError("evaluator_artifacts contain an invalid address")
        artifact_ids = [item.artifact_id for item in self.evaluator_artifacts]
        if artifact_ids != sorted(artifact_ids) or len(set(artifact_ids)) != len(artifact_ids):
            raise CoupledWitnessStateError("evaluator_artifacts must be uniquely sorted by artifact_id")
        pair_count = _positive_int(self.pair_count, "pair_count")
        _text(self.pair_order_id, "pair_order_id")
        pair_order_sha = _sha256(self.pair_order_sha256, "pair_order_sha256")
        expected_pair_order_sha = canonical_sha256(list(range(pair_count)))
        if pair_order_sha != expected_pair_order_sha:
            raise CoupledWitnessStateError("pair_order_sha256 must bind canonical contiguous pair order")
        _positive_int(self.scorer_height, "scorer_height")
        _positive_int(self.scorer_width, "scorer_width")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FROZEN_SPACE_IDENTITY_SCHEMA,
            "source_video": self.source_video.as_dict(),
            "evaluator_artifacts": [item.as_dict() for item in self.evaluator_artifacts],
            "pair_count": self.pair_count,
            "pair_order_id": self.pair_order_id,
            "pair_order_sha256": self.pair_order_sha256,
            "scorer_height": self.scorer_height,
            "scorer_width": self.scorer_width,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenSpaceIdentity:
        expected = {
            "schema",
            "source_video",
            "evaluator_artifacts",
            "pair_count",
            "pair_order_id",
            "pair_order_sha256",
            "scorer_height",
            "scorer_width",
        }
        _exact_keys(value, expected, "frozen-space identity")
        if value["schema"] != FROZEN_SPACE_IDENTITY_SCHEMA:
            raise CoupledWitnessStateError("frozen-space identity schema differs")
        artifacts = value["evaluator_artifacts"]
        if not isinstance(artifacts, list):
            raise CoupledWitnessStateError("evaluator_artifacts must be an array")
        return cls(
            source_video=ContentAddress.from_dict(value["source_video"]),
            evaluator_artifacts=tuple(ContentAddress.from_dict(item) for item in artifacts),
            pair_count=value["pair_count"],
            pair_order_id=value["pair_order_id"],
            pair_order_sha256=value["pair_order_sha256"],
            scorer_height=value["scorer_height"],
            scorer_width=value["scorer_width"],
        )


class ScientificStreamRole(StrEnum):
    """Scientific coordinates; these are not assumptions of byte independence."""

    TOPOLOGY_WORLDSHEET = "topology_worldsheet"
    BULK_BOUNDARY = "bulk_boundary"
    LANE_CHART = "lane_chart"
    MOVABLE_MYCAR = "movable_mycar"
    CELL_VALUE_PREIMAGE = "cell_value_preimage"
    POSE_TRANSPORT_FRAME0 = "pose_transport_frame0"
    IRREDUCIBLE_QUOTIENT = "irreducible_quotient"


SCIENTIFIC_STREAM_ORDER: tuple[ScientificStreamRole, ...] = tuple(ScientificStreamRole)
_STREAM_INDEX = {role: index for index, role in enumerate(SCIENTIFIC_STREAM_ORDER)}
_STREAM_DEPENDENCIES: dict[ScientificStreamRole, frozenset[ScientificStreamRole]] = {
    ScientificStreamRole.TOPOLOGY_WORLDSHEET: frozenset(),
    ScientificStreamRole.BULK_BOUNDARY: frozenset({ScientificStreamRole.TOPOLOGY_WORLDSHEET}),
    ScientificStreamRole.LANE_CHART: frozenset({ScientificStreamRole.TOPOLOGY_WORLDSHEET}),
    ScientificStreamRole.MOVABLE_MYCAR: frozenset({ScientificStreamRole.TOPOLOGY_WORLDSHEET}),
    ScientificStreamRole.CELL_VALUE_PREIMAGE: frozenset(
        {
            ScientificStreamRole.TOPOLOGY_WORLDSHEET,
            ScientificStreamRole.BULK_BOUNDARY,
            ScientificStreamRole.LANE_CHART,
            ScientificStreamRole.MOVABLE_MYCAR,
        }
    ),
    ScientificStreamRole.POSE_TRANSPORT_FRAME0: frozenset(
        {
            ScientificStreamRole.TOPOLOGY_WORLDSHEET,
            ScientificStreamRole.CELL_VALUE_PREIMAGE,
        }
    ),
    ScientificStreamRole.IRREDUCIBLE_QUOTIENT: frozenset(
        role for role in SCIENTIFIC_STREAM_ORDER if role is not ScientificStreamRole.IRREDUCIBLE_QUOTIENT
    ),
}


@dataclass(frozen=True)
class ScientificStreamDependency:
    """Exact upstream scientific payload consumed to derive one stream."""

    role: ScientificStreamRole
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ScientificStreamRole):
            raise CoupledWitnessStateError("scientific stream dependency role is invalid")
        _sha256(self.content_sha256, "dependency content_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCIENTIFIC_STREAM_DEPENDENCY_SCHEMA,
            "role": self.role.value,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScientificStreamDependency:
        _exact_keys(
            value,
            {"schema", "role", "content_sha256"},
            "scientific stream dependency",
        )
        if value["schema"] != SCIENTIFIC_STREAM_DEPENDENCY_SCHEMA:
            raise CoupledWitnessStateError("scientific-stream-dependency schema differs")
        try:
            role = ScientificStreamRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise CoupledWitnessStateError("scientific-stream-dependency role differs") from exc
        return cls(role=role, content_sha256=value["content_sha256"])


@dataclass(frozen=True)
class ScientificStream:
    """One source-derived coordinate block in the coupled scientific state."""

    role: ScientificStreamRole
    content: ContentAddress
    provenance_manifest: ContentAddress
    dependencies: tuple[ScientificStreamDependency, ...]
    lineage: str = SOURCE_DERIVED_LINEAGE
    borrowed_candidate_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.role, ScientificStreamRole):
            raise CoupledWitnessStateError("scientific stream role is invalid")
        if not isinstance(self.content, ContentAddress):
            raise CoupledWitnessStateError("scientific stream content is invalid")
        if not isinstance(self.provenance_manifest, ContentAddress):
            raise CoupledWitnessStateError("scientific stream provenance manifest is invalid")
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(item, ScientificStreamDependency) for item in self.dependencies
        ):
            raise CoupledWitnessStateError("scientific stream dependencies are invalid")
        expected_roles = sorted(_STREAM_DEPENDENCIES[self.role], key=_STREAM_INDEX.__getitem__)
        observed_roles = [item.role for item in self.dependencies]
        if observed_roles != expected_roles:
            raise CoupledWitnessStateError(f"scientific stream {self.role.value} dependency roles differ")
        if self.lineage != SOURCE_DERIVED_LINEAGE:
            raise CoupledWitnessStateError("scientific stream must use the original source-derived lineage")
        if _nonnegative_int(self.borrowed_candidate_bytes, "borrowed_candidate_bytes") != 0:
            raise CoupledWitnessStateError("borrowed candidate bytes are forbidden")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCIENTIFIC_STREAM_SCHEMA,
            "role": self.role.value,
            "content": self.content.as_dict(),
            "provenance_manifest": self.provenance_manifest.as_dict(),
            "dependencies": [item.as_dict() for item in self.dependencies],
            "lineage": self.lineage,
            "borrowed_candidate_bytes": self.borrowed_candidate_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScientificStream:
        expected = {
            "schema",
            "role",
            "content",
            "provenance_manifest",
            "dependencies",
            "lineage",
            "borrowed_candidate_bytes",
        }
        _exact_keys(value, expected, "scientific stream")
        if value["schema"] != SCIENTIFIC_STREAM_SCHEMA:
            raise CoupledWitnessStateError("scientific-stream schema differs")
        try:
            role = ScientificStreamRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise CoupledWitnessStateError("scientific-stream role differs") from exc
        dependencies = value["dependencies"]
        if not isinstance(dependencies, list):
            raise CoupledWitnessStateError("scientific-stream dependencies must be an array")
        return cls(
            role=role,
            content=ContentAddress.from_dict(value["content"]),
            provenance_manifest=ContentAddress.from_dict(value["provenance_manifest"]),
            dependencies=tuple(ScientificStreamDependency.from_dict(item) for item in dependencies),
            lineage=value["lineage"],
            borrowed_candidate_bytes=value["borrowed_candidate_bytes"],
        )


@dataclass(frozen=True)
class CoupledWitnessState:
    """Content-addressed V9 world state plus V10 realization obligations."""

    frozen_space: FrozenSpaceIdentity
    generation_seed: int
    generation_rng_id: str
    parent_state_sha256: str | None
    transition_index: int
    streams: tuple[ScientificStream, ...]
    schema: str = COUPLED_WITNESS_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COUPLED_WITNESS_STATE_SCHEMA:
            raise CoupledWitnessStateError("coupled-witness-state schema differs")
        if not isinstance(self.frozen_space, FrozenSpaceIdentity):
            raise CoupledWitnessStateError("frozen_space is invalid")
        _nonnegative_int(self.generation_seed, "generation_seed")
        _text(self.generation_rng_id, "generation_rng_id")
        index = _nonnegative_int(self.transition_index, "transition_index")
        _sha256(self.parent_state_sha256, "parent_state_sha256", optional=True)
        if index == 0 and self.parent_state_sha256 is not None:
            raise CoupledWitnessStateError("root state cannot have a parent")
        if index > 0 and self.parent_state_sha256 is None:
            raise CoupledWitnessStateError("non-root state requires a parent")
        if not isinstance(self.streams, tuple):
            raise CoupledWitnessStateError("streams must be a tuple")
        if any(not isinstance(stream, ScientificStream) for stream in self.streams):
            raise CoupledWitnessStateError("streams contain an invalid entry")
        roles = [stream.role for stream in self.streams]
        if roles != sorted(roles, key=_STREAM_INDEX.__getitem__):
            raise CoupledWitnessStateError("streams must use canonical scientific order")
        if len(set(roles)) != len(roles):
            raise CoupledWitnessStateError("scientific stream roles must be unique")
        streams_by_role = {stream.role: stream for stream in self.streams}
        present = set(streams_by_role)
        for role, stream in streams_by_role.items():
            missing = _STREAM_DEPENDENCIES[role] - present
            if missing:
                names = sorted(item.value for item in missing)
                raise CoupledWitnessStateError(f"scientific stream {role.value} is missing dependencies {names}")
            for dependency in stream.dependencies:
                if dependency.content_sha256 != streams_by_role[dependency.role].content.sha256:
                    raise CoupledWitnessStateError(
                        f"scientific stream {role.value} has a stale dependency on {dependency.role.value}"
                    )

    @classmethod
    def empty(
        cls,
        frozen_space: FrozenSpaceIdentity,
        *,
        generation_seed: int,
        generation_rng_id: str,
    ) -> CoupledWitnessState:
        return cls(frozen_space, generation_seed, generation_rng_id, None, 0, ())

    @property
    def state_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @property
    def present_roles(self) -> tuple[ScientificStreamRole, ...]:
        return tuple(stream.role for stream in self.streams)

    @property
    def next_missing_role(self) -> ScientificStreamRole | None:
        present = set(self.present_roles)
        return next((role for role in SCIENTIFIC_STREAM_ORDER if role not in present), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frozen_space": self.frozen_space.as_dict(),
            "generation_seed": self.generation_seed,
            "generation_rng_id": self.generation_rng_id,
            "parent_state_sha256": self.parent_state_sha256,
            "transition_index": self.transition_index,
            "streams": [stream.as_dict() for stream in self.streams],
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": COUPLED_WITNESS_STATE_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoupledWitnessState:
        expected = {
            "schema",
            "frozen_space",
            "generation_seed",
            "generation_rng_id",
            "parent_state_sha256",
            "transition_index",
            "streams",
        }
        _exact_keys(value, expected, "coupled witness state")
        streams = value["streams"]
        if not isinstance(streams, list):
            raise CoupledWitnessStateError("streams must be an array")
        return cls(
            schema=value["schema"],
            frozen_space=FrozenSpaceIdentity.from_dict(value["frozen_space"]),
            generation_seed=value["generation_seed"],
            generation_rng_id=value["generation_rng_id"],
            parent_state_sha256=value["parent_state_sha256"],
            transition_index=value["transition_index"],
            streams=tuple(ScientificStream.from_dict(item) for item in streams),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> CoupledWitnessState:
        envelope = decode_canonical_json(payload)
        expected = {"schema", "body", "body_sha256"}
        _exact_keys(envelope, expected, "coupled witness state envelope")
        if envelope["schema"] != COUPLED_WITNESS_STATE_ENVELOPE_SCHEMA:
            raise CoupledWitnessStateError("coupled-witness-state envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise CoupledWitnessStateError("coupled-witness-state body hash differs")
        state = cls.from_dict(envelope["body"])
        if state.state_sha256 != envelope["body_sha256"]:
            raise CoupledWitnessStateError("coupled-witness-state identity differs")
        return state


@dataclass(frozen=True)
class StatePatch:
    """Typed state transition; never a free-floating score delta."""

    patch_id: str
    expected_parent_state_sha256: str
    set_streams: tuple[ScientificStream, ...]
    remove_roles: tuple[ScientificStreamRole, ...]
    rationale: str
    provenance_ref: str
    schema: str = STATE_PATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_PATCH_SCHEMA:
            raise CoupledWitnessStateError("state-patch schema differs")
        _text(self.patch_id, "patch_id")
        _sha256(self.expected_parent_state_sha256, "expected_parent_state_sha256")
        _text(self.rationale, "rationale")
        _text(self.provenance_ref, "provenance_ref")
        if not isinstance(self.set_streams, tuple) or not isinstance(self.remove_roles, tuple):
            raise CoupledWitnessStateError("patch stream fields must be tuples")
        if any(not isinstance(stream, ScientificStream) for stream in self.set_streams):
            raise CoupledWitnessStateError("set_streams contain an invalid entry")
        if any(not isinstance(role, ScientificStreamRole) for role in self.remove_roles):
            raise CoupledWitnessStateError("remove_roles contain an invalid entry")
        set_roles = [stream.role for stream in self.set_streams]
        if set_roles != sorted(set_roles, key=_STREAM_INDEX.__getitem__) or len(set(set_roles)) != len(set_roles):
            raise CoupledWitnessStateError("set_streams must be unique and canonically ordered")
        remove_roles = list(self.remove_roles)
        if remove_roles != sorted(remove_roles, key=_STREAM_INDEX.__getitem__) or len(set(remove_roles)) != len(
            remove_roles
        ):
            raise CoupledWitnessStateError("remove_roles must be unique and canonically ordered")
        if set(set_roles) & set(remove_roles):
            raise CoupledWitnessStateError("one patch cannot set and remove the same stream")
        if not set_roles and not remove_roles:
            raise CoupledWitnessStateError("state patch must change at least one stream")

    @property
    def patch_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "patch_id": self.patch_id,
            "expected_parent_state_sha256": self.expected_parent_state_sha256,
            "set_streams": [stream.as_dict() for stream in self.set_streams],
            "remove_roles": [role.value for role in self.remove_roles],
            "rationale": self.rationale,
            "provenance_ref": self.provenance_ref,
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": STATE_PATCH_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StatePatch:
        expected = {
            "schema",
            "patch_id",
            "expected_parent_state_sha256",
            "set_streams",
            "remove_roles",
            "rationale",
            "provenance_ref",
        }
        _exact_keys(value, expected, "state patch")
        set_streams = value["set_streams"]
        remove_roles = value["remove_roles"]
        if not isinstance(set_streams, list) or not isinstance(remove_roles, list):
            raise CoupledWitnessStateError("state-patch stream fields must be arrays")
        try:
            parsed_remove_roles = tuple(ScientificStreamRole(role) for role in remove_roles)
        except (TypeError, ValueError) as exc:
            raise CoupledWitnessStateError("state-patch remove role differs") from exc
        return cls(
            schema=value["schema"],
            patch_id=value["patch_id"],
            expected_parent_state_sha256=value["expected_parent_state_sha256"],
            set_streams=tuple(ScientificStream.from_dict(item) for item in set_streams),
            remove_roles=parsed_remove_roles,
            rationale=value["rationale"],
            provenance_ref=value["provenance_ref"],
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> StatePatch:
        envelope = decode_canonical_json(payload)
        expected = {"schema", "body", "body_sha256"}
        _exact_keys(envelope, expected, "state patch envelope")
        if envelope["schema"] != STATE_PATCH_ENVELOPE_SCHEMA:
            raise CoupledWitnessStateError("state-patch envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise CoupledWitnessStateError("state-patch body hash differs")
        result = cls.from_dict(envelope["body"])
        if result.patch_sha256 != envelope["body_sha256"]:
            raise CoupledWitnessStateError("state-patch identity differs")
        return result


@dataclass(frozen=True)
class StateTransitionReceipt:
    patch_id: str
    patch_sha256: str
    from_state_sha256: str
    to_state_sha256: str
    transition_index: int
    changed_roles: tuple[ScientificStreamRole, ...]
    schema: str = STATE_TRANSITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_TRANSITION_RECEIPT_SCHEMA:
            raise CoupledWitnessStateError("transition-receipt schema differs")
        _text(self.patch_id, "patch_id")
        _sha256(self.patch_sha256, "patch_sha256")
        from_sha = _sha256(self.from_state_sha256, "from_state_sha256")
        to_sha = _sha256(self.to_state_sha256, "to_state_sha256")
        if from_sha == to_sha:
            raise CoupledWitnessStateError("state transition must change state identity")
        _positive_int(self.transition_index, "transition_index")
        if not isinstance(self.changed_roles, tuple) or not self.changed_roles:
            raise CoupledWitnessStateError("transition receipt requires changed roles")
        roles = list(self.changed_roles)
        if roles != sorted(roles, key=_STREAM_INDEX.__getitem__) or len(set(roles)) != len(roles):
            raise CoupledWitnessStateError("changed roles must be unique and canonically ordered")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "patch_id": self.patch_id,
            "patch_sha256": self.patch_sha256,
            "from_state_sha256": self.from_state_sha256,
            "to_state_sha256": self.to_state_sha256,
            "transition_index": self.transition_index,
            "changed_roles": [role.value for role in self.changed_roles],
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": STATE_TRANSITION_RECEIPT_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateTransitionReceipt:
        expected = {
            "schema",
            "patch_id",
            "patch_sha256",
            "from_state_sha256",
            "to_state_sha256",
            "transition_index",
            "changed_roles",
        }
        _exact_keys(value, expected, "state transition receipt")
        changed_roles = value["changed_roles"]
        if not isinstance(changed_roles, list):
            raise CoupledWitnessStateError("transition changed_roles must be an array")
        try:
            parsed_roles = tuple(ScientificStreamRole(role) for role in changed_roles)
        except (TypeError, ValueError) as exc:
            raise CoupledWitnessStateError("transition changed role differs") from exc
        return cls(
            schema=value["schema"],
            patch_id=value["patch_id"],
            patch_sha256=value["patch_sha256"],
            from_state_sha256=value["from_state_sha256"],
            to_state_sha256=value["to_state_sha256"],
            transition_index=value["transition_index"],
            changed_roles=parsed_roles,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> StateTransitionReceipt:
        envelope = decode_canonical_json(payload)
        expected = {"schema", "body", "body_sha256"}
        _exact_keys(envelope, expected, "state transition receipt envelope")
        if envelope["schema"] != STATE_TRANSITION_RECEIPT_ENVELOPE_SCHEMA:
            raise CoupledWitnessStateError("transition-receipt envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise CoupledWitnessStateError("transition-receipt body hash differs")
        result = cls.from_dict(envelope["body"])
        if result.receipt_sha256 != envelope["body_sha256"]:
            raise CoupledWitnessStateError("transition-receipt identity differs")
        return result

    def validate_against(
        self,
        parent: CoupledWitnessState,
        patch: StatePatch,
        child: CoupledWitnessState,
    ) -> None:
        """Replay and verify every edge identity and transition invariant."""

        if not isinstance(parent, CoupledWitnessState) or not isinstance(
            child, CoupledWitnessState
        ):
            raise CoupledWitnessStateError("transition endpoint types differ")
        if not isinstance(patch, StatePatch):
            raise CoupledWitnessStateError("transition patch type differs")
        if child.frozen_space != parent.frozen_space:
            raise CoupledWitnessStateError("transition changed frozen-space identity")
        if (
            child.generation_seed != parent.generation_seed
            or child.generation_rng_id != parent.generation_rng_id
        ):
            raise CoupledWitnessStateError("transition changed generation identity")
        if child.parent_state_sha256 != parent.state_sha256:
            raise CoupledWitnessStateError("child parent-state foreign key differs")
        if child.transition_index != parent.transition_index + 1:
            raise CoupledWitnessStateError("child transition index differs")
        expected_child, expected_receipt = apply_state_patch(parent, patch)
        if child != expected_child:
            raise CoupledWitnessStateError("child state differs from deterministic patch replay")
        if self != expected_receipt:
            raise CoupledWitnessStateError(
                "transition receipt foreign keys or changed-role semantics differ"
            )


def apply_state_patch(
    state: CoupledWitnessState,
    patch: StatePatch,
) -> tuple[CoupledWitnessState, StateTransitionReceipt]:
    """Apply one identity-bound transition and return its exact edge receipt."""

    if not isinstance(state, CoupledWitnessState) or not isinstance(patch, StatePatch):
        raise CoupledWitnessStateError("state and patch must use the typed IR")
    if patch.expected_parent_state_sha256 != state.state_sha256:
        raise CoupledWitnessStateError("patch parent state identity differs")
    by_role = {stream.role: stream for stream in state.streams}
    for role in patch.remove_roles:
        if role not in by_role:
            raise CoupledWitnessStateError(f"cannot remove absent stream {role.value}")
        del by_role[role]
    for stream in patch.set_streams:
        prior = by_role.get(stream.role)
        if prior == stream:
            raise CoupledWitnessStateError(f"patch does not change stream {stream.role.value}")
        by_role[stream.role] = stream
    streams = tuple(by_role[role] for role in SCIENTIFIC_STREAM_ORDER if role in by_role)
    child = CoupledWitnessState(
        frozen_space=state.frozen_space,
        generation_seed=state.generation_seed,
        generation_rng_id=state.generation_rng_id,
        parent_state_sha256=state.state_sha256,
        transition_index=state.transition_index + 1,
        streams=streams,
    )
    changed = tuple(
        role
        for role in SCIENTIFIC_STREAM_ORDER
        if role in set(patch.remove_roles) | {stream.role for stream in patch.set_streams}
    )
    receipt = StateTransitionReceipt(
        patch_id=patch.patch_id,
        patch_sha256=patch.patch_sha256,
        from_state_sha256=state.state_sha256,
        to_state_sha256=child.state_sha256,
        transition_index=child.transition_index,
        changed_roles=changed,
    )
    return child, receipt


@dataclass(frozen=True)
class CompileStreamPolicy:
    """Physical realization policy for one scientific stream."""

    role: ScientificStreamRole
    coder_id: str
    precision_id: str
    section_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ScientificStreamRole):
            raise CoupledWitnessStateError("compile stream role is invalid")
        _text(self.coder_id, "coder_id")
        _text(self.precision_id, "precision_id")
        _text(self.section_id, "section_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPILE_STREAM_POLICY_SCHEMA,
            "role": self.role.value,
            "coder_id": self.coder_id,
            "precision_id": self.precision_id,
            "section_id": self.section_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompileStreamPolicy:
        expected = {"schema", "role", "coder_id", "precision_id", "section_id"}
        _exact_keys(value, expected, "compile stream policy")
        if value["schema"] != COMPILE_STREAM_POLICY_SCHEMA:
            raise CoupledWitnessStateError("compile-stream-policy schema differs")
        try:
            role = ScientificStreamRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise CoupledWitnessStateError("compile stream role differs") from exc
        return cls(role, value["coder_id"], value["precision_id"], value["section_id"])


@dataclass(frozen=True)
class WitnessCompileConfig:
    """Receiver and byte-layout choices, intentionally outside scientific state."""

    container_id: str
    receiver_contract_id: str
    receiver_artifacts: tuple[ContentAddress, ...]
    r_chain_id: str
    tie_policy_id: str
    camera_height: int
    camera_width: int
    scorer_height: int
    scorer_width: int
    decoder_seed: int
    stream_policies: tuple[CompileStreamPolicy, ...]
    decoder_payload_policy: str = DECODER_PAYLOAD_POLICY
    schema: str = WITNESS_COMPILE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WITNESS_COMPILE_CONFIG_SCHEMA:
            raise CoupledWitnessStateError("witness-compile-config schema differs")
        _text(self.container_id, "container_id")
        _text(self.receiver_contract_id, "receiver_contract_id")
        _text(self.r_chain_id, "r_chain_id")
        _text(self.tie_policy_id, "tie_policy_id")
        _positive_int(self.camera_height, "camera_height")
        _positive_int(self.camera_width, "camera_width")
        _positive_int(self.scorer_height, "scorer_height")
        _positive_int(self.scorer_width, "scorer_width")
        _nonnegative_int(self.decoder_seed, "decoder_seed")
        if self.decoder_payload_policy != DECODER_PAYLOAD_POLICY:
            raise CoupledWitnessStateError("decoder payload policy weakens originality boundary")
        if not isinstance(self.receiver_artifacts, tuple) or not self.receiver_artifacts:
            raise CoupledWitnessStateError("receiver_artifacts must be a non-empty tuple")
        if any(not isinstance(item, ContentAddress) for item in self.receiver_artifacts):
            raise CoupledWitnessStateError("receiver_artifacts contain an invalid address")
        receiver_ids = [item.artifact_id for item in self.receiver_artifacts]
        if receiver_ids != sorted(receiver_ids) or len(set(receiver_ids)) != len(receiver_ids):
            raise CoupledWitnessStateError("receiver_artifacts must be uniquely sorted by artifact_id")
        if not isinstance(self.stream_policies, tuple):
            raise CoupledWitnessStateError("stream_policies must be a tuple")
        if any(not isinstance(policy, CompileStreamPolicy) for policy in self.stream_policies):
            raise CoupledWitnessStateError("stream_policies contain an invalid entry")
        roles = [policy.role for policy in self.stream_policies]
        if roles != sorted(roles, key=_STREAM_INDEX.__getitem__) or len(set(roles)) != len(roles):
            raise CoupledWitnessStateError("stream_policies must be unique and use canonical scientific order")

    @property
    def receiver_bundle_sha256(self) -> str:
        return canonical_sha256([item.as_dict() for item in self.receiver_artifacts])

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def validate_for_state(self, state: CoupledWitnessState) -> None:
        if not isinstance(state, CoupledWitnessState):
            raise CoupledWitnessStateError("compile target must be CoupledWitnessState")
        if (self.scorer_height, self.scorer_width) != (
            state.frozen_space.scorer_height,
            state.frozen_space.scorer_width,
        ):
            raise CoupledWitnessStateError("compile scorer geometry differs from frozen space")
        if tuple(policy.role for policy in self.stream_policies) != state.present_roles:
            raise CoupledWitnessStateError("compile stream policies must exactly cover the scientific state")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "container_id": self.container_id,
            "receiver_contract_id": self.receiver_contract_id,
            "receiver_artifacts": [item.as_dict() for item in self.receiver_artifacts],
            "receiver_bundle_sha256": self.receiver_bundle_sha256,
            "r_chain_id": self.r_chain_id,
            "tie_policy_id": self.tie_policy_id,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "scorer_height": self.scorer_height,
            "scorer_width": self.scorer_width,
            "decoder_seed": self.decoder_seed,
            "stream_policies": [policy.as_dict() for policy in self.stream_policies],
            "decoder_payload_policy": self.decoder_payload_policy,
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": WITNESS_COMPILE_CONFIG_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WitnessCompileConfig:
        expected = {
            "schema",
            "container_id",
            "receiver_contract_id",
            "receiver_artifacts",
            "receiver_bundle_sha256",
            "r_chain_id",
            "tie_policy_id",
            "camera_height",
            "camera_width",
            "scorer_height",
            "scorer_width",
            "decoder_seed",
            "stream_policies",
            "decoder_payload_policy",
        }
        _exact_keys(value, expected, "witness compile config")
        artifacts = value["receiver_artifacts"]
        policies = value["stream_policies"]
        if not isinstance(artifacts, list) or not isinstance(policies, list):
            raise CoupledWitnessStateError("compile artifact and policy fields must be arrays")
        result = cls(
            schema=value["schema"],
            container_id=value["container_id"],
            receiver_contract_id=value["receiver_contract_id"],
            receiver_artifacts=tuple(ContentAddress.from_dict(item) for item in artifacts),
            r_chain_id=value["r_chain_id"],
            tie_policy_id=value["tie_policy_id"],
            camera_height=value["camera_height"],
            camera_width=value["camera_width"],
            scorer_height=value["scorer_height"],
            scorer_width=value["scorer_width"],
            decoder_seed=value["decoder_seed"],
            stream_policies=tuple(CompileStreamPolicy.from_dict(item) for item in policies),
            decoder_payload_policy=value["decoder_payload_policy"],
        )
        if value["receiver_bundle_sha256"] != result.receiver_bundle_sha256:
            raise CoupledWitnessStateError("receiver bundle hash does not reconcile")
        return result

    @classmethod
    def from_bytes(cls, payload: bytes) -> WitnessCompileConfig:
        envelope = decode_canonical_json(payload)
        expected = {"schema", "body", "body_sha256"}
        _exact_keys(envelope, expected, "witness compile config envelope")
        if envelope["schema"] != WITNESS_COMPILE_CONFIG_ENVELOPE_SCHEMA:
            raise CoupledWitnessStateError("witness-compile-config envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise CoupledWitnessStateError("witness-compile-config body hash differs")
        config = cls.from_dict(envelope["body"])
        if config.config_sha256 != envelope["body_sha256"]:
            raise CoupledWitnessStateError("witness-compile-config identity differs")
        return config


@dataclass(frozen=True)
class CodecObjectManifest:
    """C0 identity scaffold joining scientific state and compile policy.

    Archive, decode, and scorer evidence are intentionally absent from this v1
    object.  They require executable edge receipts with exact foreign keys and
    cannot be created by placing parallel hashes in a manifest.  Until those
    receipts land, this object is metadata-bound and has no score authority.
    """

    state_sha256: str
    frozen_space_sha256: str
    compile_config_sha256: str
    receiver_bundle_sha256: str
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    schema: str = CODEC_OBJECT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CODEC_OBJECT_MANIFEST_SCHEMA:
            raise CoupledWitnessStateError("codec-object schema differs")
        _sha256(self.state_sha256, "state_sha256")
        _sha256(self.frozen_space_sha256, "frozen_space_sha256")
        _sha256(self.compile_config_sha256, "compile_config_sha256")
        _sha256(self.receiver_bundle_sha256, "receiver_bundle_sha256")
        if self.research_only is not True or self.score_claim is not False or self.promotion_eligible is not False:
            raise CoupledWitnessStateError("codec-object v1 is a research binding, never score or promotion authority")

    @classmethod
    def bind(
        cls,
        state: CoupledWitnessState,
        config: WitnessCompileConfig,
    ) -> CodecObjectManifest:
        config.validate_for_state(state)
        return cls(
            state_sha256=state.state_sha256,
            frozen_space_sha256=state.frozen_space.identity_sha256,
            compile_config_sha256=config.config_sha256,
            receiver_bundle_sha256=config.receiver_bundle_sha256,
        )

    @property
    def stage(self) -> str:
        return "C0_IDENTITY_SCAFFOLD"

    @property
    def object_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def validate_against(
        self,
        state: CoupledWitnessState,
        config: WitnessCompileConfig,
    ) -> None:
        config.validate_for_state(state)
        expected = CodecObjectManifest.bind(state, config)
        if self != expected:
            raise CoupledWitnessStateError("codec-object foreign keys differ")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_sha256": self.state_sha256,
            "frozen_space_sha256": self.frozen_space_sha256,
            "compile_config_sha256": self.compile_config_sha256,
            "receiver_bundle_sha256": self.receiver_bundle_sha256,
            "stage": self.stage,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": CODEC_OBJECT_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CodecObjectManifest:
        expected = {
            "schema",
            "state_sha256",
            "frozen_space_sha256",
            "compile_config_sha256",
            "receiver_bundle_sha256",
            "stage",
            "research_only",
            "score_claim",
            "promotion_eligible",
        }
        _exact_keys(value, expected, "codec object")
        result = cls(
            schema=value["schema"],
            state_sha256=value["state_sha256"],
            frozen_space_sha256=value["frozen_space_sha256"],
            compile_config_sha256=value["compile_config_sha256"],
            receiver_bundle_sha256=value["receiver_bundle_sha256"],
            research_only=value["research_only"],
            score_claim=value["score_claim"],
            promotion_eligible=value["promotion_eligible"],
        )
        if value["stage"] != result.stage:
            raise CoupledWitnessStateError("codec-object stage does not reconcile")
        return result

    @classmethod
    def from_bytes(cls, payload: bytes) -> CodecObjectManifest:
        envelope = decode_canonical_json(payload)
        expected = {"schema", "body", "body_sha256"}
        _exact_keys(envelope, expected, "codec object envelope")
        if envelope["schema"] != CODEC_OBJECT_ENVELOPE_SCHEMA:
            raise CoupledWitnessStateError("codec-object envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise CoupledWitnessStateError("codec-object body hash differs")
        result = cls.from_dict(envelope["body"])
        if result.object_sha256 != envelope["body_sha256"]:
            raise CoupledWitnessStateError("codec-object identity differs")
        return result


__all__ = [
    "CODEC_OBJECT_MANIFEST_SCHEMA",
    "COMPILE_STREAM_POLICY_SCHEMA",
    "CONTENT_ADDRESS_SCHEMA",
    "COUPLED_WITNESS_STATE_SCHEMA",
    "DECODER_PAYLOAD_POLICY",
    "FROZEN_SPACE_IDENTITY_SCHEMA",
    "SCIENTIFIC_STREAM_DEPENDENCY_SCHEMA",
    "SCIENTIFIC_STREAM_ORDER",
    "SCIENTIFIC_STREAM_SCHEMA",
    "SOURCE_DERIVED_LINEAGE",
    "STATE_PATCH_ENVELOPE_SCHEMA",
    "STATE_PATCH_SCHEMA",
    "STATE_TRANSITION_RECEIPT_ENVELOPE_SCHEMA",
    "STATE_TRANSITION_RECEIPT_SCHEMA",
    "WITNESS_COMPILE_CONFIG_SCHEMA",
    "CodecObjectManifest",
    "CompileStreamPolicy",
    "ContentAddress",
    "CoupledWitnessState",
    "CoupledWitnessStateError",
    "FrozenSpaceIdentity",
    "ScientificStream",
    "ScientificStreamDependency",
    "ScientificStreamRole",
    "StatePatch",
    "StateTransitionReceipt",
    "WitnessCompileConfig",
    "apply_state_patch",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_canonical_json",
    "sha256_bytes",
]
