# SPDX-License-Identifier: MIT
"""Strict readiness seam for the G17/G29/G33 selected-solution endpoint.

This module deliberately does not manufacture a G33 base from hashes.  It
reopens the durable G29 preflight receipts, joins them to the current dynamic
frontier pointer, and emits a machine-readable account of the exact evidence
still required before a selected-solution state can become a public endpoint.

The missing G17 whole-object and receiver/raster receipts are named schemas,
not opaque byte slots.  Until strict parsers for those schemas exist, their
roles remain unsatisfied and no base, endpoint, score authority, or candidate
claim can be constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Final, Self

from tac.canonical_frontier_pointer import (
    POINTER_SCHEMA_VERSION,
    CanonicalFrontierPointer,
    effective_frontier_score,
    recompute_effective_frontier,
)
from tac.contest_score import rate_term
from tac.witness_control.verified_continuation_certificate_v1 import (
    GENERATOR_DOMAIN_MANIFEST_SCHEMA,
    PROOF_ENVELOPE_SCHEMA,
)
from tac.witness_dsl.taskspace_public_auth_eval_closure import (
    ABI_CLOSURE_SCHEMA,
    DEPENDENCY_DISCOVERY_SCHEMA,
    GENERIC_SOURCE_AUDIT_SCHEMA,
    PLACEMENT_SCHEMA,
    PUBLIC_EXECUTION_SCHEMA,
    READINESS_SCHEMA,
    RUNTIME_COMPILE_SCHEMA,
    AuthClosureExecutionReadinessReceiptV1,
    GenericSourceAuditReceiptV1,
    InterpreterDistributionABIClosureV1,
    PayloadPlacementManifestV1,
    PublicEvaluatorExecutionReceiptV1,
    PublicRuntimeCompileReceiptV1,
    RuntimeDependencyDiscoveryReceiptV1,
)

SCHEMA: Final = "tac.taskspace_g23_g29_g33_endpoint_adapter_readiness.v1"
G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA: Final = "tac.g17_whole_object_state_receipt.v1"
G17_G29_RECEIVER_RASTER_BRIDGE_SCHEMA: Final = "tac.g17_g29_receiver_raster_bridge_receipt.v1"
G33_ACTION_CONTINUATION_RECEIPT_SCHEMA: Final = "tac.g33_action_continuation_receipt.v1"


class G36EndpointAdapterError(ValueError):
    """Raised when retained evidence is corrupt or describes different objects."""


class G36EvidenceRoleV1(StrEnum):
    G17_WHOLE_OBJECT_STATE = "G17_WHOLE_OBJECT_STATE"
    G17_G29_RECEIVER_RASTER_BRIDGE = "G17_G29_RECEIVER_RASTER_BRIDGE"
    G29_PUBLIC_RUNTIME_COMPILE = "G29_PUBLIC_RUNTIME_COMPILE"
    G29_PAYLOAD_PLACEMENT = "G29_PAYLOAD_PLACEMENT"
    G29_DECODER_ABI = "G29_DECODER_ABI"
    G29_GENERIC_SOURCE_AUDITS = "G29_GENERIC_SOURCE_AUDITS"
    G29_RUNTIME_DEPENDENCY_DISCOVERY = "G29_RUNTIME_DEPENDENCY_DISCOVERY"
    G29_EXECUTION_READINESS = "G29_EXECUTION_READINESS"
    G29_PUBLIC_EVALUATOR_EXECUTION = "G29_PUBLIC_EVALUATOR_EXECUTION"
    G33_GENERATOR_DOMAIN_MANIFEST = "G33_GENERATOR_DOMAIN_MANIFEST"
    G33_ACTION_CONTINUATION = "G33_ACTION_CONTINUATION"
    G38_CONTINUATION_PROOF = "G38_CONTINUATION_PROOF"
    DYNAMIC_FRONTIER_POINTER = "DYNAMIC_FRONTIER_POINTER"


ROLE_SCHEMA: Final[dict[G36EvidenceRoleV1, str]] = {
    G36EvidenceRoleV1.G17_WHOLE_OBJECT_STATE: G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA,
    G36EvidenceRoleV1.G17_G29_RECEIVER_RASTER_BRIDGE: G17_G29_RECEIVER_RASTER_BRIDGE_SCHEMA,
    G36EvidenceRoleV1.G29_PUBLIC_RUNTIME_COMPILE: RUNTIME_COMPILE_SCHEMA,
    G36EvidenceRoleV1.G29_PAYLOAD_PLACEMENT: PLACEMENT_SCHEMA,
    G36EvidenceRoleV1.G29_DECODER_ABI: ABI_CLOSURE_SCHEMA,
    G36EvidenceRoleV1.G29_GENERIC_SOURCE_AUDITS: GENERIC_SOURCE_AUDIT_SCHEMA,
    G36EvidenceRoleV1.G29_RUNTIME_DEPENDENCY_DISCOVERY: DEPENDENCY_DISCOVERY_SCHEMA,
    G36EvidenceRoleV1.G29_EXECUTION_READINESS: READINESS_SCHEMA,
    G36EvidenceRoleV1.G29_PUBLIC_EVALUATOR_EXECUTION: PUBLIC_EXECUTION_SCHEMA,
    G36EvidenceRoleV1.G33_GENERATOR_DOMAIN_MANIFEST: GENERATOR_DOMAIN_MANIFEST_SCHEMA,
    G36EvidenceRoleV1.G33_ACTION_CONTINUATION: G33_ACTION_CONTINUATION_RECEIPT_SCHEMA,
    G36EvidenceRoleV1.G38_CONTINUATION_PROOF: PROOF_ENVELOPE_SCHEMA,
    G36EvidenceRoleV1.DYNAMIC_FRONTIER_POINTER: POINTER_SCHEMA_VERSION,
}


class G36BlockerCodeV1(StrEnum):
    G17_STRICT_WHOLE_OBJECT_STATE_RECEIPT_OWED = "G17_STRICT_WHOLE_OBJECT_STATE_RECEIPT_OWED"
    G17_TO_G29_PUBLIC_RASTER_BRIDGE_RECEIPT_OWED = "G17_TO_G29_PUBLIC_RASTER_BRIDGE_RECEIPT_OWED"
    G29_PUBLIC_EVALUATOR_EXECUTION_RECEIPT_OWED = "G29_PUBLIC_EVALUATOR_EXECUTION_RECEIPT_OWED"
    G33_EXACT_GENERATOR_DOMAIN_MANIFEST_RECEIPT_OWED = "G33_EXACT_GENERATOR_DOMAIN_MANIFEST_RECEIPT_OWED"
    G33_ACTION_CONTINUATION_RECEIPT_OWED = "G33_ACTION_CONTINUATION_RECEIPT_OWED"
    G38_PROOF_BYTE_BUNDLE_OWED = "G38_PROOF_BYTE_BUNDLE_OWED"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise G36EndpointAdapterError("value is not finite canonical ASCII JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes or not payload:
        raise G36EndpointAdapterError("evidence payload must be nonempty immutable bytes")
    return hashlib.sha256(payload).hexdigest()


def _parse_json_strict(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise G36EndpointAdapterError("pointer payload must be nonempty immutable bytes")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G36EndpointAdapterError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                G36EndpointAdapterError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G36EndpointAdapterError("pointer payload is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise G36EndpointAdapterError("pointer payload must be one JSON object")
    return value


def _strict_pointer(payload: bytes) -> tuple[str, float]:
    value = _parse_json_strict(payload)
    if value.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise G36EndpointAdapterError("frontier pointer schema drifted")
    try:
        pointer = CanonicalFrontierPointer.from_dict(value)
        recomputed = recompute_effective_frontier(pointer)
        target = effective_frontier_score(pointer)
    except (KeyError, TypeError, ValueError) as exc:
        raise G36EndpointAdapterError("frontier pointer cannot be recomputed") from exc
    if target is None or not math.isfinite(target) or target <= 0.0:
        raise G36EndpointAdapterError("frontier pointer has no finite positive target")
    cached = value.get("effective_frontier")
    if type(cached) is not dict or _canonical_json(cached) != _canonical_json(recomputed):
        raise G36EndpointAdapterError("frontier pointer cache differs from recomputed custody rows")
    return _sha256_bytes(payload), target


@dataclass(frozen=True, slots=True)
class G36G29PreflightEvidenceBytesV1:
    compile_receipt_bytes: bytes = field(repr=False)
    placement_receipt_bytes: bytes = field(repr=False)
    decoder_abi_receipt_bytes: bytes = field(repr=False)
    generic_source_audit_receipt_bytes: tuple[bytes, ...] = field(repr=False)
    dependency_discovery_receipt_bytes: bytes = field(repr=False)
    execution_readiness_receipt_bytes: bytes = field(repr=False)
    public_evaluator_execution_receipt_bytes: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "compile_receipt_bytes",
            "placement_receipt_bytes",
            "decoder_abi_receipt_bytes",
            "dependency_discovery_receipt_bytes",
            "execution_readiness_receipt_bytes",
        ):
            _sha256_bytes(getattr(self, name))
        if type(self.generic_source_audit_receipt_bytes) is not tuple or not self.generic_source_audit_receipt_bytes:
            raise G36EndpointAdapterError("G29 evidence needs one or more source-audit receipts")
        for payload in self.generic_source_audit_receipt_bytes:
            _sha256_bytes(payload)
        if self.public_evaluator_execution_receipt_bytes is not None:
            _sha256_bytes(self.public_evaluator_execution_receipt_bytes)


@dataclass(frozen=True, slots=True)
class G36EndpointAdapterReadinessReceiptV1:
    archive_sha256: str
    archive_nbytes: int
    dynamic_pointer_sha256: str
    dynamic_pointer_nbytes: int
    dynamic_target_score: float
    exact_rate_term: float
    residual_distortion_score_budget_at_current_rate: float
    zero_pose_d_seg_intercept: float
    zero_seg_d_pose_intercept: float
    verified_input_receipts: tuple[tuple[str, str], ...]
    required_role_schemas: tuple[tuple[str, str], ...]
    satisfied_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    blockers: tuple[str, ...]
    g33_base_constructible: bool
    g33_endpoint_constructible: bool
    score_authority: bool
    candidate_archive: bool
    frontier_pointer_moved: bool
    research_only: bool
    schema: str = field(default=SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_nbytes",
            "archive_sha256",
            "blockers",
            "candidate_archive",
            "dynamic_pointer_nbytes",
            "dynamic_pointer_sha256",
            "dynamic_target_score",
            "exact_rate_term",
            "frontier_pointer_moved",
            "g33_base_constructible",
            "g33_endpoint_constructible",
            "missing_roles",
            "required_role_schemas",
            "research_only",
            "residual_distortion_score_budget_at_current_rate",
            "satisfied_roles",
            "schema",
            "score_authority",
            "verified_input_receipts",
            "zero_pose_d_seg_intercept",
            "zero_seg_d_pose_intercept",
        }
    )

    def __post_init__(self) -> None:
        if type(self.archive_sha256) is not str or len(self.archive_sha256) != 64:
            raise G36EndpointAdapterError("receipt archive SHA-256 is invalid")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise G36EndpointAdapterError("receipt archive bytes must be positive int")
        if type(self.dynamic_pointer_sha256) is not str or len(self.dynamic_pointer_sha256) != 64:
            raise G36EndpointAdapterError("receipt pointer SHA-256 is invalid")
        if type(self.dynamic_pointer_nbytes) is not int or self.dynamic_pointer_nbytes < 1:
            raise G36EndpointAdapterError("receipt pointer bytes must be positive int")
        for name in (
            "dynamic_target_score",
            "exact_rate_term",
            "residual_distortion_score_budget_at_current_rate",
            "zero_pose_d_seg_intercept",
            "zero_seg_d_pose_intercept",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise G36EndpointAdapterError(f"receipt {name} must be finite float")
        if self.exact_rate_term != rate_term(self.archive_nbytes):
            raise G36EndpointAdapterError("receipt rate term does not derive from exact archive bytes")
        residual = self.dynamic_target_score - self.exact_rate_term
        if self.residual_distortion_score_budget_at_current_rate != residual:
            raise G36EndpointAdapterError("receipt residual score budget does not derive exactly")
        positive_residual = max(0.0, residual)
        if (
            self.zero_pose_d_seg_intercept != positive_residual / 100.0
            or self.zero_seg_d_pose_intercept != positive_residual * positive_residual / 10.0
        ):
            raise G36EndpointAdapterError("receipt frontier intercepts do not derive exactly")
        for rows, label in (
            (self.verified_input_receipts, "verified input receipts"),
            (self.required_role_schemas, "required role schemas"),
        ):
            if (
                type(rows) is not tuple
                or rows != tuple(sorted(set(rows)))
                or any(
                    type(row) is not tuple
                    or len(row) != 2
                    or any(type(value) is not str or not value.isascii() or not value for value in row)
                    for row in rows
                )
            ):
                raise G36EndpointAdapterError(f"receipt {label} must be canonical string pairs")
        for values, label in (
            (self.satisfied_roles, "satisfied roles"),
            (self.missing_roles, "missing roles"),
            (self.blockers, "blockers"),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
                or any(type(value) is not str or not value.isascii() or not value for value in values)
            ):
                raise G36EndpointAdapterError(f"receipt {label} must be canonical ASCII tuple")
        all_roles = {role.value for role in G36EvidenceRoleV1}
        if set(self.satisfied_roles) & set(self.missing_roles):
            raise G36EndpointAdapterError("receipt marks an evidence role both satisfied and missing")
        if set(self.satisfied_roles) | set(self.missing_roles) != all_roles:
            raise G36EndpointAdapterError("receipt role partition differs from the closed evidence vocabulary")
        expected_schemas = tuple(sorted((role.value, schema) for role, schema in ROLE_SCHEMA.items()))
        if self.required_role_schemas != expected_schemas:
            raise G36EndpointAdapterError("receipt role/schema contract drifted")
        if (
            self.g33_base_constructible
            or self.g33_endpoint_constructible
            or self.score_authority
            or self.candidate_archive
            or self.frontier_pointer_moved
            or self.research_only is not True
        ):
            raise G36EndpointAdapterError("readiness receipt attempted to mint authority or a candidate")

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.to_receipt_bytes()).hexdigest()

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_nbytes": self.archive_nbytes,
                "archive_sha256": self.archive_sha256,
                "blockers": list(self.blockers),
                "candidate_archive": self.candidate_archive,
                "dynamic_pointer_nbytes": self.dynamic_pointer_nbytes,
                "dynamic_pointer_sha256": self.dynamic_pointer_sha256,
                "dynamic_target_score": self.dynamic_target_score,
                "exact_rate_term": self.exact_rate_term,
                "frontier_pointer_moved": self.frontier_pointer_moved,
                "g33_base_constructible": self.g33_base_constructible,
                "g33_endpoint_constructible": self.g33_endpoint_constructible,
                "missing_roles": list(self.missing_roles),
                "required_role_schemas": [list(row) for row in self.required_role_schemas],
                "research_only": self.research_only,
                "residual_distortion_score_budget_at_current_rate": (
                    self.residual_distortion_score_budget_at_current_rate
                ),
                "satisfied_roles": list(self.satisfied_roles),
                "schema": self.schema,
                "score_authority": self.score_authority,
                "verified_input_receipts": [list(row) for row in self.verified_input_receipts],
                "zero_pose_d_seg_intercept": self.zero_pose_d_seg_intercept,
                "zero_seg_d_pose_intercept": self.zero_seg_d_pose_intercept,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_json_strict(payload)
        if set(value) != cls._KEYS or value.get("schema") != SCHEMA:
            raise G36EndpointAdapterError("G36 readiness receipt fields or schema drifted")
        if _canonical_json(value) != payload:
            raise G36EndpointAdapterError("G36 readiness receipt failed canonical parse/re-emit")
        return cls(
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            dynamic_pointer_sha256=value["dynamic_pointer_sha256"],
            dynamic_pointer_nbytes=value["dynamic_pointer_nbytes"],
            dynamic_target_score=value["dynamic_target_score"],
            exact_rate_term=value["exact_rate_term"],
            residual_distortion_score_budget_at_current_rate=value["residual_distortion_score_budget_at_current_rate"],
            zero_pose_d_seg_intercept=value["zero_pose_d_seg_intercept"],
            zero_seg_d_pose_intercept=value["zero_seg_d_pose_intercept"],
            verified_input_receipts=tuple(tuple(row) for row in value["verified_input_receipts"]),
            required_role_schemas=tuple(tuple(row) for row in value["required_role_schemas"]),
            satisfied_roles=tuple(value["satisfied_roles"]),
            missing_roles=tuple(value["missing_roles"]),
            blockers=tuple(value["blockers"]),
            g33_base_constructible=value["g33_base_constructible"],
            g33_endpoint_constructible=value["g33_endpoint_constructible"],
            score_authority=value["score_authority"],
            candidate_archive=value["candidate_archive"],
            frontier_pointer_moved=value["frontier_pointer_moved"],
            research_only=value["research_only"],
        )


def audit_g36_endpoint_adapter_readiness(
    evidence: G36G29PreflightEvidenceBytesV1,
    *,
    dynamic_pointer_bytes: bytes,
) -> G36EndpointAdapterReadinessReceiptV1:
    """Reopen the current G29 packet and emit the exact G36 blocker partition."""

    if type(evidence) is not G36G29PreflightEvidenceBytesV1:
        raise G36EndpointAdapterError("G36 audit requires the exact typed G29 evidence bundle")
    try:
        compile_receipt = PublicRuntimeCompileReceiptV1.from_receipt_bytes(evidence.compile_receipt_bytes)
        placement = PayloadPlacementManifestV1.from_receipt_bytes(evidence.placement_receipt_bytes)
        decoder_abi = InterpreterDistributionABIClosureV1.from_receipt_bytes(evidence.decoder_abi_receipt_bytes)
        source_audits = tuple(
            GenericSourceAuditReceiptV1.from_receipt_bytes(payload)
            for payload in evidence.generic_source_audit_receipt_bytes
        )
        discovery = RuntimeDependencyDiscoveryReceiptV1.from_receipt_bytes(evidence.dependency_discovery_receipt_bytes)
        readiness = AuthClosureExecutionReadinessReceiptV1.from_receipt_bytes(
            evidence.execution_readiness_receipt_bytes
        )
    except (TypeError, ValueError) as exc:
        raise G36EndpointAdapterError("G29 evidence failed its strict receipt parser") from exc

    source_audit_ids = tuple(sorted(row.identity_sha256 for row in source_audits))
    if not all(row.passed for row in source_audits):
        raise G36EndpointAdapterError("G29 generic source audit did not pass")
    if (
        compile_receipt.archive_sha256 != placement.archive_sha256
        or compile_receipt.archive_nbytes != placement.archive_nbytes
        or compile_receipt.placement_identity_sha256 != placement.identity_sha256
        or compile_receipt.abi_identity_sha256 != decoder_abi.identity_sha256
        or compile_receipt.source_audit_receipt_sha256s != source_audit_ids
        or discovery.compile_receipt_sha256 != compile_receipt.identity_sha256
        or discovery.decoder_abi_identity_sha256 != decoder_abi.identity_sha256
        or readiness.compile_receipt_sha256 != compile_receipt.identity_sha256
        or readiness.decoder_abi_identity_sha256 != decoder_abi.identity_sha256
        or readiness.runtime_tree_sha256 != compile_receipt.runtime_tree_sha256
        or readiness.dependency_discovery_receipt_sha256 != discovery.identity_sha256
    ):
        raise G36EndpointAdapterError("G29 compile/placement/ABI/discovery/readiness identities do not join")

    pointer_sha256, target = _strict_pointer(dynamic_pointer_bytes)
    satisfied = {
        G36EvidenceRoleV1.G29_PUBLIC_RUNTIME_COMPILE,
        G36EvidenceRoleV1.G29_PAYLOAD_PLACEMENT,
        G36EvidenceRoleV1.G29_DECODER_ABI,
        G36EvidenceRoleV1.G29_GENERIC_SOURCE_AUDITS,
        G36EvidenceRoleV1.G29_RUNTIME_DEPENDENCY_DISCOVERY,
        G36EvidenceRoleV1.G29_EXECUTION_READINESS,
        G36EvidenceRoleV1.DYNAMIC_FRONTIER_POINTER,
    }
    verified = {
        G36EvidenceRoleV1.G29_PUBLIC_RUNTIME_COMPILE: compile_receipt.identity_sha256,
        G36EvidenceRoleV1.G29_PAYLOAD_PLACEMENT: placement.identity_sha256,
        G36EvidenceRoleV1.G29_DECODER_ABI: decoder_abi.identity_sha256,
        G36EvidenceRoleV1.G29_GENERIC_SOURCE_AUDITS: hashlib.sha256(
            _canonical_json(list(source_audit_ids))
        ).hexdigest(),
        G36EvidenceRoleV1.G29_RUNTIME_DEPENDENCY_DISCOVERY: discovery.identity_sha256,
        G36EvidenceRoleV1.G29_EXECUTION_READINESS: readiness.identity_sha256,
        G36EvidenceRoleV1.DYNAMIC_FRONTIER_POINTER: pointer_sha256,
    }
    blockers = {
        G36BlockerCodeV1.G17_STRICT_WHOLE_OBJECT_STATE_RECEIPT_OWED,
        G36BlockerCodeV1.G17_TO_G29_PUBLIC_RASTER_BRIDGE_RECEIPT_OWED,
        G36BlockerCodeV1.G33_EXACT_GENERATOR_DOMAIN_MANIFEST_RECEIPT_OWED,
        G36BlockerCodeV1.G33_ACTION_CONTINUATION_RECEIPT_OWED,
        G36BlockerCodeV1.G38_PROOF_BYTE_BUNDLE_OWED,
    }

    if evidence.public_evaluator_execution_receipt_bytes is None:
        blockers.add(G36BlockerCodeV1.G29_PUBLIC_EVALUATOR_EXECUTION_RECEIPT_OWED)
    else:
        try:
            execution = PublicEvaluatorExecutionReceiptV1.from_receipt_bytes(
                evidence.public_evaluator_execution_receipt_bytes
            )
        except (TypeError, ValueError) as exc:
            raise G36EndpointAdapterError("G29 public execution failed its strict receipt parser") from exc
        if (
            execution.archive_sha256 != compile_receipt.archive_sha256
            or execution.archive_nbytes != compile_receipt.archive_nbytes
            or execution.compile_receipt_sha256 != compile_receipt.identity_sha256
            or execution.dependency_discovery_receipt_sha256 != discovery.identity_sha256
            or execution.abi_identity_sha256 != decoder_abi.identity_sha256
        ):
            raise G36EndpointAdapterError("G29 public execution belongs to another physical object")
        satisfied.add(G36EvidenceRoleV1.G29_PUBLIC_EVALUATOR_EXECUTION)
        verified[G36EvidenceRoleV1.G29_PUBLIC_EVALUATOR_EXECUTION] = execution.identity_sha256

    missing = set(G36EvidenceRoleV1) - satisfied
    exact_rate = rate_term(compile_receipt.archive_nbytes)
    residual = target - exact_rate
    positive_residual = max(0.0, residual)
    return G36EndpointAdapterReadinessReceiptV1(
        archive_sha256=compile_receipt.archive_sha256,
        archive_nbytes=compile_receipt.archive_nbytes,
        dynamic_pointer_sha256=pointer_sha256,
        dynamic_pointer_nbytes=len(dynamic_pointer_bytes),
        dynamic_target_score=target,
        exact_rate_term=exact_rate,
        residual_distortion_score_budget_at_current_rate=residual,
        zero_pose_d_seg_intercept=positive_residual / 100.0,
        zero_seg_d_pose_intercept=positive_residual * positive_residual / 10.0,
        verified_input_receipts=tuple(sorted((role.value, digest) for role, digest in verified.items())),
        required_role_schemas=tuple(sorted((role.value, schema) for role, schema in ROLE_SCHEMA.items())),
        satisfied_roles=tuple(sorted(role.value for role in satisfied)),
        missing_roles=tuple(sorted(role.value for role in missing)),
        blockers=tuple(sorted(blocker.value for blocker in blockers)),
        g33_base_constructible=False,
        g33_endpoint_constructible=False,
        score_authority=False,
        candidate_archive=False,
        frontier_pointer_moved=False,
        research_only=True,
    )


__all__ = [
    "G17_G29_RECEIVER_RASTER_BRIDGE_SCHEMA",
    "G17_WHOLE_OBJECT_STATE_RECEIPT_SCHEMA",
    "G33_ACTION_CONTINUATION_RECEIPT_SCHEMA",
    "ROLE_SCHEMA",
    "SCHEMA",
    "G36BlockerCodeV1",
    "G36EndpointAdapterError",
    "G36EndpointAdapterReadinessReceiptV1",
    "G36EvidenceRoleV1",
    "G36G29PreflightEvidenceBytesV1",
    "audit_g36_endpoint_adapter_readiness",
]
