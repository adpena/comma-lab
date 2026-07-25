# SPDX-License-Identifier: MIT
"""Identity-bearing bridge from codec codewords to exact score transitions.

``ActionEffect`` says what changed on one evaluator authority surface.  It does
not, by itself, prove which description codeword was applied, which receiver
operator applied it, or which typed byte home paid for it.  This module is the
small executable-IR/relocation record joining those surfaces without turning
``ActionEffect`` into a codec-specific kitchen sink.

The receipt is deliberately analysis-only and fail-closed.  It cannot promote
an archive or claim a contest score.  A measured receipt must bind one base and
candidate archive, one concrete application operator, one receiver/R chain,
one typed stream home, and one finite joint ``ActionEffect`` transition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tac.analysis.action_effect import ActionEffect
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType

APPLIED_ACTION_RECEIPT_SCHEMA = "tac.applied_action_receipt.v1"
STREAM_HOME_CLAIM_SCHEMA = "tac.applied_action_stream_home.v1"


class AppliedActionReceiptError(ValueError):
    """Raised when an application receipt loses identity or accounting."""


class ApplicationStatus(StrEnum):
    """Scoped outcome of applying one identity-bound codeword."""

    DOWNHILL_FINITE = "DOWNHILL_FINITE"
    UPHILL_NULL = "UPHILL_NULL"
    CAUSAL_MISS = "CAUSAL_MISS"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    UNMEASURED = "UNMEASURED"


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppliedActionReceiptError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _require_text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise AppliedActionReceiptError(f"{name} must be a SHA-256 hex string")
    return text.lower()


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AppliedActionReceiptError(f"{name} must be a non-negative exact integer")
    return value


@dataclass(frozen=True)
class StreamHomeClaim:
    """The single independently accounted stream home charged by an action."""

    stream_type: StreamType
    layer_home: LayerHome
    byte_home_id: str
    coder_id: str
    coder_owner: str
    receiver_consumer: str
    bytes_before: int
    bytes_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.stream_type, StreamType):
            raise AppliedActionReceiptError("stream_type must be StreamType")
        if not isinstance(self.layer_home, LayerHome):
            raise AppliedActionReceiptError("layer_home must be LayerHome")
        for name, value in (
            ("byte_home_id", self.byte_home_id),
            ("coder_id", self.coder_id),
            ("coder_owner", self.coder_owner),
            ("receiver_consumer", self.receiver_consumer),
        ):
            _require_text(value, name)
        _require_nonnegative_int(self.bytes_before, "bytes_before")
        _require_nonnegative_int(self.bytes_after, "bytes_after")
        if self.stream_type is StreamType.GAUGE and self.bytes_after != 0:
            raise AppliedActionReceiptError("GAUGE stream homes must remain zero-byte")

    @property
    def delta_bytes(self) -> int:
        return self.bytes_after - self.bytes_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STREAM_HOME_CLAIM_SCHEMA,
            "stream_type": self.stream_type.value,
            "layer_home": self.layer_home.value,
            "byte_home_id": self.byte_home_id,
            "coder_id": self.coder_id,
            "coder_owner": self.coder_owner,
            "receiver_consumer": self.receiver_consumer,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "delta_bytes": self.delta_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StreamHomeClaim:
        if not isinstance(payload, Mapping):
            raise AppliedActionReceiptError("stream_home must be a mapping")
        if payload.get("schema") != STREAM_HOME_CLAIM_SCHEMA:
            raise AppliedActionReceiptError("stream_home schema differs")
        claim = cls(
            stream_type=StreamType(payload.get("stream_type")),
            layer_home=LayerHome(payload.get("layer_home")),
            byte_home_id=payload.get("byte_home_id"),
            coder_id=payload.get("coder_id"),
            coder_owner=payload.get("coder_owner"),
            receiver_consumer=payload.get("receiver_consumer"),
            bytes_before=payload.get("bytes_before"),
            bytes_after=payload.get("bytes_after"),
        )
        if payload.get("delta_bytes") != claim.delta_bytes:
            raise AppliedActionReceiptError("stream_home delta_bytes does not reconcile")
        return claim


@dataclass(frozen=True)
class AppliedActionReceipt:
    """Foreign-key receipt joining one codeword to one finite receiver edge."""

    schema: str
    receipt_id: str
    status: ApplicationStatus
    action_id: str
    codeword_id: str
    application_operator_id: str
    application_operator_version: str
    ordered_parent_action_ids: tuple[str, ...]
    base_archive_sha256: str
    candidate_archive_sha256: str
    base_payload_sha256: str | None
    candidate_payload_sha256: str | None
    base_state_sha256: str | None
    physical_edge_id: str
    edge_from_state_id: str
    edge_to_state_id: str
    pair_ids: tuple[int, ...]
    support_sha256: str | None
    bucket_id: str | None
    integer_quantum: int
    direction: int
    validity_radius: float
    receiver_schema: str
    receiver_sha256: str
    r_chain_id: str
    changed_uint8_count: int
    changed_uint8_sha256: str | None
    stream_home: StreamHomeClaim
    action_effect: ActionEffect
    authority_axis: str
    verdict_scope: str
    provenance_ref: str
    blockers: tuple[str, ...] = ()
    research_only: bool = True
    promotion_eligible: bool = False
    score_claim: bool = False

    def __post_init__(self) -> None:
        if self.schema != APPLIED_ACTION_RECEIPT_SCHEMA:
            raise AppliedActionReceiptError("applied-action receipt schema differs")
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("action_id", self.action_id),
            ("codeword_id", self.codeword_id),
            ("application_operator_id", self.application_operator_id),
            ("application_operator_version", self.application_operator_version),
            ("physical_edge_id", self.physical_edge_id),
            ("edge_from_state_id", self.edge_from_state_id),
            ("edge_to_state_id", self.edge_to_state_id),
            ("receiver_schema", self.receiver_schema),
            ("r_chain_id", self.r_chain_id),
            ("authority_axis", self.authority_axis),
            ("verdict_scope", self.verdict_scope),
            ("provenance_ref", self.provenance_ref),
        ):
            _require_text(value, name)
        if not isinstance(self.status, ApplicationStatus):
            raise AppliedActionReceiptError("status must be ApplicationStatus")
        for name, value, optional in (
            ("base_archive_sha256", self.base_archive_sha256, False),
            ("candidate_archive_sha256", self.candidate_archive_sha256, False),
            ("base_payload_sha256", self.base_payload_sha256, True),
            ("candidate_payload_sha256", self.candidate_payload_sha256, True),
            ("base_state_sha256", self.base_state_sha256, True),
            ("support_sha256", self.support_sha256, True),
            ("receiver_sha256", self.receiver_sha256, False),
            ("changed_uint8_sha256", self.changed_uint8_sha256, True),
        ):
            _require_sha256(value, name, optional=optional)
        if not isinstance(self.ordered_parent_action_ids, tuple):
            raise AppliedActionReceiptError("ordered_parent_action_ids must be a tuple")
        if any(not isinstance(value, str) or not value.strip() for value in self.ordered_parent_action_ids):
            raise AppliedActionReceiptError("ordered_parent_action_ids contain an empty identity")
        if len(set(self.ordered_parent_action_ids)) != len(self.ordered_parent_action_ids):
            raise AppliedActionReceiptError("ordered_parent_action_ids must be unique")
        if not isinstance(self.pair_ids, tuple):
            raise AppliedActionReceiptError("pair_ids must be a tuple")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.pair_ids):
            raise AppliedActionReceiptError("pair_ids must be non-negative integers")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise AppliedActionReceiptError("pair_ids must be unique")
        if isinstance(self.integer_quantum, bool) or not isinstance(self.integer_quantum, int):
            raise AppliedActionReceiptError("integer_quantum must be an exact integer")
        if self.direction not in {-1, 0, 1}:
            raise AppliedActionReceiptError("direction must be -1, 0, or 1")
        if not isinstance(self.validity_radius, (int, float)) or not math.isfinite(float(self.validity_radius)):
            raise AppliedActionReceiptError("validity_radius must be finite")
        if float(self.validity_radius) < 0.0:
            raise AppliedActionReceiptError("validity_radius must be non-negative")
        _require_nonnegative_int(self.changed_uint8_count, "changed_uint8_count")
        if self.changed_uint8_count > 0 and self.changed_uint8_sha256 is None:
            raise AppliedActionReceiptError("changed_uint8_sha256 is required when uint8 changed")
        if not isinstance(self.stream_home, StreamHomeClaim):
            raise AppliedActionReceiptError("stream_home must be StreamHomeClaim")
        if not isinstance(self.action_effect, ActionEffect):
            raise AppliedActionReceiptError("action_effect must be ActionEffect")
        if not isinstance(self.blockers, tuple):
            raise AppliedActionReceiptError("blockers must be a tuple")
        if self.promotion_eligible is not False or self.score_claim is not False:
            raise AppliedActionReceiptError("application receipts are non-promotional analysis rows")
        if self.research_only is not True:
            raise AppliedActionReceiptError("v1 application receipts must remain research_only")
        self._validate_foreign_keys()
        self._validate_outcome()

    def _validate_foreign_keys(self) -> None:
        effect = self.action_effect
        if effect.action_id != self.action_id:
            raise AppliedActionReceiptError("action_effect action_id foreign key differs")
        if tuple(effect.composed_action_ids) != self.ordered_parent_action_ids:
            raise AppliedActionReceiptError("ordered parent action identities differ")
        if effect.base_archive_sha256 != self.base_archive_sha256:
            raise AppliedActionReceiptError("action_effect base archive identity differs")
        if effect.archive_sha256 != self.candidate_archive_sha256:
            raise AppliedActionReceiptError("action_effect candidate archive identity differs")
        if effect.base_payload_sha256 != self.base_payload_sha256:
            raise AppliedActionReceiptError("action_effect base payload identity differs")
        if effect.payload_sha256 != self.candidate_payload_sha256:
            raise AppliedActionReceiptError("action_effect candidate payload identity differs")
        if effect.base_state_sha256 != self.base_state_sha256:
            raise AppliedActionReceiptError("action_effect base state identity differs")
        if effect.authority != self.authority_axis:
            raise AppliedActionReceiptError("action_effect authority axis differs")
        if effect.delta_bytes != self.stream_home.delta_bytes:
            raise AppliedActionReceiptError("stream-home byte delta differs from action effect")

    def _validate_outcome(self) -> None:
        delta = self.action_effect.delta_score_total
        measured = self.status in {
            ApplicationStatus.DOWNHILL_FINITE,
            ApplicationStatus.UPHILL_NULL,
        }
        if measured:
            if delta is None or not math.isfinite(float(delta)):
                raise AppliedActionReceiptError("measured status requires finite joint delta_score_total")
            if self.direction == 0 or self.integer_quantum == 0:
                raise AppliedActionReceiptError("measured status requires a nonzero integer move")
        if self.status is ApplicationStatus.DOWNHILL_FINITE and not float(delta) < 0.0:
            raise AppliedActionReceiptError("DOWNHILL_FINITE requires negative joint delta")
        if self.status is ApplicationStatus.UPHILL_NULL and not float(delta) >= 0.0:
            raise AppliedActionReceiptError("UPHILL_NULL requires non-negative joint delta")
        if self.status is ApplicationStatus.UNMEASURED and delta is not None:
            raise AppliedActionReceiptError("UNMEASURED must not carry a finite score transition")
        if self.status is ApplicationStatus.IDENTITY_MISMATCH and not self.blockers:
            raise AppliedActionReceiptError("IDENTITY_MISMATCH requires an explicit blocker")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "action_id": self.action_id,
            "codeword_id": self.codeword_id,
            "application_operator_id": self.application_operator_id,
            "application_operator_version": self.application_operator_version,
            "ordered_parent_action_ids": list(self.ordered_parent_action_ids),
            "base_archive_sha256": self.base_archive_sha256,
            "candidate_archive_sha256": self.candidate_archive_sha256,
            "base_payload_sha256": self.base_payload_sha256,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "base_state_sha256": self.base_state_sha256,
            "physical_edge_id": self.physical_edge_id,
            "edge_from_state_id": self.edge_from_state_id,
            "edge_to_state_id": self.edge_to_state_id,
            "pair_ids": list(self.pair_ids),
            "support_sha256": self.support_sha256,
            "bucket_id": self.bucket_id,
            "integer_quantum": self.integer_quantum,
            "direction": self.direction,
            "validity_radius": float(self.validity_radius),
            "receiver_schema": self.receiver_schema,
            "receiver_sha256": self.receiver_sha256,
            "r_chain_id": self.r_chain_id,
            "changed_uint8_count": self.changed_uint8_count,
            "changed_uint8_sha256": self.changed_uint8_sha256,
            "stream_home": self.stream_home.as_dict(),
            "action_effect": self.action_effect.as_dict(),
            "authority_axis": self.authority_axis,
            "verdict_scope": self.verdict_scope,
            "provenance_ref": self.provenance_ref,
            "blockers": list(self.blockers),
            "research_only": self.research_only,
            "promotion_eligible": self.promotion_eligible,
            "score_claim": self.score_claim,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AppliedActionReceipt:
        if not isinstance(payload, Mapping):
            raise AppliedActionReceiptError("applied-action receipt must be a mapping")
        try:
            status = ApplicationStatus(payload.get("status"))
        except (TypeError, ValueError) as exc:
            raise AppliedActionReceiptError("application status differs") from exc
        return cls(
            schema=payload.get("schema"),
            receipt_id=payload.get("receipt_id"),
            status=status,
            action_id=payload.get("action_id"),
            codeword_id=payload.get("codeword_id"),
            application_operator_id=payload.get("application_operator_id"),
            application_operator_version=payload.get("application_operator_version"),
            ordered_parent_action_ids=tuple(payload.get("ordered_parent_action_ids") or ()),
            base_archive_sha256=payload.get("base_archive_sha256"),
            candidate_archive_sha256=payload.get("candidate_archive_sha256"),
            base_payload_sha256=payload.get("base_payload_sha256"),
            candidate_payload_sha256=payload.get("candidate_payload_sha256"),
            base_state_sha256=payload.get("base_state_sha256"),
            physical_edge_id=payload.get("physical_edge_id"),
            edge_from_state_id=payload.get("edge_from_state_id"),
            edge_to_state_id=payload.get("edge_to_state_id"),
            pair_ids=tuple(payload.get("pair_ids") or ()),
            support_sha256=payload.get("support_sha256"),
            bucket_id=payload.get("bucket_id"),
            integer_quantum=payload.get("integer_quantum"),
            direction=payload.get("direction"),
            validity_radius=payload.get("validity_radius"),
            receiver_schema=payload.get("receiver_schema"),
            receiver_sha256=payload.get("receiver_sha256"),
            r_chain_id=payload.get("r_chain_id"),
            changed_uint8_count=payload.get("changed_uint8_count"),
            changed_uint8_sha256=payload.get("changed_uint8_sha256"),
            stream_home=StreamHomeClaim.from_dict(payload.get("stream_home")),
            action_effect=ActionEffect.from_dict(payload.get("action_effect")),
            authority_axis=payload.get("authority_axis"),
            verdict_scope=payload.get("verdict_scope"),
            provenance_ref=payload.get("provenance_ref"),
            blockers=tuple(str(value) for value in payload.get("blockers") or ()),
            research_only=payload.get("research_only"),
            promotion_eligible=payload.get("promotion_eligible"),
            score_claim=payload.get("score_claim"),
        )


def build_applied_action_receipt(
    *,
    receipt_id: str,
    status: ApplicationStatus,
    action_effect: ActionEffect,
    codeword_id: str,
    application_operator_id: str,
    application_operator_version: str,
    physical_edge_id: str,
    edge_from_state_id: str,
    edge_to_state_id: str,
    integer_quantum: int,
    direction: int,
    validity_radius: float,
    receiver_schema: str,
    receiver_sha256: str,
    r_chain_id: str,
    changed_uint8_count: int,
    changed_uint8_sha256: str | None,
    stream_home: StreamHomeClaim,
    verdict_scope: str,
    provenance_ref: str,
    bucket_id: str | None = None,
    blockers: Sequence[str] = (),
) -> AppliedActionReceipt:
    """Build a receipt from the canonical identity fields on ``ActionEffect``."""

    if action_effect.base_archive_sha256 is None or action_effect.archive_sha256 is None:
        raise AppliedActionReceiptError(
            "ActionEffect must bind both base and candidate archives before application receipt"
        )
    return AppliedActionReceipt(
        schema=APPLIED_ACTION_RECEIPT_SCHEMA,
        receipt_id=receipt_id,
        status=status,
        action_id=action_effect.action_id,
        codeword_id=codeword_id,
        application_operator_id=application_operator_id,
        application_operator_version=application_operator_version,
        ordered_parent_action_ids=action_effect.composed_action_ids,
        base_archive_sha256=action_effect.base_archive_sha256,
        candidate_archive_sha256=action_effect.archive_sha256,
        base_payload_sha256=action_effect.base_payload_sha256,
        candidate_payload_sha256=action_effect.payload_sha256,
        base_state_sha256=action_effect.base_state_sha256,
        physical_edge_id=physical_edge_id,
        edge_from_state_id=edge_from_state_id,
        edge_to_state_id=edge_to_state_id,
        pair_ids=action_effect.pair_ids,
        support_sha256=action_effect.support_sha256,
        bucket_id=bucket_id,
        integer_quantum=integer_quantum,
        direction=direction,
        validity_radius=validity_radius,
        receiver_schema=receiver_schema,
        receiver_sha256=receiver_sha256,
        r_chain_id=r_chain_id,
        changed_uint8_count=changed_uint8_count,
        changed_uint8_sha256=changed_uint8_sha256,
        stream_home=stream_home,
        action_effect=action_effect,
        authority_axis=action_effect.authority,
        verdict_scope=verdict_scope,
        provenance_ref=provenance_ref,
        blockers=tuple(str(value) for value in blockers if str(value).strip()),
    )


__all__ = [
    "APPLIED_ACTION_RECEIPT_SCHEMA",
    "STREAM_HOME_CLAIM_SCHEMA",
    "ApplicationStatus",
    "AppliedActionReceipt",
    "AppliedActionReceiptError",
    "StreamHomeClaim",
    "build_applied_action_receipt",
]
