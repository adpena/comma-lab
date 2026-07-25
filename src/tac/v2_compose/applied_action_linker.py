# SPDX-License-Identifier: MIT
"""Deterministic relocation of identity-complete actions into a v2 packet.

The score-side :class:`~tac.analysis.applied_action_receipt.AppliedActionReceipt`
is an executable edge receipt, but it does not say where its counted codeword
lives inside a concrete packet grammar.  This module supplies that missing
linker step for the existing ``WTNV2`` shell.

The contract is deliberately narrow.  A relocation replaces one complete,
receiver-consumed physical section.  It never appends opaque bytes, edits only
the manifest, or assumes that two logical homes sharing a physical section are
independent.  Every single-action candidate is reconstructed from the common
base and checked against the action receipt's payload/archive hashes before a
multi-action packet is emitted.  The composed packet gets a new exact byte
receipt, but no composed score claim: interactions must be remeasured through
the receiver and frozen evaluators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tac.analysis.applied_action_receipt import (
    ApplicationStatus,
    AppliedActionReceipt,
)
from tac.v2_compose.archive_grammar import (
    build_v2_archive_zip_bytes,
    generate_v2_inflate_py,
    generate_v2_inflate_sh,
    pack_v2_archive,
    unpack_v2_sections,
    validate_v2_receiver_payload,
)

V2_SECTION_REPLACEMENT_SCHEMA = "tac.v2_section_replacement.v1"
V2_PACKET_RELOCATION_ENTRY_SCHEMA = "tac.v2_packet_relocation_entry.v1"
V2_PACKET_LINK_RECEIPT_SCHEMA = "tac.v2_packet_link_receipt.v1"
V2_PACKET_LINK_ATTEMPT_SCHEMA = "tac.v2_packet_link_attempt.v1"
V2_RECEIVER_SCHEMA = "tac.v2_compose.generated_receiver.v1"
V2_R_CHAIN_ID = "tac.v2_compose.bicubic_round_uint8_R.v1"


class AppliedActionLinkError(ValueError):
    """Raised when an action cannot be relocated without inventing custody."""


class V2Section(StrEnum):
    """Physical sections consumed by the current v2 receiver."""

    STORE = "store_blob"
    RESIDUAL = "residual_inr_blob"
    POSE = "pose_sidecar_blob"
    MANIFEST = "manifest"


_SECTION_ORDER = {
    V2Section.STORE: 0,
    V2Section.RESIDUAL: 1,
    V2Section.POSE: 2,
    V2Section.MANIFEST: 3,
}
_MEASURED_STATUSES = frozenset(
    {ApplicationStatus.DOWNHILL_FINITE, ApplicationStatus.UPHILL_NULL}
)
_SECTION_RECEIVER_CONSUMER = {
    V2Section.STORE: "tac.v2.receiver.store_blob",
    V2Section.RESIDUAL: "tac.v2.receiver.residual_inr_blob",
    V2Section.POSE: "tac.v2.receiver.pose_sidecar_blob",
}
_SECTION_ARCHIVE_MARGINAL_HOME = {
    V2Section.STORE: "v2.archive_marginal/store_blob",
    V2Section.RESIDUAL: "v2.archive_marginal/residual_inr_blob",
    V2Section.POSE: "v2.archive_marginal/pose_sidecar_blob",
}


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def v2_receiver_bundle_sha256(inflate_py: bytes, inflate_sh: bytes) -> str:
    """Domain-separated identity of both runnable receiver entrypoints."""

    if not isinstance(inflate_py, bytes) or not isinstance(inflate_sh, bytes):
        raise AppliedActionLinkError("receiver bundle members must be exact bytes")
    digest = hashlib.sha256()
    digest.update(b"tac.v2.receiver.bundle.v1\x00")
    for name, payload in ((b"inflate.py", inflate_py), (b"inflate.sh", inflate_sh)):
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def v2_receiver_file_sha256s() -> dict[str, str]:
    """Per-file hashes for materialization and mutation verification."""

    return {
        "inflate.py": _sha256(generate_v2_inflate_py().encode("utf-8")),
        "inflate.sh": _sha256(generate_v2_inflate_sh().encode("utf-8")),
    }


def v2_receiver_sha256() -> str:
    """Bundle identity of the exact generated receiver paired with this linker."""

    return v2_receiver_bundle_sha256(
        generate_v2_inflate_py().encode("utf-8"),
        generate_v2_inflate_sh().encode("utf-8"),
    )


def v2_receiver_consumer(section: V2Section) -> str:
    """Canonical receiver-consumer foreign key for a mutable v2 section."""

    try:
        return _SECTION_RECEIVER_CONSUMER[section]
    except KeyError as exc:
        raise AppliedActionLinkError(
            f"section {section.value} has no receiver-mutating consumer"
        ) from exc


def v2_archive_marginal_home_id(section: V2Section) -> str:
    """Typed whole-archive marginal for one v2 section replacement.

    This is deliberately not an independently owned section byte range: all
    three sections share the DEFLATE stream of one ``0.bin`` member.
    """

    try:
        return _SECTION_ARCHIVE_MARGINAL_HOME[section]
    except KeyError as exc:
        raise AppliedActionLinkError(
            f"section {section.value} has no counted archive marginal"
        ) from exc


def applied_action_receipt_sha256(receipt: AppliedActionReceipt) -> str:
    """Canonical content identity of one complete source receipt."""

    payload = json.dumps(
        receipt.as_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(payload)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AppliedActionLinkError(f"{name} must be a SHA-256 hex string")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise AppliedActionLinkError(f"{name} must be a SHA-256 hex string")
    return lowered


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppliedActionLinkError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class V2SectionReplacement:
    """Caller-supplied relocation binding for one counted codeword."""

    receipt_id: str
    byte_home_id: str
    section: V2Section
    receiver_consumer: str
    base_section_sha256: str
    candidate_section_sha256: str
    candidate_section_bytes: bytes

    def __post_init__(self) -> None:
        _require_text(self.receipt_id, "receipt_id")
        _require_text(self.byte_home_id, "byte_home_id")
        _require_text(self.receiver_consumer, "receiver_consumer")
        if not isinstance(self.section, V2Section):
            raise AppliedActionLinkError("section must be V2Section")
        _require_sha256(self.base_section_sha256, "base_section_sha256")
        _require_sha256(self.candidate_section_sha256, "candidate_section_sha256")
        if not isinstance(self.candidate_section_bytes, bytes):
            raise AppliedActionLinkError("candidate_section_bytes must be exact bytes")
        if _sha256(self.candidate_section_bytes) != self.candidate_section_sha256:
            raise AppliedActionLinkError("candidate section bytes differ from declared SHA-256")
        if self.base_section_sha256 == self.candidate_section_sha256:
            raise AppliedActionLinkError("section replacement must change physical bytes")
        if self.section is V2Section.MANIFEST:
            raise AppliedActionLinkError(
                "manifest-only relocation is metadata, not a receiver mutation"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": V2_SECTION_REPLACEMENT_SCHEMA,
            "receipt_id": self.receipt_id,
            "byte_home_id": self.byte_home_id,
            "section": self.section.value,
            "receiver_consumer": self.receiver_consumer,
            "base_section_sha256": self.base_section_sha256,
            "candidate_section_sha256": self.candidate_section_sha256,
            "candidate_section_bytes": len(self.candidate_section_bytes),
        }


@dataclass(frozen=True)
class V2PacketRelocationEntry:
    """One parse-back-proven logical-home to physical-section relocation."""

    receipt_id: str
    source_receipt_sha256: str
    action_id: str
    codeword_id: str
    byte_home_id: str
    physical_edge_id: str
    application_status: ApplicationStatus
    application_operator_id: str
    application_operator_version: str
    receiver_schema: str
    receiver_sha256: str
    r_chain_id: str
    authority_axis: str
    provenance_ref: str
    section: V2Section
    receiver_consumer: str
    base_section_sha256: str
    candidate_section_sha256: str
    base_section_bytes: int
    candidate_section_bytes: int
    individual_candidate_payload_sha256: str
    individual_candidate_archive_sha256: str
    individual_archive_delta_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("action_id", self.action_id),
            ("codeword_id", self.codeword_id),
            ("byte_home_id", self.byte_home_id),
            ("physical_edge_id", self.physical_edge_id),
            ("application_operator_id", self.application_operator_id),
            ("application_operator_version", self.application_operator_version),
            ("receiver_schema", self.receiver_schema),
            ("r_chain_id", self.r_chain_id),
            ("authority_axis", self.authority_axis),
            ("provenance_ref", self.provenance_ref),
            ("receiver_consumer", self.receiver_consumer),
        ):
            _require_text(value, name)
        if not isinstance(self.section, V2Section):
            raise AppliedActionLinkError("relocation section must be V2Section")
        for name, value in (
            ("base_section_sha256", self.base_section_sha256),
            ("candidate_section_sha256", self.candidate_section_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
            ("receiver_sha256", self.receiver_sha256),
            (
                "individual_candidate_payload_sha256",
                self.individual_candidate_payload_sha256,
            ),
            (
                "individual_candidate_archive_sha256",
                self.individual_candidate_archive_sha256,
            ),
        ):
            _require_sha256(value, name)
        if not isinstance(self.application_status, ApplicationStatus):
            raise AppliedActionLinkError("relocation application status differs")
        if self.application_status not in _MEASURED_STATUSES:
            raise AppliedActionLinkError("relocation application status is not measured")
        if self.byte_home_id != v2_archive_marginal_home_id(self.section):
            raise AppliedActionLinkError("relocation byte home is not the native v2 marginal")
        if self.receiver_consumer != v2_receiver_consumer(self.section):
            raise AppliedActionLinkError("relocation section/receiver consumer differs")
        if self.receiver_schema != V2_RECEIVER_SCHEMA:
            raise AppliedActionLinkError("relocation receiver schema differs")
        if self.receiver_sha256 != v2_receiver_sha256():
            raise AppliedActionLinkError("relocation receiver identity differs")
        if self.r_chain_id != V2_R_CHAIN_ID:
            raise AppliedActionLinkError("relocation R chain differs")
        for name, value in (
            ("base_section_bytes", self.base_section_bytes),
            ("candidate_section_bytes", self.candidate_section_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AppliedActionLinkError(f"{name} must be a non-negative integer")
        if isinstance(self.individual_archive_delta_bytes, bool) or not isinstance(
            self.individual_archive_delta_bytes, int
        ):
            raise AppliedActionLinkError("individual_archive_delta_bytes must be an integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": V2_PACKET_RELOCATION_ENTRY_SCHEMA,
            "receipt_id": self.receipt_id,
            "source_receipt_sha256": self.source_receipt_sha256,
            "action_id": self.action_id,
            "codeword_id": self.codeword_id,
            "byte_home_id": self.byte_home_id,
            "physical_edge_id": self.physical_edge_id,
            "application_status": self.application_status.value,
            "application_operator_id": self.application_operator_id,
            "application_operator_version": self.application_operator_version,
            "receiver_schema": self.receiver_schema,
            "receiver_sha256": self.receiver_sha256,
            "r_chain_id": self.r_chain_id,
            "authority_axis": self.authority_axis,
            "provenance_ref": self.provenance_ref,
            "section": self.section.value,
            "receiver_consumer": self.receiver_consumer,
            "base_section_sha256": self.base_section_sha256,
            "candidate_section_sha256": self.candidate_section_sha256,
            "base_section_bytes": self.base_section_bytes,
            "candidate_section_bytes": self.candidate_section_bytes,
            "individual_candidate_payload_sha256": self.individual_candidate_payload_sha256,
            "individual_candidate_archive_sha256": self.individual_candidate_archive_sha256,
            "individual_archive_delta_bytes": self.individual_archive_delta_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V2PacketRelocationEntry:
        if payload.get("schema") != V2_PACKET_RELOCATION_ENTRY_SCHEMA:
            raise AppliedActionLinkError("relocation entry schema differs")
        return cls(
            receipt_id=payload.get("receipt_id"),
            source_receipt_sha256=payload.get("source_receipt_sha256"),
            action_id=payload.get("action_id"),
            codeword_id=payload.get("codeword_id"),
            byte_home_id=payload.get("byte_home_id"),
            physical_edge_id=payload.get("physical_edge_id"),
            application_status=ApplicationStatus(payload.get("application_status")),
            application_operator_id=payload.get("application_operator_id"),
            application_operator_version=payload.get("application_operator_version"),
            receiver_schema=payload.get("receiver_schema"),
            receiver_sha256=payload.get("receiver_sha256"),
            r_chain_id=payload.get("r_chain_id"),
            authority_axis=payload.get("authority_axis"),
            provenance_ref=payload.get("provenance_ref"),
            section=V2Section(payload.get("section")),
            receiver_consumer=payload.get("receiver_consumer"),
            base_section_sha256=payload.get("base_section_sha256"),
            candidate_section_sha256=payload.get("candidate_section_sha256"),
            base_section_bytes=payload.get("base_section_bytes"),
            candidate_section_bytes=payload.get("candidate_section_bytes"),
            individual_candidate_payload_sha256=payload.get(
                "individual_candidate_payload_sha256"
            ),
            individual_candidate_archive_sha256=payload.get(
                "individual_candidate_archive_sha256"
            ),
            individual_archive_delta_bytes=payload.get("individual_archive_delta_bytes"),
        )


@dataclass(frozen=True)
class V2PacketLinkReceipt:
    """Exact byte identity of a linked packet; never a composed score claim."""

    schema: str
    base_payload_sha256: str
    candidate_payload_sha256: str
    base_archive_sha256: str
    candidate_archive_sha256: str
    base_archive_bytes: int
    candidate_archive_bytes: int
    exact_archive_delta_bytes: int
    individual_archive_delta_sum: int
    archive_interaction_bytes: int
    base_state_sha256: str | None
    ordered_parent_action_ids: tuple[str, ...]
    ordered_receipt_ids: tuple[str, ...]
    ordered_action_ids: tuple[str, ...]
    relocations: tuple[V2PacketRelocationEntry, ...]
    blockers: tuple[str, ...]
    research_only: bool = True
    promotion_eligible: bool = False
    score_claim: bool = False

    def __post_init__(self) -> None:
        if self.schema != V2_PACKET_LINK_RECEIPT_SCHEMA:
            raise AppliedActionLinkError("packet link receipt schema differs")
        for name, value in (
            ("base_payload_sha256", self.base_payload_sha256),
            ("candidate_payload_sha256", self.candidate_payload_sha256),
            ("base_archive_sha256", self.base_archive_sha256),
            ("candidate_archive_sha256", self.candidate_archive_sha256),
        ):
            _require_sha256(value, name)
        if self.base_state_sha256 is not None:
            _require_sha256(self.base_state_sha256, "base_state_sha256")
        for name, value in (
            ("base_archive_bytes", self.base_archive_bytes),
            ("candidate_archive_bytes", self.candidate_archive_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AppliedActionLinkError(f"{name} must be a non-negative integer")
        for name, value in (
            ("exact_archive_delta_bytes", self.exact_archive_delta_bytes),
            ("individual_archive_delta_sum", self.individual_archive_delta_sum),
            ("archive_interaction_bytes", self.archive_interaction_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AppliedActionLinkError(f"{name} must be an exact integer")
        if self.exact_archive_delta_bytes != self.candidate_archive_bytes - self.base_archive_bytes:
            raise AppliedActionLinkError("exact archive delta does not reconcile")
        if self.archive_interaction_bytes != (
            self.exact_archive_delta_bytes - self.individual_archive_delta_sum
        ):
            raise AppliedActionLinkError("archive interaction bytes do not reconcile")
        if not isinstance(self.relocations, tuple):
            raise AppliedActionLinkError("packet link relocations must be a tuple")
        if any(not isinstance(value, V2PacketRelocationEntry) for value in self.relocations):
            raise AppliedActionLinkError("packet link contains an invalid relocation")
        if not self.relocations:
            raise AppliedActionLinkError("packet link receipt must contain a relocation")
        if self.individual_archive_delta_sum != sum(
            row.individual_archive_delta_bytes for row in self.relocations
        ):
            raise AppliedActionLinkError(
                "individual archive delta sum differs from relocations"
            )
        if not isinstance(self.blockers, tuple):
            raise AppliedActionLinkError("packet link blockers must be a tuple")
        if any(not isinstance(value, str) or not value for value in self.blockers):
            raise AppliedActionLinkError("packet link blockers contain an invalid value")
        for name, values in (
            ("ordered_parent_action_ids", self.ordered_parent_action_ids),
            ("ordered_receipt_ids", self.ordered_receipt_ids),
            ("ordered_action_ids", self.ordered_action_ids),
        ):
            if not isinstance(values, tuple):
                raise AppliedActionLinkError(f"{name} must be a tuple")
            if any(not isinstance(value, str) or not value for value in values):
                raise AppliedActionLinkError(f"{name} contains an invalid identity")
        if len(self.ordered_parent_action_ids) != len(set(self.ordered_parent_action_ids)):
            raise AppliedActionLinkError("ordered parent action identities must be unique")
        if len(self.ordered_receipt_ids) != len(set(self.ordered_receipt_ids)):
            raise AppliedActionLinkError("ordered receipt identities must be unique")
        if len(self.ordered_action_ids) != len(set(self.ordered_action_ids)):
            raise AppliedActionLinkError("ordered action identities must be unique")
        if len(self.ordered_receipt_ids) != len(self.relocations):
            raise AppliedActionLinkError("ordered receipt count differs from relocations")
        if len(self.ordered_action_ids) != len(self.relocations):
            raise AppliedActionLinkError("ordered action count differs from relocations")
        if self.ordered_receipt_ids != tuple(row.receipt_id for row in self.relocations):
            raise AppliedActionLinkError("ordered receipt identities differ from relocations")
        if self.ordered_action_ids != tuple(row.action_id for row in self.relocations):
            raise AppliedActionLinkError("ordered action identities differ from relocations")
        sections = tuple(row.section for row in self.relocations)
        if len(sections) != len(set(sections)):
            raise AppliedActionLinkError("relocations overlap one physical section")
        homes = tuple(row.byte_home_id for row in self.relocations)
        if len(homes) != len(set(homes)):
            raise AppliedActionLinkError("relocations duplicate one counted byte home")
        if self.research_only is not True or self.promotion_eligible or self.score_claim:
            raise AppliedActionLinkError("packet linker output is research-only false authority")
        if len(self.relocations) > 1 and "COMPOSED_SCORE_EFFECT_REMEASUREMENT_REQUIRED" not in self.blockers:
            raise AppliedActionLinkError("multi-action link must carry the score remeasurement blocker")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_payload_sha256": self.base_payload_sha256,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "base_archive_sha256": self.base_archive_sha256,
            "candidate_archive_sha256": self.candidate_archive_sha256,
            "base_archive_bytes": self.base_archive_bytes,
            "candidate_archive_bytes": self.candidate_archive_bytes,
            "exact_archive_delta_bytes": self.exact_archive_delta_bytes,
            "individual_archive_delta_sum": self.individual_archive_delta_sum,
            "archive_interaction_bytes": self.archive_interaction_bytes,
            "base_state_sha256": self.base_state_sha256,
            "ordered_parent_action_ids": list(self.ordered_parent_action_ids),
            "ordered_receipt_ids": list(self.ordered_receipt_ids),
            "ordered_action_ids": list(self.ordered_action_ids),
            "relocations": [row.as_dict() for row in self.relocations],
            "blockers": list(self.blockers),
            "research_only": self.research_only,
            "promotion_eligible": self.promotion_eligible,
            "score_claim": self.score_claim,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V2PacketLinkReceipt:
        return cls(
            schema=payload.get("schema"),
            base_payload_sha256=payload.get("base_payload_sha256"),
            candidate_payload_sha256=payload.get("candidate_payload_sha256"),
            base_archive_sha256=payload.get("base_archive_sha256"),
            candidate_archive_sha256=payload.get("candidate_archive_sha256"),
            base_archive_bytes=payload.get("base_archive_bytes"),
            candidate_archive_bytes=payload.get("candidate_archive_bytes"),
            exact_archive_delta_bytes=payload.get("exact_archive_delta_bytes"),
            individual_archive_delta_sum=payload.get("individual_archive_delta_sum"),
            archive_interaction_bytes=payload.get("archive_interaction_bytes"),
            base_state_sha256=payload.get("base_state_sha256"),
            ordered_parent_action_ids=tuple(payload.get("ordered_parent_action_ids") or ()),
            ordered_receipt_ids=tuple(payload.get("ordered_receipt_ids") or ()),
            ordered_action_ids=tuple(payload.get("ordered_action_ids") or ()),
            relocations=tuple(
                V2PacketRelocationEntry.from_dict(row)
                for row in payload.get("relocations") or ()
            ),
            blockers=tuple(payload.get("blockers") or ()),
            research_only=payload.get("research_only"),
            promotion_eligible=payload.get("promotion_eligible"),
            score_claim=payload.get("score_claim"),
        )


@dataclass(frozen=True)
class LinkedV2Packet:
    """In-memory exact output of a successful link."""

    payload_bytes: bytes
    archive_bytes: bytes
    receipt: V2PacketLinkReceipt


@dataclass(frozen=True)
class V2PacketLinkAttempt:
    """Deterministic success/blocker envelope for CLI and campaign consumers."""

    status: str
    input_receipt_ids: tuple[str, ...]
    receipt: V2PacketLinkReceipt | None
    blockers: tuple[str, ...]
    input_manifest_sha256: str | None = None
    research_only: bool = True
    promotion_eligible: bool = False
    score_claim: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"LINKED", "BLOCKED"}:
            raise AppliedActionLinkError("link attempt status differs")
        if (self.status == "LINKED") != (self.receipt is not None):
            raise AppliedActionLinkError("link attempt status/receipt disagree")
        if self.status == "BLOCKED" and not self.blockers:
            raise AppliedActionLinkError("blocked link attempt requires blockers")
        if not isinstance(self.input_receipt_ids, tuple):
            raise AppliedActionLinkError("link attempt input identities must be a tuple")
        if any(not isinstance(value, str) or not value for value in self.input_receipt_ids):
            raise AppliedActionLinkError("link attempt contains an invalid input identity")
        if len(self.input_receipt_ids) != len(set(self.input_receipt_ids)):
            raise AppliedActionLinkError("link attempt input identities must be unique")
        if not isinstance(self.blockers, tuple):
            raise AppliedActionLinkError("link attempt blockers must be a tuple")
        if any(not isinstance(value, str) or not value for value in self.blockers):
            raise AppliedActionLinkError("link attempt contains an invalid blocker")
        if self.receipt is not None:
            if set(self.input_receipt_ids) != set(self.receipt.ordered_receipt_ids):
                raise AppliedActionLinkError("link attempt identities differ from link receipt")
            if self.blockers != self.receipt.blockers:
                raise AppliedActionLinkError("link attempt blockers differ from link receipt")
        if self.input_manifest_sha256 is not None:
            _require_sha256(self.input_manifest_sha256, "input_manifest_sha256")
        if self.research_only is not True or self.promotion_eligible or self.score_claim:
            raise AppliedActionLinkError("link attempt is research-only false authority")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": V2_PACKET_LINK_ATTEMPT_SCHEMA,
            "status": self.status,
            "input_receipt_ids": list(self.input_receipt_ids),
            "receipt": None if self.receipt is None else self.receipt.as_dict(),
            "blockers": list(self.blockers),
            "input_manifest_sha256": self.input_manifest_sha256,
            "research_only": self.research_only,
            "promotion_eligible": self.promotion_eligible,
            "score_claim": self.score_claim,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V2PacketLinkAttempt:
        if payload.get("schema") != V2_PACKET_LINK_ATTEMPT_SCHEMA:
            raise AppliedActionLinkError("packet link attempt schema differs")
        raw_receipt = payload.get("receipt")
        if raw_receipt is not None and not isinstance(raw_receipt, Mapping):
            raise AppliedActionLinkError("packet link attempt receipt must be an object")
        return cls(
            status=payload.get("status"),
            input_receipt_ids=tuple(payload.get("input_receipt_ids") or ()),
            receipt=(
                None
                if raw_receipt is None
                else V2PacketLinkReceipt.from_dict(raw_receipt)
            ),
            blockers=tuple(payload.get("blockers") or ()),
            input_manifest_sha256=payload.get("input_manifest_sha256"),
            research_only=payload.get("research_only"),
            promotion_eligible=payload.get("promotion_eligible"),
            score_claim=payload.get("score_claim"),
        )


def _section_map(blob: bytes) -> dict[V2Section, bytes]:
    parsed = unpack_v2_sections(blob)
    return {
        V2Section.STORE: parsed.store_blob,
        V2Section.RESIDUAL: parsed.residual_inr_blob,
        V2Section.POSE: parsed.pose_sidecar_blob,
        V2Section.MANIFEST: parsed.manifest_bytes,
    }


def _pack_sections(sections: Mapping[V2Section, bytes]) -> bytes:
    return pack_v2_archive(
        sections[V2Section.STORE],
        sections[V2Section.RESIDUAL],
        sections[V2Section.POSE],
        sections[V2Section.MANIFEST],
    )


def _require_common_identity(receipts: Sequence[AppliedActionReceipt]) -> None:
    first = receipts[0]
    for receipt in receipts[1:]:
        if receipt.base_archive_sha256 != first.base_archive_sha256:
            raise AppliedActionLinkError("actions do not share one base archive identity")
        if receipt.base_payload_sha256 != first.base_payload_sha256:
            raise AppliedActionLinkError("actions do not share one base payload identity")
        if receipt.base_state_sha256 != first.base_state_sha256:
            raise AppliedActionLinkError("actions do not share one base state identity")
        if receipt.ordered_parent_action_ids != first.ordered_parent_action_ids:
            raise AppliedActionLinkError("actions do not share ordered parent identity")
        if receipt.receiver_schema != first.receiver_schema:
            raise AppliedActionLinkError("actions do not share one receiver schema")
        if receipt.receiver_sha256 != first.receiver_sha256:
            raise AppliedActionLinkError("actions do not share one receiver identity")
        if receipt.r_chain_id != first.r_chain_id:
            raise AppliedActionLinkError("actions do not share one receiver R chain")


def link_v2_applied_actions(
    *,
    base_payload: bytes,
    receipts: Sequence[AppliedActionReceipt],
    replacements: Sequence[V2SectionReplacement],
) -> LinkedV2Packet:
    """Relocate measured actions into one exact v2 packet.

    Each receipt is first replayed alone from ``base_payload``.  Its resulting
    payload and deterministic archive must match the receipt's candidate hashes.
    This is the foreign-key proof that the supplied codeword bytes are the same
    physical edge that was scored.
    """

    if not isinstance(base_payload, bytes):
        raise AppliedActionLinkError("base_payload must be exact bytes")
    receipt_rows = tuple(receipts)
    replacement_rows = tuple(replacements)
    if not receipt_rows:
        raise AppliedActionLinkError("at least one applied-action receipt is required")
    if len(receipt_rows) != len(replacement_rows):
        raise AppliedActionLinkError("every receipt requires exactly one section replacement")
    if any(not isinstance(row, AppliedActionReceipt) for row in receipt_rows):
        raise AppliedActionLinkError("receipts must be AppliedActionReceipt values")
    if any(not isinstance(row, V2SectionReplacement) for row in replacement_rows):
        raise AppliedActionLinkError("replacements must be V2SectionReplacement values")

    by_receipt = {row.receipt_id: row for row in receipt_rows}
    if len(by_receipt) != len(receipt_rows):
        raise AppliedActionLinkError("receipt identities must be unique")
    replacement_by_receipt = {row.receipt_id: row for row in replacement_rows}
    if len(replacement_by_receipt) != len(replacement_rows):
        raise AppliedActionLinkError("replacement receipt identities must be unique")
    if set(by_receipt) != set(replacement_by_receipt):
        raise AppliedActionLinkError("receipt and replacement identities differ")
    if len({row.action_id for row in receipt_rows}) != len(receipt_rows):
        raise AppliedActionLinkError("action identities must be unique")
    if len({row.codeword_id for row in receipt_rows}) != len(receipt_rows):
        raise AppliedActionLinkError("codeword identities must be unique")
    if len({row.physical_edge_id for row in receipt_rows}) != len(receipt_rows):
        raise AppliedActionLinkError("physical edge identities must be unique")

    _require_common_identity(receipt_rows)
    validate_v2_receiver_payload(base_payload)
    base_sections = _section_map(base_payload)
    base_payload_sha256 = _sha256(base_payload)
    base_archive = build_v2_archive_zip_bytes(base_payload)
    base_archive_sha256 = _sha256(base_archive)
    first = receipt_rows[0]
    canonical_receiver_sha256 = v2_receiver_sha256()
    if first.receiver_schema != V2_RECEIVER_SCHEMA:
        raise AppliedActionLinkError("receipt receiver schema is not the canonical v2 receiver")
    if first.receiver_sha256 != canonical_receiver_sha256:
        raise AppliedActionLinkError("receipt receiver SHA-256 is not the generated v2 receiver")
    if first.r_chain_id != V2_R_CHAIN_ID:
        raise AppliedActionLinkError("receipt R chain is not the canonical v2 uint8 chain")
    if first.base_payload_sha256 is None:
        raise AppliedActionLinkError("linking requires an exact base payload SHA-256")
    if first.base_payload_sha256 != base_payload_sha256:
        raise AppliedActionLinkError("base payload bytes differ from applied-action custody")
    if first.base_archive_sha256 != base_archive_sha256:
        raise AppliedActionLinkError("base archive bytes differ from applied-action custody")

    ordered = sorted(
        receipt_rows,
        key=lambda row: (
            _SECTION_ORDER[replacement_by_receipt[row.receipt_id].section],
            row.stream_home.byte_home_id,
            row.receipt_id,
        ),
    )
    sections = [replacement_by_receipt[row.receipt_id].section for row in ordered]
    if len(sections) != len(set(sections)):
        raise AppliedActionLinkError(
            "multiple logical homes target one physical v2 section; a nested receiver grammar is owed"
        )
    if len({row.stream_home.byte_home_id for row in ordered}) != len(ordered):
        raise AppliedActionLinkError("a counted byte home cannot own multiple codewords")

    entries: list[V2PacketRelocationEntry] = []
    for receipt in ordered:
        replacement = replacement_by_receipt[receipt.receipt_id]
        if receipt.blockers or receipt.action_effect.blockers:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} carries unresolved source blockers"
            )
        if receipt.status not in _MEASURED_STATUSES:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} is not a measured physical transition"
            )
        delta_score = receipt.action_effect.delta_score_total
        if delta_score is None or not math.isfinite(float(delta_score)):
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} lacks a finite joint score transition"
            )
        if receipt.changed_uint8_count <= 0:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} has no receiver-visible uint8 mutation"
            )
        if receipt.candidate_payload_sha256 is None:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} lacks candidate payload identity"
            )
        if receipt.stream_home.byte_home_id != replacement.byte_home_id:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} byte-home foreign key differs"
            )
        expected_home_id = v2_archive_marginal_home_id(replacement.section)
        if replacement.byte_home_id != expected_home_id:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} is not bound to the native v2 archive marginal"
            )
        if receipt.stream_home.receiver_consumer != replacement.receiver_consumer:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} receiver consumer differs"
            )
        expected_consumer = v2_receiver_consumer(replacement.section)
        if replacement.receiver_consumer != expected_consumer:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} section/receiver consumer foreign key differs"
            )
        base_section = base_sections[replacement.section]
        if _sha256(base_section) != replacement.base_section_sha256:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} base section identity differs"
            )

        single_sections = dict(base_sections)
        single_sections[replacement.section] = replacement.candidate_section_bytes
        single_payload = _pack_sections(single_sections)
        # Parse every receiver-consumed section before any identity claim.
        validate_v2_receiver_payload(single_payload)
        single_archive = build_v2_archive_zip_bytes(single_payload)
        single_payload_sha256 = _sha256(single_payload)
        single_archive_sha256 = _sha256(single_archive)
        if receipt.action_effect.old_bytes != len(base_archive):
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} base archive byte endpoint differs"
            )
        if receipt.action_effect.new_bytes != len(single_archive):
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} candidate archive byte endpoint differs"
            )
        if receipt.stream_home.bytes_before != len(base_archive):
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} archive-marginal base bytes differ"
            )
        if receipt.stream_home.bytes_after != len(single_archive):
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} archive-marginal candidate bytes differ"
            )
        if receipt.candidate_payload_sha256 != single_payload_sha256:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} candidate payload identity differs"
            )
        if receipt.candidate_archive_sha256 != single_archive_sha256:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} candidate archive identity differs"
            )
        archive_delta = len(single_archive) - len(base_archive)
        if archive_delta != receipt.stream_home.delta_bytes:
            raise AppliedActionLinkError(
                f"receipt {receipt.receipt_id} exact archive delta differs from byte home"
            )
        entries.append(
            V2PacketRelocationEntry(
                receipt_id=receipt.receipt_id,
                source_receipt_sha256=applied_action_receipt_sha256(receipt),
                action_id=receipt.action_id,
                codeword_id=receipt.codeword_id,
                byte_home_id=receipt.stream_home.byte_home_id,
                physical_edge_id=receipt.physical_edge_id,
                application_status=receipt.status,
                application_operator_id=receipt.application_operator_id,
                application_operator_version=receipt.application_operator_version,
                receiver_schema=receipt.receiver_schema,
                receiver_sha256=receipt.receiver_sha256,
                r_chain_id=receipt.r_chain_id,
                authority_axis=receipt.authority_axis,
                provenance_ref=receipt.provenance_ref,
                section=replacement.section,
                receiver_consumer=replacement.receiver_consumer,
                base_section_sha256=replacement.base_section_sha256,
                candidate_section_sha256=replacement.candidate_section_sha256,
                base_section_bytes=len(base_section),
                candidate_section_bytes=len(replacement.candidate_section_bytes),
                individual_candidate_payload_sha256=single_payload_sha256,
                individual_candidate_archive_sha256=single_archive_sha256,
                individual_archive_delta_bytes=archive_delta,
            )
        )

    composed_sections = dict(base_sections)
    for receipt in ordered:
        replacement = replacement_by_receipt[receipt.receipt_id]
        composed_sections[replacement.section] = replacement.candidate_section_bytes
    candidate_payload = _pack_sections(composed_sections)
    validate_v2_receiver_payload(candidate_payload)
    parsed_candidate = unpack_v2_sections(candidate_payload)
    if _pack_sections(
        {
            V2Section.STORE: parsed_candidate.store_blob,
            V2Section.RESIDUAL: parsed_candidate.residual_inr_blob,
            V2Section.POSE: parsed_candidate.pose_sidecar_blob,
            V2Section.MANIFEST: parsed_candidate.manifest_bytes,
        }
    ) != candidate_payload:
        raise AppliedActionLinkError("candidate packet failed canonical parse/re-emit")
    candidate_archive = build_v2_archive_zip_bytes(candidate_payload)
    exact_delta = len(candidate_archive) - len(base_archive)
    individual_sum = sum(row.individual_archive_delta_bytes for row in entries)
    blockers = (
        ("COMPOSED_SCORE_EFFECT_REMEASUREMENT_REQUIRED",)
        if len(entries) > 1
        else ()
    )
    receipt = V2PacketLinkReceipt(
        schema=V2_PACKET_LINK_RECEIPT_SCHEMA,
        base_payload_sha256=base_payload_sha256,
        candidate_payload_sha256=_sha256(candidate_payload),
        base_archive_sha256=base_archive_sha256,
        candidate_archive_sha256=_sha256(candidate_archive),
        base_archive_bytes=len(base_archive),
        candidate_archive_bytes=len(candidate_archive),
        exact_archive_delta_bytes=exact_delta,
        individual_archive_delta_sum=individual_sum,
        archive_interaction_bytes=exact_delta - individual_sum,
        base_state_sha256=first.base_state_sha256,
        ordered_parent_action_ids=first.ordered_parent_action_ids,
        ordered_receipt_ids=tuple(row.receipt_id for row in entries),
        ordered_action_ids=tuple(row.action_id for row in entries),
        relocations=tuple(entries),
        blockers=blockers,
    )
    return LinkedV2Packet(
        payload_bytes=candidate_payload,
        archive_bytes=candidate_archive,
        receipt=receipt,
    )


def try_link_v2_applied_actions(
    *,
    base_payload: bytes,
    receipts: Sequence[AppliedActionReceipt],
    replacements: Sequence[V2SectionReplacement],
) -> tuple[V2PacketLinkAttempt, LinkedV2Packet | None]:
    """Fail-closed envelope that preserves an exact blocker instead of faking a packet."""

    input_ids = tuple(
        sorted(
            row.receipt_id
            for row in receipts
            if isinstance(row, AppliedActionReceipt)
        )
    )
    try:
        linked = link_v2_applied_actions(
            base_payload=base_payload,
            receipts=receipts,
            replacements=replacements,
        )
    except (AppliedActionLinkError, TypeError, ValueError) as exc:
        attempt = V2PacketLinkAttempt(
            status="BLOCKED",
            input_receipt_ids=input_ids,
            receipt=None,
            blockers=(f"{type(exc).__name__}:{exc}",),
        )
        return attempt, None
    return (
        V2PacketLinkAttempt(
            status="LINKED",
            input_receipt_ids=input_ids,
            receipt=linked.receipt,
            blockers=linked.receipt.blockers,
        ),
        linked,
    )


__all__ = [
    "V2_PACKET_LINK_ATTEMPT_SCHEMA",
    "V2_PACKET_LINK_RECEIPT_SCHEMA",
    "V2_PACKET_RELOCATION_ENTRY_SCHEMA",
    "V2_RECEIVER_SCHEMA",
    "V2_R_CHAIN_ID",
    "V2_SECTION_REPLACEMENT_SCHEMA",
    "AppliedActionLinkError",
    "LinkedV2Packet",
    "V2PacketLinkAttempt",
    "V2PacketLinkReceipt",
    "V2PacketRelocationEntry",
    "V2Section",
    "V2SectionReplacement",
    "applied_action_receipt_sha256",
    "link_v2_applied_actions",
    "try_link_v2_applied_actions",
    "v2_archive_marginal_home_id",
    "v2_receiver_bundle_sha256",
    "v2_receiver_consumer",
    "v2_receiver_file_sha256s",
    "v2_receiver_sha256",
]
