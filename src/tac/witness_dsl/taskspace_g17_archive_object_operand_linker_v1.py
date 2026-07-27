# SPDX-License-Identifier: MIT
"""Fail-closed audit for the missing G17 archive-object receiver incidence.

G64 proves liveness of one logical conditional operand.  G17's canonical
placement manifest can also describe a physical byte range shared by multiple
logical owners.  It cannot, however, describe multiple owner-specific receiver
operations for one shared physical group: ``G17PhysicalCodingGroupV1`` has one
``receiver_consumer`` and one ``receiver_operation`` for the entire group.

That missing incidence type matters for a compressed ZIP.  The only honest
physical range is the whole archive, shared by all counted owners.  Recasting
that group's sole receiver as G64 would falsely say G64 consumes every co-coded
owner.  Splitting compressed ZIP bytes into owner-local ranges would be equally
false.  This module therefore verifies exact archive/builder/A-descriptor
custody, emits the precise ontology blocker, and refuses to claim the logical
operand has been linked to an exact archive object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from tac.witness_dsl.taskspace_conditional_fullframe_receiver_operation_v1 import (
    ARCHIVE_RANGE_LINK_BLOCKER,
    G17ConditionalFullFrameY0GivenY1OperationV1,
)
from tac.witness_dsl.taskspace_g17_production_envelope import (
    G17AStrictNestedParser,
    G17GStrictNestedParser,
    G17ProductionArchiveBuildV1,
    ParsedG17ProductionArchiveV1,
    build_g17_production_archive,
    parse_g17_production_archive,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ChronologicalPosePreimageV1,
    G17CompilerPlacementManifestV1,
)

if TYPE_CHECKING:
    import numpy as np

SCHEMA: Final = "tac.g17_archive_object_operand_link_blocker.v1"
OWNER_RECEIVER_INCIDENCE_BLOCKER: Final = "G17_SHARED_PHYSICAL_GROUP_OWNER_SPECIFIC_RECEIVER_INCIDENCE_TYPE_OWED"
REQUIRED_ADDITIVE_TYPE: Final = (
    "physical_group_id + logical_owner_id + receiver_consumer + receiver_operation incidence"
)


class G17ArchiveObjectOperandLinkError(ValueError):
    """The exact archive-object link is not representable by current types."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class G17ArchiveObjectOperandLinkBlockerReceiptV1:
    schema: Literal["tac.g17_archive_object_operand_link_blocker.v1"]
    baseline_archive_sha256: str
    baseline_archive_bytes: int
    baseline_member_sha256: str
    baseline_member_bytes: int
    baseline_outer_encoding: str
    placement_manifest_sha256: str
    conditional_owner_id: str
    conditional_operand_sha256: str
    conditional_operand_bytes: int
    a_descriptor_index: int
    exact_archive_builder_custody: Literal[True]
    deterministic_double_parse: Literal[True]
    conditional_operand_unique_in_a: Literal[True]
    honest_physical_group_scope: Literal["WHOLE_ARCHIVE_SHARED_MANY_TO_MANY"]
    local_zip_operand_owner_ranges_claimed: Literal[0]
    archive_object_receiver_liveness_proven: Literal[False]
    unresolved_blocker: Literal["G17_SHARED_PHYSICAL_GROUP_OWNER_SPECIFIC_RECEIVER_INCIDENCE_TYPE_OWED"]
    required_additive_type: str
    prior_g64_blocker_retained: Literal["G17_CONDITIONAL_LOGICAL_OPERAND_TO_ARCHIVE_RANGE_LINK_OWED"]
    hidden_or_direct_plane_inputs: Literal[0]
    scorer_teacher_target_inputs: Literal[0]
    research_only: Literal[True]
    candidate_claim: Literal[False]
    score_claim: Literal[False]
    pointer_moved: Literal[False]

    def as_dict(self) -> dict[str, object]:
        return {
            "a_descriptor_index": self.a_descriptor_index,
            "archive_object_receiver_liveness_proven": (self.archive_object_receiver_liveness_proven),
            "baseline_archive_bytes": self.baseline_archive_bytes,
            "baseline_archive_sha256": self.baseline_archive_sha256,
            "baseline_member_bytes": self.baseline_member_bytes,
            "baseline_member_sha256": self.baseline_member_sha256,
            "baseline_outer_encoding": self.baseline_outer_encoding,
            "candidate_claim": self.candidate_claim,
            "conditional_operand_bytes": self.conditional_operand_bytes,
            "conditional_operand_sha256": self.conditional_operand_sha256,
            "conditional_operand_unique_in_a": self.conditional_operand_unique_in_a,
            "conditional_owner_id": self.conditional_owner_id,
            "deterministic_double_parse": self.deterministic_double_parse,
            "exact_archive_builder_custody": self.exact_archive_builder_custody,
            "hidden_or_direct_plane_inputs": self.hidden_or_direct_plane_inputs,
            "honest_physical_group_scope": self.honest_physical_group_scope,
            "local_zip_operand_owner_ranges_claimed": (self.local_zip_operand_owner_ranges_claimed),
            "placement_manifest_sha256": self.placement_manifest_sha256,
            "pointer_moved": self.pointer_moved,
            "prior_g64_blocker_retained": self.prior_g64_blocker_retained,
            "required_additive_type": self.required_additive_type,
            "research_only": self.research_only,
            "schema": self.schema,
            "score_claim": self.score_claim,
            "scorer_teacher_target_inputs": self.scorer_teacher_target_inputs,
            "unresolved_blocker": self.unresolved_blocker,
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class G17ArchiveObjectOperandLinkBlockerProofV1:
    receipt: G17ArchiveObjectOperandLinkBlockerReceiptV1
    baseline_build: G17ProductionArchiveBuildV1
    baseline_parsed: ParsedG17ProductionArchiveV1


def audit_g17_conditional_operand_archive_link_blocker(
    *,
    archive_bytes: bytes,
    placement_manifest: G17CompilerPlacementManifestV1,
    conditional_operation: G17ConditionalFullFrameY0GivenY1OperationV1,
    a_active_parser: G17AStrictNestedParser,
    g_active_parser: G17GStrictNestedParser | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> G17ArchiveObjectOperandLinkBlockerProofV1:
    """Verify exact custody and classify why archive-object liveness is owed."""

    if (
        type(archive_bytes) is not bytes
        or type(placement_manifest) is not G17CompilerPlacementManifestV1
        or type(conditional_operation) is not G17ConditionalFullFrameY0GivenY1OperationV1
        or conditional_operation.placement_manifest is not placement_manifest
    ):
        raise G17ArchiveObjectOperandLinkError("audit inputs lack one exact operation/manifest/archive binding")
    if not callable(a_active_parser):
        raise G17ArchiveObjectOperandLinkError("G17_CONDITIONAL_A_SECTION_STRICT_PARSER_OWED")
    if placement_manifest.exact_archive_bytes != archive_bytes:
        raise G17ArchiveObjectOperandLinkError("placement manifest differs from exact archive object")
    parsed = parse_g17_production_archive(
        archive_bytes,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
        max_member_bytes=max_member_bytes,
    )
    reparsed = parse_g17_production_archive(
        archive_bytes,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
        max_member_bytes=max_member_bytes,
    )
    if parsed != reparsed:
        raise G17ArchiveObjectOperandLinkError("baseline archive parse-back is nondeterministic")
    build = build_g17_production_archive(
        p_section=parsed.p_section,
        g_section=parsed.g_section,
        a_section=parsed.a_section,
        terminal_section=parsed.terminal_section,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
        max_member_bytes=max_member_bytes,
    )
    if build.selected.outer.archive_bytes != archive_bytes:
        raise G17ArchiveObjectOperandLinkError("exact archive is not under the canonical STORE/DEFLATE builder")
    if (
        placement_manifest.member_name != parsed.outer.member_name
        or placement_manifest.exact_member_bytes != parsed.outer.member_bytes
    ):
        raise G17ArchiveObjectOperandLinkError("placement manifest differs from reparsed production member")
    owners = {
        id(row.logical_owner): row.logical_owner
        for row in placement_manifest.records
        if row.logical_owner.owner_id == conditional_operation.conditional_pose_owner_id
    }
    if len(owners) != 1:
        raise G17ArchiveObjectOperandLinkError("conditional owner is absent or aliases multiple logical objects")
    owner = next(iter(owners.values()))
    if type(owner.value) is not G17ChronologicalPosePreimageV1:
        raise G17ArchiveObjectOperandLinkError("conditional owner is not exact chronological-pose bytes")
    operand = owner.value.exact_bytes
    section_counts = (
        parsed.p_section.count(operand),
        parsed.g_section.count(operand),
        parsed.a_section.count(operand),
        parsed.terminal_section.count(operand),
    )
    descriptor_matches = tuple(
        index for index, descriptor in enumerate(parsed.a_packet.descriptors) if descriptor.payload.count(operand) == 1
    )
    if section_counts != (0, 0, 1, 0) or len(descriptor_matches) != 1:
        raise G17ArchiveObjectOperandLinkError("conditional operand is not uniquely linked to one active A descriptor")
    receipt = G17ArchiveObjectOperandLinkBlockerReceiptV1(
        schema=SCHEMA,
        baseline_archive_sha256=_sha256(archive_bytes),
        baseline_archive_bytes=len(archive_bytes),
        baseline_member_sha256=_sha256(parsed.outer.member_bytes),
        baseline_member_bytes=len(parsed.outer.member_bytes),
        baseline_outer_encoding=parsed.outer.encoding.value,
        placement_manifest_sha256=placement_manifest.manifest_sha256,
        conditional_owner_id=conditional_operation.conditional_pose_owner_id,
        conditional_operand_sha256=_sha256(operand),
        conditional_operand_bytes=len(operand),
        a_descriptor_index=descriptor_matches[0],
        exact_archive_builder_custody=True,
        deterministic_double_parse=True,
        conditional_operand_unique_in_a=True,
        honest_physical_group_scope="WHOLE_ARCHIVE_SHARED_MANY_TO_MANY",
        local_zip_operand_owner_ranges_claimed=0,
        archive_object_receiver_liveness_proven=False,
        unresolved_blocker=OWNER_RECEIVER_INCIDENCE_BLOCKER,
        required_additive_type=REQUIRED_ADDITIVE_TYPE,
        prior_g64_blocker_retained=ARCHIVE_RANGE_LINK_BLOCKER,
        hidden_or_direct_plane_inputs=0,
        scorer_teacher_target_inputs=0,
        research_only=True,
        candidate_claim=False,
        score_claim=False,
        pointer_moved=False,
    )
    return G17ArchiveObjectOperandLinkBlockerProofV1(
        receipt=receipt,
        baseline_build=build,
        baseline_parsed=parsed,
    )


def link_g17_conditional_operand_archive_object(
    *,
    archive_bytes: bytes,
    placement_manifest: G17CompilerPlacementManifestV1,
    conditional_operation: G17ConditionalFullFrameY0GivenY1OperationV1,
    exact_decoded_y1: np.ndarray,
    a_active_parser: G17AStrictNestedParser,
    g_active_parser: G17GStrictNestedParser | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> NoReturn:
    """Refuse the false link until owner-specific receiver incidence exists."""

    del exact_decoded_y1
    proof = audit_g17_conditional_operand_archive_link_blocker(
        archive_bytes=archive_bytes,
        placement_manifest=placement_manifest,
        conditional_operation=conditional_operation,
        a_active_parser=a_active_parser,
        g_active_parser=g_active_parser,
        max_member_bytes=max_member_bytes,
    )
    raise G17ArchiveObjectOperandLinkError(
        f"{proof.receipt.unresolved_blocker}: {proof.receipt.required_additive_type}"
    )
