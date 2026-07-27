# SPDX-License-Identifier: MIT
"""Fail-closed tests for G65's missing owner/receiver incidence type."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_conditional_fullframe_receiver_operation_v1 import (
    COUNTED_PAYLOAD_CLASS,
    OPERATION_ID,
    PARAMETER_SPELLING_FORMAT,
    RECEIVER_CONSUMER_ID,
    G17ConditionalFullFrameY0GivenY1OperationV1,
    learned_quotient_decoder_source_sha256,
)
from tac.witness_dsl.taskspace_g17_archive_object_operand_linker_v1 import (
    OWNER_RECEIVER_INCIDENCE_BLOCKER,
    G17ArchiveObjectOperandLinkError,
    audit_g17_conditional_operand_archive_link_blocker,
    link_g17_conditional_operand_archive_object,
)
from tac.witness_dsl.taskspace_g17_production_envelope import (
    G17AActiveNestedV1,
    G17ADescriptorV1,
    G17AFamily,
    G17AMode,
    G17PopulationLayout,
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_production_archive,
    build_g17_terminal_envelope,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    _parse_learned_payload,
    build_learned_irreducible_quotient_factor,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ArtifactClassV1,
    G17ChronologicalPosePreimageV1,
    G17CompilerPlacementManifestV1,
    G17CompilerPlacementRecordV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17ParameterSpellingIdentityV1,
    G17PhysicalCodingGroupV1,
    G17PlacementClassV1,
    G17PopulationSharingV1,
    G17RecursionCoordinateV1,
    G17RecursionNamespaceV1,
    G17RuntimeDependencyFileV1,
    G17RuntimeFileScopeV1,
    G17ScientificRoleV1,
    G17SemanticStreamRoleV1,
)


def _decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    values = np.frombuffer(counted_operand[-8:], dtype=np.uint8)
    output = exact_y1.copy(order="C")
    output.reshape(-1)[:8] ^= np.roll(values, -(source_pair_id - 17))
    return np.ascontiguousarray(output)


def _a_parser(
    payload: bytes,
    pair_start: int,
    pair_count: int,
    family: G17AFamily,
    mode: G17AMode,
) -> G17AActiveNestedV1:
    parsed = _parse_learned_payload(payload[8:])
    return G17AActiveNestedV1(
        payload,
        payload,
        pair_start,
        pair_count,
        family,
        mode,
        "test.g65.strict-a.v1",
        parsed,
    )


def _record(
    owner: G17LogicalOwnershipV1,
    *,
    group_id: str,
    payload_class: str,
    scientific_role: G17ScientificRoleV1,
    semantic_role: G17SemanticStreamRoleV1,
) -> G17CompilerPlacementRecordV1:
    return G17CompilerPlacementRecordV1(
        owner,
        scientific_role,
        semantic_role,
        G17RecursionCoordinateV1(
            G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
            "L1_program",
        ),
        G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
        G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC,
        payload_class,
        group_id,
        True,
    )


def _fixture() -> tuple[
    bytes,
    G17CompilerPlacementManifestV1,
    G17ConditionalFullFrameY0GivenY1OperationV1,
]:
    source_sha = learned_quotient_decoder_source_sha256(_decoder)
    operand = build_learned_irreducible_quotient_factor(
        section_id="g65-conditional-y0",
        source_pair_start=17,
        source_pair_stop_exclusive=19,
        decoder_contract_id="test.g65.decoder.v1",
        decoder_implementation_source_sha256=source_sha,
        model_family_id="g65-fixture",
        latent_codec_id="raw-u8",
        parameter_codec_id="raw-u8",
        latent_dtype="uint8",
        parameter_dtype="uint8",
        latent_payload=b"\x01\x02\x03\x04",
        parameter_payload=b"\x05\x06\x07\x08",
        source_receipt_sha256="1" * 64,
    ).payload
    p_section = b"g65-predictor"
    g_section = build_g17_g_packet(
        p_section=p_section,
        pair_start=0,
        pair_count=600,
    )
    active = _a_parser(
        b"TACX2A4\x00" + operand,
        0,
        600,
        G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
        G17AMode.QUANTIZED_XIP2,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=600,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(
            G17ADescriptorV1(
                0,
                600,
                G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
                G17AMode.QUANTIZED_XIP2,
                active,
            ),
        ),
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        a_active_parser=_a_parser,
    )
    parsed = build_g17_production_archive(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        terminal_section=terminal,
        a_active_parser=_a_parser,
    ).selected
    archive = parsed.outer.archive_bytes
    member = parsed.outer.member_bytes
    pose = G17LogicalOwnershipV1(
        "g65-pose-owner",
        G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE,
        G17ChronologicalPosePreimageV1(operand),
        parameter_spelling=G17ParameterSpellingIdentityV1(
            operand,
            PARAMETER_SPELLING_FORMAT,
        ),
    )
    common = G17LogicalOwnershipV1(
        "g65-common-owner",
        G17LogicalOwnershipKindV1.POPULATION_SHARED,
        G17PopulationSharingV1(b"shared state"),
    )
    records = (
        _record(
            pose,
            group_id="pose-range",
            payload_class=COUNTED_PAYLOAD_CLASS,
            scientific_role=G17ScientificRoleV1.POSE_TRANSPORT_FRAME0,
            semantic_role=G17SemanticStreamRoleV1.FIBER,
        ),
        _record(
            common,
            group_id="common-range",
            payload_class="G65_COMMON_STATE",
            scientific_role=G17ScientificRoleV1.BULK_BOUNDARY,
            semantic_role=G17SemanticStreamRoleV1.CONNECTION,
        ),
    )
    split = len(archive) // 2
    groups = (
        G17PhysicalCodingGroupV1(
            "pose-range",
            archive,
            parsed.outer.member_name,
            member,
            0,
            archive[:split],
            "fixture-coder",
            "fixture-container",
            RECEIVER_CONSUMER_ID,
            OPERATION_ID,
            (pose.owner_id,),
        ),
        G17PhysicalCodingGroupV1(
            "common-range",
            archive,
            parsed.outer.member_name,
            member,
            split,
            archive[split:],
            "fixture-coder",
            "fixture-container",
            "fixture-common-receiver",
            "FIXTURE_COMMON_OPERATION",
            (common.owner_id,),
        ),
    )
    manifest = G17CompilerPlacementManifestV1(
        records,
        groups,
        (pose.identity_sha256, common.identity_sha256),
        archive,
        parsed.outer.member_name,
        member,
    )
    source_path = Path(__file__).resolve()
    operation = G17ConditionalFullFrameY0GivenY1OperationV1(
        placement_manifest=manifest,
        conditional_pose_owner_id=pose.owner_id,
        source_pair_id=17,
        decoder_contract_id="test.g65.decoder.v1",
        decoder_implementation_source_sha256=source_sha,
        decoder_runtime_dependency=G17RuntimeDependencyFileV1(
            relative_path=("src/tac/witness_dsl/tests/test_taskspace_g17_archive_object_operand_linker_v1.py"),
            exact_file_bytes=source_path.read_bytes(),
            custody_owner="g65-fixture",
            scope=G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
        ),
        learned_quotient_decoder=_decoder,
    )
    return archive, manifest, operation


def test_exact_custody_emits_owner_specific_receiver_incidence_blocker() -> None:
    archive, manifest, operation = _fixture()
    proof = audit_g17_conditional_operand_archive_link_blocker(
        archive_bytes=archive,
        placement_manifest=manifest,
        conditional_operation=operation,
        a_active_parser=_a_parser,
    )
    assert proof.baseline_build.selected.outer.archive_bytes == archive
    assert proof.receipt.unresolved_blocker == OWNER_RECEIVER_INCIDENCE_BLOCKER
    assert proof.receipt.archive_object_receiver_liveness_proven is False
    assert proof.receipt.local_zip_operand_owner_ranges_claimed == 0
    assert proof.receipt.honest_physical_group_scope == ("WHOLE_ARCHIVE_SHARED_MANY_TO_MANY")
    assert proof.receipt.to_receipt_bytes() == proof.receipt.to_receipt_bytes()


def test_link_refuses_false_whole_group_g64_recast_and_foreign_archive() -> None:
    archive, manifest, operation = _fixture()
    with pytest.raises(
        G17ArchiveObjectOperandLinkError,
        match=OWNER_RECEIVER_INCIDENCE_BLOCKER,
    ):
        link_g17_conditional_operand_archive_object(
            archive_bytes=archive,
            placement_manifest=manifest,
            conditional_operation=operation,
            exact_decoded_y1=np.zeros((874, 1164, 3), dtype=np.uint8),
            a_active_parser=_a_parser,
        )
    with pytest.raises(G17ArchiveObjectOperandLinkError, match="differs"):
        audit_g17_conditional_operand_archive_link_blocker(
            archive_bytes=archive + b"x",
            placement_manifest=manifest,
            conditional_operation=operation,
            a_active_parser=_a_parser,
        )
