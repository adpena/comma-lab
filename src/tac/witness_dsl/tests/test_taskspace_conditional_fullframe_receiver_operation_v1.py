# SPDX-License-Identifier: MIT
"""Adversarial mechanics tests for the G64 conditional full-frame operation."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_conditional_fullframe_receiver_operation_v1 import (
    ARCHIVE_RANGE_LINK_BLOCKER,
    CAMERA_FRAME_SHAPE,
    COUNTED_PAYLOAD_CLASS,
    G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1,
    GENERIC_SOURCE_PROVENANCE_BLOCKER,
    OPERATION_ID,
    PARAMETER_SPELLING_FORMAT,
    RECEIVER_CONSUMER_ID,
    RUNTIME_GRAPH_LINK_BLOCKER,
    G17ConditionalFullFrameReceiverError,
    G17ConditionalFullFrameY0GivenY1OperationV1,
    conditional_fullframe_receiver_source_closure,
    execute_g17_public_receiver_operation,
    learned_quotient_decoder_source_sha256,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    build_learned_irreducible_quotient_factor,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ArtifactClassV1,
    G17ChronologicalPosePreimageV1,
    G17CompilerPlacementManifestV1,
    G17CompilerPlacementRecordV1,
    G17EntropyContextV1,
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

_DECODER_CONTRACT = "test.g64.conditional_y0_decoder.v1"
_POSE_OWNER_ID = "conditional-pose-fiber-owner"
_CONDITIONAL_GROUP_ID = "conditional-pose-shared-group"
_SOURCE_PAIR_ID = 17


def _fixture_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    values = np.frombuffer(counted_operand[-8:], dtype=np.uint8)
    values = np.roll(values, -(source_pair_id - 17))
    output = exact_y1.copy(order="C")
    flat = output.reshape(-1)
    flat[:8] = np.bitwise_xor(flat[:8], values)
    return np.ascontiguousarray(output)


def _dead_operand_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    del counted_operand
    del source_pair_id
    return exact_y1.copy(order="C")


def _nondeterministic_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    del counted_operand
    del source_pair_id
    output = exact_y1.copy(order="C")
    output.reshape(-1)[:32] = np.random.randint(
        0,
        256,
        size=32,
        dtype=np.uint8,
    )
    return output


def _mutating_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    del counted_operand
    del source_pair_id
    exact_y1.setflags(write=True)
    exact_y1[0, 0, 0] ^= 1
    return exact_y1


def _wrong_dtype_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    del counted_operand
    del source_pair_id
    return exact_y1.astype(np.float32)


_uncounted_external_value = 7


def _hidden_global_decoder(
    counted_operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    del counted_operand
    del source_pair_id
    output = exact_y1.copy(order="C")
    output[0, 0, 0] ^= _uncounted_external_value
    return output


def _zip(member: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo("program.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, member)
    return output.getvalue()


def _record(
    owner: G17LogicalOwnershipV1,
    *,
    group_id: str,
    payload_class: str,
    scientific_role: G17ScientificRoleV1,
    semantic_role: G17SemanticStreamRoleV1,
) -> G17CompilerPlacementRecordV1:
    return G17CompilerPlacementRecordV1(
        logical_owner=owner,
        scientific_role=scientific_role,
        semantic_role=semantic_role,
        recursion_coordinate=G17RecursionCoordinateV1(
            G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
            "L1_program",
        ),
        placement_class=G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
        artifact_class=G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC,
        payload_class=payload_class,
        physical_coding_group_id=group_id,
        video_specific_derivation=True,
    )


def _runtime_dependency(
    decoder: object,
) -> G17RuntimeDependencyFileV1:
    source_path = Path(str(__file__)).resolve()
    assert learned_quotient_decoder_source_sha256(decoder) == (learned_quotient_decoder_source_sha256(_fixture_decoder))
    return G17RuntimeDependencyFileV1(
        relative_path=("src/tac/witness_dsl/tests/test_taskspace_conditional_fullframe_receiver_operation_v1.py"),
        exact_file_bytes=source_path.read_bytes(),
        custody_owner="g64-test-fixture",
        scope=G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
    )


def _learned_operand(decoder: object) -> bytes:
    source_sha256 = learned_quotient_decoder_source_sha256(decoder)
    factor = build_learned_irreducible_quotient_factor(
        section_id="conditional-y0",
        source_pair_start=_SOURCE_PAIR_ID,
        source_pair_stop_exclusive=_SOURCE_PAIR_ID + 2,
        decoder_contract_id=_DECODER_CONTRACT,
        decoder_implementation_source_sha256=source_sha256,
        model_family_id="test-fullframe-conditional",
        latent_codec_id="raw-int8-test",
        parameter_codec_id="raw-int8-test",
        latent_dtype="uint8",
        parameter_dtype="uint8",
        latent_payload=b"\x01\x02\x03\x04",
        parameter_payload=b"\x05\x06\x07\x08",
        source_receipt_sha256="1" * 64,
    )
    return factor.payload


def _manifest(
    decoder: object = _fixture_decoder,
    *,
    payload_class: str = COUNTED_PAYLOAD_CLASS,
    operand_override: bytes | None = None,
    include_unrelated_group: bool = False,
) -> G17CompilerPlacementManifestV1:
    operand = _learned_operand(decoder) if operand_override is None else operand_override
    member = b"P-G-Y1-COMMON\x00" + operand + b"\x00A-E"
    archive = _zip(member)
    pose_owner = G17LogicalOwnershipV1(
        owner_id=_POSE_OWNER_ID,
        ownership_kind=G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE,
        value=G17ChronologicalPosePreimageV1(operand),
        parameter_spelling=G17ParameterSpellingIdentityV1(
            exact_parameter_bytes=operand,
            spelling_format=PARAMETER_SPELLING_FORMAT,
        ),
    )
    common_owner = G17LogicalOwnershipV1(
        owner_id="shared-y1-common-owner",
        ownership_kind=G17LogicalOwnershipKindV1.POPULATION_SHARED,
        value=G17PopulationSharingV1(b"shared common and differential state"),
    )
    records = [
        _record(
            pose_owner,
            group_id=_CONDITIONAL_GROUP_ID,
            payload_class=payload_class,
            scientific_role=G17ScientificRoleV1.POSE_TRANSPORT_FRAME0,
            semantic_role=G17SemanticStreamRoleV1.FIBER,
        ),
        _record(
            common_owner,
            group_id=_CONDITIONAL_GROUP_ID,
            payload_class="SHARED_Y1_COMMON_AND_DIFFERENTIAL_OPERAND",
            scientific_role=G17ScientificRoleV1.BULK_BOUNDARY,
            semantic_role=G17SemanticStreamRoleV1.CONNECTION,
        ),
    ]
    expected = [pose_owner.identity_sha256, common_owner.identity_sha256]
    groups: list[G17PhysicalCodingGroupV1] = []
    split = len(archive)
    conditional_owner_ids = (pose_owner.owner_id, common_owner.owner_id)
    if include_unrelated_group:
        split = len(archive) // 2
        unrelated_owner = G17LogicalOwnershipV1(
            owner_id="unrelated-terminal-owner",
            ownership_kind=G17LogicalOwnershipKindV1.ENTROPY_CONTEXT,
            value=G17EntropyContextV1(b"unrelated outer state"),
        )
        records.append(
            _record(
                unrelated_owner,
                group_id="unrelated-group",
                payload_class="UNRELATED_PRIMARY_PROGRAM_OPERAND",
                scientific_role=G17ScientificRoleV1.TOPOLOGY_WORLDSHEET,
                semantic_role=G17SemanticStreamRoleV1.SKELETON,
            )
        )
        expected.append(unrelated_owner.identity_sha256)
    groups.append(
        G17PhysicalCodingGroupV1(
            group_id=_CONDITIONAL_GROUP_ID,
            exact_archive_bytes=archive,
            member_name="program.bin",
            exact_member_bytes=member,
            archive_offset=0,
            exact_range_bytes=archive[:split],
            coder_owner="test-joint-coder",
            container_owner="test-archive",
            receiver_consumer=RECEIVER_CONSUMER_ID,
            receiver_operation=OPERATION_ID,
            logical_owner_ids=conditional_owner_ids,
        )
    )
    if include_unrelated_group:
        groups.append(
            G17PhysicalCodingGroupV1(
                group_id="unrelated-group",
                exact_archive_bytes=archive,
                member_name="program.bin",
                exact_member_bytes=member,
                archive_offset=split,
                exact_range_bytes=archive[split:],
                coder_owner="test-outer-coder",
                container_owner="test-archive",
                receiver_consumer="unrelated-primary-receiver",
                receiver_operation="UNRELATED_PRIMARY_OPERATION",
                logical_owner_ids=("unrelated-terminal-owner",),
            )
        )
    return G17CompilerPlacementManifestV1(
        records=tuple(records),
        coding_groups=tuple(groups),
        expected_object_identities=tuple(expected),
        exact_archive_bytes=archive,
        member_name="program.bin",
        exact_member_bytes=member,
    )


def _operation(
    decoder: object = _fixture_decoder,
    *,
    manifest: G17CompilerPlacementManifestV1 | None = None,
    source_sha256: str | None = None,
    runtime_dependency: G17RuntimeDependencyFileV1 | None = None,
) -> G17ConditionalFullFrameY0GivenY1OperationV1:
    return G17ConditionalFullFrameY0GivenY1OperationV1(
        placement_manifest=_manifest(decoder) if manifest is None else manifest,
        conditional_pose_owner_id=_POSE_OWNER_ID,
        source_pair_id=_SOURCE_PAIR_ID,
        decoder_contract_id=_DECODER_CONTRACT,
        decoder_implementation_source_sha256=(
            learned_quotient_decoder_source_sha256(decoder) if source_sha256 is None else source_sha256
        ),
        decoder_runtime_dependency=(_runtime_dependency(decoder) if runtime_dependency is None else runtime_dependency),
        learned_quotient_decoder=decoder,
    )


def _y1() -> np.ndarray:
    return np.zeros(CAMERA_FRAME_SHAPE, dtype=np.uint8)


def test_registered_operation_executes_fullframe_and_preserves_exact_y1() -> None:
    y1 = _y1()
    before = y1.tobytes()
    result = execute_g17_public_receiver_operation(
        OPERATION_ID,
        _operation(manifest=_manifest(include_unrelated_group=True)),
        y1,
    )
    assert y1.tobytes() == before
    assert result.y0.shape == CAMERA_FRAME_SHAPE
    assert result.y0.dtype == np.uint8
    assert result.y0.flags.writeable is False
    assert result.y0.reshape(-1)[:8].tolist() == list(range(1, 9))
    assert result.receipt.deterministic_double_decode is True
    assert result.receipt.exact_decoded_y1_immutable is True
    assert result.receipt.decoder_runtime_dependency_source_bound is True
    assert result.receipt.public_archive_admission_blockers == (
        GENERIC_SOURCE_PROVENANCE_BLOCKER,
        RUNTIME_GRAPH_LINK_BLOCKER,
        ARCHIVE_RANGE_LINK_BLOCKER,
    )
    assert result.receipt.logical_group_operand_liveness_results == ((_CONDITIONAL_GROUP_ID, "CHANGED"),)
    assert len(result.receipt.operand_mutation_results) == 4
    assert {status for _, status in result.receipt.operand_mutation_results} == {"CHANGED"}


def test_registry_is_closed_and_source_closure_is_content_addressed() -> None:
    assert tuple(G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1) == (OPERATION_ID,)
    with pytest.raises(TypeError):
        G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1["other"] = (  # type: ignore[index]
            G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1[OPERATION_ID]
        )
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="unknown"):
        execute_g17_public_receiver_operation("ARBITRARY_IMPORT", _operation(), _y1())
    first = conditional_fullframe_receiver_source_closure()
    second = conditional_fullframe_receiver_source_closure()
    assert first == second
    assert len(first["files"]) == 3
    assert first["external_reads"] == []
    assert first["scorer_teacher_target_dependencies"] == []


def test_callable_source_and_runtime_dependency_are_exact_bound() -> None:
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="source differs"):
        _operation(source_sha256="0" * 64)
    wrong_dependency = replace(
        _runtime_dependency(_fixture_decoder),
        exact_file_bytes=b"foreign source",
    )
    with pytest.raises(
        G17ConditionalFullFrameReceiverError,
        match="SUBMISSION_RUNTIME_DEPENDENCY",
    ):
        _operation(runtime_dependency=wrong_dependency)
    wrong_scope = replace(
        _runtime_dependency(_fixture_decoder),
        scope=G17RuntimeFileScopeV1.SYSTEM_RUNTIME_DEPENDENCY,
    )
    with pytest.raises(
        G17ConditionalFullFrameReceiverError,
        match="SUBMISSION_RUNTIME_DEPENDENCY",
    ):
        _operation(runtime_dependency=wrong_scope)


def test_hidden_global_default_or_callable_object_has_no_execution_path() -> None:
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="uncounted global"):
        _operation(_hidden_global_decoder)

    hidden = 3

    def closure_decoder(
        counted_operand: bytes,
        source_pair_id: int,
        exact_y1: np.ndarray,
    ) -> np.ndarray:
        del counted_operand
        del source_pair_id
        output = exact_y1.copy(order="C")
        output[0, 0, 0] ^= hidden
        return output

    with pytest.raises(G17ConditionalFullFrameReceiverError, match="closure/default"):
        _operation(closure_decoder)

    class CallableDecoder:
        def __call__(
            self,
            counted_operand: bytes,
            source_pair_id: int,
            exact_y1: np.ndarray,
        ) -> np.ndarray:
            del counted_operand
            del source_pair_id
            return exact_y1.copy(order="C")

    with pytest.raises(G17ConditionalFullFrameReceiverError, match="Python source"):
        _operation(CallableDecoder())


def test_pair_coordinate_is_explicit_and_local_imports_fail_closed() -> None:
    pair_17 = _operation().execute(_y1())
    pair_18 = replace(_operation(), source_pair_id=_SOURCE_PAIR_ID + 1).execute(_y1())
    assert not np.array_equal(pair_17.y0, pair_18.y0)
    assert pair_18.receipt.source_pair_id == _SOURCE_PAIR_ID + 1

    def local_import_decoder(
        counted_operand: bytes,
        source_pair_id: int,
        exact_y1: np.ndarray,
    ) -> np.ndarray:
        import subprocess

        del counted_operand
        del source_pair_id
        del subprocess
        return exact_y1.copy(order="C")

    with pytest.raises(G17ConditionalFullFrameReceiverError, match="local import"):
        _operation(local_import_decoder)


@pytest.mark.parametrize(
    ("decoder", "match"),
    [
        (_dead_operand_decoder, "receiver-dead"),
        (_nondeterministic_decoder, "deterministic double"),
        (_mutating_decoder, "mutated immutable Y1"),
        (_wrong_dtype_decoder, "uint8 Y0"),
    ],
)
def test_bad_receiver_physics_fail_closed(
    decoder: object,
    match: str,
) -> None:
    operation = _operation(decoder)
    with pytest.raises(G17ConditionalFullFrameReceiverError, match=match):
        operation.execute(_y1())


def test_exact_fullframe_input_contract_refuses_shape_dtype_and_layout() -> None:
    operation = _operation()
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="exact C-contiguous"):
        operation.execute(np.zeros((384, 512, 3), dtype=np.uint8))
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="exact C-contiguous"):
        operation.execute(np.zeros(CAMERA_FRAME_SHAPE, dtype=np.float32))
    noncontiguous = np.zeros(
        (CAMERA_FRAME_SHAPE[1], CAMERA_FRAME_SHAPE[0], 3),
        dtype=np.uint8,
    ).transpose(1, 0, 2)
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="exact C-contiguous"):
        operation.execute(noncontiguous)


def test_manifest_refuses_foreign_payload_direct_bytes_and_ambiguous_spelling() -> None:
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="placement"):
        _operation(
            manifest=_manifest(
                payload_class="DIRECT_FULLFRAME_PLANE",
            )
        )
    raw_direct_bytes = b"\x00" * 1024
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="compact learned"):
        _operation(
            manifest=_manifest(
                operand_override=raw_direct_bytes,
            )
        )
    valid_operand = _learned_operand(_fixture_decoder)
    manifest = _manifest()
    duplicated_member = b"P-G-Y1-COMMON\x00" + valid_operand + b"\x00" + valid_operand + b"\x00A-E"
    duplicated_archive = _zip(duplicated_member)
    group = manifest.coding_groups[0]
    duplicated_group = replace(
        group,
        exact_archive_bytes=duplicated_archive,
        exact_member_bytes=duplicated_member,
        exact_range_bytes=duplicated_archive,
    )
    ambiguous = replace(
        manifest,
        coding_groups=(duplicated_group,),
        exact_archive_bytes=duplicated_archive,
        exact_member_bytes=duplicated_member,
    )
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="ambiguous"):
        _operation(manifest=ambiguous)


def test_source_pair_and_multi_group_operand_mapping_fail_closed() -> None:
    with pytest.raises(G17ConditionalFullFrameReceiverError, match="does not address"):
        replace(_operation(), source_pair_id=_SOURCE_PAIR_ID + 2)

    manifest = _manifest(include_unrelated_group=True)
    pose_record = manifest.records[0]
    unrelated_group = manifest.coding_groups[1]
    second_pose_record = replace(
        pose_record,
        recursion_coordinate=G17RecursionCoordinateV1(
            G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
            "L2_chart",
        ),
        physical_coding_group_id=unrelated_group.group_id,
    )
    shared_unrelated_group = replace(
        unrelated_group,
        receiver_consumer=RECEIVER_CONSUMER_ID,
        receiver_operation=OPERATION_ID,
        logical_owner_ids=(
            *unrelated_group.logical_owner_ids,
            _POSE_OWNER_ID,
        ),
    )
    multi_group = replace(
        manifest,
        records=(*manifest.records, second_pose_record),
        coding_groups=(manifest.coding_groups[0], shared_unrelated_group),
    )
    with pytest.raises(
        G17ConditionalFullFrameReceiverError,
        match="SPAN_LINKER_OWED",
    ):
        _operation(manifest=multi_group)
