# SPDX-License-Identifier: MIT
"""Adversarial structural tests for the canonical G17 selected-solution compiler."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

import tac.witness_dsl.taskspace_g17_compiler_placement as compatibility
import tac.witness_dsl.taskspace_selected_solution_compiler as canonical
from tac.witness_dsl.taskspace_g17_forward_observation import (
    CANDIDATE_RECEIPT_SCHEMA,
    G17CandidateForwardObservationV1,
    G17TargetForwardObservationV1,
    parse_g17_candidate_forward_receipt,
    parse_g17_target_forward_receipt,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    C0BArchiveArtifactV1,
    C0BAuthEvalClosureV1,
    C0BDecodeReceiptV1,
    C0BObligationIRV1,
    C0BRealizedPairV1,
    C0BScoreReceiptV1,
    C0BSourceTruthV1,
    G17ArtifactClassV1,
    G17ChronologicalPosePreimageV1,
    G17CompetitiveTargetIdentityV1,
    G17CompilerBlocker,
    G17CompilerBlockerCodeV1,
    G17CompilerPlacementError,
    G17CompilerPlacementManifestV1,
    G17CompilerPlacementRecordV1,
    G17ContestCPUAuthorityEvidenceV1,
    G17ContestCUDAAuthorityEvidenceV1,
    G17DeterministicReconstructionProgramV1,
    G17EvaluatorRecursionStageV1,
    G17FrameRoleV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17ObligationCoordinateV1,
    G17ObligationCoverageModeV1,
    G17ObligationCoverageV1,
    G17PairPopulationV1,
    G17PhysicalCodingGroupV1,
    G17PlacementClassV1,
    G17PoseOwnershipV1,
    G17PosePreimageOwnershipV1,
    G17ProofDependencyDomainV1,
    G17ProofDependencySetV1,
    G17ProofDependencyV1,
    G17ProofKindV1,
    G17R10ConstraintCoordinateV1,
    G17R10ConstraintV1,
    G17R10ProsodyFeatureRelayV1,
    G17RealizationGaugeV1,
    G17ReceiverExecutionResultV1,
    G17RecursionCoordinateV1,
    G17RecursionNamespaceV1,
    G17ReopenedEvidencePacketV1,
    G17ResearchAuthorityEvidenceV1,
    G17RuntimeDependencyEdgeV1,
    G17RuntimeDependencyFileV1,
    G17RuntimeDependencyMechanismV1,
    G17RuntimeFileScopeV1,
    G17ScientificRoleV1,
    G17ScorerExecutionResultV1,
    G17SemanticStreamRoleV1,
    G17SemanticTopologyV1,
    G17TerminalCompilerPassV1,
    G17TerminalCompilerScheduleV1,
    G17VMOpcodeV1,
    G17VMOperandV1,
    G17WholeObjectStateV1,
    build_g17_whole_object_state_receipt,
    execute_g17_reconstruction_vm,
    parse_g17_whole_object_state_receipt,
    require_g17_auth_eval_public_entrypoint_closure,
)


def _zip(member: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, member)
    return output.getvalue()


def _population() -> G17PairPopulationV1:
    return G17PairPopulationV1(
        global_pair_ids=(10, 11),
        source_pair_ids=(10, 11),
        v9_pair_coordinates=(10, 11),
        pbr_pair_coordinates=(0, 1),
        obligation_ir_coordinates=(0, 1),
        v10_local_coordinates=(0, 1),
    )


def _topology_owner() -> G17LogicalOwnershipV1:
    return G17LogicalOwnershipV1(
        owner_id="topology-owner",
        ownership_kind=G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
        value=G17SemanticTopologyV1(b"topology"),
    )


def _gauge_owner() -> G17LogicalOwnershipV1:
    return G17LogicalOwnershipV1(
        owner_id="gauge-owner",
        ownership_kind=G17LogicalOwnershipKindV1.REALIZATION_GAUGE,
        value=G17RealizationGaugeV1(b"gauge"),
    )


def _record(
    owner: G17LogicalOwnershipV1,
    *,
    group_id: str,
    scientific_role: G17ScientificRoleV1,
    semantic_role: G17SemanticStreamRoleV1,
) -> G17CompilerPlacementRecordV1:
    return G17CompilerPlacementRecordV1(
        logical_owner=owner,
        scientific_role=scientific_role,
        semantic_role=semantic_role,
        recursion_coordinate=G17RecursionCoordinateV1(
            G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
            G17EvaluatorRecursionStageV1.L1_PROGRAM.value,
        ),
        placement_class=G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
        artifact_class=G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC,
        payload_class="FINITE_VIDEO_SPECIFIC_STATISTIC",
        physical_coding_group_id=group_id,
        video_specific_derivation=True,
    )


def _group(
    *,
    group_id: str,
    archive: bytes,
    member: bytes,
    start: int,
    stop: int,
    owner_ids: tuple[str, ...],
) -> G17PhysicalCodingGroupV1:
    return G17PhysicalCodingGroupV1(
        group_id=group_id,
        exact_archive_bytes=archive,
        member_name="0.bin",
        exact_member_bytes=member,
        archive_offset=start,
        exact_range_bytes=archive[start:stop],
        coder_owner="zip-store",
        container_owner="archive",
        receiver_consumer="receiver",
        receiver_operation="decode-counted-statistic",
        logical_owner_ids=owner_ids,
    )


def test_compatibility_adapter_reexports_identical_objects() -> None:
    assert compatibility.__all__ == canonical.__all__
    assert all(getattr(compatibility, name) is getattr(canonical, name) for name in canonical.__all__)


def test_scientific_semantic_and_recursion_axes_cannot_cross_cast() -> None:
    owner = _topology_owner()
    with pytest.raises(G17CompilerPlacementError):
        G17RecursionCoordinateV1(
            G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
            "L1_PROGRAM",
        )
    with pytest.raises(G17CompilerPlacementError):
        G17CompilerPlacementRecordV1(
            logical_owner=owner,
            scientific_role=G17SemanticStreamRoleV1.SKELETON,  # type: ignore[arg-type]
            semantic_role=G17SemanticStreamRoleV1.SKELETON,
            recursion_coordinate=G17RecursionCoordinateV1(
                G17RecursionNamespaceV1.TS1_INFORMATION_HOME,
                G17EvaluatorRecursionStageV1.L1_PROGRAM.value,
            ),
            placement_class=G17PlacementClassV1.COUNTED_VIDEO_STATISTIC,
            artifact_class=G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC,
            payload_class="FINITE_VIDEO_SPECIFIC_STATISTIC",
            physical_coding_group_id="g",
            video_specific_derivation=True,
        )


def test_joint_physical_group_is_charged_once_and_gap_overlap_refuse() -> None:
    member = b"member-bytes-for-physical-accounting"
    archive = _zip(member)
    topology = _topology_owner()
    gauge = _gauge_owner()
    records = (
        _record(
            topology,
            group_id="joint",
            scientific_role=G17ScientificRoleV1.TOPOLOGY_WORLDSHEET,
            semantic_role=G17SemanticStreamRoleV1.SKELETON,
        ),
        _record(
            gauge,
            group_id="joint",
            scientific_role=G17ScientificRoleV1.CELL_VALUE_PREIMAGE,
            semantic_role=G17SemanticStreamRoleV1.GAUGE,
        ),
    )
    joint = _group(
        group_id="joint",
        archive=archive,
        member=member,
        start=0,
        stop=len(archive),
        owner_ids=(topology.owner_id, gauge.owner_id),
    )
    manifest = G17CompilerPlacementManifestV1(
        records=records,
        coding_groups=(joint,),
        expected_object_identities=(topology.identity_sha256, gauge.identity_sha256),
        exact_archive_bytes=archive,
        member_name="0.bin",
        exact_member_bytes=member,
    )
    assert sum(group.byte_length for group in manifest.coding_groups) == len(archive)
    assert joint.logical_owner_ids == (topology.owner_id, gauge.owner_id)

    midpoint = len(archive) // 2
    split_records = (
        _record(
            topology,
            group_id="left",
            scientific_role=G17ScientificRoleV1.TOPOLOGY_WORLDSHEET,
            semantic_role=G17SemanticStreamRoleV1.SKELETON,
        ),
        _record(
            gauge,
            group_id="right",
            scientific_role=G17ScientificRoleV1.CELL_VALUE_PREIMAGE,
            semantic_role=G17SemanticStreamRoleV1.GAUGE,
        ),
    )
    with pytest.raises(G17CompilerPlacementError, match="gap"):
        G17CompilerPlacementManifestV1(
            records=split_records,
            coding_groups=(
                _group(
                    group_id="left",
                    archive=archive,
                    member=member,
                    start=0,
                    stop=midpoint,
                    owner_ids=(topology.owner_id,),
                ),
                _group(
                    group_id="right",
                    archive=archive,
                    member=member,
                    start=midpoint + 1,
                    stop=len(archive),
                    owner_ids=(gauge.owner_id,),
                ),
            ),
            expected_object_identities=(topology.identity_sha256, gauge.identity_sha256),
            exact_archive_bytes=archive,
            member_name="0.bin",
            exact_member_bytes=member,
        )
    with pytest.raises(G17CompilerPlacementError, match="overlap"):
        G17CompilerPlacementManifestV1(
            records=split_records,
            coding_groups=(
                _group(
                    group_id="left",
                    archive=archive,
                    member=member,
                    start=0,
                    stop=midpoint + 1,
                    owner_ids=(topology.owner_id,),
                ),
                _group(
                    group_id="right",
                    archive=archive,
                    member=member,
                    start=midpoint,
                    stop=len(archive),
                    owner_ids=(gauge.owner_id,),
                ),
            ),
            expected_object_identities=(topology.identity_sha256, gauge.identity_sha256),
            exact_archive_bytes=archive,
            member_name="0.bin",
            exact_member_bytes=member,
        )


def test_pair_population_and_obligation_coverage_are_complete_coordinates() -> None:
    population = _population()
    obligations = (
        G17ObligationCoordinateV1(10, "seg:10:0"),
        G17ObligationCoordinateV1(11, "pose:11:0"),
    )
    coverage = G17ObligationCoverageV1(
        population=population,
        mode=G17ObligationCoverageModeV1.COMPLETE,
        obligation_universe=obligations,
        predictor_owned=obligations,
        sparse_owned=(),
    )
    assert coverage.population.binding_sha256 == population.binding_sha256
    with pytest.raises(G17CompilerPlacementError, match="missing or foreign"):
        G17ObligationCoverageV1(
            population=population,
            mode=G17ObligationCoverageModeV1.COMPLETE,
            obligation_universe=obligations,
            predictor_owned=(obligations[0],),
            sparse_owned=(),
        )
    with pytest.raises(G17CompilerPlacementError):
        G17PairPopulationV1(
            global_pair_ids=(10, 11),
            source_pair_ids=(10, 11),
            v9_pair_coordinates=(10, 11),
            pbr_pair_coordinates=(0, 1),
            obligation_ir_coordinates=(1, 0),
            v10_local_coordinates=(0, 1),
        )


@dataclass(frozen=True)
class _ExactReceipt:
    schema: str
    exact_bytes: bytes

    def to_receipt_bytes(self) -> bytes:
        return self.exact_bytes


_PREIMAGE_BYTES = b'{"schema":"test.exact_preimage.v1"}'


def _parse_preimage(payload: bytes) -> _ExactReceipt:
    if payload != _PREIMAGE_BYTES:
        raise ValueError("foreign preimage")
    return _ExactReceipt("test.exact_preimage.v1", payload)


def test_pose_reverse_causal_requires_strict_reopened_exact_preimage() -> None:
    population = _population()
    with pytest.raises(G17CompilerPlacementError, match="strict-parsed preimage"):
        G17PosePreimageOwnershipV1(
            population=population,
            ownership_by_pair=(
                G17PoseOwnershipV1.REVERSE_CAUSAL_FRAME0_FROM_EXACT_Y1,
                G17PoseOwnershipV1.V9_POSE6,
            ),
            physical_coding_group_id_by_pair=("pose-0", "pose-1"),
            receiver_operation_by_pair=("reverse-causal", "v9-pose6"),
        )
    evidence = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=_PREIMAGE_BYTES,
        strict_parser=_parse_preimage,
        expected_schema="test.exact_preimage.v1",
    )
    ownership = G17PosePreimageOwnershipV1(
        population=population,
        ownership_by_pair=(
            G17PoseOwnershipV1.REVERSE_CAUSAL_FRAME0_FROM_EXACT_Y1,
            G17PoseOwnershipV1.FRAME_ZERO_RESIDUAL,
        ),
        physical_coding_group_id_by_pair=("pose-0", "pose-1"),
        receiver_operation_by_pair=("reverse-causal", "frame-zero-residual"),
        explicit_preimage_packet=evidence,
    )
    assert ownership.explicit_preimage_packet is evidence


def test_r10_names_are_not_receiver_evidence_and_named_blocker_survives() -> None:
    population = _population()
    support = (G17ObligationCoordinateV1(10, "r10:10:0"),)
    rows = tuple(
        G17R10ConstraintCoordinateV1(
            constraint=constraint,
            population=population,
            frame_role=G17FrameRoleV1.CHRONOLOGICAL_PAIR,
            scientific_role=G17ScientificRoleV1.POSE_TRANSPORT_FRAME0,
            semantic_role=G17SemanticStreamRoleV1.CONNECTION,
            exact_value_bytes=constraint.value.encode("ascii"),
            support=support,
            tolerance=0.0,
            exact_frozen_block_bytes=b"frozen-r10-block",
            exact_chronology_receipt_bytes=b"chronology-receipt",
            generic_receiver_operation="r10-feature-relay",
            physical_coding_group_id="r10-group",
            counted_operand_offset=0,
            counted_operand_bytes=b"x",
        )
        for constraint in G17R10ConstraintV1
    )
    relay = G17R10ProsodyFeatureRelayV1(rows)
    with pytest.raises(G17CompilerBlocker) as captured:
        relay.require_receiver_consumption(tuple(G17R10ConstraintV1))
    assert captured.value.code is G17CompilerBlockerCodeV1.G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED


def test_byte_vm_reconstructs_real_bytes_and_rejects_unknown_operations() -> None:
    program = G17DeterministicReconstructionProgramV1(
        bytecode=bytes(
            (
                G17VMOpcodeV1.PUSH_INPUT_SECTION,
                G17VMOpcodeV1.PUSH_LITERAL,
                G17VMOpcodeV1.CONCAT,
                G17VMOpcodeV1.EMIT_SECTION,
            )
        ),
        operands=(
            G17VMOperandV1(b"p"),
            G17VMOperandV1(b"-suffix"),
            G17VMOperandV1(struct.pack(">H", 2)),
            G17VMOperandV1(b"out"),
        ),
    )
    output, receipt = execute_g17_reconstruction_vm(program, input_sections={"p": b"prefix"})
    assert output == {"out": b"prefix-suffix"}
    assert receipt.deterministic_double_execution is True
    assert receipt.emitted_sections == (("out", hashlib.sha256(b"prefix-suffix").hexdigest(), 13),)
    with pytest.raises(G17CompilerBlocker) as captured:
        G17DeterministicReconstructionProgramV1(bytecode=b"\xff", operands=())
    assert captured.value.code is G17CompilerBlockerCodeV1.G17_UNSUPPORTED_TOPOLOGY_OR_CONSTRAINT_VM_OPERATION


def test_pointer_dependency_is_local_to_frontier_admission_and_authority_is_sealed() -> None:
    domains = (
        G17ProofDependencyDomainV1.ARCHIVE_BYTES,
        G17ProofDependencyDomainV1.DECODER_EQUALITY_ALGORITHM,
        G17ProofDependencyDomainV1.MEMBER_CONTAINER_MAPPING,
        G17ProofDependencyDomainV1.PAIR_ORDER,
        G17ProofDependencyDomainV1.RECEIVER_IMPLEMENTATION,
        G17ProofDependencyDomainV1.RECEIVER_RUNTIME,
    )
    dependencies = tuple(
        G17ProofDependencyV1(domain, domain.value.encode("ascii"))
        for domain in sorted(domains, key=lambda item: item.value)
    )
    proof = G17ProofDependencySetV1(
        proof_kind=G17ProofKindV1.ARCHIVE_DECODE_EQUALITY,
        dependencies=dependencies,
        declared_external_reads=(),
    )
    assert all(item.domain is not G17ProofDependencyDomainV1.POINTER_SNAPSHOT for item in proof.dependencies)
    with pytest.raises(G17CompilerPlacementError):
        G17ProofDependencySetV1(
            proof_kind=G17ProofKindV1.ARCHIVE_DECODE_EQUALITY,
            dependencies=tuple(
                sorted(
                    (
                        *dependencies,
                        G17ProofDependencyV1(
                            G17ProofDependencyDomainV1.POINTER_SNAPSHOT,
                            b"pointer",
                        ),
                    ),
                    key=lambda item: item.domain.value,
                )
            ),
            declared_external_reads=(G17ProofDependencyDomainV1.POINTER_SNAPSHOT,),
        )
    for authority_type in (G17ContestCPUAuthorityEvidenceV1, G17ContestCUDAAuthorityEvidenceV1):
        with pytest.raises(G17CompilerPlacementError, match="no public constructor"):
            authority_type()
    for lifecycle_type in (
        C0BArchiveArtifactV1,
        C0BAuthEvalClosureV1,
        C0BDecodeReceiptV1,
        C0BScoreReceiptV1,
    ):
        with pytest.raises(G17CompilerPlacementError):
            lifecycle_type()


def test_terminal_schedule_solves_first_and_trains_only_irreducible_quotient() -> None:
    schedule = G17TerminalCompilerScheduleV1.canonical()
    assert schedule.joint_descent_trainable_roles == (G17ScientificRoleV1.IRREDUCIBLE_QUOTIENT,)
    assert schedule.pass_order == (
        G17TerminalCompilerPassV1.MAXIMAL_INVERSE_SOLVE,
        G17TerminalCompilerPassV1.MINIMAL_IRREDUCIBLE_JOINT_DESCENT,
        G17TerminalCompilerPassV1.TERMINAL_LINK,
    )
    with pytest.raises(G17CompilerPlacementError, match="immediately before terminal link"):
        G17TerminalCompilerScheduleV1(
            inverse_solved_scientific_roles=schedule.inverse_solved_scientific_roles,
            joint_descent_trainable_roles=schedule.joint_descent_trainable_roles,
            pass_order=tuple(reversed(schedule.pass_order)),
        )
    with pytest.raises(G17CompilerPlacementError, match="only the irreducible quotient"):
        G17TerminalCompilerScheduleV1(
            inverse_solved_scientific_roles=schedule.inverse_solved_scientific_roles,
            joint_descent_trainable_roles=(G17ScientificRoleV1.BULK_BOUNDARY,),
            pass_order=schedule.pass_order,
        )


def test_auth_eval_requires_public_entrypoints_and_recursive_runtime_custody() -> None:
    runtime_file = G17RuntimeDependencyFileV1(
        relative_path="upstream/evaluate.py",
        exact_file_bytes=b"evaluator-source",
        custody_owner="upstream-evaluator",
        scope=G17RuntimeFileScopeV1.EVALUATOR_PUBLIC_ENTRYPOINT,
    )
    assert runtime_file.content_sha256 == hashlib.sha256(b"evaluator-source").hexdigest()
    with pytest.raises(G17CompilerPlacementError, match="normalized relative POSIX path"):
        G17RuntimeDependencyFileV1(
            relative_path="../inflate.py",
            exact_file_bytes=b"runtime-source",
            custody_owner="submission-runtime",
            scope=G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
        )
    with pytest.raises(G17CompilerBlocker) as captured:
        require_g17_auth_eval_public_entrypoint_closure()
    assert captured.value.code is G17CompilerBlockerCodeV1.G17_AUTH_EVAL_PUBLIC_ENTRYPOINT_CLOSURE_OWED


def test_auth_eval_graph_uses_evaluate_sh_root_and_refuses_false_evaluate_py_launcher() -> None:
    scopes = {
        "inflate.py": G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
        "inflate.sh": G17RuntimeFileScopeV1.SUBMISSION_PUBLIC_ENTRYPOINT,
        "upstream/evaluate.py": G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        "upstream/evaluate.sh": G17RuntimeFileScopeV1.EVALUATOR_PUBLIC_ENTRYPOINT,
        "upstream/frame_utils.py": G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        "upstream/modules.py": G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
    }
    runtime_files = tuple(
        G17RuntimeDependencyFileV1(
            relative_path=path,
            exact_file_bytes=f"bytes:{path}".encode(),
            custody_owner="test-owner",
            scope=scope,
        )
        for path, scope in sorted(scopes.items())
    )
    rows = (
        ("inflate.sh", "inflate.py", G17RuntimeDependencyMechanismV1.PROCESS_EXEC),
        (
            "upstream/evaluate.py",
            "upstream/frame_utils.py",
            G17RuntimeDependencyMechanismV1.PYTHON_IMPORT,
        ),
        (
            "upstream/evaluate.py",
            "upstream/modules.py",
            G17RuntimeDependencyMechanismV1.PYTHON_IMPORT,
        ),
        ("upstream/evaluate.sh", "inflate.sh", G17RuntimeDependencyMechanismV1.PROCESS_EXEC),
        (
            "upstream/evaluate.sh",
            "upstream/evaluate.py",
            G17RuntimeDependencyMechanismV1.PROCESS_EXEC,
        ),
    )
    edges = tuple(
        sorted(
            (G17RuntimeDependencyEdgeV1(*row) for row in rows),
            key=lambda item: (item.importer_path, item.dependency_path, item.mechanism.value),
        )
    )
    observed = tuple(sorted(scopes))
    file_by_path = canonical._validate_auth_eval_runtime_graph_v1(runtime_files, edges, observed)
    assert set(file_by_path) == set(scopes)

    false_edge = G17RuntimeDependencyEdgeV1(
        "upstream/evaluate.py",
        "inflate.sh",
        G17RuntimeDependencyMechanismV1.PROCESS_EXEC,
    )
    false_graph = tuple(
        sorted(
            (*edges, false_edge),
            key=lambda item: (item.importer_path, item.dependency_path, item.mechanism.value),
        )
    )
    with pytest.raises(G17CompilerPlacementError, match=r"fabricated evaluate\.py launching inflate\.sh"):
        canonical._validate_auth_eval_runtime_graph_v1(runtime_files, false_graph, observed)


def test_logical_pose_value_cannot_be_relabelled_as_semantic_topology() -> None:
    with pytest.raises(G17CompilerPlacementError, match="cross-casts"):
        G17LogicalOwnershipV1(
            owner_id="cross-cast",
            ownership_kind=G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
            value=G17ChronologicalPosePreimageV1(b"pose"),
        )


def _strict_whole_state() -> G17WholeObjectStateV1:
    population = _population()
    target_frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    target_labels = np.zeros((2, 2, 2), dtype=np.uint8)
    target_pose = np.zeros((2, 6), dtype=np.float64)
    r_numerators = np.zeros((2, 1), dtype=np.float64)
    r_denominators = np.ones((2, 1), dtype=np.float64)
    target = G17TargetForwardObservationV1(
        source_pair_ids=population.source_pair_ids,
        target_artifact_bytes=b"target-artifact",
        target_member_bytes=b"target-member",
        camera_frames=target_frames,
        exact_r_numerators=r_numerators,
        exact_r_denominators=r_denominators,
        exact_r_projected_rgb=target_frames,
        seg_labels=target_labels,
        pose6=target_pose,
        second_exact_r_numerators=r_numerators,
        second_exact_r_denominators=r_denominators,
        second_exact_r_projected_rgb=target_frames,
        second_seg_labels=target_labels,
        second_pose6=target_pose,
        frozen_scorer_bytes=b"frozen-scorer",
        scorer_runtime_environment_bytes=b"scorer-runtime",
    )
    target_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=target.receipt.to_receipt_bytes(),
        strict_parser=parse_g17_target_forward_receipt,
        expected_schema=target.receipt.schema,
    )
    source_truth = C0BSourceTruthV1(
        source_bytes=b"source",
        evaluator_source_bytes=b"evaluator-source",
        evaluator_weights_bytes=b"evaluator-weights",
        evaluator_runtime_bytes=b"evaluator-runtime",
        resize_r_implementation_bytes=b"resize-R",
        population=population,
        target_evidence=target_packet,
        originality_declaration_bytes=b"original-pact-state",
    )
    obligations = (
        G17ObligationCoordinateV1(10, "seg:10"),
        G17ObligationCoordinateV1(11, "seg:11"),
    )
    coverage = G17ObligationCoverageV1(
        population=population,
        mode=G17ObligationCoverageModeV1.COMPLETE,
        obligation_universe=obligations,
        predictor_owned=obligations,
        sparse_owned=(),
    )
    pose_ownership = G17PosePreimageOwnershipV1(
        population=population,
        ownership_by_pair=(G17PoseOwnershipV1.V9_POSE6, G17PoseOwnershipV1.V9_POSE6),
        physical_coding_group_id_by_pair=("whole", "whole"),
        receiver_operation_by_pair=("decode-pose", "decode-pose"),
    )
    exact_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=_PREIMAGE_BYTES,
        strict_parser=_parse_preimage,
        expected_schema="test.exact_preimage.v1",
    )
    obligation_ir = C0BObligationIRV1(
        source_truth=source_truth,
        obligation_ir_packet=exact_packet,
        population=population,
        coverage=coverage,
        pose_preimage_ownership=pose_ownership,
        r10_relay=None,
    )
    candidate_y0 = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    candidate_y1 = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    realized = C0BRealizedPairV1(
        obligation_ir=obligation_ir,
        camera_y0=candidate_y0,
        camera_y1=candidate_y1,
        resize_r_proof=exact_packet,
    )
    # Select an archive size for which the old multiply-before-divide rate
    # expression differs from upstream by one ULP.
    for member_nbytes in range(1, 512):
        member = b"m" * member_nbytes
        archive = _zip(member)
        if canonical.rate_term(len(archive)) != (25 * len(archive)) / 37_545_489:
            break
    else:  # pragma: no cover - binary64 invariant on supported Python builds
        raise AssertionError("failed to construct one-ULP archive fixture")
    owner = _topology_owner()
    manifest = G17CompilerPlacementManifestV1(
        records=(
            _record(
                owner,
                group_id="whole",
                scientific_role=G17ScientificRoleV1.TOPOLOGY_WORLDSHEET,
                semantic_role=G17SemanticStreamRoleV1.SKELETON,
            ),
        ),
        coding_groups=(
            _group(
                group_id="whole",
                archive=archive,
                member=member,
                start=0,
                stop=len(archive),
                owner_ids=(owner.owner_id,),
            ),
        ),
        expected_object_identities=(owner.identity_sha256,),
        exact_archive_bytes=archive,
        member_name="0.bin",
        exact_member_bytes=member,
    )
    artifact = C0BArchiveArtifactV1.from_exact_zip(
        realized_pair=realized,
        archive_bytes=archive,
        member_name="0.bin",
        expected_member_bytes=member,
        placement_manifest=manifest,
        decoder_program_bytes=b"decoder-program",
        decoder_runtime_bytes=b"decoder-runtime",
    )

    def receiver(received_archive: bytes) -> G17ReceiverExecutionResultV1:
        if received_archive != archive:
            raise ValueError("mutated archive")
        return G17ReceiverExecutionResultV1(
            decoded_output_bytes=realized.chronology_bytes,
            receiver_receipt=exact_packet,
            emitted_pair_order=population.source_pair_ids,
        )

    decode = C0BDecodeReceiptV1.from_receiver(
        archive_artifact=artifact,
        receiver=receiver,
        receiver_implementation_bytes=b"receiver-implementation",
        receiver_runtime_bytes=b"receiver-runtime",
        receiver_asset_bytes=b"receiver-assets",
        equality_algorithm_bytes=b"exact-byte-equality",
    )
    candidate_labels = target_labels.copy()
    candidate_labels[0, 0, 0] = 1
    candidate_pose = target_pose.copy()
    candidate_pose[0, 0] = 1.0
    candidate = G17CandidateForwardObservationV1(
        target=target,
        archive_bytes=archive,
        member_bytes=member,
        receiver_receipt_bytes=exact_packet.exact_packet_bytes,
        decoded_output_bytes=realized.chronology_bytes,
        camera_y1=candidate_y1,
        exact_r_numerators=r_numerators,
        exact_r_denominators=r_denominators,
        exact_r_projected_rgb=candidate_y1,
        realized_seg_labels=candidate_labels,
        realized_pose6=candidate_pose,
        second_exact_r_numerators=r_numerators,
        second_exact_r_denominators=r_denominators,
        second_exact_r_projected_rgb=candidate_y1,
        second_realized_seg_labels=candidate_labels,
        second_realized_pose6=candidate_pose,
        frozen_scorer_bytes=b"frozen-scorer",
        scorer_runtime_environment_bytes=b"scorer-runtime",
    )
    candidate_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=candidate.receipt.to_receipt_bytes(),
        strict_parser=parse_g17_candidate_forward_receipt,
        expected_schema=CANDIDATE_RECEIPT_SCHEMA,
    )
    authority = G17ResearchAuthorityEvidenceV1(
        evidence_receipt=candidate_packet,
        sample_count=2,
        axis_label="macOS-CPU-advisory",
        exact_hardware_identity_bytes=b"test-hardware",
        exact_scorer_identity_bytes=b"frozen-scorer",
        exact_runtime_identity_bytes=b"scorer-runtime",
    )

    def scorer(_decode: C0BDecodeReceiptV1) -> G17ScorerExecutionResultV1:
        return G17ScorerExecutionResultV1(observation=candidate, authority=authority)

    score = C0BScoreReceiptV1.from_research_scorer(decode_receipt=decode, scorer=scorer)
    return G17WholeObjectStateV1(
        score_receipt=score,
        competitive_target=G17CompetitiveTargetIdentityV1(
            competition_namespace="comma-video-compression",
            metric_namespace="upstream-evaluate-v1",
            selection_policy="dynamic-min-local-and-upstream",
            exact_evaluator_rules_bytes=b"frozen-evaluator-rules",
        ),
    )


def _reseal(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def test_whole_object_state_receipt_strict_parse_reemit_and_private_boundary() -> None:
    receipt = build_g17_whole_object_state_receipt(_strict_whole_state())
    reopened = parse_g17_whole_object_state_receipt(receipt.to_receipt_bytes())
    assert reopened.to_receipt_bytes() == receipt.to_receipt_bytes()
    assert reopened.state_identity_sha256 == receipt.state_identity_sha256
    assert reopened.archive_nbytes == len(reopened.archive_bytes)
    assert reopened.sample_count == 2
    assert reopened.public_rgb_bridge_proven is False
    assert reopened.total_score == canonical.compute_contest_score(
        reopened.aggregate_d_seg,
        reopened.aggregate_d_pose,
        reopened.archive_nbytes,
    )


@pytest.mark.parametrize("mutation", ["score", "axis", "archive_length", "identity", "sufficient_stat"])
def test_whole_object_state_receipt_rejects_resealed_semantic_drift(mutation: str) -> None:
    receipt = build_g17_whole_object_state_receipt(_strict_whole_state())
    payload = json.loads(receipt.to_receipt_bytes())
    if mutation == "score":
        payload["score"]["total_score_hex"] = math.nextafter(receipt.total_score, math.inf).hex()
    elif mutation == "axis":
        payload["authority"]["axis"] = "contest-CPU"
    elif mutation == "archive_length":
        payload["archive"]["archive_nbytes"] += 1
    elif mutation == "identity":
        payload["identities"]["state_identity_sha256"] = "0" * 64
    else:
        payload["observation"]["seg_mismatch_count"] += 1
    with pytest.raises(G17CompilerPlacementError):
        parse_g17_whole_object_state_receipt(_reseal(payload))


def test_whole_object_state_receipt_rejects_noncanonical_duplicate_and_wrong_types() -> None:
    receipt = build_g17_whole_object_state_receipt(_strict_whole_state())
    raw = receipt.to_receipt_bytes()
    duplicate = raw[:-1] + b',"schema":"tac.g17_whole_object_state_receipt.v1"}'
    with pytest.raises(G17CompilerPlacementError, match="repeats JSON key"):
        parse_g17_whole_object_state_receipt(duplicate)
    payload = json.loads(raw)
    payload["archive"]["archive_nbytes"] = True
    with pytest.raises(G17CompilerPlacementError):
        parse_g17_whole_object_state_receipt(_reseal(payload))


def test_whole_object_state_receipt_rejects_old_one_ulp_rate_order() -> None:
    receipt = build_g17_whole_object_state_receipt(_strict_whole_state())
    old_rate = (25 * receipt.archive_nbytes) / 37_545_489
    assert old_rate != canonical.rate_term(receipt.archive_nbytes)
    payload = json.loads(receipt.to_receipt_bytes())
    payload["score"]["rate_term_hex"] = old_rate.hex()
    payload["score"]["total_score_hex"] = (
        canonical.seg_term(receipt.aggregate_d_seg) + canonical.pose_term(receipt.aggregate_d_pose) + old_rate
    ).hex()
    with pytest.raises(G17CompilerPlacementError, match="upstream operation order"):
        parse_g17_whole_object_state_receipt(_reseal(payload))


def test_whole_object_state_receipt_rejects_mutated_or_forged_frozen_state() -> None:
    state = _strict_whole_state()
    object.__setattr__(state.score_receipt, "total_score", math.nextafter(state.score_receipt.total_score, math.inf))
    with pytest.raises(G17CompilerPlacementError, match="upstream operation order"):
        build_g17_whole_object_state_receipt(state)
    forged = object.__new__(G17WholeObjectStateV1)
    object.__setattr__(forged, "score_receipt", object())
    object.__setattr__(forged, "competitive_target", state.competitive_target)
    with pytest.raises(G17CompilerPlacementError):
        build_g17_whole_object_state_receipt(forged)
