# SPDX-License-Identifier: MIT
"""Focused mechanics for the exact G17/G49 product and its current blocker."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
    receive_carrier_compose_archive,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.taskspace_g17_g49_selected_program_product_bridge_v1 import (
    INCIDENCE_EXECUTION_BLOCKER,
    NESTED_P_CONTAINER_BLOCKER,
    SELECTED_PREIMAGE_PROGRAM_TYPE_BLOCKER,
    SEMANTIC_PROGRAM_TYPE_BLOCKER,
    SHARED_ARCHIVE_RECEIVER_CONSUMER,
    SHARED_ARCHIVE_UNPACK_OPERATION,
    G17G49SelectedProgramPreplacementProductV1,
    G17G49SelectedProgramProductError,
    G17OwnerSpecificReceiverIncidenceManifestV1,
    G17OwnerSpecificReceiverIncidenceV1,
    audit_g17_g49_selected_program_product_blockers,
    refuse_g17_g49_product_archive,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    BoundV10Factor2SelectedPreimageDecoderV1,
    GenericV10Factor2DecoderIdentityV1,
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageCompileConfigV1,
    SelectedPreimageFrameSelectorV1,
    TaskspaceSelectedPreimageProgramV1,
    build_analytic_shearlet_residual_factor,
    encode_selected_preimage_program,
    verify_v15_semantic_compile_lineage,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ArtifactClassV1,
    G17CompilerPlacementManifestV1,
    G17CompilerPlacementRecordV1,
    G17EvaluatorRecursionStageV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17PhysicalCodingGroupV1,
    G17PlacementClassV1,
    G17RealizationGaugeV1,
    G17RecursionCoordinateV1,
    G17RecursionNamespaceV1,
    G17ScientificRoleV1,
    G17SemanticStreamRoleV1,
    G17SemanticTopologyV1,
)

_MAX_PACKET_BYTES = 1 << 20


_QUARANTINE_V15_PRODUCER_PIN = pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): retired July taskspace lineage, no live consumer. "
        "The sealed V15 compile receipt's producer_custody pins "
        "src/tac/optimization/direct_description_carrier_compose.py at 3e1f69bb/156,551 B; HEAD is "
        "6fef110d/160,470 B. Drift commits: 9934d488b then 36f4b2947 (both 2026-08-20). "
        "This is NOT a hash swap: 36f4b2947 adds key non_empty_member_payload_count to "
        "prove_carrier_archive_fail_closed, which the V15 producer embeds as "
        "receipt.fail_closed_mutation_proof, so a regenerated receipt legitimately differs. "
        "FIRE TRIGGER: a scorer-authorized arm re-runs tools/measure_ddm_v15_scorer_solved_templates.py "
        "(it solves through exact R + SegNet, which ddm_ql1's charter forbade) and either refreshes "
        "producer_custody against a bit-exact compile receipt -- then DELETE this mark -- or records "
        "real output drift. strict=True means this mark FAILS the moment the lineage is repaired. "
        "Owning memos: .omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md and "
        ".omx/research/ddm_cd1_working_tree_debt_landing_20260903.md"
    ),
)


@dataclass(frozen=True)
class _RealProductFixture:
    semantic_p: bytes
    decoder: BoundV10Factor2SelectedPreimageDecoderV1
    baseline_program: TaskspaceSelectedPreimageProgramV1
    mutated_program: TaskspaceSelectedPreimageProgramV1


def _sha(character: str) -> str:
    return character * 64


def _program(
    *,
    semantic_identity: object,
    target_identity: ScorerTargetCustodyIdentityV1,
    amplitude_q4: int,
) -> TaskspaceSelectedPreimageProgramV1:
    factor = build_analytic_shearlet_residual_factor(
        section_id="g68.boundary_transport",
        source_pair_start=0,
        source_pair_stop_exclusive=1,
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        source_rgb_u8=(11, 3, 9),
        added_rgb_u8=(12, 4, 10),
        removed_rgb_u8=(10, 2, 8),
        atoms=(
            BoundaryShearletAtomV1(
                pair_index=0,
                role="Road",
                center_y=160,
                center_x=256,
                scale_y=24,
                scale_x=96,
                shear_q4=0,
                amplitude_q4=amplitude_q4,
            ),
        ),
        source_receipt_sha256=_sha("2"),
    )
    return TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=semantic_identity,
        target_custody_identity=target_identity,
        decoder_identity=GenericV10Factor2DecoderIdentityV1.current(),
        compile_config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=600,
            maximum_packet_bytes=_MAX_PACKET_BYTES,
            score_budget_receipt_sha256=_sha("4"),
            budget_rule_id="g68_structural_test_budget_v1",
        ),
        factors=(factor,),
    )


@pytest.fixture(scope="module")
def real_product_fixture() -> _RealProductFixture:
    root = Path(__file__).resolve().parents[4]
    run_dir = (
        root
        / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
        / "fresh_v15_semantic_base_n600_20260726"
    )
    receipt_path = run_dir / "ddm_v15_scorer_solved_templates_n600_receipt.json"
    semantic_p_path = run_dir / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
    if not receipt_path.is_file() or not semantic_p_path.is_file():
        pytest.skip("retained fresh V15 semantic custody artifacts are absent in this checkout")
    receipt_bytes = receipt_path.read_bytes()
    semantic_p = semantic_p_path.read_bytes()
    identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=receipt_bytes,
        compiled_semantic_archive=semantic_p,
        producer_root=root,
    )
    target_identity = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256=_sha("f"),
        target_bank_sha256=_sha("1"),
    )
    receiver = receive_carrier_compose_archive(
        semantic_p,
        verify_member_effects=False,
    )
    decoder = BoundV10Factor2SelectedPreimageDecoderV1(
        semantic_identity=identity,
        target_custody_identity=target_identity,
        carrier_receiver=receiver,
        factor2_operator=DisjointResizeOperator.build(
            camera_h=874,
            camera_w=1164,
            scorer_h=384,
            scorer_w=512,
        ),
    )
    return _RealProductFixture(
        semantic_p=semantic_p,
        decoder=decoder,
        baseline_program=_program(
            semantic_identity=identity,
            target_identity=target_identity,
            amplitude_q4=64,
        ),
        mutated_program=_program(
            semantic_identity=identity,
            target_identity=target_identity,
            amplitude_q4=96,
        ),
    )


def _product(
    fixture: _RealProductFixture,
    program: TaskspaceSelectedPreimageProgramV1,
) -> G17G49SelectedProgramPreplacementProductV1:
    return G17G49SelectedProgramPreplacementProductV1(
        semantic_p_archive=fixture.semantic_p,
        selected_program_packet=encode_selected_preimage_program(program),
        decoder=fixture.decoder,
        maximum_packet_bytes=_MAX_PACKET_BYTES,
    )


@_QUARANTINE_V15_PRODUCER_PIN
def test_real_boundv10_product_stream_is_deterministic_and_packet_mutation_is_live(
    real_product_fixture: _RealProductFixture,
) -> None:
    baseline = _product(real_product_fixture, real_product_fixture.baseline_program)
    mutated = _product(real_product_fixture, real_product_fixture.mutated_program)

    first = next(baseline.iter_factor2_pairs(pair_start=0, pair_count=1))
    changed = next(mutated.iter_factor2_pairs(pair_start=0, pair_count=1))

    assert not np.array_equal(first.scorer_y0, changed.scorer_y0)
    assert not np.array_equal(first.camera_y0, changed.camera_y0)
    assert first.proofs[0].certified_exact is True
    assert changed.proofs[0].certified_exact is True
    assert baseline.semantic_p_archive is real_product_fixture.decoder.carrier_receiver.archive
    assert baseline.program.compile_config.pair_count == 600


@_QUARANTINE_V15_PRODUCER_PIN
def test_product_refuses_wrong_p_a_decoder_type_and_decoder_source(
    real_product_fixture: _RealProductFixture,
) -> None:
    packet = encode_selected_preimage_program(real_product_fixture.baseline_program)
    with pytest.raises(G17G49SelectedProgramProductError, match="different bytes"):
        G17G49SelectedProgramPreplacementProductV1(
            semantic_p_archive=real_product_fixture.semantic_p + b"x",
            selected_program_packet=packet,
            decoder=real_product_fixture.decoder,
            maximum_packet_bytes=_MAX_PACKET_BYTES,
        )
    tampered = bytearray(packet)
    tampered[-1] ^= 1
    with pytest.raises(G17G49SelectedProgramProductError, match="strict parse"):
        G17G49SelectedProgramPreplacementProductV1(
            semantic_p_archive=real_product_fixture.semantic_p,
            selected_program_packet=bytes(tampered),
            decoder=real_product_fixture.decoder,
            maximum_packet_bytes=_MAX_PACKET_BYTES,
        )
    with pytest.raises(G17G49SelectedProgramProductError, match="concrete BoundV10"):
        G17G49SelectedProgramPreplacementProductV1(
            semantic_p_archive=real_product_fixture.semantic_p,
            selected_program_packet=packet,
            decoder=object(),  # type: ignore[arg-type]
            maximum_packet_bytes=_MAX_PACKET_BYTES,
        )
    wrong_source_program = replace(
        real_product_fixture.baseline_program,
        decoder_identity=GenericV10Factor2DecoderIdentityV1(
            implementation_source_sha256=_sha("9"),
        ),
    )
    with pytest.raises(G17G49SelectedProgramProductError, match="identity/source"):
        _product(real_product_fixture, wrong_source_program)


@_QUARANTINE_V15_PRODUCER_PIN
def test_exact_g_a_e_reopen_and_real_nested_p_refusal_emit_typed_product_blockers(
    real_product_fixture: _RealProductFixture,
) -> None:
    product = _product(real_product_fixture, real_product_fixture.baseline_program)
    proof = audit_g17_g49_selected_program_product_blockers(
        product=product,
    )

    assert proof.receipt.exact_g49_global_a_parse_reencode_twice is True
    assert proof.receipt.semantic_p_sha256 == product.semantic_p_sha256
    assert proof.receipt.selected_program_packet_sha256 == product.selected_program_packet_sha256
    assert proof.receipt.semantic_program_logical_value_type_available is False
    assert proof.receipt.selected_preimage_program_logical_value_type_available is False
    assert proof.receipt.product_structural_routing_closed is False
    assert proof.receipt.owner_specific_receiver_execution_evidence is False
    assert proof.receipt.g17_product_archive_built is False
    assert proof.receipt.g17_product_archive_reopened is False
    assert proof.receipt.open_product_blockers == (
        SEMANTIC_PROGRAM_TYPE_BLOCKER,
        SELECTED_PREIMAGE_PROGRAM_TYPE_BLOCKER,
        NESTED_P_CONTAINER_BLOCKER,
        INCIDENCE_EXECUTION_BLOCKER,
    )
    assert len(proof.receipt.selected_program_component_incidences) == 2
    framing, analytic = proof.receipt.selected_program_component_incidences
    assert framing.factor_role is None
    assert analytic.section_id == "g68.boundary_transport"
    assert analytic.factor_role == "ANALYTIC_RESIDUAL"
    assert analytic.scientific_role == G17ScientificRoleV1.BULK_BOUNDARY.value
    assert analytic.semantic_role == G17SemanticStreamRoleV1.RESIDUAL.value
    assert sum(row.byte_length for row in proof.receipt.selected_program_component_incidences) == len(
        product.selected_program_packet
    )
    assert proof.receipt.additional_unowned_or_unembedded_target_payload_inputs == 0
    assert proof.receipt.pointer_moved is False
    with pytest.raises(G17G49SelectedProgramProductError, match=NESTED_P_CONTAINER_BLOCKER):
        refuse_g17_g49_product_archive(proof)


def _stored_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, member)
    return output.getvalue()


def _incidence_placement_manifest(
    *,
    physical_receiver_operation: str = SHARED_ARCHIVE_UNPACK_OPERATION,
) -> G17CompilerPlacementManifestV1:
    topology_value = b"exact-topology-value"
    gauge_value = b"exact-gauge-value"
    member = topology_value + gauge_value
    archive = _stored_zip(member)
    topology_owner = G17LogicalOwnershipV1(
        owner_id="topology-owner",
        ownership_kind=G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
        value=G17SemanticTopologyV1(topology_value),
    )
    gauge_owner = G17LogicalOwnershipV1(
        owner_id="gauge-owner",
        ownership_kind=G17LogicalOwnershipKindV1.REALIZATION_GAUGE,
        value=G17RealizationGaugeV1(gauge_value),
    )

    def record(
        owner: G17LogicalOwnershipV1,
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
            physical_coding_group_id="joint-archive-group",
            video_specific_derivation=True,
        )

    records = (
        record(
            topology_owner,
            G17ScientificRoleV1.TOPOLOGY_WORLDSHEET,
            G17SemanticStreamRoleV1.SKELETON,
        ),
        record(
            gauge_owner,
            G17ScientificRoleV1.CELL_VALUE_PREIMAGE,
            G17SemanticStreamRoleV1.GAUGE,
        ),
    )
    group = G17PhysicalCodingGroupV1(
        group_id="joint-archive-group",
        exact_archive_bytes=archive,
        member_name="0.bin",
        exact_member_bytes=member,
        archive_offset=0,
        exact_range_bytes=archive,
        coder_owner="zip-store",
        container_owner="synthetic-incidence-mechanics",
        receiver_consumer=SHARED_ARCHIVE_RECEIVER_CONSUMER,
        receiver_operation=physical_receiver_operation,
        logical_owner_ids=(topology_owner.owner_id, gauge_owner.owner_id),
    )
    return G17CompilerPlacementManifestV1(
        records=records,
        coding_groups=(group,),
        expected_object_identities=(
            topology_owner.identity_sha256,
            gauge_owner.identity_sha256,
        ),
        exact_archive_bytes=archive,
        member_name="0.bin",
        exact_member_bytes=member,
    )


def test_owner_specific_receiver_incidences_preserve_shared_archive_truth() -> None:
    placement = _incidence_placement_manifest()
    incidences = (
        G17OwnerSpecificReceiverIncidenceV1(
            physical_group_id="joint-archive-group",
            logical_owner_id="gauge-owner",
            receiver_consumer="GAUGE_RECEIVER_V1",
            receiver_operation="REALIZE_GAUGE_V1",
        ),
        G17OwnerSpecificReceiverIncidenceV1(
            physical_group_id="joint-archive-group",
            logical_owner_id="topology-owner",
            receiver_consumer="TOPOLOGY_RECEIVER_V1",
            receiver_operation="REALIZE_TOPOLOGY_V1",
        ),
    )
    manifest = G17OwnerSpecificReceiverIncidenceManifestV1(
        placement_manifest=placement,
        incidences=incidences,
    )

    assert len(manifest.manifest_sha256) == 64
    assert placement.coding_groups[0].receiver_operation == SHARED_ARCHIVE_UNPACK_OPERATION
    assert {row.receiver_operation for row in manifest.incidences} == {
        "REALIZE_GAUGE_V1",
        "REALIZE_TOPOLOGY_V1",
    }
    wrong_group_scope = _incidence_placement_manifest(
        physical_receiver_operation="REALIZE_GAUGE_V1",
    )
    with pytest.raises(G17G49SelectedProgramProductError, match="shared physical group"):
        G17OwnerSpecificReceiverIncidenceManifestV1(
            placement_manifest=wrong_group_scope,
            incidences=incidences,
        )
    with pytest.raises(G17G49SelectedProgramProductError, match="leave a physical-group"):
        G17OwnerSpecificReceiverIncidenceManifestV1(
            placement_manifest=placement,
            incidences=incidences[:1],
        )
    duplicate_owner = tuple(
        sorted(
            (
                *incidences,
                G17OwnerSpecificReceiverIncidenceV1(
                    physical_group_id="joint-archive-group",
                    logical_owner_id="topology-owner",
                    receiver_consumer="OTHER_RECEIVER_V1",
                    receiver_operation="OTHER_OPERATION_V1",
                ),
            )
        )
    )
    with pytest.raises(G17G49SelectedProgramProductError, match="multiple competing"):
        G17OwnerSpecificReceiverIncidenceManifestV1(
            placement_manifest=placement,
            incidences=duplicate_owner,
        )
