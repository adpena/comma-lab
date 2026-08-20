# SPDX-License-Identifier: MIT
"""Behavior proofs for G107's sole conditional-Y0 product contract.

These tests use a deterministic synthetic SemanticRoot fixture and never
measure SegNet, PoseNet, a source video, or a contest score.  They prove wire,
receiver, ownership, and arithmetic behavior only.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import numpy as np
import pytest

import tac.witness_dsl.taskspace_g107_g94v2_conditional_y0_final_semanticroot_y1_v1 as g107
from tac.contest_score import compute_contest_score
from tac.witness_dsl.taskspace_g107_g94v2_conditional_y0_final_semanticroot_y1_v1 import (
    EXACT_AUTHORITY_BLOCKER,
    FRESH_SOURCE_LINEAGE_BLOCKER,
    PUBLIC_RECEIVER_BLOCKER,
    G107ConditionalY0Error,
    G107ConditionalY0ReceiverV1,
    G107ConditionalY0SourceLineageV1,
    account_g107_candidate_bytes,
    audit_conditional_operand_effects,
    bind_g107_source_lineage_to_g17,
    bind_g107_to_g17,
    compute_final_y1_binding,
    conditional_y0_g17_logical_value,
    encode_g107_conditional_y0_v1,
    encode_g107_source_lineage_manifest,
    model_g107_joint_feasibility,
    parse_g107_conditional_y0_v1,
    target_byte_budget_for_score,
)
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    PAIR_COUNT_N600,
    PairRGBGaugeV1,
    RGBGaugeOwnershipV1,
    SemanticRootY1V1,
    semantic_root_g17_logical_values,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ChronologicalPosePreimageV1,
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
)
from tac.witness_dsl.tests.test_taskspace_pfree_semantic_root_v1 import (
    _owners,
    _population,
    _root,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _behavior_y1(_root_value: SemanticRootY1V1, pair_id: int) -> np.ndarray:
    frame = np.empty((384, 512, 3), dtype=np.uint8)
    frame[:, :, 0] = 96 + pair_id % 7
    frame[:, :, 1] = 112 + pair_id % 11
    frame[:, :, 2] = 128 + pair_id % 13
    return frame


def _operands() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis = np.array(
        [
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            [
                [[-3, 2, -1], [1, -2, 3]],
                [[4, -5, 6], [-7, 8, -9]],
            ],
        ],
        dtype=np.int8,
    )
    basis_scales = np.array([4.0, 2.0], dtype=np.float32)
    rows = np.arange(PAIR_COUNT_N600, dtype=np.int16)
    coefficients = np.stack((rows % 3 + 2, rows % 5 + 1), axis=1).astype(np.int16)
    coefficient_scales = np.array([1.0, 0.75], dtype=np.float32)
    return (
        np.ascontiguousarray(basis),
        np.ascontiguousarray(basis_scales),
        np.ascontiguousarray(coefficients),
        np.ascontiguousarray(coefficient_scales),
    )


@pytest.fixture
def product(monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    root = _root()
    population = _population()

    def behavior_population_sha(root_value: SemanticRootY1V1, *, batch_size: int = 16) -> str:
        assert 1 <= batch_size <= 16
        return _sha(f"ordered-n600-y1:{root_value.packet_sha256}")

    monkeypatch.setattr(g107, "semantic_root_y1_population_sha256", behavior_population_sha)
    monkeypatch.setattr(g107, "render_semantic_root_y1_scorer", _behavior_y1)
    binding = compute_final_y1_binding(root, population=population)
    basis, basis_scales, coefficients, coefficient_scales = _operands()
    packet = encode_g107_conditional_y0_v1(
        root,
        population=population,
        expected_final_y1_binding_sha256=binding,
        basis_q=basis,
        basis_scales=basis_scales,
        coefficients_q=coefficients,
        coefficient_scales=coefficient_scales,
    )
    parsed = parse_g107_conditional_y0_v1(
        packet,
        population=population,
        expected_final_y1_binding_sha256=binding,
    )
    return root, population, binding, packet, parsed


def test_canonical_packet_parse_back_and_exact_section_spelling(product: tuple[object, ...]) -> None:
    root, _population_value, binding, packet, parsed = product
    assert parsed.root == root
    assert parsed.final_y1_binding_sha256 == binding
    assert parsed.packet_bytes == packet
    assert parsed.packet_sha256 == hashlib.sha256(packet).hexdigest()
    assert len(parsed.section_bytes) == 5
    assert parsed.rank == 2
    assert (parsed.grid_height, parsed.grid_width) == (2, 2)
    assert not parsed.basis_q.flags.writeable
    assert not parsed.coefficients_q.flags.writeable


def test_wrong_external_binding_refuses_before_decode(product: tuple[object, ...]) -> None:
    _root_value, population, _binding, packet, _parsed = product
    with pytest.raises(G107ConditionalY0Error, match="externally frozen"):
        parse_g107_conditional_y0_v1(
            packet,
            population=population,
            expected_final_y1_binding_sha256=_sha("wrong-final-y1"),
        )


def test_changed_semantic_root_cannot_reuse_frozen_final_y1_binding(
    product: tuple[object, ...],
) -> None:
    root, population, binding, _packet, _parsed = product
    changed = replace(root, background_role=type(root.background_role).LANE)
    basis, basis_scales, coefficients, coefficient_scales = _operands()
    with pytest.raises(G107ConditionalY0Error, match="wrong frozen final Y1"):
        encode_g107_conditional_y0_v1(
            changed,
            population=population,
            expected_final_y1_binding_sha256=binding,
            basis_q=basis,
            basis_scales=basis_scales,
            coefficients_q=coefficients,
            coefficient_scales=coefficient_scales,
        )


def test_duplicate_raw_gauge_population_is_forbidden(product: tuple[object, ...]) -> None:
    root, population, _binding, _packet, _parsed = product
    gauges = tuple(
        PairRGBGaugeV1(
            pair_id=pair_id,
            phase_u16=0,
            parallax_x_q8=0,
            parallax_y_q8=0,
            luma_bias=0,
            chroma_u_bias=0,
            chroma_v_bias=0,
            texture_gain_q8=256,
        )
        for pair_id in range(PAIR_COUNT_N600)
    )
    duplicate = replace(
        root,
        rgb_gauge_ownership=RGBGaugeOwnershipV1.EXPLICIT_NONOVERLAPPING_POST_GENERATOR,
        pair_rgb_gauges=gauges,
    )
    basis, basis_scales, coefficients, coefficient_scales = _operands()
    with pytest.raises(G107ConditionalY0Error, match="duplicate raw pair gauges"):
        encode_g107_conditional_y0_v1(
            duplicate,
            population=population,
            expected_final_y1_binding_sha256=_sha("unused"),
            basis_q=basis,
            basis_scales=basis_scales,
            coefficients_q=coefficients,
            coefficient_scales=coefficient_scales,
        )


def test_crc_eof_and_historical_payload_magic_fail_closed(product: tuple[object, ...]) -> None:
    _root_value, population, binding, packet, parsed = product
    tampered = bytearray(packet)
    tampered[-5] ^= 1
    with pytest.raises(G107ConditionalY0Error, match="CRC32"):
        parse_g107_conditional_y0_v1(
            bytes(tampered),
            population=population,
            expected_final_y1_binding_sha256=binding,
        )
    with pytest.raises(G107ConditionalY0Error, match="trailing bytes"):
        parse_g107_conditional_y0_v1(
            packet + b"x",
            population=population,
            expected_final_y1_binding_sha256=binding,
        )
    foreign = bytearray(packet)
    basis_offset = g107._HEADER.size + len(parsed.section_bytes[0])
    foreign[basis_offset : basis_offset + 7] = b"G95BAS1"
    with pytest.raises(G107ConditionalY0Error, match="historical/raster"):
        parse_g107_conditional_y0_v1(
            bytes(foreign),
            population=population,
            expected_final_y1_binding_sha256=binding,
        )


@pytest.mark.parametrize("operand", ["basis", "row", "rank"])
def test_dead_conditional_operand_forms_are_rejected(
    product: tuple[object, ...],
    operand: str,
) -> None:
    root, population, binding, _packet, _parsed = product
    basis, basis_scales, coefficients, coefficient_scales = _operands()
    if operand == "basis":
        basis[0] = 0
    elif operand == "row":
        coefficients[9] = 0
    else:
        coefficients[:, 1] = 0
    with pytest.raises(G107ConditionalY0Error, match=r"every conditional|every one"):
        encode_g107_conditional_y0_v1(
            root,
            population=population,
            expected_final_y1_binding_sha256=binding,
            basis_q=basis,
            basis_scales=basis_scales,
            coefficients_q=coefficients,
            coefficient_scales=coefficient_scales,
        )


def test_scorer_receiver_owns_y0_once_and_preserves_final_y1(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    result = G107ConditionalY0ReceiverV1.open(parsed).render_scorer_pair(17)
    assert result.scorer_pair.shape == (2, 384, 512, 3)
    assert result.scorer_pair.dtype == np.uint8
    assert np.array_equal(result.scorer_pair[1], _behavior_y1(parsed.root, 17))
    assert not np.array_equal(result.scorer_pair[0], result.scorer_pair[1])
    assert result.changed_y0_values > 0
    assert result.conditional_y0_owner_count == 1


def test_receiver_is_deterministic_and_parsed_arrays_cannot_drift(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    receiver = G107ConditionalY0ReceiverV1.open(parsed)
    first = receiver.render_scorer_pair(33).scorer_pair
    second = receiver.render_scorer_pair(33).scorer_pair
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        parsed.coefficients_q[33, 0] += 1


def test_full_ordered_n600_stream_has_no_gap_overlap_or_reordering(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    receiver = G107ConditionalY0ReceiverV1.open(parsed)
    starts: list[int] = []
    pair_ids: list[int] = []
    for start, batch in receiver.iter_n600(batch_size=16):
        starts.append(start)
        assert batch.shape[1:] == (2, 384, 512, 3)
        for local_index, pair in enumerate(batch):
            pair_id = start + local_index
            assert int(pair[1, 0, 0, 0]) == 96 + pair_id % 7
            pair_ids.append(pair_id)
    assert starts == list(range(0, PAIR_COUNT_N600, 16))
    assert pair_ids == list(range(PAIR_COUNT_N600))
    with pytest.raises(G107ConditionalY0Error, match="contiguous ordered"):
        receiver.render_scorer_batch((0, 2))


def test_both_planes_have_exact_v10_factor2_realization(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    result = G107ConditionalY0ReceiverV1.open(parsed).realize_v10_factor2_pair(4)
    assert result.camera_pair.shape == (2, 874, 1164, 3)
    assert result.camera_pair.dtype == np.uint8
    assert result.y0_proof.certified_exact
    assert result.y1_proof.certified_exact
    assert result.both_scorer_planes_exact


def test_each_conditional_operand_is_receiver_live_under_perturbation(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    audit = audit_conditional_operand_effects(parsed, pair_id=8)
    assert audit.basis_q_changes_y0
    assert audit.basis_scales_change_y0
    assert audit.coefficients_q_change_y0
    assert audit.coefficient_scales_change_y0
    assert audit.all_conditional_operands_receiver_live
    assert audit.score_claim is False


def test_g17_seam_retains_existing_root_owners_and_one_exact_pose_owner(
    product: tuple[object, ...],
) -> None:
    root, population, _binding, _packet, parsed = product
    root_values = semantic_root_g17_logical_values(root)
    root_owners = _owners(root_values)
    conditional_value = conditional_y0_g17_logical_value(parsed)
    conditional_owner = G17LogicalOwnershipV1(
        owner_id="g107-sole-conditional-y0",
        ownership_kind=G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE,
        value=conditional_value,
    )
    binding = bind_g107_to_g17(
        parsed,
        population=population,
        topology_owner=root_owners[0],
        realization_owner=root_owners[1],
        generator_owner=root_owners[2],
        temporal_owner=root_owners[3],
        quotient_owner=root_owners[4],
        conditional_pose_owner=conditional_owner,
    )
    assert len(binding) == 64
    wrong = G17LogicalOwnershipV1(
        owner_id="g107-wrong-conditional-y0",
        ownership_kind=G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE,
        value=G17ChronologicalPosePreimageV1(b"wrong"),
    )
    with pytest.raises(G107ConditionalY0Error, match="exactly retained"):
        bind_g107_to_g17(
            parsed,
            population=population,
            topology_owner=root_owners[0],
            realization_owner=root_owners[1],
            generator_owner=root_owners[2],
            temporal_owner=root_owners[3],
            quotient_owner=root_owners[4],
            conditional_pose_owner=wrong,
        )


def test_fresh_own_lineage_is_external_packet_bound_encoder_evidence(
    product: tuple[object, ...],
) -> None:
    _root_value, _population_value, _binding, packet, parsed = product
    lineage = G107ConditionalY0SourceLineageV1(
        packet_sha256=parsed.packet_sha256,
        final_y1_binding_sha256=parsed.final_y1_binding_sha256,
        conditional_operand_sha256=parsed.conditional_operand_sha256,
        source_video_sha256=_sha("fresh-source-video"),
        target_custody_sha256=_sha("fresh-targets-and-scorers"),
        pose_target_binding_sha256=_sha("fresh-upstream-batch16-source-pose-target"),
        pose_candidate_batch16_contract_sha256=_sha("ordered-n600-final-y0-y1-batch16"),
        compiler_source_sha256=_sha("fresh-g107-compiler"),
        compile_config_sha256=_sha("fresh-g107-config"),
        originality_declaration_sha256=_sha("no-v15-c1-pr-payload-reuse"),
        pose_target_mode="FRESH_UPSTREAM_BATCH16_TARGET",
    )
    manifest = encode_g107_source_lineage_manifest(parsed, lineage)
    assert manifest not in packet
    owner = G17LogicalOwnershipV1(
        owner_id="g107-external-lineage",
        ownership_kind=G17LogicalOwnershipKindV1.ENCODER_EVIDENCE,
        value=G17EncoderOnlyTeacherOracleEvidenceV1(manifest),
    )
    assert len(bind_g107_source_lineage_to_g17(parsed, lineage, lineage_owner=owner)) == 64
    with pytest.raises(G107ConditionalY0Error, match="does not bind"):
        encode_g107_source_lineage_manifest(
            parsed,
            replace(lineage, packet_sha256=_sha("other-packet")),
        )
    with pytest.raises(G107ConditionalY0Error, match="batch16 pose custody"):
        replace(lineage, pose_target_batch_size=1)
    with pytest.raises(G107ConditionalY0Error, match="batch16 pose custody"):
        replace(lineage, batch32_verdict_rows_are_authority=True)


def test_candidate_byte_accounting_prices_exact_selected_outer_zip(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, packet, parsed = product
    accounting = account_g107_candidate_bytes(parsed)
    assert accounting.packet_nbytes == len(packet)
    assert accounting.semantic_root_nbytes == len(parsed.section_bytes[0])
    assert accounting.conditional_coefficients_nbytes == PAIR_COUNT_N600 * parsed.rank * 2
    assert accounting.selected_member_sha256 == parsed.packet_sha256
    assert accounting.exact_outer_archive.selected.member_bytes == packet
    assert accounting.selected_archive_nbytes == min(
        accounting.stored_archive_nbytes,
        accounting.deflated_archive_nbytes,
    )
    assert accounting.research_only is True
    assert accounting.candidate_claim is False
    assert accounting.score_claim is False


def test_joint_feasibility_is_one_strict_contest_score_sublevel(product: tuple[object, ...]) -> None:
    _root_value, _population_value, _binding, _packet, parsed = product
    accounting = account_g107_candidate_bytes(parsed)
    modeled = model_g107_joint_feasibility(
        accounting,
        d_seg=0.0006,
        d_pose=0.0003,
        target_score=0.172,
        pose_measurement_batch_size=16,
    )
    assert modeled.modeled_score == compute_contest_score(0.0006, 0.0003, accounting.selected_archive_nbytes)
    assert modeled.strict_score_slack == 0.172 - modeled.modeled_score
    assert modeled.strict_sublevel_feasible is (modeled.modeled_score < 0.172)
    assert modeled.target_byte_budget is not None
    assert modeled.strict_sublevel_feasible is (accounting.selected_archive_nbytes <= modeled.target_byte_budget)
    assert modeled.authority is False
    assert modeled.score_claim is False
    assert modeled.pose_measurement_pair_count == 600
    assert modeled.pose_measurement_batch_size == 16
    assert modeled.pose_measurement_chronological is True
    assert modeled.batch32_verdict_rows_are_authority is False
    with pytest.raises(G107ConditionalY0Error, match="batch32"):
        model_g107_joint_feasibility(
            accounting,
            d_seg=0.0006,
            d_pose=0.0003,
            target_score=0.172,
            pose_measurement_batch_size=32,
        )


def test_target_byte_budget_is_strict_and_has_no_independent_pose_gate() -> None:
    budget = target_byte_budget_for_score(target_score=0.172, d_seg=0.0008, d_pose=0.0001)
    assert budget is not None
    assert compute_contest_score(0.0008, 0.0001, budget) < 0.172
    assert compute_contest_score(0.0008, 0.0001, budget + 1) >= 0.172
    assert target_byte_budget_for_score(target_score=0.172, d_seg=0.002, d_pose=0.0) is None
    field_names = {item.name for item in fields(g107.G107JointFeasibilityV1)}
    assert "max_d_pose" not in field_names
    assert "max_d_seg" not in field_names


def test_contract_truth_labels_keep_pointer_and_public_closure_unclaimed() -> None:
    assert PUBLIC_RECEIVER_BLOCKER in g107.OPEN_BLOCKERS
    assert FRESH_SOURCE_LINEAGE_BLOCKER in g107.OPEN_BLOCKERS
    assert EXACT_AUTHORITY_BLOCKER in g107.OPEN_BLOCKERS
    parsed_fields = {item.name.lower() for item in fields(g107.ParsedG107ConditionalY0V1)}
    assert not parsed_fields & {
        "source_video",
        "scorer",
        "teacher",
        "target_table",
        "raw_gauges",
        "second_temporal_state",
        "v15_payload",
        "c1_payload",
        "pr_payload",
    }
