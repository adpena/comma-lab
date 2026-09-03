# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import BoundaryShearletAtomV1
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    GENERIC_V10_FACTOR2_DECODER_ID,
    OPEN_SELECTED_SOLUTION_PRODUCT_BLOCKERS,
    V15_SEMANTIC_COMPILE_DERIVATION,
    V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
    ForbiddenSelectedPreimagePayloadClassV1,
    GenericV10Factor2DecoderIdentityV1,
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageCompileConfigV1,
    SelectedPreimageFactor2PairV1,
    SelectedPreimageFactorRoleV1,
    SelectedPreimageFrameSelectorV1,
    TaskspaceSelectedPreimageFactorV1,
    TaskspaceSelectedPreimageProgramError,
    TaskspaceSelectedPreimageProgramV1,
    V15SemanticProgramIdentityV1,
    build_analytic_shearlet_residual_factor,
    build_learned_irreducible_quotient_factor,
    compile_v9_v10_selected_preimage_program,
    decode_selected_preimage_pair,
    encode_selected_preimage_program,
    iter_selected_preimage_segment,
    parse_selected_preimage_program,
    realize_selected_preimage_pair_factor2,
    refuse_forbidden_selected_preimage_payload,
    verify_v15_semantic_compile_lineage,
)


def _sha(character: str) -> str:
    assert len(character) == 1
    return character * 64


def _test_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _semantic_identity() -> V15SemanticProgramIdentityV1:
    return V15SemanticProgramIdentityV1(
        fresh_compile_receipt_schema=V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
        fresh_compile_receipt_sha256=_sha("a"),
        compile_proof_dependency_sha256=_sha("0"),
        typed_compile_config_sha256=_sha("b"),
        compiler_source_sha256=_sha("c"),
        receiver_source_sha256=_sha("d"),
        compiled_semantic_archive_sha256=_sha("e"),
        compiled_semantic_archive_bytes=12_345,
        source_pair_start=0,
        pair_count=600,
        declared_compile_dependency_sha256s=(_sha("1"), _sha("2")),
    )


def _target_identity() -> ScorerTargetCustodyIdentityV1:
    return ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256=_sha("f"),
        target_bank_sha256=_sha("1"),
    )


def _analytic_factor(
    *,
    atoms: tuple[BoundaryShearletAtomV1, ...] | None = None,
    source_pair_stop_exclusive: int = 1,
    atom_role: str = "Road",
) -> TaskspaceSelectedPreimageFactorV1:
    if atoms is None:
        atoms = (
            BoundaryShearletAtomV1(
                pair_index=0,
                role=atom_role,
                center_y=160,
                center_x=256,
                scale_y=24,
                scale_x=96,
                shear_q4=0,
                amplitude_q4=64,
            ),
        )
    return build_analytic_shearlet_residual_factor(
        section_id="a.boundary_transport",
        source_pair_start=0,
        source_pair_stop_exclusive=source_pair_stop_exclusive,
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        source_rgb_u8=(10, 20, 30),
        added_rgb_u8=(70, 80, 90),
        removed_rgb_u8=(1, 2, 3),
        atoms=atoms,
        source_receipt_sha256=_sha("2"),
    )


def _learned_factor(
    *,
    active_pair_ranges: tuple[tuple[int, int], ...] | None = None,
    parameter_payload: bytes = b"\x01\x02\x03",
    section_id: str = "z.learned_quotient",
) -> TaskspaceSelectedPreimageFactorV1:
    return build_learned_irreducible_quotient_factor(
        section_id=section_id,
        source_pair_start=0,
        source_pair_stop_exclusive=600,
        decoder_contract_id="tac.test.compact_quotient_decoder.v1",
        decoder_implementation_source_sha256=_test_source_sha256(),
        model_family_id="compact_conditioned_pair_delta",
        latent_codec_id="delta_i8",
        parameter_codec_id="weights_i8",
        latent_dtype="int8",
        parameter_dtype="int8",
        latent_payload=b"\x01\xfe\x03\xfc",
        parameter_payload=parameter_payload,
        source_receipt_sha256=_sha("3"),
        active_pair_ranges=active_pair_ranges,
    )


def _program(
    *,
    factors: tuple[TaskspaceSelectedPreimageFactorV1, ...] | None = None,
    maximum_packet_bytes: int = 1 << 20,
) -> TaskspaceSelectedPreimageProgramV1:
    rows = factors or (_analytic_factor(), _learned_factor())
    canonical = tuple(
        sorted(
            rows,
            key=lambda factor: (
                0 if factor.role is SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL else 1,
                factor.source_pair_start,
                factor.source_pair_stop_exclusive,
                factor.section_id,
            ),
        )
    )
    return TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=_semantic_identity(),
        target_custody_identity=_target_identity(),
        decoder_identity=GenericV10Factor2DecoderIdentityV1.current(),
        factors=canonical,
        compile_config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=600,
            maximum_packet_bytes=maximum_packet_bytes,
            score_budget_receipt_sha256=_sha("4"),
            budget_rule_id="dynamic_frontier_coupled_score_budget_v1",
        ),
    )


class _Decoder:
    decoder_id = GENERIC_V10_FACTOR2_DECODER_ID

    def __init__(
        self,
        *,
        semantic_identity: V15SemanticProgramIdentityV1,
        target_identity: ScorerTargetCustodyIdentityV1,
        target_match: bool = True,
        inert_learned: bool = False,
        decoder_source_match: bool = True,
        learned_source_match: bool = True,
    ) -> None:
        self.semantic_identity = semantic_identity
        self.target_identity = target_identity
        self.target_match = target_match
        self.inert_learned = inert_learned
        self.decoder_source_match = decoder_source_match
        self.learned_source_match = learned_source_match

    @property
    def implementation_source_sha256(self) -> str:
        if not self.decoder_source_match:
            return _sha("9")
        return GenericV10Factor2DecoderIdentityV1.current().implementation_source_sha256

    def verify_semantic_program_identity(
        self,
        identity: V15SemanticProgramIdentityV1,
    ) -> bool:
        return identity == self.semantic_identity

    def verify_target_custody_identity(
        self,
        identity: ScorerTargetCustodyIdentityV1,
    ) -> bool:
        return self.target_match and identity == self.target_identity

    def decode_semantic_base_pair(
        self,
        source_pair_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del source_pair_id
        y0 = np.zeros((384, 512, 3), dtype=np.uint8)
        y1 = np.zeros((384, 512, 3), dtype=np.uint8)
        y0[160:220, 120:392] = (10, 20, 30)
        y1[150:230, 100:412] = (10, 20, 30)
        return y0, y1

    def learned_quotient_decoder_contract_id(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        return "tac.test.compact_quotient_decoder.v1"

    def learned_quotient_decoder_implementation_source_sha256(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        return _test_source_sha256() if self.learned_source_match else _sha("8")

    def apply_learned_irreducible_quotient(
        self,
        *,
        factor: TaskspaceSelectedPreimageFactorV1,
        source_pair_id: int,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
        assert 0 <= source_pair_id < 600
        output0 = scorer_y0.copy()
        output1 = scorer_y1.copy()
        if not self.inert_learned:
            output1[0, 0] = ((source_pair_id + factor.payload[-1]) % 251 + 1, 7, 9)
        return output0, output1

    def realize_factor2_pair(
        self,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> SelectedPreimageFactor2PairV1:
        operator = DisjointResizeOperator.build(
            camera_h=874,
            camera_w=1164,
            scorer_h=384,
            scorer_w=512,
        )
        camera0 = realize_factor2_uint8_scorer_plane(operator, scorer_y0)
        camera1 = realize_factor2_uint8_scorer_plane(operator, scorer_y1)
        return SelectedPreimageFactor2PairV1(
            scorer_y0=scorer_y0,
            scorer_y1=scorer_y1,
            camera_y0=camera0,
            camera_y1=camera1,
            proofs=(
                verify_factor2_uint8_scorer_plane(operator, camera0, scorer_y0),
                verify_factor2_uint8_scorer_plane(operator, camera1, scorer_y1),
            ),
        )


def _decoder(program: TaskspaceSelectedPreimageProgramV1, **kwargs: bool) -> _Decoder:
    return _Decoder(
        semantic_identity=program.semantic_program_identity,
        target_identity=program.target_custody_identity,
        **kwargs,
    )


def test_compile_encode_parse_back_is_exact_and_byte_homes_partition_packet() -> None:
    program = _program()
    packet = encode_selected_preimage_program(program)
    parsed = parse_selected_preimage_program(packet, maximum_packet_bytes=1 << 20)

    assert parsed == program
    assert parsed.packet_bytes == packet
    assert {factor.role for factor in parsed.factors} == {
        SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL,
        SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT,
    }
    homes = parsed.byte_homes()
    assert homes[0].offset == 0
    assert all(previous.offset + previous.byte_length == current.offset for previous, current in pairwise(homes))
    assert homes[-1].offset + homes[-1].byte_length == len(packet)
    assert sum(home.byte_length for home in homes) == len(packet)
    assert parsed.required_counted_input_source_bytes == (
        len(packet) + parsed.semantic_program_identity.compiled_semantic_archive_bytes
    )
    assert parsed.open_product_blockers == OPEN_SELECTED_SOLUTION_PRODUCT_BLOCKERS


def test_external_budget_pointer_control_does_not_mutate_receiver_packet_bytes() -> None:
    program = _program()
    refreshed_control = replace(
        program,
        compile_config=replace(
            program.compile_config,
            maximum_packet_bytes=2 << 20,
            score_budget_receipt_sha256=_sha("7"),
            budget_rule_id="refreshed_dynamic_frontier_budget_v2",
        ),
    )

    assert refreshed_control == program
    assert refreshed_control.packet_bytes == program.packet_bytes
    assert refreshed_control.packet_sha256 == program.packet_sha256


def test_pair_decoder_composes_behavior_distinct_analytic_and_learned_factors() -> None:
    program = _program()
    decoder = _decoder(program)
    base0, base1 = decoder.decode_semantic_base_pair(0)

    y0, y1 = decode_selected_preimage_pair(program, 0, decoder)
    repeat0, repeat1 = decode_selected_preimage_pair(program, 0, decoder)

    assert y0.dtype == y1.dtype == np.uint8
    assert not y0.flags.writeable and not y1.flags.writeable
    assert not np.array_equal(y0, base0)
    assert not np.array_equal(y1, base1)
    assert tuple(y1[0, 0]) == (4, 7, 9)
    assert np.array_equal(y0, repeat0)
    assert np.array_equal(y1, repeat1)


def test_120_pair_segment_iterator_is_resume_addressable_without_dense_bank() -> None:
    program = _program(factors=(_learned_factor(),))
    pair = next(
        iter_selected_preimage_segment(
            program,
            _decoder(program),
            segment_index=4,
        )
    )

    assert pair.segment_index == 4
    assert pair.segment_count == 5
    assert pair.pair_index == pair.source_pair_id == 480
    assert pair.program_packet_sha256 == program.packet_sha256
    assert pair.target_custody_receipt_sha256 == (program.target_custody_identity.target_custody_receipt_sha256)
    assert pair.target_bank_sha256 == program.target_custody_identity.target_bank_sha256
    assert tuple(pair.scorer_y1[0, 0]) == ((480 + 3) % 251 + 1, 7, 9)


def test_all_five_120_pair_segments_cover_n600_once_in_order() -> None:
    program = _program(factors=(_learned_factor(),))
    decoder = _decoder(program)
    observed = [
        (pair.pair_index, pair.source_pair_id)
        for segment_index in range(5)
        for pair in iter_selected_preimage_segment(
            program,
            decoder,
            segment_index=segment_index,
        )
    ]

    assert observed == [(pair_id, pair_id) for pair_id in range(600)]


def test_batch16_target_custody_is_closed_and_mismatch_fails_decode() -> None:
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="batch16"):
        replace(_target_identity(), scorer_batch_size=32)
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="batch16"):
        replace(
            _target_identity(),
            historical_batch32_targets_consumed=True,
        )

    program = _program()
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="batch16 target-custody"):
        decode_selected_preimage_pair(
            program,
            0,
            _decoder(program, target_match=False),
        )


_QUARANTINE_V15_PRODUCER_PIN = pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): retired July taskspace lineage, no live consumer. "
        "The sealed V15 compile receipt's producer_custody pins "
        "src/tac/optimization/direct_description_carrier_compose.py at 3e1f69bb/156,551 B; HEAD is "
        "6fef110d/160,470 B. Drift commits: 9934d488b then 36f4b2947 (both 2026-08-20). "
        "MEASURED, so the quarantine is not a guess: the delta is exactly 3 changed defs plus 2 added "
        "and 0 removed, all three ARE on the path the pinned producer calls, and the drifted receiver "
        "still decodes the sealed archive and refuses all 5 mutation samples with identical coverage "
        "(133,941 B) -- the sole receipt delta is an ADDED key, non_empty_member_payload_count, that "
        "no check reads. Output-equivalent is still not a pin refresh: the pin lives in a SEALED "
        "2026-07-26 custody receipt, and rewriting its producer_custody to match today's sources "
        "would make that receipt assert it was compiled by sources that did not exist at compile "
        "time. The honest cure is a NEW receipt from a fresh compile. "
        "FIRE TRIGGER: a scorer-authorized arm re-runs tools/measure_ddm_v15_scorer_solved_templates.py "
        "(it solves through exact R + SegNet, which ddm_ql1's charter forbade), emits a fresh receipt, "
        "and either confirms byte-identity -- then DELETE this mark -- or records real output drift. "
        "Note the same file text is re-checked at taskspace_selected_preimage_program_v1.py:1923 and "
        "taskspace_selected_preimage_program_v2.py:723, so only a receipt refresh cures all sites; a "
        "waiver of the :572 check alone would not. strict=True means this mark FAILS the moment the "
        "lineage is repaired. Owning memos: "
        ".omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md and "
        ".omx/research/ddm_cd1_working_tree_debt_landing_20260903.md"
    ),
)


@_QUARANTINE_V15_PRODUCER_PIN
def test_sealed_compile_lineage_accepts_fresh_deterministic_byte_identity() -> None:
    root = Path(__file__).resolve().parents[4]
    run_dir = (
        root
        / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
        / "fresh_v15_semantic_base_n600_20260726"
    )
    receipt_bytes = (run_dir / "ddm_v15_scorer_solved_templates_n600_receipt.json").read_bytes()
    archive_bytes = (run_dir / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes").read_bytes()
    identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=receipt_bytes,
        compiled_semantic_archive=archive_bytes,
        producer_root=root,
    )
    assert identity.compiled_semantic_archive_sha256 == (
        "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
    )
    assert identity.compile_derivation == V15_SEMANTIC_COMPILE_DERIVATION
    assert identity.compiled_semantic_archive_sha256 not in (identity.declared_compile_dependency_sha256s)

    program = compile_v9_v10_selected_preimage_program(
        semantic_compile_receipt_bytes=receipt_bytes,
        compiled_semantic_archive=archive_bytes,
        semantic_producer_root=root,
        target_custody_identity=_target_identity(),
        decoder_identity=GenericV10Factor2DecoderIdentityV1.current(),
        factors=(_analytic_factor(), _learned_factor()),
        config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=600,
            maximum_packet_bytes=1 << 20,
            score_budget_receipt_sha256=_sha("4"),
            budget_rule_id="dynamic_frontier_coupled_score_budget_v1",
        ),
    )
    assert program.semantic_program_identity == identity

    unsealed = json.loads(receipt_bytes)
    unsealed["producer_custody"][0]["sha256"] = _sha("9")
    tampered_receipt = json.dumps(
        unsealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="source drifted"):
        verify_v15_semantic_compile_lineage(
            compile_receipt_bytes=tampered_receipt,
            compiled_semantic_archive=archive_bytes,
            producer_root=root,
        )

    copied_input = json.loads(receipt_bytes)
    copied_input["typed_config"]["solve_archive_sha256"] = identity.compiled_semantic_archive_sha256
    copied_input["typed_config_sha256"] = hashlib.sha256(
        json.dumps(
            copied_input["typed_config"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    copied_input_receipt = json.dumps(
        copied_input,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="compile dependency"):
        verify_v15_semantic_compile_lineage(
            compile_receipt_bytes=copied_input_receipt,
            compiled_semantic_archive=archive_bytes,
            producer_root=root,
        )


def test_forbidden_payload_classes_fail_closed() -> None:
    for payload_class in ForbiddenSelectedPreimagePayloadClassV1:
        with pytest.raises(TaskspaceSelectedPreimageProgramError, match="forbidden"):
            refuse_forbidden_selected_preimage_payload(payload_class)


def test_parser_rejects_payload_tamper_trailing_bytes_and_tight_bound() -> None:
    packet = _program().packet_bytes
    tampered = bytearray(packet)
    tampered[-1] ^= 1

    with pytest.raises(TaskspaceSelectedPreimageProgramError):
        parse_selected_preimage_program(bytes(tampered), maximum_packet_bytes=1 << 20)
    with pytest.raises(TaskspaceSelectedPreimageProgramError):
        parse_selected_preimage_program(packet + b"x", maximum_packet_bytes=1 << 20)
    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="bound"):
        parse_selected_preimage_program(packet, maximum_packet_bytes=len(packet) - 1)


def test_packet_ceiling_not_arbitrary_atom_or_factor_count_caps_bounds_parse() -> None:
    atoms = tuple(
        BoundaryShearletAtomV1(
            pair_index=0,
            role="Road",
            center_y=index // 512,
            center_x=index % 512,
            scale_y=2,
            scale_x=4,
            shear_q4=0,
            amplitude_q4=1,
        )
        for index in range(16_385)
    )
    analytic = _analytic_factor(atoms=atoms)
    analytic_program = _program(
        factors=(analytic,),
        maximum_packet_bytes=8 << 20,
    )
    assert len(analytic_program.packet_bytes) > 1 << 20
    assert (
        parse_selected_preimage_program(
            analytic_program.packet_bytes,
            maximum_packet_bytes=len(analytic_program.packet_bytes),
        )
        == analytic_program
    )

    factors = tuple(
        _learned_factor(
            active_pair_ranges=((0, 1),),
            section_id=f"learned.{index:04d}",
        )
        for index in range(4_097)
    )
    factor_program = _program(
        factors=factors,
        maximum_packet_bytes=16 << 20,
    )
    assert len(factor_program.factors) == 4_097
    assert (
        parse_selected_preimage_program(
            factor_program.packet_bytes,
            maximum_packet_bytes=len(factor_program.packet_bytes),
        )
        == factor_program
    )


def test_counted_addressed_factor_cannot_be_receiver_inert() -> None:
    program = _program(factors=(_learned_factor(),))

    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="receiver-inert"):
        decode_selected_preimage_pair(
            program,
            19,
            _decoder(program, inert_learned=True),
        )


def test_sparse_learned_support_does_not_false_reject_pairs_outside_support() -> None:
    factor = _learned_factor(active_pair_ranges=((0, 1), (20, 21)))
    program = _program(factors=(factor,))
    decoder = _decoder(program, inert_learned=True)

    base0, base1 = decoder.decode_semantic_base_pair(19)
    y0, y1 = decode_selected_preimage_pair(program, 19, decoder)
    assert np.array_equal(y0, base0)
    assert np.array_equal(y1, base1)

    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="receiver-inert"):
        decode_selected_preimage_pair(program, 20, decoder)


def test_factor_stage_order_is_analytic_then_learned_not_interval_dependent() -> None:
    analytic = _analytic_factor(source_pair_stop_exclusive=600)
    learned = _learned_factor(active_pair_ranges=((0, 1),))
    program = _program(factors=(learned, analytic))

    assert [factor.role for factor in program.factors] == [
        SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL,
        SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT,
    ]


def test_decoder_source_identity_is_executably_bound() -> None:
    program = _program(factors=(_learned_factor(active_pair_ranges=((0, 1),)),))

    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="implementation source"):
        decode_selected_preimage_pair(
            program,
            0,
            _decoder(program, decoder_source_match=False),
        )


def test_learned_plugin_contract_and_source_are_executably_bound() -> None:
    program = _program(factors=(_learned_factor(active_pair_ranges=((0, 1),)),))

    with pytest.raises(TaskspaceSelectedPreimageProgramError, match="learned quotient decoder source"):
        decode_selected_preimage_pair(
            program,
            0,
            _decoder(program, learned_source_match=False),
        )


def test_analytic_donor_role_token_is_not_charged_when_receiver_ignores_it() -> None:
    road = _analytic_factor(atom_role="Road")
    boundary = _analytic_factor(atom_role="UndrivableBoundary")

    assert road.payload == boundary.payload
    assert road.payload_sha256 == boundary.payload_sha256


def test_learned_receiver_consumes_counted_payload_and_factor2_is_exact_for_both_planes() -> None:
    first = _program(
        factors=(
            _learned_factor(
                active_pair_ranges=((0, 1),),
                parameter_payload=b"\x01\x02\x03",
            ),
        )
    )
    second = _program(
        factors=(
            _learned_factor(
                active_pair_ranges=((0, 1),),
                parameter_payload=b"\x01\x02\x04",
            ),
        )
    )

    first_y = decode_selected_preimage_pair(first, 0, _decoder(first))
    second_y = decode_selected_preimage_pair(second, 0, _decoder(second))
    assert not np.array_equal(first_y[1], second_y[1])

    realized = realize_selected_preimage_pair_factor2(first, 0, _decoder(first))
    assert realized.camera_y0.shape == realized.camera_y1.shape == (874, 1164, 3)
    assert all(proof.certified_exact and proof.numerator_exact for proof in realized.proofs)
