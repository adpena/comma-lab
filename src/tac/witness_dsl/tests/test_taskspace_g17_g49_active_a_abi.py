# SPDX-License-Identifier: MIT
"""Structural G69 proofs for the canonical G17 active-A carriage of G49.

These fixtures contain synthetic identity hashes and one tiny analytic factor.
They are protocol tests only: no scorer, dense target/plane payload, candidate,
score, evaluator, or pointer claim is present.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tac.optimization.direct_description_carrier_compose import BoundaryShearletAtomV1
from tac.witness_dsl.taskspace_g17_production_envelope import (
    GA_HEADER_BYTES,
    G17ADescriptorV1,
    G17AFamily,
    G17AMode,
    G17G49SelectedPreimageStrictParserV1,
    G17PopulationLayout,
    G17ProductionEnvelopeBlocker,
    G17ProductionEnvelopeError,
    G17TerminalAFamily,
    G17TerminalAMode,
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_production_archive,
    build_g17_terminal_envelope,
    parse_g17_a_packet,
    parse_g17_production_archive,
    parse_g17_terminal_envelope,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
    GenericV10Factor2DecoderIdentityV1,
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageCompileConfigV1,
    SelectedPreimageFrameSelectorV1,
    TaskspaceSelectedPreimageProgramV1,
    V15SemanticProgramIdentityV1,
    build_analytic_shearlet_residual_factor,
)


def _sha(character: str) -> str:
    assert len(character) == 1
    return character * 64


def _program(pair_count: int) -> TaskspaceSelectedPreimageProgramV1:
    semantic = V15SemanticProgramIdentityV1(
        fresh_compile_receipt_schema=V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
        fresh_compile_receipt_sha256=_sha("a"),
        compile_proof_dependency_sha256=_sha("0"),
        typed_compile_config_sha256=_sha("b"),
        compiler_source_sha256=_sha("c"),
        receiver_source_sha256=_sha("d"),
        compiled_semantic_archive_sha256=_sha("e"),
        compiled_semantic_archive_bytes=12_345,
        source_pair_start=0,
        pair_count=pair_count,
        declared_compile_dependency_sha256s=(_sha("1"), _sha("2")),
    )
    target = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256=_sha("f"),
        target_bank_sha256=_sha("1"),
    )
    decoder = GenericV10Factor2DecoderIdentityV1.current()
    factor = build_analytic_shearlet_residual_factor(
        section_id="a.boundary_transport",
        source_pair_start=0,
        source_pair_stop_exclusive=pair_count,
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        source_rgb_u8=(10, 20, 30),
        added_rgb_u8=(70, 80, 90),
        removed_rgb_u8=(1, 2, 3),
        atoms=(
            BoundaryShearletAtomV1(
                pair_index=0,
                role="Road",
                center_y=160,
                center_x=256,
                scale_y=24,
                scale_x=96,
                shear_q4=0,
                amplitude_q4=64,
            ),
        ),
        source_receipt_sha256=_sha("2"),
    )
    return TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=semantic,
        target_custody_identity=target,
        decoder_identity=decoder,
        compile_config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=pair_count,
            maximum_packet_bytes=1 << 20,
            score_budget_receipt_sha256=_sha("4"),
            budget_rule_id="g69_structural_test_only_v1",
        ),
        factors=(factor,),
    )


def _adapter(program: TaskspaceSelectedPreimageProgramV1) -> G17G49SelectedPreimageStrictParserV1:
    return G17G49SelectedPreimageStrictParserV1(
        expected_semantic_program_identity=program.semantic_program_identity,
        expected_target_custody_identity=program.target_custody_identity,
        expected_decoder_identity=program.decoder_identity,
        maximum_packet_bytes=len(program.packet_bytes),
    )


def test_a_and_terminal_wire_values_are_append_only() -> None:
    assert {member.name: member.value for member in G17AFamily} == {
        "CANONICAL_PASS": 0,
        "NATIVE_PASS_CONDITIONAL": 1,
        "NATIVE_SELECTIVE_NO_G8": 2,
        "NATIVE_SELECTIVE_POST_G8": 3,
        "G13_PASS_SOURCE_XIP2": 4,
        "G17_GENERAL_CONDITIONAL_XIP2": 5,
        "G49_SELECTED_PREIMAGE_PROGRAM": 6,
    }
    assert {member.name: member.value for member in G17AMode} == {
        "PASS_P0": 0,
        "SPARSE_CONSTANT_RGB": 1,
        "COPY_FINAL_Y1_SUPPORT": 2,
        "GLOBAL_COPY_FINAL_Y1": 3,
        "QUANTIZED_XIP2": 4,
        "SELECTED_PREIMAGE_PROGRAM": 5,
    }
    assert {member.name: member.value for member in G17TerminalAFamily} == {
        "PASS": 0,
        "NATIVE": 1,
        "G13": 2,
        "G17_GENERAL": 3,
        "MIXED": 4,
        "G49_SELECTED_PREIMAGE": 5,
    }
    assert {member.name: member.value for member in G17TerminalAMode} == {
        "PASS": 0,
        "SPARSE_CONSTANT": 1,
        "COPY_SUPPORT": 2,
        "GLOBAL_COPY": 3,
        "QUANTIZED_XIP2": 4,
        "MIXED": 5,
        "SELECTED_PREIMAGE_PROGRAM": 6,
    }


@pytest.mark.parametrize("pair_count", (2, 600))
def test_g49_program_is_exact_global_g17_a_and_archive_at_structural_windows(
    pair_count: int,
) -> None:
    p_section = f"g69-structural-p-n{pair_count}".encode()
    g_section = build_g17_g_packet(
        p_section=p_section,
        pair_start=0,
        pair_count=pair_count,
    )
    program = _program(pair_count)
    payload = program.packet_bytes
    adapter = _adapter(program)
    active = adapter(
        payload,
        0,
        pair_count,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    descriptor = G17ADescriptorV1(
        pair_start=0,
        pair_count=pair_count,
        family=G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        mode=G17AMode.SELECTED_PREIMAGE_PROGRAM,
        active=active,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=pair_count,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(descriptor,),
    )
    parsed_a = parse_g17_a_packet(
        a_section,
        expected_p_section=p_section,
        expected_g_section=g_section,
        active_parser=adapter,
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        a_active_parser=adapter,
    )
    parsed_terminal = parse_g17_terminal_envelope(
        terminal,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        a_active_parser=adapter,
    )
    built = build_g17_production_archive(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        terminal_section=terminal,
        a_active_parser=adapter,
    )
    reopened = parse_g17_production_archive(
        built.selected.outer.archive_bytes,
        a_active_parser=adapter,
    )

    assert parsed_a.packet == a_section
    assert parsed_a.layout is G17PopulationLayout.GLOBAL
    assert parsed_a.descriptor_windows == ((0, pair_count),)
    assert parsed_a.descriptors[0].payload == payload
    assert parsed_a.descriptors[0].active is not None
    assert parsed_a.descriptors[0].active.reencoded_payload == payload
    assert parsed_a.descriptors[0].active.parsed_object == program
    assert parsed_terminal.a_family is G17TerminalAFamily.G49_SELECTED_PREIMAGE
    assert parsed_terminal.a_mode is G17TerminalAMode.SELECTED_PREIMAGE_PROGRAM
    assert reopened.a_packet.packet == a_section
    assert reopened.a_packet.descriptors[0].active is not None
    assert reopened.a_packet.descriptors[0].active.reencoded_payload == payload
    assert reopened.research_only is True
    assert reopened.candidate_claim is False
    assert reopened.score_claim is False
    assert reopened.pointer_moved is False


def test_g49_program_refuses_sharded_layout_and_tight_parser_ceiling() -> None:
    p_section = b"g69-global-only-p"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=2)
    program = _program(2)
    adapter = _adapter(program)
    active = adapter(
        program.packet_bytes,
        0,
        2,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    descriptor = G17ADescriptorV1(
        0,
        2,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
        active,
    )

    with pytest.raises(G17ProductionEnvelopeError, match="population-global"):
        build_g17_a_packet(
            p_section=p_section,
            g_section=g_section,
            pair_start=0,
            pair_count=2,
            layout=G17PopulationLayout.SHARDED,
            descriptors=(descriptor,),
        )

    tight_adapter = replace(adapter, maximum_packet_bytes=len(program.packet_bytes) - 1)
    with pytest.raises(G17ProductionEnvelopeError, match="frozen strict parser"):
        tight_adapter(
            program.packet_bytes,
            0,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )


def test_g49_adapter_refuses_mutation_foreign_family_mode_magic_and_window() -> None:
    program = _program(2)
    payload = program.packet_bytes
    adapter = _adapter(program)
    mutated = bytearray(payload)
    mutated[-1] ^= 1
    wrong_magic = b"NOTPPV1\x00" + payload[8:]

    with pytest.raises(G17ProductionEnvelopeError, match="frozen strict parser"):
        adapter(
            bytes(mutated),
            0,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )
    with pytest.raises(G17ProductionEnvelopeError, match="foreign A family or mode"):
        adapter(
            payload,
            0,
            2,
            G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )
    with pytest.raises(G17ProductionEnvelopeError, match="foreign A family or mode"):
        adapter(
            payload,
            0,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.QUANTIZED_XIP2,
        )
    with pytest.raises(G17ProductionEnvelopeError, match="magic or mode"):
        adapter(
            wrong_magic,
            0,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )
    with pytest.raises(G17ProductionEnvelopeError, match="windows differ"):
        adapter(
            payload,
            1,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )


def test_g49_descriptor_and_terminal_wire_mutations_refuse() -> None:
    p_section = b"g69-wire-mutation-p"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=2)
    program = _program(2)
    adapter = _adapter(program)
    active = adapter(
        program.packet_bytes,
        0,
        2,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=2,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(
            G17ADescriptorV1(
                0,
                2,
                G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
                G17AMode.SELECTED_PREIMAGE_PROGRAM,
                active,
            ),
        ),
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        a_active_parser=adapter,
    )
    descriptor_mutation = bytearray(a_section)
    descriptor_mutation[GA_HEADER_BYTES + 4] ^= 1
    terminal_mutation = bytearray(terminal)
    terminal_mutation[-1] ^= 1

    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_a_packet(
            bytes(descriptor_mutation),
            expected_p_section=p_section,
            expected_g_section=g_section,
            active_parser=adapter,
        )
    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_terminal_envelope(
            bytes(terminal_mutation),
            p_section=p_section,
            g_section=g_section,
            a_section=a_section,
            a_active_parser=adapter,
        )


@pytest.mark.parametrize("identity_kind", ("semantic", "target", "decoder"))
def test_g49_adapter_refuses_each_wrong_expected_identity(identity_kind: str) -> None:
    program = _program(2)
    kwargs = {
        "expected_semantic_program_identity": program.semantic_program_identity,
        "expected_target_custody_identity": program.target_custody_identity,
        "expected_decoder_identity": program.decoder_identity,
        "maximum_packet_bytes": len(program.packet_bytes),
    }
    if identity_kind == "semantic":
        kwargs["expected_semantic_program_identity"] = replace(
            program.semantic_program_identity,
            compile_proof_dependency_sha256=_sha("7"),
        )
    elif identity_kind == "target":
        kwargs["expected_target_custody_identity"] = replace(
            program.target_custody_identity,
            target_bank_sha256=_sha("8"),
        )
    else:
        kwargs["expected_decoder_identity"] = replace(
            program.decoder_identity,
            implementation_source_sha256=_sha("9"),
        )
    adapter = G17G49SelectedPreimageStrictParserV1(**kwargs)

    with pytest.raises(G17ProductionEnvelopeError, match=f"{identity_kind} identity differs"):
        adapter(
            program.packet_bytes,
            0,
            2,
            G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
            G17AMode.SELECTED_PREIMAGE_PROGRAM,
        )


def test_g49_active_a_still_requires_the_explicit_strict_adapter() -> None:
    p_section = b"g69-explicit-parser-p"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=2)
    program = _program(2)
    active = _adapter(program)(
        program.packet_bytes,
        0,
        2,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=2,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(
            G17ADescriptorV1(
                0,
                2,
                G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
                G17AMode.SELECTED_PREIMAGE_PROGRAM,
                active,
            ),
        ),
    )

    with pytest.raises(G17ProductionEnvelopeBlocker, match="strict parser"):
        parse_g17_a_packet(
            a_section,
            expected_p_section=p_section,
            expected_g_section=g_section,
        )
