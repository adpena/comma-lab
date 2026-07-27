# SPDX-License-Identifier: MIT
"""Structural tests for the exact G17 production wire and archive seams.

These fixtures are protocol objects only.  They are not n600 scientific
evidence, candidate bytes, scorer evidence, or an evaluation row.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.witness_dsl.taskspace_g17_g_descriptor_custody import (
    build_g17_g_descriptor_acquisition_custody,
    parse_g17_g_descriptor_acquisition_custody,
)
from tac.witness_dsl.taskspace_g17_production_envelope import (
    G17AActiveNestedV1,
    G17ADescriptorV1,
    G17AFamily,
    G17AMode,
    G17G8SummaryV1,
    G17GActiveNestedV1,
    G17GDescriptorV1,
    G17GFamily,
    G17GMode,
    G17PopulationLayout,
    G17ProductionEnvelopeError,
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_post_g8_population_receipt,
    build_g17_post_topology_population_receipt,
    build_g17_production_archive,
    build_g17_terminal_envelope,
    canonical_g17_shard_windows,
    parse_g17_a_packet,
    parse_g17_g_packet,
    parse_g17_post_topology_population_receipt,
    parse_g17_production_archive,
    parse_g17_terminal_envelope,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(("pair_count", "descriptor_count"), ((2, 1), (24, 6), (600, 150)))
def test_pass_wire_roundtrip_at_all_production_population_sizes(
    pair_count: int,
    descriptor_count: int,
) -> None:
    p_section = b"structural-p-section-v1"
    g_section = build_g17_g_packet(
        p_section=p_section,
        pair_start=0,
        pair_count=pair_count,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=pair_count,
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )

    parsed_g = parse_g17_g_packet(g_section, expected_p_section=p_section)
    parsed_a = parse_g17_a_packet(
        a_section,
        expected_p_section=p_section,
        expected_g_section=g_section,
    )
    parsed_terminal = parse_g17_terminal_envelope(
        terminal,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )

    assert parsed_g.packet == g_section
    assert parsed_a.packet == a_section
    assert parsed_terminal.packet == terminal
    assert len(parsed_g.descriptors) == descriptor_count
    assert len(parsed_a.descriptors) == descriptor_count
    assert parsed_g.descriptor_windows == canonical_g17_shard_windows(0, pair_count)
    assert parsed_a.descriptor_windows == canonical_g17_shard_windows(0, pair_count)


def test_wire_mutation_parent_trailing_and_alternate_partition_refuse() -> None:
    p_section = b"p-parent-a"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=24)

    mutated = bytearray(g_section)
    mutated[-5] ^= 1
    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_g_packet(bytes(mutated), expected_p_section=p_section)
    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_g_packet(g_section + b"trailing", expected_p_section=p_section)
    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_g_packet(g_section, expected_p_section=b"p-parent-b")

    alternate = tuple(
        G17GDescriptorV1.canonical_pass(start, count)
        for start, count in ((0, 4), (4, 4), (8, 4), (12, 4), (16, 4), (19, 5))
    )
    with pytest.raises(G17ProductionEnvelopeError):
        build_g17_g_packet(
            p_section=p_section,
            pair_start=0,
            pair_count=24,
            descriptors=alternate,
        )


def test_global_and_sharded_a_are_non_aliasing_even_at_n2() -> None:
    p_section = b"p-global-layout"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=2)
    payload = b"TACX2A4\x00structural-parser-fixture"
    active = G17AActiveNestedV1(
        payload=payload,
        reencoded_payload=payload,
        pair_start=0,
        pair_count=2,
        family=G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
        mode=G17AMode.GLOBAL_COPY_FINAL_Y1,
        strict_parser_id="test.structural_tacx2a4_parser.v1",
        parsed_object=("strict-parser-fixture", payload),
    )
    descriptor = G17ADescriptorV1(
        0,
        2,
        G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
        G17AMode.GLOBAL_COPY_FINAL_Y1,
        active,
    )
    sharded = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=2,
        layout=G17PopulationLayout.SHARDED,
        descriptors=(descriptor,),
    )
    global_packet = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=2,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(descriptor,),
    )

    def reopen(
        nested: bytes,
        start: int,
        count: int,
        family: G17AFamily,
        mode: G17AMode,
    ) -> G17AActiveNestedV1:
        assert (nested, start, count, family, mode) == (
            payload,
            0,
            2,
            G17AFamily.G17_GENERAL_CONDITIONAL_XIP2,
            G17AMode.GLOBAL_COPY_FINAL_Y1,
        )
        return active

    assert sharded != global_packet
    assert (
        parse_g17_a_packet(
            sharded,
            expected_p_section=p_section,
            expected_g_section=g_section,
            active_parser=reopen,
        ).layout
        is G17PopulationLayout.SHARDED
    )
    assert (
        parse_g17_a_packet(
            global_packet,
            expected_p_section=p_section,
            expected_g_section=g_section,
            active_parser=reopen,
        ).layout
        is G17PopulationLayout.GLOBAL
    )


def test_cycle_free_receipts_recompute_arrays_and_causal_custody() -> None:
    p_section = b"p-cycle-free"
    payload = b"TACPG81\x00strict-branch-fixture"
    active = G17GActiveNestedV1(
        payload=payload,
        reencoded_payload=payload,
        pair_start=0,
        pair_count=2,
        family=G17GFamily.PASS_PREDICTOR,
        mode=G17GMode.SEMANTIC_THEN_FRESH_G8,
        strict_parser_id="test.structural_pass_g8_parser.v1",
        parsed_object=("strict-parser-fixture", payload),
    )
    descriptor = G17GDescriptorV1(
        0,
        2,
        G17GFamily.PASS_PREDICTOR,
        G17GMode.SEMANTIC_THEN_FRESH_G8,
        active,
    )
    g_section = build_g17_g_packet(
        p_section=p_section,
        pair_start=0,
        pair_count=2,
        descriptors=(descriptor,),
    )

    def reopen(
        nested: bytes,
        start: int,
        count: int,
        family: G17GFamily,
        mode: G17GMode,
    ) -> G17GActiveNestedV1:
        assert (nested, start, count, family, mode) == (
            payload,
            0,
            2,
            G17GFamily.PASS_PREDICTOR,
            G17GMode.SEMANTIC_THEN_FRESH_G8,
        )
        return active

    labels = np.zeros((2, 384, 512), dtype=np.uint8)
    post_topology_y1 = np.zeros((2, 874, 1164, 3), dtype=np.uint8)
    post_g8_y1 = post_topology_y1.copy()
    post_g8_y1[1, 4, 7, 2] = 1
    causal = b"exact-causal-p-receipt"
    predictor_binding = _digest(b"predictor-state-binding")
    post_topology = build_g17_post_topology_population_receipt(
        p_section_bytes=p_section,
        g_section_bytes=g_section,
        causal_p_receipt_bytes=causal,
        predictor_state_binding_sha256=predictor_binding,
        semantic_labels=labels,
        post_topology_camera_y1=post_topology_y1,
        g_active_parser=reopen,
    )
    post_g8 = build_g17_post_g8_population_receipt(
        post_topology_receipt=post_topology,
        p_section_bytes=p_section,
        g_section_bytes=g_section,
        semantic_labels=labels,
        post_g8_camera_y1=post_g8_y1,
        g_active_parser=reopen,
    )

    assert post_g8.g8_mode == G17G8SummaryV1.FRESH.value
    assert post_g8.post_topology_population_receipt_sha256 == post_topology.receipt_sha256
    with pytest.raises(G17ProductionEnvelopeError):
        parse_g17_post_topology_population_receipt(
            post_topology.to_receipt_bytes(),
            p_section_bytes=p_section,
            g_section_bytes=g_section,
            causal_p_receipt_bytes=b"stale-causal-receipt",
            predictor_state_binding_sha256=predictor_binding,
            semantic_labels=labels,
            post_topology_camera_y1=post_topology_y1,
            g_active_parser=reopen,
        )


def test_n600_archive_has_one_four_section_member_and_one_p_role() -> None:
    p_section = b"one-counted-p-section"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=600)
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=600,
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )
    built = build_g17_production_archive(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        terminal_section=terminal,
    )
    reopened = parse_g17_production_archive(built.selected.outer.archive_bytes)

    assert len(reopened.member.sections) == 4
    assert reopened.p_section_occurrences == 1
    assert reopened.p_section == p_section
    assert len(reopened.g_packet.descriptors) == 150
    assert reopened.research_only is True
    assert reopened.candidate_claim is False
    assert reopened.score_claim is False
    assert reopened.pointer_moved is False


def test_empty_pass_descriptor_custody_covers_every_descriptor_without_fake_rows() -> None:
    p_section = b"p-empty-pass-custody"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=24)
    custody = build_g17_g_descriptor_acquisition_custody(
        p_section_bytes=p_section,
        g_section_bytes=g_section,
        evidence=(),
    )
    reopened = parse_g17_g_descriptor_acquisition_custody(
        custody.receipt_bytes,
        p_section_bytes=p_section,
        g_section_bytes=g_section,
        evidence=(),
    )
    assert reopened.receipt.empty_pass_descriptor_indices == tuple(range(6))
    assert reopened.receipt.rows == ()
    assert reopened.receipt.whole_object_contains_no_exact_control is True
