# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.witness_dsl.progressive_geometry_residual import (
    ProgressiveGeometryResidualError,
    apply_progressive_geometry_residual,
    build_progressive_geometry_residual,
    decode_progressive_geometry_residual,
    packet_accounting,
)

PROGRAM = b"counted-factorized-v9-program"
CONTRACT = "tac.factorized_v9_semantic_predictor.v1"
RENDERER_SHA = hashlib.sha256(b"renderer-source-set").hexdigest()
PAIR_IDS = (448, 449, 450, 451)


def _labels() -> tuple[np.ndarray, np.ndarray]:
    predictor = np.full((4, 8, 10), 2, dtype=np.uint8)
    target = predictor.copy()

    # Frame 0 seeds one connected island and one singleton.  Frame 1 repeats
    # both exact correction states, so they belong to the temporal stratum.
    target[0, 1:3, 1:4] = 0
    target[0, 6, 8] = 4
    target[1] = target[0]

    # A different component and singleton are born in frame 2 and persist in
    # frame 3.  This exercises all three strata and staged application.
    target[2, 4:6, 5:7] = 1
    target[2, 0, 9] = 3
    target[3] = target[2]
    return predictor, target


def _build(*, mode: str = "auto", block_size: int | None = None) -> bytes:
    predictor, target = _labels()
    return build_progressive_geometry_residual(
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        target_labels=target,
        source_pair_ids=PAIR_IDS,
        target_semantic_lineage="synthetic_fixture",
        temporal_mode=mode,  # type: ignore[arg-type]
        temporal_block_size=block_size,
    )


@pytest.mark.parametrize(
    ("mode", "block_size"),
    (("auto", None), ("row_runs", None), ("block_context", 2), ("block_context", 4)),
)
def test_all_modes_exactly_recover_target_and_are_deterministic(mode: str, block_size: int | None) -> None:
    predictor, target = _labels()
    first = _build(mode=mode, block_size=block_size)
    second = _build(mode=mode, block_size=block_size)
    assert first == second
    recovered = apply_progressive_geometry_residual(
        first,
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        source_pair_ids=PAIR_IDS,
    )
    assert np.array_equal(recovered, target)


def test_three_staged_applications_close_the_exact_error_ledger_in_order() -> None:
    predictor, target = _labels()
    packet = _build()
    accounting = packet_accounting(packet)
    strata = accounting["strata"]
    assert [row["name"] for row in strata] == [
        "temporal_boundary",
        "component_islands",
        "sparse_tail",
    ]
    assert strata[0]["corrected_cells"] > 0
    assert strata[1]["record_count"] == 2
    assert strata[2]["record_count"] == 2
    assert strata[0]["errors_before"] > strata[0]["errors_after"]
    assert strata[0]["errors_after"] > strata[1]["errors_after"]
    assert strata[1]["errors_after"] > strata[2]["errors_after"] == 0
    assert accounting["separate_dense_target_table_section_bytes"] == 0
    assert accounting["pbr2_is_target_derived"] is True
    assert accounting["pbr2_target_derived_section_bytes"] == sum(row["payload_bytes"] for row in strata)
    assert accounting["pbr2_event_count"] == strata[0]["errors_before"]
    assert accounting["pbr2_event_density_numerator"] == strata[0]["errors_before"]
    assert accounting["pbr2_event_density_denominator"] == predictor.size
    assert accounting["target_derived_residual_promotion_admitted"] is False
    assert accounting["research_only"] is True
    assert accounting["artifact_role"] == "encoder_side_conditional_entropy_measurement"
    assert accounting["candidate_archive_admissible"] is False
    assert accounting["exact_target_semantic_reconstruction"] is True
    assert accounting["target_semantic_lineage"] == "synthetic_fixture"
    assert accounting["pbr2_reconstructs_exact_gt_argmax"] is False
    assert accounting["reconstructed_target_semantic_bytes"] == predictor.size
    assert accounting["candidate_archive_blocker"] == ("lossless predictor-conditional target-semantic-table encoding")
    assert accounting["generic_apply_requires_external_predictor_semantics"] is True
    assert accounting["physical_prefix_decode_supported"] is False
    assert accounting["staged_application_requires_complete_packet"] is True
    assert accounting["decode_scorer_dependency"] is False
    assert accounting["score_claim"] is False

    prefix0 = apply_progressive_geometry_residual(
        packet,
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        source_pair_ids=PAIR_IDS,
        max_strata=0,
    )
    prefix1 = apply_progressive_geometry_residual(
        packet,
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        source_pair_ids=PAIR_IDS,
        max_strata=1,
    )
    prefix2 = apply_progressive_geometry_residual(
        packet,
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        source_pair_ids=PAIR_IDS,
        max_strata=2,
    )
    prefix3 = apply_progressive_geometry_residual(
        packet,
        predictor_program=PROGRAM,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        source_pair_ids=PAIR_IDS,
        max_strata=3,
    )
    assert np.count_nonzero(prefix0 != target) == strata[0]["errors_before"]
    assert np.count_nonzero(prefix1 != target) == strata[0]["errors_after"]
    assert np.count_nonzero(prefix2 != target) == strata[1]["errors_after"]
    assert np.array_equal(prefix3, target)


def test_program_predictor_renderer_and_pair_coordinate_substitution_refuse() -> None:
    predictor, _ = _labels()
    packet = _build()
    common = {
        "payload": packet,
        "predictor_program": PROGRAM,
        "predictor_contract_id": CONTRACT,
        "predictor_renderer_sha256": RENDERER_SHA,
        "predictor_labels": predictor,
        "source_pair_ids": PAIR_IDS,
    }
    with pytest.raises(ProgressiveGeometryResidualError, match="program identity"):
        apply_progressive_geometry_residual(**{**common, "predictor_program": PROGRAM + b"x"})
    with pytest.raises(ProgressiveGeometryResidualError, match="contract"):
        apply_progressive_geometry_residual(**{**common, "predictor_contract_id": CONTRACT + ".other"})
    with pytest.raises(ProgressiveGeometryResidualError, match="renderer identity"):
        apply_progressive_geometry_residual(**{**common, "predictor_renderer_sha256": "0" * 64})
    with pytest.raises(ProgressiveGeometryResidualError, match="pair coordinates"):
        apply_progressive_geometry_residual(**{**common, "source_pair_ids": (449, 450, 451, 452)})
    mutated = predictor.copy()
    mutated[0, 0, 0] = 1
    with pytest.raises(ProgressiveGeometryResidualError, match="semantic stream"):
        apply_progressive_geometry_residual(**{**common, "predictor_labels": mutated})


def test_packet_mutation_and_noncontiguous_coordinates_refuse() -> None:
    predictor, target = _labels()
    packet = bytearray(_build())
    packet[len(packet) // 2] ^= 1
    with pytest.raises(ProgressiveGeometryResidualError, match="CRC"):
        decode_progressive_geometry_residual(bytes(packet))
    with pytest.raises(ProgressiveGeometryResidualError, match="contiguous ordered"):
        build_progressive_geometry_residual(
            predictor_program=PROGRAM,
            predictor_contract_id=CONTRACT,
            predictor_renderer_sha256=RENDERER_SHA,
            predictor_labels=predictor,
            target_labels=target,
            source_pair_ids=(448, 449, 451, 452),
            target_semantic_lineage="synthetic_fixture",
        )


def test_auto_mode_is_no_larger_than_each_callable_temporal_form() -> None:
    automatic = _build()
    row_runs = _build(mode="row_runs")
    block_two = _build(mode="block_context", block_size=2)
    block_four = _build(mode="block_context", block_size=4)
    assert len(automatic) <= min(len(row_runs), len(block_two), len(block_four))


def test_auto_mode_minimizes_whole_packet_not_only_temporal_payload() -> None:
    predictor = np.zeros((3, 4, 4), dtype=np.uint8)
    repeated = np.asarray(
        [[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]],
        dtype=np.uint8,
    )
    target = np.broadcast_to(repeated, predictor.shape).copy()
    common = {
        "predictor_program": b"x",
        "predictor_contract_id": "c",
        "predictor_renderer_sha256": hashlib.sha256(b"r").hexdigest(),
        "predictor_labels": predictor,
        "target_labels": target,
        "source_pair_ids": range(3),
        "target_semantic_lineage": "synthetic_fixture",
    }
    automatic = build_progressive_geometry_residual(**common)
    row_runs = build_progressive_geometry_residual(**common, temporal_mode="row_runs")
    block_four = build_progressive_geometry_residual(
        **common,
        temporal_mode="block_context",
        temporal_block_size=4,
    )
    assert len(automatic) == min(len(row_runs), len(block_four))
    assert automatic == row_runs


def test_staged_application_requires_the_complete_packet() -> None:
    predictor, _ = _labels()
    packet = _build()
    common = {
        "predictor_program": PROGRAM,
        "predictor_contract_id": CONTRACT,
        "predictor_renderer_sha256": RENDERER_SHA,
        "predictor_labels": predictor,
        "source_pair_ids": PAIR_IDS,
        "max_strata": 1,
    }
    with pytest.raises(ProgressiveGeometryResidualError, match="length mismatch"):
        apply_progressive_geometry_residual(packet[:-1], **common)
    corrupted_later_stratum = bytearray(packet)
    corrupted_later_stratum[-5] ^= 1
    with pytest.raises(ProgressiveGeometryResidualError, match="CRC"):
        apply_progressive_geometry_residual(bytes(corrupted_later_stratum), **common)
