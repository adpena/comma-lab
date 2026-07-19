# SPDX-License-Identifier: MIT
"""Focused tests for the packet-only PDW2 spatial receiver."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.pdw2_spatial_receiver import (
    EXPECTED_GEOMETRY,
    PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
    PDW2SpatialReceiverError,
    build_pdw2_coefficient_only_nonidentifiability_witness,
    detect_pdw2_packet_mutation_canary,
    mutate_pdw2_packet_first_relative_coefficient,
    run_pdw2_spatial_receiver,
)
from tac.boundary_math.power_diagram_witness import (
    encode_pdw2,
    make_gauge_fixed_affine_target,
)


def _packet() -> bytes:
    target = make_gauge_fixed_affine_target(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    return encode_pdw2(target)


def _field_zero(pair_count: int = 2) -> np.ndarray:
    return np.zeros((pair_count, *EXPECTED_GEOMETRY), dtype=np.float32)


def test_receiver_decodes_packet_and_field_contracts_and_responds_with_hashes() -> None:
    packet = _packet()
    field = _field_zero(2)
    field[:, 0, 0, 0] = 0.35
    receipt = run_pdw2_spatial_receiver(packet, field, include_labels=True)
    assert receipt["schema"] == "pdw2_spatial_receiver_receipt.v1"
    assert receipt["packet_to_partition_consumed"] is True
    assert receipt["coefficient_only_through_r_equivalent"] is False
    assert receipt["through_r_authority"] is False
    assert receipt["d_seg"] is None
    assert receipt["d_pose"] is None
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["pdw2_promotion_blocker"] == PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY
    assert receipt["class_ids"] == [0, 1]
    assert receipt["partition_shape"] == list(field.shape[:-1])
    assert receipt["mapped_page_eviction_applied"] is False
    assert receipt["partition_labels"] is not None

    bad_shape = np.zeros((1, 10, 10, 4), dtype=np.float32)
    with pytest.raises(PDW2SpatialReceiverError, match="geometry"):
        run_pdw2_spatial_receiver(packet, bad_shape)

    bad_dtype = _field_zero(2).astype(np.float64)
    with pytest.raises(PDW2SpatialReceiverError, match="float32"):
        run_pdw2_spatial_receiver(packet, bad_dtype)


@pytest.mark.parametrize("empty", [b"", bytearray()])
def test_packet_missing_or_deleted_rejected(empty: bytes | bytearray) -> None:
    with pytest.raises(PDW2SpatialReceiverError, match="PDW2 packet"):
        run_pdw2_spatial_receiver(empty, _field_zero(1))


def test_packet_only_nonidentifiability_witness_and_canaries() -> None:
    packet = _packet()
    witness = build_pdw2_coefficient_only_nonidentifiability_witness(packet)
    assert witness["schema"] == "pdw2_coefficient_only_nonidentifiability_witness.v1"
    assert witness["packet_to_partition_consumed"] is True
    assert witness["coefficient_only_through_r_equivalent"] is False
    assert witness["through_r_authority"] is False
    assert witness["pdw2_promotion_blocker"] == PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY
    assert witness["witness_feature_vectors"]["class_a"] != witness["witness_feature_vectors"]["class_b"]
    assert witness["d_seg"] is None
    assert witness["d_pose"] is None

    # Mutation with no observed effect is not a canary.
    no_effect = detect_pdw2_packet_mutation_canary(packet, packet, _field_zero(1))
    assert no_effect["mutation_observed"] is False
    assert no_effect["mismatch_pixels"] == 0

    field = _field_zero(1)
    field[..., 0] = 0.1
    canary = detect_pdw2_packet_mutation_canary(
        packet,
        mutate_pdw2_packet_first_relative_coefficient(packet, 2.0),
        field,
    )
    assert canary["packet_to_partition_consumed"] is True
    assert canary["coefficient_only_through_r_equivalent"] is False
    assert canary["through_r_authority"] is False
    assert canary["pdw2_promotion_blocker"] == PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY
    assert canary["d_seg"] is None
    assert canary["d_pose"] is None

    # If a larger coefficient shift does not alter labels, synthesize a stronger
    # perturbation and require at least one canary-observed mismatch.
    if not canary["mutation_observed"]:
        canary = detect_pdw2_packet_mutation_canary(
            packet,
            mutate_pdw2_packet_first_relative_coefficient(packet, 200.0),
            field,
        )
    assert canary["mutation_observed"] is True
    assert canary["mismatch_pixels"] >= 1


def test_nonfinite_field_refuses_at_the_exact_streamed_pair() -> None:
    field = _field_zero(2)
    field[1, 4, 5, 3] = np.nan
    with pytest.raises(PDW2SpatialReceiverError, match="pair 1"):
        run_pdw2_spatial_receiver(_packet(), field)
