# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.decoder_causal_condition_transport_20260901 import (
    EQUATION_ID,
    PARENT_EQUATION_ID,
    build_decoder_causal_condition_transport_v1,
    receiver_causal_context_is_free,
    transport_floor_bytes,
)


def test_free_conditioning_requires_exact_causal_zero_byte_context() -> None:
    assert receiver_causal_context_is_free(
        exact_equivalence_class_reproducible=True,
        available_before_consumption=True,
        side_message_bytes=0,
    )
    assert not receiver_causal_context_is_free(
        exact_equivalence_class_reproducible=True,
        available_before_consumption=False,
        side_message_bytes=0,
    )
    assert not receiver_causal_context_is_free(
        exact_equivalence_class_reproducible=True,
        available_before_consumption=True,
        side_message_bytes=1,
    )


def test_transport_floor_is_ceil_bits_over_eight() -> None:
    assert transport_floor_bytes(0.0) == 0
    assert transport_floor_bytes(8.0) == 1
    assert transport_floor_bytes(8.01) == 2
    with pytest.raises(ValueError):
        transport_floor_bytes(-1.0)


def test_equation_is_wyner_ziv_operational_extension_with_two_anchors() -> None:
    equation = build_decoder_causal_condition_transport_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["parent_equation_id"] == PARENT_EQUATION_ID
    assert equation.domain_of_validity["extension_kind"] == (
        "operational_domain_extension_not_new_gate"
    )
    assert len(equation.empirical_anchors) == 2
    assert {anchor.anchor_id for anchor in equation.empirical_anchors} == {
        "qx3_encoder_only_c1_context_requires_exact_bridge_20260901",
        "gmf1_sfp1_three_encoder_only_schedule_contexts_20260901",
    }
    assert equation.domain_of_validity["score_claim"] is False
