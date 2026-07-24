from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_receiver_support_foreign_key_20260724 import (
    build_ddm_receiver_support_pf2_causal_intersection_v1,
    receiver_support_pf2_causal_intersection,
)


def test_causal_intersection_requires_exact_raw_event_identity() -> None:
    assert receiver_support_pf2_causal_intersection(
        pf2_event_ids=(1, 5, 9),
        changed_argmax_event_ids=(0, 5, 8),
    ) == (5,)
    assert (
        receiver_support_pf2_causal_intersection(
            pf2_event_ids=(1, 5, 9),
            changed_argmax_event_ids=(2, 6, 10),
        )
        == ()
    )


def test_causal_intersection_refuses_ambiguous_event_custody() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        receiver_support_pf2_causal_intersection(
            pf2_event_ids=(5, 1),
            changed_argmax_event_ids=(1,),
        )
    equation = build_ddm_receiver_support_pf2_causal_intersection_v1()
    assert equation.equation_id.endswith("_v1")
    assert equation.domain_of_validity["infeasible_quantum_semantics"] == (
        "EXPLICIT_BLOCKER_NOT_ZERO"
    )
