from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_pf2_bucket_assignment_20260724 import (
    build_ddm_pf2_bucket_assignment_join_eligibility_v1,
    pf2_bucket_assignment_join_eligible,
)


def test_assignment_join_requires_all_three_legs() -> None:
    assert pf2_bucket_assignment_join_eligible(
        pair_membership_set_equal=True,
        receiver_actuator_ids=("j2.island.track1.center_x",),
        direction_ids=("POSITIVE_ONE_QUANTUM",),
    )
    assert not pf2_bucket_assignment_join_eligible(
        pair_membership_set_equal=True,
        receiver_actuator_ids=(),
        direction_ids=(),
    )


def test_assignment_join_refuses_scalar_id_and_builds_registered_shape() -> None:
    with pytest.raises(ValueError, match="sequences"):
        pf2_bucket_assignment_join_eligible(
            pair_membership_set_equal=True,
            receiver_actuator_ids="not-a-sequence",
            direction_ids=("POSITIVE_ONE_QUANTUM",),
        )
    equation = build_ddm_pf2_bucket_assignment_join_eligibility_v1()
    assert equation.equation_id.endswith("_v1")
    assert equation.domain_of_validity["current_foreign_key_closure"] == "0/1200"
