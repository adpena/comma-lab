# SPDX-License-Identifier: MIT
from tac.canonical_equations.ddm_v16_coupled_margin_law_20260723 import describe


def test_law_exposes_authority_boundary() -> None:
    row = describe()
    assert row["authority"]["conditional_affine_QP_and_closed_KKT"] == "EXACT_WITHIN_FIXED_LOCAL_MODEL"
    assert row["authority"]["global_nonlinear_optimum"] == "NOT_CLAIMED"
    assert row["score_claim"] is False
