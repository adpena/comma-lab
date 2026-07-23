# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_v17_validity_radius_law_20260723 import (
    build_ddm_v17_realized_validity_ratio_uint8_v1,
    realized_validity_ratio,
)


def test_realized_validity_ratio_replays_signed_ratio() -> None:
    assert realized_validity_ratio(2.0, -1.0) == -0.5
    with pytest.raises(ValueError, match="predicted_reduction > 0"):
        realized_validity_ratio(0.0, 1.0)


def test_ddm_v17_equation_builds_from_sha_bound_receipt() -> None:
    equation = build_ddm_v17_realized_validity_ratio_uint8_v1()
    assert equation.equation_id == "ddm_v17_realized_validity_ratio_uint8_v1"
    assert equation.domain_of_validity["basis_conditioned"] is True
    assert equation.domain_of_validity["score_claim"] is False
    assert len(equation.empirical_anchors) == 1
