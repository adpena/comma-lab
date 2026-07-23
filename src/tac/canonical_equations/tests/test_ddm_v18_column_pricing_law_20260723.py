# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_v18_column_pricing_law_20260723 import (
    ddm_column_reduced_cost,
)


def test_reduced_cost_zero_at_byte_shadow_price() -> None:
    assert ddm_column_reduced_cost(
        singleton_objective_delta=-2.0,
        exact_coder_bytes=10,
        byte_dual_marginal=-0.2,
        conflict_dual_marginals={},
        active_conflict_keys=(),
    ) == pytest.approx(0.0)


def test_positive_constraint_dual_is_refused() -> None:
    with pytest.raises(ValueError, match="nonpositive"):
        ddm_column_reduced_cost(
            singleton_objective_delta=-1.0,
            exact_coder_bytes=1,
            byte_dual_marginal=0.1,
            conflict_dual_marginals={},
            active_conflict_keys=(),
        )


def test_duplicate_conflict_key_is_refused() -> None:
    with pytest.raises(ValueError, match="unique"):
        ddm_column_reduced_cost(
            singleton_objective_delta=-1.0,
            exact_coder_bytes=1,
            byte_dual_marginal=-0.1,
            conflict_dual_marginals={"pixel": -0.2},
            active_conflict_keys=("pixel", "pixel"),
        )
