# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_dm2_semantic_realization_exchange import (
    EQUATION_ID,
    build_ddm_dm2_semantic_to_realized_rgb_exchange_v1,
    semantic_realization_exchange,
)


def test_exchange_prices_only_positive_collateral() -> None:
    beneficial = semantic_realization_exchange(
        semantic_bytes=10,
        realized_rgb_bytes=100,
        collateral_score_delta=-2.0,
        source_video_bytes=1_000,
        rate_weight=25.0,
    )
    harmful = semantic_realization_exchange(
        semantic_bytes=10,
        realized_rgb_bytes=100,
        collateral_score_delta=0.5,
        source_video_bytes=1_000,
        rate_weight=25.0,
    )
    assert beneficial["positive_collateral_byte_equivalent_at_rate_dual"] == 0.0
    assert beneficial["effective_bytes_per_semantic_byte"] == 10.0
    assert harmful["positive_collateral_byte_equivalent_at_rate_dual"] == 20.0
    assert harmful["effective_bytes_per_semantic_byte"] == 12.0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"semantic_bytes": 0, "realized_rgb_bytes": 1, "collateral_score_delta": 0.0},
        {"semantic_bytes": 1, "realized_rgb_bytes": -1, "collateral_score_delta": 0.0},
        {"semantic_bytes": True, "realized_rgb_bytes": 1, "collateral_score_delta": 0.0},
        {"semantic_bytes": 1, "realized_rgb_bytes": 1, "collateral_score_delta": float("nan")},
    ),
)
def test_exchange_refuses_malformed_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        semantic_realization_exchange(**kwargs)  # type: ignore[arg-type]


def test_bound_receipt_builds_exact_nonpromotable_anchor() -> None:
    equation = build_ddm_dm2_semantic_to_realized_rgb_exchange_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["bound_status"].startswith("constructive")
    anchor = equation.empirical_anchors[0]
    assert anchor.residual == pytest.approx(0.0)
    assert anchor.empirical_output["effective_bytes_per_semantic_byte"] == pytest.approx(
        2524.2504780114723
    )
    assert anchor.empirical_output["fallback_pair_ids"] == [55, 60, 90]
