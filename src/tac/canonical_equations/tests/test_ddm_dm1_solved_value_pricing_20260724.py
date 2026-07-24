from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_dm1_solved_value_pricing_20260724 import (
    CODECS,
    account_dm1_semantic_prices,
    build_ddm_dm1_semantic_record_price_and_rehome_v1,
)


def test_accounting_selects_per_row_and_joint_coders() -> None:
    rows = [
        {"zlib9": 100 + index, "lzma9": 110, "context_arithmetic": 120 - index}
        for index in range(25)
    ]
    joint = {"zlib9": 1600, "lzma9": 1569, "context_arithmetic": 2387}
    result = account_dm1_semantic_prices(rows, joint_prices=joint)
    assert result["independent_best_per_row_bytes"] == sum(
        min(row.values()) for row in rows
    )
    assert result["joint_best_codec"] == "lzma9"
    assert result["joint_best_bytes"] == 1569
    assert result["score_slack_arithmetic_permitted"] is False


def test_accounting_fails_closed_on_row_or_coder_drift() -> None:
    valid = [dict.fromkeys(CODECS, 100) for _ in range(25)]
    joint = dict.fromkeys(CODECS, 200)
    with pytest.raises(ValueError, match="exactly the registered 25"):
        account_dm1_semantic_prices(valid[:-1], joint_prices=joint)
    invalid = [dict(row) for row in valid]
    invalid[0]["invented"] = 1
    with pytest.raises(ValueError, match="sealed three coders"):
        account_dm1_semantic_prices(invalid, joint_prices=joint)


def test_equation_keeps_receiver_and_score_boundaries_explicit() -> None:
    equation = build_ddm_dm1_semantic_record_price_and_rehome_v1()
    domain = equation.domain_of_validity
    assert domain["row_count"] == 25
    assert domain["boundary_home"].startswith("SKELETON/L4")
    assert domain["cell_home"].startswith("FIBER/L4")
    assert domain["connection_home"].startswith("NULL")
    assert domain["score_claim"] is False
