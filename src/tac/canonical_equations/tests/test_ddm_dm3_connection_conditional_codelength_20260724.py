from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_dm3_connection_conditional_codelength_20260724 import (
    EQUATION_ID,
    HISTORY_FAMILIES,
    account_dm3_connection_rows,
    build_ddm_dm3_heldout_connection_conditional_codelength_v1,
)
from tac.canonical_equations.evaluators import resolve_equation_value


def _rows() -> list[dict[str, object]]:
    rows = []
    for index in range(36):
        static = 200 + index
        program = 16
        residual = 100 + index
        rows.append(
            {
                "bucket_id": f"bucket-{index}",
                "bucket_type": (
                    "static_in_image" if index % 2 == 0 else "transient"
                ),
                "stratum": "boundary" if index % 3 == 0 else "cell",
                "support_size": index + 1,
                "winning_history_family": HISTORY_FAMILIES[index % 3],
                "B_static": static,
                "B_history_program": program,
                "B_residual": residual,
                "delta_B_connection": static - program - residual,
            }
        )
    return rows


def test_accounting_rederives_exact_bytes_and_family_counts() -> None:
    rows = _rows()
    result = account_dm3_connection_rows(rows)
    assert result["B_static"] == sum(row["B_static"] for row in rows)
    assert result["B_history_total"] == sum(
        row["B_history_program"] + row["B_residual"] for row in rows
    )
    assert result["delta_B_connection"] == sum(
        row["delta_B_connection"] for row in rows
    )
    assert result["winning_family_counts"] == dict.fromkeys(HISTORY_FAMILIES, 12)
    assert result["score_slack_arithmetic_permitted"] is False
    assert result["receiver_archive_bytes_inferred"] is False
    assert resolve_equation_value(EQUATION_ID, {"rows": rows}) == result


def test_accounting_fails_closed_on_scope_family_or_delta_drift() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="registered 36"):
        account_dm3_connection_rows(rows[:-1])
    rows[0]["winning_history_family"] = "invented"
    with pytest.raises(ValueError, match="introduced a history family"):
        account_dm3_connection_rows(rows)
    rows = _rows()
    rows[0]["delta_B_connection"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        account_dm3_connection_rows(rows)


def test_equation_carries_measured_anchor_and_false_authority_scope() -> None:
    equation = build_ddm_dm3_heldout_connection_conditional_codelength_v1()
    domain = equation.domain_of_validity
    assert domain["aggregate_anchor"]["delta_B_connection"] == 1188
    assert domain["aggregate_anchor"]["identity_selected"] == 34
    assert domain["score_claim"] is False
    assert "not an all-fold estimate" in domain["verdict_scope"]
