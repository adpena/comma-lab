# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (
    EQUATION_ID,
    build_ddm_ms2r_tolerance_capped_solve,
    populate_ddm_ms2r_tolerance_capped_solve,
    tolerance_capped_rung_score,
)
from tac.canonical_equations.evaluators import resolve_equation_value
from tac.canonical_equations.registry import query_equations


def _row(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "seg_errors": 100_000,
        "scored_pixels": 117_964_800,
        "d_pose": 0.001,
        "raw_compact_bytes": 150_000,
        "best_coded_bytes": 140_000,
        "allowed_errors": 136_839,
        "bundle_complete": True,
        "parseback_exact": True,
        "uint8_reverified": True,
    }
    value.update(overrides)
    return value


def test_rung_score_uses_best_coded_bytes_and_exact_error_cap() -> None:
    result = tolerance_capped_rung_score(**_row())
    assert result["admissible_inside_error_cap"] is True
    assert result["d_seg"] == 100_000 / 117_964_800
    assert result["coder_gain_bytes"] == 10_000
    assert result["best_coded_bytes"] == 140_000
    outside = tolerance_capped_rung_score(**_row(seg_errors=136_840))
    assert outside["admissible_inside_error_cap"] is False


def test_rung_score_refuses_partial_or_non_parseback_authority() -> None:
    with pytest.raises(ValueError, match="BUNDLE-COMPLETE"):
        tolerance_capped_rung_score(**_row(bundle_complete=False))
    with pytest.raises(ValueError, match="BUNDLE-COMPLETE"):
        tolerance_capped_rung_score(**_row(parseback_exact=False))
    with pytest.raises(ValueError, match="cannot exceed"):
        tolerance_capped_rung_score(
            **_row(raw_compact_bytes=100, best_coded_bytes=101)
        )


def test_equation_callable_and_locked_registry_round_trip(tmp_path: Path) -> None:
    equation = build_ddm_ms2r_tolerance_capped_solve()
    module_name, callable_name = equation.python_callable_module_path.split(":")
    assert getattr(importlib.import_module(module_name), callable_name) is (
        tolerance_capped_rung_score
    )
    resolved = resolve_equation_value(EQUATION_ID, _row())
    assert resolved["admissible_inside_error_cap"] is True

    ledger = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"
    populate_ddm_ms2r_tolerance_capped_solve(
        path=ledger,
        lock_path=lock,
        agent="codex",
        subagent_id="ddm_ms2r_tolerance_capped_solve_20260724T152730Z",
    )
    rows = query_equations(path=ledger)
    assert [row.equation_id for row in rows] == [EQUATION_ID]
    assert rows[0].domain_of_validity["current_status"].startswith(
        "BLOCKED_MS3_BUNDLE_PARTIAL"
    )
