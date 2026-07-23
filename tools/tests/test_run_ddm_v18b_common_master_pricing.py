# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

from tac.optimization.ddm_column_generation import PricedColumn
from tac.optimization.direct_description_carrier_compose import (
    RowBandScorerTemplateV1,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.run_ddm_v18b_common_master_pricing import (
    FIXED_BUDGETS,
    ColumnSpec,
    _formulation_falsified,
    _largest_prefix_not_exceeding,
    _load_bundle_rows,
    _miqp_diagonal_proposal,
    _permutation_gauge_coder_row,
    _receiver_output_noop_rejection,
    _source_resume_identity,
)


def test_template_column_audit_row_is_canonical_ijson() -> None:
    template = RowBandScorerTemplateV1(
        "Lane",
        "inner_boundary",
        0,
        128,
        1,
        1,
        b"\x33\xff\xcc",
    )
    row = ColumnSpec(
        "template",
        "realized_residual_vjp",
        1.0,
        template_index=0,
        template=template,
    ).row()
    assert row["template"]["rgb_u8_hex"] == "33ffcc"
    assert rfc8785_canonicalize(row)


def test_bound_v12_inventory_is_exactly_4096_atoms_in_353_bundles() -> None:
    rows = _load_bundle_rows()
    assert len(rows) == 353
    assert sum(int(row["atomic_obligation_count"]) for row in rows) == 4096


def test_conflict_miqp_respects_exact_byte_cap_and_conflicts() -> None:
    columns = (
        PricedColumn("a", "fixture", 4, -4.0, ("same-site",)),
        PricedColumn("b", "fixture", 4, -3.0, ("same-site",)),
        PricedColumn("c", "fixture", 3, -2.0),
    )
    assert _miqp_diagonal_proposal(columns, added_byte_budget=7) == ("a", "c")


def test_equal_byte_prefix_uses_realized_control_cap_not_nominal_rung() -> None:
    assert (
        _largest_prefix_not_exceeding(
            (100, 110, 130, 119),
            base_archive_bytes=100,
            realized_added_byte_cap=20,
        )
        == 3
    )


def test_falsifier_requires_three_clean_rounds_and_no_equal_byte_win() -> None:
    history = [
        {
            "round": index,
            "complete": True,
            "exact_pricing": True,
            "negative_reduced_cost_count": 0,
        }
        for index in range(1, 4)
    ]
    equal = [{"added_byte_budget": budget, "beats_v12": False} for budget in FIXED_BUDGETS]
    assert _formulation_falsified(history, equal) == (True, True, False)

    equal[-1]["beats_v12"] = True
    assert _formulation_falsified(history, equal) == (False, True, True)

    equal[-1]["beats_v12"] = False
    history[-1]["negative_reduced_cost_count"] = 1
    assert _formulation_falsified(history, equal) == (False, False, False)


def test_source_resume_identity_excludes_only_live_free_space() -> None:
    first = {
        "schema": "checkpoint",
        "common_master": {"sha256": "a" * 64},
        "storage_preflight": {
            "free_bytes": 10_000,
            "required_free_bytes": 100,
            "pass": True,
        },
    }
    resumed = {
        **first,
        "storage_preflight": {
            **first["storage_preflight"],
            "free_bytes": 9_000,
        },
    }
    drifted = {
        **resumed,
        "common_master": {"sha256": "b" * 64},
    }

    assert _source_resume_identity(first) == _source_resume_identity(resumed)
    assert _source_resume_identity(first) != _source_resume_identity(drifted)


def test_receiver_output_noop_is_resumable_but_other_compile_errors_fail_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tools.run_ddm_v18b_common_master_pricing._git_sha",
        lambda: "a" * 40,
    )
    spec = ColumnSpec(
        "template",
        "realized_residual_vjp",
        1.0,
        template_index=0,
        template=RowBandScorerTemplateV1(
            "Lane",
            "inner_boundary",
            0,
            128,
            1,
            1,
            b"\x33\xff\xcc",
        ),
    )
    row = _receiver_output_noop_rejection(
        error=DirectDescriptionError("fixture is a receiver-output no-op"),
        config_hash="b" * 64,
        candidate_index=7,
        spec=spec,
        current_archive=b"archive",
        accepted_count=3,
    )
    assert row["reason"] == "exact_receiver_parseback_refused_output_noop"
    assert row["accepted_count_after"] == 3
    assert row["exact_archive_before"]["sha256"]
    assert row["candidate_inventory_row_sha256"] == (
        hashlib.sha256(rfc8785_canonicalize(spec.row())).hexdigest()
    )

    try:
        _receiver_output_noop_rejection(
            error=DirectDescriptionError("unrelated parse-back failure"),
            config_hash="b" * 64,
            candidate_index=7,
            spec=spec,
            current_archive=b"archive",
            accepted_count=3,
        )
    except DirectDescriptionError as error:
        assert str(error) == "unrelated parse-back failure"
    else:
        raise AssertionError("unrelated compile errors must fail closed")


def test_permutation_gauge_coder_row_reports_exact_empty_payload_identity() -> None:
    row = _permutation_gauge_coder_row(b"archive", ())
    assert row["status"] == "MEASURED_TRIVIAL_EMPTY_SELECTED_PAYLOAD"
    assert row["as_is_archive_bytes"] == row["canonical_archive_bytes"] == 7
    assert row["canonical_minus_as_is_bytes"] == 0
    assert row["candidate_pool_is_counted_payload"] is False
