# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.analysis.nerv_modelsize_ladder import (
    NERV_MODELSIZE_LADDER_SCHEMA,
    build_nerv_modelsize_ladder,
    render_nerv_modelsize_ladder_markdown,
)


def test_nerv_modelsize_ladder_prices_hi_nerv_and_snerv_steps() -> None:
    report = build_nerv_modelsize_ladder()

    assert report["schema"] == NERV_MODELSIZE_LADDER_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["contest_byte_price_score_per_byte"] > 0.0
    assert report["objective_authority"]["objective"] == "contest_auth_eval_scorer_only"
    assert "human_visual_fidelity" in report["objective_authority"][
        "forbidden_selection_terms"
    ]
    assert "PSNR" in report["objective_authority"]["forbidden_selection_terms"]

    families = {row["family"]: row for row in report["family_rows"]}
    assert set(families) == {"hi_nerv", "snerv"}

    hi = families["hi_nerv"]
    hi_rows = hi["ladder_rows"]
    assert [row["row_id"] for row in hi_rows] == [
        "hi_nerv_local_tiny",
        "hi_nerv_local_small",
        "hi_nerv_local_base",
        "hi_nerv_local_wide",
    ]
    assert hi_rows[0]["total_parameter_count"] < hi_rows[-1]["total_parameter_count"]
    for row in hi_rows:
        estimates = row["quantized_payload_estimates"]
        assert estimates["fp32"]["payload_bytes"] > estimates["fp16"]["payload_bytes"]
        assert estimates["fp16"]["payload_bytes"] > estimates["int8"]["payload_bytes"]
        assert estimates["int8"]["payload_bytes"] > estimates["int4"]["payload_bytes"]
        assert estimates["int4"]["payload_bytes"] > estimates["int2"]["payload_bytes"]
    assert hi["marginal_gates"]
    assert all(
        gate["required_nonrate_score_improvement"] > 0.0
        for gate in hi["marginal_gates"]
    )

    snerv = families["snerv"]
    snerv_rows = snerv["ladder_rows"]
    assert len(snerv_rows) >= 4
    assert all("configured" in row["quantized_payload_estimates"] for row in snerv_rows)
    assert all(row["lf_shape"][0] <= 192 for row in snerv_rows)
    configured_bytes = [
        row["quantized_payload_estimates"]["configured"]["payload_bytes"]
        for row in snerv_rows
    ]
    assert configured_bytes == sorted(configured_bytes)
    assert {
        "snerv_measured_nonrate_modelsize_ladder_missing",
        "snerv_byte_closed_modelsize_ladder_missing",
    }.issubset(set(snerv["blockers"]))


def test_nerv_modelsize_ladder_focus_and_markdown() -> None:
    report = build_nerv_modelsize_ladder(focus_families=("snerv",), num_pairs=16)

    assert report["focus_families"] == ["snerv"]
    assert [row["family"] for row in report["family_rows"]] == ["snerv"]
    assert report["frames"] == 32
    assert report["score_claim"] is False

    markdown = render_nerv_modelsize_ladder_markdown(report)
    assert "## snerv" in markdown
    assert "Marginal gates" in markdown
    assert "modelsize_ladder_false_authority_no_nonrate_score" in markdown
