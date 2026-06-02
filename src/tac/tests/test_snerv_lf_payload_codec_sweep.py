# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.snerv_lf_payload_codec_sweep import (
    SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
    build_snerv_lf_payload_codec_sweep,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    CUT,
    DEMOTE,
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
    PROTECT,
)


def test_snerv_lf_payload_codec_sweep_is_rate_only_and_scorer_only() -> None:
    plane = np.zeros((32, 32), dtype=np.int64)
    plane[0, 0] = -1
    plane[10, 4] = 1

    report = build_snerv_lf_payload_codec_sweep(
        [plane],
        modes=("int64_lzma", "portfolio_auto", "int2"),
    )

    assert report["schema"] == SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["objective_authority"]["objective"] == "contest_auth_eval_scorer_only"
    assert "human_visual_fidelity" in report["objective_authority"][
        "forbidden_selection_terms"
    ]
    assert report["codec_proof"] == (
        "snerv_lf_quant_payload.v2_receiver_visible_exact_intn_codec"
    )
    assert report["selected_rate_only_row"]["payload_bytes"] > 0
    assert report["selected_rate_only_row"]["payload_bytes"] < report["raw_i64_bytes"]
    assert "snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay" in report[
        "blockers"
    ]
    assert report["baseline_mode"] == "int64_lzma"
    assert report["section_value_rows"]
    plan = report["byte_price_plan"]
    assert plan["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert plan["source_schema"] == SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert plan["input_row_count"] == len(report["section_value_rows"])
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert "full_video_coverage_missing" in plan["blockers"]

    by_mode = {
        row["source"]["candidate_mode"]: row for row in plan["decision_rows"]
    }
    for row in by_mode.values():
        assert row["decision"] == DEMOTE
        if row["delta_total_score"] is None:
            continue
        assert row["economic_decision"] in {CUT, PROTECT}
        if row["byte_delta"] < 0:
            assert row["economic_decision"] == CUT
        else:
            assert row["economic_decision"] == PROTECT


def test_snerv_lf_payload_codec_sweep_marks_failed_modes() -> None:
    plane = np.array([[3]], dtype=np.int64)

    report = build_snerv_lf_payload_codec_sweep([plane], modes=("int2",))
    row = report["rows"][0]

    assert row["payload_bytes"] == 0
    assert row["error"]
    assert "snerv_lf_payload_codec_mode_failed" in row["blockers"]
    plan_row = report["byte_price_plan"]["decision_rows"][0]
    assert plan_row["delta_total_score"] is None
    assert "snerv_lf_payload_codec_mode_failed" in plan_row["blockers"]


def test_snerv_lf_payload_codec_sweep_never_selects_failed_zero_byte_mode() -> None:
    plane = np.array([[3, 0], [0, 0]], dtype=np.int64)

    report = build_snerv_lf_payload_codec_sweep(
        [plane],
        modes=("int2", "int64_lzma"),
    )

    assert report["selected_rate_only_row"]["mode"] == "int64_lzma"
    assert report["selected_rate_only_row"]["payload_bytes"] > 0
    assert report["selected_rate_only_row"]["error"] is None
    assert report["failed_modes"] == [
        {
            "mode": "int2",
            "error": "signed_int2_bitpack requires values in [-2, 1]",
        }
    ]

    by_mode = {
        row["source"]["candidate_mode"]: row
        for row in report["byte_price_plan"]["decision_rows"]
    }
    assert "snerv_lf_payload_codec_mode_failed" in by_mode["int2"]["blockers"]
    assert "snerv_lf_payload_codec_mode_failed" not in by_mode["int64_lzma"][
        "blockers"
    ]
