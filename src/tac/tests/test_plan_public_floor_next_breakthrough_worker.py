from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_public_floor_next_breakthrough_worker.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("public_floor_next_breakthrough_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_strict_break_even_math_matches_c091_frontier() -> None:
    planner = _load_planner()

    assert planner.strict_bytes_needed_for_score_gap(score=planner.FRONTIER_SCORE) == 1751

    near_miss = planner.break_even(candidate_bytes=274_840)
    assert near_miss["delta_bytes_vs_frontier"] == -1641
    assert near_miss["strict_sub314_at_unchanged_components"] is False
    assert near_miss["sub314_equivalent_bytes_needed_after_candidate"] == 110
    assert near_miss["score_if_components_unchanged"] == pytest.approx(
        0.3140730757407863
    )

    byte_sufficient = planner.break_even(candidate_bytes=274_166)
    assert byte_sufficient["strict_sub314_at_unchanged_components"] is True
    assert byte_sufficient["sub314_equivalent_bytes_needed_after_candidate"] == 0


def test_opportunity_requires_break_even_and_safety_for_exact_recommendation() -> None:
    planner = _load_planner()

    unsafe = planner.opportunity(
        opportunity_id="unsafe_byte_saver",
        rank_group=1,
        title="unsafe",
        status="ready_for_exact_eval_after_claim",
        archive_bytes=planner.FRONTIER_BYTES - 2_000,
        safety={"safe_for_exact_eval_dispatch": False},
    )
    assert unsafe["economics_vs_c091"]["strict_sub314_at_unchanged_components"] is True
    assert unsafe["exact_eval_recommended"] is False

    safe = planner.opportunity(
        opportunity_id="safe_byte_saver",
        rank_group=1,
        title="safe",
        status="ready_for_exact_eval_after_claim",
        archive_bytes=planner.FRONTIER_BYTES - 2_000,
        safety={"safe_for_exact_eval_dispatch": True},
    )
    assert safe["exact_eval_recommended"] is True

    queued = planner.opportunity(
        opportunity_id="queued",
        rank_group=1,
        title="queued",
        status="ready_for_exact_eval_after_claim",
        archive_bytes=planner.FRONTIER_BYTES - 2_000,
        safety={"safe_for_exact_eval_dispatch": True},
        parent_queued=True,
    )
    assert queued["exact_eval_recommended"] is False


def test_recommendation_and_markdown_report_no_dispatch_when_rows_are_blocked() -> None:
    planner = _load_planner()
    blocked = planner.opportunity(
        opportunity_id="qzs3_b0064",
        rank_group=10,
        title="near miss",
        status="blocked_pose_safety",
        archive_bytes=274_840,
        archive_sha256="abc",
        fail_reasons=["renderer output parity failed local pose-safety"],
        safety={
            "safe_for_exact_eval_dispatch": False,
            "mean_abs_delta": 7.24,
            "rms_delta": 11.98,
            "max_abs_delta": 198.9,
        },
    )
    blocked["rank"] = 1
    plan = {
        "frontier": {
            "eval_json": {
                "strict_unchanged_component_bytes_needed": 1751,
            }
        },
        "recommendation": planner.recommendation_from([blocked]),
        "excluded_parent_queued": [],
        "ranked_opportunities": [blocked],
    }

    assert plan["recommendation"]["recommendation"] == "do_not_dispatch"
    md = planner.render_markdown(plan)
    assert "do_not_dispatch" in md
    assert "qzs3_b0064" in md
    assert "renderer output parity failed" in md
