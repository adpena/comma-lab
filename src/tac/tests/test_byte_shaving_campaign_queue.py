# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZATION_SCHEMA,
    compile_dqs1_byte_shaving_campaign,
)
from comma_lab.scheduler.experiment_queue import load_queue_definition
from tac.optimization.byte_shaving_campaign import (
    SIGNAL_SURFACE_SCHEMA,
    build_byte_shaving_campaign_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_byte_shaving_campaign_queue.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _pair_drop_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "dqs1_byte_shave_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_dqs1_byte_shave_fixture",
        "dqs1_base_pair_indices": [101, 320, 371, 501],
        "combo_beam_width": 16,
        "max_combo_count": 16,
        "units": [
            {
                "unit_id": "pair0371",
                "unit_kind": "pair",
                "candidate_saved_bytes": 1000,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.95,
                "operations": [
                    {
                        "operation_id": "drop_pair0371",
                        "operation_family": "drop_pair",
                    }
                ],
            },
            {
                "unit_id": "pair0320",
                "unit_kind": "pair",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "drop_pair0320",
                        "operation_family": "drop_pair",
                    }
                ],
            },
            {
                "unit_id": "byte_null_run_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 500,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operation_families": ["null_remove_or_seed"],
            },
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=3)


def test_compile_dqs1_byte_shaving_plan_emits_action_summary_and_blocks_unknown_ops(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    assert compiled["schema"] == MATERIALIZATION_SCHEMA
    assert compiled["score_claim"] is False
    assert compiled["executable_row_count"] >= 1
    assert compiled["blocked_row_count"] >= 1
    assert any(
        "unsupported_operation_family:null_remove_or_seed"
        in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    both = next(
        row
        for row in compiled["executable_rows"]
        if row["dropped_pair_indices"] == [320, 371]
    )
    assert both["selected_pair_indices"] == [101, 501]
    assert {unit["unit_id"] for unit in both["source_units"]} == {"pair0320", "pair0371"}
    action = next(
        row
        for row in compiled["action_summary"]["top_operator_actions"]
        if row["candidate_id"] == both["candidate_id"]
    )
    assert action["operator_next_action"] == "materialize_pairset_archive_and_run_local_controls"
    portfolio_row = next(
        row
        for row in compiled["portfolio"]["operator_action_rows"]
        if row["candidate_id"] == both["candidate_id"]
    )
    assert portfolio_row["source_metadata"]["selected_pair_indices"] == [101, 501]
    assert {unit["unit_id"] for unit in portfolio_row["source_metadata"]["source_units"]} == {
        "pair0320",
        "pair0371",
    }
    assert portfolio_row["ready_for_exact_eval_dispatch"] is False


def test_compile_dqs1_byte_shaving_plan_blocks_drop_pair_on_non_pair_unit(
    tmp_path: Path,
) -> None:
    plan = _pair_drop_plan()
    for unit in plan["ranked_units"]:
        if unit["unit_id"] == "pair0371":
            unit["unit_kind"] = "byte_range"

    compiled = compile_dqs1_byte_shaving_campaign(
        plan,
        repo_root=tmp_path,
        base_pair_indices=[101, 320, 371, 501],
        candidate_limit=8,
        portfolio_json="portfolio.json",
    )

    assert any(
        "unsupported_unit_kind:pair0371:byte_range" in row["materialization_blockers"]
        for row in compiled["blocked_rows"]
    )
    assert all(
        "pair0371" not in {unit["unit_id"] for unit in row["source_units"]}
        for row in compiled["executable_rows"]
    )


def test_byte_shaving_campaign_queue_cli_writes_dqs1_queue(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    materialization = tmp_path / "materialization.json"
    portfolio = tmp_path / "portfolio.json"
    summary = tmp_path / "action_summary.json"
    queue = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_pair_drop_plan()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan_path),
            "--materialization-out",
            str(materialization),
            "--portfolio-out",
            str(portfolio),
            "--action-summary-out",
            str(summary),
            "--queue-out",
            str(queue),
            "--repo-root",
            str(tmp_path),
            "--base-pair-indices",
            "101,320,371,501",
            "--candidate-limit",
            "4",
            "--queue-candidate-limit",
            "2",
            "--results-root",
            "results",
            "--local-cpu-concurrency",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    assert stdout["executable_row_count"] >= 1
    assert stdout["queue"]["experiment_count"] == 2
    assert materialization.is_file()
    assert portfolio.is_file()
    assert summary.is_file()
    loaded = load_queue_definition(queue)
    assert loaded["controls"]["max_concurrency"]["local_cpu"] == 2
    assert len(loaded["experiments"]) == 2
    assert all(
        experiment["id"].startswith("pairset_byte_shave_")
        for experiment in loaded["experiments"]
    )
