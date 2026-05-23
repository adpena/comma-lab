# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.comma_lab.scheduler.dqs1_local_first_queue import (
    build_queue_from_action_summary,
    candidate_slug,
    find_latest_cross_family_action_summary,
)
from src.comma_lab.scheduler.experiment_queue import ExperimentQueueError, load_queue_definition
from src.tac.optimization.local_cpu_contest_drift import EUREKA_FALSE_AUTHORITY_FIELDS


def _false_authority() -> dict[str, object]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _eureka_false_authority() -> dict[str, object]:
    return dict.fromkeys(EUREKA_FALSE_AUTHORITY_FIELDS, False)


def _action(candidate_id: str, rank: int) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_action_rank": rank,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
    }


def _row(candidate_id: str, selected_pair_indices: list[int]) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
        "source_metadata": {
            "selected_pair_count": len(selected_pair_indices),
            "selected_pair_indices": selected_pair_indices,
        },
    }


def _write_summary(repo: Path) -> Path:
    summary_dir = (
        repo
        / "experiments"
        / "results"
        / "cross_family_candidate_portfolio"
        / "20260522T202400Z_pairset_component_rank019_append_only_hardened"
    )
    summary_dir.mkdir(parents=True)
    portfolio = repo / "portfolio.json"
    portfolio.write_text(
        json.dumps(
            {
                "operator_action_rows": [
                    _row("pairset_drop_one_rank023_pair0440", [1, 2, 440]),
                    _row("pairset_drop_one_rank024_pair0112", [1, 2, 112]),
                ]
            }
        )
    )
    summary = summary_dir / "action_summary.json"
    summary.write_text(
        json.dumps(
            {
                **_false_authority(),
                "schema": "pairset_component_marginal_canonicalization_summary.v1",
                "portfolio_json": "portfolio.json",
                "top_operator_actions": [
                    _action("pairset_drop_one_rank023_pair0440", 1),
                    _action("pairset_drop_one_rank024_pair0112", 2),
                ],
            }
        )
    )
    return summary


def _write_completed_local_advisory(
    repo: Path,
    *,
    candidate_id: str = "pairset_drop_one_rank023_pair0440",
    score: float = 0.2,
) -> tuple[Path, Path, str]:
    completed = repo / "results" / "materialized" / candidate_slug(candidate_id)
    archive = completed / "submission_dir" / "archive.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    advisory = completed / "local_cpu_advisory.json"
    archive_sha = sha256(archive.read_bytes()).hexdigest()
    advisory.write_text(
        json.dumps(
            {
                **_false_authority(),
                "canonical_score": score,
                "evidence_grade": "[macOS-CPU advisory]",
                "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
                "archive_size_bytes": archive.stat().st_size,
                "n_samples": 600,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "score_axis": "cpu_advisory",
                "provenance": {
                    "archive_path": str(archive),
                    "archive_sha256": archive_sha,
                    "archive_size_bytes": archive.stat().st_size,
                },
            }
        )
    )
    return advisory, archive, archive_sha


def _write_eureka_signal(
    repo: Path,
    *,
    candidate_id: str,
    advisory: Path,
    archive_sha: str,
    local_score: float = 0.2,
    eureka_trigger: bool = False,
    recommended_action: str = "observe_only",
) -> Path:
    signal = (
        repo
        / ".omx"
        / "research"
        / f"local_cpu_contest_drift_eureka_{candidate_id}_20260522T000000Z.json"
    )
    signal.parent.mkdir(parents=True)
    signal.write_text(
        json.dumps(
            {
                **_eureka_false_authority(),
                "schema": "local_cpu_contest_drift_eureka_signal.v1",
                "candidate_id": candidate_id,
                "candidate_archive_sha256": archive_sha,
                "local_axis": "macOS-CPU advisory",
                "target_axis": "contest-CPU",
                "local_score": local_score,
                "auth_frontier_score": 0.192028,
                "eureka_trigger": eureka_trigger,
                "recommended_action": recommended_action,
                "source_artifact": str(advisory),
            }
        )
    )
    return signal


def test_dqs1_queue_builder_skips_completed_local_advisory_candidate(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )

    result = build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")

    assert result.selection.candidate_id == "pairset_drop_one_rank024_pair0112"
    assert result.selection.skipped_candidates == (
        {"candidate_id": "pairset_drop_one_rank023_pair0440", "reason": "local_advisory_exists"},
    )
    experiment = result.queue["experiments"][0]
    assert experiment["lane_id"] == "lane_dqs1_pairset_drop_one_rank024_pair0112_local_first_20260522"
    assert experiment["tags"] == ["dqs1", "pairset", "local-first", "no-score-authority"]
    assert result.queue["controls"]["max_concurrency"]["modal_gpu"] == 0
    assert {step["resources"]["kind"] for step in experiment["steps"]} == {"local_cpu"}
    steps_by_id = {step["id"]: step for step in experiment["steps"]}
    assert list(steps_by_id) == [
        "plan_packet",
        "materialize",
        "locality_controls",
        "local_cpu_advisory",
        "local_cpu_contest_drift_eureka",
    ]
    selected_pairs_arg = experiment["steps"][0]["command"][-1]
    assert selected_pairs_arg == "1,2,112"
    eureka_step = steps_by_id["local_cpu_contest_drift_eureka"]
    assert eureka_step["requires"] == ["local_cpu_advisory"]
    assert "tools/calibrate_local_cpu_contest_drift.py" in eureka_step["command"]
    assert "--auth-frontier-score-from-pointer" in eureka_step["command"]
    assert "--eureka-out" in eureka_step["command"]
    false_authority = next(
        condition
        for condition in eureka_step["postconditions"]
        if condition["type"] == "json_false_authority"
    )
    assert false_authority["required_false"] == list(EUREKA_FALSE_AUTHORITY_FIELDS)
    assert false_authority["false_or_missing"] == []
    assert any(
        condition["type"] == "json_false_authority"
        for step in experiment["steps"]
        for condition in step["postconditions"]
    )


def test_dqs1_queue_builder_refuses_to_skip_positive_eureka_candidate(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
        eureka_trigger=True,
        recommended_action="dispatch_exact_auth_anchor",
    )

    with pytest.raises(ExperimentQueueError, match="requests exact auth dispatch"):
        build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")


def test_dqs1_queue_builder_fails_closed_on_incomplete_eureka_authority(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    signal_path = _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )
    signal = json.loads(signal_path.read_text())
    signal.pop("gpu_launched")
    signal_path.write_text(json.dumps(signal))

    with pytest.raises(ExperimentQueueError, match="gpu_launched"):
        build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")


def test_dqs1_queue_builder_fails_closed_on_truthy_eureka_authority(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    signal_path = _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )
    signal = json.loads(signal_path.read_text())
    signal["score_claim_valid"] = True
    signal_path.write_text(json.dumps(signal))

    with pytest.raises(ExperimentQueueError, match="score_claim_valid"):
        build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")


def test_dqs1_queue_builder_does_not_skip_without_eureka_signal(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    _write_completed_local_advisory(tmp_path)

    result = build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")

    assert result.selection.candidate_id == "pairset_drop_one_rank023_pair0440"


def test_dqs1_queue_builder_does_not_skip_partial_local_advisory(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    completed = tmp_path / "results" / "materialized" / "drop_rank023_pair0440"
    completed.mkdir(parents=True)
    (completed / "local_cpu_advisory.json").write_text(
        json.dumps(
            {
                **_false_authority(),
                "canonical_score": 0.2,
                "score_axis": "cpu_advisory",
            }
        )
    )

    result = build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")

    assert result.selection.candidate_id == "pairset_drop_one_rank023_pair0440"


def test_dqs1_queue_builder_routes_group_pairset_candidates(tmp_path: Path) -> None:
    summary_dir = (
        tmp_path
        / "experiments"
        / "results"
        / "cross_family_candidate_portfolio"
        / "20260522T202400Z_pairset_component_group_candidates"
    )
    summary_dir.mkdir(parents=True)
    candidate_id = "pairset_drop_two_r001_002_p0371_0320"
    selected_pairs = [26, 59, 68, 98, 109, 112]
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(
        json.dumps(
            {
                "operator_action_rows": [
                    _row(candidate_id, selected_pairs),
                ]
            }
        )
    )
    summary = summary_dir / "action_summary.json"
    summary.write_text(
        json.dumps(
            {
                **_false_authority(),
                "schema": "pairset_component_marginal_canonicalization_summary.v1",
                "portfolio_json": "portfolio.json",
                "top_operator_actions": [_action(candidate_id, 7)],
            }
        )
    )

    result = build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")

    assert candidate_slug(candidate_id) == "drop_two_r001_002_p0371_0320"
    assert result.selection.candidate_id == candidate_id
    assert result.selection.operator_action_rank == 7
    experiment = result.queue["experiments"][0]
    assert experiment["priority"] == 7
    assert experiment["lane_id"] == (
        "lane_dqs1_pairset_drop_two_r001_002_p0371_0320_local_first_20260522"
    )
    assert experiment["steps"][0]["command"][-1] == "26,59,68,98,109,112"
    assert any(
        "materialized/drop_two_r001_002_p0371_0320/submission_dir" in part
        for part in experiment["steps"][1]["command"]
    )


def test_dqs1_queue_builder_threads_runtime_overrides(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        bridge_plan="bridge/custom.json",
        base_submission_dir="base/submission_dir",
        global_mutated_archive="mutated/archive.zip",
        upstream_dir="custom_upstream",
        video_names_file="custom_names.txt",
        frame_policy="single_last_frame",
        drift_calibration_json="calibration/custom.json",
        eureka_output_dir=".omx/custom_research",
        eureka_run_id="20260523T010203Z",
    )

    steps = {step["id"]: step["command"] for step in result.queue["experiments"][0]["steps"]}
    assert "bridge/custom.json" in steps["plan_packet"]
    assert "base/submission_dir/archive.zip" in steps["plan_packet"]
    assert "base/submission_dir" in steps["materialize"]
    assert "mutated/archive.zip" in steps["locality_controls"]
    assert "single_last_frame" in steps["locality_controls"]
    assert "custom_upstream" in steps["local_cpu_advisory"]
    assert "custom_names.txt" in steps["local_cpu_advisory"]
    assert "calibration/custom.json" in steps["local_cpu_contest_drift_eureka"]
    assert any(
        part
        == ".omx/custom_research/local_cpu_contest_drift_eureka_pairset_drop_one_rank023_pair0440_20260523T010203Z.json"
        for part in steps["local_cpu_contest_drift_eureka"]
    )


def test_checked_in_dqs1_queue_keeps_eureka_append_only_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue = load_queue_definition(
        repo_root / "configs" / "experiment_queues" / "dqs1_pairset_local_first.yaml"
    )
    assert queue["queue_id"] == "dqs1_pairset_local_first"
    assert len(queue["experiments"]) == 1
    experiment = queue["experiments"][0]
    candidate_id = experiment["id"]
    steps = {step["id"]: step for step in experiment["steps"]}
    eureka = steps["local_cpu_contest_drift_eureka"]
    command = eureka["command"]
    eureka_out = command[command.index("--eureka-out") + 1]
    assert eureka_out.startswith(f".omx/research/local_cpu_contest_drift_eureka_{candidate_id}_")
    assert eureka_out.endswith("Z.json")
    assert "20260522T224218Z" not in eureka_out

    false_authority = next(
        condition
        for condition in eureka["postconditions"]
        if condition["type"] == "json_false_authority"
    )
    assert false_authority["path"] == eureka_out
    assert false_authority["required_false"] == list(EUREKA_FALSE_AUTHORITY_FIELDS)
    assert false_authority["false_or_missing"] == []


def test_dqs1_queue_builder_fails_closed_on_authority_fields(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text())
    payload["top_operator_actions"][0]["score_claim"] = "true"
    summary.write_text(json.dumps(payload))

    with pytest.raises(ExperimentQueueError, match="exactly false"):
        build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")


def test_find_latest_cross_family_action_summary_uses_path_timestamp(tmp_path: Path) -> None:
    older = (
        tmp_path
        / "experiments"
        / "results"
        / "cross_family_candidate_portfolio"
        / "20260522T180032Z_observed"
    )
    newer = (
        tmp_path
        / "experiments"
        / "results"
        / "cross_family_candidate_portfolio"
        / "20260522T202400Z_hardened"
    )
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(json.dumps({"operator_action_rows": []}))
    for path, candidate_id in (
        (older / "action_summary.json", "pairset_drop_one_rank023_pair0440"),
        (newer / "action_summary.json", "pairset_drop_one_rank024_pair0112"),
    ):
        path.write_text(
            json.dumps(
                {
                    **_false_authority(),
                    "schema": "pairset_component_marginal_canonicalization_summary.v1",
                    "portfolio_json": str(portfolio),
                    "top_operator_actions": [_action(candidate_id, 1)],
                }
            )
        )

    assert find_latest_cross_family_action_summary(tmp_path) == newer / "action_summary.json"
