# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.comma_lab.scheduler.dqs1_local_first_harvest import (
    EXACT_AUTH_ANCHOR_REQUEST_SCHEMA,
    HARVEST_SCHEMA,
    build_dqs1_harvest_result,
    candidate_experiment_ids,
)
from src.comma_lab.scheduler.dqs1_local_first_queue import (
    build_queue_from_action_summary,
    candidate_slug,
    find_latest_cross_family_action_summary,
)
from src.comma_lab.scheduler.experiment_queue import ExperimentQueueError, load_queue_definition
from src.tac.optimization.local_cpu_contest_drift import EUREKA_FALSE_AUTHORITY_FIELDS
from tools.harvest_dqs1_local_first_result import _write_json as write_harvest_json


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


def _write_cross_family_summary(repo: Path) -> Path:
    summary_dir = (
        repo
        / "experiments"
        / "results"
        / "cross_family_candidate_portfolio"
        / "20260523T121036Z_full_drop_two_local_harvest"
    )
    summary_dir.mkdir(parents=True)
    portfolio = summary_dir / "portfolio.json"
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
                "schema": "cross_family_candidate_portfolio_action_summary.v1",
                "json_out": str(portfolio),
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
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                    "device": "cpu",
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
    signal.parent.mkdir(parents=True, exist_ok=True)
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
    assert result.queue["controls"]["max_concurrency"]["local_io_heavy"] == 1
    assert {step["resources"]["kind"] for step in experiment["steps"]} == {
        "local_cpu",
        "local_io_heavy",
    }
    steps_by_id = {step["id"]: step for step in experiment["steps"]}
    assert list(steps_by_id) == [
        "build_bridge_plan",
        "plan_packet",
        "materialize",
        "locality_controls",
        "local_cpu_advisory",
        "plan_raw_artifact_retention",
        "local_cpu_contest_drift_eureka",
    ]
    assert steps_by_id["build_bridge_plan"]["requires"] == []
    assert steps_by_id["plan_packet"]["requires"] == ["build_bridge_plan"]
    locality = steps_by_id["locality_controls"]
    assert locality["resources"]["kind"] == "local_io_heavy"
    assert locality["timeout_seconds"] == 960
    locality_command = locality["command"]
    assert locality_command[locality_command.index("--timeout-seconds") + 1] == "540"
    assert (
        locality_command[locality_command.index("--global-timeout-seconds") + 1]
        == "840"
    )
    assert (
        locality_command[locality_command.index("--max-inflate-parallelism") + 1]
        == "3"
    )
    assert "--reuse-existing-inflates" in locality_command
    advisory = steps_by_id["local_cpu_advisory"]
    advisory_command = advisory["command"]
    assert advisory["timeout_seconds"] == 3600
    assert advisory_command[advisory_command.index("--inflate-timeout") + 1] == "1800"
    assert advisory_command[advisory_command.index("--evaluate-timeout") + 1] == "1800"
    selected_pairs_arg = steps_by_id["plan_packet"]["command"][-1]
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
    raw_retention = steps_by_id["plan_raw_artifact_retention"]
    assert raw_retention["requires"] == ["local_cpu_advisory"]
    assert raw_retention["resources"]["kind"] == "local_io_heavy"
    assert raw_retention["timeout_seconds"] == 1200
    assert "tools/compact_experiment_artifacts.py" in raw_retention["command"]
    assert "results/materialized/drop_rank024_pair0112/raw_artifact_retention_plan.json" in raw_retention["command"]
    assert "--include-kind" not in raw_retention["command"]
    assert any(
        condition == {
            "type": "json_equals",
            "path": "results/materialized/drop_rank024_pair0112/raw_artifact_retention_plan.json",
            "key": "plan.blocked_candidate_count",
            "equals": 0,
        }
        for condition in raw_retention["postconditions"]
    )


def test_dqs1_queue_builder_skips_completed_candidate_from_extra_results_root(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    source_completed = tmp_path / "results" / "materialized" / candidate_slug(
        "pairset_drop_one_rank023_pair0440"
    )
    legacy_root = tmp_path / "legacy_results"
    legacy_completed = legacy_root / "materialized" / candidate_slug(
        "pairset_drop_one_rank023_pair0440"
    )
    legacy_completed.parent.mkdir(parents=True)
    source_completed.replace(legacy_completed)
    advisory = legacy_completed / "local_cpu_advisory.json"
    archive = legacy_completed / "submission_dir" / "archive.zip"
    payload = json.loads(advisory.read_text(encoding="utf-8"))
    payload["provenance"]["archive_path"] = str(archive)
    advisory.write_text(json.dumps(payload), encoding="utf-8")
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="external_results",
        completed_results_roots=(str(legacy_root),),
    )

    assert result.selection.candidate_id == "pairset_drop_one_rank024_pair0112"
    assert result.selection.skipped_candidates == (
        {
            "candidate_id": "pairset_drop_one_rank023_pair0440",
            "reason": "local_advisory_exists",
            "completed_results_root": str(legacy_root),
        },
    )


def test_dqs1_queue_builder_accepts_cross_family_action_summary_schema(
    tmp_path: Path,
) -> None:
    summary = _write_cross_family_summary(tmp_path)

    result = build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")

    assert result.selection.candidate_id == "pairset_drop_one_rank023_pair0440"
    assert result.selection.portfolio_path == summary.parent / "portfolio.json"
    plan_step = result.queue["experiments"][0]["steps"][1]
    assert plan_step["id"] == "plan_packet"
    assert plan_step["command"][-1] == "1,2,440"


def test_dqs1_queue_builder_can_emit_multiple_local_first_candidates(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        candidate_limit=2,
        local_cpu_concurrency=2,
    )

    assert [selection.candidate_id for selection in result.selections] == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]
    assert result.selection.candidate_id == "pairset_drop_one_rank023_pair0440"
    assert result.queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert result.queue["controls"]["max_concurrency"]["local_io_heavy"] == 1
    assert result.queue["controls"]["max_concurrency"]["modal_gpu"] == 0
    assert [experiment["id"] for experiment in result.queue["experiments"]] == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]


def test_dqs1_queue_builder_accepts_local_io_concurrency(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        candidate_limit=2,
        local_cpu_concurrency=4,
        local_io_concurrency=2,
    )

    assert result.queue["controls"]["max_concurrency"]["local_cpu"] == 4
    assert result.queue["controls"]["max_concurrency"]["local_io_heavy"] == 2
    bridge_outputs = {
        experiment["id"]: {
            step["id"]: step["command"]
            for step in experiment["steps"]
        }["build_bridge_plan"]
        for experiment in result.queue["experiments"]
    }
    assert any(
        "materialized/drop_rank023_pair0440/decoder_q_selective_window_bridge_plan.json" in part
        for part in bridge_outputs["pairset_drop_one_rank023_pair0440"]
    )
    assert any(
        "materialized/drop_rank024_pair0112/decoder_q_selective_window_bridge_plan.json" in part
        for part in bridge_outputs["pairset_drop_one_rank024_pair0112"]
    )
    materialize_commands = {
        experiment["id"]: {
            step["id"]: step["command"]
            for step in experiment["steps"]
        }["materialize"]
        for experiment in result.queue["experiments"]
    }
    assert any(
        "materialized/drop_rank023_pair0440/submission_dir" in part
        for part in materialize_commands["pairset_drop_one_rank023_pair0440"]
    )
    assert any(
        "materialized/drop_rank024_pair0112/submission_dir" in part
        for part in materialize_commands["pairset_drop_one_rank024_pair0112"]
    )
    assert all(
        condition["type"] == "json_false_authority"
        for experiment in result.queue["experiments"]
        for step in experiment["steps"]
        for condition in step["postconditions"]
        if condition["type"] == "json_false_authority"
    )


def test_dqs1_queue_builder_can_emit_local_mlx_advisory_debug_steps(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        eureka_run_id="20260523T010203Z",
        include_mlx_local_advisory_debug=True,
        allow_large_mlx_cache=True,
        mlx_reference_cache_dir="reference/full600",
        mlx_device="gpu",
    )

    experiment = result.queue["experiments"][0]
    steps_by_id = {step["id"]: step for step in experiment["steps"]}
    assert list(steps_by_id) == [
        "build_bridge_plan",
        "plan_packet",
        "materialize",
        "locality_controls",
        "local_cpu_advisory",
        "plan_raw_artifact_retention",
        "build_mlx_local_advisory_cache",
        "local_mlx_advisory_response",
        "plan_mlx_delta_cache_retention",
        "local_cpu_contest_drift_eureka",
    ]
    build_cache = steps_by_id["build_mlx_local_advisory_cache"]
    assert build_cache["requires"] == ["local_cpu_advisory"]
    assert build_cache["resources"]["kind"] == "local_cpu"
    assert "tools/build_mlx_scorer_input_cache_from_local_advisory.py" in build_cache["command"]
    assert "--allow-large-tensor-cache" in build_cache["command"]
    audit_output = build_cache["command"][build_cache["command"].index("--audit-output") + 1]
    assert audit_output == (
        ".omx/research/"
        "mlx_delta_cache_local_cpu_advisory_identity_drop_rank023_pair0440_20260523T010203Z.json"
    )
    assert any(
        condition == {"type": "json_equals", "path": audit_output, "key": "passed", "equals": True}
        for condition in build_cache["postconditions"]
    )

    mlx_response = steps_by_id["local_mlx_advisory_response"]
    assert mlx_response["requires"] == ["build_mlx_local_advisory_cache"]
    assert mlx_response["resources"]["kind"] == "local_mlx"
    assert "tools/run_mlx_scorer_response_from_local_advisory.py" in mlx_response["command"]
    assert "reference/full600" in mlx_response["command"]
    assert "--allow-gpu-research-signal" in mlx_response["command"]
    assert "--allow-local-cpu-advisory-cache-identity" in mlx_response["command"]
    false_authority = next(
        condition
        for condition in mlx_response["postconditions"]
        if condition["type"] == "json_false_authority"
    )
    assert false_authority["axis_key"] == "score_axis"
    assert false_authority["axis_equals"] == "[macOS-MLX research-signal]"
    retention = steps_by_id["plan_mlx_delta_cache_retention"]
    assert retention["requires"] == ["local_mlx_advisory_response"]
    assert retention["resources"]["kind"] == "local_io_heavy"
    assert retention["timeout_seconds"] == 1200
    assert "tools/compact_experiment_artifacts.py" in retention["command"]
    assert "mlx_scorer_input_cache" in retention["command"]
    assert any(
        condition == {
            "type": "json_equals",
            "path": "results/materialized/drop_rank023_pair0440/mlx_delta_cache_retention_plan.json",
            "key": "plan.candidate_count",
            "equals": 1,
        }
        for condition in retention["postconditions"]
    )


def test_dqs1_queue_builder_can_emit_scheduler_preflight_gate(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first",
        include_scheduler_preflight=True,
        scheduler_storage_workload_subdir="experiments/results/dqs1_local_first",
        scheduler_storage_expected_workload_root=(
            "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"
        ),
        scheduler_storage_tiers=("vertigo=/Volumes/VertigoDataTier/pact",),
        scheduler_proactive_cleanup_roots=("experiments/results", ".omx/tmp"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_cold_store_roots=(
            "/Volumes/VertigoDataTier/pact/cold_store",
        ),
    )

    assert [experiment["id"] for experiment in result.queue["experiments"]][:2] == [
        "dqs1_scheduler_preflight",
        "pairset_drop_one_rank023_pair0440",
    ]
    preflight = result.queue["experiments"][0]
    assert [step["id"] for step in preflight["steps"]] == [
        "storage_tier_plan",
        "proactive_cleanup",
    ]
    assert "tools/plan_experiment_storage.py" in preflight["steps"][0]["command"]
    assert "tools/compact_experiment_artifacts.py" in preflight["steps"][1]["command"]
    assert "--execute" in preflight["steps"][1]["command"]
    assert "--action" in preflight["steps"][1]["command"]
    assert "move" in preflight["steps"][1]["command"]
    assert preflight["steps"][1]["resources"]["kind"] == "local_io_heavy"
    assert preflight["steps"][1]["timeout_seconds"] == 1200
    candidate_steps = {
        step["id"]: step for step in result.queue["experiments"][1]["steps"]
    }
    assert candidate_steps["build_bridge_plan"]["requires"] == [
        "dqs1_scheduler_preflight.proactive_cleanup"
    ]


def test_dqs1_queue_builder_preflight_hashes_existing_storage_plan(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    storage_plan = (
        tmp_path / ".omx" / "research" / "dqs1_local_first_storage_plan_20260522.json"
    )
    storage_plan.parent.mkdir(parents=True)
    storage_plan.write_text('{"prior":true}\n', encoding="utf-8")
    expected_sha = sha256(storage_plan.read_bytes()).hexdigest()

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first",
        include_scheduler_preflight=True,
        scheduler_storage_workload_subdir="experiments/results/dqs1_local_first",
        scheduler_storage_expected_workload_root=(
            "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"
        ),
        scheduler_storage_tiers=("vertigo=/Volumes/VertigoDataTier/pact",),
        scheduler_proactive_cleanup_roots=("experiments/results", ".omx/tmp"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_cold_store_roots=(
            "/Volumes/VertigoDataTier/pact/cold_store",
        ),
    )

    command = result.queue["experiments"][0]["steps"][0]["command"]
    assert "--expected-output-sha256" in command
    assert command[command.index("--expected-output-sha256") + 1] == expected_sha


def test_dqs1_queue_builder_preflight_uses_run_id_not_stale_summary_date(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first",
        include_scheduler_preflight=True,
        scheduler_storage_expected_workload_root=(
            "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"
        ),
        scheduler_storage_tiers=("vertigo=/Volumes/VertigoDataTier/pact",),
        scheduler_proactive_cleanup_roots=("experiments/results", ".omx/tmp"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_cold_store_roots=(
            "/Volumes/VertigoDataTier/pact/cold_store",
        ),
        eureka_run_id="20260524T010203Z",
    )

    preflight = result.queue["experiments"][0]
    storage_command = preflight["steps"][0]["command"]
    cleanup_command = preflight["steps"][1]["command"]
    assert ".omx/research/dqs1_local_first_storage_plan_20260524T010203Z.json" in storage_command
    assert (
        ".omx/research/dqs1_local_first_proactive_cleanup_20260524T010203Z.json"
        in cleanup_command
    )


def test_dqs1_queue_builder_rejects_dry_run_scheduler_preflight_cleanup(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    with pytest.raises(
        ExperimentQueueError,
        match="scheduler_proactive_cleanup_execute must be true",
    ):
        build_queue_from_action_summary(
            summary,
            repo_root=tmp_path,
            results_root=str(tmp_path / "dqs1_local_first"),
            include_scheduler_preflight=True,
            scheduler_storage_expected_workload_root=str(tmp_path / "dqs1_local_first"),
        )


def test_dqs1_scheduler_preflight_move_cleanup_uses_policy_cold_store_default(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first",
        include_scheduler_preflight=True,
        scheduler_storage_expected_workload_root=(
            "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"
        ),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_action="move",
    )

    cleanup_command = result.queue["experiments"][0]["steps"][1]["command"]
    assert "/Volumes/VertigoDataTier/pact/cold_store" in cleanup_command
    assert "/Volumes/APDataStore/pact/cold_store" in cleanup_command


def test_dqs1_queue_builder_rejects_preflight_outputs_outside_workload_root(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    with pytest.raises(ExperimentQueueError, match="results_root outside"):
        build_queue_from_action_summary(
            summary,
            repo_root=tmp_path,
            results_root=str(tmp_path / "outside" / "dqs1_local_first"),
            include_scheduler_preflight=True,
            scheduler_storage_expected_workload_root=str(tmp_path / "inside"),
            scheduler_proactive_cleanup_execute=True,
            scheduler_proactive_cleanup_action="delete",
        )


def test_dqs1_queue_builder_requires_explicit_large_mlx_cache_ack(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    with pytest.raises(ExperimentQueueError, match="allow_large_mlx_cache"):
        build_queue_from_action_summary(
            summary,
            repo_root=tmp_path,
            results_root="results",
            include_mlx_local_advisory_debug=True,
        )


def test_dqs1_queue_builder_rejects_mlx_batch_shape_before_gate(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    with pytest.raises(ExperimentQueueError, match="batch-shape invariance"):
        build_queue_from_action_summary(
            summary,
            repo_root=tmp_path,
            results_root="results",
            include_mlx_local_advisory_debug=True,
            allow_large_mlx_cache=True,
            mlx_batch_pairs=2,
        )


@pytest.mark.parametrize(
    ("selected_pair_indices", "match"),
    [
        ([1, 1, 2], "contains duplicates"),
        ([-1, 2], "out of range"),
        ([600], "out of range"),
        ([2, 1], "sorted ascending"),
    ],
)
def test_dqs1_queue_builder_rejects_malformed_selected_pair_indices(
    tmp_path: Path,
    selected_pair_indices: list[int],
    match: str,
) -> None:
    summary = _write_summary(tmp_path)
    portfolio = tmp_path / "portfolio.json"
    payload = json.loads(portfolio.read_text())
    payload["operator_action_rows"][0]["source_metadata"][
        "selected_pair_indices"
    ] = selected_pair_indices
    payload["operator_action_rows"][0]["source_metadata"][
        "selected_pair_count"
    ] = len(selected_pair_indices)
    portfolio.write_text(json.dumps(payload))

    with pytest.raises(ExperimentQueueError, match=match):
        build_queue_from_action_summary(summary, repo_root=tmp_path, results_root="results")


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


def test_dqs1_harvest_observe_only_reroutes_queue(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        eureka_run_id="20260522T000000Z",
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(result.queue))
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )

    harvest = build_dqs1_harvest_result(
        queue_path=queue_path,
        repo_root=tmp_path,
        timestamp="20260523T010203Z",
        reroute_observe_only=True,
        output_queue_path=queue_path,
        expected_output_queue_sha256=sha256(queue_path.read_bytes()).hexdigest(),
        results_root="results",
    )

    assert harvest.harvest_record["schema"] == HARVEST_SCHEMA
    assert harvest.harvest_record["candidate_id"] == "pairset_drop_one_rank023_pair0440"
    assert harvest.harvest_record["recommended_action"] == "observe_only"
    assert harvest.harvest_record["score_claim"] is False
    assert harvest.harvest_record["promotion_eligible"] is False
    assert harvest.exact_auth_request is None
    assert harvest.rerouted_queue is not None
    assert harvest.rerouted_queue["experiments"][0]["id"] == "pairset_drop_one_rank024_pair0112"
    written = load_queue_definition(queue_path)
    assert written["experiments"][0]["id"] == "pairset_drop_one_rank024_pair0112"


def test_dqs1_harvest_selects_candidate_from_multi_candidate_batch(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        candidate_limit=2,
        include_scheduler_preflight=True,
        scheduler_storage_expected_workload_root=str(tmp_path / "results"),
        scheduler_proactive_cleanup_execute=True,
        scheduler_proactive_cleanup_action="delete",
        eureka_run_id="20260522T000000Z",
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(result.queue))
    candidate_ids = candidate_experiment_ids(result.queue)
    assert candidate_ids == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]
    for candidate_id in candidate_ids:
        advisory, _archive, archive_sha = _write_completed_local_advisory(
            tmp_path,
            candidate_id=candidate_id,
        )
        _write_eureka_signal(
            tmp_path,
            candidate_id=candidate_id,
            advisory=advisory,
            archive_sha=archive_sha,
        )

    prior_queue_sha = sha256(queue_path.read_bytes()).hexdigest()
    harvest = build_dqs1_harvest_result(
        queue_path=queue_path,
        repo_root=tmp_path,
        candidate_id="pairset_drop_one_rank024_pair0112",
        timestamp="20260523T010203Z",
        reroute_observe_only=False,
        results_root="results",
    )

    assert harvest.harvest_record["schema"] == HARVEST_SCHEMA
    assert harvest.harvest_record["candidate_id"] == "pairset_drop_one_rank024_pair0112"
    assert harvest.harvest_record["recommended_action"] == "observe_only"
    assert harvest.exact_auth_request is None
    assert harvest.rerouted_queue is None
    assert sha256(queue_path.read_bytes()).hexdigest() == prior_queue_sha

    with pytest.raises(ExperimentQueueError, match="multi-candidate queue"):
        build_dqs1_harvest_result(
            queue_path=queue_path,
            repo_root=tmp_path,
            candidate_id="pairset_drop_one_rank024_pair0112",
            timestamp="20260523T010204Z",
            reroute_observe_only=True,
            output_queue_path=queue_path,
            expected_output_queue_sha256=prior_queue_sha,
            results_root="results",
        )


def test_dqs1_harvest_observe_only_all_candidates_consumed_returns_no_reroute(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        eureka_run_id="20260522T000000Z",
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(result.queue))
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )
    advisory_2, _archive_2, archive_sha_2 = _write_completed_local_advisory(
        tmp_path,
        candidate_id="pairset_drop_one_rank024_pair0112",
    )
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank024_pair0112",
        advisory=advisory_2,
        archive_sha=archive_sha_2,
    )
    prior_queue_sha = sha256(queue_path.read_bytes()).hexdigest()

    harvest = build_dqs1_harvest_result(
        queue_path=queue_path,
        repo_root=tmp_path,
        timestamp="20260523T010203Z",
        reroute_observe_only=True,
        output_queue_path=queue_path,
        expected_output_queue_sha256=prior_queue_sha,
        results_root="results",
    )

    assert harvest.harvest_record["schema"] == HARVEST_SCHEMA
    assert harvest.harvest_record["candidate_id"] == "pairset_drop_one_rank023_pair0440"
    assert harvest.harvest_record["recommended_action"] == "observe_only"
    assert harvest.exact_auth_request is None
    assert harvest.rerouted_queue is None
    assert sha256(queue_path.read_bytes()).hexdigest() == prior_queue_sha


def test_dqs1_harvest_positive_eureka_creates_exact_auth_request(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        eureka_run_id="20260522T000000Z",
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(result.queue))
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
        eureka_trigger=True,
        recommended_action="dispatch_exact_auth_anchor",
    )

    harvest = build_dqs1_harvest_result(
        queue_path=queue_path,
        repo_root=tmp_path,
        timestamp="20260523T010203Z",
        reroute_observe_only=True,
        output_queue_path=queue_path,
        expected_output_queue_sha256=sha256(queue_path.read_bytes()).hexdigest(),
        results_root="results",
    )

    assert harvest.harvest_record["recommended_action"] == "dispatch_exact_auth_anchor"
    assert harvest.rerouted_queue is None
    assert harvest.exact_auth_request is not None
    request = harvest.exact_auth_request
    assert request["schema"] == EXACT_AUTH_ANCHOR_REQUEST_SCHEMA
    assert request["candidate_id"] == "pairset_drop_one_rank023_pair0440"
    assert request["requested_axes"] == ["contest-CPU", "contest-CUDA"]
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False
    assert request["ready_for_exact_eval_dispatch"] is False


def test_dqs1_harvest_reroute_refuses_unexpected_queue_overwrite(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        eureka_run_id="20260522T000000Z",
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(result.queue))
    advisory, _archive, archive_sha = _write_completed_local_advisory(tmp_path)
    _write_eureka_signal(
        tmp_path,
        candidate_id="pairset_drop_one_rank023_pair0440",
        advisory=advisory,
        archive_sha=archive_sha,
    )

    with pytest.raises(ExperimentQueueError, match="expected_output_queue_sha256"):
        build_dqs1_harvest_result(
            queue_path=queue_path,
            repo_root=tmp_path,
            timestamp="20260523T010203Z",
            reroute_observe_only=True,
            output_queue_path=queue_path,
            results_root="results",
        )

    with pytest.raises(ExperimentQueueError, match="sha256 mismatch"):
        build_dqs1_harvest_result(
            queue_path=queue_path,
            repo_root=tmp_path,
            timestamp="20260523T010204Z",
            reroute_observe_only=True,
            output_queue_path=queue_path,
            expected_output_queue_sha256="0" * 64,
            results_root="results",
        )


def test_dqs1_harvest_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "harvest.json"
    write_harvest_json(path, {"schema": "first"})

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_harvest_json(path, {"schema": "second"})

    assert json.loads(path.read_text()) == {"schema": "first"}


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
    steps = {step["id"]: step["command"] for step in experiment["steps"]}
    assert steps["plan_packet"][-1] == "26,59,68,98,109,112"
    assert any(
        "materialized/drop_two_r001_002_p0371_0320/submission_dir" in part
        for part in steps["materialize"]
    )


def test_dqs1_queue_builder_threads_runtime_overrides(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    result = build_queue_from_action_summary(
        summary,
        repo_root=tmp_path,
        results_root="results",
        mlx_effective_selection="mlx/effective_selection.json",
        decoder_q_candidate_manifest="decoder_q/mutation_manifest.json",
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
    assert "mlx/effective_selection.json" in steps["build_bridge_plan"]
    assert "decoder_q/mutation_manifest.json" in steps["build_bridge_plan"]
    assert "results/materialized/drop_rank023_pair0440/decoder_q_selective_window_bridge_plan.json" in steps["plan_packet"]
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
    candidate_experiments = [
        experiment
        for experiment in queue["experiments"]
        if experiment["id"] != "dqs1_scheduler_preflight"
    ]
    assert len(candidate_experiments) >= 2
    assert queue["controls"]["max_concurrency"]["local_cpu"] >= 2
    assert queue["controls"]["max_concurrency"]["local_io_heavy"] == 1
    for experiment in candidate_experiments:
        candidate_id = experiment["id"]
        steps = {step["id"]: step for step in experiment["steps"]}
        locality = steps["locality_controls"]
        assert locality["resources"]["kind"] == "local_io_heavy"
        assert locality["timeout_seconds"] == 960
        assert "--global-timeout-seconds" in locality["command"]
        assert "--max-inflate-parallelism" in locality["command"]
        assert "--reuse-existing-inflates" in locality["command"]
        raw_retention = steps["plan_raw_artifact_retention"]
        assert raw_retention["resources"]["kind"] == "local_io_heavy"
        assert raw_retention["timeout_seconds"] == 1200
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
