# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    FEEDBACK_REFRESH_SCHEMA,
    FrontierRateAttackFeedbackError,
    build_frontier_rate_attack_feedback_refresh,
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (
    AUTOPILOT_RESULT_SCHEMA,
    FrontierRateAttackFeedbackCycleError,
    harvest_paths_from_autopilot_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


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


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _advisory(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    raw_sha256: str,
    runtime_sha256: str,
    seg: float = 0.055,
    pose: float = 0.017,
) -> dict[str, object]:
    rate = archive_size_bytes / 10_000_000.0
    score = seg + pose + rate
    return {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "score_seg_contribution": seg,
        "score_pose_contribution": pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": archive_size_bytes,
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": runtime_sha256,
            },
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": raw_sha256,
                },
            },
        },
    }


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


def _write_action_summary(repo: Path) -> Path:
    summary_dir = repo / "experiments" / "results" / "cross_family_candidate_portfolio"
    portfolio = summary_dir / "portfolio.json"
    candidate_rows = [
        ("pairset_drop_one_rank023_pair0440", [1, 2, 440]),
        ("pairset_drop_one_rank024_pair0112", [1, 2, 112]),
        ("pairset_drop_one_rank025_pair0233", [1, 2, 233]),
    ]
    _write_json(
        portfolio,
        {"operator_action_rows": [_row(candidate, pairs) for candidate, pairs in candidate_rows]},
    )
    return _write_json(
        summary_dir / "action_summary.json",
        {
            **_false_authority(),
            "schema": "cross_family_candidate_portfolio_action_summary.v1",
            "json_out": str(portfolio),
            "top_operator_actions": [
                _action(candidate, rank)
                for rank, (candidate, _pairs) in enumerate(candidate_rows, start=1)
            ],
        },
    )


def _materializer_observation(
    *,
    observation_id: str,
    target_kind: str,
    saved_bytes: int,
    receiver_contract_satisfied: bool,
    rate_positive: bool,
) -> dict[str, object]:
    return {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        **_false_authority(),
        "observation_id": observation_id,
        "candidate_id": observation_id,
        "target_kind": target_kind,
        "materializer_id": f"{target_kind}_adapter",
        "saved_bytes": saved_bytes,
        "rate_positive": rate_positive,
        "savings_realized": rate_positive,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "inflate_parity_satisfied": False,
        "recommended_planner_action": (
            "keep_rate_positive_candidate_for_inflate_parity_gate"
            if rate_positive
            else "demote_matching_archive_class_for_materializer"
        ),
        "readiness_blockers": [] if receiver_contract_satisfied else ["runtime_adapter_missing"],
    }


def _write_materializer_feedback(root: Path) -> Path:
    sweep = _write_json(
        root / "receiver_smoke" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [
                _materializer_observation(
                    observation_id="zip_header_elide_receiver_positive",
                    target_kind="packet_member_zip_header_elide_v1",
                    saved_bytes=52,
                    receiver_contract_satisfied=True,
                    rate_positive=True,
                ),
                _materializer_observation(
                    observation_id="packet_member_recompress_no_delta",
                    target_kind="packet_member_recompress_v1",
                    saved_bytes=0,
                    receiver_contract_satisfied=True,
                    rate_positive=False,
                ),
            ],
        },
    )
    _write_jsonl(
        root / "tensor_probe" / "observations.jsonl",
        [
            _materializer_observation(
                observation_id="zip_header_elide_receiver_positive",
                target_kind="packet_member_zip_header_elide_v1",
                saved_bytes=52,
                receiver_contract_satisfied=True,
                rate_positive=True,
            ),
            _materializer_observation(
                observation_id="tensor_factorize_receiver_missing",
                target_kind="tensor_factorize_v1",
                saved_bytes=7,
                receiver_contract_satisfied=False,
                rate_positive=True,
            )
        ],
    )
    return sweep


def _dqs1_observation_row(
    *,
    candidate_id: str = "pairset_drop_one_rank023_pair0440",
    raw_output_or_cache_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        **_false_authority(),
        "candidate_id": candidate_id,
        "source_schema": "dqs1_local_first_harvest.v1",
        "sweep_config_id": "dqs1_local_first_macos_cpu_advisory",
        "optimization_pass_id": "local_cpu_advisory_harvest",
        "family": "decoder_q_pairset_drop_one",
        "observed_axis": "macos_cpu_advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "observed_score_or_delta": 0.1919,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "raw_output_or_cache_sha256": raw_output_or_cache_sha256,
        "component_deltas": {
            "segnet_delta": -0.0001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00002,
        },
        "score_delta_vs_baseline": -0.00012,
        "archive_byte_delta_vs_baseline": -4,
        "selected_pair_indices": [1, 2, 440],
    }


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key in AUTHORITY_KEYS:
        assert payload[key] is False


def test_frontier_feedback_compiler_discovers_materializers_and_refreshes_dqs1_queue(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        dqs1_observation_paths=(dqs1_observations,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_unit",
        candidate_limit=2,
    )

    assert report["schema"] == FEEDBACK_REFRESH_SCHEMA
    _assert_false_authority(report)
    assert report["materializer_feedback_payload_count"] == 2
    assert report["dqs1_observation_count"] == 1
    assert report["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    discovery = report["discovery"]
    assert discovery["discovered_feedback_count"] == 2
    assert discovery["duplicate_observation_count"] == 1
    discovered_targets = {
        target
        for row in discovery["discovered_feedback"]
        for target in row["target_kinds"]
    }
    assert discovered_targets == {
        "packet_member_zip_header_elide_v1",
        "packet_member_recompress_v1",
        "tensor_factorize_v1",
    }
    bridge = report["materializer_feedback_bridge"]
    assert bridge["materializer_observation_count"] == 3
    assert bridge["planned_dqs1_candidate_count"] == 2
    assert bridge["observed_dqs1_candidate_count"] == 1
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert bridge["recommended_next_action"] == (
        "materializer_receiver_positive_followup_before_dqs1_switch"
    )
    queue = report["queue"]
    assert queue["queue_id"] == "frontier_feedback_unit"
    assert len(queue["experiments"]) == 2
    assert all(
        experiment["metadata"]["materializer_feedback_bridge"] == bridge
        for experiment in queue["experiments"]
    )
    assert report["queue_summary"]["experiment_count"] == 2
    assert report["queue_summary"]["score_claim"] is False


def test_frontier_feedback_compiler_fails_closed_on_truthy_authority(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    bad = _materializer_observation(
        observation_id="bad_authority",
        target_kind="packet_member_recompress_v1",
        saved_bytes=1,
        receiver_contract_satisfied=True,
        rate_positive=True,
    )
    bad["score_claim"] = True
    _write_json(
        artifact_root / "bad" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [bad],
        },
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="forbidden truthy"):
        build_frontier_rate_attack_feedback_refresh(
            repo_root=tmp_path,
            frontier_artifact_roots=(artifact_root,),
        )


def test_frontier_feedback_cli_writes_valid_followup_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_rate_attack_feedback_refresh.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--dqs1-observation-jsonl",
            str(dqs1_observations),
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_cli_unit",
            "--candidate-limit",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    queue_path = output_dir / "dqs1_followup_queue.json"
    bridge_path = output_dir / "materializer_feedback_bridge.json"
    report_path = output_dir / "feedback_refresh_report.json"
    assert queue_path.exists()
    assert bridge_path.exists()
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifacts"]["dqs1_followup_queue"].endswith("dqs1_followup_queue.json")
    assert report["operator_commands"]["validate_followup_queue"][0] == ".venv/bin/python"

    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_feedback_cycle_harvests_batch_and_refreshes_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    baseline = _write_json(
        tmp_path / "baseline.json",
        _advisory(
            archive_sha256="1" * 64,
            archive_size_bytes=178_592,
            raw_sha256="2" * 64,
            runtime_sha256="3" * 64,
        ),
    )
    advisory = _write_json(
        tmp_path / "candidate" / "local_cpu_advisory.json",
        _advisory(
            archive_sha256="4" * 64,
            archive_size_bytes=178_559,
            raw_sha256="5" * 64,
            runtime_sha256="6" * 64,
            seg=0.054,
        ),
    )
    harvest = _write_json(
        tmp_path / ".omx" / "research" / "dqs1_local_first_harvest.json",
        {
            "schema": "dqs1_local_first_harvest.v1",
            **_false_authority(),
            "candidate_id": "pairset_drop_one_rank023_pair0440",
            "candidate_archive_sha256": "4" * 64,
            "local_cpu_advisory_path": str(advisory),
            "local_score": 0.191,
            "projected_contest_score": 0.190,
            "conservative_projected_contest_score": 0.192,
            "recommended_action": "observe_only",
            "eureka_trigger": False,
            "eureka_margin": -1e-6,
            "authority": "false_authority_dqs1_local_first_harvest",
            "harvested_at_utc": "20260525T010203Z",
            "dispatch_blockers": ["exact_cpu_cuda_auth_eval_required"],
        },
    )
    acquisition = _write_json(
        tmp_path / "pairset_acquisition.json",
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            "candidates": [
                {
                    "acquisition_id": "pairset_drop_one_rank023_pair0440",
                    "selector_id": "pairset_drop_one_rank023_pair0440",
                    "selector_kind": "drop_one_from_best",
                    "selected_pair_indices": [1, 2, 440],
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 23,
                        "dropped_pair_index": 440,
                    },
                }
            ],
        },
    )
    autopilot_result = _write_json(
        tmp_path / "autopilot_result.json",
        {
            "schema": AUTOPILOT_RESULT_SCHEMA,
            **_false_authority(),
            "queue_path": str(tmp_path / "initial_queue.json"),
            "execute": True,
            "stop_reason": "batch_harvested_waiting_for_portfolio_rebuild",
            "total_steps_started": 6,
            "candidates_harvested": 1,
            "rounds": [
                {
                    "harvests": [
                        {
                            **_false_authority(),
                            "candidate_id": "pairset_drop_one_rank023_pair0440",
                            "harvest_path": str(harvest),
                        }
                    ]
                }
            ],
        },
    )
    output_dir = tmp_path / "cycle_out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_frontier_rate_attack_feedback_cycle.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--autopilot-result-json",
            str(autopilot_result),
            "--pairset-acquisition",
            str(acquisition),
            "--baseline-advisory",
            str(baseline),
            "--baseline-archive-size-bytes",
            "178560",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_cycle_unit",
            "--post-harvest-queue-id",
            "frontier_feedback_cycle_unit_post",
            "--candidate-limit",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["harvest_path_count"] == 1
    assert payload["initial_selected_candidate_ids"] == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]
    assert payload["post_harvest_selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    observation_jsonl = Path(payload["observation_jsonl"])
    if not observation_jsonl.is_absolute():
        observation_jsonl = REPO_ROOT / observation_jsonl
    assert observation_jsonl.exists()
    cycle_report = json.loads(
        (output_dir / "frontier_rate_attack_feedback_cycle.json").read_text(
            encoding="utf-8"
        )
    )
    assert cycle_report["schema"] == "frontier_rate_attack_feedback_cycle.v1"
    assert cycle_report["post_harvest_refresh"]["queue_validate"]["valid"] is True
    assert "dynamic_observation_jsonl_to_refreshed_dqs1_queue" in cycle_report[
        "integration_edges"
    ]


def test_frontier_feedback_cycle_refuses_truthy_autopilot_authority(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": AUTOPILOT_RESULT_SCHEMA,
        **_false_authority(),
        "score_claim": True,
        "rounds": [],
    }

    with pytest.raises(FrontierRateAttackFeedbackCycleError, match="forbidden truthy"):
        harvest_paths_from_autopilot_payload(payload, repo_root=tmp_path)
