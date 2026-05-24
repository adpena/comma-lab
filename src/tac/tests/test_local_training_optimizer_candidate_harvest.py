# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import connect_state, initialize_queue_state
from comma_lab.scheduler.local_training_harvest import (
    LOCAL_TRAINING_HARVEST_SCHEMA,
    LocalTrainingHarvestError,
    harvest_local_training_optimizer_candidates,
)
from comma_lab.scheduler.local_training_queue import build_local_training_execution_queue
from tac.optimization.local_training_harvest_intelligence import (
    LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA,
    OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA,
)
from tac.optimization.optimizer_scheduler_registry import (
    TELEMETRY_SCHEMA,
    default_optimizer_scheduler_registry,
)
from tac.substrates._shared.trainer_skeleton import (
    write_representation_training_probe_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _plan(
    repo: Path,
    *,
    candidate_id: str,
    representation_manifest_name: str = "representation_training_manifest.json",
    optimizer_descriptor_id: str | None = None,
) -> dict:
    output_dir = repo / candidate_id
    representation_manifest = output_dir / representation_manifest_name
    config_sha256 = (
        default_optimizer_scheduler_registry().get(optimizer_descriptor_id).config_sha256
        if optimizer_descriptor_id
        else None
    )
    command = [
        ".venv/bin/python",
        "tools/run_local_training_plan.py",
        "--output",
        str(output_dir / "manifest.json"),
        "--representation-manifest",
        str(representation_manifest),
    ]
    return {
        "schema": "representation_training_probe_plan_v1",
        "candidate_id": candidate_id,
        "lane_id": "lane_local_training_harvest_fixture",
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_params": {
            "stage_index": 8,
            "seed": 23,
            "optimizer_descriptor_id": optimizer_descriptor_id,
            "optimizer_config_sha256": config_sha256,
            "parameter_group_lr_policy_id": (
                "embedding_theta1_hidden_muon_adamw"
                if optimizer_descriptor_id
                else None
            ),
            "parameter_group_lr_policy_sha256": (
                "b" * 64 if optimizer_descriptor_id else None
            ),
        },
        "recommended_execution": {
            "schema": "local_training_recommended_execution.v1",
            "tool": "tools/run_local_training_plan.py",
            "training_backend": "mlx",
            "device": "mlx",
            "resource_kind": "local_mlx",
            "output_manifest": str(output_dir / "manifest.json"),
            "representation_manifest": str(representation_manifest),
            "python_command_args": command,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }


def _write_completed_manifest(path: Path, *, candidate_id: str, seconds: float = 1.25) -> None:
    optimizer_config_sha256 = default_optimizer_scheduler_registry().get(
        "pr95_stage8_muon_adamw_mlx"
    ).config_sha256
    write_representation_training_probe_manifest(
        path,
        candidate_id=candidate_id,
        lane_id="lane_local_training_harvest_fixture",
        lane_class="local_training_harvest_fixture",
        candidate_family="local_training_harvest_fixture",
        representation_family="hnerv",
        substrate_family="nerv_family",
        profile="local_training_harvest_fixture",
        param_schema="local_training_harvest_fixture_params.v1",
        training_signal_kind="local_mlx_representation_training_runtime_probe",
        device_requested="mlx",
        device_selected="mlx",
        output_dir=str(path.parent),
        seed=23,
        stages=[{"index": 8, "module": "stage8_muon_finetune"}],
        stage_count=1,
        candidate_params={
            "stage_index": 8,
            "optimizer_descriptor_id": "pr95_stage8_muon_adamw_mlx",
            "optimizer_config_sha256": optimizer_config_sha256,
            "parameter_group_lr_policy_id": "embedding_theta1_hidden_muon_adamw",
            "parameter_group_lr_policy_sha256": "b" * 64,
            "steps": 1,
        },
        dispatch_blockers=[
            "representation_training_probe_is_proxy_signal",
            "requires_exact_cpu_cuda_auth_eval_before_score_claim",
        ],
        evidence_grade="[macOS-MLX research-signal]",
        extra_fields={
            "runtime_profile": {
                "schema": "trainer_runtime_profile_observation.v1",
                "candidate_id": candidate_id,
                "training_backend": "mlx",
                "seed": 23,
                "stage_index": 8,
                "stage_id": "stage8_muon_finetune",
                "state_bytes": 4096,
                "seconds_per_step": seconds,
                "kernel_fusion_strategy_id": "local_training_harvest_fixture",
                "kernel_fusion": {
                    "kernel_fusion_strategy_id": "local_training_harvest_fixture",
                    "operator_mix": {"conv2d": 1, "newton_schulz5": 1},
                },
                "packet_compiler_bridge": {
                    "packet_compiler_target_declared": True,
                    "runtime_consumption_proof_required": True,
                    "runtime_consumption_proof_present": False,
                    "blockers": ["runtime_consumption_proof_missing"],
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        },
    )


def _queue(repo: Path, *plans: dict) -> dict:
    return build_local_training_execution_queue(
        list(plans),
        queue_id="local_training_harvest_fixture",
        repo_root=repo,
        local_mlx_concurrency=2,
    )


def _state(repo: Path, queue: dict, statuses: dict[str, str]) -> Path:
    state_path = repo / "queue.sqlite"
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
        for experiment_id, status in statuses.items():
            conn.execute(
                """
                UPDATE step_state
                SET status = ?, updated_at_utc = '2026-05-24T00:00:00Z',
                    last_event_json = ?
                WHERE queue_id = ? AND experiment_id = ? AND step_id = 'run_local_training'
                """,
                (
                    status,
                    json.dumps({"test_status": status}),
                    queue["queue_id"],
                    experiment_id,
                ),
            )
        conn.commit()
    return state_path


def test_harvest_uses_only_succeeded_representation_manifests(tmp_path: Path) -> None:
    plan_a = _plan(tmp_path, candidate_id="candidate_a")
    plan_b = _plan(tmp_path, candidate_id="candidate_b")
    queue = _queue(tmp_path, plan_a, plan_b)
    exp_a = queue["experiments"][0]["id"]
    exp_b = queue["experiments"][1]["id"]
    _write_completed_manifest(
        tmp_path / "candidate_a" / "representation_training_manifest.json",
        candidate_id="candidate_a",
        seconds=0.5,
    )
    _write_completed_manifest(
        tmp_path / "candidate_b" / "representation_training_manifest.json",
        candidate_id="candidate_b",
        seconds=0.25,
    )
    state_path = _state(tmp_path, queue, {exp_a: "succeeded", exp_b: "failed"})

    harvested = harvest_local_training_optimizer_candidates(
        queue,
        state_path=state_path,
        repo_root=tmp_path,
    )

    assert harvested["schema"] == "optimizer_candidate_queue_v1"
    assert harvested["n_candidates"] == 1
    assert harvested["dispatch_ready_count"] == 0
    assert harvested["score_claim"] is False
    assert harvested["promotion_eligible"] is False
    assert harvested["rank_or_kill_eligible"] is False
    assert harvested["ready_for_exact_eval_dispatch"] is False
    assert harvested["top_k"][0]["candidate_id"] == "candidate_a"
    assert harvested["top_k"][0]["rank_score"] == 0.5
    assert harvested["harvest"]["schema"] == LOCAL_TRAINING_HARVEST_SCHEMA
    assert harvested["harvest"]["harvested_representation_manifest_count"] == 1
    assert harvested["harvest"]["skipped_steps"][0]["status"] == "failed"


def test_harvest_refuses_when_no_steps_succeeded(tmp_path: Path) -> None:
    plan = _plan(tmp_path, candidate_id="candidate_a")
    queue = _queue(tmp_path, plan)
    state_path = _state(tmp_path, queue, {queue["experiments"][0]["id"]: "queued"})

    with pytest.raises(LocalTrainingHarvestError, match="no succeeded"):
        harvest_local_training_optimizer_candidates(
            queue,
            state_path=state_path,
            repo_root=tmp_path,
        )


def test_harvest_refuses_plan_sidecar_even_if_state_claims_success(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        candidate_id="candidate_a",
        representation_manifest_name="representation_training_plan.json",
    )
    queue = _queue(tmp_path, plan)
    state_path = _state(tmp_path, queue, {queue["experiments"][0]["id"]: "succeeded"})

    with pytest.raises(LocalTrainingHarvestError, match="refuses non-manifest"):
        harvest_local_training_optimizer_candidates(
            queue,
            state_path=state_path,
            repo_root=tmp_path,
        )


def test_harvest_refuses_false_authority_leakage(tmp_path: Path) -> None:
    plan = _plan(tmp_path, candidate_id="candidate_a")
    queue = _queue(tmp_path, plan)
    manifest_path = tmp_path / "candidate_a" / "representation_training_manifest.json"
    _write_completed_manifest(manifest_path, candidate_id="candidate_a")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["promotion_eligible"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = _state(tmp_path, queue, {queue["experiments"][0]["id"]: "succeeded"})

    with pytest.raises(LocalTrainingHarvestError, match="promotion_eligible"):
        harvest_local_training_optimizer_candidates(
            queue,
            state_path=state_path,
            repo_root=tmp_path,
        )


def test_harvest_refuses_queue_manifest_identity_mismatch(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        candidate_id="candidate_a",
        optimizer_descriptor_id="pr95_stage8_muon_adamw_mlx",
    )
    queue = _queue(tmp_path, plan)
    manifest_path = tmp_path / "candidate_a" / "representation_training_manifest.json"
    _write_completed_manifest(manifest_path, candidate_id="candidate_a")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["candidate_params"]["optimizer_descriptor_id"] = "other_optimizer"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = _state(tmp_path, queue, {queue["experiments"][0]["id"]: "succeeded"})

    with pytest.raises(LocalTrainingHarvestError, match="identity mismatch"):
        harvest_local_training_optimizer_candidates(
            queue,
            state_path=state_path,
            repo_root=tmp_path,
        )


def test_harvest_cli_writes_optimizer_candidate_queue(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        candidate_id="candidate_a",
        optimizer_descriptor_id="pr95_stage8_muon_adamw_mlx",
    )
    queue = _queue(tmp_path, plan)
    queue_path = tmp_path / "experiment_queue.json"
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    _write_completed_manifest(
        tmp_path / "candidate_a" / "representation_training_manifest.json",
        candidate_id="candidate_a",
        seconds=0.75,
    )
    state_path = _state(tmp_path, queue, {queue["experiments"][0]["id"]: "succeeded"})
    output = tmp_path / "optimizer_candidate_queue.json"
    intelligence_output = tmp_path / "optimizer_harvest_intelligence.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "harvest_local_training_optimizer_candidates.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(output),
            "--intelligence-output",
            str(intelligence_output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    harvested = json.loads(output.read_text(encoding="utf-8"))
    intelligence = json.loads(intelligence_output.read_text(encoding="utf-8"))
    assert summary["harvested_representation_manifest_count"] == 1
    assert summary["intelligence_schema"] == LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA
    assert summary["telemetry_record_count"] == 1
    assert summary["atom_count"] == 1
    assert harvested["n_candidates"] == 1
    assert harvested["dispatch_ready_count"] == 0
    assert harvested["score_claim"] is False
    assert harvested["promotion_eligible"] is False
    assert harvested["rank_or_kill_eligible"] is False
    assert harvested["ready_for_exact_eval_dispatch"] is False
    assert harvested["top_k"][0]["rank_score_field"] == (
        "seconds_per_step_cost_signal_not_score"
    )
    assert "predicted_score_mean" not in harvested["top_k"][0]
    assert "score_mean" not in harvested["top_k"][0]
    assert "quality_evidence" not in harvested["top_k"][0]
    assert intelligence["schema"] == LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA
    assert intelligence["score_claim"] is False
    assert intelligence["promotion_eligible"] is False
    assert intelligence["ready_for_exact_eval_dispatch"] is False
    assert intelligence["atom_count"] == 1
    assert intelligence["telemetry_record_count"] == 1
    telemetry = intelligence["optimizer_scheduler_telemetry"]
    assert telemetry["schema"] == OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA
    row = telemetry["records"][0]
    assert row["schema"] == TELEMETRY_SCHEMA
    assert row["descriptor_id"] == "pr95_stage8_muon_adamw_mlx"
    assert row["axis_tag"] == "[macOS-MLX research-signal]"
    assert row["seconds_per_step"] == 0.75
    assert row["seconds_per_candidate"] is None
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["metadata"]["source_candidate_id"] == "candidate_a"
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
