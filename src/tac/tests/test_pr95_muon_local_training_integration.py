# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.pr95_muon_local_training_integration import (
    CANDIDATE_PAYLOAD_SCHEMA,
    PLAN_SCHEMA,
    PR95MuonLocalTrainingIntegrationError,
    adapt_pr95_local_training_manifest_to_candidate,
    validate_pr95_local_training_manifest,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimizer.candidate_queue import build_candidate_queue


def _manifest() -> dict[str, object]:
    return {
        "schema": "pr95_local_training_probe_manifest_v1",
        "lane_id": "lane_pr95_local_mps_source_faithful_training_probe_20260519",
        "source_tree_sha256": "a" * 64,
        "device_selected": "cpu",
        "torch_version": "2.11.0",
        "platform": "Darwin-arm64",
        "seed": 1234,
        "full_curriculum": False,
        "stage_count": 8,
        "stages": [
            {"index": 1, "module": "stage1_v328_ce"},
            {"index": 8, "module": "stage8_muon_finetune"},
        ],
        "results": [
            {
                "stage_index": 1,
                "stage_module": "stage1_v328_ce",
                "epochs_run": 1,
                "wall_seconds": 12.5,
                "best_score": 0.215,
            },
            {
                "stage_index": 8,
                "stage_module": "stage8_muon_finetune",
                "epochs_run": 2,
                "wall_seconds": 22.5,
                "best_score": 0.193,
            },
        ],
        "archive_zip": {
            "path": "experiments/results/pr95_probe/archive.zip",
            "bytes": 178417,
            "sha256": "b" * 64,
            "member_sha256": "c" * 64,
        },
        "auth_eval_bridge": {
            "ok": True,
            "score_axis": "macOS-CPU advisory",
            "auth_eval_json_sha256": "d" * 64,
            "auth_eval_canonical_score": 0.194,
            "score_comparable": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_pr95_manifest_adapter_stamps_false_authority(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(_manifest()), encoding="utf-8")

    row = adapt_pr95_local_training_manifest_to_candidate(
        _manifest(),
        source_path=source,
        repo_root=tmp_path,
    )

    assert row["candidate_id"] == "pr95_muon_hnerv_local_cpu_stages8_seed1234"
    assert row["representation_family"] == "hnerv"
    assert row["substrate_family"] == "nerv_family"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["rank_score"] == 0.193
    assert row["rank_score_field"] == "training_best_score_proxy_not_authority"
    assert row["auth_eval_bridge_score"] is None
    assert row["advisory_auth_eval_bridge_score"] == 0.194
    assert row["auth_eval_bridge_score_axis"] == "macOS-CPU advisory"
    assert row["auth_eval_bridge_score_comparable"] is False
    assert row["score_affecting_payload_changed"] is False
    assert row["charged_bits_changed"] is False
    assert validate_proxy_candidate(row) == []
    assert row["consumer_payload"]["schema"] == CANDIDATE_PAYLOAD_SCHEMA
    assert row["consumer_payload"]["pr95_muon_local_training"]["muon_partition"][
        "hidden_2d_plus_weights"
    ] == "Muon"
    assert row["solver_stack_wire_in"]["atom_wire_in"]["atom_kind"] == "meta_lagrangian"


def test_pr95_manifest_adapter_preserves_optimizer_descriptor_identity(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["optimizer_recipe"] = {
        "id": "pr95_stage8_muon_adamw_mlx",
        "optimizer_descriptor_id": "pr95_stage8_muon_adamw_mlx",
        "optimizer_config_sha256": "a" * 64,
        "optimizer_backend_status": "implemented_mlx_source_faithful",
        "parameter_group_lr_policy_id": "embedding_theta1_hidden_muon_adamw",
        "parameter_group_lr_policy_sha256": "b" * 64,
        "parameter_group_fingerprint_sha256": "c" * 64,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    row = adapt_pr95_local_training_manifest_to_candidate(
        payload,
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    params = row["candidate_params"]
    assert params["optimizer_descriptor_id"] == "pr95_stage8_muon_adamw_mlx"
    assert params["optimizer_config_sha256"] == "a" * 64
    assert params["optimizer_backend_status"] == "implemented_mlx_source_faithful"
    assert params["parameter_group_lr_policy_id"] == (
        "embedding_theta1_hidden_muon_adamw"
    )
    assert params["parameter_group_fingerprint_sha256"] == "c" * 64


def test_pr95_manifest_adapter_rejects_truthy_authority() -> None:
    payload = _manifest()
    payload["rank_or_kill_eligible"] = True

    with pytest.raises(PR95MuonLocalTrainingIntegrationError, match="rank_or_kill"):
        validate_pr95_local_training_manifest(payload)


def test_pr95_manifest_adapter_rejects_nested_truthy_authority() -> None:
    payload = _manifest()
    payload["optimizer_recipe"] = {
        "id": "unsafe",
        "ready_for_exact_eval_dispatch": 1,
    }

    with pytest.raises(
        PR95MuonLocalTrainingIntegrationError,
        match=r"optimizer_recipe\.ready_for_exact_eval_dispatch=truthy",
    ):
        validate_pr95_local_training_manifest(payload)


def test_pr95_manifest_adapter_emits_consumer_payload_for_cathedral_and_mlx_sweep(
    tmp_path: Path,
) -> None:
    row = adapt_pr95_local_training_manifest_to_candidate(
        _manifest(),
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    payload = row["consumer_payload"]["pr95_muon_local_training"]
    assert payload["stage_count"] == 8
    assert payload["archive_export"]["present"] is True
    assert payload["auth_bridge"]["axis"] == "macOS-CPU advisory"
    assert "stage8_muon_finetune" in payload["stage_modules"]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in payload[
        "missing_blockers"
    ]


def test_pr95_manifest_adapter_carries_local_runtime_profile(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["runtime_profile"] = {
        "schema": "trainer_runtime_profile_observation.v1",
        "training_backend": "mlx",
        "seconds_per_epoch": 12.0,
        "peak_memory_bytes": 4_000_000,
        "kernel_fusion_strategy_id": "pr95_mlx_stage8_profile",
        "operator_mix": {"conv2d": 0.76, "gemm": 0.11, "norm": 0.02},
        "packet_compiler_bridge": {
            "packet_compiler_target_declared": True,
            "archive_export_schema": "pr95_hnerv_archive",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_present": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    row = adapt_pr95_local_training_manifest_to_candidate(
        manifest,
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    runtime_summary = row["consumer_payload"]["pr95_muon_local_training"][
        "timing_smoke"
    ]["runtime_profile_summary"]
    assert runtime_summary["profile_count"] == 1
    assert runtime_summary["best_local_backend"] == "mlx"
    assert runtime_summary["best_timing_value_seconds"] == 12.0
    assert row["candidate_params"]["best_local_backend"] == "mlx"
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []


def test_pr95_manifest_adapter_ranks_runtime_only_signal(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["results"] = []
    manifest["runtime_profile"] = {
        "schema": "trainer_runtime_profile_observation.v1",
        "training_backend": "mlx",
        "seconds_per_epoch": 3.75,
        "peak_memory_bytes": 4_000_000,
        "kernel_fusion_strategy_id": "pr95_mlx_stage8_profile",
        "operator_mix": {"conv2d": 0.76, "gemm": 0.11, "norm": 0.02},
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    row = adapt_pr95_local_training_manifest_to_candidate(
        manifest,
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    assert row["rank_score"] == 3.75
    assert row["rank_score_field"] == "seconds_per_epoch_cost_signal_not_score"
    assert "local_training_probe_best_score_missing" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_accepts_pr95_local_training_manifest_as_planning_only(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pr95_manifest.json"
    manifest.write_text(
        json.dumps(_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_family"] == "pr95_hnerv_muon_training_probe"
    assert row["rank_score"] == 0.193
    assert row["rank_score_field"] == "training_best_score_proxy_not_authority"
    assert row["auth_eval_bridge_score"] is None
    assert row["advisory_auth_eval_bridge_score"] == 0.194
    assert row["consumer_payload"]["schema"] == CANDIDATE_PAYLOAD_SCHEMA
    assert "pr95_local_training_probe_is_proxy_signal" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_accepts_pr95_local_training_plan_schema_as_planning_only(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["schema"] = PLAN_SCHEMA
    manifest = tmp_path / "pr95_plan.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_family"] == "pr95_hnerv_muon_training_probe"
    assert row["rank_score"] == 0.193
    assert row["advisory_auth_eval_bridge_score"] == 0.194
    assert validate_proxy_candidate(row) == []
