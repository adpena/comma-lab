# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimization.representation_training_probe_integration import (
    CANDIDATE_PAYLOAD_SCHEMA,
    PLAN_SCHEMA,
    RepresentationTrainingProbeIntegrationError,
    adapt_representation_training_manifest_to_candidate,
    validate_representation_training_manifest,
)
from tac.optimizer.candidate_queue import build_candidate_queue


def _manifest() -> dict[str, object]:
    return {
        "schema": "representation_training_probe_manifest_v1",
        "candidate_id": "siren_wavelet_hybrid_training_smoke_seed17",
        "lane_id": "offline_siren_wavelet_hybrid_training_probe",
        "lane_class": "siren_wavelet_hybrid_training_proxy",
        "candidate_family": "siren_wavelet_hybrid_optimizer_probe",
        "representation_family": "siren_wavelet_hybrid",
        "substrate_family": "non_nerv_learned_plus_signal_processing",
        "profile": "generic_representation_training_probe",
        "param_schema": "representation_training_manifest_params_v1",
        "training_signal_kind": "local_representation_training_optimizer_schedule_probe",
        "seed": 17,
        "device_selected": "mlx",
        "source_tree_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "training_recipe": {"id": "micro_pair_weighted_smoke"},
        "optimizer_recipe": {"id": "adamw_muon_partitioned_candidate"},
        "scheduler_recipe": {"id": "wsd_polyak_tail"},
        "candidate_params": {
            "lr_log10_delta": -0.2,
            "pair_weight_temperature": 0.7,
            "byte_budget_lambda": 1.1,
        },
        "stage_count": 2,
        "stages": [
            {"index": 1, "module": "coarse_signal_fit"},
            {"index": 2, "module": "pair_weighted_refine"},
        ],
        "results": [
            {
                "stage_index": 1,
                "stage_module": "coarse_signal_fit",
                "epochs_run": 1,
                "wall_seconds": 8.5,
                "best_score": 0.21,
            },
            {
                "stage_index": 2,
                "stage_module": "pair_weighted_refine",
                "epochs_run": 2,
                "wall_seconds": 21.0,
                "best_score": 0.198,
            },
        ],
        "archive_zip": {
            "path": "experiments/results/siren_wavelet/archive.zip",
            "bytes": 181000,
            "sha256": "c" * 64,
            "member_sha256": "d" * 64,
        },
        "auth_eval_bridge": {
            "ok": True,
            "score_axis": "macOS-MLX research-signal",
            "auth_eval_json_sha256": "e" * 64,
            "auth_eval_canonical_score": 0.199,
            "score_comparable": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "variant_axes": [
            "source_faithful_control",
            "optimizer_recipe",
            "scheduler_recipe",
            "representation_substrate",
        ],
        "paired_modes": [
            "source_faithful_control",
            "optimizer_variant",
            "substrate_variant",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_generic_representation_training_manifest_is_substrate_agnostic_proxy(
    tmp_path: Path,
) -> None:
    row = adapt_representation_training_manifest_to_candidate(
        _manifest(),
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    assert row["candidate_family"] == "siren_wavelet_hybrid_optimizer_probe"
    assert row["representation_family"] == "siren_wavelet_hybrid"
    assert row["substrate_family"] == "non_nerv_learned_plus_signal_processing"
    assert row["rank_score"] == 0.198
    assert row["rank_score_field"] == "training_best_score_proxy_not_authority"
    assert row["auth_eval_bridge_score"] is None
    assert row["advisory_auth_eval_bridge_score"] == 0.199
    assert row["auth_eval_bridge_score_axis"] == "macOS-MLX research-signal"
    assert row["auth_eval_bridge_score_comparable"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_affecting_payload_changed"] is False
    assert row["charged_bits_changed"] is False
    assert row["consumer_payload"]["schema"] == CANDIDATE_PAYLOAD_SCHEMA
    assert row["consumer_payload"]["representation_training_probe"][
        "representation_family"
    ] == "siren_wavelet_hybrid"
    wire = row["solver_stack_wire_in"]
    assert wire["representation_family"] == "siren_wavelet_hybrid"
    assert wire["substrate_family"] == "non_nerv_learned_plus_signal_processing"
    assert wire["variant_axes"] == [
        "source_faithful_control",
        "optimizer_recipe",
        "scheduler_recipe",
        "representation_substrate",
    ]
    assert wire["probe_disambiguator_wire_in"]["paired_modes"] == [
        "source_faithful_control",
        "optimizer_variant",
        "substrate_variant",
    ]
    assert validate_proxy_candidate(row) == []


def test_generic_representation_training_manifest_carries_runtime_profile(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["runtime_profile"] = {
        "schema": "trainer_runtime_profile_observation.v1",
        "training_backend": "mlx",
        "seconds_per_epoch": 5.5,
        "peak_memory_bytes": 1_000_000,
        "kernel_fusion_strategy_id": "measured_mlx_conv_profile",
        "operator_mix": {"conv2d": 0.81, "gemm": 0.07},
        "packet_compiler_bridge": {
            "packet_compiler_target_declared": True,
            "archive_export_schema": "generic_representation_archive_v1",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_present": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    row = adapt_representation_training_manifest_to_candidate(
        payload,
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    runtime_summary = row["consumer_payload"]["representation_training_probe"][
        "timing_smoke"
    ]["runtime_profile_summary"]
    assert runtime_summary["profile_count"] == 1
    assert runtime_summary["best_local_backend"] == "mlx"
    assert runtime_summary["best_scheduler_resource_kind"] == "local_mlx"
    assert runtime_summary["best_timing_value_seconds"] == 5.5
    assert row["candidate_params"]["best_local_backend"] == "mlx"
    assert row["candidate_params"]["best_scheduler_resource_kind"] == "local_mlx"
    assert row["candidate_params"]["best_runtime_timing_value_seconds"] == 5.5
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []


def test_generic_representation_training_manifest_ranks_runtime_only_signal(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["results"] = []
    payload["runtime_profile"] = {
        "schema": "trainer_runtime_profile_observation.v1",
        "training_backend": "mlx",
        "seconds_per_epoch": 4.25,
        "peak_memory_bytes": 1_000_000,
        "kernel_fusion_strategy_id": "measured_mlx_conv_profile",
        "operator_mix": {"conv2d": 0.81, "gemm": 0.07},
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    row = adapt_representation_training_manifest_to_candidate(
        payload,
        source_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
    )

    assert row["rank_score"] == 4.25
    assert row["rank_score_field"] == "seconds_per_epoch_cost_signal_not_score"
    assert row["training_best_score"] is None
    assert "representation_training_best_score_missing" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert validate_proxy_candidate(row) == []


def test_generic_representation_training_manifest_rejects_truthy_authority() -> None:
    payload = _manifest()
    payload["auth_eval_bridge"]["promotable"] = True  # type: ignore[index]

    with pytest.raises(RepresentationTrainingProbeIntegrationError, match="promotable"):
        validate_representation_training_manifest(payload)


def test_generic_representation_training_manifest_rejects_nested_truthy_authority() -> None:
    payload = _manifest()
    payload["optimizer_recipe"] = {
        "id": "unsafe",
        "ready_for_exact_eval_dispatch": "true",
    }

    with pytest.raises(
        RepresentationTrainingProbeIntegrationError,
        match=r"optimizer_recipe\.ready_for_exact_eval_dispatch=truthy",
    ):
        validate_representation_training_manifest(payload)


def test_candidate_queue_accepts_generic_representation_training_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "representation_training_manifest.json"
    manifest.write_text(
        json.dumps(_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "siren_wavelet_hybrid_training_smoke_seed17"
    assert row["candidate_family"] == "siren_wavelet_hybrid_optimizer_probe"
    assert row["rank_score"] == 0.198
    assert row["rank_score_field"] == "training_best_score_proxy_not_authority"
    assert row["auth_eval_bridge_score"] is None
    assert row["advisory_auth_eval_bridge_score"] == 0.199
    assert "representation_training_probe_is_proxy_signal" in row["dispatch_blockers"]
    assert row["consumer_payload"]["schema"] == CANDIDATE_PAYLOAD_SCHEMA
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_accepts_generic_representation_training_plan_schema(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["schema"] = PLAN_SCHEMA
    manifest = tmp_path / "representation_training_plan.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "siren_wavelet_hybrid_training_smoke_seed17"
    assert row["rank_score"] == 0.198
    assert row["advisory_auth_eval_bridge_score"] == 0.199
    assert validate_proxy_candidate(row) == []
