# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.local_training_runtime_profile import (
    CANDIDATE_PAYLOAD_SCHEMA,
    SCHEMA,
    LocalTrainingRuntimeProfileError,
    adapt_runtime_profile_observation_to_candidate,
    normalize_runtime_profile_observation,
    runtime_profile_summary_from_training_manifest,
    validate_runtime_profile_observation,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimizer.candidate_queue import build_candidate_queue


def _profile() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "candidate_id": "boost_nerv_mlx_smoke_seed7",
        "profile_id": "boost_nerv_local_training_smoke",
        "lane_id": "lane_boost_nerv_local_mlx_training_probe_20260523",
        "representation_family": "boost_nerv",
        "substrate_family": "nerv_family",
        "training_backend": "mlx",
        "device": "mps:mlx",
        "hardware_substrate": "apple_m_series_unified_memory",
        "seed": 7,
        "stage_id": "smoke",
        "seconds_per_epoch": 3.25,
        "examples_per_second": 18.0,
        "peak_memory_bytes": 2_147_483_648,
        "state_bytes": 512_000,
        "kernel_fusion_strategy_id": "coda_like_gemm_epilogue_profile_only",
        "operator_mix": {"conv2d": 0.72, "gemm": 0.14, "norm": 0.02},
        "backend_kernel_contract": {
            "backend": "mlx",
            "coda_cuda_hopper_ineligible": True,
            "score_claim": False,
        },
        "numerical_drift_profile": {
            "max_abs_delta": 2.5e-5,
            "score_claim": False,
        },
        "kernel_fusion_measured": True,
        "packet_compiler_bridge": {
            "packet_compiler_target_declared": True,
            "archive_export_schema": "boost_nerv_bsv1",
            "archive_export_tool": "tac.substrates.boost_nerv.archive.pack_archive",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_present": False,
        },
        "local_cloud_substitution": {
            "intended_to_replace_cloud_gpu_training": True,
            "cloud_gpu_reference": "modal_h100_training_smoke",
            "local_cost_usd": 0.0,
            "cloud_cost_usd": 12.0,
            "estimated_cloud_cost_saved_usd": 12.0,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_runtime_profile_normalizes_mlx_cost_signal_without_score_authority() -> None:
    normalized = normalize_runtime_profile_observation(_profile())

    assert normalized["schema"] == SCHEMA
    assert normalized["training_backend"] == "mlx"
    assert normalized["timing_field"] == "seconds_per_epoch"
    assert normalized["timing_value_seconds"] == 3.25
    assert normalized["kernel_fusion"]["kernel_fusion_strategy_id"] == (
        "coda_like_gemm_epilogue_profile_only"
    )
    assert normalized["packet_compiler_bridge"]["packet_compiler_target_declared"] is True
    assert "local_mlx_training_profile_not_score_authority" in normalized["blockers"]
    assert "runtime_consumption_proof_missing" in normalized["blockers"]
    assert normalized["score_claim"] is False
    assert normalized["ready_for_exact_eval_dispatch"] is False


def test_runtime_profile_adapter_and_candidate_queue_are_planning_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_profile.json"
    path.write_text(json.dumps(_profile(), indent=2, sort_keys=True) + "\n")

    row = adapt_runtime_profile_observation_to_candidate(
        _profile(),
        source_path=path,
        repo_root=tmp_path,
    )
    queue = build_candidate_queue([path], repo_root=tmp_path)
    queued = queue["top_k"][0]

    assert row["candidate_id"] == "boost_nerv_mlx_smoke_seed7::runtime_profile::mlx"
    assert row["rank_score"] == 3.25
    assert row["rank_score_field"] == "seconds_per_epoch_cost_signal_not_score"
    assert row["consumer_payload"]["schema"] == CANDIDATE_PAYLOAD_SCHEMA
    assert validate_proxy_candidate(row) == []
    assert queue["dispatch_ready_count"] == 0
    assert queued["candidate_id"] == row["candidate_id"]
    assert queued["score_claim"] is False
    assert "trainer_runtime_profile_is_cost_signal_not_score" in queued[
        "dispatch_blockers"
    ]


def test_runtime_profile_rejects_truthy_nested_authority() -> None:
    profile = _profile()
    profile["numerical_drift_profile"] = {"score_claim": True}

    with pytest.raises(LocalTrainingRuntimeProfileError, match="score_claim"):
        validate_runtime_profile_observation(profile)


def test_training_manifest_runtime_profile_summary_is_generic() -> None:
    manifest = {
        "candidate_id": "same_smoke",
        "representation_family": "siren_wavelet_hybrid",
        "substrate_family": "non_nerv_learned_plus_signal_processing",
        "runtime_profiles": [
            {
                **_profile(),
                "candidate_id": "same_smoke",
                "representation_family": "siren_wavelet_hybrid",
                "substrate_family": "non_nerv_learned_plus_signal_processing",
                "training_backend": "cpu",
                "seconds_per_epoch": 9.0,
            },
            {
                **_profile(),
                "candidate_id": "same_smoke",
                "training_backend": "mlx",
                "seconds_per_epoch": 4.0,
            },
        ],
    }

    summary = runtime_profile_summary_from_training_manifest(manifest)

    assert summary["profile_count"] == 2
    assert summary["best_local_backend"] == "mlx"
    assert summary["best_timing_value_seconds"] == 4.0
    assert "coda_like_gemm_epilogue_profile_only" in summary[
        "kernel_fusion_strategy_ids"
    ]
