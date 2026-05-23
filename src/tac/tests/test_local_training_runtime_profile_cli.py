# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "canonicalize_local_training_runtime_profile.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_canonicalizes_standalone_mlx_profile_to_planning_queue(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "mlx_profile.json",
        {
            "schema": "trainer_runtime_profile_observation.v1",
            "candidate_id": "boostnerv_seed7",
            "profile_id": "seed7_mlx_fused",
            "lane_id": "boostnerv_local_training",
            "representation_family": "boostnerv",
            "substrate_family": "nerv_family",
            "training_backend": "mlx",
            "seconds_per_epoch": 3.25,
            "peak_memory_bytes": 8_000_000_000,
            "kernel_fusion": {
                "strategy_id": "mlx_compile_rmsnorm_residual_gemm",
                "operator_mix": {"matmul": 2, "rmsnorm": 1},
                "measured": True,
            },
            "packet_compiler_bridge": {
                "packet_compiler_target_declared": True,
                "archive_export_schema": "boostnerv_packetir_archive_export.v1",
                "runtime_consumption_proof_required": True,
                "runtime_consumption_proof_present": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    )
    output = tmp_path / "runtime_profile_canonical.json"
    queue_output = tmp_path / "runtime_profile_queue.json"

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--queue-output",
            str(queue_output),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "runtime_profiles=1" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_output.read_text(encoding="utf-8"))
    row = queue["top_k"][0]

    assert payload["schema"] == "local_training_runtime_profile_canonicalization.v1"
    assert payload["runtime_profile_count"] == 1
    assert payload["sources"][0]["best_local_backend"] == "mlx"
    assert queue["schema"] == "optimizer_candidate_queue_v1"
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["evidence_grade"] == "[macOS-MLX research-signal]"
    assert row["candidate_params"]["kernel_fusion_strategy_id"] == (
        "mlx_compile_rmsnorm_residual_gemm"
    )
    assert "local_mlx_training_profile_not_score_authority" in row["dispatch_blockers"]


def test_cli_canonicalizes_embedded_runtime_profiles_without_score_authority(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "representation_manifest.json",
        {
            "schema": "representation_training_probe_manifest_v1",
            "candidate_id": "hnerv_muon_seed11",
            "lane_id": "pr95_hnerv_muon",
            "representation_family": "hnerv",
            "substrate_family": "nerv_family",
            "runtime_profiles": [
                {
                    "training_backend": "cpu",
                    "seconds_per_epoch": 11.0,
                    "kernel_fusion_strategy_id": "none",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotable": False,
                },
                {
                    "training_backend": "mlx",
                    "seconds_per_epoch": 2.0,
                    "kernel_fusion_strategy_id": "mlx_graph_compile",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotable": False,
                },
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    )
    output = tmp_path / "embedded_runtime_profile_canonical.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    source_payload = payload["sources"][0]
    queue_row = payload["candidate_queue"]["top_k"][0]

    assert source_payload["profile_count"] == 2
    assert source_payload["best_local_backend"] == "mlx"
    assert source_payload["best_timing_value_seconds"] == 2.0
    assert queue_row["ready_for_exact_eval_dispatch"] is False
    assert queue_row["rank_or_kill_eligible"] is False
    assert queue_row["consumer_payload"]["representation_training_probe"]["timing_smoke"][
        "runtime_profile_summary"
    ]["profile_count"] == 2
    assert source_payload["source_authority"]["score_claim"] is False


def test_cli_rejects_embedded_runtime_profile_manifest_root_authority(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "representation_manifest.json",
        {
            "schema": "representation_training_probe_manifest_v1",
            "candidate_id": "hnerv_muon_seed11",
            "runtime_profiles": [
                {
                    "training_backend": "mlx",
                    "seconds_per_epoch": 2.0,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotable": False,
                }
            ],
            "score_claim": True,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    )
    output = tmp_path / "unsafe_runtime_profile_canonical.json"

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "score_claim=truthy" in result.stderr


def test_cli_supports_yaml_sources_for_summary_and_queue(
    tmp_path: Path,
) -> None:
    yaml = pytest.importorskip("yaml")
    source = tmp_path / "representation_manifest.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema": "representation_training_probe_manifest_v1",
                "candidate_id": "hnerv_muon_seed12",
                "lane_id": "pr95_hnerv_muon",
                "representation_family": "hnerv",
                "substrate_family": "nerv_family",
                "runtime_profiles": [
                    {
                        "training_backend": "mlx",
                        "seconds_per_epoch": 1.5,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "promotable": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "yaml_runtime_profile_canonical.json"
    queue_output = tmp_path / "yaml_runtime_profile_queue.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--queue-output",
            str(queue_output),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_output.read_text(encoding="utf-8"))
    assert payload["runtime_profile_count"] == 1
    assert queue["top_k"][0]["candidate_id"] == "hnerv_muon_seed12"
