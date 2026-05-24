# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.atom.types import AtomKind, ResolutionPath
from tac.optimization.local_training_harvest_intelligence import (
    OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA,
)
from tac.optimization.optimizer_guided_candidate_generation import (
    generate_candidate_queue,
    load_profile,
)
from tac.optimization.optimizer_scheduler_registry import (
    TELEMETRY_SCHEMA,
    default_optimizer_scheduler_registry,
)
from tac.optimization.optimizer_signal_atoms import (
    OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA,
    OptimizerSignalAtomError,
    build_atoms_from_optimizer_signal_source,
    build_optimizer_signal_atom_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "materialize_optimizer_signal_atoms.py"


def _pr95_queue() -> dict[str, object]:
    return generate_candidate_queue(
        profile=load_profile("pr95_hnerv_muon_training_smoke"),
        optimizer="cmaes",
        max_candidates=8,
        top_k=3,
        seed=20260522,
    )


def _harvested_pr95_queue() -> dict[str, object]:
    descriptor = default_optimizer_scheduler_registry().get(
        "pr95_stage5_adamw_baseline_mlx"
    )
    queue = _pr95_queue()
    row = dict(queue["top_k"][0])  # type: ignore[index]
    row["candidate_id"] = "pr95_hnerv_mlx_stage5_seed17_steps1_c36"
    row["candidate_params"] = {
        **dict(row.get("candidate_params") or {}),
        "optimizer_descriptor_id": descriptor.descriptor_id,
        "optimizer_config_sha256": descriptor.config_sha256,
        "seed": 17,
        "stage_index": 5,
        "steps": 1,
    }
    row["rank_score"] = 0.05
    row["rank_score_field"] = "seconds_per_step_cost_signal_not_score"
    row["consumer_payload"] = {
        "schema": "representation_training_candidate_payload.v1",
        "representation_training_probe": {
            "timing_smoke": {
                "runtime_profile_summary": {
                    "schema": "trainer_runtime_profile_summary.v1",
                    "profile_count": 1,
                    "best_local_backend": "mlx",
                    "best_scheduler_resource_kind": "local_mlx",
                    "best_timing_field": "seconds_per_step",
                    "best_timing_value_seconds": 0.05,
                    "kernel_fusion_strategy_ids": [
                        "native_mlx_pr95_hnerv_decoder_muon_adamw_v1"
                    ],
                    "operator_mix_keys": ["conv2d", "pixel_shuffle_2x"],
                    "blockers": ["local_mlx_training_profile_not_score_authority"],
                    "profiles": [
                        {
                            "schema": "trainer_runtime_profile_observation.v1",
                            "candidate_id": row["candidate_id"],
                            "profile_id": row["candidate_id"],
                            "training_backend": "mlx",
                            "evidence_grade": "[macOS-MLX research-signal]",
                            "seed": 17,
                            "stage_index": 5,
                            "stage_id": "stage5_c1a_l7",
                            "state_bytes": 916056,
                            "timing_field": "seconds_per_step",
                            "timing_value_seconds": 0.05,
                            "seconds_per_step": 0.05,
                            "kernel_fusion": {
                                "kernel_fusion_strategy_id": (
                                    "native_mlx_pr95_hnerv_decoder_muon_adamw_v1"
                                ),
                                "operator_mix": {"conv2d": 15},
                            },
                            "packet_compiler_bridge": {
                                "runtime_consumption_proof_required": True,
                                "runtime_consumption_proof_present": False,
                                "blockers": ["runtime_consumption_proof_missing"],
                            },
                            "blockers": ["local_mlx_training_profile_not_score_authority"],
                            "score_claim": False,
                            "score_claim_valid": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                            "promotable": False,
                            "dispatch_attempted": False,
                            "gpu_launched": False,
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            }
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    queue.update(
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [row],
            "top_k_forensic": [row],
            "n_candidates": 1,
            "dispatch_ready_count": 0,
            "dispatch_ready": [],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    return queue


def test_optimizer_guided_queue_materializes_proxy_only_atoms() -> None:
    queue = _pr95_queue()

    atoms = build_atoms_from_optimizer_signal_source(
        queue,
        source_path="experiments/results/unit/optimizer_queue.json",
    )

    assert len(atoms) == 3
    atom = atoms[0]
    assert atom.kind == AtomKind.META_LAGRANGIAN
    assert atom.resolution_path == ResolutionPath.LEARNED
    assert atom.predicted_impact_delta_s_lower == 0.0
    assert atom.predicted_impact_delta_s_upper == 0.0
    assert set(atom.wired_hooks) == {
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot_dispatch",
        "continual_learning_posterior",
        "probe_disambiguator",
    }
    assert atom.provenance["evidence_grade"] == "predicted"
    assert atom.provenance["promotion_eligible"] is False
    assert atom.provenance["score_claim_valid"] is False
    assert atom.metadata["candidate_family"] == "pr95_hnerv_muon_optimizer_recipe_smoke"
    assert atom.metadata["representation_family"] == "hnerv"
    assert atom.metadata["substrate_family"] == "nerv_family"
    assert atom.metadata["score_impact_band_status"] == (
        "neutral_until_exact_auth_or_calibrated_posterior"
    )
    assert atom.metadata["false_authority"]["score_claim"] is False
    assert atom.metadata["false_authority"]["ready_for_exact_eval_dispatch"] is False
    assert atom.metadata["parameter_group_fingerprint_sha256"]
    assert atom.metadata["proxy_objective"] is not None
    assert atom.metadata["rank_score_field"] == "proxy_objective_not_score"


def test_optimizer_signal_atom_ledger_includes_meta_lagrangian_projection() -> None:
    queue = _pr95_queue()

    ledger = build_optimizer_signal_atom_ledger(queue, source_path="queue.json", max_atoms=2)

    assert ledger["schema"] == OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA
    assert ledger["atom_count"] == 2
    assert ledger["score_claim"] is False
    assert ledger["promotion_eligible"] is False
    assert ledger["ready_for_exact_eval_dispatch"] is False
    assert len(ledger["atoms"]) == 2
    assert len(ledger["meta_lagrangian_atoms"]) == 2
    first_projection = ledger["meta_lagrangian_atoms"][0]
    assert first_projection["family"] == "atom_kind:meta_lagrangian"
    assert first_projection["family_group"] == "resolution_path:learned"
    assert first_projection["expected_score_delta"] == 0.0


def test_truthy_proxy_authority_is_rejected() -> None:
    queue = _pr95_queue()
    top_k = queue["top_k"]
    assert isinstance(top_k, list)
    row = dict(top_k[0])
    row["score_claim"] = True
    queue["top_k"] = [row]

    with pytest.raises(OptimizerSignalAtomError, match="proxy contract violations"):
        build_atoms_from_optimizer_signal_source(queue)


def test_materialize_optimizer_signal_atoms_cli(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    json_out = tmp_path / "atoms.json"
    jsonl_out = tmp_path / "atoms.jsonl"
    queue_path.write_text(
        json.dumps(_pr95_queue(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate-queue",
            str(queue_path),
            "--json-out",
            str(json_out),
            "--jsonl-out",
            str(jsonl_out),
            "--max-atoms",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "count=2" in proc.stdout
    assert "score_claim=false" in proc.stdout
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA
    assert payload["atom_count"] == 2
    lines = jsonl_out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "meta_lagrangian"
    assert first["resolution_path"] == "learned"


def test_materialize_harvested_optimizer_queue_emits_atoms_and_scheduler_telemetry(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "harvested_queue.json"
    atoms_out = tmp_path / "atoms.json"
    telemetry_out = tmp_path / "telemetry.json"
    intelligence_out = tmp_path / "intelligence.json"
    queue_path.write_text(
        json.dumps(_harvested_pr95_queue(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate-queue",
            str(queue_path),
            "--json-out",
            str(atoms_out),
            "--scheduler-telemetry-json-out",
            str(telemetry_out),
            "--intelligence-json-out",
            str(intelligence_out),
            "--max-atoms",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "count=1" in proc.stdout
    atoms = json.loads(atoms_out.read_text(encoding="utf-8"))
    telemetry = json.loads(telemetry_out.read_text(encoding="utf-8"))
    intelligence = json.loads(intelligence_out.read_text(encoding="utf-8"))
    assert atoms["atom_count"] == 1
    atom_metadata = atoms["atoms"][0]["metadata"]
    runtime_cost = atom_metadata["runtime_cost_observation"]
    assert runtime_cost["schema"] == "optimizer_signal_runtime_cost_observation.v1"
    assert runtime_cost["score_claim"] is False
    assert runtime_cost["summary"]["best_local_backend"] == "mlx"
    assert runtime_cost["summary"]["best_timing_field"] == "seconds_per_step"
    assert runtime_cost["summary"]["best_timing_value_seconds"] == 0.05
    assert runtime_cost["profiles"][0]["state_bytes"] == 916056
    assert runtime_cost["profiles"][0]["kernel_fusion_strategy_id"] == (
        "native_mlx_pr95_hnerv_decoder_muon_adamw_v1"
    )
    prior = atom_metadata["scheduler_cost_prior_hint"]
    assert prior["schema"] == "optimizer_scheduler_cost_prior_hint.v1"
    assert prior["cost_signal_only"] is True
    assert prior["training_backend"] == "mlx"
    assert prior["scheduler_resource_kind"] == "local_mlx"
    assert prior["timing_field"] == "seconds_per_step"
    assert prior["timing_value_seconds"] == 0.05
    assert prior["ready_for_exact_eval_dispatch"] is False
    assert telemetry["schema"] == OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA
    assert telemetry["record_count"] == 1
    row = telemetry["records"][0]
    assert row["schema"] == TELEMETRY_SCHEMA
    assert row["descriptor_id"] == "pr95_stage5_adamw_baseline_mlx"
    assert row["seconds_per_step"] == 0.05
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert intelligence["atom_count"] == 1
    assert intelligence["telemetry_record_count"] == 1
