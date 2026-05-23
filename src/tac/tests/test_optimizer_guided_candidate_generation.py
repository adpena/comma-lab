# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.optimizer_guided_candidate_generation import (
    QUEUE_SCHEMA,
    CandidateGenerationError,
    CandidateGenerationProfile,
    generate_candidate_queue,
    load_profile,
)
from tac.optimization.optimizer_training_signal_bridge import (
    OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA,
    validate_optimizer_training_signal_wire_in,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    validate_proxy_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_optimizer_guided_candidate_queue.py"


def test_proxy_evidence_boundary_clears_score_claim_valid() -> None:
    row = apply_proxy_evidence_boundary(
        {
            "candidate_id": "proxy_overclaim",
            "score_claim_valid": True,
            "dispatch_blockers": ["source_proxy_row"],
        }
    )

    assert row["score_claim_valid"] is False
    assert validate_proxy_candidate(row) == []


def test_pr101_bias_sidecar_cmaes_queue_is_seed_deterministic_and_proxy_only() -> None:
    profile = load_profile("pr101_bias_sidecar")

    first = generate_candidate_queue(
        profile=profile,
        optimizer="cmaes",
        max_candidates=12,
        top_k=8,
        seed=77,
    )
    second = generate_candidate_queue(
        profile=profile,
        optimizer="cmaes",
        max_candidates=12,
        top_k=8,
        seed=77,
    )

    assert first == second
    assert first["schema"] == QUEUE_SCHEMA
    assert first["dispatch_ready_count"] == 0
    assert first["dispatch_ready"] == []
    assert first["top_k"] == sorted(
        first["top_k"], key=lambda row: (row["rank_score"], row["candidate_id"])
    )
    for row in first["top_k"]:
        assert validate_proxy_candidate(row) == []
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["proxy_only"] is True
        assert row["provider_agnostic"] is True
        assert row["candidate_params"].keys() == {
            "bias_b",
            "bias_g",
            "bias_r",
            "sidecar_f1_r",
        }
        assert -1.08 <= row["candidate_params"]["bias_b"] <= -0.92
        assert -1.08 <= row["candidate_params"]["bias_g"] <= -0.92
        assert -1.08 <= row["candidate_params"]["bias_r"] <= -0.92
        assert -0.25 <= row["candidate_params"]["sidecar_f1_r"] <= 0.25
        assert "no_contest_cuda_auth_eval" in row["dispatch_blockers"]
        assert "sidecar_param_requires_archive_builder_support" in row["dispatch_blockers"]


def test_seed_changes_non_anchor_cmaes_candidates() -> None:
    profile = load_profile("pr101_bias_refine")

    first = generate_candidate_queue(
        profile=profile,
        optimizer="cmaes",
        max_candidates=10,
        seed=1,
    )
    second = generate_candidate_queue(
        profile=profile,
        optimizer="cmaes",
        max_candidates=10,
        seed=2,
    )

    first_params = [
        row["candidate_params"]
        for row in first["top_k"]
        if not row["candidate_id"].endswith("_anchor")
    ]
    second_params = [
        row["candidate_params"]
        for row in second["top_k"]
        if not row["candidate_id"].endswith("_anchor")
    ]
    assert first_params != second_params


def test_custom_profile_supports_int_param_and_optuna_style_fallback() -> None:
    profile = CandidateGenerationProfile.from_mapping(
        {
            "profile_id": "unit_int_profile",
            "lane_id": "offline_unit_int",
            "lane_class": "unit_int",
            "candidate_family": "unit_int_family",
            "param_schema": "unit_params_v1",
            "candidate_prefix": "unit",
            "score_lowering_hypothesis": "exercise bounded integer profile support",
            "parameters": [
                {"name": "q", "kind": "int", "low": 1, "high": 11, "anchor": 5},
                {"name": "bias", "low": -1.0, "high": 1.0, "anchor": 0.0},
            ],
        }
    )

    queue = generate_candidate_queue(
        profile=profile,
        optimizer="optuna",
        max_candidates=9,
        seed=123,
    )

    assert queue["optimizer_status"] == "optuna_tpe_style_stdlib"
    assert len(queue["top_k"]) == 9
    for row in queue["top_k"]:
        assert isinstance(row["candidate_params"]["q"], int)
        assert 1 <= row["candidate_params"]["q"] <= 11
        assert -1.0 <= row["candidate_params"]["bias"] <= 1.0
        assert validate_proxy_candidate(row) == []


def test_invalid_profile_bounds_fail_closed() -> None:
    with pytest.raises(CandidateGenerationError, match="high must be greater"):
        CandidateGenerationProfile.from_mapping(
            {
                "profile_id": "bad",
                "lane_id": "bad",
                "lane_class": "bad",
                "candidate_family": "bad",
                "param_schema": "bad",
                "candidate_prefix": "bad",
                "score_lowering_hypothesis": "bad",
                "parameters": [
                    {"name": "x", "low": 1.0, "high": 1.0, "anchor": 1.0},
                ],
            }
        )


def test_cli_writes_stable_queue_without_dispatching(tmp_path: Path) -> None:
    output = tmp_path / "queue.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--profile",
            "pr101_bias_refine",
            "--optimizer",
            "grid",
            "--max-candidates",
            "6",
            "--top-k",
            "4",
            "--seed",
            "20260510",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    queue = json.loads(output.read_text(encoding="utf-8"))
    assert "dispatch_ready=0" in proc.stdout
    assert "score_claim=false" in proc.stdout
    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["generated_at_utc"] == "1970-01-01T00:00:00Z"
    assert queue["dispatch_ready_count"] == 0
    assert len(queue["top_k"]) == 4
    assert all(row["score_claim"] is False for row in queue["top_k"])
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in queue["top_k"])


def test_pr95_hnerv_muon_profile_emits_full_solver_stack_wire_in() -> None:
    profile = load_profile("pr95_hnerv_muon_training_smoke")

    queue = generate_candidate_queue(
        profile=profile,
        optimizer="cmaes",
        max_candidates=8,
        top_k=5,
        seed=20260522,
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["dispatch_ready_count"] == 0
    assert queue["profile"] == "pr95_hnerv_muon_training_smoke"
    assert queue["profile_contract"]["candidate_family"] == (
        "pr95_hnerv_muon_optimizer_recipe_smoke"
    )
    assert queue["profile_contract"]["representation_family"] == "hnerv"
    assert queue["profile_contract"]["substrate_family"] == "nerv_family"
    row = queue["top_k"][0]
    assert validate_proxy_candidate(row) == []
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["rank_score_field"] == "proxy_objective_not_score"
    assert row["representation_family"] == "hnerv"
    assert row["substrate_family"] == "nerv_family"
    assert row["embedding_lr_scaling_policy"] == "theta_1_not_inverse_width"
    assert len(row["parameter_group_lr_policy_sha256"]) == 64
    assert len(row["parameter_group_fingerprint_sha256"]) == 64
    assert row["parameter_group_fingerprint"]["fingerprint_sha256"] == row[
        "parameter_group_fingerprint_sha256"
    ]
    assert row["parameter_group_fingerprint"]["fingerprint_status"] == (
        "pending_model_parameter_shape_manifest"
    )
    assert row["parameter_group_fingerprint"]["score_claim"] is False
    assert row["candidate_params"].keys() == {
        "muon_ns_steps",
        "muon_momentum",
        "hidden_weight_decay",
        "adamw_lr_ratio",
        "warmup_fraction",
        "polyak_swa_fraction",
    }
    assert "requires_master_gradient_component_marginal_anchor" in row["dispatch_blockers"]
    assert "per_pair_training_weight_schedule" in row["master_gradient_features"]
    assert "pairset_component_marginal_score_decomposition_v1" in row[
        "canonical_equation_refs"
    ]

    wire = row["solver_stack_wire_in"]
    assert wire["schema"] == OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA
    assert wire["representation_family"] == "hnerv"
    assert wire["substrate_family"] == "nerv_family"
    assert "optimizer_recipe" in wire["variant_axes"]
    assert "embedding_lr_scaling_policy" in wire["variant_axes"]
    assert "parameter_group_fingerprint" in wire["variant_axes"]
    assert "embedding_lr_scaling_policy_variant" in wire["probe_disambiguator_wire_in"][
        "paired_modes"
    ]
    assert validate_optimizer_training_signal_wire_in(wire) == []
    assert wire["false_authority"]["ready_for_exact_eval_dispatch"] is False
    assert wire["cathedral_autopilot_wire_in"]["dispatch_ready"] is False
    assert wire["atom_wire_in"]["atom_kind"] == "meta_lagrangian"
    assert wire["atom_wire_in"]["resolution_path"] == "learned"
    assert "tac.training_curriculum.master_gradient_pair_weights" in wire[
        "master_gradient_wire_in"
    ]["consumer_modules"]
    assert wire[
        "cathedral_autopilot_wire_in"
    ]["promotion_gate"] == "tools/promote_optimizer_candidate_for_exact_eval.py"
