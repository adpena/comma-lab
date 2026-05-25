# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tac.local_acceleration.pr95_hnerv_mlx_contract import (
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
)
from tac.optimization.optimizer_scheduler_registry import (
    DESCRIPTOR_SCHEMA,
    EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    FALSE_AUTHORITY_FIELDS,
    PARAMETER_GROUP_LR_POLICY_SCHEMA,
    TELEMETRY_SCHEMA,
    OptimizerSchedulerDescriptor,
    OptimizerSchedulerRegistry,
    OptimizerSchedulerRegistryError,
    build_optimizer_scheduler_telemetry_record,
    default_optimizer_scheduler_registry,
    enumerate_optimizer_scheduler_candidates,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.xray.base import CANONICAL_WIRE_IN_HOOKS

REPO_ROOT = Path(__file__).resolve().parents[3]
PR95_CURRICULUM_HELPER = REPO_ROOT / "tools" / "recover_pr95_training_curriculum.py"


def _load_pr95_curriculum_helper():
    spec = importlib.util.spec_from_file_location(
        "recover_pr95_training_curriculum_for_optimizer_registry_tests",
        PR95_CURRICULUM_HELPER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_registry_enumerates_hashed_planning_only_candidates() -> None:
    candidates = enumerate_optimizer_scheduler_candidates()

    assert len(candidates) >= 3
    for row in candidates:
        assert row["schema"] == DESCRIPTOR_SCHEMA
        assert len(row["config_sha256"]) == 64
        assert row["rank_score_field"] == "planner_priority_not_score"
        assert row["target_modes"] == ["contest_exact_eval_planning"]
        assert validate_proxy_candidate(row) == []
        assert row["parameter_group_lr_policy"]["schema"] == PARAMETER_GROUP_LR_POLICY_SCHEMA
        assert row["parameter_group_lr_policy_id"] == row["parameter_group_lr_policy"]["policy_id"]
        assert len(row["parameter_group_lr_policy_sha256"]) == 64
        assert row["false_authority"] == FALSE_AUTHORITY_FIELDS
        for key, expected in FALSE_AUTHORITY_FIELDS.items():
            assert row[key] is expected
        wire = row["solver_stack_wire_in"]
        assert wire["wire_in_hooks_engaged"] == list(CANONICAL_WIRE_IN_HOOKS)
        assert wire["cathedral_autopilot_wire_in"]["dispatch_ready"] is False
        assert wire["cathedral_autopilot_wire_in"]["score_claim"] is False
        assert wire["meta_lagrangian_wire_in"]["rank_signal"] == (
            "planner_priority_not_score"
        )
        assert wire["pareto_wire_in"]["score_claim"] is False
        assert row["canonical_equation_refs"]
        assert row["master_gradient_features"]


def test_descriptor_config_sha256_is_stable_and_config_sensitive() -> None:
    first = OptimizerSchedulerDescriptor(
        descriptor_id="unit_adamw",
        optimizer="torch.optim.AdamW",
        scheduler="constant",
        optimizer_config={"lr": 0.001, "betas": [0.9, 0.999]},
        scheduler_config={"factor": 1.0},
    )
    reordered = OptimizerSchedulerDescriptor(
        descriptor_id="unit_adamw_reordered",
        optimizer="torch.optim.AdamW",
        scheduler="constant",
        optimizer_config={"betas": [0.9, 0.999], "lr": 0.001},
        scheduler_config={"factor": 1.0},
    )
    changed = OptimizerSchedulerDescriptor(
        descriptor_id="unit_adamw_changed",
        optimizer="torch.optim.AdamW",
        scheduler="constant",
        optimizer_config={"lr": 0.002, "betas": [0.9, 0.999]},
        scheduler_config={"factor": 1.0},
    )

    assert first.config_sha256 == reordered.config_sha256
    assert first.config_sha256 != changed.config_sha256

    policy_changed = OptimizerSchedulerDescriptor(
        descriptor_id="unit_adamw_policy_changed",
        optimizer="torch.optim.AdamW",
        scheduler="constant",
        optimizer_config={"lr": 0.001, "betas": [0.9, 0.999]},
        scheduler_config={"factor": 1.0},
        parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    )
    assert first.config_sha256 != policy_changed.config_sha256


def test_registry_filters_by_declared_target_axis_and_substrate() -> None:
    registry = default_optimizer_scheduler_registry()

    macos_rows = registry.enumerate_candidates(axis_tag="[macOS-CPU advisory]")
    mlx_rows = registry.enumerate_candidates(target_mode="mlx_research_signal")
    other_substrate_rows = registry.enumerate_candidates(substrate="not_registered")

    assert macos_rows
    assert mlx_rows
    assert all("mlx_research_signal" in row["allowed_target_modes"] for row in mlx_rows)
    muon_row = next(row for row in mlx_rows if row["descriptor_id"] == "muon_adamw_cosine_representation")
    assert muon_row["parameter_group_lr_policy_id"] == "embedding_theta1_hidden_muon_adamw"
    assert muon_row["parameter_group_lr_policy"]["embedding_lr_scaling_policy"] == (
        "theta_1_not_inverse_width"
    )
    assert "latent" in muon_row["parameter_group_lr_policy"]["embedding_param_patterns"]
    assert "arxiv:2605.21486" in muon_row["parameter_group_lr_policy"]["source_refs"]
    assert other_substrate_rows == []
    with pytest.raises(OptimizerSchedulerRegistryError, match="unknown target_mode"):
        registry.enumerate_candidates(target_mode="contest_exact_eval")


def test_registry_exposes_pr95_mlx_optimizer_descriptors_fail_closed() -> None:
    registry = default_optimizer_scheduler_registry()
    stage1 = registry.get("pr95_stage1_adamw_baseline_mlx").to_planner_candidate()
    stage2 = registry.get("pr95_stage2_adamw_baseline_mlx").to_planner_candidate()
    stage8 = registry.get("pr95_stage8_muon_adamw_mlx").to_planner_candidate()
    descriptor_only = registry.get(
        "pr95_langevin_stage8_polish_descriptor_only"
    ).to_planner_candidate()

    assert stage1["optimizer_config"]["use_muon"] is False
    assert stage1["optimizer_config"]["adamw_lr"] == 1e-3
    assert stage1["training_config"]["pr95_stage_indices"] == [1]
    assert stage2["optimizer_config"]["use_muon"] is False
    assert stage2["optimizer_config"]["adamw_lr"] == 1e-3
    assert stage2["training_config"]["pr95_stage_indices"] == [2]
    assert stage2["training_config"]["stage_modules"] == ["stage2_v331_softplus"]
    assert stage2["training_config"]["stage_loss_family"] == "tau_softplus_seg_loss"
    assert stage8["optimizer_config"]["use_muon"] is True
    assert stage8["training_config"]["pr95_stage_indices"] == [8]
    assert stage8["parameter_group_lr_policy_id"] == (
        "embedding_theta1_hidden_muon_adamw"
    )
    assert stage8["training_config"]["backend_status"] == (
        "implemented_mlx_local_timing_proxy"
    )
    assert "source_video_rgb_yuv6_preprocess_coupled_timing_only" in (
        stage8["training_config"]["supported_training_fidelities"]
    )
    assert "rgb_yuv6_mse" in stage8["training_config"]["supported_loss_surfaces"]
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in (
        stage8["training_config"]["source_faithfulness_blockers"]
    )
    assert PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER in stage8["training_config"][
        "source_faithfulness_blockers"
    ]
    assert descriptor_only["training_config"]["backend_status"] == (
        "optimizer_backend_missing"
    )
    assert "optimizer_backend_missing" in descriptor_only["training_config"][
        "dispatch_blockers"
    ]
    for row in (stage1, stage2, stage8, descriptor_only):
        assert validate_proxy_candidate(row) == []
        for key, expected in FALSE_AUTHORITY_FIELDS.items():
            assert row[key] is expected


def test_pr95_mlx_stage_descriptors_match_recovered_public_curriculum() -> None:
    helper = _load_pr95_curriculum_helper()
    source = helper.recover_curriculum(helper.DEFAULT_SOURCE_DIR)
    stages_by_order = {int(row["order"]): row for row in source["stages"]}
    registry = default_optimizer_scheduler_registry()
    descriptor_by_stage = {
        1: "pr95_stage1_adamw_baseline_mlx",
        2: "pr95_stage2_adamw_baseline_mlx",
        5: "pr95_stage5_adamw_baseline_mlx",
        8: "pr95_stage8_muon_adamw_mlx",
    }

    for stage_index, descriptor_id in descriptor_by_stage.items():
        source_stage = stages_by_order[stage_index]
        row = registry.get(descriptor_id).to_planner_candidate()
        training = row["training_config"]
        optimizer = row["optimizer_config"]

        assert training["stage_modules"] == [source_stage["name"]]
        assert training["stage_epochs"] == source_stage["epochs"]
        assert training["stage_loss_family"] == source_stage["loss_family"]
        assert training["stage_cat_lambda"] == source_stage["cat_lambda"]
        assert training["stage_cat_sigma"] == source_stage["cat_sigma"]
        assert training["stage_uses_qat"] is source_stage["uses_qat"]
        assert training["stage_uses_muon"] is source_stage["uses_muon"]
        assert optimizer["use_muon"] is source_stage["uses_muon"]
        assert optimizer["adamw_lr"] == source_stage["adamw_lr"]
        if source_stage["muon_lr"] is not None:
            assert optimizer["muon_lr"] == source_stage["muon_lr"]
        if source_stage["muon_weight_decay"] is not None:
            assert optimizer["muon_weight_decay"] == source_stage["muon_weight_decay"]


def test_pr95_mlx_stage_descriptors_reject_stale_stage2_v328_ce_lore() -> None:
    registry = default_optimizer_scheduler_registry()
    payload = registry.to_dict()
    pr95_rows = [
        row
        for row in payload["descriptors"]
        if str(row["descriptor_id"]).startswith("pr95_stage")
    ]

    assert "stage2_v328_ce" not in str(pr95_rows)


def test_registry_rejects_duplicate_descriptors_and_unknown_lookup() -> None:
    descriptor = default_optimizer_scheduler_registry().descriptors[0]

    with pytest.raises(OptimizerSchedulerRegistryError, match="duplicate descriptor_id"):
        OptimizerSchedulerRegistry([descriptor, descriptor])

    with pytest.raises(OptimizerSchedulerRegistryError, match="unknown descriptor_id"):
        default_optimizer_scheduler_registry().get("missing")


def test_telemetry_record_contains_required_timing_state_and_authority_fields() -> None:
    descriptor = default_optimizer_scheduler_registry().get("adamw_cosine_micro")

    row = build_optimizer_scheduler_telemetry_record(
        descriptor=descriptor,
        axis_tag="[macOS-CPU advisory]",
        seed=123,
        seed_budget=2,
        slice_budget=16,
        seconds_per_candidate=0.75,
        seconds_per_step=0.25,
        backend="mlx",
        kernel_fusion_strategy_id="measured_mlx_conv_profile",
        backend_kernel_contract={"backend": "mlx", "score_claim": False},
        operator_mix={"conv2d": 0.7, "gemm": 0.2},
        numerical_drift_profile={"max_abs_delta": 1e-5, "score_claim": False},
        ineligible_reason="coda_cuda_hopper_path_not_applicable_to_mlx",
        state_bytes=4096,
        archive_ready=True,
        export_ready=False,
        archive_export_blockers=["export_adapter_pending"],
        metadata={"profile_id": "unit_profile"},
    )

    assert row["schema"] == TELEMETRY_SCHEMA
    assert row["descriptor_id"] == descriptor.descriptor_id
    assert row["config_sha256"] == descriptor.config_sha256
    assert row["axis_tag"] == "[macOS-CPU advisory]"
    assert row["substrate"] == descriptor.substrate
    assert row["seed"] == 123
    assert row["seed_budget"] == 2
    assert row["slice_budget"] == 16
    assert row["seconds_per_epoch"] is None
    assert row["seconds_per_candidate"] == 0.75
    assert row["seconds_per_step"] == 0.25
    assert row["backend"] == "mlx"
    assert row["kernel_fusion_strategy_id"] == "measured_mlx_conv_profile"
    assert row["operator_mix"]["conv2d"] == 0.7
    assert row["numerical_drift_profile"]["max_abs_delta"] == 1e-5
    assert row["ineligible_reason"] == "coda_cuda_hopper_path_not_applicable_to_mlx"
    assert row["state_bytes"] == 4096
    assert row["archive_ready"] is True
    assert row["export_ready"] is False
    assert row["archive_export_readiness"]["ready_for_exact_eval_dispatch"] is False
    assert "export_not_ready" in row["archive_export_readiness"]["blockers"]
    assert "contest_exact_eval_planning" in row["allowed_target_modes"]
    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_telemetry_fails_closed_without_timing_or_with_truthy_authority() -> None:
    descriptor = default_optimizer_scheduler_registry().get("adamw_cosine_micro")

    with pytest.raises(OptimizerSchedulerRegistryError, match="seconds_per_epoch"):
        build_optimizer_scheduler_telemetry_record(
            descriptor=descriptor,
            axis_tag="[macOS-CPU advisory]",
            seed=1,
            slice_budget=1,
            state_bytes=0,
        )

    with pytest.raises(
        OptimizerSchedulerRegistryError,
        match="forbidden truthy authority fields",
    ):
        build_optimizer_scheduler_telemetry_record(
            descriptor=descriptor,
            axis_tag="[macOS-CPU advisory]",
            seed=1,
            slice_budget=1,
            seconds_per_epoch=0.5,
            state_bytes=0,
            metadata={"nested": {"score_claim": True}},
        )

    with pytest.raises(OptimizerSchedulerRegistryError, match="not allowed"):
        build_optimizer_scheduler_telemetry_record(
            descriptor=descriptor,
            axis_tag="[contest-CUDA]",
            seed=1,
            slice_budget=1,
            seconds_per_epoch=0.5,
            state_bytes=0,
        )

    with pytest.raises(
        OptimizerSchedulerRegistryError,
        match=r"backend_kernel_contract.*score_claim",
    ):
        build_optimizer_scheduler_telemetry_record(
            descriptor=descriptor,
            axis_tag="[macOS-CPU advisory]",
            seed=1,
            slice_budget=1,
            seconds_per_epoch=0.5,
            state_bytes=0,
            backend_kernel_contract={"score_claim": True},
        )


def test_descriptor_rejects_authority_and_unknown_target_modes() -> None:
    with pytest.raises(
        OptimizerSchedulerRegistryError,
        match="forbidden truthy authority fields",
    ):
        OptimizerSchedulerDescriptor(
            descriptor_id="unsafe",
            optimizer="torch.optim.AdamW",
            scheduler="constant",
            optimizer_config={"lr": 0.001, "score_claim": True},
            scheduler_config={},
        )

    with pytest.raises(OptimizerSchedulerRegistryError, match="unknown allowed_target_modes"):
        OptimizerSchedulerDescriptor(
            descriptor_id="bad_target",
            optimizer="torch.optim.AdamW",
            scheduler="constant",
            optimizer_config={"lr": 0.001},
            scheduler_config={},
            allowed_target_modes=("contest_exact_eval",),
        )

    with pytest.raises(
        OptimizerSchedulerRegistryError,
        match=r"parameter_group_lr_policy.*score_claim",
    ):
        OptimizerSchedulerDescriptor(
            descriptor_id="unsafe_policy",
            optimizer="torch.optim.AdamW",
            scheduler="constant",
            optimizer_config={"lr": 0.001},
            scheduler_config={},
            parameter_group_lr_policy={
                **EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
                "score_claim": True,
            },
        )
