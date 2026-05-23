# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.optimizer_scheduler_registry import (
    DESCRIPTOR_SCHEMA,
    FALSE_AUTHORITY_FIELDS,
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


def test_default_registry_enumerates_hashed_planning_only_candidates() -> None:
    candidates = enumerate_optimizer_scheduler_candidates()

    assert len(candidates) >= 3
    for row in candidates:
        assert row["schema"] == DESCRIPTOR_SCHEMA
        assert len(row["config_sha256"]) == 64
        assert row["rank_score_field"] == "planner_priority_not_score"
        assert row["target_modes"] == ["contest_exact_eval_planning"]
        assert validate_proxy_candidate(row) == []
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


def test_registry_filters_by_declared_target_axis_and_substrate() -> None:
    registry = default_optimizer_scheduler_registry()

    macos_rows = registry.enumerate_candidates(axis_tag="[macOS-CPU advisory]")
    mlx_rows = registry.enumerate_candidates(target_mode="mlx_research_signal")
    other_substrate_rows = registry.enumerate_candidates(substrate="not_registered")

    assert macos_rows
    assert mlx_rows
    assert all("mlx_research_signal" in row["allowed_target_modes"] for row in mlx_rows)
    assert other_substrate_rows == []
    with pytest.raises(OptimizerSchedulerRegistryError, match="unknown target_mode"):
        registry.enumerate_candidates(target_mode="contest_exact_eval")


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
