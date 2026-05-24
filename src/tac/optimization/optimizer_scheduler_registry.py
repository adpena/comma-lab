# SPDX-License-Identifier: MIT
"""Planning-only optimizer/scheduler recipe registry and telemetry schema.

Learned sweep planners need a bounded recipe surface: stable optimizer +
scheduler descriptors with canonical config hashes, not arbitrary ad hoc
configs. This module is intentionally pure metadata. It never builds an
optimizer, dispatches work, or claims score authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from tac.optimization.parameter_group_lr_policy import (
    DEFAULT_PARAMETER_GROUP_LR_POLICY,
    EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    PARAMETER_GROUP_LR_POLICY_SCHEMA,
    parameter_group_lr_policy_sha256,
    validate_parameter_group_lr_policy,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_DISPATCH_BLOCKERS,
    PROXY_FALSE_AUTHORITY_FIELDS,
    PROXY_TARGET_MODES,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.xray.base import CANONICAL_WIRE_IN_HOOKS

REGISTRY_SCHEMA = "optimizer_scheduler_registry.v1"
DESCRIPTOR_SCHEMA = "optimizer_scheduler_descriptor.v1"
TELEMETRY_SCHEMA = "optimizer_scheduler_telemetry.v1"

KNOWN_TARGET_MODES: frozenset[str] = frozenset(
    {
        "local_proxy_learning",
        "macos_cpu_advisory",
        "mlx_research_signal",
        "contest_exact_eval_planning",
    }
)
DEFAULT_ALLOWED_TARGET_MODES: tuple[str, ...] = (
    "local_proxy_learning",
    "macos_cpu_advisory",
    "mlx_research_signal",
    "contest_exact_eval_planning",
)
DEFAULT_ALLOWED_AXIS_TAGS: tuple[str, ...] = (
    "[offline-proxy-planning-only]",
    "[macOS-CPU advisory]",
    "[macOS-MLX research-signal]",
)
FALSE_AUTHORITY_FIELDS: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "score_claim_valid": False,
}
DEFAULT_SCORE_CLAIM_BLOCKERS: tuple[str, ...] = (
    "optimizer_scheduler_recipe_is_planning_only",
    "requires_byte_closed_archive_export_before_dispatch_readiness",
    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
    "requires_exact_auth_eval_result_before_score_claim",
)
DEFAULT_CANONICAL_EQUATION_REFS: tuple[str, ...] = (
    "per_pair_master_gradient_score_impact_taylor_v1",
    "pairset_component_marginal_score_decomposition_v1",
    "canonical_frontier_pointer_v1",
)
DEFAULT_MASTER_GRADIENT_FEATURES: tuple[str, ...] = (
    "per_pair_master_gradient_priority",
    "component_axis_marginal",
    "optimizer_recipe_cost",
)
DEFAULT_PARETO_OBJECTIVES: tuple[str, ...] = (
    "expected_total_score_delta",
    "archive_bytes",
    "state_bytes",
    "seconds_per_candidate",
    "seconds_per_step",
)
PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY = "implemented_mlx_synthetic_timing_only"
PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY = "synthetic_timing_only"
PR95_MLX_SOURCE_FAITHFUL_BLOCKERS: tuple[str, ...] = (
    "pr95_source_video_loader_not_ported_to_mlx",
    "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx",
    "pr95_stage_hparams_and_cosine_schedules_not_all_source_matched",
    "pr95_qat_c1a_and_resume_semantics_not_ported_to_mlx",
    "pr95_export_forward_parity_not_established",
)


class OptimizerSchedulerRegistryError(ValueError):
    """Raised when a recipe or telemetry row blurs authority boundaries."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OptimizerSchedulerRegistryError("non-finite float is not JSON-safe")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_json_safe(v) for v in value]
    if isinstance(value, int | str | bool) or value is None:
        return value
    raise OptimizerSchedulerRegistryError(f"unsupported JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, dict):
        return MappingProxyType({key: _freeze_json(inner) for key, inner in safe.items()})
    if isinstance(safe, list):
        return tuple(_freeze_json(inner) for inner in safe)
    return safe


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(inner) for key, inner in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(inner) for inner in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(payload: Mapping[str, Any]) -> str:
    """Return the byte-stable SHA-256 for an optimizer/scheduler config."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _tuple_of_strings(value: Iterable[Any], *, field_name: str) -> tuple[str, ...]:
    raw_items = (value,) if isinstance(value, str) else value
    items = tuple(
        ordered_unique(str(item).strip() for item in raw_items if str(item).strip())
    )
    if not items:
        raise OptimizerSchedulerRegistryError(f"{field_name} must be non-empty")
    return items


def _validate_target_modes(modes: Sequence[str]) -> tuple[str, ...]:
    normalized = _tuple_of_strings(modes, field_name="allowed_target_modes")
    unknown = [mode for mode in normalized if mode not in KNOWN_TARGET_MODES]
    if unknown:
        raise OptimizerSchedulerRegistryError(
            "unknown allowed_target_modes: " + ", ".join(unknown)
        )
    return normalized


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise OptimizerSchedulerRegistryError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        if key in payload and payload.get(key) is not expected:
            raise OptimizerSchedulerRegistryError(
                f"{label} {key} must be {str(expected).lower()}"
            )


def _solver_stack_wire_in(
    *,
    descriptor_id: str,
    canonical_equation_refs: Sequence[str],
    master_gradient_features: Sequence[str],
    pareto_objectives: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "optimizer_scheduler_solver_stack_wire_in.v1",
        "wire_in_hooks_engaged": list(CANONICAL_WIRE_IN_HOOKS),
        "cathedral_autopilot_wire_in": {
            "consumer": "tools/cathedral_autopilot_autonomous_loop.py",
            "queue_surface": "optimizer_scheduler_registry_candidates",
            "dispatch_ready": False,
            "score_claim": False,
        },
        "meta_lagrangian_wire_in": {
            "atom_kind": "optimizer_scheduler_recipe",
            "candidate_atom_id": f"optimizer_scheduler:{descriptor_id}",
            "rank_signal": "planner_priority_not_score",
        },
        "pareto_wire_in": {
            "objectives": list(pareto_objectives),
            "rank_signal": "telemetry_cost_tradeoff_not_score",
            "score_claim": False,
        },
        "canonical_equation_refs": list(canonical_equation_refs),
        "master_gradient_wire_in": {
            "features": list(master_gradient_features),
            "required_before_promotion": [
                "per_pair_master_gradient_anchor_or_explicit_non_binding_rationale",
                "component_marginal_observation_or_explicit_non_binding_rationale",
            ],
        },
    }


@dataclass(frozen=True)
class OptimizerSchedulerDescriptor:
    """Typed optimizer/scheduler recipe exposed to learned sweep planners."""

    descriptor_id: str
    optimizer: str
    scheduler: str
    optimizer_config: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    training_config: Mapping[str, Any] = field(default_factory=dict)
    parameter_group_lr_policy: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_PARAMETER_GROUP_LR_POLICY)
    )
    substrate: str = "representation_training"
    allowed_target_modes: Sequence[str] = DEFAULT_ALLOWED_TARGET_MODES
    allowed_axis_tags: Sequence[str] = DEFAULT_ALLOWED_AXIS_TAGS
    canonical_equation_refs: Sequence[str] = DEFAULT_CANONICAL_EQUATION_REFS
    master_gradient_features: Sequence[str] = DEFAULT_MASTER_GRADIENT_FEATURES
    pareto_objectives: Sequence[str] = DEFAULT_PARETO_OBJECTIVES
    intended_use: str = "learned_sweep_planning"
    schema: str = DESCRIPTOR_SCHEMA
    config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise OptimizerSchedulerRegistryError("descriptor_id is required")
        if not self.optimizer:
            raise OptimizerSchedulerRegistryError("optimizer is required")
        if not self.scheduler:
            raise OptimizerSchedulerRegistryError("scheduler is required")
        if not self.substrate:
            raise OptimizerSchedulerRegistryError("substrate is required")
        if self.schema != DESCRIPTOR_SCHEMA:
            raise OptimizerSchedulerRegistryError("descriptor schema mismatch")

        optimizer_config = _freeze_json(self.optimizer_config)
        scheduler_config = _freeze_json(self.scheduler_config)
        training_config = _freeze_json(self.training_config)
        parameter_group_lr_policy = _freeze_json(self.parameter_group_lr_policy)
        _require_false_authority(_thaw_json(optimizer_config), label=self.descriptor_id)
        _require_false_authority(_thaw_json(scheduler_config), label=self.descriptor_id)
        _require_false_authority(_thaw_json(training_config), label=self.descriptor_id)
        self._validate_parameter_group_lr_policy(_thaw_json(parameter_group_lr_policy))

        object.__setattr__(self, "optimizer_config", optimizer_config)
        object.__setattr__(self, "scheduler_config", scheduler_config)
        object.__setattr__(self, "training_config", training_config)
        object.__setattr__(
            self,
            "parameter_group_lr_policy",
            parameter_group_lr_policy,
        )
        object.__setattr__(
            self,
            "allowed_target_modes",
            _validate_target_modes(tuple(self.allowed_target_modes)),
        )
        object.__setattr__(
            self,
            "allowed_axis_tags",
            _tuple_of_strings(self.allowed_axis_tags, field_name="allowed_axis_tags"),
        )
        object.__setattr__(
            self,
            "canonical_equation_refs",
            _tuple_of_strings(
                self.canonical_equation_refs,
                field_name="canonical_equation_refs",
            ),
        )
        object.__setattr__(
            self,
            "master_gradient_features",
            _tuple_of_strings(
                self.master_gradient_features,
                field_name="master_gradient_features",
            ),
        )
        object.__setattr__(
            self,
            "pareto_objectives",
            _tuple_of_strings(self.pareto_objectives, field_name="pareto_objectives"),
        )
        object.__setattr__(self, "config_sha256", config_sha256(self.config_payload()))

    def _validate_parameter_group_lr_policy(self, policy: Mapping[str, Any]) -> None:
        try:
            validate_parameter_group_lr_policy(policy)
        except ValueError as exc:
            raise OptimizerSchedulerRegistryError(str(exc)) from exc

    def config_payload(self) -> dict[str, Any]:
        """Return the canonical payload covered by ``config_sha256``."""

        return {
            "optimizer": self.optimizer,
            "optimizer_config": _thaw_json(self.optimizer_config),
            "parameter_group_lr_policy": _thaw_json(self.parameter_group_lr_policy),
            "scheduler": self.scheduler,
            "scheduler_config": _thaw_json(self.scheduler_config),
            "training_config": _thaw_json(self.training_config),
        }

    def to_planner_candidate(self) -> dict[str, Any]:
        """Return a planning-only candidate row for learned sweep planners."""

        solver_wire_in = _solver_stack_wire_in(
            descriptor_id=self.descriptor_id,
            canonical_equation_refs=self.canonical_equation_refs,
            master_gradient_features=self.master_gradient_features,
            pareto_objectives=self.pareto_objectives,
        )
        return {
            "schema": self.schema,
            "descriptor_id": self.descriptor_id,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "optimizer_config": _thaw_json(self.optimizer_config),
            "scheduler_config": _thaw_json(self.scheduler_config),
            "training_config": _thaw_json(self.training_config),
            "parameter_group_lr_policy": _thaw_json(self.parameter_group_lr_policy),
            "parameter_group_lr_policy_id": str(
                _thaw_json(self.parameter_group_lr_policy)["policy_id"]
            ),
            "parameter_group_lr_policy_sha256": parameter_group_lr_policy_sha256(
                _thaw_json(self.parameter_group_lr_policy)
            ),
            "config_sha256": self.config_sha256,
            "substrate": self.substrate,
            "allowed_axis_tags": list(self.allowed_axis_tags),
            "allowed_target_modes": list(self.allowed_target_modes),
            "target_modes": list(PROXY_TARGET_MODES),
            "intended_use": self.intended_use,
            "rank_score_field": "planner_priority_not_score",
            "dispatch_blockers": list(
                ordered_unique([*PROXY_DISPATCH_BLOCKERS, *DEFAULT_SCORE_CLAIM_BLOCKERS])
            ),
            "score_claim_blockers": list(DEFAULT_SCORE_CLAIM_BLOCKERS),
            "false_authority": dict(FALSE_AUTHORITY_FIELDS),
            "solver_stack_wire_in": solver_wire_in,
            "canonical_equation_refs": list(self.canonical_equation_refs),
            "master_gradient_features": list(self.master_gradient_features),
            "pareto_objectives": list(self.pareto_objectives),
            **FALSE_AUTHORITY_FIELDS,
        }


@dataclass(frozen=True)
class OptimizerSchedulerTelemetryRecord:
    """Typed timing/state telemetry for one planning-only recipe run."""

    descriptor_id: str
    config_sha256: str
    axis_tag: str
    substrate: str
    seed: int
    seed_budget: int
    slice_budget: int
    state_bytes: int
    allowed_target_modes: Sequence[str]
    seconds_per_epoch: float | None = None
    seconds_per_candidate: float | None = None
    seconds_per_step: float | None = None
    backend: str | None = None
    kernel_fusion_strategy_id: str | None = None
    backend_kernel_contract: Mapping[str, Any] = field(default_factory=dict)
    operator_mix: Mapping[str, Any] = field(default_factory=dict)
    numerical_drift_profile: Mapping[str, Any] = field(default_factory=dict)
    ineligible_reason: str | None = None
    archive_ready: bool = False
    export_ready: bool = False
    archive_export_blockers: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TELEMETRY_SCHEMA:
            raise OptimizerSchedulerRegistryError("telemetry schema mismatch")
        if not self.descriptor_id:
            raise OptimizerSchedulerRegistryError("descriptor_id is required")
        if len(self.config_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.config_sha256.lower()
        ):
            raise OptimizerSchedulerRegistryError("config_sha256 must be 64 hex chars")
        if not self.axis_tag:
            raise OptimizerSchedulerRegistryError("axis_tag is required")
        if not self.substrate:
            raise OptimizerSchedulerRegistryError("substrate is required")
        for name in ("seed", "seed_budget", "slice_budget", "state_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise OptimizerSchedulerRegistryError(f"{name} must be an integer")
        if self.seed < 0:
            raise OptimizerSchedulerRegistryError("seed must be non-negative")
        if self.seed_budget <= 0:
            raise OptimizerSchedulerRegistryError("seed_budget must be positive")
        if self.slice_budget <= 0:
            raise OptimizerSchedulerRegistryError("slice_budget must be positive")
        if self.state_bytes < 0:
            raise OptimizerSchedulerRegistryError("state_bytes must be non-negative")
        if (
            self.seconds_per_epoch is None
            and self.seconds_per_candidate is None
            and self.seconds_per_step is None
        ):
            raise OptimizerSchedulerRegistryError(
                "seconds_per_epoch, seconds_per_candidate, or seconds_per_step is required"
            )
        for name in ("seconds_per_epoch", "seconds_per_candidate", "seconds_per_step"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise OptimizerSchedulerRegistryError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise OptimizerSchedulerRegistryError(f"{name} must be positive finite")
        if self.backend is not None and not str(self.backend).strip():
            raise OptimizerSchedulerRegistryError("backend must be non-empty when provided")
        object.__setattr__(
            self,
            "allowed_target_modes",
            _validate_target_modes(tuple(self.allowed_target_modes)),
        )
        object.__setattr__(
            self,
            "archive_export_blockers",
            tuple(ordered_unique(self._auto_archive_export_blockers())),
        )
        metadata = _freeze_json(self.metadata)
        _require_false_authority(_thaw_json(metadata), label=self.descriptor_id)
        object.__setattr__(self, "metadata", metadata)
        for name in (
            "backend_kernel_contract",
            "operator_mix",
            "numerical_drift_profile",
        ):
            value = _freeze_json(getattr(self, name))
            _require_false_authority(
                _thaw_json(value),
                label=f"{self.descriptor_id}.{name}",
            )
            object.__setattr__(self, name, value)

    def _auto_archive_export_blockers(self) -> list[str]:
        blockers = [str(item) for item in self.archive_export_blockers if str(item)]
        if not self.archive_ready:
            blockers.append("archive_not_ready")
        if not self.export_ready:
            blockers.append("export_not_ready")
        blockers.extend(DEFAULT_SCORE_CLAIM_BLOCKERS)
        return blockers

    def to_dict(self) -> dict[str, Any]:
        """Return the telemetry row as JSON-safe planning-only metadata."""

        return {
            "schema": self.schema,
            "descriptor_id": self.descriptor_id,
            "config_sha256": self.config_sha256,
            "axis_tag": self.axis_tag,
            "substrate": self.substrate,
            "seed": self.seed,
            "seed_budget": self.seed_budget,
            "slice_budget": self.slice_budget,
            "seconds_per_epoch": self.seconds_per_epoch,
            "seconds_per_candidate": self.seconds_per_candidate,
            "seconds_per_step": self.seconds_per_step,
            "backend": self.backend,
            "kernel_fusion_strategy_id": self.kernel_fusion_strategy_id,
            "backend_kernel_contract": _thaw_json(self.backend_kernel_contract),
            "operator_mix": _thaw_json(self.operator_mix),
            "numerical_drift_profile": _thaw_json(self.numerical_drift_profile),
            "ineligible_reason": self.ineligible_reason,
            "archive_ready": self.archive_ready,
            "export_ready": self.export_ready,
            "archive_export_readiness": {
                "archive_ready": self.archive_ready,
                "export_ready": self.export_ready,
                "ready_for_exact_eval_dispatch": False,
                "blockers": list(self.archive_export_blockers),
            },
            "state_bytes": self.state_bytes,
            "allowed_target_modes": list(self.allowed_target_modes),
            "target_modes": list(PROXY_TARGET_MODES),
            "dispatch_blockers": list(
                ordered_unique([*PROXY_DISPATCH_BLOCKERS, *self.archive_export_blockers])
            ),
            "metadata": _thaw_json(self.metadata),
            "false_authority": dict(FALSE_AUTHORITY_FIELDS),
            **FALSE_AUTHORITY_FIELDS,
        }


@dataclass(frozen=True)
class OptimizerSchedulerRegistry:
    """Immutable registry of approved optimizer/scheduler descriptors."""

    descriptors: Sequence[OptimizerSchedulerDescriptor]
    schema: str = REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REGISTRY_SCHEMA:
            raise OptimizerSchedulerRegistryError("registry schema mismatch")
        descriptors = tuple(self.descriptors)
        seen: set[str] = set()
        for descriptor in descriptors:
            if descriptor.descriptor_id in seen:
                raise OptimizerSchedulerRegistryError(
                    f"duplicate descriptor_id: {descriptor.descriptor_id}"
                )
            seen.add(descriptor.descriptor_id)
        object.__setattr__(self, "descriptors", descriptors)

    def get(self, descriptor_id: str) -> OptimizerSchedulerDescriptor:
        for descriptor in self.descriptors:
            if descriptor.descriptor_id == descriptor_id:
                return descriptor
        raise OptimizerSchedulerRegistryError(f"unknown descriptor_id: {descriptor_id}")

    def enumerate_candidates(
        self,
        *,
        target_mode: str | None = None,
        axis_tag: str | None = None,
        substrate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return planner-consumable candidates filtered by declared support."""

        if target_mode is not None and target_mode not in KNOWN_TARGET_MODES:
            raise OptimizerSchedulerRegistryError(f"unknown target_mode: {target_mode}")
        out: list[dict[str, Any]] = []
        for descriptor in self.descriptors:
            if target_mode is not None and target_mode not in descriptor.allowed_target_modes:
                continue
            if axis_tag is not None and axis_tag not in descriptor.allowed_axis_tags:
                continue
            if substrate is not None and substrate != descriptor.substrate:
                continue
            out.append(descriptor.to_planner_candidate())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "descriptor_count": len(self.descriptors),
            "descriptors": [descriptor.to_planner_candidate() for descriptor in self.descriptors],
        }


def default_optimizer_scheduler_descriptors() -> tuple[OptimizerSchedulerDescriptor, ...]:
    """Return the built-in bounded recipe canvas for learned sweep planners."""

    return (
        OptimizerSchedulerDescriptor(
            descriptor_id="pr95_stage1_adamw_baseline_mlx",
            optimizer="mlx.optimizers.AdamW",
            scheduler="pr95_stage_static_lr",
            optimizer_config={
                "use_muon": False,
                "adamw_lr": 3e-5,
                "latent_lr_mult": 10.0,
                "adamw_betas": [0.9, 0.999],
                "adamw_eps": 1e-8,
                "adamw_weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            scheduler_config={"stage_indices": [1], "source_pr": 95},
            training_config={
                "backend_status": PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY,
                "training_fidelity": PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY,
                "source_faithful_training": False,
                "source_faithfulness_blockers": list(
                    PR95_MLX_SOURCE_FAITHFUL_BLOCKERS
                ),
                "pr95_stage_indices": [1],
                "stage_modules": ["stage1_v328_ce"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="pr95_hnerv_mlx_synthetic_timing_smoke",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="pr95_stage5_adamw_baseline_mlx",
            optimizer="mlx.optimizers.AdamW",
            scheduler="pr95_stage_static_lr",
            optimizer_config={
                "use_muon": False,
                "adamw_lr": 3e-5,
                "latent_lr_mult": 10.0,
                "adamw_betas": [0.9, 0.999],
                "adamw_eps": 1e-8,
                "adamw_weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            scheduler_config={"stage_indices": [5], "source_pr": 95},
            training_config={
                "backend_status": PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY,
                "training_fidelity": PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY,
                "source_faithful_training": False,
                "source_faithfulness_blockers": list(
                    PR95_MLX_SOURCE_FAITHFUL_BLOCKERS
                ),
                "pr95_stage_indices": [5],
                "stage_modules": ["stage5_c1a_l7"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="pr95_hnerv_mlx_synthetic_timing_smoke",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="pr95_stage8_muon_adamw_mlx",
            optimizer="tac.local_acceleration.pr95_hnerv_mlx.Muon+AdamW",
            scheduler="pr95_stage_static_lr",
            optimizer_config={
                "use_muon": True,
                "adamw_lr": 1e-5,
                "latent_lr_mult": 10.0,
                "muon_lr": 2e-4,
                "muon_momentum": 0.95,
                "muon_nesterov": True,
                "muon_ns_steps": 5,
                "muon_weight_decay": 5e-4,
                "adamw_betas": [0.9, 0.999],
                "adamw_eps": 1e-8,
                "adamw_weight_decay": 0.0,
                "grad_clip": 1.0,
                "grad_clip_muon": 1.0,
            },
            scheduler_config={"stage_indices": [8], "source_pr": 95},
            training_config={
                "backend_status": PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY,
                "training_fidelity": PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY,
                "source_faithful_training": False,
                "source_faithfulness_blockers": list(
                    PR95_MLX_SOURCE_FAITHFUL_BLOCKERS
                ),
                "pr95_stage_indices": [8],
                "stage_modules": ["stage8_muon_finetune"],
                "muon_partition": (
                    "hidden 2D+ non-stem/non-rgb weights use Muon; latents, "
                    "stem, RGB heads, biases, and scalar/norm params use AdamW"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="pr95_hnerv_mlx_synthetic_timing_smoke",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="pr95_muon_all_stages_descriptor_only",
            optimizer="Muon+AdamW",
            scheduler="descriptor_only",
            optimizer_config={"backend_status": "optimizer_backend_missing"},
            scheduler_config={"stage_indices": [1, 5, 8]},
            training_config={
                "backend_status": "optimizer_backend_missing",
                "dispatch_blockers": ["optimizer_backend_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="descriptor_only_optimizer_ablation_candidate",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="pr95_langevin_stage8_polish_descriptor_only",
            optimizer="tac.optimization.langevin_optimizer.LangevinOptimizer",
            scheduler="temperature_polish_descriptor_only",
            optimizer_config={"backend_status": "optimizer_backend_missing"},
            scheduler_config={"stage_indices": [8], "temperature_schedule": "tbd"},
            training_config={
                "backend_status": "optimizer_backend_missing",
                "dispatch_blockers": ["optimizer_backend_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="descriptor_only_stage8_polish_candidate",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="adamw_cosine_micro",
            optimizer="torch.optim.AdamW",
            scheduler="cosine_warmup",
            optimizer_config={
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "eps": 1e-8,
            },
            scheduler_config={
                "warmup_fraction": 0.05,
                "min_lr_ratio": 0.1,
                "cycle_count": 1,
            },
            training_config={"gradient_clip_norm": 1.0, "amp": False},
            intended_use="baseline_micro_sweep_recipe",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="muon_adamw_cosine_representation",
            optimizer="tac.optimization.muon.MuonOptimizer+torch.optim.AdamW",
            scheduler="cosine_warmup",
            optimizer_config={
                "muon_momentum": 0.95,
                "muon_ns_steps": 5,
                "adamw_lr_ratio": 0.2,
                "hidden_weight_decay": 0.01,
            },
            scheduler_config={
                "warmup_fraction": 0.08,
                "min_lr_ratio": 0.05,
                "polyak_swa_fraction": 0.15,
            },
            training_config={"full_renderer_required": True, "score_aware_loss_required": True},
            parameter_group_lr_policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
            intended_use="representation_training_smoke_recipe",
        ),
        OptimizerSchedulerDescriptor(
            descriptor_id="info_geom_langevin_plateau_probe",
            optimizer="tac.optimization.info_geom_langevin.InfoGeomLangevinOptimizer",
            scheduler="geman_geman_log_temperature",
            optimizer_config={
                "step_size": 0.0003,
                "fisher_ema": 0.98,
                "noise_scale": 0.03,
            },
            scheduler_config={"temperature_floor": 0.01, "burn_in_fraction": 0.1},
            training_config={"paired_seed_control": True, "exploration_only": True},
            intended_use="plateau_escape_probe_recipe",
        ),
    )


def default_optimizer_scheduler_registry() -> OptimizerSchedulerRegistry:
    """Return the default immutable optimizer/scheduler recipe registry."""

    return OptimizerSchedulerRegistry(default_optimizer_scheduler_descriptors())


def enumerate_optimizer_scheduler_candidates(
    *,
    registry: OptimizerSchedulerRegistry | None = None,
    target_mode: str | None = None,
    axis_tag: str | None = None,
    substrate: str | None = None,
) -> list[dict[str, Any]]:
    """Planner API: enumerate approved recipes without accepting raw configs."""

    active_registry = registry if registry is not None else default_optimizer_scheduler_registry()
    return active_registry.enumerate_candidates(
        target_mode=target_mode,
        axis_tag=axis_tag,
        substrate=substrate,
    )


def build_optimizer_scheduler_telemetry_record(
    *,
    descriptor: OptimizerSchedulerDescriptor,
    axis_tag: str,
    substrate: str | None = None,
    seed: int,
    seed_budget: int = 1,
    slice_budget: int,
    state_bytes: int,
    seconds_per_epoch: float | None = None,
    seconds_per_candidate: float | None = None,
    seconds_per_step: float | None = None,
    backend: str | None = None,
    kernel_fusion_strategy_id: str | None = None,
    backend_kernel_contract: Mapping[str, Any] | None = None,
    operator_mix: Mapping[str, Any] | None = None,
    numerical_drift_profile: Mapping[str, Any] | None = None,
    ineligible_reason: str | None = None,
    archive_ready: bool = False,
    export_ready: bool = False,
    archive_export_blockers: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed telemetry row for one approved descriptor."""

    if axis_tag not in descriptor.allowed_axis_tags:
        raise OptimizerSchedulerRegistryError(
            f"axis_tag {axis_tag!r} is not allowed for {descriptor.descriptor_id}"
        )
    return OptimizerSchedulerTelemetryRecord(
        descriptor_id=descriptor.descriptor_id,
        config_sha256=descriptor.config_sha256,
        axis_tag=axis_tag,
        substrate=substrate or descriptor.substrate,
        seed=seed,
        seed_budget=seed_budget,
        slice_budget=slice_budget,
        state_bytes=state_bytes,
        allowed_target_modes=descriptor.allowed_target_modes,
        seconds_per_epoch=seconds_per_epoch,
        seconds_per_candidate=seconds_per_candidate,
        seconds_per_step=seconds_per_step,
        backend=backend,
        kernel_fusion_strategy_id=kernel_fusion_strategy_id,
        backend_kernel_contract=backend_kernel_contract or {},
        operator_mix=operator_mix or {},
        numerical_drift_profile=numerical_drift_profile or {},
        ineligible_reason=ineligible_reason,
        archive_ready=archive_ready,
        export_ready=export_ready,
        archive_export_blockers=archive_export_blockers,
        metadata=metadata or {},
    ).to_dict()


__all__ = [
    "DEFAULT_ALLOWED_AXIS_TAGS",
    "DEFAULT_ALLOWED_TARGET_MODES",
    "DEFAULT_CANONICAL_EQUATION_REFS",
    "DEFAULT_MASTER_GRADIENT_FEATURES",
    "DEFAULT_PARAMETER_GROUP_LR_POLICY",
    "DEFAULT_PARETO_OBJECTIVES",
    "DEFAULT_SCORE_CLAIM_BLOCKERS",
    "DESCRIPTOR_SCHEMA",
    "EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY",
    "FALSE_AUTHORITY_FIELDS",
    "KNOWN_TARGET_MODES",
    "PARAMETER_GROUP_LR_POLICY_SCHEMA",
    "REGISTRY_SCHEMA",
    "TELEMETRY_SCHEMA",
    "OptimizerSchedulerDescriptor",
    "OptimizerSchedulerRegistry",
    "OptimizerSchedulerRegistryError",
    "OptimizerSchedulerTelemetryRecord",
    "build_optimizer_scheduler_telemetry_record",
    "config_sha256",
    "default_optimizer_scheduler_descriptors",
    "default_optimizer_scheduler_registry",
    "enumerate_optimizer_scheduler_candidates",
]
