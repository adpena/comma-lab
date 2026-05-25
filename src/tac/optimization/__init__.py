# SPDX-License-Identifier: MIT
"""Optimization and allocation utilities for contest archive atoms.

This package ranks, allocates, and composes charged archive atoms. It consumes
analysis records and emits planning policies or ledgers. Exact CUDA auth eval
remains the only score authority.

Exports are lazy so lightweight tools can import submodules such as
``tac.optimization.macos_cpu_advisory_signal`` without importing PyTorch-only
optimizer modules first.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "InfoGeomLangevinConfig": (
        "tac.optimization.info_geom_langevin",
        "InfoGeomLangevinConfig",
    ),
    "DEFAULT_PARAMETER_GROUP_LR_POLICY": (
        "tac.optimization.parameter_group_lr_policy",
        "DEFAULT_PARAMETER_GROUP_LR_POLICY",
    ),
    "EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY": (
        "tac.optimization.parameter_group_lr_policy",
        "EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY",
    ),
    "InfoGeomLangevinOptimizer": (
        "tac.optimization.info_geom_langevin",
        "InfoGeomLangevinOptimizer",
    ),
    "InformationGeometricLangevinOptimizer": (
        "tac.optimization.zen_state_frontier",
        "InformationGeometricLangevinOptimizer",
    ),
    "LangevinOptimizer": ("tac.optimization.langevin_optimizer", "LangevinOptimizer"),
    "MuonOptimizer": ("tac.optimization.muon", "MuonOptimizer"),
    "OperatingPoint": ("tac.optimization.scorer_surface_shaking", "OperatingPoint"),
    "OptimizerSchedulerDescriptor": (
        "tac.optimization.optimizer_scheduler_registry",
        "OptimizerSchedulerDescriptor",
    ),
    "OptimizerSchedulerRegistry": (
        "tac.optimization.optimizer_scheduler_registry",
        "OptimizerSchedulerRegistry",
    ),
    "OptimizerSchedulerTelemetryRecord": (
        "tac.optimization.optimizer_scheduler_registry",
        "OptimizerSchedulerTelemetryRecord",
    ),
    "OptimizerSignalAtomError": (
        "tac.optimization.optimizer_signal_atoms",
        "OptimizerSignalAtomError",
    ),
    "ByteShavingCampaignError": (
        "tac.optimization.byte_shaving_campaign",
        "ByteShavingCampaignError",
    ),
    "LocalTrainingRuntimeProfileError": (
        "tac.optimization.local_training_runtime_profile",
        "LocalTrainingRuntimeProfileError",
    ),
    "ParameterClassification": (
        "tac.optimization.parameter_group_lr_policy",
        "ParameterClassification",
    ),
    "ParameterGroupLRPolicyError": (
        "tac.optimization.parameter_group_lr_policy",
        "ParameterGroupLRPolicyError",
    ),
    "ParameterShapeRecord": (
        "tac.optimization.parameter_group_lr_policy",
        "ParameterShapeRecord",
    ),
    "PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA": (
        "tac.optimization.parameter_group_lr_policy",
        "PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA",
    ),
    "PARAMETER_GROUP_LR_POLICY_SCHEMA": (
        "tac.optimization.parameter_group_lr_policy",
        "PARAMETER_GROUP_LR_POLICY_SCHEMA",
    ),
    "ScorerSurfacePlanError": (
        "tac.optimization.scorer_surface_shaking",
        "ScorerSurfacePlanError",
    ),
    "SurfaceAtomFamily": (
        "tac.optimization.scorer_surface_shaking",
        "SurfaceAtomFamily",
    ),
    "TensorTrain": ("tac.optimization.zen_state_frontier", "TensorTrain"),
    "brenier_quantile_quantize_1d": (
        "tac.optimization.zen_state_frontier",
        "brenier_quantile_quantize_1d",
    ),
    "build_scorer_surface_shaking_plan": (
        "tac.optimization.scorer_surface_shaking",
        "build_scorer_surface_shaking_plan",
    ),
    "build_info_geom_langevin_optimizer": (
        "tac.optimization.info_geom_langevin",
        "build_info_geom_langevin_optimizer",
    ),
    "build_optimizer_scheduler_telemetry_record": (
        "tac.optimization.optimizer_scheduler_registry",
        "build_optimizer_scheduler_telemetry_record",
    ),
    "build_optimizer_signal_atom_ledger": (
        "tac.optimization.optimizer_signal_atoms",
        "build_optimizer_signal_atom_ledger",
    ),
    "dynamic_sparse_skip_mixture": (
        "tac.optimization.dynamic_sparse_gate_oracle",
        "dynamic_sparse_skip_mixture",
    ),
    "operation_set_compiler_hint_from_gate_scores": (
        "tac.optimization.dynamic_sparse_gate_oracle",
        "operation_set_compiler_hint_from_gate_scores",
    ),
    "build_local_training_harvest_intelligence": (
        "tac.optimization.local_training_harvest_intelligence",
        "build_local_training_harvest_intelligence",
    ),
    "build_mlx_effective_spend_triage_learned_sweep_candidates": (
        "tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter",
        "build_mlx_effective_spend_triage_learned_sweep_candidates",
    ),
    "build_observation_rows_from_learned_sweep_plan": (
        "tac.optimization.mlx_dynamic_learned_sweep_observation_harvest",
        "build_observation_rows_from_learned_sweep_plan",
    ),
    "execute_local_mlx_sweep_rows": (
        "tac.optimization.mlx_dynamic_learned_sweep_local_actuator",
        "execute_local_mlx_sweep_rows",
    ),
    "run_local_mlx_sweep_autopilot": (
        "tac.optimization.mlx_dynamic_learned_sweep_local_autopilot",
        "run_local_mlx_sweep_autopilot",
    ),
    "MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA": (
        "tac.optimization.mlx_learned_sweep_batch_roots",
        "MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA",
    ),
    "build_mlx_learned_sweep_autopilot_batch_root_plan": (
        "tac.optimization.mlx_learned_sweep_batch_roots",
        "build_mlx_learned_sweep_autopilot_batch_root_plan",
    ),
    "build_mlx_learned_sweep_next_surface_report": (
        "tac.optimization.mlx_learned_sweep_next_surface",
        "build_mlx_learned_sweep_next_surface_report",
    ),
    "stamp_macos_cpu_advisory_paths": (
        "tac.optimization.mlx_learned_sweep_advisory_handoff",
        "stamp_macos_cpu_advisory_paths",
    ),
    "render_mlx_learned_sweep_next_surface_markdown": (
        "tac.optimization.mlx_learned_sweep_next_surface",
        "render_mlx_learned_sweep_next_surface_markdown",
    ),
    "build_optimizer_scheduler_telemetry_from_harvest_queue": (
        "tac.optimization.local_training_harvest_intelligence",
        "build_optimizer_scheduler_telemetry_from_harvest_queue",
    ),
    "build_byte_shaving_campaign_plan": (
        "tac.optimization.byte_shaving_campaign",
        "build_byte_shaving_campaign_plan",
    ),
    "build_byte_shaving_signal_surface": (
        "tac.optimization.byte_shaving_signal_surface_builder",
        "build_byte_shaving_signal_surface",
    ),
    "observations_from_queue_observation": (
        "tac.optimization.inverse_steganalysis_acquisition",
        "observations_from_queue_observation",
    ),
    "build_signal_surface_from_candidate_queue": (
        "tac.optimization.byte_shaving_campaign",
        "build_signal_surface_from_candidate_queue",
    ),
    "build_signal_surface_from_master_gradient_anchor": (
        "tac.optimization.byte_shaving_campaign",
        "build_signal_surface_from_master_gradient_anchor",
    ),
    "adapt_runtime_profile_observation_to_candidate": (
        "tac.optimization.local_training_runtime_profile",
        "adapt_runtime_profile_observation_to_candidate",
    ),
    "build_atoms_from_optimizer_signal_source": (
        "tac.optimization.optimizer_signal_atoms",
        "build_atoms_from_optimizer_signal_source",
    ),
    "build_parameter_group_lr_policy_fingerprint": (
        "tac.optimization.parameter_group_lr_policy",
        "build_parameter_group_lr_policy_fingerprint",
    ),
    "TernaryCalibration": (
        "tac.optimization.ternary_qat",
        "TernaryCalibration",
    ),
    "TernaryQATConfig": ("tac.optimization.ternary_qat", "TernaryQATConfig"),
    "TernaryQuantizedTensor": (
        "tac.optimization.ternary_qat",
        "TernaryQuantizedTensor",
    ),
    "calibrate_ternary_tensor": (
        "tac.optimization.ternary_qat",
        "calibrate_ternary_tensor",
    ),
    "cosine_temperature_schedule": (
        "tac.optimization.langevin_optimizer",
        "cosine_temperature_schedule",
    ),
    "dequantize_ternary_tensor": (
        "tac.optimization.ternary_qat",
        "dequantize_ternary_tensor",
    ),
    "exponential_temperature_schedule": (
        "tac.optimization.langevin_optimizer",
        "exponential_temperature_schedule",
    ),
    "enumerate_optimizer_scheduler_candidates": (
        "tac.optimization.optimizer_scheduler_registry",
        "enumerate_optimizer_scheduler_candidates",
    ),
    "normalize_runtime_profile_observation": (
        "tac.optimization.local_training_runtime_profile",
        "normalize_runtime_profile_observation",
    ),
    "fisher_diagonal_ema": (
        "tac.optimization.info_geom_langevin",
        "fisher_diagonal_ema",
    ),
    "geman_geman_log_schedule": (
        "tac.optimization.langevin_optimizer",
        "geman_geman_log_schedule",
    ),
    "mps_decompose": ("tac.optimization.zen_state_frontier", "mps_decompose"),
    "mps_reconstruct": ("tac.optimization.zen_state_frontier", "mps_reconstruct"),
    "onsager_importance_weights": (
        "tac.optimization.zen_state_frontier",
        "onsager_importance_weights",
    ),
    "pack_ternary_tensor": ("tac.optimization.ternary_qat", "pack_ternary_tensor"),
    "parameter_group_lr_policy_fingerprint_sha256": (
        "tac.optimization.parameter_group_lr_policy",
        "parameter_group_lr_policy_fingerprint_sha256",
    ),
    "parameter_group_lr_policy_sha256": (
        "tac.optimization.parameter_group_lr_policy",
        "parameter_group_lr_policy_sha256",
    ),
    "partition_params_for_muon": (
        "tac.optimization.muon",
        "partition_params_for_muon",
    ),
    "precondition_gradient": (
        "tac.optimization.info_geom_langevin",
        "precondition_gradient",
    ),
    "quantize_ternary_tensor": (
        "tac.optimization.ternary_qat",
        "quantize_ternary_tensor",
    ),
    "classify_parameter_record": (
        "tac.optimization.parameter_group_lr_policy",
        "classify_parameter_record",
    ),
    "classify_parameter_records": (
        "tac.optimization.parameter_group_lr_policy",
        "classify_parameter_records",
    ),
    "sinkhorn_transport_plan": (
        "tac.optimization.zen_state_frontier",
        "sinkhorn_transport_plan",
    ),
    "runtime_profile_summary_from_training_manifest": (
        "tac.optimization.local_training_runtime_profile",
        "runtime_profile_summary_from_training_manifest",
    ),
    "ternary_ste": ("tac.optimization.ternary_qat", "ternary_ste"),
    "tropical_lora_forward": (
        "tac.optimization.zen_state_frontier",
        "tropical_lora_forward",
    ),
    "unpack_ternary_tensor": (
        "tac.optimization.ternary_qat",
        "unpack_ternary_tensor",
    ),
    "wasserstein_barycenter_diagonal_gaussians": (
        "tac.optimization.zen_state_frontier",
        "wasserstein_barycenter_diagonal_gaussians",
    ),
    "zeropower_via_newtonschulz5": (
        "tac.optimization.muon",
        "zeropower_via_newtonschulz5",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public optimization exports lazily."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
