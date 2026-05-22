# SPDX-License-Identifier: MIT
"""Wire representation-training proxy rows into the solver stack.

Optimizer and training smokes are useful only if they become structured
signal for the same consumers that rank byte candidates: master-gradient/
X-ray, Pareto, bit allocation, cathedral autopilot, continual learning, and
probe disambiguation. This bridge emits that substrate-agnostic contract while
preserving the proxy evidence boundary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.xray.base import CANONICAL_WIRE_IN_HOOKS

OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA = "optimizer_training_signal_wire_in_v1"

DEFAULT_CANONICAL_EQUATION_REFS: tuple[str, ...] = (
    "per_pair_master_gradient_score_impact_taylor_v1",
    "master_gradient_locality_violation_by_codec_v1",
    "canonical_frontier_pointer_v1",
    "pairset_component_marginal_score_decomposition_v1",
)

DEFAULT_MASTER_GRADIENT_FEATURES: tuple[str, ...] = (
    "pairset_component_marginal",
    "hard_pair_indices",
    "segnet_posenet_axis_dominance",
    "byte_locality_class",
    "rate_component_marginal",
)

DEFAULT_XRAY_PRIMITIVES: tuple[str, ...] = (
    "pairset_component_marginal",
    "per_pair_score_decomposition",
    "unified_action_principle",
    "score_lipschitz",
    "segnet_margin_polytope",
    "posenet_se3_lie_algebra",
)

DEFAULT_DETERMINISTIC_SOLUTION_REFS: tuple[str, ...] = (
    "tac.packet_compiler.deterministic_compiler",
    "tac.training.runtime_hash_identity",
    "tac.canonical_equations.scorer_input_cache_hash_identity",
)

DEFAULT_VARIANT_AXES: tuple[str, ...] = (
    "source_faithful_control",
    "optimizer_recipe",
    "scheduler_recipe",
    "normalization_or_weight_decay",
    "training_curriculum",
    "representation_substrate",
    "archive_export",
)

DEFAULT_PAIRED_MODES: tuple[str, ...] = (
    "source_faithful_control",
    "optimizer_variant",
    "scheduler_variant",
    "normalization_or_weight_decay_variant",
    "substrate_variant",
    "export_variant",
)


class OptimizerTrainingSignalBridgeError(ValueError):
    """Raised when an optimizer signal would be orphaned or over-authoritative."""


def ordered_unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OptimizerTrainingSignalBridgeError("non-finite float in optimizer signal")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_json_safe(v) for v in value]
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _false_authority_block() -> dict[str, bool]:
    return dict(PROXY_FALSE_AUTHORITY_FIELDS)


def build_optimizer_training_signal_wire_in(
    *,
    candidate_id: str,
    profile_id: str,
    lane_id: str,
    lane_class: str,
    candidate_family: str,
    param_schema: str,
    candidate_params: Mapping[str, Any],
    source_anchor: str | None,
    score_lowering_hypothesis: str,
    dispatch_blockers: Sequence[str],
    representation_family: str | None = None,
    substrate_family: str | None = None,
    training_signal_kind: str = "representation_training_proxy",
    variant_axes: Sequence[str] = DEFAULT_VARIANT_AXES,
    paired_modes: Sequence[str] = DEFAULT_PAIRED_MODES,
    canonical_equation_refs: Sequence[str] = DEFAULT_CANONICAL_EQUATION_REFS,
    master_gradient_features: Sequence[str] = DEFAULT_MASTER_GRADIENT_FEATURES,
    xray_primitives: Sequence[str] = DEFAULT_XRAY_PRIMITIVES,
    deterministic_solution_refs: Sequence[str] = DEFAULT_DETERMINISTIC_SOLUTION_REFS,
) -> dict[str, Any]:
    """Return a planning-only solver-stack handoff for one training row."""

    if not candidate_id:
        raise OptimizerTrainingSignalBridgeError("candidate_id is required")
    if not profile_id:
        raise OptimizerTrainingSignalBridgeError("profile_id is required")
    if not lane_id:
        raise OptimizerTrainingSignalBridgeError("lane_id is required")
    if not candidate_family:
        raise OptimizerTrainingSignalBridgeError("candidate_family is required")
    if not training_signal_kind:
        raise OptimizerTrainingSignalBridgeError("training_signal_kind is required")

    hooks = tuple(CANONICAL_WIRE_IN_HOOKS)
    canonical_equations = ordered_unique(canonical_equation_refs)
    master_gradient = ordered_unique(master_gradient_features)
    xray = ordered_unique(xray_primitives)
    deterministic = ordered_unique(deterministic_solution_refs)
    blockers = ordered_unique(dispatch_blockers)
    variant_axis_list = ordered_unique(variant_axes)
    paired_mode_list = ordered_unique(paired_modes)

    if len(hooks) != 6:
        raise OptimizerTrainingSignalBridgeError("all six canonical hooks must be declared")
    if not canonical_equations:
        raise OptimizerTrainingSignalBridgeError("canonical_equation_refs must be non-empty")
    if not master_gradient:
        raise OptimizerTrainingSignalBridgeError("master_gradient_features must be non-empty")
    if not xray:
        raise OptimizerTrainingSignalBridgeError("xray_primitives must be non-empty")
    if not variant_axis_list:
        raise OptimizerTrainingSignalBridgeError("variant_axes must be non-empty")
    if not paired_mode_list:
        raise OptimizerTrainingSignalBridgeError("paired_modes must be non-empty")

    payload = {
        "schema": OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA,
        "candidate_id": candidate_id,
        "profile_id": profile_id,
        "lane_id": lane_id,
        "lane_class": lane_class,
        "candidate_family": candidate_family,
        "representation_family": representation_family or "unspecified",
        "substrate_family": substrate_family or representation_family or "unspecified",
        "training_signal_kind": training_signal_kind,
        "variant_axes": variant_axis_list,
        "param_schema": param_schema,
        "candidate_params": dict(candidate_params),
        "source_anchor": source_anchor,
        "score_lowering_hypothesis": score_lowering_hypothesis,
        "evidence_semantics": "optimizer_training_proxy_signal_not_exact_auth_eval",
        "evidence_grade": "[offline-proxy-planning-only]",
        "false_authority": _false_authority_block(),
        "wire_in_hooks_engaged": list(hooks),
        "canonical_hook_aliases": {
            "atom_cathedral_hook": "cathedral_autopilot_dispatch",
            "atom_continual_learning_hook": "continual_learning_posterior",
        },
        "master_gradient_wire_in": {
            "consumer_modules": [
                "tac.master_gradient_consumers",
                "tac.master_gradient_pr101_score_response_matrix",
                "tac.training_curriculum.master_gradient_pair_weights",
                "tools/master_gradient_xray.py",
            ],
            "features": master_gradient,
            "required_before_exact_promotion": [
                "per_pair_master_gradient_anchor_or_explicit_non_binding_rationale",
                "component_marginal_observation_or_explicit_non_binding_rationale",
                "archive_runtime_identity_before_empirical_anchor",
            ],
        },
        "pareto_wire_in": {
            "axes": ["rate", "segnet", "posenet"],
            "constraints": [
                "common_seed_and_training_budget_for_paired_smokes",
                "same_archive_export_contract_before_comparing_variants",
                "contest_axis_component_deltas_required_before_promotion",
            ],
            "rank_signal": "proxy_objective_not_score",
            "scope": "representation_substrate_agnostic",
        },
        "bit_allocator_wire_in": {
            "charged_bits_status": "unchanged_until_archive_materialization",
            "allocator_features": [
                "rate_component_marginal",
                "optimizer_recipe_cost",
                "representation_recipe_cost",
                "pair_weight_schedule",
            ],
        },
        "cathedral_autopilot_wire_in": {
            "queue_schema": "optimizer_candidate_queue_v1",
            "adapter": "tools/build_optimizer_candidate_queue.py",
            "dispatch_ready": False,
            "promotion_gate": "tools/promote_optimizer_candidate_for_exact_eval.py",
            "dispatch_blockers": blockers,
        },
        "continual_learning_wire_in": {
            "observation_surface": "tac.optimization.mlx_dynamic_sweep_observations",
            "compatible_observation_surface": (
                "representation_training_probe_manifest_v1"
            ),
            "posterior_update_policy": "append_only_after_empirical_anchor",
            "score_claim": False,
        },
        "probe_disambiguator_wire_in": {
            "paired_modes": paired_mode_list,
            "arbitration_rule": "paired_budget_same_seed_then_exact_auth_gate",
        },
        "xray_wire_in": {
            "primitives": xray,
            "required_review_surfaces": [
                "pairset_component_marginal",
                "segnet_margin_polytope",
                "posenet_se3_lie_algebra",
                "score_lipschitz",
            ],
        },
        "canonical_equation_refs": canonical_equations,
        "atom_wire_in": {
            "atom_kind": "meta_lagrangian",
            "resolution_path": "learned",
            "candidate_atom_id": f"optimizer_training:{candidate_id}",
            "canonical_helper_repo_link": "src/tac/optimization/optimizer_training_signal_bridge.py",
            "wired_hooks": [
                "sensitivity_map",
                "pareto_constraint",
                "bit_allocator",
                "cathedral_autopilot_dispatch",
                "continual_learning_posterior",
                "probe_disambiguator",
            ],
        },
        "deterministic_solution_wire_in": {
            "references": deterministic,
            "requirements": [
                "deterministic_seed_recorded",
                "optimizer_config_sha_recorded",
                "training_recipe_config_sha_recorded",
                "trainer_runtime_sha_recorded",
                "archive_export_is_byte_closed_before_exact_eval",
            ],
        },
    }
    validate_optimizer_training_signal_wire_in(payload)
    return _json_safe(payload)


def validate_optimizer_training_signal_wire_in(payload: Mapping[str, Any]) -> list[str]:
    """Return violations; raise-free so queue adapters can audit rows."""

    violations: list[str] = []
    if payload.get("schema") != OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA:
        violations.append("schema_mismatch")
    hooks = payload.get("wire_in_hooks_engaged")
    if hooks != list(CANONICAL_WIRE_IN_HOOKS):
        violations.append("wire_in_hooks_must_cover_all_six")
    false_authority = payload.get("false_authority")
    if not isinstance(false_authority, Mapping):
        violations.append("false_authority_missing")
    else:
        for key, expected in PROXY_FALSE_AUTHORITY_FIELDS.items():
            if false_authority.get(key) is not expected:
                violations.append(f"false_authority.{key}_must_be_{str(expected).lower()}")
    for key in (
        "master_gradient_wire_in",
        "pareto_wire_in",
        "bit_allocator_wire_in",
        "cathedral_autopilot_wire_in",
        "continual_learning_wire_in",
        "probe_disambiguator_wire_in",
        "xray_wire_in",
        "atom_wire_in",
        "deterministic_solution_wire_in",
    ):
        value = payload.get(key)
        if not isinstance(value, Mapping) or not value:
            violations.append(f"{key}_missing")
    if not payload.get("canonical_equation_refs"):
        violations.append("canonical_equation_refs_missing")
    if not payload.get("variant_axes"):
        violations.append("variant_axes_missing")
    probe = payload.get("probe_disambiguator_wire_in")
    if isinstance(probe, Mapping) and not probe.get("paired_modes"):
        violations.append("probe_disambiguator_wire_in.paired_modes_missing")
    if violations:
        return violations
    return []


__all__ = [
    "DEFAULT_CANONICAL_EQUATION_REFS",
    "DEFAULT_DETERMINISTIC_SOLUTION_REFS",
    "DEFAULT_MASTER_GRADIENT_FEATURES",
    "DEFAULT_PAIRED_MODES",
    "DEFAULT_VARIANT_AXES",
    "DEFAULT_XRAY_PRIMITIVES",
    "OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA",
    "OptimizerTrainingSignalBridgeError",
    "build_optimizer_training_signal_wire_in",
    "ordered_unique",
    "validate_optimizer_training_signal_wire_in",
]
