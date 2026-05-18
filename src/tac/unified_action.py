# SPDX-License-Identifier: MIT
"""GR-style unified action principle for the contest pipeline (T7-T22 + Lane-12-v2 + A1).

Per ``feedback_unified_lagrangian_action_principle_GR_style_20260509`` the
council's eureka was that the contest objective decomposes as a single scalar
**action**

.. math::

    S_{\\text{total}}(\\theta;\\,b,\\,h) = \\int \\big(\\,
        \\mathcal{L}_{\\text{seg}}
      + \\mathcal{L}_{\\text{pose}}
      + \\mathcal{L}_{\\text{rate}}
      + \\sum_k \\mathcal{L}_{\\text{regularizer}}^{(k)}
      \\big)\\,dt

with each ``L`` carrying explicit Lagrange multipliers (dual variables). The
individual tracks landed in 2026-05-09's lateral-leap session each correspond to
ONE named contribution to the action — there is no separate "loss function" per
track; the existing trainers already realize variational solutions to subsets of
this action. The unified module makes the decomposition first-class and
solver-actionable.

This module is a **scaffold** — existing trainers (Phase 1 Ballé hyperprior,
Lane 12-v2 NeRV-as-renderer, etc.) keep working as-is; ``Action`` is the
migration target. Each per-track field is OPTIONAL and `None`-by-default so a
trainer can opt into one contribution at a time.

The migration path is documented as an `Action.migration_status()` table.

Cross-references
----------------

- ``feedback_unified_lagrangian_action_principle_GR_style_20260509.md`` — the meta-principle
- ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md`` — T11/T13/T19 landings
- ``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`` — non-arbitrariness
- ``tac.optimizer.meta_lagrangian.MetaLagrangianSearch`` — existing planning-only Lagrangian search
- ``tac.joint_admm_coordinator.run_admm`` — existing ADMM coordinator
- ``tac.score_geometry_shannon_floor`` — T7 closed-form floor

CLAUDE.md compliance
--------------------

- Per "Meta-Lagrangian/Pareto solver" non-negotiable: every track is a typed
  field on ``Action`` so the planner can consume it as a row.
- Per "Forbidden score claims": this module never claims a score. It returns a
  scalar ``S_total`` tagged ``[predicted; unified-action; closed-form
  weighted-sum]``. Authoritative scores come from ``upstream/evaluate.py`` on
  exact archive bytes.
- Per CLAUDE.md "FORBIDDEN device-selection defaults": no MPS fallback. Caller
  passes ``device`` explicitly; defaults to CPU for the smoke/test path only.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from typing import Any

import torch

ACTION_SCHEMA_VERSION = "tac_unified_action_v1"
ACTION_EVIDENCE_GRADE = "[predicted; unified-action; closed-form weighted-sum]"
OPTIMIZER_ANALYTICAL_BOUNDARIES_SCHEMA_VERSION = (
    "tac_optimizer_analytical_boundaries_v1"
)
OPTIMIZER_ANALYTICAL_BOUNDARIES_EVIDENCE_GRADE = (
    "[predicted; unified-action; analytical-boundary-bundle]"
)


class TrackKind(StrEnum):
    """Canonical track identifiers landed via the lateral-leap session."""

    SEG_BASELINE = "seg_baseline"  # 100 * argmax-disagreement (contest scorer canonical)
    POSE_BASELINE = "pose_baseline"  # sqrt(10 * pose_avg) (contest scorer canonical)
    RATE_BASELINE = "rate_baseline"  # 25 * archive_bytes / PR106_TOTAL_RATE_DENOM
    T7_FISHER_RAO = "t7_fisher_rao"  # Riemannian distance (seg-axis refinement)
    T8_SINKHORN_W2 = "t8_sinkhorn_w2"  # Wasserstein-2 (seg-axis refinement)
    T11_LOVASZ_HINGE = "t11_lovasz_hinge"  # Convex IoU envelope (seg-axis)
    T13_JOINT_SOURCE_RD = "t13_joint_source_rd"  # Rate-axis floor refinement
    T19_ADAPTIVE_RHO = "t19_adaptive_rho"  # Numerical-solver step (no L contribution; consumed by ADMM)
    T20_KL_POSE_DISTILL = "t20_kl_pose_distill"  # Pose-axis training-time refinement
    T22_TEMPORAL_CONSISTENCY = "t22_temporal_consistency"  # Temporal smoothness regularizer
    LANE_12_V2 = "lane_12_v2_nerv_as_renderer"  # Architecture-class atom


class SurfaceKind(StrEnum):
    """Framework-agnostic analytical surfaces feeding deterministic solvers."""

    BOUNDARY = "boundary"
    MASTER_GRADIENT_BOUNDARY = "master_gradient_boundary"
    HARD_PAIR = "hard_pair"
    SENSITIVE_BYTE = "sensitive_byte"
    XRAY_PRIMITIVE = "xray_primitive"
    SENSITIVITY_MAP = "sensitivity_map"
    VENN_CLASS = "venn_class"


@dataclass(frozen=True)
class DualVariables:
    """Per-track dual variables (Lagrange multipliers) λ_k.

    Defaults are ``1.0`` (uniform Boyd-style equal-weight init); a trainer
    that stacks tracks should adapt these via ADMM (see
    :class:`tac.joint_admm_coordinator`) or fix them per the council
    deliberation.
    """

    lambda_seg: float = 1.0
    lambda_pose: float = 1.0
    lambda_rate: float = 1.0
    lambda_t7: float = 0.0  # Off by default; activate per-experiment
    lambda_t8: float = 0.0
    lambda_t11: float = 0.0
    lambda_t13: float = 0.0  # T13 is a CONSTRAINT not a soft term; usually unused
    lambda_t20: float = 0.0
    lambda_t22: float = 0.0


@dataclass(frozen=True)
class MasterGradientBoundarySummary:
    """Planning-only boundary summary derived from a per-pair master gradient.

    The summary is the deterministic-optimizer handoff layer for the operator's
    "boundaries / hard pairs / sensitive bytes" directive. It never carries
    score, promotion, or dispatch authority; exact contest eval on emitted bytes
    is still required before any score claim.
    """

    archive_sha256: str
    n_bytes: int
    n_pairs: int
    n_axes: int
    sign_flip_byte_indices: tuple[int, ...]
    magnitude_cliff_byte_indices: tuple[int, ...]
    hard_pair_indices: tuple[int, ...]
    sensitive_byte_indices: tuple[int, ...]
    mean_seg_pose_cosine: float
    min_seg_pose_cosine: float
    max_seg_pose_cosine: float
    magnitude_cliff_ratio: float
    sensitive_byte_fraction: float
    evidence_grade: str = "[predicted; unified-action; master-gradient-boundary]"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    dispatch_packet_ready: bool = False

    def __post_init__(self) -> None:
        if (
            self.score_claim
            or self.promotion_eligible
            or self.ready_for_exact_eval_dispatch
            or self.rank_or_kill_eligible
            or self.dispatch_packet_ready
        ):
            raise ValueError(
                "MasterGradientBoundarySummary is planning-only and cannot carry "
                "score/promotion/rank/dispatch authority"
            )
        if self.n_bytes < 0 or self.n_pairs < 0 or self.n_axes < 0:
            raise ValueError("n_bytes, n_pairs, and n_axes must be non-negative")
        for name in (
            "mean_seg_pose_cosine",
            "min_seg_pose_cosine",
            "max_seg_pose_cosine",
            "magnitude_cliff_ratio",
            "sensitive_byte_fraction",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe payload for ledgers, xray tools, and autopilot notes."""
        return {
            "schema": "master_gradient_boundary_summary_v1",
            "surface_kind": SurfaceKind.MASTER_GRADIENT_BOUNDARY.value,
            "archive_sha256": self.archive_sha256,
            "n_bytes": self.n_bytes,
            "n_pairs": self.n_pairs,
            "n_axes": self.n_axes,
            "sign_flip_byte_indices": list(self.sign_flip_byte_indices),
            "magnitude_cliff_byte_indices": list(self.magnitude_cliff_byte_indices),
            "hard_pair_indices": list(self.hard_pair_indices),
            "sensitive_byte_indices": list(self.sensitive_byte_indices),
            "mean_seg_pose_cosine": self.mean_seg_pose_cosine,
            "min_seg_pose_cosine": self.min_seg_pose_cosine,
            "max_seg_pose_cosine": self.max_seg_pose_cosine,
            "magnitude_cliff_ratio": self.magnitude_cliff_ratio,
            "sensitive_byte_fraction": self.sensitive_byte_fraction,
            "evidence_grade": self.evidence_grade,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "rank_or_kill_eligible": self.rank_or_kill_eligible,
            "dispatch_packet_ready": self.dispatch_packet_ready,
        }


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for planning bundles."""
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_safe(value.as_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_json_safe(v) for v in value]
    if isinstance(value, set | frozenset):
        return [_json_safe(v) for v in sorted(value, key=str)]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float cannot be serialized: {value!r}")
        return value
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _authority_flags_are_false(surface: Mapping[str, Any], *, surface_name: str) -> None:
    """Reject score/promotion/dispatch/rank authority in optimizer inputs."""
    for flag in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_packet_ready",
        "rank_or_kill_eligible",
    ):
        if surface.get(flag) is True:
            raise ValueError(
                f"{surface_name} carries {flag}=True; optimizer analytical "
                "boundaries are planning-only"
            )


def _axis_weights_payload(weights: Any) -> dict[str, Any]:
    if hasattr(weights, "as_dict") and callable(weights.as_dict):
        payload = weights.as_dict()
    elif is_dataclass(weights) and not isinstance(weights, type):
        payload = asdict(weights)
    elif isinstance(weights, Mapping):
        payload = dict(weights)
    else:
        payload = {
            name: getattr(weights, name)
            for name in ("seg", "pose", "rate", "mixed", "operating_point_tag", "basis")
            if hasattr(weights, name)
        }
    return dict(_json_safe(payload))


def _summarize_sensitivity_weights(weights: Mapping[int, float] | None) -> dict[str, Any] | None:
    if weights is None:
        return None
    values = [float(v) for _, v in sorted(weights.items())]
    if not values:
        return {
            "schema": "sensitivity_byte_weight_summary_v1",
            "n_weights": 0,
            "min_weight": 0.0,
            "max_weight": 0.0,
            "mean_weight": 0.0,
            "n_downweighted": 0,
            "n_upweighted": 0,
        }
    return {
        "schema": "sensitivity_byte_weight_summary_v1",
        "n_weights": len(values),
        "min_weight": min(values),
        "max_weight": max(values),
        "mean_weight": sum(values) / len(values),
        "n_downweighted": sum(1 for v in values if v < 1.0),
        "n_upweighted": sum(1 for v in values if v > 1.0),
    }


@dataclass(frozen=True)
class OptimizerAnalyticalBoundaries:
    """Planning-only bundle for deterministic optimizer boundary inputs.

    This is the canonical handoff from master-gradient, xray, sensitivity-map,
    bit-allocation, and Lagrangian-dual surfaces into deterministic optimizers.
    It deliberately cannot rank/kill, claim a score, promote, or dispatch.
    """

    archive_sha256: str
    master_gradient_anchor: Mapping[str, Any]
    master_gradient_boundary_summary: MasterGradientBoundarySummary
    master_gradient_authority_violation_reason: str | None
    master_gradient_planning_usable: bool
    master_gradient_contest_authoritative: bool
    per_pair_difficulty_atlas: Mapping[str, Any] | None = None
    wyner_ziv_side_info: Mapping[str, Any] | None = None
    optimal_plan_candidate_row: Mapping[str, Any] | None = None
    null_space_basis: Mapping[str, Any] | None = None
    sensitivity_axis_weights: Mapping[str, Any] | None = None
    sensitivity_byte_weights_summary: Mapping[str, Any] | None = None
    xray_hook_inventory: Mapping[str, Sequence[str]] = field(default_factory=dict)
    xray_hook_bundles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bit_allocation_envelope: Mapping[str, Any] | None = None
    lagrangian_dual_envelope: Mapping[str, Any] | None = None
    evidence_grade: str = OPTIMIZER_ANALYTICAL_BOUNDARIES_EVIDENCE_GRADE
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    dispatch_packet_ready: bool = False

    def __post_init__(self) -> None:
        if (
            self.score_claim
            or self.promotion_eligible
            or self.ready_for_exact_eval_dispatch
            or self.rank_or_kill_eligible
            or self.dispatch_packet_ready
        ):
            raise ValueError(
                "OptimizerAnalyticalBoundaries is planning-only and cannot "
                "carry score/promotion/rank/dispatch authority"
            )
        if not self.master_gradient_planning_usable:
            raise ValueError(
                "master-gradient anchor is not usable for planning: "
                f"{self.master_gradient_authority_violation_reason}"
            )
        if self.master_gradient_contest_authoritative and self.master_gradient_authority_violation_reason:
            raise ValueError(
                "contest-authoritative anchor cannot also carry an authority "
                f"violation: {self.master_gradient_authority_violation_reason}"
            )
        for name in (
            "master_gradient_anchor",
            "per_pair_difficulty_atlas",
            "wyner_ziv_side_info",
            "optimal_plan_candidate_row",
            "null_space_basis",
            "bit_allocation_envelope",
            "lagrangian_dual_envelope",
        ):
            surface = getattr(self, name)
            if isinstance(surface, Mapping):
                _authority_flags_are_false(surface, surface_name=name)

    def as_dict(self) -> dict[str, Any]:
        """Deterministic JSON-safe payload for optimizer/research ledgers."""
        return {
            "schema": OPTIMIZER_ANALYTICAL_BOUNDARIES_SCHEMA_VERSION,
            "archive_sha256": self.archive_sha256,
            "master_gradient_anchor": _json_safe(self.master_gradient_anchor),
            "master_gradient_boundary_summary": self.master_gradient_boundary_summary.as_dict(),
            "master_gradient_authority_violation_reason": (
                self.master_gradient_authority_violation_reason
            ),
            "master_gradient_planning_usable": self.master_gradient_planning_usable,
            "master_gradient_contest_authoritative": self.master_gradient_contest_authoritative,
            "per_pair_difficulty_atlas": _json_safe(self.per_pair_difficulty_atlas),
            "wyner_ziv_side_info": _json_safe(self.wyner_ziv_side_info),
            "optimal_plan_candidate_row": _json_safe(self.optimal_plan_candidate_row),
            "null_space_basis": _json_safe(self.null_space_basis),
            "sensitivity_axis_weights": _json_safe(self.sensitivity_axis_weights),
            "sensitivity_byte_weights_summary": _json_safe(
                self.sensitivity_byte_weights_summary
            ),
            "xray_hook_inventory": _json_safe(self.xray_hook_inventory),
            "xray_hook_bundles": _json_safe(self.xray_hook_bundles),
            "bit_allocation_envelope": _json_safe(self.bit_allocation_envelope),
            "lagrangian_dual_envelope": _json_safe(self.lagrangian_dual_envelope),
            "evidence_grade": self.evidence_grade,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "rank_or_kill_eligible": self.rank_or_kill_eligible,
            "dispatch_packet_ready": self.dispatch_packet_ready,
        }


def summarize_master_gradient_boundaries(
    per_pair_gradient: torch.Tensor | Sequence[Any],
    *,
    archive_sha256: str = "",
    magnitude_cliff_ratio: float = 10.0,
    hard_pair_top_k: int = 50,
    sensitive_byte_fraction: float = 0.02,
) -> MasterGradientBoundarySummary:
    """Summarize sign flips, cliffs, hard pairs, and sensitive bytes.

    Args:
        per_pair_gradient: tensor-like shape ``(n_bytes, n_pairs, n_axes)``.
          Axis 0 is treated as SegNet sensitivity and axis 1 as PoseNet
          sensitivity for cosine statistics when present.
        archive_sha256: optional archive identity for custody. Empty is allowed
          for synthetic tests, but production callers should pass the scored
          archive SHA.
        magnitude_cliff_ratio: adjacent byte-norm ratio threshold.
        hard_pair_top_k: number of highest-norm pairs to retain.
        sensitive_byte_fraction: fraction of highest-norm bytes to retain.
    """
    if magnitude_cliff_ratio <= 1.0:
        raise ValueError("magnitude_cliff_ratio must be > 1.0")
    if hard_pair_top_k < 0:
        raise ValueError("hard_pair_top_k must be >= 0")
    if not (0.0 < sensitive_byte_fraction <= 1.0):
        raise ValueError("sensitive_byte_fraction must be in (0, 1]")

    grad = torch.as_tensor(per_pair_gradient, dtype=torch.float64)
    if grad.ndim != 3:
        raise ValueError(
            "per_pair_gradient must have shape (n_bytes, n_pairs, n_axes); "
            f"got ndim={grad.ndim}"
        )
    n_bytes, n_pairs, n_axes = (int(v) for v in grad.shape)
    if n_bytes == 0 or n_pairs == 0 or n_axes < 2:
        raise ValueError(
            "per_pair_gradient must have n_bytes>0, n_pairs>0, and n_axes>=2"
        )

    byte_vectors = grad.mean(dim=1)
    byte_norms = torch.linalg.vector_norm(byte_vectors, dim=1)
    if n_bytes >= 2:
        adjacent_dot = (byte_vectors[1:] * byte_vectors[:-1]).sum(dim=1)
        sign_flip_indices = tuple(
            int(i + 1) for i in torch.nonzero(adjacent_dot < 0, as_tuple=False).flatten()
        )
        left = byte_norms[:-1]
        right = byte_norms[1:]
        eps = torch.finfo(torch.float64).eps
        low = torch.clamp(torch.minimum(left, right), min=eps)
        high = torch.maximum(left, right)
        cliff_indices = tuple(
            int(i + 1)
            for i in torch.nonzero((high / low) >= magnitude_cliff_ratio, as_tuple=False).flatten()
        )
    else:
        sign_flip_indices = ()
        cliff_indices = ()

    pair_matrix = grad.permute(1, 0, 2).reshape(n_pairs, n_bytes * n_axes)
    pair_norms = torch.linalg.vector_norm(pair_matrix, dim=1)
    n_hard = min(n_pairs, int(hard_pair_top_k))
    hard_pair_indices = (
        tuple(int(i) for i in torch.topk(pair_norms, k=n_hard).indices.tolist())
        if n_hard > 0
        else ()
    )

    byte_matrix = grad.reshape(n_bytes, n_pairs * n_axes)
    full_byte_norms = torch.linalg.vector_norm(byte_matrix, dim=1)
    n_sensitive = max(1, min(n_bytes, math.ceil(n_bytes * sensitive_byte_fraction)))
    sensitive_byte_indices = tuple(
        int(i) for i in torch.topk(full_byte_norms, k=n_sensitive).indices.tolist()
    )

    seg = grad[:, :, 0]
    pose = grad[:, :, 1]
    denom = torch.linalg.vector_norm(seg, dim=1) * torch.linalg.vector_norm(pose, dim=1)
    valid = denom > torch.finfo(torch.float64).eps
    if bool(valid.any()):
        cosines = (seg[valid] * pose[valid]).sum(dim=1) / denom[valid]
        mean_cos = float(cosines.mean().item())
        min_cos = float(cosines.min().item())
        max_cos = float(cosines.max().item())
    else:
        mean_cos = 0.0
        min_cos = 0.0
        max_cos = 0.0

    return MasterGradientBoundarySummary(
        archive_sha256=archive_sha256,
        n_bytes=n_bytes,
        n_pairs=n_pairs,
        n_axes=n_axes,
        sign_flip_byte_indices=sign_flip_indices,
        magnitude_cliff_byte_indices=cliff_indices,
        hard_pair_indices=hard_pair_indices,
        sensitive_byte_indices=sensitive_byte_indices,
        mean_seg_pose_cosine=mean_cos,
        min_seg_pose_cosine=min_cos,
        max_seg_pose_cosine=max_cos,
        magnitude_cliff_ratio=float(magnitude_cliff_ratio),
        sensitive_byte_fraction=float(sensitive_byte_fraction),
    )


def build_optimizer_analytical_boundaries(
    *,
    archive_sha256: str | None = None,
    per_pair_gradient: torch.Tensor | Sequence[Any] | None = None,
    master_gradient_anchor: Mapping[str, Any] | None = None,
    optimal_plan_payload: Mapping[str, Any] | None = None,
    xray_targets_by_hook: Mapping[str, Mapping[str, Any]] | None = None,
    total_bit_budget: int | None = None,
) -> OptimizerAnalyticalBoundaries:
    """Build the canonical planning-only boundary bundle for optimizers.

    The function delegates to existing canonical helpers and disables sidecar
    persistence. It refuses master-gradient rows with contest-axis authority
    violations and strips every downstream surface down to non-authoritative
    planning evidence.
    """
    if per_pair_gradient is None:
        from tac.master_gradient_consumers import load_per_pair_gradient_from_anchor

        loaded_gradient, loaded_anchor = load_per_pair_gradient_from_anchor(
            archive_sha256=archive_sha256
        )
        per_pair_gradient = loaded_gradient
        if master_gradient_anchor is None:
            master_gradient_anchor = loaded_anchor
    if master_gradient_anchor is None:
        raise ValueError(
            "master_gradient_anchor is required when per_pair_gradient is supplied"
        )

    from tac.master_gradient import (
        contest_axis_authority_violation_reason,
        is_authoritative_contest_axis_anchor,
        is_usable_planning_anchor,
    )
    from tac.master_gradient_consumers import (
        load_optimal_plan_for_archive,
        optimal_plan_payload_to_candidate_row,
        per_pair_difficulty_atlas,
        wyner_ziv_side_info_covariance,
    )
    from tac.null_space_exploiter import build_null_space_basis
    from tac.sensitivity_map.axis_weights import default_axis_weights
    from tac.sensitivity_map.wyner_ziv_reweight import axis_level_reweight
    from tac.xray.wire_in import (
        aggregate_hook_evidence_grade,
        discover_primitives_by_hook,
        wire_in_for_hook,
    )

    anchor = dict(master_gradient_anchor)
    archive = str(archive_sha256 or anchor.get("archive_sha256") or "").lower()
    if not archive:
        raise ValueError("archive_sha256 is required for optimizer boundaries")

    violation = contest_axis_authority_violation_reason(anchor)
    planning_usable = is_usable_planning_anchor(anchor)
    contest_authoritative = is_authoritative_contest_axis_anchor(anchor)
    if not planning_usable:
        raise ValueError(f"master-gradient anchor is not planning-usable: {violation}")

    measurement_axis = str(anchor.get("measurement_axis", "diagnostic"))
    measurement_hardware = str(anchor.get("measurement_hardware", "unknown"))
    summary = summarize_master_gradient_boundaries(
        per_pair_gradient,
        archive_sha256=archive,
    )
    difficulty = per_pair_difficulty_atlas(
        torch.as_tensor(per_pair_gradient, dtype=torch.float64).cpu().numpy(),
        archive_sha256=archive,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        write_sidecar=False,
    )
    wz = wyner_ziv_side_info_covariance(
        torch.as_tensor(per_pair_gradient, dtype=torch.float64).cpu().numpy(),
        archive_sha256=archive,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        write_sidecar=False,
    )
    sensitivity_weights = axis_level_reweight(wz)
    null_space = build_null_space_basis(
        torch.as_tensor(per_pair_gradient, dtype=torch.float64).cpu().numpy(),
        archive_sha256=archive,
    )

    if optimal_plan_payload is None:
        optimal_plan_payload = load_optimal_plan_for_archive(archive)
    optimal_candidate = None
    if optimal_plan_payload is not None:
        optimal_candidate = optimal_plan_payload_to_candidate_row(dict(optimal_plan_payload))

    inventory = {
        str(hook): [str(name) for name in names]
        for hook, names in discover_primitives_by_hook().items()
    }
    xray_bundles: dict[str, dict[str, Any]] = {}
    targets_by_hook = {
        str(hook): dict(targets)
        for hook, targets in (xray_targets_by_hook or {}).items()
    }
    for hook in inventory:
        bundle = wire_in_for_hook(
            hook,
            targets=targets_by_hook.get(hook, {}),
            skip_on_error=True,
        )
        xray_bundles[hook] = {
            "n_primitives": int(bundle.n_primitives),
            "n_results": len(bundle.results),
            "skipped_primitives": list(bundle.skipped_primitives),
            "evidence_grade": aggregate_hook_evidence_grade(bundle),
            "result_primitives": [
                {
                    "primitive_name": result.primitive_name,
                    "evidence_grade": result.evidence_grade,
                    "confidence_band": result.confidence_band,
                    "wire_in_hooks_engaged": list(result.wire_in_hooks_engaged),
                }
                for result in bundle.results
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "dispatch_packet_ready": False,
        }

    bit_envelope = None
    if total_bit_budget is not None:
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        bit_envelope = allocate_per_pair_bits(
            archive_sha256=archive,
            total_bit_budget=total_bit_budget,
            per_pair_gradient=torch.as_tensor(per_pair_gradient, dtype=torch.float64).cpu().numpy(),
            optimal_plan=dict(optimal_plan_payload) if optimal_plan_payload is not None else None,
            sensitivity_reweight=sensitivity_weights,
            auto_load=False,
            persist_sidecar=False,
        )

    from tac.optimization.field_equation_planner import consume_per_pair_lagrangian_duals

    dual_envelope = consume_per_pair_lagrangian_duals(
        archive_sha256=archive,
        optimal_plan=dict(optimal_plan_payload) if optimal_plan_payload is not None else None,
        auto_load=False,
        persist_sidecar=False,
    )

    return OptimizerAnalyticalBoundaries(
        archive_sha256=archive,
        master_gradient_anchor=anchor,
        master_gradient_boundary_summary=summary,
        master_gradient_authority_violation_reason=violation,
        master_gradient_planning_usable=planning_usable,
        master_gradient_contest_authoritative=contest_authoritative,
        per_pair_difficulty_atlas=_json_safe(difficulty),
        wyner_ziv_side_info=_json_safe(wz),
        optimal_plan_candidate_row=_json_safe(optimal_candidate),
        null_space_basis=_json_safe(null_space),
        sensitivity_axis_weights=_axis_weights_payload(default_axis_weights()),
        sensitivity_byte_weights_summary=_summarize_sensitivity_weights(sensitivity_weights),
        xray_hook_inventory=inventory,
        xray_hook_bundles=xray_bundles,
        bit_allocation_envelope=_json_safe(bit_envelope),
        lagrangian_dual_envelope=_json_safe(dual_envelope),
    )


@dataclass
class Action:
    """A single GR-style action carrying every track Lagrangian as a named field.

    Each ``L_<track>`` is a ``Callable[[theta], torch.Tensor]`` that returns a
    scalar tensor (the per-track contribution to the action). Setting a field
    to ``None`` drops that track from the action.

    Mutable fields (the callables) are intentionally NOT frozen — a trainer
    may swap them between phases.
    """

    L_seg: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_pose: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_rate: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t7: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t8: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t11: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t13: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t20: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t22: Callable[[torch.Tensor], torch.Tensor] | None = None

    duals: DualVariables = field(default_factory=DualVariables)
    metadata: dict[str, Any] = field(default_factory=dict)

    def active_tracks(self) -> list[TrackKind]:
        """Return the list of TrackKind whose contribution is currently active."""
        out: list[TrackKind] = []
        mapping = {
            TrackKind.SEG_BASELINE: self.L_seg,
            TrackKind.POSE_BASELINE: self.L_pose,
            TrackKind.RATE_BASELINE: self.L_rate,
            TrackKind.T7_FISHER_RAO: self.L_t7,
            TrackKind.T8_SINKHORN_W2: self.L_t8,
            TrackKind.T11_LOVASZ_HINGE: self.L_t11,
            TrackKind.T13_JOINT_SOURCE_RD: self.L_t13,
            TrackKind.T20_KL_POSE_DISTILL: self.L_t20,
            TrackKind.T22_TEMPORAL_CONSISTENCY: self.L_t22,
        }
        for kind, L in mapping.items():
            if L is None:
                continue
            # T7/T8/T11/T13/T20/T22 are inactive when their dual is exactly 0
            dual_attr = {
                TrackKind.T7_FISHER_RAO: "lambda_t7",
                TrackKind.T8_SINKHORN_W2: "lambda_t8",
                TrackKind.T11_LOVASZ_HINGE: "lambda_t11",
                TrackKind.T13_JOINT_SOURCE_RD: "lambda_t13",
                TrackKind.T20_KL_POSE_DISTILL: "lambda_t20",
                TrackKind.T22_TEMPORAL_CONSISTENCY: "lambda_t22",
            }.get(kind)
            if dual_attr is not None and getattr(self.duals, dual_attr) == 0.0:
                continue
            out.append(kind)
        return out

    def S_total(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute the scalar action.

        Returns a 0-D tensor with autograd if ``theta.requires_grad``.
        """
        contributions = self._per_track_contributions(theta)
        if not contributions:
            # Degenerate: no active tracks. Return a 0-tensor sharing theta's
            # device/dtype/grad so downstream gradient calls don't crash.
            return torch.zeros((), device=theta.device, dtype=theta.dtype)
        return sum(contributions.values(), start=torch.zeros((), device=theta.device, dtype=theta.dtype))

    def _per_track_contributions(self, theta: torch.Tensor) -> dict[TrackKind, torch.Tensor]:
        """Return per-track {TrackKind -> scalar tensor} for ACTIVE tracks only."""
        contributions: dict[TrackKind, torch.Tensor] = {}
        # Baseline contest contributions ALWAYS use lambda=1.0 weighting since they
        # ARE the contest scorer. The duals only modulate REFINEMENT tracks.
        if self.L_seg is not None:
            contributions[TrackKind.SEG_BASELINE] = self.duals.lambda_seg * self.L_seg(theta)
        if self.L_pose is not None:
            contributions[TrackKind.POSE_BASELINE] = self.duals.lambda_pose * self.L_pose(theta)
        if self.L_rate is not None:
            contributions[TrackKind.RATE_BASELINE] = self.duals.lambda_rate * self.L_rate(theta)
        if self.L_t7 is not None and self.duals.lambda_t7 != 0.0:
            contributions[TrackKind.T7_FISHER_RAO] = self.duals.lambda_t7 * self.L_t7(theta)
        if self.L_t8 is not None and self.duals.lambda_t8 != 0.0:
            contributions[TrackKind.T8_SINKHORN_W2] = self.duals.lambda_t8 * self.L_t8(theta)
        if self.L_t11 is not None and self.duals.lambda_t11 != 0.0:
            contributions[TrackKind.T11_LOVASZ_HINGE] = self.duals.lambda_t11 * self.L_t11(theta)
        if self.L_t13 is not None and self.duals.lambda_t13 != 0.0:
            contributions[TrackKind.T13_JOINT_SOURCE_RD] = self.duals.lambda_t13 * self.L_t13(theta)
        if self.L_t20 is not None and self.duals.lambda_t20 != 0.0:
            contributions[TrackKind.T20_KL_POSE_DISTILL] = self.duals.lambda_t20 * self.L_t20(theta)
        if self.L_t22 is not None and self.duals.lambda_t22 != 0.0:
            contributions[TrackKind.T22_TEMPORAL_CONSISTENCY] = self.duals.lambda_t22 * self.L_t22(theta)
        return contributions

    def gradient(self, theta: torch.Tensor) -> dict[TrackKind, torch.Tensor]:
        """Per-track gradient contributions ``∂L_k/∂θ`` via autograd.

        Each entry is a tensor of the SAME shape as ``theta`` (the gradient of
        that track's contribution alone). Useful for adversarial review of
        which track is dominating the descent direction.

        Requires ``theta.requires_grad``.
        """
        if not theta.requires_grad:
            raise ValueError(
                "Action.gradient requires theta.requires_grad=True; got False"
            )
        contributions = self._per_track_contributions(theta)
        grads: dict[TrackKind, torch.Tensor] = {}
        for kind, contribution in contributions.items():
            # Short-circuit when the contribution is detached / constant — there
            # is no autograd graph to back-propagate through (track does not
            # depend on theta).
            if not contribution.requires_grad or contribution.grad_fn is None:
                grads[kind] = torch.zeros_like(theta)
                continue
            grad = torch.autograd.grad(
                contribution,
                theta,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                # Track does not depend on theta (e.g. constant rate-axis term)
                grad = torch.zeros_like(theta)
            grads[kind] = grad.detach()
        return grads

    def step(
        self,
        theta: torch.Tensor,
        lr: float,
        dual_update: Callable[[DualVariables, dict[TrackKind, torch.Tensor]], DualVariables] | None = None,
    ) -> tuple[torch.Tensor, DualVariables]:
        """One variational step: gradient descent on theta + optional dual update.

        Returns (theta_new, duals_new). The dual_update callable, if provided,
        is called with the OLD duals + the per-track gradient norms (NOT the
        gradient tensors themselves) so it can implement Boyd-style adaptive
        ρ on the dual variables. If ``None``, duals are held constant.

        Note: this is a SIMPLE variational step — production trainers should
        run inside their own optimizer (Adam/SGD/AdamW) and use ``S_total``
        as the loss. ``Action.step`` is here for the smoke-test / scaffold path.
        """
        if not theta.requires_grad:
            raise ValueError("Action.step requires theta.requires_grad=True; got False")
        if not (lr > 0.0):
            raise ValueError(f"lr must be > 0; got {lr}")

        per_track_grads = self.gradient(theta)
        # Sum the per-track gradients to get the total gradient.
        total_grad = sum(
            per_track_grads.values(),
            start=torch.zeros_like(theta),
        )
        with torch.no_grad():
            theta_new = (theta - lr * total_grad).detach().requires_grad_(theta.requires_grad)

        # Optional dual update.
        if dual_update is None:
            duals_new = self.duals
        else:
            grad_norms = {k: float(g.norm().item()) for k, g in per_track_grads.items()}
            duals_new = dual_update(self.duals, grad_norms)
            if not isinstance(duals_new, DualVariables):
                raise TypeError(
                    f"dual_update must return DualVariables; got {type(duals_new).__name__}"
                )

        return theta_new, duals_new

    def migration_status(self) -> dict[str, Any]:
        """Report which tracks are active + which trainer is the canonical migration target.

        Returns a dict suitable for JSON serialization to a research ledger.
        """
        return {
            "schema": ACTION_SCHEMA_VERSION,
            "evidence_grade": ACTION_EVIDENCE_GRADE,
            "active_tracks": [k.value for k in self.active_tracks()],
            "duals": {
                "lambda_seg": self.duals.lambda_seg,
                "lambda_pose": self.duals.lambda_pose,
                "lambda_rate": self.duals.lambda_rate,
                "lambda_t7": self.duals.lambda_t7,
                "lambda_t8": self.duals.lambda_t8,
                "lambda_t11": self.duals.lambda_t11,
                "lambda_t13": self.duals.lambda_t13,
                "lambda_t20": self.duals.lambda_t20,
                "lambda_t22": self.duals.lambda_t22,
            },
            "canonical_trainer_migration_targets": {
                "phase_1_balle_hyperprior": "experiments/train_score_gradient_pr101_finetune.py + tac.paradigm_delta_epsilon_zeta",
                "lane_12_v2_nerv_as_renderer": "tac.lane_12_v2_nerv_as_renderer.train_step",
                "score_gradient_pr101_finetune": "experiments/train_score_gradient_pr101_finetune.py",
            },
            "metadata": dict(self.metadata),
        }

    def assert_invariants(self) -> None:
        """Self-check that the action is in a consistent state.

        Raises ``ValueError`` with a clear diagnostic if any invariant fails.
        Useful as a tripwire at the top of every trainer step or as a sanity
        gate before serializing ``migration_status()`` into a research
        ledger.

        Invariants checked:

        1. Every dual variable is finite (no NaN / Inf from a bad
           ``dual_update`` callable).
        2. ``lambda_seg`` / ``lambda_pose`` / ``lambda_rate`` are non-negative
           (refinement-track duals MAY be signed when the council intentionally
           switches a constraint sense, but baselines must never invert).
        3. At least one of ``L_seg`` / ``L_pose`` / ``L_rate`` is wired (a
           degenerate Action with only refinement tracks is a trainer bug —
           the refinements have nothing to refine).
        4. ``metadata`` is a plain ``dict`` (frozen via dataclass field
           default; mutable but type-checked).

        Tagged ``[diagnostic; tac.unified_action.Action.assert_invariants]``
        per CLAUDE.md "Forbidden score claims" — this is a structural check,
        not a score claim.
        """
        import math

        for name in (
            "lambda_seg",
            "lambda_pose",
            "lambda_rate",
            "lambda_t7",
            "lambda_t8",
            "lambda_t11",
            "lambda_t13",
            "lambda_t20",
            "lambda_t22",
        ):
            v = getattr(self.duals, name)
            if not math.isfinite(float(v)):
                raise ValueError(
                    f"Action.assert_invariants: duals.{name} = {v!r} is "
                    "not finite. A bad dual_update callable produced "
                    "NaN/Inf. Snapshot the action's migration_status() "
                    "and inspect the last dual_update step."
                )
        for name in ("lambda_seg", "lambda_pose", "lambda_rate"):
            v = float(getattr(self.duals, name))
            if v < 0.0:
                raise ValueError(
                    f"Action.assert_invariants: duals.{name} = {v} < 0. "
                    "Baseline track duals must be non-negative; refinement "
                    "duals may be signed."
                )
        if (
            self.L_seg is None
            and self.L_pose is None
            and self.L_rate is None
        ):
            raise ValueError(
                "Action.assert_invariants: no baseline track wired "
                "(L_seg, L_pose, L_rate all None). Refinement tracks have "
                "nothing to refine; wire at least one baseline."
            )
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"Action.assert_invariants: metadata must be a dict, got "
                f"{type(self.metadata).__name__}"
            )


def make_action_from_track_callables(
    seg: Callable | None = None,
    pose: Callable | None = None,
    rate: Callable | None = None,
    *,
    t7_fisher_rao: Callable | None = None,
    t8_sinkhorn_w2: Callable | None = None,
    t11_lovasz_hinge: Callable | None = None,
    t13_joint_source_rd: Callable | None = None,
    t20_kl_pose_distill: Callable | None = None,
    t22_temporal_consistency: Callable | None = None,
    duals: DualVariables | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Action:
    """Convenience factory that takes per-track callables + DualVariables.

    Example::

        from tac.unified_action import make_action_from_track_callables, DualVariables

        action = make_action_from_track_callables(
            seg=lambda theta: torch.sum(theta**2),
            duals=DualVariables(lambda_seg=1.0),
        )
        S = action.S_total(theta)
    """
    return Action(
        L_seg=seg,
        L_pose=pose,
        L_rate=rate,
        L_t7=t7_fisher_rao,
        L_t8=t8_sinkhorn_w2,
        L_t11=t11_lovasz_hinge,
        L_t13=t13_joint_source_rd,
        L_t20=t20_kl_pose_distill,
        L_t22=t22_temporal_consistency,
        duals=duals or DualVariables(),
        metadata=dict(metadata) if metadata else {},
    )


__all__ = [
    "ACTION_EVIDENCE_GRADE",
    "ACTION_SCHEMA_VERSION",
    "OPTIMIZER_ANALYTICAL_BOUNDARIES_EVIDENCE_GRADE",
    "OPTIMIZER_ANALYTICAL_BOUNDARIES_SCHEMA_VERSION",
    "Action",
    "DualVariables",
    "MasterGradientBoundarySummary",
    "OptimizerAnalyticalBoundaries",
    "SurfaceKind",
    "TrackKind",
    "build_optimizer_analytical_boundaries",
    "make_action_from_track_callables",
    "summarize_master_gradient_boundaries",
]
