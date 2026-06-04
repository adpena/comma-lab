# SPDX-License-Identifier: MIT
"""Train-time dual-ascent controls for MLX score-aware NeRV training.

The contest objective is a rate-distortion Lagrangian, but the useful training
surface is vector-valued: SegNet last-frame distortion, PoseNet pair/YUV6
distortion, hard-pair errors, and rate/coder proxy terms all become active at
different times.  This module provides a small pure-Python projected
dual-ascent controller that updates loss weights from observed training metrics.

It is intentionally backend-neutral.  The MLX adapter owns differentiating the
resulting weighted loss; this controller only updates scalar multipliers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TRAIN_TIME_DUAL_ASCENT_SCHEMA = "mlx_train_time_dual_ascent.v1"
TRAIN_TIME_DUAL_ASCENT_CONSTRAINT_SCHEMA = "mlx_train_time_dual_ascent_constraint.v1"
_VALID_DIRECTIONS = frozenset({"upper_bound", "lower_bound"})
_SAFE_KEY_RE = re.compile(r"[^0-9A-Za-z_]+")
_DEFAULT_SCORER_TARGET_FRACTION = 0.985
_DEFAULT_CODER_TARGET_FRACTION = 0.98
_DEFAULT_RATCHET_FRACTION = 0.995
_FORBIDDEN_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "score_claim",
        "frontier_score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "promotable",
        "score_claim_valid",
    }
)


class TrainTimeDualAscentError(ValueError):
    """Raised when a dual-ascent controller config is malformed."""


@dataclass(frozen=True)
class DualAscentConstraint:
    """One train-time constraint and its projected dual variable settings."""

    constraint_id: str
    metric_name: str
    loss_weight_key: str
    direction: str = "upper_bound"
    target: float | None = None
    target_fraction_of_initial: float | None = None
    target_ratchet_fraction: float | None = None
    dual_lr: float = 0.1
    initial_lambda: float = 0.0
    min_lambda: float = 0.0
    max_lambda: float = 16.0
    weight_scale: float = 1.0
    metric_scale: float = 1.0
    warmup_steps: int = 0
    update_every_steps: int = 1
    bootstrap_update: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> DualAscentConstraint:
        cid = str(
            raw.get("constraint_id")
            or raw.get("id")
            or raw.get("name")
            or f"constraint_{index:04d}"
        )
        metric_name = str(raw.get("metric_name") or raw.get("metric") or "")
        loss_weight_key = str(raw.get("loss_weight_key") or raw.get("weight_key") or "")
        if not cid:
            raise TrainTimeDualAscentError("constraint_id must be non-empty")
        if not metric_name:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {cid!r} missing metric_name"
            )
        if not loss_weight_key:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {cid!r} missing loss_weight_key"
            )
        direction = str(raw.get("direction") or "upper_bound")
        target = _optional_finite_float(raw.get("target"), "target", constraint_id=cid)
        target_fraction = _optional_finite_float(
            raw.get("target_fraction_of_initial"),
            "target_fraction_of_initial",
            constraint_id=cid,
        )
        ratchet_fraction = _optional_finite_float(
            raw.get("target_ratchet_fraction"),
            "target_ratchet_fraction",
            constraint_id=cid,
        )
        out = cls(
            constraint_id=cid,
            metric_name=metric_name,
            loss_weight_key=loss_weight_key,
            direction=direction,
            target=target,
            target_fraction_of_initial=target_fraction,
            target_ratchet_fraction=ratchet_fraction,
            dual_lr=_finite_float(raw.get("dual_lr", 0.1), "dual_lr", constraint_id=cid),
            initial_lambda=_finite_float(
                raw.get("initial_lambda", raw.get("lambda", 0.0)),
                "initial_lambda",
                constraint_id=cid,
            ),
            min_lambda=_finite_float(
                raw.get("min_lambda", 0.0),
                "min_lambda",
                constraint_id=cid,
            ),
            max_lambda=_finite_float(
                raw.get("max_lambda", 16.0),
                "max_lambda",
                constraint_id=cid,
            ),
            weight_scale=_finite_float(
                raw.get("weight_scale", 1.0),
                "weight_scale",
                constraint_id=cid,
            ),
            metric_scale=_finite_float(
                raw.get("metric_scale", 1.0),
                "metric_scale",
                constraint_id=cid,
            ),
            warmup_steps=_nonnegative_int(
                raw.get("warmup_steps", 0),
                "warmup_steps",
                constraint_id=cid,
            ),
            update_every_steps=_positive_int(
                raw.get("update_every_steps", 1),
                "update_every_steps",
                constraint_id=cid,
            ),
            bootstrap_update=bool(raw.get("bootstrap_update", False)),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} direction must be "
                f"one of {sorted(_VALID_DIRECTIONS)}"
            )
        if self.target is None and self.target_fraction_of_initial is None:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} needs target or "
                "target_fraction_of_initial"
            )
        if self.target_fraction_of_initial is not None and (
            self.target_fraction_of_initial <= 0.0
        ):
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} "
                "target_fraction_of_initial must be > 0"
            )
        if self.target_ratchet_fraction is not None and (
            self.target_ratchet_fraction <= 0.0
        ):
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} "
                "target_ratchet_fraction must be > 0"
            )
        if self.dual_lr < 0.0:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} dual_lr must be >= 0"
            )
        if self.min_lambda < 0.0:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} min_lambda must be >= 0"
            )
        if self.max_lambda < self.min_lambda:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} max_lambda must be "
                ">= min_lambda"
            )
        if self.weight_scale < 0.0:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} weight_scale must be >= 0"
            )
        if self.metric_scale <= 0.0:
            raise TrainTimeDualAscentError(
                f"dual-ascent constraint {self.constraint_id!r} metric_scale must be > 0"
            )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": TRAIN_TIME_DUAL_ASCENT_CONSTRAINT_SCHEMA,
            "constraint_id": self.constraint_id,
            "metric_name": self.metric_name,
            "loss_weight_key": self.loss_weight_key,
            "direction": self.direction,
            "target": self.target,
            "target_fraction_of_initial": self.target_fraction_of_initial,
            "target_ratchet_fraction": self.target_ratchet_fraction,
            "dual_lr": self.dual_lr,
            "initial_lambda": self.initial_lambda,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "weight_scale": self.weight_scale,
            "metric_scale": self.metric_scale,
            "warmup_steps": self.warmup_steps,
            "update_every_steps": self.update_every_steps,
            "bootstrap_update": self.bootstrap_update,
        }


@dataclass
class _DualState:
    lambda_value: float
    target: float | None
    initial_metric: float | None = None
    last_metric: float | None = None
    last_violation: float | None = None
    last_weight_contribution: float = 0.0
    update_count: int = 0
    missing_metric_count: int = 0


class TrainTimeDualAscentController:
    """Projected dual-ascent controller over scalar training metrics."""

    def __init__(
        self,
        *,
        enabled: bool,
        constraints: Sequence[DualAscentConstraint],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.constraints = tuple(constraints)
        self.metadata = dict(metadata or {})
        self.step_count = 0
        self._states: dict[str, _DualState] = {
            constraint.constraint_id: _DualState(
                lambda_value=_clamp(
                    constraint.initial_lambda,
                    constraint.min_lambda,
                    constraint.max_lambda,
                ),
                target=constraint.target,
            )
            for constraint in self.constraints
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
    ) -> TrainTimeDualAscentController:
        if not config:
            return cls(enabled=False, constraints=(), metadata={})
        raw_constraints = config.get("constraints") or ()
        if not isinstance(raw_constraints, Sequence) or isinstance(
            raw_constraints,
            (str, bytes),
        ):
            raise TrainTimeDualAscentError(
                "train-time dual-ascent config constraints must be a sequence"
            )
        constraints = tuple(
            DualAscentConstraint.from_mapping(raw, index=index)
            for index, raw in enumerate(raw_constraints)
            if isinstance(raw, Mapping)
        )
        if len(constraints) != len(raw_constraints):
            raise TrainTimeDualAscentError(
                "train-time dual-ascent constraints must all be mappings"
            )
        return cls(
            enabled=bool(config.get("enabled", bool(constraints))),
            constraints=constraints,
            metadata=_strip_forbidden_artifact_metadata_keys(
                {
                    key: value
                    for key, value in dict(config).items()
                    if key != "constraints"
                }
            ),
        )

    def effective_loss_weights(
        self,
        base_loss_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        weights = {
            str(key): float(value)
            for key, value in dict(base_loss_weights or {}).items()
        }
        if not self.enabled:
            return weights
        for constraint in self.constraints:
            state = self._states[constraint.constraint_id]
            contribution = float(state.lambda_value) * float(constraint.weight_scale)
            state.last_weight_contribution = contribution
            if contribution:
                weights[constraint.loss_weight_key] = (
                    float(weights.get(constraint.loss_weight_key, 0.0))
                    + contribution
                )
        return weights

    def observe(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        if not self.enabled:
            return {}
        self.step_count += 1
        telemetry: dict[str, float] = {
            "dual_ascent_active": 1.0,
            "dual_ascent_step": float(self.step_count),
            "dual_ascent_constraint_count": float(len(self.constraints)),
        }
        for constraint in self.constraints:
            state = self._states[constraint.constraint_id]
            key = _safe_key(constraint.constraint_id)
            metric_raw = metrics.get(constraint.metric_name)
            metric = _metric_float_or_none(metric_raw)
            telemetry[f"dual_ascent_missing_metric__{key}"] = float(metric is None)
            if metric is None:
                state.missing_metric_count += 1
                telemetry[f"dual_ascent_lambda__{key}"] = state.lambda_value
                continue
            metric *= constraint.metric_scale
            state.last_metric = metric
            bootstrapped = False
            if state.target is None:
                state.initial_metric = metric
                state.target = metric * float(constraint.target_fraction_of_initial)
                bootstrapped = True
            elif (
                constraint.target_ratchet_fraction is not None
                and _constraint_satisfied(metric, state.target, constraint.direction)
            ):
                ratcheted = metric * float(constraint.target_ratchet_fraction)
                if constraint.direction == "upper_bound":
                    state.target = min(state.target, ratcheted)
                else:
                    state.target = max(state.target, ratcheted)
            violation = _constraint_violation(
                metric,
                float(state.target),
                constraint.direction,
            )
            state.last_violation = violation
            should_update = (
                self.step_count > int(constraint.warmup_steps)
                and self.step_count % int(constraint.update_every_steps) == 0
                and (constraint.bootstrap_update or not bootstrapped)
            )
            if should_update:
                state.lambda_value = _clamp(
                    state.lambda_value + float(constraint.dual_lr) * violation,
                    constraint.min_lambda,
                    constraint.max_lambda,
                )
                state.update_count += 1
            telemetry[f"dual_ascent_metric__{key}"] = metric
            telemetry[f"dual_ascent_target__{key}"] = float(state.target)
            telemetry[f"dual_ascent_violation__{key}"] = violation
            telemetry[f"dual_ascent_lambda__{key}"] = state.lambda_value
            telemetry[f"dual_ascent_update_count__{key}"] = float(state.update_count)
            telemetry[f"dual_ascent_weight_contribution__{key}"] = (
                state.last_weight_contribution
            )
        return telemetry

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema": TRAIN_TIME_DUAL_ASCENT_SCHEMA,
            "enabled": self.enabled,
            "step_count": int(self.step_count),
            "constraint_count": len(self.constraints),
            "constraints": [constraint.as_jsonable() for constraint in self.constraints],
            "state": {
                cid: {
                    "lambda": state.lambda_value,
                    "target": state.target,
                    "initial_metric": state.initial_metric,
                    "last_metric": state.last_metric,
                    "last_violation": state.last_violation,
                    "last_weight_contribution": state.last_weight_contribution,
                    "update_count": state.update_count,
                    "missing_metric_count": state.missing_metric_count,
                }
                for cid, state in sorted(self._states.items())
            },
            "metadata": _strip_forbidden_artifact_metadata_keys(self.metadata),
            "authority": "macos_mlx_research_signal_false_authority",
        }


def build_default_nerv_train_time_dual_ascent_config(
    *,
    family: str,
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    coder_qat_loss_weight_map: Mapping[str, float] | None = None,
    enabled: bool = True,
    warmup_steps: int = 0,
    update_every_steps: int = 1,
) -> dict[str, Any]:
    """Build the shared HiNeRV/SNeRV train-time constraint controller config.

    The default constraints are deliberately loss-part keyed rather than
    architecture keyed. HiNeRV and SNeRV can use different renderers, codecs,
    and modelsize controls while still pricing the same scorer-causal training
    signals: SegNet last-frame distillation, PoseNet YUV6 pair distillation,
    and decoder coder-QAT/rate proxy terms. The controller remains
    false-authority; measured archive bytes and exact CPU/CUDA replay still own
    promotion.
    """

    constraints: list[dict[str, Any]] = []
    seg_weight = _nonnegative_weight(segnet_distillation_weight)
    pose_weight = _nonnegative_weight(pose_distillation_weight)
    if seg_weight > 0.0:
        constraints.append(
            _constraint_payload(
                constraint_id=f"{family}_segnet_last_frame_distill",
                metric_name="loss_part_distill",
                loss_weight_key="distill",
                base_weight=seg_weight,
                target_fraction=_DEFAULT_SCORER_TARGET_FRACTION,
                dual_lr=0.2,
                max_lambda=6.0,
                warmup_steps=warmup_steps,
                update_every_steps=update_every_steps,
                rationale=(
                    "SegNet scores only the last frame of each pair; this "
                    "dual raises the scorer-bound distillation price when "
                    "last-frame boundary loss stalls above its active target."
                ),
            )
        )
    if pose_weight > 0.0:
        constraints.append(
            _constraint_payload(
                constraint_id=f"{family}_posenet_yuv6_pair_distill",
                metric_name="loss_part_pose_distill",
                loss_weight_key="pose_distill",
                base_weight=pose_weight,
                target_fraction=_DEFAULT_SCORER_TARGET_FRACTION,
                dual_lr=0.2,
                max_lambda=6.0,
                warmup_steps=warmup_steps,
                update_every_steps=update_every_steps,
                rationale=(
                    "PoseNet scores the pair through PR95/YUV6 preprocessing; "
                    "this dual prices pair-level pose loss separately from "
                    "SegNet boundary loss."
                ),
            )
        )

    for key, value in sorted(dict(coder_qat_loss_weight_map or {}).items()):
        weight = _nonnegative_weight(value)
        if weight <= 0.0:
            continue
        constraints.append(
            _constraint_payload(
                constraint_id=f"{family}_{_safe_key(key)}",
                metric_name=f"loss_part_{key}",
                loss_weight_key=str(key),
                base_weight=weight,
                target_fraction=_DEFAULT_CODER_TARGET_FRACTION,
                dual_lr=0.25,
                max_lambda=8.0,
                warmup_steps=warmup_steps,
                update_every_steps=update_every_steps,
                rationale=(
                    "Coder/QAT term is a train-time proxy for archive section "
                    "price; section byte authority remains post-export, but "
                    "this makes rate pressure adaptive during training."
                ),
            )
        )

    return {
        "schema": TRAIN_TIME_DUAL_ASCENT_SCHEMA,
        "enabled": bool(enabled and constraints),
        "family": str(family),
        "constraint_count": len(constraints),
        "constraints": constraints,
        "controller_kind": "projected_dual_ascent_vector_constraints",
        "contest_grounding": {
            "score_objective": (
                "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/"
                "uncompressed_total"
            ),
            "segnet_domain": "last_frame_only",
            "posenet_domain": "pair_yuv6",
            "archive_byte_price": "fixed_by_upstream_evaluate_py",
            "human_visual_fidelity_objective": False,
        },
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _constraint_payload(
    *,
    constraint_id: str,
    metric_name: str,
    loss_weight_key: str,
    base_weight: float,
    target_fraction: float,
    dual_lr: float,
    max_lambda: float,
    warmup_steps: int,
    update_every_steps: int,
    rationale: str,
) -> dict[str, Any]:
    return {
        "schema": TRAIN_TIME_DUAL_ASCENT_CONSTRAINT_SCHEMA,
        "constraint_id": constraint_id,
        "metric_name": metric_name,
        "loss_weight_key": loss_weight_key,
        "direction": "upper_bound",
        "target_fraction_of_initial": float(target_fraction),
        "target_ratchet_fraction": _DEFAULT_RATCHET_FRACTION,
        "dual_lr": float(dual_lr),
        "initial_lambda": 0.0,
        "min_lambda": 0.0,
        "max_lambda": float(max_lambda),
        "weight_scale": float(base_weight),
        "metric_scale": 1.0,
        "warmup_steps": int(warmup_steps),
        "update_every_steps": int(update_every_steps),
        "bootstrap_update": False,
        "base_loss_weight": float(base_weight),
        "rationale": rationale,
    }


def _nonnegative_weight(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out) or out <= 0.0:
        return 0.0
    return out


def _strip_forbidden_artifact_metadata_keys(value: Any) -> Any:
    """Return a JSON-like copy safe for nested MLX artifact metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip_forbidden_artifact_metadata_keys(item)
            for key, item in value.items()
            if str(key) not in _FORBIDDEN_ARTIFACT_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_strip_forbidden_artifact_metadata_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_forbidden_artifact_metadata_keys(item) for item in value]
    return value


def _constraint_satisfied(metric: float, target: float, direction: str) -> bool:
    if direction == "upper_bound":
        return metric <= target
    return metric >= target


def _constraint_violation(metric: float, target: float, direction: str) -> float:
    if direction == "upper_bound":
        return metric - target
    return target - metric


def _metric_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_float(value: Any, field: str, *, constraint_id: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainTimeDualAscentError(
            f"dual-ascent constraint {constraint_id!r} {field} must be finite float"
        ) from exc
    if not math.isfinite(out):
        raise TrainTimeDualAscentError(
            f"dual-ascent constraint {constraint_id!r} {field} must be finite float"
        )
    return out


def _optional_finite_float(
    value: Any,
    field: str,
    *,
    constraint_id: str,
) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field, constraint_id=constraint_id)


def _nonnegative_int(value: Any, field: str, *, constraint_id: str) -> int:
    out = _positive_int(value, field, constraint_id=constraint_id, allow_zero=True)
    return out


def _positive_int(
    value: Any,
    field: str,
    *,
    constraint_id: str,
    allow_zero: bool = False,
) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainTimeDualAscentError(
            f"dual-ascent constraint {constraint_id!r} {field} must be int"
        ) from exc
    if out < 0 or (out == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise TrainTimeDualAscentError(
            f"dual-ascent constraint {constraint_id!r} {field} must be {comparator}"
        )
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _safe_key(value: str) -> str:
    cleaned = _SAFE_KEY_RE.sub("_", str(value)).strip("_")
    return cleaned or "constraint"


__all__ = [
    "TRAIN_TIME_DUAL_ASCENT_SCHEMA",
    "DualAscentConstraint",
    "TrainTimeDualAscentController",
    "TrainTimeDualAscentError",
    "build_default_nerv_train_time_dual_ascent_config",
]
