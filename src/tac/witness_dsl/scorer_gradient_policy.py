# SPDX-License-Identifier: MIT
"""Typed, fail-closed policy for replacing the frozen SegNet training gradient.

Replacement evidence is cryptographically scoped to one scorer objective,
provider, anchor frame, current frame, pair, loss, and stage.  Global input-
costate agreement is mandatory.  Optional margin-annulus metrics are additional
evidence and can never authorize a costate that fails globally.

This is a contract surface, not a live-trainer flag.  Any missing, stale,
replayed, cross-frame, cross-objective, mutated-custody, or nonfinite evidence
selects ``full_teacher``.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tac.boundary_math.segnet_gradient_replacement import (
    CostateAgreementMetrics,
    TeacherStepCheck,
    array_content_sha256,
    measure_costate_agreement,
    relative_frame_displacement,
)

GradientMode = Literal[
    "full_teacher",
    "periodic_student",
    "periodic_costate",
    "trusted_jacobian_cache",
    "yopo_first_layer_costate",
]
JsonScalar = str | int | float | bool | None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_shape_and_finite(value: Any) -> tuple[tuple[int, ...], bool]:
    array_like = value
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy") and not isinstance(array_like, np.ndarray):
        array_like = array_like.numpy()
    array = np.asarray(array_like)
    return tuple(array.shape), bool(np.isfinite(array).all())


class ScorerObjectiveContext(BaseModel):
    """Everything that changes the meaning of a SegNet training costate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer_sha256: str
    preprocess_sha256: str
    receiver_r_sha256: str
    gt_targets_sha256: str
    pair_index: int = Field(ge=0)
    loss_name: str = Field(min_length=1)
    loss_parameters: dict[str, JsonScalar]
    stage_name: str = Field(min_length=1)
    stage_parameters: dict[str, JsonScalar]

    @field_validator(
        "scorer_sha256",
        "preprocess_sha256",
        "receiver_r_sha256",
        "gt_targets_sha256",
    )
    @classmethod
    def _strong_sha(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("objective component fingerprints must be lowercase SHA-256")
        return value

    @field_validator("loss_name", "stage_name")
    @classmethod
    def _no_placeholder_name(cls, value: str) -> str:
        stripped = value.strip()
        if any(token in stripped.lower() for token in ("tbd", "placeholder", "<value>")):
            raise ValueError("objective names must be concrete, not placeholders")
        return stripped

    @field_validator("loss_parameters", "stage_parameters")
    @classmethod
    def _finite_parameters(cls, value: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        if not value:
            raise ValueError("loss/stage parameters must be explicit and non-empty")
        for key, item in value.items():
            if not key.strip():
                raise ValueError("loss/stage parameter keys must be non-empty")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"objective parameter {key!r} must be finite")
        return value

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderStatFingerprint:
    """Cheap between-refresh mutation detector for already-hashed provider bytes."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> ProviderStatFingerprint:
        stat = path.stat()
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )


class ProviderCustody(BaseModel):
    """Immutable bytes defining a student, costate network, or cache."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["checkpoint", "cache"]
    path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("provider custody sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _reject_placeholder_path(self) -> ProviderCustody:
        lowered = self.path.strip().lower()
        if not lowered or any(token in lowered for token in ("<value>", "tbd", "placeholder")):
            raise ValueError("provider custody path must be concrete, not a placeholder")
        return self

    def verify_full(self) -> tuple[bool, str, ProviderStatFingerprint | None]:
        """Hash once at compile/teacher refresh, detecting mutation during the read."""

        path = Path(self.path)
        if not path.is_file():
            return False, f"provider custody artifact is missing: {path}", None
        before = ProviderStatFingerprint.capture(path)
        if before.size_bytes != self.size_bytes:
            return (
                False,
                f"provider custody byte count changed: expected={self.size_bytes}, "
                f"actual={before.size_bytes}",
                None,
            )
        actual_sha = _sha256_file(path)
        after = ProviderStatFingerprint.capture(path)
        if before != after:
            return False, "provider custody artifact changed while its SHA-256 was read", None
        if actual_sha != self.sha256:
            return (
                False,
                f"provider custody sha256 changed: expected={self.sha256}, actual={actual_sha}",
                None,
            )
        return True, "provider custody full SHA-256 verified", after

    def verify_stat(
        self, expected: ProviderStatFingerprint | None
    ) -> tuple[bool, str, ProviderStatFingerprint | None]:
        """Use one ``stat`` between refreshes; never rehash unchanged bytes per step."""

        if expected is None:
            return False, "compiled provider stat fingerprint is missing", None
        path = Path(self.path)
        if not path.is_file():
            return False, f"provider custody artifact is missing: {path}", None
        current = ProviderStatFingerprint.capture(path)
        if current != expected:
            return (
                False,
                "provider custody stat fingerprint changed between full-SHA refreshes",
                current,
            )
        return True, "provider custody cheap stat fingerprint verified", current


@dataclass(frozen=True)
class ProviderCostateEvaluation:
    """One provider output bound to the frame/objective/provider/step it evaluated."""

    costate: Any
    frame_sha256: str
    objective_context_fingerprint: str
    provider_custody_sha256: str
    evaluated_at_step: int
    split_module_path: str | None = None
    split_identity_sha256: str | None = None
    bank_source_step: int | None = None


@dataclass(frozen=True)
class TeacherGradientObservation:
    """Periodic real-teacher anchor plus provider evaluation on that exact anchor."""

    teacher_costate_at_anchor: Any
    provider_costate_at_anchor: ProviderCostateEvaluation
    anchor_frame: Any
    anchor_frame_sha256: str
    measured_at_step: int
    objective_context_fingerprint: str
    scorer_fingerprint: str
    teacher_step_check: TeacherStepCheck


@dataclass(frozen=True)
class ScorerGradientDecision:
    """Machine-readable admission result; failure always selects full teacher."""

    requested_mode: GradientMode
    selected_mode: GradientMode
    admitted: bool
    reasons: tuple[str, ...]
    global_costate_metrics: CostateAgreementMetrics | None = None
    annulus_costate_metrics: CostateAgreementMetrics | None = None
    teacher_step_check: TeacherStepCheck | None = None
    frame_relative_displacement: float | None = None
    custody_check: str | None = None

    @property
    def fallback_to_full_teacher(self) -> bool:
        return self.selected_mode == "full_teacher"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "admitted": self.admitted,
            "fallback_to_full_teacher": self.fallback_to_full_teacher,
            "reasons": list(self.reasons),
            "global_costate_metrics": (
                None
                if self.global_costate_metrics is None
                else self.global_costate_metrics.to_dict()
            ),
            "annulus_costate_metrics": (
                None
                if self.annulus_costate_metrics is None
                else self.annulus_costate_metrics.to_dict()
            ),
            "teacher_step_check": (
                None if self.teacher_step_check is None else self.teacher_step_check.to_dict()
            ),
            "frame_relative_displacement": self.frame_relative_displacement,
            "custody_check": self.custody_check,
        }


class ScorerGradientPolicy(BaseModel):
    """Source policy whose replacement fields deliberately have no defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: GradientMode
    refresh_interval_steps: int | None = Field(default=None, ge=1)
    max_staleness_steps: int | None = Field(default=None, ge=0)
    scorer_fingerprint: str | None = None
    objective_context: ScorerObjectiveContext | None = None
    objective_context_fingerprint: str | None = None
    min_costate_cosine: float | None = Field(default=None, gt=0.0, le=1.0)
    max_costate_relative_l2: float | None = Field(default=None, ge=0.0)
    min_costate_norm_ratio: float | None = Field(default=None, gt=0.0)
    max_costate_norm_ratio: float | None = Field(default=None, gt=0.0)
    max_teacher_loss_regret: float | None = Field(default=None, ge=0.0)
    provider_custody: ProviderCustody | None = None
    max_frame_relative_l2: float | None = Field(default=None, gt=0.0)
    split_module_path: str | None = None
    split_identity_sha256: str | None = None

    @model_validator(mode="after")
    def _mode_contract(self) -> ScorerGradientPolicy:
        replacement_fields = (
            "refresh_interval_steps",
            "max_staleness_steps",
            "scorer_fingerprint",
            "objective_context",
            "objective_context_fingerprint",
            "min_costate_cosine",
            "max_costate_relative_l2",
            "min_costate_norm_ratio",
            "max_costate_norm_ratio",
            "max_teacher_loss_regret",
            "provider_custody",
        )
        if self.mode == "full_teacher":
            supplied = [name for name in replacement_fields if getattr(self, name) is not None]
            if self.max_frame_relative_l2 is not None:
                supplied.append("max_frame_relative_l2")
            if self.split_module_path is not None:
                supplied.append("split_module_path")
            if self.split_identity_sha256 is not None:
                supplied.append("split_identity_sha256")
            if supplied:
                raise ValueError(
                    "full_teacher must not carry dormant replacement fields: " + ", ".join(supplied)
                )
            return self

        if self.mode == "yopo_first_layer_costate":
            yopo_base_fields = (
                "refresh_interval_steps",
                "max_staleness_steps",
                "scorer_fingerprint",
                "objective_context",
                "objective_context_fingerprint",
                "provider_custody",
            )
            missing = [name for name in yopo_base_fields if getattr(self, name) is None]
            forbidden_thresholds = (
                "min_costate_cosine",
                "max_costate_relative_l2",
                "min_costate_norm_ratio",
                "max_costate_norm_ratio",
                "max_teacher_loss_regret",
            )
            supplied_thresholds = [name for name in forbidden_thresholds if getattr(self, name) is not None]
            if supplied_thresholds:
                raise ValueError(
                    "yopo_first_layer_costate rejects universal agreement/regret thresholds: "
                    + ", ".join(supplied_thresholds)
                )
        else:
            missing = [name for name in replacement_fields if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"replacement mode {self.mode!r} requires explicit fields: {', '.join(missing)}"
            )

        assert self.refresh_interval_steps is not None
        assert self.max_staleness_steps is not None
        if self.max_staleness_steps > self.refresh_interval_steps:
            raise ValueError("max_staleness_steps must be <= refresh_interval_steps")
        assert self.scorer_fingerprint is not None
        if not _is_sha256(self.scorer_fingerprint):
            raise ValueError("scorer_fingerprint must be a lowercase 64-character SHA-256")
        assert self.objective_context is not None
        assert self.objective_context_fingerprint is not None
        if self.scorer_fingerprint != self.objective_context.scorer_sha256:
            raise ValueError("scorer_fingerprint must equal objective_context.scorer_sha256")
        if self.objective_context_fingerprint != self.objective_context.fingerprint():
            raise ValueError(
                "objective_context_fingerprint does not match the canonical context payload"
            )
        if self.mode != "yopo_first_layer_costate":
            assert self.min_costate_norm_ratio is not None
            assert self.max_costate_norm_ratio is not None
            if self.min_costate_norm_ratio > self.max_costate_norm_ratio:
                raise ValueError("costate norm-ratio lower bound exceeds upper bound")

        assert self.provider_custody is not None
        expected_kind = "cache" if self.mode in {"trusted_jacobian_cache", "yopo_first_layer_costate"} else "checkpoint"
        if self.provider_custody.kind != expected_kind:
            raise ValueError(
                f"{self.mode} requires provider_custody.kind={expected_kind!r}, got "
                f"{self.provider_custody.kind!r}"
            )
        if self.mode == "trusted_jacobian_cache":
            if self.max_frame_relative_l2 is None:
                raise ValueError("trusted_jacobian_cache requires max_frame_relative_l2")
        elif self.max_frame_relative_l2 is not None:
            raise ValueError("max_frame_relative_l2 is cache-specific")
        if self.mode == "yopo_first_layer_costate":
            if self.split_module_path != "encoder.model.blocks[0]":
                raise ValueError("yopo_first_layer_costate requires split_module_path='encoder.model.blocks[0]'")
            if not _is_sha256(self.split_identity_sha256):
                raise ValueError("yopo_first_layer_costate requires a lowercase split_identity_sha256")
            assert self.refresh_interval_steps is not None
            assert self.max_staleness_steps is not None
            if self.max_staleness_steps != self.refresh_interval_steps - 1:
                raise ValueError("yopo_first_layer_costate requires max_staleness_steps=refresh_interval_steps - 1")
        elif self.split_module_path is not None or self.split_identity_sha256 is not None:
            raise ValueError("split identity fields are exclusive to yopo_first_layer_costate")
        return self

    def compile(self) -> CompiledScorerGradientPolicy:
        """Compile after a full provider SHA-256 verification."""

        if self.mode == "full_teacher":
            return CompiledScorerGradientPolicy(
                source=self,
                _provider_stat=None,
                _compiled_objective_fingerprint=None,
            )
        assert self.objective_context is not None
        assert self.objective_context_fingerprint is not None
        compiled_objective_fingerprint = self.objective_context.fingerprint()
        if compiled_objective_fingerprint != self.objective_context_fingerprint:
            raise ValueError(
                "objective context changed between validation and policy compilation"
            )
        assert self.provider_custody is not None
        ok, reason, stat = self.provider_custody.verify_full()
        if not ok:
            raise ValueError(reason)
        return CompiledScorerGradientPolicy(
            source=self,
            _provider_stat=stat,
            _compiled_objective_fingerprint=compiled_objective_fingerprint,
        )


@dataclass
class CompiledScorerGradientPolicy:
    """Executable admission contract with refresh-scoped custody state."""

    source: ScorerGradientPolicy
    _provider_stat: ProviderStatFingerprint | None
    _compiled_objective_fingerprint: str | None

    def _verify_custody(self, *, refresh: bool) -> tuple[bool, str]:
        custody = self.source.provider_custody
        if custody is None:
            return True, "full teacher has no replacement provider custody"
        if refresh or self.source.mode == "yopo_first_layer_costate":
            ok, reason, stat = custody.verify_full()
            if ok:
                self._provider_stat = stat
            return ok, reason
        ok, reason, _ = custody.verify_stat(self._provider_stat)
        return ok, reason

    def _metric_reasons(
        self, metrics: CostateAgreementMetrics, *, label: str
    ) -> list[str]:
        policy = self.source
        if not metrics.valid:
            return [f"{label}: {reason}" for reason in (metrics.reasons or ("invalid metrics",))]
        if self.source.mode == "yopo_first_layer_costate":
            return []
        assert metrics.cosine_similarity is not None
        assert metrics.relative_l2_error is not None
        assert metrics.norm_ratio is not None
        assert policy.min_costate_cosine is not None
        assert policy.max_costate_relative_l2 is not None
        assert policy.min_costate_norm_ratio is not None
        assert policy.max_costate_norm_ratio is not None
        reasons: list[str] = []
        if metrics.cosine_similarity < policy.min_costate_cosine:
            reasons.append(
                f"{label} cosine {metrics.cosine_similarity:.9g} < "
                f"minimum {policy.min_costate_cosine:.9g}"
            )
        if metrics.relative_l2_error > policy.max_costate_relative_l2:
            reasons.append(
                f"{label} relative L2 {metrics.relative_l2_error:.9g} > "
                f"maximum {policy.max_costate_relative_l2:.9g}"
            )
        if not (
            policy.min_costate_norm_ratio
            <= metrics.norm_ratio
            <= policy.max_costate_norm_ratio
        ):
            reasons.append(
                f"{label} norm ratio {metrics.norm_ratio:.9g} outside "
                f"[{policy.min_costate_norm_ratio:.9g}, "
                f"{policy.max_costate_norm_ratio:.9g}]"
            )
        return reasons

    def _validate_provider_evaluation(
        self,
        evaluation: ProviderCostateEvaluation,
        *,
        frame: Any,
        frame_sha256: str,
        objective_context_fingerprint: str,
        evaluated_at_step: int,
        label: str,
    ) -> list[str]:
        policy = self.source
        assert policy.provider_custody is not None
        reasons: list[str] = []
        if not isinstance(evaluation, ProviderCostateEvaluation):
            return [f"{label} is not a ProviderCostateEvaluation"]
        if not _is_sha256(evaluation.frame_sha256):
            reasons.append(f"{label} frame SHA-256 is invalid")
        if not _is_sha256(evaluation.objective_context_fingerprint):
            reasons.append(f"{label} objective/context SHA-256 is invalid")
        if not _is_sha256(evaluation.provider_custody_sha256):
            reasons.append(f"{label} provider custody SHA-256 is invalid")
        if evaluation.frame_sha256 != frame_sha256:
            reasons.append(f"{label} frame hash does not match its bound frame")
        if evaluation.objective_context_fingerprint != objective_context_fingerprint:
            reasons.append(f"{label} objective/context fingerprint mismatch")
        if evaluation.provider_custody_sha256 != policy.provider_custody.sha256:
            reasons.append(f"{label} provider custody SHA-256 mismatch")
        if policy.mode == "yopo_first_layer_costate":
            if evaluation.split_module_path != policy.split_module_path:
                reasons.append(f"{label} YOPO split module path mismatch")
            if evaluation.split_identity_sha256 != policy.split_identity_sha256:
                reasons.append(f"{label} YOPO split identity SHA-256 mismatch")
            if (
                not isinstance(evaluation.bank_source_step, int)
                or isinstance(evaluation.bank_source_step, bool)
                or evaluation.bank_source_step < 0
            ):
                reasons.append(f"{label} YOPO bank source step must be an integer >= 0")
        if not isinstance(evaluation.evaluated_at_step, int) or isinstance(
            evaluation.evaluated_at_step, bool
        ):
            reasons.append(f"{label} evaluated_at_step must be an integer")
        if evaluation.evaluated_at_step != evaluated_at_step:
            reasons.append(
                f"{label} replayed step: evaluated={evaluation.evaluated_at_step}, "
                f"required={evaluated_at_step}"
            )
        try:
            frame_shape, frame_finite = _array_shape_and_finite(frame)
            costate_shape, costate_finite = _array_shape_and_finite(evaluation.costate)
        except (TypeError, ValueError) as exc:
            reasons.append(f"{label} array conversion failed: {exc}")
            return reasons
        if frame_shape != costate_shape:
            reasons.append(
                f"{label} costate/frame shape mismatch: {costate_shape} != {frame_shape}"
            )
        if not frame_finite or not costate_finite:
            reasons.append(f"{label} frame or costate contains a nonfinite value")
        return reasons

    def decide(
        self,
        *,
        current_provider_evaluation: ProviderCostateEvaluation | None,
        teacher_observation: TeacherGradientObservation | None,
        current_frame: Any | None,
        current_frame_sha256: str | None,
        current_objective_context_fingerprint: str | None,
        current_step: int,
        annulus_mask: Any | None = None,
    ) -> ScorerGradientDecision:
        policy = self.source
        if policy.mode == "full_teacher":
            return ScorerGradientDecision(
                requested_mode="full_teacher",
                selected_mode="full_teacher",
                admitted=True,
                reasons=("full teacher is the explicit baseline policy",),
            )

        reasons: list[str] = []
        global_metrics: CostateAgreementMetrics | None = None
        annulus_metrics: CostateAgreementMetrics | None = None
        step_check: TeacherStepCheck | None = None
        frame_displacement: float | None = None

        compiled_objective_fingerprint = self._compiled_objective_fingerprint
        required_objective_fingerprint: str
        if not _is_sha256(compiled_objective_fingerprint):
            reasons.append("compiled objective/context fingerprint is missing or invalid")
            required_objective_fingerprint = ""
        else:
            required_objective_fingerprint = str(compiled_objective_fingerprint)
        assert policy.objective_context is not None
        try:
            runtime_objective_fingerprint = policy.objective_context.fingerprint()
        except (TypeError, ValueError) as exc:
            reasons.append(f"policy objective context cannot be rehashed: {exc}")
        else:
            if runtime_objective_fingerprint != required_objective_fingerprint:
                reasons.append("policy objective context mutated after compilation")
        if policy.objective_context_fingerprint != required_objective_fingerprint:
            reasons.append("declared objective/context fingerprint changed after compilation")

        current_step_valid = (
            isinstance(current_step, int)
            and not isinstance(current_step, bool)
            and current_step >= 0
        )
        if not current_step_valid:
            reasons.append("current_step must be an integer >= 0")
        observation = (
            teacher_observation
            if isinstance(teacher_observation, TeacherGradientObservation)
            else None
        )
        if teacher_observation is not None and observation is None:
            reasons.append("teacher observation has the wrong contract type")
        refresh = bool(
            observation is not None
            and current_step_valid
            and observation.measured_at_step == current_step
        )
        custody_ok, custody_reason = self._verify_custody(refresh=refresh)
        if not custody_ok:
            reasons.append(custody_reason)

        if current_objective_context_fingerprint != required_objective_fingerprint:
            reasons.append("current objective/context fingerprint does not match policy")
        if current_frame is None or not _is_sha256(current_frame_sha256):
            reasons.append("current frame and its lowercase SHA-256 are required")
        else:
            try:
                actual_current_sha = array_content_sha256(current_frame)
            except (TypeError, ValueError) as exc:
                reasons.append(f"current frame content hashing failed: {exc}")
            else:
                if actual_current_sha != current_frame_sha256:
                    reasons.append("current frame content SHA-256 mismatch")

        if observation is None:
            reasons.append("missing periodic real-teacher observation")
        else:
            measured_step_valid = (
                isinstance(observation.measured_at_step, int)
                and not isinstance(observation.measured_at_step, bool)
                and observation.measured_at_step >= 0
            )
            if not measured_step_valid:
                reasons.append("teacher measured_at_step must be an integer >= 0")
            age = (
                current_step - observation.measured_at_step
                if current_step_valid and measured_step_valid
                else None
            )
            assert policy.max_staleness_steps is not None
            assert policy.refresh_interval_steps is not None
            if age is not None and age < 0:
                reasons.append("teacher observation is from the future")
            if age is not None and age > policy.max_staleness_steps:
                reasons.append(
                    f"teacher observation is stale: age={age} > "
                    f"max_staleness_steps={policy.max_staleness_steps}"
                )
            if age is not None and age >= policy.refresh_interval_steps:
                reasons.append(
                    f"teacher refresh is due: age={age} >= "
                    f"refresh_interval_steps={policy.refresh_interval_steps}"
                )
            if observation.objective_context_fingerprint != required_objective_fingerprint:
                reasons.append("teacher anchor objective/context fingerprint mismatch")
            if observation.scorer_fingerprint != policy.scorer_fingerprint:
                reasons.append("teacher anchor scorer fingerprint mismatch")
            if not _is_sha256(observation.anchor_frame_sha256):
                reasons.append("teacher anchor frame SHA-256 is invalid")
            else:
                try:
                    actual_anchor_sha = array_content_sha256(observation.anchor_frame)
                except (TypeError, ValueError) as exc:
                    reasons.append(f"teacher anchor frame hashing failed: {exc}")
                else:
                    if actual_anchor_sha != observation.anchor_frame_sha256:
                        reasons.append("teacher anchor frame content SHA-256 mismatch")

            try:
                anchor_shape, anchor_finite = _array_shape_and_finite(observation.anchor_frame)
                teacher_shape, teacher_finite = _array_shape_and_finite(
                    observation.teacher_costate_at_anchor
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"real-teacher anchor array conversion failed: {exc}")
            else:
                if anchor_shape != teacher_shape:
                    reasons.append("real-teacher anchor costate/frame shape mismatch")
                if not anchor_finite or not teacher_finite:
                    reasons.append("real-teacher anchor frame or costate is nonfinite")

            provider_anchor = observation.provider_costate_at_anchor
            if not isinstance(provider_anchor, ProviderCostateEvaluation):
                reasons.append("provider anchor evaluation has the wrong contract type")
            else:
                reasons.extend(
                    self._validate_provider_evaluation(
                        provider_anchor,
                        frame=observation.anchor_frame,
                        frame_sha256=observation.anchor_frame_sha256,
                        objective_context_fingerprint=required_objective_fingerprint,
                        evaluated_at_step=observation.measured_at_step,
                        label="provider anchor evaluation",
                    )
                )
                try:
                    global_metrics = measure_costate_agreement(
                        observation.teacher_costate_at_anchor,
                        provider_anchor.costate,
                    )
                    reasons.extend(self._metric_reasons(global_metrics, label="global costate"))
                    if policy.mode == "yopo_first_layer_costate" and (
                        array_content_sha256(observation.teacher_costate_at_anchor)
                        != array_content_sha256(provider_anchor.costate)
                    ):
                        reasons.append(
                            "YOPO refresh provider costate must be byte-identical to the exact teacher costate"
                        )
                    if annulus_mask is not None:
                        annulus_metrics = measure_costate_agreement(
                            observation.teacher_costate_at_anchor,
                            provider_anchor.costate,
                            mask=annulus_mask,
                        )
                        reasons.extend(
                            self._metric_reasons(annulus_metrics, label="annulus costate")
                        )
                except (TypeError, ValueError) as exc:
                    reasons.append(f"costate metric evaluation failed: {exc}")

            step_check = observation.teacher_step_check
            assert policy.provider_custody is not None
            if not isinstance(step_check, TeacherStepCheck):
                reasons.append("teacher step check has the wrong contract type")
            elif step_check.objective_context_fingerprint != required_objective_fingerprint:
                reasons.append("teacher step check objective/context fingerprint mismatch")
            if isinstance(step_check, TeacherStepCheck) and (
                step_check.anchor_frame_sha256 != observation.anchor_frame_sha256
            ):
                reasons.append("teacher step check belongs to a different anchor frame")
            if isinstance(step_check, TeacherStepCheck) and (
                step_check.provider_custody_sha256 != policy.provider_custody.sha256
            ):
                reasons.append("teacher step check provider custody SHA-256 mismatch")
            if isinstance(step_check, TeacherStepCheck) and (
                step_check.evaluated_at_step != observation.measured_at_step
            ):
                reasons.append("teacher step check was replayed from a different step")
            if isinstance(step_check, TeacherStepCheck) and (
                not _is_sha256(step_check.candidate_frame_sha256)
                or not _is_sha256(step_check.reference_frame_sha256)
            ):
                reasons.append("teacher step check candidate/reference frame hashes are invalid")
            if isinstance(step_check, TeacherStepCheck):
                try:
                    actual_candidate_sha = array_content_sha256(step_check.candidate_frame)
                    actual_reference_sha = array_content_sha256(step_check.reference_frame)
                    candidate_shape, candidate_finite = _array_shape_and_finite(
                        step_check.candidate_frame
                    )
                    reference_shape, reference_finite = _array_shape_and_finite(
                        step_check.reference_frame
                    )
                    anchor_shape, _ = _array_shape_and_finite(observation.anchor_frame)
                except (TypeError, ValueError) as exc:
                    reasons.append(f"teacher step check candidate provenance failed: {exc}")
                else:
                    if actual_candidate_sha != step_check.candidate_frame_sha256:
                        reasons.append(
                            "teacher step check candidate frame content SHA-256 mismatch"
                        )
                    if actual_reference_sha != step_check.reference_frame_sha256:
                        reasons.append(
                            "teacher step check reference frame content SHA-256 mismatch"
                        )
                    if candidate_shape != anchor_shape or reference_shape != anchor_shape:
                        reasons.append(
                            "teacher step check candidate/reference frame shape differs from anchor"
                        )
                    if not candidate_finite or not reference_finite:
                        reasons.append("teacher step check candidate/reference frame is nonfinite")
            if isinstance(step_check, TeacherStepCheck):
                if policy.mode == "yopo_first_layer_costate":
                    if not (
                        step_check.finite
                        and step_check.decreases_teacher_loss
                        and step_check.regret is not None
                        and math.isfinite(step_check.regret)
                    ):
                        reasons.append(
                            "YOPO real-teacher one-step check must be finite, decreasing, "
                            "and carry finite measured regret"
                        )
                else:
                    assert policy.max_teacher_loss_regret is not None
                    if not step_check.passes(max_regret=policy.max_teacher_loss_regret):
                        reasons.append(
                            "real-teacher one-step check failed: candidate must decrease teacher loss "
                            "with bounded regret"
                        )

        if current_provider_evaluation is None:
            reasons.append("replacement provider did not supply a current-frame costate evaluation")
        elif not isinstance(current_provider_evaluation, ProviderCostateEvaluation):
            reasons.append("current provider evaluation has the wrong contract type")
        elif current_frame is not None and current_frame_sha256 is not None:
            reasons.extend(
                self._validate_provider_evaluation(
                    current_provider_evaluation,
                    frame=current_frame,
                    frame_sha256=current_frame_sha256,
                    objective_context_fingerprint=required_objective_fingerprint,
                    evaluated_at_step=current_step,
                    label="current provider evaluation",
                )
            )

        if refresh and observation is not None:
            if observation.anchor_frame_sha256 != current_frame_sha256:
                reasons.append(
                    "refresh anchor frame SHA-256 does not equal the current injection frame"
                )
            provider_anchor = observation.provider_costate_at_anchor
            if isinstance(provider_anchor, ProviderCostateEvaluation) and isinstance(
                current_provider_evaluation, ProviderCostateEvaluation
            ):
                try:
                    anchor_provider_costate_sha = array_content_sha256(provider_anchor.costate)
                    current_provider_costate_sha = array_content_sha256(
                        current_provider_evaluation.costate
                    )
                except (TypeError, ValueError) as exc:
                    reasons.append(f"refresh provider costate content hashing failed: {exc}")
                else:
                    if anchor_provider_costate_sha != current_provider_costate_sha:
                        reasons.append(
                            "refresh current provider costate content differs from the "
                            "teacher-validated anchor provider costate"
                        )

        if policy.mode == "yopo_first_layer_costate" and observation is not None:
            if isinstance(current_provider_evaluation, ProviderCostateEvaluation) and (
                current_provider_evaluation.bank_source_step != observation.measured_at_step
            ):
                reasons.append("current YOPO bank source step does not match teacher anchor step")
            provider_anchor = observation.provider_costate_at_anchor
            if isinstance(provider_anchor, ProviderCostateEvaluation) and (
                provider_anchor.bank_source_step != observation.measured_at_step
            ):
                reasons.append("anchor YOPO bank source step does not match teacher anchor step")

        if policy.mode == "trusted_jacobian_cache" and observation is not None:
            if current_frame is None:
                reasons.append("trusted cache is missing current-frame trust evidence")
            else:
                try:
                    frame_displacement = relative_frame_displacement(
                        observation.anchor_frame, current_frame
                    )
                except (TypeError, ValueError) as exc:
                    frame_displacement = float("inf")
                    reasons.append(f"cache trust-radius evaluation failed: {exc}")
                assert policy.max_frame_relative_l2 is not None
                if frame_displacement > policy.max_frame_relative_l2:
                    reasons.append(
                        f"frame left cache trust radius: relative_l2={frame_displacement:.9g} "
                        f"> {policy.max_frame_relative_l2:.9g}"
                    )

        admitted = not reasons
        return ScorerGradientDecision(
            requested_mode=policy.mode,
            selected_mode=(policy.mode if admitted else "full_teacher"),
            admitted=admitted,
            reasons=("replacement admitted by all explicit gates",) if admitted else tuple(reasons),
            global_costate_metrics=global_metrics,
            annulus_costate_metrics=annulus_metrics,
            teacher_step_check=step_check,
            frame_relative_displacement=frame_displacement,
            custody_check=custody_reason,
        )


def compile_scorer_gradient_policy(payload: Mapping[str, Any]) -> CompiledScorerGradientPolicy:
    """Validate a mapping through the typed DSL, then compile it fail-closed."""

    return ScorerGradientPolicy.model_validate(payload).compile()


__all__ = [
    "CompiledScorerGradientPolicy",
    "GradientMode",
    "ProviderCostateEvaluation",
    "ProviderCustody",
    "ProviderStatFingerprint",
    "ScorerGradientDecision",
    "ScorerGradientPolicy",
    "ScorerObjectiveContext",
    "TeacherGradientObservation",
    "compile_scorer_gradient_policy",
]
