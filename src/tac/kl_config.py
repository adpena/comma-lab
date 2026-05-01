"""Frozen KL/distillation policy schema and provenance serializer.

This module is intentionally independent of the training entry points. It
centralises the policy vocabulary needed to distinguish promotion-capable
SegNet auxiliary KL from forensic-only primary, legacy SegNet-KL, and JBL
distillation paths before those paths are wired into trainers.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

KL_POLICY_FORMAT = "distillation_policy_v1"
KL_POLICY_SCHEMA_VERSION = 1

DistillationFamily: TypeAlias = Literal[
    "none",
    "segnet_aux_kl",
    "primary_scorer_kl",
    "segnet_kl_legacy",
    "jbl",
]
DistillationScope: TypeAlias = Literal["none", "segnet_aux", "primary_scorer"]
ClassWeightNormalization: TypeAlias = Literal["none", "mean_one_at_loss"]
JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple[JsonScalar, ...]
MetadataItems: TypeAlias = tuple[tuple[str, JsonValue], ...]

_ACTIVE_FAMILIES = {"segnet_aux_kl", "primary_scorer_kl", "segnet_kl_legacy", "jbl"}
_FORENSIC_ONLY_FAMILIES = {"primary_scorer_kl", "segnet_kl_legacy", "jbl"}
_KNOWN_SOURCE_KEYS = (
    "family",
    "scope",
    "loss_mode",
    "kl_distill_scope",
    "weight",
    "kl_distill_weight",
    "temperature",
    "kl_distill_temperature",
    "class_weights",
    "class_weights_metadata",
    "segnet_class_weights",
    "segnet_class_weights_metadata",
    "eval_roundtrip",
    "roundtrip_contract",
    "student_teacher_roundtrip",
    "student_eval_roundtrip",
    "teacher_eval_roundtrip",
    "student_uses_eval_roundtrip",
    "teacher_uses_eval_roundtrip",
    "kl_uses_scorer_roundtrip",
    "same_as_scorer_input",
    "promotion_eligible",
    "forensic_reason",
    "forensic_hold_reason",
    "allow_banned_primary",
    "allow_banned_primary_kl_distill",
    "controller",
    "controller_metadata",
)


class DistillationPolicyError(ValueError):
    """Raised when a KL/distillation policy is malformed or non-compliant."""


@dataclass(frozen=True, slots=True)
class ClassWeightsMetadata:
    """Serializable metadata for optional per-class SegNet weighting."""

    enabled: bool = False
    source: str = "none"
    values: tuple[float, ...] | None = None
    num_classes: int | None = None
    normalization: ClassWeightNormalization = "none"
    metadata: MetadataItems = field(default_factory=tuple)

    def __post_init__(self) -> None:
        values = _coerce_float_tuple(self.values, field_name="class_weights.values")
        enabled = _coerce_bool(self.enabled, field_name="class_weights.enabled") or values is not None
        source = _clean_string(self.source, field_name="class_weights.source") or ("explicit" if enabled else "none")
        num_classes = _coerce_optional_positive_int(self.num_classes, field_name="class_weights.num_classes")
        normalization = self.normalization
        if normalization not in ("none", "mean_one_at_loss"):
            raise DistillationPolicyError(
                f"class_weights.normalization must be 'none' or 'mean_one_at_loss', got {normalization!r}"
            )
        if values is not None:
            if not values:
                raise DistillationPolicyError("class_weights.values cannot be empty")
            if any(v < 0.0 for v in values):
                raise DistillationPolicyError("class_weights.values must be non-negative")
            if all(v == 0.0 for v in values):
                raise DistillationPolicyError("class_weights.values cannot be all zeros")
            if num_classes is None:
                num_classes = len(values)
            if len(values) != num_classes:
                raise DistillationPolicyError(
                    f"class_weights.values has {len(values)} entries but num_classes={num_classes}"
                )
            if normalization == "none":
                normalization = "mean_one_at_loss"
        elif enabled and num_classes is not None and num_classes <= 0:
            raise DistillationPolicyError("class_weights.num_classes must be positive")

        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata, field_name="class_weights.metadata"))

    def to_provenance(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "values": list(self.values) if self.values is not None else None,
            "num_classes": self.num_classes,
            "normalization": self.normalization,
            "metadata": _metadata_to_dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoundtripContract:
    """Student/teacher frame contract for scorer-roundtripped KL inputs."""

    eval_roundtrip: bool = True
    student_uses_eval_roundtrip: bool = True
    teacher_uses_eval_roundtrip: bool = True
    same_as_scorer_input: bool = True
    student_source: str = "renderer_output"
    teacher_source: str = "teacher_frames"

    def __post_init__(self) -> None:
        object.__setattr__(self, "eval_roundtrip", _coerce_bool(self.eval_roundtrip, field_name="eval_roundtrip"))
        object.__setattr__(
            self,
            "student_uses_eval_roundtrip",
            _coerce_bool(self.student_uses_eval_roundtrip, field_name="student_uses_eval_roundtrip"),
        )
        object.__setattr__(
            self,
            "teacher_uses_eval_roundtrip",
            _coerce_bool(self.teacher_uses_eval_roundtrip, field_name="teacher_uses_eval_roundtrip"),
        )
        object.__setattr__(
            self,
            "same_as_scorer_input",
            _coerce_bool(self.same_as_scorer_input, field_name="same_as_scorer_input"),
        )
        object.__setattr__(
            self,
            "student_source",
            _clean_string(self.student_source, field_name="student_source") or "renderer_output",
        )
        object.__setattr__(
            self,
            "teacher_source",
            _clean_string(self.teacher_source, field_name="teacher_source") or "teacher_frames",
        )

    @property
    def promotion_safe(self) -> bool:
        return (
            self.eval_roundtrip
            and self.student_uses_eval_roundtrip
            and self.teacher_uses_eval_roundtrip
            and self.same_as_scorer_input
        )

    def to_provenance(self) -> dict[str, Any]:
        return {
            "eval_roundtrip": self.eval_roundtrip,
            "student_uses_eval_roundtrip": self.student_uses_eval_roundtrip,
            "teacher_uses_eval_roundtrip": self.teacher_uses_eval_roundtrip,
            "same_as_scorer_input": self.same_as_scorer_input,
            "student_source": self.student_source,
            "teacher_source": self.teacher_source,
        }


@dataclass(frozen=True, slots=True)
class ControllerMetadata:
    """Serializable metadata for adaptive KL-weight controllers."""

    enabled: bool = False
    kind: str | None = None
    parameters: MetadataItems = field(default_factory=tuple)
    state: MetadataItems = field(default_factory=tuple)

    def __post_init__(self) -> None:
        enabled = _coerce_bool(self.enabled, field_name="controller.enabled")
        kind = _clean_optional_string(self.kind, field_name="controller.kind")
        if enabled and kind is None:
            kind = "unspecified"
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", _freeze_metadata(self.parameters, field_name="controller.parameters"))
        object.__setattr__(self, "state", _freeze_metadata(self.state, field_name="controller.state"))

    def to_provenance(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kind": self.kind,
            "parameters": _metadata_to_dict(self.parameters),
            "state": _metadata_to_dict(self.state),
        }


@dataclass(frozen=True, slots=True)
class DistillationPolicy:
    """Frozen Grand-Council-compatible KL/distillation policy."""

    family: DistillationFamily = "none"
    scope: DistillationScope = "none"
    weight: float = 0.0
    temperature: float = 2.0
    class_weights: ClassWeightsMetadata = field(default_factory=ClassWeightsMetadata)
    roundtrip_contract: RoundtripContract = field(default_factory=RoundtripContract)
    promotion_eligible: bool = True
    forensic_reason: str | None = None
    allow_banned_primary: bool = False
    controller: ControllerMetadata = field(default_factory=ControllerMetadata)

    def __post_init__(self) -> None:
        family = _normalize_family_value(self.family)
        scope = _normalize_scope_value(self.scope)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "weight", _coerce_finite_float(self.weight, field_name="weight", min_value=0.0))
        object.__setattr__(self, "temperature", _coerce_finite_float(self.temperature, field_name="temperature", gt=0.0))
        object.__setattr__(self, "class_weights", normalize_class_weights_metadata(self.class_weights))
        object.__setattr__(self, "roundtrip_contract", normalize_roundtrip_contract(self.roundtrip_contract))
        object.__setattr__(
            self,
            "promotion_eligible",
            _coerce_bool(self.promotion_eligible, field_name="promotion_eligible"),
        )
        object.__setattr__(
            self,
            "forensic_reason",
            _clean_optional_string(self.forensic_reason, field_name="forensic_reason"),
        )
        object.__setattr__(
            self,
            "allow_banned_primary",
            _coerce_bool(self.allow_banned_primary, field_name="allow_banned_primary"),
        )
        object.__setattr__(self, "controller", normalize_controller_metadata(self.controller))

        errors = _policy_validation_errors(self)
        if errors:
            raise DistillationPolicyError("; ".join(errors))

    @property
    def active(self) -> bool:
        return self.family in _ACTIVE_FAMILIES and self.weight > 0.0

    @property
    def promotion_capable(self) -> bool:
        return (
            self.family == "segnet_aux_kl"
            and self.promotion_eligible
            and self.weight > 0.0
            and self.temperature >= 2.0
            and self.roundtrip_contract.promotion_safe
        )

    def promotion_blockers(self) -> list[str]:
        return promotion_blockers(self)

    def to_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": KL_POLICY_SCHEMA_VERSION,
            "format": KL_POLICY_FORMAT,
            "family": self.family,
            "scope": self.scope,
            "weight": self.weight,
            "temperature": self.temperature,
            "class_weights": self.class_weights.to_provenance(),
            "roundtrip_contract": self.roundtrip_contract.to_provenance(),
            "promotion_eligible": self.promotion_eligible,
            "promotion_capable": self.promotion_capable,
            "forensic_reason": self.forensic_reason,
            "allow_banned_primary": self.allow_banned_primary,
            "controller": self.controller.to_provenance(),
            "promotion_blockers": self.promotion_blockers(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_provenance(),
            indent=indent,
            separators=(",", ": ") if indent is not None else (",", ":"),
            sort_keys=True,
        )


def normalize_distillation_policy(source: Any = None, **overrides: Any) -> DistillationPolicy:
    """Normalize legacy config/profile fields into a frozen policy.

    Accepted inputs include mappings, pydantic models with ``model_dump()``, or
    objects exposing current config-style attributes such as ``loss_mode`` and
    ``kl_distill_scope``.
    """

    if isinstance(source, DistillationPolicy) and not overrides:
        return validate_distillation_policy(source)

    data = _source_to_dict(source)
    data.update(overrides)

    explicit_family = "family" in data and data["family"] is not None
    family = _infer_family(data)

    explicit_scope = any(key in data and data[key] is not None for key in ("scope", "kl_distill_scope"))
    if family == "none" and not explicit_family:
        scope = "none"
    elif explicit_scope:
        scope = _normalize_scope_value(data.get("scope", data.get("kl_distill_scope")))
    elif explicit_family:
        scope = _default_scope_for_family(family)
    else:
        scope = "none"

    weight_key = _first_present(data, ("weight", "kl_distill_weight"))
    if family == "none" and not explicit_family:
        weight = 0.0
    else:
        weight = _coerce_finite_float(data.get(weight_key, 0.0), field_name=weight_key or "weight", min_value=0.0)

    temp_key = _first_present(data, ("temperature", "kl_distill_temperature"))
    temperature = _coerce_finite_float(data.get(temp_key, 2.0), field_name=temp_key or "temperature", gt=0.0)

    promotion_eligible = _coerce_bool(data.get("promotion_eligible", True), field_name="promotion_eligible")
    forensic_reason = data.get("forensic_reason", data.get("forensic_hold_reason"))
    allow_banned_primary = _coerce_bool(
        data.get("allow_banned_primary", data.get("allow_banned_primary_kl_distill", False)),
        field_name="allow_banned_primary",
    )

    return DistillationPolicy(
        family=family,
        scope=scope,
        weight=weight,
        temperature=temperature,
        class_weights=_extract_class_weights_metadata(data),
        roundtrip_contract=_extract_roundtrip_contract(data),
        promotion_eligible=promotion_eligible,
        forensic_reason=forensic_reason,
        allow_banned_primary=allow_banned_primary,
        controller=_extract_controller_metadata(data),
    )


def validate_distillation_policy(policy: DistillationPolicy) -> DistillationPolicy:
    if not isinstance(policy, DistillationPolicy):
        policy = normalize_distillation_policy(policy)
    errors = _policy_validation_errors(policy)
    if errors:
        raise DistillationPolicyError("; ".join(errors))
    return policy


def distillation_policy_to_provenance(policy: DistillationPolicy | Mapping[str, Any] | object) -> dict[str, Any]:
    return validate_distillation_policy(normalize_distillation_policy(policy)).to_provenance()


def distillation_policy_to_json(
    policy: DistillationPolicy | Mapping[str, Any] | object,
    *,
    indent: int | None = None,
) -> str:
    return normalize_distillation_policy(policy).to_json(indent=indent)


def distillation_policy_sha256(policy: DistillationPolicy | Mapping[str, Any] | object) -> str:
    provenance = normalize_distillation_policy(policy).to_provenance()
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def promotion_blockers(policy: DistillationPolicy) -> list[str]:
    blockers: list[str] = []
    if policy.family == "none":
        return blockers
    if policy.family == "primary_scorer_kl":
        blockers.append("primary_scorer_kl is forensic-only")
    if policy.family == "segnet_kl_legacy":
        blockers.append("segnet_kl_legacy is forensic-only until migrated and exact-eval-gated")
    if policy.family == "jbl":
        blockers.append("JBL is non-promotable unless represented as forensic")
    if not policy.promotion_eligible:
        blockers.append("promotion_eligible is false")
    if policy.weight <= 0.0:
        blockers.append("weight is not positive")
    if policy.temperature < 2.0:
        blockers.append("temperature is below 2.0")
    if not policy.roundtrip_contract.eval_roundtrip:
        blockers.append("eval_roundtrip is false")
    if not policy.roundtrip_contract.student_uses_eval_roundtrip:
        blockers.append("student input is not eval-roundtripped")
    if not policy.roundtrip_contract.teacher_uses_eval_roundtrip:
        blockers.append("teacher input is not eval-roundtripped")
    if not policy.roundtrip_contract.same_as_scorer_input:
        blockers.append("KL inputs are not declared to match scorer inputs")
    return blockers


def normalize_class_weights_metadata(value: Any = None) -> ClassWeightsMetadata:
    if isinstance(value, ClassWeightsMetadata):
        return value
    if value is None or value is False or value == "":
        return ClassWeightsMetadata()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, str):
        values = _parse_float_csv(value, field_name="class_weights")
        return ClassWeightsMetadata(enabled=True, source="csv", values=values)
    if isinstance(value, Mapping):
        enabled = value.get("enabled", "values" in value)
        values = value.get("values")
        if values is None and "weights" in value:
            values = value["weights"]
        metadata = {
            str(k): v
            for k, v in value.items()
            if k not in {"enabled", "source", "values", "weights", "num_classes", "normalization"}
        }
        return ClassWeightsMetadata(
            enabled=enabled,
            source=value.get("source", "metadata"),
            values=_coerce_float_tuple(values, field_name="class_weights.values"),
            num_classes=value.get("num_classes"),
            normalization=value.get("normalization", "none"),
            metadata=_freeze_metadata(metadata, field_name="class_weights.metadata"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ClassWeightsMetadata(enabled=True, source="sequence", values=_coerce_float_tuple(value))
    raise DistillationPolicyError(f"unsupported class_weights metadata: {value!r}")


def normalize_roundtrip_contract(value: Any = None) -> RoundtripContract:
    if isinstance(value, RoundtripContract):
        return value
    if value is None:
        return RoundtripContract()
    if isinstance(value, Mapping):
        eval_roundtrip = value.get("eval_roundtrip", True)
        return RoundtripContract(
            eval_roundtrip=eval_roundtrip,
            student_uses_eval_roundtrip=value.get(
                "student_uses_eval_roundtrip",
                value.get("student_eval_roundtrip", eval_roundtrip),
            ),
            teacher_uses_eval_roundtrip=value.get(
                "teacher_uses_eval_roundtrip",
                value.get("teacher_eval_roundtrip", eval_roundtrip),
            ),
            same_as_scorer_input=value.get(
                "same_as_scorer_input",
                value.get("kl_uses_scorer_roundtrip", eval_roundtrip),
            ),
            student_source=value.get("student_source", "renderer_output"),
            teacher_source=value.get("teacher_source", "teacher_frames"),
        )
    raise DistillationPolicyError(f"unsupported roundtrip contract: {value!r}")


def normalize_controller_metadata(value: Any = None) -> ControllerMetadata:
    if isinstance(value, ControllerMetadata):
        return value
    if value is None or value is False or value == "":
        return ControllerMetadata()
    if isinstance(value, str):
        return ControllerMetadata(enabled=True, kind=value)
    if isinstance(value, Mapping):
        parameters = value.get("parameters")
        if parameters is None:
            parameters = {
                str(k): v
                for k, v in value.items()
                if k not in {"enabled", "kind", "name", "state", "parameters"}
            }
        return ControllerMetadata(
            enabled=value.get("enabled", True),
            kind=value.get("kind", value.get("name")),
            parameters=_freeze_metadata(parameters, field_name="controller.parameters"),
            state=_freeze_metadata(value.get("state", ()), field_name="controller.state"),
        )
    raise DistillationPolicyError(f"unsupported controller metadata: {value!r}")


def _policy_validation_errors(policy: DistillationPolicy) -> list[str]:
    errors: list[str] = []
    if policy.family == "none":
        if policy.scope != "none":
            errors.append("family='none' requires scope='none'")
        if policy.weight != 0.0:
            errors.append("family='none' requires weight=0.0")
        return errors

    if policy.scope == "none":
        errors.append(f"family={policy.family!r} requires an explicit non-none scope")
    if policy.family == "segnet_aux_kl" and policy.scope != "segnet_aux":
        errors.append("family='segnet_aux_kl' requires scope='segnet_aux'")
    if policy.family in {"segnet_kl_legacy", "jbl"} and policy.scope != "segnet_aux":
        errors.append(f"family={policy.family!r} requires scope='segnet_aux'")
    if policy.family == "primary_scorer_kl" and policy.scope != "primary_scorer":
        errors.append("family='primary_scorer_kl' requires scope='primary_scorer'")
    if policy.scope == "primary_scorer" and policy.family != "primary_scorer_kl":
        errors.append("scope='primary_scorer' is only valid for family='primary_scorer_kl'")
    if policy.allow_banned_primary and policy.family != "primary_scorer_kl":
        errors.append("allow_banned_primary is only valid for family='primary_scorer_kl'")

    if policy.family == "primary_scorer_kl":
        if not policy.allow_banned_primary:
            errors.append("primary_scorer requires allow_banned_primary=True")
        if policy.promotion_eligible:
            errors.append("primary_scorer requires promotion_eligible=False")
        if policy.forensic_reason is None:
            errors.append("primary_scorer requires a forensic_reason")

    if policy.family in {"segnet_kl_legacy", "jbl"}:
        if policy.promotion_eligible:
            errors.append(f"{policy.family} requires promotion_eligible=False")
        if policy.forensic_reason is None:
            errors.append(f"{policy.family} requires forensic_reason to be explicitly represented as forensic")

    if policy.family == "segnet_aux_kl" and policy.promotion_eligible:
        if policy.weight <= 0.0:
            errors.append("promotion-capable segnet_aux_kl requires finite positive weight")
        if policy.temperature < 2.0:
            errors.append("promotion-capable segnet_aux_kl requires temperature >= 2.0")
        if not policy.roundtrip_contract.promotion_safe:
            errors.append(
                "promotion-capable segnet_aux_kl requires eval-roundtripped student and teacher inputs "
                "matching the scorer input contract"
            )

    return errors


def _infer_family(data: Mapping[str, Any]) -> DistillationFamily:
    if "family" in data and data["family"] is not None:
        raw = _clean_string(data["family"], field_name="family")
        if raw in {"kl", "kl_distill"}:
            scope = _normalize_scope_value(data.get("scope", data.get("kl_distill_scope", "segnet_aux")))
            return "primary_scorer_kl" if scope == "primary_scorer" else "segnet_aux_kl"
        return _normalize_family_value(raw)

    weight_key = _first_present(data, ("weight", "kl_distill_weight"))
    has_positive_kl_weight = False
    if weight_key is not None:
        has_positive_kl_weight = (
            _coerce_finite_float(data.get(weight_key), field_name=weight_key, min_value=0.0) > 0.0
        )
    scoped_aux_weight = (
        has_positive_kl_weight
        and _normalize_scope_value(data.get("scope", data.get("kl_distill_scope", "none"))) == "segnet_aux"
    )

    loss_mode = _clean_optional_string(data.get("loss_mode"), field_name="loss_mode")
    if loss_mode is None:
        if scoped_aux_weight:
            return "segnet_aux_kl"
        return "none"
    loss_mode_norm = loss_mode.strip().lower().replace("-", "_")
    if loss_mode_norm in {"standard", "temperature", "focal_ste", "pcgrad", "feature_match", "posenet_embedding", "logit_margin"}:
        if scoped_aux_weight:
            return "segnet_aux_kl"
        return "none"
    if loss_mode_norm in {"kl", "kl_distill"}:
        scope = _normalize_scope_value(data.get("scope", data.get("kl_distill_scope", "none")))
        if scope == "primary_scorer":
            return "primary_scorer_kl"
        return "segnet_aux_kl"
    if loss_mode_norm in {"segnet_kl", "segnet_kl_legacy", "legacy_segnet_kl"}:
        return "segnet_kl_legacy"
    if loss_mode_norm in {"jbl", "lane_j_jbl", "j_jbl"}:
        return "jbl"
    raise DistillationPolicyError(f"unsupported loss_mode for distillation policy: {loss_mode!r}")


def _normalize_family_value(value: Any) -> DistillationFamily:
    text = _clean_string(value, field_name="family").strip().lower().replace("-", "_")
    aliases: dict[str, DistillationFamily] = {
        "": "none",
        "off": "none",
        "none": "none",
        "disabled": "none",
        "segnet_aux": "segnet_aux_kl",
        "segnet_aux_kl": "segnet_aux_kl",
        "segnet_only_kl": "segnet_aux_kl",
        "hinton_segnet_aux": "segnet_aux_kl",
        "primary": "primary_scorer_kl",
        "primary_scorer": "primary_scorer_kl",
        "primary_scorer_kl": "primary_scorer_kl",
        "kl_distill_scorer_loss": "primary_scorer_kl",
        "segnet_kl": "segnet_kl_legacy",
        "legacy_segnet_kl": "segnet_kl_legacy",
        "segnet_kl_legacy": "segnet_kl_legacy",
        "jbl": "jbl",
        "lane_j_jbl": "jbl",
        "j_jbl": "jbl",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise DistillationPolicyError(f"unsupported distillation family: {value!r}") from exc


def _normalize_scope_value(value: Any) -> DistillationScope:
    text = _clean_string(value, field_name="scope").strip().lower().replace("-", "_")
    aliases: dict[str, DistillationScope] = {
        "": "none",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "segnet": "segnet_aux",
        "segnet_aux": "segnet_aux",
        "segnet_only": "segnet_aux",
        "segnet_aux_kl": "segnet_aux",
        "primary": "primary_scorer",
        "primary_scorer": "primary_scorer",
        "primary_scorer_kl": "primary_scorer",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise DistillationPolicyError(f"unsupported distillation scope: {value!r}") from exc


def _default_scope_for_family(family: DistillationFamily) -> DistillationScope:
    if family in {"segnet_aux_kl", "segnet_kl_legacy", "jbl"}:
        return "segnet_aux"
    if family == "primary_scorer_kl":
        return "primary_scorer"
    return "none"


def _extract_class_weights_metadata(data: Mapping[str, Any]) -> ClassWeightsMetadata:
    for key in ("class_weights", "class_weights_metadata", "segnet_class_weights", "segnet_class_weights_metadata"):
        if key in data and data[key] is not None:
            return normalize_class_weights_metadata(data[key])
    return ClassWeightsMetadata()


def _extract_roundtrip_contract(data: Mapping[str, Any]) -> RoundtripContract:
    for key in ("roundtrip_contract", "student_teacher_roundtrip"):
        if key in data and data[key] is not None:
            return normalize_roundtrip_contract(data[key])
    eval_roundtrip = data.get("eval_roundtrip", True)
    return normalize_roundtrip_contract(
        {
            "eval_roundtrip": eval_roundtrip,
            "student_uses_eval_roundtrip": data.get(
                "student_uses_eval_roundtrip",
                data.get("student_eval_roundtrip", eval_roundtrip),
            ),
            "teacher_uses_eval_roundtrip": data.get(
                "teacher_uses_eval_roundtrip",
                data.get("teacher_eval_roundtrip", eval_roundtrip),
            ),
            "same_as_scorer_input": data.get(
                "same_as_scorer_input",
                data.get("kl_uses_scorer_roundtrip", eval_roundtrip),
            ),
            "student_source": data.get("student_source", "renderer_output"),
            "teacher_source": data.get("teacher_source", "teacher_frames"),
        }
    )


def _extract_controller_metadata(data: Mapping[str, Any]) -> ControllerMetadata:
    for key in ("controller", "controller_metadata"):
        if key in data and data[key] is not None:
            return normalize_controller_metadata(data[key])
    return ControllerMetadata()


def _source_to_dict(source: Any) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "model_dump") and callable(source.model_dump):
        dumped = source.model_dump()
        if not isinstance(dumped, Mapping):
            raise DistillationPolicyError("model_dump() did not return a mapping")
        return dict(dumped)

    out: dict[str, Any] = {}
    for key in _KNOWN_SOURCE_KEYS:
        if hasattr(source, key):
            out[key] = getattr(source, key)
    if out:
        return out
    raise DistillationPolicyError(f"cannot normalize distillation policy from {type(source).__name__}")


def _first_present(data: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data and data[key] is not None:
            return key
    return None


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise DistillationPolicyError(f"{field_name} must be a boolean")


def _coerce_finite_float(
    value: Any,
    *,
    field_name: str,
    min_value: float | None = None,
    gt: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DistillationPolicyError(f"{field_name} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DistillationPolicyError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(out):
        raise DistillationPolicyError(f"{field_name} must be finite")
    if min_value is not None and out < min_value:
        raise DistillationPolicyError(f"{field_name} must be >= {min_value}")
    if gt is not None and out <= gt:
        raise DistillationPolicyError(f"{field_name} must be > {gt}")
    return out


def _coerce_float_tuple(value: Any, *, field_name: str = "class_weights.values") -> tuple[float, ...] | None:
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, str):
        return _parse_float_csv(value, field_name=field_name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DistillationPolicyError(f"{field_name} must be a numeric sequence")
    out = tuple(_coerce_finite_float(v, field_name=field_name) for v in value)
    return out


def _parse_float_csv(value: str, *, field_name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise DistillationPolicyError(f"{field_name} CSV cannot be empty")
    return tuple(_coerce_finite_float(part, field_name=field_name) for part in parts)


def _coerce_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DistillationPolicyError(f"{field_name} must be a positive integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise DistillationPolicyError(f"{field_name} must be a positive integer") from exc
    if out <= 0:
        raise DistillationPolicyError(f"{field_name} must be a positive integer")
    return out


def _clean_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise DistillationPolicyError(f"{field_name} must be a string")
    return value.strip()


def _clean_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    text = _clean_string(value, field_name=field_name)
    return text or None


def _freeze_metadata(value: Any, *, field_name: str) -> MetadataItems:
    if value is None or value == ():
        return ()
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        raise DistillationPolicyError(f"{field_name} must be a mapping or tuple of pairs")

    frozen: list[tuple[str, JsonValue]] = []
    for item in items:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) != 2:
            raise DistillationPolicyError(f"{field_name} entries must be key/value pairs")
        key, raw = item
        key_text = _clean_string(key, field_name=f"{field_name}.key")
        if not key_text:
            raise DistillationPolicyError(f"{field_name} keys cannot be empty")
        frozen.append((key_text, _coerce_json_value(raw, field_name=f"{field_name}.{key_text}")))
    return tuple(sorted(frozen, key=lambda kv: kv[0]))


def _coerce_json_value(value: Any, *, field_name: str) -> JsonValue:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DistillationPolicyError(f"{field_name} must be JSON-finite")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_coerce_json_scalar(v, field_name=field_name) for v in value)
    raise DistillationPolicyError(f"{field_name} must be a JSON scalar or scalar sequence")


def _coerce_json_scalar(value: Any, *, field_name: str) -> JsonScalar:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DistillationPolicyError(f"{field_name} must be JSON-finite")
        return value
    raise DistillationPolicyError(f"{field_name} sequence entries must be JSON scalars")


def _metadata_to_dict(items: MetadataItems) -> dict[str, JsonScalar | list[JsonScalar]]:
    out: dict[str, JsonScalar | list[JsonScalar]] = {}
    for key, value in items:
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


KLPolicy = DistillationPolicy
normalize_kl_policy = normalize_distillation_policy
validate_kl_policy = validate_distillation_policy
kl_policy_to_provenance = distillation_policy_to_provenance
kl_policy_to_json = distillation_policy_to_json
kl_policy_sha256 = distillation_policy_sha256


__all__ = [
    "KL_POLICY_FORMAT",
    "KL_POLICY_SCHEMA_VERSION",
    "ClassWeightsMetadata",
    "ControllerMetadata",
    "DistillationFamily",
    "DistillationPolicy",
    "DistillationPolicyError",
    "DistillationScope",
    "KLPolicy",
    "RoundtripContract",
    "distillation_policy_to_json",
    "distillation_policy_to_provenance",
    "distillation_policy_sha256",
    "kl_policy_to_json",
    "kl_policy_to_provenance",
    "kl_policy_sha256",
    "normalize_class_weights_metadata",
    "normalize_controller_metadata",
    "normalize_distillation_policy",
    "normalize_kl_policy",
    "normalize_roundtrip_contract",
    "promotion_blockers",
    "validate_distillation_policy",
    "validate_kl_policy",
]
