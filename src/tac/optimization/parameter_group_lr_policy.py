# SPDX-License-Identifier: MIT
"""Planning-only parameter-group LR policy helpers.

The embedding-Theta1 policy is an optimizer-recipe planning primitive. It
classifies generic ``(name, shape)`` records instead of framework tensors so
PyTorch and MLX callers can share the same grouping fingerprint without giving
the helper score, dispatch, rank/kill, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import (
    CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    PROXY_DISPATCH_BLOCKERS,
    PROXY_FALSE_AUTHORITY_FIELDS,
    PROXY_TARGET_MODES,
    ordered_unique,
)

PARAMETER_GROUP_LR_POLICY_SCHEMA = "parameter_group_lr_policy.v1"
PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA = "parameter_group_lr_policy_fingerprint.v1"
PARAMETER_GROUP_CLASSIFICATION_SCHEMA = "parameter_group_lr_policy_classification.v1"

ParameterClass = Literal[
    "embedding_like",
    "hidden_matrix",
    "head_scalar_norm",
    "unclassified",
]

PARAMETER_CLASSES: tuple[ParameterClass, ...] = (
    "embedding_like",
    "hidden_matrix",
    "head_scalar_norm",
    "unclassified",
)

EMBEDDING_THETA1_POLICY_ID = "embedding_theta1_hidden_muon_adamw"
DEFAULT_POLICY_ID = "single_group_baseline"

DEFAULT_PARAMETER_GROUP_LR_POLICY: dict[str, Any] = {
    "schema": PARAMETER_GROUP_LR_POLICY_SCHEMA,
    "policy_id": DEFAULT_POLICY_ID,
    "embedding_lr_scaling_policy": "same_as_base_lr",
    "width_basis": "not_width_scaled",
    "embedding_param_patterns": [],
    "hidden_param_patterns": ["*"],
    "optimizer_assignment": {"all_params": "primary_optimizer"},
    "source_refs": [],
    "falsification_probe": "paired_same_seed_parameter_group_lr_ablation",
}

EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY: dict[str, Any] = {
    "schema": PARAMETER_GROUP_LR_POLICY_SCHEMA,
    "policy_id": EMBEDDING_THETA1_POLICY_ID,
    "embedding_lr_scaling_policy": "theta_1_not_inverse_width",
    "width_basis": "hidden_width_or_latent_channel_count",
    "embedding_param_patterns": [
        "embed",
        "embedding",
        "latent",
        "codebook",
        "pos",
        "position",
        "frame_embedding",
        "pair_embedding",
    ],
    "hidden_param_patterns": ["matrix_param_ndim_ge_2_non_embedding"],
    "optimizer_assignment": {
        "embedding_like": "AdamW",
        "hidden_matrix": "Muon",
        "head_scalar_norm": "AdamW",
    },
    "source_refs": [
        "x:maximelabonne/status/2057602654151364899",
        "arxiv:2605.21486",
        "github:KellerJordan/modded-nanogpt/records/track_3_optimization",
    ],
    "falsification_probe": "switch_embedding_lr_same_seed_archive_aware_smoke",
}

FALSE_AUTHORITY_FIELDS: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "score_claim_valid": False,
}
POLICY_AUTHORITY_FIELDS: frozenset[str] = frozenset(
    [*FALSE_AUTHORITY_FIELDS, *CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS]
)

PARAMETER_GROUP_POLICY_DISPATCH_BLOCKERS: tuple[str, ...] = (
    "parameter_group_lr_policy_is_planning_only",
    "requires_same_seed_parameter_group_lr_ablation_before_recipe_authority",
    "requires_byte_closed_archive_export_before_dispatch_readiness",
    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
    "requires_exact_auth_eval_result_before_score_claim",
)

_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_EMBED_ALIAS_TOKENS: dict[str, frozenset[str]] = {
    "embed": frozenset({"embed", "embeds", "embedding", "embeddings"}),
    "embedding": frozenset({"embed", "embeds", "embedding", "embeddings"}),
    "latent": frozenset({"latent", "latents"}),
    "codebook": frozenset({"codebook", "codebooks"}),
    "pos": frozenset({"pos", "positional"}),
    "position": frozenset({"position", "positions", "positional"}),
}
_HEAD_SCALAR_NORM_PATTERNS: tuple[str, ...] = (
    "head",
    "lm_head",
    "classifier",
    "class_head",
    "logit",
    "logits",
    "readout",
    "norm",
    "layernorm",
    "rmsnorm",
    "ln",
    "bn",
    "bias",
    "scale",
    "gamma",
    "beta",
    "gain",
)


class ParameterGroupLRPolicyError(ValueError):
    """Raised when policy input would blur planning or grouping semantics."""


@dataclass(frozen=True)
class ParameterShapeRecord:
    """Framework-neutral parameter metadata consumed by LR policy helpers."""

    name: str
    shape: tuple[int, ...] | None

    @classmethod
    def from_record(cls, record: ParameterShapeRecord | Mapping[str, Any] | Sequence[Any]) -> ParameterShapeRecord:
        if isinstance(record, ParameterShapeRecord):
            return record
        if isinstance(record, Mapping):
            if "name" not in record:
                raise ParameterGroupLRPolicyError("parameter record missing name")
            name = record["name"]
            shape = record.get("shape")
        elif isinstance(record, Sequence) and not isinstance(record, str | bytes):
            if len(record) != 2:
                raise ParameterGroupLRPolicyError("tuple parameter records must be (name, shape)")
            name, shape = record
        else:
            raise ParameterGroupLRPolicyError(
                "parameter records must be ParameterShapeRecord, mapping, or (name, shape)"
            )

        text_name = str(name).strip()
        if not text_name:
            raise ParameterGroupLRPolicyError("parameter record name must be non-empty")
        return cls(name=text_name, shape=_normalize_shape(shape))

    @property
    def shape_rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    @property
    def parameter_count(self) -> int | None:
        if self.shape is None:
            return None
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": None if self.shape is None else list(self.shape),
            "shape_rank": self.shape_rank,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class ParameterClassification:
    """One deterministic grouping decision for a parameter-shape record."""

    name: str
    shape: tuple[int, ...] | None
    parameter_class: ParameterClass
    optimizer: str
    lr_scaling_policy: str
    reason: str
    policy_id: str
    schema: str = PARAMETER_GROUP_CLASSIFICATION_SCHEMA

    @property
    def shape_rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    @property
    def parameter_count(self) -> int | None:
        if self.shape is None:
            return None
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "shape": None if self.shape is None else list(self.shape),
            "shape_rank": self.shape_rank,
            "parameter_count": self.parameter_count,
            "parameter_class": self.parameter_class,
            "optimizer": self.optimizer,
            "lr_scaling_policy": self.lr_scaling_policy,
            "reason": self.reason,
            "policy_id": self.policy_id,
        }


def _normalize_shape(shape: Any) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if isinstance(shape, bool | str | bytes):
        raise ParameterGroupLRPolicyError("shape must be a sequence of non-negative integers")
    if isinstance(shape, int):
        dims = (shape,)
    else:
        try:
            dims = tuple(shape)
        except TypeError as exc:
            raise ParameterGroupLRPolicyError("shape must be a sequence of non-negative integers") from exc

    normalized: list[int] = []
    for dim in dims:
        if isinstance(dim, bool):
            raise ParameterGroupLRPolicyError("shape dimensions must be integers, not bools")
        try:
            dim_int = operator.index(dim)
        except TypeError as exc:
            raise ParameterGroupLRPolicyError("shape dimensions must be integers") from exc
        if dim_int < 0:
            raise ParameterGroupLRPolicyError("shape dimensions must be non-negative")
        normalized.append(dim_int)
    return tuple(normalized)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParameterGroupLRPolicyError("non-finite float is not JSON-safe")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, int | str | bool) or value is None:
        return value
    raise ParameterGroupLRPolicyError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable canonical JSON for policy/fingerprint payloads."""

    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def parameter_group_lr_policy_sha256(policy: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 for a validated LR policy payload."""

    validate_parameter_group_lr_policy(policy)
    return hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()


def _normalized_name_parts(name: str) -> tuple[str, tuple[str, ...]]:
    normalized = _NAME_SPLIT_RE.sub("_", name.lower()).strip("_")
    return normalized, tuple(token for token in normalized.split("_") if token)


def _matches_pattern(name: str, pattern: str) -> bool:
    normalized, tokens = _normalized_name_parts(name)
    pattern_text = _NAME_SPLIT_RE.sub("_", pattern.lower()).strip("_")
    if not pattern_text:
        return False
    if "_" in pattern_text:
        return pattern_text in normalized
    aliases = _EMBED_ALIAS_TOKENS.get(pattern_text)
    if aliases is not None:
        return any(token in aliases for token in tokens)
    return pattern_text in tokens


def _embedding_pattern_match(name: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        if _matches_pattern(name, pattern):
            return pattern
    return None


def _hidden_pattern_match(record: ParameterShapeRecord, patterns: Iterable[str]) -> str | None:
    if record.shape is None or len(record.shape) < 2:
        return None
    for pattern in patterns:
        if pattern == "matrix_param_ndim_ge_2_non_embedding":
            return pattern
        if pattern == "*":
            return pattern
        if _matches_pattern(record.name, pattern):
            return pattern
    return None


def _head_scalar_norm_pattern_match(name: str) -> str | None:
    normalized, tokens = _normalized_name_parts(name)
    for pattern in _HEAD_SCALAR_NORM_PATTERNS:
        pattern_text = _NAME_SPLIT_RE.sub("_", pattern.lower()).strip("_")
        if "_" in pattern_text and pattern_text in normalized:
            return pattern
        if pattern_text in tokens:
            return pattern
    return None


def _require_policy_authority_fields_exact_false(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            if key_text in POLICY_AUTHORITY_FIELDS and inner is not False:
                raise ParameterGroupLRPolicyError(
                    f"parameter_group_lr_policy {next_path} must be false"
                )
            _require_policy_authority_fields_exact_false(inner, path=next_path)
    elif isinstance(value, list | tuple):
        for index, inner in enumerate(value):
            _require_policy_authority_fields_exact_false(inner, path=f"{path}[{index}]")


def validate_parameter_group_lr_policy(policy: Mapping[str, Any]) -> None:
    """Fail closed when an LR policy is malformed or authority-bearing."""

    _require_policy_authority_fields_exact_false(policy, path="")
    if policy.get("schema") != PARAMETER_GROUP_LR_POLICY_SCHEMA:
        raise ParameterGroupLRPolicyError("parameter_group_lr_policy schema mismatch")
    for key in (
        "policy_id",
        "embedding_lr_scaling_policy",
        "width_basis",
        "falsification_probe",
    ):
        if not isinstance(policy.get(key), str) or not str(policy.get(key)).strip():
            raise ParameterGroupLRPolicyError(
                f"parameter_group_lr_policy {key} must be a non-empty string"
            )
    for key in ("embedding_param_patterns", "hidden_param_patterns", "source_refs"):
        value = policy.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ParameterGroupLRPolicyError(
                f"parameter_group_lr_policy {key} must be a list of strings"
            )
    optimizer_assignment = policy.get("optimizer_assignment")
    if not isinstance(optimizer_assignment, Mapping):
        raise ParameterGroupLRPolicyError(
            "parameter_group_lr_policy optimizer_assignment must be a mapping"
        )
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in optimizer_assignment.items()):
        raise ParameterGroupLRPolicyError(
            "parameter_group_lr_policy optimizer_assignment must map strings to strings"
        )


def classify_parameter_record(
    record: ParameterShapeRecord | Mapping[str, Any] | Sequence[Any],
    *,
    policy: Mapping[str, Any] = EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
) -> ParameterClassification:
    """Classify one generic parameter ``(name, shape)`` record."""

    validate_parameter_group_lr_policy(policy)
    normalized = ParameterShapeRecord.from_record(record)
    policy_id = str(policy["policy_id"])
    optimizer_assignment = policy["optimizer_assignment"]

    if normalized.shape is None:
        parameter_class: ParameterClass = "unclassified"
        reason = "shape_unknown"
    else:
        embedding_match = _embedding_pattern_match(
            normalized.name,
            policy.get("embedding_param_patterns", ()),
        )
        head_match = _head_scalar_norm_pattern_match(normalized.name)
        if embedding_match is not None:
            parameter_class = "embedding_like"
            reason = f"name_matches_embedding_pattern:{embedding_match}"
        elif head_match is not None:
            parameter_class = "head_scalar_norm"
            reason = f"name_matches_head_scalar_norm_pattern:{head_match}"
        elif len(normalized.shape) <= 1:
            parameter_class = "head_scalar_norm"
            reason = "rank_le_1_scalar_or_vector_adamw"
        else:
            hidden_match = _hidden_pattern_match(
                normalized,
                policy.get("hidden_param_patterns", ()),
            )
            if hidden_match is not None:
                parameter_class = "hidden_matrix"
                reason = f"name_or_shape_matches_hidden_pattern:{hidden_match}"
            else:
                parameter_class = "unclassified"
                reason = "no_hidden_param_pattern_matched"

    optimizer = _optimizer_for(parameter_class, optimizer_assignment)
    return ParameterClassification(
        name=normalized.name,
        shape=normalized.shape,
        parameter_class=parameter_class,
        optimizer=optimizer,
        lr_scaling_policy=_lr_scaling_policy_for(parameter_class, policy),
        reason=reason,
        policy_id=policy_id,
    )


def _optimizer_for(parameter_class: ParameterClass, optimizer_assignment: Mapping[str, Any]) -> str:
    assigned = optimizer_assignment.get(parameter_class)
    if assigned is None:
        assigned = optimizer_assignment.get("all_params")
    return str(assigned or "manual_review")


def _lr_scaling_policy_for(parameter_class: ParameterClass, policy: Mapping[str, Any]) -> str:
    optimizer_assignment = policy.get("optimizer_assignment")
    if isinstance(optimizer_assignment, Mapping) and "all_params" in optimizer_assignment:
        return str(policy["embedding_lr_scaling_policy"])
    if parameter_class == "embedding_like":
        return str(policy["embedding_lr_scaling_policy"])
    if parameter_class == "hidden_matrix":
        return "optimizer_default_hidden_matrix_policy"
    if parameter_class == "head_scalar_norm":
        return "same_as_base_lr"
    return "unassigned_manual_review"


def classify_parameter_records(
    records: Iterable[ParameterShapeRecord | Mapping[str, Any] | Sequence[Any]],
    *,
    policy: Mapping[str, Any] = EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
) -> tuple[ParameterClassification, ...]:
    """Classify records sorted by name for deterministic PyTorch/MLX parity."""

    normalized = tuple(ParameterShapeRecord.from_record(record) for record in records)
    names = [record.name for record in normalized]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ParameterGroupLRPolicyError("duplicate parameter names: " + ", ".join(duplicates))
    return tuple(
        classify_parameter_record(record, policy=policy)
        for record in sorted(normalized, key=lambda item: item.name)
    )


def summarize_parameter_classifications(
    classifications: Iterable[ParameterClassification],
) -> dict[str, dict[str, int] | int]:
    """Return deterministic class/count summaries for classified records."""

    class_counts = dict.fromkeys(PARAMETER_CLASSES, 0)
    known_parameter_counts = dict.fromkeys(PARAMETER_CLASSES, 0)
    unknown_shape_count = 0
    record_count = 0
    for classification in classifications:
        record_count += 1
        class_counts[classification.parameter_class] += 1
        parameter_count = classification.parameter_count
        if parameter_count is None:
            unknown_shape_count += 1
        else:
            known_parameter_counts[classification.parameter_class] += parameter_count
    return {
        "record_count": record_count,
        "class_counts": class_counts,
        "known_parameter_counts": known_parameter_counts,
        "unknown_shape_count": unknown_shape_count,
    }


def build_parameter_group_lr_policy_fingerprint(
    records: Iterable[ParameterShapeRecord | Mapping[str, Any] | Sequence[Any]],
    *,
    policy: Mapping[str, Any] = EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
) -> dict[str, Any]:
    """Return a JSON-safe, planning-only fingerprint for parameter grouping."""

    classifications = classify_parameter_records(records, policy=policy)
    summary = summarize_parameter_classifications(classifications)
    policy_sha256 = parameter_group_lr_policy_sha256(policy)
    payload: dict[str, Any] = {
        "schema": PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA,
        "policy": _json_safe(policy),
        "policy_id": str(policy["policy_id"]),
        "policy_sha256": policy_sha256,
        "classification_records": [classification.to_json() for classification in classifications],
        "record_count": summary["record_count"],
        "class_counts": summary["class_counts"],
        "known_parameter_counts": summary["known_parameter_counts"],
        "unknown_shape_count": summary["unknown_shape_count"],
        "planning_only": True,
        "rank_score_field": "parameter_group_lr_policy_planning_not_score",
        "target_modes": list(PROXY_TARGET_MODES),
        "dispatch_blockers": list(
            ordered_unique([*PROXY_DISPATCH_BLOCKERS, *PARAMETER_GROUP_POLICY_DISPATCH_BLOCKERS])
        ),
        "false_authority": dict(FALSE_AUTHORITY_FIELDS),
        **FALSE_AUTHORITY_FIELDS,
    }
    payload["fingerprint_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def parameter_group_lr_policy_fingerprint_sha256(
    records: Iterable[ParameterShapeRecord | Mapping[str, Any] | Sequence[Any]],
    *,
    policy: Mapping[str, Any] = EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
) -> str:
    """Return only the deterministic grouping fingerprint SHA-256."""

    return str(
        build_parameter_group_lr_policy_fingerprint(records, policy=policy)[
            "fingerprint_sha256"
        ]
    )


__all__ = [
    "DEFAULT_PARAMETER_GROUP_LR_POLICY",
    "DEFAULT_POLICY_ID",
    "EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY",
    "EMBEDDING_THETA1_POLICY_ID",
    "FALSE_AUTHORITY_FIELDS",
    "PARAMETER_CLASSES",
    "PARAMETER_GROUP_CLASSIFICATION_SCHEMA",
    "PARAMETER_GROUP_LR_POLICY_FINGERPRINT_SCHEMA",
    "PARAMETER_GROUP_LR_POLICY_SCHEMA",
    "PARAMETER_GROUP_POLICY_DISPATCH_BLOCKERS",
    "ParameterClass",
    "ParameterClassification",
    "ParameterGroupLRPolicyError",
    "ParameterShapeRecord",
    "build_parameter_group_lr_policy_fingerprint",
    "canonical_json",
    "classify_parameter_record",
    "classify_parameter_records",
    "parameter_group_lr_policy_fingerprint_sha256",
    "parameter_group_lr_policy_sha256",
    "summarize_parameter_classifications",
    "validate_parameter_group_lr_policy",
]
