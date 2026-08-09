"""Typed PR130 semantic-checkpoint metadata with legacy architecture-only reads.

Historical PR130 checkpoints stored an ancestor run's full CLI dictionary at
``checkpoint["config"]``.  The architecture fields in that dictionary are
usable, but the schedule, precision, seed, and path fields are not provenance.
This module is the only compatibility boundary: legacy dictionaries are
reduced to a validated architecture allow-list, while producing-stage
provenance is available only from the versioned schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SEMANTIC_CHECKPOINT_SCHEMA = "tac.pr130.semantic_checkpoint.v2"
ARCHITECTURE_REQUIRED_KEYS = ("width", "blocks", "frame_dim")
ARCHITECTURE_OPTIONAL_DEFAULTS = {
    "num_pairs": 600,
    "num_tokens": 5,
    "phase_y": 1,
    "phase_x": 1,
    "temporal_radius": 0,
}


class SemanticCheckpointSchemaError(ValueError):
    """Raised when checkpoint metadata is ambiguous or internally invalid."""


def _validated_architecture_config(payload: Mapping[str, Any]) -> dict[str, int]:
    missing = [key for key in ARCHITECTURE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise SemanticCheckpointSchemaError(
            f"semantic checkpoint is missing architecture keys: {missing}"
        )
    allowed = set(ARCHITECTURE_REQUIRED_KEYS) | set(ARCHITECTURE_OPTIONAL_DEFAULTS)
    config = {
        key: int(payload.get(key, default))
        for key, default in ARCHITECTURE_OPTIONAL_DEFAULTS.items()
    }
    config.update({key: int(payload[key]) for key in ARCHITECTURE_REQUIRED_KEYS})
    if config["width"] <= 0 or config["blocks"] <= 0 or config["frame_dim"] <= 0:
        raise SemanticCheckpointSchemaError(
            "width, blocks, and frame_dim must all be positive"
        )
    if config["num_pairs"] <= 0 or config["num_tokens"] <= 0:
        raise SemanticCheckpointSchemaError("num_pairs and num_tokens must be positive")
    if config["phase_y"] <= 0 or config["phase_x"] <= 0:
        raise SemanticCheckpointSchemaError("phase_y and phase_x must be positive")
    if config["temporal_radius"] < 0:
        raise SemanticCheckpointSchemaError("temporal_radius must be non-negative")
    # Never pass schedule/provenance keys through the architecture boundary.
    assert set(config) == allowed
    return config


def architecture_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    consumer: str,
) -> dict[str, int]:
    """Return only architecture facts, accepting old checkpoints explicitly.

    A v2 checkpoint must not carry the ambiguous top-level ``config`` key.
    Legacy checkpoints may carry it, but this function discards every field
    outside the architecture allow-list.  Callers that need run provenance
    must use :func:`producing_stage_config_from_checkpoint`, which refuses
    legacy objects.
    """

    if not consumer.strip():
        raise SemanticCheckpointSchemaError("consumer must be named")
    schema = checkpoint.get("schema")
    if schema == SEMANTIC_CHECKPOINT_SCHEMA:
        if "config" in checkpoint:
            raise SemanticCheckpointSchemaError(
                "v2 semantic checkpoint must not contain ambiguous top-level config"
            )
        payload = checkpoint.get("architecture_config")
        if not isinstance(payload, Mapping):
            raise SemanticCheckpointSchemaError(
                "v2 semantic checkpoint requires architecture_config"
            )
        return _validated_architecture_config(payload)
    if "architecture_config" in checkpoint:
        raise SemanticCheckpointSchemaError(
            f"unrecognized semantic checkpoint schema {schema!r} carries architecture_config"
        )
    payload = checkpoint.get("config")
    if not isinstance(payload, Mapping):
        raise SemanticCheckpointSchemaError(
            "legacy semantic checkpoint requires a mapping at top-level config"
        )
    return _validated_architecture_config(payload)


def producing_stage_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    consumer: str,
) -> dict[str, Any]:
    """Return authoritative run provenance, refusing legacy ancestor metadata."""

    if not consumer.strip():
        raise SemanticCheckpointSchemaError("consumer must be named")
    if checkpoint.get("schema") != SEMANTIC_CHECKPOINT_SCHEMA:
        raise SemanticCheckpointSchemaError(
            "legacy checkpoint config is architecture-only and must not be used as "
            "producing-stage provenance"
        )
    if "config" in checkpoint:
        raise SemanticCheckpointSchemaError(
            "v2 semantic checkpoint must not contain ambiguous top-level config"
        )
    payload = checkpoint.get("producing_stage_config")
    if not isinstance(payload, Mapping):
        raise SemanticCheckpointSchemaError(
            "v2 semantic checkpoint requires producing_stage_config"
        )
    return dict(payload)


def build_semantic_checkpoint_metadata(
    *,
    architecture_config: Mapping[str, Any],
    producing_stage_config: Mapping[str, Any],
    parent_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the unambiguous metadata prefix shared by deploy/resume checkpoints."""

    architecture = _validated_architecture_config(architecture_config)
    if not producing_stage_config:
        raise SemanticCheckpointSchemaError("producing_stage_config must not be empty")
    if not parent_checkpoint.get("sha256"):
        raise SemanticCheckpointSchemaError("parent_checkpoint.sha256 is required")
    return {
        "schema": SEMANTIC_CHECKPOINT_SCHEMA,
        "architecture_config": architecture,
        "producing_stage_config": dict(producing_stage_config),
        "parent_checkpoint": dict(parent_checkpoint),
    }


__all__ = [
    "ARCHITECTURE_OPTIONAL_DEFAULTS",
    "ARCHITECTURE_REQUIRED_KEYS",
    "SEMANTIC_CHECKPOINT_SCHEMA",
    "SemanticCheckpointSchemaError",
    "architecture_config_from_checkpoint",
    "build_semantic_checkpoint_metadata",
    "producing_stage_config_from_checkpoint",
]
