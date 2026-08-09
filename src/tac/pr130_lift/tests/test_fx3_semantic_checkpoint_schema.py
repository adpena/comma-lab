from __future__ import annotations

import pytest

from tac.pr130_lift.checkpoint_schema import (
    SEMANTIC_CHECKPOINT_SCHEMA,
    SemanticCheckpointSchemaError,
    architecture_config_from_checkpoint,
    build_semantic_checkpoint_metadata,
    producing_stage_config_from_checkpoint,
)


def test_legacy_checkpoint_exposes_architecture_only() -> None:
    legacy = {
        "config": {
            "width": 96,
            "blocks": 4,
            "frame_dim": 8,
            "steps": 3_000,
            "lr": 1e-3,
            "amp": True,
            "seed": 7,
            "out": "ancestor.json",
        }
    }
    architecture = architecture_config_from_checkpoint(
        legacy, consumer="test_legacy_checkpoint_exposes_architecture_only"
    )
    assert architecture == {
        "width": 96,
        "blocks": 4,
        "frame_dim": 8,
        "num_pairs": 600,
        "num_tokens": 5,
        "phase_y": 1,
        "phase_x": 1,
        "temporal_radius": 0,
    }
    assert not ({"steps", "lr", "amp", "seed", "out"} & set(architecture))
    with pytest.raises(SemanticCheckpointSchemaError, match="architecture-only"):
        producing_stage_config_from_checkpoint(legacy, consumer="provenance_test")


def test_v2_checkpoint_separates_architecture_and_producing_stage() -> None:
    metadata = build_semantic_checkpoint_metadata(
        architecture_config={"width": 96, "blocks": 4, "frame_dim": 8},
        producing_stage_config={"steps": 6_000, "lr": 2e-7, "seed": 20260716},
        parent_checkpoint={"sha256": "a" * 64, "path": "parent.pt"},
    )
    assert metadata["schema"] == SEMANTIC_CHECKPOINT_SCHEMA
    assert architecture_config_from_checkpoint(metadata, consumer="architecture_test")[
        "blocks"
    ] == 4
    assert producing_stage_config_from_checkpoint(metadata, consumer="provenance_test") == {
        "steps": 6_000,
        "lr": 2e-7,
        "seed": 20260716,
    }
    assert "config" not in metadata


def test_v2_checkpoint_refuses_ambiguous_config_alias() -> None:
    checkpoint = {
        "schema": SEMANTIC_CHECKPOINT_SCHEMA,
        "architecture_config": {"width": 96, "blocks": 4, "frame_dim": 8},
        "producing_stage_config": {"steps": 6_000},
        "config": {"steps": 3_000},
    }
    with pytest.raises(SemanticCheckpointSchemaError, match="ambiguous"):
        architecture_config_from_checkpoint(checkpoint, consumer="bad_schema_test")


def test_unknown_schema_cannot_smuggle_architecture_config() -> None:
    checkpoint = {
        "schema": "unknown.v1",
        "architecture_config": {"width": 96, "blocks": 4, "frame_dim": 8},
    }
    with pytest.raises(SemanticCheckpointSchemaError, match="unrecognized"):
        architecture_config_from_checkpoint(checkpoint, consumer="unknown_schema_test")
