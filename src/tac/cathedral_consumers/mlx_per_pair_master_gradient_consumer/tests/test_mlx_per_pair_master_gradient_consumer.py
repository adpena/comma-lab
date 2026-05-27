# SPDX-License-Identifier: MIT
"""Tests for the MLX per-pair master-gradient cathedral consumer (Catalog #335)."""
from __future__ import annotations

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import mlx_per_pair_master_gradient_consumer as mod


def test_consumer_satisfies_canonical_contract() -> None:
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True, reg.validation_errors
    assert reg.consumer_name == "mlx_per_pair_master_gradient_consumer"


def test_consumer_declares_canonical_hooks() -> None:
    assert HookNumber.SENSITIVITY_MAP in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS


def test_consume_candidate_no_sha_returns_tier_a_markers() -> None:
    out = mod.consume_candidate({})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert out["evidence_grade"] == "macOS-MLX research-signal"


def test_consume_candidate_no_artifact_returns_tier_a_markers() -> None:
    # A random sha with no landed artifact -> absent annotation, never promoted.
    out = mod.consume_candidate({"archive_sha256": "9" * 64})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert "no MLX per-pair heuristic-prior artifact" in out["rationale"]


def test_consume_candidate_with_landed_artifact(tmp_path, monkeypatch) -> None:
    import json

    import tac.master_gradient_mlx_pipeline as pipeline

    sha = "a" * 64
    manifest = tmp_path / "mlx_research_signal_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "npy_path": ".omx/state/landed.npy",
                "operating_point": {"d_seg": 0.0011, "d_pose": 0.0014},
                "evidence_grade": "macOS-MLX research-signal",
            },
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(pipeline, "MLX_RESEARCH_SIGNAL_MANIFEST_PATH", manifest)

    out = mod.consume_candidate({"archive_sha256": sha})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert "HEURISTIC PRIOR available" in out["rationale"]
    assert "d_seg=0.0011" in out["rationale"]
    # Rigor-gating clause surfaced in the annotation.
    assert "rigor-review verdict" in out["rationale"]
    assert "NON-PROMOTABLE" in out["rationale"]


def test_update_from_anchor_is_noop() -> None:
    assert mod.update_from_anchor({"any": "row"}) is None
