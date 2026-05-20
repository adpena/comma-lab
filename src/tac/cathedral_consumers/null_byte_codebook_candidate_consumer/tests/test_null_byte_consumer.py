# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_consumers.null_byte_codebook_candidate_consumer.

Covers canonical Protocol contract (Catalog #335) + Tier A canonical
markers (Catalog #341) + null-byte data-flow + boundary cases (zero
null bytes / all null bytes / malformed payload / file-path payload
load).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import null_byte_codebook_candidate_consumer as M


@pytest.fixture
def canonical_payload() -> dict[str, Any]:
    """Mimics the JSON shape emitted by tools/probe_null_byte_master_gradient.py."""
    return {
        "n_total_bytes": 178417,
        "n_null_bytes": 16292,
        "null_fraction": 0.0913,
        "epsilon": 1e-9,
        "per_axis_zero_counts": {
            "seg_axis_zero_count": 16638,
            "pose_axis_zero_count": 16292,
            "rate_axis_zero_count": 178417,
        },
        "grammar_detected": {
            "OUTER_MAGIC": [0, 4],
            "source_len_hdr": [4, 8],
            "source_payload": [8, 178166],
            "selector_len_hdr": [178166, 178168],
            "selector_payload": [178168, 178417],
            "selector_magic": "FEC6",
            "total_bytes": 178417,
        },
        "section_breakdown": {
            "OUTER_MAGIC": {
                "range": [0, 4],
                "length_bytes": 4,
                "n_null": 4,
                "null_fraction_within_section": 1.0,
            },
            "source_len_hdr": {
                "range": [4, 8],
                "length_bytes": 4,
                "n_null": 4,
                "null_fraction_within_section": 1.0,
            },
            "source_payload": {
                "range": [8, 178166],
                "length_bytes": 178158,
                "n_null": 16033,
                "null_fraction_within_section": 0.09,
            },
            "selector_len_hdr": {
                "range": [178166, 178168],
                "length_bytes": 2,
                "n_null": 2,
                "null_fraction_within_section": 1.0,
            },
            "selector_payload": {
                "range": [178168, 178417],
                "length_bytes": 249,
                "n_null": 249,
                "null_fraction_within_section": 1.0,
            },
        },
        "anchor_sha256": "a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc",
        "archive_zip_path": "experiments/results/.../archive.zip",
    }


def test_consumer_module_canonical_contract() -> None:
    """Catalog #335 canonical Protocol contract compliance."""
    v = validate_consumer_module(M)
    assert v.contract_compliant is True
    assert v.consumer_name == "null_byte_codebook_candidate_consumer"
    assert v.consumer_version == "0.1.0"
    assert HookNumber.SENSITIVITY_MAP in v.consumer_hook_numbers
    assert HookNumber.BIT_ALLOCATOR in v.consumer_hook_numbers
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in v.consumer_hook_numbers
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in v.consumer_hook_numbers
    assert v.validation_errors == ()


def test_consumer_consumes_master_gradient_anchors_flag() -> None:
    """Auto-trigger pattern (commit a129c8857)."""
    assert M.CONSUMES_MASTER_GRADIENT_ANCHORS is True


def test_consume_candidate_returns_canonical_markers(canonical_payload: dict[str, Any]) -> None:
    """Catalog #341 Tier A markers MUST be present in every routing-branch return."""
    out = M.consume_candidate({"null_byte_probe": canonical_payload})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_routing"


def test_consume_candidate_surfaces_signal_data(canonical_payload: dict[str, Any]) -> None:
    """Consumer payload surfaces operator-facing routing metadata."""
    out = M.consume_candidate({"null_byte_probe": canonical_payload})
    assert out["n_total_bytes"] == 178417
    assert out["n_null_bytes"] == 16292
    assert abs(out["null_fraction"] - 0.0913) < 1e-9
    assert out["epsilon"] == 1e-9
    assert out["source_anchor_sha256"].startswith("a1afce29")
    # Section top-k: sorted descending by n_null
    top = out["section_top_k"]
    assert len(top) == 5
    section_names_ordered = [row["section"] for row in top]
    assert section_names_ordered[0] == "source_payload"  # n_null=16033 (highest)
    assert section_names_ordered[-1] in {"OUTER_MAGIC", "selector_len_hdr"}


def test_consume_candidate_with_payload_path(
    tmp_path: Path, canonical_payload: dict[str, Any]
) -> None:
    """Consumer loads payload from a JSON path key if the inline mapping is absent."""
    p = tmp_path / "null_byte_probe.json"
    p.write_text(json.dumps(canonical_payload), encoding="utf-8")
    out = M.consume_candidate({"null_byte_probe_json": str(p)})
    assert out["n_null_bytes"] == 16292
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_routing"


def test_consume_candidate_no_signal_when_payload_absent() -> None:
    out = M.consume_candidate({})
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_absent"
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"


def test_consume_candidate_no_signal_when_not_mapping() -> None:
    out = M.consume_candidate("not-a-mapping")  # type: ignore[arg-type]
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_absent"


def test_consume_candidate_no_signal_when_payload_malformed() -> None:
    out = M.consume_candidate({"null_byte_probe": {"missing": "fields"}})
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_absent"


def test_consume_candidate_zero_null_bytes_boundary() -> None:
    """Boundary: zero null bytes is a valid signal (consumer must not crash)."""
    out = M.consume_candidate(
        {
            "null_byte_probe": {
                "n_total_bytes": 1000,
                "n_null_bytes": 0,
                "null_fraction": 0.0,
                "epsilon": 1e-9,
                "section_breakdown": {},
                "grammar_detected": None,
            }
        }
    )
    assert out["consumer_signal_kind"] == "null_byte_codebook_candidate_routing"
    assert out["n_null_bytes"] == 0
    assert out["section_top_k"] == []


def test_consume_candidate_all_null_bytes_boundary() -> None:
    """Boundary: all bytes null is a valid signal."""
    out = M.consume_candidate(
        {
            "null_byte_probe": {
                "n_total_bytes": 100,
                "n_null_bytes": 100,
                "null_fraction": 1.0,
                "epsilon": 1e-9,
                "section_breakdown": {
                    "all": {"range": [0, 100], "length_bytes": 100, "n_null": 100, "null_fraction_within_section": 1.0},
                },
                "grammar_detected": None,
            }
        }
    )
    assert out["n_null_bytes"] == 100
    assert out["section_top_k"][0]["section"] == "all"


def test_update_from_anchor_is_noop() -> None:
    """update_from_anchor is a no-op (probe is reconstructed per-candidate)."""
    # Should not raise on any input
    M.update_from_anchor(None)
    M.update_from_anchor({"any": "object"})
    M.update_from_anchor(object())


def test_auto_discovery_finds_this_consumer() -> None:
    """Sister regression: this consumer is auto-discovered per Catalog #335."""
    from tools.cathedral_autopilot_autonomous_loop import discover_compliant_consumer_modules

    discovered = discover_compliant_consumer_modules()
    names = {getattr(mod, "CONSUMER_NAME", "") for mod in discovered}
    assert "null_byte_codebook_candidate_consumer" in names
