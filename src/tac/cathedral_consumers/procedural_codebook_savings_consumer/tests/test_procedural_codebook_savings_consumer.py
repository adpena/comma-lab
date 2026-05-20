# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_consumers.procedural_codebook_savings_consumer.

Covers canonical Protocol contract (Catalog #335) + Tier A canonical
markers (Catalog #341) + canonical-equation savings prediction +
boundary cases.
"""
from __future__ import annotations

from typing import Any

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import (
    procedural_codebook_savings_consumer as M,
)


def test_consumer_module_satisfies_canonical_protocol() -> None:
    """Sister of Catalog #335 — auto-discovery contract compliance."""
    registration = validate_consumer_module(M)
    assert registration.contract_compliant is True, registration.validation_errors
    assert registration.consumer_name == "procedural_codebook_savings_consumer"
    assert registration.consumer_version == "0.1.0"


def test_consumer_hook_numbers_canonical_set() -> None:
    """Hooks #1 + #3 + #4 + #5 per Catalog #125 6-hook wire-in."""
    assert set(M.CONSUMER_HOOK_NUMBERS) == {
        HookNumber.SENSITIVITY_MAP,
        HookNumber.BIT_ALLOCATOR,
        HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
        HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    }


def test_consumes_master_gradient_anchors_marker() -> None:
    """Opt-in to master-gradient anchor event stream (sister auto-trigger pattern)."""
    assert getattr(M, "CONSUMES_MASTER_GRADIENT_ANCHORS", False) is True


def test_update_from_anchor_is_no_op() -> None:
    """Hook #5 entrypoint accepts arbitrary anchor; does not raise."""
    M.update_from_anchor({"any": "shape"})
    M.update_from_anchor(None)
    M.update_from_anchor("string")


# ---------------------------------------------------------------------------
# consume_candidate happy path + Tier A canonical markers
# ---------------------------------------------------------------------------


def _canonical_candidate(**overrides: Any) -> dict[str, Any]:
    """Build a canonical procedural-codebook savings candidate row."""
    payload = {
        "substrate_id": "nscs06_v8_chroma_lut",
        "n_codebook_bytes": 4096,
        "k_seed_bytes": 32,
        "generator_kind": "pcg64",
    }
    payload.update(overrides.get("payload_overrides", {}))
    return {
        "procedural_codebook_savings_candidate": payload,
        **{k: v for k, v in overrides.items() if k != "payload_overrides"},
    }


def test_consume_candidate_tier_a_markers_present() -> None:
    """Tier A canonical routing markers per Catalog #341."""
    row = M.consume_candidate(_canonical_candidate())
    assert row["predicted_delta_adjustment"] == 0.0
    assert row["promotable"] is False
    assert row["axis_tag"] == "[predicted]"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_routing"


def test_consume_candidate_predicts_canonical_formula_savings() -> None:
    """Per canonical equation procedural_codebook_from_seed_compression_savings_v1.

    NSCS06 v8 chroma LUT 4096 B → 32 B seed = 4064 B saved
    → predicted ΔS = -25 * 4064 / 37_545_489 = -0.00270589
    """
    row = M.consume_candidate(_canonical_candidate())
    assert row["bytes_saved"] == 4064
    expected_delta_s = -25.0 * 4064 / 37_545_489
    assert (
        abs(row["predicted_delta_s_per_canonical_equation"] - expected_delta_s)
        < 1e-9
    )
    assert row["actionable_above_min_bytes_saved_threshold"] is True
    assert (
        row["canonical_equation_id"]
        == "procedural_codebook_from_seed_compression_savings_v1"
    )


def test_consume_candidate_surfaces_substrate_metadata() -> None:
    """Operator-facing summary surfaces present."""
    row = M.consume_candidate(_canonical_candidate())
    assert row["substrate_id"] == "nscs06_v8_chroma_lut"
    assert row["n_codebook_bytes"] == 4096
    assert row["k_seed_bytes"] == 32
    assert row["generator_kind"] == "pcg64"


def test_consume_candidate_includes_compliance_citation_chain() -> None:
    """Cite-chain to memo Q4 + sister catalogs per Catalog #305 observability."""
    row = M.consume_candidate(_canonical_candidate())
    chain = row["compliance_citation_chain"]
    assert "upstream/evaluate.py:63" in chain
    assert "memo_q4_structurally_compliant_verdict" in chain
    assert "catalog_213" in chain
    assert "catalog_272" in chain
    assert "catalog_318" in chain


def test_consume_candidate_documents_promotion_gates() -> None:
    """Promotion-gating cite-chain per CLAUDE.md Substrate scaffolds non-negotiable."""
    row = M.consume_candidate(_canonical_candidate())
    gates = row["promotion_gates"]
    assert "catalog_325" in gates
    assert "catalog_324" in gates
    assert "catalog_272" in gates


# ---------------------------------------------------------------------------
# Boundary cases — small savings below actionable threshold
# ---------------------------------------------------------------------------


def test_consume_candidate_marks_small_savings_not_actionable() -> None:
    """Per memo §4: FEC6 selector_payload 249 B saves only -0.000166 ΔS = below threshold."""
    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "fec6_selector_payload",
                "n_codebook_bytes": 249,
                "k_seed_bytes": 16,
                "generator_kind": "lcg",
            }
        }
    )
    assert row["bytes_saved"] == 233  # 249 - 16
    assert row["actionable_above_min_bytes_saved_threshold"] is False


def test_consume_candidate_clamps_zero_when_seed_larger_than_codebook() -> None:
    """Pathological: seed larger than codebook → bytes_saved=0 (no savings)."""
    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "tiny_codebook",
                "n_codebook_bytes": 32,
                "k_seed_bytes": 64,
                "generator_kind": "pcg64",
            }
        }
    )
    assert row["bytes_saved"] == 0
    assert row["predicted_delta_s_per_canonical_equation"] == 0.0
    assert row["actionable_above_min_bytes_saved_threshold"] is False


# ---------------------------------------------------------------------------
# No-signal paths
# ---------------------------------------------------------------------------


def test_consume_candidate_no_payload_returns_no_signal() -> None:
    """Missing payload returns observability-only no-signal (does not raise)."""
    row = M.consume_candidate({})
    assert row["predicted_delta_adjustment"] == 0.0
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_absent"
    assert "no procedural_codebook_savings_candidate payload" in row["rationale"]


def test_consume_candidate_non_mapping_returns_no_signal() -> None:
    """Non-mapping candidate returns observability-only no-signal."""
    row = M.consume_candidate("not a mapping")  # type: ignore[arg-type]
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_absent"
    assert "not a mapping" in row["rationale"]


def test_consume_candidate_missing_n_codebook_bytes_no_signal() -> None:
    """Missing n_codebook_bytes returns no-signal."""
    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "test",
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
            }
        }
    )
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_absent"


def test_consume_candidate_missing_k_seed_bytes_no_signal() -> None:
    """Missing k_seed_bytes returns no-signal."""
    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "test",
                "n_codebook_bytes": 4096,
                "generator_kind": "pcg64",
            }
        }
    )
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_absent"


def test_consume_candidate_alternate_payload_keys_accepted() -> None:
    """Sister keys procedural_codebook_candidate + seed_derived_codebook_candidate."""
    for key in (
        "procedural_codebook_candidate",
        "seed_derived_codebook_candidate",
    ):
        row = M.consume_candidate(
            {
                key: {
                    "substrate_id": "atw_v2_codec",
                    "n_codebook_bytes": 3072,
                    "k_seed_bytes": 32,
                    "generator_kind": "pcg64",
                }
            }
        )
        assert row["consumer_signal_kind"] == "procedural_codebook_savings_routing"
        assert row["substrate_id"] == "atw_v2_codec"
