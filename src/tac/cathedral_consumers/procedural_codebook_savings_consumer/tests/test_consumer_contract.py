# SPDX-License-Identifier: MIT
"""Tests for procedural_codebook_savings_consumer."""

from __future__ import annotations

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import procedural_codebook_savings_consumer as M


def test_consumer_module_satisfies_canonical_contract() -> None:
    registration = validate_consumer_module(M)
    assert registration.contract_compliant is True
    assert registration.consumer_name == "procedural_codebook_savings_consumer"
    assert HookNumber.SENSITIVITY_MAP in registration.consumer_hook_numbers
    assert HookNumber.BIT_ALLOCATOR in registration.consumer_hook_numbers
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in registration.consumer_hook_numbers
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in registration.consumer_hook_numbers


def test_consume_candidate_predicts_signed_rate_savings_fail_closed() -> None:
    out = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "nscs06_v8_chroma_lut",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
            }
        }
    )

    assert out["consumer_signal_kind"] == "procedural_codebook_savings_routing"
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["bytes_saved"] == 4064
    assert out["canonical_equation_id"] == (
        "procedural_codebook_from_seed_compression_savings_v1"
    )
    assert -0.003 < out["predicted_delta_s_per_canonical_equation"] < 0.0
    assert out["actionable_above_min_bytes_saved_threshold"] is True


def test_consume_candidate_rejects_missing_or_invalid_payload() -> None:
    for candidate in (
        {},
        {"procedural_codebook_savings_candidate": {"n_codebook_bytes": 4096}},
        {"procedural_codebook_savings_candidate": {"k_seed_bytes": 32}},
        {"procedural_codebook_savings_candidate": {"n_codebook_bytes": 10, "k_seed_bytes": 0}},
    ):
        out = M.consume_candidate(candidate)
        assert out["consumer_signal_kind"] == "procedural_codebook_savings_absent"
        assert out["score_claim"] is False
        assert out["promotion_eligible"] is False


def test_consume_candidate_marks_small_savings_non_actionable() -> None:
    out = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "selector_payload",
                "n_codebook_bytes": 249,
                "k_seed_bytes": 8,
            }
        }
    )

    assert out["bytes_saved"] == 241
    assert out["actionable_above_min_bytes_saved_threshold"] is False
