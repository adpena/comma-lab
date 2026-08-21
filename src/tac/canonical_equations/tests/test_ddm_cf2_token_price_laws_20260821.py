# SPDX-License-Identifier: MIT
"""Tests for the ddm_cf2 token-field pricing laws.

Two things these tests exist to prove, both of which have bitten this registry before:

1. **The evaluators do WORK** (NO-FAKE class 1/2).  Every assertion below exercises the
   arithmetic on real inputs; none checks a constant or a marker.  Replacing any function
   body with ``return CONSTANT`` fails at least one test here.
2. **Every anchor JSON-round-trips exactly** (the #1149 red).  The anchors are re-read
   through the registry's own ``_equation_from_dict`` and compared as complete JSON
   objects, plus audited by ``audit_empirical_anchor_roundtrip_fidelity`` after a real
   locked write to a temp ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_cf2_token_price_laws_20260821 import (
    ALL_CF2_TOKEN_PRICE_BUILDERS,
    AWAY_FROM_ARGMAX,
    AWAY_TRUST_VS_ACTUAL_PRICE_BAND,
    JG1_LOGIT_RANKER_BITS_PER_TOKEN,
    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
    JG3_CONFIGURATION_BITS_PER_TOKEN,
    JG5_SHIPPED_455_BITS_PER_TOKEN,
    TOWARD_ARGMAX,
    TOWARD_TRUST_VS_MODEL_U7P75,
    TOWARD_TRUST_VS_MODEL_U12,
    bits_per_token,
    direction_trust_factor,
    greedy_margin_degrades_both_terms,
    marginal_over_average_ratio,
    populate_cf2_token_price_laws,
    predict_realized_bits_per_token,
    predict_reselection_delta_bits,
    ranker_relative_ratio,
    second_order_recapture_fraction,
)
from tac.canonical_equations.registry import (
    _equation_from_dict,
    audit_empirical_anchor_roundtrip_fidelity,
    query_equations,
)

JG5_RECEIPT = Path(
    "/Volumes/APDataStore/pact/ddm_jg5/retained/final/S1_encode_jg5_subset455.json"
)
MIRROR_RECEIPT = Path(
    "/Volumes/APDataStore/pact/ddm_fs3/reencode/retained/S1_encode_fs3_drop137.json"
)


# ---------------------------------------------------------------------------
# The shared instrument.
# ---------------------------------------------------------------------------


def test_bits_per_token_is_real_arithmetic():
    assert bits_per_token(1, 8) == pytest.approx(1.0)
    assert bits_per_token(100, 10) == pytest.approx(80.0)
    # Sign is preserved: a REMOVAL credits negative bytes.
    assert bits_per_token(-37, 1440) == pytest.approx(-0.20555555, rel=1e-6)


def test_bits_per_token_refuses_zero_denominator():
    with pytest.raises(ValueError, match="denominator"):
        bits_per_token(664, 0)


# ---------------------------------------------------------------------------
# Law (a) -- direction dependence.
# ---------------------------------------------------------------------------


def test_away_band_is_derived_from_the_two_measured_objects():
    low, high = AWAY_TRUST_VS_ACTUAL_PRICE_BAND
    assert low == pytest.approx(
        JG3_CONFIGURATION_BITS_PER_TOKEN / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN
    )
    assert high == pytest.approx(
        JG5_SHIPPED_455_BITS_PER_TOKEN / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN
    )
    # The published statements: an 8.50% overcharge on jg3's configuration, 7.84% on jg5's
    # shipped set -- i.e. the model overcharges away-from-argmax by at most ~12%.
    assert low == pytest.approx(0.921667, rel=1e-5)
    assert high == pytest.approx(0.927355, rel=1e-5)
    assert (
        pytest.approx(1.084990, rel=1e-5)
    ) == JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN / JG3_CONFIGURATION_BITS_PER_TOKEN


def test_direction_dependence_is_an_order_of_magnitude():
    away = direction_trust_factor(AWAY_FROM_ARGMAX)
    toward = direction_trust_factor(TOWARD_ARGMAX)
    assert away / toward > 10.0
    assert toward == pytest.approx(TOWARD_TRUST_VS_MODEL_U7P75)
    assert pytest.approx(0.087226, rel=1e-5) == TOWARD_TRUST_VS_MODEL_U7P75
    assert TOWARD_TRUST_VS_MODEL_U12 < 0.0


def test_direction_trust_factor_refuses_an_unmeasured_direction():
    """There is no third direction; defaulting one silently would be the fake."""
    with pytest.raises(ValueError, match="must be one of"):
        direction_trust_factor("sideways")
    with pytest.raises(ValueError):
        direction_trust_factor("")


def test_predict_realized_scales_with_its_input():
    """A marker-only implementation returning a constant fails here."""
    a = predict_realized_bits_per_token(4.0, AWAY_FROM_ARGMAX)
    b = predict_realized_bits_per_token(8.0, AWAY_FROM_ARGMAX)
    assert b == pytest.approx(2.0 * a)
    assert a == pytest.approx(4.0 * direction_trust_factor(AWAY_FROM_ARGMAX))
    # Direction is load-bearing, not decorative.
    assert predict_realized_bits_per_token(4.0, TOWARD_ARGMAX) != pytest.approx(a)


def test_second_order_recapture_reproduces_the_published_figures():
    # u = 7.75: the memo's 91.3%.
    assert second_order_recapture_fraction(1022, 11716.7) == pytest.approx(0.913, abs=5e-4)
    # u = 12.0: recapture EXCEEDS 100% -- the substitution costs bytes.
    assert second_order_recapture_fraction(-37, 2546.1) > 1.0
    # An away-direction edit realizes most of its price, so recapture is small.
    assert second_order_recapture_fraction(
        JG5_SHIPPED_455_BITS_PER_TOKEN, JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN
    ) == pytest.approx(0.0726, abs=1e-3)


def test_second_order_recapture_refuses_zero_denominator():
    with pytest.raises(ValueError):
        second_order_recapture_fraction(1.0, 0.0)


def test_ranker_relative_ratio_is_named_apart_from_the_price_ratio():
    """jg5's published realized_over_modelled is RANKER-relative, not a trust factor."""
    ranker = ranker_relative_ratio(JG5_SHIPPED_455_BITS_PER_TOKEN)
    assert ranker == pytest.approx(0.813332, rel=1e-5)
    price = JG5_SHIPPED_455_BITS_PER_TOKEN / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN
    assert price == pytest.approx(0.927355, rel=1e-5)
    # The two answer different questions and must never be conflated (rv17 W2-F14).
    assert abs(ranker - price) > 0.1
    assert JG1_LOGIT_RANKER_BITS_PER_TOKEN != JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN


# ---------------------------------------------------------------------------
# Law (b) -- average is not marginal.
# ---------------------------------------------------------------------------


def test_marginal_over_average_reproduces_the_measured_2_24x():
    add_marginal = bits_per_token(223, 300)
    assert add_marginal == pytest.approx(5.9467, abs=1e-4)
    assert marginal_over_average_ratio(add_marginal, 2.6573) == pytest.approx(2.24, abs=5e-3)


def test_removal_mirror_reproduces_the_measured_constants():
    remove_marginal = bits_per_token(664, 997)
    trimmed_average = bits_per_token(3487, 7657)
    assert remove_marginal == pytest.approx(5.3280, abs=1e-4)
    assert trimmed_average == pytest.approx(3.6432, abs=1e-4)
    assert ranker_relative_ratio(trimmed_average) == pytest.approx(0.7722, abs=1e-4)
    # Same direction as the ADD side: the margin is dearer than the set it belongs to.
    assert marginal_over_average_ratio(remove_marginal, trimmed_average) > 1.0


def test_reselection_delta_prices_only_its_delta_tokens():
    assert predict_reselection_delta_bits(300, 5.9467) == pytest.approx(1784.01, abs=0.1)
    # Round 1's error, reproduced: pricing the same +300 at the set average under-charges
    # by ~987 bits (~123 B), which is what turned a refusal into a projected 7.11x win.
    at_average = predict_reselection_delta_bits(300, 2.6573)
    realized = 223 * 8
    assert realized - at_average == pytest.approx(986.81, abs=0.1)
    assert predict_reselection_delta_bits(0, 5.9467) == 0.0


def test_greedy_margin_degrades_both_terms_is_a_real_predicate():
    set_yield, marginal_yield = 876 / 569, 116 / 300
    assert set_yield / marginal_yield == pytest.approx(3.98, abs=5e-3)
    assert greedy_margin_degrades_both_terms(2.6573, 5.9467, set_yield, marginal_yield)
    # It is falsifiable in both terms, so it cannot be a constant-returning stub.
    assert not greedy_margin_degrades_both_terms(5.9467, 2.6573, set_yield, marginal_yield)
    assert not greedy_margin_degrades_both_terms(
        2.6573, 5.9467, marginal_yield, set_yield
    )


def test_marginal_over_average_refuses_zero_average():
    with pytest.raises(ValueError):
        marginal_over_average_ratio(5.3280, 0.0)


# ---------------------------------------------------------------------------
# The equations themselves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", ALL_CF2_TOKEN_PRICE_BUILDERS)
def test_equation_builds_with_anchors_producers_and_consumers(builder):
    eq = builder()
    assert eq.empirical_anchors
    assert eq.canonical_producers and eq.canonical_consumers
    assert eq.domain_of_validity["verdict_scope"] == "FORMULATION"
    assert eq.domain_of_validity["excluded"]
    for anchor in eq.empirical_anchors:
        assert anchor.residual >= 0.0
        assert anchor.provenance.score_claim_valid is False
        assert anchor.provenance.promotion_eligible is False


@pytest.mark.parametrize("builder", ALL_CF2_TOKEN_PRICE_BUILDERS)
def test_every_anchor_json_round_trips_exactly(builder):
    """The #1149 red: an anchor that does not reconstruct byte-for-byte is a broken row."""
    eq = builder()
    payload = json.loads(json.dumps(eq.to_dict()))
    reconstructed = json.loads(json.dumps(_equation_from_dict(payload).to_dict()))
    assert reconstructed["empirical_anchors"] == payload["empirical_anchors"]
    assert reconstructed == payload


def test_callable_paths_resolve_to_the_real_evaluators():
    import importlib

    for builder in ALL_CF2_TOKEN_PRICE_BUILDERS:
        eq = builder()
        module_path, _, attr = eq.python_callable_module_path.partition(":")
        fn = getattr(importlib.import_module(module_path), attr)
        assert callable(fn)


def test_registration_and_canonical_roundtrip_audit(tmp_path):
    ledger = tmp_path / "registry.jsonl"
    built = populate_cf2_token_price_laws(
        path=ledger, lock_path=tmp_path / "registry.lock", agent="test", subagent_id="test"
    )
    assert len(built) == 2
    assert audit_empirical_anchor_roundtrip_fidelity(ledger) == ()
    ids = {eq.equation_id for eq in query_equations(path=ledger)}
    assert ids == {
        "token_rate_model_direction_dependence_v1",
        "greedy_set_average_vs_marginal_price_v1",
    }


# ---------------------------------------------------------------------------
# The constants against the retained receipts' OWN published fields.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JG5_RECEIPT.exists(), reason="SSD tier not mounted")
def test_jg5_constants_match_the_retained_receipt():
    receipt = json.loads(JG5_RECEIPT.read_text())
    assert pytest.approx(
        receipt["measured_bits_per_changed_token"]
    ) == JG5_SHIPPED_455_BITS_PER_TOKEN
    assert ranker_relative_ratio(JG5_SHIPPED_455_BITS_PER_TOKEN) == pytest.approx(
        receipt["realized_over_modelled"]
    )
    assert receipt["modelled_bits_per_changed_token_jg1"] == pytest.approx(
        JG1_LOGIT_RANKER_BITS_PER_TOKEN
    )
    assert bits_per_token(
        receipt["token_stream_delta_bytes"], receipt["tokens_changed"]
    ) == pytest.approx(receipt["measured_bits_per_changed_token"])
    assert receipt["score_claim"] is False


@pytest.mark.skipif(not MIRROR_RECEIPT.exists(), reason="SSD tier not mounted")
def test_mirror_constants_match_the_retained_receipt():
    receipt = json.loads(MIRROR_RECEIPT.read_text())
    trimmed_average = bits_per_token(
        receipt["token_stream_delta_bytes"], receipt["tokens_changed"]
    )
    assert trimmed_average == pytest.approx(receipt["measured_bits_per_changed_token"])
    assert ranker_relative_ratio(trimmed_average) == pytest.approx(
        receipt["realized_over_modelled"]
    )
    # The marginal removal is measured against the SHIPPED 455-edit baseline (113,847 B),
    # not against the receipt's zero-edit token_stream_bytes_base (109,696 B).
    assert receipt["token_stream_bytes_candidate"] == 113183
    assert bits_per_token(113847 - 113183, 8654 - receipt["tokens_changed"]) == (
        pytest.approx(5.3280, abs=1e-4)
    )
    assert receipt["delta_trustworthy"] is True
    assert receipt["score_claim"] is False
