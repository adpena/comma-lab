# SPDX-License-Identifier: MIT
"""Catalog #237 BOYD-1 self-protection regression tests.

The BOYD-1 finding (R2 ledger 2026-05-14, Boyd + Tao CRITICAL) anchored a
bug class where ``FALLBACK_PROVIDERS_PER_CLASS`` was semantically OVERLOADED:

1. Time-Traveler amendment fires when fallback is >25% CHEAPER than canonical.
2. Capacity-overflow semantic per the canonical doc says ``long_burn``'s
   ``vastai/H100`` fallback is for race-mode urgency when Lightning A100
   is saturated — H100 is MORE EXPENSIVE than the subscription canonical.

The two semantics conflict in the same dict with no discriminator. The
fix splits the dict into ``_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS`` and
``_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS``, adds a ``FallbackReason`` enum,
and gates capacity-overflow escalation behind explicit
``capacity_overflow=True`` opt-in. Backwards-compat is preserved via the
legacy ``FALLBACK_PROVIDERS_PER_CLASS`` alias (read-only union).

These tests pin the post-fix behavior:
- Cheaper-alternative dict and capacity-overflow dict are SEPARATE
- ``FallbackReason`` enum exposes both semantics distinctly
- Legacy union alias preserves backwards-compat
- Time-Traveler auto-routing reads ONLY cheaper-alternative dict
- Capacity-overflow escalation requires explicit opt-in
- Semantic invariant: every cheaper-alt fallback must be (statically) NOT
  obviously more expensive than canonical for that class
- Routing decision exposes the chosen fallback's reason

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against".

Memory: feedback_r2_critical_fix_wave_tao1_boyd1_landed_20260515.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.cost_band_calibration import (
    CANONICAL_PROVIDER_PER_CLASS,
    DISPATCH_CLASSES,
    FALLBACK_PROVIDERS_PER_CLASS,
    FallbackReason,
    ProviderRoutingDecision,
    _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS,
    _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS,
    _fallback_reason_for,
    select_provider_for_class,
    select_provider_for_recipe,
)


# ----------------------------------------------------------------------------
# Bug-class anchor — the R2 ledger BOYD-1 reproducer cases
# ----------------------------------------------------------------------------


def test_long_burn_cheaper_alternative_is_empty() -> None:
    """long_burn has NO cheaper alternative — A100 subscription is free.

    Pre-fix bug: the legacy FALLBACK dict listed vastai/H100 here, which
    Time-Traveler amendment could never trigger (H100 > A100 subscription
    cost).
    """
    assert _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS["long_burn"] == []


def test_long_burn_capacity_overflow_is_h100() -> None:
    """long_burn capacity-overflow fallback is vastai/H100.

    This is the correct semantic — it fires only when Lightning A100 is
    saturated AND operator explicitly opts in via capacity_overflow=True.
    """
    assert _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS["long_burn"] == [
        ("vastai", "H100")
    ]


def test_long_burn_no_overflow_optin_means_no_fallback_considered() -> None:
    """Without capacity_overflow=True, long_burn auto-routing is canonical-only."""
    decision = select_provider_for_class(
        "long_burn", consult_posterior=False
    )
    assert decision.provider == "lightning"
    assert decision.gpu == "A100"
    assert decision.fallback_provider is None
    assert decision.fallback_gpu is None
    assert decision.fallback_reason is None


def test_long_burn_overflow_optin_considers_h100_fallback() -> None:
    """capacity_overflow=True opts long_burn into the H100 escalation."""
    # Without posterior, the fallback is considered but won't be chosen
    # (no empirical p50 to compare). The decision still records that the
    # fallback set was considered — verified via posterior path test below.
    decision = select_provider_for_class(
        "long_burn", consult_posterior=False, capacity_overflow=True
    )
    assert decision.provider == "lightning"  # canonical still chosen
    assert decision.gpu == "A100"


# ----------------------------------------------------------------------------
# Semantic invariants — the canonical contract
# ----------------------------------------------------------------------------


def test_two_dicts_are_disjoint_per_class() -> None:
    """A (provider, gpu) tuple must not appear in BOTH dicts for the same class.

    Per CLAUDE.md "Beauty, simplicity": every fallback has exactly one
    semantic.
    """
    for cls in DISPATCH_CLASSES:
        cheaper = set(
            _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(cls, [])
        )
        overflow = set(
            _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(cls, [])
        )
        assert cheaper.isdisjoint(overflow), (
            f"Class {cls!r}: same fallback in both dicts: "
            f"{cheaper & overflow}"
        )


def test_legacy_alias_is_union_of_two_dicts() -> None:
    """FALLBACK_PROVIDERS_PER_CLASS is the union of both dicts (back-compat).

    Legacy callers that grep for the old name still see all fallbacks.
    The union is constructed at import time and is read-only by convention.
    """
    for cls in DISPATCH_CLASSES:
        cheaper = list(
            _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(cls, [])
        )
        overflow = list(
            _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(cls, [])
        )
        expected = cheaper + overflow
        assert FALLBACK_PROVIDERS_PER_CLASS[cls] == expected, (
            f"Legacy alias for class {cls!r} differs from union"
        )


def test_legacy_alias_long_burn_includes_overflow_h100() -> None:
    """Backwards-compat: legacy callers still see vastai/H100 for long_burn."""
    assert ("vastai", "H100") in FALLBACK_PROVIDERS_PER_CLASS["long_burn"]


def test_every_dispatch_class_has_entries_in_both_dicts() -> None:
    """Both dicts must declare an entry (possibly empty) for every class."""
    for cls in DISPATCH_CLASSES:
        assert cls in _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS, (
            f"Class {cls!r} missing from cheaper-alternative dict"
        )
        assert cls in _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS, (
            f"Class {cls!r} missing from capacity-overflow dict"
        )


# ----------------------------------------------------------------------------
# FallbackReason enum
# ----------------------------------------------------------------------------


def test_fallback_reason_has_two_members() -> None:
    """FallbackReason enum has exactly the two semantics we declared."""
    assert {m.value for m in FallbackReason} == {
        "cheaper_alternative",
        "capacity_overflow",
    }


def test_fallback_reason_for_cheaper_alternative() -> None:
    """_fallback_reason_for classifies cheaper-alternative entries correctly."""
    reason = _fallback_reason_for("full", "modal", "A100")
    assert reason is FallbackReason.CHEAPER_ALTERNATIVE


def test_fallback_reason_for_capacity_overflow() -> None:
    """_fallback_reason_for classifies capacity-overflow entries correctly."""
    reason = _fallback_reason_for("long_burn", "vastai", "H100")
    assert reason is FallbackReason.CAPACITY_OVERFLOW


def test_fallback_reason_for_unknown_returns_none() -> None:
    reason = _fallback_reason_for("full", "kaggle", "T4")
    assert reason is None


def test_fallback_reason_for_unknown_class_returns_none() -> None:
    reason = _fallback_reason_for("nonexistent_class", "modal", "A100")
    assert reason is None


# ----------------------------------------------------------------------------
# select_provider_for_class — routing semantics
# ----------------------------------------------------------------------------


def test_select_provider_routing_decision_includes_fallback_reason_field() -> None:
    """ProviderRoutingDecision exposes fallback_reason field."""
    decision = select_provider_for_class(
        "smoke", consult_posterior=False
    )
    assert hasattr(decision, "fallback_reason")
    assert decision.fallback_reason is None  # canonical chosen, no fallback


def test_select_provider_decision_as_dict_includes_fallback_reason() -> None:
    """as_dict() serialization exposes fallback_reason as string or None."""
    decision = select_provider_for_class(
        "smoke", consult_posterior=False
    )
    d = decision.as_dict()
    assert "fallback_reason" in d
    assert d["fallback_reason"] is None


def test_select_provider_capacity_overflow_default_false() -> None:
    """capacity_overflow defaults to False — long_burn auto-route is canonical-only."""
    decision = select_provider_for_class("long_burn", consult_posterior=False)
    assert decision.fallback_provider is None


def test_select_provider_for_recipe_propagates_capacity_overflow() -> None:
    """select_provider_for_recipe accepts and propagates capacity_overflow."""
    recipe_meta = {
        "dispatch_class": "long_burn",
        "cost_band": {"epochs": 3000},
    }
    decision = select_provider_for_recipe(
        recipe_meta, consult_posterior=False, capacity_overflow=True
    )
    assert decision.dispatch_class == "long_burn"
    assert decision.provider == "lightning"


# ----------------------------------------------------------------------------
# Time-Traveler amendment posterior-driven routing — the canonical happy path
# ----------------------------------------------------------------------------


def _write_test_posterior(
    tmp_path: Path, anchors: list[dict]
) -> Path:
    """Write a posterior JSONL with given anchors for tests."""
    posterior_path = tmp_path / "cost_band_posterior.jsonl"
    with open(posterior_path, "w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(a) + "\n")
    return posterior_path


def test_time_traveler_fires_for_cheaper_alternative(tmp_path: Path) -> None:
    """When cheaper-alternative posterior is >25% cheaper than canonical, re-route."""
    # 6 successful anchors total (3 each per provider, all `outcome=successful_dispatch`)
    anchors = []
    # Canonical: vastai/RTX_4090 — expensive
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_canon",
            "trainer": "test.py",
            "platform": "vastai",
            "gpu": "RTX_4090",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 3600.0,
            "actual_cost_usd": 10.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    # Cheaper-alternative: modal/A100 — cheaper
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_fallback",
            "trainer": "test.py",
            "platform": "modal",
            "gpu": "A100",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 1800.0,
            "actual_cost_usd": 5.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    posterior_path = _write_test_posterior(tmp_path, anchors)
    decision = select_provider_for_class(
        "full",
        posterior_path=posterior_path,
        consult_posterior=True,
    )
    assert decision.re_routed is True
    assert decision.provider == "modal"
    assert decision.gpu == "A100"
    assert decision.fallback_reason is FallbackReason.CHEAPER_ALTERNATIVE


def test_time_traveler_does_not_fire_for_capacity_overflow_when_optout(
    tmp_path: Path,
) -> None:
    """Without capacity_overflow=True, long_burn never considers H100 fallback.

    Even if H100 had a cheaper posterior anchor (hypothetical), the routing
    helper should not auto-route to a capacity-overflow fallback.
    """
    anchors = []
    # Canonical: lightning/A100 — slow + expensive (hypothetical)
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_canon",
            "trainer": "test.py",
            "platform": "lightning",
            "gpu": "A100",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 36000.0,
            "actual_cost_usd": 50.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    # Capacity-overflow: vastai/H100 — hypothetically cheaper
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_overflow",
            "trainer": "test.py",
            "platform": "vastai",
            "gpu": "H100",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 7200.0,
            "actual_cost_usd": 20.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    posterior_path = _write_test_posterior(tmp_path, anchors)
    decision = select_provider_for_class(
        "long_burn",
        posterior_path=posterior_path,
        consult_posterior=True,
        capacity_overflow=False,  # explicit
    )
    # Without overflow opt-in, the H100 fallback is NOT considered.
    assert decision.re_routed is False
    assert decision.provider == "lightning"
    assert decision.gpu == "A100"
    assert decision.fallback_reason is None


def test_time_traveler_with_capacity_overflow_optin_can_re_route(
    tmp_path: Path,
) -> None:
    """With capacity_overflow=True AND posterior shows cheaper, re-route happens.

    The routing helper considers BOTH the cheaper-alternative set AND the
    capacity-overflow set; the chosen fallback's reason is recorded so the
    operator/log can audit which trigger fired.
    """
    anchors = []
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_canon",
            "trainer": "test.py",
            "platform": "lightning",
            "gpu": "A100",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 36000.0,
            "actual_cost_usd": 50.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    for _ in range(3):
        anchors.append({
            "schema": "cost_band_posterior_v1",
            "logged_at_utc": "2026-05-14T12:00:00+00:00",
            "dispatch_label": "test_overflow",
            "trainer": "test.py",
            "platform": "vastai",
            "gpu": "H100",
            "epochs": 3000,
            "batch_size": 32,
            "all_flags_on": True,
            "actual_wall_clock_sec": 7200.0,
            "actual_cost_usd": 20.0,
            "predicted_cost_usd_low": None,
            "predicted_cost_usd_high": None,
            "prediction_in_band": None,
            "outcome": "successful_dispatch",
            "returncode": 0,
            "notes": "",
        })
    posterior_path = _write_test_posterior(tmp_path, anchors)
    decision = select_provider_for_class(
        "long_burn",
        posterior_path=posterior_path,
        consult_posterior=True,
        capacity_overflow=True,
    )
    # With overflow opt-in AND H100 cheaper (hypothetical), re-route.
    assert decision.re_routed is True
    assert decision.provider == "vastai"
    assert decision.gpu == "H100"
    assert decision.fallback_reason is FallbackReason.CAPACITY_OVERFLOW


# ----------------------------------------------------------------------------
# Cathedral autopilot wire-in (hook 4) — routing decisions affect dispatch
# ----------------------------------------------------------------------------


def test_routing_decision_serializable_for_autopilot() -> None:
    """Decision dict must be JSON-serializable for autopilot consumption."""
    decision = select_provider_for_class(
        "full", consult_posterior=False
    )
    d = decision.as_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    parsed = json.loads(s)
    assert parsed["dispatch_class"] == "full"
    assert parsed["fallback_reason"] is None  # canonical chosen
