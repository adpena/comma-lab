# SPDX-License-Identifier: MIT
"""Tests for the MPS-VIABLE pre-screen cathedral consumer.

Per `lane_mps_prescreen_cathedral_consumer_wire_in_20260519` Phase 2 +
Catalog #335 + Catalog #287 + Catalog #313 (probe-outcomes ledger
consumption) + Catalog #192/#317 (MPS non-promotion discipline).

Covers:
- Protocol satisfaction (CathedralConsumerContract)
- validate_consumer_module returns contract_compliant=True
- consume_candidate routing cascade (advisory/promotable/insufficient)
- MPS-VIABLE probe SUPERSEDED auto-fallback to paid_cuda_authoritative
- Promotable contest-axis MUST go to paid CUDA
- Mixed-signal candidates fail-closed to paid CUDA
- Insufficient signal returns route=none (cathedral autopilot fallback)
- Catalog #287 axis-tag is [predicted] / promotable=False (always)
- Catalog #335 LIVE_COUNT remains 0 after this consumer lands
- predicted_delta_adjustment is ALWAYS 0.0 (routing is not a score signal)
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    HookNumber,
    validate_consumer_module,
)


PKG_NAME = "mps_viable_prescreen_consumer"
PKG_PATH = f"tac.cathedral_consumers.{PKG_NAME}"


@pytest.fixture
def consumer_module():
    return importlib.import_module(PKG_PATH)


# ---------------------------------------------------------------------------
# Protocol + contract validation
# ---------------------------------------------------------------------------


def test_consumer_satisfies_protocol(consumer_module) -> None:
    """The MPS pre-screen consumer satisfies CathedralConsumerContract."""
    assert isinstance(consumer_module, CathedralConsumerContract)


def test_consumer_validates_clean(consumer_module) -> None:
    """validate_consumer_module returns contract_compliant=True."""
    reg = validate_consumer_module(consumer_module, module_path=PKG_PATH)
    assert reg.contract_compliant, (
        f"{PKG_NAME} validation errors: {list(reg.validation_errors)}"
    )
    assert reg.consumer_name == PKG_NAME
    assert reg.consumer_version == "1.0.0"


def test_consumer_declares_cathedral_dispatch_hook(consumer_module) -> None:
    """Catalog #125 hook #4 (CATHEDRAL_AUTOPILOT_DISPATCH) is PRIMARY."""
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer_module.CONSUMER_HOOK_NUMBERS


def test_consumer_declares_probe_disambiguator_hook(consumer_module) -> None:
    """Catalog #125 hook #6 (PROBE_DISAMBIGUATOR) is SECONDARY per design."""
    assert HookNumber.PROBE_DISAMBIGUATOR in consumer_module.CONSUMER_HOOK_NUMBERS


def test_consumer_metadata_present(consumer_module) -> None:
    """Module-level metadata exists per CathedralConsumerContract."""
    assert isinstance(consumer_module.CONSUMER_NAME, str)
    assert isinstance(consumer_module.CONSUMER_VERSION, str)
    assert isinstance(consumer_module.CONSUMER_HOOK_NUMBERS, tuple)
    # The probe ID constant is exposed for sister-subagent consumers.
    assert (
        consumer_module.MPS_VIABLE_PROBE_ID
        == "mps_phase_b_options_b_plus_c_completion_20260519T062500Z"
    )


# ---------------------------------------------------------------------------
# Routing decision cascade
# ---------------------------------------------------------------------------


def test_advisory_axis_candidate_recommends_local_mps(consumer_module) -> None:
    """Advisory-grade candidate routes to local_mps_prescreen."""
    candidate = {
        "candidate_id": "advisory_001",
        "axis_tag": "[advisory only]",
        "evidence_grade": "predicted",
    }
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_LOCAL_MPS_PRESCREEN
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    # Rationale must cite META-ASSUMPTION review per discipline
    assert "META-ASSUMPTION" in result["rationale"] or "MPS-VIABLE" in result["rationale"]


def test_mps_proxy_axis_routes_to_mps_prescreen(consumer_module) -> None:
    """[MPS-PROXY] candidate routes to local_mps_prescreen."""
    candidate = {"axis_tag": "[MPS-PROXY]", "evidence_grade": "mps_research_signal"}
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_LOCAL_MPS_PRESCREEN


def test_macos_cpu_advisory_routes_to_mps_prescreen(consumer_module) -> None:
    """[macOS-CPU advisory] candidate routes to local_mps_prescreen."""
    candidate = {"axis_tag": "[macOS-CPU advisory]", "evidence_grade": "macos_cpu_advisory"}
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_LOCAL_MPS_PRESCREEN


def test_promotable_contest_cuda_routes_to_paid_cuda(consumer_module) -> None:
    """Candidate with promotion_eligible=True routes to paid_cuda_authoritative."""
    candidate = {
        "candidate_id": "promotable_001",
        "axis_tag": "[contest-CUDA]",
        "evidence_grade": "contest_cuda",
        "promotion_eligible": True,
        "score_claim_valid": True,
    }
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_PAID_CUDA_AUTHORITATIVE
    assert "MPS auth eval is NOISE" in result["rationale"] or "forbidden" in result["rationale"].lower()


def test_score_claim_valid_with_contest_axis_routes_to_paid_cuda(consumer_module) -> None:
    """score_claim_valid=True + [contest-CUDA] axis → paid_cuda_authoritative."""
    candidate = {
        "axis_tag": "[contest-CUDA]",
        "score_claim_valid": True,
    }
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_PAID_CUDA_AUTHORITATIVE


def test_mixed_signal_contest_axis_blocks_mps_routing(consumer_module) -> None:
    """Mixed advisory + contest-axis → contest-axis wins (fail-closed)."""
    candidate = {
        "axis_tag": "[contest-CUDA]",
        "evidence_grade": "predicted",  # contradicts axis_tag
    }
    result = consumer_module.consume_candidate(candidate)
    # Contest-axis token wins; not advisory by construction; not promotable
    # (no explicit promotion signal) → falls through to route=none
    assert result["recommended_route"] in (
        consumer_module.ROUTE_NONE,
        consumer_module.ROUTE_PAID_CUDA_AUTHORITATIVE,
    )


def test_insufficient_signal_returns_route_none(consumer_module) -> None:
    """Candidate with no axis info → route=none (cathedral autopilot fallback)."""
    candidate = {"candidate_id": "unknown_001"}
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_NONE
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["confidence"] == 0.0


def test_empty_candidate_returns_route_none(consumer_module) -> None:
    """Empty candidate dict → route=none."""
    result = consumer_module.consume_candidate({})
    assert result["recommended_route"] == consumer_module.ROUTE_NONE


# ---------------------------------------------------------------------------
# Catalog #287/#323 axis-tag + promotable discipline
# ---------------------------------------------------------------------------


def test_predicted_delta_adjustment_always_zero(consumer_module) -> None:
    """Routing is not a score signal — predicted_delta_adjustment ALWAYS 0.0."""
    test_candidates = [
        {"axis_tag": "[predicted]"},
        {"axis_tag": "[contest-CUDA]", "score_claim_valid": True},
        {"axis_tag": "[MPS-PROXY]"},
        {},
        {"axis_tag": "[advisory only]", "promotion_eligible": False},
    ]
    for candidate in test_candidates:
        result = consumer_module.consume_candidate(candidate)
        assert result["predicted_delta_adjustment"] == 0.0


def test_promotable_field_always_false(consumer_module) -> None:
    """Per Catalog #287/#323: this consumer's output is NEVER promotable
    (the routing signal itself is observability-only)."""
    test_candidates = [
        {"axis_tag": "[predicted]"},
        {"axis_tag": "[contest-CUDA]", "score_claim_valid": True},
        {"axis_tag": "[advisory only]"},
        {},
    ]
    for candidate in test_candidates:
        result = consumer_module.consume_candidate(candidate)
        assert result["promotable"] is False, (
            f"candidate {candidate} got promotable=True; must always be False"
        )


def test_axis_tag_always_predicted(consumer_module) -> None:
    """Per CLAUDE.md 'Apples-to-apples': the routing signal is [predicted]."""
    test_candidates = [
        {"axis_tag": "[contest-CUDA]", "score_claim_valid": True},
        {"axis_tag": "[advisory only]"},
        {},
    ]
    for candidate in test_candidates:
        result = consumer_module.consume_candidate(candidate)
        assert result["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# Probe-outcomes ledger SUPERSEDE auto-fallback (Catalog #313)
# ---------------------------------------------------------------------------


def test_probe_supersede_to_defer_auto_fallbacks_to_paid_cuda(
    consumer_module, tmp_path, monkeypatch
) -> None:
    """When MPS-VIABLE probe is SUPERSEDED to DEFER, all routing → paid_cuda."""
    from tac.probe_outcomes_ledger import (
        register_probe_outcome,
        update_probe_outcome,
        BLOCKER_STATUS_BLOCKING,
        VERDICT_DEFER,
        VERDICT_PROCEED,
    )

    fake_ledger = tmp_path / "probe_outcomes.jsonl"
    fake_lock = tmp_path / "probe_outcomes.lock"

    monkeypatch.setattr(
        "tac.probe_outcomes_ledger.PROBE_OUTCOMES_LEDGER_PATH", fake_ledger
    )
    monkeypatch.setattr(
        "tac.probe_outcomes_ledger.PROBE_OUTCOMES_LEDGER_LOCK", fake_lock
    )

    # Register initial PROCEED then SUPERSEDE to DEFER.
    register_probe_outcome(
        probe_id=consumer_module.MPS_VIABLE_PROBE_ID,
        substrate="test_substrate",
        probe_kind="test_kind",
        recipe_path=".omx/operator_authorize_recipes/test.yaml",
        verdict=VERDICT_PROCEED,
        evidence_path="test.json",
        next_action="proceed",
        blocker_status="advisory",
        agent="test",
        metric_name="gap_relative_aggregate",
        metric_value=0.0007192662962768041,
        path=fake_ledger,
        lock_path=fake_lock,
    )
    update_probe_outcome(
        probe_id=consumer_module.MPS_VIABLE_PROBE_ID,
        event_type="superseded",
        verdict=VERDICT_DEFER,
        blocker_status=BLOCKER_STATUS_BLOCKING,
        notes="future re-measurement contradicts PROCEED",
        path=fake_ledger,
        lock_path=fake_lock,
    )

    # Now even an advisory-grade candidate should route to paid CUDA.
    candidate = {"axis_tag": "[advisory only]"}
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_PAID_CUDA_AUTHORITATIVE
    assert "SUPERSEDED" in result["rationale"] or "Catalog #313" in result["rationale"]


def test_probe_missing_falls_back_to_paid_cuda(
    consumer_module, tmp_path, monkeypatch
) -> None:
    """When MPS-VIABLE probe outcome is missing entirely, auto-fallback to paid CUDA."""
    fake_ledger = tmp_path / "probe_outcomes.jsonl"
    fake_lock = tmp_path / "probe_outcomes.lock"
    # Empty file, no outcomes registered
    fake_ledger.write_text("")

    monkeypatch.setattr(
        "tac.probe_outcomes_ledger.PROBE_OUTCOMES_LEDGER_PATH", fake_ledger
    )
    monkeypatch.setattr(
        "tac.probe_outcomes_ledger.PROBE_OUTCOMES_LEDGER_LOCK", fake_lock
    )

    candidate = {"axis_tag": "[advisory only]"}
    result = consumer_module.consume_candidate(candidate)
    assert result["recommended_route"] == consumer_module.ROUTE_PAID_CUDA_AUTHORITATIVE


# ---------------------------------------------------------------------------
# Catalog #335 LIVE_COUNT regression guard
# ---------------------------------------------------------------------------


def test_catalog_335_live_count_remains_zero() -> None:
    """Catalog #335 STRICT preflight gate has LIVE_COUNT 0 after this consumer."""
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )

    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"Catalog #335 LIVE_COUNT drift: {len(violations)} violations: {violations}"
    )


def test_directory_exists_with_canonical_layout() -> None:
    """Sanity: package directory present + __init__.py readable."""
    init_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "tac"
        / "cathedral_consumers"
        / PKG_NAME
        / "__init__.py"
    )
    assert init_path.is_file()
    text = init_path.read_text(encoding="utf-8")
    assert "CONSUMER_NAME" in text
    assert "CathedralConsumerContract" in text or "consumer_contract" in text


# ---------------------------------------------------------------------------
# Update-from-anchor is NO-OP per design (probe ledger is source of truth)
# ---------------------------------------------------------------------------


def test_update_from_anchor_is_noop(consumer_module) -> None:
    """update_from_anchor is intentional NO-OP; probe ledger is source of truth."""
    # Should not raise regardless of anchor shape
    consumer_module.update_from_anchor(None)
    consumer_module.update_from_anchor({"any": "shape"})
    consumer_module.update_from_anchor("string-anchor")
