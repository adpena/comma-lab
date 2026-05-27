# SPDX-License-Identifier: MIT
"""Tests for the META-RESURRECTION-AUDIT-V2 META-bug amplification cathedral consumer.

Per Catalog #335 canonical contract + Catalog #341 Tier-A routing markers +
Catalog #125 6-hook wire-in + op-routables Item #3.
"""
from __future__ import annotations

import tac.cathedral.consumer_contract as cc
import tac.cathedral_consumers.meta_resurrection_audit_v2_consumer as m


def test_contract_compliant():
    reg = cc.validate_consumer_module(
        m, module_path="tac.cathedral_consumers.meta_resurrection_audit_v2_consumer"
    )
    assert reg.contract_compliant, reg.validation_errors
    assert reg.validation_errors == ()


def test_consumer_name_version_hooks():
    assert m.CONSUMER_NAME == "meta_resurrection_audit_v2"
    assert isinstance(m.CONSUMER_VERSION, str)
    assert m.CONSUMER_VERSION == "1.0.0"
    assert cc.HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in m.CONSUMER_HOOK_NUMBERS
    assert cc.HookNumber.CONTINUAL_LEARNING_POSTERIOR in m.CONSUMER_HOOK_NUMBERS
    assert cc.HookNumber.PROBE_DISAMBIGUATOR in m.CONSUMER_HOOK_NUMBERS


def test_default_tier_a():
    # No CONSUMER_TIER declared -> default Tier A (observability-only).
    reg = cc.validate_consumer_module(m)
    assert reg.consumer_tier == cc.ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_all_10_meta_bug_classes_present():
    for mk in (f"M{i}" for i in range(1, 11)):
        assert mk in m.META_BUG_CLASSES, mk
        entry = m.META_BUG_CLASSES[mk]
        assert entry.get("label")
        assert entry.get("extinction")
        assert entry.get("reactivation")


def test_meta_bug_classes_with_equations_point_to_registered_ids():
    import tac.canonical_equations as ce

    registered = {e.equation_id for e in ce.query_equations()}
    for mk, entry in m.META_BUG_CLASSES.items():
        eq_id = entry["equation_id"]
        if eq_id:  # M4/M5/M6/M7 have no dedicated equation
            assert eq_id in registered, f"{mk} -> {eq_id} not registered"


def _assert_tier_a_markers(contribution):
    assert contribution["predicted_delta_adjustment"] == 0.0
    assert contribution["promotable"] is False
    assert contribution["axis_tag"] == "[predicted]"


def test_explicit_meta_bug_class_field():
    r = m.consume_candidate({"name": "lane_17_imp", "prior_verdict": "KILLED", "meta_bug_class": "M10"})
    assert r["meta_resurrection_verdict"] == "META_BUG_AMPLIFICATION_SUSPECTED"
    assert r["matched_meta_bug_class"] == "M10"
    assert r["amplification_equation_id"] == "synthetic_fallback_implementation_negative_result_amplification_v1"
    _assert_tier_a_markers(r)


def test_meta_bug_class_extended_form():
    r = m.consume_candidate({"verdict": "FALSIFIED", "meta_bug_class": "M2_cargo_cult_technique_family"})
    assert r["matched_meta_bug_class"] == "M2"
    _assert_tier_a_markers(r)


def test_meta_bug_embedded_in_notes_text():
    r = m.consume_candidate({
        "name": "apogee_int4",
        "verdict": "FALSIFIED",
        "notes": "original NAIVE-PTQ at 1.4287; META-BUG M2 cargo-cult-technique-family",
    })
    assert r["meta_resurrection_verdict"] == "META_BUG_AMPLIFICATION_SUSPECTED"
    assert r["matched_meta_bug_class"] == "M2"
    _assert_tier_a_markers(r)


def test_meta_bug_embedded_paren_form():
    r = m.consume_candidate({
        "verdict": "deferred",
        "notes": "wrong-canonical-application-surface (M8) per audit",
    })
    assert r["matched_meta_bug_class"] == "M8"


def test_negative_verdict_no_meta_bug_is_genuine_or_unclassified():
    r = m.consume_candidate({"name": "tier3", "verdict": "FALSIFIED", "notes": "genuine paradigm refutation"})
    assert r["meta_resurrection_verdict"] == "GENUINE_PARADIGM_REFUTATION_OR_NO_META_BUG"
    assert r["matched_meta_bug_class"] is None
    _assert_tier_a_markers(r)


def test_no_negative_verdict_not_a_resurrection_candidate():
    r = m.consume_candidate({"name": "active_substrate", "status": "in_progress"})
    assert r["meta_resurrection_verdict"] == "NO_NEGATIVE_VERDICT"
    _assert_tier_a_markers(r)


def test_non_mapping_returns_unknown():
    r = m.consume_candidate("not_a_mapping")
    assert r["meta_resurrection_verdict"] == "UNKNOWN"
    _assert_tier_a_markers(r)


def test_m4_no_equation_clause():
    # M4 has empty equation_id; verdict still SUSPECTED but no equation id.
    r = m.consume_candidate({"verdict": "FALSIFIED", "meta_bug_class": "M4"})
    assert r["meta_resurrection_verdict"] == "META_BUG_AMPLIFICATION_SUSPECTED"
    assert r["matched_meta_bug_class"] == "M4"
    assert r.get("amplification_equation_id") is None
    _assert_tier_a_markers(r)


def test_update_from_anchor_is_noop():
    # Hook #5: observability-only at landing; must not raise.
    assert m.update_from_anchor(None) is None
    assert m.update_from_anchor({"any": "anchor"}) is None


def test_auto_discovery_includes_consumer():
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_compliant_consumer_modules,
    )

    mods = discover_compliant_consumer_modules()
    names = {getattr(mod, "CONSUMER_NAME", None) for mod in mods}
    assert "meta_resurrection_audit_v2" in names


def test_unknown_meta_bug_key_in_text_not_matched():
    # "M11" does not exist; must not match.
    r = m.consume_candidate({"verdict": "killed", "notes": "META-BUG M11 nonexistent"})
    assert r["matched_meta_bug_class"] is None
