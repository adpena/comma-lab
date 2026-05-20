# SPDX-License-Identifier: MIT
"""Dedicated tests for Slot JJ Path B widening of the xray tool's Catalog #341
strict check at ``tools/master_gradient_xray.py`` (2026-05-19).

Per Slot II classification memo
``.omx/research/catalog_341_noncompliance_classification_20260519.md``: the
xray tool's narrow ``axis_tag == "[predicted]"`` strict check excluded the
canonical ``[MPS-PROXY]`` axis tag mandated by CLAUDE.md "MPS auth eval is
NOISE" Rule 3 + the FORBIDDEN PATTERN "Forbidden MPS-derived strategic
decision". Path B widens the check to ``axis_tag in
CANONICAL_NON_PROMOTABLE_AXES`` (a frozenset that unions ``{"[predicted]"}``
with the canonical ``NON_PROMOTABLE_TAGS`` from
``tac.continual_learning``).

Sister test file to ``test_tools_master_gradient_xray.py`` (54 existing
tests) — kept SEPARATE per Catalog #229 premise-verification discipline so
a dedicated regression surface tracks the Slot JJ Path B + Path A
remediation per Catalog #341.

6-hook wire-in declaration per Catalog #125: hook #6 probe-disambiguator =
ACTIVE — the widened check IS the canonical disambiguator between
intentionally-diagnostic-with-canonical-non-promotable-axis-tag consumers
and Catalog #341 routing-marker regression.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "master_gradient_xray.py"


@pytest.fixture(scope="module")
def xray_module():
    """Load tools/master_gradient_xray.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "master_gradient_xray_slot_jj_widening_test", TOOL_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["master_gradient_xray_slot_jj_widening_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Path B canonical-set discovery + widening verification
# ============================================================================


def test_canonical_non_promotable_axes_exists_as_module_attribute(xray_module):
    """Path B landed a module-level CANONICAL_NON_PROMOTABLE_AXES frozenset."""
    assert hasattr(xray_module, "CANONICAL_NON_PROMOTABLE_AXES"), (
        "Slot JJ Path B requires CANONICAL_NON_PROMOTABLE_AXES module-level "
        "constant per the widening landing"
    )
    assert isinstance(xray_module.CANONICAL_NON_PROMOTABLE_AXES, frozenset), (
        "CANONICAL_NON_PROMOTABLE_AXES must be frozenset (immutability per "
        "Catalog #265 + #335 canonical contract pattern)"
    )


def test_canonical_non_promotable_axes_includes_predicted_default(xray_module):
    """Backwards-compat: ``[predicted]`` (the narrow pre-Path-B default) must
    remain in the canonical set so existing compliant consumers do not
    regress."""
    assert "[predicted]" in xray_module.CANONICAL_NON_PROMOTABLE_AXES


def test_canonical_non_promotable_axes_includes_mps_proxy(xray_module):
    """Path B headline: ``[MPS-PROXY]`` is canonically non-promotable per
    CLAUDE.md "MPS auth eval is NOISE" Rule 3 + the FORBIDDEN PATTERN
    "Forbidden MPS-derived strategic decision"; the canonical
    NON_PROMOTABLE_TAGS frozenset at ``tac.continual_learning`` includes it.

    Slot II classification empirically identified the 2 MPS consumers
    (mps_diagnostic_consumer + mps_gap_experiment_consumer) as failing the
    narrow ``== "[predicted]"`` check despite carrying canonical markers
    otherwise. Path B widens the check so they pass.
    """
    assert "[MPS-PROXY]" in xray_module.CANONICAL_NON_PROMOTABLE_AXES


def test_canonical_non_promotable_axes_unions_canonical_source(xray_module):
    """The xray tool's CANONICAL_NON_PROMOTABLE_AXES MUST union the canonical
    upstream ``tac.continual_learning.NON_PROMOTABLE_TAGS`` (the source of
    truth) — never re-derive locally."""
    from tac.continual_learning import NON_PROMOTABLE_TAGS

    # Every canonical NON_PROMOTABLE_TAGS member must be in the xray
    # tool's widened set.
    for tag in NON_PROMOTABLE_TAGS:
        assert tag in xray_module.CANONICAL_NON_PROMOTABLE_AXES, (
            f"canonical NON_PROMOTABLE_TAGS member {tag!r} missing from "
            f"xray tool's CANONICAL_NON_PROMOTABLE_AXES; the tool's set must "
            f"union the canonical source per Catalog #287 single-source-of-"
            f"truth discipline"
        )


# ============================================================================
# Path B widened-strict-check correctness
# ============================================================================


def test_widened_check_accepts_mps_proxy_consumer(xray_module):
    """A consumer with ``axis_tag="[MPS-PROXY]"`` + canonical other markers
    is now Catalog #341 compliant under the Path B widening."""

    class MPSProxyConsumer:
        __name__ = "mps_proxy_test_consumer"
        CONSUMER_NAME = "mps_proxy_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            return {
                "predicted_delta_adjustment": 0.0,
                "rationale": "MPS proxy test consumer",
                "axis_tag": "[MPS-PROXY]",
                "promotable": False,
                "confidence": 0.0,
            }

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[MPSProxyConsumer],
    )
    assert len(verdicts) == 1
    assert verdicts[0]["catalog_341_markers_compliant"] is True, (
        f"Path B widening should accept [MPS-PROXY] as canonical non-"
        f"promotable axis_tag per Slot II classification; got verdict: "
        f"{verdicts[0]}"
    )


def test_widened_check_still_rejects_promotable_contest_axis(xray_module):
    """Regression guard: a consumer with ``axis_tag="[contest-CPU]"`` (a
    PROMOTABLE axis) MUST still fail the Catalog #341 check after Path B
    widening — the canonical set excludes promotable contest axes by
    construction. This is the canonical safety invariant."""

    class PromotableConsumer:
        __name__ = "promotable_test_consumer"
        CONSUMER_NAME = "promotable_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            return {
                "predicted_delta_adjustment": 0.0,
                "rationale": "promotable test consumer should fail",
                "axis_tag": "[contest-CPU]",
                "promotable": False,
                "confidence": 0.0,
            }

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[PromotableConsumer],
    )
    assert len(verdicts) == 1
    assert verdicts[0]["catalog_341_markers_compliant"] is False, (
        "[contest-CPU] is a PROMOTABLE axis and MUST be rejected by the "
        "Catalog #341 marker check even after Path B widening; this is the "
        "canonical safety invariant per CLAUDE.md 'Submission auth eval' "
        "non-negotiable"
    )


def test_widened_check_still_accepts_predicted_default(xray_module):
    """Backwards-compat regression: a consumer with ``axis_tag="[predicted]"``
    (the narrow pre-Path-B default) MUST still be accepted. No existing
    compliant consumer should regress."""

    class PredictedConsumer:
        __name__ = "predicted_test_consumer"
        CONSUMER_NAME = "predicted_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            return {
                "predicted_delta_adjustment": 0.0,
                "rationale": "predicted test consumer",
                "axis_tag": "[predicted]",
                "promotable": False,
                "confidence": 0.0,
            }

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[PredictedConsumer],
    )
    assert len(verdicts) == 1
    assert verdicts[0]["catalog_341_markers_compliant"] is True, (
        "Pre-Path-B compliant [predicted] consumer regressed; backwards-"
        "compat regression"
    )


def test_widened_check_rejects_unknown_axis_tag(xray_module):
    """A consumer with an axis_tag NOT in the canonical set (e.g. a
    typo or invented tag) MUST be rejected. The canonical set bounds
    permissible non-promotable axes; arbitrary strings are not accepted."""

    class UnknownAxisConsumer:
        __name__ = "unknown_axis_test_consumer"
        CONSUMER_NAME = "unknown_axis_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            return {
                "predicted_delta_adjustment": 0.0,
                "rationale": "unknown axis tag test consumer",
                "axis_tag": "[invented-noise-tag]",
                "promotable": False,
                "confidence": 0.0,
            }

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[UnknownAxisConsumer],
    )
    assert len(verdicts) == 1
    assert verdicts[0]["catalog_341_markers_compliant"] is False, (
        "An invented axis_tag NOT in CANONICAL_NON_PROMOTABLE_AXES must "
        "be rejected; the canonical set is bounded"
    )


# ============================================================================
# Path A waiver verification (Slot JJ) — regression guard for the 2 MPS consumers
# ============================================================================


def test_path_a_mps_diagnostic_consumer_imports_cleanly():
    """Path A regression guard: ``mps_diagnostic_consumer`` MUST still import
    cleanly post-waiver-line-2 addition. The Catalog #335 canonical contract
    fields must remain importable."""
    from tac.cathedral_consumers import mps_diagnostic_consumer

    assert mps_diagnostic_consumer.CONSUMER_NAME == "mps_diagnostic_consumer"
    assert mps_diagnostic_consumer.CONSUMER_VERSION == "0.1.0"
    assert mps_diagnostic_consumer.CONSUMER_HOOK_NUMBERS, (
        "CONSUMER_HOOK_NUMBERS must be non-empty per Catalog #335"
    )
    # Verify consume_candidate still returns canonical dict shape
    result = mps_diagnostic_consumer.consume_candidate({})
    assert result["axis_tag"] == "[MPS-PROXY]", (
        "axis_tag must remain [MPS-PROXY] per CLAUDE.md 'MPS auth eval "
        "is NOISE' Rule 3 + the canonical waiver rationale documents WHY"
    )
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False


def test_path_a_mps_gap_experiment_consumer_imports_cleanly():
    """Path A regression guard: ``mps_gap_experiment_consumer`` MUST still
    import cleanly post-waiver-line-2 addition."""
    from tac.cathedral_consumers import mps_gap_experiment_consumer

    assert mps_gap_experiment_consumer.CONSUMER_NAME == "mps_gap_experiment_consumer"
    assert mps_gap_experiment_consumer.CONSUMER_VERSION == "0.1.0"
    assert mps_gap_experiment_consumer.CONSUMER_HOOK_NUMBERS, (
        "CONSUMER_HOOK_NUMBERS must be non-empty per Catalog #335"
    )
    # Verify consume_candidate still returns canonical dict shape
    result = mps_gap_experiment_consumer.consume_candidate({})
    assert result["axis_tag"] == "[MPS-PROXY]", (
        "axis_tag must remain [MPS-PROXY] per CLAUDE.md 'MPS auth eval "
        "is NOISE' Rule 3 + active sister mps_phase_b_fire_and_harvest_"
        "20260519 owns per-experiment verdict harvest"
    )
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False


def test_path_a_mps_consumers_have_canonical_waiver_in_init():
    """Path A canonical-waiver-discovery regression: both MPS consumers
    carry ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` waivers per
    Catalog #335 ``discover_waiver_in_init`` helper. The Slot II
    classification recommends this as the structural documentation of
    the canonical tension between Catalog #341 marker-uniformity and
    CLAUDE.md MPS-tag-mandation."""
    from tac.cathedral.consumer_contract import discover_waiver_in_init

    for name in ("mps_diagnostic_consumer", "mps_gap_experiment_consumer"):
        path = (
            REPO_ROOT / "src" / "tac" / "cathedral_consumers" / name / "__init__.py"
        )
        waiver = discover_waiver_in_init(path)
        assert waiver is not None, (
            f"{name} must carry # CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale> "
            f"waiver per Slot II classification + Catalog #335"
        )
        # discover_waiver_in_init returns (rationale, valid) tuple
        rationale, valid = waiver
        assert valid, (
            f"{name} waiver rationale must be non-placeholder; got: {rationale}"
        )
        assert "MPS" in rationale, (
            f"{name} waiver rationale must cite MPS canonical discipline; "
            f"got: {rationale}"
        )


# ============================================================================
# Live-repo regression guard
# ============================================================================


def test_live_repo_catalog_341_compliance_at_100_percent(xray_module):
    """Live-repo regression guard: post-Path-B widening, the live cathedral
    consumer surface (~34 consumers) must show 100% Catalog #341 compliance.

    Pre-remediation: 32/34 = 94.1% (the 2 MPS consumers were non-compliant
    due to narrow ``== "[predicted]"`` check).
    Post-Path-B: 34/34 = 100% (widened check accepts ``[MPS-PROXY]``).

    Per Slot II classification memo + sister EE plot 3 empirical anchor.
    """
    verdicts = xray_module._collect_consumer_verdicts(
        {
            "candidate_id": "slot_jj_path_b_live_regression_guard",
            "archive_sha256": "6bae0201fb082457" + "0" * 48,
        }
    )
    n_total = len(verdicts)
    n_compliant = sum(1 for v in verdicts if v.get("catalog_341_markers_compliant"))
    n_non_compliant = n_total - n_compliant

    # We expect AT LEAST 30 contract-compliant consumers (per existing
    # sister test_consumer_verdict_matrix_returns_canonical_metrics) and
    # 100% Catalog #341 compliance post-Path-B.
    assert n_total >= 30, (
        f"expected >=30 contract-compliant consumers, got {n_total}; "
        f"sister regression"
    )
    assert n_non_compliant == 0, (
        f"Path B widening should drive Catalog #341 non-compliance to 0; "
        f"got {n_non_compliant}/{n_total} non-compliant. Non-compliant: "
        f"{[v['consumer_name'] for v in verdicts if not v.get('catalog_341_markers_compliant')]}"
    )
