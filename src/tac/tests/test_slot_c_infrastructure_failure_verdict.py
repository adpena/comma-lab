# SPDX-License-Identifier: MIT
"""Tests for Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O3 (INFRASTRUCTURE_FAILURE verdict).

Per Slot C canonical-2-landing-pattern corrective-action cascade 2026-05-28
op-routable #3 (Catalog #313 verdict taxonomy extension):

Adds NEW canonical verdict ``INFRASTRUCTURE_FAILURE`` distinct from
``INDEPENDENT`` for the F19 segfault empirical anchor class. Per CLAUDE.md
"Apples-to-apples evidence discipline" non-negotiable:

* ``INDEPENDENT`` = paradigm-disambiguation empirical null ("probe ran cleanly,
  no signal in conditioning axis")
* ``INFRASTRUCTURE_FAILURE`` = probe could not be measured because code/runtime/
  dispatch infrastructure crashed (segfault, OOM, malformed input, NaN
  propagation, etc.)

The prior conflation poisoned downstream Catalog #307 paradigm-vs-implementation
classification. The F19 anchor
(``v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pair_segfault_20260518``)
SEGFAULTS at both 100-pair smoke AND 600-pair full on REAL A1 softmax data;
the V1/V2 sister Faiss codecs disambiguated Shannon zones cleanly, so the
correct verdict is INFRASTRUCTURE_FAILURE (hand-rolled V4 specifically) rather
than INDEPENDENT (Faiss-paradigm null).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    BLOCKER_STATUS_BLOCKING,
    BLOCKING_VERDICTS,
    EVENT_ADJUDICATED,
    EVENT_RATIFIED,
    EVENT_SUPERSEDED,
    SCHEMA_VERSION,
    VALID_VERDICTS,
    VERDICT_DEFER,
    VERDICT_INDEPENDENT,
    VERDICT_INFRASTRUCTURE_FAILURE,
    VERDICT_KILL,
    latest_blocking_outcome_by_recipe,
    latest_blocking_outcome_by_substrate,
    query_blocking_outcomes,
    query_by_probe_id,
    register_probe_outcome,
    update_probe_outcome,
)


@pytest.fixture
def tmp_ledger(tmp_path: Path):
    path = tmp_path / "probe_outcomes.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    return path, lock


# ---------------------------------------------------------------------------
# Verdict taxonomy invariants
# ---------------------------------------------------------------------------


def test_verdict_infrastructure_failure_constant_pinned() -> None:
    """The canonical token is pinned to the exact string."""
    assert VERDICT_INFRASTRUCTURE_FAILURE == "INFRASTRUCTURE_FAILURE"


def test_infrastructure_failure_in_valid_verdicts() -> None:
    """The new verdict is part of the canonical VALID_VERDICTS frozenset."""
    assert VERDICT_INFRASTRUCTURE_FAILURE in VALID_VERDICTS


def test_infrastructure_failure_in_blocking_verdicts() -> None:
    """Per Slot A FIX O3 design: INFRASTRUCTURE_FAILURE is BLOCKING because
    re-running the exact same infrastructure-broken probe would just re-crash
    and waste paid GPU spend. Resolution requires sister probe with corrected
    infrastructure OR operator override per Catalog #313 paired-env bypass."""
    assert VERDICT_INFRASTRUCTURE_FAILURE in BLOCKING_VERDICTS


def test_blocking_verdicts_canonical_4_member_set() -> None:
    """BLOCKING_VERDICTS contains exactly the 4 canonical blocking tokens."""
    assert BLOCKING_VERDICTS == frozenset({
        VERDICT_INDEPENDENT,
        VERDICT_KILL,
        VERDICT_DEFER,
        VERDICT_INFRASTRUCTURE_FAILURE,
    })


def test_infrastructure_failure_semantically_distinct_from_independent() -> None:
    """INDEPENDENT and INFRASTRUCTURE_FAILURE are DIFFERENT tokens (sanity)."""
    assert VERDICT_INDEPENDENT != VERDICT_INFRASTRUCTURE_FAILURE
    assert VERDICT_INDEPENDENT == "INDEPENDENT"
    assert VERDICT_INFRASTRUCTURE_FAILURE == "INFRASTRUCTURE_FAILURE"


# ---------------------------------------------------------------------------
# register_probe_outcome behavior with new verdict
# ---------------------------------------------------------------------------


def test_register_infrastructure_failure_outcome(tmp_ledger) -> None:
    """A probe outcome with verdict INFRASTRUCTURE_FAILURE registers cleanly
    and defaults to blocker_status=blocking."""
    path, lock = tmp_ledger
    row = register_probe_outcome(
        probe_id="test_infra_failure_v1",
        substrate="test_substrate",
        recipe_path=".omx/operator_authorize_recipes/substrate_test.yaml",
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        threshold=0.0,
        threshold_token="INFRASTRUCTURE_INTACT",
        evidence_path=".omx/research/test_segfault_anchor.md",
        next_action="route_to_v5_sister_with_corrected_implementation",
        notes="hand-rolled implementation SEGFAULTS at 100-pair and 600-pair on real data",
        agent="test",
        path=path,
        lock_path=lock,
    )
    assert row["verdict"] == VERDICT_INFRASTRUCTURE_FAILURE
    assert row["blocker_status"] == BLOCKER_STATUS_BLOCKING
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["event_type"] == EVENT_ADJUDICATED


def test_infrastructure_failure_blocks_dispatch_via_recipe_query(tmp_ledger) -> None:
    """``latest_blocking_outcome_by_recipe`` returns an INFRASTRUCTURE_FAILURE
    outcome as a blocker (sister gate Catalog #313 will refuse dispatch)."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_recipe_blocker_v1",
        substrate="test_substrate",
        recipe_path=".omx/operator_authorize_recipes/test_recipe.yaml",
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        notes="V4 hand-rolled segfaults; route to V5",
        agent="test",
        path=path,
        lock_path=lock,
    )
    blocker = latest_blocking_outcome_by_recipe(
        ".omx/operator_authorize_recipes/test_recipe.yaml",
        path=path,
    )
    assert blocker is not None
    assert blocker.verdict == VERDICT_INFRASTRUCTURE_FAILURE
    assert blocker.blocker_status == BLOCKER_STATUS_BLOCKING


def test_infrastructure_failure_blocks_dispatch_via_substrate_query(tmp_ledger) -> None:
    """``latest_blocking_outcome_by_substrate`` returns an INFRASTRUCTURE_FAILURE
    outcome as a blocker."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_substrate_blocker_v1",
        substrate="test_substrate_for_substrate_query",
        recipe_path=".omx/operator_authorize_recipes/some_recipe.yaml",
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        notes="V4 hand-rolled segfaults",
        agent="test",
        path=path,
        lock_path=lock,
    )
    blocker = latest_blocking_outcome_by_substrate(
        "test_substrate_for_substrate_query",
        path=path,
    )
    assert blocker is not None
    assert blocker.verdict == VERDICT_INFRASTRUCTURE_FAILURE


def test_query_blocking_outcomes_includes_infrastructure_failure(tmp_ledger) -> None:
    """``query_blocking_outcomes`` returns INFRASTRUCTURE_FAILURE rows in the
    set of active blockers (within staleness window)."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_query_v1",
        substrate="test_substrate",
        recipe_path=".omx/operator_authorize_recipes/test.yaml",
        probe_kind="codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        agent="test",
        path=path,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(path=path)
    assert any(
        r.get("verdict") == VERDICT_INFRASTRUCTURE_FAILURE for r in blockers
    )


# ---------------------------------------------------------------------------
# Lifecycle transitions (ratified, superseded, operator_override)
# ---------------------------------------------------------------------------


def test_infrastructure_failure_can_be_superseded_by_sister_probe(tmp_ledger) -> None:
    """A V5/V6/V7/V8 sister probe with corrected infrastructure can supersede
    the INFRASTRUCTURE_FAILURE row."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="v4_hand_rolled_segfault_test_v1",
        substrate="atw_codec_v2_1_faiss_ivf_pq_v4_hand_rolled",
        recipe_path=".omx/operator_authorize_recipes/v4.yaml",
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        next_action="advance_to_v5_wavelet_multi_scale",
        agent="test",
        path=path,
        lock_path=lock,
    )
    # Sister V5 probe supersedes (e.g. wavelet-based replacement runs cleanly)
    update_probe_outcome(
        probe_id="v4_hand_rolled_segfault_test_v1",
        event_type=EVENT_SUPERSEDED,
        notes="superseded by v5_wavelet_multi_scale_probe which ran without segfault",
        agent="test",
        path=path,
        lock_path=lock,
    )
    rows = query_by_probe_id("v4_hand_rolled_segfault_test_v1", path=path)
    assert len(rows) == 2
    assert rows[0]["event_type"] == EVENT_ADJUDICATED
    assert rows[1]["event_type"] == EVENT_SUPERSEDED


def test_infrastructure_failure_can_be_operator_overridden(tmp_ledger) -> None:
    """Per Catalog #313 paired-env bypass discipline: operator can explicitly
    clear an INFRASTRUCTURE_FAILURE blocker."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_operator_override_v1",
        substrate="test_substrate",
        recipe_path=".omx/operator_authorize_recipes/test.yaml",
        probe_kind="codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        agent="test",
        path=path,
        lock_path=lock,
    )
    update_probe_outcome(
        probe_id="test_operator_override_v1",
        event_type="operator_override",
        notes="operator confirmed sister probe infrastructure is corrected; clear blocker",
        agent="operator",
        path=path,
        lock_path=lock,
    )
    rows = query_by_probe_id("test_operator_override_v1", path=path)
    assert len(rows) == 2
    # Latest event is operator_override; blocker_status auto-transitions to advisory.
    assert rows[1]["event_type"] == "operator_override"
    assert rows[1]["blocker_status"] == "advisory"
    # The latest row's verdict is preserved (still INFRASTRUCTURE_FAILURE; the
    # OVERRIDE removes blocker_status, not the verdict).
    assert rows[1]["verdict"] == VERDICT_INFRASTRUCTURE_FAILURE


def test_infrastructure_failure_distinct_from_independent_in_query(tmp_ledger) -> None:
    """INDEPENDENT and INFRASTRUCTURE_FAILURE both block but are distinguishable
    in the query results (operator-facing diagnostics)."""
    path, lock = tmp_ledger
    register_probe_outcome(
        probe_id="v1_independent_probe",
        substrate="substrate_A",
        recipe_path=".omx/operator_authorize_recipes/sub_a.yaml",
        probe_kind="paradigm_disambiguation",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        agent="test",
        path=path,
        lock_path=lock,
    )
    register_probe_outcome(
        probe_id="v4_infra_failure_probe",
        substrate="substrate_B",
        recipe_path=".omx/operator_authorize_recipes/sub_b.yaml",
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        metric_name="segfault_signal",
        metric_value=1.0,
        agent="test",
        path=path,
        lock_path=lock,
    )
    blockers = query_blocking_outcomes(path=path)
    verdicts_by_id = {r["probe_id"]: r["verdict"] for r in blockers}
    assert verdicts_by_id["v1_independent_probe"] == VERDICT_INDEPENDENT
    assert verdicts_by_id["v4_infra_failure_probe"] == VERDICT_INFRASTRUCTURE_FAILURE


# ---------------------------------------------------------------------------
# F19 empirical anchor migration regression
# ---------------------------------------------------------------------------


def test_f19_anchor_migration_via_supersession(tmp_ledger) -> None:
    """Canonical APPEND-ONLY migration: the F19 anchor was registered as
    INDEPENDENT but its semantic verdict is INFRASTRUCTURE_FAILURE.
    Migration appends a new ``superseded`` event (NOT in-place mutation) so
    the canonical historical audit trail remains per Catalog #110/#113.

    The latest row's effective verdict is INFRASTRUCTURE_FAILURE; the
    original adjudicated row's INDEPENDENT verdict is preserved verbatim."""
    path, lock = tmp_ledger
    # Simulate the F19 anchor as it currently lives in canonical posterior:
    register_probe_outcome(
        probe_id="v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pair_segfault_20260518",
        substrate="atw_codec_v2_1_faiss_ivf_pq_v4_hand_rolled",
        recipe_path=None,
        probe_kind="hand_rolled_codec_validation",
        verdict=VERDICT_INDEPENDENT,
        metric_name="segfault_signal",
        metric_value=1.0,
        next_action="advance_to_v5_wavelet_multi_scale",
        notes="originally adjudicated INDEPENDENT; sister probes V1/V2 disambiguate Shannon zones",
        agent="test",
        path=path,
        lock_path=lock,
    )
    # Apply the canonical migration: APPEND a superseded event with the
    # corrected verdict.
    update_probe_outcome(
        probe_id="v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pair_segfault_20260518",
        event_type=EVENT_SUPERSEDED,
        verdict=VERDICT_INFRASTRUCTURE_FAILURE,
        notes=(
            "Slot C Catalog #313 verdict taxonomy migration 2026-05-28: "
            "re-classify from INDEPENDENT to INFRASTRUCTURE_FAILURE per "
            "canonical anti-pattern "
            "paradigm_vs_implementation_falsification_distinct_from_infrastructure_failure_v1; "
            "the V4 hand-rolled SEGFAULT is an infrastructure-implementation failure, "
            "NOT a paradigm-disambiguation empirical null. APPEND-ONLY per Catalog #110/#113."
        ),
        agent="claude",
        path=path,
        lock_path=lock,
    )
    rows = query_by_probe_id(
        "v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pair_segfault_20260518",
        path=path,
    )
    # APPEND-ONLY: 2 rows.
    assert len(rows) == 2
    # Original verdict preserved verbatim.
    assert rows[0]["verdict"] == VERDICT_INDEPENDENT
    # Migrated verdict on the superseded event.
    assert rows[1]["verdict"] == VERDICT_INFRASTRUCTURE_FAILURE
    assert rows[1]["event_type"] == EVENT_SUPERSEDED
    # The latest row's effective state (per the gate's latest-wins semantics).
    blocker = latest_blocking_outcome_by_substrate(
        "atw_codec_v2_1_faiss_ivf_pq_v4_hand_rolled",
        path=path,
    )
    assert blocker is not None
    assert blocker.verdict == VERDICT_INFRASTRUCTURE_FAILURE
