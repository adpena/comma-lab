# SPDX-License-Identifier: MIT
"""Behavior tests for the pairwise commutator ledger over ActionEffect v1.

These tests assert BEHAVIOR, not constants: a marker-stub that returned canonical
fields without computing the real commutator arithmetic would FAIL the
hand-checked-arithmetic, basis-consistency, authority-mismatch, classification,
and queue-emission tests below.

All numeric inputs are SYNTHETIC FIXTURES (hand-chosen distortion/byte endpoints)
exercised through the REAL ``ActionEffect`` + ``tac.score_geometry.contest_score``
scoring path; they carry no empirical / contest-score authority.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.action_commutator import (
    ACTION_COMMUTATOR_LEDGER_SCHEMA,
    ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA,
    ACTION_COMMUTATOR_SCHEMA,
    BASIS_NONRATE,
    BASIS_TOTAL,
    CLASSIFICATION_ADDITIVE,
    CLASSIFICATION_CONFLICTING,
    CLASSIFICATION_SYNERGISTIC,
    ActionCommutatorError,
    build_commutator_ledger,
    commutator_value,
    ledger_from_dict,
)
from tac.analysis.action_effect import (
    ACTION_EFFECT_V1_SCHEMA,
    ActionEffect,
    append_action_effect,
)
from tac.analysis.pr110_baseline_reproduction import (
    BLOCKER_AUTHORITY,
    BLOCKER_GLOBAL_K,
    BLOCKER_MISSING,
    BLOCKER_REPLAY_ROW_MISSING,
    BLOCKER_SELECTOR_PAIR_COUNT,
    MENU_ILP_BASELINE_BLOCKER,
    PR110_K16_BASELINE_REPRODUCTION_SCHEMA,
    build_pr110_k16_baseline_reproduction_from_action_effects,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

# ── fixture builders (SYNTHETIC; no empirical authority) ────────────────────


def _seg_effect(
    action_id: str,
    *,
    authority: str = "fakequant_mlx",
    normalization_scope: str | None = None,
    old_d_seg: float = 0.10,
    new_d_seg: float = 0.10,
    bytes_: int = 1000,
    archive_sha256: str | None = None,
    payload_sha256: str | None = None,
    base_state_sha256: str | None = None,
) -> ActionEffect:
    """A byte-priced (total-basis) effect that moves ONLY d_seg.

    pose held constant so ``delta_score_total == 100*(new_d_seg-old_d_seg)``
    (bytes unchanged ⇒ zero rate term), which makes commutator arithmetic
    hand-checkable.
    """

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        authority=authority,
        normalization_scope=normalization_scope,
        producer="fixture",
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=0.10,
        new_d_pose=0.10,
        old_bytes=bytes_,
        new_bytes=bytes_,
        archive_sha256=archive_sha256,
        payload_sha256=payload_sha256,
        base_state_sha256=base_state_sha256,
    )


def _nonrate_only_effect(action_id: str, *, old_d_seg: float, new_d_seg: float) -> ActionEffect:
    """A distortion-only effect (no bytes ⇒ delta_score_total is None)."""

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        authority="fakequant_mlx",
        producer="fixture",
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=0.10,
        new_d_pose=0.10,
        old_bytes=None,
        new_bytes=None,
    )


# ── 1. exact comm arithmetic (hand-checked, total basis) ────────────────────


def test_commutator_exact_synergistic_arithmetic_total_basis():
    # a: 100*(0.08-0.10) = -2.0 ; b: 100*(0.09-0.10) = -1.0 ; ab: -5.0
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    assert a.delta_score_total == pytest.approx(-2.0)
    assert b.delta_score_total == pytest.approx(-1.0)
    assert ab.delta_score_total == pytest.approx(-5.0)
    row = commutator_value(a, b, ab)
    # comm = -5.0 - (-2.0) - (-1.0) = -2.0
    assert row["comm"] == pytest.approx(-2.0)
    assert row["synergy_score_units"] == pytest.approx(2.0)
    assert row["basis"] == BASIS_TOTAL
    assert row["classification"] == CLASSIFICATION_SYNERGISTIC
    assert row["macro_action_recommended"] is True
    assert row["schema"] == ACTION_COMMUTATOR_SCHEMA


def test_commutator_exact_conflicting_arithmetic():
    # a: -2.0 ; b: -1.0 ; ab only -1.0 (composite gave back score the parts
    # promised) => comm = -1.0 - (-2.0) - (-1.0) = +2.0  => conflicting
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.09)
    row = commutator_value(a, b, ab)
    assert row["comm"] == pytest.approx(2.0)
    assert row["synergy_score_units"] == pytest.approx(-2.0)
    assert row["classification"] == CLASSIFICATION_CONFLICTING
    assert row["macro_action_recommended"] is False


def test_commutator_exact_additive_arithmetic():
    # ab delta == a + b exactly => comm == 0 => additive
    a = _seg_effect("A", new_d_seg=0.08)  # -2.0
    b = _seg_effect("B", new_d_seg=0.09)  # -1.0
    ab = _seg_effect("A__then__B", new_d_seg=0.07)  # -3.0 == (-2)+(-1)
    row = commutator_value(a, b, ab)
    assert row["comm"] == pytest.approx(0.0, abs=1e-9)
    assert row["classification"] == CLASSIFICATION_ADDITIVE
    assert row["macro_action_recommended"] is False


def test_commutator_noop_identity_is_additive_both_orders():
    noop = _seg_effect("noop", new_d_seg=0.10)
    a = _seg_effect("A", new_d_seg=0.08)
    a_then_noop = _seg_effect("A__then__noop", new_d_seg=0.08)
    noop_then_a = _seg_effect("noop__then__A", new_d_seg=0.08)

    assert noop.delta_score_total == pytest.approx(0.0)
    assert noop.delta_bytes == 0

    right_identity = commutator_value(a, noop, a_then_noop)
    left_identity = commutator_value(noop, a, noop_then_a)
    assert right_identity["comm"] == pytest.approx(0.0, abs=1e-9)
    assert left_identity["comm"] == pytest.approx(0.0, abs=1e-9)
    assert right_identity["classification"] == CLASSIFICATION_ADDITIVE
    assert left_identity["classification"] == CLASSIFICATION_ADDITIVE


# ── 2. classification thresholds honor eps ──────────────────────────────────


def test_classification_thresholds_respect_eps():
    a = _seg_effect("A", new_d_seg=0.10)  # delta 0
    b = _seg_effect("B", new_d_seg=0.10)  # delta 0
    # ab moves seg by -0.0005 -> delta -0.05 ; comm = -0.05 - 0 - 0 = -0.05
    ab = _seg_effect("A__then__B", new_d_seg=0.0995)
    # with a small eps it is synergistic
    small = commutator_value(a, b, ab, eps=1e-3)
    assert small["comm"] == pytest.approx(-0.05)
    assert small["classification"] == CLASSIFICATION_SYNERGISTIC
    # with a large eps (band wider than |comm|) it reads additive
    large = commutator_value(a, b, ab, eps=1.0)
    assert large["classification"] == CLASSIFICATION_ADDITIVE


def test_negative_eps_rejected():
    a = _seg_effect("A")
    with pytest.raises(ValueError):
        commutator_value(a, a, a, eps=-1e-9)


# ── 3. basis consistency rule (NEVER mix total and nonrate) ─────────────────


def test_basis_falls_back_to_nonrate_when_any_row_lacks_bytes():
    # a and ab are byte-priced (total available); b is distortion-only (no bytes).
    # The rule: if ANY of the three lacks total, use nonrate for ALL THREE.
    a = _seg_effect("A", new_d_seg=0.08)  # total -2.0, nonrate -2.0
    b = _nonrate_only_effect("B", old_d_seg=0.10, new_d_seg=0.09)  # total None, nonrate -1.0
    ab = _seg_effect("A__then__B", new_d_seg=0.05)  # total -5.0, nonrate -5.0
    assert b.delta_score_total is None
    assert b.delta_score_nonrate == pytest.approx(-1.0)
    row = commutator_value(a, b, ab)
    assert row["basis"] == BASIS_NONRATE
    # nonrate deltas: a=-2.0, b=-1.0, ab=-5.0 => comm -2.0
    assert row["delta_a"] == pytest.approx(-2.0)
    assert row["delta_b"] == pytest.approx(-1.0)
    assert row["delta_ab"] == pytest.approx(-5.0)
    assert row["comm"] == pytest.approx(-2.0)


def test_basis_uses_total_when_all_three_byte_priced():
    # Add a real rate movement so total != nonrate, proving total is the basis used.
    a = ActionEffect.build(
        action_id="A", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=0.10, new_d_seg=0.10, old_d_pose=0.10, new_d_pose=0.10,
        old_bytes=1000, new_bytes=2000,  # +1000 bytes -> positive rate delta
    )
    b = _seg_effect("B", new_d_seg=0.10)  # all-zero delta
    ab = ActionEffect.build(
        action_id="A__then__B", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=0.10, new_d_seg=0.10, old_d_pose=0.10, new_d_pose=0.10,
        old_bytes=1000, new_bytes=2000,
    )
    row = commutator_value(a, b, ab)
    assert row["basis"] == BASIS_TOTAL
    # a total = rate delta only (nonzero); nonrate would be 0 -> proves total used
    assert a.delta_score_nonrate == pytest.approx(0.0)
    assert a.delta_score_total != pytest.approx(0.0)
    assert row["delta_a"] == pytest.approx(a.delta_score_total)


def test_commutator_undefined_when_no_consistent_basis_raises():
    # ab has neither total nor nonrate (no distortion endpoints, no bytes).
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = ActionEffect.build(
        action_id="A__then__B", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=None, new_d_seg=None, old_d_pose=None, new_d_pose=None,
        old_bytes=None, new_bytes=None,
    )
    assert ab.delta_score_total is None
    assert ab.delta_score_nonrate is None
    with pytest.raises(ActionCommutatorError):
        commutator_value(a, b, ab)


# ── 4. authority is a type (mismatch raises) ────────────────────────────────


def test_authority_mismatch_raises():
    a = _seg_effect("A", new_d_seg=0.08, authority="fakequant_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="inflated_torch_cpu")
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="fakequant_mlx")
    with pytest.raises(ActionCommutatorError) as exc:
        commutator_value(a, b, ab)
    assert "authority" in str(exc.value).lower()


def test_authority_matches_returns_that_authority():
    a = _seg_effect("A", new_d_seg=0.08, authority="parseback_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="parseback_mlx")
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="parseback_mlx")
    row = commutator_value(a, b, ab)
    assert row["authority"] == "parseback_mlx"


def test_measured_commutator_rejects_normalization_scope_mismatch():
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="same_advisory_surface",
        normalization_scope="batch_local",
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="same_advisory_surface",
        normalization_scope="full_video_exact",
    )
    ab = _seg_effect(
        "A__then__B",
        new_d_seg=0.05,
        authority="same_advisory_surface",
        normalization_scope="batch_local",
    )

    with pytest.raises(ActionCommutatorError, match="normalization_scope mismatch"):
        commutator_value(a, b, ab)


def test_measured_commutator_rejects_mismatched_base_hashes():
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="contest_cpu",
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="contest_cpu",
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )
    ab = _seg_effect(
        "A__then__B",
        new_d_seg=0.05,
        authority="contest_cpu",
        archive_sha256="c" * 64,
        payload_sha256="b" * 64,
    )

    with pytest.raises(ActionCommutatorError, match="archive_sha256 mismatch"):
        commutator_value(a, b, ab)


def test_measured_commutator_rejects_partially_missing_base_hashes():
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="contest_cpu",
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="contest_cpu",
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="contest_cpu")

    with pytest.raises(ActionCommutatorError, match="archive_sha256 missing"):
        commutator_value(a, b, ab)


def test_measured_commutator_carries_shared_base_hashes_when_present():
    archive_hash = "a" * 64
    payload_hash = "b" * 64
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="contest_cpu",
        archive_sha256=archive_hash,
        payload_sha256=payload_hash,
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="contest_cpu",
        archive_sha256=archive_hash,
        payload_sha256=payload_hash,
    )
    ab = _seg_effect(
        "A__then__B",
        new_d_seg=0.05,
        authority="contest_cpu",
        archive_sha256=archive_hash,
        payload_sha256=payload_hash,
    )

    row = commutator_value(a, b, ab)
    assert row["base_archive_sha256"] == archive_hash
    assert row["base_payload_sha256"] == payload_hash


# ── 5. false-authority markers present (planning row, never a score claim) ───


def test_commutator_row_carries_false_authority_markers():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    row = commutator_value(a, b, ab)
    for key, expected in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False


# ── 6. ledger: measured vs needs-measurement queue ──────────────────────────


def test_ledger_emits_measured_row_and_queue_for_missing_reverse_pair():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)  # only the (A,B) order measured
    ledger = build_commutator_ledger([a, b], [ab])
    assert ledger["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    assert ledger["ordered_pair_count"] == 2  # (A,B) and (B,A)
    assert ledger["measured_commutator_count"] == 1
    assert ledger["needs_measurement_count"] == 1
    # the measured one is the synergistic (A,B)
    assert ledger["macro_action_candidates"][0]["first_action_id"] == "A"
    assert ledger["macro_action_candidates"][0]["second_action_id"] == "B"
    # the queued one is the (B,A) order
    q = ledger["measurement_queue"][0]
    assert q["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA
    assert q["first_action_id"] == "B"
    assert q["second_action_id"] == "A"
    assert q["proposed_composite_action_id"] == "B__then__A"
    assert q["comm"] is None  # NEVER fabricated
    assert q["additive_delta_score_total"] == pytest.approx(a.delta_score_total + b.delta_score_total)
    assert q["additive_delta_bytes"] == 0
    assert q["byte_cost"] == 0
    assert q["first_measurement_command"].startswith("uv run python tools/run_pr110_commutator_ledger.py")
    assert q["measurement_command_blockers"] == [
        "composite_action_effect_row_missing",
        "action_effect_base_archive_hash_missing",
        "action_effect_base_payload_hash_missing",
        "action_effect_base_state_hash_missing",
    ]


def test_ledger_matches_measured_composites_by_order_not_unordered_pair():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ba = _seg_effect("B__then__A", new_d_seg=0.095)
    ledger = build_commutator_ledger([a, b], [ba])

    assert ledger["ordered_pair_count"] == 2
    assert ledger["measured_commutator_count"] == 1
    measured = ledger["rows"][0]
    assert measured["first_action_id"] == "B"
    assert measured["second_action_id"] == "A"
    assert measured["composed_action_id"] == "B__then__A"
    queued = ledger["measurement_queue"][0]
    assert queued["first_action_id"] == "A"
    assert queued["second_action_id"] == "B"
    assert queued["proposed_composite_action_id"] == "A__then__B"


def test_ledger_queue_when_no_pair_effects_at_all():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    c = _seg_effect("C", new_d_seg=0.07)
    ledger = build_commutator_ledger([a, b, c], [])
    # 3 distinct singles -> 3*2 = 6 ordered pairs, all unmeasured
    assert ledger["ordered_pair_count"] == 6
    assert ledger["measured_commutator_count"] == 0
    assert ledger["needs_measurement_count"] == 6
    assert all(r["comm"] is None for r in ledger["measurement_queue"])
    assert all("first_measurement_command" in r for r in ledger["measurement_queue"])
    assert [r["measurement_priority_rank"] for r in ledger["measurement_queue"]] == list(range(1, 7))
    assert all(r["measurement_priority_basis"] == "total" for r in ledger["measurement_queue"])
    assert ledger["measurement_queue"][0]["additive_score_improvement_total"] == pytest.approx(5.0)
    assert {
        ledger["measurement_queue"][0]["first_action_id"],
        ledger["measurement_queue"][0]["second_action_id"],
    } == {"A", "C"}
    assert ledger["policy"]["measurement_queue_ranked_by_expected_additive_score_authority_and_byte_cost"] is True


def test_ledger_queues_pair_with_incompatible_authority_never_fabricates():
    a = _seg_effect("A", new_d_seg=0.08, authority="fakequant_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="inflated_torch_cpu")
    # measured composite exists but parts disagree on authority -> queued, not measured
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="fakequant_mlx")
    ledger = build_commutator_ledger([a, b], [ab])
    assert ledger["measured_commutator_count"] == 0
    assert ledger["needs_measurement_count"] == 2
    incompat = [r for r in ledger["measurement_queue"] if r["first_action_id"] == "A" and r["second_action_id"] == "B"]
    assert len(incompat) == 1
    assert incompat[0]["authority_compatible"] is False
    assert "incompatible" in incompat[0]["reason"]
    assert "action_effect_authority_mismatch" in incompat[0]["measurement_command_blockers"]


def test_measurement_queue_blocks_normalization_scope_mismatch() -> None:
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="same_advisory_surface",
        normalization_scope="batch_local",
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="same_advisory_surface",
        normalization_scope="full_video_exact",
    )
    ledger = build_commutator_ledger([a, b], [])

    row = ledger["measurement_queue"][0]
    assert row["authority_compatible"] is True
    assert row["normalization_scope_compatible"] is False
    assert "action_effect_authority_mismatch" not in row["measurement_command_blockers"]
    assert "action_effect_normalization_scope_mismatch" in row["measurement_command_blockers"]


def test_measurement_queue_carries_base_identity_and_blocks_when_missing():
    a = _seg_effect("A", new_d_seg=0.08, authority="contest_cpu")
    b = _seg_effect("B", new_d_seg=0.09, authority="contest_cpu")
    ledger = build_commutator_ledger([a, b], [])

    row = ledger["measurement_queue"][0]
    assert row["first_authority"] == "contest_cpu"
    assert row["second_authority"] == "contest_cpu"
    assert row["first_normalization_scope"] == "full_video_exact"
    assert row["second_normalization_scope"] == "full_video_exact"
    assert row["normalization_scope_compatible"] is True
    assert row["base_archive_hash"] is None
    assert row["base_payload_hash"] is None
    assert "action_effect_base_archive_hash_missing" in row["measurement_command_blockers"]
    assert "action_effect_base_payload_hash_missing" in row["measurement_command_blockers"]
    assert "action_effect_base_state_hash_missing" in row["measurement_command_blockers"]


def test_measurement_queue_exposes_shared_base_hashes_when_present():
    archive_hash = "a" * 64
    payload_hash = "b" * 64
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="contest_cpu",
        archive_sha256=archive_hash,
        payload_sha256=payload_hash,
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="contest_cpu",
        archive_sha256=archive_hash,
        payload_sha256=payload_hash,
    )
    ledger = build_commutator_ledger([a, b], [])

    row = ledger["measurement_queue"][0]
    assert row["base_archive_hash"] == archive_hash
    assert row["base_payload_hash"] == payload_hash
    assert "action_effect_base_archive_hash_missing" not in row["measurement_command_blockers"]
    assert "action_effect_base_payload_hash_missing" not in row["measurement_command_blockers"]
    assert "action_effect_base_state_hash_missing" in row["measurement_command_blockers"]
    assert "action_effect_base_archive_hash_mismatch" not in row["measurement_command_blockers"]
    assert "action_effect_base_payload_hash_mismatch" not in row["measurement_command_blockers"]


def test_measurement_queue_accepts_shared_batch_local_base_state_hash():
    base_state_hash = "e" * 64
    a = _seg_effect("A", new_d_seg=0.08, base_state_sha256=base_state_hash)
    b = _seg_effect("B", new_d_seg=0.09, base_state_sha256=base_state_hash)
    ledger = build_commutator_ledger([a, b], [])

    row = ledger["measurement_queue"][0]
    assert row["base_state_hash"] == base_state_hash
    assert "action_effect_base_archive_hash_missing" not in row["measurement_command_blockers"]
    assert "action_effect_base_payload_hash_missing" not in row["measurement_command_blockers"]
    assert "action_effect_base_state_hash_missing" not in row["measurement_command_blockers"]


def test_measurement_queue_blocks_when_base_hashes_differ():
    a = _seg_effect(
        "A",
        new_d_seg=0.08,
        authority="contest_cpu",
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )
    b = _seg_effect(
        "B",
        new_d_seg=0.09,
        authority="contest_cpu",
        archive_sha256="c" * 64,
        payload_sha256="d" * 64,
    )
    ledger = build_commutator_ledger([a, b], [])

    row = ledger["measurement_queue"][0]
    assert row["base_archive_hash"] is None
    assert row["base_payload_hash"] is None
    assert "action_effect_base_archive_hash_mismatch" in row["measurement_command_blockers"]
    assert "action_effect_base_payload_hash_mismatch" in row["measurement_command_blockers"]


def test_measurement_queue_ranks_authority_compatible_before_cross_authority():
    weak_compatible = _seg_effect("weakA", new_d_seg=0.099, authority="contest_cpu")
    weak_compatible_b = _seg_effect("weakB", new_d_seg=0.099, authority="contest_cpu")
    strong_cross = _seg_effect("strongX", new_d_seg=0.001, authority="receiver_closed_frontier_rate_attack")
    ledger = build_commutator_ledger([weak_compatible, weak_compatible_b, strong_cross], [])

    queue = ledger["measurement_queue"]
    assert queue[0]["authority_compatible"] is True
    assert {queue[0]["first_action_id"], queue[0]["second_action_id"]} == {"weakA", "weakB"}
    incompatible = [row for row in queue if row["authority_compatible"] is False]
    assert incompatible
    assert min(row["measurement_priority_rank"] for row in incompatible) > 2


def test_ledger_separates_synergistic_and_conflicting_and_sorts():
    a = _seg_effect("A", new_d_seg=0.08)  # -2.0
    b = _seg_effect("B", new_d_seg=0.09)  # -1.0
    # (A,B) strongly synergistic: ab -6.0 -> comm -3.0
    ab = _seg_effect("A__then__B", new_d_seg=0.04)
    # (B,A) conflicting: ba -0.5 -> comm = -0.5 -(-1)-(-2) = +2.5
    ba = _seg_effect("B__then__A", new_d_seg=0.095)
    ledger = build_commutator_ledger([a, b], [ab, ba])
    assert ledger["measured_commutator_count"] == 2
    assert ledger["synergistic_count"] == 1
    assert ledger["conflicting_count"] == 1
    assert ledger["macro_action_candidates"][0]["comm"] == pytest.approx(-3.0)
    assert ledger["conflict_pairs"][0]["comm"] == pytest.approx(2.5)


def test_ledger_macro_candidates_sorted_most_synergistic_first():
    a = _seg_effect("A", new_d_seg=0.10)
    b = _seg_effect("B", new_d_seg=0.10)
    c = _seg_effect("C", new_d_seg=0.10)
    # A->B comm -1.0 ; A->C comm -3.0 (more synergistic) -> C should rank first
    ab = _seg_effect("A__then__B", new_d_seg=0.099)  # delta -0.1 ; comm -0.1
    ac = _seg_effect("A__then__C", new_d_seg=0.097)  # delta -0.3 ; comm -0.3
    ledger = build_commutator_ledger([a, b, c], [ab, ac])
    macros = ledger["macro_action_candidates"]
    assert len(macros) == 2
    assert macros[0]["comm"] <= macros[1]["comm"]
    assert macros[0]["second_action_id"] == "C"


def test_ledger_respects_top_k_caps():
    singles = [_seg_effect(f"A{i}", new_d_seg=0.10) for i in range(5)]
    # build several synergistic composites
    pairs = []
    for i in range(4):
        pairs.append(_seg_effect(f"A0__then__A{i + 1}", new_d_seg=0.10 - 0.001 * (i + 1)))
    ledger = build_commutator_ledger(singles, pairs, macro_action_limit=2, conflict_pair_limit=2)
    assert len(ledger["macro_action_candidates"]) <= 2


# ── 7. duplicate / self-composition handling ────────────────────────────────


def test_ledger_skips_self_and_duplicate_ids():
    a = _seg_effect("A", new_d_seg=0.08)
    a_dup = _seg_effect("A", new_d_seg=0.09)  # same id, different deltas
    ledger = build_commutator_ledger([a, a_dup], [])
    # one unique id -> zero ordered pairs
    assert ledger["ordered_pair_count"] == 0
    assert ledger["needs_measurement_count"] == 0


# ── 8. input type guards ────────────────────────────────────────────────────


def test_build_ledger_rejects_non_action_effect_inputs():
    with pytest.raises(TypeError):
        build_commutator_ledger([{"action_id": "A"}], [])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        build_commutator_ledger("not a sequence", [])  # type: ignore[arg-type]


def test_commutator_value_rejects_non_action_effect():
    a = _seg_effect("A", new_d_seg=0.08)
    with pytest.raises(TypeError):
        commutator_value(a, a, {"action_id": "A__then__A"})  # type: ignore[arg-type]


# ── 9. JSONL ledger round-trip ──────────────────────────────────────────────


def test_ledger_jsonl_round_trip(tmp_path: Path):
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    ledger = build_commutator_ledger([a, b], [ab])
    out = tmp_path / "ledger.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in ledger["rows"]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        for row in ledger["measurement_queue"]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    reloaded = []
    with open(out, encoding="utf-8") as fh:
        for line in fh:
            reloaded.append(json.loads(line))
    measured = [r for r in reloaded if r["schema"] == ACTION_COMMUTATOR_SCHEMA]
    queued = [r for r in reloaded if r["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA]
    assert len(measured) == 1
    assert measured[0]["comm"] == pytest.approx(-2.0)
    assert len(queued) == 1
    assert queued[0]["comm"] is None


def test_ledger_from_dict_validates_schema():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    ledger = build_commutator_ledger([a, b], [ab])
    round_tripped = ledger_from_dict(json.loads(json.dumps(ledger)))
    assert round_tripped["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    with pytest.raises(ValueError):
        ledger_from_dict({"schema": "not_a_commutator_ledger"})


# ── 10. CLI smoke on tmp fixtures (fixtures labeled synthetic) ──────────────


def _write_effect_jsonl(effects: list[ActionEffect], path: Path) -> None:
    """Write ActionEffect rows via the canonical fcntl-locked appender."""

    for effect in effects:
        append_action_effect(effect, path)


def test_cli_smoke_emits_ledger_and_summary(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "out"

    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    _write_effect_jsonl([a, b], singles_path)
    _write_effect_jsonl([ab], pairs_path)

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--pair-effects",
        str(pairs_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    assert summary["measured_commutator_count"] == 1
    assert summary["needs_measurement_count"] == 1
    assert summary["menu_ilp_allowed"] is False
    assert summary["macro_action_promotion_allowed"] is False
    assert summary["menu_ilp_blockers"] == [MENU_ILP_BASELINE_BLOCKER, BLOCKER_MISSING]
    assert summary["macro_action_candidates"][0]["comm"] == pytest.approx(-2.0)
    assert summary["macro_action_candidates"][0]["menu_ilp_blockers"] == [
        MENU_ILP_BASELINE_BLOCKER,
        BLOCKER_MISSING,
    ]
    # false-authority markers propagate to the summary
    assert summary["score_claim"] is False

    jsonl_rows = [
        json.loads(line) for line in (out_dir / "commutator_ledger.jsonl").read_text().splitlines() if line.strip()
    ]
    assert any(r["schema"] == ACTION_COMMUTATOR_SCHEMA for r in jsonl_rows)
    assert any(r["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA for r in jsonl_rows)
    assert all(MENU_ILP_BASELINE_BLOCKER in r["menu_ilp_blockers"] for r in jsonl_rows)


def test_cli_smoke_no_pair_effects_all_queued(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    out_dir = tmp_path / "out2"
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    _write_effect_jsonl([a, b], singles_path)

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["measured_commutator_count"] == 0
    assert summary["needs_measurement_count"] == 2
    assert summary["menu_ilp_allowed"] is False
    assert summary["measurement_queue"][0]["first_measurement_command"] == (
        "uv run python tools/run_pr110_commutator_ledger.py "
        f"--action-effects {singles_path.as_posix()} "
        f"--pair-effects {(out_dir / 'pr110_composite_action_effects.jsonl').as_posix()} "
        f"--output {out_dir.as_posix()}"
    )
    assert summary["measurement_queue"][0]["additive_delta_score_total"] == pytest.approx(
        a.delta_score_total + b.delta_score_total
    )
    assert summary["measurement_queue"][0]["menu_ilp_blockers"] == [MENU_ILP_BASELINE_BLOCKER, BLOCKER_MISSING]


def _write_pr110_k16_baseline(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema": PR110_K16_BASELINE_REPRODUCTION_SCHEMA,
        "passed": True,
        "global_k": 16,
        "selector_id": "synthetic-pr110-global-k16",
        "authority": "contest_cpu",
        "expected_archive_bytes": 178517,
        "actual_archive_bytes": 178517,
        "byte_error_abs": 0,
        "byte_tolerance": 0,
        "expected_score": 0.192051,
        "actual_score": 0.192051,
        "score_error_abs": 0.0,
        "score_tolerance": 0.0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pr110_selector_effect(
    *,
    action_id: str = "pr110-global-k16-selector",
    action_kind: str = "selector_replay",
    pair_count: int = 600,
    payload_sections: tuple[str, ...] = ("selector_global_k16",),
) -> ActionEffect:
    return ActionEffect.build(
        action_id=action_id,
        family="pr110",
        action_kind=action_kind,
        authority="contest_cpu",
        normalization_scope="full_video_exact",
        producer="unit_test_pr110_replay",
        pair_ids=range(pair_count),
        payload_sections=payload_sections,
        old_d_seg=0.001,
        new_d_seg=0.0009,
        old_d_pose=0.0001,
        new_d_pose=0.00009,
        old_bytes=178_517,
        new_bytes=178_517,
        archive_sha256="a" * 64,
        payload_sha256="b" * 64,
    )


def test_pr110_k16_baseline_builder_accepts_full_selector_replay_action_effect() -> None:
    proof = build_pr110_k16_baseline_reproduction_from_action_effects(
        [_pr110_selector_effect()],
        expected_global_k=16,
        expected_pair_count=600,
        byte_tolerance=0,
        score_tolerance=0.0,
        source="unit_test",
    )

    assert proof["schema"] == PR110_K16_BASELINE_REPRODUCTION_SCHEMA
    assert proof["passed"] is True
    assert proof["blockers"] == []
    assert proof["global_k"] == 16
    assert proof["pair_count"] == 600
    assert proof["selector_bits"] == 2400
    assert proof["score_claim"] is False


def test_pr110_k16_baseline_builder_rejects_sparse_selector_mode() -> None:
    proof = build_pr110_k16_baseline_reproduction_from_action_effects(
        [
            _pr110_selector_effect(
                action_id="pr110-sparse-selector",
                action_kind="selector_mode",
                pair_count=1,
                payload_sections=("selector_k1",),
            )
        ],
        expected_global_k=16,
        expected_pair_count=600,
        source="unit_test",
    )

    assert proof["passed"] is False
    assert BLOCKER_GLOBAL_K in proof["blockers"]
    assert BLOCKER_SELECTOR_PAIR_COUNT in proof["blockers"]
    assert proof["global_k"] == 1
    assert proof["pair_count"] == 1


def test_pr110_k16_baseline_builder_rejects_missing_replay_row() -> None:
    proof = build_pr110_k16_baseline_reproduction_from_action_effects([], source="unit_test")

    assert proof["passed"] is False
    assert BLOCKER_REPLAY_ROW_MISSING in proof["blockers"]


def test_cli_valid_k16_baseline_unblocks_menu_ilp(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    out_dir = tmp_path / "out_valid_baseline"
    baseline_path = tmp_path / "baseline.json"
    _write_effect_jsonl([_seg_effect("A", new_d_seg=0.08), _seg_effect("B", new_d_seg=0.09)], singles_path)
    _write_pr110_k16_baseline(baseline_path)

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--baseline-reproduction",
        str(baseline_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["baseline_reproduction"]["passed"] is True
    assert summary["menu_ilp_allowed"] is True
    assert summary["macro_action_promotion_allowed"] is True
    assert summary["menu_ilp_blockers"] == []
    assert all(row["menu_ilp_allowed"] is True for row in summary["measurement_queue"])
    assert f"--baseline-reproduction {baseline_path.as_posix()}" in summary["measurement_queue"][0][
        "first_measurement_command"
    ]


def test_cli_invalid_k16_baseline_keeps_exact_blockers(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    out_dir = tmp_path / "out_invalid_baseline"
    baseline_path = tmp_path / "baseline_bad.json"
    _write_effect_jsonl([_seg_effect("A", new_d_seg=0.08), _seg_effect("B", new_d_seg=0.09)], singles_path)
    _write_pr110_k16_baseline(baseline_path, global_k=8, authority="")

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--baseline-reproduction",
        str(baseline_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["baseline_reproduction"]["passed"] is False
    assert summary["menu_ilp_allowed"] is False
    assert summary["menu_ilp_blockers"] == [
        MENU_ILP_BASELINE_BLOCKER,
        BLOCKER_GLOBAL_K,
        BLOCKER_AUTHORITY,
    ]
    assert summary["measurement_queue"][0]["menu_ilp_blockers"] == summary["menu_ilp_blockers"]


def test_cli_missing_action_effects_file_errors(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(tmp_path / "does_not_exist.jsonl"),
        "--output",
        str(tmp_path / "out3"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode != 0


# ── 11. round-trip fidelity: schema constants are the real v1 names ──────────


def test_schema_constants_are_distinct_and_present():
    # guards against accidental schema collision with the ActionEffect surface
    assert ACTION_COMMUTATOR_SCHEMA == "tac.action_commutator.v1"
    assert ACTION_COMMUTATOR_LEDGER_SCHEMA == "tac.action_commutator_ledger.v1"
    assert ACTION_EFFECT_V1_SCHEMA == "tac.action_effect.v1"
    assert ACTION_COMMUTATOR_SCHEMA != ACTION_EFFECT_V1_SCHEMA


def test_synergy_is_exact_negative_of_comm():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    row = commutator_value(a, b, ab)
    assert math.isclose(row["synergy_score_units"], -row["comm"], rel_tol=0, abs_tol=0)
