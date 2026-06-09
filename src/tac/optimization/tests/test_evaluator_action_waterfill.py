"""Tests for the evaluator-action waterfilling law (every atom pays rent)."""

from __future__ import annotations

from tac.optimization.evaluator_action_waterfill import (
    CandidateActionEvaluation,
    action_commutator,
    waterfill_select_actions,
)


def _sidecar_degenerate() -> CandidateActionEvaluation:
    # The 2026-06-08 incident: backend survives (low d_seg, fewer bytes); the
    # sidecar WORSENS d_seg (lost region wins) AND adds ~7.3 KB -> negative twice.
    return CandidateActionEvaluation(
        action_id="sidecar_v9",
        action_kind="target_region_action",
        base_archive_sha256="base_backend_only",
        with_action_archive_sha256="base_plus_sidecar",
        d_seg_base=0.050,
        d_pose_base=0.001,
        bytes_base=9064,
        d_seg_with_action=0.108,  # worse: ~11303 region wins lost
        d_pose_with_action=0.001,
        bytes_with_action=16326,  # +7262 bytes
        scorer_effect_survived=False,
        backend_wrong_to_target=11306,
        with_action_wrong_to_target=3,
    )


def test_degenerate_sidecar_is_rejected_by_rent_law() -> None:
    ev = _sidecar_degenerate()
    assert ev.delta_score_total > 0.0  # score WORSENS
    assert ev.delta_bytes == 7262
    assert ev.pays_rent is False
    assert ev.to_row()["verdict"] == "reject"


def test_backend_only_strip_is_the_admitted_double_win() -> None:
    # Dropping the sidecar = the inverse action: base = with-sidecar, action =
    # strip -> backend-only.  Lowers d_seg (wins restored) AND frees bytes.
    strip = CandidateActionEvaluation(
        action_id="strip_sidecar_v9",
        action_kind="drop_target_region_action",
        base_archive_sha256="base_plus_sidecar",
        with_action_archive_sha256="base_backend_only",
        d_seg_base=0.108,
        d_pose_base=0.001,
        bytes_base=16326,
        d_seg_with_action=0.050,  # better: wins restored
        d_pose_with_action=0.001,
        bytes_with_action=9064,  # -7262 bytes
        scorer_effect_survived=True,
    )
    assert strip.delta_score_total < 0.0  # DOUBLE WIN
    assert strip.delta_bytes == -7262
    assert strip.pays_rent is True
    # byte-freeing action with score reduction => +inf value-per-byte (ranked first)
    assert strip.value_per_byte == float("inf")


def test_rent_paying_action_admitted() -> None:
    ev = CandidateActionEvaluation(
        action_id="good_atom",
        action_kind="margin_normal_pixel",
        base_archive_sha256="P",
        with_action_archive_sha256="P_plus_a",
        d_seg_base=0.100,
        d_pose_base=0.001,
        bytes_base=9000,
        d_seg_with_action=0.090,  # -0.01 d_seg => -1.0 score
        d_pose_with_action=0.001,
        bytes_with_action=9100,  # +100 bytes => +0.0000666 score
        scorer_effect_survived=True,
    )
    assert ev.pays_rent is True
    assert ev.delta_score_total < 0.0
    assert ev.value_per_byte is not None and ev.value_per_byte > 0.0


def test_anti_drift_staleness_when_base_changes() -> None:
    ev = _sidecar_degenerate()
    # Same base -> not stale.
    assert ev.is_stale_for_base("base_backend_only") is False
    # Different base (the phantom-base failure mode) -> STALE, must re-measure.
    assert ev.is_stale_for_base("a_different_base") is True
    # scorer-state-hash drift also triggers staleness.
    ev2 = CandidateActionEvaluation(
        action_id="x", action_kind="k", base_archive_sha256="P",
        with_action_archive_sha256="Pa", d_seg_base=0.0, d_pose_base=0.0, bytes_base=1,
        d_seg_with_action=0.0, d_pose_with_action=0.0, bytes_with_action=1,
        base_scorer_state_hash="scorer_v1",
    )
    assert ev2.is_stale_for_base("P", current_base_scorer_state_hash="scorer_v2") is True
    assert ev2.is_stale_for_base("P", current_base_scorer_state_hash="scorer_v1") is False


def test_action_commutator_signs() -> None:
    assert action_commutator(-1.0, -1.0, -3.0) == -1.0  # synergistic
    assert action_commutator(-1.0, -1.0, -1.0) == 1.0  # interfering
    assert action_commutator(-1.0, -1.0, -2.0) == 0.0  # additive


def test_waterfill_select_filters_ranks_and_drops_stale() -> None:
    good = CandidateActionEvaluation(
        action_id="good", action_kind="k", base_archive_sha256="P",
        with_action_archive_sha256="Pg", d_seg_base=0.1, d_pose_base=0.0, bytes_base=9000,
        d_seg_with_action=0.09, d_pose_with_action=0.0, bytes_with_action=9100,
        scorer_effect_survived=True,
    )
    # A degenerate atom ON BASE P (rejected for rent, NOT stale).
    bad = CandidateActionEvaluation(
        action_id="bad", action_kind="target_region_action", base_archive_sha256="P",
        with_action_archive_sha256="Pb", d_seg_base=0.05, d_pose_base=0.001, bytes_base=9064,
        d_seg_with_action=0.108, d_pose_with_action=0.001, bytes_with_action=16326,
        scorer_effect_survived=False,
    )
    stale = CandidateActionEvaluation(
        action_id="stale", action_kind="k", base_archive_sha256="OLD_BASE",
        with_action_archive_sha256="x", d_seg_base=0.1, d_pose_base=0.0, bytes_base=9000,
        d_seg_with_action=0.05, d_pose_with_action=0.0, bytes_with_action=9001,
        scorer_effect_survived=True,
    )
    out = waterfill_select_actions([good, bad, stale], current_base_archive_sha256="P")
    assert out["n_stale"] == 1  # OLD_BASE dropped
    assert out["best_action_id"] == "good"  # only rent-payer on base P
    assert out["n_admissible"] == 1
    assert out["requires_recompute_after_accept"] is True
