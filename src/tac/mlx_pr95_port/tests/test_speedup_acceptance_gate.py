# SPDX-License-Identifier: MIT
"""Tests for the BOTH-TERMS speedup acceptance gate.

The load-bearing test is ``test_d_pose_divergent_candidate_is_rejected`` — it
proves the gate REJECTS the exact n600 failure (perfect d_seg, diverging
d_pose). If that test ever fails, the gate has regressed to the d_seg-only blind
spot the incident exposed.

The trajectories here are SYNTHETIC by design: the gate is PURE acceptance
LOGIC over already-measured trajectories, so its contract is tested with crafted
trajectories that reproduce the documented regimes (clean descent, pose
divergence, seg gap, n8-provisional). The REAL torch-CPU measurement that feeds
the gate lives in ``experiments/measure_descent_equivalence.py`` and the
speedup-gate CLI; this suite tests the verdict logic, not the scorer.
"""

from __future__ import annotations

import pytest

from tac.mlx_pr95_port.speedup_acceptance_gate import (
    DEFAULT_MIN_TRUSTWORTHY_N,
    DSegOnlyGateMisuse,
    EpochMetric,
    GateConfig,
    evaluate_descent_equivalence,
    gradient_cosine_precheck_verdict,
)


def _clean_descent_traj(seg0=0.030, seg1=0.0040, pose0=0.10, pose1=3.0e-5):
    """A trajectory where BOTH terms descend monotonically (a good speedup)."""
    return [
        {"epoch": 0, "exact_d_seg": seg0, "mean_d_pose": pose0},
        {"epoch": 10, "exact_d_seg": (seg0 + seg1) / 2, "mean_d_pose": (pose0 + pose1) / 2},
        {"epoch": 20, "exact_d_seg": seg1, "mean_d_pose": pose1},
    ]


# --- the canonical n600 incident: seg-correct, pose-diverging ----------------

def test_d_pose_divergent_candidate_is_rejected():
    """THE load-bearing test: perfect d_seg + diverging d_pose => REJECT.

    This is the exact n600 incident reproduced as a trajectory: d_seg descends
    fine, d_pose explodes 0.835 -> 6.94 -> 36.46. A d_seg-only gate would PASS
    this; the BOTH-terms gate MUST reject it.
    """
    baseline = _clean_descent_traj()
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.028, "mean_d_pose": 0.835},
        {"epoch": 20, "exact_d_seg": 0.0042, "mean_d_pose": 36.46},  # seg matches baseline final!
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is False
    assert verdict.seg.tracks_within_tol is True  # seg is fine — the trap
    assert verdict.pose.tracks_within_tol is False
    assert verdict.pose.diverged is True
    # The reason must name the BOTH-terms lesson explicitly.
    joined = " ".join(verdict.reasons)
    assert "n600" in joined
    assert "d_seg-only" in joined


def test_seg_only_pass_would_have_been_wrong_is_flagged():
    """When seg passes but pose fails, the verdict explicitly notes the trap."""
    baseline = _clean_descent_traj()
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.020, "mean_d_pose": 7.0},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 40.0},
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is False
    assert any("d_seg passed but d_pose did NOT" in r for r in verdict.reasons)


# --- structural refusal of the d_seg-only gate -------------------------------

def test_d_seg_only_trajectory_dict_is_refused():
    """A trajectory dict with no d_pose/mean_d_pose key is structurally refused."""
    baseline = [
        {"epoch": 0, "exact_d_seg": 0.030},  # NO pose
        {"epoch": 20, "exact_d_seg": 0.004},
    ]
    candidate = _clean_descent_traj()
    with pytest.raises(DSegOnlyGateMisuse):
        evaluate_descent_equivalence(baseline, candidate, n_pairs=600)


def test_d_seg_only_candidate_trajectory_is_refused():
    """Refusal applies to the candidate arm too, not only the baseline."""
    baseline = _clean_descent_traj()
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030},
        {"epoch": 20, "exact_d_seg": 0.004},
    ]
    with pytest.raises(DSegOnlyGateMisuse):
        evaluate_descent_equivalence(baseline, candidate, n_pairs=600)


def test_epoch_metric_rejects_none_pose():
    """The typed record refuses a None pose at construction (no silent 0)."""
    with pytest.raises(DSegOnlyGateMisuse):
        EpochMetric(epoch=0, d_seg=0.03, d_pose=None)  # type: ignore[arg-type]


# --- the happy path: a genuinely descent-equivalent speedup ------------------

def test_clean_descent_equivalent_candidate_passes_at_n600():
    """Both terms track the authority within tolerance at the real n => PASS."""
    baseline = _clean_descent_traj()
    # Candidate is a faithful fast backend: tiny per-epoch drift, both descend.
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.0171, "mean_d_pose": 0.050001},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 3.1e-5},
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is True
    assert verdict.generalization_warning is False
    assert verdict.seg.tracks_within_tol is True
    assert verdict.pose.tracks_within_tol is True


def test_typed_epoch_metric_records_also_accepted():
    """The gate accepts EpochMetric records (not only dicts)."""
    baseline = [
        EpochMetric(0, 0.030, 0.10),
        EpochMetric(20, 0.0040, 3.0e-5),
    ]
    candidate = [
        EpochMetric(0, 0.030, 0.10),
        EpochMetric(20, 0.0041, 3.1e-5),
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is True


# --- the n8-does-not-generalize lesson ---------------------------------------

def test_small_n_pass_is_provisional():
    """A PASS at n8 sets generalization_warning (n8 does NOT generalize to n600)."""
    baseline = _clean_descent_traj()
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 3.1e-5},
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=8)
    assert verdict.passed is True  # it tracks at n8...
    assert verdict.generalization_warning is True  # ...but provisionally
    assert any("does NOT generalize" in r or "PROVISIONAL" in r for r in verdict.reasons)
    assert DEFAULT_MIN_TRUSTWORTHY_N == 600


def test_n600_pass_not_flagged_provisional():
    baseline = _clean_descent_traj()
    candidate = _clean_descent_traj(seg1=0.0041, pose1=3.1e-5)
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is True
    assert verdict.generalization_warning is False


# --- seg gap (not divergence) also rejects -----------------------------------

def test_seg_gap_too_large_rejects():
    """A candidate whose d_seg never descends (big final gap) is rejected."""
    baseline = _clean_descent_traj()
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 20, "exact_d_seg": 0.025, "mean_d_pose": 3.1e-5},  # seg barely moved
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is False
    assert verdict.seg.tracks_within_tol is False
    assert "GAP too large" in verdict.seg.reason


def test_pose_gap_without_divergence_rejects():
    """A bounded but too-large pose gap (no blow-up) still rejects on tolerance."""
    baseline = _clean_descent_traj(pose0=0.10, pose1=3.0e-5)
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.017, "mean_d_pose": 0.060},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 0.050},  # stuck high, not exploding
    ]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is False
    assert verdict.pose.diverged is False  # not a blow-up...
    assert verdict.pose.tracks_within_tol is False  # ...but a real gap


# --- alignment / input hygiene ------------------------------------------------

def test_non_overlapping_epochs_raises():
    baseline = [{"epoch": 0, "exact_d_seg": 0.03, "mean_d_pose": 0.1},
                {"epoch": 10, "exact_d_seg": 0.01, "mean_d_pose": 0.01}]
    candidate = [{"epoch": 5, "exact_d_seg": 0.03, "mean_d_pose": 0.1},
                 {"epoch": 15, "exact_d_seg": 0.01, "mean_d_pose": 0.01}]
    with pytest.raises(ValueError):
        evaluate_descent_equivalence(baseline, candidate, n_pairs=600)


def test_empty_trajectory_raises():
    with pytest.raises(ValueError):
        evaluate_descent_equivalence([], _clean_descent_traj(), n_pairs=600)


def test_canonical_field_names_accepted():
    """Accepts the canonical d_seg/d_pose keys as well as exact_d_seg/mean_d_pose."""
    baseline = [{"epoch": 0, "d_seg": 0.03, "d_pose": 0.1},
                {"epoch": 20, "d_seg": 0.004, "d_pose": 3e-5}]
    candidate = [{"epoch": 0, "d_seg": 0.03, "d_pose": 0.1},
                 {"epoch": 20, "d_seg": 0.0041, "d_pose": 3.1e-5}]
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert verdict.passed is True


# --- the gradient-cosine BOTH-paths pre-check --------------------------------

def test_cosine_precheck_both_paths_must_pass():
    """Both seg-path AND pose-path cosines must clear the bar."""
    ok = gradient_cosine_precheck_verdict(0.99999, 0.99998)
    assert ok.passed is True
    assert ok.seg_ok and ok.pose_ok


def test_cosine_precheck_seg_ok_pose_bad_is_rejected():
    """The native strided-grouped VJP signature: seg cosine high, pose cosine ~0."""
    v = gradient_cosine_precheck_verdict(0.99999, 0.025)
    assert v.passed is False
    assert v.seg_ok is True
    assert v.pose_ok is False


def test_cosine_precheck_is_necessary_not_sufficient_doc():
    """A cosine PASS does not admit the speedup — the gate docstring is explicit.

    This guards the documented contract: the custom backward had per-layer cosine
    1.0 and still diverged at n600. We assert the pre-check object is SEPARATE
    from the gate verdict (it cannot stand in for it).
    """
    from tac.mlx_pr95_port import speedup_acceptance_gate as mod

    # The "necessary, not sufficient" contract is documented on the pre-check type.
    assert "necessary" in mod.GradientCosinePrecheck.__doc__.lower()
    assert "does not admit" in mod.gradient_cosine_precheck_verdict.__doc__.lower()
    # The pre-check type has no gate-pass attribute — it can't stand in for the gate.
    v = gradient_cosine_precheck_verdict(1.0, 1.0)
    assert not hasattr(v, "passed_gate")


def test_custom_threshold_config_tightens_pose():
    """A tighter pose tolerance can reject a candidate the default would pass."""
    baseline = _clean_descent_traj(pose0=0.10, pose1=3.0e-5)
    candidate = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 8.0e-4},  # within default abs_tol 1e-3
    ]
    default = evaluate_descent_equivalence(baseline, candidate, n_pairs=600)
    assert default.pose.tracks_within_tol is True
    strict = evaluate_descent_equivalence(
        baseline, candidate, n_pairs=600,
        config=GateConfig(pose_abs_tol=1.0e-4, pose_rel_tol=0.005),
    )
    assert strict.pose.tracks_within_tol is False
    assert strict.passed is False
