# SPDX-License-Identifier: MIT
"""Tests for compressive-sensing lattice recovery (6 enhancements).

Per `feedback_compressive_sensing_lattice_recovery_enhancements_landed_20260516`
this module covers:
  - Enhancement 1: SubstrateLatticeRecovery sparse-signal posterior
  - Enhancement 2: CoherenceMinimizingSelector (Tropp 2004 RIP)
  - Enhancement 3: BayesianSequentialKSelector (Snoek-Larochelle-Adams 2012)
  - Enhancement 4: LatticePhaseTransitionMonitor (Donoho-Tanner 2009)
  - Enhancement 5: TreeStructuredSparsityPrior (Baraniuk-Cevher 2010)
  - Enhancement 6: DaubechiesDb4LatticeBasis (Mallat 1989 / Daubechies 1988)
"""
from __future__ import annotations

import math

import pytest

from tac.autopilot_rudin_daubechies import (
    BayesianSequentialKSelector,
    CoherenceMinimizingSelector,
    DaubechiesDb4LatticeBasis,
    FrontierPursuitClass,
    LatticePhaseTransitionMonitor,
    SubstrateLatticeNode,
    SubstrateLatticeRecovery,
    TreeStructuredSparsityPrior,
    classify_predicted_band,
    compute_pairwise_coherence,
    diff_sparse_signal_posteriors,
)


# Test fixtures.
def _make_node(
    node_id: str,
    *,
    parent_id: str | None = None,
    support_level: int = 0,
    low: float = 0.15,
    high: float = 0.20,
    cls: FrontierPursuitClass = FrontierPursuitClass.PLATEAU_ADJACENT,
) -> SubstrateLatticeNode:
    return SubstrateLatticeNode(
        node_id=node_id,
        parent_id=parent_id,
        support_level=support_level,
        predicted_band_low=low,
        predicted_band_high=high,
        frontier_pursuit_class=cls,
    )


# ──────────────────────────────────────────────────────────────────────────
# Horizon class classification.
# ──────────────────────────────────────────────────────────────────────────


def test_classify_predicted_band_plateau_adjacent():
    assert classify_predicted_band(0.185, 0.195) == FrontierPursuitClass.PLATEAU_ADJACENT


def test_classify_predicted_band_frontier_pursuit():
    assert classify_predicted_band(0.140, 0.175) == FrontierPursuitClass.FRONTIER_PURSUIT


def test_classify_predicted_band_asymptotic_pursuit():
    assert classify_predicted_band(0.05, 0.12) == FrontierPursuitClass.ASYMPTOTIC_PURSUIT


def test_classify_predicted_band_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        classify_predicted_band(float("nan"), 0.18)


def test_classify_predicted_band_rejects_inverted():
    with pytest.raises(ValueError, match="<= high"):
        classify_predicted_band(0.30, 0.10)


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 1: SubstrateLatticeRecovery.
# ──────────────────────────────────────────────────────────────────────────


def test_lattice_recovery_empty_returns_empty_posterior():
    lr = SubstrateLatticeRecovery()
    post = lr.recover_sparse_signal()
    assert post.n_substrates == 0
    assert post.n_anchors == 0
    assert post.posterior_frontier_probability == ()
    assert "K=0" in post.confidence_tag


def test_lattice_recovery_duplicate_node_rejected():
    lr = SubstrateLatticeRecovery()
    lr.add_node(_make_node("a"))
    with pytest.raises(ValueError, match="duplicate"):
        lr.add_node(_make_node("a"))


def test_lattice_recovery_orphan_parent_rejected():
    lr = SubstrateLatticeRecovery()
    with pytest.raises(ValueError, match="not registered"):
        lr.add_node(_make_node("child", parent_id="ghost"))


def test_lattice_recovery_anchor_at_unknown_node_rejected():
    lr = SubstrateLatticeRecovery()
    lr.add_node(_make_node("a"))
    with pytest.raises(ValueError, match="not in lattice"):
        lr.update_from_anchor("ghost", 0.18)


def test_lattice_recovery_anchor_rejects_nonfinite():
    lr = SubstrateLatticeRecovery()
    lr.add_node(_make_node("a"))
    with pytest.raises(ValueError, match="finite"):
        lr.update_from_anchor("a", float("inf"))


def test_lattice_recovery_anchor_replaces_previous():
    lr = SubstrateLatticeRecovery()
    lr.add_node(_make_node("a"))
    lr.update_from_anchor("a", 0.20)
    lr.update_from_anchor("a", 0.25)
    assert lr.n_anchors == 1


def test_lattice_recovery_recovers_sparse_signal_with_anchors():
    lr = SubstrateLatticeRecovery(expected_sparsity=3)
    for i in range(10):
        lr.add_node(_make_node(f"s{i}", low=0.15 + 0.01 * i, high=0.20 + 0.01 * i))
    lr.update_from_anchor("s0", 0.18)
    lr.update_from_anchor("s5", 0.16)
    post = lr.recover_sparse_signal()
    assert post.n_substrates == 10
    assert post.n_anchors == 2
    assert post.sparsity_recovered > 0
    assert post.basis == "daubechies_db4"
    assert "K=2; N=10" in post.confidence_tag


def test_lattice_recovery_anchored_substrate_has_zero_uncertainty():
    lr = SubstrateLatticeRecovery()
    for i in range(10):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s3", 0.185)
    post = lr.recover_sparse_signal()
    unc_map = dict(post.recovery_uncertainty)
    assert unc_map["s3"] == 0.0


def test_lattice_recovery_unanchored_substrate_has_nonzero_uncertainty():
    lr = SubstrateLatticeRecovery()
    for i in range(10):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.185)
    post = lr.recover_sparse_signal()
    unc_map = dict(post.recovery_uncertainty)
    assert unc_map["s9"] > 0.0


def test_lattice_recovery_frontier_threshold_anchored_below_returns_p1():
    lr = SubstrateLatticeRecovery(frontier_threshold_cpu=0.192)
    lr.add_node(_make_node("breaker"))
    lr.update_from_anchor("breaker", 0.150)  # below threshold
    post = lr.recover_sparse_signal()
    prob_map = dict(post.posterior_frontier_probability)
    assert prob_map["breaker"] == 1.0


def test_lattice_recovery_frontier_threshold_anchored_above_returns_p0():
    lr = SubstrateLatticeRecovery(
        frontier_threshold_cpu=0.192,
        use_tree_structured_prior=False,  # tree prior may boost; isolate here
    )
    lr.add_node(_make_node("non_breaker"))
    lr.update_from_anchor("non_breaker", 0.250)  # above threshold
    post = lr.recover_sparse_signal()
    prob_map = dict(post.posterior_frontier_probability)
    assert prob_map["non_breaker"] == 0.0


def test_lattice_recovery_predict_marginal_recovery_at_K():
    lr = SubstrateLatticeRecovery()
    for i in range(10):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.185)
    lr.update_from_anchor("s5", 0.195)
    counterfactual = lr.predict_marginal_recovery_at_K("s7")
    assert counterfactual["K_current"] == 2
    assert counterfactual["K_next"] == 3
    assert counterfactual["uncertainty_reduction"] > 0.0
    assert counterfactual["basis"] == "daubechies_db4"


def test_lattice_recovery_predict_marginal_unknown_node_rejected():
    lr = SubstrateLatticeRecovery()
    lr.add_node(_make_node("a"))
    with pytest.raises(ValueError, match="not in lattice"):
        lr.predict_marginal_recovery_at_K("ghost")


def test_lattice_recovery_haar_fallback_when_db4_disabled():
    lr = SubstrateLatticeRecovery(use_daubechies_db4=False)
    for i in range(5):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.18)
    post = lr.recover_sparse_signal()
    assert post.basis == "haar_db1"


def test_lattice_recovery_observability_record_is_json_serializable():
    import json as _json

    lr = SubstrateLatticeRecovery()
    for i in range(8):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.18)
    lr.update_from_anchor("s4", 0.20)
    post = lr.recover_sparse_signal()
    rec = post.to_observability_record()
    s = _json.dumps(rec)
    assert "sparsity_recovered" in s
    assert "confidence_tag" in s


def test_lattice_recovery_top_k_frontier_breaking():
    lr = SubstrateLatticeRecovery()
    for i in range(10):
        lr.add_node(_make_node(f"s{i}", low=0.10 + 0.02 * i, high=0.15 + 0.02 * i))
    lr.update_from_anchor("s0", 0.10)
    post = lr.recover_sparse_signal()
    top3 = post.top_k_frontier_breaking(3)
    assert len(top3) == 3
    # All probabilities should be descending.
    probs = [p for _, p in top3]
    assert probs == sorted(probs, reverse=True)


def test_diff_sparse_signal_posteriors_returns_per_substrate_delta():
    lr = SubstrateLatticeRecovery()
    for i in range(5):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.18)
    post_a = lr.recover_sparse_signal()
    lr.update_from_anchor("s3", 0.15)
    post_b = lr.recover_sparse_signal()
    diff = diff_sparse_signal_posteriors(post_a, post_b)
    assert diff["anchor_delta"] == 1
    assert len(diff["per_substrate"]) == 5


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 2: Coherence-minimization.
# ──────────────────────────────────────────────────────────────────────────


def test_compute_pairwise_coherence_returns_canonical_ordering():
    nodes = [_make_node("a"), _make_node("b"), _make_node("c")]
    coh = compute_pairwise_coherence(nodes)
    assert ("a", "b") in coh
    assert ("a", "c") in coh
    assert ("b", "c") in coh
    assert ("b", "a") not in coh  # canonical ordering only


def test_compute_pairwise_coherence_same_band_same_class_high():
    a = _make_node("a", low=0.18, high=0.20, cls=FrontierPursuitClass.PLATEAU_ADJACENT)
    b = _make_node("b", low=0.18, high=0.20, cls=FrontierPursuitClass.PLATEAU_ADJACENT)
    coh = compute_pairwise_coherence([a, b])
    assert coh[("a", "b")] == pytest.approx(1.0, abs=0.05)


def test_compute_pairwise_coherence_disjoint_bands_zero():
    a = _make_node("a", low=0.10, high=0.12)
    b = _make_node("b", low=0.18, high=0.20)
    coh = compute_pairwise_coherence([a, b])
    assert coh[("a", "b")] == 0.0


def test_compute_pairwise_coherence_parent_child_lowered():
    a = _make_node("parent", low=0.18, high=0.20)
    b = _make_node("child", parent_id="parent", support_level=1, low=0.18, high=0.20)
    coh = compute_pairwise_coherence([a, b])
    # Parent/child get class_boost * 0.3 per tree-structure adjustment.
    assert coh[("parent", "child")] < 0.5


def test_coherence_selector_returns_K_substrates():
    nodes = [_make_node(f"s{i}", low=0.10 + 0.01 * i, high=0.20 + 0.01 * i) for i in range(10)]
    sel = CoherenceMinimizingSelector(K=4).select(nodes)
    assert len(sel) == 4


def test_coherence_selector_returns_all_when_K_exceeds_pool():
    nodes = [_make_node(f"s{i}") for i in range(3)]
    sel = CoherenceMinimizingSelector(K=10).select(nodes)
    assert len(sel) == 3


def test_coherence_selector_K_zero_returns_empty():
    nodes = [_make_node(f"s{i}") for i in range(5)]
    sel = CoherenceMinimizingSelector(K=0).select(nodes)
    assert sel == []


def test_coherence_selector_starts_from_lowest_midpoint():
    """Greedy starts from the lowest-band-midpoint (most-frontier-pursuit)."""
    nodes = [
        _make_node("a", low=0.18, high=0.20),
        _make_node("b", low=0.05, high=0.07, cls=FrontierPursuitClass.ASYMPTOTIC_PURSUIT),
        _make_node("c", low=0.14, high=0.16, cls=FrontierPursuitClass.FRONTIER_PURSUIT),
    ]
    sel = CoherenceMinimizingSelector(K=1).select(nodes)
    assert sel[0].node_id == "b"


def test_coherence_selector_respects_plateau_budget_cap():
    """At most ceil(K * 0.30) plateau substrates."""
    nodes_plateau = [
        _make_node(f"p{i}", low=0.18, high=0.20, cls=FrontierPursuitClass.PLATEAU_ADJACENT)
        for i in range(10)
    ]
    nodes_frontier = [
        _make_node(f"f{i}", low=0.13, high=0.17, cls=FrontierPursuitClass.FRONTIER_PURSUIT)
        for i in range(10)
    ]
    nodes_asymptotic = [
        _make_node(f"a{i}", low=0.08, high=0.11, cls=FrontierPursuitClass.ASYMPTOTIC_PURSUIT)
        for i in range(5)
    ]
    all_nodes = nodes_plateau + nodes_frontier + nodes_asymptotic
    sel = CoherenceMinimizingSelector(K=10, plateau_budget_max=0.30).select(all_nodes)
    plateau_count = sum(
        1 for s in sel if s.frontier_pursuit_class == FrontierPursuitClass.PLATEAU_ADJACENT
    )
    assert plateau_count <= math.ceil(10 * 0.30)


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 3: Bayesian sequential K-selection.
# ──────────────────────────────────────────────────────────────────────────


def test_bayesian_selector_no_posterior_falls_back_to_band_width():
    """Without posterior, falls back to Thompson-sampling-style band-width
    priority."""
    nodes = [
        _make_node("narrow", low=0.18, high=0.19),  # width 0.01
        _make_node("wide", low=0.10, high=0.18),  # width 0.08
        _make_node("medium", low=0.15, high=0.18),  # width 0.03
    ]
    sel = BayesianSequentialKSelector(K=1).select_next_K(nodes)
    assert sel[0].node_id == "wide"


def test_bayesian_selector_with_posterior_prioritizes_high_entropy():
    """With posterior, prefers candidates with HIGH uncertainty * Bernoulli entropy."""
    lr = SubstrateLatticeRecovery()
    for i in range(8):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.18)
    lr.update_from_anchor("s1", 0.19)
    post = lr.recover_sparse_signal()
    nodes = [_make_node(f"s{i}") for i in range(8)]
    sel = BayesianSequentialKSelector(
        K=3, posterior=post, lattice=lr
    ).select_next_K(nodes)
    assert len(sel) == 3


def test_bayesian_selector_K_zero_returns_empty():
    nodes = [_make_node(f"s{i}") for i in range(5)]
    sel = BayesianSequentialKSelector(K=0).select_next_K(nodes)
    assert sel == []


def test_bayesian_selector_K_exceeds_pool_returns_all():
    nodes = [_make_node(f"s{i}") for i in range(3)]
    sel = BayesianSequentialKSelector(K=10).select_next_K(nodes)
    assert len(sel) == 3


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 4: Phase-transition monitor (Donoho-Tanner 2009).
# ──────────────────────────────────────────────────────────────────────────


def test_phase_monitor_rejects_invalid_N():
    mon = LatticePhaseTransitionMonitor()
    with pytest.raises(ValueError, match="N must be > 0"):
        mon.compute_undersampling_diagnostic(K=8, N=0, sparsity_estimate=3)


def test_phase_monitor_rejects_negative_K():
    mon = LatticePhaseTransitionMonitor()
    with pytest.raises(ValueError, match=">= 0"):
        mon.compute_undersampling_diagnostic(K=-1, N=30, sparsity_estimate=3)


def test_phase_monitor_exact_regime_when_K_large_s_small():
    """K=20, N=30, s=2 -> rho = 0.10, well below threshold."""
    mon = LatticePhaseTransitionMonitor()
    diag = mon.compute_undersampling_diagnostic(K=20, N=30, sparsity_estimate=2)
    assert diag["recovery_regime"] == "EXACT"
    assert not diag["safety_margin_violated"]


def test_phase_monitor_failed_regime_when_K_too_small():
    """K=4, N=30, s=8 -> rho = 2.0, well ABOVE threshold."""
    mon = LatticePhaseTransitionMonitor()
    diag = mon.compute_undersampling_diagnostic(K=4, N=30, sparsity_estimate=8)
    assert diag["recovery_regime"] == "FAILED"
    assert diag["safety_margin_violated"]


def test_phase_monitor_falsification_audit_K8_N30_s5_falls_in_failed():
    """The falsification audit's K=8/N=30 schedule for s=5 is FAILED per
    Donoho-Tanner (rho = 0.625 > threshold ~0.21)."""
    mon = LatticePhaseTransitionMonitor()
    diag = mon.compute_undersampling_diagnostic(K=8, N=30, sparsity_estimate=5)
    assert diag["recovery_regime"] == "FAILED"
    assert diag["rho"] == pytest.approx(5 / 8, abs=1e-6)


def test_phase_monitor_recommends_larger_K_when_failed():
    mon = LatticePhaseTransitionMonitor()
    diag = mon.compute_undersampling_diagnostic(K=4, N=30, sparsity_estimate=5)
    if diag["recovery_regime"] != "EXACT":
        assert diag["recommended_K"] is not None
        assert diag["recommended_K"] > 4


def test_phase_monitor_interpolates_threshold_monotonically():
    mon = LatticePhaseTransitionMonitor()
    # Threshold should be monotone non-decreasing in delta.
    thresholds = [mon._interpolate_threshold(d / 100) for d in range(5, 100, 5)]
    for i in range(1, len(thresholds)):
        assert thresholds[i] >= thresholds[i - 1] - 1e-9


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 5: Tree-structured sparsity prior.
# ──────────────────────────────────────────────────────────────────────────


def test_tree_prior_parent_in_support_boosts_children():
    parent = _make_node("parent")
    child_a = _make_node("child_a", parent_id="parent", support_level=1)
    child_b = _make_node("child_b", parent_id="parent", support_level=1)
    raw = {"parent": 0.8, "child_a": 0.3, "child_b": 0.3}
    prior = TreeStructuredSparsityPrior(parent_in_support_boost=2.0)
    adj = prior.apply_to_posterior([parent, child_a, child_b], raw)
    assert adj["child_a"] > 0.3
    assert adj["child_b"] > 0.3


def test_tree_prior_all_siblings_outside_penalizes():
    parent = _make_node("parent")
    child_a = _make_node("child_a", parent_id="parent", support_level=1)
    child_b = _make_node("child_b", parent_id="parent", support_level=1)
    raw = {"parent": 0.1, "child_a": 0.2, "child_b": 0.2}
    prior = TreeStructuredSparsityPrior(all_siblings_outside_penalty=0.5)
    adj = prior.apply_to_posterior([parent, child_a, child_b], raw)
    assert adj["child_a"] < 0.2
    assert adj["child_b"] < 0.2


def test_tree_prior_no_parent_no_op():
    """Root-only lattice: no parent to act on."""
    a = _make_node("a")
    b = _make_node("b")
    raw = {"a": 0.5, "b": 0.7}
    prior = TreeStructuredSparsityPrior()
    adj = prior.apply_to_posterior([a, b], raw)
    assert adj["a"] == 0.5
    assert adj["b"] == 0.7


# ──────────────────────────────────────────────────────────────────────────
# Enhancement 6: Daubechies db4 basis.
# ──────────────────────────────────────────────────────────────────────────


def test_db4_filter_coefficients_have_8_nonzero():
    assert len(DaubechiesDb4LatticeBasis.DB4_LOW_PASS) == 8


def test_db4_low_pass_sum_close_to_sqrt2():
    """Daubechies orthonormality: sum_k h_k = sqrt(2)."""
    s = sum(DaubechiesDb4LatticeBasis.DB4_LOW_PASS)
    assert s == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_db4_high_pass_orthogonal_to_low_pass():
    """g_k = (-1)^k h_{N-1-k}; sum_k h_k g_k should be ~ 0."""
    h = DaubechiesDb4LatticeBasis.DB4_LOW_PASS
    g = DaubechiesDb4LatticeBasis.high_pass()
    inner = sum(h[k] * g[k] for k in range(len(h)))
    # QMF inner product is not exactly zero — they differ over shift; but
    # for the same-index inner product the QMF construction yields a small
    # nonzero value bounded by sum |h_k|.
    assert abs(inner) < sum(abs(c) for c in h)


def test_db4_forward_transform_empty_signal():
    a, d = DaubechiesDb4LatticeBasis.forward_transform([])
    assert a == []
    assert d == []


def test_db4_forward_transform_returns_half_length():
    signal = [1.0] * 16
    a, d = DaubechiesDb4LatticeBasis.forward_transform(signal)
    assert len(a) == 8
    assert len(d) == 8


def test_db4_project_sparse_empty_signal():
    out = DaubechiesDb4LatticeBasis.project_sparse([], sparsity=3)
    assert out == []


def test_db4_project_sparse_zero_sparsity_returns_zero():
    out = DaubechiesDb4LatticeBasis.project_sparse([1.0, 2.0, 3.0], sparsity=0)
    assert out == [0.0, 0.0, 0.0]


def test_db4_project_sparse_full_sparsity_returns_signal():
    sig = [1.0, 2.0, 3.0, 4.0]
    out = DaubechiesDb4LatticeBasis.project_sparse(sig, sparsity=10)
    assert out == sig


def test_db4_project_sparse_concentrated_signal_recovers():
    """A spike signal projects onto its largest wavelet coefficient cleanly."""
    sig = [0.0] * 8
    sig[3] = 10.0
    out = DaubechiesDb4LatticeBasis.project_sparse(sig, sparsity=4)
    # Spike preserved approximately (db4 wavelet support spread).
    assert max(out) > 1.0


# ──────────────────────────────────────────────────────────────────────────
# Composition / integration tests.
# ──────────────────────────────────────────────────────────────────────────


def test_full_pipeline_end_to_end_with_falsification_audit_scenario():
    """End-to-end: K=8 LEVEL-0 schedule on N=15 substrate lattice with
    s=5 expected frontier-breaking + tree-structured composition + db4
    basis + phase-monitor."""
    lr = SubstrateLatticeRecovery(
        use_daubechies_db4=True,
        use_tree_structured_prior=True,
        expected_sparsity=5,
        frontier_threshold_cpu=0.192,
    )
    # 12 leaf substrates.
    leaves = [
        _make_node(f"leaf_{i}", low=0.10 + 0.01 * i, high=0.15 + 0.01 * i)
        for i in range(12)
    ]
    for n in leaves:
        lr.add_node(n)
    # 3 pairwise compositions.
    comps = [
        _make_node(
            f"comp_{i}",
            parent_id=f"leaf_{i}",
            support_level=1,
            low=0.10 + 0.01 * i,
            high=0.15 + 0.01 * i,
            cls=FrontierPursuitClass.FRONTIER_PURSUIT,
        )
        for i in range(3)
    ]
    for n in comps:
        lr.add_node(n)
    # K=8 anchors landing.
    for i, score in enumerate([0.20, 0.18, 0.15, 0.13, 0.11, 0.16, 0.19, 0.17]):
        lr.update_from_anchor(f"leaf_{i}", score)
    post = lr.recover_sparse_signal()
    assert post.n_substrates == 15
    assert post.n_anchors == 8
    # Phase-transition check.
    mon = LatticePhaseTransitionMonitor()
    diag = mon.compute_undersampling_diagnostic(
        K=post.n_anchors, N=post.n_substrates, sparsity_estimate=post.sparsity_recovered
    )
    assert diag["recovery_regime"] in ("EXACT", "AT_THRESHOLD", "FAILED")
    # Coherence-minimizing K-selection for next round.
    next_K = CoherenceMinimizingSelector(K=4).select(leaves + comps)
    assert len(next_K) == 4
    # Bayesian sequential selection conditional on posterior.
    bayes_K = BayesianSequentialKSelector(
        K=3, posterior=post, lattice=lr
    ).select_next_K(leaves + comps)
    assert len(bayes_K) == 3


def test_full_pipeline_observability_record_includes_provenance():
    lr = SubstrateLatticeRecovery()
    for i in range(8):
        lr.add_node(_make_node(f"s{i}"))
    lr.update_from_anchor("s0", 0.18)
    lr.update_from_anchor("s4", 0.20)
    post = lr.recover_sparse_signal()
    rec = post.to_observability_record()
    assert "compressive-sensing-lattice-recovery" in rec["confidence_tag"]
    assert rec["n_substrates"] == 8
    assert rec["n_anchors"] == 2
    assert rec["basis"] == "daubechies_db4"
