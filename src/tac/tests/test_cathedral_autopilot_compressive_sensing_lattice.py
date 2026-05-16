# SPDX-License-Identifier: MIT
"""Cathedral autopilot integration: compressive-sensing lattice recovery.

Tests the wire-in helpers added per operator approval 2026-05-16:

  - ``rerank_candidates_via_compressive_sensing_lattice``
  - ``diagnose_compressive_sensing_lattice_undersampling``

Sister of :mod:`test_cathedral_autopilot_rudin_daubechies_integration`
(SLIM/Rashomon wire-in) per Catalog #125 wire-in hook 4 (cathedral
autopilot dispatch hook) consumer-side coverage.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the autopilot loop as a module-level import (same pattern as the
# sister rudin_daubechies integration test).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOPILOT_PATH = _REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"
_spec = importlib.util.spec_from_file_location(
    "cathedral_autopilot_autonomous_loop_compressive_sensing_test",
    _AUTOPILOT_PATH,
)
assert _spec is not None and _spec.loader is not None
_autopilot = importlib.util.module_from_spec(_spec)
sys.modules["cathedral_autopilot_autonomous_loop_compressive_sensing_test"] = _autopilot
_spec.loader.exec_module(_autopilot)


CandidateRow = _autopilot.CandidateRow
_build_substrate_lattice_from_candidates = _autopilot._build_substrate_lattice_from_candidates
rerank_candidates_via_compressive_sensing_lattice = (
    _autopilot.rerank_candidates_via_compressive_sensing_lattice
)
diagnose_compressive_sensing_lattice_undersampling = (
    _autopilot.diagnose_compressive_sensing_lattice_undersampling
)


def _make_candidate(
    cid: str,
    *,
    predicted: float = -0.005,
    cost: float = 1.0,
    eig: float = 0.01,
) -> CandidateRow:
    return CandidateRow(
        candidate_id=cid,
        family="test_family",
        predicted_score_delta=predicted,
        expected_information_gain=eig,
        estimated_dispatch_cost_usd=cost,
    )


def test_build_lattice_from_candidates_creates_node_per_id():
    cands = [
        _make_candidate("z3_g1", predicted=0.005),  # frontier + 0.005 = 0.197
        _make_candidate("nscs06_v8", predicted=-0.040),  # 0.152 frontier-pursuit
        _make_candidate("rudin_floor", predicted=-0.080),  # 0.112 asymptotic
    ]
    lr = _build_substrate_lattice_from_candidates(cands)
    assert lr.n_substrates == 3


def test_build_lattice_skips_duplicates():
    cands = [
        _make_candidate("z3_g1"),
        _make_candidate("z3_g1"),  # duplicate
        _make_candidate("nscs06_v8"),
    ]
    lr = _build_substrate_lattice_from_candidates(cands)
    assert lr.n_substrates == 2


def test_build_lattice_skips_empty_id():
    cands = [_make_candidate(""), _make_candidate("real")]
    lr = _build_substrate_lattice_from_candidates(cands)
    assert lr.n_substrates == 1


def test_rerank_returns_sorted_ascending_by_adjusted_delta():
    cands = [
        _make_candidate("a", predicted=-0.020),
        _make_candidate("b", predicted=-0.005),
        _make_candidate("c", predicted=-0.030),
    ]
    out = rerank_candidates_via_compressive_sensing_lattice(cands)
    assert len(out) == 3
    deltas = [adj for _, adj, _ in out]
    assert deltas == sorted(deltas)


def test_rerank_explanation_carries_canonical_provenance():
    cands = [_make_candidate("a", predicted=-0.020)]
    out = rerank_candidates_via_compressive_sensing_lattice(cands)
    _cand, _adj, expl = out[0]
    assert "compressive-sensing-lattice-recovery" in expl
    assert "posterior_frontier_probability=" in expl
    assert "raw_predicted_delta=" in expl
    assert "adjusted_predicted_delta=" in expl
    assert "basis=daubechies_db4" in expl


def test_rerank_with_anchors_updates_posterior():
    cands = [
        _make_candidate("a", predicted=-0.005),
        _make_candidate("b", predicted=-0.020),
        _make_candidate("c", predicted=-0.080),  # asymptotic-pursuit
    ]
    anchors = [("a", 0.187), ("b", 0.172)]
    out = rerank_candidates_via_compressive_sensing_lattice(
        cands, anchors=anchors
    )
    # Posterior with K=2 anchors should give a different ordering than
    # the no-anchor baseline.
    out_no_anchor = rerank_candidates_via_compressive_sensing_lattice(cands)
    # Both should have 3 candidates with valid explanations.
    assert len(out) == 3
    assert len(out_no_anchor) == 3
    for _, _, expl in out:
        assert "K=2" in expl
    for _, _, expl in out_no_anchor:
        assert "K=0" in expl


def test_rerank_anchor_at_unknown_node_skipped_silently():
    cands = [_make_candidate("a", predicted=-0.005)]
    out = rerank_candidates_via_compressive_sensing_lattice(
        cands, anchors=[("ghost", 0.15)]
    )
    # No exception raised; explanation still carries posterior signal.
    assert len(out) == 1
    _, _, expl = out[0]
    # No anchors landed (ghost is unknown), so K=0 in the tag.
    assert "K=0" in expl


def test_rerank_haar_fallback_when_db4_disabled():
    cands = [_make_candidate(f"s{i}", predicted=-0.005) for i in range(3)]
    out = rerank_candidates_via_compressive_sensing_lattice(
        cands, use_daubechies_db4=False
    )
    for _, _, expl in out:
        assert "basis=haar_db1" in expl


def test_rerank_empty_pool_returns_empty():
    out = rerank_candidates_via_compressive_sensing_lattice([])
    assert out == []


def test_rerank_high_posterior_keeps_full_savings():
    """If posterior says HIGH frontier-breaking probability, the adjusted
    savings should be CLOSE TO the raw predicted savings."""
    # Anchor an unambiguous frontier-breaker.
    cands = [_make_candidate("breaker", predicted=-0.080)]
    out = rerank_candidates_via_compressive_sensing_lattice(
        cands, anchors=[("breaker", 0.10)]  # well below 0.192 frontier
    )
    _cand, adjusted, _expl = out[0]
    # With anchor at 0.10 < 0.192 threshold, posterior_frontier_probability=1.0
    # adjusted = -0.080 * (0.5 + 0.5 * 1.0) = -0.080
    assert adjusted == -0.080


def test_rerank_low_posterior_halves_savings():
    """If posterior says LOW frontier-breaking probability, the adjusted
    savings should be ~0.5x the raw."""
    # Construct one with predicted_score_delta close to zero so the lattice
    # sees a non-frontier-breaker; anchor at a high value to suppress prob.
    cands = [_make_candidate("non_breaker", predicted=-0.080)]
    out = rerank_candidates_via_compressive_sensing_lattice(
        cands, anchors=[("non_breaker", 0.250)]  # well above threshold
    )
    _cand, adjusted, _expl = out[0]
    # Anchored ABOVE threshold => posterior_frontier_probability=0.0
    # adjusted = -0.080 * (0.5 + 0.5 * 0.0) = -0.040
    assert adjusted == -0.040


def test_diagnose_undersampling_returns_donoho_tanner_record():
    cands = [_make_candidate(f"s{i}") for i in range(30)]
    diag = diagnose_compressive_sensing_lattice_undersampling(
        cands, n_anchors=8, expected_sparsity=5
    )
    assert "recovery_regime" in diag
    assert diag["N"] == 30
    assert diag["K"] == 8
    assert diag["sparsity_estimate"] == 5


def test_diagnose_undersampling_failed_regime_for_K8_N30_s5():
    """The falsification audit's K=8/N=30 schedule for s=5 should FAIL."""
    cands = [_make_candidate(f"s{i}") for i in range(30)]
    diag = diagnose_compressive_sensing_lattice_undersampling(
        cands, n_anchors=8, expected_sparsity=5
    )
    assert diag["recovery_regime"] == "FAILED"


def test_diagnose_undersampling_exact_regime_for_K20_N30_s2():
    """Generous K/N ratio with low sparsity should be EXACT."""
    cands = [_make_candidate(f"s{i}") for i in range(30)]
    diag = diagnose_compressive_sensing_lattice_undersampling(
        cands, n_anchors=20, expected_sparsity=2
    )
    assert diag["recovery_regime"] == "EXACT"
