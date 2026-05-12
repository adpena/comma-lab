"""W/I/A decisions implementation regression tests (2026-05-12).

Tests for `feedback_wia_decisions_impl_landed_20260512`:

- I-1: autonomous-loop ↔ continual-learning posterior wire-in
- I-3: sensitivity-map context surfacing on planner load
- I-4: iter_layer_pairs documented as research-only (smoke import)

Per CLAUDE.md "Subagent coherence-by-default" the regression test below
exercises wire-in hook 5 (continual-learning posterior) directly: a
newly-appended anchor changes the next ranking pass's output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cathedral_autopilot_autonomous_loop as loop  # noqa: E402


def _cand(
    cid: str, *, family: str = "hnerv_lc_v2",
    delta: float = -0.005, eig: float = 0.5, cost: float = 5.0,
) -> loop.CandidateRow:
    return loop.CandidateRow(
        candidate_id=cid,
        family=family,
        predicted_score_delta=delta,
        expected_information_gain=eig,
        estimated_dispatch_cost_usd=cost,
    )


# ── I-1: posterior-correction factor ──────────────────────────────────────


class _StubPosterior:
    """Minimal posterior stub for testing rank_candidates correction.

    Real `tac.continual_learning.posterior_query_track_correction` signature
    is (posterior, track_kind, *, default=1.0) -> (factor, n_obs).
    """

    def __init__(self, family_to_factor: dict[str, tuple[float, int]]):
        # Public attribute mirroring real ContinualLearningPosterior surface
        # used by load_planner_posterior_for_loop diagnostic.
        self.schema = "stub_posterior_v0"
        self.accepted_anchor_count = sum(n for _, n in family_to_factor.values())
        self.refused_anchor_count = 0
        self.track_correction_posteriors = {k: object() for k in family_to_factor}
        self._family = family_to_factor

    def query(self, kind: str) -> tuple[float, int]:
        return self._family.get(kind, (1.0, 0))


@pytest.fixture
def stub_query_track_correction(monkeypatch):
    """Monkeypatch the loop module's posterior_query_track_correction symbol.

    The loop module imported the symbol at module load; we replace it in-place
    so tests can drive the correction factor without a real posterior file.
    """
    def _go(posterior_to_families: dict):
        def _fake_query(post, kind, *, default=1.0):
            if hasattr(post, "_family"):
                return post.query(kind)
            return default, 0
        monkeypatch.setattr(loop, "posterior_query_track_correction", _fake_query)
    return _go


def test_i1_rank_candidates_no_posterior_unchanged():
    """Without a posterior, ranking is the legacy pure-EIG/$ order."""
    cands = [
        _cand("c1", eig=2.0, cost=10.0),   # 0.2
        _cand("c2", eig=1.0, cost=10.0),   # 0.1
        _cand("c3", eig=0.5, cost=10.0),   # 0.05
    ]
    ranked = loop.rank_candidates(cands, rank_axis="eig_per_dollar")
    assert [c.candidate_id for c in ranked] == ["c1", "c2", "c3"]


def test_i1_rank_candidates_posterior_reweights_eig(stub_query_track_correction):
    """Family-keyed posterior factor reweights EIG/$ sort.

    Family A has correction 2.0x; family B has correction 0.5x. The two
    candidates have equal raw EIG/$ but family A's effective EIG/$ is 4x.
    Family A must sort first.
    """
    cands = [
        _cand("c_b", family="family_b", eig=1.0, cost=10.0),  # raw 0.1
        _cand("c_a", family="family_a", eig=1.0, cost=10.0),  # raw 0.1
    ]
    posterior = _StubPosterior({
        "family_a": (2.0, 3),
        "family_b": (0.5, 3),
    })
    stub_query_track_correction(posterior)
    ranked = loop.rank_candidates(
        cands, rank_axis="eig_per_dollar", continual_posterior=posterior,
    )
    assert [c.candidate_id for c in ranked] == ["c_a", "c_b"]


def test_i1_newly_appended_anchor_changes_next_ranking(stub_query_track_correction):
    """The wire-in regression: an updated posterior changes ranking on the
    next call.

    This is the canonical "every autopilot RANK pass should consume the
    posterior" test from the W/I/A I-1 directive.
    """
    cands = [
        _cand("c_a", family="family_a", eig=1.0, cost=10.0),
        _cand("c_b", family="family_b", eig=1.0, cost=10.0),
    ]

    # Pass 1: no anchors for either family — order is stable.
    posterior_v0 = _StubPosterior({})
    stub_query_track_correction(posterior_v0)
    ranked_v0 = loop.rank_candidates(
        cands, rank_axis="eig_per_dollar", continual_posterior=posterior_v0,
    )
    # With equal factors, sorted() preserves input order.
    assert [c.candidate_id for c in ranked_v0] == ["c_a", "c_b"]

    # Pass 2: simulate an anchor harvest for family_b that biases the
    # corrected EIG/$ for family_b higher than family_a.
    posterior_v1 = _StubPosterior({"family_b": (3.0, 1)})
    stub_query_track_correction(posterior_v1)
    ranked_v1 = loop.rank_candidates(
        cands, rank_axis="eig_per_dollar", continual_posterior=posterior_v1,
    )
    # Family B's effective EIG/$ is now 3x family A's — must sort first.
    assert [c.candidate_id for c in ranked_v1] == ["c_b", "c_a"]

    # Pass 3: a different anchor harvest for family_a that boosts it 5x —
    # family_a must now sort first again. This is the "next ranking pass
    # consumes the latest posterior" property.
    posterior_v2 = _StubPosterior({
        "family_a": (5.0, 2),
        "family_b": (3.0, 1),
    })
    stub_query_track_correction(posterior_v2)
    ranked_v2 = loop.rank_candidates(
        cands, rank_axis="eig_per_dollar", continual_posterior=posterior_v2,
    )
    assert [c.candidate_id for c in ranked_v2] == ["c_a", "c_b"]


def test_i1_rank_predicted_score_delta_with_posterior(stub_query_track_correction):
    """Sister axis: predicted_score_delta ranking also honors posterior."""
    cands = [
        _cand("c_a", family="family_a", delta=-0.010),
        _cand("c_b", family="family_b", delta=-0.005),
    ]
    # Family A's predicted-Δ is 2× family B's (-0.010 vs -0.005); without
    # posterior, A wins (most-negative first). Apply a 0.2× correction to A
    # — the corrected delta is -0.002, which is LESS negative than B's
    # uncorrected -0.005. Family B wins.
    posterior = _StubPosterior({"family_a": (0.2, 4)})
    stub_query_track_correction(posterior)
    ranked = loop.rank_candidates(
        cands, rank_axis="predicted_score_delta",
        continual_posterior=posterior,
    )
    assert [c.candidate_id for c in ranked] == ["c_b", "c_a"]


def test_i1_correction_factor_helper_zero_anchors():
    """_posterior_correction_factor returns (1.0, 0, "") for empty posterior."""
    c = _cand("c1", family="unknown_family")
    factor, n, key = loop._posterior_correction_factor(c, None)
    assert factor == 1.0
    assert n == 0
    assert key == ""


def test_i1_correction_factor_helper_with_match(stub_query_track_correction):
    posterior = _StubPosterior({"hnerv_lc_v2": (1.5, 7)})
    stub_query_track_correction(posterior)
    c = _cand("c1", family="hnerv_lc_v2")
    factor, n, key = loop._posterior_correction_factor(c, posterior)
    assert factor == pytest.approx(1.5)
    assert n == 7
    assert key == "hnerv_lc_v2"


def test_i1_correction_factor_negative_or_zero_rejected(stub_query_track_correction):
    """Defensive: negative / zero correction factors fall back to (1.0, 0).

    A bad anchor should not flip predicted_score_delta signs.
    """
    posterior = _StubPosterior({"bad_family": (-2.5, 5)})
    stub_query_track_correction(posterior)
    c = _cand("c1", family="bad_family")
    factor, n, key = loop._posterior_correction_factor(c, posterior)
    assert factor == 1.0
    assert n == 0


def test_i1_run_one_loop_iteration_passes_posterior(stub_query_track_correction):
    """End-to-end: run_one_loop_iteration honors the injected posterior."""
    cands = [
        _cand("c_a", family="family_a", eig=1.0, cost=10.0),
        _cand("c_b", family="family_b", eig=1.0, cost=10.0),
    ]
    posterior = _StubPosterior({"family_b": (4.0, 3)})
    stub_query_track_correction(posterior)
    report = loop.run_one_loop_iteration(
        cands,
        iteration=1,
        continual_posterior=posterior,
    )
    assert report.n_candidates_seen == 2
    # The first halt event should be the family_b candidate.
    assert report.halt_events[0].candidate_id == "c_b"


def test_i1_load_planner_posterior_returns_context_dict():
    """load_planner_posterior_for_loop emits a context dict (no crashes)."""
    posterior, context = loop.load_planner_posterior_for_loop(
        continual_posterior_path=Path("/nonexistent/path/does/not/exist.jsonl"),
    )
    # When path does not exist, the actual tac.continual_learning may
    # gracefully return an empty posterior OR raise; either way context is
    # JSON-safe and reports the loaded flag.
    assert isinstance(context, dict)
    assert "loaded" in context


# ── I-3: sensitivity-map wire-in surfaces metadata ─────────────────────────


def test_i3_load_planner_posterior_includes_track_correction_count():
    """Sister wire-in: posterior context exposes track_correction_count.

    This is the autopilot-visible metadata that lets the planner enumerate
    sensitivity-relevant track keys when ranking. The track_correction_count
    proxies for per-axis sensitivity-map richness.
    """
    posterior, context = loop.load_planner_posterior_for_loop()
    # Even with a real posterior, the context payload must carry the key.
    assert isinstance(context, dict)
    if context.get("loaded"):
        assert "track_correction_count" in context
        assert isinstance(context["track_correction_count"], int)


# ── I-4: iter_layer_pairs is documented research-only ─────────────────────


def test_i4_iter_layer_pairs_importable_and_research_only():
    """iter_layer_pairs is still a real callable (research-only utility).

    Per the audit it has 0 consumers. We document it as research-only
    pending the Lane Ω Phase 3 QAT use case.
    """
    from tac.frozen_bit_quant import iter_layer_pairs
    # The function exists; we don't exercise it (no model fixture). The
    # presence of the import + the inline docstring mention of
    # "Lane Ω Phase 3 QAT setup" is the documented reactivation criterion.
    assert callable(iter_layer_pairs)
    # The docstring should name the Phase 3 reactivation criterion.
    doc = (iter_layer_pairs.__doc__ or "")
    assert "Phase 3" in doc or "research" in doc.lower() or "Lane" in doc
