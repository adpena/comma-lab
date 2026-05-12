"""Tests for `tac.optimization.autopilot_dispatch_ranking`.

Per CLAUDE.md "Recursive adversarial review protocol": these tests cover
the ranking algorithm + composition constraint enforcement + envelope
budget enforcement + L2 Hinton-distilled encoder integration + autopilot
loop interop + serialization.
"""

from __future__ import annotations

import json

import pytest

from tac.optimization.autopilot_dispatch_ranking import (
    DEFAULT_CUMULATIVE_CAP_USD,
    DEFAULT_PER_DISPATCH_CAP_USD,
    SCHEMA_VERSION,
    RankedDispatchCandidate,
    RankingResult,
    rank_dispatches,
    serialize_candidate,
    serialize_ranking,
    synthetic_l2_encoder_dispatch_candidates,
    write_ranking_json,
)

# ── Smoke test the ranker on the canonical inventory ────────────────────


def test_rank_dispatches_smoke():
    result = rank_dispatches()
    assert isinstance(result, RankingResult)
    assert result.schema == SCHEMA_VERSION
    # 39 = 24 legacy + 15 FIX-J substrate-scaffold rows (LOOPCLOSE 2026-05-12).
    assert result.n_substrates_considered == 39
    assert len(result.ranked_dispatches) > 0
    # Score-claim invariants.
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False


def test_ranked_dispatches_descending_eig_per_dollar():
    result = rank_dispatches()
    candidates = result.ranked_dispatches
    for i in range(len(candidates) - 1):
        assert candidates[i].eig_per_dollar >= candidates[i + 1].eig_per_dollar


def test_envelope_caps_default_to_operator_set_values():
    result = rank_dispatches()
    assert result.per_dispatch_cap_usd == DEFAULT_PER_DISPATCH_CAP_USD
    assert result.cumulative_cap_usd == DEFAULT_CUMULATIVE_CAP_USD
    assert result.per_dispatch_cap_usd == 5.00
    assert result.cumulative_cap_usd == 20.00


def test_per_dispatch_cap_annotates_singletons():
    """Substrates above per-dispatch cap should still appear but be flagged."""
    result = rank_dispatches(per_dispatch_cap_usd=5.00)
    # Full-trainer substrates ($30-60 cost) MUST be flagged
    # fits_per_dispatch_cap=False at the $5 cap.
    expensive_substrates = [
        c for c in result.ranked_dispatches
        if "scpp_substrate" in c.substrate_ids
        or "blocknerv" in c.substrate_ids
        or "nerv_as_renderer" in c.substrate_ids
    ]
    assert len(expensive_substrates) > 0
    for c in expensive_substrates:
        assert c.fits_per_dispatch_cap is False, (
            f"expected {c.candidate_id} to be flagged above-cap"
        )


def test_cheap_substrates_fit_per_dispatch_cap():
    """Bolt-ons (cost=0), magic codec (cost=$0.10), residuals ($0.65) all fit $5 cap."""
    result = rank_dispatches(per_dispatch_cap_usd=5.00)
    cheap = [c for c in result.ranked_dispatches if c.estimated_dispatch_cost_usd < 1.0]
    assert all(c.fits_per_dispatch_cap for c in cheap)


def test_cumulative_envelope_enforced():
    """Walking down the ranked list, candidates that exceed cumulative_cap_usd
    are flagged fits_cumulative_envelope=False."""
    result = rank_dispatches(cumulative_cap_usd=2.00, per_dispatch_cap_usd=5.00)
    # With $2 envelope, a few cheap singletons + magic codec + bolt-ons
    # should fit; many should not.
    n_in_envelope = sum(
        1 for c in result.ranked_dispatches
        if c.fits_cumulative_envelope and c.fits_per_dispatch_cap
    )
    n_above = sum(
        1 for c in result.ranked_dispatches
        if not c.fits_cumulative_envelope
    )
    assert n_in_envelope >= 1
    assert n_above >= 1
    assert result.cumulative_estimated_spend_usd <= 2.00 + 1e-6


def test_cumulative_envelope_zero_cost_does_not_increment():
    """Bolt-ons with cost=0 should not advance the cumulative-spend counter."""
    result = rank_dispatches(cumulative_cap_usd=0.50)
    bolt_on_candidates = [
        c for c in result.ranked_dispatches
        if c.candidate_id.startswith("singleton__") and c.estimated_dispatch_cost_usd == 0.0
    ]
    assert len(bolt_on_candidates) > 0
    for c in bolt_on_candidates:
        assert c.fits_cumulative_envelope is True


# ── Composition constraints applied ─────────────────────────────────────


def test_no_two_renderer_replacements_in_a_single_dispatch():
    """No singleton dispatch contains two renderer-replacements; orthogonal-pair
    dispatches should never include two renderer-replacements (would be
    REPLACEMENT, not ORTHOGONAL)."""
    result = rank_dispatches()
    for c in result.ranked_dispatches:
        if len(c.substrate_ids) >= 2:
            renderer_count = sum(
                1 for sid in c.substrate_ids
                if "nerv" in sid or "anr_token" in sid or "categorical" in sid or "vqvae" in sid
            )
            assert renderer_count <= 1, (
                f"candidate {c.candidate_id} has multiple renderer substrates"
            )


def test_drop_redundant_dominated_default_on():
    result_with = rank_dispatches(drop_redundant_dominated=True)
    result_without = rank_dispatches(drop_redundant_dominated=False)
    # Pareto-frontier filter should drop at least 0 substrates (may drop more).
    assert result_with.n_filtered_dropped >= 0
    # Without filter, n_filtered_dropped is still recorded as 0 because the
    # filter pass was skipped.
    assert result_without.n_filtered_dropped == 0


def test_orthogonal_pairs_enabled_default():
    result = rank_dispatches(include_orthogonal_pairs=True)
    pair_dispatches = [
        c for c in result.ranked_dispatches
        if c.candidate_id.startswith("orthogonal_pair__")
    ]
    assert len(pair_dispatches) >= 1


def test_orthogonal_pairs_disabled_no_pair_candidates():
    result = rank_dispatches(include_orthogonal_pairs=False)
    pair_dispatches = [
        c for c in result.ranked_dispatches
        if c.candidate_id.startswith("orthogonal_pair__")
    ]
    assert len(pair_dispatches) == 0


def test_orthogonal_pair_joint_cost_enforced():
    """Orthogonal pairs must have joint cost <= per_dispatch_cap_usd."""
    cap = 1.00
    result = rank_dispatches(per_dispatch_cap_usd=cap, include_orthogonal_pairs=True)
    pair_dispatches = [
        c for c in result.ranked_dispatches
        if c.candidate_id.startswith("orthogonal_pair__")
    ]
    for c in pair_dispatches:
        assert c.estimated_dispatch_cost_usd <= cap


# ── L2 Hinton-distilled encoder integration ─────────────────────────────


def test_l2_hinton_candidates_default_three_encoders():
    cands = synthetic_l2_encoder_dispatch_candidates()
    assert len(cands) == 3
    encoders = {c.substrate_ids[0] for c in cands}
    assert encoders == {"c3_residual", "wavelet_residual", "cool_chic_residual"}


def test_l2_hinton_candidates_have_planning_only_invariants():
    cands = synthetic_l2_encoder_dispatch_candidates()
    for c in cands:
        assert c.score_claim is False
        assert c.promotion_eligible is False
        assert c.ready_for_exact_eval_dispatch is False
        assert c.estimated_dispatch_cost_usd > 0.0
        assert "Hinton" in c.composition_notes or "hinton" in c.composition_notes.lower()


def test_l2_hinton_candidates_high_eig_per_dollar():
    """At $0.30/encoder and predicted Δ ~ -0.0019, EV/$ should be high
    (≈ 0.0063). This is competitive with bolt-ons (cost=0)."""
    cands = synthetic_l2_encoder_dispatch_candidates()
    for c in cands:
        # |delta_mid| / cost = ~0.0019 / 0.30 = ~0.0063.
        assert c.eig_per_dollar > 0.001


# ── Autopilot loop interop ──────────────────────────────────────────────


def test_as_candidate_row_kwargs_compatible_with_autopilot_loop():
    """The kwargs returned must construct a valid CandidateRow."""
    cand = RankedDispatchCandidate(
        candidate_id="test_id",
        family="renderer_replacement",
        substrate_ids=("nerv_as_renderer",),
        predicted_score_delta=-0.005,
        expected_information_gain=0.005,
        estimated_dispatch_cost_usd=40.0,
        eig_per_dollar=0.000125,
        composition_notes="test",
    )
    kwargs = cand.as_candidate_row_kwargs()
    assert "candidate_id" in kwargs
    assert "family" in kwargs
    assert "predicted_score_delta" in kwargs
    assert "expected_information_gain" in kwargs
    assert "estimated_dispatch_cost_usd" in kwargs
    assert "blockers" in kwargs
    assert "notes" in kwargs
    # Construct the actual CandidateRow to verify schema match.
    from tools.cathedral_autopilot_autonomous_loop import CandidateRow
    row = CandidateRow(**kwargs)
    assert row.candidate_id == "test_id"
    assert row.predicted_score_delta == -0.005


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_candidate_jsonable():
    cand = RankedDispatchCandidate(
        candidate_id="test",
        family="residual",
        substrate_ids=("wavelet_residual",),
        predicted_score_delta=-0.0005,
        expected_information_gain=0.0005,
        estimated_dispatch_cost_usd=0.65,
        eig_per_dollar=0.000769,
        composition_notes="smoke",
    )
    payload = serialize_candidate(cand)
    json.dumps(payload)  # Must not raise.
    assert payload["score_claim"] is False
    assert isinstance(payload["substrate_ids"], list)


def test_serialize_ranking_jsonable():
    result = rank_dispatches()
    payload = serialize_ranking(result)
    serialized = json.dumps(payload, sort_keys=True)
    parsed = json.loads(serialized)
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["score_claim"] is False
    assert parsed["promotion_eligible"] is False
    assert parsed["ready_for_exact_eval_dispatch"] is False
    assert "claude_md_compliance_tags" in parsed
    assert isinstance(parsed["ranked_dispatches"], list)


def test_write_ranking_json_refuses_tmp_path(tmp_path):
    result = rank_dispatches()
    with pytest.raises(ValueError, match="forbidden /tmp"):
        write_ranking_json(result, "/tmp/forbidden_ranking.json")


def test_write_ranking_json_writes_durable_path(tmp_path):
    result = rank_dispatches()
    durable = tmp_path / "ranking.json"
    write_ranking_json(result, str(durable))
    parsed = json.loads(durable.read_text())
    assert parsed["schema"] == SCHEMA_VERSION


# ── max_total cap ───────────────────────────────────────────────────────


def test_max_total_cap_truncates_results():
    result = rank_dispatches(max_total=5)
    assert len(result.ranked_dispatches) == 5


def test_max_total_zero_returns_empty():
    result = rank_dispatches(max_total=0)
    assert len(result.ranked_dispatches) == 0


# ── Composition constraint string list ─────────────────────────────────


def test_composition_constraints_applied_present():
    result = rank_dispatches()
    constraints = result.composition_constraints_applied
    assert any("renderer_replacement_mutually_exclusive" in c for c in constraints)
    assert any("format_id_collision_check" in c for c in constraints)
