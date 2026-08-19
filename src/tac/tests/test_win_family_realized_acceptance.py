# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.win_families.realized_acceptance` and its ranker layer (F1).

The central positive control is EXECUTED: a descent on a synthetic objective with a
known optimum must actually reach that optimum and report the convergence proof.  The
central negative control is also executed: a ranker must never be able to cause a
non-improving move to be accepted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tac.win_families import proposal_rankers as pr
from tac.win_families import realized_acceptance as ra


# --- a tiny family: minimise squared distance to a target on an integer lattice ---


def _make_engine(target, **kwargs):
    generator = ra.LatticeNeighbourGenerator(offsets=(-1, 1), low=-10, high=10)

    def realize(proposals, coordinate):
        return np.stack([p.state for p in proposals])

    def objective(realized, proposals, coordinate):
        return ((realized - target) ** 2).sum(axis=1).astype(np.float64)

    kwargs.setdefault("proposal_generator", generator)
    kwargs.setdefault("realization_path", realize)
    kwargs.setdefault("joint_objective", objective)
    return ra.RealizedAcceptanceEngine(**kwargs)


TARGET = np.array([3, -5, 2], dtype=np.int64)


# --- the descent itself ------------------------------------------------------


def test_descent_reaches_the_known_optimum():
    """EXECUTED POSITIVE CONTROL: the engine must actually find the optimum."""
    report = _make_engine(TARGET).descend({0: np.zeros(3, dtype=np.int64)})
    assert np.array_equal(report.outcomes[0].state, TARGET)
    assert report.outcomes[0].final_value == 0.0


def test_descent_reports_the_convergence_proof():
    report = _make_engine(TARGET).descend({0: np.zeros(3, dtype=np.int64)})
    assert report.converged is True


def test_descent_starting_at_the_optimum_accepts_nothing():
    report = _make_engine(TARGET).descend({0: TARGET.copy()})
    assert report.outcomes[0].accepted_moves == 0
    assert report.outcomes[0].converged is True


def test_max_passes_cap_reports_not_converged():
    report = _make_engine(TARGET, max_passes=1).descend({0: np.zeros(3, dtype=np.int64)})
    assert report.converged is False


def test_improvement_is_positive_and_matches_the_history():
    outcome = _make_engine(TARGET).descend({0: np.zeros(3, dtype=np.int64)}).outcomes[0]
    assert outcome.improvement == outcome.start_value - outcome.final_value
    assert outcome.history[0] == outcome.start_value
    assert outcome.history[-1] == outcome.final_value


def test_history_is_monotone_non_increasing():
    outcome = _make_engine(TARGET).descend({0: np.zeros(3, dtype=np.int64)}).outcomes[0]
    assert all(b <= a for a, b in zip(outcome.history, outcome.history[1:], strict=False))


def test_multiple_coordinates_are_each_descended():
    engine = _make_engine(TARGET)
    report = engine.descend({0: np.zeros(3, dtype=np.int64), 1: np.ones(3, dtype=np.int64)})
    assert len(report.outcomes) == 2
    assert all(np.array_equal(o.state, TARGET) for o in report.outcomes)


def test_missing_start_state_refuses():
    with pytest.raises(ra.RealizedAcceptanceError, match="no start state"):
        _make_engine(TARGET).descend({0: np.zeros(3, dtype=np.int64)}, coordinates=[0, 5])


# --- vacuity ----------------------------------------------------------------


def test_empty_run_is_not_reported_as_converged():
    """all([]) is True; a run that measured nothing has proved nothing."""
    report = ra.DescentReport(outcomes=(), wall_clock_seconds=0.0)
    assert report.converged is False


# --- acceptance rule ---------------------------------------------------------


def test_acceptance_requires_strict_improvement():
    rule = ra.AcceptanceRule()
    assert rule.accepts(1.0, 0.9) is True
    assert rule.accepts(1.0, 1.0) is False
    assert rule.accepts(1.0, 1.1) is False


def test_acceptance_threshold_refuses_moves_below_it():
    rule = ra.AcceptanceRule(min_improvement=0.5)
    assert rule.accepts(1.0, 0.6) is False
    assert rule.accepts(1.0, 0.4) is True


def test_negative_threshold_refuses():
    with pytest.raises(ra.RealizedAcceptanceError):
        ra.AcceptanceRule(min_improvement=-1.0)


def test_threshold_stops_the_descent_early():
    loose = _make_engine(TARGET, acceptance_rule=ra.AcceptanceRule(min_improvement=100.0))
    outcome = loose.descend({0: np.zeros(3, dtype=np.int64)}).outcomes[0]
    assert outcome.accepted_moves == 0


# --- objective contract ------------------------------------------------------


def test_objective_returning_wrong_shape_refuses():
    engine = _make_engine(
        TARGET, joint_objective=lambda realized, proposals, coordinate: np.zeros(1)
    )
    with pytest.raises(ra.RealizedAcceptanceError, match="one score-unit value per proposal"):
        engine.descend({0: np.zeros(3, dtype=np.int64)})


def test_objective_returning_non_finite_refuses():
    def bad(realized, proposals, coordinate):
        return np.full(len(proposals), np.nan)

    with pytest.raises(ra.RealizedAcceptanceError, match="non-finite"):
        _make_engine(TARGET, joint_objective=bad).descend({0: np.zeros(3, dtype=np.int64)})


# --- proposals + the lattice generator ---------------------------------------


def test_proposal_refuses_non_array_state():
    with pytest.raises(ra.RealizedAcceptanceError, match="ndarray state"):
        ra.Proposal(coordinate=0, label="x", state=[1, 2, 3])


def test_lattice_generator_emits_every_in_box_move():
    generator = ra.LatticeNeighbourGenerator(offsets=(-1, 1), low=-10, high=10)
    proposals = generator(np.zeros(3, dtype=np.int64), 0)
    assert len(proposals) == 3 * 2


def test_lattice_generator_clamps_at_the_box_edge():
    """A move that would leave the box is NOT proposed -- it must not read as a tie."""
    generator = ra.LatticeNeighbourGenerator(offsets=(-1, 1), low=0, high=0)
    assert generator(np.zeros(2, dtype=np.int64), 0) == []


def test_lattice_generator_refuses_zero_offset():
    with pytest.raises(ra.RealizedAcceptanceError, match="incumbent"):
        ra.LatticeNeighbourGenerator(offsets=(0, 1))


def test_lattice_generator_refuses_empty_offsets():
    with pytest.raises(ra.RealizedAcceptanceError):
        ra.LatticeNeighbourGenerator(offsets=())


def test_lattice_generator_refuses_empty_box():
    with pytest.raises(ra.RealizedAcceptanceError, match="empty box"):
        ra.LatticeNeighbourGenerator(low=5, high=1)


def test_lattice_generator_does_not_mutate_the_incumbent():
    state = np.zeros(3, dtype=np.int64)
    ra.LatticeNeighbourGenerator(offsets=(1,), low=-5, high=5)(state, 0)
    assert np.array_equal(state, np.zeros(3, dtype=np.int64))


# --- the free training corpus -------------------------------------------------


def test_events_are_emitted_for_accepts_and_rejects():
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    assert any(e.accepted for e in events)
    assert any(not e.accepted for e in events)


def test_rejects_outnumber_accepts_in_a_real_descent():
    """A ranker trained only on accepts never learns what a bad move looks like."""
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    assert sum(not e.accepted for e in events) > sum(e.accepted for e in events)


def test_event_delta_is_realized_minus_incumbent():
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    for event in events:
        assert event.delta == pytest.approx(event.realized_value - event.incumbent_value)


def test_accepted_events_have_negative_delta():
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    assert all(e.delta < 0 for e in events if e.accepted)


def test_jsonl_event_sink_writes_one_row_per_event(tmp_path):
    sink = ra.JsonlEventSink(tmp_path / "events.jsonl")
    events: list[ra.AcceptanceEvent] = []

    def both(event):
        events.append(event)
        sink(event)

    _make_engine(TARGET, event_sink=both).descend({0: np.zeros(3, dtype=np.int64)})
    rows = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(rows) == len(events)
    assert json.loads(rows[0])["coordinate"] == 0


# --- checkpoint / resume (P0) -------------------------------------------------


def test_checkpoint_is_written_per_coordinate(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    _make_engine(TARGET, checkpoint_path=path).descend(
        {0: np.zeros(3, dtype=np.int64), 1: np.zeros(3, dtype=np.int64)}
    )
    assert len(path.read_text().strip().splitlines()) == 2


def test_resume_skips_completed_coordinates(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    _make_engine(TARGET, checkpoint_path=path).descend({0: np.zeros(3, dtype=np.int64)})
    second = _make_engine(TARGET, checkpoint_path=path).descend({0: np.zeros(3, dtype=np.int64)})
    assert second.resumed_coordinates == 1
    assert second.outcomes[0].realizations > 0


def test_resume_restores_the_solved_state(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    _make_engine(TARGET, checkpoint_path=path).descend({0: np.zeros(3, dtype=np.int64)})
    resumed = _make_engine(TARGET, checkpoint_path=path).descend({0: np.zeros(3, dtype=np.int64)})
    assert np.array_equal(np.asarray(resumed.outcomes[0].state), TARGET)


def test_resume_spends_no_new_realizations(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    calls = {"n": 0}

    def counting_realize(proposals, coordinate):
        calls["n"] += 1
        return np.stack([p.state for p in proposals])

    _make_engine(TARGET, checkpoint_path=path, realization_path=counting_realize).descend(
        {0: np.zeros(3, dtype=np.int64)}
    )
    before = calls["n"]
    _make_engine(TARGET, checkpoint_path=path, realization_path=counting_realize).descend(
        {0: np.zeros(3, dtype=np.int64)}
    )
    assert calls["n"] == before


def test_torn_checkpoint_line_is_survivable(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text('{"coordinate": 0, "start_value": 1.0, "final_value": 0.0, "passes": 1,'
                    ' "realizations": 1, "converged": true, "accepted_moves": 0,'
                    ' "state": [3, -5, 2], "history": [1.0]}\n{"coordi')
    report = _make_engine(TARGET, checkpoint_path=path).descend({0: np.zeros(3, dtype=np.int64)})
    assert report.resumed_coordinates == 1


def test_absent_checkpoint_reads_as_empty(tmp_path):
    engine = _make_engine(TARGET, checkpoint_path=tmp_path / "missing.jsonl")
    assert engine.completed_coordinates() == {}


def test_negative_max_passes_refuses():
    with pytest.raises(ra.RealizedAcceptanceError):
        _make_engine(TARGET, max_passes=-1)


# --- rankers: the safety boundary --------------------------------------------


def test_ranker_cannot_cause_a_non_improving_move_to_be_accepted():
    """NEGATIVE CONTROL: an adversarial ranker that puts the WORST move first.

    Acceptance is realized-only, so the engine must still reach the true optimum.
    """

    def worst_first(proposals, coordinate):
        return list(reversed(list(proposals)))

    report = _make_engine(TARGET, ranker=worst_first).descend({0: np.zeros(3, dtype=np.int64)})
    assert np.array_equal(report.outcomes[0].state, TARGET)
    assert report.outcomes[0].final_value == 0.0


def test_identity_ranker_is_a_no_op():
    proposals = [ra.Proposal(0, "a", np.zeros(2, dtype=np.int64))]
    assert pr.IdentityRanker()(proposals, 0) == proposals


def test_identity_ranker_does_not_weaken_the_convergence_proof():
    assert pr.IdentityRanker.config.convergence_proof_weakened is False


def test_truncating_ranker_declares_the_weakened_proof():
    assert pr.RankerConfig(top_k=3).convergence_proof_weakened is True


def test_ranker_config_refuses_negative_top_k():
    with pytest.raises(pr.RankerError):
        pr.RankerConfig(top_k=-1)


def test_ranker_truncation_limits_the_candidate_list():
    config = pr.RankerConfig(top_k=2)
    assert len(config.apply([1, 2, 3, 4])) == 2


# --- geometric rankers --------------------------------------------------------


def _margin_setup():
    incumbent = np.zeros((2, 2), dtype=np.int64)
    margins = np.array([[9.0, 0.1], [5.0, 7.0]])
    low = ra.Proposal(0, "low", np.array([[0, 1], [0, 0]], dtype=np.int64))
    high = ra.Proposal(0, "high", np.array([[1, 0], [0, 0]], dtype=np.int64))
    return incumbent, margins, [high, low]


def test_margin_ranker_puts_the_low_margin_move_first():
    incumbent, margins, proposals = _margin_setup()
    ranker = pr.MarginSaliencyRanker(
        margin_field={0: margins}, incumbent_field={0: incumbent}
    )
    assert [p.label for p in ranker(proposals, 0)] == ["low", "high"]


def test_margin_ranker_records_the_margin_as_the_rank_score():
    incumbent, margins, proposals = _margin_setup()
    ranker = pr.MarginSaliencyRanker(
        margin_field={0: margins}, incumbent_field={0: incumbent}
    )
    assert ranker(proposals, 0)[0].rank_score == pytest.approx(0.1)


def test_margin_ranker_refuses_a_missing_margin_field():
    incumbent, _, proposals = _margin_setup()
    ranker = pr.MarginSaliencyRanker(margin_field={}, incumbent_field={0: incumbent})
    with pytest.raises(pr.RankerError, match="no margin field"):
        ranker(proposals, 0)


def test_margin_ranker_refuses_a_missing_incumbent():
    """Without an incumbent every proposal scores the field minimum -- a constant."""
    _, margins, proposals = _margin_setup()
    ranker = pr.MarginSaliencyRanker(margin_field={0: margins}, incumbent_field={})
    with pytest.raises(pr.RankerError, match="constant wearing a geometry"):
        ranker(proposals, 0)


def test_margin_ranker_refuses_a_lattice_mismatch():
    incumbent = np.zeros((2, 2), dtype=np.int64)
    ranker = pr.MarginSaliencyRanker(
        margin_field={0: np.zeros((3, 3))}, incumbent_field={0: incumbent}
    )
    with pytest.raises(pr.RankerError, match="different lattices"):
        ranker([ra.Proposal(0, "a", np.zeros((2, 2), dtype=np.int64))], 0)


def test_margin_ranker_sends_no_op_proposals_last():
    incumbent = np.zeros((2, 2), dtype=np.int64)
    margins = np.array([[9.0, 0.1], [5.0, 7.0]])
    noop = ra.Proposal(0, "noop", incumbent.copy())
    real = ra.Proposal(0, "real", np.array([[0, 1], [0, 0]], dtype=np.int64))
    ranker = pr.MarginSaliencyRanker(
        margin_field={0: margins}, incumbent_field={0: incumbent}
    )
    assert [p.label for p in ranker([noop, real], 0)] == ["real", "noop"]


def test_jacobian_ranker_prefers_the_high_leverage_coefficient():
    jacobian = np.array([[0.01, 5.0]])  # coefficient 1 buys far more objective per step
    ranker = pr.JacobianConditioningRanker(jacobians={0: jacobian})
    proposals = [
        ra.Proposal(0, "d0+1", np.array([1, 0], dtype=np.int64)),
        ra.Proposal(0, "d1+1", np.array([0, 1], dtype=np.int64)),
    ]
    assert [p.label for p in ranker(proposals, 0)] == ["d1+1", "d0+1"]


def test_jacobian_ranker_refuses_a_missing_jacobian():
    with pytest.raises(pr.RankerError, match="no Jacobian"):
        pr.JacobianConditioningRanker(jacobians={})([], 0)


def test_jacobian_ranker_refuses_a_non_matrix_jacobian():
    ranker = pr.JacobianConditioningRanker(jacobians={0: np.zeros(3)})
    with pytest.raises(pr.RankerError, match="2-D"):
        ranker([ra.Proposal(0, "d0+1", np.zeros(3, dtype=np.int64))], 0)


def test_jacobian_ranker_composes_with_the_lattice_generator_labels():
    """The slot is recovered from the generator's own ``dN+M`` label, not guessed."""
    generator = ra.LatticeNeighbourGenerator(offsets=(1,), low=-5, high=5)
    proposals = generator(np.zeros(3, dtype=np.int64), 0)
    jacobian = np.array([[1.0, 9.0, 2.0]])
    ranked = pr.JacobianConditioningRanker(jacobians={0: jacobian})(proposals, 0)
    assert ranked[0].label == "d1+1"


# --- learned ranker: interface only -------------------------------------------


def test_learned_ranker_refuses_without_a_model():
    with pytest.raises(pr.RankerError, match="requires a trained model"):
        pr.LearnedRanker(model=None)


def test_learned_ranker_refuses_a_non_callable_model():
    with pytest.raises(pr.RankerError, match="callable"):
        pr.LearnedRanker(model=42)


def test_learned_ranker_orders_by_model_score():
    ranker = pr.LearnedRanker(model=lambda features: np.array([2.0, 1.0]))
    proposals = [
        ra.Proposal(0, "a", np.zeros(2, dtype=np.int64)),
        ra.Proposal(0, "b", np.zeros(2, dtype=np.int64)),
    ]
    assert [p.label for p in ranker(proposals, 0)] == ["b", "a"]


def test_learned_ranker_refuses_a_bad_model_output_shape():
    ranker = pr.LearnedRanker(model=lambda features: np.zeros(5))
    with pytest.raises(pr.RankerError, match="returned shape"):
        ranker([ra.Proposal(0, "a", np.zeros(2, dtype=np.int64))], 0)


def test_learned_ranker_on_empty_proposals_is_empty():
    assert pr.LearnedRanker(model=lambda f: np.zeros(len(f)))([], 0) == []


# --- the training table --------------------------------------------------------


def test_training_table_covers_every_event():
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    features, labels, names = pr.training_table_from_events(events)
    assert features.shape == (len(events), len(names))
    assert labels.shape == (len(events),)


def test_training_table_labels_are_the_realized_deltas():
    events: list[ra.AcceptanceEvent] = []
    _make_engine(TARGET, event_sink=events.append).descend({0: np.zeros(3, dtype=np.int64)})
    _, labels, _ = pr.training_table_from_events(events)
    assert labels[0] == pytest.approx(events[0].delta)


def test_training_table_refuses_an_empty_log():
    with pytest.raises(pr.RankerError, match="empty event log"):
        pr.training_table_from_events([])


def test_feature_names_are_a_stable_contract():
    assert pr.FEATURE_NAMES == (
        "coordinate",
        "pass_index",
        "incumbent_value",
        "rank_score",
    )


# --- ranker quality -------------------------------------------------------------


def test_perfect_ranker_finds_the_best_first():
    quality = pr.ranker_quality([0.0, 1.0, 2.0], [1.0, 5.0, 9.0])
    assert quality.top1_is_best is True
    assert quality.realizations_to_best == 1
    assert quality.spearman == pytest.approx(1.0)


def test_worst_ranker_finds_the_best_last():
    quality = pr.ranker_quality([0.0, 1.0, 2.0], [9.0, 5.0, 1.0])
    assert quality.top1_is_best is False
    assert quality.realizations_to_best == 3
    assert quality.spearman == pytest.approx(-1.0)


def test_ranker_quality_refuses_mismatched_lengths():
    with pytest.raises(pr.RankerError, match="disagree"):
        pr.ranker_quality([1.0, 2.0], [1.0])


def test_ranker_quality_refuses_empty():
    with pytest.raises(pr.RankerError, match="zero proposals"):
        pr.ranker_quality([], [])


def test_ranker_quality_on_a_single_proposal_is_defined():
    assert pr.ranker_quality([0.0], [1.0]).spearman == 1.0


def test_ranker_quality_json_carries_the_cost_note():
    assert "IdentityRanker" in pr.ranker_quality([0.0], [1.0]).to_json()["note"]
