"""Tests for tools/cathedral_autopilot_autonomous_loop.py.

Covers:
  - rank_candidates by eig_per_dollar (descending)
  - rank_candidates by predicted_score_delta (most-negative first)
  - rank_candidates rejects unknown axis
  - eig_per_dollar = +inf for zero-cost candidate
  - make_dispatch_halt_event sets requires_approval=True when DISPATCH gated
  - make_dispatch_halt_event sets requires_approval=False when not gated
  - make_kill_halt_event ALWAYS sets requires_approval=True (CLAUDE.md non-neg)
  - inject_operator_decision records decision + ts
  - inject_operator_decision raises on double-decide
  - check_dispatch_claim_conflict returns False when claims file missing
  - check_dispatch_claim_conflict returns True when candidate present
  - run_one_loop_iteration emits halt events for non-conflicted candidates
  - run_one_loop_iteration excludes conflicted candidates
  - run_one_loop_iteration race_mode trims to negative-delta only
  - run_one_loop_iteration max_dispatch_recommendations cap
  - run_continuous_loop iterates N times
  - run_continuous_loop refuses zero iterations
  - operator_decision_callback is invoked on every requires_approval event
  - default DEFER decision when no callback
  - HALT-and-ASK pattern: pretend operator approves
  - HALT-and-ASK pattern: pretend operator rejects
  - serialize_report produces JSON-safe dict
  - write_report writes a parseable JSON
  - load_candidates_from_jsonl parses correctly
  - main returns 0 on valid input
  - main returns 2 on missing candidates file
  - main returns 2 on zero iterations
  - main writes output when --output given
  - _parse_approval_flags rejects unknown event class
  - main accepts multiple --require-operator-approval-on flags
  - schema version constant stable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cathedral_autopilot_autonomous_loop as loop  # noqa: E402


def _cand(cid: str = "c1", *, family: str = "hnerv_lc_v2",
          predicted_delta: float = -0.005,
          eig: float = 0.5,
          cost_usd: float = 5.0,
          blockers: list[str] | None = None) -> loop.CandidateRow:
    return loop.CandidateRow(
        candidate_id=cid,
        family=family,
        predicted_score_delta=predicted_delta,
        expected_information_gain=eig,
        estimated_dispatch_cost_usd=cost_usd,
        blockers=list(blockers or []),
    )


# ── rank_candidates ────────────────────────────────────────────────────────


def test_rank_by_eig_per_dollar_desc():
    c1 = _cand("c1", eig=1.0, cost_usd=10.0)  # 0.1
    c2 = _cand("c2", eig=2.0, cost_usd=5.0)   # 0.4
    c3 = _cand("c3", eig=0.5, cost_usd=10.0)  # 0.05
    ranked = loop.rank_candidates([c1, c2, c3], rank_axis="eig_per_dollar")
    assert [c.candidate_id for c in ranked] == ["c2", "c1", "c3"]


def test_rank_by_predicted_score_delta_most_negative_first():
    c1 = _cand("c1", predicted_delta=-0.005)
    c2 = _cand("c2", predicted_delta=-0.020)
    c3 = _cand("c3", predicted_delta=+0.010)
    ranked = loop.rank_candidates([c1, c2, c3], rank_axis="predicted_score_delta")
    assert [c.candidate_id for c in ranked] == ["c2", "c1", "c3"]


def test_rank_unknown_axis_raises():
    with pytest.raises(ValueError, match="rank_axis"):
        loop.rank_candidates([], rank_axis="something_made_up")


def test_eig_per_dollar_inf_for_zero_cost():
    c = _cand(eig=1.0, cost_usd=0.0)
    assert c.eig_per_dollar() == float("inf")


def test_eig_per_dollar_inf_for_negative_cost():
    c = _cand(eig=1.0, cost_usd=-1.0)
    assert c.eig_per_dollar() == float("inf")


# ── HALT events ────────────────────────────────────────────────────────────


def test_dispatch_halt_event_requires_approval_when_gated():
    c = _cand()
    e = loop.make_dispatch_halt_event(
        c, requires_approval_classes=frozenset({loop.EventClass.DISPATCH})
    )
    assert e.requires_approval is True
    assert e.event_class == loop.EventClass.DISPATCH


def test_dispatch_halt_event_no_approval_when_not_gated():
    c = _cand()
    e = loop.make_dispatch_halt_event(
        c, requires_approval_classes=frozenset()
    )
    assert e.requires_approval is False


def test_dispatch_halt_event_carries_blockers():
    c = _cand()
    e = loop.make_dispatch_halt_event(
        c, requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        blockers=["dispatch_claim_active"],
    )
    assert "dispatch_claim_active" in e.blockers


def test_kill_halt_event_always_requires_approval():
    e = loop.make_kill_halt_event("c1", "advisory test")
    assert e.requires_approval is True
    assert e.event_class == loop.EventClass.KILL


# ── inject_operator_decision ───────────────────────────────────────────────


def test_inject_decision_records_decision_and_ts():
    e = loop.make_kill_halt_event("c1", "test")
    loop.inject_operator_decision(e, loop.OperatorDecision.REJECT, "no")
    assert e.decision == loop.OperatorDecision.REJECT
    assert e.decision_at_utc is not None
    assert e.decision_notes == "no"


def test_inject_decision_double_decide_raises():
    e = loop.make_kill_halt_event("c1", "test")
    loop.inject_operator_decision(e, loop.OperatorDecision.REJECT)
    with pytest.raises(ValueError, match="already"):
        loop.inject_operator_decision(e, loop.OperatorDecision.APPROVE)


# ── Dispatch-claim conflict check ──────────────────────────────────────────


def test_check_dispatch_claim_no_file_returns_false(tmp_path):
    has, _ = loop.check_dispatch_claim_conflict(
        "c1", claims_path=tmp_path / "missing.md"
    )
    assert has is False


def test_check_dispatch_claim_present_returns_true(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text("c1 some active claim row\n", encoding="utf-8")
    has, reason = loop.check_dispatch_claim_conflict("c1", claims_path=p)
    assert has is True
    assert "c1" in reason


def test_check_dispatch_claim_absent_in_file_returns_false(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text("c2 some other claim\n", encoding="utf-8")
    has, _ = loop.check_dispatch_claim_conflict("c1", claims_path=p)
    assert has is False


# ── Loop iteration ────────────────────────────────────────────────────────


def test_one_iteration_emits_halt_events(tmp_path):
    cands = [_cand("a"), _cand("b")]
    rep = loop.run_one_loop_iteration(
        cands, claims_path=tmp_path / "no_claims.md",
    )
    assert rep.n_candidates_seen == 2
    assert rep.n_candidates_ranked == 2
    assert len(rep.halt_events) == 2
    for e in rep.halt_events:
        assert e.requires_approval is True


def test_one_iteration_blocks_conflicted_candidate(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text("a active\n", encoding="utf-8")
    cands = [_cand("a"), _cand("b")]
    rep = loop.run_one_loop_iteration(cands, claims_path=p)
    assert rep.n_candidates_blocked_by_dispatch_claim == 1
    assert rep.n_candidates_ranked == 1
    assert all(e.candidate_id != "a" for e in rep.halt_events)


def test_one_iteration_race_mode_trims_to_negative_delta(tmp_path):
    cands = [
        _cand("a", predicted_delta=-0.005, cost_usd=2.0),
        _cand("b", predicted_delta=+0.001, cost_usd=1.0),  # positive: dropped
    ]
    rep = loop.run_one_loop_iteration(
        cands, race_mode=True, claims_path=tmp_path / "n.md",
    )
    assert rep.n_candidates_ranked == 1
    assert any("race-mode" in n for n in rep.notes)
    ids = {e.candidate_id for e in rep.halt_events}
    assert ids == {"a"}


def test_one_iteration_max_recommendations_cap(tmp_path):
    cands = [_cand(f"c{i}") for i in range(10)]
    rep = loop.run_one_loop_iteration(
        cands, max_dispatch_recommendations=3,
        claims_path=tmp_path / "n.md",
    )
    assert rep.n_candidates_ranked == 3
    assert len(rep.halt_events) == 3


def test_one_iteration_empty_candidates(tmp_path):
    rep = loop.run_one_loop_iteration(
        [], claims_path=tmp_path / "n.md",
    )
    assert rep.n_candidates_seen == 0
    assert rep.n_candidates_ranked == 0


# ── Continuous loop ───────────────────────────────────────────────────────


def test_continuous_loop_iterates_n_times(tmp_path):
    state = {"call_count": 0}

    def src() -> list[loop.CandidateRow]:
        state["call_count"] += 1
        return [_cand(f"c{state['call_count']}")]

    reports = loop.run_continuous_loop(
        src, iterations=3, claims_path=tmp_path / "n.md",
    )
    assert len(reports) == 3
    assert state["call_count"] == 3
    assert reports[0].iteration == 1
    assert reports[2].iteration == 3


def test_continuous_loop_refuses_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        loop.run_continuous_loop(lambda: [], iterations=0)


def test_continuous_loop_default_decision_is_defer(tmp_path):
    cands = [_cand("a")]
    reports = loop.run_continuous_loop(
        lambda: cands, iterations=1, claims_path=tmp_path / "n.md",
    )
    assert reports[0].halt_events[0].decision == loop.OperatorDecision.DEFER


def test_continuous_loop_callback_approve(tmp_path):
    cands = [_cand("a")]

    def approve(_e):
        return loop.OperatorDecision.APPROVE

    reports = loop.run_continuous_loop(
        lambda: cands, iterations=1,
        operator_decision_callback=approve,
        claims_path=tmp_path / "n.md",
    )
    assert reports[0].halt_events[0].decision == loop.OperatorDecision.APPROVE


def test_continuous_loop_callback_reject(tmp_path):
    cands = [_cand("a")]

    def reject(_e):
        return loop.OperatorDecision.REJECT

    reports = loop.run_continuous_loop(
        lambda: cands, iterations=1,
        operator_decision_callback=reject,
        claims_path=tmp_path / "n.md",
    )
    assert reports[0].halt_events[0].decision == loop.OperatorDecision.REJECT


def test_callback_invoked_only_on_requires_approval(tmp_path):
    cands = [_cand("a")]
    invocations: list[str] = []

    def cb(e):
        invocations.append(e.candidate_id)
        return loop.OperatorDecision.APPROVE

    # All halt events with requires_approval=True call the callback;
    # passing empty approval set means no events block.
    loop.run_continuous_loop(
        lambda: cands, iterations=1,
        requires_approval_on=frozenset(),
        operator_decision_callback=cb,
        claims_path=tmp_path / "n.md",
    )
    assert invocations == []  # no events required approval


# ── Serialization ──────────────────────────────────────────────────────────


def test_serialize_report_is_json_safe(tmp_path):
    cands = [_cand("a")]
    rep = loop.run_one_loop_iteration(cands, claims_path=tmp_path / "n.md")
    payload = loop.serialize_report(rep)
    json.dumps(payload)  # must not raise
    assert payload["schema"] == loop.AUTONOMOUS_LOOP_SCHEMA
    assert payload["halt_events"][0]["event_class"] == "dispatch"


def test_write_report_round_trip(tmp_path):
    cands = [_cand("a")]
    rep = loop.run_one_loop_iteration(cands, claims_path=tmp_path / "n.md")
    out = tmp_path / "rep.json"
    loop.write_report(rep, out)
    raw = json.loads(out.read_text())
    assert raw["iteration"] == 1


# ── Loaders + CLI ─────────────────────────────────────────────────────────


def test_load_candidates_from_jsonl(tmp_path):
    p = tmp_path / "cands.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "a",
            "family": "hnerv_lc_v2",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 5.0,
        }) + "\n"
        + json.dumps({
            "candidate_id": "b",
            "family": "balle_scale_hyperprior",
            "predicted_score_delta": -0.010,
            "expected_information_gain": 0.7,
            "estimated_dispatch_cost_usd": 10.0,
            "blockers": ["needs_phase2_anchor"],
            "notes": "interim",
        }) + "\n",
        encoding="utf-8",
    )
    rows = loop.load_candidates_from_jsonl(p)
    assert len(rows) == 2
    assert rows[0].candidate_id == "a"
    assert rows[1].blockers == ["needs_phase2_anchor"]


def test_main_returns_0_on_valid_input(tmp_path, capsys):
    cands_path = tmp_path / "cands.jsonl"
    cands_path.write_text(
        json.dumps({
            "candidate_id": "a",
            "family": "hnerv_lc_v2",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 5.0,
        }) + "\n",
        encoding="utf-8",
    )
    rc = loop.main([
        "--candidates-jsonl", str(cands_path),
        "--iterations", "1",
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["iterations_run"] == 1


def test_main_returns_2_on_missing_file(tmp_path, capsys):
    rc = loop.main([
        "--candidates-jsonl", str(tmp_path / "nope.jsonl"),
        "--iterations", "1",
    ])
    assert rc == 2


def test_main_returns_2_on_zero_iterations(tmp_path, capsys):
    cands_path = tmp_path / "cands.jsonl"
    cands_path.write_text("", encoding="utf-8")
    rc = loop.main([
        "--candidates-jsonl", str(cands_path),
        "--iterations", "0",
    ])
    assert rc == 2


def test_main_writes_output_file(tmp_path):
    cands_path = tmp_path / "cands.jsonl"
    cands_path.write_text(
        json.dumps({
            "candidate_id": "a", "family": "x",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 5.0,
        }) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out" / "report.json"
    rc = loop.main([
        "--candidates-jsonl", str(cands_path),
        "--iterations", "1",
        "--output", str(out_path),
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert "claude_md_compliance_tags" in payload
    assert "operator_gate_non_negotiable_at_every_dispatch" in payload[
        "claude_md_compliance_tags"
    ]


def test_parse_approval_flags_rejects_unknown(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        loop._parse_approval_flags(["unknown_event"])


def test_parse_approval_flags_accepts_multiple():
    s = loop._parse_approval_flags(["dispatch", "promote"])
    assert loop.EventClass.DISPATCH in s
    assert loop.EventClass.PROMOTE in s


def test_main_multiple_approval_classes(tmp_path):
    cands_path = tmp_path / "cands.jsonl"
    cands_path.write_text(
        json.dumps({
            "candidate_id": "a", "family": "x",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 5.0,
        }) + "\n",
        encoding="utf-8",
    )
    rc = loop.main([
        "--candidates-jsonl", str(cands_path),
        "--iterations", "1",
        "--require-operator-approval-on", "dispatch",
        "--require-operator-approval-on", "kill",
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0


# ── Schema constants ──────────────────────────────────────────────────────


def test_schema_constant_stable():
    assert loop.AUTONOMOUS_LOOP_SCHEMA == "tac_cathedral_autopilot_autonomous_loop_v1"


def test_event_class_enum_members():
    assert loop.EventClass.DISPATCH.value == "dispatch"
    assert loop.EventClass.KILL.value == "kill"
    assert loop.EventClass.PROMOTE.value == "promote"
    assert loop.EventClass.POSTERIOR_REWEIGHT.value == "posterior_reweight"
    assert loop.EventClass.RACE_MODE_TOGGLE.value == "race_mode_toggle"


def test_operator_decision_enum_members():
    assert loop.OperatorDecision.APPROVE.value == "approve"
    assert loop.OperatorDecision.REJECT.value == "reject"
    assert loop.OperatorDecision.DEFER.value == "defer"
