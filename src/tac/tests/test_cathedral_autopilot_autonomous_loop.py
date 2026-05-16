# SPDX-License-Identifier: MIT
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
          blockers: list[str] | None = None,
          dispatch_packet_ready: bool = True,
          lane_id: str | None = None,
          target_modes: list[str] | None = None,
          dispatch_packet_sha256: str | None = None) -> loop.CandidateRow:
    return loop.CandidateRow(
        candidate_id=cid,
        family=family,
        predicted_score_delta=predicted_delta,
        expected_information_gain=eig,
        estimated_dispatch_cost_usd=cost_usd,
        blockers=list(blockers or []),
        dispatch_packet_ready=dispatch_packet_ready,
        lane_id=lane_id if lane_id is not None else f"lane_{cid}",
        target_modes=(
            list(target_modes)
            if target_modes is not None
            else [loop.AUTOPILOT_CONTEST_TARGET_MODE]
        ),
        dispatch_packet_sha256=(
            dispatch_packet_sha256
            if dispatch_packet_sha256 is not None
            else f"dispatch_packet_sha256_for_{cid}"
        ),
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


def test_check_dispatch_claim_present_by_lane_id_returns_true(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text("lane_exact_eval_a active claim row\n", encoding="utf-8")
    has, reason = loop.check_dispatch_claim_conflict(
        "candidate_a", claim_keys=["lane_exact_eval_a"], claims_path=p
    )
    assert has is True
    assert "lane_exact_eval_a" in reason


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


def test_one_iteration_blocks_conflicted_lane_id(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text("lane_a active\n", encoding="utf-8")
    cands = [_cand("candidate_a", lane_id="lane_a"), _cand("candidate_b")]
    rep = loop.run_one_loop_iteration(cands, claims_path=p)
    assert rep.n_candidates_blocked_by_dispatch_claim == 1
    assert rep.n_candidates_ranked == 1
    assert all(e.candidate_id != "candidate_a" for e in rep.halt_events)


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
    assert rows[0].dispatch_packet_ready is False
    assert rows[0].target_modes == []
    assert rows[1].blockers == ["needs_phase2_anchor"]


def _write_probe_payload(tmp_path: Path, payload: dict | None = None) -> Path:
    payload = payload or {
        "schema": "zen_floor_disambiguator_v1",
        "tool": "tools/probe_zen_floor_disambiguator.py",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "autopilot_rows": [
            {
                "candidate_id": "lane_zen_floor_probe_disambiguator_20260514",
                "family": "zen_floor_planning_probe",
                "predicted_score_delta": 0.0,
                "expected_information_gain": 2.0,
                "estimated_dispatch_cost_usd": 0.0,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "mdl_tier_c_density": 0.25,
                "composition_alpha": 0.8,
                "blockers": ["byte_closed_codec_candidate_required_before_dispatch"],
                "notes": "[proxy] selected_interpretation=proxy_static_floor",
            }
        ],
    }
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_probe_disambiguator_autopilot_rows_read_only(tmp_path):
    p = _write_probe_payload(tmp_path)
    rows = loop.load_candidates_from_probe_disambiguator_output(p)
    assert len(rows) == 1
    row = rows[0]
    assert row.candidate_id == "lane_zen_floor_probe_disambiguator_20260514"
    assert row.family == "zen_floor_planning_probe"
    assert "byte_closed_codec_candidate_required_before_dispatch" in row.blockers
    assert loop.PLANNING_ONLY_SOURCE_BLOCKER in row.blockers
    assert row.mdl_tier_c_density == pytest.approx(0.25)
    assert row.composition_alpha == pytest.approx(0.8)
    assert "[probe-disambiguator; read-only planning]" in row.notes


def test_load_probe_disambiguator_refuses_score_claim(tmp_path):
    payload = {
        "schema": "zen_floor_disambiguator_v1",
        "score_claim": True,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "autopilot_rows": [],
    }
    p = _write_probe_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="score_claim=True"):
        loop.load_candidates_from_probe_disambiguator_output(p)


def test_main_accepts_probe_disambiguator_source(tmp_path):
    p = _write_probe_payload(tmp_path)
    out_path = tmp_path / "probe_report.json"
    rc = loop.main([
        "--probe-disambiguator-json", str(p),
        "--iterations", "1",
        "--output", str(out_path),
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert "probe_disambiguator_read_only_source" in payload[
        "claude_md_compliance_tags"
    ]
    assert payload["probe_disambiguator_source"]["read_only_consumer"] is True


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


# ── Operator-authorized le-$5/individual mode (2026-05-11) ────────────────


def _auth_mode(
    tmp_path,
    *,
    enabled: bool = True,
    per_dispatch_cap: float = 5.0,
    cumulative_cap: float = 20.0,
    helper_exists: bool = True,
) -> loop.OperatorAuthorizedModeConfig:
    helper = tmp_path / "claim_lane_dispatch.py"
    if helper_exists:
        helper.write_text("# canonical helper\n", encoding="utf-8")
    journal = tmp_path / "autopilot_journal.jsonl"
    return loop.OperatorAuthorizedModeConfig(
        enabled=enabled,
        per_dispatch_cap_usd=per_dispatch_cap,
        cumulative_cap_usd=cumulative_cap,
        canonical_helper_script=helper,
        journal_path=journal,
    )


def test_authorized_mode_disabled_by_default():
    cfg = loop.OperatorAuthorizedModeConfig()
    assert cfg.enabled is False
    assert cfg.per_dispatch_cap_usd == loop.DEFAULT_PER_DISPATCH_CAP_USD
    assert cfg.cumulative_cap_usd == loop.DEFAULT_CUMULATIVE_CAP_USD
    assert cfg.cumulative_spent_usd == 0.0


def test_authorized_mode_default_caps_match_operator_directive():
    # Operator directive 2026-05-11: le-$5/individual + le-$20 cumulative.
    assert loop.DEFAULT_PER_DISPATCH_CAP_USD == 5.0
    assert loop.DEFAULT_CUMULATIVE_CAP_USD == 20.0


def test_can_authorize_refuses_when_disabled(tmp_path):
    cfg = _auth_mode(tmp_path, enabled=False)
    c = _cand(cost_usd=1.0)
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "OFF" in reason


def test_can_authorize_refuses_above_per_dispatch_cap(tmp_path):
    cfg = _auth_mode(tmp_path, per_dispatch_cap=5.0)
    c = _cand(cost_usd=5.01)
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "per-dispatch cap" in reason


def test_can_authorize_refuses_above_cumulative_envelope(tmp_path):
    cfg = _auth_mode(tmp_path, per_dispatch_cap=5.0, cumulative_cap=10.0)
    cfg.cumulative_spent_usd = 7.0
    c = _cand(cost_usd=4.0)  # 7+4=11 > 10
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "envelope" in reason


def test_can_authorize_refuses_when_blockers_present(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=1.0, blockers=["unresolved_dep"])
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "blockers" in reason


def test_can_authorize_refuses_when_helper_missing(tmp_path):
    cfg = _auth_mode(tmp_path, helper_exists=False)
    c = _cand(cost_usd=1.0)
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "helper" in reason


def test_can_authorize_refuses_non_positive_cost(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=0.0)
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "non-positive" in reason


def test_can_authorize_refuses_planning_row_without_dispatch_packet(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_ready=False,
        lane_id="",
        target_modes=[],
        dispatch_packet_sha256="",
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "dispatch-authority packet" in reason
    assert "dispatch_packet_ready_false" in reason
    assert "lane_id_required_for_dispatch_packet" in reason
    assert "contest_exact_eval_target_mode_required" in reason


def test_can_authorize_approves_within_caps(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=4.5)
    ok, reason = cfg.can_authorize(c)
    assert ok is True
    assert reason == ""


def test_record_authorization_increments_cumulative(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=2.5)
    cfg.record_authorization(c)
    assert cfg.cumulative_spent_usd == 2.5
    cfg.record_authorization(c)
    assert cfg.cumulative_spent_usd == 5.0


def test_make_dispatch_halt_event_authorized_when_mode_on_and_env_set(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=3.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
    )
    assert e.autopilot_authorized is True
    assert e.autopilot_tag == loop.AUTOPILOT_AUTHORIZED_TAG
    assert e.requires_approval is False
    assert cfg.cumulative_spent_usd == 3.0


def test_make_dispatch_halt_event_refused_when_env_not_set(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=3.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=False,
    )
    assert e.autopilot_authorized is False
    assert e.requires_approval is True  # falls back to operator gate
    assert "env-var" in e.autopilot_refused_reason
    assert cfg.cumulative_spent_usd == 0.0


def test_make_dispatch_halt_event_refused_when_over_per_dispatch_cap(tmp_path):
    cfg = _auth_mode(tmp_path, per_dispatch_cap=5.0)
    c = _cand(cost_usd=7.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
    )
    assert e.autopilot_authorized is False
    assert e.requires_approval is True
    assert "per-dispatch cap" in e.autopilot_refused_reason


def test_make_dispatch_halt_event_no_auth_mode_keeps_existing_behaviour(tmp_path):
    c = _cand(cost_usd=3.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=None,
        env_authorized=True,
    )
    assert e.autopilot_authorized is False
    assert e.requires_approval is True


def test_kill_event_never_auto_authorized(tmp_path):
    e = loop.make_kill_halt_event("c1", "advisory")
    # KILL events bypass dispatch helpers; they always require approval.
    assert e.requires_approval is True
    assert e.autopilot_authorized is False


def test_loop_iteration_journal_row_written_on_authorization(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand("a", cost_usd=2.0)
    rep = loop.run_one_loop_iteration(
        [c],
        claims_path=tmp_path / "no_claims.md",
        auth_mode=cfg,
        env_authorized=True,
    )
    assert rep.halt_events[0].autopilot_authorized is True
    assert cfg.journal_path.is_file()
    lines = cfg.journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["candidate_id"] == "a"
    assert row["autopilot_authorized"] is True
    assert row["autopilot_tag"] == loop.AUTOPILOT_AUTHORIZED_TAG


def test_loop_iteration_journal_appends_one_row_per_authorization(tmp_path):
    cfg = _auth_mode(tmp_path)
    cands = [_cand("a", cost_usd=2.0), _cand("b", cost_usd=3.0)]
    rep = loop.run_one_loop_iteration(
        cands,
        claims_path=tmp_path / "no_claims.md",
        auth_mode=cfg,
        env_authorized=True,
    )
    assert all(e.autopilot_authorized for e in rep.halt_events)
    lines = cfg.journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert cfg.cumulative_spent_usd == 5.0


def test_loop_iteration_cumulative_envelope_halts_excess(tmp_path):
    cfg = _auth_mode(tmp_path, cumulative_cap=10.0)
    cands = [
        _cand("a", cost_usd=5.0),
        _cand("b", cost_usd=4.0),
        _cand("c", cost_usd=4.0),  # 5+4+4=13 > 10
    ]
    rep = loop.run_one_loop_iteration(
        cands,
        claims_path=tmp_path / "no_claims.md",
        auth_mode=cfg,
        env_authorized=True,
        rank_axis="predicted_score_delta",
    )
    # First two authorized, third refused on envelope.
    authorized = [e for e in rep.halt_events if e.autopilot_authorized]
    refused = [e for e in rep.halt_events if not e.autopilot_authorized]
    assert len(authorized) == 2
    assert len(refused) == 1
    assert "envelope" in refused[0].autopilot_refused_reason
    assert refused[0].requires_approval is True
    assert cfg.cumulative_spent_usd == 9.0


def test_env_authorizes_mode_reads_real_env(monkeypatch):
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "1")
    assert loop._env_authorizes_mode() is True
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "0")
    assert loop._env_authorizes_mode() is False
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    assert loop._env_authorizes_mode() is False


def test_env_authorizes_mode_accepts_injected_dict():
    assert loop._env_authorizes_mode(
        {loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR: "1"}
    ) is True
    assert loop._env_authorizes_mode(
        {loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR: "true"}
    ) is False
    assert loop._env_authorizes_mode({}) is False


def test_main_refuses_authorized_mode_without_journal(tmp_path, capsys):
    cand_file = tmp_path / "c.jsonl"
    cand_file.write_text(
        json.dumps({
            "candidate_id": "test_auth_mode_uniq_abc123",
            "family": "hnerv",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 1.0,
        }) + "\n",
        encoding="utf-8",
    )
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        # No --journal-path
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--journal-path" in err


def test_main_authorized_mode_with_journal_succeeds(tmp_path, monkeypatch, capsys):
    cand_file = tmp_path / "c.jsonl"
    cand_file.write_text(
        json.dumps({
            "candidate_id": "test_auth_mode_uniq_abc123",
            "family": "hnerv",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 2.0,
            "dispatch_packet_ready": True,
            "dispatch_packet_sha256": "dispatch_packet_sha256_for_test_auth_mode_uniq_abc123",
            "lane_id": "lane_test_auth_mode_uniq_abc123",
            "target_modes": [loop.AUTOPILOT_CONTEST_TARGET_MODE],
        }) + "\n",
        encoding="utf-8",
    )
    helper = tmp_path / "claim_lane_dispatch.py"
    helper.write_text("# canonical\n", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "1")
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        "--journal-path", str(journal),
        "--canonical-helper-script", str(helper),
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_authorized_mode"]["enabled"] is True
    assert payload["operator_authorized_mode"]["env_authorized"] is True
    assert payload["operator_authorized_mode"]["cumulative_spent_usd"] == 2.0
    assert journal.is_file()


def test_main_authorized_mode_without_env_warns_but_still_runs(tmp_path, monkeypatch, capsys):
    cand_file = tmp_path / "c.jsonl"
    cand_file.write_text(
        json.dumps({
            "candidate_id": "test_auth_mode_uniq_abc123",
            "family": "hnerv",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 1.5,
        }) + "\n",
        encoding="utf-8",
    )
    helper = tmp_path / "claim_lane_dispatch.py"
    helper.write_text("# canonical\n", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        "--journal-path", str(journal),
        "--canonical-helper-script", str(helper),
        "--claims-path", str(tmp_path / "no_claims.md"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "env-var" in captured.err
    payload = json.loads(captured.out)
    assert payload["operator_authorized_mode"]["env_authorized"] is False
    # No candidate should self-authorize when env is missing.
    halt_events = payload["reports"][0]["halt_events"]
    assert all(not e["autopilot_authorized"] for e in halt_events)


def test_authorized_tag_constant_matches_directive():
    # The structured tag enforces the operator-set audit identifier.
    assert loop.AUTOPILOT_AUTHORIZED_TAG == "[autopilot-claude-le-5-dollar]"


def test_canonical_helper_script_relpath_is_claim_lane_dispatch():
    assert loop.CANONICAL_HELPER_SCRIPT_RELPATH == "tools/claim_lane_dispatch.py"


def test_env_var_name_stable():
    # Used by preflight + docs; if this string changes, every doc must update.
    assert loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR == "CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE"


def test_authorized_event_serialization_round_trip(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand("a", cost_usd=2.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
    )
    report = loop.LoopIterationReport(
        iteration=1,
        started_at_utc="x",
        ended_at_utc="y",
        n_candidates_seen=1,
        n_candidates_blocked_by_dispatch_claim=0,
        n_candidates_ranked=1,
        halt_events=[e],
    )
    serialized = loop.serialize_report(report)
    assert serialized["halt_events"][0]["autopilot_authorized"] is True
    assert serialized["halt_events"][0]["autopilot_tag"] == loop.AUTOPILOT_AUTHORIZED_TAG
    # JSON round-trip:
    assert json.loads(json.dumps(serialized))["halt_events"][0]["autopilot_authorized"] is True
