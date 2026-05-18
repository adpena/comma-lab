# SPDX-License-Identifier: MIT
"""Tests for tools/cathedral_autopilot_autonomous_loop.py.

Covers:
  - rank_candidates by eig_per_dollar (descending)
  - rank_candidates by predicted_score_delta (most-negative first)
  - rank_candidates rejects unknown axis
  - malformed dispatch costs fail closed before ranking
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

import hashlib
import json
import math
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cathedral_autopilot_autonomous_loop as loop  # noqa: E402

from tac.optimizer.exact_readiness import runtime_dependency_manifest  # noqa: E402

TEST_JOURNAL_ROOT = (
    loop.REPO_ROOT / ".omx" / "state" / "pytest_cathedral_autopilot_journals"
)


@pytest.fixture(autouse=True)
def _cleanup_repo_local_test_journals():
    shutil.rmtree(TEST_JOURNAL_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(TEST_JOURNAL_ROOT, ignore_errors=True)


def _sha(seed: str) -> str:
    return (seed * 64)[:64]


def _write_exact_ready_archive(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _exact_ready_cand(
    tmp_path: Path,
    cid: str = "exact",
    *,
    cost_usd: float = 1.0,
) -> loop.CandidateRow:
    submission = tmp_path / f"submission_{cid}"
    archive_bytes, archive_sha = _write_exact_ready_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("cathedral exact custody fixture\n", encoding="utf-8")
    manifest = submission / "archive_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": archive_sha,
                "candidate_archive_bytes": archive_bytes,
                "candidate_archive": {"member_name": "0.bin"},
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_sha = runtime_dependency_manifest(submission, loop.REPO_ROOT)[
        "runtime_tree_sha256"
    ]
    return _cand(
        cid,
        cost_usd=cost_usd,
        lane_id=f"lane_{tmp_path.name}_{cid}",
        dispatch_packet_sha256="",
        archive_sha256=archive_sha,
        runtime_tree_sha256=runtime_sha,
        archive_path=str(submission / "archive.zip"),
        submission_dir=str(submission),
        archive_manifest_path=str(manifest),
        candidate_archive_bytes=archive_bytes,
        deployment_target="t4_contest_runtime",
        score_affecting_payload_changed=True,
        charged_bits_changed=True,
    )


def _repo_local_journal(tmp_path: Path, filename: str = "autopilot_journal.jsonl") -> Path:
    return TEST_JOURNAL_ROOT / tmp_path.name / filename


def _cand(cid: str = "c1", *, family: str = "hnerv_lc_v2",
          predicted_delta: float = -0.005,
          eig: float = 0.5,
          cost_usd: float = 5.0,
          blockers: list[str] | None = None,
          dispatch_packet_ready: bool = True,
          lane_id: str | None = None,
          target_modes: list[str] | None = None,
          dispatch_packet_sha256: str | None = None,
          archive_path: str = "",
          submission_dir: str = "",
          archive_manifest_path: str = "",
          archive_sha256: str | None = None,
          candidate_archive_bytes: int | None = None,
          runtime_tree_sha256: str | None = None,
          deployment_target: str = "",
          score_affecting_payload_changed: bool = False,
          charged_bits_changed: bool = False,
          ready_for_exact_eval_dispatch: bool = True) -> loop.CandidateRow:
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
            else _sha("a")
        ),
        archive_path=archive_path,
        submission_dir=submission_dir,
        archive_manifest_path=archive_manifest_path,
        archive_sha256=archive_sha256 if archive_sha256 is not None else _sha("b"),
        candidate_archive_bytes=candidate_archive_bytes,
        runtime_tree_sha256=(
            runtime_tree_sha256 if runtime_tree_sha256 is not None else _sha("c")
        ),
        deployment_target=deployment_target,
        score_affecting_payload_changed=score_affecting_payload_changed,
        charged_bits_changed=charged_bits_changed,
        ready_for_exact_eval_dispatch=ready_for_exact_eval_dispatch,
    )


def _claim_row(
    *,
    timestamp: str = "2026-05-16T00:00:00Z",
    lane_id: str,
    job_id: str = "job_1",
    status: str = "active_dispatch",
    notes: str = "",
) -> str:
    return (
        f"| {timestamp} | test-agent | {lane_id} | modal | {job_id} |  | "
        f"{status} | {notes} |"
    )


def _claims_text(*rows: str) -> str:
    return "\n".join([
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ])


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


def test_rank_by_predicted_score_delta_neutralizes_suppressed_prediction_band():
    blocked = _cand(
        "blocked_l5",
        predicted_delta=-0.050,
        blockers=["prediction_band_rank_reward_suppressed"],
    )
    clean = _cand("clean_small", predicted_delta=-0.001)

    ranked = loop.rank_candidates(
        [blocked, clean],
        rank_axis="predicted_score_delta",
    )

    assert [c.candidate_id for c in ranked] == ["clean_small", "blocked_l5"]


def test_rank_unknown_axis_raises():
    with pytest.raises(ValueError, match="rank_axis"):
        loop.rank_candidates([], rank_axis="something_made_up")


def test_eig_per_dollar_sorts_zero_cost_planning_rows_last():
    c = _cand(eig=1.0, cost_usd=0.0, blockers=["cost_estimation_required"])
    assert c.eig_per_dollar() == 0.0


@pytest.mark.parametrize("bad_cost", [-1.0, math.nan, math.inf, -math.inf])
def test_eig_per_dollar_refuses_malformed_cost(bad_cost):
    c = _cand(eig=1.0, cost_usd=bad_cost)
    with pytest.raises(ValueError, match="finite nonnegative estimated_dispatch_cost_usd"):
        c.eig_per_dollar()


def test_rank_candidates_keeps_zero_cost_planning_rows_visible():
    zero = _cand("zero", eig=1.0, cost_usd=0.0, blockers=["cost_estimation_required"])
    priced = _cand("priced", eig=0.1, cost_usd=1.0)

    ranked = loop.rank_candidates([zero, priced], rank_axis="eig_per_dollar")

    assert [candidate.candidate_id for candidate in ranked] == ["priced", "zero"]


@pytest.mark.parametrize("bad_cost", [-1.0, math.nan, math.inf, -math.inf])
def test_rank_candidates_refuses_malformed_cost_before_sort(bad_cost):
    c = _cand(eig=1.0, cost_usd=bad_cost)
    with pytest.raises(ValueError, match="finite nonnegative estimated_dispatch_cost_usd"):
        loop.rank_candidates([c], rank_axis="eig_per_dollar")


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
    p.write_text(_claims_text(_claim_row(lane_id="c1")), encoding="utf-8")
    has, reason = loop.check_dispatch_claim_conflict("c1", claims_path=p)
    assert has is True
    assert "c1" in reason


def test_check_dispatch_claim_present_by_lane_id_returns_true(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(
        _claims_text(_claim_row(lane_id="lane_exact_eval_a")),
        encoding="utf-8",
    )
    has, reason = loop.check_dispatch_claim_conflict(
        "candidate_a", claim_keys=["lane_exact_eval_a"], claims_path=p
    )
    assert has is True
    assert "lane_exact_eval_a" in reason


def test_check_dispatch_claim_absent_in_file_returns_false(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(_claims_text(_claim_row(lane_id="c2")), encoding="utf-8")
    has, _ = loop.check_dispatch_claim_conflict("c1", claims_path=p)
    assert has is False


def test_check_dispatch_claim_terminal_row_does_not_block(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(
        _claims_text(
            _claim_row(
                timestamp="2026-05-16T00:00:00Z",
                lane_id="lane_done",
                job_id="job_done",
                status="active_dispatch",
            ),
            _claim_row(
                timestamp="2026-05-16T00:10:00Z",
                lane_id="lane_done",
                job_id="job_done",
                status="completed_modal_auth_eval",
            ),
        ),
        encoding="utf-8",
    )
    has, reason = loop.check_dispatch_claim_conflict(
        "candidate_done",
        claim_keys=["lane_done"],
        claims_path=p,
        now_utc=loop.dt.datetime(2026, 5, 16, 1, 0, tzinfo=loop.dt.UTC),
    )
    assert has is False
    assert reason == ""


def test_check_dispatch_claim_exact_lane_id_matching_only(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(
        _claims_text(
            _claim_row(
                lane_id="lane_exact_eval_a_extra",
                notes="mentions lane_exact_eval_a in notes only",
            )
        ),
        encoding="utf-8",
    )
    has, reason = loop.check_dispatch_claim_conflict(
        "candidate_a", claim_keys=["lane_exact_eval_a"], claims_path=p
    )
    assert has is False
    assert reason == ""


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


def test_one_iteration_surfaces_zero_cost_blocked_planning_row(tmp_path):
    cands = [_cand("zero", cost_usd=0.0, blockers=["cost_estimation_required"])]

    rep = loop.run_one_loop_iteration(
        cands,
        claims_path=tmp_path / "no_claims.md",
    )

    assert rep.n_candidates_seen == 1
    assert rep.n_candidates_ranked == 1
    assert rep.halt_events[0].candidate_id == "zero"
    assert "cost_estimation_required" in rep.halt_events[0].blockers


def test_one_iteration_blocks_conflicted_candidate(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(_claims_text(_claim_row(lane_id="a")), encoding="utf-8")
    cands = [_cand("a"), _cand("b")]
    rep = loop.run_one_loop_iteration(cands, claims_path=p)
    assert rep.n_candidates_blocked_by_dispatch_claim == 1
    assert rep.n_candidates_ranked == 1
    assert all(e.candidate_id != "a" for e in rep.halt_events)


def test_one_iteration_blocks_conflicted_lane_id(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(_claims_text(_claim_row(lane_id="lane_a")), encoding="utf-8")
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


def test_one_iteration_race_mode_applies_prediction_band_suppression_before_trim(
    tmp_path,
):
    cands = [
        _cand(
            "blocked_l5",
            predicted_delta=-0.050,
            cost_usd=0.1,
            blockers=["prediction_band_rank_reward_suppressed"],
        ),
        _cand("clean_small", predicted_delta=-0.001, cost_usd=2.0),
    ]

    rep = loop.run_one_loop_iteration(
        cands,
        race_mode=True,
        claims_path=tmp_path / "n.md",
    )

    assert rep.n_candidates_seen == 1
    assert rep.n_candidates_ranked == 1
    assert {e.candidate_id for e in rep.halt_events} == {"clean_small"}


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


def test_load_candidates_from_jsonl_suppresses_legacy_prediction_rank_reward(tmp_path):
    p = tmp_path / "legacy_prediction.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "lane_time_traveler_l5_autonomy_substrate_20260513",
            "family": "time_traveler_l5_packet",
            "predicted_score_delta": -0.040,
            "expected_information_gain": 4.5,
            "estimated_dispatch_cost_usd": 4.50,
            "blockers": ["modal_a100_dispatch_smoke_before_full_pending"],
            "notes": "[prediction; time_traveler band [0.150, 0.170]]",
        })
        + "\n",
        encoding="utf-8",
    )

    rows = loop.load_candidates_from_jsonl(p)

    assert rows[0].expected_information_gain == 0.0
    assert "prediction_band_rank_reward_suppressed" in rows[0].blockers
    assert "prediction_band_rank_reward_suppressed" in rows[0].notes


def test_load_candidates_from_jsonl_preserves_valid_prediction_band_rank_reward(tmp_path):
    p = tmp_path / "valid_prediction.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "rank_valid",
            "family": "exact_anchor_family",
            "predicted_score_delta": -0.010,
            "expected_information_gain": 0.7,
            "estimated_dispatch_cost_usd": 10.0,
            "notes": "[prediction; structured band custody present]",
            "prediction_band_verdict": {"valid_for_rank_reward": True},
        })
        + "\n",
        encoding="utf-8",
    )

    rows = loop.load_candidates_from_jsonl(p)

    assert rows[0].expected_information_gain == 0.7
    assert "prediction_band_rank_reward_suppressed" not in rows[0].blockers


def test_load_candidates_from_jsonl_blocks_unscoped_literature_rank_reward(tmp_path):
    p = tmp_path / "unscoped_literature.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "unscoped_lit",
            "family": "predictive_coding",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 1.0,
            "lane_class": "substrate_class_shift predictive_receiver",
            "literature_anchor": "Rao-Ballard predictive coding",
        })
        + "\n",
        encoding="utf-8",
    )

    row = loop.load_candidates_from_jsonl(p)[0]

    assert any(
        blocker.startswith("literature_anchor_source_scope_missing:")
        for blocker in row.blockers
    )
    assert loop.apply_z1_empirical_revision_to_candidate_delta(row) == pytest.approx(
        -0.005
    )


def test_load_candidates_from_jsonl_preserves_scoped_literature_rank_reward(tmp_path):
    p = tmp_path / "scoped_literature.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "scoped_lit",
            "family": "predictive_coding",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 1.0,
            "lane_class": "substrate_class_shift predictive_receiver",
            "literature_anchor": "Rao-Ballard predictive coding",
            "source_supports": "Predictive error feedback as a neuroscience model.",
            "paper_claim_scope": "Analogy for planning, not Pact score evidence.",
            "pact_must_prove": "Byte-closed paired CPU/CUDA exact eval.",
            "decode_complexity_evidence": "T4 timing smoke and runtime custody.",
        })
        + "\n",
        encoding="utf-8",
    )

    row = loop.load_candidates_from_jsonl(p)[0]

    assert not any(
        blocker.startswith("literature_anchor_source_scope_missing:")
        for blocker in row.blockers
    )
    assert loop.apply_z1_empirical_revision_to_candidate_delta(row) == pytest.approx(
        -0.035
    )


def test_load_candidates_from_jsonl_requires_literal_true_prediction_rank_reward(
    tmp_path,
):
    p = tmp_path / "string_false_prediction.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "rank_string_false",
            "family": "exact_anchor_family",
            "predicted_score_delta": -0.010,
            "expected_information_gain": 0.7,
            "estimated_dispatch_cost_usd": 10.0,
            "notes": "[prediction; malformed structured band verdict]",
            "prediction_band_verdict": {"valid_for_rank_reward": "false"},
        })
        + "\n",
        encoding="utf-8",
    )

    rows = loop.load_candidates_from_jsonl(p)

    assert rows[0].expected_information_gain == 0.0
    assert "prediction_band_rank_reward_suppressed" in rows[0].blockers
    assert "prediction_band_rank_reward_suppressed" in rows[0].notes


def test_load_candidates_from_jsonl_carries_timing_smoke_to_halt_event(tmp_path):
    p = tmp_path / "timing_smoke.jsonl"
    command = ".venv/bin/python tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --epochs 1"
    p.write_text(
        json.dumps({
            "candidate_id": "timed",
            "family": "time_traveler_l5_packet",
            "predicted_score_delta": -0.010,
            "expected_information_gain": 0.0,
            "estimated_dispatch_cost_usd": 5.0,
            "timing_smoke_command": command,
        })
        + "\n",
        encoding="utf-8",
    )

    candidate = loop.load_candidates_from_jsonl(p)[0]
    event = loop.make_dispatch_halt_event(
        candidate,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
    )
    payload = loop.serialize_report(
        loop.LoopIterationReport(
            iteration=1,
            started_at_utc="1970-01-01T00:00:00Z",
            ended_at_utc="1970-01-01T00:00:01Z",
            n_candidates_seen=1,
            n_candidates_blocked_by_dispatch_claim=0,
            n_candidates_ranked=1,
            halt_events=[event],
        )
    )

    assert candidate.timing_smoke_command == command
    assert event.timing_smoke_command == command
    assert payload["halt_events"][0]["timing_smoke_command"] == command


@pytest.mark.parametrize(
    "flag",
    ["score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"],
)
def test_load_candidates_from_jsonl_refuses_authority_flags(tmp_path, flag):
    p = tmp_path / "cands.jsonl"
    raw = {
        "candidate_id": "authority_row",
        "family": "hnerv_lc_v2",
        "predicted_score_delta": -0.005,
        "expected_information_gain": 0.5,
        "estimated_dispatch_cost_usd": 5.0,
        flag: True,
    }
    p.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{flag}=True"):
        loop.load_candidates_from_jsonl(p)


def test_load_candidates_from_jsonl_refuses_string_authority_bool(tmp_path):
    p = tmp_path / "cands.jsonl"
    raw = {
        "candidate_id": "authority_row",
        "family": "hnerv_lc_v2",
        "predicted_score_delta": -0.005,
        "expected_information_gain": 0.5,
        "estimated_dispatch_cost_usd": 5.0,
        "dispatch_packet_ready": "false",
    }
    p.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-boolean dispatch_packet_ready"):
        loop.load_candidates_from_jsonl(p)


@pytest.mark.parametrize("field", ["license_ok", "sideinfo_consumed", "exact_duplicate"])
def test_load_candidates_from_jsonl_refuses_string_review_bools(tmp_path, field):
    p = tmp_path / "cands.jsonl"
    raw = {
        "candidate_id": "review_bool_row",
        "family": "hnerv_lc_v2",
        "predicted_score_delta": -0.005,
        "expected_information_gain": 0.5,
        "estimated_dispatch_cost_usd": 5.0,
        field: "false",
    }
    p.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"non-boolean {field}"):
        loop.load_candidates_from_jsonl(p)


def test_load_exact_ready_queue_preserves_authority_after_live_audit(
    tmp_path,
    monkeypatch,
):
    p = tmp_path / "exact_ready_queue.json"
    raw = dict(_exact_ready_cand(tmp_path, "exact_ready_row").__dict__)
    p.write_text(
        json.dumps({
            "schema": loop.EXACT_READY_QUEUE_SCHEMA,
            "dispatch_ready": [raw],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loop,
        "_audit_exact_ready_queue",
        lambda *args, **kwargs: {"stale_ready_rows": []},
    )

    rows = loop.load_candidates_from_exact_ready_queue(p)

    assert rows[0].ready_for_exact_eval_dispatch is True
    assert rows[0].dispatch_packet_ready is True
    assert "dispatch_packet_ready_false" not in rows[0].dispatch_authority_blockers()
    assert rows[0].score_claim is False
    assert rows[0].promotion_eligible is False
    auth = _auth_mode(tmp_path)
    ok, reason = auth.can_authorize(rows[0])
    assert ok, reason


def test_load_exact_ready_queue_refuses_string_authority_bools(
    tmp_path,
    monkeypatch,
):
    p = tmp_path / "exact_ready_queue.json"
    raw = {
        "candidate_id": "exact_ready_row",
        "family": "hnerv_lc_v2",
        "predicted_score_delta": -0.005,
        "expected_information_gain": 0.5,
        "estimated_dispatch_cost_usd": 5.0,
        "dispatch_packet_ready": "false",
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": _sha("a"),
        "runtime_tree_sha256": _sha("b"),
        "target_modes": [loop.AUTOPILOT_CONTEST_TARGET_MODE],
    }
    p.write_text(
        json.dumps({
            "schema": loop.EXACT_READY_QUEUE_SCHEMA,
            "dispatch_ready": [raw],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loop,
        "_audit_exact_ready_queue",
        lambda *args, **kwargs: {"stale_ready_rows": []},
    )

    with pytest.raises(ValueError, match="non-boolean dispatch_packet_ready"):
        loop.load_candidates_from_exact_ready_queue(p)


@pytest.mark.parametrize("bad_cost", ["NaN", "Infinity", "-Infinity", "0"])
def test_load_candidates_from_jsonl_refuses_non_finite_or_nonpositive_cost(
    tmp_path,
    bad_cost,
):
    p = tmp_path / "cands.jsonl"
    raw = {
        "candidate_id": "bad_cost",
        "family": "hnerv_lc_v2",
        "predicted_score_delta": -0.005,
        "expected_information_gain": 0.5,
        "estimated_dispatch_cost_usd": bad_cost,
    }
    p.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite positive estimated_dispatch_cost_usd"):
        loop.load_candidates_from_jsonl(p)


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
                "estimated_dispatch_cost_usd": 0.01,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "mdl_tier_c_density": 0.25,
                "composition_alpha": 0.8,
                "blockers": ["byte_closed_codec_candidate_required_before_dispatch"],
                "literature_anchor": "Rao-Ballard predictive coding",
                "source_supports": "Paper supports predictive error feedback.",
                "paper_claim_scope": "Analogy for L5 planning, not Pact evidence.",
                "pact_must_prove": "Paired exact CPU/CUDA byte-closed archive.",
                "decode_complexity_evidence": "T4 timing smoke required.",
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
    assert row.literature_anchor == "Rao-Ballard predictive coding"
    assert row.source_supports == "Paper supports predictive error feedback."
    assert row.paper_claim_scope == "Analogy for L5 planning, not Pact evidence."
    assert row.pact_must_prove == "Paired exact CPU/CUDA byte-closed archive."
    assert row.decode_complexity_evidence == "T4 timing smoke required."
    assert "[probe-disambiguator; read-only planning]" in row.notes


def test_load_probe_disambiguator_blocks_unscoped_literature_rank_reward(tmp_path):
    p = _write_probe_payload(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw = payload["autopilot_rows"][0]
    raw.pop("source_supports")
    raw.pop("paper_claim_scope")
    raw.pop("pact_must_prove")
    raw.pop("decode_complexity_evidence")
    raw.pop("mdl_tier_c_density")
    raw.pop("composition_alpha")
    raw["lane_class"] = "substrate_class_shift predictive_receiver"
    p.write_text(json.dumps(payload), encoding="utf-8")

    row = loop.load_candidates_from_probe_disambiguator_output(p)[0]

    assert any(
        blocker.startswith(f"{loop.LITERATURE_SOURCE_SCOPE_BLOCKER_PREFIX}:")
        for blocker in row.blockers
    )
    assert loop.LITERATURE_SOURCE_SCOPE_BLOCKER_PREFIX in row.notes
    assert loop.apply_z1_empirical_revision_to_candidate_delta(row) == pytest.approx(
        0.0
    )


def test_load_probe_disambiguator_suppresses_prediction_band_rank_reward(tmp_path):
    p = _write_probe_payload(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["autopilot_rows"][0]["prediction_band"] = {
        "band_id": "probe-band",
        "subject_id": "lane_zen_floor_probe_disambiguator_20260514",
    }
    payload["autopilot_rows"][0]["expected_information_gain"] = 9.0
    p.write_text(json.dumps(payload), encoding="utf-8")

    rows = loop.load_candidates_from_probe_disambiguator_output(p)

    assert rows[0].expected_information_gain == 0.0
    assert "prediction_band_rank_reward_suppressed" in rows[0].blockers
    assert "prediction_band_rank_reward_suppressed" in rows[0].notes


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


@pytest.mark.parametrize("bad_value", ["false", "", 0, 1])
def test_load_probe_disambiguator_refuses_malformed_top_level_dispatch_attempted(
    tmp_path,
    bad_value,
):
    p = _write_probe_payload(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["dispatch_attempted"] = bad_value
    p.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-boolean dispatch_attempted"):
        loop.load_candidates_from_probe_disambiguator_output(p)


@pytest.mark.parametrize("bad_value", ["false", "", 0, 1])
def test_load_probe_disambiguator_refuses_malformed_row_dispatch_attempted(
    tmp_path,
    bad_value,
):
    p = _write_probe_payload(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["autopilot_rows"][0]["dispatch_attempted"] = bad_value
    p.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-boolean dispatch_attempted"):
        loop.load_candidates_from_probe_disambiguator_output(p)


@pytest.mark.parametrize("field", ["license_ok", "sideinfo_consumed", "exact_duplicate"])
def test_load_probe_disambiguator_refuses_string_review_bools(tmp_path, field):
    p = _write_probe_payload(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["autopilot_rows"][0][field] = "false"
    p.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"non-boolean {field}"):
        loop.load_candidates_from_probe_disambiguator_output(p)


def test_load_substrate_composition_alpha_index_absent_file_is_empty(tmp_path):
    assert loop.load_substrate_composition_alpha_index(
        tmp_path / "missing_matrix.json"
    ) == {}


def test_load_substrate_composition_alpha_index_invalid_json_fails_closed(tmp_path):
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    matrix_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        loop.load_substrate_composition_alpha_index(matrix_path)


def test_load_substrate_composition_alpha_index_bad_alpha_fails_closed(tmp_path):
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "entries": {
                    "a__x__b": [
                        {
                            "written_at_utc": "2026-05-16T00:00:00Z",
                            "alpha": "not-a-number",
                            "score_claim": False,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-numeric alpha"):
        loop.load_substrate_composition_alpha_index(matrix_path)


def test_load_substrate_composition_alpha_index_string_score_claim_fails_closed(
    tmp_path,
):
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "entries": {
                    "a__x__b": [
                        {
                            "written_at_utc": "2026-05-16T00:00:00Z",
                            "alpha": 0.5,
                            "score_claim": "false",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-boolean score_claim"):
        loop.load_substrate_composition_alpha_index(matrix_path)


def test_substrate_composition_matrix_preserves_zero_alpha(tmp_path):
    """alpha=0.0 is a real saturation signal, not a missing value."""
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "entries": {
                    "a__x__b": [
                        {
                            "written_at_utc": "2026-05-16T00:00:00Z",
                            "alpha": 0.0,
                            "score_claim": False,
                        }
                    ],
                    "b__x__a": [
                        {
                            "written_at_utc": "2026-05-16T00:01:00Z",
                            "alpha": 0.9,
                            "score_claim": False,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    candidate = _cand("candidate_ab")

    loop.apply_substrate_composition_matrix_to_candidates(
        [candidate],
        substrate_ids_by_candidate={"candidate_ab": ("a", "b")},
        matrix_path=matrix_path,
    )

    assert candidate.composition_alpha == pytest.approx(0.0)


def test_substrate_composition_matrix_fallback_matches_exact_pair_ids(tmp_path):
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "entries": {
                    "aa__x__b": [
                        {
                            "written_at_utc": "2026-05-16T00:00:00Z",
                            "alpha": 0.2,
                            "score_claim": False,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    candidate_ab = _cand("candidate_ab")
    candidate_aab = _cand("candidate_aab")

    loop.apply_substrate_composition_matrix_to_candidates(
        [candidate_ab, candidate_aab],
        substrate_ids_by_candidate={
            "candidate_ab": ("a", "b"),
            "candidate_aab": ("aa", "b"),
        },
        matrix_path=matrix_path,
    )

    assert candidate_ab.composition_alpha is None
    assert candidate_aab.composition_alpha == pytest.approx(0.2)


def test_load_substrate_composition_ranking_requires_schema(tmp_path):
    p = tmp_path / "ranking.json"
    p.write_text(
        json.dumps(
            {
                "matrix_schema": "tac_autopilot_dispatch_ranking_v1",
                "score_claim": False,
                "ranked_dispatches": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema unsupported"):
        loop.load_candidates_from_substrate_composition_ranking(p)


def _write_substrate_ranking_payload(tmp_path: Path, row: dict | None = None) -> Path:
    p = tmp_path / "ranking.json"
    base_row = {
        "candidate_id": "singleton__time_traveler_l5_autonomy",
        "family": "renderer_replacement",
        "substrate_ids": ["time_traveler_l5_autonomy"],
        "predicted_score_delta": -0.020,
        "expected_information_gain": 0.0,
        "estimated_dispatch_cost_usd": 5.0,
        "composition_notes": "planning row",
        "fits_per_dispatch_cap": True,
        "fits_cumulative_envelope": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "campaign_metadata": [
            "lane_id=lane_time_traveler_l5_autonomy_substrate_20260513",
            "campaign_id=campaign_time_traveler_l5_v2_staircase_20260516",
        ],
        "mdl_tier_c_density": 0.25,
        "predicted_dispatch_risk": 12.5,
    }
    if row:
        base_row.update(row)
    p.write_text(
        json.dumps(
            {
                "schema": loop.SUBSTRATE_COMPOSITION_RANKING_SCHEMA,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "ranked_dispatches": [base_row],
            }
        ),
        encoding="utf-8",
    )
    return p


def test_ranking_loader_promotes_campaign_lane_id_to_claim_keys(tmp_path):
    p = _write_substrate_ranking_payload(tmp_path)

    rows = loop.load_candidates_from_substrate_composition_ranking(p)

    assert len(rows) == 1
    row = rows[0]
    assert row.lane_id == "lane_time_traveler_l5_autonomy_substrate_20260513"
    assert "lane_time_traveler_l5_autonomy_substrate_20260513" in row.claim_keys
    assert row.mdl_tier_c_density == pytest.approx(0.25)
    assert row.predicted_dispatch_risk == pytest.approx(12.5)


def test_ranking_loader_blocks_active_lane_claim_by_campaign_lane_id(tmp_path):
    ranking_path = _write_substrate_ranking_payload(tmp_path)
    claims_path = tmp_path / "claims.md"
    lane_id = "lane_time_traveler_l5_autonomy_substrate_20260513"
    claims_path.write_text(_claims_text(_claim_row(lane_id=lane_id)), encoding="utf-8")
    rows = loop.load_candidates_from_substrate_composition_ranking(ranking_path)

    report = loop.run_one_loop_iteration(rows, claims_path=claims_path)

    assert report.n_candidates_blocked_by_dispatch_claim == 1
    assert report.n_candidates_ranked == 0
    assert any(lane_id in note for note in report.notes)


def test_candidate_substrate_ids_from_ranking_requires_schema(tmp_path):
    p = tmp_path / "ranking.json"
    p.write_text(
        json.dumps(
            {
                "matrix_schema": "tac_autopilot_dispatch_ranking_v1",
                "ranked_dispatches": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema unsupported"):
        loop.candidate_substrate_ids_from_ranking(p)


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


def test_main_surfaces_compressive_sensing_lattice_diagnostic(tmp_path, capsys):
    cands_path = tmp_path / "cands.jsonl"
    cands_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "candidate_id": f"c{i}",
                    "family": "l5_v2",
                    "predicted_score_delta": -0.01 * (i + 1),
                    "expected_information_gain": 0.5,
                    "estimated_dispatch_cost_usd": 1.0,
                }
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    rc = loop.main([
        "--candidates-jsonl", str(cands_path),
        "--iterations", "1",
        "--claims-path", str(tmp_path / "no_claims.md"),
        "--use-compressive-sensing-lattice",
        "--lattice-anchor-count", "2",
        "--lattice-expected-sparsity", "2",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    lattice = payload["compressive_sensing_lattice"]
    assert lattice["enabled"] is True
    assert lattice["score_claim"] is False
    assert lattice["ready_for_exact_eval_dispatch"] is False
    assert lattice["diagnostic"]["K"] == 2
    assert lattice["diagnostic"]["N"] == 6
    assert "recovery_regime" in lattice["diagnostic"]
    assert lattice["dispatch_blocker"].startswith(
        "compressive_sensing_lattice_recovery_regime_"
    )
    blocker = lattice["dispatch_blocker"]
    halt_blockers = payload["reports"][0]["halt_events"][0]["blockers"]
    assert blocker in halt_blockers


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
    helper = TOOLS_DIR / "claim_lane_dispatch.py"
    if not helper_exists:
        helper = tmp_path / "missing_claim_lane_dispatch.py"
    journal = _repo_local_journal(tmp_path)
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
    assert "not finite-positive" in reason


@pytest.mark.parametrize("bad_cost", [math.nan, math.inf, -math.inf])
def test_can_authorize_refuses_non_finite_cost(tmp_path, bad_cost):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=bad_cost)
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "not finite-positive" in reason


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


def test_can_authorize_refuses_malformed_dispatch_packet_hash(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(cost_usd=1.0, dispatch_packet_sha256="not-a-sha256")
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "dispatch_packet_sha256_malformed" in reason


def test_can_authorize_refuses_contest_exact_hash_only_without_archive_runtime(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256=_sha("a"),
        archive_sha256="",
        runtime_tree_sha256="",
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "contest_exact_eval_requires_archive_and_runtime_hash" in reason


def test_can_authorize_refuses_placeholder_dispatch_packet_hash_even_with_pair(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256="dispatch_packet_sha256_for_a",
        archive_sha256=_sha("b"),
        runtime_tree_sha256=_sha("c"),
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "dispatch_packet_sha256_malformed" in reason


def test_can_authorize_refuses_hash_pair_without_live_custody(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256="",
        archive_sha256=_sha("b"),
        runtime_tree_sha256=_sha("c"),
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "exact_dispatch_authority:archive_path_missing" in reason
    assert "exact_dispatch_authority:archive_bytes_missing_or_invalid" in reason
    assert "exact_dispatch_authority:score_affecting_change_proof_missing" in reason


def test_can_authorize_refuses_contest_exact_without_exact_ready_authority(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256="",
        archive_sha256=_sha("b"),
        runtime_tree_sha256=_sha("c"),
        ready_for_exact_eval_dispatch=False,
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "contest_exact_eval_requires_ready_for_exact_eval_dispatch" in reason


def test_can_authorize_refuses_malformed_archive_runtime_hash_pair(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256="",
        archive_sha256=_sha("b"),
        runtime_tree_sha256="runtime-tree-sha-placeholder",
    )
    c.ready_for_exact_eval_dispatch = True
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "runtime_tree_sha256_malformed" in reason
    assert "ready_for_exact_eval_dispatch_requires_archive_and_runtime_hash" in reason


def test_can_authorize_refuses_placeholder_archive_runtime_hash_pair(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _cand(
        cost_usd=1.0,
        dispatch_packet_sha256="",
        archive_sha256="archive_sha256_for_a",
        runtime_tree_sha256="runtime_tree_sha256_for_a",
    )
    ok, reason = cfg.can_authorize(c)
    assert ok is False
    assert "archive_sha256_malformed" in reason
    assert "runtime_tree_sha256_malformed" in reason
    assert "contest_exact_eval_requires_archive_and_runtime_hash" in reason


def test_can_authorize_approves_within_caps(tmp_path):
    cfg = _auth_mode(tmp_path)
    c = _exact_ready_cand(tmp_path, "within_caps", cost_usd=4.5)
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
    c = _exact_ready_cand(tmp_path, "authorized", cost_usd=3.0)
    claims_path = tmp_path / "claims.md"
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
        claims_path=claims_path,
    )
    assert e.autopilot_authorized is True
    assert e.autopilot_tag == loop.AUTOPILOT_AUTHORIZED_TAG
    assert e.requires_approval is False
    assert e.autopilot_claim_recorded is True
    assert e.autopilot_claim_instance_job_id
    assert cfg.cumulative_spent_usd == 3.0
    claim_text = claims_path.read_text(encoding="utf-8")
    assert c.lane_id in claim_text
    assert loop.AUTOPILOT_CLAIM_STATUS in claim_text


def test_make_dispatch_halt_event_refuses_tmp_journal_before_claim(tmp_path):
    cfg = _auth_mode(tmp_path)
    cfg.journal_path = (
        Path(tempfile.gettempdir())
        / f"cathedral_autopilot_tmp_journal_{tmp_path.name}.jsonl"
    )
    claims_path = tmp_path / "claims.md"
    e = loop.make_dispatch_halt_event(
        _cand("tmp_journal", cost_usd=3.0),
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
        claims_path=claims_path,
    )
    assert e.autopilot_authorized is False
    assert e.autopilot_claim_recorded is False
    assert e.requires_approval is True
    assert "refusing transient path" in e.autopilot_refused_reason
    assert "operator_authorized_mode_config_invalid" in e.blockers
    assert not claims_path.exists()


def test_make_dispatch_halt_event_self_authorization_requires_successful_claim(tmp_path):
    bad_helper = tmp_path / "bad_claim_helper.py"
    bad_helper.write_text(
        "import sys\nprint('claim failed intentionally', file=sys.stderr)\nsys.exit(3)\n",
        encoding="utf-8",
    )
    cfg = loop.OperatorAuthorizedModeConfig(
        enabled=True,
        per_dispatch_cap_usd=5.0,
        cumulative_cap_usd=20.0,
        canonical_helper_script=bad_helper,
        journal_path=_repo_local_journal(tmp_path, "bad_helper_journal.jsonl"),
    )
    c = _exact_ready_cand(tmp_path, "claim_required", cost_usd=3.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
        claims_path=tmp_path / "claims.md",
    )
    assert e.autopilot_authorized is False
    assert e.autopilot_claim_recorded is False
    assert e.requires_approval is True
    assert "dispatch claim is required" in e.autopilot_refused_reason
    assert "claim failed intentionally" in e.autopilot_refused_reason
    assert "dispatch_claim_required_for_self_authorization" in e.blockers
    assert cfg.cumulative_spent_usd == 0.0


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
        claims_path=tmp_path / "claims.md",
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
    c = _exact_ready_cand(tmp_path, "a", cost_usd=2.0)
    c.source_supports = "Paper supports the planning hypothesis only."
    c.paper_claim_scope = "Literature scope, not Pact score evidence."
    c.pact_must_prove = "Byte-closed contest eval."
    c.decode_complexity_evidence = "T4 timing smoke required."
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
    assert row["source_supports"].startswith("Paper supports")
    assert row["paper_claim_scope"].startswith("Literature scope")
    assert row["pact_must_prove"].startswith("Byte-closed")
    assert row["decode_complexity_evidence"].startswith("T4")
    assert row["archive_path"] == c.archive_path
    assert row["candidate_archive_bytes"] == c.candidate_archive_bytes


def test_loop_iteration_journal_appends_one_row_per_authorization(tmp_path):
    cfg = _auth_mode(tmp_path)
    cands = [
        _exact_ready_cand(tmp_path, "a", cost_usd=2.0),
        _exact_ready_cand(tmp_path, "b", cost_usd=3.0),
    ]
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


@pytest.mark.parametrize(
    ("per_dispatch_cap", "cumulative_cap"),
    [
        (math.nan, 20.0),
        (math.inf, 20.0),
        (5.0, math.nan),
        (5.0, math.inf),
    ],
)
def test_loop_iteration_refuses_non_finite_caps_before_claim_or_journal(
    tmp_path,
    per_dispatch_cap,
    cumulative_cap,
):
    cfg = _auth_mode(
        tmp_path,
        per_dispatch_cap=per_dispatch_cap,
        cumulative_cap=cumulative_cap,
    )
    claims_path = tmp_path / "claims.md"
    with pytest.raises(ValueError, match="finite positive"):
        loop.run_one_loop_iteration(
            [_cand("bad_cap", cost_usd=1.0)],
            claims_path=claims_path,
            auth_mode=cfg,
            env_authorized=True,
        )
    assert not claims_path.exists()
    assert not cfg.journal_path.exists()


def test_loop_iteration_refuses_malformed_cost_before_claim_or_journal(tmp_path):
    cfg = _auth_mode(tmp_path)
    claims_path = tmp_path / "claims.md"
    with pytest.raises(ValueError, match="finite nonnegative estimated_dispatch_cost_usd"):
        loop.run_one_loop_iteration(
            [_cand("bad_cost", cost_usd=math.nan)],
            claims_path=claims_path,
            auth_mode=cfg,
            env_authorized=True,
        )
    assert not claims_path.exists()
    assert not cfg.journal_path.exists()


def test_loop_iteration_cumulative_envelope_halts_excess(tmp_path):
    cfg = _auth_mode(tmp_path, cumulative_cap=10.0)
    cands = [
        _exact_ready_cand(tmp_path, "a", cost_usd=5.0),
        _exact_ready_cand(tmp_path, "b", cost_usd=4.0),
        _exact_ready_cand(tmp_path, "c", cost_usd=4.0),  # 5+4+4=13 > 10
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
    cand_file = tmp_path / "exact_ready_queue.json"
    row = dict(_exact_ready_cand(tmp_path, "test_auth_mode_uniq_abc123", cost_usd=2.0).__dict__)
    cand_file.write_text(
        json.dumps({
            "schema": loop.EXACT_READY_QUEUE_SCHEMA,
            "dispatch_ready": [row],
        }),
        encoding="utf-8",
    )
    helper = TOOLS_DIR / "claim_lane_dispatch.py"
    journal = _repo_local_journal(tmp_path, "main_journal.jsonl")
    claims_path = tmp_path / "claims.md"
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "1")
    monkeypatch.setattr(
        loop,
        "_audit_exact_ready_queue",
        lambda *args, **kwargs: {"stale_ready_rows": []},
    )
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        "--journal-path", str(journal),
        "--canonical-helper-script", str(helper),
        "--claims-path", str(claims_path),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_authorized_mode"]["enabled"] is True
    assert payload["operator_authorized_mode"]["env_authorized"] is True
    assert payload["operator_authorized_mode"]["cumulative_spent_usd"] == 2.0
    assert journal.is_file()
    assert row["lane_id"] in claims_path.read_text(encoding="utf-8")


def test_main_authorized_mode_refuses_tmp_journal_before_claim(
    tmp_path,
    monkeypatch,
    capsys,
):
    cand_file = tmp_path / "c.jsonl"
    cand_file.write_text(
        json.dumps({
            "candidate_id": "test_auth_mode_tmp_journal",
            "family": "hnerv",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 2.0,
            "dispatch_packet_ready": True,
            "dispatch_packet_sha256": _sha("e"),
            "lane_id": "lane_test_auth_mode_tmp_journal",
            "target_modes": [loop.AUTOPILOT_CONTEST_TARGET_MODE],
        }) + "\n",
        encoding="utf-8",
    )
    claims_path = tmp_path / "claims.md"
    journal = (
        Path(tempfile.gettempdir())
        / f"cathedral_autopilot_cli_tmp_journal_{tmp_path.name}.jsonl"
    )
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "1")
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        "--journal-path", str(journal),
        "--canonical-helper-script", str(TOOLS_DIR / "claim_lane_dispatch.py"),
        "--claims-path", str(claims_path),
    ])
    assert rc == 2
    assert "refusing transient path" in capsys.readouterr().err
    assert not claims_path.exists()


@pytest.mark.parametrize(
    ("cap_flag", "cap_value"),
    [
        ("--per-dispatch-cap-usd", "nan"),
        ("--per-dispatch-cap-usd", "inf"),
        ("--cumulative-cap-usd", "nan"),
        ("--cumulative-cap-usd", "inf"),
    ],
)
def test_main_authorized_mode_refuses_non_finite_caps_before_claim(
    tmp_path,
    monkeypatch,
    capsys,
    cap_flag,
    cap_value,
):
    cand_file = tmp_path / "c.jsonl"
    cand_file.write_text(
        json.dumps({
            "candidate_id": "test_auth_mode_bad_cap",
            "family": "hnerv",
            "predicted_score_delta": -0.005,
            "expected_information_gain": 0.5,
            "estimated_dispatch_cost_usd": 2.0,
            "dispatch_packet_ready": True,
            "dispatch_packet_sha256": _sha("f"),
            "lane_id": "lane_test_auth_mode_bad_cap",
            "target_modes": [loop.AUTOPILOT_CONTEST_TARGET_MODE],
        }) + "\n",
        encoding="utf-8",
    )
    claims_path = tmp_path / "claims.md"
    monkeypatch.setenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, "1")
    rc = loop.main([
        "--candidates-jsonl", str(cand_file),
        "--iterations", "1",
        "--operator-authorized-le-5-dollar-mode",
        "--journal-path", str(_repo_local_journal(tmp_path, "bad_cap_journal.jsonl")),
        "--canonical-helper-script", str(TOOLS_DIR / "claim_lane_dispatch.py"),
        "--claims-path", str(claims_path),
        cap_flag, cap_value,
    ])
    assert rc == 2
    assert "finite positive" in capsys.readouterr().err
    assert not claims_path.exists()


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
    journal = _repo_local_journal(tmp_path, "no_env_journal.jsonl")
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
    c = _exact_ready_cand(tmp_path, "a", cost_usd=2.0)
    e = loop.make_dispatch_halt_event(
        c,
        requires_approval_classes=frozenset({loop.EventClass.DISPATCH}),
        auth_mode=cfg,
        env_authorized=True,
        claims_path=tmp_path / "claims.md",
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
    assert serialized["halt_events"][0]["autopilot_claim_recorded"] is True
    # JSON round-trip:
    assert json.loads(json.dumps(serialized))["halt_events"][0]["autopilot_authorized"] is True


def test_master_gradient_rerank_uses_structured_archive_sha256(tmp_path):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    archive_sha = _sha("d")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    # The reranker does not load the tensor yet; a real path is still clearer.
    sidecar.write_bytes(b"not-loaded")
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("e"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=600,
            n_pairs_total=600,
            operating_point=OperatingPoint(
                d_seg=0.0005,
                d_pose=0.00003,
                rate=178_517 / 37_545_489,
                score=0.192,
            ),
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_modal_cpu",
            measurement_call_id="call-full",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand("not_a_hash", predicted_delta=-0.01, archive_sha256=archive_sha)

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cpu",
        ledger_path=ledger,
    )

    assert ranked[0][1] == pytest.approx(-0.01)
    assert "master-gradient-anchor-present" in ranked[0][2]
    assert "zip_inner_member_payload" in ranked[0][2]


def test_master_gradient_rerank_rejects_subset_anchor(tmp_path):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    archive_sha = _sha("f")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    sidecar.write_bytes(b"not-loaded")
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("a"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=8,
            n_pairs_total=600,
            operating_point=OperatingPoint(
                d_seg=0.0005,
                d_pose=0.00003,
                rate=178_517 / 37_545_489,
                score=0.192,
            ),
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_modal_cpu",
            measurement_call_id="call-subset",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand("not_a_hash", predicted_delta=-0.01, archive_sha256=archive_sha)

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cpu",
        ledger_path=ledger,
    )

    assert ranked[0][1] == pytest.approx(-0.01)
    assert "master-gradient-anchor-rejected" in ranked[0][2]
    assert "pair subset" in ranked[0][2]


def test_master_gradient_rerank_reports_advisory_correction_for_stale_contest_row(
    tmp_path,
):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    archive_sha = _sha("1")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    sidecar.write_bytes(b"not-loaded")
    op = OperatingPoint(
        d_seg=0.0005,
        d_pose=0.00003,
        rate=178_517 / 37_545_489,
        score=0.192,
    )
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("2"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=8,
            n_pairs_total=600,
            operating_point=op,
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_8pair_subset",
            measurement_axis="[contest-CPU]",
            measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
            measurement_call_id="call-stale",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("2"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=8,
            n_pairs_total=600,
            operating_point=op,
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_8pair_subset_axis_correction",
            measurement_axis="[macOS-CPU advisory]",
            measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
            measurement_call_id="call-correction",
            measurement_utc="2026-05-18T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand("not_a_hash", predicted_delta=-0.01, archive_sha256=archive_sha)

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cpu",
        ledger_path=ledger,
    )

    assert ranked[0][1] == pytest.approx(-0.01)
    assert "master-gradient-anchor-diagnostic-only" in ranked[0][2]
    assert "[macOS-CPU advisory]" in ranked[0][2]


def test_master_gradient_rerank_rejects_false_contest_cuda_mps_anchor(tmp_path):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    archive_sha = _sha("9")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    sidecar.write_bytes(b"not-loaded")
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("8"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=600,
            n_pairs_total=600,
            operating_point=OperatingPoint(
                d_seg=0.0005,
                d_pose=0.00003,
                rate=178_517 / 37_545_489,
                score=0.192,
            ),
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_full",
            measurement_axis="[contest-CUDA]",
            measurement_hardware="darwin_arm64_m5_max_macos_mps_advisory",
            measurement_call_id="call-mps",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand("not_a_hash", predicted_delta=-0.01, archive_sha256=archive_sha)

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cuda",
        ledger_path=ledger,
    )

    assert "master-gradient-anchor-rejected" in ranked[0][2]
    assert "[contest-CUDA]" in ranked[0][2]
    assert "advisory/local/proxy hardware" in ranked[0][2]


def test_master_gradient_rerank_reports_different_authoritative_axis(tmp_path):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    archive_sha = _sha("7")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    sidecar.write_bytes(b"not-loaded")
    append_anchor_locked(
        MasterGradient(
            archive_sha256=archive_sha,
            scored_archive_sha256=archive_sha,
            scored_archive_bytes=178_517,
            gradient_subject_sha256=_sha("6"),
            gradient_subject_bytes=178_417,
            gradient_byte_domain="zip_inner_member_payload",
            n_pairs_used=600,
            n_pairs_total=600,
            operating_point=OperatingPoint(
                d_seg=0.0005,
                d_pose=0.00003,
                rate=178_517 / 37_545_489,
                score=0.192,
            ),
            gradient_array_path=str(sidecar),
            n_bytes=178_417,
            measurement_method="autograd_per_parameter_projected_full",
            measurement_axis="[contest-CUDA]",
            measurement_hardware="linux_x86_64_cuda_t4",
            measurement_call_id="call-cuda",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand("not_a_hash", predicted_delta=-0.01, archive_sha256=archive_sha)

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cpu",
        ledger_path=ledger,
    )

    assert "master-gradient-anchor-different-authoritative-axis" in ranked[0][2]
    assert "[contest-CUDA]" in ranked[0][2]
    assert "[contest-CPU]" in ranked[0][2]


def test_master_gradient_rerank_does_not_scrape_notes_or_candidate_id(tmp_path):
    from tac.master_gradient import MasterGradient, OperatingPoint, append_anchor_locked

    anchor_sha = _sha("a")
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    sidecar = tmp_path / "grad.npy"
    sidecar.write_bytes(b"not-loaded")
    append_anchor_locked(
        MasterGradient(
            archive_sha256=anchor_sha,
            scored_archive_sha256=anchor_sha,
            scored_archive_bytes=100,
            operating_point=OperatingPoint(
                d_seg=0.0005,
                d_pose=0.00003,
                rate=100 / 37_545_489,
                score=0.192,
            ),
            gradient_array_path=str(sidecar),
            n_bytes=100,
            measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_modal_cpu",
            measurement_call_id="call",
            measurement_utc="2026-05-17T12:00:00+00:00",
        ),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    cand = _cand(anchor_sha, predicted_delta=-0.01, archive_sha256="")
    cand.notes = f"archive_sha256 {anchor_sha}"

    ranked = loop.rerank_candidates_via_master_gradient(
        [cand],
        panel_axis="contest_cpu",
        ledger_path=ledger,
    )

    assert "structured archive_sha256 missing or malformed" in ranked[0][2]


def _optimal_plan_payload_for_cathedral_loader(sha: str, delta: float) -> dict:
    return {
        "archive_sha256": sha,
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": 15,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "treatment_catalog_sha": "catsha",
        "predicted_score_delta": delta,
        "predicted_score_delta_confidence_interval": [delta - 0.001, delta + 0.001],
        "budget": {"compute_usd": 0.75},
        "kkt_residual": 0.002,
        "feasibility_certificate": {"archive_bytes": True},
        "is_pareto_feasible": True,
    }


def test_load_candidates_from_master_gradient_optimal_plans_latest_only(tmp_path):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "8" * 64
    (root / f"optimal_plan_{sha[:12]}_20260518T100000.json").write_text(
        json.dumps(_optimal_plan_payload_for_cathedral_loader(sha, -0.001)),
        encoding="utf-8",
    )
    (root / f"optimal_plan_{sha[:12]}_20260518T200000.json").write_text(
        json.dumps(_optimal_plan_payload_for_cathedral_loader(sha, -0.009)),
        encoding="utf-8",
    )
    rows = loop.load_candidates_from_master_gradient_optimal_plans(root=root)
    assert len(rows) == 1
    row = rows[0]
    assert row.archive_sha256 == sha
    assert row.predicted_score_delta == -0.009
    assert row.score_claim is False
    assert row.promotion_eligible is False
    assert row.ready_for_exact_eval_dispatch is False
    assert "planning_only_master_gradient_optimal_plan_no_dispatch_packet" in row.blockers


def test_per_pair_sister_817_sidecar_rewards_make_negative_delta_more_negative(
    tmp_path, monkeypatch
):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "3" * 64
    (root / f"per_pair_bit_allocation_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps({"cascade_path_used": "optimal_plan"}),
        encoding="utf-8",
    )
    (root / f"per_pair_fisher_importance_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps({"aggregate_fisher_l1": 3.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)

    adjusted = loop.adjust_predicted_delta_for_per_pair_sister_817_sidecars(-0.01, sha)

    assert adjusted == pytest.approx(-0.01 * 1.05 * 1.03)
    rewarded = _cand("rewarded", predicted_delta=-0.01, archive_sha256=sha)
    plain = _cand("plain", predicted_delta=-0.0105, archive_sha256="7" * 64)

    ranked = loop.rank_candidates([plain, rewarded], rank_axis="predicted_score_delta")

    assert [cand.candidate_id for cand in ranked] == ["rewarded", "plain"]


def test_per_pair_difficulty_atlas_sidecar_rewards_pose_hard_signal(tmp_path, monkeypatch):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "4" * 64
    payload = {
        "archive_sha256": sha,
        "schema": "master_gradient_consumer_per_pair_difficulty_v1",
        "consumer_id": "per_pair_difficulty_atlas",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "aggregate_gradient_norm_l2": 12.0,
        "top_k_hardest_with_axis_breakdown": [
            {
                "pair_index": 7,
                "gradient_norm_l2": 8.0,
                "seg_axis_l1": 1.0,
                "pose_axis_l1": 5.0,
                "rate_axis_l1": 0.1,
                "seg_axis_score_l1": 1.0,
                "pose_axis_score_l1": 5.0,
                "rate_axis_score_l1": 0.1,
            }
        ],
    }
    (root / f"per_pair_difficulty_atlas_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)

    adjusted = loop.adjust_predicted_delta_for_per_pair_difficulty_atlas(-0.01, sha)

    assert adjusted == pytest.approx(-0.01 * 1.06)


def test_per_pair_difficulty_atlas_raw_axis_only_gets_generic_reward(
    tmp_path, monkeypatch
):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "8" * 64
    (root / f"per_pair_difficulty_atlas_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "schema": "master_gradient_consumer_per_pair_difficulty_v1",
                "consumer_id": "per_pair_difficulty_atlas",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "aggregate_gradient_norm_l2": 12.0,
                "top_k_hardest_with_axis_breakdown": [
                    {"seg_axis_l1": 1.0, "pose_axis_l1": 5.0, "rate_axis_l1": 0.1}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)

    assert loop.adjust_predicted_delta_for_per_pair_difficulty_atlas(-0.01, sha) == pytest.approx(
        -0.01 * 1.04
    )


def test_per_pair_difficulty_atlas_malformed_or_authority_bearing_passthrough(
    tmp_path, monkeypatch
):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "5" * 64
    (root / f"per_pair_difficulty_atlas_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "schema": "master_gradient_consumer_per_pair_difficulty_v1",
                "consumer_id": "per_pair_difficulty_atlas",
                "score_claim": True,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "aggregate_gradient_norm_l2": 12.0,
                "top_k_hardest_with_axis_breakdown": [
                    {"seg_axis_l1": 1.0, "pose_axis_l1": 5.0, "rate_axis_l1": 0.1}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)

    assert loop.adjust_predicted_delta_for_per_pair_difficulty_atlas(-0.01, sha) == pytest.approx(-0.01)


def test_per_pair_difficulty_atlas_archive_mismatch_passthrough(tmp_path, monkeypatch):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "9" * 64
    (root / f"per_pair_difficulty_atlas_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "schema": "master_gradient_consumer_per_pair_difficulty_v1",
                "consumer_id": "per_pair_difficulty_atlas",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "aggregate_gradient_norm_l2": 12.0,
                "top_k_hardest_with_axis_breakdown": [
                    {
                        "seg_axis_score_l1": 1.0,
                        "pose_axis_score_l1": 5.0,
                        "rate_axis_score_l1": 0.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)

    assert loop.adjust_predicted_delta_for_per_pair_difficulty_atlas(-0.01, sha) == pytest.approx(-0.01)


def test_score_adjusted_for_all_signal_consumes_difficulty_atlas_after_existing_cascades(
    tmp_path, monkeypatch
):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "6" * 64
    (root / f"per_pair_difficulty_atlas_{sha[:12]}_20260518T191403.json").write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "schema": "master_gradient_consumer_per_pair_difficulty_v1",
                "consumer_id": "per_pair_difficulty_atlas",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "aggregate_gradient_norm_l2": 10.0,
                "top_k_hardest_with_axis_breakdown": [
                    {"seg_axis_l1": 5.0, "pose_axis_l1": 1.0, "rate_axis_l1": 0.1}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_PER_PAIR_SIDECAR_SCAN_ROOT", root)
    cand = _cand("difficulty_atlas", predicted_delta=-0.01, archive_sha256=sha)

    assert loop.apply_z1_empirical_revision_to_candidate_delta(cand) == pytest.approx(-0.01 * 1.04)


def test_report_only_appends_master_gradient_optimal_plan_rows(tmp_path, capsys):
    root = tmp_path / "master_gradient_consumers"
    root.mkdir()
    sha = "9" * 64
    (root / f"optimal_plan_{sha[:12]}_20260518T200000.json").write_text(
        json.dumps(_optimal_plan_payload_for_cathedral_loader(sha, -0.011)),
        encoding="utf-8",
    )
    candidates_jsonl = tmp_path / "candidates.jsonl"
    candidates_jsonl.write_text("", encoding="utf-8")
    out = tmp_path / "report.json"

    rc = loop.main([
        "--candidates-jsonl",
        str(candidates_jsonl),
        "--include-master-gradient-optimal-plans",
        "--master-gradient-optimal-plan-root",
        str(root),
        "--report-only",
        "--output",
        str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_candidates_total"] == 1
    assert payload["top_candidates"][0]["candidate_id"].startswith("optimal_plan_")
    assert payload["top_candidates"][0]["score_claim"] is False
    assert "planning_only_master_gradient_optimal_plan_no_dispatch_packet" in (
        payload["top_candidates"][0]["blockers"]
    )
    capsys.readouterr()
