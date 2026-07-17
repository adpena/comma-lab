"""Tests for the #513 Modal single-flight runtime guard + surface gate.

Two-landing structural enforcement of the operator binding 2026-07-15
(memory ``modal_single_flight_dual_ledger_policy_20260715``):

* runtime half — ``tac.deploy.modal.single_flight.assert_modal_single_flight``
  (pre-spawn refusal every Modal dispatch entry point calls) + the dual-ledger
  terminality blocker invoked from ``update_call_id_outcome``;
* static half — ``tac.preflight.check_modal_dispatch_single_flight`` (every
  dispatch surface with a real ``.spawn(`` call routes through the guard).

Sisters: test_modal_single_flight_ledger.py (ledger-STATE gate) +
test_claim_lane_dispatch_modal_single_flight.py (claim-time refusal rc=5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.deploy.modal import single_flight as sf
from tac.deploy.modal.single_flight import (
    ModalSingleFlightRefusal,
    active_modal_claims,
    assert_modal_single_flight,
    dual_ledger_terminality_blockers,
    emit_dual_ledger_terminality_blocker_if_needed,
    live_modal_call_rows,
    single_flight_findings,
)
from tac.preflight import PreflightError, check_modal_dispatch_single_flight

_CLAIMS_HEADER = (
    "# Active lane dispatch claims\n\n"
    "| timestamp_utc | agent | lane_id | platform | instance/job_id "
    "| predicted_eta_utc | status | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _write_ledger(root: Path, rows: list[dict]) -> Path:
    path = root / ".omx/state/modal_call_id_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _write_claims(root: Path, rows: list[tuple[str, ...]]) -> Path:
    path = root / ".omx/state/active_lane_dispatch_claims.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows)
    path.write_text(_CLAIMS_HEADER + body)
    return path


def _claim_row(
    lane: str, job: str, status: str, platform: str = "modal", notes: str = "-",
) -> tuple[str, ...]:
    return ("2026-07-17T00:00:00Z", "claude", lane, platform, job, "-", status, notes)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / ".omx/state").mkdir(parents=True)
    return tmp_path


# ── runtime guard: surfaces ─────────────────────────────────────────────────


def test_clear_ledgers_no_findings_no_raise(root):
    _write_ledger(root, [])
    _write_claims(root, [])
    assert assert_modal_single_flight(repo_root=root, check_cloud=False) == []


def test_missing_ledger_files_fail_open(root):
    # No ledger, no claims file at all -> zero findings (fail-open surfaces).
    assert single_flight_findings(repo_root=root, check_cloud=False) == []


def test_live_ledger_row_refuses(root):
    _write_ledger(root, [{
        "call_id": "fc-live1", "status": "dispatched", "label": "lblA",
        "lane_id": "laneA",
    }])
    with pytest.raises(ModalSingleFlightRefusal) as exc:
        assert_modal_single_flight(repo_root=root, check_cloud=False)
    assert "fc-live1" in str(exc.value)
    assert "2026-07-15" in str(exc.value)
    assert "reconcile" in str(exc.value)


def test_terminal_outcome_clears_live_ledger_row(root):
    _write_ledger(root, [
        {"call_id": "fc-1", "status": "dispatched"},
        {"call_id": "fc-1", "status": "harvested"},
    ])
    assert live_modal_call_rows(repo_root=root) == []
    assert assert_modal_single_flight(repo_root=root, check_cloud=False) == []


def test_same_lane_live_ledger_row_still_refuses(root):
    # A non-terminal call_id on the CALLER'S OWN lane is exactly the
    # un-harvested duplicate-breeder — never excluded.
    _write_ledger(root, [{
        "call_id": "fc-own", "status": "running", "lane_id": "laneA",
    }])
    with pytest.raises(ModalSingleFlightRefusal):
        assert_modal_single_flight(lane_id="laneA", repo_root=root, check_cloud=False)


def test_corrupt_ledger_lines_skipped_valid_row_still_counts(root):
    path = root / ".omx/state/modal_call_id_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "NOT-JSON\n"
        + json.dumps({"call_id": "fc-ok", "status": "dispatched"}) + "\n"
    )
    rows = live_modal_call_rows(repo_root=root)
    assert [r["call_id"] for r in rows] == ["fc-ok"]


def test_active_modal_claim_other_lane_refuses(root):
    _write_claims(root, [_claim_row("laneB", "job-b", "active_dispatched")])
    with pytest.raises(ModalSingleFlightRefusal) as exc:
        assert_modal_single_flight(lane_id="laneA", repo_root=root, check_cloud=False)
    assert "laneB" in str(exc.value)


def test_own_lane_claim_excluded(root):
    # Healthy path: the dispatcher claimed its own lane first, then dispatches.
    _write_claims(root, [_claim_row("laneA", "job-a", "active_dispatched")])
    assert assert_modal_single_flight(
        lane_id="laneA", repo_root=root, check_cloud=False,
    ) == []


def test_non_modal_platform_claim_ignored(root):
    _write_claims(root, [_claim_row("laneB", "job-b", "active", platform="vastai")])
    assert single_flight_findings(repo_root=root, check_cloud=False) == []


def test_terminal_claim_ignored(root):
    _write_claims(root, [_claim_row("laneB", "job-b", "completed_harvested")])
    assert single_flight_findings(repo_root=root, check_cloud=False) == []


def test_latest_row_wins_newest_first_claims(root):
    # Claims file is newest-first: a terminal row ABOVE an older active row for
    # the same (lane, job) means the claim is closed.
    _write_claims(root, [
        _claim_row("laneB", "job-b", "completed_harvested"),
        _claim_row("laneB", "job-b", "active_dispatched"),
    ])
    assert active_modal_claims(repo_root=root) == []


# ── runtime guard: override escape ──────────────────────────────────────────


def test_force_rationale_returns_findings_no_raise(root, capsys):
    _write_ledger(root, [{"call_id": "fc-live", "status": "dispatched"}])
    findings = assert_modal_single_flight(
        repo_root=root,
        check_cloud=False,
        force_rationale="operator GO 2026-07-17: paired A/B needs 2nd concurrent job",
    )
    assert len(findings) == 1
    err = capsys.readouterr().err
    assert "OPERATOR OVERRIDE" in err


def test_placeholder_force_rationale_still_refuses(root):
    _write_ledger(root, [{"call_id": "fc-live", "status": "dispatched"}])
    with pytest.raises(ModalSingleFlightRefusal):
        assert_modal_single_flight(
            repo_root=root, check_cloud=False, force_rationale="<rationale>",
        )


def test_short_force_rationale_still_refuses(root):
    _write_ledger(root, [{"call_id": "fc-live", "status": "dispatched"}])
    with pytest.raises(ModalSingleFlightRefusal):
        assert_modal_single_flight(
            repo_root=root, check_cloud=False, force_rationale="ok",
        )


def test_env_force_rationale_escape(root, monkeypatch, capsys):
    _write_ledger(root, [{"call_id": "fc-live", "status": "dispatched"}])
    monkeypatch.setenv(
        "TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE",
        "operator override quoted in claim notes 2026-07-17",
    )
    findings = assert_modal_single_flight(repo_root=root, check_cloud=False)
    assert findings and "OPERATOR OVERRIDE" in capsys.readouterr().err


# ── runtime guard: cloud surface ────────────────────────────────────────────


def test_cloud_check_findings_included(root, monkeypatch):
    monkeypatch.setattr(
        sf, "cloud_live_modal_apps",
        lambda **kw: ["appX (state=deployed, tasks=2)"],
    )
    with pytest.raises(ModalSingleFlightRefusal) as exc:
        assert_modal_single_flight(repo_root=root, check_cloud=True)
    assert "appX" in str(exc.value)


def test_skip_cloud_env_skips_cloud_check(root, monkeypatch):
    def _boom(**kw):  # pragma: no cover - must not be called
        raise AssertionError("cloud check must be skipped")

    monkeypatch.setattr(sf, "cloud_live_modal_apps", _boom)
    monkeypatch.setenv("TAC_MODAL_SINGLE_FLIGHT_SKIP_CLOUD", "1")
    assert assert_modal_single_flight(repo_root=root, check_cloud=True) == []


def test_check_cloud_false_skips_cloud_check(root, monkeypatch):
    def _boom(**kw):  # pragma: no cover - must not be called
        raise AssertionError("cloud check must be skipped")

    monkeypatch.setattr(sf, "cloud_live_modal_apps", _boom)
    assert assert_modal_single_flight(repo_root=root, check_cloud=False) == []


# ── dual-ledger terminality ─────────────────────────────────────────────────


def test_terminality_blocker_when_claim_still_active(root):
    _write_claims(root, [_claim_row("laneA", "fc-dead", "active_dispatched")])
    blockers = dual_ledger_terminality_blockers(
        call_id="fc-dead", repo_root=root,
    )
    assert len(blockers) == 1
    assert "fc-dead" in blockers[0]
    assert "claim_lane_dispatch.py" in blockers[0]


def test_terminality_blocker_matches_by_lane_id(root):
    _write_claims(root, [_claim_row("laneA", "job-x", "active_dispatched")])
    blockers = dual_ledger_terminality_blockers(
        call_id="fc-other", lane_id="laneA", repo_root=root,
    )
    assert len(blockers) == 1


def test_no_terminality_blocker_when_claim_terminal(root):
    _write_claims(root, [_claim_row("laneA", "fc-dead", "completed_harvested")])
    assert dual_ledger_terminality_blockers(call_id="fc-dead", repo_root=root) == []


def test_emit_hook_non_terminal_record_no_blocker(root):
    _write_claims(root, [_claim_row("laneA", "fc-1", "active_dispatched")])
    ledger = root / ".omx/state/modal_call_id_ledger.jsonl"
    out = emit_dual_ledger_terminality_blocker_if_needed(
        record={"call_id": "fc-1", "status": "dispatched"}, ledger_path=ledger,
    )
    assert out == []


def test_emit_hook_derives_repo_root_from_ledger_path(root, capsys):
    _write_claims(root, [_claim_row("laneA", "fc-1", "active_dispatched")])
    ledger = root / ".omx/state/modal_call_id_ledger.jsonl"
    out = emit_dual_ledger_terminality_blocker_if_needed(
        record={"call_id": "fc-1", "status": "failed"}, ledger_path=ledger,
    )
    assert len(out) == 1
    assert "BLOCKER" in capsys.readouterr().err


def test_update_call_id_outcome_emits_blocker(root, capsys):
    from tac.deploy.modal.call_id_ledger import update_call_id_outcome

    _write_claims(root, [_claim_row("laneA", "fc-int", "active_dispatched")])
    ledger = root / ".omx/state/modal_call_id_ledger.jsonl"
    update_call_id_outcome(
        call_id="fc-int",
        status="failed",
        path=ledger,
        lock_path=root / ".omx/state/.ledger-lock",
    )
    assert "BLOCKER" in capsys.readouterr().err


# ── static gate: check_modal_dispatch_single_flight ─────────────────────────


def _write_surface(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_gate_flags_unrouted_spawn(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_new_dispatch.py",
        "import modal\n\ncall = fn.spawn(archive_bytes)\n",
    )
    violations = check_modal_dispatch_single_flight(repo_root=tmp_path)
    assert len(violations) == 1
    assert "assert_modal_single_flight" in violations[0]


def test_gate_passes_guarded_file(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_new_dispatch.py",
        "import modal\n"
        "from tac.deploy.modal.single_flight import assert_modal_single_flight\n"
        "assert_modal_single_flight(label='x', lane_id='y')\n"
        "call = fn.spawn(archive_bytes)\n",
    )
    assert check_modal_dispatch_single_flight(repo_root=tmp_path) == []


def test_gate_same_line_waiver_honored(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_new_dispatch.py",
        "import modal\n"
        "call = fn.spawn(x)  # MODAL_SINGLE_FLIGHT_OK:fanout-child-of-claimed-parent\n",
    )
    assert check_modal_dispatch_single_flight(repo_root=tmp_path) == []


def test_gate_placeholder_waiver_rejected(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_new_dispatch.py",
        "import modal\n"
        "call = fn.spawn(x)  # MODAL_SINGLE_FLIGHT_OK:<rationale>\n",
    )
    assert len(check_modal_dispatch_single_flight(repo_root=tmp_path)) == 1


def test_gate_prose_and_comment_spawn_not_flagged(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_harvest_only.py",
        'import modal\n'
        '"""Recovers artifacts dispatched via `.spawn()` earlier."""\n'
        "# the dispatcher used fn.spawn( elsewhere\n"
        "x = 'text mentioning .spawn() with empty parens'\n",
    )
    assert check_modal_dispatch_single_flight(repo_root=tmp_path) == []


def test_gate_non_modal_file_ignored(tmp_path):
    _write_surface(
        tmp_path, "tools/random_helper.py",
        "children = seed_seq.spawn(10)\n",
    )
    assert check_modal_dispatch_single_flight(repo_root=tmp_path) == []


def test_gate_test_files_and_guard_module_skipped(tmp_path):
    _write_surface(
        tmp_path, "experiments/test_modal_thing.py",
        "import modal\ncall = fn.spawn(x)\n",
    )
    _write_surface(
        tmp_path, "src/tac/deploy/modal/single_flight.py",
        "import modal\ncall = fn.spawn(x)\n",
    )
    assert check_modal_dispatch_single_flight(repo_root=tmp_path) == []


def test_gate_strict_raises(tmp_path):
    _write_surface(
        tmp_path, "experiments/modal_new_dispatch.py",
        "import modal\ncall = fn.spawn(x)\n",
    )
    with pytest.raises(PreflightError):
        check_modal_dispatch_single_flight(repo_root=tmp_path, strict=True)


def test_gate_live_repo_count_is_zero():
    violations = check_modal_dispatch_single_flight()
    assert violations == [], "\n".join(violations)
