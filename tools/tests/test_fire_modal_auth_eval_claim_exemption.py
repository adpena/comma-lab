# SPDX-License-Identifier: MIT
"""The fire tool must not terminal-close the claim its own dispatch is about to consume.

THE DEFECT (measured, rc2 T4 fire, 2026-08-20). Stage 4's phantom auto-closer fires on
"an active Modal claim while the call-id ledger has ZERO live rows". A claim PRE-STAGED
for the dispatch that is one stage away satisfies that trigger *by construction*: it is
active, and its call does not exist yet. So the closer closed it, and the worker's
`--claim-policy require_active` then refused the fire:

    /Volumes/APDataStore/pact/ddm_rc2/t4_row_r1/FIRE_REFUSED.json
      stage4_claims.closed = [{lane_ddm_rc2_composed_cuda_20260820,
                               modal:ddm_rc2_composed_cuda_r1, rc 0}]
      entrypoint_refusal_lines[0] = "... could not find an active lane claim: newest
        matching claim is terminal: ... status=stale_superseded_reconciled_no_live_call"
      refusal_rc = 5  ("dispatch produced no spawn record — the fire DID NOT take")

Same lane, same job, same run: the tool disarmed its own guard. The documented
workaround was `--claim-policy open`, i.e. turning the guard off.

THE CURE is an identity + freshness exemption, and these tests execute BOTH directions:
a fresh claim carrying this invocation's lane_id AND instance_job_id survives; anything
else — a different lane/job, or the same lane/job written outside the window — still
closes. Identity alone would be too weak (a job id reused after an abandoned fire is a
real phantom); freshness alone would be too weak (a sister lane's minutes-old claim is
still a single-flight blocker).

Every test drives the REAL `tools/claim_lane_dispatch.py` CLI against a tmp claims file
and a tmp ledger. Nothing here touches `.omx/state/active_lane_dispatch_claims.md`.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import fire_modal_auth_eval as fire  # noqa: E402

CLAIM_CLI = REPO / "tools" / "claim_lane_dispatch.py"
VENV_PY = REPO / ".venv" / "bin" / "python"


def _stage_claim(
    claims_path: Path,
    *,
    lane: str,
    job: str,
    status: str = "active_paid_dispatch",
    age_hours: float = 0.0,
    override: bool = False,
) -> None:
    """Append a claim through the canonical CLI, optionally stamped in the past."""

    stamp = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=age_hours)
    argv = [
        str(VENV_PY), str(CLAIM_CLI), "claim",
        "--claims-path", str(claims_path),
        "--lane-id", lane,
        "--platform", "modal",
        "--instance-job-id", job,
        "--agent", "TEST",
        "--status", status,
        "--now-utc", stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--notes", "pre-staged by the exemption test",
    ]
    if override:
        # A second concurrent active Modal claim trips the single-flight refusal; the
        # test needs that exact two-claim state, so it opts in explicitly.
        argv += ["--override", "--force"]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    claims = tmp_path / "claims.md"
    ledger = tmp_path / "modal_call_id_ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    return claims, ledger


def _reconcile(claims: Path, ledger: Path, **kwargs) -> dict:
    return fire.reconcile_claims(
        "TEST-AUTOCLOSE", False, claims_path=claims, modal_ledger=ledger, **kwargs
    )


def _newest_status(claims: Path, lane: str, job: str) -> str:
    """Newest-first table: the first row matching lane+job wins."""

    for line in claims.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and cells[2] == lane and cells[4] == job:
            return cells[6]
    raise AssertionError(f"no claim row for {lane} / {job}")


# --------------------------------------------------------------------------------------
# POSITIVE CONTROL — the claim this fire pre-staged SURVIVES.
# --------------------------------------------------------------------------------------

def test_pre_staged_claim_for_this_dispatch_survives_reconciliation(tmp_path: Path) -> None:
    claims, ledger = _fixture(tmp_path)
    lane, job = "lane_ddm_rc2_composed_cuda_20260820", "modal:ddm_rc2_composed_cuda_r1"
    _stage_claim(claims, lane=lane, job=job)

    action = _reconcile(claims, ledger, self_lane_id=lane, self_instance_job_id=job)

    assert action["closed"] == [], action
    assert [(e["lane"], e["job"]) for e in action["exempt"]] == [(lane, job)]
    # The receipt is not the proof — the ledger is. The claim must still be usable by
    # `--claim-policy require_active`, which reads the newest matching row.
    assert _newest_status(claims, lane, job) == "active_paid_dispatch"
    assert "stale_superseded_reconciled_no_live_call" not in claims.read_text()


def test_the_reconciler_still_reports_the_problem_it_no_longer_acts_on(tmp_path: Path) -> None:
    """Exempting is not hiding: the operator-visible problem line is still recorded."""

    claims, ledger = _fixture(tmp_path)
    lane, job = "lane_self", "modal:self"
    _stage_claim(claims, lane=lane, job=job)

    action = _reconcile(claims, ledger, self_lane_id=lane, self_instance_job_id=job)

    assert any("NO live ledger call_id" in line for line in action["reconcile_output_tail"])


# --------------------------------------------------------------------------------------
# INVERSE CONTROLS — everything that is NOT this fire's fresh claim still closes.
# --------------------------------------------------------------------------------------

def test_a_phantom_claim_on_another_lane_is_still_terminal_closed(tmp_path: Path) -> None:
    claims, ledger = _fixture(tmp_path)
    _stage_claim(claims, lane="lane_someone_else", job="modal:someone_else")

    action = _reconcile(
        claims, ledger, self_lane_id="lane_mine", self_instance_job_id="modal:mine"
    )

    assert [(c["lane"], c["job"]) for c in action["closed"]] == [
        ("lane_someone_else", "modal:someone_else")
    ]
    assert action["exempt"] == []
    assert _newest_status(claims, "lane_someone_else", "modal:someone_else") == (
        "stale_superseded_reconciled_no_live_call"
    )


def test_same_lane_but_a_different_job_is_not_exempt(tmp_path: Path) -> None:
    """Identity is lane AND job. A stale sibling job on our own lane is still a phantom."""

    claims, ledger = _fixture(tmp_path)
    lane = "lane_shared"
    _stage_claim(claims, lane=lane, job="modal:previous_attempt")

    action = _reconcile(
        claims, ledger, self_lane_id=lane, self_instance_job_id="modal:this_attempt"
    )

    assert [c["job"] for c in action["closed"]] == ["modal:previous_attempt"]
    assert action["exempt"] == []


def test_same_lane_and_job_but_older_than_the_window_is_not_exempt(tmp_path: Path) -> None:
    """The freshness leg. Identity alone must not grant an exemption forever."""

    claims, ledger = _fixture(tmp_path)
    lane, job = "lane_reused", "modal:reused_job_id"
    _stage_claim(claims, lane=lane, job=job, age_hours=fire.SELF_CLAIM_MAX_AGE_HOURS + 18.0)

    action = _reconcile(claims, ledger, self_lane_id=lane, self_instance_job_id=job)

    assert [(c["lane"], c["job"]) for c in action["closed"]] == [(lane, job)]
    assert action["exempt"] == []
    assert _newest_status(claims, lane, job) == "stale_superseded_reconciled_no_live_call"


def test_the_exemption_covers_exactly_one_claim_not_the_whole_sweep(tmp_path: Path) -> None:
    """Two active claims, one ours: ours survives, the other closes, in the same pass."""

    claims, ledger = _fixture(tmp_path)
    lane, job = "lane_mine", "modal:mine"
    _stage_claim(claims, lane="lane_phantom", job="modal:phantom")
    _stage_claim(claims, lane=lane, job=job, override=True)

    action = _reconcile(claims, ledger, self_lane_id=lane, self_instance_job_id=job)

    assert [(c["lane"], c["job"]) for c in action["closed"]] == [
        ("lane_phantom", "modal:phantom")
    ]
    assert [(e["lane"], e["job"]) for e in action["exempt"]] == [(lane, job)]
    assert _newest_status(claims, lane, job) == "active_paid_dispatch"
    assert _newest_status(claims, "lane_phantom", "modal:phantom") == (
        "stale_superseded_reconciled_no_live_call"
    )


# --------------------------------------------------------------------------------------
# PRECONDITIONS the cure must not weaken.
# --------------------------------------------------------------------------------------

def test_a_live_ledger_row_still_suppresses_every_auto_close(tmp_path: Path) -> None:
    claims, ledger = _fixture(tmp_path)
    _stage_claim(claims, lane="lane_phantom", job="modal:phantom")
    ledger.write_text(
        json.dumps(
            {
                "call_id": "fc-LIVE",
                "status": "dispatched",
                "written_at_utc": dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    action = _reconcile(
        claims, ledger, self_lane_id="lane_mine", self_instance_job_id="modal:mine"
    )

    assert action["closed"] == []
    assert action["live_modal_call_ids"] == ["fc-LIVE"]
    assert _newest_status(claims, "lane_phantom", "modal:phantom") == "active_paid_dispatch"


def test_an_unparseable_reconcile_closes_nothing_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail SAFE, never silent: closing on a guess is how the rc2 refusal happened."""

    class _Broken:
        returncode = 1
        stdout = "MODAL RECONCILE: exploded"
        stderr = "Traceback ..."

    monkeypatch.setattr(fire.subprocess, "run", lambda *a, **k: _Broken())
    action = fire.reconcile_claims("TEST", False, self_lane_id="l", self_instance_job_id="j")

    assert action["closed"] == []
    assert action["reconcile_unparseable"] is True
    assert action["reconcile_output_tail"]


def test_dry_run_never_closes_a_claim_on_disk(tmp_path: Path) -> None:
    claims, ledger = _fixture(tmp_path)
    _stage_claim(claims, lane="lane_phantom", job="modal:phantom")

    action = fire.reconcile_claims(
        "TEST", True, self_lane_id="lane_mine", self_instance_job_id="modal:mine",
        claims_path=claims, modal_ledger=ledger,
    )

    assert action["closed"] == [
        {"lane": "lane_phantom", "job": "modal:phantom", "dry_run": True}
    ]
    assert _newest_status(claims, "lane_phantom", "modal:phantom") == "active_paid_dispatch"


# --------------------------------------------------------------------------------------
# WIRING — a cure the entry point does not pass is an orphan cure.
# --------------------------------------------------------------------------------------

def test_main_passes_its_own_lane_and_job_into_the_reconciler() -> None:
    """The helper can exempt only what main() tells it about. Assert the call site."""

    tree = ast.parse((REPO / "tools" / "fire_modal_auth_eval.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reconcile_claims"
    ]
    assert calls, "main() no longer calls reconcile_claims"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "self_lane_id" in kwargs and "self_instance_job_id" in kwargs, (
            "reconcile_claims call site dropped the self-identity — the exemption is "
            "inert and the rc2 self-defeat can recur"
        )
        assert isinstance(kwargs["self_lane_id"], ast.Attribute)
        assert kwargs["self_lane_id"].attr == "lane_id"
        assert isinstance(kwargs["self_instance_job_id"], ast.Attribute)
        assert kwargs["self_instance_job_id"].attr == "instance_job_id"


def test_reconciler_reads_the_timestamp_the_exemption_depends_on(tmp_path: Path) -> None:
    """The freshness leg is only expressible because reconcile emits per-claim ages."""

    claims, ledger = _fixture(tmp_path)
    _stage_claim(claims, lane="lane_x", job="modal:x", age_hours=3.0)
    proc = subprocess.run(
        [
            str(VENV_PY), str(CLAIM_CLI), "reconcile",
            "--claims-path", str(claims),
            "--modal-ledger", str(ledger),
            "--format", "json",
        ],
        capture_output=True, text=True, cwd=REPO,
    )
    row = json.loads(proc.stdout)["active_modal_claims"][0]

    assert row["timestamp_utc"]
    assert 2.9 <= row["age_hours"] <= 3.1
