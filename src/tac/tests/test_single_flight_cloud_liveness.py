"""What "a live Modal app" MEANS to the single-flight guard.

MEASURED 2026-08-20 (``ddm_rr7``). The guard treated ``modal app list``'s ``tasks`` count as
liveness. That count is not liveness for ephemeral/detached apps: one whose containers exited
but which was never explicitly stopped keeps a non-zero count indefinitely. On the operator's
account three apps reported "running tasks" at ages of **272.8 h / 168.9 h / 143.5 h** while
``modal container list`` showed exactly ONE active container.

The two defects compose, which is why both halves are tested here rather than one:

* the cloud leg had been SKIPPING entirely (the CLI was not on PATH — see
  ``test_single_flight_modal_bin_resolution.py``), so nobody saw the bad predicate;
* restoring the leg without fixing the predicate would have turned a silent skip into a FALSE
  REFUSE on every future dispatch, blocking correct work with week-old ghosts.

So an app counts live only when ``tasks > 0`` AND it is not stopped AND it owns a running
container. The container query is authoritative; the tasks predicate is kept as a necessary
condition so the change can only ever REMOVE false positives.
"""

from __future__ import annotations

import json

from tac.deploy.modal import single_flight

APP_WITH_CONTAINER = "ap-live"
APP_WITHOUT_CONTAINER = "ap-ghost"


def _fake_cli(monkeypatch, *, apps: list[dict], containers: object) -> None:
    """Stub both CLI reads. ``containers`` may be a list, or an rc!=0 sentinel."""

    def fake_run(cmd, **_kwargs):
        class R:
            returncode = 0
            stdout = ""

        r = R()
        if "container" in cmd:
            if containers is None:
                r.returncode = 1
                return r
            r.stdout = json.dumps(containers)
            return r
        r.stdout = json.dumps(apps)
        return r

    monkeypatch.setattr(single_flight, "_resolve_modal_bin", lambda: "/fake/modal")
    monkeypatch.setattr(single_flight.subprocess, "run", fake_run)


def test_stale_task_count_without_a_container_is_not_live(monkeypatch) -> None:
    """The 6-to-11-day ghost case: tasks>0, never stopped, no container. NOT live."""
    _fake_cli(
        monkeypatch,
        apps=[{"app_id": APP_WITHOUT_CONTAINER, "description": "comma-ddm-sa1-t4-sign-gate",
               "state": "ephemeral (detached)", "tasks": "2"}],
        containers=[],
    )
    assert single_flight.cloud_live_modal_apps() == []


def test_a_running_container_is_live(monkeypatch) -> None:
    """The genuine case must still refuse a second fire."""
    _fake_cli(
        monkeypatch,
        apps=[{"app_id": APP_WITH_CONTAINER, "description": "comma-auth-eval",
               "state": "ephemeral (detached)", "tasks": "1"}],
        containers=[{"app_id": APP_WITH_CONTAINER, "app_name": "comma-auth-eval"}],
    )
    live = single_flight.cloud_live_modal_apps()
    assert len(live) == 1
    assert "comma-auth-eval" in live[0]


def test_ghost_and_real_are_separated_in_one_pass(monkeypatch) -> None:
    """The measured shape on 2026-08-20: ghosts alongside one real in-flight row."""
    _fake_cli(
        monkeypatch,
        apps=[
            {"app_id": APP_WITHOUT_CONTAINER, "description": "comma-dali-av-gt-diff",
             "state": "ephemeral", "tasks": "1"},
            {"app_id": APP_WITH_CONTAINER, "description": "comma-auth-eval",
             "state": "ephemeral (detached)", "tasks": "1"},
        ],
        containers=[{"app_id": APP_WITH_CONTAINER}],
    )
    live = single_flight.cloud_live_modal_apps()
    assert len(live) == 1
    assert "comma-auth-eval" in live[0]
    assert all("dali" not in entry for entry in live)


def test_unavailable_container_query_keeps_the_conservative_predicate(monkeypatch) -> None:
    """FAIL-CLOSED: if liveness is unknowable, over-report rather than under-report.

    Over-reporting costs a refusal the operator can override with a rationale; under-reporting
    costs a double fire. The guard must degrade toward keeping itself.
    """
    _fake_cli(
        monkeypatch,
        apps=[{"app_id": APP_WITHOUT_CONTAINER, "description": "comma-ddm-sa1-t4-sign-gate",
               "state": "ephemeral (detached)", "tasks": "2"}],
        containers=None,  # rc != 0
    )
    live = single_flight.cloud_live_modal_apps()
    assert len(live) == 1, "an unknowable container query must not silently clear the guard"


def test_stopped_apps_are_still_excluded(monkeypatch) -> None:
    """The pre-existing predicate half is preserved, not replaced."""
    _fake_cli(
        monkeypatch,
        apps=[{"app_id": APP_WITH_CONTAINER, "description": "comma-auth-eval",
               "state": "stopped", "tasks": "3"}],
        containers=[{"app_id": APP_WITH_CONTAINER}],
    )
    assert single_flight.cloud_live_modal_apps() == []
