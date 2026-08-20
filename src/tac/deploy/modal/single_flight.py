"""Modal SINGLE-FLIGHT pre-spawn runtime guard + dual-ledger terminality (#513).

Operator binding 2026-07-15 (memory ``modal_single_flight_dual_ledger_policy_
20260715``): ONE Modal job in flight at a time unless EXPLICIT operator
override; before ANY Modal dispatch verify zero non-terminal Modal work on all
three surfaces (local call-id ledger + cross-agent claims file + live
``modal app list``); terminal rows are written to BOTH local ledgers in the
same turn.

This module is the RUNTIME half of the #513 two-landing (the static halves are
``check_modal_single_flight_ledger_consistency`` — ledger-STATE consistency —
and ``check_modal_dispatch_single_flight`` — dispatch-SURFACE routing — in
``src/tac/preflight.py``). The common claim helper performs a precise guard
before it writes the dispatcher-owned claim. Every Modal dispatch entry point
also calls :func:`assert_modal_single_flight` immediately BEFORE ``.spawn()``:

    from tac.deploy.modal.single_flight import assert_modal_single_flight
    assert_modal_single_flight(label=label, lane_id=lane_id)

Refusal semantics (fail-CLOSED on real findings, fail-OPEN on broken
observability surfaces — a corrupt ledger must not be able to block a
legitimate refusal-free dispatch, and the cloud check must not require
network health to dispatch):

* refuses when the local call-id ledger has ANY live (non-terminal) call_id;
* refuses when the claims file has an ACTIVE Modal claim that is not the
  caller's own lane-agent-job claim (legacy immediate pre-spawn rechecks may
  omit the agent and exclude their already-claimed lane);
* refuses when the live ``modal app list`` cross-check (opt-out via
  ``check_cloud=False`` or env ``TAC_MODAL_SINGLE_FLIGHT_SKIP_CLOUD=1``)
  reports a deployed app with running tasks (JSON ``tasks > 0`` and state not
  stopped — the coordinator-corrected predicate, commit ``adf04cdc18``);
* the ONLY escape is an explicit operator-override rationale
  (``force_rationale=...`` or env ``TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE``)
  — placeholder rationales are rejected per Catalog #287.

Dual-ledger terminality: :func:`dual_ledger_terminality_blockers` is invoked
by ``call_id_ledger.update_call_id_outcome`` after every TERMINAL outcome row
so a terminal call_id whose claims-file row is still active emits a LOUD
blocker in the same turn (the stale-active claim is the duplicate-breeder).

Sisters: CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + "CROSS-AGENT DISPATCH
COORDINATION" · ``tools/claim_lane_dispatch.py`` (claim-time refusal rc=5 +
``reconcile`` subcommand) · task #512 (local launcher same-outdir guard —
this module is its cloud-side twin).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "ModalSingleFlightRefusal",
    "active_modal_claims",
    "assert_modal_single_flight",
    "cloud_live_modal_apps",
    "dual_ledger_terminality_blockers",
    "emit_dual_ledger_terminality_blocker_if_needed",
    "live_modal_call_rows",
    "single_flight_findings",
]

_MODAL_LEDGER_REL = ".omx/state/modal_call_id_ledger.jsonl"
_MODAL_CLAIMS_REL = ".omx/state/active_lane_dispatch_claims.md"

# Non-terminal ledger states (mirror of the claim tool + preflight vocab; a
# row whose LATEST state is one of these is "live"). Everything else
# (harvested/failed/stale/manually_terminated/pre_spawn_fatal/...) is terminal.
_NON_TERMINAL_LEDGER_STATES: frozenset[str] = frozenset(
    {
        "dispatched",
        "active",
        "running",
        "spawned",
        "in_progress",
        "pending",
        "queued",
    }
)

# Claims-file status-token vocab (mirror of check_modal_single_flight_ledger_
# consistency in src/tac/preflight.py — keep in sync).
_CLAIM_TERMINAL_TOKENS: tuple[str, ...] = (
    "failed",
    "stopped",
    "completed",
    "complete",
    "harvested",
    "refused",
    "stale",
    "cancelled",
    "canceled",
    "terminal",
    "timeout",
    "timed_out",
    "error",
    "killed",
    "superseded",
    "oom",
)
_CLAIM_ACTIVE_TOKENS: tuple[str, ...] = (
    "active",
    "dispatched",
    "running",
    "spawned",
    "eval",
)

# Placeholder rationales rejected per Catalog #287 (data-content layer).
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {
        "<rationale>",
        "<reason>",
        "rationale",
        "reason",
        "tbd",
        "todo",
        "placeholder",
        "x",
        "-",
        "none",
        "n/a",
    }
)

_FORCE_ENV = "TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE"
_SKIP_CLOUD_ENV = "TAC_MODAL_SINGLE_FLIGHT_SKIP_CLOUD"
_CLOUD_TIMEOUT_SECONDS = 45.0
_MODAL_CALL_ID_RE = re.compile(r"\bfc-[A-Za-z0-9]+\b", re.IGNORECASE)


class ModalSingleFlightRefusal(RuntimeError):
    """Raised by :func:`assert_modal_single_flight` when a live Modal job exists.

    Carries ``findings`` (one string per conflicting surface row) so callers can
    surface the exact conflicts in their own FATAL banner.
    """

    def __init__(self, findings: list[str]):
        self.findings = list(findings)
        joined = "\n  - ".join(self.findings)
        super().__init__(
            "MODAL SINGLE-FLIGHT REFUSAL (operator binding 2026-07-15): a live "
            "Modal job / active claim already exists — HARVEST or close it "
            "first; never fire a second. Conflicts:\n  - " + joined + "\n"
            "Escape (operator override ONLY): pass force_rationale=... / set "
            f"{_FORCE_ENV}=<substantive rationale>, and quote the rationale in "
            "the claim notes. Reconcile first: .venv/bin/python "
            "tools/claim_lane_dispatch.py reconcile"
        )


def _repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root)
    # src/tac/deploy/modal/single_flight.py -> repo root is 4 parents up from
    # this file's directory (modal -> deploy -> tac -> src -> root).
    return Path(__file__).resolve().parents[4]


def live_modal_call_rows(*, repo_root: Path | str | None = None) -> list[dict[str, Any]]:
    """Latest-row-wins per call_id; return rows whose latest state is non-terminal.

    Fail-OPEN: a missing or unreadable ledger yields ``[]`` (the STATE gate +
    reconcile tool own corruption reporting; the runtime guard must not brick
    dispatch on observability breakage).
    """
    path = _repo_root(repo_root) / _MODAL_LEDGER_REL
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                call_id = str(row.get("call_id") or "").strip()
                if not call_id:
                    continue
                latest[call_id] = row  # file order is chronological (append-only)
    except OSError:
        return []
    live: list[dict[str, Any]] = []
    for row in latest.values():
        state = str(row.get("status") or row.get("event_type") or "").strip().lower()
        if state in _NON_TERMINAL_LEDGER_STATES:
            live.append(row)
    return live


def active_modal_claims(
    *,
    repo_root: Path | str | None = None,
) -> list[dict[str, str]]:
    """ACTIVE (non-terminal) Modal claim rows from the cross-agent claims file.

    Latest-row-wins per ``(lane_id, instance_job_id)`` — the claims file is
    newest-first, so the FIRST row seen for a job key wins. Returns dicts with
    ``lane_id`` / ``platform`` / ``instance_job_id`` / ``status`` / ``notes``.
    Fail-OPEN on a missing/unreadable file.
    """
    path = _repo_root(repo_root) / _MODAL_CLAIMS_REL
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    seen: set[tuple[str, str]] = set()
    active: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        if cells[0].lower() in ("timestamp_utc", ""):
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        row = {
            "timestamp_utc": cells[0],
            "agent": cells[1],
            "lane_id": cells[2],
            "platform": cells[3],
            "instance_job_id": cells[4],
            "status": cells[6],
            "notes": cells[7],
        }
        key = (row["lane_id"].lower(), row["instance_job_id"].lower())
        if key in seen:
            continue  # newest-first file: first row per job key is latest
        seen.add(key)
        if "modal" not in row["platform"].lower():
            continue
        status_l = row["status"].lower()
        has_terminal = any(t in status_l for t in _CLAIM_TERMINAL_TOKENS)
        has_active = any(t in status_l for t in _CLAIM_ACTIVE_TOKENS)
        if has_active and not has_terminal:
            active.append(row)
    return active


def _resolve_modal_bin() -> str | None:
    """Locate the modal CLI the way the rest of this repo does: venv FIRST, then PATH.

    MEASURED 2026-08-20 (``ddm_rr7``): ``shutil.which("modal")`` alone returned None on the
    operator's machine because ``modal`` lives at ``<repo>/.venv/bin/modal`` and ``.venv/bin``
    is not on PATH. The cloud cross-check therefore SKIPPED on every dispatch from this repo,
    degrading single-flight to local-ledgers-only — and it announced that with one stderr line
    that scrolls past inside Modal's own image-build output, so the degradation was invisible
    in practice.

    That mattered on the same day: the local leg was simultaneously poisoned by unit-test
    fixtures writing phantom-live ``fc-test-*`` rows into the production call-id ledger. Both
    legs of the guard were compromised at once, in opposite directions — one inert, one lying.
    ``tools/fire_modal_auth_eval.py`` already resolves ``REPO/.venv/bin/modal`` explicitly;
    this makes the guard agree with the tool that actually dispatches.

    Still fail-OPEN when genuinely absent: dispatch must not brick on CLI availability.
    """
    venv_modal = Path(__file__).resolve().parents[4] / ".venv" / "bin" / "modal"
    if venv_modal.is_file():
        return str(venv_modal)
    return shutil.which("modal")


def _running_container_app_ids(
    modal_bin: str, *, timeout_seconds: float = _CLOUD_TIMEOUT_SECONDS
) -> set[str] | None:
    """App ids that have an ACTUALLY RUNNING container, or None if unknowable.

    WHY THIS EXISTS (MEASURED 2026-08-20, ``ddm_rr7``). ``modal app list``'s ``tasks`` count
    is NOT a liveness signal for ephemeral/detached apps: a detached app whose containers
    exited but which was never explicitly stopped keeps a non-zero count indefinitely. On the
    operator's account it reported three live apps aged **272.8 h, 168.9 h and 143.5 h**
    (``comma-dali-av-gt-diff``, ``comma-ddm-sa1-t4-sign-gate``,
    ``comma-ddm-re1-round1-t4-sign-gate``) while ``modal container list`` showed exactly ONE
    active container — the auth-eval row being fired at that moment. Six-to-eleven-day-old
    "running tasks" for jobs that take minutes are stale metadata, not compute.

    That matters because this leg had been silently skipping (the CLI was not on PATH). Simply
    restoring it would have traded a silent SKIP for a false REFUSE on every future dispatch —
    a guard that blocks correct work is not an improvement over one that greens by not looking.
    So the ``app list`` predicate is kept as a NECESSARY condition and intersected with the
    authoritative one.

    Fail-CLOSED toward the old behaviour: when the container query is unavailable this returns
    None and the caller applies the ``tasks``-only predicate, which over-reports. Over-reporting
    costs a refusal the operator can override with a rationale; under-reporting costs a double
    fire. The conservative direction is the one that keeps the guard.
    """
    try:
        proc = subprocess.run(
            [modal_bin, "container", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        containers = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(containers, list):
        return None
    return {
        str(c.get("app_id") or c.get("App ID") or "")
        for c in containers
        if isinstance(c, dict) and (c.get("app_id") or c.get("App ID"))
    }


def cloud_live_modal_apps(*, timeout_seconds: float = _CLOUD_TIMEOUT_SECONDS) -> list[str] | None:
    """Live cross-check: ``modal app list --json`` apps with running tasks.

    Predicate (coordinator-corrected, commit ``adf04cdc18``): an app is LIVE
    when ``tasks > 0`` AND its state is not stopped (stopped apps keep history
    rows and must not count). Returns ``None`` (fail-OPEN, loud stderr warning)
    when the CLI is unavailable, errors, or times out — the caller records the
    skipped surface but does not brick dispatch on network health.
    """
    modal_bin = _resolve_modal_bin()
    if modal_bin is None:
        print(
            "[modal-single-flight] WARNING: `modal` CLI not found — cloud cross-check SKIPPED (local ledgers only).",
            file=sys.stderr,
        )
        return None
    try:
        proc = subprocess.run(
            [modal_bin, "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(
            f"[modal-single-flight] WARNING: `modal app list` failed "
            f"({type(exc).__name__}) — cloud cross-check SKIPPED.",
            file=sys.stderr,
        )
        return None
    if proc.returncode != 0:
        print(
            f"[modal-single-flight] WARNING: `modal app list` rc={proc.returncode} — cloud cross-check SKIPPED.",
            file=sys.stderr,
        )
        return None
    try:
        apps = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        print(
            "[modal-single-flight] WARNING: `modal app list --json` returned non-JSON — cloud cross-check SKIPPED.",
            file=sys.stderr,
        )
        return None
    running_app_ids = _running_container_app_ids(modal_bin, timeout_seconds=timeout_seconds)
    live: list[str] = []
    if isinstance(apps, list):
        for app in apps:
            if not isinstance(app, dict):
                continue
            state = str(app.get("State") or app.get("state") or "").strip().lower()
            tasks_raw = app.get("Tasks", app.get("tasks", 0))
            try:
                tasks = int(str(tasks_raw).strip() or "0")
            except ValueError:
                tasks = 0
            if tasks > 0 and "stopped" not in state:
                app_id = str(app.get("App ID") or app.get("app_id") or "")
                if running_app_ids is not None and app_id and app_id not in running_app_ids:
                    # Stale task count, not live compute. See _running_container_app_ids.
                    continue
                name = str(
                    app.get("Description")
                    or app.get("description")
                    or app.get("Name")
                    or app.get("name")
                    or app.get("App ID")
                    or app.get("app_id")
                    or "<unnamed>"
                )
                live.append(f"{name} (state={state or '?'}, tasks={tasks})")
    return live


def single_flight_findings(
    *,
    label: str = "",
    lane_id: str = "",
    claim_agent: str = "",
    repo_root: Path | str | None = None,
    check_cloud: bool = True,
) -> list[str]:
    """All single-flight conflicts across the (up to) three surfaces.

    The caller's OWN active claim is excluded.  When ``claim_agent`` is
    supplied, lane, agent, and a supplied job label must match; an external or
    stale same-lane pre-claim is a conflict, not the dispatcher's own row.  The
    legacy lane-only exclusion remains when no agent is supplied so already-
    claimed dispatchers can perform their immediate pre-spawn recheck.
    Live LEDGER rows are never excluded: a non-terminal call_id on the same
    lane is exactly the un-harvested duplicate-breeder this guard exists for.
    """
    findings: list[str] = []
    for row in live_modal_call_rows(repo_root=repo_root):
        findings.append(
            "call-id ledger: live (non-terminal) call_id="
            f"{row.get('call_id')} label={row.get('label') or '<none>'} "
            f"lane={row.get('lane_id') or '<none>'} status="
            f"{row.get('status') or row.get('event_type')}"
        )
    lane_l = (lane_id or "").strip().lower()
    agent_l = (claim_agent or "").strip().lower()
    label_l = (label or "").strip().lower()
    for claim in active_modal_claims(repo_root=repo_root):
        same_lane = lane_l and claim["lane_id"].strip().lower() == lane_l
        same_agent = agent_l and claim["agent"].strip().lower() == agent_l
        same_label = label_l and claim["instance_job_id"].strip().lower() == label_l
        if same_lane and (not agent_l or (same_agent and (not label_l or same_label))):
            continue  # caller's own claim; never an unrelated same-lane preclaim
        findings.append(
            "claims file: ACTIVE Modal claim lane="
            f"{claim['lane_id']} job={claim['instance_job_id']} "
            f"status={claim['status']}"
        )
    skip_cloud_env = os.environ.get(_SKIP_CLOUD_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if check_cloud and not skip_cloud_env:
        cloud = cloud_live_modal_apps()
        if cloud:
            for entry in cloud:
                findings.append(f"modal app list: live app with running tasks: {entry}")
    return findings


def _rationale_is_substantive(rationale: str | None) -> bool:
    if not isinstance(rationale, str):
        return False
    text = rationale.strip()
    if len(text) < 8:
        return False
    return text.lower() not in _PLACEHOLDER_RATIONALES


def assert_modal_single_flight(
    *,
    label: str = "",
    lane_id: str = "",
    claim_agent: str = "",
    force_rationale: str | None = None,
    repo_root: Path | str | None = None,
    check_cloud: bool = True,
) -> list[str]:
    """Refuse a Modal dispatch unless zero live Modal work exists (or override).

    Call IMMEDIATELY before ``.spawn()``. Returns the (possibly empty) findings
    list on success so the caller can record any override context in the claim
    notes. Raises :class:`ModalSingleFlightRefusal` when findings exist and no
    substantive ``force_rationale`` (param or env
    ``TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE``) is given.
    """
    findings = single_flight_findings(
        label=label,
        lane_id=lane_id,
        claim_agent=claim_agent,
        repo_root=repo_root,
        check_cloud=check_cloud,
    )
    if not findings:
        return []
    rationale = force_rationale
    if not _rationale_is_substantive(rationale):
        rationale = os.environ.get(_FORCE_ENV)
    if _rationale_is_substantive(rationale):
        print(
            "[modal-single-flight] ⚠ OPERATOR OVERRIDE — dispatching label="
            f"{label or '<none>'} lane={lane_id or '<none>'} DESPITE "
            f"{len(findings)} live-Modal conflict(s). Rationale: {rationale!r}. "
            "Per operator binding 2026-07-15 concurrent Modal jobs require an "
            "EXPLICIT operator override quoted in the claim notes — record it "
            "there NOW. Conflicts:\n  - " + "\n  - ".join(findings),
            file=sys.stderr,
        )
        return findings
    raise ModalSingleFlightRefusal(findings)


# ─────────────────────────────────────────────────────────────────────────
# Dual-ledger terminality (invoked by update_call_id_outcome)
# ─────────────────────────────────────────────────────────────────────────


def dual_ledger_terminality_blockers(
    *,
    call_id: str,
    label: str | None = None,
    lane_id: str | None = None,
    repo_root: Path | str | None = None,
) -> list[str]:
    """Blockers for a TERMINAL call_id whose claims-file row is still active.

    Operator binding 2026-07-15 rule 3: whoever observes a terminal state
    appends BOTH the ledger outcome row AND the claim terminal row in the same
    turn. This checker runs after the ledger half lands and reports the claims
    half still owed. Matching is call-id-first: if an active claim names any
    Modal call ID, only that exact terminal call ID can match it. Label/lane
    fallback is reserved for legacy claim rows that contain no call ID. This
    prevents an older terminal attempt from matching a newer live attempt on
    the same lane and emitting a false blocker.
    """
    blockers: list[str] = []
    fallback_tokens = [t.strip().lower() for t in (label or "", lane_id or "") if isinstance(t, str) and t.strip()]
    call_id_l = call_id.strip().lower()
    if not call_id_l and not fallback_tokens:
        return []
    for claim in active_modal_claims(repo_root=repo_root):
        claim_text = (f"{claim['lane_id']} {claim['instance_job_id']} {claim['notes']}").lower()
        claim_call_ids = {match.group(0).lower() for match in _MODAL_CALL_ID_RE.finditer(claim_text)}
        if claim_call_ids:
            matched = bool(call_id_l) and call_id_l in claim_call_ids
        else:
            matched = any(tok in claim_text for tok in fallback_tokens)
        if matched:
            blockers.append(
                f"claims file still ACTIVE for terminal call_id={call_id}: "
                f"lane={claim['lane_id']} job={claim['instance_job_id']} "
                f"status={claim['status']} — append the terminal claim row NOW "
                "(same turn): .venv/bin/python tools/claim_lane_dispatch.py "
                f"claim --force --lane-id {claim['lane_id']} --platform modal "
                f"--instance-job-id {claim['instance_job_id']} --agent <agent> "
                "--status completed_or_failed_... --notes '<outcome>'"
            )
    return blockers


def emit_dual_ledger_terminality_blocker_if_needed(
    *,
    record: dict[str, Any],
    ledger_path: Path | None = None,
) -> list[str]:
    """Print a LOUD blocker when a terminal outcome row leaves an active claim.

    Called from ``call_id_ledger.update_call_id_outcome`` (fail-quiet wrapper
    there; the ledger write has already succeeded). ``ledger_path`` (when the
    caller wrote to a non-default ledger, e.g. tests) locates the repo root as
    ``<root>/.omx/state/<ledger>`` so the sibling claims file is found.
    """
    status = str(record.get("status") or record.get("event_type") or "").strip().lower()
    if status in _NON_TERMINAL_LEDGER_STATES:
        return []
    repo_root: Path | None = None
    if ledger_path is not None:
        parent = Path(ledger_path).resolve().parent
        # .../<root>/.omx/state/<file> → root is 2 levels above `state`
        if parent.name == "state" and parent.parent.name == ".omx":
            repo_root = parent.parent.parent
    blockers = dual_ledger_terminality_blockers(
        call_id=str(record.get("call_id") or ""),
        label=record.get("label"),
        lane_id=record.get("lane_id"),
        repo_root=repo_root,
    )
    for blocker in blockers:
        print(
            f"[modal-dual-ledger] ⛔ BLOCKER (operator binding 2026-07-15): {blocker}",
            file=sys.stderr,
        )
    return blockers
