#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""arch_shrink_x0.4_quantizr_class — Lightning Job harvest companion.

Polls the Lightning Studio Job submitted by
``experiments/arch_shrink_x0.4_lightning_full.py``, downloads the trained
archive + ``contest_auth_eval.json`` once the Job lands, emits a
``[contest-CUDA]`` evidence row to
``reports/cathedral_autopilot_evidence.jsonl``, and updates the dispatch
claim ledger to a terminal status (``completed_score_*`` or
``failed_*``).

Workflow
--------
1. Read ``.omx/state/lightning_active_jobs.json``; locate the row for
   ``--job-name`` (or the most recent ``arch_shrink_x0.4_lightning`` row
   if ``--job-name`` omitted).
2. Resolve the Lightning Job via ``lightning_sdk.Job(name=..., ...)``;
   poll status with ``--poll-interval-sec``.  ``--once`` returns
   immediately after a single status check.
3. On terminal status (``completed`` / ``stopped`` / ``failed``):
   a. rsync ``experiments/results/lightning_batch/<job_name>/`` via
      ``--ssh-target``.  The harvester tries both the live Studio checkout and
      the persisted ``/teamspace/jobs/<sdk-job>/artifacts/pact/...`` mirror.
   b. Parse ``contest_auth_eval.json`` (the Job emits it from the
      ``RESULT_JSON`` line in ``auth_eval.log``).
   c. Append a ``[contest-CUDA]`` evidence row to
      ``reports/cathedral_autopilot_evidence.jsonl``.
   d. File a terminal claim row via ``tools/claim_lane_dispatch.py
      claim --force --status completed_score_<score>`` (or
      ``failed_<reason>``).
4. Mark the active-jobs row as terminal in
   ``.omx/state/lightning_active_jobs.json`` so future harvests skip it.

CLAUDE.md compliance
--------------------
- ``[contest-CUDA]`` tag is emitted ONLY after the auth-eval JSON parses
  with a numeric score and the archive bytes match the Job's reported
  archive.
- Forensic rsync uses the canonical Lightning SSH target; never bare
  ``ssh.lightning.ai``.
- Terminal claim status uses ``completed_score_<rounded>`` so the
  ledger reflects an actual measurement.

Usage
-----
.. code-block:: bash

    # Poll continuously until the Job lands:
    .venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py

    # Single-shot check:
    .venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --once
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.lightning.batch_jobs import lightning_sdk_job_name
from tac.deploy.lightning.defaults import (
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_remote_pact,
    default_ssh_target,
    default_teamspace,
    default_user,
)
from tac.deploy.lightning.harvest_env import (
    LightningHarvestRsyncError,
    require_lightning_harvest_values,
    rsync_progress_args,
)
from tac.deploy.lightning.round3_harvest import (
    auth_eval_archive_size,
    auth_eval_posenet_distortion,
    auth_eval_rate,
    auth_eval_score,
    auth_eval_segnet_distortion,
    classify_lightning_no_score_failure,
    lightning_round3_remote_artifact_dirs,
    require_contest_cuda_auth_eval,
    sha256_file,
    write_no_score_failure_classification,
)

LANE_ID = "arch_shrink_x0.4_lightning"
LIGHTNING_ACTIVE_JOBS_PATH = REPO_ROOT / ".omx" / "state" / "lightning_active_jobs.json"
DEFAULT_EVIDENCE_OUT = REPO_ROOT / "reports" / "cathedral_autopilot_evidence.jsonl"
DEFAULT_POLL_INTERVAL_SEC = 300  # 5 minutes
TERMINAL_STATUSES = {"completed", "succeeded", "failed", "stopped", "cancelled"}


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _load_active_jobs() -> list[dict[str, object]]:
    """Read-only snapshot of the active jobs.

    Safe outside the lock for read-only inspection; ``os.replace`` semantics
    in the locked-write path guarantee no partial reads. For
    load→mutate→save cycles, use
    :func:`tac.deploy.lightning.active_jobs_state.update_active_jobs_locked`
    instead — see ``_mark_row_terminal_locked`` below.
    """
    # SAVE_POSTERIOR_LOCKED_OK: read-only path; the corresponding write
    # goes through ``_mark_row_terminal_locked`` which uses fcntl serialization.
    if not LIGHTNING_ACTIVE_JOBS_PATH.exists():
        return []
    try:
        rows = json.loads(LIGHTNING_ACTIVE_JOBS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"FATAL: corrupt {LIGHTNING_ACTIVE_JOBS_PATH}: {exc}")
    if not isinstance(rows, list):
        return []
    return rows


def _mark_row_terminal_locked(
    *,
    job_name: str,
    status_lower: str,
    score: float | None,
) -> None:
    """Atomic mark-terminal under fcntl lock — sister-dispatcher safe.

    Routes through
    :func:`tac.deploy.lightning.active_jobs_state.update_active_jobs_locked`
    (catalog #131) so the load→mutate→save cycle is atomic. Without this,
    a concurrent dispatcher writing a NEW row would see a stale read here
    and silently drop the new row when this writer's ``os.replace`` lands.
    """
    from tac.deploy.lightning.active_jobs_state import update_active_jobs_locked

    def _mutate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        _mark_row_terminal(
            rows,
            job_name=job_name,
            status_lower=status_lower,
            score=score,
        )
        return rows

    update_active_jobs_locked(_mutate)


def _select_target_row(
    rows: list[dict[str, object]], *, job_name: str | None
) -> dict[str, object] | None:
    candidates = [
        r for r in rows
        if r.get("lane_id") == LANE_ID
        and r.get("terminal_status") is None
    ]
    if job_name is not None:
        for r in candidates:
            if r.get("job_name") == job_name:
                return r
        # Fall through: also allow re-poll of already-terminal row by name.
        for r in rows:
            if r.get("job_name") == job_name:
                return r
        return None
    if not candidates:
        return None
    # Most recent submission first.
    candidates.sort(key=lambda r: str(r.get("submitted_at_utc", "")), reverse=True)
    return candidates[0]


def _resolve_lightning_job(*, name: str, teamspace: str, user: str) -> object:
    """Resolve a Job handle via Teamspace.jobs iteration.

    Direct ``Job(name=...)`` construction returns a 400 on the current SDK
    (2026.05.06post2) for newly-created Jobs that haven't propagated through
    the API. Teamspace.jobs iteration is the working alternative — it
    enumerates the live jobs and we match on the SDK-canonical name.
    """
    os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
    try:
        from lightning_sdk import Teamspace  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - env-dependent
        sys.exit(
            f"FATAL: lightning_sdk import failed; install via "
            f"`uv pip install lightning-sdk` ({exc})"
        )
    sdk_name = lightning_sdk_job_name(name).lower()
    try:
        ts = Teamspace(name=teamspace, user=user)
    except Exception as exc:  # pragma: no cover - SDK runtime
        sys.exit(f"FATAL: cannot resolve Teamspace {teamspace!r}: {exc}")
    for job in ts.jobs:
        if str(getattr(job, "name", "")).lower() == sdk_name:
            return job
    sys.exit(
        f"FATAL: no Lightning Job named {sdk_name!r} found under "
        f"teamspace={teamspace!r} user={user!r}"
    )


def _job_status_lower(job: object) -> str:
    raw = getattr(job, "status", None)
    if raw is None:
        return "unknown"
    return str(raw).split(".")[-1].lower()


def _rsync_artifacts(
    *,
    ssh_target: str,
    remote_pact: str,
    job_name: str,
) -> Path:
    """rsync live Studio and persisted SDK artifact dirs back to local."""
    if not shutil.which("rsync"):
        sys.exit("FATAL: rsync not on PATH; required for Lightning artifact harvest")
    local_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, str]] = []
    successes: list[str] = []
    for remote_dir in lightning_round3_remote_artifact_dirs(
        remote_pact=remote_pact,
        job_name=job_name,
    ):
        remote_path = f"{ssh_target}:{remote_dir}"
        cmd = [
            "rsync",
            "-az",
            *rsync_progress_args("rsync"),
            "-e",
            "ssh -o ConnectTimeout=30 -o ServerAliveInterval=30",
            remote_path,
            str(local_dir) + "/",
        ]
        print(f"[harvest] rsync {remote_path} -> {local_dir}/")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if result.returncode == 0:
            successes.append(remote_path)
            continue
        failures.append((result.returncode, remote_path))
        print(
            f"[harvest] rsync fallback candidate failed rc={result.returncode}: "
            f"{remote_path}"
        )
    if not successes:
        rc, remote_path = failures[-1] if failures else (1, "<no artifact roots>")
        if len(failures) > 1:
            remote_path = "; ".join(path for _rc, path in failures)
        raise LightningHarvestRsyncError(
            returncode=rc,
            remote_path=remote_path,
        )
    return local_dir


def _parse_auth_eval_json(local_dir: Path) -> dict[str, object] | None:
    """Locate and parse the contest_auth_eval.json artifact."""
    candidates = [
        local_dir / "contest_auth_eval.json",
        *sorted(local_dir.glob("**/contest_auth_eval*.json")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[harvest] WARNING: corrupt {candidate}: {exc}")
                continue
            if isinstance(data, dict):
                data["_auth_eval_json_path"] = str(candidate)
                return data
    return None


def _archive_path_stats(archive_path: Path | None) -> tuple[int | None, str | None]:
    if archive_path is None or not archive_path.is_file():
        return None, None
    return archive_path.stat().st_size, sha256_file(archive_path)


def _local_archive_path(local_dir: Path) -> Path | None:
    candidates = [
        local_dir / "archive.zip",
        local_dir / "auth_eval_work" / "archive.zip",
        *sorted(local_dir.glob("*.zip")),
        *sorted(local_dir.glob("**/archive.zip")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _emit_evidence_row(
    *,
    evidence_out: Path,
    auth_eval: dict[str, object],
    job_name: str,
    archive_path: Path | None,
) -> dict[str, object]:
    """Append a [contest-CUDA] row to the cathedral_autopilot evidence JSONL."""
    score = auth_eval_score(auth_eval)
    if not isinstance(score, (int, float)):
        raise SystemExit(
            "FATAL: refusing [contest-CUDA] evidence row without numeric auth_eval score"
        )
    archive_bytes = auth_eval_archive_size(auth_eval)
    if not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise SystemExit(
            "FATAL: refusing [contest-CUDA] evidence row without positive integer archive_bytes"
        )
    if archive_path is None or not archive_path.is_file():
        raise SystemExit(
            "FATAL: refusing [contest-CUDA] evidence row without local archive.zip custody"
        )
    expected_archive_bytes, expected_archive_sha256 = _archive_path_stats(archive_path)
    require_contest_cuda_auth_eval(
        auth_eval,
        expected_archive_bytes=expected_archive_bytes,
        expected_archive_sha256=expected_archive_sha256,
    )
    provenance = auth_eval.get("provenance") if isinstance(auth_eval.get("provenance"), dict) else {}
    archive_sha256 = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    auth_eval_json_path = auth_eval.get("_auth_eval_json_path")
    if isinstance(auth_eval_json_path, str) and auth_eval_json_path:
        auth_eval_json_display = str(_display_path(Path(auth_eval_json_path)))
    else:
        auth_eval_json_display = (
            f"experiments/results/lightning_batch/{job_name}/contest_auth_eval.json"
        )
    row = {
        "technique": "arch_shrink_x0.4_quantizr_class",
        "lane_id": LANE_ID,
        "job_name": job_name,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": False,
        "promotion_eligible": False,
        "contest_dispatch_verdict": "completed",
        "score_contest_cuda": score,
        "empirical_archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_bytes_match_expected": expected_archive_bytes is not None,
        "archive_sha256_match_expected": expected_archive_sha256 is not None,
        "posenet_distortion": auth_eval_posenet_distortion(auth_eval),
        "segnet_distortion": auth_eval_segnet_distortion(auth_eval),
        "rate": auth_eval_rate(auth_eval),
        "archive_path": str(archive_path) if archive_path is not None else None,
        "source": (
            f"[contest-CUDA] Lightning T4 g4dn.2xlarge job={job_name} "
            f"profile=q_faithful_dilated_88k auth_eval_json="
            f"{auth_eval_json_display}"
        ),
        "timestamp": _utc_now_iso(),
    }
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    with evidence_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[evidence] appended row to {_display_path(evidence_out)}")
    return row


def _terminal_claim(
    *,
    job_name: str,
    status: str,
    notes: str,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        LANE_ID,
        "--agent",
        "claude_lab",
        "--platform",
        "lightning",
        "--instance-job-id",
        job_name,
        "--status",
        status,
        "--notes",
        notes,
        "--force",
    ]
    print(f"[terminal-claim] status={status} job={job_name}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"terminal claim failed for job={job_name} status={status}: "
            f"claim_lane_dispatch.py rc={result.returncode}"
        )


def _mark_row_terminal(
    rows: list[dict[str, object]],
    *,
    job_name: str,
    status_lower: str,
    score: float | None,
) -> None:
    for r in rows:
        if r.get("job_name") == job_name:
            r["terminal_status"] = status_lower
            r["terminated_at_utc"] = _utc_now_iso()
            if score is not None:
                r["score_contest_cuda"] = score
            break


def _close_terminal_failure(
    *,
    job_name: str,
    status: str,
    notes: str,
) -> None:
    _terminal_claim(job_name=job_name, status=status, notes=notes)
    _mark_row_terminal_locked(job_name=job_name, status_lower=status, score=None)
    sys.exit(notes)


def _harvest_terminal(
    *,
    target: dict[str, object],
    args: argparse.Namespace,
) -> int:
    job_name = str(target["job_name"])
    print(f"[harvest] Job {job_name} reached terminal status; rsync + parse + emit")
    require_lightning_harvest_values(
        ssh_target=args.ssh_target,
        remote_pact=args.remote_pact,
        require_provider=False,
        require_rsync=True,
        context="artifact-rsync",
    )

    try:
        local_dir = _rsync_artifacts(
            ssh_target=args.ssh_target,
            remote_pact=args.remote_pact,
            job_name=job_name,
        )
    except LightningHarvestRsyncError as exc:
        status = f"failed_artifact_rsync_rc_{exc.returncode}"
        notes = (
            f"harvest: terminal Lightning job but artifact rsync failed rc={exc.returncode}; "
            f"remote={exc.remote_path}"
        )
        _terminal_claim(job_name=job_name, status=status, notes=notes)
        _mark_row_terminal_locked(job_name=job_name, status_lower=status, score=None)
        sys.exit(f"FATAL: {notes}")

    auth_eval = _parse_auth_eval_json(local_dir)
    if auth_eval is None:
        classification = classify_lightning_no_score_failure(local_dir)
        classification_path = write_no_score_failure_classification(
            local_dir,
            classification,
        )
        status = str(classification["terminal_status"])
        notes = (
            f"harvest: no contest_auth_eval.json found in {local_dir}; "
            f"failure_class={classification['failure_class']} "
            f"failure_domain={classification['failure_domain']} "
            f"classification={_display_path(classification_path)}"
        )
        _close_terminal_failure(job_name=job_name, status=status, notes=notes)

    archive_path = _local_archive_path(local_dir)

    try:
        row = _emit_evidence_row(
            evidence_out=args.evidence_out,
            auth_eval=auth_eval,
            job_name=job_name,
            archive_path=archive_path,
        )
    except SystemExit as exc:
        reason = str(exc) or "strict auth-eval custody validation failed"
        notes = (
            "harvest: contest_auth_eval.json present but rejected before evidence emission; "
            f"reason={reason}; artifact=experiments/results/lightning_batch/{job_name}/"
            "contest_auth_eval.json"
        )
        _close_terminal_failure(
            job_name=job_name,
            status="failed_invalid_auth_eval_custody",
            notes=notes,
        )

    score = auth_eval_score(auth_eval)
    if isinstance(score, (int, float)):
        score_tag = f"{float(score):.6f}".rstrip("0").rstrip(".")
        terminal_status = f"completed_score_{score_tag}"
    else:
        terminal_status = "completed_score_unknown"

    notes = (
        f"contest-CUDA score={score} archive_bytes={row.get('empirical_archive_bytes')} "
        f"artifact=experiments/results/lightning_batch/{job_name}/contest_auth_eval.json"
    )
    _terminal_claim(job_name=job_name, status=terminal_status, notes=notes)

    _mark_row_terminal_locked(
        job_name=job_name,
        status_lower=terminal_status,
        score=float(score) if isinstance(score, (int, float)) else None,
    )

    print(json.dumps(row, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--job-name",
        default=None,
        help="Lightning Job name (default: most recent active arch_shrink row)",
    )
    p.add_argument(
        "--ssh-target",
        default=default_ssh_target(),
        help="Lightning Studio SSH target (defaults to $LIGHTNING_SSH_TARGET / "
             "$LIGHTNING_REMOTE / $REMOTE)",
    )
    p.add_argument(
        "--remote-pact",
        default=default_remote_pact(),
        help=f"Remote pact dir (default $LIGHTNING_REMOTE_PACT or "
             f"{DEFAULT_LIGHTNING_REMOTE_PACT})",
    )
    p.add_argument(
        "--teamspace",
        default=default_teamspace(),
        help="Lightning teamspace name (default $LIGHTNING_TEAMSPACE)",
    )
    p.add_argument(
        "--user",
        default=default_user(),
        help="Lightning user (default $LIGHTNING_USER)",
    )
    p.add_argument(
        "--evidence-out",
        type=Path,
        default=DEFAULT_EVIDENCE_OUT,
        help="JSONL evidence sink (default reports/cathedral_autopilot_evidence.jsonl)",
    )
    p.add_argument(
        "--poll-interval-sec",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Polling interval in seconds (default {DEFAULT_POLL_INTERVAL_SEC})",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Single status check; do not loop",
    )
    p.add_argument(
        "--force-harvest",
        action="store_true",
        help=(
            "Skip the Lightning Job status check and harvest immediately. "
            "Use only if the Job is known terminal but the SDK status query "
            "fails."
        ),
    )
    args = p.parse_args(argv)

    rows = _load_active_jobs()
    target = _select_target_row(rows, job_name=args.job_name)
    if target is None:
        if args.job_name:
            sys.exit(
                f"FATAL: no row in {LIGHTNING_ACTIVE_JOBS_PATH} for "
                f"job_name={args.job_name!r}"
            )
        sys.exit(
            f"FATAL: no active arch_shrink_x0.4_lightning row in "
            f"{LIGHTNING_ACTIVE_JOBS_PATH}"
        )

    job_name = str(target["job_name"])
    print(f"[harvest] target job_name={job_name}")

    if args.force_harvest:
        return _harvest_terminal(target=target, args=args)

    require_lightning_harvest_values(
        teamspace=args.teamspace,
        user=args.user,
        require_provider=True,
        require_rsync=False,
        context="provider",
    )

    while True:
        job = _resolve_lightning_job(
            name=job_name,
            teamspace=args.teamspace,
            user=args.user,
        )
        status_lower = _job_status_lower(job)
        print(f"[harvest] {job_name} status={status_lower} ({_utc_now_iso()})")
        if status_lower in TERMINAL_STATUSES:
            return _harvest_terminal(target=target, args=args)
        if args.once:
            print(f"[harvest] --once set; status={status_lower}; exiting non-terminal")
            return 0
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
