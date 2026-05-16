# SPDX-License-Identifier: MIT
"""Provider-neutral lane-dispatch claim helpers.

Cloud actuators own provider APIs. This module owns the shared command shape
for the mandatory ``tools/claim_lane_dispatch.py claim`` guard so Modal,
Vast.ai, Lightning, Kaggle, Azure, AWS, and GCP wrappers do not drift on lane
custody semantics.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

TERMINAL_PREFIXES = (
    "completed_",
    "failed_",
    "timed_out",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)


@dataclass(frozen=True)
class DispatchClaimSpec:
    """Metadata required to open or close a remote dispatch claim."""

    lane_id: str
    instance_job_id: str
    agent: str
    platform: str
    predicted_eta_utc: str = ""
    force: bool = False
    notes: str = ""


def utc_now() -> str:
    """Return a compact UTC timestamp for custody metadata."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def predicted_eta(hours: float = 3.0) -> str:
    """Return a conservative UTC ETA for claim rows."""

    return (
        datetime.now(UTC).replace(microsecond=0) + timedelta(hours=float(hours))
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_terminal_status(status: str) -> bool:
    """Return true when a dispatch-claim status closes a lane/job row."""

    return any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def active_claim_row(
    claims_path: Path,
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, str]:
    """Return the newest matching active claim row or raise ``ValueError``."""

    if not claims_path.is_file():
        raise ValueError(f"missing lane-claim ledger: {claims_path}")
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        row = {
            "timestamp_utc": cells[0],
            "agent": cells[1],
            "lane_id": cells[2],
            "platform": cells[3],
            "instance_job_id": cells[4],
            "predicted_eta_utc": cells[5],
            "status": cells[6],
            "notes": cells[7],
        }
        if row["lane_id"] != lane_id or row["instance_job_id"] != instance_job_id:
            continue
        if is_terminal_status(row["status"]):
            raise ValueError(
                "newest matching claim is terminal: "
                f"lane_id={lane_id} instance_job_id={instance_job_id} "
                f"status={row['status']}"
            )
        return row
    raise ValueError(
        "no active lane claim found for "
        f"lane_id={lane_id} instance_job_id={instance_job_id}"
    )


def dispatch_claim_command(
    *,
    spec: DispatchClaimSpec,
    status: str,
    python_executable: str = ".venv/bin/python",
    claim_tool: str | Path = "tools/claim_lane_dispatch.py",
    claims_path: str | Path | None = None,
    default_notes: str = "",
) -> list[str]:
    """Build the canonical lane-claim command without executing it."""

    lane_id = spec.lane_id.strip()
    instance_job_id = spec.instance_job_id.strip()
    platform = spec.platform.strip()
    agent = spec.agent.strip()
    if not lane_id:
        raise ValueError("dispatch claim requires lane_id")
    if not instance_job_id:
        raise ValueError("dispatch claim requires instance_job_id")
    if not platform:
        raise ValueError("dispatch claim requires platform")
    if not agent:
        raise ValueError("dispatch claim requires agent")
    if not status.strip():
        raise ValueError("dispatch claim requires status")

    cmd = [
        python_executable,
        str(claim_tool),
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        platform,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--predicted-eta-utc",
        spec.predicted_eta_utc or predicted_eta(),
        "--status",
        status.strip(),
        "--notes",
        spec.notes or default_notes,
    ]
    if claims_path is not None:
        cmd.extend(["--claims-path", str(claims_path)])
    if spec.force:
        cmd.append("--force")
    return cmd


def record_dispatch_claim(
    *,
    repo_root: Path,
    spec: DispatchClaimSpec,
    status: str,
    default_notes: str = "",
    python_executable: str = ".venv/bin/python",
    claim_tool: str | Path = "tools/claim_lane_dispatch.py",
    claims_path: str | Path | None = None,
) -> None:
    """Execute the canonical lane-claim command and fail closed on conflicts."""

    cmd = dispatch_claim_command(
        spec=spec,
        status=status,
        python_executable=python_executable,
        claim_tool=claim_tool,
        claims_path=claims_path,
        default_notes=default_notes,
    )
    proc = subprocess.run(cmd, cwd=repo_root, text=True, check=False)
    if proc.returncode:
        raise SystemExit(
            f"FATAL: dispatch claim failed rc={proc.returncode}; aborting before provider spend"
        )


def terminal_dispatch_claim(
    *,
    repo_root: Path,
    spec: DispatchClaimSpec,
    status: str,
    notes: str,
    python_executable: str = ".venv/bin/python",
    claim_tool: str | Path = "tools/claim_lane_dispatch.py",
    claims_path: str | Path | None = None,
) -> None:
    """Append a terminal claim row for a dispatch claim."""

    terminal = DispatchClaimSpec(
        lane_id=spec.lane_id,
        instance_job_id=spec.instance_job_id,
        agent=spec.agent,
        platform=spec.platform,
        predicted_eta_utc=utc_now(),
        force=True,
        notes=notes,
    )
    record_dispatch_claim(
        repo_root=repo_root,
        spec=terminal,
        status=status,
        python_executable=python_executable,
        claim_tool=claim_tool,
        claims_path=claims_path,
    )


__all__ = [
    "TERMINAL_PREFIXES",
    "DispatchClaimSpec",
    "active_claim_row",
    "dispatch_claim_command",
    "is_terminal_status",
    "predicted_eta",
    "record_dispatch_claim",
    "terminal_dispatch_claim",
    "utc_now",
]
