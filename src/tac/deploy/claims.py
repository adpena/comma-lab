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


def dispatch_claim_command(
    *,
    spec: DispatchClaimSpec,
    status: str,
    python_executable: str = ".venv/bin/python",
    claim_tool: str | Path = "tools/claim_lane_dispatch.py",
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
) -> None:
    """Execute the canonical lane-claim command and fail closed on conflicts."""

    cmd = dispatch_claim_command(
        spec=spec,
        status=status,
        python_executable=python_executable,
        claim_tool=claim_tool,
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
    )


__all__ = [
    "DispatchClaimSpec",
    "dispatch_claim_command",
    "predicted_eta",
    "record_dispatch_claim",
    "terminal_dispatch_claim",
    "utc_now",
]
