"""Fail-closed environment checks for Lightning harvest companions."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


class LightningHarvestRsyncError(RuntimeError):
    """Raised when terminal-job artifact rsync cannot complete."""

    def __init__(self, *, returncode: int, remote_path: str) -> None:
        super().__init__(f"Lightning artifact rsync failed rc={returncode}: {remote_path}")
        self.returncode = returncode
        self.remote_path = remote_path


def _is_missing(value: str | None) -> bool:
    return value is None or not str(value).strip()


def _format_missing(items: Sequence[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def rsync_progress_args_from_help(help_text: str) -> list[str]:
    """Return progress args supported by the local rsync build."""
    if "--info=" in help_text or "--info=FLAGS" in help_text:
        return ["--info=progress2"]
    return ["--progress"]


def rsync_progress_args(rsync_path: str = "rsync") -> list[str]:
    """Return deterministic progress args for GNU rsync and macOS rsync."""
    result = subprocess.run(
        [rsync_path, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    return rsync_progress_args_from_help(
        "\n".join(part for part in (result.stdout, result.stderr) if part)
    )


def missing_lightning_harvest_values(
    *,
    teamspace: str | None = None,
    user: str | None = None,
    ssh_target: str | None = None,
    remote_pact: str | None = None,
    require_provider: bool,
    require_rsync: bool,
) -> list[str]:
    """Return missing CLI/env knobs needed for Lightning harvest work.

    ``require_provider`` protects calls into ``lightning_sdk.Teamspace``.
    ``require_rsync`` protects artifact custody once a job reaches a terminal
    state or the operator uses a force-harvest path.
    """
    missing: list[str] = []
    if require_provider:
        if _is_missing(teamspace):
            missing.append("--teamspace / $LIGHTNING_TEAMSPACE")
        if _is_missing(user):
            missing.append("--user / $LIGHTNING_USER")
    if require_rsync:
        if _is_missing(ssh_target):
            missing.append("--ssh-target / $LIGHTNING_SSH_TARGET")
        if _is_missing(remote_pact):
            missing.append("--remote-pact / $LIGHTNING_REMOTE_PACT")
    return missing


def require_lightning_harvest_values(
    *,
    teamspace: str | None = None,
    user: str | None = None,
    ssh_target: str | None = None,
    remote_pact: str | None = None,
    require_provider: bool,
    require_rsync: bool,
    context: str,
) -> None:
    """Raise ``SystemExit`` before opaque SDK/rsync failures can occur."""
    missing = missing_lightning_harvest_values(
        teamspace=teamspace,
        user=user,
        ssh_target=ssh_target,
        remote_pact=remote_pact,
        require_provider=require_provider,
        require_rsync=require_rsync,
    )
    if missing:
        raise SystemExit(
            f"FATAL: missing required Lightning {context} values:\n"
            f"{_format_missing(missing)}"
        )


__all__ = [
    "LightningHarvestRsyncError",
    "missing_lightning_harvest_values",
    "require_lightning_harvest_values",
    "rsync_progress_args",
    "rsync_progress_args_from_help",
]
