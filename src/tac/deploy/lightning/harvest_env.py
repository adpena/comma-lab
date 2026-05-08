"""Fail-closed environment checks for Lightning harvest companions."""

from __future__ import annotations

from collections.abc import Sequence


def _is_missing(value: str | None) -> bool:
    return value is None or not str(value).strip()


def _format_missing(items: Sequence[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


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
    "missing_lightning_harvest_values",
    "require_lightning_harvest_values",
]
