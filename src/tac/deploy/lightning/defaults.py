# SPDX-License-Identifier: MIT
"""Lightning Studio environment defaults.

Public code must not carry operator-specific Studio names, user IDs, or direct
SSH targets. Command surfaces should import this module and then fail loudly
when the required environment variable or CLI flag is missing.
"""

from __future__ import annotations

import os

DEFAULT_LIGHTNING_USER = ""
DEFAULT_LIGHTNING_TEAMSPACE = ""
DEFAULT_LIGHTNING_STUDIO = ""
DEFAULT_LIGHTNING_SSH_TARGET = ""
DEFAULT_LIGHTNING_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
DEFAULT_LIGHTNING_REMOTE_TAC = "/teamspace/studios/this_studio/tac"


def default_ssh_target() -> str:
    """Return the preferred Studio SSH target, honoring legacy env overrides."""
    return (
        os.environ.get("LIGHTNING_SSH_TARGET")
        or os.environ.get("LIGHTNING_REMOTE")
        or os.environ.get("REMOTE")
        or DEFAULT_LIGHTNING_SSH_TARGET
    )


def default_studio() -> str:
    return os.environ.get("LIGHTNING_STUDIO", DEFAULT_LIGHTNING_STUDIO)


def default_teamspace() -> str:
    return os.environ.get("LIGHTNING_TEAMSPACE", DEFAULT_LIGHTNING_TEAMSPACE)


def default_user() -> str:
    return os.environ.get("LIGHTNING_USER", DEFAULT_LIGHTNING_USER)


def default_remote_pact() -> str:
    return os.environ.get("LIGHTNING_REMOTE_PACT", DEFAULT_LIGHTNING_REMOTE_PACT)


def default_remote_tac() -> str:
    return os.environ.get("LIGHTNING_REMOTE_TAC", DEFAULT_LIGHTNING_REMOTE_TAC)


__all__ = [
    "DEFAULT_LIGHTNING_REMOTE_PACT",
    "DEFAULT_LIGHTNING_REMOTE_TAC",
    "DEFAULT_LIGHTNING_SSH_TARGET",
    "DEFAULT_LIGHTNING_STUDIO",
    "DEFAULT_LIGHTNING_TEAMSPACE",
    "DEFAULT_LIGHTNING_USER",
    "default_remote_pact",
    "default_remote_tac",
    "default_ssh_target",
    "default_studio",
    "default_teamspace",
    "default_user",
]
