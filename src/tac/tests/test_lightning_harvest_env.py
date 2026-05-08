from __future__ import annotations

import pytest

from tac.deploy.lightning.harvest_env import (
    missing_lightning_harvest_values,
    require_lightning_harvest_values,
    rsync_progress_args_from_help,
)


def test_lightning_harvest_provider_env_fails_before_sdk_access() -> None:
    missing = missing_lightning_harvest_values(
        teamspace="",
        user=" ",
        require_provider=True,
        require_rsync=False,
    )

    assert missing == [
        "--teamspace / $LIGHTNING_TEAMSPACE",
        "--user / $LIGHTNING_USER",
    ]
    with pytest.raises(SystemExit, match="missing required Lightning provider values"):
        require_lightning_harvest_values(
            teamspace="",
            user="",
            require_provider=True,
            require_rsync=False,
            context="provider",
        )


def test_lightning_harvest_rsync_env_fails_before_artifact_custody() -> None:
    missing = missing_lightning_harvest_values(
        ssh_target="",
        remote_pact="",
        require_provider=False,
        require_rsync=True,
    )

    assert missing == [
        "--ssh-target / $LIGHTNING_SSH_TARGET",
        "--remote-pact / $LIGHTNING_REMOTE_PACT",
    ]
    with pytest.raises(SystemExit, match="missing required Lightning artifact-rsync values"):
        require_lightning_harvest_values(
            ssh_target="",
            remote_pact="",
            require_provider=False,
            require_rsync=True,
            context="artifact-rsync",
        )


def test_lightning_harvest_env_accepts_complete_values() -> None:
    require_lightning_harvest_values(
        teamspace="team",
        user="user",
        ssh_target="studio-alias",
        remote_pact="/teamspace/studios/this_studio/pact",
        require_provider=True,
        require_rsync=True,
        context="full-harvest",
    )


def test_rsync_progress_args_support_gnu_and_macos_rsync() -> None:
    assert rsync_progress_args_from_help("rsync  version 3.2.7\n    --info=FLAGS") == [
        "--info=progress2"
    ]
    assert rsync_progress_args_from_help("rsync  version 2.6.9 protocol version 29") == [
        "--progress"
    ]
