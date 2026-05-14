# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "configure_lightning_ssh.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("configure_lightning_ssh_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["configure_lightning_ssh_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_block_uses_hardened_lightning_options():
    mod = _load_module()
    block = mod.render_block(alias="scratch-studio-devbox", user="s_abc")
    assert "Host scratch-studio-devbox" in block
    assert "HostName ssh.lightning.ai" in block
    assert "User s_abc" in block
    assert "BatchMode yes" in block
    assert "PasswordAuthentication no" in block
    assert "KbdInteractiveAuthentication no" in block
    assert "ConnectTimeout 20" in block
    assert "ConnectionAttempts 3" in block
    assert "ServerAliveInterval 15" in block
    assert "ServerAliveCountMax 4" in block
    assert "TCPKeepAlive yes" in block
    assert "ControlMaster auto" in block
    assert "StrictHostKeyChecking accept-new" in block
    assert "UserKnownHostsFile ~/.ssh/lightning_known_hosts" in block
    assert "StrictHostKeyChecking no" not in block
    assert "UserKnownHostsFile /dev/null" not in block


def test_render_block_rejects_bare_host_alias():
    mod = _load_module()
    with pytest.raises(ValueError, match="bare ssh.lightning.ai"):
        mod.render_block(alias="ssh.lightning.ai", user="s_abc")


def test_replace_managed_block_is_idempotent():
    mod = _load_module()
    first = mod.render_block(alias="lightning-pact", user="s_one")
    second = mod.render_block(alias="lightning-pact", user="s_two")
    existing = "Host unrelated\n  HostName example.com\n\n" + first + "\nHost tail\n  HostName tail\n"
    updated = mod.replace_managed_block(existing, second)
    assert "User s_one" not in updated
    assert "User s_two" in updated
    assert "Host unrelated" in updated
    assert "Host tail" in updated
    assert updated.count(mod.BEGIN) == 1
    assert updated.count(mod.END) == 1


def test_prune_duplicate_host_stanzas_keeps_managed_block():
    mod = _load_module()
    block = mod.render_block(alias="scratch-studio-devbox", user="s_one")
    existing = (
        block
        + "\nHost scratch-studio-devbox lightning-pact\n"
        + "  HostName ssh.lightning.ai\n"
        + "  StrictHostKeyChecking no\n"
        + "\nHost unrelated\n"
        + "  HostName example.com\n"
    )

    updated = mod.prune_duplicate_host_stanzas(existing, alias="scratch-studio-devbox")

    assert updated.count("Host scratch-studio-devbox") == 1
    assert "StrictHostKeyChecking no" not in updated
    assert "Host unrelated" in updated
    assert updated.count(mod.BEGIN) == 1
    assert updated.count(mod.END) == 1
