#!/usr/bin/env python3
"""Install a hardened Lightning AI SSH alias.

This intentionally replaces the ad hoc Lightning UI one-liner for contest
work. The UI helper is convenient, but it may write permissive host-key policy.
This script writes a small managed block that is safe for reproducible staging,
Batch Job preflights, and artifact harvests.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ALIAS = "scratch-studio-devbox"
DEFAULT_HOST = "ssh.lightning.ai"  # HostName ssh.lightning.ai — canonical alias installer constant
DEFAULT_IDENTITY_FILE = "~/.ssh/lightning_rsa"
DEFAULT_KNOWN_HOSTS = "~/.ssh/lightning_known_hosts"
BEGIN = "# >>> pact lightning ssh alias >>>"
END = "# <<< pact lightning ssh alias <<<"


def _require_clean_token(value: str, *, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} is required")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{label} must not contain whitespace")
    if any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def render_block(
    *,
    alias: str,
    user: str,
    host: str = DEFAULT_HOST,
    identity_file: str = DEFAULT_IDENTITY_FILE,
    known_hosts: str = DEFAULT_KNOWN_HOSTS,
    connect_timeout: int = 20,
    connection_attempts: int = 3,
    server_alive_interval: int = 15,
    server_alive_count_max: int = 4,
    control_persist: str = "10m",
) -> str:
    alias = _require_clean_token(alias, label="alias")
    user = _require_clean_token(user, label="user")
    host = _require_clean_token(host, label="host")
    identity_file = _require_clean_token(identity_file, label="identity_file")
    known_hosts = _require_clean_token(known_hosts, label="known_hosts")
    if host == DEFAULT_HOST and alias == DEFAULT_HOST:
        raise ValueError("invalid alias — not bare ssh.lightning.ai is permitted; use a config alias")
    if connect_timeout <= 0:
        raise ValueError("connect_timeout must be positive")
    if connection_attempts <= 0:
        raise ValueError("connection_attempts must be positive")
    if server_alive_interval <= 0:
        raise ValueError("server_alive_interval must be positive")
    if server_alive_count_max <= 0:
        raise ValueError("server_alive_count_max must be positive")
    lines = [
        BEGIN,
        f"Host {alias}",
        f"  HostName {host}",
        f"  User {user}",
        f"  IdentityFile {identity_file}",
        "  IdentitiesOnly yes",
        "  BatchMode yes",
        "  PreferredAuthentications publickey",
        "  PubkeyAuthentication yes",
        "  PasswordAuthentication no",
        "  KbdInteractiveAuthentication no",
        f"  ConnectTimeout {connect_timeout}",
        f"  ConnectionAttempts {connection_attempts}",
        f"  ServerAliveInterval {server_alive_interval}",
        f"  ServerAliveCountMax {server_alive_count_max}",
        "  TCPKeepAlive yes",
        "  ControlMaster auto",
        "  ControlPath ~/.ssh/cm/lightning-%r@%h:%p",
        f"  ControlPersist {control_persist}",
        "  StrictHostKeyChecking accept-new",
        f"  UserKnownHostsFile {known_hosts}",
        END,
        "",
    ]
    return "\n".join(lines)


def replace_managed_block(existing: str, block: str) -> str:
    start = existing.find(BEGIN)
    end = existing.find(END)
    if start == -1 and end == -1:
        suffix = existing.lstrip("\n")
        return block + ("\n" + suffix if suffix else "")
    if start == -1 or end == -1 or end < start:
        raise ValueError("ssh config contains a partial pact lightning managed block")
    end += len(END)
    suffix_start = end
    if suffix_start < len(existing) and existing[suffix_start : suffix_start + 1] == "\n":
        suffix_start += 1
    prefix = existing[:start].rstrip()
    suffix = existing[suffix_start:].lstrip("\n")
    out = (prefix + "\n\n" if prefix else "") + block
    if suffix:
        out += "\n" + suffix
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default=os.environ.get("LIGHTNING_SSH_ALIAS", DEFAULT_ALIAS))
    parser.add_argument("--user", default=os.environ.get("LIGHTNING_USER"))
    parser.add_argument("--host", default=os.environ.get("LIGHTNING_HOST", DEFAULT_HOST))
    parser.add_argument("--identity-file", default=os.environ.get("LIGHTNING_IDENTITY_FILE", DEFAULT_IDENTITY_FILE))
    parser.add_argument("--known-hosts", default=os.environ.get("LIGHTNING_KNOWN_HOSTS", DEFAULT_KNOWN_HOSTS))
    parser.add_argument("--config", default=str(Path.home() / ".ssh" / "config"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.user:
        raise SystemExit("--user or LIGHTNING_USER is required")
    block = render_block(
        alias=args.alias,
        user=args.user,
        host=args.host,
        identity_file=args.identity_file,
        known_hosts=args.known_hosts,
    )
    config = Path(args.config).expanduser()
    existing = config.read_text() if config.exists() else ""
    updated = replace_managed_block(existing, block)
    if args.dry_run:
        print(updated, end="")
        return 0
    config.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    config.write_text(updated)
    config.chmod(0o600)
    cm_dir = Path.home() / ".ssh" / "cm"
    cm_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    print(f"LIGHTNING_SSH_ALIAS_CONFIGURED alias={args.alias} config={config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
