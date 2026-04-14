#!/usr/bin/env python3
"""Canonical bat00 interaction script — handles PowerShell quoting correctly.

Runs commands on bat00 (Windows + WSL2) via Tailscale SSH without
quoting hell. All commands go through this script.

Usage:
    # PowerShell commands:
    python scripts/bat00.py ps "hostname"
    python scripts/bat00.py ps "nvidia-smi"
    python scripts/bat00.py ps "Get-Process python*"

    # WSL2 bash commands (preferred for training):
    python scripts/bat00.py wsl "hostname && nvidia-smi && python3 --version"
    python scripts/bat00.py wsl "cd ~/pact && git pull"
    python scripts/bat00.py wsl "cd ~/pact && nohup scripts/bat00_runner.sh > /dev/null 2>&1 &"

    # Upload a script and run it:
    python scripts/bat00.py upload scripts/bat00_runner.sh
    python scripts/bat00.py run-wsl "bash ~/bat00_runner.sh"

    # Status check:
    python scripts/bat00.py status

    # Sync repo to bat00:
    python scripts/bat00.py sync
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BAT00_IP = "100.120.99.124"
BAT00_USER = "adpena"
# Port 22 = Windows OpenSSH (PowerShell) — fragile, rate-limited
# Port 2222 = WSL2 sshd (Linux bash) — reliable, preferred
WIN_PORT = 22
WSL_PORT = 2222

def _ssh_opts(port: int = WSL_PORT) -> list[str]:
    return [
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=5",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
    ]

REPO_ROOT = Path(__file__).resolve().parent.parent


def ssh_target() -> str:
    return f"{BAT00_USER}@{BAT00_IP}"


def run_ssh(cmd: str, timeout: int = 60, port: int = WSL_PORT) -> subprocess.CompletedProcess:
    """Run a command on bat00 via SSH. Default port 2222 = WSL2 Linux."""
    full = ["ssh", *_ssh_opts(port), ssh_target(), cmd]
    label = "wsl" if port == WSL_PORT else "win"
    print(f"[bat00:{label}] $ {cmd}")
    result = subprocess.run(full, capture_output=False, timeout=timeout)
    return result


def cmd_ps(args: argparse.Namespace) -> int:
    """Run a PowerShell command on bat00 (port 22, Windows OpenSSH)."""
    return run_ssh(args.command, timeout=args.timeout, port=WIN_PORT).returncode


def cmd_wsl(args: argparse.Namespace) -> int:
    """Run a bash command inside WSL2 on bat00 (port 2222, direct Linux SSH).

    This connects directly to WSL2's sshd — no PowerShell, no quoting issues.
    Requires bat00_wsl_setup.ps1 to have been run once on bat00.
    """
    return run_ssh(args.command, timeout=args.timeout, port=WSL_PORT).returncode


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload a file to bat00 via scp."""
    src = Path(args.file)
    dest = args.dest or f"C:/Users/{BAT00_USER}/Desktop/{src.name}"
    scp_cmd = ["scp", *_ssh_opts(WIN_PORT), str(src), f"{ssh_target()}:{dest}"]
    print(f"[bat00] scp {src} -> {dest}")
    return subprocess.run(scp_cmd).returncode


def cmd_status(args: argparse.Namespace) -> int:
    """Quick health check of bat00. Tries WSL2 (port 2222) first, falls back to Windows (port 22)."""
    print("=== bat00 Status ===")

    # Try WSL2 direct (port 2222) — preferred
    print("\n  [WSL2 port 2222]:")
    r = run_ssh(
        "hostname && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader && python3 --version && echo 'WSL2 SSH OK'",
        timeout=15, port=WSL_PORT,
    )
    if r.returncode == 0:
        print("  WSL2 direct SSH: WORKING")
        return 0

    # Fallback: Windows (port 22)
    print("\n  [WSL2 not reachable, trying Windows port 22]:")
    r = run_ssh("hostname", timeout=15, port=WIN_PORT)
    if r.returncode == 0:
        print("  Windows SSH: WORKING (but WSL2 sshd not configured)")
        print("  Run scripts/bat00_wsl_setup.ps1 on bat00 to enable WSL2 direct SSH")
    else:
        print("  BOTH ports unreachable. Check Tailscale: tailscale status")
    return r.returncode


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync the repo to bat00 via rsync over Tailscale SSH.

    Uses rsync to WSL2's filesystem via the Windows mount.
    """
    # rsync to Windows path, then accessible from WSL2 at /mnt/c/Users/...
    dest = f"{ssh_target()}:C:/Users/{BAT00_USER}/pact/"
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "--exclude", ".venv", "--exclude", "__pycache__",
        "--exclude", "*.pyc", "--exclude", ".git",
        "--exclude", "upstream", "--exclude", "dist",
        "--exclude", "build", "--exclude", "*.egg-info",
        "-e", f"ssh {' '.join(_ssh_opts(WIN_PORT))}",
        f"{REPO_ROOT}/", dest,
    ]
    print(f"[bat00] rsync {REPO_ROOT}/ -> {dest}")
    return subprocess.run(rsync_cmd).returncode


def cmd_run_script(args: argparse.Namespace) -> int:
    """Upload a script to bat00 and run it in WSL2."""
    script = Path(args.script)
    if not script.exists():
        print(f"Script not found: {script}")
        return 1

    # Upload to Windows Desktop
    win_path = f"C:/Users/{BAT00_USER}/Desktop/{script.name}"
    wsl_path = f"/mnt/c/Users/{BAT00_USER}/Desktop/{script.name}"

    print(f"[1/3] Uploading {script.name}...")
    scp_cmd = ["scp", *_ssh_opts(WIN_PORT), str(script), f"{ssh_target()}:{win_path}"]
    r = subprocess.run(scp_cmd)
    if r.returncode != 0:
        return r.returncode

    print(f"[2/3] Making executable in WSL2...")
    run_ssh(f'wsl chmod +x {wsl_path}', timeout=15)

    print(f"[3/3] Running in WSL2 (detached)...")
    # Use PowerShell Start-Process to launch WSL detached
    ps_cmd = (
        f'Start-Process wsl -ArgumentList "bash","{wsl_path}" '
        f'-WindowStyle Hidden'
    )
    return run_ssh(ps_cmd, timeout=30).returncode



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical bat00 interaction — handles PowerShell quoting correctly"
    )
    parser.add_argument("--timeout", type=int, default=60, help="SSH timeout in seconds")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_ps = sub.add_parser("ps", help="Run PowerShell command")
    p_ps.add_argument("command", help="PowerShell command string")
    p_ps.set_defaults(func=cmd_ps)

    p_wsl = sub.add_parser("wsl", help="Run bash command in WSL2")
    p_wsl.add_argument("command", help="Bash command string")
    p_wsl.set_defaults(func=cmd_wsl)

    p_upload = sub.add_parser("upload", help="Upload file via scp")
    p_upload.add_argument("file", help="Local file path")
    p_upload.add_argument("--dest", help="Remote Windows path (default: Desktop)")
    p_upload.set_defaults(func=cmd_upload)

    p_status = sub.add_parser("status", help="Health check")
    p_status.set_defaults(func=cmd_status)

    p_sync = sub.add_parser("sync", help="Rsync repo to bat00")
    p_sync.set_defaults(func=cmd_sync)

    p_run = sub.add_parser("run-script", help="Upload + run script in WSL2")
    p_run.add_argument("script", help="Local script path")
    p_run.set_defaults(func=cmd_run_script)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
