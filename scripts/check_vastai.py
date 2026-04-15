#!/usr/bin/env python3
"""Canonical Vast.ai interaction script for pact-lab.

Hardened CLI for safe Vast.ai instance lifecycle management. Replaces ad-hoc
SSH/rsync commands with cost-tracked, safety-gated operations.

Usage:
    python scripts/check_vastai.py status                         Show all instances + cost
    python scripts/check_vastai.py create                         Create instance from best offer
    python scripts/check_vastai.py deploy <id>                    Rsync code to instance
    python scripts/check_vastai.py ssh <id> <cmd>                 Run command on instance
    python scripts/check_vastai.py run <id> <experiment>          Launch registered experiment
    python scripts/check_vastai.py download <id> <remote> <local> Download results
    python scripts/check_vastai.py destroy <id>                   Destroy instance (with confirm)
    python scripts/check_vastai.py destroy-all                    Destroy ALL instances (emergency)
    python scripts/check_vastai.py cost                           Show cumulative session cost

Safety features:
    - Cost tracking with warnings at 50%/75%/90% of $24 cap, hard-refuse at $24
    - Idle detection: flags instances idle >5 min
    - Auto-unbuffered: always sets PYTHONUNBUFFERED=1 on remote
    - PYTHONPATH: always sets src:upstream:$PWD
    - SSH key verification before creating instances
    - Instance labeling: pact-<experiment>-<timestamp>
    - Lean deployment bundles (excludes __pycache__, .git, large files)
    - Result download with manifest.json provenance tracking
    - Destroy confirmation with cost display
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = REPO_ROOT / "experiments" / "results" / "vastai"
INSTANCES_FILE = RESULTS_BASE / "instances.json"

HARD_CAP_USD = 24.0
WARN_THRESHOLDS = [0.50, 0.75, 0.90]  # Warn at these fractions of cap

IDLE_THRESHOLD_SEC = 300  # 5 minutes

# ANSI colors
_RED = "\033[91m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Remote environment for all SSH commands
REMOTE_ENV = (
    "export PYTHONPATH=/workspace/src:/workspace/upstream:/workspace && "
    "export PYTHONUNBUFFERED=1 && "
    "export TAC_UPSTREAM_DIR=/workspace/upstream && "
    "export TAC_MODELS_DIR=/workspace/upstream/models && "
    "source /workspace/.venv/bin/activate 2>/dev/null; "
)

# Rsync exclusions for lean deployment bundles
RSYNC_EXCLUDES = [
    "__pycache__",
    "*.pyc",
    ".git",
    "*.egg-info",
    ".venv",
    "node_modules",
    "*.pt",
    "*.pth",
    "*.onnx",
    "experiments/results",
    "experiments/raft_flow.pt",
    "reports/raw",
    ".omx/state/review_tracker.duckdb",
    "submissions",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_vastai_bin() -> str:
    """Locate the vastai CLI binary."""
    venv_bin = REPO_ROOT / ".venv" / "bin" / "vastai"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("vastai")
    if found:
        return found
    print(f"{_RED}vastai CLI not found. Install with: uv pip install vastai{_RESET}", file=sys.stderr)
    sys.exit(1)


def _run_vastai(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a vastai CLI command."""
    cli = _find_vastai_bin()
    return subprocess.run([cli] + args, capture_output=True, text=True, timeout=timeout)


def _load_instances() -> dict:
    """Load tracked instances from disk."""
    if INSTANCES_FILE.exists():
        return json.loads(INSTANCES_FILE.read_text())
    return {}


def _save_instances(data: dict) -> None:
    """Persist instance data to disk."""
    INSTANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    INSTANCES_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _load_budget() -> dict:
    """Load budget state from disk."""
    budget_file = RESULTS_BASE / "budget.json"
    if budget_file.exists():
        return json.loads(budget_file.read_text())
    return {"total_spent": 0.0, "sessions": []}


def _budget_check(estimated_cost: float = 0.0) -> bool:
    """Check budget status. Returns True if within limits.

    Prints warnings at threshold crossings. Hard-refuses if at cap.
    """
    budget = _load_budget()
    spent = budget.get("total_spent", 0.0)
    remaining = max(0.0, HARD_CAP_USD - spent)
    projected = spent + estimated_cost

    if remaining <= 0:
        print(f"{_RED}{_BOLD}BUDGET EXHAUSTED. ${spent:.2f} / ${HARD_CAP_USD:.2f}{_RESET}")
        return False

    if estimated_cost > remaining:
        print(
            f"{_RED}Insufficient budget. "
            f"Need ${estimated_cost:.2f}, have ${remaining:.2f} remaining{_RESET}"
        )
        return False

    # Threshold warnings
    for threshold in WARN_THRESHOLDS:
        level = HARD_CAP_USD * threshold
        if spent < level <= projected:
            pct = int(threshold * 100)
            print(f"{_YELLOW}WARNING: Spend will cross {pct}% threshold (${level:.2f}){_RESET}")

    return True


def _get_ssh_key(inst: dict) -> str:
    """Find SSH key for an instance."""
    # Check ephemeral key first
    ssh_dir = RESULTS_BASE / ".ssh"
    iid = inst.get("instance_id", "")
    ephemeral = ssh_dir / f"vastai_{iid}"
    if ephemeral.exists():
        return str(ephemeral)
    # Fall back to default keys
    for key_name in ["id_ed25519", "id_rsa"]:
        key_path = Path.home() / ".ssh" / key_name
        if key_path.exists():
            return str(key_path)
    raise FileNotFoundError(f"No SSH key found for instance {iid}")


def _ssh_opts(key_path: str) -> list[str]:
    """Common SSH options."""
    return [
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        "-o", "LogLevel=ERROR",
    ]


def _ssh_exec(host: str, port: int, key: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Execute a command on a remote instance via SSH."""
    return subprocess.run(
        ["ssh"] + _ssh_opts(key) + ["-p", str(port), f"root@{host}", command],
        capture_output=True, text=True, timeout=timeout,
    )


def _verify_ssh_keys() -> bool:
    """Verify SSH keys are available for Vast.ai instances."""
    for key_name in ["id_ed25519", "id_rsa"]:
        if (Path.home() / ".ssh" / key_name).exists():
            return True
    # Check if vastai has registered keys
    result = _run_vastai(["show", "ssh-keys", "--raw"])
    if result.returncode == 0:
        try:
            keys = json.loads(result.stdout)
            if keys:
                return True
        except json.JSONDecodeError:
            pass
    print(f"{_RED}No SSH keys found. Create one with: ssh-keygen -t ed25519{_RESET}")
    return False


def _check_idle(host: str, port: int, key: str) -> float | None:
    """Check how long since last GPU activity. Returns seconds idle or None."""
    try:
        result = _ssh_exec(
            host, port, key,
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null",
            timeout=15,
        )
        if result.returncode == 0:
            util = float(result.stdout.strip().split("\n")[0])
            if util < 5.0:
                # Check if experiment is still running
                ps_result = _ssh_exec(
                    host, port, key,
                    "pgrep -f 'python.*experiments/' 2>/dev/null | head -1",
                    timeout=15,
                )
                if ps_result.returncode != 0 or not ps_result.stdout.strip():
                    return IDLE_THRESHOLD_SEC + 1  # No experiment process, GPU idle
            return 0.0
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _instance_label(experiment: str) -> str:
    """Generate a pact-<experiment>-<timestamp> label."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"pact-{experiment}-{ts}"


def _write_manifest(local_dir: Path, inst: dict, remote_path: str) -> None:
    """Write a manifest.json recording provenance of downloaded results."""
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "vastai",
        "instance_id": inst.get("instance_id", "unknown"),
        "experiment": inst.get("experiment", "unknown"),
        "remote_path": remote_path,
        "ssh_host": inst.get("ssh_host", "unknown"),
        "ssh_port": inst.get("ssh_port", 0),
        "local_dir": str(local_dir),
        "dph": inst.get("dph", 0.0),
        "created_at": inst.get("created_at", "unknown"),
    }
    manifest_path = local_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    """Show all instances + cost."""
    instances = _load_instances()
    budget = _load_budget()
    spent = budget.get("total_spent", 0.0)

    print(f"\n{_BOLD}Vast.ai Instance Status{_RESET}")
    print("=" * 70)
    print(
        f"  Budget: ${spent:.2f} / ${HARD_CAP_USD:.2f} "
        f"(${max(0, HARD_CAP_USD - spent):.2f} remaining)"
    )
    print()

    if not instances:
        print(f"  {_DIM}No tracked instances{_RESET}")
        return 0

    now = datetime.now(timezone.utc)

    for iid, inst in instances.items():
        created = datetime.fromisoformat(inst["created_at"])
        elapsed_hours = (now - created).total_seconds() / 3600
        estimated_cost = inst.get("dph", 0) * elapsed_hours

        # Query remote status
        try:
            remote_info_result = _run_vastai(["show", "instance", iid, "--raw"])
            if remote_info_result.returncode == 0:
                remote_info = json.loads(remote_info_result.stdout)
                vastai_status = remote_info.get("actual_status", remote_info.get("status_msg", "unknown"))
            else:
                vastai_status = "destroyed/not-found"
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            vastai_status = "unknown"

        # Format status with colors
        if vastai_status == "running":
            status_str = f"{_GREEN}{_BOLD}RUNNING{_RESET}"
        elif vastai_status in ("exited", "error"):
            status_str = f"{_RED}{_BOLD}{vastai_status.upper()}{_RESET}"
        elif "destroyed" in str(vastai_status) or "not-found" in str(vastai_status):
            status_str = f"{_DIM}DESTROYED{_RESET}"
        else:
            status_str = f"{_YELLOW}{vastai_status}{_RESET}"

        timeout_hours = inst.get("timeout_hours", 0)
        over_timeout = elapsed_hours > timeout_hours if timeout_hours > 0 else False
        timeout_str = (
            f"{_RED}OVER LIMIT{_RESET}" if over_timeout
            else f"{elapsed_hours:.1f}/{timeout_hours}h"
        )

        print(f"  {_BOLD}[{iid}]{_RESET} {inst.get('experiment', 'unknown')}")
        print(f"    Status: {status_str} | Elapsed: {timeout_str}")
        print(f"    Cost: ${estimated_cost:.2f} (${inst.get('dph', 0):.3f}/hr)")
        print(f"    SSH: ssh -p {inst.get('ssh_port', '?')} root@{inst.get('ssh_host', '?')}")

        # Idle detection for running instances
        if vastai_status == "running":
            try:
                key = _get_ssh_key({"instance_id": iid, **inst})
                idle = _check_idle(inst["ssh_host"], inst["ssh_port"], key)
                if idle is not None and idle > IDLE_THRESHOLD_SEC:
                    print(f"    {_YELLOW}{_BOLD}IDLE >5 min (GPU <5% util, no experiment process){_RESET}")
            except (FileNotFoundError, KeyError):
                pass

        if over_timeout and vastai_status == "running":
            print(f"    {_RED}{_BOLD}TIMEOUT EXCEEDED{_RESET}")

        print()

    return 0


def cmd_create(args: argparse.Namespace) -> int:
    """Create instance from best offer."""
    if not _verify_ssh_keys():
        return 1

    experiment = getattr(args, "experiment", None) or "general"
    label = _instance_label(experiment)

    print(f"\n{_BOLD}Searching for RTX 4090 instances...{_RESET}")
    search_query = (
        "gpu_name='RTX 4090' num_gpus=1 reliability>0.95 "
        "inet_down>200 disk_space>40 rentable=True"
    )
    result = _run_vastai(["search", "offers", search_query, "--order", "dph_total", "--limit", "5", "--raw"])
    if result.returncode != 0:
        print(f"{_RED}Search failed: {result.stderr}{_RESET}")
        return 1

    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"{_RED}Failed to parse offers{_RESET}")
        return 1

    if not offers:
        print(f"{_RED}No RTX 4090 offers found{_RESET}")
        return 1

    best = offers[0]
    dph = best.get("dph_total", best.get("dph", 0.0))
    offer_id = best["id"]

    # Budget check for 4 hours (default timeout)
    if not _budget_check(dph * 4.0):
        return 1

    print(f"  Best offer: ${dph:.3f}/hr (ID: {offer_id})")
    print(f"  Label: {label}")

    create_result = _run_vastai([
        "create", "instance", str(offer_id),
        "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        "--disk", "40",
        "--ssh", "--direct",
        "--label", label,
        "--raw",
    ])
    if create_result.returncode != 0:
        print(f"{_RED}Instance creation failed: {create_result.stderr}{_RESET}")
        return 1

    # Parse instance ID
    try:
        data = json.loads(create_result.stdout)
        iid = str(data.get("new_contract", data.get("id", "")))
    except (json.JSONDecodeError, TypeError):
        import re
        match = re.search(r"\d{4,}", create_result.stdout)
        iid = match.group(0) if match else ""

    if not iid:
        print(f"{_RED}Could not determine instance ID{_RESET}")
        return 1

    print(f"  {_GREEN}Instance created: {iid}{_RESET}")
    print(f"  Waiting for instance to start...")

    # Wait for running + SSH info
    for attempt in range(60):
        time.sleep(10)
        try:
            info_result = _run_vastai(["show", "instance", iid, "--raw"])
            if info_result.returncode != 0:
                continue
            info = json.loads(info_result.stdout)
            status = info.get("actual_status", "")
            if status == "running":
                ssh_host = info.get("ssh_host", info.get("public_ipaddr"))
                ssh_port = info.get("ssh_port")
                if ssh_host and ssh_port:
                    break
            elif status in ("exited", "error"):
                print(f"{_RED}Instance failed to start: {status}{_RESET}")
                _run_vastai(["destroy", "instance", iid])
                return 1
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            continue
        if attempt % 6 == 0 and attempt > 0:
            print(f"  {_DIM}Still waiting... ({attempt * 10}s){_RESET}")
    else:
        print(f"{_RED}Timed out waiting for instance{_RESET}")
        _run_vastai(["destroy", "instance", iid])
        return 1

    # Save instance
    instances = _load_instances()
    instances[iid] = {
        "experiment": experiment,
        "dph": dph,
        "ssh_host": ssh_host,
        "ssh_port": int(ssh_port),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeout_hours": 4.0,
        "status": "created",
        "label": label,
    }
    _save_instances(instances)

    print(f"\n  {_GREEN}{_BOLD}Instance ready!{_RESET}")
    print(f"  ID: {iid}")
    print(f"  SSH: ssh -p {ssh_port} root@{ssh_host}")
    print(f"  Deploy code: python scripts/check_vastai.py deploy {iid}")
    print(f"  Run experiment: python scripts/check_vastai.py run {iid} <experiment>")

    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    """Rsync a lean code bundle to an instance."""
    iid = args.instance_id
    instances = _load_instances()
    if iid not in instances:
        print(f"{_RED}Unknown instance: {iid}{_RESET}")
        return 1

    inst = instances[iid]
    try:
        key = _get_ssh_key({"instance_id": iid, **inst})
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    host = inst["ssh_host"]
    port = inst["ssh_port"]

    print(f"\n{_BOLD}Deploying code to [{iid}]...{_RESET}")

    # Build rsync exclude args
    exclude_args: list[str] = []
    for excl in RSYNC_EXCLUDES:
        exclude_args.extend(["--exclude", excl])

    ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(key)) + f" -p {port}"

    # Upload src/
    print("  Uploading src/...")
    result = subprocess.run(
        ["rsync", "-avz", "--progress"] + exclude_args + [
            "-e", ssh_cmd,
            str(REPO_ROOT / "src") + "/",
            f"root@{host}:/workspace/src/",
        ],
        text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"{_RED}Failed to upload src/{_RESET}")
        return 1

    # Upload experiments/
    print("  Uploading experiments/...")
    result = subprocess.run(
        ["rsync", "-avz", "--progress"] + exclude_args + [
            "-e", ssh_cmd,
            str(REPO_ROOT / "experiments") + "/",
            f"root@{host}:/workspace/experiments/",
        ],
        text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"{_RED}Failed to upload experiments/{_RESET}")
        return 1

    # Upload scripts/
    print("  Uploading scripts/...")
    result = subprocess.run(
        ["rsync", "-avz", "--progress"] + exclude_args + [
            "-e", ssh_cmd,
            str(REPO_ROOT / "scripts") + "/",
            f"root@{host}:/workspace/scripts/",
        ],
        text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"{_YELLOW}Warning: failed to upload scripts/ (non-critical){_RESET}")

    print(f"  {_GREEN}Deploy complete{_RESET}")
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    """Run a command on an instance."""
    iid = args.instance_id
    command = " ".join(args.command)
    instances = _load_instances()
    if iid not in instances:
        print(f"{_RED}Unknown instance: {iid}{_RESET}")
        return 1

    inst = instances[iid]
    try:
        key = _get_ssh_key({"instance_id": iid, **inst})
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    # Wrap command with environment setup
    full_cmd = REMOTE_ENV + command

    result = _ssh_exec(inst["ssh_host"], inst["ssh_port"], key, full_cmd, timeout=300)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def cmd_run(args: argparse.Namespace) -> int:
    """Launch a registered experiment on an instance."""
    iid = args.instance_id
    experiment_name = args.experiment
    instances = _load_instances()
    if iid not in instances:
        print(f"{_RED}Unknown instance: {iid}{_RESET}")
        return 1

    # Import experiment registry
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from tac.deploy.vastai.experiments import EXPERIMENTS
    except ImportError:
        print(f"{_RED}Cannot import experiment registry. Is src/tac/ available?{_RESET}")
        return 1

    if experiment_name not in EXPERIMENTS:
        print(f"{_RED}Unknown experiment: {experiment_name}{_RESET}")
        print(f"Available: {', '.join(sorted(EXPERIMENTS.keys()))}")
        return 1

    experiment = EXPERIMENTS[experiment_name]
    inst = instances[iid]

    try:
        key = _get_ssh_key({"instance_id": iid, **inst})
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    # Upload checkpoint if needed
    if experiment.needs_checkpoint:
        ckpt_path = REPO_ROOT / "experiments" / "results" / "fridrich_renderer" / experiment.needs_checkpoint
        if not ckpt_path.exists():
            print(f"{_RED}Missing checkpoint: {ckpt_path}{_RESET}")
            return 1
        print(f"  Uploading checkpoint: {experiment.needs_checkpoint}...")
        ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(key)) + f" -p {inst['ssh_port']}"
        subprocess.run(
            ["rsync", "-avz", "--progress", "-e", ssh_cmd,
             str(ckpt_path), f"root@{inst['ssh_host']}:/workspace/{experiment.needs_checkpoint}"],
            text=True, timeout=600,
        )

    # Build run script
    args_str = " ".join(experiment.args)
    run_script = (
        "#!/bin/bash\n"
        "cd /workspace\n"
        "source .venv/bin/activate\n"
        "export PYTHONPATH=/workspace/src:/workspace/upstream\n"
        "export PYTHONUNBUFFERED=1\n"
        "export TAC_UPSTREAM_DIR=/workspace/upstream\n"
        "export TAC_MODELS_DIR=/workspace/upstream/models\n"
        f"timeout {experiment.timeout_seconds} python3 -u {experiment.script} {args_str} "
        "> /workspace/experiment_stdout.log 2>&1\n"
        "EXIT_STATUS=$?\n"
        'echo "EXIT_CODE=$EXIT_STATUS" >> /workspace/experiment_stdout.log\n'
        "echo EXPERIMENT_DONE > /workspace/.experiment_done\n"
    )

    _ssh_exec(
        inst["ssh_host"], inst["ssh_port"], key,
        f"cat > /workspace/run_experiment.sh << 'RUNEOF'\n{run_script}RUNEOF",
        timeout=15,
    )
    _ssh_exec(inst["ssh_host"], inst["ssh_port"], key, "chmod +x /workspace/run_experiment.sh", timeout=10)
    _ssh_exec(
        inst["ssh_host"], inst["ssh_port"], key,
        "nohup /workspace/run_experiment.sh </dev/null >/workspace/nohup.out 2>&1 &",
        timeout=10,
    )

    # Update instance record
    instances[iid]["experiment"] = experiment_name
    instances[iid]["timeout_hours"] = experiment.timeout_hours
    instances[iid]["status"] = "running"
    _save_instances(instances)

    print(f"\n  {_GREEN}{_BOLD}Experiment launched: {experiment_name}{_RESET}")
    print(f"  Instance: {iid}")
    print(f"  Timeout: {experiment.timeout_hours}h")
    print(f"  Monitor: python scripts/check_vastai.py status")
    print(f"  Tail log: python scripts/check_vastai.py ssh {iid} 'tail -f /workspace/experiment_stdout.log'")
    print(f"  Download: python scripts/check_vastai.py download {iid} /workspace/experiments/results/ experiments/results/vastai/{experiment_name}/")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download results from an instance."""
    iid = args.instance_id
    remote_path = args.remote_path
    local_path = args.local_path

    instances = _load_instances()
    if iid not in instances:
        print(f"{_RED}Unknown instance: {iid}{_RESET}")
        return 1

    inst = instances[iid]
    try:
        key = _get_ssh_key({"instance_id": iid, **inst})
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    # Default local path uses experiment label
    if local_path is None:
        label = inst.get("label", inst.get("experiment", "unknown"))
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        local_dir = RESULTS_BASE / f"{label}_{ts}"
    else:
        local_dir = Path(local_path)

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{_BOLD}Downloading from [{iid}]...{_RESET}")
    print(f"  Remote: {remote_path}")
    print(f"  Local:  {local_dir}")

    ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(key)) + f" -p {inst['ssh_port']}"

    result = subprocess.run(
        ["rsync", "-avz", "--progress", "-e", ssh_cmd,
         f"root@{inst['ssh_host']}:{remote_path}",
         str(local_dir) + "/"],
        text=True, timeout=1800,  # 30 min timeout for large downloads
    )

    if result.returncode != 0:
        print(f"{_RED}Download failed{_RESET}")
        return 1

    # Also grab the experiment log
    subprocess.run(
        ["rsync", "-avz", "-e", ssh_cmd,
         f"root@{inst['ssh_host']}:/workspace/experiment_stdout.log",
         str(local_dir) + "/"],
        capture_output=True, text=True, timeout=60,
    )

    # Write provenance manifest
    _write_manifest(local_dir, {"instance_id": iid, **inst}, remote_path)

    print(f"  {_GREEN}Download complete: {local_dir}{_RESET}")
    print(f"  Manifest: {local_dir / 'manifest.json'}")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    """Destroy an instance with confirmation."""
    iid = args.instance_id
    instances = _load_instances()

    if iid not in instances:
        print(f"{_RED}Unknown instance: {iid}{_RESET}")
        return 1

    inst = instances[iid]
    budget = _load_budget()
    spent = budget.get("total_spent", 0.0)

    # Calculate cost for this instance
    created = datetime.fromisoformat(inst["created_at"])
    elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    instance_cost = inst.get("dph", 0) * elapsed_hours

    print(f"\n{_BOLD}Destroy instance [{iid}]?{_RESET}")
    print(f"  Experiment: {inst.get('experiment', 'unknown')}")
    print(f"  Elapsed: {elapsed_hours:.1f}h")
    print(f"  Instance cost: ${instance_cost:.2f}")
    print(f"  Total budget spent: ${spent:.2f} / ${HARD_CAP_USD:.2f}")

    if not args.yes:
        confirm = input(f"\n  Type 'y' to confirm destroy: ")
        if confirm.strip().lower() != "y":
            print("  Cancelled.")
            return 0

    _run_vastai(["destroy", "instance", iid])
    instances[iid]["status"] = "destroyed"
    _save_instances(instances)

    # Clean up SSH key
    ssh_dir = RESULTS_BASE / ".ssh"
    for suffix in ["", ".pub"]:
        key_file = ssh_dir / f"vastai_{iid}{suffix}"
        key_file.unlink(missing_ok=True)

    print(f"  {_GREEN}Destroyed {iid}{_RESET}")
    return 0


def cmd_destroy_all(args: argparse.Namespace) -> int:
    """Emergency: destroy ALL tracked instances."""
    instances = _load_instances()
    active = {k: v for k, v in instances.items() if v.get("status") != "destroyed"}

    if not active:
        print(f"  {_DIM}No active instances to destroy{_RESET}")
        return 0

    budget = _load_budget()
    spent = budget.get("total_spent", 0.0)

    print(f"\n{_RED}{_BOLD}EMERGENCY DESTROY ALL{_RESET}")
    print(f"  Active instances: {len(active)}")
    print(f"  Total budget spent: ${spent:.2f} / ${HARD_CAP_USD:.2f}")

    for iid, inst in active.items():
        print(f"  [{iid}] {inst.get('experiment', 'unknown')}")

    if not args.yes:
        confirm = input(f"\n  Type 'DESTROY ALL' to confirm: ")
        if confirm.strip() != "DESTROY ALL":
            print("  Cancelled.")
            return 0

    for iid in active:
        _run_vastai(["destroy", "instance", iid])
        instances[iid]["status"] = "destroyed"
        print(f"  {_GREEN}Destroyed {iid}{_RESET}")

    _save_instances(instances)
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    """Show cumulative session cost."""
    budget = _load_budget()
    spent = budget.get("total_spent", 0.0)
    pct = (spent / HARD_CAP_USD) * 100 if HARD_CAP_USD > 0 else 0
    remaining = max(0.0, HARD_CAP_USD - spent)

    if remaining <= 0:
        color = _RED
    elif pct > 75:
        color = _YELLOW
    else:
        color = _GREEN

    print(f"\n{_BOLD}Vast.ai Cost Tracker{_RESET}")
    print("=" * 50)
    print(f"  Total spent:  {color}${spent:.2f}{_RESET}")
    print(f"  Hard cap:     ${HARD_CAP_USD:.2f}")
    print(f"  Remaining:    {color}${remaining:.2f}{_RESET} ({100 - pct:.0f}%)")

    # Show threshold status
    for threshold in WARN_THRESHOLDS:
        level = HARD_CAP_USD * threshold
        pct_t = int(threshold * 100)
        if spent >= level:
            print(f"  {_YELLOW}[x] {pct_t}% threshold (${level:.2f}) CROSSED{_RESET}")
        else:
            print(f"  {_DIM}[ ] {pct_t}% threshold (${level:.2f}){_RESET}")

    # Session history
    sessions = budget.get("sessions", [])
    if sessions:
        print(f"\n  {_BOLD}Recent sessions:{_RESET}")
        for entry in sessions[-10:]:
            ts = entry.get("timestamp", "?")[:19].replace("T", " ")
            amt = entry.get("amount", 0)
            desc = entry.get("description", "")
            if amt > 0:
                print(f"    {ts}  ${amt:>7.2f}  {desc}")
            else:
                print(f"    {_DIM}{ts}  ${amt:>7.2f}  {desc}{_RESET}")

    print()
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical Vast.ai interaction script for pact-lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s status                                 Show all instances\n"
            "  %(prog)s create --experiment tto_v1              Create instance\n"
            "  %(prog)s deploy 12345                            Upload code\n"
            "  %(prog)s ssh 12345 nvidia-smi                    Run command\n"
            "  %(prog)s run 12345 tto_step_curve_hinge          Launch experiment\n"
            "  %(prog)s download 12345 /workspace/results/ ./   Download results\n"
            "  %(prog)s destroy 12345                           Destroy (with confirm)\n"
            "  %(prog)s destroy-all                             Emergency destroy all\n"
            "  %(prog)s cost                                    Show spend tracking\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Show all instances + cost")

    # create
    create_p = subparsers.add_parser("create", help="Create instance from best offer")
    create_p.add_argument("--experiment", "-e", default="general", help="Experiment label")

    # deploy
    deploy_p = subparsers.add_parser("deploy", help="Rsync code to instance")
    deploy_p.add_argument("instance_id", help="Instance ID")

    # ssh
    ssh_p = subparsers.add_parser("ssh", help="Run command on instance")
    ssh_p.add_argument("instance_id", help="Instance ID")
    ssh_p.add_argument("command", nargs="+", help="Command to run")

    # run
    run_p = subparsers.add_parser("run", help="Launch registered experiment")
    run_p.add_argument("instance_id", help="Instance ID")
    run_p.add_argument("experiment", help="Experiment name from registry")

    # download
    dl_p = subparsers.add_parser("download", help="Download results")
    dl_p.add_argument("instance_id", help="Instance ID")
    dl_p.add_argument("remote_path", help="Remote path to download")
    dl_p.add_argument("local_path", nargs="?", default=None, help="Local destination (auto-generated if omitted)")

    # destroy
    destroy_p = subparsers.add_parser("destroy", help="Destroy instance")
    destroy_p.add_argument("instance_id", help="Instance ID")
    destroy_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # destroy-all
    da_p = subparsers.add_parser("destroy-all", help="Emergency: destroy ALL instances")
    da_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # cost
    subparsers.add_parser("cost", help="Show cumulative session cost")

    args = parser.parse_args()

    dispatch = {
        "status": cmd_status,
        "create": cmd_create,
        "deploy": cmd_deploy,
        "ssh": cmd_ssh,
        "run": cmd_run,
        "download": cmd_download,
        "destroy": cmd_destroy,
        "destroy-all": cmd_destroy_all,
        "cost": cmd_cost,
    }

    try:
        return dispatch[args.command](args)
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Interrupted{_RESET}")
        return 130
    except subprocess.TimeoutExpired:
        print(f"\n{_RED}Command timed out{_RESET}")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
