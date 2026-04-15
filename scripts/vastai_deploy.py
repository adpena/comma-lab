#!/usr/bin/env python3
"""Vast.ai deployment script for RTX 4090 experiments.

Full lifecycle management: search -> create -> upload -> run -> monitor -> download -> destroy.

Usage:
    python scripts/vastai_deploy.py search                          # Find best 4090 offers
    python scripts/vastai_deploy.py launch --experiment tto_v1      # Spin up and run
    python scripts/vastai_deploy.py status                          # Check all running instances
    python scripts/vastai_deploy.py results                         # Download results from completed
    python scripts/vastai_deploy.py destroy --all                   # Tear down everything
    python scripts/vastai_deploy.py budget                          # Show spend tracking

Budget: Hard cap at $24 of $25 total. Auto-destroy after experiment timeout.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "vastai"
BUDGET_FILE = RESULTS_DIR / "budget.json"
INSTANCES_FILE = RESULTS_DIR / "instances.json"
SSH_KEY_DIR = RESULTS_DIR / ".ssh"
API_KEY_FILE = Path.home() / ".vast_api_key"

# Budget limits (USD)
BUDGET_HARD_CAP = 24.0
BUDGET_WARN_THRESHOLD = 20.0
BUDGET_TOTAL = 25.0

# Search constraints for RTX 4090
SEARCH_CONSTRAINTS = {
    "gpu_name": "RTX 4090",
    "reliability": 0.95,
    "inet_down": 200.0,     # Mbps minimum
    "disk_space": 30.0,     # GB minimum
    "num_gpus": 1,
}

# Docker image
DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

# Instance label prefix for tracking
LABEL_PREFIX = "pact-lab"

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Experiment configs ─────────────────────────────────────────────────────────

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "tto_v1": {
        "script": "experiments/renderer_tto.py",
        "args": (
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--simulate-resize"
        ),
        "needs_checkpoint": "renderer_best.pt",
        "needs_upstream": True,
        "timeout_hours": 2,
    },
    "tto_v1_seg_odd": {
        "script": "experiments/renderer_tto.py",
        "args": (
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--seg-odd-only --simulate-resize"
        ),
        "needs_checkpoint": "renderer_best.pt",
        "needs_upstream": True,
        "timeout_hours": 2,
    },
    "sensitivity_map": {
        "script": "experiments/analysis/posenet_sensitivity.py",
        "args": "--device cuda --n-frames 1200",
        "needs_upstream": True,
        "timeout_hours": 1,
    },
    "warp_baseline": {
        "script": "experiments/warp_gen_baseline.py",
        "args": "--device cuda --n-frames 1200",
        "needs_upstream": True,
        "timeout_hours": 1,
    },
    "gt_sparse_tto": {
        "script": "experiments/gt_sparse_tto.py",
        "args": (
            "--device cuda --n-frames 1200 --n-patches 50 "
            "--n-restarts 20 --steps-per-restart 500"
        ),
        "needs_upstream": True,
        "timeout_hours": 4,
    },
    "joint_pair_train": {
        "script": "experiments/train_joint_pair.py",
        "args": "--device cuda --epochs 100",
        "needs_upstream": True,
        "needs_checkpoint": "renderer_best.pt",
        "timeout_hours": 20,
    },
}

# Onstart script: runs when the instance boots.
# The Docker image already has PyTorch 2.5.1 + CUDA 12.4 installed system-wide.
# We create a venv with --system-site-packages to inherit torch, then only install
# the missing dependencies. This saves ~2GB download and several minutes.
ONSTART_SCRIPT = """\
#!/bin/bash
set -ex

# Install uv for package management
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv inheriting system torch from the Docker image
cd /workspace
uv venv --python 3.12 --system-site-packages
source .venv/bin/activate

# Only install deps NOT in the base image (torch/torchvision already present)
uv pip install pyav safetensors segmentation-models-pytorch timm einops pydantic click

# Clone upstream scorer (PoseNet/SegNet models + GT video)
if [ ! -d /workspace/upstream ]; then
    apt-get update && apt-get install -y git-lfs
    git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /workspace/upstream
    cd /workspace/upstream && git lfs pull
fi

echo "SETUP_COMPLETE" > /workspace/.setup_done
"""


# ── Utility functions ──────────────────────────────────────────────────────────


def vastai_bin() -> str:
    """Find the vastai CLI binary."""
    venv_vastai = REPO_ROOT / ".venv" / "bin" / "vastai"
    if venv_vastai.exists():
        return str(venv_vastai)
    found = shutil.which("vastai")
    if found:
        return found
    print(f"{RED}vastai CLI not found. Install with: uv pip install vastai{RESET}")
    sys.exit(1)


def run_vastai(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a vastai CLI command."""
    cli = vastai_bin()
    cmd = [cli] + args
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    else:
        result = subprocess.run(cmd, text=True, timeout=120)
    return result


def run_vastai_json(args: list[str]) -> Any:
    """Run a vastai CLI command and parse JSON output.

    Note: used by cmd_status and other commands that need structured data.
    """
    result = run_vastai(args)
    if result.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Some vastai commands return non-JSON; return raw stdout
        return result.stdout.strip()


def ensure_dirs():
    """Create necessary directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
    SSH_KEY_DIR.chmod(0o700)


def check_api_key():
    """Verify Vast.ai API key exists."""
    if not API_KEY_FILE.exists():
        print(f"{RED}No Vast.ai API key found at {API_KEY_FILE}{RESET}")
        print("Set it with: vastai set api-key <your-key>")
        sys.exit(1)


# ── Budget management ──────────────────────────────────────────────────────────


def load_budget() -> dict:
    """Load budget tracking data."""
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text())
    return {
        "total_spent": 0.0,
        "sessions": [],
    }


def save_budget(budget: dict):
    """Persist budget tracking data."""
    ensure_dirs()
    BUDGET_FILE.write_text(json.dumps(budget, indent=2) + "\n")


def record_spend(budget: dict, instance_id: str, amount: float, description: str):
    """Record a spending event."""
    budget["total_spent"] += amount
    budget["sessions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id,
        "amount": amount,
        "description": description,
        "cumulative": budget["total_spent"],
    })
    save_budget(budget)


def check_budget_remaining(budget: dict, estimated_cost: float) -> bool:
    """Check if we have budget for estimated_cost. Returns True if OK."""
    remaining = BUDGET_HARD_CAP - budget["total_spent"]
    if remaining <= 0:
        print(f"{RED}{BOLD}BUDGET EXHAUSTED. Spent ${budget['total_spent']:.2f} / ${BUDGET_HARD_CAP:.2f}{RESET}")
        return False
    if estimated_cost > remaining:
        print(f"{RED}Insufficient budget. Need ${estimated_cost:.2f}, have ${remaining:.2f} remaining{RESET}")
        return False
    if budget["total_spent"] + estimated_cost > BUDGET_WARN_THRESHOLD:
        print(f"{YELLOW}WARNING: This will bring spend to ${budget['total_spent'] + estimated_cost:.2f} "
              f"(warn threshold: ${BUDGET_WARN_THRESHOLD:.2f}){RESET}")
    return True


# ── Instance tracking ──────────────────────────────────────────────────────────


def load_instances() -> dict[str, dict]:
    """Load instance tracking data. Keys are instance IDs (strings)."""
    if INSTANCES_FILE.exists():
        return json.loads(INSTANCES_FILE.read_text())
    return {}


def save_instances(instances: dict[str, dict]):
    """Persist instance tracking data."""
    ensure_dirs()
    INSTANCES_FILE.write_text(json.dumps(instances, indent=2) + "\n")


def register_instance(instance_id: str, experiment: str, dph: float, ssh_host: str,
                      ssh_port: int, timeout_hours: float):
    """Register a new instance in tracking."""
    instances = load_instances()
    instances[str(instance_id)] = {
        "experiment": experiment,
        "dph": dph,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeout_hours": timeout_hours,
        "status": "created",
    }
    save_instances(instances)


def update_instance_status(instance_id: str, status: str):
    """Update instance status."""
    instances = load_instances()
    key = str(instance_id)
    if key in instances:
        instances[key]["status"] = status
        save_instances(instances)


# ── SSH key management ─────────────────────────────────────────────────────────


def generate_ssh_key(instance_id: str) -> tuple[str, str]:
    """Generate an ephemeral SSH key pair for an instance.

    Returns (private_key_path, public_key_content).
    """
    ensure_dirs()
    key_path = SSH_KEY_DIR / f"vastai_{instance_id}"
    if key_path.exists():
        key_path.unlink()
    if key_path.with_suffix(".pub").exists():
        key_path.with_suffix(".pub").unlink()

    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", f"pact-lab-{instance_id}"],
        capture_output=True, check=True,
    )
    key_path.chmod(0o600)
    pub_content = key_path.with_suffix(".pub").read_text().strip()
    return str(key_path), pub_content


def cleanup_ssh_key(instance_id: str):
    """Remove ephemeral SSH key pair."""
    key_path = SSH_KEY_DIR / f"vastai_{instance_id}"
    key_path.unlink(missing_ok=True)
    key_path.with_suffix(".pub").unlink(missing_ok=True)


# ── SSH / rsync helpers ────────────────────────────────────────────────────────


def ssh_opts(key_path: str) -> list[str]:
    """Common SSH options for Vast.ai connections."""
    return [
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        "-o", "LogLevel=ERROR",
    ]


def wait_for_ssh(ssh_host: str, ssh_port: int, key_path: str,
                 max_wait: int = 600, interval: int = 15) -> bool:
    """Wait for SSH to become available on the instance.

    Polls every `interval` seconds up to `max_wait` seconds.
    Returns True if SSH is reachable, False on timeout.
    """
    start = time.time()
    attempt = 0
    while time.time() - start < max_wait:
        attempt += 1
        try:
            result = subprocess.run(
                ["ssh"] + ssh_opts(key_path) + [
                    "-p", str(ssh_port),
                    f"root@{ssh_host}",
                    "echo ok",
                ],
                capture_output=True, text=True, timeout=35,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                elapsed = time.time() - start
                print(f"  {GREEN}SSH ready after {elapsed:.0f}s ({attempt} attempts){RESET}")
                return True
        except subprocess.TimeoutExpired:
            pass  # SSH hung, retry
        except OSError:
            pass  # Connection refused, retry
        if attempt % 4 == 0:
            print(f"  {DIM}Waiting for SSH... ({int(time.time() - start)}s elapsed){RESET}")
        time.sleep(interval)
    print(f"  {RED}SSH timeout after {max_wait}s{RESET}")
    return False


def wait_for_setup(ssh_host: str, ssh_port: int, key_path: str,
                   max_wait: int = 900, interval: int = 20) -> bool:
    """Wait for the onstart script to finish (checks for .setup_done marker).

    Default 900s (15min) to allow time for torch download (~2GB) via uv.
    """
    start = time.time()
    while time.time() - start < max_wait:
        try:
            result = subprocess.run(
                ["ssh"] + ssh_opts(key_path) + [
                    "-p", str(ssh_port),
                    f"root@{ssh_host}",
                    "cat /workspace/.setup_done 2>/dev/null || echo PENDING",
                ],
                capture_output=True, text=True, timeout=35,
            )
            if result.returncode == 0 and "SETUP_COMPLETE" in result.stdout:
                elapsed = time.time() - start
                print(f"  {GREEN}Setup complete after {elapsed:.0f}s{RESET}")
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass  # SSH hung or refused, retry
        elapsed = int(time.time() - start)
        if elapsed % 60 < interval:
            print(f"  {DIM}Waiting for setup... ({elapsed}s elapsed){RESET}")
        time.sleep(interval)
    print(f"  {RED}Setup timeout after {max_wait}s{RESET}")
    return False


def rsync_upload(ssh_host: str, ssh_port: int, key_path: str,
                 local_path: str, remote_path: str, max_retries: int = 3) -> bool:
    """Upload files via rsync over SSH with retry logic."""
    ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in ssh_opts(key_path)) + f" -p {ssh_port}"
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(
            [
                "rsync", "-avz", "--progress",
                "--exclude", "__pycache__",
                "--exclude", "*.pyc",
                "--exclude", ".git",
                "--exclude", "*.egg-info",
                "--exclude", "experiments/results",
                "--exclude", "experiments/raft_flow.pt",
                "--exclude", ".venv",
                "--exclude", "node_modules",
                "-e", ssh_cmd,
                local_path,
                f"root@{ssh_host}:{remote_path}",
            ],
            text=True, timeout=600,
        )
        if result.returncode == 0:
            return True
        if attempt < max_retries:
            print(f"  {YELLOW}rsync attempt {attempt} failed, retrying in 10s...{RESET}")
            time.sleep(10)
    print(f"  {RED}rsync failed after {max_retries} attempts{RESET}")
    return False


def rsync_download(ssh_host: str, ssh_port: int, key_path: str,
                   remote_path: str, local_path: str, max_retries: int = 3) -> bool:
    """Download files via rsync over SSH with retry logic."""
    ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in ssh_opts(key_path)) + f" -p {ssh_port}"
    Path(local_path).mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(
            [
                "rsync", "-avz", "--progress",
                "-e", ssh_cmd,
                f"root@{ssh_host}:{remote_path}",
                local_path,
            ],
            text=True, timeout=600,
        )
        if result.returncode == 0:
            return True
        if attempt < max_retries:
            print(f"  {YELLOW}rsync download attempt {attempt} failed, retrying in 10s...{RESET}")
            time.sleep(10)
    print(f"  {RED}rsync download failed after {max_retries} attempts{RESET}")
    return False


def ssh_exec(ssh_host: str, ssh_port: int, key_path: str, command: str,
             timeout: int = 30) -> subprocess.CompletedProcess:
    """Execute a command on the remote instance via SSH."""
    return subprocess.run(
        ["ssh"] + ssh_opts(key_path) + [
            "-p", str(ssh_port),
            f"root@{ssh_host}",
            command,
        ],
        capture_output=True, text=True, timeout=timeout,
    )


# ── Core commands ──────────────────────────────────────────────────────────────


def cmd_search(args: argparse.Namespace) -> int:
    """Search for best RTX 4090 offers."""
    print(f"\n{BOLD}Searching for RTX 4090 instances...{RESET}\n")

    query = (
        f"gpu_name='RTX 4090' "
        f"num_gpus=1 "
        f"reliability>{SEARCH_CONSTRAINTS['reliability']} "
        f"inet_down>{SEARCH_CONSTRAINTS['inet_down']} "
        f"disk_space>{SEARCH_CONSTRAINTS['disk_space']} "
        f"rentable=True"
    )

    result = run_vastai(["search", "offers", query, "--order", "dph_total", "--limit", "15"])
    if result.returncode != 0:
        print(f"{RED}Search failed: {result.stderr}{RESET}")
        return 1

    # Print raw output (vastai formats it as a table)
    print(result.stdout)

    # Also show budget context
    budget = load_budget()
    remaining = BUDGET_HARD_CAP - budget["total_spent"]
    print(f"\n{BOLD}Budget:{RESET} ${budget['total_spent']:.2f} spent, "
          f"${remaining:.2f} remaining of ${BUDGET_HARD_CAP:.2f} cap")

    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch an experiment on a Vast.ai RTX 4090 instance."""
    experiment_name = args.experiment
    if experiment_name not in EXPERIMENTS:
        print(f"{RED}Unknown experiment: {experiment_name}{RESET}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        return 1

    config = EXPERIMENTS[experiment_name]
    timeout_hours = config["timeout_hours"]

    # ── Budget check ──
    budget = load_budget()

    # Search for cheapest offer to estimate cost
    print(f"\n{BOLD}Launching experiment: {experiment_name}{RESET}")
    print(f"  Timeout: {timeout_hours}h")
    print(f"  Script: {config['script']}")
    print()

    query = (
        f"gpu_name='RTX 4090' "
        f"num_gpus=1 "
        f"reliability>{SEARCH_CONSTRAINTS['reliability']} "
        f"inet_down>{SEARCH_CONSTRAINTS['inet_down']} "
        f"disk_space>{SEARCH_CONSTRAINTS['disk_space']} "
        f"rentable=True"
    )

    result = run_vastai(["search", "offers", query, "--order", "dph_total", "--limit", "5", "--raw"])
    if result.returncode != 0:
        print(f"{RED}No 4090 offers found: {result.stderr}{RESET}")
        return 1

    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"{RED}Failed to parse offers{RESET}")
        print(result.stdout[:500])
        return 1

    if not offers:
        print(f"{RED}No RTX 4090 offers matching constraints{RESET}")
        return 1

    # Pick cheapest offer
    best = offers[0]
    offer_id = best["id"]
    dph = best.get("dph_total", best.get("dph", 0.0))
    estimated_cost = dph * timeout_hours
    machine_location = best.get("geolocation", "unknown")

    print(f"  Best offer: ID {offer_id}")
    print(f"  Rate: ${dph:.3f}/hr")
    print(f"  Estimated cost: ${estimated_cost:.2f} ({timeout_hours}h)")
    print(f"  Location: {machine_location}")
    print(f"  VRAM: {best.get('gpu_ram', '?')} GB")
    print(f"  Disk: {best.get('disk_space', '?')} GB")
    print()

    if not check_budget_remaining(budget, estimated_cost):
        return 1

    # Extra warning for expensive experiments (>30% of total budget)
    if estimated_cost > BUDGET_HARD_CAP * 0.30:
        print(f"{YELLOW}{BOLD}CAUTION: This experiment costs ${estimated_cost:.2f} "
              f"({estimated_cost/BUDGET_HARD_CAP*100:.0f}% of budget cap){RESET}")

    # ── Check for checkpoint if needed ──
    if config.get("needs_checkpoint"):
        ckpt_name = config["needs_checkpoint"]
        ckpt_path = REPO_ROOT / "experiments" / "results" / "fridrich_renderer" / ckpt_name
        if not ckpt_path.exists():
            print(f"{RED}Missing checkpoint: {ckpt_path}{RESET}")
            print("Download from Modal first: modal volume get tac-asymmetric-results <tag>/renderer_best.pt")
            return 1
        print(f"  {GREEN}Checkpoint found: {ckpt_path}{RESET}")

    # ── Create instance ──
    print(f"\n{BOLD}Creating instance...{RESET}")
    label = f"{LABEL_PREFIX}-{experiment_name}"

    create_args = [
        "create", "instance", str(offer_id),
        "--image", DOCKER_IMAGE,
        "--disk", "40",
        "--onstart-cmd", ONSTART_SCRIPT,
        "--label", label,
        "--raw",
    ]

    result = run_vastai(create_args)
    if result.returncode != 0:
        print(f"{RED}Instance creation failed: {result.stderr}{RESET}")
        # Try relaxed constraints
        print(f"\n{YELLOW}Retrying with relaxed constraints...{RESET}")
        relaxed_query = (
            "gpu_name='RTX 4090' "
            "num_gpus=1 "
            "reliability>0.90 "
            "inet_down>100 "
            "disk_space>25 "
            "rentable=True"
        )
        result2 = run_vastai(["search", "offers", relaxed_query, "--order", "dph_total", "--limit", "3", "--raw"])
        if result2.returncode != 0 or not result2.stdout.strip():
            print(f"{RED}No offers even with relaxed constraints{RESET}")
            return 1
        try:
            relaxed_offers = json.loads(result2.stdout)
        except json.JSONDecodeError:
            print(f"{RED}Failed to parse relaxed offers{RESET}")
            return 1
        if not relaxed_offers:
            print(f"{RED}No offers available{RESET}")
            return 1
        offer_id = relaxed_offers[0]["id"]
        dph = relaxed_offers[0].get("dph_total", relaxed_offers[0].get("dph", 0.0))
        estimated_cost = dph * timeout_hours
        if not check_budget_remaining(budget, estimated_cost):
            return 1
        create_args[2] = str(offer_id)
        result = run_vastai(create_args)
        if result.returncode != 0:
            print(f"{RED}Instance creation failed on retry: {result.stderr}{RESET}")
            return 1

    # Parse instance ID from response
    # vastai returns: {"success": true, "new_contract": 12345} or just text
    try:
        create_resp = json.loads(result.stdout)
        if isinstance(create_resp, dict):
            instance_id = str(create_resp.get("new_contract", create_resp.get("id", "")))
        else:
            instance_id = str(create_resp)
    except (json.JSONDecodeError, TypeError):
        # Try to extract ID from text response (e.g., "Started. 12345")
        id_match = re.search(r'\d{4,}', result.stdout)
        instance_id = id_match.group(0) if id_match else ""

    if not instance_id or instance_id == "":
        print(f"{RED}Could not determine instance ID from response:{RESET}")
        print(result.stdout)
        return 1

    print(f"  {GREEN}Instance created: {instance_id}{RESET}")

    # ── Wait for instance to be running ──
    print(f"\n{BOLD}Waiting for instance to start...{RESET}")
    ssh_host = None
    ssh_port = None
    for _ in range(60):  # up to 10 minutes
        time.sleep(10)
        result = run_vastai(["show", "instance", instance_id, "--raw"])
        if result.returncode != 0:
            continue
        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue

        status = info.get("actual_status", info.get("status_msg", ""))
        if status == "running":
            ssh_host = info.get("ssh_host", info.get("public_ipaddr"))
            ssh_port = info.get("ssh_port", info.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort", 22))
            if ssh_host and ssh_port:
                print(f"  {GREEN}Instance running: {ssh_host}:{ssh_port}{RESET}")
                break
        elif status in ("exited", "error"):
            print(f"  {RED}Instance failed to start: {status}{RESET}")
            run_vastai(["destroy", "instance", instance_id])
            return 1

    if not ssh_host or not ssh_port:
        print(f"{RED}Timed out waiting for instance to start{RESET}")
        run_vastai(["destroy", "instance", instance_id])
        return 1

    # ── Generate ephemeral SSH key ──
    print(f"\n{BOLD}Setting up SSH access...{RESET}")
    key_path, pub_key = generate_ssh_key(instance_id)

    # Upload public key to instance via vastai CLI
    result = run_vastai(["ssh-key", "put", pub_key])
    if result.returncode != 0:
        print(f"  {YELLOW}Warning: Could not push SSH key via API: {result.stderr}{RESET}")
        # Fall back: the instance may use the default Vast.ai SSH key
        # Check if ~/.ssh/id_ed25519 or ~/.ssh/id_rsa exists
        for default_key in [Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"]:
            if default_key.exists():
                key_path = str(default_key)
                print(f"  {YELLOW}Using default SSH key: {key_path}{RESET}")
                break

    # ── Wait for SSH ──
    if not wait_for_ssh(ssh_host, int(ssh_port), key_path):
        print(f"{RED}Cannot reach instance via SSH. Destroying...{RESET}")
        run_vastai(["destroy", "instance", instance_id])
        cleanup_ssh_key(instance_id)
        return 1

    # Register instance
    register_instance(instance_id, experiment_name, dph, ssh_host, int(ssh_port), timeout_hours)

    # ── Wait for onstart setup to complete ──
    print(f"\n{BOLD}Waiting for setup script...{RESET}")
    if not wait_for_setup(ssh_host, int(ssh_port), key_path):
        print(f"{YELLOW}Setup may still be running, proceeding with upload...{RESET}")

    # ── Upload code ──
    print(f"\n{BOLD}Uploading code...{RESET}")

    # Upload src/tac/
    if not rsync_upload(ssh_host, int(ssh_port), key_path,
                        str(REPO_ROOT / "src" / "tac") + "/", "/workspace/src/tac/"):
        print(f"{RED}Failed to upload src/tac{RESET}")
        destroy_instance(instance_id)
        return 1

    # Upload experiments/ (excluding large files)
    if not rsync_upload(ssh_host, int(ssh_port), key_path,
                        str(REPO_ROOT / "experiments") + "/", "/workspace/experiments/"):
        print(f"{RED}Failed to upload experiments{RESET}")
        destroy_instance(instance_id)
        return 1

    # Upload checkpoint if needed
    if config.get("needs_checkpoint"):
        ckpt_name = config["needs_checkpoint"]
        ckpt_path = REPO_ROOT / "experiments" / "results" / "fridrich_renderer" / ckpt_name
        print(f"  Uploading checkpoint: {ckpt_name}")
        if not rsync_upload(ssh_host, int(ssh_port), key_path,
                            str(ckpt_path), f"/workspace/{ckpt_name}"):
            print(f"{RED}Failed to upload checkpoint{RESET}")
            destroy_instance(instance_id)
            return 1

    print(f"  {GREEN}Upload complete{RESET}")

    # ── Run the experiment ──
    print(f"\n{BOLD}Starting experiment: {experiment_name}{RESET}")
    timeout_seconds = int(timeout_hours * 3600)

    # Build the run script on the remote machine, then execute it detached.
    # Writing a script file avoids shell quoting issues with nohup + SSH.
    run_script = (
        "#!/bin/bash\n"
        "cd /workspace\n"
        "source .venv/bin/activate\n"
        "export PYTHONPATH=/workspace/src:/workspace/upstream\n"
        "export PYTHONUNBUFFERED=1\n"
        "export TAC_UPSTREAM_DIR=/workspace/upstream\n"
        "export TAC_MODELS_DIR=/workspace/upstream/models\n"
        f"timeout {timeout_seconds} python {config['script']} {config['args']} "
        "> /workspace/experiment_stdout.log 2>&1\n"
        "EXIT_STATUS=$?\n"
        'echo "EXIT_CODE=$EXIT_STATUS" >> /workspace/experiment_stdout.log\n'
        "echo EXPERIMENT_DONE > /workspace/.experiment_done\n"
    )

    # Upload the run script, then launch it fully detached
    ssh_exec(ssh_host, int(ssh_port), key_path,
             f"cat > /workspace/run_experiment.sh << 'RUNEOF'\n{run_script}RUNEOF",
             timeout=15)
    ssh_exec(ssh_host, int(ssh_port), key_path,
             "chmod +x /workspace/run_experiment.sh", timeout=10)

    # Use nohup + redirect + disown pattern for reliable detachment
    ssh_exec(ssh_host, int(ssh_port), key_path,
             "nohup /workspace/run_experiment.sh </dev/null >/workspace/nohup.out 2>&1 &",
             timeout=10)

    update_instance_status(instance_id, "running")
    print(f"  {GREEN}Experiment launched in background{RESET}")
    print(f"  Instance ID: {instance_id}")
    print(f"  SSH: ssh -i {key_path} -p {ssh_port} root@{ssh_host}")
    print(f"  Monitor: python scripts/vastai_deploy.py status")
    print(f"  Results: python scripts/vastai_deploy.py results")
    print(f"  Timeout: {timeout_hours}h (auto-destroy at {timeout_seconds}s)")

    # Record estimated cost
    record_spend(budget, instance_id, 0.0,
                 f"launched {experiment_name} at ${dph:.3f}/hr (estimated ${estimated_cost:.2f})")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Check status of all tracked instances."""
    instances = load_instances()
    budget = load_budget()

    print(f"\n{BOLD}Vast.ai Instance Status{RESET}")
    print(f"{'=' * 70}")
    print(f"  Budget: ${budget['total_spent']:.2f} / ${BUDGET_HARD_CAP:.2f} "
          f"(${BUDGET_HARD_CAP - budget['total_spent']:.2f} remaining)")
    print()

    if not instances:
        print(f"  {DIM}No tracked instances{RESET}")
        return 0

    now = datetime.now(timezone.utc)
    instances_to_destroy = []

    for iid, info in instances.items():
        experiment = info["experiment"]
        dph = info["dph"]
        created = datetime.fromisoformat(info["created_at"])
        elapsed_hours = (now - created).total_seconds() / 3600
        estimated_cost = dph * elapsed_hours
        timeout_hours = info["timeout_hours"]
        local_status = info.get("status", "unknown")

        # Check if still running on Vast.ai
        result = run_vastai(["show", "instance", iid, "--raw"])
        if result.returncode != 0:
            vastai_status = "destroyed/not-found"
        else:
            try:
                remote_info = json.loads(result.stdout)
                vastai_status = remote_info.get("actual_status", remote_info.get("status_msg", "unknown"))
            except json.JSONDecodeError:
                vastai_status = "unknown"

        # Check for experiment completion via SSH
        experiment_done = False
        if vastai_status == "running" and info.get("ssh_host"):
            key_path = str(SSH_KEY_DIR / f"vastai_{iid}")
            if not Path(key_path).exists():
                # Try default keys
                for default_key in [Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"]:
                    if default_key.exists():
                        key_path = str(default_key)
                        break
            try:
                check = ssh_exec(info["ssh_host"], info["ssh_port"], key_path,
                                 "cat /workspace/.experiment_done 2>/dev/null", timeout=15)
                if "EXPERIMENT_DONE" in check.stdout:
                    experiment_done = True
                    local_status = "completed"
                    update_instance_status(iid, "completed")
            except (subprocess.TimeoutExpired, OSError):
                pass

        # Color-code status
        if vastai_status == "running" and not experiment_done:
            status_str = f"{GREEN}{BOLD}RUNNING{RESET}"
        elif experiment_done:
            status_str = f"{BLUE}{BOLD}COMPLETED{RESET}"
        elif vastai_status in ("exited", "error"):
            status_str = f"{RED}{BOLD}{vastai_status.upper()}{RESET}"
        elif "destroyed" in vastai_status or "not-found" in vastai_status:
            status_str = f"{DIM}DESTROYED{RESET}"
        else:
            status_str = f"{YELLOW}{vastai_status}{RESET}"

        # Check timeout
        over_timeout = elapsed_hours > timeout_hours
        timeout_str = f"{RED}OVER LIMIT{RESET}" if over_timeout else f"{elapsed_hours:.1f}/{timeout_hours}h"

        print(f"  {BOLD}[{iid}]{RESET} {experiment}")
        print(f"    Status: {status_str} | Elapsed: {timeout_str}")
        print(f"    Cost: ${estimated_cost:.2f} (${dph:.3f}/hr)")
        if info.get("ssh_host"):
            print(f"    SSH: root@{info['ssh_host']}:{info['ssh_port']}")

        # Auto-destroy on timeout
        if over_timeout and vastai_status == "running":
            print(f"    {RED}{BOLD}TIMEOUT EXCEEDED — marking for auto-destroy{RESET}")
            instances_to_destroy.append(iid)

        # Budget overrun check
        if budget["total_spent"] + estimated_cost > BUDGET_HARD_CAP and vastai_status == "running":
            print(f"    {RED}{BOLD}BUDGET OVERRUN — marking for auto-destroy{RESET}")
            instances_to_destroy.append(iid)

        print()

    # Auto-destroy timed out / over-budget instances
    if instances_to_destroy:
        print(f"{RED}{BOLD}Auto-destroying {len(instances_to_destroy)} instance(s):{RESET}")
        for iid in set(instances_to_destroy):
            # Try to download results first
            info = instances[iid]
            if info.get("ssh_host"):
                print(f"  Saving partial results from {iid}...")
                _download_instance_results(iid, info)
            destroy_instance(iid)

    return 0


def cmd_results(args: argparse.Namespace) -> int:
    """Download results from completed (or running) instances."""
    instances = load_instances()
    budget = load_budget()

    print(f"\n{BOLD}Downloading results...{RESET}\n")

    if not instances:
        print(f"  {DIM}No tracked instances{RESET}")
        return 0

    downloaded = 0
    for iid, info in instances.items():
        if info.get("status") in ("destroyed", "results_downloaded"):
            continue

        print(f"  {BOLD}[{iid}]{RESET} {info['experiment']}")
        success = _download_instance_results(iid, info)
        if success:
            downloaded += 1
            update_instance_status(iid, "results_downloaded")

            # If experiment is done, record final cost and destroy
            if info.get("status") == "completed":
                created = datetime.fromisoformat(info["created_at"])
                elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
                final_cost = info["dph"] * elapsed_hours
                record_spend(budget, iid, final_cost,
                             f"final cost for {info['experiment']}: {elapsed_hours:.1f}h")
                print(f"    Final cost: ${final_cost:.2f}")

                if not args.keep:
                    print(f"    Destroying instance...")
                    destroy_instance(iid)

    print(f"\n  Downloaded from {downloaded} instance(s)")
    return 0


def _download_instance_results(instance_id: str, info: dict) -> bool:
    """Download results from a single instance."""
    ssh_host = info.get("ssh_host")
    ssh_port = info.get("ssh_port")
    if not ssh_host or not ssh_port:
        print(f"    {YELLOW}No SSH info available{RESET}")
        return False

    key_path = str(SSH_KEY_DIR / f"vastai_{instance_id}")
    if not Path(key_path).exists():
        for default_key in [Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"]:
            if default_key.exists():
                key_path = str(default_key)
                break

    # Check if instance is reachable
    try:
        check = ssh_exec(ssh_host, ssh_port, key_path, "ls /workspace/", timeout=15)
        if check.returncode != 0:
            print(f"    {RED}Instance not reachable via SSH{RESET}")
            return False
    except (subprocess.TimeoutExpired, OSError):
        print(f"    {RED}SSH timeout{RESET}")
        return False

    # Create local results directory
    experiment = info["experiment"]
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    local_dir = str(RESULTS_DIR / f"{experiment}_{ts}")

    # Download experiment results
    print(f"    Downloading to {local_dir}/")

    # Get stdout log
    rsync_download(ssh_host, ssh_port, key_path,
                   "/workspace/experiment_stdout.log", local_dir + "/")

    # Get any results the experiment produced
    # Check what result directories exist
    try:
        ls_result = ssh_exec(ssh_host, ssh_port, key_path,
                             "ls /workspace/experiments/results/ 2>/dev/null", timeout=15)
        if ls_result.returncode == 0 and ls_result.stdout.strip():
            rsync_download(ssh_host, ssh_port, key_path,
                           "/workspace/experiments/results/", local_dir + "/results/")
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Download any .pt files in /workspace
    try:
        pt_result = ssh_exec(ssh_host, ssh_port, key_path,
                             "ls /workspace/*.pt 2>/dev/null", timeout=15)
        if pt_result.returncode == 0 and pt_result.stdout.strip():
            for pt_file in pt_result.stdout.strip().split("\n"):
                pt_file = pt_file.strip()
                if pt_file:
                    rsync_download(ssh_host, ssh_port, key_path,
                                   pt_file, local_dir + "/")
    except (subprocess.TimeoutExpired, OSError):
        pass

    print(f"    {GREEN}Results saved to {local_dir}{RESET}")
    return True


def cmd_destroy(args: argparse.Namespace) -> int:
    """Destroy instances."""
    instances = load_instances()
    budget = load_budget()

    if args.all:
        targets = list(instances.keys())
    elif args.instance_id:
        targets = [args.instance_id]
    else:
        # Destroy only completed/error instances
        targets = [iid for iid, info in instances.items()
                   if info.get("status") in ("completed", "error", "results_downloaded")]

    if not targets:
        print(f"  {DIM}No instances to destroy{RESET}")
        return 0

    print(f"\n{BOLD}Destroying {len(targets)} instance(s)...{RESET}\n")

    for iid in targets:
        info = instances.get(iid, {})

        # Record final cost before destroying
        if info.get("created_at") and info.get("dph"):
            created = datetime.fromisoformat(info["created_at"])
            elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            final_cost = info["dph"] * elapsed_hours
            # Only record if not already recorded
            already_recorded = any(s.get("instance_id") == iid and s.get("amount", 0) > 0
                                   for s in budget.get("sessions", []))
            if not already_recorded:
                record_spend(budget, iid, final_cost,
                             f"final cost for {info.get('experiment', 'unknown')}: {elapsed_hours:.1f}h")

        destroy_instance(iid)
        print(f"  {GREEN}Destroyed {iid} ({info.get('experiment', 'unknown')}){RESET}")

    return 0


def destroy_instance(instance_id: str):
    """Destroy a single instance and clean up SSH keys."""
    result = run_vastai(["destroy", "instance", str(instance_id)])
    if result.returncode != 0:
        # Instance might already be destroyed — that is fine
        pass
    cleanup_ssh_key(instance_id)
    update_instance_status(instance_id, "destroyed")


def cmd_budget(args: argparse.Namespace) -> int:
    """Show budget tracking."""
    budget = load_budget()

    remaining = BUDGET_HARD_CAP - budget["total_spent"]
    pct = (budget["total_spent"] / BUDGET_HARD_CAP) * 100 if BUDGET_HARD_CAP > 0 else 0

    print(f"\n{BOLD}Vast.ai Budget Tracker{RESET}")
    print(f"{'=' * 50}")

    if remaining < 0:
        color = RED
    elif budget["total_spent"] > BUDGET_WARN_THRESHOLD:
        color = YELLOW
    else:
        color = GREEN

    print(f"  Total spent:  {color}${budget['total_spent']:.2f}{RESET}")
    print(f"  Hard cap:     ${BUDGET_HARD_CAP:.2f}")
    print(f"  Remaining:    {color}${remaining:.2f}{RESET} ({100 - pct:.0f}%)")
    print(f"  Total budget: ${BUDGET_TOTAL:.2f}")

    if budget.get("sessions"):
        print(f"\n  {BOLD}Session history:{RESET}")
        for session in budget["sessions"][-10:]:  # last 10
            ts = session["timestamp"][:19].replace("T", " ")
            amt = session["amount"]
            desc = session["description"]
            if amt > 0:
                print(f"    {ts}  ${amt:>7.2f}  {desc}")
            else:
                print(f"    {DIM}{ts}  ${amt:>7.2f}  {desc}{RESET}")
        if len(budget["sessions"]) > 10:
            print(f"    {DIM}... and {len(budget['sessions']) - 10} earlier entries{RESET}")

    print()
    return 0


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> int:
    check_api_key()
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="Vast.ai RTX 4090 deployment for pact-lab experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s search                          Find best 4090 offers
  %(prog)s launch --experiment tto_v1      Launch TTO v1 experiment
  %(prog)s status                          Check instance statuses
  %(prog)s results                         Download results
  %(prog)s destroy --all                   Destroy all instances
  %(prog)s budget                          Show spend tracking

Available experiments:
  {chr(10).join(f'  {name:20s} ({cfg["timeout_hours"]}h timeout)' for name, cfg in EXPERIMENTS.items())}
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search
    subparsers.add_parser("search", help="Find best RTX 4090 offers")

    # launch
    launch_p = subparsers.add_parser("launch", help="Launch an experiment")
    launch_p.add_argument("--experiment", "-e", required=True, choices=list(EXPERIMENTS.keys()),
                          help="Experiment to run")

    # status
    subparsers.add_parser("status", help="Check all instance statuses")

    # results
    results_p = subparsers.add_parser("results", help="Download results")
    results_p.add_argument("--keep", action="store_true",
                           help="Keep instances after downloading (don't auto-destroy)")

    # destroy
    destroy_p = subparsers.add_parser("destroy", help="Destroy instances")
    destroy_p.add_argument("--all", action="store_true", help="Destroy ALL tracked instances")
    destroy_p.add_argument("--instance-id", "-i", help="Destroy a specific instance")

    # budget
    subparsers.add_parser("budget", help="Show spend tracking")

    args = parser.parse_args()

    dispatch = {
        "search": cmd_search,
        "launch": cmd_launch,
        "status": cmd_status,
        "results": cmd_results,
        "destroy": cmd_destroy,
        "budget": cmd_budget,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
