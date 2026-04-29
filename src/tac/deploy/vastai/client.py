"""Vast.ai client: SSH, rsync, and instance lifecycle management.

Encapsulates all interaction with the ``vastai`` CLI and remote instances.
Provides context-manager support for automatic instance teardown.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from tac.deploy.base import ExperimentConfig, InstanceSpec, repo_root
from tac.deploy.vastai.budget import BudgetTracker

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_PREFIX = "pact-lab"
"""Instance label prefix for tracking our instances on Vast.ai."""

API_KEY_FILE = Path.home() / ".vast_api_key"
"""Default location where ``vastai set api-key`` stores the key."""

SSH_CONNECT_TIMEOUT_SEC = 30
SSH_ALIVE_INTERVAL_SEC = 15
SSH_ALIVE_COUNT_MAX = 4
SSH_POLL_INTERVAL_SEC = 15
SSH_MAX_WAIT_SEC = 600
SETUP_POLL_INTERVAL_SEC = 20
SETUP_MAX_WAIT_SEC = 900
RSYNC_MAX_RETRIES = 3
RSYNC_RETRY_DELAY_SEC = 10
RSYNC_TIMEOUT_SEC = 600
INSTANCE_POLL_INTERVAL_SEC = 10
INSTANCE_POLL_MAX_ATTEMPTS = 60

# ANSI colors
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Onstart script template: runs when the instance boots.
# The Docker image already has PyTorch 2.5.1 + CUDA 12.4 installed system-wide.
# We create a venv with --system-site-packages to inherit torch, then only
# install the missing dependencies.
ONSTART_SCRIPT = """\
#!/bin/bash
set -ex

# Install uv for package management
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv inheriting system torch from the Docker image
cd /workspace
uv venv --system-site-packages
source .venv/bin/activate

# Only install deps NOT in the base image (torch/torchvision already present)
uv pip install av safetensors segmentation-models-pytorch timm einops pydantic click

# Clone upstream scorer (PoseNet/SegNet models + GT video)
if [ ! -d /workspace/upstream ]; then
    apt-get update && apt-get install -y git-lfs
    git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /workspace/upstream
    cd /workspace/upstream && git lfs pull
fi

echo "SETUP_COMPLETE" > /workspace/.setup_done
"""


@dataclass
class InstanceInfo:
    """Tracked state for a running Vast.ai instance."""

    instance_id: str
    experiment: str
    dph: float
    ssh_host: str
    ssh_port: int
    created_at: str
    timeout_hours: float
    status: str = "created"
    ssh_key_path: str | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "experiment": self.experiment,
            "dph": self.dph,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "created_at": self.created_at,
            "timeout_hours": self.timeout_hours,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, instance_id: str, data: dict) -> InstanceInfo:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            instance_id=instance_id,
            experiment=data["experiment"],
            dph=data["dph"],
            ssh_host=data["ssh_host"],
            ssh_port=data["ssh_port"],
            created_at=data["created_at"],
            timeout_hours=data["timeout_hours"],
            status=data.get("status", "unknown"),
        )


class VastClient:
    """High-level client for Vast.ai instance lifecycle management.

    Parameters
    ----------
    budget:
        Budget tracker for cost enforcement.
    results_dir:
        Directory for instance tracking files and SSH keys.
    """

    def __init__(
        self,
        budget: BudgetTracker | None = None,
        results_dir: Path | None = None,
    ) -> None:
        if results_dir is None:
            results_dir = repo_root() / "experiments" / "results" / "vastai"
        self._results_dir = Path(results_dir)
        self._instances_file = self._results_dir / "instances.json"
        self._ssh_key_dir = self._results_dir / ".ssh"
        self._budget = budget or BudgetTracker()
        self._ensure_dirs()

    # ── Context manager for instances ─────────────────────────────────────

    @contextmanager
    def create_instance(
        self,
        spec: InstanceSpec,
        experiment: ExperimentConfig,
    ) -> Generator[InstanceInfo, None, None]:
        """Provision a Vast.ai instance and destroy it on exit.

        Usage::

            with client.create_instance(spec, experiment) as inst:
                client.upload_code(inst)
                client.run_experiment(inst, experiment)

        The instance is destroyed when the context exits, even on exception.
        """
        inst: InstanceInfo | None = None
        try:
            inst = self._provision(spec, experiment)
            yield inst
        finally:
            if inst is not None:
                self.destroy(inst.instance_id)

    # ── Public methods ────────────────────────────────────────────────────

    @property
    def budget(self) -> BudgetTracker:
        """Access the budget tracker."""
        return self._budget

    def check_api_key(self) -> None:
        """Verify that the Vast.ai API key file exists. Exits on failure."""
        if not API_KEY_FILE.exists():
            print(
                f"{_RED}No Vast.ai API key found at {API_KEY_FILE}{_RESET}\n"
                "Set it with: vastai set api-key <your-key>",
                file=sys.stderr,
            )
            sys.exit(1)

    def search_offers(
        self,
        spec: InstanceSpec,
        limit: int = 15,
        raw: bool = False,
    ) -> list[dict[str, Any]] | str:
        """Search for Vast.ai offers matching *spec*.

        Returns parsed JSON list when *raw* is True, or the CLI's
        human-readable table string otherwise.
        """
        query = self._build_search_query(spec)
        extra_args = ["--order", "dph_total", "--limit", str(limit)]
        if raw:
            extra_args.append("--raw")

        result = self._run_cli(["search", "offers", query] + extra_args)
        if result.returncode != 0:
            raise RuntimeError(f"Offer search failed: {result.stderr}")

        if raw:
            return json.loads(result.stdout)
        return result.stdout

    def show_instance(self, instance_id: str) -> dict[str, Any]:
        """Query the Vast.ai API for current instance state."""
        result = self._run_cli(["show", "instance", instance_id, "--raw"])
        if result.returncode != 0:
            raise RuntimeError(f"show instance {instance_id} failed: {result.stderr}")
        return json.loads(result.stdout)

    def destroy(self, instance_id: str) -> None:
        """Destroy an instance and clean up local SSH keys."""
        self._run_cli(["destroy", "instance", str(instance_id)])
        self._cleanup_ssh_key(instance_id)
        self._update_instance_status(instance_id, "destroyed")

    def upload_code(self, inst: InstanceInfo) -> None:
        """Upload ``src/tac/`` and ``experiments/`` to the remote instance."""
        root = repo_root()
        key = self._resolve_ssh_key(inst)

        # Upload src/tac/
        if not self._rsync_upload(
            inst.ssh_host, inst.ssh_port, key,
            str(root / "src" / "tac") + "/",
            "/workspace/src/tac/",
        ):
            raise RuntimeError("Failed to upload src/tac")

        # Upload experiments/
        if not self._rsync_upload(
            inst.ssh_host, inst.ssh_port, key,
            str(root / "experiments") + "/",
            "/workspace/experiments/",
        ):
            raise RuntimeError("Failed to upload experiments")

    def upload_checkpoint(
        self,
        inst: InstanceInfo,
        checkpoint_name: str,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Upload a checkpoint file to ``/workspace/`` on the instance.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing the checkpoint, relative to repo root.
            Defaults to the canonical v5 Lagrangian renderer directory.
        """
        from tac.checkpoint import CANONICAL_CHECKPOINT_DIR, verify_checkpoint_identity

        root = repo_root()
        ckpt_dir = checkpoint_dir or CANONICAL_CHECKPOINT_DIR
        ckpt_path = root / ckpt_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {ckpt_path}\n"
                "Download from Modal first: modal volume get tac-asymmetric-results <tag>/renderer_best.pt"
            )

        # Sanity check: verify this is the trained model, not a smoke-test
        md5 = verify_checkpoint_identity(ckpt_path)
        print(f"  {_GREEN}Checkpoint verified (MD5: {md5}){_RESET}")

        key = self._resolve_ssh_key(inst)
        if not self._rsync_upload(
            inst.ssh_host, inst.ssh_port, key,
            str(ckpt_path),
            f"/workspace/{checkpoint_name}",
        ):
            raise RuntimeError(f"Failed to upload checkpoint {checkpoint_name}")

    def run_experiment(self, inst: InstanceInfo, experiment: ExperimentConfig) -> None:
        """Launch the experiment script on the remote instance (detached).

        The script runs under ``nohup`` so it survives SSH disconnection.
        """
        key = self._resolve_ssh_key(inst)
        args_str = " ".join(experiment.args)

        run_script = (
            "#!/bin/bash\n"
            "cd /workspace\n"
            "source .venv/bin/activate\n"
            "export PYTHONPATH=/workspace/src:/workspace/upstream\n"
            "export PYTHONUNBUFFERED=1\n"
            "export TAC_UPSTREAM_DIR=/workspace/upstream\n"
            "export TAC_MODELS_DIR=/workspace/upstream/models\n"
            f"timeout {experiment.timeout_seconds} python {experiment.script} {args_str} "
            "> /workspace/experiment_stdout.log 2>&1\n"
            "EXIT_STATUS=$?\n"
            'echo "EXIT_CODE=$EXIT_STATUS" >> /workspace/experiment_stdout.log\n'
            "echo EXPERIMENT_DONE > /workspace/.experiment_done\n"
        )

        self._ssh_exec(
            inst.ssh_host, inst.ssh_port, key,
            f"cat > /workspace/run_experiment.sh << 'RUNEOF'\n{run_script}RUNEOF",
            timeout=15,
        )
        self._ssh_exec(
            inst.ssh_host, inst.ssh_port, key,
            "chmod +x /workspace/run_experiment.sh",
            timeout=10,
        )
        self._ssh_exec(
            inst.ssh_host, inst.ssh_port, key,
            "nohup /workspace/run_experiment.sh </dev/null >/workspace/nohup.out 2>&1 &",
            timeout=10,
        )
        self._update_instance_status(inst.instance_id, "running")

    def check_experiment_done(self, inst: InstanceInfo) -> bool:
        """Check if the remote experiment has written its completion marker."""
        key = self._resolve_ssh_key(inst)
        try:
            result = self._ssh_exec(
                inst.ssh_host, inst.ssh_port, key,
                "cat /workspace/.experiment_done 2>/dev/null",
                timeout=15,
            )
            return "EXPERIMENT_DONE" in result.stdout
        except (subprocess.TimeoutExpired, OSError):
            return False

    def download_results(self, inst: InstanceInfo) -> Path:
        """Download experiment results to a timestamped local directory.

        Returns the local directory path where results were saved.
        """
        key = self._resolve_ssh_key(inst)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        local_dir = self._results_dir / f"{inst.experiment}_{ts}"

        # Stdout log
        self._rsync_download(
            inst.ssh_host, inst.ssh_port, key,
            "/workspace/experiment_stdout.log",
            str(local_dir) + "/",
        )

        # Experiment result directories
        try:
            ls_result = self._ssh_exec(
                inst.ssh_host, inst.ssh_port, key,
                "ls /workspace/experiments/results/ 2>/dev/null",
                timeout=15,
            )
            if ls_result.returncode == 0 and ls_result.stdout.strip():
                self._rsync_download(
                    inst.ssh_host, inst.ssh_port, key,
                    "/workspace/experiments/results/",
                    str(local_dir / "results") + "/",
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Checkpoint files
        try:
            pt_result = self._ssh_exec(
                inst.ssh_host, inst.ssh_port, key,
                "ls /workspace/*.pt 2>/dev/null",
                timeout=15,
            )
            if pt_result.returncode == 0 and pt_result.stdout.strip():
                for pt_file in pt_result.stdout.strip().split("\n"):
                    pt_file = pt_file.strip()
                    if pt_file:
                        self._rsync_download(
                            inst.ssh_host, inst.ssh_port, key,
                            pt_file,
                            str(local_dir) + "/",
                        )
        except (subprocess.TimeoutExpired, OSError):
            pass

        return local_dir

    def wait_for_ssh(self, inst: InstanceInfo) -> bool:
        """Block until SSH is reachable on the instance.

        Returns True on success, False on timeout.
        """
        key = self._resolve_ssh_key(inst)
        start = time.time()
        attempt = 0
        while time.time() - start < SSH_MAX_WAIT_SEC:
            attempt += 1
            try:
                result = subprocess.run(
                    ["ssh"] + self._ssh_opts(key) + [
                        "-p", str(inst.ssh_port),
                        f"root@{inst.ssh_host}",
                        "echo ok",
                    ],
                    capture_output=True, text=True, timeout=35,
                )
                if result.returncode == 0 and "ok" in result.stdout:
                    elapsed = time.time() - start
                    print(f"  {_GREEN}SSH ready after {elapsed:.0f}s ({attempt} attempts){_RESET}")
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            if attempt % 4 == 0:
                print(f"  {_DIM}Waiting for SSH... ({int(time.time() - start)}s elapsed){_RESET}")
            time.sleep(SSH_POLL_INTERVAL_SEC)

        print(f"  {_RED}SSH timeout after {SSH_MAX_WAIT_SEC}s{_RESET}")
        return False

    def wait_for_setup(self, inst: InstanceInfo) -> bool:
        """Block until the onstart script writes its completion marker.

        Returns True on success, False on timeout.
        """
        key = self._resolve_ssh_key(inst)
        start = time.time()
        while time.time() - start < SETUP_MAX_WAIT_SEC:
            try:
                result = subprocess.run(
                    ["ssh"] + self._ssh_opts(key) + [
                        "-p", str(inst.ssh_port),
                        f"root@{inst.ssh_host}",
                        "cat /workspace/.setup_done 2>/dev/null || echo PENDING",
                    ],
                    capture_output=True, text=True, timeout=35,
                )
                if result.returncode == 0 and "SETUP_COMPLETE" in result.stdout:
                    elapsed = time.time() - start
                    print(f"  {_GREEN}Setup complete after {elapsed:.0f}s{_RESET}")
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            elapsed = int(time.time() - start)
            if elapsed % 60 < SETUP_POLL_INTERVAL_SEC:
                print(f"  {_DIM}Waiting for setup... ({elapsed}s elapsed){_RESET}")
            time.sleep(SETUP_POLL_INTERVAL_SEC)

        print(f"  {_RED}Setup timeout after {SETUP_MAX_WAIT_SEC}s{_RESET}")
        return False

    def load_instances(self) -> dict[str, InstanceInfo]:
        """Load all tracked instances from disk."""
        if not self._instances_file.exists():
            return {}
        raw = json.loads(self._instances_file.read_text())
        return {
            iid: InstanceInfo.from_dict(iid, data)
            for iid, data in raw.items()
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _provision(
        self,
        spec: InstanceSpec,
        experiment: ExperimentConfig,
    ) -> InstanceInfo:
        """Full provisioning flow: search, create, wait for SSH + setup."""
        # Find cheapest offer
        offers = self.search_offers(spec, limit=5, raw=True)
        assert isinstance(offers, list)
        if not offers:
            raise RuntimeError(f"No {spec.gpu_type} offers matching constraints")

        best = offers[0]
        offer_id = best["id"]
        dph = best.get("dph_total", best.get("dph", 0.0))
        estimated_cost = dph * experiment.timeout_hours

        # Budget check
        if not self._budget.check_remaining(estimated_cost):
            raise RuntimeError("Budget check failed")

        # Create instance
        label = f"{LABEL_PREFIX}-{experiment.name}"
        create_args = [
            "create", "instance", str(offer_id),
            "--image", spec.docker_image,
            "--disk", str(int(spec.disk_gb)),
            "--ssh", "--direct",  # Enable direct SSH (not proxy) on port 22
            "--onstart-cmd", ONSTART_SCRIPT,
            "--label", label,
            "--raw",
        ]
        result = self._run_cli(create_args)
        if result.returncode != 0:
            raise RuntimeError(f"Instance creation failed: {result.stderr}")

        instance_id = self._parse_instance_id(result.stdout)
        if not instance_id:
            raise RuntimeError(f"Could not determine instance ID from: {result.stdout}")

        print(f"  {_GREEN}Instance created: {instance_id}{_RESET}")

        # Wait for running state + SSH info
        ssh_host, ssh_port = self._wait_for_running(instance_id)

        # Register ephemeral SSH key with Vast.ai account
        # The `create ssh-key` command adds the key globally (auto-applied to all instances).
        # See: https://docs.vast.ai/api-reference/accounts/create-ssh-key
        key_path, pub_key = self._generate_ssh_key(instance_id)
        key_result = self._run_cli(["create", "ssh-key", pub_key])
        if key_result.returncode != 0:
            print(f"  {_YELLOW}SSH key registration failed, falling back to default key{_RESET}")
            key_path = None  # Will use default ~/.ssh/id_* keys

        inst = InstanceInfo(
            instance_id=instance_id,
            experiment=experiment.name,
            dph=dph,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            created_at=datetime.now(timezone.utc).isoformat(),
            timeout_hours=experiment.timeout_hours,
            status="created",
            ssh_key_path=key_path,
        )
        self._save_instance(inst)

        # Centralized vastai_active_instances tracker (CLAUDE.md non-negotiable
        # + memory feedback_oneshot_vastai_subagent_failure_pattern). The
        # `vastai_active_instances.json` file is the single source of truth
        # that `tools/vastai_orphan_cleanup.py` reads to detect orphans
        # (instances created but never destroyed). _save_instance above is
        # this client's per-deploy bookkeeping; this is cross-script audit.
        try:
            from tac.vastai_tracker import register_instance
            register_instance(
                instance_id=instance_id,
                label=label,
                metadata={
                    "experiment": experiment.name,
                    "dph": float(dph),
                    "ssh_host": ssh_host,
                    "ssh_port": int(ssh_port),
                    "source_script": "src/tac/deploy/vastai/client.py",
                },
            )
        except Exception as e:  # pragma: no cover — tracker must not block launch
            print(f"  {_YELLOW}vastai_active_instances tracker write failed: {e!r}{_RESET}")

        # Wait for SSH + setup
        if not self.wait_for_ssh(inst):
            self.destroy(instance_id)
            raise RuntimeError("SSH timeout")

        print(f"\n{_BOLD}Waiting for setup script...{_RESET}")
        if not self.wait_for_setup(inst):
            print(f"{_YELLOW}Setup may still be running, proceeding...{_RESET}")

        return inst

    def _wait_for_running(self, instance_id: str) -> tuple[str, int]:
        """Poll until the instance reaches 'running' status. Returns (host, port)."""
        for _ in range(INSTANCE_POLL_MAX_ATTEMPTS):
            time.sleep(INSTANCE_POLL_INTERVAL_SEC)
            try:
                info = self.show_instance(instance_id)
            except RuntimeError:
                continue

            status = info.get("actual_status", info.get("status_msg", ""))
            if status == "running":
                ssh_host = info.get("ssh_host", info.get("public_ipaddr"))
                ssh_port = info.get("ssh_port")
                # Vast.ai uses high-numbered mapped ports, never port 22.
                # If ssh_port is missing, instance isn't fully provisioned yet.
                if not ssh_port:
                    continue
                if ssh_host and ssh_port:
                    print(f"  {_GREEN}Instance running: {ssh_host}:{ssh_port}{_RESET}")
                    return str(ssh_host), int(ssh_port)
            elif status in ("exited", "error"):
                self._run_cli(["destroy", "instance", instance_id])
                raise RuntimeError(f"Instance failed to start: {status}")

        self._run_cli(["destroy", "instance", instance_id])
        raise RuntimeError("Timed out waiting for instance to start")

    def _ensure_dirs(self) -> None:
        """Create results and SSH key directories."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._ssh_key_dir.mkdir(parents=True, exist_ok=True)
        self._ssh_key_dir.chmod(0o700)

    def _find_vastai_bin(self) -> str:
        """Locate the ``vastai`` CLI binary."""
        venv_bin = repo_root() / ".venv" / "bin" / "vastai"
        if venv_bin.exists():
            return str(venv_bin)
        found = shutil.which("vastai")
        if found:
            return found
        print(f"{_RED}vastai CLI not found. Install with: uv pip install vastai{_RESET}", file=sys.stderr)
        sys.exit(1)

    def _run_cli(
        self,
        args: list[str],
        capture: bool = True,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a ``vastai`` CLI command."""
        cli = self._find_vastai_bin()
        cmd = [cli] + args
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # subprocess-no-check-OK: wrapper returns CompletedProcess; every call site inspects .returncode
        return subprocess.run(cmd, text=True, timeout=timeout)  # subprocess-no-check-OK: wrapper returns CompletedProcess; every call site inspects .returncode

    @staticmethod
    def _build_search_query(spec: InstanceSpec) -> str:
        """Build a Vast.ai search query string from an InstanceSpec."""
        return (
            f"gpu_name={spec.gpu_type.replace(' ', '_')} "
            f"num_gpus={spec.num_gpus} "
            f"reliability>{spec.min_reliability} "
            f"inet_down>{spec.min_download_mbps} "
            f"disk_space>{spec.disk_gb} "
            f"rentable=True"
        )

    @staticmethod
    def _parse_instance_id(response_text: str) -> str:
        """Extract an instance ID from the Vast.ai create response."""
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                return str(data.get("new_contract", data.get("id", "")))
            return str(data)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\d{4,}", response_text)
            return match.group(0) if match else ""

    def _generate_ssh_key(self, instance_id: str) -> tuple[str, str]:
        """Generate an ephemeral ed25519 key pair. Returns (private_path, pub_content)."""
        self._ensure_dirs()
        key_path = self._ssh_key_dir / f"vastai_{instance_id}"
        key_path.unlink(missing_ok=True)
        key_path.with_suffix(".pub").unlink(missing_ok=True)

        subprocess.run(
            [
                "ssh-keygen", "-t", "ed25519",
                "-f", str(key_path),
                "-N", "",
                "-C", f"pact-lab-{instance_id}",
            ],
            capture_output=True, check=True,
        )
        key_path.chmod(0o600)
        pub_content = key_path.with_suffix(".pub").read_text().strip()
        return str(key_path), pub_content

    def _cleanup_ssh_key(self, instance_id: str) -> None:
        """Remove the ephemeral SSH key pair for an instance."""
        key_path = self._ssh_key_dir / f"vastai_{instance_id}"
        key_path.unlink(missing_ok=True)
        key_path.with_suffix(".pub").unlink(missing_ok=True)

    def _resolve_ssh_key(self, inst: InstanceInfo) -> str:
        """Find the SSH key for an instance, falling back to user defaults."""
        if inst.ssh_key_path and Path(inst.ssh_key_path).exists():
            return inst.ssh_key_path

        ephemeral = self._ssh_key_dir / f"vastai_{inst.instance_id}"
        if ephemeral.exists():
            return str(ephemeral)

        for default_key in [Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"]:
            if default_key.exists():
                return str(default_key)

        raise FileNotFoundError(f"No SSH key found for instance {inst.instance_id}")

    @staticmethod
    def _ssh_opts(key_path: str) -> list[str]:
        """Common SSH options for Vast.ai connections."""
        return [
            "-i", key_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SEC}",
            "-o", f"ServerAliveInterval={SSH_ALIVE_INTERVAL_SEC}",
            "-o", f"ServerAliveCountMax={SSH_ALIVE_COUNT_MAX}",
            "-o", "LogLevel=ERROR",
        ]

    def _ssh_exec(
        self,
        ssh_host: str,
        ssh_port: int,
        key_path: str,
        command: str,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        """Execute a command on a remote instance via SSH."""
        return subprocess.run(  # subprocess-no-check-OK: wrapper returns CompletedProcess; every caller inspects .returncode
            ["ssh"] + self._ssh_opts(key_path) + [
                "-p", str(ssh_port),
                f"root@{ssh_host}",
                command,
            ],
            capture_output=True, text=True, timeout=timeout,
        )

    def _rsync_upload(
        self,
        ssh_host: str,
        ssh_port: int,
        key_path: str,
        local_path: str,
        remote_path: str,
    ) -> bool:
        """Upload files via rsync over SSH with retry logic."""
        ssh_cmd = (
            "ssh " + " ".join(shlex.quote(o) for o in self._ssh_opts(key_path))
            + f" -p {ssh_port}"
        )
        for attempt in range(1, RSYNC_MAX_RETRIES + 1):
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
                text=True, timeout=RSYNC_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                return True
            if attempt < RSYNC_MAX_RETRIES:
                print(f"  {_YELLOW}rsync attempt {attempt} failed, retrying in {RSYNC_RETRY_DELAY_SEC}s...{_RESET}")
                time.sleep(RSYNC_RETRY_DELAY_SEC)

        print(f"  {_RED}rsync failed after {RSYNC_MAX_RETRIES} attempts{_RESET}")
        return False

    def _rsync_download(
        self,
        ssh_host: str,
        ssh_port: int,
        key_path: str,
        remote_path: str,
        local_path: str,
    ) -> bool:
        """Download files via rsync over SSH with retry logic."""
        ssh_cmd = (
            "ssh " + " ".join(shlex.quote(o) for o in self._ssh_opts(key_path))
            + f" -p {ssh_port}"
        )
        Path(local_path).mkdir(parents=True, exist_ok=True)
        for attempt in range(1, RSYNC_MAX_RETRIES + 1):
            result = subprocess.run(
                [
                    "rsync", "-avz", "--progress",
                    "-e", ssh_cmd,
                    f"root@{ssh_host}:{remote_path}",
                    local_path,
                ],
                text=True, timeout=RSYNC_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                return True
            if attempt < RSYNC_MAX_RETRIES:
                print(f"  {_YELLOW}rsync download attempt {attempt} failed, retrying in {RSYNC_RETRY_DELAY_SEC}s...{_RESET}")
                time.sleep(RSYNC_RETRY_DELAY_SEC)

        print(f"  {_RED}rsync download failed after {RSYNC_MAX_RETRIES} attempts{_RESET}")
        return False

    def _save_instance(self, inst: InstanceInfo) -> None:
        """Persist instance info to the tracking file."""
        instances = self._load_raw_instances()
        instances[inst.instance_id] = inst.to_dict()
        self._write_raw_instances(instances)

    def _update_instance_status(self, instance_id: str, status: str) -> None:
        """Update a single instance's status on disk."""
        instances = self._load_raw_instances()
        key = str(instance_id)
        if key in instances:
            instances[key]["status"] = status
            self._write_raw_instances(instances)

    def _load_raw_instances(self) -> dict[str, dict]:
        """Load raw instance tracking dict from disk."""
        if self._instances_file.exists():
            return json.loads(self._instances_file.read_text())
        return {}

    def _write_raw_instances(self, data: dict[str, dict]) -> None:
        """Write raw instance tracking dict to disk."""
        self._instances_file.parent.mkdir(parents=True, exist_ok=True)
        self._instances_file.write_text(json.dumps(data, indent=2) + "\n")
