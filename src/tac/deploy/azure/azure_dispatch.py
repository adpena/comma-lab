"""Azure VM-spot dispatch — wire Azure as a 4th GPU dispatch platform.

Mirrors the Vast.ai pattern (provision → SSH → run lane → harvest →
deprovision) because for single-instance lane scripts the heavyweight
Azure ML SDK is overkill. Spot VMs at 60-80% off on-demand make the $200
free-credits stretch.

Pricing reference (2026-04-30 Azure US East spot):
    Standard_NC6s_v3        1× V100  16GB   $3.06/hr → ~$0.50/hr spot
    Standard_NC24ads_A100_v4 1× A100 80GB   $3.67/hr → ~$1.10/hr spot
    Standard_NC40ads_H100_v5 1× H100 80GB   $6.98/hr → ~$2.30/hr spot

Constants in ``AZURE_GPU_PRICING`` reflect the published list+spot rates;
operators MUST verify spot pricing for their region at dispatch time
(spot prices are dynamic).

Pre-flight requirement: the operator must run ``az login`` once. The
helper ``ensure_azure_logged_in()`` raises a clear ``AzureNotLoggedIn``
error if no account is active so this module never silently dispatches
against the wrong subscription.

Mandatory non-negotiables (per CLAUDE.md):
- ALL commits via ``tools/subagent_commit_serializer.py``
- This module DOES NOT spawn VMs at import time
- Dispatch goes through ``scripts/launch_lane_azure.py`` (Pattern A nohup)
- Cost accounting must record spend to ``.omx/state/azure_active_vms.json``
  so the cleanup tooling can detect orphan VMs (analogous to
  ``.omx/state/vastai_active_instances.json``).
"""
from __future__ import annotations

import dataclasses
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

# ── Repo layout ──────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """Locate the repo root by searching upward for pyproject.toml."""
    candidate = Path(__file__).resolve().parent
    for _ in range(12):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError("Cannot locate repo root (no pyproject.toml found)")


REPO_ROOT = _repo_root()
ACTIVE_VMS_PATH = REPO_ROOT / ".omx" / "state" / "azure_active_vms.json"
# Codex round 5 HIGH 3 fix (2026-05-09, catalog #141): the helper module
# ``tac.deploy.azure.active_vms_state`` has its own module-level
# ``ACTIVE_VMS_PATH`` constant. When tests monkeypatched
# ``azd.ACTIVE_VMS_PATH`` to isolate writes, the override never reached the
# helper because the calls below did not thread ``path=ACTIVE_VMS_PATH`` /
# ``lock_path=ACTIVE_VMS_LOCK``. Tests' tracker round-trip silently observed
# the canonical file instead of the temp path. The same trap applied to
# real recovery/cleanup tooling that constructs an alternative tracker
# location for forensic inspection.
#
# Canonical sibling lockfile derived from the dispatch module's
# ``ACTIVE_VMS_PATH`` (same convention the helper uses internally).
ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix(ACTIVE_VMS_PATH.suffix + ".lock")


# ── Pricing reference ────────────────────────────────────────────────────────

# Reference prices in USD/hour. Spot rates are TYPICAL; real-time pricing
# is dynamic — operators should verify with `az vm list-skus -l <region>`
# or the Azure pricing API before any large dispatch.
AZURE_GPU_PRICING: dict[str, dict[str, float]] = {
    "Standard_NC6s_v3": {  # 1× V100 16GB
        "on_demand_usd_per_hour": 3.06,
        "spot_usd_per_hour": 0.50,
        "vram_gb": 16,
        "gpu": "V100",
    },
    "Standard_NC24ads_A100_v4": {  # 1× A100 80GB
        "on_demand_usd_per_hour": 3.67,
        "spot_usd_per_hour": 1.10,
        "vram_gb": 80,
        "gpu": "A100",
    },
    "Standard_NC40ads_H100_v5": {  # 1× H100 80GB
        "on_demand_usd_per_hour": 6.98,
        "spot_usd_per_hour": 2.30,
        "vram_gb": 80,
        "gpu": "H100",
    },
}

DEFAULT_GPU_TYPE = "Standard_NC6s_v3"  # cheapest spot for smoke + quick experiments
DEFAULT_REGION = "eastus"
DEFAULT_RESOURCE_GROUP = "pact-gpu-rg"
DEFAULT_IMAGE = (
    # Ubuntu 22.04 LTS Gen2 image (works with NVIDIA GPU drivers when the
    # nvidia-driver extension is installed at runtime)
    "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
)
DEFAULT_VM_USERNAME = "azureuser"
# $200 / 200hr ≈ $1/hr if we burn it evenly; cap at $0.50/hr default for spot V100
DEFAULT_SPOT_MAX_PRICE_USD = 0.50
# Safety: never exceed total $200 free-credit budget. This is checked
# against the running cost-projection ledger (cost_projection.md is the
# operator-facing source of truth; this constant is a hard ceiling).
AZURE_HARD_CAP_USD = 200.00


# ── Errors ───────────────────────────────────────────────────────────────────


class AzureNotLoggedIn(RuntimeError):
    """Raised when ``az`` CLI has no active subscription."""


class AzureCLIMissing(RuntimeError):
    """Raised when ``az`` is not on PATH."""


class AzureCommandError(RuntimeError):
    """Raised when an ``az`` command fails."""


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class AzureVMSpec:
    """Desired Azure VM provisioning spec."""

    label: str
    gpu_type: str = DEFAULT_GPU_TYPE
    region: str = DEFAULT_REGION
    resource_group: str = DEFAULT_RESOURCE_GROUP
    image: str = DEFAULT_IMAGE
    username: str = DEFAULT_VM_USERNAME
    spot: bool = True
    spot_max_price_usd: float = DEFAULT_SPOT_MAX_PRICE_USD
    ssh_pubkey_path: str = "~/.ssh/id_ed25519.pub"

    @property
    def vm_name(self) -> str:
        """Sanitized VM name (Azure limits: 1-64 chars, alnum + hyphen)."""
        slug = re.sub(r"[^A-Za-z0-9-]+", "-", self.label).strip("-")
        return slug[:50] or "pact-vm"

    @property
    def estimated_cost_per_hour(self) -> float:
        info = AZURE_GPU_PRICING.get(self.gpu_type)
        if not info:
            return 0.0
        return info["spot_usd_per_hour"] if self.spot else info["on_demand_usd_per_hour"]


@dataclasses.dataclass(frozen=True)
class AzureVMHandle:
    """Handle to a provisioned Azure VM (returned by provision_spot_vm)."""

    vm_name: str
    resource_group: str
    region: str
    public_ip: str
    username: str
    ssh_port: int = 22
    gpu_type: str = DEFAULT_GPU_TYPE
    spot: bool = True
    estimated_cost_per_hour: float = 0.0

    @property
    def ssh_target(self) -> str:
        return f"{self.username}@{self.public_ip}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _which_az() -> str:
    """Return the absolute path to the ``az`` CLI or raise."""
    proc = subprocess.run(
        ["which", "az"], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise AzureCLIMissing(
            "az CLI not found on PATH. Install with `brew install azure-cli` "
            "(macOS) or follow https://learn.microsoft.com/cli/azure/install-azure-cli."
        )
    return proc.stdout.strip()


def ensure_azure_logged_in() -> dict:
    """Verify that ``az`` has an active subscription. Returns the account dict.

    Raises ``AzureNotLoggedIn`` with a remediation message otherwise. This
    is the single chokepoint every Azure operation should call before any
    side-effecting ``az`` invocation.
    """
    _which_az()
    proc = subprocess.run(
        ["az", "account", "show", "--output", "json"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AzureNotLoggedIn(
            "az is installed but no active subscription. Run `az login` "
            "interactively, then re-dispatch. Output:\n"
            f"  stdout: {proc.stdout.strip()[:200]}\n"
            f"  stderr: {proc.stderr.strip()[:200]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AzureNotLoggedIn(
            f"az account show returned non-JSON output: {e}"
        ) from e


def _run_az(
    args: list[str],
    *,
    timeout: float = 300.0,
    check: bool = True,
    capture: bool = True,
) -> tuple[int, str, str]:
    """Run ``az ARGS`` and return (rc, stdout, stderr).

    Honors the platform-agnostic timeout and surfaces clear errors when
    ``az`` fails. Never silently swallows non-zero exit unless
    ``check=False``.
    """
    cmd = ["az", *args]
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AzureCommandError(
            f"az {' '.join(args[:3])} ... failed (rc={proc.returncode}):\n"
            f"  stdout: {proc.stdout.strip()[:400]}\n"
            f"  stderr: {proc.stderr.strip()[:400]}"
        )
    return proc.returncode, proc.stdout, proc.stderr


def _load_active_vms() -> list[dict]:
    """Read .omx/state/azure_active_vms.json (returns [] if missing/invalid).

    Lenient read for non-mutating callers; for mutating callers use
    ``tac.deploy.azure.active_vms_state.update_active_vms_locked`` (which
    holds an fcntl lock and uses the strict reload path inside it).

    Codex round 5 HIGH 3 fix (catalog #141): threads
    ``path=ACTIVE_VMS_PATH`` so tests that monkeypatch
    ``azure_dispatch.ACTIVE_VMS_PATH`` see the override. Pre-fix, the
    helper's module-level path was always used and tracker-isolation
    tests broke silently.
    """
    from tac.deploy.azure.active_vms_state import load_active_vms

    return load_active_vms(path=ACTIVE_VMS_PATH)


def register_active_vm(handle: AzureVMHandle, label: str) -> None:
    """Register a newly provisioned VM in .omx/state/azure_active_vms.json.

    This is REQUIRED for every successful provision_spot_vm call so the
    cleanup scripts can detect orphan VMs (analog to the Vast.ai
    .omx/state/vastai_active_instances.json contract).

    Codex round 4 HIGH 1 fix (2026-05-09, catalog #133): writes are now
    serialized through ``tac.deploy.azure.active_vms_state.register_active_vm_record``
    which acquires an fcntl exclusive lock + reloads strict + writes atomically.
    Previously the load+append+write_text path raced with concurrent provisions
    and silently dropped VM rows. The previous module's listing in catalog
    #131's exempt list as "already locked" was a false-green; the fix
    landed the actual locked writer.

    Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
    """
    from tac.deploy.azure.active_vms_state import register_active_vm_record

    record = {
        "vm_name": handle.vm_name,
        "resource_group": handle.resource_group,
        "region": handle.region,
        "public_ip": handle.public_ip,
        "username": handle.username,
        "ssh_port": handle.ssh_port,
        "gpu_type": handle.gpu_type,
        "spot": handle.spot,
        "estimated_cost_per_hour": handle.estimated_cost_per_hour,
        "label": label,
        "provisioned_at": int(time.time()),
    }
    # Codex round 5 HIGH 3 fix (catalog #141): thread `path=` and
    # `lock_path=` so monkeypatched ``ACTIVE_VMS_PATH`` overrides reach
    # the helper. Pre-fix the helper's module-level path won; test isolation
    # silently failed.
    register_active_vm_record(record, path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK)


def unregister_active_vm(vm_name: str) -> None:
    """Remove a deprovisioned VM from the active tracker.

    Codex round 4 HIGH 1 fix (2026-05-09, catalog #133): now routed through
    ``tac.deploy.azure.active_vms_state.unregister_active_vm_by_name`` which
    holds the fcntl lock for the entire load+remove+write transaction.
    """
    from tac.deploy.azure.active_vms_state import unregister_active_vm_by_name

    # Codex round 5 HIGH 3 fix (catalog #141): thread `path=` and
    # `lock_path=` so monkeypatched ``ACTIVE_VMS_PATH`` overrides reach
    # the helper. Pre-fix tests' tracker monkeypatch silently observed
    # the canonical file instead of the temp path.
    unregister_active_vm_by_name(vm_name, path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK)


# ── Provision / SSH / Run / Harvest / Deprovision ────────────────────────────


def ensure_resource_group(name: str, region: str) -> None:
    """Idempotently create a resource group if missing."""
    ensure_azure_logged_in()
    rc, _, _ = _run_az(
        ["group", "show", "--name", name, "--output", "json"],
        check=False,
    )
    if rc == 0:
        return
    _run_az(
        ["group", "create",
         "--name", name,
         "--location", region,
         "--output", "json"],
        timeout=120,
    )


def provision_spot_vm(spec: AzureVMSpec, dry_run: bool = False) -> AzureVMHandle:
    """Provision a spot VM matching ``spec``. Returns an AzureVMHandle.

    On success, also writes the VM into the active tracker so cleanup
    scripts can detect orphans. This function performs an actual
    side-effecting ``az vm create`` — wrapper scripts should pass
    ``dry_run=True`` for plan-only invocations.

    Args:
        spec: AzureVMSpec with desired GPU, region, label.
        dry_run: If True, print the equivalent CLI command and return a
            placeholder handle without creating the VM. Skips the
            ``az login`` check too, so dry-run works on machines that
            haven't authenticated yet (the auth check is the operator's
            responsibility before any non-dry-run dispatch).
    """
    if not dry_run:
        ensure_azure_logged_in()
    if spec.gpu_type not in AZURE_GPU_PRICING:
        raise ValueError(
            f"Unknown GPU type {spec.gpu_type!r}. Known: "
            f"{', '.join(AZURE_GPU_PRICING)}"
        )
    cost_per_hour = spec.estimated_cost_per_hour
    if cost_per_hour <= 0:
        raise ValueError(f"Refusing to dispatch with zero cost estimate for {spec.gpu_type}")

    if not dry_run:
        ensure_resource_group(spec.resource_group, spec.region)

    az_args = [
        "vm", "create",
        "--resource-group", spec.resource_group,
        "--name", spec.vm_name,
        "--location", spec.region,
        "--size", spec.gpu_type,
        "--image", spec.image,
        "--admin-username", spec.username,
        "--ssh-key-values", spec.ssh_pubkey_path,
        "--public-ip-sku", "Standard",
        "--output", "json",
    ]
    if spec.spot:
        az_args += [
            "--priority", "Spot",
            "--max-price", f"{spec.spot_max_price_usd:.4f}",
            "--eviction-policy", "Deallocate",
        ]

    if dry_run:
        print(f"[dry_run] az {' '.join(shlex.quote(a) for a in az_args)}")
        return AzureVMHandle(
            vm_name=spec.vm_name,
            resource_group=spec.resource_group,
            region=spec.region,
            public_ip="0.0.0.0",
            username=spec.username,
            gpu_type=spec.gpu_type,
            spot=spec.spot,
            estimated_cost_per_hour=cost_per_hour,
        )

    _, stdout, _ = _run_az(az_args, timeout=600)
    info = json.loads(stdout)
    public_ip = info.get("publicIpAddress")
    if not public_ip:
        raise AzureCommandError(
            f"vm create succeeded but no publicIpAddress in response: {info!r}"
        )
    handle = AzureVMHandle(
        vm_name=spec.vm_name,
        resource_group=spec.resource_group,
        region=spec.region,
        public_ip=public_ip,
        username=spec.username,
        gpu_type=spec.gpu_type,
        spot=spec.spot,
        estimated_cost_per_hour=cost_per_hour,
    )
    register_active_vm(handle, spec.label)
    return handle


def ssh_in(handle: AzureVMHandle, command: str, *, timeout: float = 120.0) -> tuple[int, str, str]:
    """Run a single SSH command on the VM and return (rc, stdout, stderr).

    Caller is responsible for any quoting / escaping of ``command``. This
    helper prepends the standard SSH options that match our Vast.ai
    pattern (StrictHostKeyChecking=no for first-connect convenience).
    """
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-p", str(handle.ssh_port),
        handle.ssh_target,
        command,
    ]
    proc = subprocess.run(
        ssh_args, capture_output=True, text=True, timeout=timeout, check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_lane(
    handle: AzureVMHandle,
    lane_script_path: str,
    *,
    env_vars: Optional[dict] = None,
    remote_workdir: str = "/home/azureuser/pact",
    log_path: str = "/tmp/lane_run.log",
) -> tuple[int, str]:
    """Detach-launch a lane script on the VM (Pattern A nohup).

    Mirrors CLAUDE.md "Pattern A — Detached BG bash". This emits the
    nohup + bash -c + disown wrapper and returns the remote PID after
    backgrounding so the caller can poll the log file.

    Args:
        handle: provisioned VM
        lane_script_path: path on VM to the lane script (already SCP'd)
        env_vars: optional dict of env vars exported before invoking
            the script
        remote_workdir: cwd for the script
        log_path: where to tee combined stdout+stderr on the VM
    Returns:
        (rc, remote_log_path) — rc is 0 if the launch succeeded; the
        actual lane script runs DETACHED on the VM so caller polls
        ``log_path`` separately to monitor.
    """
    env_block = ""
    if env_vars:
        # Build "export KEY=VAL\n" pairs; values must be safely quoted.
        env_block = "".join(
            f"export {k}={shlex.quote(str(v))}\n" for k, v in env_vars.items()
        )
    # Pattern A: nohup + bash -c + disown — survives the parent SSH session
    # closing without harming the lane.
    inner = (
        f"cd {shlex.quote(remote_workdir)} && "
        f"{env_block}"
        f"bash {shlex.quote(lane_script_path)}"
    )
    detach_wrapper = (
        f"mkdir -p $(dirname {shlex.quote(log_path)}); "
        f"nohup bash -c {shlex.quote(inner)} "
        f"< /dev/null > {shlex.quote(log_path)} 2>&1 & disown; "
        f"echo PID=$!"
    )
    rc, stdout, stderr = ssh_in(handle, detach_wrapper, timeout=60)
    if rc != 0:
        return rc, f"ssh launch failed: rc={rc} stderr={stderr.strip()[:200]}"
    return 0, log_path


def harvest(handle: AzureVMHandle, remote_results_dir: str, local_dest: str) -> int:
    """Rsync the remote results directory back to ``local_dest``.

    Returns rsync's exit code; caller decides whether to deprovision.
    """
    Path(local_dest).mkdir(parents=True, exist_ok=True)
    src = f"{handle.ssh_target}:{remote_results_dir}/"
    rsync_args = [
        "rsync", "-az", "--partial",
        "-e",
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {handle.ssh_port}",
        src, local_dest,
    ]
    proc = subprocess.run(
        rsync_args, capture_output=True, text=True, timeout=1800, check=False,
    )
    return proc.returncode


def deprovision(handle: AzureVMHandle, *, force_delete_rg: bool = False) -> int:
    """Delete the VM (and its disk) and remove from active tracker.

    By default ``vm delete --yes --no-wait`` is used so the call returns
    quickly. The disk + NIC + public IP are NOT deleted automatically by
    that command on Azure — operators should periodically prune the
    resource group with ``az group delete --name <rg> --yes``. If
    ``force_delete_rg=True``, the WHOLE resource group is deleted (use
    only when the RG is dedicated to this single VM).
    """
    ensure_azure_logged_in()
    if force_delete_rg:
        rc, _, _ = _run_az(
            ["group", "delete",
             "--name", handle.resource_group,
             "--yes", "--no-wait", "--output", "json"],
            check=False,
            timeout=120,
        )
    else:
        rc, _, _ = _run_az(
            ["vm", "delete",
             "--resource-group", handle.resource_group,
             "--name", handle.vm_name,
             "--yes", "--no-wait", "--output", "json"],
            check=False,
            timeout=120,
        )
    unregister_active_vm(handle.vm_name)
    return rc


# ── Cost estimation helpers ──────────────────────────────────────────────────


def estimate_cost(gpu_type: str, hours: float, *, spot: bool = True) -> float:
    """Return the estimated USD cost for ``gpu_type`` for ``hours`` hours."""
    info = AZURE_GPU_PRICING.get(gpu_type)
    if not info:
        raise ValueError(f"Unknown gpu_type: {gpu_type!r}")
    rate = info["spot_usd_per_hour"] if spot else info["on_demand_usd_per_hour"]
    return rate * hours


def remaining_budget_usd(spent_so_far_usd: float = 0.0) -> float:
    """Return remaining USD against the $200 free-credit cap."""
    return max(0.0, AZURE_HARD_CAP_USD - spent_so_far_usd)


__all__ = [
    "AZURE_GPU_PRICING",
    "AZURE_HARD_CAP_USD",
    "DEFAULT_GPU_TYPE",
    "DEFAULT_REGION",
    "DEFAULT_SPOT_MAX_PRICE_USD",
    "AzureCLIMissing",
    "AzureCommandError",
    "AzureNotLoggedIn",
    "AzureVMHandle",
    "AzureVMSpec",
    "deprovision",
    "ensure_azure_logged_in",
    "ensure_resource_group",
    "estimate_cost",
    "harvest",
    "provision_spot_vm",
    "register_active_vm",
    "remaining_budget_usd",
    "run_lane",
    "ssh_in",
    "unregister_active_vm",
]
