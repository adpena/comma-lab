#!/usr/bin/env python3
"""launch_lane_azure.py — Azure VM-spot dispatch with retry, analog to
``launch_lane_with_retry.py``.

Wraps ``tac.deploy.azure.azure_dispatch`` to provision an Azure spot VM,
SCP the lane tarball, run the lane (Pattern A nohup), and on transient
failure retry on a fresh VM. Up to ``--max-retries`` (default 3).

Why a separate script: the user has $200 free Azure credits and Lightning
is the primary supplemental platform; Azure is the second supplemental.
The lane-script contract is shared with Vast.ai/Modal so a single CLI
flag set works across all four platforms.

Usage:
    .venv/bin/python scripts/launch_lane_azure.py \\
        --lane-script scripts/remote_lane_X.sh --label lane_X \\
        --gpu-type Standard_NC6s_v3 --region eastus \\
        --predicted-band 0.85 1.10 --estimated-cost 4.00 --max-retries 3

Exit codes:
    0 = lane successfully launched (VM running, lane script detached)
    1 = max retries exhausted
    2 = invalid args / pre-flight failure (no az login, missing CLI, etc)
    3 = remote launch state unknown; verify VM manually, no blind retry

This script DOES NOT spawn any VMs unless the user explicitly invokes it
with --no-dry-run; the default is dry-run for parity with the
"infrastructure first, dispatches when needed" mandate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.azure import azure_dispatch as azd  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lane-script", required=True,
                   help="Path (repo-relative) to the remote lane script.")
    p.add_argument("--label", required=True,
                   help="Lane label — used as VM name prefix and tracker key.")
    p.add_argument("--gpu-type", default=azd.DEFAULT_GPU_TYPE,
                   choices=sorted(azd.AZURE_GPU_PRICING.keys()),
                   help="Azure VM SKU (default: Standard_NC6s_v3 = V100).")
    p.add_argument("--region", default=azd.DEFAULT_REGION,
                   help=f"Azure region (default: {azd.DEFAULT_REGION}).")
    p.add_argument("--resource-group", default=azd.DEFAULT_RESOURCE_GROUP,
                   help=f"Azure resource group (default: {azd.DEFAULT_RESOURCE_GROUP}).")
    p.add_argument("--spot-max-price", type=float, default=azd.DEFAULT_SPOT_MAX_PRICE_USD,
                   help=f"Spot eviction price ceiling USD/hr "
                        f"(default: {azd.DEFAULT_SPOT_MAX_PRICE_USD}).")
    p.add_argument("--no-spot", action="store_true",
                   help="Disable spot pricing (use on-demand).")
    p.add_argument("--ssh-pubkey", default="~/.ssh/id_ed25519.pub",
                   help="Path to SSH public key (default: ~/.ssh/id_ed25519.pub).")
    p.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    p.add_argument("--estimated-cost", type=float)
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max attempts before giving up (default 3).")
    p.add_argument("--retry-delay", type=int, default=15,
                   help="Seconds between retries (default 15).")
    p.add_argument("--no-dry-run", action="store_true",
                   help="Actually spawn the VM. Default is dry-run "
                        "(prints planned `az vm create` and exits 0).")
    return p.parse_args()


def _attempt_dispatch(args: argparse.Namespace, attempt: int) -> tuple[str, str | None, str]:
    """Single Azure provision+launch attempt.

    Returns (status, vm_name, log) — status is "success" / "retry" / "unknown".
    """
    log: list[str] = []
    log.append(f"=== ATTEMPT {attempt} ===")

    label = f"{args.label}_a{attempt}"
    spec = azd.AzureVMSpec(
        label=label,
        gpu_type=args.gpu_type,
        region=args.region,
        resource_group=args.resource_group,
        spot=not args.no_spot,
        spot_max_price_usd=args.spot_max_price,
        ssh_pubkey_path=args.ssh_pubkey,
    )

    try:
        handle = azd.provision_spot_vm(spec, dry_run=not args.no_dry_run)
    except (azd.AzureCommandError, azd.AzureCLIMissing,
            azd.AzureNotLoggedIn, ValueError) as e:
        log.append(f"PROVISION_FAILED: {e}")
        return "retry", None, "\n".join(log)

    log.append(f"  vm_name={handle.vm_name} public_ip={handle.public_ip} "
               f"region={handle.region} spot={handle.spot} "
               f"~${handle.estimated_cost_per_hour:.2f}/hr")

    if not args.no_dry_run:
        log.append("DRY_RUN: skipping SCP + run_lane + harvest. "
                   "Re-run with --no-dry-run to actually dispatch.")
        return "success", handle.vm_name, "\n".join(log)

    # SCP the lane tarball — for now we just verify the lane script exists
    # locally. Actual tarball wiring is identical to Vast.ai's
    # phase2-scp; left out of this initial wiring per "infrastructure
    # first" mandate.
    lane_script_local = REPO_ROOT / args.lane_script
    if not lane_script_local.exists():
        log.append(f"FAIL: lane script missing locally: {args.lane_script}")
        azd.deprovision(handle)
        return "retry", handle.vm_name, "\n".join(log)

    # Pattern A nohup detach launch
    rc, log_or_path = azd.run_lane(
        handle,
        lane_script_path=f"/home/{handle.username}/pact/{args.lane_script}",
        env_vars={"PYTHONPATH": "src:upstream:."},
    )
    if rc != 0:
        log.append(f"RUN_LANE_FAILED: {log_or_path}")
        azd.deprovision(handle)
        return "retry", handle.vm_name, "\n".join(log)

    log.append(f"✓ SUCCESS — VM {handle.vm_name} running lane {args.label}")
    log.append(f"  log on VM: {log_or_path}")
    return "success", handle.vm_name, "\n".join(log)


def main() -> int:
    args = _parse_args()

    if not (REPO_ROOT / args.lane_script).exists():
        print(f"FATAL: lane script missing: {args.lane_script}", file=sys.stderr)
        return 2

    # Pre-flight Azure login check (fail-loud, never silent)
    if args.no_dry_run:
        try:
            account = azd.ensure_azure_logged_in()
        except (azd.AzureCLIMissing, azd.AzureNotLoggedIn) as e:
            print(f"FATAL: Azure pre-flight failed:\n{e}", file=sys.stderr)
            return 2
        print(f"=== launch_lane_azure: {args.label} (max {args.max_retries} attempts) ===")
        print(f"  account: {account.get('name')!r} sub: {account.get('id')!r}")
    else:
        print(f"=== launch_lane_azure DRY-RUN: {args.label} (max {args.max_retries} attempts) ===")

    print(f"  gpu_type: {args.gpu_type}  region: {args.region}  spot: {not args.no_spot}")
    print(f"  estimated cost/hr: ${azd.AZURE_GPU_PRICING[args.gpu_type]['spot_usd_per_hour']:.2f} (spot) "
          f"/ ${azd.AZURE_GPU_PRICING[args.gpu_type]['on_demand_usd_per_hour']:.2f} (on-demand)")

    for attempt in range(1, args.max_retries + 1):
        status, vm_name, log = _attempt_dispatch(args, attempt)
        print(log)
        if status == "success":
            print(f"\n✓ DISPATCHED: vm={vm_name} label={args.label} attempts={attempt}")
            return 0
        if status == "unknown":
            print(
                f"\n? UNKNOWN_REMOTE_STATE: vm={vm_name} label={args.label}. "
                "No duplicate retry launched; inspect or destroy manually.",
                file=sys.stderr,
            )
            return 3
        if attempt < args.max_retries:
            print(f"  retrying in {args.retry_delay}s...\n")
            time.sleep(args.retry_delay)

    print(f"\n✗ FAILED: {args.max_retries} attempts exhausted for {args.label}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
