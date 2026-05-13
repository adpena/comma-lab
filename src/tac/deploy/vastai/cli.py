"""CLI entry point for Vast.ai deployment.

Dispatches subcommands to :class:`VastClient` and :class:`BudgetTracker`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from tac.deploy.base import InstanceSpec
from tac.deploy.vastai.budget import BudgetTracker
from tac.deploy.vastai.client import VastClient
from tac.deploy.vastai.experiments import EXPERIMENTS

# ANSI colors
_RED = "\033[91m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def cmd_search(client: VastClient, args: argparse.Namespace) -> int:
    """Search for best RTX 4090 offers."""
    print(f"\n{_BOLD}Searching for RTX 4090 instances...{_RESET}\n")
    spec = InstanceSpec()

    try:
        output = client.search_offers(spec, limit=15, raw=False)
    except RuntimeError as exc:
        print(f"{_RED}Search failed: {exc}{_RESET}")
        return 1

    print(output)

    budget = client.budget
    print(
        f"\n{_BOLD}Budget:{_RESET} ${budget.total_spent:.2f} spent, "
        f"${budget.remaining:.2f} remaining of ${budget.hard_cap:.2f} cap"
    )
    return 0


def cmd_launch(client: VastClient, args: argparse.Namespace) -> int:
    """Launch an experiment on a Vast.ai RTX 4090 instance."""
    experiment_name: str = args.experiment
    if experiment_name not in EXPERIMENTS:
        print(f"{_RED}Unknown experiment: {experiment_name}{_RESET}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        return 1

    experiment = EXPERIMENTS[experiment_name]
    spec = InstanceSpec()

    print(f"\n{_BOLD}Launching experiment: {experiment_name}{_RESET}")
    print(f"  Timeout: {experiment.timeout_hours}h")
    print(f"  Script: {experiment.script}")
    print()

    try:
        with client.create_instance(spec, experiment) as inst:
            # Upload code
            print(f"\n{_BOLD}Uploading code...{_RESET}")
            client.upload_code(inst)

            # Upload checkpoint if needed
            if experiment.needs_checkpoint:
                print(f"  Uploading checkpoint: {experiment.needs_checkpoint}")
                client.upload_checkpoint(inst, experiment.needs_checkpoint)

            print(f"  {_GREEN}Upload complete{_RESET}")

            # Launch experiment (detached)
            print(f"\n{_BOLD}Starting experiment: {experiment_name}{_RESET}")
            client.run_experiment(inst, experiment)

            print(f"  {_GREEN}Experiment launched in background{_RESET}")
            print(f"  Instance ID: {inst.instance_id}")
            print(f"  SSH: ssh -p {inst.ssh_port} root@{inst.ssh_host}")
            print("  Monitor: python scripts/vastai_deploy.py status")
            print("  Results: python scripts/vastai_deploy.py results")
            print(f"  Timeout: {experiment.timeout_hours}h")

            # Record estimated cost
            estimated_cost = inst.dph * experiment.timeout_hours
            client.budget.record_spend(
                inst.instance_id, 0.0,
                f"launched {experiment_name} at ${inst.dph:.3f}/hr (estimated ${estimated_cost:.2f})",
            )

            # NOTE: context manager will destroy the instance on exit.
            # For long-running experiments, the user should use the
            # non-context-manager flow (launch -> status -> results -> destroy).
            # This CLI uses the simple flow for correctness.

    except RuntimeError as exc:
        print(f"{_RED}Launch failed: {exc}{_RESET}")
        return 1
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    return 0


def cmd_status(client: VastClient, args: argparse.Namespace) -> int:
    """Check status of all tracked instances."""
    instances = client.load_instances()
    budget = client.budget

    print(f"\n{_BOLD}Vast.ai Instance Status{_RESET}")
    print("=" * 70)
    print(
        f"  Budget: ${budget.total_spent:.2f} / ${budget.hard_cap:.2f} "
        f"(${budget.remaining:.2f} remaining)"
    )
    print()

    if not instances:
        print(f"  {_DIM}No tracked instances{_RESET}")
        return 0

    now = datetime.now(timezone.utc)
    instances_to_destroy: list[str] = []

    for iid, inst in instances.items():
        # Defensive: created_at may be missing/None for instances persisted by
        # an older client version. Fall back to "now" so the loop continues
        # rather than crashing the entire status command. The elapsed=0
        # produces a $0 estimated cost for that instance — the operator gets
        # a visible warning instead of a silent crash mid-fleet.
        try:
            created = datetime.fromisoformat(inst.created_at)
        except (TypeError, ValueError):
            print(
                f"  {_YELLOW}[warn] {iid}: missing/invalid created_at "
                f"({inst.created_at!r}); using now() — cost estimate will read 0{_RESET}"
            )
            created = now
        elapsed_hours = (now - created).total_seconds() / 3600
        estimated_cost = inst.dph * elapsed_hours

        # Query remote status
        try:
            remote_info = client.show_instance(iid)
            vastai_status = remote_info.get("actual_status", remote_info.get("status_msg", "unknown"))
        except RuntimeError:
            vastai_status = "destroyed/not-found"

        # Check experiment completion
        experiment_done = False
        if vastai_status == "running":
            experiment_done = client.check_experiment_done(inst)
            if experiment_done:
                inst.status = "completed"

        # Format status
        if vastai_status == "running" and not experiment_done:
            status_str = f"{_GREEN}{_BOLD}RUNNING{_RESET}"
        elif experiment_done:
            status_str = f"{_BLUE}{_BOLD}COMPLETED{_RESET}"
        elif vastai_status in ("exited", "error"):
            status_str = f"{_RED}{_BOLD}{vastai_status.upper()}{_RESET}"
        elif "destroyed" in vastai_status or "not-found" in vastai_status:
            status_str = f"{_DIM}DESTROYED{_RESET}"
        else:
            status_str = f"{_YELLOW}{vastai_status}{_RESET}"

        over_timeout = elapsed_hours > inst.timeout_hours
        timeout_str = (
            f"{_RED}OVER LIMIT{_RESET}" if over_timeout
            else f"{elapsed_hours:.1f}/{inst.timeout_hours}h"
        )

        print(f"  {_BOLD}[{iid}]{_RESET} {inst.experiment}")
        print(f"    Status: {status_str} | Elapsed: {timeout_str}")
        print(f"    Cost: ${estimated_cost:.2f} (${inst.dph:.3f}/hr)")
        print(f"    SSH: root@{inst.ssh_host}:{inst.ssh_port}")

        if over_timeout and vastai_status == "running":
            print(f"    {_RED}{_BOLD}TIMEOUT EXCEEDED - marking for auto-destroy{_RESET}")
            instances_to_destroy.append(iid)

        if budget.total_spent + estimated_cost > budget.hard_cap and vastai_status == "running":
            print(f"    {_RED}{_BOLD}BUDGET OVERRUN - marking for auto-destroy{_RESET}")
            instances_to_destroy.append(iid)

        print()

    if instances_to_destroy:
        print(f"{_RED}{_BOLD}Auto-destroying {len(instances_to_destroy)} instance(s):{_RESET}")
        for iid in set(instances_to_destroy):
            inst = instances[iid]
            print(f"  Saving partial results from {iid}...")
            try:
                client.download_results(inst)
            except (RuntimeError, OSError):
                pass
            client.destroy(iid)

    return 0


def cmd_results(client: VastClient, args: argparse.Namespace) -> int:
    """Download results from completed (or running) instances."""
    instances = client.load_instances()

    print(f"\n{_BOLD}Downloading results...{_RESET}\n")

    if not instances:
        print(f"  {_DIM}No tracked instances{_RESET}")
        return 0

    downloaded = 0
    for iid, inst in instances.items():
        if inst.status in ("destroyed", "results_downloaded"):
            continue

        print(f"  {_BOLD}[{iid}]{_RESET} {inst.experiment}")
        try:
            local_dir = client.download_results(inst)
            downloaded += 1
            print(f"    {_GREEN}Results saved to {local_dir}{_RESET}")

            if inst.status == "completed":
                try:
                    created = datetime.fromisoformat(inst.created_at)
                except (TypeError, ValueError):
                    print(
                        f"    {_YELLOW}[warn] missing/invalid created_at "
                        f"({inst.created_at!r}); skipping cost record{_RESET}"
                    )
                    created = None
                if created is None:
                    if not args.keep:
                        print("    Destroying instance...")
                        client.destroy(iid)
                    continue
                elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
                final_cost = inst.dph * elapsed_hours
                client.budget.record_spend(
                    iid, final_cost,
                    f"final cost for {inst.experiment}: {elapsed_hours:.1f}h",
                )
                print(f"    Final cost: ${final_cost:.2f}")

                if not args.keep:
                    print("    Destroying instance...")
                    client.destroy(iid)

        except (RuntimeError, OSError) as exc:
            print(f"    {_RED}Download failed: {exc}{_RESET}")

    print(f"\n  Downloaded from {downloaded} instance(s)")
    return 0


def cmd_destroy(client: VastClient, args: argparse.Namespace) -> int:
    """Destroy instances."""
    instances = client.load_instances()

    if args.all:
        targets = list(instances.keys())
    elif args.instance_id:
        targets = [args.instance_id]
    else:
        targets = [
            iid for iid, inst in instances.items()
            if inst.status in ("completed", "error", "results_downloaded")
        ]

    if not targets:
        print(f"  {_DIM}No instances to destroy{_RESET}")
        return 0

    print(f"\n{_BOLD}Destroying {len(targets)} instance(s)...{_RESET}\n")

    for iid in targets:
        inst = instances.get(iid)
        experiment_name = inst.experiment if inst else "unknown"

        if inst and inst.dph > 0:
            try:
                created = datetime.fromisoformat(inst.created_at)
            except (TypeError, ValueError):
                # Skip cost record but still destroy — see status() handler.
                print(
                    f"  {_YELLOW}[warn] {iid}: missing/invalid created_at "
                    f"({inst.created_at!r}); destroying without cost record{_RESET}"
                )
                client.destroy(iid)
                print(f"  {_GREEN}Destroyed {iid} ({experiment_name}){_RESET}")
                continue
            elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            final_cost = inst.dph * elapsed_hours
            # Only record if not already recorded
            already_recorded = any(
                s.get("instance_id") == iid and s.get("amount", 0) > 0
                for s in client.budget.sessions
            )
            if not already_recorded:
                client.budget.record_spend(
                    iid, final_cost,
                    f"final cost for {experiment_name}: {elapsed_hours:.1f}h",
                )

        client.destroy(iid)
        print(f"  {_GREEN}Destroyed {iid} ({experiment_name}){_RESET}")

    return 0


def cmd_budget(client: VastClient, args: argparse.Namespace) -> int:
    """Show budget tracking."""
    client.budget.print_summary()
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vast.ai RTX 4090 deployment for pact-lab experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s search                          Find best 4090 offers\n"
            "  %(prog)s launch --experiment tto_v1      Launch TTO v1 experiment\n"
            "  %(prog)s status                          Check instance statuses\n"
            "  %(prog)s results                         Download results\n"
            "  %(prog)s destroy --all                   Destroy all instances\n"
            "  %(prog)s budget                          Show spend tracking\n"
            "\n"
            "Available experiments:\n"
            + "\n".join(
                f"  {name:20s} ({cfg.timeout_hours}h timeout)"
                for name, cfg in EXPERIMENTS.items()
            )
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("search", help="Find best RTX 4090 offers")

    launch_p = subparsers.add_parser("launch", help="Launch an experiment")
    launch_p.add_argument(
        "--experiment", "-e", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run",
    )

    subparsers.add_parser("status", help="Check all instance statuses")

    results_p = subparsers.add_parser("results", help="Download results")
    results_p.add_argument(
        "--keep", action="store_true",
        help="Keep instances after downloading (don't auto-destroy)",
    )

    destroy_p = subparsers.add_parser("destroy", help="Destroy instances")
    destroy_p.add_argument("--all", action="store_true", help="Destroy ALL tracked instances")
    destroy_p.add_argument("--instance-id", "-i", help="Destroy a specific instance")

    subparsers.add_parser("budget", help="Show spend tracking")

    args = parser.parse_args()

    # Initialize client
    budget = BudgetTracker()
    client = VastClient(budget=budget)
    client.check_api_key()

    dispatch = {
        "search": cmd_search,
        "launch": cmd_launch,
        "status": cmd_status,
        "results": cmd_results,
        "destroy": cmd_destroy,
        "budget": cmd_budget,
    }

    return dispatch[args.command](client, args)
