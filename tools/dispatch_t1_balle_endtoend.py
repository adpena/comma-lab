#!/usr/bin/env python3
"""Dispatcher for T1 — Ballé hyperprior + 128K decoder end-to-end.

This is the **actuator** for T1 per CLAUDE.md "Race-mode rigor inversion +
parallel-dispatch first" Rule 1: every ranker / sanity gate must name the
parallel actuator that consumes its output. T1's actuator is THIS file.

Pipeline
--------

1. Dry-run scaffold planning only. Real provider launch is refused until Phase
   2 lands scorer-domain training plus a contest-hermetic runtime/export
   contract.
2. Cost gate: $80 hard cap (operator-allocated for T1 per Phase 2 council
   memo §1 EIG/$ ranking).
3. Choose provider:
   - ``--provider modal`` → dry-run metadata only until Phase 2 closes the
     contest-hermetic runtime/export contract.
   - ``--provider vastai`` → dry-run metadata only. The dispatcher deliberately
     does not emit a provider CLI command until the current Vast.ai argparse
     surface is inspected in the same turn.
4. Phase-2 non-dry-run work will run
   ``scripts/remote_lane_t1_balle_endtoend.sh`` on the remote (which delegates
   bootstrap to ``scripts/remote_archive_only_eval.sh``).
5. Phase-2 harvest may queue GHA ``[contest-CPU]`` evaluation only after an
   archive has exact identity/custody; Phase 1 does not launch it.

CLAUDE.md compliance
--------------------

- NEVER inline ``uv run --with torch`` (per ``forbidden_remote_bootstrap_inline``)
  — the remote_lane script delegates to bootstrap_runtime_deps().
- NEVER bare BG bash for codex/long-running commands (per
  ``codex_invocation`` Pattern A) — this dispatcher is a foreground tool;
  the remote_lane script runs inside the provider's own session.
- NEVER MPS device for the dispatched run — refuses ``--device mps`` and
  forces ``--device cuda`` on the remote.
- NEVER leave phantom lane claims: this scaffold dispatcher refuses non-dry-run
  before writing any active claim because it does not create a provider job.
- NEVER claim a score without the contest_auth_eval.json provenance.

Exit codes
----------

0 = scaffold plan written; no provider job was created.
1 = pre-dispatch validation failed (cost cap, claim refused, etc.).
2 = non-dry-run refused; no GPU spend incurred and no lane claim written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LANE_ID = "t1_balle_128k_endtoend"
SCAFFOLD_VERSION_REL = "src/tac/paradigm_delta_epsilon_zeta/__init__.py"
DEFAULT_VASTAI_DISK_GB = 60
SCAFFOLD_PLAN_STATUS = "refused_dispatch_phase1_scaffold_only"
SCAFFOLD_PLAN_REASON = (
    "phase1_scaffold_only_no_provider_job_no_gpu_work_no_auth_eval; "
    "blocked_by_noop_scorer_loss_synthetic_smoke_and_non_hermetic_runtime"
)


HOURLY_RATES = {
    "modal_t4": 0.59,
    "modal_a10g": 1.10,
    "modal_a100": 4.00,
    "vastai_4090": 0.42,
    "vastai_a100": 1.40,
    "lightning_t4": 0.66,
    "lightning_a10g": 1.10,
}

DEFAULT_GPU_TIER_BY_PROVIDER = {
    "modal": "t4",
    "vastai": "4090",
}


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_compact_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class DispatchPlan:
    provider: str
    gpu_tier: str
    estimated_hours: float
    cost_cap_usd: float
    estimated_cost_usd: float
    output_dir: Path
    epochs: int
    vastai_disk_gb: int


def hourly_rate(provider: str, gpu_tier: str) -> float:
    key = f"{provider}_{gpu_tier.lower()}"
    try:
        return HOURLY_RATES[key]
    except KeyError as exc:
        valid = sorted(
            k.removeprefix(f"{provider}_")
            for k in HOURLY_RATES
            if k.startswith(f"{provider}_")
        )
        raise ValueError(
            f"unsupported provider/gpu-tier combination: {provider}/{gpu_tier}; "
            f"valid tiers for {provider}: {', '.join(valid) or 'none'}"
        ) from exc


def estimated_cost(provider: str, gpu_tier: str, hours: float) -> float:
    return float(hourly_rate(provider, gpu_tier)) * float(hours)


def claim_lane(
    *,
    provider: str,
    instance_or_job_id: str,
    agent: str,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id", LANE_ID,
        "--platform", provider,
        "--instance-job-id", instance_or_job_id,
        "--agent", agent,
        "--status", SCAFFOLD_PLAN_STATUS,
        "--notes", (
            "T1 Phase 1 scaffold plan only; no provider job created; "
            "no GPU work; no score claim"
        ),
    ]
    if dry_run:
        cmd.append("--dry-run")
    print(f"[t1-dispatch] {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, check=False).returncode


def write_dispatch_metadata(plan: DispatchPlan, *, extra: dict) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "lane_id": LANE_ID,
        "scaffold_version_relpath": SCAFFOLD_VERSION_REL,
        "provider": plan.provider,
        "gpu_tier": plan.gpu_tier,
        "estimated_hours": plan.estimated_hours,
        "cost_cap_usd": plan.cost_cap_usd,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "epochs": plan.epochs,
        "planned_at_utc": utc_stamp(),
        "dispatched_at_utc": None,
        "dispatch_status": SCAFFOLD_PLAN_STATUS,
        "dispatch_reason": SCAFFOLD_PLAN_REASON,
        "provider_job_created": False,
        "gpu_work_created": False,
        "auth_eval_created": False,
        "score_claim": False,
        "score_band": "[scaffold-plan only; Phase 1 not empirical]",
        **extra,
    }
    path = plan.output_dir / f"{plan.provider}_scaffold_plan.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"[t1-dispatch] wrote {path}")
    return path


def build_remote_command(plan: DispatchPlan) -> list[str]:
    """Return the bash command the remote runner will exec."""
    return [
        "bash",
        "-c",
        " && ".join([
            "cd /workspace/pact",
            "test \"$(git branch --show-current)\" = main",
            "git pull --ff-only origin main",
            "bash scripts/remote_lane_t1_balle_endtoend.sh",
        ]),
    ]


def dispatch_modal(plan: DispatchPlan, *, dry_run: bool) -> int:
    """Modal scaffold-plan writer; never spawns a provider job in Phase 1.

    Phase 1 NOTE: this is the build-only scaffold. The Modal app definition
    lives elsewhere (existing infra under ``experiments/modal_*.py``). This
    dispatcher deliberately writes a *_scaffold_plan.json file, not
    modal_metadata.json, so harvesters do not treat it as real GPU work.
    """
    metadata_extra = {
        "modal_app_name": "comma-t1-balle-endtoend",
        "modal_spawn_status": SCAFFOLD_PLAN_STATUS,
        "modal_call_id": None,
        "remote_command_template_only": shlex.join(build_remote_command(plan)),
        "harvester_path": None,
        "harvester_invocation": None,
    }
    if not dry_run:
        print("[t1-dispatch] FATAL: non-dry-run Modal dispatch should have been refused")
        return 2
    metadata_extra["call_id"] = None
    write_dispatch_metadata(plan, extra=metadata_extra)
    return 0


def dispatch_vastai(plan: DispatchPlan, *, dry_run: bool) -> int:
    """Vast.ai dry-run metadata writer.

    Phase 1 NOTE: actual ``vastai create instance --disk 60`` requires the operator's
    API key + active credit. Per AGENTS.md "Never invent CLI flags", this
    scaffold does not print a provider command unless that command has been
    inspected and wired in a future non-dry-run implementation.
    """
    label = f"t1_balle_endtoend_{utc_compact_stamp()}"
    metadata_extra = {
        "vastai_label": label,
        "vastai_required_disk_gb": int(plan.vastai_disk_gb),
        "vastai_required_create_flags": ["--disk", str(int(plan.vastai_disk_gb)), "--label", label],
        "vastai_create_command": None,
        "vastai_create_command_status": (
            "refused_phase1_scaffold_only; inspect current Vast.ai CLI help "
            "only after scorer/runtime blockers close"
        ),
        "vastai_remote_command_template_only": shlex.join(build_remote_command(plan)),
        "active_instances_path": ".omx/state/vastai_active_instances.json",
    }
    if not dry_run:
        print("[t1-dispatch] FATAL: non-dry-run Vast.ai dispatch should have been refused")
        return 2
    metadata_extra["instance_id"] = None
    write_dispatch_metadata(plan, extra=metadata_extra)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", required=True, choices=["modal", "vastai"],
        help="Where to dispatch the run.",
    )
    parser.add_argument(
        "--gpu-tier",
        default=None,
        help=(
            "GPU tier for cost estimation. Defaults to t4 for Modal and 4090 "
            "for Vast.ai. Unsupported provider/tier combinations fail closed."
        ),
    )
    parser.add_argument(
        "--estimated-hours", type=float, default=24.0,
        help="Estimated wall-clock hours (used for cost gate).",
    )
    parser.add_argument(
        "--cost-cap-usd", type=float, default=80.0,
        help="Hard cap on dispatch cost. Default $80 per Phase 2 council §1.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000,
        help=(
            "Planned Phase-2 trainer epochs for metadata. Phase-1 dry-runs do "
            "not launch training."
        ),
    )
    parser.add_argument(
        "--vastai-disk-gb",
        type=int,
        default=DEFAULT_VASTAI_DISK_GB,
        help=(
            "Required Vast.ai disk allocation for any future non-dry-run actuator. "
            "Values below 60 fail closed because chain evals have exceeded the "
            "provider default disk size."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiments" / "results" / "t1_balle_endtoend_scaffold_plan",
        help="Where to write scaffold-plan metadata. No provider job is created.",
    )
    parser.add_argument(
        "--agent", default=os.environ.get("SUBAGENT_LABEL", "claude-opus-4.7-1m"),
        help="Lane-claim agent label.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print + write metadata but do NOT actually dispatch.",
    )
    parser.add_argument(
        "--skip-claim", action="store_true",
        help="Skip the pre-dispatch lane claim (use ONLY in tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_tier is None:
        args.gpu_tier = DEFAULT_GPU_TIER_BY_PROVIDER[args.provider]

    # Cost gate.
    try:
        est = estimated_cost(args.provider, args.gpu_tier, args.estimated_hours)
    except ValueError as exc:
        print(f"[t1-dispatch] FATAL: {exc}")
        return 1
    plan = DispatchPlan(
        provider=args.provider,
        gpu_tier=args.gpu_tier,
        estimated_hours=args.estimated_hours,
        cost_cap_usd=args.cost_cap_usd,
        estimated_cost_usd=est,
        output_dir=args.output_dir,
        epochs=args.epochs,
        vastai_disk_gb=args.vastai_disk_gb,
    )
    print(f"[t1-dispatch] estimated cost: ${est:.2f} (cap: ${args.cost_cap_usd:.2f})")
    if est > args.cost_cap_usd:
        print(f"[t1-dispatch] FATAL: estimated cost ${est:.2f} > cap ${args.cost_cap_usd:.2f}")
        return 1
    if args.provider == "vastai" and args.vastai_disk_gb < DEFAULT_VASTAI_DISK_GB:
        print(
            "[t1-dispatch] FATAL: Vast.ai dispatch requires "
            f"--vastai-disk-gb >= {DEFAULT_VASTAI_DISK_GB}; "
            "the provider default disk has failed chain evals."
        )
        return 1

    if not args.dry_run:
        print(
            "[t1-dispatch] FATAL: T1 Phase 1 is scaffold-only. This "
            "dispatcher writes dry-run scaffold plans only until Phase 2 "
            "closes scorer loss, hermetic inflate, and exact-eval custody. "
            "No lane claim or provider job was created."
        )
        return 2

    # Dry-run refused-dispatch preview only. Real dispatch must claim with the
    # actual provider job id after the scaffold graduates from Phase 1.
    if not args.skip_claim:
        instance_or_job_id = (
            f"{args.provider}_scaffold_plan_{utc_compact_stamp()}"
        )
        rc = claim_lane(
            provider=args.provider,
            instance_or_job_id=instance_or_job_id,
            agent=args.agent,
            dry_run=args.dry_run,
        )
        if rc != 0:
            print(f"[t1-dispatch] FATAL: lane claim refused (rc={rc})")
            return 1

    # Dispatch.
    if args.provider == "modal":
        return dispatch_modal(plan, dry_run=args.dry_run)
    if args.provider == "vastai":
        return dispatch_vastai(plan, dry_run=args.dry_run)
    print(f"[t1-dispatch] FATAL: unknown provider {args.provider!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
