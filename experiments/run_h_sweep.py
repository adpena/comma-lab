#!/usr/bin/env python
"""Karpathy channel-width sweep: find the optimal base_ch/mid_ch for AsymmetricPairGenerator.

Reads experiments/configs/h_sweep.json and plans one Modal training run per
config (tiny/small/medium/large/xlarge). Collects proxy/advisory scores and
produces a Pareto frontier plot of rate vs quality.

Uses the existing Modal asymmetric warp deploy infrastructure as the canonical
provider actuator. This legacy sweep surface is plan-only by default; paid Modal
execution requires ``--execute-modal``.

Usage:
    # Plan all 5 configs (default; no provider job)
    .venv/bin/python experiments/run_h_sweep.py

    # Plan a single config
    .venv/bin/python experiments/run_h_sweep.py --config medium

    # Collect results only (after all training is done)
    .venv/bin/python experiments/run_h_sweep.py --collect-only

    # Explicitly spend on Modal via the canonical deploy actuator
    .venv/bin/python experiments/run_h_sweep.py --config medium --execute-modal

    # Canonical Modal actuator used by this planner
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py \
        --tag h_sweep_medium --extra-args '--base-ch 36 --mid-ch 60'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    PROXY_DISPATCH_BLOCKERS,
    proxy_authority_fields,
)

SWEEP_CONFIG = REPO_ROOT / "experiments" / "configs" / "h_sweep.json"
DEPLOY_SCRIPT = REPO_ROOT / "src" / "tac" / "deploy" / "modal" / "modal_asymmetric_warp_deploy.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "h_sweep"
PLAN_STATUS = "planned_only_no_provider_job_created"
MODAL_EXECUTION_FLAG = "--execute-modal"
MODAL_PLATFORM = "modal"
DEFAULT_DISPATCH_CLAIMS_PATH = REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
MODAL_EXECUTE_PREFLIGHT = "h_sweep_modal_execute_preflight.json"


class ExecuteGateError(ValueError):
    """Raised when a paid/provider execution path is missing custody gates."""


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def false_authority_metadata(
    *,
    surface: str,
    evidence_grade: str,
    dispatch_attempted: bool = False,
    claim_gate_satisfied: bool = False,
) -> dict:
    """Canonical no-score/no-promotion metadata for this legacy sweep surface."""

    authority = proxy_authority_fields()
    authority["dispatch_attempted"] = dispatch_attempted
    blockers = list(PROXY_DISPATCH_BLOCKERS)
    if claim_gate_satisfied:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "requires_lane_dispatch_claim_before_gpu_or_remote_eval"
        ]
        blockers.append("dispatch_claim_verified_but_no_adjudicated_exact_cuda_result")

    return {
        **authority,
        "score_claim_valid": False,
        "proxy_only": True,
        "score_authority": "none",
        "evidence_grade": evidence_grade,
        "evidence_boundary": (
            "proxy_or_plan_only_not_score_evidence; exact CUDA promotion requires "
            "claimed remote eval and adjudicated contest_auth_eval artifact"
        ),
        "surface": surface,
        "dispatch_blockers": [
            *blockers,
            "legacy_h_sweep_surface_is_plan_only_by_default",
            "modal_execution_requires_explicit_execute_modal_flag",
            "local_mps_cpu_results_are_advisory_only",
        ],
    }


def _git_output(args: list[str]) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_custody_metadata() -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    diff_bytes = diff.stdout if diff.returncode == 0 else b""
    status = _git_output(["status", "--short"]) or ""
    return {
        "repo_git_head": _git_output(["rev-parse", "HEAD"]),
        "repo_git_branch": _git_output(["branch", "--show-current"]),
        "repo_dirty": bool(status.strip()),
        "repo_status_short": status.splitlines(),
        "repo_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _claim_cost_usd(notes: str) -> float | None:
    match = re.search(r"\bcost\s*=\s*\$?([0-9]+(?:\.[0-9]+)?)\b", notes)
    if not match:
        return None
    return float(match.group(1))


def _load_dispatch_claim_summary(claims_path: Path) -> dict:
    if not claims_path.is_file():
        raise ExecuteGateError(f"missing dispatch claims ledger: {claims_path}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "summary",
        "--claims-path",
        str(claims_path),
        "--format",
        "json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
        raise ExecuteGateError(f"dispatch claim summary failed rc={proc.returncode}: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExecuteGateError(f"dispatch claim summary returned invalid JSON: {exc}") from exc
    if not isinstance(payload.get("active"), list):
        raise ExecuteGateError("dispatch claim summary missing active claim list")
    return payload


def require_modal_execute_gate(
    args: argparse.Namespace,
    configs: list[dict],
) -> dict:
    """Fail closed before paid Modal work unless claim, cost, and custody gates pass."""

    if not args.execute_modal or args.dry_run:
        return {}

    blockers: list[str] = []
    if len(configs) != 1:
        blockers.append("--execute-modal requires exactly one --config so one claim maps to one provider run")
    if not args.lane_id:
        blockers.append("--execute-modal requires --lane-id")
    if not args.instance_job_id:
        blockers.append("--execute-modal requires --instance-job-id")
    if args.estimated_cost_usd is None:
        blockers.append("--execute-modal requires --estimated-cost-usd")
    elif args.estimated_cost_usd <= 0:
        blockers.append("--estimated-cost-usd must be positive")
    if args.cost_cap_usd is None:
        blockers.append("--execute-modal requires --cost-cap-usd")
    elif args.cost_cap_usd <= 0:
        blockers.append("--cost-cap-usd must be positive")
    if (
        args.estimated_cost_usd is not None
        and args.cost_cap_usd is not None
        and args.estimated_cost_usd > args.cost_cap_usd
    ):
        blockers.append(
            f"estimated cost ${args.estimated_cost_usd:.2f} exceeds cap ${args.cost_cap_usd:.2f}"
        )
    if blockers:
        raise ExecuteGateError("; ".join(blockers))

    summary = _load_dispatch_claim_summary(args.dispatch_claims_path)
    active = summary["active"]
    matches = [
        row
        for row in active
        if row.get("lane_id") == args.lane_id
        and row.get("instance_job_id") == args.instance_job_id
        and row.get("platform") == MODAL_PLATFORM
    ]
    if not matches:
        raise ExecuteGateError(
            "missing active Modal dispatch claim for "
            f"lane_id={args.lane_id!r} instance_job_id={args.instance_job_id!r}"
        )
    claim = matches[0]
    claim_cost = _claim_cost_usd(str(claim.get("notes") or ""))
    if claim_cost is None:
        raise ExecuteGateError("active dispatch claim notes must include cost=$<usd>")
    if abs(claim_cost - float(args.estimated_cost_usd)) > 0.01:
        raise ExecuteGateError(
            f"claim cost ${claim_cost:.2f} does not match --estimated-cost-usd "
            f"${args.estimated_cost_usd:.2f}"
        )

    return {
        "provider_launch_allowed": True,
        "dispatch_claim_verified": True,
        "dispatch_claim": claim,
        "dispatch_claims_path": str(args.dispatch_claims_path),
        "lane_id": args.lane_id,
        "instance_job_id": args.instance_job_id,
        "estimated_cost_usd": float(args.estimated_cost_usd),
        "cost_cap_usd": float(args.cost_cap_usd),
        "claim_cost_usd": claim_cost,
        "cost_gate_status": "passed",
        **_git_custody_metadata(),
    }


def write_modal_execute_preflight(
    configs: list[dict],
    sweep: dict,
    output_dir: Path,
    gate: dict,
) -> Path:
    """Persist custody/cost metadata before launching the provider command."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "sweep_name": sweep["sweep_name"],
        "written_at_utc": utc_stamp(),
        "provider": MODAL_PLATFORM,
        "platform": "Modal T4",
        "execution_required_flag": MODAL_EXECUTION_FLAG,
        "provider_launch_allowed": True,
        "provider_job_created": False,
        "gpu_work_created": False,
        "auth_eval_created": False,
        "configs": [build_modal_plan_entry(cfg, sweep) for cfg in configs],
        "execute_gate": gate,
        **false_authority_metadata(
            surface="h_sweep_modal_execute_preflight",
            evidence_grade="[Modal execute preflight; no score claim]",
            claim_gate_satisfied=True,
        ),
    }
    path = output_dir / MODAL_EXECUTE_PREFLIGHT
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  [custody] wrote {path}")
    return path


def provider_env(gate: dict) -> dict[str, str] | None:
    if not gate:
        return None
    env = os.environ.copy()
    env.update(
        {
            "PACT_DISPATCH_LANE_ID": str(gate["lane_id"]),
            "PACT_DISPATCH_INSTANCE_JOB_ID": str(gate["instance_job_id"]),
            "PACT_DISPATCH_CLAIMS_PATH": str(gate["dispatch_claims_path"]),
            "PACT_ESTIMATED_COST_USD": f"{float(gate['estimated_cost_usd']):.2f}",
            "PACT_COST_CAP_USD": f"{float(gate['cost_cap_usd']):.2f}",
        }
    )
    return env


def run_provider_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)


def load_sweep_config() -> dict:
    """Load and validate the sweep configuration."""
    with open(SWEEP_CONFIG) as f:
        config = json.load(f)

    required_keys = ["configs", "training", "fixed_params", "deployment"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in {SWEEP_CONFIG}")

    return config


def build_modal_command(
    cfg: dict,
    sweep: dict,
    tag: str,
) -> list[str]:
    """Build the modal run command for a single sweep config.

    The extra-args override base_ch and mid_ch in the deploy script's
    TRAINING_CMD_TEMPLATE. All other training params are inherited from
    the template (council-approved defaults).
    """
    training = sweep["training"]

    extra_args = [
        "--base-ch", str(cfg["base_ch"]),
        "--mid-ch", str(cfg["mid_ch"]),
        "--epochs", str(training["epochs"]),
    ]

    # Build the modal run command
    cmd = [
        sys.executable, "-m", "modal", "run",
        str(DEPLOY_SCRIPT),
        "--tag", tag,
        "--extra-args", " ".join(extra_args),
    ]

    return cmd


def build_modal_plan_entry(cfg: dict, sweep: dict) -> dict:
    tag = f"h_sweep_{cfg['name']}"
    cmd = build_modal_command(cfg, sweep, tag)
    return {
        "name": cfg["name"],
        "tag": tag,
        "base_ch": cfg["base_ch"],
        "mid_ch": cfg["mid_ch"],
        "total_params": cfg["total_params"],
        "fp4_kb": cfg["fp4_kb"],
        "rate_term": cfg["rate_term"],
        "modal_actuator": str(DEPLOY_SCRIPT.relative_to(REPO_ROOT)),
        "modal_command_template_only": shlex.join(cmd),
        "provider_job_created": False,
        "gpu_work_created": False,
        "auth_eval_created": False,
        "execution_required_flag": MODAL_EXECUTION_FLAG,
        **false_authority_metadata(
            surface="h_sweep_modal_plan",
            evidence_grade="[provider-plan only; no remote job]",
        ),
    }


def write_modal_plan(configs: list[dict], sweep: dict, output_dir: Path) -> Path:
    """Write plan-only Modal metadata without creating a provider job."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "sweep_name": sweep["sweep_name"],
        "planned_at_utc": utc_stamp(),
        "plan_status": PLAN_STATUS,
        "provider": "modal",
        "platform": "Modal T4",
        "provider_job_created": False,
        "gpu_work_created": False,
        "auth_eval_created": False,
        "modal_actuator": str(DEPLOY_SCRIPT.relative_to(REPO_ROOT)),
        "execution_required_flag": MODAL_EXECUTION_FLAG,
        "estimated_cost_usd": len(configs) * float(
            sweep["deployment"].get("est_cost_per_config_usd", 0.50)
        ),
        "configs": [build_modal_plan_entry(cfg, sweep) for cfg in configs],
        **false_authority_metadata(
            surface="h_sweep_modal_plan",
            evidence_grade="[provider-plan only; no remote job]",
        ),
    }
    path = output_dir / "h_sweep_modal_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  [plan] wrote {path}")
    print(
        "  [authority] score_claim=false proxy_only=true "
        "ready_for_exact_eval_dispatch=false promotion_eligible=false"
    )
    return path


def build_local_command(
    cfg: dict,
    sweep: dict,
    tag: str,
    device: str = "mps",
) -> list[str]:
    """Build a local training command (for smoke testing on M5 Max)."""
    training = sweep["training"]
    fixed = sweep["fixed_params"]

    cmd = [
        sys.executable, "experiments/train_renderer_fridrich.py",
        "--pair-mode", fixed["pair_mode"],
        "--base-ch", str(cfg["base_ch"]),
        "--mid-ch", str(cfg["mid_ch"]),
        "--embed-dim", str(fixed["embed_dim"]),
        "--motion-hidden", str(fixed["motion_hidden"]),
        "--max-flow-px", str(fixed["max_flow_px"]),
        "--max-residual", str(fixed["max_residual"]),
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--lr", str(training["lr"]),
        "--seg-boundary", str(training["seg_boundary"]),
        "--pose-boundary", str(training["pose_boundary"]),
        "--rho-init", str(training["rho_init"]),
        "--rho-growth", str(training["rho_growth"]),
        "--tv-weight", str(training["tv_weight"]),
        "--flow-weight", str(training["flow_weight"]),
        "--rate-weight", str(training["rate_weight"]),
        "--target-bytes", str(training["target_bytes"]),
        "--gate-reg-weight", str(training["gate_reg_weight"]),
        "--phase2-mse-weight", str(training["phase2_mse_weight"]),
        "--eval-every", str(training["eval_every"]),
        "--checkpoint-every", str(training["checkpoint_every"]),
        "--log-every", str(training["log_every"]),
        "--max-hours", str(training["max_hours"]),
        "--device", device,
        "--seed", str(training["seed"]),
    ]
    if training.get("even_pairs_only"):
        cmd.append("--even-pairs-only")

    return cmd


def collect_results(sweep: dict) -> list[dict]:
    """Collect results from completed sweep runs.

    Looks for auth_eval JSON files on the Modal volume (downloaded locally)
    or in experiments/results/h_sweep/<config_name>/.
    """
    results = []

    for cfg in sweep["configs"]:
        tag = f"h_sweep_{cfg['name']}"
        result_dir = RESULTS_DIR / cfg["name"]

        # Look for auth eval results
        result_entry = {
            "name": cfg["name"],
            "base_ch": cfg["base_ch"],
            "mid_ch": cfg["mid_ch"],
            "total_params": cfg["total_params"],
            "fp4_kb": cfg["fp4_kb"],
            "rate_term": cfg["rate_term"],
            "tag": tag,
            **false_authority_metadata(
                surface="h_sweep_collect_results",
                evidence_grade="[legacy sweep collected result; advisory until adjudicated exact CUDA]",
            ),
        }

        # Check local results directory
        if result_dir.exists():
            # Find auth eval JSON
            auth_files = sorted(result_dir.glob("auth_eval_*.json"))
            if auth_files:
                with open(auth_files[-1]) as f:
                    auth = json.load(f)
                result_entry["seg_score"] = auth.get("seg_score")
                result_entry["pose_score"] = auth.get("pose_score")
                result_entry["rate"] = auth.get("rate")
                result_entry["total_score"] = auth.get("total_score")
                result_entry["auth_eval_file"] = str(auth_files[-1])

            # Find training summary
            summary_files = sorted(result_dir.glob("training_summary*.json"))
            if summary_files:
                with open(summary_files[-1]) as f:
                    summary = json.load(f)
                result_entry["proxy_score"] = summary.get("best_score") or summary.get("proxy_score")
                result_entry["epochs_trained"] = summary.get("epochs_trained") or summary.get("epoch")
                result_entry["summary_file"] = str(summary_files[-1])

        results.append(result_entry)

    return results


def plot_pareto_frontier(results: list[dict], output_path: Path) -> None:
    """Generate Pareto frontier plot: rate vs quality.

    X-axis: rate term (FP4 model size / total pixels)
    Y-axis: quality = 100*seg + sqrt(10*pose)  (lower is better)

    Points on the Pareto frontier are connected with a line.
    Each point is labeled with its config name.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib not available, skipping plot")
        return

    # Filter to results that have auth scores
    scored = [r for r in results if r.get("total_score") is not None]
    if not scored:
        # Fall back to proxy scores
        scored = [r for r in results if r.get("proxy_score") is not None]
        if not scored:
            print("  [plot] No scored results found, skipping plot")
            return
        metric = "proxy_score"
        ylabel = "Proxy Score (lower is better)"
    else:
        metric = "total_score"
        ylabel = "Total Score (lower is better)"

    names = [r["name"] for r in scored]
    rates = [r["rate_term"] for r in scored]
    scores = [r[metric] for r in scored]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Scatter all points
    ax.scatter(rates, scores, s=100, zorder=5, color="steelblue", edgecolors="black")

    # Label each point
    for name, rate, score in zip(names, rates, scores):
        ax.annotate(
            name,
            (rate, score),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
            fontweight="bold",
        )

    # Compute and plot Pareto frontier
    # A point is Pareto-optimal if no other point has both lower rate AND lower score
    pareto_idx = []
    for i, (r, s) in enumerate(zip(rates, scores)):
        dominated = False
        for j, (r2, s2) in enumerate(zip(rates, scores)):
            if i != j and r2 <= r and s2 <= s and (r2 < r or s2 < s):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    if pareto_idx:
        pareto_points = sorted(pareto_idx, key=lambda i: rates[i])
        pareto_rates = [rates[i] for i in pareto_points]
        pareto_scores = [scores[i] for i in pareto_points]
        ax.plot(pareto_rates, pareto_scores, "r--", linewidth=2, alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("Rate Term (FP4 size / total pixels)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Karpathy Channel Width Sweep: Rate vs Quality", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add param count as secondary labels
    for r in scored:
        ax.annotate(
            f"{r['total_params'] / 1000:.0f}K",
            (r["rate_term"], r[metric]),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
            color="gray",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved Pareto frontier to {output_path}")


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("KARPATHY CHANNEL WIDTH SWEEP — RESULTS (PROXY/ADVISORY)")
    print("=" * 100)
    print("score_claim=false proxy_only=true ready_for_exact_eval_dispatch=false")

    header = f"{'Config':<8} {'base_ch':>7} {'mid_ch':>6} {'Params':>9} {'FP4 KB':>7} {'Rate':>7} {'Proxy':>7} {'Auth':>7} {'Status':<12}"
    print(header)
    print("-" * 100)

    for r in results:
        proxy = f"{r['proxy_score']:.3f}" if r.get("proxy_score") is not None else "  —"
        auth = f"{r['total_score']:.3f}" if r.get("total_score") is not None else "  —"

        if r.get("total_score") is not None:
            status = "auth_eval"
        elif r.get("proxy_score") is not None:
            status = "trained"
        else:
            status = "pending"

        print(
            f"{r['name']:<8} {r['base_ch']:>7} {r['mid_ch']:>6} "
            f"{r['total_params']:>9,} {r['fp4_kb']:>7.1f} {r['rate_term']:>7.4f} "
            f"{proxy:>7} {auth:>7} {status:<12}"
        )

    print("=" * 100)

    # Identify winner
    scored = [r for r in results if r.get("total_score") is not None]
    if scored:
        best = min(scored, key=lambda r: r["total_score"])
        print(f"\nBest config (auth/advisory): {best['name']} — score {best['total_score']:.3f}")
        print(f"  base_ch={best['base_ch']}, mid_ch={best['mid_ch']}, {best['total_params']:,} params")
        print("  Not promotion evidence from this legacy surface: score_claim=false")
    else:
        proxy_scored = [r for r in results if r.get("proxy_score") is not None]
        if proxy_scored:
            best = min(proxy_scored, key=lambda r: r["proxy_score"])
            print(f"\nBest config (proxy/advisory): {best['name']} — proxy {best['proxy_score']:.3f}")
            print(f"  base_ch={best['base_ch']}, mid_ch={best['mid_ch']}, {best['total_params']:,} params")
            print("  Not promotion evidence from this legacy surface: score_claim=false")
        else:
            print("\nNo results yet. Run the sweep first.")


def main():
    parser = argparse.ArgumentParser(
        description="Karpathy channel-width sweep for AsymmetricPairGenerator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run only this config (tiny/small/medium/large/xlarge). Default: all.",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Skip training, just collect and display results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compatibility alias for the default plan-only mode; no command execution.",
    )
    parser.add_argument(
        "--execute-modal", action="store_true",
        help=(
            "Actually launch Modal runs via the canonical deploy actuator. "
            "Default is plan-only and creates no provider job."
        ),
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of on Modal. Local/MPS results are advisory only.",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device for local runs (default: mps).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Override epochs to 50 for quick validation of the sweep harness.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="Directory for plan/run metadata. Defaults to experiments/results/h_sweep.",
    )
    parser.add_argument(
        "--lane-id",
        default=None,
        help="Required with --execute-modal: active dispatch-claim lane id.",
    )
    parser.add_argument(
        "--instance-job-id",
        default=None,
        help="Required with --execute-modal: active dispatch-claim instance/job id.",
    )
    parser.add_argument(
        "--estimated-cost-usd",
        type=float,
        default=None,
        help="Required with --execute-modal: estimated provider cost for this run.",
    )
    parser.add_argument(
        "--cost-cap-usd",
        type=float,
        default=None,
        help="Required with --execute-modal: hard provider cost cap for this run.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=DEFAULT_DISPATCH_CLAIMS_PATH,
        help="Dispatch claim ledger to verify before --execute-modal launches.",
    )
    args = parser.parse_args()
    if args.execute_modal and args.local:
        print("--execute-modal cannot be combined with --local")
        sys.exit(2)

    sweep = load_sweep_config()
    configs = sweep["configs"]

    if args.config:
        configs = [c for c in configs if c["name"] == args.config]
        if not configs:
            valid = [c["name"] for c in sweep["configs"]]
            print(f"Unknown config '{args.config}'. Valid: {valid}")
            sys.exit(1)

    if args.collect_only:
        results = collect_results(sweep)
        print_results_table(results)
        plot_path = RESULTS_DIR / "pareto_frontier.png"
        plot_pareto_frontier(results, plot_path)
        return

    try:
        execute_gate = require_modal_execute_gate(args, configs)
    except ExecuteGateError as exc:
        print(f"FATAL: Modal execute refused before provider launch: {exc}")
        sys.exit(2)

    # Smoke test: override epochs
    if args.smoke:
        sweep["training"]["epochs"] = 50
        sweep["training"]["eval_every"] = 10
        sweep["training"]["checkpoint_every"] = 25
        sweep["training"]["max_hours"] = 0.5

    print(f"=== Karpathy Channel Width Sweep ===")
    print(f"  Configs: {[c['name'] for c in configs]}")
    print(f"  Platform: {'local advisory' if args.local else 'Modal T4'}")
    print(f"  Epochs: {sweep['training']['epochs']}")
    print(f"  Est. cost: ${len(configs) * 0.50:.2f} (Modal T4)")
    print(
        "  Authority: score_claim=false proxy_only=true "
        "ready_for_exact_eval_dispatch=false"
    )
    if not args.local and not args.execute_modal:
        print(f"  Mode: plan-only default; pass {MODAL_EXECUTION_FLAG} to launch Modal")
    elif args.execute_modal and not args.dry_run:
        print(
            "  Mode: Modal execute allowed after claim/cost gate "
            f"lane_id={execute_gate['lane_id']} job={execute_gate['instance_job_id']} "
            f"cost=${execute_gate['estimated_cost_usd']:.2f}"
        )
    print()

    if not args.local and not args.execute_modal:
        write_modal_plan(configs, sweep, args.output_dir)
        return
    if args.execute_modal and not args.dry_run:
        write_modal_execute_preflight(configs, sweep, args.output_dir, execute_gate)

    run_results = []
    for i, cfg in enumerate(configs):
        tag = f"h_sweep_{cfg['name']}"
        print(f"\n--- [{i+1}/{len(configs)}] Config: {cfg['name']} ---")
        print(f"  base_ch={cfg['base_ch']}, mid_ch={cfg['mid_ch']}")
        print(f"  {cfg['total_params']:,} params, {cfg['fp4_kb']:.1f} KB FP4")
        print(f"  Tag: {tag}")

        if args.local:
            cmd = build_local_command(cfg, sweep, tag, device=args.device)
        else:
            cmd = build_modal_command(cfg, sweep, tag)

        if args.dry_run:
            print(f"  [plan-only] {shlex.join(cmd)}")
            continue

        print(f"  Command: {' '.join(cmd[:6])} ...")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        t0 = time.monotonic()
        result = run_provider_command(cmd, env=provider_env(execute_gate))
        elapsed = time.monotonic() - t0

        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed/60:.1f} min) — {status}")

        run_results.append({
            "name": cfg["name"],
            "tag": tag,
            "returncode": result.returncode,
            "elapsed_min": elapsed / 60,
            "platform": "local_advisory" if args.local else "modal",
            **false_authority_metadata(
                surface="h_sweep_local_advisory" if args.local else "h_sweep_modal_execute",
                evidence_grade=(
                    "[local/MPS advisory]" if args.local else "[Modal run; requires separate adjudication]"
                ),
                dispatch_attempted=True,
                claim_gate_satisfied=bool(execute_gate),
            ),
        })

    if not args.dry_run:
        # Save run metadata
        args.output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = args.output_dir / "sweep_runs.json"
        with open(meta_path, "w") as f:
            json.dump({
                "sweep_name": sweep["sweep_name"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **false_authority_metadata(
                    surface="h_sweep_runs",
                    evidence_grade=(
                        "[local/MPS advisory]" if args.local else "[Modal run; requires separate adjudication]"
                    ),
                    dispatch_attempted=bool(run_results),
                    claim_gate_satisfied=bool(execute_gate),
                ),
                "execute_gate": execute_gate if args.execute_modal and not args.dry_run else None,
                "runs": run_results,
            }, f, indent=2)
        print(f"\n  Run metadata saved to {meta_path}")

        # Collect and display results
        results = collect_results(sweep)
        print_results_table(results)
        plot_path = RESULTS_DIR / "pareto_frontier.png"
        plot_pareto_frontier(results, plot_path)


if __name__ == "__main__":
    main()
