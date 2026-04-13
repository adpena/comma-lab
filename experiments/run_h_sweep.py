#!/usr/bin/env python
"""Karpathy channel-width sweep: find the optimal base_ch/mid_ch for AsymmetricPairGenerator.

Reads experiments/configs/h_sweep.json and launches one Modal training run per
config (tiny/small/medium/large/xlarge). Collects proxy scores and produces a
Pareto frontier plot of rate vs quality.

Uses the existing Modal asymmetric warp deploy infrastructure. Each config gets
its own tag on the results volume so runs can be resumed independently.

Usage:
    # Run all 5 configs (sequential, ~27.5h total)
    .venv/bin/python experiments/run_h_sweep.py

    # Run a single config
    .venv/bin/python experiments/run_h_sweep.py --config medium

    # Collect results only (after all training is done)
    .venv/bin/python experiments/run_h_sweep.py --collect-only

    # Dry run (show commands without executing)
    .venv/bin/python experiments/run_h_sweep.py --dry-run

    # Run via Modal directly (bypasses this orchestrator)
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py \
        --tag h_sweep_medium --extra-args '--base-ch 36 --mid-ch 60'
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_CONFIG = REPO_ROOT / "experiments" / "configs" / "h_sweep.json"
DEPLOY_SCRIPT = REPO_ROOT / "src" / "tac" / "deploy" / "modal" / "modal_asymmetric_warp_deploy.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "h_sweep"


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
    print("KARPATHY CHANNEL WIDTH SWEEP — RESULTS")
    print("=" * 100)

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
        print(f"\nBest config (auth): {best['name']} — score {best['total_score']:.3f}")
        print(f"  base_ch={best['base_ch']}, mid_ch={best['mid_ch']}, {best['total_params']:,} params")
    else:
        proxy_scored = [r for r in results if r.get("proxy_score") is not None]
        if proxy_scored:
            best = min(proxy_scored, key=lambda r: r["proxy_score"])
            print(f"\nBest config (proxy): {best['name']} — proxy {best['proxy_score']:.3f}")
            print(f"  base_ch={best['base_ch']}, mid_ch={best['mid_ch']}, {best['total_params']:,} params")
        else:
            print("\nNo results yet. Run the sweep first.")


def main():
    import argparse

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
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of on Modal (for smoke testing).",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device for local runs (default: mps).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Override epochs to 50 for quick validation of the sweep harness.",
    )
    args = parser.parse_args()

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

    # Smoke test: override epochs
    if args.smoke:
        sweep["training"]["epochs"] = 50
        sweep["training"]["eval_every"] = 10
        sweep["training"]["checkpoint_every"] = 25
        sweep["training"]["max_hours"] = 0.5

    print(f"=== Karpathy Channel Width Sweep ===")
    print(f"  Configs: {[c['name'] for c in configs]}")
    print(f"  Platform: {'local' if args.local else 'Modal T4'}")
    print(f"  Epochs: {sweep['training']['epochs']}")
    print(f"  Est. cost: ${len(configs) * 0.50:.2f} (Modal T4)")
    print()

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
            print(f"  [dry-run] {' '.join(cmd)}")
            continue

        print(f"  Command: {' '.join(cmd[:6])} ...")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = time.monotonic() - t0

        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed/60:.1f} min) — {status}")

        run_results.append({
            "name": cfg["name"],
            "tag": tag,
            "returncode": result.returncode,
            "elapsed_min": elapsed / 60,
        })

    if not args.dry_run:
        # Save run metadata
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = RESULTS_DIR / "sweep_runs.json"
        with open(meta_path, "w") as f:
            json.dump({
                "sweep_name": sweep["sweep_name"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
