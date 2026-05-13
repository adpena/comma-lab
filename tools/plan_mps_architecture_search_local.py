#!/usr/bin/env python3
"""MPS architecture search planner — [MPS-research-signal] only.

Operator directive 2026-05-13 LOCAL HARDWARE MAXIMIZATION SWEEP Stream 3.

Emits a structured search-grid plan + cost estimate for a small HNeRV-family
architecture sweep on MPS. The plan is consumed by
``tools/build_mps_research_signal_manifest.py`` after live runs land their
observation JSONL.

Per CLAUDE.md "MPS auth eval is NOISE" — output of this plan is research
signal only:
  - evidence_grade = "MPS-research-signal"
  - score_claim = False
  - promotion_eligible = False
  - ready_for_exact_eval_dispatch = False

The 128GB unified-memory advantage of M5 Max allows huge-batch sweeps that
GPU-budget-constrained Modal/Vast.ai cannot match cheaply — at the cost of
the 23x PoseNet noise floor + 2.5x score noise documented in CLAUDE.md.
Therefore the value of this sweep is CURVE-SHAPE PRIORS, not absolute
scores: which architecture-configs are PARETO-DOMINATED on MPS likely
correlate qualitatively with paired CUDA priors.

Search grid (default):
    decoder_param_count ∈ {25000, 50000, 75000, 100000, 150000}
    latent_dim          ∈ {16, 24, 32}
    foveation_grid      ∈ {(4, 4), (8, 8)}
    seed                ∈ {0}
Total: 30 configs.

Cost on M5 Max MPS (rough estimate):
    200 epochs @ batch_size=32 on 50K-param config: ~12-15 min
    30 configs * 13 min ≈ 6.5 hours wall-clock single-process
    Parallelizable to 2-3 configs concurrent (memory permits) → ~2-3 hours.

Usage:
    .venv/bin/python tools/plan_mps_architecture_search_local.py \\
        --output-dir experiments/results/<lane>_<UTC>
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LANE_ID = "lane_local_hardware_maximization_sweep_20260513"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--decoder-param-counts",
        type=int,
        nargs="+",
        default=[25000, 50000, 75000, 100000, 150000],
    )
    p.add_argument(
        "--latent-dims", type=int, nargs="+", default=[16, 24, 32]
    )
    p.add_argument(
        "--foveation-grids",
        type=str,
        nargs="+",
        default=["4x4", "8x8"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs-per-config", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--est-min-per-config",
        type=float,
        default=13.0,
        help="Estimated wall-clock minutes per config on M5 Max MPS.",
    )
    args = p.parse_args(argv)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    configs = []
    for dp, ld, fg, sd in itertools.product(
        args.decoder_param_counts,
        args.latent_dims,
        args.foveation_grids,
        args.seeds,
    ):
        configs.append({
            "config_id": f"mps_arch_dp{dp}_ld{ld}_fov{fg}_seed{sd}",
            "decoder_param_count": dp,
            "latent_dim": ld,
            "foveation_grid": fg,
            "seed": sd,
            "epochs": args.epochs_per_config,
            "batch_size": args.batch_size,
            "device": "mps",
            "est_wall_clock_minutes": args.est_min_per_config,
            "evidence_grade": "MPS-research-signal",
            "evidence_tag": "[MPS-research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        })

    total_min = len(configs) * args.est_min_per_config
    plan = {
        "schema": "mps_architecture_search_plan_v1",
        "lane_id": LANE_ID,
        "stamped_at_utc": _utc_stamp(),
        "platform": {
            "node": platform.node(),
            "machine": platform.machine(),
        },
        "num_configs": len(configs),
        "total_est_wall_clock_minutes_serial": total_min,
        "total_est_wall_clock_hours_serial": total_min / 60.0,
        "parallelism_assumption": (
            "2-3 configs concurrent on 128GB M5 Max unified memory; "
            "~2-3 hours wall-clock with concurrency=2."
        ),
        "cost_usd": 0.0,
        "evidence_grade": "MPS-research-signal",
        "evidence_tag": "[MPS-research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "next_step": (
            "Land trainer at experiments/train_mps_arch_search.py + run "
            "configs via the plan JSON, emit observation JSONL, feed to "
            "tools/build_mps_research_signal_manifest.py for downstream "
            "atom-ledger feed (CLAUDE.md MPS-research-signal canonical path)."
        ),
        "operator_routable": True,
        "configs": configs,
    }
    out_path = out / "mps_architecture_search_plan.json"
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {out_path}")
    print(
        f"num_configs={plan['num_configs']}  "
        f"est_serial_hours={plan['total_est_wall_clock_hours_serial']:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
