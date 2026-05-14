#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DARTS-SuperNet search over the time-traveler L5 autonomy architecture axes.

Per operator directive 2026-05-13 OPT4: build differentiable architecture
search over the 5 axes specified in the time-traveler memos.

Usage:
    .venv/bin/python experiments/search_time_traveler_supernet.py \\
        --total-steps 300 --top-k 5 \\
        --output reports/raw/2026-05-13-darts-supernet-time-traveler/

Per CLAUDE.md "Forbidden device-selection defaults": device defaults to
``cpu`` (this is a CPU smoke search on a proxy SuperNet). MPS is intentionally
NOT supported. ``--device cuda`` requires CUDA to be available; the search
core is so cheap (~1s for 300 CPU steps) that CUDA gives negligible win.

Per CLAUDE.md "Score-claim discipline": the proxy SuperNet's predicted
score is NOT an authoritative measurement. The output JSON carries
``score_claim=False`` / ``promotion_eligible=False`` /
``ready_for_exact_eval_dispatch=False``. The deliverable is a RANKED LIST
of top-k candidate architectures for downstream substrate engineering.

Source memos:
    - .omx/research/time_traveler_architecture_reverse_engineered_20260513.md
    - .omx/research/expert_team_hardware_physics_future_ledgers/07_time_traveler_2032_l5_autonomy_secret.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.composition.darts_supernet import (  # noqa: E402
    DartsSearchResult,
    default_search_axes,
    run_supernet_search,
    write_provenance,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_summary(result: DartsSearchResult, total_seconds: float) -> None:
    """Print a one-screen summary of the search outcome."""
    print()
    print("=" * 72)
    print(" DARTS-SuperNet time-traveler architecture search — RESULTS")
    print("=" * 72)
    print(f" wall-clock: {total_seconds:.2f}s for {result.total_steps} steps")
    print(f" final predicted score (low-T discrete): {result.final_score:.4f}")
    print()
    print(" Discovered architecture (argmax-α per axis):")
    print(f"   {'axis':22s} {'value':12s} {'name':12s} {'KL nats':10s} verdict")
    for axis, value in result.discovered_values.items():
        name = result.discovered[axis]
        kl = result.kl_per_axis[axis]
        verdict = (
            "decisive" if kl >= 2.0
            else "moderate" if kl >= 1.0
            else "inconclusive"
        )
        print(
            f"   {axis:22s} {str(value):12s} {name:12s} "
            f"{kl:.3f}     [{verdict}]"
        )
    print()
    print(f" Top {len(result.ranked_top_k)} architectures (by marginal-α × predicted score):")
    for i, (s, arch) in enumerate(result.ranked_top_k):
        print(f"   #{i + 1}  predicted_score={s:.4f}")
        for k, v in arch.items():
            print(f"          {k} = {v}")
    print()
    print(" Per CLAUDE.md score-claim discipline:")
    print("   - score_claim:               False")
    print("   - promotion_eligible:        False")
    print("   - ready_for_exact_eval:      False")
    print("   - evidence_grade:            MPS-research-signal")
    print()
    print(" The top-k ranking is a DESIGN-SPACE SIGNAL for the substrate-")
    print(" engineering lane that consumes it. Empirical contest-CUDA +")
    print(" contest-CPU anchors are required before any architecture is")
    print(" promoted to dispatch.")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=300,
        help="Number of DARTS alternating-SGD steps (default: 300).",
    )
    parser.add_argument(
        "--arch-lr",
        type=float,
        default=3e-4,
        help="Adam lr for the α architecture parameters (default: 3e-4).",
    )
    parser.add_argument(
        "--weight-lr",
        type=float,
        default=3e-3,
        help="SGD lr for the AxisOp MLP weights (default: 3e-3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Replicas per forward (latent broadcasted). Default 1.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top architectures to enumerate (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="torch.manual_seed (default: 1234).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help=(
            "Compute device. Default 'cpu' per CLAUDE.md MPS-falsification-"
            "trap. Pass 'cuda' explicitly if you have a CUDA GPU and want "
            "the (negligible) speedup. MPS is intentionally NOT supported."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/raw") / f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}-darts-supernet-time-traveler",
        help=(
            "Output directory for the provenance JSON + summary log. "
            "Defaults to a timestamped reports/raw/ subdir."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = default_search_axes()
    print(f"[{_utc_now_iso()}] starting DARTS-SuperNet search:")
    print(f"  axes: {cfg.axis_names()}")
    print(f"  total enumerated archs: {cfg.total_architectures()}")
    print(f"  search steps: {args.total_steps}")
    print(f"  device: {args.device}")
    print(f"  seed: {args.seed}")
    print(f"  output: {args.output}")
    print()
    t0 = time.time()
    _, result = run_supernet_search(
        config=cfg,
        total_steps=args.total_steps,
        arch_lr=args.arch_lr,
        weight_lr=args.weight_lr,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
    )
    dt = time.time() - t0
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output / "supernet_search_provenance.json"
    write_provenance(result, provenance_path)
    # Also write a search_command.txt for reproducibility.
    cmd_path = args.output / "search_command.txt"
    cmd_path.write_text(
        " ".join(
            [
                ".venv/bin/python",
                "experiments/search_time_traveler_supernet.py",
                f"--total-steps {args.total_steps}",
                f"--arch-lr {args.arch_lr}",
                f"--weight-lr {args.weight_lr}",
                f"--batch-size {args.batch_size}",
                f"--top-k {args.top_k}",
                f"--seed {args.seed}",
                f"--device {args.device}",
                f"--output {args.output}",
            ]
        )
        + "\n"
    )
    summary_path = args.output / "summary.json"
    summary_payload = {
        "started_at_utc": _utc_now_iso(),
        "wall_clock_seconds": dt,
        "total_steps": result.total_steps,
        "discovered": result.discovered,
        "discovered_values": {
            k: (v if isinstance(v, str) else int(v))
            for k, v in result.discovered_values.items()
        },
        "final_score_predicted": result.final_score,
        "kl_nats_per_axis": dict(result.kl_per_axis),
        "ranked_top_k": [
            {"predicted_score": float(s), "architecture": dict(arch)}
            for s, arch in result.ranked_top_k
        ],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "MPS-research-signal",
        "source_memos": [
            ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
            ".omx/research/expert_team_hardware_physics_future_ledgers/07_time_traveler_2032_l5_autonomy_secret.md",
        ],
        "lane_id": "lane_darts_supernet_time_traveler_architecture_search_20260513",
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    _print_summary(result, dt)
    print()
    print(f"[{_utc_now_iso()}] artifacts written to {args.output}:")
    print(f"  - {provenance_path.relative_to(args.output)}")
    print(f"  - {summary_path.relative_to(args.output)}")
    print(f"  - {cmd_path.relative_to(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
