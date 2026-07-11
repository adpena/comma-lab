#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#426 costate-organ backtest CLI — the honesty gate, runnable any time ($0, read-only).

Runs on a witness run directory's telemetry (READ-ONLY — never writes into the run dir):
  1. the 4-architecture λ-net tournament (``tac.witness_control.lambda_net.benchmark_all``),
  2. the 5-mode routing benchmark (nested LOO; ``costate_panel.routing_benchmark``),
  3. the panel verdict at the declared routing mode (+ Rashomon escalations),
and writes one durable JSON artifact under ``experiments/results/costate_organ_backtests/``.

Every number: [macOS advisory] NON-PROMOTABLE, score_claim=false. The organ is MEANS.

Usage:
  .venv/bin/python tools/lambda_net_backtest.py \
      --run-dir experiments/results/levelset_v752_baseline_20260710T185913Z
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-routing", action="store_true",
                    help="skip the (slow, ~7 min) nested-LOO routing benchmark")
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/costate_organ_backtests"))
    args = ap.parse_args(argv)

    from tac.witness_control.costate_panel import routing_benchmark, run_panel
    from tac.witness_control.lambda_net import benchmark_all, read_trajectory

    traj = read_trajectory(args.run_dir)
    print(f"trajectory: {traj.n_verdicts} verdicts, {len(traj.loss_terms)} loss rows, "
          f"{len(traj.lever_names)} levers  [{traj.run_dir}]")

    print("\n== λ-net architecture tournament (LOO backtest vs persistence heuristic) ==")
    arch_reports = benchmark_all(traj, seed=args.seed)
    for arch, r in sorted(arch_reports.items(), key=lambda kv: kv[1].forecast_mae_model):
        print(f"  {arch:16s} fMAE {r.forecast_mae_model:.6f} (heur {r.forecast_mae_heuristic:.6f})"
              f" | per-class {r.forecast_perclass_mae_model:.6f}"
              f" | binding AUROC {r.binding_auroc_model} vs {r.binding_auroc_magnitude_heuristic}"
              f" | pass={r.passed}")

    routing = None
    if not args.skip_routing:
        print("\n== routing benchmark (nested LOO; 5 modes + solo ablation) ==")
        routing = routing_benchmark(traj, seed=args.seed)
        for m, v in sorted(routing.per_mode_mae.items(), key=lambda kv: kv[1]):
            print(f"  {m:26s} {v:.6f} (per-class {routing.per_mode_perclass_mae[m]:.6f})")
        print("  solo:", {k: round(v, 6) for k, v in
                          sorted(routing.per_lens_solo_mae.items(), key=lambda kv: kv[1])})
        print("  winner:", routing.winner)

    print("\n== panel verdict (EVIDENCE_SHRUNK_STACKING) ==")
    verdict = run_panel(traj, routing_mode="EVIDENCE_SHRUNK_STACKING", seed=args.seed)
    for r in verdict.reports:
        print(f"  {r.spec.name:15s} act={r.activation:.3f} :: {r.insight[:100]}")
    print(f"  disagreement {verdict.disagreement:.2f}; spawn tickets {len(verdict.spawn_tickets)}")
    for t in verdict.spawn_tickets:
        print(f"    SYSTEM-2: {t.trigger}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"costate_organ_backtest_{stamp}.json"
    payload = {
        "run_dir": traj.run_dir, "seed": args.seed, "generated_at": stamp,
        "axis_tag": "[macOS advisory] NON-PROMOTABLE", "score_claim": False,
        "promotable": False,
        "architecture_tournament": {k: v.to_dict() for k, v in arch_reports.items()},
        "routing_benchmark": routing.to_dict() if routing else None,
        "panel": {
            "routing_mode": verdict.routing_mode,
            "disagreement": verdict.disagreement,
            "consensus_lambda": verdict.consensus_lambda,
            "activations": {r.spec.name: r.activation for r in verdict.reports},
            "spawn_triggers": [t.trigger for t in verdict.spawn_tickets],
        },
    }
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
