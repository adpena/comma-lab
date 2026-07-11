#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#430 coherent-schedule backtest CLI + scorer-model-arms tournament ($0, read-only).

One command, three measured products (each durable JSON under
``experiments/results/costate_organ_backtests/``):
  1. the scorer-model arms' tournament rows (H/I/J/K/L/M vs the incumbents) + the
     early-fold walk-forward instrument (the n=1-fragility question),
  2. the ball-agreement faithfulness audit of the margin-surrogate boundary model
     (faithful-KD 2306.04431 acceptance, advection-ball protocol),
  3. the #430 schedule replay (hand vs selective vs always-on vs uniform; 2607.08716
     selective-intervention shape) + the emitted OperatorGoTicket (advisory — the
     ticket structurally cannot execute; live A/B stays in gates_owed).

Every number [macOS advisory] NON-PROMOTABLE, score_claim=False. The organ is MEANS;
the pointer moves only through byte-closed exact eval.

Usage:
  .venv/bin/python tools/schedule_bundle_backtest.py \
      --run-dir experiments/results/levelset_v752_baseline_20260710T185913Z
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NEW_ARMS = ("H_smoothed_argmax", "I_comma10k_regime", "J_adv_boundary",
            "K_perclass_v8", "L_priormean_comma10k", "M_priormean_advb")
BASELINES = ("A_ridge_solve", "G_ridge_scorerprior", "E_prototype",
             "E_prototype_bregman")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-archs", default="E_prototype_bregman,A_ridge_solve,K_perclass_v8")
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/costate_organ_backtests"))
    args = ap.parse_args(argv)

    from tac.witness_control.lambda_net import backtest, read_trajectory
    from tac.witness_control.schedule_backtest import (
        backtest_schedule_430, compose_430_ticket)
    from tac.witness_control.scorer_model_arms import (
        ball_agreement_audit, walkforward_per_fold)

    traj = read_trajectory(args.run_dir)
    print(f"trajectory: {traj.n_verdicts} verdicts, {len(traj.loss_terms)} loss rows "
          f"[{traj.run_dir}]")

    print("\n== scorer-model arms tournament (LOO + WALK-FORWARD vs persistence) ==")
    arch_reports = {}
    for arch in NEW_ARMS + BASELINES:
        try:
            arch_reports[arch], _ = backtest(traj, architecture=arch, seed=args.seed)
            r = arch_reports[arch]
            print(f"  {arch:22s} LOO {r.forecast_mae_model:.6f} | WF "
                  f"{r.walkforward_mae_model:.6f} (heur {r.walkforward_mae_heuristic:.6f})"
                  f" | AUROC {r.binding_auroc_model} | pass={r.passed}")
        except Exception as exc:                      # honest per-arch failure
            print(f"  {arch:22s} FAILED: {type(exc).__name__}: {exc}")

    print("\n== early-fold walk-forward (the n=1-fragility instrument) ==")
    early = {}
    for arch in NEW_ARMS + ("A_ridge_solve", "E_prototype"):
        try:
            early[arch] = walkforward_per_fold(traj, arch, seed=args.seed)
            e = early[arch]
            print(f"  {arch:22s} early MAE {e['early_mae_model']:.6f} "
                  f"(heur {e['early_mae_heuristic']:.6f}) | all {e['mae_model']:.6f}")
        except Exception as exc:
            print(f"  {arch:22s} FAILED: {type(exc).__name__}: {exc}")

    print("\n== ball-agreement faithfulness audit (2306.04431, advection ball) ==")
    audit = ball_agreement_audit()
    print(f"  iou={audit['iou']:.3f} precision={audit['precision']:.3f} "
          f"recall={audit['recall']:.3f} faithful={audit['faithful']}")

    print("\n== #430 schedule replay (hand / selective / always_on / uniform) ==")
    reports = []
    for arch in args.replay_archs.split(","):
        r = backtest_schedule_430(traj, arch=arch.strip(), seed=args.seed)
        reports.append(r)
        print(f"  model {r.model_arch}: self-replay MAE {r.self_replay_mae:.5f} "
              f"(final gap {r.self_replay_final_gap:.5f})")
        for res in r.results:
            print(f"    {res.policy:10s} final {res.final_weighted_dseg:.5f} "
                  f"∫dseg·dep {res.dseg_epoch_integral:.3f} "
                  f"({len(res.interventions)} interventions)")
        print(f"    selective beats hand: {r.selective_beats_hand} | "
              f"beats always-on: {r.selective_beats_always_on}")

    ticket = compose_430_ticket(traj, reports)
    print("\n== OperatorGoTicket (#430; advisory — cannot execute) ==")
    print(f"  action: {ticket.action} | actuation: {ticket.actuation}")
    print(f"  gates_owed: {ticket.gates_owed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"schedule_430_backtest_{stamp}.json"
    out.write_text(json.dumps({
        "run_dir": traj.run_dir, "seed": args.seed, "generated_at": stamp,
        "axis_tag": "[macOS advisory] NON-PROMOTABLE", "score_claim": False,
        "promotable": False,
        "arms_tournament": {k: v.to_dict() for k, v in arch_reports.items()},
        "early_fold_walkforward": early,
        "ball_agreement_audit": audit,
        "schedule_430": [r.to_dict() for r in reports],
        "ticket": dataclasses.asdict(ticket),
    }, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
