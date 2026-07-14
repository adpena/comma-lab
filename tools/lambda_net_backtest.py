#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#426 costate-organ backtest CLI — the honesty gate, runnable any time ($0, read-only).

Runs on a witness run directory's telemetry (READ-ONLY — never writes into the run dir):
  1. the λ-net architecture tournament (LOO + WALK-FORWARD vs the persistence heuristic),
  2. the 5-mode routing benchmark (nested LOO; ``costate_panel.routing_benchmark``),
  3. the panel verdict at the declared routing mode (+ Rashomon escalations),
  4. the PRISM faithfulness audit (stated attribution vs suppression counterfactual),
and writes one durable JSON artifact under ``experiments/results/costate_organ_backtests/``
PLUS (default ON — the continual-learning compounding; ``--no-record`` opts out with the
recorded reason "benchmark-only run") one FEED-426-organ block in the triality ledger
``.omx/research/costate_organ_trajectory_ledger.md`` so the NEXT session inherits a
smarter organ (graph-memory indexes the block into regime nodes).

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
                    help="skip the (slow) nested-LOO routing benchmark")
    ap.add_argument("--routing-folds", type=str, default=None,
                    help="comma-separated outer fold ids (chunked-resumable execution)")
    ap.add_argument("--archs", type=str, default=None,
                    help="comma-separated architecture subset (chunked-resumable "
                         "execution — the full 8-arm tournament exceeds a single "
                         "harness window; run numpy arms + torch arms separately)")
    ap.add_argument("--no-record", action="store_true",
                    help="do NOT append the organ-ledger compounding record "
                         "(benchmark-only run; recording is the default so every "
                         "backtest makes the next session smarter)")
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/costate_organ_backtests"))
    args = ap.parse_args(argv)

    import numpy as np

    from tac.witness_control.costate_panel import routing_benchmark, run_panel
    from tac.witness_control.lambda_net import (
        benchmark_all,
        build_intervals,
        fit_score_composition,
        lever_features,
        read_trajectory,
    )

    traj = read_trajectory(args.run_dir)
    print(f"trajectory: {traj.n_verdicts} verdicts, {len(traj.loss_terms)} loss rows, "
          f"{len(traj.lever_names)} levers  [{traj.run_dir}]")

    print("\n== λ-net architecture tournament (LOO + WALK-FORWARD vs persistence) ==")
    if args.archs:
        from tac.witness_control.lambda_net import backtest
        arch_reports = {}
        for arch in args.archs.split(","):
            arch_reports[arch] = backtest(traj, architecture=arch, seed=args.seed)[0]
    else:
        arch_reports = benchmark_all(traj, seed=args.seed)
    for arch, r in sorted(arch_reports.items(), key=lambda kv: kv[1].forecast_mae_model):
        print(f"  {arch:20s} LOO {r.forecast_mae_model:.6f} (heur {r.forecast_mae_heuristic:.6f})"
              f" | WF {r.walkforward_mae_model:.6f} (heur {r.walkforward_mae_heuristic:.6f})"
              f" | per-class {r.forecast_perclass_mae_model:.6f}"
              f" | AUROC {r.binding_auroc_model} vs {r.binding_auroc_magnitude_heuristic}"
              f" | pass={r.passed} (wf={r.passed_walkforward})")

    routing = None
    if not args.skip_routing:
        folds = (tuple(int(x) for x in args.routing_folds.split(","))
                 if args.routing_folds else None)
        print("\n== routing benchmark (nested LOO; modes + solo ablation) ==")
        routing = routing_benchmark(traj, seed=args.seed, folds=folds)
        for m, v in sorted(routing.per_mode_mae.items(), key=lambda kv: kv[1]):
            print(f"  {m:26s} {v:.6f} (per-class {routing.per_mode_perclass_mae[m]:.6f})")
        print("  solo:", {k: round(v, 6) for k, v in
                          sorted(routing.per_lens_solo_mae.items(), key=lambda kv: kv[1])})
        print("  winner:", routing.winner)

    # regime-conditional self-dispatch backtest (#436): does per-STATE dispatch beat the
    # global-single-best arm walk-forward? (past-only, no look-ahead; the honest arbiter)
    dispatch = None
    live_dispatch = None
    print("\n== regime-conditional dispatch backtest (#436; walk-forward vs global-single-best) ==")
    try:
        from tac.witness_control.regime_dispatch import (
            backtest_dispatch,
            dispatch_for_trajectory,
        )
        dispatch = backtest_dispatch(traj, seed=args.seed)
        print(f"  dispatcher WF {dispatch.dispatcher_wf_mae:.6f} "
              f"(no-meta-guard {dispatch.dispatcher_wf_mae_no_meta_guard:.6f}) | "
              f"global-single-best {dispatch.global_single_best_arm} "
              f"{dispatch.global_single_best_wf_mae:.6f} | persistence "
              f"{dispatch.persistence_wf_mae:.6f}")
        print(f"  beats_persistence={dispatch.beats_persistence} "
              f"beats_global_single_best={dispatch.beats_global_single_best}")
        for r in dispatch.fold_rows:
            print(f"    ep{r['epoch']:.0f} {r['regime']:9s}→{r['tool']:24s} "
                  f"err {r['dispatcher_err']:.5f} (oracle {r['oracle_arm']}"
                  f"{' ✓' if r['route_matches_oracle'] else ''})")
        print(f"  VERDICT: {dispatch.verdict}")
        live_dispatch = dispatch_for_trajectory(traj, seed=args.seed)
        print(f"  LIVE: {live_dispatch.explain()}")
        cert = live_dispatch.classification.gate_certificate
        if cert is not None:
            print(f"  GATE: {cert.gate_dtype}; slope margin "
                  f"{cert.slope_abs_margin} ({cert.slope_margin_ulps} ULP guards); "
                  f"stable={cert.stable_beyond_float32_roundoff}")
    except Exception as exc:
        print(f"  dispatch backtest unavailable ({type(exc).__name__}: {exc})")

    print("\n== panel verdict (EVIDENCE_SHRUNK_STACKING) ==")
    verdict = run_panel(traj, routing_mode="EVIDENCE_SHRUNK_STACKING", seed=args.seed)
    for r in verdict.reports:
        print(f"  {r.spec.name:15s} act={r.activation:.3f} :: {r.insight[:100]}")
    print(f"  disagreement {verdict.disagreement:.2f}; spawn tickets {len(verdict.spawn_tickets)}")
    for t in verdict.spawn_tickets:
        print(f"    SYSTEM-2: {t.trigger}")

    # PRISM faithfulness audit (2607.08046: audit the interpretable head's own story)
    print("\n== PRISM faithfulness audit (stated vs suppression counterfactual) ==")
    faith = None
    try:
        from tac.witness_control.prototype_router import PrototypeRouterLens
        comp = fit_score_composition(traj.verdicts)
        intervals = build_intervals(traj)
        phis = np.stack([lever_features(n) for n in traj.lever_names])
        lens = PrototypeRouterLens()
        lens.fit(intervals, phis, seed=args.seed)
        x = intervals[-1].x1
        faith = lens.faithfulness_audit(x, intervals[-1].ctx, comp.grad_s_wrt_state(),
                                        lever_features("seg"))
        print(f"  faithful={faith['faithful']} max_rel_gap={faith['max_rel_gap']:.4f} "
              f"(tol {faith['rel_tol']}) over {len(faith['rows'])} fired prototypes")
    except Exception as exc:
        print(f"  audit unavailable ({type(exc).__name__}: {exc})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"costate_organ_backtest_{stamp}.json"

    # Molt-pattern router stability: content-address the current DECIDE selection,
    # replay that exact selection at advisory APPLY, and fail closed on IS until a
    # separately-custodied visited-live density exists.  No run-dir writes or actuation.
    router_stability = None
    if dispatch is not None and live_dispatch is not None:
        from tac.witness_control.router_stability import (
            append_decide_record,
            calibrate_router_forecast,
            importance_weighted_architecture_eval,
            make_decision_record,
            replay_apply,
        )
        cert = live_dispatch.classification.gate_certificate
        if cert is None:
            raise RuntimeError("dispatch omitted router gate certificate")
        replay_path = out_dir / f"costate_router_replay_{stamp}.jsonl"
        record = make_decision_record(
            run_ref=Path(traj.run_dir).name,
            decision_epoch=float(traj.verdicts[-1]["epoch"]),
            selected_regime=live_dispatch.classification.regime,
            selected_tool=live_dispatch.tool,
            certificate=cert,
        )
        append_decide_record(replay_path, record)
        replay = replay_apply(replay_path, record.decision_id)
        is_report = importance_weighted_architecture_eval(
            dispatch.fold_rows,
            custody=None,
            clip_bounds=None,
        )
        forecast_calibration = calibrate_router_forecast(dispatch.fold_rows)
        router_stability = {
            "gate_certificate": cert.to_dict(),
            "decide_record": record.to_dict(),
            "apply_replay": replay.to_dict(),
            "replay_ledger": str(replay_path),
            "importance_weighted_architecture_eval": is_report.to_dict(),
            "forecast_calibration": forecast_calibration.to_dict(),
        }
        print("\n== Molt-pattern router stability (advisory) ==")
        print(f"  gate={cert.gate_dtype} stable={cert.stable_beyond_float32_roundoff}; "
              f"replay={replay.status} tool={replay.selected_tool}")
        print(f"  IS={is_report.status}: {is_report.blocker}")
        print(f"  forecast_calibration={forecast_calibration.status}; "
              f"posterior_match_p="
              f"{forecast_calibration.terminal_posterior_match_probability:.6f}; "
              f"compute={forecast_calibration.allocation_verdict}")
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
        "faithfulness_audit": faith,
        "regime_dispatch_backtest": dispatch.to_dict() if dispatch else None,
        "router_stability": router_stability,
    }
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out}")

    if not args.no_record:
        try:
            from tac.witness_control.continual_costate import (
                append_trajectory_record,
                compose_trajectory_record,
                organ_summary,
            )
            protos = lens.prototypes if faith is not None else []
            rec = compose_trajectory_record(traj, arch_reports, protos)
            append_trajectory_record(rec)
            s = organ_summary()
            print(f"organ ledger compounded: {s['n_records']} record(s), "
                  f"{s['n_regimes']} regime(s), recommended={s['recommended_architecture']}")
        except Exception as exc:
            print(f"organ-ledger record FAILED ({type(exc).__name__}: {exc}) — "
                  "benchmark artifact still written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
