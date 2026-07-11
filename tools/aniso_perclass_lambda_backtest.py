#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#433 anisotropic-coupled per-class-λ backtest — the P0 honesty gate ($0, read-only).

Runs, on the live #205 trajectory telemetry (READ-ONLY; the run slot is untouched):

  1. the MEASURED physics coverage report (per-class bulk-vs-boundary Fisher-regime
     split at the measured flip temperature, per-class-pair σ_cc′/power-diagram
     coupling, per-pair along/across-tangent anisotropy — the UNION residual framing:
     Road-mass + Lane + ALL class edges, never Lane-only);
  2. the P0 tournament: ANISOTROPIC-COUPLED arms (N/P) vs the ISOTROPIC-INDEPENDENT
     baselines (A ridge, Q the same-formulation ablation) vs the earlier arms (K
     isotropic-coupled, L the failed empirical-Bayes prior-mean) — LOO + WALK-FORWARD
     + binding AUROC + early-fold walk-forward (the n=1-fragility instrument);
  3. thread 1: the OPEN comma10k-family member R (score-law-pinned scale + anisotropy);
  4. thread 2: the openpilot ISOLATED arms (O reweight / S prior-mean) + the measured
     ego-geometry prior;
  5. thread 3: SAO trust-region walk-forward + GEPA cycle 2 on the extended tournament.

Writes ONE durable JSON artifact under experiments/results/costate_organ_backtests/
and (default ON) compounds the organ ledger. Every number [macOS advisory]
NON-PROMOTABLE, score_claim=false. The organ is MEANS; pointer UNMOVED.

Usage:
  .venv/bin/python tools/aniso_perclass_lambda_backtest.py \
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

#: the P0 comparison set: physics arms + their honest baselines
ARMS = ("A_ridge_solve", "K_perclass_v8", "L_priormean_comma10k",
        "N_aniso_coupled", "P_priormean_aniso", "Q_priormean_iso",
        "R_priormean_c10k_scorelaw", "O_openpilot_geom", "S_priormean_openpilot")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--no-record", action="store_true",
                    help="skip the organ-ledger compounding record")
    ap.add_argument("--out-dir",
                    default=str(REPO / "experiments/results/costate_organ_backtests"))
    args = ap.parse_args(argv)

    from tac.witness_control.aniso_perclass_lambda import (
        CLASS_NAMES,
        measure_aniso_class_profiles,
        measure_flip_temperature,
        measure_openpilot_geometry_prior,
        sao_trustregion_walkforward,
        smoothed_grad_per_class,
    )
    from tac.witness_control.lambda_net import backtest, read_trajectory
    from tac.witness_control.scorer_model_arms import walkforward_per_fold

    traj = read_trajectory(args.run_dir)
    print(f"trajectory: {traj.n_verdicts} verdicts, {len(traj.loss_terms)} loss rows, "
          f"{len(traj.lever_names)} levers  [{traj.run_dir}]")

    # ── 1. the measured physics coverage report ────────────────────────────────
    print("\n== MEASURED physics profiles (flip temperature; UNION residual) ==")
    eps_flip = measure_flip_temperature()
    profiles = measure_aniso_class_profiles()
    op_prior = measure_openpilot_geometry_prior()
    g = smoothed_grad_per_class()
    print(f"  eps_flip {eps_flip:.4f} | annulus area {profiles.annulus_area_frac:.4f} "
          f"| annulus susceptibility {profiles.annulus_susc_frac:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s} susc-share {profiles.total_susc_share[c]:.3f} "
              f"bulk/boundary {profiles.bulk_frac[c]:.3f}/"
              f"{1 - profiles.bulk_frac[c]:.3f} "
              f"g^eps {g[c]:.2e} op-addressable {op_prior.addressable_frac[c]:.3f}")
    C = profiles.coupling_matrix()
    print("  coupling C_phys (rows respond, cols lever-target):")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:10s} " + " ".join(f"{v:.3f}" for v in C[c]))

    # ── 2-4. the tournament (LOO + WF + AUROC) + early-fold instrument ─────────
    print("\n== tournament (LOO + WALK-FORWARD vs persistence; early folds) ==")
    arch_reports: dict = {}
    per_fold: dict = {}
    for arch in args.arms.split(","):
        report, _field = backtest(traj, architecture=arch, seed=args.seed)
        arch_reports[arch] = report
        per_fold[arch] = walkforward_per_fold(traj, arch, seed=args.seed)
    for arch, r in sorted(arch_reports.items(),
                          key=lambda kv: kv[1].walkforward_mae_model):
        pf = per_fold[arch]
        print(f"  {arch:26s} LOO {r.forecast_mae_model:.6f} "
              f"| WF {r.walkforward_mae_model:.6f} (heur {r.walkforward_mae_heuristic:.6f})"
              f" | early-WF {pf['early_mae_model']:.6f} (heur {pf['early_mae_heuristic']:.6f})"
              f" | AUROC {r.binding_auroc_model} | pass={r.passed} wf={r.passed_walkforward}")

    # ── 5. thread 3: SAO trust region + GEPA cycle 2 ───────────────────────────
    print("\n== SAO trust region (pre-registered radii, all reported) ==")
    sao = sao_trustregion_walkforward(traj, seed=args.seed)
    print(f"  ridge plain WF {sao['wf_mae_ridge_plain']:.6f} | "
          f"persistence {sao['wf_mae_persistence']:.6f} | " +
          " ".join(f"r={k}: {v:.6f}" for k, v in sao["radii"].items()))

    print("\n== GEPA cycle 2 (extended tournament; reflection proposes, backtest disposes) ==")
    from tac.witness_control.gepa_reflection import run_gepa_cycle
    gepa = run_gepa_cycle(traj, arch_reports, incumbent="E_prototype_bregman",
                          seed=args.seed)
    print(f"  incumbent E_prototype_bregman WF {gepa.incumbent_wf}")
    for c in gepa.candidates:
        print(f"  {c.status:8s} {c.name:40s} wf={c.measured.get('wf_mae')}")

    # ── durable artifact ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"aniso_perclass_lambda_backtest_{stamp}.json"
    payload = {
        "task": "#433 anisotropic-coupled per-class lambda (P0)",
        "run_dir": traj.run_dir, "seed": args.seed, "generated_at": stamp,
        "axis_tag": "[macOS advisory] NON-PROMOTABLE", "score_claim": False,
        "promotable": False,
        "flip_temperature": eps_flip,
        "physics_profiles": profiles.to_jsonable(),
        "smoothed_grad_per_class_at_flip_eps": [float(v) for v in g],
        "openpilot_prior": op_prior.to_jsonable(),
        "tournament": {k: v.to_dict() for k, v in arch_reports.items()},
        "walkforward_per_fold": per_fold,
        "sao_trust_region": sao,
        "gepa_cycle2": {
            "incumbent": gepa.incumbent, "incumbent_wf": gepa.incumbent_wf,
            "adopted": gepa.adopted,
            "candidates": [{"name": c.name, "status": c.status,
                            "measured": c.measured, "reflection": c.reflection}
                           for c in gepa.candidates],
        },
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
            rec = compose_trajectory_record(traj, arch_reports, [])
            append_trajectory_record(rec)
            s = organ_summary()
            print(f"organ ledger compounded: {s['n_records']} record(s), "
                  f"recommended={s['recommended_architecture']}")
        except Exception as exc:
            print(f"organ-ledger record FAILED ({type(exc).__name__}: {exc}) — "
                  "artifact still written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
