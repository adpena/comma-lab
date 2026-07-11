#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#434 THE TRANSIENT FORGE — the honest adoption-gate CLI ($0, read-only).

Runs the synthetic-trajectory engine (``tac.witness_control.transient_forge``) against a
witness run directory's REAL trajectory and answers ONE question with a MEASURED verdict:

    does synthetic (regime→λ) training data confer REAL chronological walk-forward skill
    over the real-only ablation, persistence, AND the incumbent?

The Forge sees ONLY the real prefix ≤ k at each fold (chronological hygiene). Synthetic
data is the TREATMENT; the REAL folds are the TEST. Synthetic-fold / in-sample wins are
NEVER adoption evidence (NO-FAKE class #3). A non-adopting result is an HONEST NEGATIVE
(the engine is built; the synthetic data does not yet confer skill → the iteration target
is named). Writes one durable JSON under ``experiments/results/transient_forge_backtests/``.

CONTAINMENT: pure numpy simulation; NO scorer forward, NO GPU, NO dispatch, NO witness
training (tier-2 real micro-runs are operator-GO and are NOT fired here). Every number is
``[macOS advisory] NON-PROMOTABLE, score_claim=false``. The organ is MEANS.

Usage:
  .venv/bin/python tools/transient_forge_backtest.py \
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
    ap.add_argument("--n-candidates", type=int, default=40,
                    help="PDR parallel breadth (candidate trajectories per fold)")
    ap.add_argument("--max-synthetic-intervals", type=int, default=120)
    ap.add_argument("--out-dir",
                    default=str(REPO / "experiments/results/transient_forge_backtests"))
    args = ap.parse_args(argv)

    from tac.witness_control.lambda_net import read_trajectory
    from tac.witness_control.transient_forge import ForgeConfig, adoption_backtest

    traj = read_trajectory(args.run_dir)
    print(f"trajectory: {traj.n_verdicts} verdicts, {len(traj.lever_names)} levers "
          f"[{traj.run_dir}]")
    cfg = ForgeConfig(n_candidate_trajectories=args.n_candidates,
                      max_synthetic_intervals=args.max_synthetic_intervals,
                      seed=args.seed)

    print("\n== THE TRANSIENT FORGE — real chronological walk-forward adoption gate ==")
    rep = adoption_backtest(traj, cfg)
    print(f"  persistence (null)      {rep.wf_persistence:.6f}")
    print(f"  incumbent (bregman)     {rep.wf_incumbent:.6f}")
    print(f"  real-only ridge (c)     {rep.wf_real_only_ridge:.6f}")
    print(f"  forge naive-concat      {rep.wf_forge_ridge:.6f}  [diagnostic — synthetic swamps]")
    print(f"  forge prior-mean (d)    {rep.wf_forge_ridge_pruned:.6f}  [PRIMARY, optimal form]")
    print(f"  beats persistence={rep.beats_persistence}  incumbent={rep.beats_incumbent}  "
          f"real-only={rep.beats_real_only}  => ADOPTED={rep.adopted}")

    print("\n  per-fold (real_only vs forge prior-mean; selected λ = trust-in-synthetic):")
    for f in rep.per_fold:
        print(f"    ep{f['epoch']:.0f}: real {f['real_only']:.5f} vs forge "
              f"{f['forge_priormean']:.5f} (λ={f['selected_prior_strength']}) "
              f"{'✓forge' if f['forge_beats_real'] else ''}")

    print("\n  corpus (per fold: generated→regret→diversity→intervals; coverage; eff-rank; mem):")
    for c in rep.corpus_stats:
        print(f"    k{c['k']}: {c['n_generated']}→{c['n_after_regret']}→"
              f"{c['n_after_diversity']}→{c['n_intervals']}iv "
              f"(pruned-out {c['n_pruned_out']}); cov {c['archive_coverage']}; "
              f"effrank {c['effective_rank']:.2f}; mem-excess {c['memorization_excess']:+.3f}")

    a = rep.aniso_acid
    print(f"\n  #433 aniso acid test: P(aniso) {a['P_aniso_wf_mae']:.6f} vs "
          f"Q(iso) {a['Q_iso_wf_mae']:.6f}; separation {a['separation']:+.6f} "
          f"(aniso_helps={a['aniso_helps']})")

    print(f"\n  VERDICT: {rep.verdict}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"transient_forge_backtest_{stamp}.json"
    out.write_text(json.dumps(rep.to_dict(), indent=1, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
