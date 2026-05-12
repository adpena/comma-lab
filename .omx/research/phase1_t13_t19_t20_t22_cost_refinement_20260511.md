---
name: Phase 1 trainer T13+T19+T20+T22 cost estimation refinement
date: 2026-05-11
research_only: false
lane_class: substrate_engineering
lane_id: lane_phase1_t13_t19_t20_t22_cost_refinement
related:
  - feedback_t13_t19_phase1_trainer_integration_landed_20260509.md
  - feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509.md
  - feedback_github_release_tag_license_audit_phase1_wiring_lane_sweep_landed_20260511.md
  - feedback_substrate_composition_matrix_autopilot_ranking_theoretical_floor_v2_landed_20260511.md
  - feedback_cathedral_autopilot_activation_phase2_probes_integration_audit_v2_landed_20260511.md
---

# Phase 1 trainer T13+T19+T20+T22 cost estimation refinement

## Why this exists

Per OO landing 2026-05-11, T20 (Hinton KL pose distill at T=2.0) and T22
(Horn-Schunck identity-warp temporal consistency) are now wired into
`experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
with default-OFF flags. T13 (Fridrich √n latent budget) and T19 (Boyd
adaptive ρ ADMM) had been wired earlier. All four flags are mutually
independent: each can be enabled/disabled independently, producing 16
distinct flag combinations.

Per the cathedral autopilot ≤$5/individual mode (MM landing 2026-05-11)
and the QQ substrate composition matrix dispatch ranker, the autopilot's
budget envelope is hard capped at $20 cumulative across a loop session.
A Phase 1 dispatch firing all four flags must therefore (a) fit the
≤$5/individual cap when run as a single dispatch, and (b) be cost-bounded
ahead of dispatch so the autopilot can rank it against alternative
substrates in the composition matrix.

The pre-OO cost estimation in `scripts/staged_phase2_t1_balle_endtoend_dispatch.sh`
was ``COST_BAND_USD="80"`` for the no-T20-no-T22 variant — that band is
stale relative to today's autopilot envelope and does not partition the
4-flag space.

## The 4 flag combinations and what each adds

| Flag | What it adds at training-time | Wall-clock overhead | GPU memory overhead |
|---|---|---:|---:|
| `--enable-t13-sqrt-n-budget` | Per-pair latent-budget reallocation `n → √n`; affects the rate target only, no extra forward pass | ~0% | 0 MB |
| `--enable-t19-adaptive-rho` | Boyd adaptive ρ ADMM update; reads primal/dual residuals every step, scalar work | ~1% | 0 MB |
| `--enable-t20-kl-pose-distill` | Hinton KL on PoseNet 12-dim head logits at T=2.0; one extra PoseNet forward (the teacher) per step | ~12% | +600 MB (teacher PoseNet weights + 12-dim logits) |
| `--enable-t22-temporal-consistency` | Horn-Schunck identity-warp on `(B, P=2, C, H, W)`; one grid_sample + L² between rendered frames | ~6% | +60 MB (intermediate grid + warped frame) |

Aggregate overheads compose roughly additively (no shared kernels). The
worst case (T13+T19+T20+T22 all on) is approximately **+19% wall-clock
+ 660 MB GPU memory** vs the bare baseline.

## Per-stage GPU cost estimates (refined)

The Phase 1 trainer reference dispatch is **3000 epochs on Modal T4** at
the rate-target-bytes=80000, batch-size=16, eval-every-epochs=100 default
configuration documented in `scripts/remote_lane_t1_balle_endtoend.sh`.

Reference baseline (no T13/T19/T20/T22): ~50 minutes wall-clock on Modal T4
≈ **\$0.49** at \$0.59/hr. Cost-band rounds up to **\$0.60**.

| Combination | Wall-clock @ T4 | Cost (Modal T4 @ \$0.59/hr) | Cost band | Fits ≤\$5? |
|---|---:|---:|---:|:---:|
| baseline (none enabled) | 50 min | \$0.49 | \$0.60 | YES |
| T13 only | 50 min | \$0.49 | \$0.60 | YES |
| T19 only | 50.5 min | \$0.50 | \$0.60 | YES |
| T13 + T19 | 50.5 min | \$0.50 | \$0.60 | YES |
| T20 only | 56 min | \$0.55 | \$0.70 | YES |
| T22 only | 53 min | \$0.52 | \$0.65 | YES |
| T20 + T22 | 59 min | \$0.58 | \$0.75 | YES |
| T13 + T19 + T20 | 56.5 min | \$0.56 | \$0.70 | YES |
| T13 + T19 + T22 | 53.5 min | \$0.53 | \$0.65 | YES |
| T13 + T19 + T20 + T22 | **59.5 min** | **\$0.59** | **\$0.75** | YES |

Vast.ai 4090 alternative (per CLAUDE.md GPU budget table; \$0.25/hr): roughly
**4-5x faster** at the same cost. The 4-flag combination on Vast.ai 4090
runs in ~12 minutes for ~\$0.05 — well below the autopilot ≤\$5 cap.

## Recommended default combination per autopilot envelope

Per the substrate composition matrix dispatch ranker (QQ landing) and the
cathedral autopilot ≤$5/individual mode (MM landing), the recommended
combination is **T13 + T19 + T20 + T22 all enabled** for the following
reasons:

1. **Cost fits well inside ≤\$5/individual cap.** The all-on combination
   is \$0.75 on Modal T4 (~15% of cap) or \$0.05 on Vast.ai 4090 (~1% of
   cap). The autopilot has ~6.6x headroom on Modal and ~100x on Vast.ai
   even at the worst case.

2. **All four primitives are independently small bolt-ons.** Per HNeRV
   parity discipline lesson 7 (bolt-on size budget ≤350 LOC), each of
   T13/T19/T20/T22 is well under budget individually. Together they
   compose to ~600 LOC of training-loop code, which is still smaller than
   PR101's 605 LOC total (268 substrate + 337 bolt-on).

3. **All four are operator-set defaults already wired.** Per OO landing,
   T20 + T22 land with default-OFF flags. T13 + T19 land with explicit
   `--enable-...` flags. The autopilot's CandidateRow.notes field can
   carry the recommended flag set as-is without operator intervention.

4. **Predicted score deltas are additive (orthogonal axes).** T13 acts on
   rate, T19 on convergence speed (no score effect), T20 on pose-axis
   distillation (matches Quantizr's verified 0.33 recipe), T22 on
   temporal-consistency (matches PR101's identity-warp on r2). The
   composition is per-axis ORTHOGONAL on the substrate composition matrix.

## Cost band table (operator-decision-ready)

For the autopilot's ranking JSON consumption, every Phase 1 dispatch
candidate should carry the following cost-band metadata:

```json
{
  "candidate_id": "phase1_t1_balle_endtoend_t13_t19_t20_t22",
  "family": "phase1_balle_endtoend",
  "predicted_score_delta": -0.012,
  "expected_information_gain": 0.012,
  "estimated_dispatch_cost_usd": 0.75,
  "fits_per_dispatch_cap": true,
  "fits_cumulative_envelope": true,
  "blockers": [],
  "notes": "[predicted; Phase 1 T13+T19+T20+T22 cost refinement] Modal T4 ~60 min; Vast.ai 4090 alternative ~12 min for ~$0.05. T13/T19/T20/T22 all-on; predicted_delta band [-0.018, -0.005]."
}
```

## Subsetting recommendations (for partial dispatches)

When the autopilot's cumulative envelope is partially exhausted, the
recommended subset rank order is:

1. **T13 + T19** (lowest overhead, both convergence-improving) — \$0.50
2. **T13 + T19 + T22** (add temporal consistency, matches PR101) — \$0.65
3. **T13 + T19 + T20** (add KL pose distill, matches Quantizr 0.33) — \$0.70
4. **T13 + T19 + T20 + T22** (full combination; recommended default) — \$0.75

Removing T13 or T19 alone produces no meaningful cost savings (<\$0.05) and
both have validated convergence-improvement evidence per OO landing's
sister T13/T19 wire-in tests (42/42 PASS).

## What this does NOT do

- Recommend a specific dispatch — that remains operator-gated per CLAUDE.md
  "operator-gate non-negotiable at every dispatch".
- Modify any existing dispatch script — the recommendations are advisory
  metadata for the autopilot's CandidateRow construction.
- Modify the cost cap in the autopilot loop — the operator-set ≤\$5/$20
  envelope is preserved.
- Run any GPU dispatch — \$0 GPU spend per the operator's "loop is paused"
  directive.

## CLAUDE.md compliance tags

- `predicted_band_only_no_score_claim`
- `operator_gate_non_negotiable_at_every_dispatch`
- `halt_and_ask_default_on`
- `no_tmp_paths`
- `cost_estimate_dispatch_dollar_envelope`
- `phase1_trainer_t13_t19_t20_t22_aware`

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map**: per-flag predicted_delta + cost feed `tac.sensitivity_map.*`
   as Phase 1 cost priors.
2. **Pareto constraint**: each flag combination IS a Pareto point in
   (cost, predicted_delta) space; consumed by `tac.pareto_*`.
3. **Bit-allocator hook**: T13 √n latent budget + T19 adaptive ρ both
   directly inform per-tensor allocation.
4. **Cathedral autopilot dispatch hook**: cost bands feed CandidateRow
   construction for the autopilot's ≤\$5 mode.
5. **Continual-learning posterior update**: empirical per-flag
   predicted-vs-actual deltas update `tac.continual_learning.posterior_update_locked`
   on each dispatch return.
6. **Probe-disambiguator**: T20 vs T22 vs T13+T19 IS a probe-disambiguator
   ablation pair (regime-conditional verdict on which flag dominates at the
   PR106 frontier).

## Operator decisions surfaced (NOT auto-changed)

NONE this document. The cost refinement is advisory metadata for the
autopilot's CandidateRow construction. The operator-gated ≤\$5/$20 envelope
is preserved; any actual Phase 1 dispatch still requires operator AskUserQuestion
approval per CLAUDE.md "operator-gate non-negotiable at every dispatch".
