# Retroactive Sweep — PR95 Curriculum MLX-LOCAL Smoke Validation (m9-v3 op-routable #2 pre-flight) — 2026-05-30

**Per Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP**
**Lane**: `lane_pr95_curriculum_mlx_local_smoke_validation_20260530`
**Anchor commit**: TBD (this landing)
**Predecessor commits**: c91481212 (canonical helper) + 49f41e22c (sister-wave wire-in)

## 4-field contract per Catalog #348

### 1. Bug-class symptom signature

The pre-flight validation is NOT a bug-fix gate — it's a structural verification that the canonical 3-layer sister-wave wire-in (long_training_canonical → harness → adapter) actually executes the canonical PR95 8-stage curriculum + Muon-final-stage-only per L14/L15. The bug class this validation extincts is "canonical helper landed + sister-wave wire-in landed + paid-CUDA dispatch fires WITHOUT empirical proof that the 3 layers actually compose end-to-end" — i.e. the Slot EEE FAKE pattern (function claims to apply a curriculum but the per-stage hparam transitions never reach the optimizer/loss). Symptom signature: any historical verdict that DEPENDED on "PR95 curriculum is wired into z6_v2 / z8 / dreamer_v3_rssm" assumption WITHOUT having empirically verified the 8-stage transitions + Muon-final-stage-only + hparam transitions through the observability surface.

### 2. Pre-validation window

Window = commits since predecessor c91481212 (2026-05-30 14:27 -0500) through this validation commit. Specifically the sister-wave wire-in commit 49f41e22c (2026-05-30 14:55 -0500) ASSUMED the 3-layer architecture composes end-to-end per the 41/41 helper tests; THIS validation is the empirical anchor that confirms the composition.

### 3. Historical KILL / DEFER / FALSIFY search results

Searched `.omx/state/probe_outcomes.jsonl` + `.omx/research/*killed*.md` + `*falsified*.md` + `*deferred*.md` for verdicts citing `pr95_faithful_curriculum` / `m9_v3` / `pr95-faithful-curriculum-enabled` tokens that might be invalidated by the validation evidence. Findings:

- **NO historical KILL verdicts** invalidated by this validation (predecessor c91481212 is <24h old; sister-wave wire-in 49f41e22c is <8h old; canonical helper has 0 prior KILL anchors)
- **NO historical DEFER verdicts** invalidated (the m9-v3 sister-wave wire-in's own PROCEED probe outcome 2026-05-30T19:56:36Z + the predecessor's c91481212 PROCEED 14-day advisory are BOTH ratified by this validation evidence, NOT invalidated)
- **NO historical FALSIFY verdicts** invalidated (no prior anchors falsify the canonical-helper-routes-curriculum-via-notify_global_epoch contract)

### 4. Per-finding RE-EVAL-priority assignment

Zero findings require RE-EVAL. The validation strengthens the existing PROCEED chain (predecessor → sister-wave-wire-in → THIS smoke validation) rather than invalidating it. The validation evidence empirically anchors canonical equation `pr95_faithful_curriculum_cross_substrate_compounding_savings_v1` (residual=0.0; predicted=empirical=8 stages per L14) and supports operator-routing m9-v3 op-routable #2 paired-CUDA RATIFICATION on Modal A100 ~$6-15.

## Operator-routable next steps

1. **Highest-EV first paired-CUDA target = z8** per Catalog #312 canonical quadruple binding-depth + active Z8 Phase 2 build per memory entry `[[z8-hierarchical-predictive-coding]]` 2026-05-29
2. **Z6_v2 SECOND** (L1 cargo-cult-unwind + 32-pair LONG-RUN per trainer docstring)
3. **Dreamer_v3_rssm THIRD** (L0 SCAFFOLD; requires L1 advancement per Catalog #233 first)

mission_predicted_contribution per Catalog #300: `apparatus_maintenance` (closes the empirical verification gap between sister-wave wire-in + paid-CUDA dispatch; the canonical-equation EmpiricalAnchor + probe outcome PROCEED 14-day advisory both inherit from this validation evidence).
