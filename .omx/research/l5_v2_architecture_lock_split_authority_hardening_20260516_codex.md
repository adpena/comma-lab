# L5 v2 architecture-lock split-authority hardening

- schema: `l5_v2_architecture_lock_split_authority_hardening_v1`
- created_at_utc: `2026-05-16T22:16:36Z`
- surface: `src/tac/optimization/l5_staircase_v2.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

Readiness and the architecture-lock packet had diverged. The canonical
architecture-lock packet required all gate evidence plus:

- paired CPU/CUDA side-info effect curve;
- first-anchor timing-smoke custody;
- exact or diagnostic paired anchor custody.

The intermediate TT5L readiness surface exposed `architecture_lock_allowed`
after only side-info/probe/paired-axis-plan evidence plus the side-info curve.
That meant operator briefing or autopilot could observe a true architecture-lock
boolean while the stricter packet still refused lock. This is a false-authority
hazard in the L5 v2 staircase.

## Fix

`_l5_v2_tt5l_campaign_readiness_from_dispatch_readiness()` now uses the same
authority threshold as `l5_v2_architecture_lock_packet()`:

- `sideinfo_effect_curve_allowed`
- `sideinfo_effect_curve_valid`
- `probe_valid`
- `paired_axis_plan_valid`
- `timing_smoke_valid`
- `anchor_pair_valid`

The next-action ladder still advances through the missing artifacts in order:
materialize first-anchor timing-smoke custody first, then materialize the exact
or diagnostic anchor pair. What changed is only the authority boolean: it no
longer turns true during the intermediate state.

## Regression

`src/tac/tests/test_l5_staircase_v2.py` now asserts:

- side-info curve present but timing-smoke missing keeps
  `architecture_lock_allowed=false`;
- timing-smoke present but anchor-pair missing still keeps
  `architecture_lock_allowed=false`;
- the architecture-lock packet reports
  `readiness_architecture_lock_allowed=false` in the same intermediate state.

Full lock remains allowed only after the full custody bundle exists.
