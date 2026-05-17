# FIX-WAVE-R1 Catalog #315 C6 Join Closure - 2026-05-17

**Lane:** `lane_fix_wave_r1_catalog315_c6_join_closure_20260517_codex`
**Source finding:** `.omx/research/recursive_adversarial_review_r1_post_provenance_z6_c6_wave_20260517.md` F1.
**Verdict:** concrete guard closure; no score claim; no dispatch.

## Finding

Catalog #315 was intended to bind substrate dispatch readiness to the latest
sextet/grand-council verdict. The live C6 IBPS sextet row
`council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517` was emitted with
`deferred_substrate_id=null`, so the gate could not join the positive C6 verdict
back to `lane_c6_e4_mdl_ibps_substrate_20260514`. The lane appeared as
`no_council_anchor`, which is a false-authority failure: the dispatch-safety gate
had evidence, but could not see it structurally.

## Closure

Implemented a narrow historical backfill in `src/tac/preflight.py`:

- `council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517` maps to
  `c6_e4_mdl_ibps_substrate` when `deferred_substrate_id` is missing.
- Catalog #315 family tokens now include `c6`, `c6_e4`, `c6_e4_mdl_ibps`,
  and `mdl_ibps`.
- The existing C6 scope tokens were retained and the comment was corrected so
  scope membership and family-token join are not conflated.
- FIX-WAVE/review lanes are now explicitly out of Catalog #315 scope so a
  review lane named with substrate families, such as `z6_c6`, cannot inherit a
  substrate council verdict.

Generic `time_traveler` tokens were deliberately avoided: the Z6
PROCEED_WITH_REVISIONS row must not bind to older non-Z6 time-traveler lanes.

This avoids rewriting append-only `.omx/state/council_deliberation_posterior.jsonl`
while making the guard robust to the already-emitted malformed row.

## Verification

Focused tests added to
`src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py` cover:

- C6 lanes are in Catalog #315 scope.
- C6 family-token fuzzy join resolves live-style surface IDs.
- The null-`deferred_substrate_id` C6 council row backfills to the C6 lane.
- The backfill is not a positive-only shim: a malformed C6
  `PROCEED_WITH_REVISIONS` row still blocks dispatch.

Live check after the patch: Catalog #315 still returns one violation, and it is
the expected Z6-v2 PROCEED_WITH_REVISIONS blocker. C6 no longer depends on a
`no_council_anchor` false negative.

## Assumption Audit

- HARD-EARNED: `deliberation_id` is immutable enough to key the historical
  backfill; it is already the unique row identifier in the council posterior.
- HARD-EARNED: C6/MDL IBPS is an in-scope substrate-class-shift dispatch family.
- CARGO-CULTED avoided: treating the absence of `deferred_substrate_id` as
  absence of council evidence.

## Catalog #125 Hooks

1. Sensitivity-map contribution: N/A - guard closure only.
2. Pareto constraint: N/A - no candidate score or rank emitted.
3. Bit-allocator hook: N/A - no allocation decision emitted.
4. Cathedral autopilot dispatch hook: ACTIVE - Catalog #315 is a pre-dispatch
   refusal surface consumed by operator-authorize/autopilot flows.
5. Continual-learning posterior update: N/A - no empirical anchor emitted.
6. Probe-disambiguator: N/A - no design ambiguity; this is deterministic
   historical-row repair.
